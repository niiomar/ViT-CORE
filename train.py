import argparse
import atexit
import csv
import logging
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from augmentations import get_transform
from datasets import TrainDataset
from loss import consistency_loss_cosine, consistency_loss_mse
from metrics import compute_auc, compute_tdr
from model_utils import ModelEma, build_eval_transform, build_model, build_param_groups, validate_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--val-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6, help="LR floor at the end of cosine decay")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lambda-consistency", type=float, default=2.0)
    p.add_argument("--consistency-loss", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="Use automatic mixed precision on CUDA (default: on)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--ema", dest="ema", action="store_true", default=True,
                    help="Track an EMA of weights for validation/checkpointing (default: on)")
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--early-stopping-patience", type=int, default=10,
                    help="Stop after this many epochs with no val-AUC improvement. 0 disables.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    if not os.path.exists(path):
        return 0, 0.0, None
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_auc", 0.0), ckpt.get("ema_state_dict")


def save_checkpoint(path, model, optimizer, epoch, best_auc, **extra) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auc": best_auc,
        **extra,
    }, path)


def make_lr_lambda(warmup_epochs: int, total_epochs: int, lr: float, min_lr: float):
    floor = min_lr / lr

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        return floor + (1 - floor) * cosine

    return lr_lambda


def should_stop_early(epochs_since_improvement: int, patience: int) -> bool:
    """True once `patience` epochs have passed with no val-AUC improvement. patience <= 0 disables."""
    return patience > 0 and epochs_since_improvement >= patience


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_paths({
        "train-csv": args.train_csv,
        "train-dir": args.train_dir,
        "val-csv": args.val_csv,
        "val-dir": args.val_dir,
    })
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    logger.info(f"Device: {device}  AMP: {use_amp}  EMA: {args.ema}")

    t1 = get_transform("raaug")
    t2 = get_transform("dfdcselim")
    val_transform = build_eval_transform()

    train_ds = TrainDataset(args.train_csv, args.train_dir, t1, t2)
    val_ds = TrainDataset(args.val_csv, args.val_dir, val_transform, None)

    labels = [lbl for _, lbl in train_ds.samples]
    counts = Counter(labels)
    weights = torch.DoubleTensor([1.0 / counts[label] for label in labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker
        loader_kwargs["generator"] = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(num_classes=2, pretrained=True)
    optimizer = optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_epochs, args.epochs, args.lr, args.min_lr)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_path = os.path.join(args.output_dir, "vitcore_latest.pth")
    best_path = os.path.join(args.output_dir, "vitcore_best.pth")
    csv_path = os.path.join(args.output_dir, "vitcore_losses.csv")

    start_epoch, best_auc, ema_state = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
    model.to(device)

    ema = None
    if args.ema:
        ema = ModelEma(model, decay=args.ema_decay).to(device)
        if ema_state is not None:
            ema.load_state_dict(ema_state)
    eval_model = ema.module if ema is not None else model

    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    cons_fn = consistency_loss_mse if args.consistency_loss == "mse" else consistency_loss_cosine

    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    atexit.register(writer.close)

    exit_state = {"epoch": start_epoch, "best_auc": best_auc}

    def save_on_exit():
        extra = {"scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict()}
        if ema is not None:
            extra["ema_state_dict"] = ema.state_dict()
        save_checkpoint(
            os.path.join(args.output_dir, "vitcore_exit.pth"), model, optimizer,
            exit_state["epoch"], exit_state["best_auc"], **extra,
        )

    atexit.register(save_on_exit)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "total_loss", "total_ce", "total_cons",
                                    "accuracy", "val_auc", "tdr@0.1", "tdr@0.01", "lr"])

    epochs_since_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = total_ce = total_cons = correct = total = 0
        acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, ((v1, v2), lbls) in enumerate(pbar, start=1):
            v1, v2, lbls = v1.to(device), v2.to(device), lbls.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                p1 = model(v1)
                p2 = model(v2)
                loss_ce = ce_loss(p1, lbls)
                loss_cons = cons_fn(p1, p2)
                loss = loss_ce + args.lambda_consistency * loss_cons

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_cons += loss_cons.item()
            _, pred = torch.max(p1, 1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)

            acc = 100 * correct / total
            pbar.set_postfix({"loss": f"{total_loss/batch_idx:.3f}", "acc": f"{acc:.2f}%"})

        eval_model.eval()
        val_labels, val_probs, val_correct, val_total = [], [], 0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"[VAL] Epoch {epoch+1}"):
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    preds = eval_model(imgs)
                _, predicted = torch.max(preds, 1)
                val_correct += (predicted == lbls).sum().item()
                val_total += lbls.size(0)
                val_probs.extend(torch.softmax(preds, 1)[:, 1].detach().float().cpu().numpy())
                val_labels.extend(lbls.detach().cpu().numpy())

        val_auc = compute_auc(val_labels, val_probs)
        tdr01 = compute_tdr(val_labels, val_probs, 0.1)
        tdr001 = compute_tdr(val_labels, val_probs, 0.01)
        val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[VAL] Acc: {val_acc:.2f}%  AUC: {val_auc:.4f}  "
                    f"TDR@0.1: {tdr01:.4f}  TDR@0.01: {tdr001:.4f}  LR: {current_lr:.2e}")

        writer.add_scalar("train/loss_total", total_loss / batch_idx, epoch + 1)
        writer.add_scalar("train/loss_ce", total_ce / batch_idx, epoch + 1)
        writer.add_scalar("train/loss_consistency", total_cons / batch_idx, epoch + 1)
        writer.add_scalar("train/accuracy", acc, epoch + 1)
        writer.add_scalar("val/accuracy", val_acc, epoch + 1)
        writer.add_scalar("val/auc", val_auc, epoch + 1)
        writer.add_scalar("val/tdr_at_0.1", tdr01, epoch + 1)
        writer.add_scalar("val/tdr_at_0.01", tdr001, epoch + 1)
        writer.add_scalar("lr", current_lr, epoch + 1)

        scheduler.step()

        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        extra_state = {
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        if ema is not None:
            extra_state["ema_state_dict"] = ema.state_dict()

        if improved:
            save_checkpoint(best_path, model, optimizer, epoch + 1, best_auc, **extra_state)

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, best_auc, **extra_state)
        exit_state["epoch"] = epoch + 1
        exit_state["best_auc"] = best_auc

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, round(total_loss, 4), round(total_ce, 4),
                                    round(total_cons, 4), round(acc, 2),
                                    round(val_auc, 4), round(tdr01, 4), round(tdr001, 4),
                                    round(current_lr, 8)])

        if should_stop_early(epochs_since_improvement, args.early_stopping_patience):
            logger.info(f"No val-AUC improvement in {epochs_since_improvement} epochs, stopping early.")
            break


if __name__ == "__main__":
    main()
