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
from torchvision import transforms
from tqdm.auto import tqdm
from timm.models import vit_small_patch16_224

from augmentations import get_transform
from datasets import TrainDataset
from loss import consistency_loss_mse, consistency_loss_cosine
from metrics import compute_auc, compute_tdr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
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
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--lambda-consistency", type=float, default=2.0)
    p.add_argument("--consistency-loss", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="Use automatic mixed precision on CUDA (default: on)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_auc", 0.0)

def save_checkpoint(path, model, optimizer, epoch, best_auc, **extra):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auc": best_auc,
        **extra,
    }, path)

def make_lr_lambda(warmup_epochs, total_epochs, lr, min_lr):
    floor = min_lr / lr

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        return floor + (1 - floor) * cosine

    return lr_lambda

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    logger.info(f"Device: {device}  AMP: {use_amp}")

    t1 = get_transform("raaug")
    t2 = get_transform("dfdcselim")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

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

    model = vit_small_patch16_224(pretrained=True)
    model.head = nn.Linear(model.head.in_features, 2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.warmup_epochs, args.epochs, args.lr, args.min_lr)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_path = os.path.join(args.output_dir, "vitcore_latest.pth")
    best_path = os.path.join(args.output_dir, "vitcore_best.pth")
    csv_path = os.path.join(args.output_dir, "vitcore_losses.csv")

    start_epoch, best_auc = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    cons_fn = consistency_loss_mse if args.consistency_loss == "mse" else consistency_loss_cosine

    exit_state = {"epoch": start_epoch, "best_auc": best_auc}

    def save_on_exit():
        save_checkpoint(
            os.path.join(args.output_dir, "vitcore_exit.pth"), model, optimizer,
            exit_state["epoch"], exit_state["best_auc"],
            scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(),
        )

    atexit.register(save_on_exit)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "total_loss", "total_ce", "total_cons",
                                    "accuracy", "val_auc", "tdr@0.1", "tdr@0.01", "lr"])

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

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_cons += loss_cons.item()
            _, pred = torch.max(p1, 1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)

            acc = 100 * correct / total
            pbar.set_postfix({"loss": f"{total_loss/batch_idx:.3f}", "acc": f"{acc:.2f}%"})

        model.eval()
        val_labels, val_probs, val_correct, val_total = [], [], 0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"[VAL] Epoch {epoch+1}"):
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    preds = model(imgs)
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

        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(best_path, model, optimizer, epoch + 1, best_auc,
                            scheduler_state_dict=scheduler.state_dict(),
                            scaler_state_dict=scaler.state_dict())

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, best_auc,
                        scheduler_state_dict=scheduler.state_dict(),
                        scaler_state_dict=scaler.state_dict())
        exit_state["epoch"] = epoch + 1
        exit_state["best_auc"] = best_auc

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, round(total_loss, 4), round(total_ce, 4),
                                    round(total_cons, 4), round(acc, 2),
                                    round(val_auc, 4), round(tdr01, 4), round(tdr001, 4),
                                    round(current_lr, 8)])

if __name__ == "__main__":
    main()
