import sys
sys.path.append("/content/deit")

import argparse
import os
import csv
import atexit
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from collections import Counter
from tqdm.auto import tqdm
from timm.models import vit_small_patch16_224

from augmentations import get_transform
from datasets import TrainDataset
from metrics import compute_tdr
from loss import consistency_loss_mse, consistency_loss_cosine
from sklearn.metrics import roc_auc_score

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
    p.add_argument("--lambda-consistency", type=float, default=2.0)
    p.add_argument("--consistency-loss", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_auc", 0.0)

def save_checkpoint(path, model, optimizer, epoch, best_auc, **extra):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auc": best_auc,
        **extra
    }, path)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    weights = torch.DoubleTensor([1.0 / counts[l] for l in labels])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = vit_small_patch16_224(pretrained=True)
    model.head = nn.Linear(model.head.in_features, 2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.output_dir, "vitcore_latest.pth")
    best_path = os.path.join(args.output_dir, "vitcore_best.pth")
    csv_path = os.path.join(args.output_dir, "vitcore_losses.csv")

    start_epoch, best_auc = load_checkpoint(ckpt_path, model, optimizer, device)
    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    cons_fn = consistency_loss_mse if args.consistency_loss == "mse" else consistency_loss_cosine

    atexit.register(lambda: save_checkpoint(
        os.path.join(args.output_dir, "vitcore_exit.pth"), model, optimizer, start_epoch, best_auc
    ))

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "total_loss", "total_ce", "total_cons",
                                    "accuracy", "val_auc", "tdr@0.1", "tdr@0.01"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = total_ce = total_cons = correct = total = 0
        all_labels, all_probs = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for (v1, v2), lbls in pbar:
            v1, v2, lbls = v1.to(device), v2.to(device), lbls.to(device)

            optimizer.zero_grad()
            p1 = model(v1)
            p2 = model(v2)

            loss_ce = ce_loss(p1, lbls)
            loss_cons = cons_fn(p1, p2)
            loss = loss_ce + args.lambda_consistency * loss_cons

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_cons += loss_cons.item()
            _, pred = torch.max(p1, 1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)

            all_probs.extend(torch.softmax(p1, 1)[:, 1].detach().cpu().numpy())
            all_labels.extend(lbls.detach().cpu().numpy())

            acc = 100 * correct / total
            pbar.set_postfix({"loss": f"{total_loss/(total/args.batch_size):.3f}", "acc": f"{acc:.2f}%"})

        model.eval()
        val_labels, val_probs, val_correct, val_total = [], [], 0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"[VAL] Epoch {epoch+1}"):
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs)
                _, predicted = torch.max(preds, 1)
                val_correct += (predicted == lbls).sum().item()
                val_total += lbls.size(0)
                val_probs.extend(torch.softmax(preds, 1)[:, 1].detach().cpu().numpy())
                val_labels.extend(lbls.detach().cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_probs)
        tdr01 = compute_tdr(val_labels, val_probs, 0.1)
        tdr001 = compute_tdr(val_labels, val_probs, 0.01)
        val_acc = 100 * val_correct / val_total
        print(f"[VAL] Acc: {val_acc:.2f}%  AUC: {val_auc:.4f}  TDR@0.1: {tdr01:.4f}  TDR@0.01: {tdr001:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "epoch": epoch+1, "best_auc": best_auc}, best_path)

        save_checkpoint(ckpt_path, model, optimizer, epoch+1, best_auc)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, round(total_loss, 4), round(total_ce, 4),
                                    round(total_cons, 4), round(acc, 2),
                                    round(val_auc, 4), round(tdr01, 4), round(tdr001, 4)])

if __name__ == "__main__":
    main()
