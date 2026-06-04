import sys
sys.path.append("/content/deit")

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
from tqdm.auto import tqdm
from timm.models import vit_small_patch16_224

from datasets import TestDataset
from metrics import compute_tdr


DATASET_CONFIGS = {
    "FF++ (In-Domain)": {
        "csv": "/content/drive/MyDrive/ViT-CORE-Datasets/ffpp/test_filtered.csv",
        "dir": "/content/drive/MyDrive/ViT-CORE-Datasets/ffpp/test",
    },
    "Celeb-DF": {
        "csv": "/content/drive/MyDrive/ViT-CORE-Datasets/celebdf/test_filtered.csv",
        "dir": "/content/drive/MyDrive/ViT-CORE-Datasets/celebdf/test",
    },
    "DFDC-P": {
        "csv": "/content/drive/MyDrive/ViT-CORE-Datasets/dfdc/test_filtered.csv",
        "dir": "/content/drive/MyDrive/ViT-CORE-Datasets/dfdc/test",
    },
    "WildDeepfake": {
        "csv": "/content/drive/MyDrive/ViT-CORE-Datasets/wilddeepfake/test_filtered.csv",
        "dir": "/content/drive/MyDrive/ViT-CORE-Datasets/wilddeepfake/test",
    },
}

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


def load_model(checkpoint_path, device):
    model = vit_small_patch16_224(pretrained=False, num_classes=2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_predictions(model, loader, device):
    labels, scores, preds = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Predicting"):
            imgs = imgs.to(device)
            out = model(imgs)
            probs = torch.softmax(out, 1)[:, 1]
            scores.extend(probs.cpu().numpy())
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(labels), np.array(scores), np.array(preds)


def plot_roc_curves(roc_results, output_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, res), color in zip(roc_results.items(), colors):
        plt.plot(res["fpr"], res["tpr"], color=color, lw=2,
                 label=f"{name} (AUC={res['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — Cross-Domain Generalisation", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.show()


def plot_confusion_matrix(labels, preds, name, output_dir):
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=["Real", "Fake"], columns=["Real", "Fake"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    plt.title(f"Confusion Matrix — {name}", fontsize=14, pad=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.show()


def plot_score_distribution(labels, scores, name, output_dir):
    real = scores[labels == 0]
    fake = scores[labels == 1]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(fake, color="coral", label="Fake", kde=True, bins=50)
    sns.histplot(real, color="dodgerblue", label="Real", kde=True, bins=50)
    plt.title(f"Confidence Score Distribution — {name}", fontsize=14)
    plt.xlabel("Predicted Probability of Being Fake", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_scores.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.show()


def plot_training_curves(log_path, output_dir, max_epoch=None):
    df = pd.read_csv(log_path)
    if max_epoch:
        df = df[df["epoch"] <= max_epoch]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["total_loss"], label="Total Loss", color="blue", marker="o")
    plt.plot(df["epoch"], df["total_ce"], label="CE Loss", color="coral", marker="o")
    plt.plot(df["epoch"], df["total_cons"], label="Consistency Loss", color="green", marker="o")
    plt.title("Training Loss Breakdown", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-path", type=str, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    roc_results = {}

    for name, cfg in DATASET_CONFIGS.items():
        print(f"\n--- {name} ---")
        ds = TestDataset(cfg["csv"], cfg["dir"], TEST_TRANSFORM)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        labels, scores, preds = get_predictions(model, loader, device)

        acc = accuracy_score(labels, preds)
        auc_score = roc_auc_score(labels, scores)
        tdr01 = compute_tdr(labels, scores, 0.1)
        tdr001 = compute_tdr(labels, scores, 0.01)
        print(f"Acc: {acc:.4f}  AUC: {auc_score:.4f}  TDR@0.1: {tdr01:.4f}  TDR@0.01: {tdr001:.4f}")

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_results[name] = {"fpr": fpr, "tpr": tpr, "auc": auc_score}

        plot_confusion_matrix(labels, preds, name, args.output_dir)
        plot_score_distribution(labels, scores, name, args.output_dir)

    plot_roc_curves(roc_results, args.output_dir)

    if args.log_path and os.path.exists(args.log_path):
        plot_training_curves(args.log_path, args.output_dir)


if __name__ == "__main__":
    main()
