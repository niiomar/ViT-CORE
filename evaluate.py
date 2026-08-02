import argparse
import json
import logging
import os
from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import TestDataset
from metrics import compute_auc, compute_tdr
from model_utils import build_eval_transform, load_inference_model, validate_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset_config.example.json"
)

TEST_TRANSFORM = build_eval_transform()

SHOW_PLOTS = True


def maybe_show() -> None:
    """Display the current figure unless --no-show was passed, in which case just close it."""
    if SHOW_PLOTS:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close()


def load_dataset_configs(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_dataset_metrics(labels, scores, preds) -> dict:
    """Compute the accuracy/AUC/TDR summary dict for one dataset's predictions."""
    return {
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": float(compute_auc(labels, scores)),
        "tdr@0.1": float(compute_tdr(labels, scores, 0.1)),
        "tdr@0.01": float(compute_tdr(labels, scores, 0.01)),
    }


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
    import matplotlib.pyplot as plt
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
    maybe_show()

def plot_confusion_matrix(labels, preds, name, output_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt
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
    maybe_show()

def plot_score_distribution(labels, scores, name, output_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt
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
    maybe_show()

def plot_training_curves(log_path, output_dir, max_epoch=None):
    import matplotlib.pyplot as plt
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
    maybe_show()

def main():
    global SHOW_PLOTS

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-path", type=str, default=None)
    p.add_argument("--dataset-config", type=str, default="dataset_config.json",
                    help="JSON file mapping dataset display names to {csv, dir}. "
                         "See dataset_config.example.json for the expected format.")
    p.add_argument("--no-show", action="store_true",
                    help="Save plots without displaying them (use for headless/CI runs).")
    args = p.parse_args()

    SHOW_PLOTS = not args.no_show
    if args.no_show:
        matplotlib.use("Agg")

    if not os.path.exists(args.dataset_config):
        logger.error(
            f"Dataset config not found: {args.dataset_config}\n"
            f"Copy dataset_config.example.json to dataset_config.json and update the "
            f"paths for your own Google Drive layout, or pass --dataset-config explicitly."
        )
        raise SystemExit(1)
    dataset_configs = load_dataset_configs(args.dataset_config)

    validate_paths({"checkpoint": args.checkpoint})
    validate_paths({
        f"{name} ({key})": path
        for name, cfg in dataset_configs.items()
        for key, path in cfg.items()
    })

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_inference_model(args.checkpoint, device)

    roc_results = {}
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {},
    }

    for name, cfg in dataset_configs.items():
        logger.info(f"--- {name} ---")
        ds = TestDataset(cfg["csv"], cfg["dir"], TEST_TRANSFORM)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

        labels, scores, preds = get_predictions(model, loader, device)

        metrics = summarize_dataset_metrics(labels, scores, preds)
        logger.info(f"Acc: {metrics['accuracy']:.4f}  AUC: {metrics['auc']:.4f}  "
                    f"TDR@0.1: {metrics['tdr@0.1']:.4f}  TDR@0.01: {metrics['tdr@0.01']:.4f}")
        summary["datasets"][name] = metrics

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_results[name] = {"fpr": fpr, "tpr": tpr, "auc": metrics["auc"]}

        plot_confusion_matrix(labels, preds, name, args.output_dir)
        plot_score_distribution(labels, scores, name, args.output_dir)

    plot_roc_curves(roc_results, args.output_dir)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary metrics to {results_path}")

    if args.log_path and os.path.exists(args.log_path):
        plot_training_curves(args.log_path, args.output_dir)


if __name__ == "__main__":
    main()
