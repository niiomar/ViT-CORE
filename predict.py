"""Run a trained ViT-CORE checkpoint on a single image or a folder of images."""

import argparse
import csv
import logging
import os

import torch
from PIL import Image

from datasets import VALID_EXTS
from model_utils import build_eval_transform, load_inference_model, validate_paths

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="A single image, or a directory of images")
    p.add_argument("--threshold", type=float, default=0.5, help="p(fake) threshold for the real/fake label")
    p.add_argument("--output", type=str, default=None, help="Optional CSV path to write predictions to")
    return p.parse_args()


def collect_image_paths(path: str) -> list:
    if os.path.isfile(path):
        return [path]
    return sorted(
        os.path.join(path, fname) for fname in os.listdir(path)
        if fname.lower().endswith(VALID_EXTS)
    )


def predict_one(model, transform, image_path: str, device: torch.device) -> float:
    """Return p(fake) in [0, 1] for a single image."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        return torch.softmax(logits, 1)[0, 1].item()


def main() -> None:
    args = parse_args()
    validate_paths({"checkpoint": args.checkpoint, "input": args.input})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_inference_model(args.checkpoint, device)
    transform = build_eval_transform()

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found at: {args.input}")

    rows = []
    for path in image_paths:
        prob_fake = predict_one(model, transform, path, device)
        label = "fake" if prob_fake >= args.threshold else "real"
        logger.info(f"{path}: {label}  (p_fake={prob_fake:.4f})")
        rows.append((path, label, f"{prob_fake:.6f}"))

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "p_fake"])
            writer.writerows(rows)
        logger.info(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
