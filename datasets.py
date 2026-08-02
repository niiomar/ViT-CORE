import glob
import logging
import os
from typing import Callable, Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VALID_EXTS = (".png", ".jpg", ".jpeg")
MAX_OPEN_RETRIES = 5


def _load_samples_from_csv(csv_path: str, root_dir: str) -> list:
    """Read a manifest CSV (path, label) into a flat list of (image_path, label) pairs.

    A `path` may point at a single image, or at a directory — in which case every
    image file inside it is loaded with that row's label.
    """
    samples = []
    df = pd.read_csv(csv_path)
    for row in df.itertuples(index=False):
        entry = os.path.join(root_dir, row.path)
        label = int(row.label)
        if os.path.isfile(entry) and entry.lower().endswith(VALID_EXTS):
            samples.append((entry, label))
        else:
            imgs = [p for p in glob.glob(os.path.join(entry, "*"))
                    if p.lower().endswith(VALID_EXTS)]
            if not imgs:
                logger.warning(f"No images found in: {entry}")
                continue
            for p in imgs:
                samples.append((p, label))
    return samples


def _safe_open(path: str, idx: int, samples: list) -> Image.Image:
    """Open an image, retrying at subsequent samples if it's unreadable/corrupt."""
    next_idx = idx
    for _ in range(MAX_OPEN_RETRIES):
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            logger.warning(f"Skipping unreadable image: {path}")
            next_idx = (next_idx + 1) % len(samples)
            path = samples[next_idx][0]
    raise OSError(
        f"Could not find a readable image after {MAX_OPEN_RETRIES} attempts "
        f"starting from index {idx}"
    )


class TrainDataset(Dataset):
    """Dual-view dataset for training with two augmentation transforms."""

    def __init__(self, csv_path: str, root_dir: str, transform1: Callable, transform2: Optional[Callable]):
        self.transform1 = transform1
        self.transform2 = transform2
        self.samples = _load_samples_from_csv(csv_path, root_dir)
        logger.info(f"TrainDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = _safe_open(path, idx, self.samples)
        v1 = self.transform1(img)
        v2 = self.transform2(img) if self.transform2 else None
        return (v1, v2) if v2 is not None else v1, label


class TestDataset(Dataset):
    """Single-view dataset for evaluation."""

    def __init__(self, csv_path: str, root_dir: str, transform: Callable):
        self.transform = transform
        self.samples = _load_samples_from_csv(csv_path, root_dir)
        logger.info(f"TestDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = _safe_open(path, idx, self.samples)
        return self.transform(img), label
