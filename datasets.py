import os
import glob
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

VALID_EXTS = (".png", ".jpg", ".jpeg")

def _load_samples_from_csv(csv_path, root_dir):
    samples = []
    df = pd.read_csv(csv_path)
    for row in df.itertuples(index=False):
        entry = os.path.join(root_dir, row.path)
        label = int(row.label)
        if os.path.isfile(entry) and entry.lower().endswith(VALID_EXTS):
            samples.append((entry, label))
        else:
            imgs = []
            for ext in VALID_EXTS:
                imgs.extend(glob.glob(os.path.join(entry, f"*{ext}")))
            if not imgs:
                print(f"[WARN] No images found in: {entry}")
                continue
            for p in imgs:
                samples.append((p, label))
    return samples

def _safe_open(path, idx, samples):
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        print(f"[!] Skipping unreadable image: {path}")
        next_idx = (idx + 1) % len(samples)
        return Image.open(samples[next_idx][0]).convert("RGB")

class TrainDataset(Dataset):
    """Dual-view dataset for training with two augmentation transforms."""

    def __init__(self, csv_path, root_dir, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2
        self.samples = _load_samples_from_csv(csv_path, root_dir)
        print(f"[INFO] TrainDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = _safe_open(path, idx, self.samples)
        v1 = self.transform1(img)
        v2 = self.transform2(img) if self.transform2 else None
        return (v1, v2) if v2 is not None else v1, label

class TestDataset(Dataset):
    """Single-view dataset for evaluation."""

    def __init__(self, csv_path, root_dir, transform):
        self.transform = transform
        self.samples = _load_samples_from_csv(csv_path, root_dir)
        print(f"[INFO] TestDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = _safe_open(path, idx, self.samples)
        return self.transform(img), label
