# ViT-CORE

A weighted dual-view Vision Transformer pipeline for deepfake detection, built in PyTorch and designed to run in Google Colab.

## Architecture

- **Model:** ViT-Small (patch16/224) via timm, pretrained on ImageNet
- **Training:** Dual-view with cross-entropy + consistency loss (MSE or cosine), mixed precision, cosine LR schedule with warmup, gradient clipping
- **Augmentations:** RaAug (view 1) and DFDCselim (view 2)
- **Evaluation:** AUC, TDR@0.1, TDR@0.01, confusion matrix, ROC curves
- **Environment:** Google Colab (GPU recommended)

## Project Structure

```
ViT-CORE/
├── ViT-CORE.ipynb              # Colab orchestration notebook
├── train.py                    # Training loop with CLI args
├── evaluate.py                 # Evaluation across all datasets
├── datasets.py                 # TrainDataset and TestDataset
├── augmentations.py            # RaAug and DFDCselim transforms
├── loss.py                     # Consistency loss functions
├── metrics.py                  # AUC and TDR computation
├── dataset_config.example.json # Template for evaluate.py's dataset list
├── tests/                      # pytest suite (metrics, augmentations, datasets)
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # + pytest, ruff
├── README.md
└── LICENSE
```

## Setup

### 1. Clone the repo and open the notebook

Upload or clone this repo to Google Drive, then open `ViT-CORE.ipynb` in Google Colab.

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The notebook's setup cell runs this for you. Outside Colab (e.g. running the scripts directly), pin dev tooling too:

```bash
pip install -r requirements-dev.txt
```

### 4. Add your datasets to Google Drive

> **Note on paths:** All paths below reflect the folder structure used during development. You will need to update them to match your own Google Drive layout. Anywhere you see `/content/drive/MyDrive/...`, replace it with the actual path to your data.

Organise your datasets in Drive using this structure (or adjust `dataset_config.json`, see below, to match your own):

```
MyDrive/<your-datasets-folder>/
├── ffpp/
│   ├── train_filtered.csv
│   ├── val_filtered.csv
│   ├── test_filtered.csv
│   └── train/  val/  test/
├── celebdf/
├── dfdc/
└── wilddeepfake/
```

Each manifest CSV has `path` and `label` columns (0 = real, 1 = fake), relative to the dataset's root directory. Two row formats are supported:

- **`path` is a file** (e.g. `train/real/0001.png`) — one row per image.
- **`path` is a directory** (e.g. `train/real/clip_0001/`) — every `.png`/`.jpg`/`.jpeg` file inside (case-insensitive) is loaded as a sample with that row's label. Useful when your manifest is per-clip rather than per-frame.

## Training

Replace the paths below with your own before running:

```bash
python train.py \
  --train-csv "/content/drive/MyDrive/<your-datasets-folder>/ffpp/train_filtered.csv" \
  --train-dir "/content/drive/MyDrive/<your-datasets-folder>/ffpp/train" \
  --val-csv   "/content/drive/MyDrive/<your-datasets-folder>/ffpp/val_filtered.csv" \
  --val-dir   "/content/drive/MyDrive/<your-datasets-folder>/ffpp/val" \
  --output-dir "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore" \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --lambda-consistency 5
```

Checkpoints (`vitcore_latest.pth`, `vitcore_best.pth`, `vitcore_exit.pth` — all one schema, all resumable) and a loss log CSV are saved to `--output-dir` automatically.

Other flags worth knowing about:

| Flag | Default | Purpose |
|---|---|---|
| `--num-workers` | `4` | DataLoader worker processes |
| `--warmup-epochs` | `3` | Linear LR warmup before cosine decay |
| `--min-lr` | `1e-6` | LR floor at the end of cosine decay |
| `--grad-clip-norm` | `1.0` | Max gradient norm |
| `--no-amp` | off | Disable mixed precision (on by default on CUDA) |
| `--seed` | `42` | Seeds torch/numpy/random plus CUDA and DataLoader workers |

## Evaluation

`evaluate.py` reads its dataset list from a JSON config instead of hardcoded paths. Copy the template and point it at your own test splits:

```bash
cp dataset_config.example.json dataset_config.json
# then edit dataset_config.json with your own csv/dir paths
```

```bash
python evaluate.py \
  --checkpoint "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore/vitcore_best.pth" \
  --output-dir "/content/drive/MyDrive/<your-experiments-folder>/charts" \
  --log-path   "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore/vitcore_losses.csv" \
  --dataset-config "dataset_config.json"
```

Runs evaluation on every dataset listed in the config (FF++, Celeb-DF, DFDC-Preview, and WildDeepfake by default). Saves ROC curves, confusion matrices, and score distribution plots. Pass `--no-show` for headless/CI runs where there's no display to render figures to.

The notebook generates `dataset_config.json` for you from the `DATASETS_DIR` variable in its setup cell, so notebook users don't need the manual copy step above.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite covers `metrics.py`, `augmentations.py`, and the CSV/glob sample-loading logic in `datasets.py` — no GPU or dataset download required. CI (`.github/workflows/ci.yml`) runs this plus `ruff check .` and a smoke import of `train.py`/`evaluate.py` on every push and pull request.

## Requirements

- Google Colab (GPU recommended, free tier works)
- Python 3.10+
- See `requirements.txt` for pinned package versions (PyTorch, torchvision, timm, scikit-learn, seaborn, tqdm, etc.)
- Datasets stored in Google Drive (not included)

## License

MIT License
