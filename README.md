# ViT-CORE

[![CI](https://github.com/niiomar/ViT-CORE/actions/workflows/ci.yml/badge.svg)](https://github.com/niiomar/ViT-CORE/actions/workflows/ci.yml)

A weighted dual-view Vision Transformer pipeline for deepfake detection, built in PyTorch and designed to run in Google Colab.

## Architecture

- **Model:** ViT-Small (patch16/224) via timm, pretrained on ImageNet
- **Training:** Dual-view with cross-entropy (label smoothing) + consistency loss (MSE or cosine), AdamW with decoupled weight decay, mixed precision, cosine LR schedule with warmup, gradient clipping, EMA of weights, early stopping
- **Augmentations:** RaAug (view 1) and DFDCselim (view 2)
- **Evaluation:** AUC, TDR@0.1, TDR@0.01, confusion matrix, ROC curves, machine-readable `results.json`
- **Inference:** `predict.py` scores a single image or a folder, no CSV manifest required
- **Environment:** Google Colab (GPU recommended)

## Project Structure

```
ViT-CORE/
├── ViT-CORE.ipynb              # Colab orchestration notebook
├── train.py                    # Training loop with CLI args
├── evaluate.py                 # Evaluation across all datasets
├── predict.py                  # Single-image / folder inference
├── model_utils.py              # Shared model, transform, checkpoint, and EMA helpers
├── datasets.py                 # TrainDataset and TestDataset
├── augmentations.py            # RaAug and DFDCselim transforms
├── loss.py                     # Consistency loss functions
├── metrics.py                  # AUC and TDR computation
├── dataset_config.example.json # Template for evaluate.py's dataset list
├── tests/                      # pytest suite
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # + pytest, pytest-cov, ruff, pre-commit
├── .pre-commit-config.yaml     # Runs ruff on commit
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

Checkpoints (`vitcore_latest.pth`, `vitcore_best.pth`, `vitcore_exit.pth` — all one schema, all resumable) and a loss log CSV are saved to `--output-dir` automatically. Bad paths are checked up front, so a typo in `--train-csv` fails immediately instead of after the first batch.

Other flags worth knowing about:

| Flag | Default | Purpose |
|---|---|---|
| `--num-workers` | `4` | DataLoader worker processes |
| `--warmup-epochs` | `3` | Linear LR warmup before cosine decay |
| `--min-lr` | `1e-6` | LR floor at the end of cosine decay |
| `--weight-decay` | `0.05` | AdamW weight decay (skipped for biases and norm params) |
| `--label-smoothing` | `0.1` | Cross-entropy label smoothing |
| `--grad-clip-norm` | `1.0` | Max gradient norm |
| `--no-amp` | off | Disable mixed precision (on by default on CUDA) |
| `--no-ema` | off | Disable EMA of weights (on by default) |
| `--ema-decay` | `0.999` | EMA decay rate |
| `--early-stopping-patience` | `10` | Stop after this many epochs with no val-AUC improvement (`0` disables) |
| `--seed` | `42` | Seeds torch/numpy/random plus CUDA and DataLoader workers |

When EMA is on (the default), validation each epoch runs on the EMA shadow weights rather than the raw model, and every checkpoint stores both (`model_state_dict` for resuming, `ema_state_dict` for inference) — `evaluate.py` and `predict.py` automatically prefer the EMA weights when loading a checkpoint.

### Watching training live

Training metrics are also logged to TensorBoard under `<output-dir>/tensorboard/`:

```bash
tensorboard --logdir "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore/tensorboard"
```

In Colab, run `%load_ext tensorboard` then `%tensorboard --logdir ...` in a cell instead.

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

Runs evaluation on every dataset listed in the config (FF++, Celeb-DF, DFDC-Preview, and WildDeepfake by default). Saves ROC curves, confusion matrices, score distribution plots, and a `results.json` summary (per-dataset accuracy/AUC/TDR, sample counts, checkpoint path, timestamp) to `--output-dir` — the machine-readable file to diff between runs or gate CI on, rather than parsing logs. Pass `--no-show` for headless/CI runs where there's no display to render figures to.

The notebook generates `dataset_config.json` for you from the `DATASETS_DIR` variable in its setup cell, so notebook users don't need the manual copy step above.

## Inference

To score a single image or a folder of images without building a CSV manifest:

```bash
python predict.py \
  --checkpoint "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore/vitcore_best.pth" \
  --input "/path/to/image_or_folder" \
  --output predictions.csv
```

Prints a real/fake label and `p(fake)` per image; `--threshold` (default `0.5`) controls the label cutoff and `--output` is optional.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest --cov=. --cov-report=term-missing
```

The suite covers `metrics.py`, `augmentations.py`, `model_utils.py`, the pure-function parts of `train.py`/`evaluate.py`/`predict.py`, and the CSV/glob sample-loading logic in `datasets.py` — no GPU or dataset download required. CI (`.github/workflows/ci.yml`) runs this plus `ruff check .` and a smoke import of every entry point on every push and pull request.

### Development

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Runs `ruff` automatically on each commit. Dependency versions in `requirements.txt`/`requirements-dev.txt` are pinned deliberately and bumped by hand — there is no automated dependency-update bot on this repo.

## Requirements

- Google Colab (GPU recommended, free tier works)
- Python 3.10+
- See `requirements.txt` for pinned package versions (PyTorch, torchvision, timm, scikit-learn, seaborn, tqdm, etc.)
- Datasets stored in Google Drive (not included)

## License

MIT License
