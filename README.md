# ViT-CORE

A weighted dual-view Vision Transformer pipeline for deepfake detection, built in PyTorch and designed to run in Google Colab.

## Architecture

- **Model:** ViT-Small (patch16/224) via timm, pretrained on ImageNet
- **Training:** Dual-view with cross-entropy + consistency loss (MSE or cosine)
- **Augmentations:** RaAug (view 1) and DFDCselim (view 2)
- **Evaluation:** AUC, TDR@0.1, TDR@0.01, confusion matrix, ROC curves
- **Environment:** Google Colab (GPU recommended)

## Project Structure

```
ViT-CORE/
├── ViT-CORE.ipynb    # Colab orchestration notebook
├── train.py          # Training loop with CLI args
├── evaluate.py       # Evaluation across all datasets
├── datasets.py       # TrainDataset and TestDataset
├── augmentations.py  # RaAug and DFDCselim transforms
├── loss.py           # Consistency loss functions
├── metrics.py        # AUC and TDR computation
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

### 3. Clone DeiT and install dependencies

```bash
git clone https://github.com/facebookresearch/deit.git
pip install timm submitit opencv-python torch torchvision scikit-learn seaborn
```

### 4. Copy modules into the DeiT directory

The notebook handles this automatically via the setup cell.

### 5. Add your datasets to Google Drive

> **Note on paths:** All paths below reflect the folder structure used during development. You will need to update them to match your own Google Drive layout. Anywhere you see `/content/drive/MyDrive/...`, replace it with the actual path to your data.

Organise your datasets in Drive using this structure (or adjust the paths in `evaluate.py` to match your own):

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

Each CSV must have `path` and `label` columns (0 = real, 1 = fake). The `path` column should be relative to the dataset's root directory.

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

Checkpoints and a loss log CSV are saved to `--output-dir` automatically.

## Evaluation

Replace the paths below with your own before running:

```bash
python evaluate.py \
  --checkpoint "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore/vitcore_best.pth" \
  --output-dir "/content/drive/MyDrive/<your-experiments-folder>/charts" \
  --log-path   "/content/drive/MyDrive/<your-experiments-folder>/ffpp_vitcore/vitcore_losses.csv"
```

Runs evaluation on FF++, Celeb-DF, DFDC-Preview, and WildDeepfake. Saves ROC curves, confusion matrices, and score distribution plots.

> **Note:** Dataset paths for each evaluation split are defined at the top of `evaluate.py` under `DATASET_CONFIGS`. Update those entries to point to your own test CSVs and directories.

## Requirements

- Google Colab (GPU recommended, free tier works)
- Python 3.10+
- PyTorch, torchvision, timm, scikit-learn, seaborn, tqdm
- Datasets stored in Google Drive (not included)

## License

MIT License
