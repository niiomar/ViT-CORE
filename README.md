# ViT-CORE

Vision Transformer (ViT) implementation and training pipeline using PyTorch in Google Colab.

---

## Overview

This project implements a Vision Transformer model and demonstrates an end-to-end workflow including data loading, preprocessing, training, and evaluation.

The project is designed for experimentation and understanding transformer-based architectures in computer vision.

---

## Project Structure

```
.
├── ViT-CORE.ipynb   # Main notebook
├── README.md        # Documentation
```

---

## Getting Started

### 1. Open in Google Colab

* Go to Google Colab
* Upload `ViT-CORE.ipynb`
* Open and run the notebook

---

### 2. Install Dependencies

Run inside Colab:

```
pip install torch torchvision numpy matplotlib
```

---

## Dataset Setup (Important)

This project originally uses datasets stored in Google Drive.

If you are running this yourself, you will need to:

### Option 1 — Use Your Own Dataset

1. Upload your dataset to Colab or Google Drive
2. Update file paths in the notebook

Example:

```python
data_path = "/your/path/to/dataset"
```

---

### Option 2 — Mount Google Drive

If using Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then update paths accordingly.

---

## How to Run

Run all cells in order:

1. Load dataset
2. Build model
3. Train model
4. Evaluate results

---

## Results

The notebook includes:

* Training loss
* Accuracy metrics
* Model predictions

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Google Colab

---

## Limitations

* Dataset is not included in the repository
* Paths must be configured manually
* Notebook-based (not modularized yet)

---

## Future Improvements

* Convert notebook into modular Python scripts
* Add dataset loader abstraction
* Improve reproducibility
* Add pretrained weights

---

## License

MIT License
