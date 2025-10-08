# Eye Disease Classification from Retinal Scans using Deep Learning

This repository contains experiments and notebooks for classifying diabetic retinopathy severity from retinal fundus images using deep learning (APTOS 2019 dataset). The work explores preprocessing strategies (including CLAHE), model architectures (ResNet50, EfficientNet-B0), training and evaluation pipelines, and visualization of results.

## Repository structure

- `experiment_code.ipynb` — Main notebook with end-to-end experiments: dataset download (Kaggle), Dataset class, train/validation split, data augmentation, training loops for ResNet50 and EfficientNet-B0 (various training strategies), evaluation and plotting (loss/accuracy curves, confusion matrices).
- `experiement_with_CLAHE.ipynb` — Notebook focusing on CLAHE-based preprocessing and experiments repeating training/evaluation with CLAHE-applied images.
- `data/aptos/` — Expected dataset folder (not included). Contains `train.csv`, `train_images/`, `test_images/`, `train_split.csv`, `val_split.csv`, `test.csv`, and `sample_submission.csv`.
- `env/` — Local Python virtual environment used for development (already present in this workspace).
- `output_images/` — Exported plots and confusion matrices from experiments.

## Quick summary of the notebooks

- `experiment_code.ipynb`:
  - Uses a custom `APTOSDataset` PyTorch `Dataset` class to load images and labels from CSVs.
  - Defines transforms for training and validation (resize, crop, normalization) and visual helpers.
  - Creates stratified train/validation splits and DataLoaders.
  - Trains and evaluates multiple models:
    - ResNet50 (full training and fine-tuning strategies)
    - EfficientNet-B0 (classifier finetuning; experiments with 10 and 30 epochs)
  - Visualizes sample images, loss/accuracy curves, and confusion matrices.

- `experiement_with_CLAHE.ipynb`:
  - Extends `APTOSDataset` to optionally apply CLAHE to the L channel in LAB color space using OpenCV.
  - Builds DataLoaders with CLAHE pre-processing and runs experiments similar to `experiment_code.ipynb`.
  - Includes visualization of CLAHE-applied samples and evaluation (confusion matrices and metric curves).

## Requirements & environment

- Python: 3.8+ (project uses Python 3.11 in `env/`)
- GPU recommended for training (CUDA + PyTorch). Training on CPU will be slow.

Install dependencies (recommended inside a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer the included `env/` virtual environment, activate it accordingly (macOS/zsh):

```bash
source env/bin/activate
```

### Kaggle dataset setup

1. Place your `kaggle.json` (Kaggle API token) somewhere on your machine (e.g., `~/Downloads/kaggle.json`).
2. The notebooks expect the token to be copied to `~/.kaggle/kaggle.json` with permissions `600` (there is a cell in `experiment_code.ipynb` that copies and sets permissions).
3. The dataset code uses the `kaggle` CLI to download the APTOS dataset into `data/aptos`.

## How to run the notebooks

Open either notebook in Jupyter or VS Code and run cells in order. High-level steps:

1. Configure Kaggle credentials (or skip download if you already have the dataset in `data/aptos`).
2. Ensure `data/aptos/train.csv` and `data/aptos/train_images/` exist.
3. Run the preprocessing / data-split cells to create `train_split.csv` and `val_split.csv`.
4. Run model training cells. Adjust `batch_size`, `num_epochs`, and `num_workers` to fit your machine.

Notes:
- Use GPU (CUDA) for reasonable training times. If `torch.cuda.is_available()` is False, training will use CPU and be slow.
- The notebooks include many inline `pip install` cells for missing packages; installing via `requirements.txt` is recommended for reproducibility.

## Reproducibility & recommended changes

- Save model checkpoints during training (not saved by default). Add `torch.save()` calls in training loops.
- Consider using Git LFS for storing large image files (`train_images/`) or keep data outside the repository (recommended).
- Add a small script-based training runner (`train.py`) for headless execution on a server.

## Results

The `output_images/` directory contains example outputs (confusion matrices and training curves) generated during experimentation. See files like `confusion_matrix_resnet50.png` and `output_curves_efficientnet.png`.

## Files added in this commit

- `README.md` — this file

## Contact / next steps

If you want, I can:
- Add a `requirements.txt` (I included one in this commit) or a `environment.yml` for conda.
- Create a small `train.py` script to run experiments from the command line.
- Add model checkpoint saving and evaluation scripts.
