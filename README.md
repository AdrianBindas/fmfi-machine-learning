# FMFI Machine learning project

Feature extraction for skin lesion classification

## Overview

This project extracts local and global features from dermoscopic images and classifies them into 7 skin lesion types.

## Project Structure

```
├── data/
│   ├── ISIC2018_Task3_Training_Input/
│   ├── ISIC2018_Task3_Validation_Input/
│   └── ISIC2018_Task3_Test_Input/
├── segmentation.py          # Image preprocessing pipeline
├── feature_engineering.py   # Morphological feature extraction
├── resnet.py                # Local feature extraction
├── training.py              # Model training and evaluation
├── feature_analysis.py      # PCA reverse mapping
├── utils.py                 # Helper functions and Pipeline class
└── nb.ipynb                 # Segmentation and global feature extraction
```

## Usage

### 1. Download dataset from https://challenge2018.isic-archive.com/

### 2. Install requirements using pip or uv

### 3. Extract Segmentation Features

Run cells in nb.ipynb.
This creates parquet files with segmentation features in:
- `train_features/train_segmentation_features.parquet`
- `validation_features/valid_segmentation_features.parquet`
- `test_features/test_segmentation_features.parquet`

### 4. Extract ResNet Features

This creates parquet files with local features.
```bash
python resnet.py data/ISIC2018_Task3_Training_Input train_features
python resnet.py data/ISIC2018_Task3_Validation_Input validation_features
python resnet.py data/ISIC2018_Task3_Test_Input test_features
```

### 5. Train Models

```bash
python training.py
```

Results saved to `results/` directory:
- Confusion matrices
- Training history plots
- PCA variance analysis
- Feature importance rankings

## Requirements
- Python libraries listed in pyproject.toml
- CUDA