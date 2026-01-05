import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import numpy as np
import polars as pl
import cv2
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import sys
from PIL import Image

import utils


warnings.filterwarnings('ignore')

if not torch.cuda.is_available():
    raise Exception("Install CUDA")

if len(sys.argv) != 3:
    raise Exception("Invalid number of arguments. Usage: python resnet.py <input_folder> <output_folder>")

INPUT_DIR = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Model config
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image config
TARGET_SIZE = 640
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


# Define input transformations
transform = transforms.Compose([
    transforms.Pad(padding=(0,95), fill=0, padding_mode='constant'),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])


class CustomDataset(Dataset):
    """Dataset for loading images from a folder."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return idx, (img_path, image)


def extract_features(model, dataloader, device):
    features_list = []
    indices_list = []
    img_paths_list = []
    
    model.eval()
    
    with torch.no_grad():
        for indices, (img_paths, images) in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)

            features = model(images)
            
            features = features.squeeze(-1).squeeze(-1)
            
            features_list.append(features.cpu().numpy())
            indices_list.append(indices.numpy())
            img_paths_list.append(img_paths)
    
    all_features = np.vstack(features_list)
    all_indices = np.concatenate(indices_list)
    all_img_paths = np.concatenate(img_paths_list)
    
    return all_features, all_indices, all_img_paths


if __name__ == "__main__":
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model = model.to(DEVICE)

    image_paths = utils.get_image_paths(INPUT_DIR)

    dataset = CustomDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    features, indices, img_paths = extract_features(model, dataloader, device=DEVICE)


    output_path = OUTPUT_DIR / "features.parquet"

    results_df = pl.DataFrame({
        'indices': indices,
        'img_paths': img_paths,
        'features': features,
    })
    results_df.write_parquet(output_path)
