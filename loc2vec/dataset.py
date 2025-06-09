import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset


def load_image(filename) -> ImageFile:
    return Image.open(filename).convert("RGB")

def list_tiles_to_df(tiles_dir):
    files = []
    for file in Path(tiles_dir).resolve().glob("**/*.png"):
        # if filesize is equal to 103 bytes, skip it
        if file.stat().st_size == 103: # empty tile
            continue

        files.append((file.parent.name, file.stem, file.parent.parent.name, str(file)))

    return pd.DataFrame(files, columns=["x", "y", "zoom", "filename"])

def take_positive(tiles_dir, x, y, zoom, max_attempts=10):
    """Get a positive sample within radius of the anchor tile."""
    attempts = 0
    while attempts < max_attempts:
        x_shift, y_shift = random.sample([-1, 0, 1], 2)
        pos_x = int(x) + x_shift
        pos_y = int(y) + y_shift
        
        pos_path = os.path.join(tiles_dir, zoom, str(pos_x), f"{pos_y}.png")
        
        if os.path.exists(pos_path) and os.path.getsize(pos_path) > 103:  # Skip empty tiles
            return pos_path
            
        attempts += 1
    
    # If no valid positive found after max attempts, return original tile as fallback
    return os.path.join(tiles_dir, zoom, str(int(x)), f"{int(y)}.png")

class TilesDataset(Dataset):
    def __init__(self, tiles_root_dir, pos_radius=1, neg_radius_min=10, transform=None):
        self.tiles_root_dir = tiles_root_dir
        self.df = list_tiles_to_df(tiles_root_dir)
        self.pos_radius = pos_radius
        self.neg_radius_min = neg_radius_min
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y, zoom, filename = row['x'], row['y'], row['zoom'], row['filename']

        # Load the anchor image
        image = load_image(filename)

        # Get a positive sample (geographically close)
        pos_path = take_positive(self.tiles_root_dir, x, y, zoom)
        pos_image = load_image(pos_path)
        
        # Get a negative sample (geographically distant)
        # Filter dataframe to find tiles that are far away
        try:
            x_int, y_int = int(x), int(y)
            neg_candidates = self.df[
                ((self.df['zoom'] == zoom) & 
                 ((abs(self.df['x'].astype(int) - x_int) > self.neg_radius_min) | 
                  (abs(self.df['y'].astype(int) - y_int) > self.neg_radius_min)))
            ]
            
            # If no suitable negative found, just sample randomly from the whole dataset
            if len(neg_candidates) == 0:
                neg_candidates = self.df
            
            # Make sure we don't select the anchor tile as negative
            neg_candidates = neg_candidates[neg_candidates['filename'] != filename]
            
            # If still no candidates, use a random tile
            if len(neg_candidates) == 0:
                neg_candidates = self.df[self.df['filename'] != filename]
                
            neg = neg_candidates.sample(1)
            neg_image = load_image(neg['filename'].values[0])
        except Exception as e:
            # Fallback: use a random sample that's not the anchor
            fallback_candidates = self.df[self.df['filename'] != filename]
            neg = fallback_candidates.sample(1)
            neg_image = load_image(neg['filename'].values[0])

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        return {
            'anchor_image': image,
            'pos_image': pos_image,
            'neg_image': neg_image,
            'x': x,
            'y': y,
            'zoom': zoom,
            'filename': filename
        }

if __name__ == "__main__":
    dataset = TilesDataset("../full", pos_radius=1, transform=T.ToTensor())
    print(len(dataset))
    print(f"Dataset length: {len(dataset)}")

    sample = random.choice(dataset)
    print(f"Sample: {sample['filename']}, Coordinates: ({sample['x']}, {sample['y']}), Zoom: {sample['zoom']}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample['anchor_image'].permute(1, 2, 0))
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(sample['pos_image'].permute(1, 2, 0))
    plt.title("Transformed Image")
    plt.subplot(1, 3, 3)
    plt.imshow(sample['neg_image'].permute(1, 2, 0))
    plt.title("Negative Image")
    plt.show()





