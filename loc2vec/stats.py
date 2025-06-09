import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import numpy as np

# update path before running
tiles = sorted(Path("../full/17").rglob("*.png"))
sum_, sum_sq, count = 0., 0., 0.

for path in tiles:
    # x = (str(path)).float()          # (C,H,W) in 0-255
    x = Image.open(path).convert("RGB")
    x = torch.tensor(np.array(x)).permute(2, 0, 1)  # Convert to tensor and rearrange to (C, H, W)
    x = x / 255.0
    sum_   += x.mean(dim=(1,2))
    sum_sq += (x ** 2).mean(dim=(1,2))
    count  += 1

mean = sum_ / count
std  = (sum_sq / count - mean**2).sqrt()

# torch.save({'mean': mean, 'std': std}, "norm_stats.pt")

print("Mean:", mean)
print("Std:", std)

# todo: add arguments handling for paths, etc.
