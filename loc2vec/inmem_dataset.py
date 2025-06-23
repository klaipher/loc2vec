import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch


class InMemoryTilesDataset(Dataset):
    def __init__(
        self,
        metadata_filename,
        npy_data_filename,
        neg_radius_min=10,
        pos_radius=2,
        transform=None,
    ):
        self.neg_radius_min = neg_radius_min
        self.pos_radius = pos_radius
        self.transform = transform
        
        # Load metadata
        self.df = pd.read_csv(metadata_filename)
        self.dataset_length = len(self.df)
        
        # Load all images into memory
        print(f"Loading {self.dataset_length} images into memory...")
        memmap_file = np.memmap(
            npy_data_filename,
            dtype=np.uint8,
            mode='r',
            shape=(self.dataset_length, 3, 128, 128)
        )
        
        # Copy everything to RAM
        self.images = memmap_file.copy()
        del memmap_file  # Free the memmap
        print(f"Loaded {self.images.nbytes / (1024**3):.2f} GB into memory")
        
        # Extract coordinates as numpy arrays for fast access
        self.coordinates = self.df[['x', 'y', 'zoom']].astype(np.int32).values
        
        # Build spatial and negative indices
        self._build_spatial_index()
        self._build_negative_index()

    def _build_spatial_index(self):
        """Build spatial index for fast positive sample lookup."""
        self.spatial_index = {}
        
        for idx, (x, y, zoom) in enumerate(self.coordinates):
            key = (int(x), int(y), int(zoom))
            if key not in self.spatial_index:
                self.spatial_index[key] = []
            self.spatial_index[key].append(idx)
        
        # Convert to numpy arrays for faster access
        for key in self.spatial_index:
            self.spatial_index[key] = np.array(self.spatial_index[key], dtype=np.int32)

    def _build_negative_index(self):
        """Build negative index for fast negative sample lookup."""
        self.negative_index = {}
        
        unique_zooms = np.unique(self.coordinates[:, 2])
        
        for zoom in unique_zooms:
            mask = self.coordinates[:, 2] == zoom
            indices = np.where(mask)[0].astype(np.int32)
            coords = self.coordinates[mask]
            
            self.negative_index[int(zoom)] = {
                'indices': indices,
                'x': coords[:, 0],
                'y': coords[:, 1]
            }

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Get anchor coordinates
        x, y, zoom = self.coordinates[idx]
        x, y, zoom = int(x), int(y), int(zoom)
        
        # Find positive and negative samples
        pos_idx = self._take_positive(x, y, zoom, idx)
        neg_idx = self._take_negative(x, y, zoom, idx)
        
        # Get image data
        anchor_data = self.images[idx]
        pos_data = self.images[pos_idx]
        neg_data = self.images[neg_idx]
        
        # Convert to tensors
        if self.transform:
            anchor_tensor = self._array_to_tensor_via_pil(anchor_data)
            pos_tensor = self._array_to_tensor_via_pil(pos_data)
            neg_tensor = self._array_to_tensor_via_pil(neg_data)
        else:
            anchor_tensor = torch.from_numpy(anchor_data).float().div_(255.0)
            pos_tensor = torch.from_numpy(pos_data).float().div_(255.0)
            neg_tensor = torch.from_numpy(neg_data).float().div_(255.0)

        return {
            "anchor_image": anchor_tensor,
            "pos_image": pos_tensor,
            "neg_image": neg_tensor,
            "x": x,
            "y": y,
            "zoom": zoom,
        }

    def _take_positive(self, x, y, zoom, anchor_idx, max_attempts=10):
        """Find positive sample within radius."""
        for _ in range(max_attempts):
            x_shift = random.randint(-self.pos_radius, self.pos_radius)
            y_shift = random.randint(-self.pos_radius, self.pos_radius)

            if x_shift == 0 and y_shift == 0:
                continue

            key = (x + x_shift, y + y_shift, zoom)
            candidates = self.spatial_index.get(key)
            
            if candidates is not None and len(candidates) > 0:
                return candidates[random.randint(0, len(candidates) - 1)]

        # Fallback: return anchor
        return anchor_idx

    def _take_negative(self, x, y, zoom, anchor_idx):
        """Find negative sample outside radius."""
        if zoom not in self.negative_index:
            return anchor_idx

        neg_data = self.negative_index[zoom]
        
        # Vectorized distance calculation
        x_dist = np.abs(neg_data['x'] - x)
        y_dist = np.abs(neg_data['y'] - y)
        
        valid_mask = (x_dist > self.neg_radius_min) | (y_dist > self.neg_radius_min)
        valid_indices = neg_data['indices'][valid_mask]
        
        if len(valid_indices) > 0:
            return valid_indices[random.randint(0, len(valid_indices) - 1)]
        
        # Fallback: return anchor
        return anchor_idx

    def _array_to_tensor_via_pil(self, array_data):
        """Convert array to tensor via PIL."""
        pil_img = Image.fromarray(array_data.transpose(1, 2, 0))
        tensor = self.transform(pil_img)
        return tensor

    def get_memory_info(self):
        """Get memory usage information."""
        image_size_gb = self.images.nbytes / (1024**3)
        print(f"Images in memory: {image_size_gb:.2f} GB")
        print(f"Dataset length: {self.dataset_length}")
        return image_size_gb
