import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class KyivTileDataset(Dataset):
    """
    Dataset for loading and processing Kyiv city tiles for loc2vec training.

    Each sample contains:
    - Anchor tile: current location
    - Positive tile: nearby location (within radius)
    - Negative tile: distant location
    """

    def __init__(
        self,
        tiles_root: str = "12_layer_tiles/tiles",
        tile_size: int = 256,
        positive_radius: int = 2,  # tiles within this radius are positive
        negative_radius: int = 10,  # tiles outside this radius are negative
        max_samples_per_epoch: int = 10000,
    ):
        """
        Args:
            tiles_root: Path to the tiles directory
            tile_size: Size to resize tiles to
            positive_radius: Radius for positive samples (in tile coordinates)
            negative_radius: Minimum radius for negative samples
            max_samples_per_epoch: Maximum number of triplets per epoch
        """
        self.tiles_root = tiles_root
        self.tile_size = tile_size
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius
        self.max_samples_per_epoch = max_samples_per_epoch

        # Get all layer names
        self.layers = [
            name
            for name in os.listdir(tiles_root)
            if os.path.isdir(os.path.join(tiles_root, name))
        ]

        print(f"Found {len(self.layers)} layers: {self.layers}")

        # Load tile coordinates for all layers
        self.tile_coords = self._load_tile_coordinates()

        # Find common coordinates across all layers
        self.common_coords = self._find_common_coordinates()

        print(f"Found {len(self.common_coords)} tiles with data across all layers")

    def _load_tile_coordinates(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Load all available tile coordinates for each layer."""
        coords = {}

        for layer in self.layers:
            layer_coords = []
            layer_path = os.path.join(self.tiles_root, layer)

            # Assuming zoom level 17 based on the HTML file
            zoom_path = os.path.join(layer_path, "17")
            if not os.path.exists(zoom_path):
                continue

            for x_dir in os.listdir(zoom_path):
                x_path = os.path.join(zoom_path, x_dir)
                if not os.path.isdir(x_path):
                    continue

                try:
                    x = int(x_dir)
                except ValueError:
                    continue

                for y_file in os.listdir(x_path):
                    if y_file.endswith(".png"):
                        try:
                            y = int(y_file.replace(".png", ""))
                            layer_coords.append((17, x, y))  # (zoom, x, y)
                        except ValueError:
                            continue

            coords[layer] = layer_coords
            print(f"Layer {layer}: {len(layer_coords)} tiles")

        return coords

    def _find_common_coordinates(self) -> List[Tuple[int, int, int]]:
        """Find coordinates that exist across all layers."""
        if not self.tile_coords:
            return []

        # Start with coordinates from first layer
        common = set(self.tile_coords[self.layers[0]])

        # Find intersection with all other layers
        for layer in self.layers[1:]:
            common = common.intersection(set(self.tile_coords[layer]))

        return list(common)

    def _load_tile_stack(self, coord: Tuple[int, int, int]) -> np.ndarray:
        """
        Load all 12 layers for a given coordinate and stack them.

        Returns:
            np.ndarray: Shape (12, H, W) - stacked tile layers
        """
        zoom, x, y = coord
        layers_data = []

        for layer in self.layers:
            tile_path = os.path.join(
                self.tiles_root, layer, str(zoom), str(x), f"{y}.png"
            )

            if os.path.exists(tile_path):
                # Load image
                img = Image.open(tile_path).convert("RGB")
                img = img.resize((self.tile_size, self.tile_size))
                img_array = np.array(img)

                # Convert to grayscale and normalize
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                img_array = img_array.astype(np.float32) / 255.0
                layers_data.append(img_array)
            else:
                # If tile doesn't exist, create empty tile
                empty_tile = np.zeros(
                    (self.tile_size, self.tile_size), dtype=np.float32
                )
                layers_data.append(empty_tile)

        return np.stack(layers_data, axis=0)  # Shape: (12, H, W)

    def _get_nearby_coordinates(
        self, coord: Tuple[int, int, int], radius: int
    ) -> List[Tuple[int, int, int]]:
        """Get coordinates within a given radius."""
        zoom, x, y = coord
        nearby = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:  # Skip the center coordinate
                    continue

                new_coord = (zoom, x + dx, y + dy)
                if new_coord in self.common_coords:
                    nearby.append(new_coord)

        return nearby

    def _get_distance(
        self, coord1: Tuple[int, int, int], coord2: Tuple[int, int, int]
    ) -> float:
        """Calculate Euclidean distance between two tile coordinates."""
        _, x1, y1 = coord1
        _, x2, y2 = coord2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __len__(self):
        return self.max_samples_per_epoch

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a triplet sample: anchor, positive, negative.

        Returns:
            Dict containing:
                - anchor: torch.Tensor (12, H, W)
                - positive: torch.Tensor (12, H, W)
                - negative: torch.Tensor (12, H, W)
                - anchor_coord: Tuple[int, int, int]
                - positive_coord: Tuple[int, int, int]
                - negative_coord: Tuple[int, int, int]
        """
        # Randomly select anchor coordinate
        anchor_coord = random.choice(self.common_coords)

        # Find positive samples (nearby)
        positive_candidates = self._get_nearby_coordinates(
            anchor_coord, self.positive_radius
        )
        if not positive_candidates:
            # If no nearby coordinates, use a random one as fallback
            positive_coord = random.choice(
                [c for c in self.common_coords if c != anchor_coord]
            )
        else:
            positive_coord = random.choice(positive_candidates)

        # Find negative samples (distant)
        negative_candidates = [
            c
            for c in self.common_coords
            if self._get_distance(anchor_coord, c) > self.negative_radius
        ]
        if not negative_candidates:
            # If no distant coordinates, use a random one as fallback
            negative_coord = random.choice(
                [
                    c
                    for c in self.common_coords
                    if c != anchor_coord and c != positive_coord
                ]
            )
        else:
            negative_coord = random.choice(negative_candidates)

        # Load tile stacks
        anchor_tiles = self._load_tile_stack(anchor_coord)
        positive_tiles = self._load_tile_stack(positive_coord)
        negative_tiles = self._load_tile_stack(negative_coord)

        return {
            "anchor": torch.from_numpy(anchor_tiles),
            "positive": torch.from_numpy(positive_tiles),
            "negative": torch.from_numpy(negative_tiles),
            "anchor_coord": anchor_coord,
            "positive_coord": positive_coord,
            "negative_coord": negative_coord,
        }


def create_data_loader(
    tiles_root: str = "12_layer_tiles/tiles",
    batch_size: int = 32,
    num_workers: int = 8,
    tile_size: int = 256,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for the Kyiv tile dataset."""

    dataset = KyivTileDataset(tiles_root=tiles_root, tile_size=tile_size, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the data loader
    loader = create_data_loader(batch_size=2)

    print("Testing data loader...")
    for i, batch in enumerate(loader):
        print(f"Batch {i + 1}:")
        print(f"  Anchor shape: {batch['anchor'].shape}")
        print(f"  Positive shape: {batch['positive'].shape}")
        print(f"  Negative shape: {batch['negative'].shape}")
        print(f"  Anchor coords: {batch['anchor_coord']}")
        print(f"  Positive coords: {batch['positive_coord']}")
        print(f"  Negative coords: {batch['negative_coord']}")

        if i >= 2:  # Just test a few batches
            break
