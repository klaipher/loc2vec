import math
import random
import json
from typing import Tuple, Dict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class OSMTileHandler:
    """Class to handle locally stored OpenStreetMap tile loading and processing."""

    def __init__(self, tile_dir: str, zoom: int, patch_size: int = 64):
        """
        Initialize the OpenStreetMap tile handler for locally stored tiles.

        Args:
            tile_dir: Directory containing the pre-downloaded tiles.
            zoom: Zoom level of the tiles.
            patch_size: Size of the image patches to extract.
        """
        self.tile_dir = Path(tile_dir)
        if not self.tile_dir.is_dir():
            raise FileNotFoundError(f"Tile directory not found: {self.tile_dir}")

        self.zoom = zoom
        self.patch_size = patch_size

        # Calculate max tile coordinates for this zoom level
        self.max_tile = 2**zoom - 1

    def latlon_to_tile(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert latitude/longitude to OSM tile coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple[int, int]: (x, y) tile coordinates
        """
        lat_rad = math.radians(lat)
        n = 2.0**self.zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def tile_to_latlon(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert OSM tile coordinates to latitude/longitude.

        Args:
            x: Tile x coordinate
            y: Tile y coordinate

        Returns:
            Tuple[float, float]: (latitude, longitude) in degrees
        """
        n = 2.0**self.zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def get_tile_path(self, x: int, y: int) -> Path:
        """Get the expected path for a tile file."""
        return self.tile_dir / f"tile_{self.zoom}_{x}_{y}.png"

    def get_tile(self, x: int, y: int) -> np.ndarray:
        """
        Load a tile from the local tile directory.

        Args:
            x: Tile x coordinate
            y: Tile y coordinate

        Returns:
            np.ndarray: Tile image as a numpy array with shape (channels, height, width).
                        Returns a blank tile if the file is missing or corrupted.
        """
        tile_file = self.get_tile_path(x, y)

        if tile_file.exists():
            try:
                img = Image.open(tile_file)
                # Convert to numpy array and transpose to (channels, height, width)
                # Ensure image has 3 channels (e.g., handle grayscale)
                img_array = np.array(img)
                if img_array.ndim == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]  # Drop alpha channel

                return img_array.transpose(2, 0, 1)
            except Exception as e:
                print(
                    f"Warning: Error loading tile {tile_file}: {e}. Returning blank tile."
                )
                # Return a blank tile as fallback
                blank_tile = np.zeros((3, 256, 256), dtype=np.uint8)
                return blank_tile
        else:
            print(f"Warning: Tile file not found: {tile_file}. Returning blank tile.")
            # Return a blank tile as fallback
            blank_tile = np.zeros((3, 256, 256), dtype=np.uint8)
            return blank_tile

    def extract_patch(self, tile_x: int, tile_y: int, px: int, py: int) -> np.ndarray:
        """
        Extract a patch from a specific position within a tile, potentially
        loading adjacent tiles if the patch crosses boundaries.

        Args:
            tile_x: Tile x coordinate
            tile_y: Tile y coordinate
            px: Pixel x coordinate within the tile (0-255)
            py: Pixel y coordinate within the tile (0-255)

        Returns:
            np.ndarray: Image patch of shape (channels, patch_size, patch_size)
        """
        # Ensure pixel coordinates are within valid range
        px = max(0, min(px, 255))
        py = max(0, min(py, 255))

        # Get the primary tile
        tile = self.get_tile(tile_x, tile_y)
        tile_height, tile_width = tile.shape[1:]

        patch = np.zeros(
            (tile.shape[0], self.patch_size, self.patch_size), dtype=tile.dtype
        )

        # Calculate the portion of the patch coming from each of the four potential tiles
        x_start_in_patch = max(0, -px)
        y_start_in_patch = max(0, -py)
        x_end_in_patch = min(self.patch_size, tile_width - px)
        y_end_in_patch = min(self.patch_size, tile_height - py)

        # Top-left tile (primary)
        patch[:, y_start_in_patch:y_end_in_patch, x_start_in_patch:x_end_in_patch] = (
            tile[
                :,
                max(0, py) : min(tile_height, py + self.patch_size),
                max(0, px) : min(tile_width, px + self.patch_size),
            ]
        )

        # Top-right tile (if needed)
        if px + self.patch_size > tile_width:
            right_tile = self.get_tile(tile_x + 1, tile_y)
            x_start_in_patch = tile_width - px
            x_end_in_patch = self.patch_size

            patch[
                :, y_start_in_patch:y_end_in_patch, x_start_in_patch:x_end_in_patch
            ] = right_tile[
                :,
                max(0, py) : min(tile_height, py + self.patch_size),
                0 : min(tile_width, self.patch_size - (tile_width - px)),
            ]

        # Bottom-left tile (if needed)
        if py + self.patch_size > tile_height:
            bottom_tile = self.get_tile(tile_x, tile_y + 1)
            y_start_in_patch = tile_height - py
            y_end_in_patch = self.patch_size

            patch[
                :, y_start_in_patch:y_end_in_patch, x_start_in_patch:x_end_in_patch
            ] = bottom_tile[
                :,
                0 : min(tile_height, self.patch_size - (tile_height - py)),
                max(0, px) : min(tile_width, px + self.patch_size),
            ]

        # Bottom-right tile (if needed)
        if px + self.patch_size > tile_width and py + self.patch_size > tile_height:
            bottom_right_tile = self.get_tile(tile_x + 1, tile_y + 1)
            x_start_in_patch = tile_width - px
            y_start_in_patch = tile_height - py
            x_end_in_patch = self.patch_size
            y_end_in_patch = self.patch_size

            patch[
                :, y_start_in_patch:y_end_in_patch, x_start_in_patch:x_end_in_patch
            ] = bottom_right_tile[
                :,
                0 : min(tile_height, self.patch_size - (tile_height - py)),
                0 : min(tile_width, self.patch_size - (tile_width - px)),
            ]

        return patch

    def extract_patch_from_latlon(
        self, lat: float, lon: float
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract a patch centered at given latitude and longitude from local tiles.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple containing:
                - np.ndarray: Image patch of shape (channels, patch_size, patch_size)
                - Tuple[int, int, int, int]: (tile_x, tile_y, pixel_x, pixel_y) coordinates of top-left corner
        """
        # Convert lat/lon to tile coordinates
        center_tile_x, center_tile_y = self.latlon_to_tile(lat, lon)

        # Calculate pixel coordinates of the center within its tile
        n = 2.0**self.zoom
        lat_rad = math.radians(lat)
        x_pixel_center = ((lon + 180.0) / 360.0 * n - center_tile_x) * 256
        y_pixel_center = (
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n - center_tile_y
        ) * 256

        # Calculate top-left corner of the patch in pixel coordinates relative to the top-left of the center tile
        px_in_tile = x_pixel_center - self.patch_size // 2
        py_in_tile = y_pixel_center - self.patch_size // 2

        # Determine the primary tile (top-left tile the patch overlaps)
        tile_x = center_tile_x
        tile_y = center_tile_y
        px = int(px_in_tile)
        py = int(py_in_tile)

        # Adjust tile coordinates and pixel offsets if patch starts in an adjacent tile
        if px < 0:
            tile_x -= 1
            px += 256
        if py < 0:
            tile_y -= 1
            py += 256

        # Extract the patch using the potentially adjusted tile coordinates and pixel offset
        patch = self.extract_patch(tile_x, tile_y, px, py)

        return patch, (tile_x, tile_y, px, py)

    def get_region_bounds(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> Tuple[int, int, int, int]:
        """
        Get the tile bounds for a geographic region.

        Args:
            min_lat: Minimum latitude
            min_lon: Minimum longitude
            max_lat: Maximum latitude
            max_lon: Maximum longitude

        Returns:
            Tuple[int, int, int, int]: (min_tile_x, min_tile_y, max_tile_x, max_tile_y)
        """
        min_tile_x, max_tile_y = self.latlon_to_tile(min_lat, min_lon)
        max_tile_x, min_tile_y = self.latlon_to_tile(max_lat, max_lon)

        return min_tile_x, min_tile_y, max_tile_x, max_tile_y


class OSMLoc2VecDataset(Dataset):
    """Dataset for training Loc2Vec with triplet loss using pre-prepared OpenStreetMap data."""

    def __init__(
        self,
        prepared_data_dir: str,
        transform=None,
        max_distance_positive: float = 0.001,  # degrees
        min_distance_negative: float = 0.01,  # degrees
    ):
        """
        Initialize the OSM-based Loc2Vec dataset from prepared data.

        Args:
            prepared_data_dir: Directory containing pre-downloaded tiles and metadata.json
            transform: PyTorch transforms to apply to the images
            max_distance_positive: Maximum distance (in degrees) for positive samples
            min_distance_negative: Minimum distance (in degrees) for negative samples
        """
        self.prepared_data_dir = Path(prepared_data_dir)
        self.metadata_path = self.prepared_data_dir / "metadata.json"
        self.tile_dir = self.prepared_data_dir / "tiles"

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.tile_dir.is_dir():
            raise FileNotFoundError(f"Tile directory not found: {self.tile_dir}")

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.zoom = self.metadata["zoom"]
        self.patch_size = self.metadata["patch_size"]
        self.region_bounds = tuple(self.metadata["region_bounds"])
        self.sample_points = self.metadata[
            "sample_coordinates"
        ]  # List of [lat, lon] pairs

        print(
            f"Loaded metadata for {len(self.sample_points)} samples from {self.metadata_path}"
        )
        print(f"Using tiles from: {self.tile_dir}")
        print(f"Zoom: {self.zoom}, Patch Size: {self.patch_size}")
        print(f"Region Bounds: {self.region_bounds}")

        # Initialize tile handler with the local tile directory
        self.tile_handler = OSMTileHandler(
            tile_dir=str(self.tile_dir), zoom=self.zoom, patch_size=self.patch_size
        )

        self.transform = transform
        self.max_distance_positive = max_distance_positive
        self.min_distance_negative = min_distance_negative

    def __len__(self) -> int:
        return len(self.sample_points)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a triplet (anchor, positive, negative) for training.

        Args:
            idx: Index of the sample

        Returns:
            Dict containing 'anchor', 'positive', and 'negative' tensors
        """
        # Get anchor coordinates from pre-generated list
        anchor_lat, anchor_lon = self.sample_points[idx]

        # Extract anchor patch
        anchor_patch, anchor_coords = self.tile_handler.extract_patch_from_latlon(
            anchor_lat, anchor_lon
        )

        # Find positive sample (nearby location)
        # Random offset for positive sample (small distance)
        offset_lat = random.uniform(
            -self.max_distance_positive, self.max_distance_positive
        )
        offset_lon = random.uniform(
            -self.max_distance_positive, self.max_distance_positive
        )

        positive_lat = anchor_lat + offset_lat
        positive_lon = anchor_lon + offset_lon

        # Keep within region bounds
        positive_lat = max(
            self.region_bounds[0], min(positive_lat, self.region_bounds[2])
        )
        positive_lon = max(
            self.region_bounds[1], min(positive_lon, self.region_bounds[3])
        )

        # Extract positive patch
        positive_patch, positive_coords = self.tile_handler.extract_patch_from_latlon(
            positive_lat, positive_lon
        )

        # Find negative sample (distant location)
        while True:
            # Generate a candidate negative sample far from anchor
            # We can sample from the pre-generated list or randomly within bounds
            # Sampling randomly within bounds might be simpler here
            negative_lat = random.uniform(self.region_bounds[0], self.region_bounds[2])
            negative_lon = random.uniform(self.region_bounds[1], self.region_bounds[3])

            # Calculate distance to ensure it's far enough
            lat_distance = abs(anchor_lat - negative_lat)
            lon_distance = abs(anchor_lon - negative_lon)
            distance = math.sqrt(lat_distance**2 + lon_distance**2)

            # If far enough, use this as negative
            if distance > self.min_distance_negative:
                break

        # Extract negative patch
        negative_patch, negative_coords = self.tile_handler.extract_patch_from_latlon(
            negative_lat, negative_lon
        )

        # Convert to torch tensors
        anchor_tensor = torch.from_numpy(anchor_patch).float()
        positive_tensor = torch.from_numpy(positive_patch).float()
        negative_tensor = torch.from_numpy(negative_patch).float()

        # Apply transforms if specified
        if self.transform:
            anchor_tensor = self.transform(anchor_tensor)
            positive_tensor = self.transform(positive_tensor)
            negative_tensor = self.transform(negative_tensor)

        # Normalize to [0, 1] if not already done by transforms
        # OSM tiles are usually already 0-255 uint8
        if anchor_tensor.max() > 1.0:
            anchor_tensor = anchor_tensor / 255.0
            positive_tensor = positive_tensor / 255.0
            negative_tensor = negative_tensor / 255.0

        return {
            "anchor": anchor_tensor,
            "positive": positive_tensor,
            "negative": negative_tensor,
            "anchor_coords": (anchor_lat, anchor_lon),
            "positive_coords": (positive_lat, positive_lon),
            "negative_coords": (negative_lat, negative_lon),
        }


def create_osm_dataloader(
    prepared_data_dir: str,
    batch_size: int = 32,
    transform=None,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Loc2Vec training using pre-prepared OpenStreetMap data.

    Args:
        prepared_data_dir: Directory containing pre-downloaded tiles and metadata.json
        batch_size: Batch size for training
        transform: PyTorch transforms to apply
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for OSMLoc2VecDataset (e.g., distances)

    Returns:
        DataLoader: PyTorch DataLoader for training
    """
    dataset = OSMLoc2VecDataset(
        prepared_data_dir=prepared_data_dir, transform=transform, **kwargs
    )

    # Pin memory only if CUDA is available
    pin_memory_flag = torch.cuda.is_available()
    print(f"DataLoader: Setting pin_memory={pin_memory_flag}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,  # Use the conditional flag
    )
