import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class KyivTileDataset(Dataset):
    """
    Dataset for loading and processing Kyiv city tiles for loc2vec training.

    Each sample contains:
    - Anchor tile: current location
    - Positive tile: nearby location (within radius)
    - Negative tile: distant location

    Supports two modes:
    1. 'all_layers': Include all individual layers + full layer (if available)
    2. 'full_only': Include only the full layer
    """

    def __init__(
        self,
        tiles_root: str = "tiles/",
        tile_size: int = 256,
        positive_radius: int = 2,  # tiles within this radius are positive
        negative_radius: int = 10,  # tiles outside this radius are negative
        max_samples_per_epoch: int = 10000,
        mode: str = "all_layers",  # 'all_layers' or 'full_only'
    ):
        """
        Args:
            tiles_root: Path to the tiles directory
            tile_size: Size to resize tiles to
            positive_radius: Radius for positive samples (in tile coordinates)
            negative_radius: Minimum radius for negative samples
            max_samples_per_epoch: Maximum number of triplets per epoch
            mode: Loading mode - 'all_layers' (all individual layers + full) or 'full_only' (just full layer)
        """
        self.tiles_root = tiles_root
        self.tile_size = tile_size
        self.positive_radius = positive_radius
        self.negative_radius = negative_radius
        self.max_samples_per_epoch = max_samples_per_epoch
        self.mode = mode

        if mode not in ["all_layers", "full_only"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose 'all_layers' or 'full_only'"
            )

        # Get all layer names
        all_dirs = [
            name
            for name in os.listdir(tiles_root)
            if os.path.isdir(os.path.join(tiles_root, name))
        ]

        # Separate full layer from individual layers
        self.has_full_layer = "full" in all_dirs
        self.individual_layers = [name for name in all_dirs if name != "full"]

        # Set layers to use based on mode
        if mode == "full_only":
            if not self.has_full_layer:
                raise ValueError(
                    "Mode 'full_only' requires a 'full' layer directory, but it was not found"
                )
            self.layers = ["full"]
            self.input_channels = 1
        else:  # all_layers
            self.layers = self.individual_layers.copy()
            if self.has_full_layer:
                self.layers.append("full")
            self.input_channels = len(self.layers)

        print(f"Mode: {mode}")
        print(f"Found {len(all_dirs)} total layer directories: {sorted(all_dirs)}")
        print(f"Has full layer: {self.has_full_layer}")
        print(f"Using {len(self.layers)} layers: {sorted(self.layers)}")
        print(f"Input channels: {self.input_channels}")

        # Load tile coordinates for all layers in use
        self.tile_coords = self._load_tile_coordinates()

        # Find common coordinates across all layers in use
        self.common_coords = self._find_common_coordinates()

        print(
            f"Found {len(self.common_coords)} tiles with data across all required layers"
        )

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
        Load tile layers for a given coordinate and stack them.

        In 'all_layers' mode: loads all individual layers + full layer (if available)
        In 'full_only' mode: loads only the full layer

        Returns:
            np.ndarray: Shape (num_channels, H, W) - stacked tile layers
                       where num_channels depends on mode:
                       - 'full_only': (1, H, W)
                       - 'all_layers': (num_individual_layers + 1, H, W) if full layer exists
                                      or (num_individual_layers, H, W) if no full layer
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

        return np.stack(layers_data, axis=0)  # Shape: (num_channels, H, W)

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
                - anchor: torch.Tensor (num_channels, H, W)
                - positive: torch.Tensor (num_channels, H, W)
                - negative: torch.Tensor (num_channels, H, W)
                - anchor_coord: Tuple[int, int, int]
                - positive_coord: Tuple[int, int, int]
                - negative_coord: Tuple[int, int, int]

            where num_channels depends on mode:
            - 'full_only': 1 channel
            - 'all_layers': number of individual layers + 1 (if full layer exists)
        """
        # Randomly select anchor coordinate
        anchor_coord = random.choice(self.common_coords)

        # Find positive samples (immediate neighbors only - radius 1)
        neighbor_candidates = self._get_nearby_coordinates(anchor_coord, 1)
        if not neighbor_candidates:
            # If no immediate neighbors, use a random nearby tile as fallback
            fallback_candidates = self._get_nearby_coordinates(
                anchor_coord, self.positive_radius
            )
            if fallback_candidates:
                positive_coord = random.choice(fallback_candidates)
            else:
                # If still no candidates, use a random one as final fallback
                positive_coord = random.choice(
                    [c for c in self.common_coords if c != anchor_coord]
                )
        else:
            positive_coord = random.choice(neighbor_candidates)

        # Find negative samples (distant - keep current approach)
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
    tiles_root: str = "tiles/",
    batch_size: int = 32,
    num_workers: int = 8,
    tile_size: int = 256,
    mode: str = "all_layers",
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for the Kyiv tile dataset.

    Args:
        tiles_root: Path to the tiles directory
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        tile_size: Size to resize tiles to
        mode: Loading mode - 'all_layers' or 'full_only'
        **kwargs: Additional arguments passed to KyivTileDataset
    """

    dataset = KyivTileDataset(
        tiles_root=tiles_root, tile_size=tile_size, mode=mode, **kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def visualize_tile_stack(tile_stack: np.ndarray, title: str = "") -> np.ndarray:
    """
    Visualize a tile stack by combining layers into a single image.

    Args:
        tile_stack: Shape (num_channels, H, W) - stack of layers
        title: Title for the visualization

    Returns:
        Combined visualization as RGB image
    """
    num_channels = tile_stack.shape[0]

    # Method 1: Average all layers
    avg_layer = np.mean(tile_stack, axis=0)

    # Method 2: Create RGB from selected layers
    if num_channels >= 3:
        # Use first, middle, and last layers
        rgb_layers = np.stack(
            [
                tile_stack[0],  # Red channel
                tile_stack[num_channels // 2],  # Green channel
                tile_stack[-1],  # Blue channel
            ],
            axis=2,
        )
    elif num_channels == 2:
        # Use both layers and duplicate one
        rgb_layers = np.stack(
            [
                tile_stack[0],  # Red channel
                tile_stack[1],  # Green channel
                tile_stack[0],  # Blue channel (duplicate first)
            ],
            axis=2,
        )
    else:  # num_channels == 1
        # Single channel - convert to RGB by repeating
        rgb_layers = np.stack(
            [
                tile_stack[0],  # Red channel
                tile_stack[0],  # Green channel
                tile_stack[0],  # Blue channel
            ],
            axis=2,
        )

    # Method 3: Create a grid showing all layers
    if num_channels == 1:
        # Special case for single channel
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(tile_stack[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title} - Single Layer")
        ax.axis("off")
    else:
        # Multiple channels - create grid
        cols = min(4, num_channels)
        rows = (num_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        fig.suptitle(f"{title} - All {num_channels} Layers", fontsize=16)

        if rows == 1:
            axes = axes if num_channels > 1 else [axes]
        else:
            axes = axes.flatten()

        for i in range(num_channels):
            axes[i].imshow(tile_stack[i], cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(f"Layer {i}")
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis("off")

    plt.tight_layout()
    return fig, avg_layer, rgb_layers


def create_triplet_visualization(
    anchor, positive, negative, anchor_coord, positive_coord, negative_coord
):
    """
    Create a visualization showing anchor, positive, and negative samples.

    Args:
        anchor, positive, negative: torch.Tensor or np.ndarray of shape (num_channels, H, W)
        anchor_coord, positive_coord, negative_coord: coordinate tuples
    """
    # Convert to numpy if needed
    if isinstance(anchor, torch.Tensor):
        anchor = anchor.numpy()
    if isinstance(positive, torch.Tensor):
        positive = positive.numpy()
    if isinstance(negative, torch.Tensor):
        negative = negative.numpy()

    num_channels = anchor.shape[0]

    # Create combined visualization
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

    # Top row: Average of all layers for each sample
    samples = [
        (anchor, f"Anchor\n{anchor_coord}", 0),
        (positive, f"Positive\n{positive_coord}", 1),
        (negative, f"Negative\n{negative_coord}", 2),
    ]

    for i, (sample, title, col) in enumerate(samples):
        # Average visualization
        ax_avg = fig.add_subplot(gs[0, col])
        avg_img = np.mean(sample, axis=0)
        ax_avg.imshow(avg_img, cmap="viridis", vmin=0, vmax=1)
        ax_avg.set_title(
            f"{title}\nAverage of {num_channels} Layer{'s' if num_channels != 1 else ''}"
        )
        ax_avg.axis("off")

        # RGB combination
        ax_rgb = fig.add_subplot(gs[1, col])

        if num_channels >= 3:
            # Use first, middle, and last layers
            rgb_img = np.stack(
                [
                    sample[0],  # Red
                    sample[num_channels // 2],  # Green
                    sample[-1],  # Blue
                ],
                axis=2,
            )
            rgb_title = (
                f"RGB Combination\n(Layers 0,{num_channels // 2},{num_channels - 1})"
            )
        elif num_channels == 2:
            # Use both layers and duplicate first
            rgb_img = np.stack(
                [
                    sample[0],  # Red
                    sample[1],  # Green
                    sample[0],  # Blue (duplicate)
                ],
                axis=2,
            )
            rgb_title = "RGB Combination\n(Layers 0,1,0)"
        else:  # num_channels == 1
            # Single channel - convert to RGB by repeating
            rgb_img = np.stack(
                [
                    sample[0],  # Red
                    sample[0],  # Green
                    sample[0],  # Blue
                ],
                axis=2,
            )
            rgb_title = "Single Layer\n(Grayscale as RGB)"

        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(rgb_title)
        ax_rgb.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test both loading modes
    print("=" * 60)
    print("TESTING DATA LOADER WITH DIFFERENT MODES")
    print("=" * 60)

    # Create output directory for demo images
    demo_dir = "demo_visualizations"
    os.makedirs(demo_dir, exist_ok=True)

    # Test both modes
    modes_to_test = ["all_layers", "full_only"]

    for mode in modes_to_test:
        print(f"\n{'=' * 40}")
        print(f"TESTING MODE: {mode.upper()}")
        print(f"{'=' * 40}")

        try:
            # Create data loader for current mode
            loader = create_data_loader(batch_size=2, mode=mode)

            print(f"Testing data loader in {mode} mode...")
            print(f"Input channels: {loader.dataset.input_channels}")

            # Test a few batches
            for i, batch in enumerate(loader):
                print(f"\nBatch {i + 1}:")
                print(f"  Anchor shape: {batch['anchor'].shape}")
                print(f"  Positive shape: {batch['positive'].shape}")
                print(f"  Negative shape: {batch['negative'].shape}")
                print(f"  Anchor coords: {batch['anchor_coord']}")
                print(f"  Positive coords: {batch['positive_coord']}")
                print(f"  Negative coords: {batch['negative_coord']}")

                # Create visualization for first sample in first batch
                if i == 0:
                    sample_idx = 0
                    anchor = batch["anchor"][sample_idx]
                    positive = batch["positive"][sample_idx]
                    negative = batch["negative"][sample_idx]

                    anchor_coord = batch["anchor_coord"][sample_idx]
                    positive_coord = batch["positive_coord"][sample_idx]
                    negative_coord = batch["negative_coord"][sample_idx]

                    print(f"\nCreating demo visualizations for {mode} mode...")

                    # Create triplet visualization
                    triplet_fig = create_triplet_visualization(
                        anchor,
                        positive,
                        negative,
                        anchor_coord,
                        positive_coord,
                        negative_coord,
                    )
                    triplet_path = os.path.join(
                        demo_dir, f"triplet_demo_{mode}_batch_{i + 1}.png"
                    )
                    triplet_fig.savefig(triplet_path, dpi=150, bbox_inches="tight")
                    print(f"Saved triplet visualization: {triplet_path}")
                    plt.close(triplet_fig)

                    # Create detailed layer visualization for anchor
                    anchor_fig, avg_layer, rgb_layers = visualize_tile_stack(
                        anchor.numpy(), f"Anchor Sample {anchor_coord} ({mode} mode)"
                    )
                    anchor_path = os.path.join(
                        demo_dir, f"anchor_layers_{mode}_batch_{i + 1}.png"
                    )
                    anchor_fig.savefig(anchor_path, dpi=150, bbox_inches="tight")
                    print(f"Saved anchor layers visualization: {anchor_path}")
                    plt.close(anchor_fig)

                    # Save individual combined images
                    avg_path = os.path.join(
                        demo_dir, f"anchor_average_{mode}_batch_{i + 1}.png"
                    )
                    plt.figure(figsize=(8, 8))
                    plt.imshow(avg_layer, cmap="viridis")
                    num_channels = anchor.shape[0]
                    plt.title(
                        f"Anchor Average - {num_channels} Layer{'s' if num_channels != 1 else ''} ({mode} mode)\nCoord: {anchor_coord}"
                    )
                    plt.axis("off")
                    plt.savefig(avg_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"Saved anchor average: {avg_path}")

                    rgb_path = os.path.join(
                        demo_dir, f"anchor_rgb_{mode}_batch_{i + 1}.png"
                    )
                    plt.figure(figsize=(8, 8))
                    plt.imshow(rgb_layers)
                    plt.title(
                        f"Anchor RGB Combination ({mode} mode)\nCoord: {anchor_coord}"
                    )
                    plt.axis("off")
                    plt.savefig(rgb_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"Saved anchor RGB: {rgb_path}")

                if i >= 1:  # Just test a couple of batches per mode
                    break

        except Exception as e:
            print(f"Error testing {mode} mode: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("DEMO COMPLETE")
    print(f"{'=' * 60}")
    print(f"Demo visualizations saved in '{demo_dir}' directory")
    print("Generated files for each mode:")
    print(
        "- triplet_demo_{mode}_batch_X.png: Shows anchor, positive, negative triplets"
    )
    print("- anchor_layers_{mode}_batch_X.png: Shows all layers of anchor sample")
    print("- anchor_average_{mode}_batch_X.png: Average of all layers")
    print("- anchor_rgb_{mode}_batch_X.png: RGB combination of select layers")
    print("\nModes:")
    print("- all_layers: Uses all individual layers + full layer (if available)")
    print("- full_only: Uses only the full layer (1 channel)")
