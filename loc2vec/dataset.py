import random
from pathlib import Path
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
        if file.stat().st_size == 103:  # empty tile
            continue

        files.append((file.parent.name, file.stem, file.parent.parent.name, str(file)))

    return pd.DataFrame(files, columns=["x", "y", "zoom", "filename"])


class OptimizedTilesDataset(Dataset):
    def __init__(
        self,
        tiles_root_dir,
        pos_radius=1,
        neg_radius_min=10,
        transform=None,
        preload_images=True,
    ):
        self.tiles_root_dir = tiles_root_dir
        self.df = list_tiles_to_df(tiles_root_dir)
        self.pos_radius = pos_radius
        self.neg_radius_min = neg_radius_min
        self.transform = transform
        self.preload_images = preload_images

        print(f"Found {len(self.df)} valid tiles")

        # Convert coordinates to integers once
        self.df["x_int"] = self.df["x"].astype(int)
        self.df["y_int"] = self.df["y"].astype(int)

        # Preload all images if requested
        if preload_images:
            print("Preloading all images into memory...")
            self.images = {}
            self.positive_candidates = {}
            self.negative_candidates = {}

            self._preload_images()
            self._precompute_candidates()
            print("Preloading complete!")
        else:
            self.images = None
            self._precompute_candidates_lazy()

    def _preload_images(self):
        """Load all images into memory during initialization."""
        from tqdm import tqdm

        for idx, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Loading images"
        ):
            filename = row["filename"]
            try:
                self.images[idx] = load_image(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                # Create a blank image as fallback
                self.images[idx] = Image.new("RGB", (256, 256), color="black")

    def _precompute_candidates(self):
        """Precompute positive and negative candidates for each sample."""
        from tqdm import tqdm

        for idx, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Computing candidates"
        ):
            x, y, zoom = row["x_int"], row["y_int"], row["zoom"]

            # Find positive candidates (within pos_radius)
            pos_mask = (
                (self.df["zoom"] == zoom)
                & (abs(self.df["x_int"] - x) <= self.pos_radius)
                & (abs(self.df["y_int"] - y) <= self.pos_radius)
                & (self.df.index != idx)  # Don't include self
            )
            pos_indices = self.df[pos_mask].index.tolist()

            # If no positive candidates, use self as fallback
            if not pos_indices:
                pos_indices = [idx]

            self.positive_candidates[idx] = pos_indices

            # Find negative candidates (outside neg_radius_min)
            neg_mask = (
                (self.df["zoom"] == zoom)
                & (
                    (abs(self.df["x_int"] - x) > self.neg_radius_min)
                    | (abs(self.df["y_int"] - y) > self.neg_radius_min)
                )
                & (self.df.index != idx)  # Don't include self
            )
            neg_indices = self.df[neg_mask].index.tolist()

            # If no suitable negatives, use all others as candidates
            if not neg_indices:
                neg_indices = [i for i in self.df.index if i != idx]

            self.negative_candidates[idx] = neg_indices

    def _precompute_candidates_lazy(self):
        """Lighter version that precomputes candidate indices without loading images."""
        from tqdm import tqdm

        self.positive_candidates = {}
        self.negative_candidates = {}

        for idx, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Computing candidates"
        ):
            x, y, zoom = row["x_int"], row["y_int"], row["zoom"]

            # Find positive candidates
            pos_mask = (
                (self.df["zoom"] == zoom)
                & (abs(self.df["x_int"] - x) <= self.pos_radius)
                & (abs(self.df["y_int"] - y) <= self.pos_radius)
                & (self.df.index != idx)
            )
            pos_indices = self.df[pos_mask].index.tolist()
            if not pos_indices:
                pos_indices = [idx]
            self.positive_candidates[idx] = pos_indices

            # Find negative candidates
            neg_mask = (
                (self.df["zoom"] == zoom)
                & (
                    (abs(self.df["x_int"] - x) > self.neg_radius_min)
                    | (abs(self.df["y_int"] - y) > self.neg_radius_min)
                )
                & (self.df.index != idx)
            )
            neg_indices = self.df[neg_mask].index.tolist()
            if not neg_indices:
                neg_indices = [i for i in self.df.index if i != idx]
            self.negative_candidates[idx] = neg_indices

    def _get_image(self, idx):
        """Get image either from preloaded cache or load on demand."""
        if self.preload_images:
            return self.images[idx]
        else:
            filename = self.df.iloc[idx]["filename"]
            try:
                return load_image(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                return Image.new("RGB", (256, 256), color="black")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get anchor image
        anchor_image = self._get_image(idx)

        # Get positive sample (random choice from precomputed candidates)
        pos_idx = random.choice(self.positive_candidates[idx])
        pos_image = self._get_image(pos_idx)

        # Get negative sample (random choice from precomputed candidates)
        neg_idx = random.choice(self.negative_candidates[idx])
        neg_image = self._get_image(neg_idx)

        # Apply transforms if specified
        if self.transform:
            anchor_image = self.transform(anchor_image)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        # Get metadata
        row = self.df.iloc[idx]

        return {
            "anchor_image": anchor_image,
            "pos_image": pos_image,
            "neg_image": neg_image,
            "x": row["x"],
            "y": row["y"],
            "zoom": row["zoom"],
            "filename": row["filename"],
        }


# Backward compatibility - use optimized version by default
class TilesDataset(OptimizedTilesDataset):
    def __init__(self, tiles_root_dir, pos_radius=1, neg_radius_min=10, transform=None):
        # Default to preloading images for maximum performance
        super().__init__(
            tiles_root_dir, pos_radius, neg_radius_min, transform, preload_images=True
        )


# Memory-efficient version for very large datasets
class LazyTilesDataset(OptimizedTilesDataset):
    def __init__(self, tiles_root_dir, pos_radius=1, neg_radius_min=10, transform=None):
        # Don't preload images, but still precompute candidates
        super().__init__(
            tiles_root_dir, pos_radius, neg_radius_min, transform, preload_images=False
        )


if __name__ == "__main__":
    dataset = TilesDataset("../full", pos_radius=1, transform=T.ToTensor())
    print(len(dataset))
    print(f"Dataset length: {len(dataset)}")

    sample = random.choice(dataset)
    print(
        f"Sample: {sample['filename']}, Coordinates: ({sample['x']}, {sample['y']}), Zoom: {sample['zoom']}"
    )

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample["anchor_image"].permute(1, 2, 0))
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(sample["pos_image"].permute(1, 2, 0))
    plt.title("Transformed Image")
    plt.subplot(1, 3, 3)
    plt.imshow(sample["neg_image"].permute(1, 2, 0))
    plt.title("Negative Image")
    plt.show()
