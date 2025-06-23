import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch


class MemmapTripletTilesDataset(Dataset):
    def __init__(
        self,
        metadata_filename,
        npy_data_filename,
        neg_radius_min=10,
        pos_radius=2,
        transform=None,
    ):
        self.metadata_filename = metadata_filename
        self.npy_data_filename = npy_data_filename
        self.neg_radius_min = neg_radius_min
        self.pos_radius = pos_radius
        self.transform = transform

        self.df = pd.read_csv(metadata_filename)
        self.dataset_length = len(self.df)

        self.coordinates = self.df[["x", "y", "zoom"]].astype(np.int32).values

        self._build_spatial_index()
        self._build_negative_index()

        self.images = None
        self._worker_id = None

    def _init_worker_data(self):
        """Read the file per worker."""
        if self.images is None:
            self.images = np.memmap(
                self.npy_data_filename,
                dtype=np.uint8,
                mode="r",
                shape=(self.dataset_length, 6, 128, 128),
            )

    def _build_spatial_index(self):
        """Build spatial index using lightweight numpy arrays."""
        self.spatial_index = {}

        for idx, (x, y, zoom) in enumerate(self.coordinates):
            key = (int(x), int(y), int(zoom))
            if key not in self.spatial_index:
                self.spatial_index[key] = []
            self.spatial_index[key].append(idx)

        for key in self.spatial_index:
            self.spatial_index[key] = np.array(self.spatial_index[key], dtype=np.int32)

    def _build_negative_index(self):
        """Build negative index using numpy arrays only."""
        self.negative_index = {}

        unique_zooms = np.unique(self.coordinates[:, 2])

        for zoom in unique_zooms:
            mask = self.coordinates[:, 2] == zoom
            indices = np.where(mask)[0].astype(np.int32)
            coords = self.coordinates[mask]

            self.negative_index[int(zoom)] = {
                "indices": indices,
                "x": coords[:, 0],
                "y": coords[:, 1],
            }

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        self._init_worker_data()

        x, y, zoom = self.coordinates[idx]
        x, y, zoom = int(x), int(y), int(zoom)

        pos_idx = self._select_positive_sample(x, y, zoom, idx)
        neg_idx = self._select_negative_sample(x, y, zoom, idx)

        anchor_data = self.images[idx]
        pos_data = self.images[pos_idx]
        neg_data = self.images[neg_idx]

        # print(f"anchor shape: {anchor_data.shape}, pos shape: {pos_data.shape}, neg shape: {neg_data.shape}")

        if self.transform:
            # Convert to tensor first, then apply other transforms
            anchor_tensor = torch.from_numpy(np.asarray(anchor_data).astype(np.float32)).div_(255.0)
            pos_tensor = torch.from_numpy(np.asarray(pos_data).astype(np.float32)).div_(255.0)
            neg_tensor = torch.from_numpy(np.asarray(neg_data).astype(np.float32)).div_(255.0)
            
            anchor_tensor = self.transform(anchor_tensor)
            pos_tensor = self.transform(pos_tensor)
            neg_tensor = self.transform(neg_tensor)
        else:
            anchor_tensor = (
                torch.from_numpy(np.asarray(anchor_data).astype(np.float32)).div_(255.0)
            )
            pos_tensor = torch.from_numpy(np.asarray(pos_data).astype(np.float32)).div_(255.0)
            neg_tensor = torch.from_numpy(np.asarray(neg_data).astype(np.float32)).div_(255.0)

        return {
            "anchor_image": anchor_tensor,
            "pos_image": pos_tensor,
            "neg_image": neg_tensor,
            "x": x,
            "y": y,
            "zoom": zoom,
        }

    def _array_to_tensor_via_pil(self, array_data):
        """Convert array to tensor via PIL with minimal memory usage."""
        pil_img = Image.fromarray(np.asarray(array_data).transpose(1, 2, 0))
        tensor = self.transform(pil_img)
        return tensor

    def _select_positive_sample(self, x, y, zoom, anchor_idx, max_attempts=10):
        """Memory-efficient positive sample selection."""
        for _ in range(max_attempts):
            x_shift = random.randint(-self.pos_radius, self.pos_radius)
            y_shift = random.randint(-self.pos_radius, self.pos_radius)

            if x_shift == 0 and y_shift == 0:
                continue

            key = (x + x_shift, y + y_shift, zoom)
            candidates = self.spatial_index.get(key)

            if candidates is not None and len(candidates) > 0:
                return candidates[random.randint(0, len(candidates) - 1)]

        return anchor_idx

    def _select_negative_sample(self, x, y, z, anchor_idx, max_trials: int = 10) -> int:
        block = self.negative_index[z]
        xs, ys, idxs = block["x"], block["y"], block["indices"]

        for _ in range(max_trials):
            j = random.randrange(len(idxs))
            if (
                abs(xs[j] - x) > self.neg_radius_min
                or abs(ys[j] - y) > self.neg_radius_min
            ):
                return int(idxs[j])

        mask = (np.abs(xs - x) > self.neg_radius_min) | (
            np.abs(ys - y) > self.neg_radius_min
        )
        valid = idxs[mask]
        return int(random.choice(valid)) if len(valid) else anchor_idx
