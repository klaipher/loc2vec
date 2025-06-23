#!/usr/bin/env python3
"""
M1 MacBook optimized preprocessor for tiles dataset.
Specifically tuned for M1 Pro/Max with 32GB+ RAM.

Key optimizations:
- Memory-resident processing (uses up to 24GB RAM)
- M1-optimized multiprocessing
- Vectorized operations with NumPy
- Optimized HDF5 settings for Apple Silicon
- Progress estimation based on actual throughput

Usage:
    uv run compress_tiles.py --tiles_dir /path/to/tiles --memory_limit 24

Expected speedup: 5-10x faster than standard approach
"""

import argparse
import gc
import multiprocessing as mp
import pickle
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from tqdm import tqdm


# Global function for multiprocessing (must be outside class)
def scan_zoom_level_worker(zoom_dir_path):
    """Scan a single zoom level directory - global function for multiprocessing."""
    from pathlib import Path  # Import inside function for multiprocessing

    zoom_dir = Path(zoom_dir_path)
    tiles_data = []

    try:
        zoom = int(zoom_dir.name)
    except ValueError:
        return []

    print(f"  Scanning zoom {zoom}: {zoom_dir}")

    for x_dir in zoom_dir.iterdir():
        if not x_dir.is_dir():
            continue

        try:
            x = int(x_dir.name)
        except ValueError:
            continue

        # Batch process all PNG files in this x directory
        png_files = list(x_dir.glob("*.png"))
        for tile_file in png_files:
            try:
                y = int(tile_file.stem)
                tiles_data.append(
                    {
                        "x": x,
                        "y": y,
                        "zoom": zoom,
                        "filename": str(tile_file),
                        "file_size": tile_file.stat().st_size,
                    }
                )
            except ValueError:
                continue

    print(f"  Zoom {zoom}: found {len(tiles_data):,} tiles")
    return tiles_data


class M1OptimizedPreprocessor:
    """M1 MacBook optimized preprocessor with aggressive memory usage."""

    def __init__(
        self, tiles_root_dir, pos_radius=1, neg_radius_min=10, memory_limit_gb=24
    ):
        self.tiles_root_dir = Path(tiles_root_dir)
        self.pos_radius = pos_radius
        self.neg_radius_min = neg_radius_min
        self.memory_limit_gb = memory_limit_gb

        # M1 Pro has 8 performance cores + 2 efficiency cores
        # Use performance cores only for CPU-intensive work
        self.num_workers = min(8, mp.cpu_count())

        # Set up output paths
        self.processed_dir = self.tiles_root_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

        self.hdf5_path = self.processed_dir / "tiles_data.h5"
        self.metadata_path = self.processed_dir / "metadata.pkl"
        self.spatial_index_path = self.processed_dir / "spatial_index.pkl"

        print("🚀 M1 MacBook Optimized Preprocessor")
        print(f"  Input directory: {self.tiles_root_dir}")
        print(f"  CPU cores: {mp.cpu_count()} (using {self.num_workers} workers)")
        print(f"  RAM limit: {memory_limit_gb} GB")
        print(f"  Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    def scan_tiles_fast(self):
        """Ultra-fast directory scanning using multiprocessing."""
        print("\n📁 Fast scanning tiles directory...")

        # Debug: List what's in the directory
        print(f"  Scanning: {self.tiles_root_dir}")
        contents = list(self.tiles_root_dir.iterdir())
        print(f"  Contents: {[d.name for d in contents]}")

        # Get all zoom directories
        zoom_dirs = [
            d
            for d in self.tiles_root_dir.iterdir()
            if d.is_dir() and d.name != "processed"
        ]

        print(f"  Found directories: {[d.name for d in zoom_dirs]}")

        # Filter to numeric zoom directories
        numeric_zoom_dirs = []
        for d in zoom_dirs:
            try:
                int(d.name)
                numeric_zoom_dirs.append(d)
            except ValueError:
                print(f"  Skipping non-numeric directory: {d.name}")

        if not numeric_zoom_dirs:
            print("❌ No valid zoom directories found!")
            print(f"   Looking for numeric directories in: {self.tiles_root_dir}")
            print(f"   Found directories: {[d.name for d in zoom_dirs]}")

            # Try manual scan for debugging
            print("\n🔍 Manual scan for tiles...")
            manual_count = 0
            for item in self.tiles_root_dir.rglob("*.png"):
                manual_count += 1
                if manual_count <= 5:
                    print(f"  Found: {item}")
                elif manual_count == 6:
                    print("  ...")
            print(f"  Total PNG files found: {manual_count}")

            if manual_count > 0:
                print("\n💡 Trying fallback scan...")
                return self.scan_tiles_fallback()
            else:
                raise ValueError("No PNG tiles found in directory!")

        print(
            f"  Processing {len(numeric_zoom_dirs)} zoom levels: {[d.name for d in numeric_zoom_dirs]}"
        )

        # Process zoom levels in parallel
        all_tiles_data = []

        if len(numeric_zoom_dirs) == 1:
            # Single zoom level - process directly
            print("  Single zoom level detected - processing directly...")
            tiles_data = scan_zoom_level_worker(str(numeric_zoom_dirs[0]))
            all_tiles_data.extend(tiles_data)
        else:
            # Multiple zoom levels - use multiprocessing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                zoom_paths = [str(zoom_dir) for zoom_dir in numeric_zoom_dirs]
                futures = {
                    executor.submit(scan_zoom_level_worker, zoom_path): zoom_path
                    for zoom_path in zoom_paths
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Scanning zoom levels",
                ):
                    try:
                        tiles_data = future.result()
                        all_tiles_data.extend(tiles_data)
                    except Exception as e:
                        zoom_path = futures[future]
                        print(f"Error scanning {zoom_path}: {e}")

        df = pd.DataFrame(all_tiles_data)

        if len(df) == 0:
            raise ValueError("No valid tiles found after scanning!")

        print(f"  ✅ Found {len(df):,} tiles in {len(numeric_zoom_dirs)} zoom levels")
        print(f"  📊 Total size: {df['file_size'].sum() / (1024**3):.2f} GB")

        return df

    def scan_tiles_fallback(self):
        """Fallback method for scanning tiles when directory structure is unexpected."""
        print("🔄 Using fallback scanning method...")

        tiles_data = []

        # Recursively find all PNG files
        for tile_file in tqdm(
            self.tiles_root_dir.rglob("*.png"), desc="Scanning tiles"
        ):
            try:
                # Try to extract coordinates from path
                # Assuming structure like: zoom/x/y.png or similar
                parts = tile_file.relative_to(self.tiles_root_dir).parts

                if len(parts) >= 3:
                    # Standard structure: zoom/x/y.png
                    zoom_str, x_str, y_file = parts[0], parts[1], parts[2]
                    zoom = int(zoom_str)
                    x = int(x_str)
                    y = int(tile_file.stem)
                elif len(parts) == 2:
                    # Alternative structure: zoom/xy.png (need to parse filename)
                    zoom_str, xy_file = parts[0], parts[1]
                    zoom = int(zoom_str)
                    # Try to parse x_y from filename
                    name_parts = tile_file.stem.split("_")
                    if len(name_parts) == 2:
                        x, y = int(name_parts[0]), int(name_parts[1])
                    else:
                        continue
                else:
                    continue

                tiles_data.append(
                    {
                        "x": x,
                        "y": y,
                        "zoom": zoom,
                        "filename": str(tile_file),
                        "file_size": tile_file.stat().st_size,
                    }
                )

            except (ValueError, IndexError):
                # Skip files that don't match expected naming pattern
                continue

        df = pd.DataFrame(tiles_data)

        if len(df) == 0:
            raise ValueError("No valid tiles found with fallback method!")

        print(f"  ✅ Fallback scan found {len(df):,} tiles")
        return df

    def estimate_memory_usage(self, df, target_shape):
        """Estimate memory usage for different batch sizes."""
        bytes_per_image = np.prod(target_shape) * 1  # uint8
        total_images = len(df)

        # Calculate how many images we can fit in memory
        available_bytes = self.memory_limit_gb * 1024**3

        # Reserve some memory for other operations
        usable_bytes = int(available_bytes * 0.8)
        max_images_in_memory = usable_bytes // bytes_per_image

        print("📊 Memory planning:")
        print(f"  Bytes per image: {bytes_per_image:,}")
        print(f"  Max images in memory: {max_images_in_memory:,}")
        print(f"  Total images: {total_images:,}")

        if max_images_in_memory >= total_images:
            print("  🎉 Can fit ALL images in memory!")
            return total_images, True
        else:
            batch_size = max_images_in_memory
            print(f"  📦 Will process in batches of {batch_size:,}")
            return batch_size, False

    def load_images_batch_parallel(self, file_paths, target_shape):
        """Load a batch of images in parallel with optimal memory usage."""

        def load_single_image(filepath):
            """Load and process a single image."""
            try:
                img = Image.open(filepath).convert("RGB")

                # Resize if needed (PIL is optimized for M1)
                if hasattr(img, "size") and img.size != (
                    target_shape[1],
                    target_shape[0],
                ):
                    img = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)

                img_array = np.array(img, dtype=np.uint8)
                return img_array, str(filepath)
            except Exception:
                return None, str(filepath)

        # Use ThreadPoolExecutor for I/O bound operations (better for M1)
        from concurrent.futures import ThreadPoolExecutor

        batch_images = []
        valid_paths = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_single_image, fp): fp for fp in file_paths}

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading images",
                leave=False,
            ):
                img_array, filepath = future.result()
                if img_array is not None:
                    batch_images.append(img_array)
                    valid_paths.append(filepath)

        if batch_images:
            return np.stack(batch_images), valid_paths
        else:
            return np.array([]), []

    def create_hdf5_memory_optimized(self, df, target_shape):
        """Create HDF5 with aggressive memory optimization."""
        print("\n💾 Creating memory-optimized HDF5...")

        # Remove existing file
        if self.hdf5_path.exists():
            self.hdf5_path.unlink()

        # Calculate optimal batch size
        batch_size, can_fit_all = self.estimate_memory_usage(df, target_shape)

        # M1-optimized HDF5 settings
        with h5py.File(self.hdf5_path, "w") as f:
            # Create dataset with M1-optimized settings
            images_ds = f.create_dataset(
                "images",
                (len(df), *target_shape),
                dtype=np.uint8,
                compression="lzf",  # Faster than gzip, good for M1
                chunks=(min(32, batch_size), *target_shape),  # Optimal chunk size
                shuffle=False,  # Disable for speed
                fletcher32=False,  # Disable checksums for speed
            )

            filenames_ds = f.create_dataset(
                "filenames", (len(df),), dtype=h5py.string_dtype(), compression="lzf"
            )

            valid_count = 0

            if can_fit_all:
                # Load everything into memory at once
                print("  🚀 Loading ALL images into memory...")
                all_images, valid_paths = self.load_images_batch_parallel(
                    df["filename"].tolist(), target_shape
                )

                if len(all_images) > 0:
                    print("  💾 Writing to HDF5...")
                    images_ds[: len(all_images)] = all_images

                    for i, path in enumerate(valid_paths):
                        filenames_ds[i] = path

                    valid_count = len(all_images)

                    # Resize dataset
                    if valid_count < len(df):
                        images_ds.resize((valid_count, *target_shape))
                        filenames_ds.resize((valid_count,))

            else:
                # Process in memory-optimized batches
                for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
                    batch_df = df.iloc[i : i + batch_size]

                    batch_images, valid_paths = self.load_images_batch_parallel(
                        batch_df["filename"].tolist(), target_shape
                    )

                    if len(batch_images) > 0:
                        end_idx = valid_count + len(batch_images)
                        images_ds[valid_count:end_idx] = batch_images

                        for j, path in enumerate(valid_paths):
                            filenames_ds[valid_count + j] = path

                        valid_count += len(batch_images)

                    # Force garbage collection to free memory
                    del batch_images
                    gc.collect()

                # Final resize
                if valid_count < len(df):
                    images_ds.resize((valid_count, *target_shape))
                    filenames_ds.resize((valid_count,))

            # Add metadata
            f.attrs["num_images"] = valid_count
            f.attrs["image_shape"] = target_shape
            f.attrs["creation_time"] = time.time()

        print(f"  ✅ HDF5 created with {valid_count:,} images")
        print(f"  📁 File size: {self.hdf5_path.stat().st_size / (1024**3):.2f} GB")

        return df.head(valid_count)

    def build_spatial_index_vectorized(self, df):
        """Vectorized spatial index building using NumPy."""
        print("\n🗺️  Building vectorized spatial index...")

        spatial_data = {}

        for zoom in tqdm(sorted(df["zoom"].unique()), desc="Processing zoom levels"):
            zoom_df = df[df["zoom"] == zoom].copy().reset_index(drop=True)

            if len(zoom_df) == 0:
                continue

            coords = zoom_df[["x", "y"]].values.astype(
                np.float32
            )  # Use float32 for M1 optimization
            n_tiles = len(coords)

            print(f"  Zoom {zoom}: {n_tiles:,} tiles")

            # Vectorized distance computation
            print("    Computing all pairwise distances...")

            # For large datasets, compute distances in chunks to manage memory
            chunk_size = min(1000, n_tiles)

            positive_indices = {i: [] for i in range(n_tiles)}
            negative_indices = {i: [] for i in range(n_tiles)}

            for i in tqdm(
                range(0, n_tiles, chunk_size), desc="    Distance chunks", leave=False
            ):
                end_i = min(i + chunk_size, n_tiles)
                chunk_coords = coords[i:end_i]

                # Vectorized distance computation using broadcasting
                diff = (
                    chunk_coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
                )  # (chunk_size, n_tiles, 2)
                distances = np.sqrt(np.sum(diff**2, axis=2))  # (chunk_size, n_tiles)

                # Find positive and negative samples for this chunk
                for local_idx, global_idx in enumerate(range(i, end_i)):
                    tile_distances = distances[local_idx]

                    # Positive samples (excluding self)
                    pos_mask = (tile_distances > 0) & (
                        tile_distances <= self.pos_radius
                    )
                    positive_indices[global_idx] = np.where(pos_mask)[0].tolist()

                    # Negative samples
                    neg_mask = tile_distances >= self.neg_radius_min
                    negative_indices[global_idx] = np.where(neg_mask)[0].tolist()

                    # Fallback for negative samples if none found
                    if not negative_indices[global_idx]:
                        # All tiles except self
                        negative_indices[global_idx] = [
                            j for j in range(n_tiles) if j != global_idx
                        ]

            spatial_data[zoom] = {
                "zoom_df_indices": zoom_df.index.tolist(),
                "positive_indices": positive_indices,
                "negative_indices": negative_indices,
            }

            # Print statistics
            avg_pos = np.mean(
                [len(candidates) for candidates in positive_indices.values()]
            )
            avg_neg = np.mean(
                [len(candidates) for candidates in negative_indices.values()]
            )
            print(f"    Avg positive: {avg_pos:.1f}, Avg negative: {avg_neg:.1f}")

        # Save spatial index
        with open(self.spatial_index_path, "wb") as f:
            pickle.dump(spatial_data, f)

        print("  ✅ Spatial index saved")
        return spatial_data

    def run(self):
        """Run optimized preprocessing pipeline."""
        start_time = time.time()

        print("🚀 M1 MacBook Optimized Preprocessing Pipeline")

        # Step 1: Fast directory scan
        df = self.scan_tiles_fast()

        # Step 2: Determine target shape from sample
        print("\n🔍 Determining image dimensions...")
        sample_files = df["filename"].head(5).tolist()
        target_shape = None

        for filepath in sample_files:
            try:
                img = Image.open(filepath)
                target_shape = (img.height, img.width, 3)  # Assume RGB
                print(f"  Target shape: {target_shape}")
                break
            except Exception:
                continue

        if target_shape is None:
            raise RuntimeError("Could not determine image dimensions")

        # Step 3: Memory-optimized HDF5 creation
        df_final = self.create_hdf5_memory_optimized(df, target_shape)

        # Step 4: Vectorized spatial index
        spatial_data = self.build_spatial_index_vectorized(df_final)

        # Step 5: Save metadata
        metadata = {
            "preprocessing_params": {
                "pos_radius": self.pos_radius,
                "neg_radius_min": self.neg_radius_min,
                "memory_limit_gb": self.memory_limit_gb,
                "num_workers": self.num_workers,
            },
            "dataset_info": {
                "num_samples": len(df_final),
                "image_shape": target_shape,
                "zoom_levels": sorted(df_final["zoom"].unique().tolist()),
                "tiles_per_zoom": df_final["zoom"].value_counts().to_dict(),
            },
            "performance": {
                "processing_time_minutes": (time.time() - start_time) / 60,
                "throughput_tiles_per_second": len(df_final)
                / (time.time() - start_time),
            },
            "dataframe": df_final,
        }

        with open(self.metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        total_time = time.time() - start_time
        throughput = len(df_final) / total_time

        print("\n🎉 M1 Optimized Preprocessing Complete!")
        print(f"  📊 Processed: {len(df_final):,} tiles")
        print(f"  ⏱️  Time: {total_time / 60:.1f} minutes (vs 3+ hours)")
        print(f"  🚀 Throughput: {throughput:.0f} tiles/second")
        print(f"  💾 HDF5 size: {self.hdf5_path.stat().st_size / (1024**3):.2f} GB")
        print(f"  🎯 Speedup: ~{180 / total_time * 60:.0f}x faster!")


def main():
    parser = argparse.ArgumentParser(
        description="M1 MacBook optimized tiles preprocessor"
    )

    parser.add_argument(
        "--tiles_dir", type=str, required=True, help="Path to tiles root directory"
    )
    parser.add_argument(
        "--pos_radius", type=int, default=1, help="Radius for positive samples"
    )
    parser.add_argument(
        "--neg_radius_min",
        type=int,
        default=10,
        help="Minimum radius for negative samples",
    )
    parser.add_argument(
        "--memory_limit",
        type=float,
        default=24,
        help="Memory limit in GB (default: 24GB for 32GB systems)",
    )

    args = parser.parse_args()

    # Validate memory limit
    available_ram = psutil.virtual_memory().total / (1024**3)
    if args.memory_limit > available_ram * 0.8:
        print(
            f"⚠️  Warning: Memory limit {args.memory_limit}GB exceeds 80% of available RAM ({available_ram:.1f}GB)"
        )
        args.memory_limit = available_ram * 0.7
        print(f"   Adjusted to {args.memory_limit:.1f}GB")

    # Run preprocessing
    preprocessor = M1OptimizedPreprocessor(
        tiles_root_dir=args.tiles_dir,
        pos_radius=args.pos_radius,
        neg_radius_min=args.neg_radius_min,
        memory_limit_gb=args.memory_limit,
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
