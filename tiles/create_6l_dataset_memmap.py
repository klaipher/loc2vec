#!/usr/bin/env python3

from argparse import ArgumentParser
import h5py
import numpy as np
from numpy.random import f
from tqdm import tqdm
from pathlib import Path

import os

import pandas as pd
from PIL import Image

def load_full_image(filename):
    return Image.open(filename).convert("RGB").resize((128, 128))

def load_layer_image(layer_tile_path):
    """
    Load a single tile image from the specified layer directory.
    """
    # tile_path = Path(layer_dir) / str(zoom) / str(x) / f"{y}.png"
    if not Path(layer_tile_path).exists():
        print(f"Tile {layer_tile_path} does not exist. Returning empty tile.")
        return Image.new("L", (128, 128), (255))
    
    return Image.open(layer_tile_path).convert("L").resize((128, 128))


def list_tiles_to_df(tiles_dir, full_layer_dir="/full"):
    files = []

    full_layer_path = Path(tiles_dir + full_layer_dir).resolve()

    for file in full_layer_path.glob("**/*.png"):
        # if filesize is equal to 103 bytes, skip it
        if file.stat().st_size == 103:  # empty tile
            continue

        files.append((file.parent.name, file.stem, file.parent.parent.name, str(file.relative_to(full_layer_path))))

    return pd.DataFrame(files, columns=["x", "y", "zoom", "filename"])


args = ArgumentParser(description="Create a dataset from tiles directory")
args.add_argument(
    "--tiles-dir",
    default="./full",
    required=True,
    type=str,
    help="Path to the tiles directory.",
)
args.add_argument(
    "--metadata-file",
    type=str,
    default="tiles_metadata.csv",
    help="Output CSV file for the metadata.",
)
args.add_argument(
    "--output-file",
    type=str,
    default="tiles_dataset.npy",
    help="Output .npy file for the dataset.",
)
args.add_argument(
    "--chunk-size",
    type=int,
    default=1000,
    help="Number of tiles to process in each chunk.",
)

args = args.parse_args()
print(f"Creating dataset from tiles in {args.tiles_dir}...")

if Path(args.metadata_file).exists():
    print(f"Metadata file {args.metadata_file} already exists. Loading it...")
    df = pd.read_csv(args.metadata_file)
else: 
    df = list_tiles_to_df(args.tiles_dir)
    df.to_csv(args.metadata_file, index=False)

print(f"Creating .npy file from the {len(df)} tiles...")


layers = os.listdir(args.tiles_dir)

print(f"Found {len(layers)} layers: {layers}")

chunk_size = args.chunk_size
output_file = args.output_file

images = []

for filename in tqdm(df["filename"], desc="Processing tiles", unit="tile"):
    image = load_full_image(args.tiles_dir + "/full/" + filename)
    building = load_layer_image(args.tiles_dir + "/buildings/" + filename)
    amenities = load_layer_image(args.tiles_dir + "/amenities/" + filename)
    # roads = load_layer_image(args.tiles_dir + "/roads/" + filename)
    railways = load_layer_image(args.tiles_dir + "/railways/" + filename)
    # landcover = load_layer_image(args.tiles_dir + "/landcover/" + filename)

    image = np.array(image).transpose((2, 0, 1)).astype(np.uint8)
    image = np.concatenate(
        [
            image,
            np.array(building).reshape(1, 128, 128).astype(np.uint8),  # Building layer
            np.array(amenities).reshape(1, 128, 128).astype(np.uint8),  # Amenities layer
            np.array(railways).reshape(1, 128, 128).astype(np.uint8),  # Railways layer
        ],
        axis=0
    )

    images.append(image)

images = np.array(images, dtype=np.uint8)
images = np.ascontiguousarray(images)
np.save(output_file, images)

print(f"Done {len(images)} images saved to {output_file}.")
