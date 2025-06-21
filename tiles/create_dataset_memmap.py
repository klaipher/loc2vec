#!/usr/bin/env python3

from argparse import ArgumentParser
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

import pandas as pd
from PIL import Image

def load_image(filename):
    return Image.open(filename).convert("RGB").resize((128, 128))


def list_tiles_to_df(tiles_dir):
    files = []
    for file in Path(tiles_dir).resolve().glob("**/*.png"):
        # if filesize is equal to 103 bytes, skip it
        if file.stat().st_size == 103:  # empty tile
            continue

        files.append((file.parent.name, file.stem, file.parent.parent.name, str(file)))

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
    default="tiles_dataset.h5",
    help="Output HDF5 file for the dataset.",
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

print(f"Creating h5 file from the {len(df)} tiles...")

chunk_size = args.chunk_size
output_file = args.output_file

images = []
for filename in tqdm(df["filename"], desc="Processing tiles", unit="tile"):
    image = load_image(filename)
    image = np.array(image).transpose((2, 0, 1)).astype(np.uint8)
    images.append(image)

images = np.array(images, dtype=np.uint8)
np.save(output_file, images)

print(f"Done {len(images)} images saved to {output_file}.")
