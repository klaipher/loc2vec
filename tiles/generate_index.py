"""
Generate an index file for the tiles directory.
This script scans the tiles directory and creates a CSV file
with the coordinates and paths of all tiles.
"""

import argparse
import csv
import os
import random
from pathlib import Path

from tqdm import tqdm


# path tiles/full/17/76285/43744.png
def generate_triplets_index(tiles_dir: Path, output_file: Path):
    for dirpath, dirnames, filenames in os.walk(tiles_dir):
        for filename in tqdm(filenames, desc=f"Processing {dirpath}"):
            if filename.endswith('.png'):
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, tiles_dir)
                parts = relative_path.split(os.sep)

                if len(parts) >= 3:
                    zoom = parts[-3]
                    x = parts[-2]
                    y = parts[-1].replace('.png', '')

                    # take any of the tiles that are at least `neg_distance` away as negative
                    neg_path = None
                    while not neg_path and not (os.path.exists(neg_path)):
                        neg_x = int(x) + random.randint(-neg_distance, neg_distance)
                        neg_y = int(y) + random.randint(-neg_distance, neg_distance)

                        if abs(neg_x - int(x)) >= neg_distance or abs(neg_y - int(y)) >= neg_distance:
                            neg_path = os.path.join(tiles_dir, zoom, str(neg_x), f"{neg_y}.png")

                    with open(output_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([zoom, x, y, full_path])
                else:
                    print(f"Skipping file with unexpected path format: {relative_path}")


def take_positive(tiles_dir, x, y, zoom):
    pos_path = None
    # take any of the nearest tiles as positive
    while not pos_path and not (os.path.exists(pos_path)):
        x_shift, y_shift = random.sample([-1, 0, 1], 2)
        pos_x = int(x) + x_shift
        pos_y = int(y) + y_shift

        pos_path = os.path.join(tiles_dir, zoom, str(pos_x), f"{pos_y}.png")
    return pos_path


def take_negative(tiles_dir, x, y, zoom, neg_distance=40):
    neg_path = None
    # take any of the tiles that are at least `neg_distance` away as negative
    while not neg_path and not (os.path.exists(neg_path)):
        neg_x = int(x) + random.randint(-neg_distance, neg_distance)
        neg_y = int(y) + random.randint(-neg_distance, neg_distance)

        if abs(neg_x - int(x)) >= neg_distance or abs(neg_y - int(y)) >= neg_distance:
            neg_path = os.path.join(tiles_dir, zoom, str(neg_x), f"{neg_y}.png")
    return neg_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate index for tiles directory")
    parser.add_argument(
        "--tiles-dir",
        default="./full",
        type=str,
        # required=True,
        help="Path to the tiles directory."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="tiles_index.csv",
        help="Output CSV file for the index."
    )

    args = parser.parse_args()

    tiles_loc = Path(args.tiles_dir)
    output_loc = Path(args.output_file)

    if not tiles_loc.exists() or not tiles_loc.is_dir():
        raise ValueError(f"Tiles directory '{tiles_loc}' does not exist or is not a directory.")

    generate_triplets_index(tiles_loc, output_loc)
