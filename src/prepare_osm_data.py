import argparse
import json
import random
import time
import math
import io
from pathlib import Path
from typing import Tuple, List

import requests
from PIL import Image

# --- Coordinate Conversion Utilities (adapted from OSMTileHandler) ---


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert latitude/longitude to OSM tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def get_region_tile_bounds(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
) -> Tuple[int, int, int, int]:
    """Get the tile bounds for a geographic region."""
    min_tile_x, max_tile_y = latlon_to_tile(min_lat, min_lon, zoom)
    max_tile_x, min_tile_y = latlon_to_tile(max_lat, max_lon, zoom)
    return min_tile_x, min_tile_y, max_tile_x, max_tile_y


# --- Tile Downloading Function ---


def download_tile(
    session: requests.Session,
    server_ip: str,
    server_port: int,
    server_path: str,
    zoom: int,
    x: int,
    y: int,
    cache_dir: Path,
) -> bool:
    """
    Downloads a single tile if it doesn't exist in the cache.
    Returns True if the tile exists or was downloaded successfully, False otherwise.
    """
    cache_file = cache_dir / f"tile_{zoom}_{x}_{y}.png"
    if cache_file.exists():
        # print(f"Tile {zoom}/{x}/{y} found in cache.")
        return True  # Already cached

    max_tile = 2**zoom - 1
    x = max(0, min(x, max_tile))
    y = max(0, min(y, max_tile))

    url = f"http://{server_ip}:{server_port}" + server_path.format(z=zoom, x=x, y=y)

    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Verify content looks like an image before saving
            try:
                img = Image.open(io.BytesIO(response.content))
                img.verify()  # Check if Pillow can decode it
                # Save the raw content to cache
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    f.write(response.content)
                # print(f"Downloaded tile {zoom}/{x}/{y} to {cache_file}")
                return True
            except (Image.UnidentifiedImageError, IOError) as img_err:
                print(
                    f"Warning: Content for tile {zoom}/{x}/{y} from {url} is not a valid image: {img_err}"
                )
                return False  # Treat as failed download

        except requests.exceptions.RequestException as e:
            print(
                f"Error downloading tile {zoom}/{x}/{y} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(
                    f"Failed to download tile {zoom}/{x}/{y} after {max_retries} attempts."
                )
                return False
    return False  # Should not be reached if retries loop finishes


def download_tiles_for_region(
    server_ip: str,
    server_port: int,
    server_path: str,
    zoom: int,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    cache_dir: Path,
):
    """
    Download all OSM tiles within the specified geographic region using direct requests.
    """
    min_tile_x, min_tile_y, max_tile_x, max_tile_y = get_region_tile_bounds(
        min_lat, min_lon, max_lat, max_lon, zoom
    )

    total_tiles = (max_tile_x - min_tile_x + 1) * (max_tile_y - min_tile_y + 1)
    print(
        f"Region covers tiles from ({min_tile_x}, {min_tile_y}) to ({max_tile_x}, {max_tile_y})"
    )
    print(f"Total tiles to check/download: {total_tiles}")

    processed_count = 0
    success_count = 0
    start_time = time.time()

    # Use a session for potential connection pooling
    with requests.Session() as session:
        for y in range(min_tile_y, max_tile_y + 1):
            for x in range(min_tile_x, max_tile_x + 1):
                success = download_tile(
                    session, server_ip, server_port, server_path, zoom, x, y, cache_dir
                )
                processed_count += 1
                if success:
                    success_count += 1

                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(
                        f"Processed {processed_count}/{total_tiles} tiles... Success: {success_count} ({rate:.1f} tiles/s, {elapsed:.1f}s elapsed)"
                    )

    print(
        f"Finished checking/downloading tiles. Processed: {processed_count}, Succeeded/Found: {success_count}"
    )


def generate_sample_coordinates(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    num_samples: int,
    zoom: int,  # Needed for tile boundary check
) -> List[Tuple[float, float]]:
    """
    Generate random sample coordinates within the region bounds.
    """
    sample_points = []
    min_tile_x, min_tile_y, max_tile_x, max_tile_y = get_region_tile_bounds(
        min_lat, min_lon, max_lat, max_lon, zoom
    )

    print(f"Generating {num_samples} random sample points within the region...")
    attempts = 0
    max_attempts = (
        num_samples * 5
    )  # Limit attempts to avoid infinite loop if region is tiny

    while len(sample_points) < num_samples and attempts < max_attempts:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        attempts += 1

        # Check if the point falls within the valid tile range for the region
        tile_x, tile_y = latlon_to_tile(lat, lon, zoom)
        if min_tile_x <= tile_x <= max_tile_x and min_tile_y <= tile_y <= max_tile_y:
            sample_points.append((lat, lon))

    if attempts >= max_attempts:
        print(
            f"Warning: Reached max attempts ({max_attempts}) while generating samples. Generated {len(sample_points)} points."
        )

    return sample_points


def main(args):
    # Create output directory for prepared data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define tile cache directory within the output directory
    cache_dir = output_dir / "tiles"
    cache_dir.mkdir(exist_ok=True)

    print(
        f"Preparing data for region: Lat ({args.min_lat}, {args.max_lat}), Lon ({args.min_lon}, {args.max_lon})"
    )
    print(f"Output directory: {output_dir}")
    print(f"Tile cache directory: {cache_dir}")

    # --- Download Tiles ---
    print("\n--- Downloading Tiles ---")
    download_tiles_for_region(
        server_ip=args.server_ip,
        server_port=args.server_port,
        server_path=args.server_path,
        zoom=args.zoom,
        min_lat=args.min_lat,
        min_lon=args.min_lon,
        max_lat=args.max_lat,
        max_lon=args.max_lon,
        cache_dir=cache_dir,
    )

    # --- Generate Sample Coordinates ---
    print("\n--- Generating Sample Coordinates ---")
    sample_coordinates = generate_sample_coordinates(
        min_lat=args.min_lat,
        min_lon=args.min_lon,
        max_lat=args.max_lat,
        max_lon=args.max_lon,
        num_samples=args.num_samples,
        zoom=args.zoom,  # Pass zoom
    )

    # --- Save Metadata ---
    # Save metadata (coordinates and config)
    metadata = {
        # Store server info for reference, though not used by loader
        "server_ip_source": args.server_ip,
        "server_port_source": args.server_port,
        "server_path_source": args.server_path,
        "zoom": args.zoom,
        "patch_size": args.patch_size,  # The patch size intended for training
        "region_bounds": (args.min_lat, args.min_lon, args.max_lat, args.max_lon),
        "num_samples": len(sample_coordinates),
        "sample_coordinates": sample_coordinates,  # List of [lat, lon] pairs
    }

    metadata_path = output_dir / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(
            f"\nSaved {len(sample_coordinates)} sample coordinates and metadata to {metadata_path}"
        )
    except Exception as e:
        print(f"Error saving metadata file: {e}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare OpenStreetMap data for Loc2Vec training"
    )

    # OSM server parameters (for download)
    parser.add_argument(
        "--server_ip",
        type=str,
        required=True,
        help="IP address of the OpenStreetMap server",
    )
    parser.add_argument(
        "--server_port", type=int, default=80, help="Port of the OpenStreetMap server"
    )
    parser.add_argument(
        "--server_path",
        type=str,
        default="/hot/{z}/{x}/{y}.png",
        help="Path template for OSM tiles",
    )
    parser.add_argument(
        "--zoom", type=int, default=17, help="Zoom level to download tiles for"
    )

    # Region parameters
    parser.add_argument(
        "--min_lat", type=float, default=49.8, help="Minimum latitude for the region"
    )
    parser.add_argument(
        "--min_lon", type=float, default=29.2, help="Minimum longitude for the region"
    )
    parser.add_argument(
        "--max_lat", type=float, default=51.5, help="Maximum latitude for the region"
    )
    parser.add_argument(
        "--max_lon", type=float, default=32.2, help="Maximum longitude for the region"
    )

    # Data parameters
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Size of image patches intended for training (saved in metadata)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of sample coordinates (potential anchors) to generate",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prepared data (tiles/ and metadata.json)",
    )

    args = parser.parse_args()
    main(args)
