import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import random
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import folium
from folium.plugins import MarkerCluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model import Loc2Vec
from osm_data import OSMTileHandler


def load_model(model_path: str, device: torch.device) -> Tuple[Loc2Vec, dict]:
    """
    Load a trained Loc2Vec model.

    Args:
        model_path: Path to saved model file
        device: Device to load model on

    Returns:
        Tuple containing the model and metadata
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model metadata from the checkpoint itself
    model_metadata = {
        "embedding_dim": checkpoint.get("embedding_dim", 64),
        "input_channels": checkpoint.get("input_channels", 3),
        "patch_size": checkpoint.get("patch_size", 64),
    }
    print(f"Model metadata from checkpoint: {model_metadata}")

    # Create model
    model = Loc2Vec(
        input_channels=model_metadata["input_channels"],
        embedding_dim=model_metadata["embedding_dim"],
    ).to(device)

    # Load state dict
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        # Maybe the checkpoint only contains the state dict
        state_dict = checkpoint
        print("Checkpoint seems to contain only state_dict.")
        if (
            "loc2vec.encoder.0.weight" not in state_dict
            and "encoder.0.weight" not in state_dict
        ):
            raise ValueError("Could not find model weights in checkpoint or state_dict")

    # Handle checkpoints saved from TripletLoc2Vec
    if "loc2vec.encoder.0.weight" in state_dict:
        print("Loading weights from TripletLoc2Vec checkpoint structure.")
        # Extract just the Loc2Vec part
        new_state_dict = {}
        for key in state_dict:
            if key.startswith("loc2vec."):
                new_key = key[len("loc2vec.") :]
                new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_metadata


def generate_embeddings(
    model: Loc2Vec,
    tile_handler: OSMTileHandler,
    coordinates: List[Tuple[float, float]],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings for a list of coordinates using local tiles.

    Args:
        model: Trained Loc2Vec model
        tile_handler: OSM tile handler configured for local tiles
        coordinates: List of (latitude, longitude) coordinates
        device: Device to run inference on
        batch_size: Batch size for processing (currently processes one by one)

    Returns:
        Tuple of (embeddings, coordinates) arrays for successfully processed points
    """
    embeddings = []
    processed_coords = []
    failed_count = 0

    # Process coordinates one by one (batching could be added for efficiency)
    with torch.no_grad():
        for i, (lat, lon) in enumerate(coordinates):
            if (i + 1) % 100 == 0:
                print(
                    f"Processing coordinate {i + 1}/{len(coordinates)} (Failed: {failed_count})"
                )

            try:
                # Extract patch using the local tile handler
                patch, _ = tile_handler.extract_patch_from_latlon(lat, lon)

                # Convert to tensor and normalize
                patch_tensor = torch.from_numpy(patch).float()

                # Ensure patch is the correct size (sometimes edge patches might be smaller)
                if patch_tensor.shape[1:] != (
                    tile_handler.patch_size,
                    tile_handler.patch_size,
                ):
                    # Handle incorrect size, e.g., skip or pad
                    # print(f"Warning: Skipping patch at ({lat:.5f}, {lon:.5f}) due to unexpected size {patch_tensor.shape[1:]}")
                    # For now, just skip
                    # failed_count += 1
                    # continue
                    # Or try padding (might introduce artifacts)
                    from torchvision.transforms.functional import pad

                    padding_h = tile_handler.patch_size - patch_tensor.shape[1]
                    padding_w = tile_handler.patch_size - patch_tensor.shape[2]
                    # Pad right and bottom
                    patch_tensor = pad(patch_tensor, [0, 0, padding_w, padding_h])

                if patch_tensor.max() > 1.0:
                    patch_tensor = patch_tensor / 255.0

                # Add batch dimension and send to device
                patch_tensor = patch_tensor.unsqueeze(0).to(device)

                # Compute embedding
                embedding = model(patch_tensor).cpu().numpy()[0]
                embeddings.append(embedding)
                processed_coords.append((lat, lon))

            except Exception as e:
                print(f"Error processing coordinate ({lat}, {lon}): {e}")
                failed_count += 1
                continue

    print(
        f"Finished generating embeddings. Successfully processed: {len(processed_coords)}, Failed: {failed_count}"
    )
    return np.array(embeddings), np.array(processed_coords)


def reduce_dimensions(
    embeddings: np.ndarray, method: str = "pca", n_components: int = 3
) -> np.ndarray:
    """
    Reduce dimensionality of embeddings for visualization.

    Args:
        embeddings: Array of embeddings
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components to reduce to

    Returns:
        Reduced embeddings
    """
    if embeddings.shape[0] < n_components:
        print(
            f"Warning: Number of samples ({embeddings.shape[0]}) is less than n_components ({n_components}). Cannot reduce dimensions."
        )
        # Return original embeddings or handle appropriately
        return embeddings  # Or maybe return None or raise error

    print(f"Reducing dimensions using {method.upper()} to {n_components} components...")
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        # t-SNE can be slow for large datasets
        perplexity = min(
            30.0, max(5.0, embeddings.shape[0] / 5.0)
        )  # Adjust perplexity based on data size
        print(f"Using t-SNE with perplexity={perplexity}")
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=perplexity,
            n_iter=300,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(embeddings)
    print("Dimensionality reduction complete.")
    return reduced


def create_interactive_map(
    coordinates: np.ndarray,
    reduced_embeddings: np.ndarray,
    output_path: str,
    zoom_start: int = 10,
):
    """
    Create an interactive map visualization with Folium.

    Args:
        coordinates: Array of (latitude, longitude) coordinates
        reduced_embeddings: Reduced embeddings (2D or 3D)
        output_path: Path to save the HTML map
        zoom_start: Initial zoom level
    """
    if coordinates.shape[0] == 0:
        print("No coordinates to plot. Skipping map generation.")
        return

    # Calculate center of the map
    center_lat = np.mean(coordinates[:, 0])
    center_lon = np.mean(coordinates[:, 1])

    print(f"Creating interactive map centered at ({center_lat:.5f}, {center_lon:.5f})")
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Normalize embedding values for coloring
    num_components = reduced_embeddings.shape[1]
    colors = []

    if num_components >= 3:
        print("Using RGB coloring based on first 3 components.")
        # Use RGB coloring from the first 3 components
        norm_r = Normalize(
            vmin=np.min(reduced_embeddings[:, 0]), vmax=np.max(reduced_embeddings[:, 0])
        )(reduced_embeddings[:, 0])
        norm_g = Normalize(
            vmin=np.min(reduced_embeddings[:, 1]), vmax=np.max(reduced_embeddings[:, 1])
        )(reduced_embeddings[:, 1])
        norm_b = Normalize(
            vmin=np.min(reduced_embeddings[:, 2]), vmax=np.max(reduced_embeddings[:, 2])
        )(reduced_embeddings[:, 2])

        colors = []
        for r, g, b in zip(norm_r, norm_g, norm_b):
            # Convert to hex color, handling potential NaN/Inf
            r_int = int(np.nan_to_num(r) * 255)
            g_int = int(np.nan_to_num(g) * 255)
            b_int = int(np.nan_to_num(b) * 255)
            color = "#{:02x}{:02x}{:02x}".format(r_int, g_int, b_int)
            colors.append(color)
    elif num_components > 0:
        print(f"Using single component ({num_components}) coloring with Viridis map.")
        # Use a single component for coloring
        norm = Normalize(
            vmin=np.min(reduced_embeddings[:, 0]), vmax=np.max(reduced_embeddings[:, 0])
        )(reduced_embeddings[:, 0])
        cmap_colors = plt.cm.viridis(norm)

        # Convert to hex colors
        colors = [
            "#{:02x}{:02x}{:02x}".format(
                int(np.nan_to_num(r) * 255),
                int(np.nan_to_num(g) * 255),
                int(np.nan_to_num(b) * 255),
            )
            for r, g, b, _ in cmap_colors
        ]
    else:
        # Default color if no components
        print("Warning: Reduced embeddings have 0 components. Using default color.")
        colors = ["#808080"] * len(coordinates)  # Gray

    # Add markers using clustering for performance
    marker_cluster = MarkerCluster().add_to(m)
    print(f"Adding {len(coordinates)} markers to the map...")

    for i, (lat, lon) in enumerate(coordinates):
        # Create popup with embedding info
        popup_text = f"Lat: {lat:.5f}, Lon: {lon:.5f}<br>"
        for j in range(num_components):
            popup_text += f"Comp {j + 1}: {reduced_embeddings[i, j]:.4f}<br>"

        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=popup_text,
            color=colors[i],
            fill=True,
            fill_color=colors[i],
            fill_opacity=0.8,
        ).add_to(marker_cluster)

    # Save map to HTML
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")


def create_embedding_json(
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    reduced_embeddings: Optional[np.ndarray],
    output_path: str,
):
    """
    Create a GeoJSON file of the embeddings for visualization in GIS software.

    Args:
        embeddings: Original embeddings
        coordinates: Latitude/longitude coordinates
        reduced_embeddings: Reduced (2D or 3D) embeddings, if available
        output_path: Path to save the GeoJSON file
    """
    if coordinates.shape[0] == 0:
        print("No coordinates to save. Skipping GeoJSON generation.")
        return

    # Create feature collection
    features = []
    print(f"Creating GeoJSON output at {output_path}...")

    has_reduced = (
        reduced_embeddings is not None
        and reduced_embeddings.shape[0] == embeddings.shape[0]
    )
    if not has_reduced:
        print(
            "Warning: Reduced embeddings not available or dimension mismatch. GeoJSON will only contain original embeddings."
        )

    for i, (lat, lon) in enumerate(coordinates):
        # Create feature properties
        properties = {
            "lat": float(lat),
            "lon": float(lon),
        }

        # Add reduced embeddings if available and valid
        if has_reduced:
            for j in range(reduced_embeddings.shape[1]):
                properties[f"component_{j + 1}"] = float(reduced_embeddings[i, j])

        # Add original embeddings (first few dimensions)
        max_dim = min(5, embeddings.shape[1])  # Save only first 5 dims of original
        for j in range(max_dim):
            properties[f"embedding_{j + 1}"] = float(embeddings[i, j])

        # Create feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],  # GeoJSON standard: lon, lat
            },
            "properties": properties,
        }

        features.append(feature)

    # Create GeoJSON structure
    geojson = {"type": "FeatureCollection", "features": features}

    # Save to file
    try:
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"Successfully created GeoJSON file with {len(features)} features.")
    except Exception as e:
        print(f"Error writing GeoJSON file: {e}")


def main(args):
    # Set device
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif not args.no_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Prepared data directory
    prepared_data_dir = Path(args.prepared_data_dir)
    metadata_path = prepared_data_dir / "metadata.json"
    tile_dir = prepared_data_dir / "tiles"

    if not metadata_path.is_file():
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    if not tile_dir.is_dir():
        print(f"Error: Tile directory not found: {tile_dir}")
        return

    # Load metadata from prepared data
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "r") as f:
        data_metadata = json.load(f)

    zoom = data_metadata["zoom"]
    patch_size_from_data = data_metadata["patch_size"]
    sample_coordinates = data_metadata["sample_coordinates"]  # List of [lat, lon]
    region_bounds = data_metadata.get("region_bounds")  # Optional in metadata

    print(f"Data details - Zoom: {zoom}, Patch Size: {patch_size_from_data}")
    if region_bounds:
        print(f"Region Bounds: {region_bounds}")
    print(f"Loaded {len(sample_coordinates)} coordinates from metadata.")

    # Create output directory for visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    model, model_metadata = load_model(args.model_path, device)

    # Verify model patch size matches data patch size
    model_patch_size = model_metadata["patch_size"]
    if model_patch_size != patch_size_from_data:
        print(
            f"Warning: Model patch size ({model_patch_size}) differs from data preparation patch size ({patch_size_from_data}). Using model patch size for tile handler."
        )
        # Decide which patch size to trust - model's is usually safer for inference
        patch_size_to_use = model_patch_size
    else:
        patch_size_to_use = patch_size_from_data

    # Create OSM tile handler using the local tile directory
    tile_handler = OSMTileHandler(
        tile_dir=str(tile_dir), zoom=zoom, patch_size=patch_size_to_use
    )

    # Select coordinates to process
    coordinates_to_process = sample_coordinates
    if args.max_samples and len(coordinates_to_process) > args.max_samples:
        print(
            f"Sampling {args.max_samples} coordinates from the available {len(coordinates_to_process)}."
        )
        coordinates_to_process = random.sample(coordinates_to_process, args.max_samples)
    else:
        print(
            f"Processing all {len(coordinates_to_process)} coordinates from metadata."
        )

    # Generate embeddings
    print("\n--- Generating Embeddings ---")
    embeddings, processed_coordinates = generate_embeddings(
        model=model,
        tile_handler=tile_handler,
        coordinates=coordinates_to_process,
        device=device,
        batch_size=args.batch_size,
    )

    if len(embeddings) == 0:
        print("No embeddings were generated. Exiting visualization.")
        return

    # Reduce dimensions for visualization
    print("\n--- Reducing Dimensions ---")
    reduced_embeddings = reduce_dimensions(
        embeddings, method=args.reduction_method, n_components=args.n_components
    )

    # Create interactive map
    print("\n--- Creating Interactive Map ---")
    create_interactive_map(
        processed_coordinates,  # Use only the coordinates for which embeddings were generated
        reduced_embeddings,
        output_path=str(output_dir / "embeddings_map.html"),
        zoom_start=args.zoom_start,
    )

    # Create GeoJSON for GIS visualization
    print("\n--- Creating GeoJSON Output ---")
    create_embedding_json(
        embeddings,
        processed_coordinates,
        reduced_embeddings,
        output_path=str(output_dir / "embeddings.geojson"),
    )

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Loc2Vec embeddings using pre-prepared OpenStreetMap data"
    )

    # Input paths
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model file (.pt)"
    )
    parser.add_argument(
        "--prepared_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing prepared OSM data (tiles/ and metadata.json)",
    )

    # Processing parameters
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of sample coordinates from metadata to process (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generating embeddings (currently processes one-by-one, future enhancement)",
    )

    # Visualization parameters
    parser.add_argument(
        "--reduction_method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="Number of components for dimensionality reduction (e.g., 3 for RGB coloring)",
    )
    parser.add_argument(
        "--zoom_start",
        type=int,
        default=12,
        help="Initial zoom level for interactive Folium map",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Output directory for saving map.html and embeddings.geojson",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA for model inference"
    )

    args = parser.parse_args()
    main(args)
