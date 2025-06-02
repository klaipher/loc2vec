"""
Embedding processor for pre-computing embeddings and building Annoy indices.
This module handles the heavy computation outside of the main Streamlit app.
"""

import pickle
import numpy as np
import torch
from annoy import AnnoyIndex
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class EmbeddingProcessor:
    """Process and manage embeddings with Annoy index for fast similarity search."""

    def __init__(self, cache_dir: str = "embedding_cache"):
        """Initialize the embedding processor.

        Args:
            cache_dir: Directory to store cached embeddings and indices
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def compute_embeddings(
        self,
        model,
        dataset,
        device,
        tiles_root: str,
        max_tiles: Optional[int] = None,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """Compute or load cached embeddings.

        Args:
            model: The trained model
            dataset: The tile dataset
            device: Torch device
            tiles_root: Root directory of tiles
            max_tiles: Maximum number of tiles to process
            force_recompute: Force recomputation even if cache exists

        Returns:
            Dictionary containing embeddings, coordinates, metadata, and tile images
        """
        # Create cache key based on parameters
        cache_key = f"embeddings_{abs(hash(tiles_root))}_{max_tiles or 'all'}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists() and not force_recompute:
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print("Computing embeddings...")
        coords = dataset.common_coords.copy()

        if max_tiles and max_tiles < len(coords):
            import random

            coords = random.sample(coords, max_tiles)

        # Process tiles in batches
        batch_size = 64
        all_embeddings = []
        all_coordinates = []
        all_metadata = []
        all_tile_images = []

        for i in tqdm(range(0, len(coords), batch_size), desc="Processing tiles"):
            batch_coords = coords[i : i + batch_size]
            batch_data = self._process_tile_batch(batch_coords, dataset, tiles_root)

            if not batch_data:
                continue

            # Generate embeddings
            batch_tiles = [item["tile_stack"] for item in batch_data]
            batch_tensor = torch.from_numpy(np.stack(batch_tiles, axis=0)).to(device)

            with torch.no_grad():
                batch_embeddings = model.encode(batch_tensor)
                embeddings_np = batch_embeddings.cpu().numpy()

            all_embeddings.append(embeddings_np)

            for j, item in enumerate(batch_data):
                all_coordinates.append(item["coord"])
                all_metadata.append(item["metadata"])
                all_tile_images.append(item["tile_image"])

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

        result = {
            "embeddings": embeddings,
            "coordinates": all_coordinates,
            "metadata": all_metadata,
            "tile_images": all_tile_images,
        }

        # Cache the results
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        print(f"Cached embeddings to {cache_file}")
        return result

    def build_annoy_index(
        self,
        embeddings: np.ndarray,
        metric: str = "angular",
        n_trees: int = 100,
        force_rebuild: bool = False,
    ) -> AnnoyIndex:
        """Build or load Annoy index for fast similarity search.

        Args:
            embeddings: The embedding vectors
            metric: Distance metric ('angular', 'euclidean', 'manhattan', 'hamming', 'dot')
            n_trees: Number of trees for the index
            force_rebuild: Force rebuilding even if index exists

        Returns:
            Annoy index
        """
        # Create cache key
        embedding_hash = abs(hash(embeddings.tobytes()))
        index_file = (
            self.cache_dir / f"annoy_index_{embedding_hash}_{metric}_{n_trees}.ann"
        )

        embedding_dim = embeddings.shape[1]
        annoy_index = AnnoyIndex(embedding_dim, metric)

        if index_file.exists() and not force_rebuild:
            print(f"Loading cached Annoy index from {index_file}")
            annoy_index.load(str(index_file))
            return annoy_index

        print(f"Building Annoy index with {n_trees} trees...")

        # Add all embeddings to the index
        for i, embedding in enumerate(
            tqdm(embeddings, desc="Adding embeddings to index")
        ):
            annoy_index.add_item(i, embedding.tolist())

        # Build the index
        annoy_index.build(n_trees)

        # Save the index
        annoy_index.save(str(index_file))
        print(f"Saved Annoy index to {index_file}")

        return annoy_index

    def compute_dimensionality_reductions(
        self,
        embeddings: np.ndarray,
        algorithms: List[str] = ["PCA", "TSNE", "UMAP"],
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute dimensionality reduction projections.

        Args:
            embeddings: The embedding vectors
            algorithms: List of algorithms to compute
            force_recompute: Force recomputation even if cache exists

        Returns:
            Dictionary mapping algorithm names to 2D projections
        """
        # Create cache key
        embedding_hash = abs(hash(embeddings.tobytes()))
        cache_file = (
            self.cache_dir / f"projections_{embedding_hash}_{'_'.join(algorithms)}.pkl"
        )

        if cache_file.exists() and not force_recompute:
            print(f"Loading cached projections from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print("Computing dimensionality reductions...")
        projections = {}

        for algorithm in algorithms:
            print(f"Computing {algorithm}...")
            start_time = time.time()

            if algorithm == "PCA":
                reducer = PCA(n_components=2, random_state=42)
                projection = reducer.fit_transform(embeddings)
            elif algorithm == "TSNE":
                reducer = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embeddings) - 1),
                )
                projection = reducer.fit_transform(embeddings)
            elif algorithm == "UMAP":
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(15, len(embeddings) - 1),
                )
                projection = reducer.fit_transform(embeddings)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            projections[algorithm] = projection
            print(f"{algorithm} completed in {time.time() - start_time:.2f}s")

        # Cache the results
        with open(cache_file, "wb") as f:
            pickle.dump(projections, f)

        print(f"Cached projections to {cache_file}")
        return projections

    def find_similar_tiles(
        self,
        annoy_index: AnnoyIndex,
        query_idx: int,
        n_neighbors: int = 10,
        search_k: int = -1,
    ) -> Tuple[List[int], List[float]]:
        """Find similar tiles using Annoy index.

        Args:
            annoy_index: The Annoy index
            query_idx: Index of the query tile
            n_neighbors: Number of neighbors to return
            search_k: Search parameter for Annoy (-1 for default)

        Returns:
            Tuple of (neighbor_indices, distances)
        """
        neighbors, distances = annoy_index.get_nns_by_item(
            query_idx, n_neighbors, search_k=search_k, include_distances=True
        )
        return neighbors, distances

    def find_similar_by_vector(
        self,
        annoy_index: AnnoyIndex,
        query_vector: np.ndarray,
        n_neighbors: int = 10,
        search_k: int = -1,
    ) -> Tuple[List[int], List[float]]:
        """Find similar tiles using a query vector.

        Args:
            annoy_index: The Annoy index
            query_vector: The query embedding vector
            n_neighbors: Number of neighbors to return
            search_k: Search parameter for Annoy (-1 for default)

        Returns:
            Tuple of (neighbor_indices, distances)
        """
        neighbors, distances = annoy_index.get_nns_by_vector(
            query_vector.tolist(),
            n_neighbors,
            search_k=search_k,
            include_distances=True,
        )
        return neighbors, distances

    def _process_tile_batch(
        self, coords_batch: List[Tuple], dataset, tiles_root: str
    ) -> List[Dict]:
        """Process a batch of tiles for embeddings and metadata."""
        batch_data = []

        for coord in coords_batch:
            try:
                # Load tile stack
                tile_stack = dataset._load_tile_stack(coord)

                # Analyze content (simplified version)
                meta = self._analyze_tile_content(coord, dataset.layers, tiles_root)

                # Create tile image (simplified)
                tile_image = self._create_tile_image_base64(
                    coord, dataset.layers, tiles_root
                )

                batch_data.append(
                    {
                        "coord": coord,
                        "tile_stack": tile_stack,
                        "metadata": meta,
                        "tile_image": tile_image,
                    }
                )
            except Exception as e:
                print(f"Error processing tile {coord}: {e}")
                continue

        return batch_data

    def _analyze_tile_content(
        self, coord: Tuple, layers: List[str], tiles_root: str
    ) -> Dict:
        """Simplified tile content analysis."""
        zoom, x, y = coord

        # Try to analyze layer content like the main app does
        layer_content = {}
        total_content = 0

        try:
            from PIL import Image
            import os

            for layer in layers:
                tile_path = os.path.join(
                    tiles_root, layer, str(zoom), str(x), f"{y}.png"
                )

                if os.path.exists(tile_path):
                    try:
                        img = Image.open(tile_path).convert("RGB")
                        img_array = np.array(img)
                        # Calculate content density
                        content_density = (
                            np.sum(np.mean(img_array, axis=2) > 30)
                            / img_array[:, :, 0].size
                        )
                        layer_content[layer] = content_density
                        total_content += content_density
                    except:
                        layer_content[layer] = 0.0
                else:
                    layer_content[layer] = 0.0
        except:
            # Fallback: assume some content for all layers
            for layer in layers:
                layer_content[layer] = 0.1
            total_content = len(layers) * 0.1

        # Find dominant layer
        dominant_layer = (
            max(layer_content, key=layer_content.get) if layer_content else "unknown"
        )

        # Basic metadata
        return {
            "zoom": zoom,
            "x": x,
            "y": y,
            "coordinate_str": f"{zoom}/{x}/{y}",
            "dominant_layer": dominant_layer,
            "total_content": total_content,
            "layer_content": layer_content,
            "geographic_zone": self._get_geographic_zone(x, y),
        }

    def _get_geographic_zone(self, x: int, y: int) -> str:
        """Determine geographic zone based on coordinates."""
        # Match the logic from the main streamlit app
        if x < 76650:
            zone = "West"
        elif x > 76700:
            zone = "East"
        else:
            zone = "Central"

        if y < 44180:
            zone += "_North"
        elif y > 44220:
            zone += "_South"
        else:
            zone += "_Center"

        return zone

    def _create_tile_image_base64(
        self, coord: Tuple, layers: List[str], tiles_root: str
    ) -> str:
        """Create a simple base64 encoded tile image."""
        # Try to create a simple composite image
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            import os

            zoom, x, y = coord

            # Try to load the first available layer as a preview
            for layer in ["buildings", "roads", "natural-features", "water"]:
                if layer in layers:
                    tile_path = os.path.join(
                        tiles_root, layer, str(zoom), str(x), f"{y}.png"
                    )
                    if os.path.exists(tile_path):
                        try:
                            img = Image.open(tile_path).convert("RGB")
                            img = img.resize((32, 32), Image.Resampling.LANCZOS)

                            # Convert to base64
                            buffer = BytesIO()
                            img.save(buffer, format="PNG")
                            img_str = base64.b64encode(buffer.getvalue()).decode()
                            return f"data:image/png;base64,{img_str}"
                        except:
                            continue

            # Fallback: create a simple placeholder
            img = Image.new("RGB", (32, 32), color="lightgray")
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        except:
            # Ultimate fallback: 1x1 transparent image
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def main():
    """Command line interface for pre-computing embeddings."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings and build Annoy index"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tiles_root",
        type=str,
        default="12_layer_tiles/tiles",
        help="Root directory of tiles",
    )
    parser.add_argument(
        "--max_tiles", type=int, default=None, help="Maximum number of tiles to process"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="embedding_cache",
        help="Directory to store cached results",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation even if cache exists",
    )
    parser.add_argument(
        "--annoy_trees", type=int, default=100, help="Number of trees for Annoy index"
    )
    parser.add_argument(
        "--annoy_metric",
        type=str,
        default="angular",
        choices=["angular", "euclidean", "manhattan", "hamming", "dot"],
        help="Annoy distance metric",
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from model import create_model
    from data_loader import KyivTileDataset

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = create_model(embedding_dim=128)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    # Load dataset
    dataset = KyivTileDataset(
        tiles_root=args.tiles_root, tile_size=256, max_samples_per_epoch=None
    )

    # Initialize processor
    processor = EmbeddingProcessor(args.cache_dir)

    # Compute embeddings
    embedding_data = processor.compute_embeddings(
        model, dataset, device, args.tiles_root, args.max_tiles, args.force_recompute
    )

    embeddings = embedding_data["embeddings"]
    print(f"Computed embeddings shape: {embeddings.shape}")

    # Build Annoy index
    annoy_index = processor.build_annoy_index(
        embeddings, args.annoy_metric, args.annoy_trees, args.force_recompute
    )

    # Compute dimensionality reductions
    projections = processor.compute_dimensionality_reductions(
        embeddings, ["PCA", "TSNE", "UMAP"], args.force_recompute
    )

    print("Pre-computation complete!")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Projections: {list(projections.keys())}")
    print(f"Annoy index: {annoy_index.get_n_items()} items")


if __name__ == "__main__":
    main()
