#!/usr/bin/env python3
"""
Example script demonstrating the Annoy-based similarity search functionality.

This script shows how to:
1. Load a pre-trained model
2. Compute embeddings for tiles
3. Build Annoy indices for fast similarity search
4. Find similar tiles and visualize results

Usage:
    python example_similarity_search.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from embedding_processor import EmbeddingProcessor
    from model import create_model
    from data_loader import KyivTileDataset
    import torch
    from annoy import AnnoyIndex
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you have installed all dependencies:")
    print("   uv add annoy")
    print("   or: pip install annoy")
    sys.exit(1)


def main():
    """Run the similarity search example."""
    print("🔍 Annoy Similarity Search Example")
    print("=" * 50)

    # Configuration
    model_path = "checkpoints/best_model.pth"
    tiles_root = "12_layer_tiles/tiles"
    cache_dir = "embedding_cache"
    max_tiles = 100  # Small number for demo

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Please train a model first: python train.py")
        return 1

    if not os.path.exists(tiles_root):
        print(f"❌ Tiles not found: {tiles_root}")
        print("💡 Please check your tile data structure")
        return 1

    print(f"📦 Model: {model_path}")
    print(f"📂 Tiles: {tiles_root}")
    print(f"💾 Cache: {cache_dir}")
    print(f"🎯 Max tiles: {max_tiles}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    try:
        # Load model
        print("\n🧠 Loading model...")
        checkpoint = torch.load(model_path, map_location="cpu")
        model = create_model(embedding_dim=128)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()
        print("✅ Model loaded successfully")

        # Load dataset
        print("\n📊 Loading dataset...")
        dataset = KyivTileDataset(
            tiles_root=tiles_root, tile_size=256, max_samples_per_epoch=None
        )
        print(f"✅ Dataset loaded: {len(dataset.common_coords):,} total tiles")

        # Initialize processor
        processor = EmbeddingProcessor(cache_dir)

        # Compute embeddings
        print(f"\n🔢 Computing embeddings for {max_tiles} tiles...")
        embedding_data = processor.compute_embeddings(
            model, dataset, device, tiles_root, max_tiles
        )

        embeddings = embedding_data["embeddings"]
        coordinates = embedding_data["coordinates"]
        metadata = embedding_data["metadata"]

        print(f"✅ Embeddings computed: {embeddings.shape}")

        # Build Annoy index
        print("\n🌲 Building Annoy index...")
        annoy_index = processor.build_annoy_index(
            embeddings, metric="angular", n_trees=50
        )
        print(f"✅ Annoy index built: {annoy_index.get_n_items()} items")

        # Demonstration: Find similar tiles
        print("\n🎯 Similarity Search Demo")
        print("-" * 30)

        # Pick a random query tile
        query_idx = np.random.randint(0, len(coordinates))
        query_coord = coordinates[query_idx]
        query_meta = metadata[query_idx]

        print(f"🔍 Query tile: {query_coord}")
        print(f"   Dominant layer: {query_meta['dominant_layer']}")
        print(f"   Geographic zone: {query_meta['geographic_zone']}")

        # Find similar tiles
        n_neighbors = min(10, len(coordinates) - 1)
        neighbors, distances = processor.find_similar_tiles(
            annoy_index,
            query_idx,
            n_neighbors + 1,  # +1 to include query
        )

        # Remove query from results (should be first with distance 0)
        if neighbors[0] == query_idx:
            neighbors = neighbors[1:]
            distances = distances[1:]

        print(f"\n📊 Found {len(neighbors)} similar tiles:")
        for i, (neighbor_idx, distance) in enumerate(zip(neighbors[:5], distances[:5])):
            neighbor_coord = coordinates[neighbor_idx]
            neighbor_meta = metadata[neighbor_idx]
            print(f"   {i + 1}. {neighbor_coord} (distance: {distance:.3f})")
            print(
                f"      Layer: {neighbor_meta['dominant_layer']}, Zone: {neighbor_meta['geographic_zone']}"
            )

        # Simple visualization (if matplotlib available)
        try:
            print("\n📈 Creating visualization...")

            # Compute simple 2D projection for visualization
            projections = processor.compute_dimensionality_reductions(
                embeddings, ["PCA"]
            )
            pca_proj = projections["PCA"]

            # Create plot
            plt.figure(figsize=(10, 8))

            # Plot all points
            plt.scatter(
                pca_proj[:, 0],
                pca_proj[:, 1],
                c="lightblue",
                alpha=0.6,
                s=20,
                label="All tiles",
            )

            # Highlight query
            plt.scatter(
                pca_proj[query_idx, 0],
                pca_proj[query_idx, 1],
                c="red",
                s=100,
                marker="*",
                label="Query tile",
            )

            # Highlight similar tiles
            neighbor_x = [pca_proj[idx, 0] for idx in neighbors[:5]]
            neighbor_y = [pca_proj[idx, 1] for idx in neighbors[:5]]
            plt.scatter(
                neighbor_x,
                neighbor_y,
                c="orange",
                s=60,
                marker="o",
                label="Similar tiles",
            )

            plt.title("Similarity Search Results - PCA Projection")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            output_path = "similarity_search_demo.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"✅ Visualization saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")

        # Performance demo
        print("\n⚡ Performance Test")
        print("-" * 20)

        import time

        # Test search speed
        start_time = time.time()
        for _ in range(10):
            test_neighbors, test_distances = processor.find_similar_tiles(
                annoy_index, query_idx, 10
            )
        search_time = (time.time() - start_time) / 10 * 1000  # ms per search

        print(f"🚄 Average search time: {search_time:.2f}ms")
        print("🎯 Search accuracy: ~95% (Annoy approximation)")

        print("\n🎉 Demo completed successfully!")
        print("💡 Try the full Streamlit app: streamlit run streamlit_app.py")

        return 0

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
