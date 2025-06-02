#!/usr/bin/env python3
"""
Pre-compute embeddings and build Annoy indices for fast similarity search.

This script computes embeddings for all tiles, builds Annoy indices for fast
similarity search, and computes dimensionality reduction projections. The results
are cached to disk for fast loading in the Streamlit app.

Usage:
    python precompute_embeddings.py                    # Basic usage
    python precompute_embeddings.py --max_tiles 1000   # Limit tiles for testing
    python precompute_embeddings.py --annoy_trees 500  # High quality index
    python precompute_embeddings.py --skip_embeddings  # Only rebuild indices
"""

import argparse
import time
from embedding_processor import EmbeddingProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings and build Annoy indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Basic usage with defaults
  %(prog)s --max_tiles 1000             # Process only 1000 tiles for testing
  %(prog)s --annoy_trees 500            # High quality index (slower)
  %(prog)s --skip_embeddings            # Only rebuild indices, skip embeddings
  %(prog)s --force_recompute            # Force recomputation of everything
        """,
    )

    # Data settings
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--tiles_root",
        type=str,
        default="12_layer_tiles/tiles",
        help="Root directory of tiles (default: 12_layer_tiles/tiles)",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=None,
        help="Maximum number of tiles to process (default: all)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="embedding_cache",
        help="Directory to store cached results (default: embedding_cache)",
    )

    # Computation settings
    parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip embedding computation, only build indices from existing embeddings",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation even if cache exists",
    )
    parser.add_argument(
        "--dr_algorithms",
        nargs="+",
        default=["PCA", "UMAP"],
        choices=["PCA", "TSNE", "UMAP"],
        help="Dimensionality reduction algorithms to compute (default: PCA UMAP)",
    )

    # Annoy settings
    parser.add_argument(
        "--annoy_trees",
        type=int,
        default=100,
        help="Number of trees for Annoy index (default: 100)",
    )
    parser.add_argument(
        "--annoy_metrics",
        nargs="+",
        default=["angular"],
        choices=["angular", "euclidean", "manhattan", "hamming", "dot"],
        help="Annoy distance metrics to build indices for (default: angular)",
    )

    args = parser.parse_args()

    print("🚀 Pre-computing Embeddings and Annoy Indices")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Tiles root: {args.tiles_root}")
    print(f"Max tiles: {args.max_tiles or 'all'}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Skip embeddings: {args.skip_embeddings}")
    print(f"Force recompute: {args.force_recompute}")
    print(f"DR algorithms: {args.dr_algorithms}")
    print(f"Annoy trees: {args.annoy_trees}")
    print(f"Annoy metrics: {args.annoy_metrics}")
    print()

    start_time = time.time()

    # Import here to avoid issues if dependencies aren't available
    try:
        import torch
        from model import create_model
        from data_loader import KyivTileDataset
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print(
            "Make sure you're in the correct directory and dependencies are installed"
        )
        return

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Metal Performance Shaders")
    else:
        device = torch.device("cpu")
        print("🔧 Using CPU")

    # Initialize processor
    processor = EmbeddingProcessor(args.cache_dir)

    embeddings = None

    if not args.skip_embeddings:
        # Load model
        try:
            print(f"\n📥 Loading model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location="cpu")
            model = create_model(embedding_dim=128)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device).eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return

        # Load dataset
        try:
            print(f"\n📊 Loading dataset from {args.tiles_root}")
            dataset = KyivTileDataset(
                tiles_root=args.tiles_root, tile_size=256, max_samples_per_epoch=None
            )
            print(f"✅ Dataset loaded: {len(dataset.common_coords)} tiles available")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return

        # Compute embeddings
        print("\n🧠 Computing embeddings...")
        try:
            embedding_data = processor.compute_embeddings(
                model,
                dataset,
                device,
                args.tiles_root,
                args.max_tiles,
                args.force_recompute,
            )
            embeddings = embedding_data["embeddings"]
            print(f"✅ Embeddings computed: {embeddings.shape}")
        except Exception as e:
            print(f"❌ Error computing embeddings: {e}")
            return

        # Compute dimensionality reductions
        print(f"\n📐 Computing dimensionality reductions: {args.dr_algorithms}")
        try:
            projections = processor.compute_dimensionality_reductions(
                embeddings, args.dr_algorithms, args.force_recompute
            )
            print(f"✅ Projections computed: {list(projections.keys())}")
        except Exception as e:
            print(f"❌ Error computing projections: {e}")
            # Continue anyway, projections are not critical

    else:
        # Try to load existing embeddings
        print("\n🔍 Looking for existing embeddings...")
        try:
            # We need a dummy model and dataset just to get the cache key right
            # but we won't actually use them
            import os

            if not os.path.exists(args.model_path):
                print(f"❌ Model file not found: {args.model_path}")
                return

            if not os.path.exists(args.tiles_root):
                print(f"❌ Tiles directory not found: {args.tiles_root}")
                return

            # Load minimal data just to get cache key
            cache_key = (
                f"embeddings_{abs(hash(args.tiles_root))}_{args.max_tiles or 'all'}"
            )
            cache_file = processor.cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                import pickle

                with open(cache_file, "rb") as f:
                    embedding_data = pickle.load(f)
                embeddings = embedding_data["embeddings"]
                print(f"✅ Found existing embeddings: {embeddings.shape}")
            else:
                print(f"❌ No existing embeddings found at {cache_file}")
                print("   Run without --skip_embeddings to compute them first")
                return

        except Exception as e:
            print(f"❌ Error loading existing embeddings: {e}")
            return

    # Build Annoy indices for all requested metrics
    print("\n🌲 Building Annoy indices...")
    for metric in args.annoy_metrics:
        print(f"  Building {metric} index with {args.annoy_trees} trees...")
        try:
            annoy_index = processor.build_annoy_index(
                embeddings, metric, args.annoy_trees, args.force_recompute
            )
            print(f"  ✅ {metric} index: {annoy_index.get_n_items()} items")
        except Exception as e:
            print(f"  ❌ Error building {metric} index: {e}")
            continue

    total_time = time.time() - start_time
    print(f"\n🎉 Pre-computation completed in {total_time:.1f} seconds!")
    print(f"📁 Results cached in: {processor.cache_dir}")
    print("\n💡 Now you can run the Streamlit app for fast similarity search:")
    print("   streamlit run streamlit_app.py")
    print("\n🔍 To check what's cached, run:")
    print("   python check_cache.py")


if __name__ == "__main__":
    main()
