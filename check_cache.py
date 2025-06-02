#!/usr/bin/env python3
"""
Quick script to check what pre-computed data is available.

Usage:
    python check_cache.py
"""

from embedding_processor import EmbeddingProcessor


def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def main():
    print("🔍 Checking Pre-computed Cache")
    print("=" * 40)

    # Initialize processor to get cache directory
    processor = EmbeddingProcessor()
    cache_dir = processor.cache_dir

    print(f"📁 Cache directory: {cache_dir}")

    if not cache_dir.exists():
        print("❌ Cache directory does not exist")
        print("💡 Run: python precompute_embeddings.py")
        return

    # Check for different types of files
    embedding_files = list(cache_dir.glob("embeddings_*.pkl"))
    projection_files = list(cache_dir.glob("projections_*.pkl"))
    annoy_files = list(cache_dir.glob("annoy_index_*.ann"))

    total_size = 0

    print(f"\n🧠 Embedding Files ({len(embedding_files)}):")
    if embedding_files:
        for file in embedding_files:
            size = file.stat().st_size
            total_size += size
            print(f"  ✅ {file.name} ({format_size(size)})")
    else:
        print("  ❌ No embedding files found")

    print(f"\n📐 Projection Files ({len(projection_files)}):")
    if projection_files:
        for file in projection_files:
            size = file.stat().st_size
            total_size += size
            print(f"  ✅ {file.name} ({format_size(size)})")
    else:
        print("  ❌ No projection files found")

    print(f"\n🌲 Annoy Index Files ({len(annoy_files)}):")
    if annoy_files:
        for file in annoy_files:
            size = file.stat().st_size
            total_size += size
            # Parse filename to extract info
            parts = file.stem.split("_")
            if len(parts) >= 4:
                metric = parts[2]
                trees = parts[3]
                print(f"  ✅ {metric} metric, {trees} trees ({format_size(size)})")
            else:
                print(f"  ✅ {file.name} ({format_size(size)})")
    else:
        print("  ❌ No Annoy index files found")

    print("\n📊 Summary:")
    print(
        f"  Total files: {len(embedding_files) + len(projection_files) + len(annoy_files)}"
    )
    print(f"  Total size: {format_size(total_size)}")

    if embedding_files and annoy_files:
        print("\n🚀 Status: Ready for fast similarity search!")
        print("   Launch: streamlit run streamlit_app.py")
    elif embedding_files:
        print("\n⚠️  Status: Embeddings ready, but no Annoy indices")
        print("   Run: python precompute_embeddings.py --skip_embeddings")
    else:
        print("\n❌ Status: No pre-computed data found")
        print("   Run: python precompute_embeddings.py")

    print("\n💡 Tips:")
    print("  • For faster startup: python precompute_embeddings.py --max_tiles 1000")
    print("  • For better quality: python precompute_embeddings.py --annoy_trees 500")
    print(
        "  • For all algorithms: python precompute_embeddings.py --dr_algorithms PCA UMAP TSNE"
    )


if __name__ == "__main__":
    main()
