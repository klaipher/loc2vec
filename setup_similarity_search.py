#!/usr/bin/env python3
"""
Interactive setup script for Annoy-based similarity search.

This script guides you through setting up fast similarity search for the best
Streamlit experience. It will pre-compute embeddings and build Annoy indices.

Usage:
    python setup_similarity_search.py
"""

import os
import sys
import time


def main():
    print("🚀 Similarity Search Setup Wizard")
    print("=" * 50)
    print("This script will help you set up blazing-fast similarity search")
    print("for the best Streamlit experience!")
    print()

    # Check dependencies
    try:
        import torch
        import numpy as np
        from annoy import AnnoyIndex

        print("✅ All dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please install missing packages:")
        print("   uv add annoy")
        print("   or: pip install annoy torch numpy")
        return 1

    # Check files
    model_path = "checkpoints/best_model.pth"
    tiles_root = "12_layer_tiles/tiles"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Please train a model first:")
        print("   python train.py --epochs 20")
        return 1

    if not os.path.exists(tiles_root):
        print(f"❌ Tiles not found: {tiles_root}")
        print("💡 Please check your tile data structure")
        return 1

    print("✅ Model and tiles found")

    # Interactive configuration
    print("\n🎛️ Configuration")
    print("-" * 20)

    # Ask about number of tiles
    print("How many tiles should we process?")
    print("  1. All tiles (best quality, slower)")
    print("  2. 10,000 tiles (good balance)")
    print("  3. 5,000 tiles (faster)")
    print("  4. 1,000 tiles (quickest for testing)")

    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            max_tiles = None
            break
        elif choice == "2":
            max_tiles = 10000
            break
        elif choice == "3":
            max_tiles = 5000
            break
        elif choice == "4":
            max_tiles = 1000
            break
        else:
            print("Please enter 1, 2, 3, or 4")

    # Ask about quality vs speed
    print("\nWhat's more important to you?")
    print("  1. Speed (100 trees, ~95% accuracy)")
    print("  2. Quality (200 trees, ~99% accuracy)")
    print("  3. Maximum quality (500 trees, ~99.5% accuracy)")

    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice == "1":
            n_trees = 100
            break
        elif choice == "2":
            n_trees = 200
            break
        elif choice == "3":
            n_trees = 500
            break
        else:
            print("Please enter 1, 2, or 3")

    # Ask about algorithms
    print("\nWhich visualization algorithms should we pre-compute?")
    print("  1. Essential (PCA + UMAP)")
    print("  2. All (PCA + UMAP + t-SNE)")

    while True:
        choice = input("Enter choice (1-2): ").strip()
        if choice == "1":
            algorithms = ["PCA", "UMAP"]
            break
        elif choice == "2":
            algorithms = ["PCA", "UMAP", "TSNE"]
            break
        else:
            print("Please enter 1 or 2")

    # Summary
    print("\n📋 Configuration Summary")
    print("-" * 25)
    print(f"Max tiles: {max_tiles or 'All'}")
    print(f"Annoy trees: {n_trees}")
    print(f"Algorithms: {', '.join(algorithms)}")

    # Estimate time
    base_time = (
        2
        if max_tiles == 1000
        else 5
        if max_tiles == 5000
        else 10
        if max_tiles == 10000
        else 20
    )
    tree_time = n_trees // 50
    algo_time = len(algorithms) * 2
    total_time = base_time + tree_time + algo_time

    print(f"Estimated time: ~{total_time} minutes")

    # Confirm
    confirm = input("\nProceed with setup? (y/n): ").strip().lower()
    if confirm not in ["y", "yes"]:
        print("Setup cancelled")
        return 0

    # Run pre-computation
    print("\n🚀 Starting pre-computation...")
    print("This may take a while. Feel free to grab a coffee! ☕")

    # Build command
    cmd_parts = [
        "python",
        "precompute_embeddings.py",
        "--annoy_trees",
        str(n_trees),
        "--dr_algorithms",
    ]
    cmd_parts.extend(algorithms)

    if max_tiles:
        cmd_parts.extend(["--max_tiles", str(max_tiles)])

    cmd = " ".join(cmd_parts)
    print(f"Running: {cmd}")

    # Execute
    start_time = time.time()
    exit_code = os.system(cmd)
    elapsed = time.time() - start_time

    if exit_code == 0:
        print("\n🎉 Setup completed successfully!")
        print(f"⏱️ Total time: {elapsed / 60:.1f} minutes")
        print("\n🚀 You're all set! Now you can:")
        print("   1. Launch Streamlit: streamlit run streamlit_app.py")
        print("   2. Select 'Similarity Search (Annoy)' mode")
        print("   3. Enjoy blazing-fast similarity search!")
        print("\n💡 Pro tip: Try different distance metrics (angular, euclidean)")
        print("   for different types of similarity!")

    else:
        print(f"\n❌ Setup failed with exit code {exit_code}")
        print("💡 Try running the command manually to see detailed errors:")
        print(f"   {cmd}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user")
        sys.exit(1)
