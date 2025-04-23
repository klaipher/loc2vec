import argparse


def main():
    """
    Main entry point for Loc2Vec implementation using OpenStreetMap data.

    Provides commands to prepare OSM data, train the model, and visualize embeddings.
    """
    parser = argparse.ArgumentParser(
        description="Loc2Vec: Learning Deep Representations of Location from OSM Data",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    # --- OpenStreetMap Commands ---
    prep_osm_parser = subparsers.add_parser(
        "prepare-osm", help="Download OSM tiles and prepare data for training"
    )
    prep_osm_parser.add_argument(
        "--server_ip",
        type=str,
        required=True,
        help="IP address of the OpenStreetMap tile server",
    )
    prep_osm_parser.add_argument("--server_port", type=int, default=80)
    prep_osm_parser.add_argument(
        "--server_path",
        type=str,
        default="/hot/{z}/{x}/{y}.png",
        help="Path template for OSM tiles on the server",
    )  # Updated default
    prep_osm_parser.add_argument("--zoom", type=int, default=17)
    prep_osm_parser.add_argument(
        "--min_lat", type=float, default=49.8, help="Minimum latitude for region"
    )
    prep_osm_parser.add_argument(
        "--min_lon", type=float, default=29.2, help="Minimum longitude for region"
    )
    prep_osm_parser.add_argument(
        "--max_lat", type=float, default=51.5, help="Maximum latitude for region"
    )
    prep_osm_parser.add_argument(
        "--max_lon", type=float, default=32.2, help="Maximum longitude for region"
    )
    prep_osm_parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Patch size used for training (saved in metadata)",
    )
    prep_osm_parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of sample anchor coordinates to generate",
    )
    prep_osm_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prepared data (tiles/ and metadata.json)",
    )

    train_osm_parser = subparsers.add_parser(
        "train-osm", help="Train Loc2Vec using pre-prepared OSM data"
    )
    train_osm_parser.add_argument(
        "--prepared_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing prepared OSM data",
    )
    train_osm_parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        help="Path to prepared validation data directory (optional)",
    )
    train_osm_parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_osm",
        help="Output directory for models and logs",
    )
    train_osm_parser.add_argument("--epochs", type=int, default=50)
    train_osm_parser.add_argument("--batch_size", type=int, default=32)
    train_osm_parser.add_argument("--embedding_dim", type=int, default=64)
    train_osm_parser.add_argument("--lr", type=float, default=0.001)
    train_osm_parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Patch size (usually loaded from metadata)",
    )
    train_osm_parser.add_argument("--input_channels", type=int, default=3)
    train_osm_parser.add_argument(
        "--max_distance_positive",
        type=float,
        default=0.001,
        help="Max distance (degrees) for positive samples",
    )
    train_osm_parser.add_argument(
        "--min_distance_negative",
        type=float,
        default=0.01,
        help="Min distance (degrees) for negative samples",
    )
    train_osm_parser.add_argument("--margin", type=float, default=0.3)
    train_osm_parser.add_argument("--weight_decay", type=float, default=1e-5)
    train_osm_parser.add_argument("--num_workers", type=int, default=4)
    train_osm_parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA/MPS"
    )

    viz_osm_parser = subparsers.add_parser(
        "visualize-osm", help="Visualize Loc2Vec embeddings using pre-prepared OSM data"
    )
    viz_osm_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model file (.pt)"
    )
    viz_osm_parser.add_argument(
        "--prepared_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing prepared OSM data",
    )
    viz_osm_parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations_osm",
        help="Output directory for visualizations (map.html, embeddings.geojson)",
    )
    viz_osm_parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max number of points to visualize (from metadata)",
    )
    viz_osm_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation (currently 1)",
    )
    viz_osm_parser.add_argument(
        "--reduction_method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Dimensionality reduction method",
    )
    viz_osm_parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="Number of components for reduction (e.g., 3 for RGB color)",
    )
    viz_osm_parser.add_argument(
        "--zoom_start", type=int, default=12, help="Initial zoom for Folium map"
    )
    viz_osm_parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA/MPS for inference"
    )

    # Parse args
    args = parser.parse_args()

    # Execute command
    if args.command == "prepare-osm":
        print("Starting OSM data preparation...")
        try:
            from prepare_osm_data import main as prep_osm_main

            prep_osm_main(args)
        except ImportError:
            print(
                "Error: Could not import the OSM data preparation script 'prepare_osm_data.py'"
            )
        except Exception as e:
            print(f"Error running OSM data preparation: {e}")

    elif args.command == "train-osm":
        print("Starting Loc2Vec training with prepared OSM data...")
        try:
            from train_osm import main as train_osm_main

            train_osm_main(args)
        except ImportError:
            print("Error: Could not import the OSM training script 'train_osm.py'")
        except Exception as e:
            print(f"Error running OSM training: {e}")

    elif args.command == "visualize-osm":
        print("Visualizing Loc2Vec embeddings with prepared OSM data...")
        try:
            from visualize_osm import main as viz_osm_main

            viz_osm_main(args)
        except ImportError:
            print(
                "Error: Could not import the OSM visualization script 'visualize_osm.py'"
            )
        except Exception as e:
            print(f"Error running OSM visualization: {e}")

    else:
        # Should not happen due to `required=True` in subparsers
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
