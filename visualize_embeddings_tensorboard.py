import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

# Clustering and dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.cluster import DBSCAN
import hdbscan

# Metrics
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# Visualization
import matplotlib.pyplot as plt

from loc2vec.model import Loc2VecModel
from loc2vec.dataset import TilesDataset
from loc2vec.embeddings import log_embeddings_to_tensorboard


def extract_embeddings_and_metadata(model, dataloader, device, max_samples=2000):
    """Extract embeddings and metadata from the model"""
    model.eval()

    embeddings_list = []
    metadata = {"x_coords": [], "y_coords": [], "zoom_levels": [], "filenames": []}

    sample_count = 0
    print("Extracting embeddings...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            if sample_count >= max_samples:
                break

            anchor_images = batch["anchor_image"].to(device)
            embeddings = model(anchor_images)
            embeddings_list.append(embeddings.cpu().numpy())

            # Store metadata
            batch_size = len(batch["x"])
            metadata["x_coords"].extend([int(x) for x in batch["x"]])
            metadata["y_coords"].extend([int(y) for y in batch["y"]])
            metadata["zoom_levels"].extend([int(z) for z in batch["zoom"]])
            metadata["filenames"].extend(batch["filename"])

            sample_count += batch_size

    embeddings = np.vstack(embeddings_list)[:max_samples]
    for key in metadata:
        metadata[key] = metadata[key][:max_samples]

    return embeddings, metadata


def geographic_coherence_score(embeddings, coordinates):
    """Calculate correlation between embedding distance and geographic distance"""
    geo_distances = pdist(coordinates)
    embedding_distances = pdist(embeddings)
    correlation, p_value = spearmanr(geo_distances, embedding_distances)
    return correlation, p_value


def calculate_clustering_metrics(embeddings, labels):
    """Calculate various clustering quality metrics"""
    metrics = {}

    unique_labels = np.unique(labels)
    print(
        f"  Calculating metrics for {len(unique_labels)} unique labels: {unique_labels}"
    )

    # Skip if we don't have enough unique labels
    if len(unique_labels) < 2:
        print(f"  Skipping metrics: only {len(unique_labels)} unique label(s)")
        metrics["note"] = (
            f"Only {len(unique_labels)} unique label(s), metrics not applicable"
        )
        return metrics

    try:
        print("  Computing silhouette score...")
        metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        print("  Computing Calinski-Harabasz score...")
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(embeddings, labels)
        print("  Computing Davies-Bouldin score...")
        metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, labels)
        print("  All metrics computed successfully")
    except Exception as e:
        print(f"  Error calculating clustering metrics: {e}")
        metrics["error"] = str(e)

    return metrics


def perform_clustering_analysis(embeddings, metadata):
    """Perform clustering analysis using HDBSCAN and DBSCAN"""
    results = {}

    print("Performing clustering analysis...")

    # HDBSCAN clustering
    try:
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        hdbscan_labels = hdbscan_clusterer.fit_predict(embeddings)

        results["hdbscan"] = {
            "labels": hdbscan_labels,
            "n_clusters": len(np.unique(hdbscan_labels[hdbscan_labels >= 0])),
            "n_noise": np.sum(hdbscan_labels == -1),
            "cluster_probabilities": hdbscan_clusterer.probabilities_,
        }

        # Calculate metrics for HDBSCAN (excluding noise points)
        if results["hdbscan"]["n_clusters"] > 1:
            valid_mask = hdbscan_labels >= 0
            if np.sum(valid_mask) > 0:
                results["hdbscan"]["metrics"] = calculate_clustering_metrics(
                    embeddings[valid_mask], hdbscan_labels[valid_mask]
                )
    except Exception as e:
        print(f"HDBSCAN failed: {e}")

    # DBSCAN clustering
    try:
        dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan_clusterer.fit_predict(embeddings)

        results["dbscan"] = {
            "labels": dbscan_labels,
            "n_clusters": len(np.unique(dbscan_labels[dbscan_labels >= 0])),
            "n_noise": np.sum(dbscan_labels == -1),
        }

        # Calculate metrics for DBSCAN (excluding noise points)
        if results["dbscan"]["n_clusters"] > 1:
            valid_mask = dbscan_labels >= 0
            if np.sum(valid_mask) > 0:
                results["dbscan"]["metrics"] = calculate_clustering_metrics(
                    embeddings[valid_mask], dbscan_labels[valid_mask]
                )
    except Exception as e:
        print(f"DBSCAN failed: {e}")

    return results


def perform_dimensionality_reduction(embeddings):
    """Perform various dimensionality reduction techniques"""
    results = {}

    print("Performing dimensionality reduction...")

    # PCA
    try:
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(embeddings)
        results["pca"] = {
            "embeddings_2d": pca_2d,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_variance_explained": np.sum(pca.explained_variance_ratio_),
        }
    except Exception as e:
        print(f"PCA failed: {e}")

    # MDS
    try:
        mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean", n_init=1)
        mds_2d = mds.fit_transform(embeddings)
        results["mds"] = {"embeddings_2d": mds_2d, "stress": mds.stress_}
    except Exception as e:
        print(f"MDS failed: {e}")

    # t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_2d = tsne.fit_transform(embeddings)
        results["tsne"] = {"embeddings_2d": tsne_2d}
    except Exception as e:
        print(f"t-SNE failed: {e}")

    return results


def calculate_embedding_quality_metrics(embeddings, metadata, reduced_embeddings):
    """Calculate comprehensive embedding quality metrics"""
    results = {}

    print("Calculating embedding quality metrics...")

    # Geographic coherence
    if "x_coords" in metadata and "y_coords" in metadata:
        coordinates = np.column_stack([metadata["x_coords"], metadata["y_coords"]])
        geo_corr, geo_p_val = geographic_coherence_score(embeddings, coordinates)
        results["geographic_coherence"] = {
            "correlation": geo_corr,
            "p_value": geo_p_val,
        }

    # Trustworthiness
    try:
        if "x_coords" in metadata and "y_coords" in metadata:
            coordinates = np.column_stack([metadata["x_coords"], metadata["y_coords"]])
            trust_score = trustworthiness(coordinates, embeddings, n_neighbors=10)
            results["trustworthiness"] = trust_score
    except Exception as e:
        print(f"Trustworthiness calculation failed: {e}")

    # Metrics on reduced dimensions
    results["reduced_space_metrics"] = {}

    for method_name, method_data in reduced_embeddings.items():
        if "embeddings_2d" in method_data:
            method_results = {}

            print(f"Calculating metrics for {method_name.upper()}...")

            # Create zoom-based labels for clustering evaluation
            if "zoom_levels" in metadata:
                zoom_labels = np.array(metadata["zoom_levels"])
                print(
                    f"  Zoom levels: {np.unique(zoom_labels)} (total: {len(zoom_labels)} samples)"
                )

                if len(np.unique(zoom_labels)) > 1:
                    method_results["zoom_clustering"] = calculate_clustering_metrics(
                        method_data["embeddings_2d"], zoom_labels
                    )
                else:
                    print(
                        "  Only one unique zoom level, creating artificial clusters for evaluation..."
                    )
                    # Create artificial clusters using k-means for evaluation
                    from sklearn.cluster import KMeans

                    try:
                        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                        artificial_labels = kmeans.fit_predict(
                            method_data["embeddings_2d"]
                        )
                        method_results["artificial_clustering"] = (
                            calculate_clustering_metrics(
                                method_data["embeddings_2d"], artificial_labels
                            )
                        )
                    except Exception as e:
                        print(f"  Error creating artificial clusters: {e}")

            # Also calculate silhouette score on original embeddings if we have coordinates
            if "x_coords" in metadata and "y_coords" in metadata:
                coordinates = np.column_stack(
                    [metadata["x_coords"], metadata["y_coords"]]
                )
                # Create spatial clusters based on geographic proximity
                try:
                    from sklearn.cluster import KMeans

                    geo_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    geo_clusters = geo_kmeans.fit_predict(coordinates)
                    print(
                        f"  Geographic clusters: {len(np.unique(geo_clusters))} clusters"
                    )
                    method_results["geographic_clustering"] = (
                        calculate_clustering_metrics(
                            method_data["embeddings_2d"], geo_clusters
                        )
                    )
                except Exception as e:
                    print(f"  Error calculating geographic clustering: {e}")

            results["reduced_space_metrics"][method_name] = method_results

    return results


def create_visualizations(
    embeddings,
    metadata,
    reduced_embeddings,
    clustering_results,
    output_dir="analysis_plots",
):
    """Create comprehensive visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    print("Creating visualizations...")

    # Plot reduced embeddings
    methods = ["pca", "mds", "tsne"]
    available_methods = [method for method in methods if method in reduced_embeddings]
    n_methods = len(available_methods)

    if n_methods == 0:
        print("No dimensionality reduction methods succeeded, skipping visualization")
        return

    # Create subplot grid based on number of available methods
    if n_methods == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    elif n_methods == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

    fig.suptitle("Dimensionality Reduction Results", fontsize=16)

    colors = metadata.get("zoom_levels", [1] * len(embeddings))
    scatter = None  # To hold the last scatter plot for colorbar

    for i, method in enumerate(available_methods):
        if i >= len(axes):
            break

        ax = axes[i]
        coords_2d = reduced_embeddings[method]["embeddings_2d"]
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1], c=colors, cmap="viridis", alpha=0.6, s=20
        )
        ax.set_title(f"{method.upper()}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        if method == "pca":
            var_ratio = reduced_embeddings[method]["explained_variance_ratio"]
            ax.set_xlabel(f"PC1 ({var_ratio[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({var_ratio[1]:.2%} variance)")
        elif method == "mds":
            stress = reduced_embeddings[method]["stress"]
            ax.set_title(f"MDS (Stress: {stress:.3f})")

    # Hide unused subplots
    if len(axes) > n_methods:
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)

    # Add colorbar if we have a scatter plot
    if scatter is not None:
        plt.colorbar(scatter, ax=axes[:n_methods], label="Zoom Level")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/dimensionality_reduction.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot clustering results
    if clustering_results:
        # Filter methods that have labels
        valid_clustering = {
            k: v for k, v in clustering_results.items() if "labels" in v
        }

        if valid_clustering:
            n_clusters = len(valid_clustering)
            if n_clusters == 1:
                fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                axes = [axes]
            else:
                fig, axes = plt.subplots(1, min(n_clusters, 2), figsize=(15, 6))
                if n_clusters == 1:
                    axes = [axes]

            fig.suptitle("Clustering Results", fontsize=16)

            for i, (method, results) in enumerate(valid_clustering.items()):
                if i >= len(axes):
                    break

                ax = axes[i] if len(axes) > 1 else axes[0]

                # Use PCA coordinates for visualization if available
                if "pca" in reduced_embeddings:
                    coords_2d = reduced_embeddings["pca"]["embeddings_2d"]
                else:
                    coords_2d = embeddings[:, :2]  # Use first 2 dimensions

                labels = results["labels"]
                scatter = ax.scatter(
                    coords_2d[:, 0],
                    coords_2d[:, 1],
                    c=labels,
                    cmap="tab10",
                    alpha=0.6,
                    s=20,
                )
                ax.set_title(
                    f"{method.upper()}: {results['n_clusters']} clusters, "
                    f"{results['n_noise']} noise points"
                )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/clustering_results.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            print("No valid clustering results to visualize")


def save_results_to_file(
    quality_metrics,
    clustering_results,
    reduced_embeddings,
    output_file="embedding_analysis_results.txt",
):
    """Save all results to a text file"""
    with open(output_file, "w") as f:
        f.write("=== EMBEDDING ANALYSIS RESULTS ===\n\n")

        # Geographic coherence
        if "geographic_coherence" in quality_metrics:
            gc = quality_metrics["geographic_coherence"]
            f.write("Geographic Coherence:\n")
            f.write(f"  Correlation: {gc['correlation']:.4f}\n")
            f.write(f"  P-value: {gc['p_value']:.4e}\n\n")

        # Trustworthiness
        if "trustworthiness" in quality_metrics:
            f.write(
                f"Trustworthiness Score: {quality_metrics['trustworthiness']:.4f}\n\n"
            )

        # Dimensionality reduction results
        f.write("=== DIMENSIONALITY REDUCTION ===\n")
        for method, results in reduced_embeddings.items():
            f.write(f"\n{method.upper()}:\n")
            if "total_variance_explained" in results:
                f.write(
                    f"  Total Variance Explained: {results['total_variance_explained']:.4f}\n"
                )
            if "stress" in results:
                f.write(f"  Stress: {results['stress']:.4f}\n")

        # Clustering results
        f.write("\n=== CLUSTERING RESULTS ===\n")
        for method, results in clustering_results.items():
            f.write(f"\n{method.upper()}:\n")
            f.write(f"  Number of clusters: {results['n_clusters']}\n")
            f.write(f"  Number of noise points: {results['n_noise']}\n")

            if "metrics" in results:
                metrics = results["metrics"]
                f.write("  Clustering Metrics:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"    {metric_name}: {metric_value:.4f}\n")

        # Reduced space metrics
        if "reduced_space_metrics" in quality_metrics:
            f.write("\n=== REDUCED SPACE METRICS ===\n")
            for method, metrics in quality_metrics["reduced_space_metrics"].items():
                f.write(f"\n{method.upper()}:\n")
                for metric_type, metric_values in metrics.items():
                    f.write(f"  {metric_type}:\n")
                    if isinstance(metric_values, dict):
                        for metric_name, metric_value in metric_values.items():
                            if isinstance(metric_value, (int, float)):
                                f.write(f"    {metric_name}: {metric_value:.4f}\n")
                            else:
                                f.write(f"    {metric_name}: {metric_value}\n")
                    else:
                        f.write(f"    {metric_values}\n")


def load_model_and_visualize():
    """Load the saved model and visualize embeddings in TensorBoard"""

    # Device setup
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Model path
    model_path = "/Users/klaipher/University/UCU/Machine Learning/Project/saved_models/best_Custom Loc2Vec_Adam.pth"

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load the model state
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Initialize the model with the same parameters used during training
    # From the error, we can see the embedding dimension should be 16, not 64
    model = Loc2VecModel(input_channels=3, embedding_dim=16, dropout_rate=0.5)

    # Load the model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model from checkpoint with model_state_dict")
    else:
        # If the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")

    # Move model to device
    model.to(device)
    print("Model loaded successfully!")

    # Create dataset and dataloader
    print("Creating dataset...")

    # Use the same transforms as during training
    # Based on embeddings.py, they use different normalization values
    transform = T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize([0.8107, 0.8611, 0.7814], [0.1215, 0.0828, 0.1320]),
        ]
    )

    # Check for tiles directory
    tiles_dir = "full"
    if not os.path.exists(tiles_dir):
        print(f"Error: Tiles directory not found at {tiles_dir}")
        return

    dataset = TilesDataset(tiles_dir, pos_radius=1, transform=transform)

    print(f"Dataset loaded with {len(dataset)} samples")

    # Create dataloader with smaller batch size for embedding visualization
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # Log embeddings to TensorBoard
    print("Extracting embeddings and logging to TensorBoard...")
    log_embeddings_to_tensorboard(
        model=model,
        dataloader=dataloader,
        device=device,
        log_dir="logs/embeddings",
        max_samples=4000,  # Visualize samples
    )

    print("\n" + "=" * 60)
    print("COMPREHENSIVE EMBEDDING ANALYSIS")
    print("=" * 60)

    # Extract embeddings and metadata for analysis
    embeddings, metadata = extract_embeddings_and_metadata(
        model=model,
        dataloader=dataloader,
        device=device,
        max_samples=2000,  # Use smaller sample for analysis
    )

    print(
        f"Analyzing {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions"
    )

    # Perform dimensionality reduction
    reduced_embeddings = perform_dimensionality_reduction(embeddings)

    # Perform clustering analysis
    clustering_results = perform_clustering_analysis(embeddings, metadata)

    # Calculate quality metrics
    quality_metrics = calculate_embedding_quality_metrics(
        embeddings, metadata, reduced_embeddings
    )

    # Create visualizations
    create_visualizations(embeddings, metadata, reduced_embeddings, clustering_results)

    # Save results
    save_results_to_file(quality_metrics, clustering_results, reduced_embeddings)

    # Print summary results
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    # Geographic coherence
    if "geographic_coherence" in quality_metrics:
        gc = quality_metrics["geographic_coherence"]
        print(f"Geographic Coherence: {gc['correlation']:.4f} (p={gc['p_value']:.4e})")

    # Trustworthiness
    if "trustworthiness" in quality_metrics:
        print(f"Trustworthiness Score: {quality_metrics['trustworthiness']:.4f}")

    # Dimensionality reduction summary
    print("\nDimensionality Reduction:")
    for method, results in reduced_embeddings.items():
        if "total_variance_explained" in results:
            print(
                f"  {method.upper()}: {results['total_variance_explained']:.2%} variance explained"
            )
        elif "stress" in results:
            print(f"  {method.upper()}: Stress = {results['stress']:.4f}")
        else:
            print(f"  {method.upper()}: Completed successfully")

    # Clustering summary
    print("\nClustering Results:")
    for method, results in clustering_results.items():
        print(
            f"  {method.upper()}: {results['n_clusters']} clusters, {results['n_noise']} noise points"
        )
        if "metrics" in results and "silhouette_score" in results["metrics"]:
            print(f"    Silhouette Score: {results['metrics']['silhouette_score']:.4f}")

    print("\nDetailed results saved to: embedding_analysis_results.txt")
    print("Visualizations saved to: analysis_plots/")

    print("\n" + "=" * 60)
    print("TENSORBOARD VISUALIZATION")
    print("=" * 60)
    print("To view embeddings in TensorBoard, run:")
    print("tensorboard --logdir=logs/embeddings")
    print("\nThen navigate to http://localhost:6006 in your browser")
    print("Go to the 'PROJECTOR' tab to see the embeddings")


if __name__ == "__main__":
    load_model_and_visualize()
