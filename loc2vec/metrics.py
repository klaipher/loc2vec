import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
try:
    from sklearn.manifold import trustworthiness
except ImportError:
    # Fallback for older sklearn versions
    def trustworthiness(*args, **kwargs):
        return 0.0

from sklearn.cluster import DBSCAN
import hdbscan

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from loc2vec.model import Loc2VecModel
from loc2vec.memmap_tiles_dataset import MemmapTripletTilesDataset


def extract_embeddings_and_metadata(model, dataloader, device, max_samples=10000):
    """Extract embeddings and metadata from the model"""
    model.eval()

    embeddings_list = []
    metadata = {"x_coords": [], "y_coords": [], "zoom_levels": []}

    sample_count = 0
    print("Extracting embeddings...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            if sample_count >= max_samples:
                break

            anchor_images = batch["anchor_image"].to(device)
            embeddings = model(anchor_images)
            embeddings_list.append(embeddings.cpu().numpy())

            # Store metadata - fix the batch access
            batch_size = len(batch["x"])
            metadata["x_coords"].extend(batch["x"].tolist())
            metadata["y_coords"].extend(batch["y"].tolist())  
            metadata["zoom_levels"].extend(batch["zoom"].tolist())

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
    print(f"  Calculating metrics for {len(unique_labels)} unique labels: {unique_labels}")

    # Skip if we don't have enough unique labels
    if len(unique_labels) < 2:
        print(f"  Skipping metrics: only {len(unique_labels)} unique label(s)")
        metrics["note"] = f"Only {len(unique_labels)} unique label(s), metrics not applicable"
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
        print("  PCA completed successfully")
    except Exception as e:
        print(f"PCA failed: {e}")

    # # MDS
    # try:
    #     mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean", n_init=1)
    #     mds_2d = mds.fit_transform(embeddings)
    #     results["mds"] = {"embeddings_2d": mds_2d, "stress": mds.stress_}
    #     print("  MDS completed successfully")
    # except Exception as e:
    #     print(f"MDS failed: {e}")

    # t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_2d = tsne.fit_transform(embeddings)
        results["tsne"] = {"embeddings_2d": tsne_2d}
        print("  t-SNE completed successfully")
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
                print(f"  Zoom levels: {np.unique(zoom_labels)} (total: {len(zoom_labels)} samples)")

                if len(np.unique(zoom_labels)) > 1:
                    method_results["zoom_clustering"] = calculate_clustering_metrics(
                        method_data["embeddings_2d"], zoom_labels
                    )

            # Calculate silhouette score on original embeddings with geographic clusters
            if "x_coords" in metadata and "y_coords" in metadata:
                coordinates = np.column_stack([metadata["x_coords"], metadata["y_coords"]])
                # Create spatial clusters based on geographic proximity
                try:
                    from sklearn.cluster import KMeans
                    geo_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    geo_clusters = geo_kmeans.fit_predict(coordinates)
                    print(f"  Geographic clusters: {len(np.unique(geo_clusters))} clusters")
                    method_results["geographic_clustering"] = calculate_clustering_metrics(
                        method_data["embeddings_2d"], geo_clusters
                    )
                except Exception as e:
                    print(f"  Error calculating geographic clustering: {e}")

            results["reduced_space_metrics"][method_name] = method_results

    return results


def create_visualizations(embeddings, metadata, reduced_embeddings, clustering_results, output_dir="analysis_plots"):
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

    # Create subplot grid
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
    scatter = None

    for i, method in enumerate(available_methods):
        if i >= len(axes):
            break

        ax = axes[i]
        coords_2d = reduced_embeddings[method]["embeddings_2d"]
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, cmap="viridis", alpha=0.6, s=20)
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

    # Add colorbar
    if scatter is not None:
        plt.colorbar(scatter, ax=axes[:n_methods], label="Zoom Level")

    # plt.tight_layout()
    plt.savefig(f"{output_dir}/dimensionality_reduction.png", dpi=300, bbox_inches="tight")
    plt.show()  # Show the plot in notebook


def save_results_to_file(quality_metrics, clustering_results, reduced_embeddings, output_file="embedding_analysis_results.txt"):
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
            f.write(f"Trustworthiness Score: {quality_metrics['trustworthiness']:.4f}\n\n")

        # Dimensionality reduction results
        f.write("=== DIMENSIONALITY REDUCTION ===\n")
        for method, results in reduced_embeddings.items():
            f.write(f"\n{method.upper()}:\n")
            if "total_variance_explained" in results:
                f.write(f"  Total Variance Explained: {results['total_variance_explained']:.4f}\n")
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


def load_model_and_visualize(model_path, dataloader, model, device):
    """Load the saved model and visualize embeddings"""

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model from checkpoint with model_state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")

    model.to(device)
    print("Model loaded successfully!")

    # Extract embeddings
    embeddings, metadata = extract_embeddings_and_metadata(
        model=model,
        dataloader=dataloader,
        device=device,
        max_samples=10000,
    )

    print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Perform analysis
    reduced_embeddings = perform_dimensionality_reduction(embeddings)
    clustering_results = perform_clustering_analysis(embeddings, metadata)
    quality_metrics = calculate_embedding_quality_metrics(embeddings, metadata, reduced_embeddings)

    # Create visualizations and save results
    create_visualizations(embeddings, metadata, reduced_embeddings, clustering_results)
    save_results_to_file(quality_metrics, clustering_results, reduced_embeddings)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    if "geographic_coherence" in quality_metrics:
        gc = quality_metrics["geographic_coherence"]
        print(f"Geographic Coherence: {gc['correlation']:.4f} (p={gc['p_value']:.4e})")

    if "trustworthiness" in quality_metrics:
        print(f"Trustworthiness Score: {quality_metrics['trustworthiness']:.4f}")

    print("\nDimensionality Reduction:")
    for method, results in reduced_embeddings.items():
        if "total_variance_explained" in results:
            print(f"  {method.upper()}: {results['total_variance_explained']:.2%} variance explained")
        elif "stress" in results:
            print(f"  {method.upper()}: Stress = {results['stress']:.4f}")
        else:
            print(f"  {method.upper()}: Completed successfully")

    print("\nClustering Results:")
    for method, results in clustering_results.items():
        print(f"  {method.upper()}: {results['n_clusters']} clusters, {results['n_noise']} noise points")
        if "metrics" in results and "silhouette_score" in results["metrics"]:
            print(f"    Silhouette Score: {results['metrics']['silhouette_score']:.4f}")

    print("\nDetailed results saved to: embedding_analysis_results.txt")

    return embeddings, metadata, reduced_embeddings, clustering_results, quality_metrics