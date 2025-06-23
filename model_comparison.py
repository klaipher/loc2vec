import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

def calculate_silhouette_scores(X, cluster_range=(2, 11), models=None):
    """
    Calculate silhouette scores for different clustering models and cluster numbers.
    
    Args:
        X: Input data for clustering
        cluster_range: Tuple of (min_clusters, max_clusters)
        models: Dict of clustering models to compare (default: KMeans only)
    
    Returns:
        Dict containing silhouette scores for each model and cluster count
    """
    if models is None:
        models = {'KMeans': KMeans}
    
    results = {}
    
    for model_name, model_class in models.items():
        results[model_name] = {}
        
        for n_clusters in range(cluster_range[0], cluster_range[1]):
            # Initialize and fit the model
            model = model_class(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X)
            
            # Calculate silhouette score
            score = silhouette_score(X, cluster_labels)
            results[model_name][n_clusters] = score
            
            print(f"{model_name} with {n_clusters} clusters: {score:.4f}")
    
    return results

def plot_silhouette_scores(results):
    """Plot silhouette scores for comparison."""
    plt.figure(figsize=(10, 6))
    
    for model_name, scores in results.items():
        clusters = list(scores.keys())
        silhouette_scores = list(scores.values())
        plt.plot(clusters, silhouette_scores, marker='o', label=model_name)
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data or load your data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # Calculate silhouette scores
    results = calculate_silhouette_scores(X)
    
    # Plot results
    plot_silhouette_scores(results)