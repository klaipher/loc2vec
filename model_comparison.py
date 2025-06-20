import torch
import torch.optim as optim
import time
import psutil
import gc
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from loc2vec.train import train as train_epoch

from loc2vec.dataset import TilesDataset

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from loc2vec.model import (
    Loc2VecModel,
    EfficientNetV2SLoc2Vec,
    EfficientNetV2MLoc2Vec,
    ResNetLoc2Vec,
    MobileNetV3SmallLoc2Vec,
    SoftmaxTripletLoss,
)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_embeddings(model, train_loader, device, max_samples=500):
    """Evaluate embedding quality using silhouette score"""
    model.eval()
    embeddings = []
    spatial_labels = []

    with torch.no_grad():
        sample_count = 0
        for batch_data in train_loader:
            if sample_count >= max_samples:
                break

            try:
                anchor = batch_data["anchor_image"].to(device)
                anchor_emb = model(anchor).cpu().numpy()
                print("Processing batch with shape:", anchor_emb.shape)

                # Check for NaN or infinite values
                if np.any(np.isnan(anchor_emb)) or np.any(np.isinf(anchor_emb)):
                    print("Warning: NaN/Inf detected in embeddings, skipping batch")
                    continue

                embeddings.append(anchor_emb)

                # Create spatial pseudo-labels based on coordinates if available
                # If no coordinates, create labels based on batch position as approximation
                if "coordinates" in batch_data:
                    coords = batch_data["coordinates"].numpy()
                    # Discretize coordinates into spatial bins for clustering
                    lat_bins = np.digitize(
                        coords[:, 0],
                        bins=np.linspace(coords[:, 0].min(), coords[:, 0].max(), 10),
                    )
                    lon_bins = np.digitize(
                        coords[:, 1],
                        bins=np.linspace(coords[:, 1].min(), coords[:, 1].max(), 10),
                    )
                    labels = lat_bins * 10 + lon_bins
                else:
                    # Fallback: use simple sequential labeling
                    labels = np.full(
                        anchor_emb.shape[0], len(embeddings) % 5
                    )  # Create 5 clusters

                spatial_labels.append(labels)
                sample_count += anchor_emb.shape[0]

            except Exception as e:
                print(f"Warning: Error processing batch in embedding evaluation: {e}")
                continue

    if len(embeddings) < 2:
        print("Warning: Not enough valid embeddings for silhouette score")
        return 0.0  # Not enough data for silhouette score

    try:
        # Concatenate all embeddings and labels
        all_embeddings = np.vstack(embeddings)
        all_labels = np.concatenate(spatial_labels)

        # Final check for NaN values
        if np.any(np.isnan(all_embeddings)) or np.any(np.isinf(all_embeddings)):
            print("Warning: NaN/Inf found in concatenated embeddings")
            return 0.0

        # If we don't have real spatial labels, use KMeans clustering
        if "coordinates" not in batch_data:
            n_clusters = min(
                5, max(2, len(np.unique(all_labels)))
            )  # Ensure 2-5 clusters
            if n_clusters > 1 and len(all_embeddings) >= n_clusters:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(all_embeddings)
                except Exception as e:
                    print(f"Warning: KMeans clustering failed: {e}")
                    return 0.0
            else:
                return 0.0
        else:
            cluster_labels = all_labels

        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1 and len(all_embeddings) > 1:
            try:
                # Ensure we have enough samples per cluster
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                if np.all(counts >= 1) and len(unique_labels) >= 2:
                    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
                    return float(silhouette_avg)
                else:
                    return 0.0
            except Exception as e:
                print(f"Warning: Silhouette score calculation failed: {e}")
                return 0.0
        else:
            return 0.0

    except Exception as e:
        print(f"Warning: Error in embedding evaluation: {e}")
        return 0.0


def benchmark_model(model_class, model_name, train_loader, device, epochs=3):
    """Benchmark a single model configuration"""
    results = []

    base_lr = 1e-4  # Standard LR for pre-trained models

    # Test different optimizers and schedulers
    configs = [
        {"optimizer": "Adam", "lr": base_lr, "scheduler": None},
        {"optimizer": "AdamW", "lr": base_lr, "scheduler": None},
    ]

    for config in configs:
        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Initialize model
            model = model_class(input_channels=3, embedding_dim=16, dropout_rate=0.5)
            model.to(device)

            # Count parameters
            param_count = count_parameters(model)

            # Setup optimizer
            if config["optimizer"] == "AdamW":
                optimizer = optim.Adam(model.parameters(), lr=config["lr"])
            else:  # AdamW
                optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

            # Setup loss function (Step 3 - More stable loss)
            loss_fn = SoftmaxTripletLoss()

            # Memory before training
            memory_before = get_memory_usage()

            # Training with gradient clipping (Step 2)
            start_time = time.time()
            epoch_losses = []

            for epoch in range(epochs):
                epoch_loss = train_epoch(
                    model, train_loader, optimizer, loss_fn, device, scheduler=None
                )
                # Add gradient clipping after training step
                epoch_losses.append(epoch_loss)

            training_time = time.time() - start_time

            # Better loss analysis
            avg_loss = np.mean(epoch_losses)
            final_loss = epoch_losses[-1] if epoch_losses else 0
            loss_std = np.std(epoch_losses) if len(epoch_losses) > 1 else 0
            min_loss = np.min(epoch_losses) if epoch_losses else 0
            max_loss = np.max(epoch_losses) if epoch_losses else 0
            loss_trend = (
                "Improving"
                if len(epoch_losses) > 1 and epoch_losses[-1] < epoch_losses[0]
                else "Stable/Degrading"
            )
            loss_string = ", ".join([f"{loss:.4f}" for loss in epoch_losses])

            # Memory after training
            memory_after = get_memory_usage()
            memory_used = memory_after - memory_before

            # Evaluate embedding quality
            silhouette_avg = 0  # evaluate_embeddings(model, train_loader, device)

            # Record results
            result = {
                "Model": model_name,
                "Optimizer": config["optimizer"],
                "Scheduler": config["scheduler"] or "None",
                "Parameters (M)": param_count / 1e6,
                "Training Time (s)": training_time,
                "Memory Used (MB)": memory_used,
                "Final Loss": final_loss,
                "Avg Loss": avg_loss,
                "Min Loss": min_loss,
                "Max Loss": max_loss,
                "Loss Std": loss_std,
                "Loss Trend": loss_trend,
                "All Losses": loss_string,
                "Silhouette Score": silhouette_avg,
                "Time per Epoch (s)": training_time / epochs,
            }
            results.append(result)

            print(
                f"✓ {model_name} - {config['optimizer']} - {config['scheduler'] or 'None'}"
            )

        except Exception as e:
            print(
                f"✗ {model_name} - {config['optimizer']} - {config['scheduler'] or 'None'}: {str(e)}"
            )
            continue

    return results


def run_comprehensive_comparison(train_loader, device):
    """Run comprehensive comparison of all models"""

    # Model configurations
    models_to_test = [
        (Loc2VecModel, "Custom Loc2Vec"),
        # (EfficientNetLoc2Vec, "EfficientNet B0"),
        (EfficientNetV2SLoc2Vec, "EfficientNetV2-S"),
        (EfficientNetV2MLoc2Vec, "EfficientNetV2-M"),
        (ResNetLoc2Vec, "ResNet50"),
        # (ConvNeXtLoc2Vec, "ConvNeXt-Small"),
        # (SwinTransformerLoc2Vec, "Swin-Small"),
        # (MobileNetV3Loc2Vec, "MobileNetV3-Large"),
        (MobileNetV3SmallLoc2Vec, "MobileNetV3-Small"),
    ]

    all_results = []

    print("Starting comprehensive model comparison...")
    print(f"Device: {device}")
    print("-" * 50)

    for model_class, model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            results = benchmark_model(model_class, model_name, train_loader, device)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            continue
        else:
            all_results.extend(results)

    # Create DataFrame and sort by performance
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80)

    # Display results
    print("\n📊 FULL RESULTS TABLE:")
    print(df.round(4).to_string(index=False))

    # Best performers analysis
    print("\n🏆 BEST PERFORMERS BY CATEGORY:")
    print("-" * 40)

    best_speed = df.loc[df["Training Time (s)"].idxmin()]
    print(
        f"⚡ Fastest: {best_speed['Model']} ({best_speed['Optimizer']}) - {best_speed['Training Time (s)']:.2f}s"
    )

    best_memory = df.loc[df["Memory Used (MB)"].idxmin()]
    print(
        f"💾 Most Memory Efficient: {best_memory['Model']} ({best_memory['Optimizer']}) - {best_memory['Memory Used (MB)']:.1f}MB"
    )

    best_final_loss = df.loc[df["Final Loss"].idxmin()]
    print(
        f"🎯 Best Final Loss: {best_final_loss['Model']} ({best_final_loss['Optimizer']}) - {best_final_loss['Final Loss']:.4f}"
    )

    most_stable = df.loc[df["Loss Std"].idxmin()]
    print(
        f"📈 Most Stable Training: {most_stable['Model']} ({most_stable['Optimizer']}) - Std: {most_stable['Loss Std']:.4f}"
    )

    best_silhouette = df.loc[df["Silhouette Score"].idxmax()]
    print(
        f"🎨 Best Embeddings: {best_silhouette['Model']} ({best_silhouette['Optimizer']}) - {best_silhouette['Silhouette Score']:.4f}"
    )

    smallest_model = df.loc[df["Parameters (M)"].idxmin()]
    print(
        f"📦 Smallest Model: {smallest_model['Model']} - {smallest_model['Parameters (M)']:.1f}M params"
    )
    return df


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = TilesDataset(
        "full",
        pos_radius=1,
        transform=T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize([0.8107, 0.8611, 0.7814], [0.1215, 0.0828, 0.1320]),
            ]
        ),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        prefetch_factor=10,
        persistent_workers=True,
    )
    run_comprehensive_comparison(train_loader, device)
