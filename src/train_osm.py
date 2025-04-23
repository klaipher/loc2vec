import argparse
import time
from pathlib import Path
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from model import TripletLoc2Vec
from osm_data import create_osm_dataloader


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: Loc2Vec model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)

        # Forward pass
        optimizer.zero_grad()
        _, loss = model(anchor, positive, negative)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
            elapsed = time.time() - start_time
            batches_processed = batch_idx + 1
            print(
                f"Epoch {epoch} [{batches_processed}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Avg Loss (last 10): {total_loss / batches_processed:.4f} "
                f"Time/10 batches: {elapsed:.2f}s"
            )
            start_time = time.time()

            if writer:
                writer.add_scalar(
                    "train/batch_loss",
                    loss.item(),
                    epoch * len(dataloader) + batches_processed,
                )

    # Calculate average loss over the entire epoch
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")

    if writer and avg_loss > 0:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

    return avg_loss


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None,
) -> float:
    """
    Validate the model using a separate pre-prepared validation dataset.

    Args:
        model: Loc2Vec model
        dataloader: Validation dataloader
        device: Device to validate on
        epoch: Current epoch number
        writer: TensorBoard writer

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0

    if dataloader is None:
        print("No validation data provided, skipping validation.")
        return float("inf")  # Return infinity if no validation

    print("Starting validation...")
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            # Forward pass
            _, loss = model(anchor, positive, negative)

            # Logging
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    print(f"Validation - Epoch {epoch} complete. Average loss: {avg_loss:.4f}")

    if writer and avg_loss > 0:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)

    return avg_loss


def main(args):
    # Set device
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif not args.no_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Training data directory
    train_data_dir = Path(args.prepared_data_dir)
    if not train_data_dir.is_dir():
        print(f"Error: Prepared training data directory not found: {train_data_dir}")
        return

    # Load patch size from training metadata
    train_metadata_path = train_data_dir / "metadata.json"
    if not train_metadata_path.exists():
        print(
            f"Error: Metadata file not found in training data directory: {train_metadata_path}"
        )
        return
    with open(train_metadata_path, "r") as f:
        train_metadata = json.load(f)
    patch_size = train_metadata.get(
        "patch_size", args.patch_size
    )  # Use metadata value if available
    print(f"Using patch size: {patch_size} (from metadata or args)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # Data transforms (Normalization based on ImageNet stats, common for OSM tiles too)
    transform = transforms.Compose(
        [
            # Assuming tiles are loaded as 0-255 numpy arrays
            # Normalization happens in the Dataset __getitem__ after converting to float tensor
            # If needed, add more transforms here, e.g., random flips
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    # Create training dataloader
    print(f"Loading training data from: {train_data_dir}")
    train_dataloader = create_osm_dataloader(
        prepared_data_dir=str(train_data_dir),
        batch_size=args.batch_size,
        transform=transform,
        num_workers=args.num_workers,
        # Pass distance parameters from args
        max_distance_positive=args.max_distance_positive,
        min_distance_negative=args.min_distance_negative,
    )

    # Create validation dataloader if validation data dir is provided
    val_dataloader = None
    if args.val_data_dir:
        val_data_dir = Path(args.val_data_dir)
        if val_data_dir.is_dir() and (val_data_dir / "metadata.json").exists():
            print(f"Loading validation data from: {val_data_dir}")
            # Use same transforms for validation?
            val_transform = transforms.Compose(
                [
                    # Typically no augmentation for validation
                ]
            )
            val_dataloader = create_osm_dataloader(
                prepared_data_dir=str(val_data_dir),
                batch_size=args.batch_size,  # Use same batch size or smaller for validation?
                transform=val_transform,  # Potentially different transforms
                num_workers=args.num_workers,
                max_distance_positive=args.max_distance_positive,
                min_distance_negative=args.min_distance_negative,
            )
        else:
            print(
                f"Warning: Validation data directory not found or missing metadata: {val_data_dir}"
            )

    # Create model
    model = TripletLoc2Vec(
        input_channels=args.input_channels,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
    ).to(device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Removed verbose=True as it's deprecated
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    best_val_loss = float("inf")
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        # Train
        train_loss = train(model, train_dataloader, optimizer, device, epoch, writer)

        # Validate
        val_loss = validate(model, val_dataloader, device, epoch, writer)

        # Update learning rate based on validation loss (or train loss if no validation)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss if val_dataloader else train_loss)
        if optimizer.param_groups[0]["lr"] < current_lr:
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss if val_dataloader else None,
        }

        # Save latest model
        checkpoint_path = output_dir / "model_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved latest checkpoint to {checkpoint_path}")

        # Save best model based on validation loss
        if val_dataloader and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_dir / "model_best.pt"
            torch.save(checkpoint, best_checkpoint_path)
            print(
                f"*** New best validation model saved! Validation loss: {val_loss:.4f} ***"
            )
        elif (
            not val_dataloader and train_loss < best_val_loss
        ):  # Use train loss if no validation
            best_val_loss = train_loss
            best_checkpoint_path = output_dir / "model_best.pt"
            torch.save(checkpoint, best_checkpoint_path)
            print(
                f"*** New best training model saved! Training loss: {train_loss:.4f} ***"
            )

    # Save final model state dict along with metadata
    final_model_path = output_dir / "model_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "input_channels": args.input_channels,
            "patch_size": patch_size,  # Save the actual patch size used
            "training_args": vars(args),  # Save training arguments
        },
        final_model_path,
    )

    print(f"\nTraining complete! Final model saved to {final_model_path}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Loc2Vec model using pre-prepared OpenStreetMap data"
    )

    # Data parameters
    parser.add_argument(
        "--prepared_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing prepared OSM data (tiles and metadata.json)",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        help="Path to the directory containing prepared validation OSM data (optional)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        help="Size of image patches (override metadata if needed, otherwise loaded from metadata)",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="Number of input channels expected by the model (usually 3 for RGB OSM tiles)",
    )
    parser.add_argument(
        "--max_distance_positive",
        type=float,
        default=0.001,
        help="Maximum distance (degrees) for positive samples during training",
    )
    parser.add_argument(
        "--min_distance_negative",
        type=float,
        default=0.01,
        help="Minimum distance (degrees) for negative samples during training",
    )

    # Model parameters
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="Dimension of location embeddings"
    )
    parser.add_argument(
        "--margin", type=float, default=0.3, help="Margin for triplet loss"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for models and logs",
    )

    args = parser.parse_args()
    main(args)
