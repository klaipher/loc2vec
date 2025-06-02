import argparse
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import create_data_loader
from model import create_model


class Loc2VecTrainer:
    """
    Trainer class for the loc2vec model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-3,
        device=None,
        log_dir="runs",
        checkpoint_dir="checkpoints",
        save_every=10,
    ):
        """
        Args:
            model: Loc2VecModel instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate for optimizer
            device: Device to train on
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for model checkpoints
            save_every: Save checkpoint every N epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(self.device)
        self.save_every = save_every

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        # Track learning rate for custom verbose logging
        self.last_lr = learning_rate

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"loc2vec_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)

        # Setup checkpoints
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "pos_dist": [],
            "neg_dist": [],
            "triplets_used": [],
        }

    def save_checkpoint(self, filename=None, is_best=False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"loc2vec_epoch_{self.epoch}.pth"

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.epoch}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_pos_dist = 0
        total_neg_dist = 0
        total_triplets_used = 0
        total_triplets = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            anchor = batch["anchor"].to(self.device)
            positive = batch["positive"].to(self.device)
            negative = batch["negative"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            anchor_emb, pos_emb, neg_emb, loss, metrics = self.model(
                anchor, positive, negative
            )

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_pos_dist += metrics["pos_dist_mean"]
            total_neg_dist += metrics["neg_dist_mean"]
            total_triplets_used += metrics["triplets_used"]
            total_triplets += metrics["total_triplets"]

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Pos Dist": f"{metrics['pos_dist_mean']:.4f}",
                    "Neg Dist": f"{metrics['neg_dist_mean']:.4f}",
                    "Triplets": f"{metrics['triplets_used']}/{metrics['total_triplets']}",
                }
            )

            # Log to tensorboard every 100 batches
            if batch_idx % 100 == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("Train/Loss_Step", loss.item(), global_step)
                self.writer.add_scalar(
                    "Train/PosDist_Step", metrics["pos_dist_mean"], global_step
                )
                self.writer.add_scalar(
                    "Train/NegDist_Step", metrics["neg_dist_mean"], global_step
                )

        # Calculate epoch averages
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_pos_dist = total_pos_dist / num_batches
        avg_neg_dist = total_neg_dist / num_batches
        triplet_ratio = (
            total_triplets_used / total_triplets if total_triplets > 0 else 0
        )

        return avg_loss, avg_pos_dist, avg_neg_dist, triplet_ratio

    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None, None, None, None

        self.model.eval()

        total_loss = 0
        total_pos_dist = 0
        total_neg_dist = 0
        total_triplets_used = 0
        total_triplets = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                anchor = batch["anchor"].to(self.device)
                positive = batch["positive"].to(self.device)
                negative = batch["negative"].to(self.device)

                # Forward pass
                anchor_emb, pos_emb, neg_emb, loss, metrics = self.model(
                    anchor, positive, negative
                )

                # Update statistics
                total_loss += loss.item()
                total_pos_dist += metrics["pos_dist_mean"]
                total_neg_dist += metrics["neg_dist_mean"]
                total_triplets_used += metrics["triplets_used"]
                total_triplets += metrics["total_triplets"]

        # Calculate averages
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_pos_dist = total_pos_dist / num_batches
        avg_neg_dist = total_neg_dist / num_batches
        triplet_ratio = (
            total_triplets_used / total_triplets if total_triplets > 0 else 0
        )

        return avg_loss, avg_pos_dist, avg_neg_dist, triplet_ratio

    def train(self, num_epochs):
        """Train the model for specified number of epochs."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples per epoch: {len(self.train_loader.dataset)}")

        start_epoch = self.epoch

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_pos_dist, train_neg_dist, train_triplet_ratio = (
                self.train_epoch()
            )

            # Validate
            val_loss, val_pos_dist, val_neg_dist, val_triplet_ratio = self.validate()

            epoch_time = time.time() - start_time

            # Update learning rate and detect changes
            current_lr = self.optimizer.param_groups[0]["lr"]
            if val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)

            # Check if learning rate changed and log it
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                print(f"  Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
            self.last_lr = new_lr

            # Log to tensorboard
            self.writer.add_scalar("Train/Loss_Epoch", train_loss, epoch)
            self.writer.add_scalar("Train/PosDist_Epoch", train_pos_dist, epoch)
            self.writer.add_scalar("Train/NegDist_Epoch", train_neg_dist, epoch)
            self.writer.add_scalar("Train/TripletRatio", train_triplet_ratio, epoch)
            self.writer.add_scalar(
                "Train/LearningRate", self.optimizer.param_groups[0]["lr"], epoch
            )

            if val_loss is not None:
                self.writer.add_scalar("Val/Loss_Epoch", val_loss, epoch)
                self.writer.add_scalar("Val/PosDist_Epoch", val_pos_dist, epoch)
                self.writer.add_scalar("Val/NegDist_Epoch", val_neg_dist, epoch)
                self.writer.add_scalar("Val/TripletRatio", val_triplet_ratio, epoch)

            # Update training history
            self.training_history["train_loss"].append(train_loss)
            if val_loss is not None:
                self.training_history["val_loss"].append(val_loss)
            self.training_history["pos_dist"].append(train_pos_dist)
            self.training_history["neg_dist"].append(train_neg_dist)
            self.training_history["triplets_used"].append(train_triplet_ratio)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Pos/Neg Dist: {train_pos_dist:.4f}/{train_neg_dist:.4f}")
            print(f"  Train Triplet Ratio: {train_triplet_ratio:.3f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Pos/Neg Dist: {val_pos_dist:.4f}/{val_neg_dist:.4f}")
                print(f"  Val Triplet Ratio: {val_triplet_ratio:.3f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()

            # Save best model
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.save_checkpoint(is_best=True)
                print(f"  New best model saved! Loss: {current_loss:.4f}")

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final model
        self.save_checkpoint(filename="final_model.pth")

        # Close tensorboard writer
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Loc2Vec model")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--pos_radius", type=int, default=2, help="Positive radius")
    parser.add_argument("--neg_radius", type=int, default=10, help="Negative radius")
    parser.add_argument(
        "--max_samples", type=int, default=10000, help="Max samples per epoch"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="softpn",
        choices=["triplet", "softpn"],
        help="Type of triplet loss",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="cnn",
        choices=["cnn", "resnet18"],
        help="Type of encoder (cnn: custom CNN, resnet18: ResNet18 with pre-trained weights)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pre-trained weights for ResNet18 (only applicable when --encoder_type=resnet18)",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_false",
        dest="pretrained",
        help="Don't use pre-trained weights for ResNet18",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all_layers",
        choices=["all_layers", "full_only"],
        help="Data loading mode - 'all_layers' (all individual layers + full) or 'full_only' (just full layer)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.encoder_type != "resnet18" and not args.pretrained:
        print(
            "Warning: --pretrained flag is only applicable with --encoder_type=resnet18"
        )

    # Create data loaders first to get the correct number of input channels
    print("Creating data loaders...")
    train_loader = create_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tile_size=args.tile_size,
        positive_radius=args.pos_radius,
        negative_radius=args.neg_radius,
        max_samples_per_epoch=args.max_samples,
        mode=args.mode,
    )

    # Get the number of input channels from the dataset
    input_channels = train_loader.dataset.input_channels
    print(f"Mode: {args.mode}")
    print(f"Input channels: {input_channels}")

    # Create model with the correct number of input channels
    model = create_model(
        input_channels=input_channels,
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        loss_type=args.loss_type,
        dropout_rate=args.dropout,
        encoder_type=args.encoder_type,
        pretrained=args.pretrained,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created with encoder type: {args.encoder_type}")
    if args.encoder_type == "resnet18":
        print(f"Pre-trained weights: {'Yes' if args.pretrained else 'No'}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Loc2VecTrainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.lr,
        save_every=args.save_every,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train the model
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
