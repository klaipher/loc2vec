import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Loc2VecEncoder(nn.Module):
    """
    CNN encoder for location tiles following the loc2vec methodology.

    Takes 12-channel input (12 layers of map data) and produces a
    fixed-size embedding vector representing the location.
    """

    def __init__(
        self,
        input_channels: int = 12,
        embedding_dim: int = 128,
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            input_channels: Number of input channels (12 for the map layers)
            embedding_dim: Dimension of the output embedding
            dropout_rate: Dropout rate for regularization
        """
        super(Loc2VecEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Convolutional layers with batch normalization and dropout
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            # Fifth conv block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )

        # Calculate the size after conv layers
        # For 256x256 input -> 8x8 output with 512 channels
        conv_output_size = 8 * 8 * 512

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            # Final embedding layer (no activation for better embedding space)
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, 12, H, W)

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        # L2 normalize the embeddings for better triplet loss performance
        x = F.normalize(x, p=2, dim=1)

        return x


class TripletLoss(nn.Module):
    """
    Triplet loss with online hard negative mining.

    Implements both the standard triplet loss and the SoftPN triplet loss
    as mentioned in the research papers.
    """

    def __init__(
        self,
        margin: float = 0.2,
        loss_type: str = "triplet",  # 'triplet' or 'softpn'
        hard_mining: bool = True,
    ):
        """
        Args:
            margin: Margin for triplet loss
            loss_type: Type of loss ('triplet' or 'softpn')
            hard_mining: Whether to use hard negative mining
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_type = loss_type
        self.hard_mining = hard_mining

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute pairwise distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        if self.loss_type == "triplet":
            # Standard triplet loss
            loss = F.relu(pos_dist - neg_dist + self.margin)

        elif self.loss_type == "softpn":
            # SoftPN triplet loss as described in the papers
            # Also consider distance between positive and negative
            pn_dist = F.pairwise_distance(positive, negative, p=2)
            min_neg_dist = torch.min(neg_dist, pn_dist)
            loss = F.relu(pos_dist - min_neg_dist + self.margin)

        # Apply hard mining if enabled
        if self.hard_mining:
            # Only consider the hardest examples (positive loss)
            hard_mask = loss > 0
            if hard_mask.sum() > 0:
                loss = loss[hard_mask].mean()
            else:
                loss = torch.tensor(0.0, device=anchor.device, requires_grad=True)
        else:
            loss = loss.mean()

        # Calculate metrics
        metrics = {
            "pos_dist_mean": pos_dist.mean().item(),
            "neg_dist_mean": neg_dist.mean().item(),
            "triplets_used": (loss > 0).sum().item() if self.hard_mining else len(loss),
            "total_triplets": len(pos_dist),
        }

        return loss, metrics


class Loc2VecModel(nn.Module):
    """
    Complete loc2vec model combining encoder and triplet loss.
    """

    def __init__(
        self,
        input_channels: int = 12,
        embedding_dim: int = 128,
        margin: float = 0.2,
        loss_type: str = "softpn",
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            input_channels: Number of input channels
            embedding_dim: Dimension of embeddings
            margin: Triplet loss margin
            loss_type: Type of triplet loss ('triplet' or 'softpn')
            dropout_rate: Dropout rate
        """
        super(Loc2VecModel, self).__init__()

        self.encoder = Loc2VecEncoder(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        self.triplet_loss = TripletLoss(
            margin=margin, loss_type=loss_type, hard_mining=True
        )

        self.embedding_dim = embedding_dim

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]
    ]:
        """
        Forward pass through the complete model.

        Args:
            anchor: Anchor tiles (batch_size, 12, H, W)
            positive: Positive tiles (batch_size, 12, H, W)
            negative: Negative tiles (batch_size, 12, H, W)

        Returns:
            Tuple of (anchor_emb, positive_emb, negative_emb, loss, metrics)
        """
        # Get embeddings for all three inputs
        anchor_emb = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)

        # Compute triplet loss
        loss, metrics = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

        return anchor_emb, positive_emb, negative_emb, loss, metrics

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Encode tiles to embeddings (for inference).

        Args:
            tiles: Input tiles (batch_size, 12, H, W)

        Returns:
            Embeddings (batch_size, embedding_dim)
        """
        with torch.no_grad():
            return self.encoder(tiles)


def create_model(
    embedding_dim: int = 128,
    margin: float = 0.2,
    loss_type: str = "softpn",
    dropout_rate: float = 0.5,
) -> Loc2VecModel:
    """
    Create a loc2vec model with specified parameters.

    Args:
        embedding_dim: Dimension of location embeddings
        margin: Triplet loss margin
        loss_type: Type of triplet loss
        dropout_rate: Dropout rate for regularization

    Returns:
        Initialized Loc2VecModel
    """
    model = Loc2VecModel(
        input_channels=12,
        embedding_dim=embedding_dim,
        margin=margin,
        loss_type=loss_type,
        dropout_rate=dropout_rate,
    )

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    model = create_model(embedding_dim=128).to(device)

    # Create dummy data
    batch_size = 4
    anchor = torch.randn(batch_size, 12, 256, 256).to(device)
    positive = torch.randn(batch_size, 12, 256, 256).to(device)
    negative = torch.randn(batch_size, 12, 256, 256).to(device)

    print("Testing model forward pass...")
    anchor_emb, pos_emb, neg_emb, loss, metrics = model(anchor, positive, negative)

    print(f"Anchor embedding shape: {anchor_emb.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test encoding
    print("\nTesting encoding...")
    embeddings = model.encode(anchor)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nModel created successfully!")
