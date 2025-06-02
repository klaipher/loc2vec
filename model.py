import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18Encoder(nn.Module):
    """
    ResNet18-based encoder for location tiles with pre-trained weights.

    Uses a pre-trained ResNet18 backbone and adapts it for 12-channel input
    and the desired embedding dimension.
    """

    def __init__(
        self,
        input_channels: int = 12,
        embedding_dim: int = 128,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
    ):
        """
        Args:
            input_channels: Number of input channels (12 for the map layers)
            embedding_dim: Dimension of the output embedding
            dropout_rate: Dropout rate for regularization
            pretrained: Whether to use pre-trained weights
        """
        super(ResNet18Encoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Load ResNet18 with updated weights parameter
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Modify first conv layer to handle 12 channels instead of 3
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Initialize new conv layer weights intelligently
            with torch.no_grad():
                if input_channels >= 3:
                    # For the first 3 channels, use pre-trained weights
                    self.backbone.conv1.weight[:, :3, :, :] = (
                        original_conv1.weight.clone()
                    )

                    # For additional channels, initialize with mean of RGB channels
                    if input_channels > 3:
                        mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                        for i in range(3, input_channels):
                            self.backbone.conv1.weight[:, i : i + 1, :, :] = (
                                mean_weights.clone()
                            )
                else:
                    # For fewer than 3 channels, use mean or subset of pre-trained weights
                    if input_channels == 1:
                        # Use mean of RGB channels for single channel
                        mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
                        self.backbone.conv1.weight[:, 0:1, :, :] = mean_weights.clone()
                    elif input_channels == 2:
                        # Use first two channels of pre-trained weights
                        self.backbone.conv1.weight[:, :2, :, :] = original_conv1.weight[
                            :, :2, :, :
                        ].clone()

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Add custom head for embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(512, 512),  # ResNet18 outputs 512 features
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, 12, H, W)

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Pass through ResNet18 backbone
        x = self.backbone(x)

        # Pass through embedding head
        x = self.embedding_head(x)

        # L2 normalize the embeddings for better triplet loss performance
        x = F.normalize(x, p=2, dim=1)

        return x


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
        encoder_type: str = "cnn",
        pretrained: bool = True,
    ):
        """
        Args:
            input_channels: Number of input channels
            embedding_dim: Dimension of embeddings
            margin: Triplet loss margin
            loss_type: Type of triplet loss ('triplet' or 'softpn')
            dropout_rate: Dropout rate
            encoder_type: Type of encoder ('cnn' or 'resnet18')
            pretrained: Whether to use pre-trained weights (only for resnet18)
        """
        super(Loc2VecModel, self).__init__()

        if encoder_type == "resnet18":
            self.encoder = ResNet18Encoder(
                input_channels=input_channels,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                pretrained=pretrained,
            )
        elif encoder_type == "cnn":
            self.encoder = Loc2VecEncoder(
                input_channels=input_channels,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. Choose 'cnn' or 'resnet18'"
            )

        self.triplet_loss = TripletLoss(
            margin=margin, loss_type=loss_type, hard_mining=True
        )

        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type

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
    input_channels: int = 12,
    embedding_dim: int = 128,
    margin: float = 0.2,
    loss_type: str = "softpn",
    dropout_rate: float = 0.5,
    encoder_type: str = "cnn",
    pretrained: bool = True,
) -> Loc2VecModel:
    """
    Create a loc2vec model with specified parameters.

    Args:
        input_channels: Number of input channels (depends on data loading mode)
        embedding_dim: Dimension of location embeddings
        margin: Triplet loss margin
        loss_type: Type of triplet loss
        dropout_rate: Dropout rate for regularization
        encoder_type: Type of encoder ('cnn' or 'resnet18')
        pretrained: Whether to use pre-trained weights (only for resnet18)

    Returns:
        Initialized Loc2VecModel
    """
    model = Loc2VecModel(
        input_channels=input_channels,
        embedding_dim=embedding_dim,
        margin=margin,
        loss_type=loss_type,
        dropout_rate=dropout_rate,
        encoder_type=encoder_type,
        pretrained=pretrained,
    )

    # Initialize weights for custom CNN encoder
    if encoder_type == "cnn":

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model.apply(init_weights)
    elif encoder_type == "resnet18":
        # Only initialize the embedding head for ResNet18 (backbone is pre-trained)
        def init_embedding_head(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model.encoder.embedding_head.apply(init_embedding_head)

    return model


if __name__ == "__main__":
    # Test both encoder types
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # Test CNN encoder
    print("\n=== Testing CNN Encoder ===")
    model_cnn = create_model(embedding_dim=128, encoder_type="cnn").to(device)
    cnn_params = sum(p.numel() for p in model_cnn.parameters())
    print(f"CNN Model parameters: {cnn_params:,}")

    # Test ResNet18 encoder
    print("\n=== Testing ResNet18 Encoder ===")
    model_resnet = create_model(
        embedding_dim=128, encoder_type="resnet18", pretrained=True
    ).to(device)
    resnet_params = sum(p.numel() for p in model_resnet.parameters())
    print(f"ResNet18 Model parameters: {resnet_params:,}")

    # Create dummy data
    batch_size = 4
    anchor = torch.randn(batch_size, 12, 256, 256).to(device)
    positive = torch.randn(batch_size, 12, 256, 256).to(device)
    negative = torch.randn(batch_size, 12, 256, 256).to(device)

    # Test CNN model
    print("\nTesting CNN model forward pass...")
    anchor_emb, pos_emb, neg_emb, loss, metrics = model_cnn(anchor, positive, negative)
    print(f"CNN - Anchor embedding shape: {anchor_emb.shape}")
    print(f"CNN - Loss: {loss.item():.4f}")
    print(f"CNN - Metrics: {metrics}")

    # Test ResNet18 model
    print("\nTesting ResNet18 model forward pass...")
    anchor_emb, pos_emb, neg_emb, loss, metrics = model_resnet(
        anchor, positive, negative
    )
    print(f"ResNet18 - Anchor embedding shape: {anchor_emb.shape}")
    print(f"ResNet18 - Loss: {loss.item():.4f}")
    print(f"ResNet18 - Metrics: {metrics}")

    # Test encoding
    print("\nTesting encoding...")
    embeddings_cnn = model_cnn.encode(anchor)
    embeddings_resnet = model_resnet.encode(anchor)
    print(f"CNN Embeddings shape: {embeddings_cnn.shape}")
    print(f"ResNet18 Embeddings shape: {embeddings_resnet.shape}")

    print("\nParameter comparison:")
    print(f"CNN Model: {cnn_params:,} parameters")
    print(f"ResNet18 Model: {resnet_params:,} parameters")
    print(f"Difference: {resnet_params - cnn_params:,} parameters")

    print("\nBoth models created successfully!")
