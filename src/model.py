import torch.nn as nn
import torch.nn.functional as F


class Loc2Vec(nn.Module):
    """
    Loc2Vec model implementation based on the paper "Learning Deep Representations of Location".

    This model takes satellite imagery patches as input and encodes them into a vector space
    where similar locations have similar representations.
    """

    def __init__(self, input_channels=3, embedding_dim=64):
        """
        Initialize the Loc2Vec model.

        Args:
            input_channels (int): Number of input channels in the imagery (3 for RGB)
            embedding_dim (int): Dimension of the output location embedding
        """
        super(Loc2Vec, self).__init__()

        # Encoder network (CNN)
        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate output size after convolutional layers
        # Assuming input size of 64x64
        self.feature_size = 256 * 4 * 4

        # Fully connected layers for embedding
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Location embedding of shape [batch_size, embedding_dim]
        """
        # Pass through convolutional encoder
        x = self.encoder(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers to get embedding
        x = self.fc(x)

        # Normalize embedding to unit length
        x = F.normalize(x, p=2, dim=1)

        return x


class TripletLoc2Vec(nn.Module):
    """
    Loc2Vec model with triplet loss for training.

    This model takes triplets of images (anchor, positive, negative) and computes
    embeddings that bring the anchor closer to the positive and farther from the negative.
    """

    def __init__(self, input_channels=3, embedding_dim=64, margin=0.3):
        """
        Initialize the TripletLoc2Vec model.

        Args:
            input_channels (int): Number of input channels in the imagery
            embedding_dim (int): Dimension of the output location embedding
            margin (float): Margin for triplet loss
        """
        super(TripletLoc2Vec, self).__init__()

        # Base Loc2Vec model
        self.loc2vec = Loc2Vec(input_channels, embedding_dim)

        # Triplet loss with margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet-based training.

        Args:
            anchor (torch.Tensor): Anchor image
            positive (torch.Tensor): Positive image (similar location)
            negative (torch.Tensor): Negative image (dissimilar location)

        Returns:
            tuple: Tuple containing (embeddings, loss)
                - embeddings: Tuple of (anchor_emb, positive_emb, negative_emb)
                - loss: Triplet loss value
        """
        # Compute embeddings
        anchor_emb = self.loc2vec(anchor)
        positive_emb = self.loc2vec(positive)
        negative_emb = self.loc2vec(negative)

        # Compute triplet loss
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

        return (anchor_emb, positive_emb, negative_emb), loss
