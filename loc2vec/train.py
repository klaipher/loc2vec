import random

import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from loc2vec.dataset import TilesDataset
from loc2vec.model import Loc2VecModel


def train(model, train_loader, optimizer, loss_fn, device, scheduler=None):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The Pytorch model instance
        train_loader (DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        loss_fn (nn.Module): Loss function to compute the loss
        device (torch.device): Device to run the training on

    Returns:
        float: Average loss for the epoch
    """
    model.train()

    loss_each = 2000 # epochs
    loss_history = []
    total_loss = 0.0

    interm_loss = 0.0

    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        anchor = batch["anchor_image"].to(device)
        positive = batch["pos_image"].to(device)
        negative = batch["neg_image"].to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = loss_fn(anchor_out, positive_out, negative_out)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        interm_loss += loss.item()

        if batch_idx % loss_each == 0:
            # tqdm.write(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            loss_history.append(loss.item() / loss_each)
            interm_loss = 0.0
            

    return total_loss / len(train_loader), loss_history

def evaluate(model, val_loader, loss_fn, device):
    """
    Evaluate the model on validation data.

    Args:
        model (nn.Module): The Pytorch model instance
        val_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to run the evaluation on

    Returns:
        float: Average loss for the validation set
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", total=len(val_loader)):
            anchor = batch["anchor_image"].to(device)
            positive = batch["pos_image"].to(device)
            negative = batch["neg_image"].to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = loss_fn(anchor_out, positive_out, negative_out)
            total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Loc2VecModel(input_channels=3, embedding_dim=64, dropout_rate=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = TripletLoc2Vec(embedding_dim=64, margin=0.3)
    loss_fn = nn.TripletMarginLoss(margin=0.3, p=2)
    dataset = TilesDataset(
        "../tiles/full",
        pos_radius=1,
        transform=T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],  # image normalization values for ImageNet
                    [0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    sample = random.choice(dataset)
    print(f"Input shape: {sample['anchor_image'].shape}")

    print(f"Training on device: {device}")

    model.to(device)

    # model = torch.compile(model)

    for epoch in range(10):
        # Assuming train_loader is defined and provides the training data
        avg_loss = train(model, train_loader, optimizer, loss_fn, device=device)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
