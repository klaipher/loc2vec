import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from loc2vec.dataset import load_image


def log_embeddings_to_tensorboard(model, dataloader, device, log_dir='logs/embeddings', max_samples=200):
    """
    Log embeddings from a model to TensorBoard for visualization.

    Args:
        model: The model from which to extract embeddings
        dataloader: DataLoader providing batches of data
        device: Device to run computations on (e.g., 'cuda' or 'cpu')
        log_dir: Directory where TensorBoard logs will be stored
        max_samples: Maximum number of samples to log
    """
    import numpy as np
    
    # Clear existing log directory
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Set model to evaluation mode
    model.eval()

    # Lists to store embeddings, images, and metadata
    embeddings_list = []
    images_list = []
    metadata = []

    # Count samples processed
    sample_count = 0
    
    # Define denormalization transform - using the same values you normalized with
    denormalize = T.Normalize(
        mean=[-0.8107/0.1215, -0.8611/0.0828, -0.7814/0.1320],
        std=[1/0.1215, 1/0.0828, 1/0.1320]
    )

    # Extract embeddings
    with torch.no_grad():
        for batch in dataloader:
            # For TilesDataset, we get a dictionary with 'anchor_image', etc.
            anchor_images = batch['anchor_image'].to(device)
            
            # Get embeddings from the model for anchor images only
            embeddings = model(anchor_images)

            # Store embeddings
            embeddings_list.append(embeddings.cpu())

            # Denormalize images for visualization to look like original
            display_images = torch.stack([denormalize(img) for img in anchor_images.cpu()])
            
            # Clamp values to valid range [0, 1]
            display_images = torch.clamp(display_images, 0, 1)
            
            # Store images for visualization
            images_list.append(display_images)

            # Store metadata: coordinates and zoom level
            batch_metadata = []
            for i in range(len(batch['x'])):
                batch_metadata.append(f"x={batch['x'][i]},y={batch['y'][i]},z={batch['zoom'][i]}")
            metadata.extend(batch_metadata)

            # Update sample count
            sample_count += anchor_images.size(0)

            # Stop if we've collected enough samples
            if sample_count >= max_samples:
                break

    # Concatenate all embeddings and images and truncate to max_samples
    all_embeddings = torch.cat(embeddings_list, dim=0)[:max_samples]
    all_images = torch.cat(images_list, dim=0)[:max_samples]

    # Log embeddings to TensorBoard
    writer.add_embedding(
        all_embeddings,
        metadata=metadata[:max_samples] if metadata else None,
        label_img=all_images,
        global_step=0
    )

    # Close the writer
    writer.close()

    print(f"Embeddings logged to TensorBoard. View with: tensorboard --logdir={log_dir}")
    print(f"Total samples: {len(all_embeddings)}")
