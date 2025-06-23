import os
import shutil

from numpy.random import f
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.xpu import max_memory_allocated
import torchvision.transforms as T

from loc2vec.dataset import load_image


def log_embeddings_to_tensorboard(model, dataloader, device, log_dir='logs/embeddings', max_samples=200):
    import numpy as np
    
    # Clear existing log directory
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    model.eval()

    embeddings_list = []
    images_list = []
    metadata = []
    sample_count = 0
    
    denormalize = T.Normalize(
        mean=[-0.8107/0.1215, -0.8611/0.0828, -0.7814/0.1320],
        std=[1/0.1215, 1/0.0828, 1/0.1320]
    )

    print(f"Starting embedding extraction, max_samples={max_samples}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            anchor_images = batch['anchor_image'].to(device)
            
            embeddings = model(anchor_images)
            
            embeddings_list.append(embeddings.cpu())

            # print(anchor_images.shape, embeddings.shape)

            resize = T.Resize((64, 64))

            anchor_images = resize(anchor_images[:, :3, :, :])  # Ensure only RGB channels are used
            
            display_images = torch.stack([denormalize(img) for img in anchor_images.cpu()])
            display_images = torch.clamp(display_images, 0, 1)
            images_list.append(display_images)
            
            batch_metadata = []
            for i in range(len(batch['x'])):
                batch_metadata.append(f"x={batch['x'][i]},y={batch['y'][i]},z={batch['zoom'][i]}")
            
            metadata.extend(batch_metadata)
            
            sample_count += anchor_images.size(0)
            
            if sample_count >= max_samples:
                break

    all_embeddings = torch.cat(embeddings_list, dim=0)
    all_images = torch.cat(images_list, dim=0)
    all_embeddings = all_embeddings[:max_samples]
    all_images = all_images[:max_samples]
    metadata_truncated = metadata[:max_samples]

    writer.add_embedding(
        all_embeddings,
        metadata=metadata_truncated,
        label_img=all_images,
        global_step=0
    )

    writer.close()
    print(f"Embeddings logged successfully!")