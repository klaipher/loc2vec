# Visualizer Module

This module provides a tool for generating ResNet-18 embeddings for a folder of images and visualizing them in 2-D or 3-D using Plotly and Dash, with interactive tooltips for each image.

## Features
- Loads images from a specified folder and computes embeddings using a pretrained ResNet-18 model.
- Supports dimensionality reduction using PCA or t-SNE (2D or 3D visualization).
- Interactive visualization with Plotly and Dash, including image tooltips on hover.
- Customizable batch size, number of images, and device (CPU, CUDA, or MPS).

## Usage
Run the script from the command line:

```bash
python visualizer.py --image_folder <path_to_images> [--reduction_method pca|tsne] [-k 2|3] [--batch_size N] [--max_images N] [--no_cuda]
```

### Examples
- 2D PCA visualization:
  ```bash
  python visualizer.py --image_folder imgs --reduction_method pca -k 2
  ```
- 3D t-SNE visualization:
  ```bash
  python visualizer.py --image_folder imgs --reduction_method tsne -k 3
  ```

## Arguments
- `--image_folder`: Path to the folder containing images (required).
- `--reduction_method`: Dimensionality reduction method (`pca` or `tsne`). Default: `pca`.
- `-k`, `--n_components`: Number of components for visualization (2 or 3). Default: 3.
- `--batch_size`: Batch size for embedding computation. Default: 256.
- `--max_images`: Maximum number of images to process.
- `--no_cuda`: Force CPU even if CUDA/MPS is available.

## Output
A Dash web app will open in your browser at http://127.0.0.1:8050 with the interactive embedding visualization.
