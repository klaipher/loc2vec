# Loc2Vec: Learning Deep Representations of Location from OSM Data

A PyTorch implementation of the Loc2Vec approach for learning representations of
geographical locations from OpenStreetMap (OSM) tile data. This implementation
is based on the paper "Learning Deep Representations of Location" and uses
PyTorch 2.6.

## Overview

Loc2Vec is a deep learning model that learns to encode OSM map tile patches into
a vector space where similar locations have similar representations. This allows
for tasks like:

- Finding similar locations based on their map appearance
- Clustering locations with similar map characteristics (e.g., urban, rural,
  forest)
- Using location embeddings as features for downstream tasks

This implementation uses a two-step workflow: preparing the data by downloading
tiles from an OSM server, and then training/visualizing using the locally
prepared data.

## Requirements

- uv, [how to install](https://docs.astral.sh/uv/getting-started/installation/)
- PyTorch 2.6 or higher
- An accessible OpenStreetMap tile server (e.g., running locally or hosted)
- CUDA-capable GPU (recommended for faster training) or macOS with MPS support.

## Prepare the environment

Install python:

```bash
uv python install 3.12
```

Install dependencies:

```bash
make init
```

## Implementation Details

The implementation consists of several key components:

- **Model**: `src/model.py`: CNN-based architecture that transforms image
  patches into location embeddings.
- **OSM Data Handling**: `src/osm_data.py`: Utilities for loading OpenStreetMap
  tiles from a prepared local cache and creating training triplets.
- **OSM Data Preparation**: `src/prepare_osm_data.py`: Script to download OSM
  tiles for a region from a server and create metadata for training.
- **Training**: `src/train_osm.py`: Triplet loss-based training script using
  prepared OSM data.
- **Visualization**: `src/visualize_osm.py`: Tools to visualize embeddings from
  prepared OSM data using Folium interactive maps.
- **Main Interface**: `src/main.py`: Command-line interface to access different
  functionalities.

## Usage Workflow (Two Steps)

This implementation uses tiles fetched from an OpenStreetMap tile server (e.g.,
one running locally). It involves a preparation step followed by
training/visualization.

### Step 1: Prepare Data

Download OSM tiles for a specific region (e.g., Kyiv Oblast or Kyiv City) and
generate metadata. This command connects to your OSM server, downloads tiles
into the specified output directory (`--output_dir`), and creates a
`metadata.json` file with sample coordinates.

**Example for Kyiv Oblast (Wider Region):**

```bash
python -m src.main prepare-osm \
    --server_ip YOUR_SERVER_IP \
    --server_port 80 \
    --min_lat 49.8 --max_lat 51.5 --min_lon 29.2 --max_lon 32.2 \
    --zoom 17 \
    --num_samples 20000 \
    --output_dir ./prepared_osm_data/kyiv_oblast_z17
```

**Example for Kyiv City (Focused Region):**

```bash
python -m src.main prepare-osm \
    --server_ip YOUR_SERVER_IP \
    --server_port 80 \
    --min_lat 50.3 --max_lat 50.6 --min_lon 30.3 --max_lon 30.7 \
    --zoom 18 \
    --num_samples 15000 \
    --output_dir ./prepared_osm_data/kyiv_city_z18
```

- Replace `YOUR_SERVER_IP` with the IP address of your OSM tile server.
- Adjust `--min_lat`, `--max_lat`, `--min_lon`, `--max_lon` to define your
  desired region.
- Set `--zoom` to the desired tile zoom level (higher zoom = more detail, more
  tiles).
- `--num_samples` defines how many random anchor points to generate within the
  region.
- `--output_dir` is where the downloaded tiles (in a `tiles/` subdirectory) and
  `metadata.json` will be saved.

### Step 2a: Train using Prepared Data

Train a Loc2Vec model using the pre-downloaded tiles and metadata from the
preparation step:

```bash
python -m src.main train-osm \
    --prepared_data_dir ./prepared_osm_data/kyiv_oblast_z17 \
    --output_dir ./output_osm/kyiv_oblast_z17 \
    --epochs 50
```

- `--prepared_data_dir` points to the directory created in Step 1.
- Training uses the tiles and coordinates from the specified directory.
- Optionally provide a separate prepared directory for validation using
  `--val_data_dir`.

### Step 2b: Visualize using Prepared Data

Visualize embeddings generated from a model trained on prepared OSM data:

```bash
python -m src.main visualize-osm \
    --model_path ./output_osm/kyiv_oblast_z17/model_best.pt \
    --prepared_data_dir ./prepared_osm_data/kyiv_oblast_z17 \
    --output_dir ./visualizations_osm/kyiv_oblast_z17 \
    --max_samples 500
```

- `--model_path` points to the trained model.
- `--prepared_data_dir` points to the directory created in Step 1 (used to load
  coordinates and tile paths).
- `--max_samples` limits how many points are visualized (for performance).
- This generates an interactive HTML map (`embeddings_map.html`) and a
  `embeddings.geojson` file in the output directory.

## Training Considerations

- **Data Preparation**: The `prepare-osm` step can take significant time and
  disk space depending on the region size and zoom level. Run this step once per
  dataset configuration.
- **Zoom Level**: Higher zoom levels (e.g., 17-19) provide more detailed map
  imagery but drastically increase the number of tiles to download and store,
  and slow down training.
- **Hyperparameters**: Experiment with `embedding_dim` (32-256),
  `learning_rate`, `batch_size`, and `epochs` in the `train-osm` command.
- **Validation**: For robust evaluation, prepare a separate validation dataset
  (using `prepare-osm` for a different region or subset) and use the
  `--val_data_dir` argument during training.

## Hardware Requirements

- **GPU**: Highly recommended (CUDA or MPS). At least 8GB VRAM for reasonable
  batch sizes.
- **RAM**: 16GB+ recommended.
- **Storage**: Potentially large amounts needed for prepared OSM data (tens or
  hundreds of GB depending on region size and zoom level). SSD recommended for
  faster I/O during training.
