# Loc2Vec: Learning Location Representations from Map Data

This project helps you teach AI to understand geographic locations by looking at
map images. Think of it like training a computer to recognize whether two places
look similar - like both being cities, forests, or residential areas.

## What does this do?

Loc2Vec takes satellite map images from OpenStreetMap and converts them into
numbers that capture what each place "looks like." Once trained, the AI can find
similar places, group locations by their characteristics, or help with other
location-based tasks.

For example, it might learn that downtown areas look similar to each other, even
in different cities, or that forest areas have distinctive patterns.

## What you need

You need Python 3.12 and a computer with good graphics (GPU recommended). The
project works on Windows, Mac, and Linux.

## Getting started

First, install the Python package manager called `uv`:

```bash
# Visit https://docs.astral.sh/uv/getting-started/installation/ for instructions
```

Then set up the project:

```bash
uv python install 3.12
make init
```

## How to use it

The process has two main steps: get the map data, then train the AI.

### Step 1: Download map images

You need to download map tiles from an OpenStreetMap server. This command grabs
thousands of map images from a specific area:

```bash
python -m src.main prepare-osm \
    --server_ip YOUR_SERVER_IP \
    --server_port 80 \
    --min_lat 50.3 --max_lat 50.6 --min_lon 30.3 --max_lon 30.7 \
    --zoom 18 \
    --num_samples 15000 \
    --output_dir ./map_data/kyiv
```

Replace `YOUR_SERVER_IP` with your map server's address. The coordinates shown
here cover Kyiv city - change them to your area of interest.

### Step 2: Train the AI

Now train the model using your downloaded map images:

```bash
python -m src.main train-osm \
    --prepared_data_dir ./map_data/kyiv \
    --output_dir ./trained_models/kyiv \
    --epochs 50
```

This teaches the AI to recognize patterns in your map images. It might take a
while depending on your computer.

### Step 3: See the results

Create an interactive map showing how the AI groups similar locations:

```bash
python -m src.main visualize-osm \
    --model_path ./trained_models/kyiv/model_best.pt \
    --prepared_data_dir ./map_data/kyiv \
    --output_dir ./results/kyiv \
    --max_samples 500
```

This creates an HTML file you can open in your browser to explore the results.

## Important notes

**Storage space**: Map data can take up a lot of disk space (several GB),
especially for large areas or high zoom levels.

**Training time**: Training can take hours or days depending on your computer
and the amount of data.

**Memory**: You'll need at least 8GB of RAM, preferably 16GB or more.

**Graphics card**: A GPU makes training much faster. The project supports NVIDIA
CUDA and Apple Metal.

## Project structure

The main code lives in two folders:

**src/**: The main application with commands for downloading data, training, and
visualization.

**loc2vec/**: Core machine learning components including the neural network
model and data handling.

## Alternative training method

There's also a Jupyter notebook approach in the `loc2vec/` folder if you prefer
working with notebooks instead of command-line tools.

## Development and experimentation

We used several additional tools during development:

**Transfer learning experiments**: `loc2vec_kaggle.ipynb` contains comprehensive
comparisons of different transfer learning approaches (EfficientNet, ResNet,
MobileNet, etc.) that we ran on Kaggle's free GPUs.

**Model evaluation**: `visualize_embeddings_tensorboard.py` helps you analyze
trained models by building embeddings and evaluating their quality using various
metrics and TensorBoard visualization.

## Need help?

Check the `saved_models/` folder for examples of trained models, or look at the
Jupyter notebooks for interactive examples of how everything works.
