# Loc2Vec: Location Embeddings for Kyiv City

This project implements the **loc2vec** paper using PyTorch 2.6, creating
location embeddings from 12-layer map tiles of Kyiv city. The implementation
uses a CNN encoder with triplet loss to learn spatial representations that
capture geographic and semantic relationships.

## 🗺️ Overview

The system processes 12 different map layers (buildings, roads, water, etc.) to
create location embeddings that can be visualized and analyzed to understand
spatial patterns in Kyiv.

**Key Features:**

- **12-layer input**: natural-features, boundaries, landcover, buildings,
  amenities, railways, aeroways, leisure, places, water, roads, power
- **Triplet loss training**: with spatial distance-based positive/negative
  sampling
- **Interactive visualizations**: TensorFlow Projector-like interface with
  t-SNE, PCA, and UMAP
- **PyTorch 2.6**: Modern implementation with proper GPU support

## 📋 Requirements

- Python 3.12+
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- CUDA-capable GPU (recommended for training)

## 🚀 Quick Start

### Option 1: Automated Quick Start (Recommended)

Use the provided script for an automated end-to-end experience:

```bash
# Complete pipeline: test data → train → visualize
python run_example.py --epochs 20

# Just test data loading
python run_example.py --test-only

# Train only (20 epochs, good for testing)
python run_example.py --train-only --epochs 20

# Create visualizations from existing model
python run_example.py --visualize-only --max-tiles 3000
```

The script will:

1. ✅ Check your tile data structure
2. 🧪 Test data loading
3. 🏋️ Train the model with reasonable defaults
4. 🎨 Generate interactive visualizations
5. 📊 Show you where to find the results

### Option 2: Manual Step-by-Step

### 1. Environment Setup

```bash
# Install Python 3.12
uv python install 3.12

# Initialize the environment
make init

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 2. Data Structure

Ensure your tile data is organized as follows:

```
12_layer_tiles/
└── tiles/
    ├── natural-features/
    │   └── 17/
    │       └── {x}/
    │           └── {y}.png
    ├── boundaries/
    │   └── 17/
    │       └── {x}/
    │           └── {y}.png
    ├── ... (other layers)
    └── power/
        └── 17/
            └── {x}/
                └── {y}.png
```

**Note**: All 12 layers must have tiles for the same coordinates for training to
work properly.

### 3. Test Data Loading

```bash
# Test if the data loader works correctly
python data_loader.py
```

This will show you:

- Number of layers found
- Number of tiles per layer
- Common coordinates across all layers
- Sample batch shapes

### 4. Train the Model

#### Basic Training

```bash
python train.py --epochs 50 --batch_size 16
```

#### Advanced Training with Custom Parameters

```bash
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --embedding_dim 256 \
  --margin 0.3 \
  --dropout 0.3 \
  --pos_radius 3 \
  --neg_radius 15 \
  --max_samples 20000 \
  --loss_type softpn \
  --save_every 5
```

#### Resume Training from Checkpoint

```bash
python train.py --resume checkpoints/best_model.pth --epochs 25
```

### 5. Monitor Training

Training logs are saved to TensorBoard. Open in another terminal:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 to monitor:

- Training/validation loss
- Positive/negative distances
- Triplet usage ratio
- Learning rate changes

### 6. Create Interactive Visualizations

We provide two powerful ways to explore your trained embeddings with interactive
visualizations:

#### Option 1: Live Streamlit App (Recommended)

Real-time exploration with live controls:

```bash
# Launch the interactive Streamlit app
python run_interactive_viz.py streamlit

# With custom model and tiles
python run_interactive_viz.py streamlit \
  --model_path checkpoints/best_model.pth \
  --tiles_root 12_layer_tiles/tiles
```

**Features:**

- 🎛️ **Real-time algorithm switching** (PCA, t-SNE, UMAP)
- 🔄 **Live 2D/3D toggle** with smooth rotation
- 🎨 **Dynamic coloring** (layer, zone, content density)
- 🖼️ **Tile previews** on hover
- 📊 **Progress tracking** during computation
- 📈 **Statistics dashboard**

#### Option 2: Static HTML Dashboard

Generate shareable visualizations:

```bash
# Create interactive HTML files
python run_interactive_viz.py html --max_tiles 500

# With custom output directory
python run_interactive_viz.py html \
  --model_path checkpoints/best_model.pth \
  --max_tiles 300 \
  --output_dir my_visualizations
```

**Features:**

- 🌐 **Self-contained HTML files** (no server needed)
- 🔧 **Master dashboard** with algorithm switching
- 🖼️ **Embedded tile previews**
- 📱 **Mobile-friendly** responsive design
- 🔗 **Easy sharing** via file links

#### Legacy Static Visualizations

For comparison with original implementation:

```bash
# Basic visualization (original method)
python visualize.py --model_path checkpoints/best_model.pth

# Custom visualization with fewer tiles (faster)
python visualize.py \
  --model_path checkpoints/best_model.pth \
  --max_tiles 5000 \
  --output_dir my_visualizations \
  --save_embeddings
```

## 🎮 Interactive Features

### 🔍 Algorithm Comparison

- **PCA**: Fast, linear, shows global variance structure
- **t-SNE**: Excellent clustering, preserves local neighborhoods
- **UMAP**: Balanced approach, preserves local and global structure

### 🌐 3D Exploration

- **Mouse controls**: Drag to rotate, scroll to zoom
- **Smooth transitions**: Real-time algorithm switching
- **Depth perception**: Better cluster separation visualization

### 🖼️ Tile Previews

- **Hover to see tiles**: Actual map tile images appear on hover
- **Smart layer selection**: Shows most representative layer per tile
- **Fast rendering**: Optimized 64x64 previews for performance

### 🎨 Smart Coloring

- **Dominant Layer**: Color by most prominent map feature
- **Geographic Zone**: Color by spatial regions (North/South/East/West)
- **Content Density**: Color by amount of map information

## 🔍 Fast Similarity Search with Annoy

**NEW**: We've added blazing-fast similarity search using
[Spotify's Annoy library](https://github.com/spotify/annoy) for approximate
nearest neighbor search. This allows you to find similar map tiles in
milliseconds rather than seconds!

### 🚀 Pre-computing for Speed

For the best experience, pre-compute embeddings and indices before using the
Streamlit app:

```bash
# Pre-compute everything (recommended for first use)
python precompute_embeddings.py

# Pre-compute with custom parameters
python precompute_embeddings.py \
  --max_tiles 10000 \
  --annoy_trees 200 \
  --annoy_metrics angular euclidean \
  --dr_algorithms PCA UMAP TSNE

# Just update Annoy indices with more trees
python precompute_embeddings.py \
  --skip_embeddings \
  --skip_projections \
  --annoy_trees 500 \
  --force_recompute
```

**What this does:**

- 🧠 **Computes embeddings** for all tiles (cached for reuse)
- 🌲 **Builds Annoy indices** with different distance metrics
- 📐 **Pre-computes PCA, t-SNE, UMAP** projections
- 💾 **Caches everything** for instant loading in Streamlit

### 🎯 Using Similarity Search

In the Streamlit app, select **"🔍 Similarity Search (Annoy)"** mode:

#### Query Selection Methods

1. **🎲 Random Tile**: Explore random tiles for interesting patterns
2. **📍 Coordinate Input**: Search for specific (x, y) coordinates
3. **🖼️ Browse Gallery**: Visual tile selection from preview grid

#### Search Parameters

- **Distance Metric**: Angular, Euclidean, Manhattan, or Dot product
- **Number of Trees**: More trees = better accuracy (100-500 recommended)
- **Search Quality**: Higher values = better results but slower search
- **Number of Results**: How many similar tiles to find (5-50)

#### Advanced Features

- **📊 Real-time visualization** of results in embedding space
- **🖼️ Side-by-side tile comparison** with distance metrics
- **🔍 Detailed tile analysis** for each result
- **📈 Interactive scatter plots** highlighting query and results

### 🎛️ Distance Metrics Explained

| Metric          | Best For              | Characteristics                              |
| --------------- | --------------------- | -------------------------------------------- |
| **Angular**     | Semantic similarity   | Measures angle between vectors (cosine-like) |
| **Euclidean**   | Spatial similarity    | Traditional distance in embedding space      |
| **Manhattan**   | Feature similarity    | Sum of absolute differences                  |
| **Dot Product** | Magnitude + direction | Good for normalized embeddings               |

### 🚄 Performance Comparison

| Method                | Search Time  | Accuracy | Best Use Case                |
| --------------------- | ------------ | -------- | ---------------------------- |
| **Brute Force**       | ~1-5 seconds | 100%     | Small datasets (<1K tiles)   |
| **Annoy (50 trees)**  | ~1-10ms      | ~95%     | Interactive exploration      |
| **Annoy (200 trees)** | ~5-20ms      | ~99%     | Production similarity search |

### 💡 Tips for Best Results

1. **Start with Angular distance** - works well for most map data
2. **Use 100-200 trees** for good speed/accuracy balance
3. **Pre-compute indices** for multiple metrics to compare results
4. **Increase search quality** for more precise results when needed
5. **Try different algorithms** (PCA vs UMAP) for visualization

### 🔄 Workflow Integration

**Recommended workflow:**

1. **Pre-compute** embeddings and indices: `python precompute_embeddings.py`
2. **Launch Streamlit**: `streamlit run streamlit_app.py`
3. **Select Similarity Search** mode
4. **Pick a query tile** using any method
5. **Explore results** and adjust parameters
6. **Switch to regular projector** mode for global overview

## 📊 Training Parameters

| Parameter         | Default  | Description                       |
| ----------------- | -------- | --------------------------------- |
| `--epochs`        | 50       | Number of training epochs         |
| `--batch_size`    | 16       | Batch size                        |
| `--lr`            | 0.001    | Learning rate                     |
| `--embedding_dim` | 128      | Embedding dimension               |
| `--margin`        | 0.2      | Triplet loss margin               |
| `--dropout`       | 0.5      | Dropout rate                      |
| `--pos_radius`    | 2        | Positive sample radius (tiles)    |
| `--neg_radius`    | 10       | Negative sample minimum radius    |
| `--max_samples`   | 10000    | Samples per epoch                 |
| `--loss_type`     | 'softpn' | Loss type ('triplet' or 'softpn') |

## 🔍 Understanding the Outputs

### Visualization Modes Explained

#### 🧮 PCA (Principal Component Analysis)

- **Best for**: Initial data exploration, understanding variance
- **Characteristics**: Linear transformation, fast computation
- **Use when**: You want to see the main axes of variation

#### 🌀 t-SNE (t-Distributed Stochastic Neighbor Embedding)

- **Best for**: Discovering clusters and local structure
- **Characteristics**: Non-linear, emphasizes local neighborhoods
- **Use when**: You want to find grouped similar tiles

#### 🗺️ UMAP (Uniform Manifold Approximation)

- **Best for**: Balanced local and global structure preservation
- **Characteristics**: Fast, theoretically grounded, good for large datasets
- **Use when**: You want the best of both worlds

### Interactive Navigation Tips

1. **Start with t-SNE 3D** for initial exploration
2. **Switch to UMAP** for better global structure
3. **Use PCA** to understand main variation axes
4. **Toggle 2D/3D** to compare perspectives
5. **Color by layer** to see functional patterns
6. **Color by zone** to see geographic patterns

### What to Look For

- **Tight clusters**: Similar map regions (residential, commercial)
- **Smooth transitions**: Gradual change between related areas
- **Clear separation**: Different functional zones well separated
- **Geographic coherence**: Nearby tiles should be near in embedding space

## 🏗️ Architecture Details

### Model Architecture

```
Input: (batch_size, 12, 256, 256)
├── Conv Block 1: 12 → 64 channels
├── Conv Block 2: 64 → 128 channels
├── Conv Block 3: 128 → 256 channels
├── Conv Block 4: 256 → 512 channels
├── Conv Block 5: 512 → 512 channels
├── FC Layer 1: 32768 → 1024
├── FC Layer 2: 1024 → 512
└── Embedding: 512 → embedding_dim
```

### Triplet Loss

- **Standard Triplet**: `max(0, d(a,p) - d(a,n) + margin)`
- **SoftPN Triplet**: Considers positive-negative distance for better mining

## 🛠️ Troubleshooting

### Visualization Issues

**1. Streamlit App Won't Start**

```bash
# Install missing dependencies
pip install streamlit plotly-express umap-learn kaleido

# Check if port is available
python run_interactive_viz.py streamlit
```

**2. HTML Files Too Large**

```bash
# Reduce number of tiles
python run_interactive_viz.py html --max_tiles 200

# Use lower resolution if needed
```

**3. Slow 3D Rendering**

- Reduce `--max_tiles` parameter
- Use 2D mode for faster interaction
- Close other browser tabs to free memory

**4. Missing Tile Previews**

- Check tile directory structure
- Verify PNG files exist and are readable
- Some tiles may show gray placeholders if corrupted

**5. Similarity Search Issues**

```bash
# If Annoy import fails
pip install annoy

# If pre-computation is slow
python precompute_embeddings.py --max_tiles 1000

# If index loading fails
python precompute_embeddings.py --force_recompute
```

**6. Slow Similarity Search**

- Use pre-computed indices: `python precompute_embeddings.py`
- Reduce number of trees for faster (but less accurate) search
- Lower search quality parameter for speed

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size
python train.py --batch_size 8

# Or reduce tile size
python train.py --tile_size 128
```

**2. No Common Coordinates Found**

- Check that all 12 layers have tiles for overlapping areas
- Verify tile directory structure matches expected format

**3. Low Triplet Usage Ratio**

- Increase `--pos_radius` and `--neg_radius`
- Adjust `--margin` parameter
- Use `--loss_type softpn` for better hard mining

**4. Slow Training**

- Increase `--num_workers` for faster data loading
- Use smaller `--max_samples` per epoch
- Consider reducing `--tile_size`

### Performance Tips

- **GPU Memory**: Start with smaller batch sizes and increase gradually
- **Data Loading**: Use more workers if you have multiple CPU cores
- **Convergence**: Monitor the triplet usage ratio - should be 20-80%
- **Embeddings**: Higher dimensions (256, 512) may work better for complex areas

## 📁 Project Structure

```
├── data_loader.py           # Dataset and data loading logic
├── model.py                # CNN encoder and triplet loss
├── train.py               # Training script with TensorBoard logging
├── visualize.py           # Legacy visualization generation
├── interactive_visualizer.py # Advanced HTML visualizations
├── streamlit_app.py       # Live Streamlit app
├── run_interactive_viz.py # Easy runner for both modes
├── requirements.txt       # Python dependencies
├── Makefile              # Setup commands
└── 12_layer_tiles/       # Tile data directory
    └── tiles/            # Individual layer directories
```

## 🎯 Expected Results

After successful training, you should see:

- **Loss decreasing** over epochs (typically 0.1-0.5 range)
- **Positive distances < Negative distances** (good separation)
- **Meaningful clusters** in visualizations corresponding to:
  - Geographic regions (city center, suburbs, outskirts)
  - Functional zones (residential, commercial, industrial)
  - Infrastructure patterns (transportation hubs, parks)

### Visualization Quality Indicators

**Good Embeddings Show:**

- 🎯 **Clear clustering** of similar map regions
- 🌊 **Smooth transitions** between related areas
- 🗺️ **Geographic coherence** (nearby tiles cluster together)
- 🏗️ **Functional separation** (residential vs commercial distinct)

**Poor Embeddings Show:**

- ❌ **Random scattered points** with no clear structure
- ❌ **Disconnected similar regions**
- ❌ **Poor geographic coherence**
- ❌ **Mixed functional zones**

## 📚 References

- [Original loc2vec paper](https://www.sentiance.com/blog/sentiance-research-paper-location-embeddings-nlp-techniques-geospatial-world/)
  by Sentiance
- [PyTorch Triplet Loss Implementation](https://omoindrot.github.io/triplet-loss)
- [UMAP: Uniform Manifold Approximation](https://umap-learn.readthedocs.io/)
