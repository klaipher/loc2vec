import streamlit as st
import torch
import numpy as np
import plotly.express as px
import pandas as pd
import os
from PIL import Image
import base64
from io import BytesIO
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from threading import Lock
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "1"  # Suppress OpenMP warnings

from model import create_model
from data_loader import KyivTileDataset
from embedding_processor import EmbeddingProcessor

# Set page config
st.set_page_config(
    page_title="Loc2Vec TensorFlow Projector",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global cache for thread safety
cache_lock = Lock()


@st.cache_resource
def load_model_and_data(model_path, tiles_root):
    """Load model and dataset (cached)."""
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model = create_model(embedding_dim=128)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    # Load dataset
    dataset = KyivTileDataset(
        tiles_root=tiles_root,
        tile_size=256,
        max_samples_per_epoch=None,  # Load all available samples
    )

    return model, dataset, device


def load_layer_png(coord, layer, tiles_root):
    """Load a specific layer PNG file."""
    zoom, x, y = coord
    tile_path = os.path.join(tiles_root, layer, str(zoom), str(x), f"{y}.png")

    if os.path.exists(tile_path):
        try:
            img = Image.open(tile_path).convert("RGB")
            return img
        except:
            return None
    return None


def create_composite_image(coord, layers, tiles_root, size=256):
    """Create a composite image from multiple layers."""
    zoom, x, y = coord

    # Priority layers for better visualization
    priority_layers = ["buildings", "roads", "natural-features", "water", "land-use"]

    # Start with a white background
    base_img = Image.new("RGB", (size, size), color="white")

    # Add layers in priority order
    for layer in priority_layers:
        if layer in layers:
            layer_img = load_layer_png(coord, layer, tiles_root)
            if layer_img:
                layer_img = layer_img.resize((size, size), Image.Resampling.LANCZOS)

                # Blend with existing image
                if layer_img.mode == "RGBA":
                    base_img = Image.alpha_composite(
                        base_img.convert("RGBA"), layer_img
                    ).convert("RGB")
                else:
                    # Create a mask based on non-white pixels
                    layer_array = np.array(layer_img)
                    mask = np.any(layer_array < 240, axis=2)  # Non-white pixels

                    if np.any(mask):
                        base_array = np.array(base_img)
                        layer_array = np.array(layer_img)

                        # Blend where mask is True
                        blended = base_array.copy()
                        blended[mask] = (
                            0.7 * base_array[mask] + 0.3 * layer_array[mask]
                        ).astype(np.uint8)
                        base_img = Image.fromarray(blended)

    return base_img


def create_tile_image_base64(coord, layers, tiles_root, size=32):
    """Create a small tile image encoded as base64."""
    composite = create_composite_image(coord, layers, tiles_root, size=size)

    # Convert to base64
    buffer = BytesIO()
    composite.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def process_tile_batch(coords_batch, layers, tiles_root, dataset):
    """Process a batch of tiles for embeddings and metadata."""
    batch_data = []

    for coord in coords_batch:
        try:
            # Load tile stack
            tile_stack = dataset._load_tile_stack(coord)

            # Analyze content
            meta = analyze_tile_content(coord, layers, tiles_root)

            # Create tile image
            tile_image = create_tile_image_base64(coord, layers, tiles_root, size=32)

            batch_data.append(
                {
                    "coord": coord,
                    "tile_stack": tile_stack,
                    "metadata": meta,
                    "tile_image": tile_image,
                }
            )
        except Exception as e:
            print(f"Error processing tile {coord}: {e}")
            continue

    return batch_data


@st.cache_data
def generate_all_embeddings(_model, _dataset, _device, tiles_root, max_tiles=None):
    """Generate embeddings for all tiles with multithreading."""
    coords = _dataset.common_coords.copy()

    if max_tiles and max_tiles < len(coords):
        # Sample randomly to maintain diversity
        coords = random.sample(coords, max_tiles)

    st.info(f"Processing {len(coords)} tiles...")

    # Process in batches with multithreading
    batch_size = 64  # Larger batches for efficiency
    all_embeddings = []
    all_coordinates = []
    all_metadata = []
    all_tile_images = []

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_batches = len(coords) // batch_size + (
        1 if len(coords) % batch_size > 0 else 0
    )

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]

        # Process batch data
        batch_data = process_tile_batch(
            batch_coords, _dataset.layers, tiles_root, _dataset
        )

        if not batch_data:
            continue

        # Extract tile stacks for embedding generation
        batch_tiles = [item["tile_stack"] for item in batch_data]
        batch_tensor = torch.from_numpy(np.stack(batch_tiles, axis=0)).to(_device)

        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = _model.encode(batch_tensor)
            embeddings_np = batch_embeddings.cpu().numpy()

        # Store results
        all_embeddings.append(embeddings_np)

        for j, item in enumerate(batch_data):
            all_coordinates.append(item["coord"])
            all_metadata.append(item["metadata"])
            all_tile_images.append(item["tile_image"])

        # Update progress
        progress = (i // batch_size + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(
            f"Processing batch {i // batch_size + 1}/{total_batches} ({len(all_coordinates)} tiles completed)"
        )

    progress_bar.empty()
    status_text.empty()

    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
        st.success(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings, all_coordinates, all_metadata, all_tile_images
    else:
        st.error("No embeddings generated")
        return None, None, None, None


def analyze_tile_content(coord, layers, tiles_root):
    """Analyze tile content for metadata."""
    zoom, x, y = coord
    layer_content = {}
    total_content = 0

    for layer in layers:
        tile_path = os.path.join(tiles_root, layer, str(zoom), str(x), f"{y}.png")

        if os.path.exists(tile_path):
            try:
                img = Image.open(tile_path).convert("RGB")
                img_array = np.array(img)
                # Calculate content density
                content_density = (
                    np.sum(np.mean(img_array, axis=2) > 30) / img_array[:, :, 0].size
                )
                layer_content[layer] = content_density
                total_content += content_density
            except:
                layer_content[layer] = 0.0
        else:
            layer_content[layer] = 0.0

    dominant_layer = (
        max(layer_content, key=layer_content.get) if layer_content else "unknown"
    )
    zone = get_geographic_zone(x, y)

    return {
        "zoom": zoom,
        "x": x,
        "y": y,
        "coordinate_str": f"{zoom}/{x}/{y}",
        "dominant_layer": dominant_layer,
        "total_content": total_content,
        "layer_content": layer_content,
        "geographic_zone": zone,
    }


def get_geographic_zone(x, y):
    """Estimate geographic zone."""
    if x < 76650:
        zone = "West"
    elif x > 76700:
        zone = "East"
    else:
        zone = "Central"

    if y < 44180:
        zone += "_North"
    elif y > 44220:
        zone += "_South"
    else:
        zone += "_Center"

    return zone


@st.cache_data
def compute_projections(embeddings, algorithms):
    """Compute dimensionality reduction projections."""
    projections = {}

    with st.spinner("Computing projections..."):
        progress_container = st.container()

        for i, algorithm in enumerate(algorithms):
            with progress_container:
                st.write(f"Computing {algorithm}...")

            if algorithm == "PCA":
                # PCA 2D and 3D
                pca_2d = PCA(n_components=2, random_state=42)
                pca_3d = PCA(n_components=3, random_state=42)
                projections["PCA_2D"] = pca_2d.fit_transform(embeddings)
                projections["PCA_3D"] = pca_3d.fit_transform(embeddings)

            elif algorithm == "t-SNE":
                # t-SNE with different perplexities for large datasets
                perplexity = min(30, len(embeddings) // 4)
                tsne_2d = TSNE(
                    n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1
                )
                tsne_3d = TSNE(
                    n_components=3, perplexity=perplexity, random_state=42, n_jobs=-1
                )
                projections["t-SNE_2D"] = tsne_2d.fit_transform(embeddings)
                projections["t-SNE_3D"] = tsne_3d.fit_transform(embeddings)

            elif algorithm == "UMAP":
                # UMAP with parameters suitable for large datasets
                n_neighbors = min(15, len(embeddings) // 10)
                # For parallelism, we remove random_state (UMAP can't parallelize with fixed seed)
                if len(embeddings) > 1000:  # Use parallelism for large datasets
                    umap_2d = umap.UMAP(
                        n_components=2, n_neighbors=n_neighbors, min_dist=0.1, n_jobs=-1
                    )
                    umap_3d = umap.UMAP(
                        n_components=3, n_neighbors=n_neighbors, min_dist=0.1, n_jobs=-1
                    )
                else:  # Use deterministic results for small datasets
                    umap_2d = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        random_state=42,
                    )
                    umap_3d = umap.UMAP(
                        n_components=3,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        random_state=42,
                    )

                projections["UMAP_2D"] = umap_2d.fit_transform(embeddings)
                projections["UMAP_3D"] = umap_3d.fit_transform(embeddings)

    return projections


def create_projector_plot(
    projections,
    metadata,
    tile_images,
    coordinates,
    algorithm,
    dimensions,
    color_by,
    point_size,
    show_images,
):
    """Create TensorFlow Projector-like visualization."""
    projection_key = f"{algorithm}_{dimensions}"
    projection = projections[projection_key]

    # Create base DataFrame
    plot_data = {
        "coordinate": [meta["coordinate_str"] for meta in metadata],
        "dominant_layer": [meta["dominant_layer"] for meta in metadata],
        "geographic_zone": [meta["geographic_zone"] for meta in metadata],
        "total_content": [meta["total_content"] for meta in metadata],
    }

    if dimensions == "2D":
        plot_data.update({"x": projection[:, 0], "y": projection[:, 1]})
    else:  # 3D
        plot_data.update(
            {"x": projection[:, 0], "y": projection[:, 1], "z": projection[:, 2]}
        )

    df = pd.DataFrame(plot_data)

    # Create the plot
    if dimensions == "2D":
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by,
            hover_data=[
                "coordinate",
                "dominant_layer",
                "geographic_zone",
                "total_content",
            ],
            title=f"{algorithm} 2D Projection - {len(df)} tiles",
            height=700,
        )
    else:  # 3D
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color=color_by,
            hover_data=[
                "coordinate",
                "dominant_layer",
                "geographic_zone",
                "total_content",
            ],
            title=f"{algorithm} 3D Projection - {len(df)} tiles",
            height=700,
        )

    # Update markers
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=0.5, color="white"), opacity=0.8),
        hovertemplate="""
        <b>%{customdata[0]}</b><br>
        <b>Layer:</b> %{customdata[1]}<br>
        <b>Zone:</b> %{customdata[2]}<br>
        <b>Content:</b> %{customdata[3]:.3f}<br>
        <extra></extra>
        """,
    )

    # Add tile images as custom markers if requested
    if show_images and len(df) <= 1000:  # Limit for performance
        # Sample points to show images
        sample_indices = np.random.choice(len(df), min(100, len(df)), replace=False)

        for idx in sample_indices:
            try:
                if dimensions == "2D":
                    fig.add_layout_image(
                        dict(
                            source=tile_images[idx],
                            xref="x",
                            yref="y",
                            x=df.iloc[idx]["x"],
                            y=df.iloc[idx]["y"],
                            sizex=0.1
                            * (projection[:, 0].max() - projection[:, 0].min()),
                            sizey=0.1
                            * (projection[:, 1].max() - projection[:, 1].min()),
                            sizing="contain",
                            opacity=0.8,
                            layer="above",
                        )
                    )
            except:
                continue

    # Update layout
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
    )

    if dimensions == "3D":
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
                zaxis=dict(showgrid=True, gridcolor="lightgray"),
                bgcolor="white",
            )
        )

    return fig, df


def display_tile_details(coord, layers, tiles_root, metadata):
    """Display detailed tile information with individual layer PNGs."""
    zoom, x, y = coord

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["🖼️ Composite View", "📚 Individual Layers", "📊 Layer Analysis"]
    )

    with tab1:
        st.subheader("Composite Tile Image")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Show large composite image
            composite_img = create_composite_image(coord, layers, tiles_root, size=512)
            st.image(
                composite_img,
                caption=f"Composite view - {metadata['coordinate_str']}",
                width=512,
            )

        with col2:
            st.write("**Tile Information:**")
            st.write(f"**Coordinate:** {metadata['coordinate_str']}")
            st.write(f"**Dominant Layer:** {metadata['dominant_layer']}")
            st.write(f"**Geographic Zone:** {metadata['geographic_zone']}")
            st.write(f"**Total Content Score:** {metadata['total_content']:.3f}")

            st.write("**Layer Content Scores:**")
            sorted_layers = sorted(
                metadata["layer_content"].items(), key=lambda x: x[1], reverse=True
            )
            for layer, score in sorted_layers:
                if score > 0:
                    st.write(f"  • **{layer}:** {score:.3f}")

    with tab2:
        st.subheader("Individual Layer Images")

        # Create a grid of individual layer images
        cols_per_row = 3
        layer_cols = st.columns(cols_per_row)

        for i, layer in enumerate(layers):
            with layer_cols[i % cols_per_row]:
                layer_img = load_layer_png(coord, layer, tiles_root)

                if layer_img:
                    # Resize for display
                    display_img = layer_img.resize((200, 200), Image.Resampling.LANCZOS)
                    st.image(
                        display_img,
                        caption=f"{layer}\n({metadata['layer_content'].get(layer, 0):.3f})",
                    )
                else:
                    # Show placeholder
                    placeholder = Image.new("RGB", (200, 200), color="lightgray")
                    st.image(placeholder, caption=f"{layer}\n(No data)")

    with tab3:
        st.subheader("Layer Content Analysis")

        # Create bar chart of layer content
        layer_data = []
        for layer, content in metadata["layer_content"].items():
            layer_data.append(
                {"Layer": layer, "Content Score": content, "Has Data": content > 0}
            )

        if layer_data:
            layer_df = pd.DataFrame(layer_data)

            # Color by whether layer has data
            fig = px.bar(
                layer_df,
                x="Content Score",
                y="Layer",
                orientation="h",
                color="Has Data",
                title=f"Layer Content Analysis - {metadata['coordinate_str']}",
                color_discrete_map={True: "steelblue", False: "lightgray"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data
            st.write("**Raw Layer Data:**")
            st.dataframe(layer_df, use_container_width=True)


def show_tile_gallery(coordinates, metadata, layers, tiles_root, sample_size=20):
    """Show a gallery of tiles as actual PNG images."""
    st.subheader(
        f"🖼️ Tile Gallery ({min(sample_size, len(coordinates))} random samples)"
    )

    # Ensure we don't sample more than available
    actual_sample_size = min(sample_size, len(coordinates))

    # Sample indices that exist in both coordinates and metadata
    available_indices = list(range(min(len(coordinates), len(metadata))))
    sample_indices = random.sample(available_indices, actual_sample_size)

    # Create gallery grid
    cols_per_row = 4

    for i in range(0, len(sample_indices), cols_per_row):
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            if i + j < len(sample_indices):
                idx = sample_indices[i + j]
                coord = coordinates[idx]
                meta = metadata[idx]

                with cols[j]:
                    try:
                        # Create composite image
                        composite_img = create_composite_image(
                            coord, layers, tiles_root, size=200
                        )
                        st.image(
                            composite_img,
                            caption=f"{meta['coordinate_str']}\n{meta['dominant_layer']}\nScore: {meta['total_content']:.2f}",
                            width=200,
                        )

                        # Add button to view details
                        if st.button("Details", key=f"detail_{idx}_{i}_{j}"):
                            st.session_state.selected_tile_for_details = idx
                            st.session_state.show_tile_details = True
                            st.session_state.detail_coord = coord
                            st.session_state.detail_meta = meta
                    except Exception as e:
                        st.error(f"Error displaying tile {coord}: {str(e)}")


def create_gallery_data(dataset, tiles_root, sample_size):
    """Create properly aligned coordinates and metadata for gallery."""
    # Sample coordinates first
    sample_coords = random.sample(
        dataset.common_coords, min(sample_size, len(dataset.common_coords))
    )

    # Generate metadata for the sampled coordinates
    sample_metadata = []
    for coord in sample_coords:
        try:
            meta = analyze_tile_content(coord, dataset.layers, tiles_root)
            sample_metadata.append(meta)
        except Exception as e:
            st.warning(f"Error analyzing tile {coord}: {e}")
            # Create minimal metadata
            sample_metadata.append(
                {
                    "coordinate_str": f"{coord[0]}/{coord[1]}/{coord[2]}",
                    "dominant_layer": "unknown",
                    "total_content": 0.0,
                    "geographic_zone": "unknown",
                }
            )

    return sample_coords, sample_metadata


def main():
    # App title
    st.title("🗺️ Loc2Vec TensorFlow Projector")
    st.markdown(
        "**Explore 16K+ map tile embeddings with interactive dimensionality reduction**"
    )

    # Sidebar controls
    st.sidebar.header("🎛️ Configuration")

    # File inputs
    st.sidebar.subheader("📁 Input Files")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )

    tiles_root = st.sidebar.text_input(
        "Tiles Root",
        value="12_layer_tiles/tiles",
        help="Root directory containing tiles",
    )

    # Performance settings
    st.sidebar.subheader("⚡ Performance Settings")
    max_tiles = st.sidebar.selectbox(
        "Max Tiles",
        options=[None, 1000, 5000, 10000, 16000],
        format_func=lambda x: "All tiles" if x is None else f"{x:,} tiles",
        index=0,
        help="Limit number of tiles for faster processing",
    )

    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return

    if not os.path.exists(tiles_root):
        st.error(f"Tiles directory not found: {tiles_root}")
        return

    # Load model and data
    try:
        with st.spinner("Loading model and dataset..."):
            model, dataset, device = load_model_and_data(model_path, tiles_root)
        st.sidebar.success(f"✅ Model loaded on {device}")
        st.sidebar.info(f"📊 Dataset: {len(dataset.common_coords):,} total tiles")
        st.sidebar.info(f"🗂️ Layers: {', '.join(dataset.layers)}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Main navigation
    view_mode = st.selectbox(
        "Select View Mode:",
        options=[
            "🎯 Embedding Projector",
            "🔍 Similarity Search (Annoy)",
            "🖼️ Tile Gallery",
            "🔍 Tile Explorer",
        ],
        help="Choose how you want to explore the tiles",
    )

    if view_mode == "🔍 Similarity Search (Annoy)":
        # Show similarity search mode
        st.header("🔍 Similarity Search with Annoy")
        st.markdown(
            "**Find similar tiles using fast approximate nearest neighbor search**"
        )

        # Performance tip
        st.info("""
        🚀 **Performance Tip**: For the best experience, pre-compute embeddings and indices:

        ```bash
        python precompute_embeddings.py --max_tiles 5000 --annoy_trees 200
        ```

        This will cache everything for instant loading! ⚡
        """)

        # Initialize embedding processor
        processor = EmbeddingProcessor()

        # Annoy settings (moved to sidebar first)
        st.sidebar.subheader("🎛️ Annoy Settings")
        annoy_metric = st.sidebar.selectbox(
            "Distance Metric",
            options=["angular", "euclidean", "manhattan", "dot"],
            index=0,
            help="Distance metric for Annoy index",
        )

        n_trees = st.sidebar.slider(
            "Number of Trees",
            min_value=10,
            max_value=500,
            value=100,
            help="More trees = better accuracy but larger index",
        )

        # Check if we have pre-computed data
        cache_key = f"{model_path}_{tiles_root}_{max_tiles}"

        # Try to load pre-computed embeddings first
        try:
            st.info("🔍 Checking for pre-computed embeddings...")
            embedding_data = processor.compute_embeddings(
                model, dataset, device, tiles_root, max_tiles, force_recompute=False
            )
            embeddings = embedding_data["embeddings"]
            coordinates = embedding_data["coordinates"]
            metadata = embedding_data["metadata"]
            tile_images = embedding_data["tile_images"]

            st.success(f"✅ Loaded pre-computed embeddings: {embeddings.shape}")

            # Try to load pre-computed Annoy index
            try:
                st.info(f"🔍 Checking for pre-computed {annoy_metric} Annoy index...")
                annoy_index = processor.build_annoy_index(
                    embeddings, annoy_metric, n_trees, force_rebuild=False
                )
                st.success(
                    f"✅ Loaded pre-computed Annoy index: {annoy_index.get_n_items()} items"
                )

            except Exception:
                st.warning(
                    f"⚠️ No pre-computed {annoy_metric} index found, building now..."
                )
                annoy_index = processor.build_annoy_index(
                    embeddings, annoy_metric, n_trees, force_rebuild=True
                )
                st.success(
                    f"✅ Built new Annoy index: {annoy_index.get_n_items()} items"
                )

            # Try to load pre-computed projections
            try:
                st.info("🔍 Checking for pre-computed projections...")
                projections = processor.compute_dimensionality_reductions(
                    embeddings, ["PCA", "UMAP"], force_recompute=False
                )
                st.success(
                    f"✅ Loaded pre-computed projections: {list(projections.keys())}"
                )

            except Exception:
                st.warning("⚠️ No pre-computed projections found, computing now...")
                projections = processor.compute_dimensionality_reductions(
                    embeddings, ["PCA", "UMAP"], force_recompute=True
                )
                st.success(f"✅ Computed projections: {list(projections.keys())}")

        except Exception:
            st.error(
                "❌ No pre-computed embeddings found. Computing fresh embeddings..."
            )
            st.info("💡 To avoid this delay, run: python precompute_embeddings.py")

            with st.spinner("Computing embeddings and building Annoy index..."):
                # Compute embeddings
                embedding_data = processor.compute_embeddings(
                    model, dataset, device, tiles_root, max_tiles, force_recompute=True
                )

                embeddings = embedding_data["embeddings"]
                coordinates = embedding_data["coordinates"]
                metadata = embedding_data["metadata"]
                tile_images = embedding_data["tile_images"]

                # Build Annoy index
                annoy_index = processor.build_annoy_index(
                    embeddings, annoy_metric, n_trees, force_rebuild=True
                )

                # Compute projections for visualization
                projections = processor.compute_dimensionality_reductions(
                    embeddings, ["PCA", "UMAP"], force_recompute=True
                )

        st.sidebar.success(f"✅ {len(embeddings):,} embeddings ready")
        st.sidebar.info(f"🌲 Annoy index: {annoy_index.get_n_items()} items")

        # Show cache info
        with st.sidebar.expander("📁 Cache Information"):
            st.write(f"**Cache Directory:** {processor.cache_dir}")
            cache_files = list(processor.cache_dir.glob("*.pkl")) + list(
                processor.cache_dir.glob("*.ann")
            )
            if cache_files:
                st.write(f"**Cached Files:** {len(cache_files)}")
                for cache_file in cache_files:
                    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
                    st.write(f"  • {cache_file.name} ({file_size:.1f} MB)")
            else:
                st.write("**No cache files found**")
                st.info("Run `python precompute_embeddings.py` to create cache files")

        # Similarity search interface
        st.subheader("🎯 Find Similar Tiles")

        # Query selection methods
        query_method = st.radio(
            "How do you want to select the query tile?",
            options=["Random Tile", "Coordinate Input", "Browse Gallery"],
            horizontal=True,
        )

        query_idx = None
        query_coord = None

        if query_method == "Random Tile":
            if (
                st.button("🎲 Select Random Tile")
                or "random_query_idx" not in st.session_state
            ):
                st.session_state.random_query_idx = random.randint(
                    0, len(coordinates) - 1
                )
            query_idx = st.session_state.random_query_idx
            query_coord = coordinates[query_idx]

        elif query_method == "Coordinate Input":
            col1, col2, col3 = st.columns(3)
            with col1:
                zoom = st.number_input(
                    "Zoom", value=17, min_value=1, max_value=20, key="sim_zoom"
                )
            with col2:
                x = st.number_input(
                    "X Coordinate", value=76680, min_value=0, key="sim_x"
                )
            with col3:
                y = st.number_input(
                    "Y Coordinate", value=44200, min_value=0, key="sim_y"
                )

            query_coord = (zoom, x, y)
            if query_coord in coordinates:
                query_idx = coordinates.index(query_coord)
            else:
                st.warning(f"Coordinate {query_coord} not found in dataset")
                # Find nearest coordinate
                distances = []
                for i, coord in enumerate(coordinates):
                    z, cx, cy = coord
                    if z == zoom:
                        dist = abs(cx - x) + abs(cy - y)  # Manhattan distance
                        distances.append((dist, i))

                if distances:
                    distances.sort()
                    nearest_idx = distances[0][1]
                    nearest_coord = coordinates[nearest_idx]
                    st.info(f"Using nearest coordinate: {nearest_coord}")
                    query_idx = nearest_idx
                    query_coord = nearest_coord

        elif query_method == "Browse Gallery":
            # Show a mini gallery to select from
            gallery_size = min(20, len(coordinates))
            if "gallery_indices" not in st.session_state:
                st.session_state.gallery_indices = random.sample(
                    range(len(coordinates)), gallery_size
                )

            if st.button("🔄 Refresh Gallery"):
                st.session_state.gallery_indices = random.sample(
                    range(len(coordinates)), gallery_size
                )

            # Display gallery in grid
            cols = st.columns(5)
            for i, idx in enumerate(st.session_state.gallery_indices):
                with cols[i % 5]:
                    coord = coordinates[idx]
                    meta = metadata[idx]

                    # Create small preview
                    preview_img = create_composite_image(
                        coord, dataset.layers, tiles_root, size=100
                    )
                    st.image(preview_img, caption=f"{coord[1]}, {coord[2]}", width=100)

                    if st.button("Select", key=f"gallery_select_{idx}"):
                        query_idx = idx
                        query_coord = coord
                        st.rerun()

        # Perform similarity search if query is selected
        if query_idx is not None and query_coord is not None:
            st.subheader(f"🎯 Query Tile: {query_coord}")

            # Display query tile
            col1, col2 = st.columns([1, 2])

            with col1:
                query_meta = metadata[query_idx]
                query_img = create_composite_image(
                    query_coord, dataset.layers, tiles_root, size=200
                )
                st.image(query_img, caption="Query Tile", width=200)

                st.write(f"**Coordinate:** {query_meta['coordinate_str']}")
                st.write(f"**Dominant Layer:** {query_meta['dominant_layer']}")
                st.write(f"**Geographic Zone:** {query_meta['geographic_zone']}")
                st.write(f"**Content Score:** {query_meta['total_content']:.3f}")

            with col2:
                # Similarity search parameters
                n_neighbors = st.slider(
                    "Number of Similar Tiles",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="How many similar tiles to find",
                )

                search_k = st.slider(
                    "Search Quality",
                    min_value=-1,
                    max_value=1000,
                    value=-1,
                    help="Higher = better quality but slower (-1 for default)",
                )

                # Perform search
                try:
                    neighbors, distances = processor.find_similar_tiles(
                        annoy_index,
                        query_idx,
                        n_neighbors + 1,
                        search_k,  # +1 to include query itself
                    )

                    # Remove query tile from results
                    if neighbors[0] == query_idx:
                        neighbors = neighbors[1:]
                        distances = distances[1:]
                    else:
                        neighbors = neighbors[:-1]
                        distances = distances[:-1]

                    st.success(f"Found {len(neighbors)} similar tiles!")

                except Exception as e:
                    st.error(f"Error performing similarity search: {str(e)}")
                    return

            # Display similar tiles
            st.subheader("🔍 Similar Tiles")

            # Show results in grid
            cols_per_row = 5
            for i in range(0, len(neighbors), cols_per_row):
                cols = st.columns(cols_per_row)

                for j, col in enumerate(cols):
                    if i + j < len(neighbors):
                        neighbor_idx = neighbors[i + j]
                        neighbor_coord = coordinates[neighbor_idx]
                        neighbor_meta = metadata[neighbor_idx]
                        distance = distances[i + j]

                        with col:
                            # Create preview image
                            neighbor_img = create_composite_image(
                                neighbor_coord, dataset.layers, tiles_root, size=120
                            )
                            st.image(neighbor_img, width=120)

                            # Show details
                            st.caption(f"**Rank:** {i + j + 1}")
                            st.caption(f"**Distance:** {distance:.3f}")
                            st.caption(
                                f"**Coord:** {neighbor_coord[1]}, {neighbor_coord[2]}"
                            )
                            st.caption(f"**Layer:** {neighbor_meta['dominant_layer']}")

                            # Detail button
                            if st.button(
                                "🔍 Details", key=f"detail_neighbor_{neighbor_idx}"
                            ):
                                st.session_state.show_tile_details = True
                                st.session_state.detail_coord = neighbor_coord
                                st.session_state.detail_meta = neighbor_meta

            # Visualization of similar tiles in embedding space
            st.subheader("📊 Similarity Visualization")

            # Create visualization with query and similar tiles highlighted
            viz_algorithm = st.selectbox(
                "Projection Algorithm",
                options=["PCA", "UMAP"],
                help="Algorithm for 2D projection",
            )

            if viz_algorithm in projections:
                projection_data = projections[viz_algorithm]

                # Create DataFrame for plotting
                viz_df = pd.DataFrame(
                    {
                        "x": projection_data[:, 0],
                        "y": projection_data[:, 1],
                        "coordinate": [meta["coordinate_str"] for meta in metadata],
                        "dominant_layer": [meta["dominant_layer"] for meta in metadata],
                        "type": ["other"] * len(metadata),
                    }
                )

                # Mark query and similar tiles
                viz_df.loc[query_idx, "type"] = "query"
                for neighbor_idx in neighbors:
                    viz_df.loc[neighbor_idx, "type"] = "similar"

                # Create plot
                color_map = {"query": "red", "similar": "orange", "other": "lightblue"}
                size_map = {"query": 12, "similar": 8, "other": 4}

                fig = px.scatter(
                    viz_df,
                    x="x",
                    y="y",
                    color="type",
                    color_discrete_map=color_map,
                    size=[size_map[t] for t in viz_df["type"]],
                    hover_data=["coordinate", "dominant_layer"],
                    title=f"Similarity Search Results - {viz_algorithm} Projection",
                )

                fig.update_layout(
                    showlegend=True,
                    legend=dict(title="Tile Type", itemsizing="constant"),
                )

                st.plotly_chart(fig, use_container_width=True)

        # Show detailed tile analysis if requested
        if st.session_state.get("show_tile_details", False):
            st.subheader("🔬 Detailed Tile Analysis")
            display_tile_details(
                st.session_state.detail_coord,
                dataset.layers,
                tiles_root,
                st.session_state.detail_meta,
            )

            if st.button("✕ Close Details"):
                st.session_state.show_tile_details = False

        return

    if view_mode == "🖼️ Tile Gallery":
        # Show tile gallery mode
        st.header("🖼️ Tile Gallery Mode")

        gallery_size = st.slider("Gallery Size", min_value=10, max_value=100, value=20)

        if (
            st.button("🎲 Generate Random Gallery")
            or "gallery_coords" not in st.session_state
        ):
            with st.spinner("Generating tile gallery..."):
                gallery_coords, gallery_metadata = create_gallery_data(
                    dataset, tiles_root, gallery_size
                )
                st.session_state.gallery_coords = gallery_coords
                st.session_state.gallery_metadata = gallery_metadata

        if "gallery_coords" in st.session_state:
            show_tile_gallery(
                st.session_state.gallery_coords,
                st.session_state.gallery_metadata,
                dataset.layers,
                tiles_root,
                len(st.session_state.gallery_coords),
            )

            # Show detailed tile analysis if requested
            if st.session_state.get("show_tile_details", False):
                st.subheader("🔬 Detailed Tile Analysis")
                display_tile_details(
                    st.session_state.detail_coord,
                    dataset.layers,
                    tiles_root,
                    st.session_state.detail_meta,
                )

                if st.button("✕ Close Details"):
                    st.session_state.show_tile_details = False

        return

    elif view_mode == "🔍 Tile Explorer":
        # Show tile explorer mode
        st.header("🔍 Tile Explorer Mode")

        # Coordinate input
        col1, col2, col3 = st.columns(3)
        with col1:
            zoom = st.number_input("Zoom", value=17, min_value=1, max_value=20)
        with col2:
            x = st.number_input("X Coordinate", value=76680, min_value=0)
        with col3:
            y = st.number_input("Y Coordinate", value=44200, min_value=0)

        coord = (zoom, x, y)

        if coord in dataset.common_coords:
            meta = analyze_tile_content(coord, dataset.layers, tiles_root)
            display_tile_details(coord, dataset.layers, tiles_root, meta)
        else:
            st.warning(f"Coordinate {coord} not found in dataset")

            # Show nearby coordinates
            st.write("**Available coordinates near your input:**")
            nearby = []
            for c in dataset.common_coords:
                z, cx, cy = c
                if abs(cx - x) <= 5 and abs(cy - y) <= 5 and z == zoom:
                    nearby.append(c)

            if nearby:
                st.write(f"Found {len(nearby)} nearby coordinates:")
                for i, c in enumerate(nearby[:10]):  # Show first 10
                    if st.button(f"View {c[1]}, {c[2]}", key=f"nearby_{i}"):
                        meta = analyze_tile_content(c, dataset.layers, tiles_root)
                        display_tile_details(c, dataset.layers, tiles_root, meta)
                        break
            else:
                st.write("No nearby coordinates found")

        return

    # Continue with embedding projector mode
    # Generate embeddings
    cache_key = f"{model_path}_{tiles_root}_{max_tiles}"
    if f"embeddings_{cache_key}" not in st.session_state:
        try:
            with st.spinner("Generating embeddings for all tiles..."):
                embeddings, coordinates, metadata, tile_images = (
                    generate_all_embeddings(
                        model, dataset, device, tiles_root, max_tiles
                    )
                )

            if embeddings is not None:
                st.session_state[f"embeddings_{cache_key}"] = embeddings
                st.session_state[f"coordinates_{cache_key}"] = coordinates
                st.session_state[f"metadata_{cache_key}"] = metadata
                st.session_state[f"tile_images_{cache_key}"] = tile_images
            else:
                st.error("Failed to generate embeddings")
                return
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return
    else:
        embeddings = st.session_state[f"embeddings_{cache_key}"]
        coordinates = st.session_state[f"coordinates_{cache_key}"]
        metadata = st.session_state[f"metadata_{cache_key}"]
        tile_images = st.session_state[f"tile_images_{cache_key}"]

    st.sidebar.success(f"✅ {len(embeddings):,} embeddings ready")

    # Visualization controls
    st.sidebar.subheader("📊 Visualization")

    # Algorithm selection
    available_algorithms = ["PCA", "t-SNE", "UMAP"]
    selected_algorithms = st.sidebar.multiselect(
        "Algorithms to compute",
        options=available_algorithms,
        default=["PCA", "UMAP"],
        help="Select which algorithms to compute (t-SNE is slower for large datasets)",
    )

    if not selected_algorithms:
        st.error("Please select at least one algorithm")
        return

    # Compute projections
    projection_cache_key = f"{cache_key}_{'_'.join(selected_algorithms)}"
    if f"projections_{projection_cache_key}" not in st.session_state:
        try:
            projections = compute_projections(embeddings, selected_algorithms)
            st.session_state[f"projections_{projection_cache_key}"] = projections
        except Exception as e:
            st.error(f"Error computing projections: {str(e)}")
            return
    else:
        projections = st.session_state[f"projections_{projection_cache_key}"]

    # Plot controls
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        options=selected_algorithms,
        help="Select algorithm for visualization",
    )

    dimensions = st.sidebar.selectbox(
        "Dimensions", options=["2D", "3D"], index=1, help="2D or 3D visualization"
    )

    color_by = st.sidebar.selectbox(
        "Color By",
        options=["dominant_layer", "geographic_zone", "total_content"],
        help="How to color the points",
    )

    point_size = st.sidebar.slider(
        "Point Size",
        min_value=1,
        max_value=10,
        value=3,
        help="Size of points in the plot",
    )

    show_images = st.sidebar.checkbox(
        "Show tile images",
        value=False,
        help="Overlay tile images on plot (only for <1000 points)",
    )

    # Create main visualization
    st.subheader(f"🎯 {algorithm} {dimensions} Projection")

    try:
        fig, df = create_projector_plot(
            projections,
            metadata,
            tile_images,
            coordinates,
            algorithm,
            dimensions,
            color_by,
            point_size,
            show_images,
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, key="main_projector")

        # Interactive point selection
        st.subheader("🔍 Explore Specific Points")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Search functionality
            search_term = st.text_input(
                "Search coordinates (e.g., '76680/44200')",
                help="Search for specific coordinates or layers",
            )

            if search_term:
                # Filter points based on search
                mask = df["coordinate"].str.contains(search_term, case=False) | df[
                    "dominant_layer"
                ].str.contains(search_term, case=False)
                filtered_df = df[mask]

                if len(filtered_df) > 0:
                    st.write(f"Found {len(filtered_df)} matching points:")
                    selected_idx = st.selectbox(
                        "Select point:",
                        options=filtered_df.index,
                        format_func=lambda x: f"{df.loc[x, 'coordinate']} ({df.loc[x, 'dominant_layer']})",
                    )
                else:
                    st.write("No matching points found")
                    selected_idx = None
            else:
                # Random or manual selection
                selected_idx = st.selectbox(
                    "Select point to explore:",
                    options=range(len(df)),
                    format_func=lambda x: f"{df.loc[x, 'coordinate']} ({df.loc[x, 'dominant_layer']})",
                )

        with col2:
            if selected_idx is not None:
                coord = coordinates[selected_idx]
                meta = metadata[selected_idx]

                # Quick preview
                st.write(f"**Point {selected_idx}**")
                st.write(f"**Coordinate:** {meta['coordinate_str']}")
                st.write(f"**Dominant Layer:** {meta['dominant_layer']}")
                st.write(f"**Geographic Zone:** {meta['geographic_zone']}")
                st.write(f"**Content Score:** {meta['total_content']:.3f}")

                # Show small preview
                composite_preview = create_composite_image(
                    coord, dataset.layers, tiles_root, size=150
                )
                st.image(composite_preview, caption="Quick Preview", width=150)

                # Button for detailed view
                if st.button(
                    "🔍 View Detailed Analysis", key=f"detail_main_{selected_idx}"
                ):
                    st.session_state.show_tile_details = True
                    st.session_state.detail_coord = coord
                    st.session_state.detail_meta = meta

        # Show detailed tile analysis if requested
        if st.session_state.get("show_tile_details", False):
            st.subheader("🔬 Detailed Tile Analysis")
            display_tile_details(
                st.session_state.detail_coord,
                dataset.layers,
                tiles_root,
                st.session_state.detail_meta,
            )

            if st.button("✕ Close Details"):
                st.session_state.show_tile_details = False

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return

    # Statistics and analysis
    st.subheader("📊 Dataset Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tiles", f"{len(embeddings):,}")

    with col2:
        unique_layers = len(set(meta["dominant_layer"] for meta in metadata))
        st.metric("Unique Layers", unique_layers)

    with col3:
        unique_zones = len(set(meta["geographic_zone"] for meta in metadata))
        st.metric("Geographic Zones", unique_zones)

    with col4:
        avg_content = np.mean([meta["total_content"] for meta in metadata])
        st.metric("Avg Content Score", f"{avg_content:.3f}")

    # Distribution plots
    with st.expander("📈 Data Distributions"):
        col1, col2 = st.columns(2)

        with col1:
            # Layer distribution
            layer_counts = {}
            for meta in metadata:
                layer = meta["dominant_layer"]
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

            layer_df = pd.DataFrame(
                list(layer_counts.items()), columns=["Layer", "Count"]
            )
            fig_layer = px.bar(
                layer_df,
                x="Count",
                y="Layer",
                orientation="h",
                title="Distribution by Dominant Layer",
            )
            st.plotly_chart(fig_layer, use_container_width=True)

        with col2:
            # Zone distribution
            zone_counts = {}
            for meta in metadata:
                zone = meta["geographic_zone"]
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

            zone_df = pd.DataFrame(list(zone_counts.items()), columns=["Zone", "Count"])
            fig_zone = px.bar(
                zone_df,
                x="Count",
                y="Zone",
                orientation="h",
                title="Distribution by Geographic Zone",
            )
            st.plotly_chart(fig_zone, use_container_width=True)

    # Export functionality
    with st.expander("💾 Export Data"):
        st.write("Export embeddings and projections for further analysis:")

        if st.button("Prepare Export Data"):
            export_data = {
                "embeddings": embeddings,
                "coordinates": coordinates,
                "metadata": metadata,
                "projections": projections,
            }

            # Convert to downloadable format
            export_df = pd.DataFrame(
                {
                    "coordinate": [meta["coordinate_str"] for meta in metadata],
                    "dominant_layer": [meta["dominant_layer"] for meta in metadata],
                    "geographic_zone": [meta["geographic_zone"] for meta in metadata],
                    "total_content": [meta["total_content"] for meta in metadata],
                }
            )

            # Add embedding dimensions
            for i in range(embeddings.shape[1]):
                export_df[f"embedding_{i}"] = embeddings[:, i]

            # Add projections
            for proj_name, proj_data in projections.items():
                for i in range(proj_data.shape[1]):
                    export_df[f"{proj_name}_dim_{i}"] = proj_data[:, i]

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="loc2vec_embeddings_projections.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
