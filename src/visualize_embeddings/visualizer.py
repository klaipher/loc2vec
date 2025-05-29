#!/usr/bin/env python3
# img_embed_dash.py
"""
Generate ResNet-18 embeddings for a folder of images and visualise them
in 2-D or 3-D with Plotly + Dash tooltips (no custom JS).

Run:
    python img_embed_dash.py --image_folder imgs        # 2-D PCA
    python img_embed_dash.py --image_folder imgs -k 3   # 3-D PCA
"""

import argparse, random, ssl, base64, io
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, no_update, callback

# ─── Embedding helpers ─────────────────────────────────────────────────
def load_pretrained_model(device):
    ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        w = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=w)
        tfm   = w.transforms()
    finally:
        ssl._create_default_https_context = ctx
    model.fc = nn.Identity()
    model.to(device).eval()
    return model, tfm

def preprocess(p: Path, tfm):
    try:
        return tfm(Image.open(p).convert("RGB"))
    except Exception as e:
        print(f" ! {p}: {e}")
        return None

def get_embeddings(folder: Path, tfm, model, device,
                   batch=64, max_imgs=None):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    if max_imgs and len(paths) > max_imgs:
        paths = random.sample(paths, max_imgs)

    embs, buf = [], []
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            t = preprocess(p, tfm)
            if t is not None:
                buf.append(t)
            if len(buf) == batch or i == len(paths):
                embs.extend(model(torch.stack(buf).to(device)).cpu().numpy())
                buf = []
    return np.asarray(embs), paths

def reduce(x, k, method="pca"):
    if x.shape[0] < k:
        return x[:, :k]
    if method == "pca":
        return PCA(k).fit_transform(x)
    if method == "tsne":
        perp = max(5, min(30, x.shape[0]-1))
        return TSNE(k, perplexity=perp, init="pca", random_state=42,
                    n_iter=400).fit_transform(x)
    raise ValueError(method)

# ─── Dash app ──────────────────────────────────────────────────────────
def build_dash(points, k):
    coords = np.array([p["coords"] for p in points])
    marker = dict(size=5 if k==2 else 6,
                  color=np.arange(len(points)),
                  colorscale="Viridis",
                  opacity=.85)

    trace_cls = go.Scatter if k==2 else go.Scatter3d
    trace_kwargs = {"x": coords[:,0], "y": coords[:,1]}
    if k == 3:
        trace_kwargs["z"] = coords[:,2]

    fig = go.Figure(trace_cls(
        **trace_kwargs,
        mode="markers",
        customdata=list(range(len(points))),     # stable index
        marker=marker,
        hoverinfo="none", hovertemplate=None     # disable default
    ))

    if k == 2:
        fig.update_layout(xaxis_title="Comp 1", yaxis_title="Comp 2")
    else:
        fig.update_layout(scene=dict(
            xaxis_title="Comp 1", yaxis_title="Comp 2", zaxis_title="Comp 3"))

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True,
                  style={"height": "90vh"}),
        dcc.Tooltip(id="graph-tooltip", direction="right")
    ])

    @callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_tooltip(hoverData):
        if hoverData is None:
            return False, no_update, no_update
        pt   = hoverData["points"][0]
        idx  = pt.get("customdata", pt["pointNumber"])   # works 2-D & 3-D
        bbox = pt["bbox"]
        p    = points[idx]

        children = html.Div([
            html.Img(src=p["img"], style={"width": "150px", "display":"block"}),
            html.P(p["file"], style={"textAlign":"center", "margin":0})
        ], style={"white-space":"normal"})
        return True, bbox, children

    return app

# ─── CLI ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--reduction_method", choices=["pca","tsne"], default="pca")
    ap.add_argument("-k", "--n_components", type=int, choices=[2,3], default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_images", type=int)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() and not args.no_cuda else
              "mps"  if torch.backends.mps.is_available() and not args.no_cuda
                      else "cpu")
    print("Device:", device)

    model, tfm = load_pretrained_model(device)
    embs, paths = get_embeddings(Path(args.image_folder), tfm, model, device,
                                 args.batch_size, args.max_images)
    reduced = reduce(embs, args.n_components, args.reduction_method)

    # build tooltip points ------------------------------------------------
    pts = []
    for vec, p in zip(reduced, paths):
        img = Image.open(p).convert("RGB")
        img.thumbnail((150,150), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85); buf.seek(0)
        img_b64 = "data:image/jpeg;base64," + \
                  base64.b64encode(buf.read()).decode()
        pts.append({"coords": vec.tolist(), "img": img_b64, "file": p.name})

    dash_app = build_dash(pts, args.n_components)
    dash_app.run(debug=True, port=8050)

if __name__ == "__main__":
    main()
