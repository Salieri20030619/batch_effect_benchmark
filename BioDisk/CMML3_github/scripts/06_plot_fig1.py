#!/usr/bin/env python
"""
Figure 1: UMAP comparison across 5 batch correction methods
============================================================
2 rows x 5 cols:
  Row 1: colored by batch (assay) - shows batch mixing
  Row 2: colored by cell_type_major - shows bio conservation

Input:  data/pancreas_integrated_eval.h5ad
Output: figures/fig1_umap.png, .pdf
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT = os.path.join(PROJECT_ROOT, "data", "pancreas_integrated_eval.h5ad")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Plot config
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

METHOD_ORDER = ["unintegrated", "combat", "harmony", "scvi", "seurat_cca"]
METHOD_LABELS = {
    "unintegrated": "Unintegrated",
    "combat":       "ComBat",
    "harmony":      "Harmony",
    "scvi":         "scVI",
    "seurat_cca":   "Seurat CCA",
}

print("Loading AnnData...", flush=True)
adata = sc.read_h5ad(INPUT)
print(f"Shape: {adata.shape}")

# ----- Color palettes -----
# Batches (5 categories): use a perceptually uniform palette
batches = sorted(adata.obs["batch"].astype(str).unique())
n_batches = len(batches)
batch_palette = plt.get_cmap("tab10")(np.arange(n_batches))
batch_colors = dict(zip(batches, batch_palette))

# Cell types (~14-20 major): use tab20
celltypes = adata.obs["cell_type_major"].cat.categories.tolist() if hasattr(
    adata.obs["cell_type_major"], "cat") else sorted(adata.obs["cell_type_major"].unique())
ct_palette = plt.get_cmap("tab20")(np.arange(len(celltypes)) % 20)
ct_colors = dict(zip(celltypes, ct_palette))

# ----- Create figure -----
n_methods = len(METHOD_ORDER)
fig, axes = plt.subplots(2, n_methods, figsize=(n_methods * 2.8, 5.8))

# Shuffle indices once for consistent (deterministic) plotting order
rng = np.random.default_rng(42)
shuffle_idx = rng.permutation(adata.n_obs)

for col_idx, method in enumerate(METHOD_ORDER):
    umap_key = f"X_umap_{method}"
    if umap_key not in adata.obsm:
        # Use unintegrated UMAP as placeholder; report missing
        print(f"  [warn] {umap_key} not found in adata.obsm; skipping {method}")
        for row in (0, 1):
            axes[row, col_idx].text(0.5, 0.5, "N/A",
                ha="center", va="center", transform=axes[row, col_idx].transAxes)
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])
        continue

    umap = adata.obsm[umap_key][shuffle_idx]

    # Row 0: by batch
    ax = axes[0, col_idx]
    batch_vals = adata.obs["batch"].astype(str).values[shuffle_idx]
    for b in batches:
        m = batch_vals == b
        ax.scatter(umap[m, 0], umap[m, 1], c=[batch_colors[b]], s=1.0,
                   alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(METHOD_LABELS[method], fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if col_idx == 0:
        ax.set_ylabel("Colored by batch (assay)", fontsize=10, fontweight="bold")

    # Row 1: by cell type
    ax = axes[1, col_idx]
    ct_vals = adata.obs["cell_type_major"].astype(str).values[shuffle_idx]
    for ct in celltypes:
        m = ct_vals == ct
        if m.sum() == 0:
            continue
        ax.scatter(umap[m, 0], umap[m, 1], c=[ct_colors[ct]], s=1.0,
                   alpha=0.6, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    if col_idx == 0:
        ax.set_ylabel("Colored by cell type", fontsize=10, fontweight="bold")

# Legends below figure - two compact rows
# Batch legend
batch_handles = [Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=batch_colors[b], markersize=6, label=b)
                 for b in batches]
fig.legend(handles=batch_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
           ncol=min(len(batches), 5), title="Batch",
           frameon=False, fontsize=8, title_fontsize=9)

# Cell type legend - put on the right side (vertical)
ct_handles = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=ct_colors[ct], markersize=6, label=ct)
              for ct in celltypes]
# Make celltype legend extend wider, accept many columns
fig.legend(handles=ct_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
           ncol=min(len(celltypes), 8), title="Cell type",
           frameon=False, fontsize=7, title_fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])

# Save
out_png = os.path.join(FIG_DIR, "fig1_umap.png")
out_pdf = os.path.join(FIG_DIR, "fig1_umap.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"Figure size: {os.path.getsize(out_png)/1024:.0f} KB (PNG)")
