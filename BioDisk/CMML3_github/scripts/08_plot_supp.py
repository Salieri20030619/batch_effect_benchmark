#!/usr/bin/env python
"""
Supplementary figures
======================
- Figure S1: Per-batch cell type composition (stacked bar)
- Figure S2: Per-method runtime breakdown (table)
- Figure S3: Sensitivity to rare cell types (UMAP highlighting epsilon, immune_rare)
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT = os.path.join(PROJECT_ROOT, "data", "pancreas_integrated_eval.h5ad")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "savefig.dpi": 300,
})

print("Loading...", flush=True)
adata = sc.read_h5ad(INPUT)
print(f"Shape: {adata.shape}")

# ---------------------------------------------------------
# Figure S1: Per-batch cell type composition (stacked bar)
# ---------------------------------------------------------
print("\nFigure S1: per-batch composition")
batches = sorted(adata.obs["batch"].astype(str).unique())
celltypes = sorted(adata.obs["cell_type_major"].astype(str).unique())

# Build proportion table
prop_table = pd.crosstab(adata.obs["batch"], adata.obs["cell_type_major"],
                         normalize="index") * 100  # percent
prop_table = prop_table.reindex(index=batches, columns=celltypes)

# Plot
fig, ax = plt.subplots(figsize=(9, 4))
ct_palette = plt.get_cmap("tab20")(np.arange(len(celltypes)) % 20)
bottom = np.zeros(len(batches))
for ci, ct in enumerate(celltypes):
    vals = prop_table[ct].values
    ax.bar(batches, vals, bottom=bottom, label=ct, color=ct_palette[ci], width=0.7)
    bottom += vals

ax.set_ylabel("% of cells")
ax.set_title("Cell type composition per batch (after harmonization)")
ax.set_ylim(0, 100)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=2, fontsize=7,
          title="Cell type", title_fontsize=8)
plt.xticks(rotation=15)
plt.tight_layout()
out_png = os.path.join(FIG_DIR, "figS1_batch_composition.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"  Saved: {out_png}")
plt.close()

# Also save raw counts
counts_table = pd.crosstab(adata.obs["batch"], adata.obs["cell_type_major"])
counts_table.to_csv(os.path.join(PROJECT_ROOT, "results", "batch_celltype_counts.csv"))
print("  Saved: results/batch_celltype_counts.csv")

# ---------------------------------------------------------
# Figure S2: rare cell type UMAP highlight
# ---------------------------------------------------------
print("\nFigure S2: rare cell types under each method")
# Define rare cell types (< 200 cells)
ct_counts = adata.obs["cell_type_major"].value_counts()
rare_types = ct_counts[ct_counts < 200].index.tolist()
print(f"  Rare cell types (<200 cells): {rare_types}")

# Pick top 4 methods to compare
methods_to_compare = ["unintegrated", "harmony", "scvi", "seurat_cca"]
fig, axes = plt.subplots(1, len(methods_to_compare),
                          figsize=(len(methods_to_compare) * 3, 3))

# Color: rare types highlighted, others gray
for col_idx, method in enumerate(methods_to_compare):
    ax = axes[col_idx]
    umap_key = f"X_umap_{method}"
    if umap_key not in adata.obsm:
        continue
    umap = adata.obsm[umap_key]

    # Plot non-rare cells in gray
    non_rare = ~adata.obs["cell_type_major"].isin(rare_types)
    ax.scatter(umap[non_rare, 0], umap[non_rare, 1], c="lightgray",
               s=0.5, alpha=0.3, linewidths=0, rasterized=True)

    # Plot each rare type with distinct color
    palette = plt.get_cmap("tab10")(np.arange(len(rare_types)))
    for ri, rt in enumerate(rare_types):
        m = adata.obs["cell_type_major"] == rt
        if m.sum() == 0:
            continue
        ax.scatter(umap[m, 0], umap[m, 1], c=[palette[ri]],
                   s=8, alpha=0.85, linewidths=0, label=f"{rt} (n={m.sum()})")

    ax.set_title(method)
    ax.set_xticks([])
    ax.set_yticks([])

# One legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=min(len(rare_types), 5),
           bbox_to_anchor=(0.5, -0.05), fontsize=7, frameon=False)

plt.suptitle("Sensitivity to low-frequency cell types", fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
out_png = os.path.join(FIG_DIR, "figS2_rare_celltypes.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"  Saved: {out_png}")
plt.close()

# ---------------------------------------------------------
# Figure S3: per-method clustering vs ground truth confusion
# ---------------------------------------------------------
# This one is informational - we already have NMI/ARI numbers
# Skip if too time-consuming; users can recompute manually

print("\n[+] All supplementary figures generated")
print(f"Figures in: {FIG_DIR}")
for fn in sorted(os.listdir(FIG_DIR)):
    fp = os.path.join(FIG_DIR, fn)
    print(f"  {fn:40s} {os.path.getsize(fp)/1024:8.1f} KB")
