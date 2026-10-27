#!/usr/bin/env python
"""
Figure 2 (v2): Metrics heatmap + Scalability
Layout fixes from v1:
  - Move category labels (Batch / Bio) and Overall column out of overlap zones
  - Separate the heatmap and a small "Overall" bar
  - Adjusted gridspec
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
METRICS_CSV = os.path.join(PROJECT_ROOT, "results", "metrics.csv")
SCAL_DIR    = os.path.join(PROJECT_ROOT, "results", "scalability")
FIG_DIR     = os.path.join(PROJECT_ROOT, "figures")

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 100, "savefig.dpi": 300,
})

METHOD_LABELS = {
    "unintegrated": "Unintegrated", "combat": "ComBat",
    "harmony": "Harmony", "scvi": "scVI", "seurat_cca": "Seurat CCA",
}
METHOD_COLORS = {
    "harmony": "#1f77b4", "scvi": "#d62728", "seurat_cca": "#2ca02c",
}
METRIC_LABELS = {
    "batch_asw": "Batch ASW", "ilisi": "iLISI",
    "graph_connectivity": "Graph conn.", "kbet": "kBET",
    "celltype_asw": "Cell type ASW", "nmi": "NMI",
    "ari": "ARI", "clisi": "cLISI",
}

# ----- Load -----
df_metrics = pd.read_csv(METRICS_CSV, index_col=0)
METHOD_ORDER = ["unintegrated", "combat", "harmony", "scvi", "seurat_cca"]
df_metrics = df_metrics.reindex(METHOD_ORDER)

heatmap_cols = ["batch_asw", "ilisi", "graph_connectivity", "kbet",
                "celltype_asw", "nmi", "ari", "clisi"]
heatmap_data = df_metrics[heatmap_cols].copy()
heatmap_norm = heatmap_data.copy()
for col in heatmap_cols:
    v = heatmap_data[col].values
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmax > vmin:
        heatmap_norm[col] = (v - vmin) / (vmax - vmin)
    else:
        heatmap_norm[col] = 0.5

scal_rows = []
for f in glob.glob(os.path.join(SCAL_DIR, "*.json")):
    with open(f) as fp:
        d = json.load(fp)
    scal_rows.append(d)
df_scal = pd.DataFrame(scal_rows)
scale_order = {"21k": 1, "50k": 2, "100k": 3, "186k": 4}
df_scal["scale_idx"] = df_scal["scale"].map(scale_order)
df_scal = df_scal.sort_values(["method", "scale_idx"])

# ----- Figure -----
fig = plt.figure(figsize=(13.5, 5.2))
# 4 columns: heatmap, overall_bar, runtime, memory
gs = fig.add_gridspec(
    1, 4,
    width_ratios=[2.4, 0.6, 1.2, 1.2],
    wspace=0.55
)

cmap = plt.get_cmap("viridis")

# ===== Panel (a): heatmap =====
ax_hm = fig.add_subplot(gs[0, 0])
im = ax_hm.imshow(heatmap_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.values[i, j]
        norm_val = heatmap_norm.values[i, j]
        # viridis: low = dark (need white text), high = bright yellow (need black)
        text_color = "white" if norm_val < 0.55 else "black"
        ax_hm.text(j, i, f"{val:.3f}", ha="center", va="center",
                   color=text_color, fontsize=7.5)

ax_hm.set_xticks(range(len(heatmap_cols)))
ax_hm.set_xticklabels([METRIC_LABELS[c] for c in heatmap_cols],
                      rotation=35, ha="right")
ax_hm.set_yticks(range(len(METHOD_ORDER)))
ax_hm.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])

# Vertical line separating batch and bio metrics
ax_hm.axvline(3.5, color="black", linewidth=1.5)

# Place category labels ABOVE the heatmap (between title and first row)
# Using axes coordinates anchored at y > 1 puts them just above the matrix
ax_hm.text(0.22, 1.02, "Batch correction", transform=ax_hm.transAxes,
           ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_hm.text(0.72, 1.02, "Bio conservation", transform=ax_hm.transAxes,
           ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_hm.set_title("(a) Metric scores", loc="left", fontweight="bold", pad=22)

# Horizontal colorbar below the heatmap, anchored inside the axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax_hm, width="55%", height="3.5%",
                 loc="lower center",
                 bbox_to_anchor=(0.0, -0.50, 1.0, 1.0),
                 bbox_transform=ax_hm.transAxes,
                 borderpad=0)
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("Per-metric normalized score (low \u2192 high)",
               fontsize=7.5, labelpad=2)
cbar.ax.tick_params(labelsize=6.5)

# ===== Overall bar (separate small panel) =====
ax_ov = fig.add_subplot(gs[0, 1])
overall_vals = df_metrics["overall"].values
y_pos = np.arange(len(METHOD_ORDER))

# Color bars by score with same colormap
overall_norm = (overall_vals - overall_vals.min()) / (overall_vals.max() - overall_vals.min() + 1e-9)
bar_colors = cmap(overall_norm)

bars = ax_ov.barh(y_pos, overall_vals, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
for i, v in enumerate(overall_vals):
    ax_ov.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

ax_ov.set_yticks(y_pos)
ax_ov.set_yticklabels([])  # method names are on heatmap
ax_ov.invert_yaxis()
ax_ov.set_xlim(0.50, max(overall_vals) * 1.10)
ax_ov.set_xlabel("Overall score", fontsize=8)
ax_ov.set_title("(b) Overall", loc="left", fontweight="bold", pad=8)
ax_ov.spines["top"].set_visible(False)
ax_ov.spines["right"].set_visible(False)
ax_ov.grid(axis="x", alpha=0.3, linestyle="--")

# ===== Panel (c): Runtime =====
ax_rt = fig.add_subplot(gs[0, 2])
for method in ["harmony", "scvi", "seurat_cca"]:
    sub = df_scal[df_scal["method"] == method]
    ax_rt.plot(sub["n_cells"], sub["runtime_sec"] / 60,
               marker="o", markersize=6, linewidth=2,
               color=METHOD_COLORS[method], label=METHOD_LABELS[method])
ax_rt.set_xscale("log")
ax_rt.set_yscale("log")
ax_rt.set_xlabel("Number of cells")
ax_rt.set_ylabel("Runtime (min)")
ax_rt.set_title("(c) Scalability: runtime", loc="left", fontweight="bold")
ax_rt.grid(True, which="both", alpha=0.3, linestyle="--")
ax_rt.legend(loc="upper left", frameon=True, fontsize=7)

# ===== Panel (d): Memory =====
ax_mem = fig.add_subplot(gs[0, 3])
for method in ["harmony", "scvi", "seurat_cca"]:
    sub = df_scal[df_scal["method"] == method]
    ax_mem.plot(sub["n_cells"], sub["peak_mem_mb"] / 1024,
                marker="o", markersize=6, linewidth=2,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
ax_mem.set_xscale("log")
ax_mem.set_yscale("log")
ax_mem.set_xlabel("Number of cells")
ax_mem.set_ylabel("Peak memory (GB)")
ax_mem.set_title("(d) Scalability: memory", loc="left", fontweight="bold")
ax_mem.grid(True, which="both", alpha=0.3, linestyle="--")
ax_mem.legend(loc="upper left", frameon=True, fontsize=7)

plt.tight_layout()
out_png = os.path.join(FIG_DIR, "fig2_metrics_scalability.png")
out_pdf = os.path.join(FIG_DIR, "fig2_metrics_scalability.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

# Re-save scalability summary
df_scal_clean = df_scal[["method", "scale", "n_cells", "runtime_sec", "peak_mem_mb"]].copy()
df_scal_clean["runtime_min"] = (df_scal_clean["runtime_sec"] / 60).round(2)
df_scal_clean["peak_mem_gb"] = (df_scal_clean["peak_mem_mb"] / 1024).round(2)
df_scal_clean.to_csv(os.path.join(PROJECT_ROOT, "results", "scalability_summary.csv"), index=False)
