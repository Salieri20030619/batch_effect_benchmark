#!/usr/bin/env python
"""
Step 3: Preprocessing
======================
QC -> normalize -> log1p -> HVG -> scale -> PCA

After this script, the AnnData has:
  - adata.layers["counts"]:  raw integer counts (for scVI)
  - adata.X:                 log-normalized (for ComBat, Harmony, Seurat CCA)
  - adata.var["highly_variable"]: HVG flag (top 2000)
  - adata.obsm["X_pca"]:    50-dim PCA (unintegrated baseline)
  - adata.obsm["X_umap_raw"]: UMAP from raw PCA (for "before correction" plot)

Input:  data/pancreas_census.h5ad
Output: data/pancreas_preprocessed.h5ad
"""

import os
import time
import numpy as np
import scanpy as sc

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

# --------- Config ---------
PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_census.h5ad")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")

# QC thresholds (relatively permissive - tech platforms differ a lot)
MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3

# HVG
N_HVG = 2000

# PCA
N_PCS = 50

# UMAP
N_NEIGHBORS = 15

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white')

log("=" * 60)
log("Loading data")
log("=" * 60)
adata = sc.read_h5ad(INPUT_PATH)
log(f"Input shape: {adata.shape}")
log(f"Layers: {list(adata.layers.keys())}")
log(f"X dtype: {adata.X.dtype}")

# --------- Stage 1: QC ---------
log("=" * 60)
log("Stage 1: QC filtering")
log("=" * 60)

# Always filter on raw counts, not on normalized X
# Temporarily swap X to counts for QC, then swap back
adata_qc = adata.copy()
adata_qc.X = adata_qc.layers["counts"]

n_before = adata_qc.n_obs
sc.pp.filter_cells(adata_qc, min_genes=MIN_GENES_PER_CELL)
log(f"Cell filter (>= {MIN_GENES_PER_CELL} genes): {n_before} -> {adata_qc.n_obs}")

n_before = adata_qc.n_vars
sc.pp.filter_genes(adata_qc, min_cells=MIN_CELLS_PER_GENE)
log(f"Gene filter (>= {MIN_CELLS_PER_GENE} cells): {n_before} -> {adata_qc.n_vars}")

# Annotate QC metrics for reporting (not for filtering)
adata_qc.var["mt"] = adata_qc.var_names.str.startswith(("MT-", "mt-"))
sc.pp.calculate_qc_metrics(
    adata_qc, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

log(f"QC metrics annotated. n_genes_by_counts range: "
    f"{adata_qc.obs['n_genes_by_counts'].min()} - {adata_qc.obs['n_genes_by_counts'].max()}")
log(f"pct_counts_mt range: {adata_qc.obs['pct_counts_mt'].min():.1f}% - "
    f"{adata_qc.obs['pct_counts_mt'].max():.1f}%")

# Per-batch QC summary
_print()
_print("Per-batch QC summary (median):")
qc_summary = adata_qc.obs.groupby("batch", observed=True).agg(
    n_cells=("batch", "count"),
    median_n_genes=("n_genes_by_counts", "median"),
    median_total_counts=("total_counts", "median"),
    median_pct_mt=("pct_counts_mt", "median"),
).round(1)
_print(qc_summary.to_string())
_print()

adata = adata_qc
del adata_qc

# --------- Stage 2: Normalize + log1p ---------
log("=" * 60)
log("Stage 2: Normalize and log1p")
log("=" * 60)

# Make sure we start from raw counts
adata.X = adata.layers["counts"].copy()

# Normalize to 10,000 counts per cell
sc.pp.normalize_total(adata, target_sum=1e4)
log("Normalized total counts to 10,000 per cell")

# log1p
sc.pp.log1p(adata)
log("log1p transformed")

# Sanity check X is no longer integer
sample = adata.X[:5, :100].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :100]
log(f"X after log1p: min={sample.min():.3f}, max={sample.max():.3f}")

# --------- Stage 3: HVG selection ---------
log("=" * 60)
log("Stage 3: HVG selection")
log("=" * 60)

# Use seurat_v3 flavor on raw counts (best practice for cross-batch HVG)
# Pass counts layer explicitly
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=N_HVG,
    flavor="seurat_v3",
    layer="counts",
    batch_key="batch",
    subset=False,  # keep all genes, just flag HVG
)
n_hvg = adata.var["highly_variable"].sum()
log(f"Selected {n_hvg} HVGs (target {N_HVG})")
log(f"HVGs detected in multiple batches:")
if "highly_variable_nbatches" in adata.var.columns:
    nbatch_counts = adata.var.loc[adata.var["highly_variable"], "highly_variable_nbatches"].value_counts().sort_index()
    _print(nbatch_counts.to_string())

# --------- Stage 4: Scale + PCA (on HVG only) ---------
log("=" * 60)
log("Stage 4: Scale + PCA on HVG subset")
log("=" * 60)

# Build a HVG-only view for PCA without modifying main adata
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
log(f"HVG subset shape: {adata_hvg.shape}")

# Scale (zero mean, unit variance) - clipped to 10 std to avoid outlier domination
sc.pp.scale(adata_hvg, max_value=10)
log("Scaled HVG matrix")

# PCA
sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="arpack")
log(f"PCA done, n_comps={N_PCS}")

# Copy PCA back to main adata
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
# Variance explained for reference
adata.uns["pca"] = adata_hvg.uns["pca"]

# --------- Stage 5: Baseline UMAP (no batch correction) ---------
log("=" * 60)
log("Stage 5: Baseline neighbors + UMAP (no correction)")
log("=" * 60)

sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
sc.tl.umap(adata)
adata.obsm["X_umap_raw"] = adata.obsm["X_umap"].copy()
log("Baseline UMAP saved to obsm['X_umap_raw']")
# Remove X_umap so the slot is free for later methods to fill
del adata.obsm["X_umap"]

# --------- Stage 6: Save ---------
log("=" * 60)
log("Stage 6: Save")
log("=" * 60)

# Set a string version of cell_type for ground truth (more portable)
adata.obs["cell_type_str"] = adata.obs["cell_type"].astype(str)

# Print final summary
_print()
_print("Final preprocessed AnnData:")
_print(f"  Shape: {adata.shape}")
_print(f"  Layers: {list(adata.layers.keys())}")
_print(f"  obsm: {list(adata.obsm.keys())}")
_print(f"  HVGs: {adata.var['highly_variable'].sum()}")
_print()
_print("Cell counts per batch:")
_print(adata.obs["batch"].value_counts().to_string())
_print()

log(f"Saving to {OUTPUT_PATH}...")
adata.write_h5ad(OUTPUT_PATH, compression="gzip")
size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
log(f"Saved ({size_mb:.1f} MB)")

_print()
_print("=" * 60)
_print(f"[+] Done in {time.time()-t0:.0f}s")
_print(f"    Output: {OUTPUT_PATH}")
_print("=" * 60)
