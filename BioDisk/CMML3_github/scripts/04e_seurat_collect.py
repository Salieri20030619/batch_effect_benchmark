#!/usr/bin/env python
"""
Method 5/5: Seurat - Part 3 of 3
=================================
Load Seurat CCA embedding from R output, run UMAP, save in our
standard format alongside the other methods.

Input:  results/seurat_output/seurat_cca_embedding.csv (from R)
        data/pancreas_preprocessed.h5ad (for cell ordering and UMAP graph)
Output: results/embeddings/seurat_cca.npy
        results/umaps/seurat_cca.npy
"""

import os
import time
import numpy as np
import pandas as pd
import scanpy as sc

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")
SEURAT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "seurat_output")
EMB_DIR = os.path.join(PROJECT_ROOT, "results", "embeddings")
UMAP_DIR = os.path.join(PROJECT_ROOT, "results", "umaps")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(UMAP_DIR, exist_ok=True)

METHOD = "seurat_cca"
N_NEIGHBORS = 15

log(f"Method: {METHOD}")

# --------- Load Seurat embedding ---------
emb_csv = os.path.join(SEURAT_OUT_DIR, "seurat_cca_embedding.csv")
log(f"Loading Seurat embedding: {emb_csv}")
emb_df = pd.read_csv(emb_csv, index_col=0)
log(f"Embedding shape: {emb_df.shape}")
log(f"Index sample: {list(emb_df.index[:3])}")

# --------- Load preprocessed AnnData for cell ordering ---------
log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")

# --------- Align embedding to adata's cell order ---------
log("Aligning Seurat embedding to AnnData cell order")
missing = set(adata.obs_names) - set(emb_df.index)
extra = set(emb_df.index) - set(adata.obs_names)
if missing:
    log(f"  [!] {len(missing)} cells in adata are missing from Seurat output")
if extra:
    log(f"  [!] {len(extra)} cells in Seurat output not in adata")

# Reorder to match adata
emb_aligned = emb_df.loc[adata.obs_names]
embedding = emb_aligned.values.astype(np.float32)
log(f"Aligned embedding shape: {embedding.shape}")

# --------- UMAP ---------
log("Computing UMAP from Seurat embedding...")
adata.obsm["X_seurat"] = embedding
sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_seurat")
sc.tl.umap(adata)
umap_coords = adata.obsm["X_umap"]
log(f"UMAP shape: {umap_coords.shape}")

# --------- Save ---------
emb_path = os.path.join(EMB_DIR, f"{METHOD}.npy")
umap_path = os.path.join(UMAP_DIR, f"{METHOD}.npy")
np.save(emb_path, embedding)
np.save(umap_path, umap_coords)
log(f"Saved embedding: {emb_path}")
log(f"Saved UMAP: {umap_path}")

_print()
_print("=" * 60)
_print(f"[+] {METHOD} loaded and UMAP-ed in {time.time()-t0:.1f}s")
_print("=" * 60)
