#!/usr/bin/env python
"""
Method 1/5: Unintegrated baseline
===================================
This is the "no batch correction" reference. We just use the PCA
computed during preprocessing (no integration applied).

Input:  data/pancreas_preprocessed.h5ad
Output: results/embeddings/unintegrated.npy
        results/umaps/unintegrated.npy
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
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")
EMB_DIR = os.path.join(PROJECT_ROOT, "results", "embeddings")
UMAP_DIR = os.path.join(PROJECT_ROOT, "results", "umaps")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(UMAP_DIR, exist_ok=True)

METHOD = "unintegrated"
N_PCS = 50
N_NEIGHBORS = 15

log(f"Method: {METHOD}")

# --------- Load ---------
log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")

# --------- Use existing PCA as embedding ---------
log("Using preprocessing PCA as embedding (no correction)")
embedding = adata.obsm["X_pca"].copy()
log(f"Embedding shape: {embedding.shape}")

# UMAP was already computed during preprocessing
log("UMAP was already computed in preprocessing step (obsm['X_umap_raw'])")
umap_coords = adata.obsm["X_umap_raw"].copy()

# --------- Save ---------
emb_path = os.path.join(EMB_DIR, f"{METHOD}.npy")
umap_path = os.path.join(UMAP_DIR, f"{METHOD}.npy")
np.save(emb_path, embedding)
np.save(umap_path, umap_coords)
log(f"Saved embedding: {emb_path}")
log(f"Saved UMAP: {umap_path}")

_print()
_print("=" * 60)
_print(f"[+] {METHOD} done in {time.time()-t0:.1f}s")
_print(f"    Embedding: {embedding.shape}")
_print(f"    UMAP: {umap_coords.shape}")
_print("=" * 60)
