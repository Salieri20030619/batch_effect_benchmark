#!/usr/bin/env python
"""
Method 2/5: ComBat
===================
Empirical Bayes method (Johnson et al. 2007). Corrects the expression
matrix directly using a location-and-scale model per batch.

Pipeline:
  1. Subset to HVGs
  2. ComBat correction on log-normalized X
  3. PCA on corrected expression
  4. UMAP

Input:  data/pancreas_preprocessed.h5ad
Output: results/embeddings/combat.npy
        results/umaps/combat.npy
"""

import os
import time
import numpy as np
import scanpy as sc

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")
EMB_DIR = os.path.join(PROJECT_ROOT, "results", "embeddings")
UMAP_DIR = os.path.join(PROJECT_ROOT, "results", "umaps")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(UMAP_DIR, exist_ok=True)

METHOD = "combat"
N_PCS = 50
N_NEIGHBORS = 15
BATCH_KEY = "batch"

log(f"Method: {METHOD}")

# --------- Load ---------
log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")
log(f"Batches: {adata.obs[BATCH_KEY].value_counts().to_dict()}")

# --------- Subset to HVGs ---------
log("Subsetting to HVGs")
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
log(f"HVG shape: {adata_hvg.shape}")

# --------- Run ComBat ---------
log("Running ComBat (this can take 2-5 min on 17K x 2000)...")
t_method = time.time()

# scanpy's combat expects log-normalized X and dense matrix in some versions
# Convert sparse to dense
if hasattr(adata_hvg.X, "toarray"):
    log("Densifying HVG matrix for ComBat...")
    adata_hvg.X = adata_hvg.X.toarray()

sc.pp.combat(adata_hvg, key=BATCH_KEY)
runtime = time.time() - t_method
log(f"ComBat done in {runtime:.1f}s")
log(f"X after ComBat: min={adata_hvg.X.min():.2f}, max={adata_hvg.X.max():.2f}, "
    f"mean={adata_hvg.X.mean():.3f}")

# --------- PCA on corrected expression ---------
log("Computing PCA on ComBat-corrected expression...")
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="arpack")
embedding = adata_hvg.obsm["X_pca"]
log(f"Embedding shape: {embedding.shape}")

# --------- UMAP ---------
log("Computing UMAP...")
sc.pp.neighbors(adata_hvg, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, use_rep="X_pca")
sc.tl.umap(adata_hvg)
umap_coords = adata_hvg.obsm["X_umap"]
log(f"UMAP shape: {umap_coords.shape}")

# --------- Save ---------
emb_path = os.path.join(EMB_DIR, f"{METHOD}.npy")
umap_path = os.path.join(UMAP_DIR, f"{METHOD}.npy")
np.save(emb_path, embedding)
np.save(umap_path, umap_coords)
log(f"Saved embedding: {emb_path}")
log(f"Saved UMAP: {umap_path}")

# Also save runtime for benchmarking
import json
stats_path = os.path.join(PROJECT_ROOT, "results", f"{METHOD}_stats.json")
with open(stats_path, "w") as f:
    json.dump({"method": METHOD, "runtime_sec": runtime, "embedding_dim": embedding.shape[1]}, f)
log(f"Saved stats: {stats_path}")

_print()
_print("=" * 60)
_print(f"[+] {METHOD} done in {time.time()-t0:.1f}s total ({runtime:.1f}s for ComBat)")
_print("=" * 60)
