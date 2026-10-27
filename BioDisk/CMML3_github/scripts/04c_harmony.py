#!/usr/bin/env python
"""
Method 3/5: Harmony (v2 - direct harmonypy call)
=================================================
Bypass the buggy scanpy.external wrapper. Call harmonypy directly.
"""

import os
import time
import json
import numpy as np
import scanpy as sc
import harmonypy as hm

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

METHOD = "harmony"
N_NEIGHBORS = 15
BATCH_KEY = "batch"

log(f"Method: {METHOD}")

# --------- Load ---------
log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")
log(f"PCA available: X_pca shape={adata.obsm['X_pca'].shape}")

# --------- Run Harmony directly via harmonypy ---------
log(f"harmonypy version: {hm.__version__}")
log("Running Harmony via direct harmonypy.run_harmony() call...")
t_method = time.time()

# harmonypy.run_harmony signature:
#   run_harmony(data_mat, meta_data, vars_use, ...)
# data_mat: cells x PCs (n_obs x n_pcs)
harmony_out = hm.run_harmony(
    data_mat=adata.obsm["X_pca"],
    meta_data=adata.obs,
    vars_use=[BATCH_KEY],
    max_iter_harmony=30,
    random_state=42,
)
runtime = time.time() - t_method
log(f"Harmony done in {runtime:.1f}s")

# Inspect output shape to know what to do
log(f"harmony_out.Z_corr.shape: {harmony_out.Z_corr.shape}")
log(f"harmony_out.Z_corr.T.shape: {harmony_out.Z_corr.T.shape}")
log(f"Expected: ({adata.n_obs}, 50)")

# Figure out the right orientation
n_cells = adata.n_obs
n_pcs = adata.obsm["X_pca"].shape[1]

if harmony_out.Z_corr.shape == (n_cells, n_pcs):
    log("Z_corr is already (cells, PCs) - use as-is")
    embedding = harmony_out.Z_corr.copy()
elif harmony_out.Z_corr.shape == (n_pcs, n_cells):
    log("Z_corr is (PCs, cells) - transpose")
    embedding = harmony_out.Z_corr.T.copy()
else:
    raise ValueError(f"Unexpected Z_corr shape: {harmony_out.Z_corr.shape}")

log(f"Final embedding shape: {embedding.shape}")
assert embedding.shape == (n_cells, n_pcs), f"Shape mismatch: {embedding.shape}"

# --------- UMAP ---------
log("Computing UMAP on corrected embedding...")
adata.obsm["X_pca_harmony"] = embedding
sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca_harmony")
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

stats_path = os.path.join(PROJECT_ROOT, "results", f"{METHOD}_stats.json")
with open(stats_path, "w") as f:
    json.dump({"method": METHOD, "runtime_sec": runtime, "embedding_dim": embedding.shape[1]}, f)
log(f"Saved stats: {stats_path}")

_print()
_print("=" * 60)
_print(f"[+] {METHOD} done in {time.time()-t0:.1f}s total ({runtime:.1f}s for Harmony)")
_print("=" * 60)
