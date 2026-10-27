#!/usr/bin/env python
"""
Collect all method embeddings into a single AnnData
=====================================================
Reads results/embeddings/{method}.npy for each method and stores
into obsm slots of the preprocessed adata. Also stores UMAPs.

Input:  data/pancreas_preprocessed.h5ad
        results/embeddings/*.npy
        results/umaps/*.npy
Output: data/pancreas_integrated.h5ad
"""

import os
import time
import json
import numpy as np
import scanpy as sc

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_integrated.h5ad")
EMB_DIR = os.path.join(PROJECT_ROOT, "results", "embeddings")
UMAP_DIR = os.path.join(PROJECT_ROOT, "results", "umaps")

METHODS = ["unintegrated", "combat", "harmony", "scvi", "seurat_cca"]

log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")

# --------- Load each method's embedding and UMAP ---------
for method in METHODS:
    emb_path = os.path.join(EMB_DIR, f"{method}.npy")
    umap_path = os.path.join(UMAP_DIR, f"{method}.npy")

    if not os.path.exists(emb_path):
        log(f"[skip] {method}: no embedding file at {emb_path}")
        continue
    if not os.path.exists(umap_path):
        log(f"[skip] {method}: no umap file at {umap_path}")
        continue

    emb = np.load(emb_path)
    umap = np.load(umap_path)
    log(f"[{method:15s}] embedding: {emb.shape}, umap: {umap.shape}")

    if emb.shape[0] != adata.n_obs:
        log(f"  [!] cell count mismatch ({emb.shape[0]} vs {adata.n_obs}); skipping")
        continue

    adata.obsm[f"X_emb_{method}"] = emb.astype(np.float32)
    adata.obsm[f"X_umap_{method}"] = umap.astype(np.float32)

# --------- Aggregate runtime stats ---------
log("Aggregating runtime stats...")
stats_all = {}
for method in METHODS:
    stats_file = os.path.join(PROJECT_ROOT, "results", f"{method}_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            stats_all[method] = json.load(f)

adata.uns["benchmark_stats"] = stats_all

_print()
_print("Method summary:")
_print(f"  {'method':<15} {'embedding':<15} {'umap':<15} {'runtime_sec':<12}")
for method in METHODS:
    emb_key = f"X_emb_{method}"
    umap_key = f"X_umap_{method}"
    if emb_key in adata.obsm:
        rt = stats_all.get(method, {}).get("runtime_sec", None)
        rt_str = f"{rt:.1f}" if rt is not None else "n/a"
        _print(f"  {method:<15} {str(adata.obsm[emb_key].shape):<15} "
               f"{str(adata.obsm[umap_key].shape):<15} {rt_str:<12}")
    else:
        _print(f"  {method:<15} MISSING")

# --------- Save ---------
_print()
log(f"Saving to {OUTPUT_PATH}...")
adata.write_h5ad(OUTPUT_PATH, compression="gzip")
size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
log(f"Saved ({size_mb:.1f} MB)")

_print()
_print("=" * 60)
_print(f"[+] Done in {time.time()-t0:.1f}s")
_print(f"    Output: {OUTPUT_PATH}")
_print("=" * 60)
