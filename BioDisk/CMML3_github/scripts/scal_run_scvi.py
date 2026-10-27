#!/usr/bin/env python
"""
Scalability runner: scVI
Usage: python scal_run_scvi.py <scale_name>
"""

import sys
import os
import time
import json
import resource
import numpy as np
import torch
import scanpy as sc

torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
SCALE = sys.argv[1]
INPUT = os.path.join(PROJECT_ROOT, "data", "scalability", f"pancreas_{SCALE}.h5ad")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "scalability")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, f"scvi_{SCALE}.json")

print(f"=== scVI scalability test: scale={SCALE} ===", flush=True)

t0 = time.time()
print(f"[t={time.time()-t0:.1f}s] Loading {INPUT}", flush=True)
adata = sc.read_h5ad(INPUT)
print(f"[t={time.time()-t0:.1f}s] Shape: {adata.shape}", flush=True)
n_cells = adata.n_obs

# Subset to HVGs (standard scVI workflow)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
print(f"[t={time.time()-t0:.1f}s] HVG: {adata_hvg.shape}", flush=True)

mem_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

import scvi
print(f"[t={time.time()-t0:.1f}s] Setting up scVI", flush=True)
scvi.model.SCVI.setup_anndata(adata_hvg, layer="counts", batch_key="batch")

print(f"[t={time.time()-t0:.1f}s] Training scVI...", flush=True)
t_method = time.time()
model = scvi.model.SCVI(adata_hvg, n_latent=30, n_hidden=128, n_layers=2,
                         gene_likelihood="zinb")
model.train(
    max_epochs=100,
    early_stopping=True,
    early_stopping_patience=20,
    accelerator="cpu",
    check_val_every_n_epoch=1,
)
runtime = time.time() - t_method
print(f"[t={time.time()-t0:.1f}s] Done in {runtime:.1f}s", flush=True)

# Force get latent to ensure full pipeline tested
embedding = model.get_latent_representation()
print(f"Latent: {embedding.shape}", flush=True)

peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

stats = {
    "method": "scvi",
    "scale": SCALE,
    "n_cells": int(n_cells),
    "runtime_sec": float(runtime),
    "peak_mem_mb": float(peak_mem_mb),
    "mem_after_load_mb": float(mem_after_load),
}
with open(OUT, "w") as f:
    json.dump(stats, f, indent=2)
print(f"Saved: {OUT}", flush=True)
print(json.dumps(stats, indent=2), flush=True)
