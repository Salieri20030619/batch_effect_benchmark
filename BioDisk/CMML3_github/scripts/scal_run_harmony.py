#!/usr/bin/env python
"""
Scalability runner: Harmony
Usage: python scal_run_harmony.py <scale_name>
Example: python scal_run_harmony.py 50k

Records: runtime, peak memory (resident set size).
Output: results/scalability/harmony_{scale}.json
"""

import sys
import os
import time
import json
import resource
import numpy as np
import scanpy as sc
import harmonypy as hm

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
SCALE = sys.argv[1]
INPUT = os.path.join(PROJECT_ROOT, "data", "scalability", f"pancreas_{SCALE}.h5ad")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "scalability")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, f"harmony_{SCALE}.json")

print(f"=== Harmony scalability test: scale={SCALE} ===", flush=True)

t0 = time.time()
print(f"[t={time.time()-t0:.1f}s] Loading {INPUT}", flush=True)
adata = sc.read_h5ad(INPUT)
print(f"[t={time.time()-t0:.1f}s] Shape: {adata.shape}", flush=True)
n_cells = adata.n_obs

# Record memory after load
mem_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

print(f"[t={time.time()-t0:.1f}s] Running Harmony...", flush=True)
t_method = time.time()
harmony_out = hm.run_harmony(
    data_mat=adata.obsm["X_pca"],
    meta_data=adata.obs,
    vars_use=["batch"],
    max_iter_harmony=30,
    random_state=42,
)
runtime = time.time() - t_method
print(f"[t={time.time()-t0:.1f}s] Done in {runtime:.1f}s", flush=True)

# Record peak memory
peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

stats = {
    "method": "harmony",
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
