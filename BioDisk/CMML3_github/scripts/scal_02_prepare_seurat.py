#!/usr/bin/env python
"""
Prepare Seurat 10X-format input for each scale.
Usage: python scal_02_prepare_seurat.py
"""

import os
import sys
import time
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
SCALES = ["21k", "50k", "100k", "186k"]

for scale in SCALES:
    in_h5ad = os.path.join(PROJECT_ROOT, "data", "scalability", f"pancreas_{scale}.h5ad")
    out_dir = os.path.join(PROJECT_ROOT, "data", "scalability", "seurat_input", scale)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== {scale} ===", flush=True)
    t0 = time.time()
    adata = sc.read_h5ad(in_h5ad)
    print(f"  Loaded {adata.shape} in {time.time()-t0:.1f}s")

    counts = adata.layers["counts"]
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    counts_t = counts.T.tocsr()

    print(f"  Writing matrix.mtx ({counts_t.shape})...")
    mmwrite(os.path.join(out_dir, "matrix.mtx"), counts_t)

    with open(os.path.join(out_dir, "features.tsv"), "w") as f:
        for g in adata.var_names:
            f.write(f"{g}\t{g}\tGene Expression\n")

    with open(os.path.join(out_dir, "barcodes.tsv"), "w") as f:
        for b in adata.obs_names:
            f.write(f"{b}\n")

    obs_keep = ["batch", "donor_id", "assay", "cell_type", "cell_type_major", "dataset_id"]
    obs_keep = [c for c in obs_keep if c in adata.obs.columns]
    obs_df = adata.obs[obs_keep].copy()
    for col in obs_df.columns:
        obs_df[col] = obs_df[col].astype(str)
    obs_df.to_csv(os.path.join(out_dir, "obs.csv"))

    hvg = adata.var_names[adata.var["highly_variable"]].tolist()
    with open(os.path.join(out_dir, "hvg.txt"), "w") as f:
        for g in hvg:
            f.write(g + "\n")

    sizes = {fn: os.path.getsize(os.path.join(out_dir, fn))/1e6
             for fn in os.listdir(out_dir)}
    print(f"  Files written ({time.time()-t0:.1f}s total):", flush=True)
    for fn, sz in sorted(sizes.items()):
        print(f"    {fn}: {sz:.1f} MB")
    del adata

print("\n[+] All done")
