#!/usr/bin/env python
"""
Method 5/5: Seurat - Part 1 of 3
=================================
Export data for R/Seurat.

This script writes:
  1. counts matrix as 10X-format (mtx + features.tsv + barcodes.tsv)
  2. obs (cell metadata) as csv
  3. var (gene metadata) as csv with HVG flag

R script (04e_seurat.R) will:
  - read these files
  - build Seurat v5 object with layers split by batch
  - run IntegrateLayers (CCAIntegration)
  - write the corrected embedding back to csv

Then 04e_seurat_collect.py loads the csv back into the AnnData ecosystem.

Input:  data/pancreas_preprocessed.h5ad
Output: results/seurat_input/
"""

import os
import time
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "seurat_input")
os.makedirs(OUTPUT_DIR, exist_ok=True)

log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")

# --------- Use raw counts (Seurat expects raw counts) ---------
log("Using raw counts from layers['counts']")
counts = adata.layers["counts"]
if not sparse.issparse(counts):
    counts = sparse.csr_matrix(counts)
log(f"counts shape: {counts.shape}, nnz: {counts.nnz}, dtype: {counts.dtype}")

# Seurat expects features in rows, cells in columns
log("Transposing counts to genes x cells (Seurat convention)")
counts_t = counts.T.tocsr()
log(f"Transposed: {counts_t.shape}")

# --------- Write 10X-format ---------
log(f"Writing matrix.mtx to {OUTPUT_DIR}/...")
mtx_path = os.path.join(OUTPUT_DIR, "matrix.mtx")
mmwrite(mtx_path, counts_t)
log(f"  Wrote {os.path.getsize(mtx_path)/1e6:.1f} MB")

log("Writing features.tsv and barcodes.tsv...")
# features.tsv: gene_id<TAB>gene_symbol<TAB>Gene Expression
with open(os.path.join(OUTPUT_DIR, "features.tsv"), "w") as f:
    for g in adata.var_names:
        f.write(f"{g}\t{g}\tGene Expression\n")

# barcodes.tsv: cell barcode
with open(os.path.join(OUTPUT_DIR, "barcodes.tsv"), "w") as f:
    for b in adata.obs_names:
        f.write(f"{b}\n")

# --------- Write metadata ---------
log("Writing obs (cell metadata)...")
# Only keep columns Seurat will actually use
obs_keep = ["batch", "batch_dataset", "donor_id", "assay", "cell_type", "dataset_id"]
obs_keep = [c for c in obs_keep if c in adata.obs.columns]
obs_df = adata.obs[obs_keep].copy()
# Cast all to string to avoid R/Python ontology issues
for col in obs_df.columns:
    obs_df[col] = obs_df[col].astype(str)
obs_df.to_csv(os.path.join(OUTPUT_DIR, "obs.csv"))
log(f"  Wrote {len(obs_df)} rows, cols: {list(obs_df.columns)}")

log("Writing HVG list...")
hvg = adata.var_names[adata.var["highly_variable"]].tolist()
with open(os.path.join(OUTPUT_DIR, "hvg.txt"), "w") as f:
    for g in hvg:
        f.write(g + "\n")
log(f"  Wrote {len(hvg)} HVGs")

_print()
_print("=" * 60)
_print(f"[+] Seurat input prepared in {time.time()-t0:.1f}s")
_print(f"    Files in: {OUTPUT_DIR}/")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, fn)
    _print(f"      {fn:30s} {os.path.getsize(fp)/1e6:8.2f} MB")
_print("=" * 60)
_print()
_print("Next: run the R script in R_4.5.2 env")
_print(f"  conda activate R_4.5.2")
_print(f"  Rscript {PROJECT_ROOT}/scripts/04e_seurat.R")
