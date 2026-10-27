#!/usr/bin/env python
"""
Step 2c: Subsample pancreas cells from local h5ad files and merge
==================================================================
Reads the 6 downloaded source h5ad files, filters each to
pancreas-related cells (tissue in [pancreas, islet of Langerhans]),
normal disease, primary data only. Then stratified-samples ~18K
cells by assay and writes a single merged AnnData.

Output: data/pancreas_census.h5ad

This step is purely local - no network calls.
Expected runtime: 3-8 minutes (mostly loading the 600K-cell file).
"""

import os
import sys
import time
import gc

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

# --------- Config ---------
PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
SRC_DIR = os.path.join(PROJECT_ROOT, "data", "source_h5ads")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_census.h5ad")

# Source datasets to load (file basename + intended sampling)
DATASETS = [
    "31f657dc-1875-4c4b-a5ca-ce63b3ef3a82",
    "37b21763-7f0f-41ae-9001-60bad6e2841d",
    "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf",
    "3294d050-6eeb-4a00-b24c-71aacc9b777f",
    "66d15835-5dc8-4e96-b0eb-f48971cb65e8",
    "b07e5164-baf6-43d2-bdba-5a249d0da879",
]

# Target cells per assay (None = take all)
SAMPLING_TARGETS = {
    "10x 3' v2":     6000,
    "10x 3' v3":     4000,
    "microwell-seq": 3500,
    "Smart-seq2":    None,
    "CEL-seq2":      None,
}

# Filter criteria
PANCREAS_TISSUES = {"pancreas", "islet of Langerhans"}
NORMAL_DISEASE_ID = "PATO:0000461"  # 'normal' in PATO ontology

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

log("Script started")
log(f"Source dir: {SRC_DIR}")
log(f"Output:     {OUTPUT_PATH}")

# --------- Stage 1: per-dataset filter + sample ---------
# Strategy: load each h5ad fully (or partly), keep only pancreas+normal+primary
# cells, drop large X to save memory while we collect indices, then in stage 2
# we re-read the X for the selected cells.

log("=" * 60)
log("Stage 1: filter each dataset down to pancreas cells")
log("=" * 60)

# Per-dataset filtered AnnData (small - only pancreas cells, raw counts kept)
filtered_adatas = []

for dataset_id in DATASETS:
    fp = os.path.join(SRC_DIR, f"{dataset_id}.h5ad")
    log(f"Loading {dataset_id[:8]}...")
    
    # Load in backed mode first to check size
    a_backed = ad.read_h5ad(fp, backed="r")
    total = a_backed.n_obs
    log(f"  Total cells in file: {total:,}")
    
    # Build filter mask from obs (cheap - obs is already in memory in backed mode)
    obs = a_backed.obs
    mask = (
        obs["tissue"].isin(PANCREAS_TISSUES)
        & (obs["disease_ontology_term_id"] == NORMAL_DISEASE_ID)
        & (obs["is_primary_data"] == True)
    )
    n_keep = mask.sum()
    log(f"  After filter (pancreas+normal+primary): {n_keep:,}")
    
    if n_keep == 0:
        log(f"  [skip] no cells kept")
        a_backed.file.close()
        continue
    
    # Close backed handle, then load only filtered cells into memory
    a_backed.file.close()
    
    # Read just the rows we need (this loads X for those rows)
    keep_idx = np.where(mask.values)[0]
    log(f"  Loading {len(keep_idx):,} cells fully into memory...")
    a_full = ad.read_h5ad(fp)
    a_sub = a_full[keep_idx].copy()
    del a_full
    gc.collect()
    
    log(f"  Subset shape: {a_sub.shape}")
    log(f"  Memory check: X dtype={a_sub.X.dtype}, nnz~{a_sub.X.nnz if hasattr(a_sub.X, 'nnz') else 'dense'}")
    
    # Add provenance
    a_sub.obs["dataset_id"] = dataset_id
    
    filtered_adatas.append(a_sub)
    log(f"  [done]")

log(f"Stage 1 done. {len(filtered_adatas)} datasets loaded.")

# --------- Stage 2: stratified sampling per assay ---------
log("=" * 60)
log("Stage 2: stratified sampling by assay")
log("=" * 60)

# First, aggregate obs across all filtered datasets to get global view
all_obs_list = []
for i, a in enumerate(filtered_adatas):
    df = a.obs[["dataset_id", "assay", "donor_id"]].copy()
    df["adata_idx"] = i
    df["local_idx"] = np.arange(len(df))
    all_obs_list.append(df)
global_obs = pd.concat(all_obs_list, ignore_index=True)
log(f"Total pancreas cells pooled: {len(global_obs):,}")
log(f"Assays available: {global_obs['assay'].value_counts().to_dict()}")

# Sample by assay
selected_indices = []  # list of (adata_idx, local_idx) tuples
sampling_report = []
rng = np.random.default_rng(RANDOM_SEED)

for assay, target_n in SAMPLING_TARGETS.items():
    pool = global_obs[global_obs["assay"] == assay]
    available = len(pool)
    
    if available == 0:
        log(f"  [warn] no cells for assay '{assay}'")
        continue
    
    if target_n is None or target_n >= available:
        take = pool
    else:
        # Stratify within assay by (dataset_id, donor_id)
        groups = pool.groupby(["dataset_id", "donor_id"], observed=True)
        n_groups = len(groups)
        # Sample proportionally to group size
        per_group = []
        for (ds, donor), g in groups:
            n_g = int(round(target_n * len(g) / available))
            n_g = max(1, n_g)
            n_g = min(n_g, len(g))
            sampled = g.sample(n=n_g, random_state=RANDOM_SEED)
            per_group.append(sampled)
        take = pd.concat(per_group)
        # Adjust if we overshot
        if len(take) > target_n:
            take = take.sample(n=target_n, random_state=RANDOM_SEED)
    
    selected_indices.extend(list(zip(take["adata_idx"].values, take["local_idx"].values)))
    sampling_report.append({"assay": assay, "available": available, "sampled": len(take)})

sampling_df = pd.DataFrame(sampling_report)
_print()
_print("Sampling summary:")
_print(sampling_df.to_string(index=False))
_print()
total_sampled = sampling_df["sampled"].sum()
log(f"Total cells to assemble: {total_sampled:,}")

# --------- Stage 3: assemble selected cells into a single AnnData ---------
log("=" * 60)
log("Stage 3: assemble merged AnnData")
log("=" * 60)

# Group selected indices by adata_idx for efficient subsetting
sel_by_adata = {}
for ai, li in selected_indices:
    sel_by_adata.setdefault(ai, []).append(li)

# Subset each filtered AnnData and collect
parts = []
for ai, locs in sel_by_adata.items():
    a = filtered_adatas[ai][np.array(locs)].copy()
    log(f"  Dataset {DATASETS[ai][:8]}: kept {a.n_obs} cells")
    parts.append(a)

# Free original filtered_adatas memory
for i in range(len(filtered_adatas)):
    filtered_adatas[i] = None
del filtered_adatas
gc.collect()

# Concatenate. Use 'inner' join on var to keep only common genes.
log("Concatenating (inner join on genes)...")
adata = ad.concat(
    parts,
    axis=0,
    join="inner",
    merge="unique",
    label="source_idx",
    keys=[f"src_{i}" for i in range(len(parts))],
    index_unique="-",
)
log(f"Merged shape: {adata.shape} (cells x common genes)")

# Free parts memory
del parts
gc.collect()

# --------- Stage 4: post-merge setup ---------
log("=" * 60)
log("Stage 4: setup batch labels and counts layer")
log("=" * 60)

# Census source h5ads typically have raw counts in raw.X (or X if no raw),
# and normalized data in X. Let's check.
log(f"X dtype: {adata.X.dtype}")
sample_vals = adata.X[:5, :100].toarray() if hasattr(adata.X, 'toarray') else adata.X[:5, :100]
log(f"X first 5x100 head: min={sample_vals.min():.3f}, max={sample_vals.max():.3f}")

# If X looks like normalized (floats < 50), check raw
if adata.raw is not None:
    log(f"Raw available: {adata.raw.X.dtype}, shape={adata.raw.shape}")
    raw_sample = adata.raw.X[:5, :100].toarray() if hasattr(adata.raw.X, 'toarray') else adata.raw.X[:5, :100]
    log(f"Raw head: min={raw_sample.min():.3f}, max={raw_sample.max():.3f}")
    
    # Census conventions: raw.X has raw counts on all genes.
    # We need to align raw to the same vars as X (common genes after inner join).
    common_genes = adata.var_names
    raw_aligned = adata.raw[:, common_genes].X
    adata.layers["counts"] = raw_aligned.copy()
    log(f"Stored raw counts in layers['counts']")
else:
    # No raw available - X itself might be raw counts
    is_int = np.all(np.equal(sample_vals % 1, 0))
    if is_int:
        adata.layers["counts"] = adata.X.copy()
        log("X looks like raw counts; copied to layers['counts']")
    else:
        log("[!] WARNING: no raw counts available. scVI may not work.")

# Batch labels
adata.obs["batch"] = adata.obs["assay"].astype("category")
adata.obs["batch_dataset"] = adata.obs["dataset_id"].astype("category")
adata.obs["batch_donor"] = (
    adata.obs["dataset_id"].astype(str) + "_" + adata.obs["donor_id"].astype(str)
).astype("category")

# Clean up cell_type as categorical (will be the ground truth label)
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

# --------- Stage 5: summary + save ---------
log("=" * 60)
log("Stage 5: summary")
log("=" * 60)

_print()
_print("Final AnnData:")
_print(f"  Shape: {adata.shape}")
_print()
_print("Batch (assay) distribution:")
_print(adata.obs["batch"].value_counts().to_string())
_print()
_print("Dataset distribution:")
_print(adata.obs["batch_dataset"].value_counts().to_string())
_print()
_print(f"Unique donors: {adata.obs['donor_id'].nunique()}")
_print()
_print(f"Cell types: {adata.obs['cell_type'].nunique()}")
_print("Top 15 cell types:")
_print(adata.obs["cell_type"].value_counts().head(15).to_string())
_print()

# Save
log(f"Saving to {OUTPUT_PATH}...")
adata.write_h5ad(OUTPUT_PATH, compression="gzip")
size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
log(f"Saved ({size_mb:.1f} MB)")

_print()
_print("=" * 60)
_print(f"[+] Done in {time.time()-t0:.0f}s")
_print(f"    Output: {OUTPUT_PATH}")
_print(f"    Size:   {size_mb:.1f} MB")
_print("=" * 60)
