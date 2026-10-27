#!/usr/bin/env python
"""
Step 2c (v2): Subsample pancreas cells with proper raw counts preservation
=============================================================================
Fix: ad.concat drops .raw attribute when datasets have heterogeneous raw.
We move raw.X to layers['counts'] PER DATASET (aligned to common genes
after we know what they are) before concat.
"""

import os
import sys
import time
import gc

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

# --------- Config ---------
PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
SRC_DIR = os.path.join(PROJECT_ROOT, "data", "source_h5ads")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_census.h5ad")

DATASETS = [
    "31f657dc-1875-4c4b-a5ca-ce63b3ef3a82",
    "37b21763-7f0f-41ae-9001-60bad6e2841d",
    "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf",
    "3294d050-6eeb-4a00-b24c-71aacc9b777f",
    "66d15835-5dc8-4e96-b0eb-f48971cb65e8",
    "b07e5164-baf6-43d2-bdba-5a249d0da879",
]

SAMPLING_TARGETS = {
    "10x 3' v2":     6000,
    "10x 3' v3":     4000,
    "microwell-seq": 3500,
    "Smart-seq2":    None,
    "CEL-seq2":      None,
}

PANCREAS_TISSUES = {"pancreas", "islet of Langerhans"}
NORMAL_DISEASE_ID = "PATO:0000461"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

log("Script started (v2: with raw counts preservation)")
log(f"Source dir: {SRC_DIR}")
log(f"Output:     {OUTPUT_PATH}")

# --------- Stage 1: filter + sample (using only obs metadata) ---------
log("=" * 60)
log("Stage 1: build filter+sample plan (obs only, no X load yet)")
log("=" * 60)

per_dataset_keep_idx = {}  # dataset_id -> array of local indices to keep

for dataset_id in DATASETS:
    fp = os.path.join(SRC_DIR, f"{dataset_id}.h5ad")
    log(f"Reading obs of {dataset_id[:8]}...")
    
    a_backed = ad.read_h5ad(fp, backed="r")
    obs = a_backed.obs
    mask = (
        obs["tissue"].isin(PANCREAS_TISSUES)
        & (obs["disease_ontology_term_id"] == NORMAL_DISEASE_ID)
        & (obs["is_primary_data"] == True)
    )
    log(f"  Pancreas+normal+primary: {mask.sum():,} / {len(obs):,}")
    per_dataset_keep_idx[dataset_id] = np.where(mask.values)[0]
    
    # Save obs subset for sampling stage (no X load yet)
    a_backed.file.close()

# Aggregate filtered obs from all datasets for sampling
log("Building global obs view for sampling")
global_obs_parts = []
for dataset_id in DATASETS:
    fp = os.path.join(SRC_DIR, f"{dataset_id}.h5ad")
    a_backed = ad.read_h5ad(fp, backed="r")
    keep_idx = per_dataset_keep_idx[dataset_id]
    df = a_backed.obs.iloc[keep_idx][["assay", "donor_id"]].copy()
    df["dataset_id"] = dataset_id
    df["local_idx"] = keep_idx
    global_obs_parts.append(df.reset_index(drop=True))
    a_backed.file.close()
global_obs = pd.concat(global_obs_parts, ignore_index=True)
log(f"Total pancreas pool: {len(global_obs):,}")
log(f"Assays: {global_obs['assay'].value_counts().to_dict()}")

# --------- Stage 2: sample by assay ---------
log("=" * 60)
log("Stage 2: stratified sampling per assay")
log("=" * 60)

selected_per_dataset = {ds: [] for ds in DATASETS}
sampling_report = []

for assay, target_n in SAMPLING_TARGETS.items():
    pool = global_obs[global_obs["assay"] == assay]
    available = len(pool)
    if available == 0:
        continue
    
    if target_n is None or target_n >= available:
        take = pool
    else:
        groups = pool.groupby(["dataset_id", "donor_id"], observed=True)
        parts = []
        for _, g in groups:
            n_g = max(1, min(len(g), int(round(target_n * len(g) / available))))
            parts.append(g.sample(n=n_g, random_state=RANDOM_SEED))
        take = pd.concat(parts)
        if len(take) > target_n:
            take = take.sample(n=target_n, random_state=RANDOM_SEED)
    
    sampling_report.append({"assay": assay, "available": available, "sampled": len(take)})
    # Group by dataset to find local indices
    for ds_id, group in take.groupby("dataset_id", observed=True):
        selected_per_dataset[ds_id].extend(group["local_idx"].tolist())

sampling_df = pd.DataFrame(sampling_report)
_print()
_print("Sampling summary:")
_print(sampling_df.to_string(index=False))
_print()
total_sampled = sampling_df["sampled"].sum()
log(f"Total to assemble: {total_sampled:,}")

# Sort indices for each dataset (helps with backed read)
for ds in selected_per_dataset:
    selected_per_dataset[ds] = np.sort(np.array(selected_per_dataset[ds], dtype=np.int64))
    log(f"  {ds[:8]}: {len(selected_per_dataset[ds]):,} cells selected")

# --------- Stage 3: load selected cells per dataset, KEEPING raw ---------
log("=" * 60)
log("Stage 3: load selected cells with raw counts preserved")
log("=" * 60)

per_dataset_adata = []
for ds_id in DATASETS:
    sel = selected_per_dataset[ds_id]
    if len(sel) == 0:
        continue
    
    fp = os.path.join(SRC_DIR, f"{ds_id}.h5ad")
    log(f"Loading {ds_id[:8]} fully (need raw access)...")
    a_full = ad.read_h5ad(fp)
    a_sub = a_full[sel].copy()
    del a_full
    gc.collect()
    
    log(f"  Loaded {a_sub.n_obs} cells, {a_sub.n_vars} genes")
    
    # CRITICAL FIX: move raw.X to layers['counts'] BEFORE concat
    # The raw of CELLxGENE source h5ad is on the full gene set,
    # while .X is on a filtered gene set. We need to align raw to the
    # same vars as .X so they're consistent.
    if a_sub.raw is not None:
        raw_var_names = a_sub.raw.var_names
        x_var_names = a_sub.var_names
        # Find intersection - genes that are in both
        common = x_var_names.intersection(raw_var_names)
        if len(common) == a_sub.n_vars:
            # All X vars are present in raw - safe to subset raw to X vars
            log(f"  Aligning raw to .var ({a_sub.n_vars} genes)")
            raw_sub = a_sub.raw[:, x_var_names].X
            a_sub.layers["counts"] = raw_sub.copy()
        else:
            log(f"  [warn] X has {a_sub.n_vars} vars but only {len(common)} are in raw")
            # Use intersection - drop X vars not in raw
            common_idx = [i for i, v in enumerate(x_var_names) if v in set(raw_var_names)]
            a_sub = a_sub[:, common_idx].copy()
            a_sub.layers["counts"] = a_sub.raw[:, a_sub.var_names].X.copy()
        # Clear raw to save memory and avoid concat issues
        del a_sub.raw
    else:
        log(f"  [warn] no .raw, assuming .X is raw counts")
        # Quick sanity check
        sample = a_sub.X[:100, :].toarray() if hasattr(a_sub.X, 'toarray') else a_sub.X[:100, :]
        if np.all(np.equal(sample % 1, 0)):
            a_sub.layers["counts"] = a_sub.X.copy()
            log(f"  X is integer-valued, using as counts")
        else:
            log(f"  [!] X is not integer-valued. scVI will fail on this dataset.")
            continue
    
    # Verify counts are integer
    counts_sample = a_sub.layers["counts"][:100, :].toarray() if hasattr(a_sub.layers["counts"], 'toarray') else a_sub.layers["counts"][:100, :]
    is_int = np.all(np.equal(counts_sample % 1, 0))
    log(f"  counts layer integer-valued: {is_int}, max value: {counts_sample.max():.0f}")
    
    a_sub.obs["dataset_id"] = ds_id
    per_dataset_adata.append(a_sub)

log(f"All {len(per_dataset_adata)} datasets prepared")

# --------- Stage 4: concat ---------
log("=" * 60)
log("Stage 4: concatenate")
log("=" * 60)

log("Concatenating with inner join on genes...")
adata = ad.concat(
    per_dataset_adata,
    axis=0,
    join="inner",
    merge="unique",
    label="src",
    keys=[f"src_{i}" for i in range(len(per_dataset_adata))],
    index_unique="-",
)
log(f"Merged shape: {adata.shape}")

# Free memory
del per_dataset_adata
gc.collect()

# Verify counts layer survived concat
if "counts" in adata.layers:
    log(f"[+] counts layer preserved: shape={adata.layers['counts'].shape}")
    counts_sample = adata.layers["counts"][:100, :].toarray() if sparse.issparse(adata.layers["counts"]) else adata.layers["counts"][:100, :]
    log(f"    integer-valued: {np.all(np.equal(counts_sample % 1, 0))}, max: {counts_sample.max():.0f}")
else:
    log("[!] counts layer LOST after concat - this is a bug")

# --------- Stage 5: setup labels ---------
log("=" * 60)
log("Stage 5: batch labels")
log("=" * 60)

adata.obs["batch"] = adata.obs["assay"].astype("category")
adata.obs["batch_dataset"] = adata.obs["dataset_id"].astype("category")
adata.obs["batch_donor"] = (
    adata.obs["dataset_id"].astype(str) + "_" + adata.obs["donor_id"].astype(str)
).astype("category")
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

# --------- Stage 6: summary + save ---------
_print()
_print("Final AnnData:")
_print(f"  Shape: {adata.shape}")
_print(f"  Layers: {list(adata.layers.keys())}")
_print()
_print("Batch (assay) distribution:")
_print(adata.obs["batch"].value_counts().to_string())
_print()
_print("Dataset distribution:")
_print(adata.obs["batch_dataset"].value_counts().to_string())
_print()
_print(f"Unique donors: {adata.obs['donor_id'].nunique()}")
_print(f"Cell types: {adata.obs['cell_type'].nunique()}")
_print()
_print("Top 15 cell types:")
_print(adata.obs["cell_type"].value_counts().head(15).to_string())
_print()

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
