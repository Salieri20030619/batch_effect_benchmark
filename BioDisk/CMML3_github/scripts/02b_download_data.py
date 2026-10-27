#!/usr/bin/env python
"""
Step 2b: Download balanced pancreas subset for batch correction benchmark
==========================================================================
Strategy: Stratified sampling by assay to get ~20K cells across 6 platforms.
This ensures cross-technology batch effects are well-represented without
overloading scVI / Seurat training time.

Sampling targets:
  - 10x 3' v2:      6000 cells (pooled from multiple datasets)
  - 10x 3' v3:      4000 cells (pooled from multiple datasets)
  - sci-RNA-seq3:   3500 cells
  - microwell-seq:  3500 cells
  - Smart-seq2:     2544 cells (all - small dataset)
  - CEL-seq2:       2126 cells (all - small dataset)
  Total: ~21,670 cells

Output: data/pancreas_census.h5ad
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Unbuffered prints
import builtins
_print = builtins.print
def log(msg):
    _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

t0 = time.time()
log("Script started")

# --------- Config ---------
CENSUS_VERSION = "2025-01-30"
PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_H5AD = os.path.join(DATA_DIR, "pancreas_census.h5ad")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Target cells per assay (None means take all available)
SAMPLING_TARGETS = {
    "10x 3' v2":     6000,
    "10x 3' v3":     4000,
    "sci-RNA-seq3":  3500,
    "microwell-seq": 3500,
    "Smart-seq2":    None,   # take all (~2544)
    "CEL-seq2":      None,   # take all (~2126)
}

# Reproducibility
RANDOM_SEED = 42

# --------- Load metadata from prior exploration ---------
log("Loading metadata from previous exploration step")
obs_csv = os.path.join(LOG_DIR, "census_pancreas_obs.csv.gz")
if not os.path.exists(obs_csv):
    print(f"[!] Metadata file not found: {obs_csv}")
    print(f"    Please run 02a_explore_census_v2.py first")
    sys.exit(1)

obs_meta = pd.read_csv(obs_csv)
log(f"Loaded {len(obs_meta):,} cells of metadata")

# Keep only normal cells (the project goal is batch effect, not disease)
obs_meta = obs_meta[obs_meta["disease"] == "normal"].copy()
log(f"After filtering to normal: {len(obs_meta):,} cells")

# Sanity check assays
present_assays = set(obs_meta["assay"].unique())
target_assays = set(SAMPLING_TARGETS.keys())
missing = target_assays - present_assays
if missing:
    print(f"[!] These target assays are missing from data: {missing}")
    sys.exit(1)

# --------- Stratified sampling ---------
log("Performing stratified sampling by assay")
rng = np.random.default_rng(RANDOM_SEED)

selected_joinids = []
sampling_report = []

for assay, target_n in SAMPLING_TARGETS.items():
    pool = obs_meta[obs_meta["assay"] == assay]
    available = len(pool)

    if target_n is None or target_n >= available:
        take_n = available
        chosen = pool["soma_joinid"].values
    else:
        take_n = target_n
        # Stratify within assay by (dataset_id, donor_id) so each
        # dataset/donor is represented proportionally
        # Use groupby + sample with weights
        chosen_idx = (
            pool.groupby(["dataset_id", "donor_id"], group_keys=False, observed=True)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(round(target_n * len(g) / available))),
                    random_state=RANDOM_SEED,
                ),
                include_groups=False,
            )
            .index
        )
        # If rounding gave us too many/few, adjust
        chosen_pool = pool.loc[chosen_idx]
        if len(chosen_pool) > target_n:
            chosen_pool = chosen_pool.sample(n=target_n, random_state=RANDOM_SEED)
        chosen = chosen_pool["soma_joinid"].values
        take_n = len(chosen)

    selected_joinids.extend(chosen.tolist())
    sampling_report.append({"assay": assay, "available": available, "sampled": take_n})

sampling_df = pd.DataFrame(sampling_report)
print()
print("Sampling summary:")
print(sampling_df.to_string(index=False))
total_sampled = sampling_df["sampled"].sum()
print(f"\nTotal cells to download: {total_sampled:,}")
print()

# --------- Download expression data via Census ---------
log("Connecting to Census")
import cellxgene_census

census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
log("Census opened")

try:
    log(f"Downloading {total_sampled:,} cells (this may take 5-15 minutes)")
    # Use soma_joinid filter for exact cell selection - much faster than re-running
    # the tissue_general filter
    joinid_list = [int(x) for x in selected_joinids]

    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_coords=joinid_list,
        column_names={
            "obs": [
                "soma_joinid",
                "dataset_id",
                "donor_id",
                "assay",
                "cell_type",
                "disease",
                "suspension_type",
                "sex",
                "development_stage",
                "tissue",
            ],
        },
    )
    log(f"Downloaded AnnData: {adata.shape}")

finally:
    log("Closing census")
    census.close()

# --------- Add convenient batch labels ---------
log("Adding batch label columns")

# Primary batch: assay (technology-level)
adata.obs["batch"] = adata.obs["assay"].astype("category")

# Fine-grained batch: dataset_id (study-level, nested within assay)
adata.obs["batch_dataset"] = adata.obs["dataset_id"].astype("category")

# Combined: assay + donor (most granular, for nested batch analysis)
adata.obs["batch_donor"] = (
    adata.obs["dataset_id"].astype(str) + "_" + adata.obs["donor_id"].astype(str)
).astype("category")

# Ensure cell_type is categorical with stable codes
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

# --------- Quick sanity check ---------
print()
print("=" * 70)
print("Downloaded AnnData summary")
print("=" * 70)
print(f"Shape: {adata.shape}")
print()
print("Batch (assay) distribution:")
print(adata.obs["batch"].value_counts().to_string())
print()
print("Dataset distribution:")
print(adata.obs["batch_dataset"].value_counts().to_string())
print()
print(f"Total unique donors: {adata.obs['donor_id'].nunique()}")
print()
print("Top 15 cell types:")
print(adata.obs["cell_type"].value_counts().head(15).to_string())

# Check raw counts presence
print()
if "raw" in adata.layers or adata.X.dtype in (np.int32, np.int64):
    log("Raw counts present (good for downstream scVI)")
else:
    # Census returns counts in .X by default
    is_int = np.allclose(adata.X.data[:1000] % 1, 0) if hasattr(adata.X, 'data') else False
    log(f"X looks like raw counts: {is_int}")

# --------- Save ---------
log(f"Saving to {OUTPUT_H5AD}")
adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
size_mb = os.path.getsize(OUTPUT_H5AD) / 1024 / 1024
log(f"Saved ({size_mb:.1f} MB)")

print()
print("=" * 70)
print(f"[+] Done in {time.time()-t0:.0f}s total")
print(f"    Output: {OUTPUT_H5AD}")
print(f"    Size:   {size_mb:.1f} MB")
print("=" * 70)
print()
print("Next: preprocessing + batch correction (Step 3)")
