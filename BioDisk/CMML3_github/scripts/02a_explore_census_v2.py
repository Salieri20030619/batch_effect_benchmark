#!/usr/bin/env python
"""
Step 2a (v2): Explore pancreas data using fast obs.read() path
================================================================
Uses the same low-level API as the diagnostic script which finished
in 127 seconds. Streams metadata in chunks instead of materializing
all 290K rows at once through the get_obs() helper.
"""

import os
import sys
import time
import pandas as pd

# Force unbuffered output
print = lambda *a, **kw: __builtins__.print(*a, **{**kw, "flush": True})

t0 = time.time()
def log(msg):
    print(f"[t={time.time()-t0:.1f}s] {msg}")

# --------- Config ---------
CENSUS_VERSION = "2025-01-30"
PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Explore Census pancreas data (fast streaming version)")
print(f"Census version: {CENSUS_VERSION}")
print("=" * 70)

import cellxgene_census
log(f"cellxgene-census version: {cellxgene_census.__version__}")

log("Opening census...")
census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
log("Census opened")

try:
    human = census["census_data"]["homo_sapiens"]
    log("Accessed homo_sapiens experiment")

    # --- Use obs.read() directly (proven to work in diagnose script) ---
    log("Querying pancreas cells (this took 127s in the diagnostic run)...")
    reader = human.obs.read(
        column_names=[
            "soma_joinid",
            "dataset_id",
            "donor_id",
            "assay",
            "cell_type",
            "disease",
            "suspension_type",
            "sex",
        ],
        value_filter="tissue_general == 'pancreas' and is_primary_data == True",
    )

    log("Concatenating chunks into Arrow table...")
    arrow_table = reader.concat()
    log(f"Arrow table: {len(arrow_table):,} rows")

    log("Converting to pandas DataFrame...")
    obs_df = arrow_table.to_pandas()
    log(f"DataFrame shape: {obs_df.shape}")
    print()

finally:
    log("Closing census")
    census.close()

# --------- Analysis (all in-memory, fast) ---------
print()
print("=" * 70)
print("Summary statistics")
print("=" * 70)

# 1. Datasets
print()
print("[1/5] Datasets containing pancreas cells:")
dataset_summary = (
    obs_df.groupby("dataset_id")
    .size()
    .reset_index(name="n_cells")
    .sort_values("n_cells", ascending=False)
)
print(f"      Total datasets: {len(dataset_summary)}")
print()
print(dataset_summary.to_string(index=False))

# 2. Assays
print()
print("[2/5] Assays (technical platforms):")
assay_summary = obs_df["assay"].value_counts()
print(assay_summary.to_string())

# 3. Donors per (dataset, assay)
print()
print("[3/5] Donors per (dataset, assay):")
donor_summary = obs_df.groupby(["dataset_id", "assay"])["donor_id"].nunique()
print(f"      Total unique donors: {obs_df['donor_id'].nunique()}")
print(donor_summary.to_string())

# 4. Diseases
print()
print("[4/5] Disease distribution:")
disease_summary = obs_df["disease"].value_counts()
print(disease_summary.to_string())

# 5. Cell types (top 20)
print()
print("[5/5] Cell type distribution (top 20):")
celltype_summary = obs_df["cell_type"].value_counts().head(20)
print(celltype_summary.to_string())

# --- Save full obs ---
obs_csv = os.path.join(OUTPUT_DIR, "census_pancreas_obs.csv.gz")
log(f"Saving full obs to {obs_csv}...")
obs_df.to_csv(obs_csv, index=False, compression="gzip")
log("Saved")

# --- Recommendation ---
print()
print("=" * 70)
print("Recommendation for benchmark subset")
print("=" * 70)

normal_obs = obs_df[obs_df["disease"] == "normal"]
print(f"  Normal-only cells: {len(normal_obs):,}")

normal_assay_counts = normal_obs["assay"].value_counts()
print(f"  Assays in normal cells:")
print(normal_assay_counts.to_string())
print()

# Pick eligible assays with at least 500 normal cells
eligible_assays = normal_assay_counts[normal_assay_counts >= 500]
print(f"  Assays with >= 500 normal cells: {len(eligible_assays)}")

normal_dataset_counts = (
    normal_obs.groupby(["dataset_id", "assay"])
    .size()
    .reset_index(name="n_cells")
    .sort_values("n_cells", ascending=False)
)
print()
print("  Top (dataset, assay) combinations for normal cells:")
print(normal_dataset_counts.head(15).to_string(index=False))

print()
log(f"All done in {time.time()-t0:.0f}s total")
