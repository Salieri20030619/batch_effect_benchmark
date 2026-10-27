#!/usr/bin/env python
"""
Scalability Step 1: Pre-generate 4 datasets at different scales
==================================================================
Subsamples from the 6 source h5ads at 4 target sizes:
  - 21K  (current, smallest)
  - 50K  (2.4x)
  - 100K (4.8x)
  - 186K (all available normal pancreas cells)

Each output is a self-contained h5ad with:
  - X = log-normalized
  - layers['counts'] = raw counts (for scVI)
  - var['highly_variable'] = HVG flag
  - obs['batch'], obs['cell_type_major']

Output: data/scalability/pancreas_{N}.h5ad
"""

import os
import sys
import time
import gc
import json
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
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "scalability")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    "31f657dc-1875-4c4b-a5ca-ce63b3ef3a82",
    "37b21763-7f0f-41ae-9001-60bad6e2841d",
    "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf",
    "3294d050-6eeb-4a00-b24c-71aacc9b777f",
    "66d15835-5dc8-4e96-b0eb-f48971cb65e8",
    "b07e5164-baf6-43d2-bdba-5a249d0da879",
]

PANCREAS_TISSUES = {"pancreas", "islet of Langerhans"}
NORMAL_DISEASE_ID = "PATO:0000461"

# Available cells per assay (from your 02a output)
AVAILABLE = {
    "10x 3' v2":     126305,
    "10x 3' v3":     26781,
    "microwell-seq": 28473,
    "Smart-seq2":    2544,
    "CEL-seq2":      2126,
}
TOTAL_AVAILABLE = sum(AVAILABLE.values())  # 186,229

# Sampling targets per scale (proportional to assay sizes,
# minimum 1000 per assay if possible)
def get_sampling_targets(total_target):
    """Get per-assay sampling targets that sum to total_target,
    proportional to available cells but with minimums for small assays."""
    if total_target >= TOTAL_AVAILABLE:
        return {assay: n for assay, n in AVAILABLE.items()}
    targets = {}
    # First pass: proportional
    for assay, n in AVAILABLE.items():
        targets[assay] = int(total_target * n / TOTAL_AVAILABLE)
    # Adjust by giving smaller assays a minimum, taking from largest
    # Small assays (Smart-seq2, CEL-seq2) should always be fully included if possible
    for small in ["Smart-seq2", "CEL-seq2"]:
        if AVAILABLE[small] <= 3000 and targets[small] < AVAILABLE[small]:
            # Take from the largest assay
            diff = AVAILABLE[small] - targets[small]
            targets[small] = AVAILABLE[small]
            targets["10x 3' v2"] -= diff
    return targets

SCALES = {
    "21k":  21000,
    "50k":  50000,
    "100k": 100000,
    "186k": TOTAL_AVAILABLE,  # all available
}

RANDOM_SEED = 42

# --------- Cell type harmonization (same map as 05_metrics_v3) ---------
CELLTYPE_MAP = {
    "pancreatic acinar cell": "acinar", "acinar cell": "acinar",
    "epithelial cell of exocrine pancreas": "acinar", "epithelial cell": "acinar",
    "pancreatic A cell": "alpha", "type B pancreatic cell": "beta",
    "pancreatic D cell": "delta", "pancreatic PP cell": "pp", "PP cell": "pp",
    "pancreatic epsilon cell": "epsilon",
    "pancreatic endocrine cell": "endocrine_other", "endocrine cell": "endocrine_other",
    "pancreatic ductal cell": "ductal",
    "type A enteroendocrine cell": "enteroendocrine",
    "type D enteroendocrine cell": "enteroendocrine",
    "enterocyte": "enteroendocrine",
    "pancreatic stellate cell": "stellate",
    "mesenchymal stem cell": "mesenchymal", "mesenchymal cell": "mesenchymal",
    "stromal cell": "mesenchymal", "fibroblast": "mesenchymal",
    "endothelial cell": "endothelial",
    "smooth muscle cell": "smooth_muscle", "cell of skeletal muscle": "smooth_muscle",
    "T cell": "t_cell", "B cell": "b_cell", "plasma cell": "b_cell",
    "macrophage": "macrophage", "dendritic cell": "dendritic", "monocyte": "monocyte",
    "mast cell": "immune_rare", "neutrophil": "immune_rare",
    "professional antigen presenting cell": "immune_rare",
    "cord blood hematopoietic stem cell": "immune_rare",
    "neuron": "rare", "erythroid lineage cell": "rare",
    "erythroid progenitor cell": "rare", "primordial germ cell": "rare",
    "unknown": None,
}

# --------- Step 1: Load all source data + harmonize raw counts ---------
log("=" * 60)
log("Step 1: Load and harmonize 6 source datasets")
log("=" * 60)

all_subs = []
for ds_id in DATASETS:
    fp = os.path.join(SRC_DIR, f"{ds_id}.h5ad")
    log(f"Loading {ds_id[:8]}...")

    # Backed mode to filter
    a_backed = ad.read_h5ad(fp, backed="r")
    obs = a_backed.obs
    mask = (
        obs["tissue"].isin(PANCREAS_TISSUES)
        & (obs["disease_ontology_term_id"] == NORMAL_DISEASE_ID)
        & (obs["is_primary_data"] == True)
    )
    keep_idx = np.where(mask.values)[0]
    a_backed.file.close()
    log(f"  {len(keep_idx):,} pancreas cells")

    # Load fully and subset
    a_full = ad.read_h5ad(fp)
    a_sub = a_full[keep_idx].copy()
    del a_full
    gc.collect()

    # Move raw to layers['counts']
    if a_sub.raw is not None:
        common = a_sub.var_names.intersection(a_sub.raw.var_names)
        if len(common) == a_sub.n_vars:
            a_sub.layers["counts"] = a_sub.raw[:, a_sub.var_names].X.copy()
        else:
            common_idx = [i for i, v in enumerate(a_sub.var_names) if v in set(a_sub.raw.var_names)]
            a_sub = a_sub[:, common_idx].copy()
            a_sub.layers["counts"] = a_sub.raw[:, a_sub.var_names].X.copy()
        del a_sub.raw
    else:
        # X is already raw counts (case for 66d15835)
        a_sub.layers["counts"] = a_sub.X.copy()

    a_sub.obs["dataset_id"] = ds_id
    all_subs.append(a_sub)

log(f"Loaded {len(all_subs)} datasets")

# --------- Step 2: Build per-scale obs metadata for sampling decisions ---------
log("=" * 60)
log("Step 2: Generate sampling indices for each scale")
log("=" * 60)

# Build a global obs dataframe with (adata_idx, local_idx) pointers
obs_global = []
for ai, a in enumerate(all_subs):
    df = a.obs[["assay", "donor_id", "dataset_id"]].copy()
    df["adata_idx"] = ai
    df["local_idx"] = np.arange(len(df))
    obs_global.append(df.reset_index(drop=True))
obs_global = pd.concat(obs_global, ignore_index=True)
log(f"Total pool: {len(obs_global):,}")

rng = np.random.default_rng(RANDOM_SEED)

# For each scale, decide which cells to sample
scale_indices = {}
for scale_name, target_total in SCALES.items():
    targets = get_sampling_targets(target_total)
    log(f"Scale {scale_name}: target = {target_total:,}")
    for assay, t in targets.items():
        log(f"  {assay}: target {t} / available {AVAILABLE[assay]}")

    # Sample per assay, stratified by (dataset_id, donor_id)
    selected = []
    for assay, target_n in targets.items():
        pool = obs_global[obs_global["assay"] == assay]
        if target_n >= len(pool):
            take = pool
        else:
            # Stratify by donor
            groups = pool.groupby(["dataset_id", "donor_id"], observed=True)
            parts = []
            for _, g in groups:
                n_g = max(1, min(len(g), int(round(target_n * len(g) / len(pool)))))
                parts.append(g.sample(n=n_g, random_state=RANDOM_SEED))
            take = pd.concat(parts)
            if len(take) > target_n:
                take = take.sample(n=target_n, random_state=RANDOM_SEED)
        selected.append(take)
    selected_df = pd.concat(selected)
    scale_indices[scale_name] = selected_df
    log(f"  Final: {len(selected_df):,} cells")

# --------- Step 3: Find common genes across all datasets ---------
log("=" * 60)
log("Step 3: Find common genes across 6 datasets")
log("=" * 60)
common_genes = set(all_subs[0].var_names)
for a in all_subs[1:]:
    common_genes &= set(a.var_names)
common_genes = sorted(common_genes)
log(f"Common genes: {len(common_genes)}")

# --------- Step 4: For each scale, build merged adata + preprocess ---------
for scale_name, target_total in SCALES.items():
    log("=" * 60)
    log(f"Step 4: Building scale '{scale_name}' ({target_total:,} cells)")
    log("=" * 60)

    sel_df = scale_indices[scale_name]
    # Group by adata_idx for efficient subsetting
    parts = []
    for ai, group in sel_df.groupby("adata_idx", observed=True):
        local_idx = np.sort(group["local_idx"].values)
        a_part = all_subs[ai][local_idx].copy()
        # Subset to common genes
        a_part = a_part[:, common_genes].copy()
        parts.append(a_part)

    log("Concatenating...")
    adata = ad.concat(parts, axis=0, join="inner", merge="unique",
                      label="src", keys=[str(i) for i in range(len(parts))],
                      index_unique="-")
    del parts
    gc.collect()
    log(f"  Merged: {adata.shape}")

    # Harmonize cell types
    adata.obs["cell_type_major"] = adata.obs["cell_type"].astype(str).map(
        lambda x: CELLTYPE_MAP.get(x, "rare")
    )
    keep = adata.obs["cell_type_major"].notna()
    adata = adata[keep].copy()

    # Add batch labels
    adata.obs["batch"] = adata.obs["assay"].astype("category")
    adata.obs["dataset_id"] = adata.obs["dataset_id"].astype("category")

    # ---- Preprocess: QC + normalize + HVG + PCA ----
    log("  QC filtering...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    log(f"  After QC: {adata.shape}")

    # CRITICAL: reset X to raw counts before normalizing.
    # The Census source h5ads have heterogeneous X content - some are
    # already log-normalized, others are raw. We always start from counts.
    log("  Resetting X to raw counts from layers['counts']")
    adata.X = adata.layers["counts"].copy()

    # Sanity check: X should be integer-valued now
    sample = adata.X[:50, :100].toarray() if hasattr(adata.X, "toarray") else adata.X[:50, :100]
    is_int = np.all(np.equal(sample % 1, 0))
    log(f"  X integer check: {is_int}, min={sample.min():.1f}, max={sample.max():.1f}")
    if not is_int:
        log("  [!] X not integer; counts may be corrupted")

    log("  Normalizing X...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG
    log("  HVG selection...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                layer="counts", batch_key="batch", subset=False)

    # PCA on HVG (needed for Harmony and as baseline)
    log("  PCA on HVG...")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50, svd_solver="arpack")
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    del adata_hvg
    gc.collect()

    # Save
    out_path = os.path.join(OUT_DIR, f"pancreas_{scale_name}.h5ad")
    log(f"  Saving to {out_path}...")
    adata.write_h5ad(out_path, compression="gzip")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    log(f"  Saved ({size_mb:.1f} MB)")

    del adata
    gc.collect()

_print()
_print("=" * 60)
_print(f"[+] All scales prepared in {time.time()-t0:.0f}s")
_print("=" * 60)
_print()
_print("Output files:")
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    _print(f"  {fn:30s} {os.path.getsize(fp)/1024/1024:8.1f} MB")
_print()
_print("Next: run scalability benchmark")
