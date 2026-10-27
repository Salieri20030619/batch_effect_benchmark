#!/usr/bin/env python
"""
Step 5 v3: Metrics with cell type harmonization
=================================================
Improvements over v2:
  1. Merge 39 fine-grained cell types into ~14 major categories
     (different datasets use different naming conventions; e.g.
     "pancreatic acinar cell" and "acinar cell" are the same thing).
  2. Exclude "unknown" cells from evaluation (they lack ground truth).
  3. Replace global kbet with kbet_per_label (more meaningful for
     high celltype diversity).
"""

import os
import time
import json
import numpy as np
import pandas as pd
import scanpy as sc

import scib_metrics
from scib_metrics.nearest_neighbors import pynndescent

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_integrated.h5ad")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS = ["unintegrated", "combat", "harmony", "scvi", "seurat_cca"]
BATCH_KEY = "batch"
CELLTYPE_KEY = "cell_type_str"

# --------- Cell type harmonization map ---------
# Maps fine-grained labels (39 types) to major categories (~14 types).
# "unknown" is excluded from evaluation entirely.
CELLTYPE_MAP = {
    # Acinar (pancreatic exocrine)
    "pancreatic acinar cell":               "acinar",
    "acinar cell":                          "acinar",
    "epithelial cell of exocrine pancreas": "acinar",
    "epithelial cell":                      "acinar",
    # Endocrine islet cells
    "pancreatic A cell":                    "alpha",
    "type B pancreatic cell":               "beta",
    "pancreatic D cell":                    "delta",
    "pancreatic PP cell":                   "pp",
    "PP cell":                              "pp",
    "pancreatic epsilon cell":              "epsilon",
    "pancreatic endocrine cell":            "endocrine_other",
    "endocrine cell":                       "endocrine_other",
    # Ductal
    "pancreatic ductal cell":               "ductal",
    # Enteroendocrine (intestinal contamination or shared lineage)
    "type A enteroendocrine cell":          "enteroendocrine",
    "type D enteroendocrine cell":          "enteroendocrine",
    "enterocyte":                           "enteroendocrine",
    # Stromal
    "pancreatic stellate cell":             "stellate",
    "mesenchymal stem cell":                "mesenchymal",
    "mesenchymal cell":                     "mesenchymal",
    "stromal cell":                         "mesenchymal",
    "fibroblast":                           "mesenchymal",
    # Vascular
    "endothelial cell":                     "endothelial",
    "smooth muscle cell":                   "smooth_muscle",
    "cell of skeletal muscle":              "smooth_muscle",
    # Immune
    "T cell":                               "t_cell",
    "B cell":                               "b_cell",
    "macrophage":                           "macrophage",
    "dendritic cell":                       "dendritic",
    "monocyte":                             "monocyte",
    "plasma cell":                          "b_cell",
    "mast cell":                            "immune_rare",
    "neutrophil":                           "immune_rare",
    "professional antigen presenting cell": "immune_rare",
    "cord blood hematopoietic stem cell":   "immune_rare",
    # Other rare types
    "neuron":                               "rare",
    "erythroid lineage cell":               "rare",
    "erythroid progenitor cell":            "rare",
    "primordial germ cell":                 "rare",
    # Will be excluded
    "unknown":                              None,
}

log("=" * 60)
log("Loading integrated AnnData")
log("=" * 60)
adata = sc.read_h5ad(INPUT_PATH)
log(f"Original shape: {adata.shape}")
log(f"Original cell types: {adata.obs[CELLTYPE_KEY].nunique()}")

# --------- Harmonize cell types ---------
log("=" * 60)
log("Harmonizing cell types")
log("=" * 60)

# Check coverage
orig_types = set(adata.obs[CELLTYPE_KEY].unique())
mapped_types = set(CELLTYPE_MAP.keys())
unmapped = orig_types - mapped_types
if unmapped:
    log(f"[!] Unmapped types (will be set to 'rare'): {unmapped}")

# Apply mapping
adata.obs["cell_type_major"] = adata.obs[CELLTYPE_KEY].map(
    lambda x: CELLTYPE_MAP.get(x, "rare")
)

# Show mapping result
_print()
_print("Major cell type distribution:")
_print(adata.obs["cell_type_major"].value_counts(dropna=False).to_string())
_print()

# Drop cells with 'unknown' (None in map)
keep_mask = adata.obs["cell_type_major"].notna()
n_dropped = (~keep_mask).sum()
log(f"Dropping {n_dropped} 'unknown' cells")
adata = adata[keep_mask].copy()
log(f"After drop: {adata.shape}, {adata.obs['cell_type_major'].nunique()} major types")

# Also drop extremely rare types (< 10 cells across all batches; too few for silhouette)
type_counts = adata.obs["cell_type_major"].value_counts()
rare_to_drop = type_counts[type_counts < 10].index.tolist()
if rare_to_drop:
    log(f"Dropping cells of ultra-rare types (< 10 cells): {rare_to_drop}")
    keep = ~adata.obs["cell_type_major"].isin(rare_to_drop)
    adata = adata[keep].copy()
    log(f"After drop: {adata.shape}")

_print()
_print("Final major cell type distribution for evaluation:")
_print(adata.obs["cell_type_major"].value_counts().to_string())
_print()
log(f"Batches: {adata.obs[BATCH_KEY].value_counts().to_dict()}")

from sklearn.preprocessing import LabelEncoder
batch_enc = LabelEncoder().fit_transform(adata.obs[BATCH_KEY].astype(str).values)
celltype_enc = LabelEncoder().fit_transform(adata.obs["cell_type_major"].astype(str).values)
log(f"Batch labels: {len(np.unique(batch_enc))} unique")
log(f"Cell type labels: {len(np.unique(celltype_enc))} unique")

# --------- Compute metrics ---------
all_results = []

for method in METHODS:
    log("=" * 60)
    log(f"Method: {method}")
    log("=" * 60)

    emb_key = f"X_emb_{method}"
    if emb_key not in adata.obsm:
        log(f"  [skip] {emb_key} not found")
        continue

    emb = adata.obsm[emb_key].astype(np.float32)
    log(f"  Embedding shape: {emb.shape}")

    results = {"method": method}

    log("  Building kNN graph (n_neighbors=90)...")
    t_knn = time.time()
    knn_results = pynndescent(emb, n_neighbors=90, random_state=42, n_jobs=4)
    log(f"  kNN done in {time.time()-t_knn:.1f}s")

    # ============ BATCH METRICS ============

    log("  [1/8] batch_asw...")
    t = time.time()
    val = scib_metrics.silhouette_batch(emb, labels=celltype_enc, batch=batch_enc)
    results["batch_asw"] = float(val)
    log(f"        = {val:.4f}  ({time.time()-t:.1f}s)")

    log("  [2/8] ilisi...")
    t = time.time()
    val = scib_metrics.ilisi_knn(knn_results, batch_enc)
    results["ilisi"] = float(val)
    log(f"        = {val:.4f}  ({time.time()-t:.1f}s)")

    log("  [3/8] graph_connectivity...")
    t = time.time()
    val = scib_metrics.graph_connectivity(knn_results, labels=celltype_enc)
    results["graph_connectivity"] = float(val)
    log(f"        = {val:.4f}  ({time.time()-t:.1f}s)")

    # Use kbet_per_label instead of global kbet (more meaningful)
    log("  [4/8] kbet_per_label (per-celltype kbet)...")
    t = time.time()
    try:
        out = scib_metrics.kbet_per_label(
            knn_results, batches=batch_enc, labels=celltype_enc
        )
        if isinstance(out, tuple):
            kbet_val = float(out[0])
        else:
            kbet_val = float(out)
        results["kbet"] = kbet_val
        log(f"        = {kbet_val:.4f}  ({time.time()-t:.1f}s)")
    except Exception as e:
        log(f"        FAILED: {e}")
        results["kbet"] = np.nan

    # ============ BIO METRICS ============

    log("  [5/8] celltype_asw...")
    t = time.time()
    val = scib_metrics.silhouette_label(emb, labels=celltype_enc)
    results["celltype_asw"] = float(val)
    log(f"        = {val:.4f}  ({time.time()-t:.1f}s)")

    log("  [6,7/8] nmi + ari (Leiden)...")
    t = time.time()
    nmi_ari = scib_metrics.nmi_ari_cluster_labels_leiden(
        knn_results, labels=celltype_enc, optimize_resolution=True,
        n_jobs=4, seed=42,
    )
    results["nmi"] = float(nmi_ari["nmi"])
    results["ari"] = float(nmi_ari["ari"])
    log(f"        nmi = {nmi_ari['nmi']:.4f}, ari = {nmi_ari['ari']:.4f}  ({time.time()-t:.1f}s)")

    log("  [8/8] clisi...")
    t = time.time()
    val = scib_metrics.clisi_knn(knn_results, labels=celltype_enc)
    results["clisi"] = float(val)
    log(f"        = {val:.4f}  ({time.time()-t:.1f}s)")

    all_results.append(results)
    log(f"  [done]")

# --------- Aggregate ---------
log("=" * 60)
log("Aggregating")
log("=" * 60)
df = pd.DataFrame(all_results).set_index("method")
_print()
_print("Raw metrics (all already in [0,1], higher = better):")
_print(df.round(4).to_string())

BATCH_METRICS = ["batch_asw", "ilisi", "graph_connectivity", "kbet"]
BIO_METRICS   = ["celltype_asw", "nmi", "ari", "clisi"]

df["batch_score"] = df[BATCH_METRICS].mean(axis=1, skipna=True)
df["bio_score"]   = df[BIO_METRICS].mean(axis=1, skipna=True)
df["overall"]     = 0.4 * df["batch_score"] + 0.6 * df["bio_score"]

df = df.sort_values("overall", ascending=False)

_print()
_print("With aggregate scores:")
_print(df.round(4).to_string())

# --------- Save ---------
csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
df.to_csv(csv_path)
log(f"Saved: {csv_path}")

json_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(json_path, "w") as f:
    json.dump(df.to_dict(orient="index"), f, indent=2)
log(f"Saved: {json_path}")

# Save the major cell type mapping for reproducibility
mapping_path = os.path.join(RESULTS_DIR, "celltype_harmonization.json")
with open(mapping_path, "w") as f:
    json.dump(CELLTYPE_MAP, f, indent=2)
log(f"Saved celltype mapping: {mapping_path}")

_print()
_print("=" * 60)
_print(f"[+] Done in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
_print("=" * 60)
_print()
_print("Final ranking (by Overall score):")
for i, (m, row) in enumerate(df.iterrows(), 1):
    _print(f"  {i}. {m:<15}  overall={row['overall']:.3f}  "
           f"(batch={row['batch_score']:.3f}, bio={row['bio_score']:.3f})")

# Also save a copy of adata with the harmonized labels for plotting later
adata.write_h5ad(
    os.path.join(PROJECT_ROOT, "data", "pancreas_integrated_eval.h5ad"),
    compression="gzip"
)
log("Saved evaluation-ready adata with cell_type_major")
