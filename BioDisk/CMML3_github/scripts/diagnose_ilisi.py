#!/usr/bin/env python
"""Diagnose iLISI distribution for each method"""
import os
import numpy as np
import scanpy as sc
from scib_metrics.nearest_neighbors import pynndescent
from scib_metrics import lisi_knn
from sklearn.preprocessing import LabelEncoder
import pandas as pd

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_integrated_eval.h5ad")

adata = sc.read_h5ad(INPUT_PATH)
batch_enc = LabelEncoder().fit_transform(adata.obs["batch"].astype(str).values)
n_batches = len(np.unique(batch_enc))
print(f"n_batches = {n_batches}")
print(f"Batch sizes: {pd.Series(batch_enc).value_counts().to_dict()}")
print()

for method in ["unintegrated", "harmony", "scvi", "seurat_cca"]:
    emb_key = f"X_emb_{method}"
    if emb_key not in adata.obsm:
        continue
    emb = adata.obsm[emb_key].astype(np.float32)
    knn = pynndescent(emb, n_neighbors=90, random_state=42, n_jobs=4)
    
    # Per-cell LISI
    lisi = lisi_knn(knn, batch_enc)
    
    # Per-cell scaled iLISI
    lisi_scaled = (lisi - 1) / (n_batches - 1)
    
    print(f"=== {method} ===")
    print(f"  Raw LISI    - min: {lisi.min():.3f}, "
          f"q25: {np.percentile(lisi, 25):.3f}, "
          f"median: {np.median(lisi):.3f}, "
          f"mean: {np.mean(lisi):.3f}, "
          f"q75: {np.percentile(lisi, 75):.3f}, "
          f"max: {lisi.max():.3f}")
    print(f"  Scaled iLISI- min: {lisi_scaled.min():.3f}, "
          f"q25: {np.percentile(lisi_scaled, 25):.3f}, "
          f"median: {np.median(lisi_scaled):.3f}, "
          f"mean: {np.mean(lisi_scaled):.3f}, "
          f"q75: {np.percentile(lisi_scaled, 75):.3f}, "
          f"max: {lisi_scaled.max():.3f}")
    
    # Distribution check: how many cells have iLISI > 0.5?
    well_mixed_pct = (lisi_scaled > 0.5).mean() * 100
    print(f"  % cells with scaled iLISI > 0.5: {well_mixed_pct:.1f}%")
    print()
