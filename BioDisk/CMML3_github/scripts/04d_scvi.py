#!/usr/bin/env python
"""
Method 4/5: scVI
=================
Lopez et al. 2018, Nat Methods. Variational autoencoder that models
raw counts with a zero-inflated negative binomial likelihood, with
batch as a conditioning variable. Outputs a latent embedding.

Pipeline:
  1. Subset to HVGs (scVI is efficient on selected features)
  2. Set up scVI with batch_key
  3. Train (10-30 min on CPU for 17K cells)
  4. Get latent representation
  5. UMAP

Input:  data/pancreas_preprocessed.h5ad (needs layers['counts'])
Output: results/embeddings/scvi.npy
        results/umaps/scvi.npy
"""

import os
import time
import numpy as np
import scanpy as sc

import builtins
_print = builtins.print
t0 = time.time()
def log(msg): _print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

PROJECT_ROOT = "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pancreas_preprocessed.h5ad")
EMB_DIR = os.path.join(PROJECT_ROOT, "results", "embeddings")
UMAP_DIR = os.path.join(PROJECT_ROOT, "results", "umaps")
MODEL_DIR = os.path.join(PROJECT_ROOT, "results", "scvi_model")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(UMAP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

METHOD = "scvi"
BATCH_KEY = "batch"
N_NEIGHBORS = 15
N_LATENT = 30           # default for scVI
N_HIDDEN = 128          # default
N_LAYERS = 2            # default
MAX_EPOCHS = 100        # default; will early stop if converged
RANDOM_SEED = 42

log(f"Method: {METHOD}")

# --------- Reproducibility ---------
import torch
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------- Load ---------
log("Loading preprocessed AnnData...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"Shape: {adata.shape}")
log(f"counts layer present: {'counts' in adata.layers}")
log(f"Batches: {adata.obs[BATCH_KEY].value_counts().to_dict()}")

# --------- Subset to HVGs ---------
log("Subsetting to HVGs for efficiency")
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
log(f"HVG shape: {adata_hvg.shape}")

# --------- Setup scVI ---------
log("Importing scvi-tools...")
import scvi
log(f"scvi-tools version: {scvi.__version__}")

log("Setting up AnnData for scVI...")
scvi.model.SCVI.setup_anndata(
    adata_hvg,
    layer="counts",          # raw counts
    batch_key=BATCH_KEY,
)

# --------- Build and train model ---------
log("Building scVI model...")
model = scvi.model.SCVI(
    adata_hvg,
    n_latent=N_LATENT,
    n_hidden=N_HIDDEN,
    n_layers=N_LAYERS,
    gene_likelihood="zinb",  # zero-inflated negative binomial
)
log(f"Model: {model}")

log(f"Training for max {MAX_EPOCHS} epochs (with early stopping)...")
log("This is the slowest step; expect 10-30 min on CPU")
t_method = time.time()

model.train(
    max_epochs=MAX_EPOCHS,
    early_stopping=True,
    early_stopping_patience=20,
    accelerator="cpu",
    check_val_every_n_epoch=1,
)
runtime = time.time() - t_method
log(f"Training done in {runtime:.1f}s ({runtime/60:.1f} min)")

# --------- Save model for reproducibility ---------
log(f"Saving model to {MODEL_DIR}...")
model.save(MODEL_DIR, overwrite=True)

# --------- Get latent representation ---------
log("Extracting latent representation...")
embedding = model.get_latent_representation()
log(f"Latent shape: {embedding.shape}")

# --------- UMAP ---------
log("Computing UMAP from scVI latent...")
adata_hvg.obsm["X_scvi"] = embedding
sc.pp.neighbors(adata_hvg, n_neighbors=N_NEIGHBORS, use_rep="X_scvi")
sc.tl.umap(adata_hvg)
umap_coords = adata_hvg.obsm["X_umap"]
log(f"UMAP shape: {umap_coords.shape}")

# --------- Save ---------
emb_path = os.path.join(EMB_DIR, f"{METHOD}.npy")
umap_path = os.path.join(UMAP_DIR, f"{METHOD}.npy")
np.save(emb_path, embedding)
np.save(umap_path, umap_coords)
log(f"Saved embedding: {emb_path}")
log(f"Saved UMAP: {umap_path}")

import json
stats_path = os.path.join(PROJECT_ROOT, "results", f"{METHOD}_stats.json")
with open(stats_path, "w") as f:
    json.dump({
        "method": METHOD,
        "runtime_sec": runtime,
        "embedding_dim": embedding.shape[1],
        "n_epochs_trained": model.history["elbo_train"].shape[0] if "elbo_train" in model.history else None,
    }, f)
log(f"Saved stats: {stats_path}")

_print()
_print("=" * 60)
_print(f"[+] {METHOD} done in {time.time()-t0:.1f}s total ({runtime:.1f}s for training)")
_print("=" * 60)
