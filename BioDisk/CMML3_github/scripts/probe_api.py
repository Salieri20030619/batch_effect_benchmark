#!/usr/bin/env python
"""
30-second API probe
===================
Goal: figure out correct cellxgene-census API for fetching a small
slice quickly, before we commit to a 21K-cell download.
"""

import time
import inspect

t0 = time.time()
def log(msg): print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

log("Starting API probe")

import cellxgene_census
log(f"cellxgene-census version: {cellxgene_census.__version__}")

# --- Show the actual signature of get_anndata ---
print()
print("=" * 70)
print("get_anndata signature")
print("=" * 70)
sig = inspect.signature(cellxgene_census.get_anndata)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'any'}"
          f"  (default={param.default if param.default != inspect.Parameter.empty else '<required>'})")
print()

# --- Show the docstring ---
print("=" * 70)
print("get_anndata docstring (first 50 lines)")
print("=" * 70)
doc = cellxgene_census.get_anndata.__doc__ or ""
for i, line in enumerate(doc.split("\n")[:60]):
    print(line)
print()

# --- Try a very small real fetch to verify what works ---
print("=" * 70)
print("Trying small fetch: 100 cells via obs_value_filter")
print("=" * 70)

log("Opening census")
census = cellxgene_census.open_soma(census_version="2025-01-30")
log("Census opened")

try:
    log("Fetching: tissue_general='pancreas' AND assay='CEL-seq2' (all ~2126 cells)")
    # Using a small specific assay so this is fast
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=(
            "tissue_general == 'pancreas' "
            "and assay == 'CEL-seq2' "
            "and is_primary_data == True"
        ),
    )
    log(f"[+] Got AnnData: {adata.shape}")
    log(f"    obs columns: {list(adata.obs.columns)[:10]}...")
    log(f"    X dtype: {adata.X.dtype}")
    print()
    print("Cell type counts in this small slice:")
    print(adata.obs["cell_type"].value_counts().head(10).to_string())

finally:
    census.close()
    log("Census closed")

log(f"Total time: {time.time()-t0:.0f}s")
