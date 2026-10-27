#!/usr/bin/env python
"""
Diagnostic version with flushed prints
Locates exactly where the original script hangs.
"""

import sys
import time

# Force unbuffered output so every print appears immediately
print("[t=0.0s] Script started", flush=True)
t0 = time.time()

def log(msg):
    print(f"[t={time.time()-t0:.1f}s] {msg}", flush=True)

log("Importing standard libs...")
import os

log("Importing pandas...")
import pandas as pd

log("Importing cellxgene_census...")
import cellxgene_census
log(f"  cellxgene-census version: {cellxgene_census.__version__}")

log("About to call open_soma() -- this may take 30-60s on first connect")
log("  It downloads ~few MB of TileDB schema metadata")

CENSUS_VERSION = "2025-01-30"

try:
    log("Calling open_soma()...")
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    log("[+] open_soma() returned successfully")
except Exception as e:
    log(f"[!] open_soma() failed: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    log("Listing top-level census structure...")
    keys = list(census.keys())
    log(f"  Top-level keys: {keys}")

    log("Accessing census_data['homo_sapiens']...")
    human = census["census_data"]["homo_sapiens"]
    log(f"  human experiment: {type(human).__name__}")

    log("Counting total cells via obs.count (cheap)...")
    obs_handle = human.obs
    total_count = obs_handle.count
    log(f"  Total human cells in census: {total_count:,}")

    log("Trying small filtered query: tissue_general == 'pancreas' (just count)...")
    test_query = obs_handle.read(
        column_names=["soma_joinid"],
        value_filter="tissue_general == 'pancreas' and is_primary_data == True",
    )
    arrow_tbl = test_query.concat()
    n_pancreas = len(arrow_tbl)
    log(f"  [+] Pancreas cell count: {n_pancreas:,}")

except Exception as e:
    log(f"[!] Query failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    log("Closing census...")
    census.close()
    log("[+] census closed cleanly")

log("All tests passed. The actual exploration script should work.")
