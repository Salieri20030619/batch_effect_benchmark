#!/bin/bash
# ============================================================
# Direct download of source h5ad files from CELLxGENE Census S3
# This bypasses the slow get_anndata() Python API.
#
# Strategy: download 6 source h5ad files for the 6 datasets that
# contributed pancreas cells. After download, run 02c_subsample.py
# to load them, filter to pancreas + normal, and stratified-sample
# ~21K cells.
# ============================================================

set -e

PROJECT_ROOT="/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
DATA_DIR="${PROJECT_ROOT}/data/source_h5ads"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Census release version (must match what 02a explored)
CENSUS_VERSION="2025-01-30"
S3_BASE="https://cellxgene-data-public.s3.us-west-2.amazonaws.com/cell-census/${CENSUS_VERSION}/h5ads"

# Six datasets that contain pancreas cells (from 02a exploration)
# Format: dataset_id|n_pancreas_cells|primary_assay
DATASETS=(
    "31f657dc-1875-4c4b-a5ca-ce63b3ef3a82|79367|10x 3' v2"
    "37b21763-7f0f-41ae-9001-60bad6e2841d|69645|10x 3' v3 + 10x 3' v2"
    "f7c1c579-2dc0-47e2-ba19-8165c5a0e353|45653|sci-RNA-seq3"
    "2adb1f8a-a6b1-4909-8ee8-484814e2d4bf|28473|microwell-seq"
    "3294d050-6eeb-4a00-b24c-71aacc9b777f|26474|10x 3' v2"
    "6725ee8e-ef5b-4e68-8901-61bd14a1fe73|24793|10x 3' v3 + 10x 3' v2"
    "53d208b0-2cfd-4366-9866-c3c6114081bc|14140|10x 3' v3"
    "66d15835-5dc8-4e96-b0eb-f48971cb65e8|2544|Smart-seq2"
    "b07e5164-baf6-43d2-bdba-5a249d0da879|2126|CEL-seq2"
)

echo "============================================================"
echo "Downloading 9 source h5ad files"
echo "Target dir: ${DATA_DIR}"
echo "S3 base:    ${S3_BASE}"
echo "============================================================"
echo ""

# Pick downloader: aria2c (best - multithread) > wget (good) > curl (fallback)
if command -v aria2c &> /dev/null; then
    DOWNLOADER="aria2c"
    echo "Using aria2c (multithread)"
elif command -v wget &> /dev/null; then
    DOWNLOADER="wget"
    echo "Using wget"
else
    DOWNLOADER="curl"
    echo "Using curl"
fi
echo ""

TOTAL=${#DATASETS[@]}
i=0
for entry in "${DATASETS[@]}"; do
    i=$((i+1))
    IFS='|' read -r dataset_id n_cells assay <<< "$entry"
    url="${S3_BASE}/${dataset_id}.h5ad"
    out="${DATA_DIR}/${dataset_id}.h5ad"

    echo "[${i}/${TOTAL}] ${dataset_id}"
    echo "    Cells: ${n_cells} (pancreas only, full dataset may be larger)"
    echo "    Assay: ${assay}"

    # Skip if already downloaded
    if [ -f "$out" ]; then
        size_mb=$(du -m "$out" | cut -f1)
        echo "    [skip] Already downloaded (${size_mb} MB)"
        echo ""
        continue
    fi

    # HEAD request first to get expected size and verify URL exists
    echo "    HEAD request..."
    content_length=$(curl -sIL "$url" 2>/dev/null | grep -i "^content-length:" | tail -1 | awk '{print $2}' | tr -d '\r')
    if [ -n "$content_length" ]; then
        size_mb=$(echo "scale=1; $content_length / 1048576" | bc 2>/dev/null || echo "?")
        echo "    Expected size: ${size_mb} MB"
    else
        echo "    [!] Could not get Content-Length"
    fi

    # Download
    case "$DOWNLOADER" in
        aria2c)
            aria2c -x 8 -s 8 --console-log-level=warn --summary-interval=10 \
                   -d "$DATA_DIR" -o "${dataset_id}.h5ad" "$url"
            ;;
        wget)
            wget --progress=dot:giga -O "$out" "$url"
            ;;
        curl)
            curl -L --progress-bar -o "$out" "$url"
            ;;
    esac

    actual_mb=$(du -m "$out" | cut -f1)
    echo "    [+] Downloaded ${actual_mb} MB"
    echo ""
done

echo "============================================================"
echo "Download summary"
echo "============================================================"
ls -lh "${DATA_DIR}"
echo ""

total_mb=$(du -sm "${DATA_DIR}" | cut -f1)
echo "Total downloaded: ${total_mb} MB"
echo ""
echo "Next: run 02c_subsample.py to load these, filter to pancreas,"
echo "      stratified-sample 21K cells, and save as a single h5ad"
