#!/bin/bash
# ============================================================
# Master scalability benchmark
# Runs all 3 methods × 4 scales = 12 tests sequentially
# Each test in a fresh subprocess so memory is fully released.
# 
# Estimated total: 6-10 hours.
# Recommended to run in tmux/screen for resilience.
# ============================================================

set -u  # Catch unset vars but NOT non-zero exits (don't use -e here - we want to keep going)

PROJECT_ROOT="/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
cd "${PROJECT_ROOT}"

SCALES=("21k" "50k" "100k" "186k")
LOG_DIR="${PROJECT_ROOT}/logs/scalability"
mkdir -p "${LOG_DIR}"

# Paths to env executables
PY_BIN="/mnt/BioDisk/wangmingyuan/miniconda3/envs/batch_bench/bin/python"
RSCRIPT_BIN="/mnt/BioDisk/wangmingyuan/miniconda3/envs/R_4.5.2/bin/Rscript"

echo "============================================================"
echo "Scalability benchmark - $(date)"
echo "============================================================"

run_one() {
    local method="$1"
    local scale="$2"
    local log_file="${LOG_DIR}/${method}_${scale}.log"

    echo ""
    echo ">>> [$(date +%H:%M)] Starting ${method} @ scale ${scale}"
    echo "    Log: ${log_file}"
    local t_start=$(date +%s)

    case "${method}" in
        harmony)
            ${PY_BIN} -u scripts/scal_run_harmony.py "${scale}" > "${log_file}" 2>&1
            ;;
        scvi)
            ${PY_BIN} -u scripts/scal_run_scvi.py "${scale}" > "${log_file}" 2>&1
            ;;
        seurat)
            ${RSCRIPT_BIN} scripts/scal_run_seurat.R "${scale}" > "${log_file}" 2>&1
            ;;
    esac

    local exit_code=$?
    local t_end=$(date +%s)
    local elapsed=$((t_end - t_start))

    if [ ${exit_code} -eq 0 ]; then
        echo "    [+] Done in ${elapsed}s (exit ${exit_code})"
    else
        echo "    [!] FAILED in ${elapsed}s (exit ${exit_code})"
        echo "    See log: ${log_file}"
    fi
}

# ---------- Run all 12 tests ----------
# Order: Harmony (fast) first for all scales, then scVI, then Seurat
# This way you see Harmony results across all scales quickly,
# and slower methods can run overnight.

for scale in "${SCALES[@]}"; do
    run_one harmony "${scale}"
done

for scale in "${SCALES[@]}"; do
    run_one scvi "${scale}"
done

for scale in "${SCALES[@]}"; do
    run_one seurat "${scale}"
done

echo ""
echo "============================================================"
echo "All done - $(date)"
echo "============================================================"
ls -lh "${PROJECT_ROOT}/results/scalability/"
