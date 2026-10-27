#!/usr/bin/env Rscript
# ============================================================
# Scalability runner: Seurat v5 CCA Integration
# Usage: Rscript scal_run_seurat.R <scale_name>
# ============================================================

suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
SCALE <- args[1]

PROJECT_ROOT <- "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
SEURAT_INPUT_DIR <- file.path(PROJECT_ROOT, "data", "scalability", "seurat_input", SCALE)
OUT_DIR <- file.path(PROJECT_ROOT, "results", "scalability")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
OUT_JSON <- file.path(OUT_DIR, sprintf("seurat_%s.json", SCALE))

N_DIMS <- 30
set.seed(42)

t0 <- Sys.time()
log_msg <- function(msg) {
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("[t=%.1fs] %s\n", elapsed, msg))
    flush.console()
}

log_msg(sprintf("=== Seurat scalability test: scale=%s ===", SCALE))

# Load 10X-format data
log_msg("Loading data...")
mtx <- readMM(file.path(SEURAT_INPUT_DIR, "matrix.mtx"))
features <- read.table(file.path(SEURAT_INPUT_DIR, "features.tsv"),
                       sep="\t", stringsAsFactors=FALSE, header=FALSE)
barcodes <- readLines(file.path(SEURAT_INPUT_DIR, "barcodes.tsv"))
rownames(mtx) <- features$V1
colnames(mtx) <- barcodes
mtx <- as(mtx, "CsparseMatrix")

obs <- read.csv(file.path(SEURAT_INPUT_DIR, "obs.csv"),
                row.names = 1, stringsAsFactors = FALSE)
hvg <- readLines(file.path(SEURAT_INPUT_DIR, "hvg.txt"))

n_cells <- ncol(mtx)
log_msg(sprintf("Matrix: %d genes x %d cells", nrow(mtx), n_cells))

# Build Seurat object
log_msg("Creating Seurat object")
seu <- CreateSeuratObject(counts = mtx, meta.data = obs)
seu[["RNA"]] <- split(seu[["RNA"]], f = seu@meta.data[["batch"]])
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, nfeatures = length(hvg), verbose = FALSE)
VariableFeatures(seu) <- intersect(hvg, rownames(seu))
seu <- ScaleData(seu, features = VariableFeatures(seu), verbose = FALSE)
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = N_DIMS, verbose = FALSE)

log_msg("Running IntegrateLayers (CCA)...")
t_method <- Sys.time()
seu <- IntegrateLayers(
    object = seu, method = CCAIntegration,
    orig.reduction = "pca", new.reduction = "integrated.cca",
    features = VariableFeatures(seu), dims = 1:N_DIMS,
    verbose = FALSE
)
runtime <- as.numeric(difftime(Sys.time(), t_method, units = "secs"))
log_msg(sprintf("Done in %.1fs", runtime))

# Get peak memory (from /proc/self/status, in KB)
get_peak_mem_mb <- function() {
    status <- readLines("/proc/self/status")
    vm_peak <- grep("^VmPeak:", status, value = TRUE)
    if (length(vm_peak) > 0) {
        kb <- as.numeric(gsub("[^0-9]", "", vm_peak))
        return(kb / 1024)
    }
    return(NA)
}

peak_mem_mb <- get_peak_mem_mb()

stats <- list(
    method = "seurat_cca",
    scale = SCALE,
    n_cells = n_cells,
    runtime_sec = runtime,
    peak_mem_mb = peak_mem_mb
)
write_json(stats, OUT_JSON, pretty = TRUE, auto_unbox = TRUE)
log_msg(sprintf("Saved: %s", OUT_JSON))
print(stats)
