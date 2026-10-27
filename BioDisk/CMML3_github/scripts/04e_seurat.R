#!/usr/bin/env Rscript
# ============================================================
# Method 5/5: Seurat v5 CCA Integration
# Run with: conda activate R_4.5.2 && Rscript 04e_seurat.R
#
# Strategy: Use Seurat v5's IntegrateLayers() with CCAIntegration.
# This is the v5 replacement for the old anchor-based workflow.
# ============================================================

suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
})

cat("Seurat version:", as.character(packageVersion("Seurat")), "\n")
cat("R version:", R.version.string, "\n\n")

# --------- Config ---------
PROJECT_ROOT <- "/mnt/BioDisk/wangmingyuan/batch_effect_benchmark"
INPUT_DIR  <- file.path(PROJECT_ROOT, "results", "seurat_input")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "results", "seurat_output")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

BATCH_KEY <- "batch"   # column in obs to split layers by
N_DIMS    <- 30        # CCA components
RANDOM_SEED <- 42
set.seed(RANDOM_SEED)

t0 <- Sys.time()
log_msg <- function(msg) {
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("[t=%.1fs] %s\n", elapsed, msg))
    flush.console()
}

# --------- Load 10X-format counts ---------
log_msg("Loading counts matrix")
mtx <- readMM(file.path(INPUT_DIR, "matrix.mtx"))
features <- read.table(file.path(INPUT_DIR, "features.tsv"), sep="\t",
                       stringsAsFactors=FALSE, header=FALSE)
barcodes <- readLines(file.path(INPUT_DIR, "barcodes.tsv"))

rownames(mtx) <- features$V1
colnames(mtx) <- barcodes
mtx <- as(mtx, "CsparseMatrix")
log_msg(sprintf("Counts matrix: %d genes x %d cells", nrow(mtx), ncol(mtx)))

# --------- Load metadata ---------
log_msg("Loading metadata")
obs <- read.csv(file.path(INPUT_DIR, "obs.csv"), row.names = 1,
                stringsAsFactors = FALSE)
log_msg(sprintf("obs: %d rows, columns: %s",
                nrow(obs), paste(colnames(obs), collapse=", ")))
log_msg("Batches:")
print(table(obs[[BATCH_KEY]]))

# --------- Load HVG list ---------
hvg <- readLines(file.path(INPUT_DIR, "hvg.txt"))
log_msg(sprintf("HVGs: %d", length(hvg)))

# --------- Build Seurat v5 object with split layers ---------
log_msg("Creating Seurat object")
seu <- CreateSeuratObject(counts = mtx, meta.data = obs)

# Split counts layer by batch (the v5 way for integration)
log_msg(sprintf("Splitting layers by '%s'", BATCH_KEY))
seu[["RNA"]] <- split(seu[["RNA"]], f = seu@meta.data[[BATCH_KEY]])
log_msg(sprintf("Layers after split: %s",
                paste(Layers(seu, search = "counts"), collapse = ", ")))

# --------- Normalize each layer ---------
log_msg("NormalizeData per layer")
seu <- NormalizeData(seu, verbose = FALSE)
log_msg("FindVariableFeatures per layer")
seu <- FindVariableFeatures(seu, nfeatures = length(hvg), verbose = FALSE)
# Force HVG set to match our Python HVGs for fair comparison
log_msg("Overriding HVGs with the precomputed Python HVG set")
VariableFeatures(seu) <- intersect(hvg, rownames(seu))
log_msg(sprintf("  After intersection with seu rows: %d HVGs",
                length(VariableFeatures(seu))))

log_msg("ScaleData on HVGs")
seu <- ScaleData(seu, features = VariableFeatures(seu), verbose = FALSE)

log_msg("RunPCA on HVGs (needed before IntegrateLayers)")
seu <- RunPCA(seu, features = VariableFeatures(seu), npcs = N_DIMS,
              verbose = FALSE)

# --------- IntegrateLayers (CCAIntegration) ---------
log_msg("Running IntegrateLayers (CCAIntegration)")
log_msg("This is the main integration step; can take 5-15 min")
t_method <- Sys.time()

seu <- IntegrateLayers(
    object             = seu,
    method             = CCAIntegration,
    orig.reduction     = "pca",
    new.reduction      = "integrated.cca",
    features           = VariableFeatures(seu),
    dims               = 1:N_DIMS,
    k.weight           = 100,    # default
    verbose            = TRUE
)

runtime <- as.numeric(difftime(Sys.time(), t_method, units = "secs"))
log_msg(sprintf("Integration done in %.1fs (%.1f min)", runtime, runtime/60))

# --------- Re-join layers ---------
log_msg("Re-joining layers")
seu[["RNA"]] <- JoinLayers(seu[["RNA"]])

# --------- Extract integrated embedding ---------
log_msg("Extracting integrated.cca embedding")
emb <- Embeddings(seu, reduction = "integrated.cca")
log_msg(sprintf("Embedding shape: %d x %d", nrow(emb), ncol(emb)))

# Sanity: order should match input barcodes
stopifnot(all(rownames(emb) == colnames(mtx)))

# --------- Save embedding ---------
emb_path <- file.path(OUTPUT_DIR, "seurat_cca_embedding.csv")
write.csv(emb, emb_path, row.names = TRUE)
log_msg(sprintf("Saved embedding: %s (%.1f MB)",
                emb_path, file.info(emb_path)$size / 1e6))

# --------- Save runtime stats ---------
stats <- list(method = "seurat_cca",
              runtime_sec = runtime,
              embedding_dim = ncol(emb))
saveRDS(stats, file.path(OUTPUT_DIR, "seurat_cca_stats.rds"))
# Also write JSON for Python compatibility
stats_json <- sprintf(
    '{"method":"seurat_cca","runtime_sec":%.2f,"embedding_dim":%d}',
    runtime, ncol(emb))
writeLines(stats_json, file.path(PROJECT_ROOT, "results", "seurat_cca_stats.json"))

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat(sprintf("[+] Seurat CCA done in %.1fs total (%.1fs for integration)\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            runtime))
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("\nNext: load result back into Python with:\n")
cat("  python scripts/04e_seurat_collect.py\n")
