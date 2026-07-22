#!/usr/bin/env Rscript
# get_crowell.R -- fetch Crowell et al. LPS mouse cortex (muscData::Crowell19_4vs4)
# and export it in a form the Python pipeline can assemble into an h5ad.
#
# Exports counts as MatrixMarket plus colData/rowData as TSV, deliberately
# avoiding zellkonverter/basilisk (which would pull a whole second Python env).
# This mirrors the interchange already used by bench_mast_run.R.

suppressPackageStartupMessages({
  library(ExperimentHub)
  library(muscData)
  library(SingleCellExperiment)
  library(Matrix)
})

outdir <- file.path("data", "crowell")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

setExperimentHubOption("ASK", FALSE)
cat("[crowell] loading Crowell19_4vs4 (downloads on first use) ...\n")
sce <- Crowell19_4vs4()

cat("[crowell] dim:", paste(dim(sce), collapse = " x "), "\n")
cat("[crowell] assays:", paste(assayNames(sce), collapse = ", "), "\n")
cat("[crowell] colData columns:\n")
cd <- as.data.frame(colData(sce))
for (n in colnames(cd)) {
  v <- cd[[n]]
  u <- unique(v)
  cat(sprintf("   %-16s n_unique=%-6d %s\n", n, length(u),
              paste(utils::head(as.character(u), 6), collapse = ", ")))
}
cat("[crowell] rowData columns:", paste(colnames(rowData(sce)), collapse = ", "), "\n")

# ---- design tables -------------------------------------------------------
if (all(c("group_id", "sample_id") %in% colnames(cd))) {
  cat("\n[crowell] sample_id x group_id:\n")
  print(table(cd$sample_id, cd$group_id))
}
if (all(c("cluster_id", "group_id") %in% colnames(cd))) {
  cat("\n[crowell] cluster_id x group_id:\n")
  print(table(cd$cluster_id, cd$group_id))
}

# ---- export --------------------------------------------------------------
cnt <- assay(sce, "counts")
cat(sprintf("\n[crowell] counts: %s, nnz=%s, integer=%s\n",
            class(cnt)[1], format(length(cnt@x), big.mark = ","),
            all(cnt@x == round(cnt@x))))

writeMM(as(cnt, "CsparseMatrix"), file.path(outdir, "counts.mtx"))
write.table(cd, file.path(outdir, "coldata.tsv"), sep = "\t",
            quote = FALSE, row.names = TRUE, col.names = NA)
write.table(as.data.frame(rowData(sce)), file.path(outdir, "rowdata.tsv"),
            sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)
writeLines(rownames(sce), file.path(outdir, "genes.txt"))
writeLines(colnames(sce), file.path(outdir, "cells.txt"))

cat("[crowell] wrote ->", normalizePath(outdir), "\n")
cat("[crowell] counts.mtx size:",
    round(file.size(file.path(outdir, "counts.mtx")) / 1e6, 1), "MB\n")
