#!/usr/bin/env Rscript
# simulate_replicates.R -- N independent muscat simulations for replicate-level
# variability. prepSim() is the expensive step (~25 min) and depends only on the
# reference, so it runs ONCE and is cached to disk; each replicate is then a fast
# simData() draw with its own seed.
#
# Usage: Rscript simulate_replicates.R [n_reps] [ng] [nc] [ns]

suppressPackageStartupMessages({
  library(muscat); library(SingleCellExperiment); library(Matrix)
  library(muscData); library(ExperimentHub)
})

# Primary data lives outside the checkout (see README). Resolve it the same
# way the Python scripts do, so both agree on where replicates are written.
inputs <- Sys.getenv("DUET_INPUTS", unset = file.path("..", "..", "inputs"))
args <- commandArgs(trailingOnly = TRUE)
nrep <- if (length(args) >= 1) as.integer(args[1]) else 5L
ng   <- if (length(args) >= 2) as.integer(args[2]) else 4000L
nc   <- if (length(args) >= 3) as.integer(args[3]) else 24000L
ns   <- if (length(args) >= 4) as.integer(args[4]) else 4L

cache <- file.path(inputs, "data", "sim_ref_prepped.rds")
if (file.exists(cache)) {
  cat("[sim] loading cached prepSim reference\n")
  ref <- readRDS(cache)
} else {
  cat("[sim] prepSim() on Crowell19_4vs4 (slow, cached afterwards) ...\n")
  setExperimentHubOption("ASK", FALSE)
  ref <- prepSim(Crowell19_4vs4(), verbose = FALSE)
  saveRDS(ref, cache)
}
cat("[sim] reference:", paste(dim(ref), collapse = " x "), "\n")

p_dd <- c(0.80, 0.10, 0.025, 0.025, 0.025, 0.025)   # ee, ep, de, dp, dm, db

for (r in seq_len(nrep)) {
  seed <- 100L + r
  outdir <- file.path(inputs, "data", sprintf("sim_rep%02d", r))
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  # seed.txt is written last, so it marks a complete replicate (MO task 16 also asks
  # for the seed to be recorded). A directory without it was interrupted mid-write and
  # is regenerated from the same seed.
  if (file.exists(file.path(outdir, "seed.txt"))) {
    cat(sprintf("[sim] rep %d already present, skipping\n", r)); next
  }
  set.seed(seed)
  cat(sprintf("[sim] replicate %d/%d (seed %d) ...\n", r, nrep, seed))
  sim <- simData(ref, ng = ng, nc = nc, ns = ns, nk = 1L, p_dd = p_dd, force = TRUE)

  writeMM(as(counts(sim), "CsparseMatrix"), file.path(outdir, "counts.mtx"))
  write.table(as.data.frame(colData(sim)), file.path(outdir, "coldata.tsv"),
              sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)
  write.table(metadata(sim)$gene_info, file.path(outdir, "gene_info.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  writeLines(rownames(sim), file.path(outdir, "genes.txt"))
  writeLines(colnames(sim), file.path(outdir, "cells.txt"))
  writeLines(as.character(seed), file.path(outdir, "seed.txt"))
  cat(sprintf("[sim]   -> %s  (%s)\n", outdir,
              paste(table(metadata(sim)$gene_info$category), collapse = "/")))
}
cat("[sim] done\n")
