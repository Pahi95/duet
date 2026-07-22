#!/usr/bin/env Rscript
# simulate_muscat.R -- ground-truth simulation for the DE benchmark.
#
# muscat::simData() simulates a multi-sample, multi-group scRNA-seq dataset from a
# real reference, with each gene assigned a KNOWN differential category:
#   ee = equally expressed        (null)
#   ep = differential proportion  (null for a mean-shift test: same overall mean)
#   de = differential expression  (mean shift        -> TRUE positive)
#   dp = differential proportion of a high state     -> TRUE positive
#   dm = differential modality                       -> TRUE positive
#   db = both / bimodal                              -> TRUE positive
#
# Only `de` is an unambiguous mean shift, so the primary truth label used
# downstream is `de` vs `ee`; the other categories are exported too and can be
# scored separately (they are exactly where a hurdle model should beat a t-test).
#
# The reference is Crowell19_4vs4 (muscat's own reference), so the simulated
# variance structure -- including between-sample variance -- is realistic. That
# matters: a simulation without sample-level variance would make every cell-level
# method look perfect and would not test the thing we care about.
#
# Usage: Rscript simulate_muscat.R [n_genes] [n_cells] [n_samples_per_group] [seed]

suppressPackageStartupMessages({
  library(muscat)
  library(SingleCellExperiment)
  library(Matrix)
  library(muscData)
  library(ExperimentHub)
})

args <- commandArgs(trailingOnly = TRUE)
ng   <- if (length(args) >= 1) as.integer(args[1]) else 4000L
nc   <- if (length(args) >= 2) as.integer(args[2]) else 24000L
nsg  <- if (length(args) >= 3) as.integer(args[3]) else 4L
seed <- if (length(args) >= 4) as.integer(args[4]) else 1L

outdir <- file.path("data", "sim")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

cat("[sim] loading reference (Crowell19_4vs4) ...\n")
setExperimentHubOption("ASK", FALSE)
ref <- Crowell19_4vs4()

# muscat's prepSim wants raw counts and drops poor cells/genes
cat("[sim] prepSim() ...\n")
ref <- prepSim(ref, verbose = FALSE)
cat("[sim] reference after prepSim:", paste(dim(ref), collapse = " x "), "\n")

# 10% of genes differential, split evenly across the four DE flavours
p_dd <- c(0.80, 0.10, 0.025, 0.025, 0.025, 0.025)   # ee, ep, de, dp, dm, db
names(p_dd) <- c("ee", "ep", "de", "dp", "dm", "db")
cat("[sim] p_dd:", paste(sprintf("%s=%.3f", names(p_dd), p_dd), collapse = " "), "\n")

cat(sprintf("[sim] simData(ng=%d, nc=%d, ns=%d/group, seed=%d) ...\n", ng, nc, nsg, seed))
sim <- simData(ref, ng = ng, nc = nc, ns = nsg, nk = 1L,
               p_dd = as.numeric(p_dd), force = TRUE)

cat("[sim] dim:", paste(dim(sim), collapse = " x "), "\n")
cd <- as.data.frame(colData(sim))
cat("[sim] colData:", paste(colnames(cd), collapse = ", "), "\n")
cat("\n[sim] sample_id x group_id:\n"); print(table(cd$sample_id, cd$group_id))
cat("\n[sim] cluster_id:\n"); print(table(cd$cluster_id))

gi <- metadata(sim)$gene_info
cat("\n[sim] gene_info columns:", paste(colnames(gi), collapse = ", "), "\n")
cat("[sim] truth categories:\n"); print(table(gi$category))

# ---- export ---------------------------------------------------------------
cnt <- counts(sim)
writeMM(as(cnt, "CsparseMatrix"), file.path(outdir, "counts.mtx"))
write.table(cd, file.path(outdir, "coldata.tsv"), sep = "\t",
            quote = FALSE, row.names = TRUE, col.names = NA)
write.table(gi, file.path(outdir, "gene_info.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)
writeLines(rownames(sim), file.path(outdir, "genes.txt"))
writeLines(colnames(sim), file.path(outdir, "cells.txt"))

cat("\n[sim] wrote ->", normalizePath(outdir), "\n")
cat("[sim] counts.mtx:", round(file.size(file.path(outdir, "counts.mtx")) / 1e6, 1), "MB\n")
