#!/usr/bin/env Rscript
# bench_mast_run.R -- step 2 of the DUET vs R MAST head-to-head benchmark.
#
# Reads the export written by bench_mast_export.py and runs R MAST on the SAME
# cells, the SAME genes and the SAME design (~1+condition, no covariates, which
# is what DUET's mast_compat preset uses).
#
# Three clocks are recorded separately, because they answer different questions:
#   read_s  - readMM + building the SingleCellAssay. This is the disk round-trip
#             DUET does not pay at all; it belongs in the end-to-end number, not
#             in the "who fits faster" number.
#   fit_s   - zlm() only. The core model fit.
#   lrt_s   - lrTest() + assembling the result table.
# Peak memory is taken from gc()'s max-used counters.
#
# Usage: Rscript bench_mast_run.R <bench_dir> [n_threads]

suppressPackageStartupMessages({
  library(Matrix)
  library(MAST)
  library(SingleCellExperiment)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript bench_mast_run.R <bench_dir> [n_threads]")
bd <- args[1]
nthreads <- if (length(args) >= 2) as.integer(args[2]) else 1L
options(mc.cores = nthreads)

cat(sprintf("[mast] dir=%s threads=%d\n", bd, nthreads))
cat(sprintf("[mast] MAST %s, R %s\n", as.character(packageVersion("MAST")),
            paste0(R.version$major, ".", R.version$minor)))

gc(reset = TRUE, full = TRUE)

## ---- read ----------------------------------------------------------------
t0 <- proc.time()[["elapsed"]]
expr <- readMM(file.path(bd, "expr.mtx"))                 # genes x cells
genes <- readLines(file.path(bd, "genes.txt"))
cells <- readLines(file.path(bd, "cells.txt"))
cond  <- readLines(file.path(bd, "cond.txt"))
expr <- as.matrix(expr)                                   # MAST wants dense
rownames(expr) <- genes
colnames(expr) <- cells

# Condition levels come from labels.json when the caller supplies it (datasets
# other than the pancreas one use ctrl/stim, Vehicle/LPS, ...). The REFERENCE
# level must be first so zlm's condition coefficient is test-vs-reference.
lab_file <- file.path(bd, "labels.json")
if (file.exists(lab_file)) {
  labs <- jsonlite::fromJSON(lab_file)
  lev <- c(labs$ref, labs$test)
} else {
  lev <- c("Reference", "pancreas")
}
if (!all(unique(cond) %in% lev))
  stop(sprintf("cond.txt has values %s not in levels %s",
               paste(setdiff(unique(cond), lev), collapse = "/"),
               paste(lev, collapse = "/")))
cat(sprintf("[mast] condition levels: %s (reference first)\n", paste(lev, collapse = " -> ")))

cdat <- data.frame(wellKey = cells,
                   condition = factor(cond, levels = lev),
                   row.names = cells)
fdat <- data.frame(primerid = genes, row.names = genes)

sca <- FromMatrix(exprsArray = expr, cData = cdat, fData = fdat,
                  check_sanity = FALSE)
read_s <- proc.time()[["elapsed"]] - t0
cat(sprintf("[mast] read + SingleCellAssay: %.2fs  (%d genes x %d cells)\n",
            read_s, length(genes), length(cells)))

## ---- fit -----------------------------------------------------------------
# ~1+condition with NO covariates, to match DUET's mast_compat preset exactly.
# (The pipeline's production MAST adds cngeneson/CDR; adding it here would make
# the two engines solve different problems.)
t0 <- proc.time()[["elapsed"]]
zz <- zlm(~condition, sca)
fit_s <- proc.time()[["elapsed"]] - t0
cat(sprintf("[mast] zlm fit: %.2fs\n", fit_s))

## ---- LRT -----------------------------------------------------------------
t0 <- proc.time()[["elapsed"]]
lrt <- lrTest(zz, "condition")
# lrt is genes x {cont,disc,hurdle} x {lambda,df,Pr(>Chisq)}
res <- data.table(
  gene       = dimnames(lrt)[[1]],
  stat_hurdle = as.numeric(lrt[, "hurdle", "lambda"]),
  df_hurdle   = as.numeric(lrt[, "hurdle", "df"]),
  pvalue      = as.numeric(lrt[, "hurdle", "Pr(>Chisq)"]),
  stat_cont   = as.numeric(lrt[, "cont",   "lambda"]),
  stat_disc   = as.numeric(lrt[, "disc",   "lambda"])
)
# condition coefficient from the discrete+continuous hurdle -> MAST's logFC
lf <- logFC(zz)
lfdt <- data.table(gene = as.character(dimnames(lf$logFC)[[1]]),
                   coef = as.numeric(lf$logFC[, 1]))
res <- merge(res, lfdt, by = "gene", all.x = TRUE)
res[, fdr := p.adjust(pvalue, method = "BH")]
lrt_s <- proc.time()[["elapsed"]] - t0
cat(sprintf("[mast] lrTest + logFC: %.2fs\n", lrt_s))

# gc() returns a matrix whose 6th column is "max used (Mb)", one row each for
# Ncells (cons cells) and Vcells (vector storage); their sum is peak R heap.
g <- gc(full = TRUE)
peak_mb <- sum(g[, 6])

fwrite(res, file.path(bd, "mast_results.csv"))
timing <- list(read_s = round(read_s, 3), fit_s = round(fit_s, 3),
               lrt_s = round(lrt_s, 3),
               mast_total_s = round(read_s + fit_s + lrt_s, 3),
               peak_mb = round(peak_mb, 1),
               n_genes = length(genes), n_cells = length(cells),
               threads = nthreads,
               mast_version = as.character(packageVersion("MAST")),
               design = "~1+condition (no covariates)")
writeLines(jsonlite::toJSON(timing, auto_unbox = TRUE, pretty = TRUE),
           file.path(bd, "mast_timing.json"))
cat(sprintf("[mast] TOTAL %.2fs (read %.2f + fit %.2f + lrt %.2f), peak %.0f MB\n",
            read_s + fit_s + lrt_s, read_s, fit_s, lrt_s, peak_mb))
cat(sprintf("[mast] wrote %s\n", file.path(bd, "mast_results.csv")))
