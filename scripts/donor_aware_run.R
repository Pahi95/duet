#!/usr/bin/env Rscript
# donor_aware_run.R -- a donor-aware single-cell DE method on one export (MO task 12).
# Companion of donor_aware_methods.py.
#
# Usage: Rscript donor_aware_run.R <dir> <method> [lib]
#   method  glimes_poisson  GLIMES::poisson_glmm_DE  (Poisson GLMM on raw UMIs, donor random intercept)
#           glimes_binomial GLIMES::binomial_glmm_DE (detection indicator, donor random intercept)
#           mast_glmer      MAST zlm(~condition + (1|donor), method = "glmer", ebayes = FALSE) + lrTest
#   <dir>   counts.mtx (genes x cells, raw UMIs), expr.mtx (genes x cells, log-normalised),
#           genes.txt, cells.txt, coldata.csv (cell, donor, condition), labels.json
#   lib     library holding GLIMES (default: <project>/../_Rlib)
# Writes <method>.csv (gene, pvalue, fdr, log2fc, stat, status) and <method>_timing.json.

suppressPackageStartupMessages({ library(Matrix); library(SingleCellExperiment) })
a <- commandArgs(trailingOnly = TRUE)
dir <- a[1]; method <- a[2]
lib <- if (length(a) >= 3) a[3] else normalizePath(file.path(dirname(sub("^--file=", "",
         grep("^--file=", commandArgs(FALSE), value = TRUE))), "..", "..", "_Rlib"), mustWork = FALSE)
genes <- readLines(file.path(dir, "genes.txt")); cells <- readLines(file.path(dir, "cells.txt"))
cd <- read.csv(file.path(dir, "coldata.csv"), row.names = 1)[cells, , drop = FALSE]
labs <- jsonlite::fromJSON(file.path(dir, "labels.json"))
cd$condition <- factor(cd$condition, levels = c(labs$ref, labs$test))
cd$donor <- factor(cd$donor)
t0 <- proc.time()[["elapsed"]]
gc(reset = TRUE)

if (startsWith(method, "glimes")) {
  suppressPackageStartupMessages(library(GLIMES, lib.loc = lib))
  counts <- readMM(file.path(dir, "counts.mtx")); dimnames(counts) <- list(genes, cells)
  sce <- SingleCellExperiment(assays = list(counts = as(counts, "CsparseMatrix")), colData = cd)
  fn <- if (method == "glimes_poisson") poisson_glmm_DE else binomial_glmm_DE
  r <- fn(sce, comparison = "condition", replicates = "donor")
  out <- data.frame(gene = r$genes, pvalue = r$pval, fdr = r$BH, log2fc = r$log2FC,
                    stat = r$beta_comparison, status = r$status)
} else if (method == "mast_glmer") {
  suppressPackageStartupMessages({ library(MAST); library(lme4) })
  expr <- as.matrix(readMM(file.path(dir, "expr.mtx"))); dimnames(expr) <- list(genes, cells)
  sca <- FromMatrix(exprsArray = expr, cData = data.frame(wellKey = cells, cd),
                    fData = data.frame(primerid = genes, row.names = genes), check_sanity = FALSE)
  zz <- zlm(~condition + (1 | donor), sca, method = "glmer", ebayes = FALSE, strictConvergence = FALSE)
  lrt <- lrTest(zz, "condition")
  term <- paste0("condition", labs$test)
  lf <- tryCatch(logFC(zz)$logFC[, 1], error = function(e) setNames(rep(NA_real_, length(genes)), genes))
  out <- data.frame(gene = dimnames(lrt)[[1]], pvalue = lrt[, "hurdle", "Pr(>Chisq)"], fdr = NA_real_,
                    log2fc = lf[dimnames(lrt)[[1]]] / log(2), stat = lrt[, "hurdle", "lambda"],
                    status = ifelse(is.finite(lrt[, "hurdle", "lambda"]), "done", "failed"))
  out$fdr <- p.adjust(out$pvalue, method = "BH")
} else stop("unknown method ", method)

elapsed <- proc.time()[["elapsed"]] - t0
g <- gc()
write.csv(out, file.path(dir, paste0(method, ".csv")), row.names = FALSE)
writeLines(jsonlite::toJSON(list(method = method, elapsed_s = round(elapsed, 2), r_heap_peak_mb = sum(g[, 6]),
                                 n_genes = length(genes), n_cells = length(cells)), auto_unbox = TRUE),
           file.path(dir, paste0(method, "_timing.json")))
cat(sprintf("[donor-aware] %s: %d genes x %d cells in %.1fs\n", method, length(genes), length(cells), elapsed))
