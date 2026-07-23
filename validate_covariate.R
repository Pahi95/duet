#!/usr/bin/env Rscript
# validate_covariate.R -- MAST side of the covariate check.
#
# Fits ~1 + cngeneson + condition on exactly the cells, genes and CDR values that
# validate_covariate.py exported, so the only thing that can differ between the
# two engines is the model fit itself. cngeneson is the scaled CDR, matching
# MAST's own recommended workflow; DUET z-scores its CDR column, so the two are
# equivalent up to a linear rescaling, which does not change the condition LRT.

suppressPackageStartupMessages({
  library(Matrix); library(MAST); library(SingleCellExperiment); library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
bd <- args[1]
cat(sprintf("[mast] dir=%s  MAST %s\n", bd, as.character(packageVersion("MAST"))))

expr  <- as.matrix(readMM(file.path(bd, "expr.mtx")))
genes <- readLines(file.path(bd, "genes.txt"))
cells <- readLines(file.path(bd, "cells.txt"))
cond  <- readLines(file.path(bd, "cond.txt"))
cdr   <- as.numeric(readLines(file.path(bd, "cdr.txt")))
labs  <- jsonlite::fromJSON(file.path(bd, "labels.json"))
rownames(expr) <- genes; colnames(expr) <- cells

stopifnot(length(cdr) == length(cells))
lev <- c(labs$ref, labs$test)
cat(sprintf("[mast] condition levels: %s (reference first)\n", paste(lev, collapse = " -> ")))

cdat <- data.frame(
  wellKey   = cells,
  condition = factor(cond, levels = lev),
  cngeneson = as.numeric(scale(cdr)),      # MAST's standard scaled CDR
  row.names = cells)
fdat <- data.frame(primerid = genes, row.names = genes)

sca <- FromMatrix(exprsArray = expr, cData = cdat, fData = fdat, check_sanity = FALSE)

t0 <- proc.time()[["elapsed"]]
zz <- zlm(~ cngeneson + condition, sca)
cat(sprintf("[mast] zlm(~cngeneson + condition) fit: %.1fs\n", proc.time()[["elapsed"]] - t0))

t0 <- proc.time()[["elapsed"]]
lrt <- lrTest(zz, "condition")
res <- data.table(
  gene        = dimnames(lrt)[[1]],
  stat_hurdle = as.numeric(lrt[, "hurdle", "lambda"]),
  pvalue      = as.numeric(lrt[, "hurdle", "Pr(>Chisq)"]))
# logFC() returns ONE COLUMN PER NON-INTERCEPT COEFFICIENT. With a covariate in
# the model that is c("cngeneson", "conditionstim"), so column 1 is the COVARIATE,
# not the condition. Select by name; taking [, 1] silently compares the wrong term.
lf <- logFC(zz)
cn <- dimnames(lf$logFC)[[2]]
ci <- grep("^condition", cn)
if (length(ci) != 1)
  stop(sprintf("cannot identify the condition column in logFC(): [%s]",
               paste(cn, collapse = ", ")))
cat(sprintf("[mast] logFC columns [%s]; using '%s'\n",
            paste(cn, collapse = ", "), cn[ci]))
res <- merge(res, data.table(gene = as.character(dimnames(lf$logFC)[[1]]),
                             coef = as.numeric(lf$logFC[, ci])), by = "gene", all.x = TRUE)
res[, fdr := p.adjust(pvalue, method = "BH")]
cat(sprintf("[mast] lrTest + logFC: %.1fs\n", proc.time()[["elapsed"]] - t0))

fwrite(res, file.path(bd, "mast_cov_results.csv"))
cat(sprintf("[mast] wrote %s (%d genes)\n", file.path(bd, "mast_cov_results.csv"), nrow(res)))
