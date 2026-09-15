#!/usr/bin/env Rscript
# mast_components.R -- R MAST with every setting spelled out, returning each hurdle
# component separately (MO task 3). Companion of mast_settings_check.py.
#
# Usage: Rscript mast_components.R <dir> <method: bayesglm|glm> <ebayes: TRUE|FALSE>
#   <dir> holds expr.mtx (genes x cells, log-normalised), genes.txt, cells.txt,
#   cond.txt and labels.json ({"ref": ..., "test": ...}); writes
#   mast_<method>_eb<ebayes>.csv there.
#
# Settings, all explicit rather than defaulted:
#   formula  ~condition, reference level first (no CDR / cngeneson covariate)
#   method   bayesglm (MAST's default: weakly informative Cauchy prior on the
#            logistic coefficients) or glm (plain maximum likelihood)
#   ebayes   empirical-Bayes shrinkage of the continuous-part variance (default TRUE)
#   test     lrTest(zz, "condition"): discrete, continuous and hurdle likelihood ratios
#   genes    exactly the genes listed in genes.txt; MAST applies no further filter
#   BH       p.adjust(method = "BH") over those genes

suppressPackageStartupMessages({ library(Matrix); library(MAST); library(SingleCellExperiment) })
a <- commandArgs(trailingOnly = TRUE)
dir <- a[1]; method <- a[2]; eb <- as.logical(a[3])
expr <- as.matrix(readMM(file.path(dir, "expr.mtx")))
genes <- readLines(file.path(dir, "genes.txt")); cells <- readLines(file.path(dir, "cells.txt"))
cond <- readLines(file.path(dir, "cond.txt"))
labs <- jsonlite::fromJSON(file.path(dir, "labels.json"))
dimnames(expr) <- list(genes, cells)
cd <- data.frame(wellKey = cells, condition = factor(cond, levels = c(labs$ref, labs$test)), row.names = cells)
sca <- FromMatrix(exprsArray = expr, cData = cd, fData = data.frame(primerid = genes, row.names = genes),
                  check_sanity = FALSE)
t0 <- proc.time()[["elapsed"]]
zz <- zlm(~condition, sca, method = method, ebayes = eb)
lrt <- lrTest(zz, "condition")
term <- paste0("condition", labs$test)
cD <- coef(zz, "D"); cC <- coef(zz, "C")
lf <- logFC(zz)$logFC[, 1]
out <- data.frame(
  gene = dimnames(lrt)[[1]],
  stat_disc = lrt[, "disc", "lambda"], df_disc = lrt[, "disc", "df"], p_disc = lrt[, "disc", "Pr(>Chisq)"],
  stat_cont = lrt[, "cont", "lambda"], df_cont = lrt[, "cont", "df"], p_cont = lrt[, "cont", "Pr(>Chisq)"],
  stat_hurdle = lrt[, "hurdle", "lambda"], df_hurdle = lrt[, "hurdle", "df"],
  pvalue = lrt[, "hurdle", "Pr(>Chisq)"])
out$coef_disc <- cD[out$gene, term]
out$coef_cont <- cC[out$gene, term]
out$logFC <- lf[out$gene]
out$fdr <- p.adjust(out$pvalue, method = "BH")
f <- file.path(dir, sprintf("mast_%s_eb%s.csv", method, eb))
write.csv(out, f, row.names = FALSE)
cat(sprintf("[mast] %s ebayes=%s: %d genes x %d cells in %.1fs -> %s\n", method, eb, length(genes),
            length(cells), proc.time()[["elapsed"]] - t0, f))
