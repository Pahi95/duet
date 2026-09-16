#!/usr/bin/env Rscript
# sim_pb_edger_voom.R -- edgeR quasi-likelihood and limma-voom pseudobulk on the saved
# muscat replicates (reviewer request: further sample-level comparators).
#
# Counts are summed per simulated sample (sample_id), exactly the pseudobulk unit of the
# DESeq2 arm; design ~group_id (the simulated samples of the two groups are distinct units).
# Genes are filtered with edgeR::filterByExpr(group = group_id); TMM normalization.
#   edgeR: glmQLFit(robust = TRUE) + glmQLFTest
#   voom : voom -> lmFit -> eBayes (robust = FALSE)
# Output per replicate, in the sim_calls.py column layout:
#   results/sim_reps100/calls/<sim_repNN>/edger.csv and voom.csv
#   gene, tested, skip_reason, pvalue, fdr, score (-log10 p), log2fc, stat
# Resumable: an existing file is skipped. Simulations are not regenerated.
#
# Usage: Rscript sim_pb_edger_voom.R <inputs/data> <results/sim_reps100/calls> [reps, e.g. 1-100]

suppressPackageStartupMessages({ library(Matrix); library(edgeR); library(limma) })
a <- commandArgs(trailingOnly = TRUE)
data_dir <- a[1]; calls_dir <- a[2]
rr <- if (length(a) >= 3) as.integer(strsplit(a[3], "-")[[1]]) else c(1L, 100L)

write_calls <- function(path, genes, keep, p, lfc, stat) {
  out <- data.frame(gene = genes, tested = keep, skip_reason = ifelse(keep, "", "filterByExpr"),
                    pvalue = NA_real_, fdr = NA_real_, score = NA_real_, log2fc = NA_real_, stat = NA_real_)
  out$pvalue[keep] <- p
  out$fdr[keep] <- p.adjust(p, "BH")
  out$score[keep] <- -log10(p)
  out$log2fc[keep] <- lfc
  out$stat[keep] <- stat
  write.csv(out, path, row.names = FALSE, na = "")
}

for (r in rr[1]:rr[2]) {
  rep <- sprintf("sim_rep%02d", r)
  src <- file.path(data_dir, rep)
  dst <- file.path(calls_dir, rep)
  f_e <- file.path(dst, "edger.csv"); f_v <- file.path(dst, "voom.csv")
  if (file.exists(f_e) && file.exists(f_v)) next
  if (!file.exists(file.path(src, "seed.txt"))) next
  X <- as(readMM(file.path(src, "counts.mtx")), "CsparseMatrix")     # genes x cells
  genes <- readLines(file.path(src, "genes.txt"))
  cd <- read.delim(file.path(src, "coldata.tsv"), row.names = 1)
  smp <- factor(cd$sample_id)
  P <- as.matrix(X %*% t(fac2sparse(smp)))                            # genes x samples
  rownames(P) <- genes; colnames(P) <- levels(smp)
  grp <- factor(cd$group_id[match(colnames(P), cd$sample_id)], levels = c("A", "B"))
  y <- DGEList(P, group = grp)
  keep <- filterByExpr(y, group = grp)
  y <- normLibSizes(y[keep, , keep.lib.sizes = FALSE])
  design <- model.matrix(~grp)
  fit <- glmQLFit(y, design, robust = TRUE)
  qt <- glmQLFTest(fit, coef = 2)$table
  dir.create(dst, showWarnings = FALSE, recursive = TRUE)
  write_calls(f_e, genes, keep, qt$PValue, qt$logFC, qt$F * sign(qt$logFC))
  v <- voom(y, design)
  tt <- topTable(eBayes(lmFit(v, design)), coef = 2, number = Inf, sort.by = "none")
  write_calls(f_v, genes, keep, tt$P.Value, tt$logFC, tt$t)
  cat(sprintf("[pb] %s: %d samples, %d/%d genes kept\n", rep, ncol(P), sum(keep), length(genes)))
}
cat(sprintf("[pb] edgeR %s, limma %s\n", packageVersion("edgeR"), packageVersion("limma")))
