# pancreas_paired_patients_limma.R -- the pancreas pseudobulk with the patient as a random block.
#
# DESeq2 has no random effect, and a fixed patient term would leave the unpaired samples without
# information on the condition. limma-voom with duplicateCorrelation(block = patient) keeps every
# sample and models the within-patient correlation of the tumor and adjacent non-tumor samples.
# The same fit without the block isolates what the pairing changes.
#
# Input:  results/pancreas_20260925/paired_patients/{pb,meta}_<ct>.csv (pancreas_paired_patients.py)
# Output: results/pancreas_20260925/paired_patients/limma_<ct>.csv, limma_summary.csv
# Usage:  Rscript scripts/pancreas_paired_patients_limma.R  (limma and edgeR from R_LIBS_USER or _Rlib46)

lib <- file.path(dirname(normalizePath(".")), "_Rlib46")
if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({ library(limma); library(edgeR) })

out <- file.path("results", "pancreas_20260925", "paired_patients")
rows <- list()
for (tag in c("ductal", "endothelial", "stellate")) {
  pb <- as.matrix(read.csv(file.path(out, sprintf("pb_%s.csv", tag)), row.names = 1, check.names = FALSE))
  meta <- read.csv(file.path(out, sprintf("meta_%s.csv", tag)))
  stopifnot(identical(rownames(pb), meta$donor))
  y <- DGEList(t(pb))
  y <- calcNormFactors(y)
  cond <- factor(meta$condition, levels = c("Reference", "pancreas"))
  design <- model.matrix(~cond)
  v <- voom(y, design)
  dc <- duplicateCorrelation(v, design, block = meta$patient)
  v2 <- voom(y, design, block = meta$patient, correlation = dc$consensus.correlation)
  dc2 <- duplicateCorrelation(v2, design, block = meta$patient)
  fb <- eBayes(lmFit(v2, design, block = meta$patient, correlation = dc2$consensus.correlation))
  fu <- eBayes(lmFit(v, design))
  tb <- topTable(fb, coef = 2, number = Inf, sort.by = "none")
  tu <- topTable(fu, coef = 2, number = Inf, sort.by = "none")
  res <- data.frame(gene = rownames(tb), logFC_blocked = tb$logFC, fdr_blocked = tb$adj.P.Val,
                    logFC_unblocked = tu$logFC, fdr_unblocked = tu$adj.P.Val)
  write.csv(res, file.path(out, sprintf("limma_%s.csv", tag)), row.names = FALSE)
  sb <- res$gene[res$fdr_blocked < 0.05]; su <- res$gene[res$fdr_unblocked < 0.05]
  rows[[tag]] <- data.frame(
    celltype = tag, n_samples = nrow(meta),
    n_paired_patients = length(unique(meta$patient[as.logical(meta$paired_in_celltype)])),
    within_patient_correlation = dc2$consensus.correlation, n_genes = nrow(res),
    n_sig_unblocked = length(su), n_sig_blocked = length(sb),
    jaccard = length(intersect(sb, su)) / max(length(union(sb, su)), 1),
    logFC_pearson = cor(res$logFC_blocked, res$logFC_unblocked))
  print(rows[[tag]])
}
write.csv(do.call(rbind, rows), file.path(out, "limma_summary.csv"), row.names = FALSE)
cat("limma", as.character(packageVersion("limma")), "edgeR", as.character(packageVersion("edgeR")), "\n")
