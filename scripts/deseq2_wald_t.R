# deseq2_wald_t.R -- R DESeq2 Wald test with a normal (default) and a t reference distribution.
#
# For every counts_<key>.csv / coldata_<key>.csv pair in a folder (genes x samples; coldata with a
# `condition` column, reference level first in `levels.txt`, one line "ref,test"), fits ~condition
# once and tests it twice: nbinomWaldTest(useT = FALSE) and nbinomWaldTest(useT = TRUE), the latter
# with DESeq2's default df = samples - coefficients. Writes out_<key>.csv per pair.
# Usage: Rscript scripts/deseq2_wald_t.R <folder>
suppressPackageStartupMessages(library(DESeq2))
dir <- commandArgs(trailingOnly = TRUE)[1]
lv <- strsplit(readLines(file.path(dir, "levels.txt"))[1], ",")[[1]]
for (f in sort(list.files(dir, pattern = "^counts_.*[.]csv$"))) {
  key <- sub("^counts_(.*)[.]csv$", "\\1", f)
  out <- file.path(dir, sprintf("out_%s.csv", key))
  if (file.exists(out)) next
  cts <- read.csv(file.path(dir, f), row.names = 1, check.names = FALSE, na.strings = character(0))
  cd <- read.csv(file.path(dir, sprintf("coldata_%s.csv", key)), row.names = 1, colClasses = "character")
  stopifnot(identical(rownames(cd), colnames(cts)))
  cd$condition <- factor(cd$condition, levels = lv)
  dds <- DESeqDataSetFromMatrix(as.matrix(cts), cd, ~condition)
  dds <- estimateDispersions(estimateSizeFactors(dds), quiet = TRUE)
  rz <- results(nbinomWaldTest(dds, quiet = TRUE), contrast = c("condition", lv[2], lv[1]))
  rt <- results(nbinomWaldTest(dds, useT = TRUE, quiet = TRUE), contrast = c("condition", lv[2], lv[1]))
  write.csv(data.frame(gene = rownames(rz), log2FoldChange = rz$log2FoldChange,
                       pvalue_z = rz$pvalue, padj_z = rz$padj, pvalue_t = rt$pvalue, padj_t = rt$padj),
            out, row.names = FALSE)
}
cat("DESeq2", as.character(packageVersion("DESeq2")), "\n")
