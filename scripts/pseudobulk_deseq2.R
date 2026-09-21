#!/usr/bin/env Rscript
# pseudobulk_deseq2.R -- R DESeq2 on a pseudobulk matrix written by
# validate_pseudobulk_engines.py, so the Python arm can be checked against the
# reference implementation on identical input.
#
# Usage: Rscript pseudobulk_deseq2.R <counts.csv> <coldata.csv> <design> <ref> <test> <out.csv>
#   counts.csv  genes x samples, first column = gene
#   coldata.csv sample, donor, condition
#   design      "~condition" or "~donor + condition"

suppressPackageStartupMessages(library(DESeq2))
a <- commandArgs(trailingOnly = TRUE)
if (length(a) != 6) stop("usage: pseudobulk_deseq2.R counts.csv coldata.csv design ref test out.csv")
cts <- read.csv(a[1], row.names = 1, check.names = FALSE, na.strings = character(0))  # a gene named "NA" stays a name
cd <- read.csv(a[2], row.names = 1, check.names = FALSE)
cd <- cd[colnames(cts), , drop = FALSE]
cd$condition <- factor(cd$condition, levels = c(a[4], a[5]))
cd$donor <- factor(cd$donor)
dds <- DESeqDataSetFromMatrix(as.matrix(round(cts)), colData = cd, design = as.formula(a[3]))
dds <- DESeq(dds, quiet = TRUE)
res <- results(dds, contrast = c("condition", a[5], a[4]))      # default independent filtering + Cook's
out <- data.frame(gene = rownames(res), baseMean = res$baseMean, log2FC = res$log2FoldChange, stat = res$stat,
                  pvalue = res$pvalue, padj = res$padj)
write.csv(out, a[6], row.names = FALSE)
cat(sprintf("[R-DESeq2] %d genes, %d samples, design %s, DESeq2 %s\n", nrow(out), ncol(cts), a[3],
            as.character(packageVersion("DESeq2"))))
