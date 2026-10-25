# simulate_batch_nb_wald_t.R -- R DESeq2 on the NB global-null pseudobulk matrices: default Wald test
# (normal reference) against the Wald test with a t reference (useT = TRUE, df = n - p).
# Input: a folder of counts_<scenario>_<rep>.csv / coldata_<scenario>_<rep>.csv written by
#        simulate_batch_nb_rdeseq2.py --dump; Output: <folder>/wald_t_summary.csv
suppressPackageStartupMessages(library(DESeq2))
dir <- commandArgs(trailingOnly = TRUE)[1]
rows <- list()
for (f in sort(list.files(dir, pattern = "^counts_.*[.]csv$"))) {
  key <- sub("^counts_(.*)[.]csv$", "\\1", f)
  cts <- read.csv(file.path(dir, f), row.names = 1, check.names = FALSE)
  cd <- read.csv(file.path(dir, sprintf("coldata_%s.csv", key)), row.names = 1, colClasses = "character")
  cd$condition <- factor(cd$condition, levels = c("A", "B"))
  dds <- DESeqDataSetFromMatrix(as.matrix(cts), cd, ~condition)
  dds <- estimateDispersions(estimateSizeFactors(dds), quiet = TRUE)
  for (useT in c(FALSE, TRUE)) {
    d2 <- nbinomWaldTest(dds, useT = useT, quiet = TRUE)
    r <- results(d2, contrast = c("condition", "B", "A"))
    rows[[length(rows) + 1]] <- data.frame(key = key, useT = useT, n_disc = sum(r$padj < 0.05, na.rm = TRUE),
                                           raw_type1 = mean(r$pvalue < 0.05, na.rm = TRUE))
  }
}
write.csv(do.call(rbind, rows), file.path(dir, "wald_t_summary.csv"), row.names = FALSE)
