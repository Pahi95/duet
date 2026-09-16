#!/usr/bin/env Rscript
# bench_mast_steps.R -- R MAST, one stage at a time, for the timing and memory
# benchmark. Driven by bench_harness.py.
#
# Usage: Rscript bench_mast_steps.R <dir> <input: sparse|dense> <ebayes: TRUE|FALSE>
#                                   [method: bayesglm|glm] [design: condition|cdr]
#   <dir> holds expr.mtx (genes x cells), genes.txt, cells.txt, cond.txt, labels.json
#   (written by bench_duet_run.py --export).
#
# Every stage prints "[stage-start] <name> <unix time>" and "[stage] <name>
# elapsed=<s> rheap_mb=<MB>", so the harness can attribute process-tree memory to
# stages. `input` decides what is handed to MAST: `sparse` keeps the dgCMatrix that
# readMM returns, `dense` converts it with as.matrix() first -- which is what the
# earlier benchmark wrapper did, and what task 1 asks to separate from MAST itself.
# A stage that fails prints "[stage-error] <name> <message>" and exits with status 3.

suppressPackageStartupMessages({ library(Matrix); library(MAST); library(SingleCellExperiment) })
a <- commandArgs(trailingOnly = TRUE)
dir <- a[1]; input <- a[2]; eb <- as.logical(a[3])
method <- if (length(a) >= 4) a[4] else "bayesglm"
design <- if (length(a) >= 5) a[5] else "condition"
options(mc.cores = 1L)

now <- function() as.numeric(Sys.time())
stage <- function(name, expr) {
  gc(reset = TRUE, full = TRUE)
  cat(sprintf("[stage-start] %s %.6f\n", name, now())); flush(stdout())
  pt0 <- proc.time(); t0 <- pt0[["elapsed"]]
  val <- tryCatch(force(expr), error = function(e) {
    cat(sprintf("[stage-error] %s %s\n", name, conditionMessage(e))); flush(stdout()); quit(status = 3)
  })
  pt1 <- proc.time(); el <- pt1[["elapsed"]] - t0
  cpu <- (pt1[["user.self"]] + pt1[["sys.self"]]) - (pt0[["user.self"]] + pt0[["sys.self"]])
  g <- gc(full = TRUE)
  cat(sprintf("[stage] %s elapsed=%.3f cpu=%.3f rheap_mb=%.1f end=%.6f\n", name, el, cpu, sum(g[, 6]), now()))
  flush(stdout())
  val
}
cat(sprintf("[mark] baseline %.6f\n", now())); flush(stdout())

expr <- stage("read", {
  m <- readMM(file.path(dir, "expr.mtx"))
  dimnames(m) <- list(readLines(file.path(dir, "genes.txt")), readLines(file.path(dir, "cells.txt")))
  if (input == "dense") as.matrix(m) else as(m, "CsparseMatrix")
})
labs <- jsonlite::fromJSON(file.path(dir, "labels.json"))
cond <- factor(readLines(file.path(dir, "cond.txt")), levels = c(labs$ref, labs$test))
sca <- stage("singlecellassay", {
  cd <- data.frame(wellKey = colnames(expr), condition = cond, row.names = colnames(expr))
  fd <- data.frame(primerid = rownames(expr), row.names = rownames(expr))
  s <- if (input == "dense") {
    FromMatrix(exprsArray = expr, cData = cd, fData = fd, check_sanity = FALSE)
  } else {
    # FromMatrix() rejects a dgCMatrix ("exprsArray must be matrix, 3-D array, list or
    # SimpleList"), so the sparse route goes through a SingleCellExperiment
    SceToSingleCellAssay(SingleCellExperiment(assays = list(et = expr), colData = cd, rowData = fd),
                         check_sanity = FALSE)
  }
  if (design == "cdr") colData(s)$cngeneson <- as.numeric(scale(Matrix::colSums(assay(s) > 0)))
  s
})
cat(sprintf("[info] assay class after SingleCellAssay: %s\n", class(assay(sca))[1])); flush(stdout())
rm(expr)
form <- if (design == "cdr") ~condition + cngeneson else ~condition
zz <- stage("zlm", zlm(form, sca, method = method, ebayes = eb))
lrt <- stage("lrtest", lrTest(zz, "condition"))
res <- stage("assemble", {
  lf <- logFC(zz)$logFC[, 1]
  data.frame(gene = dimnames(lrt)[[1]], stat_hurdle = lrt[, "hurdle", "lambda"],
             df_hurdle = lrt[, "hurdle", "df"], pvalue = lrt[, "hurdle", "Pr(>Chisq)"],
             coef = lf[dimnames(lrt)[[1]]])
})
invisible(stage("write", write.csv(res, file.path(dir, "mast_bench_results.csv"), row.names = FALSE)))
cat("[done]\n")
