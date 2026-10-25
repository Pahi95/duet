#!/usr/bin/env python
"""
null_permutations_rdeseq2.py -- the pseudobulk arm of the label-permutation null with R DESeq2.

null_permutations.py permutes the condition label across the cells of a cell type, so every donor
contributes to both permuted groups and its pseudobulk arm fits ~donor + condition with PyDESeq2,
whose paired fit depends on sample order. This script rebuilds the identical pseudobulk matrices
(same seeds, rng([seed, cell-type index]); donor x permuted condition, >= 10 cells, genes with >= 10
summed counts, donors sorted) and fits them with R DESeq2 (default DESeq() and results(), as
duet.pseudobulk engine="R"). One R session per cell type fits all permutations.

Outputs: results/null/permutations_rdeseq2/<dataset>__<celltype>.csv (one row per seed),
         results/null/permutation_summary_rdeseq2.csv (same columns as permutation_summary.csv)
Usage:   python scripts/null_permutations_rdeseq2.py [--datasets kang,crowell] [--n-perm 100]
         (RSCRIPT=... selects Rscript; R_LIBS_USER must reach DESeq2)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import shutil                                              # noqa: E402
import subprocess                                          # noqa: E402
import tempfile                                            # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS                # noqa: E402
from null_permutations import TEST                        # noqa: E402
from null_calibration import summarize                    # noqa: E402

ALPHA = 0.05
MIN_CELLS, MIN_GENE_COUNTS = 10, 10

R_LOOP = r"""
suppressPackageStartupMessages(library(DESeq2))
a <- commandArgs(trailingOnly = TRUE)
dir <- a[1]; ref <- a[2]; test <- a[3]
for (f in sort(list.files(dir, pattern = "^counts_\\d+\\.csv$"))) {
  s <- sub("^counts_(\\d+)\\.csv$", "\\1", f)
  cts <- read.csv(file.path(dir, f), row.names = 1, check.names = FALSE, na.strings = character(0))
  cd <- read.csv(file.path(dir, sprintf("coldata_%s.csv", s)), row.names = 1, check.names = FALSE,
                 colClasses = "character")
  stopifnot(identical(rownames(cd), colnames(cts)))
  cd$condition <- factor(cd$condition, levels = c(ref, test))
  cd$donor <- factor(cd$donor, levels = sort(unique(cd$donor)))
  dds <- DESeqDataSetFromMatrix(as.matrix(round(cts)), colData = cd, design = ~donor + condition)
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, contrast = c("condition", test, ref))
  write.csv(data.frame(gene = rownames(res), log2FoldChange = res$log2FoldChange,
                       pvalue = res$pvalue, padj = res$padj),
            file.path(dir, sprintf("out_%s.csv", s)), row.names = FALSE)
}
cat("DESeq2", as.character(packageVersion("DESeq2")), "\n")
"""


def pseudobulk(X, donors, cond, genes, ref, test):
    rows, names, meta = [], [], []
    for d in sorted(pd.unique(donors)):
        for lab in (ref, test):
            m = (donors == d) & (cond == lab)
            if int(m.sum()) < MIN_CELLS:
                continue
            rows.append(np.asarray(X[m].sum(axis=0)).ravel())
            names.append(f"{d}|{lab}")
            meta.append((d, lab))
    pb = pd.DataFrame(np.rint(np.vstack(rows)).astype(int), index=names, columns=genes)
    meta = pd.DataFrame(meta, columns=["donor", "condition"], index=names)
    assert (meta.groupby("donor")["condition"].nunique() == 2).all()      # paired, as in the PyDESeq2 arm
    return pb.loc[:, pb.sum(axis=0) >= MIN_GENE_COUNTS], meta


def run_celltype(name, celltype, n_perm, outdir, rscript):
    cfg = DATASETS[name]
    ref, test = cfg["ref"], TEST[name]
    dest = Path(outdir) / f"{name}__{celltype.replace(' ', '_').replace('.', '')}.csv"
    if dest.is_file():
        return f"[{name}/{celltype}] already done"
    t0 = time.time()
    adm = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == celltype).to_numpy())
    real = np.where(adm.obs[cfg["cond"]].astype(str) == ref, ref, test)
    donors = adm.obs[cfg["donor"]].astype(str).to_numpy()
    X = adm.layers["counts"]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    genes = adm.var_names.astype(str)
    ct_index = cfg["celltypes"].index(celltype)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as tmp:
        t = Path(tmp)
        for seed in range(1, n_perm + 1):
            c = np.random.default_rng([seed, ct_index]).permutation(real)
            pb, meta = pseudobulk(X, donors, c, genes, ref, test)
            pb.T.to_csv(t / f"counts_{seed:03d}.csv")
            meta.to_csv(t / f"coldata_{seed:03d}.csv")
        (t / "loop.R").write_text(R_LOOP, encoding="utf-8")
        r = subprocess.run([rscript, str(t / "loop.R"), str(t), ref, test], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"R DESeq2 failed:\n{r.stderr[-1500:]}")
        rows = []
        for seed in range(1, n_perm + 1):
            o = pd.read_csv(t / f"out_{seed:03d}.csv", dtype={"gene": str}, keep_default_na=False,
                            na_values={"pvalue": ["NA"], "padj": ["NA"], "log2FoldChange": ["NA"]})
            df = pd.DataFrame({"gene": o["gene"], "pvalue": pd.to_numeric(o["pvalue"], errors="coerce"),
                               "fdr": pd.to_numeric(o["padj"], errors="coerce"), "tested": True})
            s = summarize(df, "DESeq2-pseudobulk (R)", celltype, "permutation", name)
            s.update(seed=seed)
            rows.append(s)
    pd.DataFrame(rows).to_csv(dest, index=False)
    return f"[{name}/{celltype}] {n_perm} permutations, {adm.n_obs} cells, {time.time() - t0:.0f}s; {r.stdout.strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kang,crowell")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--outdir", default=str(HERE / "results" / "null" / "permutations_rdeseq2"))
    a = ap.parse_args()
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for n in a.datasets.split(","):
        for ct in DATASETS[n]["celltypes"]:
            print(run_celltype(n, ct, a.n_perm, str(out), rscript), flush=True)
    D = pd.concat([pd.read_csv(f) for f in sorted(out.glob("*.csv"))], ignore_index=True)
    S = D.groupby(["dataset", "celltype", "method"], sort=False).agg(
        n_perm=("seed", "nunique"), n_tested_median=("n_tested", "median"),
        type1_mean=("type1_at_alpha05", "mean"), type1_min=("type1_at_alpha05", "min"),
        type1_max=("type1_at_alpha05", "max"), median_p_mean=("median_p", "mean"),
        perms_with_any_discovery=("n_sig_fdr05", lambda x: int((x > 0).sum())),
        discoveries_max=("n_sig_fdr05", "max")).reset_index()
    S.to_csv(out.parent / "permutation_summary_rdeseq2.csv", index=False)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
