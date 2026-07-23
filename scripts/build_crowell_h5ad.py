#!/usr/bin/env python
"""
build_crowell_h5ad.py
=====================
Assemble the muscData Crowell19_4vs4 export (written by get_crowell.R) into an
AnnData in the shape run_de_comparison.py expects:

    layers['counts'] = raw integer counts
    X                = log1p(normalize_total(counts, 1e4))
    obs['celltype']  = cluster_id
    obs['Sample']    = group_id      (Vehicle / LPS)
    obs['SourceFile']= sample_id     (the 8 mice -> DESeq2 pseudobulk replicates)

Normalization is applied here rather than reusing muscData's `logcounts`, so the
preprocessing is identical to what the pancreas object received and the two
datasets stay comparable.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(INPUTS / "data" / "crowell"))
    ap.add_argument("--out", default=str(INPUTS / "data" / "crowell_4vs4.h5ad"))
    a = ap.parse_args()
    d = Path(a.indir)

    print("[build] reading counts.mtx (genes x cells) ...", flush=True)
    m = mmread(str(d / "counts.mtx"))          # genes x cells
    X = sp.csr_matrix(m.T)                     # -> cells x genes
    genes = [l.strip() for l in (d / "genes.txt").read_text(encoding="utf-8").splitlines()]
    cells = [l.strip() for l in (d / "cells.txt").read_text(encoding="utf-8").splitlines()]
    print(f"[build] X = {X.shape[0]} cells x {X.shape[1]} genes, nnz={X.nnz:,}", flush=True)
    assert X.shape == (len(cells), len(genes)), "shape/label mismatch"

    cd = pd.read_csv(d / "coldata.tsv", sep="\t", index_col=0, low_memory=False)
    rd = pd.read_csv(d / "rowdata.tsv", sep="\t", index_col=0, low_memory=False)
    cd.index = pd.Index(cells, name=None)
    rd.index = pd.Index(genes, name=None)

    obs = pd.DataFrame(index=cd.index)
    obs["celltype"] = cd["cluster_id"].astype(str)
    obs["Sample"] = cd["group_id"].astype(str)          # Vehicle / LPS
    obs["SourceFile"] = cd["sample_id"].astype(str)     # the 8 mice
    for keep in ("nCount_RNA", "nFeature_RNA", "barcode"):
        if keep in cd.columns:
            obs[keep] = cd[keep].values

    A = ad.AnnData(X=X, obs=obs, var=rd.copy())
    A.var_names = pd.Index(genes)
    A.var_names_make_unique()
    A.obs_names_make_unique()

    # raw counts kept aside, then normalize in place -- same recipe as the
    # pancreas object (counts layer + log1p CP10K in X)
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    print("\n[build] design check")
    print(pd.crosstab(A.obs["SourceFile"], A.obs["Sample"]).to_string())
    print("\n[build] cells per cell type x condition")
    print(pd.crosstab(A.obs["celltype"], A.obs["Sample"]).to_string())

    cnts = A.layers["counts"]
    print(f"\n[build] counts integer: {bool(np.all(np.abs(cnts.data - np.round(cnts.data)) < 1e-9))}"
          f" | max {float(cnts.max()):.0f}")
    print(f"[build] X log-normalized: max {float(A.X.max()):.2f}")

    A.write_h5ad(a.out, compression="gzip")
    print(f"\n[build] wrote {a.out} ({Path(a.out).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
