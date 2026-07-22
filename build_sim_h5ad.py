#!/usr/bin/env python
"""
build_sim_h5ad.py
=================
Assemble the muscat simulation (written by simulate_muscat.R) into an AnnData in
the shape run_de_comparison.py expects, and carry the ground-truth gene labels
through in `var` so the evaluation can join on them.

    layers['counts'] = simulated raw counts
    X                = log1p(normalize_total(counts, 1e4))
    obs['celltype']  = cluster_id
    obs['Sample']    = group_id       (A / B)
    obs['SourceFile']= sample_id      (the simulated samples -> pseudobulk units)
    var['category']  = ee / ep / de / dp / dm / db   <- GROUND TRUTH
    var['is_de']     = category == 'de'              <- primary truth label
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
# categories that represent a genuine mean shift the tests should detect
TRUE_DE = {"de", "dp", "dm", "db"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(HERE / "data" / "sim"))
    ap.add_argument("--out", default=str(HERE / "data" / "sim_muscat.h5ad"))
    a = ap.parse_args()
    d = Path(a.indir)

    print("[build] reading counts.mtx ...", flush=True)
    X = sp.csr_matrix(mmread(str(d / "counts.mtx")).T)      # -> cells x genes
    genes = [l.strip() for l in (d / "genes.txt").read_text(encoding="utf-8").splitlines()]
    cells = [l.strip() for l in (d / "cells.txt").read_text(encoding="utf-8").splitlines()]
    print(f"[build] {X.shape[0]} cells x {X.shape[1]} genes, nnz={X.nnz:,}")
    assert X.shape == (len(cells), len(genes))

    cd = pd.read_csv(d / "coldata.tsv", sep="\t", index_col=0, low_memory=False)
    gi = pd.read_csv(d / "gene_info.tsv", sep="\t", low_memory=False)
    cd.index = pd.Index(cells)

    obs = pd.DataFrame(index=cd.index)
    obs["celltype"] = cd["cluster_id"].astype(str)
    obs["Sample"] = cd["group_id"].astype(str)
    obs["SourceFile"] = cd["sample_id"].astype(str)

    # gene_info is one row per (gene, cluster); with nk=1 that is one row per gene
    gcol = "gene" if "gene" in gi.columns else gi.columns[0]
    gi = gi.drop_duplicates(subset=[gcol]).set_index(gcol)
    var = pd.DataFrame(index=pd.Index(genes, name=None))
    var["category"] = gi["category"].reindex(var.index).astype(str).values
    var["is_de"] = var["category"].isin(TRUE_DE)
    var["is_null"] = var["category"].isin(["ee", "ep"])
    if "logFC" in gi.columns:
        var["true_logFC"] = pd.to_numeric(gi["logFC"], errors="coerce").reindex(var.index).values

    A = ad.AnnData(X=X, obs=obs, var=var)
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    print("\n[build] design")
    print(pd.crosstab(A.obs["SourceFile"], A.obs["Sample"]).to_string())
    print("\n[build] ground truth")
    print(A.var["category"].value_counts().to_string())
    print(f"\n  true DE  ({'/'.join(sorted(TRUE_DE))}): {int(A.var['is_de'].sum())}")
    print(f"  true null (ee/ep)              : {int(A.var['is_null'].sum())}")
    unl = int((~A.var['is_de'] & ~A.var['is_null']).sum())
    if unl:
        print(f"  unlabelled                     : {unl}  <-- excluded from scoring")

    A.write_h5ad(a.out, compression="gzip")
    print(f"\n[build] wrote {a.out} ({Path(a.out).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
