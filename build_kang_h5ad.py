#!/usr/bin/env python
"""
build_kang_h5ad.py
==================
Reshape the Kang et al. 2018 PBMC / IFN-beta dataset (data/kang_2018.h5ad, as
distributed by pertpy from exampledata.scverse.org) into the layout
run_de_comparison.py expects:

    layers['counts'] = raw integer counts   (the file ships raw counts in X)
    X                = log1p(normalize_total(counts, 1e4))
    obs['celltype']  = cell_type
    obs['Sample']    = label       (ctrl / stim)
    obs['SourceFile']= replicate   (the 8 patients -> pseudobulk replicates)

Identical normalization recipe to the pancreas and Crowell objects, so the three
datasets stay comparable.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(HERE / "data" / "kang_2018.h5ad"))
    ap.add_argument("--out", default=str(HERE / "data" / "kang_8donors.h5ad"))
    a = ap.parse_args()

    A = sc.read_h5ad(a.inp)
    print(f"[build] in: {A.n_obs} cells x {A.n_vars} genes")

    A.obs["celltype"] = A.obs["cell_type"].astype(str)
    A.obs["Sample"] = A.obs["label"].astype(str)           # ctrl / stim
    A.obs["SourceFile"] = A.obs["replicate"].astype(str)   # 8 patients

    X = A.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    A.X = X
    assert np.all(np.abs(X.data - np.round(X.data)) < 1e-9), \
        "X is not integer-valued -- expected raw counts"

    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    print("\n[build] design check (paired?)")
    print(pd.crosstab(A.obs["SourceFile"], A.obs["Sample"]).to_string())
    both = (pd.crosstab(A.obs["SourceFile"], A.obs["Sample"]) > 0).sum(axis=1)
    print(f"\ndonors with BOTH conditions: {int((both == 2).sum())} of {len(both)}")
    print("\n[build] cells per cell type x condition")
    print(pd.crosstab(A.obs["celltype"], A.obs["Sample"]).to_string())

    c = A.layers["counts"]
    print(f"\n[build] counts integer: True | max {float(c.max()):.0f}")
    print(f"[build] X log-normalized: max {float(A.X.max()):.2f}")

    A.write_h5ad(a.out, compression="gzip")
    print(f"\n[build] wrote {a.out} ({Path(a.out).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
