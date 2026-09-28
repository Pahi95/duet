#!/usr/bin/env python
"""
hpap_run_overlap.py -- do two sequencing runs of one HPAP library count the same molecules?

Each HPAP count file is one run, UMI-collapsed on its own. If run B re-sequences the library of
run A, a molecule can be counted in both, and summing the files counts it twice.

Test, per pair of runs of one donor, on the cells (barcodes) present in both. The 50 highest-count
genes are left out so that a few endocrine transcripts do not dominate. Model: the library holds
N molecules of a gene in a cell, and run r detects each one independently with probability p_r.
With r_c = UMIs_B / UMIs_A per cell,
    S = sum (b - r_c a)^2 / sum b  =  1 + r - 2 r p_A.
Independent molecule sets (p_A -> 0, or separate libraries) give S >= 1 + r; S < 1 + r means shared
molecules. The fitted p_A, p_B give the fraction of the summed UMIs that are repeats,
p_A p_B / (p_A + p_B). The per-run p is re-estimated in every pair, so agreement across the pairs of
a donor checks the model.

Output: evidence/hpap_run_overlap.csv
"""
from __future__ import annotations

import itertools
import os
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))

import importlib.util
_spec = importlib.util.spec_from_file_location("bpc", HERE / "scripts" / "build_pancreas_corrected.py")
bpc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bpc)


def main():
    A = ad.read_h5ad(INPUTS / "PancreasIntegratedAnnotated.h5ad", backed="r")
    src = A.obs["SourceFile"].astype(str).str.replace("_matrix.mtx", "", regex=False).to_numpy()
    idx = np.where(~pd.Series(src).str.startswith("GSM").to_numpy())[0]
    X = sp.csr_matrix(A[idx].to_memory().X)
    tc = A.obs["total_counts"].to_numpy(np.float64)[idx]
    C, _ = bpc.recover_counts(X, tc)
    src = src[idx]
    bc = pd.Series(A.obs_names[idx].astype(str)).str.extract(r"([ACGT]{14,16})", expand=False).to_numpy()
    info = {f: bpc.parse_hpap(bpc.FILE_TO_HPAP[f]) for f in np.unique(src)}
    o = pd.DataFrame({"file": src, "donor": [info[f][0] for f in src], "run": [info[f][2] for f in src],
                      "bc": bc, "i": np.arange(len(src))})
    rows = []
    for d, g in o.groupby("donor"):
        files = sorted(g["file"].unique(), key=int)
        for fa, fb in itertools.combinations(files, 2):
            ga = g[g["file"] == fa].set_index("bc"); gb = g[g["file"] == fb].set_index("bc")
            sh = ga.index.intersection(gb.index)
            row = dict(donor=d, file_a=f"{fa}_matrix.mtx", run_a=info[fa][2], file_b=f"{fb}_matrix.mtx",
                       run_b=info[fb][2], shared_cells=len(sh))
            if len(sh) >= 50:
                Xa = C[ga.loc[sh, "i"].to_numpy()].astype(np.float64)
                Xb = C[gb.loc[sh, "i"].to_numpy()].astype(np.float64)
                tot = np.asarray(Xa.sum(0) + Xb.sum(0)).ravel()
                keepg = tot > 0; keepg[np.argsort(-tot)[:50]] = False
                Xa, Xb = Xa[:, keepg], Xb[:, keepg]
                ta = np.asarray(Xa.sum(1)).ravel(); tb = np.asarray(Xb.sum(1)).ravel()
                ok = (ta > 0) & (tb > 0)
                Xa, Xb, ta, tb = Xa[ok], Xb[ok], ta[ok], tb[ok]
                r = tb / ta
                S = float((Xb - sp.diags(r) @ Xa).power(2).sum() / Xb.sum())
                rw = float((r * tb).sum() / tb.sum())
                pa = (1 + rw - S) / (2 * rw); pb = pa * float(Xb.sum() / Xa.sum())
                row.update(median_umi_a=float(np.median(ta)), median_umi_b=float(np.median(tb)),
                           depth_ratio_b_over_a=rw, S=S, S_if_independent=1 + rw,
                           p_a=pa, p_b=pb, repeat_fraction_of_sum=pa * pb / (pa + pb))
            rows.append(row)
    out = pd.DataFrame(rows)
    path = HERE / "evidence" / "hpap_run_overlap.csv"
    out.to_csv(path, index=False, float_format="%.4g")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"[overlap] wrote {path}")


if __name__ == "__main__":
    main()
