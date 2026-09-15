#!/usr/bin/env python
"""
fdr_scale_diag.py -- why the FDR of DUET/MAST differs so much from Wilcoxon / DESeq2
(Table 6). Reads the per-gene results sim_calls.py wrote; nothing is refitted.

Per muscat replicate, for the TRUE-NULL genes only (ee and ep, decision A4 -- an ep
gene has the same mixture in both groups), and per method:
  * raw type-I error at alpha = 0.05, median p and the KS distance from U(0,1):
    is the p-value itself valid before BH ever runs?
  * the BH denominator the method actually used (genes with a p-value / an adjusted p)
  * the false-positive rate at BH < 0.05, overall and by quartile of the per-gene
    design effect DE = 1 + (mbar - 1) * ICC, with ICC estimated from the between- and
    within-sample variance of the log1p-normalised expression.

Outputs: results/fdr_scale/fdr_scale_diag_summary.csv, fdr_scale_diag_pergene.csv
Usage:   python scripts/fdr_scale_diag.py [--calls results/sim_reps100/calls] [--out results/fdr_scale]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402
from scipy.stats import kstest                             # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", REPO.parent / "inputs"))
sys.path.insert(0, str(REPO))
from evaluate_sim_replicates import build                  # noqa: E402

NULL_CATS = ("ee", "ep")
FILES = {"DUET": "duet", "MAST": "mast", "wilcoxon": "wilcoxon", "DESeq2": "deseq2", "diffxpy": "diffxpy"}


def donor_stats(A, genes):
    """Between-sample (donor) and within-sample variance of log1p-normalised
    expression per gene, pooled within condition; ICC and design effect."""
    gi = pd.Index(A.var_names).get_indexer(np.asarray(genes, dtype=str))
    X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
    X = X[:, gi]
    smp = A.obs["SourceFile"].astype(str).to_numpy()
    grp = A.obs["Sample"].astype(str).to_numpy()
    means, ns, sgrp = [], [], []
    within_ss = np.zeros(X.shape[1])
    within_df = 0
    for s in pd.unique(smp):
        m = smp == s
        Xs = X[m]
        n = int(m.sum())
        mu = np.asarray(Xs.mean(axis=0)).ravel()
        within_ss += np.asarray(Xs.multiply(Xs).sum(axis=0)).ravel() - n * mu ** 2
        within_df += n - 1
        means.append(mu)
        ns.append(n)
        sgrp.append(grp[m][0])
    M, ns, sgrp = np.vstack(means), np.asarray(ns), np.asarray(sgrp)
    within_var = within_ss / max(within_df, 1)
    btw, bdf = np.zeros(X.shape[1]), 0
    for g in np.unique(sgrp):
        k = sgrp == g
        if k.sum() >= 2:
            btw += ((M[k] - M[k].mean(axis=0)) ** 2).sum(axis=0)
            bdf += int(k.sum()) - 1
    mbar = float(ns.mean())
    sigma_b2 = np.maximum(btw / max(bdf, 1) - within_var / mbar, 0.0)   # var(sample mean) = sb2 + sw2/mbar
    icc = sigma_b2 / np.maximum(sigma_b2 + within_var, 1e-12)
    return pd.DataFrame(dict(gene=np.asarray(genes, dtype=str), within_var=within_var, sigma_b2=sigma_b2,
                             icc=icc, design_effect=1.0 + (mbar - 1.0) * icc)), mbar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default=str(REPO / "results" / "sim_reps100" / "calls"))
    ap.add_argument("--out", default=str(REPO / "results" / "fdr_scale"))
    a = ap.parse_args()
    calls, out = Path(a.calls), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pergene, summary = [], []
    for repdir in sorted(calls.glob("sim_rep*")):
        tag = repdir.name
        if not all((repdir / f"{f}.csv").is_file() for f in FILES.values()):
            print(f"[diag] {tag}: not all methods present, skipped", flush=True)
            continue
        t0 = time.time()
        A = build(INPUTS / "data" / tag)
        cat = A.var["category"].astype(str)
        null_genes = cat.index[cat.isin(NULL_CATS)].astype(str)
        ds, mbar = donor_stats(A, null_genes)
        ds["category"] = cat.reindex(ds["gene"]).to_numpy()
        ds["de_quartile"] = pd.qcut(ds["design_effect"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        for m, f in FILES.items():
            d = pd.read_csv(repdir / f"{f}.csv", low_memory=False)
            d = d[d["tested"].astype(bool)].drop_duplicates("gene")
            d = d.set_index(pd.Index(d["gene"].astype(str), name=None))
            sub = d.reindex(ds["gene"])
            p, q = sub["pvalue"].to_numpy(float), sub["fdr"].to_numpy(float)
            ok = np.isfinite(p)
            called = np.nan_to_num(q < 0.05, nan=False).astype(bool)
            row = dict(replicate=tag, method=m, n_null=int(ok.sum()),
                       m_universe=int(d["pvalue"].notna().sum()), m_adjusted=int(d["fdr"].notna().sum()),
                       typeI_raw_p05=float((p[ok] < 0.05).mean()), median_p=float(np.median(p[ok])),
                       ks_uniform=float(kstest(np.clip(p[ok], 0, 1), "uniform").statistic),
                       fp_rate_fdr05=float(called[ok].mean()), mean_cells_per_sample=mbar)
            for c in NULL_CATS:
                sel = ok & (ds["category"].to_numpy() == c)
                row[f"typeI_raw_p05_{c}"] = float((p[sel] < 0.05).mean()) if sel.any() else np.nan
            for k in (1, 2, 3, 4):
                sel = (ds["de_quartile"].to_numpy() == k) & ok
                row[f"fp_q{k}"] = float(called[sel].mean())
            summary.append(row)
            g = ds.copy()
            g["replicate"], g["method"], g["pvalue"], g["fdr"] = tag, m, p, q
            pergene.append(g)
        print(f"[diag] {tag}: mbar={mbar:.0f}, median design effect {ds.design_effect.median():.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del A
    S = pd.DataFrame(summary)
    S.to_csv(out / "fdr_scale_diag_summary.csv", index=False)
    pd.concat(pergene).to_csv(out / "fdr_scale_diag_pergene.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["m_universe", "m_adjusted", "typeI_raw_p05", "median_p", "ks_uniform", "fp_rate_fdr05",
            "fp_q1", "fp_q2", "fp_q3", "fp_q4"]
    print(f"\n=== mean over {S.replicate.nunique()} replicates (true-null genes: {'+'.join(NULL_CATS)}) ===")
    print(S.groupby("method")[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
