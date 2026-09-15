#!/usr/bin/env python
"""
eb_logspace_report.py -- MO task 5: what removing DUET's 1e-300 clip changes.

The empirical-Bayes step used to clip the two-sided moderated-t p-value at 1e-300,
capping the continuous statistic at chi2.isf(1e-300, 1) = 1373.87. It now carries
the tail in log space (duet.core._moderated_t_to_chi2).

Part A -- accuracy. The log tail and the resulting chi2_1 statistic against mpmath
          over df from 3 to 1e6 and tails from 1e-300 to beyond 1e-10000.
Part B -- real data. DUET on the pancreas Ductal population, old (clipped) versus
          new engine: genes at the old cap, ordering inside that set, global
          ranking, top-k lists and significance calls. Run on the corrected object
          and, for continuity with the manuscript's 1.19%, on the original one.
Part C -- p-value underflow after the fix: share of genes whose pvalue is stored
          as 0, and how far a p-value ranking (ties broken by gene order, over
          random permutations of that order) departs from the neglog10p ranking.

Outputs: evidence/eb_logspace_accuracy.csv, evidence/eb_logspace_realdata.csv,
         evidence/eb_logspace_underflow.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr, t as student_t

HERE = Path(__file__).resolve().parent.parent            # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
import duet.core as core                                  # noqa: E402

warnings.filterwarnings("ignore")
OLD_CAP = float(chi2.isf(1e-300, 1))
TOPK = (20, 100, 500)


def _old_moderated_t_to_chi2(tval, post_df):
    """The engine as it was before the revision: clip at 1e-300."""
    p = np.clip(2.0 * student_t.sf(np.abs(tval), post_df), 1e-300, 1.0)
    return p, chi2.isf(p, 1), np.log(p)


# ---------------------------------------------------------------- Part A
def _mp_log_tail(t, df, mp):
    t, nu = mp.mpf(abs(t)), mp.mpf(df)
    if df <= 1e4:
        x = nu / (nu + t * t)
        return mp.log(mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True))
    const = mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2) - mp.log(nu * mp.pi) / 2
    logpdf = lambda u: const - (nu + 1) / 2 * mp.log1p(u * u / nu)      # noqa: E731
    ref = logpdf(t)
    pts = sorted({mp.mpf(0), 1 / t, 10 / t, 100 / t, t / 10, t, 10 * t})
    return mp.log(2) + ref + mp.log(mp.quad(lambda s: mp.exp(logpdf(t + s) - ref), pts + [mp.inf]))


def part_a() -> pd.DataFrame:
    import mpmath as mp
    mp.mp.dps = 60
    rows = []
    for df in (3.0, 10.0, 30.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        if df <= 3:
            ts = np.array([1e110, 1e150, 1e200, 1e300])
        else:
            base = float(student_t.isf(1e-300, df))
            ts = base * np.array([1.0001, 1.05, 1.5, 3.0, 10.0, 50.0])
        _, stat, logp = core._moderated_t_to_chi2(ts, df)
        for t, s, lp in zip(ts, stat, logp):
            ref = _mp_log_tail(t, df, mp)
            # chi2_1 statistic with the same tail: erfc(sqrt(s/2)) = p
            s_ref = mp.findroot(lambda x: mp.log(mp.erfc(mp.sqrt(x / 2))) - ref, mp.mpf(s))
            rows.append(dict(df=df, t=t, log10_p=float(ref / mp.log(10)),
                             rel_err_log_p=abs(lp - float(ref)) / abs(float(ref)),
                             chi2_stat=s, rel_err_chi2=abs(s - float(s_ref)) / float(s_ref),
                             old_chi2_stat=OLD_CAP))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part B / C
def _ductal(h5ad: Path):
    import anndata as ad
    a = ad.read_h5ad(h5ad, backed="r")
    m = (a.obs["celltype"].astype(str) == "Ductal cell").to_numpy()
    sub = a[m].to_memory()
    a.file.close()
    for k in list(sub.layers):
        del sub.layers[k]
    return sub


def _run(adata, engine) -> pd.DataFrame:
    orig = core._moderated_t_to_chi2
    core._moderated_t_to_chi2 = engine
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = core.run_duet(adata, output_dir=tmp, celltype_col="celltype", sample_col="Sample",
                                ref_label="Reference", test_label="pancreas", mast_compat=True,
                                vectorized=True, memory_log=False, output_name="d.csv")
            d = pd.read_csv(out, low_memory=False)
    finally:
        core._moderated_t_to_chi2 = orig
    return d[d.tested == True].set_index("gene")                            # noqa: E712


def _overlap(a: pd.Series, b: pd.Series, k: int) -> int:
    return len(set(a.nlargest(k).index) & set(b.nlargest(k).index))


def part_bc(label: str, h5ad: Path, n_perm: int = 50):
    A = _ductal(h5ad)
    n_ref = int((A.obs["Sample"].astype(str) == "Reference").sum())
    print(f"[{label}] Ductal: {A.n_obs} cells ({n_ref} reference), {A.n_vars} genes", flush=True)
    old = _run(A, _old_moderated_t_to_chi2)
    new = _run(A, core._moderated_t_to_chi2).reindex(old.index)
    sat = old["stat_continuous"] >= OLD_CAP - 1e-6
    unsat = ~sat
    row = dict(
        dataset=label, n_cells=A.n_obs, n_reference=n_ref, n_tested=len(old),
        n_at_old_cap=int(sat.sum()), pct_at_old_cap=round(100 * sat.mean(), 3),
        distinct_new_stats_among_capped=int(new.loc[sat, "stat_continuous"].round(6).nunique()),
        max_new_stat_continuous=float(new["stat_continuous"].max()),
        spearman_within_capped=float(spearmanr(old.loc[sat, "neglog10p"], new.loc[sat, "neglog10p"]).statistic)
        if sat.sum() > 2 else np.nan,
        spearman_global=float(spearmanr(old["neglog10p"], new["neglog10p"]).statistic),
        max_abs_change_uncapped=float((new.loc[unsat, "neglog10p"] - old.loc[unsat, "neglog10p"]).abs().max()),
        sig_calls_changed=int(((old["fdr"] < 0.05) != (new["fdr"] < 0.05)).sum()),
    )
    for k in TOPK:
        row[f"top{k}_overlap_old_vs_new"] = _overlap(old["neglog10p"], new["neglog10p"], k)

    # Part C: pvalue underflow in the new output
    zero = new["pvalue"] == 0
    rng = np.random.default_rng(0)
    under = dict(dataset=label, n_tested=len(new), n_pvalue_zero=int(zero.sum()),
                 pct_pvalue_zero=round(100 * zero.mean(), 3),
                 neglog10p_min_among_zero=float(new.loc[zero, "neglog10p"].min()) if zero.any() else np.nan,
                 neglog10p_max_among_zero=float(new.loc[zero, "neglog10p"].max()) if zero.any() else np.nan)
    for k in TOPK:
        ov = []
        for _ in range(n_perm):
            order = rng.permutation(len(new))
            by_p = new.iloc[order].sort_values("pvalue", kind="stable").index[:k]
            ov.append(len(set(by_p) & set(new["neglog10p"].nlargest(k).index)))
        under[f"top{k}_overlap_pvalue_vs_neglog10p_min"] = int(np.min(ov))
        under[f"top{k}_overlap_pvalue_vs_neglog10p_median"] = float(np.median(ov))
        under[f"top{k}_overlap_pvalue_vs_neglog10p_max"] = int(np.max(ov))
    return row, under


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected", default=str(INPUTS / "PancreasCorrected.h5ad"))
    ap.add_argument("--original", default=str(INPUTS / "PancreasIntegratedAnnotated.h5ad"))
    ap.add_argument("--outdir", default=str(HERE / "evidence"))
    a = ap.parse_args()
    out = Path(a.outdir)

    acc = part_a()
    acc.to_csv(out / "eb_logspace_accuracy.csv", index=False)
    print(f"[A] max relative error: log p {acc.rel_err_log_p.max():.2e}, chi2 {acc.rel_err_chi2.max():.2e} "
          f"over {len(acc)} points (log10 p down to {acc.log10_p.min():.0f})", flush=True)

    rows, unders = [], []
    for label, path in (("corrected", a.corrected), ("original", a.original)):
        if not Path(path).is_file():
            print(f"[{label}] {path} missing, skipped")
            continue
        r, u = part_bc(label, Path(path))
        rows.append(r); unders.append(u)
        print(pd.Series(r).to_string(), "\n", pd.Series(u).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(out / "eb_logspace_realdata.csv", index=False)
    pd.DataFrame(unders).to_csv(out / "eb_logspace_underflow.csv", index=False)
    print(f"[eb] wrote {out / 'eb_logspace_accuracy.csv'}, eb_logspace_realdata.csv, eb_logspace_underflow.csv")


if __name__ == "__main__":
    main()
