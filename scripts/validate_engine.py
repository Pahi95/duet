#!/usr/bin/env python
"""
validate_engine.py — is `numpy_irls` a safe replacement for `statsmodels` as the
logistic engine on the covariate path?

Profiling showed statsmodels' GLM accounts for ~78% of the covariate path's
runtime, and DUET's own IRLS produces the same fit roughly 6x more cheaply. Before
changing the default, the two engines must be shown to agree on more than one
dataset -- separation handling is where they could differ: statsmodels raises
PerfectSeparationError, while _logit_irls flags it heuristically (|beta| > 25 or
non-convergence).

Runs both engines on the same cells with a CDR covariate and reports timing plus
agreement on every quantity the paper relies on.

Usage: python validate_engine.py --h5ad data/crowell_4vs4.h5ad --tag crowell
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
from duet import run_duet                       # noqa: E402

ENGINES = ("statsmodels", "numpy_irls")


def run(A, outdir: Path, engine: str, **kw) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    p = run_duet(A, output_dir=str(outdir / engine), covariates=("CDR",),
                 mast_compat=False, memory_log=False, workers=1,
                 logistic_engine=engine, output_name=f"duet_{engine}.csv", **kw)
    return pd.read_csv(p, low_memory=False), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--celltype-col", default="celltype")
    ap.add_argument("--sample-col", default="Sample")
    ap.add_argument("--condition-col", default=None)
    ap.add_argument("--ref", default="Reference")
    ap.add_argument("--test", default="pancreas")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    out = Path(a.outdir or (HERE / "bench" / f"engine_{a.tag}"))
    out.mkdir(parents=True, exist_ok=True)

    A = sc.read_h5ad(a.h5ad)
    print(f"[{a.tag}] {A.shape[0]:,} cells x {A.shape[1]:,} genes", flush=True)

    kw = dict(celltype_col=a.celltype_col, sample_col=a.sample_col,
              ref_label=a.ref, test_label=a.test)
    if a.condition_col:
        kw["condition_col"] = a.condition_col

    res, times = {}, {}
    for eng in ENGINES:
        res[eng], times[eng] = run(A, out, eng, **kw)
        print(f"[{a.tag}] {eng:12s} {times[eng]:8.1f}s", flush=True)

    # Key on (gene, celltype), NOT gene alone: with more than one cell type the
    # gene labels repeat, and a gene-only reindex silently misaligns the two
    # frames. That bug reported a perfect Jaccard of 1.0 on Crowell and hid three
    # genuinely disagreeing calls.
    KEY = ["gene", "celltype"]
    s, n = res["statsmodels"].set_index(KEY), res["numpy_irls"].set_index(KEY)
    assert s.index.is_unique and n.index.is_unique, "(gene, celltype) is not unique"
    n = n.reindex(s.index)
    m = (s["tested"] == True) & (n["tested"] == True)          # noqa: E712
    s, n = s[m], n[m]

    row = {"dataset": a.tag, "n_genes_tested": int(m.sum()),
           "seconds_statsmodels": round(times["statsmodels"], 1),
           "seconds_numpy_irls": round(times["numpy_irls"], 1),
           "speedup": round(times["statsmodels"] / max(times["numpy_irls"], 1e-9), 2)}

    # agreement on every quantity a claim rests on
    for c in ("coef", "stat_hurdle", "neglog10p", "fdr"):
        if c in s.columns and c in n.columns:
            x, y = s[c].to_numpy(float), n[c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            row[f"{c}_max_absdiff"] = float(np.abs(x[ok] - y[ok]).max()) if ok.any() else np.nan
            row[f"{c}_pearson"] = (round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 6)
                                   if ok.sum() > 2 and x[ok].std() > 0 else np.nan)

    if "fdr" in s.columns:
        ss, sn = set(s.index[s.fdr < 0.05]), set(n.index[n.fdr < 0.05])
        row["sig_statsmodels"] = len(ss)
        row["sig_numpy_irls"] = len(sn)
        row["sig_jaccard"] = round(len(ss & sn) / max(len(ss | sn), 1), 6)
        row["sig_only_statsmodels"] = len(ss - sn)
        row["sig_only_numpy_irls"] = len(sn - ss)

    # separation is the one place the engines genuinely differ -- count it
    for eng, d in (("statsmodels", s), ("numpy_irls", n)):
        if "skip_reason" in res[eng].columns:
            sk = res[eng]["skip_reason"].astype(str)
            row[f"sep_{eng}"] = int((sk == "logistic_separation").sum())

    df = pd.DataFrame([row])
    df.to_csv(out / "engine_agreement.csv", index=False)
    print("\n" + "=" * 68)
    print(f"ENGINE EQUIVALENCE — {a.tag}")
    print("=" * 68)
    for k, v in row.items():
        print(f"  {k:26s} {v}")
    return df


if __name__ == "__main__":
    main()
