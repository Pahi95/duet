#!/usr/bin/env python
"""
run_mast_dataset.py
===================
Run R MAST as the 5th method on a whole dataset, one cell type at a time, and
write `de_MAST_all_celltypes.csv` in the harmonized schema next to the other
per-method files.

This generalizes the two-step benchmark scripts (bench_mast_export.py +
bench_mast_run.R) from a single cell type to the cell types the Python pipeline
actually used, so the five methods are compared on the same footing:

  * cell types are read from the existing de_DUET_all_celltypes.csv, so MAST sees
    exactly the cell types the other methods ran on;
  * genes are the ones DUET tested for that cell type (its mast_compat gate plus
    the min.pct filter), so neither engine is charged for filtering the other
    would not do;
  * the design is ~1+condition with no covariates, matching DUET's mast_compat.

MAST's `coef` is a natural-log mean difference -> log2 via / ln2. MAST keeps its
hurdle likelihood-ratio statistic (lrTest lambda, written by bench_mast_run.R as
stat_hurdle with df_hurdle), so `neglog10p` is recovered from it with the same
underflow-safe chi-square tail DUET uses; a p-value stored as 0.0 still ranks.

Usage:
  python run_mast_dataset.py --h5ad data/kang_8donors.h5ad \
      --results-dir results/kang --ref-label ctrl --test-label stim
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
import anndata as ad

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
import run_de_comparison as R          # noqa: E402

LN2 = math.log(2.0)
RSCRIPT_DEFAULT = r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ref-label", required=True)
    ap.add_argument("--test-label", required=True)
    ap.add_argument("--celltype-col", default="celltype")
    ap.add_argument("--sample-col", default="Sample")
    ap.add_argument("--rscript", default=RSCRIPT_DEFAULT)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    res = Path(a.results_dir)
    duet_csv = res / "de_DUET_all_celltypes.csv"
    if not duet_csv.is_file():
        raise SystemExit(f"need {duet_csv} first (run run_de_comparison.py)")
    duet = pd.read_csv(duet_csv, low_memory=False)
    duet = duet[duet["tested"] == True]                       # noqa: E712
    celltypes = list(duet["celltype"].dropna().unique())
    comparison = str(duet["comparison"].dropna().iloc[0])
    print(f"[mast] cell types: {celltypes}")
    print(f"[mast] comparison: {comparison}")

    work = Path(a.workdir) if a.workdir else res / "_mast_work"
    work.mkdir(parents=True, exist_ok=True)

    from _runtime import load_rows, maybe_background, read_obs
    from duet.core import _neglog10_chi2_sf
    maybe_background()
    obs = read_obs(a.h5ad)                         # rows are read per cell type below
    ct_all = obs[a.celltype_col].astype(str).to_numpy()
    samp = obs[a.sample_col].astype(str)
    cond_all = np.where(samp.str.contains(a.ref_label, case=False, na=False),
                        a.ref_label, a.test_label)

    frames, timings = [], []
    for ct in celltypes:
        genes = duet.loc[duet.celltype == ct, "gene"].astype(str).to_numpy()
        idx = np.flatnonzero(ct_all == ct)
        cond = cond_all[idx]
        n_ref = int((cond == a.ref_label).sum())
        n_test = int((cond == a.test_label).sum())
        d = work / ct.replace(" ", "_").replace("/", "-")
        d.mkdir(exist_ok=True)
        print(f"\n[mast] === {ct}: {len(idx)} cells "
              f"(ref={n_ref}, test={n_test}), {len(genes)} genes ===", flush=True)

        # export exactly DUET's gene set, genes x cells, as MAST expects
        t0 = time.perf_counter()
        view = load_rows(a.h5ad, ct_all == ct, layers=())
        X = view.X
        X = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
        gi = pd.Index(view.var_names).get_indexer(genes)
        assert (gi >= 0).all(), f"gene mismatch in {ct}"
        mmwrite(str(d / "expr.mtx"), X[:, gi].T.tocoo(), field="real", precision=7)
        np.savetxt(d / "genes.txt", genes, fmt="%s")
        np.savetxt(d / "cells.txt", np.asarray(view.obs_names, dtype=str), fmt="%s")
        # R side expects the reference level first; write the raw labels
        np.savetxt(d / "cond.txt", cond, fmt="%s")
        (d / "labels.json").write_text(json.dumps(
            {"ref": a.ref_label, "test": a.test_label}), encoding="utf-8")
        print(f"[mast] exported in {time.perf_counter()-t0:.1f}s", flush=True)

        r = subprocess.run([a.rscript, str(HERE / "scripts" / "bench_mast_run.R"), str(d), "1"],
                           capture_output=True, text=True)
        sys.stdout.write("\n".join(l for l in r.stdout.splitlines()
                                   if l.startswith("[mast]")) + "\n")
        if r.returncode != 0:
            print(f"[mast] !! {ct} FAILED (rc={r.returncode})")
            print(r.stderr[-1500:])
            continue

        m = pd.read_csv(d / "mast_results.csv")
        tj = json.loads((d / "mast_timing.json").read_text(encoding="utf-8"))
        tj.update(celltype=ct, n_cells=len(idx))
        timings.append(tj)
        tested = m["pvalue"].notna() & m["coef"].notna()
        frames.append(pd.DataFrame({
            "method": "MAST", "celltype": ct, "comparison": comparison,
            "gene": m["gene"].astype(str),
            "pvalue": pd.to_numeric(m["pvalue"], errors="coerce"),
            "fdr": pd.to_numeric(m["fdr"], errors="coerce"),
            "log2fc": pd.to_numeric(m["coef"], errors="coerce") / LN2,
            "neglog10p": _neglog10_chi2_sf(pd.to_numeric(m["stat_hurdle"], errors="coerce").to_numpy(float),
                                           pd.to_numeric(m["df_hurdle"], errors="coerce").to_numpy(float)),
            "stat": pd.to_numeric(m["stat_hurdle"], errors="coerce"), "n_ref": n_ref, "n_test": n_test,
            "tested": tested.values,
            "skip_reason": np.where(tested.values, "", "untested_or_na"),
        })[R.HARMON])

    if not frames:
        raise SystemExit("[mast] no cell type succeeded")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(res / "de_MAST_all_celltypes.csv", index=False)
    pd.DataFrame(timings).to_csv(res / "mast_timings.csv", index=False)
    print(f"\n[mast] wrote {res/'de_MAST_all_celltypes.csv'} "
          f"({len(out)} rows, tested={int(out.tested.sum())})")
    print(pd.DataFrame(timings)[["celltype", "n_cells", "n_genes", "fit_s",
                                 "lrt_s", "mast_total_s", "peak_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
