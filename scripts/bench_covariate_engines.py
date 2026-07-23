#!/usr/bin/env python
"""
bench_covariate_engines.py — a defensible timing for the covariate path.

The manuscript's covariate speed claim (DUET 210 s vs MAST 118 s) came from runs
made at different times under different machine load, and a later measurement put
MAST at 63 s on the same input. Load skews both sides, so the two figures cannot be
compared. This script times all three configurations back to back in ONE process
with nothing else running, which is the only comparison worth quoting.

  DUET  statsmodels   (the current default)
  DUET  numpy_irls    (same model, DUET's own IRLS)
  MAST  zlm + lrTest  (fit and test only, no I/O)

Run it on an otherwise idle machine:
    python bench_covariate_engines.py --repeats 2
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
import scanpy as sc

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
from duet import run_duet                       # noqa: E402
from duet.core import _compute_cdr              # noqa: E402

RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")


def time_duet(A, engine, ref, test, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="Sample",
                     ref_label=ref, test_label=test, covariates=("CDR",),
                     mast_compat=False, min_cells_per_group=1, min_total_cells=20,
                     min_positive=2, partial_hurdle=True, eb_shrinkage=True,
                     vectorized=True, memory_log=False, workers=1,
                     logistic_engine=engine, output_name="d.csv")
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "data" / "kang_8donors.h5ad"))
    ap.add_argument("--celltype", default="CD14+ Monocytes")
    ap.add_argument("--ref", default="ctrl")
    ap.add_argument("--test", default="stim")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--outdir", default=str(HERE / "bench" / "covariate_timing"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    A = sc.read_h5ad(a.h5ad)
    A = A[A.obs["celltype"].astype(str) == a.celltype].copy()
    cond = A.obs["Sample"].astype(str).to_numpy()
    print(f"[bench] {a.celltype}: {A.n_obs} cells x {A.n_vars} genes, "
          f"repeats={a.repeats}", flush=True)

    rows = []
    for eng in ("statsmodels", "numpy_irls"):
        ts = time_duet(A, eng, a.ref, a.test, a.repeats)
        rows.append(dict(method=f"DUET ({eng})", best=min(ts), median=float(np.median(ts)),
                         runs=json.dumps([round(t, 1) for t in ts])))
        print(f"[bench] DUET {eng:12s} best {min(ts):7.1f}s  runs={[round(t,1) for t in ts]}",
              flush=True)

    # MAST on the identical cells/genes/covariate, fit + test only
    duet_genes = None
    with tempfile.TemporaryDirectory() as tmp:
        p = run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="Sample",
                     ref_label=a.ref, test_label=a.test, covariates=("CDR",),
                     mast_compat=False, min_cells_per_group=1, min_total_cells=20,
                     min_positive=2, partial_hurdle=True, eb_shrinkage=True,
                     vectorized=True, memory_log=False, workers=1,
                     logistic_engine="numpy_irls", output_name="d.csv")
        d = pd.read_csv(p)
    d = d[d["tested"] == True]                                     # noqa: E712
    d = d[d[["detect_rate_ref", "detect_rate_test"]].max(axis=1) >= 0.10]
    duet_genes = d["gene"].astype(str).to_numpy()

    X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
    gi = pd.Index(A.var_names).get_indexer(duet_genes)
    mmwrite(str(out / "expr.mtx"), X[:, gi].T.tocoo(), field="real", precision=7)
    np.savetxt(out / "genes.txt", duet_genes, fmt="%s")
    np.savetxt(out / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
    np.savetxt(out / "cond.txt", cond, fmt="%s")
    np.savetxt(out / "cdr.txt", _compute_cdr(A.X, A.n_vars), fmt="%.10g")
    (out / "labels.json").write_text(json.dumps({"ref": a.ref, "test": a.test}),
                                     encoding="utf-8")

    mast_ts = []
    for _ in range(a.repeats):
        r = subprocess.run([RSCRIPT, str(HERE / "validate_covariate.R"), str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:]); raise SystemExit("MAST failed")
        secs = [float(l.split(":")[-1].strip().rstrip("s"))
                for l in r.stdout.splitlines()
                if l.startswith("[mast]") and ("fit:" in l or "lrTest" in l)]
        mast_ts.append(sum(secs))
        print(f"[bench] MAST  fit+test    {sum(secs):7.1f}s  ({secs})", flush=True)
    rows.append(dict(method="MAST (zlm+lrTest)", best=min(mast_ts),
                     median=float(np.median(mast_ts)),
                     runs=json.dumps([round(t, 1) for t in mast_ts])))

    df = pd.DataFrame(rows)
    df["n_genes"] = len(duet_genes)
    df["n_cells"] = A.n_obs
    sm = df.loc[df.method.str.contains("statsmodels"), "best"].iloc[0]
    ni = df.loc[df.method.str.contains("numpy_irls"), "best"].iloc[0]
    ma = df.loc[df.method.str.contains("MAST"), "best"].iloc[0]
    df["vs_mast"] = [round(ma / sm, 2), round(ma / ni, 2), 1.0]
    df.to_csv(out / "covariate_timing.csv", index=False)

    print("\n" + "=" * 66)
    print("COVARIATE PATH — best of", a.repeats, "runs, isolated")
    print("=" * 66)
    print(df[["method", "best", "median", "vs_mast"]].to_string(index=False))
    print("\nvs_mast > 1 means DUET is faster than MAST.")
    print(f"engine speed-up: {sm/ni:.2f}x")


if __name__ == "__main__":
    main()
