#!/usr/bin/env python
"""
validate_covariate.py — does DUET still reproduce R MAST when a covariate is present?

Every benchmark so far used mast_compat=True, which sets covariates=() — and DUET's
vectorised fast path applies ONLY when there are no covariates. So the general
per-gene path, which is what runs whenever a covariate is included, has never been
checked against MAST. MAST's own recommended workflow includes the cellular
detection rate (`cngeneson`), so this is the configuration many users will run.

Design here: ~1 + cngeneson + condition on both sides, same cells, same genes.
The CDR is computed once in Python (DUET's own definition: positive genes / total
genes) and exported, so MAST receives the identical covariate values and any
difference is attributable to the model fit rather than to the covariate.
Both engines z-score / scale it, which does not change the condition LRT.

Usage: python validate_covariate.py --celltype "CD14+ Monocytes"
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from scipy.stats import spearmanr
import anndata as ad
import scanpy as sc

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
from duet import run_duet                       # noqa: E402
from duet.core import _compute_cdr              # noqa: E402

RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
LN2 = np.log(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "data" / "kang_8donors.h5ad"))
    ap.add_argument("--celltype", default="CD14+ Monocytes")
    ap.add_argument("--ref", default="ctrl")
    ap.add_argument("--test", default="stim")
    ap.add_argument("--min-detect-frac", type=float, default=0.10)
    ap.add_argument("--outdir", default=str(HERE / "bench" / "covariate"))
    # The logistic engine dominates this path: statsmodels' GLM accounts for ~78%
    # of the runtime, and DUET's own IRLS gives the same fit far more cheaply.
    ap.add_argument("--engine", default="statsmodels",
                    choices=["statsmodels", "numpy_irls"])
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    A = sc.read_h5ad(a.h5ad)
    A = A[A.obs["celltype"].astype(str) == a.celltype].copy()
    cond = A.obs["Sample"].astype(str).to_numpy()
    A.obs["condition"] = cond
    n_ref = int((cond == a.ref).sum()); n_test = int((cond == a.test).sum())
    print(f"[cov] {a.celltype}: {A.n_obs} cells (ref={n_ref}, test={n_test}), "
          f"{A.n_vars} genes", flush=True)

    # ---- DUET with the CDR covariate -------------------------------------
    # mast_compat would force covariates=(), so its gating is set explicitly and
    # mast_compat left off; everything else matches the benchmark configuration.
    print("[cov] DUET  ~1 + CDR + condition (general per-gene path) ...", flush=True)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        p = run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="Sample",
                     ref_label=a.ref, test_label=a.test,
                     covariates=("CDR",), mast_compat=False,
                     min_cells_per_group=1, min_total_cells=20, min_positive=2,
                     partial_hurdle=True, eb_shrinkage=True, vectorized=True,
                     memory_log=False, output_name="duet_cov.csv",
                     logistic_engine=a.engine)
        duet = pd.read_csv(p)
    duet_s = time.perf_counter() - t0
    duet = duet[duet["tested"] == True].copy()                    # noqa: E712
    maxdet = duet[["detect_rate_ref", "detect_rate_test"]].max(axis=1)
    duet = duet[maxdet >= a.min_detect_frac].copy()
    genes = duet["gene"].astype(str).to_numpy()
    print(f"[cov] DUET: {len(genes)} genes in {duet_s:.1f}s", flush=True)
    duet.to_csv(out / "duet_cov_results.csv", index=False)

    # ---- export the same cells, genes and CDR for MAST -------------------
    X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
    cdr = _compute_cdr(A.X, A.n_vars)          # DUET's own definition
    gi = pd.Index(A.var_names).get_indexer(genes)
    assert (gi >= 0).all()
    mmwrite(str(out / "expr.mtx"), X[:, gi].T.tocoo(), field="real", precision=7)
    np.savetxt(out / "genes.txt", genes, fmt="%s")
    np.savetxt(out / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
    np.savetxt(out / "cond.txt", cond, fmt="%s")
    np.savetxt(out / "cdr.txt", cdr, fmt="%.10g")
    (out / "labels.json").write_text(json.dumps({"ref": a.ref, "test": a.test}),
                                     encoding="utf-8")
    print(f"[cov] exported {len(genes)} genes x {A.n_obs} cells + CDR", flush=True)

    # ---- MAST with cngeneson --------------------------------------------
    r = subprocess.run([RSCRIPT, str(HERE / "validate_covariate.R"), str(out)],
                       capture_output=True, text=True)
    sys.stdout.write("\n".join(l for l in r.stdout.splitlines()
                               if l.startswith("[mast]")) + "\n")
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit("MAST failed")
    mast = pd.read_csv(out / "mast_cov_results.csv")

    # ---- compare ---------------------------------------------------------
    j = mast.merge(duet, on="gene", suffixes=("_m", "_d")).dropna(
        subset=["pvalue_m", "pvalue_d", "coef_m", "coef_d"])
    lfc_m = j["coef_m"] / LN2
    lfc_d = j["coef_d"] / LN2
    nlp_m = -np.log10(j["pvalue_m"].clip(1e-300))
    nlp_d = j["neglog10p"] if "neglog10p" in j else -np.log10(j["pvalue_d"].clip(1e-300))
    sm = j["fdr_m"] < 0.05
    sd = j["fdr_d"] < 0.05
    res = dict(
        celltype=a.celltype, model="~1 + CDR/cngeneson + condition",
        engine=a.engine, n_genes=len(j), duet_seconds=round(duet_s, 1),
        log2fc_pearson=round(float(lfc_m.corr(lfc_d)), 4),
        log2fc_spearman=round(float(lfc_m.corr(lfc_d, method="spearman")), 4),
        sign_agree_pct=round(100 * float((np.sign(lfc_m) == np.sign(lfc_d)).mean()), 2),
        neglog10p_spearman=round(float(spearmanr(nlp_m, nlp_d).statistic), 4),
        top100_overlap=len(set(nlp_d.nlargest(100).index) & set(nlp_m.nlargest(100).index)),
        sig_jaccard=round(float((sm & sd).sum() / max((sm | sd).sum(), 1)), 4),
        sig_duet=int(sd.sum()), sig_mast=int(sm.sum()))
    pd.DataFrame([res]).to_csv(out / "covariate_agreement.csv", index=False)
    print("\n" + "=" * 70)
    print("DUET vs R MAST WITH A COVARIATE (general per-gene path)")
    print("=" * 70)
    for k, v in res.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
