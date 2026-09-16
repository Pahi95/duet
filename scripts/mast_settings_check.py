#!/usr/bin/env python
"""
mast_settings_check.py -- how closely does DUET reproduce MAST, component
by component, and under which MAST settings?

For each dataset: DUET (mast_compat: ~condition, no covariates, EB variance
moderation) and R MAST (scripts/mast_components.R) on the same cells, the same genes
(DUET's tested genes detected in >= 10% of cells of at least one group) and the same
contrast. MAST runs three ways: bayesglm + ebayes (its default), glm + ebayes, and
glm without ebayes.

Compared per gene: the detection (discrete) statistic, the continuous statistic, the
hurdle statistic, their degrees of freedom, the detection coefficient (DUET: the
logit difference of the two detection rates, which is the two-group logistic MLE),
the continuous coefficient, MAST's logFC vs DUET's coef, raw p and BH-adjusted p.

Tolerance fixed before running: two quantities agree when their relative difference
|a - b| / max(|a|, |b|, 1e-8) is at most 1e-3. Reported per component: Pearson and
Spearman correlation, the largest absolute and relative difference, the share of
genes within tolerance, and the genes furthest apart. High correlation alone is not
taken as evidence of the same test.

Outputs: evidence/mast_component_agreement.csv, results/mast_components/<dataset>/...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import math                                                # noqa: E402
import os                                                  # noqa: E402
import shutil                                              # noqa: E402
import subprocess                                          # noqa: E402
import tempfile                                            # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402
from scipy.io import mmwrite                               # noqa: E402
from scipy.stats import pearsonr, spearmanr                # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS                 # noqa: E402

RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
TEST = {"kang": "stim", "crowell": "LPS", "pancreas": "pancreas"}
CASES = [("kang", "CD14+ Monocytes"), ("crowell", "Astrocytes"), ("pancreas", "Ductal cell")]
CONFIGS = [("bayesglm", True), ("glm", True), ("glm", False)]
TOL = 1e-3
warnings.filterwarnings("ignore")


def rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def run_duet_case(A, ref, test):
    from duet import run_duet
    with tempfile.TemporaryDirectory() as tmp:
        out = run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="cond", ref_label=ref,
                       test_label=test, mast_compat=True, vectorized=True, memory_log=False, output_name="d.csv")
        d = pd.read_csv(out, low_memory=False)
    d = d[d.tested == True].copy()                                                   # noqa: E712
    d = d[d[["detect_rate_ref", "detect_rate_test"]].max(axis=1) >= 0.10]
    return d.set_index(d["gene"].astype(str))


def compare(duet: pd.DataFrame, mast: pd.DataFrame) -> list[dict]:
    g = duet.index.intersection(mast.index)
    D, M = duet.loc[g], mast.loc[g]
    pairs = {
        "stat_detect": (D["stat_detect"], M["stat_disc"]),
        "stat_continuous": (D["stat_continuous"], M["stat_cont"]),
        "stat_hurdle": (D["stat_hurdle"], M["stat_hurdle"]),
        "df_hurdle": (D["df_hurdle"], M["df_hurdle"]),
        "coef_detect": (logit(D["detect_rate_test"]) - logit(D["detect_rate_ref"]), M["coef_disc"]),
        "coef_continuous": (D["coef_continuous"], M["coef_cont"]),
        "logFC": (D["coef"], M["logFC"]),
        "neglog10_p": (D["neglog10p"], -np.log10(M["pvalue"].clip(lower=1e-300))),
        "neglog10_fdr": (-np.log10(D["fdr"].clip(lower=1e-300)), -np.log10(M["fdr"].clip(lower=1e-300))),
    }
    rows = []
    for comp, (a, b) in pairs.items():
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if comp.startswith("neglog10"):
            # a p-value stored as 0.0 carries no magnitude; compare where both are above the floor
            ok &= (a < 299) & (b < 299)
        r = rel_diff(a[ok], b[ok])
        worst = np.argsort(-r)[:3]
        rows.append(dict(component=comp, n_genes=int(ok.sum()),
                         pearson=pearsonr(a[ok], b[ok]).statistic if ok.sum() > 2 else np.nan,
                         spearman=spearmanr(a[ok], b[ok]).statistic if ok.sum() > 2 else np.nan,
                         max_abs_diff=float(np.abs(a[ok] - b[ok]).max()) if ok.any() else np.nan,
                         max_rel_diff=float(r.max()) if ok.any() else np.nan,
                         median_rel_diff=float(np.median(r)) if ok.any() else np.nan,
                         share_within_tol=float((r <= TOL).mean()) if ok.any() else np.nan,
                         worst_genes=";".join(np.asarray(g)[ok][worst])))
    sa, sb = set(g[(D["fdr"] < 0.05).to_numpy()]), set(g[(M["fdr"] < 0.05).to_numpy()])
    rows.append(dict(component="sig_calls_jaccard", n_genes=len(g),
                     share_within_tol=len(sa & sb) / max(len(sa | sb), 1),
                     max_abs_diff=float(len(sa ^ sb))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "results" / "mast_components"))
    ap.add_argument("--out", default=str(HERE / "evidence" / "mast_component_agreement.csv"))
    a = ap.parse_args()
    rows = []
    for name, ct in CASES:
        cfg = DATASETS[name]
        ref, test = cfg["ref"], TEST[name]
        A = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == ct).to_numpy(), layers=())
        A.obs["cond"] = A.obs[cfg["cond"]].astype(str)
        A = A[A.obs.cond.isin([ref, test])].copy()
        duet = run_duet_case(A, ref, test)
        d = Path(a.outdir) / name
        d.mkdir(parents=True, exist_ok=True)
        duet.to_csv(d / "duet.csv")
        genes = duet.index.to_numpy()
        X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
        mmwrite(str(d / "expr.mtx"), X[:, pd.Index(A.var_names.astype(str)).get_indexer(genes)].T.tocoo(),
                field="real", precision=7)
        np.savetxt(d / "genes.txt", genes, fmt="%s")
        np.savetxt(d / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
        np.savetxt(d / "cond.txt", A.obs.cond.to_numpy(), fmt="%s")
        (d / "labels.json").write_text(json.dumps({"ref": ref, "test": test}), encoding="utf-8")
        for method, eb in CONFIGS:
            f = d / f"mast_{method}_eb{str(eb).upper()}.csv"
            if not f.is_file():
                r = subprocess.run([RSCRIPT, str(HERE / "scripts" / "mast_components.R"), str(d), method,
                                    str(eb).upper()], capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[{name}] MAST {method}/{eb} FAILED: {r.stderr[-600:]}", flush=True)
                    continue
                print(r.stdout.strip().splitlines()[-1], flush=True)
            m = pd.read_csv(f)
            m = m.set_index(m["gene"].astype(str))
            for row in compare(duet, m):
                rows.append(dict(dataset=name, celltype=ct, n_cells=A.n_obs, mast_method=method,
                                 mast_ebayes=eb, **row))
        (d / "expr.mtx").unlink(missing_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print(R[["dataset", "mast_method", "mast_ebayes", "component", "n_genes", "pearson", "max_rel_diff",
             "share_within_tol"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
