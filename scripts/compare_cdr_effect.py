#!/usr/bin/env python
"""
compare_cdr_effect.py — does omitting the CDR covariate change the gene list?

The benchmark in the paper runs `~1+condition` because R MAST was run with the
same model, which makes the comparison apples-to-apples. For BIOLOGICAL analysis
the question is different: the cellular detection rate differs between the two
arms of the pancreas dataset (Endothelial: 0.0229 vs 0.0411, Cohen d = +1.01),
and a gene can look differential simply because one arm detects more genes
overall. This script measures how much that actually matters.

Two cell types are run deliberately:
  * Endothelial — large CDR imbalance (d = +1.01), where an effect is expected
  * Ductal      — small imbalance (d = -0.29), the built-in control where it is
                  not, so any difference seen in Endothelial cannot be blamed on
                  the covariate path being different in some other way

For each, `~1+condition` (vectorised path) is compared with `~1+CDR+condition`
(general path). The diagnostic that matters is not the overall overlap but
whether the genes that DISAGREE are the ones whose detection rate tracks the
global CDR shift — that is the signature of a detection-rate artefact.

Usage: python scripts/compare_cdr_effect.py [--celltypes "Endothelial cell,Ductal cell"]
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
from duet import run_duet                                    # noqa: E402
from duet.core import _compute_cdr                           # noqa: E402


def run(A, celltype, covariates, tag):
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(output_dir=tmp, celltype_col="celltype", sample_col="Sample",
                  ref_label="Reference", test_label="pancreas",
                  memory_log=False, workers=1, output_name=f"{tag}.csv")
        if covariates:
            # general per-gene path: mast_compat would clear the covariates
            kw.update(covariates=covariates, mast_compat=False, eb_shrinkage=True,
                      min_cells_per_group=1, min_total_cells=20, min_positive=2,
                      partial_hurdle=True)
        else:
            kw.update(mast_compat=True)          # clears covariates -> vectorised
        d = pd.read_csv(run_duet(A, **kw))
    secs = time.perf_counter() - t0
    d = d[d.tested == True].copy()                            # noqa: E712
    # the min.pct gate every benchmark in this project applies
    d = d[d[["detect_rate_ref", "detect_rate_test"]].max(axis=1) >= 0.10]
    print(f"    {tag:22s} {len(d):5d} genes tested, {secs:7.1f}s", flush=True)
    return d.set_index("gene"), secs


def compare(celltype, A, rows):
    print(f"\n=== {celltype} ===", flush=True)
    sub = A[A.obs.celltype.astype(str) == celltype].copy()
    cdr = _compute_cdr(sub.X, sub.n_vars)
    cond = np.where(sub.obs.Sample.astype(str).str.contains("Reference"),
                    "Reference", "pancreas")
    c_ref, c_test = cdr[cond == "Reference"], cdr[cond == "pancreas"]
    d_cohen = ((c_test.mean() - c_ref.mean())
               / np.sqrt((c_ref.var() + c_test.var()) / 2))
    print(f"    CDR ref {c_ref.mean():.4f} vs tumour {c_test.mean():.4f} "
          f"({c_test.mean()/c_ref.mean():.2f}x, Cohen d = {d_cohen:+.2f})", flush=True)

    no, s_no = run(sub, celltype, (), "no_covariate")
    yes, s_yes = run(sub, celltype, ("CDR",), "with_CDR")

    common = no.index.intersection(yes.index)
    a, b = no.loc[common], yes.loc[common]
    sa = set(a.index[a.fdr < 0.05])
    sb = set(b.index[b.fdr < 0.05])
    only_no, only_yes = sa - sb, sb - sa

    # Do the genes that LOSE significance when CDR is added look like
    # detection-rate artefacts? Compare their detection-rate shift against the
    # genes both models agree on.
    shift = (b.detect_rate_test - b.detect_rate_ref)
    agreed = sa & sb
    u_p = np.nan
    if len(only_no) >= 5 and len(agreed) >= 5:
        u_p = float(mannwhitneyu(shift.loc[list(only_no)],
                                 shift.loc[list(agreed)],
                                 alternative="two-sided").pvalue)

    row = dict(
        celltype=celltype, n_cells=int(sub.n_obs), cdr_cohen_d=round(float(d_cohen), 3),
        n_common=len(common),
        sig_no_covariate=len(sa), sig_with_CDR=len(sb),
        jaccard=round(len(sa & sb) / max(len(sa | sb), 1), 4),
        lost_when_CDR_added=len(only_no), gained_when_CDR_added=len(only_yes),
        neglog10p_spearman=round(float(spearmanr(a.neglog10p, b.neglog10p).statistic), 4),
        log2fc_pearson=round(float(np.corrcoef(a.coef.astype(float),
                                               b.coef.astype(float))[0, 1]), 4),
        median_detect_shift_lost=round(float(shift.loc[list(only_no)].median()), 4)
        if only_no else np.nan,
        median_detect_shift_agreed=round(float(shift.loc[list(agreed)].median()), 4)
        if agreed else np.nan,
        mannwhitney_p=u_p,
        seconds_no_covariate=round(s_no, 1), seconds_with_CDR=round(s_yes, 1),
    )
    rows.append(row)
    for k, v in row.items():
        print(f"      {k:28s} {v}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "PancreasIntegratedAnnotated.h5ad"))
    ap.add_argument("--celltypes", default="Endothelial cell,Ductal cell")
    ap.add_argument("--out", default=str(HERE / "evidence" / "cdr_covariate_effect.csv"))
    a = ap.parse_args()

    cts = [c.strip() for c in a.celltypes.split(",")]
    print(f"[cdr] loading {a.h5ad}", flush=True)
    A = sc.read_h5ad(a.h5ad)
    A = A[A.obs.celltype.astype(str).isin(cts)].copy()
    A.obs["condition"] = np.where(
        A.obs.Sample.astype(str).str.contains("Reference"), "Reference", "pancreas")

    rows = []
    for ct in cts:
        compare(ct, A, rows)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
