#!/usr/bin/env python
"""
simulate_batch_nb_rdeseq2.py -- the pseudobulk arm of the negative-binomial global nulls with R DESeq2.

simulate_batch_nb.py fitted the pseudobulk arm (~condition, 4 vs 4 donors x 3,000 cells) with
PyDESeq2. Under the 'none' scenario it still made a discovery in 5 of 20 replicates, above the
nominal 0.05. This script refits the same replicates (inputs/data/batchsim/<scenario>_repNN) with
duet.pseudobulk.run_pseudobulk(engine="R"), same arguments otherwise, to see whether the excess
belongs to the engine.

Outputs: results/batchsim/per_rep_rdeseq2.csv, results/batchsim/summary_rdeseq2.csv
Usage:   python scripts/simulate_batch_nb_rdeseq2.py [--reps 20]   (RSCRIPT=... selects Rscript)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.limit_threads(1)

import argparse                                            # noqa: E402
import os                                                  # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from simulate_batch_nb import OUT_DATA, OUT_RES, SCENARIOS, ALPHA   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--dump", default=None, help="only write the pseudobulk matrices of the 'none' and "
                    "'donor' scenarios to this folder (input of simulate_batch_nb_wald_t.R)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    from evaluate_sim_replicates import build
    if a.dump:
        out = Path(a.dump)
        out.mkdir(parents=True, exist_ok=True)
        for sc in ("none", "donor"):
            for rep in range(1, a.reps + 1):
                A = build(OUT_DATA / f"{sc}_rep{rep:02d}")
                donor = A.obs["SourceFile"].astype(str).to_numpy()
                ds = sorted(pd.unique(donor))
                pb = pd.DataFrame(np.vstack([np.asarray(A.layers["counts"][donor == d].sum(axis=0)).ravel()
                                             for d in ds]).astype(int), index=ds, columns=A.var_names)
                pb = pb.loc[:, pb.sum(axis=0) >= 10]
                pb.T.to_csv(out / f"counts_{sc}_{rep:02d}.csv")
                pd.DataFrame({"condition": [d.split(".")[-1] for d in ds]}, index=ds).to_csv(
                    out / f"coldata_{sc}_{rep:02d}.csv")
        return
    from duet.pseudobulk import run_pseudobulk
    rscript = os.environ.get("RSCRIPT")
    rows = []
    for sc in SCENARIOS:
        for rep in range(1, a.reps + 1):
            src = OUT_DATA / f"{sc}_rep{rep:02d}"
            A = build(src)
            A.obs["condition"] = A.obs["Sample"].astype(str)
            pb = run_pseudobulk(A, donor_col="SourceFile", condition_col="condition", ref_label="A",
                                test_label="B", counts_layer="counts", min_cells_per_donor=10,
                                min_donors_per_group=2, engine="R", rscript=rscript)
            pb = pb[pb.pb_tested & pb.pb_pvalue.notna()]
            n_d = int((pb.pb_padj < ALPHA).sum())
            rows.append(dict(scenario=sc, replicate=rep, method="DESeq2-pseudobulk (R)", n_tested=len(pb),
                             n_discoveries=n_d, rejection_pct=100 * n_d / max(len(pb), 1),
                             raw_type1=float((pb.pb_pvalue < ALPHA).mean()), any_discovery=n_d > 0))
            print(f"{sc} rep{rep:02d}: tested={len(pb)} discoveries={n_d}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT_RES / "per_rep_rdeseq2.csv", index=False)
    S = D.groupby(["scenario", "method"], sort=False).agg(
        n_reps=("replicate", "nunique"), rejection_pct_mean=("rejection_pct", "mean"),
        rejection_pct_max=("rejection_pct", "max"), raw_type1_mean=("raw_type1", "mean"),
        share_reps_with_any_discovery=("any_discovery", "mean"),
        total_discoveries=("n_discoveries", "sum")).reset_index()
    S.to_csv(OUT_RES / "summary_rdeseq2.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
