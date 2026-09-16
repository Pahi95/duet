#!/usr/bin/env python
"""
sim_score_pb_extra.py -- score the edgeR and limma-voom pseudobulk calls on the saved replicates.

The five-method evaluation (sim_score.py) is final and is not changed. The additional
sample-level methods are scored on exactly the same per-replicate gene universe (the genes all
five original methods tested), with the same metrics and the same calibration split
(thresholds on replicates 1-50, evaluation on 51-100). Genes removed by filterByExpr count as
not called (score 0). DESeq2 is rescored alongside as a check against per_rep_metrics.csv.

Outputs: results/sim_reps100/extra_pb/{per_rep_metrics,summary_metrics,by_category,fdr_matched}.csv
Usage:   python scripts/sim_score_pb_extra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_score as S                                      # noqa: E402

EXTRA = {"edgeR": "edger", "limma-voom": "voom"}


def main():
    calls = S.HERE / "results" / "sim_reps100" / "calls"
    out = S.HERE / "results" / "sim_reps100" / "extra_pb"
    rows, scored, filtered = [], {}, []
    for repdir in sorted(calls.glob("sim_rep*")):
        rep = repdir.name
        res = S.load_rep(calls, rep)
        if res is None or not all((repdir / f"{f}.csv").is_file() for f in EXTRA.values()):
            continue
        truth = S.truth_table(rep)
        have = {m: set(d.index[d["tested"].astype(bool) & d["pvalue"].notna()]) for m, d in res.items()}
        common = sorted(set.intersection(*have.values()) & set(truth.index))
        ext = {"DESeq2": res["DESeq2"]}
        for m, f in EXTRA.items():
            d = pd.read_csv(repdir / f"{f}.csv", low_memory=False)
            d["gene"] = d["gene"].astype(str)
            d = d.drop_duplicates("gene").set_index("gene")
            ok = d["tested"].astype(bool) & d["pvalue"].notna()
            filtered.append(dict(replicate=rep, method=m, n_common=len(common),
                                 n_common_filtered=int((~ok.reindex(common).fillna(False)).sum())))
            ext[m] = d
        for r in S.rep_metrics(ext, truth, common):
            r["replicate"], r["rep_no"] = rep, int(rep.replace("sim_rep", ""))
            rows.append(r)
        y = truth.loc[common, "is_alt"].to_numpy()
        scored[rep] = {m: (np.nan_to_num(d.reindex(common)["score"].to_numpy(float), nan=0.0), y)
                       for m, d in ext.items()}
    if not rows:
        raise SystemExit("no replicate has edgeR and voom calls yet")
    out.mkdir(parents=True, exist_ok=True)
    M = pd.DataFrame(rows)
    M.to_csv(out / "per_rep_metrics.csv", index=False)
    pd.DataFrame(filtered).to_csv(out / "filtered_in_common_universe.csv", index=False)

    ref = pd.read_csv(S.HERE / "results" / "sim_reps100" / "per_rep_metrics.csv")
    chk = ref[ref.method == "DESeq2"].set_index("replicate")[["sensitivity", "observed_fdr", "roc_auc"]]
    mine = M[M.method == "DESeq2"].set_index("replicate")[["sensitivity", "observed_fdr", "roc_auc"]]
    diff = float((chk.loc[mine.index] - mine).abs().to_numpy().max())
    print(f"[extra] DESeq2 rescored vs saved per_rep_metrics: max |difference| = {diff:.3g}", flush=True)
    assert diff < 1e-9, "common gene universe or metrics differ from sim_score.py"

    summ = []
    for m, g in M.groupby("method", sort=False):
        r = dict(method=m, n_replicates=len(g))
        for k in ("roc_auc", "pr_auc", "sensitivity", "observed_fdr", "n_called"):
            lo, hi = S.boot_ci(g[k])
            r.update({f"{k}_mean": g[k].mean(), f"{k}_ci_lo": lo, f"{k}_ci_hi": hi,
                      f"{k}_min": g[k].min(), f"{k}_max": g[k].max()})
        summ.append(r)
    pd.DataFrame(summ).to_csv(out / "summary_metrics.csv", index=False)
    cat_rows = [dict(method=m, category=c, label=lab, pct_called_mean=g[f"pct_called_{c}"].mean())
                for m, g in M.groupby("method", sort=False) for c, lab in S.CATEGORY_LABEL.items()]
    pd.DataFrame(cat_rows).to_csv(out / "by_category.csv", index=False)

    cal = [r for r in scored if int(r.replace("sim_rep", "")) <= 50]
    ev = [r for r in scored if int(r.replace("sim_rep", "")) > 50]
    fm = []
    for m in ext:
        pool = np.concatenate([scored[r][m][0] for r in cal])
        grid = np.unique(np.quantile(pool[np.isfinite(pool)], np.linspace(0.5, 0.99999, 800)))
        mean_fdr = np.array([np.mean([S.fdr_at(scored[r][m][0], scored[r][m][1], th)[0] for r in cal])
                             for th in grid])
        ok = np.where(mean_fdr <= 0.05)[0]
        thr = float(grid[ok.min()]) if ok.size else np.nan
        vals = [S.fdr_at(scored[r][m][0], scored[r][m][1], thr) for r in ev] if np.isfinite(thr) else []
        fm.append(dict(method=m, n_calibration=len(cal), n_evaluation=len(ev), score_threshold=thr,
                       eval_fdr_mean=np.mean([v[0] for v in vals]) if vals else np.nan,
                       eval_sensitivity_mean=np.mean([v[1] for v in vals]) if vals else np.nan,
                       eval_sensitivity_ci=S.boot_ci([v[1] for v in vals]) if vals else (np.nan, np.nan)))
    pd.DataFrame(fm).to_csv(out / "fdr_matched.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summ)[["method", "n_replicates", "sensitivity_mean", "observed_fdr_mean", "roc_auc_mean"]]
          .round(4).to_string(index=False))
    print(pd.DataFrame(fm).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
