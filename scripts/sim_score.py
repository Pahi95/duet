#!/usr/bin/env python
"""
sim_score.py -- score the muscat replicates against their ground truth
. Reads the per-gene files written by sim_calls.py.

Truth, fixed before scoring (decision A4):
  primary    alternative = de, dp, dm, db; null = ee AND ep. In muscat:::.sim an `ep`
             gene receives the same two-component mixture in both groups, so its
             distribution is identical between them -- a null for every estimand.
  secondary  from the simulation parameters, never from the realised data (which
             carry the donor noise the calibration question is about):
             mean_changed  = |log2(sim_mean.B / sim_mean.A)| > 0.1
             shape_changed = category in {dp, dm, db} (the two groups get different
                             mixtures); label = none / mean only / shape only / mean + shape.

Per replicate and method, at BH-adjusted p < 0.05: TP, FP, FN, TN, sensitivity,
observed FDR; ROC-AUC and PR-AUC from the underflow-safe score. The combined
DUET + pseudobulk rule (decision A3) is scored as its own method, DUET+PB:
significant = DESeq2 padj < 0.05 and sign(DESeq2 log2FC) = sign(DUET log2FC).

Comparison at equal actual FDR (A8b): on the calibration replicates each method's
score threshold is set so that its mean observed FDR is 0.05; sensitivity and FDR
are then measured on the separate evaluation replicates. Genes are scored on the
set every method returned a result for; missing_genes.csv says which genes each
method lacks and why (the 3989-of-4000 question).

Outputs (results/sim_reps100/):
  manifest.csv, per_rep_metrics.csv, summary_metrics.csv, by_category.csv,
  by_secondary_label.csv, fdr_matched.csv, fdr_curves.csv, missing_genes.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent.parent            # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
ALT = {"de", "dp", "dm", "db"}
NULL = {"ee", "ep"}
SHAPE = {"dp", "dm", "db"}
ALPHA = 0.05
METHODS = ["DUET", "MAST", "wilcoxon", "DESeq2", "diffxpy"]
FILES = {"DUET": "duet", "MAST": "mast", "wilcoxon": "wilcoxon", "DESeq2": "deseq2", "diffxpy": "diffxpy"}
CATEGORY_LABEL = {"de": "mean shift", "dp": "diff. proportion", "dm": "diff. modality",
                  "db": "bimodal / both", "ep": "bimodal, no change", "ee": "no change"}


def truth_table(rep: str) -> pd.DataFrame:
    gi = pd.read_csv(INPUTS / "data" / rep / "gene_info.tsv", sep="\t")
    gi = gi.drop_duplicates("gene").set_index("gene")
    lr = np.log2(gi["sim_mean.B"].clip(lower=1e-12) / gi["sim_mean.A"].clip(lower=1e-12))
    t = pd.DataFrame({"category": gi["category"].astype(str), "sim_log2_mean_ratio": lr})
    t["is_alt"] = t["category"].isin(ALT)
    t["mean_changed"] = lr.abs() > 0.1
    t["shape_changed"] = t["category"].isin(SHAPE)
    t["secondary"] = np.select(
        [t.mean_changed & t.shape_changed, t.mean_changed, t.shape_changed],
        ["mean + shape", "mean only", "shape only"], default="none")
    return t


def load_rep(calls: Path, rep: str) -> dict[str, pd.DataFrame] | None:
    out = {}
    for m, f in FILES.items():
        p = calls / rep / f"{f}.csv"
        if not p.is_file():
            return None
        d = pd.read_csv(p, low_memory=False)
        d["gene"] = d["gene"].astype(str)
        out[m] = d.drop_duplicates("gene").set_index("gene")
    return out


def combined_rule(res: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d, p = res["DUET"], res["DESeq2"]
    genes = d.index.union(p.index)
    d, p = d.reindex(genes), p.reindex(genes)
    pb_sig = p["fdr"].notna() & (p["fdr"] < ALPHA)
    same_dir = np.sign(p["log2fc"]) == np.sign(d["log2fc"])
    sig = pb_sig & same_dir
    duet_sig = d["fdr"].notna() & (d["fdr"] < ALPHA)
    call = np.where(sig, "significant", np.where(pb_sig & ~same_dir, "direction_conflict",
                    np.where(duet_sig, "ranked_only", "ns")))
    # the combined output ranks by the hurdle; significance comes from the rule
    return pd.DataFrame({"tested": d["tested"].fillna(False).astype(bool) & p["pvalue"].notna(),
                         "fdr": np.where(sig, 0.0, 1.0), "score": d["score"], "call": call},
                        index=genes)


def rep_metrics(res, truth, common) -> list[dict]:
    t = truth.loc[common]
    y = t["is_alt"].to_numpy()
    rows = []
    for m, d in res.items():
        d = d.reindex(common)
        called = (d["fdr"].to_numpy(float) < ALPHA)
        called = np.nan_to_num(called, nan=False).astype(bool)
        s = np.nan_to_num(d["score"].to_numpy(float), nan=0.0, posinf=1e308)
        tp, fp = int((called & y).sum()), int((called & ~y).sum())
        fn, tn = int((~called & y).sum()), int((~called & ~y).sum())
        row = dict(method=m, n_scored=len(y), n_alt=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                   sensitivity=tp / max(tp + fn, 1), observed_fdr=fp / max(tp + fp, 1),
                   n_called=tp + fp, roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
        if m == "DUET+PB":
            row["n_direction_conflict"] = int((d["call"] == "direction_conflict").sum())
            row["n_ranked_only"] = int((d["call"] == "ranked_only").sum())
        for cat in CATEGORY_LABEL:
            mk = (t["category"] == cat).to_numpy()
            row[f"pct_called_{cat}"] = 100 * called[mk].mean() if mk.any() else np.nan
            row[f"n_{cat}"] = int(mk.sum())
        for lab in ("none", "mean only", "shape only", "mean + shape"):
            mk = (t["secondary"] == lab).to_numpy()
            row[f"pct_called_sec_{lab}"] = 100 * called[mk].mean() if mk.any() else np.nan
            row[f"n_sec_{lab}"] = int(mk.sum())
        rows.append(row)
    return rows


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fdr_at(scores, y, thr):
    called = scores >= thr
    tp, fp = int((called & y).sum()), int((called & ~y).sum())
    return fp / max(tp + fp, 1), tp / max(int(y.sum()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default=str(HERE / "results" / "sim_reps100" / "calls"))
    ap.add_argument("--outdir", default=str(HERE / "results" / "sim_reps100"))
    ap.add_argument("--calibration", default="1-50")
    ap.add_argument("--evaluation", default="51-100")
    a = ap.parse_args()
    calls, out = Path(a.calls), Path(a.outdir)
    rng_of = lambda s: set(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))  # noqa: E731
    cal, ev = rng_of(a.calibration), rng_of(a.evaluation)

    manifest, rows, missing, scored, rule_agree = [], [], [], {}, []
    for repdir in sorted(calls.glob("sim_rep*")):
        rep = repdir.name
        res = load_rep(calls, rep)
        if res is None:
            continue
        truth = truth_table(rep)
        seed = (INPUTS / "data" / rep / "seed.txt").read_text().strip()
        manifest.append(dict(replicate=rep, seed=int(seed), **{f"n_{k}": int((truth.category == k).sum())
                                                                for k in CATEGORY_LABEL}))
        have = {m: set(d.index[d["tested"].astype(bool) & d["pvalue"].notna()]) for m, d in res.items()}
        common = sorted(set.intersection(*have.values()) & set(truth.index))
        for m, d in res.items():
            for g in sorted(set(truth.index) - have[m]):
                why = (d.loc[g, "skip_reason"] if g in d.index and "skip_reason" in d and
                       isinstance(d.loc[g, "skip_reason"], str) and d.loc[g, "skip_reason"] else
                       ("p-value NA (independent filtering / Cook)" if g in d.index else "not returned (method filter)"))
                missing.append(dict(replicate=rep, method=m, gene=g, category=truth.loc[g, "category"], reason=why))
        own = calls / rep / "duet_pb.csv"
        rule = combined_rule(res)                    # the rule recomputed here from the DUET and DESeq2 files
        if own.is_file():                            # DUET's own arm (duet.pseudobulk.combine_calls)
            o = pd.read_csv(own, low_memory=False).drop_duplicates("gene").set_index("gene")
            o.index = o.index.astype(str)
            sig = o["call"].eq("significant")
            res["DUET+PB"] = pd.DataFrame({"tested": o["tested"].astype(bool), "fdr": np.where(sig, 0.0, 1.0),
                                           "score": o["score"], "call": o["call"]}, index=o.index)
            g = rule.index.intersection(o.index)
            rule_agree.append(dict(replicate=rep, genes=len(g),
                                   differing_calls=int((rule.loc[g, "call"].eq("significant")
                                                        != o.loc[g, "call"].eq("significant")).sum())))
        else:
            res["DUET+PB"] = rule
        for r in rep_metrics(res, truth, common):
            r["replicate"] = rep
            r["rep_no"] = int(rep.replace("sim_rep", ""))
            rows.append(r)
        scored[rep] = {m: (res[m].reindex(common)["score"].to_numpy(float), truth.loc[common, "is_alt"].to_numpy())
                       for m in METHODS}
    if not rows:
        raise SystemExit("no replicate has all five methods yet")
    out.mkdir(parents=True, exist_ok=True)
    M = pd.DataFrame(rows)
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    M.to_csv(out / "per_rep_metrics.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "missing_genes.csv", index=False)
    if rule_agree:
        RA = pd.DataFrame(rule_agree)
        RA.to_csv(out / "duet_pb_vs_recomputed_rule.csv", index=False)
        print(f"[score] DUET's own pseudobulk arm vs the recomputed rule: {int(RA.differing_calls.sum())} differing "
              f"calls over {len(RA)} replicates", flush=True)

    # ---- summary with bootstrap CIs over replicates -----------------------------------
    summ = []
    for m, g in M.groupby("method", sort=False):
        r = dict(method=m, n_replicates=len(g))
        for k in ("roc_auc", "pr_auc", "sensitivity", "observed_fdr", "n_called"):
            lo, hi = boot_ci(g[k])
            r.update({f"{k}_mean": g[k].mean(), f"{k}_sd": g[k].std(), f"{k}_ci_lo": lo, f"{k}_ci_hi": hi,
                      f"{k}_min": g[k].min(), f"{k}_max": g[k].max()})
        r["n_reps_fdr_le_0.10"] = int((g["observed_fdr"] <= 0.10).sum())
        summ.append(r)
    pd.DataFrame(summ).to_csv(out / "summary_metrics.csv", index=False)

    cat_rows = []
    for m, g in M.groupby("method", sort=False):
        for cat, lab in CATEGORY_LABEL.items():
            cat_rows.append(dict(method=m, category=cat, label=lab, n_mean=g[f"n_{cat}"].mean(),
                                 pct_called_mean=g[f"pct_called_{cat}"].mean(),
                                 pct_called_sd=g[f"pct_called_{cat}"].std(), truth="alternative" if cat in ALT else "null"))
    pd.DataFrame(cat_rows).to_csv(out / "by_category.csv", index=False)
    sec_rows = []
    for m, g in M.groupby("method", sort=False):
        for lab in ("none", "mean only", "shape only", "mean + shape"):
            sec_rows.append(dict(method=m, secondary=lab, n_mean=g[f"n_sec_{lab}"].mean(),
                                 pct_called_mean=g[f"pct_called_sec_{lab}"].mean(),
                                 pct_called_sd=g[f"pct_called_sec_{lab}"].std()))
    pd.DataFrame(sec_rows).to_csv(out / "by_secondary_label.csv", index=False)

    # ---- equal actual FDR: calibrate on one set of replicates, evaluate on another -----
    reps_cal = [r for r in scored if int(r.replace("sim_rep", "")) in cal]
    reps_ev = [r for r in scored if int(r.replace("sim_rep", "")) in ev]
    fm, curves = [], []
    if reps_cal and reps_ev:
        for m in METHODS:
            pool = np.concatenate([scored[r][m][0] for r in reps_cal])
            grid = np.unique(np.quantile(pool[np.isfinite(pool)], np.linspace(0.5, 0.99999, 800)))
            mean_fdr = np.array([np.mean([fdr_at(np.nan_to_num(scored[r][m][0]), scored[r][m][1], th)[0]
                                          for r in reps_cal]) for th in grid])
            ok = np.where(mean_fdr <= 0.05)[0]
            thr = float(grid[ok.min()]) if ok.size else np.nan
            ev_vals = [fdr_at(np.nan_to_num(scored[r][m][0]), scored[r][m][1], thr) for r in reps_ev] \
                if np.isfinite(thr) else []
            f_ev = [v[0] for v in ev_vals]
            s_ev = [v[1] for v in ev_vals]
            fm.append(dict(method=m, n_calibration=len(reps_cal), n_evaluation=len(reps_ev),
                           score_threshold=thr, calibration_mean_fdr=float(mean_fdr[ok.min()]) if ok.size else np.nan,
                           eval_fdr_mean=np.mean(f_ev) if f_ev else np.nan,
                           eval_fdr_ci=boot_ci(f_ev) if f_ev else (np.nan, np.nan),
                           eval_fdr_max=np.max(f_ev) if f_ev else np.nan,
                           eval_sensitivity_mean=np.mean(s_ev) if s_ev else np.nan,
                           eval_sensitivity_ci=boot_ci(s_ev) if s_ev else (np.nan, np.nan)))
            for th in grid[::8]:
                vals = [fdr_at(np.nan_to_num(scored[r][m][0]), scored[r][m][1], th) for r in reps_ev]
                curves.append(dict(method=m, score_threshold=th, fdr_mean=np.mean([v[0] for v in vals]),
                                   sensitivity_mean=np.mean([v[1] for v in vals])))
    pd.DataFrame(fm).to_csv(out / "fdr_matched.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "fdr_curves.csv", index=False)

    pd.set_option("display.width", 220)
    S = pd.DataFrame(summ).set_index("method")
    print(f"[score] {M.replicate.nunique()} replicates scored")
    print(S[["n_replicates", "roc_auc_mean", "pr_auc_mean", "sensitivity_mean", "observed_fdr_mean",
             "observed_fdr_min", "observed_fdr_max", "n_reps_fdr_le_0.10"]].round(4).to_string())
    if fm:
        print(pd.DataFrame(fm)[["method", "score_threshold", "eval_fdr_mean", "eval_sensitivity_mean"]].round(4).to_string())
    print(f"[score] missing (gene, method) pairs: {len(missing)}; by reason:")
    if missing:
        print(pd.DataFrame(missing).groupby(["method", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
