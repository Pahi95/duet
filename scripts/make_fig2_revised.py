#!/usr/bin/env python
"""
make_fig2_revised.py -- Figure 2 from the revision's results only (MO tasks 6-8, 11).

  (a) ROC against muscat ground truth, averaged over the replicates: each replicate's
      curve is interpolated onto a common false-positive-rate grid and the true-
      positive rates averaged. Truth: de/dp/dm/db alternative, ee and ep null.
  (b) Sensitivity against observed FDR at BH-adjusted p < 0.05, one point per
      replicate, with the mean +/- SD; includes the combined DUET + pseudobulk rule.
  (c) Percentage called per simulated category (mean +/- SD over replicates); the
      two null categories are on the right, where every call is a false positive.
  (d) Donor-swap global null: rejection proportion per unique donor partition
      (not an FDR: every discovery is false), by dataset and cell type.

Inputs: results/sim_reps100/{calls,per_rep_metrics.csv,by_category.csv},
        results/null/donor_swap/*.csv
Outputs: evidence/figures/fig2_revised.png (+ .pdf) and fig2_revised_<a..d>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import matplotlib                                          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
from sklearn.metrics import roc_curve                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
from sim_score import FILES, truth_table                   # noqa: E402

SIM = HERE / "results" / "sim_reps100"
FIG = HERE / "evidence" / "figures"
COLORS = {"DUET": "#1f77b4", "MAST": "#9467bd", "wilcoxon": "#ff7f0e", "DESeq2": "#2ca02c",
          "diffxpy": "#8c564b", "DUET+PB": "#d62728", "DESeq2-pseudobulk": "#2ca02c"}
CATS = [("de", "mean shift"), ("dp", "diff. proportion"), ("dm", "diff. modality"), ("db", "bimodal / both"),
        ("ep", "bimodal, no change"), ("ee", "no change")]


def mean_roc(grid=np.linspace(0, 1, 201)):
    tprs, aucs = {m: [] for m in FILES}, {m: [] for m in FILES}
    for rep in sorted((SIM / "calls").glob("sim_rep*")):
        if not all((rep / f"{f}.csv").is_file() for f in FILES.values()):
            continue
        truth = truth_table(rep.name)
        res = {m: pd.read_csv(rep / f"{f}.csv", low_memory=False).drop_duplicates("gene").set_index("gene")
               for m, f in FILES.items()}
        common = sorted(set.intersection(*(set(d.index[d["tested"].astype(bool) & d["pvalue"].notna()])
                                            for d in res.values())) & set(truth.index))
        y = truth.loc[common, "is_alt"].to_numpy()
        for m, d in res.items():
            s = np.nan_to_num(d.reindex(common)["score"].to_numpy(float), nan=0.0, posinf=1e308)
            fpr, tpr, _ = roc_curve(y, s)
            tprs[m].append(np.interp(grid, fpr, tpr))
            aucs[m].append(np.trapezoid(tpr, fpr))
    return grid, {m: np.mean(v, axis=0) for m, v in tprs.items() if v}, aucs


def panel_a(ax, roc):
    grid, tpr, aucs = roc
    for m, t in tpr.items():
        ax.plot(grid, t, color=COLORS[m], lw=1.4,
                label=f"{m} ({np.mean(aucs[m]):.3f} ± {np.std(aucs[m], ddof=1):.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.6, ls=":")
    n = len(next(iter(aucs.values())))
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate",
           title=f"(a) mean ROC over {n} replicates (AUC mean ± SD)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def panel_b(ax, M):
    for m, g in M.groupby("method", sort=False):
        c = COLORS.get(m, "0.4")
        ax.scatter(g.observed_fdr, g.sensitivity, s=8, color=c, alpha=0.35, lw=0)
        ax.errorbar(g.observed_fdr.mean(), g.sensitivity.mean(), xerr=g.observed_fdr.std(),
                    yerr=g.sensitivity.std(), color=c, marker="o", ms=5, capsize=2, lw=1.2, label=m)
    ax.axvline(0.05, color="0.5", lw=0.6, ls="--")
    ax.set(xlabel="observed FDR at BH-adjusted p < 0.05", ylabel="sensitivity", xlim=(-0.02, 1.0),
           title="(b) sensitivity vs observed FDR, per replicate")
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def panel_c(ax, C):
    methods = [m for m in COLORS if m in set(C.method)]
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        g = C[C.method == m].set_index("category").reindex([c for c, _ in CATS])
        x = np.arange(len(CATS)) + (i - (len(methods) - 1) / 2) * w
        ax.bar(x, g.pct_called_mean, w, yerr=g.pct_called_sd, color=COLORS[m], label=m,
               error_kw=dict(lw=0.6, capsize=1))
    ax.axvline(3.5, color="0.3", lw=0.8)
    ax.text(3.6, 113, "true nulls: every call is false", fontsize=7, va="top")
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([lab for _, lab in CATS], rotation=30, ha="right", fontsize=7)
    ax.set(ylabel="% called (BH-adjusted p < 0.05)", ylim=(0, 105), title="(c) calls by simulated category")
    ax.set_ylim(0, 125)                       # headroom so the legend clears the bars
    ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=6, frameon=False, ncol=6, loc="upper center")


def panel_d(ax, D):
    D = D[D.method.isin(["DUET", "wilcoxon", "DESeq2-pseudobulk", "DUET+PB"])]
    groups = list(D.groupby(["dataset", "celltype"], sort=False).groups)
    meths = ["DUET", "wilcoxon", "DESeq2-pseudobulk", "DUET+PB"]
    w = 0.8 / len(meths)
    rng = np.random.default_rng(0)
    for i, m in enumerate(meths):
        for j, key in enumerate(groups):
            v = D[(D.dataset == key[0]) & (D.celltype == key[1]) & (D.method == m)]["rejection_pct"].to_numpy()
            x = j + (i - (len(meths) - 1) / 2) * w + rng.uniform(-w / 4, w / 4, len(v))
            ax.scatter(x, v, s=6, color=COLORS[m], alpha=0.6, lw=0, label=m if j == 0 else None)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{d}\n{c}" for d, c in groups], rotation=45, ha="right", fontsize=6)
    ax.set(ylabel="% of genes rejected (BH < 0.05)", title="(d) donor-swap null, one point per unique partition")
    ax.legend(fontsize=7, frameon=False, loc="upper left", markerscale=2)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(SIM / "per_rep_metrics.csv")
    C = pd.read_csv(SIM / "by_category.csv")
    D = pd.concat([pd.read_csv(f) for f in sorted((HERE / "results" / "null" / "donor_swap").glob("*.csv"))])
    roc = mean_roc()
    draws = [(panel_a, roc), (panel_b, M), (panel_c, C), (panel_d, D)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (fn, data) in zip(axes.ravel(), draws):
        fn(ax, data)
    fig.savefig(FIG / "fig2_revised.png", dpi=200)
    fig.savefig(FIG / "fig2_revised.pdf")
    plt.close(fig)
    for (fn, data), key in zip(draws, "abcd"):                     # one plot per image
        f, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        fn(ax, data)
        f.savefig(FIG / f"fig2_revised_{key}.png", dpi=200)
        plt.close(f)
    print(f"[fig2] wrote {FIG / 'fig2_revised.png'} and 4 single panels")


if __name__ == "__main__":
    main()
