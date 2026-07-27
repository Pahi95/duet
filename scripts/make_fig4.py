#!/usr/bin/env python
"""
make_fig4.py — null calibration across datasets, methods and null designs.

Two problems with the first version, both of which hid real results:

(a) plotted the percentage of genes surviving FDR correction under the
    permutation null. That is 0.00 for every method in all nine dataset x
    cell-type combinations -- a true and important result, but drawn as 27 bars
    of height zero it conveys nothing. The panel now plots the quantity that IS
    informative here, the raw type-I error at alpha = 0.05, against its nominal
    value; the FDR result is stated once in the panel title instead.

(b) plotted 0-100% on a linear axis, on which pseudobulk's 0.00-0.85% range is
    indistinguishable from zero -- so the control that passes looked like no data
    at all. The axis is now symlog: linear below 1%, logarithmic above, so both
    "0.85 vs 0.08" and "100 vs 5" are readable on one axis.

All numbers are read from results/null/*.csv; nothing is hardcoded.
"""
from __future__ import annotations
import glob
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
OUT = HERE / "evidence" / "figures" / "fig4_null_calibration.png"

COL = {"DUET": "#b64b34", "wilcoxon": "#2b6685", "DESeq2-pseudobulk": "#3d7a55"}
MORDER = ["DUET", "wilcoxon", "DESeq2-pseudobulk"]
ORDER = ["pancreas", "kang", "crowell"]
NICE = {"pancreas": "Pancreas\n16 donors, 8v8\nseparate matrices",
        "kang": "Kang 2018\n8 donors, 4v4\nmultiplexed pool",
        "crowell": "Crowell 4v4\n4 mice, 2v2\nseparate preps"}


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(HERE / "results" / "null" / "null_*_matched.csv")))
    if not files:
        raise SystemExit("no results/null/null_*_matched.csv — run null_calibration.py first")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["celltype"] = (d["celltype"].str.replace(" cells", "", regex=False)
                                  .str.replace(" cell", "", regex=False))
    return d


def grouped_bars(ax, d, value_col):
    """One cluster per cell type, three bars per cluster, datasets separated."""
    xs, labels, bounds = [], [], []
    pos = 0.0
    for ds in ORDER:
        sub = d[d.dataset == ds]
        cts = list(dict.fromkeys(sub.celltype))
        start = pos
        for ct in cts:
            for k, m in enumerate(MORDER):
                r = sub[(sub.celltype == ct) & (sub.method == m)]
                if not len(r):
                    continue
                ax.bar(pos + (k - 1) * 0.27, float(r[value_col].iloc[0]), 0.27,
                       color=COL[m], zorder=3,
                       label=m if (ds == ORDER[0] and ct == cts[0]) else None)
            xs.append(pos)
            labels.append(ct)
            pos += 1.0
        bounds.append((start, pos - 1.0, ds))
        pos += 0.7
    for _, end, _ in bounds[:-1]:
        ax.axvline(end + 0.85, color="0.85", lw=1, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    return bounds


def main():
    D = load()
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.4))
    fig.patch.set_facecolor("white")

    # ---- (a) permutation null: type-I error, the quantity that varies --------
    ax = axes[0]
    perm = D[D.design == "permutation"]
    bounds = grouped_bars(ax, perm, "type1_at_alpha05")
    ax.axhline(0.05, ls="--", lw=1.2, c="0.35", zorder=4)
    ax.text(0.985, 0.05, "nominal α = 0.05 ", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="0.35", va="bottom", ha="right")
    ax.set_ylabel("type-I error at α = 0.05\n(uncorrected, matched genes)", fontsize=9)

    n_comb = perm.groupby(["dataset", "celltype"]).ngroups
    worst = perm.pct_sig_fdr05.max()
    ax.set_title(
        "(a) permutation null — shuffle labels within samples\n"
        f"after BH correction 0 genes survive in {n_comb}/{n_comb} combinations "
        f"for every method\n(largest single value {worst:.2f}%), so the uncorrected "
        "rate is what carries information",
        fontsize=10.5, loc="left", linespacing=1.6)
    ax.set_ylim(0, float(perm.type1_at_alpha05.max()) * 1.28)

    # ---- (b) donor-swap null: symlog so the passing control is visible -------
    ax = axes[1]
    swap = D[D.design == "donor_swap"]
    grouped_bars(ax, swap, "pct_sig_fdr05")
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.9)
    ax.set_ylim(0, 260)
    ax.set_yticks([0, 0.25, 0.5, 1, 5, 10, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 or v == 0 else f"{v:.2f}"))
    ax.axhspan(0, 1.0, color="#f2efe9", zorder=0)
    ax.axhline(5, ls="--", lw=1.2, c="0.35", zorder=4)
    ax.text(0.985, 5, "5% ", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="0.35", va="bottom", ha="right")
    ax.set_ylabel("% of tested genes called significant\n(FDR < 0.05, matched genes)",
                  fontsize=9)

    pb = swap[swap.method == "DESeq2-pseudobulk"].pct_sig_fdr05
    ax.set_title(
        "(b) donor-swap null — split the reference samples at random\n"
        "symlog axis: linear below 1% (shaded band), logarithmic above, because\n"
        f"pseudobulk's 0.00–{pb.max():.2f}% range is otherwise indistinguishable "
        "from an empty panel",
        fontsize=10.5, loc="left", linespacing=1.6)

    # value labels — rotated in (b), where neighbouring bars are close together
    for a, rot in ((axes[0], 0), (axes[1], 90)):
        for p in a.patches:
            v = p.get_height()
            a.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}",
                       (p.get_x() + p.get_width() / 2, v),
                       textcoords="offset points", xytext=(0, 3), ha="center",
                       va="bottom", fontsize=6, color="0.3", rotation=rot, zorder=5)

    for a in axes:
        a.grid(axis="y", alpha=.25, lw=.5, zorder=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=8)
        for s_, e_, ds in bounds:
            a.text((s_ + e_) / 2, -0.30, NICE[ds], transform=a.get_xaxis_transform(),
                   ha="center", va="top", fontsize=7.5, color="0.35")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 1.055),
               ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Null calibration — no condition effect anywhere, so the correct "
                 "answer is ~0. Pseudobulk is the control that passes.",
                 fontsize=12.5, y=1.055, x=0.012, ha="left")
    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.24)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
