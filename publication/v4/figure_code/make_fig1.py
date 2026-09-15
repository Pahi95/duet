#!/usr/bin/env python
"""
make_fig1.py — Figure 1 for the DUET manuscript.

(a) is a walkthrough of ONE real gene rather than a generic box diagram, because
    the point of a hurdle model is *why* two components are needed: the condition
    shifts both the fraction of cells expressing the gene and the level among
    those that do. A schematic cannot show that; real data can.
(e) and (f) come from the matched benchmark (results/bench/fair.csv, results/final/table_T1.csv).

Default exemplar: IFIT3 in Kang CD14+ Monocytes — the top DUET hit, and an
interferon-stimulated gene, so the biology is unambiguous.
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figutil import maybe_split   # noqa: E402

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
REF_C, TEST_C = "#5c6675", "#b64b34"
DET_C, EXP_C = "#2b6685", "#b64b34"


def panel_a(axes, gene, adata, ref, test, refname, testname):
    """Raw data -> detection component -> continuous component -> summed statistic."""
    x = adata[:, gene].X
    x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
    cond = adata.obs["Sample"].astype(str).to_numpy()
    a, b = x[cond == ref], x[cond == test]

    # --- a1: the data, zero spike separated from the positive part ---
    ax = axes[0]
    for v, c, lab in ((a, REF_C, refname), (b, TEST_C, testname)):
        pos = v[v > 0]
        ax.hist(pos, bins=28, alpha=.55, color=c, label=f"{lab} (detected)",
                density=True, zorder=3)
    ax.set_xlabel("log-normalised expression\n(detected cells only)", fontsize=8)
    ax.set_ylabel("density", fontsize=8.5)
    ax.set_title(f"(a) one gene: {gene}\n{len(a)+len(b):,} cells, two conditions",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.text(.03, .97, f"zeros: {100*(a==0).mean():.0f}% vs {100*(b==0).mean():.0f}%",
            transform=ax.transAxes, fontsize=7.5, va="top", color="0.3")

    # --- a2: detection component ---
    ax = axes[1]
    dr = [100 * (a > 0).mean(), 100 * (b > 0).mean()]
    bars = ax.bar([0, 1], dr, .55, color=[REF_C, TEST_C], zorder=3)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8.5, padding=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels([refname, testname], fontsize=8.5)
    ax.set_ylabel("% of cells detecting", fontsize=8.5)
    ax.set_ylim(0, max(dr) * 1.35)
    ax.set_title("(b) detection component\nlogistic LRT", fontsize=9.5, loc="left",
                 color=DET_C)
    ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)

    # --- a3: continuous component ---
    ax = axes[2]
    parts = ax.violinplot([a[a > 0], b[b > 0]], positions=[0, 1], widths=.7,
                          showmeans=True, showextrema=False)
    for pc, c in zip(parts["bodies"], (REF_C, TEST_C)):
        pc.set_facecolor(c); pc.set_alpha(.6)
    parts["cmeans"].set_color("0.2"); parts["cmeans"].set_linewidth(1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels([refname, testname], fontsize=8.5)
    ax.set_ylabel("expression | detected", fontsize=8.5)
    ax.set_title("(c) expression component\nGaussian LRT", fontsize=9.5, loc="left",
                 color=EXP_C)
    ax.grid(axis="y", alpha=.25, lw=.5)
    return dr


def panel_a_sum(ax, row):
    """Schematic of the moderated hurdle; no model fitting for the figure."""
    ax.axis("off")
    ax.set_title("(d) combined hurdle statistic", loc="left", fontsize=9.5)
    ax.text(.03, .85, "Detection: likelihood-ratio statistic\n\nExpression: moderated t tail\nconverted to a χ²₁ statistic\n\nSum → χ² with active-component df\n\nRank by −log₁₀ p (descending)", va="top", fontsize=10, linespacing=1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(INPUTS / "data" / "kang_8donors.h5ad"))
    ap.add_argument("--results", default=str(HERE / "figure_data"))
    ap.add_argument("--celltype", default="CD14+ Monocytes")
    ap.add_argument("--gene", default="IFIT3")
    ap.add_argument("--ref", default="ctrl")
    ap.add_argument("--test", default="stim")
    ap.add_argument("--out", default=str(HERE / "figures" / "fig1_model_and_performance.png"))
    ap.add_argument("--split", action="store_true",
                    help="also write each panel as its own image")
    a = ap.parse_args()

    if a.gene != "IFIT3" or a.celltype != "CD14+ Monocytes":
        raise SystemExit("The archived example is IFIT3 in Kang CD14+ Monocytes.")
    fixture = pd.read_csv(HERE / "figure_data" / "ifit3_expression.csv", float_precision="round_trip")
    A = sc.AnnData(fixture.expression.to_numpy().reshape(-1, 1),
                   obs=pd.DataFrame({"Sample": fixture.condition.to_numpy(), "celltype": a.celltype}, index=[str(i) for i in range(len(fixture))]),
                   var=pd.DataFrame(index=[a.gene]))
    A = A[A.obs["celltype"].astype(str) == a.celltype]
    d = pd.read_csv(Path(a.results) / "de_DUET_all_celltypes.csv", low_memory=False)
    d = d[(d.tested == True) & (d.celltype == a.celltype)]          # noqa: E712
    row = d[d.gene == a.gene]
    if not len(row):
        raise SystemExit(f"{a.gene} not in results")
    row = row.iloc[0]

    fig = plt.figure(figsize=(14.2, 7.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=.52, wspace=.34)
    ax_a = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_d = fig.add_subplot(gs[0, 3])
    panel_a(ax_a, a.gene, A, a.ref, a.test, a.ref, a.test)
    panel_a_sum(ax_d, row)

    # ---- bottom row: performance + agreement, from the matched benchmark (bench_harness.py) ----
    # Both engines receive the same cells, genes, gene order and model; 5 timed runs per point after a
    # warm-up, single-threaded. The machine was in use, so CPU time is the timing that is reported.
    B = pd.read_csv(HERE / "figure_data" / "fair.csv")
    B = B[~B.warmup.astype(bool)]
    tot = B[B.stage == "TOTAL_RUN"].groupby(["input", "engine"])["cpu_s"].median().unstack()
    comp = (B[B.stage.isin(["fit_test", "zlm", "lrtest"])].groupby(["input", "engine", "run"])["cpu_s"].sum()
            .groupby(["input", "engine"]).median().unstack())
    T1 = pd.read_csv(HERE / "figure_data" / "agreement_raw.csv")
    ladder = T1[T1.dataset.str.contains("benchmark ladder")].set_index("celltype")
    tot["cells"] = ladder.loc[tot.index, "cells"].to_numpy()
    comp["cells"] = tot["cells"]

    ax = fig.add_subplot(gs[1, :2])
    for series, mk in (("ductal", "o"), ("pooled", "s")):
        t = tot[tot.index.str.startswith(series)].sort_values("cells")
        c = comp.loc[t.index]
        mfc = None if series == "ductal" else "white"
        ax.plot(t.cells, t.mast, mk + "-", c="#2b6685", lw=2, ms=6, mfc=mfc, label=f"R MAST, end-to-end ({series})")
        ax.plot(c.cells, c.mast, mk + "--", c="#2b6685", lw=1.2, ms=5, mfc=mfc, alpha=.6,
                label=f"R MAST, zlm + lrTest ({series})")
        ax.plot(t.cells, t.duet, mk + "-", c="#b64b34", lw=2, ms=6, mfc=mfc, label=f"DUET, end-to-end ({series})")
        for cells, r in zip(t.cells, (t.mast / t.duet)):
            ax.annotate(f"{r:.0f}×", (cells, float(t.loc[t.cells == cells, "duet"].iloc[0])),
                        textcoords="offset points", xytext=(0, 9 if series == "ductal" else -14), ha="center",
                        fontsize=7, color="#b64b34", fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cells", fontsize=8.5); ax.set_ylabel("CPU time (s), median of 5 runs", fontsize=8.5)
    ax.set_title("(e) runtime on identical input; annotated: end-to-end CPU-time ratio\n"
                 "     filled: pancreas Ductal; open: three cell types pooled", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="center right", ncol=1)
    ax.grid(alpha=.25, lw=.5, which="both")

    ax = fig.add_subplot(gs[1, 2:])
    lad = ladder.sort_values("cells")
    for series, mk in (("ductal", "o"), ("pooled", "s")):
        l = lad[lad.index.str.startswith(series)]
        mfc = None if series == "ductal" else "white"
        ax.plot(l.cells, l.nlp_spearman, mk + "-", c="#2b6685", lw=1.6, ms=5, mfc=mfc,
                label=f"$-\\log_{{10}}p$ Spearman ({series})")
        ax.plot(l.cells, l.sig_jaccard, mk + "--", c="0.45", lw=1.4, ms=5, mfc=mfc,
                label=f"significant-gene Jaccard ({series})")
        ax.plot(l.cells, l.log2fc_r, mk + ":", c="#b64b34", lw=1.6, ms=5, mfc=mfc, label=f"log2FC Pearson ({series})")
    ax.set_xscale("log"); ax.set_ylim(.995, 1.0005)
    ax.set_xlabel("cells", fontsize=8.5)
    ax.set_ylabel("agreement with R MAST", fontsize=8.5)
    ax.set_title("(f) agreement on the benchmark inputs (same cells, genes and design)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.grid(alpha=.25, lw=.5)

    for ax_ in fig.get_axes():
        ax_.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")
    maybe_split(fig, a, Path(a.out).parent / "panels", "fig1")



if __name__ == "__main__":
    main()
