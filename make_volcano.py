#!/usr/bin/env python
"""
make_volcano.py
===============
Volcano plots (x = log2FC, y = -log10 p; red = up in pancreas, blue = down,
grey = ns at fdr<0.05) for each DE method, from the per-method result CSVs.

Outputs (in results/):
  volcano_<celltype>.png    -- one figure PER CELL TYPE (methods side by side)
  volcano_grid.png          -- cell types (rows) x methods (cols) overview grid
  volcano_all_panel.png     -- all cell types pooled (methods side by side)

The log2FC x-axis is SHARED across panels, so MAST shows up as a narrow cloud
(its log-mean-diff log2FC is compressed).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["DUET", "MAST", "diffxpy", "DESeq2", "wilcoxon"]
HERE = Path(__file__).resolve().parent


def _draw(ax, d, *, fdr, xlim, ycap, log_y=False, title=None, ylabel=None, small_tag=False):
    """One volcano onto ax from a harmonized per-method DataFrame `d`.

    With `log_y` the y-axis is log-scaled and NOT capped: now that `neglog10p` is
    exact past the float64 p-value floor, the significance tail really does span
    ~4 orders of magnitude (up to -log10 p ~ 3260 here), which a linear axis can
    only render as a flat line at the cap.
    """
    if d is None or not len(d):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8)
        ax.set_xlim(-xlim, xlim)
        if title:
            ax.set_title(title, fontsize=9)
        return
    lfc = d["log2fc"].to_numpy(float)
    x = lfc.clip(-xlim, xlim)
    # Prefer the underflow-safe -log10(p). Ranking/plotting on the raw p-value
    # pins every gene whose p underflowed to 0.0 onto a single horizontal line
    # (the "flat top") -- ~7% of genes at 11k cells, spanning a true 10x range.
    if "neglog10p" in d.columns and d["neglog10p"].notna().any():
        nlp = d["neglog10p"].to_numpy(float)
    else:
        nlp = -np.log10(d["pvalue"].to_numpy(float).clip(1e-300))
    if log_y:
        # 0.05 floor keeps p~1 genes on a log axis without distorting the tail
        y = np.maximum(nlp, 0.05)
        capped = np.zeros(len(nlp), dtype=bool)
    else:
        y = np.minimum(nlp, ycap)
        capped = nlp > ycap
    sig = d["fdr"].to_numpy(float) < fdr
    up, dn, ns = sig & (lfc > 0), sig & (lfc < 0), ~sig
    ax.scatter(x[ns], y[ns], s=3, c="lightgrey", alpha=0.25, linewidths=0, rasterized=True)
    ax.scatter(x[dn], y[dn], s=4, c="#1f77b4", alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(x[up], y[up], s=4, c="#d62728", alpha=0.5, linewidths=0, rasterized=True)
    # Draw whatever is still saturated as hollow markers on the cap line, so the
    # remaining flat top is visibly a rendering limit rather than a real plateau.
    if capped.any():
        ax.scatter(x[capped & (lfc > 0)], y[capped & (lfc > 0)], s=13, facecolors="none",
                   edgecolors="#d62728", linewidths=0.45, rasterized=True)
        ax.scatter(x[capped & (lfc < 0)], y[capped & (lfc < 0)], s=13, facecolors="none",
                   edgecolors="#1f77b4", linewidths=0.45, rasterized=True)
    if sig.any():
        pthr = d.loc[sig, "pvalue"].max()
        if pthr > 0:
            ax.axhline(-np.log10(pthr), ls="--", lw=0.7, c="0.4")
    ax.axvline(0, ls="-", lw=0.5, c="0.7")
    for v in (-1, 1):
        ax.axvline(v, ls=":", lw=0.6, c="0.85")
    ax.set_xlim(-xlim, xlim)
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(labelsize=7)
    tag = f"sig={int(sig.sum())} (↑{int(up.sum())} ↓{int(dn.sum())})"
    if capped.any():
        tag += f"\n{int(capped.sum())} above y-cap (max {np.nanmax(nlp):,.0f})"
    if small_tag:
        if title:
            ax.set_title(title, fontsize=9)
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, fontsize=6.5, va="top",
                ha="left", color="0.25")
    else:
        ax.set_title((title + "\n" if title else "") + tag, fontsize=9)
    ax.set_xlabel("log2FC", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def _panel(methods, get_df, suptitle, outpath, *, fdr, xlim, ycap, log_y=False):
    """1 x len(methods) volcano figure; get_df(m) -> DataFrame for method m."""
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods) + 1, 4.4), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for k, m in enumerate(methods):
        _draw(axes[k], get_df(m), fdr=fdr, xlim=xlim, ycap=ycap, log_y=log_y, title=m,
              ylabel="-log10(p-value)" if k == 0 else None)
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--xlim", type=float, default=8.0)
    ap.add_argument("--ycap", type=float, default=300.0)
    ap.add_argument("--log-y", action="store_true",
                    help="Log-scale the y axis and drop the cap. Now that neglog10p is exact "
                         "past the float64 floor, the tail spans ~4 decades; a linear axis can "
                         "only render that as a flat line at the cap. Writes *_logy.png.")
    ap.add_argument("--celltype", default=None, help="only this one cell type")
    args = ap.parse_args()
    res = Path(args.results_dir)

    methods, frames = [], {}
    for m in METHODS:
        f = res / f"de_{m}_all_celltypes.csv"
        if not f.is_file():
            continue
        d = pd.read_csv(f, low_memory=False)
        d = d[d["tested"] == True].dropna(subset=["log2fc", "pvalue"])  # noqa: E712
        if len(d):
            methods.append(m); frames[m] = d
    celltypes = sorted(set().union(*[set(frames[m]["celltype"].dropna().unique()) for m in methods]))
    if args.celltype:
        celltypes = [c for c in celltypes if c == args.celltype]
    print(f"methods: {methods}\ncell types: {celltypes}")

    kw = dict(fdr=args.fdr, xlim=args.xlim, ycap=args.ycap, log_y=args.log_y)
    sfx = "_logy" if args.log_y else ""   # keep linear and log figures side by side

    # (1) one figure PER CELL TYPE
    for ct in celltypes:
        safe = ct.replace(" ", "_").replace("/", "-")
        _panel(methods, lambda m, ct=ct: frames[m][frames[m]["celltype"] == ct],
               f"Volcano — {ct} (red=up in pancreas, blue=down, grey=ns @ fdr<{args.fdr})",
               res / f"volcano_{safe}{sfx}.png", **kw)

    # (2) cell types (rows) x methods (cols) grid overview
    if len(celltypes) > 1:
        fig, axes = plt.subplots(len(celltypes), len(methods), sharex=True,
                                 figsize=(3.0 * len(methods) + 1, 3.4 * len(celltypes) + 1))
        axes = np.atleast_2d(axes)
        for i, ct in enumerate(celltypes):
            for j, m in enumerate(methods):
                d = frames[m][frames[m]["celltype"] == ct]
                _draw(axes[i, j], d, title=(m if i == 0 else None),
                      ylabel=(f"{ct}\n-log10(p)" if j == 0 else None), small_tag=True, **kw)
        fig.suptitle("DE method comparison — volcano plots: cell types (rows) x methods (cols); "
                     "shared log2FC axis (MAST narrow = compressed)", fontsize=12, y=1.005)
        fig.tight_layout()
        p = res / f"volcano_grid{sfx}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {p}")

    # (3) pooled (kept for reference)
    if not args.celltype:
        _panel(methods, lambda m: frames[m],
               f"Volcano — ALL cell types pooled (red=up, blue=down, ns @ fdr<{args.fdr})",
               res / f"volcano_all_panel{sfx}.png", **kw)


if __name__ == "__main__":
    main()
