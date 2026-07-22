#!/usr/bin/env python
"""
make_heatmaps.py
================
Build one log2FC heatmap per DE method (DUET, MAST, diffxpy, DESeq2, Wilcoxon) from
comparison_per_gene.csv. All four heatmaps use the SAME genes in the SAME order
(rows) and the 3 cell types as columns, so the methods can be compared directly.

Genes shown: a manageable set of the strongest DE genes (top up + top down by
Wilcoxon log2FC) that were tested by ALL four methods in ALL cell types.

Outputs (in results/):
  heatmap_<method>.png        -- one per method
  heatmap_all4_panel.png      -- the four side by side
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# methods shown as panels (order = display order). Only those with a
# de_<method>_all_celltypes.csv present are kept.
METHODS = ["DUET", "MAST", "diffxpy", "DESeq2", "wilcoxon"]
RANK_METHODS = ["diffxpy", "DESeq2", "wilcoxon"]  # linear-scale, all-gene methods used to rank/order genes
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gene", default=str(HERE / "results" / "comparison_per_gene.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--n-up", type=int, default=20)
    ap.add_argument("--n-down", type=int, default=20)
    ap.add_argument("--rank-method", default="wilcoxon", help="method used to rank/order genes")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_gene, low_memory=False)
    df["gene"] = df["gene"].astype(str)
    celltypes = sorted(df["celltype"].dropna().unique())
    # keep only methods that are actually present in the comparison table
    methods = [m for m in METHODS if f"log2fc_{m}" in df.columns]
    print(f"cell types: {celltypes}\nmethods: {methods}")

    # per-method gene x celltype log2FC and fdr matrices
    mats = {m: df.pivot_table(index="gene", columns="celltype",
                              values=f"log2fc_{m}", aggfunc="first").reindex(columns=celltypes)
            for m in methods}
    fdrs = {m: df.pivot_table(index="gene", columns="celltype",
                              values=f"fdr_{m}", aggfunc="first").reindex(columns=celltypes)
            for m in methods}

    # genes tested by ALL methods in ALL cell types (complete cases)
    complete = None
    for m in methods:
        ok = mats[m].notna().all(axis=1)
        complete = ok if complete is None else (complete & ok)
    genes_complete = complete[complete].index

    # robust selection (avoid lowly-expressed artifacts that explode log2FC):
    #   (a) significant (fdr<0.05) in ALL methods in >=1 cell type,
    #   (b) artifact guard: max |log2FC| across methods/cell types < 8,
    #   (c) rank by the MEDIAN of the well-scaled, all-gene methods (RANK_METHODS);
    #       DUET/MAST shown but excluded from ranking (both report the compressed
    #       log-mean-difference, a different scale from the linear-log2FC methods).
    rank_methods = [m for m in RANK_METHODS if m in methods] or methods
    sig_all = pd.Series(True, index=genes_complete)
    for m in methods:
        sig_all &= (fdrs[m].loc[genes_complete] < 0.05).any(axis=1)
    maxabs = pd.concat([mats[m].loc[genes_complete].abs().max(axis=1) for m in methods],
                       axis=1).max(axis=1)
    eff = pd.concat([mats[m].loc[genes_complete].mean(axis=1)
                     for m in rank_methods], axis=1).median(axis=1)
    cand = eff[sig_all & (maxabs < 8.0)].sort_values()
    print(f"complete-case genes={len(genes_complete)}; sig-in-all & non-artifact={len(cand)}")

    down = cand.head(args.n_down).index.tolist()
    up = cand.tail(args.n_up).index.tolist()
    order = cand.loc[down + up].sort_values(ascending=False).index.tolist()  # up at top
    print(f"showing {len(order)} genes ({args.n_up} up + {args.n_down} down by median log2FC)")

    def draw(ax, m, show_yticks=True, cbar=True, vmax=None):
        M = mats[m].loc[order].to_numpy(dtype=float)
        if vmax is None:
            vmax = np.nanpercentile(np.abs(M), 98) or 1.0
            title = f"{m}\n(|log2FC| scale ±{vmax:.2f})"
        else:
            title = m  # shared scale -> stated once on the figure
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(celltypes)))
        ax.set_xticklabels([c.replace(" cell", "") for c in celltypes], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)))
        if show_yticks:
            ax.set_yticklabels(order, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=9)
        # annotate values
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        if cbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log2FC")
        return im

    # 1) one PNG per method
    for m in methods:
        fig, ax = plt.subplots(figsize=(3.6, 0.22 * len(order) + 1.5))
        draw(ax, m, show_yticks=True, cbar=True)
        fig.suptitle(f"{m} — log2FC (pancreas vs Reference)", fontsize=10, y=1.005)
        fig.tight_layout()
        p = out / f"heatmap_{m}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {p}")

    # 2a) combined panel, PER-METHOD scale (each panel auto-scaled -> patterns visible)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.1 * len(methods) + 1, 0.22 * len(order) + 1.8))
    if len(methods) == 1:
        axes = [axes]
    for k, m in enumerate(methods):
        draw(axes[k], m, show_yticks=(k == 0), cbar=True)
    fig.suptitle("DE method comparison — log2FC of top DE genes (rows) x cell types (cols); "
                 "PER-METHOD colour scale (compare PATTERN, not intensity)", fontsize=11, y=1.01)
    fig.tight_layout()
    p = out / "heatmap_all_panel.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")

    # 2b) combined panel, SHARED scale (same colour == same log2FC everywhere;
    #     MAST looks paler because its log-mean-diff log2FC is compressed)
    allvals = np.concatenate([mats[m].loc[order].to_numpy(dtype=float).ravel() for m in methods])
    gvmax = float(np.nanpercentile(np.abs(allvals), 98)) or 1.0
    fig, axes = plt.subplots(1, len(methods), figsize=(3.0 * len(methods) + 1.5, 0.22 * len(order) + 1.8))
    if len(methods) == 1:
        axes = [axes]
    im = None
    for k, m in enumerate(methods):
        im = draw(axes[k], m, show_yticks=(k == 0), cbar=False, vmax=gvmax)
    fig.suptitle(f"DE method comparison — SHARED colour scale (±{gvmax:.2f} log2FC for all panels); "
                 "MAST is paler = compressed log-mean-diff scale", fontsize=11, y=1.01)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes((0.945, 0.15, 0.012, 0.7))
    fig.colorbar(im, cax=cax, label="log2FC (shared)")
    p = out / "heatmap_all_panel_shared.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
