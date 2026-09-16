#!/usr/bin/env python
"""
evaluate_sim.py
===============
Score every DE method against the muscat ground truth: ROC/AUC, power at a nominal
FDR, and -- the number that actually matters -- the OBSERVED false discovery rate
against the nominal one.

Concordance between methods says only that they agree. This says whether they are
right, and it is the difference between "agrees with MAST" and "is accurate".

Scoring rules, all deliberate:
  * positives = genes muscat labelled de/dp/dm/db (a real distributional shift);
    negatives = ee/ep. Genes with no label are excluded.
  * every method is scored on the SAME gene set (the intersection of what the
    methods tested), so differing gene filters cannot flatter anyone.
  * ranking uses `neglog10p` where present, so the float64 underflow does not
    silently scramble the top of the ROC curve.
  * observed FDR is computed at the method's OWN nominal FDR<0.05 call set.

Usage: python evaluate_sim.py --results-dir results/sim --h5ad ../inputs/data/sim_muscat.h5ad
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figutil import save_panels   # noqa: E402

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
PREFERRED = ["DUET", "MAST", "diffxpy", "DESeq2", "wilcoxon"]
COLORS = {"DUET": "#b64b34", "MAST": "#2b6685", "diffxpy": "#9c7222",
          "DESeq2": "#3d7a55", "wilcoxon": "#7a5c8f"}


def roc_curve_xy(y, s):
    o = np.argsort(-s)
    y = np.asarray(y)[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return fp / max(fp[-1], 1), tp / max(tp[-1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results" / "sim"))
    ap.add_argument("--h5ad", default=str(INPUTS / "data" / "sim_muscat.h5ad"))
    ap.add_argument("--fdr", type=float, default=0.05)
    a = ap.parse_args()
    res = Path(a.results_dir)

    A = sc.read_h5ad(a.h5ad, backed="r")
    truth = A.var[["category", "is_de", "is_null"]].copy()
    truth.index.name = "gene"
    truth = truth.reset_index()

    # muscat's `ep` = differential PROPORTION with the same overall mean. That is a
    # null for a mean-shift test but a real distributional difference, which a
    # hurdle model may legitimately detect. Scoring it as a false positive would
    # unfairly penalise DUET/MAST, so `ep` is EXCLUDED from both classes here and
    # reported separately; `ee` alone is the unambiguous null.
    n_ep = int((truth.category == "ep").sum())
    truth = truth[truth.is_de | (truth.category == "ee")]
    print(f"[eval] truth: {int(truth.is_de.sum())} DE (de/dp/dm/db), "
          f"{int((truth.category == 'ee').sum())} unambiguous null (ee); "
          f"{n_ep} 'ep' genes excluded (same mean, different proportion)")

    frames = {}
    for m in PREFERRED:
        f = res / f"de_{m}_all_celltypes.csv"
        if not f.is_file():
            continue
        d = pd.read_csv(f, low_memory=False)
        d = d[d["tested"] == True].dropna(subset=["pvalue"])       # noqa: E712
        if len(d):
            frames[m] = d
    if not frames:
        raise SystemExit(f"[eval] no per-method CSVs in {res}")

    # score everyone on the same genes
    common = set.intersection(*(set(d.gene) for d in frames.values())) & set(truth.gene)
    print(f"[eval] scoring {len(common)} genes common to {list(frames)}")
    t = truth[truth.gene.isin(common)].set_index("gene")

    rows, curves = [], {}
    for m, d in frames.items():
        d = d[d.gene.isin(common)].drop_duplicates(subset="gene").set_index("gene")
        d = d.reindex(t.index)
        y = t["is_de"].to_numpy().astype(int)
        score = (d["neglog10p"] if "neglog10p" in d.columns and d["neglog10p"].notna().any()
                 else -np.log10(d["pvalue"].clip(1e-300))).to_numpy(float)
        score = np.nan_to_num(score, nan=0.0)
        called = (d["fdr"].to_numpy(float) < a.fdr)
        called = np.nan_to_num(called, nan=False).astype(bool)

        tp = int((called & (y == 1)).sum()); fp = int((called & (y == 0)).sum())
        fn = int((~called & (y == 1)).sum())
        obs_fdr = fp / max(tp + fp, 1)
        rows.append(dict(
            method=m, n_scored=len(y), n_called=int(called.sum()),
            auc=round(float(roc_auc_score(y, score)), 4),
            auprc=round(float(average_precision_score(y, score)), 4),
            tpr_at_fdr=round(tp / max(tp + fn, 1), 4),
            observed_fdr=round(obs_fdr, 4),
            fdr_control="OK" if obs_fdr <= a.fdr * 1.5 else "INFLATED",
            tp=tp, fp=fp, fn=fn))
        curves[m] = roc_curve_xy(y, score)

    S = pd.DataFrame(rows).sort_values("auc", ascending=False)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 86)
    print(f"GROUND-TRUTH EVALUATION  (nominal FDR < {a.fdr})")
    print("=" * 86)
    print(S.to_string(index=False))
    S.to_csv(res / "sim_evaluation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.patch.set_facecolor("white")
    ax = axes[0]
    for m, (x, yv) in curves.items():
        auc = float(S.loc[S.method == m, "auc"].iloc[0])
        ax.plot(x, yv, lw=2, c=COLORS.get(m, "0.4"), label=f"{m}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=.8, c="0.6")
    ax.set_xlabel("false positive rate", fontsize=9)
    ax.set_ylabel("true positive rate", fontsize=9)
    ax.set_title("(a) ROC against muscat ground truth", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.25, lw=.5)

    ax = axes[1]
    xs = np.arange(len(S)); w = .36
    b1 = ax.bar(xs - w/2, S["tpr_at_fdr"], w, color="#2b6685", label="power (TPR)", zorder=3)
    b2 = ax.bar(xs + w/2, S["observed_fdr"], w, color="#b64b34", label="observed FDR", zorder=3)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(a.fdr, ls="--", lw=1, c="0.35", zorder=4)
    ax.text(len(S) - .45, a.fdr + .02, f"nominal FDR {a.fdr}", fontsize=7.5,
            color="0.35", ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(S["method"], fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_title(f"(b) power vs actual false discovery rate", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)
    for ax_ in axes:
        ax_.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)

    fig.suptitle("Ground truth: muscat simulation from the Crowell reference "
                 "(4 vs 4 samples, realistic between-sample variance)",
                 fontsize=11.5, y=1.03, x=.02, ha="left")
    fig.tight_layout()
    out = res / "fig_simulation.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    n = save_panels(fig, HERE / "evidence" / "figures" / "panels", "fig_simulation", dpi=155)
    print(f"  + {len(n)} single-panel files")
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
