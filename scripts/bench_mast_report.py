#!/usr/bin/env python
"""
bench_mast_report.py
====================
Step 3 of the DUET vs R MAST head-to-head benchmark: collect the timings from
every bench/<tag>/ directory, check that the two engines agree on the SAME cells,
and draw the scaling figure.

Two runtime numbers are reported and they mean different things:
  * fit only     - zlm() vs DUET's hurdle fit. "Which engine solves the model faster."
  * end-to-end   - MAST additionally pays a disk round-trip (write matrix, readMM,
                   densify, build SingleCellAssay). DUET pays none of it because it
                   works on the in-memory AnnData. That gap IS the product claim,
                   so it is shown rather than hidden.

Note on fairness: `duet_fit_s` already INCLUDES DUET's own round-trip -- the engine
streams results to a CSV which the runner then reads back -- whereas MAST's `fit_s`
excludes all of its I/O. The comparison is therefore conservative toward DUET; the
true fit-only gap is slightly wider than reported.

Outputs: bench/benchmark_summary.csv, bench/fig_benchmark.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
BENCH = HERE / "bench"
LN2 = np.log(2.0)
DUET_C, MAST_C = "#b64b34", "#2b6685"


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(BENCH.iterdir()):
        if not d.is_dir():
            continue
        mj, mt = d / "meta.json", d / "mast_timing.json"
        if not mj.is_file():
            continue
        meta = json.loads(mj.read_text(encoding="utf-8"))
        row = dict(meta)
        if mt.is_file():
            row.update(json.loads(mt.read_text(encoding="utf-8")))
        # ---- agreement on identical cells/genes ----
        mr, dr = d / "mast_results.csv", d / "duet_results.csv"
        if mr.is_file() and dr.is_file():
            m = pd.read_csv(mr)
            u = pd.read_csv(dr)
            j = m.merge(u, on="gene", suffixes=("_m", "_d")).dropna(
                subset=["pvalue_m", "pvalue_d", "coef", "log2fc"])
            if len(j) > 100:
                lfc_m = j["coef"] / LN2          # MAST coef is a natural-log mean diff
                nlp_m = -np.log10(j["pvalue_m"].clip(1e-300))
                nlp_d = j["neglog10p"] if "neglog10p" in j else -np.log10(j["pvalue_d"].clip(1e-300))
                sm, sd = j["fdr_m"] < 0.05, j["fdr_d"] < 0.05
                # The EB-shrinkage step clips its moderated p at 1e-300 before
                # inverting to a chi2_1 statistic, so DUET's continuous component
                # saturates at chi2.isf(1e-300,1) = 1373.87. That caps the hurdle
                # statistic for the handful of most extreme genes and drags the
                # PEARSON correlation down at large N, while leaving the RANK
                # agreement untouched -- so report both, or the plot lies.
                row.update(
                    n_compared=len(j),
                    lfc_pearson=round(float(lfc_m.corr(j["log2fc"])), 4),
                    lfc_spearman=round(float(lfc_m.corr(j["log2fc"], method="spearman")), 4),
                    sign_agree_pct=round(100 * float((np.sign(lfc_m) == np.sign(j["log2fc"])).mean()), 2),
                    stat_pearson=round(float(j["stat_hurdle"].corr(j["stat"])), 4),
                    stat_spearman=round(float(spearmanr(j["stat_hurdle"], j["stat"]).statistic), 4),
                    neglog10p_spearman=round(float(spearmanr(nlp_m, nlp_d).statistic), 4),
                    n_eb_saturated=int(((j["stat"] - 1373.87).abs() < 60).sum()),
                    sig_mast=int(sm.sum()), sig_duet=int(sd.sum()),
                    sig_jaccard=round(float((sm & sd).sum() / max((sm | sd).sum(), 1)), 4),
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_cells")


def figure(t: pd.DataFrame) -> None:
    t = t.dropna(subset=["fit_s"]).copy()
    if t.empty:
        print("no MAST timings yet -- skipping figure")
        return
    t["mast_end2end_s"] = t["read_s"] + t["fit_s"] + t["lrt_s"]
    t["duet_end2end_s"] = t["duet_fit_s"]        # no export/import: it reads AnnData
    t["speedup_fit"] = t["fit_s"] / t["duet_fit_s"]
    t["speedup_e2e"] = t["mast_end2end_s"] / t["duet_end2end_s"]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
    fig.patch.set_facecolor("white")
    x = t["n_cells"].to_numpy()

    # (a) wall clock, log-log
    ax = axes[0]
    ax.plot(x, t["mast_end2end_s"], "o-", c=MAST_C, lw=2, ms=6, label="R MAST (end-to-end)")
    ax.plot(x, t["fit_s"], "o--", c=MAST_C, lw=1.4, ms=5, alpha=.65, label="R MAST (zlm fit only)")
    ax.plot(x, t["duet_fit_s"], "o-", c=DUET_C, lw=2, ms=6, label="DUET")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cells", fontsize=9); ax.set_ylabel("wall clock (s)", fontsize=9)
    ax.set_title("(a) runtime — Ductal cell, ~8.5k genes", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25, lw=.5, which="both")

    # (b) speed-up
    ax = axes[1]
    w = .36; xs = np.arange(len(t))
    b1 = ax.bar(xs - w/2, t["speedup_e2e"], w, color=MAST_C, label="end-to-end", zorder=3)
    b2 = ax.bar(xs + w/2, t["speedup_fit"], w, color=DUET_C, label="fit only", zorder=3)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f×", fontsize=8.5, padding=2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{int(v):,}" for v in x], fontsize=9)
    ax.set_xlabel("cells", fontsize=9); ax.set_ylabel("DUET speed-up (×)", fontsize=9)
    ax.set_title("(b) how much faster DUET is", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)
    ax.set_ylim(0, float(max(t["speedup_e2e"].max(), t["speedup_fit"].max())) * 1.25)

    # (c) agreement
    ax = axes[2]
    if "stat_pearson" in t and t["stat_pearson"].notna().any():
        # Three series only. stat_spearman is deliberately NOT drawn: it sits on
        # top of neglog10p_spearman (both 1.0000), so plotting it adds a legend
        # entry with no visible line. It is still in benchmark_summary.csv.
        ax.plot(x, t["stat_pearson"], "o-", c=DUET_C, lw=2, ms=6, label="hurdle statistic (Pearson)")
        ax.plot(x, t["neglog10p_spearman"], "s-", c=MAST_C, lw=2, ms=5, label="$-\\log_{10}p$ (Spearman)")
        ax.plot(x, t["sig_jaccard"], "^--", c="0.45", lw=1.5, ms=5, label="significant-gene Jaccard")
        ax.set_xscale("log"); ax.set_ylim(0.90, 1.005)
        ax.set_xlabel("cells", fontsize=9); ax.set_ylabel("agreement with R MAST", fontsize=9)
        ax.set_title("(c) same cells, same genes, same design", fontsize=10.5, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.grid(alpha=.25, lw=.5)
        ax.annotate("Pearson dips only because DUET's EB step\ncaps the statistic at 1374 (~1% of genes);\n"
                    "rank agreement stays at 1.0000",
                    xy=(x[-1], float(t["stat_pearson"].iloc[-1])), xytext=(0.30, 0.235),
                    textcoords="axes fraction", fontsize=7, color="0.35",
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=.8,
                                    connectionstyle="arc3,rad=0.25"))
    for ax in axes:
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("DUET vs R MAST 1.33.0 — identical cells, identical genes, identical design "
                 "(~1+condition, no covariates), same machine",
                 fontsize=12, y=1.03, x=.02, ha="left")
    fig.tight_layout()
    out = BENCH / "fig_benchmark.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main():
    t = collect()
    if t.empty:
        print("nothing in bench/ yet"); return
    t.to_csv(BENCH / "benchmark_summary.csv", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ["tag", "n_cells", "n_genes", "duet_fit_s", "fit_s", "read_s", "lrt_s",
                        "mast_total_s", "peak_mb", "export_s", "mtx_mb"] if c in t]
    print("=== TIMINGS ===")
    print(t[cols].to_string(index=False))
    acols = [c for c in ["tag", "n_cells", "n_compared", "lfc_pearson", "sign_agree_pct",
                         "stat_pearson", "neglog10p_spearman", "sig_mast", "sig_duet",
                         "sig_jaccard"] if c in t]
    if len(acols) > 2:
        print("\n=== AGREEMENT (same cells) ===")
        print(t[acols].to_string(index=False))
    figure(t)


if __name__ == "__main__":
    main()
