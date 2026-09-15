#!/usr/bin/env python
"""
make_fig_perf_summary.py -- one figure that summarises the matched DUET vs MAST benchmark (MO tasks 1, 2, 13),
from bench_harness.py output only (medians over the timed runs, CPU time: the machine was in use).

  (a) MAST/DUET CPU-time ratio at every benchmark point: end-to-end and fit + test only, against cells;
      the covariate (~CDR + condition) points are marked separately
  (b) where MAST's CPU time goes on the pancreas series, stage by stage (DUET end-to-end for comparison)
  (c) which MAST step needs the memory: peak resident memory per stage, dense vs sparse input, ebayes on/off
      (pancreas ductal cells, 8,020 x 9,058), with DUET's peak on the same input
  (d) peak resident memory against cells, both engines, pancreas series and simulated grid

Inputs: results/bench/{fair,grid,grid_cdr,real_cdr,mem_*}.csv, inputs/bench/*/data.h5ad (sizes)
Outputs: evidence/figures/fig_perf_summary.png (+ .pdf) and fig_perf_summary_<a..d>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import os                                                  # noqa: E402

import h5py                                                # noqa: E402
import matplotlib                                          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
B = HERE / "figure_data"
FIG = HERE / "figures"
DUET_C, MAST_C = "#b64b34", "#2b6685"


def cells(name: str) -> int:
    return int(pd.read_csv(B / "benchmark_inputs.csv").set_index("input").loc[name, "cells"])

def timed(csv: str) -> pd.DataFrame:
    d = pd.read_csv(B / csv)
    return d[~d.warmup.astype(bool)]


def per_point(csv: str) -> pd.DataFrame:
    """input x engine: median end-to-end CPU, median fit+test CPU, median peak memory."""
    d = timed(csv)
    tot = d[d.stage == "TOTAL_RUN"].groupby(["input", "engine"]).agg(cpu=("cpu_s", "median"), peak=("peak_rss_mb", "median"))
    comp = (d[d.stage.isin(["fit_test", "fit_test_write", "zlm", "lrtest"])].groupby(["input", "engine", "run"]).cpu_s.sum()
            .groupby(["input", "engine"]).median().rename("compute"))
    out = tot.join(comp).reset_index()
    out["cells"] = out["input"].map(cells)
    return out


def ratios(pts: pd.DataFrame) -> pd.DataFrame:
    w = pts.pivot(index=["input", "cells"], columns="engine", values=["cpu", "compute"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["e2e"] = w.cpu_mast / w.cpu_duet
    # Windows counts CPU time in ~16 ms ticks: a DUET fit below 0.5 s CPU gives no reliable ratio
    w["fit"] = np.where(w.compute_duet >= 0.5, w.compute_mast / w.compute_duet, np.nan)
    return w


def panel_a(ax):
    series = [("fair.csv", "pancreas series", "o", None), ("grid.csv", "simulated grid", "s", None),
              ("grid_cdr.csv", "~CDR + condition (pancreas 8,020 and simulated 10,000 cells)", "D", "cdr"),
              ("real_cdr.csv", None, "D", "cdr")]
    for csv, label, mk, kind in series:
        if kind == "cdr" or not (B / csv).is_file():
            continue
        r = ratios(per_point(csv))
        mfc = "white" if kind == "cdr" else None
        # with the covariate both ratios are about equal: nudge the end-to-end marker left so both show
        ax.scatter(r.cells * (0.9 if kind == "cdr" else 1.0), r.e2e, marker=mk, s=34, facecolors=mfc or MAST_C, edgecolors=MAST_C, lw=1.2,
                   label=f"end-to-end, {label}" if label else "_nolegend_", zorder=3)
        ok = r.fit.notna()
        ax.scatter(r.cells[ok], r.fit[ok], marker=mk, s=34, facecolors=mfc or DUET_C, edgecolors=DUET_C, lw=1.2,
                   label=f"fit + test only, {label}" if label else "_nolegend_", zorder=3)
        if csv == "fair.csv":
            # DUET's fit is too short for a CPU ratio on this series: show the elapsed-time ratio of the stage
            d = timed(csv)
            st = d[d.stage.isin(["fit_test", "zlm", "lrtest"])].groupby(["input", "engine", "run"]).elapsed_s.sum() \
                .groupby(["input", "engine"]).median().unstack()
            wall = (st.mast / st.duet).rename("r").reset_index()
            wall["cells"] = wall["input"].map(cells)
            ax.scatter(wall.cells, wall.r, marker="o", s=34, facecolors="none", edgecolors=DUET_C, lw=1.0,
                       label="fit + test only, pancreas series (elapsed time)", zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="cells", ylabel="MAST / DUET (median of 5 runs)",
           title="(a) how much faster: whole run vs model fit and test")
    ax.axhline(1, color="0.6", lw=0.8)
    ax.grid(alpha=.25, lw=.5, which="both")
    ax.legend(fontsize=6.3, frameon=False, loc="lower left", ncol=1)


def panel_b(ax):
    d = timed("fair.csv")
    m = d[d.engine == "mast"]
    # parts are computed run by run and then summarised by their median, so the start-up residual
    # (process CPU minus its timed stages) is never negative
    runs = m.groupby(["input", "run", "process", "stage"]).cpu_s.sum().unstack(["process", "stage"])
    r_stages = [c for c in runs.columns if c[0] == "R" and c[1] != "TOTAL_PROCESS"]
    per_run = pd.DataFrame({
        "Python export": runs[("export", "TOTAL_PROCESS")],
        "R read + SingleCellAssay": runs[("R", "read")] + runs[("R", "singlecellassay")],
        "zlm (model fit)": runs[("R", "zlm")],
        "lrTest": runs[("R", "lrtest")],
        "assemble + write": runs[("R", "assemble")] + runs[("R", "write")],
        "R start-up and packages": (runs[("R", "TOTAL_PROCESS")] - runs[r_stages].sum(axis=1)).clip(lower=0),
    })
    med = per_run.groupby("input").median()
    tot = m[m.stage == "TOTAL_RUN"].groupby("input").cpu_s.median()
    duet = d[(d.engine == "duet") & (d.stage == "TOTAL_RUN")].groupby("input").cpu_s.median()
    order = sorted(tot.index, key=lambda n: (n.split("_")[0], cells(n)))
    parts = {k: med[k] for k in med.columns}
    colors = ["#c9c9c9", "#8fb3c9", MAST_C, "#6f97b3", "#e0e0e0", "#a0a0a0"]
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for (name, s), c in zip(parts.items(), colors):
        vals = s.reindex(order).to_numpy()
        ax.barh(y, vals, left=left, color=c, label=name, edgecolor="white", lw=0.4)
        left += vals
    for k, n in enumerate(order):
        ax.text(left[k] + 8, k, f"{tot[n]:.0f} s   (DUET {duet[n]:.1f} s)", va="center", fontsize=6.5)
    ax.set_yticks(y, [f"{n}  ({cells(n):,} cells)" for n in order], fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="MAST CPU time (s), median of 5 runs", xlim=(0, tot.max() * 1.45),
           title="(b) where MAST's time goes (pancreas series)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")


def panel_c(ax):
    stages = ["read", "singlecellassay", "zlm", "lrtest"]
    labels = ["R read", "SingleCellAssay", "zlm", "lrTest"]
    settings = [("dense", "TRUE", "dense, ebayes on (default)"), ("dense", "FALSE", "dense, ebayes off"),
                ("sparse", "TRUE", "sparse, ebayes on"), ("sparse", "FALSE", "sparse, ebayes off")]
    colors = [MAST_C, "#7fa6c0", "#8a5a9e", "#c2a5cf"]
    x = np.arange(len(stages))
    w = 0.2
    for k, ((inp, eb, lab), c) in enumerate(zip(settings, colors)):
        d = timed(f"mem_{inp}_eb{eb}.csv")
        s = d[d.process == "R"].groupby("stage").peak_rss_mb.median().reindex(stages)
        ax.bar(x + (k - 1.5) * w, s.to_numpy(), w, color=c, label=f"MAST, {lab}")
    duet = timed("fair.csv")
    duet_peak = duet[(duet.engine == "duet") & (duet.input == "ductal_all") & (duet.stage == "TOTAL_RUN")].peak_rss_mb.median()
    ax.axhline(duet_peak, color=DUET_C, lw=1.6, ls="--", label=f"DUET, whole run ({duet_peak:,.0f} MiB)")
    ax.set_xticks(x, labels)
    ax.set(ylabel="peak resident memory (MiB)", title="(c) which MAST step needs the memory (8,020 × 9,058)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.25, lw=.5)


def panel_d(ax):
    for csv, prefix, mk, lab in (("fair.csv", "ductal", "o", "pancreas Ductal"), ("fair.csv", "pooled", "o", "pancreas pooled"),
                                 ("grid.csv", "cells_", "s", "simulated, 8,000 genes")):
        p = per_point(csv)
        p = p[p["input"].str.startswith(prefix)]
        for eng, c in (("duet", DUET_C), ("mast", MAST_C)):
            q = p[p.engine == eng].sort_values("cells")
            ax.plot(q.cells, q.peak, marker=mk, ms=4, lw=1.2, color=c, ls="-" if csv == "grid.csv" else "--",
                    mfc="white" if prefix == "pooled" else c, label=f"{eng.upper() if eng == 'mast' else 'DUET'}, {lab}")
    ax.set(xscale="log", yscale="log", xlabel="cells", ylabel="peak resident memory (MiB)",
           title="(d) peak memory against cells")
    ax.grid(alpha=.25, lw=.5, which="both")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    panel_a(axes[0, 0]); panel_b(axes[0, 1]); panel_c(axes[1, 0]); panel_d(axes[1, 1])
    for ax in axes.flat:
        ax.tick_params(labelsize=7.5)
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(FIG / "fig_perf_summary.png", dpi=200)
    fig.savefig(FIG / "fig_perf_summary.pdf")
    from _figutil import save_panels
    save_panels(fig, FIG, "fig_perf_summary")
    plt.close(fig)
    print(f"[perf-fig] wrote {FIG / 'fig_perf_summary.png'}")


if __name__ == "__main__":
    main()
