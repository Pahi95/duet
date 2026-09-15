#!/usr/bin/env python
"""
make_fig_bench.py -- the benchmark figure (replaces Figure 1e; MO tasks 2 and 13),
from bench_harness.py output only. Medians over the timed runs; bars span the
interquartile range.

  (a) end-to-end wall time against cells, real pancreas ladder (Ductal; pooled three
      cell types), DUET and MAST, log-log
  (b) peak minus baseline resident memory of the process tree against cells
  (c) synthetic grid, cells varied at 8,000 genes          } one factor at a time
  (d) synthetic grid, genes varied at 10,000 cells        }
  (e) synthetic grid, zero fraction varied                }
  (f) ~condition vs ~CDR + condition at 10,000 cells x 8,000 genes

Inputs: results/bench/fair.csv (real ladder), results/bench/grid.csv (synthetic grid),
        results/bench/grid_cdr.csv (CDR model), inputs/bench/*/data.h5ad (sizes)
Outputs: evidence/figures/fig_bench.png (+ .pdf) and fig_bench_<a..f>.png
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
COL = {"duet": "#1f77b4", "mast": "#9467bd"}
NAME = {"duet": "DUET", "mast": "MAST"}


def sizes(name: str) -> dict:
    return pd.read_csv(B / "benchmark_inputs.csv").set_index("input").loc[name].to_dict()

def summarise(csv: Path) -> pd.DataFrame | None:
    if not csv.is_file():
        return None
    d = pd.read_csv(csv)
    d = d[(~d.warmup) & (d.stage == "TOTAL_RUN")]
    # the machine was in use during the benchmark: CPU time (single-threaded runs) is the primary timing
    col = "cpu_s" if "cpu_s" in d.columns and d["cpu_s"].notna().all() else "elapsed_s"
    g = d.groupby(["input", "engine", "design"]).agg(
        t=(col, "median"), t_q1=(col, lambda x: x.quantile(0.25)),
        t_q3=(col, lambda x: x.quantile(0.75)), mem=("delta_rss_mb", "median"),
        mem_q1=("delta_rss_mb", lambda x: x.quantile(0.25)), mem_q3=("delta_rss_mb", lambda x: x.quantile(0.75)),
        n=("run", "nunique")).reset_index()
    return g.join(pd.DataFrame([sizes(i) for i in g["input"]], index=g.index))


def line(ax, g, x, y="t", q=("t_q1", "t_q3"), label_suffix=""):
    for eng, h in g.groupby("engine"):
        h = h.sort_values(x)
        ax.errorbar(h[x], h[y], yerr=[h[y] - h[q[0]], h[q[1]] - h[y]], color=COL[eng], marker="o", ms=4,
                    capsize=2, lw=1.2, label=f"{NAME[eng]}{label_suffix}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    real, grid, cdr = summarise(B / "fair.csv"), summarise(B / "grid.csv"), summarise(B / "grid_cdr.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    (a, b, c), (d, e, f) = axes
    if real is not None:
        for series, ls in (("ductal", "-"), ("pooled", "--")):
            s = real[real.input.str.startswith(series)]
            for eng, h in s.groupby("engine"):
                h = h.sort_values("cells")
                a.plot(h.cells, h.t, ls=ls, marker="o", ms=4, color=COL[eng], label=f"{NAME[eng]} ({series})")
                b.plot(h.cells, h.mem, ls=ls, marker="o", ms=4, color=COL[eng], label=f"{NAME[eng]} ({series})")
        a.set(xscale="log", yscale="log", xlabel="cells", ylabel="end-to-end CPU time (s)",
              title="(a) real ladder: time")
        b.set(xscale="log", xlabel="cells", ylabel="peak - baseline RSS (MiB)", title="(b) real ladder: memory")
        a.legend(fontsize=7, frameon=False)
        b.legend(fontsize=7, frameon=False)
    if grid is not None:
        line(c, grid[grid.input.str.startswith("cells_")], "cells")
        c.set(xscale="log", yscale="log", xlabel="cells (8,000 genes)", ylabel="CPU time (s)",
              title="(c) cells varied, genes fixed")
        line(d, grid[grid.input.str.startswith("genes_")], "genes")
        d.set(xscale="log", yscale="log", xlabel="genes (10,000 cells)", ylabel="CPU time (s)",
              title="(d) genes varied, cells fixed")
        line(e, grid[grid.input.str.startswith("sparsity_")], "zero_fraction")
        e.set(yscale="log", xlabel="zero fraction (10,000 cells x 8,000 genes)", ylabel="CPU time (s)",
              title="(e) sparsity varied")
        for ax, x in ((c, "cells"), (d, "genes")):                   # label the grid points themselves
            pts = sorted(grid.loc[grid.input.str.startswith(x + "_"), x].unique())
            ax.set_xticks(pts, [f"{v:,}" for v in pts])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for ax in (c, d, e):
            ax.legend(fontsize=7, frameon=False)
    if False:  # Unmatched historical CDR results are excluded.
        base = grid[grid.input == "sparsity_mid_c10000"].assign(model="~condition")
        withc = cdr.assign(model="~CDR + condition")
        both = pd.concat([base, withc])
        x = np.arange(2)
        for k, (eng, h) in enumerate(both.groupby("engine")):
            h = h.set_index("model").reindex(["~condition", "~CDR + condition"])
            f.bar(x + (k - 0.5) * 0.35, h.t, 0.35, yerr=[h.t - h.t_q1, h.t_q3 - h.t], color=COL[eng], label=NAME[eng])
        f.set_xticks(x)
        f.set_xticklabels(["~condition", "~CDR + condition"])
        f.set(yscale="log", ylabel="CPU time (s)", title="(f) covariate model (10,000 x 8,000)")
        f.legend(fontsize=7, frameon=False)
    f.axis("off")
    f.text(.5, .5, "(f) CDR timing excluded\nEB and gating settings were unmatched\n(see supplementary Table S5)", ha="center", va="center", fontsize=11)
    fig.savefig(FIG / "fig_bench.png", dpi=200)
    fig.savefig(FIG / "fig_bench.pdf")
    from _figutil import save_panels                              # one plot per image, neighbours hidden
    save_panels(fig, FIG, "fig_bench")
    plt.close(fig)
    print(f"[bench-fig] wrote {FIG / 'fig_bench.png'}")


if __name__ == "__main__":
    main()
