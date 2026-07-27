#!/usr/bin/env python
"""
make_all_panels.py — regenerate every figure so that each image holds one plot.

Composed multi-panel figures are convenient in a manuscript but awkward to
inspect, and journals often want the panels as separate files. This runs each
generator in split mode and gathers the result in evidence/figures/panels/.

Three kinds of figure are handled differently:

  * fig1, fig2, fig4, the benchmark and the simulation are composed figures with
    a generator in this repository, so they are re-rendered panel by panel from
    the live figure object -- full resolution, each panel keeping its own title,
    axis labels and legend.
  * the heatmaps and volcano plots are already produced one per method and one
    per cell type by make_heatmaps.py / make_volcano.py; only the "grid" and
    "all_panel" composites are multi-panel, so the existing single files are
    copied rather than re-derived.
  * figS1_pvalue_underflow.png has no generator in this repository. It is
    reported as such rather than being cropped, because cropping a finished PNG
    would lose resolution and silently produce something that cannot be
    regenerated.

Usage: python scripts/make_all_panels.py [--skip-slow]
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
SCRIPTS = HERE / "scripts"
PANELS = HERE / "evidence" / "figures" / "panels"

# generator, arguments, whether it needs the big h5ad (and so is slow)
GENERATORS = [
    ("make_fig4.py", ["--split"], False),
    ("make_fig2.py", ["--split"], False),
    ("make_fig1.py", ["--split"], True),
    ("bench_mast_report.py", [], False),
    ("evaluate_sim.py", [], True),
]

# already single-panel, just collect them
COPY_SINGLES = [
    ("results/pancreas", "heatmap_*.png", ("heatmap_all_panel.png",
                                          "heatmap_all_panel_shared.png")),
    ("results/pancreas", "volcano_*.png", ("volcano_grid.png", "volcano_grid_logy.png",
                                          "volcano_all_panel.png",
                                          "volcano_all_panel_logy.png")),
]

NO_GENERATOR = ["figS1_pvalue_underflow.png"]


def run(script: str, args: list[str]) -> bool:
    print(f"\n--- {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                       capture_output=True, text=True, cwd=str(HERE))
    for line in r.stdout.splitlines():
        if any(k in line for k in ("wrote", "single-panel", "split into", "fig")):
            print("   ", line.strip())
    if r.returncode != 0:
        print(f"    FAILED ({r.returncode}): {r.stderr.strip().splitlines()[-1:]}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip generators that load the large h5ad objects")
    a = ap.parse_args()

    PANELS.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for script, args, slow in GENERATORS:
        if slow and a.skip_slow:
            print(f"\n--- {script}  SKIPPED (--skip-slow)")
            continue
        if not (SCRIPTS / script).is_file():
            print(f"\n--- {script}  MISSING")
            fail += 1
            continue
        ok += run(script, args)

    print("\n--- collecting figures that are already single-panel")
    for subdir, pattern, composites in COPY_SINGLES:
        src = HERE / subdir
        if not src.is_dir():
            print(f"    {subdir} not present, skipped")
            continue
        for f in sorted(src.glob(pattern)):
            if f.name in composites:
                continue
            shutil.copy2(f, PANELS / f.name)
            print(f"    {f.name}")

    n = len(list(PANELS.glob("*.png")))
    print(f"\n{n} single-panel images in {PANELS}")
    for name in NO_GENERATOR:
        if (HERE / "evidence" / "figures" / name).is_file():
            print(f"NOTE: {name} has no generator in this repository and was left "
                  f"as it is.")


if __name__ == "__main__":
    main()
