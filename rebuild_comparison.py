#!/usr/bin/env python
"""
rebuild_comparison.py
=====================
Rebuild comparison_per_gene / comparison_concordance / consensus_genes /
comparison_summary from whatever per-method `de_<method>_all_celltypes.csv`
files are present in a results directory.

Needed because run_de_comparison.py rebuilds the comparison from only the methods
it just ran -- so re-running a single method to fix it would silently replace a
five-method comparison with a one-method one. This reassembles from disk instead.

Usage:
  python rebuild_comparison.py --results-dir results_kang
  python rebuild_comparison.py --results-dir results --methods DUET,MAST,diffxpy,DESeq2,wilcoxon
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_de_comparison as R          # noqa: E402

PREFERRED = ["DUET", "MAST", "diffxpy", "DESeq2", "wilcoxon"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--methods", default=None,
                    help="comma-separated; default = every de_*_all_celltypes.csv found")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=100)
    a = ap.parse_args()

    res = Path(a.results_dir)
    if a.methods:
        names = [m.strip() for m in a.methods.split(",") if m.strip()]
    else:
        found = {p.name[len("de_"):-len("_all_celltypes.csv")]
                 for p in res.glob("de_*_all_celltypes.csv")}
        names = [m for m in PREFERRED if m in found] + sorted(found - set(PREFERRED))

    parts = []
    for m in names:
        f = res / f"de_{m}_all_celltypes.csv"
        if not f.is_file():
            print(f"[rebuild] missing, skipping: {f.name}")
            continue
        d = pd.read_csv(f, low_memory=False)
        n_tested = int((d["tested"] == True).sum())            # noqa: E712
        print(f"[rebuild] {m:10s} rows={len(d):7d} tested={n_tested:7d} "
              f"celltypes={sorted(d.loc[d.tested == True, 'celltype'].dropna().unique())}")  # noqa: E712
        if n_tested == 0:
            print(f"[rebuild]   ^ WARNING: {m} has no tested genes; it will appear "
                  f"in the method list but contribute nothing")
        parts.append(d)

    if not parts:
        raise SystemExit(f"[rebuild] no per-method CSVs in {res}")

    full = pd.concat(parts, ignore_index=True)
    R.build_comparison(full, fdr_thr=a.fdr, top_k=a.top_k, out_dir=res)
    print(f"\n[rebuild] rebuilt comparison_* in {res} from {len(parts)} methods")
    summ = res / "comparison_summary.txt"
    if summ.is_file():
        print()
        print("\n".join(summ.read_text(encoding="utf-8").splitlines()[:14]))


if __name__ == "__main__":
    main()
