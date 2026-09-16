#!/usr/bin/env python
"""
bench_duet_run.py -- one engine run for the timing/memory benchmark.
Driven by bench_harness.py; run in a fresh process each time.

  python bench_duet_run.py <data.h5ad> <workdir> [--design condition|cdr]
      DUET from the .h5ad on disk: read -> fit + test -> write.
  python bench_duet_run.py <data.h5ad> <workdir> --export
      The Python half of the MAST pipeline: read the same .h5ad and write the
      genes x cells matrix, gene and cell lists and conditions that R reads.

Both start from the same file, so the two engines see the same cells, genes, gene
order and model. Stages print "[stage-start] <name> <unix time>" and
"[stage] <name> elapsed=<s> end=<unix time>"; "[mark] baseline" follows the imports.
DUET's fit and test are one closed-form computation on the two-group path, so they
are one stage ("fit_test"); with --design cdr the general per-gene path runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"[stage-start] {self.name} {time.time():.6f}", flush=True)
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()                  # CPU time of this process (user + system)

    def __exit__(self, *exc):
        print(f"[stage] {self.name} elapsed={time.perf_counter() - self.t0:.3f} "
              f"cpu={time.process_time() - self.c0:.3f} end={time.time():.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("workdir")
    ap.add_argument("--design", choices=["condition", "cdr"], default="condition")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    import anndata as ad
    import scipy.sparse as sp
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    if not a.export:
        import duet.core as core                      # imported before the baseline mark
    print(f"[mark] baseline {time.time():.6f}", flush=True)

    with Stage("read"):
        A = ad.read_h5ad(a.h5ad)
        ref, test = A.uns["labels"]["ref"], A.uns["labels"]["test"]

    if a.export:
        from scipy.io import mmwrite
        with Stage("export"):
            X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
            mmwrite(str(wd / "expr.mtx"), X.T.tocoo(), field="real", precision=7)
            np.savetxt(wd / "genes.txt", np.asarray(A.var_names, dtype=str), fmt="%s")
            np.savetxt(wd / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
            np.savetxt(wd / "cond.txt", A.obs["condition"].astype(str).to_numpy(), fmt="%s")
            (wd / "labels.json").write_text(json.dumps({"ref": ref, "test": test}), encoding="utf-8")
        print("[done]", flush=True)
        return

    cond01 = (A.obs["condition"].astype(str).to_numpy() == test).astype(np.int8)
    if a.design == "condition":
        with Stage("fit_test"):
            df = core._fit_celltype_vectorized(
                A.X, A.var_names, cond01, cond01 == 0, cond01 == 1, comparison=f"{ref}_vs_{test}",
                # the mast_compat settings: min_positive=2, partial hurdle, EB moderation
                ref_label=ref, test_label=test, min_positive=2, partial_hurdle=True, eb_shrinkage=True,
                apply_log1p=False, celltype="bench")
            t = df["tested"].to_numpy(bool)
            df.loc[t, "fdr"] = core._bh_fdr(df.loc[t, "pvalue"].to_numpy())
        with Stage("write"):
            df.to_csv(wd / "duet_bench_results.csv", index=False)
    else:
        A.obs["celltype"] = "bench"
        with Stage("fit_test_write"):
            core.run_duet(A, output_dir=str(wd), celltype_col="celltype", sample_col="condition",
                          ref_label=ref, test_label=test, covariates=("CDR",), vectorized=False,
                          memory_log=False, output_name="duet_bench_results.csv")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
