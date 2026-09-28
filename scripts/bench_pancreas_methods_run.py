#!/usr/bin/env python
"""One timed diffxpy, PyDESeq2, or Wilcoxon run on a prepared ductal input."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_duet_run import Stage  # noqa: E402
import run_de_comparison as methods  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True, choices=("diffxpy", "deseq2", "wilcoxon"))
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.engine == "diffxpy":
        import diffxpy.api  # noqa: F401
        import dask

        dask.config.set(scheduler="synchronous", num_workers=1)
    elif args.engine == "deseq2":
        from pydeseq2.dds import DeseqDataSet  # noqa: F401
        from pydeseq2.ds import DeseqStats  # noqa: F401
    else:
        import scanpy  # noqa: F401
    print(f"[mark] baseline {time.time():.6f}", flush=True)
    with Stage("read"):
        obj = ad.read_h5ad(args.input)
        labels = obj.uns.get("labels", {"ref": "Reference", "test": "pancreas"})
        ref, test = labels["ref"], labels["test"]
        cond = obj.obs["condition"].astype(str).to_numpy()
        n_ref, n_test = int((cond == ref).sum()), int((cond == test).sum())
        if n_ref + n_test != obj.n_obs or min(n_ref, n_test) == 0:
            raise ValueError("invalid condition labels in benchmark input")
        if args.engine != "wilcoxon":
            if "DonorID" not in obj.obs:
                raise ValueError("raw-count input has no DonorID")
            if not np.array_equal(obj.X.data, np.rint(obj.X.data)):
                raise ValueError("raw-count input is not integer-valued")
            obj.layers["counts"] = obj.X
    kw = dict(ref_label=ref, test_label=test, n_ref=n_ref, n_test=n_test)
    with Stage("fit_test"):
        if args.engine == "diffxpy":
            result = methods.run_diffxpy(
                obj, "bench", counts_layer="counts", condition=cond, min_cells=1,
                max_cells_per_group=0, max_genes=0, nproc=1, seed=0, **kw,
            )
        elif args.engine == "deseq2":
            result = methods.run_deseq2(
                obj, "bench", counts_layer="counts", sample_col="DonorID",
                condition=cond, min_samples_per_group=2, min_gene_counts=10,
                min_cells_per_sample=10, **kw,
            )
        else:
            result = methods.run_wilcoxon(obj, "bench", **kw)
    if len(result) <= 1 or not result["tested"].all() or not result["gene"].is_unique:
        raise ValueError("incomplete or invalid method output")
    if not set(result["gene"]).issubset(set(obj.var_names)):
        raise ValueError("output genes do not match the input")
    if args.engine in ("diffxpy", "wilcoxon") and len(result) != obj.n_vars:
        raise ValueError("method did not return all input genes")
    info = {
        "engine": args.engine, "n_cells": obj.n_obs, "n_genes_input": obj.n_vars,
        "n_genes_output": len(result), "n_ref": n_ref, "n_test": n_test,
        "n_pvalue_finite": int(np.isfinite(result.pvalue).sum()),
        "source_sha256": obj.uns.get("meta", {}).get("source_sha256", ""),
    }
    if args.engine == "deseq2":
        sizes = obj.obs.groupby(["DonorID", "condition"], observed=True).size()
        accepted = sizes[sizes >= 10]
        info.update(
            n_donors=int(obj.obs["DonorID"].nunique()),
            n_pseudobulk_samples=len(accepted), n_cells_aggregated=int(accepted.sum()),
            aggregation="DonorID x condition", min_cells_per_sample=10,
            min_gene_counts=10,
        )
    if info["n_pvalue_finite"] == 0:
        raise ValueError("no finite p-values")
    print("[info] " + json.dumps(info), flush=True)
    with Stage("write"):
        result.to_csv(args.out, index=False)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
