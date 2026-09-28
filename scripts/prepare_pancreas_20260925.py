#!/usr/bin/env python
"""Prepare the four versioned ductal inputs from the 2026-09-25 pancreas object.

The original object and the manuscript's previous benchmark inputs are never changed.
Each output has a normalized data.h5ad for DUET/MAST and a counts.h5ad for
count-based methods. Run this step before any analysis in pancreas_20260925.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime  # noqa: E402

_runtime.limit_threads(1)

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from anndata.io import read_elem, sparse_dataset  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
# The integrated pancreas object (469,979 cells) produced by the Scanpy pipeline described in the manuscript.
SOURCE = Path(os.environ.get("DUET_PANCREAS_SOURCE",
                             HERE.parent / "inputs" / "PancreasIntegratedAnnotated_20260925.h5ad"))
ROOT = HERE.parent / "inputs" / "bench" / "pancreas_20260925"
SIZES = (1000, 2500, 5000, None)
REF, TEST = "Reference", "pancreas"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(obs: pd.DataFrame, x: sp.csr_matrix, counts: sp.csr_matrix,
                    label: str = "ductal") -> None:
    required = {"Sample", "DonorID", "SourceFile", "celltype", "total_counts", "n_genes_by_counts"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"missing obs columns: {sorted(missing)}")
    if not obs.index.is_unique:
        raise ValueError("ductal cell identifiers are not unique")
    labels = set(obs["Sample"].astype(str))
    if labels != {REF, TEST}:
        raise ValueError(f"unexpected conditions: {labels}")
    if obs.groupby("DonorID", observed=True)["Sample"].nunique().max() != 1:
        raise ValueError("a donor appears in both conditions; check donor mapping")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("counts contain non-finite or negative values")
    if not np.array_equal(counts.data, np.rint(counts.data)):
        raise ValueError("counts are not integer-valued")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if not np.array_equal(totals, obs["total_counts"].to_numpy()):
        raise ValueError("counts row sums differ from obs.total_counts")
    if not np.array_equal(np.diff(counts.indptr), obs["n_genes_by_counts"].to_numpy()):
        raise ValueError("count nonzeros differ from obs.n_genes_by_counts")
    if (totals <= 0).any():
        raise ValueError("zero-count ductal cell")
    if not np.array_equal(x.indptr, counts.indptr) or not np.array_equal(x.indices, counts.indices):
        raise ValueError("X and counts have different sparsity patterns")
    max_err = 0.0
    for start in range(0, x.shape[0], 1000):
        end = min(start + 1000, x.shape[0])
        lo, hi = x.indptr[start], x.indptr[end]
        scale = np.repeat(1e4 / totals[start:end], np.diff(x.indptr[start : end + 1]))
        predicted = np.log1p(counts.data[lo:hi].astype(np.float64) * scale)
        if len(predicted):
            max_err = max(max_err, float(np.max(np.abs(x.data[lo:hi] - predicted))))
    if max_err > 1e-5:
        raise ValueError(f"X is not log1p(CP10k) from counts: max error {max_err:.3g}")
    print(f"[prepare] validated {len(obs)} {label} cells; max normalization error {max_err:.3g}", flush=True)


def gene_gate(x: sp.csr_matrix, condition: np.ndarray) -> np.ndarray:
    positive = x > 0
    ref = condition == REF
    test = condition == TEST
    n_ref = np.asarray(positive[ref].sum(axis=0)).ravel()
    n_test = np.asarray(positive[test].sum(axis=0)).ravel()
    return (n_ref >= 1) & (n_test >= 1) & (
        np.maximum(n_ref / ref.sum(), n_test / test.sum()) >= 0.10
    )


def write_atomic(obj: ad.AnnData, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    if part.exists():
        raise FileExistsError(f"partial output exists; inspect it before retrying: {part}")
    obj.write_h5ad(part, compression="lzf")
    os.replace(part, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output folder is not empty: {args.out}")

    with h5py.File(args.source, "r") as h5:
        all_obs = read_elem(h5["obs"])
        selected = np.flatnonzero(all_obs["celltype"].astype(str).to_numpy() == "Ductal cell")
        obs = all_obs.iloc[selected].copy()
        var = read_elem(h5["var"])
        x = sparse_dataset(h5["X"])[selected].tocsr()
        counts = sparse_dataset(h5["layers"]["counts"])[selected].tocsr()
    validate_source(obs, x, counts)

    condition = obs["Sample"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    source_hash = sha256(args.source)
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "source_shape": list(all_obs.shape[:1]) + [len(var)],
        "celltype": "Ductal cell",
        "seed": args.seed,
        "ref": REF,
        "test": TEST,
        "gene_gate": "detected in both groups and in >=10% of at least one group",
        "samples_for_pseudobulk": "DonorID x Sample",
        "inputs": [],
    }
    args.out.mkdir(parents=True)
    for n in SIZES:
        name = f"ductal_{n if n is not None else 'all'}"
        if n is None or n >= len(obs):
            chosen = np.arange(len(obs))
        else:
            n_ref = int(round(n * np.mean(condition == REF)))
            n_test = n - n_ref
            chosen = np.sort(np.concatenate([
                rng.choice(np.flatnonzero(condition == REF), n_ref, replace=False),
                rng.choice(np.flatnonzero(condition == TEST), n_test, replace=False),
            ]))
        keep = gene_gate(x[chosen], condition[chosen])
        genes = var.index[keep].astype(str)
        common_obs = pd.DataFrame({
            "condition": condition[chosen],
            "DonorID": obs.iloc[chosen]["DonorID"].astype(str).to_numpy(),
            "SourceFile": obs.iloc[chosen]["SourceFile"].astype(str).to_numpy(),
            "total_counts_full": obs.iloc[chosen]["total_counts"].to_numpy(np.float64),
        }, index=obs.index[chosen])
        normalized = ad.AnnData(
            X=x[chosen][:, keep].astype(np.float32),
            obs=common_obs[["condition"]].copy(),
            var=pd.DataFrame(index=genes),
        )
        normalized.uns["labels"] = {"ref": REF, "test": TEST}
        normalized.uns["meta"] = {
            "source_sha256": source_hash,
            "celltype": "Ductal cell",
            "zero_fraction": float(1 - normalized.X.nnz / np.prod(normalized.shape)),
        }
        raw = ad.AnnData(
            X=counts[chosen][:, keep].astype(np.int32),
            obs=common_obs,
            var=pd.DataFrame(index=genes),
        )
        raw.uns["labels"] = {"ref": REF, "test": TEST}
        raw.uns["meta"] = {"source_sha256": source_hash, "celltype": "Ductal cell"}
        dest = args.out / name
        dest.mkdir()
        write_atomic(normalized, dest / "data.h5ad")
        write_atomic(raw, dest / "counts.h5ad")
        info = {
            "name": name,
            "cells": int(len(chosen)),
            "genes": int(len(genes)),
            "n_ref": int(np.sum(condition[chosen] == REF)),
            "n_test": int(np.sum(condition[chosen] == TEST)),
            "ref_donors": int(common_obs.loc[common_obs.condition == REF, "DonorID"].nunique()),
            "test_donors": int(common_obs.loc[common_obs.condition == TEST, "DonorID"].nunique()),
            "cell_ids_sha256": hashlib.sha256("\n".join(map(str, common_obs.index)).encode()).hexdigest(),
            "gene_ids_sha256": hashlib.sha256("\n".join(map(str, genes)).encode()).hexdigest(),
            "normalized": str((dest / "data.h5ad").resolve()),
            "counts": str((dest / "counts.h5ad").resolve()),
        }
        manifest["inputs"].append(info)
        print(f"[prepare] {name}: {info['cells']} cells, {info['genes']} genes, "
              f"ref={info['n_ref']}, test={info['n_test']}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
