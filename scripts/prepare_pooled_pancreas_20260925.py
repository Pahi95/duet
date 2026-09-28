#!/usr/bin/env python
"""Append five pooled pancreas benchmark inputs to the 2026-09-25 manifest.

Pooled means Ductal + Endothelial + Stellate cells, with stratified Reference/
pancreas sampling. This only prepares inputs; it never runs an analysis.
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
from anndata.io import read_elem, sparse_dataset  # noqa: E402

from prepare_pancreas_20260925 import (  # noqa: E402
    REF, ROOT, SOURCE, TEST, gene_gate, sha256, validate_source, write_atomic,
)


CELLTYPES = ("Ductal cell", "Endothelial cell", "Stellate cell")
SIZES = (2500, 5000, 10000, 15000, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest_path = args.out / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare ductal inputs first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(args.source.resolve()) != manifest["source"]:
        raise ValueError("pooled and ductal inputs must use the same source path")
    if sha256(args.source) != manifest["source_sha256"]:
        raise ValueError("source h5ad changed since ductal input preparation")
    names = [f"pooled_{n if n is not None else 'all'}" for n in SIZES]
    existing_names = {item["name"] for item in manifest["inputs"]}
    if any(name in existing_names or (args.out / name).exists() for name in names):
        raise FileExistsError("a pooled input already exists; inspect it before retrying")

    with h5py.File(args.source, "r") as h5:
        all_obs = read_elem(h5["obs"])
        selected = np.flatnonzero(all_obs["celltype"].astype(str).isin(CELLTYPES).to_numpy())
        obs = all_obs.iloc[selected].copy()
        var = read_elem(h5["var"])
        x = sparse_dataset(h5["X"])[selected].tocsr()
        counts = sparse_dataset(h5["layers"]["counts"])[selected].tocsr()
    validate_source(obs, x, counts, label="pooled")
    condition = obs["Sample"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    source_hash = manifest["source_sha256"]
    for n, name in zip(SIZES, names):
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
            "celltypes": list(CELLTYPES),
            "zero_fraction": float(1 - normalized.X.nnz / np.prod(normalized.shape)),
        }
        raw = ad.AnnData(
            X=counts[chosen][:, keep].astype(np.int32),
            obs=common_obs,
            var=pd.DataFrame(index=genes),
        )
        raw.uns["labels"] = {"ref": REF, "test": TEST}
        raw.uns["meta"] = {"source_sha256": source_hash, "celltypes": list(CELLTYPES)}
        dest = args.out / name
        dest.mkdir()
        write_atomic(normalized, dest / "data.h5ad")
        write_atomic(raw, dest / "counts.h5ad")
        info = {
            "name": name, "celltypes": list(CELLTYPES),
            "cells": int(len(chosen)), "genes": int(len(genes)),
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
    manifest.pop("celltype", None)
    manifest["celltypes_by_series"] = {
        "ductal": ["Ductal cell"], "pooled": list(CELLTYPES),
    }
    tmp = manifest_path.with_name("manifest.json.part")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)


if __name__ == "__main__":
    main()
