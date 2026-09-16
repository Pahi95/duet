#!/usr/bin/env python
"""
bench_prepare.py -- write the benchmark inputs once, so every timed run starts from
the same file: the same cells, genes, gene order and conditions for both engines
. Output: inputs/bench/<name>/data.h5ad, X = log1p(CP10k) CSR,
obs.condition, uns.labels, uns.meta.

Real ladder, from the corrected pancreas object (Reference vs pancreas):
  ductal_<n>   Ductal cells, n in 1000, 2500, 5000 and all; cells subsampled with
               seed 0, keeping the two groups' proportions
  pooled_<n>   Ductal + Endothelial + Stellate cells, n in 2500, 5000, 10000, 15000, all
  genes        DUET's mast_compat gate on that subset (>= 1 positive cell per group,
               >= 20 cells) and detected in >= 10% of the cells of at least one group

Synthetic grid, one factor at a time, negative-binomial counts with gene
(mean, dispersion) pairs resampled from the Crowell reference (results/batchsim/params.json),
two equal groups, 5% of genes with a two-fold mean change:
  cells_<n>_g8000         n in 1000, 2500, 5000, 10000, 25000, 50000 -- the gene list fixed
  genes_<g>_c10000        g in 2000, 8000, 20000
  sparsity_<s>_c10000     library size x4 / x1 / x0.25 -> low / mid / high zero fraction
  (the ~CDR + condition model is a run-time option of bench_harness.py, not separate data)

Usage: python scripts/bench_prepare.py [--real] [--synthetic]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import os                                                  # noqa: E402
import zlib                                                # noqa: E402

import anndata as ad                                       # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
OUT = INPUTS / "bench"
REF, TEST = "Reference", "pancreas"


def done(name: str) -> bool:
    """Already written: rerunning the preparation skips it (stop/resume safe; writes are atomic)."""
    return (OUT / name / "data.h5ad").is_file()


def write(name: str, X, obs: pd.DataFrame, var_names, meta: dict) -> None:
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    A = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(var_names).astype(str)))
    A.uns["labels"] = {"ref": meta["ref"], "test": meta["test"]}
    A.uns["meta"] = {k: v for k, v in meta.items() if k not in ("ref", "test")}
    tmp = dest / "data.h5ad.part"
    A.write_h5ad(tmp)
    os.replace(tmp, dest / "data.h5ad")
    print(f"[prepare] {name:24s} {A.n_obs:6d} cells x {A.n_vars:6d} genes  zero fraction "
          f"{meta.get('zero_fraction', float('nan')):.3f}", flush=True)


def gene_gate(X, cond):
    X = X.tocsc()
    pos = X > 0
    n_ref, n_test = int((cond == 0).sum()), int((cond == 1).sum())
    c_ref = np.asarray(pos[cond == 0].sum(axis=0)).ravel()
    c_test = np.asarray(pos[cond == 1].sum(axis=0)).ravel()
    return (c_ref >= 1) & (c_test >= 1) & (np.maximum(c_ref / n_ref, c_test / n_test) >= 0.10)


def real_ladder():
    h5 = INPUTS / "PancreasCorrected.h5ad"
    for label, cts, sizes in (("ductal", ["Ductal cell"], [1000, 2500, 5000, None]),
                              ("pooled", ["Ductal cell", "Endothelial cell", "Stellate cell"],
                               [2500, 5000, 10000, 15000, None])):
        A = _runtime.load_rows(h5, lambda o: o["celltype"].astype(str).isin(cts).to_numpy(), layers=())
        cond_all = (A.obs["Sample"].astype(str).to_numpy() == TEST).astype(int)
        rng = np.random.default_rng(0)
        for n in sizes:
            if done(f"{label}_{'all' if n is None else n}"):
                print(f"[prepare] {label}_{'all' if n is None else n} exists, skipped", flush=True)
                continue
            if n is None or n >= A.n_obs:
                idx = np.arange(A.n_obs)
            else:
                idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(cond_all == g),
                                                         int(round(n * (cond_all == g).mean())), replace=False)
                                              for g in (0, 1)]))
            X = A.X[idx]
            cond = cond_all[idx]
            keep = gene_gate(X, cond)
            X = X[:, keep]
            obs = pd.DataFrame({"condition": np.where(cond == 1, TEST, REF)}, index=A.obs_names[idx])
            zf = 1 - X.nnz / (X.shape[0] * X.shape[1])
            write(f"{label}_{'all' if n is None else n}", X, obs, A.var_names[keep],
                  dict(ref=REF, test=TEST, source="PancreasCorrected.h5ad", celltypes=cts, zero_fraction=zf))


def nb_matrix(n_cells, mu, phi, lib_scale, rng, chunk=5000):
    """Two equal groups; 5% of genes two-fold up in the second. Returns log1p(CP10k) CSR."""
    G = len(mu)
    de = np.zeros(G, bool)
    de[rng.choice(G, max(1, G // 20), replace=False)] = True
    cond = np.repeat([0, 1], [n_cells // 2, n_cells - n_cells // 2])
    blocks = []
    for s in range(0, n_cells, chunk):
        c = cond[s:s + chunk]
        lib = np.exp(rng.normal(0.0, 0.5, len(c))) * lib_scale
        lam = lib[:, None] * mu[None, :] * np.where(de[None, :] & (c[:, None] == 1), 2.0, 1.0)
        size = 1.0 / phi
        blocks.append(sp.csr_matrix(rng.negative_binomial(size[None, :], size[None, :] / (size[None, :] + lam))
                                    .astype(np.float32)))
    C = sp.vstack(blocks).tocsr()
    tot = np.asarray(C.sum(axis=1)).ravel()
    X = sp.diags(1e4 / np.maximum(tot, 1)) @ C
    X.data = np.log1p(X.data)
    zf = 1 - C.nnz / (C.shape[0] * C.shape[1])
    return X.tocsr(), cond, zf


def synthetic_grid():
    P = json.loads((HERE / "results" / "batchsim" / "params.json").read_text())
    mu0, phi0 = np.asarray(P["mu"]), np.asarray(P["phi"])
    specs = [(f"cells_{n}_g8000", n, 8000, 1.0) for n in (1000, 2500, 5000, 10000, 25000, 50000)]
    specs += [(f"genes_{g}_c10000", 10000, g, 1.0) for g in (2000, 8000, 20000)]
    specs += [(f"sparsity_{lab}_c10000", 10000, 8000, s) for lab, s in (("low", 4.0), ("mid", 1.0), ("high", 0.25))]
    for name, n, g, scale in specs:
        if done(name):
            print(f"[prepare] {name} exists, skipped", flush=True)
            continue
        rng = np.random.default_rng(zlib.crc32(name.encode()))       # deterministic, unlike hash()
        pick = np.random.default_rng(g).choice(len(mu0), g, replace=True)    # same gene list for a given g
        X, cond, zf = nb_matrix(n, mu0[pick], phi0[pick], scale, rng)
        obs = pd.DataFrame({"condition": np.where(cond == 1, "B", "A")}, index=[f"c{i}" for i in range(n)])
        write(name, X, obs, [f"g{j}" for j in range(g)],
              dict(ref="A", test="B", source="NB grid, Crowell-derived gene parameters", library_scale=scale,
                   zero_fraction=zf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if not (a.real or a.synthetic):
        a.real = a.synthetic = True
    if a.real:
        real_ladder()
    if a.synthetic:
        synthetic_grid()


if __name__ == "__main__":
    main()
