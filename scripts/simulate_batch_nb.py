#!/usr/bin/env python
"""
simulate_batch_nb.py -- MO task 14: donor and batch effects, one at a time, by simulation.

A parametric negative-binomial simulator whose gene means, dispersions and library
sizes come from a real experiment (Crowell, Vehicle excitatory neurons), with the
donor effect switched on and off explicitly. Three scenarios, all GLOBAL NULLS
(no gene differs between the groups), 4 vs 4 donors x 3,000 cells, 4,000 genes:

  none         no donor or batch effect:   mu_gcd = mu_g * s_c
  donor        donor effect independent of the group:
               log mu_gd = log mu_g + b_gd,  b_gd ~ N(0, sigma_d^2); donors are
               assigned to groups at random, so the effect is not aligned with them
  batch_group  every group processed in its own batch, i.e. batch = group:
               log mu_gd = log mu_g + b_gd + beta_g * batch(d),  beta_g ~ N(0, sigma_b^2)

sigma_d is estimated from the reference (between-sample SD of log sample means, net
of sampling noise); sigma_b = sigma_d. Counts ~ NB(mean = mu_gcd, size = 1 / phi_g).

In the first two scenarios every call is a false positive, so the rejection share,
the raw type-I error and the share of replicates with at least one discovery are
error rates. In batch_group the group and batch effects are the same contrast and
cannot be separated by any method; no group effect is estimated there. Instead the
script shows that the calls follow the simulated batch shift (correlation of the
estimated log2FC with beta_g), whichever method makes them.

Replicates are written in the muscat-replicate format (evaluate_sim_replicates.build
reads them) to inputs/data/batchsim/<scenario>_repNN/, with seed.txt.

Usage: python scripts/simulate_batch_nb.py [--reps 20] [--evaluate] [--workers 2]
Outputs: results/batchsim/per_rep.csv, results/batchsim/summary.csv, results/batchsim/params.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.limit_threads(1)

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import os                                                  # noqa: E402
import warnings                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import scipy.sparse as sp                                  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
OUT_DATA = INPUTS / "data" / "batchsim"
OUT_RES = HERE / "results" / "batchsim"
SCENARIOS = ("none", "donor", "batch_group")
N_GENES, N_DONORS, CELLS_PER_DONOR = 4000, 8, 3000
ALPHA = 0.05
warnings.filterwarnings("ignore")


def estimate_params(seed: int = 0) -> dict:
    """Gene means, NB dispersions, library sizes and the donor SD from the reference."""
    A = _runtime.load_rows(INPUTS / "data" / "crowell_4vs4.h5ad",
                           lambda o: ((o["celltype"].astype(str) == "Excit. Neuron")
                                      & (o["Sample"].astype(str) == "Vehicle")).to_numpy())
    C = sp.csr_matrix(A.layers["counts"]).astype(np.float64)
    lib = np.asarray(C.sum(axis=1)).ravel()
    s = lib / np.median(lib)
    Cn = sp.diags(1.0 / s) @ C                                 # size-factor normalised counts
    mu = np.asarray(Cn.mean(axis=0)).ravel()
    ex2 = np.asarray(Cn.multiply(Cn).mean(axis=0)).ravel()
    var = ex2 - mu ** 2
    phi = np.clip((var - mu) / np.maximum(mu, 1e-12) ** 2, 1e-3, 20.0)
    ok = mu > 0.05                                             # genes worth simulating
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.flatnonzero(ok), N_GENES, replace=False))
    donors = A.obs["SourceFile"].astype(str).to_numpy()
    means, ns = [], []
    for d in np.unique(donors):
        m = donors == d
        means.append(np.asarray(Cn[m][:, idx].mean(axis=0)).ravel())
        ns.append(int(m.sum()))
    means, ns = np.vstack(means), np.array(ns)
    keep = (means > 0).all(axis=0) & (mu[idx] > 0.2)
    lm = np.log(means[:, keep])
    # sampling variance of a log sample mean ~ (1/mu + phi) / n_cells (delta method)
    samp = ((1.0 / mu[idx][keep] + phi[idx][keep])[None, :] / ns[:, None]).mean(axis=0)
    between = np.maximum(lm.var(axis=0, ddof=1) - samp, 0.0)
    sigma_d = float(np.sqrt(np.median(between)))
    return dict(mu=mu[idx].tolist(), phi=phi[idx].tolist(), genes=list(map(str, A.var_names[idx])),
                log_lib_mean=float(np.log(lib).mean()), log_lib_sd=float(np.log(lib).std()),
                median_lib=float(np.median(lib)), sigma_d=sigma_d, n_genes_for_sigma=int(keep.sum()))


def simulate(scenario: str, rep: int, P: dict) -> Path:
    dest = OUT_DATA / f"{scenario}_rep{rep:02d}"
    if (dest / "seed.txt").is_file():
        return dest
    seed = 1000 * (SCENARIOS.index(scenario) + 1) + rep
    rng = np.random.default_rng(seed)
    mu, phi = np.asarray(P["mu"]), np.asarray(P["phi"])
    G = len(mu)
    donors = [f"donor{i + 1}" for i in range(N_DONORS)]
    group = rng.permutation(np.repeat(["A", "B"], N_DONORS // 2))
    b = rng.normal(0.0, P["sigma_d"], (N_DONORS, G)) if scenario != "none" else np.zeros((N_DONORS, G))
    beta = rng.normal(0.0, P["sigma_d"], G) if scenario == "batch_group" else np.zeros(G)
    rows, cells, obs = [], [], []
    for i, d in enumerate(donors):
        n = CELLS_PER_DONOR
        lib = np.exp(rng.normal(P["log_lib_mean"], P["log_lib_sd"], n))
        s = lib / P["median_lib"]
        log_mu_d = np.log(mu) + b[i] + beta * (group[i] == "B")
        lam = s[:, None] * np.exp(log_mu_d)[None, :]
        size = 1.0 / phi
        X = rng.negative_binomial(size[None, :], size[None, :] / (size[None, :] + lam))
        rows.append(sp.csr_matrix(X.astype(np.int32)))
        cells += [f"{d}_c{j}" for j in range(n)]
        obs += [("cluster1", f"{d}.{group[i]}", group[i])] * n
    M = sp.vstack(rows).tocsr()
    dest.mkdir(parents=True, exist_ok=True)
    from scipy.io import mmwrite
    mmwrite(str(dest / "counts.mtx"), M.T.tocoo(), field="integer")          # genes x cells, like muscat
    pd.DataFrame(obs, columns=["cluster_id", "sample_id", "group_id"], index=cells).to_csv(dest / "coldata.tsv", sep="\t")
    genes = [f"gene{j + 1}" for j in range(G)]
    pd.DataFrame({"gene": genes, "category": "ee", "batch_shift_log": beta, "ref_gene": P["genes"]}).to_csv(
        dest / "gene_info.tsv", sep="\t", index=False)
    (dest / "genes.txt").write_text("\n".join(genes))
    (dest / "cells.txt").write_text("\n".join(cells))
    (dest / "seed.txt").write_text(str(seed))
    return dest


def evaluate(scenario: str, rep: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    _runtime.lower_priority()
    from scipy.stats import pearsonr
    from evaluate_sim_replicates import build
    from donor_aware_methods import python_arms
    import run_de_comparison as R
    src = OUT_DATA / f"{scenario}_rep{rep:02d}"
    A = build(src)
    A.obs["condition"] = A.obs["Sample"].astype(str)
    gi = pd.read_csv(src / "gene_info.tsv", sep="\t").set_index("gene")
    res = python_arms(A, "SourceFile", "A", "B")
    wi = R.run_wilcoxon(A, "cluster1", ref_label="A", test_label="B", n_ref=0, n_test=0).set_index("gene")
    res["wilcoxon"] = wi[["pvalue", "fdr", "log2fc"]]
    rows = []
    for m, d in res.items():
        d = d[d["fdr"].notna()]
        n_t, n_d = len(d), int((d["fdr"] < ALPHA).sum())
        raw = float((d["pvalue"] < ALPHA).mean()) if d["pvalue"].notna().any() else np.nan
        shift = gi["batch_shift_log"].reindex(d.index).to_numpy(float)
        r = (pearsonr(d["log2fc"].to_numpy(float), shift).statistic
             if scenario == "batch_group" and np.isfinite(d["log2fc"]).sum() > 2 else np.nan)
        rows.append(dict(scenario=scenario, replicate=rep, method=m, n_tested=n_t, n_discoveries=n_d,
                         rejection_pct=100 * n_d / max(n_t, 1), raw_type1=raw, any_discovery=n_d > 0,
                         r_log2fc_vs_batch_shift=r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    _runtime.lower_priority()
    OUT_RES.mkdir(parents=True, exist_ok=True)
    pfile = OUT_RES / "params.json"
    if pfile.is_file():
        P = json.loads(pfile.read_text())
    else:
        P = estimate_params()
        pfile.write_text(json.dumps(P))
    print(f"[batchsim] sigma_d = {P['sigma_d']:.3f} (from {P['n_genes_for_sigma']} reference genes), "
          f"median library {P['median_lib']:.0f}", flush=True)
    for sc in SCENARIOS:
        for r in range(1, a.reps + 1):
            simulate(sc, r, P)
    print(f"[batchsim] {len(SCENARIOS) * a.reps} replicates in {OUT_DATA}", flush=True)
    if not a.evaluate:
        return
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(evaluate, sc, r) for sc in SCENARIOS for r in range(1, a.reps + 1)]
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
            except Exception as e:
                print(f"[batchsim] FAIL {type(e).__name__}: {e}", flush=True)
    D = pd.DataFrame(rows).sort_values(["scenario", "method", "replicate"])
    D.to_csv(OUT_RES / "per_rep.csv", index=False)
    S = D.groupby(["scenario", "method"], sort=False).agg(
        n_reps=("replicate", "nunique"), rejection_pct_mean=("rejection_pct", "mean"),
        rejection_pct_max=("rejection_pct", "max"), raw_type1_mean=("raw_type1", "mean"),
        share_reps_with_any_discovery=("any_discovery", "mean"),
        r_log2fc_vs_batch_shift_mean=("r_log2fc_vs_batch_shift", "mean")).reset_index()
    S.to_csv(OUT_RES / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
