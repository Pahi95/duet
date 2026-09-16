#!/usr/bin/env python
"""
underflow_check.py -- does a p-value stored as 0.0 actually damage the
gene ranking in the packages we compare, and what does each package rank by?

For pancreas Ductal (corrected object) and Kang CD14+ monocytes, on one gene set
(DUET's tested genes detected in >= 10% of one group), per package:
  MAST 1.33.0  lrTest, default settings (bayesglm, ebayes) -- the run written by
               mast_settings_check.py; lambda (the hurdle LR statistic) is retained
  diffxpy      Wald test, called directly; coef_mle / coef_sd is retained
  Scanpy       rank_genes_groups(method="wilcoxon"); `scores` (a z) is retained
  DUET         neglog10p
For each: the share of p-values stored as exactly 0, the usable statistic kept
beside the p-value, and the order in which the package returns genes (checked, not
assumed). Rankings compared: by p-value (ties broken by input order, as a stable
sort does) against the retained statistic, over several random input gene orders;
top-20/100/500 overlaps and the stability of the p-value top lists across orders.

Scanpy, diffxpy and DUET are rerun on each permuted gene order. MAST fits each gene
independently, so permuting its output rows is equivalent; one real rerun on a
permuted order (Kang) checks that equivalence.

Outputs: evidence/underflow_by_package.csv, evidence/underflow_topk.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _runtime                                            # noqa: E402

_runtime.background(1)

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import math                                                # noqa: E402
import os                                                  # noqa: E402
import shutil                                              # noqa: E402
import subprocess                                          # noqa: E402
import warnings                                            # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
from scipy.stats import norm                               # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from donor_swap_partitions import DATASETS                 # noqa: E402
from mast_settings_check import TEST, run_duet_case        # noqa: E402

RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
CASES = [("kang", "CD14+ Monocytes"), ("pancreas", "Ductal cell")]
TOPK = (20, 100, 500)
LN10 = math.log(10.0)
warnings.filterwarnings("ignore")


def nlp_normal(z):
    return -(math.log(2.0) + norm.logsf(np.abs(np.asarray(z, float)))) / LN10


def wilcoxon(A, ref, test):
    import scanpy as sc
    import anndata as ad
    a = ad.AnnData(X=A.X, obs=pd.DataFrame({"condition": pd.Categorical(A.obs.cond, categories=[ref, test])},
                                           index=A.obs_names), var=pd.DataFrame(index=A.var_names))
    sc.tl.rank_genes_groups(a, "condition", groups=[test], reference=ref, method="wilcoxon")
    d = sc.get.rank_genes_groups_df(a, group=test)
    return pd.DataFrame({"gene": d["names"].astype(str), "pvalue": d["pvals"].astype(float),
                         "stat": d["scores"].astype(float), "retained": nlp_normal(d["scores"])})


def diffxpy(A, ref, test):
    import anndata as ad
    import run_de_comparison as R
    R._patch_dask_auto_chunks()
    import diffxpy.api as de
    C = A.layers["counts"]
    sub = ad.AnnData(X=np.ascontiguousarray(C.toarray(), dtype=np.float64),
                     obs=pd.DataFrame({"condition": pd.Categorical(A.obs.cond.to_numpy(), categories=[ref, test])}),
                     var=pd.DataFrame(index=A.var_names.astype(str)))
    s = de.test.wald(data=sub, formula_loc="~1+condition", factor_loc_totest="condition",
                     train_args={"nproc": 1}).summary()
    z = s["coef_mle"].astype(float) / s["coef_sd"].astype(float)
    return pd.DataFrame({"gene": s["gene"].astype(str), "pvalue": s["pval"].astype(float),
                         "stat": z, "retained": nlp_normal(z)})


def duet(A, ref, test):
    d = run_duet_case(A, ref, test).reset_index(drop=True)
    return pd.DataFrame({"gene": d["gene"].astype(str), "pvalue": d["pvalue"], "stat": d["stat_hurdle"],
                         "retained": d["neglog10p"]})


def mast_from_file(dataset):
    f = HERE / "results" / "mast_components" / dataset / "mast_bayesglm_ebTRUE.csv"
    if not f.is_file():
        raise FileNotFoundError(f"{f} -- run mast_settings_check.py first")
    from duet.core import _neglog10_chi2_sf
    m = pd.read_csv(f)
    return pd.DataFrame({"gene": m["gene"].astype(str), "pvalue": m["pvalue"], "stat": m["stat_hurdle"],
                         "retained": _neglog10_chi2_sf(m["stat_hurdle"].to_numpy(float),
                                                       m["df_hurdle"].to_numpy(float))})


def mast_permuted_rerun(A, perm_genes, m_orig, ref, test) -> dict:
    """Run MAST (default settings) on a permuted gene order and compare gene by gene."""
    import tempfile
    import scipy.sparse as sp
    from scipy.io import mmwrite
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
        mmwrite(str(d / "expr.mtx"), X[:, pd.Index(A.var_names.astype(str)).get_indexer(perm_genes)].T.tocoo(),
                field="real", precision=7)
        np.savetxt(d / "genes.txt", np.asarray(perm_genes), fmt="%s")
        np.savetxt(d / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
        np.savetxt(d / "cond.txt", A.obs.cond.to_numpy(), fmt="%s")
        (d / "labels.json").write_text(json.dumps({"ref": ref, "test": test}), encoding="utf-8")
        r = subprocess.run([RSCRIPT, str(HERE / "scripts" / "mast_components.R"), str(d), "bayesglm", "TRUE"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return dict(dataset="kang", package="MAST (permuted rerun)", note=f"FAILED: {r.stderr[-300:]}")
        p = pd.read_csv(d / "mast_bayesglm_ebTRUE.csv").set_index("gene")
    o = m_orig.set_index("gene").reindex(p.index)
    return dict(dataset="kang", package="MAST (permuted rerun)", n_genes=len(p),
                note=(f"max |d lambda| = {float((p['stat_hurdle'] - o['stat']).abs().max()):.2e}, "
                      f"max |d p| = {float((p['pvalue'] - o['pvalue']).abs().max()):.2e}; "
                      f"output order = input order: {list(p.index) == list(perm_genes)}"))


def topk_by_p(d: pd.DataFrame, k: int) -> list[str]:
    return d.sort_values("pvalue", kind="stable")["gene"].head(k).tolist()


def topk_by_stat(d: pd.DataFrame, k: int) -> list[str]:
    return d.sort_values("retained", ascending=False, kind="stable")["gene"].head(k).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=5)
    ap.add_argument("--diffxpy-perms", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pkg_rows, top_rows = [], []
    for name, ct in CASES:
        cfg = DATASETS[name]
        ref, test = cfg["ref"], TEST[name]
        A0 = _runtime.load_rows(cfg["h5ad"], lambda o: (o["celltype"].astype(str) == ct).to_numpy())
        A0.obs["cond"] = A0.obs[cfg["cond"]].astype(str)
        A0 = A0[A0.obs.cond.isin([ref, test])].copy()
        genes = duet(A0, ref, test)["gene"].tolist()              # the common gene set
        A0 = A0[:, genes].copy()
        runners = {"DUET": duet, "Scanpy wilcoxon": wilcoxon, "diffxpy": diffxpy}
        orders = [np.arange(len(genes))] + [rng.permutation(len(genes)) for _ in range(a.perms)]
        results = {}
        for pkg, fn in runners.items():
            n_orders = (a.diffxpy_perms + 1) if pkg == "diffxpy" else len(orders)
            results[pkg] = []
            for o in orders[:n_orders]:
                A = A0[:, o].copy()
                d = fn(A, ref, test)
                d["input_rank"] = pd.Index(A.var_names.astype(str)).get_indexer(d["gene"])
                results[pkg].append(d)
                print(f"[{name}] {pkg}: order {len(results[pkg])}/{n_orders} done", flush=True)
        m = mast_from_file(name)
        m = m[m.gene.isin(genes)]
        results["MAST"] = []
        for o in orders:
            order_genes = [genes[i] for i in o]
            mm = m.set_index("gene").reindex(order_genes).dropna(subset=["pvalue"]).reset_index()
            mm["input_rank"] = np.arange(len(mm))
            results["MAST"].append(mm)
        for pkg, runs in results.items():
            d0 = runs[0]
            # how does the package itself order its output?
            if (np.diff(d0["input_rank"].to_numpy()) > 0).all():
                default_order = "input gene order"
            elif (np.diff(d0["stat"].to_numpy()) <= 1e-12).all():
                default_order = "descending retained statistic"
            else:
                default_order = "other"
            zero = d0["pvalue"] == 0
            pkg_rows.append(dict(dataset=name, celltype=ct, package=pkg, n_genes=len(d0),
                                 n_pvalue_zero=int(zero.sum()), pct_pvalue_zero=100 * zero.mean(),
                                 retained_statistic={"DUET": "hurdle chi2 (neglog10p)", "MAST": "lrTest lambda",
                                                     "Scanpy wilcoxon": "scores (z)",
                                                     "diffxpy": "coef_mle / coef_sd (Wald z)"}[pkg],
                                 retained_finite_where_p0=bool(np.isfinite(d0.loc[zero, "retained"]).all()),
                                 retained_range_where_p0=(f"{d0.loc[zero, 'retained'].min():.0f}-"
                                                          f"{d0.loc[zero, 'retained'].max():.0f}") if zero.any() else "",
                                 default_output_order=default_order, n_input_orders=len(runs)))
            for k in TOPK:
                ov = [len(set(topk_by_p(d, k)) & set(topk_by_stat(d, k))) for d in runs]
                ps = [set(topk_by_p(d, k)) for d in runs]
                jac = [len(x & y) / len(x | y) for x, y in zip(ps[:-1], ps[1:])] if len(ps) > 1 else [1.0]
                top_rows.append(dict(dataset=name, package=pkg, k=k, overlap_p_vs_statistic_min=min(ov),
                                     overlap_p_vs_statistic_median=float(np.median(ov)),
                                     overlap_p_vs_statistic_max=max(ov),
                                     p_ranked_topk_jaccard_across_orders_min=min(jac),
                                     statistic_ranked_topk_identical_across_orders=all(
                                         topk_by_stat(d, k) == topk_by_stat(runs[0], k) for d in runs)))
        if name == "kang":                                     # MAST: one real rerun on a permuted gene order
            pkg_rows.append(mast_permuted_rerun(A0, [genes[i] for i in orders[1]], m, ref, test))
    pd.DataFrame(pkg_rows).to_csv(HERE / "evidence" / "underflow_by_package.csv", index=False)
    pd.DataFrame(top_rows).to_csv(HERE / "evidence" / "underflow_topk.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(pkg_rows).to_string(index=False))
    print(pd.DataFrame(top_rows).to_string(index=False))


if __name__ == "__main__":
    main()
