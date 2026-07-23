#!/usr/bin/env python
"""
evaluate_sim_replicates.py — score every simulation replicate and report the
spread, so the ground-truth numbers are means with variability rather than one draw.

For each data/sim_rep*/ directory: assemble the AnnData, run the methods, join to
the muscat truth labels and compute AUC, power and the OBSERVED false discovery
rate. diffxpy is skipped by default — it costs ~4.5 min per replicate and is not
one of the methods the paper's claims rest on.

Scoring matches evaluate_sim.py: positives are de/dp/dm/db, negatives are ee, and
`ep` (same mean, different proportion) is excluded from both classes because it is
a null for a mean-shift test but a real distributional difference a hurdle may
legitimately detect.

Usage: python evaluate_sim_replicates.py [--methods duet,deseq2,wilcoxon] [--mast]
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.io import mmread, mmwrite
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent.parent   # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
import run_de_comparison as R          # noqa: E402

TRUE_DE = {"de", "dp", "dm", "db"}

# Resolve Rscript from PATH so this runs off this machine; RSCRIPT=... overrides for
# a non-standard install. The Windows default is only a last resort.
RSCRIPT = (os.environ.get("RSCRIPT")
           or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")


def build(repdir: Path) -> ad.AnnData:
    X = sp.csr_matrix(mmread(str(repdir / "counts.mtx")).T)
    genes = [l.strip() for l in (repdir / "genes.txt").read_text(encoding="utf-8").splitlines()]
    cells = [l.strip() for l in (repdir / "cells.txt").read_text(encoding="utf-8").splitlines()]
    cd = pd.read_csv(repdir / "coldata.tsv", sep="\t", index_col=0, low_memory=False)
    gi = pd.read_csv(repdir / "gene_info.tsv", sep="\t", low_memory=False)
    cd.index = pd.Index(cells)
    obs = pd.DataFrame(index=cd.index)
    obs["celltype"] = cd["cluster_id"].astype(str)
    obs["Sample"] = cd["group_id"].astype(str)
    obs["SourceFile"] = cd["sample_id"].astype(str)
    gcol = "gene" if "gene" in gi.columns else gi.columns[0]
    gi = gi.drop_duplicates(subset=[gcol]).set_index(gcol)
    var = pd.DataFrame(index=pd.Index(genes))
    var["category"] = gi["category"].reindex(var.index).astype(str).values
    A = ad.AnnData(X=X, obs=obs, var=var)
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    return A


def score(res: dict, truth: pd.DataFrame, alpha=0.05) -> list[dict]:
    common = set.intersection(*(set(d.gene) for d in res.values())) & set(truth.index)
    t = truth.loc[sorted(common)]
    y = t["is_de"].to_numpy().astype(int)
    out = []
    for m, d in res.items():
        d = d[d.gene.isin(common)].drop_duplicates("gene").set_index("gene").reindex(t.index)
        s = (d["neglog10p"] if "neglog10p" in d and d["neglog10p"].notna().any()
             else -np.log10(d["pvalue"].clip(1e-300))).to_numpy(float)
        s = np.nan_to_num(s)
        called = np.nan_to_num(d["fdr"].to_numpy(float) < alpha, nan=False).astype(bool)
        tp = int((called & (y == 1)).sum()); fp = int((called & (y == 0)).sum())
        fn = int((~called & (y == 1)).sum())
        out.append(dict(method=m, n_scored=len(y), auc=round(float(roc_auc_score(y, s)), 4),
                        power=round(tp / max(tp + fn, 1), 4),
                        observed_fdr=round(fp / max(tp + fp, 1), 4), tp=tp, fp=fp, fn=fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="duet,deseq2,wilcoxon")
    ap.add_argument("--mast", action="store_true", default=True)
    ap.add_argument("--no-mast", dest="mast", action="store_false")
    ap.add_argument("--outdir", default=str(HERE / "results_sim_reps"))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]

    reps = sorted(INPUTS.glob("data/sim_rep*"))
    if not reps:
        raise SystemExit("no data/sim_rep*/ directories — run simulate_replicates.R first")
    print(f"[eval] {len(reps)} replicates: {[r.name for r in reps]}")

    rows = []
    for rep in reps:
        tag = rep.name
        print(f"\n=== {tag} ===", flush=True)
        A = build(rep)
        truth = A.var[["category"]].copy()
        truth["is_de"] = truth.category.isin(TRUE_DE)
        truth = truth[truth.is_de | (truth.category == "ee")]
        A.obs["condition"] = A.obs["Sample"].astype(str)
        n_ref = int((A.obs.Sample == "A").sum()); n_test = int((A.obs.Sample == "B").sum())

        res = {}
        t0 = time.time()
        if "duet" in methods:
            res["DUET"] = R.run_duet(A, "cluster1", sample_col="Sample", ref_label="A",
                                     test_label="B", n_ref=n_ref, n_test=n_test,
                                     min_detect_frac=0.0)
        if "wilcoxon" in methods:
            res["wilcoxon"] = R.run_wilcoxon(A, "cluster1", ref_label="A", test_label="B",
                                             n_ref=n_ref, n_test=n_test)
        if "deseq2" in methods:
            res["DESeq2"] = R.run_deseq2(A, "cluster1", counts_layer="counts",
                                         sample_col="SourceFile",
                                         condition=A.obs.Sample.astype(str).to_numpy(),
                                         ref_label="A", test_label="B",
                                         n_ref=n_ref, n_test=n_test,
                                         min_samples_per_group=2, min_gene_counts=10,
                                         min_cells_per_sample=10)
        res = {k: v[v.tested == True].dropna(subset=["pvalue"]) for k, v in res.items()}  # noqa: E712
        print(f"[eval]   python methods in {time.time()-t0:.0f}s", flush=True)

        if a.mast:
            d = out / tag; d.mkdir(exist_ok=True)
            genes = res["DUET"].gene.astype(str).to_numpy()
            X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
            gi = pd.Index(A.var_names).get_indexer(genes)
            mmwrite(str(d / "expr.mtx"), X[:, gi].T.tocoo(), field="real", precision=7)
            np.savetxt(d / "genes.txt", genes, fmt="%s")
            np.savetxt(d / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
            np.savetxt(d / "cond.txt", A.obs.Sample.astype(str).to_numpy(), fmt="%s")
            (d / "labels.json").write_text('{"ref": "A", "test": "B"}', encoding="utf-8")
            t0 = time.time()
            r = subprocess.run([RSCRIPT, str(HERE / "bench_mast_run.R"), str(d), "1"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                m = pd.read_csv(d / "mast_results.csv")
                res["MAST"] = pd.DataFrame({
                    "gene": m.gene.astype(str), "pvalue": m.pvalue, "fdr": m.fdr,
                    "neglog10p": -np.log10(m.pvalue.clip(1e-300))})
                print(f"[eval]   MAST in {time.time()-t0:.0f}s", flush=True)
            else:
                print(f"[eval]   MAST FAILED: {r.stderr[-400:]}", flush=True)

        for row in score(res, truth):
            row["replicate"] = tag
            rows.append(row)
            print(f"    {row['method']:9s} AUC={row['auc']:.4f} power={row['power']:.3f} "
                  f"obsFDR={row['observed_fdr']:.3f}", flush=True)
        del A

    D = pd.DataFrame(rows)
    D.to_csv(out / "sim_replicates.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print(f"GROUND TRUTH OVER {len(reps)} SIMULATION REPLICATES")
    print("=" * 78)
    agg = D.groupby("method")[["auc", "power", "observed_fdr"]].agg(["mean", "std", "min", "max"]).round(4)
    print(agg.to_string())
    print(f"\n-> {out / 'sim_replicates.csv'}")


if __name__ == "__main__":
    main()
