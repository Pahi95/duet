#!/usr/bin/env python
"""
donor_aware_methods.py -- MO task 12: DUET (ranking) + pseudobulk (calling) against
methods that model the donor directly: GLIMES (Poisson and binomial GLMM with a donor
random intercept; Wu, Zhou & Chen 2025) and MAST with a glmer donor random effect.

The mixed models fit one GLMM per gene and cost seconds per gene at tens of
thousands of cells, so cells are subsampled -- to --cells-per-sample per simulated
sample, or --cells-per-donor per donor in the null -- and every method, DUET and
pseudobulk included, is run on the same subsample.

  --mode sim   muscat replicates (default 51-70, the evaluation half), scored against
               the same truth as sim_score.py; also records runtime and R heap per method
  --mode null  unique donor-swap splits (donor_swap_partitions.py) for the given
               dataset/cell types; every discovery is false
  --probe      one replicate or split only, to time the methods before a full run

Outputs: results/donor_aware/<mode>/<unit>/<method>.csv, results/donor_aware/<mode>_metrics.csv
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
import shutil                                              # noqa: E402
import subprocess                                          # noqa: E402
import tempfile                                            # noqa: E402
import time                                                # noqa: E402
import warnings                                            # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
RLIB = str(HERE.parent / "_Rlib")
R_METHODS = ("glimes_poisson", "glimes_binomial", "mast_glmer")
ALPHA = 0.05


def subsample(A, unit_col: str, n_per_unit: int, seed: int):
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in A.obs.groupby(unit_col, observed=True).indices.items():
        keep.extend(idx if len(idx) <= n_per_unit else rng.choice(idx, n_per_unit, replace=False))
    return A[np.sort(np.asarray(keep))].copy()


def python_arms(A, donor_col, ref, test) -> dict[str, pd.DataFrame]:
    """DUET, DESeq2 pseudobulk and the combined rule on exactly these cells."""
    from duet import run_duet
    from duet.pseudobulk import run_pseudobulk
    with tempfile.TemporaryDirectory() as tmp:
        out = run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="condition", ref_label=ref,
                       test_label=test, mast_compat=True, vectorized=True, memory_log=False, output_name="d.csv")
        d = pd.read_csv(out, low_memory=False)
    d = d[d.tested == True]                                                         # noqa: E712
    du = pd.DataFrame({"gene": d.gene.astype(str), "pvalue": d.pvalue, "fdr": d.fdr, "score": d.neglog10p,
                       "log2fc": d.coef}).set_index("gene")
    pb = run_pseudobulk(A, donor_col=donor_col, condition_col="condition", ref_label=ref, test_label=test,
                        counts_layer="counts", min_cells_per_donor=10, min_donors_per_group=2, n_cpus=1)
    pb = pb[pb.pb_tested & pb.pb_pvalue.notna()]
    pbd = pd.DataFrame({"gene": pb.gene.astype(str), "pvalue": pb.pb_pvalue, "fdr": pb.pb_padj,
                        "score": pb.pb_neglog10p, "log2fc": pb.pb_log2FC}).set_index("gene")
    g = du.index.union(pbd.index)
    pb_sig = pbd.reindex(g)["fdr"] < ALPHA
    same = np.sign(pbd.reindex(g)["log2fc"]) == np.sign(du.reindex(g)["log2fc"])
    sig = pb_sig & same
    call = np.where(sig, "significant", np.where(pb_sig & ~same, "direction_conflict",
                    np.where(du.reindex(g)["fdr"] < ALPHA, "ranked_only", "ns")))
    comb = pd.DataFrame({"pvalue": np.nan, "fdr": np.where(sig, 0.0, 1.0), "score": du.reindex(g)["score"],
                         "log2fc": du.reindex(g)["log2fc"], "call": call}, index=g)
    return {"DUET": du, "DESeq2-pseudobulk": pbd, "DUET+PB": comb}


def r_arm(A, donor_col, ref, test, method, workdir: Path) -> tuple[pd.DataFrame, dict]:
    import scipy.sparse as sp
    from scipy.io import mmwrite
    workdir.mkdir(parents=True, exist_ok=True)
    genes = np.asarray(A.var_names, dtype=str)
    if not (workdir / "genes.txt").is_file():
        C = A.layers["counts"]
        mmwrite(str(workdir / "counts.mtx"), sp.csr_matrix(C).T.tocoo(), field="integer")
        mmwrite(str(workdir / "expr.mtx"), sp.csr_matrix(A.X).T.tocoo(), field="real", precision=7)
        np.savetxt(workdir / "genes.txt", genes, fmt="%s")
        np.savetxt(workdir / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
        pd.DataFrame({"donor": A.obs[donor_col].astype(str).to_numpy(),
                      "condition": A.obs["condition"].astype(str).to_numpy()},
                     index=A.obs_names).to_csv(workdir / "coldata.csv")
        (workdir / "labels.json").write_text(json.dumps({"ref": ref, "test": test}), encoding="utf-8")
    r = subprocess.run([RSCRIPT, str(HERE / "scripts" / "donor_aware_run.R"), str(workdir), method, RLIB],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{method}: {r.stderr[-600:]}")
    d = pd.read_csv(workdir / f"{method}.csv")
    timing = json.loads((workdir / f"{method}_timing.json").read_text())
    d["score"] = -np.log10(d["pvalue"].clip(lower=1e-300))
    return d.set_index(d["gene"].astype(str))[["pvalue", "fdr", "score", "log2fc", "status"]], timing


def load_sim(rep: int, cells_per_sample: int):
    from evaluate_sim_replicates import build
    A = build(INPUTS / "data" / f"sim_rep{rep:02d}")
    A.obs["condition"] = A.obs["Sample"].astype(str)
    return subsample(A, "SourceFile", cells_per_sample, seed=rep)


def run_sim_unit(rep: int, methods: list[str], cells_per_sample: int, outdir: str,
                 max_genes: int | None = None) -> list[dict]:
    warnings.filterwarnings("ignore")
    _runtime.lower_priority()
    from sim_score import truth_table, rep_metrics
    tag = f"sim_rep{rep:02d}"
    dest = Path(outdir) / (tag if max_genes is None else f"{tag}_probe")
    dest.mkdir(parents=True, exist_ok=True)
    A = load_sim(rep, cells_per_sample)
    if max_genes:                                      # probe: time a slice, extrapolate per gene
        A = A[:, :max_genes].copy()
    res, timing = {}, {}
    t0 = time.time()
    for m, d in python_arms(A, "SourceFile", "A", "B").items():
        res[m] = d
    timing["DUET+pseudobulk_s"] = round(time.time() - t0, 1)
    # one work directory per replicate AND method set: two concurrent jobs on the same
    # replicate (e.g. GLIMES and MAST glmer) must never share it, or one job's cleanup
    # deletes the other's files mid-run (this cost ten MAST glmer replicates once)
    work = Path(tempfile.gettempdir()) / f"da_{tag}_{'+'.join(methods)}"
    for m in methods:
        f = dest / f"{m}.csv"
        if f.is_file():
            res[m] = pd.read_csv(f, index_col=0)
            continue
        d, t = r_arm(A, "SourceFile", "A", "B", m, work)
        d.to_csv(f)
        res[m] = d
        timing[f"{m}_s"] = t["elapsed_s"]
        timing[f"{m}_r_heap_mb"] = t["r_heap_peak_mb"]
    shutil.rmtree(work, ignore_errors=True)
    truth = truth_table(tag)
    common =sorted(set.intersection(*(set(d.index[d["score"].notna()]) for d in res.values())) & set(truth.index))
    rows = rep_metrics(res, truth, common)
    for r in rows:
        r.update(replicate=tag, n_cells=A.n_obs, **timing)
    return rows


def run_null_unit(name: str, celltype: str, split_ids: list[int] | None, methods, cells_per_donor, outdir) -> list[dict]:
    warnings.filterwarnings("ignore")
    _runtime.lower_priority()
    from donor_swap_partitions import DATASETS, unique_splits
    cfg = DATASETS[name]
    sub = _runtime.load_rows(cfg["h5ad"], lambda o: ((o["celltype"].astype(str) == celltype)
                                                     & (o[cfg["cond"]].astype(str) == cfg["ref"])).to_numpy())
    counts = sub.obs[cfg["donor"]].astype(str).value_counts()
    donors = sorted(counts.index[counts >= 10])
    sub = subsample(sub[sub.obs[cfg["donor"]].astype(str).isin(donors)], cfg["donor"], cells_per_donor, seed=0)
    rows = []
    for i, (arm_a, arm_b) in enumerate(unique_splits(donors)):
        if split_ids and (i + 1) not in split_ids:
            continue
        tag = f"{name}__{celltype.replace(' ', '_')}__split{i + 1:02d}"
        done = Path(outdir) / f"{tag}__{'+'.join(methods)}.csv"
        if done.is_file():                  # one file per split: a stopped run resumes where it left off
            rows.extend(pd.read_csv(done).to_dict("records"))
            continue
        A = sub[sub.obs[cfg["donor"]].astype(str).isin(arm_a + arm_b)].copy()
        A.obs["condition"] = np.where(A.obs[cfg["donor"]].astype(str).isin(arm_a), "armA", "armB")
        res = python_arms(A, cfg["donor"], "armA", "armB")
        work = Path(tempfile.gettempdir()) / f"da_{tag}_{'+'.join(methods)}"     # see run_sim_unit
        timing = {}
        for m in methods:
            d, t = r_arm(A, cfg["donor"], "armA", "armB", m, work)
            res[m] = d
            timing[m] = t["elapsed_s"]
        shutil.rmtree(work, ignore_errors=True)
        split_rows = []
        for m, d in res.items():
            n_t = int(d["fdr"].notna().sum())
            n_d = int((d["fdr"] < ALPHA).sum())
            split_rows.append(dict(dataset=name, celltype=celltype, split=i + 1, arm_a=";".join(arm_a),
                                   arm_b=";".join(arm_b), n_cells=A.n_obs, method=m, n_tested=n_t,
                                   n_discoveries=n_d, rejection_pct=100 * n_d / max(n_t, 1),
                                   any_discovery=n_d > 0, elapsed_s=timing.get(m, np.nan)))
        Path(outdir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(split_rows).to_csv(done.with_suffix(".part"), index=False)
        os.replace(done.with_suffix(".part"), done)
        rows.extend(split_rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "null"], default="sim")
    ap.add_argument("--reps", default="51-70")
    ap.add_argument("--null-cases", default="kang:CD14+ Monocytes,crowell:Astrocytes,pancreas:Ductal cell")
    ap.add_argument("--methods", default=",".join(R_METHODS))
    ap.add_argument("--cells-per-sample", type=int, default=500)
    ap.add_argument("--cells-per-donor", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2, help="maximum parallel units")
    ap.add_argument("--max-load", type=float, default=75.0, help="start no new unit above this CPU load %%")
    ap.add_argument("--min-free-gb", type=float, default=8.0, help="start no new unit below this free RAM")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--outdir", default=str(HERE / "results" / "donor_aware"))
    a = ap.parse_args()
    _runtime.lower_priority()
    methods = [m for m in a.methods.split(",") if m]
    out = Path(a.outdir) / a.mode
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "sim":
        lo, hi = map(int, a.reps.split("-"))
        units = [r for r in range(lo, hi + 1) if (INPUTS / "data" / f"sim_rep{r:02d}" / "seed.txt").is_file()]
        units = units[:1] if a.probe else units
        jobs = [(run_sim_unit, (r, methods, a.cells_per_sample, str(out), 300 if a.probe else None))
                for r in units]
    else:
        cases = [c.split(":") for c in a.null_cases.split(",")]
        jobs = [(run_null_unit, (n, ct, [1] if a.probe else None, methods, a.cells_per_donor, str(out)))
                for n, ct in (cases[:1] if a.probe else cases)]
    rows = []
    # start a new unit only while the machine has spare CPU and RAM (user request)
    for args, r in _runtime.adaptive_run(jobs, max_workers=a.workers, max_load=a.max_load,
                                         min_free_gb=a.min_free_gb):
        if isinstance(r, Exception):
            print(f"[donor-aware] FAIL {args[:2]}: {type(r).__name__}: {r}", flush=True)
            continue
        rows.extend(r)
        df = pd.DataFrame(r)
        keep = [c for c in df.columns if c in ("replicate", "dataset", "split", "method", "n_scored", "n_tested",
                                               "sensitivity", "observed_fdr", "roc_auc", "n_discoveries",
                                               "rejection_pct") or c.endswith("_s") or c.endswith("_mb")]
        print(df[keep].to_string(index=False), flush=True)
        if not a.probe:                        # after every unit, so a stop loses nothing
            pd.DataFrame(rows).to_csv(Path(a.outdir) / f"{a.mode}_metrics_{'+'.join(methods)}.csv", index=False)


if __name__ == "__main__":
    main()
