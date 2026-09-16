#!/usr/bin/env python
"""
sim_calls.py -- per-gene results of every method on every muscat replicate
.

One file per replicate and method:
    results/sim_reps100/calls/<sim_repNN>/<method>.csv
with columns gene, tested, skip_reason, pvalue, fdr, score, log2fc, stat.
`score` is the underflow-safe -log10 p where the method's output allows it
(DUET neglog10p; MAST from its retained lambda; Wilcoxon and DESeq2 from their z);
`log2fc` is kept for its sign, which the combined DUET + pseudobulk rule uses.
DUET keeps its untested genes with the reason, so missing genes can be explained.

Resumable: a (replicate, method) pair whose file exists is skipped, so the job can
be stopped and restarted and methods can be added later. A replicate is used only
once its seed.txt exists (simulate_replicates.R writes it last).

Methods: duet, wilcoxon, deseq2 (pseudobulk, one profile per simulated sample),
diffxpy, mast (R; bench_mast_run.R on DUET's tested genes, as in the earlier runs).
Replicates run in parallel worker processes, each limited to one BLAS thread.

Usage: python scripts/sim_calls.py [--reps 1-100] [--methods duet,wilcoxon,deseq2,diffxpy,mast]
                                   [--workers 10] [--outdir results/sim_reps100/calls]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMBA_NUM_THREADS",
           # dask (diffxpy/batchglm) otherwise starts one thread per CPU
           "DASK_NUM_WORKERS",
           # PyDESeq2 runs a joblib/loky pool with one process per CPU by default; inside
           # ten workers that meant hundreds of processes and froze the machine. It
           # ignores LOKY_MAX_CPU_COUNT, so run_de_comparison passes DUET_N_CPUS as n_cpus.
           "LOKY_MAX_CPU_COUNT", "DUET_N_CPUS"):
    os.environ.setdefault(_v, "1")                    # one thread / process per worker


def _lower_priority():
    """Below-normal CPU priority for this process; on Windows child processes
    (workers, Rscript) inherit it, so the machine stays usable. Not for timing runs."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as e:                                   # pragma: no cover
        print(f"[calls] could not lower priority: {e}", flush=True)

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent            # project root; this file is in scripts/
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")
LN2 = math.log(2.0)
ALL_METHODS = ("duet", "wilcoxon", "deseq2", "diffxpy", "mast", "duet_pb")
COLS = ["gene", "tested", "skip_reason", "pvalue", "fdr", "score", "log2fc", "stat"]


def parse_reps(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _harmon(d: pd.DataFrame) -> pd.DataFrame:
    """run_de_comparison output -> the columns kept here (tested rows only)."""
    return pd.DataFrame({"gene": d["gene"].astype(str), "tested": d["tested"].astype(bool),
                         "skip_reason": d["skip_reason"].fillna("").astype(str),
                         "pvalue": d["pvalue"].astype(float), "fdr": d["fdr"].astype(float),
                         "score": d["neglog10p"].astype(float), "log2fc": d["log2fc"].astype(float),
                         "stat": d["stat"].astype(float)})[COLS]


def _duet(A) -> pd.DataFrame:
    from duet import run_duet
    with tempfile.TemporaryDirectory() as tmp:
        out = run_duet(A, output_dir=tmp, celltype_col="celltype", sample_col="Sample",
                       ref_label="A", test_label="B", mast_compat=True, eb_shrinkage=True,
                       vectorized=True, logistic_engine="numpy_irls", memory_log=False,
                       output_name="duet.csv")
        d = pd.read_csv(out, low_memory=False)
    res = pd.DataFrame({"gene": d["gene"].astype(str), "tested": d["tested"].astype(bool),
                        "skip_reason": d["skip_reason"].fillna("").astype(str),
                        "pvalue": d["pvalue"], "fdr": d["fdr"], "score": d["neglog10p"],
                        "log2fc": d["coef"] / LN2, "stat": d["stat_hurdle"]})
    res["stat_detect"] = d["stat_detect"]
    res["stat_continuous"] = d["stat_continuous"]
    res["detect_rate_ref"] = d["detect_rate_ref"]
    res["detect_rate_test"] = d["detect_rate_test"]
    return res


def _mast(A, duet: pd.DataFrame, workdir: Path) -> pd.DataFrame:
    import scipy.sparse as sp
    from scipy.io import mmwrite
    from duet.core import _neglog10_chi2_sf
    genes = duet.loc[duet["tested"], "gene"].astype(str).to_numpy()
    workdir.mkdir(parents=True, exist_ok=True)
    X = A.X.tocsc() if sp.issparse(A.X) else sp.csc_matrix(A.X)
    gi = pd.Index(A.var_names).get_indexer(genes)
    mmwrite(str(workdir / "expr.mtx"), X[:, gi].T.tocoo(), field="real", precision=7)
    np.savetxt(workdir / "genes.txt", genes, fmt="%s")
    np.savetxt(workdir / "cells.txt", np.asarray(A.obs_names, dtype=str), fmt="%s")
    np.savetxt(workdir / "cond.txt", A.obs["Sample"].astype(str).to_numpy(), fmt="%s")
    (workdir / "labels.json").write_text('{"ref": "A", "test": "B"}', encoding="utf-8")
    r = subprocess.run([RSCRIPT, str(HERE / "scripts" / "bench_mast_run.R"), str(workdir), "1"],
                       capture_output=True, text=True)
    (workdir / "expr.mtx").unlink(missing_ok=True)
    if r.returncode != 0 or not (workdir / "mast_results.csv").is_file():
        raise RuntimeError(f"MAST failed rc={r.returncode}: {r.stderr[-600:]}")
    m = pd.read_csv(workdir / "mast_results.csv")
    return pd.DataFrame({"gene": m["gene"].astype(str), "tested": True, "skip_reason": "",
                         "pvalue": m["pvalue"], "fdr": m["fdr"],
                         # MAST keeps its likelihood-ratio lambda, so the tail is recoverable
                         "score": _neglog10_chi2_sf(m["stat_hurdle"].to_numpy(float),
                                                    m["df_hurdle"].to_numpy(float)),
                         "log2fc": m["coef"] / LN2, "stat": m["stat_hurdle"]})[COLS]


def _duet_pb(A, duet: pd.DataFrame) -> pd.DataFrame:
    """DUET's own sample-level arm and combined rule (duet.pseudobulk), on raw counts summed
    per simulated sample. `call` is the rule the manuscript states: significant only when
    pseudobulk padj < 0.05 and both arms agree on the direction."""
    from duet.pseudobulk import combine_calls, run_pseudobulk
    pb = run_pseudobulk(A, donor_col="SourceFile", condition_col="Sample", ref_label="A", test_label="B",
                        counts_layer="counts", celltype="cluster1", min_cells_per_donor=10,
                        min_donors_per_group=2, min_gene_counts=10, n_cpus=1)
    cell = pd.DataFrame({"gene": duet["gene"].astype(str), "celltype": "cluster1", "fdr": duet["fdr"],
                         "coef": duet["log2fc"], "score": duet["score"]})
    comb = combine_calls(cell, pb)
    return pd.DataFrame({"gene": comb["gene"].astype(str), "tested": comb["pb_tested"].fillna(False).astype(bool),
                         "skip_reason": comb["pb_skip_reason"].fillna("").astype(str),
                         "pvalue": comb["pb_pvalue"], "fdr": comb["pb_padj"], "score": comb["score"],
                         "log2fc": comb["pb_log2FC"], "stat": comb["pb_stat"], "call": comb["call"]})


def run_replicate(rep: int, methods: list[str], outdir: str) -> list[str]:
    """Run the requested methods on one replicate; returns log lines."""
    import warnings
    warnings.filterwarnings("ignore")
    import run_de_comparison as R
    from evaluate_sim_replicates import build

    tag = f"sim_rep{rep:02d}"
    src = INPUTS / "data" / tag
    dest = Path(outdir) / tag
    dest.mkdir(parents=True, exist_ok=True)
    todo = [m for m in methods if not (dest / f"{m}.csv").is_file()]
    if ({"mast", "duet_pb"} & set(todo)) and "duet" not in todo and not (dest / "duet.csv").is_file():
        todo.insert(0, "duet")
    if not todo:
        return [f"[{tag}] all requested methods present"]
    log = []
    A = build(src)
    A.obs["condition"] = A.obs["Sample"].astype(str)
    n_ref = int((A.obs.Sample == "A").sum())
    n_test = int((A.obs.Sample == "B").sum())
    duet = pd.read_csv(dest / "duet.csv") if (dest / "duet.csv").is_file() else None
    for m in todo:
        t0 = time.time()
        try:
            if m == "duet":
                res = duet = _duet(A)
            elif m == "wilcoxon":
                res = _harmon(R.run_wilcoxon(A, "cluster1", ref_label="A", test_label="B",
                                             n_ref=n_ref, n_test=n_test))
            elif m == "deseq2":
                res = _harmon(R.run_deseq2(A, "cluster1", counts_layer="counts", sample_col="SourceFile",
                                           condition=A.obs.Sample.astype(str).to_numpy(), ref_label="A",
                                           test_label="B", n_ref=n_ref, n_test=n_test,
                                           min_samples_per_group=2, min_gene_counts=10,
                                           min_cells_per_sample=10))
            elif m == "diffxpy":
                res = _harmon(R.run_diffxpy(A, "cluster1", counts_layer="counts",
                                            condition=A.obs.Sample.astype(str).to_numpy(),
                                            ref_label="A", test_label="B", n_ref=n_ref,
                                            n_test=n_test, min_cells=10))
            elif m == "duet_pb":
                res = _duet_pb(A, duet)
            elif m == "mast":
                res = _mast(A, duet, Path(tempfile.gettempdir()) / f"mast_{tag}")
            else:
                raise ValueError(m)
            # write-then-rename: an interrupted run never leaves a partial file that
            # the resume logic would mistake for a finished one
            tmp = dest / f"{m}.csv.part"
            res.to_csv(tmp, index=False)
            os.replace(tmp, dest / f"{m}.csv")
            log.append(f"[{tag}] {m:8s} OK   {time.time() - t0:7.1f}s  tested={int(res['tested'].sum())}")
        except Exception as e:                                   # keep going; the pair is retried next run
            log.append(f"[{tag}] {m:8s} FAIL {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="1-100")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel replicates for the Python methods (~1-2 GB each)")
    ap.add_argument("--mast-workers", type=int, default=2,
                    help="parallel MAST runs; each densifies ~24,000 x 4,000 and peaks near 5 GB")
    ap.add_argument("--priority", choices=["low", "normal"], default="low")
    ap.add_argument("--outdir", default=str(HERE / "results" / "sim_reps100" / "calls"))
    a = ap.parse_args()
    if a.priority == "low":
        _lower_priority()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    bad = set(methods) - set(ALL_METHODS)
    if bad:
        raise SystemExit(f"unknown methods {bad}")
    reps = [r for r in parse_reps(a.reps) if (INPUTS / "data" / f"sim_rep{r:02d}" / "seed.txt").is_file()]
    # MAST runs in its own phase with fewer workers: it is the memory-heavy step
    phases = [([m for m in methods if m != "mast"], a.workers), (["mast"] if "mast" in methods else [], a.mast_workers)]
    t0 = time.time()
    for meths, workers in phases:
        if not meths:
            continue
        print(f"[calls] {len(reps)} complete replicates, methods={meths}, workers={workers} -> {a.outdir}",
              flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_replicate, r, meths, a.outdir): r for r in reps}
            for f in as_completed(futs):
                try:
                    for line in f.result():
                        print(line, flush=True)
                except Exception as e:
                    print(f"[sim_rep{futs[f]:02d}] WORKER FAIL {type(e).__name__}: {e}", flush=True)
    print(f"[calls] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
