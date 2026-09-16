#!/usr/bin/env python
"""
make_all_results.py -- one entry point for the analyses and benchmarks behind the DUET manuscript.
Each step is a script that can also be run on its own; the scripts skip work that is already done,
so a step can be repeated safely.

  python scripts/make_all_results.py --dry-run          list every step and its command
  python scripts/make_all_results.py --analyses         analyses, in the background
                                                        (below-normal priority, one thread each)
  python scripts/make_all_results.py --benchmarks       timing/memory benchmarks, one at a time
  ... [--only name1,name2] [--from name]

Order matters within --analyses (the simulations feed the scoring, the pancreas object feeds every
pancreas step). Environment: DUET_INPUTS (default ../inputs), RSCRIPT. Results go to results/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
PY = sys.executable
RS = os.environ.get("RSCRIPT") or shutil.which("Rscript") or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe"
S = str(HERE / "scripts")
PANC = str(INPUTS / "PancreasCorrected.h5ad")
TMPROOT = HERE.parent / "_tmp"
STOP_FILE = Path(os.environ.get("DUET_STOP_FILE", TMPROOT / "STOP"))
DONE_DIR = TMPROOT / "steps_done"
STOPPED = 99
INCOMPLETE = 98       # a benchmark step ran but could not collect all usable runs


def de_compare(tag, h5ad, ref, test, donor, cts):
    return [
        (f"de_{tag}", [PY, f"{S}/run_de_comparison.py", "--h5ad", h5ad, "--output-dir", f"results/{tag}",
                       "--celltypes", cts, "--source-col", donor, "--ref-label", ref, "--test-label", test,
                       "--methods", "duet,diffxpy,deseq2,wilcoxon"]),
        (f"mast_{tag}", [PY, f"{S}/run_mast_dataset.py", "--h5ad", h5ad, "--results-dir", f"results/{tag}",
                         "--ref-label", ref, "--test-label", test]),
    ]


ANALYSES = [
    ("pancreas_object", [PY, f"{S}/build_pancreas_corrected.py"]),
    ("simulate", [RS, f"{S}/simulate_replicates.R", "100", "4000", "24000", "4"]),
    ("sim_calls", [PY, f"{S}/sim_calls.py", "--reps", "1-100"]),
    ("sim_score", [PY, f"{S}/sim_score.py"]),
    ("fdr_scale", [PY, f"{S}/fdr_scale_diag.py"]),
    ("arm_offset", [PY, f"{S}/arm_offset.py"]),
    ("donor_swap", [PY, f"{S}/donor_swap_partitions.py", "--workers", "2"]),
    ("pseudobulk_engines", [PY, f"{S}/validate_pseudobulk_engines.py"]),
    ("mast_settings", [PY, f"{S}/mast_settings_check.py"]),
    ("underflow", [PY, f"{S}/underflow_check.py"]),
    ("eb_logspace", [PY, f"{S}/eb_logspace_report.py"]),
    ("batchsim", [PY, f"{S}/simulate_batch_nb.py", "--reps", "20", "--evaluate"]),
    ("donor_aware_sim_glimes", [PY, f"{S}/donor_aware_methods.py", "--mode", "sim", "--reps", "51-70",
                                "--methods", "glimes_poisson,glimes_binomial", "--workers", "2"]),
    ("donor_aware_sim_mastglmer", [PY, f"{S}/donor_aware_methods.py", "--mode", "sim", "--reps", "51-70",
                                   "--methods", "mast_glmer", "--workers", "8"]),
    ("donor_aware_null_glimes", [PY, f"{S}/donor_aware_methods.py", "--mode", "null",
                                 "--methods", "glimes_poisson,glimes_binomial", "--workers", "3"]),
    *de_compare("kang_rev", str(INPUTS / "data" / "kang_8donors.h5ad"), "ctrl", "stim", "replicate",
                "B cells,CD14+ Monocytes,CD4 T cells"),
    *de_compare("crowell_rev", str(INPUTS / "data" / "crowell_4vs4.h5ad"), "Vehicle", "LPS", "SourceFile",
                "Astrocytes,Excit. Neuron,Inhib. Neuron"),
    *de_compare("pancreas_corrected", PANC, "Reference", "pancreas", "donor",
                "Ductal cell,Endothelial cell,Stellate cell"),
    # the covariate agreement with R MAST on Kang CD14+ monocytes (general per-gene path)
    ("covariate_agreement", [PY, f"{S}/validate_covariate.py", "--outdir", "results/covariate_v5"]),
    ("irls_engine", [PY, f"{S}/validate_engine.py"]),
    # 100 label permutations per cell type; donor partitions with sex-chromosome genes
    ("permutations", [PY, f"{S}/null_permutations.py", "--n-perm", "100", "--workers", "3"]),
    ("donor_swap_sex", [PY, f"{S}/donor_swap_sexgenes.py", "--workers", "3"]),
    # further sample-level methods on the simulations
    ("sim_edger_voom", [RS, f"{S}/sim_pb_edger_voom.R", str(INPUTS / "data"), "results/sim_reps100/calls", "1-100"]),
    ("sim_score_edger_voom", [PY, f"{S}/sim_score_pb_extra.py"]),
]

LADDER = "ductal_1000,ductal_2500,ductal_5000,ductal_all,pooled_2500,pooled_5000,pooled_10000,pooled_15000,pooled_all"
GRID = ("cells_1000_g8000,cells_2500_g8000,cells_5000_g8000,cells_10000_g8000,cells_25000_g8000,cells_50000_g8000,"
        "genes_2000_c10000,genes_8000_c10000,genes_20000_c10000,sparsity_low_c10000,sparsity_mid_c10000,"
        "sparsity_high_c10000")
H = [PY, f"{S}/bench_harness.py", "--shared"]   # the machine stays in use (user request)
BENCHMARKS = [
    ("bench_prepare", [PY, f"{S}/bench_prepare.py"]),
    # task 2: the fair comparison on the real ladder
    ("fair", H + ["--inputs", LADDER, "--engines", "duet,mast", "--repeats", "5", "--tag", "fair"]),
    # task 1: which MAST step needs the memory -- dense vs sparse input, ebayes on/off
    *[(f"mem_{inp}_eb{eb}", H + ["--inputs", "ductal_all", "--engines", "mast", "--mast-input", inp,
                                "--ebayes", eb, "--repeats", "3", "--tag", f"mem_{inp}_eb{eb}"])
      for inp in ("dense", "sparse") for eb in ("TRUE", "FALSE")],
    # task 13: one factor at a time
    ("grid", H + ["--inputs", GRID, "--engines", "duet,mast", "--repeats", "5", "--tag", "grid"]),
    ("grid_cdr", H + ["--inputs", "sparsity_mid_c10000", "--engines", "duet,mast", "--design", "cdr",
                      "--repeats", "5", "--tag", "grid_cdr"]),
    # the covariate path on real data (the synthetic grid_cdr point alone does not cover it)
    ("real_cdr", H + ["--inputs", "ductal_all", "--engines", "duet,mast", "--design", "cdr",
                      "--repeats", "5", "--tag", "real_cdr"]),
    # DUET with MAST-matched detection gates and empirical Bayes on the covariate path (Table S5)
    ("real_cdr_matched", H + ["--inputs", "ductal_all", "--engines", "duet", "--design", "cdr",
                              "--duet-driver", "bench_duet_run_matched.py", "--repeats", "5", "--tag", "real_cdr_matched"]),
    ("grid_cdr_matched", H + ["--inputs", "sparsity_mid_c10000", "--engines", "duet", "--design", "cdr",
                              "--duet-driver", "bench_duet_run_matched.py", "--repeats", "5", "--tag", "grid_cdr_matched"]),
]

def background_env() -> dict:
    env = dict(os.environ, DUET_BACKGROUND="1")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS", "DASK_NUM_WORKERS", "LOKY_MAX_CPU_COUNT", "DUET_N_CPUS"):
        env[v] = "1"
    return env


def run(steps, env, a, low_priority: bool):
    """Run steps in order. A step that finished writes a marker in _tmp/steps_done and is skipped
    next time; the scripts themselves resume inside a step. A STOP file ends the queue cleanly."""
    names = [n for n, _ in steps]
    if a.only:
        steps = [s for s in steps if s[0] in a.only.split(",")]
    if a.start:
        steps = steps[names.index(a.start):]
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    for name, cmd in steps:
        marker = DONE_DIR / f"{a.queue}__{name}"
        if marker.is_file() and not a.redo:
            print(f"[all] {name}: already done ({marker.read_text().strip()}), skipped", flush=True)
            continue
        if STOP_FILE.is_file():
            print(f"[all] STOP file found -- stopping before {name}; rerun the same command to resume", flush=True)
            raise SystemExit(STOPPED)
        print(f"\n[all] === {name} ===\n[all] {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=str(HERE), env=env)
        if low_priority:
            try:
                import psutil
                psutil.Process(p.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
            except Exception:
                pass
        rc = p.wait()
        print(f"[all] {name}: exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc == STOPPED or STOP_FILE.is_file():
            print("[all] stopped on request; rerun the same command to resume", flush=True)
            raise SystemExit(STOPPED)
        if rc == INCOMPLETE:
            print(f"[all] {name}: INCOMPLETE (the machine was busy) -- not marked done; stopping the queue. "
                  "Rerun on an idle machine to resume.", flush=True)
            raise SystemExit(INCOMPLETE)
        if rc == 0:
            marker.write_text(time.strftime("%Y-%m-%d %H:%M"))
        elif not a.keep_going:
            raise SystemExit(f"[all] step {name} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses", action="store_true")
    ap.add_argument("--benchmarks", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--redo", action="store_true", help="ignore step-done markers")
    a = ap.parse_args()
    a.queue = "all"
    if a.dry_run and not (a.analyses or a.benchmarks):
        a.analyses = a.benchmarks = True
    if a.analyses:
        run(ANALYSES, background_env(), a, low_priority=True)
    if a.benchmarks:
        run(BENCHMARKS, dict(os.environ), a, low_priority=False)


if __name__ == "__main__":
    main()
