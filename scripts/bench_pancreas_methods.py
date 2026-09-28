#!/usr/bin/env python
"""Versioned timing/memory benchmark for diffxpy, PyDESeq2, and Wilcoxon.

Runs one engine at a time in fresh processes, with one warm-up and five measured
runs by default. Uses only pancreas_20260925 inputs and writes only to its own
results directory. No historical benchmark row is treated as a new measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

import bench_harness as bh


HERE = Path(__file__).resolve().parent.parent
INPUTS = HERE.parent / "inputs" / "bench" / "pancreas_20260925"
OUTPUT = HERE / "results" / "pancreas_20260925" / "bench_extra"
NAMES = (
    "ductal_1000", "ductal_2500", "ductal_5000", "ductal_all",
    "pooled_2500", "pooled_5000", "pooled_10000", "pooled_15000", "pooled_all",
)
ENGINES = ("wilcoxon", "deseq2", "diffxpy")
DRIVER = Path(__file__).with_name("bench_pancreas_methods_run.py")


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(part, path)


def read_runs(out: Path, name: str, engine: str, source_hash: str,
              driver_hash: str) -> list[dict]:
    files = sorted((out / "runs" / name / engine).glob("run_*.json"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    for run in runs:
        if run.get("source_sha256") != source_hash or run.get("driver_sha256") != driver_hash:
            raise ValueError(f"run for another source/driver found at {out / 'runs' / name / engine}")
    return runs


def usable(run: dict, shared: bool) -> bool:
    return (not run["warmup"] and run["returncode"] == 0 and
            run["sampler_errors"] == 0 and (shared or not run["contaminated"]))


def export_csv(out: Path) -> None:
    rows = []
    for path in sorted((out / "runs").glob("*/*/run_*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        base = {k: v for k, v in run.items() if k not in ("stages", "errors", "info", "tail")}
        base["errors"] = " | ".join(run["errors"])
        base["info"] = " | ".join(run["info"])
        rows.append(dict(base, stage="TOTAL_RUN", elapsed_s=run["wall_s"]))
        for stage in run["stages"]:
            rows.append(dict(base, **stage))
    if rows:
        part = out / "measurements.csv.part"
        pd.DataFrame(rows).to_csv(part, index=False)
        os.replace(part, out / "measurements.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", default=",".join(NAMES))
    p.add_argument("--engines", default=",".join(ENGINES))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--outdir", type=Path, default=OUTPUT)
    p.add_argument("--shared", action="store_true", help="record CPU time on a shared machine")
    p.add_argument("--max-load", type=float, default=25.0)
    p.add_argument("--max-background", type=float, default=20.0)
    args = p.parse_args()
    names = args.inputs.split(",")
    engines = args.engines.split(",")
    if not set(names) <= set(NAMES) or not set(engines) <= set(ENGINES):
        p.error("unknown input or engine")
    manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    source_hash = manifest["source_sha256"]
    lookup = {x["name"]: x for x in manifest["inputs"]}
    for name in names:
        for key in ("normalized", "counts"):
            if not Path(lookup[name][key]).is_file():
                raise FileNotFoundError(lookup[name][key])

    args.outdir.mkdir(parents=True, exist_ok=True)
    psutil.Process().cpu_affinity(list(range(16)))
    if args.shared:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "DASK_NUM_WORKERS",
                "LOKY_MAX_CPU_COUNT", "DUET_N_CPUS"):
        env[key] = "1"
    driver_hash = hashlib.sha256(DRIVER.read_bytes()).hexdigest()
    atomic_json(args.outdir / "protocol.json", {
        "source_sha256": source_hash, "manifest": str(INPUTS / "manifest.json"),
        "driver_sha256": driver_hash, "engines": engines, "inputs": names,
        "repeats": args.repeats, "shared": args.shared, "cpu_affinity": "0-15",
        "python": sys.executable, "pseudobulk_sample": "DonorID",
    })
    for name in names:
        for round_no in range(args.repeats + 1):
            warmup = round_no == 0
            for engine in engines:
                old = read_runs(args.outdir, name, engine, source_hash, driver_hash)
                if warmup and any(r["warmup"] and r["returncode"] == 0 for r in old):
                    continue
                if not warmup and sum(usable(r, args.shared) for r in old) >= round_no:
                    continue
                for attempt in range(2 * args.repeats):
                    if (args.outdir / "STOP").is_file():
                        print("[extra] STOP requested before next run", flush=True)
                        raise SystemExit(bh.STOPPED)
                    load = psutil.cpu_percent(interval=2) if args.shared else bh.wait_until_idle(args.max_load)
                    old = read_runs(args.outdir, name, engine, source_hash, driver_hash)
                    run_no = len(old) + 1
                    dest = args.outdir / "runs" / name / engine
                    dest.mkdir(parents=True, exist_ok=True)
                    data = Path(lookup[name]["normalized" if engine == "wilcoxon" else "counts"])
                    started = time.time()
                    output = dest / f"run_{run_no:02d}_genes.csv"
                    result = bh.run_child([
                        sys.executable, "-B", "-u", str(DRIVER), "--engine", engine,
                        "--input", str(data), "--out", str(output),
                    ], env)
                    result.update(
                        input=name, engine=engine, run=run_no, warmup=warmup,
                        started=started, source_input=str(data), source_sha256=source_hash,
                        driver_sha256=driver_hash, cpu_load_before=load,
                        contaminated=(not args.shared and
                                      result["background_cpu_mean"] > args.max_background),
                        shared_machine=args.shared, threads=1, cpus="0-15",
                    )
                    atomic_json(dest / f"run_{run_no:02d}.json", result)
                    export_csv(args.outdir)
                    print(f"[extra] {name} {engine} run={run_no} warmup={warmup} "
                          f"cpu={result['cpu_s']:.1f}s rc={result['returncode']}", flush=True)
                    if result["returncode"] != 0 or result["sampler_errors"]:
                        raise RuntimeError(result["tail"])
                    if warmup or not result["contaminated"]:
                        break
                else:
                    raise RuntimeError(f"too many contaminated runs: {name}/{engine}")
    print(f"[extra] completed; measurements in {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
