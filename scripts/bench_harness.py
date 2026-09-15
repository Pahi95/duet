#!/usr/bin/env python
"""
bench_harness.py -- the timing and memory benchmark (MO tasks 1, 2 and 13).

Run on an otherwise idle machine and at normal priority. Before every run it WAITS
while system CPU load exceeds --max-load (default 25%; the idle desktop sits near 10%);
during every run it measures the CPU used by other processes, and a run above
--max-background (default 20%) is marked contaminated and repeated (at most repeats
extra runs). Stop at any time: create DUET/_tmp/STOP (the run in progress finishes,
exit code 99) or kill the process (the run in progress is lost). Rerunning the same
command resumes: usable runs already in the output CSV are kept and not repeated. (Measurements taken under parallel load have had
to be retracted twice.)

For every input and configuration: one warm-up run, discarded, then --repeats runs,
each in fresh processes. Memory is the resident set size of the whole process tree
-- the child and all of its descendants, which matters on Windows where Rscript
starts a second process -- sampled every 20 ms. Reported per run and stage: elapsed
time and peak RSS; per run: baseline RSS (after imports, before any data is read),
peak RSS and peak - baseline, and wall-clock time. Every child gets --threads
BLAS/OpenMP threads (default 1), the same for both engines.

Engines (both start from the same inputs/bench/<name>/data.h5ad):
  duet   python bench_duet_run.py <data.h5ad> <tmp> [--design cdr]
  mast   python bench_duet_run.py <data.h5ad> <tmp> --export      (Python half)
         Rscript bench_mast_steps.R <tmp> <sparse|dense> <ebayes> <method> <design>

Usage:
  python scripts/bench_harness.py --inputs ductal_1000,ductal_2500 --engines duet,mast \
         [--mast-input dense] [--ebayes TRUE] [--method bayesglm] [--design condition] \
         [--repeats 5] [--tag fair] [--out results/bench/<tag>.csv]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent.parent
INPUTS = Path(os.environ.get("DUET_INPUTS", HERE.parent / "inputs"))
STOP_FILE = Path(os.environ.get("DUET_STOP_FILE", HERE.parent / "_tmp" / "STOP"))
RSCRIPT = (os.environ.get("RSCRIPT") or shutil.which("Rscript")
           or r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe")


class TreeSampler(threading.Thread):
    """RSS of a process and all its descendants every `dt` seconds, plus -- once a
    second -- how much of the machine's CPU was used by anything ELSE (system load
    minus the measured tree), so a run disturbed by other work can be recognised."""

    def __init__(self, pid: int, dt: float = 0.02):
        super().__init__(daemon=True)
        # not `_stop`: that name is threading.Thread's own internal method
        self.root, self.dt, self.samples, self._halt = psutil.Process(pid), dt, [], threading.Event()
        self.background, self._procs, self._ncpu = [], {}, psutil.cpu_count(logical=True) or 1
        self.cpu_last = {}                                    # pid -> last seen user+system seconds
        self.peak_wset, self.peak_private = {}, {}            # pid -> OS peak counters (exact, not sampled)
        self.errors = 0                                       # samples lost to OS errors

    def _cpu_tick(self, procs):
        system = psutil.cpu_percent(interval=None)
        tree = 0.0
        for p in procs:
            q = self._procs.setdefault(p.pid, p)
            try:
                tree += q.cpu_percent(interval=None)          # % of one core, since the last call
            except psutil.Error:
                pass
        self.background.append(max(0.0, system - tree / self._ncpu))

    def run(self):
        psutil.cpu_percent(interval=None)                     # prime the system counter
        last_cpu = time.time()
        while not self._halt.is_set():
            try:
                procs = [self.root] + self.root.children(recursive=True)
                rss = 0
                for p in procs:
                    try:
                        mi = p.memory_info()
                        rss += mi.rss
                        if hasattr(mi, "peak_wset"):          # Windows keeps the true peak itself
                            self.peak_wset[p.pid] = max(self.peak_wset.get(p.pid, 0), mi.peak_wset)
                            self.peak_private[p.pid] = max(self.peak_private.get(p.pid, 0),
                                                           getattr(mi, "peak_pagefile", 0))
                        ct = p.cpu_times()
                        self.cpu_last[p.pid] = ct.user + ct.system
                    except psutil.Error:
                        pass
                self.samples.append((time.time(), rss / 2 ** 20))
                if time.time() - last_cpu >= 1.0:
                    self._cpu_tick(procs)
                    last_cpu = time.time()
            except psutil.Error:
                break
            except OSError:
                # e.g. WinError 1455 (paging file too small) when the machine runs out of commit memory:
                # skip this sample instead of letting the sampler thread die; the run is flagged
                self.errors += 1
            time.sleep(self.dt)

    def stop(self):
        self._halt.set()


def run_child(cmd: list[str], env: dict) -> dict:
    """Run one process, sample its tree, parse its stage markers."""
    t_start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    s = TreeSampler(p.pid)
    s.start()
    lines = [(time.time(), line.rstrip()) for line in p.stdout]
    p.wait()
    s.stop()
    s.join()
    t_end = time.time()
    starts, stages, baseline_t, errors, info = {}, [], None, [], []
    for _, ln in lines:
        parts = ln.split()
        if ln.startswith("[info]"):
            info.append(ln[7:])
        elif ln.startswith("[stage-start]"):
            starts[parts[1]] = float(parts[2])
        elif ln.startswith("[stage]"):
            kv = dict(x.split("=") for x in parts[2:] if "=" in x)
            stages.append(dict(stage=parts[1], elapsed_s=float(kv["elapsed"]), t0=starts.get(parts[1]),
                               cpu_s=float(kv["cpu"]) if "cpu" in kv else None,
                               t1=float(kv["end"]), rheap_mb=float(kv["rheap_mb"]) if "rheap_mb" in kv else None))
        elif ln.startswith("[mark] baseline"):
            baseline_t = float(parts[2])
        elif ln.startswith("[stage-error]") or ln.startswith("Error") or "Traceback" in ln:
            errors.append(ln)
    smp = pd.DataFrame(s.samples, columns=["t", "rss_mb"]) if s.samples else pd.DataFrame({"t": [], "rss_mb": []})
    for st in stages:
        w = smp[(smp.t >= (st["t0"] or t_start)) & (smp.t <= st["t1"] + 0.05)]
        st["peak_rss_mb"] = float(w.rss_mb.max()) if len(w) else float("nan")
    base = (float(smp.iloc[(smp.t - baseline_t).abs().argmin()].rss_mb) if baseline_t and len(smp) else float("nan"))
    bg = s.background
    return dict(returncode=p.returncode, wall_s=t_end - t_start, baseline_rss_mb=base,
                cpu_s=float(sum(s.cpu_last.values())),
                sampler_errors=int(s.errors),
                os_peak_wset_mb=float(max(s.peak_wset.values(), default=0)) / 2 ** 20,
                os_peak_private_mb=float(max(s.peak_private.values(), default=0)) / 2 ** 20,
                background_cpu_mean=float(sum(bg) / len(bg)) if bg else 0.0,
                background_cpu_max=float(max(bg)) if bg else 0.0,
                peak_rss_mb=float(smp.rss_mb.max()) if len(smp) else float("nan"), stages=stages,
                errors=errors, info=info, tail="\n".join(l for _, l in lines[-5:]))


def one_run(engine: str, data: Path, a, env: dict, keep: Path | None = None) -> list[dict]:
    """One timed run in fresh processes. `keep`: copy the per-gene results there
    (the agreement table needs them; timing runs otherwise leave nothing behind)."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMP")) as tmp:
        rows = _one_run(engine, data, a, env, tmp)
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            for f in ("duet_bench_results.csv", "mast_bench_results.csv"):
                if (Path(tmp) / f).is_file():
                    shutil.copy2(Path(tmp) / f, keep / f)
        return rows


def _one_run(engine: str, data: Path, a, env: dict, tmp: str) -> list[dict]:
    py = [sys.executable, str(HERE / "scripts" / "bench_duet_run.py"), str(data), tmp]
    if engine == "duet":
        parts = [("duet", run_child(py + ["--design", a.design], env))]
    else:
        exp = run_child(py + ["--export"], env)
        parts = [("export", exp)]
        if exp["returncode"] == 0:
            parts.append(("R", run_child([RSCRIPT, str(HERE / "scripts" / "bench_mast_steps.R"), tmp,
                                          a.mast_input, a.ebayes, a.method, a.design], env)))
    rows = []
    for proc, r in parts:
        for st in r["stages"]:
            rows.append(dict(process=proc, stage=st["stage"], elapsed_s=st["elapsed_s"], cpu_s=st.get("cpu_s"),
                             peak_rss_mb=st["peak_rss_mb"], rheap_mb=st["rheap_mb"]))
        rows.append(dict(process=proc, stage="TOTAL_PROCESS", elapsed_s=r["wall_s"], cpu_s=r["cpu_s"],
                         os_peak_wset_mb=r["os_peak_wset_mb"], os_peak_private_mb=r["os_peak_private_mb"],
                         peak_rss_mb=r["peak_rss_mb"],
                         baseline_rss_mb=r["baseline_rss_mb"], delta_rss_mb=r["peak_rss_mb"] - r["baseline_rss_mb"],
                         returncode=r["returncode"], errors=" | ".join(r["errors"])[:500],
                         info=" | ".join(r["info"])[:500]))
    tot_wall = sum(r["wall_s"] for _, r in parts)
    rows.append(dict(process="ALL", stage="TOTAL_RUN", elapsed_s=tot_wall, cpu_s=sum(r["cpu_s"] for _, r in parts),
                     os_peak_wset_mb=max(r["os_peak_wset_mb"] for _, r in parts),
                     os_peak_private_mb=max(r["os_peak_private_mb"] for _, r in parts),
                     peak_rss_mb=max(r["peak_rss_mb"] for _, r in parts),
                     baseline_rss_mb=parts[0][1]["baseline_rss_mb"],
                     delta_rss_mb=max(r["peak_rss_mb"] - r["baseline_rss_mb"] for _, r in parts),
                     returncode=max(r["returncode"] for _, r in parts),
                     sampler_errors=sum(r.get("sampler_errors", 0) for _, r in parts),
                     background_cpu_mean=max(r["background_cpu_mean"] for _, r in parts),
                     background_cpu_max=max(r["background_cpu_max"] for _, r in parts)))
    return rows


STOPPED = 99          # exit code: stopped on request (STOP file), not a failure
INCOMPLETE = 98       # exit code: finished, but some inputs lack usable runs (machine was busy) -- rerun


def stop_requested() -> bool:
    return STOP_FILE.is_file()


def wait_until_idle(max_load: float, poll_s: float = 30.0) -> float:
    """Block until system CPU load is below `max_load`; return the load. Never exits:
    a busy machine pauses the benchmark instead of ending it."""
    said = False
    while True:
        if stop_requested():
            print("[bench] STOP file found -- stopping before the next run", flush=True)
            raise SystemExit(STOPPED)
        load = psutil.cpu_percent(interval=5.0)
        if load <= max_load:
            if said:
                print(f"[bench] machine idle again ({load:.0f}%), continuing", flush=True)
            return load
        if not said:
            print(f"[bench] system CPU load {load:.0f}% > {max_load:.0f}%: waiting for an idle machine "
                  f"(checking every {poll_s:.0f} s)", flush=True)
            said = True
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="comma-separated names under inputs/bench/")
    ap.add_argument("--engines", default="duet,mast")
    ap.add_argument("--mast-input", choices=["sparse", "dense"], default="dense")
    ap.add_argument("--ebayes", choices=["TRUE", "FALSE"], default="TRUE")
    ap.add_argument("--method", choices=["bayesglm", "glm"], default="bayesglm")
    ap.add_argument("--design", choices=["condition", "cdr"], default="condition")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-load", type=float, default=25.0, help="wait while system CPU load is above this %%")
    ap.add_argument("--max-background", type=float, default=20.0,
                    help="a run whose mean CPU use by OTHER processes exceeds this %% is marked contaminated "
                         "and repeated")
    ap.add_argument("--force", action="store_true", help="do not wait for an idle machine (smoke tests only)")
    ap.add_argument("--shared", action="store_true",
                    help="the machine is in use: below-normal priority, no waiting, no repeats; CPU time "
                         "(unaffected by other load for single-threaded runs) is the primary timing, "
                         "wall time and background load are recorded as secondary")
    ap.add_argument("--cpus", default="0-15",
                    help="logical CPUs the benchmark may use (inherited by all children). 0-15 are the P-cores of "
                         "this i9-14900K; 16-31 are E-cores, about twice as slow per thread")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or HERE / "results" / "bench" / f"{a.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS", "DASK_NUM_WORKERS", "LOKY_MAX_CPU_COUNT"):
        env[v] = str(a.threads)
    env.pop("DUET_BACKGROUND", None)
    if a.cpus:
        lo, _, hi = a.cpus.partition("-")
        cpus = list(range(int(lo), int(hi or lo) + 1))
        psutil.Process().cpu_affinity(cpus)
        print(f"[bench] pinned to logical CPUs {cpus[0]}-{cpus[-1]}", flush=True)
    if a.shared:
        a.force = True                                    # never wait, never mark contaminated
        try:
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except Exception:
            pass

    # ---- resume: keep everything already measured, count the usable timed runs ----
    rows = pd.read_csv(out).to_dict("records") if out.is_file() else []
    def usable(name, engine):
        return sum(1 for r in rows if r.get("stage") == "TOTAL_RUN" and r.get("input") == name
                   and r.get("engine") == engine and not bool(r.get("warmup")) and r.get("returncode") == 0
                   and (a.shared or not bool(r.get("contaminated"))))
    def next_run_no(name, engine):
        runs = [int(r["run"]) for r in rows if r.get("input") == name and r.get("engine") == engine]
        return max(runs) + 1 if runs else 1

    engines = a.engines.split(",")

    def measure(name, engine, data, is_warm):
        load = psutil.cpu_percent(interval=2.0) if a.force else wait_until_idle(a.max_load)
        t0 = time.time()
        run_no = next_run_no(name, engine)
        keep = (out.parent / a.tag / name / engine) if (not is_warm and usable(name, engine) == 0) else None
        new = one_run(engine, data, a, env, keep)
        tot = new[-1]
        contaminated = (not a.force) and tot["background_cpu_mean"] > a.max_background
        for r in new:
            rows.append(dict(tag=a.tag, input=name, engine=engine, mast_input=a.mast_input if engine == "mast"
                             else "", ebayes=a.ebayes if engine == "mast" else "", method=a.method
                             if engine == "mast" else "", design=a.design, threads=a.threads, run=run_no,
                             warmup=is_warm, cpu_load_before=load, contaminated=contaminated,
                             shared_machine=a.shared, cpus=a.cpus, started=t0, **r))
        pd.DataFrame(rows).to_csv(out, index=False)        # after every run: nothing lost on a stop
        print(f"[bench] {name:22s} {engine:5s} run {run_no} {'(warm-up) ' if is_warm else ''}"
              f"{tot['elapsed_s']:8.1f}s wall {tot['cpu_s']:8.1f}s CPU peak {tot['peak_rss_mb']:8.0f} MB "
              f"(OS peak {tot['os_peak_wset_mb']:.0f} MB) rc={tot['returncode']} "
              f"background {tot['background_cpu_mean']:.1f}%{'  CONTAMINATED' if contaminated else ''}", flush=True)
        if stop_requested():
            print("[bench] STOP file found -- stopped after a completed run; rerun to resume", flush=True)
            raise SystemExit(STOPPED)
        return contaminated

    for name in a.inputs.split(","):
        data = INPUTS / "bench" / name / "data.h5ad"
        if not data.is_file():
            raise SystemExit(f"{data} missing -- run bench_prepare.py first")
        todo = [e for e in engines if usable(name, e) < a.repeats]
        for e in engines:
            if e not in todo:
                print(f"[bench] {name:22s} {e:5s} already has {usable(name, e)} usable runs, skipped", flush=True)
        # a fresh warm-up of each engine after every (re)start, so a resumed series is measured like a new one
        for e in todo:
            for _ in range(a.warmup):
                measure(name, e, data, True)
        # then alternate the engines run by run, so a change in machine load hits both alike
        attempts = {e: 0 for e in todo}
        while True:
            pending = [e for e in todo if usable(name, e) < a.repeats and attempts[e] < 2 * a.repeats]
            if not pending:
                break
            for e in pending:
                attempts[e] += 1
                measure(name, e, data, False)

    short = [(n, e, usable(n, e)) for n in a.inputs.split(",") for e in a.engines.split(",")
             if usable(n, e) < a.repeats]
    print(f"[bench] wrote {out}")
    if short:
        print("[bench] INCOMPLETE -- too many contaminated runs (machine not idle) for: "
              + ", ".join(f"{n}/{e} {k}/{a.repeats}" for n, e, k in short)
              + "; rerun on an idle machine to fill in the missing runs", flush=True)
        raise SystemExit(INCOMPLETE)


if __name__ == "__main__":
    main()
