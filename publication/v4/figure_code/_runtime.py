"""
Shared runtime settings for the NON-benchmark analysis scripts.

Import this before numpy/scipy and call ``background()``: it caps BLAS, numba, dask
and joblib at one thread/process each (so parallel workers do not multiply into
hundreds of threads) and lowers the process priority, which child processes -- worker
processes and Rscript alike -- inherit on Windows. The machine stays usable while the
analyses run.

Timing benchmarks (bench_*.py) must not use it: they need a quiet machine and normal
priority, and their thread counts are set explicitly.
"""
from __future__ import annotations

import os

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS", "DASK_NUM_WORKERS", "LOKY_MAX_CPU_COUNT",
                # read by run_de_comparison and passed to PyDESeq2 as n_cpus
                "DUET_N_CPUS")


def limit_threads(n: int = 1) -> None:
    for v in _THREAD_VARS:
        os.environ.setdefault(v, str(n))


def lower_priority() -> None:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as e:                                    # pragma: no cover
        print(f"[runtime] could not lower priority: {e}", flush=True)


def background(threads: int = 1) -> None:
    limit_threads(threads)
    lower_priority()


def maybe_background() -> None:
    """For scripts that also serve as benchmarks: lower the priority only when
    DUET_BACKGROUND=1 (thread limits must then come from the environment, since
    numpy is already imported)."""
    if os.environ.get("DUET_BACKGROUND") == "1":
        lower_priority()


def read_obs(path):
    """Only the obs table of an .h5ad (no matrices, no layers)."""
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:                                       # pragma: no cover
        from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        return read_elem(f["obs"])


def adaptive_run(jobs, max_workers: int, max_load: float = 75.0, min_free_gb: float = 8.0, poll_s: float = 30.0):
    """Run (fn, args) jobs in worker processes, starting a new one only while the
    machine has room: system CPU load below `max_load` % and at least `min_free_gb`
    of free RAM. Yields (args, result_or_exception) as jobs finish. The user's own
    work keeps priority: when the load rises, fewer jobs run."""
    import time
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    import psutil
    pending, running = list(jobs), {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        while pending or running:
            load = psutil.cpu_percent(interval=1.0)
            free = psutil.virtual_memory().available / 2 ** 30
            if pending and len(running) < max_workers and (not running or (load < max_load and free > min_free_gb)):
                fn, args = pending.pop(0)
                running[ex.submit(fn, *args)] = args
                continue
            if running:
                done, _ = wait(list(running), timeout=poll_s, return_when=FIRST_COMPLETED)
                for f in done:
                    args = running.pop(f)
                    try:
                        yield args, f.result()
                    except Exception as e:                 # noqa: BLE001 - reported to the caller
                        yield args, e
            else:
                time.sleep(poll_s)


def load_rows(path, obs_mask=None, *, layers=("counts",)):
    """Read only the selected cells of a large .h5ad (CSR rows), including layers.

    anndata's backed mode still loads every layer into memory; this reads the row
    slices directly, so a few thousand cells of a 200k-cell object cost megabytes.
    ``obs_mask`` is a boolean array over all cells, or a callable taking obs.
    """
    import anndata as ad
    import h5py
    import numpy as np
    try:
        from anndata.io import read_elem, sparse_dataset
    except ImportError:                                       # pragma: no cover
        from anndata.experimental import read_elem, sparse_dataset
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        mask = obs_mask(obs) if callable(obs_mask) else obs_mask
        idx = np.arange(len(obs)) if mask is None else np.flatnonzero(np.asarray(mask, bool))
        X = sparse_dataset(f["X"])[idx]
        lay = {k: sparse_dataset(f["layers"][k])[idx] for k in layers if "layers" in f and k in f["layers"]}
    out = ad.AnnData(X=X, obs=obs.iloc[idx].copy(), var=var)
    for k, v in lay.items():
        out.layers[k] = v
    return out
