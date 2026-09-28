#!/usr/bin/env python
"""List or explicitly run pancreas DE and ductal benchmarks for the 2026-09-25 input.

Without --run this script only prints the commands. Every analysis writes under
results/pancreas_20260925. Completed steps are validated and skipped on restart;
interrupted benchmarks resume their saved measurements. Historical results are
not read or overwritten. The nine prepared inputs must already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from pancreas_resume import ResumeStore, STOPPED, execution_lock, fingerprint


HERE = Path(__file__).resolve().parent.parent
SCRIPTS = HERE / "scripts"
INPUTS = HERE.parent / "inputs" / "bench" / "pancreas_20260925"
RESULTS = HERE / "results" / "pancreas_20260925"
DUCTAL = [f"pancreas_20260925/{name}" for name in
          ("ductal_1000", "ductal_2500", "ductal_5000", "ductal_all")]
POOLED = [f"pancreas_20260925/{name}" for name in
          ("pooled_2500", "pooled_5000", "pooled_10000", "pooled_15000", "pooled_all")]
NAMES = DUCTAL + POOLED


def steps(source: str) -> list[tuple[str, list[str]]]:
    py = sys.executable
    bench = [py, str(SCRIPTS / "bench_harness.py"), "--shared"]
    out = RESULTS / "bench"
    all_name = DUCTAL[-1]
    work = [
        ("de", [py, str(SCRIPTS / "run_de_comparison.py"), "--h5ad", source,
                "--output-dir", str(RESULTS / "de"), "--celltypes",
                "Ductal cell,Endothelial cell,Stellate cell",
                "--source-col", "DonorID", "--ref-label", "Reference",
                "--test-label", "pancreas", "--methods", "duet,diffxpy,deseq2,wilcoxon"]),
        ("mast", [py, str(SCRIPTS / "run_mast_dataset.py"), "--h5ad", source,
                  "--results-dir", str(RESULTS / "de"), "--ref-label", "Reference",
                  "--test-label", "pancreas"]),
        ("null_single", [
            py, str(SCRIPTS / "null_calibration.py"), "--h5ad", source,
            "--source-col", "DonorID", "--celltypes",
            "Ductal cell,Endothelial cell,Stellate cell",
            "--tag", "pancreas_20260925", "--outdir", str(RESULTS / "null"),
        ]),
        ("pseudobulk_check", [
            py, str(SCRIPTS / "validate_pseudobulk_engines.py"),
            "--datasets", "pancreas", "--out", str(RESULTS / "pseudobulk_engine_agreement.csv"),
        ]),
        ("mast_settings", [
            py, str(SCRIPTS / "mast_settings_check.py"), "--datasets", "pancreas",
            "--outdir", str(RESULTS / "mast_components"),
            "--out", str(RESULTS / "mast_component_agreement.csv"),
        ]),
        ("underflow", [
            py, str(SCRIPTS / "underflow_check.py"), "--datasets", "pancreas",
            "--mast-dir", str(RESULTS / "mast_components"),
            "--outdir", str(RESULTS / "evidence"),
        ]),
        ("fair", bench + ["--inputs", ",".join(NAMES), "--engines", "duet,mast",
                           "--repeats", "5", "--tag", "pancreas_20260925_fair",
                           "--out", str(out / "fair.csv")]),
    ]
    for inp in ("dense", "sparse"):
        for eb in ("TRUE", "FALSE"):
            tag = f"mem_{inp}_eb{eb}"
            work.append((tag, bench + [
                "--inputs", all_name, "--engines", "mast", "--mast-input", inp,
                "--ebayes", eb, "--repeats", "3", "--tag", f"pancreas_20260925_{tag}",
                "--out", str(out / f"{tag}.csv"),
            ]))
    work.extend([
        ("real_cdr", bench + ["--inputs", all_name, "--engines", "duet,mast",
                              "--design", "cdr", "--repeats", "5",
                              "--tag", "pancreas_20260925_real_cdr",
                              "--out", str(out / "real_cdr.csv")]),
        ("real_cdr_matched", bench + [
            "--inputs", all_name, "--engines", "duet", "--design", "cdr",
            "--duet-driver", "bench_duet_run_matched.py", "--repeats", "5",
            "--tag", "pancreas_20260925_real_cdr_matched",
            "--out", str(out / "real_cdr_matched.csv"),
        ]),
        ("extra_methods", [py, str(SCRIPTS / "bench_pancreas_methods.py"),
                           "--outdir", str(RESULTS / "bench_extra"), "--shared"]),
    ])
    return work


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="actually start the selected analyses")
    ap.add_argument("--only", default=None, help="comma-separated step names; mast needs de first")
    ap.add_argument("--register-existing", action="store_true",
                    help="register validated outputs of the currently running pre-checkpoint job; no analyses")
    args = ap.parse_args()
    if args.run and args.register_existing:
        ap.error("--register-existing does not start analyses; use it without --run")
    manifest_path = INPUTS / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"prepare the inputs first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    for item in manifest["inputs"]:
        for kind in ("normalized", "counts"):
            if not Path(item[kind]).is_file():
                raise SystemExit(f"missing input: {item[kind]}")
    commands = steps(source)
    requested = set(args.only.split(",")) if args.only else {name for name, _ in commands}
    unknown = requested - {name for name, _ in commands}
    if unknown:
        ap.error(f"unknown steps: {sorted(unknown)}")
    all_commands = commands
    commands = [(name, cmd) for name, cmd in commands if name in requested]
    store = ResumeStore(RESULTS, manifest["source_sha256"])
    for name, cmd in commands:
        print(f"[{name}] {subprocess.list2cmdline(cmd)}")
        record = store.record(name)
        if record:
            print(f"[saved] {name}: {record['state']}")
    if not args.run and not args.register_existing:
        print("[plan] listing only; no analysis started")
        return
    if args.register_existing:
        verify_source(source, manifest)
        signatures = {name: fingerprint(cmd, manifest_path, HERE) for name, cmd in all_commands}
        store.register_existing(all_commands, signatures)
        print("[resume] existing-run plan saved; no analysis started", flush=True)
        return
    with execution_lock(RESULTS):
        verify_source(source, manifest)
        run_commands(commands, source, manifest_path, store)


def verify_source(source, manifest):
    print("[input] verifying source SHA-256...", flush=True)
    digest = hashlib.sha256()
    with open(source, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["source_sha256"]:
        raise SystemExit("source h5ad changed since the ductal inputs were prepared")


def run_commands(commands, source, manifest_path, store):
    env = dict(os.environ)
    tmp = HERE.parent / "_tmp"
    for key in ("TMP", "TEMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        env[key] = str(tmp)
    env["NUMBA_CACHE_DIR"] = str(tmp / "numba")
    env["MPLCONFIGDIR"] = str(tmp / "mpl")
    env["XDG_CACHE_HOME"] = str(tmp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["DUET_INPUTS"] = str(HERE.parent / "inputs")
    env["DUET_PANCREAS_H5AD"] = source
    env["DUET_PANCREAS_DONOR_COL"] = "DonorID"
    env["DUET_PANCREAS_CELLTYPES"] = "Ductal cell,Endothelial cell,Stellate cell"
    RESULTS.mkdir(parents=True, exist_ok=True)
    try:                                   # optional local progress monitor (not part of the repository)
        from pancreas_status import start_monitor
    except ImportError:
        pass
    else:
        start_monitor(os.getpid(), env, selected=[name for name, _ in commands])
    for name, cmd in commands:
        signature = fingerprint(cmd, manifest_path, HERE)
        try:
            store.run_step(name, signature, cmd, HERE, env)
        except subprocess.CalledProcessError as error:
            if error.returncode == STOPPED:
                print(f"[paused] {name}: completed measurements saved; rerun to resume", flush=True)
                raise SystemExit(STOPPED) from None
            raise
    print("[done] all selected steps complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[paused] interrupted; rerun the same command to resume", flush=True)
        raise SystemExit(130) from None
