"""Validated, atomic step checkpoints for the versioned pancreas analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CELLTYPES = {"Ductal cell", "Endothelial cell", "Stellate cell"}
STOPPED = 99


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return str(value).lower() in {"true", "1", "1.0"}


def fingerprint(command, manifest_path, repo):
    """Tie checkpoints to the exact input manifest, command and analysis code."""
    repo = Path(repo)
    # Include the analysis helpers and R drivers, but not the launcher/checkpoint
    # code: improving resume handling must not invalidate scientific results.
    primary = Path(command[1])
    helpers = {
        "run_mast_dataset.py": ["bench_mast_run.R"],
        "null_calibration.py": ["run_de_comparison.py"],
        "validate_pseudobulk_engines.py": ["donor_swap_partitions.py"],
        "mast_settings_check.py": ["donor_swap_partitions.py", "mast_components.R"],
        "underflow_check.py": ["donor_swap_partitions.py", "run_de_comparison.py",
                               "mast_settings_check.py", "mast_components.R"],
        "bench_harness.py": ["bench_duet_run.py", "bench_duet_run_matched.py",
                             "bench_mast_steps.R"],
        "bench_pancreas_methods.py": ["bench_pancreas_methods_run.py", "bench_harness.py"],
    }
    files = sorted((repo / "duet").glob("*.py"))
    files += [primary, repo / "scripts" / "_runtime.py"]
    files += [repo / "scripts" / helper for helper in helpers.get(primary.name, [])]
    payload = dict(command=command, manifest_sha256=sha256(manifest_path),
                   code={str(p.relative_to(repo)): sha256(p) for p in files})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def outputs(name, root):
    root = Path(root)
    mapping = {
        "de": [f"de/de_{method}_all_celltypes.csv" for method in
               ("DUET", "diffxpy", "DESeq2", "wilcoxon")] + [
                   "de/comparison_per_gene.csv", "de/comparison_concordance.csv",
                   "de/consensus_genes.csv", "de/comparison_summary.txt"],
        "mast": ["de/de_MAST_all_celltypes.csv", "de/mast_timings.csv"],
        "null_single": ["null/null_pancreas_20260925.csv",
                        "null/null_pancreas_20260925_matched.csv"],
        "pseudobulk_check": ["pseudobulk_engine_agreement.csv"],
        "mast_settings": ["mast_component_agreement.csv"],
        "underflow": ["evidence/underflow_by_package.csv", "evidence/underflow_topk.csv"],
        "fair": ["bench/fair.csv"],
        "real_cdr": ["bench/real_cdr.csv"],
        "real_cdr_matched": ["bench/real_cdr_matched.csv"],
        "extra_methods": ["bench_extra/measurements.csv", "bench_extra/protocol.json"],
    }
    if name.startswith("mem_"):
        return [root / "bench" / f"{name}.csv"]
    return [root / p for p in mapping[name]]


def validate_outputs(name, root, command):
    """Exit code zero is insufficient: some scientific scripts catch failures."""
    paths = outputs(name, root)
    if any(not p.is_file() or not p.stat().st_size for p in paths):
        return False
    try:
        if name in {"de", "mast"}:
            for path in paths[:4] if name == "de" else paths[:1]:
                rows = read_csv(path)
                if any(r.get("skip_reason", "").startswith("failed:") for r in rows):
                    return False
                tested = {r.get("celltype") for r in rows if truth(r.get("tested"))}
                if tested != CELLTYPES or any(r.get("comparison") != "Reference_vs_pancreas"
                                             for r in rows):
                    return False
            if name == "mast":
                return {r["celltype"] for r in read_csv(paths[1])} == CELLTYPES
            return "DONE. Outputs in" in (Path(root) / "de/run.log").read_text(encoding="utf-8")
        if name == "null_single":
            expected = {(ct, design, method) for ct in CELLTYPES
                        for design in ("permutation", "donor_swap")
                        for method in ("DUET", "wilcoxon", "DESeq2-pseudobulk")}
            for path, count_col in zip(paths, ("n_tested", "n_common")):
                rows = read_csv(path)
                actual = {(r["celltype"], r["design"], r["method"]) for r in rows
                          if float(r[count_col]) > 0 and r["dataset"] == "pancreas_20260925"}
                if actual != expected:
                    return False
            return True
        if name == "pseudobulk_check":
            rows = read_csv(paths[0])
            return {r["celltype"] for r in rows if r["dataset"] == "pancreas"
                    and r["design"] == "~condition" and "aggregation_identical" in r} == CELLTYPES
        if name == "mast_settings":
            rows = read_csv(paths[0])
            actual = {(r["mast_method"], truth(r["mast_ebayes"]), r["component"])
                      for r in rows if r["dataset"] == "pancreas"}
            components = {"stat_detect", "stat_continuous", "stat_hurdle", "df_hurdle",
                          "coef_detect", "coef_continuous", "logFC", "neglog10_p",
                          "neglog10_fdr", "sig_calls_jaccard"}
            return actual == {(method, eb, component) for method, eb in
                              (("bayesglm", True), ("glm", True), ("glm", False))
                              for component in components}
        if name == "underflow":
            expected = {"DUET", "Scanpy wilcoxon", "diffxpy", "MAST"}
            return all({r["package"] for r in read_csv(p) if r["dataset"] == "pancreas"}
                       == expected for p in paths)
        # Benchmarks require the full number of successful, measured repetitions.
        if name == "extra_methods":
            protocol = json.loads(paths[1].read_text(encoding="utf-8"))
            names, engines, repeats = protocol["inputs"], protocol["engines"], protocol["repeats"]
        else:
            names = command[command.index("--inputs") + 1].split(",")
            engines = command[command.index("--engines") + 1].split(",")
            repeats = int(command[command.index("--repeats") + 1])
        rows = read_csv(paths[0])
        for input_name in names:
            for engine in engines:
                successful = {r["run"] for r in rows if r.get("stage") == "TOTAL_RUN"
                              and r.get("input") == input_name and r.get("engine") == engine
                              and not truth(r.get("warmup")) and float(r["returncode"]) == 0
                              and float(r.get("sampler_errors") or 0) == 0}
                if len(successful) < repeats:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class ResumeStore:
    def __init__(self, root, source_sha256):
        self.root = Path(root)
        self.directory = self.root / "_progress"
        self.source = source_sha256

    def record(self, name):
        path = self.directory / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def save(self, name, signature, state, **extra):
        atomic_json(self.directory / f"{name}.json", dict(
            step=name, fingerprint=signature, source_sha256=self.source,
            state=state, updated=now(), **extra))

    def complete(self, name, signature, adopted=False):
        artifacts = {str(p.relative_to(self.root)): sha256(p) for p in outputs(name, self.root)}
        self.save(name, signature, "complete", artifacts=artifacts, adopted=adopted)

    def ready(self, name, signature, command):
        record = self.record(name)
        if record:
            if record["fingerprint"] != signature or record["source_sha256"] != self.source:
                raise RuntimeError(f"{name}: input, settings or analysis code changed; "
                                   "existing results cannot be resumed with this configuration")
            if record["state"] == "complete":
                if all((self.root / p).is_file() and sha256(self.root / p) == digest
                       for p, digest in record["artifacts"].items()):
                    return True
                raise RuntimeError(f"{name}: a saved completed output is missing or changed; "
                                   "restore it before resuming")
            return False
        # Migrate the already running pre-checkpoint job, using a code/input
        # snapshot registered during that job and strict output coverage checks.
        legacy = self.directory / "legacy_plan.json"
        if legacy.is_file():
            plan = json.loads(legacy.read_text(encoding="utf-8"))
            if (plan["source_sha256"] == self.source
                    and plan["steps"].get(name) == signature
                    and validate_outputs(name, self.root, command)):
                self.complete(name, signature, adopted=True)
                print(f"[resume] {name}: validated existing results registered", flush=True)
                return True
        return False

    def register_existing(self, commands, signatures):
        path = self.directory / "legacy_plan.json"
        plan = dict(source_sha256=self.source, steps=signatures, registered=now())
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old["source_sha256"] != self.source or old["steps"] != signatures:
                raise RuntimeError("the registered legacy run has a different input or analysis code")
        else:
            atomic_json(path, plan)
        for name, command in commands:
            if self.ready(name, signatures[name], command):
                print(f"[resume] {name}: complete", flush=True)

    def run_step(self, name, signature, command, cwd, env):
        if self.ready(name, signature, command):
            print(f"[skip] {name}: already complete", flush=True)
            return
        self.save(name, signature, "running", command=command, started=now())
        print(f"[run] {name}", flush=True)
        try:
            child = subprocess.Popen(command, cwd=cwd, env=env)
            try:
                returncode = child.wait()
            except KeyboardInterrupt:
                # Also stop descendants such as Rscript's R process. The next
                # launcher must not race a surviving child writing the same files.
                try:
                    import psutil                               # optional dependency
                except ImportError:
                    psutil = None
                if psutil is not None:
                    try:
                        tree = psutil.Process(child.pid).children(recursive=True)
                        for process in reversed(tree):
                            try:
                                process.kill()
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                child.kill()
                child.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            if not validate_outputs(name, self.root, command):
                raise RuntimeError(f"{name}: output is incomplete; step was not marked complete")
            self.complete(name, signature)
            print(f"[done] {name}: checkpoint saved", flush=True)
        except BaseException as error:
            stopped = (isinstance(error, KeyboardInterrupt) or
                       isinstance(error, subprocess.CalledProcessError) and error.returncode == STOPPED)
            self.save(name, signature, "interrupted" if stopped
                      else "failed", error=str(error))
            raise


@contextmanager
def execution_lock(root):
    """OS lock releases automatically even if the process is killed (msvcrt on Windows, flock elsewhere)."""
    if os.name == "nt":
        import msvcrt

        def lock(stream):
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

        def unlock(stream):
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        def lock(stream):
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def unlock(stream):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    directory = Path(root) / "_progress"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as stream:
        if not stream.seek(0, 2):
            stream.write(b"0")
            stream.flush()
        try:
            lock(stream)
        except OSError:
            raise RuntimeError("another pancreas analysis runner is already active") from None
        try:
            # The old runner predates the lock. Detect it too, when its command
            # line is accessible (same-user processes on the desktop); needs psutil.
            try:
                import psutil                                   # optional dependency
            except ImportError:
                psutil = None
            for process in (psutil.process_iter() if psutil is not None else []):
                if process.pid == os.getpid():
                    continue
                try:
                    args = process.cmdline()
                    if (any(Path(arg).name == "pancreas_20260925.py" for arg in args)
                            and "--run" in args):
                        raise RuntimeError(f"pancreas runner PID {process.pid} is still active; "
                                           "stop that run before starting another")
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            yield
        finally:
            unlock(stream)
