"""Resume/stop regression tests; no scientific analyses are launched."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pancreas_resume import (CELLTYPES, ResumeStore, atomic_json, execution_lock,
                             outputs, validate_outputs)


def patch_psutil(target, **kwargs):
    """psutil is an optional dependency: patch it only where it is installed."""
    return patch(target, **kwargs) if find_spec("psutil") else nullcontext()


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[2]
        tmp = workspace / "_tmp"
        tmp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=tmp, prefix="resume_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ResumeStore(self.root, "source-hash")
        self.command = [sys.executable, "fake-analysis.py"]

    def finish_de(self):
        rows = [dict(celltype=ct, comparison="Reference_vs_pancreas", gene="G1",
                     tested=True, skip_reason="") for ct in sorted(CELLTYPES)]
        for path in outputs("de", self.root):
            if path.name.startswith("de_"):
                csv_file(path, rows)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("completed output\n", encoding="utf-8")
        (self.root / "de/run.log").write_text("DONE. Outputs in test\n", encoding="utf-8")

    def success(self, *args, **kwargs):
        self.finish_de()
        return Mock(wait=Mock(return_value=0))

    def test_completed_step_is_never_spawned_again(self):
        with patch("pancreas_resume.subprocess.Popen", side_effect=self.success) as spawn:
            self.store.run_step("de", "config", self.command, self.root, {})
            self.store.run_step("de", "config", self.command, self.root, {})
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(self.store.record("de")["state"], "complete")

    def test_failed_process_is_retried_on_restart(self):
        with patch("pancreas_resume.subprocess.Popen", return_value=Mock(wait=Mock(return_value=1))):
            with self.assertRaises(subprocess.CalledProcessError):
                self.store.run_step("de", "config", self.command, self.root, {})
        self.assertEqual(self.store.record("de")["state"], "failed")
        with patch("pancreas_resume.subprocess.Popen", side_effect=self.success):
            self.store.run_step("de", "config", self.command, self.root, {})
        self.assertEqual(self.store.record("de")["state"], "complete")

    def test_interrupted_process_is_retried_and_child_is_stopped(self):
        child = Mock(pid=123456, wait=Mock(side_effect=[KeyboardInterrupt, 0]))
        with patch("pancreas_resume.subprocess.Popen", return_value=child), \
                patch_psutil("psutil.Process", return_value=Mock(children=Mock(return_value=[]))):
            with self.assertRaises(KeyboardInterrupt):
                self.store.run_step("de", "config", self.command, self.root, {})
        child.kill.assert_called_once()
        self.assertEqual(self.store.record("de")["state"], "interrupted")
        with patch("pancreas_resume.subprocess.Popen", side_effect=self.success):
            self.store.run_step("de", "config", self.command, self.root, {})
        self.assertEqual(self.store.record("de")["state"], "complete")

    def test_zero_exit_with_partial_outputs_is_not_complete(self):
        with patch("pancreas_resume.subprocess.Popen", return_value=Mock(wait=Mock(return_value=0))):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.store.run_step("de", "config", self.command, self.root, {})
        self.assertEqual(self.store.record("de")["state"], "failed")

    def test_existing_complete_job_is_adopted_without_spawning(self):
        self.finish_de()
        self.store.register_existing([("de", self.command)], {"de": "config"})
        with patch("pancreas_resume.subprocess.Popen") as spawn:
            self.store.run_step("de", "config", self.command, self.root, {})
        spawn.assert_not_called()
        self.assertTrue(self.store.record("de")["adopted"])

    def test_existing_partial_or_failed_method_is_not_adopted(self):
        self.finish_de()
        csv_file(outputs("de", self.root)[0], [dict(celltype="Ductal cell",
                 comparison="Reference_vs_pancreas", tested=False, skip_reason="failed:ValueError")])
        self.store.register_existing([("de", self.command)], {"de": "config"})
        self.assertFalse(self.store.ready("de", "config", self.command))
        self.assertIsNone(self.store.record("de"))

    def test_changed_configuration_is_rejected(self):
        self.finish_de()
        self.store.complete("de", "original")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.store.ready("de", "different", self.command)

    def test_changed_completed_output_is_rejected(self):
        self.finish_de()
        self.store.complete("de", "config")
        outputs("de", self.root)[0].write_text("truncated", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            self.store.ready("de", "config", self.command)

    def test_partial_benchmark_and_duplicate_run_do_not_count_as_complete(self):
        command = self.command + ["--inputs", "a,b", "--engines", "duet,mast", "--repeats", "2"]
        rows = [dict(input=name, engine=engine, stage="TOTAL_RUN", warmup=False,
                     returncode=0, run=1) for name in ("a", "b") for engine in ("duet", "mast")]
        path = outputs("fair", self.root)[0]
        csv_file(path, rows + rows)
        self.assertFalse(validate_outputs("fair", self.root, command))
        csv_file(path, rows + [dict(row, run=2) for row in rows])
        self.assertTrue(validate_outputs("fair", self.root, command))

    def test_underflow_package_labels_match_analysis_outputs(self):
        rows = [dict(dataset="pancreas", package=package) for package in
                ("DUET", "Scanpy wilcoxon", "diffxpy", "MAST")]
        for path in outputs("underflow", self.root):
            csv_file(path, rows)
        self.assertTrue(validate_outputs("underflow", self.root, self.command))
        csv_file(outputs("underflow", self.root)[1], rows[:-1])
        self.assertFalse(validate_outputs("underflow", self.root, self.command))

    def test_failed_atomic_replace_keeps_previous_checkpoint(self):
        path = self.root / "checkpoint.json"
        atomic_json(path, {"state": "complete"})
        with patch.object(Path, "replace", side_effect=OSError("interrupted write")):
            with self.assertRaises(OSError):
                atomic_json(path, {"state": "running"})
        self.assertEqual(json.loads(path.read_text())["state"], "complete")

    def test_second_runner_cannot_acquire_active_lock(self):
        with patch_psutil("psutil.process_iter", return_value=[]):
            with execution_lock(self.root):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with execution_lock(self.root):
                        self.fail("second runner acquired the lock")


if __name__ == "__main__":
    unittest.main()
