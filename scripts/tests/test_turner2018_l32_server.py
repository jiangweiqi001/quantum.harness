"""Focused tests for the staged Turner L=32 server workflow."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from turner2018_l32_server import (  # noqa: E402
    FULL_FSA_SHELLS,
    L32_FULL_DIMENSION,
    L32_SECTOR_DIMENSION,
    SYMMETRY_FOLDED_FSA_SHELLS,
    atomic_write_json,
    dense_resource_estimate,
    require_stage,
)


class TurnerL32PlanTests(unittest.TestCase):
    def test_known_dimensions_and_dense_memory_estimate(self):
        estimate = dense_resource_estimate(L32_SECTOR_DIMENSION)

        self.assertEqual(L32_FULL_DIMENSION, 4_870_847)
        self.assertEqual(L32_SECTOR_DIMENSION, 77_436)
        self.assertEqual(FULL_FSA_SHELLS, 33)
        self.assertEqual(SYMMETRY_FOLDED_FSA_SHELLS, 17)
        self.assertEqual(estimate["bytes_per_dense_array"], 47_970_672_768)
        self.assertAlmostEqual(estimate["gb_per_dense_array"], 47.970672768)

    def test_atomic_json_write_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.json"
            atomic_write_json(target, {"status": "complete"})

            self.assertEqual(json.loads(target.read_text()), {"status": "complete"})
            self.assertFalse(target.with_name("manifest.json.partial").exists())

    def test_restart_guard_requires_completed_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "basis.*not complete"):
                require_stage(root, "basis")

            atomic_write_json(
                root / "stages" / "basis.json",
                {"stage": "basis", "status": "complete"},
            )
            manifest = require_stage(root, "basis")
            self.assertEqual(manifest["stage"], "basis")

    def test_plan_cli_writes_fail_closed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "turner2018_l32_server.py"),
                    "--stage",
                    "plan",
                    "--length",
                    "32",
                    "--dry-run",
                    "--output-dir",
                    directory,
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["readiness"], "blocked")
            self.assertEqual(manifest["basis"]["sector_dimension"], 77_436)
            self.assertEqual(manifest["solver"]["driver"], "evd")
            self.assertFalse(manifest["solver"]["check_finite"])
            self.assertTrue(manifest["solver"]["overwrite_a"])
            self.assertEqual(manifest["artifacts"]["hamiltonian"], "hamiltonian.csr.npz")
            self.assertEqual(manifest["artifacts"]["results"], "results.h5")

    def test_l32_basis_refuses_without_local_quspin_validation_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "turner2018_l32_server.py"),
                    "--stage",
                    "basis",
                    "--length",
                    "32",
                    "--output-dir",
                    directory,
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validated QuSpin 1.0.1 imported-basis proof", result.stderr)


class TurnerL32SlurmTests(unittest.TestCase):
    def test_qdagnormal_script_has_exact_required_resources_and_no_secrets(self):
        text = (SCRIPTS / "turner2018_l32_qdagnormal.sbatch").read_text()

        for directive in (
            "#SBATCH --partition=qdagnormal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=64",
            "#SBATCH --mem=512G",
            "#SBATCH --time=12:00:00",
            "#SBATCH --gres=gpu:A800:1",
        ):
            self.assertIn(directive, text)
        self.assertIn("OPENBLAS_NUM_THREADS=64", text)
        self.assertIn("MKL_NUM_THREADS=64", text)
        self.assertIn("TURNER_OFFLINE_IMAGE", text)
        self.assertIn("TURNER_OUTPUT_DIR", text)
        self.assertNotRegex(text, r"(?i)(password|api[_-]?key|access[_-]?token)\s*=")

    def test_documented_submission_is_test_only(self):
        text = (REPO / "tracks" / "ed" / "README.md").read_text()
        expected = (
            "scripts/harness_slurm.sh --profile "
            "skills/using-slurm/profiles/qdeshell.toml submit --test-only "
            "--script scripts/turner2018_l32_qdagnormal.sbatch"
        )
        self.assertIn(expected, text)
        self.assertNotIn(
            "submit --script scripts/turner2018_l32_qdagnormal.sbatch",
            text,
        )


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
    "local NumPy/SciPy stack is not installed",
)
class TurnerSmallLEquivalenceTests(unittest.TestCase):
    def test_small_l_reference_stage_matches_existing_implementation(self):
        from turner2018_l32_server import validate_small_l

        metrics = validate_small_l(10, official_data_dir=None)
        self.assertLess(metrics["sector_matrix_max_abs"], 1e-12)
        self.assertLess(metrics["eigenvalue_max_abs"], 1e-12)
        self.assertLess(metrics["overlap_max_abs"], 1e-12)
        self.assertLess(metrics["fsa_beta_max_abs"], 1e-12)
        self.assertEqual(metrics["full_fsa_shell_count"], 11)
        self.assertEqual(metrics["folded_fsa_shell_count"], 6)


if __name__ == "__main__":
    unittest.main()
