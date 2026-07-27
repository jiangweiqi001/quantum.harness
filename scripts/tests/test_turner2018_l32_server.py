"""Focused tests for the staged Turner L=32 server workflow."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    main,
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
    def test_small_l_gate_detects_perturbed_independent_matrix(self):
        import turner2018_l32_server as server

        original = server.assemble_reduced_hamiltonian

        def perturbed(basis):
            matrix = original(basis).tolil()
            matrix[0, 1] += 1e-6
            matrix[1, 0] += 1e-6
            return matrix.tocsr()

        with mock.patch.object(server, "assemble_reduced_hamiltonian", perturbed):
            result = server.validate_small_l(10, official_data_dir=None)

        self.assertFalse(result["passed"])
        self.assertFalse(result["metrics"]["reduced_matrix"]["passed"])
        self.assertGreater(result["metrics"]["reduced_matrix"]["value"], 1e-11)

    def test_validate_local_cli_writes_failed_atomic_gate_and_returns_nonzero(self):
        failed = {
            "length": 12,
            "passed": False,
            "metrics": {
                "complete_spectrum": {
                    "value": 2e-10,
                    "tolerance": 1e-10,
                    "passed": False,
                }
            },
        }
        passed = {
            "length": 10,
            "passed": True,
            "metrics": {
                "complete_spectrum": {
                    "value": 0.0,
                    "tolerance": 1e-10,
                    "passed": True,
                }
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "turner2018_l32_server.validate_small_l",
                side_effect=[passed, failed],
            ):
                return_code = main(
                    [
                        "--validate-local",
                        "10",
                        "12",
                        "--output-dir",
                        directory,
                    ]
                )

            summary_path = Path(directory) / "validation" / "local-equivalence.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual(return_code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["requested_lengths"], [10, 12])
            self.assertFalse(summary["results"][1]["metrics"]["complete_spectrum"]["passed"])
            self.assertIn("artifact_hashes", summary)
            self.assertIn("provenance", summary)
            self.assertFalse(summary_path.with_name(summary_path.name + ".partial").exists())

    def test_validate_local_cli_runs_every_requested_even_length(self):
        lengths = [10, 12, 14, 16, 18, 20]

        def passing(length, official_data_dir):
            return {"length": length, "passed": True, "metrics": {}}

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "turner2018_l32_server.validate_small_l",
                side_effect=passing,
            ) as validate:
                return_code = main(
                    [
                        "--validate-local",
                        *(str(length) for length in lengths),
                        "--output-dir",
                        directory,
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(
                [call.args[0] for call in validate.call_args_list],
                lengths,
            )
            summary = json.loads(
                (
                    Path(directory) / "validation" / "local-equivalence.json"
                ).read_text()
            )
            self.assertEqual(summary["status"], "passed")
            self.assertTrue(summary["passed"])


if __name__ == "__main__":
    unittest.main()
