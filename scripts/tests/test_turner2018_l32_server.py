"""Focused tests for the staged Turner L=32 server workflow."""

from __future__ import annotations

import importlib.util
import json
import os
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
    validation_lengths = [10, 12, 14, 16, 18, 20]

    @staticmethod
    def _passing_result(length):
        return {"length": length, "passed": True, "metrics": {}}

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
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "turner2018_l32_server.validate_small_l",
                side_effect=[
                    self._passing_result(10),
                    failed,
                    *(self._passing_result(length) for length in (14, 16, 18, 20)),
                ],
            ):
                return_code = main(
                    [
                        "--validate-local",
                        *(str(length) for length in self.validation_lengths),
                        "--output-dir",
                        directory,
                    ]
                )

            summary_path = Path(directory) / "validation" / "local-equivalence.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual(return_code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["requested_lengths"], self.validation_lengths)
            self.assertFalse(summary["results"][1]["metrics"]["complete_spectrum"]["passed"])
            self.assertIn("artifact_hashes", summary)
            self.assertIn("provenance", summary)
            self.assertFalse(summary_path.with_name(summary_path.name + ".partial").exists())

    def test_validate_local_cli_runs_every_requested_even_length(self):
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
                        *(str(length) for length in self.validation_lengths),
                        "--output-dir",
                        directory,
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(
                [call.args[0] for call in validate.call_args_list],
                self.validation_lengths,
            )
            summary = json.loads(
                (
                    Path(directory) / "validation" / "local-equivalence.json"
                ).read_text()
            )
            self.assertEqual(summary["status"], "passed")
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["requested_lengths"], self.validation_lengths)
            self.assertEqual(
                [result["length"] for result in summary["results"]],
                self.validation_lengths,
            )

    def test_validate_local_rejects_noncanonical_length_sets_before_work(self):
        invalid_length_lists = [
            [10, 12, 14, 16, 18],
            [10, 12, 14, 16, 18, 18],
            [10, 12, 14, 16, 18, 20, 22],
            [12, 10, 14, 16, 18, 20],
        ]
        for lengths in invalid_length_lists:
            with self.subTest(lengths=lengths):
                with mock.patch(
                    "turner2018_l32_server.run_local_validation",
                    side_effect=AssertionError("scientific work must not start"),
                ) as run:
                    with self.assertRaisesRegex(SystemExit, "exactly.*10.*20"):
                        main(["--validate-local", *(str(value) for value in lengths)])
                run.assert_not_called()

    def test_validate_local_rejects_conflicting_execution_modes_before_work(self):
        conflicts = [
            ["--stage", "plan"],
            ["--dry-run"],
            ["--length", "20"],
            ["--chunk-columns", "8"],
            ["--validate-small-l", "10"],
        ]
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                with mock.patch(
                    "turner2018_l32_server.run_local_validation",
                    side_effect=AssertionError("scientific work must not start"),
                ) as run:
                    with self.assertRaises(SystemExit) as raised:
                        main(
                            [
                                "--validate-local",
                                *(str(value) for value in self.validation_lengths),
                                *conflict,
                            ]
                        )
                    if conflict[0] != "--stage":
                        self.assertIn("mutually exclusive", str(raised.exception))
                run.assert_not_called()

    def test_validate_local_rejects_equals_form_execution_options_before_work(self):
        conflicts = [
            ["--stage=plan"],
            ["--length=32"],
            ["--chunk-columns=32"],
            ["--validate-small-l=10"],
            ["--official-data-dir=/tmp/official"],
        ]
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                with mock.patch(
                    "turner2018_l32_server.run_local_validation",
                    side_effect=AssertionError("scientific work must not start"),
                ) as run:
                    with self.assertRaises(SystemExit):
                        main(
                            [
                                "--validate-local",
                                *(str(value) for value in self.validation_lengths),
                                *conflict,
                            ]
                        )
                run.assert_not_called()

    def test_argparse_abbreviations_are_rejected_before_work(self):
        with mock.patch(
            "turner2018_l32_server.run_local_validation",
            side_effect=AssertionError("scientific work must not start"),
        ) as run:
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--validate-loc",
                        *(str(value) for value in self.validation_lengths),
                    ]
                )
        run.assert_not_called()

    def test_running_marker_replaces_stale_pass_before_scientific_work(self):
        import turner2018_l32_server as server

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "validation" / "local-equivalence.json"
            atomic_write_json(path, {"status": "passed", "passed": True})
            observed = []

            def validate(length, official_data_dir):
                observed.append(json.loads(path.read_text()))
                return self._passing_result(length)

            with mock.patch.object(server, "validate_small_l", side_effect=validate):
                server.run_local_validation(
                    self.validation_lengths,
                    output,
                    None,
                    ["--validate-local"],
                )

            self.assertTrue(observed)
            self.assertEqual(observed[0]["status"], "running")
            self.assertFalse(observed[0]["passed"])

    def test_source_hash_failure_publishes_minimal_failed_summary(self):
        import turner2018_l32_server as server

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                mock.patch.object(
                    server,
                    "validate_small_l",
                    side_effect=lambda length, _: self._passing_result(length),
                ),
                mock.patch.object(
                    server,
                    "_source_hashes",
                    side_effect=OSError("hash injection"),
                ),
            ):
                _path, summary = server.run_local_validation(
                    self.validation_lengths, output, None, ["--validate-local"]
                )

            persisted = json.loads(
                (output / "validation" / "local-equivalence.json").read_text()
            )
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(persisted["status"], "failed")
            self.assertFalse(persisted["passed"])
            self.assertIn("hash injection", persisted["error"])

    def test_real_git_failure_publishes_minimal_failed_summary(self):
        import turner2018_l32_server as server

        git_results = [
            subprocess.CompletedProcess(
                args=["git"],
                returncode=128,
                stdout="",
                stderr="fatal: injected git failure",
            ),
            subprocess.CompletedProcess(
                args=["git"],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]
        for git_result in git_results:
            with self.subTest(
                returncode=git_result.returncode,
                stdout=git_result.stdout,
            ):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    with (
                        mock.patch.object(
                            server.subprocess,
                            "run",
                            return_value=git_result,
                        ),
                        mock.patch.object(
                            server,
                            "validate_small_l",
                            side_effect=AssertionError(
                                "scientific work must not start"
                            ),
                        ) as validate,
                    ):
                        _path, summary = server.run_local_validation(
                            self.validation_lengths,
                            output,
                            None,
                            ["--validate-local"],
                        )

                    persisted = json.loads(
                        (
                            output / "validation" / "local-equivalence.json"
                        ).read_text()
                    )
                    self.assertEqual(summary["status"], "failed")
                    self.assertEqual(summary["failure_phase"], "provenance")
                    self.assertIn("git revision", summary["error"])
                    self.assertEqual(persisted, summary)
                    validate.assert_not_called()

    def test_final_publication_failure_leaves_running_marker(self):
        import turner2018_l32_server as server

        original = server.atomic_write_json
        calls = 0

        def fail_final(path, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("final publication injection")
            original(path, payload)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                mock.patch.object(
                    server,
                    "validate_small_l",
                    side_effect=lambda length, _: self._passing_result(length),
                ),
                mock.patch.object(server, "atomic_write_json", side_effect=fail_final),
            ):
                with self.assertRaisesRegex(OSError, "final publication injection"):
                    server.run_local_validation(
                        self.validation_lengths, output, None, ["--validate-local"]
                    )

            persisted = json.loads(
                (output / "validation" / "local-equivalence.json").read_text()
            )
            self.assertEqual(calls, 2)
            self.assertEqual(persisted["status"], "running")
            self.assertFalse(persisted["passed"])

    def test_initial_marker_failure_is_not_reported_as_recovered(self):
        import turner2018_l32_server as server

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                mock.patch.object(
                    server,
                    "atomic_write_json",
                    side_effect=OSError("marker publication injection"),
                ),
                mock.patch.object(server, "_source_hashes") as hashes,
            ):
                with self.assertRaisesRegex(OSError, "marker publication injection"):
                    server.run_local_validation(
                        self.validation_lengths, output, None, ["--validate-local"]
                    )
            hashes.assert_not_called()

    def test_fsa_shape_metrics_record_exact_candidate_and_reference_shapes(self):
        from turner2018_l32_server import validate_small_l

        result = validate_small_l(10, official_data_dir=None)
        metrics = result["metrics"]
        self.assertEqual(
            metrics["fsa_beta_shape"],
            {
                "expected": [10],
                "candidate": [10],
                "reference": [10],
                "passed": True,
            },
        )
        self.assertEqual(
            metrics["fsa_projected_shell_shape"]["expected"],
            [6, 14],
        )
        self.assertTrue(metrics["fsa_projected_shell_shape"]["passed"])
        self.assertEqual(
            metrics["fsa_reduced_hamiltonian_shape"]["expected"],
            [6, 6],
        )
        self.assertTrue(metrics["fsa_reduced_hamiltonian_shape"]["passed"])

    def test_fsa_shape_mismatch_fails_without_broadcasting(self):
        import turner2018_ed_observables as observables
        from turner2018_l32_server import validate_small_l

        original = observables.compute_observables

        def wrong_beta_shape(*args, **kwargs):
            result = original(*args, **kwargs)
            result["fsa_beta_full_chain"] = result["fsa_beta_full_chain"][:-1]
            return result

        with mock.patch.object(
            observables,
            "compute_observables",
            side_effect=wrong_beta_shape,
        ):
            result = validate_small_l(10, official_data_dir=None)

        self.assertFalse(result["passed"])
        self.assertFalse(result["metrics"]["fsa_beta_shape"]["passed"])
        self.assertFalse(result["metrics"]["fsa_beta"]["passed"])

    def test_git_revision_is_resolved_from_script_repository_outside_cwd(self):
        import turner2018_l32_server as server

        expected = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                actual = server._git_revision()
            finally:
                os.chdir(previous)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
