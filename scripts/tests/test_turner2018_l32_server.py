"""Focused tests for the staged Turner L=32 server workflow."""

from __future__ import annotations

import importlib.util
import hashlib
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
    build_execution_fingerprint,
    build_plan,
    dense_resource_estimate,
    main,
    parse_memory_bytes,
    resolve_declared_memory_bytes,
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

    def test_plan_has_dense_estimates_for_all_production_lengths(self):
        for length, dimension in ((28, 13_201), (30, 31_836), (32, 77_436)):
            with self.subTest(length=length):
                plan = build_plan(length, Path("."), [])
                self.assertEqual(plan["basis"]["sector_dimension"], dimension)
                self.assertGreater(plan["resources"]["minimum_requested_bytes"], 0)

    def test_execution_fingerprint_is_exact_and_checkout_scoped(self):
        fingerprint = build_execution_fingerprint()
        self.assertEqual(fingerprint["python"], [3, 12])
        self.assertEqual(set(fingerprint["packages"]), {"numpy", "scipy", "h5py"})
        self.assertIn("turner2018_ed_artifacts.py", fingerprint["sources"])
        self.assertEqual(len(fingerprint["uv_lock_sha256"]), 64)

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

            main(
                [
                    "--stage",
                    "basis",
                    "--length",
                    "10",
                    "--output-dir",
                    str(root),
                ]
            )
            manifest = require_stage(root, "basis")
            self.assertEqual(manifest["stage"], "basis")

    def test_plan_cli_reports_native_l32_readiness(self):
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
            self.assertEqual(manifest["readiness"], "ready")
            self.assertEqual(manifest["basis"]["sector_dimension"], 77_436)
            self.assertEqual(manifest["solver"]["driver"], "evd")
            self.assertFalse(manifest["solver"]["check_finite"])
            self.assertTrue(manifest["solver"]["overwrite_a"])
            self.assertEqual(
                manifest["resources"]["minimum_requested_bytes"],
                239_853_363_840,
            )
            self.assertEqual(manifest["artifacts"]["hamiltonian"], "hamiltonian.csr.npz")
            self.assertEqual(manifest["artifacts"]["eigensystem"], "eigensystem.h5")
            self.assertEqual(manifest["artifacts"]["observables"], "observables.h5")

    def test_l32_basis_has_no_unconditional_quspin_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "turner2018_l32_server.build_orbit_basis",
                side_effect=RuntimeError("native basis reached"),
            ):
                with self.assertRaisesRegex(RuntimeError, "native basis reached"):
                    main(
                        [
                            "--stage",
                            "basis",
                            "--length",
                            "32",
                            "--output-dir",
                            directory,
                        ]
                    )

    def test_memory_parser_accepts_slurm_units_and_per_cpu(self):
        self.assertEqual(parse_memory_bytes("512G"), 512 * 2**30)
        self.assertEqual(parse_memory_bytes("512GB"), 512 * 10**9)
        self.assertEqual(parse_memory_bytes("524288M"), 524288 * 2**20)
        self.assertEqual(parse_memory_bytes("524288"), 524288 * 2**20)
        self.assertEqual(
            resolve_declared_memory_bytes(
                {"SLURM_MEM_PER_CPU": "8192", "SLURM_CPUS_PER_TASK": "64"}
            ),
            8192 * 64 * 2**20,
        )

    def test_scheduler_memory_caps_cli_overclaim(self):
        self.assertEqual(
            resolve_declared_memory_bytes(
                {"SLURM_MEM_PER_NODE": "1024"},
                "2G",
            ),
            1024 * 2**20,
        )

    def test_scheduler_memory_rejects_malformed_cpu_counts(self):
        for environment in (
            {"SLURM_MEM_PER_CPU": "1024", "SLURM_CPUS_PER_TASK": "bad"},
            {"SLURM_MEM_PER_CPU": "1024", "SLURM_CPUS_ON_NODE": "0"},
            {"SLURM_MEM_PER_CPU": "bogus", "SLURM_CPUS_PER_TASK": "2"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaises((ValueError, RuntimeError)):
                    resolve_declared_memory_bytes(environment)


@unittest.skipUnless(
    importlib.util.find_spec("numpy")
    and importlib.util.find_spec("scipy")
    and importlib.util.find_spec("h5py"),
    "local NumPy/SciPy/h5py stack is not installed",
)
class TurnerRestartWorkflowTests(unittest.TestCase):
    def run_stage(self, output: Path, stage: str, *extra: str) -> tuple[int, str]:
        stdout = []
        with mock.patch("builtins.print", side_effect=lambda *args, **_: stdout.append(" ".join(map(str, args)))):
            code = main(
                [
                    "--stage",
                    stage,
                    "--length",
                    "10",
                    "--output-dir",
                    str(output),
                    "--declared-memory",
                    "1G",
                    "--chunk-columns",
                    "4",
                    *extra,
                ]
            )
        return code, "\n".join(stdout)

    def test_all_computational_stages_restart_and_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            code, first = self.run_stage(output, "all")
            self.assertEqual(code, 0)
            self.assertIn("completed stage=validate", first)
            self.assertFalse((output / "stages" / "figures.json").exists())

            mtimes = {
                path.name: path.stat().st_mtime_ns
                for path in (
                    output / "basis.npz",
                    output / "hamiltonian.csr.npz",
                    output / "eigensystem.h5",
                    output / "observables.h5",
                    output / "validation" / "metrics.json",
                )
            }
            code, restarted = self.run_stage(output, "all")
            self.assertEqual(code, 0)
            for stage in ("plan", "basis", "hamiltonian", "diagonalize", "observables", "validate"):
                self.assertIn(f"skipped stage={stage}", restarted)
            self.assertEqual(
                mtimes,
                {
                    path.name: path.stat().st_mtime_ns
                    for path in (
                        output / "basis.npz",
                        output / "hamiltonian.csr.npz",
                        output / "eigensystem.h5",
                        output / "observables.h5",
                        output / "validation" / "metrics.json",
                    )
                },
            )

    def test_predecessor_guard_rejects_out_of_order_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "required stage 'basis'.*not complete"):
                self.run_stage(Path(directory), "hamiltonian")

    def test_hamiltonian_reuses_persisted_orbit_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "basis")
            with mock.patch(
                "turner2018_l32_server.build_orbit_basis",
                side_effect=AssertionError("basis must not be rebuilt"),
            ):
                self.run_stage(output, "hamiltonian")

    def test_corrupt_completed_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "basis")
            with (output / "basis.npz").open("ab") as handle:
                handle.write(b"corrupt")

            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                self.run_stage(output, "all")
            failure = json.loads(
                (output / "stages" / "basis.failure.json").read_text()
            )
            self.assertEqual(failure["status"], "failed")
            self.assertIn("sha256 mismatch", failure["error"])

    def test_rebuilt_basis_invalidates_and_recomputes_clean_downstream(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            old = {
                stage: json.loads((output / "stages" / f"{stage}.json").read_text())
                for stage in ("basis", "hamiltonian", "diagonalize", "observables", "validate")
            }
            (output / "stages" / "basis.json").unlink()
            self.run_stage(output, "basis")
            code, restarted = self.run_stage(output, "all")
            self.assertEqual(code, 0)
            for stage in ("hamiltonian", "diagonalize", "observables", "validate"):
                self.assertIn(f"completed stage={stage}", restarted)
                current = json.loads(
                    (output / "stages" / f"{stage}.json").read_text()
                )
                self.assertNotEqual(current["generation_id"], old[stage]["generation_id"])

    def test_direct_observables_rejects_corrupt_hamiltonian(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            with (output / "hamiltonian.csr.npz").open("ab") as handle:
                handle.write(b"corrupt")
            (output / "stages" / "observables.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "hamiltonian.*sha256"):
                self.run_stage(output, "observables")

    def test_changed_execution_fingerprint_prevents_old_stage_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            changed = json.loads(json.dumps(build_execution_fingerprint()))
            changed["sources"]["turner2018_ed_engine.py"] = "f" * 64
            with mock.patch(
                "turner2018_l32_server.build_execution_fingerprint",
                return_value=changed,
            ):
                code, restarted = self.run_stage(output, "all")
            self.assertEqual(code, 0)
            for stage in (
                "plan",
                "basis",
                "hamiltonian",
                "diagonalize",
                "observables",
                "validate",
            ):
                self.assertIn(f"completed stage={stage}", restarted)

    def test_recursive_validation_hashes_eigensystem_once_per_invocation(self):
        import turner2018_ed_artifacts as artifacts

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            original = artifacts._sha256
            count = 0

            def counting_sha(path):
                nonlocal count
                if Path(path).name == "eigensystem.h5":
                    count += 1
                return original(path)

            with mock.patch.object(artifacts, "_sha256", counting_sha):
                self.run_stage(output, "all")
            self.assertEqual(count, 1)

    def test_eigensystem_and_observables_are_separate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            eigensystem = output / "eigensystem.h5"
            observables = output / "observables.h5"
            self.assertTrue(eigensystem.is_file())
            self.assertTrue(observables.is_file())
            before = eigensystem.read_bytes()
            self.assertNotEqual(eigensystem, observables)
            self.assertEqual(before, eigensystem.read_bytes())

    def test_figures_stage_publishes_only_after_fig3_and_fig4_accept(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            fig3_artifact = output / "figures" / "fig3.png"
            fig4_artifact = output / "figures" / "fig4.png"

            def render(path, payload, passed=True):
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(payload)
                path.with_suffix(".npz").write_bytes(b"npz-" + payload)
                path.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "source": "independent-ed",
                            "acceptance": {
                                "passed": passed,
                                "generated_source": "independent-ed",
                            },
                            "generation_id": path.stem,
                            "generation_assets": {
                                "png_sha256": hashlib.sha256(payload).hexdigest(),
                                "npz_sha256": hashlib.sha256(
                                    b"npz-" + payload
                                ).hexdigest(),
                            },
                        }
                    )
                )
                return path

            with mock.patch(
                "turner2018_l32_server.FIG3_RENDERER_ADAPTER",
                side_effect=lambda *_: render(fig3_artifact, b"fig3"),
            ) as fig3_renderer, mock.patch(
                "turner2018_l32_server.FIG4_RENDERER_ADAPTER",
                side_effect=lambda *_: render(fig4_artifact, b"fig4"),
            ) as fig4_renderer:
                self.run_stage(output, "figures")

            fig3_renderer.assert_called_once_with(output, 10)
            fig4_renderer.assert_called_once_with(output, 10)
            manifest = require_stage(output, "figures")
            summary = json.loads((output / manifest["artifact"]["path"]).read_text())
            self.assertEqual(summary["fig3"]["sha256"], hashlib.sha256(b"fig3").hexdigest())
            self.assertEqual(summary["fig4"]["sha256"], hashlib.sha256(b"fig4").hexdigest())
            self.assertTrue(summary["acceptance"]["passed"])

    def test_figures_stage_does_not_publish_when_fig4_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            fig3 = output / "figures" / "fig3.png"
            fig4 = output / "figures" / "fig4.png"

            def render(path, passed):
                path.parent.mkdir(exist_ok=True)
                payload = path.name.encode()
                path.write_bytes(payload)
                path.with_suffix(".npz").write_bytes(b"npz-" + payload)
                path.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "source": "independent-ed",
                            "acceptance": {
                                "passed": passed,
                                "generated_source": "independent-ed",
                            },
                            "generation_assets": {
                                "png_sha256": hashlib.sha256(payload).hexdigest(),
                                "npz_sha256": hashlib.sha256(
                                    b"npz-" + payload
                                ).hexdigest(),
                            },
                        }
                    )
                )
                return path

            with mock.patch(
                "turner2018_l32_server.FIG3_RENDERER_ADAPTER",
                side_effect=lambda *_: render(fig3, True),
            ), mock.patch(
                "turner2018_l32_server.FIG4_RENDERER_ADAPTER",
                side_effect=lambda *_: render(fig4, False),
            ):
                with self.assertRaisesRegex(RuntimeError, "Fig. 4 acceptance"):
                    self.run_stage(output, "figures")

            self.assertFalse((output / "stages" / "figures.json").exists())

    def test_observables_and_validate_never_full_slice_eigenvectors(self):
        import h5py

        original = h5py.Dataset.__getitem__

        def reject_full_slice(dataset, key):
            if dataset.name.endswith("/vectors"):
                full = key == slice(None)
                if isinstance(key, tuple) and len(key) == 2:
                    full = all(
                        isinstance(part, slice) and part == slice(None)
                        for part in key
                    )
                if full:
                    raise AssertionError("full eigenvector slicing is forbidden")
            return original(dataset, key)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "basis")
            self.run_stage(output, "hamiltonian")
            self.run_stage(output, "diagonalize")
            with mock.patch.object(h5py.Dataset, "__getitem__", reject_full_slice):
                self.run_stage(output, "observables")
                self.run_stage(output, "validate")

    def test_validate_writes_real_bounded_scientific_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            metrics = json.loads(
                (output / "validation" / "metrics.json").read_text()
            )
            self.assertTrue(metrics["passed"])
            for name in (
                "basis_full_dimension",
                "basis_sector_dimension",
                "hamiltonian_hermiticity",
                "energies_sorted",
                "eigenvector_norm",
                "eigenpair_residual",
                "eigenvector_orthogonality",
                "z2_overlap_sum",
                "observables_finite",
            ):
                self.assertIn(name, metrics["metrics"])
                self.assertIn("passed", metrics["metrics"][name])
            self.assertEqual(metrics["eigenvector_access"], "column-chunked")

    def test_validate_publishes_failed_metrics_for_scientific_corruption(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            with h5py.File(output / "observables.h5", "r+") as handle:
                handle["observables/overlap_z2"][:] = 0.0
            manifest_path = output / "stages" / "observables.json"
            manifest = json.loads(manifest_path.read_text())
            with (output / "observables.h5").open("rb") as handle:
                manifest["artifact"]["sha256"] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
            atomic_write_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "scientific validation failed"):
                self.run_stage(output, "validate")
            metrics = json.loads(
                (output / "validation" / "metrics.json").read_text()
            )
            self.assertFalse(metrics["passed"])
            self.assertFalse(metrics["metrics"]["z2_overlap_sum"]["passed"])

    def test_insufficient_declared_memory_fails_before_dense_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "basis")
            self.run_stage(output, "hamiltonian")
            with mock.patch("scipy.sparse.csr_matrix.toarray", side_effect=AssertionError("dense conversion")):
                with self.assertRaises(MemoryError):
                    main(
                        [
                            "--stage",
                            "diagonalize",
                            "--length",
                            "10",
                            "--output-dir",
                            str(output),
                            "--declared-memory",
                            "1K",
                        ]
                    )
            self.assertFalse((output / "stages" / "diagonalize.json").exists())
            self.assertTrue((output / "stages" / "diagonalize.failure.json").exists())

    def test_failure_record_does_not_replace_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "basis")
            manifest = (output / "stages" / "basis.json").read_bytes()
            with mock.patch(
                "turner2018_l32_server.build_orbit_basis",
                side_effect=RuntimeError("injected basis rerun failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "basis rerun failure"):
                    self.run_stage(output, "basis", "--rebuild")
            self.assertEqual(manifest, (output / "stages" / "basis.json").read_bytes())
            failure = json.loads(
                (output / "stages" / "basis.failure.json").read_text()
            )
            self.assertEqual(failure["status"], "failed")


class TurnerL32SlurmTests(unittest.TestCase):
    def test_qdagnormal_script_has_exact_required_resources_and_no_secrets(self):
        text = (SCRIPTS / "turner2018_l32_qdagnormal.sbatch").read_text()

        for directive in (
            "#SBATCH --partition=qdagnormal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=64",
            "#SBATCH --mem=512G",
            "#SBATCH --time=24:00:00",
            "#SBATCH --gres=gpu:A800:1",
        ):
            self.assertIn(directive, text)
        self.assertIn("OPENBLAS_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"", text)
        self.assertIn("MKL_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"", text)
        self.assertIn("TURNER_LENGTH", text)
        self.assertRegex(text, r"28\|30\|32")
        self.assertIn("--stage all", text.replace("\n", " "))
        self.assertNotIn("--declared-memory", text)
        self.assertIn("TURNER_OFFLINE_IMAGE", text)
        self.assertIn("TURNER_OUTPUT_DIR", text)
        self.assertNotRegex(text, r"(?i)(password|api[_-]?key|access[_-]?token)\s*=")

    def test_scnet_script_is_provider_neutral_and_cpu_only_by_default(self):
        text = (SCRIPTS / "turner2018_l32_scnet.sbatch").read_text()
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=64",
            "#SBATCH --mem=512G",
            "#SBATCH --time=24:00:00",
        ):
            self.assertIn(directive, text)
        self.assertNotRegex(text, r"#SBATCH\s+--partition")
        self.assertNotRegex(text, r"#SBATCH\s+--account")
        self.assertNotRegex(text, r"#SBATCH\s+--qos")
        self.assertNotRegex(text, r"#SBATCH\s+--gres")
        self.assertIn("TURNER_LENGTH", text)
        self.assertRegex(text, r"28\|30\|32")
        self.assertIn("OPENBLAS_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"", text)
        self.assertNotIn("--declared-memory", text)

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
        self.assertIn('sinfo -o "%P %c %m %G %l %a"', text)
        self.assertIn("scontrol show partition", text)
        self.assertIn("turner2018_l32_scnet.sbatch", text)
        self.assertIn("SCNET_PARTITION", text)
        self.assertIn("SCNET_ACCOUNT", text)
        self.assertIn("SCNET_QOS", text)


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
            ["--declared-memory=512G"],
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
