"""Focused tests for the staged Turner L=32 server workflow."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
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
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            fig3_artifact = output / "figures" / "fig3_independent_L10.png"
            fig4_overview = output / "figures" / "fig4_independent_all.png"
            fig4_size = output / "figures" / "fig4_independent_L10.png"

            def render(path, payload, passed=True, *, lengths=None, generation_id=None):
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(payload)
                generation_id = generation_id or path.stem
                figure_arrays = {}
                if path == fig3_artifact:
                    figure_arrays = {
                        "panel_b_L10_shell": np.arange(6),
                        "panel_b_L10_exact_weights": np.full(6, 0.05),
                        "panel_b_L10_fsa_weights": np.full(6, 1.0 / 6.0),
                        "panel_c_L10_shell": np.arange(6),
                        "panel_c_L10_exact_weights": np.full(6, 0.04),
                        "panel_c_L10_fsa_weights": np.full(6, 1.0 / 6.0),
                    }
                np.savez(
                    path.with_suffix(".npz"),
                    generation_id=np.asarray(generation_id),
                    **figure_arrays,
                    **(
                        {f"L{lengths[0]}_source": np.asarray("independent-ed")}
                        if lengths and len(lengths) == 1
                        else {}
                    ),
                )
                metrics = {
                    "source": "independent-ed",
                    "acceptance": {
                        "passed": passed,
                        "generated_source": "independent-ed",
                        "provenance_passed": True,
                        "render_passed": True,
                        "statistics_passed": True,
                        "statistics_required": False,
                    },
                    "generation_id": generation_id,
                    "generation_assets": {
                        "png_sha256": hashlib.sha256(payload).hexdigest(),
                        "npz_sha256": hashlib.sha256(
                            path.with_suffix(".npz").read_bytes()
                        ).hexdigest(),
                    },
                }
                if path == fig3_artifact:
                    selected = [
                        {
                            "panel": "b",
                            "role": "lowest-matched-scar",
                            "exact_index": 1,
                            "exact_energy": -3.0,
                            "fsa_index": 0,
                            "fsa_energy": -2.9,
                            "match_strength": 0.3,
                            "zero_tolerance": 1e-10,
                            "full_fsa_shell_count": 11,
                            "plotted_folded_shell_count": 6,
                            "folding": (
                                "k=0 inversion-even: n and L-n are symmetry-related"
                            ),
                            "exact_weight_sum": 0.3,
                            "fsa_weight_sum": 1.0,
                        },
                        {
                            "panel": "c",
                            "role": "negative-adjacent-to-zero",
                            "exact_index": 5,
                            "exact_energy": -0.1,
                            "fsa_index": 2,
                            "fsa_energy": -0.09,
                            "match_strength": 0.24,
                            "zero_tolerance": 1e-10,
                            "full_fsa_shell_count": 11,
                            "plotted_folded_shell_count": 6,
                            "folding": (
                                "k=0 inversion-even: n and L-n are symmetry-related"
                            ),
                            "exact_weight_sum": 0.24,
                            "fsa_weight_sum": 1.0,
                        },
                    ]
                    metrics.update(
                        {
                            "primary_length": 10,
                            "selected_panel_states": selected,
                            "plot_conventions": {
                                panel: {
                                    "exact": "black circles, solid line",
                                    "fsa": "red crosses, dashed line",
                                    "x": "folded FSA shell index n=0..L/2",
                                    "y": "squared shell weight, linear",
                                }
                                for panel in ("panel_b", "panel_c")
                            },
                        }
                    )
                if lengths is not None:
                    metrics.update(
                        {
                            "available_independent_lengths": lengths,
                            "layout": {"lengths": lengths},
                            "lengths": {
                                str(item): {
                                    "provenance_acceptance": {"passed": True},
                                    "histogram_acceptance": {"passed": True},
                                }
                                for item in lengths
                            },
                            "series": {
                                f"L{item}_source": {
                                    "source": "independent-ed",
                                    "length": item,
                                }
                                for item in lengths
                            },
                        }
                    )
                path.with_suffix(".json").write_text(
                    json.dumps(metrics)
                )
                return path

            def render_fig4(*_args):
                generation_id = "fig4-set-generation"
                render(
                    fig4_overview,
                    b"fig4-overview",
                    lengths=[10],
                    generation_id=generation_id,
                )
                render(
                    fig4_size,
                    b"fig4-size",
                    lengths=[10],
                    generation_id=generation_id,
                )
                return fig4_overview

            with mock.patch(
                "turner2018_l32_server.FIG4_RENDERER_ADAPTER",
                side_effect=render_fig4,
            ) as fig4_renderer:
                self.run_stage(output, "figures")

            fig4_renderer.assert_called_once_with(output, 10)
            manifest = require_stage(output, "figures")
            summary = json.loads((output / manifest["artifact"]["path"]).read_text())
            self.assertEqual(
                summary["fig3"]["path"],
                "figures/fig3_independent_L10.png",
            )
            self.assertEqual(
                summary["fig4"]["overview"]["sha256"],
                hashlib.sha256(b"fig4-overview").hexdigest(),
            )
            self.assertEqual(set(summary["fig4"]["sizes"]), {"10"})
            self.assertEqual(
                summary["fig4"]["sizes"]["10"]["sha256"],
                hashlib.sha256(b"fig4-size").hexdigest(),
            )
            self.assertTrue(summary["acceptance"]["passed"])

            referenced = (
                fig3_artifact,
                fig3_artifact.with_suffix(".npz"),
                fig3_artifact.with_suffix(".json"),
                fig4_overview,
                fig4_overview.with_suffix(".npz"),
                fig4_overview.with_suffix(".json"),
                fig4_size,
                fig4_size.with_suffix(".npz"),
                fig4_size.with_suffix(".json"),
            )
            for target in referenced:
                with self.subTest(target=target.name):
                    original = target.read_bytes()
                    target.write_bytes(original + b"corrupt")
                    try:
                        with self.assertRaisesRegex(
                            RuntimeError, "figure.*hash|asset hash|JSON"
                        ):
                            self.run_stage(output, "figures")
                    finally:
                        target.write_bytes(original)

    def test_figures_stage_does_not_publish_when_fig4_rejects(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            fig3 = output / "figures" / "fig3_independent_L10.png"
            fig4 = output / "figures" / "fig4.png"

            def render(path, passed):
                path.parent.mkdir(exist_ok=True)
                payload = path.name.encode()
                path.write_bytes(payload)
                generation_id = path.stem
                figure_arrays = {}
                if path == fig3:
                    figure_arrays = {
                        "panel_b_L10_shell": np.arange(6),
                        "panel_b_L10_exact_weights": np.full(6, 0.05),
                        "panel_b_L10_fsa_weights": np.full(6, 1.0 / 6.0),
                        "panel_c_L10_shell": np.arange(6),
                        "panel_c_L10_exact_weights": np.full(6, 0.04),
                        "panel_c_L10_fsa_weights": np.full(6, 1.0 / 6.0),
                    }
                np.savez(
                    path.with_suffix(".npz"),
                    generation_id=np.asarray(generation_id),
                    **figure_arrays,
                )
                metrics = {
                    "source": "independent-ed",
                    "acceptance": {
                        "passed": passed,
                        "generated_source": "independent-ed",
                    },
                    "generation_id": generation_id,
                    "generation_assets": {
                        "png_sha256": hashlib.sha256(payload).hexdigest(),
                        "npz_sha256": hashlib.sha256(
                            path.with_suffix(".npz").read_bytes()
                        ).hexdigest(),
                    },
                }
                if path == fig3:
                    metrics.update(
                        {
                            "primary_length": 10,
                            "selected_panel_states": [
                                {
                                    "panel": "b",
                                    "role": "lowest-matched-scar",
                                    "exact_index": 1,
                                    "exact_energy": -3.0,
                                    "fsa_index": 0,
                                    "fsa_energy": -2.9,
                                    "match_strength": 0.3,
                                    "zero_tolerance": 1e-10,
                                    "full_fsa_shell_count": 11,
                                    "plotted_folded_shell_count": 6,
                                    "folding": (
                                        "k=0 inversion-even: n and L-n are symmetry-related"
                                    ),
                                    "exact_weight_sum": 0.3,
                                    "fsa_weight_sum": 1.0,
                                },
                                {
                                    "panel": "c",
                                    "role": "negative-adjacent-to-zero",
                                    "exact_index": 5,
                                    "exact_energy": -0.1,
                                    "fsa_index": 2,
                                    "fsa_energy": -0.09,
                                    "match_strength": 0.24,
                                    "zero_tolerance": 1e-10,
                                    "full_fsa_shell_count": 11,
                                    "plotted_folded_shell_count": 6,
                                    "folding": (
                                        "k=0 inversion-even: n and L-n are symmetry-related"
                                    ),
                                    "exact_weight_sum": 0.24,
                                    "fsa_weight_sum": 1.0,
                                },
                            ],
                            "plot_conventions": {
                                panel: {
                                    "exact": "black circles, solid line",
                                    "fsa": "red crosses, dashed line",
                                    "x": "folded FSA shell index n=0..L/2",
                                    "y": "squared shell weight, linear",
                                }
                                for panel in ("panel_b", "panel_c")
                            },
                        }
                    )
                if path == fig4:
                    metrics.update(
                        {
                            "available_independent_lengths": [10],
                            "layout": {"lengths": [10]},
                            "lengths": {"10": {}},
                            "series": {},
                        }
                    )
                path.with_suffix(".json").write_text(json.dumps(metrics))
                return path

            with mock.patch(
                "turner2018_l32_server.FIG4_RENDERER_ADAPTER",
                side_effect=lambda *_: render(fig4, False),
            ):
                with self.assertRaisesRegex(RuntimeError, "Fig. 4 acceptance"):
                    self.run_stage(output, "figures")

            self.assertFalse((output / "stages" / "figures.json").exists())

    def test_figure_acceptance_rejects_mixed_json_npz_generation_identity(self):
        import numpy as np
        import turner2018_l32_server as server

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fig4.png"
            npz = path.with_suffix(".npz")
            path.write_bytes(b"png")
            np.savez(npz, generation_id=np.asarray("npz-generation"))
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "source": "independent-ed",
                        "generation_id": "json-generation",
                        "acceptance": {
                            "passed": True,
                            "generated_source": "independent-ed",
                            "provenance_passed": True,
                            "render_passed": True,
                            "statistics_passed": False,
                            "statistics_required": False,
                        },
                        "generation_assets": {
                            "png_sha256": hashlib.sha256(b"png").hexdigest(),
                            "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                        },
                    }
                )
            )

            with self.assertRaisesRegex(RuntimeError, "generation identity"):
                server._accepted_figure(path, "Fig. 4", length=20)

    def test_figures_stage_never_skips_independently_corrupt_fig3_semantics(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_stage(output, "all")
            self.run_stage(output, "figures")
            figure = output / "figures" / "fig3_independent_L10.png"
            metrics_path = figure.with_suffix(".json")
            arrays_path = figure.with_suffix(".npz")
            summary_path = output / "figures" / "manifest.json"
            stage_path = output / "stages" / "figures.json"
            originals = {
                path: path.read_bytes()
                for path in (metrics_path, arrays_path, summary_path, stage_path)
            }
            with np.load(arrays_path, allow_pickle=False) as archive:
                required_source_arrays = {
                    "selection_L10_exact_shell_amplitudes",
                    "selection_L10_fsa_hamiltonian_sector",
                    "selection_L10_match_exact_indices",
                }
                self.assertTrue(required_source_arrays.issubset(archive.files))

            def json_mutation(field, value, panel=0):
                def mutate(metrics, _arrays):
                    metrics["selected_panel_states"][panel][field] = value(
                        metrics["selected_panel_states"][panel][field]
                    )

                return mutate

            def array_mutation(name, operation):
                def mutate(_metrics, arrays):
                    arrays[name] = operation(arrays[name].copy())

                return mutate

            corruptions = (
                ("exact index", json_mutation("exact_index", lambda value: value + 1)),
                (
                    "FSA index",
                    json_mutation("fsa_index", lambda value: (value + 1) % 6, panel=1),
                ),
                (
                    "exact energy",
                    json_mutation("exact_energy", lambda value: value + 0.125),
                ),
                (
                    "FSA energy",
                    json_mutation("fsa_energy", lambda value: value + 0.125, panel=1),
                ),
                (
                    "match strength",
                    json_mutation("match_strength", lambda value: value * 0.5),
                ),
                ("role", json_mutation("role", lambda _value: "interior-special", panel=1)),
                (
                    "full shell count",
                    json_mutation("full_fsa_shell_count", lambda value: value + 1),
                ),
                (
                    "folded shell count",
                    json_mutation(
                        "plotted_folded_shell_count", lambda value: value + 1, panel=1
                    ),
                ),
                (
                    "exact normalization",
                    json_mutation("exact_weight_sum", lambda value: value + 0.1),
                ),
                (
                    "FSA normalization",
                    json_mutation("fsa_weight_sum", lambda _value: 0.9, panel=1),
                ),
                (
                    "FSA nearest gap",
                    json_mutation("fsa_nearest_gap", lambda value: value + 0.1),
                ),
                (
                    "exact shell weights",
                    array_mutation(
                        "panel_b_L10_exact_weights", lambda values: np.roll(values, 1)
                    ),
                ),
                (
                    "FSA shell weights",
                    array_mutation(
                        "panel_c_L10_fsa_weights", lambda values: np.roll(values, 1)
                    ),
                ),
                (
                    "exact amplitude source",
                    array_mutation(
                        "selection_L10_exact_shell_amplitudes",
                        lambda values: values
                        + np.eye(*values.shape, dtype=values.dtype) * 1e-4,
                    ),
                ),
                (
                    "FSA Hamiltonian source",
                    array_mutation(
                        "selection_L10_fsa_hamiltonian_sector",
                        lambda values: values
                        + np.eye(values.shape[0], dtype=values.dtype) * 1e-4,
                    ),
                ),
                (
                    "matched indices source",
                    array_mutation(
                        "selection_L10_match_exact_indices",
                        lambda values: np.roll(values, 1),
                    ),
                ),
            )

            for label, mutate in corruptions:
                with self.subTest(label=label):
                    for path, payload in originals.items():
                        path.write_bytes(payload)
                    metrics = json.loads(metrics_path.read_text())
                    with np.load(arrays_path, allow_pickle=False) as archive:
                        arrays = {name: archive[name].copy() for name in archive.files}
                    mutate(metrics, arrays)
                    with arrays_path.open("wb") as handle:
                        np.savez(handle, **arrays)
                    metrics["generation_assets"]["npz_sha256"] = hashlib.sha256(
                        arrays_path.read_bytes()
                    ).hexdigest()
                    metrics_path.write_text(json.dumps(metrics))

                    summary = json.loads(summary_path.read_text())
                    summary["fig3"]["metrics_sha256"] = hashlib.sha256(
                        metrics_path.read_bytes()
                    ).hexdigest()
                    summary["fig3"]["arrays_sha256"] = hashlib.sha256(
                        arrays_path.read_bytes()
                    ).hexdigest()
                    summary_path.write_text(json.dumps(summary))
                    stage = json.loads(stage_path.read_text())
                    stage["artifact"]["sha256"] = hashlib.sha256(
                        summary_path.read_bytes()
                    ).hexdigest()
                    stage_path.write_text(json.dumps(stage))

                    with self.assertRaisesRegex(
                        RuntimeError, "Fig. 3.*selection|semantic|source evidence"
                    ):
                        self.run_stage(output, "figures")

    def test_figures_stage_rejects_coordinated_fig3_rewrite_against_upstream(self):
        import h5py
        import numpy as np
        import turner2018_fig3 as fig3

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_execution_fingerprint()
            self.run_stage(output, "all")
            self.run_stage(output, "figures")
            figure = output / "figures" / "fig3_independent_L10.png"
            metrics_path = figure.with_suffix(".json")
            arrays_path = figure.with_suffix(".npz")
            summary_path = output / "figures" / "manifest.json"
            stage_path = output / "stages" / "figures.json"

            metrics = json.loads(metrics_path.read_text())
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            energies = arrays["panel_a_L10_energies"] + 0.125
            amplitudes = arrays["selection_L10_exact_shell_amplitudes"] * 0.99
            hamiltonian = arrays["selection_L10_fsa_hamiltonian_sector"] + (
                np.eye(6) * 0.125
            )
            tower = arrays["selection_L10_match_exact_indices"]
            states = fig3.select_fig3_shell_panel_states(
                length=10,
                energies=energies,
                exact_shell_amplitudes=amplitudes,
                fsa_hamiltonian_sector=hamiltonian,
                matched_tower={"tower": tower},
            )
            fsa_energies, fsa_vectors = np.linalg.eigh(hamiltonian)
            projection = np.abs(amplitudes.T @ fsa_vectors) ** 2

            arrays["panel_a_L10_energies"] = energies
            arrays["panel_a_L10_fsa_energies"] = fsa_energies
            arrays["selection_L10_exact_shell_amplitudes"] = amplitudes
            arrays["selection_L10_fsa_hamiltonian_sector"] = hamiltonian
            for state in states:
                prefix = f"panel_{state.panel}_L10_"
                arrays[prefix + "shell"] = np.asarray(state.shell)
                arrays[prefix + "exact_weights"] = np.asarray(state.exact_weights)
                arrays[prefix + "fsa_weights"] = np.asarray(state.fsa_weights)
            with arrays_path.open("wb") as handle:
                np.savez(handle, **arrays)

            selected = [state.to_metadata_dict() for state in states]
            fsa = metrics["lengths"]["10"]["fsa"]
            metrics["selected_panel_states"] = selected
            fsa["selected_states"] = selected
            fsa["match_exact_energies"] = energies[tower].tolist()
            fsa["fsa_energies"] = fsa_energies.tolist()
            fsa["match_strengths"] = projection[
                tower, np.arange(len(tower))
            ].tolist()
            fsa["shell_amplitudes_sha256"] = hashlib.sha256(
                amplitudes.tobytes()
            ).hexdigest()
            fsa["fsa_hamiltonian_sha256"] = hashlib.sha256(
                hamiltonian.tobytes()
            ).hexdigest()
            metrics["generation_assets"]["npz_sha256"] = hashlib.sha256(
                arrays_path.read_bytes()
            ).hexdigest()
            metrics_path.write_text(json.dumps(metrics))

            summary = json.loads(summary_path.read_text())
            summary["fig3"]["metrics_sha256"] = hashlib.sha256(
                metrics_path.read_bytes()
            ).hexdigest()
            summary["fig3"]["arrays_sha256"] = hashlib.sha256(
                arrays_path.read_bytes()
            ).hexdigest()
            summary_path.write_text(json.dumps(summary))
            stage = json.loads(stage_path.read_text())
            stage["artifact"]["sha256"] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
            stage_path.write_text(json.dumps(stage))

            original_getitem = h5py.Dataset.__getitem__

            def reject_eigenvectors(dataset, key):
                if dataset.name.endswith("/vectors"):
                    raise AssertionError("semantic restart must not load eigenvectors")
                return original_getitem(dataset, key)

            with mock.patch.object(h5py.Dataset, "__getitem__", reject_eigenvectors):
                with self.assertRaisesRegex(
                    RuntimeError, "Fig. 3.*upstream|source evidence"
                ):
                    self.run_stage(output, "figures")

    def test_production_fig4_gate_requires_accepted_statistics(self):
        import numpy as np
        import turner2018_l32_server as server

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fig4.png"
            npz = path.with_suffix(".npz")
            path.write_bytes(b"png")
            np.savez(npz, generation_id=np.asarray("same-generation"))
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "source": "independent-ed",
                        "generation_id": "same-generation",
                        "acceptance": {
                            "passed": True,
                            "generated_source": "independent-ed",
                            "provenance_passed": True,
                            "render_passed": True,
                            "statistics_passed": False,
                            "statistics_required": False,
                        },
                        "available_independent_lengths": [28],
                        "layout": {"lengths": [28]},
                        "lengths": {
                            "28": {
                                "provenance_acceptance": {"passed": True},
                                "histogram_acceptance": {"passed": False},
                            }
                        },
                        "series": {
                            "L28_source": {
                                "source": "independent-ed",
                                "length": 28,
                            }
                        },
                        "generation_assets": {
                            "png_sha256": hashlib.sha256(b"png").hexdigest(),
                            "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                        },
                    }
                )
            )

            with self.assertRaisesRegex(RuntimeError, "statistics acceptance"):
                server._accepted_figure(path, "Fig. 4", length=28)

            metrics_path = path.with_suffix(".json")
            metrics = json.loads(metrics_path.read_text())
            metrics["acceptance"]["statistics_required"] = True
            metrics["acceptance"]["statistics_passed"] = True
            metrics["lengths"]["28"]["histogram_acceptance"]["passed"] = True
            for field in ("provenance_passed", "render_passed"):
                with self.subTest(field=field):
                    candidate = json.loads(json.dumps(metrics))
                    candidate["acceptance"][field] = False
                    metrics_path.write_text(json.dumps(candidate))
                    with self.assertRaisesRegex(
                        RuntimeError, "provenance/render acceptance"
                    ):
                        server._accepted_figure(path, "Fig. 4", length=28)

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
    DZESHELL_ROOT = "/work/share/giggleliu/jiangweiqi"
    DZESHELL_CLASSES = {
        "turner2018_dzeshell_l22_28.sbatch": {
            "lengths": "22|24|26|28",
            "cpus": 8,
            "gpus": 1,
            "memory": "60000M",
        },
        "turner2018_dzeshell_l30.sbatch": {
            "lengths": "30",
            "cpus": 16,
            "gpus": 2,
            "memory": "120000M",
        },
        "turner2018_dzeshell_l32.sbatch": {
            "lengths": "32",
            "cpus": 32,
            "gpus": 4,
            "memory": "240000M",
        },
    }

    def test_dzeshell_wrappers_have_immutable_length_resource_classes(self):
        self.assertFalse((SCRIPTS / "turner2018_l32_qdagnormal.sbatch").exists())
        for name, expected in self.DZESHELL_CLASSES.items():
            with self.subTest(name=name):
                text = (SCRIPTS / name).read_text()
                for directive in (
                    "#SBATCH --partition=dzagnormal",
                    "#SBATCH --nodes=1",
                    "#SBATCH --ntasks=1",
                    f"#SBATCH --cpus-per-task={expected['cpus']}",
                    f"#SBATCH --mem={expected['memory']}",
                    "#SBATCH --time=24:00:00",
                    (
                        "#SBATCH --gres=gpu:NVIDIAA80080GBPCIeLC:"
                        f"{expected['gpus']}"
                    ),
                ):
                    self.assertIn(directive, text)
                self.assertIn(
                    f'TURNER_ALLOWED_LENGTHS="{expected["lengths"]}"', text
                )
                self.assertIn(
                    'source "$canonical_runner"',
                    text,
                )
                self.assertNotIn("BASH_SOURCE", text)
                self.assertNotIn("turner2018_l32_server.py", text)
                source_position = text.index('source "$canonical_runner"')
                self.assertLess(text.index("realpath -e"), source_position)
                self.assertLess(text.index("canonical_submit_dir="), source_position)
                self.assertLess(text.index("canonical_runner="), source_position)

    def _synthetic_root_wrapper(self, root: Path) -> Path:
        wrapper = SCRIPTS / "turner2018_dzeshell_l30.sbatch"
        spool = root.parent / "slurm-spool-copy"
        spool.write_text(
            wrapper.read_text().replace(self.DZESHELL_ROOT, str(root))
        )
        return spool

    @staticmethod
    def _write_sentinel_runner(submit: Path, sentinel: Path) -> None:
        scripts = submit / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "turner2018_dzeshell_run.sh").write_text(
            'printf "%s" "$TURNER_LENGTH" > "$RUNNER_SENTINEL"\n'
        )

    def test_copied_spool_wrapper_finds_runner_in_valid_shared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            shared_root = temporary / "shared-root"
            submit = shared_root / "reviewed-checkout"
            sentinel = temporary / "runner-called"
            self._write_sentinel_runner(submit, sentinel)
            spool = self._synthetic_root_wrapper(shared_root)

            result = subprocess.run(
                ["bash", str(spool)],
                env={
                    **os.environ,
                    "SLURM_SUBMIT_DIR": str(submit),
                    "RUNNER_SENTINEL": str(sentinel),
                },
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sentinel.read_text(), "30")

    def test_wrapper_rejects_outside_runner_without_side_effect(self):
        wrapper = SCRIPTS / "turner2018_dzeshell_l30.sbatch"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attacker = root / "attacker-checkout"
            sentinel = root / "attacker-ran"
            self._write_sentinel_runner(attacker, sentinel)

            result = subprocess.run(
                ["bash", str(wrapper)],
                env={
                    **os.environ,
                    "SLURM_SUBMIT_DIR": str(attacker),
                    "RUNNER_SENTINEL": str(sentinel),
                },
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn(self.DZESHELL_ROOT, result.stderr)
            self.assertFalse(sentinel.exists())

    def test_wrapper_rejects_unset_nonexistent_and_symlink_escaped_submit_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            shared_root = temporary / "shared-root"
            shared_root.mkdir()
            spool = self._synthetic_root_wrapper(shared_root)

            unset_environment = dict(os.environ)
            unset_environment.pop("SLURM_SUBMIT_DIR", None)
            unset = subprocess.run(
                ["bash", str(spool)],
                env=unset_environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(unset.returncode, 2)
            self.assertIn("SLURM_SUBMIT_DIR is required", unset.stderr)

            nonexistent = subprocess.run(
                ["bash", str(spool)],
                env={
                    **os.environ,
                    "SLURM_SUBMIT_DIR": str(shared_root / "missing"),
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(nonexistent.returncode, 2)
            self.assertIn("existing directory", nonexistent.stderr)

            attacker = temporary / "attacker-checkout"
            sentinel = temporary / "symlink-attacker-ran"
            self._write_sentinel_runner(attacker, sentinel)
            escaped = shared_root / "escaped-checkout"
            escaped.symlink_to(attacker, target_is_directory=True)
            symlink_result = subprocess.run(
                ["bash", str(spool)],
                env={
                    **os.environ,
                    "SLURM_SUBMIT_DIR": str(escaped),
                    "RUNNER_SENTINEL": str(sentinel),
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(symlink_result.returncode, 2)
            self.assertIn("outside", symlink_result.stderr)
            self.assertFalse(sentinel.exists())

    def test_dzeshell_common_runner_uses_shared_offline_paths(self):
        text = (SCRIPTS / "turner2018_dzeshell_run.sh").read_text()
        root = "/work/share/giggleliu/jiangweiqi"
        self.assertIn(f'readonly TURNER_SHARED_ROOT="{root}"', text)
        self.assertIn(
            'TURNER_REPO="${TURNER_REPO:-$TURNER_SHARED_ROOT/quantum.harness}"',
            text,
        )
        self.assertIn(
            'TURNER_RUNTIME="${TURNER_RUNTIME:-$TURNER_SHARED_ROOT/python/cpython-3.12}"',
            text,
        )
        self.assertIn(
            'TURNER_OUTPUT_DIR="${TURNER_OUTPUT_DIR:-$TURNER_SHARED_ROOT/results/turner-l${TURNER_LENGTH}}"',
            text,
        )
        self.assertIn(
            'TURNER_PYTHON="${TURNER_PYTHON:-$TURNER_REPO/.venv/bin/python}"',
            text,
        )
        self.assertIn("realpath -m", text)
        self.assertIn("turner2018_wheelhouse.py", text)
        self.assertIn("--check-runtime", text)
        self.assertIn("--stage all", text.replace("\n", " "))
        self.assertNotIn("--declared-memory", text)
        self.assertIn('OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"', text)

    def test_dzeshell_runner_rejects_exported_paths_outside_shared_root(self):
        runner = SCRIPTS / "turner2018_dzeshell_run.sh"
        base = {
            **os.environ,
            "TURNER_ALLOWED_LENGTHS": "30",
            "TURNER_LENGTH": "30",
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_MEM_PER_NODE": "120000M",
            "SLURM_GPUS_ON_NODE": "2",
        }
        rejected = {
            "TURNER_REPO": "/tmp/checkout",
            "TURNER_OUTPUT_DIR": str(Path.home() / "results"),
            "TURNER_PYTHON": "/usr/bin/python3",
            "TURNER_RUNTIME": "/opt/cpython-3.12",
            "TURNER_OFFLINE_IMAGE": "/tmp/runtime.sif",
        }
        for key, value in rejected.items():
            with self.subTest(key=key):
                environment = {**base, key: value}
                result = subprocess.run(
                    ["bash", str(runner)],
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(key, result.stderr)
                self.assertIn("/work/share/giggleliu/jiangweiqi", result.stderr)

    def test_dzeshell_runner_rejects_traversal_out_of_shared_root(self):
        runner = SCRIPTS / "turner2018_dzeshell_run.sh"
        environment = {
            **os.environ,
            "TURNER_ALLOWED_LENGTHS": "32",
            "TURNER_LENGTH": "32",
            "SLURM_CPUS_PER_TASK": "32",
            "SLURM_MEM_PER_NODE": "240000M",
            "SLURM_GPUS_ON_NODE": "4",
            "TURNER_REPO": (
                "/work/share/giggleliu/jiangweiqi/quantum.harness/../../../../tmp"
            ),
        }
        result = subprocess.run(
            ["bash", str(runner)],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("TURNER_REPO", result.stderr)

    def test_dzeshell_tracked_files_have_no_literal_connection_secrets(self):
        profile_path = (
            REPO / "skills" / "using-slurm" / "profiles" / "qdeshell.toml"
        )
        profile = tomllib.loads(profile_path.read_text())
        self.assertEqual(profile["connection"]["ssh"], {"alias": "qdeshell"})
        forbidden_profile_keys = {
            "host",
            "hostname",
            "port",
            "user",
            "username",
            "key",
            "key_path",
            "identity_file",
            "password",
            "token",
            "secret",
        }

        def walk_profile(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(key.lower(), forbidden_profile_keys)
                    walk_profile(child)
            elif isinstance(value, list):
                for child in value:
                    walk_profile(child)

        walk_profile(profile)

        tracked = [
            profile_path,
            SCRIPTS / "turner2018_dzeshell_run.sh",
            *(SCRIPTS / name for name in self.DZESHELL_CLASSES),
        ]
        assignment = re.compile(
            r"(?im)^\s*(?:export\s+|readonly\s+)?"
            r"(?P<name>[A-Z][A-Z0-9_]*)=(?P<value>[^\n#]*)$"
        )
        secret_name_parts = {
            "HOST",
            "HOSTNAME",
            "PORT",
            "USER",
            "USERNAME",
            "KEY",
            "KEY_PATH",
            "PRIVATE_KEY",
            "IDENTITY_FILE",
            "PASSWORD",
            "PASSWD",
            "TOKEN",
            "SECRET",
        }
        forbidden_literals = (
            "-----BEGIN",
            "qdeshell_rsa",
            "~/.ssh",
            r"C:\\",
        )
        for path in tracked:
            with self.subTest(path=path.name):
                text = path.read_text()
                for match in assignment.finditer(text):
                    name = match.group("name")
                    value = match.group("value").strip().strip("\"'")
                    segments = name.split("_")
                    looks_secret = (
                        name in secret_name_parts
                        or any(part in secret_name_parts for part in segments)
                        or any(name.endswith(f"_{part}") for part in secret_name_parts)
                    )
                    if looks_secret and value and not value.startswith("$"):
                        self.fail(f"{path.name}: literal value assigned to {name}")
                for literal in forbidden_literals:
                    self.assertNotIn(literal, text)
                self.assertIsNone(
                    re.search(r"(?:ssh|https?)://[^/\s:@]+:[^@\s]+@", text)
                )

    def test_dzeshell_common_runner_rejects_unsupported_lengths_and_bad_resources(self):
        runner = SCRIPTS / "turner2018_dzeshell_run.sh"
        base = {
            **os.environ,
            "TURNER_ALLOWED_LENGTHS": "30",
            "TURNER_LENGTH": "28",
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_MEM_PER_NODE": "120000M",
            "SLURM_GPUS_ON_NODE": "2",
        }
        unsupported = subprocess.run(
            ["bash", str(runner)], env=base, text=True, capture_output=True
        )
        self.assertEqual(unsupported.returncode, 2)
        self.assertIn("not allowed", unsupported.stderr)

        for key, value in (
            ("SLURM_CPUS_PER_TASK", "8"),
            ("SLURM_MEM_PER_NODE", "60000M"),
            ("SLURM_GPUS_ON_NODE", "1"),
        ):
            with self.subTest(key=key):
                environment = {
                    **base,
                    "TURNER_LENGTH": "30",
                    key: value,
                }
                result = subprocess.run(
                    ["bash", str(runner)],
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(key, result.stderr)

    def test_dzeshell_common_runner_requires_scheduler_environment(self):
        runner = SCRIPTS / "turner2018_dzeshell_run.sh"
        environment = {
            **os.environ,
            "TURNER_ALLOWED_LENGTHS": "32",
            "TURNER_LENGTH": "32",
        }
        for key in (
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE",
            "SLURM_GPUS_ON_NODE",
        ):
            environment.pop(key, None)
        result = subprocess.run(
            ["bash", str(runner)],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SLURM_CPUS_PER_TASK", result.stderr)

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
        documented_lengths = {
            "turner2018_dzeshell_l22_28.sbatch": 28,
            "turner2018_dzeshell_l30.sbatch": 30,
            "turner2018_dzeshell_l32.sbatch": 32,
        }
        for script, length in documented_lengths.items():
            expected = (
                f"TURNER_LENGTH={length} scripts/harness_slurm.sh --profile "
                "skills/using-slurm/profiles/qdeshell.toml submit --test-only "
                f"--script scripts/{script}"
            )
            self.assertIn(expected, text)
            self.assertNotIn(f"submit --script scripts/{script}", text)
        self.assertNotIn("turner2018_l32_qdagnormal.sbatch", text)
        self.assertNotIn("qdagnormal", text)
        self.assertIn('sinfo -o "%P %c %m %G %l %a"', text)
        self.assertIn("scontrol show partition", text)
        self.assertIn("turner2018_l32_scnet.sbatch", text)
        self.assertIn("SCNET_PARTITION", text)
        self.assertIn("SCNET_ACCOUNT", text)
        self.assertIn("SCNET_QOS", text)

    def test_documented_dzeshell_commands_scope_each_length(self):
        text = (REPO / "tracks" / "ed" / "README.md").read_text()
        match = re.search(
            r"```bash\n(# L=22, 24, 26, or 28:.*?"
            r"turner2018_dzeshell_l32\.sbatch)\n```",
            text,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            calls = root / "calls"
            harness = scripts / "harness_slurm.sh"
            harness.write_text(
                '#!/usr/bin/env bash\nprintf "%s|%s\\n" '
                '"${TURNER_LENGTH-unset}" "$*" >> "$CALLS"\n'
            )
            harness.chmod(0o755)
            result = subprocess.run(
                ["bash", "-eu", "-o", "pipefail", "-c", match.group(1)],
                cwd=root,
                env={**os.environ, "TURNER_LENGTH": "99", "CALLS": str(calls)},
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = [line.split("|", 1)[0] for line in calls.read_text().splitlines()]
            self.assertEqual(observed, ["28", "30", "32"])


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
