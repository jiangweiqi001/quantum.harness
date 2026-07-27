#!/usr/bin/env python3
"""Restartable, fail-closed Turner PXP ED/FSA server workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from turner2018_ed_artifacts import (
    validate_stage,
    write_basis_artifact,
    write_eigensystem,
    write_hamiltonian_artifact,
    write_observables,
)
from turner2018_ed_engine import (
    OrbitBasis,
    assemble_reduced_hamiltonian,
    build_orbit_basis,
)
from turner2018_ed_observables import compute_observables
from turner2018_ed_solver import estimate_dense_resources, solve_full_eigensystem

L32_FULL_DIMENSION = 4_870_847
L32_SECTOR_DIMENSION = 77_436
FULL_FSA_SHELLS = 33
SYMMETRY_FOLDED_FSA_SHELLS = 17
SCHEMA_VERSION = "turner-l32-ed-fsa-v1"
LOCAL_EQUIVALENCE_SCHEMA_VERSION = "turner-local-equivalence-v1"
QUANTUM_MODEL = "H=sum_j P_(j-1) X_j P_(j+1), PBC"
STAGES = (
    "plan",
    "basis",
    "hamiltonian",
    "diagonalize",
    "observables",
    "validate",
    "figures",
    "all",
)
LOCAL_VALIDATION_DIMENSIONS = {
    10: (123, 14),
    12: (322, 26),
    14: (843, 49),
    16: (2207, 99),
    18: (5778, 209),
    20: (15127, 455),
}
LOCAL_VALIDATION_LENGTHS = tuple(LOCAL_VALIDATION_DIMENSIONS)
LOCAL_VALIDATION_TOLERANCES = {
    "reduced_matrix": 1e-11,
    "complete_spectrum": 1e-10,
    "degenerate_total_z2": 1e-10,
    "degenerate_projector_diagonal": 1e-10,
    "degenerate_subspace": 1e-10,
    "fsa_beta": 1e-11,
    "fsa_projected_shell": 1e-11,
    "pr2_isolated": 1e-10,
}
STAGE_PREDECESSOR = {
    "hamiltonian": "basis",
    "diagonalize": "hamiltonian",
    "observables": "diagonalize",
    "validate": "observables",
    "figures": "validate",
}
COMPUTATIONAL_STAGES = (
    "basis",
    "hamiltonian",
    "diagonalize",
    "observables",
    "validate",
)
FIGURES_ADAPTER: Callable[[Path, int], Path] | None = None


def dense_resource_estimate(dimension: int) -> dict[str, int | float]:
    """Return decimal-GB storage estimates for real dense arrays."""
    bytes_per_array = int(dimension) ** 2 * 8
    return {
        "dimension": int(dimension),
        "bytes_per_dense_array": bytes_per_array,
        "gb_per_dense_array": bytes_per_array / 1e9,
        "gib_per_dense_array": bytes_per_array / 2**30,
        "matrix_plus_eigenvectors_gb": 2 * bytes_per_array / 1e9,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling .partial file and atomically rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def require_stage(output_dir: str | Path, stage: str) -> dict[str, Any]:
    """Load a hash-validated completed stage or fail closed."""
    path = Path(output_dir) / "stages" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"required stage {stage!r} is not complete: missing {path}")
    try:
        return validate_stage(Path(output_dir), stage)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(
            f"required stage {stage!r} is not complete or valid: {error}"
        ) from error


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git_revision() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    revision = (result.stdout or "").strip()
    if result.returncode != 0 or not revision:
        detail = (result.stderr or "").strip()
        message = "unable to resolve git revision"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message)
    return revision


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "h5py", "quspin"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


_MEMORY_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([KMGTPE]?)(I?B)?\s*$",
    re.IGNORECASE,
)


def parse_memory_bytes(value: str) -> int:
    """Parse Slurm-style memory; a bare value is interpreted as MiB."""
    match = _MEMORY_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid memory declaration: {value!r}")
    amount = float(match.group(1))
    prefix = match.group(2).upper()
    unit = (match.group(3) or "").upper()
    if not prefix:
        multiplier = 2**20 if not unit else 1
    else:
        exponent = "KMGTPE".index(prefix) + 1
        multiplier = (1000 if unit == "B" else 1024) ** exponent
    result = int(amount * multiplier)
    if result < 1:
        raise ValueError("declared memory must be positive")
    return result


def resolve_declared_memory_bytes(
    environment: dict[str, str] | os._Environ[str],
    explicit: str | None = None,
) -> int:
    """Resolve total declared memory from CLI or standard Slurm variables."""
    if explicit is not None:
        return parse_memory_bytes(explicit)
    per_node = environment.get("SLURM_MEM_PER_NODE")
    if per_node:
        return parse_memory_bytes(per_node)
    per_cpu = environment.get("SLURM_MEM_PER_CPU")
    if per_cpu:
        cpus = environment.get("SLURM_CPUS_ON_NODE")
        if cpus is None:
            cpus_per_task = int(environment.get("SLURM_CPUS_PER_TASK", "1"))
            tasks = int(environment.get("SLURM_NTASKS", "1"))
            cpu_count = cpus_per_task * tasks
        else:
            cpu_count = int(cpus)
        if cpu_count < 1:
            raise ValueError("declared Slurm CPU count must be positive")
        return parse_memory_bytes(per_cpu) * cpu_count
    raise RuntimeError(
        "dense diagonalization requires --declared-memory or "
        "SLURM_MEM_PER_NODE/SLURM_MEM_PER_CPU"
    )


def _max_abs_array(values: Any) -> float:
    import numpy as np

    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _max_abs_sparse(values: Any) -> float:
    return _max_abs_array(values.data) if values.nnz else 0.0


def _threshold_metric(value: float, tolerance: float) -> dict[str, float | bool]:
    return {
        "value": float(value),
        "tolerance": float(tolerance),
        "passed": bool(value <= tolerance),
    }


def _dimension_metric(
    expected: int,
    candidate: int,
    reference: int,
) -> dict[str, int | bool]:
    return {
        "expected": int(expected),
        "candidate": int(candidate),
        "reference": int(reference),
        "passed": bool(candidate == reference == expected),
    }


def _shape_metric(
    expected: tuple[int, ...],
    candidate: tuple[int, ...],
    reference: tuple[int, ...],
) -> dict[str, list[int] | bool]:
    return {
        "expected": list(expected),
        "candidate": list(candidate),
        "reference": list(reference),
        "passed": bool(candidate == reference == expected),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "pxp_ed.py",
        "turner2018_fig3.py",
        "turner2018_ed_engine.py",
        "turner2018_ed_solver.py",
        "turner2018_ed_observables.py",
        "turner2018_ed_validation.py",
        "turner2018_l32_server.py",
    )
    return {name: _sha256(root / name) for name in names}


def _local_validation_provenance(invocation: list[str]) -> dict[str, Any]:
    return {
        "invocation": invocation,
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "candidate_engine": (
            "turner2018_ed_engine direct dihedral-orbit assembly with "
            "turner2018_ed_solver and external-eigensystem observables"
        ),
        "reference_engine": (
            "pxp_ed full constrained basis transformed by "
            "symmetry_basis_k0_inversion_even with turner2018_fig3 FSA"
        ),
    }


def build_plan(length: int, output_dir: Path, argv: list[str]) -> dict[str, Any]:
    """Build the immutable scientific/resource plan."""
    dimension = L32_SECTOR_DIMENSION if length == 32 else None
    packages = _package_versions()
    resources = dense_resource_estimate(dimension or 0)
    if dimension is not None:
        task3_estimate = estimate_dense_resources(dimension, vectors=True)
        resources.update(
            {
                "matrix_bytes": task3_estimate.matrix_bytes,
                "eigenvector_bytes": task3_estimate.eigenvector_bytes,
                "minimum_requested_bytes": task3_estimate.minimum_requested_bytes,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "readiness": "ready",
        "blocker": None,
        "invocation": argv,
        "model": {
            "hamiltonian": QUANTUM_MODEL,
            "length": length,
            "boundary": "periodic",
            "momentum": 0,
            "inversion": "even",
        },
        "basis": {
            "constructor": "native direct dihedral-orbit basis",
            "enumeration": "direct constrained states; never scan 2**L",
            "symmetry_reduction": "direct k=0 inversion-even dihedral orbits",
            "full_constrained_dimension": (
                L32_FULL_DIMENSION if length == 32 else None
            ),
            "sector_dimension": dimension,
        },
        "fsa": {
            "full_shell_count": length + 1,
            "symmetry_folded_shell_count": length // 2 + 1,
        },
        "solver": {
            "dense_dtype": "float64",
            "dense_order": "Fortran",
            "routine": "scipy.linalg.eigh",
            "driver": "evd",
            "overwrite_a": True,
            "check_finite": False,
        },
        "observables": {
            "pr2": "sum_i |V_ij|**4, streamed by eigenvector-column chunks",
        },
        "resources": resources,
        "artifacts": {
            "basis": "basis.npz",
            "hamiltonian": "hamiltonian.csr.npz",
            "eigensystem": "eigensystem.h5",
            "observables": "observables.h5",
            "validation": "validation/metrics.json",
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "quspin_role": "optional cross-check only",
            "git_revision": _git_revision(),
        },
    }


def _write_hashed_json_stage(
    output_dir: Path,
    stage: str,
    relative_path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    target = output_dir / relative_path
    atomic_write_json(target, payload)
    stage_payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": "complete",
        "artifact": {
            "path": relative_path,
            "sha256": _sha256(target),
            "shape": [],
            "dtype": "json",
            "conventions": ["atomic-json", "sha256-validated"],
        },
    }
    atomic_write_json(
        output_dir / "stages" / f"{stage}.json",
        stage_payload,
    )
    return stage_payload


def write_plan(length: int, output_dir: Path, argv: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(length, output_dir, argv)
    _write_hashed_json_stage(output_dir, "plan", "manifest.json", plan)
    return plan


def _load_plan(output_dir: Path, length: int) -> dict[str, Any]:
    require_stage(output_dir, "plan")
    plan = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    planned_length = plan.get("model", {}).get("length")
    if planned_length != length:
        raise RuntimeError(
            f"output directory was planned for L={planned_length}, not L={length}"
        )
    return plan


def run_basis(length: int, output_dir: Path) -> None:
    basis = build_orbit_basis(length)
    write_basis_artifact(
        output_dir,
        basis=basis.constrained_states,
        representatives=basis.representatives,
        orbit_sizes=basis.orbit_sizes,
        length=length,
    )


def _load_basis(path: Path) -> OrbitBasis:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        required = {"basis", "representatives", "orbit_sizes", "length"}
        if not required.issubset(data.files):
            raise RuntimeError("basis artifact lacks restartable orbit metadata")
        return OrbitBasis(
            length=int(data["length"]),
            constrained_states=np.asarray(data["basis"], dtype=np.uint64),
            representatives=np.asarray(data["representatives"], dtype=np.uint64),
            orbit_sizes=np.asarray(data["orbit_sizes"], dtype=np.uint8),
        )


def run_hamiltonian(length: int, output_dir: Path) -> None:
    require_stage(output_dir, "basis")
    basis = _load_basis(output_dir / "basis.npz")
    if basis.length != length:
        raise RuntimeError("basis artifact length does not match requested length")
    write_hamiltonian_artifact(
        output_dir,
        hamiltonian=assemble_reduced_hamiltonian(basis),
    )


def run_diagonalize(
    length: int,
    output_dir: Path,
    declared_memory_bytes: int,
    chunk_columns: int,
) -> None:
    import scipy.sparse as sp

    require_stage(output_dir, "hamiltonian")
    matrix = sp.load_npz(output_dir / "hamiltonian.csr.npz").tocsr()
    estimate = estimate_dense_resources(matrix.shape[0], vectors=True)
    if declared_memory_bytes < estimate.minimum_requested_bytes:
        raise MemoryError(
            "declared memory is insufficient for dense diagonalization "
            f"({declared_memory_bytes} < {estimate.minimum_requested_bytes})"
        )
    energies, vectors = solve_full_eigensystem(
        matrix,
        vectors=True,
        declared_memory_bytes=declared_memory_bytes,
        validation_chunk_columns=chunk_columns,
    )
    assert vectors is not None
    write_eigensystem(
        output_dir,
        energies=energies,
        vectors=vectors,
        stage="diagonalize",
    )


def run_observables(length: int, output_dir: Path, chunk_columns: int) -> None:
    import h5py
    import scipy.sparse as sp

    require_stage(output_dir, "diagonalize")
    require_stage(output_dir, "basis")
    require_stage(output_dir, "hamiltonian")
    basis = _load_basis(output_dir / "basis.npz")
    matrix = sp.load_npz(output_dir / "hamiltonian.csr.npz").tocsr()
    with h5py.File(output_dir / "eigensystem.h5", "r") as handle:
        energies = handle["eigensystem/energies"][:]
        vectors = handle["eigensystem/vectors"][:]
    result = compute_observables(
        basis,
        matrix,
        energies,
        vectors,
        chunk_columns=chunk_columns,
    )
    arrays = {
        name: value
        for name, value in result.items()
        if name not in {"energies", "eigenvectors", "validation_metadata"}
    }
    write_observables(output_dir, arrays)


def run_validate(length: int, output_dir: Path) -> None:
    validated = {
        stage: require_stage(output_dir, stage)["artifact"]["sha256"]
        for stage in ("basis", "hamiltonian", "diagonalize", "observables")
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "length": length,
        "validated_stage_sha256": validated,
    }
    _write_hashed_json_stage(
        output_dir,
        "validate",
        "validation/metrics.json",
        metrics,
    )


def run_figures(length: int, output_dir: Path) -> None:
    require_stage(output_dir, "validate")
    if FIGURES_ADAPTER is None:
        raise RuntimeError(
            "figure renderer adapter is unavailable until Tasks 7/8; "
            "figures stage remains incomplete"
        )
    artifact = Path(FIGURES_ADAPTER(output_dir, length))
    if not artifact.is_file():
        raise RuntimeError(f"figure renderer did not produce its artifact: {artifact}")


def validate_small_l(
    length: int,
    official_data_dir: str | Path | None,
) -> dict[str, Any]:
    """Compare one independent-engine result to the existing full-basis oracle."""
    import numpy as np

    from pxp_ed import (
        basis_state_vector,
        constrained_basis,
        density_wave_state,
        pxp_hamiltonian,
        symmetry_basis_k0_inversion_even,
    )
    from turner2018_ed_observables import (
        compare_degenerate_invariants,
        compute_observables,
        project_product_state,
    )
    from turner2018_ed_solver import estimate_dense_resources, solve_full_eigensystem
    from turner2018_fig3 import fsa_basis

    if length not in LOCAL_VALIDATION_DIMENSIONS:
        raise ValueError(
            "local equivalence validation supports even lengths 10 through 20"
        )
    expected_full, expected_sector = LOCAL_VALIDATION_DIMENSIONS[length]

    # Candidate: direct constrained-state enumeration and direct dihedral-orbit
    # Hamiltonian assembly, followed by the Task 3/4 eigensystem and observables.
    candidate_basis = build_orbit_basis(length)
    candidate_matrix = assemble_reduced_hamiltonian(candidate_basis)
    estimate = estimate_dense_resources(candidate_matrix.shape[0], vectors=True)
    candidate_energies, candidate_vectors = solve_full_eigensystem(
        candidate_matrix,
        vectors=True,
        declared_memory_bytes=estimate.minimum_requested_bytes,
    )
    assert candidate_vectors is not None
    candidate = compute_observables(
        candidate_basis,
        candidate_matrix,
        candidate_energies,
        candidate_vectors,
    )

    # Oracle: the established full-basis pxp_ed transform and Turner Fig. 3 FSA.
    reference_states = constrained_basis(length, pbc=True)
    reference_full_matrix = pxp_hamiltonian(reference_states, length, pbc=True)
    reference_transform = symmetry_basis_k0_inversion_even(reference_states, length)
    reference_matrix = (
        reference_transform.T @ reference_full_matrix @ reference_transform
    ).tocsr()
    reference_energies, reference_vectors = np.linalg.eigh(reference_matrix.toarray())
    z2_state = density_wave_state(length, 2)
    reference_z2_full = basis_state_vector(reference_states, z2_state)
    reference_z2_sector = np.asarray(
        reference_transform.T @ reference_z2_full
    ).ravel()
    candidate_z2_sector = project_product_state(candidate_basis, z2_state)
    reference_shells_full, reference_beta = fsa_basis(
        reference_full_matrix,
        reference_states,
        z2_state,
        length,
    )
    reference_shells = np.asarray(
        reference_transform.T @ reference_shells_full.T
    ).T[: length // 2 + 1]
    reference_shells = np.asarray(reference_shells.real, dtype=np.float64)
    reference_shells /= np.linalg.norm(reference_shells, axis=1, keepdims=True)
    reference_beta = np.asarray(reference_beta)
    candidate_shells = np.asarray(
        candidate["fsa_shell_vectors_sector"],
        dtype=np.float64,
    )
    candidate_beta = np.asarray(candidate["fsa_beta_full_chain"])
    candidate_fsa_hamiltonian = np.asarray(candidate["fsa_hamiltonian_sector"])
    reference_fsa_hamiltonian = np.asarray(
        reference_shells @ reference_matrix @ reference_shells.T
    )
    expected_shell_count = length // 2 + 1
    fsa_shape_metrics = {
        "fsa_beta_shape": _shape_metric(
            (length,),
            candidate_beta.shape,
            reference_beta.shape,
        ),
        "fsa_projected_shell_shape": _shape_metric(
            (expected_shell_count, expected_sector),
            candidate_shells.shape,
            reference_shells.shape,
        ),
        "fsa_reduced_hamiltonian_shape": _shape_metric(
            (expected_shell_count, expected_shell_count),
            candidate_fsa_hamiltonian.shape,
            reference_fsa_hamiltonian.shape,
        ),
    }
    fsa_shapes_pass = all(metric["passed"] for metric in fsa_shape_metrics.values())
    failed_value = sys.float_info.max
    if fsa_shapes_pass:
        aligned_candidate_shells = candidate_shells.copy()
        for shell in range(reference_shells.shape[0]):
            if (
                np.vdot(reference_shells[shell], aligned_candidate_shells[shell]).real
                < 0
            ):
                aligned_candidate_shells[shell] *= -1.0
        fsa_beta_error = _max_abs_array(candidate_beta - reference_beta)
        fsa_shell_error = _max_abs_array(
            aligned_candidate_shells - reference_shells
        )
    else:
        fsa_beta_error = failed_value
        fsa_shell_error = failed_value

    matrix_error = _max_abs_sparse(candidate_matrix - reference_matrix)
    spectrum_error = _max_abs_array(candidate_energies - reference_energies)
    invariant_error: str | None = None
    try:
        invariants = compare_degenerate_invariants(
            reference_energies=reference_energies,
            reference_vectors=reference_vectors,
            candidate_energies=candidate_energies,
            candidate_vectors=candidate_vectors,
            reference_z2_sector_state=reference_z2_sector,
            candidate_z2_sector_state=candidate_z2_sector,
            tolerance=LOCAL_VALIDATION_TOLERANCES["complete_spectrum"],
        )
    except ValueError as error:
        invariant_error = str(error)
        invariants = {
            "max_total_z2_diff": failed_value,
            "max_projector_diag_diff": failed_value,
            "max_subspace_sine": failed_value,
            "max_pr2_diff_isolated": failed_value,
            "degenerate_group_count": 0,
            "isolated_group_count": 0,
        }

    metrics: dict[str, dict[str, Any]] = {
        "full_dimension": _dimension_metric(
            expected_full,
            len(candidate_basis.constrained_states),
            len(reference_states),
        ),
        "sector_dimension": _dimension_metric(
            expected_sector,
            candidate_matrix.shape[0],
            reference_matrix.shape[0],
        ),
        "reduced_matrix": _threshold_metric(
            matrix_error,
            LOCAL_VALIDATION_TOLERANCES["reduced_matrix"],
        ),
        "complete_spectrum": _threshold_metric(
            spectrum_error,
            LOCAL_VALIDATION_TOLERANCES["complete_spectrum"],
        ),
        "degenerate_total_z2": _threshold_metric(
            float(invariants["max_total_z2_diff"]),
            LOCAL_VALIDATION_TOLERANCES["degenerate_total_z2"],
        ),
        "degenerate_projector_diagonal": _threshold_metric(
            float(invariants["max_projector_diag_diff"]),
            LOCAL_VALIDATION_TOLERANCES["degenerate_projector_diagonal"],
        ),
        "degenerate_subspace": _threshold_metric(
            float(invariants["max_subspace_sine"]),
            LOCAL_VALIDATION_TOLERANCES["degenerate_subspace"],
        ),
        "fsa_beta": _threshold_metric(
            fsa_beta_error,
            LOCAL_VALIDATION_TOLERANCES["fsa_beta"],
        ),
        "fsa_projected_shell": _threshold_metric(
            fsa_shell_error,
            LOCAL_VALIDATION_TOLERANCES["fsa_projected_shell"],
        ),
        "pr2_isolated": _threshold_metric(
            float(invariants["max_pr2_diff_isolated"]),
            LOCAL_VALIDATION_TOLERANCES["pr2_isolated"],
        ),
        **fsa_shape_metrics,
    }
    result: dict[str, Any] = {
        "length": length,
        "passed": bool(all(metric["passed"] for metric in metrics.values())),
        "metrics": metrics,
        "full_fsa_shell_count": length + 1,
        "folded_fsa_shell_count": length // 2 + 1,
        "degenerate_group_count": int(invariants["degenerate_group_count"]),
        "isolated_group_count": int(invariants["isolated_group_count"]),
        "pr2_policy": "compared only for isolated one-dimensional eigenspaces",
        "official_data": "not-requested",
    }
    if invariant_error is not None:
        result["invariant_error"] = invariant_error
    if official_data_dir is not None:
        official = Path(official_data_dir)
        archive = official / "eigendecomposition.zip"
        result["official_data"] = "available" if archive.is_file() else "missing"
        if not archive.is_file():
            raise RuntimeError(
                f"official small-L comparison requested but archive is missing: {archive}"
            )
    return result


def run_local_validation(
    lengths: list[int],
    output_dir: Path,
    official_data_dir: Path | None,
    invocation: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Run every requested local comparison and atomically publish one gate."""
    if tuple(lengths) != LOCAL_VALIDATION_LENGTHS:
        raise ValueError(
            "--validate-local requires exactly 10 12 14 16 18 20 in that order"
        )
    canonical_lengths = list(LOCAL_VALIDATION_LENGTHS)
    path = output_dir / "validation" / "local-equivalence.json"
    running = {
        "schema_version": LOCAL_EQUIVALENCE_SCHEMA_VERSION,
        "status": "running",
        "passed": False,
        "requested_lengths": canonical_lengths,
    }
    # Initial publication failure is unrecoverable: without this marker, an old
    # passing summary may still be authoritative.
    atomic_write_json(path, running)

    phase = "source_hashing"
    try:
        source_hashes = _source_hashes()
        phase = "provenance"
        provenance = _local_validation_provenance(invocation)
        phase = "per_length_validation"
        results: list[dict[str, Any]] = []
        for length in canonical_lengths:
            try:
                result = validate_small_l(length, official_data_dir)
            except Exception as error:
                result = {
                    "length": length,
                    "passed": False,
                    "metrics": {
                        "execution": {
                            "passed": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    },
                }
            results.append(result)
        phase = "aggregate_construction"
        passed = all(result["passed"] for result in results)
        summary = {
            "schema_version": LOCAL_EQUIVALENCE_SCHEMA_VERSION,
            "status": "passed" if passed else "failed",
            "passed": passed,
            "requested_lengths": canonical_lengths,
            "results": results,
            "artifact_hashes": source_hashes,
            "provenance": provenance,
        }
        phase = "serialization"
        json.dumps(summary, sort_keys=True, allow_nan=False)
    except Exception as error:
        failed = {
            "schema_version": LOCAL_EQUIVALENCE_SCHEMA_VERSION,
            "status": "failed",
            "passed": False,
            "requested_lengths": canonical_lengths,
            "failure_phase": phase,
            "error": f"{type(error).__name__}: {error}",
        }
        atomic_write_json(path, failed)
        return path, failed

    # Keep the running marker authoritative if final atomic publication fails.
    atomic_write_json(path, summary)
    return path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--stage", choices=STAGES)
    execution_mode.add_argument("--validate-local", type=int, nargs="+")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tracks/ed/results/turner-2018/l32-server"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-columns", type=int, default=32)
    parser.add_argument(
        "--declared-memory",
        help="total memory allocation (for example 512G); defaults to Slurm variables",
    )
    parser.add_argument("--validate-small-l", type=int)
    parser.add_argument("--official-data-dir", type=Path)
    return parser


def _record_failure(output_dir: Path, stage: str, error: Exception) -> None:
    atomic_write_json(
        output_dir / "stages" / f"{stage}.failure.json",
        {
            "stage": stage,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    invoked = list(argv) if argv is not None else sys.argv[1:]
    if args.validate_local is not None:
        if tuple(args.validate_local) != LOCAL_VALIDATION_LENGTHS:
            raise SystemExit(
                "--validate-local requires exactly 10 12 14 16 18 20 in that order"
            )
        conflicting_options = {
            "--dry-run",
            "--length",
            "--chunk-columns",
            "--declared-memory",
            "--validate-small-l",
            "--official-data-dir",
        }
        raw_options = {
            token.split("=", 1)[0]
            for token in invoked
            if token.startswith("--")
        }
        if raw_options & conflicting_options:
            raise SystemExit(
                "--validate-local is mutually exclusive with execution-mode options"
            )
        path, summary = run_local_validation(
            list(LOCAL_VALIDATION_LENGTHS),
            args.output_dir,
            args.official_data_dir,
            invoked,
        )
        print(path, flush=True)
        return 0 if summary["passed"] else 1
    if args.stage is None:
        raise SystemExit("--stage is required unless --validate-local is used")
    if args.length < 4 or args.length % 2:
        raise SystemExit("--length must be an even integer >= 4")
    if args.chunk_columns < 1:
        raise SystemExit("--chunk-columns must be positive")

    plan_manifest = args.output_dir / "stages" / "plan.json"
    try:
        if plan_manifest.is_file():
            _load_plan(args.output_dir, args.length)
            print("skipped stage=plan", flush=True)
        else:
            write_plan(args.length, args.output_dir, invoked)
            print("completed stage=plan", flush=True)
    except Exception as error:
        _record_failure(args.output_dir, "plan", error)
        raise
    if args.validate_small_l is not None:
        metrics = validate_small_l(args.validate_small_l, args.official_data_dir)
        atomic_write_json(args.output_dir / "validation" / "metrics.json", metrics)
    if args.stage == "plan" or args.dry_run:
        print(args.output_dir / "manifest.json", flush=True)
        return 0

    requested = (
        COMPUTATIONAL_STAGES
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in requested:
        if stage == "plan":
            continue
        try:
            stage_manifest = args.output_dir / "stages" / f"{stage}.json"
            if stage_manifest.is_file():
                validate_stage(args.output_dir, stage)
                print(f"skipped stage={stage}", flush=True)
                continue
            predecessor = STAGE_PREDECESSOR.get(stage)
            if predecessor is not None:
                require_stage(args.output_dir, predecessor)
            if stage == "basis":
                run_basis(args.length, args.output_dir)
            elif stage == "hamiltonian":
                run_hamiltonian(args.length, args.output_dir)
            elif stage == "diagonalize":
                declared_memory = resolve_declared_memory_bytes(
                    os.environ,
                    args.declared_memory,
                )
                run_diagonalize(
                    args.length,
                    args.output_dir,
                    declared_memory,
                    args.chunk_columns,
                )
            elif stage == "observables":
                run_observables(args.length, args.output_dir, args.chunk_columns)
            elif stage == "validate":
                run_validate(args.length, args.output_dir)
            elif stage == "figures":
                run_figures(args.length, args.output_dir)
            print(f"completed stage={stage}", flush=True)
        except Exception as error:
            _record_failure(args.output_dir, stage, error)
            raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"turner2018_l32_server: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
