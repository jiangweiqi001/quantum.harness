#!/usr/bin/env python3
"""Restartable, fail-closed Turner PXP ED/FSA server workflow.

The L=32 path is intentionally blocked until this checkout contains a local
QuSpin 1.0.1 imported/user-basis validation proof. Small systems remain
available for validating the stage mechanics and existing reference code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from turner2018_ed_engine import assemble_reduced_hamiltonian, build_orbit_basis

L32_FULL_DIMENSION = 4_870_847
L32_SECTOR_DIMENSION = 77_436
FULL_FSA_SHELLS = 33
SYMMETRY_FOLDED_FSA_SHELLS = 17
SCHEMA_VERSION = "turner-l32-ed-fsa-v1"
LOCAL_EQUIVALENCE_SCHEMA_VERSION = "turner-local-equivalence-v1"
QUANTUM_MODEL = "H=sum_j P_(j-1) X_j P_(j+1), PBC"
STAGES = ("plan", "basis", "hamiltonian", "diagonalize", "observables", "all")
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
}


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
    """Load a completed stage manifest or fail before touching downstream data."""
    path = Path(output_dir) / "stages" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"required stage {stage!r} is not complete: missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or payload.get("status") != "complete":
        raise RuntimeError(f"required stage {stage!r} is not complete: {path}")
    return payload


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
    """Build the immutable scientific/resource plan and current readiness result."""
    dimension = L32_SECTOR_DIMENSION if length == 32 else None
    packages = _package_versions()
    quspin_proof_present = packages["quspin"] == "1.0.1" and (
        output_dir / "validation" / "quspin_imported_basis_proof.json"
    ).is_file()
    readiness = "ready" if length != 32 else "blocked"
    blocker = None
    if readiness == "blocked":
        blocker = (
            "L=32 direct QuSpin imported/user-basis translation/reflection "
            "reduction is not locally proven; a proof file alone does not "
            "enable the unimplemented adapter"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "readiness": readiness,
        "blocker": blocker,
        "invocation": argv,
        "model": {
            "hamiltonian": QUANTUM_MODEL,
            "length": length,
            "boundary": "periodic",
            "momentum": 0,
            "inversion": "even",
        },
        "basis": {
            "constructor": "QuSpin 1.0.1 imported/user basis",
            "enumeration": "direct constrained states; never scan 2**L",
            "symmetry_reduction": "direct translation then reflection",
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
        "resources": dense_resource_estimate(dimension or 0),
        "artifacts": {
            "basis": "basis.npz",
            "hamiltonian": "hamiltonian.csr.npz",
            "results": "results.h5",
            "validation": "validation/metrics.json",
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "quspin_proof_present": quspin_proof_present,
            "git_revision": _git_revision(),
        },
    }


def write_plan(length: int, output_dir: Path, argv: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(length, output_dir, argv)
    atomic_write_json(output_dir / "manifest.json", plan)
    atomic_write_json(
        output_dir / "stages" / "plan.json",
        {"stage": "plan", "status": "complete", "readiness": plan["readiness"]},
    )
    return plan


def _refuse_unproven_l32(length: int, output_dir: Path) -> None:
    if length != 32:
        return
    proof = output_dir / "validation" / "quspin_imported_basis_proof.json"
    packages = _package_versions()
    if packages["quspin"] != "1.0.1" or not proof.is_file():
        raise RuntimeError(
            "refusing L=32: no validated QuSpin 1.0.1 imported-basis proof; "
            "run and inspect local small-L equivalence first"
        )
    payload = json.loads(proof.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("quspin_version") != "1.0.1":
        raise RuntimeError(
            "refusing L=32: validated QuSpin 1.0.1 imported-basis proof is incomplete"
        )
    # A proof file alone is deliberately insufficient while the direct imported
    # basis adapter remains unimplemented in this checkout.
    raise RuntimeError(
        "refusing L=32: QuSpin direct imported-basis adapter is not locally proven"
    )


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, path)


def run_basis(length: int, output_dir: Path) -> None:
    _refuse_unproven_l32(length, output_dir)
    from pxp_ed import constrained_basis, symmetry_basis_k0_inversion_even

    basis = constrained_basis(length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(basis, length)
    _atomic_save_npz(
        output_dir / "basis.npz",
        states=basis,
        transform_data=transform.data,
        transform_indices=transform.indices,
        transform_indptr=transform.indptr,
        transform_shape=transform.shape,
    )
    atomic_write_json(
        output_dir / "stages" / "basis.json",
        {
            "stage": "basis",
            "status": "complete",
            "full_dimension": len(basis),
            "sector_dimension": transform.shape[1],
            "artifact_sha256": _sha256(output_dir / "basis.npz"),
        },
    )


def _load_basis(path: Path):
    import numpy as np
    import scipy.sparse as sp

    with np.load(path, allow_pickle=False) as data:
        states = data["states"]
        shape = tuple(int(value) for value in data["transform_shape"])
        transform = sp.csc_matrix(
            (data["transform_data"], data["transform_indices"], data["transform_indptr"]),
            shape=shape,
        )
    return states, transform


def run_hamiltonian(length: int, output_dir: Path) -> None:
    import scipy.sparse as sp

    require_stage(output_dir, "basis")
    _refuse_unproven_l32(length, output_dir)
    from pxp_ed import pxp_hamiltonian

    states, transform = _load_basis(output_dir / "basis.npz")
    full = pxp_hamiltonian(states, length, pbc=True)
    reduced = (transform.T @ full @ transform).tocsr()
    target = output_dir / "hamiltonian.csr.npz"
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as handle:
        sp.save_npz(handle, reduced)
    os.replace(partial, target)
    atomic_write_json(
        output_dir / "stages" / "hamiltonian.json",
        {
            "stage": "hamiltonian",
            "status": "complete",
            "shape": list(reduced.shape),
            "nnz": reduced.nnz,
            "artifact_sha256": _sha256(target),
        },
    )


def run_diagonalize(length: int, output_dir: Path) -> None:
    import h5py
    import numpy as np
    import scipy.linalg
    import scipy.sparse as sp

    require_stage(output_dir, "hamiltonian")
    _refuse_unproven_l32(length, output_dir)
    sparse = sp.load_npz(output_dir / "hamiltonian.csr.npz")
    dense = np.asfortranarray(sparse.toarray(), dtype=np.float64)
    if not dense.flags.f_contiguous or dense.dtype != np.float64:
        raise RuntimeError("dense Hamiltonian must be Fortran-contiguous float64")
    energies, vectors = scipy.linalg.eigh(
        dense,
        driver="evd",
        overwrite_a=True,
        check_finite=False,
    )
    target = output_dir / "results.h5"
    partial = target.with_name(target.name + ".partial")
    with h5py.File(partial, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        eig = handle.create_group("eigensystem")
        eig.create_dataset("energies", data=energies)
        eig.create_dataset("vectors", data=vectors, chunks=(vectors.shape[0], 1))
    os.replace(partial, target)
    atomic_write_json(
        output_dir / "stages" / "diagonalize.json",
        {
            "stage": "diagonalize",
            "status": "complete",
            "dimension": len(energies),
            "artifact_sha256": _sha256(target),
        },
    )


def _stream_pr2(vectors: Any, chunk_columns: int) -> Any:
    import numpy as np

    result = np.empty(vectors.shape[1], dtype=np.float64)
    for start in range(0, vectors.shape[1], chunk_columns):
        stop = min(start + chunk_columns, vectors.shape[1])
        block = vectors[:, start:stop]
        squared = np.square(block, dtype=np.float64)
        result[start:stop] = np.sum(squared * squared, axis=0)
    return result


def run_observables(length: int, output_dir: Path, chunk_columns: int) -> None:
    import h5py
    import numpy as np

    require_stage(output_dir, "diagonalize")
    _refuse_unproven_l32(length, output_dir)
    from pxp_ed import density_wave_state, pxp_hamiltonian
    from turner2018_fig3 import fsa_basis

    states, transform = _load_basis(output_dir / "basis.npz")
    z2_state = density_wave_state(length, 2)
    full_h = pxp_hamiltonian(states, length, pbc=True)
    full_shells, beta = fsa_basis(full_h, states, z2_state, length)
    folded = np.asarray(transform.T @ full_shells.T).T[: length // 2 + 1]
    folded /= np.linalg.norm(folded, axis=1)[:, None]

    path = output_dir / "results.h5"
    partial = path.with_name(path.name + ".partial")
    with h5py.File(path, "r") as source, h5py.File(partial, "w") as target:
        source.copy("eigensystem", target)
        vectors = source["eigensystem/vectors"]
        observables = target.create_group("observables")
        observables.create_dataset("pr2", data=_stream_pr2(vectors, chunk_columns))
        fsa = target.create_group("fsa")
        fsa.create_dataset("full_shell_vectors", data=full_shells)
        fsa.create_dataset("full_beta", data=beta)
        fsa.create_dataset("symmetry_folded_shell_vectors", data=folded)
        fsa.attrs["full_shell_count"] = length + 1
        fsa.attrs["symmetry_folded_shell_count"] = length // 2 + 1
    os.replace(partial, path)
    atomic_write_json(
        output_dir / "stages" / "observables.json",
        {
            "stage": "observables",
            "status": "complete",
            "pr2_chunk_columns": chunk_columns,
            "full_fsa_shell_count": length + 1,
            "folded_fsa_shell_count": length // 2 + 1,
            "artifact_sha256": _sha256(path),
        },
    )


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
    parser.add_argument("--validate-small-l", type=int)
    parser.add_argument("--official-data-dir", type=Path)
    return parser


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

    write_plan(args.length, args.output_dir, invoked)
    if args.validate_small_l is not None:
        metrics = validate_small_l(args.validate_small_l, args.official_data_dir)
        atomic_write_json(args.output_dir / "validation" / "metrics.json", metrics)
    if args.stage == "plan" or args.dry_run:
        print(args.output_dir / "manifest.json", flush=True)
        return 0

    requested = (
        ("basis", "hamiltonian", "diagonalize", "observables")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in requested:
        predecessor = STAGE_PREDECESSOR.get(stage)
        if predecessor is not None:
            require_stage(args.output_dir, predecessor)
        if stage == "basis":
            run_basis(args.length, args.output_dir)
        elif stage == "hamiltonian":
            run_hamiltonian(args.length, args.output_dir)
        elif stage == "diagonalize":
            run_diagonalize(args.length, args.output_dir)
        elif stage == "observables":
            run_observables(args.length, args.output_dir, args.chunk_columns)
        print(f"completed stage={stage}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"turner2018_l32_server: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
