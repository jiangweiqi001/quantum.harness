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
import uuid

from turner2018_ed_artifacts import (
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
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
from turner2018_ed_observables import compute_observables, compute_observables_from_h5
from turner2018_ed_solver import estimate_dense_resources, solve_full_eigensystem

L32_FULL_DIMENSION = 4_870_847
L32_SECTOR_DIMENSION = 77_436
FULL_FSA_SHELLS = 33
SYMMETRY_FOLDED_FSA_SHELLS = 17
SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION
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
PRODUCTION_DIMENSIONS = {
    10: (123, 14),
    28: (710_647, 13_201),
    30: (1_860_498, 31_836),
    32: (4_870_847, 77_436),
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
STAGE_DEPENDENCIES = {
    "plan": (),
    "basis": (),
    "hamiltonian": ("basis",),
    "diagonalize": ("hamiltonian",),
    "observables": ("basis", "hamiltonian", "diagonalize"),
    "validate": ("basis", "hamiltonian", "diagonalize", "observables"),
    "figures": ("validate",),
}
COMPUTATIONAL_STAGES = (
    "basis",
    "hamiltonian",
    "diagonalize",
    "observables",
    "validate",
)


def _render_fig3_adapter(output_dir: Path, length: int) -> Path:
    from turner2018_fig3 import render_fig3_stage

    return render_fig3_stage(output_dir, length)


def _render_fig4_adapter(output_dir: Path, length: int) -> Path:
    from turner2018_fig4 import render_fig4_stage

    return render_fig4_stage(output_dir, length)


FIG3_RENDERER_ADAPTER: Callable[[Path, int], Path] = _render_fig3_adapter
FIG4_RENDERER_ADAPTER: Callable[[Path, int], Path] = _render_fig4_adapter


class StaleStageError(RuntimeError):
    """A structurally valid stage no longer matches its dependencies."""


class StalePlanError(StaleStageError):
    """The checkout/runtime no longer matches the stored execution plan."""


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


def require_stage(
    output_dir: str | Path,
    stage: str,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a hash-validated completed stage or fail closed."""
    path = Path(output_dir) / "stages" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"required stage {stage!r} is not complete: missing {path}")
    try:
        return _validate_current_stage(Path(output_dir), stage, validation_cache)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(
            f"required stage {stage!r} is not complete or valid: {error}"
        ) from error


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _stage_manifest_sha256(output_dir: Path, stage: str) -> str:
    return _sha256(output_dir / "stages" / f"{stage}.json")


def _plan_config_sha256(output_dir: Path) -> str:
    plan = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    value = plan.get("scientific_config_sha256")
    if not isinstance(value, str):
        raise RuntimeError("plan is missing scientific_config_sha256")
    return value


def _stage_inputs(output_dir: Path, stage: str) -> dict[str, str]:
    return {
        dependency: _stage_manifest_sha256(output_dir, dependency)
        for dependency in STAGE_DEPENDENCIES[stage]
    }


def _validate_current_stage(
    output_dir: Path,
    stage: str,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if validation_cache is not None and stage in validation_cache:
        return validation_cache[stage]
    payload = validate_stage(output_dir, stage)
    for dependency in STAGE_DEPENDENCIES[stage]:
        _validate_current_stage(output_dir, dependency, validation_cache)
    expected_inputs = _stage_inputs(output_dir, stage)
    if payload.get("inputs") != expected_inputs:
        raise StaleStageError(
            f"stage={stage} input hashes are stale: "
            f"recorded={payload.get('inputs')} current={expected_inputs}"
        )
    expected_plan = _plan_config_sha256(output_dir)
    if payload.get("plan_sha256") != expected_plan:
        raise StaleStageError(f"stage={stage} scientific plan hash is stale")
    if stage == "figures":
        _validate_figures_stage_references(output_dir, payload)
    if validation_cache is not None:
        validation_cache[stage] = payload
    return payload


def _git_revision() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
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
    explicit_bytes = None if explicit is None else parse_memory_bytes(explicit)
    scheduler_bytes: int | None = None
    per_node = environment.get("SLURM_MEM_PER_NODE")
    if per_node:
        scheduler_bytes = parse_memory_bytes(per_node)
    elif environment.get("SLURM_MEM_PER_CPU"):
        per_cpu = parse_memory_bytes(environment["SLURM_MEM_PER_CPU"])
        raw_cpus = environment.get("SLURM_CPUS_PER_TASK") or environment.get(
            "SLURM_CPUS_ON_NODE"
        )
        if raw_cpus is None:
            raise RuntimeError(
                "SLURM_MEM_PER_CPU requires SLURM_CPUS_PER_TASK "
                "or SLURM_CPUS_ON_NODE"
            )
        try:
            cpu_count = int(raw_cpus)
        except ValueError as error:
            raise ValueError(f"invalid declared Slurm CPU count: {raw_cpus!r}") from error
        if cpu_count < 1:
            raise ValueError("declared Slurm CPU count must be positive")
        scheduler_bytes = per_cpu * cpu_count
    if scheduler_bytes is not None and explicit_bytes is not None:
        return min(scheduler_bytes, explicit_bytes)
    if scheduler_bytes is not None:
        return scheduler_bytes
    if explicit_bytes is not None:
        return explicit_bytes
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
        "turner2018_fig4.py",
        "turner2018_ed_artifacts.py",
        "turner2018_ed_engine.py",
        "turner2018_ed_solver.py",
        "turner2018_ed_observables.py",
        "turner2018_ed_validation.py",
        "turner2018_l32_server.py",
    )
    return {name: _sha256(root / name) for name in names}


def build_execution_fingerprint() -> dict[str, Any]:
    """Fingerprint only production code, lock bytes, and numerical runtime."""
    repository_root = Path(__file__).resolve().parents[1]
    versions = _package_versions()
    return {
        "sources": _source_hashes(),
        "uv_lock_sha256": _sha256(repository_root / "uv.lock"),
        "python": [sys.version_info.major, sys.version_info.minor],
        "packages": {
            name: versions[name]
            for name in ("numpy", "scipy", "h5py")
        },
    }


def _fingerprint_sha256(fingerprint: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            fingerprint,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
    production_dimensions = {
        production_length: sector_dimension
        for production_length, (_, sector_dimension) in PRODUCTION_DIMENSIONS.items()
        if production_length in {28, 30, 32}
    }
    dimension = production_dimensions.get(length)
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
    plan = {
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
                PRODUCTION_DIMENSIONS[length][0]
                if length in production_dimensions
                else None
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
    fingerprint = build_execution_fingerprint()
    plan["execution_fingerprint"] = fingerprint
    plan["execution_fingerprint_sha256"] = _fingerprint_sha256(fingerprint)
    scientific_config = {
        key: plan[key]
        for key in (
            "model",
            "basis",
            "fsa",
            "solver",
            "observables",
            "execution_fingerprint_sha256",
        )
    }
    plan["scientific_config_sha256"] = hashlib.sha256(
        json.dumps(
            scientific_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return plan


def _write_hashed_json_stage(
    output_dir: Path,
    stage: str,
    relative_path: str,
    payload: dict[str, Any],
    *,
    inputs: dict[str, str],
    plan_sha256: str,
) -> dict[str, Any]:
    target = output_dir / relative_path
    atomic_write_json(target, payload)
    stage_payload = {
        "schema_version": SCHEMA_VERSION,
        "generation_id": str(uuid.uuid4()),
        "stage": stage,
        "status": "complete",
        "inputs": inputs,
        "plan_sha256": plan_sha256,
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
    _write_hashed_json_stage(
        output_dir,
        "plan",
        "manifest.json",
        plan,
        inputs={},
        plan_sha256=plan["scientific_config_sha256"],
    )
    return plan


def _load_plan(
    output_dir: Path,
    length: int,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require_stage(output_dir, "plan", validation_cache)
    plan = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    planned_length = plan.get("model", {}).get("length")
    if planned_length != length:
        raise RuntimeError(
            f"output directory was planned for L={planned_length}, not L={length}"
        )
    current_fingerprint = build_execution_fingerprint()
    if (
        plan.get("execution_fingerprint") != current_fingerprint
        or plan.get("execution_fingerprint_sha256")
        != _fingerprint_sha256(current_fingerprint)
    ):
        raise StalePlanError("stored plan execution fingerprint is stale")
    return plan


def run_basis(length: int, output_dir: Path) -> None:
    basis = build_orbit_basis(length)
    write_basis_artifact(
        output_dir,
        basis=basis.constrained_states,
        representatives=basis.representatives,
        orbit_sizes=basis.orbit_sizes,
        length=length,
        inputs=_stage_inputs(output_dir, "basis"),
        plan_sha256=_plan_config_sha256(output_dir),
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


def run_hamiltonian(
    length: int,
    output_dir: Path,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
    require_stage(output_dir, "basis", validation_cache)
    basis = _load_basis(output_dir / "basis.npz")
    if basis.length != length:
        raise RuntimeError("basis artifact length does not match requested length")
    write_hamiltonian_artifact(
        output_dir,
        hamiltonian=assemble_reduced_hamiltonian(basis),
        inputs=_stage_inputs(output_dir, "hamiltonian"),
        plan_sha256=_plan_config_sha256(output_dir),
    )


def run_diagonalize(
    length: int,
    output_dir: Path,
    declared_memory_bytes: int,
    chunk_columns: int,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
    import scipy.sparse as sp

    require_stage(output_dir, "hamiltonian", validation_cache)
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
        inputs=_stage_inputs(output_dir, "diagonalize"),
        plan_sha256=_plan_config_sha256(output_dir),
    )


def run_observables(
    length: int,
    output_dir: Path,
    chunk_columns: int,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
    import h5py
    import scipy.sparse as sp

    require_stage(output_dir, "diagonalize", validation_cache)
    require_stage(output_dir, "basis", validation_cache)
    require_stage(output_dir, "hamiltonian", validation_cache)
    basis = _load_basis(output_dir / "basis.npz")
    matrix = sp.load_npz(output_dir / "hamiltonian.csr.npz").tocsr()
    with h5py.File(output_dir / "eigensystem.h5", "r") as handle:
        energies = handle["eigensystem/energies"][()]
        result = compute_observables_from_h5(
            basis,
            matrix,
            energies,
            handle["eigensystem/vectors"],
            chunk_columns=chunk_columns,
        )
    arrays = {
        name: value
        for name, value in result.items()
        if name not in {"energies", "eigenvectors", "validation_metadata"}
    }
    write_observables(
        output_dir,
        arrays,
        metadata=result["validation_metadata"],
        inputs=_stage_inputs(output_dir, "observables"),
        plan_sha256=_plan_config_sha256(output_dir),
    )


def run_validate(
    length: int,
    output_dir: Path,
    validation_cache: dict[str, dict[str, Any]] | None = None,
) -> None:
    import h5py
    import numpy as np
    import scipy.sparse as sp

    validated = {}
    for stage in ("basis", "hamiltonian", "diagonalize", "observables"):
        validated[stage] = require_stage(
            output_dir, stage, validation_cache
        )["artifact"]["sha256"]
    checks: dict[str, dict[str, Any]] = {}
    try:
        basis = _load_basis(output_dir / "basis.npz")
        matrix = sp.load_npz(output_dir / "hamiltonian.csr.npz").tocsr()
        expected_full, expected_sector = PRODUCTION_DIMENSIONS.get(
            length,
            (len(basis.constrained_states), len(basis.representatives)),
        )
        checks["basis_full_dimension"] = {
            "value": len(basis.constrained_states),
            "expected": expected_full,
            "passed": len(basis.constrained_states) == expected_full,
        }
        checks["basis_sector_dimension"] = {
            "value": len(basis.representatives),
            "expected": expected_sector,
            "passed": len(basis.representatives) == expected_sector,
        }
        matrix_values_finite = bool(np.all(np.isfinite(matrix.data)))
        checks["hamiltonian_shape"] = {
            "value": list(matrix.shape),
            "expected": [expected_sector, expected_sector],
            "passed": matrix.shape == (expected_sector, expected_sector),
        }
        checks["hamiltonian_finite"] = {
            "value": matrix_values_finite,
            "expected": True,
            "passed": matrix_values_finite,
        }

        with h5py.File(output_dir / "eigensystem.h5", "r") as handle:
            energies_dataset = handle["eigensystem/energies"]
            vectors = handle["eigensystem/vectors"]
            energies = energies_dataset[()]
            checks["energies_finite"] = {
                "value": bool(np.all(np.isfinite(energies))),
                "expected": True,
                "passed": bool(np.all(np.isfinite(energies))),
            }
            sorted_energies = bool(
                energies.shape == (expected_sector,)
                and np.all(np.diff(energies) >= 0)
            )
            checks["energies_sorted"] = {
                "value": sorted_energies,
                "expected_length": expected_sector,
                "passed": sorted_energies,
            }
            vector_metadata_passed = (
                vectors.shape == (expected_sector, expected_sector)
                and vectors.dtype == np.float64
                and vectors.chunks == (expected_sector, 1)
            )
            checks["eigenvector_metadata"] = {
                "value": {
                    "shape": list(vectors.shape),
                    "dtype": str(vectors.dtype),
                    "chunks": list(vectors.chunks or ()),
                },
                "expected": {
                    "shape": [expected_sector, expected_sector],
                    "dtype": "float64",
                    "chunks": [expected_sector, 1],
                },
                "passed": vector_metadata_passed,
            }
            streamed = compute_observables_from_h5(
                basis,
                matrix,
                energies,
                vectors,
                chunk_columns=32,
            )

        metadata = streamed["validation_metadata"]
        checks["hamiltonian_hermiticity"] = _threshold_metric(
            float(metadata["max_hermiticity_error"]), 1e-10
        )
        checks["eigenvector_norm"] = _threshold_metric(
            float(metadata["max_norm_error"]), 1e-10
        )
        checks["eigenpair_residual"] = _threshold_metric(
            float(metadata["max_eigenpair_residual"]), 1e-10
        )
        checks["eigenvector_orthogonality"] = _threshold_metric(
            float(metadata["max_sample_overlap"]), 1e-10
        )
        overlap_sum_error = abs(float(np.sum(streamed["overlap_z2"])) - 0.5)
        checks["z2_overlap_sum"] = _threshold_metric(overlap_sum_error, 1e-10)

        expected_shapes = {
            "overlap_z2": (expected_sector,),
            "participation_ratio": (expected_sector,),
            "exact_shell_amplitudes": (length // 2 + 1, expected_sector),
            "fsa_shell_vectors_sector": (length // 2 + 1, expected_sector),
            "fsa_hamiltonian_sector": (length // 2 + 1, length // 2 + 1),
            "fsa_beta_full_chain": (length,),
        }
        observable_shapes_passed = True
        observables_finite = True
        max_observable_error = 0.0
        persisted_overlap_sum = 0.0
        with h5py.File(output_dir / "observables.h5", "r") as handle:
            group = handle["observables"]
            recorded_metadata = json.loads(group.attrs["validation_metadata"])
            for name, expected_shape in expected_shapes.items():
                dataset = group[name]
                observable_shapes_passed &= dataset.shape == expected_shape
                for start in range(0, dataset.shape[0], 32):
                    block = dataset[start : start + 32]
                    observables_finite &= bool(np.all(np.isfinite(block)))
                    expected_block = np.asarray(streamed[name])[start : start + 32]
                    max_observable_error = max(
                        max_observable_error,
                        float(np.max(np.abs(block - expected_block))),
                    )
                if name == "overlap_z2":
                    persisted_overlap_sum = float(np.sum(dataset[()]))
            metadata_passed = (
                recorded_metadata.get("eigenvector_access") == "column-chunked"
                and recorded_metadata.get("finite_columns_checked")
                == expected_sector
                and recorded_metadata.get("residual_columns_checked")
                == expected_sector
            )
        checks["observables_shapes"] = {
            "value": observable_shapes_passed,
            "expected": True,
            "passed": observable_shapes_passed,
        }
        checks["observables_finite"] = {
            "value": observables_finite,
            "expected": True,
            "passed": observables_finite,
        }
        checks["observables_metadata"] = {
            "value": metadata_passed,
            "expected": True,
            "passed": metadata_passed,
        }
        checks["observables_consistency"] = _threshold_metric(
            max_observable_error,
            1e-12,
        )
        checks["z2_overlap_sum"] = _threshold_metric(
            abs(persisted_overlap_sum - 0.5),
            1e-10,
        )
        passed = all(check["passed"] for check in checks.values())
    except Exception as error:
        checks["execution"] = {
            "value": f"{type(error).__name__}: {error}",
            "passed": False,
        }
        passed = False

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "length": length,
        "validated_stage_sha256": validated,
        "metrics": checks,
        "eigenvector_access": "column-chunked",
    }
    if not passed:
        atomic_write_json(output_dir / "validation" / "metrics.json", metrics)
        raise RuntimeError("internal scientific validation failed")
    _write_hashed_json_stage(
        output_dir,
        "validate",
        "validation/metrics.json",
        metrics,
        inputs=_stage_inputs(output_dir, "validate"),
        plan_sha256=_plan_config_sha256(output_dir),
    )


def _validate_fig3_shell_selection(
    metrics: dict[str, Any],
    arrays: dict[str, Any],
    *,
    length: int,
    upstream: dict[str, Any],
) -> None:
    import numpy as np

    failure = "Fig. 3 shell selection metadata is invalid"
    if metrics.get("primary_length") != length:
        raise RuntimeError(f"{failure}: stale primary length")
    expected_plot_convention = {
        "exact": "black circles, solid line",
        "fsa": "red crosses, dashed line",
        "x": "folded FSA shell index n=0..L/2",
        "y": "squared shell weight, linear",
    }
    conventions = metrics.get("plot_conventions", {})
    if any(
        conventions.get(f"panel_{panel}") != expected_plot_convention
        for panel in ("b", "c")
    ):
        raise RuntimeError(f"{failure}: plot conventions")
    selected = metrics.get("selected_panel_states")
    if not isinstance(selected, list) or len(selected) != 2:
        raise RuntimeError(f"{failure}: panel records")
    expected_roles = {
        "b": "lowest-matched-scar",
        "c": "negative-adjacent-to-zero",
    }
    plotted_count = length // 2 + 1
    folding = "k=0 inversion-even: n and L-n are symmetry-related"
    selected_by_panel: dict[str, dict[str, Any]] = {}
    for record in selected:
        if not isinstance(record, dict):
            raise RuntimeError(f"{failure}: panel record type")
        panel = record.get("panel")
        if panel not in expected_roles or panel in selected_by_panel:
            raise RuntimeError(f"{failure}: panel identities")
        selected_by_panel[panel] = record
        if (
            record.get("role") != expected_roles[panel]
            or record.get("full_fsa_shell_count") != length + 1
            or record.get("plotted_folded_shell_count") != plotted_count
            or record.get("folding") != folding
        ):
            raise RuntimeError(f"{failure}: role or shell counts")
        integer_fields = ("exact_index", "fsa_index")
        if any(
            isinstance(record.get(field), bool)
            or not isinstance(record.get(field), int)
            or record[field] < 0
            for field in integer_fields
        ):
            raise RuntimeError(f"{failure}: selected indices")
        if record["fsa_index"] >= plotted_count:
            raise RuntimeError(f"{failure}: FSA index exceeds shell count")
        numeric_fields = (
            "exact_energy",
            "fsa_energy",
            "match_strength",
            "zero_tolerance",
            "exact_weight_sum",
            "fsa_weight_sum",
        )
        if any(
            isinstance(record.get(field), bool)
            or not isinstance(record.get(field), (int, float))
            or not np.isfinite(record[field])
            for field in numeric_fields
        ):
            raise RuntimeError(f"{failure}: nonfinite numerical evidence")
        if (
            record["zero_tolerance"] <= 0
            or record["match_strength"] < 0
            or record["exact_weight_sum"] <= 0
            or record["fsa_weight_sum"] <= 0
            or record["match_strength"] > record["exact_weight_sum"] + 1e-12
        ):
            raise RuntimeError(f"{failure}: normalization evidence")
        if panel == "c" and not (
            record["exact_energy"] < -record["zero_tolerance"]
        ):
            raise RuntimeError(f"{failure}: negative adjacent state")

        prefix = f"panel_{panel}_L{length}_"
        try:
            shell = np.asarray(arrays[prefix + "shell"])
            exact_weights = np.asarray(arrays[prefix + "exact_weights"])
            fsa_weights = np.asarray(arrays[prefix + "fsa_weights"])
        except KeyError as error:
            raise RuntimeError(f"{failure}: missing plotted arrays") from error
        if (
            shell.shape != (plotted_count,)
            or exact_weights.shape != (plotted_count,)
            or fsa_weights.shape != (plotted_count,)
            or not np.array_equal(shell, np.arange(plotted_count))
            or not np.all(np.isfinite(exact_weights))
            or not np.all(np.isfinite(fsa_weights))
            or np.any(exact_weights < 0)
            or np.any(fsa_weights < 0)
        ):
            raise RuntimeError(f"{failure}: plotted shell arrays")
        if not np.isclose(
            np.sum(exact_weights),
            record["exact_weight_sum"],
            atol=1e-12,
            rtol=1e-10,
        ) or not np.isclose(
            np.sum(fsa_weights),
            record["fsa_weight_sum"],
            atol=1e-12,
            rtol=1e-10,
        ):
            raise RuntimeError(f"{failure}: normalization mismatch")
        if not np.isclose(record["fsa_weight_sum"], 1.0, atol=1e-12, rtol=1e-10):
            raise RuntimeError(f"{failure}: FSA normalization")

    per_length = metrics.get("lengths", {}).get(str(length))
    if not isinstance(per_length, dict):
        raise RuntimeError(f"{failure}: missing per-length source evidence")
    fsa = per_length.get("fsa", {})
    if (
        fsa.get("full_fsa_shell_count") != length + 1
        or fsa.get("plotted_folded_shell_count") != plotted_count
        or fsa.get("folding") != folding
        or fsa.get("selected_states") != selected
    ):
        raise RuntimeError(f"{failure}: inconsistent per-length evidence")

    source_prefix = f"selection_L{length}_"
    try:
        sidecar_energies = np.asarray(arrays[f"panel_a_L{length}_energies"])
        sidecar_amplitudes = np.asarray(
            arrays[source_prefix + "exact_shell_amplitudes"]
        )
        sidecar_hamiltonian = np.asarray(
            arrays[source_prefix + "fsa_hamiltonian_sector"]
        )
        sidecar_tower = np.asarray(arrays[source_prefix + "match_exact_indices"])
    except KeyError as error:
        raise RuntimeError(f"{failure}: missing NPZ source evidence") from error
    if (
        upstream.get("length") != length
        or per_length.get("source_hashes") != upstream.get("source_hashes")
    ):
        raise RuntimeError(f"{failure}: current upstream source hashes mismatch")
    energies = np.asarray(upstream["energies"])
    amplitudes = np.asarray(upstream["exact_shell_amplitudes"])
    hamiltonian = np.asarray(upstream["fsa_hamiltonian_sector"])
    tower = np.asarray(upstream["selector"]["tower"], dtype=np.int64)
    if (
        not np.array_equal(sidecar_energies, energies)
        or not np.array_equal(sidecar_amplitudes, amplitudes)
        or not np.array_equal(sidecar_hamiltonian, hamiltonian)
        or not np.array_equal(sidecar_tower, tower)
    ):
        raise RuntimeError(f"{failure}: NPZ evidence differs from current upstream")
    if (
        hashlib.sha256(amplitudes.tobytes()).hexdigest()
        != fsa.get("shell_amplitudes_sha256")
        or hashlib.sha256(hamiltonian.tobytes()).hexdigest()
        != fsa.get("fsa_hamiltonian_sha256")
        or hashlib.sha256(tower.astype(np.int64).tobytes()).hexdigest()
        != fsa.get("match_indices_sha256")
    ):
        raise RuntimeError(f"{failure}: source evidence hash mismatch")
    if (
        fsa.get("match_exact_indices") != tower.tolist()
        or fsa.get("match_fsa_indices") != list(range(plotted_count))
        or fsa.get("shell_dimensions") != list(amplitudes.shape)
    ):
        raise RuntimeError(f"{failure}: source indices or dimensions mismatch")
    fsa_energies, fsa_vectors = np.linalg.eigh(hamiltonian)
    projection = np.abs(amplitudes.T @ fsa_vectors) ** 2
    if (
        not np.array_equal(
            energies[tower],
            np.asarray(fsa.get("match_exact_energies")),
        )
        or not np.array_equal(
            fsa_energies,
            np.asarray(fsa.get("fsa_energies")),
        )
        or not np.array_equal(
            projection[tower, np.arange(plotted_count)],
            np.asarray(fsa.get("match_strengths")),
        )
    ):
        raise RuntimeError(f"{failure}: JSON source evidence mismatch")

    from turner2018_fig3 import (
        FSA_EIGENVALUE_GAP_TOLERANCE,
        select_fig3_shell_panel_states,
    )

    try:
        expected_states = select_fig3_shell_panel_states(
            length=length,
            energies=energies,
            exact_shell_amplitudes=amplitudes,
            fsa_hamiltonian_sector=hamiltonian,
            matched_tower={"tower": tower},
            zero_tolerance=1e-10,
            fsa_gap_tolerance=FSA_EIGENVALUE_GAP_TOLERANCE,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{failure}: source selection cannot be reproduced") from error
    expected_metadata = [
        state.to_metadata_dict()
        for state in expected_states
    ]
    if selected != expected_metadata or fsa.get("selected_states") != expected_metadata:
        raise RuntimeError(f"{failure}: semantic selection mismatch")
    for state in expected_states:
        prefix = f"panel_{state.panel}_L{length}_"
        if (
            not np.array_equal(arrays[prefix + "shell"], state.shell)
            or not np.array_equal(
                arrays[prefix + "exact_weights"], state.exact_weights
            )
            or not np.array_equal(
                arrays[prefix + "fsa_weights"], state.fsa_weights
            )
        ):
            raise RuntimeError(f"{failure}: shell weights mismatch source evidence")


def _accepted_figure(
    path: Path,
    label: str,
    *,
    length: int,
    expected_lengths: list[int] | None = None,
) -> dict[str, Any]:
    import numpy as np

    if not path.is_file():
        raise RuntimeError(f"{label} renderer did not produce its artifact: {path}")
    json_path = path.with_suffix(".json")
    npz_path = path.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise RuntimeError(f"{label} renderer did not produce complete sidecars")
    try:
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} JSON sidecar is invalid") from error
    if (
        metrics.get("source") != "independent-ed"
        or metrics.get("acceptance", {}).get("passed") is not True
        or metrics.get("acceptance", {}).get("generated_source") != "independent-ed"
    ):
        raise RuntimeError(f"{label} acceptance/provenance check failed")
    generation_id = metrics.get("generation_id")
    if not isinstance(generation_id, str):
        raise RuntimeError(f"{label} generation identity is missing")
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            npz_generation_id = arrays["generation_id"].item()
            array_names = tuple(arrays.files)
            fig3_arrays = (
                {
                    name: np.asarray(arrays[name]).copy()
                    for name in array_names
                    if name.startswith(
                        ("panel_a_", "panel_b_", "panel_c_", "selection_")
                    )
                }
                if label == "Fig. 3"
                else {}
            )
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"{label} NPZ generation identity is invalid") from error
    if npz_generation_id != generation_id:
        raise RuntimeError(f"{label} JSON/NPZ generation identity mismatch")
    assets = metrics.get("generation_assets", {})
    if (
        assets.get("png_sha256") != _sha256(path)
        or assets.get("npz_sha256") != _sha256(npz_path)
    ):
        raise RuntimeError(f"{label} generation asset hash check failed")
    if label == "Fig. 3":
        from turner2018_fig3 import _load_independent_length

        try:
            upstream = _load_independent_length(path.parents[1])
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "Fig. 3 current upstream validated snapshot is invalid"
            ) from error
        _validate_fig3_shell_selection(
            metrics,
            fig3_arrays,
            length=length,
            upstream=upstream,
        )
    if label == "Fig. 4":
        acceptance = metrics["acceptance"]
        if (
            acceptance.get("provenance_passed") is not True
            or acceptance.get("render_passed") is not True
        ):
            raise RuntimeError(f"{label} provenance/render acceptance is invalid")
        scope = metrics.get("available_independent_lengths")
        expected_scope = expected_lengths or [length]
        if (
            scope != expected_scope
            or metrics.get("layout", {}).get("lengths") != expected_scope
            or set(metrics.get("lengths", {}))
            != {str(item) for item in expected_scope}
            or any(
                entry.get("length") not in set(expected_scope)
                for entry in metrics.get("series", {}).values()
            )
        ):
            raise RuntimeError(f"{label} scoped lengths/series are invalid")
        if len(expected_scope) == 1:
            prefix = f"L{expected_scope[0]}_"
            official_prefix = f"official_L{expected_scope[0]}_"
            if any(
                name != "generation_id"
                and not name.startswith(prefix)
                and not name.startswith(official_prefix)
                for name in array_names
            ):
                raise RuntimeError(f"{label} NPZ scoped lengths are invalid")
        production_lengths = [
            item for item in expected_scope if item in {28, 30, 32}
        ]
        if production_lengths and (
            acceptance.get("statistics_required") is not True
            or acceptance.get("statistics_passed") is not True
            or any(
                metrics.get("lengths", {})
                .get(str(item), {})
                .get("provenance_acceptance", {})
                .get("passed")
                is not True
                or metrics.get("lengths", {})
                .get(str(item), {})
                .get("histogram_acceptance", {})
                .get("passed")
                is not True
                for item in production_lengths
            )
        ):
            raise RuntimeError(
                f"{label} statistics acceptance is required for production "
                f"lengths={production_lengths}"
            )
    return {
        "path": str(path.relative_to(path.parents[1])),
        "sha256": _sha256(path),
        "metrics_path": str(json_path.relative_to(json_path.parents[1])),
        "metrics_sha256": _sha256(json_path),
        "arrays_path": str(npz_path.relative_to(npz_path.parents[1])),
        "arrays_sha256": _sha256(npz_path),
        "generation_id": generation_id,
    }


def _accepted_fig4_set(path: Path, *, length: int) -> dict[str, Any]:
    try:
        overview_metrics = json.loads(path.with_suffix(".json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Fig. 4 overview JSON sidecar is invalid") from error
    available = overview_metrics.get("available_independent_lengths")
    if (
        not isinstance(available, list)
        or not available
        or any(not isinstance(item, int) for item in available)
        or length not in available
    ):
        raise RuntimeError("Fig. 4 overview has invalid available lengths")
    overview = _accepted_figure(
        path,
        "Fig. 4",
        length=length,
        expected_lengths=available,
    )
    sizes: dict[str, dict[str, Any]] = {}
    for size in available:
        size_path = path.parent / f"fig4_independent_L{size}.png"
        sizes[str(size)] = _accepted_figure(
            size_path,
            "Fig. 4",
            length=length,
            expected_lengths=[size],
        )
    generation_ids = {
        overview["generation_id"],
        *(record["generation_id"] for record in sizes.values()),
    }
    if len(generation_ids) != 1:
        raise RuntimeError("Fig. 4 output set has mixed generation identities")
    return {
        "generation_id": overview["generation_id"],
        "overview": overview,
        "sizes": sizes,
    }


def _resolve_figure_reference(output_dir: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str):
        raise RuntimeError("figure reference path must be a string")
    resolved = (output_dir / relative_path).resolve()
    figures_root = (output_dir / "figures").resolve()
    if figures_root not in resolved.parents:
        raise RuntimeError(f"figure reference escapes figures directory: {relative_path}")
    return resolved


def _validate_figures_stage_references(
    output_dir: Path,
    stage_payload: dict[str, Any],
) -> None:
    manifest_path = _resolve_figure_reference(
        output_dir, stage_payload["artifact"]["path"]
    )
    try:
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("combined figures manifest JSON is invalid") from error
    length = summary.get("length")
    if (
        not isinstance(length, int)
        or summary.get("source") != "independent-ed"
        or summary.get("status") != "passed"
        or summary.get("passed") is not True
        or summary.get("acceptance", {}).get("passed") is not True
    ):
        raise RuntimeError("combined figures manifest acceptance is invalid")
    for key, label in (("fig3", "Fig. 3"), ("fig4", "Fig. 4")):
        recorded = summary.get(key)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"combined figures manifest is missing {key}")
        if key == "fig3":
            path = _resolve_figure_reference(output_dir, recorded.get("path"))
            actual = _accepted_figure(path, label, length=length)
        else:
            overview = recorded.get("overview")
            if not isinstance(overview, dict):
                raise RuntimeError("combined figures manifest is missing Fig. 4 overview")
            path = _resolve_figure_reference(output_dir, overview.get("path"))
            actual = _accepted_fig4_set(path, length=length)
        if actual != recorded:
            raise RuntimeError(f"{label} figure reference hash/provenance mismatch")


def run_figures(length: int, output_dir: Path) -> None:
    require_stage(output_dir, "validate")
    fig3_artifact = Path(FIG3_RENDERER_ADAPTER(output_dir, length))
    fig3 = _accepted_figure(fig3_artifact, "Fig. 3", length=length)
    fig4_artifact = Path(FIG4_RENDERER_ADAPTER(output_dir, length))
    fig4 = _accepted_fig4_set(fig4_artifact, length=length)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "length": length,
        "source": "independent-ed",
        "fig3": fig3,
        "fig4": fig4,
        "acceptance": {"passed": True},
    }
    _write_hashed_json_stage(
        output_dir,
        "figures",
        "figures/manifest.json",
        summary,
        inputs=_stage_inputs(output_dir, "figures"),
        plan_sha256=_plan_config_sha256(output_dir),
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
    parser.add_argument("--rebuild", action="store_true")
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
            "--rebuild",
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

    validation_cache: dict[str, dict[str, Any]] = {}
    plan_manifest = args.output_dir / "stages" / "plan.json"
    try:
        if plan_manifest.is_file():
            try:
                _load_plan(args.output_dir, args.length, validation_cache)
            except StalePlanError:
                if args.stage not in {"all", "plan"} and not args.rebuild:
                    raise
                write_plan(args.length, args.output_dir, invoked)
                validation_cache.clear()
                print("completed stage=plan", flush=True)
            else:
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
            if stage_manifest.is_file() and not args.rebuild:
                try:
                    _validate_current_stage(
                        args.output_dir,
                        stage,
                        validation_cache,
                    )
                except StaleStageError:
                    pass
                else:
                    print(f"skipped stage={stage}", flush=True)
                    continue
            for dependency in STAGE_DEPENDENCIES[stage]:
                require_stage(args.output_dir, dependency, validation_cache)
            if stage == "basis":
                run_basis(args.length, args.output_dir)
            elif stage == "hamiltonian":
                run_hamiltonian(args.length, args.output_dir, validation_cache)
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
                    validation_cache,
                )
            elif stage == "observables":
                run_observables(
                    args.length,
                    args.output_dir,
                    args.chunk_columns,
                    validation_cache,
                )
            elif stage == "validate":
                run_validate(args.length, args.output_dir, validation_cache)
            elif stage == "figures":
                run_figures(args.length, args.output_dir)
            print(f"completed stage={stage}", flush=True)
            validation_cache.clear()
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
