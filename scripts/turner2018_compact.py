#!/usr/bin/env python3
"""Export and validate vector-free Turner independent-ED plotting packages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import h5py
import numpy as np

from turner2018_ed_artifacts import SCHEMA_VERSION

COMPACT_SCHEMA_VERSION = "turner2018-independent-ed-compact-v1"
COMPACT_MANIFEST = "compact.json"
EXPECTED_MODEL = "H=sum_j P_(j-1) X_j P_(j+1), PBC"
STAGES = ("plan", "basis", "hamiltonian", "diagonalize", "observables", "validate")
DEPENDENCIES = {
    "plan": (),
    "basis": (),
    "hamiltonian": ("basis",),
    "diagonalize": ("hamiltonian",),
    "observables": ("basis", "hamiltonian", "diagonalize"),
    "validate": ("basis", "hamiltonian", "diagonalize", "observables"),
}
REQUIRED_OBSERVABLES = {
    "exact_shell_amplitudes",
    "fsa_beta_full_chain",
    "fsa_hamiltonian_sector",
    "fsa_shell_vectors_sector",
    "overlap_z2",
    "participation_ratio",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _validate_manifest_chain(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = {
        stage: _read_json(directory / "stages" / f"{stage}.json")
        for stage in STAGES
    }
    manifest_hashes = {
        stage: _sha256(directory / "stages" / f"{stage}.json") for stage in STAGES
    }
    plan = _read_json(directory / "manifest.json")
    plan_sha = plan.get("scientific_config_sha256")
    if not isinstance(plan_sha, str) or len(plan_sha) != 64:
        raise RuntimeError("scientific plan hash is invalid")
    for stage, payload in manifests.items():
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("stage") != stage
            or payload.get("status") != "complete"
            or payload.get("plan_sha256") != plan_sha
        ):
            raise RuntimeError(f"stage manifest is invalid: {stage}")
        expected_inputs = {
            dependency: manifest_hashes[dependency]
            for dependency in DEPENDENCIES[stage]
        }
        if payload.get("inputs") != expected_inputs:
            raise RuntimeError(f"stage dependency hashes are stale: {stage}")
    if manifests["plan"]["artifact"].get("sha256") != _sha256(
        directory / "manifest.json"
    ):
        raise RuntimeError("plan artifact hash mismatch")
    return plan, manifests


def _compact_manifest_payload(
    directory: Path,
    *,
    source_eigensystem_sha256: str,
    source_eigensystem_bytes: int,
    vector_shape: list[int],
) -> dict[str, Any]:
    files = (
        "manifest.json",
        "basis.npz",
        "hamiltonian.csr.npz",
        "eigenvalues.h5",
        "observables.h5",
        "validation/metrics.json",
        *(f"stages/{stage}.json" for stage in STAGES),
    )
    return {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "mode": "validated-vector-free-plotting-package",
        "compact_artifact": {
            "path": "eigenvalues.h5",
            "sha256": _sha256(directory / "eigenvalues.h5"),
        },
        "source_eigensystem": {
            "path": "eigensystem.h5",
            "sha256": source_eigensystem_sha256,
            "byte_size": source_eigensystem_bytes,
            "vector_shape": vector_shape,
            "vectors_local": False,
        },
        "file_sha256": {name: _sha256(directory / name) for name in files},
    }


def bind_compact_package(directory: str | Path) -> dict[str, Any]:
    """Write a hash manifest for an already exported trusted compact package."""
    directory = Path(directory)
    loaded = _validate_compact_core(directory, require_compact_manifest=False)
    payload = _compact_manifest_payload(
        directory,
        source_eigensystem_sha256=loaded["source_eigensystem_sha256"],
        source_eigensystem_bytes=loaded["source_eigensystem_bytes"],
        vector_shape=loaded["source_vector_shape"],
    )
    _write_json_atomic(directory / COMPACT_MANIFEST, payload)
    return payload


def export_compact_package(
    source_directory: str | Path, destination_directory: str | Path
) -> dict[str, Any]:
    """Export energies and validated observables without copying eigenvectors."""
    source = Path(source_directory)
    destination = Path(destination_directory)
    if destination.exists():
        raise FileExistsError(destination)
    if (source / COMPACT_MANIFEST).is_file() and not (
        source / "eigensystem.h5"
    ).is_file():
        load_compact_package(source)
        manifest = _read_json(source / COMPACT_MANIFEST)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.compact-", dir=destination.parent
            )
        )
        try:
            names = [*manifest["file_sha256"], COMPACT_MANIFEST]
            for name in names:
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
            load_compact_package(temporary)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return _read_json(destination / COMPACT_MANIFEST)

    from turner2018_l32_server import require_stage

    cache: dict[str, dict[str, Any]] = {}
    require_stage(source, "validate", cache)
    diagonalize = cache["diagonalize"]
    source_eigensystem = source / "eigensystem.h5"
    source_sha = diagonalize["artifact"]["sha256"]
    source_bytes = source_eigensystem.stat().st_size
    vector_shape = list(diagonalize["vector_shape"])
    with h5py.File(source_eigensystem, "r") as handle:
        energies = handle["eigensystem/energies"][()]

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.compact-", dir=destination.parent)
    )
    try:
        for name in (
            "manifest.json",
            "basis.npz",
            "hamiltonian.csr.npz",
            "observables.h5",
        ):
            shutil.copy2(source / name, temporary / name)
        shutil.copytree(source / "stages", temporary / "stages")
        shutil.copytree(source / "validation", temporary / "validation")
        with h5py.File(temporary / "eigenvalues.h5", "w") as handle:
            handle.attrs["source_artifact_sha256"] = source_sha
            handle.attrs["source_artifact_bytes"] = source_bytes
            handle.attrs["source_vector_shape"] = vector_shape
            group = handle.create_group("eigensystem")
            dataset = group.create_dataset("energies", data=energies)
            dataset.attrs["source_dataset"] = "eigensystem/energies"
        bind_compact_package(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _read_json(destination / COMPACT_MANIFEST)


def _validate_compact_core(
    directory: Path, *, require_compact_manifest: bool
) -> dict[str, Any]:
    plan, stages = _validate_manifest_chain(directory)
    validation = _read_json(directory / "validation" / "metrics.json")
    model = plan.get("model", {})
    if (
        model.get("hamiltonian") != EXPECTED_MODEL
        or model.get("boundary") != "periodic"
        or model.get("momentum") != 0
        or model.get("inversion") != "even"
    ):
        raise RuntimeError("compact scientific model is invalid")
    length = model.get("length")
    sector_dimension = plan.get("basis", {}).get("sector_dimension")
    full_dimension = plan.get("basis", {}).get("full_constrained_dimension")
    checks = validation.get("metrics", {})
    if sector_dimension is None:
        sector_dimension = checks.get("basis_sector_dimension", {}).get("value")
    if full_dimension is None:
        full_dimension = checks.get("basis_full_dimension", {}).get("value")
    if (
        not isinstance(length, int)
        or not isinstance(sector_dimension, int)
        or not isinstance(full_dimension, int)
    ):
        raise RuntimeError("compact scientific dimensions are invalid")

    for stage, name in (
        ("basis", "basis.npz"),
        ("hamiltonian", "hamiltonian.csr.npz"),
        ("observables", "observables.h5"),
        ("validate", "validation/metrics.json"),
    ):
        expected = stages[stage]["artifact"]["sha256"]
        if _sha256(directory / name) != expected:
            label = "validation" if stage == "validate" else stage
            raise RuntimeError(f"{label} artifact sha256 mismatch")
    if (
        validation.get("status") != "passed"
        or validation.get("passed") is not True
        or validation.get("length") != length
        or any(
            check.get("passed") is not True
            for check in validation.get("metrics", {}).values()
        )
    ):
        raise RuntimeError("compact package validation did not pass")
    validated = validation.get("validated_stage_sha256", {})
    for stage in ("basis", "hamiltonian", "diagonalize", "observables"):
        if validated.get(stage) != stages[stage]["artifact"]["sha256"]:
            raise RuntimeError(f"validation source hash mismatch: {stage}")

    eigenvalues_path = directory / "eigenvalues.h5"
    with h5py.File(eigenvalues_path, "r") as handle:
        if set(handle.keys()) != {"eigensystem"}:
            raise RuntimeError("compact eigenvalue groups are invalid")
        group = handle["eigensystem"]
        if set(group.keys()) != {"energies"}:
            raise RuntimeError("compact eigenvalue dataset set is invalid")
        dataset = group["energies"]
        energies = dataset[()]
        source_sha = handle.attrs.get("source_artifact_sha256")
        source_bytes = int(handle.attrs.get("source_artifact_bytes", -1))
        source_vector_shape = [
            int(value) for value in handle.attrs.get("source_vector_shape", [])
        ]
        if dataset.attrs.get("source_dataset") != "eigensystem/energies":
            raise RuntimeError("compact energy source dataset is invalid")
    diagonalize = stages["diagonalize"]
    if (
        source_sha != diagonalize["artifact"]["sha256"]
        or source_bytes < energies.nbytes
        or source_vector_shape != [sector_dimension, sector_dimension]
        or diagonalize.get("vector_shape") != source_vector_shape
        or diagonalize.get("has_vectors") is not True
    ):
        raise RuntimeError("compact source eigensystem reference is invalid")
    if (
        energies.shape != (sector_dimension,)
        or energies.dtype != np.float64
        or not np.all(np.isfinite(energies))
        or np.any(np.diff(energies) < 0)
    ):
        raise RuntimeError("compact energies must have exact length, be finite and sorted")

    expected_shapes = {
        "overlap_z2": (sector_dimension,),
        "participation_ratio": (sector_dimension,),
        "exact_shell_amplitudes": (length // 2 + 1, sector_dimension),
        "fsa_shell_vectors_sector": (length // 2 + 1, sector_dimension),
        "fsa_hamiltonian_sector": (length // 2 + 1, length // 2 + 1),
        "fsa_beta_full_chain": (length,),
    }
    observables: dict[str, np.ndarray] = {}
    with h5py.File(directory / "observables.h5", "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("compact observables schema is invalid")
        group = handle["observables"]
        if set(group.keys()) != REQUIRED_OBSERVABLES:
            raise RuntimeError("compact observables dataset set is invalid")
        metadata = json.loads(group.attrs["validation_metadata"])
        if (
            metadata.get("finite_columns_checked") != sector_dimension
            or metadata.get("residual_columns_checked") != sector_dimension
        ):
            raise RuntimeError("compact observables validation metadata is invalid")
        for name, shape in expected_shapes.items():
            values = group[name][()]
            if (
                values.shape != shape
                or values.dtype != np.float64
                or not np.all(np.isfinite(values))
            ):
                raise RuntimeError(f"compact observable is invalid: {name}")
            observables[name] = values

    compact_file_hashes: dict[str, str] = {}
    if require_compact_manifest:
        compact = _read_json(directory / COMPACT_MANIFEST)
        if compact.get("schema_version") != COMPACT_SCHEMA_VERSION:
            raise RuntimeError("compact manifest schema is invalid")
        expected_compact_hash = compact.get("compact_artifact", {}).get("sha256")
        if expected_compact_hash != _sha256(eigenvalues_path):
            raise RuntimeError("compact artifact sha256 mismatch")
        compact_file_hashes = compact.get("file_sha256", {})
        if not isinstance(compact_file_hashes, dict):
            raise RuntimeError("compact manifest file hashes are invalid")
        for name, expected in compact_file_hashes.items():
            if _sha256(directory / name) != expected:
                label = "validation" if name == "validation/metrics.json" else name
                raise RuntimeError(f"compact manifest hash mismatch: {label}")
        source = compact.get("source_eigensystem", {})
        if (
            source.get("sha256") != source_sha
            or source.get("byte_size") != source_bytes
            or source.get("vector_shape") != source_vector_shape
            or source.get("vectors_local") is not False
        ):
            raise RuntimeError("compact source eigensystem manifest is invalid")

    return {
        "source": "independent-ed",
        "directory": directory,
        "length": length,
        "full_dimension": full_dimension,
        "sector_dimension": sector_dimension,
        "energies": energies,
        **observables,
        "validation": validation,
        "source_eigensystem_sha256": source_sha,
        "source_eigensystem_bytes": source_bytes,
        "source_vector_shape": source_vector_shape,
        "source_hashes": {
            **compact_file_hashes,
            "eigenvalues.h5": _sha256(eigenvalues_path),
            "source_eigensystem.h5": source_sha,
            "observables.h5": _sha256(directory / "observables.h5"),
            "validation_metrics": _sha256(
                directory / "validation" / "metrics.json"
            ),
        },
        "eigenvector_metadata": {
            "shape": source_vector_shape,
            "dtype": "float64",
            "chunks": [sector_dimension, 1],
            "access": "remote-source-reference-only",
            "local": False,
        },
        "energy_dataset_metadata": {
            "shape": [sector_dimension],
            "dtype": "float64",
            "access": "full-energy-vector-only",
        },
    }


def load_compact_package(directory: str | Path) -> dict[str, Any]:
    """Load a fully hash-bound compact package for plotting."""
    return _validate_compact_core(Path(directory), require_compact_manifest=True)
