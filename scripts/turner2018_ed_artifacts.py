"""Immutable artifact writers and stage validation for independent ED."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import h5py
import numpy as np
import scipy
import scipy.sparse as sp


SCHEMA_VERSION = "turner2018-independent-ed-v1"
STAGE_ARTIFACT_NAMES = {
    "plan": "manifest.json",
    "basis": "basis.npz",
    "hamiltonian": "hamiltonian.csr.npz",
    "eigensystem": "eigensystem.h5",
    "diagonalize": "eigensystem.h5",
    "observables": "observables.h5",
    "validate": "validation/metrics.json",
    "figures": "figures/manifest.json",
}
STAGE_INPUT_NAMES = {
    "plan": set(),
    "basis": set(),
    "hamiltonian": {"basis"},
    "eigensystem": {"hamiltonian"},
    "diagonalize": {"hamiltonian"},
    "observables": {"basis", "hamiltonian", "diagonalize"},
    "validate": {"basis", "hamiltonian", "diagonalize", "observables"},
    "figures": {"validate"},
}


@dataclass(frozen=True)
class ArtifactMeta:
    path: str
    sha256: str
    shape: list[int]
    dtype: str
    conventions: list[str]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_parent_directory(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_if_exists_durable(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _fsync_parent_directory(path)
    return True


def _cleanup_partial(path: Path) -> None:
    _unlink_if_exists_durable(path)


def _cleanup_temp(path: Path) -> None:
    _unlink_if_exists_durable(path)


def _remove_path_durable(path: Path) -> None:
    _unlink_if_exists_durable(path)


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision or None


def _provenance() -> dict[str, Any]:
    slurm = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_")
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
        "git_revision": _git_revision(),
        "slurm_environment": slurm,
    }


def _write_stage_manifest(
    output_dir: Path,
    stage: str,
    artifact: ArtifactMeta,
    extra: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    declared_inputs = (
        {name: "0" * 64 for name in STAGE_INPUT_NAMES[stage]}
        if inputs is None
        else dict(inputs)
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation_id": str(uuid.uuid4()),
        "stage": stage,
        "status": "complete",
        "inputs": declared_inputs,
        "plan_sha256": plan_sha256 or "0" * 64,
        "artifact": {
            "path": artifact.path,
            "sha256": artifact.sha256,
            "shape": artifact.shape,
            "dtype": artifact.dtype,
            "conventions": artifact.conventions,
        },
        "provenance": _provenance(),
    }
    if extra:
        payload.update(extra)
    target = output_dir / "stages" / f"{stage}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        with partial.open("wb") as handle:
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
        _fsync_parent_directory(target)
    except Exception:
        _cleanup_partial(partial)
        raise
    return payload


def _atomic_write_h5(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with h5py.File(partial, "w") as handle:
            writer(handle)
            handle.flush()
        _fsync_file(partial)
        os.replace(partial, path)
        _fsync_parent_directory(path)
    except Exception:
        _cleanup_partial(partial)
        raise


def _atomic_write_npz(path: Path, writer: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_parent_directory(path)
    except Exception:
        _cleanup_partial(partial)
        raise


def _backup_previous_stage_state(
    output_dir: Path,
    stage: str,
    artifact_path: Path,
) -> bool:
    manifest_path = output_dir / "stages" / f"{stage}.json"
    if not manifest_path.is_file() and not artifact_path.is_file():
        return False
    artifact_backup = artifact_path.with_name(artifact_path.name + ".backup")
    manifest_backup = manifest_path.with_name(manifest_path.name + ".backup")
    _cleanup_temp(artifact_backup)
    _cleanup_temp(manifest_backup)
    try:
        if artifact_path.is_file():
            os.link(artifact_path, artifact_backup)
            _fsync_parent_directory(artifact_backup)
        if manifest_path.is_file():
            os.link(manifest_path, manifest_backup)
            _fsync_parent_directory(manifest_backup)
    except Exception:
        _cleanup_temp(artifact_backup)
        _cleanup_temp(manifest_backup)
        raise
    return True


def _restore_stage_state(
    *,
    artifact_path: Path,
    manifest_path: Path,
) -> None:
    artifact_backup = artifact_path.with_name(artifact_path.name + ".backup")
    manifest_backup = manifest_path.with_name(manifest_path.name + ".backup")
    try:
        for backup, target in (
            (artifact_backup, artifact_path),
            (manifest_backup, manifest_path),
        ):
            if backup.is_file():
                _unlink_if_exists_durable(target)
                os.link(backup, target)
                _fsync_parent_directory(target)
            else:
                _unlink_if_exists_durable(target)
    finally:
        _cleanup_temp(artifact_backup)
        _cleanup_temp(manifest_backup)
        _cleanup_temp(artifact_path.with_name(artifact_path.name + ".partial"))
        _cleanup_temp(manifest_path.with_name(manifest_path.name + ".partial"))


def _publish_stage_transactional(
    *,
    output_dir: Path,
    stage: str,
    artifact_path: Path,
    write_artifact: Any,
    build_artifact: Any,
    extra: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "stages" / f"{stage}.json"
    try:
        had_previous = _backup_previous_stage_state(
            output_dir, stage, artifact_path
        )
    except Exception:
        _cleanup_temp(artifact_path.with_name(artifact_path.name + ".backup"))
        _cleanup_temp(manifest_path.with_name(manifest_path.name + ".backup"))
        raise
    try:
        write_artifact()
        artifact = build_artifact()
        return _write_stage_manifest(
            output_dir,
            stage,
            artifact,
            extra,
            inputs=inputs,
            plan_sha256=plan_sha256,
        )
    except Exception:
        if had_previous:
            _restore_stage_state(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
            )
        else:
            _remove_path_durable(artifact_path)
            _remove_path_durable(manifest_path)
        raise
    finally:
        _cleanup_temp(artifact_path.with_name(artifact_path.name + ".backup"))
        _cleanup_temp(manifest_path.with_name(manifest_path.name + ".backup"))
        _cleanup_temp(artifact_path.with_name(artifact_path.name + ".partial"))
        _cleanup_temp(manifest_path.with_name(manifest_path.name + ".partial"))


def write_basis_artifact(
    output_dir: Path,
    *,
    basis: np.ndarray,
    representatives: np.ndarray | None = None,
    orbit_sizes: np.ndarray | None = None,
    length: int | None = None,
    inputs: dict[str, str] | None = None,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "basis.npz"
    basis_array = np.asarray(basis, dtype=np.uint64)

    representatives_array = (
        None if representatives is None else np.asarray(representatives, dtype=np.uint64)
    )
    orbit_sizes_array = (
        None if orbit_sizes is None else np.asarray(orbit_sizes, dtype=np.uint8)
    )
    if (representatives_array is None) != (orbit_sizes_array is None):
        raise ValueError("representatives and orbit_sizes must be provided together")
    if representatives_array is not None and representatives_array.shape != orbit_sizes_array.shape:
        raise ValueError("representatives and orbit_sizes must have the same shape")

    def write_artifact() -> None:
        arrays: dict[str, np.ndarray] = {"basis": basis_array}
        if representatives_array is not None and orbit_sizes_array is not None:
            arrays["representatives"] = representatives_array
            arrays["orbit_sizes"] = orbit_sizes_array
        if length is not None:
            arrays["length"] = np.asarray(int(length), dtype=np.int64)
        _atomic_write_npz(target, lambda handle: np.savez(handle, **arrays))

    def build_artifact() -> ArtifactMeta:
        return ArtifactMeta(
            path=target.name,
            sha256=_sha256(target),
            shape=[int(basis_array.shape[0])],
            dtype=str(basis_array.dtype),
            conventions=[
                "sorted-constrained-states",
                *(
                    ["direct-dihedral-orbit-metadata"]
                    if representatives_array is not None
                    else []
                ),
            ],
        )

    return _publish_stage_transactional(
        output_dir=output_dir,
        stage="basis",
        artifact_path=target,
        write_artifact=write_artifact,
        build_artifact=build_artifact,
        inputs=inputs,
        plan_sha256=plan_sha256,
    )


def write_hamiltonian_artifact(
    output_dir: Path,
    *,
    hamiltonian: sp.csr_matrix,
    inputs: dict[str, str] | None = None,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "hamiltonian.csr.npz"

    def write_artifact() -> None:
        _atomic_write_npz(target, lambda handle: sp.save_npz(handle, hamiltonian, compressed=False))

    def build_artifact() -> ArtifactMeta:
        return ArtifactMeta(
            path=target.name,
            sha256=_sha256(target),
            shape=[int(hamiltonian.shape[0]), int(hamiltonian.shape[1])],
            dtype=str(hamiltonian.dtype),
            conventions=["csr-real-symmetric"],
        )

    return _publish_stage_transactional(
        output_dir=output_dir,
        stage="hamiltonian",
        artifact_path=target,
        write_artifact=write_artifact,
        build_artifact=build_artifact,
        extra={"nnz": int(hamiltonian.nnz)},
        inputs=inputs,
        plan_sha256=plan_sha256,
    )


def write_eigensystem(
    output_dir: Path,
    *,
    energies: np.ndarray,
    vectors: np.ndarray | None,
    stage: str = "eigensystem",
    inputs: dict[str, str] | None = None,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "eigensystem.h5"
    energies_arr = np.asarray(energies, dtype=np.float64)
    vectors_arr = None if vectors is None else np.asarray(vectors, dtype=np.float64, order="F")

    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        eig = handle.create_group("eigensystem")
        eig.create_dataset("energies", data=energies_arr)
        if vectors_arr is not None:
            eig.create_dataset(
                "vectors",
                data=vectors_arr,
                chunks=(vectors_arr.shape[0], 1),
            )

    def write_artifact() -> None:
        _atomic_write_h5(target, writer)

    def build_artifact() -> ArtifactMeta:
        return ArtifactMeta(
            path=target.name,
            sha256=_sha256(target),
            shape=[int(energies_arr.shape[0])],
            dtype=str(energies_arr.dtype),
            conventions=["eigenvectors-are-columns", "float64", "hdf5"],
        )

    extra: dict[str, Any] = {"has_vectors": vectors_arr is not None}
    if vectors_arr is not None:
        extra["vector_shape"] = [int(vectors_arr.shape[0]), int(vectors_arr.shape[1])]
    return _publish_stage_transactional(
        output_dir=output_dir,
        stage=stage,
        artifact_path=target,
        write_artifact=write_artifact,
        build_artifact=build_artifact,
        extra=extra,
        inputs=inputs,
        plan_sha256=plan_sha256,
    )


def write_observables(
    output_dir: Path,
    observables: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    plan_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "observables.h5"
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in observables.items()}

    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        group = handle.create_group("observables")
        group.attrs["validation_metadata"] = json.dumps(metadata or {}, sort_keys=True)
        for name, array in arrays.items():
            group.create_dataset(name, data=array)

    def write_artifact() -> None:
        _atomic_write_h5(target, writer)

    def build_artifact() -> ArtifactMeta:
        return ArtifactMeta(
            path=target.name,
            sha256=_sha256(target),
            shape=[sum(int(array.size) for array in arrays.values())],
            dtype="float64",
            conventions=["observables-only", "separate-from-eigensystem"],
        )

    return _publish_stage_transactional(
        output_dir=output_dir,
        stage="observables",
        artifact_path=target,
        write_artifact=write_artifact,
        build_artifact=build_artifact,
        extra={"datasets": sorted(arrays.keys())},
        inputs=inputs,
        plan_sha256=plan_sha256,
    )


def _resolve_stage_artifact_path(output_dir: Path, stage: str, artifact_path_raw: str) -> Path:
    artifact_relative = Path(artifact_path_raw)
    if artifact_relative.is_absolute():
        raise RuntimeError(f"absolute artifact paths are forbidden: {artifact_path_raw}")
    if ".." in artifact_relative.parts:
        raise RuntimeError(f"path traversal is forbidden: {artifact_path_raw}")
    expected = STAGE_ARTIFACT_NAMES.get(stage)
    if expected is not None and artifact_relative.as_posix() != expected:
        raise RuntimeError(
            f"stage/artifact-name mismatch: stage={stage} expected={expected} actual={artifact_relative.as_posix()}"
        )
    if expected is None and len(artifact_relative.parts) != 1:
        raise RuntimeError(f"nested artifact paths are forbidden: {artifact_path_raw}")

    output_resolved = output_dir.resolve()
    artifact_path = (output_dir / artifact_relative).resolve()
    try:
        artifact_path.relative_to(output_resolved)
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes output directory: {artifact_path_raw}") from exc
    return artifact_path


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_internal_artifact(
    stage: str,
    artifact_path: Path,
    artifact: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    expected_conventions = {
        "basis": ["sorted-constrained-states", "direct-dihedral-orbit-metadata"],
        "hamiltonian": ["csr-real-symmetric"],
        "eigensystem": ["eigenvectors-are-columns", "float64", "hdf5"],
        "diagonalize": ["eigenvectors-are-columns", "float64", "hdf5"],
        "observables": ["observables-only", "separate-from-eigensystem"],
        "plan": ["atomic-json", "sha256-validated"],
        "validate": ["atomic-json", "sha256-validated"],
        "figures": ["atomic-json", "sha256-validated"],
    }
    if artifact.get("conventions") != expected_conventions.get(stage):
        raise RuntimeError(f"invalid artifact conventions for stage={stage}")

    if stage == "basis":
        with np.load(artifact_path, allow_pickle=False) as data:
            required = {"basis", "representatives", "orbit_sizes", "length"}
            if set(data.files) != required:
                raise RuntimeError("basis artifact has invalid NPZ structure")
            basis = data["basis"]
            representatives = data["representatives"]
            orbit_sizes = data["orbit_sizes"]
            if basis.dtype != np.uint64 or representatives.dtype != np.uint64:
                raise RuntimeError("basis artifact arrays must be uint64")
            if orbit_sizes.dtype != np.uint8 or representatives.shape != orbit_sizes.shape:
                raise RuntimeError("basis orbit metadata is invalid")
            if (
                data["length"].shape != ()
                or not np.all(basis[1:] > basis[:-1])
                or not np.all(representatives[1:] > representatives[:-1])
            ):
                raise RuntimeError("basis artifact ordering or length metadata is invalid")
            actual_shape = [int(basis.size)]
            actual_dtype = "uint64"
    elif stage == "hamiltonian":
        matrix = sp.load_npz(artifact_path)
        if not sp.isspmatrix_csr(matrix) or matrix.shape[0] != matrix.shape[1]:
            raise RuntimeError("Hamiltonian artifact must be square CSR")
        if matrix.dtype != np.float64 or not np.all(np.isfinite(matrix.data)):
            raise RuntimeError("Hamiltonian artifact must be finite float64")
        asymmetry = matrix - matrix.T
        if asymmetry.nnz and float(np.max(np.abs(asymmetry.data))) > 1e-13:
            raise RuntimeError("Hamiltonian artifact must be symmetric")
        actual_shape = [int(matrix.shape[0]), int(matrix.shape[1])]
        actual_dtype = "float64"
    elif stage in {"eigensystem", "diagonalize"}:
        with h5py.File(artifact_path, "r") as handle:
            if handle.attrs.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("eigensystem HDF5 schema version mismatch")
            if "eigensystem/energies" not in handle:
                raise RuntimeError("eigensystem artifact is missing energies")
            energies = handle["eigensystem/energies"]
            dimension = int(energies.shape[0]) if energies.ndim == 1 else -1
            if energies.dtype != np.float64 or dimension < 1:
                raise RuntimeError("eigensystem energies metadata is invalid")
            energy_values = energies[()]
            if (
                not np.all(np.isfinite(energy_values))
                or np.any(np.diff(energy_values) < 0)
            ):
                raise RuntimeError("eigensystem energies must be finite and sorted")
            if payload.get("has_vectors"):
                if "eigensystem/vectors" not in handle:
                    raise RuntimeError("eigensystem artifact is missing vectors")
                vectors = handle["eigensystem/vectors"]
                if (
                    vectors.shape != (dimension, dimension)
                    or vectors.dtype != np.float64
                    or vectors.chunks != (dimension, 1)
                ):
                    raise RuntimeError("eigensystem vectors metadata is invalid")
            actual_shape = [dimension]
            actual_dtype = "float64"
    elif stage == "observables":
        with h5py.File(artifact_path, "r") as handle:
            if handle.attrs.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("observables HDF5 schema version mismatch")
            if "observables" not in handle:
                raise RuntimeError("observables group is missing")
            group = handle["observables"]
            datasets = sorted(group.keys())
            if datasets != payload.get("datasets"):
                raise RuntimeError("observables dataset manifest mismatch")
            total = 0
            for name in datasets:
                dataset = group[name]
                if dataset.dtype != np.float64:
                    raise RuntimeError("observables datasets must be float64")
                if dataset.ndim == 0:
                    finite = bool(np.isfinite(dataset[()]))
                else:
                    finite = all(
                        bool(np.all(np.isfinite(dataset[start : start + 256])))
                        for start in range(0, dataset.shape[0], 256)
                    )
                if not finite:
                    raise RuntimeError("observables datasets must be finite")
                total += int(dataset.size)
            actual_shape = [total]
            actual_dtype = "float64"
    elif stage in {"plan", "validate", "figures"}:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise RuntimeError(f"{stage} artifact must be a JSON object")
        if stage == "validate" and (
            document.get("status") != "passed" or document.get("passed") is not True
        ):
            raise RuntimeError("validate artifact does not record a passing gate")
        actual_shape = []
        actual_dtype = "json"
    else:
        raise RuntimeError(f"unsupported stage validation: {stage}")

    if artifact.get("shape") != actual_shape:
        raise RuntimeError(f"artifact shape metadata mismatch for stage={stage}")
    if artifact.get("dtype") != actual_dtype:
        raise RuntimeError(f"artifact dtype metadata mismatch for stage={stage}")


def validate_stage(output_dir: Path, stage: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    path = output_dir / "stages" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"missing stage manifest: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"stage manifest schema version mismatch: {path}")
    if payload.get("stage") != stage or payload.get("status") != "complete":
        raise RuntimeError(f"stage manifest is incomplete: {path}")
    try:
        uuid.UUID(payload.get("generation_id", ""))
    except (ValueError, TypeError, AttributeError) as error:
        raise RuntimeError(f"stage manifest generation id is invalid: {path}") from error
    if not _is_sha256(payload.get("plan_sha256")):
        raise RuntimeError(f"stage manifest plan hash is invalid: {path}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or any(
        not isinstance(name, str) or not _is_sha256(value)
        for name, value in inputs.items()
    ):
        raise RuntimeError(f"stage manifest input hashes are invalid: {path}")
    if set(inputs) != STAGE_INPUT_NAMES.get(stage):
        raise RuntimeError(f"stage manifest input names are invalid: {path}")

    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError(f"stage manifest is missing artifact metadata: {path}")
    artifact_path_raw = artifact.get("path")
    if not isinstance(artifact_path_raw, str):
        raise RuntimeError(f"stage manifest artifact path must be a string: {path}")
    artifact_path = _resolve_stage_artifact_path(output_dir, stage, artifact_path_raw)
    if not artifact_path.is_file():
        raise RuntimeError(f"stage artifact missing: {artifact_path}")

    expected_sha = artifact.get("sha256")
    actual_sha = _sha256(artifact_path)
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"artifact sha256 mismatch for stage={stage}: expected={expected_sha} actual={actual_sha}"
        )
    _validate_internal_artifact(stage, artifact_path, artifact, payload)
    return payload
