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

import h5py
import numpy as np
import scipy
import scipy.sparse as sp


SCHEMA_VERSION = "turner2018-independent-ed-v1"
STAGE_ARTIFACT_NAMES = {
    "basis": "basis.npz",
    "hamiltonian": "hamiltonian.csr.npz",
    "eigensystem": "eigensystem.h5",
    "observables": "observables.h5",
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
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": "complete",
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


def _load_previous_stage_state(
    output_dir: Path,
    stage: str,
    artifact_path: Path,
) -> tuple[bytes | None, bytes | None]:
    manifest_path = output_dir / "stages" / f"{stage}.json"
    if not manifest_path.is_file() or not artifact_path.is_file():
        return None, None

    try:
        payload = validate_stage(output_dir, stage)
    except RuntimeError:
        return None, None

    manifest_artifact = payload.get("artifact", {}).get("path")
    if manifest_artifact != artifact_path.name:
        return None, None
    return artifact_path.read_bytes(), manifest_path.read_bytes()


def _restore_stage_state(
    *,
    artifact_path: Path,
    manifest_path: Path,
    old_artifact_bytes: bytes,
    old_manifest_bytes: bytes,
) -> None:
    artifact_backup = artifact_path.with_name(artifact_path.name + ".backup")
    manifest_backup = manifest_path.with_name(manifest_path.name + ".backup")
    try:
        with artifact_backup.open("wb") as handle:
            handle.write(old_artifact_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(artifact_backup, artifact_path)
        _fsync_parent_directory(artifact_path)

        with manifest_backup.open("wb") as handle:
            handle.write(old_manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_backup, manifest_path)
        _fsync_parent_directory(manifest_path)
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
) -> dict[str, Any]:
    manifest_path = output_dir / "stages" / f"{stage}.json"
    old_artifact_bytes, old_manifest_bytes = _load_previous_stage_state(
        output_dir, stage, artifact_path
    )

    write_artifact()
    artifact = build_artifact()
    try:
        return _write_stage_manifest(output_dir, stage, artifact, extra)
    except Exception:
        if old_artifact_bytes is not None and old_manifest_bytes is not None:
            _restore_stage_state(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                old_artifact_bytes=old_artifact_bytes,
                old_manifest_bytes=old_manifest_bytes,
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


def write_basis_artifact(output_dir: Path, *, basis: np.ndarray) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "basis.npz"
    basis_array = np.asarray(basis, dtype=np.uint64)

    def write_artifact() -> None:
        _atomic_write_npz(target, lambda handle: np.savez(handle, basis=basis_array))

    def build_artifact() -> ArtifactMeta:
        return ArtifactMeta(
            path=target.name,
            sha256=_sha256(target),
            shape=[int(basis_array.shape[0])],
            dtype=str(basis_array.dtype),
            conventions=["sorted-constrained-states"],
        )

    return _publish_stage_transactional(
        output_dir=output_dir,
        stage="basis",
        artifact_path=target,
        write_artifact=write_artifact,
        build_artifact=build_artifact,
    )


def write_hamiltonian_artifact(output_dir: Path, *, hamiltonian: sp.csr_matrix) -> dict[str, Any]:
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
    )


def write_eigensystem(
    output_dir: Path,
    *,
    energies: np.ndarray,
    vectors: np.ndarray | None,
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
        stage="eigensystem",
        artifact_path=target,
        write_artifact=write_artifact,
        build_artifact=build_artifact,
        extra=extra,
    )


def write_observables(output_dir: Path, observables: dict[str, np.ndarray]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "observables.h5"
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in observables.items()}

    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        group = handle.create_group("observables")
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
    )


def _resolve_stage_artifact_path(output_dir: Path, stage: str, artifact_path_raw: str) -> Path:
    artifact_relative = Path(artifact_path_raw)
    if artifact_relative.is_absolute():
        raise RuntimeError(f"absolute artifact paths are forbidden: {artifact_path_raw}")
    if ".." in artifact_relative.parts:
        raise RuntimeError(f"path traversal is forbidden: {artifact_path_raw}")
    if len(artifact_relative.parts) != 1:
        raise RuntimeError(f"nested artifact paths are forbidden: {artifact_path_raw}")

    expected = STAGE_ARTIFACT_NAMES.get(stage)
    if expected is not None and artifact_relative.name != expected:
        raise RuntimeError(
            f"stage/artifact-name mismatch: stage={stage} expected={expected} actual={artifact_relative.name}"
        )

    output_resolved = output_dir.resolve()
    artifact_path = (output_dir / artifact_relative).resolve()
    try:
        artifact_path.relative_to(output_resolved)
    except ValueError as exc:
        raise RuntimeError(f"artifact path escapes output directory: {artifact_path_raw}") from exc
    return artifact_path


def validate_stage(output_dir: Path, stage: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    path = output_dir / "stages" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"missing stage manifest: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or payload.get("status") != "complete":
        raise RuntimeError(f"stage manifest is incomplete: {path}")

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
    return payload
