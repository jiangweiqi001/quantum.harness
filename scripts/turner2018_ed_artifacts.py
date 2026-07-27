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


def _cleanup_partial(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


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
    except Exception:
        _cleanup_partial(partial)
        raise


def write_basis_artifact(output_dir: Path, *, basis: np.ndarray) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "basis.npz"
    partial = target.with_name(target.name + ".partial")
    try:
        with partial.open("wb") as handle:
            np.savez(handle, basis=np.asarray(basis, dtype=np.uint64))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
    except Exception:
        _cleanup_partial(partial)
        raise
    artifact = ArtifactMeta(
        path=target.name,
        sha256=_sha256(target),
        shape=[int(np.asarray(basis).shape[0])],
        dtype=str(np.asarray(basis).dtype),
        conventions=["sorted-constrained-states"],
    )
    return _write_stage_manifest(output_dir, "basis", artifact)


def write_hamiltonian_artifact(output_dir: Path, *, hamiltonian: sp.csr_matrix) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "hamiltonian.csr.npz"
    partial = target.with_name(target.name + ".partial")
    try:
        with partial.open("wb") as handle:
            sp.save_npz(handle, hamiltonian, compressed=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
    except Exception:
        _cleanup_partial(partial)
        raise
    artifact = ArtifactMeta(
        path=target.name,
        sha256=_sha256(target),
        shape=[int(hamiltonian.shape[0]), int(hamiltonian.shape[1])],
        dtype=str(hamiltonian.dtype),
        conventions=["csr-real-symmetric"],
    )
    return _write_stage_manifest(output_dir, "hamiltonian", artifact, {"nnz": int(hamiltonian.nnz)})


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

    _atomic_write_h5(target, writer)
    artifact = ArtifactMeta(
        path=target.name,
        sha256=_sha256(target),
        shape=[int(energies_arr.shape[0])],
        dtype=str(energies_arr.dtype),
        conventions=["eigenvectors-are-columns", "float64", "hdf5"],
    )
    extra: dict[str, Any] = {"has_vectors": vectors_arr is not None}
    if vectors_arr is not None:
        extra["vector_shape"] = [int(vectors_arr.shape[0]), int(vectors_arr.shape[1])]
    return _write_stage_manifest(output_dir, "eigensystem", artifact, extra)


def write_observables(output_dir: Path, observables: dict[str, np.ndarray]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target = output_dir / "observables.h5"
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in observables.items()}

    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        group = handle.create_group("observables")
        for name, array in arrays.items():
            group.create_dataset(name, data=array)

    _atomic_write_h5(target, writer)
    artifact = ArtifactMeta(
        path=target.name,
        sha256=_sha256(target),
        shape=[sum(int(array.size) for array in arrays.values())],
        dtype="float64",
        conventions=["observables-only", "separate-from-eigensystem"],
    )
    return _write_stage_manifest(
        output_dir,
        "observables",
        artifact,
        {"datasets": sorted(arrays.keys())},
    )


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
    artifact_path = output_dir / str(artifact.get("path", ""))
    if not artifact_path.is_file():
        raise RuntimeError(f"stage artifact missing: {artifact_path}")

    expected_sha = artifact.get("sha256")
    actual_sha = _sha256(artifact_path)
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"artifact sha256 mismatch for stage={stage}: expected={expected_sha} actual={actual_sha}"
        )
    return payload
