import json
from pathlib import Path
import stat

import h5py
import numpy as np
import pytest

import turner2018_ed_artifacts as artifacts
from turner2018_ed_artifacts import validate_stage, write_eigensystem, write_observables


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def test_write_eigensystem_atomic_failure_preserves_target_and_cleans_partial(
    tmp_path, monkeypatch
):
    output_dir = tmp_path
    target = output_dir / "eigensystem.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset("sentinel", data=np.asarray([7.0]))
    original = _read_bytes(target)

    def fail_create_dataset(self, name, *args, **kwargs):
        if name == "vectors":
            raise RuntimeError("simulated hdf5 failure")
        return original_create_dataset(self, name, *args, **kwargs)

    original_create_dataset = h5py.Group.create_dataset
    monkeypatch.setattr(h5py.Group, "create_dataset", fail_create_dataset)

    energies = np.asarray([0.0, 1.0], dtype=np.float64)
    vectors = np.eye(2, dtype=np.float64)
    with pytest.raises(RuntimeError, match="simulated hdf5 failure"):
        write_eigensystem(output_dir, energies=energies, vectors=vectors)

    assert _read_bytes(target) == original
    assert not (output_dir / "eigensystem.h5.partial").exists()


def test_validate_stage_rejects_one_byte_artifact_mutation(tmp_path):
    output_dir = tmp_path
    energies = np.asarray([0.0, 1.0], dtype=np.float64)
    vectors = np.eye(2, dtype=np.float64)
    write_eigensystem(output_dir, energies=energies, vectors=vectors)

    stage_path = output_dir / "stages" / "eigensystem.json"
    payload = json.loads(stage_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    validate_stage(output_dir, "eigensystem")

    artifact_path = output_dir / payload["artifact"]["path"]
    blob = bytearray(_read_bytes(artifact_path))
    blob[-1] ^= 0x01
    artifact_path.write_bytes(bytes(blob))

    with pytest.raises(RuntimeError, match="sha256"):
        validate_stage(output_dir, "eigensystem")


def test_write_observables_does_not_modify_eigensystem(tmp_path):
    output_dir = tmp_path
    energies = np.asarray([0.0, 1.0], dtype=np.float64)
    vectors = np.eye(2, dtype=np.float64)
    write_eigensystem(output_dir, energies=energies, vectors=vectors)
    eigensystem_path = output_dir / "eigensystem.h5"
    before = _read_bytes(eigensystem_path)

    write_observables(output_dir, {"pr2": np.asarray([1.0, 1.0], dtype=np.float64)})

    assert (output_dir / "observables.h5").is_file()
    assert _read_bytes(eigensystem_path) == before
    validate_stage(output_dir, "observables")


def test_manifest_publication_failure_rolls_back_prior_stage_and_cleans_temp_files(
    tmp_path, monkeypatch
):
    output_dir = tmp_path
    old_energies = np.asarray([0.0, 1.0], dtype=np.float64)
    old_vectors = np.eye(2, dtype=np.float64)
    write_eigensystem(output_dir, energies=old_energies, vectors=old_vectors)

    target = output_dir / "eigensystem.h5"
    manifest = output_dir / "stages" / "eigensystem.json"
    old_target_bytes = _read_bytes(target)
    old_manifest_bytes = _read_bytes(manifest)

    original_replace = artifacts.os.replace

    def fail_manifest_replace(source, destination):
        src = Path(source)
        dst = Path(destination)
        if dst == manifest:
            raise RuntimeError("simulated manifest publication failure")
        return original_replace(src, dst)

    monkeypatch.setattr(artifacts.os, "replace", fail_manifest_replace)

    new_energies = np.asarray([2.0, 3.0], dtype=np.float64)
    new_vectors = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="simulated manifest publication failure"):
        write_eigensystem(output_dir, energies=new_energies, vectors=new_vectors)

    assert _read_bytes(target) == old_target_bytes
    assert _read_bytes(manifest) == old_manifest_bytes
    assert not (output_dir / "eigensystem.h5.partial").exists()
    assert not (output_dir / "eigensystem.h5.backup").exists()
    assert not (output_dir / "stages" / "eigensystem.json.partial").exists()
    assert not (output_dir / "stages" / "eigensystem.json.backup").exists()


def test_validate_stage_rejects_absolute_artifact_path(tmp_path):
    output_dir = tmp_path
    write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))
    stage_path = output_dir / "stages" / "eigensystem.json"
    payload = json.loads(stage_path.read_text(encoding="utf-8"))
    payload["artifact"]["path"] = "/tmp/evil.h5"
    stage_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="absolute"):
        validate_stage(output_dir, "eigensystem")


def test_validate_stage_rejects_parent_traversal_path(tmp_path):
    output_dir = tmp_path
    write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))
    stage_path = output_dir / "stages" / "eigensystem.json"
    payload = json.loads(stage_path.read_text(encoding="utf-8"))
    payload["artifact"]["path"] = "../escape.h5"
    stage_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="traversal"):
        validate_stage(output_dir, "eigensystem")


def test_validate_stage_rejects_nested_artifact_path(tmp_path):
    output_dir = tmp_path
    write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))
    stage_path = output_dir / "stages" / "eigensystem.json"
    payload = json.loads(stage_path.read_text(encoding="utf-8"))
    payload["artifact"]["path"] = "nested/eigensystem.h5"
    stage_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="nested"):
        validate_stage(output_dir, "eigensystem")


def test_validate_stage_rejects_stage_artifact_name_mismatch(tmp_path):
    output_dir = tmp_path
    write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))
    stage_path = output_dir / "stages" / "eigensystem.json"
    payload = json.loads(stage_path.read_text(encoding="utf-8"))
    payload["artifact"]["path"] = "observables.h5"
    stage_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mismatch"):
        validate_stage(output_dir, "eigensystem")


def test_write_eigensystem_fsyncs_parent_directories_after_replace(tmp_path, monkeypatch):
    output_dir = tmp_path
    seen_directory_fsync = {"count": 0}
    original_fsync = artifacts.os.fsync

    def tracking_fsync(fd: int):
        mode = stat.S_IFMT(artifacts.os.fstat(fd).st_mode)
        if mode == stat.S_IFDIR:
            seen_directory_fsync["count"] += 1
        return original_fsync(fd)

    monkeypatch.setattr(artifacts.os, "fsync", tracking_fsync)
    write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))
    assert seen_directory_fsync["count"] >= 2
