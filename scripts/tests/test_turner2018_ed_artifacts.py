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


def test_transactional_rollback_never_reads_giant_artifacts_into_memory(
    tmp_path, monkeypatch
):
    output_dir = tmp_path
    write_eigensystem(output_dir, energies=np.asarray([0.0]), vectors=np.eye(1))
    target = output_dir / "eigensystem.h5"
    original_sha = artifacts._sha256(target)
    manifest = output_dir / "stages" / "eigensystem.json"
    original_replace = artifacts.os.replace

    def forbidden_read_bytes(self):
        raise AssertionError("transaction rollback must not call read_bytes")

    def fail_manifest_replace(source, destination):
        if Path(destination) == manifest:
            raise RuntimeError("manifest failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    monkeypatch.setattr(artifacts.os, "replace", fail_manifest_replace)
    with pytest.raises(RuntimeError, match="manifest failure"):
        write_eigensystem(
            output_dir,
            energies=np.asarray([2.0]),
            vectors=np.eye(1),
        )
    assert artifacts._sha256(target) == original_sha
    validate_stage(output_dir, "eigensystem")


def test_second_backup_link_failure_cleans_first_and_preserves_fixed_files(
    tmp_path, monkeypatch
):
    write_eigensystem(tmp_path, energies=np.asarray([0.0]), vectors=np.eye(1))
    artifact = tmp_path / "eigensystem.h5"
    manifest = tmp_path / "stages" / "eigensystem.json"
    artifact_sha = artifacts._sha256(artifact)
    manifest_bytes = manifest.read_bytes()
    original_link = artifacts.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second backup link failure")
        return original_link(source, destination)

    monkeypatch.setattr(artifacts.os, "link", fail_second_link)
    with pytest.raises(OSError, match="second backup"):
        write_eigensystem(tmp_path, energies=np.asarray([1.0]), vectors=np.eye(1))

    assert artifacts._sha256(artifact) == artifact_sha
    assert manifest.read_bytes() == manifest_bytes
    assert not (tmp_path / "eigensystem.h5.backup").exists()
    assert not (tmp_path / "stages" / "eigensystem.json.backup").exists()


def test_validate_stage_enforces_schema_metadata_inputs_and_hdf5_structure(tmp_path):
    output_dir = tmp_path
    write_eigensystem(
        output_dir,
        energies=np.asarray([0.0, 1.0]),
        vectors=np.eye(2),
        inputs={"hamiltonian": "a" * 64},
        plan_sha256="b" * 64,
    )
    manifest = output_dir / "stages" / "eigensystem.json"
    original = json.loads(manifest.read_text())
    validate_stage(output_dir, "eigensystem")

    mutations = (
        ("schema_version", "wrong"),
        ("inputs", {"hamiltonian": "bad"}),
        ("plan_sha256", "bad"),
        ("artifact.shape", [999]),
        ("artifact.dtype", "float32"),
        ("artifact.conventions", []),
    )
    for dotted, value in mutations:
        payload = json.loads(json.dumps(original))
        target = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        manifest.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError):
            validate_stage(output_dir, "eigensystem")
    manifest.write_text(json.dumps(original))

    with h5py.File(output_dir / "eigensystem.h5", "r+") as handle:
        del handle["eigensystem/energies"]
    payload = json.loads(manifest.read_text())
    payload["artifact"]["sha256"] = artifacts._sha256(output_dir / "eigensystem.h5")
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="energies"):
        validate_stage(output_dir, "eigensystem")


def test_first_publish_manifest_failure_rolls_back_to_no_stage_and_fsyncs_cleanup(
    tmp_path, monkeypatch
):
    output_dir = tmp_path
    target = output_dir / "eigensystem.h5"
    manifest = output_dir / "stages" / "eigensystem.json"

    original_replace = artifacts.os.replace
    original_open = artifacts.os.open
    original_close = artifacts.os.close
    original_fsync = artifacts.os.fsync
    open_directories: dict[int, Path] = {}
    directory_fsyncs: list[Path] = []

    def fail_manifest_replace(source, destination):
        src = Path(source)
        dst = Path(destination)
        if dst == manifest:
            raise RuntimeError("simulated first publish manifest failure")
        return original_replace(src, dst)

    def tracking_open(path, flags, *args):
        fd = original_open(path, flags, *args)
        open_directories[fd] = Path(path).resolve()
        return fd

    def tracking_close(fd: int):
        open_directories.pop(fd, None)
        return original_close(fd)

    def tracking_fsync(fd: int):
        mode = stat.S_IFMT(artifacts.os.fstat(fd).st_mode)
        if mode == stat.S_IFDIR:
            directory_fsyncs.append(open_directories.get(fd, Path("UNKNOWN")))
        return original_fsync(fd)

    monkeypatch.setattr(artifacts.os, "replace", fail_manifest_replace)
    monkeypatch.setattr(artifacts.os, "open", tracking_open)
    monkeypatch.setattr(artifacts.os, "close", tracking_close)
    monkeypatch.setattr(artifacts.os, "fsync", tracking_fsync)

    with pytest.raises(RuntimeError, match="simulated first publish manifest failure"):
        write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))

    assert not target.exists()
    assert not manifest.exists()
    assert not (output_dir / "eigensystem.h5.partial").exists()
    assert not (output_dir / "eigensystem.h5.backup").exists()
    assert not (output_dir / "stages" / "eigensystem.json.partial").exists()
    assert not (output_dir / "stages" / "eigensystem.json.backup").exists()
    fsynced_directories = set(directory_fsyncs)
    assert output_dir.resolve() in fsynced_directories
    assert (output_dir / "stages").resolve() in fsynced_directories


def test_first_publish_artifact_replace_failure_leaves_no_fixed_target(tmp_path, monkeypatch):
    output_dir = tmp_path
    target = output_dir / "eigensystem.h5"
    original_replace = artifacts.os.replace

    def fail_artifact_replace(source, destination):
        src = Path(source)
        dst = Path(destination)
        if dst == target:
            raise RuntimeError("simulated artifact replace failure")
        return original_replace(src, dst)

    monkeypatch.setattr(artifacts.os, "replace", fail_artifact_replace)

    with pytest.raises(RuntimeError, match="simulated artifact replace failure"):
        write_eigensystem(output_dir, energies=np.asarray([1.0]), vectors=np.eye(1))

    assert not target.exists()
    assert not (output_dir / "eigensystem.h5.partial").exists()
    assert not (output_dir / "eigensystem.h5.backup").exists()
    assert not (output_dir / "stages" / "eigensystem.json").exists()


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
