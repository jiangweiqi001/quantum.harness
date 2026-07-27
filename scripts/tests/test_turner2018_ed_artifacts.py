import json
from pathlib import Path

import h5py
import numpy as np
import pytest

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
