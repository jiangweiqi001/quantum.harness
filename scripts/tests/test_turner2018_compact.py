from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import turner2018_fig3 as fig3
import turner2018_fig4 as fig4
from turner2018_compact import (
    export_compact_package,
    load_compact_package,
)
from turner2018_l32_server import main as server_main, require_stage


def _full_package(root: Path) -> Path:
    output = root / "full"
    assert server_main(
        [
            "--length",
            "10",
            "--stage",
            "all",
            "--declared-memory",
            "1G",
            "--chunk-columns",
            "4",
            "--output-dir",
            str(output),
        ]
    ) == 0
    return output


def test_compact_export_loads_without_local_vectors_and_keeps_full_validation_strict(
    tmp_path,
):
    full = _full_package(tmp_path)
    compact = tmp_path / "compact-root" / "L10"
    compact.parent.mkdir()

    manifest = export_compact_package(full, compact)
    loaded = load_compact_package(compact)

    assert not (compact / "eigensystem.h5").exists()
    assert manifest["compact_artifact"]["sha256"] == loaded["source_hashes"][
        "eigenvalues.h5"
    ]
    assert loaded["source_hashes"]["source_eigensystem.h5"] == json.loads(
        (full / "stages" / "diagonalize.json").read_text()
    )["artifact"]["sha256"]
    assert loaded["energies"].shape == (14,)
    assert np.all(np.diff(loaded["energies"]) >= 0)
    assert loaded["eigenvector_metadata"]["access"] == "remote-source-reference-only"
    assert loaded["validation"]["passed"] is True
    fig3_loaded = fig3.load_independent_results(compact.parent)[10]
    fig4_loaded = fig4.load_independent_fig4_results(compact.parent)[10]
    np.testing.assert_array_equal(fig3_loaded["energies"], loaded["energies"])
    np.testing.assert_array_equal(fig4_loaded["energies"], loaded["energies"])
    assert fig3_loaded["eigenvector_metadata"]["local"] is False
    assert fig4_loaded["eigenvector_metadata"]["local"] is False
    clone = tmp_path / "cloned-root" / "L10"
    clone.parent.mkdir()
    export_compact_package(compact, clone)
    cloned = load_compact_package(clone)
    np.testing.assert_array_equal(cloned["energies"], loaded["energies"])
    assert cloned["source_hashes"] == loaded["source_hashes"]
    with pytest.raises(RuntimeError, match="stage artifact missing"):
        require_stage(compact, "diagonalize")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda root: (root / "eigenvalues.h5").open("ab").write(b"corrupt"),
            "compact artifact sha256",
        ),
        (
            lambda root: _replace_energies(root, np.array([0.0] * 14)),
            "compact artifact sha256",
        ),
        (
            lambda root: _mark_validation_failed(root),
            "validation",
        ),
    ],
)
def test_compact_loader_fails_closed_on_provenance_or_validation_changes(
    tmp_path, mutation, message
):
    full = _full_package(tmp_path)
    compact = tmp_path / "compact"
    export_compact_package(full, compact)

    mutation(compact)

    with pytest.raises(RuntimeError, match=message):
        load_compact_package(compact)


def _replace_energies(root: Path, values: np.ndarray) -> None:
    with h5py.File(root / "eigenvalues.h5", "r+") as handle:
        handle["eigensystem/energies"][:] = values


def _mark_validation_failed(root: Path) -> None:
    path = root / "validation" / "metrics.json"
    payload = json.loads(path.read_text())
    payload["passed"] = False
    path.write_text(json.dumps(payload))
