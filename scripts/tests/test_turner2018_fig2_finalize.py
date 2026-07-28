from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pxp_itebd import ITEBDConfig, build_imps, evolve_imps, save_checkpoint
from turner2018_fig2_finalize import finalize_checkpoint
from turner2018_fig2_itebd import _load_result


def _checkpoint(path: Path) -> tuple[ITEBDConfig, bytes]:
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=1.0,
    )
    psi = build_imps("Z2", config)
    samples = evolve_imps(psi, config, start_time=0.0, target_time=0.1)
    save_checkpoint(
        path,
        psi,
        "Z2",
        config,
        0.1,
        samples[-1].discarded_total,
        samples,
    )
    return config, path.read_bytes()


def test_finalize_checkpoint_is_read_only_exact_and_sha_bound(tmp_path):
    checkpoint = tmp_path / "fig2_itebd_Z2_checkpoint.h5"
    config, source_bytes = _checkpoint(checkpoint)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    result = tmp_path / "fig2_itebd_Z2.h5"

    summary = finalize_checkpoint(
        checkpoint,
        result,
        state="Z2",
        config=config,
        expected_checkpoint_sha256=source_sha256,
    )

    assert checkpoint.read_bytes() == source_bytes
    assert summary["source_checkpoint_sha256"] == source_sha256
    assert summary["time_range"] == [0.0, 0.1]
    state, arrays, metadata = _load_result(result)
    assert state == "Z2"
    np.testing.assert_array_equal(arrays["time"], [0.0, 0.1])
    provenance = metadata["provenance"]["checkpoint_finalization"]
    assert provenance["source_checkpoint_sha256"] == source_sha256
    assert provenance["mode"] == "read-only-snapshot"
    with h5py.File(checkpoint, "r") as handle:
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], handle[f"samples/{name}"][()])


def test_finalize_checkpoint_rejects_wrong_hash_without_touching_output(tmp_path):
    checkpoint = tmp_path / "fig2_itebd_Z2_checkpoint.h5"
    config, source_bytes = _checkpoint(checkpoint)
    result = tmp_path / "fig2_itebd_Z2.h5"
    result.write_bytes(b"keep")

    with pytest.raises(RuntimeError, match="checkpoint sha256"):
        finalize_checkpoint(
            checkpoint,
            result,
            state="Z2",
            config=config,
            expected_checkpoint_sha256="0" * 64,
        )

    assert checkpoint.read_bytes() == source_bytes
    assert result.read_bytes() == b"keep"


def test_finalize_checkpoint_rejects_unaccepted_last_blockade(tmp_path):
    checkpoint = tmp_path / "fig2_itebd_Z2_checkpoint.h5"
    config, source_bytes = _checkpoint(checkpoint)
    with h5py.File(checkpoint, "r+") as handle:
        handle["samples/blockade_violation"][-1] = 2e-5
    changed = checkpoint.read_bytes()

    with pytest.raises(RuntimeError, match="blockade"):
        finalize_checkpoint(
            checkpoint,
            tmp_path / "result.h5",
            state="Z2",
            config=config,
            expected_checkpoint_sha256=hashlib.sha256(changed).hexdigest(),
        )

    assert checkpoint.read_bytes() == changed
