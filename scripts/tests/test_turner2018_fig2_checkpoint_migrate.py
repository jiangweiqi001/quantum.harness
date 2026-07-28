from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import turner2018_fig2_checkpoint_migrate as migration
from pxp_itebd import (
    ITEBDConfig,
    build_imps,
    configuration_fingerprint,
    evolve_imps,
    load_checkpoint,
    measure_sample,
    save_checkpoint,
)


def _configs() -> tuple[ITEBDConfig, ITEBDConfig]:
    old = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        svd_min=1e-12,
        sample_dt=0.1,
        checkpoint_dt=1.0,
    )
    return old, replace(old, dt=0.025)


def _source_checkpoint(path: Path, state: str = "Z2"):
    old, _ = _configs()
    psi = build_imps(state, old)
    samples = evolve_imps(psi, old, 0.0, 0.1)
    discarded = samples[-1].discarded_total
    save_checkpoint(path, psi, state, old, 0.1, discarded, samples)
    return psi, discarded, samples


def _assert_samples_exact(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.time == want.time
        np.testing.assert_array_equal(got.entropy, want.entropy)
        np.testing.assert_array_equal(got.zz, want.zz)
        assert got.max_chi == want.max_chi
        assert got.discarded_interval == want.discarded_interval
        assert got.discarded_total == want.discarded_total
        assert got.blockade_violation == want.blockade_violation


def test_migration_preserves_checkpoint_and_binds_hash_provenance(tmp_path):
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"
    original_psi, original_discarded, original_samples = _source_checkpoint(source)
    old, new = _configs()
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    summary = migration.migrate_checkpoint_timestep(
        source, destination, "Z2", old, new
    )

    assert source.read_bytes() == source_bytes
    assert summary == {
        "destination_fingerprint": configuration_fingerprint("Z2", new),
        "destination_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_fingerprint": configuration_fingerprint("Z2", old),
        "source_sha256": source_sha256,
        "state": "Z2",
        "time": 0.1,
    }
    with h5py.File(destination, "r") as handle:
        root_provenance = json.loads(handle.attrs["migration_provenance"])
        payload = migration.hdf5_io.load_from_hdf5(handle)
    assert payload["migration_provenance"] == root_provenance
    assert root_provenance == {
        "destination": {
            "configuration_fingerprint": configuration_fingerprint("Z2", new),
            "configuration_payload": migration.configuration_payload("Z2", new),
            "time": 0.1,
        },
        "migration_schema": "turner2018.fig2.checkpoint-migration.v1",
        "migration_tool": "scripts/turner2018_fig2_checkpoint_migrate.py",
        "source": {
            "configuration_fingerprint": configuration_fingerprint("Z2", old),
            "configuration_payload": migration.configuration_payload("Z2", old),
            "sha256": source_sha256,
            "time": 0.1,
        },
    }

    with pytest.raises(ValueError, match="configuration fingerprint"):
        load_checkpoint(destination, "Z2", old)
    restored, time, discarded, restored_samples = load_checkpoint(
        destination, "Z2", new
    )
    assert time == 0.1
    assert discarded == original_discarded
    _assert_samples_exact(restored_samples, original_samples)

    original_observables = measure_sample(
        original_psi, time, 0.0, original_discarded
    )
    restored_observables = measure_sample(restored, time, 0.0, discarded)
    np.testing.assert_allclose(
        restored_observables.entropy,
        original_observables.entropy,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        restored_observables.zz,
        original_observables.zz,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        restored.norm_test(), original_psi.norm_test(), rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("old_dt", "new_dt", "message"),
    [
        (0.05, 0.05, "strictly smaller"),
        (0.05, 0.1, "strictly smaller"),
        (0.05, 0.03, "positive integer"),
        (True, 0.025, "finite real number"),
        (0.05, False, "finite real number"),
        (np.nan, 0.025, "finite real number"),
        (0.05, np.inf, "finite real number"),
        (0.0, 0.025, "positive"),
        (0.05, 0.0, "positive"),
        (-0.05, 0.025, "positive"),
    ],
)
def test_migration_rejects_unsafe_timestep_changes(
    tmp_path, old_dt, new_dt, message
):
    old, new = _configs()
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"

    with pytest.raises((TypeError, ValueError), match=message):
        migration.migrate_checkpoint_timestep(
            source,
            destination,
            "Z2",
            replace(old, dt=old_dt),
            replace(new, dt=new_dt),
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_state", "Z3", "same initial_state"),
        ("unit_cell", 24, "same unit_cell"),
        ("chi_max", 9, "same chi_max"),
        ("svd_min", 1e-10, "same svd_min"),
        ("sample_dt", 0.2, "same sample_dt"),
        ("checkpoint_dt", 2.0, "same checkpoint_dt"),
    ],
)
def test_migration_rejects_non_dt_configuration_changes(
    tmp_path, field, value, message
):
    old, new = _configs()
    old_state = "Z2"
    new_state = "Z2"
    if field == "initial_state":
        new_state = value
    else:
        new = replace(new, **{field: value})

    with pytest.raises(ValueError, match=message):
        migration.validate_refinement(old_state, new_state, old, new)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_dt", 0.11),
        ("checkpoint_dt", 1.01),
    ],
)
def test_migration_requires_new_step_to_divide_cadences(field, value):
    old, new = _configs()
    old = replace(old, **{field: value})
    new = replace(new, **{field: value})

    with pytest.raises(ValueError, match=f"{field} / new dt"):
        migration.validate_refinement("Z2", "Z2", old, new)


def test_migration_rejects_same_missing_mismatched_and_existing_paths(
    tmp_path, monkeypatch
):
    old, new = _configs()
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"

    with pytest.raises(ValueError, match="distinct"):
        migration.migrate_checkpoint_timestep(source, source, "Z2", old, new)
    with pytest.raises(FileNotFoundError, match="source checkpoint"):
        migration.migrate_checkpoint_timestep(
            source, destination, "Z2", old, new
        )

    _source_checkpoint(source)
    destination.write_bytes(b"keep me")
    with pytest.raises(FileExistsError, match="destination"):
        migration.migrate_checkpoint_timestep(
            source, destination, "Z2", old, new
        )
    assert destination.read_bytes() == b"keep me"

    destination.unlink()
    monkeypatch.setattr(
        migration.hdf5_io,
        "load_from_hdf5",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must reject before deserialization")
        ),
    )
    with pytest.raises(ValueError, match="configuration fingerprint"):
        migration.migrate_checkpoint_timestep(
            source, destination, "Z3", old, new
        )


def test_migration_source_named_like_legacy_partial_is_safe(tmp_path):
    destination = tmp_path / "destination.h5"
    source = destination.with_name(f".{destination.name}.migration-part")
    _source_checkpoint(source)
    source_bytes = source.read_bytes()
    old, new = _configs()

    migration.migrate_checkpoint_timestep(source, destination, "Z2", old, new)

    assert source.read_bytes() == source_bytes
    assert destination.exists()


def test_migration_failure_preserves_destination_state_and_cleans_partial(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"
    _source_checkpoint(source)
    old, new = _configs()

    monkeypatch.setattr(
        migration,
        "_verify_checkpoint_equality",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated verification failure")
        ),
    )
    with pytest.raises(RuntimeError, match="verification failure"):
        migration.migrate_checkpoint_timestep(
            source, destination, "Z2", old, new
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".destination.h5.migration-*.part")) == []
    assert source.is_file()


def test_cli_emits_concise_json_summary(tmp_path, capsys):
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"
    _source_checkpoint(source)

    assert migration.main(
        [
            str(source),
            str(destination),
            "--state",
            "Z2",
            "--old-dt",
            "0.05",
            "--new-dt",
            "0.025",
            "--chi-max",
            "8",
            "--svd-min",
            "1e-12",
            "--sample-dt",
            "0.1",
            "--checkpoint-dt",
            "1.0",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert summary["destination_sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    assert summary["state"] == "Z2"
