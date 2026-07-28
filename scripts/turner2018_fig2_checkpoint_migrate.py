#!/usr/bin/env python3
"""Safely refine a trusted Turner Fig. 2 iTEBD checkpoint timestep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from pxp_itebd import (
    ITEBDConfig,
    ITEBDSample,
    configuration_fingerprint,
    load_checkpoint,
    measure_sample,
)
from tenpy.tools import hdf5_io

MIGRATION_SCHEMA = "turner2018.fig2.checkpoint-migration.v1"
MIGRATION_TOOL = "scripts/turner2018_fig2_checkpoint_migrate.py"
HAMILTONIAN = "sum_i P_(i-1) X_i P_(i+1)"
STATES = ("vacuum", "Z2", "Z3", "Z4")
RATIO_TOLERANCE = 1e-10
PHYSICAL_TOLERANCE = 1e-12


def configuration_payload(
    state: str, config: ITEBDConfig
) -> dict[str, str | float | int]:
    return {
        "initial_state": state,
        "hamiltonian": HAMILTONIAN,
        **asdict(config),
    }


def _require_finite_real(value: object, label: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _require_positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{label} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _validate_config_types(config: ITEBDConfig, label: str) -> None:
    _require_positive_integer(config.unit_cell, f"{label} unit_cell")
    _require_positive_integer(config.chi_max, f"{label} chi_max")
    _require_finite_real(config.dt, f"{label} dt", positive=True)
    _require_finite_real(config.svd_min, f"{label} svd_min", positive=True)
    _require_finite_real(config.sample_dt, f"{label} sample_dt", positive=True)
    _require_finite_real(
        config.checkpoint_dt, f"{label} checkpoint_dt", positive=True
    )


def _integer_ratio(numerator: float, denominator: float, label: str) -> int:
    ratio = numerator / denominator
    rounded = round(ratio)
    if rounded < 1 or abs(ratio - rounded) > RATIO_TOLERANCE:
        raise ValueError(f"{label} must be a positive integer within tolerance")
    return rounded


def validate_refinement(
    old_state: str,
    new_state: str,
    old_config: ITEBDConfig,
    new_config: ITEBDConfig,
) -> None:
    """Validate that only a safe integrator timestep refinement is requested."""
    _validate_config_types(old_config, "old")
    _validate_config_types(new_config, "new")
    if old_state != new_state:
        raise ValueError("migration requires the same initial_state")
    if old_state not in STATES:
        raise ValueError(f"unknown initial state: {old_state}")
    for field in ("unit_cell", "chi_max", "svd_min", "sample_dt", "checkpoint_dt"):
        if getattr(old_config, field) != getattr(new_config, field):
            raise ValueError(f"migration requires the same {field}")
    if new_config.dt >= old_config.dt:
        raise ValueError("new dt must be strictly smaller than old dt")
    _integer_ratio(old_config.dt, new_config.dt, "old dt / new dt")
    _integer_ratio(
        new_config.sample_dt, new_config.dt, "sample_dt / new dt"
    )
    _integer_ratio(
        new_config.checkpoint_dt, new_config.dt, "checkpoint_dt / new dt"
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _samples_to_arrays(
    samples: list[ITEBDSample], unit_cell: int
) -> dict[str, np.ndarray]:
    return {
        "time": np.asarray([sample.time for sample in samples], dtype=float),
        "entropy_by_bond": (
            np.stack([sample.entropy for sample in samples])
            if samples
            else np.empty((0, unit_cell), dtype=float)
        ),
        "zz_by_bond": (
            np.stack([sample.zz for sample in samples])
            if samples
            else np.empty((0, unit_cell), dtype=float)
        ),
        "max_chi": np.asarray(
            [sample.max_chi for sample in samples], dtype=np.int64
        ),
        "discarded_interval": np.asarray(
            [sample.discarded_interval for sample in samples], dtype=float
        ),
        "discarded_total": np.asarray(
            [sample.discarded_total for sample in samples], dtype=float
        ),
        "blockade_violation": np.asarray(
            [sample.blockade_violation for sample in samples], dtype=float
        ),
    }


def _assert_samples_exact(
    actual: list[ITEBDSample], expected: list[ITEBDSample]
) -> None:
    if len(actual) != len(expected):
        raise RuntimeError("migrated checkpoint changed the sample count")
    for got, want in zip(actual, expected, strict=True):
        scalar_fields = (
            "time",
            "max_chi",
            "discarded_interval",
            "discarded_total",
            "blockade_violation",
        )
        if any(getattr(got, field) != getattr(want, field) for field in scalar_fields):
            raise RuntimeError("migrated checkpoint changed historical samples")
        if not np.array_equal(got.entropy, want.entropy) or not np.array_equal(
            got.zz, want.zz
        ):
            raise RuntimeError("migrated checkpoint changed historical samples")


def _verify_checkpoint_equality(
    path: Path,
    state: str,
    config: ITEBDConfig,
    source_psi,
    source_time: float,
    source_discarded: float,
    source_samples: list[ITEBDSample],
    provenance: dict[str, object],
) -> None:
    restored, time, discarded, samples = load_checkpoint(path, state, config)
    if time != source_time or discarded != source_discarded:
        raise RuntimeError("migrated checkpoint changed time or discarded_total")
    _assert_samples_exact(samples, source_samples)

    for site in range(source_psi.L):
        if not np.array_equal(
            restored.get_B(site).to_ndarray(),
            source_psi.get_B(site).to_ndarray(),
        ) or not np.array_equal(restored.get_SL(site), source_psi.get_SL(site)):
            raise RuntimeError("migrated checkpoint changed MPS tensors")

    source_observables = measure_sample(
        source_psi, source_time, 0.0, source_discarded
    )
    restored_observables = measure_sample(
        restored, source_time, 0.0, source_discarded
    )
    np.testing.assert_allclose(
        restored_observables.entropy,
        source_observables.entropy,
        rtol=0.0,
        atol=PHYSICAL_TOLERANCE,
    )
    np.testing.assert_allclose(
        restored_observables.zz,
        source_observables.zz,
        rtol=0.0,
        atol=PHYSICAL_TOLERANCE,
    )
    np.testing.assert_allclose(
        restored.norm_test(),
        source_psi.norm_test(),
        rtol=0.0,
        atol=PHYSICAL_TOLERANCE,
    )
    if (
        restored_observables.max_chi != source_observables.max_chi
        or abs(
            restored_observables.blockade_violation
            - source_observables.blockade_violation
        )
        > PHYSICAL_TOLERANCE
    ):
        raise RuntimeError("migrated checkpoint changed MPS diagnostics")

    with h5py.File(path, "r") as handle:
        root_provenance = json.loads(handle.attrs["migration_provenance"])
        payload = hdf5_io.load_from_hdf5(handle)
    if root_provenance != provenance or payload.get("migration_provenance") != provenance:
        raise RuntimeError("migration provenance is not bound to both HDF5 layers")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def migrate_checkpoint_timestep(
    source: str | Path,
    destination: str | Path,
    state: str,
    old_config: ITEBDConfig,
    new_config: ITEBDConfig,
) -> dict[str, str | float]:
    """Migrate a trusted local checkpoint to a safely refined timestep.

    TeNPy checkpoint deserialization reconstructs Python objects. The caller
    must only pass a trusted local source checkpoint.
    """
    validate_refinement(state, state, old_config, new_config)
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination paths must be distinct")
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"destination directory does not exist: {destination.parent}"
        )

    partial_fd, partial_name = tempfile.mkstemp(
        prefix=f".{destination.name}.migration-",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(partial_fd)
    partial = Path(partial_name)
    source_sha256 = _sha256(source)
    old_fingerprint = configuration_fingerprint(state, old_config)
    new_fingerprint = configuration_fingerprint(state, new_config)
    published = False
    try:
        psi, time, discarded_total, samples = load_checkpoint(
            source, state, old_config
        )
        if _sha256(source) != source_sha256:
            raise RuntimeError("source checkpoint changed during migration")
        provenance = {
            "migration_schema": MIGRATION_SCHEMA,
            "migration_tool": MIGRATION_TOOL,
            "source": {
                "sha256": source_sha256,
                "configuration_payload": configuration_payload(state, old_config),
                "configuration_fingerprint": old_fingerprint,
                "time": time,
            },
            "destination": {
                "configuration_payload": configuration_payload(state, new_config),
                "configuration_fingerprint": new_fingerprint,
                "time": time,
            },
        }
        payload = {
            "psi": psi,
            "configuration_fingerprint": new_fingerprint,
            "time": time,
            "discarded_total": discarded_total,
            "samples": _samples_to_arrays(samples, new_config.unit_cell),
            "migration_provenance": provenance,
        }
        with h5py.File(partial, "w") as handle:
            hdf5_io.save_to_hdf5(handle, payload)
            handle.attrs["configuration_fingerprint"] = new_fingerprint
            handle.attrs["migration_provenance"] = json.dumps(
                provenance, sort_keys=True, separators=(",", ":")
            )
            handle.flush()
        _fsync_file(partial)
        _verify_checkpoint_equality(
            partial,
            state,
            new_config,
            psi,
            time,
            discarded_total,
            samples,
            provenance,
        )
        if _sha256(source) != source_sha256:
            raise RuntimeError("source checkpoint changed during migration")

        os.link(partial, destination)
        published = True
        partial.unlink()
        _fsync_directory(destination.parent)
    except Exception:
        if published:
            try:
                destination.unlink()
                _fsync_directory(destination.parent)
            except OSError:
                pass
        raise
    finally:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "destination_fingerprint": new_fingerprint,
        "destination_sha256": _sha256(destination),
        "source_fingerprint": old_fingerprint,
        "source_sha256": source_sha256,
        "state": state,
        "time": time,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--state", required=True, choices=STATES)
    parser.add_argument("--old-dt", required=True, type=float)
    parser.add_argument("--new-dt", required=True, type=float)
    parser.add_argument("--chi-max", type=int, default=400)
    parser.add_argument("--svd-min", type=float, default=1e-12)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--checkpoint-dt", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixed = {
        "chi_max": args.chi_max,
        "svd_min": args.svd_min,
        "sample_dt": args.sample_dt,
        "checkpoint_dt": args.checkpoint_dt,
    }
    summary = migrate_checkpoint_timestep(
        args.source,
        args.destination,
        args.state,
        ITEBDConfig(dt=args.old_dt, **fixed),
        ITEBDConfig(dt=args.new_dt, **fixed),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
