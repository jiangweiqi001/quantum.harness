#!/usr/bin/env python3
"""Finalize trusted Fig. 2 checkpoints without evolving or mutating them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np

from pxp_itebd import (
    ITEBDConfig,
    _samples_to_arrays,
    load_checkpoint,
    write_results,
)
from turner2018_fig2_itebd import STATES, _load_result

FINALIZATION_SCHEMA = "turner2018.fig2.checkpoint-finalization.v1"
FINALIZATION_TOOL = "scripts/turner2018_fig2_finalize.py"
BLOCKADE_VIOLATION_LIMIT = 1e-5


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def finalize_checkpoint(
    checkpoint_path: str | Path,
    result_path: str | Path,
    *,
    state: str,
    config: ITEBDConfig,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Copy validated checkpoint samples into a normal result artifact."""
    checkpoint_path = Path(checkpoint_path)
    result_path = Path(result_path)
    if state not in STATES:
        raise ValueError(f"unknown initial state: {state}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    source_sha256 = _sha256(checkpoint_path)
    if source_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            "checkpoint sha256 mismatch: "
            f"expected={expected_checkpoint_sha256} actual={source_sha256}"
        )
    _psi, checkpoint_time, discarded_total, samples = load_checkpoint(
        checkpoint_path, state, config
    )
    if _sha256(checkpoint_path) != source_sha256:
        raise RuntimeError("checkpoint changed while it was read")
    if not samples or samples[-1].time != checkpoint_time:
        raise RuntimeError("checkpoint samples do not end at checkpoint time")
    if samples[-1].discarded_total != discarded_total:
        raise RuntimeError("checkpoint discarded_total does not match final sample")
    if samples[-1].blockade_violation > BLOCKADE_VIOLATION_LIMIT:
        raise RuntimeError(
            "last checkpoint sample blockade violation exceeds accepted limit"
        )
    expected_arrays = _samples_to_arrays(samples, config.unit_cell)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result_path.name}.finalize-",
        suffix=".part",
        dir=result_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_results(temporary, state, config, samples)
        with h5py.File(temporary, "r+") as handle:
            provenance = json.loads(handle.attrs["provenance"])
            provenance["checkpoint_finalization"] = {
                "schema": FINALIZATION_SCHEMA,
                "tool": FINALIZATION_TOOL,
                "mode": "read-only-snapshot",
                "source_checkpoint_path": str(checkpoint_path),
                "source_checkpoint_sha256": source_sha256,
                "checkpoint_time": checkpoint_time,
            }
            handle.attrs["provenance"] = json.dumps(provenance, sort_keys=True)
            handle.flush()
        loaded_state, arrays, _metadata = _load_result(temporary)
        if loaded_state != state:
            raise RuntimeError("finalized result state changed")
        for name, expected in expected_arrays.items():
            if not np.array_equal(arrays[name], expected):
                raise RuntimeError(f"finalized result changed checkpoint samples: {name}")
        if _sha256(checkpoint_path) != source_sha256:
            raise RuntimeError("checkpoint changed during finalization")
        _fsync_file(temporary)
        os.replace(temporary, result_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": FINALIZATION_SCHEMA,
        "state": state,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": source_sha256,
        "sample_count": len(samples),
        "time_range": [float(samples[0].time), float(samples[-1].time)],
        "last_blockade_violation": float(samples[-1].blockade_violation),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--state", required=True, choices=STATES)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--chi-max", type=int, default=400)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--checkpoint-dt", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize_checkpoint(
        args.checkpoint,
        args.result,
        state=args.state,
        config=ITEBDConfig(
            dt=args.dt,
            chi_max=args.chi_max,
            sample_dt=args.sample_dt,
            checkpoint_dt=args.checkpoint_dt,
        ),
        expected_checkpoint_sha256=args.expected_sha256,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
