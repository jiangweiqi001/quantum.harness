#!/usr/bin/env python3
"""Render a read-only partial Fig. 2 snapshot from copied live checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from pxp_itebd import ITEBDConfig, load_checkpoint, write_results
from turner2018_fig2_itebd import (
    DEFAULT_OFFICIAL_DATA,
    DEFAULT_OUTPUT,
    STATES,
    render_figures,
)


DEFAULT_SOURCE = DEFAULT_OUTPUT
DEFAULT_OUTPUT_SNAPSHOT = Path("tracks/ed/results/turner-2018/fig2-itebd-partial")
DEFAULT_PARTIAL_STATUS = "PARTIAL SNAPSHOT"


def _checkpoint_path(directory: Path, state: str) -> Path:
    return directory / f"fig2_itebd_{state}_checkpoint.h5"


def _result_path(directory: Path, state: str) -> Path:
    return directory / f"fig2_itebd_{state}.h5"


def _manifest_path(directory: Path) -> Path:
    return directory / "fig2_itebd_snapshot_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path | None) -> None:
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


def _copy_checkpoint_atomic(
    source_path: Path, destination_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    partial_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    digest = hashlib.sha256()
    try:
        partial_path.unlink(missing_ok=True)
        with source_path.open("rb") as source, partial_path.open("wb") as copied:
            source_stat = os.fstat(source.fileno())
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                copied.write(chunk)
        os.replace(partial_path, destination_path)
    finally:
        partial_path.unlink(missing_ok=True)
    source_sha256 = digest.hexdigest()
    copied_sha256 = _sha256_file(destination_path)
    if copied_sha256 != source_sha256:
        raise ValueError(
            f"copied checkpoint hash mismatch for {source_path.name}: "
            f"{source_sha256} != {copied_sha256}"
        )
    return (
        {
            "path": str(source_path),
            "sha256": source_sha256,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        {
            "path": str(destination_path),
            "sha256": copied_sha256,
        },
    )


def _validated_checkpoint_time(checkpoint_time: float, target_time: float, state: str) -> float:
    if not math.isfinite(checkpoint_time):
        raise ValueError(f"{state} has nonfinite checkpoint time")
    if checkpoint_time > target_time + 1e-10:
        raise ValueError(
            f"{state} checkpoint time {checkpoint_time} exceeds requested target "
            f"{target_time}"
        )
    return float(checkpoint_time)


def _replace_path_prefix(value: object, old_prefix: str, new_prefix: str) -> object:
    if isinstance(value, dict):
        return {
            key: _replace_path_prefix(item, old_prefix, new_prefix)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_path_prefix(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and (
        value == old_prefix or value.startswith(old_prefix + os.sep)
    ):
        return new_prefix + value[len(old_prefix) :]
    return value


def _rewrite_metrics_paths(metrics_path: Path, staging_dir: Path, output_dir: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rewritten = _replace_path_prefix(metrics, str(staging_dir), str(output_dir))
    metrics_path.write_text(
        json.dumps(rewritten, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_snapshot_directory(staging_dir: Path, output_dir: Path) -> None:
    backup_dir = output_dir.with_name(output_dir.name + ".backup")
    _remove_tree(backup_dir)
    if output_dir.exists():
        os.replace(output_dir, backup_dir)
        try:
            os.replace(staging_dir, output_dir)
        except Exception:
            os.replace(backup_dir, output_dir)
            raise
        _remove_tree(backup_dir)
        return
    os.replace(staging_dir, output_dir)


def create_snapshot(
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT_SNAPSHOT,
    target_time: float,
    config: ITEBDConfig,
    official_data_dir: str | Path = DEFAULT_OFFICIAL_DATA,
    partial_status_text: str = DEFAULT_PARTIAL_STATUS,
    partial_status_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if target_time < 0.0:
        raise ValueError("target_time must be nonnegative")
    config.validate()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from source_dir for read-only snapshots")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    missing = [
        state
        for state in STATES
        if not _checkpoint_path(source_dir, state).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing checkpoint(s): " + ", ".join(missing)
        )

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        result_paths: list[Path] = []
        state_horizons: dict[str, float] = {}
        comparison_intervals: dict[str, list[float]] = {}
        manifest_states: dict[str, dict[str, object]] = {}

        for state in STATES:
            source_path = _checkpoint_path(source_dir, state)
            copied_path = _checkpoint_path(staging_dir, state)
            source_metadata, copied_metadata = _copy_checkpoint_atomic(
                source_path, copied_path
            )
            _, checkpoint_time, _, samples = load_checkpoint(copied_path, state, config)
            checkpoint_time = _validated_checkpoint_time(
                checkpoint_time, target_time, state
            )
            source_metadata["checkpoint_time"] = checkpoint_time
            copied_metadata["checkpoint_time"] = checkpoint_time
            copied_metadata["path"] = str(_checkpoint_path(output_dir, state))

            result_path = _result_path(staging_dir, state)
            write_results(result_path, state, config, samples)

            state_horizons[state] = checkpoint_time
            comparison_intervals[state] = [0.0, checkpoint_time]
            manifest_states[state] = {
                "source_checkpoint": source_metadata,
                "copied_checkpoint": copied_metadata,
                "result": {
                    "path": str(_result_path(output_dir, state)),
                    "checkpoint_time": checkpoint_time,
                },
            }
            result_paths.append(result_path)

        effective_fit_window = [0.0, min(target_time, state_horizons["Z2"])]
        if effective_fit_window[0] >= effective_fit_window[1]:
            raise ValueError("effective fit window requires completed Z2 data past t=0")

        status_metadata = dict(partial_status_metadata or {})
        status_metadata.update(
            {
                "requested_target_time": float(target_time),
                "effective_fit_window": effective_fit_window,
                "state_horizons": state_horizons,
                "comparison_intervals": comparison_intervals,
            }
        )
        paper_path, diagnostics_path = render_figures(
            result_paths,
            official_data_dir,
            staging_dir,
            fit_window=(effective_fit_window[0], effective_fit_window[1]),
            partial_status_text=partial_status_text,
            partial_status_metadata=status_metadata,
            x_limit=target_time,
        )
        metrics_path = staging_dir / "fig2_itebd_metrics.json"
        _rewrite_metrics_paths(metrics_path, staging_dir, output_dir)
        manifest_path = _manifest_path(staging_dir)
        manifest = {
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "target_time": float(target_time),
            "effective_fit_window": effective_fit_window,
            "comparison_intervals": comparison_intervals,
            "state_horizons": state_horizons,
            "paper_path": str(output_dir / Path(paper_path).name)
            if paper_path is not None
            else None,
            "diagnostics_path": str(output_dir / Path(diagnostics_path).name),
            "metrics_path": str(output_dir / metrics_path.name),
            "states": manifest_states,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_snapshot_directory(staging_dir, output_dir)
    except Exception:
        _remove_tree(staging_dir)
        raise
    return {
        "paper_path": str(output_dir / Path(paper_path).name) if paper_path is not None else None,
        "diagnostics_path": str(output_dir / Path(diagnostics_path).name),
        "metrics_path": str(output_dir / "fig2_itebd_metrics.json"),
        "manifest_path": str(output_dir / "fig2_itebd_snapshot_manifest.json"),
        "effective_fit_window": effective_fit_window,
        "state_horizons": state_horizons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_SNAPSHOT)
    parser.add_argument("--official-data-dir", type=Path, default=DEFAULT_OFFICIAL_DATA)
    parser.add_argument("--target-time", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--chi-max", type=int, default=400)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--checkpoint-dt", type=float, default=1.0)
    parser.add_argument(
        "--partial-status-text",
        default=DEFAULT_PARTIAL_STATUS,
        help="visible banner drawn on snapshot figures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ITEBDConfig(
        dt=args.dt,
        chi_max=args.chi_max,
        sample_dt=args.sample_dt,
        checkpoint_dt=args.checkpoint_dt,
    )
    snapshot = create_snapshot(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_time=args.target_time,
        config=config,
        official_data_dir=args.official_data_dir,
        partial_status_text=args.partial_status_text,
    )
    for key in ("paper_path", "diagnostics_path", "metrics_path", "manifest_path"):
        value = snapshot[key]
        if value is not None:
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
