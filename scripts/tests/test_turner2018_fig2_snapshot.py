import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pxp_itebd import ITEBDConfig
from turner2018_fig2_itebd import STATES, load_checkpoint, run_state
import turner2018_fig2_snapshot
from turner2018_fig2_snapshot import _copy_checkpoint_atomic, create_snapshot


def _write_official_data(path: Path, *, stop: float = 1.0) -> None:
    path.mkdir()
    members = {
        "vacuum": "Ent_Z1.dat",
        "Z2": "Ent_Z2.dat",
        "Z3": "Ent_Z3.dat",
        "Z4": "Ent_Z4.dat",
    }
    times = np.arange(0.0, stop + 0.05, 0.1)
    with ZipFile(path / "entanglement_entropy_growth.zip", "w") as archive:
        for member in members.values():
            archive.writestr(
                member,
                "".join(f"{time} {3.0 * time + 2.0}\n" for time in times),
            )
    (path / "corzz_Neel.dat").write_text(
        "".join(f"{time} {1.2 - 0.1 * time}\n" for time in times),
        encoding="utf-8",
    )


def _write_checkpoints(
    output_dir: Path,
    config: ITEBDConfig,
    targets: dict[str, float],
) -> dict[str, Path]:
    paths = {}
    for state, target_time in targets.items():
        run_state(
            state,
            target_time=target_time,
            config=config,
            output_dir=output_dir,
        )
        paths[state] = output_dir / f"fig2_itebd_{state}_checkpoint.h5"
    return paths


def test_snapshot_copies_sources_writes_results_and_uses_completed_horizons(
    tmp_path,
    monkeypatch,
):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=16,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    target_time = 12.0
    checkpoint_targets = {"vacuum": 0.4, "Z2": 0.3, "Z3": 0.2, "Z4": 0.1}
    checkpoint_paths = _write_checkpoints(source_dir, config, checkpoint_targets)
    _write_official_data(official_dir, stop=target_time)

    source_metadata = {}
    for state, path in checkpoint_paths.items():
        stat = path.stat()
        _, checkpoint_time, _, _ = load_checkpoint(path, state, config)
        source_metadata[state] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime_ns": stat.st_mtime_ns,
            "checkpoint_time": checkpoint_time,
        }

    loaded_paths = []
    original_load_checkpoint = turner2018_fig2_snapshot.load_checkpoint

    def record_load(path, state, locked_config):
        loaded_paths.append(Path(path))
        return original_load_checkpoint(path, state, locked_config)

    monkeypatch.setattr(turner2018_fig2_snapshot, "load_checkpoint", record_load)
    plt.close("all")
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    snapshot = create_snapshot(
        source_dir=source_dir,
        output_dir=output_dir,
        target_time=target_time,
        config=config,
        official_data_dir=official_dir,
        partial_status_text="PARTIAL SNAPSHOT",
        partial_status_metadata={"requested_target_time": target_time},
    )

    for key in ("paper_path", "diagnostics_path", "metrics_path", "manifest_path"):
        assert Path(snapshot[key]).is_file()
    assert loaded_paths
    assert all(path.parent != source_dir for path in loaded_paths)

    metrics = json.loads(Path(snapshot["metrics_path"]).read_text(encoding="utf-8"))
    effective_fit_window = [0.0, checkpoint_targets["Z2"]]
    comparison_intervals = {
        state: [0.0, horizon] for state, horizon in checkpoint_targets.items()
    }
    assert metrics["fit_window"] == pytest.approx(effective_fit_window)
    assert metrics["partial_status"]["text"] == "PARTIAL SNAPSHOT"
    partial_metadata = metrics["partial_status"]["metadata"]
    assert partial_metadata["requested_target_time"] == pytest.approx(target_time)
    assert partial_metadata["effective_fit_window"] == pytest.approx(
        effective_fit_window
    )
    for state, horizon in checkpoint_targets.items():
        assert partial_metadata["state_horizons"][state] == pytest.approx(horizon)
        assert partial_metadata["comparison_intervals"][state] == pytest.approx(
            comparison_intervals[state]
        )
    for state, horizon in checkpoint_targets.items():
        state_metrics = metrics["states"][state]
        assert state_metrics["time_range"] == pytest.approx([0.0, horizon])
        assert state_metrics["sample_count"] == round(horizon / config.sample_dt) + 1
        assert state_metrics["entropy_comparison"]["interval"] == pytest.approx(
            [0.0, horizon]
        )
    assert metrics["states"]["Z2"]["correlation_comparison"]["interval"] == pytest.approx(
        [0.0, checkpoint_targets["Z2"]]
    )

    manifest = json.loads(Path(snapshot["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["effective_fit_window"] == pytest.approx(effective_fit_window)
    assert manifest["target_time"] == pytest.approx(target_time)
    assert set(manifest["states"]) == set(STATES)
    for state, horizon in checkpoint_targets.items():
        assert manifest["state_horizons"][state] == pytest.approx(horizon)
        assert manifest["comparison_intervals"][state] == pytest.approx(
            comparison_intervals[state]
        )
    for state in STATES:
        source_path = checkpoint_paths[state]
        source = manifest["states"][state]["source_checkpoint"]
        copied = manifest["states"][state]["copied_checkpoint"]
        result = manifest["states"][state]["result"]
        assert Path(source["path"]) == source_path
        assert source["sha256"] == source_metadata[state]["sha256"]
        assert source["mtime_ns"] == source_metadata[state]["mtime_ns"]
        assert source["checkpoint_time"] == pytest.approx(
            source_metadata[state]["checkpoint_time"]
        )
        assert Path(copied["path"]).is_file()
        assert copied["sha256"] == source["sha256"]
        assert Path(result["path"]).is_file()
        with h5py.File(result["path"], "r") as handle:
            checkpoint_time = source_metadata[state]["checkpoint_time"]
            np.testing.assert_allclose(
                handle["time"][...],
                np.arange(
                    0.0,
                    checkpoint_time + 0.5 * config.sample_dt,
                    config.sample_dt,
                ),
            )
            assert handle["time"][-1] == pytest.approx(
                source_metadata[state]["checkpoint_time"]
            )
        assert source_path.stat().st_mtime_ns == source_metadata[state]["mtime_ns"]
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == source_metadata[state]["sha256"]
        )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    paper_figure = next(figure for figure in figures if len(figure.axes) == 3)
    diagnostics_figure = next(figure for figure in figures if len(figure.axes) == 4)
    for figure in (paper_figure, diagnostics_figure):
        for axis in figure.axes:
            assert axis.get_xlim()[1] == pytest.approx(target_time)
    for figure in (paper_figure, diagnostics_figure):
        for axis in figure.axes:
            for line in axis.lines:
                label = line.get_label()
                if label.startswith("generated"):
                    state = next(
                        state_name
                        for state_name in STATES
                        if state_name in label
                    )
                    assert np.max(line.get_xdata()) == pytest.approx(
                        checkpoint_targets[state]
                    )
                if label.startswith("official Turner et al."):
                    assert np.max(line.get_xdata()) == pytest.approx(target_time)
    plt.close("all")


def test_snapshot_requires_all_four_states(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    _write_checkpoints(source_dir, config, {"vacuum": 0.1, "Z2": 0.1, "Z3": 0.1})
    _write_official_data(official_dir)

    with pytest.raises(FileNotFoundError, match="Z4"):
        create_snapshot(
            source_dir=source_dir,
            output_dir=output_dir,
            target_time=0.1,
            config=config,
            official_data_dir=official_dir,
        )


def test_snapshot_rejects_checkpoint_beyond_requested_target(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    _write_checkpoints(
        source_dir,
        config,
        {"vacuum": 0.1, "Z2": 0.2, "Z3": 0.1, "Z4": 0.1},
    )
    _write_official_data(official_dir)

    with pytest.raises(ValueError, match="requested target"):
        create_snapshot(
            source_dir=source_dir,
            output_dir=output_dir,
            target_time=0.1,
            config=config,
            official_data_dir=official_dir,
        )


def test_snapshot_rejects_nonfinite_checkpoint_time(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    checkpoints = _write_checkpoints(
        source_dir,
        config,
        {state: 0.1 for state in STATES},
    )
    _write_official_data(official_dir)
    with h5py.File(checkpoints["Z2"], "r+") as handle:
        handle["time"][...] = np.nan

    with pytest.raises(ValueError, match="nonfinite checkpoint time"):
        create_snapshot(
            source_dir=source_dir,
            output_dir=output_dir,
            target_time=0.1,
            config=config,
            official_data_dir=official_dir,
        )


def test_snapshot_rejects_fingerprint_mismatch(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    checkpoints = _write_checkpoints(
        source_dir,
        config,
        {state: 0.1 for state in STATES},
    )
    _write_official_data(official_dir)
    with h5py.File(checkpoints["Z3"], "r+") as handle:
        handle.attrs["configuration_fingerprint"] = "mismatch"

    with pytest.raises(ValueError, match="fingerprint"):
        create_snapshot(
            source_dir=source_dir,
            output_dir=output_dir,
            target_time=0.1,
            config=config,
            official_data_dir=official_dir,
        )


def test_copy_checkpoint_atomic_uses_single_descriptor_for_bytes_and_mtime(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "source.h5"
    destination_path = tmp_path / "copied.h5"
    replacement_path = tmp_path / "replacement.h5"
    source_path.write_bytes(b"old-checkpoint")
    replacement_path.write_bytes(b"new-checkpoint")
    old_ns = (1_700_000_000_000_000_000, 1_700_000_000_000_000_000)
    new_ns = (1_800_000_000_000_000_000, 1_800_000_000_000_000_000)
    os.utime(source_path, ns=old_ns)
    os.utime(replacement_path, ns=new_ns)

    original_open = Path.open
    swapped = False

    def race_open(self, *args, **kwargs):
        nonlocal swapped
        if self == source_path and not swapped:
            swapped = True
            os.replace(replacement_path, source_path)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race_open)
    source_metadata, copied_metadata = _copy_checkpoint_atomic(
        source_path, destination_path
    )

    expected_sha = hashlib.sha256(b"new-checkpoint").hexdigest()
    assert source_metadata["sha256"] == expected_sha
    assert copied_metadata["sha256"] == expected_sha
    assert source_metadata["mtime_ns"] == new_ns[1]
    assert destination_path.read_bytes() == b"new-checkpoint"


def test_snapshot_cleans_staging_and_leaves_no_output_on_write_failure(
    tmp_path,
    monkeypatch,
):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    _write_checkpoints(source_dir, config, {state: 0.1 for state in STATES})
    _write_official_data(official_dir)

    original_write_results = turner2018_fig2_snapshot.write_results

    def fail_after_partial_write(path, state, locked_config, samples):
        original_write_results(path, state, locked_config, samples)
        if state == "Z4":
            raise RuntimeError("simulated staged write failure")

    monkeypatch.setattr(
        turner2018_fig2_snapshot,
        "write_results",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError, match="simulated staged write failure"):
        create_snapshot(
            source_dir=source_dir,
            output_dir=output_dir,
            target_time=0.1,
            config=config,
            official_data_dir=official_dir,
        )

    assert not output_dir.exists()
    assert not any(tmp_path.glob("snapshot.staging-*"))


def test_snapshot_failure_leaves_existing_output_untouched(tmp_path, monkeypatch):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "snapshot"
    official_dir = tmp_path / "official"
    _write_checkpoints(source_dir, config, {state: 0.1 for state in STATES})
    _write_official_data(official_dir)
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep-me", encoding="utf-8")

    def fail_render(*args, **kwargs):
        staging_dir = Path(args[2])
        (staging_dir / "transient.txt").write_text("staged", encoding="utf-8")
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(turner2018_fig2_snapshot, "render_figures", fail_render)

    with pytest.raises(RuntimeError, match="simulated render failure"):
        create_snapshot(
            source_dir=source_dir,
            output_dir=output_dir,
            target_time=0.1,
            config=config,
            official_data_dir=official_dir,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep-me"
    assert sorted(path.name for path in output_dir.iterdir()) == ["sentinel.txt"]
    assert not any(tmp_path.glob("snapshot.staging-*"))
