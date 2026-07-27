import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import h5py
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

import turner2018_fig2_itebd
from pxp_ed import (
    basis_state_vector,
    constrained_basis,
    density_wave_state,
    pxp_hamiltonian,
)
from pxp_itebd import (
    ITEBDConfig,
    build_imps,
    configuration_fingerprint,
    evolve_imps,
    load_checkpoint,
)
from turner2018_fig2 import evolve_observables, half_chain_entropy
from turner2018_fig2_itebd import render_figures, run_state


def _open_chain_center_entropy(length, times):
    basis = constrained_basis(length, pbc=False)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=False)
    initial = basis_state_vector(basis, density_wave_state(length, 2))
    evolved = expm_multiply(
        -1j * hamiltonian,
        initial,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    return np.asarray(
        [half_chain_entropy(state, basis, length) for state in evolved]
    )


def test_z2_short_time_mean_zz_matches_periodic_ed():
    times = np.arange(0.0, 0.5 + 0.025, 0.05)
    config = ITEBDConfig(dt=0.05, sample_dt=0.05, chi_max=64)
    samples = evolve_imps(
        build_imps("Z2", config),
        config,
        start_time=0.0,
        target_time=0.5,
    )
    ed = evolve_observables(
        length=14,
        initial_period=2,
        times=times,
        compute_entropy=False,
    )

    itebd_zz = np.asarray([np.mean(sample.zz) for sample in samples])
    assert np.max(np.abs(itebd_zz - ed["zz"])) < 5e-3


def test_z2_short_time_center_entropy_matches_open_chain_ed():
    times = np.arange(0.0, 0.3 + 0.025, 0.05)
    config = ITEBDConfig(dt=0.05, sample_dt=0.05, chi_max=64)
    samples = evolve_imps(
        build_imps("Z2", config),
        config,
        start_time=0.0,
        target_time=0.3,
    )

    open_ed_entropy = _open_chain_center_entropy(12, times)
    itebd_entropy = np.asarray([sample.entropy[6] for sample in samples])
    assert np.max(np.abs(itebd_entropy - open_ed_entropy)) < 5e-3


def test_run_state_writes_result_and_checkpoint(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )

    result_path = run_state(
        "Z2",
        target_time=0.1,
        config=config,
        output_dir=tmp_path,
    )
    checkpoint_path = tmp_path / "fig2_itebd_Z2_checkpoint.h5"

    assert result_path.is_file()
    assert checkpoint_path.is_file()
    with h5py.File(result_path) as handle:
        assert handle["entropy_by_bond"].shape == (2, 12)
        assert handle["zz_by_bond"].shape == (2, 12)


def test_resume_continues_without_duplicating_boundary_sample(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    run_state("Z2", target_time=0.1, config=config, output_dir=tmp_path)

    result_path = run_state(
        "Z2",
        target_time=0.2,
        config=config,
        output_dir=tmp_path,
        resume=True,
    )

    with h5py.File(result_path) as handle:
        np.testing.assert_allclose(handle["time"][...], [0.0, 0.1, 0.2])


def test_resumed_result_matches_uninterrupted_values_and_discarded_continuity(
    tmp_path,
):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    uninterrupted_path = run_state(
        "Z2",
        target_time=0.2,
        config=config,
        output_dir=uninterrupted,
    )
    run_state("Z2", target_time=0.1, config=config, output_dir=resumed)
    resumed_path = run_state(
        "Z2",
        target_time=0.2,
        config=config,
        output_dir=resumed,
        resume=True,
    )

    with h5py.File(uninterrupted_path) as expected, h5py.File(resumed_path) as actual:
        assert set(actual) == set(expected)
        for name in expected:
            np.testing.assert_allclose(actual[name][...], expected[name][...])
        discarded_interval = actual["discarded_interval"][...]
        discarded_total = actual["discarded_total"][...]
    assert np.all(np.diff(discarded_total) >= 0.0)
    np.testing.assert_allclose(
        discarded_total[1:] - discarded_total[:-1],
        discarded_interval[1:],
    )


def test_checkpoint_cadence_and_exact_progress_format(tmp_path, monkeypatch, capsys):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    checkpoint_times = []
    original = turner2018_fig2_itebd.save_checkpoint

    def record_checkpoint(path, psi, state, config, time, discarded, samples):
        checkpoint_times.append(time)
        return original(path, psi, state, config, time, discarded, samples)

    monkeypatch.setattr(
        turner2018_fig2_itebd,
        "save_checkpoint",
        record_checkpoint,
    )
    run_state("Z2", target_time=0.2, config=config, output_dir=tmp_path)

    np.testing.assert_allclose(checkpoint_times, [0.1, 0.2])
    lines = capsys.readouterr().out.splitlines()
    assert lines
    pattern = re.compile(
        r"^state=Z2 time=\d+\.\d{3} max_chi=\d+ "
        r"discarded_total=\d\.\d{6}e[+-]\d{2} "
        r"blockade=\d\.\d{6}e[+-]\d{2}$"
    )
    assert all(pattern.fullmatch(line) for line in lines)


def test_checkpoint_failure_preserves_last_checkpoint_and_result(tmp_path, monkeypatch):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.1,
        checkpoint_dt=0.1,
    )
    result_path = run_state(
        "Z2",
        target_time=0.1,
        config=config,
        output_dir=tmp_path,
    )
    checkpoint_path = tmp_path / "fig2_itebd_Z2_checkpoint.h5"
    old_result = result_path.read_bytes()
    old_checkpoint = checkpoint_path.read_bytes()
    original = turner2018_fig2_itebd.save_checkpoint

    def fail_at_next_checkpoint(path, psi, state, config, time, discarded, samples):
        if time > 0.1:
            raise RuntimeError("simulated orchestration interruption")
        return original(path, psi, state, config, time, discarded, samples)

    monkeypatch.setattr(
        turner2018_fig2_itebd,
        "save_checkpoint",
        fail_at_next_checkpoint,
    )
    with pytest.raises(RuntimeError, match="simulated orchestration interruption"):
        run_state(
            "Z2",
            target_time=0.2,
            config=config,
            output_dir=tmp_path,
            resume=True,
        )

    assert result_path.read_bytes() == old_result
    assert checkpoint_path.read_bytes() == old_checkpoint
    _, checkpoint_time, _, _ = load_checkpoint(checkpoint_path, "Z2", config)
    assert checkpoint_time == pytest.approx(0.1)


def test_checkpoint_interval_must_preserve_global_sample_grid(tmp_path):
    config = ITEBDConfig(
        dt=0.05,
        chi_max=8,
        sample_dt=0.2,
        checkpoint_dt=0.1,
    )

    with pytest.raises(ValueError, match="checkpoint_dt.*sample_dt"):
        run_state(
            "Z2",
            target_time=0.2,
            config=config,
            output_dir=tmp_path,
        )


def _write_result(
    path: Path,
    state: str,
    *,
    time: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    time = np.array([0.0, 1.0, 2.0, 3.0]) if time is None else np.asarray(time)
    entropy = np.column_stack(
        [2.0 * time + 1.0]
        + [2.0 * time + 10.0 + bond for bond in range(1, 12)]
    )
    zz = np.repeat((1.0 - 0.1 * time)[:, None], 12, axis=1)
    config = ITEBDConfig()
    fingerprint = configuration_fingerprint(state, config)
    arrays = {
        "time": time,
        "entropy_by_bond": entropy,
        "zz_by_bond": zz,
        "max_chi": np.arange(1, len(time) + 1, dtype=np.int64),
        "discarded_interval": np.arange(len(time), dtype=float) * 1e-8,
        "discarded_total": np.cumsum(np.arange(len(time), dtype=float) * 1e-8),
        "blockade_violation": (np.arange(len(time)) + 1.0) * 1e-16,
    }
    with h5py.File(path, "w") as handle:
        for name, values in arrays.items():
            handle.create_dataset(name, data=values)
        handle.attrs["configuration"] = json.dumps(
            {
                "initial_state": state,
                "hamiltonian": "sum_i P_(i-1) X_i P_(i+1)",
                **config.fingerprint_payload(),
            }
        )
        handle.attrs["provenance"] = json.dumps(
            {
                "configuration_fingerprint": fingerprint,
                "format_version": 1,
                "generator": "scripts/pxp_itebd.py",
            }
        )
    return arrays


def _write_official_data(path: Path, *, stop: float = 3.0) -> None:
    path.mkdir()
    members = {
        "vacuum": "Ent_Z1.dat",
        "Z2": "Ent_Z2.dat",
        "Z3": "Ent_Z3.dat",
        "Z4": "Ent_Z4.dat",
    }
    times = np.arange(0.0, stop + 0.5, 1.0)
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


def test_render_figures_selects_state_specific_entropy_cuts(tmp_path, monkeypatch):
    states = ("vacuum", "Z2", "Z3", "Z4")
    result_paths = []
    generated = {}
    for state in states:
        path = tmp_path / f"fig2_itebd_{state}.h5"
        generated[state] = _write_result(path, state)
        result_paths.append(path)
    official_dir = tmp_path / "official"
    _write_official_data(official_dir)

    plt.close("all")
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    render_figures(
        result_paths,
        official_dir,
        tmp_path,
        fit_window=(0.0, 3.0),
    )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    paper_figure = next(figure for figure in figures if len(figure.axes) == 3)
    assert paper_figure.axes[0].get_ylabel() == "entropy at state-specific cut"
    for state in states:
        line = next(
            line
            for line in paper_figure.axes[0].lines
            if line.get_label() == f"generated iTEBD {state}"
        )
        expected_cut = 11 if state == "Z4" else 0
        np.testing.assert_allclose(
            line.get_ydata(),
            generated[state]["entropy_by_bond"][:, expected_cut],
        )
        assert not np.allclose(
            line.get_ydata(),
            generated[state]["entropy_by_bond"][:, 0 if expected_cut == 11 else 11],
        )

    z2_residual_line = next(
        line
        for line in paper_figure.axes[1].lines
        if line.get_label() == "generated iTEBD Z2"
    )
    z2_time = generated["Z2"]["time"]
    z2_cut0 = generated["Z2"]["entropy_by_bond"][:, 0]
    selected = (z2_time >= 0.0) & (z2_time <= 3.0)
    slope, intercept = np.polyfit(z2_time[selected], z2_cut0[selected], deg=1)
    np.testing.assert_allclose(
        z2_residual_line.get_ydata(),
        z2_cut0 - (slope * z2_time + intercept),
    )

    metrics = json.loads(
        (tmp_path / "fig2_itebd_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["selected_entropy_cut"] == 0
    assert metrics["entropy_cut_by_state"] == {
        "vacuum": 0,
        "Z2": 0,
        "Z3": 0,
        "Z4": 11,
    }
    for state in states:
        expected_cut = 11 if state == "Z4" else 0
        assert metrics["states"][state]["selected_entropy_cut"] == expected_cut

    official_times = np.arange(0.0, 3.0 + 0.5, 1.0)
    official_values = 3.0 * official_times + 2.0
    z4_time = generated["Z4"]["time"]
    z4_cut11 = generated["Z4"]["entropy_by_bond"][:, 11]
    interpolated = np.interp(official_times, z4_time, z4_cut11)
    expected_z4_rmse = float(
        np.sqrt(np.mean((interpolated - official_values) ** 2))
    )
    assert metrics["states"]["Z4"]["official_entropy_rmse"] == pytest.approx(
        expected_z4_rmse
    )
    z4_cut0 = generated["Z4"]["entropy_by_bond"][:, 0]
    cut0_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.interp(official_times, z4_time, z4_cut0) - official_values
                )
                ** 2
            )
        )
    )
    assert cut0_rmse != pytest.approx(expected_z4_rmse)
    plt.close("all")


def test_render_figures_records_styles_cut_metrics_diagnostics_and_provenance(
    tmp_path,
    monkeypatch,
):
    states = ("vacuum", "Z2", "Z3", "Z4")
    result_paths = []
    generated = {}
    for state in states:
        path = tmp_path / f"fig2_itebd_{state}.h5"
        generated[state] = _write_result(path, state)
        result_paths.append(path)
    official_dir = tmp_path / "official"
    _write_official_data(official_dir)
    plotted = []
    original_plot = Axes.plot

    def record_plot(self, *args, **kwargs):
        plotted.append(kwargs.copy())
        return original_plot(self, *args, **kwargs)

    plt.close("all")
    monkeypatch.setattr(Axes, "plot", record_plot)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    paper, diagnostics = render_figures(
        result_paths,
        official_dir,
        tmp_path,
        fit_window=(0.0, 3.0),
    )

    assert paper.is_file()
    assert diagnostics.is_file()
    labels = {entry.get("label"): entry for entry in plotted}
    assert labels["generated iTEBD Z2"].get("linestyle", "-") != "--"
    assert labels["official Turner et al. Z2"]["linestyle"] == "--"
    assert labels["official Turner et al. Z2 correlation"]["linestyle"] == "--"

    figures = [plt.figure(number) for number in plt.get_fignums()]
    paper_figure = next(figure for figure in figures if len(figure.axes) == 3)
    diagnostics_figure = next(figure for figure in figures if len(figure.axes) == 4)
    z2_line = next(
        line
        for line in paper_figure.axes[0].lines
        if line.get_label() == "generated iTEBD Z2"
    )
    np.testing.assert_allclose(
        z2_line.get_ydata(),
        generated["Z2"]["entropy_by_bond"][:, 0],
    )
    assert not np.allclose(
        z2_line.get_ydata(),
        np.mean(generated["Z2"]["entropy_by_bond"], axis=1),
    )
    assert [len(axis.lines) for axis in diagnostics_figure.axes] == [4, 4, 8, 4]
    assert diagnostics_figure.axes[3].get_yscale() == "log"
    assert len(diagnostics_figure.axes[0].collections) == 4

    metrics = json.loads(
        (tmp_path / "fig2_itebd_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["selected_entropy_cut"] == 0
    assert metrics["entropy_cut_by_state"] == {
        "vacuum": 0,
        "Z2": 0,
        "Z3": 0,
        "Z4": 11,
    }
    assert metrics["fit_window"] == [0.0, 3.0]
    assert metrics["official_data_complete"] is True
    assert metrics["comparison"]["interpolation_method"] == "linear numpy.interp"
    assert metrics["official_sources"]["entropy"]["path"].endswith(
        "entanglement_entropy_growth.zip"
    )
    assert metrics["official_sources"]["entropy"]["sha256"] == hashlib.sha256(
        (official_dir / "entanglement_entropy_growth.zip").read_bytes()
    ).hexdigest()
    z2_metrics = metrics["states"]["Z2"]
    assert z2_metrics["configuration_fingerprint"] == configuration_fingerprint(
        "Z2",
        ITEBDConfig(),
    )
    assert z2_metrics["generated_fit"]["intercept"] == pytest.approx(1.0)
    assert z2_metrics["generated_fit"]["slope"] == pytest.approx(2.0)
    assert z2_metrics["official_fit"]["intercept"] == pytest.approx(2.0)
    assert z2_metrics["official_fit"]["slope"] == pytest.approx(3.0)
    assert z2_metrics["official_entropy_rmse"] == pytest.approx(
        np.sqrt(np.mean((np.arange(4, dtype=float) + 1.0) ** 2))
    )
    assert z2_metrics["official_correlation_rmse"] == pytest.approx(0.2)
    assert z2_metrics["entropy_comparison"] == {
        "count": 4,
        "interval": [0.0, 3.0],
        "interpolation_method": "linear numpy.interp",
    }
    assert set(metrics["states"]) == set(states)
    plt.close("all")


def test_missing_official_data_fails_by_default(tmp_path):
    result = tmp_path / "fig2_itebd_vacuum.h5"
    _write_result(result, "vacuum")

    with pytest.raises(FileNotFoundError, match="official Turner"):
        render_figures(
            [result],
            tmp_path / "missing",
            tmp_path,
            fit_window=(0.0, 3.0),
        )


def test_allow_missing_official_marks_outputs_incomplete(tmp_path, monkeypatch):
    result = tmp_path / "fig2_itebd_Z2.h5"
    _write_result(result, "Z2")
    saved_figures = []
    original_savefig = plt.Figure.savefig

    def record_figure(self, *args, **kwargs):
        saved_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", record_figure)
    paper, diagnostics = render_figures(
        [result],
        tmp_path / "missing",
        tmp_path,
        fit_window=(0.0, 3.0),
        allow_missing_official=True,
    )

    assert paper.is_file()
    assert diagnostics.is_file()
    assert any(
        "INCOMPLETE" in text.get_text()
        for figure in saved_figures
        for text in figure.texts
    )
    metrics = json.loads(
        (tmp_path / "fig2_itebd_metrics_Z2.json").read_text(encoding="utf-8")
    )
    assert metrics["official_data_complete"] is False
    assert metrics["incomplete_reason"]
    assert metrics["official_sources"]["entropy"]["available"] is False


def test_render_figures_records_partial_status_banner_and_metadata(
    tmp_path,
    monkeypatch,
):
    result_paths = []
    for state in ("vacuum", "Z2", "Z3", "Z4"):
        path = tmp_path / f"fig2_itebd_{state}.h5"
        _write_result(path, state)
        result_paths.append(path)
    official_dir = tmp_path / "official"
    _write_official_data(official_dir)

    plt.close("all")
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    paper, diagnostics = render_figures(
        result_paths,
        official_dir,
        tmp_path,
        fit_window=(0.0, 3.0),
        partial_status_text="PARTIAL SNAPSHOT",
        partial_status_metadata={"mode": "snapshot", "fit_stop": 3.0},
    )

    assert paper.is_file()
    assert diagnostics.is_file()
    figures = [plt.figure(number) for number in plt.get_fignums()]
    for figure in figures:
        assert any(
            text.get_text() == "PARTIAL SNAPSHOT" for text in figure.texts
        )
    metrics = json.loads(
        (tmp_path / "fig2_itebd_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["partial_status"] == {
        "text": "PARTIAL SNAPSHOT",
        "metadata": {"mode": "snapshot", "fit_stop": 3.0},
    }
    plt.close("all")


@pytest.mark.parametrize("state", ["vacuum", "Z3", "Z4"])
def test_non_z2_state_renders_only_distinct_diagnostics(state, tmp_path):
    result = tmp_path / f"fig2_itebd_{state}.h5"
    _write_result(result, state)
    official = tmp_path / "official"
    _write_official_data(official)

    paper, diagnostics = render_figures(
        [result],
        official,
        tmp_path,
        fit_window=(0.0, 3.0),
    )

    assert paper is None
    assert diagnostics == tmp_path / f"fig2_itebd_diagnostics_{state}.png"
    assert diagnostics.is_file()
    assert not (tmp_path / f"fig2_itebd_paper_{state}.png").exists()


@pytest.mark.parametrize(
    ("generated_stop", "official_stop", "message"),
    [(2.0, 3.0, "generated.*fit window"), (3.0, 2.0, "official.*fit window")],
)
def test_fit_window_requires_generated_and_official_endpoint_coverage(
    generated_stop,
    official_stop,
    message,
    tmp_path,
):
    result = tmp_path / "fig2_itebd_Z2.h5"
    _write_result(
        result,
        "Z2",
        time=np.arange(0.0, generated_stop + 0.5, 1.0),
    )
    official = tmp_path / "official"
    _write_official_data(official, stop=official_stop)

    with pytest.raises(ValueError, match=message):
        render_figures(
            [result],
            official,
            tmp_path,
            fit_window=(0.0, 3.0),
        )


def test_render_rejects_duplicate_result_states(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_result(first, "Z2")
    _write_result(second, "Z2")

    with pytest.raises(ValueError, match="duplicate.*Z2"):
        render_figures(
            [first, second],
            tmp_path,
            tmp_path,
            fit_window=(0.0, 3.0),
            allow_missing_official=True,
        )


def _replace_entropy_with_wrong_width(handle):
    data = handle["entropy_by_bond"][...][:, :11]
    del handle["entropy_by_bond"]
    handle.create_dataset("entropy_by_bond", data=data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda handle: handle.__delitem__("zz_by_bond"), "schema"),
        (_replace_entropy_with_wrong_width, "entropy_by_bond.*shape"),
        (
            lambda handle: handle["time"].__setitem__(1, np.nan),
            "finite",
        ),
        (
            lambda handle: handle.attrs.__setitem__(
                "provenance",
                json.dumps(
                    {
                        "format_version": 1,
                        "generator": "scripts/pxp_itebd.py",
                    }
                ),
            ),
            "configuration_fingerprint",
        ),
    ],
)
def test_render_rejects_invalid_result_schema(tmp_path, mutate, message):
    result = tmp_path / "result.h5"
    _write_result(result, "Z2")
    with h5py.File(result, "r+") as handle:
        mutate(handle)

    with pytest.raises((TypeError, ValueError), match=message):
        render_figures(
            [result],
            tmp_path,
            tmp_path,
            fit_window=(0.0, 3.0),
            allow_missing_official=True,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("configuration", [], "configuration.*object"),
        ("provenance", [], "provenance.*object"),
        (
            "configuration",
            {
                "initial_state": "Z2",
                "hamiltonian": "wrong Hamiltonian",
                **ITEBDConfig().fingerprint_payload(),
            },
            "Hamiltonian",
        ),
        (
            "provenance",
            {
                "configuration_fingerprint": configuration_fingerprint(
                    "Z2", ITEBDConfig()
                ),
                "format_version": 2,
                "generator": "scripts/pxp_itebd.py",
            },
            "format_version",
        ),
        (
            "provenance",
            {
                "configuration_fingerprint": configuration_fingerprint(
                    "Z2", ITEBDConfig()
                ),
                "format_version": 1,
                "generator": "unexpected.py",
            },
            "generator",
        ),
    ],
)
def test_render_rejects_invalid_metadata_contract(
    tmp_path,
    attribute,
    value,
    message,
):
    result = tmp_path / "result.h5"
    _write_result(result, "Z2")
    with h5py.File(result, "r+") as handle:
        handle.attrs[attribute] = json.dumps(value)

    with pytest.raises(ValueError, match=message):
        render_figures(
            [result],
            tmp_path,
            tmp_path,
            fit_window=(0.0, 3.0),
            allow_missing_official=True,
        )


def test_cli_defaults_and_all_options():
    defaults = turner2018_fig2_itebd.build_parser().parse_args([])
    assert defaults.state == "all"
    assert defaults.target_time == 12.0
    assert defaults.dt == 0.05
    assert defaults.chi_max == 400
    assert defaults.sample_dt == 0.1
    assert defaults.checkpoint_dt == 1.0
    assert defaults.resume is False
    assert defaults.fit_start == 0.0
    assert defaults.fit_stop == 12.0
    assert defaults.allow_missing_official is False

    explicit = turner2018_fig2_itebd.build_parser().parse_args(
        [
            "--state",
            "Z3",
            "--target-time",
            "3",
            "--dt",
            "0.025",
            "--chi-max",
            "64",
            "--sample-dt",
            "0.05",
            "--checkpoint-dt",
            "0.5",
            "--resume",
            "--fit-start",
            "1",
            "--fit-stop",
            "3",
            "--output-dir",
            "out",
            "--official-data-dir",
            "official",
            "--allow-missing-official",
        ]
    )
    assert explicit.state == "Z3"
    assert explicit.target_time == 3.0
    assert explicit.dt == 0.025
    assert explicit.chi_max == 64
    assert explicit.sample_dt == 0.05
    assert explicit.checkpoint_dt == 0.5
    assert explicit.resume is True
    assert explicit.fit_start == 1.0
    assert explicit.fit_stop == 3.0
    assert explicit.output_dir == Path("out")
    assert explicit.official_data_dir == Path("official")
    assert explicit.allow_missing_official is True


def test_cli_preflights_official_sources_before_any_evolution(tmp_path, monkeypatch):
    calls = []

    def forbidden_run_state(*args, **kwargs):
        calls.append("run_state")
        raise AssertionError("run_state must not be called")

    def forbidden_evolution(*args, **kwargs):
        calls.append("evolve_imps")
        raise AssertionError("evolution must not be called")

    monkeypatch.setattr(
        turner2018_fig2_itebd,
        "run_state",
        forbidden_run_state,
    )
    monkeypatch.setattr(
        turner2018_fig2_itebd,
        "evolve_imps",
        forbidden_evolution,
    )

    with pytest.raises(FileNotFoundError, match="official Turner"):
        turner2018_fig2_itebd.main(
            [
                "--state",
                "all",
                "--official-data-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert calls == []


@pytest.mark.parametrize("state", ["vacuum", "Z2", "Z3", "Z4"])
def test_cli_each_state_completes_with_distinct_files(state, tmp_path):
    assert (
        turner2018_fig2_itebd.main(
            [
                "--state",
                state,
                "--target-time",
                "0.1",
                "--dt",
                "0.05",
                "--chi-max",
                "8",
                "--fit-stop",
                "0.1",
                "--allow-missing-official",
                "--official-data-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / f"fig2_itebd_{state}.h5").is_file()
    assert (tmp_path / f"fig2_itebd_{state}_checkpoint.h5").is_file()
    assert (tmp_path / f"fig2_itebd_diagnostics_{state}.png").is_file()
    assert (tmp_path / f"fig2_itebd_metrics_{state}.json").is_file()
    if state == "Z2":
        assert (tmp_path / "fig2_itebd_paper_Z2.png").is_file()
    else:
        assert not (tmp_path / f"fig2_itebd_paper_{state}.png").exists()


def test_make_targets_pin_all_required_cli_content():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    for target, target_time in (
        ("turner-fig2-itebd-stage1", "12"),
        ("turner-fig2-itebd-stage2", "30"),
    ):
        recipe = makefile.split(f"{target}:", 1)[1].split("\n\n", 1)[0]
        assert "scripts/turner2018_fig2_itebd.py" in recipe
        assert "--state all" in recipe
        assert f"--target-time {target_time}" in recipe
        assert "--dt 0.05" in recipe
        assert "--chi-max 400" in recipe
        assert "--resume" in recipe
        assert "--allow-missing-official" not in recipe
    readme = Path("tracks/ed/README.md").read_text(encoding="utf-8")
    convergence = readme.split(
        "Use a separate output directory for the time-step convergence run:",
        1,
    )[1].split("```", 2)[1]
    assert "--target-time 3" in convergence
    assert "--fit-stop 3" in convergence


def test_state_all_runs_states_sequentially(monkeypatch, tmp_path):
    calls = []

    def fake_run_state(state, **kwargs):
        calls.append(state)
        return tmp_path / f"{state}.h5"

    monkeypatch.setattr(turner2018_fig2_itebd, "run_state", fake_run_state)
    monkeypatch.setattr(
        turner2018_fig2_itebd,
        "render_figures",
        lambda *args, **kwargs: (
            tmp_path / "paper.png",
            tmp_path / "diagnostics.png",
        ),
    )

    assert (
        turner2018_fig2_itebd.main(
            [
                "--state",
                "all",
                "--target-time",
                "0.1",
                "--dt",
                "0.05",
                "--chi-max",
                "8",
                "--allow-missing-official",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == ["vacuum", "Z2", "Z3", "Z4"]
