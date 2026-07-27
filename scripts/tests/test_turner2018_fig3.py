import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pxp_ed import constrained_basis, density_wave_state, pxp_hamiltonian
import turner2018_fig3 as fig3
import turner2018_l32_server as server
from turner2018_official import pr2_fsa_averages, select_fig3_pr2_states
from turner2018_l32_server import main as server_main
from turner2018_fig3 import (
    analyze_spectrum,
    fsa_basis,
    participation_ratio,
    plot_official_pr2_scaling,
)


def test_fsa_basis_is_orthonormal_and_resolves_hamming_distance():
    length = 10
    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    z2 = density_wave_state(length, 2)

    vectors, beta = fsa_basis(hamiltonian, basis, z2, length)

    np.testing.assert_allclose(vectors.conj() @ vectors.T, np.eye(length + 1))
    assert len(beta) == length
    for distance, vector in enumerate(vectors):
        support = np.flatnonzero(np.abs(vector) > 1e-12)
        assert all((int(basis[i]) ^ z2).bit_count() == distance for i in support)


def test_overlap_weights_and_participation_ratios_are_normalized():
    result = analyze_spectrum(length=10)

    assert len(result["energies"]) == len(result["overlap_z2"])
    np.testing.assert_allclose(np.sum(result["overlap_z2"]), 0.5, atol=1e-12)
    assert np.all(result["participation_ratio"] > 0.0)
    assert np.all(result["participation_ratio"] <= 1.0)
    assert len(result["fsa_energies"]) == 6
    np.testing.assert_allclose(np.sum(result["fsa_overlap_z2"]), 0.5, atol=1e-12)
    assert result["exact_shell_amplitudes"].shape == (6, len(result["energies"]))
    assert result["fsa_eigenvectors"].shape == (6, 6)


def test_participation_ratio_distinguishes_product_and_uniform_states():
    product = np.array([1.0, 0.0, 0.0, 0.0])
    uniform = np.ones(4) / 2

    assert participation_ratio(product) == 1.0
    assert participation_ratio(uniform) == 0.25


def test_official_pr2_available_sizes_are_unconnected_markers():
    figure, axis = plt.subplots()
    scaling = {
        "length": np.array([26, 28, 30, 32]),
        "other": np.array([0.1, np.nan, np.nan, 0.01]),
        "special": np.array([0.2, np.nan, np.nan, 0.02]),
        "available": np.array([True, False, False, True]),
    }

    lines = plot_official_pr2_scaling(axis, scaling)

    assert len(lines) == 2
    assert all(line.get_linestyle() == "None" for line in lines)
    for line in lines:
        np.testing.assert_array_equal(line.get_xdata(), [26, 32])
    plt.close(figure)


def _independent_result(root, length=10):
    output = root / f"L{length}"
    assert server_main(
        [
            "--stage",
            "all",
            "--length",
            str(length),
            "--output-dir",
            str(output),
            "--declared-memory",
            "1G",
            "--chunk-columns",
            "4",
        ]
    ) == 0
    with h5py.File(output / "eigensystem.h5", "r") as handle:
        energies = handle["eigensystem/energies"][()]
    with h5py.File(output / "observables.h5", "r") as handle:
        observables = {
            name: handle[f"observables/{name}"][()]
            for name in (
                "overlap_z2",
                "participation_ratio",
                "exact_shell_amplitudes",
                "fsa_hamiltonian_sector",
            )
        }
    fsa_energies, fsa_vectors = np.linalg.eigh(
        observables["fsa_hamiltonian_sector"]
    )
    selector = select_fig3_pr2_states(
        energies=energies,
        exact_shell_amplitudes=observables["exact_shell_amplitudes"],
        fsa_eigenvectors=fsa_vectors,
    )
    other_mean, special_mean = pr2_fsa_averages(
        energies=energies,
        pr2=observables["participation_ratio"],
        exact_shell_amplitudes=observables["exact_shell_amplitudes"],
        fsa_eigenvectors=fsa_vectors,
    )
    return output, {
        "energies": energies,
        **observables,
        "fsa_energies": fsa_energies,
        "fsa_vectors": fsa_vectors,
        "selector": selector,
        "other_mean": other_mean,
        "special_mean": special_mean,
    }


def test_independent_renderer_uses_only_validated_artifacts_for_generated_series(
    tmp_path, monkeypatch
):
    independent_root = tmp_path / "independent"
    _output, independent = _independent_result(independent_root)

    official_root = tmp_path / "official"
    official_root.mkdir()
    for name in (
        "energy_eigenvalues.zip",
        "forward-scattering.zip",
        "participation_ratios.zip",
    ):
        (official_root / name).touch()
    with ZipFile(official_root / "overlaps_with_Neel_state.zip", "w") as archive:
        archive.writestr("oneel_periodic_N10_k0_p0.dat", "synthetic")
    with ZipFile(official_root / "eigendecomposition.zip", "w") as archive:
        archive.writestr("eigs_periodic_N10_k0_p0.h5", b"synthetic")
    official_energies = independent["energies"] + 9.0
    official_overlap = independent["overlap_z2"] + 7.0
    monkeypatch.setattr(
        fig3,
        "load_fig3_overlap",
        lambda _root, *, length: (official_energies, official_overlap),
    )
    monkeypatch.setattr(
        fig3,
        "load_fig3_pr2_scaling",
        lambda _root: {
            "length": np.array([10]),
            "other": np.array([0.91]),
            "special": np.array([0.92]),
            "available": np.array([True]),
        },
    )
    original_getitem = h5py.Dataset.__getitem__

    def reject_full_eigenvector_read(dataset, key):
        if dataset.name.endswith("/vectors"):
            full = key == slice(None)
            if isinstance(key, tuple):
                full = all(part == slice(None) for part in key)
            if full:
                raise AssertionError("renderer must not materialize eigenvectors")
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_full_eigenvector_read)

    figure_path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=official_root,
    )

    assert figure_path.name == "fig3_independent_L10.png"
    with np.load(figure_path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        np.testing.assert_array_equal(
            sidecar["panel_a_L10_energies"], independent["energies"]
        )
        np.testing.assert_array_equal(
            sidecar["panel_a_L10_overlap_z2"], independent["overlap_z2"]
        )
        assert not np.array_equal(
            sidecar["panel_a_L10_energies"], official_energies
        )
        np.testing.assert_array_equal(
            sidecar["official_L10_energies"], official_energies
        )
        assert sidecar["panel_a_L10_source"].item() == "independent-ed"
        assert sidecar["panel_d_source"].item() == "independent-ed"
        assert sidecar["official_L10_source"].item() == "official-doi"
        sidecar_generation = sidecar["generation_id"].item()
        np.testing.assert_allclose(
            sidecar["panel_d_other"], [independent["other_mean"]]
        )
        np.testing.assert_allclose(
            sidecar["panel_d_special"], [independent["special_mean"]]
        )
        selector = independent["selector"]
        panel_indices = (
            int(selector["sorted_tower"][0]),
            int(
                selector["special"][
                    np.argmin(
                        np.abs(independent["energies"][selector["special"]])
                    )
                ]
            ),
        )
        for panel, exact_index in zip(("b", "c"), panel_indices):
            fsa_index = int(
                np.flatnonzero(selector["tower"] == exact_index)[0]
            )
            np.testing.assert_allclose(
                sidecar[f"panel_{panel}_L10_exact_weights"],
                np.abs(
                    independent["exact_shell_amplitudes"][:, exact_index]
                )
                ** 2,
            )
            np.testing.assert_allclose(
                sidecar[f"panel_{panel}_L10_fsa_weights"],
                np.abs(independent["fsa_vectors"][:, fsa_index]) ** 2,
            )

    metrics = json.loads(figure_path.with_suffix(".json").read_text())
    assert metrics["generation_id"] == sidecar_generation
    assert metrics["generation_assets"]["png_sha256"] == hashlib.sha256(
        figure_path.read_bytes()
    ).hexdigest()
    assert metrics["generation_assets"]["npz_sha256"] == hashlib.sha256(
        figure_path.with_suffix(".npz").read_bytes()
    ).hexdigest()
    assert metrics["source"] == "independent-ed"
    assert metrics["lengths"]["10"]["overlap_sum"] == pytest.approx(
        float(np.sum(independent["overlap_z2"]))
    )
    assert metrics["lengths"]["10"]["pr2"]["all_values_sha256"] == hashlib.sha256(
        independent["participation_ratio"].tobytes()
    ).hexdigest()
    assert all(
        series["source"] == "independent-ed"
        for name, series in metrics["series"].items()
        if not name.startswith("official_")
    )
    assert all(
        series["source"] == "official-doi"
        for name, series in metrics["series"].items()
        if name.startswith("official_")
    )
    mismatch = metrics["lengths"]["10"]["official_mismatch"]
    assert mismatch["energies_max_abs"] == pytest.approx(9.0)
    assert mismatch["overlap_max_abs"] == pytest.approx(7.0)
    fsa = metrics["lengths"]["10"]["fsa"]
    assert fsa["shell_dimensions"] == [6, 14]
    assert fsa["normalization_convention"] == "unit-norm projected shells"
    assert fsa["sign_convention"] == "raw persisted real shell amplitudes"
    assert len(fsa["match_strengths"]) == 6
    assert all(value > 0 for value in fsa["match_strengths"])
    assert {
        "basis.npz",
        "hamiltonian.csr.npz",
        "eigensystem.h5",
        "observables.h5",
    }.issubset(metrics["lengths"]["10"]["source_hashes"])
    assert {
        state["selection_role"] for state in metrics["selected_panel_states"]
    } == {"tower-ground", "interior-special"}
    assert "near E=0" not in json.dumps(metrics)


def test_missing_independent_size_is_recorded_without_official_substitution(tmp_path):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root, length=10)
    _independent_result(independent_root, length=14)

    figure_path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=None,
    )

    with np.load(figure_path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        np.testing.assert_array_equal(sidecar["panel_d_lengths"], [10, 14])
        assert not any(name.startswith("official_") for name in sidecar.files)
    metrics = json.loads(figure_path.with_suffix(".json").read_text())
    assert metrics["missing_lengths_within_independent_range"] == [12]
    assert metrics["available_independent_lengths"] == [10, 14]


def test_official_overlap_overlay_is_skipped_when_requested_member_is_absent(
    tmp_path, monkeypatch
):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root, length=10)
    official_root = tmp_path / "official"
    official_root.mkdir()
    with ZipFile(official_root / "overlaps_with_Neel_state.zip", "w") as archive:
        archive.writestr("oneel_periodic_N22_k0_p0.dat", "0.0\n")
    with ZipFile(official_root / "eigendecomposition.zip", "w") as archive:
        archive.writestr("unrelated", b"")
    monkeypatch.setattr(
        fig3,
        "load_fig3_overlap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("absent official member must not be loaded")
        ),
    )

    path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=official_root,
    )

    with np.load(path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        assert not any(name.startswith("official_L10") for name in sidecar.files)


def test_renderer_rejects_stale_current_execution_fingerprint(tmp_path, monkeypatch):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root)
    changed = json.loads(json.dumps(server.build_execution_fingerprint()))
    changed["sources"]["turner2018_fig3.py"] = "f" * 64
    monkeypatch.setattr(server, "build_execution_fingerprint", lambda: changed)

    with pytest.raises(RuntimeError, match="execution fingerprint"):
        fig3.render_independent_fig3(
            independent_root,
            output_dir=tmp_path / "figure",
            official_data_dir=None,
        )


def test_renderer_consumes_the_exact_validated_hdf5_handles(tmp_path, monkeypatch):
    independent_root = tmp_path / "independent"
    output, independent = _independent_result(independent_root)
    replacement = tmp_path / "replacement-observables.h5"
    shutil.copyfile(output / "observables.h5", replacement)
    with h5py.File(replacement, "r+") as handle:
        overlap = handle["observables/overlap_z2"]
        overlap[:] = np.roll(overlap[()], 1)

    original_require_stage = server.require_stage
    swapped = False

    def swap_after_validation(directory, stage, validation_cache=None):
        nonlocal swapped
        result = original_require_stage(directory, stage, validation_cache)
        if stage == "validate" and not swapped:
            os.replace(replacement, output / "observables.h5")
            swapped = True
        return result

    monkeypatch.setattr(server, "require_stage", swap_after_validation)
    path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=None,
    )

    with np.load(path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        np.testing.assert_array_equal(
            sidecar["panel_a_L10_overlap_z2"],
            independent["overlap_z2"],
        )


def test_generation_publish_failure_restores_all_previous_files(
    tmp_path, monkeypatch
):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root)
    figure_dir = tmp_path / "figure"
    path = fig3.render_independent_fig3(
        independent_root,
        output_dir=figure_dir,
        official_data_dir=None,
    )
    targets = (path, path.with_suffix(".npz"), path.with_suffix(".json"))
    before = {target: target.read_bytes() for target in targets}
    original_replace = fig3.os.replace

    def fail_json_publish(source, destination):
        if Path(destination) == path.with_suffix(".json"):
            raise OSError("injected JSON publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(fig3.os, "replace", fail_json_publish)
    with pytest.raises(OSError, match="JSON publish failure"):
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )

    assert {target: target.read_bytes() for target in targets} == before
    assert not list(figure_dir.glob("*.partial*"))
    assert not list(figure_dir.glob("*.backup"))


def test_first_generation_publish_failure_leaves_no_generation(tmp_path, monkeypatch):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root)
    figure_dir = tmp_path / "figure"
    expected = figure_dir / "fig3_independent_L10.png"
    original_replace = fig3.os.replace

    def fail_json_publish(source, destination):
        if Path(destination) == expected.with_suffix(".json"):
            raise OSError("injected first JSON publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(fig3.os, "replace", fail_json_publish)
    with pytest.raises(OSError, match="first JSON publish failure"):
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )

    assert not expected.exists()
    assert not expected.with_suffix(".npz").exists()
    assert not expected.with_suffix(".json").exists()
    assert not list(figure_dir.glob("*.partial*"))
    assert not list(figure_dir.glob("*.backup"))


def test_generation_backup_failure_preserves_previous_files(tmp_path, monkeypatch):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root)
    figure_dir = tmp_path / "figure"
    path = fig3.render_independent_fig3(
        independent_root,
        output_dir=figure_dir,
        official_data_dir=None,
    )
    targets = (path, path.with_suffix(".npz"), path.with_suffix(".json"))
    before = {target: target.read_bytes() for target in targets}
    original_link = fig3.os.link
    calls = 0

    def fail_second_backup(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected backup failure")
        return original_link(source, destination)

    monkeypatch.setattr(fig3.os, "link", fail_second_backup)
    with pytest.raises(OSError, match="backup failure"):
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )

    assert {target: target.read_bytes() for target in targets} == before
    assert not list(figure_dir.glob("*.partial*"))
    assert not list(figure_dir.glob("*.backup"))


def test_length_and_independent_results_root_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        fig3.build_parser().parse_args(
            ["--length", "10", "--independent-results-root", "results"]
        )


def test_legacy_small_l_renderer_still_writes_analyze_spectrum_arrays(tmp_path):
    expected = analyze_spectrum(10)

    path = fig3.run_figure(10, output_dir=tmp_path, official_data_dir=None)

    assert path.name == "fig3_L10.png"
    with np.load(tmp_path / "fig3_L10.npz", allow_pickle=False) as persisted:
        np.testing.assert_allclose(persisted["energies"], expected["energies"])
        np.testing.assert_allclose(persisted["overlap_z2"], expected["overlap_z2"])
