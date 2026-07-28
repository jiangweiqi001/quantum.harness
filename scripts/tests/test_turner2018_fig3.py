import hashlib
import json
import os
from pathlib import Path
import shutil
from dataclasses import FrozenInstanceError, is_dataclass
from zipfile import ZipFile

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pxp_ed import (
    constrained_basis,
    density_wave_state,
    pxp_hamiltonian,
    symmetry_basis_k0_inversion_even,
)
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


def test_unfold_folded_shell_amplitudes_preserves_norm_and_reflection_parity():
    folded = np.asarray([np.sqrt(2.0), 2.0 * np.sqrt(2.0), 3.0])

    even = fig3.unfold_folded_shell_amplitudes(folded, length=4)
    odd = fig3.unfold_folded_shell_amplitudes(
        folded, length=4, reflection_parity=-1
    )

    np.testing.assert_allclose(even, [1.0, 2.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(odd, [1.0, 2.0, 3.0, -2.0, -1.0])
    assert np.vdot(even, even) == pytest.approx(np.vdot(folded, folded))
    assert np.vdot(odd, odd) == pytest.approx(np.vdot(folded, folded))


def test_symmetry_unfold_matches_direct_small_l_selected_shell_probabilities():
    length = 10
    result = analyze_spectrum(length)
    fsa_vectors = np.asarray(result["fsa_eigenvectors"]).real
    tower, _resolutions = fig3.match_independent_fsa_tower(
        result["energies"],
        result["exact_shell_amplitudes"],
        fsa_vectors,
        fsa_energies=result["fsa_energies"],
    )
    selections = fig3.select_fig3_shell_panel_states(
        length=length,
        energies=result["energies"],
        exact_shell_amplitudes=result["exact_shell_amplitudes"],
        fsa_hamiltonian_sector=np.asarray(
            result["fsa_hamiltonian_sector"]
        ).real,
        matched_tower={"tower": tower},
    )

    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(basis, length)
    reduced = (transform.T @ hamiltonian @ transform).toarray()
    energies, eigenvectors = np.linalg.eigh(reduced)
    full_shells, _beta = fsa_basis(
        hamiltonian, basis, density_wave_state(length, 2), length
    )
    direct_shells = np.asarray(transform.T @ full_shells.T).T
    scale = length / 2.0
    for selection in selections:
        exact_index = selection["exact_index"]
        assert energies[exact_index] == pytest.approx(selection["exact_energy"])
        direct = direct_shells @ eigenvectors[:, exact_index]
        unfolded = fig3.unfold_folded_shell_amplitudes(
            result["exact_shell_amplitudes"][:, exact_index], length
        )
        np.testing.assert_allclose(
            scale * np.abs(unfolded) ** 2,
            scale * np.abs(direct) ** 2,
            atol=0.025,
            rtol=0.0,
        )
        plotted = scale * np.abs(unfolded) ** 2
        np.testing.assert_allclose(plotted, plotted[::-1], atol=0.0, rtol=0.0)
        if selection["panel"] == "b":
            assert abs(int(np.argmax(plotted)) - length // 2) <= 1
        else:
            local_turns = np.count_nonzero(
                np.diff(np.sign(np.diff(plotted))) != 0
            )
            assert local_turns >= length // 2


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


def _synthetic_shell_panel_inputs(length=6):
    shell_count = length // 2 + 1
    exact_count = max(8, shell_count + 4)
    energies = np.linspace(-9.0, 8.0, exact_count)
    energies[1:6] = [-4.0, -1.0, 0.0, 0.0, 0.4]
    energies.sort()
    fsa_diagonal = np.linspace(-3.0, 3.0, shell_count)
    permutation = np.roll(np.arange(shell_count), 1)
    fsa_hamiltonian = np.diag(fsa_diagonal[permutation])
    _fsa_energies, fsa_vectors = np.linalg.eigh(fsa_hamiltonian)
    tower = np.arange(shell_count, dtype=np.int64) + 1
    exact_shell_amplitudes = np.zeros((shell_count, exact_count))
    exact_shell_amplitudes[:, tower] = 0.8 * fsa_vectors
    return {
        "length": length,
        "energies": energies,
        "exact_shell_amplitudes": exact_shell_amplitudes,
        "fsa_hamiltonian_sector": fsa_hamiltonian,
        "matched_tower": {
            "tower": tower,
            "sorted_tower": tower[np.argsort(energies[tower], kind="stable")],
        },
    }


def test_shell_panel_selector_uses_lowest_and_negative_adjacent_matched_states():
    inputs = _synthetic_shell_panel_inputs()

    panel_b, panel_c = fig3.select_fig3_shell_panel_states(**inputs)

    assert panel_b["role"] == "lowest-matched-scar"
    assert panel_c["role"] == "negative-adjacent-to-zero"
    assert panel_b["exact_index"] == 1
    assert panel_c["exact_index"] == 2
    assert inputs["energies"][panel_c["exact_index"]] < -panel_c["zero_tolerance"]
    assert panel_b["full_fsa_shell_count"] == inputs["length"] + 1
    assert panel_b["plotted_folded_shell_count"] == inputs["length"] // 2 + 1
    assert panel_b["folding"] == (
        "normalized inversion-even amplitudes unfolded as "
        "(|n>+|L-n>)/sqrt(2), with n=L/2 unchanged"
    )
    assert panel_b["displayed_full_shell_count"] == inputs["length"] + 1
    assert panel_b["vertical_scale_factor"] == inputs["length"] / 2
    assert panel_b["exact_weight_sum"] == pytest.approx(0.64)
    assert panel_b["fsa_weight_sum"] == pytest.approx(1.0)
    assert panel_b["match_strength"] == pytest.approx(0.64)
    assert panel_c["exact_energy"] == pytest.approx(-1.0)
    assert panel_b["fsa_index"] != panel_b["exact_index"]
    assert panel_b["fsa_gap_tolerance"] == pytest.approx(1e-10)
    assert panel_b["fsa_nearest_gap"] > panel_b["fsa_gap_tolerance"]
    assert panel_c["fsa_nearest_gap"] > panel_c["fsa_gap_tolerance"]
    np.testing.assert_array_equal(
        panel_b["shell"], np.arange(inputs["length"] + 1)
    )


def test_shell_panel_selector_excludes_central_fsa_state_for_panel_c():
    energies = np.asarray([-4.0, -2.0, -0.1, 2.0, 4.0])
    amplitudes = np.eye(5)
    panel_b, panel_c = fig3.select_fig3_shell_panel_states(
        length=8,
        energies=energies,
        exact_shell_amplitudes=amplitudes,
        fsa_hamiltonian_sector=np.diag([-4.0, -2.0, 0.0, 2.0, 4.0]),
        matched_tower={"tower": np.arange(5)},
    )

    assert panel_b["fsa_index"] == 0
    assert panel_c["fsa_index"] == 1
    assert panel_c["exact_energy"] == -2.0


def test_shell_panel_selector_result_is_structurally_immutable():
    panel_b, _panel_c = fig3.select_fig3_shell_panel_states(
        **_synthetic_shell_panel_inputs()
    )

    assert is_dataclass(panel_b)
    with pytest.raises(FrozenInstanceError):
        panel_b.role = "changed"
    for field in ("shell", "exact_weights", "fsa_weights"):
        stored = getattr(panel_b, field)
        assert isinstance(stored, tuple)
        assert not hasattr(stored, "setflags")
        with pytest.raises(TypeError):
            stored[0] = 99
        local = np.asarray(stored)
        local.setflags(write=True)
        local[0] = 99
        assert getattr(panel_b, field) == stored
    metadata = panel_b.to_metadata_dict()
    metadata["role"] = "changed-at-serialization-boundary"
    assert panel_b.role == "lowest-matched-scar"


def test_shell_panel_selector_rejects_rotated_degenerate_selected_fsa_basis(
    monkeypatch,
):
    inputs = _synthetic_shell_panel_inputs()
    original_eigh = np.linalg.eigh
    _energies, baseline_vectors = original_eigh(inputs["fsa_hamiltonian_sector"])
    degenerate_energies = np.asarray([-1.0, -1.0, 1.0, 2.0])

    for angle in (0.0, np.pi / 7.0):
        rotation = np.eye(4)
        rotation[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        rotated_vectors = baseline_vectors @ rotation
        monkeypatch.setattr(
            fig3.np.linalg,
            "eigh",
            lambda _matrix, e=degenerate_energies, v=rotated_vectors: (e, v),
        )
        with pytest.raises(ValueError, match="degenerate.*FSA eigenvalue"):
            fig3.select_fig3_shell_panel_states(**inputs)


@pytest.mark.parametrize(("length", "plotted"), [(20, 11), (32, 17)])
def test_shell_panel_selector_records_folded_shell_count(length, plotted):
    inputs = _synthetic_shell_panel_inputs(length)

    panel_b, panel_c = fig3.select_fig3_shell_panel_states(**inputs)

    assert panel_b["full_fsa_shell_count"] == length + 1
    assert panel_b["plotted_folded_shell_count"] == plotted
    assert panel_c["plotted_folded_shell_count"] == plotted
    assert panel_b["displayed_full_shell_count"] == length + 1
    assert len(panel_b["shell"]) == length + 1


def test_shell_panel_selector_excludes_zero_modes_and_requires_negative_candidate():
    inputs = _synthetic_shell_panel_inputs()
    inputs["fsa_hamiltonian_sector"] = np.diag([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="negative.*adjacent"):
        fig3.select_fig3_shell_panel_states(**inputs)


def test_shell_panel_selector_rejects_duplicate_or_stale_matches():
    inputs = _synthetic_shell_panel_inputs()
    inputs["matched_tower"]["tower"][1] = inputs["matched_tower"]["tower"][0]

    with pytest.raises(ValueError, match="one-to-one|duplicate"):
        fig3.select_fig3_shell_panel_states(**inputs)


def test_shell_panel_selector_rejects_nonfinite_weights_and_wrong_shell_count():
    inputs = _synthetic_shell_panel_inputs()
    inputs["exact_shell_amplitudes"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fig3.select_fig3_shell_panel_states(**inputs)

    inputs = _synthetic_shell_panel_inputs()
    inputs["exact_shell_amplitudes"] = inputs["exact_shell_amplitudes"][:-1]
    with pytest.raises(ValueError, match="shell count"):
        fig3.select_fig3_shell_panel_states(**inputs)


def test_shell_panel_selector_rejects_ambiguous_projection_matches():
    inputs = _synthetic_shell_panel_inputs()
    exact_index = int(inputs["matched_tower"]["tower"][0])
    inputs["exact_shell_amplitudes"][:, exact_index + 4] = (
        inputs["exact_shell_amplitudes"][:, exact_index]
    )

    with pytest.raises(ValueError, match="tied|ambiguous"):
        fig3.select_fig3_shell_panel_states(**inputs)


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


def test_particle_hole_tie_resolution_selects_negative_representative():
    energies = np.array([-2.0, -0.1, 0.1, 2.0])
    amplitudes = np.zeros((3, 4))
    amplitudes[0, 0] = 1.0
    amplitudes[1, 1:3] = np.sqrt(0.4)
    amplitudes[2, 3] = 1.0

    tower, resolutions = fig3.match_independent_fsa_tower(
        energies,
        amplitudes,
        np.eye(3),
        fsa_energies=np.array([-2.0, 0.0, 2.0]),
    )

    np.testing.assert_array_equal(tower, [0, 1, 3])
    assert resolutions == [
        {
            "fsa_index": 1,
            "policy": "particle-hole-pair-negative-representative",
            "candidate_indices": [1, 2],
            "candidate_energies": [-0.1, 0.1],
        }
    ]


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
    rendered_panels = {}
    original_write_png = fig3._write_png_partial

    def inspect_shell_panels(partial, figure):
        rendered_panels["label_positions"] = {
            label: next(
                text.get_position()
                for text in panel.texts
                if text.get_text() == f"({label})"
            )
            for label, panel in zip(("a", "b", "c", "d"), figure.axes)
        }
        for label, panel in zip(("b", "c"), figure.axes[1:3]):
            rendered_panels[label] = {
                "title": panel.get_title(),
                "xlabel": panel.get_xlabel(),
                "ylabel": panel.get_ylabel(),
                "xlim": panel.get_xlim(),
                "xticks": panel.get_xticks(),
                "ylim": panel.get_ylim(),
                "yticks": panel.get_yticks(),
                "panel_label_position": next(
                    text.get_position()
                    for text in panel.texts
                    if text.get_text() == f"({label})"
                ),
                "lines": [
                    {
                        "color": line.get_color(),
                        "marker": line.get_marker(),
                        "linestyle": line.get_linestyle(),
                        "label": line.get_label(),
                        "x": np.asarray(line.get_xdata()),
                        "y": np.asarray(line.get_ydata()),
                    }
                    for line in panel.lines
                ],
            }
        original_write_png(partial, figure)

    monkeypatch.setattr(fig3, "_write_png_partial", inspect_shell_panels)

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
        np.testing.assert_array_equal(
            sidecar["selection_L10_exact_shell_amplitudes"],
            independent["exact_shell_amplitudes"],
        )
        np.testing.assert_array_equal(
            sidecar["selection_L10_fsa_hamiltonian_sector"],
            independent["fsa_hamiltonian_sector"],
        )
        np.testing.assert_array_equal(
            sidecar["selection_L10_match_exact_indices"],
            independent["selector"]["tower"],
        )
        tower = np.asarray(independent["selector"]["tower"])
        expected_b_fsa = min(
            range(len(tower)),
            key=lambda index: (
                independent["energies"][tower[index]],
                int(tower[index]),
            ),
        )
        negative_fsa = [
            index
            for index in range(len(tower))
            if independent["fsa_energies"][index] < -1e-10
        ]
        expected_c_fsa = min(
            negative_fsa,
            key=lambda index: (
                abs(independent["fsa_energies"][index]),
                int(tower[index]),
            ),
        )
        for panel, fsa_index in zip(("b", "c"), (expected_b_fsa, expected_c_fsa)):
            exact_index = int(tower[fsa_index])
            np.testing.assert_allclose(
                sidecar[f"panel_{panel}_L10_exact_weights"],
                5.0
                * np.abs(
                    fig3.unfold_folded_shell_amplitudes(
                        independent["exact_shell_amplitudes"][:, exact_index],
                        10,
                    )
                )
                ** 2,
            )
            np.testing.assert_allclose(
                sidecar[f"panel_{panel}_L10_fsa_weights"],
                5.0
                * np.abs(
                    fig3.unfold_folded_shell_amplitudes(
                        independent["fsa_vectors"][:, fsa_index], 10
                    )
                )
                ** 2,
            )
            np.testing.assert_array_equal(
                sidecar[f"panel_{panel}_L10_shell"], np.arange(11)
            )
            rendered = rendered_panels[panel]
            assert rendered["xlabel"] == "n"
            assert rendered["ylabel"] == r"$(L/2)|\langle n|\psi\rangle|^2$"
            assert rendered["xlim"] == pytest.approx((0.0, 10.0))
            np.testing.assert_array_equal(rendered["xticks"], [0.0, 10.0])
            expected_ylim = (0.0, 4.2) if panel == "b" else (0.0, 2.1)
            expected_yticks = (
                [0.0, 2.0, 4.0] if panel == "b" else [0.0, 1.0, 2.0]
            )
            assert rendered["ylim"] == pytest.approx(expected_ylim)
            np.testing.assert_array_equal(rendered["yticks"], expected_yticks)
            assert len(rendered["lines"]) == 2
            exact_line, fsa_line = rendered["lines"]
            assert exact_line["color"] == "black"
            assert exact_line["marker"] == "o"
            assert exact_line["linestyle"] == "-"
            assert exact_line["label"] == "exact"
            assert fsa_line["color"] == "red"
            assert fsa_line["marker"] == "o"
            assert fsa_line["linestyle"] == "-"
            assert fsa_line["label"] == "FSA"
            assert len(exact_line["x"]) == len(fsa_line["x"]) == 11
            np.testing.assert_array_equal(exact_line["x"], np.arange(11))
        assert rendered_panels["b"]["title"] == ""
        assert rendered_panels["c"]["title"] == ""
        assert rendered_panels["label_positions"] == {
            label: (0.02, 1.04) for label in ("a", "b", "c", "d")
        }

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
    assert fsa["full_fsa_shell_count"] == 11
    assert fsa["plotted_folded_shell_count"] == 6
    assert fsa["displayed_full_shell_count"] == 11
    assert fsa["folding"] == (
        "normalized inversion-even amplitudes unfolded as "
        "(|n>+|L-n>)/sqrt(2), with n=L/2 unchanged"
    )
    assert fsa["normalization_convention"] == (
        "unit-norm folded projections; paired amplitudes divided by sqrt(2), "
        "center unchanged; plotted by (L/2)|amplitude|^2"
    )
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
        state["role"] for state in metrics["selected_panel_states"]
    } == {"lowest-matched-scar", "negative-adjacent-to-zero"}
    for state in metrics["selected_panel_states"]:
        assert state["full_fsa_shell_count"] == 11
        assert state["plotted_folded_shell_count"] == 6
        assert state["exact_weight_sum"] > 0
        assert state["fsa_weight_sum"] == pytest.approx(1.0)
        assert state["zero_tolerance"] == pytest.approx(1e-10)
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


def test_independent_discovery_ignores_completed_figures_manifest(tmp_path):
    independent_root = tmp_path / "independent"
    output, independent = _independent_result(independent_root, length=10)
    figures = output / "figures"
    figures.mkdir()
    (figures / "manifest.json").write_text('{"not": "an ED plan"}')

    loaded = fig3.load_independent_results(independent_root)

    assert set(loaded) == {10}
    np.testing.assert_array_equal(loaded[10]["energies"], independent["energies"])


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


def test_shared_publication_preserves_backups_when_restore_link_fails(
    tmp_path, monkeypatch
):
    targets = tuple(tmp_path / f"asset-{index}" for index in range(3))
    partials = {
        target: target.with_name(target.name + ".partial") for target in targets
    }
    for index, target in enumerate(targets):
        target.write_bytes(f"old-{index}".encode())
        partials[target].write_bytes(f"new-{index}".encode())
    original_replace = fig3.os.replace
    original_link = fig3.os.link

    def fail_second_publish(source, destination):
        if Path(source).suffix == ".partial" and Path(destination) == targets[1]:
            raise OSError("injected publish failure")
        return original_replace(source, destination)

    def fail_first_restore(source, destination):
        if str(source).endswith(".backup") and Path(destination) == targets[0]:
            raise OSError("injected restore link failure")
        return original_link(source, destination)

    monkeypatch.setattr(fig3.os, "replace", fail_second_publish)
    monkeypatch.setattr(fig3.os, "link", fail_first_restore)

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        fig3._publish_generation(partials)

    backup = targets[0].with_name(targets[0].name + ".backup")
    assert backup.read_bytes() == b"old-0"


def test_shared_publication_preserves_backups_when_rollback_unlink_fails(
    tmp_path, monkeypatch
):
    targets = tuple(tmp_path / f"asset-{index}" for index in range(3))
    partials = {
        target: target.with_name(target.name + ".partial") for target in targets
    }
    for index, target in enumerate(targets):
        target.write_bytes(f"old-{index}".encode())
        partials[target].write_bytes(f"new-{index}".encode())
    original_replace = fig3.os.replace
    original_unlink = fig3._unlink_durable

    def fail_second_publish(source, destination):
        if Path(source).suffix == ".partial" and Path(destination) == targets[1]:
            raise OSError("injected publish failure")
        return original_replace(source, destination)

    def fail_rollback_unlink(path):
        if Path(path) == targets[0]:
            raise OSError("injected rollback unlink failure")
        return original_unlink(path)

    monkeypatch.setattr(fig3.os, "replace", fail_second_publish)
    monkeypatch.setattr(fig3, "_unlink_durable", fail_rollback_unlink)

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        fig3._publish_generation(partials)

    backups = [target.with_name(target.name + ".backup") for target in targets]
    assert all(backup.is_file() for backup in backups)


def test_shared_publication_preserves_backups_when_restore_fsync_fails(
    tmp_path, monkeypatch
):
    targets = tuple(tmp_path / f"asset-{index}" for index in range(3))
    partials = {
        target: target.with_name(target.name + ".partial") for target in targets
    }
    for index, target in enumerate(targets):
        target.write_bytes(f"old-{index}".encode())
        partials[target].write_bytes(f"new-{index}".encode())
    original_replace = fig3.os.replace
    original_fsync_parent = fig3._fsync_parent
    target_zero_fsyncs = 0

    def fail_second_publish(source, destination):
        if Path(source).suffix == ".partial" and Path(destination) == targets[1]:
            raise OSError("injected publish failure")
        return original_replace(source, destination)

    def fail_restore_fsync(path):
        nonlocal target_zero_fsyncs
        if Path(path) == targets[0]:
            target_zero_fsyncs += 1
            if target_zero_fsyncs == 3:
                raise OSError("injected restore fsync failure")
        return original_fsync_parent(path)

    monkeypatch.setattr(fig3.os, "replace", fail_second_publish)
    monkeypatch.setattr(fig3, "_fsync_parent", fail_restore_fsync)

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        fig3._publish_generation(partials)

    assert targets[0].read_bytes() == b"old-0"
    assert targets[0].with_name("asset-0.backup").read_bytes() == b"old-0"


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_shared_publication_records_recovery_state_for_each_cleanup_failure(
    tmp_path, monkeypatch, failure_index
):
    targets = tuple(tmp_path / f"asset-{index}" for index in range(3))
    partials = {
        target: target.with_name(target.name + ".partial") for target in targets
    }
    for index, target in enumerate(targets):
        target.write_bytes(f"old-{index}".encode())
        partials[target].write_bytes(f"new-{index}".encode())
    original_unlink = fig3._unlink_durable
    failed_backup = f"asset-{failure_index}.backup"

    def fail_existing_backup_cleanup(path):
        path = Path(path)
        if path.name == failed_backup and path.is_file():
            raise OSError("injected backup cleanup failure")
        return original_unlink(path)

    monkeypatch.setattr(fig3, "_unlink_durable", fail_existing_backup_cleanup)

    with pytest.raises(RuntimeError, match="published backup cleanup"):
        fig3._publish_generation(partials)

    assert all(
        target.read_bytes() == f"new-{index}".encode()
        for index, target in enumerate(targets)
    )
    recovery_paths = [
        path for path in tmp_path.iterdir() if path.name.endswith(".recovery.json")
    ]
    assert len(recovery_paths) == 1
    recovery = json.loads(recovery_paths[0].read_text())
    assert recovery["state"] == "backup-cleanup-failed"
    assert all(
        item["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        for item, target in zip(recovery["current_generation"], targets)
    )
    backup_state = {
        Path(item["path"]).name: item["present"]
        for item in recovery["backups"]
    }
    assert backup_state == {
        f"asset-{index}.backup": index >= failure_index for index in range(3)
    }
    assert Path(
        next(item["path"] for item in recovery["backups"] if item["present"])
    ).is_file()


@pytest.mark.parametrize("with_prior_generation", [False, True])
@pytest.mark.parametrize("artifact", ["png", "npz", "json"])
@pytest.mark.parametrize("failure_phase", ["creation", "write", "fsync"])
def test_partial_failures_are_owned_and_cleaned(
    tmp_path,
    monkeypatch,
    with_prior_generation,
    artifact,
    failure_phase,
):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root)
    figure_dir = tmp_path / "figure"
    png = figure_dir / "fig3_independent_L10.png"
    targets = {
        "png": png,
        "npz": png.with_suffix(".npz"),
        "json": png.with_suffix(".json"),
    }
    if with_prior_generation:
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )
        before = {target: target.read_bytes() for target in targets.values()}
    else:
        before = {}
    target = targets[artifact]
    partial = target.with_name(target.name + ".partial")

    def injected_failure(actual_partial, _payload):
        assert actual_partial == partial
        if failure_phase != "creation":
            with actual_partial.open("wb") as handle:
                handle.write(b"incomplete")
                handle.flush()
            if failure_phase == "fsync":
                raise OSError(f"injected {artifact} fsync failure")
        raise OSError(f"injected {artifact} {failure_phase} failure")

    monkeypatch.setattr(
        fig3,
        f"_write_{artifact}_partial",
        injected_failure,
        raising=False,
    )
    with pytest.raises(OSError, match=f"{artifact} {failure_phase} failure"):
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )

    if with_prior_generation:
        assert {
            output: output.read_bytes() for output in targets.values()
        } == before
    else:
        assert not any(output.exists() for output in targets.values())
    assert not list(figure_dir.glob("*.partial*"))
    assert not list(figure_dir.glob("*.backup"))


@pytest.mark.parametrize("with_prior_generation", [False, True])
@pytest.mark.parametrize("artifact", ["png", "npz", "json"])
def test_real_partial_fsync_failure_is_cleaned(
    tmp_path, monkeypatch, with_prior_generation, artifact
):
    independent_root = tmp_path / "independent"
    _independent_result(independent_root)
    figure_dir = tmp_path / "figure"
    png = figure_dir / "fig3_independent_L10.png"
    targets = {
        "png": png,
        "npz": png.with_suffix(".npz"),
        "json": png.with_suffix(".json"),
    }
    if with_prior_generation:
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )
        before = {target: target.read_bytes() for target in targets.values()}
    else:
        before = {}
    failed_partial = targets[artifact].with_name(
        targets[artifact].name + ".partial"
    )
    original_fsync = fig3.os.fsync

    def fail_target_fsync(descriptor):
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        try:
            opened_path = Path(os.readlink(descriptor_path))
        except OSError:
            opened_path = None
        if opened_path == failed_partial:
            raise OSError(f"injected real {artifact} fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(fig3.os, "fsync", fail_target_fsync)
    with pytest.raises(OSError, match=f"real {artifact} fsync failure"):
        fig3.render_independent_fig3(
            independent_root,
            output_dir=figure_dir,
            official_data_dir=None,
        )

    if with_prior_generation:
        assert {
            output: output.read_bytes() for output in targets.values()
        } == before
    else:
        assert not any(output.exists() for output in targets.values())
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


def test_independent_panel_a_is_logarithmic_without_replacing_zero_overlaps(
    tmp_path, monkeypatch
):
    root = tmp_path / "independent"
    _output, expected = _independent_result(root)
    loaded = fig3.load_independent_results(root)
    overlap = np.zeros_like(expected["overlap_z2"])
    overlap[0] = 0.5 - 1e-12
    overlap[1] = 1e-12
    loaded[10]["overlap_z2"] = overlap
    fsa_overlap = loaded[10]["fsa_overlap_z2"].copy()
    fsa_overlap[0] = 0.0
    loaded[10]["fsa_overlap_z2"] = fsa_overlap
    monkeypatch.setattr(fig3, "load_independent_results", lambda _root: loaded)
    captured = {}
    original_write = fig3._write_png_partial

    def inspect_panel(partial, figure):
        panel = figure.axes[0]
        captured["yscale"] = panel.get_yscale()
        captured["ed_y"] = np.asarray(panel.collections[0].get_offsets())[:, 1]
        captured["fsa_y"] = np.asarray(panel.collections[1].get_offsets())[:, 1]
        captured["xlim"] = panel.get_xlim()
        captured["xticks"] = panel.get_xticks()
        captured["ylim"] = panel.get_ylim()
        captured["yticks"] = panel.get_yticks()
        captured["ylabel"] = panel.get_ylabel()
        original_write(partial, figure)

    monkeypatch.setattr(fig3, "_write_png_partial", inspect_panel)

    path = fig3.render_independent_fig3(
        root,
        output_dir=tmp_path / "figure",
        official_data_dir=None,
    )

    assert captured["yscale"] == "linear"
    assert captured["ed_y"][0] == pytest.approx(np.log10(overlap[0]))
    assert captured["ed_y"][1] == pytest.approx(-12.0)
    assert np.count_nonzero(~np.isfinite(captured["ed_y"])) == overlap.size - 2
    assert captured["xlim"] == pytest.approx((-20.0, 20.0))
    np.testing.assert_array_equal(captured["xticks"], [-20, -10, 0, 10, 20])
    assert captured["ylim"] == pytest.approx((-10.0, 0.0))
    np.testing.assert_array_equal(captured["yticks"], [-10, -8, -6, -4, -2, 0])
    assert captured["ylabel"] == r"$\log_{10}|\langle Z_2|\psi\rangle|^2$"
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        np.testing.assert_array_equal(sidecar["panel_a_L10_overlap_z2"], overlap)
    metrics = json.loads(path.with_suffix(".json").read_text())
    assert metrics["plot_conventions"]["panel_a"] == {
        "x_range": [-20.0, 20.0],
        "x_ticks": [-20.0, -10.0, 0.0, 10.0, 20.0],
        "y_coordinate": "log10(overlap)",
        "y_range": [-10.0, 0.0],
        "y_ticks": [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0],
        "nonpositive": "NaN-masked",
        "stored_overlap_values": "unmodified",
    }


def test_legacy_panel_a_uses_paper_logarithmic_overlap_axis(tmp_path, monkeypatch):
    captured = {}
    original_savefig = plt.Figure.savefig

    def inspect_panel(figure, *args, **kwargs):
        captured["yscale"] = figure.axes[0].get_yscale()
        captured["ylim"] = figure.axes[0].get_ylim()
        captured["yticks"] = figure.axes[0].get_yticks()
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", inspect_panel)

    fig3.run_figure(10, output_dir=tmp_path, official_data_dir=None)

    assert captured["yscale"] == "linear"
    assert captured["ylim"] == pytest.approx((-10.0, 0.0))
    np.testing.assert_array_equal(captured["yticks"], [-10, -8, -6, -4, -2, 0])


def test_legacy_official_l32_shell_panels_use_thirty_three_unfolded_points(
    tmp_path, monkeypatch
):
    official_root = tmp_path / "official"
    official_root.mkdir()
    for name in (
        "energy_eigenvalues.zip",
        "forward-scattering.zip",
        "overlaps_with_Neel_state.zip",
        "participation_ratios.zip",
    ):
        (official_root / name).touch()
    fixture = _synthetic_shell_panel_inputs(length=32)
    fsa_energies, fsa_vectors = np.linalg.eigh(
        fixture["fsa_hamiltonian_sector"]
    )
    exact_weights = np.abs(fixture["exact_shell_amplitudes"]) ** 2
    monkeypatch.setattr(
        fig3,
        "load_fig3_overlap",
        lambda *_args, **_kwargs: (
            fixture["energies"],
            np.full(fixture["energies"].shape, 0.5 / len(fixture["energies"])),
        ),
    )
    monkeypatch.setattr(
        fig3,
        "load_fig3_fsa",
        lambda *_args, **_kwargs: {
            "energies": fsa_energies,
            "overlap": 0.5 * np.abs(fsa_vectors[0]) ** 2,
            "exact_shell_amplitudes": fixture["exact_shell_amplitudes"],
            "exact_shell_weights": exact_weights,
            "vectors": fsa_vectors,
            "hamiltonian": fixture["fsa_hamiltonian_sector"],
        },
    )
    monkeypatch.setattr(
        fig3,
        "load_fig3_pr2_scaling",
        lambda *_args, **_kwargs: {
            "length": np.asarray([32]),
            "other": np.asarray([0.1]),
            "special": np.asarray([0.2]),
            "available": np.asarray([True]),
        },
    )
    captured = {}
    original_savefig = plt.Figure.savefig

    def inspect_panels(figure, *args, **kwargs):
        for label, panel in zip(("b", "c"), figure.axes[1:3]):
            captured[label] = {
                "title": panel.get_title(),
                "lines": [
                    {
                        "x": np.asarray(line.get_xdata()),
                        "y": np.asarray(line.get_ydata()),
                        "color": line.get_color(),
                        "marker": line.get_marker(),
                        "linestyle": line.get_linestyle(),
                        "label": line.get_label(),
                    }
                    for line in panel.lines
                ],
                "panel_label_position": next(
                    text.get_position()
                    for text in panel.texts
                    if text.get_text() == f"({label})"
                ),
            }
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", inspect_panels)

    fig3.run_figure(10, output_dir=tmp_path, official_data_dir=official_root)

    tower = np.asarray(fixture["matched_tower"]["tower"])
    expected_b_fsa = min(
        range(len(tower)),
        key=lambda index: (fixture["energies"][tower[index]], int(tower[index])),
    )
    expected_c_fsa = min(
        (
            index for index in range(len(tower)) if fsa_energies[index] < -1e-10
        ),
        key=lambda index: (
            abs(fsa_energies[index]),
            int(tower[index]),
        ),
    )
    for panel, fsa_index in (
        ("b", expected_b_fsa),
        ("c", expected_c_fsa),
    ):
        exact_index = int(tower[fsa_index])
        record = captured[panel]
        assert record["title"] == ""
        assert len(record["lines"]) == 2
        exact_line, fsa_line = record["lines"]
        np.testing.assert_array_equal(exact_line["x"], np.arange(33))
        np.testing.assert_allclose(
            exact_line["y"],
            16.0
            * np.abs(
                fig3.unfold_folded_shell_amplitudes(
                    fixture["exact_shell_amplitudes"][:, exact_index], 32
                )
            )
            ** 2,
        )
        np.testing.assert_allclose(
            fsa_line["y"],
            16.0
            * np.abs(
                fig3.unfold_folded_shell_amplitudes(fsa_vectors[:, fsa_index], 32)
            )
            ** 2,
        )
        assert (exact_line["color"], exact_line["marker"], exact_line["linestyle"]) == (
            "black",
            "o",
            "-",
        )
        assert (fsa_line["color"], fsa_line["marker"], fsa_line["linestyle"]) == (
            "red",
            "o",
            "-",
        )
        assert (exact_line["label"], fsa_line["label"]) == ("exact", "FSA")
    assert captured["c"]["panel_label_position"] != pytest.approx((0.02, 0.96))
