from pathlib import Path

import numpy as np
import pytest

from turner2018_official import (
    load_fig2_entropy,
    load_fig3_fsa,
    load_fig3_overlap,
    load_fig3_pr2_scaling,
    load_fig4_histograms,
    match_fsa_tower,
    pr2_fsa_averages,
    select_fig3_pr2_states,
)

DATA_DIR = Path(".external/official-data/turner-2018")
REQUIRED_ARCHIVES = (
    "entanglement_entropy_growth.zip",
    "energy_eigenvalues.zip",
    "forward-scattering.zip",
    "level_statistics.zip",
    "overlaps_with_Neel_state.zip",
    "participation_ratios.zip",
)
official_data_required = pytest.mark.skipif(
    not all((DATA_DIR / name).is_file() for name in REQUIRED_ARCHIVES),
    reason="complete official Turner 2018 figure data not downloaded",
)


@official_data_required
def test_load_fig2_official_entropy_curves():
    curves = load_fig2_entropy(DATA_DIR)

    assert set(curves) == {"vacuum", "Z2", "Z3", "Z4"}
    for time, entropy in curves.values():
        assert time[0] == 0.0
        assert entropy[0] == 0.0
        assert np.all(np.diff(time) > 0)


@official_data_required
def test_load_fig3_official_overlap_and_fsa():
    energies, overlap = load_fig3_overlap(DATA_DIR, length=32)
    fsa = load_fig3_fsa(DATA_DIR, length=32)
    fsa_l26 = load_fig3_fsa(DATA_DIR, length=26)

    assert len(energies) == len(overlap) == 77436
    np.testing.assert_allclose(np.sum(overlap), 0.5, atol=1e-10)
    assert len(fsa["energies"]) == 17
    assert len(fsa_l26["energies"]) == 14
    np.testing.assert_allclose(np.sum(fsa["overlap"]), 0.5)
    assert fsa["exact_shell_weights"].shape == (17, 77436)
    assert fsa["exact_shell_amplitudes"].shape == (17, 77436)


@official_data_required
def test_load_fig3_official_pr2_scaling():
    scaling = load_fig3_pr2_scaling(DATA_DIR)

    np.testing.assert_array_equal(scaling["length"], np.arange(12, 33, 2))
    np.testing.assert_array_equal(
        scaling["available"], np.isin(scaling["length"], [26, 32])
    )
    assert np.all(np.isnan(scaling["special"][~scaling["available"]]))
    assert np.all(np.isnan(scaling["other"][~scaling["available"]]))
    assert scaling["special"][scaling["length"] == 26] == pytest.approx(
        0.021911368, abs=5e-10
    )
    assert scaling["special"][scaling["length"] == 32] == pytest.approx(
        0.00685444590, abs=5e-12
    )
    assert scaling["other"][scaling["length"] == 26] == pytest.approx(
        0.0009593895839844039, abs=5e-16
    )
    assert scaling["other"][scaling["length"] == 32] == pytest.approx(
        5.1613385015568644e-05, abs=5e-18
    )


def test_match_fsa_tower_uses_unique_maximum_projection_matches():
    amplitudes = np.zeros((3, 6))
    amplitudes[0, 4] = 1.0
    amplitudes[1, 1] = 1.0
    amplitudes[2, 5] = 1.0

    matched = match_fsa_tower(amplitudes, np.eye(3))

    np.testing.assert_array_equal(matched, [4, 1, 5])


def test_match_fsa_tower_rejects_duplicate_exact_matches():
    amplitudes = np.zeros((2, 3))
    amplitudes[:, 1] = 1.0

    with pytest.raises(ValueError, match="one-to-one"):
        match_fsa_tower(amplitudes, np.eye(2))


def test_match_fsa_tower_rejects_nearly_tied_column_maximum():
    amplitudes = np.array([[1.0, np.sqrt(1.0 - 5e-9)]])

    with pytest.raises(ValueError, match="tied"):
        match_fsa_tower(amplitudes, np.eye(1))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_match_fsa_tower_rejects_nonfinite_projection(bad_value):
    amplitudes = np.array([[1.0, bad_value]])

    with pytest.raises(ValueError, match="finite"):
        match_fsa_tower(amplitudes, np.eye(1))


def test_selector_trims_middle_two_thirds_and_excludes_zero():
    amplitudes = np.zeros((6, 9))
    tower = np.array([7, 1, 5, 3, 8, 0])
    amplitudes[np.arange(6), tower] = 1.0
    energies = np.array([5.0, -3.0, 0.4, 0.0, 0.8, -1.0, 2.0, -5.0, 3.0])

    selected = select_fig3_pr2_states(
        energies=energies,
        exact_shell_amplitudes=amplitudes,
        fsa_eigenvectors=np.eye(6),
    )

    np.testing.assert_array_equal(selected["sorted_tower"], [7, 1, 5, 3, 8, 0])
    np.testing.assert_array_equal(selected["special"], [1, 5, 8])


def test_other_states_are_nonzero_complement_of_full_untrimmed_tower():
    amplitudes = np.zeros((6, 9))
    tower = np.array([7, 1, 5, 3, 8, 0])
    amplitudes[np.arange(6), tower] = 1.0
    energies = np.array([5.0, -3.0, 0.4, 0.0, 0.8, -1.0, 0.0, -5.0, 3.0])
    pr2 = np.arange(1.0, 10.0)

    other, special = pr2_fsa_averages(
        energies=energies,
        pr2=pr2,
        exact_shell_amplitudes=amplitudes,
        fsa_eigenvectors=np.eye(6),
    )

    assert other == pytest.approx(np.mean(pr2[[2, 4]]))
    assert special == pytest.approx(np.mean(pr2[[1, 5, 8]]))


@pytest.mark.parametrize("length", [28, 30])
def test_official_fsa_matching_is_explicitly_unavailable(length):
    with pytest.raises(ValueError, match=f"L={length}.*unavailable"):
        load_fig3_fsa(DATA_DIR, length=length)


@official_data_required
def test_load_fig4_official_histograms():
    histograms = load_fig4_histograms(DATA_DIR)

    assert set(histograms) == {28, 30, 32}
    for spacing, density in histograms.values():
        assert spacing.shape == density.shape == (25,)
        assert np.all(density >= 0)
