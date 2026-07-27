import numpy as np
import matplotlib.pyplot as plt

from pxp_ed import constrained_basis, density_wave_state, pxp_hamiltonian
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
