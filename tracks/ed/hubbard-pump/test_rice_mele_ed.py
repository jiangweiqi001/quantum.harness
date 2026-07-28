from __future__ import annotations

import json
from math import comb

import numpy as np

from run_rice_mele_ed import build_rice_mele_hamiltonian, diagonalize_rice_mele, write_artifacts


PARAMETERS = {"L": 6, "delta": 0.5, "Delta": 0.3, "theta": 2.0 * np.pi, "t": 1.0}


def test_l6_half_filled_zero_magnetization_full_eigensystem():
    energies, vectors, diagnostics = diagonalize_rice_mele(**PARAMETERS)
    assert diagnostics["basis_dimension"] == comb(6, 3) ** 2 == 400
    assert energies.shape == (400,)
    assert vectors.shape == (400, 400)
    assert np.all(np.diff(energies) >= -1e-12)


def test_l6_hamiltonian_is_hermitian():
    _, H = build_rice_mele_hamiltonian(**PARAMETERS)
    matrix = H.toarray()
    assert np.max(np.abs(matrix - matrix.conj().T)) < 1e-12


def test_twist_two_pi_matches_zero_for_matrix_and_spectrum():
    zero_parameters = {**PARAMETERS, "theta": 0.0}
    _, H_zero = build_rice_mele_hamiltonian(**zero_parameters)
    _, H_two_pi = build_rice_mele_hamiltonian(**PARAMETERS)
    assert np.allclose(H_zero.toarray(), H_two_pi.toarray(), atol=1e-12)

    energies_zero, _, _ = diagonalize_rice_mele(**zero_parameters)
    energies_two_pi, _, _ = diagonalize_rice_mele(**PARAMETERS)
    assert np.allclose(energies_zero, energies_two_pi, atol=1e-12)


def test_write_artifacts_persists_complete_eigensystem(tmp_path):
    files = write_artifacts(tmp_path, **PARAMETERS)
    assert set(files) == {"eigenvalues", "eigenvectors", "manifest"}

    energies = np.load(tmp_path / files["eigenvalues"])
    vectors = np.load(tmp_path / files["eigenvectors"])
    assert energies.shape == (400,)
    assert vectors.shape == (400, 400)

    manifest = json.loads((tmp_path / files["manifest"]).read_text())
    assert manifest["parameters"] == PARAMETERS
    assert manifest["diagnostics"]["basis_dimension"] == 400
    assert manifest["diagnostics"]["eigenvalue_shape"] == [400]
    assert manifest["diagnostics"]["eigenvector_shape"] == [400, 400]
    assert manifest["diagnostics"]["hermiticity_error"] < 1e-12


def test_l8_half_filled_sector_dimension():
    basis, _ = build_rice_mele_hamiltonian(L=8, delta=0.5, Delta=0.3, theta=0.0)
    assert basis.Ns == comb(8, 4) ** 2 == 4900
