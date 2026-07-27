import numpy as np
import pytest
import scipy.sparse as sp

from turner2018_ed_engine import assemble_reduced_hamiltonian, build_orbit_basis
from turner2018_ed_solver import estimate_dense_resources, solve_full_eigensystem


def _reduced_hamiltonian(length: int) -> sp.csr_matrix:
    return assemble_reduced_hamiltonian(build_orbit_basis(length))


def test_l32_dense_resource_estimate():
    estimate = estimate_dense_resources(77_436, vectors=True)
    assert estimate.matrix_bytes == 47_970_672_768
    assert estimate.eigenvector_bytes == 47_970_672_768
    assert estimate.minimum_requested_bytes >= 4 * estimate.matrix_bytes


def test_dense_solver_rejects_nonsymmetric_input():
    matrix = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="real symmetric"):
        solve_full_eigensystem(matrix, vectors=True)


def test_dense_solver_rejects_insufficient_declared_memory():
    matrix = _reduced_hamiltonian(10)
    estimate = estimate_dense_resources(matrix.shape[0], vectors=True)
    with pytest.raises(MemoryError, match="declared memory"):
        solve_full_eigensystem(
            matrix,
            vectors=True,
            declared_memory_bytes=estimate.minimum_requested_bytes - 1,
        )


def test_dense_solver_matches_numpy_l10_with_vectors():
    matrix = _reduced_hamiltonian(10)
    energies, vectors = solve_full_eigensystem(matrix, vectors=True)
    expected_energies, expected_vectors = np.linalg.eigh(matrix.toarray())

    assert vectors is not None
    np.testing.assert_allclose(energies, expected_energies, atol=1e-12, rtol=0.0)
    residual = np.max(
        np.abs(matrix @ vectors - vectors * energies[np.newaxis, :])
    )
    assert float(residual) <= 1e-11
    orthogonality = np.max(np.abs(vectors.T @ vectors - np.eye(vectors.shape[1])))
    assert float(orthogonality) <= 1e-11

    alignment = np.abs(np.sum(expected_vectors * vectors, axis=0))
    np.testing.assert_allclose(alignment, np.ones_like(alignment), atol=1e-8, rtol=0.0)


def test_dense_solver_vectors_false_returns_no_vectors_and_matches_numpy():
    matrix = _reduced_hamiltonian(10)
    energies, vectors = solve_full_eigensystem(matrix, vectors=False)
    expected_energies = np.linalg.eigvalsh(matrix.toarray())

    assert vectors is None
    np.testing.assert_allclose(energies, expected_energies, atol=1e-12, rtol=0.0)
