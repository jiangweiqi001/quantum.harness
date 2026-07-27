import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from pxp_ed import (
    basis_state_vector,
    constrained_basis,
    density_wave_state,
    is_blockaded,
    lucas_number,
    pxp_hamiltonian,
)


@pytest.mark.parametrize("length", range(3, 13))
def test_pbc_basis_dimension_is_lucas_number(length):
    basis = constrained_basis(length, pbc=True)

    assert len(basis) == lucas_number(length)
    assert np.all(basis[:-1] < basis[1:])
    assert all(is_blockaded(int(state), length, pbc=True) for state in basis)


def test_pxp_hamiltonian_is_real_symmetric_and_contains_only_legal_flips():
    length = 8
    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    index = {int(state): i for i, state in enumerate(basis)}

    assert (hamiltonian - hamiltonian.T).nnz == 0
    rows, columns = hamiltonian.nonzero()
    for row, column in zip(rows, columns):
        before = int(basis[column])
        after = int(basis[row])
        difference = before ^ after
        assert difference.bit_count() == 1
        assert after in index
        assert is_blockaded(after, length, pbc=True)


def test_sparse_extremal_spectrum_matches_dense_diagonalization():
    basis = constrained_basis(10, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, 10, pbc=True)

    dense = np.linalg.eigvalsh(hamiltonian.toarray())
    sparse = eigsh(
        hamiltonian.astype(float), k=3, which="LA", return_eigenvectors=False
    )

    np.testing.assert_allclose(np.sort(sparse), dense[-3:], atol=1e-10)
    np.testing.assert_allclose(dense, -dense[::-1], atol=1e-10)


@pytest.mark.parametrize("period", [2, 3, 4])
def test_density_wave_states_are_normalized_blockaded_basis_vectors(period):
    length = 12
    basis = constrained_basis(length, pbc=True)
    state = density_wave_state(length, period)
    vector = basis_state_vector(basis, state)

    assert is_blockaded(state, length, pbc=True)
    assert np.count_nonzero(vector) == 1
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_density_wave_rejects_incompatible_periodic_ring():
    with pytest.raises(ValueError, match="blockade"):
        density_wave_state(5, period=2)
