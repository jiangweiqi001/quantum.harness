import numpy as np

from run_ssh_ed import diagonalize_ssh


def test_l8_single_particle_sector_has_eight_states():
    energies, vectors, diagnostics = diagonalize_ssh()
    assert diagnostics["basis_dimension"] == 8
    assert energies.shape == (8,)
    assert vectors.shape == (8, 8)


def test_ssh_hamiltonian_is_hermitian_and_particle_hole_symmetric():
    energies, _, diagnostics = diagonalize_ssh()
    assert diagnostics["hermiticity_error"] < 1e-12
    assert diagnostics["particle_hole_error"] < 1e-12
    assert np.allclose(energies, -energies[::-1], atol=1e-12)


def test_two_lowest_absolute_energy_states_are_edge_mode_candidates():
    energies, _, diagnostics = diagonalize_ssh()
    candidate_indices = np.argsort(np.abs(energies))[:2]
    assert diagnostics["edge_mode_indices"] == sorted(candidate_indices.tolist())

    edge_weights = np.asarray(diagnostics["edge_weights"])
    assert len(edge_weights) == 8
    assert np.array_equal(np.argsort(edge_weights)[-2:], candidate_indices)
