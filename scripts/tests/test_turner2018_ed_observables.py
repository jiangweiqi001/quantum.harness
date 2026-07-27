import numpy as np
import pytest

from pxp_ed import (
    constrained_basis,
    density_wave_state,
    pxp_hamiltonian,
    symmetry_basis_k0_inversion_even,
)
from turner2018_ed_engine import assemble_reduced_hamiltonian, build_orbit_basis
from turner2018_ed_observables import (
    compare_degenerate_invariants,
    project_product_state,
    stream_fsa_shells,
    stream_pr2,
)
from turner2018_fig3 import fsa_basis


def _align_global_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    overlap = float(np.dot(reference, candidate))
    if overlap < 0.0:
        return -candidate
    return candidate


def _oracle_project_shells(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orbit = build_orbit_basis(length)
    reduced = assemble_reduced_hamiltonian(orbit).toarray()
    constrained = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(constrained, length, pbc=True)
    z2 = density_wave_state(length, 2)
    shells_full, beta = fsa_basis(hamiltonian, constrained, z2, length)

    transform = symmetry_basis_k0_inversion_even(constrained, length).toarray()
    projected_complex = (transform.T @ shells_full.T).T[: length // 2 + 1]
    np.testing.assert_allclose(projected_complex.imag, 0.0, atol=1e-14, rtol=0.0)
    projected_shells = np.asarray(projected_complex.real, dtype=np.float64)
    projected_shells /= np.linalg.norm(projected_shells, axis=1, keepdims=True)
    projected_h = projected_shells @ reduced @ projected_shells.T
    projected_h = 0.5 * (projected_h + projected_h.T)
    return np.asarray(beta, dtype=np.float64), projected_shells, projected_h


def test_z2_sector_projection_has_half_weight():
    basis = build_orbit_basis(12)
    z2 = project_product_state(basis, density_wave_state(12, 2))
    assert np.vdot(z2, z2).real == pytest.approx(0.5)


def test_streamed_pr2_matches_definition():
    vectors = np.linalg.qr(np.arange(1, 26, dtype=float).reshape(5, 5))[0]
    np.testing.assert_allclose(
        stream_pr2(vectors, chunk_columns=2),
        np.sum(np.abs(vectors) ** 4, axis=0),
    )


@pytest.mark.parametrize("length", [10, 12, 14, 16])
def test_streamed_fsa_matches_full_basis_oracle(length: int):
    orbit = build_orbit_basis(length)
    z2 = density_wave_state(length, 2)
    streamed = stream_fsa_shells(orbit, z2)
    oracle_beta, oracle_shells, oracle_h = _oracle_project_shells(length)

    np.testing.assert_allclose(streamed.beta, oracle_beta, atol=1e-12, rtol=0.0)

    for shell_index in range(oracle_shells.shape[0]):
        aligned = _align_global_sign(
            oracle_shells[shell_index], streamed.projected_shells[shell_index]
        )
        np.testing.assert_allclose(
            aligned,
            oracle_shells[shell_index],
            atol=1e-12,
            rtol=0.0,
        )

    np.testing.assert_allclose(
        streamed.reduced_fsa_hamiltonian,
        oracle_h,
        atol=1e-11,
        rtol=0.0,
    )


def test_degenerate_invariant_comparison_uses_projectors_not_vector_pr2():
    energies = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    reference = np.eye(3, dtype=np.float64)
    angle = np.pi / 4.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    candidate = reference @ rotation
    z2 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    diagnostics = compare_degenerate_invariants(
        reference_energies=energies,
        reference_vectors=reference,
        candidate_energies=energies,
        candidate_vectors=candidate,
        z2_sector_state=z2,
    )

    assert diagnostics["max_total_z2_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["max_projector_diag_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["degenerate_group_count"] == 1
    assert diagnostics["isolated_group_count"] == 1
    # Rotating inside the degenerate two-state manifold changes vector-level PR2.
    assert diagnostics["max_pr2_diff_isolated"] == pytest.approx(0.0, abs=1e-12)
