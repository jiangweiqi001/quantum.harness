"""Independent observables and streamed FSA diagnostics for Turner 2018 ED."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from pxp_ed import density_wave_state
from turner2018_ed_engine import OrbitBasis, assemble_reduced_hamiltonian, canonical_dihedral


def _orbit_projection_data(basis: OrbitBasis) -> tuple[np.ndarray, np.ndarray]:
    canonical = canonical_dihedral(basis.constrained_states, basis.length)
    orbit_index_by_state = basis.index_of(canonical).astype(np.int64, copy=False)
    inverse_sqrt_orbit_sizes = 1.0 / np.sqrt(basis.orbit_sizes.astype(np.float64))
    return orbit_index_by_state, inverse_sqrt_orbit_sizes


def _project_full_vector_with_data(
    full_vector: np.ndarray,
    *,
    orbit_index_by_state: np.ndarray,
    inverse_sqrt_orbit_sizes: np.ndarray,
) -> np.ndarray:
    weighted = np.bincount(
        orbit_index_by_state,
        weights=np.asarray(full_vector, dtype=np.float64),
        minlength=inverse_sqrt_orbit_sizes.shape[0],
    )
    return weighted * inverse_sqrt_orbit_sizes


def project_product_state(basis: OrbitBasis, product_state: int) -> np.ndarray:
    """Project one constrained product state into the orbit basis."""
    states = basis.constrained_states
    position = int(np.searchsorted(states, np.uint64(product_state)))
    if position >= len(states) or int(states[position]) != int(product_state):
        raise ValueError("product_state must belong to the constrained basis")

    full_vector = np.zeros(len(states), dtype=np.float64)
    full_vector[position] = 1.0
    orbit_index_by_state, inverse_sqrt_orbit_sizes = _orbit_projection_data(basis)
    return _project_full_vector_with_data(
        full_vector,
        orbit_index_by_state=orbit_index_by_state,
        inverse_sqrt_orbit_sizes=inverse_sqrt_orbit_sizes,
    )


def stream_pr2(vectors: np.ndarray, chunk_columns: int = 256) -> np.ndarray:
    """Compute PR2 column-wise using bounded temporary memory."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 1:
        return np.asarray([np.sum(np.abs(array) ** 4)], dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("vectors must be a one- or two-dimensional array")
    if chunk_columns < 1:
        raise ValueError("chunk_columns must be positive")

    result = np.empty(array.shape[1], dtype=np.float64)
    for start in range(0, array.shape[1], chunk_columns):
        stop = min(start + chunk_columns, array.shape[1])
        block = array[:, start:stop]
        result[start:stop] = np.sum(np.abs(block) ** 4, axis=0)
    return result


@dataclass(frozen=True)
class StreamedFSAResult:
    beta: np.ndarray
    projected_shells: np.ndarray
    reduced_fsa_hamiltonian: np.ndarray
    full_dimension: int
    reduced_dimension: int
    full_buffer_shape: tuple[int, ...]


def stream_fsa_shells(
    basis: OrbitBasis,
    initial_state: int,
    *,
    reduced_hamiltonian: sp.csr_matrix | np.ndarray | None = None,
    max_shell: int | None = None,
) -> StreamedFSAResult:
    """Stream FSA shells on the full constrained basis and project each shell."""
    if max_shell is not None and max_shell < 0:
        raise ValueError("max_shell must be non-negative when provided")

    states = np.asarray(basis.constrained_states, dtype=np.uint64)
    full_dimension = len(states)
    position = int(np.searchsorted(states, np.uint64(initial_state)))
    if position >= full_dimension or int(states[position]) != int(initial_state):
        raise ValueError("initial_state must belong to the constrained basis")

    length = basis.length
    shell_limit = length if max_shell is None else min(int(max_shell), length)
    projected_shell_count = min(length // 2 + 1, shell_limit + 1)
    reduced_dimension = len(basis.representatives)
    projected_shells = np.zeros(
        (projected_shell_count, reduced_dimension),
        dtype=np.float64,
    )
    beta = np.empty(shell_limit, dtype=np.float64)

    orbit_index_by_state, inverse_sqrt_orbit_sizes = _orbit_projection_data(basis)
    distances = np.asarray(
        [(int(state) ^ int(initial_state)).bit_count() for state in states],
        dtype=np.int16,
    )

    current = np.zeros(full_dimension, dtype=np.float64)
    next_shell = np.zeros(full_dimension, dtype=np.float64)
    current[position] = 1.0

    shell0 = _project_full_vector_with_data(
        current,
        orbit_index_by_state=orbit_index_by_state,
        inverse_sqrt_orbit_sizes=inverse_sqrt_orbit_sizes,
    )
    shell0_norm = float(np.linalg.norm(shell0))
    if shell0_norm < 1e-14:
        raise RuntimeError("projected shell 0 has zero norm")
    projected_shells[0] = shell0 / shell0_norm

    for shell in range(shell_limit):
        next_shell.fill(0.0)
        support = np.flatnonzero(np.abs(current) > 0.0)
        for source_index in support:
            source_state = int(states[source_index])
            source_distance = int(distances[source_index])
            amplitude = float(current[source_index])
            for site in range(length):
                left = (site - 1) % length
                right = (site + 1) % length
                if ((source_state >> left) & 1) or ((source_state >> right) & 1):
                    continue
                destination_state = source_state ^ (1 << site)
                destination_index = int(np.searchsorted(states, np.uint64(destination_state)))
                if destination_index >= full_dimension:
                    continue
                if int(states[destination_index]) != destination_state:
                    continue
                if int(distances[destination_index]) != source_distance + 1:
                    continue
                next_shell[destination_index] += amplitude

        norm = float(np.linalg.norm(next_shell))
        if norm < 1e-14:
            raise RuntimeError("forward-scattering chain terminated before distance L")
        beta[shell] = norm
        next_shell /= norm

        projected_index = shell + 1
        if projected_index < projected_shell_count:
            projected = _project_full_vector_with_data(
                next_shell,
                orbit_index_by_state=orbit_index_by_state,
                inverse_sqrt_orbit_sizes=inverse_sqrt_orbit_sizes,
            )
            projected_norm = float(np.linalg.norm(projected))
            if projected_norm < 1e-14:
                raise RuntimeError("projected shell has zero norm")
            projected_shells[projected_index] = projected / projected_norm
        current, next_shell = next_shell, current

    if reduced_hamiltonian is None:
        reduced_dense = assemble_reduced_hamiltonian(basis).toarray()
    elif sp.issparse(reduced_hamiltonian):
        reduced_dense = reduced_hamiltonian.toarray()
    else:
        reduced_dense = np.asarray(reduced_hamiltonian, dtype=np.float64)
    reduced_fsa_hamiltonian = projected_shells @ reduced_dense @ projected_shells.T
    reduced_fsa_hamiltonian = 0.5 * (reduced_fsa_hamiltonian + reduced_fsa_hamiltonian.T)

    return StreamedFSAResult(
        beta=beta,
        projected_shells=projected_shells,
        reduced_fsa_hamiltonian=np.asarray(reduced_fsa_hamiltonian, dtype=np.float64),
        full_dimension=full_dimension,
        reduced_dimension=reduced_dimension,
        full_buffer_shape=current.shape,
    )


def compare_degenerate_invariants(
    *,
    reference_energies: np.ndarray,
    reference_vectors: np.ndarray,
    candidate_energies: np.ndarray,
    candidate_vectors: np.ndarray,
    z2_sector_state: np.ndarray,
    tolerance: float = 1e-10,
) -> dict[str, float | int]:
    """Compare eigensystems with degeneracy-safe Z2/projector invariants."""
    energies_ref = np.asarray(reference_energies, dtype=np.float64)
    energies_candidate = np.asarray(candidate_energies, dtype=np.float64)
    vectors_ref = np.asarray(reference_vectors, dtype=np.float64)
    vectors_candidate = np.asarray(candidate_vectors, dtype=np.float64)
    z2 = np.asarray(z2_sector_state, dtype=np.float64)

    if energies_ref.shape != energies_candidate.shape:
        raise ValueError("reference and candidate energies must have the same shape")
    if vectors_ref.shape != vectors_candidate.shape:
        raise ValueError("reference and candidate vectors must have the same shape")
    if vectors_ref.shape[0] != z2.shape[0]:
        raise ValueError("z2_sector_state size must match vector rows")
    if not np.allclose(energies_ref, energies_candidate, atol=tolerance, rtol=0.0):
        raise ValueError("reference and candidate energies must agree within tolerance")

    overlap_ref = np.abs(vectors_ref.T @ z2) ** 2
    overlap_candidate = np.abs(vectors_candidate.T @ z2) ** 2
    pr2_ref = stream_pr2(vectors_ref)
    pr2_candidate = stream_pr2(vectors_candidate)

    groups: list[np.ndarray] = []
    start = 0
    for index in range(1, energies_ref.size + 1):
        if index == energies_ref.size or (energies_ref[index] - energies_ref[index - 1]) > tolerance:
            groups.append(np.arange(start, index, dtype=np.int64))
            start = index

    max_total_z2_diff = 0.0
    max_projector_diag_diff = 0.0
    max_pr2_diff_isolated = 0.0
    degenerate_groups = 0
    isolated_groups = 0

    for group in groups:
        total_z2_diff = abs(
            float(np.sum(overlap_ref[group]) - np.sum(overlap_candidate[group]))
        )
        max_total_z2_diff = max(max_total_z2_diff, total_z2_diff)

        projector_diag_ref = np.sum(np.abs(vectors_ref[:, group]) ** 2, axis=1)
        projector_diag_candidate = np.sum(np.abs(vectors_candidate[:, group]) ** 2, axis=1)
        projector_diag_diff = float(
            np.max(np.abs(projector_diag_ref - projector_diag_candidate))
        )
        max_projector_diag_diff = max(max_projector_diag_diff, projector_diag_diff)

        if len(group) == 1:
            isolated_groups += 1
            pr2_diff = abs(float(pr2_ref[group[0]] - pr2_candidate[group[0]]))
            max_pr2_diff_isolated = max(max_pr2_diff_isolated, pr2_diff)
        else:
            degenerate_groups += 1

    return {
        "max_total_z2_diff": max_total_z2_diff,
        "max_projector_diag_diff": max_projector_diag_diff,
        "max_pr2_diff_isolated": max_pr2_diff_isolated,
        "degenerate_group_count": degenerate_groups,
        "isolated_group_count": isolated_groups,
    }


def compute_observables(
    basis: OrbitBasis,
    reduced_hamiltonian: sp.csr_matrix | np.ndarray,
    *,
    z2_pattern: int = 2,
) -> dict[str, np.ndarray | dict[str, float | int]]:
    """Compute overlap/PR2/FSA observables in the reduced orbit basis."""
    initial_state = density_wave_state(basis.length, z2_pattern)
    z2_sector = project_product_state(basis, initial_state)

    reduced_dense = (
        reduced_hamiltonian.toarray()
        if sp.issparse(reduced_hamiltonian)
        else np.asarray(reduced_hamiltonian, dtype=np.float64)
    )
    energies, eigenvectors = np.linalg.eigh(reduced_dense)
    overlap_z2 = np.abs(eigenvectors.T @ z2_sector) ** 2
    pr2 = stream_pr2(eigenvectors)

    streamed_fsa = stream_fsa_shells(
        basis,
        initial_state,
        reduced_hamiltonian=reduced_dense,
    )
    fsa_energies, fsa_eigenvectors = np.linalg.eigh(streamed_fsa.reduced_fsa_hamiltonian)
    z2_norm = float(np.vdot(z2_sector, z2_sector).real)
    fsa_overlap_z2 = z2_norm * np.abs(fsa_eigenvectors[0, :]) ** 2

    exact_shell_amplitudes = streamed_fsa.projected_shells @ eigenvectors
    invariants = compare_degenerate_invariants(
        reference_energies=energies,
        reference_vectors=eigenvectors,
        candidate_energies=energies,
        candidate_vectors=eigenvectors,
        z2_sector_state=z2_sector,
    )
    return {
        "energies": np.asarray(energies, dtype=np.float64),
        "eigenvectors": np.asarray(eigenvectors, dtype=np.float64),
        "overlap_z2": np.asarray(overlap_z2, dtype=np.float64),
        "participation_ratio": np.asarray(pr2, dtype=np.float64),
        "fsa_energies": np.asarray(fsa_energies, dtype=np.float64),
        "fsa_overlap_z2": np.asarray(fsa_overlap_z2, dtype=np.float64),
        "fsa_eigenvectors": np.asarray(fsa_eigenvectors, dtype=np.float64),
        "exact_shell_amplitudes": np.asarray(exact_shell_amplitudes, dtype=np.float64),
        "fsa_shell_vectors_sector": np.asarray(streamed_fsa.projected_shells, dtype=np.float64),
        "fsa_hamiltonian_sector": np.asarray(
            streamed_fsa.reduced_fsa_hamiltonian, dtype=np.float64
        ),
        "fsa_beta_full_chain": np.asarray(streamed_fsa.beta, dtype=np.float64),
        "invariant_diagnostics": invariants,
    }
