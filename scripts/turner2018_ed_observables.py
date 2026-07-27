"""Independent observables and streamed FSA diagnostics for Turner 2018 ED."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np
import scipy.sparse as sp

from pxp_ed import density_wave_state
from turner2018_ed_engine import OrbitBasis, assemble_reduced_hamiltonian, canonical_dihedral
from turner2018_ed_validation import (
    ORTHOGONALITY_PAIR_MEMORY_POLICY,
    ORTHOGONALITY_SAMPLING_POLICY,
    positive_integer,
    sample_pair_batches,
    uint64_state,
    validate_eigenpair_residuals,
    validate_hermitian,
    validate_orthonormal_columns,
)


_BYTE_POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def _validate_chunk_size(chunk_size: int) -> int:
    return positive_integer(chunk_size, "chunk_size")


def _validate_max_shell(max_shell: int | None, length: int) -> int:
    if max_shell is None:
        return length
    if isinstance(max_shell, (bool, np.bool_)):
        raise TypeError("max_shell must be an integer in [0, L]")
    try:
        value = operator.index(max_shell)
    except TypeError as error:
        raise TypeError("max_shell must be an integer in [0, L]") from error
    if not 0 <= value <= length:
        raise ValueError("max_shell must be an integer in [0, L]")
    return value


def _hamming_distances_chunked(
    states: np.ndarray,
    initial_state: int,
    *,
    chunk_size: int,
) -> np.ndarray:
    distances = np.empty(states.size, dtype=np.uint8)
    initial = np.uint64(initial_state)
    for start in range(0, states.size, chunk_size):
        stop = min(start + chunk_size, states.size)
        xor_bytes = np.ascontiguousarray(states[start:stop] ^ initial).view(np.uint8)
        xor_bytes = xor_bytes.reshape(stop - start, np.dtype(np.uint64).itemsize)
        distances[start:stop] = np.sum(
            _BYTE_POPCOUNT[xor_bytes],
            axis=1,
            dtype=np.uint8,
        )
    return distances


def _project_full_vector_chunked(
    basis: OrbitBasis,
    full_vector: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    states = np.asarray(basis.constrained_states, dtype=np.uint64)
    projected = np.zeros(len(basis.representatives), dtype=np.float64)
    for start in range(0, states.size, chunk_size):
        stop = min(start + chunk_size, states.size)
        amplitudes = full_vector[start:stop]
        canonical = canonical_dihedral(states[start:stop], basis.length)
        orbit_indices = basis.index_of(canonical)
        np.add.at(projected, orbit_indices, amplitudes)
    projected /= np.sqrt(basis.orbit_sizes.astype(np.float64))
    return projected


def project_product_state(
    basis: OrbitBasis,
    product_state: int,
    *,
    chunk_size: int = 262144,
) -> np.ndarray:
    """Project one constrained product state into the orbit basis."""
    _validate_chunk_size(chunk_size)
    product_state = uint64_state(product_state, "product_state")
    states = basis.constrained_states
    position = int(np.searchsorted(states, np.uint64(product_state)))
    if position >= len(states) or int(states[position]) != int(product_state):
        raise ValueError("product_state must belong to the constrained basis")

    canonical = canonical_dihedral(
        np.asarray([np.uint64(product_state)]),
        basis.length,
    )
    orbit_index = int(basis.index_of(canonical)[0])
    projected = np.zeros(len(basis.representatives), dtype=np.float64)
    projected[orbit_index] = 1.0 / np.sqrt(float(basis.orbit_sizes[orbit_index]))
    return projected


def stream_pr2(vectors: np.ndarray, chunk_columns: int = 256) -> np.ndarray:
    """Compute PR2 column-wise using bounded temporary memory."""
    chunk_columns = positive_integer(chunk_columns, "chunk_columns")
    array = np.asarray(vectors)
    if array.ndim == 1:
        return np.asarray([np.sum(np.abs(array) ** 4)], dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("vectors must be a one- or two-dimensional array")
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
    chunk_size: int
    max_source_chunk: int
    max_projection_chunk: int
    distance_dtype: str


def stream_fsa_shells(
    basis: OrbitBasis,
    initial_state: int,
    *,
    reduced_hamiltonian: sp.csr_matrix | np.ndarray | None = None,
    max_shell: int | None = None,
    chunk_size: int = 262144,
) -> StreamedFSAResult:
    """Stream FSA shells on the full constrained basis and project each shell."""
    chunk_size = _validate_chunk_size(chunk_size)
    initial_state = uint64_state(initial_state, "initial_state")

    states = np.asarray(basis.constrained_states, dtype=np.uint64)
    full_dimension = len(states)
    position = int(np.searchsorted(states, np.uint64(initial_state)))
    if position >= full_dimension or int(states[position]) != int(initial_state):
        raise ValueError("initial_state must belong to the constrained basis")

    length = basis.length
    shell_limit = _validate_max_shell(max_shell, length)
    projected_shell_count = min(length // 2 + 1, shell_limit + 1)
    reduced_dimension = len(basis.representatives)
    projected_shells = np.zeros(
        (projected_shell_count, reduced_dimension),
        dtype=np.float64,
    )
    beta = np.empty(shell_limit, dtype=np.float64)

    distances = _hamming_distances_chunked(
        states,
        initial_state,
        chunk_size=chunk_size,
    )

    current = np.zeros(full_dimension, dtype=np.float64)
    next_shell = np.zeros(full_dimension, dtype=np.float64)
    current[position] = 1.0
    max_source_chunk = 0

    shell0 = _project_full_vector_chunked(
        basis,
        current,
        chunk_size=chunk_size,
    )
    shell0_norm = float(np.linalg.norm(shell0))
    if shell0_norm < 1e-14:
        raise RuntimeError("projected shell 0 has zero norm")
    projected_shells[0] = shell0 / shell0_norm

    for shell in range(shell_limit):
        next_shell.fill(0.0)
        for start in range(0, full_dimension, chunk_size):
            stop = min(start + chunk_size, full_dimension)
            chunk_amplitudes = current[start:stop]
            support = np.flatnonzero(chunk_amplitudes)
            if support.size == 0:
                continue
            max_source_chunk = max(max_source_chunk, int(support.size))
            source_states = states[start:stop][support]
            source_distances = distances[start:stop][support]
            source_amplitudes = chunk_amplitudes[support]

            for site in range(length):
                left = (site - 1) % length
                right = (site + 1) % length
                legal = (
                    ((source_states >> np.uint64(left)) & np.uint64(1)) == 0
                ) & (
                    ((source_states >> np.uint64(right)) & np.uint64(1)) == 0
                )
                if not np.any(legal):
                    continue
                destination_states = (
                    source_states[legal] ^ (np.uint64(1) << np.uint64(site))
                )
                destination_indices = np.searchsorted(states, destination_states)
                in_basis = destination_indices < full_dimension
                if not np.any(in_basis):
                    continue
                destination_states = destination_states[in_basis]
                destination_indices = destination_indices[in_basis]
                source_legal_indices = np.flatnonzero(legal)[in_basis]
                exact_match = states[destination_indices] == destination_states
                if not np.any(exact_match):
                    continue
                destination_indices = destination_indices[exact_match]
                source_legal_indices = source_legal_indices[exact_match]
                increases_distance = (
                    distances[destination_indices]
                    == source_distances[source_legal_indices] + np.uint8(1)
                )
                if not np.any(increases_distance):
                    continue
                np.add.at(
                    next_shell,
                    destination_indices[increases_distance],
                    source_amplitudes[source_legal_indices[increases_distance]],
                )

        norm = float(np.linalg.norm(next_shell))
        if norm < 1e-14:
            raise RuntimeError("forward-scattering chain terminated before distance L")
        beta[shell] = norm
        next_shell /= norm

        projected_index = shell + 1
        if projected_index < projected_shell_count:
            projected = _project_full_vector_chunked(
                basis,
                next_shell,
                chunk_size=chunk_size,
            )
            projected_norm = float(np.linalg.norm(projected))
            if projected_norm < 1e-14:
                raise RuntimeError("projected shell has zero norm")
            projected_shells[projected_index] = projected / projected_norm
        current, next_shell = next_shell, current

    if reduced_hamiltonian is None:
        reduced_matrix = assemble_reduced_hamiltonian(basis)
    else:
        reduced_matrix = reduced_hamiltonian
    if reduced_matrix.shape != (reduced_dimension, reduced_dimension):
        raise ValueError("reduced_hamiltonian shape must match the orbit basis")
    if sp.issparse(reduced_matrix):
        h_shells = reduced_matrix @ projected_shells.T
    else:
        h_shells = np.asarray(reduced_matrix) @ projected_shells.T
    reduced_fsa_hamiltonian = projected_shells @ h_shells
    reduced_fsa_hamiltonian = 0.5 * (reduced_fsa_hamiltonian + reduced_fsa_hamiltonian.T)

    return StreamedFSAResult(
        beta=beta,
        projected_shells=projected_shells,
        reduced_fsa_hamiltonian=np.asarray(reduced_fsa_hamiltonian, dtype=np.float64),
        full_dimension=full_dimension,
        reduced_dimension=reduced_dimension,
        full_buffer_shape=current.shape,
        chunk_size=chunk_size,
        max_source_chunk=max_source_chunk,
        max_projection_chunk=min(chunk_size, full_dimension),
        distance_dtype=distances.dtype.name,
    )


def _projector_diagonal_chunked(
    vectors: np.ndarray,
    group: np.ndarray,
    *,
    chunk_columns: int,
) -> np.ndarray:
    diagonal = np.zeros(vectors.shape[0], dtype=np.float64)
    for start in range(0, group.size, chunk_columns):
        stop = min(start + chunk_columns, group.size)
        block = vectors[:, group[start:stop]]
        diagonal += np.sum(np.abs(block) ** 2, axis=1)
    return diagonal


def _two_way_subspace_residual(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    chunk_columns: int,
) -> float:
    maximum = 0.0
    for source, target in ((candidate, reference), (reference, candidate)):
        for start in range(0, source.shape[1], chunk_columns):
            stop = min(start + chunk_columns, source.shape[1])
            block = source[:, start:stop]
            projected = target @ (target.conj().T @ block)
            residual_norms = np.sqrt(
                np.sum(np.abs(block - projected) ** 2, axis=0)
            )
            maximum = max(maximum, float(np.max(residual_norms)))
    return maximum


def compare_degenerate_invariants(
    *,
    reference_energies: np.ndarray,
    reference_vectors: np.ndarray,
    candidate_energies: np.ndarray,
    candidate_vectors: np.ndarray,
    reference_z2_sector_state: np.ndarray,
    candidate_z2_sector_state: np.ndarray,
    tolerance: float = 1e-10,
    chunk_columns: int = 256,
    orthogonality_samples: int = 4096,
) -> dict[str, float | int | str]:
    """Compare complete eigenspaces using bounded two-way projector residuals."""
    chunk_columns = positive_integer(chunk_columns, "chunk_columns")
    orthogonality_samples = positive_integer(
        orthogonality_samples, "orthogonality_samples"
    )
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not np.isscalar(tolerance)
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be finite and positive")

    raw_energies_ref = np.asarray(reference_energies)
    raw_energies_candidate = np.asarray(candidate_energies)
    if np.iscomplexobj(raw_energies_ref) or np.iscomplexobj(raw_energies_candidate):
        raise ValueError("energy arrays must be real")
    energies_ref = np.asarray(raw_energies_ref, dtype=np.float64)
    energies_candidate = np.asarray(raw_energies_candidate, dtype=np.float64)
    for name, energies in (
        ("reference", energies_ref),
        ("candidate", energies_candidate),
    ):
        if energies.ndim != 1:
            raise ValueError(f"{name} energies must be one-dimensional")
        if energies.size == 0:
            raise ValueError(f"{name} energies must be non-empty")
        if not np.all(np.isfinite(energies)):
            raise ValueError(f"{name} energies must be finite")
        if np.any(np.diff(energies) < 0):
            raise ValueError(f"{name} energies must be sorted")
    if energies_ref.shape != energies_candidate.shape:
        raise ValueError("reference and candidate energies must have the same shape")

    vectors_ref = np.asarray(reference_vectors)
    vectors_candidate = np.asarray(candidate_vectors)
    for name, vectors, energies in (
        ("reference", vectors_ref, energies_ref),
        ("candidate", vectors_candidate, energies_candidate),
    ):
        if vectors.ndim != 2:
            raise ValueError(f"{name} vectors must be two-dimensional")
        if vectors.shape[1] != energies.size:
            raise ValueError(f"{name} vector columns must match energies")
        validate_orthonormal_columns(
            vectors,
            chunk_columns=chunk_columns,
            orthogonality_samples=orthogonality_samples,
            tolerance=tolerance,
            name=name,
        )
    if vectors_ref.shape[0] != vectors_candidate.shape[0]:
        raise ValueError("reference and candidate vector row dimensions must match")

    z2_ref = np.asarray(reference_z2_sector_state)
    z2_candidate = np.asarray(candidate_z2_sector_state)
    for name, z2 in (("reference", z2_ref), ("candidate", z2_candidate)):
        if z2.ndim != 1 or z2.shape[0] != vectors_ref.shape[0]:
            raise ValueError(f"{name} Z2 state must match vector rows")
        if not np.all(np.isfinite(z2)):
            raise ValueError(f"{name} Z2 state must be finite")
    if not np.allclose(z2_ref, z2_candidate, atol=tolerance, rtol=0.0):
        raise ValueError("reference and candidate Z2 states must match")

    reference_boundaries = np.diff(energies_ref) > tolerance
    candidate_boundaries = np.diff(energies_candidate) > tolerance
    if not np.array_equal(reference_boundaries, candidate_boundaries):
        raise ValueError("reference and candidate degeneracy partitions must match")
    if not np.allclose(energies_ref, energies_candidate, atol=tolerance, rtol=0.0):
        raise ValueError("reference and candidate energies must agree within tolerance")

    overlap_ref = np.abs(vectors_ref.conj().T @ z2_ref) ** 2
    overlap_candidate = np.abs(vectors_candidate.conj().T @ z2_candidate) ** 2
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
    max_subspace_sine = 0.0
    max_pr2_diff_isolated = 0.0
    degenerate_groups = 0
    isolated_groups = 0

    for group in groups:
        total_z2_diff = abs(
            float(np.sum(overlap_ref[group]) - np.sum(overlap_candidate[group]))
        )
        max_total_z2_diff = max(max_total_z2_diff, total_z2_diff)

        projector_diag_ref = _projector_diagonal_chunked(
            vectors_ref,
            group,
            chunk_columns=chunk_columns,
        )
        projector_diag_candidate = _projector_diagonal_chunked(
            vectors_candidate,
            group,
            chunk_columns=chunk_columns,
        )
        projector_diag_diff = float(
            np.max(np.abs(projector_diag_ref - projector_diag_candidate))
        )
        max_projector_diag_diff = max(max_projector_diag_diff, projector_diag_diff)

        group_slice = slice(int(group[0]), int(group[-1]) + 1)
        subspace_sine = _two_way_subspace_residual(
            vectors_ref[:, group_slice],
            vectors_candidate[:, group_slice],
            chunk_columns=chunk_columns,
        )
        max_subspace_sine = max(max_subspace_sine, subspace_sine)

        if len(group) == 1:
            isolated_groups += 1
            pr2_diff = abs(float(pr2_ref[group[0]] - pr2_candidate[group[0]]))
            max_pr2_diff_isolated = max(max_pr2_diff_isolated, pr2_diff)
        else:
            degenerate_groups += 1

    return {
        "max_total_z2_diff": max_total_z2_diff,
        "max_projector_diag_diff": max_projector_diag_diff,
        "max_subspace_sine": max_subspace_sine,
        "max_pr2_diff_isolated": max_pr2_diff_isolated,
        "degenerate_group_count": degenerate_groups,
        "isolated_group_count": isolated_groups,
        "validation_chunk_columns": min(chunk_columns, vectors_ref.shape[1]),
        "subspace_chunk_columns": min(chunk_columns, vectors_ref.shape[1]),
        "orthogonality_sample_count": min(
            orthogonality_samples,
            vectors_ref.shape[1] * (vectors_ref.shape[1] - 1) // 2,
        ),
        "orthogonality_sampling_policy": ORTHOGONALITY_SAMPLING_POLICY,
        "orthogonality_pair_batch_size": min(
            chunk_columns,
            orthogonality_samples,
            vectors_ref.shape[1] * (vectors_ref.shape[1] - 1) // 2,
        ),
        "orthogonality_pair_metadata_memory": ORTHOGONALITY_PAIR_MEMORY_POLICY,
    }


def compute_observables(
    basis: OrbitBasis,
    reduced_hamiltonian: sp.csr_matrix | np.ndarray,
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    z2_pattern: int = 2,
    chunk_size: int = 262144,
    chunk_columns: int = 256,
    orthogonality_samples: int = 4096,
) -> dict[str, object]:
    """Compute observables from a validated Task 3 eigensystem without solving."""
    chunk_size = _validate_chunk_size(chunk_size)
    chunk_columns = positive_integer(chunk_columns, "chunk_columns")
    orthogonality_samples = positive_integer(
        orthogonality_samples, "orthogonality_samples"
    )
    representatives = np.asarray(basis.representatives)
    orbit_sizes = np.asarray(basis.orbit_sizes)
    constrained_states = np.asarray(basis.constrained_states)
    if (
        representatives.ndim != 1
        or orbit_sizes.ndim != 1
        or constrained_states.ndim != 1
        or representatives.size != orbit_sizes.size
    ):
        raise ValueError("basis arrays have inconsistent dimensions")

    reduced_dimension = representatives.size
    if reduced_hamiltonian.ndim != 2 or reduced_hamiltonian.shape != (
        reduced_dimension,
        reduced_dimension,
    ):
        raise ValueError("Hamiltonian dimensions must match the orbit basis")
    matrix_values = (
        reduced_hamiltonian.data
        if sp.issparse(reduced_hamiltonian)
        else np.asarray(reduced_hamiltonian)
    )
    if not np.all(np.isfinite(matrix_values)):
        raise ValueError("Hamiltonian values must be finite")
    hermiticity_error = validate_hermitian(
        reduced_hamiltonian,
        chunk_columns=chunk_columns,
        tolerance=1e-10,
    )

    energy_array = np.asarray(energies)
    if np.iscomplexobj(energy_array):
        raise ValueError("energies must be real")
    energy_array = np.asarray(energy_array, dtype=np.float64)
    if energy_array.ndim != 1 or energy_array.size != reduced_dimension:
        raise ValueError("energies must be one-dimensional and match the basis")
    if not np.all(np.isfinite(energy_array)):
        raise ValueError("energies must be finite")
    if np.any(np.diff(energy_array) < 0):
        raise ValueError("energies must be sorted")

    vector_array = np.asarray(eigenvectors)
    if vector_array.ndim != 2 or vector_array.shape != (
        reduced_dimension,
        reduced_dimension,
    ):
        raise ValueError("eigenvectors dimensions must match the basis and energies")
    validation = validate_orthonormal_columns(
        vector_array,
        chunk_columns=chunk_columns,
        orthogonality_samples=orthogonality_samples,
        tolerance=1e-10,
        name="eigenvectors",
    )
    residual_error = validate_eigenpair_residuals(
        reduced_hamiltonian,
        energy_array,
        vector_array,
        chunk_columns=chunk_columns,
        tolerance=1e-10,
    )

    initial_state = density_wave_state(basis.length, z2_pattern)
    z2_sector = project_product_state(
        basis,
        initial_state,
        chunk_size=chunk_size,
    )
    overlap_z2 = np.abs(vector_array.conj().T @ z2_sector) ** 2
    pr2 = stream_pr2(vector_array, chunk_columns=chunk_columns)

    streamed_fsa = stream_fsa_shells(
        basis,
        initial_state,
        reduced_hamiltonian=reduced_hamiltonian,
        chunk_size=chunk_size,
    )

    exact_shell_amplitudes = streamed_fsa.projected_shells @ vector_array
    return {
        "energies": energy_array,
        "eigenvectors": vector_array,
        "overlap_z2": np.asarray(overlap_z2, dtype=np.float64),
        "participation_ratio": np.asarray(pr2, dtype=np.float64),
        "exact_shell_amplitudes": np.asarray(exact_shell_amplitudes),
        "fsa_shell_vectors_sector": np.asarray(streamed_fsa.projected_shells, dtype=np.float64),
        "fsa_hamiltonian_sector": np.asarray(
            streamed_fsa.reduced_fsa_hamiltonian, dtype=np.float64
        ),
        "fsa_beta_full_chain": np.asarray(streamed_fsa.beta, dtype=np.float64),
        "validation_metadata": {
            "chunk_columns": min(chunk_columns, reduced_dimension),
            "finite_columns_checked": validation["columns_checked"],
            "norm_columns_checked": validation["columns_checked"],
            "residual_columns_checked": reduced_dimension,
            "orthogonality_sample_count": validation[
                "orthogonality_sample_count"
            ],
            "orthogonality_sampling_policy": validation[
                "orthogonality_sampling_policy"
            ],
            "orthogonality_pair_batch_size": validation[
                "orthogonality_pair_batch_size"
            ],
            "orthogonality_pair_metadata_memory": validation[
                "orthogonality_pair_metadata_memory"
            ],
            "max_norm_error": validation["max_norm_error"],
            "max_sample_overlap": validation["max_sample_overlap"],
            "max_hermiticity_error": hermiticity_error,
            "max_eigenpair_residual": residual_error,
        },
    }


def compute_observables_from_h5(
    basis: OrbitBasis,
    reduced_hamiltonian: sp.csr_matrix,
    energies: np.ndarray,
    vectors: object,
    *,
    chunk_size: int = 262144,
    chunk_columns: int = 256,
    orthogonality_samples: int = 4096,
) -> dict[str, object]:
    """Compute and validate observables while reading HDF5 vectors by columns."""
    chunk_size = _validate_chunk_size(chunk_size)
    chunk_columns = positive_integer(chunk_columns, "chunk_columns")
    orthogonality_samples = positive_integer(
        orthogonality_samples, "orthogonality_samples"
    )
    dimension = len(basis.representatives)
    energy_array = np.asarray(energies, dtype=np.float64)
    if energy_array.shape != (dimension,) or not np.all(np.isfinite(energy_array)):
        raise ValueError("energies must be finite and match the basis")
    if np.any(np.diff(energy_array) < 0):
        raise ValueError("energies must be sorted")
    if getattr(vectors, "shape", None) != (dimension, dimension):
        raise ValueError("eigenvector dataset shape must match the basis")

    hermiticity_error = validate_hermitian(
        reduced_hamiltonian,
        chunk_columns=chunk_columns,
        tolerance=1e-10,
    )
    initial_state = density_wave_state(basis.length, 2)
    z2_sector = project_product_state(basis, initial_state, chunk_size=chunk_size)
    streamed_fsa = stream_fsa_shells(
        basis,
        initial_state,
        reduced_hamiltonian=reduced_hamiltonian,
        chunk_size=chunk_size,
    )

    overlap_z2 = np.empty(dimension, dtype=np.float64)
    participation_ratio = np.empty(dimension, dtype=np.float64)
    exact_shell_amplitudes = np.empty(
        (streamed_fsa.projected_shells.shape[0], dimension),
        dtype=np.float64,
    )
    max_norm_error = 0.0
    max_residual = 0.0
    for start in range(0, dimension, chunk_columns):
        stop = min(start + chunk_columns, dimension)
        block = np.asarray(vectors[:, start:stop], dtype=np.float64)
        if not np.all(np.isfinite(block)):
            raise ValueError("eigenvectors must be finite")
        norms = np.sum(block * block, axis=0)
        max_norm_error = max(max_norm_error, float(np.max(np.abs(norms - 1.0))))
        residual = (
            reduced_hamiltonian @ block
            - block * energy_array[np.newaxis, start:stop]
        )
        max_residual = max(max_residual, float(np.max(np.abs(residual))))
        overlap_z2[start:stop] = np.abs(block.T @ z2_sector) ** 2
        participation_ratio[start:stop] = np.sum(block**4, axis=0)
        exact_shell_amplitudes[:, start:stop] = (
            streamed_fsa.projected_shells @ block
        )
    if max_norm_error > 1e-10:
        raise ValueError("eigenvector norms exceed tolerance")
    if max_residual > 1e-10:
        raise ValueError("eigenpair residual exceeds tolerance")

    max_sample_overlap = 0.0
    samples_checked = 0
    for left, right in sample_pair_batches(
        dimension,
        sample_count=orthogonality_samples,
        batch_size=chunk_columns,
    ):
        selected = np.unique(np.concatenate((left, right)))
        block = np.asarray(vectors[:, selected], dtype=np.float64)
        positions = {int(column): index for index, column in enumerate(selected)}
        overlaps = np.sum(
            block[:, [positions[int(column)] for column in left]]
            * block[:, [positions[int(column)] for column in right]],
            axis=0,
        )
        samples_checked += int(left.size)
        if overlaps.size:
            max_sample_overlap = max(
                max_sample_overlap,
                float(np.max(np.abs(overlaps))),
            )
    if max_sample_overlap > 1e-10:
        raise ValueError("eigenvector orthogonality exceeds tolerance")

    return {
        "overlap_z2": overlap_z2,
        "participation_ratio": participation_ratio,
        "exact_shell_amplitudes": exact_shell_amplitudes,
        "fsa_shell_vectors_sector": streamed_fsa.projected_shells,
        "fsa_hamiltonian_sector": streamed_fsa.reduced_fsa_hamiltonian,
        "fsa_beta_full_chain": streamed_fsa.beta,
        "validation_metadata": {
            "chunk_columns": min(chunk_columns, dimension),
            "finite_columns_checked": dimension,
            "norm_columns_checked": dimension,
            "residual_columns_checked": dimension,
            "orthogonality_sample_count": samples_checked,
            "orthogonality_sampling_policy": ORTHOGONALITY_SAMPLING_POLICY,
            "orthogonality_pair_batch_size": min(chunk_columns, samples_checked),
            "orthogonality_pair_metadata_memory": ORTHOGONALITY_PAIR_MEMORY_POLICY,
            "max_norm_error": max_norm_error,
            "max_sample_overlap": max_sample_overlap,
            "max_hermiticity_error": hermiticity_error,
            "max_eigenpair_residual": max_residual,
            "eigenvector_access": "column-chunked",
        },
    }
