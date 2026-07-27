"""Independent constrained basis and orbit metadata for Turner 2018 ED."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def _validate_length(length: int) -> None:
    if length < 1:
        raise ValueError("length must be positive")
    if length > 64:
        raise ValueError("length must be at most 64 for uint64 states")


def enumerate_constrained_states(length: int) -> np.ndarray:
    """Enumerate periodic blockade-allowed bitstrings in sorted uint64 order."""
    _validate_length(length)

    def count(site: int, previous_occupied: bool, first_occupied: bool) -> int:
        if site == length:
            return 0 if (previous_occupied and first_occupied) else 1
        total = count(site + 1, False, first_occupied)
        if not previous_occupied:
            total += count(site + 1, True, True if site == 0 else first_occupied)
        return total

    total_states = count(0, False, False)
    states = np.empty(total_states, dtype=np.uint64)
    cursor = 0

    def write(
        site: int,
        state: np.uint64,
        previous_occupied: bool,
        first_occupied: bool,
    ) -> None:
        nonlocal cursor
        if site == length:
            if not (previous_occupied and first_occupied):
                states[cursor] = state
                cursor += 1
            return

        write(site + 1, state, False, first_occupied)
        if not previous_occupied:
            write(
                site + 1,
                state | (np.uint64(1) << np.uint64(site)),
                True,
                True if site == 0 else first_occupied,
            )

    write(0, np.uint64(0), False, False)
    states.sort()

    assert states.ndim == 1
    assert states.dtype == np.uint64
    assert np.all(states[1:] > states[:-1])
    return states


def _bitmask(length: int) -> np.uint64:
    if length == 64:
        return np.uint64(np.iinfo(np.uint64).max)
    return np.uint64((1 << length) - 1)


def _rotate_scalar(state: int, shift: int, length: int) -> np.uint64:
    mask = int(_bitmask(length))
    shift %= length
    if shift == 0:
        return np.uint64(state & mask)
    return np.uint64((((state << shift) & mask) | (state >> (length - shift))) & mask)


def _rotate_array(states: np.ndarray, shift: int, length: int) -> np.ndarray:
    shift %= length
    mask = _bitmask(length)
    rotated = states & mask
    if shift == 0:
        return rotated
    return ((states << np.uint64(shift)) & mask) | (states >> np.uint64(length - shift))


def _reflect_array(states: np.ndarray, length: int) -> np.ndarray:
    reflected = np.zeros(states.shape, dtype=np.uint64)
    for site in range(length):
        reflected |= ((states >> np.uint64(site)) & np.uint64(1)) << np.uint64((-site) % length)
    return reflected


def dihedral_members(state: int, length: int) -> np.ndarray:
    """Return sorted unique members of the D_L orbit of one state."""
    _validate_length(length)
    base = np.uint64(state)
    reflected = _reflect_array(np.asarray([base], dtype=np.uint64), length)[0]
    members = np.empty(2 * length, dtype=np.uint64)
    for shift in range(length):
        members[shift] = _rotate_scalar(int(base), shift, length)
        members[length + shift] = _rotate_scalar(int(reflected), shift, length)
    return np.unique(members)


def canonical_dihedral(states: np.ndarray, length: int) -> np.ndarray:
    """Canonical representative (minimum D_L image) for each state."""
    _validate_length(length)
    vector = np.asarray(states, dtype=np.uint64)
    if vector.ndim != 1:
        raise ValueError("states must be a one-dimensional array")
    if vector.size == 0:
        return vector.copy()

    canonical = vector.copy()
    for shift in range(length):
        canonical = np.minimum(canonical, _rotate_array(vector, shift, length))

    reflected = _reflect_array(vector, length)
    for shift in range(length):
        canonical = np.minimum(canonical, _rotate_array(reflected, shift, length))
    return canonical


@dataclass(frozen=True)
class OrbitBasis:
    length: int
    constrained_states: np.ndarray
    representatives: np.ndarray
    orbit_sizes: np.ndarray

    def index_of(self, representatives: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.representatives, representatives)
        if np.any(positions == len(self.representatives)):
            raise KeyError("representative is outside this orbit basis")
        if np.any(self.representatives[positions] != representatives):
            raise KeyError("representative is outside this orbit basis")
        return positions


def build_orbit_basis(length: int, chunk_size: int = 262144) -> OrbitBasis:
    """Construct sorted dihedral representatives and orbit-size metadata."""
    _validate_length(length)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    constrained_states = enumerate_constrained_states(length)
    canonical = np.empty_like(constrained_states)
    for start in range(0, len(constrained_states), chunk_size):
        stop = min(start + chunk_size, len(constrained_states))
        canonical[start:stop] = canonical_dihedral(constrained_states[start:stop], length)

    keep = constrained_states == canonical
    representatives = constrained_states[keep]
    orbit_sizes = np.empty(len(representatives), dtype=np.uint8)
    for index, representative in enumerate(representatives):
        orbit_sizes[index] = np.uint8(len(dihedral_members(int(representative), length)))

    return OrbitBasis(
        length=length,
        constrained_states=constrained_states,
        representatives=representatives,
        orbit_sizes=orbit_sizes,
    )


def assemble_reduced_hamiltonian(basis: OrbitBasis) -> sp.csr_matrix:
    """Assemble the direct reduced PXP Hamiltonian in the k=0, inversion-even basis."""
    length = basis.length
    _validate_length(length)

    representatives = np.asarray(basis.representatives, dtype=np.uint64)
    orbit_sizes = np.asarray(basis.orbit_sizes, dtype=np.float64)
    dimension = len(representatives)

    max_entries = dimension * length
    rows = np.empty(max_entries, dtype=np.int64)
    columns = np.empty(max_entries, dtype=np.int64)
    values = np.empty(max_entries, dtype=np.float64)
    cursor = 0

    for source_index, source_representative in enumerate(representatives):
        source_state = int(source_representative)
        destination_counts: dict[int, int] = {}

        for site in range(length):
            left = (site - 1) % length
            right = (site + 1) % length
            left_empty = ((source_state >> left) & 1) == 0
            right_empty = ((source_state >> right) & 1) == 0
            if not (left_empty and right_empty):
                continue

            flipped_state = np.uint64(source_state ^ (1 << site))
            destination_representative = int(
                canonical_dihedral(
                    np.asarray([flipped_state], dtype=np.uint64),
                    length,
                )[0]
            )
            destination_counts[destination_representative] = (
                destination_counts.get(destination_representative, 0) + 1
            )

        if not destination_counts:
            continue

        destination_representatives = np.asarray(
            sorted(destination_counts.keys()),
            dtype=np.uint64,
        )
        destination_indices = basis.index_of(destination_representatives)
        source_orbit_size = orbit_sizes[source_index]

        for destination_representative, destination_index in zip(
            destination_representatives, destination_indices, strict=True
        ):
            multiplicity = destination_counts[int(destination_representative)]
            destination_orbit_size = orbit_sizes[destination_index]
            value = multiplicity * np.sqrt(source_orbit_size / destination_orbit_size)
            rows[cursor] = int(destination_index)
            columns[cursor] = source_index
            values[cursor] = float(value)
            cursor += 1

    reduced = sp.coo_matrix(
        (values[:cursor], (rows[:cursor], columns[:cursor])),
        shape=(dimension, dimension),
    )
    reduced_csr = reduced.tocsr()
    reduced_csr.sum_duplicates()
    reduced_csr.eliminate_zeros()
    reduced_csr.sort_indices()
    return reduced_csr
