import numpy as np
import pytest
import scipy.sparse as sp

from pxp_ed import (
    _reflect,
    _rotate,
    constrained_basis,
    pxp_hamiltonian,
    symmetry_basis_k0_inversion_even,
)
from turner2018_ed_engine import (
    OrbitBasis,
    assemble_reduced_hamiltonian,
    build_orbit_basis,
    canonical_dihedral,
    dihedral_members,
    enumerate_constrained_states,
)

EXPECTED_SECTOR_DIMENSIONS = {
    10: 14,
    12: 26,
    14: 49,
    16: 99,
    18: 209,
    20: 455,
}


def _scalar_rotate_left(value: int, shift: int, length: int) -> int:
    mask = (1 << length) - 1
    shift_mod = shift % length
    value_masked = value & mask
    return ((value_masked << shift_mod) | (value_masked >> (length - shift_mod))) & mask


def _scalar_reflect(value: int, length: int) -> int:
    reflected = 0
    for index in range(length):
        if (value >> index) & 1:
            reflected |= 1 << ((-index) % length)
    return reflected


def _scalar_dihedral_members(value: int, length: int) -> set[int]:
    reflected = _scalar_reflect(value, length)
    rotations = {_scalar_rotate_left(value, shift, length) for shift in range(length)}
    reflected_rotations = {
        _scalar_rotate_left(reflected, shift, length) for shift in range(length)
    }
    return rotations | reflected_rotations


def _reference_column_data(length: int):
    constrained = constrained_basis(length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(constrained, length).toarray()

    representatives: list[int] = []
    supports_by_representative: dict[int, set[int]] = {}
    orbit_sizes: list[int] = []
    for column in range(transform.shape[1]):
        support_indices = np.flatnonzero(np.abs(transform[:, column]) > 0)
        support_states = constrained[support_indices].astype(np.uint64)
        representative = int(np.min(support_states))
        orbit_size = int(len(support_states))
        expected_amplitude = 1.0 / np.sqrt(orbit_size)
        np.testing.assert_allclose(
            transform[support_indices, column],
            expected_amplitude,
            atol=1e-14,
        )
        representatives.append(representative)
        orbit_sizes.append(orbit_size)
        supports_by_representative[representative] = set(int(x) for x in support_states)

    gram = transform.T @ transform
    np.testing.assert_allclose(gram, np.eye(transform.shape[1]), atol=1e-14)
    return constrained, np.asarray(representatives), np.asarray(orbit_sizes), supports_by_representative


def max_abs_sparse(matrix: sp.spmatrix) -> float:
    if matrix.nnz == 0:
        return 0.0
    return float(np.max(np.abs(matrix.data)))


def _reference_reduced_hamiltonian(length: int) -> sp.csr_matrix:
    full_states = constrained_basis(length, pbc=True)
    full_h = pxp_hamiltonian(full_states, length, pbc=True)
    q = symmetry_basis_k0_inversion_even(full_states, length)
    return (q.T @ full_h @ q).tocsr()


@pytest.mark.parametrize("length", [10, 12, 14, 16])
def test_direct_reduced_hamiltonian_matches_qt_h_q(length):
    orbit = build_orbit_basis(length)
    actual = assemble_reduced_hamiltonian(orbit)
    expected = _reference_reduced_hamiltonian(length)
    difference = actual - expected
    assert max_abs_sparse(difference) <= 1e-11


@pytest.mark.parametrize("length", [10, 12, 14, 16, 18, 20])
def test_reduced_hamiltonian_is_symmetric_and_off_diagonal(length):
    matrix = assemble_reduced_hamiltonian(build_orbit_basis(length))
    assert max_abs_sparse(matrix - matrix.T) <= 1e-13
    assert np.count_nonzero(matrix.diagonal()) == 0


@pytest.mark.parametrize("length", [10, 12, 14, 16, 18, 20])
def test_reduced_hamiltonian_csr_layout_is_deterministic(length):
    first = assemble_reduced_hamiltonian(build_orbit_basis(length))
    second = assemble_reduced_hamiltonian(build_orbit_basis(length))

    assert first.shape == second.shape
    assert first.nnz == second.nnz
    assert first.data.tobytes() == second.data.tobytes()
    assert first.indices.tobytes() == second.indices.tobytes()
    assert first.indptr.tobytes() == second.indptr.tobytes()


@pytest.mark.parametrize("length", [10, 12, 14, 16, 18, 20])
def test_reduced_hamiltonian_dimensions_match_expected_sector_sizes(length):
    basis = build_orbit_basis(length)
    matrix = assemble_reduced_hamiltonian(basis)
    assert len(basis.constrained_states) == len(constrained_basis(length, pbc=True))
    assert len(basis.representatives) == EXPECTED_SECTOR_DIMENSIONS[length]
    assert matrix.shape == (
        EXPECTED_SECTOR_DIMENSIONS[length],
        EXPECTED_SECTOR_DIMENSIONS[length],
    )


@pytest.mark.parametrize("length", [18, 20])
def test_reduced_hamiltonian_matches_full_sparse_reference_l18_l20(length):
    matrix = assemble_reduced_hamiltonian(build_orbit_basis(length))
    expected = _reference_reduced_hamiltonian(length)
    difference = matrix - expected
    assert max_abs_sparse(difference) <= 1e-11


def test_scalar_reflect_matches_site_centered_reference_convention():
    cases = [
        (8, 0b10110000),
        (8, 0b00100101),
        (32, (1 << 31) | (1 << 30) | (1 << 9)),
        (32, (1 << 31) | (1 << 3) | 1),
    ]
    for length, state in cases:
        assert _scalar_reflect(state, length) == _reflect(state, length)


def test_compact_states_match_reference_at_l10():
    actual = enumerate_constrained_states(10)
    expected = constrained_basis(10, pbc=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint64


def test_dihedral_convention_matches_reference_rotation_and_reflection():
    state = 0b001001
    members = dihedral_members(state, 6)
    expected = {_rotate(state, shift, 6) for shift in range(6)} | {
        _rotate(_reflect(state, 6), shift, 6) for shift in range(6)
    }
    assert set(members.tolist()) == expected


def test_canonical_dihedral_maps_to_minimum_member():
    states = np.asarray([0b001001, 0b100100, 0b010010], dtype=np.uint64)
    canonical = canonical_dihedral(states, 6)
    assert canonical.dtype == np.uint64
    assert np.all(canonical == np.min(dihedral_members(int(states[0]), 6)))


def test_canonical_dihedral_matches_scalar_dihedral_minimum_l32_high_bits():
    length = 32
    states = np.asarray(
        [
            (1 << 31) | (1 << 3) | 1,
            (1 << 30) | (1 << 16) | (1 << 5),
            (1 << 31) | (1 << 30) | (1 << 9),
            (1 << 31) | (1 << 30) | (1 << 12),
        ],
        dtype=np.uint64,
    )

    expected = np.asarray(
        [min(_scalar_dihedral_members(int(state), length)) for state in states],
        dtype=np.uint64,
    )

    for state in states:
        rotations = {_scalar_rotate_left(int(state), shift, length) for shift in range(length)}
        reflected_rotations = {
            _scalar_rotate_left(_scalar_reflect(int(state), length), shift, length)
            for shift in range(length)
        }
        assert reflected_rotations - rotations

    canonical = canonical_dihedral(states, length)
    assert canonical.dtype == np.uint64
    np.testing.assert_array_equal(canonical, expected)


def test_dihedral_convention_includes_nontrivial_reflection_orbit_members():
    length = 8
    state = 0b10110000

    rotation_orbit = {_scalar_rotate_left(state, shift, length) for shift in range(length)}
    reflected_orbit = {
        _scalar_rotate_left(_scalar_reflect(state, length), shift, length)
        for shift in range(length)
    }
    assert reflected_orbit - rotation_orbit

    expected = rotation_orbit | reflected_orbit
    members = dihedral_members(state, length)
    assert set(members.tolist()) == expected


@pytest.mark.parametrize("length", [10, 12, 14])
def test_orbit_basis_columns_match_reference_transform(length):
    constrained, reference_representatives, reference_sizes, reference_supports = (
        _reference_column_data(length)
    )

    orbit_basis = build_orbit_basis(length, chunk_size=32)
    assert isinstance(orbit_basis, OrbitBasis)
    assert orbit_basis.length == length
    np.testing.assert_array_equal(orbit_basis.constrained_states, constrained)
    np.testing.assert_array_equal(orbit_basis.representatives, reference_representatives)
    np.testing.assert_array_equal(
        orbit_basis.orbit_sizes.astype(np.int64), reference_sizes.astype(np.int64)
    )
    assert orbit_basis.orbit_sizes.dtype == np.uint8

    for representative, orbit_size in zip(
        orbit_basis.representatives, orbit_basis.orbit_sizes, strict=True
    ):
        members = np.unique(dihedral_members(int(representative), length))
        assert len(members) == int(orbit_size)
        assert set(int(member) for member in members) == reference_supports[int(representative)]


def test_build_orbit_basis_is_invariant_to_chunk_boundaries():
    length = 14
    baseline = build_orbit_basis(length, chunk_size=262144)

    for chunk_size in (1, 2, 7, 31, 32, 33):
        basis = build_orbit_basis(length, chunk_size=chunk_size)
        np.testing.assert_array_equal(basis.constrained_states, baseline.constrained_states)
        np.testing.assert_array_equal(basis.representatives, baseline.representatives)
        np.testing.assert_array_equal(basis.orbit_sizes, baseline.orbit_sizes)


def test_orbit_basis_index_of_validates_representatives():
    basis = build_orbit_basis(10)
    indices = basis.index_of(basis.representatives[:5])
    np.testing.assert_array_equal(indices, np.arange(5))

    with pytest.raises(KeyError, match="outside this orbit basis"):
        basis.index_of(np.asarray([np.uint64((1 << 10) - 1)], dtype=np.uint64))
