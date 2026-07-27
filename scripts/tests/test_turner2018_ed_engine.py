import numpy as np
import pytest

from pxp_ed import _reflect, _rotate, constrained_basis, symmetry_basis_k0_inversion_even
from turner2018_ed_engine import (
    OrbitBasis,
    build_orbit_basis,
    canonical_dihedral,
    dihedral_members,
    enumerate_constrained_states,
)


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


def test_orbit_basis_index_of_validates_representatives():
    basis = build_orbit_basis(10)
    indices = basis.index_of(basis.representatives[:5])
    np.testing.assert_array_equal(indices, np.arange(5))

    with pytest.raises(KeyError, match="outside this orbit basis"):
        basis.index_of(np.asarray([np.uint64((1 << 10) - 1)], dtype=np.uint64))
