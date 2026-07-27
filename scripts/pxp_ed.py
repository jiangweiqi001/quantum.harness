#!/usr/bin/env python3
"""Numerical building blocks for finite-size PXP exact diagonalization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp


def lucas_number(length: int) -> int:
    """Return the Lucas number L_length with L_0=2 and L_1=1."""
    if length < 0:
        raise ValueError("length must be non-negative")
    a, b = 2, 1
    for _ in range(length):
        a, b = b, a + b
    return a


def is_blockaded(state: int, length: int, *, pbc: bool = True) -> bool:
    """Whether a bitstring has no adjacent excitations."""
    if length < 1 or state < 0 or state >= 1 << length:
        return False
    if state & (state << 1):
        return False
    return not (pbc and (state & 1) and (state & (1 << (length - 1))))


def constrained_basis(length: int, *, pbc: bool = True) -> np.ndarray:
    """Generate sorted blockade-allowed bitstrings without scanning 2**L states."""
    if length < 1:
        raise ValueError("length must be positive")
    states: list[int] = []

    def visit(site: int, state: int, previous_occupied: bool) -> None:
        if site == length:
            if not pbc or not ((state & 1) and (state & (1 << (length - 1)))):
                states.append(state)
            return
        visit(site + 1, state, False)
        if not previous_occupied:
            visit(site + 1, state | (1 << site), True)

    visit(0, 0, False)
    states.sort()
    return np.asarray(states, dtype=np.uint64)


def _rotate(state: int, shift: int, length: int) -> int:
    mask = (1 << length) - 1
    shift %= length
    return ((state << shift) & mask) | (state >> (length - shift))


def _reflect(state: int, length: int) -> int:
    reflected = 0
    for site in range(length):
        if (state >> site) & 1:
            reflected |= 1 << ((-site) % length)
    return reflected


def _translation_orbit(state: int, length: int) -> tuple[int, ...]:
    return tuple(sorted({_rotate(state, shift, length) for shift in range(length)}))


def symmetry_basis_k0_inversion_even(
    basis: np.ndarray,
    length: int,
) -> sp.csc_matrix:
    """Map the k=0, inversion-even orbit basis into the bitstring basis."""
    states = [int(state) for state in basis]
    state_index = {state: index for index, state in enumerate(states)}
    orbit_by_representative: dict[int, tuple[int, ...]] = {}
    representative_for_state: dict[int, int] = {}
    for state in states:
        if state in representative_for_state:
            continue
        orbit = _translation_orbit(state, length)
        representative = min(orbit)
        orbit_by_representative[representative] = orbit
        for member in orbit:
            representative_for_state[member] = representative

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    visited: set[int] = set()
    column = 0
    for representative in sorted(orbit_by_representative):
        if representative in visited:
            continue
        partner = representative_for_state[_reflect(representative, length)]
        orbit = orbit_by_representative[representative]
        partner_orbit = orbit_by_representative[partner]
        members = orbit if partner == representative else orbit + partner_orbit
        normalization = np.sqrt(len(members))
        for member in members:
            rows.append(state_index[member])
            columns.append(column)
            values.append(1.0 / normalization)
        visited.update((representative, partner))
        column += 1

    return sp.coo_matrix(
        (values, (rows, columns)), shape=(len(basis), column)
    ).tocsc()


def pxp_hamiltonian(
    basis: Sequence[int] | np.ndarray,
    length: int,
    *,
    pbc: bool = True,
) -> sp.csr_matrix:
    """Construct H=sum_j P_(j-1) X_j P_(j+1) in a constrained basis."""
    states = np.asarray(basis, dtype=np.uint64)
    state_index = {int(state): index for index, state in enumerate(states)}
    rows: list[int] = []
    columns: list[int] = []

    for column, raw_state in enumerate(states):
        state = int(raw_state)
        for site in range(length):
            left = (site - 1) % length
            right = (site + 1) % length
            if not pbc:
                left_empty = site == 0 or not (state >> left) & 1
                right_empty = site == length - 1 or not (state >> right) & 1
            else:
                left_empty = not (state >> left) & 1
                right_empty = not (state >> right) & 1
            if not (left_empty and right_empty):
                continue
            flipped = state ^ (1 << site)
            row = state_index.get(flipped)
            if row is not None:
                rows.append(row)
                columns.append(column)

    data = np.ones(len(rows), dtype=float)
    return sp.coo_matrix(
        (data, (rows, columns)), shape=(len(states), len(states))
    ).tocsr()


def density_wave_state(length: int, period: int, *, offset: int = 0) -> int:
    """Return the period-p density-wave bitstring on a periodic ring."""
    if period < 2:
        raise ValueError("period must be at least two")
    state = 0
    for site in range(offset % period, length, period):
        state |= 1 << site
    if not is_blockaded(state, length, pbc=True):
        raise ValueError("density wave violates the periodic blockade")
    return state


def basis_state_vector(
    basis: Sequence[int] | np.ndarray,
    state: int,
    *,
    dtype: type = complex,
) -> np.ndarray:
    """Represent one product state in the constrained basis."""
    states = np.asarray(basis)
    matches = np.flatnonzero(states == state)
    if len(matches) != 1:
        raise ValueError(f"state {state} is not present exactly once in the basis")
    vector = np.zeros(len(states), dtype=dtype)
    vector[matches[0]] = 1
    return vector
