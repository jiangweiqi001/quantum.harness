from __future__ import annotations

import numpy as np
from quspin.basis import spinless_fermion_basis_1d
from quspin.operators import hamiltonian


def build_ssh_hamiltonian(L: int, t1: float, t2: float):
    basis = spinless_fermion_basis_1d(L, Nf=1)
    hoppings = []
    for bond in range(L - 1):
        coeff = -(t1 if bond % 2 == 0 else t2)
        hoppings.append([coeff, bond, bond + 1])
        hoppings.append([coeff, bond + 1, bond])
    static = [["+-", hoppings]]
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)
    return basis, H


def diagonalize_ssh(L: int = 8, t1: float = 0.6, t2: float = 1.0):
    basis, H = build_ssh_hamiltonian(L, t1, t2)
    energies, vectors = H.eigh()
    matrix = H.toarray()
    hermiticity_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    particle_hole_error = float(np.max(np.abs(energies + energies[::-1])))
    edge_mode_indices = sorted(np.argsort(np.abs(energies))[:2].tolist())
    edge_weights = (np.abs(vectors[0, :]) ** 2 + np.abs(vectors[-1, :]) ** 2).tolist()
    diagnostics = {
        "basis_dimension": basis.Ns,
        "hermiticity_error": hermiticity_error,
        "particle_hole_error": particle_hole_error,
        "edge_mode_indices": edge_mode_indices,
        "edge_weights": edge_weights,
    }
    if basis.Ns != L or hermiticity_error >= 1e-12 or particle_hole_error >= 1e-12:
        raise RuntimeError("SSH diagnostic failed")
    return energies, vectors, diagnostics
