#!/usr/bin/env python3
"""Finite-size overlap, FSA, and participation-ratio analysis for Fig. 3."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy.sparse as sp

from pxp_ed import (
    basis_state_vector,
    constrained_basis,
    density_wave_state,
    pxp_hamiltonian,
    symmetry_basis_k0_inversion_even,
)
from turner2018_official import (
    load_fig3_fsa,
    load_fig3_overlap,
    load_fig3_pr2_scaling,
    pr2_fsa_averages,
    select_fig3_pr2_states,
)

DEFAULT_OUTPUT = Path("tracks/ed/results/turner-2018/fig3")
DEFAULT_OFFICIAL_DATA = Path(".external/official-data/turner-2018")


def participation_ratio(coefficients: np.ndarray) -> np.ndarray | float:
    """Paper's second participation ratio PR2=sum_i |c_i|^4."""
    values = np.sum(np.abs(coefficients) ** 4, axis=0)
    return float(values) if np.ndim(values) == 0 else values


def fsa_basis(
    hamiltonian: sp.csr_matrix,
    basis: np.ndarray,
    initial_state: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build forward-scattering vectors on successive Hamming-distance shells."""
    distances = np.asarray(
        [(int(state) ^ initial_state).bit_count() for state in basis], dtype=int
    )
    coo = hamiltonian.tocoo()
    forward = distances[coo.row] == distances[coo.col] + 1
    h_plus = sp.coo_matrix(
        (coo.data[forward], (coo.row[forward], coo.col[forward])),
        shape=hamiltonian.shape,
    ).tocsr()

    current = basis_state_vector(basis, initial_state)
    vectors = [current]
    beta: list[float] = []
    for _ in range(length):
        candidate = h_plus @ current
        norm = float(np.linalg.norm(candidate))
        if norm < 1e-14:
            raise RuntimeError("forward-scattering chain terminated before distance L")
        beta.append(norm)
        current = candidate / norm
        vectors.append(current)
    return np.asarray(vectors), np.asarray(beta)


def fsa_hamiltonian(beta: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(beta) + 1, len(beta) + 1), dtype=float)
    indices = np.arange(len(beta))
    matrix[indices, indices + 1] = beta
    matrix[indices + 1, indices] = beta
    return matrix


def analyze_spectrum(length: int) -> dict[str, np.ndarray]:
    """Dense small-system spectrum in the k=0, inversion-even PBC sector."""
    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(basis, length)
    reduced = (transform.T @ hamiltonian @ transform).toarray()
    energies, eigenvectors = np.linalg.eigh(reduced)
    z2_state = density_wave_state(length, 2)
    z2 = basis_state_vector(basis, z2_state)
    z2_sector = np.asarray(transform.T @ z2).ravel()
    overlap_z2 = np.abs(eigenvectors.conj().T @ z2_sector) ** 2
    vectors, beta = fsa_basis(hamiltonian, basis, z2_state, length)
    projected_shells = np.asarray(transform.T @ vectors.T).T[: length // 2 + 1]
    projected_shells /= np.linalg.norm(projected_shells, axis=1)[:, None]
    projected_fsa = projected_shells @ reduced @ projected_shells.T
    projected_fsa = 0.5 * (projected_fsa + projected_fsa.T)
    fsa_energies, fsa_eigenvectors = np.linalg.eigh(projected_fsa)
    fsa_z2_overlap = 0.5 * np.abs(fsa_eigenvectors[0, :]) ** 2
    exact_shell_amplitudes = projected_shells @ eigenvectors
    return {
        "raw_basis_states": basis,
        "sector_dimension": np.asarray([transform.shape[1]]),
        "energies": energies,
        "overlap_z2": overlap_z2,
        "participation_ratio": participation_ratio(eigenvectors),
        "fsa_energies": fsa_energies,
        "fsa_overlap_z2": fsa_z2_overlap,
        "fsa_eigenvectors": fsa_eigenvectors,
        "exact_shell_amplitudes": exact_shell_amplitudes,
        "fsa_shell_vectors_sector": projected_shells,
        "fsa_hamiltonian_sector": projected_fsa,
        "fsa_beta_full_chain": beta,
    }


def plot_official_pr2_scaling(panel, scaling: dict[str, np.ndarray]) -> tuple:
    """Plot only available DOI PR2 sizes, without bridging missing sizes."""
    available = scaling["available"]
    other_line, = panel.semilogy(
        scaling["length"][available],
        scaling["other"][available],
        marker="o",
        linestyle="none",
        color="tab:orange",
        label="DOI matched other (L=26,32)",
    )
    special_line, = panel.semilogy(
        scaling["length"][available],
        scaling["special"][available],
        marker="s",
        linestyle="none",
        color="tab:red",
        label="DOI matched FSA special (L=26,32)",
    )
    return other_line, special_line


def run_figure(
    length: int,
    output_dir: str | Path = DEFAULT_OUTPUT,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_spectrum(length)
    np.savez(output_dir / f"fig3_L{length}.npz", **result)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    panel_a, panel_b, panel_c, panel_d = axes.ravel()
    panel_a.scatter(
        result["energies"],
        result["overlap_z2"],
        s=9,
        alpha=0.5,
        label=f"ED L={length}",
    )
    panel_a.scatter(
        result["fsa_energies"],
        result["fsa_overlap_z2"],
        marker="x",
        color="tab:blue",
        label=f"FSA L={length}",
    )
    official_available = (
        official_data_dir is not None
        and (Path(official_data_dir) / "energy_eigenvalues.zip").is_file()
        and (Path(official_data_dir) / "forward-scattering.zip").is_file()
        and (Path(official_data_dir) / "overlaps_with_Neel_state.zip").is_file()
        and (Path(official_data_dir) / "participation_ratios.zip").is_file()
    )
    if official_available:
        official_energies, official_overlap = load_fig3_overlap(
            official_data_dir, length=32
        )
        official_fsa = load_fig3_fsa(official_data_dir, length=32)
        official_scaling = load_fig3_pr2_scaling(official_data_dir)
        panel_a.scatter(
            official_energies,
            official_overlap,
            s=2,
            alpha=0.3,
            color="tab:orange",
            rasterized=True,
            label="official ED L=32",
        )
        panel_a.scatter(
            official_fsa["energies"],
            official_fsa["overlap"],
            marker="x",
            color="black",
            label="official FSA L=32",
        )

        official_selected = select_fig3_pr2_states(
            energies=official_energies,
            exact_shell_amplitudes=official_fsa["exact_shell_amplitudes"],
            fsa_eigenvectors=official_fsa["vectors"],
        )
        special_index = official_selected["special"][
            np.argmin(np.abs(official_energies[official_selected["special"]]))
        ]
        state_indices = (official_selected["sorted_tower"][0], special_index)
        titles = ("matched tower ground state", "matched special state near E=0")
        for panel, state_index, title in zip(
            (panel_b, panel_c), state_indices, titles
        ):
            target_energy = official_energies[state_index]
            fsa_index = int(np.flatnonzero(
                official_selected["tower"] == state_index
            )[0])
            shell = np.arange(official_fsa["exact_shell_weights"].shape[0])
            panel.plot(
                shell,
                official_fsa["exact_shell_weights"][:, state_index],
                "o-",
                markersize=3,
                label="official exact",
            )
            panel.plot(
                shell,
                np.abs(official_fsa["vectors"][:, fsa_index]) ** 2,
                "x--",
                color="black",
                label="official FSA",
            )
            panel.set_title(f"{title}, E={target_energy:.2f}")
            panel.set_xlabel("FSA shell n")
            panel.set_ylabel("shell weight")
            panel.legend(fontsize=8)

        plot_official_pr2_scaling(panel_d, official_scaling)
        panel_d.text(
            0.02,
            0.08,
            "DOI FSA projections unavailable at L=28,30",
            transform=panel_d.transAxes,
            fontsize=7,
        )
    else:
        panel_b.text(0.5, 0.5, "official FSA data unavailable", ha="center")
        panel_c.text(0.5, 0.5, "official FSA data unavailable", ha="center")

    local_other, local_special = pr2_fsa_averages(
        energies=result["energies"],
        pr2=result["participation_ratio"],
        exact_shell_amplitudes=result["exact_shell_amplitudes"],
        fsa_eigenvectors=result["fsa_eigenvectors"],
    )
    panel_d.semilogy(
        [length],
        [local_other],
        "o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="tab:blue",
        label=f"our matched other L={length}",
    )
    panel_d.semilogy(
        [length],
        [local_special],
        "s",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="tab:blue",
        label=f"our matched FSA special L={length}",
    )
    panel_a.set_xlabel("energy")
    panel_a.set_ylabel(r"$|\langle E|Z_2\rangle|^2$")
    panel_a.legend(fontsize=8)
    panel_d.set_xlabel("system size L")
    panel_d.set_ylabel(r"$PR_2=\sum_\alpha |c_\alpha|^4$")
    panel_d.legend(fontsize=8)
    for label, panel in zip(("a", "b", "c", "d"), axes.ravel()):
        panel.text(
            0.02,
            0.96,
            f"({label})",
            transform=panel.transAxes,
            va="top",
            fontweight="bold",
        )
    figure.tight_layout()
    path = output_dir / f"fig3_L{length}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-data-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_DATA,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_figure(args.length, args.output_dir, args.official_data_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
