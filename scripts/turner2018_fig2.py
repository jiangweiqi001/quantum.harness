#!/usr/bin/env python3
"""Finite-size ED reproduction of Turner et al. (2018), Fig. 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.sparse.linalg import expm_multiply

from pxp_ed import (
    basis_state_vector,
    constrained_basis,
    density_wave_state,
    pxp_hamiltonian,
)
from turner2018_official import load_fig2_entropy

DEFAULT_OUTPUT = Path("tracks/ed/results/turner-2018/fig2")
DEFAULT_OFFICIAL_DATA = Path(".external/official-data/turner-2018")
DEFAULT_OFFICIAL_CORRELATION = Path(
    ".external/official-data/turner-2018/corzz_Neel.dat"
)


def half_chain_entropy(
    state_vector: np.ndarray,
    basis: np.ndarray,
    length: int,
) -> float:
    """Von Neumann entropy across the central bond, using natural logarithms."""
    left_length = length // 2
    right_length = length - left_length
    coefficient_matrix = np.zeros(
        (1 << right_length, 1 << left_length), dtype=complex
    )
    left_mask = (1 << left_length) - 1
    for amplitude, raw_state in zip(state_vector, basis):
        state = int(raw_state)
        coefficient_matrix[state >> left_length, state & left_mask] = amplitude
    singular_values = np.linalg.svd(coefficient_matrix, compute_uv=False)
    probabilities = singular_values**2
    probabilities = probabilities[probabilities > 1e-14]
    return float(-np.sum(probabilities * np.log(probabilities)))


def _zz_diagonal(basis: np.ndarray, length: int) -> np.ndarray:
    diagonal = np.empty(len(basis), dtype=float)
    for index, raw_state in enumerate(basis):
        state = int(raw_state)
        total = 0.0
        for site in range(length):
            z_site = 1 - 2 * ((state >> site) & 1)
            z_next = 1 - 2 * ((state >> ((site + 1) % length)) & 1)
            total += z_site * z_next
        diagonal[index] = total / length
    return diagonal


def _evolve(
    hamiltonian,
    initial: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    if len(times) == 1:
        return np.asarray([expm_multiply(-1j * times[0] * hamiltonian, initial)])
    steps = np.diff(times)
    if times[0] == 0 and np.allclose(steps, steps[0]):
        return expm_multiply(
            -1j * hamiltonian,
            initial,
            start=float(times[0]),
            stop=float(times[-1]),
            num=len(times),
            endpoint=True,
        )
    return np.asarray(
        [expm_multiply(-1j * float(time) * hamiltonian, initial) for time in times]
    )


def evolve_observables(
    *,
    length: int,
    initial_period: int | None,
    times: np.ndarray,
    compute_entropy: bool = True,
) -> dict[str, np.ndarray]:
    """Evolve a density wave, or the vacuum when initial_period is None."""
    times = np.asarray(times, dtype=float)
    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    product_state = (
        0 if initial_period is None else density_wave_state(length, initial_period)
    )
    initial = basis_state_vector(basis, product_state)
    evolved = _evolve(hamiltonian, initial, times)
    zz_diagonal = _zz_diagonal(basis, length)
    fidelity = np.abs(evolved @ initial.conj()) ** 2
    zz = np.sum(np.abs(evolved) ** 2 * zz_diagonal[None, :], axis=1).real
    result = {"time": times, "fidelity": fidelity.real, "zz": zz}
    if compute_entropy:
        result["entropy"] = np.asarray(
            [half_chain_entropy(vector, basis, length) for vector in evolved]
        )
    return result


def load_official_correlation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def detrend_linear(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Subtract the least-squares linear background used to expose oscillations."""
    slope, intercept = np.polyfit(time, values, deg=1)
    return values - (slope * time + intercept)


def run_figure(
    length: int,
    t_max: float,
    dt: float,
    output_dir: str | Path = DEFAULT_OUTPUT,
    official_correlation_path: str | Path | None = DEFAULT_OFFICIAL_CORRELATION,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> Path:
    if length % 12:
        raise ValueError(
            "Figure 2 comparison length must be divisible by 12 so Z2, Z3, "
            "and Z4 are all commensurate with the periodic ring"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"fig2_L{length}_comparison.json"
    comparison_path.unlink(missing_ok=True)
    times = np.arange(0.0, t_max + 0.5 * dt, dt)
    periods = {"vacuum": None, "Z2": 2, "Z3": 3, "Z4": 4}
    results = {
        label: evolve_observables(
            length=length,
            initial_period=period,
            times=times,
            compute_entropy=True,
        )
        for label, period in periods.items()
    }
    z2 = results["Z2"]
    arrays = {"time": times}
    for label, result in results.items():
        prefix = label.lower()
        for observable, values in result.items():
            if observable != "time":
                arrays[f"{prefix}_{observable}"] = values
        arrays[f"{prefix}_entropy_per_cut"] = 0.5 * result["entropy"]
    np.savez(output_dir / f"fig2_L{length}.npz", **arrays)

    colors = {"vacuum": "tab:gray", "Z2": "tab:blue", "Z3": "tab:orange", "Z4": "tab:purple"}
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 8.2))
    comparison: dict[str, float | int] = {}
    official_entropy = {}
    if official_data_dir is not None:
        entropy_archive = Path(official_data_dir) / "entanglement_entropy_growth.zip"
        if entropy_archive.is_file():
            official_entropy = load_fig2_entropy(official_data_dir)
    for label, result in results.items():
        entropy_per_cut = 0.5 * result["entropy"]
        axes[0].plot(
            times,
            entropy_per_cut,
            color=colors[label],
            label=f"ED {label}",
        )
        if label in official_entropy:
            official_time, official_values = official_entropy[label]
            selected = (official_time >= times[0]) & (official_time <= times[-1])
            official_time = official_time[selected]
            official_values = official_values[selected]
            if len(official_time):
                axes[0].plot(
                    official_time,
                    official_values,
                    color=colors[label],
                    linestyle="--",
                    linewidth=1.2,
                    label=f"official {label}",
                )
                interpolated = np.interp(official_time, times, entropy_per_cut)
                comparison[f"{label.lower()}_entropy_rmse"] = float(
                    np.sqrt(np.mean((interpolated - official_values) ** 2))
                )
    axes[1].plot(
        times,
        detrend_linear(times, 0.5 * z2["entropy"]),
        color=colors["Z2"],
        label="ED Z2",
    )
    if "Z2" in official_entropy:
        official_time, official_values = official_entropy["Z2"]
        selected = (official_time >= times[0]) & (official_time <= times[-1])
        official_time = official_time[selected]
        official_values = official_values[selected]
        if len(official_time):
            axes[1].plot(
                official_time,
                detrend_linear(official_time, official_values),
                color="tab:green",
                linestyle="--",
                linewidth=1.5,
                label="official Z2",
            )
    axes[2].plot(times, z2["zz"], color=colors["Z2"], label="ED Z2")
    if official_correlation_path is not None and Path(
        official_correlation_path
    ).is_file():
        official_time, official_zz = load_official_correlation(
            official_correlation_path
        )
        selected = (official_time >= times[0]) & (official_time <= times[-1])
        official_time = official_time[selected]
        official_zz = official_zz[selected]
        if len(official_time):
            interpolated = np.interp(official_time, times, z2["zz"])
            comparison["zz_official_points"] = int(len(official_time))
            comparison["zz_rmse"] = float(
                np.sqrt(np.mean((interpolated - official_zz) ** 2))
            )
            axes[2].plot(
                official_time,
                official_zz,
                color="tab:green",
                linestyle="--",
                marker="o",
                markersize=2.5,
                markevery=max(1, len(official_time) // 30),
                linewidth=1.4,
                zorder=3,
                label="official iTEBD",
            )
    if comparison:
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
    axes[0].set_ylabel("entanglement entropy per cut S")
    axes[1].set_ylabel(r"$\Delta S$ (linear trend removed)")
    axes[2].set_ylabel(r"$\langle Z_i Z_{i+1}\rangle$")
    axes[2].set_xlabel("time")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].legend()
    axes[2].legend()
    figure.tight_layout()
    path = output_dir / f"fig2_L{length}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--t-max", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-correlation",
        type=Path,
        default=DEFAULT_OFFICIAL_CORRELATION,
    )
    parser.add_argument(
        "--official-data-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_DATA,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_figure(
        args.length,
        args.t_max,
        args.dt,
        args.output_dir,
        args.official_correlation,
        args.official_data_dir,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
