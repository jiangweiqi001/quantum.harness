"""Readers for the official Turner et al. 2018 figure source data."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np

FIG4_EXPECTED_DIMENSIONS = {28: 13201, 30: 31836, 32: 77436}
FIG4_ENERGY_MEMBERS = {
    length: f"evals_periodic_N{length}_k0_p0.npy.txt"
    for length in FIG4_EXPECTED_DIMENSIONS
}
FIG4_HISTOGRAM_MEMBERS = {
    length: f"xydataL{length}.dat" for length in FIG4_EXPECTED_DIMENSIONS
}
FIG3_FSA_LENGTHS = frozenset({26, 32})
FSA_MATCH_TIE_RTOL = 1e-8
FSA_MATCH_TIE_ATOL = 1e-12


def _load_text_member(archive_path: Path, member: str) -> np.ndarray:
    with ZipFile(archive_path) as archive:
        return np.loadtxt(BytesIO(archive.read(member)))


def _load_h5_member(archive_path: Path, member: str) -> dict[str, np.ndarray]:
    with ZipFile(archive_path) as archive:
        payload = BytesIO(archive.read(member))
    with h5py.File(payload, "r") as handle:
        return {key: handle[key][...] for key in handle.keys()}


def load_fig2_entropy(
    data_dir: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    archive = Path(data_dir) / "entanglement_entropy_growth.zip"
    names = {
        "vacuum": "Ent_Z1.dat",
        "Z2": "Ent_Z2.dat",
        "Z3": "Ent_Z3.dat",
        "Z4": "Ent_Z4.dat",
    }
    curves = {}
    for label, member in names.items():
        data = _load_text_member(archive, member)
        curves[label] = (data[:, 0], data[:, 1])
    return curves


def load_fig3_overlap(
    data_dir: str | Path,
    *,
    length: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    amplitudes = _load_text_member(
        data_dir / "overlaps_with_Neel_state.zip",
        f"oneel_periodic_N{length}_k0_p0.dat",
    )
    if length >= 26:
        energies = _load_text_member(
            data_dir / "energy_eigenvalues.zip",
            f"evals_periodic_N{length}_k0_p0.npy.txt",
        )
    else:
        energies = _load_h5_member(
            data_dir / "eigendecomposition.zip",
            f"eigs_periodic_N{length}_k0_p0.h5",
        )["evals"]
    energies = np.asarray(energies)
    overlap = np.abs(amplitudes) ** 2
    if energies.shape != overlap.shape:
        raise ValueError(
            f"official L={length} energies and overlaps have different shapes: "
            f"{energies.shape} != {overlap.shape}"
        )
    return energies, overlap


def load_fig3_fsa(
    data_dir: str | Path,
    *,
    length: int = 32,
) -> dict[str, np.ndarray]:
    if length not in FIG3_FSA_LENGTHS:
        raise ValueError(
            f"official Fig. 3 FSA projections for L={length} are unavailable; "
            "the DOI archive contains only L=26 and L=32"
        )
    values = _load_h5_member(
        Path(data_dir) / "forward-scattering.zip",
        f"fscat_periodic_N{length}_k0_p0.h5",
    )
    beta = np.asarray(values["beta"])
    valid = np.isfinite(beta) & (beta > 1e-12)
    invalid = np.flatnonzero(~valid)
    prefix_length = int(invalid[0]) if len(invalid) else len(beta)
    if np.any(valid[prefix_length:]):
        raise ValueError("official FSA beta coefficients are not a contiguous prefix")
    dimension = prefix_length + 1
    alpha = np.asarray(values["alpha"][:dimension])
    beta = beta[: dimension - 1]
    matrix = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    energies, vectors = np.linalg.eigh(matrix)
    exact_shell_amplitudes = np.asarray(values["lanczos_ed"][:dimension, :])
    return {
        "energies": energies,
        "overlap": 0.5 * np.abs(vectors[0, :]) ** 2,
        "vectors": vectors,
        "exact_shell_amplitudes": exact_shell_amplitudes,
        "exact_shell_weights": np.abs(exact_shell_amplitudes) ** 2,
    }


def match_fsa_tower(
    exact_shell_amplitudes: np.ndarray,
    fsa_eigenvectors: np.ndarray,
) -> np.ndarray:
    """Match FSA states by unique maxima separated beyond 1e-8 relative tolerance.

    The absolute tolerance is 1e-12. The smallest DOI top-two gap is about
    0.223 (L=32), so these tolerances reject numerical near-ties without
    affecting the source-backed L=26/L=32 matches.
    """
    amplitudes = np.asarray(exact_shell_amplitudes)
    vectors = np.asarray(fsa_eigenvectors)
    if amplitudes.ndim != 2:
        raise ValueError("exact shell amplitudes must be a two-dimensional array")
    shell_count = amplitudes.shape[0]
    if vectors.shape != (shell_count, shell_count):
        raise ValueError(
            "FSA eigenvectors must be square and match the shell-amplitude rows"
        )
    projection = np.abs(amplitudes.T @ vectors) ** 2
    if not np.all(np.isfinite(projection)):
        raise ValueError("FSA-to-exact projection matrix must be finite")
    if projection.shape[0] > 1:
        top_two = np.partition(projection, -2, axis=0)[-2:]
        second, maximum = top_two
        tie_tolerance = FSA_MATCH_TIE_ATOL + FSA_MATCH_TIE_RTOL * np.maximum(
            np.abs(maximum), np.abs(second)
        )
        if np.any(maximum - second <= tie_tolerance):
            raise ValueError(
                "FSA-to-exact projection has a tied or near-tied column maximum"
            )
    matched = np.argmax(projection, axis=0)
    if len(np.unique(matched)) != shell_count:
        raise ValueError(
            "FSA-to-exact maximum-projection matches are not one-to-one"
        )
    return matched


def select_fig3_pr2_states(
    *,
    energies: np.ndarray,
    exact_shell_amplitudes: np.ndarray,
    fsa_eigenvectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select the thesis-defined special tower middle and its exact complement."""
    energies = np.asarray(energies)
    amplitudes = np.asarray(exact_shell_amplitudes)
    if energies.ndim != 1 or amplitudes.ndim != 2:
        raise ValueError("energies and exact shell amplitudes must be 1D and 2D")
    if amplitudes.shape[1] != len(energies):
        raise ValueError("exact shell amplitudes must have one column per energy")
    tower = match_fsa_tower(amplitudes, fsa_eigenvectors)
    sorted_tower = tower[np.argsort(energies[tower], kind="stable")]
    shell_count = amplitudes.shape[0]
    trim = shell_count // 6
    stop = shell_count - trim if trim else shell_count
    special = sorted_tower[trim:stop]
    nonzero = np.abs(energies) > 1e-10
    special = special[nonzero[special]]
    other_mask = nonzero.copy()
    other_mask[tower] = False
    return {
        "tower": tower,
        "sorted_tower": sorted_tower,
        "special": special,
        "other": np.flatnonzero(other_mask),
    }


def pr2_fsa_averages(
    *,
    energies: np.ndarray,
    pr2: np.ndarray,
    exact_shell_amplitudes: np.ndarray,
    fsa_eigenvectors: np.ndarray,
) -> tuple[float, float]:
    """Return unweighted PR2 means for other and special exact states."""
    pr2 = np.asarray(pr2)
    energies = np.asarray(energies)
    if pr2.shape != energies.shape:
        raise ValueError("energies and PR2 must have matching shapes")
    selected = select_fig3_pr2_states(
        energies=energies,
        exact_shell_amplitudes=exact_shell_amplitudes,
        fsa_eigenvectors=fsa_eigenvectors,
    )
    if not len(selected["special"]) or not len(selected["other"]):
        raise ValueError("special and other state sets must both be nonempty")
    return (
        float(np.mean(pr2[selected["other"]])),
        float(np.mean(pr2[selected["special"]])),
    )


def load_fig3_pr2_scaling(data_dir: str | Path) -> dict[str, np.ndarray]:
    archive_path = Path(data_dir) / "participation_ratios.zip"
    rows: list[tuple[int, float, float, bool]] = []
    with ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if re.fullmatch(r"pr_periodic_N\d+_k0_p0\.h5", name)
        ]
    for member in sorted(members, key=lambda name: int(re.search(r"N(\d+)", name)[1])):
        length = int(re.search(r"N(\d+)", member)[1])
        values = _load_h5_member(archive_path, member)
        available = length in FIG3_FSA_LENGTHS
        if available:
            fsa = load_fig3_fsa(data_dir, length=length)
            other, special = pr2_fsa_averages(
                energies=values["evals"],
                pr2=values["ipr_sum"],
                exact_shell_amplitudes=fsa["exact_shell_amplitudes"],
                fsa_eigenvectors=fsa["vectors"],
            )
        else:
            other = special = np.nan
        rows.append((length, other, special, available))
    return {
        "length": np.asarray([row[0] for row in rows], dtype=int),
        "other": np.asarray([row[1] for row in rows]),
        "special": np.asarray([row[2] for row in rows]),
        "available": np.asarray([row[3] for row in rows], dtype=bool),
    }


def load_fig4_histograms(
    data_dir: str | Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    archive = Path(data_dir) / "level_statistics.zip"
    histograms = {}
    for length, member in FIG4_HISTOGRAM_MEMBERS.items():
        data = _load_text_member(archive, member)
        histograms[length] = (data[:, 0], data[:, 1])
    return histograms


def load_fig4_energies(
    data_dir: str | Path,
    *,
    length: int,
) -> np.ndarray:
    """Load and sort an official Fig. 4 k=0, inversion-even spectrum."""
    if length not in FIG4_EXPECTED_DIMENSIONS:
        raise ValueError("official Fig. 4 energies are available only for L=28, 30, 32")
    energies = _load_text_member(
        Path(data_dir) / "energy_eigenvalues.zip",
        FIG4_ENERGY_MEMBERS[length],
    )
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise ValueError(f"official L={length} spectrum must be one-dimensional and finite")
    expected_dimension = FIG4_EXPECTED_DIMENSIONS[length]
    if len(energies) != expected_dimension:
        raise ValueError(
            f"official L={length} spectrum has {len(energies)} levels; "
            f"expected {expected_dimension}"
        )
    return np.sort(energies)
