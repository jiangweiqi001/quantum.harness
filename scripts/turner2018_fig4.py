#!/usr/bin/env python3
"""Level statistics for the PXP k=0, inversion-even symmetry sector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from pxp_ed import (
    constrained_basis,
    pxp_hamiltonian,
    symmetry_basis_k0_inversion_even,
)
from turner2018_official import (
    FIG4_ENERGY_MEMBERS,
    FIG4_HISTOGRAM_MEMBERS,
    load_fig4_energies,
    load_fig4_histograms,
)

DEFAULT_OUTPUT = Path("tracks/ed/results/turner-2018/fig4")
DEFAULT_OFFICIAL_DATA = Path(".external/official-data/turner-2018")
PAPER_LENGTHS = (28, 30, 32)
PAPER_HISTOGRAM_EDGES = np.linspace(0.0, 4.8, 25)
PAPER_HISTOGRAM_CENTERS = np.arange(0.1, 4.8, 0.2)
PAPER_UNFOLDING_DEGREE = 3
PAPER_EDGE_TRIM = 50
OFFICIAL_DENSITY_TOLERANCE = 1e-12
FIG4_EXACT_SCHEMA_VERSION = "turner-fig4-exact-metrics-v1"
FIG4_EXACT_GENERATOR = {
    "name": "quantum.harness.turner2018_fig4",
    "version": "1.0.0",
}
PAPER_LENGTH_COLORS = {28: "tab:blue", 30: "tab:orange", 32: "tab:green"}
THEORY_CURVE_STYLES = {
    "poisson": {"color": "0.45", "linestyle": "--", "linewidth": 1.2},
    "semi-poisson": {"color": "0.55", "linestyle": ":", "linewidth": 1.2},
    "goe": {"color": "0.35", "linestyle": "-.", "linewidth": 1.2},
}
THEORY_CURVE_LABELS = {
    "poisson": "Poisson",
    "semi-poisson": "Semi-Poisson",
    "goe": "Wigner-Dyson (GOE)",
}


def adjacent_gap_ratios(
    energies: np.ndarray,
    *,
    degeneracy_tolerance: float = 1e-12,
) -> np.ndarray:
    """Folded adjacent-gap ratios without joining levels across degeneracies."""
    spacings = np.diff(np.sort(np.asarray(energies, dtype=float)))
    if len(spacings) < 2:
        return np.asarray([], dtype=float)
    valid = (spacings[:-1] > degeneracy_tolerance) & (
        spacings[1:] > degeneracy_tolerance
    )
    left = spacings[:-1][valid]
    right = spacings[1:][valid]
    return np.minimum(left, right) / np.maximum(left, right)


def unfold_spectrum(energies: np.ndarray, *, degree: int = 5) -> np.ndarray:
    """Polynomial unfolding of the integrated density of states."""
    energies = np.sort(np.asarray(energies, dtype=float))
    if len(energies) <= degree:
        raise ValueError("spectrum must contain more levels than polynomial degree")
    staircase = np.arange(len(energies), dtype=float)
    fit = np.polynomial.Polynomial.fit(energies, staircase, deg=degree)
    unfolded = fit(energies)
    if np.any(np.diff(unfolded) <= 0):
        raise ValueError("polynomial unfolding is not monotonic on this window")
    unfolded = (unfolded - unfolded[0]) / np.mean(np.diff(unfolded))
    return unfolded


def paper_exact_level_statistics(energies: np.ndarray) -> dict[str, np.ndarray | int]:
    """Reconstruct the paper's Fig. 4 histogram without spacing renormalization."""
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1:
        raise ValueError("paper-exact spectrum must be one-dimensional")
    if not np.all(np.isfinite(energies)):
        raise ValueError("paper-exact spectrum must contain only finite values")
    if np.any(np.diff(energies) < 0):
        raise ValueError("paper-exact spectrum must be sorted in nondecreasing order")

    dimension = len(energies)
    lower = dimension // 5
    upper = dimension // 2 - 500
    window = energies[lower:upper]
    minimum_window = 2 * PAPER_EDGE_TRIM + 2
    if len(window) < minimum_window:
        raise ValueError(
            "paper window must contain at least "
            f"{minimum_window} levels before edge trimming"
        )

    staircase = np.arange(len(window), dtype=float)
    coefficients = np.polyfit(window, staircase, deg=PAPER_UNFOLDING_DEGREE)
    unfolded = np.polyval(coefficients, window)
    trimmed = unfolded[PAPER_EDGE_TRIM:-PAPER_EDGE_TRIM]
    spacings = np.diff(trimmed)
    if not np.all(np.isfinite(spacings)):
        raise ValueError("paper-exact unfolding produced non-finite spacings")
    if np.any(spacings <= 0):
        raise ValueError("paper-exact cubic unfolding is not strictly monotonic")

    histogram_counts, histogram_edges = np.histogram(
        spacings, bins=PAPER_HISTOGRAM_EDGES
    )
    histogram_density, _ = np.histogram(
        spacings, bins=PAPER_HISTOGRAM_EDGES, density=True
    )
    return {
        "original_dimension": dimension,
        "window_energies": window,
        "polynomial_coefficients": coefficients,
        "unfolded": unfolded,
        "trimmed_unfolded": trimmed,
        "spacings": spacings,
        "histogram_edges": histogram_edges,
        "histogram_centers": PAPER_HISTOGRAM_CENTERS.copy(),
        "histogram_counts": histogram_counts,
        "histogram_density": histogram_density,
    }


def compare_official_histogram(
    reconstruction: dict[str, np.ndarray | int],
    official_xy: np.ndarray,
) -> dict[str, float | int]:
    """Validate official xydata structure and quantify density mismatch."""
    official_xy = np.asarray(official_xy, dtype=float)
    if official_xy.shape != (25, 2):
        raise ValueError("official Fig. 4 xydata must have shape (25, 2)")
    if not np.array_equal(official_xy[0], np.array([0.0, 0.0])):
        raise ValueError("official Fig. 4 xydata must begin with sentinel (0, 0)")
    if not np.allclose(
        official_xy[1:, 0],
        PAPER_HISTOGRAM_CENTERS,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("official Fig. 4 x values do not match histogram centers")

    density = np.asarray(reconstruction["histogram_density"])
    if density.shape != (24,):
        raise ValueError("reconstructed histogram density must have 24 bins")
    official_density = official_xy[:, 1]
    if not np.all(np.isfinite(official_density)):
        raise ValueError("official Fig. 4 densities must be finite")
    if np.any(official_density < 0):
        raise ValueError("official Fig. 4 densities must be nonnegative")
    if not np.all(np.isfinite(density)):
        raise ValueError("reconstructed Fig. 4 densities must be finite")
    if np.any(density < 0):
        raise ValueError("reconstructed Fig. 4 densities must be nonnegative")
    mismatch = density - official_density[1:]
    if not np.all(np.isfinite(mismatch)):
        raise ValueError("official Fig. 4 density mismatches must be finite")
    max_abs_mismatch = float(np.max(np.abs(mismatch)))
    if max_abs_mismatch > OFFICIAL_DENSITY_TOLERANCE:
        raise ValueError(
            "official Fig. 4 density mismatch exceeds tolerance "
            f"{OFFICIAL_DENSITY_TOLERANCE:.1e}: {max_abs_mismatch:.17g}"
        )
    counts = np.asarray(reconstruction["histogram_counts"])
    spacings = np.asarray(reconstruction["spacings"])
    return {
        "window_size": int(len(np.asarray(reconstruction["window_energies"]))),
        "spacing_count": int(len(spacings)),
        "in_range_spacing_count": int(np.sum(counts)),
        "mean_spacing": float(np.mean(spacings)),
        "max_abs_density_mismatch": max_abs_mismatch,
        "rms_density_mismatch": float(np.sqrt(np.mean(mismatch**2))),
    }


def theoretical_spacing(spacing: np.ndarray, distribution: str) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=float)
    if distribution == "poisson":
        return np.exp(-spacing)
    if distribution == "semi-poisson":
        return 4.0 * spacing * np.exp(-2.0 * spacing)
    if distribution == "goe":
        return 0.5 * np.pi * spacing * np.exp(-0.25 * np.pi * spacing**2)
    raise ValueError(f"unknown spacing distribution: {distribution}")


def _paper_or_small_system_window(energies: np.ndarray) -> tuple[np.ndarray, str]:
    energies = np.sort(np.asarray(energies, dtype=float))
    dimension = len(energies)
    lower = dimension // 5
    upper = dimension // 2 - 500
    if upper > lower:
        return energies[lower:upper], "paper"
    lower = max(1, dimension // 10)
    upper = dimension // 2 - max(1, dimension // 20)
    return energies[lower:upper], "finite-size-bulk"


def analyze_level_statistics(length: int) -> dict[str, np.ndarray | str | float]:
    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(basis, length)
    reduced = (transform.T @ hamiltonian @ transform).toarray()
    energies = np.linalg.eigvalsh(reduced)
    window, window_kind = _paper_or_small_system_window(energies)
    window = window[np.abs(window) > 1e-10]
    ratios = adjacent_gap_ratios(window)
    unfolded = unfold_spectrum(window, degree=min(5, len(window) - 1))
    return {
        "energies": energies,
        "window_energies": window,
        "gap_ratios": ratios,
        "mean_gap_ratio": float(np.mean(ratios)),
        "unfolded": unfolded,
        "spacings": np.diff(unfolded),
        "window_kind": window_kind,
    }


def run_figure(
    length: int,
    output_dir: str | Path = DEFAULT_OUTPUT,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_level_statistics(length)
    np.savez(output_dir / f"fig4_L{length}.npz", **result)

    spacing = np.asarray(result["spacings"])
    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    axis.hist(
        spacing,
        bins=35,
        range=(0, 4),
        density=True,
        alpha=0.35,
        color="tab:blue",
        label=f"our ED L={length}",
    )
    if official_data_dir is not None and (
        Path(official_data_dir) / "level_statistics.zip"
    ).is_file():
        for official_length, (official_spacing, official_density) in (
            load_fig4_histograms(official_data_dir).items()
        ):
            axis.plot(
                official_spacing,
                official_density,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label=f"official ED L={official_length}",
            )
    grid = np.linspace(0, 4, 500)
    for distribution, label in (
        ("poisson", "Poisson"),
        ("semi-poisson", "Semi-Poisson"),
        ("goe", "Wigner-Dyson (GOE)"),
    ):
        axis.plot(grid, theoretical_spacing(grid, distribution), label=label)
    axis.set_xlabel("unfolded spacing s")
    axis.set_ylabel("P(s)")
    axis.set_title(
        f"L={length}, mean r={result['mean_gap_ratio']:.3f}, "
        f"window={result['window_kind']}"
    )
    axis.legend()
    figure.tight_layout()
    path = output_dir / f"fig4_L{length}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _archive_provenance(path: Path, members: list[str]) -> dict[str, object]:
    with path.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "resolved_path": str(path.resolve()),
        "sha256": sha256,
        "byte_size": path.stat().st_size,
        "members": members,
    }


def _plot_theory_reference_curves(axis: plt.Axes, *, grid: np.ndarray | None = None) -> None:
    if grid is None:
        grid = np.linspace(0.0, 4.8, 500)
    for distribution in ("poisson", "semi-poisson", "goe"):
        axis.plot(
            grid,
            theoretical_spacing(grid, distribution),
            label=THEORY_CURVE_LABELS[distribution],
            **THEORY_CURVE_STYLES[distribution],
        )


def _plot_reconstructed_and_official(
    axis: plt.Axes,
    length: int,
    reconstruction: dict[str, np.ndarray | int],
    official_xy: np.ndarray,
    *,
    show_legend: bool = True,
) -> None:
    color = PAPER_LENGTH_COLORS[length]
    axis.step(
        PAPER_HISTOGRAM_CENTERS,
        np.asarray(reconstruction["histogram_density"]),
        where="mid",
        linewidth=1.8,
        color=color,
        label=f"L={length} reconstructed",
    )
    axis.plot(
        official_xy[1:, 0],
        official_xy[1:, 1],
        linestyle="none",
        marker="o",
        markersize=3.2,
        markerfacecolor="none",
        markeredgecolor=color,
        color=color,
        label=f"L={length} official",
    )
    if show_legend:
        axis.legend(fontsize=8)


def save_comparison_figures(
    reconstructions: dict[int, dict[str, np.ndarray | int]],
    official_xy_by_length: dict[int, np.ndarray],
    metrics: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write one all-sizes overview and three size-specific comparison PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 4.8, 500)
    paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    for length in PAPER_LENGTHS:
        _plot_reconstructed_and_official(
            axis,
            length,
            reconstructions[length],
            official_xy_by_length[length],
            show_legend=False,
        )
    _plot_theory_reference_curves(axis, grid=grid)
    axis.set_xlim(0.0, 4.8)
    axis.set_xlabel("unfolded spacing s")
    axis.set_ylabel("P(s)")
    axis.set_title("Turner et al. (2018) Fig. 4 — all sizes")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    paths["figure_all"] = output_dir / "fig4_comparison_all.png"
    figure.savefig(paths["figure_all"], dpi=180)
    plt.close(figure)

    for length in PAPER_LENGTHS:
        figure, axis = plt.subplots(figsize=(6.4, 4.5))
        _plot_reconstructed_and_official(
            axis,
            length,
            reconstructions[length],
            official_xy_by_length[length],
            show_legend=False,
        )
        _plot_theory_reference_curves(axis, grid=grid)
        mismatch = metrics["lengths"][str(length)]["max_abs_density_mismatch"]
        axis.set_xlim(0.0, 4.8)
        axis.set_xlabel("unfolded spacing s")
        axis.set_ylabel("P(s)")
        axis.set_title(f"L={length}\nmax |ΔP|={mismatch:.2e}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        paths[f"figure_L{length}"] = output_dir / f"fig4_comparison_L{length}.png"
        figure.savefig(paths[f"figure_L{length}"], dpi=180)
        plt.close(figure)

    return paths


def run_paper_exact_reconstruction(
    official_data_dir: str | Path = DEFAULT_OFFICIAL_DATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Path]:
    """Recompute and compare all official L=28/30/32 Fig. 4 distributions."""
    official_data_dir = Path(official_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_histograms = load_fig4_histograms(official_data_dir)
    energy_archive = official_data_dir / "energy_eigenvalues.zip"
    histogram_archive = official_data_dir / "level_statistics.zip"

    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {
        "schema_version": FIG4_EXACT_SCHEMA_VERSION,
        "generator": FIG4_EXACT_GENERATOR,
        "invocation": {
            "lengths": list(PAPER_LENGTHS),
            "official_data_dir": str(official_data_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "density_mismatch_tolerance": OFFICIAL_DENSITY_TOLERANCE,
            "unfolding_degree": PAPER_UNFOLDING_DEGREE,
            "edge_trim_levels": PAPER_EDGE_TRIM,
            "spacing_renormalized": False,
            "histogram_edges": PAPER_HISTOGRAM_EDGES.tolist(),
            "histogram_density": True,
        },
        "source_archives": {
            "energy_eigenvalues": _archive_provenance(
                energy_archive,
                [FIG4_ENERGY_MEMBERS[length] for length in PAPER_LENGTHS],
            ),
            "level_statistics": _archive_provenance(
                histogram_archive,
                [FIG4_HISTOGRAM_MEMBERS[length] for length in PAPER_LENGTHS],
            ),
        },
        "method": {
            "window": "sorted_energies[D//5:D//2-500]",
            "unfolding_degree": PAPER_UNFOLDING_DEGREE,
            "edge_trim_levels": PAPER_EDGE_TRIM,
            "spacing_renormalized": False,
            "histogram_edges": PAPER_HISTOGRAM_EDGES.tolist(),
            "density": True,
        },
        "lengths": {},
    }
    reconstructions: dict[int, dict[str, np.ndarray | int]] = {}
    official_xy_by_length: dict[int, np.ndarray] = {}
    for length in PAPER_LENGTHS:
        energies = load_fig4_energies(official_data_dir, length=length)
        reconstruction = paper_exact_level_statistics(energies)
        official_x, official_density = official_histograms[length]
        official_xy = np.column_stack((official_x, official_density))
        comparison = compare_official_histogram(reconstruction, official_xy)
        comparison["original_dimension"] = int(
            reconstruction["original_dimension"]
        )
        metrics["lengths"][str(length)] = comparison
        reconstructions[length] = reconstruction
        official_xy_by_length[length] = official_xy

        prefix = f"L{length}_"
        arrays[prefix + "energies"] = energies
        for key, value in reconstruction.items():
            arrays[prefix + key] = np.asarray(value)
        arrays[prefix + "official_xy"] = official_xy

    arrays_path = output_dir / "fig4_paper_exact.npz"
    np.savez_compressed(arrays_path, **arrays)
    metrics_path = output_dir / "fig4_paper_exact_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    for axis, length in zip(axes, PAPER_LENGTHS, strict=True):
        reconstruction = reconstructions[length]
        official_xy = official_xy_by_length[length]
        axis.step(
            PAPER_HISTOGRAM_CENTERS,
            np.asarray(reconstruction["histogram_density"]),
            where="mid",
            linewidth=1.8,
            label="reconstructed from energies",
        )
        axis.plot(
            official_xy[1:, 0],
            official_xy[1:, 1],
            linestyle="none",
            marker="o",
            markersize=3.2,
            markerfacecolor="none",
            label="official xydata",
        )
        mismatch = metrics["lengths"][str(length)]["max_abs_density_mismatch"]
        axis.set_title(f"L={length}\nmax |ΔP|={mismatch:.2e}")
        axis.set_xlabel("unfolded spacing s")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("P(s)")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Turner et al. (2018) Fig. 4 — paper-exact reconstruction\n"
        "cubic unfolding; 50 levels trimmed per edge; no mean-spacing rescaling"
    )
    figure.tight_layout()
    figure_path = output_dir / "fig4_paper_exact.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    comparison_paths = save_comparison_figures(
        reconstructions,
        official_xy_by_length,
        metrics,
        output_dir,
    )
    return {
        "figure": figure_path,
        "arrays": arrays_path,
        "metrics": metrics_path,
        **comparison_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-data-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_DATA,
    )
    parser.add_argument(
        "--paper-exact",
        action="store_true",
        help="reconstruct all official L=28/30/32 histograms from source energies",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.paper_exact:
        paths = run_paper_exact_reconstruction(
            args.official_data_dir,
            args.output_dir,
        )
        for path in paths.values():
            print(path)
        return 0
    path = run_figure(args.length, args.output_dir, args.official_data_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
