#!/usr/bin/env python3
"""Level statistics for the PXP k=0, inversion-even symmetry sector."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid
from zipfile import BadZipFile, ZipFile

import h5py
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
from turner2018_ed_artifacts import SCHEMA_VERSION as ED_ARTIFACT_SCHEMA_VERSION
from turner2018_fig3 import (
    EXPECTED_MODEL,
    INDEPENDENT_SOURCE,
    OFFICIAL_SOURCE,
    _fsync_file,
    _fsync_parent,
    _hash_open_file,
    _publish_generation,
    _read_open_file,
    _scientific_dimensions,
    _sha256,
    _unlink_durable,
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
FIG4_INDEPENDENT_SCHEMA_VERSION = "turner2018-independent-fig4-v1"
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


def _zip_has_member(path: Path, member: str) -> bool:
    if not path.is_file():
        return False
    try:
        with ZipFile(path) as archive:
            return member in archive.namelist()
    except BadZipFile:
        return False


def _load_independent_fig4_length(directory: Path) -> dict[str, Any]:
    """Load one validated energy vector from stable open Task 6 handles."""
    from turner2018_l32_server import (
        _fingerprint_sha256,
        build_execution_fingerprint,
        require_stage,
    )

    manifest_stages = (
        "plan",
        "basis",
        "hamiltonian",
        "diagonalize",
        "observables",
        "validate",
    )
    with ExitStack() as stack:
        plan_handle = stack.enter_context((directory / "manifest.json").open("rb"))
        validation_handle = stack.enter_context(
            (directory / "validation" / "metrics.json").open("rb")
        )
        basis_handle = stack.enter_context((directory / "basis.npz").open("rb"))
        hamiltonian_handle = stack.enter_context(
            (directory / "hamiltonian.csr.npz").open("rb")
        )
        eigensystem_handle = stack.enter_context(
            (directory / "eigensystem.h5").open("rb")
        )
        observables_handle = stack.enter_context(
            (directory / "observables.h5").open("rb")
        )
        stage_handles = {
            stage: stack.enter_context(
                (directory / "stages" / f"{stage}.json").open("rb")
            )
            for stage in manifest_stages
        }
        stage_bytes = {
            stage: _read_open_file(handle)
            for stage, handle in stage_handles.items()
        }
        stage_hashes = {
            stage: hashlib.sha256(payload).hexdigest()
            for stage, payload in stage_bytes.items()
        }
        plan_bytes = _read_open_file(plan_handle)
        validation_bytes = _read_open_file(validation_handle)

        cache: dict[str, dict[str, Any]] = {}
        require_stage(directory, "plan", cache)
        validate_manifest = require_stage(directory, "validate", cache)
        for stage, payload in stage_bytes.items():
            if json.loads(payload) != cache[stage]:
                raise RuntimeError(
                    f"stage manifest changed during validation: stage={stage}"
                )
        diagonalize_manifest = cache["diagonalize"]
        observables_manifest = cache["observables"]
        open_artifacts = {
            "manifest.json": (plan_handle, cache["plan"]["artifact"]["sha256"]),
            "basis.npz": (basis_handle, cache["basis"]["artifact"]["sha256"]),
            "hamiltonian.csr.npz": (
                hamiltonian_handle,
                cache["hamiltonian"]["artifact"]["sha256"],
            ),
            "eigensystem.h5": (
                eigensystem_handle,
                diagonalize_manifest["artifact"]["sha256"],
            ),
            "observables.h5": (
                observables_handle,
                observables_manifest["artifact"]["sha256"],
            ),
            "validation_metrics": (
                validation_handle,
                validate_manifest["artifact"]["sha256"],
            ),
        }
        for name, (handle, expected_hash) in open_artifacts.items():
            if _hash_open_file(handle) != expected_hash:
                raise RuntimeError(
                    f"validated {name} bytes do not match the open file handle"
                )

        plan = json.loads(plan_bytes)
        current_fingerprint = build_execution_fingerprint()
        if (
            plan.get("execution_fingerprint") != current_fingerprint
            or plan.get("execution_fingerprint_sha256")
            != _fingerprint_sha256(current_fingerprint)
        ):
            raise RuntimeError("stored plan execution fingerprint is not current")
        model = plan.get("model", {})
        if (
            model.get("hamiltonian") != EXPECTED_MODEL
            or model.get("boundary") != "periodic"
            or model.get("momentum") != 0
            or model.get("inversion") != "even"
        ):
            raise RuntimeError(
                f"scientific model attributes are invalid in {directory}"
            )
        length = model.get("length")
        if not isinstance(length, int) or length < 4 or length % 2:
            raise RuntimeError(f"scientific length is invalid in {directory}")

        validation = json.loads(validation_bytes)
        checks = validation.get("metrics")
        if (
            validation.get("status") != "passed"
            or validation.get("passed") is not True
            or validation.get("length") != length
            or not isinstance(checks, dict)
            or not checks
            or any(check.get("passed") is not True for check in checks.values())
        ):
            raise RuntimeError(f"scientific validation did not pass for L={length}")
        validated_hashes = validation.get("validated_stage_sha256", {})
        for stage in ("basis", "hamiltonian", "diagonalize", "observables"):
            if validated_hashes.get(stage) != cache[stage]["artifact"]["sha256"]:
                raise RuntimeError(
                    f"validation source hash does not match stage={stage} for L={length}"
                )
        full_dimension, sector_dimension = _scientific_dimensions(
            directory, validation, plan
        )

        eigensystem_handle.seek(0)
        with h5py.File(eigensystem_handle, "r") as handle:
            if handle.attrs.get("schema_version") != ED_ARTIFACT_SCHEMA_VERSION:
                raise RuntimeError(f"eigensystem HDF5 schema is invalid for L={length}")
            if set(handle.keys()) != {"eigensystem"}:
                raise RuntimeError(f"eigensystem HDF5 groups are invalid for L={length}")
            group = handle["eigensystem"]
            if set(group.keys()) != {"energies", "vectors"}:
                raise RuntimeError(
                    f"eigensystem dataset set is invalid for L={length}"
                )
            energies_dataset = group["energies"]
            vectors = group["vectors"]
            if (
                energies_dataset.shape != (sector_dimension,)
                or energies_dataset.dtype != np.float64
            ):
                raise RuntimeError(
                    f"energy dataset metadata is invalid for L={length}"
                )
            if (
                vectors.shape != (sector_dimension, sector_dimension)
                or vectors.dtype != np.float64
                or vectors.chunks != (sector_dimension, 1)
            ):
                raise RuntimeError(
                    f"eigenvector metadata is invalid for L={length}"
                )
            energies = energies_dataset[()]
            energy_metadata = {
                "shape": list(energies_dataset.shape),
                "dtype": str(energies_dataset.dtype),
                "access": "full-energy-vector-only",
            }
            eigenvector_metadata = {
                "shape": list(vectors.shape),
                "dtype": str(vectors.dtype),
                "chunks": list(vectors.chunks or ()),
                "access": "metadata-only",
            }

        observables_handle.seek(0)
        with h5py.File(observables_handle, "r") as handle:
            if handle.attrs.get("schema_version") != ED_ARTIFACT_SCHEMA_VERSION:
                raise RuntimeError(f"observables HDF5 schema is invalid for L={length}")

        stable_hashes = {
            name: _hash_open_file(handle)
            for name, (handle, _expected) in open_artifacts.items()
        }
        stable_hashes.update(
            {
                f"{stage}_manifest": _hash_open_file(handle)
                for stage, handle in stage_handles.items()
            }
        )
        for name, (_handle, expected_hash) in open_artifacts.items():
            if stable_hashes[name] != expected_hash:
                raise RuntimeError(f"{name} changed while its snapshot was consumed")
        for stage in manifest_stages:
            if stable_hashes[f"{stage}_manifest"] != stage_hashes[stage]:
                raise RuntimeError(
                    f"stage manifest changed while consumed: stage={stage}"
                )

    if (
        energies.shape != (sector_dimension,)
        or not np.all(np.isfinite(energies))
        or np.any(np.diff(energies) < 0)
    ):
        raise RuntimeError(f"energy array is invalid for L={length}")
    return {
        "source": INDEPENDENT_SOURCE,
        "directory": directory,
        "length": length,
        "full_dimension": full_dimension,
        "sector_dimension": sector_dimension,
        "energies": energies,
        "source_hashes": {
            **stable_hashes,
            "execution_fingerprint": plan["execution_fingerprint_sha256"],
        },
        "energy_dataset_metadata": energy_metadata,
        "eigenvector_metadata": eigenvector_metadata,
    }


def load_independent_fig4_results(
    root: str | Path,
) -> dict[int, dict[str, Any]]:
    """Discover and validate independent Task 6 spectra below one root."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"independent results root is not a directory: {root}")
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise RuntimeError(f"no independent Task 6 manifests found below {root}")
    results: dict[int, dict[str, Any]] = {}
    for manifest in manifests:
        result = _load_independent_fig4_length(manifest.parent)
        length = result["length"]
        if length in results:
            raise RuntimeError(f"duplicate independent result for L={length}")
        results[length] = result
    return dict(sorted(results.items()))


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


def _load_available_official_histograms(
    official_data_dir: str | Path | None,
    lengths: list[int],
) -> dict[int, np.ndarray]:
    if official_data_dir is None:
        return {}
    root = Path(official_data_dir)
    archive = root / "level_statistics.zip"
    available = [
        length
        for length in lengths
        if length in FIG4_HISTOGRAM_MEMBERS
        and _zip_has_member(archive, FIG4_HISTOGRAM_MEMBERS[length])
    ]
    if not available:
        return {}
    loaded = load_fig4_histograms(root)
    return {
        length: np.column_stack(loaded[length])
        for length in available
    }


def _independent_statistics(
    energies: np.ndarray,
) -> tuple[dict[str, np.ndarray | int] | None, dict[str, Any]]:
    dimension = len(energies)
    lower = dimension // 5
    upper = dimension // 2 - 500
    window = energies[lower:upper]
    minimum_window = 2 * PAPER_EDGE_TRIM + 2
    convention = {
        "window_expression": "sorted_energies[D//5:D//2-500]",
        "window_bounds": [lower, upper],
        "window_count": int(len(window)),
        "unfolding_degree": PAPER_UNFOLDING_DEGREE,
        "edge_trim_levels": PAPER_EDGE_TRIM,
        "zero_mode_convention": (
            "no additional zero-energy filter; the negative-energy paper "
            "window excludes the central zero modes"
        ),
        "spacing_renormalized": False,
    }
    if len(window) < minimum_window:
        return None, {
            **convention,
            "statistics_available": False,
            "unavailable_reason": (
                f"paper window has {len(window)} levels; at least "
                f"{minimum_window} are required for the exact edge trim"
            ),
        }
    statistics = paper_exact_level_statistics(energies)
    return statistics, {
        **convention,
        "statistics_available": True,
        "unavailable_reason": None,
    }


def _official_mismatch(
    statistics: dict[str, np.ndarray | int] | None,
    official_xy: np.ndarray | None,
) -> dict[str, float | int] | None:
    if statistics is None or official_xy is None:
        return None
    official_xy = np.asarray(official_xy, dtype=float)
    if official_xy.shape != (25, 2):
        raise RuntimeError("official Fig. 4 overlay must have shape (25, 2)")
    if not np.array_equal(official_xy[0], np.array([0.0, 0.0])):
        raise RuntimeError("official Fig. 4 overlay has an invalid sentinel")
    if not np.allclose(
        official_xy[1:, 0],
        PAPER_HISTOGRAM_CENTERS,
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("official Fig. 4 overlay has invalid bin centers")
    official_density = official_xy[1:, 1]
    independent_density = np.asarray(statistics["histogram_density"])
    if (
        not np.all(np.isfinite(official_density))
        or np.any(official_density < 0)
    ):
        raise RuntimeError("official Fig. 4 overlay densities are invalid")
    delta = independent_density - official_density
    return {
        "max_abs_density": float(np.max(np.abs(delta))),
        "rms_density": float(np.sqrt(np.mean(delta**2))),
        "compared_bin_count": int(len(delta)),
    }


def _write_png_partial(partial: Path, figure: Any) -> None:
    figure.savefig(partial, dpi=180, format="png")
    _fsync_file(partial)


def _write_json_partial(partial: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_partial(partial: Path, arrays: dict[str, np.ndarray]) -> None:
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_figure_generation(
    path: Path,
    figure: Any,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    generation_id = str(uuid.uuid4())
    generation_arrays = {**arrays, "generation_id": np.asarray(generation_id)}
    generation_metrics = json.loads(json.dumps(metrics, allow_nan=False))
    generation_metrics["generation_id"] = generation_id
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")
    partials = {
        target: target.with_name(target.name + ".partial")
        for target in (path, npz_path, json_path)
    }
    try:
        _write_png_partial(partials[path], figure)
        plt.close(figure)
        figure = None
        _write_npz_partial(partials[npz_path], generation_arrays)
        generation_metrics["generation_assets"] = {
            "png_sha256": _sha256(partials[path]),
            "npz_sha256": _sha256(partials[npz_path]),
        }
        _write_json_partial(partials[json_path], generation_metrics)
        _publish_generation(partials)
    except Exception:
        for partial in partials.values():
            _unlink_durable(partial)
        raise
    finally:
        if figure is not None:
            plt.close(figure)


def _plot_independent_length(
    axis: plt.Axes,
    *,
    length: int,
    statistics: dict[str, np.ndarray | int] | None,
    official_xy: np.ndarray | None,
    color: Any,
) -> None:
    if statistics is None:
        axis.text(
            0.5,
            0.58,
            "Exact paper window too short\n(no substituted spectrum)",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    else:
        axis.step(
            np.asarray(statistics["histogram_centers"]),
            np.asarray(statistics["histogram_density"]),
            where="mid",
            linewidth=1.8,
            color=color,
            label=f"independent ED L={length}",
        )
    if official_xy is not None:
        axis.plot(
            official_xy[1:, 0],
            official_xy[1:, 1],
            linestyle="none",
            marker="o",
            markersize=3.2,
            markerfacecolor="none",
            markeredgecolor=color,
            label=f"official DOI L={length}",
        )


def _finish_independent_axis(axis: plt.Axes, title: str) -> None:
    _plot_theory_reference_curves(axis)
    axis.set_xlim(0.0, 4.8)
    axis.set_xlabel("unfolded spacing s")
    axis.set_ylabel("P(s)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)


def render_independent_fig4(
    independent_results_root: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> dict[str, Path]:
    """Render Fig. 4 from validated independent spectra only."""
    results = load_independent_fig4_results(independent_results_root)
    lengths = list(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_by_length = _load_available_official_histograms(
        official_data_dir, lengths
    )
    statistics_by_length: dict[int, dict[str, np.ndarray | int] | None] = {}
    metrics_by_length: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    series: dict[str, dict[str, Any]] = {}

    for length, result in results.items():
        statistics, convention = _independent_statistics(result["energies"])
        statistics_by_length[length] = statistics
        official_xy = official_by_length.get(length)
        mismatch = _official_mismatch(statistics, official_xy)
        entry: dict[str, Any] = {
            "source": INDEPENDENT_SOURCE,
            "source_directory": str(result["directory"]),
            "source_hashes": result["source_hashes"],
            "full_dimension": result["full_dimension"],
            "sector_dimension": result["sector_dimension"],
            "energy_dataset_metadata": result["energy_dataset_metadata"],
            "eigenvector_metadata": result["eigenvector_metadata"],
            **convention,
            "unfolding_coefficients": None,
            "spacing_count": None,
            "mean_spacing": None,
            "histogram_edges": PAPER_HISTOGRAM_EDGES.tolist(),
            "histogram_counts": None,
            "histogram_density": None,
            "official_mismatch": mismatch,
            "acceptance": {
                "passed": True,
                "basis": (
                    "recursive manifests current and scientific validation passed"
                ),
            },
        }
        arrays[f"L{length}_source"] = np.asarray(INDEPENDENT_SOURCE)
        series[f"L{length}_source"] = {
            "source": INDEPENDENT_SOURCE,
            "length": length,
        }
        if statistics is not None:
            entry.update(
                {
                    "unfolding_coefficients": np.asarray(
                        statistics["polynomial_coefficients"]
                    ).tolist(),
                    "spacing_count": int(len(np.asarray(statistics["spacings"]))),
                    "mean_spacing": float(
                        np.mean(np.asarray(statistics["spacings"]))
                    ),
                    "histogram_counts": np.asarray(
                        statistics["histogram_counts"]
                    ).tolist(),
                    "histogram_density": np.asarray(
                        statistics["histogram_density"]
                    ).tolist(),
                }
            )
            for name in (
                "window_energies",
                "polynomial_coefficients",
                "spacings",
                "histogram_edges",
                "histogram_centers",
                "histogram_counts",
                "histogram_density",
            ):
                key = f"L{length}_{name}"
                arrays[key] = np.asarray(statistics[name])
                series[key] = {
                    "source": INDEPENDENT_SOURCE,
                    "length": length,
                }
        if official_xy is not None:
            arrays[f"L{length}_official_xy"] = official_xy
            arrays[f"L{length}_official_source"] = np.asarray(OFFICIAL_SOURCE)
            series[f"official_L{length}_xy"] = {
                "source": OFFICIAL_SOURCE,
                "length": length,
            }
        metrics_by_length[str(length)] = entry

    contiguous = np.arange(lengths[0], lengths[-1] + 1, 2)
    missing = np.setdiff1d(contiguous, np.asarray(lengths)).tolist()
    base_metrics = {
        "schema_version": FIG4_INDEPENDENT_SCHEMA_VERSION,
        "source": INDEPENDENT_SOURCE,
        "independent_results_root": str(Path(independent_results_root)),
        "available_independent_lengths": lengths,
        "missing_lengths_within_independent_range": missing,
        "missing_length_range": (
            {"start": lengths[0], "stop": lengths[-1], "step": 2}
            if len(lengths) > 1
            else None
        ),
        "missing_size_policy": (
            "reports only even gaps within the independent range; never bridges, "
            "connects, or substitutes official data"
        ),
        "method": {
            "window": "sorted_energies[D//5:D//2-500]",
            "unfolding_degree": PAPER_UNFOLDING_DEGREE,
            "edge_trim_levels": PAPER_EDGE_TRIM,
            "zero_mode_filter": False,
            "spacing_renormalized": False,
            "histogram_edges": PAPER_HISTOGRAM_EDGES.tolist(),
        },
        "series": series,
        "lengths": metrics_by_length,
        "official_data": (
            {"source": OFFICIAL_SOURCE, "root": str(Path(official_data_dir))}
            if official_data_dir is not None
            else None
        ),
        "acceptance": {
            "passed": True,
            "validated_lengths": lengths,
            "generated_source": INDEPENDENT_SOURCE,
        },
    }

    colors = plt.get_cmap("tab10")
    paths: dict[str, Path] = {}
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for index, length in enumerate(lengths):
        _plot_independent_length(
            axis,
            length=length,
            statistics=statistics_by_length[length],
            official_xy=official_by_length.get(length),
            color=colors(index % 10),
        )
    _finish_independent_axis(
        axis, "Turner et al. (2018) Fig. 4 — independent ED sizes"
    )
    figure.tight_layout()
    all_path = output_dir / "fig4_independent_all.png"
    _publish_figure_generation(
        all_path,
        figure,
        arrays,
        {**base_metrics, "layout": {"kind": "all-sizes", "lengths": lengths}},
    )
    paths["figure_all"] = all_path

    for index, length in enumerate(lengths):
        figure, axis = plt.subplots(figsize=(6.4, 4.5))
        _plot_independent_length(
            axis,
            length=length,
            statistics=statistics_by_length[length],
            official_xy=official_by_length.get(length),
            color=colors(index % 10),
        )
        _finish_independent_axis(axis, f"Independent ED Fig. 4 comparison — L={length}")
        figure.tight_layout()
        path = output_dir / f"fig4_independent_L{length}.png"
        length_arrays = {
            name: value
            for name, value in arrays.items()
            if name.startswith(f"L{length}_")
        }
        _publish_figure_generation(
            path,
            figure,
            length_arrays,
            {
                **base_metrics,
                "layout": {"kind": "single-size", "lengths": [length]},
            },
        )
        paths[f"figure_L{length}"] = path
    return paths


def render_fig4_stage(
    output_dir: str | Path,
    length: int,
    *,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> Path:
    """Task 6 figures-stage hook for one validated length directory."""
    output_dir = Path(output_dir)
    paths = render_independent_fig4(
        output_dir,
        output_dir=output_dir / "figures",
        official_data_dir=official_data_dir,
    )
    expected = output_dir / "figures" / f"fig4_independent_L{length}.png"
    if paths.get(f"figure_L{length}") != expected:
        raise RuntimeError(f"Fig. 4 renderer did not produce expected L={length}")
    return paths["figure_all"]


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
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument("--length", type=int, default=16)
    input_mode.add_argument(
        "--paper-exact",
        action="store_true",
        help="reconstruct all official L=28/30/32 histograms from source energies",
    )
    input_mode.add_argument(
        "--independent-results-root",
        type=Path,
        help="root containing hash-validated Task 6 length directories",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-data-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_DATA,
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
    if args.independent_results_root is not None:
        paths = render_independent_fig4(
            args.independent_results_root,
            output_dir=args.output_dir,
            official_data_dir=args.official_data_dir,
        )
        for path in paths.values():
            print(path)
        return 0
    path = run_figure(args.length, args.output_dir, args.official_data_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
