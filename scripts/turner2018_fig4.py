#!/usr/bin/env python3
"""Level statistics for the PXP k=0, inversion-even symmetry sector."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from io import BytesIO
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
from scipy.interpolate import PchipInterpolator

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
    _discover_task6_directories,
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
    "poisson": "P",
    "semi-poisson": "SP",
    "goe": "WD",
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
    if (directory / "compact.json").is_file() and not (
        directory / "eigensystem.h5"
    ).is_file():
        from turner2018_compact import load_compact_package

        compact = load_compact_package(directory)
        return {
            key: compact[key]
            for key in (
                "source",
                "directory",
                "length",
                "full_dimension",
                "sector_dimension",
                "energies",
                "source_hashes",
                "energy_dataset_metadata",
                "eigenvector_metadata",
            )
        }

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
    results: dict[int, dict[str, Any]] = {}
    for directory in _discover_task6_directories(root):
        result = _load_independent_fig4_length(directory)
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
) -> dict[int, dict[str, Any]]:
    if official_data_dir is None:
        return {}
    root = Path(official_data_dir)
    archive = root / "level_statistics.zip"
    if not archive.is_file():
        return {}
    requested = [
        length for length in lengths if length in FIG4_HISTOGRAM_MEMBERS
    ]
    if not requested:
        return {}
    records: dict[int, dict[str, Any]] = {}
    with archive.open("rb") as archive_handle:
        archive_sha256 = _hash_open_file(archive_handle)
        archive_byte_size = os.fstat(archive_handle.fileno()).st_size
        archive_handle.seek(0)
        try:
            with ZipFile(archive_handle) as zip_handle:
                members = set(zip_handle.namelist())
                member_snapshots = {
                    length: zip_handle.read(FIG4_HISTOGRAM_MEMBERS[length])
                    for length in requested
                    if FIG4_HISTOGRAM_MEMBERS[length] in members
                }
        except BadZipFile:
            return {}
        if _hash_open_file(archive_handle) != archive_sha256:
            raise RuntimeError(
                "official Fig. 4 archive changed while its open snapshot was consumed"
            )
        for length, member_bytes in member_snapshots.items():
            member = FIG4_HISTOGRAM_MEMBERS[length]
            xy = np.loadtxt(BytesIO(member_bytes))
            if (
                xy.shape != (25, 2)
                or not np.all(np.isfinite(xy))
                or np.any(xy[:, 1] < 0)
            ):
                raise RuntimeError(
                    f"official Fig. 4 member is invalid for L={length}"
                )
            records[length] = {
                "xy": xy,
                "provenance": {
                    "source": OFFICIAL_SOURCE,
                    "archive_path": str(archive.resolve()),
                    "archive_sha256": archive_sha256,
                    "archive_byte_size": archive_byte_size,
                    "member": member,
                    "member_sha256": hashlib.sha256(member_bytes).hexdigest(),
                    "member_byte_size": len(member_bytes),
                    "snapshot_semantics": (
                        "array and hashes consumed from one stable open archive handle"
                    ),
                },
            }
    return records


def _official_xy(record: Any) -> np.ndarray | None:
    if record is None:
        return None
    if isinstance(record, dict):
        return np.asarray(record["xy"])
    return np.asarray(record)


def _official_source_provenance(record: Any) -> dict[str, Any] | None:
    if isinstance(record, dict):
        return record["provenance"]
    return None


def _independent_statistics(
    energies: np.ndarray,
) -> tuple[dict[str, np.ndarray | int] | None, dict[str, Any]]:
    dimension = len(energies)
    lower = dimension // 5
    upper = dimension // 2 - 500
    window = energies[lower:upper]
    resolved_lower, resolved_upper, _step = slice(lower, upper).indices(dimension)
    minimum_window = 2 * PAPER_EDGE_TRIM + 2
    convention = {
        "window_expression": "sorted_energies[D//5:D//2-500]",
        "window_bounds": [lower, upper],
        "raw_window_bounds": [lower, upper],
        "resolved_window_bounds": [resolved_lower, resolved_upper],
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


def _write_figure_generation_partials(
    path: Path,
    figure: Any,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
    generation_id: str,
    partials: dict[Path, Path],
) -> None:
    generation_arrays = {**arrays, "generation_id": np.asarray(generation_id)}
    generation_metrics = json.loads(json.dumps(metrics, allow_nan=False))
    generation_metrics["generation_id"] = generation_id
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")
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
    finally:
        if figure is not None:
            plt.close(figure)


def _cleanup_generation_partials(partials: dict[Path, Path]) -> None:
    for partial in partials.values():
        try:
            _unlink_durable(partial)
        except Exception as error:
            raise RuntimeError(
                "Fig. 4 partial cleanup incomplete; recoverable partials preserved"
            ) from error


def _plot_independent_length(
    axis: plt.Axes,
    *,
    length: int,
    statistics: dict[str, np.ndarray | int] | None,
    official_xy: np.ndarray | None,
    color: Any,
) -> None:
    if statistics is None:
        axis.plot(
            [],
            [],
            color=color,
            linestyle=":",
            linewidth=1.2,
            label=f"L={length} unavailable (paper window)",
        )
    else:
        curve_x, curve_y = _smooth_density_curve(
            np.asarray(statistics["histogram_centers"]),
            np.asarray(statistics["histogram_density"]),
        )
        axis.plot(
            curve_x,
            curve_y,
            linewidth=1.8,
            linestyle="-",
            color=color,
            label=f"L={length}",
        )
    if official_xy is not None:
        official_x, official_y = _smooth_density_curve(
            official_xy[1:, 0], official_xy[1:, 1]
        )
        axis.plot(
            official_x,
            official_y,
            linestyle="--",
            linewidth=1.2,
            label=f"L={length} official",
        )


def _smooth_density_curve(
    centers: np.ndarray, density: np.ndarray, *, samples_per_bin: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Shape-preserving display interpolation through persisted histogram points."""
    x = np.asarray(centers, dtype=np.float64)
    y = np.asarray(density, dtype=np.float64)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError("density curve points must be finite and strictly ordered")
    grid = np.linspace(x[0], x[-1], samples_per_bin * (len(x) - 1) + 1)
    values = PchipInterpolator(x, y)(grid)
    return grid, np.maximum(values, 0.0)


def _plot_density_of_states_inset(
    axis: plt.Axes, energies: np.ndarray
) -> plt.Axes:
    """Render the paper's L=32 density-of-states inset."""
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("density-of-states energies must be finite and one-dimensional")
    density, edges = np.histogram(
        values, bins=np.linspace(-20.0, 20.0, 121), density=True
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    curve_x, curve_y = _smooth_density_curve(centers, density)
    inset = axis.inset_axes([0.59, 0.57, 0.37, 0.37])
    inset.plot(curve_x, curve_y, color="black", linewidth=1.0)
    inset.set_xlim(-20.0, 20.0)
    inset.set_xticks([-20.0, -10.0, 0.0, 10.0, 20.0])
    inset.set_ylim(0.0, 0.12)
    inset.set_yticks([0.0, 0.1])
    inset.set_xlabel("$E$", fontsize=7)
    inset.set_ylabel(r"$\rho(E)$", fontsize=7)
    inset.tick_params(labelsize=6)
    return inset


def _finish_independent_axis(axis: plt.Axes, title: str) -> None:
    _plot_theory_reference_curves(axis)
    axis.set_xlim(0.0, 5.0)
    axis.set_xticks(np.arange(0.0, 5.1, 1.0))
    axis.set_ylim(0.0, 1.05)
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_xlabel(r"$s$")
    axis.set_ylabel(r"$P(s)$")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)


def _acceptance_for_lengths(
    lengths: list[int],
    statistics_by_length: dict[int, dict[str, np.ndarray | int] | None],
) -> dict[str, Any]:
    required_lengths = [length for length in lengths if length in PAPER_LENGTHS]
    statistics_required = bool(required_lengths)
    checked_lengths = required_lengths if statistics_required else lengths
    statistics_passed = all(
        statistics_by_length[length] is not None for length in checked_lengths
    )
    return {
        "passed": bool(not statistics_required or statistics_passed),
        "provenance_passed": True,
        "render_passed": True,
        "statistics_passed": statistics_passed,
        "statistics_required": statistics_required,
        "mode": (
            "statistics-required" if statistics_required else "provenance-only"
        ),
        "validated_lengths": lengths,
        "generated_source": INDEPENDENT_SOURCE,
    }


def _scoped_metrics(
    base_metrics: dict[str, Any],
    scope: list[int],
    statistics_by_length: dict[int, dict[str, np.ndarray | int] | None],
    *,
    layout_kind: str,
) -> dict[str, Any]:
    metrics = json.loads(json.dumps(base_metrics, allow_nan=False))
    scope_set = set(scope)
    metrics["available_independent_lengths"] = scope
    contiguous = np.arange(scope[0], scope[-1] + 1, 2)
    metrics["missing_lengths_within_independent_range"] = np.setdiff1d(
        contiguous, np.asarray(scope)
    ).tolist()
    metrics["missing_length_range"] = (
        {"start": scope[0], "stop": scope[-1], "step": 2}
        if len(scope) > 1
        else None
    )
    metrics["lengths"] = {
        str(length): metrics["lengths"][str(length)] for length in scope
    }
    metrics["series"] = {
        name: entry
        for name, entry in metrics["series"].items()
        if entry.get("length") in scope_set
    }
    if metrics["official_data"] is not None:
        metrics["official_data"]["overlays"] = {
            str(length): provenance
            for length, provenance in metrics["official_data"]["overlays"].items()
            if int(length) in scope_set
        }
    metrics["acceptance"] = _acceptance_for_lengths(
        scope, statistics_by_length
    )
    metrics["layout"] = {"kind": layout_kind, "lengths": scope}
    return metrics


def _scoped_arrays(
    arrays: dict[str, np.ndarray], scope: list[int]
) -> dict[str, np.ndarray]:
    prefixes = tuple(f"L{length}_" for length in scope)
    return {
        name: value for name, value in arrays.items() if name.startswith(prefixes)
    }


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
        official_record = official_by_length.get(length)
        official_xy = _official_xy(official_record)
        mismatch = _official_mismatch(statistics, official_xy)
        statistics_available = statistics is not None
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
            "official_source_provenance": _official_source_provenance(
                official_record
            ),
            "provenance_acceptance": {
                "passed": True,
                "basis": (
                    "recursive manifests current and scientific validation passed"
                ),
            },
            "histogram_acceptance": {
                "passed": statistics_available,
                "statistics_available": statistics_available,
                "required_for_task6_production": length in PAPER_LENGTHS,
                "basis": (
                    "exact paper window and unfolding accepted"
                    if statistics_available
                    else convention["unavailable_reason"]
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
        "plot_conventions": {
            "histogram_rendering": (
                "shape-preserving PCHIP display interpolation through persisted "
                "bin-center densities; histogram values remain unmodified"
            ),
            "generated": "solid curves",
            "official": "dashed curves",
            "theory_labels": dict(THEORY_CURVE_LABELS),
            "x_range": [0.0, 5.0],
            "x_ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y_range": [0.0, 1.05],
            "y_ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "density_of_states_inset": {
                "length": 32,
                "x_range": [-20.0, 20.0],
                "x_ticks": [-20.0, -10.0, 0.0, 10.0, 20.0],
                "y_range": [0.0, 0.12],
                "y_ticks": [0.0, 0.1],
            },
        },
        "series": series,
        "lengths": metrics_by_length,
        "official_data": (
            {
                "source": OFFICIAL_SOURCE,
                "root": str(Path(official_data_dir)),
                "overlays": {
                    str(length): provenance
                    for length, record in official_by_length.items()
                    if (
                        provenance := _official_source_provenance(record)
                    ) is not None
                },
            }
            if official_data_dir is not None
            else None
        ),
    }

    colors = plt.get_cmap("tab10")
    paper_lengths = [length for length in PAPER_LENGTHS if length in results]
    paper_path = output_dir / "fig4_paper_comparison.png"
    supplemental_path = output_dir / "fig4_supplemental_L22-L32.png"
    paths = {
        "figure_supplemental": supplemental_path,
        **{
            f"figure_L{length}": output_dir / f"fig4_independent_L{length}.png"
            for length in lengths
        },
    }
    if paper_lengths:
        paths = {"figure_paper": paper_path, **paths}
    targets = tuple(
        target
        for path in paths.values()
        for target in (path, path.with_suffix(".npz"), path.with_suffix(".json"))
    )
    for path in paths.values():
        triplet = (path, path.with_suffix(".npz"), path.with_suffix(".json"))
        existing = [target.is_file() for target in triplet]
        if any(existing) and not all(existing):
            raise RuntimeError(
                "refusing to replace an incomplete prior Fig. 4 output triplet"
            )
    partials = {
        target: target.with_name(target.name + ".partial") for target in targets
    }
    generation_id = str(uuid.uuid4())
    try:
        aggregate_specs = [
            (
                supplemental_path,
                lengths,
                "supplemental-finite-size",
                "Supplemental finite-size comparison — L=22–32",
            )
        ]
        if paper_lengths:
            aggregate_specs.insert(
                0, (paper_path, paper_lengths, "paper-comparison", "")
            )
        for path, scope, layout_kind, title in aggregate_specs:
            figure, axis = plt.subplots(figsize=(7.2, 5.0))
            for index, length in enumerate(scope):
                _plot_independent_length(
                    axis,
                    length=length,
                    statistics=statistics_by_length[length],
                    official_xy=_official_xy(official_by_length.get(length)),
                    color=colors(index % 10),
                )
            _finish_independent_axis(axis, title)
            if 32 in scope:
                _plot_density_of_states_inset(axis, results[32]["energies"])
                axis.legend(loc="lower right", fontsize=7)
            figure.tight_layout()
            _write_figure_generation_partials(
                path,
                figure,
                _scoped_arrays(arrays, scope),
                _scoped_metrics(
                    base_metrics,
                    scope,
                    statistics_by_length,
                    layout_kind=layout_kind,
                ),
                generation_id,
                partials,
            )

        for index, length in enumerate(lengths):
            figure, axis = plt.subplots(figsize=(6.4, 4.5))
            _plot_independent_length(
                axis,
                length=length,
                statistics=statistics_by_length[length],
                official_xy=_official_xy(official_by_length.get(length)),
                color=colors(index % 10),
            )
            _finish_independent_axis(
                axis, f"Independent ED Fig. 4 comparison — L={length}"
            )
            figure.tight_layout()
            path = paths[f"figure_L{length}"]
            length_arrays = {
                name: value
                for name, value in arrays.items()
                if name.startswith(f"L{length}_")
            }
            _write_figure_generation_partials(
                path,
                figure,
                length_arrays,
                _scoped_metrics(
                    base_metrics,
                    [length],
                    statistics_by_length,
                    layout_kind="single-size",
                ),
                generation_id,
                partials,
            )
    except Exception as write_error:
        try:
            _cleanup_generation_partials(partials)
        except RuntimeError as cleanup_error:
            raise RuntimeError(
                "Fig. 4 generation failed and partial cleanup was incomplete"
            ) from ExceptionGroup(
                "Fig. 4 write and cleanup errors", [write_error, cleanup_error]
            )
        raise
    try:
        _publish_generation(partials, allow_mixed_previous=True)
    except Exception as publish_error:
        try:
            _cleanup_generation_partials(partials)
        except RuntimeError as cleanup_error:
            raise RuntimeError(
                "Fig. 4 publication failed and partial cleanup was incomplete"
            ) from ExceptionGroup(
                "Fig. 4 publication and cleanup errors",
                [publish_error, cleanup_error],
            )
        raise
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
    return paths["figure_supplemental"]


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
    axis.plot(
        PAPER_HISTOGRAM_CENTERS,
        np.asarray(reconstruction["histogram_density"]),
        linewidth=1.8,
        linestyle="-",
        color=color,
        label=f"L={length} reconstructed",
    )
    axis.plot(
        official_xy[1:, 0],
        official_xy[1:, 1],
        linestyle="--",
        linewidth=1.2,
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
