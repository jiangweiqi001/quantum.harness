#!/usr/bin/env python3
"""Finite-size overlap, FSA, and participation-ratio analysis for Fig. 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import h5py
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
INDEPENDENT_SOURCE = "independent-ed"
OFFICIAL_SOURCE = "official-doi"
FIG3_SCHEMA_VERSION = "turner2018-independent-fig3-v1"
EXPECTED_MODEL = "H=sum_j P_(j-1) X_j P_(j+1), PBC"
REQUIRED_OBSERVABLES = {
    "exact_shell_amplitudes",
    "fsa_beta_full_chain",
    "fsa_hamiltonian_sector",
    "fsa_shell_vectors_sector",
    "overlap_z2",
    "participation_ratio",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, path)


def _maximum_mismatch(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return None
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _zip_has_member(path: Path, member: str) -> bool:
    if not path.is_file():
        return False
    try:
        with ZipFile(path) as archive:
            return member in archive.namelist()
    except BadZipFile:
        return False


def _scientific_dimensions(
    directory: Path,
    validation: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[int, int]:
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"validation metrics are missing in {directory}")
    full = metrics.get("basis_full_dimension", {}).get("value")
    sector = metrics.get("basis_sector_dimension", {}).get("value")
    if not isinstance(full, int) or not isinstance(sector, int):
        raise RuntimeError(f"validation dimensions are invalid in {directory}")
    planned_sector = plan["basis"].get("sector_dimension")
    planned_full = plan["basis"].get("full_constrained_dimension")
    if planned_sector is not None and planned_sector != sector:
        raise RuntimeError(f"planned sector dimension is stale in {directory}")
    if planned_full is not None and planned_full != full:
        raise RuntimeError(f"planned full dimension is stale in {directory}")
    return full, sector


def _load_independent_length(directory: Path) -> dict[str, Any]:
    """Load one current Task 6 result without materializing eigenvectors."""
    from turner2018_l32_server import require_stage

    cache: dict[str, dict[str, Any]] = {}
    validate_manifest = require_stage(directory, "validate", cache)
    diagonalize_manifest = cache["diagonalize"]
    observables_manifest = cache["observables"]
    plan = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    model = plan.get("model", {})
    if (
        model.get("hamiltonian") != EXPECTED_MODEL
        or model.get("boundary") != "periodic"
        or model.get("momentum") != 0
        or model.get("inversion") != "even"
    ):
        raise RuntimeError(f"scientific model attributes are invalid in {directory}")
    length = model.get("length")
    if not isinstance(length, int) or length < 4 or length % 2:
        raise RuntimeError(f"scientific length is invalid in {directory}")

    validation = json.loads(
        (directory / "validation" / "metrics.json").read_text(encoding="utf-8")
    )
    if validation.get("passed") is not True or validation.get("length") != length:
        raise RuntimeError(f"scientific validation did not pass for L={length}")
    validated_hashes = validation.get("validated_stage_sha256", {})
    for stage, manifest in (
        ("diagonalize", diagonalize_manifest),
        ("observables", observables_manifest),
    ):
        if validated_hashes.get(stage) != manifest["artifact"]["sha256"]:
            raise RuntimeError(
                f"validation source hash does not match stage={stage} for L={length}"
            )
    full_dimension, sector_dimension = _scientific_dimensions(
        directory, validation, plan
    )

    with h5py.File(directory / "eigensystem.h5", "r") as handle:
        energies_dataset = handle["eigensystem/energies"]
        vectors = handle["eigensystem/vectors"]
        expected_vector_metadata = (
            vectors.shape == (sector_dimension, sector_dimension)
            and vectors.dtype == np.float64
            and vectors.chunks == (sector_dimension, 1)
        )
        if not expected_vector_metadata:
            raise RuntimeError(f"eigenvector metadata is invalid for L={length}")
        energies = energies_dataset[()]
        eigenvector_metadata = {
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "chunks": list(vectors.chunks or ()),
            "access": "metadata-only",
        }

    expected_shells = length // 2 + 1
    expected_shapes = {
        "overlap_z2": (sector_dimension,),
        "participation_ratio": (sector_dimension,),
        "exact_shell_amplitudes": (expected_shells, sector_dimension),
        "fsa_shell_vectors_sector": (expected_shells, sector_dimension),
        "fsa_hamiltonian_sector": (expected_shells, expected_shells),
        "fsa_beta_full_chain": (length,),
    }
    observables: dict[str, np.ndarray] = {}
    with h5py.File(directory / "observables.h5", "r") as handle:
        group = handle["observables"]
        if set(group.keys()) != REQUIRED_OBSERVABLES:
            raise RuntimeError(f"observable dataset set is invalid for L={length}")
        metadata = json.loads(group.attrs["validation_metadata"])
        if (
            metadata.get("eigenvector_access") != "column-chunked"
            or metadata.get("finite_columns_checked") != sector_dimension
            or metadata.get("residual_columns_checked") != sector_dimension
        ):
            raise RuntimeError(f"observable scientific attributes are invalid for L={length}")
        for name, shape in expected_shapes.items():
            dataset = group[name]
            if dataset.shape != shape or dataset.dtype != np.float64:
                raise RuntimeError(
                    f"observable {name} metadata is invalid for L={length}"
                )
            observables[name] = dataset[()]

    if energies.shape != (sector_dimension,) or np.any(np.diff(energies) < 0):
        raise RuntimeError(f"energy array is invalid for L={length}")
    overlap_sum = float(np.sum(observables["overlap_z2"]))
    if abs(overlap_sum - 0.5) > 1e-10:
        raise RuntimeError(f"Z2 overlap sum is invalid for L={length}: {overlap_sum}")
    fsa_hamiltonian = observables["fsa_hamiltonian_sector"]
    if not np.allclose(fsa_hamiltonian, fsa_hamiltonian.T, atol=1e-12, rtol=0):
        raise RuntimeError(f"FSA Hamiltonian is not symmetric for L={length}")
    fsa_energies, fsa_vectors = np.linalg.eigh(fsa_hamiltonian)
    fsa_overlap = 0.5 * np.abs(fsa_vectors[0]) ** 2
    selected = select_fig3_pr2_states(
        energies=energies,
        exact_shell_amplitudes=observables["exact_shell_amplitudes"],
        fsa_eigenvectors=fsa_vectors,
    )
    other_pr2, special_pr2 = pr2_fsa_averages(
        energies=energies,
        pr2=observables["participation_ratio"],
        exact_shell_amplitudes=observables["exact_shell_amplitudes"],
        fsa_eigenvectors=fsa_vectors,
    )
    return {
        "source": INDEPENDENT_SOURCE,
        "directory": directory,
        "length": length,
        "full_dimension": full_dimension,
        "sector_dimension": sector_dimension,
        "energies": energies,
        **observables,
        "fsa_energies": fsa_energies,
        "fsa_eigenvectors": fsa_vectors,
        "fsa_overlap_z2": fsa_overlap,
        "selector": selected,
        "pr2_other": other_pr2,
        "pr2_special": special_pr2,
        "overlap_sum": overlap_sum,
        "source_hashes": {
            "eigensystem.h5": diagonalize_manifest["artifact"]["sha256"],
            "observables.h5": observables_manifest["artifact"]["sha256"],
            "validate_manifest": _sha256(directory / "stages" / "validate.json"),
            "validation_metrics": validate_manifest["artifact"]["sha256"],
        },
        "eigenvector_metadata": eigenvector_metadata,
    }


def load_independent_results(root: str | Path) -> dict[int, dict[str, Any]]:
    """Discover and load current Task 6 length directories below one root."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"independent results root is not a directory: {root}")
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise RuntimeError(f"no independent Task 6 manifests found below {root}")
    results: dict[int, dict[str, Any]] = {}
    for manifest in manifests:
        result = _load_independent_length(manifest.parent)
        length = result["length"]
        if length in results:
            raise RuntimeError(f"duplicate independent result for L={length}")
        results[length] = result
    return dict(sorted(results.items()))


def _series_entry(
    source: str,
    *,
    panel: str,
    length: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"source": source, "panel": panel}
    if length is not None:
        entry["length"] = length
    return entry


def render_independent_fig3(
    independent_results_root: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> Path:
    """Render Fig. 3 exclusively from validated independent ED artifacts."""
    results = load_independent_results(independent_results_root)
    lengths = np.asarray(sorted(results), dtype=int)
    primary_length = int(lengths[-1])
    primary = results[primary_length]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig3_independent_L{primary_length}"
    path = output_dir / f"{stem}.png"
    arrays: dict[str, np.ndarray] = {}
    series: dict[str, dict[str, Any]] = {}

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    panel_a, panel_b, panel_c, panel_d = axes.ravel()
    panel_a.scatter(
        primary["energies"],
        primary["overlap_z2"],
        s=9,
        alpha=0.55,
        color="tab:blue",
        label=f"independent ED L={primary_length}",
    )
    panel_a.scatter(
        primary["fsa_energies"],
        primary["fsa_overlap_z2"],
        marker="x",
        color="tab:red",
        label=f"independent FSA L={primary_length}",
    )
    for name, values in (
        ("energies", primary["energies"]),
        ("overlap_z2", primary["overlap_z2"]),
        ("fsa_energies", primary["fsa_energies"]),
        ("fsa_overlap_z2", primary["fsa_overlap_z2"]),
    ):
        key = f"panel_a_L{primary_length}_{name}"
        arrays[key] = np.asarray(values)
        series[key] = _series_entry(
            INDEPENDENT_SOURCE, panel="a", length=primary_length
        )
    arrays[f"panel_a_L{primary_length}_source"] = np.asarray(INDEPENDENT_SOURCE)

    selector = primary["selector"]
    special_index = int(
        selector["special"][
            np.argmin(np.abs(primary["energies"][selector["special"]]))
        ]
    )
    state_indices = (int(selector["sorted_tower"][0]), special_index)
    panel_titles = ("matched tower ground state", "matched special state near E=0")
    selected_details: list[dict[str, Any]] = []
    shell = np.arange(primary["exact_shell_amplitudes"].shape[0])
    for label, panel, state_index, title in zip(
        ("b", "c"), (panel_b, panel_c), state_indices, panel_titles
    ):
        fsa_index = int(np.flatnonzero(selector["tower"] == state_index)[0])
        exact_weights = np.abs(
            primary["exact_shell_amplitudes"][:, state_index]
        ) ** 2
        fsa_weights = np.abs(primary["fsa_eigenvectors"][:, fsa_index]) ** 2
        panel.plot(shell, exact_weights, "o-", markersize=3, label="independent exact")
        panel.plot(
            shell,
            fsa_weights,
            "x--",
            color="tab:red",
            label="independent FSA",
        )
        panel.set_title(
            f"{title}, E={primary['energies'][state_index]:.3f}"
        )
        panel.set_xlabel("FSA shell n")
        panel.set_ylabel("shell weight")
        panel.legend(fontsize=8)
        for name, values in (
            ("shell", shell),
            ("exact_weights", exact_weights),
            ("fsa_weights", fsa_weights),
        ):
            key = f"panel_{label}_L{primary_length}_{name}"
            arrays[key] = np.asarray(values)
            series[key] = _series_entry(
                INDEPENDENT_SOURCE, panel=label, length=primary_length
            )
        selected_details.append(
            {
                "panel": label,
                "exact_index": state_index,
                "exact_energy": float(primary["energies"][state_index]),
                "fsa_index": fsa_index,
                "fsa_energy": float(primary["fsa_energies"][fsa_index]),
            }
        )

    independent_other = np.asarray(
        [results[int(length)]["pr2_other"] for length in lengths]
    )
    independent_special = np.asarray(
        [results[int(length)]["pr2_special"] for length in lengths]
    )
    panel_d.semilogy(
        lengths,
        independent_other,
        "o",
        linestyle="none",
        color="tab:blue",
        label="independent other",
    )
    panel_d.semilogy(
        lengths,
        independent_special,
        "s",
        linestyle="none",
        color="tab:red",
        label="independent FSA special",
    )
    arrays["panel_d_lengths"] = lengths
    arrays["panel_d_other"] = independent_other
    arrays["panel_d_special"] = independent_special
    arrays["panel_d_source"] = np.asarray(INDEPENDENT_SOURCE)
    for name in ("lengths", "other", "special"):
        series[f"panel_d_{name}"] = _series_entry(INDEPENDENT_SOURCE, panel="d")

    official_by_length: dict[int, dict[str, Any]] = {}
    official_scaling: dict[str, np.ndarray] | None = None
    official_root = None if official_data_dir is None else Path(official_data_dir)
    if official_root is not None:
        overlap_archives = (
            official_root / "overlaps_with_Neel_state.zip",
            official_root / "energy_eigenvalues.zip",
            official_root / "eigendecomposition.zip",
        )
        if overlap_archives[0].is_file():
            for length in lengths:
                length_value = int(length)
                energy_archive = (
                    overlap_archives[1] if length_value >= 26 else overlap_archives[2]
                )
                overlap_member = f"oneel_periodic_N{length_value}_k0_p0.dat"
                energy_member = (
                    f"evals_periodic_N{length_value}_k0_p0.npy.txt"
                    if length_value >= 26
                    else f"eigs_periodic_N{length_value}_k0_p0.h5"
                )
                if not (
                    _zip_has_member(overlap_archives[0], overlap_member)
                    and _zip_has_member(energy_archive, energy_member)
                ):
                    continue
                official_energies, official_overlap = load_fig3_overlap(
                    official_root, length=length_value
                )
                official_by_length[length_value] = {
                    "energies": official_energies,
                    "overlap": official_overlap,
                }
                arrays[f"official_L{length_value}_energies"] = official_energies
                arrays[f"official_L{length_value}_overlap_z2"] = official_overlap
                arrays[f"official_L{length_value}_source"] = np.asarray(OFFICIAL_SOURCE)
                for name in ("energies", "overlap_z2"):
                    series[f"official_L{length_value}_{name}"] = _series_entry(
                        OFFICIAL_SOURCE, panel="a", length=length_value
                    )
        if (official_root / "participation_ratios.zip").is_file():
            official_scaling = load_fig3_pr2_scaling(official_root)
            available = official_scaling["available"]
            plot_official_pr2_scaling(panel_d, official_scaling)
            for name in ("length", "other", "special", "available"):
                arrays[f"official_pr2_{name}"] = np.asarray(official_scaling[name])
                series[f"official_pr2_{name}"] = _series_entry(
                    OFFICIAL_SOURCE, panel="d"
                )
        primary_official = official_by_length.get(primary_length)
        if primary_official is not None:
            panel_a.scatter(
                primary_official["energies"],
                primary_official["overlap"],
                s=3,
                alpha=0.3,
                color="tab:orange",
                rasterized=True,
                label=f"official ED L={primary_length}",
            )
        if (
            primary_length in {26, 32}
            and _zip_has_member(
                official_root / "forward-scattering.zip",
                f"fscat_periodic_N{primary_length}_k0_p0.h5",
            )
        ):
            official_fsa = load_fig3_fsa(official_root, length=primary_length)
            official_by_length.setdefault(primary_length, {})["fsa"] = official_fsa
            panel_a.scatter(
                official_fsa["energies"],
                official_fsa["overlap"],
                marker="+",
                color="black",
                label=f"official FSA L={primary_length}",
            )
            for name in ("energies", "overlap"):
                arrays[f"official_fsa_L{primary_length}_{name}"] = official_fsa[name]
                series[f"official_fsa_L{primary_length}_{name}"] = _series_entry(
                    OFFICIAL_SOURCE, panel="a", length=primary_length
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
    image_partial = path.with_name(path.stem + ".partial.png")
    figure.savefig(image_partial, dpi=180)
    plt.close(figure)
    os.replace(image_partial, path)

    per_length: dict[str, Any] = {}
    for length in lengths:
        result = results[int(length)]
        selected = result["selector"]
        official = official_by_length.get(int(length), {})
        official_pr2_mismatch = None
        if official_scaling is not None:
            match = np.flatnonzero(official_scaling["length"] == length)
            if len(match) and bool(official_scaling["available"][match[0]]):
                index = int(match[0])
                official_pr2_mismatch = {
                    "other_abs": abs(
                        result["pr2_other"] - float(official_scaling["other"][index])
                    ),
                    "special_abs": abs(
                        result["pr2_special"]
                        - float(official_scaling["special"][index])
                    ),
                }
        per_length[str(int(length))] = {
            "source": INDEPENDENT_SOURCE,
            "source_directory": str(result["directory"]),
            "source_hashes": result["source_hashes"],
            "full_dimension": result["full_dimension"],
            "sector_dimension": result["sector_dimension"],
            "eigenvector_metadata": result["eigenvector_metadata"],
            "overlap_sum": result["overlap_sum"],
            "fsa": {
                "match_exact_indices": selected["tower"].tolist(),
                "match_exact_energies": result["energies"][
                    selected["tower"]
                ].tolist(),
                "fsa_energies": result["fsa_energies"].tolist(),
                "shell_amplitudes_sha256": hashlib.sha256(
                    result["exact_shell_amplitudes"].tobytes()
                ).hexdigest(),
                "selector": "turner2018_official.select_fig3_pr2_states",
                "tie_rtol": 1e-8,
                "tie_atol": 1e-12,
            },
            "pr2": {
                "other_mean": result["pr2_other"],
                "special_mean": result["pr2_special"],
                "other_indices": selected["other"].tolist(),
                "special_indices": selected["special"].tolist(),
                "zero_energy_excluded_indices": np.flatnonzero(
                    np.abs(result["energies"]) <= 1e-10
                ).tolist(),
                "all_values_sha256": hashlib.sha256(
                    result["participation_ratio"].tobytes()
                ).hexdigest(),
                "selector": "turner2018_official.select_fig3_pr2_states",
            },
            "official_mismatch": {
                "energies_max_abs": (
                    _maximum_mismatch(result["energies"], official["energies"])
                    if "energies" in official
                    else None
                ),
                "overlap_max_abs": (
                    _maximum_mismatch(result["overlap_z2"], official["overlap"])
                    if "overlap" in official
                    else None
                ),
                "fsa_energies_max_abs": (
                    _maximum_mismatch(
                        result["fsa_energies"], official["fsa"]["energies"]
                    )
                    if "fsa" in official
                    else None
                ),
                "pr2": official_pr2_mismatch,
            },
            "acceptance": {
                "passed": True,
                "basis": "recursive manifests current and scientific validation passed",
            },
        }

    contiguous = np.arange(int(lengths[0]), int(lengths[-1]) + 1, 2)
    missing = np.setdiff1d(contiguous, lengths).tolist()
    metrics = {
        "schema_version": FIG3_SCHEMA_VERSION,
        "source": INDEPENDENT_SOURCE,
        "independent_results_root": str(Path(independent_results_root)),
        "primary_length": primary_length,
        "available_independent_lengths": lengths.tolist(),
        "missing_independent_sizes": missing,
        "missing_size_policy": "never connected or substituted from official data",
        "series": series,
        "selected_panel_states": selected_details,
        "lengths": per_length,
        "official_data": (
            {"source": OFFICIAL_SOURCE, "root": str(official_root)}
            if official_root is not None
            else None
        ),
        "acceptance": {
            "passed": True,
            "validated_lengths": lengths.tolist(),
            "generated_source": INDEPENDENT_SOURCE,
        },
    }
    _atomic_npz(path.with_suffix(".npz"), arrays)
    _atomic_json(path.with_suffix(".json"), metrics)
    return path


def render_fig3_stage(
    output_dir: str | Path,
    length: int,
    *,
    official_data_dir: str | Path | None = DEFAULT_OFFICIAL_DATA,
) -> Path:
    """Task 6 figures-stage hook for one validated length directory."""
    output_dir = Path(output_dir)
    rendered = render_independent_fig3(
        output_dir,
        output_dir=output_dir / "figures",
        official_data_dir=official_data_dir,
    )
    if rendered.name != f"fig3_independent_L{length}.png":
        raise RuntimeError(
            f"Fig. 3 renderer selected {rendered.name}, expected L={length}"
        )
    return rendered


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
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--length", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--independent-results-root",
        type=Path,
        help="root containing hash-validated Task 6 length directories",
    )
    parser.add_argument(
        "--official-data-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_DATA,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.independent_results_root is None:
        path = run_figure(args.length, args.output_dir, args.official_data_dir)
    else:
        path = render_independent_fig3(
            args.independent_results_root,
            output_dir=args.output_dir,
            official_data_dir=args.official_data_dir,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
