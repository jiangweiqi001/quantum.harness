import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pytest

import turner2018_fig4 as fig4
import turner2018_l32_server as server
from pxp_ed import _reflect, _rotate, constrained_basis, pxp_hamiltonian
from turner2018_fig4 import (
    FIG4_EXACT_SCHEMA_VERSION,
    OFFICIAL_DENSITY_TOLERANCE,
    PAPER_HISTOGRAM_CENTERS,
    PAPER_HISTOGRAM_EDGES,
    adjacent_gap_ratios,
    analyze_level_statistics,
    compare_official_histogram,
    paper_exact_level_statistics,
    run_paper_exact_reconstruction,
    symmetry_basis_k0_inversion_even,
    theoretical_spacing,
    unfold_spectrum,
)
from turner2018_official import load_fig4_energies, load_fig4_histograms
from turner2018_l32_server import main as server_main


DATA_DIR = Path(".external/official-data/turner-2018")
HAS_FIG4_ARCHIVES = all(
    (DATA_DIR / name).is_file()
    for name in ("energy_eigenvalues.zip", "level_statistics.zip")
)
EXPECTED_WINDOW_SIZES = {28: 3460, 30: 9051, 32: 22731}
EXPECTED_SPACING_COUNTS = {28: 3359, 30: 8950, 32: 22630}
EXPECTED_DIMENSIONS = {28: 13201, 30: 31836, 32: 77436}
EXPECTED_HISTOGRAM_COUNTS = {
    28: [171, 464, 506, 480, 417, 329, 276, 181, 132, 99, 80, 57,
         39, 26, 19, 15, 20, 6, 10, 7, 4, 1, 9, 4],
    30: [420, 997, 1359, 1365, 1169, 953, 736, 524, 416, 277, 224, 145,
         101, 80, 51, 32, 34, 23, 7, 13, 5, 5, 2, 3],
    32: [803, 2269, 3193, 3393, 3229, 2700, 2128, 1599, 1120, 777, 509,
         329, 206, 118, 106, 57, 35, 19, 14, 11, 4, 6, 3, 1],
}


def test_k0_inversion_even_basis_is_orthonormal_and_reduces_hamiltonian():
    length = 10
    basis = constrained_basis(length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(basis, length)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    reduced = transform.T @ hamiltonian @ transform

    np.testing.assert_allclose((transform.T @ transform).toarray(), np.eye(transform.shape[1]))
    np.testing.assert_allclose(reduced.toarray(), reduced.toarray().T)
    assert 0 < reduced.shape[0] < hamiltonian.shape[0]
    index = {int(state): i for i, state in enumerate(basis)}
    dense_transform = transform.toarray()
    for operation in (
        lambda state: _rotate(state, 1, length),
        lambda state: _reflect(state, length),
    ):
        permutation = [index[operation(int(state))] for state in basis]
        transformed = np.empty_like(dense_transform)
        transformed[permutation, :] = dense_transform
        np.testing.assert_allclose(transformed, dense_transform)


def test_adjacent_gap_ratios_ignore_exact_degeneracies():
    energies = np.array([0.0, 1.0, 1.0, 3.0, 6.0, 10.0])

    ratios = adjacent_gap_ratios(energies)

    np.testing.assert_allclose(ratios, [2.0 / 3.0, 0.75])


def test_unfolded_spectrum_has_unit_mean_spacing():
    indices = np.arange(1, 401)
    energies = np.sqrt(indices)

    unfolded = unfold_spectrum(energies, degree=5)

    np.testing.assert_allclose(np.mean(np.diff(unfolded)), 1.0, atol=1e-12)
    assert np.all(np.diff(unfolded) > 0)


def test_theoretical_spacing_curves_are_normalized():
    spacing = np.linspace(0.0, 12.0, 100_001)

    for distribution in ("poisson", "semi-poisson", "goe"):
        density = theoretical_spacing(spacing, distribution)
        np.testing.assert_allclose(np.trapezoid(density, spacing), 1.0, atol=2e-4)


def test_small_pxp_sector_uses_a_monotonic_finite_size_unfolding():
    result = analyze_level_statistics(16)

    assert result["window_kind"] == "finite-size-bulk"
    assert np.all(result["spacings"] > 0)
    np.testing.assert_allclose(
        result["gap_ratios"], adjacent_gap_ratios(result["window_energies"])
    )
    np.testing.assert_allclose(np.mean(result["spacings"]), 1.0)


def test_paper_exact_pipeline_matches_the_stated_operations_without_rescaling():
    energies = np.linspace(-2.5, 3.0, 3000) ** 3
    energies.sort()
    dimension = len(energies)
    expected_window = energies[dimension // 5 : dimension // 2 - 500]
    expected_unfolded = np.polyval(
        np.polyfit(
            expected_window,
            np.arange(len(expected_window), dtype=float),
            deg=3,
        ),
        expected_window,
    )

    result = paper_exact_level_statistics(energies)

    np.testing.assert_array_equal(result["window_energies"], expected_window)
    np.testing.assert_allclose(result["unfolded"], expected_unfolded)
    np.testing.assert_allclose(result["trimmed_unfolded"], expected_unfolded[50:-50])
    np.testing.assert_allclose(result["spacings"], np.diff(expected_unfolded[50:-50]))
    np.testing.assert_array_equal(result["histogram_edges"], np.linspace(0.0, 4.8, 25))
    np.testing.assert_array_equal(result["histogram_centers"], np.arange(0.1, 4.8, 0.2))
    np.testing.assert_allclose(
        np.sum(result["histogram_density"] * np.diff(result["histogram_edges"])),
        1.0,
    )
    assert not np.isclose(np.mean(result["spacings"]), 1.0)


def test_paper_exact_pipeline_does_not_remove_zero_energy():
    energies = np.arange(-800.0, 2200.0)

    result = paper_exact_level_statistics(energies)

    assert 0.0 in result["window_energies"]
    assert len(result["window_energies"]) == 400


@pytest.mark.parametrize(
    ("energies", "message"),
    [
        (np.zeros((2, 1000)), "one-dimensional"),
        (np.r_[np.arange(1099.0), np.nan], "finite"),
        (np.arange(1000.0)[::-1], "sorted"),
        (np.arange(1000.0), "paper window"),
    ],
)
def test_paper_exact_pipeline_rejects_invalid_spectra(energies, message):
    with pytest.raises(ValueError, match=message):
        paper_exact_level_statistics(energies)


def test_official_comparison_validates_sentinel_and_centers():
    reconstruction = paper_exact_level_statistics(np.arange(3000.0) ** 1.1)
    centers = 0.5 * (PAPER_HISTOGRAM_EDGES[:-1] + PAPER_HISTOGRAM_EDGES[1:])
    official = np.column_stack(
        (
            np.r_[0.0, centers],
            np.r_[0.0, reconstruction["histogram_density"]],
        )
    )

    metrics = compare_official_histogram(reconstruction, official)

    assert metrics["max_abs_density_mismatch"] == pytest.approx(0.0)
    bad_sentinel = official.copy()
    bad_sentinel[0, 1] = 1.0
    with pytest.raises(ValueError, match="sentinel"):
        compare_official_histogram(reconstruction, bad_sentinel)
    bad_centers = official.copy()
    bad_centers[1, 0] += 0.01
    with pytest.raises(ValueError, match="centers"):
        compare_official_histogram(reconstruction, bad_centers)


@pytest.mark.parametrize(
    ("row", "value", "message"),
    [
        (1, np.nan, "finite"),
        (1, -0.01, "nonnegative"),
        (1, 1e-3, "tolerance"),
    ],
)
def test_official_comparison_fails_closed_on_invalid_density(row, value, message):
    reconstruction = paper_exact_level_statistics(np.arange(3000.0) ** 1.1)
    official = np.column_stack(
        (
            np.r_[0.0, PAPER_HISTOGRAM_CENTERS],
            np.r_[0.0, reconstruction["histogram_density"]],
        )
    )
    if message == "tolerance":
        official[row, 1] += value
    else:
        official[row, 1] = value

    with pytest.raises(ValueError, match=message):
        compare_official_histogram(reconstruction, official)


def test_official_energy_loader_rejects_wrong_dimension(tmp_path):
    archive = tmp_path / "energy_eigenvalues.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr(
            "evals_periodic_N28_k0_p0.npy.txt",
            "\n".join(str(value) for value in range(10)),
        )

    with pytest.raises(ValueError, match="13201"):
        load_fig4_energies(tmp_path, length=28)


@pytest.mark.skipif(not HAS_FIG4_ARCHIVES, reason="official Fig. 4 archives missing")
@pytest.mark.parametrize("length", [28, 30, 32])
def test_paper_exact_real_data_reproduces_official_histogram(length):
    energies = load_fig4_energies(DATA_DIR, length=length)
    official_x, official_density = load_fig4_histograms(DATA_DIR)[length]

    result = paper_exact_level_statistics(energies)
    official = np.column_stack((official_x, official_density))
    metrics = compare_official_histogram(result, official)

    assert len(energies) == EXPECTED_DIMENSIONS[length]
    assert result["original_dimension"] == EXPECTED_DIMENSIONS[length]
    assert len(result["window_energies"]) == EXPECTED_WINDOW_SIZES[length]
    assert len(result["spacings"]) == EXPECTED_SPACING_COUNTS[length]
    np.testing.assert_allclose(
        result["histogram_centers"], official_x[1:], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.sum(result["histogram_density"] * np.diff(PAPER_HISTOGRAM_EDGES)),
        1.0,
    )
    np.testing.assert_array_equal(
        result["histogram_counts"], EXPECTED_HISTOGRAM_COUNTS[length]
    )
    np.testing.assert_allclose(
        result["histogram_density"], official_density[1:], rtol=0.0, atol=4e-16
    )
    assert metrics["max_abs_density_mismatch"] <= 4e-16


@pytest.mark.skipif(not HAS_FIG4_ARCHIVES, reason="official Fig. 4 archives missing")
def test_paper_exact_runner_writes_pickle_free_outputs(tmp_path):
    paths = run_paper_exact_reconstruction(DATA_DIR, tmp_path)

    assert paths["figure"].name == "fig4_paper_exact.png"
    assert paths["figure"].is_file()
    with np.load(paths["arrays"], allow_pickle=False) as arrays:
        assert arrays["L28_spacings"].shape == (3359,)
        assert arrays["L30_histogram_counts"].shape == (24,)
        assert arrays["L32_official_xy"].shape == (25, 2)
        for name in arrays.files:
            assert arrays[name].dtype.kind != "O", name
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["schema_version"] == FIG4_EXACT_SCHEMA_VERSION
    assert metrics["generator"] == {
        "name": "quantum.harness.turner2018_fig4",
        "version": "1.0.0",
    }
    assert set(metrics["lengths"]) == {"28", "30", "32"}
    assert metrics["invocation"] == {
        "lengths": [28, 30, 32],
        "official_data_dir": str(DATA_DIR.resolve()),
        "output_dir": str(tmp_path.resolve()),
        "density_mismatch_tolerance": OFFICIAL_DENSITY_TOLERANCE,
        "unfolding_degree": 3,
        "edge_trim_levels": 50,
        "spacing_renormalized": False,
        "histogram_edges": PAPER_HISTOGRAM_EDGES.tolist(),
        "histogram_density": True,
    }
    expected_archives = {
        "energy_eigenvalues": (
            DATA_DIR / "energy_eigenvalues.zip",
            [
                "evals_periodic_N28_k0_p0.npy.txt",
                "evals_periodic_N30_k0_p0.npy.txt",
                "evals_periodic_N32_k0_p0.npy.txt",
            ],
        ),
        "level_statistics": (
            DATA_DIR / "level_statistics.zip",
            ["xydataL28.dat", "xydataL30.dat", "xydataL32.dat"],
        ),
    }
    for label, (archive, members) in expected_archives.items():
        provenance = metrics["source_archives"][label]
        assert provenance["resolved_path"] == str(archive.resolve())
        assert provenance["byte_size"] == archive.stat().st_size
        assert provenance["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert provenance["members"] == members
    for length, dimension in EXPECTED_DIMENSIONS.items():
        assert metrics["lengths"][str(length)]["original_dimension"] == dimension
    assert metrics["lengths"]["32"]["max_abs_density_mismatch"] <= 4e-16


@pytest.mark.skipif(not HAS_FIG4_ARCHIVES, reason="official Fig. 4 archives missing")
def test_paper_exact_runner_writes_four_comparison_figures(tmp_path):
    paths = run_paper_exact_reconstruction(DATA_DIR, tmp_path)

    expected = {
        "figure_all": "fig4_comparison_all.png",
        "figure_L28": "fig4_comparison_L28.png",
        "figure_L30": "fig4_comparison_L30.png",
        "figure_L32": "fig4_comparison_L32.png",
    }
    for key, filename in expected.items():
        assert key in paths, f"missing return key {key!r}"
        assert paths[key].name == filename
        assert paths[key].parent == tmp_path
        assert paths[key].is_file()
        assert paths[key].stat().st_size > 0


@pytest.mark.skipif(not HAS_FIG4_ARCHIVES, reason="official Fig. 4 archives missing")
def test_paper_exact_runner_rejects_nonfinite_json_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "turner2018_fig4.compare_official_histogram",
        lambda reconstruction, official_xy: {
            "max_abs_density_mismatch": float("nan"),
        },
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        run_paper_exact_reconstruction(DATA_DIR, tmp_path)


def _validated_independent_result(root: Path, length: int = 10) -> tuple[Path, np.ndarray]:
    output = root / f"L{length}"
    assert server_main(
        [
            "--stage",
            "all",
            "--length",
            str(length),
            "--output-dir",
            str(output),
            "--declared-memory",
            "1G",
            "--chunk-columns",
            "4",
        ]
    ) == 0
    with h5py.File(output / "eigensystem.h5", "r") as handle:
        energies = handle["eigensystem/energies"][()]
    return output, energies


def _synthetic_independent(length: int, energies: np.ndarray) -> dict[str, object]:
    return {
        "source": "independent-ed",
        "directory": Path(f"/validated/L{length}"),
        "length": length,
        "full_dimension": 999_999,
        "sector_dimension": len(energies),
        "energies": energies,
        "source_hashes": {
            "eigensystem.h5": "a" * 64,
            "validation_metrics": "b" * 64,
            "execution_fingerprint": "c" * 64,
        },
        "energy_dataset_metadata": {
            "shape": [len(energies)],
            "dtype": "float64",
            "access": "full-energy-vector-only",
        },
        "eigenvector_metadata": {
            "shape": [len(energies), len(energies)],
            "dtype": "float64",
            "chunks": [len(energies), 1],
            "access": "metadata-only",
        },
    }


def test_independent_renderer_uses_independent_spectra_for_all_generated_metrics(
    tmp_path, monkeypatch
):
    independent_energies = np.linspace(-4.0, 3.0, 4000) ** 3
    independent_energies.sort()
    official_energies = np.linspace(-6.0, 2.0, 4000) ** 3
    official_energies.sort()
    independent_stats = paper_exact_level_statistics(independent_energies)
    official_stats = paper_exact_level_statistics(official_energies)
    official_xy = np.column_stack(
        (
            np.r_[0.0, PAPER_HISTOGRAM_CENTERS],
            np.r_[0.0, official_stats["histogram_density"]],
        )
    )
    monkeypatch.setattr(
        fig4,
        "load_independent_fig4_results",
        lambda _root: {28: _synthetic_independent(28, independent_energies)},
    )
    monkeypatch.setattr(
        fig4,
        "_load_available_official_histograms",
        lambda _root, _lengths: {28: official_xy},
    )

    paths = fig4.render_independent_fig4(
        tmp_path / "independent",
        output_dir=tmp_path / "figure",
        official_data_dir=tmp_path / "official",
    )

    overview = paths["figure_all"]
    with np.load(overview.with_suffix(".npz"), allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["L28_histogram_counts"],
            independent_stats["histogram_counts"],
        )
        np.testing.assert_array_equal(arrays["L28_official_xy"], official_xy)
        assert arrays["L28_source"].item() == "independent-ed"
        assert arrays["L28_official_source"].item() == "official-doi"
    metrics = json.loads(overview.with_suffix(".json").read_text())
    generated = metrics["lengths"]["28"]
    assert generated["source"] == "independent-ed"
    assert generated["window_count"] == len(independent_stats["window_energies"])
    assert generated["spacing_count"] == len(independent_stats["spacings"])
    assert generated["mean_spacing"] == pytest.approx(
        float(np.mean(independent_stats["spacings"]))
    )
    assert generated["histogram_counts"] == independent_stats[
        "histogram_counts"
    ].tolist()
    assert generated["official_mismatch"]["max_abs_density"] == pytest.approx(
        float(
            np.max(
                np.abs(
                    independent_stats["histogram_density"]
                    - official_stats["histogram_density"]
                )
            )
        )
    )
    assert paths["figure_L28"].is_file()
    assert all(
        item["source"] == "independent-ed"
        for name, item in metrics["series"].items()
        if not name.startswith("official_")
    )


def test_independent_renderer_records_missing_sizes_and_small_exact_window(
    tmp_path, monkeypatch
):
    small = np.linspace(-2.0, 2.0, 455)
    large = np.linspace(-3.0, 4.0, 4000) ** 3
    large.sort()
    monkeypatch.setattr(
        fig4,
        "load_independent_fig4_results",
        lambda _root: {
            20: _synthetic_independent(20, small),
            24: _synthetic_independent(24, large),
        },
    )
    monkeypatch.setattr(
        fig4,
        "_load_available_official_histograms",
        lambda _root, _lengths: {},
    )

    paths = fig4.render_independent_fig4(
        tmp_path / "independent",
        output_dir=tmp_path / "figure",
        official_data_dir=None,
    )

    assert set(paths) == {"figure_all", "figure_L20", "figure_L24"}
    metrics = json.loads(paths["figure_all"].with_suffix(".json").read_text())
    assert metrics["available_independent_lengths"] == [20, 24]
    assert metrics["missing_lengths_within_independent_range"] == [22]
    assert metrics["lengths"]["20"]["statistics_available"] is False
    assert metrics["lengths"]["20"]["window_bounds"] == [91, -273]
    assert metrics["lengths"]["20"]["window_count"] == 91
    assert metrics["lengths"]["20"]["histogram_counts"] is None
    assert metrics["lengths"]["24"]["statistics_available"] is True
    assert not (tmp_path / "figure" / "fig4_independent_L22.png").exists()


def test_independent_loader_validates_stable_snapshot_without_reading_vectors(
    tmp_path, monkeypatch
):
    root = tmp_path / "independent"
    _output, expected = _validated_independent_result(root)
    original_getitem = h5py.Dataset.__getitem__

    def reject_vector_read(dataset, key):
        if dataset.name.endswith("/vectors"):
            raise AssertionError("Fig. 4 must never read eigenvectors")
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_vector_read)

    result = fig4.load_independent_fig4_results(root)[10]

    np.testing.assert_array_equal(result["energies"], expected)
    assert result["energy_dataset_metadata"]["access"] == "full-energy-vector-only"
    assert result["eigenvector_metadata"]["access"] == "metadata-only"


def test_independent_loader_rejects_stale_current_execution_fingerprint(
    tmp_path, monkeypatch
):
    root = tmp_path / "independent"
    _validated_independent_result(root)
    changed = json.loads(json.dumps(server.build_execution_fingerprint()))
    changed["sources"]["turner2018_fig4.py"] = "f" * 64
    monkeypatch.setattr(server, "build_execution_fingerprint", lambda: changed)

    with pytest.raises(RuntimeError, match="execution fingerprint"):
        fig4.load_independent_fig4_results(root)


def test_independent_loader_rejects_artifact_hash_corruption(tmp_path):
    root = tmp_path / "independent"
    output, _expected = _validated_independent_result(root)
    with (output / "eigensystem.h5").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        fig4.load_independent_fig4_results(root)


def test_independent_loader_rejects_wrong_scientific_model(tmp_path, monkeypatch):
    root = tmp_path / "independent"
    _validated_independent_result(root)
    monkeypatch.setattr(fig4, "EXPECTED_MODEL", "wrong Hamiltonian")

    with pytest.raises(RuntimeError, match="scientific model"):
        fig4.load_independent_fig4_results(root)


def test_independent_loader_consumes_open_validated_eigensystem_snapshot(
    tmp_path, monkeypatch
):
    root = tmp_path / "independent"
    output, expected = _validated_independent_result(root)
    replacement = tmp_path / "replacement-eigensystem.h5"
    replacement.write_bytes((output / "eigensystem.h5").read_bytes())
    with h5py.File(replacement, "r+") as handle:
        energies = handle["eigensystem/energies"]
        energies[:] = energies[()] + 100.0
    original_require_stage = server.require_stage
    swapped = False

    def swap_after_validation(directory, stage, validation_cache=None):
        nonlocal swapped
        result = original_require_stage(directory, stage, validation_cache)
        if stage == "validate" and not swapped:
            os.replace(replacement, output / "eigensystem.h5")
            swapped = True
        return result

    monkeypatch.setattr(server, "require_stage", swap_after_validation)

    loaded = fig4.load_independent_fig4_results(root)[10]

    np.testing.assert_array_equal(loaded["energies"], expected)


def test_fig4_cli_input_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        fig4.build_parser().parse_args(
            ["--length", "10", "--independent-results-root", "results"]
        )
    with pytest.raises(SystemExit):
        fig4.build_parser().parse_args(
            ["--paper-exact", "--independent-results-root", "results"]
        )


def test_independent_generation_publish_failure_restores_previous_triplet(
    tmp_path, monkeypatch
):
    energies = np.linspace(-3.0, 4.0, 4000) ** 3
    energies.sort()
    monkeypatch.setattr(
        fig4,
        "load_independent_fig4_results",
        lambda _root: {28: _synthetic_independent(28, energies)},
    )
    monkeypatch.setattr(
        fig4,
        "_load_available_official_histograms",
        lambda _root, _lengths: {},
    )
    output = tmp_path / "figure"
    paths = fig4.render_independent_fig4(
        tmp_path / "independent",
        output_dir=output,
        official_data_dir=None,
    )
    target = paths["figure_all"]
    triplet = (target, target.with_suffix(".npz"), target.with_suffix(".json"))
    before = {path: path.read_bytes() for path in triplet}
    original_replace = fig4.os.replace

    def fail_json(source, destination):
        if Path(destination) == target.with_suffix(".json"):
            raise OSError("injected Fig. 4 JSON rename failure")
        return original_replace(source, destination)

    monkeypatch.setattr(fig4.os, "replace", fail_json)
    with pytest.raises(OSError, match="JSON rename failure"):
        fig4.render_independent_fig4(
            tmp_path / "independent",
            output_dir=output,
            official_data_dir=None,
        )

    assert {path: path.read_bytes() for path in triplet} == before
    assert not list(output.glob("*.partial"))
    assert not list(output.glob("*.backup"))


@pytest.mark.parametrize("artifact", ["png", "npz", "json"])
def test_independent_first_generation_partial_failure_leaves_no_triplet(
    tmp_path, monkeypatch, artifact
):
    energies = np.linspace(-3.0, 4.0, 4000) ** 3
    energies.sort()
    monkeypatch.setattr(
        fig4,
        "load_independent_fig4_results",
        lambda _root: {28: _synthetic_independent(28, energies)},
    )
    monkeypatch.setattr(
        fig4,
        "_load_available_official_histograms",
        lambda _root, _lengths: {},
    )
    output = tmp_path / "figure"
    expected = output / "fig4_independent_all.png"

    def fail_partial(partial, _payload):
        with partial.open("wb") as handle:
            handle.write(b"incomplete")
            handle.flush()
            os.fsync(handle.fileno())
        raise OSError(f"injected {artifact} write failure")

    monkeypatch.setattr(fig4, f"_write_{artifact}_partial", fail_partial)
    with pytest.raises(OSError, match=f"{artifact} write failure"):
        fig4.render_independent_fig4(
            tmp_path / "independent",
            output_dir=output,
            official_data_dir=None,
        )

    assert not expected.exists()
    assert not expected.with_suffix(".npz").exists()
    assert not expected.with_suffix(".json").exists()
    assert not list(output.glob("*.partial"))
    assert not list(output.glob("*.backup"))


@pytest.mark.parametrize("with_prior_generation", [False, True])
@pytest.mark.parametrize("artifact", ["png", "npz", "json"])
@pytest.mark.parametrize("failure_phase", ["creation", "flush", "fsync"])
def test_independent_partial_creation_and_fsync_failures_are_cleaned(
    tmp_path,
    monkeypatch,
    with_prior_generation,
    artifact,
    failure_phase,
):
    energies = np.linspace(-3.0, 4.0, 4000) ** 3
    energies.sort()
    monkeypatch.setattr(
        fig4,
        "load_independent_fig4_results",
        lambda _root: {28: _synthetic_independent(28, energies)},
    )
    monkeypatch.setattr(
        fig4,
        "_load_available_official_histograms",
        lambda _root, _lengths: {},
    )
    output = tmp_path / "figure"
    png = output / "fig4_independent_all.png"
    targets = {
        "png": png,
        "npz": png.with_suffix(".npz"),
        "json": png.with_suffix(".json"),
    }
    if with_prior_generation:
        fig4.render_independent_fig4(
            tmp_path / "independent",
            output_dir=output,
            official_data_dir=None,
        )
        before = {path: path.read_bytes() for path in targets.values()}
    else:
        before = {}
    partial = targets[artifact].with_name(targets[artifact].name + ".partial")

    def injected_failure(actual_partial, _payload):
        assert actual_partial == partial
        if failure_phase in {"flush", "fsync"}:
            with actual_partial.open("wb") as handle:
                handle.write(b"incomplete")
                handle.flush()
            raise OSError(f"injected {artifact} {failure_phase} failure")
        raise OSError(f"injected {artifact} creation failure")

    monkeypatch.setattr(fig4, f"_write_{artifact}_partial", injected_failure)
    with pytest.raises(OSError, match=f"{artifact} {failure_phase} failure"):
        fig4.render_independent_fig4(
            tmp_path / "independent",
            output_dir=output,
            official_data_dir=None,
        )

    if with_prior_generation:
        assert {path: path.read_bytes() for path in targets.values()} == before
    else:
        assert not any(path.exists() for path in targets.values())
    assert not list(output.glob("*.partial"))
    assert not list(output.glob("*.backup"))


def test_independent_generation_backup_failure_preserves_previous_triplet(
    tmp_path, monkeypatch
):
    energies = np.linspace(-3.0, 4.0, 4000) ** 3
    energies.sort()
    monkeypatch.setattr(
        fig4,
        "load_independent_fig4_results",
        lambda _root: {28: _synthetic_independent(28, energies)},
    )
    monkeypatch.setattr(
        fig4,
        "_load_available_official_histograms",
        lambda _root, _lengths: {},
    )
    output = tmp_path / "figure"
    paths = fig4.render_independent_fig4(
        tmp_path / "independent",
        output_dir=output,
        official_data_dir=None,
    )
    target = paths["figure_all"]
    triplet = (target, target.with_suffix(".npz"), target.with_suffix(".json"))
    before = {path: path.read_bytes() for path in triplet}
    original_link = fig4.os.link
    calls = 0

    def fail_second_backup(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected Fig. 4 backup failure")
        return original_link(source, destination)

    monkeypatch.setattr(fig4.os, "link", fail_second_backup)
    with pytest.raises(OSError, match="backup failure"):
        fig4.render_independent_fig4(
            tmp_path / "independent",
            output_dir=output,
            official_data_dir=None,
        )

    assert {path: path.read_bytes() for path in triplet} == before
    assert not list(output.glob("*.partial"))
    assert not list(output.glob("*.backup"))
