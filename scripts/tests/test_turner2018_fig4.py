import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

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
