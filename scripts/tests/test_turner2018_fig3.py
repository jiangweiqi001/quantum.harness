import hashlib
import json
from zipfile import ZipFile

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pxp_ed import constrained_basis, density_wave_state, pxp_hamiltonian
import turner2018_fig3 as fig3
from turner2018_l32_server import atomic_write_json, main as server_main
from turner2018_fig3 import (
    analyze_spectrum,
    fsa_basis,
    participation_ratio,
    plot_official_pr2_scaling,
)


def test_fsa_basis_is_orthonormal_and_resolves_hamming_distance():
    length = 10
    basis = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
    z2 = density_wave_state(length, 2)

    vectors, beta = fsa_basis(hamiltonian, basis, z2, length)

    np.testing.assert_allclose(vectors.conj() @ vectors.T, np.eye(length + 1))
    assert len(beta) == length
    for distance, vector in enumerate(vectors):
        support = np.flatnonzero(np.abs(vector) > 1e-12)
        assert all((int(basis[i]) ^ z2).bit_count() == distance for i in support)


def test_overlap_weights_and_participation_ratios_are_normalized():
    result = analyze_spectrum(length=10)

    assert len(result["energies"]) == len(result["overlap_z2"])
    np.testing.assert_allclose(np.sum(result["overlap_z2"]), 0.5, atol=1e-12)
    assert np.all(result["participation_ratio"] > 0.0)
    assert np.all(result["participation_ratio"] <= 1.0)
    assert len(result["fsa_energies"]) == 6
    np.testing.assert_allclose(np.sum(result["fsa_overlap_z2"]), 0.5, atol=1e-12)
    assert result["exact_shell_amplitudes"].shape == (6, len(result["energies"]))
    assert result["fsa_eigenvectors"].shape == (6, 6)


def test_participation_ratio_distinguishes_product_and_uniform_states():
    product = np.array([1.0, 0.0, 0.0, 0.0])
    uniform = np.ones(4) / 2

    assert participation_ratio(product) == 1.0
    assert participation_ratio(uniform) == 0.25


def test_official_pr2_available_sizes_are_unconnected_markers():
    figure, axis = plt.subplots()
    scaling = {
        "length": np.array([26, 28, 30, 32]),
        "other": np.array([0.1, np.nan, np.nan, 0.01]),
        "special": np.array([0.2, np.nan, np.nan, 0.02]),
        "available": np.array([True, False, False, True]),
    }

    lines = plot_official_pr2_scaling(axis, scaling)

    assert len(lines) == 2
    assert all(line.get_linestyle() == "None" for line in lines)
    for line in lines:
        np.testing.assert_array_equal(line.get_xdata(), [26, 32])
    plt.close(figure)


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rewrite_manifest(path, update):
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    atomic_write_json(path, payload)


def _synthetic_independent_result(root, length=10):
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

    with h5py.File(output / "eigensystem.h5", "r+") as handle:
        energies = handle["eigensystem/energies"]
        energies[:] = energies[:] + 0.375
    with h5py.File(output / "observables.h5", "r+") as handle:
        overlap = handle["observables/overlap_z2"]
        synthetic_overlap = overlap[()][::-1]
        overlap[:] = synthetic_overlap
        pr2 = handle["observables/participation_ratio"]
        synthetic_pr2 = np.linspace(0.011, 0.029, pr2.size)
        pr2[:] = synthetic_pr2

    diagonalize_manifest = output / "stages" / "diagonalize.json"
    _rewrite_manifest(
        diagonalize_manifest,
        lambda payload: payload["artifact"].update(
            sha256=_sha256(output / "eigensystem.h5")
        ),
    )
    observables_manifest = output / "stages" / "observables.json"
    _rewrite_manifest(
        observables_manifest,
        lambda payload: (
            payload["artifact"].update(sha256=_sha256(output / "observables.h5")),
            payload["inputs"].update(
                diagonalize=_sha256(diagonalize_manifest)
            ),
        ),
    )
    validation_path = output / "validation" / "metrics.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["validated_stage_sha256"]["diagonalize"] = _sha256(
        output / "eigensystem.h5"
    )
    validation["validated_stage_sha256"]["observables"] = _sha256(
        output / "observables.h5"
    )
    atomic_write_json(validation_path, validation)
    validate_manifest = output / "stages" / "validate.json"
    _rewrite_manifest(
        validate_manifest,
        lambda payload: (
            payload["artifact"].update(sha256=_sha256(validation_path)),
            payload["inputs"].update(
                diagonalize=_sha256(diagonalize_manifest),
                observables=_sha256(observables_manifest),
            ),
        ),
    )
    return output, synthetic_overlap, synthetic_pr2


def test_independent_renderer_uses_only_validated_artifacts_for_generated_series(
    tmp_path, monkeypatch
):
    independent_root = tmp_path / "independent"
    output, synthetic_overlap, synthetic_pr2 = _synthetic_independent_result(
        independent_root
    )
    with h5py.File(output / "eigensystem.h5", "r") as handle:
        synthetic_energies = handle["eigensystem/energies"][()]

    official_root = tmp_path / "official"
    official_root.mkdir()
    for name in (
        "energy_eigenvalues.zip",
        "forward-scattering.zip",
        "participation_ratios.zip",
    ):
        (official_root / name).touch()
    with ZipFile(official_root / "overlaps_with_Neel_state.zip", "w") as archive:
        archive.writestr("oneel_periodic_N10_k0_p0.dat", "synthetic")
    with ZipFile(official_root / "eigendecomposition.zip", "w") as archive:
        archive.writestr("eigs_periodic_N10_k0_p0.h5", b"synthetic")
    official_energies = synthetic_energies + 9.0
    official_overlap = synthetic_overlap + 7.0
    monkeypatch.setattr(
        fig3,
        "load_fig3_overlap",
        lambda _root, *, length: (official_energies, official_overlap),
    )
    monkeypatch.setattr(
        fig3,
        "load_fig3_pr2_scaling",
        lambda _root: {
            "length": np.array([10]),
            "other": np.array([0.91]),
            "special": np.array([0.92]),
            "available": np.array([True]),
        },
    )
    original_getitem = h5py.Dataset.__getitem__

    def reject_full_eigenvector_read(dataset, key):
        if dataset.name.endswith("/vectors"):
            full = key == slice(None)
            if isinstance(key, tuple):
                full = all(part == slice(None) for part in key)
            if full:
                raise AssertionError("renderer must not materialize eigenvectors")
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_full_eigenvector_read)

    figure_path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=official_root,
    )

    assert figure_path.name == "fig3_independent_L10.png"
    with np.load(figure_path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        np.testing.assert_array_equal(
            sidecar["panel_a_L10_energies"], synthetic_energies
        )
        np.testing.assert_array_equal(
            sidecar["panel_a_L10_overlap_z2"], synthetic_overlap
        )
        assert not np.array_equal(
            sidecar["panel_a_L10_energies"], official_energies
        )
        np.testing.assert_array_equal(
            sidecar["official_L10_energies"], official_energies
        )
        assert sidecar["panel_a_L10_source"].item() == "independent-ed"
        assert sidecar["panel_d_source"].item() == "independent-ed"
        assert sidecar["official_L10_source"].item() == "official-doi"

    metrics = json.loads(figure_path.with_suffix(".json").read_text())
    assert metrics["source"] == "independent-ed"
    assert metrics["lengths"]["10"]["overlap_sum"] == pytest.approx(
        float(np.sum(synthetic_overlap))
    )
    assert metrics["lengths"]["10"]["pr2"]["all_values_sha256"] == hashlib.sha256(
        synthetic_pr2.tobytes()
    ).hexdigest()
    assert all(
        series["source"] == "independent-ed"
        for name, series in metrics["series"].items()
        if not name.startswith("official_")
    )
    assert all(
        series["source"] == "official-doi"
        for name, series in metrics["series"].items()
        if name.startswith("official_")
    )
    mismatch = metrics["lengths"]["10"]["official_mismatch"]
    assert mismatch["energies_max_abs"] == pytest.approx(9.0)
    assert mismatch["overlap_max_abs"] == pytest.approx(7.0)


def test_missing_independent_size_is_recorded_without_official_substitution(tmp_path):
    independent_root = tmp_path / "independent"
    _synthetic_independent_result(independent_root, length=10)
    _synthetic_independent_result(independent_root, length=14)

    figure_path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=None,
    )

    with np.load(figure_path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        np.testing.assert_array_equal(sidecar["panel_d_lengths"], [10, 14])
        assert not any(name.startswith("official_") for name in sidecar.files)
    metrics = json.loads(figure_path.with_suffix(".json").read_text())
    assert metrics["missing_independent_sizes"] == [12]
    assert metrics["available_independent_lengths"] == [10, 14]


def test_official_overlap_overlay_is_skipped_when_requested_member_is_absent(
    tmp_path, monkeypatch
):
    independent_root = tmp_path / "independent"
    _synthetic_independent_result(independent_root, length=10)
    official_root = tmp_path / "official"
    official_root.mkdir()
    with ZipFile(official_root / "overlaps_with_Neel_state.zip", "w") as archive:
        archive.writestr("oneel_periodic_N22_k0_p0.dat", "0.0\n")
    with ZipFile(official_root / "eigendecomposition.zip", "w") as archive:
        archive.writestr("unrelated", b"")
    monkeypatch.setattr(
        fig3,
        "load_fig3_overlap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("absent official member must not be loaded")
        ),
    )

    path = fig3.render_independent_fig3(
        independent_root,
        output_dir=tmp_path / "figure",
        official_data_dir=official_root,
    )

    with np.load(path.with_suffix(".npz"), allow_pickle=False) as sidecar:
        assert not any(name.startswith("official_L10") for name in sidecar.files)


def test_legacy_small_l_renderer_still_writes_analyze_spectrum_arrays(tmp_path):
    expected = analyze_spectrum(10)

    path = fig3.run_figure(10, output_dir=tmp_path, official_data_dir=None)

    assert path.name == "fig3_L10.png"
    with np.load(tmp_path / "fig3_L10.npz", allow_pickle=False) as persisted:
        np.testing.assert_allclose(persisted["energies"], expected["energies"])
        np.testing.assert_allclose(persisted["overlap_z2"], expected["overlap_z2"])
