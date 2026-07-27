import json
from pathlib import Path

import numpy as np
import pytest

from pxp_ed import constrained_basis, density_wave_state
from turner2018_fig2 import (
    detrend_linear,
    evolve_observables,
    half_chain_entropy,
    load_official_correlation,
    run_figure,
)


def test_half_chain_entropy_of_product_and_cat_states():
    length = 8
    basis = constrained_basis(length, pbc=True)
    product = np.zeros(len(basis), dtype=complex)
    product[np.flatnonzero(basis == density_wave_state(length, 2))[0]] = 1
    cat = product.copy()
    cat[np.flatnonzero(basis == density_wave_state(length, 2, offset=1))[0]] = 1
    cat /= np.linalg.norm(cat)

    assert half_chain_entropy(product, basis, length) == pytest.approx(0.0)
    assert half_chain_entropy(cat, basis, length) == pytest.approx(np.log(2))


def test_linear_detrending_removes_a_linear_background():
    time = np.linspace(0, 5, 51)
    np.testing.assert_allclose(detrend_linear(time, 2.5 * time + 0.7), 0.0, atol=1e-12)


def test_z2_dynamics_starts_from_expected_observables():
    result = evolve_observables(
        length=10,
        initial_period=2,
        times=np.array([0.0, 0.1]),
        compute_entropy=True,
    )

    assert result["fidelity"][0] == pytest.approx(1.0)
    assert result["zz"][0] == pytest.approx(-1.0)
    assert result["entropy"][0] == pytest.approx(0.0)


@pytest.mark.skipif(
    not Path(".external/official-data/turner-2018/corzz_Neel.dat").is_file(),
    reason="official Turner 2018 correlation data not downloaded",
)
def test_finite_ed_matches_official_z2_correlation_at_early_time():
    official_time, official_zz = load_official_correlation(
        ".external/official-data/turner-2018/corzz_Neel.dat"
    )
    mask = official_time <= 1.0
    result = evolve_observables(
        length=14,
        initial_period=2,
        times=official_time[mask],
        compute_entropy=False,
    )

    np.testing.assert_allclose(result["zz"], official_zz[mask], atol=2e-3)


def test_figure_outputs_are_pickle_free_and_include_official_comparison(tmp_path):
    official = tmp_path / "official.dat"
    official.write_text("0.0 -1.0\n0.1 -0.9800669\n", encoding="utf-8")

    run_figure(
        length=12,
        t_max=0.2,
        dt=0.1,
        output_dir=tmp_path,
        official_correlation_path=official,
        official_data_dir=None,
    )

    with np.load(tmp_path / "fig2_L12.npz", allow_pickle=False) as data:
        assert {
            "time",
            "z2_fidelity",
            "z2_entropy",
            "z2_zz",
            "z3_entropy",
            "z4_entropy",
            "vacuum_entropy",
        } <= set(data.files)
    comparison = json.loads(
        (tmp_path / "fig2_L12_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["zz_official_points"] == 2
    assert comparison["zz_rmse"] < 1e-4


def test_figure_requires_a_ring_commensurate_with_z2_z3_and_z4(tmp_path):
    with pytest.raises(ValueError, match="divisible by 12"):
        run_figure(
            length=14,
            t_max=0.2,
            dt=0.1,
            output_dir=tmp_path,
            official_correlation_path=None,
            official_data_dir=None,
        )


def test_figure_skips_official_comparison_when_time_ranges_do_not_overlap(tmp_path):
    official = tmp_path / "official.dat"
    official.write_text("10.0 -1.0\n10.1 -0.9\n", encoding="utf-8")
    comparison_path = tmp_path / "fig2_L12_comparison.json"
    comparison_path.write_text('{"stale": true}\n', encoding="utf-8")

    run_figure(
        length=12,
        t_max=0.2,
        dt=0.1,
        output_dir=tmp_path,
        official_correlation_path=official,
        official_data_dir=None,
    )

    assert not comparison_path.exists()
