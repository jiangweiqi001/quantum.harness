from dataclasses import replace
import json

import h5py
import numpy as np
import pytest
from tenpy.networks.mps import MPS

import pxp_itebd
from pxp_itebd import (
    ITEBDConfig,
    ITEBDSample,
    apply_three_site_gate,
    build_imps,
    configuration_fingerprint,
    evolve_imps,
    itebd_step,
    load_checkpoint,
    measure_sample,
    product_pattern,
    pxp_local_term,
    save_checkpoint,
    three_site_gate,
    trotter_schedule,
    write_results,
)


def test_pxp_local_gate_is_hermitian_and_unitary():
    local = pxp_local_term()
    gate = three_site_gate(0.05)
    assert local.shape == (8, 8)
    np.testing.assert_allclose(local, local.conj().T)
    np.testing.assert_allclose(gate.conj().T @ gate, np.eye(8), atol=1e-13)


def test_second_order_schedule_covers_three_disjoint_colors():
    schedule = trotter_schedule(12, 0.05)
    half_step = 0.025
    expected_layers = [
        [(11, half_step), (2, half_step), (5, half_step), (8, half_step)],
        [(0, half_step), (3, half_step), (6, half_step), (9, half_step)],
        [(1, 0.05), (4, 0.05), (7, 0.05), (10, 0.05)],
        [(0, half_step), (3, half_step), (6, half_step), (9, half_step)],
        [(11, half_step), (2, half_step), (5, half_step), (8, half_step)],
    ]
    assert schedule == [entry for layer in expected_layers for entry in layer]
    assert expected_layers == expected_layers[::-1]

    for layer in expected_layers:
        supports = [
            {(start + offset) % 12 for offset in range(3)}
            for start, _ in layer
        ]
        assert set().union(*supports) == set(range(12))
        assert sum(map(len, supports)) == len(set().union(*supports))

    weighted = {center: 0.0 for center in range(12)}
    for start, tau in schedule:
        weighted[(start + 1) % 12] += tau
    for value in weighted.values():
        assert value == pytest.approx(0.05)


def test_itebd_step_builds_each_distinct_gate_once(monkeypatch):
    generated = []

    def record_gate(tau):
        generated.append(tau)
        return np.eye(8)

    monkeypatch.setattr(pxp_itebd, "three_site_gate", record_gate)
    monkeypatch.setattr(pxp_itebd, "apply_three_site_gate", lambda *args: 0.0)

    assert itebd_step(None, ITEBDConfig(dt=0.05)) == 0.0
    assert generated == [0.025, 0.05]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("vacuum", ["up"] * 12),
        ("Z2", ["down", "up"] * 6),
        ("Z3", ["down", "up", "up"] * 4),
        ("Z4", ["down", "up", "up", "up"] * 3),
    ],
)
def test_product_patterns(label, expected):
    assert product_pattern(label) == expected


def test_builds_canonical_twelve_site_infinite_product_state():
    psi = build_imps("Z2", ITEBDConfig(chi_max=8))
    assert psi.bc == "infinite"
    assert psi.L == 12
    assert max(psi.chi) == 1
    assert np.max(psi.norm_test()) < 1e-12


def test_z2_initial_measurements():
    psi = build_imps("Z2", ITEBDConfig(chi_max=8))
    sample = measure_sample(psi, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(sample.entropy, 0.0, atol=1e-14)
    np.testing.assert_allclose(sample.zz, -1.0)
    assert sample.blockade_violation < 1e-14
    assert sample.max_chi == 1


def test_z3_and_vacuum_measurements_follow_bit_and_wrap_convention():
    config = ITEBDConfig(chi_max=8)
    z3 = measure_sample(build_imps("Z3", config), 0.0, 0.0, 0.0)
    vacuum = measure_sample(build_imps("vacuum", config), 0.0, 0.0, 0.0)

    np.testing.assert_array_equal(z3.zz, [-1.0, 1.0, -1.0] * 4)
    assert z3.zz[-1] == -1.0
    np.testing.assert_array_equal(vacuum.zz, 1.0)
    assert z3.blockade_violation == pytest.approx(0.0, abs=1e-14)
    assert vacuum.blockade_violation == pytest.approx(0.0, abs=1e-14)


def test_evolution_samples_target_and_tracks_partial_interval(monkeypatch):
    config = ITEBDConfig(dt=0.05, sample_dt=0.1, chi_max=16)
    discarded = iter([0.1, 0.2, 0.4])
    monkeypatch.setattr(pxp_itebd, "itebd_step", lambda *args: next(discarded))
    callbacks = []
    samples = evolve_imps(
        build_imps("Z2", config),
        config,
        start_time=0.0,
        target_time=0.15,
        callback=callbacks.append,
    )
    np.testing.assert_allclose([sample.time for sample in samples], [0.0, 0.1, 0.15])
    np.testing.assert_allclose(
        [sample.discarded_interval for sample in samples],
        [0.0, 0.3, 0.4],
    )
    np.testing.assert_allclose(
        [sample.discarded_total for sample in samples],
        [0.0, 0.3, 0.7],
    )
    assert callbacks == samples


def test_evolution_rejects_nonfinite_state_tensor():
    psi = build_imps("Z2", ITEBDConfig())
    psi.set_B(0, psi.get_B(0) * np.nan)

    with pytest.raises(RuntimeError, match="non-finite tensors"):
        evolve_imps(psi, ITEBDConfig(), start_time=0.0, target_time=0.0)


def test_evolution_rejects_blockade_violation():
    config = ITEBDConfig()
    sites = build_imps("vacuum", config).sites
    violating = MPS.from_product_state(
        sites,
        ["down", "down"] + ["up"] * 10,
        bc="infinite",
        unit_cell_width=12,
    )

    with pytest.raises(RuntimeError, match="blockade violation exceeds 1e-8"):
        evolve_imps(violating, config, start_time=0.0, target_time=0.0)


@pytest.mark.parametrize(
    "field",
    [
        "time",
        "max_chi",
        "discarded_interval",
        "discarded_total",
        "blockade_violation",
    ],
)
def test_evolution_rejects_nonfinite_sample_scalars(monkeypatch, field):
    sample = ITEBDSample(
        time=0.0,
        entropy=np.zeros(12),
        zz=np.ones(12),
        max_chi=1,
        discarded_interval=0.0,
        discarded_total=0.0,
        blockade_violation=0.0,
    )
    monkeypatch.setattr(
        pxp_itebd,
        "measure_sample",
        lambda *args: replace(sample, **{field: np.nan}),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        evolve_imps(
            build_imps("vacuum", ITEBDConfig()),
            ITEBDConfig(),
            start_time=0.0,
            target_time=0.0,
        )


@pytest.mark.parametrize("field", ["discarded_interval", "discarded_total"])
def test_evolution_rejects_negative_discarded_weights(monkeypatch, field):
    sample = ITEBDSample(
        time=0.0,
        entropy=np.zeros(12),
        zz=np.ones(12),
        max_chi=1,
        discarded_interval=0.0,
        discarded_total=0.0,
        blockade_violation=0.0,
    )
    monkeypatch.setattr(
        pxp_itebd,
        "measure_sample",
        lambda *args: replace(sample, **{field: -1e-6}),
    )

    with pytest.raises(RuntimeError, match="discarded weights must be nonnegative"):
        evolve_imps(
            build_imps("vacuum", ITEBDConfig()),
            ITEBDConfig(),
            start_time=0.0,
            target_time=0.0,
        )


@pytest.mark.parametrize(
    ("config", "target_time", "message"),
    [
        (ITEBDConfig(dt=0.05, sample_dt=0.075), 0.2, "sample_dt / dt"),
        (ITEBDConfig(dt=0.05, sample_dt=0.1), 0.175, "target_time - start_time"),
    ],
)
def test_evolution_rejects_times_off_the_step_grid(config, target_time, message):
    with pytest.raises(ValueError, match=message):
        evolve_imps(
            build_imps("Z2", config),
            config,
            start_time=0.0,
            target_time=target_time,
        )


def test_one_step_preserves_cell_and_bond_limit():
    config = ITEBDConfig(dt=0.05, chi_max=8, svd_min=1e-13)
    psi = build_imps("Z2", config)
    discarded = itebd_step(psi, config)
    assert psi.L == 12
    assert max(psi.chi) <= 8
    assert discarded >= 0.0
    assert np.max(psi.norm_test()) < 1e-9


@pytest.mark.parametrize("start", [1, 11])
def test_three_site_gate_matches_dense_amplitudes_including_wrapping(start):
    config = ITEBDConfig(chi_max=16, svd_min=1e-13)
    psi = build_imps("Z2", config)
    gate = three_site_gate(0.05)
    local_pattern = product_pattern("Z2")[start:] + product_pattern("Z2")[:start]
    local_pattern = local_pattern[:3]
    basis_index = sum(
        (state == "down") << (2 - offset)
        for offset, state in enumerate(local_pattern)
    )
    expected = gate @ np.eye(8, dtype=complex)[:, basis_index]

    discarded = apply_three_site_gate(psi, start, gate, config)

    theta = psi.get_theta(start, n=3)
    actual = theta.to_ndarray().reshape(-1)
    phase = np.vdot(expected, actual)
    actual *= np.exp(-1j * np.angle(phase))
    assert discarded >= 0.0
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_tenpy_gate_uses_output_legs_before_input_legs():
    config = ITEBDConfig(chi_max=16, svd_min=1e-13)
    psi = build_imps("Z2", config)
    start = 1
    gate = np.roll(np.eye(8, dtype=complex), shift=1, axis=0)
    expected = gate @ np.eye(8, dtype=complex)[:, 2]

    apply_three_site_gate(psi, start, gate, config)

    actual = psi.get_theta(start, n=3).to_ndarray().reshape(-1)
    phase = np.vdot(expected, actual)
    actual *= np.exp(-1j * np.angle(phase))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_checkpoint_round_trip_and_configuration_guard(tmp_path):
    config = ITEBDConfig(dt=0.05, chi_max=8)
    psi = build_imps("Z2", config)
    samples = evolve_imps(psi, config, 0.0, 0.1)
    path = tmp_path / "checkpoint.h5"

    save_checkpoint(path, psi, "Z2", config, 0.1, 0.0, samples)
    with h5py.File(path, "r") as handle:
        root_fingerprint = handle.attrs["configuration_fingerprint"]
        serialized_payload = pxp_itebd.hdf5_io.load_from_hdf5(handle)
    assert serialized_payload["configuration_fingerprint"] == root_fingerprint
    restored, time, discarded, restored_samples = load_checkpoint(
        path, "Z2", config
    )

    assert time == 0.1
    assert discarded == 0.0
    assert len(restored_samples) == len(samples)
    for site in range(psi.L):
        np.testing.assert_allclose(
            restored.get_B(site).to_ndarray(), psi.get_B(site).to_ndarray()
        )
        np.testing.assert_allclose(restored.get_SL(site), psi.get_SL(site))
    restored_observables = measure_sample(restored, time, 0.0, discarded)
    original_observables = measure_sample(psi, time, 0.0, discarded)
    np.testing.assert_allclose(
        restored_observables.entropy, original_observables.entropy
    )
    np.testing.assert_allclose(restored_observables.zz, original_observables.zz)
    assert restored_observables.max_chi == original_observables.max_chi
    assert restored_observables.blockade_violation == pytest.approx(
        original_observables.blockade_violation
    )
    for actual, expected in zip(restored_samples, samples, strict=True):
        assert actual.time == expected.time
        np.testing.assert_array_equal(actual.entropy, expected.entropy)
        np.testing.assert_array_equal(actual.zz, expected.zz)
        assert actual.max_chi == expected.max_chi
        assert actual.discarded_interval == expected.discarded_interval
        assert actual.discarded_total == expected.discarded_total
        assert actual.blockade_violation == expected.blockade_violation
    assert np.max(restored.norm_test()) < 1e-9
    with pytest.raises(ValueError, match="configuration fingerprint"):
        load_checkpoint(path, "Z2", ITEBDConfig(dt=0.025, chi_max=8))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unit_cell", 24),
        ("dt", 0.025),
        ("chi_max", 9),
        ("svd_min", 1e-10),
        ("sample_dt", 0.2),
        ("checkpoint_dt", 2.0),
    ],
)
def test_configuration_fingerprint_is_sensitive_to_every_config_field(
    field, value
):
    config = ITEBDConfig(dt=0.05, chi_max=8)

    assert configuration_fingerprint("Z2", config) == configuration_fingerprint(
        "Z2", config
    )
    assert configuration_fingerprint(
        "Z2", config
    ) != configuration_fingerprint("Z3", config)
    assert configuration_fingerprint(
        "Z2", config
    ) != configuration_fingerprint("Z2", replace(config, **{field: value}))


def test_checkpoint_fingerprint_is_checked_before_tenpy_deserialization(
    tmp_path, monkeypatch
):
    config = ITEBDConfig(dt=0.05, chi_max=8)
    path = tmp_path / "checkpoint.h5"
    save_checkpoint(path, build_imps("Z2", config), "Z2", config, 0.0, 0.0, [])
    with h5py.File(path, "r+") as handle:
        handle.attrs["configuration_fingerprint"] = "mismatch"

    def reject_deserialization(*args, **kwargs):
        raise AssertionError("TeNPy deserialization must not run")

    monkeypatch.setattr(
        pxp_itebd.hdf5_io, "load_from_hdf5", reject_deserialization
    )
    with pytest.raises(ValueError, match="configuration fingerprint"):
        load_checkpoint(path, "Z2", config)


def test_checkpoint_rejects_disagreement_with_serialized_fingerprint(
    tmp_path, monkeypatch
):
    config = ITEBDConfig()
    path = tmp_path / "checkpoint.h5"
    save_checkpoint(path, build_imps("Z2", config), "Z2", config, 0.0, 0.0, [])
    monkeypatch.setattr(
        pxp_itebd.hdf5_io,
        "load_from_hdf5",
        lambda *args, **kwargs: {"configuration_fingerprint": "mismatch"},
    )

    with pytest.raises(ValueError, match="serialized.*fingerprint"):
        load_checkpoint(path, "Z2", config)


def test_checkpoint_loader_documents_trusted_local_input_requirement():
    assert "trusted local" in load_checkpoint.__doc__.lower()


def test_interrupted_checkpoint_write_preserves_valid_checkpoint(
    tmp_path, monkeypatch
):
    config = ITEBDConfig(dt=0.05, chi_max=8)
    psi = build_imps("Z2", config)
    path = tmp_path / "checkpoint.h5"
    save_checkpoint(path, psi, "Z2", config, 0.0, 0.0, [])
    valid_contents = path.read_bytes()

    def fail_after_partial_file(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        pxp_itebd.hdf5_io, "save_to_hdf5", fail_after_partial_file
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        save_checkpoint(path, psi, "Z2", config, 1.0, 0.25, [])

    assert path.read_bytes() == valid_contents
    assert not path.with_suffix(path.suffix + ".part").exists()


def test_checkpoint_conversion_failure_cleans_partial_file(
    tmp_path, monkeypatch
):
    config = ITEBDConfig(dt=0.05, chi_max=8)
    psi = build_imps("Z2", config)
    path = tmp_path / "checkpoint.h5"
    partial_path = path.with_suffix(path.suffix + ".part")
    partial_path.write_bytes(b"stale partial data")

    def fail_conversion(*args, **kwargs):
        raise RuntimeError("simulated conversion failure")

    monkeypatch.setattr(pxp_itebd, "_samples_to_arrays", fail_conversion)
    with pytest.raises(RuntimeError, match="simulated conversion failure"):
        save_checkpoint(path, psi, "Z2", config, 0.0, 0.0, [])

    assert not partial_path.exists()


def test_checkpoint_cleanup_failure_does_not_mask_write_failure(
    tmp_path, monkeypatch
):
    config = ITEBDConfig(dt=0.05, chi_max=8)
    psi = build_imps("Z2", config)
    path = tmp_path / "checkpoint.h5"
    save_checkpoint(path, psi, "Z2", config, 0.0, 0.0, [])
    valid_contents = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise RuntimeError("original write failure")

    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup failure")

    monkeypatch.setattr(pxp_itebd.hdf5_io, "save_to_hdf5", fail_write)
    monkeypatch.setattr(pxp_itebd.Path, "unlink", fail_cleanup)
    with pytest.raises(RuntimeError, match="original write failure"):
        save_checkpoint(path, psi, "Z2", config, 1.0, 0.0, [])

    assert path.read_bytes() == valid_contents


def test_checkpoint_cleanup_propagates_non_oserror(tmp_path, monkeypatch):
    config = ITEBDConfig()
    path = tmp_path / "checkpoint.h5"

    def fail_cleanup(*args, **kwargs):
        raise RuntimeError("unexpected cleanup failure")

    monkeypatch.setattr(pxp_itebd.Path, "unlink", fail_cleanup)
    with pytest.raises(RuntimeError, match="unexpected cleanup failure"):
        save_checkpoint(
            path,
            build_imps("Z2", config),
            "Z2",
            config,
            0.0,
            0.0,
            [],
        )


def _valid_checkpoint_sample_arrays():
    return {
        "time": np.array([0.0]),
        "entropy_by_bond": np.zeros((1, 12)),
        "zz_by_bond": np.ones((1, 12)),
        "max_chi": np.array([1], dtype=np.int64),
        "discarded_interval": np.array([0.0]),
        "discarded_total": np.array([0.0]),
        "blockade_violation": np.array([0.0]),
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda arrays: arrays.pop("zz_by_bond"), "sample schema"),
        (
            lambda arrays: arrays.__setitem__("unexpected", np.zeros(1)),
            "sample schema",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "time", np.array([object()], dtype=object)
            ),
            "numeric dtype",
        ),
        (
            lambda arrays: arrays.__setitem__("time", np.zeros((1, 1))),
            "shape",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "entropy_by_bond", np.zeros((1, 11))
            ),
            "shape",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "max_chi", np.ones(2, dtype=np.int64)
            ),
            "shape",
        ),
        (
            lambda arrays: arrays.__setitem__("max_chi", np.array([1.0])),
            "integer dtype",
        ),
        (
            lambda arrays: arrays.__setitem__("discarded_total", np.array([np.nan])),
            "finite",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "discarded_interval", np.array([-1.0])
            ),
            "nonnegative",
        ),
    ],
)
def test_checkpoint_rejects_invalid_sample_schema(
    tmp_path, monkeypatch, mutate, message
):
    config = ITEBDConfig()
    path = tmp_path / "checkpoint.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["configuration_fingerprint"] = configuration_fingerprint(
            "Z2", config
        )
    arrays = _valid_checkpoint_sample_arrays()
    mutate(arrays)
    monkeypatch.setattr(
        pxp_itebd.hdf5_io,
        "load_from_hdf5",
        lambda *args, **kwargs: {
            "psi": build_imps("Z2", config),
            "configuration_fingerprint": configuration_fingerprint("Z2", config),
            "time": 0.0,
            "discarded_total": 0.0,
            "samples": arrays,
        },
    )

    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        load_checkpoint(path, "Z2", config)


def test_results_are_plain_numerical_hdf5(tmp_path):
    config = ITEBDConfig(dt=0.05, chi_max=8)
    samples = evolve_imps(build_imps("Z2", config), config, 0.0, 0.1)
    path = tmp_path / "results.h5"

    write_results(path, "Z2", config, samples)

    expected = {
        "time": ((len(samples),), "f"),
        "entropy_by_bond": ((len(samples), 12), "f"),
        "zz_by_bond": ((len(samples), 12), "f"),
        "max_chi": ((len(samples),), "i"),
        "discarded_interval": ((len(samples),), "f"),
        "discarded_total": ((len(samples),), "f"),
        "blockade_violation": ((len(samples),), "f"),
    }
    with h5py.File(path, "r") as handle:
        assert set(handle) == set(expected)
        for name, (shape, dtype_kind) in expected.items():
            assert handle[name].shape == shape
            assert handle[name].dtype.kind == dtype_kind
        configuration = json.loads(handle.attrs["configuration"])
        provenance = json.loads(handle.attrs["provenance"])

    assert configuration["initial_state"] == "Z2"
    assert configuration["dt"] == pytest.approx(config.dt)
    assert provenance["configuration_fingerprint"] == configuration_fingerprint(
        "Z2", config
    )
