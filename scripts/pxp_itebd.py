from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import expm
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.tools import hdf5_io


@dataclass(frozen=True)
class ITEBDConfig:
    unit_cell: int = 12
    dt: float = 0.05
    chi_max: int = 400
    svd_min: float = 1e-12
    sample_dt: float = 0.1
    checkpoint_dt: float = 1.0

    def validate(self) -> None:
        if self.unit_cell != 12:
            raise ValueError("the Figure 2 iTEBD unit cell must be 12")
        if self.dt <= 0 or self.sample_dt <= 0 or self.checkpoint_dt <= 0:
            raise ValueError("time intervals must be positive")
        if self.chi_max < 1:
            raise ValueError("chi_max must be positive")

    def fingerprint_payload(self) -> dict[str, float | int]:
        return asdict(self)


_BLOCKADE_VIOLATION_LIMIT = 1e-5


@dataclass
class ITEBDSample:
    time: float
    entropy: np.ndarray
    zz: np.ndarray
    max_chi: int
    discarded_interval: float
    discarded_total: float
    blockade_violation: float

    def validate(self) -> None:
        scalars = (
            self.time,
            self.max_chi,
            self.discarded_interval,
            self.discarded_total,
            self.blockade_violation,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise RuntimeError("iTEBD sample contains non-finite scalar fields")
        if self.discarded_interval < 0.0 or self.discarded_total < 0.0:
            raise RuntimeError("iTEBD discarded weights must be nonnegative")
        if not np.all(np.isfinite(self.entropy)) or not np.all(
            np.isfinite(self.zz)
        ):
            raise RuntimeError("iTEBD sample contains non-finite observables")


_HAMILTONIAN = "sum_i P_(i-1) X_i P_(i+1)"
_SAMPLE_KEYS = frozenset(
    {
        "time",
        "entropy_by_bond",
        "zz_by_bond",
        "max_chi",
        "discarded_interval",
        "discarded_total",
        "blockade_violation",
    }
)
_SAMPLE_SCALAR_KEYS = _SAMPLE_KEYS - {"entropy_by_bond", "zz_by_bond"}


def _configuration_payload(
    initial_state: str, config: ITEBDConfig
) -> dict[str, str | float | int]:
    return {
        "initial_state": initial_state,
        "hamiltonian": _HAMILTONIAN,
        **config.fingerprint_payload(),
    }


def configuration_fingerprint(initial_state: str, config: ITEBDConfig) -> str:
    encoded = json.dumps(
        _configuration_payload(initial_state, config), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sample(sample: ITEBDSample) -> None:
    sample.validate()


def _samples_to_arrays(
    samples: list[ITEBDSample], unit_cell: int
) -> dict[str, np.ndarray]:
    for sample in samples:
        _validate_sample(sample)
    count = len(samples)
    entropy = (
        np.stack([sample.entropy for sample in samples]).astype(float, copy=False)
        if samples
        else np.empty((0, unit_cell), dtype=float)
    )
    zz = (
        np.stack([sample.zz for sample in samples]).astype(float, copy=False)
        if samples
        else np.empty((0, unit_cell), dtype=float)
    )
    expected_shape = (count, unit_cell)
    if entropy.shape != expected_shape or zz.shape != expected_shape:
        raise ValueError(f"sample observables must have shape ({unit_cell},)")
    return {
        "time": np.asarray([sample.time for sample in samples], dtype=float),
        "entropy_by_bond": entropy,
        "zz_by_bond": zz,
        "max_chi": np.asarray([sample.max_chi for sample in samples], dtype=np.int64),
        "discarded_interval": np.asarray(
            [sample.discarded_interval for sample in samples], dtype=float
        ),
        "discarded_total": np.asarray(
            [sample.discarded_total for sample in samples], dtype=float
        ),
        "blockade_violation": np.asarray(
            [sample.blockade_violation for sample in samples], dtype=float
        ),
    }


def _samples_from_arrays(
    arrays: dict[str, np.ndarray], unit_cell: int
) -> list[ITEBDSample]:
    if not isinstance(arrays, dict) or set(arrays) != _SAMPLE_KEYS:
        raise ValueError("checkpoint sample schema has incorrect keys")
    for name, values in arrays.items():
        if not isinstance(values, np.ndarray) or values.dtype.kind not in "iuf":
            raise TypeError(
                f"checkpoint sample {name!r} must have a numeric dtype"
            )
    if arrays["max_chi"].dtype.kind not in "iu":
        raise TypeError("checkpoint sample 'max_chi' must have an integer dtype")
    if arrays["time"].ndim != 1:
        raise ValueError("checkpoint sample 'time' has invalid shape")
    count = len(arrays["time"])
    for name in _SAMPLE_SCALAR_KEYS:
        if arrays[name].shape != (count,):
            raise ValueError(f"checkpoint sample {name!r} has invalid shape")
    for name in ("entropy_by_bond", "zz_by_bond"):
        if arrays[name].shape != (count, unit_cell):
            raise ValueError(f"checkpoint sample {name!r} has invalid shape")
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("checkpoint sample arrays must contain finite values")

    samples = [
        ITEBDSample(
            time=float(arrays["time"][index]),
            entropy=np.asarray(arrays["entropy_by_bond"][index], dtype=float),
            zz=np.asarray(arrays["zz_by_bond"][index], dtype=float),
            max_chi=int(arrays["max_chi"][index]),
            discarded_interval=float(arrays["discarded_interval"][index]),
            discarded_total=float(arrays["discarded_total"][index]),
            blockade_violation=float(arrays["blockade_violation"][index]),
        )
        for index in range(len(arrays["time"]))
    ]
    for sample in samples:
        sample.validate()
    return samples


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def save_checkpoint(
    path: str | Path,
    psi: MPS,
    initial_state: str,
    config: ITEBDConfig,
    time: float,
    discarded_total: float,
    samples: list[ITEBDSample],
) -> None:
    path = Path(path)
    partial_path = path.with_suffix(path.suffix + ".part")
    try:
        _remove_partial_file(partial_path)
        fingerprint = configuration_fingerprint(initial_state, config)
        payload = {
            "psi": psi,
            "configuration_fingerprint": fingerprint,
            "time": float(time),
            "discarded_total": float(discarded_total),
            "samples": _samples_to_arrays(samples, config.unit_cell),
        }
        with h5py.File(partial_path, "w") as handle:
            hdf5_io.save_to_hdf5(handle, payload)
            handle.attrs["configuration_fingerprint"] = fingerprint
            handle.flush()
        partial_path.replace(path)
    finally:
        _remove_partial_file(partial_path)


def load_checkpoint(
    path: str | Path,
    initial_state: str,
    config: ITEBDConfig,
) -> tuple[MPS, float, float, list[ITEBDSample]]:
    """Load a checkpoint created locally from a trusted local input.

    TeNPy reconstructs Python objects while deserializing. Do not load
    checkpoints obtained from untrusted sources.
    """
    expected = configuration_fingerprint(initial_state, config)
    with h5py.File(path, "r") as handle:
        actual = handle.attrs.get("configuration_fingerprint")
        if actual != expected:
            raise ValueError("checkpoint configuration fingerprint does not match")
        payload = hdf5_io.load_from_hdf5(handle)
    if payload.get("configuration_fingerprint") != actual:
        raise ValueError(
            "serialized payload fingerprint does not match root fingerprint"
        )
    return (
        payload["psi"],
        float(payload["time"]),
        float(payload["discarded_total"]),
        _samples_from_arrays(payload["samples"], config.unit_cell),
    )


def write_results(
    path: str | Path,
    initial_state: str,
    config: ITEBDConfig,
    samples: list[ITEBDSample],
) -> None:
    arrays = _samples_to_arrays(samples, config.unit_cell)
    configuration = _configuration_payload(initial_state, config)
    provenance = {
        "configuration_fingerprint": configuration_fingerprint(
            initial_state, config
        ),
        "format_version": 1,
        "generator": "scripts/pxp_itebd.py",
    }
    with h5py.File(path, "w") as handle:
        for name, values in arrays.items():
            handle.create_dataset(name, data=values)
        handle.attrs["configuration"] = json.dumps(configuration, sort_keys=True)
        handle.attrs["provenance"] = json.dumps(provenance, sort_keys=True)


def pxp_local_term() -> np.ndarray:
    projector = np.diag([1.0, 0.0])
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    return np.kron(np.kron(projector, flip), projector)


def three_site_gate(delta_t: float) -> np.ndarray:
    return expm(-1j * delta_t * pxp_local_term())


def _layer_starts(unit_cell: int, color: int) -> list[int]:
    return [((center - 1) % unit_cell) for center in range(color, unit_cell, 3)]


def trotter_schedule(unit_cell: int, delta_t: float) -> list[tuple[int, float]]:
    if unit_cell % 3:
        raise ValueError("unit cell must be divisible by three")
    schedule: list[tuple[int, float]] = []
    for color, weight in ((0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)):
        schedule.extend(
            (start, weight * delta_t) for start in _layer_starts(unit_cell, color)
        )
    return schedule


def product_pattern(initial_state: str, unit_cell: int = 12) -> list[str]:
    periods = {"vacuum": None, "Z2": 2, "Z3": 3, "Z4": 4}
    if initial_state not in periods:
        raise ValueError(f"unknown initial state: {initial_state}")
    period = periods[initial_state]
    if period is None:
        return ["up"] * unit_cell
    if unit_cell % period:
        raise ValueError(f"{initial_state} does not fit unit cell {unit_cell}")
    return ["down" if site % period == 0 else "up" for site in range(unit_cell)]


def build_imps(initial_state: str, config: ITEBDConfig) -> MPS:
    config.validate()
    sites = [SpinHalfSite(conserve=None)] * config.unit_cell
    return MPS.from_product_state(
        sites,
        product_pattern(initial_state, config.unit_cell),
        bc="infinite",
    )


def _tenpy_gate(psi: MPS, start: int, gate: np.ndarray) -> npc.Array:
    legs = [psi.sites[(start + offset) % psi.L].leg for offset in range(3)]
    return npc.Array.from_ndarray(
        gate.reshape((2, 2, 2, 2, 2, 2)),
        legs + [leg.conj() for leg in legs],
        labels=["p0", "p1", "p2", "p0*", "p1*", "p2*"],
    )


def apply_three_site_gate(
    psi: MPS,
    start: int,
    gate: np.ndarray,
    config: ITEBDConfig,
) -> float:
    operator = _tenpy_gate(psi, start, gate)
    psi.apply_local_op(
        start,
        operator,
        unitary=True,
        cutoff=0.0,
        understood_infinite=True,
    )
    error = psi.compress_svd(
        {"chi_max": config.chi_max, "svd_min": config.svd_min}
    )
    if max(psi.chi) > config.chi_max:
        raise RuntimeError("iTEBD bond dimension exceeded chi_max")
    return float(error.eps)


def itebd_step(psi: MPS, config: ITEBDConfig) -> float:
    discarded = 0.0
    gate_cache: dict[float, np.ndarray] = {}
    for start, tau in trotter_schedule(config.unit_cell, config.dt):
        if tau not in gate_cache:
            gate_cache[tau] = three_site_gate(tau)
        gate = gate_cache[tau]
        discarded += apply_three_site_gate(psi, start, gate, config)
    return discarded


def _register_measurement_operators(psi: MPS) -> None:
    bit_z = np.diag([1.0, -1.0])
    bit_occupied = np.diag([0.0, 1.0])
    for site in {id(site): site for site in psi.sites}.values():
        if "BitZ" not in site.opnames:
            site.add_op("BitZ", bit_z)
        if "BitOccupied" not in site.opnames:
            site.add_op("BitOccupied", bit_occupied)


def measure_sample(
    psi: MPS,
    time: float,
    discarded_interval: float,
    discarded_total: float,
) -> ITEBDSample:
    _register_measurement_operators(psi)
    entropy = np.asarray(psi.entanglement_entropy(), dtype=float)
    zz = np.asarray(
        [
            np.real_if_close(
                psi.expectation_value_term(
                    [("BitZ", site), ("BitZ", site + 1)]
                )
            )
            for site in range(psi.L)
        ],
        dtype=float,
    )
    blockade = np.asarray(
        [
            np.real_if_close(
                psi.expectation_value_term(
                    [("BitOccupied", site), ("BitOccupied", site + 1)]
                )
            )
            for site in range(psi.L)
        ],
        dtype=float,
    )
    sample = ITEBDSample(
        time=float(time),
        entropy=entropy,
        zz=zz,
        max_chi=max(psi.chi),
        discarded_interval=float(discarded_interval),
        discarded_total=float(discarded_total),
        blockade_violation=float(np.max(np.abs(blockade))),
    )
    _validate_sample(sample)
    return sample


def _integer_steps(value: float, dt: float, label: str) -> int:
    ratio = value / dt
    steps = round(ratio)
    if abs(ratio - steps) > 1e-10:
        raise ValueError(f"{label} must be an integer within 1e-10")
    return steps


def _state_is_finite(psi: MPS) -> bool:
    return all(
        np.all(np.isfinite(psi.get_B(site).to_ndarray()))
        and np.all(np.isfinite(psi.get_SL(site)))
        for site in range(psi.L)
    )


def evolve_imps(
    psi: MPS,
    config: ITEBDConfig,
    start_time: float,
    target_time: float,
    callback: Callable[[ITEBDSample], None] | None = None,
) -> list[ITEBDSample]:
    config.validate()
    sample_steps = _integer_steps(config.sample_dt, config.dt, "sample_dt / dt")
    total_steps = _integer_steps(
        target_time - start_time,
        config.dt,
        "(target_time - start_time) / dt",
    )
    if sample_steps < 1:
        raise ValueError("sample_dt must span at least one evolution step")
    if total_steps < 0:
        raise ValueError("target_time must not precede start_time")

    samples: list[ITEBDSample] = []
    discarded_interval = 0.0
    discarded_total = 0.0

    def record(time: float) -> None:
        if not _state_is_finite(psi):
            raise RuntimeError("iTEBD state contains non-finite tensors")
        sample = measure_sample(
            psi,
            time,
            discarded_interval,
            discarded_total,
        )
        _validate_sample(sample)
        if sample.blockade_violation > _BLOCKADE_VIOLATION_LIMIT:
            raise RuntimeError(
                "iTEBD blockade violation exceeds 1e-5: "
                f"{sample.blockade_violation:.3e}"
            )
        samples.append(sample)
        if callback is not None:
            callback(sample)

    record(start_time)
    for step in range(1, total_steps + 1):
        discarded = itebd_step(psi, config)
        discarded_interval += discarded
        discarded_total += discarded
        if not _state_is_finite(psi):
            raise RuntimeError("iTEBD state contains non-finite tensors")
        if step % sample_steps == 0:
            record(start_time + step * config.dt)
            discarded_interval = 0.0

    if total_steps % sample_steps:
        record(target_time)

    return samples
