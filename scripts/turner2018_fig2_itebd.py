#!/usr/bin/env python3
"""Resumable thermodynamic-limit iTEBD reproduction of Turner et al. Fig. 2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from pxp_itebd import (
    ITEBDConfig,
    ITEBDSample,
    build_imps,
    configuration_fingerprint,
    evolve_imps,
    load_checkpoint,
    save_checkpoint,
    write_results,
)
from turner2018_official import load_fig2_entropy


STATES = ("vacuum", "Z2", "Z3", "Z4")
DEFAULT_OUTPUT = Path("tracks/ed/results/turner-2018/fig2-itebd")
DEFAULT_OFFICIAL_DATA = Path(".external/official-data/turner-2018")
ENTROPY_CUT = 0
COLORS = {
    "vacuum": "tab:gray",
    "Z2": "tab:blue",
    "Z3": "tab:orange",
    "Z4": "tab:purple",
}
RESULT_DATASETS = frozenset(
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
INTERPOLATION_METHOD = "linear numpy.interp"
PXP_HAMILTONIAN = "sum_i P_(i-1) X_i P_(i+1)"
RESULT_FORMAT_VERSION = 1
RESULT_GENERATOR = "scripts/pxp_itebd.py"


def _result_path(output_dir: Path, state: str) -> Path:
    return output_dir / f"fig2_itebd_{state}.h5"


def _checkpoint_path(output_dir: Path, state: str) -> Path:
    return output_dir / f"fig2_itebd_{state}_checkpoint.h5"


def _progress(state: str, sample: ITEBDSample) -> None:
    print(
        f"state={state} time={sample.time:.3f} max_chi={sample.max_chi} "
        f"discarded_total={sample.discarded_total:.6e} "
        f"blockade={sample.blockade_violation:.6e}",
        flush=True,
    )


def _offset_discarded_total(
    samples: list[ITEBDSample], offset: float
) -> list[ITEBDSample]:
    for sample in samples:
        sample.discarded_total += offset
    return samples


def run_state(
    state: str,
    *,
    target_time: float,
    config: ITEBDConfig,
    output_dir: str | Path = DEFAULT_OUTPUT,
    resume: bool = False,
) -> Path:
    """Run one initial state, checkpointing without duplicating resume samples."""
    if state not in STATES:
        raise ValueError(f"unknown initial state: {state}")
    if target_time < 0:
        raise ValueError("target_time must be nonnegative")
    config.validate()
    checkpoint_samples = config.checkpoint_dt / config.sample_dt
    if (
        checkpoint_samples < 1
        or abs(checkpoint_samples - round(checkpoint_samples)) > 1e-10
    ):
        raise ValueError(
            "checkpoint_dt must be an integer multiple of sample_dt so "
            "checkpointing preserves the global sample grid"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(output_dir, state)
    checkpoint_path = _checkpoint_path(output_dir, state)

    if resume and checkpoint_path.is_file():
        psi, current_time, discarded_total, samples = load_checkpoint(
            checkpoint_path,
            state,
            config,
        )
    else:
        psi = build_imps(state, config)
        current_time = 0.0
        discarded_total = 0.0
        samples = []

    tolerance = 1e-10
    if target_time < current_time - tolerance:
        raise ValueError(
            f"target_time {target_time} precedes checkpoint time {current_time}"
        )

    if not samples:
        initial = evolve_imps(
            psi,
            config,
            start_time=current_time,
            target_time=current_time,
        )
        initial = _offset_discarded_total(initial, discarded_total)
        samples.extend(initial)
        for sample in initial:
            _progress(state, sample)

    checkpoint_saved = False
    while current_time < target_time - tolerance:
        checkpoint_index = math.floor(
            (current_time + tolerance) / config.checkpoint_dt
        ) + 1
        next_checkpoint = checkpoint_index * config.checkpoint_dt
        chunk_target = min(target_time, next_checkpoint)
        evolved = evolve_imps(
            psi,
            config,
            start_time=current_time,
            target_time=chunk_target,
        )
        evolved = _offset_discarded_total(evolved, discarded_total)
        if samples and evolved and np.isclose(
            evolved[0].time,
            samples[-1].time,
            rtol=0.0,
            atol=tolerance,
        ):
            evolved = evolved[1:]
        samples.extend(evolved)
        if evolved:
            discarded_total = evolved[-1].discarded_total
            for sample in evolved:
                _progress(state, sample)
        current_time = chunk_target
        save_checkpoint(
            checkpoint_path,
            psi,
            state,
            config,
            current_time,
            discarded_total,
            samples,
        )
        checkpoint_saved = True

    if not checkpoint_saved:
        save_checkpoint(
            checkpoint_path,
            psi,
            state,
            config,
            current_time,
            discarded_total,
            samples,
        )
    write_results(result_path, state, config, samples)
    return result_path


def _load_result(
    path: str | Path,
) -> tuple[str, dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(path, "r") as handle:
        if set(handle) != RESULT_DATASETS:
            raise ValueError(
                f"result schema must contain exactly {sorted(RESULT_DATASETS)}"
            )
        try:
            configuration = json.loads(handle.attrs["configuration"])
            provenance = json.loads(handle.attrs["provenance"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("result metadata schema is invalid") from error
        arrays = {name: handle[name][...] for name in handle}
    if not isinstance(configuration, dict):
        raise ValueError("result configuration metadata must be a JSON object")
    if not isinstance(provenance, dict):
        raise ValueError("result provenance metadata must be a JSON object")
    state = configuration.get("initial_state")
    if state not in STATES:
        raise ValueError(f"result configuration has invalid initial_state: {state!r}")
    if configuration.get("hamiltonian") != PXP_HAMILTONIAN:
        raise ValueError("result Hamiltonian label does not match the PXP model")
    if provenance.get("format_version") != RESULT_FORMAT_VERSION:
        raise ValueError(
            f"result provenance format_version must be {RESULT_FORMAT_VERSION}"
        )
    if provenance.get("generator") != RESULT_GENERATOR:
        raise ValueError(
            f"result provenance generator must be {RESULT_GENERATOR!r}"
        )
    count = len(arrays["time"]) if arrays["time"].ndim == 1 else -1
    if count < 1:
        raise ValueError("result time must be a nonempty one-dimensional array")
    for name in ("entropy_by_bond", "zz_by_bond"):
        if arrays[name].shape != (count, 12):
            raise ValueError(f"{name} must have shape ({count}, 12)")
    for name in RESULT_DATASETS - {"entropy_by_bond", "zz_by_bond"}:
        if arrays[name].shape != (count,):
            raise ValueError(f"{name} must have shape ({count},)")
    if arrays["max_chi"].dtype.kind not in "iu":
        raise TypeError("max_chi must have an integer dtype")
    if any(values.dtype.kind not in "iuf" for values in arrays.values()):
        raise TypeError("result datasets must have numeric dtypes")
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("result datasets must contain only finite values")
    if np.any(np.diff(arrays["time"]) <= 0.0):
        raise ValueError("result time values must be strictly increasing")
    if np.any(arrays["discarded_interval"] < 0.0) or np.any(
        arrays["discarded_total"] < 0.0
    ):
        raise ValueError("discarded weights must be nonnegative")
    if np.any(np.diff(arrays["discarded_total"]) < -1e-15):
        raise ValueError("discarded_total must be nondecreasing")
    fingerprint = provenance.get("configuration_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("result provenance requires configuration_fingerprint")
    config_fields = ITEBDConfig().fingerprint_payload()
    try:
        config = ITEBDConfig(
            **{name: configuration[name] for name in config_fields}
        )
    except (KeyError, TypeError) as error:
        raise ValueError("result configuration schema is invalid") from error
    expected_fingerprint = configuration_fingerprint(state, config)
    if fingerprint != expected_fingerprint:
        raise ValueError("result configuration_fingerprint does not match metadata")
    return state, arrays, {
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "provenance": provenance,
        "path": str(Path(path)),
    }


def _fit_residual(
    time: np.ndarray,
    values: np.ndarray,
    fit_window: tuple[float, float],
    source: str,
) -> tuple[np.ndarray, tuple[float, float]]:
    tolerance = 1e-10
    if (
        time[0] > fit_window[0] + tolerance
        or time[-1] < fit_window[1] - tolerance
    ):
        raise ValueError(
            f"{source} curve does not fully cover requested fit window "
            f"[{fit_window[0]}, {fit_window[1]}]"
        )
    selected = (time >= fit_window[0]) & (time <= fit_window[1])
    if np.count_nonzero(selected) < 2:
        raise ValueError(
            f"{source} fit window must contain at least two points"
        )
    slope, intercept = np.polyfit(time[selected], values[selected], deg=1)
    return values - (slope * time + intercept), (float(slope), float(intercept))


def _comparison(
    generated_time: np.ndarray,
    generated_values: np.ndarray,
    official_time: np.ndarray,
    official_values: np.ndarray,
) -> tuple[float | None, dict[str, object]]:
    selected = (official_time >= generated_time[0]) & (
        official_time <= generated_time[-1]
    )
    if not np.any(selected):
        return None, {
            "count": 0,
            "interval": None,
            "interpolation_method": INTERPOLATION_METHOD,
        }
    interpolated = np.interp(
        official_time[selected],
        generated_time,
        generated_values,
    )
    rmse = float(np.sqrt(np.mean((interpolated - official_values[selected]) ** 2)))
    comparison = {
        "count": int(np.count_nonzero(selected)),
        "interval": [
            float(official_time[selected][0]),
            float(official_time[selected][-1]),
        ],
        "interpolation_method": INTERPOLATION_METHOD,
    }
    return rmse, comparison


def _load_official_correlation(
    official_data_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = official_data_dir / "corzz_Neel.dat"
    if not path.is_file():
        return None
    data = np.loadtxt(path)
    return np.asarray(data[:, 0]), np.asarray(data[:, 1])


def _source_metadata(path: Path) -> dict[str, object]:
    available = path.is_file()
    return {
        "path": str(path),
        "available": available,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if available else None,
    }


def _required_official_sources(
    states: tuple[str, ...],
    official_data_dir: Path,
) -> tuple[Path, ...]:
    sources = [official_data_dir / "entanglement_entropy_growth.zip"]
    if "Z2" in states:
        sources.append(official_data_dir / "corzz_Neel.dat")
    return tuple(sources)


def _preflight_official_sources(
    states: tuple[str, ...],
    official_data_dir: str | Path,
    *,
    allow_missing_official: bool,
) -> tuple[Path, ...]:
    required = _required_official_sources(states, Path(official_data_dir))
    missing = tuple(path for path in required if not path.is_file())
    if missing and not allow_missing_official:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"official Turner source data missing: {paths}; "
            "use --allow-missing-official only for incomplete smoke runs"
        )
    return missing


def _artifact_suffix(states: tuple[str, ...]) -> str:
    return "" if states == STATES else "_" + "-".join(states)


def _mark_incomplete(figure: plt.Figure, reason: str) -> None:
    figure.text(
        0.5,
        0.995,
        f"INCOMPLETE: {reason}",
        ha="center",
        va="top",
        color="crimson",
        fontsize=9,
        fontweight="bold",
    )


def render_figures(
    result_paths: list[str | Path] | tuple[str | Path, ...],
    official_data_dir: str | Path,
    output_dir: str | Path,
    fit_window: tuple[float, float],
    *,
    allow_missing_official: bool = False,
) -> tuple[Path | None, Path]:
    """Render the paper comparison and numerical diagnostics figures."""
    if fit_window[0] >= fit_window[1]:
        raise ValueError("fit window must have fit_start < fit_stop")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, np.ndarray]] = {}
    result_metadata: dict[str, dict[str, object]] = {}
    for path in result_paths:
        state, arrays, metadata = _load_result(path)
        if state in results:
            raise ValueError(f"duplicate result state: {state}")
        results[state] = arrays
        result_metadata[state] = metadata
    if not results:
        raise ValueError("at least one result path is required")
    states = tuple(state for state in STATES if state in results)
    suffix = _artifact_suffix(states)

    official_data_dir = Path(official_data_dir)
    entropy_archive = official_data_dir / "entanglement_entropy_growth.zip"
    correlation_path = official_data_dir / "corzz_Neel.dat"
    missing_sources = _preflight_official_sources(
        states,
        official_data_dir,
        allow_missing_official=allow_missing_official,
    )
    incomplete_reason = (
        "official Turner source data unavailable: "
        + ", ".join(path.name for path in missing_sources)
        if missing_sources
        else ""
    )
    official_entropy = (
        load_fig2_entropy(official_data_dir) if entropy_archive.is_file() else {}
    )
    official_correlation = _load_official_correlation(official_data_dir)

    metrics: dict[str, object] = {
        "selected_entropy_cut": ENTROPY_CUT,
        "fit_window": [float(fit_window[0]), float(fit_window[1])],
        "official_data_complete": not missing_sources,
        "official_sources": {
            "entropy": _source_metadata(entropy_archive),
            "correlation": _source_metadata(correlation_path),
        },
        "comparison": {"interpolation_method": INTERPOLATION_METHOD},
        "states": {},
    }
    if incomplete_reason:
        metrics["incomplete_reason"] = incomplete_reason

    for state in states:
        arrays = results[state]
        time = arrays["time"]
        entropy = arrays["entropy_by_bond"][:, ENTROPY_CUT]
        state_metrics: dict[str, object] = {
            "selected_entropy_cut": ENTROPY_CUT,
            "fit_window": [float(fit_window[0]), float(fit_window[1])],
            "sample_count": int(len(time)),
            "time_range": [float(time[0]), float(time[-1])],
            "configuration_fingerprint": result_metadata[state][
                "configuration_fingerprint"
            ],
            "result_path": result_metadata[state]["path"],
        }
        if state in official_entropy:
            official_time, official_values = official_entropy[state]
            rmse, comparison = _comparison(
                time,
                entropy,
                official_time,
                official_values,
            )
            state_metrics["entropy_comparison"] = comparison
            if rmse is not None:
                state_metrics["official_entropy_rmse"] = rmse
        metrics["states"][state] = state_metrics

    diagnostics, diagnostic_axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(7.4, 9.2),
    )
    for state in states:
        arrays = results[state]
        time = arrays["time"]
        entropy = arrays["entropy_by_bond"]
        color = COLORS[state]
        diagnostic_axes[0].fill_between(
            time,
            np.min(entropy, axis=1),
            np.max(entropy, axis=1),
            color=color,
            alpha=0.12,
        )
        diagnostic_axes[0].plot(
            time,
            np.mean(entropy, axis=1),
            color=color,
            label=f"generated {state} mean (min–max band)",
        )
        diagnostic_axes[1].plot(
            time,
            arrays["max_chi"],
            color=color,
            label=f"generated {state}",
        )
        diagnostic_axes[2].plot(
            time,
            arrays["discarded_interval"],
            color=color,
            label=f"generated {state} interval",
        )
        diagnostic_axes[2].plot(
            time,
            arrays["discarded_total"],
            color=color,
            linestyle="--",
            label=f"generated {state} cumulative",
        )
        diagnostic_axes[3].semilogy(
            time,
            np.maximum(arrays["blockade_violation"], np.finfo(float).tiny),
            color=color,
            label=f"generated {state}",
        )
    diagnostic_axes[0].set_ylabel("entropy")
    diagnostic_axes[1].set_ylabel("max bond dimension")
    diagnostic_axes[2].set_ylabel("discarded weight")
    diagnostic_axes[3].set_ylabel("blockade violation")
    diagnostic_axes[3].set_xlabel("time")
    for axis in diagnostic_axes:
        axis.legend(ncol=2, fontsize=7)
        axis.grid(alpha=0.2)
    if incomplete_reason:
        _mark_incomplete(diagnostics, incomplete_reason)
    diagnostics.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    diagnostics_path = output_dir / f"fig2_itebd_diagnostics{suffix}.png"
    diagnostics.savefig(diagnostics_path, dpi=180)
    plt.close(diagnostics)

    paper_path: Path | None = None
    if "Z2" in results:
        paper, axes = plt.subplots(3, 1, sharex=True, figsize=(7.4, 8.4))
        for state in states:
            arrays = results[state]
            time = arrays["time"]
            entropy = arrays["entropy_by_bond"][:, ENTROPY_CUT]
            axes[0].plot(
                time,
                entropy,
                color=COLORS[state],
                linewidth=2.0,
                label=f"generated iTEBD {state}",
            )
            if state in official_entropy:
                official_time, official_values = official_entropy[state]
                axes[0].plot(
                    official_time,
                    official_values,
                    color=COLORS[state],
                    linestyle="--",
                    linewidth=1.4,
                    label=f"official Turner et al. {state}",
                )

        z2 = results["Z2"]
        z2_time = z2["time"]
        z2_entropy = z2["entropy_by_bond"][:, ENTROPY_CUT]
        z2_residual, generated_fit = _fit_residual(
            z2_time,
            z2_entropy,
            fit_window,
            "generated Z2",
        )
        axes[1].plot(
            z2_time,
            z2_residual,
            color=COLORS["Z2"],
            linewidth=2.0,
            label="generated iTEBD Z2",
        )
        metrics["states"]["Z2"]["generated_fit"] = {
            "slope": generated_fit[0],
            "intercept": generated_fit[1],
        }
        if "Z2" in official_entropy:
            official_time, official_values = official_entropy["Z2"]
            official_residual, official_fit = _fit_residual(
                official_time,
                official_values,
                fit_window,
                "official Z2",
            )
            axes[1].plot(
                official_time,
                official_residual,
                color="tab:green",
                linestyle="--",
                linewidth=1.5,
                label="official Turner et al. Z2",
            )
            metrics["states"]["Z2"]["official_fit"] = {
                "slope": official_fit[0],
                "intercept": official_fit[1],
            }

        generated_zz = np.mean(z2["zz_by_bond"], axis=1)
        axes[2].plot(
            z2_time,
            generated_zz,
            color=COLORS["Z2"],
            linewidth=2.0,
            label="generated iTEBD Z2 correlation",
        )
        if official_correlation is not None:
            official_time, official_zz = official_correlation
            axes[2].plot(
                official_time,
                official_zz,
                color="tab:green",
                linestyle="--",
                linewidth=1.5,
                label="official Turner et al. Z2 correlation",
            )
            rmse, comparison = _comparison(
                z2_time,
                generated_zz,
                official_time,
                official_zz,
            )
            metrics["states"]["Z2"]["correlation_comparison"] = comparison
            if rmse is not None:
                metrics["states"]["Z2"]["official_correlation_rmse"] = rmse

        axes[0].set_ylabel(f"entropy at cut {ENTROPY_CUT}")
        axes[1].set_ylabel("entropy residual")
        axes[2].set_ylabel(r"$\langle Z_i Z_{i+1}\rangle$")
        axes[2].set_xlabel("time")
        for axis in axes:
            axis.legend(ncol=2, fontsize=7)
            axis.grid(alpha=0.2)
        if incomplete_reason:
            _mark_incomplete(paper, incomplete_reason)
        paper.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        paper_path = output_dir / f"fig2_itebd_paper{suffix}.png"
        paper.savefig(paper_path, dpi=180)
        plt.close(paper)

    metrics_path = output_dir / f"fig2_itebd_metrics{suffix}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paper_path, diagnostics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=(*STATES, "all"), default="all")
    parser.add_argument("--target-time", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--chi-max", type=int, default=400)
    parser.add_argument("--sample-dt", type=float, default=0.1)
    parser.add_argument("--checkpoint-dt", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fit-start", type=float, default=0.0)
    parser.add_argument("--fit-stop", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--official-data-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_DATA,
    )
    parser.add_argument(
        "--allow-missing-official",
        action="store_true",
        help="permit visibly incomplete smoke-test figures without official data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    states = STATES if args.state == "all" else (args.state,)
    _preflight_official_sources(
        states,
        args.official_data_dir,
        allow_missing_official=args.allow_missing_official,
    )
    config = ITEBDConfig(
        dt=args.dt,
        chi_max=args.chi_max,
        sample_dt=args.sample_dt,
        checkpoint_dt=args.checkpoint_dt,
    )
    result_paths = []
    for state in states:
        result_paths.append(
            run_state(
                state,
                target_time=args.target_time,
                config=config,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        )
        gc.collect()
    paper, diagnostics = render_figures(
        result_paths,
        args.official_data_dir,
        args.output_dir,
        fit_window=(args.fit_start, args.fit_stop),
        allow_missing_official=args.allow_missing_official,
    )
    if paper is not None:
        print(paper)
    print(diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
