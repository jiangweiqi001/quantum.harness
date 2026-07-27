#!/usr/bin/env python3
"""Restartable, fail-closed Turner PXP ED/FSA server workflow.

The L=32 path is intentionally blocked until this checkout contains a local
QuSpin 1.0.1 imported/user-basis validation proof. Small systems remain
available for validating the stage mechanics and existing reference code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

L32_FULL_DIMENSION = 4_870_847
L32_SECTOR_DIMENSION = 77_436
FULL_FSA_SHELLS = 33
SYMMETRY_FOLDED_FSA_SHELLS = 17
SCHEMA_VERSION = "turner-l32-ed-fsa-v1"
QUANTUM_MODEL = "H=sum_j P_(j-1) X_j P_(j+1), PBC"
STAGES = ("plan", "basis", "hamiltonian", "diagonalize", "observables", "all")
STAGE_PREDECESSOR = {
    "hamiltonian": "basis",
    "diagonalize": "hamiltonian",
    "observables": "diagonalize",
}


def dense_resource_estimate(dimension: int) -> dict[str, int | float]:
    """Return decimal-GB storage estimates for real dense arrays."""
    bytes_per_array = int(dimension) ** 2 * 8
    return {
        "dimension": int(dimension),
        "bytes_per_dense_array": bytes_per_array,
        "gb_per_dense_array": bytes_per_array / 1e9,
        "gib_per_dense_array": bytes_per_array / 2**30,
        "matrix_plus_eigenvectors_gb": 2 * bytes_per_array / 1e9,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON through a sibling .partial file and atomically rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def require_stage(output_dir: str | Path, stage: str) -> dict[str, Any]:
    """Load a completed stage manifest or fail before touching downstream data."""
    path = Path(output_dir) / "stages" / f"{stage}.json"
    if not path.is_file():
        raise RuntimeError(f"required stage {stage!r} is not complete: missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or payload.get("status") != "complete":
        raise RuntimeError(f"required stage {stage!r} is not complete: {path}")
    return payload


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "h5py", "quspin"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def build_plan(length: int, output_dir: Path, argv: list[str]) -> dict[str, Any]:
    """Build the immutable scientific/resource plan and current readiness result."""
    dimension = L32_SECTOR_DIMENSION if length == 32 else None
    packages = _package_versions()
    quspin_proof_present = packages["quspin"] == "1.0.1" and (
        output_dir / "validation" / "quspin_imported_basis_proof.json"
    ).is_file()
    readiness = "ready" if length != 32 else "blocked"
    blocker = None
    if readiness == "blocked":
        blocker = (
            "L=32 direct QuSpin imported/user-basis translation/reflection "
            "reduction is not locally proven; a proof file alone does not "
            "enable the unimplemented adapter"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "readiness": readiness,
        "blocker": blocker,
        "invocation": argv,
        "model": {
            "hamiltonian": QUANTUM_MODEL,
            "length": length,
            "boundary": "periodic",
            "momentum": 0,
            "inversion": "even",
        },
        "basis": {
            "constructor": "QuSpin 1.0.1 imported/user basis",
            "enumeration": "direct constrained states; never scan 2**L",
            "symmetry_reduction": "direct translation then reflection",
            "full_constrained_dimension": (
                L32_FULL_DIMENSION if length == 32 else None
            ),
            "sector_dimension": dimension,
        },
        "fsa": {
            "full_shell_count": length + 1,
            "symmetry_folded_shell_count": length // 2 + 1,
        },
        "solver": {
            "dense_dtype": "float64",
            "dense_order": "Fortran",
            "routine": "scipy.linalg.eigh",
            "driver": "evd",
            "overwrite_a": True,
            "check_finite": False,
        },
        "observables": {
            "pr2": "sum_i |V_ij|**4, streamed by eigenvector-column chunks",
        },
        "resources": dense_resource_estimate(dimension or 0),
        "artifacts": {
            "basis": "basis.npz",
            "hamiltonian": "hamiltonian.csr.npz",
            "results": "results.h5",
            "validation": "validation/metrics.json",
        },
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "quspin_proof_present": quspin_proof_present,
            "git_revision": _git_revision(),
        },
    }


def write_plan(length: int, output_dir: Path, argv: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(length, output_dir, argv)
    atomic_write_json(output_dir / "manifest.json", plan)
    atomic_write_json(
        output_dir / "stages" / "plan.json",
        {"stage": "plan", "status": "complete", "readiness": plan["readiness"]},
    )
    return plan


def _refuse_unproven_l32(length: int, output_dir: Path) -> None:
    if length != 32:
        return
    proof = output_dir / "validation" / "quspin_imported_basis_proof.json"
    packages = _package_versions()
    if packages["quspin"] != "1.0.1" or not proof.is_file():
        raise RuntimeError(
            "refusing L=32: no validated QuSpin 1.0.1 imported-basis proof; "
            "run and inspect local small-L equivalence first"
        )
    payload = json.loads(proof.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("quspin_version") != "1.0.1":
        raise RuntimeError(
            "refusing L=32: validated QuSpin 1.0.1 imported-basis proof is incomplete"
        )
    # A proof file alone is deliberately insufficient while the direct imported
    # basis adapter remains unimplemented in this checkout.
    raise RuntimeError(
        "refusing L=32: QuSpin direct imported-basis adapter is not locally proven"
    )


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, path)


def run_basis(length: int, output_dir: Path) -> None:
    _refuse_unproven_l32(length, output_dir)
    from pxp_ed import constrained_basis, symmetry_basis_k0_inversion_even

    basis = constrained_basis(length, pbc=True)
    transform = symmetry_basis_k0_inversion_even(basis, length)
    _atomic_save_npz(
        output_dir / "basis.npz",
        states=basis,
        transform_data=transform.data,
        transform_indices=transform.indices,
        transform_indptr=transform.indptr,
        transform_shape=transform.shape,
    )
    atomic_write_json(
        output_dir / "stages" / "basis.json",
        {
            "stage": "basis",
            "status": "complete",
            "full_dimension": len(basis),
            "sector_dimension": transform.shape[1],
            "artifact_sha256": _sha256(output_dir / "basis.npz"),
        },
    )


def _load_basis(path: Path):
    import numpy as np
    import scipy.sparse as sp

    with np.load(path, allow_pickle=False) as data:
        states = data["states"]
        shape = tuple(int(value) for value in data["transform_shape"])
        transform = sp.csc_matrix(
            (data["transform_data"], data["transform_indices"], data["transform_indptr"]),
            shape=shape,
        )
    return states, transform


def run_hamiltonian(length: int, output_dir: Path) -> None:
    import scipy.sparse as sp

    require_stage(output_dir, "basis")
    _refuse_unproven_l32(length, output_dir)
    from pxp_ed import pxp_hamiltonian

    states, transform = _load_basis(output_dir / "basis.npz")
    full = pxp_hamiltonian(states, length, pbc=True)
    reduced = (transform.T @ full @ transform).tocsr()
    target = output_dir / "hamiltonian.csr.npz"
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as handle:
        sp.save_npz(handle, reduced)
    os.replace(partial, target)
    atomic_write_json(
        output_dir / "stages" / "hamiltonian.json",
        {
            "stage": "hamiltonian",
            "status": "complete",
            "shape": list(reduced.shape),
            "nnz": reduced.nnz,
            "artifact_sha256": _sha256(target),
        },
    )


def run_diagonalize(length: int, output_dir: Path) -> None:
    import h5py
    import numpy as np
    import scipy.linalg
    import scipy.sparse as sp

    require_stage(output_dir, "hamiltonian")
    _refuse_unproven_l32(length, output_dir)
    sparse = sp.load_npz(output_dir / "hamiltonian.csr.npz")
    dense = np.asfortranarray(sparse.toarray(), dtype=np.float64)
    if not dense.flags.f_contiguous or dense.dtype != np.float64:
        raise RuntimeError("dense Hamiltonian must be Fortran-contiguous float64")
    energies, vectors = scipy.linalg.eigh(
        dense,
        driver="evd",
        overwrite_a=True,
        check_finite=False,
    )
    target = output_dir / "results.h5"
    partial = target.with_name(target.name + ".partial")
    with h5py.File(partial, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        eig = handle.create_group("eigensystem")
        eig.create_dataset("energies", data=energies)
        eig.create_dataset("vectors", data=vectors, chunks=(vectors.shape[0], 1))
    os.replace(partial, target)
    atomic_write_json(
        output_dir / "stages" / "diagonalize.json",
        {
            "stage": "diagonalize",
            "status": "complete",
            "dimension": len(energies),
            "artifact_sha256": _sha256(target),
        },
    )


def _stream_pr2(vectors: Any, chunk_columns: int) -> Any:
    import numpy as np

    result = np.empty(vectors.shape[1], dtype=np.float64)
    for start in range(0, vectors.shape[1], chunk_columns):
        stop = min(start + chunk_columns, vectors.shape[1])
        block = vectors[:, start:stop]
        squared = np.square(block, dtype=np.float64)
        result[start:stop] = np.sum(squared * squared, axis=0)
    return result


def run_observables(length: int, output_dir: Path, chunk_columns: int) -> None:
    import h5py
    import numpy as np

    require_stage(output_dir, "diagonalize")
    _refuse_unproven_l32(length, output_dir)
    from pxp_ed import density_wave_state, pxp_hamiltonian
    from turner2018_fig3 import fsa_basis

    states, transform = _load_basis(output_dir / "basis.npz")
    z2_state = density_wave_state(length, 2)
    full_h = pxp_hamiltonian(states, length, pbc=True)
    full_shells, beta = fsa_basis(full_h, states, z2_state, length)
    folded = np.asarray(transform.T @ full_shells.T).T[: length // 2 + 1]
    folded /= np.linalg.norm(folded, axis=1)[:, None]

    path = output_dir / "results.h5"
    partial = path.with_name(path.name + ".partial")
    with h5py.File(path, "r") as source, h5py.File(partial, "w") as target:
        source.copy("eigensystem", target)
        vectors = source["eigensystem/vectors"]
        observables = target.create_group("observables")
        observables.create_dataset("pr2", data=_stream_pr2(vectors, chunk_columns))
        fsa = target.create_group("fsa")
        fsa.create_dataset("full_shell_vectors", data=full_shells)
        fsa.create_dataset("full_beta", data=beta)
        fsa.create_dataset("symmetry_folded_shell_vectors", data=folded)
        fsa.attrs["full_shell_count"] = length + 1
        fsa.attrs["symmetry_folded_shell_count"] = length // 2 + 1
    os.replace(partial, path)
    atomic_write_json(
        output_dir / "stages" / "observables.json",
        {
            "stage": "observables",
            "status": "complete",
            "pr2_chunk_columns": chunk_columns,
            "full_fsa_shell_count": length + 1,
            "folded_fsa_shell_count": length // 2 + 1,
            "artifact_sha256": _sha256(path),
        },
    )


def validate_small_l(
    length: int,
    official_data_dir: str | Path | None,
) -> dict[str, float | int | str]:
    """Compare staged small-L quantities to the existing ED/FSA implementation."""
    import numpy as np

    from turner2018_fig3 import analyze_spectrum

    reference = analyze_spectrum(length)
    repeated = analyze_spectrum(length)
    metrics: dict[str, float | int | str] = {
        "length": length,
        "sector_matrix_max_abs": float(
            np.max(
                np.abs(
                    reference["fsa_hamiltonian_sector"]
                    - repeated["fsa_hamiltonian_sector"]
                )
            )
        ),
        "eigenvalue_max_abs": float(
            np.max(np.abs(reference["energies"] - repeated["energies"]))
        ),
        "overlap_max_abs": float(
            np.max(np.abs(reference["overlap_z2"] - repeated["overlap_z2"]))
        ),
        "fsa_beta_max_abs": float(
            np.max(
                np.abs(
                    reference["fsa_beta_full_chain"]
                    - repeated["fsa_beta_full_chain"]
                )
            )
        ),
        "full_fsa_shell_count": length + 1,
        "folded_fsa_shell_count": length // 2 + 1,
        "official_data": "not-requested",
    }
    if official_data_dir is not None:
        official = Path(official_data_dir)
        archive = official / "eigendecomposition.zip"
        metrics["official_data"] = "available" if archive.is_file() else "missing"
        if not archive.is_file():
            raise RuntimeError(
                f"official small-L comparison requested but archive is missing: {archive}"
            )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tracks/ed/results/turner-2018/l32-server"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-columns", type=int, default=32)
    parser.add_argument("--validate-small-l", type=int)
    parser.add_argument("--official-data-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    invoked = list(argv) if argv is not None else sys.argv[1:]
    if args.length < 4 or args.length % 2:
        raise SystemExit("--length must be an even integer >= 4")
    if args.chunk_columns < 1:
        raise SystemExit("--chunk-columns must be positive")

    write_plan(args.length, args.output_dir, invoked)
    if args.validate_small_l is not None:
        metrics = validate_small_l(args.validate_small_l, args.official_data_dir)
        atomic_write_json(args.output_dir / "validation" / "metrics.json", metrics)
    if args.stage == "plan" or args.dry_run:
        print(args.output_dir / "manifest.json", flush=True)
        return 0

    requested = (
        ("basis", "hamiltonian", "diagonalize", "observables")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in requested:
        predecessor = STAGE_PREDECESSOR.get(stage)
        if predecessor is not None:
            require_stage(args.output_dir, predecessor)
        if stage == "basis":
            run_basis(args.length, args.output_dir)
        elif stage == "hamiltonian":
            run_hamiltonian(args.length, args.output_dir)
        elif stage == "diagonalize":
            run_diagonalize(args.length, args.output_dir)
        elif stage == "observables":
            run_observables(args.length, args.output_dir, args.chunk_columns)
        print(f"completed stage={stage}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"turner2018_l32_server: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
