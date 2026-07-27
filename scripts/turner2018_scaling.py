#!/usr/bin/env python3
"""Profile constrained-basis and sparse-matvec costs before large ED runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

from pxp_ed import constrained_basis, pxp_hamiltonian

DEFAULT_OUTPUT = Path("tracks/ed/results/turner-2018/scaling.json")


def profile_sizes(lengths: list[int]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for length in lengths:
        start = time.perf_counter()
        basis = constrained_basis(length, pbc=True)
        basis_seconds = time.perf_counter() - start

        start = time.perf_counter()
        hamiltonian = pxp_hamiltonian(basis, length, pbc=True)
        hamiltonian_seconds = time.perf_counter() - start

        vector = np.ones(len(basis), dtype=float) / np.sqrt(len(basis))
        start = time.perf_counter()
        hamiltonian @ vector
        matvec_seconds = time.perf_counter() - start
        start = time.perf_counter()
        expm_multiply(-0.1j * hamiltonian, vector)
        time_propagation_seconds = time.perf_counter() - start
        sparse_bytes = (
            hamiltonian.data.nbytes
            + hamiltonian.indices.nbytes
            + hamiltonian.indptr.nbytes
        )
        rows.append(
            {
                "length": length,
                "basis_dimension": len(basis),
                "hamiltonian_nnz": hamiltonian.nnz,
                "basis_seconds": basis_seconds,
                "hamiltonian_seconds": hamiltonian_seconds,
                "matvec_seconds": matvec_seconds,
                "time_propagation_seconds": time_propagation_seconds,
                "sparse_megabytes": sparse_bytes / 1024**2,
                "dense_matrix_gib": len(basis) ** 2 * 8 / 1024**3,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="12,14,16,18,20")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lengths = [int(value) for value in args.lengths.split(",")]
    rows = profile_sizes(lengths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"L={row['length']}: D={row['basis_dimension']}, "
            f"nnz={row['hamiltonian_nnz']}, "
            f"CSR={row['sparse_megabytes']:.2f} MiB, "
            f"dense={row['dense_matrix_gib']:.2f} GiB"
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
