"""Dense ED solver helpers for the independent Turner 2018 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from turner2018_ed_validation import (
    positive_integer,
    validate_eigenpair_residuals,
    validate_orthonormal_columns,
)


@dataclass(frozen=True)
class DenseResourceEstimate:
    """Byte-level resource estimate for dense diagonalization."""

    dimension: int
    matrix_bytes: int
    eigenvector_bytes: int
    minimum_requested_bytes: int


def estimate_dense_resources(dimension: int, *, vectors: bool) -> DenseResourceEstimate:
    """Estimate dense memory requirements for one full solve."""
    if dimension < 1:
        raise ValueError("dimension must be positive")

    matrix_bytes = int(dimension) * int(dimension) * 8
    eigenvector_bytes = matrix_bytes if vectors else 0
    # Workspace floor: one dense input matrix plus conservative LAPACK work arrays.
    minimum_requested_bytes = 4 * matrix_bytes + eigenvector_bytes
    return DenseResourceEstimate(
        dimension=int(dimension),
        matrix_bytes=matrix_bytes,
        eigenvector_bytes=eigenvector_bytes,
        minimum_requested_bytes=minimum_requested_bytes,
    )


def _validate_real_symmetric_csr(matrix: sp.csr_matrix) -> None:
    if not sp.isspmatrix_csr(matrix):
        raise TypeError("matrix must be a scipy.sparse.csr_matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not np.isrealobj(matrix.data):
        raise ValueError("matrix must be real symmetric CSR")

    asymmetry = matrix - matrix.T
    if asymmetry.nnz:
        max_abs = float(np.max(np.abs(asymmetry.data)))
        if max_abs > 1e-13:
            raise ValueError("matrix must be real symmetric CSR")


def solve_full_eigensystem(
    matrix: sp.csr_matrix,
    *,
    vectors: bool,
    declared_memory_bytes: int,
    validation_chunk_columns: int = 256,
    orthogonality_samples: int = 4096,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Solve the full dense eigensystem from a deterministic real symmetric CSR."""
    validation_chunk_columns = positive_integer(
        validation_chunk_columns, "validation_chunk_columns"
    )
    orthogonality_samples = positive_integer(
        orthogonality_samples, "orthogonality_samples"
    )
    _validate_real_symmetric_csr(matrix)
    estimate = estimate_dense_resources(matrix.shape[0], vectors=vectors)
    if declared_memory_bytes < estimate.minimum_requested_bytes:
        raise MemoryError(
            "declared memory is insufficient for dense diagonalization "
            f"({declared_memory_bytes} < {estimate.minimum_requested_bytes})"
        )

    dense = np.asfortranarray(matrix.toarray(), dtype=np.float64)
    if dense.dtype != np.float64 or not dense.flags.f_contiguous:
        raise RuntimeError("dense conversion must produce Fortran-contiguous float64")

    if vectors:
        energies, eigenvectors = scipy.linalg.eigh(
            dense,
            driver="evd",
            overwrite_a=True,
            check_finite=False,
        )
        validate_eigenpair_residuals(
            matrix,
            energies,
            eigenvectors,
            chunk_columns=validation_chunk_columns,
            tolerance=1e-11,
            error_type=RuntimeError,
        )
        try:
            validate_orthonormal_columns(
                eigenvectors,
                chunk_columns=validation_chunk_columns,
                orthogonality_samples=orthogonality_samples,
                tolerance=1e-11,
                name="eigenvectors",
            )
        except ValueError as error:
            raise RuntimeError(
                "eigenvector orthogonality exceeds tolerance"
            ) from error
        return np.asarray(energies, dtype=np.float64), np.asarray(eigenvectors, dtype=np.float64)

    energies = scipy.linalg.eigvalsh(
        dense,
        driver="evd",
        overwrite_a=True,
        check_finite=False,
    )
    return np.asarray(energies, dtype=np.float64), None
