"""Bounded-memory validation helpers for full ED eigenvector matrices."""

from __future__ import annotations

import operator

import numpy as np
import scipy.sparse as sp


def positive_integer(value: int, name: str) -> int:
    """Return a strict positive integer, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def uint64_state(value: int, name: str) -> int:
    """Validate an integer state before any uint64 conversion."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer state in uint64 range")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer state in uint64 range") from error
    if result < 0 or result > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must be an integer state in uint64 range")
    return result


def _sample_pairs(column_count: int, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    total_pairs = column_count * (column_count - 1) // 2
    target = min(sample_count, total_pairs)
    if target == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if target == total_pairs:
        left = np.empty(target, dtype=np.int64)
        right = np.empty(target, dtype=np.int64)
        cursor = 0
        for first in range(column_count):
            width = column_count - first - 1
            left[cursor : cursor + width] = first
            right[cursor : cursor + width] = np.arange(
                first + 1, column_count, dtype=np.int64
            )
            cursor += width
        return left, right

    pairs: set[tuple[int, int]] = set()
    for first in range(min(column_count - 1, target)):
        pairs.add((first, first + 1))
    state = 0x9E3779B97F4A7C15
    while len(pairs) < target:
        state = (state * 6364136223846793005 + 1442695040888963407) & (
            (1 << 64) - 1
        )
        first = state % column_count
        state = (state * 6364136223846793005 + 1442695040888963407) & (
            (1 << 64) - 1
        )
        second = state % (column_count - 1)
        if second >= first:
            second += 1
        pairs.add((min(first, second), max(first, second)))
    ordered = sorted(pairs)
    return (
        np.fromiter((pair[0] for pair in ordered), dtype=np.int64),
        np.fromiter((pair[1] for pair in ordered), dtype=np.int64),
    )


def validate_orthonormal_columns(
    vectors: np.ndarray,
    *,
    chunk_columns: int,
    orthogonality_samples: int,
    tolerance: float,
    name: str,
) -> dict[str, int | float]:
    """Validate finiteness, every norm, and bounded deterministic cross samples."""
    chunk_columns = positive_integer(chunk_columns, "chunk_columns")
    orthogonality_samples = positive_integer(
        orthogonality_samples, "orthogonality_samples"
    )
    column_count = vectors.shape[1]
    max_norm_error = 0.0
    for start in range(0, column_count, chunk_columns):
        stop = min(start + chunk_columns, column_count)
        block = vectors[:, start:stop]
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} vectors must be finite")
        norms = np.sum(np.abs(block) ** 2, axis=0)
        max_norm_error = max(max_norm_error, float(np.max(np.abs(norms - 1.0))))
    if max_norm_error > tolerance:
        raise ValueError(f"{name} vectors must be orthonormal")

    left, right = _sample_pairs(column_count, orthogonality_samples)
    max_sample_overlap = 0.0
    for start in range(0, left.size, chunk_columns):
        stop = min(start + chunk_columns, left.size)
        overlaps = np.sum(
            vectors[:, left[start:stop]].conj() * vectors[:, right[start:stop]],
            axis=0,
        )
        if overlaps.size:
            max_sample_overlap = max(
                max_sample_overlap, float(np.max(np.abs(overlaps)))
            )
    if max_sample_overlap > tolerance:
        raise ValueError(f"{name} vectors must be orthonormal")
    return {
        "chunk_columns": chunk_columns,
        "columns_checked": column_count,
        "orthogonality_sample_count": int(left.size),
        "max_norm_error": max_norm_error,
        "max_sample_overlap": max_sample_overlap,
    }


def validate_hermitian(
    matrix: sp.spmatrix | np.ndarray,
    *,
    chunk_columns: int,
    tolerance: float,
) -> float:
    """Check Hermiticity using only bounded row/column strips."""
    dimension = matrix.shape[0]
    max_error = 0.0
    for start in range(0, dimension, chunk_columns):
        stop = min(start + chunk_columns, dimension)
        if sp.issparse(matrix):
            rows = matrix[start:stop, :]
            conjugate_columns = matrix[:, start:stop].getH()
            difference = rows - conjugate_columns
            if difference.nnz:
                max_error = max(max_error, float(np.max(np.abs(difference.data))))
        else:
            rows = np.asarray(matrix[start:stop, :])
            conjugate_columns = np.asarray(matrix[:, start:stop]).conj().T
            max_error = max(
                max_error, float(np.max(np.abs(rows - conjugate_columns)))
            )
    if max_error > tolerance:
        raise ValueError(f"Hamiltonian must be Hermitian; error={max_error}")
    return max_error


def validate_eigenpair_residuals(
    matrix: sp.spmatrix | np.ndarray,
    energies: np.ndarray,
    vectors: np.ndarray,
    *,
    chunk_columns: int,
    tolerance: float,
    error_type: type[Exception] = ValueError,
) -> float:
    """Check every eigenpair in bounded column chunks."""
    max_residual = 0.0
    for start in range(0, vectors.shape[1], chunk_columns):
        stop = min(start + chunk_columns, vectors.shape[1])
        block = vectors[:, start:stop]
        residual = matrix @ block - block * energies[np.newaxis, start:stop]
        max_residual = max(max_residual, float(np.max(np.abs(residual))))
    if max_residual > tolerance:
        raise error_type(f"eigenvector residual exceeds tolerance: {max_residual}")
    return max_residual
