"""Bounded-memory validation helpers for full ED eigenvector matrices."""

from __future__ import annotations

from collections.abc import Iterator
import math
import operator

import numpy as np
import scipy.sparse as sp


ORTHOGONALITY_SAMPLING_POLICY = (
    "midpoint-stratified first indices with SplitMix64 partner slots when "
    "samples <= D-1; otherwise midpoint-stratified canonical pair ranks"
)
ORTHOGONALITY_PAIR_MEMORY_POLICY = (
    "bounded by two int64 arrays of pair_batch_size"
)


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


def _canonical_pair_from_rank(column_count: int, rank: int) -> tuple[int, int]:
    """Unrank one lexicographically ordered pair without pair-sized storage."""
    diagonal = 2 * column_count - 1
    discriminant = diagonal * diagonal - 8 * rank
    root = math.isqrt(discriminant)
    ceiling_root = root if root * root == discriminant else root + 1
    first = (diagonal - ceiling_root) // 2
    row_start = first * (2 * column_count - first - 1) // 2
    second = first + 1 + rank - row_start
    return first, second


def _splitmix64(value: int) -> int:
    """Return one deterministic well-mixed uint64 value."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def sample_pair_batches(
    column_count: int,
    *,
    sample_count: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream deterministic spectrum-wide canonical pairs in bounded batches."""
    sample_count = positive_integer(sample_count, "sample_count")
    batch_size = positive_integer(batch_size, "batch_size")
    total_pairs = column_count * (column_count - 1) // 2
    target = min(sample_count, total_pairs)
    for start in range(0, target, batch_size):
        stop = min(start + batch_size, target)
        left = np.empty(stop - start, dtype=np.int64)
        right = np.empty(stop - start, dtype=np.int64)
        for offset, ordinal in enumerate(range(start, stop)):
            if target <= column_count - 1:
                first = ((2 * ordinal + 1) * (column_count - 1)) // (2 * target)
                partner_width = column_count - first - 1
                left[offset] = first
                right[offset] = first + 1 + _splitmix64(ordinal) % partner_width
            else:
                rank = ((2 * ordinal + 1) * total_pairs) // (2 * target)
                left[offset], right[offset] = _canonical_pair_from_rank(
                    column_count,
                    rank,
                )
        yield left, right


def validate_orthonormal_columns(
    vectors: np.ndarray,
    *,
    chunk_columns: int,
    orthogonality_samples: int,
    tolerance: float,
    name: str,
) -> dict[str, int | float | str]:
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

    total_pairs = column_count * (column_count - 1) // 2
    target = min(orthogonality_samples, total_pairs)
    max_sample_overlap = 0.0
    samples_checked = 0
    for left, right in sample_pair_batches(
        column_count,
        sample_count=orthogonality_samples,
        batch_size=chunk_columns,
    ):
        overlaps = np.sum(
            vectors[:, left].conj() * vectors[:, right],
            axis=0,
        )
        samples_checked += left.size
        if overlaps.size:
            max_sample_overlap = max(
                max_sample_overlap, float(np.max(np.abs(overlaps)))
            )
    if max_sample_overlap > tolerance:
        raise ValueError(f"{name} vectors must be orthonormal")
    return {
        "chunk_columns": chunk_columns,
        "columns_checked": column_count,
        "orthogonality_sample_count": samples_checked,
        "orthogonality_sampling_policy": ORTHOGONALITY_SAMPLING_POLICY,
        "orthogonality_pair_batch_size": min(chunk_columns, target),
        "orthogonality_pair_metadata_memory": ORTHOGONALITY_PAIR_MEMORY_POLICY,
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
