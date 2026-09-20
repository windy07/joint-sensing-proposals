"""Deterministic Jacobi algebra for small symmetric audit/runtime matrices."""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from typing import Any

import numpy as np


def _safe_windows_backend_required() -> bool:
    return (
        os.environ.get("RADAR_FORCE_SAFE_LINALG", "0") == "1"
        or (
            os.name == "nt"
            and "torch" in sys.modules
            and os.environ.get("RADAR_ALLOW_MIXED_NATIVE_LINALG", "0") != "1"
        )
    )


def platform_matmul(left: Any, right: Any) -> Any:
    """Matrix product with a deterministic Windows fallback for small arrays."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if not _safe_windows_backend_required():
        return left_array @ right_array
    if left_array.ndim not in (1, 2) or right_array.ndim not in (1, 2):
        raise ValueError("platform_matmul supports one- or two-dimensional arrays")
    left_inner = left_array.shape[-1]
    right_inner = right_array.shape[0]
    if left_inner != right_inner:
        raise ValueError("matrix product dimension mismatch")

    def dot(row: np.ndarray, column: np.ndarray) -> float:
        return math.fsum(
            float(row[index]) * float(column[index])
            for index in range(left_inner)
        )

    if left_array.ndim == 1 and right_array.ndim == 1:
        return np.float64(dot(left_array, right_array))
    if left_array.ndim == 2 and right_array.ndim == 1:
        return np.asarray(
            [dot(left_array[row], right_array) for row in range(left_array.shape[0])],
            dtype=np.float64,
        )
    if left_array.ndim == 1 and right_array.ndim == 2:
        return np.asarray(
            [dot(left_array, right_array[:, column]) for column in range(right_array.shape[1])],
            dtype=np.float64,
        )
    result = np.empty(
        (left_array.shape[0], right_array.shape[1]), dtype=np.float64
    )
    for row in range(left_array.shape[0]):
        for column in range(right_array.shape[1]):
            result[row, column] = dot(
                left_array[row], right_array[:, column]
            )
    return result


def _symmetry_tolerance(matrix: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(matrix))))
    return max(
        1.0e-12,
        8.0 * np.finfo(np.float64).eps * matrix.shape[0] * scale,
    )


def symmetric_eigh(
    value: Any, *, tolerance: float = 1.0e-13, max_sweeps: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite and nonempty")
    if float(np.max(np.abs(matrix - matrix.T))) > _symmetry_tolerance(matrix):
        raise ValueError("matrix must be symmetric")
    work = 0.5 * (matrix + matrix.T)
    vectors = np.eye(work.shape[0], dtype=np.float64)
    dimension = work.shape[0]
    scale = max(1.0, float(np.max(np.abs(work))))
    for _ in range(max_sweeps * max(1, dimension * dimension)):
        off_diagonal = np.triu(np.abs(work), 1)
        flat_index = int(np.argmax(off_diagonal))
        row, column = divmod(flat_index, dimension)
        if float(off_diagonal[row, column]) <= tolerance * scale:
            break
        app = float(work[row, row])
        aqq = float(work[column, column])
        apq = float(work[row, column])
        tau = (aqq - app) / (2.0 * apq)
        tangent = (
            math.copysign(1.0, tau)
            / (abs(tau) + math.sqrt(1.0 + tau * tau))
            if tau != 0.0
            else 1.0
        )
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        for index in range(dimension):
            if index in (row, column):
                continue
            aip = float(work[index, row])
            aiq = float(work[index, column])
            work[index, row] = work[row, index] = cosine * aip - sine * aiq
            work[index, column] = work[column, index] = sine * aip + cosine * aiq
        work[row, row] = app - tangent * apq
        work[column, column] = aqq + tangent * apq
        work[row, column] = work[column, row] = 0.0
        old_row = vectors[:, row].copy()
        old_column = vectors[:, column].copy()
        vectors[:, row] = cosine * old_row - sine * old_column
        vectors[:, column] = sine * old_row + cosine * old_column
    else:
        raise RuntimeError("Jacobi eigensolver did not converge")
    values = np.diag(work).copy()
    order = np.argsort(values)
    return values[order], vectors[:, order]


def symmetric_pinv(value: Any, *, rcond: float = 1.0e-10) -> np.ndarray:
    values, vectors = symmetric_eigh(value)
    threshold = rcond * max(1.0, float(np.max(np.abs(values))))
    reciprocal = np.asarray(
        [0.0 if abs(item) <= threshold else 1.0 / item for item in values],
        dtype=np.float64,
    )
    return platform_matmul(vectors * reciprocal[None, :], vectors.T)


def symmetric_inverse(value: Any, *, minimum_eigenvalue: float = 1.0e-12) -> np.ndarray:
    values, vectors = symmetric_eigh(value)
    if float(values.min()) <= minimum_eigenvalue:
        raise ValueError("matrix is not numerically positive definite")
    return platform_matmul(vectors * (1.0 / values)[None, :], vectors.T)


def platform_symmetric_eigh(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Use audited Jacobi algebra on Windows and optimized LAPACK elsewhere."""
    matrix = np.asarray(value, dtype=np.float64)
    if _safe_windows_backend_required():
        return symmetric_eigh(matrix)
    return np.linalg.eigh(0.5 * (matrix + matrix.T))


def platform_symmetric_eigvalsh(value: Any) -> np.ndarray:
    if _safe_windows_backend_required():
        return symmetric_eigh(value)[0]
    matrix = np.asarray(value, dtype=np.float64)
    return np.linalg.eigvalsh(0.5 * (matrix + matrix.T))


def platform_cholesky(value: Any) -> np.ndarray:
    """Return a lower Cholesky factor without Windows LAPACK dependency."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite and nonempty")
    if float(np.max(np.abs(matrix - matrix.T))) > _symmetry_tolerance(matrix):
        raise ValueError("matrix must be symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    if not _safe_windows_backend_required():
        return np.linalg.cholesky(symmetric)
    dimension = symmetric.shape[0]
    lower = np.zeros_like(symmetric)
    for row in range(dimension):
        for column in range(row + 1):
            accumulated = math.fsum(
                float(lower[row, index]) * float(lower[column, index])
                for index in range(column)
            )
            if row == column:
                residual = float(symmetric[row, row]) - accumulated
                if not math.isfinite(residual) or residual <= 0.0:
                    raise np.linalg.LinAlgError("matrix is not positive definite")
                lower[row, column] = math.sqrt(residual)
            else:
                pivot = float(lower[column, column])
                if pivot <= 0.0:
                    raise np.linalg.LinAlgError("matrix is not positive definite")
                lower[row, column] = (
                    float(symmetric[row, column]) - accumulated
                ) / pivot
    return lower


def platform_symmetric_pinv(value: Any, *, rcond: float = 1.0e-10) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    symmetric = 0.5 * (matrix + matrix.T)
    if _safe_windows_backend_required():
        return symmetric_pinv(symmetric, rcond=rcond)
    return np.linalg.pinv(symmetric, rcond=rcond)


def platform_norm(value: Any) -> float:
    """Euclidean/Frobenius norm without dispatching to Windows LAPACK."""

    array = np.asarray(value)
    if not np.isfinite(array).all():
        return float("nan")
    if not _safe_windows_backend_required():
        return float(np.linalg.norm(array))
    squared = sum(float(abs(item)) ** 2 for item in array.reshape(-1))
    return math.sqrt(squared)


@lru_cache(maxsize=32)
def platform_gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights via the symmetric Jacobi matrix."""

    count = int(order)
    if count <= 0:
        raise ValueError("quadrature order must be positive")
    if not _safe_windows_backend_required():
        nodes, weights = np.polynomial.legendre.leggauss(count)
    else:
        jacobi = np.zeros((count, count), dtype=np.float64)
        for index in range(1, count):
            beta = index / math.sqrt(4.0 * index * index - 1.0)
            jacobi[index - 1, index] = beta
            jacobi[index, index - 1] = beta
        nodes, vectors = symmetric_eigh(jacobi)
        weights = 2.0 * vectors[0, :] ** 2
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def platform_singular_values(value: Any) -> np.ndarray:
    """Descending singular values through a symmetric Gram eigenproblem."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("singular-value input must be a finite matrix")
    if not _safe_windows_backend_required():
        return np.linalg.svd(matrix, compute_uv=False)
    gram = platform_matmul(matrix.T, matrix)
    gram = 0.5 * (gram + gram.T)
    eigenvalues = symmetric_eigh(gram)[0]
    values = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    return values[: min(matrix.shape)]


def platform_svd(
    value: Any, *, full_matrices: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD with a Gram-spectrum fallback for mixed Windows runtimes."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("SVD input must be a finite matrix")
    if not _safe_windows_backend_required():
        return np.linalg.svd(matrix, full_matrices=full_matrices)
    row_count, column_count = matrix.shape
    rank_dimension = min(row_count, column_count)
    left_gram = platform_matmul(matrix, matrix.T)
    right_gram = platform_matmul(matrix.T, matrix)
    left_values, left_vectors = symmetric_eigh(
        0.5 * (left_gram + left_gram.T)
    )
    right_values, right_vectors = symmetric_eigh(
        0.5 * (right_gram + right_gram.T)
    )
    left_order = np.argsort(left_values)[::-1]
    right_order = np.argsort(right_values)[::-1]
    singular_values = np.sqrt(
        np.maximum(right_values[right_order[:rank_dimension]], 0.0)
    )
    left = left_vectors[:, left_order]
    right_transpose = right_vectors[:, right_order].T
    if not full_matrices:
        left = left[:, :rank_dimension]
        right_transpose = right_transpose[:rank_dimension, :]
    return left, singular_values, right_transpose


def platform_solve(matrix: Any, right: Any) -> np.ndarray:
    """Solve a small dense system without Windows LAPACK process aborts."""
    coefficient = np.asarray(matrix, dtype=np.float64)
    value = np.asarray(right, dtype=np.float64)
    if not _safe_windows_backend_required():
        return np.linalg.solve(coefficient, value)
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient matrix must be square")
    dimension = coefficient.shape[0]
    if value.shape[0] != dimension:
        raise ValueError("right-hand side dimension mismatch")
    was_vector = value.ndim == 1
    rhs = value.reshape(dimension, -1).copy()

    # Cholesky-based radar solves call this routine with exact triangular
    # factors.  Preserving that structure avoids unnecessary pivoting and uses
    # compensated dot products in the forward/back substitutions.
    strictly_upper = coefficient[np.triu_indices(dimension, 1)]
    strictly_lower = coefficient[np.tril_indices(dimension, -1)]
    lower_triangular = not np.any(strictly_upper)
    upper_triangular = not np.any(strictly_lower)
    if lower_triangular or upper_triangular:
        solution = np.zeros_like(rhs)
        rows = range(dimension) if lower_triangular else range(dimension - 1, -1, -1)
        coefficient_scale = max(1.0, float(np.max(np.abs(coefficient))))
        for row in rows:
            pivot = float(coefficient[row, row])
            if abs(pivot) <= np.finfo(float).eps * coefficient_scale:
                raise np.linalg.LinAlgError("singular triangular matrix")
            indices = range(row) if lower_triangular else range(row + 1, dimension)
            for column in range(rhs.shape[1]):
                accumulated = math.fsum(
                    float(coefficient[row, index])
                    * float(solution[index, column])
                    for index in indices
                )
                solution[row, column] = (
                    float(rhs[row, column]) - accumulated
                ) / pivot
        return solution[:, 0] if was_vector else solution.reshape(value.shape)

    work = coefficient.copy()
    scale = max(1.0, float(np.max(np.abs(work))))
    for pivot_index in range(dimension):
        selected = pivot_index + int(np.argmax(np.abs(work[pivot_index:, pivot_index])))
        if abs(float(work[selected, pivot_index])) <= np.finfo(float).eps * scale:
            raise np.linalg.LinAlgError("singular matrix")
        if selected != pivot_index:
            work[[pivot_index, selected]] = work[[selected, pivot_index]]
            rhs[[pivot_index, selected]] = rhs[[selected, pivot_index]]
        pivot = float(work[pivot_index, pivot_index])
        work[pivot_index] /= pivot
        rhs[pivot_index] /= pivot
        for row in range(pivot_index + 1, dimension):
            factor = float(work[row, pivot_index])
            if factor == 0.0:
                continue
            work[row] -= factor * work[pivot_index]
            rhs[row] -= factor * rhs[pivot_index]
    solution = np.zeros_like(rhs)
    for row in range(dimension - 1, -1, -1):
        solution[row] = rhs[row] - platform_matmul(
            work[row, row + 1 :], solution[row + 1 :]
        )
    return solution[:, 0] if was_vector else solution.reshape(value.shape)
