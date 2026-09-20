from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .credible_grid import credible_region, cell_areas
from ._validation import signal_1d, positive, grid_1d, NoUsableSignal


def _solve_complex_system_regularized(
    gram: list[list[complex]],
    rhs: list[complex],
    *,
    ridge: float = 1.0e-10,
) -> np.ndarray:
    """Solve a small complex normal-equation system without LAPACK.

    The PBR direct-path dictionary has only a handful of columns.  Some Windows
    BLAS/LAPACK builds abort inside ``lstsq`` on these tiny systems, so the ECA
    path uses an explicit pivoted Gauss-Jordan solve instead of delegating to
    ``np.linalg``.  The ridge is tiny relative to the diagonal scale and only
    regularizes nearly duplicate lag atoms.
    """
    n = len(rhs)
    if n == 0:
        return np.zeros(0, dtype=complex)
    diag_scale = max(float(abs(gram[i][i])) for i in range(n))
    diag_scale = max(diag_scale, 1.0)
    regularizer = float(ridge) * diag_scale
    aug = []
    for i in range(n):
        row = [complex(gram[i][j]) for j in range(n)]
        row[i] += regularizer
        row.append(complex(rhs[i]))
        aug.append(row)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row_idx: abs(aug[row_idx][col]))
        if abs(aug[pivot][col]) <= 1.0e-24:
            aug[col][col] += regularizer if regularizer > 0.0 else 1.0e-12
            pivot = col
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        if abs(pivot_value) <= 1.0e-30:
            continue
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row_idx in range(n):
            if row_idx == col:
                continue
            factor = aug[row_idx][col]
            if abs(factor) <= 1.0e-30:
                continue
            for j in range(col, n + 1):
                aug[row_idx][j] -= factor * aug[col][j]
    return np.asarray([aug[i][n] for i in range(n)], dtype=complex)


def _complex_lstsq_coef(A: np.ndarray, y: np.ndarray, *, ridge: float = 1.0e-10) -> np.ndarray:
    design = np.asarray(A, dtype=complex)
    target = np.asarray(y, dtype=complex).reshape(-1)
    if design.ndim != 2:
        raise ValueError("direct-path dictionary must be two-dimensional")
    if design.shape[0] != target.shape[0]:
        raise ValueError("direct-path dictionary and surveillance length mismatch")
    n_rows, n_cols = design.shape
    gram = [[0.0j for _ in range(n_cols)] for _ in range(n_cols)]
    rhs = [0.0j for _ in range(n_cols)]
    for row in range(n_rows):
        target_value = complex(target[row])
        for i in range(n_cols):
            ai_conj = complex(design[row, i]).conjugate()
            rhs[i] += ai_conj * target_value
            for j in range(n_cols):
                gram[i][j] += ai_conj * complex(design[row, j])
    return _solve_complex_system_regularized(gram, rhs, ridge=ridge)


def _dictionary_product(A: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.asarray(A, dtype=complex)
    weights = np.asarray(coef, dtype=complex).reshape(-1)
    out = np.zeros(design.shape[0], dtype=complex)
    for row in range(design.shape[0]):
        total = 0.0j
        for col in range(design.shape[1]):
            total += complex(design[row, col]) * complex(weights[col])
        out[row] = total
    return out


def _complex_column_rank(A: np.ndarray, *, tol: float = 1.0e-9) -> int:
    design = np.asarray(A, dtype=complex)
    if design.ndim != 2 or design.shape[1] == 0:
        return 0
    n_rows, n_cols = design.shape
    work = [[complex(design[row, col]) for col in range(n_cols)] for row in range(n_rows)]
    max_abs = max((abs(value) for row in work for value in row), default=0.0)
    threshold = float(tol) * max(max_abs, 1.0)
    rank = 0
    for col in range(n_cols):
        pivot = max(range(rank, n_rows), key=lambda row_idx: abs(work[row_idx][col]), default=rank)
        if pivot >= n_rows or abs(work[pivot][col]) <= threshold:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][col]
        for j in range(col, n_cols):
            work[rank][j] /= pivot_value
        for row_idx in range(n_rows):
            if row_idx == rank:
                continue
            factor = work[row_idx][col]
            if abs(factor) <= threshold:
                continue
            for j in range(col, n_cols):
                work[row_idx][j] -= factor * work[rank][j]
        rank += 1
        if rank == n_rows:
            break
    return int(rank)


@dataclass
class PBR2DFactor:
    mean_delay_s: float
    mean_doppler_hz: float
    covariance: np.ndarray
    coverage_mass: float
    credible_cell_count: int
    truth_inside_if_known_for_audit: bool | None
    nll_if_known_for_audit: float | None
    mode: str
    R_inflation: float
    posterior_hash: str
    posterior_shape: tuple[int, int]
    support_audit: dict | None = None
    calibrated_likelihood: bool = False
    semantic_role: str = "heuristic_CAF_posterior_includes_tracking_prior_not_new_likelihood"

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        data["covariance"] = np.asarray(self.covariance, dtype=float).tolist()
        data["posterior_shape"] = list(self.posterior_shape)
        return data


def caf_likelihood(
    reference: np.ndarray,
    surveillance: np.ndarray,
    fs: float,
    delay_grid_samples: np.ndarray,
    doppler_grid_hz: np.ndarray,
) -> np.ndarray:
    n = len(surveillance)
    t = np.arange(n, dtype=float) / float(fs)
    out = np.zeros((len(delay_grid_samples), len(doppler_grid_hz)), dtype=float)
    for i, delay in enumerate(delay_grid_samples):
        shifted = finite_window_shift(reference, float(delay), n)
        for j, fd in enumerate(doppler_grid_hz):
            replica = shifted * np.exp(2j * np.pi * float(fd) * t)
            out[i, j] = float(abs(np.vdot(replica, surveillance)) ** 2)
    return out


def caf_likelihood_circular_negative_control(
    reference: np.ndarray,
    surveillance: np.ndarray,
    fs: float,
    delay_grid_samples: np.ndarray,
    doppler_grid_hz: np.ndarray,
) -> np.ndarray:
    n = len(reference)
    t = np.arange(n, dtype=float) / float(fs)
    out = np.zeros((len(delay_grid_samples), len(doppler_grid_hz)), dtype=float)
    ref_fft = np.fft.fft(reference)
    freqs = np.fft.fftfreq(n)
    for i, delay in enumerate(delay_grid_samples):
        shifted = np.fft.ifft(ref_fft * np.exp(-2j * np.pi * freqs * float(delay)))
        for j, fd in enumerate(doppler_grid_hz):
            replica = shifted * np.exp(2j * np.pi * float(fd) * t)
            out[i, j] = float(abs(np.vdot(replica, surveillance)) ** 2)
    return out


def _frac_delay_kernel(frac: float, taps: int = 41) -> np.ndarray:
    n = np.arange(taps, dtype=float) - (int(taps) - 1) / 2.0
    h = np.sinc(n - float(frac)) * np.hamming(int(taps))
    return h / max(float(np.sum(h)), 1.0e-12)


def finite_window_shift(reference: np.ndarray, delay_samples: float, out_len: int) -> np.ndarray:
    x = np.asarray(reference, dtype=complex)
    k = int(np.floor(float(delay_samples)))
    frac = float(delay_samples) - k
    h = _frac_delay_kernel(frac)
    y_full = np.convolve(x, h.astype(complex), mode="full")
    y = np.zeros(int(out_len), dtype=complex)
    start = k - (len(h) - 1) // 2
    for idx, val in enumerate(y_full):
        out_idx = start + idx
        if 0 <= out_idx < int(out_len):
            y[out_idx] += val
    return y


def direct_path_dictionary_windowed(reference: np.ndarray, out_len: int, lags: np.ndarray | list[float]) -> np.ndarray:
    cols = [finite_window_shift(reference, float(lag), int(out_len)) for lag in lags]
    return np.column_stack(cols)


def suppress_direct_path(reference: np.ndarray, surveillance: np.ndarray, order: int = 3) -> np.ndarray:
    lag_values: list[float] = [0.0]
    for lag in range(1, int(order)):
        lag_values.extend([float(lag), -float(lag)])
    A = direct_path_dictionary_windowed(reference, len(surveillance), lag_values)
    coef = _complex_lstsq_coef(A, surveillance)
    return np.asarray(surveillance, dtype=complex) - _dictionary_product(A, coef)


def suppress_direct_path_circular_negative_control(
    reference: np.ndarray, surveillance: np.ndarray, order: int = 3
) -> np.ndarray:
    dictionary = [np.asarray(reference, dtype=complex)]
    for lag in range(1, int(order)):
        dictionary.append(np.roll(reference, lag))
        dictionary.append(np.roll(reference, -lag))
    A = np.column_stack(dictionary)
    coef = _complex_lstsq_coef(A, surveillance)
    return np.asarray(surveillance, dtype=complex) - _dictionary_product(A, coef)


def posterior_from_likelihood_and_prior(log_like: np.ndarray, log_prior: np.ndarray, temperature: float) -> np.ndarray:
    temperature = positive(temperature, "temperature")
    z = np.asarray(log_like, dtype=float) / temperature + np.asarray(log_prior, dtype=float)
    z -= float(np.max(z))
    p = np.exp(z)
    return p / max(float(np.sum(p)), 1.0e-300)


def build_pbr_factor(
    reference: np.ndarray,
    surveillance: np.ndarray,
    fs: float,
    delay_grid_s: np.ndarray,
    doppler_grid_hz: np.ndarray,
    tracker_delay_mean_s: float,
    tracker_delay_std_s: float,
    tracker_doppler_mean_hz: float,
    tracker_doppler_std_hz: float,
    *,
    truth: tuple[float, float] | None = None,
    temperature: float = 0.35,
    credible_mass: float = 0.90,
) -> tuple[PBR2DFactor, np.ndarray]:
    # This retained CAF score is NOT re-labelled as a noise-calibrated likelihood.
    reference = signal_1d(reference,"reference")
    surveillance = signal_1d(surveillance,"surveillance")
    fs = positive(fs,"fs_hz")
    if not np.any(np.abs(reference)>0) or not np.any(np.abs(surveillance)>0):
        raise NoUsableSignal("zero received channel; no credible CAF audit")
    delay_grid_s = grid_1d(delay_grid_s,"delay_grid_s")
    doppler_grid_hz = grid_1d(doppler_grid_hz,"doppler_grid_hz")
    positive(tracker_delay_std_s,"tracker_delay_std_s")
    positive(tracker_doppler_std_hz,"tracker_doppler_std_hz")
    if not np.isfinite(tracker_delay_mean_s) or not np.isfinite(tracker_doppler_mean_hz):
        raise ValueError("tracking prior means must be finite")
    # The prior density is integrated over explicitly truncated midpoint cells.
    areas = cell_areas(delay_grid_s,doppler_grid_hz)
    clean = suppress_direct_path(reference, surveillance, order=3)
    power = caf_likelihood(reference, clean, fs, delay_grid_s * fs, doppler_grid_hz)
    log_like = np.log(power + 1.0e-300)
    D, V = np.meshgrid(delay_grid_s, doppler_grid_hz, indexing="ij")
    log_prior = -0.5 * ((D - float(tracker_delay_mean_s)) / max(float(tracker_delay_std_s), 1.0e-12)) ** 2
    log_prior += -0.5 * ((V - float(tracker_doppler_mean_hz)) / max(float(tracker_doppler_std_hz), 1.0e-9)) ** 2
    posterior = posterior_from_likelihood_and_prior(log_like, log_prior + np.log(areas), temperature)
    mean_delay = float(np.sum(posterior * D))
    mean_doppler = float(np.sum(posterior * V))
    dd = D - mean_delay
    vv = V - mean_doppler
    covariance = np.array(
        [
            [float(np.sum(posterior * dd * dd)), float(np.sum(posterior * dd * vv))],
            [float(np.sum(posterior * dd * vv)), float(np.sum(posterior * vv * vv))],
        ],
        dtype=float,
    )
    audit = credible_region(posterior,delay_grid_s,doppler_grid_hz,credible_mass,truth)
    cells = audit["cell_count"]
    truth_inside = audit["truth_inside_credible_region"]
    nll = audit["nll_cell_mass"]
    mode = "soft_mixture"
    inflation = 1.0
    if cells > 0.4 * posterior.size:
        mode = "inflated_R"
        inflation = max(2.0, cells / max(1.0, 0.1 * posterior.size))
    post_hash = hashlib.sha256(np.ascontiguousarray(posterior, dtype=np.float64).tobytes()).hexdigest()
    return (
        PBR2DFactor(
            mean_delay_s=mean_delay,
            mean_doppler_hz=mean_doppler,
            covariance=covariance,
            coverage_mass=float(credible_mass),
            credible_cell_count=cells,
            truth_inside_if_known_for_audit=truth_inside,
            nll_if_known_for_audit=nll,
            mode=mode,
            R_inflation=float(inflation),
            posterior_hash=post_hash,
            posterior_shape=posterior.shape,
            support_audit=audit,
        ),
        posterior,
    )
