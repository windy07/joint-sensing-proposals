from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ._validation import NoUsableSignal, signal_1d, positive


@dataclass(frozen=True)
class ClockPosterior:
    mean_s: np.ndarray
    cov_s2: np.ndarray
    source: str = "calibration_factor_or_filter"

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        data["mean_s"] = np.asarray(self.mean_s, dtype=float).tolist()
        data["cov_s2"] = np.asarray(self.cov_s2, dtype=float).tolist()
        return data


def update_clock_posterior(prior: ClockPosterior, z_s: np.ndarray, h: np.ndarray, r: np.ndarray) -> ClockPosterior:
    p = np.asarray(prior.cov_s2, dtype=float)
    h = np.asarray(h, dtype=float)
    r = np.asarray(r, dtype=float)
    s = h @ p @ h.T + r
    k = p @ h.T @ np.linalg.pinv(s)
    mean = np.asarray(prior.mean_s, dtype=float) + k @ (np.asarray(z_s, dtype=float) - h @ np.asarray(prior.mean_s, dtype=float))
    cov = (np.eye(p.shape[0]) - k @ h) @ p
    return ClockPosterior(mean, 0.5 * (cov + cov.T), "linear_gaussian_clock_calibration_filter")


def estimate_toa_matched_filter(
    iq: np.ndarray,
    waveform: np.ndarray,
    fs: float,
    search_s: tuple[float, float] | None = None,
    interpolate: bool = False,
) -> float:
    """Known/reference-template TOA point estimate; no calibrated variance.

    A supplied window is a closed physical interval in seconds. Only integer
    lags inside it are searched. Subsample interpolation additionally requires
    both neighbors inside the same interval. This is not a passive algorithm
    supplied with an unknown emitter's exact waveform.
    """
    iq = signal_1d(iq, "iq")
    waveform = signal_1d(waveform, "known_or_measured_reference")
    fs = positive(fs, "fs_hz")
    corr = np.abs(np.correlate(iq, waveform, mode="full"))
    lags = np.arange(-len(waveform) + 1, len(iq))
    valid = np.ones(lags.size, dtype=bool)
    if search_s is not None:
        bounds = np.asarray(search_s, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
            raise ValueError("search_s must be an ordered finite pair in seconds")
        valid = (lags / fs >= bounds[0]) & (lags / fs <= bounds[1])
    indices = np.flatnonzero(valid)
    if not indices.size:
        raise ValueError("search window contains no sampled correlation lag")
    # Retain the actual selected global index, including when equal peaks exist.
    peak_idx = int(indices[int(np.argmax(corr[indices]))])
    if corr[peak_idx] <= 0:
        raise NoUsableSignal("zero correlation within the requested window")
    lag = float(lags[peak_idx])
    if interpolate and 0 < peak_idx < corr.size - 1 and valid[peak_idx-1] and valid[peak_idx+1]:
        y0, y1, y2 = corr[peak_idx-1:peak_idx+2]
        denom = float(y0 - 2*y1 + y2)
        if denom < -np.finfo(float).eps * max(float(y0),float(y1),float(y2)):
            lag += float(np.clip(.5*(y0-y2)/denom,-.5,.5))
    return float(lag/fs)


def estimate_tdoa_cross_correlation(
    received_i: np.ndarray, received_reference: np.ndarray, fs: float,
    search_s: tuple[float,float] | None=None, interpolate: bool=True,
) -> float:
    """Two actually received streams; positive output means i arrives later.

    The two ADCs must have a common nominal sampling rate and acquisition epoch.
    Clock/group-delay differences remain in the output. This bounded-delay
    reference does not compensate FDOA and provides no posterior covariance.
    """
    return estimate_toa_matched_filter(received_i, received_reference, fs, search_s, interpolate)


def tdoa_trials_from_raw_iq(
    iq_i_trials: list[np.ndarray],
    iq_j_trials: list[np.ndarray],
    waveform: np.ndarray,
    fs: float,
    clock_post: ClockPosterior,
    i: int,
    j: int,
    pred_window_s: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(iq_i_trials) != len(iq_j_trials):
        raise ValueError("paired trial lists must have equal length")
    raw: list[float] = []
    corrected: list[float] = []
    clock_mean = np.asarray(clock_post.mean_s, dtype=float)
    for iq_i, iq_j in zip(iq_i_trials, iq_j_trials):
        ti = estimate_toa_matched_filter(iq_i, waveform, fs, pred_window_s)
        tj = estimate_toa_matched_filter(iq_j, waveform, fs, pred_window_s)
        z = ti - tj
        raw.append(z)
        corrected.append(z - (clock_mean[i] - clock_mean[j]))
    return np.asarray(raw, dtype=float), np.asarray(corrected, dtype=float)

