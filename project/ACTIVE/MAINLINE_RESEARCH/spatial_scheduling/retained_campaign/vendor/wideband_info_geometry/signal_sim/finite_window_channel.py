from __future__ import annotations

import numpy as np


def fractional_delay_kernel(frac: float, taps: int = 41, window: str = "hann") -> np.ndarray:
    if taps % 2 == 0:
        taps += 1
    n = np.arange(taps, dtype=float) - taps // 2
    h = np.sinc(n - float(frac))
    if window == "hann":
        h *= np.hanning(taps)
    total = float(np.sum(h))
    if abs(total) < 1.0e-12:
        raise ValueError("fractional delay kernel has near-zero DC gain")
    return h / total


def causal_fractional_delay_window(
    signal: np.ndarray,
    delay_samples: float,
    capture_len: int,
    capture_start_sample: float = 0.0,
    taps: int = 41,
) -> np.ndarray:
    """Finite-window causal receive model with no circular wrap-around."""

    s = np.asarray(signal, dtype=np.complex128)
    if int(capture_len) <= 0:
        raise ValueError("capture_len must be positive")
    rel = float(delay_samples) - float(capture_start_sample)
    k0 = int(np.floor(rel))
    frac = rel - k0
    h = fractional_delay_kernel(frac, taps=taps)
    delayed = np.convolve(s, h.astype(np.complex128), mode="full")
    y = np.zeros(int(capture_len), dtype=np.complex128)
    out_start = k0 - (len(h) // 2)
    for idx, value in enumerate(delayed):
        j = out_start + idx
        if 0 <= j < y.size:
            y[j] += value
    return y


def edge_loss_fraction(
    signal: np.ndarray,
    delay_samples: float,
    capture_len: int,
    capture_start_sample: float = 0.0,
    taps: int = 41,
) -> float:
    s = np.asarray(signal, dtype=np.complex128)
    full = causal_fractional_delay_window(
        s,
        delay_samples,
        capture_len=max(int(capture_len) + s.size + 2 * taps, int(capture_len) + 1),
        capture_start_sample=float(capture_start_sample) - taps,
        taps=taps,
    )
    captured = causal_fractional_delay_window(s, delay_samples, capture_len, capture_start_sample, taps)
    full_energy = float(np.vdot(full, full).real)
    captured_energy = float(np.vdot(captured, captured).real)
    if full_energy <= 1.0e-30:
        return 0.0
    return float(max(0.0, 1.0 - captured_energy / full_energy))


def causal_multipath_window(
    signal: np.ndarray,
    paths: list[tuple[complex, float]],
    capture_len: int,
    capture_start_sample: float = 0.0,
    taps: int = 41,
) -> np.ndarray:
    out = np.zeros(int(capture_len), dtype=np.complex128)
    for amplitude, delay_samples in paths:
        out += complex(amplitude) * causal_fractional_delay_window(
            signal,
            delay_samples=float(delay_samples),
            capture_len=capture_len,
            capture_start_sample=capture_start_sample,
            taps=taps,
        )
    return out
