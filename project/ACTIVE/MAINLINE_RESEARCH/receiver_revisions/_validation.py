"""Strict unit-domain validation shared only by the revised references."""
from __future__ import annotations
import numpy as np

class NoUsableSignal(ValueError):
    """A point estimator cannot choose a meaningful peak from zero data."""

def signal_1d(x, name: str) -> np.ndarray:
    a=np.asarray(x,dtype=np.complex128)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a nonempty finite one-dimensional complex array')
    return a

def positive(x, name: str) -> float:
    y=float(x)
    if not np.isfinite(y) or y <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return y

def grid_1d(x, name: str, minimum: int=2) -> np.ndarray:
    a=np.asarray(x,dtype=float)
    if a.ndim != 1 or a.size < minimum or not np.all(np.isfinite(a)) or np.any(np.diff(a)<=0):
        raise ValueError(f'{name} must be a strictly increasing finite grid of length >= {minimum}')
    return a

def probabilities(x, n: int, name: str) -> np.ndarray:
    a=np.asarray(x,dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)) or np.any(a<0) or a.sum()<=0:
        raise ValueError(f'{name} must be finite nonnegative weights of length {n} with positive sum')
    return a/a.sum()
