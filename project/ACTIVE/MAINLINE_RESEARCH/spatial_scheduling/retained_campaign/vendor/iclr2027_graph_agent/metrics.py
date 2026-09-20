"""Metrics with explicit estimands and tail reporting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, median
from typing import Iterable, Sequence


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires nonempty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    x = sorted(float(v) for v in values)
    pos = q * (len(x) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return x[lo]
    return x[lo] * (hi - pos) + x[hi] * (pos - lo)


def wilson_lower_bound(successes: int, trials: int, z: float = 1.6448536269514722) -> float:
    """One-sided Wilson lower confidence bound; z defaults to 95%."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    p = successes / trials
    z2 = z * z
    center = p + z2 / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return max(0.0, (center - radius) / (1.0 + z2 / trials))


@dataclass(frozen=True, slots=True)
class ProposalMetrics:
    candidate_budget: int
    worlds: int
    conditionally_feasible_worlds: int
    qualifying_hits: int
    conditional_recall: float | None
    mean_exact_regret: float | None
    p95_exact_regret: float | None
    duplicate_fraction: float
    mean_exact_evaluations: float


@dataclass(frozen=True, slots=True)
class TailSummary:
    count: int
    mean: float
    median: float
    p95: float
    maximum: float


def tail_summary(values: Iterable[float]) -> TailSummary:
    x = [float(v) for v in values]
    if not x:
        raise ValueError("tail_summary requires nonempty values")
    return TailSummary(len(x), mean(x), median(x), quantile(x, 0.95), max(x))
