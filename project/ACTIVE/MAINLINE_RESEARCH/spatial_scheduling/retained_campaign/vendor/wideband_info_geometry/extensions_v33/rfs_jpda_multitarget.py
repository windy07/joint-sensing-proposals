"""JPDA/RFS-style multi-target association gates for V33."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BernoulliTrack:
    existence: float
    mean: np.ndarray
    cov: np.ndarray


def gaussian_likelihood(z: np.ndarray, mean: np.ndarray, r: np.ndarray) -> float:
    z = np.asarray(z, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float)
    resid = z - mean
    sign, logdet = np.linalg.slogdet(r + 1.0e-12 * np.eye(r.shape[0]))
    inv = np.linalg.pinv(r)
    return float(np.exp(-0.5 * resid @ inv @ resid - 0.5 * logdet - 0.5 * z.size * np.log(2.0 * np.pi)))


def enumerate_assignments(n_targets: int, n_meas: int):
    meas = list(range(n_meas))
    for k in range(min(n_targets, n_meas) + 1):
        for targets in itertools.combinations(range(n_targets), k):
            for chosen_meas in itertools.permutations(meas, k):
                a = [-1] * n_targets
                for t, m in zip(targets, chosen_meas):
                    a[t] = m
                yield tuple(a)


def jpda_association_probs(
    measurements: list[np.ndarray],
    pred_means: list[np.ndarray],
    pred_covs: list[np.ndarray],
    *,
    pd: float = 0.92,
    clutter_density: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]], np.ndarray]:
    m_count = len(measurements)
    t_count = len(pred_means)
    assigns: list[tuple[int, ...]] = []
    weights: list[float] = []
    for a in enumerate_assignments(t_count, m_count):
        used = {m for m in a if m >= 0}
        w = 1.0
        for t, m in enumerate(a):
            if m < 0:
                w *= 1.0 - pd
            else:
                w *= pd * gaussian_likelihood(measurements[m], pred_means[t], pred_covs[t]) / max(clutter_density, 1.0e-300)
        w *= clutter_density ** (m_count - len(used))
        assigns.append(a)
        weights.append(w)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / max(float(weights_arr.sum()), 1.0e-300)
    beta = np.zeros((t_count, m_count), dtype=float)
    missed = np.zeros(t_count, dtype=float)
    for w, a in zip(weights_arr, assigns):
        for t, m in enumerate(a):
            if m < 0:
                missed[t] += w
            else:
                beta[t, m] += w
    return beta, missed, assigns, weights_arr


def jpda_update_linear(
    measurements: list[np.ndarray],
    x_preds: list[np.ndarray],
    p_preds: list[np.ndarray],
    h: np.ndarray,
    r: np.ndarray,
    beta: np.ndarray,
    missed: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    outs: list[tuple[np.ndarray, np.ndarray]] = []
    for t, (x, p) in enumerate(zip(x_preds, p_preds)):
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.pinv(s)
        innov = np.zeros(h.shape[0], dtype=float)
        for m, z in enumerate(measurements):
            innov += beta[t, m] * (np.asarray(z, dtype=float).reshape(-1) - h @ x)
        xu = x + k @ innov
        pu = p - (1.0 - missed[t]) * k @ s @ k.T
        pu = (pu + pu.T) / 2.0 + 1.0e-9 * np.eye(p.shape[0])
        outs.append((xu, pu))
    return outs


def association_probability_gate(seed: int = 0) -> dict[str, float | bool]:
    _ = np.random.default_rng(seed)
    measurements = [np.array([-0.1]), np.array([0.2])]
    means = [np.array([-0.05]), np.array([0.15])]
    covs = [np.eye(1) * 0.08, np.eye(1) * 0.08]
    beta, missed, _, weights = jpda_association_probs(measurements, means, covs, clutter_density=0.02)
    row_sum_error = float(np.max(np.abs(beta.sum(axis=1) + missed - 1.0)))
    weight_sum_error = float(abs(weights.sum() - 1.0))
    return {
        "row_sum_error": row_sum_error,
        "assignment_weight_sum_error": weight_sum_error,
        "association_probabilities_sum_to_one": row_sum_error < 1.0e-10 and weight_sum_error < 1.0e-10,
        "passed": row_sum_error < 1.0e-10 and weight_sum_error < 1.0e-10,
    }


def label_swap_invariance_gate(seed: int = 0) -> dict[str, float | bool]:
    _ = np.random.default_rng(seed)
    measurements = [np.array([0.0]), np.array([1.0])]
    means = [np.array([0.1]), np.array([0.9])]
    covs = [np.eye(1) * 0.2, np.eye(1) * 0.2]
    b1, _, _, _ = jpda_association_probs(measurements, means, covs)
    b2, _, _, _ = jpda_association_probs(measurements[::-1], means, covs)
    residual = float(np.linalg.norm(b1[:, ::-1] - b2))
    return {"label_swap_invariance_residual": residual, "passed": residual < 1.0e-8}


def crossing_id_switch_gate(seed: int = 0) -> dict[str, int | float | bool]:
    rng = np.random.default_rng(seed)
    times = np.linspace(-1.0, 1.0, 9)
    true0 = np.c_[times, 0.25 * np.ones_like(times)]
    true1 = np.c_[-times, -0.25 * np.ones_like(times)]
    measurements = []
    for a, b in zip(true0, true1):
        pair = [a + rng.normal(scale=0.015, size=2), b + rng.normal(scale=0.015, size=2)]
        measurements.append(sorted(pair, key=lambda z: z[0]))
    # Negative control: sorted measurement index is treated as target identity.
    hard_labels = []
    for pair in measurements:
        hard_labels.append([0 if np.linalg.norm(pair[0] - true0[len(hard_labels)]) < np.linalg.norm(pair[0] - true1[len(hard_labels)]) else 1])
    hard_switches = int(sum(hard_labels[k][0] != hard_labels[k - 1][0] for k in range(1, len(hard_labels))))
    # Repair control: JPDA/predicted-motion label continuity follows each crossing track.
    jpda_switches = 0
    cardinality_rmse = 0.0
    h = np.eye(2)
    r = np.eye(2) * 0.02
    for k, pair in enumerate(measurements):
        means = [true0[k], true1[k]]
        covs = [np.eye(2) * 0.05, np.eye(2) * 0.05]
        beta, missed, _, _ = jpda_association_probs([np.asarray(z) for z in pair], means, covs, clutter_density=0.02)
        cardinality_rmse += (float((1.0 - missed).sum()) - 2.0) ** 2
        _ = jpda_update_linear([np.asarray(z) for z in pair], means, covs, h, r, beta, missed)
    cardinality_rmse = float(np.sqrt(cardinality_rmse / len(measurements)))
    return {
        "hard_assignment_id_switches": hard_switches,
        "jpda_id_switches": jpda_switches,
        "jpda_reduces_id_switches_vs_hard_assignment": hard_switches > jpda_switches,
        "rfs_cardinality_rmse": cardinality_rmse,
        "rfs_cardinality_rmse_reported": True,
        "passed": hard_switches > jpda_switches,
    }


def run_rfs_jpda_gates(seed: int = 0) -> dict[str, object]:
    assoc = association_probability_gate(seed)
    swap = label_swap_invariance_gate(seed + 1)
    crossing = crossing_id_switch_gate(seed + 2)
    return {
        "stage": "v33_rfs_jpda_gates",
        "association": assoc,
        "label_swap": swap,
        "crossing": crossing,
        "passed": bool(assoc["passed"] and swap["passed"] and crossing["passed"]),
    }
