from __future__ import annotations

"""Advanced proposal solvers with exact robust-Schur verification downstream."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .information import prior_information_normalized, schedule_feasibility
from .measurement import StateLayout, build_action_factor
from .schema import PhysicalWorld


@dataclass(frozen=True)
class RelaxationResult:
    action_ids: tuple[int, ...]
    weights: tuple[float, ...]
    solver: str
    status: str
    objective: float
    proposal_only: bool = True


@dataclass(frozen=True)
class ZeroSumScheduleGameResult:
    radar_mixed_strategy: tuple[float, ...]
    game_value: float
    deterministic_maximin_value: float
    randomization_gain: float
    adversary_best_response_index: int
    radar_support_size: int
    radar_policy_entropy_nats: float
    solver: str
    status: str
    candidate_pool_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "radar_mixed_strategy": list(self.radar_mixed_strategy),
            "game_value": self.game_value,
            "deterministic_maximin_value": self.deterministic_maximin_value,
            "randomization_gain": self.randomization_gain,
            "adversary_best_response_index": self.adversary_best_response_index,
            "radar_support_size": self.radar_support_size,
            "radar_policy_entropy_nats": self.radar_policy_entropy_nats,
            "solver": self.solver,
            "status": self.status,
            "candidate_pool_only": self.candidate_pool_only,
            "exact_full_action_space_game_claim_allowed": False,
        }


def solve_zero_sum_schedule_game(payoff: np.ndarray) -> ZeroSumScheduleGameResult:
    """Solve max_p min_j p^T U[:,j] over the supplied candidate pool."""

    matrix = np.asarray(payoff, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("zero-sum payoff must be a finite nonempty candidate-by-tactic matrix")
    candidate_count, tactic_count = matrix.shape
    pure_worst = np.min(matrix, axis=1)
    deterministic_value = float(np.max(pure_worst))
    try:
        from scipy.optimize import linprog

        objective = np.zeros(candidate_count + 1, dtype=float)
        objective[-1] = -1.0
        a_ub = np.zeros((tactic_count, candidate_count + 1), dtype=float)
        a_ub[:, :candidate_count] = -matrix.T
        a_ub[:, -1] = 1.0
        b_ub = np.zeros(tactic_count, dtype=float)
        a_eq = np.zeros((1, candidate_count + 1), dtype=float)
        a_eq[0, :candidate_count] = 1.0
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=np.ones(1),
            bounds=[(0.0, 1.0)] * candidate_count + [(None, None)],
            method="highs",
        )
        if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"linprog failed: {result.message}")
        strategy = np.maximum(np.asarray(result.x[:candidate_count], dtype=float), 0.0)
        strategy /= max(float(strategy.sum()), 1e-300)
        expected_by_tactic = strategy @ matrix
        game_value = float(np.min(expected_by_tactic))
        status = "optimal"
        solver = "scipy_highs_linear_program"
    except Exception as exc:
        best = int(np.argmax(pure_worst))
        strategy = np.zeros(candidate_count, dtype=float)
        strategy[best] = 1.0
        expected_by_tactic = strategy @ matrix
        game_value = float(np.min(expected_by_tactic))
        status = f"deterministic_fallback_after_{type(exc).__name__}"
        solver = "pure_maximin_fallback"
    positive = strategy[strategy > 1e-12]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    return ZeroSumScheduleGameResult(
        tuple(float(value) for value in strategy),
        game_value,
        deterministic_value,
        max(0.0, game_value - deterministic_value),
        int(np.argmin(expected_by_tactic)),
        int(np.count_nonzero(strategy > 1e-8)),
        entropy,
        solver,
        status,
    )


def _round_feasible(world: PhysicalWorld, weights: np.ndarray) -> tuple[int, ...]:
    ranked = [int(i) for i in np.argsort(-weights)]
    selected: list[int] = []
    by_id = {a.action_id: a for a in world.actions}
    for action_id in ranked:
        proposal = [*selected, action_id]
        feasible, _ = schedule_feasibility(world, [by_id[x] for x in proposal])
        if feasible:
            selected.append(action_id)
        if len(selected) >= world.max_schedule_size:
            break
    return tuple(sorted(selected))


def _project_capped_simplex(values: np.ndarray, budget: float, iterations: int = 80) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min() - 1.0), float(values.max())
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        projected = np.clip(values - mid, 0.0, 1.0)
        if projected.sum() > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(values - hi, 0.0, 1.0)


def d_optimal_relaxation_proposal(world: PhysicalWorld, *, prefer_cvxpy: bool = False) -> RelaxationResult:
    """Continuous complete-joint D-optimal relaxation used only as a proposal.

    Nuisance-marginal robust utility is nontrivial and is evaluated exactly by
    the downstream V58 reranker.  This relaxation never labels itself the final
    optimizer or an exact large-world oracle.
    """

    layout = StateLayout.from_world(world)
    prior = prior_information_normalized(world, layout)
    factors = [build_action_factor(world, action, None).observed_information_normalized for action in world.actions]
    n = len(factors)
    energy = np.asarray([a.energy_cost_j for a in world.actions], dtype=float)
    dwell = np.asarray([world.waveform(a.waveform_id).dwell_s for a in world.actions], dtype=float)
    try:
        if not prefer_cvxpy:
            raise ImportError("CVXPY path disabled for bulk world generation")
        import cvxpy as cp  # type: ignore

        weights = cp.Variable(n)
        joint = prior
        for i, factor in enumerate(factors):
            joint = joint + weights[i] * factor
        constraints = [weights >= 0.0, weights <= 1.0, cp.sum(weights) <= world.max_schedule_size, energy @ weights <= world.max_total_energy_j, dwell @ weights <= world.max_total_dwell_s]
        problem = cp.Problem(cp.Maximize(cp.log_det(joint)), constraints)
        installed = set(cp.installed_solvers())
        solver = "CLARABEL" if "CLARABEL" in installed else "SCS" if "SCS" in installed else None
        problem.solve(solver=solver, verbose=False) if solver else problem.solve(verbose=False)
        values = np.asarray(weights.value, dtype=float).reshape(-1)
        if values.size != n or not np.all(np.isfinite(values)):
            raise RuntimeError(f"convex solver returned invalid weights with status={problem.status}")
        action_ids = _round_feasible(world, values)
        return RelaxationResult(action_ids, tuple(float(x) for x in values), f"cvxpy_{solver or 'default'}", str(problem.status), float(problem.value))
    except Exception as exc:
        # Dependency-free projected-gradient fallback.  It optimizes the same
        # continuous joint logdet surrogate under cardinality, then feasibility
        # rounding enforces the full resource constraints.
        weights = np.full(n, min(1.0, world.max_schedule_size / max(1, n)), dtype=float)
        for step in range(120):
            joint = prior.copy()
            for i, factor in enumerate(factors):
                joint += weights[i] * factor
            inverse = np.linalg.pinv(joint)
            gradient = np.asarray([np.trace(inverse @ factor) for factor in factors], dtype=float)
            weights = _project_capped_simplex(weights + 0.08 / np.sqrt(step + 1.0) * gradient / max(np.linalg.norm(gradient), 1e-12), world.max_schedule_size)
        action_ids = _round_feasible(world, weights)
        joint = prior.copy()
        for i, factor in enumerate(factors):
            joint += weights[i] * factor
        objective = float(np.linalg.slogdet(joint)[1])
        return RelaxationResult(action_ids, tuple(float(x) for x in weights), "projected_gradient_fallback", f"fallback_after:{type(exc).__name__}", objective)
