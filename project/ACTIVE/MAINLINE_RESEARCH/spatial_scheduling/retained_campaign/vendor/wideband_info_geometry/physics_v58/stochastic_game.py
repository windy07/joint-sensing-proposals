from __future__ import annotations

"""Partially observed action-reactive radar/EW stochastic game controls.

The game state is the hidden semi-Markov threat mode carried by
``dynamic_episode.AdversaryDynamicState``. Radar decisions are bounded candidate
schedules and adversary decisions are tactic modes.  This module intentionally
keeps several controls separate:

* static matrix minimax (history free),
* Bayesian finite-horizon belief control,
* deterministic Stackelberg leader/follower timing,
* distributionally robust/CVaR scoring under belief ambiguity,
* online Dirichlet transition estimation, and
* an independently trained tabular recurrent-belief Q policy.

The learned policy receives rewards and observations from its own simulated
episodes.  It is never initialized from, distilled from, or labeled by the
model-based controller.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .cognitive_ew import HiddenTacticModel, observation_distribution, update_tactic_belief
from .optimization import solve_zero_sum_schedule_game


@dataclass(frozen=True)
class GameRewardTerms:
    estimation_information: float
    task_loss: float
    tail_risk: float
    action_cost: float
    switching_cost: float
    emission_cost: float
    communication_cost: float
    future_resource_value: float
    threat_information_value: float

    @property
    def total(self) -> float:
        """Each operational term enters exactly once."""

        return float(
            self.estimation_information
            - self.task_loss
            - self.tail_risk
            - self.action_cost
            - self.switching_cost
            - self.emission_cost
            - self.communication_cost
            + self.future_resource_value
            + self.threat_information_value
        )


@dataclass
class OnlineOpponentModel:
    tactic_count: int
    radar_action_count: int
    prior_count: float = 1.0
    forgetting: float = 0.995
    counts: np.ndarray = field(init=False)
    duration_counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.tactic_count <= 0 or self.radar_action_count <= 0:
            raise ValueError("opponent model dimensions must be positive")
        if self.prior_count <= 0.0 or not 0.0 < self.forgetting <= 1.0:
            raise ValueError("opponent prior/forgetting is invalid")
        self.counts = np.full(
            (self.radar_action_count, self.tactic_count, self.tactic_count),
            float(self.prior_count),
            dtype=float,
        )
        self.duration_counts = np.full((self.tactic_count, 16), float(self.prior_count), dtype=float)

    def update(self, radar_action: int, previous_tactic: int, next_tactic: int, duration_steps: int) -> None:
        if not 0 <= radar_action < self.radar_action_count:
            raise IndexError("radar action index out of range")
        if not 0 <= previous_tactic < self.tactic_count or not 0 <= next_tactic < self.tactic_count:
            raise IndexError("tactic index out of range")
        self.counts *= self.forgetting
        self.duration_counts *= self.forgetting
        self.counts[radar_action, previous_tactic, next_tactic] += 1.0
        duration_bin = min(max(int(duration_steps), 1), self.duration_counts.shape[1]) - 1
        self.duration_counts[next_tactic, duration_bin] += 1.0

    def transition(self) -> np.ndarray:
        return self.counts / np.sum(self.counts, axis=-1, keepdims=True)

    def duration_pmf(self) -> np.ndarray:
        return self.duration_counts / np.sum(self.duration_counts, axis=-1, keepdims=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "transition_by_radar_action": self.transition().tolist(),
            "duration_pmf": self.duration_pmf().tolist(),
            "forgetting": float(self.forgetting),
            "prior_count": float(self.prior_count),
        }


def lower_tail_mean(values: np.ndarray, probability: float, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not 0.0 < probability <= 1.0:
        raise ValueError("tail values/probability are invalid")
    if weights is None:
        weights = np.full(values.size, 1.0 / values.size)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape != values.shape or np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("tail weights must be a probability vector")
    order = np.argsort(values)
    mass_remaining = float(probability)
    accumulation = 0.0
    used = 0.0
    for index in order:
        take = min(float(weights[index]), mass_remaining)
        accumulation += take * float(values[index])
        used += take
        mass_remaining -= take
        if mass_remaining <= 1e-15:
            break
    return float(accumulation / max(used, 1e-15))


def l1_worst_case_belief(expectation_values: np.ndarray, nominal: np.ndarray, radius: float) -> np.ndarray:
    """Minimize expected radar utility over an L1 ball around a belief.

    Probability mass is shifted from high-utility to low-utility states.  Moving
    mass ``delta`` changes L1 distance by ``2*delta``; this greedy transport is
    exact for a linear objective on the simplex with an L1 ambiguity ball.
    """

    values = np.asarray(expectation_values, dtype=float).reshape(-1)
    probability = np.asarray(nominal, dtype=float).reshape(-1).copy()
    if values.shape != probability.shape or np.any(probability < 0.0) or not np.isclose(np.sum(probability), 1.0):
        raise ValueError("belief/value shape or probability is invalid")
    budget = float(np.clip(radius, 0.0, 2.0)) / 2.0
    low = list(np.argsort(values))
    high = list(np.argsort(values)[::-1])
    i = j = 0
    while budget > 1e-15 and i < len(low) and j < len(high):
        receiver = int(low[i])
        donor = int(high[j])
        if values[receiver] >= values[donor] - 1e-15:
            break
        move = min(1.0 - probability[receiver], probability[donor], budget)
        probability[receiver] += move
        probability[donor] -= move
        budget -= move
        if probability[receiver] >= 1.0 - 1e-15:
            i += 1
        if probability[donor] <= 1e-15:
            j += 1
    probability = np.maximum(probability, 0.0)
    probability /= np.sum(probability)
    return probability


def distributionally_robust_scores(
    payoff: np.ndarray,
    belief: np.ndarray,
    *,
    ambiguity_radius: float = 0.20,
    cvar_probability: float = 0.20,
    cvar_weight: float = 0.50,
) -> dict[str, object]:
    matrix = np.asarray(payoff, dtype=float)
    probability = np.asarray(belief, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != probability.size:
        raise ValueError("payoff and belief dimensions disagree")
    rows: list[dict[str, object]] = []
    for action, values in enumerate(matrix):
        worst_belief = l1_worst_case_belief(values, probability, ambiguity_radius)
        expectation = float(worst_belief @ values)
        cvar = lower_tail_mean(values, cvar_probability, worst_belief)
        score = (1.0 - cvar_weight) * expectation + cvar_weight * cvar
        rows.append({
            "radar_action": int(action),
            "worst_case_belief": worst_belief.tolist(),
            "worst_case_expectation": expectation,
            "lower_tail_cvar_utility": cvar,
            "distributionally_robust_score": float(score),
        })
    selected = max(rows, key=lambda row: (float(row["distributionally_robust_score"]), -int(row["radar_action"])))
    return {
        "selected_action": int(selected["radar_action"]),
        "rows": rows,
        "ambiguity_radius_l1": float(ambiguity_radius),
        "lower_tail_probability": float(cvar_probability),
        "cvar_weight": float(cvar_weight),
    }


def stackelberg_leader_solution(payoff: np.ndarray, adversary_response_cost: np.ndarray | None = None) -> dict[str, object]:
    """Radar commits first; adversary observes and minimizes radar utility.

    ``adversary_response_cost`` is a nonnegative cost paid by the adversary and
    therefore added to radar utility when deciding the follower response.  This
    timing model is kept separate from simultaneous minimax.
    """

    matrix = np.asarray(payoff, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Stackelberg payoff must be finite and two-dimensional")
    cost = np.zeros_like(matrix) if adversary_response_cost is None else np.asarray(adversary_response_cost, dtype=float)
    if cost.shape != matrix.shape or np.any(cost < 0.0):
        raise ValueError("adversary response cost is invalid")
    follower = np.argmin(matrix + cost, axis=1)
    leader_values = matrix[np.arange(matrix.shape[0]), follower]
    selected = int(np.argmax(leader_values))
    return {
        "selected_radar_action": selected,
        "adversary_best_response_by_radar_action": follower.astype(int).tolist(),
        "leader_value_by_radar_action": leader_values.tolist(),
        "leader_value": float(leader_values[selected]),
        "timing_assumption": "radar_commits_then_adversary_observes_and_best_responds",
    }


def exact_belief_control(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    belief: np.ndarray,
    *,
    horizon: int = 3,
    discount: float = 0.95,
    safety_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Exact small-world Bayesian POMDP control by observation-tree recursion."""

    matrix = np.asarray(payoff, dtype=float)
    prior = np.asarray(belief, dtype=float).reshape(-1)
    model.validate(matrix.shape[0])
    if prior.size != matrix.shape[1] or np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0):
        raise ValueError("belief is invalid")
    if horizon <= 0 or not 0.0 <= discount <= 1.0:
        raise ValueError("belief horizon/discount is invalid")
    mask = np.ones(matrix.shape[0], dtype=bool) if safety_mask is None else np.asarray(safety_mask, dtype=bool)
    if mask.shape != (matrix.shape[0],) or not np.any(mask):
        raise ValueError("belief controller requires at least one safe action")

    safe_actions = np.flatnonzero(mask)
    cache: dict[tuple[tuple[float, ...], int], float] = {}
    counters = {"leaf_value_evaluations": 0, "internal_value_evaluations": 0}

    def action_values(current: np.ndarray, remaining: int) -> np.ndarray:
        # At the terminal horizon, V_1(b)=max_a r_a^T b exactly.  Computing
        # all terminal action values in one matrix product avoids constructing
        # and retaining an exponential number of one-use leaf cache entries.
        q = np.full(matrix.shape[0], -math.inf)
        q[safe_actions] = matrix[safe_actions] @ current
        if remaining == 1:
            counters["leaf_value_evaluations"] += 1
            return q
        counters["internal_value_evaluations"] += 1
        for action in safe_actions:
            future = 0.0
            distribution = observation_distribution(current, model, int(action))
            for observation, probability in enumerate(distribution):
                if probability <= 0.0:
                    continue
                posterior = update_tactic_belief(current, model, int(action), int(observation))
                future += float(probability) * state_value(posterior, remaining - 1)
            q[action] += discount * future
        return q

    def state_value(current: np.ndarray, remaining: int) -> float:
        if remaining == 1:
            return float(np.max(action_values(current, remaining)))
        key = (tuple(float(value) for value in current), remaining)
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = float(np.max(action_values(current, remaining)))
        cache[key] = value
        return value

    q = action_values(prior, int(horizon))
    selected = int(np.argmax(q))
    value = float(q[selected])
    return {
        "selected_action": selected,
        "value": value,
        "q_values": q.tolist(),
        "horizon": int(horizon),
        "discount": float(discount),
        "solver_kind": "exact_reachable_belief_tree",
        "is_exact": True,
        "exact_scope": "finite hidden-mode observation tree over registered bounded radar actions",
        "solver_revision": "exact_reachable_belief_tree_vectorized_terminal_no_leaf_cache_r2",
        "internal_belief_cache_entries": len(cache),
        "leaf_cache_entries": 0,
        **counters,
    }


def exact_resource_belief_control(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    belief: np.ndarray,
    *,
    horizon: int = 3,
    discount: float = 0.95,
    safety_mask: np.ndarray | None = None,
    action_resource_costs: np.ndarray,
    initial_resource_fraction: float = 1.0,
    switching_costs: np.ndarray,
    previous_action: int = -1,
) -> dict[str, object]:
    """Exact finite-horizon belief control with resource and switching state.

    This solver is deliberately restricted to bounded calibration problems.
    Its sufficient state is ``(belief, remaining_resource, previous_action)``;
    therefore its recurrence is exactly the one approximated by the online
    search rather than a resource-free surrogate.
    """

    matrix = np.asarray(payoff, dtype=float)
    prior = np.asarray(belief, dtype=float).reshape(-1)
    model.validate(matrix.shape[0])
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("belief-control payoff must be finite and two-dimensional")
    if (
        prior.size != matrix.shape[1]
        or np.any(prior < 0.0)
        or not np.isclose(np.sum(prior), 1.0)
    ):
        raise ValueError("belief is invalid")
    if horizon <= 0 or not 0.0 <= discount <= 1.0:
        raise ValueError("belief horizon/discount is invalid")
    mask = (
        np.ones(matrix.shape[0], dtype=bool)
        if safety_mask is None
        else np.asarray(safety_mask, dtype=bool)
    )
    if mask.shape != (matrix.shape[0],) or not np.any(mask):
        raise ValueError("belief controller requires at least one safe action")
    costs = np.asarray(action_resource_costs, dtype=float).reshape(-1)
    if (
        costs.shape != (matrix.shape[0],)
        or np.any(~np.isfinite(costs))
        or np.any(costs < 0.0)
    ):
        raise ValueError("exact belief action-resource costs are invalid")
    resource0 = float(initial_resource_fraction)
    if not math.isfinite(resource0) or resource0 < 0.0:
        raise ValueError("exact belief initial resource fraction is invalid")
    switch = np.asarray(switching_costs, dtype=float)
    if (
        switch.shape != (matrix.shape[0] + 1, matrix.shape[0])
        or np.any(~np.isfinite(switch))
        or np.any(switch < 0.0)
    ):
        raise ValueError("exact belief switching-cost matrix is invalid")
    previous0 = matrix.shape[0] if previous_action < 0 else int(previous_action)
    if not 0 <= previous0 <= matrix.shape[0]:
        raise ValueError("exact belief previous action is invalid")

    cache: dict[tuple[tuple[float, ...], str, int, int], float] = {}
    counters = {
        "leaf_value_evaluations": 0,
        "internal_value_evaluations": 0,
    }

    def feasible_actions(resource: float) -> np.ndarray:
        return np.flatnonzero(mask & (costs <= float(resource) + 1.0e-12))

    def action_values(
        current: np.ndarray,
        resource: float,
        previous: int,
        remaining: int,
    ) -> np.ndarray:
        feasible = feasible_actions(resource)
        q = np.full(matrix.shape[0], -math.inf)
        if not feasible.size:
            return q
        q[feasible] = matrix[feasible] @ current - switch[previous, feasible]
        if remaining == 1:
            counters["leaf_value_evaluations"] += 1
            return q
        counters["internal_value_evaluations"] += 1
        for action in feasible:
            future = 0.0
            distribution = observation_distribution(current, model, int(action))
            next_resource = max(0.0, resource - float(costs[action]))
            for observation, probability in enumerate(distribution):
                if probability <= 0.0:
                    continue
                posterior = update_tactic_belief(
                    current,
                    model,
                    int(action),
                    int(observation),
                )
                future += float(probability) * state_value(
                    posterior,
                    next_resource,
                    int(action),
                    remaining - 1,
                )
            q[action] += discount * future
        return q

    def state_value(
        current: np.ndarray,
        resource: float,
        previous: int,
        remaining: int,
    ) -> float:
        feasible = feasible_actions(resource)
        if not feasible.size:
            return 0.0
        key = (
            tuple(float(value) for value in current),
            float(resource).hex(),
            int(previous),
            int(remaining),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = float(
            np.max(action_values(current, resource, previous, remaining))
        )
        cache[key] = value
        return value

    root_feasible = feasible_actions(resource0)
    if not root_feasible.size:
        raise ValueError("belief controller requires a resource-feasible root action")
    q = action_values(prior, resource0, previous0, int(horizon))
    selected = int(np.argmax(q))
    return {
        "selected_action": selected,
        "value": float(q[selected]),
        "q_values": [
            float(value) if math.isfinite(float(value)) else None for value in q
        ],
        "horizon": int(horizon),
        "discount": float(discount),
        "solver_kind": "exact_resource_reachable_belief_tree",
        "is_exact": True,
        "exact_scope": (
            "finite hidden-mode observation tree over bounded actions with "
            "registered resource depletion and switching state"
        ),
        "solver_revision": "exact_resource_reachable_belief_tree_r1",
        "internal_belief_resource_cache_entries": len(cache),
        "action_resource_costs": costs.tolist(),
        "initial_resource_fraction": resource0,
        "switching_costs": switch.tolist(),
        "previous_action": int(previous0),
        "resource_feasibility_is_propagated_through_tree": True,
        **counters,
    }


@dataclass(frozen=True)
class OnlineBeliefTreeSearchConfig:
    """Fixed-budget online POMDP search used beyond the exact-tree regime.

    Tree construction and the independent root-action evaluation use disjoint
    deterministic random streams.  The reported intervals quantify Monte Carlo
    evaluation uncertainty for the frozen tree policy; they are not bounds on
    the gap to the unknown finite-horizon optimum.
    """

    simulations: int = 4096
    evaluation_rollouts_per_action: int = 128
    exploration_constant: float = 1.35
    seed: int = 6502
    rollout_policy: Literal["myopic_belief"] = "myopic_belief"

    def validate(self) -> None:
        if self.simulations <= 0 or self.evaluation_rollouts_per_action <= 1:
            raise ValueError("online belief search requires positive construction and evaluation budgets")
        if not math.isfinite(self.exploration_constant) or self.exploration_constant < 0.0:
            raise ValueError("online belief search exploration constant must be finite and nonnegative")
        if self.rollout_policy != "myopic_belief":
            raise ValueError("unsupported online belief-search rollout policy")


@dataclass
class _BeliefSearchNode:
    visits: np.ndarray
    value_sum: np.ndarray
    value_square_sum: np.ndarray


def online_pomcp_belief_control(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    belief: np.ndarray,
    *,
    horizon: int,
    discount: float = 0.95,
    safety_mask: np.ndarray | None = None,
    config: OnlineBeliefTreeSearchConfig | None = None,
    action_resource_costs: np.ndarray | None = None,
    initial_resource_fraction: float = 1.0,
    switching_costs: np.ndarray | None = None,
    previous_action: int = -1,
) -> dict[str, object]:
    """Approximate a finite-horizon belief policy with online Monte Carlo search.

    The hidden tactic is sampled only inside the generative simulator.  Every
    action chosen by the tree or rollout policy is a function of observable
    history and the Bayesian belief, never of the sampled hidden tactic.  Exact
    immediate expectations and Bayesian updates are retained; only the future
    max-expectation tree is approximated under a declared simulation budget.
    """

    matrix = np.asarray(payoff, dtype=float)
    prior = np.asarray(belief, dtype=float).reshape(-1)
    model.validate(matrix.shape[0])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("belief-control payoff must be finite and two-dimensional")
    if prior.size != matrix.shape[1] or np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0):
        raise ValueError("belief is invalid")
    if horizon <= 0 or not 0.0 <= discount <= 1.0:
        raise ValueError("belief horizon/discount is invalid")
    mask = np.ones(matrix.shape[0], dtype=bool) if safety_mask is None else np.asarray(safety_mask, dtype=bool)
    if mask.shape != (matrix.shape[0],) or not np.any(mask):
        raise ValueError("belief controller requires at least one safe action")
    cfg = config or OnlineBeliefTreeSearchConfig()
    cfg.validate()
    costs = (
        np.zeros(matrix.shape[0], dtype=float)
        if action_resource_costs is None
        else np.asarray(action_resource_costs, dtype=float).reshape(-1)
    )
    if (
        costs.shape != (matrix.shape[0],)
        or np.any(~np.isfinite(costs))
        or np.any(costs < 0.0)
    ):
        raise ValueError("POMCP action-resource costs are invalid")
    resource0 = float(initial_resource_fraction)
    if not math.isfinite(resource0) or resource0 < 0.0:
        raise ValueError("POMCP initial resource fraction is invalid")
    switch = (
        np.zeros((matrix.shape[0] + 1, matrix.shape[0]), dtype=float)
        if switching_costs is None
        else np.asarray(switching_costs, dtype=float)
    )
    if (
        switch.shape != (matrix.shape[0] + 1, matrix.shape[0])
        or np.any(~np.isfinite(switch))
        or np.any(switch < 0.0)
    ):
        raise ValueError("POMCP switching-cost matrix is invalid")
    previous0 = matrix.shape[0] if previous_action < 0 else int(previous_action)
    if not 0 <= previous0 <= matrix.shape[0]:
        raise ValueError("POMCP previous action is invalid")

    def feasible_actions(resource: float) -> np.ndarray:
        return np.flatnonzero(mask & (costs <= float(resource) + 1.0e-12))

    safe_actions = feasible_actions(resource0)
    if not safe_actions.size:
        raise ValueError("belief controller requires a resource-feasible root action")
    reward_scale = max(float(np.ptp(matrix)), float(np.std(matrix)), 1.0e-9)
    nodes: dict[tuple[int, ...], _BeliefSearchNode] = {}
    construction_rng = np.random.default_rng(cfg.seed)

    def new_node() -> _BeliefSearchNode:
        return _BeliefSearchNode(
            visits=np.zeros(matrix.shape[0], dtype=np.int64),
            value_sum=np.zeros(matrix.shape[0], dtype=float),
            value_square_sum=np.zeros(matrix.shape[0], dtype=float),
        )

    def reward_value(action: int, tactic: int, previous: int) -> float:
        return float(matrix[action, tactic] - switch[previous, action])

    def myopic_action(current: np.ndarray, resource: float, previous: int) -> int:
        feasible = feasible_actions(resource)
        if not feasible.size:
            return -1
        values = matrix[feasible] @ current - switch[previous, feasible]
        return int(feasible[int(np.argmax(values))])

    def transition(
        current: np.ndarray,
        tactic: int,
        action: int,
        rng: np.random.Generator,
    ) -> tuple[int, int, np.ndarray]:
        next_tactic = int(rng.choice(matrix.shape[1], p=model.transition_by_radar_action[action, tactic]))
        observation = int(rng.choice(len(model.observation_ids), p=model.observation_likelihood[next_tactic]))
        posterior = update_tactic_belief(current, model, action, observation)
        return next_tactic, observation, posterior

    def rollout(
        current: np.ndarray,
        tactic: int,
        resource: float,
        previous: int,
        remaining: int,
        rng: np.random.Generator,
    ) -> float:
        total = 0.0
        weight = 1.0
        for _ in range(remaining):
            action = myopic_action(current, resource, previous)
            if action < 0:
                break
            total += weight * reward_value(action, tactic, previous)
            resource = max(0.0, resource - float(costs[action]))
            if weight == 0.0:
                break
            tactic, _observation, current = transition(current, tactic, action, rng)
            previous = action
            weight *= discount
        return float(total)

    def simulate(
        history: tuple[int, ...],
        current: np.ndarray,
        tactic: int,
        resource: float,
        previous: int,
        remaining: int,
        rng: np.random.Generator,
    ) -> float:
        if remaining <= 0:
            return 0.0
        node = nodes.setdefault(history, new_node())
        feasible = feasible_actions(resource)
        if not feasible.size:
            return 0.0
        unvisited = feasible[node.visits[feasible] == 0]
        if unvisited.size:
            action = int(unvisited[0])
        else:
            means = node.value_sum[feasible] / node.visits[feasible]
            total_visits = max(int(np.sum(node.visits[feasible])), 1)
            bonuses = (
                cfg.exploration_constant
                * reward_scale
                * np.sqrt(math.log(total_visits + 1.0) / node.visits[feasible])
            )
            action = int(feasible[int(np.argmax(means + bonuses))])
        reward = reward_value(action, tactic, previous)
        next_resource = max(0.0, resource - float(costs[action]))
        future = 0.0
        if remaining > 1:
            next_tactic, observation, posterior = transition(current, tactic, action, rng)
            child_history = (*history, action, observation)
            if child_history in nodes:
                future = simulate(
                    child_history,
                    posterior,
                    next_tactic,
                    next_resource,
                    action,
                    remaining - 1,
                    rng,
                )
            else:
                nodes[child_history] = new_node()
                future = rollout(
                    posterior,
                    next_tactic,
                    next_resource,
                    action,
                    remaining - 1,
                    rng,
                )
        value = reward + discount * future
        node.visits[action] += 1
        node.value_sum[action] += value
        node.value_square_sum[action] += value * value
        return float(value)

    root = nodes.setdefault((), new_node())
    for _ in range(cfg.simulations):
        tactic = int(construction_rng.choice(matrix.shape[1], p=prior))
        simulate(
            (),
            prior,
            tactic,
            resource0,
            previous0,
            int(horizon),
            construction_rng,
        )

    root_internal_means = np.full(matrix.shape[0], math.nan)
    visited_root = root.visits > 0
    root_internal_means[visited_root] = (
        root.value_sum[visited_root] / root.visits[visited_root]
    )
    selectable_root = safe_actions[visited_root[safe_actions]]
    if not selectable_root.size:
        raise RuntimeError("online belief search did not visit a feasible root action")
    selected = int(
        selectable_root[
            int(np.argmax(root_internal_means[selectable_root]))
        ]
    )

    def frozen_tree_action(
        history: tuple[int, ...],
        current: np.ndarray,
        resource: float,
        previous: int,
    ) -> int:
        feasible = feasible_actions(resource)
        if not feasible.size:
            return -1
        node = nodes.get(history)
        if node is None or not np.any(node.visits[feasible] > 0):
            return myopic_action(current, resource, previous)
        visited = feasible[node.visits[feasible] > 0]
        means = node.value_sum[visited] / node.visits[visited]
        return int(visited[int(np.argmax(means))])

    def evaluate_root_action(root_action: int, rng: np.random.Generator) -> float:
        current = prior.copy()
        tactic = int(rng.choice(matrix.shape[1], p=current))
        resource = resource0
        previous = previous0
        history: tuple[int, ...] = ()
        total = 0.0
        weight = 1.0
        for depth in range(int(horizon)):
            action = (
                int(root_action)
                if depth == 0
                else frozen_tree_action(history, current, resource, previous)
            )
            if action < 0:
                break
            total += weight * reward_value(action, tactic, previous)
            resource = max(0.0, resource - float(costs[action]))
            if depth + 1 >= int(horizon):
                break
            tactic, observation, current = transition(current, tactic, action, rng)
            history = (*history, action, observation)
            previous = action
            weight *= discount
        return float(total)

    q = np.full(matrix.shape[0], -math.inf)
    standard_errors = np.full(matrix.shape[0], math.nan)
    confidence_intervals: list[list[float] | None] = [None] * matrix.shape[0]
    evaluation_count = int(cfg.evaluation_rollouts_per_action)
    for action in safe_actions:
        evaluation_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(action), 0xB3113F]))
        samples = np.asarray(
            [evaluate_root_action(int(action), evaluation_rng) for _ in range(evaluation_count)],
            dtype=float,
        )
        mean = float(np.mean(samples))
        standard_error = float(np.std(samples, ddof=1) / math.sqrt(samples.size))
        q[int(action)] = mean
        standard_errors[int(action)] = standard_error
        confidence_intervals[int(action)] = [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
    return {
        "selected_action": selected,
        "value": float(q[selected]),
        "q_values": q.tolist(),
        "horizon": int(horizon),
        "discount": float(discount),
        "solver_kind": "online_pomcp_mcts",
        "is_exact": False,
        "exact_scope": None,
        "approximation_scope": "fixed-budget online history-tree search with exact Bayesian belief updates",
        "simulation_budget": int(cfg.simulations),
        "evaluation_rollouts_per_safe_action": evaluation_count,
        "tree_node_count": len(nodes),
        "root_action_visits": root.visits.astype(int).tolist(),
        "root_internal_search_means": root_internal_means.tolist(),
        "selection_q_values": root_internal_means.tolist(),
        "root_action_standard_errors": standard_errors.tolist(),
        "root_action_monte_carlo_95_intervals": confidence_intervals,
        "selection_rule": "argmax_root_internal_search_mean",
        "independent_root_evaluation_used_for_selection": False,
        "uncertainty_semantics": "independent frozen-tree policy Monte Carlo interval; not an optimality-gap bound",
        "rollout_policy": cfg.rollout_policy,
        "seed": int(cfg.seed),
        "solver_revision": "observable_history_pomcp_root_search_selection_r2",
        "action_resource_costs": costs.tolist(),
        "initial_resource_fraction": resource0,
        "switching_costs": switch.tolist(),
        "resource_feasibility_is_propagated_through_tree": True,
    }


def scalable_belief_control(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    belief: np.ndarray,
    *,
    horizon: int = 3,
    discount: float = 0.95,
    safety_mask: np.ndarray | None = None,
    solver: Literal["auto", "exact", "online_pomcp"] = "auto",
    exact_horizon_limit: int = 4,
    online_config: OnlineBeliefTreeSearchConfig | None = None,
    action_resource_costs: np.ndarray | None = None,
    initial_resource_fraction: float = 1.0,
    switching_costs: np.ndarray | None = None,
    previous_action: int = -1,
) -> dict[str, object]:
    """Select the exact or declared approximate solver before tree expansion."""

    if solver not in {"auto", "exact", "online_pomcp"}:
        raise ValueError("unknown belief-control solver")
    if exact_horizon_limit <= 0:
        raise ValueError("exact horizon limit must be positive")
    dynamic_resource_model = (
        action_resource_costs is not None or switching_costs is not None
    )
    selected_solver = (
        "exact"
        if solver == "exact"
        or (
            solver == "auto"
            and horizon <= exact_horizon_limit
            and not dynamic_resource_model
        )
        else "online_pomcp"
    )
    if selected_solver == "exact":
        if dynamic_resource_model:
            matrix = np.asarray(payoff, dtype=float)
            costs = (
                np.zeros(matrix.shape[0], dtype=float)
                if action_resource_costs is None
                else np.asarray(action_resource_costs, dtype=float)
            )
            switch = (
                np.zeros((matrix.shape[0] + 1, matrix.shape[0]), dtype=float)
                if switching_costs is None
                else np.asarray(switching_costs, dtype=float)
            )
            result = exact_resource_belief_control(
                payoff,
                model,
                belief,
                horizon=horizon,
                discount=discount,
                safety_mask=safety_mask,
                action_resource_costs=costs,
                initial_resource_fraction=initial_resource_fraction,
                switching_costs=switch,
                previous_action=previous_action,
            )
        else:
            result = exact_belief_control(
                payoff,
                model,
                belief,
                horizon=horizon,
                discount=discount,
                safety_mask=safety_mask,
            )
    else:
        result = online_pomcp_belief_control(
            payoff,
            model,
            belief,
            horizon=horizon,
            discount=discount,
            safety_mask=safety_mask,
            config=online_config,
            action_resource_costs=action_resource_costs,
            initial_resource_fraction=initial_resource_fraction,
            switching_costs=switching_costs,
            previous_action=previous_action,
        )
    result["solver_selection"] = {
        "requested": solver,
        "selected": selected_solver,
        "exact_horizon_limit": int(exact_horizon_limit),
        "horizon": int(horizon),
        "silent_horizon_reduction": False,
        "dynamic_resource_model_requested": bool(dynamic_resource_model),
    }
    return result


def calibrate_online_belief_control(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    belief: np.ndarray,
    *,
    exact_horizon: int = 4,
    discount: float = 0.95,
    safety_mask: np.ndarray | None = None,
    simulations: int = 2048,
    evaluation_rollouts_per_action: int = 96,
    exploration_constant: float = 1.35,
    seeds: Sequence[int] = (6600, 6601, 6602, 6603),
    action_resource_costs: np.ndarray | None = None,
    initial_resource_fraction: float = 1.0,
    switching_costs: np.ndarray | None = None,
    previous_action: int = -1,
) -> dict[str, object]:
    """Calibrate online-search decisions on a horizon with an exact solution."""

    identities = tuple(int(seed) for seed in seeds)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("belief-search calibration seeds must be nonempty and unique")
    dynamic_resource_model = (
        action_resource_costs is not None or switching_costs is not None
    )
    matrix = np.asarray(payoff, dtype=float)
    costs = (
        np.zeros(matrix.shape[0], dtype=float)
        if action_resource_costs is None
        else np.asarray(action_resource_costs, dtype=float)
    )
    switch = (
        np.zeros((matrix.shape[0] + 1, matrix.shape[0]), dtype=float)
        if switching_costs is None
        else np.asarray(switching_costs, dtype=float)
    )
    exact = (
        exact_resource_belief_control(
            payoff,
            model,
            belief,
            horizon=exact_horizon,
            discount=discount,
            safety_mask=safety_mask,
            action_resource_costs=costs,
            initial_resource_fraction=initial_resource_fraction,
            switching_costs=switch,
            previous_action=previous_action,
        )
        if dynamic_resource_model
        else exact_belief_control(
            payoff,
            model,
            belief,
            horizon=exact_horizon,
            discount=discount,
            safety_mask=safety_mask,
        )
    )
    exact_q = np.asarray(exact["q_values"], dtype=float)
    finite = np.isfinite(exact_q)
    exact_selected = int(exact["selected_action"])
    rows: list[dict[str, object]] = []
    for seed in identities:
        approximate = online_pomcp_belief_control(
            payoff,
            model,
            belief,
            horizon=exact_horizon,
            discount=discount,
            safety_mask=safety_mask,
            config=OnlineBeliefTreeSearchConfig(
                simulations=simulations,
                evaluation_rollouts_per_action=evaluation_rollouts_per_action,
                exploration_constant=exploration_constant,
                seed=seed,
            ),
            action_resource_costs=(costs if dynamic_resource_model else None),
            initial_resource_fraction=initial_resource_fraction,
            switching_costs=(switch if dynamic_resource_model else None),
            previous_action=previous_action,
        )
        approximate_q = np.asarray(approximate["q_values"], dtype=float)
        search_q = np.asarray(approximate["selection_q_values"], dtype=float)
        search_finite = finite & np.isfinite(search_q)
        if not np.any(search_finite):
            raise RuntimeError("online calibration has no finite root-search values")
        selected = int(approximate["selected_action"])
        rows.append(
            {
                "seed": seed,
                "selected_action": selected,
                "matches_exact_selected_action": selected == exact_selected,
                "exact_objective_regret": float(exact_q[exact_selected] - exact_q[selected]),
                "q_rmse_against_exact_optimal_continuation": float(
                    math.sqrt(np.mean((approximate_q[finite] - exact_q[finite]) ** 2))
                ),
                "root_search_q_rmse_against_exact_optimal_continuation": float(
                    math.sqrt(
                        np.mean(
                            (search_q[search_finite] - exact_q[search_finite]) ** 2
                        )
                    )
                ),
                "online_selection_rule": str(approximate["selection_rule"]),
                "independent_root_evaluation_used_for_selection": bool(
                    approximate["independent_root_evaluation_used_for_selection"]
                ),
                "selected_action_monte_carlo_95_interval": approximate[
                    "root_action_monte_carlo_95_intervals"
                ][selected],
                "tree_node_count": int(approximate["tree_node_count"]),
            }
        )
    regrets = np.asarray([float(row["exact_objective_regret"]) for row in rows], dtype=float)
    rmses = np.asarray([float(row["q_rmse_against_exact_optimal_continuation"]) for row in rows], dtype=float)
    agreement = float(np.mean([bool(row["matches_exact_selected_action"]) for row in rows]))
    return {
        "stage": "online_belief_search_exact_small_world_calibration",
        "status": "passed" if agreement >= 0.75 and float(np.max(regrets)) <= 0.05 else "failed",
        "exact_horizon": int(exact_horizon),
        "target_long_horizon_not_declared_exact": True,
        "exact_selected_action": exact_selected,
        "exact_q_values": [float(value) if math.isfinite(float(value)) else None for value in exact_q],
        "replicate_count": len(rows),
        "selected_action_agreement_rate": agreement,
        "mean_exact_objective_regret": float(np.mean(regrets)),
        "maximum_exact_objective_regret": float(np.max(regrets)),
        "mean_q_rmse_against_exact_optimal_continuation": float(np.mean(rmses)),
        "rows": rows,
        "configuration": {
            "simulations": int(simulations),
            "evaluation_rollouts_per_action": int(evaluation_rollouts_per_action),
            "exploration_constant": float(exploration_constant),
            "seeds": list(identities),
            "dynamic_resource_and_switching_state_matched": bool(
                dynamic_resource_model
            ),
            "action_resource_costs": (
                costs.tolist() if dynamic_resource_model else None
            ),
            "initial_resource_fraction": float(initial_resource_fraction),
            "switching_costs": (
                switch.tolist() if dynamic_resource_model else None
            ),
            "previous_action": int(previous_action),
        },
        "interpretation": {
            "decision_agreement_is_primary": True,
            "q_rmse_compares_a_frozen_approximate_policy_with_exact_optimal_continuation": True,
            "calibration_does_not_prove_long_horizon_optimality": True,
            "calibration_does_not_convert_online_search_into_an_exact_solver": True,
        },
    }


@dataclass(frozen=True)
class IndependentRLConfig:
    episodes: int = 600
    steps_per_episode: int = 20
    confidence_bins: int = 5
    resource_bins: int = 5
    learning_rate: float = 0.10
    discount: float = 0.94
    initial_epsilon: float = 0.45
    final_epsilon: float = 0.03
    seed: int = 6501

    def validate(self) -> None:
        if min(self.episodes, self.steps_per_episode, self.confidence_bins, self.resource_bins) <= 0:
            raise ValueError("RL counts must be positive")
        if not 0.0 < self.learning_rate <= 1.0 or not 0.0 <= self.discount <= 1.0:
            raise ValueError("RL learning rate/discount is invalid")


@dataclass
class IndependentBeliefQPolicy:
    q_values: np.ndarray
    config: IndependentRLConfig
    tactic_count: int
    action_count: int
    training_episode_ids: tuple[str, ...]
    shield_interventions: int
    rewards_seen: int
    action_resource_costs: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    switching_costs: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float)
    )
    reward_positive_scale: float = 1.0
    reward_transform: str = (
        "(physical_payoff_minus_physical_switching_cost)/positive_scale"
    )

    def state_index(self, belief: np.ndarray, resource_fraction: float, previous_action: int) -> int:
        probability = np.asarray(belief, dtype=float).reshape(-1)
        leader = int(np.argmax(probability))
        confidence = min(self.config.confidence_bins - 1, int(float(np.max(probability)) * self.config.confidence_bins))
        resource = min(self.config.resource_bins - 1, max(0, int(float(np.clip(resource_fraction, 0.0, 0.999999)) * self.config.resource_bins)))
        previous = self.action_count if previous_action < 0 else int(previous_action)
        return (((leader * self.config.confidence_bins + confidence) * self.config.resource_bins + resource) * (self.action_count + 1) + previous)

    def select(self, belief: np.ndarray, resource_fraction: float, previous_action: int, safety_mask: np.ndarray) -> int:
        mask = np.asarray(safety_mask, dtype=bool)
        if mask.shape != (self.action_count,) or not np.any(mask):
            raise ValueError("RL policy requires at least one safe action")
        if self.action_resource_costs.size:
            if self.action_resource_costs.shape != (self.action_count,):
                raise ValueError("RL policy resource-cost vector is invalid")
            mask = mask & (
                self.action_resource_costs <= float(resource_fraction) + 1.0e-12
            )
        if not np.any(mask):
            raise ValueError("RL policy has no resource-feasible action")
        index = self.state_index(belief, resource_fraction, previous_action)
        values = np.where(mask, self.q_values[index], -math.inf)
        return int(np.argmax(values))


def train_independent_belief_q_policy(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    *,
    safety_mask_fn: Callable[[float, int], np.ndarray] | None = None,
    config: IndependentRLConfig | None = None,
    training_episode_ids: Sequence[str] | None = None,
    action_resource_costs: np.ndarray | None = None,
    initial_resource_fraction: float = 1.0,
    switching_costs: np.ndarray | None = None,
) -> IndependentBeliefQPolicy:
    """Train only from simulated transitions/rewards, never model-controller labels."""

    cfg = config or IndependentRLConfig()
    cfg.validate()
    matrix = np.asarray(payoff, dtype=float)
    model.validate(matrix.shape[0])
    tactic_count = matrix.shape[1]
    action_count = matrix.shape[0]
    costs = (
        0.015
        + 0.025 * np.arange(action_count, dtype=float) / max(action_count - 1, 1)
        if action_resource_costs is None
        else np.asarray(action_resource_costs, dtype=float).reshape(-1)
    )
    if (
        costs.shape != (action_count,)
        or np.any(~np.isfinite(costs))
        or np.any(costs < 0.0)
    ):
        raise ValueError("RL action-resource costs are invalid")
    switch = (
        np.asarray(
            [
                [0.04 if previous < action_count and previous != action else 0.0 for action in range(action_count)]
                for previous in range(action_count + 1)
            ],
            dtype=float,
        )
        if switching_costs is None
        else np.asarray(switching_costs, dtype=float)
    )
    if (
        switch.shape != (action_count + 1, action_count)
        or np.any(~np.isfinite(switch))
        or np.any(switch < 0.0)
    ):
        raise ValueError("RL switching-cost matrix is invalid")
    resource0 = float(initial_resource_fraction)
    if not math.isfinite(resource0) or resource0 < 0.0:
        raise ValueError("RL initial resource fraction is invalid")
    state_count = tactic_count * cfg.confidence_bins * cfg.resource_bins * (action_count + 1)
    q = np.zeros((state_count, action_count), dtype=float)
    rng = np.random.default_rng(cfg.seed)
    dummy = IndependentBeliefQPolicy(q, cfg, tactic_count, action_count, (), 0, 0)
    # A positive multiplicative scale preserves the finite-horizon control
    # objective. Per-step centering does not when actions change the resource-
    # limited stopping time, and switching costs must remain in the same units
    # as the physical payoff before scaling.
    reward_scale = max(
        float(np.max(np.abs(matrix))),
        float(np.std(matrix)),
        1.0e-8,
    )
    interventions = 0
    rewards_seen = 0
    identities = tuple(training_episode_ids or (f"rl_train_{index:05d}" for index in range(cfg.episodes)))
    for episode in range(cfg.episodes):
        belief = np.full(tactic_count, 1.0 / tactic_count)
        tactic = int(rng.choice(tactic_count, p=belief))
        resource = resource0
        previous = -1
        epsilon = cfg.final_epsilon + (cfg.initial_epsilon - cfg.final_epsilon) * (1.0 - episode / max(cfg.episodes - 1, 1))
        for _ in range(cfg.steps_per_episode):
            state = dummy.state_index(belief, resource, previous)
            mask = np.ones(action_count, dtype=bool) if safety_mask_fn is None else np.asarray(safety_mask_fn(resource, previous), dtype=bool)
            if mask.shape != (action_count,):
                raise ValueError("RL safety mask is invalid")
            mask = mask & (costs <= resource + 1.0e-12)
            if not np.any(mask):
                break
            proposed = int(rng.integers(action_count)) if rng.random() < epsilon else int(np.argmax(q[state]))
            action = proposed
            if not mask[action]:
                interventions += 1
                action = int(np.flatnonzero(mask)[np.argmax(q[state, mask])])
            previous_index = action_count if previous < 0 else previous
            switching = float(switch[previous_index, action])
            consumption = float(costs[action])
            reward = (
                float(matrix[action, tactic]) - switching
            ) / reward_scale
            next_resource = max(0.0, resource - consumption)
            next_tactic = int(rng.choice(tactic_count, p=model.transition_by_radar_action[action, tactic]))
            observation = int(rng.choice(len(model.observation_ids), p=model.observation_likelihood[next_tactic]))
            next_belief = update_tactic_belief(belief, model, action, observation)
            next_state = dummy.state_index(next_belief, next_resource, action)
            next_mask = np.ones(action_count, dtype=bool) if safety_mask_fn is None else np.asarray(safety_mask_fn(next_resource, action), dtype=bool)
            if next_mask.shape != (action_count,):
                raise ValueError("RL next safety mask is invalid")
            next_mask = next_mask & (costs <= next_resource + 1.0e-12)
            target = reward
            if np.any(next_mask):
                target += cfg.discount * float(np.max(q[next_state, next_mask]))
            q[state, action] += cfg.learning_rate * (target - q[state, action])
            belief, tactic, resource, previous = next_belief, next_tactic, next_resource, action
            rewards_seen += 1
    return IndependentBeliefQPolicy(
        q,
        cfg,
        tactic_count,
        action_count,
        identities,
        interventions,
        rewards_seen,
        action_resource_costs=costs,
        switching_costs=switch,
        reward_positive_scale=reward_scale,
    )


def evaluate_independent_policy(
    policy: IndependentBeliefQPolicy,
    payoff: np.ndarray,
    model: HiddenTacticModel,
    *,
    episode_ids: Sequence[str],
    steps: int = 20,
    seed: int = 6601,
    safety_mask_fn: Callable[[float, int], np.ndarray] | None = None,
) -> dict[str, object]:
    if set(episode_ids).intersection(policy.training_episode_ids):
        raise ValueError("RL evaluation episodes overlap training episodes")
    matrix = np.asarray(payoff, dtype=float)
    model.validate(matrix.shape[0])
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    regrets: list[float] = []
    interventions = 0
    for _episode_id in episode_ids:
        belief = np.full(policy.tactic_count, 1.0 / policy.tactic_count)
        tactic = int(rng.choice(policy.tactic_count, p=belief))
        resource = 1.0
        previous = -1
        for _ in range(int(steps)):
            mask = np.ones(policy.action_count, dtype=bool) if safety_mask_fn is None else np.asarray(safety_mask_fn(resource, previous), dtype=bool)
            if policy.action_resource_costs.size:
                mask = mask & (
                    policy.action_resource_costs <= resource + 1.0e-12
                )
            if not np.any(mask):
                break
            proposed = int(np.argmax(policy.q_values[policy.state_index(belief, resource, previous)]))
            if not mask[proposed]:
                interventions += 1
            action = policy.select(belief, resource, previous, mask)
            previous_index = policy.action_count if previous < 0 else previous
            switching = (
                float(policy.switching_costs[previous_index, action])
                if policy.switching_costs.size
                else 0.0
            )
            reward = float(matrix[action, tactic]) - switching
            safe_indices = np.flatnonzero(mask)
            oracle = float(
                np.max(
                    matrix[safe_indices, tactic]
                    - (
                        policy.switching_costs[previous_index, safe_indices]
                        if policy.switching_costs.size
                        else 0.0
                    )
                )
            )
            rewards.append(reward)
            regrets.append(oracle - reward)
            consumption = (
                float(policy.action_resource_costs[action])
                if policy.action_resource_costs.size
                else 0.015
                + 0.025 * action / max(policy.action_count - 1, 1)
            )
            resource = max(0.0, resource - consumption)
            next_tactic = int(rng.choice(policy.tactic_count, p=model.transition_by_radar_action[action, tactic]))
            observation = int(rng.choice(len(model.observation_ids), p=model.observation_likelihood[next_tactic]))
            belief = update_tactic_belief(belief, model, action, observation)
            previous, tactic = action, next_tactic
    values = np.asarray(rewards)
    return {
        "episode_count": len(tuple(episode_ids)),
        "sample_count": int(values.size),
        "mean_utility": float(np.mean(values)),
        "cvar_10pct_utility": lower_tail_mean(values, 0.10),
        "mean_clairvoyant_regret": float(np.mean(regrets)),
        "safety_interventions": int(interventions),
        "safety_intervention_rate": float(interventions / max(values.size, 1)),
        "training_episode_overlap": False,
        "sample_efficiency_rewards_seen": int(policy.rewards_seen),
    }


def action_reactivity_report(radar_actions: Sequence[int], adversary_modes: Sequence[int]) -> dict[str, object]:
    """Likelihood-ratio/conditional-distribution evidence for action reactivity."""

    actions = np.asarray(radar_actions, dtype=int).reshape(-1)
    modes = np.asarray(adversary_modes, dtype=int).reshape(-1)
    if actions.size != modes.size or actions.size < 2:
        raise ValueError("reactivity sequences must have equal nontrivial length")
    action_values = sorted(set(actions.tolist()))
    mode_values = sorted(set(modes.tolist()))
    table = np.ones((len(action_values), len(mode_values)), dtype=float)  # finite Dirichlet smoothing
    a_index = {value: index for index, value in enumerate(action_values)}
    m_index = {value: index for index, value in enumerate(mode_values)}
    for action, mode in zip(actions, modes):
        table[a_index[int(action)], m_index[int(mode)]] += 1.0
    conditional = table / np.sum(table, axis=1, keepdims=True)
    marginal = np.sum(table, axis=0) / np.sum(table)
    action_weight = np.sum(table, axis=1) / np.sum(table)
    mutual_information = 0.0
    for i in range(conditional.shape[0]):
        mutual_information += float(action_weight[i] * np.sum(conditional[i] * np.log(conditional[i] / marginal)))
    maximum_pairwise_l1 = max(
        (float(np.sum(np.abs(conditional[i] - conditional[j]))) for i in range(len(action_values)) for j in range(i + 1, len(action_values))),
        default=0.0,
    )
    return {
        "conditional_adversary_distribution_by_radar_action": conditional.tolist(),
        "action_values": action_values,
        "mode_values": mode_values,
        "mutual_information_nats": float(max(mutual_information, 0.0)),
        "maximum_pairwise_conditional_l1": maximum_pairwise_l1,
        "action_reactive": bool(mutual_information > 1e-3 and maximum_pairwise_l1 > 0.05),
    }


def game_baseline_report(payoff: np.ndarray, model: HiddenTacticModel, belief: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(payoff, dtype=float)
    minimax = solve_zero_sum_schedule_game(matrix)
    bayesian = exact_belief_control(matrix, model, belief, horizon=3)
    stackelberg = stackelberg_leader_solution(matrix)
    robust = distributionally_robust_scores(matrix, belief)
    return {
        "stage": "v65_partially_observed_action_reactive_stochastic_game",
        "static_simultaneous_minimax": minimax.to_dict(),
        "bayesian_exact_small_world_belief_control": bayesian,
        "stackelberg_leader_follower": stackelberg,
        "distributionally_robust_cvar": robust,
        "model_contract": {
            "hidden_state": "semi_Markov_threat_mode_and_duration",
            "radar_action": "bounded_physically_feasible_schedule",
            "adversary_action": "jammer_or_deception_tactic_with_resource_and_latency",
            "observation": "noisy_threat_indicator_and_measurement_quality",
            "static_minimax_is_history_free_baseline": True,
            "clairvoyance_not_available_to_operational_policy": True,
        },
    }
