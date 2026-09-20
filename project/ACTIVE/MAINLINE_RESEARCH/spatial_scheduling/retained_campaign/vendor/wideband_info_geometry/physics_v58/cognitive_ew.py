from __future__ import annotations

"""Partially observed radar-jammer games and learning baselines."""

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from .information import DecisionPolicy, evaluate_schedule
from .optimization import solve_zero_sum_schedule_game
from .schema import Jammer, PhysicalWorld


@dataclass(frozen=True)
class HiddenTacticModel:
    tactic_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    transition_by_radar_action: np.ndarray
    observation_likelihood: np.ndarray

    def validate(self, radar_action_count: int) -> None:
        tactic_count = len(self.tactic_ids)
        observation_count = len(self.observation_ids)
        transition = np.asarray(self.transition_by_radar_action, dtype=float)
        likelihood = np.asarray(self.observation_likelihood, dtype=float)
        if transition.shape != (radar_action_count, tactic_count, tactic_count):
            raise ValueError("tactic transition must be radar-action by current-state by next-state")
        if likelihood.shape != (tactic_count, observation_count):
            raise ValueError("observation likelihood shape is inconsistent")
        if np.any(transition < 0.0) or np.any(likelihood < 0.0):
            raise ValueError("game probabilities must be nonnegative")
        if not np.allclose(np.sum(transition, axis=-1), 1.0, atol=1e-10):
            raise ValueError("every tactic-transition row must sum to one")
        if not np.allclose(np.sum(likelihood, axis=-1), 1.0, atol=1e-10):
            raise ValueError("every observation row must sum to one")


def tactic_identifier(jammer: Jammer | None) -> str:
    return "nominal" if jammer is None else f"{jammer.kind}:{jammer.jammer_id}"


def schedule_tactic_payoff_matrix(
    world: PhysicalWorld,
    candidate_action_ids: Sequence[Sequence[int]],
    tactics: Sequence[Jammer | None],
    *,
    policy: DecisionPolicy | None = None,
) -> np.ndarray:
    decision_policy = policy or DecisionPolicy()
    matrix = np.empty((len(candidate_action_ids), len(tactics)), dtype=float)
    for candidate_index, action_ids in enumerate(candidate_action_ids):
        # The same likelihood factors and resource cost apply to the matched
        # scenario vector.  Evaluating all tactics once is exactly equivalent
        # to one single-tactic call per column because a one-scenario CVaR/mean
        # objective reduces to that scenario information gain minus the common
        # decision cost.  This removes redundant factor construction without
        # changing the finite game.
        evaluation = evaluate_schedule(
            world,
            action_ids,
            adversary_scenarios=tactics,
            policy=decision_policy,
        )
        scenario = np.asarray(evaluation.scenario_information_gains, dtype=float)
        if scenario.shape != (len(tactics),):
            raise ValueError("scenario payoff dimension disagrees with tactics")
        matrix[candidate_index] = scenario - float(evaluation.decision_cost)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("radar-jammer payoff contains non-finite utilities")
    return matrix


def _observation_profile(tactic_id: str) -> np.ndarray:
    # quiet, broadband, narrowband, coherent-deception
    lowered = tactic_id.lower()
    if "nominal" in lowered:
        return np.asarray([0.82, 0.06, 0.08, 0.04])
    if "barrage" in lowered or "swept" in lowered:
        return np.asarray([0.05, 0.76, 0.15, 0.04])
    if "spot" in lowered:
        return np.asarray([0.05, 0.14, 0.76, 0.05])
    if "drfm" in lowered or "false" in lowered or "decept" in lowered:
        return np.asarray([0.04, 0.08, 0.13, 0.75])
    return np.full(4, 0.25)


def build_hidden_tactic_model(
    payoff: np.ndarray,
    tactic_ids: Sequence[str],
    *,
    persistence: float = 0.72,
    best_response_strength: float = 0.22,
    observation_quality: float = 0.88,
) -> HiddenTacticModel:
    matrix = np.asarray(payoff, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(tactic_ids):
        raise ValueError("payoff and tactic identifiers are inconsistent")
    if not 0.0 <= persistence <= 1.0 or not 0.0 <= best_response_strength <= 1.0:
        raise ValueError("invalid jammer persistence or response strength")
    if not 0.0 <= observation_quality <= 1.0:
        raise ValueError("observation quality must be in [0,1]")
    tactic_count = len(tactic_ids)
    base = np.full((tactic_count, tactic_count), (1.0 - persistence) / tactic_count)
    base += persistence * np.eye(tactic_count)
    transition = np.empty((matrix.shape[0], tactic_count, tactic_count), dtype=float)
    for action_index in range(matrix.shape[0]):
        best_response = int(np.argmin(matrix[action_index]))
        response = np.zeros(tactic_count, dtype=float)
        response[best_response] = 1.0
        transition[action_index] = (1.0 - best_response_strength) * base + best_response_strength * response[None, :]
    canonical = np.vstack([_observation_profile(str(tactic_id)) for tactic_id in tactic_ids])
    observation = observation_quality * canonical + (1.0 - observation_quality) * np.full_like(canonical, 0.25)
    observation /= np.sum(observation, axis=1, keepdims=True)
    model = HiddenTacticModel(
        tuple(str(value) for value in tactic_ids),
        ("quiet", "broadband_power", "narrowband_power", "coherent_deception"),
        transition,
        observation,
    )
    model.validate(matrix.shape[0])
    return model


def predict_tactic_belief(belief: np.ndarray, model: HiddenTacticModel, radar_action: int) -> np.ndarray:
    prior = np.asarray(belief, dtype=float).reshape(-1)
    if np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0):
        raise ValueError("tactic belief must be a probability vector")
    return prior @ model.transition_by_radar_action[int(radar_action)]


def update_tactic_belief(
    belief: np.ndarray,
    model: HiddenTacticModel,
    radar_action: int,
    observation_index: int,
) -> np.ndarray:
    prediction = predict_tactic_belief(belief, model, radar_action)
    posterior = prediction * model.observation_likelihood[:, int(observation_index)]
    normalizer = float(np.sum(posterior))
    if normalizer <= 0.0:
        raise ValueError("observation has zero probability under the tactic model")
    return posterior / normalizer


def observation_distribution(belief: np.ndarray, model: HiddenTacticModel, radar_action: int) -> np.ndarray:
    prediction = predict_tactic_belief(belief, model, radar_action)
    return prediction @ model.observation_likelihood


def finite_horizon_belief_policy(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    belief: np.ndarray,
    *,
    horizon: int = 2,
    discount: float = 0.95,
) -> dict[str, object]:
    matrix = np.asarray(payoff, dtype=float)
    model.validate(matrix.shape[0])
    initial = np.asarray(belief, dtype=float).reshape(-1)
    if horizon < 1 or not 0.0 <= discount <= 1.0:
        raise ValueError("invalid finite-horizon game settings")

    def value(current: np.ndarray, remaining: int) -> tuple[float, int, np.ndarray]:
        q_values = np.empty(matrix.shape[0], dtype=float)
        for action in range(matrix.shape[0]):
            immediate = float(current @ matrix[action])
            future = 0.0
            if remaining > 1:
                probabilities = observation_distribution(current, model, action)
                for observation_index, probability in enumerate(probabilities):
                    if probability <= 0.0:
                        continue
                    posterior = update_tactic_belief(current, model, action, observation_index)
                    next_value, _, _ = value(posterior, remaining - 1)
                    future += float(probability) * next_value
            q_values[action] = immediate + discount * future
        selected = int(np.argmax(q_values))
        return float(q_values[selected]), selected, q_values

    total, selected, q_values = value(initial, int(horizon))
    myopic_values = matrix @ initial
    myopic_action = int(np.argmax(myopic_values))
    no_current_observation_q = np.empty(matrix.shape[0], dtype=float)
    for action in range(matrix.shape[0]):
        immediate = float(initial @ matrix[action])
        if horizon > 1:
            prediction = predict_tactic_belief(initial, model, action)
            future_without_current_observation, _, _ = value(prediction, int(horizon) - 1)
        else:
            future_without_current_observation = 0.0
        no_current_observation_q[action] = immediate + discount * future_without_current_observation
    value_of_current_observation = max(0.0, float(total - np.max(no_current_observation_q)))
    return {
        "selected_action": selected,
        "finite_horizon_value": total,
        "q_values": [float(value) for value in q_values],
        "myopic_action": myopic_action,
        "myopic_expected_utility": float(myopic_values[myopic_action]),
        "finite_horizon_advantage_over_myopic_action": float(total - q_values[myopic_action]),
        "no_current_observation_q_values": [float(value) for value in no_current_observation_q],
        "value_of_current_observation": value_of_current_observation,
        "horizon": int(horizon),
        "discount": float(discount),
    }


def _belief_state_index(belief: np.ndarray, tactic_count: int, confidence_bins: int) -> int:
    leader = int(np.argmax(belief))
    confidence = float(np.max(belief))
    level = min(confidence_bins - 1, int(confidence * confidence_bins))
    return leader * confidence_bins + level


def train_belief_q_policy(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    *,
    episodes: int = 1_200,
    steps_per_episode: int = 20,
    confidence_bins: int = 4,
    learning_rate: float = 0.12,
    discount: float = 0.92,
    seed: int = 0,
) -> dict[str, object]:
    matrix = np.asarray(payoff, dtype=float)
    model.validate(matrix.shape[0])
    rng = np.random.default_rng(seed)
    tactic_count = matrix.shape[1]
    q = np.zeros((tactic_count * confidence_bins, matrix.shape[0]), dtype=float)
    reward_center = float(np.mean(matrix))
    reward_scale = max(float(np.std(matrix)), 1e-6)
    for episode in range(int(episodes)):
        belief = np.full(tactic_count, 1.0 / tactic_count)
        tactic = int(rng.choice(tactic_count, p=belief))
        epsilon = max(0.04, 0.45 * (1.0 - episode / max(episodes, 1)))
        for _ in range(int(steps_per_episode)):
            state = _belief_state_index(belief, tactic_count, confidence_bins)
            action = int(rng.integers(matrix.shape[0])) if rng.random() < epsilon else int(np.argmax(q[state]))
            reward = (float(matrix[action, tactic]) - reward_center) / reward_scale
            next_tactic = int(rng.choice(tactic_count, p=model.transition_by_radar_action[action, tactic]))
            observation = int(rng.choice(len(model.observation_ids), p=model.observation_likelihood[next_tactic]))
            next_belief = update_tactic_belief(belief, model, action, observation)
            next_state = _belief_state_index(next_belief, tactic_count, confidence_bins)
            target = reward + discount * float(np.max(q[next_state]))
            q[state, action] += learning_rate * (target - q[state, action])
            tactic = next_tactic
            belief = next_belief
    return {
        "q_values": q,
        "confidence_bins": int(confidence_bins),
        "episodes": int(episodes),
        "steps_per_episode": int(steps_per_episode),
        "reward_center": reward_center,
        "reward_scale": reward_scale,
    }


def simulate_hidden_tactic_policy(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    policy: Callable[[np.ndarray], int],
    *,
    episodes: int = 200,
    steps: int = 20,
    seed: int = 0,
) -> dict[str, float | int]:
    matrix = np.asarray(payoff, dtype=float)
    model.validate(matrix.shape[0])
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    belief_entropies: list[float] = []
    for _ in range(int(episodes)):
        belief = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
        tactic = int(rng.choice(matrix.shape[1], p=belief))
        for _ in range(int(steps)):
            action = int(policy(belief.copy()))
            rewards.append(float(matrix[action, tactic]))
            next_tactic = int(rng.choice(matrix.shape[1], p=model.transition_by_radar_action[action, tactic]))
            observation = int(rng.choice(len(model.observation_ids), p=model.observation_likelihood[next_tactic]))
            belief = update_tactic_belief(belief, model, action, observation)
            positive = belief[belief > 0.0]
            belief_entropies.append(float(-np.sum(positive * np.log(positive))))
            tactic = next_tactic
    array = np.asarray(rewards, dtype=float)
    return {
        "sample_count": int(array.size),
        "mean_utility": float(np.mean(array)),
        "lower_tail_10pct_utility": float(np.mean(np.sort(array)[: max(1, int(math.ceil(0.1 * array.size)))])),
        "utility_std": float(np.std(array)),
        "mean_tactic_belief_entropy_nats": float(np.mean(belief_entropies)),
    }


def cognitive_ew_game_report(
    payoff: np.ndarray,
    tactic_ids: Sequence[str],
    *,
    seed: int = 0,
    q_episodes: int = 800,
) -> dict[str, object]:
    matrix = np.asarray(payoff, dtype=float)
    model = build_hidden_tactic_model(matrix, tactic_ids)
    initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    matrix_game = solve_zero_sum_schedule_game(matrix)
    horizon_policy = finite_horizon_belief_policy(matrix, model, initial, horizon=2)
    q_result = train_belief_q_policy(matrix, model, episodes=q_episodes, seed=seed)
    q_values = np.asarray(q_result.pop("q_values"), dtype=float)
    deterministic_maximin_action = int(np.argmax(np.min(matrix, axis=1)))
    myopic = lambda belief: int(np.argmax(matrix @ belief))
    lookahead = lambda belief: int(finite_horizon_belief_policy(matrix, model, belief, horizon=2)["selected_action"])
    q_policy = lambda belief: int(np.argmax(q_values[_belief_state_index(belief, matrix.shape[1], int(q_result["confidence_bins"]))]))
    static_policy = lambda belief: deterministic_maximin_action
    random_rng = np.random.default_rng(seed + 99)
    random_policy = lambda belief: int(random_rng.integers(matrix.shape[0]))
    simulations = {
        "belief_lookahead": simulate_hidden_tactic_policy(matrix, model, lookahead, seed=seed + 1),
        "belief_myopic": simulate_hidden_tactic_policy(matrix, model, myopic, seed=seed + 1),
        "q_learning": simulate_hidden_tactic_policy(matrix, model, q_policy, seed=seed + 1),
        "static_maximin": simulate_hidden_tactic_policy(matrix, model, static_policy, seed=seed + 1),
        "random": simulate_hidden_tactic_policy(matrix, model, random_policy, seed=seed + 1),
    }
    lookahead_mean = float(simulations["belief_lookahead"]["mean_utility"])
    static_mean = float(simulations["static_maximin"]["mean_utility"])
    random_mean = float(simulations["random"]["mean_utility"])
    q_mean = float(simulations["q_learning"]["mean_utility"])
    posterior_checks = []
    for observation_index in range(len(model.observation_ids)):
        posterior = update_tactic_belief(initial, model, 0, observation_index)
        posterior_checks.append(float(np.sum(posterior)))
    return {
        "stage": "v64_partially_observed_radar_jammer_game",
        "status": "passed" if np.allclose(posterior_checks, 1.0, atol=1e-12) and matrix_game.game_value + 1e-8 >= matrix_game.deterministic_maximin_value else "failed_closed",
        "payoff_shape": list(matrix.shape),
        "tactic_ids": list(model.tactic_ids),
        "observation_ids": list(model.observation_ids),
        "static_zero_sum_game": matrix_game.to_dict(),
        "initial_belief_policy": horizon_policy,
        "learning_configuration": q_result,
        "policy_simulations": simulations,
        "policy_effects": {
            "belief_lookahead_minus_static_maximin_mean_utility": lookahead_mean - static_mean,
            "belief_lookahead_minus_random_mean_utility": lookahead_mean - random_mean,
            "q_learning_minus_static_maximin_mean_utility": q_mean - static_mean,
            "belief_lookahead_outperforms_static_in_registered_simulation": lookahead_mean > static_mean,
            "Q_learning_outperforms_static_in_registered_simulation": q_mean > static_mean,
        },
        "belief_normalization_max_abs_residual": float(max(abs(value - 1.0) for value in posterior_checks)),
        "theory_guidance": {
            "adversarial_worst_case": "use the zero-sum mixed strategy when tactic dynamics are not learnable",
            "predictable_hidden_tactic": "use Bayesian belief control when observations carry stable tactic information",
            "learning_role": "Q-learning approximates the belief policy but does not replace the exact small-game benchmark",
        },
        "claim_boundary": {
            "finite_candidate_pool_game": True,
            "universal_EW_equilibrium": False,
            "RL_policy_mature": False,
            "adversary_model_is_hypothesis_not_ground_truth": True,
        },
    }
