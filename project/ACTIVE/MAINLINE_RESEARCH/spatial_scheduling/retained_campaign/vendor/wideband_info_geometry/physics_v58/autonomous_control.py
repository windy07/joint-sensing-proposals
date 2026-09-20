from __future__ import annotations

"""Receding-horizon autonomous radar control under changing threats."""

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Any, Literal, Sequence

import numpy as np

from .cognitive_ew import HiddenTacticModel, build_hidden_tactic_model
from .information import (
    DecisionPolicy,
    decision_cost_for_actions,
    propagate_normalized_covariance,
    schedule_feasibility,
    schedule_resource_usage,
    stable_inverse_spd,
)
from .measurement import StateLayout, build_action_factor, world_propagated_to_time
from .schema import Jammer, PhysicalWorld, RadarAction


@dataclass
class ResourceLedger:
    energy_j: float
    emission_j: float
    communication_mb: float

    def can_execute(self, world: PhysicalWorld, actions: Sequence[RadarAction], reserve_fraction: float = 0.0) -> bool:
        usage = schedule_resource_usage(world, actions)
        reserve = float(np.clip(reserve_fraction, 0.0, 1.0))
        return (
            usage.energy_j <= self.energy_j * (1.0 - reserve) + 1e-12
            and usage.emission_exposure_j <= self.emission_j * (1.0 - reserve) + 1e-12
            and usage.communication_mb <= self.communication_mb * (1.0 - reserve) + 1e-12
        )

    def execute(self, world: PhysicalWorld, actions: Sequence[RadarAction]) -> dict[str, float]:
        usage = schedule_resource_usage(world, actions)
        if not self.can_execute(world, actions):
            raise ValueError("resource ledger rejected an infeasible autonomous action")
        self.energy_j -= usage.energy_j
        self.emission_j -= usage.emission_exposure_j
        self.communication_mb -= usage.communication_mb
        return {
            "energy_j": float(usage.energy_j),
            "emission_j": float(usage.emission_exposure_j),
            "communication_mb": float(usage.communication_mb),
            "latency_s": float(usage.decision_latency_s),
        }

    def preserves_backup_invariant(
        self,
        world: PhysicalWorld,
        actions: Sequence[RadarAction],
        backup_actions: Sequence[RadarAction],
        remaining_epochs: int,
    ) -> bool:
        """Keep enough campaign resources for a feasible low-cost backup policy."""

        current = schedule_resource_usage(world, actions)
        backup = schedule_resource_usage(world, backup_actions)
        epochs = max(0, int(remaining_epochs))
        return (
            current.energy_j + epochs * backup.energy_j <= self.energy_j + 1e-12
            and current.emission_exposure_j + epochs * backup.emission_exposure_j <= self.emission_j + 1e-12
            and current.communication_mb + epochs * backup.communication_mb <= self.communication_mb + 1e-12
        )


@dataclass
class ThreatMemory:
    transition_counts: np.ndarray
    previous_belief: np.ndarray
    cusum: float = 0.0
    alarm_count: int = 0
    maximum_cusum: float = 0.0
    confirmed_tactic: int | None = None
    candidate_tactic: int | None = None
    candidate_run_length: int = 0
    refractory_epochs_remaining: int = 0
    confirmed_probability_collapse_run_length: int = 0

    @classmethod
    def initialize(
        cls,
        tactic_count: int,
        prior_count: float = 1.0,
        *,
        initial_belief: np.ndarray | None = None,
        confirmed_tactic: int | None = None,
    ) -> "ThreatMemory":
        belief = (
            np.full(tactic_count, 1.0 / tactic_count)
            if initial_belief is None
            else np.asarray(initial_belief, dtype=float).reshape(-1)
        )
        if belief.size != tactic_count or np.any(belief < 0.0) or not np.isclose(np.sum(belief), 1.0):
            raise ValueError("initial threat belief must be a probability vector")
        if confirmed_tactic is not None and not 0 <= int(confirmed_tactic) < tactic_count:
            raise ValueError("confirmed tactic index is out of range")
        return cls(
            np.full((tactic_count, tactic_count), float(prior_count)),
            belief.copy(),
            confirmed_tactic=None if confirmed_tactic is None else int(confirmed_tactic),
        )

    def learned_transition(self) -> np.ndarray:
        return self.transition_counts / np.sum(self.transition_counts, axis=1, keepdims=True)

    def update(
        self,
        posterior: np.ndarray,
        observation_probability: float,
        expected_surprise: float,
        *,
        forgetting: float = 0.985,
        cusum_slack: float = 0.12,
        cusum_threshold: float = 1.40,
        minimum_mode_dwell: int = 2,
        mode_confidence_threshold: float = 0.38,
        alarm_refractory_epochs: int = 3,
        confirmed_probability_floor: float = 0.25,
        collapse_minimum_dwell: int = 2,
    ) -> tuple[bool, float, str]:
        """Update long-memory transition learning and a calibrated Page statistic.

        The increment is the realized surprisal minus its one-step predictive
        expectation.  Unlike a raw ``-log p(y)`` threshold, this has zero mean
        under a calibrated stationary observation model before the positive
        Page reset and therefore does not confuse ordinary observation entropy
        with a regime change.
        """

        posterior = np.asarray(posterior, dtype=float)
        self.transition_counts *= float(forgetting)
        self.transition_counts += np.outer(self.previous_belief, posterior)
        surprise = -math.log(max(float(observation_probability), 1e-12))
        innovation = surprise - float(expected_surprise) - float(cusum_slack)
        self.cusum = max(0.0, self.cusum + innovation)
        self.maximum_cusum = max(self.maximum_cusum, self.cusum)
        in_refractory = self.refractory_epochs_remaining > 0
        if in_refractory:
            self.refractory_epochs_remaining -= 1
            self.cusum = 0.0
        page_evidence = (not in_refractory) and self.cusum >= float(cusum_threshold)
        map_tactic = int(np.argmax(posterior))
        if map_tactic == self.candidate_tactic:
            self.candidate_run_length += 1
        else:
            self.candidate_tactic = map_tactic
            self.candidate_run_length = 1
        mode_alarm = False
        if (
            self.candidate_run_length >= int(minimum_mode_dwell)
            and float(posterior[map_tactic]) >= float(mode_confidence_threshold)
        ):
            if self.confirmed_tactic is None:
                self.confirmed_tactic = map_tactic
            elif map_tactic != self.confirmed_tactic:
                if not in_refractory:
                    mode_alarm = True
                    self.confirmed_tactic = map_tactic
        collapse_alarm = False
        if self.confirmed_tactic is not None:
            if float(posterior[int(self.confirmed_tactic)]) < float(confirmed_probability_floor):
                self.confirmed_probability_collapse_run_length += 1
            else:
                self.confirmed_probability_collapse_run_length = 0
            collapse_alarm = (
                not in_refractory
                and self.confirmed_probability_collapse_run_length >= int(collapse_minimum_dwell)
                and map_tactic != self.confirmed_tactic
            )
        # A Page excursion is evidence against the stationary null, but one
        # improbable observation is not yet a regime declaration.  Requiring
        # posterior persistence prevents a rare sample and the subsequent
        # Bayesian correction from becoming two change alarms.
        alarm = mode_alarm or collapse_alarm
        if alarm:
            self.confirmed_tactic = map_tactic
            self.candidate_tactic = map_tactic
            self.candidate_run_length = 1
            self.confirmed_probability_collapse_run_length = 0
            self.alarm_count += 1
            self.cusum = 0.0
            self.refractory_epochs_remaining = max(0, int(alarm_refractory_epochs))
        self.previous_belief = posterior.copy()
        reason = (
            "page_cusum_supported_persistent_hmm_mode_switch"
            if page_evidence and mode_alarm
            else (
                "persistent_hmm_mode_switch"
                if mode_alarm
                else (
                    "page_cusum_supported_confirmed_mode_posterior_collapse"
                    if page_evidence and collapse_alarm
                    else (
                        "confirmed_mode_posterior_collapse"
                        if collapse_alarm
                        else ("page_cusum_evidence_only" if page_evidence else "none")
                    )
                )
            )
        )
        return alarm, float(innovation), reason


def adaptive_threat_model(base: HiddenTacticModel, memory: ThreatMemory, blend: float = 0.35) -> HiddenTacticModel:
    learned = memory.learned_transition()
    transition = (1.0 - blend) * base.transition_by_radar_action + blend * learned[None, :, :]
    transition /= np.sum(transition, axis=-1, keepdims=True)
    return replace(base, transition_by_radar_action=transition)


def candidate_observation_likelihood(
    world: PhysicalWorld,
    action_ids: Sequence[int],
    base: HiddenTacticModel,
) -> tuple[np.ndarray, float]:
    by_id = {action.action_id: action for action in world.actions}
    actions = [by_id[int(action_id)] for action_id in action_ids]
    passive = sum(action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"} for action in actions)
    band_count = len({world.waveform(action.waveform_id).carrier_hz for action in actions})
    reference_count = len({action.reference_node_id for action in actions if action.reference_node_id is not None})
    quality = float(np.clip(0.35 + 0.08 * passive + 0.06 * band_count + 0.05 * reference_count, 0.35, 0.96))
    uniform = np.full_like(base.observation_likelihood, 1.0 / len(base.observation_ids))
    likelihood = quality * base.observation_likelihood + (1.0 - quality) * uniform
    likelihood /= np.sum(likelihood, axis=1, keepdims=True)
    return likelihood, quality


def bayes_update_with_likelihood(
    belief: np.ndarray,
    transition: np.ndarray,
    likelihood: np.ndarray,
    observation: int,
) -> tuple[np.ndarray, float]:
    prediction = np.asarray(belief, dtype=float) @ np.asarray(transition, dtype=float)
    observation_probability = float(prediction @ likelihood[:, int(observation)])
    posterior = prediction * likelihood[:, int(observation)] / max(observation_probability, 1e-12)
    posterior /= max(float(np.sum(posterior)), 1e-12)
    return posterior, observation_probability


def tactic_information_gain(
    belief: np.ndarray,
    transition: np.ndarray,
    likelihood: np.ndarray,
) -> float:
    prediction = np.asarray(belief, dtype=float) @ np.asarray(transition, dtype=float)
    observation_probability = prediction @ likelihood
    positive_observation = observation_probability[observation_probability > 0.0]
    h_observation = -float(np.sum(positive_observation * np.log(positive_observation)))
    conditional = 0.0
    for probability, row in zip(prediction, likelihood):
        positive = row[row > 0.0]
        conditional += float(probability) * (-float(np.sum(positive * np.log(positive))))
    return max(0.0, h_observation - conditional)


def _physical_position_covariance(covariance: np.ndarray, layout: StateLayout) -> np.ndarray:
    scales = layout.scales()[:3]
    return np.diag(scales) @ covariance[:3, :3] @ np.diag(scales)


def _posterior_for_tactic(
    covariance: np.ndarray,
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    tactic: Jammer | None,
    factor_information_cache: dict[tuple[int, str], np.ndarray] | None = None,
) -> np.ndarray:
    information = stable_inverse_spd(covariance, name="autonomous_predicted_covariance")
    for action in actions:
        tactic_id = "nominal" if tactic is None else tactic.jammer_id
        key = (int(action.action_id), tactic_id)
        if factor_information_cache is not None and key in factor_information_cache:
            increment = factor_information_cache[key]
        else:
            increment = build_action_factor(world, action, tactic).observed_information_normalized
            if factor_information_cache is not None:
                factor_information_cache[key] = increment
        information += increment
    return stable_inverse_spd(information, name="autonomous_posterior_information")


def candidate_dynamic_statistics(
    covariance: np.ndarray,
    world: PhysicalWorld,
    action_ids: Sequence[int],
    tactics: Sequence[Jammer | None],
    belief: np.ndarray,
    *,
    policy: DecisionPolicy,
    factor_information_cache: dict[tuple[int, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    by_id = {action.action_id: action for action in world.actions}
    actions = [by_id[int(action_id)] for action_id in action_ids]
    feasible, violations = schedule_feasibility(world, actions)
    if not feasible:
        return {"feasible": False, "violations": list(violations)}
    prior_position = _physical_position_covariance(covariance, StateLayout.from_world(world))
    prior_sign, prior_logdet = np.linalg.slogdet(prior_position)
    if prior_sign <= 0.0:
        raise ValueError("autonomous prior position covariance must be positive definite")
    gains: list[float] = []
    posteriors: list[np.ndarray] = []
    traces: list[float] = []
    for tactic in tactics:
        posterior = _posterior_for_tactic(
            covariance,
            world,
            actions,
            tactic,
            factor_information_cache=factor_information_cache,
        )
        position = _physical_position_covariance(posterior, StateLayout.from_world(world))
        sign, logdet = np.linalg.slogdet(position)
        if sign <= 0.0:
            raise ValueError("autonomous posterior position covariance must be positive definite")
        gains.append(float(prior_logdet - logdet))
        traces.append(float(np.trace(position)))
        posteriors.append(posterior)
    gains_array = np.asarray(gains, dtype=float)
    probability = np.asarray(belief, dtype=float)
    expected_gain = float(probability @ gains_array)
    worst_gain = float(np.min(gains_array))
    robust_gain = (1.0 - policy.cvar_weight) * expected_gain + policy.cvar_weight * worst_gain
    expected_posterior = sum(float(weight) * posterior for weight, posterior in zip(probability, posteriors))
    return {
        "feasible": True,
        "violations": [],
        "scenario_information_gains": gains_array,
        "expected_information_gain": expected_gain,
        "worst_information_gain": worst_gain,
        "robust_information_gain": robust_gain,
        "expected_position_trace_m2": float(probability @ np.asarray(traces)),
        "expected_posterior_covariance": expected_posterior,
        "decision_cost": decision_cost_for_actions(world, actions, policy),
    }


def select_receding_horizon_action(
    covariance: np.ndarray,
    world: PhysicalWorld,
    candidates: Sequence[Sequence[int]],
    tactics: Sequence[Jammer | None],
    model: HiddenTacticModel,
    belief: np.ndarray,
    ledger: ResourceLedger,
    *,
    policy: DecisionPolicy | None = None,
    horizon: int = 2,
    discount: float = 0.90,
    dual_control_weight: float = 0.18,
    reserve_fraction: float = 0.02,
    change_alarm: bool = False,
    lookahead_width: int = 3,
    remaining_epochs: int = 0,
) -> dict[str, Any]:
    """Choose the first action of a newly solved finite-horizon problem."""

    decision_policy = policy or DecisionPolicy()
    if model.transition_by_radar_action.shape[0] != len(candidates):
        raise ValueError("threat model action dimension must match autonomous candidates")
    by_id = {action.action_id: action for action in world.actions}
    factor_information_cache: dict[tuple[int, str], np.ndarray] = {}
    feasible_backup_rows: list[tuple[float, int, list[RadarAction]]] = []
    for candidate_index, ids in enumerate(candidates):
        actions = [by_id[int(action_id)] for action_id in ids]
        feasible, _ = schedule_feasibility(world, actions)
        if not feasible:
            continue
        usage = schedule_resource_usage(world, actions)
        normalized_cost = (
            usage.energy_j / world.max_total_energy_j
            + usage.emission_exposure_j / world.max_total_emission_j
            + usage.communication_mb / world.max_total_communication_mb
        )
        feasible_backup_rows.append((float(normalized_cost), candidate_index, actions))
    if not feasible_backup_rows:
        raise RuntimeError("autonomous controller has no physically feasible backup candidate")
    _, backup_candidate_index, backup_actions = min(feasible_backup_rows, key=lambda row: (row[0], row[1]))

    def one_step(
        current_covariance: np.ndarray,
        current_belief: np.ndarray,
        risk_boost: float,
        candidate_indices: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        indices = range(len(candidates)) if candidate_indices is None else candidate_indices
        for candidate_index in indices:
            ids = candidates[int(candidate_index)]
            actions = [by_id[int(action_id)] for action_id in ids]
            if not ledger.can_execute(world, actions, reserve_fraction=reserve_fraction):
                rows.append({"candidate_index": candidate_index, "feasible": False, "reason": "campaign_resource_filter"})
                continue
            if not ledger.preserves_backup_invariant(
                world,
                actions,
                backup_actions,
                remaining_epochs=max(0, int(remaining_epochs)),
            ):
                rows.append({
                    "candidate_index": candidate_index,
                    "feasible": False,
                    "reason": "terminal_backup_resource_invariant",
                })
                continue
            statistics = candidate_dynamic_statistics(
                current_covariance,
                world,
                ids,
                tactics,
                current_belief,
                policy=decision_policy,
                factor_information_cache=factor_information_cache,
            )
            if not statistics["feasible"]:
                rows.append({"candidate_index": candidate_index, **statistics})
                continue
            likelihood, sensor_quality = candidate_observation_likelihood(world, ids, model)
            transition = model.transition_by_radar_action[candidate_index]
            information_gain = tactic_information_gain(current_belief, transition, likelihood)
            robust_weight = min(0.95, decision_policy.cvar_weight + risk_boost)
            control_gain = (
                (1.0 - robust_weight) * float(statistics["expected_information_gain"])
                + robust_weight * float(statistics["worst_information_gain"])
            )
            score = control_gain - float(statistics["decision_cost"]) + dual_control_weight * information_gain
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "action_ids": [int(value) for value in ids],
                    "feasible": True,
                    "score": float(score),
                    "control_gain": float(control_gain),
                    "tactic_information_gain_nats": float(information_gain),
                    "sensor_quality": sensor_quality,
                    **statistics,
                }
            )
        return rows

    risk_boost = 0.25 if change_alarm else 0.0
    rows = one_step(covariance, np.asarray(belief, dtype=float), risk_boost)
    feasible_rows = [row for row in rows if row.get("feasible")]
    if not feasible_rows:
        raise RuntimeError("autonomous controller has no recursively feasible candidate")
    if horizon > 1:
        expanded_rows = sorted(feasible_rows, key=lambda item: float(item["score"]), reverse=True)[: max(1, int(lookahead_width))]
        expanded_indices = [int(item["candidate_index"]) for item in expanded_rows]
        for row in expanded_rows:
            candidate_index = int(row["candidate_index"])
            transition = model.transition_by_radar_action[candidate_index]
            likelihood, _ = candidate_observation_likelihood(world, row["action_ids"], model)
            observation_probability = (np.asarray(belief) @ transition) @ likelihood
            future_value = 0.0
            for observation_index, probability in enumerate(observation_probability):
                if probability <= 0.0:
                    continue
                posterior_belief, _ = bayes_update_with_likelihood(
                    np.asarray(belief), transition, likelihood, observation_index
                )
                future_rows = one_step(
                    np.asarray(row["expected_posterior_covariance"]),
                    posterior_belief,
                    0.0,
                    expanded_indices,
                )
                future_scores = [float(item["score"]) for item in future_rows if item.get("feasible")]
                if future_scores:
                    future_value += float(probability) * max(future_scores)
            row["future_value"] = float(future_value)
            row["receding_horizon_score"] = float(row["score"] + discount * future_value)
        feasible_rows = expanded_rows
    else:
        for row in feasible_rows:
            row["future_value"] = 0.0
            row["receding_horizon_score"] = float(row["score"])
    selected = max(feasible_rows, key=lambda item: (float(item["receding_horizon_score"]), -int(item["candidate_index"])))
    return {
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_action_ids": list(selected["action_ids"]),
        "selected_score": float(selected["receding_horizon_score"]),
        "backup_candidate_index": int(backup_candidate_index),
        "terminal_backup_resource_invariant_satisfied": True,
        "change_alarm_risk_boost_applied": bool(change_alarm),
        "candidate_diagnostics": [
            {
                key: value
                for key, value in row.items()
                if key not in {"expected_posterior_covariance", "scenario_information_gains"}
            }
            for row in rows
        ],
        "controller_contract": "belief_space_receding_horizon_recomputed_each_epoch",
    }


def _initial_covariance(world: PhysicalWorld) -> np.ndarray:
    from .information import prior_information_normalized

    layout = StateLayout.from_world(world)
    return stable_inverse_spd(prior_information_normalized(world, layout), name="autonomous_initial_prior")


def _match_change_alarms(
    change_steps: Sequence[int],
    alarm_steps: Sequence[int],
    maximum_delay: int,
) -> tuple[list[int | None], list[int], list[int]]:
    """Maximum-cardinality one-to-one matching with minimum total delay."""

    changes = tuple(int(value) for value in change_steps)
    alarms = tuple(int(value) for value in alarm_steps)
    # Store alarm indices rather than alarm values so duplicate timestamps remain
    # distinguishable and unmatched-alarm accounting uses the same assignment.
    best_assignment: list[int | None] = [None] * len(changes)
    best_objective = (-1, math.inf, math.inf)

    def search(index: int, used: set[int], assignment: list[int | None]) -> None:
        nonlocal best_assignment, best_objective
        if index == len(changes):
            matched = sum(value is not None for value in assignment)
            total_delay = sum(
                alarms[int(value)] - changes[i]
                for i, value in enumerate(assignment)
                if value is not None
            )
            alarm_sum = sum(alarms[int(value)] for value in assignment if value is not None)
            objective = (matched, total_delay, alarm_sum)
            if (
                objective[0] > best_objective[0]
                or (objective[0] == best_objective[0] and objective[1] < best_objective[1])
                or (
                    objective[0] == best_objective[0]
                    and objective[1] == best_objective[1]
                    and objective[2] < best_objective[2]
                )
            ):
                best_objective = objective
                best_assignment = list(assignment)
            return
        search(index + 1, used, [*assignment, None])
        change = changes[index]
        for alarm_index, alarm in enumerate(alarms):
            if alarm_index in used or not change <= alarm <= change + int(maximum_delay):
                continue
            search(index + 1, used | {alarm_index}, [*assignment, alarm_index])

    search(0, set(), [])
    matched_indices = {int(value) for value in best_assignment if value is not None}
    matched = [alarms[int(value)] for value in best_assignment if value is not None]
    unmatched = [alarm for index, alarm in enumerate(alarms) if index not in matched_indices]
    delays = [
        None if alarm_index is None else int(alarms[int(alarm_index)] - changes[index])
        for index, alarm_index in enumerate(best_assignment)
    ]
    return delays, matched, unmatched


def _simulate_controller(
    world: PhysicalWorld,
    candidates: Sequence[Sequence[int]],
    tactics: Sequence[Jammer | None],
    true_tactic_path: Sequence[int],
    *,
    mode: Literal[
        "adaptive_receding_horizon",
        "fixed_initial_policy",
        "static_worst_case_policy",
        "clairvoyant_one_step_reference",
    ],
    seed: int,
    dt_s: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    layout = StateLayout.from_world(world)
    covariance = _initial_covariance(world)
    placeholder_payoff = np.zeros((len(candidates), len(tactics)), dtype=float)
    base_model = build_hidden_tactic_model(placeholder_payoff, ["nominal" if value is None else value.kind for value in tactics], best_response_strength=0.0)
    belief = np.full(len(tactics), 0.20 / max(len(tactics) - 1, 1))
    belief[0] = 0.80 if len(tactics) > 1 else 1.0
    memory = ThreatMemory.initialize(
        len(tactics),
        initial_belief=belief,
        confirmed_tactic=0,
    )
    steps = len(true_tactic_path)
    ledger = ResourceLedger(
        energy_j=world.max_total_energy_j * max(2.0, 0.76 * steps),
        emission_j=world.max_total_emission_j * max(1.5, 0.56 * steps),
        communication_mb=world.max_total_communication_mb * max(2.0, 0.76 * steps),
    )
    initial_resources = (ledger.energy_j, ledger.emission_j, ledger.communication_mb)
    fixed_index: int | None = None
    records: list[dict[str, Any]] = []
    previous_selected: int | None = None
    pending_alarm = False
    first_safety_fallback_step: int | None = None
    for step, tactic_index in enumerate(true_tactic_path):
        current_world = world_propagated_to_time(world, step * dt_s)
        covariance = propagate_normalized_covariance(covariance, current_world, layout, dt_s)
        adaptive = mode == "adaptive_receding_horizon"
        model = adaptive_threat_model(base_model, memory, blend=0.55 if pending_alarm else 0.35) if adaptive else base_model
        if mode in {"adaptive_receding_horizon", "clairvoyant_one_step_reference"} or fixed_index is None:
            decision_belief = np.asarray(belief, dtype=float)
            decision_policy = DecisionPolicy()
            horizon = 2 if adaptive else 1
            dual_control_weight = 0.18 if adaptive else 0.0
            if mode == "static_worst_case_policy":
                decision_policy = DecisionPolicy(cvar_weight=1.0)
            elif mode == "clairvoyant_one_step_reference":
                decision_belief = np.eye(len(tactics), dtype=float)[int(tactic_index)]
            decision = select_receding_horizon_action(
                covariance,
                current_world,
                candidates,
                tactics,
                model,
                decision_belief,
                ledger,
                policy=decision_policy,
                horizon=horizon,
                dual_control_weight=dual_control_weight,
                change_alarm=pending_alarm if adaptive else False,
                remaining_epochs=steps - step - 1,
            )
            selected_index = int(decision["selected_candidate_index"])
            if fixed_index is None:
                fixed_index = selected_index
        else:
            selected_index = int(fixed_index)
            ids = candidates[selected_index]
            by_id = {action.action_id: action for action in current_world.actions}
            actions = [by_id[int(value)] for value in ids]
            if not ledger.can_execute(current_world, actions):
                feasible_indices = [
                    index
                    for index, candidate in enumerate(candidates)
                    if ledger.can_execute(current_world, [by_id[int(value)] for value in candidate])
                ]
                if not feasible_indices:
                    raise RuntimeError("fixed controller exhausted every feasible campaign action")
                selected_index = min(feasible_indices)
                if first_safety_fallback_step is None:
                    first_safety_fallback_step = step
            decision = {"selected_candidate_index": selected_index, "selected_action_ids": list(candidates[selected_index]), "selected_score": None}
        action_ids = [int(value) for value in candidates[selected_index]]
        by_id = {action.action_id: action for action in current_world.actions}
        actions = [by_id[action_id] for action_id in action_ids]
        usage = ledger.execute(current_world, actions)
        prior_trace = float(np.trace(_physical_position_covariance(covariance, layout)))
        prior_position = _physical_position_covariance(covariance, layout)
        prior_sign, prior_logdet = np.linalg.slogdet(prior_position)
        if prior_sign <= 0.0:
            raise ValueError("autonomous prior physical covariance is not positive definite")
        true_tactic = tactics[int(tactic_index)]
        covariance = _posterior_for_tactic(covariance, current_world, actions, true_tactic)
        posterior_position = _physical_position_covariance(covariance, layout)
        posterior_trace = float(np.trace(posterior_position))
        posterior_sign, posterior_logdet = np.linalg.slogdet(posterior_position)
        if posterior_sign <= 0.0:
            raise ValueError("autonomous posterior physical covariance is not positive definite")
        realized_information_gain = float(prior_logdet - posterior_logdet)
        realized_decision_cost = float(decision_cost_for_actions(current_world, actions, DecisionPolicy()))
        realized_net_reward = realized_information_gain - realized_decision_cost
        likelihood, quality = candidate_observation_likelihood(current_world, action_ids, model)
        predictive_tactic = np.asarray(belief, dtype=float) @ model.transition_by_radar_action[selected_index]
        predictive_observation = predictive_tactic @ likelihood
        if memory.confirmed_tactic is None:
            detector_null = predictive_observation
        else:
            detector_null = likelihood[int(memory.confirmed_tactic)]
        positive_null = detector_null[detector_null > 0.0]
        expected_surprise = -float(np.sum(positive_null * np.log(positive_null)))
        observation = int(rng.choice(len(model.observation_ids), p=likelihood[int(tactic_index)]))
        posterior_belief, observation_probability = bayes_update_with_likelihood(
            belief,
            model.transition_by_radar_action[selected_index],
            likelihood,
            observation,
        )
        alarm, change_innovation, alarm_reason = memory.update(
            posterior_belief,
            float(detector_null[observation]),
            expected_surprise,
        )
        selected_changed = previous_selected is not None and selected_index != previous_selected
        records.append(
            {
                "step": step,
                "true_tactic_index": int(tactic_index),
                "true_tactic_id": "nominal" if true_tactic is None else true_tactic.kind,
                "selected_candidate_index": selected_index,
                "selected_action_ids": action_ids,
                "selected_changed": selected_changed,
                "active_count": sum(action.kind.startswith("active_") for action in actions),
                "passive_count": sum(action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"} for action in actions),
                "prior_position_trace_m2": prior_trace,
                "posterior_position_trace_m2": posterior_trace,
                "realized_information_gain_nats": realized_information_gain,
                "realized_decision_cost": realized_decision_cost,
                "realized_net_reward": realized_net_reward,
                "observation": model.observation_ids[observation],
                "observation_probability": observation_probability,
                "change_detector_null_probability": float(detector_null[observation]),
                "expected_surprise_nats": expected_surprise,
                "change_innovation_nats": change_innovation,
                "sensor_quality": quality,
                "belief": [float(value) for value in posterior_belief],
                "posterior_probability_of_true_tactic": float(posterior_belief[int(tactic_index)]),
                "belief_entropy_nats": float(-np.sum(posterior_belief[posterior_belief > 0.0] * np.log(posterior_belief[posterior_belief > 0.0]))),
                "change_alarm": alarm,
                "change_alarm_reason": alarm_reason,
                "cusum_after_update": float(memory.cusum),
                "change_alarm_refractory_epochs_remaining": int(memory.refractory_epochs_remaining),
                "controller_score_at_selection": decision.get("selected_score"),
                "terminal_backup_resource_invariant_satisfied": bool(
                    decision.get("terminal_backup_resource_invariant_satisfied", mode not in {"adaptive_receding_horizon", "clairvoyant_one_step_reference"})
                ),
                "remaining_energy_j": float(ledger.energy_j),
                "remaining_emission_j": float(ledger.emission_j),
                "remaining_communication_mb": float(ledger.communication_mb),
                **{f"used_{key}": value for key, value in usage.items()},
            }
        )
        belief = posterior_belief
        previous_selected = selected_index
        pending_alarm = alarm
    return {
        "mode": mode,
        "records": records,
        "mean_posterior_position_trace_m2": float(np.mean([row["posterior_position_trace_m2"] for row in records])),
        "final_posterior_position_trace_m2": float(records[-1]["posterior_position_trace_m2"]),
        "policy_switch_count": sum(bool(row["selected_changed"]) for row in records),
        "change_alarm_count": sum(bool(row["change_alarm"]) for row in records),
        "unique_candidate_count": len({int(row["selected_candidate_index"]) for row in records}),
        "resource_constraints_satisfied": min(ledger.energy_j, ledger.emission_j, ledger.communication_mb) >= -1e-9,
        "terminal_backup_resource_invariant_satisfied": all(
            bool(row["terminal_backup_resource_invariant_satisfied"]) for row in records
        ),
        "safety_fallback_used": first_safety_fallback_step is not None,
        "first_safety_fallback_step": first_safety_fallback_step,
        "cumulative_realized_information_gain_nats": float(sum(row["realized_information_gain_nats"] for row in records)),
        "cumulative_realized_net_reward": float(sum(row["realized_net_reward"] for row in records)),
        "mean_true_tactic_posterior_probability": float(np.mean([row["posterior_probability_of_true_tactic"] for row in records])),
        "maximum_page_cusum": float(memory.maximum_cusum),
        "initial_resources": {
            "energy_j": initial_resources[0],
            "emission_j": initial_resources[1],
            "communication_mb": initial_resources[2],
        },
        "remaining_resources": {
            "energy_j": ledger.energy_j,
            "emission_j": ledger.emission_j,
            "communication_mb": ledger.communication_mb,
        },
        "learned_transition": memory.learned_transition().tolist(),
    }


def autonomous_control_report(
    world: PhysicalWorld,
    candidates: Sequence[Sequence[int]],
    tactics: Sequence[Jammer | None],
    *,
    seed: int = 0,
    dt_s: float = 0.25,
    maximum_detection_delay_epochs: int = 5,
) -> dict[str, object]:
    if len(tactics) < 2 or len(candidates) < 2:
        raise ValueError("autonomous control requires multiple tactics and candidates")
    nominal = 0
    deceptive = next((index for index, value in enumerate(tactics) if value is not None and value.kind == "drfm_false_target"), len(tactics) - 1)
    persistent = next((index for index, value in enumerate(tactics) if value is not None and value.kind in {"swept", "spot", "barrage"}), 1)
    true_path = [nominal] * 4 + [deceptive] * 4 + [persistent] * 8
    adaptive = _simulate_controller(
        world, candidates, tactics, true_path, mode="adaptive_receding_horizon", seed=seed, dt_s=dt_s
    )
    fixed = _simulate_controller(
        world, candidates, tactics, true_path, mode="fixed_initial_policy", seed=seed, dt_s=dt_s
    )
    static_worst = _simulate_controller(
        world, candidates, tactics, true_path, mode="static_worst_case_policy", seed=seed, dt_s=dt_s
    )
    clairvoyant = _simulate_controller(
        world, candidates, tactics, true_path, mode="clairvoyant_one_step_reference", seed=seed, dt_s=dt_s
    )
    mean_improvement = float(fixed["mean_posterior_position_trace_m2"] - adaptive["mean_posterior_position_trace_m2"])
    reward_improvement = float(adaptive["cumulative_realized_net_reward"] - fixed["cumulative_realized_net_reward"])
    change_steps = [index for index in range(1, len(true_path)) if true_path[index] != true_path[index - 1]]
    alarm_steps = [int(row["step"]) for row in adaptive["records"] if row["change_alarm"]]
    detection_delays, matched_alarm_steps, unmatched_alarms = _match_change_alarms(
        change_steps,
        alarm_steps,
        int(maximum_detection_delay_epochs),
    )
    change_recall = float(sum(value is not None for value in detection_delays) / max(len(change_steps), 1))
    autonomous = (
        adaptive["unique_candidate_count"] > 1
        and adaptive["policy_switch_count"] > 0
        and adaptive["resource_constraints_satisfied"]
        and adaptive["terminal_backup_resource_invariant_satisfied"]
        and change_recall == 1.0
        and len(unmatched_alarms) <= 1
    )
    return {
        "stage": "v64_autonomous_receding_horizon_engineering_control",
        "status": "passed" if autonomous else "failed_closed",
        "true_tactic_path": true_path,
        "adaptive": adaptive,
        "fixed_policy_negative_control": fixed,
        "static_worst_case_negative_control": static_worst,
        "clairvoyant_one_step_reference": clairvoyant,
        "adaptive_minus_fixed_mean_trace_improvement_m2": mean_improvement,
        "adaptive_minus_fixed_cumulative_net_reward": reward_improvement,
        "adaptive_to_clairvoyant_one_step_reference_net_reward_ratio": float(
            adaptive["cumulative_realized_net_reward"]
            / max(float(clairvoyant["cumulative_realized_net_reward"]), 1e-12)
        ),
        "adaptive_policy_is_history_dependent": adaptive["unique_candidate_count"] > 1,
        "adaptive_policy_replans_online": adaptive["policy_switch_count"] > 0,
        "abrupt_change_detector_activated": adaptive["change_alarm_count"] > 0,
        "true_change_steps": change_steps,
        "change_alarm_steps": alarm_steps,
        "matched_change_alarm_steps": matched_alarm_steps,
        "unmatched_change_alarm_steps": unmatched_alarms,
        "change_detection_delays_epochs": detection_delays,
        "maximum_registered_detection_delay_epochs": int(maximum_detection_delay_epochs),
        "change_detection_recall_within_registered_delay": change_recall,
        "change_detection_false_alarm_count": len(unmatched_alarms),
        "long_horizon_threat_model_updated": not np.allclose(
            np.asarray(adaptive["learned_transition"]),
            np.full((len(tactics), len(tactics)), 1.0 / len(tactics)),
        ),
        "all_campaign_resource_constraints_satisfied": bool(
            adaptive["resource_constraints_satisfied"]
            and fixed["resource_constraints_satisfied"]
            and static_worst["resource_constraints_satisfied"]
            and clairvoyant["resource_constraints_satisfied"]
        ),
        "control_theory_contract": {
            "state": "target_covariance+tactic_belief+threat_memory+resource_ledger",
            "estimator": "Bayesian_hidden_tactic_filter+information_form_target_update",
            "controller": "receding_horizon_dual_control_with_robust_tail_and_resource_filter",
            "adaptation": "forgetting_transition_counts+predictive_entropy_centered_Page_CUSUM",
            "safety": "recursive_candidate_feasibility+terminal_backup_resource_invariant+campaign_resource_ledger",
            "fixed_policy_is_only_a_negative_control": True,
            "static_worst_case_policy_is_history_free_negative_control": True,
            "clairvoyant_one_step_policy_is_a_reference_not_a_reachable_controller": True,
        },
        "claim_boundary": {
            "global_closed_loop_stability_proved": False,
            "finite_scenario_recursive_feasibility_tested": True,
            "field_autonomy_validated": False,
        },
    }


def autonomous_control_ensemble_report(
    world: PhysicalWorld,
    candidates: Sequence[Sequence[int]],
    tactics: Sequence[Jammer | None],
    *,
    observation_seeds: Sequence[int],
    workers: int = 1,
    dt_s: float = 0.25,
) -> dict[str, object]:
    """Evaluate stochastic closed-loop evidence without cherry-picking a path."""

    seeds = tuple(dict.fromkeys(int(value) for value in observation_seeds))
    if not seeds:
        raise ValueError("autonomous ensemble requires at least one observation seed")

    def run(seed: int) -> dict[str, object]:
        return autonomous_control_report(world, candidates, tactics, seed=seed, dt_s=dt_s)

    if int(workers) > 1 and len(seeds) > 1:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(seeds))) as executor:
            reports = list(executor.map(run, seeds))
    else:
        reports = [run(seed) for seed in seeds]
    rows = []
    for seed, report in zip(seeds, reports):
        rows.append({
            "observation_seed": seed,
            "single_trajectory_status": report["status"],
            "adaptive_minus_fixed_cumulative_net_reward": float(
                report["adaptive_minus_fixed_cumulative_net_reward"]
            ),
            "adaptive_minus_fixed_mean_trace_improvement_m2": float(
                report["adaptive_minus_fixed_mean_trace_improvement_m2"]
            ),
            "change_detection_recall": float(
                report["change_detection_recall_within_registered_delay"]
            ),
            "change_detection_false_alarm_count": int(report["change_detection_false_alarm_count"]),
            "change_detection_delays_epochs": list(report["change_detection_delays_epochs"]),
            "policy_switch_count": int(report["adaptive"]["policy_switch_count"]),
            "resource_constraints_satisfied": bool(
                report["all_campaign_resource_constraints_satisfied"]
            ),
        })
    reward = np.asarray([row["adaptive_minus_fixed_cumulative_net_reward"] for row in rows], dtype=float)
    trace = np.asarray([row["adaptive_minus_fixed_mean_trace_improvement_m2"] for row in rows], dtype=float)
    recall = np.asarray([row["change_detection_recall"] for row in rows], dtype=float)
    false_alarms = np.asarray([row["change_detection_false_alarm_count"] for row in rows], dtype=float)
    pass_fraction = float(np.mean([row["single_trajectory_status"] == "passed" for row in rows]))
    checks = {
        "change_detection_mean_recall_at_least_0_90": float(np.mean(recall)) >= 0.90,
        "mean_false_alarms_at_most_1_per_16_epoch_campaign": float(np.mean(false_alarms)) <= 1.0,
        "adaptive_reward_positive_fraction_at_least_0_90": float(np.mean(reward > 0.0)) >= 0.90,
        "adaptive_trace_improvement_positive_fraction_at_least_0_90": float(np.mean(trace > 0.0)) >= 0.90,
        "single_trajectory_strict_pass_fraction_at_least_0_80": pass_fraction >= 0.80,
        "all_campaigns_preserve_resource_constraints": all(
            bool(row["resource_constraints_satisfied"]) for row in rows
        ),
    }
    return {
        "stage": "v64_autonomous_control_stochastic_ensemble",
        "status": "passed" if all(checks.values()) else "failed_closed",
        "observation_seed_count": len(seeds),
        "observation_seeds": list(seeds),
        "checks": checks,
        "trajectory_rows": rows,
        "strict_single_trajectory_pass_fraction": pass_fraction,
        "adaptive_minus_fixed_cumulative_net_reward_mean": float(np.mean(reward)),
        "adaptive_minus_fixed_cumulative_net_reward_minimum": float(np.min(reward)),
        "adaptive_reward_positive_fraction": float(np.mean(reward > 0.0)),
        "adaptive_minus_fixed_mean_trace_improvement_m2_mean": float(np.mean(trace)),
        "adaptive_minus_fixed_mean_trace_improvement_m2_minimum": float(np.min(trace)),
        "adaptive_trace_improvement_positive_fraction": float(np.mean(trace > 0.0)),
        "change_detection_mean_recall": float(np.mean(recall)),
        "change_detection_mean_false_alarm_count": float(np.mean(false_alarms)),
        "change_detection_maximum_false_alarm_count": int(np.max(false_alarms)),
        "representative_report": reports[0],
        "claim_boundary": {
            "registered_observation_noise_ensemble": True,
            "unknown_threat_model_generalization_validated": False,
            "field_detection_probability_established": False,
        },
    }
