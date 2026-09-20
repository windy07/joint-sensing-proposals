from __future__ import annotations

"""Action-conditioned recurrent ensemble closure model for V65.

This is a learned multi-observable transition closure, not a utility regressor.
A typed scene/factor encoder maps targets, threat duration/belief, topology,
spectrum/reference state, campaign resources and previous/current actions into a
registered vector.  A constant-velocity/persistent-regime physical transport is
the leading term.  Each ensemble member learns only the recurrent interaction
residual around that transport, with heteroscedastic closure covariance.
Between-member dispersion is epistemic uncertainty; residual covariance is
aleatoric uncertainty.  Mahalanobis OOD distance inflates uncertainty and can
truncate planning rollouts.  Training consumes complete chronological episodes
only and never exact utility, future labels or oracle action indices as inputs.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.dynamic_episode import EpisodeState, ThreatMode
from wideband_info_geometry.physics_v58.schema import PhysicalWorld, RadarAction


THREAT_ORDER = tuple(ThreatMode)

INTERACTION_HISTORY_FEATURE_NAMES = (
    "communication_cycle_density",
    "observation_cycle_density",
    "observation_pair_overlap_density",
    "observation_incidence_density",
    "connected_pair_motif_density",
    "connected_triple_motif_density",
    "repeated_participant_fraction",
    "interaction_tree_excess_fraction",
)


@dataclass(frozen=True)
class TypedSceneGraph:
    epoch: int
    target_nodes: np.ndarray
    sensor_nodes: np.ndarray
    threat_node: np.ndarray
    spectrum_nodes: np.ndarray
    resource_node: np.ndarray
    association_node: np.ndarray
    action_nodes: np.ndarray
    observation_edges: np.ndarray
    communication_edges: np.ndarray
    feature_schema: Mapping[str, tuple[str, ...]]
    dt_s: float = 1.0

    def validate(self) -> None:
        arrays = (
            self.target_nodes,
            self.sensor_nodes,
            self.threat_node,
            self.spectrum_nodes,
            self.resource_node,
            self.association_node,
            self.action_nodes,
            self.observation_edges,
            self.communication_edges,
        )
        if any(not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays):
            raise ValueError("typed scene graph contains non-finite features")


@dataclass(frozen=True)
class WorldModelTransition:
    episode_id: str
    epoch: int
    scene: TypedSceneGraph
    target_vector: np.ndarray
    next_observable: np.ndarray
    next_observable_names: tuple[str, ...]
    action_ids: tuple[int, ...]
    split: str
    generator_version: str

    def __post_init__(self) -> None:
        target = np.asarray(self.target_vector, dtype=float).reshape(-1)
        outcome = np.asarray(self.next_observable, dtype=float).reshape(-1)
        if outcome.size != len(self.next_observable_names):
            raise ValueError("world-model output schema disagrees")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(outcome)):
            raise ValueError("world-model transition contains non-finite values")
        object.__setattr__(self, "target_vector", target)
        object.__setattr__(self, "next_observable", outcome)
        self.scene.validate()


@dataclass(frozen=True)
class WorldModelPrediction:
    mean: np.ndarray
    aleatoric_variance: np.ndarray
    epistemic_variance: np.ndarray
    total_variance: np.ndarray
    member_means: np.ndarray
    hidden_states: np.ndarray
    stochastic_latents: np.ndarray
    ood_score: float
    uncertainty_inflation: float
    rollout_truncated: bool
    observable_names: tuple[str, ...]
    closure_baseline: np.ndarray | None = None
    closure_residual_mean: np.ndarray | None = None
    closure_error_bound: np.ndarray | None = None
    interaction_cumulant_inflation: float = 1.0
    projection_correction_norm: float = 0.0
    maximum_preprojection_violation: float = 0.0
    physical_projection_applied: bool = False
    connected_interaction_pressure: float = 0.0


@dataclass(frozen=True)
class WorldModelMetrics:
    sample_count: int
    one_step_rmse: float
    negative_log_likelihood: float
    coverage_50: float
    coverage_90: float
    expected_calibration_error: float
    mean_ood_score: float
    maximum_ood_score: float
    multi_step_rmse_by_horizon: tuple[float, ...]
    teacher_forced_multi_step_rmse_by_horizon: tuple[float, ...]
    analytic_multi_step_rmse_by_horizon: tuple[float, ...]
    analytic_one_step_rmse: float
    learned_closure_residual_rmse: float
    closure_target_rms: float
    mean_predicted_closure_error_bound: float
    closure_gain_over_analytic: float
    open_loop_uses_future_environment_state: bool
    calibration_scale: float
    split: str
    mean_projection_correction_norm: float
    maximum_preprojection_violation: float
    physical_projection_activation_rate: float
    mean_connected_interaction_pressure: float


@dataclass(frozen=True)
class WorldModelConfig:
    hidden_dim: int = 32
    stochastic_dim: int = 12
    ensemble_size: int = 5
    ridge: float = 1e-3
    bootstrap_fraction: float = 0.85
    seed: int = 6500
    ood_threshold: float = 4.0
    ood_variance_gain: float = 0.35
    rollout_truncation_threshold: float = 8.0
    minimum_variance: float = 1e-6
    interaction_cycle_variance_gain: float = 0.50
    projection_variance_gain: float = 1.0

    def validate(self) -> None:
        if min(self.hidden_dim, self.stochastic_dim, self.ensemble_size) <= 0:
            raise ValueError("world-model dimensions must be positive")
        if self.ridge <= 0.0 or not 0.0 < self.bootstrap_fraction <= 1.0:
            raise ValueError("world-model regularization/bootstrap is invalid")
        if self.ood_threshold <= 0.0 or self.rollout_truncation_threshold <= self.ood_threshold:
            raise ValueError("world-model OOD thresholds are invalid")
        if self.interaction_cycle_variance_gain < 0.0:
            raise ValueError("interaction-cycle variance gain must be nonnegative")
        if self.projection_variance_gain < 0.0:
            raise ValueError("projection variance gain must be nonnegative")


@dataclass
class _Member:
    recurrent_input: np.ndarray
    recurrent_hidden: np.ndarray
    stochastic_input: np.ndarray
    output_weights: np.ndarray
    log_variance: np.ndarray
    hidden_dim: int
    stochastic_dim: int

    def step(self, encoded_input: np.ndarray, previous_hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.tanh(self.recurrent_input @ encoded_input + self.recurrent_hidden @ previous_hidden)
        stochastic_mean = np.tanh(self.stochastic_input @ np.r_[encoded_input, hidden])
        features = np.r_[1.0, encoded_input, hidden, stochastic_mean]
        mean = features @ self.output_weights
        return mean, hidden, stochastic_mean


def _project_probability_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the unit probability simplex."""

    vector = np.asarray(value, dtype=float).reshape(-1)
    if not vector.size:
        return vector.copy()
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    support = np.flatnonzero(
        ordered - cumulative / np.arange(1, vector.size + 1) > 0.0
    )
    if not support.size:
        return np.full(vector.size, 1.0 / vector.size)
    rho = int(support[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / max(float(np.sum(projected)), 1e-300)


def project_observable_state(
    value: np.ndarray,
    observable_names: Sequence[str],
    *,
    current_baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a learned closure output onto the registered physical state set.

    The finite campaign model has no resource replenishment transition.
    Energy, emission, communication and compute fractions therefore cannot
    increase over the current analytic baseline. Threat probabilities are
    projected jointly, rather than clipped component by component.
    """

    raw = np.asarray(value, dtype=float).reshape(-1)
    if raw.size != len(observable_names) or not np.all(np.isfinite(raw)):
        raise ValueError("observable projection received an invalid state")
    baseline = (
        np.asarray(current_baseline, dtype=float).reshape(raw.shape)
        if current_baseline is not None
        else None
    )
    projected = raw.copy()
    index = {str(name): position for position, name in enumerate(observable_names)}
    bounded = (
        "target_existence",
        "threat_duration_fraction",
        "spectrum_occupancy",
        "reference_quality",
        "connectivity",
        "node_health_fraction",
        "association_information_retention",
        "target_count_fraction",
        "energy_fraction",
        "emission_fraction",
        "communication_fraction",
        "compute_fraction",
    )
    for name in bounded:
        if name in index:
            projected[index[name]] = float(np.clip(projected[index[name]], 0.0, 1.0))
    threat_indices = [
        index[f"threat_mode_{mode.value}"]
        for mode in THREAT_ORDER
        if f"threat_mode_{mode.value}" in index
    ]
    if len(threat_indices) == len(THREAT_ORDER):
        projected[threat_indices] = _project_probability_simplex(projected[threat_indices])
    if "false_alarm_intensity" in index:
        location = index["false_alarm_intensity"]
        projected[location] = max(0.0, float(projected[location]))
    if baseline is not None:
        for name in ("energy_fraction", "emission_fraction", "communication_fraction", "compute_fraction"):
            if name in index:
                location = index[name]
                projected[location] = min(
                    projected[location],
                    float(np.clip(baseline[location], 0.0, 1.0)),
                )
    return projected, projected - raw


@dataclass
class RecurrentEnsembleWorldModel:
    config: WorldModelConfig
    input_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    members: list[_Member] = field(default_factory=list)
    input_mean: np.ndarray | None = None
    input_scale: np.ndarray | None = None
    input_covariance_inverse: np.ndarray | None = None
    output_mean: np.ndarray | None = None
    output_scale: np.ndarray | None = None
    calibration_scale: float = 1.0
    training_episode_ids: tuple[str, ...] = ()
    model_hash: str = ""
    controller_qualified: bool = False
    controller_qualification_reason: str = "not_evaluated"
    baseline_input_start: int | None = None
    interaction_input_start: int | None = None
    closure_residual_training_rmse: float = math.nan
    closure_model: str = "absolute_transition_legacy"

    def __post_init__(self) -> None:
        self.config.validate()

    @property
    def fitted(self) -> bool:
        return bool(self.members and self.input_mean is not None and self.output_mean is not None)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("world model is not fitted")

    def _normalize_input(self, value: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.input_mean is not None and self.input_scale is not None
        return (np.asarray(value, dtype=float) - self.input_mean) / self.input_scale

    def _ood_score(self, normalized_input: np.ndarray) -> float:
        assert self.input_covariance_inverse is not None
        return float(math.sqrt(max(normalized_input @ self.input_covariance_inverse @ normalized_input / max(normalized_input.size, 1), 0.0)))

    def predict(
        self,
        encoded_input: np.ndarray,
        *,
        hidden_states: np.ndarray | None = None,
        seed: int = 0,
        stochastic: bool = False,
    ) -> WorldModelPrediction:
        self._check_fitted()
        raw_input = np.asarray(encoded_input, dtype=float).reshape(-1)
        normalized = self._normalize_input(raw_input)
        if normalized.size != len(self.input_names):
            raise ValueError("world-model input dimension disagrees")
        if hidden_states is None:
            hidden_states = np.zeros((len(self.members), self.config.hidden_dim), dtype=float)
        hidden_states = np.asarray(hidden_states, dtype=float)
        if hidden_states.shape != (len(self.members), self.config.hidden_dim):
            raise ValueError("world-model hidden-state dimension disagrees")
        member_means: list[np.ndarray] = []
        projection_corrections: list[np.ndarray] = []
        next_hidden: list[np.ndarray] = []
        stochastic_latents: list[np.ndarray] = []
        aleatoric: list[np.ndarray] = []
        rng = np.random.default_rng(seed)
        assert self.output_mean is not None and self.output_scale is not None
        closure_baseline = (
            raw_input[
                self.baseline_input_start :
                self.baseline_input_start + len(self.observable_names)
            ]
            if self.baseline_input_start is not None
            else np.zeros(len(self.observable_names), dtype=float)
        )
        for index, member in enumerate(self.members):
            mean_normalized, hidden, stochastic_mean = member.step(normalized, hidden_states[index])
            variance_normalized = np.exp(member.log_variance)
            latent = stochastic_mean + (rng.normal(size=stochastic_mean.size) * 0.15 if stochastic else 0.0)
            # Stochastic latent perturbation enters through the learned output
            # features at the next recurrent call.  Current predictive density
            # is centered on the deterministic posterior mean.
            residual_mean = self.output_mean + self.output_scale * mean_normalized
            raw_mean = closure_baseline + residual_mean
            mean, projection_correction = project_observable_state(
                raw_mean,
                self.observable_names,
                current_baseline=closure_baseline,
            )
            variance = (self.output_scale**2) * variance_normalized
            variance += self.config.projection_variance_gain * projection_correction**2
            member_means.append(mean)
            projection_corrections.append(projection_correction)
            aleatoric.append(variance)
            next_hidden.append(hidden)
            stochastic_latents.append(latent)
        means = np.asarray(member_means)
        mean = np.mean(means, axis=0)
        aleatoric_variance = np.mean(np.asarray(aleatoric), axis=0)
        epistemic_variance = np.var(means, axis=0, ddof=1) if len(self.members) > 1 else np.zeros_like(mean)
        score = self._ood_score(normalized)
        ood_inflation = 1.0 + self.config.ood_variance_gain * max(
            0.0, score - self.config.ood_threshold
        ) ** 2
        interaction_cycle_pressure = 0.0
        if self.interaction_input_start is not None:
            interaction_features = np.clip(
                raw_input[
                    self.interaction_input_start :
                    self.interaction_input_start
                    + len(INTERACTION_HISTORY_FEATURE_NAMES)
                ],
                0.0,
                1.0,
            )
            interaction_weights = np.asarray(
                [0.20, 0.20, 0.12, 0.04, 0.14, 0.14, 0.08, 0.08],
                dtype=float,
            )
            if interaction_features.size != interaction_weights.size:
                raise ValueError("registered connected-interaction feature width changed")
            interaction_cycle_pressure = float(interaction_features @ interaction_weights)
        interaction_inflation = (
            1.0
            + self.config.interaction_cycle_variance_gain
            * interaction_cycle_pressure
        )
        inflation = ood_inflation * interaction_inflation
        total = self.calibration_scale * inflation * (aleatoric_variance + epistemic_variance)
        total = np.maximum(total, self.config.minimum_variance)
        residual_mean = mean - closure_baseline
        correction_matrix = np.asarray(projection_corrections, dtype=float)
        correction_norm = float(math.sqrt(np.mean(correction_matrix**2)))
        maximum_correction = float(np.max(np.abs(correction_matrix)))
        return WorldModelPrediction(
            mean=mean,
            aleatoric_variance=np.maximum(self.calibration_scale * inflation * aleatoric_variance, self.config.minimum_variance),
            epistemic_variance=np.maximum(self.calibration_scale * inflation * epistemic_variance, 0.0),
            total_variance=total,
            member_means=means,
            hidden_states=np.asarray(next_hidden),
            stochastic_latents=np.asarray(stochastic_latents),
            ood_score=score,
            uncertainty_inflation=inflation,
            rollout_truncated=bool(score >= self.config.rollout_truncation_threshold),
            observable_names=self.observable_names,
            closure_baseline=closure_baseline.copy(),
            closure_residual_mean=residual_mean,
            closure_error_bound=np.sqrt(total),
            interaction_cumulant_inflation=interaction_inflation,
            projection_correction_norm=correction_norm,
            maximum_preprojection_violation=maximum_correction,
            physical_projection_applied=bool(maximum_correction > 1e-12),
            connected_interaction_pressure=interaction_cycle_pressure,
        )

    def rollout(
        self,
        initial_input: np.ndarray,
        action_conditioned_inputs: Sequence[np.ndarray],
        *,
        seed: int = 0,
        truncate_on_ood: bool = True,
    ) -> tuple[WorldModelPrediction, ...]:
        hidden = np.zeros((len(self.members), self.config.hidden_dim), dtype=float)
        predictions: list[WorldModelPrediction] = []
        # ``initial_input`` is the first current-state/action encoding.
        # Subsequent encodings are scenario-tree contexts produced without
        # future observations.  The former implementation silently discarded
        # the initial input, shortening every rollout by one step.
        inputs = [np.asarray(initial_input, dtype=float), *(np.asarray(value, dtype=float) for value in action_conditioned_inputs)]
        for step, current in enumerate(inputs):
            prediction = self.predict(current, hidden_states=hidden, seed=seed + step, stochastic=True)
            predictions.append(prediction)
            hidden = prediction.hidden_states
            if truncate_on_ood and prediction.rollout_truncated:
                break
        return tuple(predictions)

    def save(self, directory: Path) -> dict[str, str]:
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "input_mean": np.asarray(self.input_mean),
            "input_scale": np.asarray(self.input_scale),
            "input_covariance_inverse": np.asarray(self.input_covariance_inverse),
            "output_mean": np.asarray(self.output_mean),
            "output_scale": np.asarray(self.output_scale),
        }
        for index, member in enumerate(self.members):
            arrays[f"member_{index}_recurrent_input"] = member.recurrent_input
            arrays[f"member_{index}_recurrent_hidden"] = member.recurrent_hidden
            arrays[f"member_{index}_stochastic_input"] = member.stochastic_input
            arrays[f"member_{index}_output_weights"] = member.output_weights
            arrays[f"member_{index}_log_variance"] = member.log_variance
        npz_path = directory / "v65_world_model.npz"
        np.savez_compressed(npz_path, **arrays)
        metadata = {
            "config": asdict(self.config),
            "input_names": self.input_names,
            "observable_names": self.observable_names,
            "calibration_scale": self.calibration_scale,
            "training_episode_ids": self.training_episode_ids,
            "model_hash": self.model_hash,
            "controller_qualified": self.controller_qualified,
            "controller_qualification_reason": self.controller_qualification_reason,
            "baseline_input_start": self.baseline_input_start,
            "interaction_input_start": self.interaction_input_start,
            "closure_residual_training_rmse": self.closure_residual_training_rmse,
            "closure_model": self.closure_model,
            "role": "analytic_transport_plus_recurrent_interaction_closure_not_utility_regressor",
        }
        metadata_path = directory / "v65_world_model.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"weights": str(npz_path), "metadata": str(metadata_path)}


def _action_features(world: PhysicalWorld, action_ids: Sequence[int], previous_action_ids: Sequence[int]) -> tuple[np.ndarray, tuple[str, ...]]:
    by_id = {action.action_id: action for action in world.actions}
    actions = [by_id[value] for value in action_ids if value in by_id]
    waveforms = [world.waveform(action.waveform_id) for action in actions]
    active = sum(action.kind.startswith("active") for action in actions)
    passive = sum(action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"} for action in actions)
    calibration = sum(action.kind.startswith("calibration") for action in actions)
    count = max(len(actions), 1)
    values = np.asarray(
        [
            active / count,
            passive / count,
            calibration / count,
            sum(action.energy_cost_j for action in actions),
            sum(action.emission_exposure_j for action in actions),
            sum(action.communication_cost_mb for action in actions),
            sum(waveform.bandwidth_hz * waveform.dwell_s for waveform in waveforms),
            sum(waveform.dwell_s for waveform in waveforms),
            len(set(action_ids).symmetric_difference(previous_action_ids)),
            float(np.mean([waveform.carrier_hz for waveform in waveforms])) if waveforms else 0.0,
        ],
        dtype=float,
    )
    names = (
        "action_active_fraction",
        "action_passive_fraction",
        "action_calibration_fraction",
        "action_energy_j",
        "action_emission_j",
        "action_communication_mb",
        "action_bandwidth_time_hz_s",
        "action_dwell_s",
        "action_switch_count",
        "action_mean_carrier_hz",
    )
    return values, names


def _cycle_rank_from_edges(
    node_count: int,
    edges: Sequence[tuple[int, int]],
) -> int:
    adjacency = {index: set() for index in range(int(node_count))}
    for left, right in edges:
        if left == right or left not in adjacency or right not in adjacency:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = 0
    visited: set[int] = set()
    for root in adjacency:
        if root in visited:
            continue
        components += 1
        stack = [root]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node].difference(visited))
    edge_count = sum(len(value) for value in adjacency.values()) // 2
    return max(0, edge_count - int(node_count) + components)


def _interaction_history_summary(
    sensor_count: int,
    communication_edges: np.ndarray,
    observation_edges: np.ndarray,
) -> np.ndarray:
    communication_pairs = [
        (int(row[0]), int(row[1]))
        for row in np.asarray(communication_edges, dtype=float)
        if len(row) >= 2
    ]
    communication_cycle_rank = _cycle_rank_from_edges(
        sensor_count, communication_pairs
    )
    incidence_edges: list[tuple[int, int]] = []
    hyperedge_count = int(np.asarray(observation_edges).shape[0])
    for edge_index, row in enumerate(np.asarray(observation_edges, dtype=float)):
        hyperedge_node = sensor_count + edge_index
        for value in row:
            node = int(value)
            if 0 <= node < sensor_count:
                incidence_edges.append((node, hyperedge_node))
    observation_cycle_rank = _cycle_rank_from_edges(
        sensor_count + hyperedge_count,
        incidence_edges,
    )
    total_pairs = hyperedge_count * (hyperedge_count - 1) // 2
    edge_sets = [
        {int(value) for value in row if 0 <= int(value) < sensor_count}
        for row in np.asarray(observation_edges, dtype=float)
    ]
    factor_adjacency = np.zeros((hyperedge_count, hyperedge_count), dtype=float)
    for left in range(len(edge_sets)):
        for right in range(left + 1, len(edge_sets)):
            if edge_sets[left].intersection(edge_sets[right]):
                factor_adjacency[left, right] = 1.0
                factor_adjacency[right, left] = 1.0
    overlap_pairs = int(np.sum(factor_adjacency) // 2)
    degrees = np.sum(factor_adjacency, axis=1)
    wedges = float(np.sum(degrees * np.maximum(degrees - 1.0, 0.0) / 2.0))
    triangles = (
        float(np.trace(factor_adjacency @ factor_adjacency @ factor_adjacency) / 6.0)
        if hyperedge_count >= 3
        else 0.0
    )
    connected_triples = max(0.0, wedges - 2.0 * triangles)
    total_triples = math.comb(hyperedge_count, 3) if hyperedge_count >= 3 else 0
    participant_degrees = np.zeros(max(sensor_count, 1), dtype=float)
    for edge in edge_sets:
        for participant in edge:
            participant_degrees[participant] += 1.0
    repeated_incidence = float(np.sum(np.maximum(participant_degrees - 1.0, 0.0)))
    communication_capacity = max(
        sensor_count * (sensor_count - 1) // 2 - sensor_count + 1, 1
    )
    observation_capacity = max(len(incidence_edges) - sensor_count + 1, 1)
    summary = np.asarray(
        [
            communication_cycle_rank / communication_capacity,
            observation_cycle_rank / observation_capacity,
            overlap_pairs / max(total_pairs, 1),
            len(incidence_edges) / max(sensor_count * max(hyperedge_count, 1), 1),
            overlap_pairs / max(total_pairs, 1),
            connected_triples / max(total_triples, 1),
            repeated_incidence / max(len(incidence_edges), 1),
            observation_cycle_rank / max(len(incidence_edges), 1),
        ],
        dtype=float,
    )
    if summary.size != len(INTERACTION_HISTORY_FEATURE_NAMES):
        raise AssertionError("connected-interaction summary schema is inconsistent")
    return np.clip(summary, 0.0, 1.0)


def _observable_from_visible_scene(
    graph: TypedSceneGraph,
) -> tuple[np.ndarray, tuple[str, ...]]:
    target = np.asarray(graph.target_nodes[0], dtype=float)
    threat = np.asarray(graph.threat_node, dtype=float)
    sensor = np.asarray(graph.sensor_nodes, dtype=float)
    spectrum = np.asarray(graph.spectrum_nodes, dtype=float)
    association = np.asarray(graph.association_node, dtype=float)
    resource = np.asarray(graph.resource_node, dtype=float)
    sensor_count = max(sensor.shape[0], 1)
    connectivity = float(graph.communication_edges.shape[0]) / max(
        sensor_count * (sensor_count - 1) / 2.0, 1.0
    )
    values = np.r_[
        target[0:3],
        target[3:6],
        target[9],
        threat[: len(THREAT_ORDER)],
        threat[len(THREAT_ORDER)],
        float(np.mean(spectrum[:, 2])) if spectrum.size else 0.0,
        float(np.mean(spectrum[:, 4])) if spectrum.size else 0.0,
        connectivity,
        float(np.mean(sensor[:, 0] < 1.0)) if sensor.size else 0.0,
        association[5],
        association[6],
        association[0],
        resource[0],
        resource[1],
        resource[4],
        resource[5],
    ]
    names = tuple(
        [f"target_position_{axis}" for axis in "xyz"]
        + [f"target_velocity_{axis}" for axis in "xyz"]
        + ["target_existence"]
        + [f"threat_mode_{item.value}" for item in THREAT_ORDER]
        + [
            "threat_duration_fraction",
            "spectrum_occupancy",
            "reference_quality",
            "connectivity",
            "node_health_fraction",
            "association_information_retention",
            "false_alarm_intensity",
            "target_count_fraction",
            "energy_fraction",
            "emission_fraction",
            "communication_fraction",
            "compute_fraction",
        ]
    )
    return values, names


def encode_typed_scene(
    state: EpisodeState,
    world: PhysicalWorld,
    action_ids: Sequence[int],
    *,
    target_mean_override_m_m_s: np.ndarray | None = None,
    threat_mode_probabilities_override: np.ndarray | None = None,
    threat_duration_fraction_override: float | None = None,
) -> tuple[TypedSceneGraph, np.ndarray, tuple[str, ...]]:
    existing_targets = [target for target in state.targets if target.exists]
    target_rows = []
    encoded_targets = existing_targets
    if target_mean_override_m_m_s is not None:
        override = np.asarray(target_mean_override_m_m_s, dtype=float).reshape(6)
        if not np.all(np.isfinite(override)):
            raise ValueError("target belief override must be finite 6-D kinematics")
        encoded_targets = [state.primary_target]
    for target in encoded_targets:
        position = np.asarray(target.position_m, dtype=float)
        velocity = np.asarray(target.velocity_m_s, dtype=float)
        acceleration = np.asarray(target.acceleration_m_s2, dtype=float)
        if target_mean_override_m_m_s is not None:
            position = override[:3]
            velocity = override[3:]
            acceleration = np.zeros(3)
        target_rows.append(
            np.r_[
                position / 10_000.0,
                velocity / 300.0,
                acceleration / 10.0,
                target.existence_probability,
                math.log(max(target.mean_rcs_m2, 1e-9)),
                math.sin(target.aspect_state_rad),
                math.cos(target.aspect_state_rad),
                target.mode_age_steps / max(target.mode_duration_steps, 1),
            ]
        )
    target_nodes = np.asarray(target_rows if target_rows else [np.zeros(14)], dtype=float)
    sensor_rows = []
    health_order = {"healthy": 0.0, "degraded": 0.33, "recovering": 0.67, "failed": 1.0}
    for node in state.nodes:
        sensor_rows.append(
            [
                health_order[node.health.value],
                node.clock_bias_s / 1e-6,
                node.clock_drift_s_per_s / 1e-8,
                node.cfo_hz / 1000.0,
                node.calibration_age_s / 60.0,
                node.packet_delay_steps / 8.0,
                node.packet_loss_probability,
                node.queue_megabytes / 10.0,
                node.compute_load_fraction,
                float(node.communication_available),
                float(node.receiver_available),
                float(node.transmitter_available),
            ]
        )
    sensor_nodes = np.asarray(sensor_rows, dtype=float)
    if threat_mode_probabilities_override is None:
        threat_one_hot = np.asarray([float(state.adversary.mode is item) for item in THREAT_ORDER], dtype=float)
    else:
        threat_one_hot = np.asarray(threat_mode_probabilities_override, dtype=float).reshape(-1)
        if threat_one_hot.shape != (len(THREAT_ORDER),) or np.any(threat_one_hot < 0.0) or not np.all(np.isfinite(threat_one_hot)):
            raise ValueError("threat belief override must be a finite probability vector")
        mass = float(np.sum(threat_one_hot))
        if mass <= 0.0:
            raise ValueError("threat belief override has zero mass")
        threat_one_hot = threat_one_hot / mass
    duration_fraction = (
        state.adversary.mode_age_steps / max(state.adversary.mode_duration_steps, 1)
        if threat_duration_fraction_override is None
        else float(np.clip(threat_duration_fraction_override, 0.0, 1.0))
    )
    threat_node = np.r_[
        threat_one_hot,
        duration_fraction,
        state.adversary.energy_remaining_j / 500.0,
        state.adversary.deception_budget / 20.0,
        state.adversary.internal_belief_active_probability,
        state.adversary.adaptation_memory.get("active_fraction_ema", 0.0),
        state.adversary.adaptation_memory.get("passive_fraction_ema", 0.0),
        state.adversary.adaptation_memory.get("switch_rate_ema", 0.0),
    ]
    spectrum_rows = []
    for waveform in world.waveforms:
        spectrum_rows.append(
            [
                waveform.carrier_hz / 10e9,
                waveform.bandwidth_hz / 1e9,
                state.spectrum.occupancy_by_waveform.get(waveform.waveform_id, 1.0),
                float(state.spectrum.illuminator_available_by_waveform.get(waveform.waveform_id, False)),
                state.spectrum.reference_quality_by_waveform.get(waveform.waveform_id, 0.0),
                state.spectrum.endogeneity_memory.get(waveform.waveform_id, 0.0),
            ]
        )
    spectrum_nodes = np.asarray(spectrum_rows, dtype=float)
    resources = state.resources
    resource_node = np.asarray(
        [
            resources.energy_remaining_j / max(resources.energy_remaining_j + resources.reserve_energy_j, 1.0),
            resources.emission_remaining_j / max(resources.emission_remaining_j + resources.reserve_emission_j, 1.0),
            resources.bandwidth_time_remaining_hz_s / 1e9,
            resources.dwell_remaining_s / 10.0,
            resources.communication_remaining_mb / 100.0,
            resources.compute_remaining_s / 20.0,
            resources.switching_cost_accumulated,
            resources.calibration_time_accumulated_s,
        ],
        dtype=float,
    )
    association = state.association_context
    association_node = np.asarray(
        [
            association.get("existing_target_count", 1.0) / 3.0,
            min(float(association.get("minimum_target_separation_m", 1.0e9)), 10_000.0) / 10_000.0,
            association.get("ambiguity_probability", 0.0),
            association.get("correct_association_probability", 1.0),
            association.get("association_entropy_nats", 0.0) / math.log(2.0),
            association.get("information_retention", 1.0),
            association.get("false_alarm_intensity", 0.0),
        ],
        dtype=float,
    )
    action_values, action_names = _action_features(world, action_ids, resources.last_action_ids)
    action_nodes = action_values.reshape(1, -1)
    node_index = {node.node_id: index for index, node in enumerate(state.nodes)}
    observation_edges = np.asarray(
        [[node_index[node_id] for node_id in edge] + [-1] * (3 - len(edge)) for edge in state.observation_hyperedges if len(edge) <= 3],
        dtype=float,
    ) if state.observation_hyperedges else np.zeros((0, 3), dtype=float)
    communication_edges = np.asarray(
        [[node_index[left], node_index[right]] for left, right in state.communication_edges], dtype=float
    ) if state.communication_edges else np.zeros((0, 2), dtype=float)
    schema = {
        "target": ("position_xyz_scaled", "velocity_xyz_scaled", "acceleration_xyz_scaled", "existence", "log_rcs", "sin_aspect", "cos_aspect", "duration_fraction"),
        "sensor": ("health", "clock", "drift", "cfo", "calibration_age", "packet_delay", "packet_loss", "queue", "compute", "communication", "receiver", "transmitter"),
        "threat": tuple([f"mode_{item.value}" for item in THREAT_ORDER] + ["duration_fraction", "energy", "deception_budget", "belief_active", "active_ema", "passive_ema", "switch_ema"]),
        "spectrum": ("carrier", "bandwidth", "occupancy", "illuminator", "reference_quality", "endogeneity"),
        "resource": ("energy", "emission", "bandwidth_time", "dwell", "communication", "compute", "switch_cost", "calibration_time"),
        "association": ("target_count", "minimum_separation", "ambiguity", "correct_probability", "entropy", "information_retention", "false_alarm_intensity"),
        "action": action_names,
        "interaction_history": INTERACTION_HISTORY_FEATURE_NAMES,
    }
    graph = TypedSceneGraph(
        epoch=state.epoch,
        target_nodes=target_nodes,
        sensor_nodes=sensor_nodes,
        threat_node=threat_node,
        spectrum_nodes=spectrum_nodes,
        resource_node=resource_node,
        association_node=association_node,
        action_nodes=action_nodes,
        observation_edges=observation_edges,
        communication_edges=communication_edges,
        feature_schema=schema,
        dt_s=float(state.dt_s),
    )
    # Permutation-invariant pooling for the dependency-light recurrent backend.
    def pooled(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.zeros(0, dtype=float)
        return np.r_[np.mean(values, axis=0), np.std(values, axis=0), np.max(values, axis=0)]
    interaction_summary = _interaction_history_summary(
        len(state.nodes),
        communication_edges,
        observation_edges,
    )
    visible_observable, visible_observable_names = _observable_from_visible_scene(
        graph
    )
    analytic_closure_baseline = analytical_transition_baseline(
        visible_observable,
        dt_s=float(state.dt_s),
        observable_names=visible_observable_names,
    )
    vector_parts = [
        pooled(target_nodes),
        pooled(sensor_nodes),
        threat_node,
        pooled(spectrum_nodes),
        resource_node,
        association_node,
        action_values,
        np.asarray([
            len(state.communication_edges) / max(len(state.nodes) * (len(state.nodes) - 1) / 2.0, 1.0),
            len(state.observation_hyperedges) / max(len(world.actions), 1),
        ]),
        interaction_summary,
        analytic_closure_baseline,
    ]
    vector = np.concatenate(vector_parts)
    names: list[str] = []
    for group, values in (
        ("target_pool", pooled(target_nodes)),
        ("sensor_pool", pooled(sensor_nodes)),
        ("threat", threat_node),
        ("spectrum_pool", pooled(spectrum_nodes)),
        ("resource", resource_node),
        ("association", association_node),
        ("action", action_values),
        ("edge_summary", np.zeros(2)),
        ("interaction_history", interaction_summary),
        ("analytic_closure_baseline", analytic_closure_baseline),
    ):
        names.extend(f"{group}_{index}" for index in range(values.size))
    graph.validate()
    return graph, vector, tuple(names)


def counterfactual_scene_input(
    previous_input: np.ndarray,
    prediction: WorldModelPrediction,
    scene: TypedSceneGraph,
    planned_action_input: np.ndarray,
) -> np.ndarray:
    """Advance an encoded scene without reading a future environment state.

    Only predicted observables and the registered next action are inserted.
    Static/latent fields that the model did not predict are held persistent.
    This makes multi-step evaluation and MPC genuinely open loop instead of
    teacher-forcing the simulator's future target, threat, topology or resource
    state into every recurrent call.
    """

    value = np.asarray(previous_input, dtype=float).copy()
    planned = np.asarray(planned_action_input, dtype=float).reshape(value.shape)
    names = {name: index for index, name in enumerate(prediction.observable_names)}
    predicted = np.asarray(prediction.mean, dtype=float)

    target_width = int(scene.target_nodes.shape[1])
    sensor_width = int(scene.sensor_nodes.shape[1])
    spectrum_width = int(scene.spectrum_nodes.shape[1])
    target_start = 0
    sensor_start = target_start + 3 * target_width
    threat_start = sensor_start + 3 * sensor_width
    spectrum_start = threat_start + int(scene.threat_node.size)
    resource_start = spectrum_start + 3 * spectrum_width
    association_start = resource_start + int(scene.resource_node.size)
    action_start = association_start + int(scene.association_node.size)
    action_width = int(scene.action_nodes.size)
    edge_start = action_start + action_width
    interaction_start = edge_start + 2
    interaction_width = len(INTERACTION_HISTORY_FEATURE_NAMES)
    baseline_start = interaction_start + interaction_width
    baseline_width = len(prediction.observable_names)
    if baseline_start + baseline_width != value.size:
        raise ValueError("typed-scene pooled vector layout changed without an open-loop adapter update")

    target_fields = [
        ("target_position_x", 0), ("target_position_y", 1), ("target_position_z", 2),
        ("target_velocity_x", 3), ("target_velocity_y", 4), ("target_velocity_z", 5),
        ("target_existence", 9),
    ]
    for name, local in target_fields:
        if name in names:
            for pool_offset in (0, 2 * target_width):
                value[target_start + pool_offset + local] = predicted[names[name]]

    for mode_index, mode in enumerate(THREAT_ORDER):
        name = f"threat_mode_{mode.value}"
        if name in names:
            value[threat_start + mode_index] = max(0.0, predicted[names[name]])
    mode_mass = float(np.sum(value[threat_start : threat_start + len(THREAT_ORDER)]))
    if mode_mass > 0.0:
        value[threat_start : threat_start + len(THREAT_ORDER)] /= mode_mass
    if "threat_duration_fraction" in names:
        value[threat_start + len(THREAT_ORDER)] = float(np.clip(predicted[names["threat_duration_fraction"]], 0.0, 1.0))

    for output_name, local in (("spectrum_occupancy", 2), ("reference_quality", 4)):
        if output_name in names:
            for pool_offset in (0, 2 * spectrum_width):
                value[spectrum_start + pool_offset + local] = predicted[names[output_name]]
    for output_name, local in (("energy_fraction", 0), ("emission_fraction", 1), ("communication_fraction", 4), ("compute_fraction", 5)):
        if output_name in names:
            value[resource_start + local] = predicted[names[output_name]]
    for output_name, local in (("target_count_fraction", 0), ("association_information_retention", 5), ("false_alarm_intensity", 6)):
        if output_name in names:
            value[association_start + local] = predicted[names[output_name]]
    if "connectivity" in names:
        value[edge_start] = predicted[names["connectivity"]]
    value[action_start:edge_start] = planned[action_start:edge_start]
    value[baseline_start:] = analytical_transition_baseline(
        predicted,
        dt_s=float(scene.dt_s),
        observable_names=prediction.observable_names,
    )
    if not np.all(np.isfinite(value)):
        raise ValueError("open-loop scene update produced non-finite features")
    return value


def observable_from_state(state: EpisodeState, world: PhysicalWorld) -> tuple[np.ndarray, tuple[str, ...]]:
    target = state.primary_target
    mode = np.asarray([float(state.adversary.mode is item) for item in THREAT_ORDER], dtype=float)
    occupancy = float(np.mean(list(state.spectrum.occupancy_by_waveform.values())))
    reference = float(np.mean(list(state.spectrum.reference_quality_by_waveform.values())))
    connectivity = len(state.communication_edges) / max(len(state.nodes) * (len(state.nodes) - 1) / 2.0, 1.0)
    health = sum(node.health.value != "failed" for node in state.nodes) / max(len(state.nodes), 1)
    resource = state.resources
    values = np.r_[
        target.position_m / 10_000.0,
        target.velocity_m_s / 300.0,
        target.existence_probability,
        mode,
        state.adversary.mode_age_steps / max(state.adversary.mode_duration_steps, 1),
        occupancy,
        reference,
        connectivity,
        health,
        state.association_context.get("information_retention", 1.0),
        state.association_context.get("false_alarm_intensity", 0.0),
        state.association_context.get("existing_target_count", 1.0) / 3.0,
        resource.energy_remaining_j / max(resource.energy_remaining_j + resource.reserve_energy_j, 1.0),
        resource.emission_remaining_j / max(resource.emission_remaining_j + resource.reserve_emission_j, 1.0),
        resource.communication_remaining_mb / 100.0,
        resource.compute_remaining_s / 20.0,
    ]
    names = tuple(
        [f"target_position_{axis}" for axis in "xyz"]
        + [f"target_velocity_{axis}" for axis in "xyz"]
        + ["target_existence"]
        + [f"threat_mode_{item.value}" for item in THREAT_ORDER]
        + ["threat_duration_fraction", "spectrum_occupancy", "reference_quality", "connectivity", "node_health_fraction", "association_information_retention", "false_alarm_intensity", "target_count_fraction", "energy_fraction", "emission_fraction", "communication_fraction", "compute_fraction"]
    )
    return values, names


def simulated_online_scene_observation(
    state: EpisodeState,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate controller-visible scene beliefs from latent simulator state.

    The latent state is used only by this observation generator.  Downstream
    training receives a noisy kinematic estimate and a soft threat-mode
    likelihood, matching the information form used by the online controller.
    """

    rng = np.random.default_rng(int(seed))
    target = state.primary_target
    retention = float(np.clip(state.association_context.get("information_retention", 1.0), 0.0, 1.0))
    position_std = 20.0 + 180.0 * (1.0 - retention)
    velocity_std = 2.0 + 18.0 * (1.0 - retention)
    estimate = np.r_[target.position_m, target.velocity_m_s] + rng.normal(
        0.0,
        np.asarray([position_std] * 3 + [velocity_std] * 3),
    )
    correct_probability = 0.84
    mode_probability = np.full(
        len(THREAT_ORDER),
        (1.0 - correct_probability) / max(len(THREAT_ORDER) - 1, 1),
        dtype=float,
    )
    mode_probability[THREAT_ORDER.index(state.adversary.mode)] = correct_probability
    duration = float(
        np.clip(
            state.adversary.mode_age_steps / max(state.adversary.mode_duration_steps, 1)
            + rng.normal(0.0, 0.05),
            0.0,
            1.0,
        )
    )
    return estimate, mode_probability, duration


def transitions_from_trajectory(
    trajectory: Any,
    world: PhysicalWorld,
    *,
    split: str,
    observation_seed: int | None = None,
) -> tuple[WorldModelTransition, ...]:
    rows: list[WorldModelTransition] = []
    for epoch, action_ids in enumerate(trajectory.actions):
        state = trajectory.states[epoch]
        next_state = trajectory.states[epoch + 1]
        overrides: dict[str, Any] = {}
        if observation_seed is not None:
            target_estimate, mode_probability, duration = simulated_online_scene_observation(
                state,
                seed=int(observation_seed) + state.epoch * 1009,
            )
            overrides = {
                "target_mean_override_m_m_s": target_estimate,
                "threat_mode_probabilities_override": mode_probability,
                "threat_duration_fraction_override": duration,
            }
        scene, vector, _ = encode_typed_scene(state, world, action_ids, **overrides)
        observable, names = observable_from_state(next_state, world)
        rows.append(
            WorldModelTransition(
                episode_id=state.episode_id,
                epoch=state.epoch,
                scene=scene,
                target_vector=vector,
                next_observable=observable,
                next_observable_names=names,
                action_ids=tuple(action_ids),
                split=split,
                generator_version=state.generator_version,
            )
        )
    return tuple(rows)


def fit_world_model(
    transitions: Sequence[WorldModelTransition],
    *,
    config: WorldModelConfig | None = None,
    validation_transitions: Sequence[WorldModelTransition] = (),
) -> RecurrentEnsembleWorldModel:
    config = config or WorldModelConfig()
    config.validate()
    if not transitions:
        raise ValueError("world-model training requires transitions")
    if any(row.split != "train" for row in transitions):
        raise ValueError("world-model fitting accepts train transitions only")
    observable_names = transitions[0].next_observable_names
    if any(row.next_observable_names != observable_names for row in transitions):
        raise ValueError("world-model output schemas differ")
    x = np.vstack([row.target_vector for row in transitions])
    y = np.vstack([row.next_observable for row in transitions])
    if x.shape[1] <= y.shape[1]:
        raise ValueError(
            "typed-scene input does not contain the registered analytic closure baseline"
        )
    baseline_input_start = x.shape[1] - y.shape[1]
    interaction_input_start = baseline_input_start - len(INTERACTION_HISTORY_FEATURE_NAMES)
    if interaction_input_start < 0:
        raise ValueError(
            "typed-scene input does not contain interaction-history closure features"
        )
    analytic_baseline = x[:, baseline_input_start:]
    if analytic_baseline.shape != y.shape or not np.all(
        np.isfinite(analytic_baseline)
    ):
        raise ValueError("analytic closure baseline schema disagrees")
    closure_target = y - analytic_baseline
    input_mean = np.mean(x, axis=0)
    input_scale = np.std(x, axis=0)
    input_scale = np.where(input_scale > 1e-8, input_scale, 1.0)
    xn = (x - input_mean) / input_scale
    covariance = np.cov(xn, rowvar=False, ddof=0) + 0.05 * np.eye(xn.shape[1])
    inverse_covariance = np.linalg.pinv(covariance, rcond=1e-10)
    output_mean = np.mean(closure_target, axis=0)
    output_scale = np.std(closure_target, axis=0)
    output_scale = np.where(output_scale > 1e-8, output_scale, 1.0)
    yn = (closure_target - output_mean) / output_scale
    input_names = tuple(f"typed_scene_feature_{index}" for index in range(x.shape[1]))
    model = RecurrentEnsembleWorldModel(
        config=config,
        input_names=input_names,
        observable_names=observable_names,
        input_mean=input_mean,
        input_scale=input_scale,
        input_covariance_inverse=inverse_covariance,
        output_mean=output_mean,
        output_scale=output_scale,
        training_episode_ids=tuple(sorted({row.episode_id for row in transitions})),
        baseline_input_start=baseline_input_start,
        interaction_input_start=interaction_input_start,
        closure_model="analytic_constant_velocity_persistent_regime_plus_connected_cumulant_residual_with_physical_projection",
    )
    rng = np.random.default_rng(config.seed)
    episode_groups: dict[str, list[int]] = {}
    for index, row in enumerate(transitions):
        episode_groups.setdefault(row.episode_id, []).append(index)
    episode_ids = sorted(episode_groups)
    for member_index in range(config.ensemble_size):
        member_rng = np.random.default_rng(config.seed + member_index * 1009)
        recurrent_input = member_rng.normal(0.0, 1.0 / math.sqrt(xn.shape[1]), (config.hidden_dim, xn.shape[1]))
        recurrent_hidden = member_rng.normal(0.0, 0.25 / math.sqrt(config.hidden_dim), (config.hidden_dim, config.hidden_dim))
        stochastic_input = member_rng.normal(0.0, 1.0 / math.sqrt(xn.shape[1] + config.hidden_dim), (config.stochastic_dim, xn.shape[1] + config.hidden_dim))
        bootstrap_episode_count = max(1, int(math.ceil(config.bootstrap_fraction * len(episode_ids))))
        sampled_episodes = member_rng.choice(episode_ids, size=bootstrap_episode_count, replace=True)
        sample_indices = [index for episode_id in sampled_episodes for index in episode_groups[str(episode_id)]]
        hidden_by_episode: dict[str, np.ndarray] = {}
        design_rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index in sample_indices:
            row = transitions[index]
            previous = hidden_by_episode.setdefault(row.episode_id, np.zeros(config.hidden_dim))
            hidden = np.tanh(recurrent_input @ xn[index] + recurrent_hidden @ previous)
            latent = np.tanh(stochastic_input @ np.r_[xn[index], hidden])
            hidden_by_episode[row.episode_id] = hidden
            design_rows.append(np.r_[1.0, xn[index], hidden, latent])
            targets.append(yn[index])
        design = np.vstack(design_rows)
        target_matrix = np.vstack(targets)
        gram = design.T @ design + config.ridge * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ target_matrix)
        residual = target_matrix - design @ weights
        variance = np.maximum(np.mean(residual**2, axis=0), config.minimum_variance)
        model.members.append(_Member(recurrent_input, recurrent_hidden, stochastic_input, weights, np.log(variance), config.hidden_dim, config.stochastic_dim))
    training_errors: list[np.ndarray] = []
    training_hidden: dict[str, np.ndarray] = {}
    for row in transitions:
        prediction = model.predict(
            row.target_vector,
            hidden_states=training_hidden.get(row.episode_id),
        )
        training_hidden[row.episode_id] = prediction.hidden_states
        training_errors.append(row.next_observable - prediction.mean)
    model.closure_residual_training_rmse = float(
        math.sqrt(np.mean(np.vstack(training_errors) ** 2))
    )
    # Calibrate aggregate variance on temporally separate validation transitions.
    if validation_transitions:
        if any(row.split != "validation" for row in validation_transitions):
            raise ValueError("world-model calibration accepts validation transitions only")
        squared_errors: list[np.ndarray] = []
        predicted_variances: list[np.ndarray] = []
        hidden_by_episode: dict[str, np.ndarray] = {}
        for row in validation_transitions:
            hidden = hidden_by_episode.get(row.episode_id)
            prediction = model.predict(row.target_vector, hidden_states=hidden)
            hidden_by_episode[row.episode_id] = prediction.hidden_states
            squared_errors.append((row.next_observable - prediction.mean) ** 2)
            predicted_variances.append(prediction.total_variance)
        ratio = np.concatenate(squared_errors) / np.maximum(np.concatenate(predicted_variances), config.minimum_variance)
        model.calibration_scale = float(np.clip(np.median(ratio), 0.25, 16.0))
    payload = {
        "config": asdict(config),
        "input_names": input_names,
        "observable_names": observable_names,
        "training_episode_ids": model.training_episode_ids,
        "input_mean": input_mean.tolist(),
        "input_scale": input_scale.tolist(),
        "calibration_scale": model.calibration_scale,
        "baseline_input_start": model.baseline_input_start,
        "interaction_input_start": model.interaction_input_start,
        "closure_model": model.closure_model,
        "closure_residual_training_rmse": model.closure_residual_training_rmse,
    }
    model.model_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return model


def evaluate_world_model(
    model: RecurrentEnsembleWorldModel,
    transitions: Sequence[WorldModelTransition],
    *,
    split: str,
    multi_step_horizon: int = 3,
) -> WorldModelMetrics:
    if not transitions:
        raise ValueError("world-model evaluation requires transitions")
    if any(row.split != split for row in transitions):
        raise ValueError("world-model evaluation split mismatch")
    errors: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    analytic_errors: list[np.ndarray] = []
    closure_target_rows: list[np.ndarray] = []
    closure_prediction_errors: list[np.ndarray] = []
    closure_bounds: list[np.ndarray] = []
    ood: list[float] = []
    projection_correction_norms: list[float] = []
    preprojection_violations: list[float] = []
    projection_activations: list[float] = []
    connected_interaction_pressures: list[float] = []
    hidden_by_episode: dict[str, np.ndarray] = {}
    by_episode: dict[str, list[WorldModelTransition]] = {}
    for row in transitions:
        by_episode.setdefault(row.episode_id, []).append(row)
        prediction = model.predict(row.target_vector, hidden_states=hidden_by_episode.get(row.episode_id))
        hidden_by_episode[row.episode_id] = prediction.hidden_states
        errors.append(row.next_observable - prediction.mean)
        variances.append(prediction.total_variance)
        if model.baseline_input_start is None:
            baseline = np.zeros_like(row.next_observable)
        else:
            baseline = row.target_vector[
                model.baseline_input_start :
                model.baseline_input_start + len(model.observable_names)
            ]
        closure_target = row.next_observable - baseline
        closure_prediction = (
            np.asarray(prediction.closure_residual_mean, dtype=float)
            if prediction.closure_residual_mean is not None
            else prediction.mean
        )
        analytic_errors.append(row.next_observable - baseline)
        closure_target_rows.append(closure_target)
        closure_prediction_errors.append(closure_target - closure_prediction)
        if prediction.closure_error_bound is not None:
            closure_bounds.append(
                np.asarray(prediction.closure_error_bound, dtype=float)
            )
        ood.append(prediction.ood_score)
        projection_correction_norms.append(prediction.projection_correction_norm)
        preprojection_violations.append(prediction.maximum_preprojection_violation)
        projection_activations.append(float(prediction.physical_projection_applied))
        connected_interaction_pressures.append(prediction.connected_interaction_pressure)
    error = np.vstack(errors)
    variance = np.vstack(variances)
    nll = 0.5 * np.mean(np.log(2.0 * math.pi * variance) + error**2 / variance)
    standardized = np.abs(error) / np.sqrt(variance)
    coverage_50 = float(np.mean(standardized <= 0.67448975))
    coverage_90 = float(np.mean(standardized <= 1.64485363))
    ece = 0.5 * (abs(coverage_50 - 0.50) + abs(coverage_90 - 0.90))
    multi_step: list[float] = []
    teacher_forced_multi_step: list[float] = []
    analytic_multi_step: list[float] = []
    for horizon in range(1, multi_step_horizon + 1):
        horizon_errors: list[float] = []
        teacher_forced_errors: list[float] = []
        analytic_horizon_errors: list[float] = []
        for rows in by_episode.values():
            ordered = sorted(rows, key=lambda item: item.epoch)
            for start in range(0, len(ordered) - horizon + 1):
                hidden = None
                prediction: WorldModelPrediction | None = None
                teacher_hidden = None
                teacher_prediction: WorldModelPrediction | None = None
                current_input = ordered[start].target_vector.copy()
                if model.baseline_input_start is None:
                    analytic_prediction = np.zeros_like(
                        ordered[start].next_observable
                    )
                else:
                    analytic_prediction = ordered[start].target_vector[
                        model.baseline_input_start :
                        model.baseline_input_start + len(model.observable_names)
                    ].copy()
                for step in range(horizon):
                    row = ordered[start + step]
                    prediction = model.predict(current_input, hidden_states=hidden)
                    hidden = prediction.hidden_states
                    teacher_prediction = model.predict(row.target_vector, hidden_states=teacher_hidden)
                    teacher_hidden = teacher_prediction.hidden_states
                    if step + 1 < horizon:
                        current_input = counterfactual_scene_input(
                            current_input,
                            prediction,
                            row.scene,
                            ordered[start + step + 1].target_vector,
                        )
                        analytic_prediction = analytical_transition_baseline(
                            analytic_prediction,
                            dt_s=float(row.scene.dt_s),
                            observable_names=model.observable_names,
                        )
                assert prediction is not None
                assert teacher_prediction is not None
                truth = ordered[start + horizon - 1].next_observable
                horizon_errors.append(float(np.mean((truth - prediction.mean) ** 2)))
                teacher_forced_errors.append(float(np.mean((truth - teacher_prediction.mean) ** 2)))
                analytic_horizon_errors.append(
                    float(np.mean((truth - analytic_prediction) ** 2))
                )
        multi_step.append(math.sqrt(float(np.mean(horizon_errors))) if horizon_errors else float("nan"))
        teacher_forced_multi_step.append(math.sqrt(float(np.mean(teacher_forced_errors))) if teacher_forced_errors else float("nan"))
        analytic_multi_step.append(
            math.sqrt(float(np.mean(analytic_horizon_errors)))
            if analytic_horizon_errors
            else float("nan")
        )
    analytic_error = np.vstack(analytic_errors)
    closure_target_matrix = np.vstack(closure_target_rows)
    closure_prediction_error = np.vstack(closure_prediction_errors)
    analytic_rmse = float(math.sqrt(np.mean(analytic_error**2)))
    learned_rmse = float(math.sqrt(np.mean(error**2)))
    return WorldModelMetrics(
        sample_count=len(transitions),
        one_step_rmse=float(math.sqrt(np.mean(error**2))),
        negative_log_likelihood=float(nll),
        coverage_50=coverage_50,
        coverage_90=coverage_90,
        expected_calibration_error=float(ece),
        mean_ood_score=float(np.mean(ood)),
        maximum_ood_score=float(np.max(ood)),
        multi_step_rmse_by_horizon=tuple(multi_step),
        teacher_forced_multi_step_rmse_by_horizon=tuple(teacher_forced_multi_step),
        analytic_multi_step_rmse_by_horizon=tuple(analytic_multi_step),
        analytic_one_step_rmse=analytic_rmse,
        learned_closure_residual_rmse=float(
            math.sqrt(np.mean(closure_prediction_error**2))
        ),
        closure_target_rms=float(math.sqrt(np.mean(closure_target_matrix**2))),
        mean_predicted_closure_error_bound=float(
            np.mean(np.vstack(closure_bounds))
        )
        if closure_bounds
        else float("nan"),
        closure_gain_over_analytic=analytic_rmse - learned_rmse,
        open_loop_uses_future_environment_state=False,
        calibration_scale=model.calibration_scale,
        split=split,
        mean_projection_correction_norm=float(np.mean(projection_correction_norms)),
        maximum_preprojection_violation=float(np.max(preprojection_violations)),
        physical_projection_activation_rate=float(np.mean(projection_activations)),
        mean_connected_interaction_pressure=float(np.mean(connected_interaction_pressures)),
    )


def analytical_transition_baseline(
    current_observable: np.ndarray,
    *,
    dt_s: float,
    observable_names: Sequence[str],
) -> np.ndarray:
    """Non-neural constant-velocity/persistent-regime baseline."""

    value = np.asarray(current_observable, dtype=float).copy()
    index = {name: i for i, name in enumerate(observable_names)}
    for axis in "xyz":
        p = index.get(f"target_position_{axis}")
        v = index.get(f"target_velocity_{axis}")
        if p is not None and v is not None:
            # Position and velocity are scaled by 10 km and 300 m/s.
            value[p] += value[v] * 300.0 * dt_s / 10_000.0
    return value


def ood_overconfidence_negative_control(
    model: RecurrentEnsembleWorldModel,
    in_distribution_input: np.ndarray,
    *,
    displacement_sigma: float = 20.0,
) -> dict[str, Any]:
    in_prediction = model.predict(in_distribution_input)
    assert model.input_scale is not None
    ood_input = np.asarray(in_distribution_input, dtype=float) + displacement_sigma * model.input_scale
    out_prediction = model.predict(ood_input)
    passed = bool(
        out_prediction.ood_score > in_prediction.ood_score
        and float(np.mean(out_prediction.total_variance)) > float(np.mean(in_prediction.total_variance))
        and out_prediction.rollout_truncated
    )
    return {
        "status": "passed" if passed else "failed",
        "in_distribution_ood_score": in_prediction.ood_score,
        "out_of_distribution_ood_score": out_prediction.ood_score,
        "in_distribution_mean_variance": float(np.mean(in_prediction.total_variance)),
        "out_of_distribution_mean_variance": float(np.mean(out_prediction.total_variance)),
        "out_of_distribution_rollout_truncated": out_prediction.rollout_truncated,
        "negative_control": "would fail if OOD uncertainty inflation were removed",
    }
