from __future__ import annotations

"""V65 long-horizon, safe, history-dependent radar autonomy.

The operational loop is

    posterior belief -> candidate proposal -> learned/analytic prediction ->
    exact likelihood/Schur/risk rerank -> safety shield -> world transition ->
    delayed observation -> Bayesian update/change detection.

No learned component can bypass ``evaluate_schedule`` or the campaign-resource
shield.  The finite candidate domain supports an explicit backup invariant; no
global stability claim is made.
"""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from wideband_info_geometry.learning.v65_world_model import (
    RecurrentEnsembleWorldModel,
    counterfactual_scene_input,
    encode_typed_scene,
)

from .adversary import canonical_adversary_scenarios
from .autonomous_control import ThreatMemory, bayes_update_with_likelihood, tactic_information_gain
from .cognitive_ew import HiddenTacticModel, build_hidden_tactic_model, tactic_identifier
from .dynamic_episode import (
    CampaignResources,
    EpisodeConfig,
    EpisodeState,
    ThreatMode,
    action_is_dynamically_feasible,
    episode_state_to_world,
    initial_episode_state,
    transition_episode_state,
)
from .information import (
    DecisionPolicy,
    evaluate_schedule,
    prior_information_normalized,
    propagate_normalized_covariance,
    schedule_feasibility,
    schedule_resource_usage,
    schur_effective_information,
    stable_inverse_spd,
)
from .measurement import (
    StateLayout,
    build_action_factor,
    nonlinear_measurement_delta,
    normalized_world_state_error,
)
from .schema import Jammer, PhysicalWorld, RadarAction
from .stochastic_game import (
    GameRewardTerms,
    IndependentBeliefQPolicy,
    OnlineBeliefTreeSearchConfig,
    OnlineOpponentModel,
    exact_belief_control,
    lower_tail_mean,
    scalable_belief_control,
)

PolicyKind = Literal[
    "fixed_policy",
    "static_minimax",
    "myopic_belief",
    "belief_mpc",
    "world_model_mpc",
    "independent_rl",
    "random",
    "clairvoyant_reference",
]


@dataclass(frozen=True)
class MissionIR:
    mission_id: str
    position_weight: float = 1.0
    velocity_weight: float = 0.30
    tail_risk_weight: float = 0.50
    threat_information_weight: float = 0.10
    switching_cost_weight: float = 0.08
    future_resource_weight: float = 0.12
    world_model_future_weight: float = 0.08
    world_model_uncertainty_weight: float = 0.05
    forbidden_action_kinds: tuple[str, ...] = ()
    forbidden_action_ids: tuple[int, ...] = ()
    maximum_energy_per_epoch_j: float | None = None
    maximum_emission_per_epoch_j: float | None = None
    maximum_communication_per_epoch_mb: float | None = None
    minimum_energy_reserve_j: float = 0.0
    minimum_emission_reserve_j: float = 0.0
    require_terminal_backup: bool = True
    maximum_decision_latency_s: float = 0.75
    target_classes: tuple[str, ...] = ("low_altitude_air_target",)
    #: A usable update means that a target-dependent measurement is delivered
    #: and associated with the tracked target.  These probabilities define the
    #: mission-level continuity requirement in each causal track phase.
    minimum_update_probability_acquisition: float = 0.0
    minimum_update_probability_confirmed: float = 0.0
    minimum_update_probability_reacquisition: float = 0.0
    #: Absolute probability difference below which two *infeasible* schedules
    #: have no registered mission distinction.  A feasible schedule always
    #: retains priority over an infeasible one.  The band prevents floating-
    #: point-sized forecast differences from acting as an unstated utility
    #: function when every candidate misses the chance constraint.
    track_update_probability_indifference_band: float = 0.0
    track_update_shortfall_weight: float = 0.0
    confirmation_min_distinct_detection_epochs: int = 2
    confirmation_radius_fraction_of_initial: float = 0.80
    #: Optional truth-isolated directional confirmation certificate.  The
    #: controller computes the largest semi-axis of the Gaussian position
    #: credible ellipsoid from the sole graph posterior.  When this limit is
    #: present, relative D-optimal contraction alone cannot confirm a track.
    confirmation_max_position_ellipsoid_semi_axis_m: float | None = None
    confirmation_position_credible_probability: float = 0.95
    loss_radius_fraction_of_initial: float = 1.25
    maximum_coast_steps: int = 2

    def validate(self) -> None:
        if not self.mission_id or not self.target_classes:
            raise ValueError("mission identity and target classes are required")
        nonnegative = (
            self.position_weight,
            self.velocity_weight,
            self.tail_risk_weight,
            self.threat_information_weight,
            self.switching_cost_weight,
            self.future_resource_weight,
            self.world_model_future_weight,
            self.world_model_uncertainty_weight,
            self.minimum_energy_reserve_j,
            self.minimum_emission_reserve_j,
            self.track_update_probability_indifference_band,
            self.track_update_shortfall_weight,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in nonnegative):
            raise ValueError("mission weights/reserves must be finite and nonnegative")
        for value in (
            self.maximum_energy_per_epoch_j,
            self.maximum_emission_per_epoch_j,
            self.maximum_communication_per_epoch_mb,
        ):
            if value is not None and value < 0.0:
                raise ValueError("mission resource maxima must be nonnegative")
        if self.maximum_decision_latency_s <= 0.0:
            raise ValueError("mission decision deadline must be positive")
        probabilities = (
            self.minimum_update_probability_acquisition,
            self.minimum_update_probability_confirmed,
            self.minimum_update_probability_reacquisition,
        )
        if any(
            not 0.0 <= value <= 1.0 or not math.isfinite(value)
            for value in probabilities
        ):
            raise ValueError("track-update probability requirements must lie in [0, 1]")
        if self.track_update_probability_indifference_band > 1.0:
            raise ValueError(
                "track-update probability indifference band must lie in [0, 1]"
            )
        if self.confirmation_min_distinct_detection_epochs <= 0:
            raise ValueError("track confirmation requires a positive detection-epoch count")
        if (
            self.confirmation_radius_fraction_of_initial <= 0.0
            or self.loss_radius_fraction_of_initial <= 0.0
            or not math.isfinite(self.confirmation_radius_fraction_of_initial)
            or not math.isfinite(self.loss_radius_fraction_of_initial)
        ):
            raise ValueError("track-phase radius fractions must be finite and positive")
        if (
            self.confirmation_max_position_ellipsoid_semi_axis_m is not None
            and (
                self.confirmation_max_position_ellipsoid_semi_axis_m <= 0.0
                or not math.isfinite(
                    self.confirmation_max_position_ellipsoid_semi_axis_m
                )
            )
        ):
            raise ValueError(
                "position confirmation ellipsoid limit must be finite and positive"
            )
        if not (
            math.isfinite(self.confirmation_position_credible_probability)
            and 0.0 < self.confirmation_position_credible_probability < 1.0
        ):
            raise ValueError(
                "position confirmation credible probability must lie in (0, 1)"
            )
        if (
            self.loss_radius_fraction_of_initial
            < self.confirmation_radius_fraction_of_initial
        ):
            raise ValueError("track-loss radius must not be tighter than confirmation")
        if self.maximum_coast_steps < 0:
            raise ValueError("maximum coast steps must be nonnegative")


def compile_mission_ir(specification: Mapping[str, Any]) -> MissionIR:
    """Compile a typed goal contract; free-form language never reaches control."""

    allowed = set(MissionIR.__dataclass_fields__)
    unknown = sorted(set(specification).difference(allowed))
    if unknown:
        raise ValueError(f"unknown mission fields: {unknown}")
    payload = dict(specification)
    for key in ("forbidden_action_kinds", "forbidden_action_ids", "target_classes"):
        if key in payload:
            payload[key] = tuple(payload[key])
    mission = MissionIR(**payload)
    mission.validate()
    return mission


@dataclass(frozen=True)
class CandidateLibrary:
    schedules: tuple[tuple[int, ...], ...]
    generation_world_id: str
    generation_hash: str
    role_by_candidate: tuple[tuple[str, ...], ...]
    exact_bounded_domain: bool = True

    def validate(self, world: PhysicalWorld) -> None:
        if not self.schedules or len(set(self.schedules)) != len(self.schedules):
            raise ValueError("candidate library must be nonempty and unique")
        by_id = {action.action_id: action for action in world.actions}
        for schedule in self.schedules:
            if any(action_id not in by_id for action_id in schedule):
                raise ValueError("candidate library references unknown action")


def _action_role(action: RadarAction) -> str:
    if action.kind.startswith("active"):
        return "active_transmitter"
    if action.kind == "passive_bistatic":
        return "illuminator_reference_processor"
    if action.kind.startswith("calibration"):
        return "calibration_agent"
    return "passive_receiver"


def build_dynamic_candidate_library(
    world: PhysicalWorld,
    state: EpisodeState,
    *,
    mission: MissionIR | None = None,
    maximum_candidates: int = 9,
) -> CandidateLibrary:
    """Build a world-specific bounded domain with active/passive/calibration coverage."""

    mission = mission or MissionIR("default")
    mission.validate()
    dynamic_world = episode_state_to_world(world, state)
    eligible = [
        action
        for action in dynamic_world.actions
        if action.allowed
        and action.kind not in mission.forbidden_action_kinds
        and action.action_id not in mission.forbidden_action_ids
    ]
    if not eligible:
        schedules = ((),)
        roles = ((),)
    else:
        tactics = canonical_adversary_scenarios(dynamic_world)
        singleton_rows: list[tuple[float, RadarAction]] = []
        for action in eligible:
            usage = schedule_resource_usage(dynamic_world, (action,))
            if mission.maximum_energy_per_epoch_j is not None and usage.energy_j > mission.maximum_energy_per_epoch_j:
                continue
            if mission.maximum_emission_per_epoch_j is not None and usage.emission_exposure_j > mission.maximum_emission_per_epoch_j:
                continue
            if mission.maximum_communication_per_epoch_mb is not None and usage.communication_mb > mission.maximum_communication_per_epoch_mb:
                continue
            # Candidate generation is a proposal stage.  Use the exact current
            # likelihood/Schur factor plus explicit resource cost here and
            # reserve the more expensive multi-tactic risk evaluation for the
            # final shortlisted rerank.  This preserves the physics contract
            # while avoiding redundant full schedule evaluations.
            factor = build_action_factor(dynamic_world, action)
            posterior = prior_information_normalized(dynamic_world) + factor.observed_information_normalized
            effective, _ = schur_effective_information(posterior)
            information_proxy = float(np.trace(effective))
            cost_proxy = (
                usage.energy_j / max(dynamic_world.max_total_energy_j, 1e-12)
                + usage.emission_exposure_j / max(dynamic_world.max_total_emission_j, 1e-12)
                + usage.communication_mb / max(dynamic_world.max_total_communication_mb, 1e-12)
            )
            singleton_rows.append((information_proxy - 0.05 * cost_proxy, action))
        singleton_rows.sort(key=lambda item: (-item[0], item[1].action_id))
        selected: list[tuple[int, ...]] = []
        # Guarantee role coverage before filling by utility.
        for role in ("active_transmitter", "passive_receiver", "illuminator_reference_processor", "calibration_agent"):
            match = next((action for _, action in singleton_rows if _action_role(action) == role), None)
            if match is not None:
                selected.append((match.action_id,))
        selected.extend((action.action_id,) for _, action in singleton_rows[: max(4, maximum_candidates // 2)])
        top = [action for _, action in singleton_rows[:6]]
        for left_index, left in enumerate(top):
            for right in top[left_index + 1 :]:
                if _action_role(left) == _action_role(right) and left.resource_slot == right.resource_slot:
                    continue
                ids = tuple(sorted((left.action_id, right.action_id)))
                actions = [left, right]
                feasible, _ = schedule_feasibility(dynamic_world, actions)
                dynamic_feasible, _ = action_is_dynamically_feasible(world, state, ids)
                if feasible and dynamic_feasible:
                    selected.append(ids)
        # Stable deduplication; a no-op is a last-resort shield action, not a normal proposal.
        schedules_list: list[tuple[int, ...]] = []
        for schedule in selected:
            if schedule not in schedules_list:
                schedules_list.append(schedule)
            if len(schedules_list) >= maximum_candidates:
                break
        schedules = tuple(schedules_list or [()])
        by_id = {action.action_id: action for action in dynamic_world.actions}
        roles = tuple(tuple(sorted({_action_role(by_id[action_id]) for action_id in schedule})) for schedule in schedules)
    payload = json.dumps({"world": dynamic_world.world_id, "schedules": schedules, "epoch": state.epoch}, sort_keys=True)
    library = CandidateLibrary(schedules, dynamic_world.world_id, hashlib.sha256(payload.encode()).hexdigest(), roles)
    library.validate(dynamic_world)
    return library


@dataclass(frozen=True)
class ShieldDecision:
    proposed_candidate: int
    selected_candidate: int
    selected_action_ids: tuple[int, ...]
    intervened: bool
    reasons: tuple[str, ...]
    backup_candidate: int
    recursive_feasibility_verified: bool


class CampaignSafetyShield:
    def __init__(self, world: PhysicalWorld, library: CandidateLibrary, mission: MissionIR) -> None:
        self.world = world
        self.library = library
        self.mission = mission
        mission.validate()

    def _usage(self, dynamic_world: PhysicalWorld, candidate: int):
        by_id = {action.action_id: action for action in dynamic_world.actions}
        return schedule_resource_usage(dynamic_world, [by_id[action_id] for action_id in self.library.schedules[candidate]])

    def safe_mask(self, state: EpisodeState, *, remaining_epochs: int) -> np.ndarray:
        dynamic_world = episode_state_to_world(self.world, state)
        mask = np.zeros(len(self.library.schedules), dtype=bool)
        for index, ids in enumerate(self.library.schedules):
            feasible, _ = action_is_dynamically_feasible(self.world, state, ids)
            if not feasible:
                continue
            by_id = {action.action_id: action for action in dynamic_world.actions}
            actions = [by_id[action_id] for action_id in ids]
            base_feasible, _ = schedule_feasibility(dynamic_world, actions)
            if not base_feasible:
                continue
            usage = schedule_resource_usage(dynamic_world, actions)
            if self.mission.maximum_energy_per_epoch_j is not None and usage.energy_j > self.mission.maximum_energy_per_epoch_j:
                continue
            if self.mission.maximum_emission_per_epoch_j is not None and usage.emission_exposure_j > self.mission.maximum_emission_per_epoch_j:
                continue
            if self.mission.maximum_communication_per_epoch_mb is not None and usage.communication_mb > self.mission.maximum_communication_per_epoch_mb:
                continue
            if state.resources.energy_remaining_j - usage.energy_j < max(state.resources.reserve_energy_j, self.mission.minimum_energy_reserve_j) - 1e-9:
                continue
            if state.resources.emission_remaining_j - usage.emission_exposure_j < max(state.resources.reserve_emission_j, self.mission.minimum_emission_reserve_j) - 1e-9:
                continue
            mask[index] = True
        if self.mission.require_terminal_backup and np.any(mask):
            backup = self.backup_candidate(state, mask=mask)
            backup_usage = self._usage(dynamic_world, backup)
            for index in np.flatnonzero(mask):
                usage = self._usage(dynamic_world, int(index))
                epochs = max(0, int(remaining_epochs))
                keep = (
                    usage.energy_j + epochs * backup_usage.energy_j <= state.resources.energy_remaining_j + 1e-9
                    and usage.emission_exposure_j + epochs * backup_usage.emission_exposure_j <= state.resources.emission_remaining_j + 1e-9
                    and usage.communication_mb + epochs * backup_usage.communication_mb <= state.resources.communication_remaining_mb + 1e-9
                )
                mask[index] = bool(keep)
        return mask

    def backup_candidate(self, state: EpisodeState, *, mask: np.ndarray | None = None) -> int:
        dynamic_world = episode_state_to_world(self.world, state)
        available = np.ones(len(self.library.schedules), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        rows: list[tuple[float, int]] = []
        for index in np.flatnonzero(available):
            usage = self._usage(dynamic_world, int(index))
            active_penalty = sum(
                dynamic_world.action(action_id).kind.startswith("active") if hasattr(dynamic_world, "action") else next(a for a in dynamic_world.actions if a.action_id == action_id).kind.startswith("active")
                for action_id in self.library.schedules[int(index)]
            )
            normalized = (
                usage.energy_j / max(dynamic_world.max_total_energy_j, 1e-12)
                + usage.emission_exposure_j / max(dynamic_world.max_total_emission_j, 1e-12)
                + usage.communication_mb / max(dynamic_world.max_total_communication_mb, 1e-12)
                + 0.05 * active_penalty
            )
            rows.append((float(normalized), int(index)))
        if not rows:
            raise RuntimeError("no safe backup candidate exists")
        return min(rows)[1]

    def apply(self, proposed: int, state: EpisodeState, *, remaining_epochs: int) -> ShieldDecision:
        mask = self.safe_mask(state, remaining_epochs=remaining_epochs)
        if not np.any(mask):
            # Fail closed: execute no RF/communication action. The transition still advances.
            return ShieldDecision(int(proposed), -1, (), True, ("no_registered_candidate_preserves_campaign_invariant",), -1, True)
        backup = self.backup_candidate(state, mask=mask)
        if 0 <= proposed < len(mask) and mask[proposed]:
            return ShieldDecision(int(proposed), int(proposed), self.library.schedules[int(proposed)], False, (), backup, True)
        reasons = ("proposal_not_dynamically_or_recursively_feasible",)
        return ShieldDecision(int(proposed), backup, self.library.schedules[backup], True, reasons, backup, True)


@dataclass
class ControllerBelief:
    covariance_normalized: np.ndarray
    target_mean_m_m_s: np.ndarray
    threat_belief: np.ndarray
    threat_memory: ThreatMemory
    opponent_model: OnlineOpponentModel
    previous_candidate: int = -1
    previous_action_ids: tuple[int, ...] = ()
    topology_signature: tuple[tuple[str, str], ...] = ()
    node_health_signature: tuple[str, ...] = ()
    change_alarm: bool = False
    change_alarm_reason: str = "none"
    change_alarm_steps: list[int] = field(default_factory=list)
    world_model_hidden_states: np.ndarray | None = None
    last_world_prediction: Any | None = None
    estimate_history: list[np.ndarray] = field(default_factory=list)
    inferred_tactic_age_steps: int = 1


@dataclass(frozen=True)
class ControllerConfig:
    horizon: int = 3
    discount: float = 0.94
    shortlist_size: int = 3
    replanning_interval_steps: int = 2
    maximum_candidates: int = 8
    seed: int = 6502
    compute_deadline_s: float = 0.75
    belief_solver: Literal["auto", "exact", "online_pomcp"] = "auto"
    exact_belief_horizon_limit: int = 4
    belief_search_simulations: int = 4096
    belief_search_evaluation_rollouts_per_action: int = 128
    belief_search_exploration_constant: float = 1.35

    def validate(self) -> None:
        if min(
            self.horizon,
            self.shortlist_size,
            self.replanning_interval_steps,
            self.maximum_candidates,
            self.exact_belief_horizon_limit,
            self.belief_search_simulations,
            self.belief_search_evaluation_rollouts_per_action,
        ) <= 0:
            raise ValueError("controller dimensions must be positive")
        if not 0.0 <= self.discount <= 1.0 or self.compute_deadline_s <= 0.0:
            raise ValueError("controller discount/deadline is invalid")
        if self.belief_solver not in {"auto", "exact", "online_pomcp"}:
            raise ValueError("controller belief solver is invalid")
        if not math.isfinite(self.belief_search_exploration_constant) or self.belief_search_exploration_constant < 0.0:
            raise ValueError("controller belief-search exploration constant is invalid")


def _mode_observation_index(mode: ThreatMode, rng: np.random.Generator, noise_probability: float = 0.12) -> int:
    canonical = {
        ThreatMode.QUIET: 0,
        ThreatMode.BARRAGE: 1,
        ThreatMode.SWEPT: 1,
        ThreatMode.SPOT: 2,
        ThreatMode.DRFM: 3,
        ThreatMode.REFERENCE_CONTAMINATION: 3,
        ThreatMode.INTERMITTENT_DECEPTION: 3,
    }[mode]
    return int(rng.integers(4)) if rng.random() < noise_probability else canonical


def _map_mode_to_tactic(mode: ThreatMode, tactics: Sequence[Jammer | None]) -> int:
    desired = {
        ThreatMode.QUIET: "nominal",
        ThreatMode.BARRAGE: "barrage",
        ThreatMode.SPOT: "spot",
        ThreatMode.SWEPT: "swept",
        ThreatMode.DRFM: "drfm",
        ThreatMode.REFERENCE_CONTAMINATION: "drfm",
        ThreatMode.INTERMITTENT_DECEPTION: "drfm",
    }[mode]
    for index, tactic in enumerate(tactics):
        identifier = tactic_identifier(tactic).lower()
        if desired in identifier:
            return index
    return 0


def _candidate_payoff_matrix(world: PhysicalWorld, library: CandidateLibrary, tactics: Sequence[Jammer | None]) -> np.ndarray:
    matrix = np.empty((len(library.schedules), len(tactics)), dtype=float)
    for candidate, ids in enumerate(library.schedules):
        evaluation = evaluate_schedule(
            world,
            ids,
            adversary_scenarios=tactics,
            policy=DecisionPolicy(cvar_weight=0.0),
        )
        scenario = np.asarray(evaluation.scenario_information_gains, dtype=float)
        if scenario.shape != (len(tactics),):
            raise ValueError("candidate tactic payoff dimension disagrees")
        matrix[candidate] = scenario - float(evaluation.decision_cost)
    return matrix


def candidate_tactic_payoff_matrix(
    world: PhysicalWorld,
    library: CandidateLibrary,
    tactics: Sequence[Jammer | None],
) -> np.ndarray:
    """Public exact finite-game payoff adapter used by training/evaluation."""

    library.validate(world)
    return _candidate_payoff_matrix(world, library, tactics)


def _resource_fraction(resources: CampaignResources) -> float:
    energy = (resources.energy_remaining_j - resources.reserve_energy_j) / max(resources.energy_remaining_j, 1.0)
    emission = (resources.emission_remaining_j - resources.reserve_emission_j) / max(resources.emission_remaining_j, 1.0)
    return float(np.clip(min(energy, emission), 0.0, 1.0))


def _normalized_world_error(reference: PhysicalWorld, truth: PhysicalWorld, layout: StateLayout) -> np.ndarray:
    """Return the simulated truth-minus-filter-linearization state in normalized units."""

    return normalized_world_state_error(reference, truth, layout)


class SafeAutonomyController:
    def __init__(
        self,
        world: PhysicalWorld,
        initial_state: EpisodeState,
        *,
        mission: MissionIR,
        policy_kind: PolicyKind = "belief_mpc",
        config: ControllerConfig | None = None,
        world_model: RecurrentEnsembleWorldModel | None = None,
        rl_policy: IndependentBeliefQPolicy | None = None,
        candidate_library: CandidateLibrary | None = None,
        base_payoff: np.ndarray | None = None,
    ) -> None:
        self.world = world
        self.mission = mission
        self.policy_kind = policy_kind
        self.config = config or ControllerConfig(compute_deadline_s=mission.maximum_decision_latency_s)
        self.config.validate()
        initial_world = episode_state_to_world(world, initial_state)
        self.library = candidate_library or build_dynamic_candidate_library(
            world,
            initial_state,
            mission=mission,
            maximum_candidates=self.config.maximum_candidates,
        )
        self.library.validate(initial_world)
        self.shield = CampaignSafetyShield(world, self.library, mission)
        self.tactics = tuple(canonical_adversary_scenarios(initial_world))
        if base_payoff is None:
            self.base_payoff = _candidate_payoff_matrix(initial_world, self.library, self.tactics)
        else:
            self.base_payoff = np.asarray(base_payoff, dtype=float).copy()
            if self.base_payoff.shape != (len(self.library.schedules), len(self.tactics)) or not np.all(np.isfinite(self.base_payoff)):
                raise ValueError("precomputed candidate payoff has the wrong finite-game contract")
        self.tactic_model = build_hidden_tactic_model(self.base_payoff, [tactic_identifier(value) for value in self.tactics])
        layout = StateLayout.from_world(initial_world)
        covariance = stable_inverse_spd(prior_information_normalized(initial_world, layout), name="v65_initial_information")
        target_mean = np.r_[initial_state.primary_target.position_m, initial_state.primary_target.velocity_m_s]
        initial_threat = np.full(len(self.tactics), 1.0 / len(self.tactics))
        self.belief = ControllerBelief(
            covariance,
            target_mean,
            initial_threat,
            ThreatMemory.initialize(len(self.tactics), initial_belief=initial_threat),
            OnlineOpponentModel(len(self.tactics), len(self.library.schedules)),
            topology_signature=initial_state.communication_edges,
            node_health_signature=tuple(node.health.value for node in initial_state.nodes),
        )
        self.world_model = world_model
        self.world_model_controller_enabled = bool(
            world_model is not None and world_model.fitted and world_model.controller_qualified
        )
        self.rl_policy = rl_policy
        if policy_kind == "world_model_mpc" and world_model is not None and not world_model.fitted:
            raise ValueError("world_model_mpc received an unfitted world model")
        if policy_kind == "independent_rl" and rl_policy is None:
            raise ValueError("independent_rl requires a separately trained policy")
        if rl_policy is not None and rl_policy.action_count != len(self.library.schedules):
            raise ValueError("RL policy action domain differs from controller candidate library")
        static_min = np.min(self.base_payoff, axis=1)
        self.static_minimax_candidate = int(np.argmax(static_min))
        self.fixed_candidate = int(np.argmax(self.base_payoff @ initial_threat))
        self.records: list[dict[str, Any]] = []

    def _threat_mode_belief(self) -> np.ndarray:
        probability = np.zeros(len(ThreatMode), dtype=float)
        index = {mode: position for position, mode in enumerate(ThreatMode)}
        for tactic_probability, tactic in zip(self.belief.threat_belief, self.tactics):
            identifier = tactic_identifier(tactic).lower()
            if "nominal" in identifier:
                mode = ThreatMode.QUIET
            elif "barrage" in identifier:
                mode = ThreatMode.BARRAGE
            elif "spot" in identifier:
                mode = ThreatMode.SPOT
            elif "swept" in identifier:
                mode = ThreatMode.SWEPT
            elif "drfm" in identifier:
                mode = ThreatMode.DRFM
            else:
                mode = ThreatMode.INTERMITTENT_DECEPTION
            probability[index[mode]] += float(tactic_probability)
        if float(np.sum(probability)) <= 0.0:
            probability[index[ThreatMode.QUIET]] = 1.0
        return probability / np.sum(probability)

    def _belief_world(self, state: EpisodeState) -> PhysicalWorld:
        dynamic_world = episode_state_to_world(self.world, state)
        return replace(
            dynamic_world,
            target=replace(
                dynamic_world.target,
                position_m=tuple(float(value) for value in self.belief.target_mean_m_m_s[:3]),
                velocity_m_s=tuple(float(value) for value in self.belief.target_mean_m_m_s[3:]),
            ),
        )

    def _dynamic_proposal_scores(self, state: EpisodeState, safe_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        matrix = self.base_payoff.copy()
        mean_occupancy = float(np.mean(list(state.spectrum.occupancy_by_waveform.values())))
        reference = float(np.mean(list(state.spectrum.reference_quality_by_waveform.values())))
        health = sum(node.health.value != "failed" for node in state.nodes) / max(len(state.nodes), 1)
        # Observable state modifies a registered baseline, not the exact final score.
        for index, roles in enumerate(self.library.role_by_candidate):
            if "active_transmitter" in roles:
                matrix[index] -= 0.35 * mean_occupancy
            if "illuminator_reference_processor" in roles or "passive_receiver" in roles:
                matrix[index] += 0.25 * reference
            matrix[index] += 0.15 * health
        metadata: dict[str, Any] = {}
        if self.policy_kind == "fixed_policy":
            scores = np.full(len(self.library.schedules), -math.inf)
            scores[self.fixed_candidate] = 1.0
        elif self.policy_kind == "static_minimax":
            scores = np.min(matrix, axis=1)
        elif self.policy_kind == "myopic_belief":
            scores = matrix @ self.belief.threat_belief
        elif self.policy_kind in {"belief_mpc", "world_model_mpc"}:
            adaptive_model = replace(self.tactic_model, transition_by_radar_action=self.belief.opponent_model.transition())
            control = scalable_belief_control(
                matrix,
                adaptive_model,
                self.belief.threat_belief,
                horizon=self.config.horizon,
                discount=self.config.discount,
                safety_mask=safe_mask,
                solver=self.config.belief_solver,
                exact_horizon_limit=self.config.exact_belief_horizon_limit,
                online_config=OnlineBeliefTreeSearchConfig(
                    simulations=self.config.belief_search_simulations,
                    evaluation_rollouts_per_action=self.config.belief_search_evaluation_rollouts_per_action,
                    exploration_constant=self.config.belief_search_exploration_constant,
                    seed=self.config.seed + state.epoch * 100_003,
                ),
            )
            scores = np.asarray(control["q_values"], dtype=float)
            metadata["belief_mpc"] = control
        elif self.policy_kind == "independent_rl":
            assert self.rl_policy is not None
            index = self.rl_policy.state_index(self.belief.threat_belief, _resource_fraction(state.resources), self.belief.previous_candidate)
            scores = self.rl_policy.q_values[index].copy()
        elif self.policy_kind == "random":
            rng = np.random.default_rng(self.config.seed + state.epoch * 7919)
            scores = rng.random(len(self.library.schedules))
        elif self.policy_kind == "clairvoyant_reference":
            tactic = _map_mode_to_tactic(state.adversary.mode, self.tactics)
            scores = matrix[:, tactic]
        else:
            raise ValueError(f"unknown policy kind {self.policy_kind}")
        if self.policy_kind == "world_model_mpc" and self.world_model_controller_enabled:
            assert self.world_model is not None
            future_bonus = np.full(len(self.library.schedules), -math.inf)
            predictions: dict[int, Any] = {}
            rollout_lengths: dict[int, int] = {}
            for index in np.flatnonzero(safe_mask):
                schedule = self.library.schedules[int(index)]
                dynamic_world = self._belief_world(state)
                scene, encoded, _ = encode_typed_scene(
                    state,
                    dynamic_world,
                    schedule,
                    target_mean_override_m_m_s=self.belief.target_mean_m_m_s,
                    threat_mode_probabilities_override=self._threat_mode_belief(),
                    threat_duration_fraction_override=0.0,
                )
                current = encoded
                hidden = None if self.belief.world_model_hidden_states is None else self.belief.world_model_hidden_states.copy()
                rollout: list[Any] = []
                for rollout_step in range(max(1, self.config.horizon)):
                    prediction = self.world_model.predict(
                        current,
                        hidden_states=hidden,
                        seed=self.config.seed + state.epoch * 100_003 + int(index) * 1009 + rollout_step,
                        stochastic=True,
                    )
                    rollout.append(prediction)
                    hidden = prediction.hidden_states
                    if prediction.rollout_truncated:
                        break
                    current = counterfactual_scene_input(current, prediction, scene, encoded)
                if not rollout:
                    continue
                discounted_favorable = 0.0
                discounted_uncertainty = 0.0
                for rollout_step, prediction in enumerate(rollout):
                    names = {name: position for position, name in enumerate(prediction.observable_names)}
                    favorable = 0.0
                    for name in (
                        "reference_quality",
                        "connectivity",
                        "node_health_fraction",
                        "association_information_retention",
                        "energy_fraction",
                        "emission_fraction",
                    ):
                        if name in names:
                            favorable += float(prediction.mean[names[name]])
                    for name in ("spectrum_occupancy", "false_alarm_intensity"):
                        if name in names:
                            favorable -= float(prediction.mean[names[name]])
                    discount = self.config.discount**rollout_step
                    discounted_favorable += discount * favorable
                    discounted_uncertainty += discount * float(np.mean(np.sqrt(prediction.total_variance)))
                bonus = (
                    self.mission.world_model_future_weight * discounted_favorable
                    - self.mission.world_model_uncertainty_weight * discounted_uncertainty
                )
                if rollout[-1].rollout_truncated:
                    bonus -= 1.0
                future_bonus[int(index)] = bonus
                predictions[int(index)] = rollout[-1]
                rollout_lengths[int(index)] = len(rollout)
            scores = scores + future_bonus
            metadata["world_model_future_bonus"] = future_bonus.tolist()
            metadata["world_model_rollout_lengths"] = rollout_lengths
            metadata["world_model_predictions"] = predictions
            metadata["future_environment_state_read"] = False
            metadata["world_model_controller_qualified"] = True
        elif self.policy_kind == "world_model_mpc":
            metadata["world_model_controller_qualified"] = False
            metadata["world_model_fallback"] = "belief_mpc_without_learned_future_bonus"
            metadata["future_environment_state_read"] = False
        return np.where(safe_mask, scores, -math.inf), metadata

    def _exact_rerank(
        self,
        state: EpisodeState,
        proposal_scores: np.ndarray,
        safe_mask: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> tuple[int, list[dict[str, Any]]]:
        dynamic_world = self._belief_world(state)
        safe_indices = np.flatnonzero(safe_mask)
        shortlist = safe_indices[np.argsort(proposal_scores[safe_indices])[::-1][: self.config.shortlist_size]]
        rows: list[dict[str, Any]] = []
        for index in shortlist:
            ids = self.library.schedules[int(index)]
            evaluation = evaluate_schedule(dynamic_world, ids, adversary_scenarios=self.tactics)
            usage = schedule_resource_usage(
                dynamic_world,
                [next(action for action in dynamic_world.actions if action.action_id == action_id) for action_id in ids],
            )
            switching = self.mission.switching_cost_weight * len(set(ids).symmetric_difference(self.belief.previous_action_ids))
            remaining_energy_fraction = (state.resources.energy_remaining_j - usage.energy_j - state.resources.reserve_energy_j) / max(state.resources.energy_remaining_j, 1.0)
            remaining_emission_fraction = (state.resources.emission_remaining_j - usage.emission_exposure_j - state.resources.reserve_emission_j) / max(state.resources.emission_remaining_j, 1.0)
            future_resource = self.mission.future_resource_weight * min(remaining_energy_fraction, remaining_emission_fraction)
            likelihood = self.tactic_model.observation_likelihood
            threat_info = self.mission.threat_information_weight * tactic_information_gain(
                self.belief.threat_belief,
                self.belief.opponent_model.transition()[int(index)],
                likelihood,
            )
            association_retention = float(np.clip(state.association_context.get("information_retention", 1.0), 0.0, 1.0))
            scenario = association_retention * np.asarray(evaluation.scenario_information_gains, dtype=float)
            retained_information = association_retention * float(evaluation.mean_information_gain)
            tail_penalty = self.mission.tail_risk_weight * max(0.0, retained_information - lower_tail_mean(scenario, 0.25))
            terms = GameRewardTerms(
                estimation_information=retained_information,
                task_loss=float(self.mission.position_weight * (1.0 - association_retention)),
                tail_risk=float(tail_penalty),
                action_cost=float(max(evaluation.decision_cost - (usage.emission_exposure_j / max(dynamic_world.max_total_emission_j, 1e-12)) * DecisionPolicy().emission_weight - (usage.communication_mb / max(dynamic_world.max_total_communication_mb, 1e-12)) * DecisionPolicy().communication_weight, 0.0)),
                switching_cost=float(switching),
                emission_cost=float((usage.emission_exposure_j / max(dynamic_world.max_total_emission_j, 1e-12)) * DecisionPolicy().emission_weight),
                communication_cost=float((usage.communication_mb / max(dynamic_world.max_total_communication_mb, 1e-12)) * DecisionPolicy().communication_weight),
                future_resource_value=float(future_resource),
                threat_information_value=float(threat_info),
            )
            world_bonus = 0.0
            if self.policy_kind == "world_model_mpc" and self.world_model_controller_enabled:
                world_bonus = float(metadata["world_model_future_bonus"][int(index)])
            exact_score = terms.total + world_bonus
            rows.append({
                "candidate_index": int(index),
                "action_ids": list(ids),
                "proposal_score": float(proposal_scores[int(index)]),
                "exact_score": float(exact_score),
                "exact_schedule_evaluation": evaluation.to_dict(),
                "reward_terms": {**asdict(terms), "total": terms.total, "world_model_future_bonus": world_bonus},
                "association_context": dict(state.association_context),
                "fidelity": dynamic_world.simulator_fidelity,
                "target_state_source": "controller_posterior_mean_not_simulator_truth",
                "target_mean_used_m_m_s": self.belief.target_mean_m_m_s.tolist(),
            })
        if not rows:
            raise RuntimeError("exact rerank received no safe shortlisted candidate")
        selected = max(rows, key=lambda row: (float(row["exact_score"]), -int(row["candidate_index"])))
        return int(selected["candidate_index"]), rows

    def select(self, state: EpisodeState, *, remaining_epochs: int) -> dict[str, Any]:
        start = time.perf_counter()
        safe_mask = self.shield.safe_mask(state, remaining_epochs=remaining_epochs)
        trigger_reasons: list[str] = []
        topology = state.communication_edges
        health = tuple(node.health.value for node in state.nodes)
        if state.epoch == 0:
            trigger_reasons.append("campaign_start")
        if self.belief.change_alarm:
            trigger_reasons.append("change_alarm")
        if topology != self.belief.topology_signature:
            trigger_reasons.append("topology_change")
        if health != self.belief.node_health_signature:
            trigger_reasons.append("node_health_change")
        if state.epoch % self.config.replanning_interval_steps == 0:
            trigger_reasons.append("periodic_replan")
        if _resource_fraction(state.resources) < 0.25:
            trigger_reasons.append("resource_reserve_proximity")
        history_free = self.policy_kind in {"fixed_policy", "static_minimax"}
        persistent = (
            not trigger_reasons
            and self.belief.previous_candidate >= 0
            and self.belief.previous_candidate < safe_mask.size
            and safe_mask[self.belief.previous_candidate]
            and not history_free
        )
        if not np.any(safe_mask):
            proposed = -1
            proposal_scores = np.full(len(self.library.schedules), -math.inf)
            proposal_metadata = {
                "no_safe_candidate": True,
                "fail_closed_action": "execute_no_RF_or_communication_action",
                "solver_invoked": False,
            }
            rerank_rows = []
        elif persistent:
            proposed = self.belief.previous_candidate
            proposal_scores = np.full(len(self.library.schedules), -math.inf)
            proposal_scores[proposed] = 0.0
            proposal_metadata: dict[str, Any] = {"action_persistence": True}
            rerank_rows: list[dict[str, Any]] = []
        else:
            proposal_scores, proposal_metadata = self._dynamic_proposal_scores(state, safe_mask)
            if not np.any(np.isfinite(proposal_scores)):
                proposed = -1
                rerank_rows = []
            else:
                proposed, rerank_rows = self._exact_rerank(state, proposal_scores, safe_mask, proposal_metadata)
        shield = self.shield.apply(proposed, state, remaining_epochs=remaining_epochs)
        latency = time.perf_counter() - start
        deadline_miss = latency > min(self.config.compute_deadline_s, self.mission.maximum_decision_latency_s)
        if deadline_miss and shield.backup_candidate >= 0:
            shield = ShieldDecision(
                shield.proposed_candidate,
                shield.backup_candidate,
                self.library.schedules[shield.backup_candidate],
                True,
                (*shield.reasons, "decision_deadline_miss_backup"),
                shield.backup_candidate,
                True,
            )
        selected_prediction = None
        if self.policy_kind == "world_model_mpc" and self.world_model_controller_enabled and shield.selected_candidate >= 0:
            selected_prediction = proposal_metadata.get("world_model_predictions", {}).get(shield.selected_candidate)
        return {
            "policy_kind": self.policy_kind,
            "selected_candidate": int(shield.selected_candidate),
            "selected_action_ids": tuple(shield.selected_action_ids),
            "proposed_candidate": int(shield.proposed_candidate),
            "shield": asdict(shield),
            "safe_mask": safe_mask.tolist(),
            "trigger_reasons": trigger_reasons,
            "replanned": not persistent,
            "proposal_scores": proposal_scores.tolist(),
            "proposal_metadata": {key: value for key, value in proposal_metadata.items() if key != "world_model_predictions"},
            "exact_rerank_rows": rerank_rows,
            "decision_latency_s": float(latency),
            "deadline_miss": bool(deadline_miss),
            "selected_world_prediction": selected_prediction,
        }

    def update(
        self,
        previous_state: EpisodeState,
        next_state: EpisodeState,
        decision: Mapping[str, Any],
        *,
        observation_index: int,
        epoch_seed: int,
    ) -> dict[str, Any]:
        selected_candidate = int(decision["selected_candidate"])
        ids = tuple(int(value) for value in decision["selected_action_ids"])
        dynamic_world = episode_state_to_world(self.world, previous_state)
        layout = StateLayout.from_world(dynamic_world)
        covariance = propagate_normalized_covariance(self.belief.covariance_normalized, dynamic_world, layout, previous_state.dt_s)
        information = stable_inverse_spd(covariance, name="v65_belief_predicted_covariance")
        predicted_mean = np.asarray(self.belief.target_mean_m_m_s, dtype=float).copy()
        predicted_mean[:3] += predicted_mean[3:] * previous_state.dt_s
        belief_world = replace(
            dynamic_world,
            target=replace(
                dynamic_world.target,
                position_m=tuple(float(value) for value in predicted_mean[:3]),
                velocity_m_s=tuple(float(value) for value in predicted_mean[3:]),
            ),
        )
        truth_world = episode_state_to_world(self.world, next_state)
        normalized_truth_error = _normalized_world_error(belief_world, truth_world, layout)
        actual_jammer = truth_world.jammers[0] if truth_world.jammers else None
        by_id = {action.action_id: action for action in belief_world.actions}
        rng = np.random.default_rng(epoch_seed)
        information_vector = np.zeros(layout.dimension, dtype=float)
        measurement_records: list[dict[str, Any]] = []
        for action_id in ids:
            action = by_id[action_id]
            assumed = build_action_factor(belief_world, action, None)
            simulated = build_action_factor(truth_world, action, actual_jammer)
            observation_probability = float(
                np.clip(
                    simulated.link_budget.detection_probability
                    * simulated.link_budget.correct_association_probability
                    * next_state.association_context.get("information_retention", 1.0),
                    0.0,
                    1.0,
                )
            )
            detected = bool(rng.random() <= observation_probability)
            if not detected:
                measurement_records.append({
                    "action_id": int(action_id),
                    "detected_and_associated": False,
                    "observation_probability": observation_probability,
                })
                continue
            h = assumed.jacobian_physical * layout.scales()[None, :]
            true_covariance = 0.5 * (simulated.covariance + simulated.covariance.T)
            filter_covariance = 0.5 * (assumed.covariance + assumed.covariance.T)
            true_covariance += 1e-15 * np.eye(true_covariance.shape[0])
            filter_covariance += 1e-15 * np.eye(filter_covariance.shape[0])
            exact_delta = nonlinear_measurement_delta(belief_world, action, normalized_truth_error, layout)
            residual = exact_delta + rng.multivariate_normal(np.zeros(exact_delta.size), true_covariance)
            precision = stable_inverse_spd(filter_covariance, name="v65_filter_measurement_covariance")
            information += h.T @ precision @ h
            information_vector += h.T @ precision @ residual
            measurement_records.append({
                "action_id": int(action_id),
                "detected_and_associated": True,
                "observation_probability": observation_probability,
                "measurement_names": list(assumed.measurement_names),
                "innovation": residual.tolist(),
                "innovation_mahalanobis2": float(residual @ precision @ residual),
                "provenance": f"simulated_measurement:{next_state.episode_id}:{next_state.epoch}:{action_id}",
            })
        posterior_covariance = stable_inverse_spd(information, name="v65_belief_posterior_information")
        normalized_correction = posterior_covariance @ information_vector
        estimate = predicted_mean + layout.scales()[:6] * normalized_correction[:6]
        if selected_candidate >= 0:
            transition = self.belief.opponent_model.transition()[selected_candidate]
            likelihood = self.tactic_model.observation_likelihood
            prediction = self.belief.threat_belief @ transition
            observation_probability = float(prediction @ likelihood[:, observation_index])
            expected_surprise = -float(np.sum((prediction @ likelihood) * np.log(np.maximum(prediction @ likelihood, 1e-12))))
            posterior, _ = bayes_update_with_likelihood(
                self.belief.threat_belief,
                transition,
                likelihood,
                observation_index,
            )
            alarm, innovation, alarm_reason = self.belief.threat_memory.update(
                posterior,
                observation_probability,
                expected_surprise,
            )
        else:
            posterior = self.belief.threat_belief.copy()
            observation_probability = 1.0
            innovation = 0.0
            alarm = False
            alarm_reason = "no_action_no_threat_update"
        previous_tactic = int(np.argmax(self.belief.threat_belief))
        next_tactic = int(np.argmax(posterior))
        if selected_candidate >= 0:
            self.belief.opponent_model.update(
                selected_candidate,
                previous_tactic,
                next_tactic,
                self.belief.inferred_tactic_age_steps,
            )
        self.belief.inferred_tactic_age_steps = (
            self.belief.inferred_tactic_age_steps + 1 if next_tactic == previous_tactic else 1
        )
        scales = layout.scales()[:6]
        physical_covariance = np.diag(scales) @ posterior_covariance[:6, :6] @ np.diag(scales)
        truth = np.r_[next_state.primary_target.position_m, next_state.primary_target.velocity_m_s]
        prediction_record = decision.get("selected_world_prediction")
        if prediction_record is not None:
            self.belief.world_model_hidden_states = prediction_record.hidden_states
        self.belief.covariance_normalized = posterior_covariance
        self.belief.target_mean_m_m_s = estimate
        self.belief.threat_belief = posterior
        self.belief.previous_candidate = selected_candidate
        self.belief.previous_action_ids = ids
        self.belief.topology_signature = next_state.communication_edges
        self.belief.node_health_signature = tuple(node.health.value for node in next_state.nodes)
        self.belief.change_alarm = bool(alarm)
        self.belief.change_alarm_reason = alarm_reason
        if alarm:
            self.belief.change_alarm_steps.append(next_state.epoch)
        self.belief.estimate_history.append(estimate)
        return {
            "observation_index": int(observation_index),
            "observation_probability": float(observation_probability),
            "posterior_threat_belief": posterior.tolist(),
            "change_alarm": bool(alarm),
            "change_alarm_reason": alarm_reason,
            "page_innovation": float(innovation),
            "target_prediction_m_m_s": predicted_mean.tolist(),
            "target_estimate_m_m_s": estimate.tolist(),
            "target_truth_m_m_s": truth.tolist(),
            "measurement_updates": measurement_records,
            "measurement_update_count": sum(bool(row["detected_and_associated"]) for row in measurement_records),
            "truth_directly_added_to_estimate": False,
            "estimator_contract": "predict_then_measurement_residual_information_update",
            "opponent_model_update_source": "posterior_tactic_belief_not_simulator_threat_mode",
            "position_error_m": float(np.linalg.norm(estimate[:3] - truth[:3])),
            "velocity_error_m_s": float(np.linalg.norm(estimate[3:] - truth[3:])),
            "posterior_position_covariance_trace_m2": float(np.trace(physical_covariance[:3, :3])),
        }


def match_changes_to_alarms_one_to_one(
    change_steps: Sequence[int],
    alarm_steps: Sequence[int],
    *,
    maximum_delay: int,
) -> dict[str, object]:
    remaining = list(sorted(int(value) for value in alarm_steps))
    matches: list[tuple[int, int] | None] = []
    for change in sorted(int(value) for value in change_steps):
        match_index = next((index for index, alarm in enumerate(remaining) if change <= alarm <= change + maximum_delay), None)
        if match_index is None:
            matches.append(None)
        else:
            alarm = remaining.pop(match_index)
            matches.append((change, alarm))
    return {
        "matches": [None if value is None else {"change_step": value[0], "alarm_step": value[1], "delay_steps": value[1] - value[0]} for value in matches],
        "unmatched_alarms": remaining,
        "one_alarm_used_at_most_once": True,
        "recall": float(sum(value is not None for value in matches) / max(len(matches), 1)),
    }


def run_closed_loop_policy(
    world: PhysicalWorld,
    *,
    policy_kind: PolicyKind,
    episode_config: EpisodeConfig,
    mission: MissionIR,
    seed: int,
    world_model: RecurrentEnsembleWorldModel | None = None,
    rl_policy: IndependentBeliefQPolicy | None = None,
    action_reactive_adversary: bool = True,
    controller_config: ControllerConfig | None = None,
    candidate_library: CandidateLibrary | None = None,
    base_payoff: np.ndarray | None = None,
) -> dict[str, Any]:
    state = initial_episode_state(world, episode_config, seed=seed)
    controller = SafeAutonomyController(
        world,
        state,
        mission=mission,
        policy_kind=policy_kind,
        config=controller_config,
        world_model=world_model,
        rl_policy=rl_policy,
        candidate_library=candidate_library,
        base_payoff=base_payoff,
    )
    records: list[dict[str, Any]] = []
    resource_violations = 0
    for step in range(episode_config.horizon_steps):
        decision = controller.select(state, remaining_epochs=episode_config.horizon_steps - step - 1)
        next_state = transition_episode_state(
            world,
            state,
            decision["selected_action_ids"],
            config=episode_config,
            seed=seed * 1_000_003 + step * 1009,
            action_reactive_adversary=action_reactive_adversary,
            compute_cost_s=min(float(decision["decision_latency_s"]), 0.05),
        )
        rng = np.random.default_rng(seed * 65537 + step)
        observation = _mode_observation_index(next_state.adversary.mode, rng)
        update = controller.update(state, next_state, decision, observation_index=observation, epoch_seed=seed * 8191 + step)
        if not next_state.resources.feasible:
            resource_violations += 1
        records.append({
            "step": step,
            "state_epoch": state.epoch,
            "next_state_epoch": next_state.epoch,
            "adversary_mode": next_state.adversary.mode.value,
            "adversary_action_reactive": next_state.adversary.action_reactive,
            "resources": asdict(next_state.resources),
            "events_started": [asdict(event) for event in next_state.events_started],
            **{key: value for key, value in decision.items() if key != "selected_world_prediction"},
            "belief_update": update,
        })
        state = next_state
    position_errors = np.asarray([row["belief_update"]["position_error_m"] for row in records], dtype=float)
    velocity_errors = np.asarray([row["belief_update"]["velocity_error_m_s"] for row in records], dtype=float)
    selected = [int(row["selected_candidate"]) for row in records]
    changes = [
        row["next_state_epoch"]
        for row in records
        if any(event["kind"] in {"attack_onset", "attack_offset", "combined_stress"} for event in row["events_started"])
    ]
    alarm_match = match_changes_to_alarms_one_to_one(changes, controller.belief.change_alarm_steps, maximum_delay=4)
    deadline_misses = sum(bool(row["deadline_miss"]) for row in records)
    shield_count = sum(bool(row["shield"]["intervened"]) for row in records)
    return {
        "stage": "v65_safe_long_horizon_closed_loop",
        "status": "passed" if resource_violations == 0 else "failed_closed",
        "policy_kind": policy_kind,
        "episode_id": state.episode_id,
        "seed": int(seed),
        "action_reactive_adversary": bool(action_reactive_adversary),
        "candidate_library": asdict(controller.library),
        "records": records,
        "metrics": {
            "position_rmse_m": float(math.sqrt(np.mean(position_errors**2))),
            "velocity_rmse_m_s": float(math.sqrt(np.mean(velocity_errors**2))),
            "mean_covariance_trace_m2": float(np.mean([row["belief_update"]["posterior_position_covariance_trace_m2"] for row in records])),
            "unique_candidate_count": len(set(selected)),
            "policy_switch_count": sum(selected[index] != selected[index - 1] for index in range(1, len(selected))),
            "shield_intervention_count": int(shield_count),
            "shield_intervention_rate": float(shield_count / max(len(records), 1)),
            "deadline_miss_count": int(deadline_misses),
            "deadline_miss_rate": float(deadline_misses / max(len(records), 1)),
            "resource_recursive_feasibility_violations": int(resource_violations),
            "minimum_energy_margin_j": float(min(row["resources"]["energy_remaining_j"] - row["resources"]["reserve_energy_j"] for row in records)),
            "minimum_emission_margin_j": float(min(row["resources"]["emission_remaining_j"] - row["resources"]["reserve_emission_j"] for row in records)),
            "change_alarm_count": len(controller.belief.change_alarm_steps),
            "change_alarm_matching": alarm_match,
            "final_threat_belief": controller.belief.threat_belief.tolist(),
        },
        "separate_evidence_contract": {
            "factor_graph_output": "external_dynamic_factor_graph_report",
            "world_model_output": (
                "analytical_belief_mpc_fallback"
                if policy_kind == "world_model_mpc" and not controller.world_model_controller_enabled
                else None if world_model is None else world_model.model_hash
            ),
            "world_model_controller_enabled": controller.world_model_controller_enabled,
            "future_environment_state_read_by_controller": False,
            "rl_training_episode_ids": [] if rl_policy is None else list(rl_policy.training_episode_ids),
            "exact_physics_output": "per_epoch_exact_rerank_rows",
            "safety_output": "per_epoch_shield_and_campaign_margins",
        },
        "claim_boundary": {
            "recursive_feasibility": "tested_on_registered_finite_action_domain",
            "global_stability_proved": False,
            "field_autonomy_validated": False,
            "clairvoyant_policy_operational": False,
            "fixed_policy_labeled_autonomous": False,
            "policy_adapts_online_from_belief": policy_kind in {"belief_mpc", "world_model_mpc", "independent_rl"},
        },
    }


def history_branch_decision(
    payoff: np.ndarray,
    model: HiddenTacticModel,
    observation_history_a: Sequence[int],
    observation_history_b: Sequence[int],
    *,
    action_history: Sequence[int],
    horizon: int = 3,
) -> dict[str, object]:
    """Acceptance control: identical current resources, distinct observation histories."""

    matrix = np.asarray(payoff, dtype=float)
    if len(observation_history_a) != len(action_history) or len(observation_history_b) != len(action_history):
        raise ValueError("history lengths disagree")

    def posterior(observations: Sequence[int]) -> np.ndarray:
        belief = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
        for action, observation in zip(action_history, observations):
            transition = model.transition_by_radar_action[int(action)]
            belief, _ = bayes_update_with_likelihood(belief, transition, model.observation_likelihood, int(observation))
        return belief

    belief_a = posterior(observation_history_a)
    belief_b = posterior(observation_history_b)
    decision_a = exact_belief_control(matrix, model, belief_a, horizon=horizon)
    decision_b = exact_belief_control(matrix, model, belief_b, horizon=horizon)
    fixed = int(np.argmax(matrix @ np.full(matrix.shape[1], 1.0 / matrix.shape[1])))
    return {
        "belief_a": belief_a.tolist(),
        "belief_b": belief_b.tolist(),
        "history_dependent_action_a": int(decision_a["selected_action"]),
        "history_dependent_action_b": int(decision_b["selected_action"]),
        "history_changes_action": int(decision_a["selected_action"]) != int(decision_b["selected_action"]),
        "fixed_policy_action_a": fixed,
        "fixed_policy_action_b": fixed,
        "fixed_policy_unchanged": True,
    }
