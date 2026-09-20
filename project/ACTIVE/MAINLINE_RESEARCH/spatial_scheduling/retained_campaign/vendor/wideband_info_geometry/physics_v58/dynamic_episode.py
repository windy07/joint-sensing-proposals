from __future__ import annotations

"""Action-conditioned hybrid episode dynamics for the active radar ecosystem.

The state is neither a shuffled static-world row nor a tactic label.  It carries
continuous kinematics/synchronization/resources and discrete semi-Markov modes,
node health, topology, spectrum/reference regimes and adversary memory.  The
transition is an executable approximation to

    p(s[t+1] | s[t], a[t], u_adv[t])

with registered event hazards and duration laws.  The adapter at the bottom
projects the dynamic state into the existing :class:`PhysicalWorld` contract so
all established V58/V64 likelihood, Schur and candidate code remains active.
"""

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .propagation import spectral_overlap_fraction
from .schema import Jammer, PhysicalWorld, RadarAction, SensorNode, TargetState


class ManeuverMode(str, Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    COORDINATED_TURN_LEFT = "coordinated_turn_left"
    COORDINATED_TURN_RIGHT = "coordinated_turn_right"
    ACCELERATE = "accelerate"
    EVADE = "evade"


class ThreatMode(str, Enum):
    QUIET = "quiet"
    BARRAGE = "barrage"
    SPOT = "spot"
    SWEPT = "swept"
    DRFM = "drfm_false_target"
    REFERENCE_CONTAMINATION = "reference_contamination"
    INTERMITTENT_DECEPTION = "intermittent_deception"


class NodeHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class EventKind(str, Enum):
    ATTACK_ONSET = "attack_onset"
    ATTACK_OFFSET = "attack_offset"
    TARGET_MANEUVER = "target_maneuver"
    TARGET_BIRTH = "target_birth"
    TARGET_DEATH = "target_death"
    NODE_LOSS = "node_loss"
    NODE_RECOVERY = "node_recovery"
    SPECTRUM_HOLE = "spectrum_hole"
    REFERENCE_DEGRADATION = "reference_degradation"
    NETWORK_CLOCK_DESYNCHRONIZATION = "network_clock_desynchronization"
    COMBINED_STRESS = "combined_stress"


TRANSPORT_BACKGROUND_PROFILES = (
    "native_stochastic",
    "registered_outage_marginal_packet_loss",
)


@dataclass(frozen=True)
class TargetDynamicState:
    target_id: str
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    existence_probability: float
    exists: bool
    maneuver_mode: ManeuverMode
    mode_age_steps: int
    mode_duration_steps: int
    mean_rcs_m2: float
    aspect_state_rad: float
    frequency_scattering_slope: float
    age_steps: int = 0
    maneuver_severity: float = 0.0
    maneuver_source: str = "initial_constant_velocity"

    def __post_init__(self) -> None:
        for name in ("position_m", "velocity_m_s", "acceleration_m_s2"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite SI values")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.existence_probability <= 1.0:
            raise ValueError("existence_probability must be in [0,1]")
        if self.mode_duration_steps <= 0 or self.mode_age_steps < 0 or self.age_steps < 0:
            raise ValueError("target duration/age state is invalid")
        if self.mean_rcs_m2 <= 0.0:
            raise ValueError("target RCS must be positive")
        if not 0.0 <= self.maneuver_severity <= 1.0:
            raise ValueError("target maneuver severity must be in [0,1]")
        if not self.maneuver_source:
            raise ValueError("target maneuver source cannot be empty")


@dataclass(frozen=True)
class AdversaryDynamicState:
    mode: ThreatMode
    mode_age_steps: int
    mode_duration_steps: int
    energy_remaining_j: float
    deception_budget: int
    response_latency_steps: int
    pending_response: ThreatMode | None
    pending_response_age_steps: int
    adaptation_memory: Mapping[str, float]
    detected_radar_band_hz: float | None
    action_reactive: bool
    internal_belief_active_probability: float

    def validate(self) -> None:
        if self.mode_duration_steps <= 0 or self.mode_age_steps < 0:
            raise ValueError("adversary duration state is invalid")
        if self.energy_remaining_j < -1e-9 or self.deception_budget < 0:
            raise ValueError("adversary resources are invalid")
        if self.response_latency_steps < 0 or self.pending_response_age_steps < 0:
            raise ValueError("adversary response latency is invalid")
        if not 0.0 <= self.internal_belief_active_probability <= 1.0:
            raise ValueError("adversary belief must be in [0,1]")


@dataclass(frozen=True)
class NodeDynamicState:
    node_id: str
    health: NodeHealth
    health_age_steps: int
    clock_bias_s: float
    clock_drift_s_per_s: float
    cfo_hz: float
    calibration_age_s: float
    packet_delay_steps: int
    packet_loss_probability: float
    queue_megabytes: float
    compute_load_fraction: float
    communication_available: bool
    receiver_available: bool
    transmitter_available: bool

    def validate(self) -> None:
        if self.health_age_steps < 0 or self.calibration_age_s < 0.0 or self.packet_delay_steps < 0:
            raise ValueError("node age/delay state is invalid")
        if not 0.0 <= self.packet_loss_probability <= 1.0:
            raise ValueError("packet loss probability must be in [0,1]")
        if self.queue_megabytes < 0.0 or not 0.0 <= self.compute_load_fraction <= 1.5:
            raise ValueError("node queue/compute state is invalid")


@dataclass(frozen=True)
class SpectrumDynamicState:
    occupancy_by_waveform: Mapping[str, float]
    illuminator_available_by_waveform: Mapping[str, bool]
    reference_quality_by_waveform: Mapping[str, float]
    clutter_regime: str
    persistent_regime_age_steps: int
    endogeneity_memory: Mapping[str, float]

    def validate(self) -> None:
        if self.persistent_regime_age_steps < 0:
            raise ValueError("spectrum regime age is invalid")
        for value in self.occupancy_by_waveform.values():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("spectrum occupancy must be in [0,1]")
        for value in self.reference_quality_by_waveform.values():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("reference quality must be in [0,1]")


@dataclass(frozen=True)
class CampaignResources:
    energy_remaining_j: float
    emission_remaining_j: float
    bandwidth_time_remaining_hz_s: float
    dwell_remaining_s: float
    communication_remaining_mb: float
    compute_remaining_s: float
    reserve_energy_j: float
    reserve_emission_j: float
    last_action_ids: tuple[int, ...] = ()
    switching_cost_accumulated: float = 0.0
    calibration_time_accumulated_s: float = 0.0

    def validate(self) -> None:
        values = (
            self.energy_remaining_j,
            self.emission_remaining_j,
            self.bandwidth_time_remaining_hz_s,
            self.dwell_remaining_s,
            self.communication_remaining_mb,
            self.compute_remaining_s,
            self.reserve_energy_j,
            self.reserve_emission_j,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("campaign resources must be finite")
        if self.reserve_energy_j < 0.0 or self.reserve_emission_j < 0.0:
            raise ValueError("campaign reserves must be nonnegative")

    @property
    def feasible(self) -> bool:
        return bool(
            self.energy_remaining_j >= self.reserve_energy_j - 1e-9
            and self.emission_remaining_j >= self.reserve_emission_j - 1e-9
            and self.bandwidth_time_remaining_hz_s >= -1e-9
            and self.dwell_remaining_s >= -1e-9
            and self.communication_remaining_mb >= -1e-9
            and self.compute_remaining_s >= -1e-9
        )


@dataclass(frozen=True)
class DynamicEvent:
    event_id: str
    epoch: int
    kind: EventKind
    entity_id: str
    severity: float
    abrupt: bool
    persistent: bool
    duration_steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeState:
    episode_id: str
    epoch: int
    time_s: float
    dt_s: float
    targets: tuple[TargetDynamicState, ...]
    adversary: AdversaryDynamicState
    nodes: tuple[NodeDynamicState, ...]
    spectrum: SpectrumDynamicState
    resources: CampaignResources
    communication_edges: tuple[tuple[str, str], ...]
    observation_hyperedges: tuple[tuple[str, ...], ...]
    events_active: tuple[DynamicEvent, ...]
    events_started: tuple[DynamicEvent, ...]
    action_history: tuple[tuple[int, ...], ...]
    adversary_action_history: tuple[str, ...]
    transition_log_probability: float
    association_context: Mapping[str, float] = field(default_factory=dict)
    effective_delay_s_by_configuration: Mapping[str, float] = field(default_factory=dict)
    receive_chain_calibration_age_s_by_configuration: Mapping[str, float] = field(
        default_factory=dict
    )
    calibration_path_bias_s_by_node: Mapping[str, float] = field(default_factory=dict)
    transition_random_stream_receipt: Mapping[str, Any] = field(default_factory=dict)
    generator_version: str = "v75_r18_controlled_intervention_episode_1"

    def validate(self) -> None:
        if self.epoch < 0 or self.time_s < 0.0 or self.dt_s <= 0.0:
            raise ValueError("episode time state is invalid")
        if not self.targets:
            raise ValueError("episode requires at least one target state")
        for target in self.targets:
            target.__post_init__()
        self.adversary.validate()
        for node in self.nodes:
            node.validate()
        self.spectrum.validate()
        self.resources.validate()
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.communication_edges:
            if len(edge) != 2 or not set(edge).issubset(node_ids):
                raise ValueError("communication edge references unknown node")
        for edge in self.observation_hyperedges:
            if not set(edge).issubset(node_ids):
                raise ValueError("observation hyperedge references unknown node")
        numeric_context = [value for value in self.association_context.values() if isinstance(value, (int, float, np.floating))]
        if any(not math.isfinite(float(value)) for value in numeric_context):
            raise ValueError("association context must be finite")
        retention = float(self.association_context.get("information_retention", 1.0))
        if not 0.0 <= retention <= 1.0:
            raise ValueError("association information retention must be in [0,1]")
        for name, values in (
            (
                "effective_delay_s_by_configuration",
                self.effective_delay_s_by_configuration,
            ),
            (
                "receive_chain_calibration_age_s_by_configuration",
                self.receive_chain_calibration_age_s_by_configuration,
            ),
            ("calibration_path_bias_s_by_node", self.calibration_path_bias_s_by_node),
        ):
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"{name} must contain finite SI values")
        if self.transition_random_stream_receipt:
            required_streams = {
                "adversary_response",
                "target_dynamics",
                "node_dynamics",
                "spectrum_dynamics",
                "receive_chain_delay",
            }
            observed_streams = set(
                self.transition_random_stream_receipt.get("component_tags", {})
            )
            if observed_streams != required_streams:
                raise ValueError(
                    "transition random-stream receipt does not cover every physical component"
                )
        if any(
            float(value) < 0.0
            for value in self.receive_chain_calibration_age_s_by_configuration.values()
        ):
            raise ValueError(
                "receive-chain calibration ages must be nonnegative"
            )
        for name, events in (
            ("events_active", self.events_active),
            ("events_started", self.events_started),
        ):
            event_ids = [event.event_id for event in events]
            if len(event_ids) != len(set(event_ids)):
                raise ValueError(f"{name} contains duplicate event identifiers")
            for event in events:
                if event.duration_steps <= 0:
                    raise ValueError(f"{name} contains a nonpositive duration")
                if not 0.0 <= float(event.severity) <= 1.0:
                    raise ValueError(f"{name} contains an invalid severity")
                if event.epoch < 0 or event.epoch > self.epoch:
                    raise ValueError(f"{name} contains a future/invalid event")
        if any(event.epoch != self.epoch for event in self.events_started):
            raise ValueError("events_started must contain only current-epoch realizations")
        if any(
            not (event.epoch <= self.epoch < event.epoch + event.duration_steps)
            for event in self.events_active
        ):
            raise ValueError("events_active contains an event outside its active interval")

    @property
    def primary_target(self) -> TargetDynamicState:
        existing = [target for target in self.targets if target.exists]
        return existing[0] if existing else self.targets[0]


@dataclass(frozen=True)
class EpisodeConfig:
    horizon_steps: int = 18
    dt_s: float = 1.0
    target_birth_probability: float = 0.02
    target_death_probability: float = 0.005
    maneuver_hazard: float = 0.08
    node_failure_hazard: float = 0.015
    node_recovery_hazard: float = 0.20
    spectrum_hole_hazard: float = 0.04
    reference_degradation_hazard: float = 0.05
    attack_hazard: float = 0.08
    quiet_duration_mean_steps: float = 5.0
    attack_duration_mean_steps: float = 6.0
    initial_adversary_energy_j: float = 500.0
    initial_deception_budget: int = 12
    response_latency_steps: int = 1
    campaign_energy_multiplier: float = 5.0
    campaign_emission_multiplier: float = 5.0
    campaign_communication_multiplier: float = 8.0
    campaign_compute_s: float = 20.0
    reserve_fraction: float = 0.12
    max_targets: int = 3
    target_birth_offset_mean_m: tuple[float, float, float] = (500.0, -300.0, 80.0)
    target_birth_offset_std_m: tuple[float, float, float] = (80.0, 80.0, 20.0)
    target_birth_velocity_std_m_s: float = 15.0
    event_schedule: tuple[tuple[int, str, str, float, int], ...] = ()
    spontaneous_target_mode_transitions_enabled: bool = True
    spontaneous_adversary_mode_transitions_enabled: bool = True
    component_keyed_random_streams: bool = False
    constant_velocity_acceleration_std_m_s2: float = 0.25
    transport_background_profile: str = "native_stochastic"
    spectrum_external_correlation_time_s: float = 4.0
    spectrum_self_recovery_time_s: float = 0.05
    spectrum_occupancy_noise_std: float = 0.025
    registered_network_clock_step_rms_s: float = 100.0e-9

    def validate(self) -> None:
        if self.horizon_steps <= 0 or self.dt_s <= 0.0 or self.max_targets <= 0:
            raise ValueError("episode horizon, step and target limit must be positive")
        for value in (
            self.target_birth_probability,
            self.target_death_probability,
            self.maneuver_hazard,
            self.node_failure_hazard,
            self.node_recovery_hazard,
            self.spectrum_hole_hazard,
            self.reference_degradation_hazard,
            self.attack_hazard,
            self.reserve_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("episode hazards and reserve fraction must be in [0,1]")
        birth_mean = np.asarray(self.target_birth_offset_mean_m, dtype=float)
        birth_std = np.asarray(self.target_birth_offset_std_m, dtype=float)
        if birth_mean.shape != (3,) or birth_std.shape != (3,):
            raise ValueError("target birth offset mean/std must contain three axes")
        if not np.all(np.isfinite(birth_mean)) or not np.all(np.isfinite(birth_std)):
            raise ValueError("target birth offset parameters must be finite")
        if np.any(birth_std < 0.0):
            raise ValueError("target birth offset standard deviations cannot be negative")
        if (
            self.target_birth_velocity_std_m_s < 0.0
            or not math.isfinite(self.target_birth_velocity_std_m_s)
        ):
            raise ValueError("target birth velocity standard deviation is invalid")
        if not isinstance(self.spontaneous_target_mode_transitions_enabled, bool):
            raise ValueError("target mode-transition switch must be boolean")
        if not isinstance(
            self.spontaneous_adversary_mode_transitions_enabled, bool
        ):
            raise ValueError("adversary mode-transition switch must be boolean")
        if not isinstance(self.component_keyed_random_streams, bool):
            raise ValueError("component-keyed random-stream flag must be boolean")
        if self.transport_background_profile not in TRANSPORT_BACKGROUND_PROFILES:
            raise ValueError(
                "unknown episode transport-background profile "
                f"{self.transport_background_profile!r}"
            )
        if (
            not math.isfinite(self.constant_velocity_acceleration_std_m_s2)
            or self.constant_velocity_acceleration_std_m_s2 < 0.0
        ):
            raise ValueError(
                "constant-velocity acceleration standard deviation must be finite "
                "and nonnegative"
            )
        if (
            not math.isfinite(self.spectrum_external_correlation_time_s)
            or self.spectrum_external_correlation_time_s <= 0.0
        ):
            raise ValueError(
                "spectrum external correlation time must be finite and positive"
            )
        if (
            not math.isfinite(self.spectrum_self_recovery_time_s)
            or self.spectrum_self_recovery_time_s <= 0.0
        ):
            raise ValueError(
                "spectrum self-recovery time must be finite and positive"
            )
        if (
            not math.isfinite(self.spectrum_occupancy_noise_std)
            or self.spectrum_occupancy_noise_std < 0.0
        ):
            raise ValueError(
                "spectrum occupancy noise standard deviation must be finite "
                "and nonnegative"
            )
        if (
            not math.isfinite(self.registered_network_clock_step_rms_s)
            or self.registered_network_clock_step_rms_s <= 0.0
        ):
            raise ValueError(
                "registered network clock-step RMS must be finite and positive"
            )
        seen_schedule_keys: set[tuple[int, str, str]] = set()
        for row in self.event_schedule:
            if len(row) != 5:
                raise ValueError("each event schedule row must have five fields")
            start, kind, entity_id, severity, duration = row
            if int(start) < 1:
                raise ValueError("scheduled event epochs must start at one")
            event_kind = EventKind(str(kind))
            if not str(entity_id):
                raise ValueError("scheduled event entity cannot be empty")
            if not 0.0 <= float(severity) <= 1.0:
                raise ValueError("scheduled event severity must be in [0,1]")
            if int(duration) < 1:
                raise ValueError("scheduled event duration must be positive")
            key = (int(start), event_kind.value, str(entity_id))
            if key in seen_schedule_keys:
                raise ValueError("event schedule contains a duplicate intervention")
            seen_schedule_keys.add(key)


@dataclass(frozen=True)
class EpisodeTrajectory:
    states: tuple[EpisodeState, ...]
    actions: tuple[tuple[int, ...], ...]
    adversary_actions: tuple[str, ...]
    observations: tuple[Mapping[str, Any], ...]
    events: tuple[DynamicEvent, ...]
    seed: int
    family: str
    generator_version: str

    def validate(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("trajectory must contain one more state than actions")
        if len(self.actions) != len(self.adversary_actions) or len(self.actions) != len(self.observations):
            raise ValueError("trajectory action/adversary/observation lengths disagree")
        for state in self.states:
            state.validate()


def _geometric_duration(rng: np.random.Generator, mean_steps: float, minimum: int = 1) -> int:
    probability = min(max(1.0 / max(mean_steps, 1.0), 1e-4), 1.0)
    return int(minimum + rng.geometric(probability) - 1)


def _node_availability(node: NodeDynamicState) -> bool:
    return node.health in {NodeHealth.HEALTHY, NodeHealth.DEGRADED} and node.communication_available


def _communication_edges(world: PhysicalWorld, nodes: Sequence[NodeDynamicState], *, range_m: float = 7000.0) -> tuple[tuple[str, str], ...]:
    dynamic = {node.node_id: node for node in nodes}
    edges: list[tuple[str, str]] = []
    for i, left in enumerate(world.nodes):
        for right in world.nodes[i + 1 :]:
            if not _node_availability(dynamic[left.node_id]) or not _node_availability(dynamic[right.node_id]):
                continue
            distance = float(np.linalg.norm(np.asarray(left.position_m) - np.asarray(right.position_m)))
            reliability = (1.0 - dynamic[left.node_id].packet_loss_probability) * (1.0 - dynamic[right.node_id].packet_loss_probability)
            if distance <= range_m and reliability >= 0.25:
                edges.append(tuple(sorted((left.node_id, right.node_id))))
    return tuple(sorted(set(edges)))


def _observation_hyperedges(world: PhysicalWorld, nodes: Sequence[NodeDynamicState]) -> tuple[tuple[str, ...], ...]:
    by_id = {node.node_id: node for node in nodes}
    edges: set[tuple[str, ...]] = set()
    for action in world.actions:
        participants = tuple(sorted({node_id for node_id in (action.tx_node_id, action.reference_node_id, *action.rx_node_ids) if node_id}))
        if participants and all(_node_availability(by_id[node_id]) for node_id in participants):
            edges.add(participants)
    return tuple(sorted(edges))


def _external_spectrum_occupancy_targets(
    world: PhysicalWorld,
) -> dict[str, float]:
    """Return the physical busy fraction induced by declared emitters.

    The same overlap-by-duty construction is used by the measurement model for
    ambient interference.  It is repeated here only to initialize and evolve
    the slowly varying *external* occupancy state.  Radar transmissions made by
    this network are not external emitters and therefore do not enter this
    term.
    """

    targets: dict[str, float] = {}
    for waveform in world.waveforms:
        idle_probability = 1.0
        for emitter in world.spectrum_emitters:
            overlap = spectral_overlap_fraction(
                waveform.carrier_hz,
                waveform.bandwidth_hz,
                emitter.center_frequency_hz,
                emitter.bandwidth_hz,
            )
            busy_fraction = float(
                np.clip(overlap * emitter.duty_cycle, 0.0, 1.0)
            )
            idle_probability *= 1.0 - busy_fraction
        targets[waveform.waveform_id] = float(
            np.clip(1.0 - idle_probability, 0.0, 1.0)
        )
    return targets


def _association_context(
    targets: Sequence[TargetDynamicState],
    adversary: AdversaryDynamicState,
    spectrum: SpectrumDynamicState,
) -> dict[str, Any]:
    """Observable ambiguity summary shared by inference, learning and planning.

    The exact bounded JPDA module consumes detections separately.  This state
    summary deliberately excludes oracle assignments; it is a deterministic
    function of target geometry, observable threat mode and spectrum occupancy.
    """

    existing = [target for target in targets if target.exists]
    if len(existing) >= 2:
        distances = [
            float(np.linalg.norm(existing[left].position_m - existing[right].position_m))
            for left in range(len(existing))
            for right in range(left + 1, len(existing))
        ]
        minimum_separation_m = min(distances)
        proximity = math.exp(-minimum_separation_m / 350.0)
    else:
        minimum_separation_m = 1.0e9
        proximity = 0.0
    mean_occupancy = float(np.mean(list(spectrum.occupancy_by_waveform.values()))) if spectrum.occupancy_by_waveform else 0.0
    deception = 1.0 if adversary.mode in {ThreatMode.DRFM, ThreatMode.INTERMITTENT_DECEPTION} else 0.0
    ambiguity = float(np.clip(0.58 * proximity + 0.30 * deception + 0.22 * mean_occupancy, 0.0, 0.95))
    correct = float(np.clip(1.0 - 0.78 * ambiguity, 0.05, 0.999))
    probabilities = np.asarray([correct, 1.0 - correct], dtype=float)
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    entropy_retention = max(0.0, 1.0 - entropy / math.log(2.0))
    retention = float(np.clip(correct * entropy_retention + (1.0 - correct) / 2.0, 0.0, 1.0))
    return {
        "existing_target_count": float(len(existing)),
        "minimum_target_separation_m": float(minimum_separation_m),
        "ambiguity_probability": ambiguity,
        "correct_association_probability": correct,
        "association_entropy_nats": entropy,
        "information_retention": retention,
        "false_alarm_intensity": float(0.01 + 0.18 * deception + 0.08 * mean_occupancy),
        "source": "observable_geometry_threat_spectrum_proxy_no_oracle_assignment",
    }


def initial_episode_state(world: PhysicalWorld, config: EpisodeConfig | None = None, *, seed: int | None = None) -> EpisodeState:
    config = config or EpisodeConfig()
    config.validate()
    rng = np.random.default_rng(world.seed if seed is None else seed)
    target = TargetDynamicState(
        target_id=world.target.target_id,
        position_m=np.asarray(world.target.position_m, dtype=float),
        velocity_m_s=np.asarray(world.target.velocity_m_s, dtype=float),
        acceleration_m_s2=np.zeros(3),
        existence_probability=0.995,
        exists=True,
        maneuver_mode=ManeuverMode.CONSTANT_VELOCITY,
        mode_age_steps=0,
        mode_duration_steps=_geometric_duration(rng, 8.0, 2),
        mean_rcs_m2=world.target.mean_rcs_m2,
        aspect_state_rad=float(rng.uniform(-math.pi, math.pi)),
        frequency_scattering_slope=float(rng.normal(0.0, 0.08)),
    )
    nodes = tuple(
        NodeDynamicState(
            node_id=node.node_id,
            health=NodeHealth.HEALTHY,
            health_age_steps=0,
            clock_bias_s=node.clock_bias_s,
            clock_drift_s_per_s=node.clock_drift_s_per_s,
            cfo_hz=node.frequency_bias_hz,
            calibration_age_s=0.0,
            packet_delay_steps=0,
            packet_loss_probability=0.01,
            queue_megabytes=0.0,
            compute_load_fraction=0.15,
            communication_available=True,
            receiver_available=True,
            transmitter_available=node.tx_power_w > 0.0,
        )
        for node in world.nodes
    )
    occupancy = _external_spectrum_occupancy_targets(world)
    availability = {waveform.waveform_id: True for waveform in world.waveforms}
    quality = {waveform.waveform_id: 0.95 for waveform in world.waveforms}
    energy = config.campaign_energy_multiplier * world.max_total_energy_j
    emission = config.campaign_emission_multiplier * world.max_total_emission_j
    resources = CampaignResources(
        energy_remaining_j=energy,
        emission_remaining_j=emission,
        bandwidth_time_remaining_hz_s=sum(w.bandwidth_hz * w.dwell_s for w in world.waveforms) * config.horizon_steps * 2.0,
        dwell_remaining_s=world.max_total_dwell_s * config.horizon_steps,
        communication_remaining_mb=config.campaign_communication_multiplier * world.max_total_communication_mb,
        compute_remaining_s=config.campaign_compute_s,
        reserve_energy_j=config.reserve_fraction * energy,
        reserve_emission_j=config.reserve_fraction * emission,
    )
    adversary = AdversaryDynamicState(
        mode=ThreatMode.QUIET,
        mode_age_steps=0,
        mode_duration_steps=_geometric_duration(rng, config.quiet_duration_mean_steps, 2),
        energy_remaining_j=config.initial_adversary_energy_j,
        deception_budget=config.initial_deception_budget,
        response_latency_steps=config.response_latency_steps,
        pending_response=None,
        pending_response_age_steps=0,
        adaptation_memory={"active_fraction_ema": 0.0, "switch_rate_ema": 0.0, "passive_fraction_ema": 0.0},
        detected_radar_band_hz=None,
        action_reactive=True,
        internal_belief_active_probability=0.2,
    )
    spectrum_state = SpectrumDynamicState(occupancy, availability, quality, "nominal", 0, {wid: 0.0 for wid in occupancy})
    state = EpisodeState(
        episode_id=f"{world.world_id}:episode:{world.seed if seed is None else seed}",
        epoch=0,
        time_s=0.0,
        dt_s=config.dt_s,
        targets=(target,),
        adversary=adversary,
        nodes=nodes,
        spectrum=spectrum_state,
        resources=resources,
        communication_edges=_communication_edges(world, nodes),
        observation_hyperedges=_observation_hyperedges(world, nodes),
        events_active=(),
        events_started=(),
        action_history=(),
        adversary_action_history=(),
        transition_log_probability=0.0,
        association_context=_association_context((target,), adversary, spectrum_state),
        effective_delay_s_by_configuration={
            item.configuration_id: float(item.effective_delay_s)
            for item in world.receive_chain_configurations
        },
        receive_chain_calibration_age_s_by_configuration={
            item.configuration_id: float(item.calibration_age_s)
            for item in world.receive_chain_configurations
        },
        calibration_path_bias_s_by_node={
            node.node_id: float(node.calibration_path_bias_s)
            for node in world.nodes
        },
    )
    state.validate()
    return state


def _selected_actions(world: PhysicalWorld, action_ids: Iterable[int]) -> tuple[RadarAction, ...]:
    by_id = {action.action_id: action for action in world.actions}
    selected: list[RadarAction] = []
    for value in sorted(set(int(item) for item in action_ids)):
        if value not in by_id:
            raise KeyError(f"unknown action {value}")
        selected.append(by_id[value])
    return tuple(selected)


def _action_summary(world: PhysicalWorld, actions: Sequence[RadarAction], previous: Sequence[int]) -> dict[str, Any]:
    active = [action for action in actions if action.kind.startswith("active")]
    passive = [action for action in actions if action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"}]
    calibration = [action for action in actions if action.kind.startswith("calibration")]
    waveforms = [world.waveform(action.waveform_id) for action in actions]
    active_airtime_s_by_waveform = {
        waveform.waveform_id: float(
            sum(
                world.waveform(action.waveform_id).dwell_s
                for action in active
                if action.waveform_id == waveform.waveform_id
            )
        )
        for waveform in world.waveforms
    }
    return {
        "active_count": len(active),
        "passive_count": len(passive),
        "calibration_count": len(calibration),
        "active_fraction": len(active) / max(len(actions), 1),
        "passive_fraction": len(passive) / max(len(actions), 1),
        "energy_j": float(sum(action.energy_cost_j for action in actions)),
        "emission_j": float(sum(action.emission_exposure_j for action in actions)),
        "communication_mb": float(sum(action.communication_cost_mb for action in actions)),
        "dwell_s": float(sum(waveform.dwell_s for waveform in waveforms)),
        "bandwidth_time_hz_s": float(sum(waveform.bandwidth_hz * waveform.dwell_s for waveform in waveforms)),
        "bands_hz": tuple(float(waveform.carrier_hz) for waveform in waveforms),
        "waveform_ids": tuple(waveform.waveform_id for waveform in waveforms),
        "active_airtime_s_by_waveform": active_airtime_s_by_waveform,
        "switch_count": len(set(previous).symmetric_difference(action.action_id for action in actions)),
    }


def _reactive_mode(summary: Mapping[str, Any], adversary: AdversaryDynamicState, rng: np.random.Generator) -> ThreatMode:
    active_fraction = float(summary["active_fraction"])
    passive_fraction = float(summary["passive_fraction"])
    switch_count = int(summary["switch_count"])
    if active_fraction >= 0.50:
        if switch_count <= 1 and adversary.deception_budget > 0:
            return ThreatMode.DRFM
        return ThreatMode.SPOT if len(set(summary["waveform_ids"])) <= 1 else ThreatMode.SWEPT
    if passive_fraction >= 0.60:
        return ThreatMode.REFERENCE_CONTAMINATION
    return ThreatMode.BARRAGE if rng.random() < 0.55 else ThreatMode.INTERMITTENT_DECEPTION


def _event_from_schedule(config: EpisodeConfig, episode_id: str, epoch: int) -> list[DynamicEvent]:
    events: list[DynamicEvent] = []
    for start, kind, entity_id, severity, duration in config.event_schedule:
        if int(start) == epoch:
            event_kind = EventKind(str(kind))
            events.append(DynamicEvent(
                event_id=f"{episode_id}:{epoch}:{event_kind.value}:{entity_id}",
                epoch=epoch,
                kind=event_kind,
                entity_id=str(entity_id),
                severity=float(severity),
                abrupt=True,
                persistent=int(duration) > 1,
                duration_steps=max(1, int(duration)),
                metadata={"source": "registered_event_schedule"},
            ))
    return events


def _target_acceleration(
    mode: ManeuverMode,
    target: TargetDynamicState,
    rng: np.random.Generator,
    severity: float,
    constant_velocity_acceleration_std_m_s2: float,
) -> np.ndarray:
    velocity = np.asarray(target.velocity_m_s, dtype=float)
    speed = max(float(np.linalg.norm(velocity[:2])), 1.0)
    heading = math.atan2(velocity[1], velocity[0])
    if mode is ManeuverMode.CONSTANT_VELOCITY:
        return rng.normal(
            0.0,
            float(constant_velocity_acceleration_std_m_s2),
            size=3,
        )
    if mode in {ManeuverMode.COORDINATED_TURN_LEFT, ManeuverMode.COORDINATED_TURN_RIGHT}:
        sign = 1.0 if mode is ManeuverMode.COORDINATED_TURN_LEFT else -1.0
        turn_rate = sign * math.radians(2.0 + 6.0 * float(severity))
        return np.asarray([-speed * turn_rate * math.sin(heading), speed * turn_rate * math.cos(heading), 0.0])
    if mode is ManeuverMode.ACCELERATE:
        acceleration = 1.0 + 3.0 * float(severity)
        return np.asarray(
            [acceleration * math.cos(heading), acceleration * math.sin(heading), 0.0]
        )
    lateral_acceleration = 2.0 + 4.0 * float(severity)
    return np.asarray(
        [
            lateral_acceleration * math.cos(heading + math.pi / 2.0),
            lateral_acceleration * math.sin(heading + math.pi / 2.0),
            rng.normal(0.0, 0.2 + 0.6 * float(severity)),
        ]
    )


def _transition_targets(
    targets: Sequence[TargetDynamicState],
    config: EpisodeConfig,
    epoch: int,
    rng: np.random.Generator,
    forced_events: Sequence[DynamicEvent],
) -> tuple[tuple[TargetDynamicState, ...], list[DynamicEvent], float]:
    output: list[TargetDynamicState] = []
    events: list[DynamicEvent] = []
    logp = 0.0
    forced_maneuver = {
        event.entity_id: event
        for event in forced_events
        if event.kind is EventKind.TARGET_MANEUVER
    }
    forced_death = {
        event.entity_id: event
        for event in forced_events
        if event.kind is EventKind.TARGET_DEATH
    }
    for target_index, target in enumerate(targets):
        forced_maneuver_event = (
            forced_maneuver.get(target.target_id)
            or (
                forced_maneuver.get("primary")
                if target_index == 0
                else None
            )
            or forced_maneuver.get("*")
        )
        forced_death_event = (
            forced_death.get(target.target_id)
            or (forced_death.get("primary") if target_index == 0 else None)
            or forced_death.get("*")
        )
        death_selected = forced_death_event is not None
        exists = target.exists and not death_selected
        if target.exists and forced_death_event is not None:
            events.append(
                DynamicEvent(
                    f"death:{epoch}:{target.target_id}",
                    epoch,
                    EventKind.TARGET_DEATH,
                    target.target_id,
                    float(forced_death_event.severity),
                    True,
                    False,
                    1,
                    {
                        "source": forced_death_event.metadata.get(
                            "source", "registered_event_schedule"
                        )
                    },
                )
            )
        if exists and rng.random() < config.target_death_probability:
            exists = False
            events.append(DynamicEvent(f"death:{epoch}:{target.target_id}", epoch, EventKind.TARGET_DEATH, target.target_id, 1.0, True, False, 1, {"source": "hazard"}))
            logp += math.log(max(config.target_death_probability, 1e-12))
        mode = target.maneuver_mode
        mode_age = target.mode_age_steps + 1
        duration = target.mode_duration_steps
        severity = float(target.maneuver_severity)
        maneuver_source = str(target.maneuver_source)
        mode_expired = mode_age >= duration
        spontaneous_switch = bool(
            config.spontaneous_target_mode_transitions_enabled
            and (mode_expired or rng.random() < config.maneuver_hazard)
        )
        controlled_return = bool(
            not config.spontaneous_target_mode_transitions_enabled
            and mode is not ManeuverMode.CONSTANT_VELOCITY
            and mode_expired
        )
        if forced_maneuver_event is not None and exists:
            old_mode = mode
            alternatives = [
                item
                for item in ManeuverMode
                if item is not ManeuverMode.CONSTANT_VELOCITY and item is not mode
            ]
            if not alternatives:
                alternatives = [
                    item
                    for item in ManeuverMode
                    if item is not ManeuverMode.CONSTANT_VELOCITY
                ]
            mode = alternatives[int(rng.integers(0, len(alternatives)))]
            mode_age = 0
            duration = max(1, int(forced_maneuver_event.duration_steps))
            severity = float(forced_maneuver_event.severity)
            maneuver_source = str(
                forced_maneuver_event.metadata.get(
                    "source", "registered_event_schedule"
                )
            )
            events.append(
                DynamicEvent(
                    f"maneuver:{epoch}:{target.target_id}",
                    epoch,
                    EventKind.TARGET_MANEUVER,
                    target.target_id,
                    severity,
                    True,
                    duration > 1,
                    duration,
                    {
                        "old_mode": old_mode.value,
                        "new_mode": mode.value,
                        "source": maneuver_source,
                        "registered_duration_honored": True,
                        "registered_severity_honored": True,
                    },
                )
            )
        elif controlled_return and exists:
            old_mode = mode
            mode = ManeuverMode.CONSTANT_VELOCITY
            mode_age = 0
            duration = max(config.horizon_steps + 1, 2)
            severity = 0.0
            maneuver_source = "controlled_return_after_registered_duration"
            events.append(
                DynamicEvent(
                    f"maneuver_return:{epoch}:{target.target_id}",
                    epoch,
                    EventKind.TARGET_MANEUVER,
                    target.target_id,
                    0.0,
                    False,
                    False,
                    1,
                    {
                        "old_mode": old_mode.value,
                        "new_mode": mode.value,
                        "source": maneuver_source,
                    },
                )
            )
        elif spontaneous_switch and exists:
            old_mode = mode
            alternatives = [item for item in ManeuverMode if item is not mode]
            mode = alternatives[int(rng.integers(0, len(alternatives)))]
            mode_age = 0
            duration = _geometric_duration(rng, 5.0, 2)
            severity = (
                0.0
                if mode is ManeuverMode.CONSTANT_VELOCITY
                else float(rng.uniform(0.35, 0.85))
            )
            maneuver_source = "spontaneous_semi_markov_transition"
            events.append(
                DynamicEvent(
                    f"maneuver:{epoch}:{target.target_id}",
                    epoch,
                    EventKind.TARGET_MANEUVER,
                    target.target_id,
                    severity,
                    True,
                    duration > 1,
                    duration,
                    {
                        "old_mode": old_mode.value,
                        "new_mode": mode.value,
                        "source": maneuver_source,
                    },
                )
            )
        acceleration = (
            _target_acceleration(
                mode,
                target,
                rng,
                severity,
                config.constant_velocity_acceleration_std_m_s2,
            )
            if exists
            else np.zeros(3)
        )
        dt = config.dt_s
        velocity = np.asarray(target.velocity_m_s) + acceleration * dt
        speed = float(np.linalg.norm(velocity))
        if speed > 320.0:
            velocity *= 320.0 / speed
        position = np.asarray(target.position_m) + np.asarray(target.velocity_m_s) * dt + 0.5 * acceleration * dt * dt
        existence_probability = float(np.clip(target.existence_probability + (0.02 if exists else -0.20), 0.0, 1.0))
        output.append(replace(
            target,
            position_m=position,
            velocity_m_s=velocity,
            acceleration_m_s2=acceleration,
            existence_probability=existence_probability,
            exists=exists,
            maneuver_mode=mode,
            mode_age_steps=mode_age,
            mode_duration_steps=duration,
            maneuver_severity=severity,
            maneuver_source=maneuver_source,
            aspect_state_rad=float((target.aspect_state_rad + math.atan2(velocity[1], velocity[0]) * 0.05) % (2.0 * math.pi)),
            age_steps=target.age_steps + 1,
        ))
    forced_births = [event for event in forced_events if event.kind is EventKind.TARGET_BIRTH]
    random_birth = len(output) < config.max_targets and rng.random() < config.target_birth_probability
    if (forced_births or random_birth) and len(output) < config.max_targets:
        index = len(output)
        forced_birth = forced_births[0] if forced_births else None
        identity = forced_birth.entity_id if forced_birth is not None else f"birth_{epoch}_{index}"
        anchor = output[0]
        new = TargetDynamicState(
            target_id=identity,
            position_m=np.asarray(anchor.position_m)
            + rng.normal(
                np.asarray(config.target_birth_offset_mean_m, dtype=float),
                np.asarray(config.target_birth_offset_std_m, dtype=float),
            ),
            velocity_m_s=np.asarray(anchor.velocity_m_s)
            + rng.normal(0.0, config.target_birth_velocity_std_m_s, 3),
            acceleration_m_s2=np.zeros(3),
            existence_probability=0.65,
            exists=True,
            maneuver_mode=ManeuverMode.CONSTANT_VELOCITY,
            mode_age_steps=0,
            mode_duration_steps=_geometric_duration(rng, 6.0, 2),
            mean_rcs_m2=max(anchor.mean_rcs_m2 * float(rng.uniform(0.3, 1.2)), 0.01),
            aspect_state_rad=float(rng.uniform(-math.pi, math.pi)),
            frequency_scattering_slope=float(rng.normal(0.0, 0.1)),
        )
        output.append(new)
        birth_duration = (
            max(1, int(forced_birth.duration_steps))
            if forced_birth is not None
            else max(1, config.horizon_steps - epoch)
        )
        events.append(
            DynamicEvent(
                f"birth:{epoch}:{identity}",
                epoch,
                EventKind.TARGET_BIRTH,
                identity,
                float(forced_birth.severity) if forced_birth is not None else 0.7,
                True,
                birth_duration > 1,
                birth_duration,
                {
                    "source": (
                        forced_birth.metadata.get(
                            "source", "registered_event_schedule"
                        )
                        if forced_birth is not None
                        else "hazard"
                    )
                },
            )
        )
    return tuple(output), events, logp


def _transition_nodes(
    world: PhysicalWorld,
    nodes: Sequence[NodeDynamicState],
    actions: Sequence[RadarAction],
    config: EpisodeConfig,
    epoch: int,
    rng: np.random.Generator,
    forced_events: Sequence[DynamicEvent],
) -> tuple[tuple[NodeDynamicState, ...], list[DynamicEvent], float]:
    events: list[DynamicEvent] = []
    action_node_ids = {node_id for action in actions for node_id in (action.tx_node_id, action.reference_node_id, *action.rx_node_ids) if node_id}
    forced_loss_events = {
        event.entity_id: event
        for event in forced_events
        if event.kind is EventKind.NODE_LOSS
    }
    forced_recovery_events = {
        event.entity_id: event
        for event in forced_events
        if event.kind is EventKind.NODE_RECOVERY
    }
    clock_events = [
        event
        for event in forced_events
        if event.kind is EventKind.NETWORK_CLOCK_DESYNCHRONIZATION
    ]
    if len(clock_events) > 1:
        raise ValueError(
            "at most one registered network clock-desynchronization event may "
            "be active in an epoch"
        )
    clock_step_by_node: dict[str, float] = {}
    if clock_events and clock_events[0].epoch == epoch:
        event = clock_events[0]
        if event.entity_id not in {"network", "*"}:
            raise ValueError(
                "network clock desynchronization must target 'network' or '*'"
            )
        node_ids = sorted(node.node_id for node in nodes)
        if len(node_ids) < 2:
            raise ValueError(
                "network clock desynchronization requires at least two nodes"
            )
        seed_document = (
            f"{event.event_id}|registered_network_clock_step|{','.join(node_ids)}"
        ).encode("utf-8")
        keyed_seed = int.from_bytes(
            hashlib.sha256(seed_document).digest()[:8], "big", signed=False
        )
        keyed_rng = np.random.default_rng(keyed_seed)
        standardized = keyed_rng.normal(0.0, 1.0, size=len(node_ids))
        standardized -= float(np.mean(standardized))
        realized_rms = float(np.sqrt(np.mean(np.square(standardized))))
        if realized_rms <= np.finfo(float).eps:
            standardized = np.linspace(-1.0, 1.0, num=len(node_ids))
            standardized -= float(np.mean(standardized))
            realized_rms = float(np.sqrt(np.mean(np.square(standardized))))
        requested_rms = float(
            event.severity * config.registered_network_clock_step_rms_s
        )
        steps = requested_rms * standardized / realized_rms
        clock_step_by_node = {
            node_id: float(value) for node_id, value in zip(node_ids, steps)
        }
        source = str(
            event.metadata.get("source", "registered_event_schedule")
        )
        events.append(
            DynamicEvent(
                f"network_clock_desynchronization:{epoch}:network",
                epoch,
                EventKind.NETWORK_CLOCK_DESYNCHRONIZATION,
                "network",
                float(event.severity),
                True,
                False,
                max(1, int(event.duration_steps)),
                {
                    "source": source,
                    "registered_duration_honored": True,
                    "registered_severity_honored": True,
                    "persistent_state_change": True,
                    "clock_step_gauge": "zero_network_mean",
                    "clock_step_rms_s": float(
                        np.sqrt(
                            np.mean(
                                np.square(list(clock_step_by_node.values()))
                            )
                        )
                    ),
                    "clock_step_mean_s": float(
                        np.mean(list(clock_step_by_node.values()))
                    ),
                    "clock_step_peak_absolute_s": float(
                        max(abs(value) for value in clock_step_by_node.values())
                    ),
                    "clock_step_by_node_s": dict(clock_step_by_node),
                    "truth_only_event_metadata": True,
                },
            )
        )
    output: list[NodeDynamicState] = []
    logp = 0.0
    sqrt_dt = math.sqrt(config.dt_s)
    common_clock_increment = (
        float(
            rng.normal(
                0.0,
                world.clock_common_random_walk_std_s_sqrt_s * sqrt_dt,
            )
        )
        if world.clock_common_random_walk_std_s_sqrt_s > 0.0
        else 0.0
    )
    common_frequency_increment = (
        float(
            rng.normal(
                0.0,
                world.frequency_common_random_walk_std_hz_sqrt_s * sqrt_dt,
            )
        )
        if world.frequency_common_random_walk_std_hz_sqrt_s > 0.0
        else 0.0
    )
    for node in nodes:
        health = node.health
        age = node.health_age_steps + 1
        forced_loss_event = (
            forced_loss_events.get(node.node_id)
            or forced_loss_events.get("*")
        )
        forced_recovery_event = (
            forced_recovery_events.get(node.node_id)
            or forced_recovery_events.get("*")
        )
        random_loss = bool(
            forced_loss_event is None
            and health in {NodeHealth.HEALTHY, NodeHealth.DEGRADED}
            and rng.random() < config.node_failure_hazard
        )
        random_recovery = bool(
            forced_recovery_event is None
            and health is NodeHealth.FAILED
            and rng.random() < config.node_recovery_hazard
        )
        if forced_loss_event is not None or random_loss:
            event = forced_loss_event
            is_start = event is None or event.epoch == epoch
            health = NodeHealth.FAILED
            age = 0 if is_start else age
            if is_start:
                duration = max(1, int(event.duration_steps)) if event is not None else 2
                severity = float(event.severity) if event is not None else 1.0
                source = (
                    str(event.metadata.get("source", "registered_event_schedule"))
                    if event is not None
                    else "hazard"
                )
                events.append(
                    DynamicEvent(
                        f"node_loss:{epoch}:{node.node_id}",
                        epoch,
                        EventKind.NODE_LOSS,
                        node.node_id,
                        severity,
                        True,
                        duration > 1,
                        duration,
                        {
                            "source": source,
                            "registered_duration_honored": event is not None,
                            "registered_severity_honored": event is not None,
                        },
                    )
                )
        elif forced_recovery_event is not None or random_recovery:
            event = forced_recovery_event
            is_start = event is None or event.epoch == epoch
            health = NodeHealth.RECOVERING
            age = 0 if is_start else age
            if is_start:
                duration = max(1, int(event.duration_steps)) if event is not None else 2
                severity = float(event.severity) if event is not None else 0.5
                source = (
                    str(event.metadata.get("source", "registered_event_schedule"))
                    if event is not None
                    else "hazard"
                )
                events.append(
                    DynamicEvent(
                        f"node_recovery:{epoch}:{node.node_id}",
                        epoch,
                        EventKind.NODE_RECOVERY,
                        node.node_id,
                        severity,
                        True,
                        duration > 1,
                        duration,
                        {
                            "source": source,
                            "registered_duration_honored": event is not None,
                            "registered_severity_honored": event is not None,
                        },
                    )
                )
        elif health is NodeHealth.RECOVERING and age >= 2:
            health = NodeHealth.DEGRADED
            age = 0
        elif health is NodeHealth.DEGRADED and age >= 3:
            health = NodeHealth.HEALTHY
            age = 0
        used = node.node_id in action_node_ids
        if (
            config.transport_background_profile
            == "registered_outage_marginal_packet_loss"
        ):
            # Ordinary packet loss is already marginalized by the observation
            # arrival model. Reusing it as a Bernoulli hard node outage counts
            # one mechanism twice and can erase every action in a nominal
            # cycle. Hard unavailability is therefore event-driven here.
            degraded = health in {NodeHealth.DEGRADED, NodeHealth.RECOVERING}
            packet_loss = float(
                np.clip(
                    1.0
                    if health is NodeHealth.FAILED
                    else (0.10 if degraded else 0.005)
                    + (0.03 if degraded else 0.02)
                    * node.compute_load_fraction,
                    0.0,
                    1.0,
                )
            )
            delay = 1 if degraded else 0
            communication_available = health is not NodeHealth.FAILED
        else:
            packet_loss = float(
                np.clip(
                    0.01
                    + (
                        0.18
                        if health is NodeHealth.DEGRADED
                        else 0.95
                        if health is NodeHealth.FAILED
                        else 0.0
                    )
                    + 0.08 * node.compute_load_fraction,
                    0.0,
                    1.0,
                )
            )
            delay = int(
                np.clip(
                    round(
                        node.queue_megabytes / 2.0
                        + rng.poisson(
                            0.3 if health is NodeHealth.HEALTHY else 1.5
                        )
                    ),
                    0,
                    8,
                )
            )
            communication_available = (
                health is not NodeHealth.FAILED and rng.random() > packet_loss
            )
        queue = max(0.0, 0.65 * node.queue_megabytes + (0.25 if used else 0.0) - 0.20)
        compute = float(np.clip(0.72 * node.compute_load_fraction + (0.25 if used else 0.02), 0.0, 1.5))
        calibration_used = any(action.kind.startswith("calibration") and node.node_id in action.rx_node_ids for action in actions)
        calibration_age = 0.0 if calibration_used else node.calibration_age_s + config.dt_s
        clock_bias = (
            node.clock_bias_s
            + node.clock_drift_s_per_s * config.dt_s
            + common_clock_increment
            + rng.normal(
                0.0,
                world.node(node.node_id).clock_random_walk_std_s_sqrt_s * sqrt_dt,
            )
            + clock_step_by_node.get(node.node_id, 0.0)
        )
        cfo = (
            node.cfo_hz
            + common_frequency_increment
            + rng.normal(
                0.0,
                world.node(node.node_id).frequency_random_walk_std_hz_sqrt_s
                * sqrt_dt,
            )
        )
        available = health is not NodeHealth.FAILED
        output.append(replace(
            node,
            health=health,
            health_age_steps=age,
            clock_bias_s=float(clock_bias),
            cfo_hz=float(cfo),
            calibration_age_s=float(calibration_age),
            packet_delay_steps=delay,
            packet_loss_probability=packet_loss,
            queue_megabytes=queue,
            compute_load_fraction=compute,
            communication_available=communication_available,
            receiver_available=available,
            transmitter_available=available and world.node(node.node_id).tx_power_w > 0.0,
        ))
    return tuple(output), events, logp


def _transition_spectrum(
    world: PhysicalWorld,
    spectrum: SpectrumDynamicState,
    summary: Mapping[str, Any],
    adversary_mode: ThreatMode,
    config: EpisodeConfig,
    epoch: int,
    episode_id: str,
    rng: np.random.Generator,
    forced_events: Sequence[DynamicEvent],
) -> tuple[SpectrumDynamicState, list[DynamicEvent], float]:
    occupancy = {key: float(value) for key, value in spectrum.occupancy_by_waveform.items()}
    quality = {key: float(value) for key, value in spectrum.reference_quality_by_waveform.items()}
    availability = dict(spectrum.illuminator_available_by_waveform)
    endogeneity = {key: float(value) for key, value in spectrum.endogeneity_memory.items()}
    events: list[DynamicEvent] = []
    forced_holes = [
        event for event in forced_events
        if event.kind in {EventKind.SPECTRUM_HOLE, EventKind.COMBINED_STRESS}
    ]
    forced_references = [
        event for event in forced_events
        if event.kind in {EventKind.REFERENCE_DEGRADATION, EventKind.COMBINED_STRESS}
    ]
    external_targets = _external_spectrum_occupancy_targets(world)
    external_memory = math.exp(
        -float(config.dt_s) / float(config.spectrum_external_correlation_time_s)
    )
    self_memory = math.exp(
        -float(config.dt_s) / float(config.spectrum_self_recovery_time_s)
    )
    active_airtime_by_waveform = {
        waveform_id: float(
            summary["active_airtime_s_by_waveform"].get(waveform_id, 0.0)
        )
        for waveform_id in occupancy
    }
    for waveform_id in occupancy:
        selected = waveform_id in summary["waveform_ids"]
        self_busy_fraction = float(
            np.clip(
                active_airtime_by_waveform[waveform_id]
                / max(float(config.dt_s), 1.0e-12),
                0.0,
                1.0,
            )
        )
        jammer_busy_fraction = 0.0
        if adversary_mode in {ThreatMode.BARRAGE, ThreatMode.SWEPT}:
            jammer_busy_fraction = 0.30
        elif adversary_mode in {ThreatMode.SPOT, ThreatMode.DRFM} and selected:
            jammer_busy_fraction = 0.45
        endogeneity[waveform_id] = float(
            np.clip(
                self_memory * endogeneity[waveform_id] + self_busy_fraction,
                0.0,
                1.0,
            )
        )
        # The occupancy target is the union of external traffic, adversarial
        # traffic and the network's finite on-air duty during the preceding
        # decision interval.  This replaces the former action-count recurrence,
        # which could turn microsecond pulses into seconds of self-blocking.
        target_occupancy = 1.0 - (
            (1.0 - external_targets.get(waveform_id, 0.0))
            * (1.0 - jammer_busy_fraction)
            * (1.0 - endogeneity[waveform_id])
        )
        occupancy[waveform_id] = float(
            np.clip(
                external_memory * occupancy[waveform_id]
                + (1.0 - external_memory) * target_occupancy
                + rng.normal(0.0, config.spectrum_occupancy_noise_std),
                0.0,
                1.0,
            )
        )
        if adversary_mode in {ThreatMode.REFERENCE_CONTAMINATION, ThreatMode.INTERMITTENT_DECEPTION}:
            quality[waveform_id] = float(np.clip(quality[waveform_id] - 0.20, 0.0, 1.0))
        else:
            quality[waveform_id] = float(np.clip(quality[waveform_id] + 0.04, 0.0, 1.0))
    forced_hole_event = forced_holes[0] if forced_holes else None
    random_hole = bool(
        forced_hole_event is None and rng.random() < config.spectrum_hole_hazard
    )
    if forced_hole_event is not None or random_hole:
        forced_id = forced_hole_event.entity_id if forced_hole_event is not None else ""
        waveform_id = (
            forced_id
            if forced_id in occupancy
            else sorted(occupancy)[sum(ord(value) for value in forced_id) % len(occupancy)]
            if forced_id
            else sorted(occupancy)[int(rng.integers(0, len(occupancy)))]
        )
        severity = (
            float(forced_hole_event.severity)
            if forced_hole_event is not None
            else 0.6
        )
        occupancy_ceiling = max(0.01, 0.25 * (1.0 - severity))
        occupancy[waveform_id] = min(occupancy[waveform_id], occupancy_ceiling)
        is_start = forced_hole_event is None or forced_hole_event.epoch == epoch
        if is_start:
            duration = (
                max(1, int(forced_hole_event.duration_steps))
                if forced_hole_event is not None
                else 2
            )
            source = (
                str(
                    forced_hole_event.metadata.get(
                        "source", "registered_event_schedule"
                    )
                )
                if forced_hole_event is not None
                else "hazard"
            )
            events.append(
                DynamicEvent(
                    f"spectrum_hole:{epoch}:{waveform_id}",
                    epoch,
                    EventKind.SPECTRUM_HOLE,
                    waveform_id,
                    severity,
                    True,
                    duration > 1,
                    duration,
                    {
                        "source": source,
                        "parent_event_kind": (
                            forced_hole_event.kind.value
                            if forced_hole_event is not None
                            else EventKind.SPECTRUM_HOLE.value
                        ),
                        "registered_duration_honored": source
                        == "registered_event_schedule",
                        "registered_severity_honored": source
                        == "registered_event_schedule",
                    },
                )
            )
    forced_reference_event = forced_references[0] if forced_references else None
    random_reference = bool(
        forced_reference_event is None
        and rng.random() < config.reference_degradation_hazard
    )
    if forced_reference_event is not None or random_reference:
        forced_id = (
            forced_reference_event.entity_id
            if forced_reference_event is not None
            else ""
        )
        waveform_id = (
            forced_id
            if forced_id in quality
            else sorted(quality)[sum(ord(value) for value in forced_id) % len(quality)]
            if forced_id
            else sorted(quality)[int(rng.integers(0, len(quality)))]
        )
        severity = (
            float(forced_reference_event.severity)
            if forced_reference_event is not None
            else 0.8
        )
        quality[waveform_id] = min(quality[waveform_id], max(0.0, 1.0 - severity))
        availability[waveform_id] = bool(quality[waveform_id] > 0.05)
        is_start = (
            forced_reference_event is None or forced_reference_event.epoch == epoch
        )
        if is_start:
            duration = (
                max(1, int(forced_reference_event.duration_steps))
                if forced_reference_event is not None
                else 3
            )
            source = (
                str(
                    forced_reference_event.metadata.get(
                        "source", "registered_event_schedule"
                    )
                )
                if forced_reference_event is not None
                else "hazard"
            )
            events.append(
                DynamicEvent(
                    f"reference_degradation:{epoch}:{waveform_id}",
                    epoch,
                    EventKind.REFERENCE_DEGRADATION,
                    waveform_id,
                    severity,
                    True,
                    duration > 1,
                    duration,
                    {
                        "source": source,
                        "parent_event_kind": (
                            forced_reference_event.kind.value
                            if forced_reference_event is not None
                            else EventKind.REFERENCE_DEGRADATION.value
                        ),
                        "registered_duration_honored": source
                        == "registered_event_schedule",
                        "registered_severity_honored": source
                        == "registered_event_schedule",
                    },
                )
            )
    clutter = "high" if float(np.mean(list(occupancy.values()))) > 0.65 else "moderate" if float(np.mean(list(occupancy.values()))) > 0.30 else "nominal"
    return SpectrumDynamicState(occupancy, availability, quality, clutter, spectrum.persistent_regime_age_steps + 1, endogeneity), events, 0.0


def _transition_adversary(
    adversary: AdversaryDynamicState,
    summary: Mapping[str, Any],
    config: EpisodeConfig,
    epoch: int,
    episode_id: str,
    rng: np.random.Generator,
    forced_events: Sequence[DynamicEvent],
    *,
    action_reactive: bool,
) -> tuple[AdversaryDynamicState, list[DynamicEvent], float]:
    events: list[DynamicEvent] = []
    memory = dict(adversary.adaptation_memory)
    memory["active_fraction_ema"] = 0.82 * memory.get("active_fraction_ema", 0.0) + 0.18 * float(summary["active_fraction"])
    memory["passive_fraction_ema"] = 0.82 * memory.get("passive_fraction_ema", 0.0) + 0.18 * float(summary["passive_fraction"])
    memory["switch_rate_ema"] = 0.82 * memory.get("switch_rate_ema", 0.0) + 0.18 * min(float(summary["switch_count"]) / 4.0, 1.0)
    belief_active = float(np.clip(0.75 * adversary.internal_belief_active_probability + 0.25 * float(summary["active_fraction"]), 0.0, 1.0))
    mode = adversary.mode
    age = adversary.mode_age_steps + 1
    duration = adversary.mode_duration_steps
    pending = adversary.pending_response
    pending_age = adversary.pending_response_age_steps
    forced_onset_event = next(
        (
            event
            for event in forced_events
            if event.kind in {EventKind.ATTACK_ONSET, EventKind.COMBINED_STRESS}
        ),
        None,
    )
    forced_offset_event = next(
        (event for event in forced_events if event.kind is EventKind.ATTACK_OFFSET),
        None,
    )
    desired = mode
    should_change = False
    transition_source = ""
    transition_severity = 0.0
    transition_duration = 0
    transition_immediate = False
    if forced_offset_event is not None:
        should_change = True
        desired = ThreatMode.QUIET
        transition_source = str(
            forced_offset_event.metadata.get("source", "registered_event_schedule")
        )
        transition_severity = float(forced_offset_event.severity)
        transition_duration = max(1, int(forced_offset_event.duration_steps))
        transition_immediate = True
    elif forced_onset_event is not None:
        should_change = True
        if action_reactive:
            desired = _reactive_mode(summary, adversary, rng)
        else:
            scripted = (
                ThreatMode.BARRAGE,
                ThreatMode.DRFM,
                ThreatMode.SPOT,
                ThreatMode.SWEPT,
            )
            desired = scripted[(epoch // 3) % len(scripted)]
        transition_source = str(
            forced_onset_event.metadata.get("source", "registered_event_schedule")
        )
        transition_severity = float(forced_onset_event.severity)
        transition_duration = max(1, int(forced_onset_event.duration_steps))
    elif pending is None and config.spontaneous_adversary_mode_transitions_enabled:
        should_change = bool(
            age >= duration
            or (
                mode is ThreatMode.QUIET
                and rng.random() < config.attack_hazard
            )
        )
        if should_change:
            if action_reactive:
                desired = _reactive_mode(summary, adversary, rng)
            else:
                scripted = (
                    ThreatMode.BARRAGE,
                    ThreatMode.DRFM,
                    ThreatMode.SPOT,
                    ThreatMode.SWEPT,
                )
                desired = scripted[(epoch // 3) % len(scripted)]
            transition_source = "spontaneous_semi_markov_transition"
            transition_severity = 0.0 if desired is ThreatMode.QUIET else 0.9
    elif (
        pending is None
        and mode is not ThreatMode.QUIET
        and age >= duration
    ):
        # In a controlled intervention experiment, a registered attack ends
        # after its registered duration instead of mutating into another
        # endogenous attack mode.
        should_change = True
        desired = ThreatMode.QUIET
        transition_source = "controlled_return_after_registered_duration"
        transition_severity = 0.0
        transition_duration = 1
        transition_immediate = True
    if should_change and (
        pending is None
        or forced_onset_event is not None
        or forced_offset_event is not None
    ):
        pending = desired
        pending_age = 0
        memory["pending_transition_duration_steps"] = float(transition_duration)
        memory["pending_transition_severity"] = float(transition_severity)
        memory["pending_transition_registered"] = float(
            transition_source == "registered_event_schedule"
        )
        memory["pending_transition_controlled_return"] = float(
            transition_source == "controlled_return_after_registered_duration"
        )
        memory["pending_transition_spontaneous"] = float(
            transition_source == "spontaneous_semi_markov_transition"
        )
        if transition_immediate:
            pending_age = adversary.response_latency_steps
    if pending is not None:
        pending_age += 1
        if pending_age > adversary.response_latency_steps:
            old = mode
            mode = pending
            pending = None
            pending_age = 0
            age = 0
            requested_duration = int(
                round(memory.pop("pending_transition_duration_steps", 0.0))
            )
            event_severity = float(memory.pop("pending_transition_severity", 0.9))
            was_registered = bool(memory.pop("pending_transition_registered", 0.0))
            was_controlled_return = bool(
                memory.pop("pending_transition_controlled_return", 0.0)
            )
            was_spontaneous = bool(memory.pop("pending_transition_spontaneous", 0.0))
            event_source = (
                "registered_event_schedule"
                if was_registered
                else "controlled_return_after_registered_duration"
                if was_controlled_return
                else "spontaneous_semi_markov_transition"
                if was_spontaneous
                else "state_transition"
            )
            duration = (
                requested_duration
                if requested_duration > 0 and mode is not ThreatMode.QUIET
                else max(config.horizon_steps + 1, 2)
                if (
                    mode is ThreatMode.QUIET
                    and not config.spontaneous_adversary_mode_transitions_enabled
                )
                else _geometric_duration(
                    rng,
                    config.quiet_duration_mean_steps if mode is ThreatMode.QUIET else config.attack_duration_mean_steps,
                    2,
                )
            )
            kind = EventKind.ATTACK_OFFSET if mode is ThreatMode.QUIET else EventKind.ATTACK_ONSET
            event_duration = duration if kind is EventKind.ATTACK_ONSET else 1
            if mode is not old or was_registered or was_controlled_return:
                events.append(
                    DynamicEvent(
                        f"adv:{epoch}:{old.value}->{mode.value}",
                        epoch,
                        kind,
                        "adversary",
                        float(np.clip(event_severity, 0.0, 1.0)),
                        True,
                        event_duration > 1,
                        event_duration,
                        {
                            "old_mode": old.value,
                            "new_mode": mode.value,
                            "action_reactive": action_reactive,
                            "source": event_source,
                            "registered_duration_honored": was_registered,
                            "registered_severity_honored": was_registered,
                        },
                    )
                )
    mode_power = {
        ThreatMode.QUIET: 0.0,
        ThreatMode.BARRAGE: 20.0,
        ThreatMode.SPOT: 12.0,
        ThreatMode.SWEPT: 15.0,
        ThreatMode.DRFM: 8.0,
        ThreatMode.REFERENCE_CONTAMINATION: 5.0,
        ThreatMode.INTERMITTENT_DECEPTION: 4.0,
    }[mode]
    energy = max(0.0, adversary.energy_remaining_j - mode_power * config.dt_s)
    deception = adversary.deception_budget - (1 if mode in {ThreatMode.DRFM, ThreatMode.INTERMITTENT_DECEPTION} and adversary.deception_budget > 0 else 0)
    if energy <= 0.0 and mode is not ThreatMode.QUIET:
        old = mode
        mode = ThreatMode.QUIET
        age = 0
        duration = max(config.horizon_steps + 1, 2)
        pending = None
        pending_age = 0
        events.append(
            DynamicEvent(
                f"adv_energy_offset:{epoch}:{old.value}->quiet",
                epoch,
                EventKind.ATTACK_OFFSET,
                "adversary",
                0.0,
                False,
                False,
                1,
                {
                    "old_mode": old.value,
                    "new_mode": ThreatMode.QUIET.value,
                    "action_reactive": action_reactive,
                    "source": "energy_depletion",
                },
            )
        )
    detected_band = float(np.mean(summary["bands_hz"])) if summary["bands_hz"] else adversary.detected_radar_band_hz
    return replace(
        adversary,
        mode=mode,
        mode_age_steps=age,
        mode_duration_steps=duration,
        energy_remaining_j=energy,
        deception_budget=deception,
        pending_response=pending,
        pending_response_age_steps=pending_age,
        adaptation_memory=memory,
        detected_radar_band_hz=detected_band,
        action_reactive=action_reactive,
        internal_belief_active_probability=belief_active,
    ), events, 0.0


def _transition_resources(
    resources: CampaignResources,
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    summary: Mapping[str, Any],
    *,
    compute_cost_s: float,
) -> CampaignResources:
    switch_cost = 0.025 * float(summary["switch_count"])
    calibration_time = sum(world.waveform(action.waveform_id).dwell_s for action in actions if action.kind.startswith("calibration"))
    return replace(
        resources,
        energy_remaining_j=resources.energy_remaining_j - float(summary["energy_j"]) - switch_cost,
        emission_remaining_j=resources.emission_remaining_j - float(summary["emission_j"]),
        bandwidth_time_remaining_hz_s=resources.bandwidth_time_remaining_hz_s - float(summary["bandwidth_time_hz_s"]),
        dwell_remaining_s=resources.dwell_remaining_s - float(summary["dwell_s"]),
        communication_remaining_mb=resources.communication_remaining_mb - float(summary["communication_mb"]),
        compute_remaining_s=resources.compute_remaining_s - float(compute_cost_s),
        last_action_ids=tuple(sorted(action.action_id for action in actions)),
        switching_cost_accumulated=resources.switching_cost_accumulated + switch_cost,
        calibration_time_accumulated_s=resources.calibration_time_accumulated_s + calibration_time,
    )


def transition_episode_state(
    world: PhysicalWorld,
    state: EpisodeState,
    action_ids: Iterable[int],
    *,
    config: EpisodeConfig | None = None,
    seed: int | None = None,
    action_reactive_adversary: bool = True,
    compute_cost_s: float = 0.01,
) -> EpisodeState:
    config = config or EpisodeConfig(dt_s=state.dt_s)
    config.validate()
    state.validate()
    epoch = state.epoch + 1
    transition_seed = int(
        (world.seed + 1) * 1_000_003 + epoch * 65537
        if seed is None
        else seed
    )
    component_tags = {
        "adversary_response": 0xAD0E25A1,
        "target_dynamics": 0x7A263701,
        "node_dynamics": 0xA0DE5001,
        "spectrum_dynamics": 0x5EEC7A01,
        "receive_chain_delay": 0xDE1A7001,
    }
    shared_rng = np.random.default_rng(transition_seed)

    def component_rng(name: str) -> np.random.Generator:
        if not config.component_keyed_random_streams:
            return shared_rng
        return np.random.default_rng(
            np.random.SeedSequence([transition_seed, component_tags[name]])
        )

    adversary_rng = component_rng("adversary_response")
    target_rng = component_rng("target_dynamics")
    node_rng = component_rng("node_dynamics")
    spectrum_rng = component_rng("spectrum_dynamics")
    receive_chain_rng = component_rng("receive_chain_delay")
    actions = _selected_actions(world, action_ids)
    summary = _action_summary(world, actions, state.resources.last_action_ids)
    scheduled = _event_from_schedule(config, state.episode_id, epoch)
    continuing = [
        event
        for event in state.events_active
        if epoch < event.epoch + max(1, event.duration_steps)
    ]
    adversary, adversary_events, logp_adv = _transition_adversary(
        state.adversary,
        summary,
        config,
        epoch,
        state.episode_id,
        adversary_rng,
        scheduled,
        action_reactive=action_reactive_adversary,
    )
    targets, target_events, logp_target = _transition_targets(
        state.targets, config, epoch, target_rng, scheduled
    )
    persistent_node_events = [
        event
        for event in continuing
        if event.kind in {EventKind.NODE_LOSS, EventKind.NODE_RECOVERY}
    ]
    persistent_spectrum_events = [
        event
        for event in continuing
        if event.kind in {EventKind.SPECTRUM_HOLE, EventKind.REFERENCE_DEGRADATION, EventKind.COMBINED_STRESS}
    ]
    nodes, node_events, logp_nodes = _transition_nodes(
        world,
        state.nodes,
        actions,
        config,
        epoch,
        node_rng,
        [*scheduled, *persistent_node_events],
    )
    spectrum, spectrum_events, logp_spectrum = _transition_spectrum(
        world,
        state.spectrum,
        summary,
        adversary.mode,
        config,
        epoch,
        state.episode_id,
        spectrum_rng,
        [*scheduled, *persistent_spectrum_events],
    )
    resources = _transition_resources(state.resources, world, actions, summary, compute_cost_s=compute_cost_s)
    configuration_by_id = {
        item.configuration_id: item
        for item in world.receive_chain_configurations
    }
    effective_delay_s_by_configuration: dict[str, float] = {}
    calibrated_configuration_ids = {
        str(configuration_id)
        for action in actions
        if action.kind.startswith("calibration")
        for configuration_id in action.metadata.get(
            "receive_configuration_ids",
            (),
        )
    }
    receive_chain_calibration_age_s_by_configuration: dict[str, float] = {}
    for configuration_id, configuration in configuration_by_id.items():
        current = float(
            state.effective_delay_s_by_configuration.get(
                configuration_id,
                configuration.effective_delay_s,
            )
        )
        effective_delay_s_by_configuration[configuration_id] = float(
            current
            + configuration.effective_delay_drift_s_per_s * config.dt_s
            + receive_chain_rng.normal(
                0.0,
                configuration.delay_random_walk_std_s_sqrt_s
                * math.sqrt(config.dt_s),
            )
        )
        current_age = float(
            state.receive_chain_calibration_age_s_by_configuration.get(
                configuration_id,
                configuration.calibration_age_s,
            )
        )
        receive_chain_calibration_age_s_by_configuration[configuration_id] = (
            0.0
            if configuration_id in calibrated_configuration_ids
            else current_age + config.dt_s
        )
    # The schedule is an exogenous intervention request, not evidence that a
    # physical event happened.  Only subsystem-confirmed state changes enter
    # the realized ledger.  This prevents schedule/realization double counting
    # and makes delayed adversary responses auditable at their true epoch.
    started = tuple(
        [*adversary_events, *target_events, *node_events, *spectrum_events]
    )
    active = tuple(
        event
        for event in [*continuing, *started]
        if epoch < event.epoch + max(1, event.duration_steps)
    )
    next_state = EpisodeState(
        episode_id=state.episode_id,
        epoch=epoch,
        time_s=state.time_s + config.dt_s,
        dt_s=config.dt_s,
        targets=targets,
        adversary=adversary,
        nodes=nodes,
        spectrum=spectrum,
        resources=resources,
        communication_edges=_communication_edges(world, nodes),
        observation_hyperedges=_observation_hyperedges(world, nodes),
        events_active=active,
        events_started=started,
        action_history=(*state.action_history, tuple(sorted(action.action_id for action in actions))),
        adversary_action_history=(*state.adversary_action_history, adversary.mode.value),
        transition_log_probability=float(logp_adv + logp_target + logp_nodes + logp_spectrum),
        association_context=_association_context(targets, adversary, spectrum),
        effective_delay_s_by_configuration=effective_delay_s_by_configuration,
        receive_chain_calibration_age_s_by_configuration=(
            receive_chain_calibration_age_s_by_configuration
        ),
        calibration_path_bias_s_by_node=dict(state.calibration_path_bias_s_by_node),
        transition_random_stream_receipt={
            "mode": (
                "component_keyed_common_random_numbers"
                if config.component_keyed_random_streams
                else "legacy_shared_sequential_stream"
            ),
            "transition_seed": transition_seed,
            "component_tags": {
                key: int(value) for key, value in component_tags.items()
            },
            "target_stream_is_action_independent": bool(
                config.component_keyed_random_streams
            ),
            "stream_keys_exclude_planning_probability_arm": True,
        },
    )
    next_state.validate()
    return next_state


def episode_state_to_world(world: PhysicalWorld, state: EpisodeState) -> PhysicalWorld:
    """Project a hybrid episode state into the established static factor API."""

    state.validate()
    primary = state.primary_target
    target = replace(
        world.target,
        target_id=primary.target_id,
        position_m=tuple(float(value) for value in primary.position_m),
        velocity_m_s=tuple(float(value) for value in primary.velocity_m_s),
        mean_rcs_m2=float(primary.mean_rcs_m2),
    )
    dynamic_nodes = {node.node_id: node for node in state.nodes}
    nodes = tuple(
        replace(
            node,
            clock_bias_s=dynamic_nodes[node.node_id].clock_bias_s,
            clock_drift_s_per_s=dynamic_nodes[node.node_id].clock_drift_s_per_s,
            frequency_bias_hz=dynamic_nodes[node.node_id].cfo_hz,
            calibration_path_bias_s=float(
                state.calibration_path_bias_s_by_node.get(
                    node.node_id,
                    node.calibration_path_bias_s,
                )
            ),
            noise_figure_db=node.noise_figure_db + (3.0 if dynamic_nodes[node.node_id].health is NodeHealth.DEGRADED else 20.0 if dynamic_nodes[node.node_id].health is NodeHealth.FAILED else 0.0),
        )
        for node in world.nodes
    )
    configurations = tuple(
        replace(
            configuration,
            effective_delay_s=float(
                state.effective_delay_s_by_configuration.get(
                    configuration.configuration_id,
                    configuration.effective_delay_s,
                )
            ),
            calibration_age_s=(
                state.receive_chain_calibration_age_s_by_configuration.get(
                    configuration.configuration_id,
                    configuration.calibration_age_s
                    + max(float(state.time_s), 0.0),
                )
            ),
        )
        for configuration in world.receive_chain_configurations
    )
    available_nodes = {
        node.node_id
        for node in state.nodes
        if node.receiver_available and node.communication_available and node.health is not NodeHealth.FAILED
    }
    actions = tuple(
        replace(
            action,
            allowed=bool(
                action.allowed
                and all(node_id in available_nodes for node_id in (action.tx_node_id, action.reference_node_id, *action.rx_node_ids) if node_id)
                and state.spectrum.occupancy_by_waveform.get(action.waveform_id, 1.0) < 0.98
                and (action.kind != "passive_bistatic" or state.spectrum.illuminator_available_by_waveform.get(action.waveform_id, False))
            ),
            metadata={
                **action.metadata,
                "calibration_age_s_max": max(
                    (
                        state.receive_chain_calibration_age_s_by_configuration.get(
                            str(configuration_id),
                            0.0,
                        )
                        for configuration_id in action.metadata.get(
                            "receive_configuration_ids",
                            (),
                        )
                    ),
                    default=0.0,
                ),
                "dynamic_epoch": state.epoch,
                "spectrum_occupancy": state.spectrum.occupancy_by_waveform.get(action.waveform_id, 0.0),
                "dynamic_reference_quality": state.spectrum.reference_quality_by_waveform.get(action.waveform_id, 1.0),
                "node_health_snapshot": {node_id: dynamic_nodes[node_id].health.value for node_id in (action.tx_node_id, action.reference_node_id, *action.rx_node_ids) if node_id},
            },
        )
        for action in world.actions
    )
    jammer: tuple[Jammer, ...] = ()
    mode = state.adversary.mode
    if mode not in {ThreatMode.QUIET, ThreatMode.REFERENCE_CONTAMINATION, ThreatMode.INTERMITTENT_DECEPTION}:
        kind = {
            ThreatMode.BARRAGE: "barrage",
            ThreatMode.SPOT: "spot",
            ThreatMode.SWEPT: "swept",
            ThreatMode.DRFM: "drfm_false_target",
        }[mode]
        waveform = world.waveforms[0]
        jammer = (
            Jammer(
                jammer_id=f"dynamic_{mode.value}_{state.epoch}",
                kind=kind,  # type: ignore[arg-type]
                position_m=tuple(float(x) for x in (np.asarray(target.position_m) + np.asarray([1500.0, -800.0, 150.0]))),
                tx_power_w=max(1.0, min(100.0, state.adversary.energy_remaining_j / max(state.dt_s, 1e-9))),
                antenna_gain_linear=8.0,
                center_frequency_hz=state.adversary.detected_radar_band_hz or waveform.carrier_hz,
                bandwidth_hz=max(waveform.bandwidth_hz, 1.0),
                duty_cycle=0.8,
                false_target_delay_s=1.2e-6 if mode is ThreatMode.DRFM else 0.0,
                false_target_doppler_hz=180.0 if mode is ThreatMode.DRFM else 0.0,
                deception_count=2 if mode is ThreatMode.DRFM else 0,
                instantaneous_bandwidth_hz=max(waveform.bandwidth_hz * 0.15, 1.0) if mode is ThreatMode.SWEPT else None,
            ),
        )
    provenance = {
        **world.external_channel_provenance,
        "dynamic_episode_id": state.episode_id,
        "dynamic_epoch": state.epoch,
        "action_conditioned_transition": True,
        "adversary_action_reactive": state.adversary.action_reactive,
        "multi_target_truth_count": sum(target.exists for target in state.targets),
        "resource_state": asdict(state.resources),
        "fidelity_boundary": "hybrid_action_conditioned_synthetic_episode_not_field_environment",
    }
    projected = replace(
        world,
        nodes=nodes,
        target=target,
        actions=actions,
        jammers=jammer,
        receive_chain_configurations=configurations,
        external_channel_provenance=provenance,
    )
    projected.validate()
    return projected


def action_is_dynamically_feasible(world: PhysicalWorld, state: EpisodeState, action_ids: Iterable[int]) -> tuple[bool, tuple[str, ...]]:
    dynamic_world = episode_state_to_world(world, state)
    actions = _selected_actions(dynamic_world, action_ids)
    reasons: list[str] = []
    if any(not action.allowed for action in actions):
        reasons.append("node_spectrum_or_reference_unavailable")
    summary = _action_summary(dynamic_world, actions, state.resources.last_action_ids)
    if state.resources.energy_remaining_j - summary["energy_j"] < state.resources.reserve_energy_j:
        reasons.append("energy_reserve")
    if state.resources.emission_remaining_j - summary["emission_j"] < state.resources.reserve_emission_j:
        reasons.append("emission_reserve")
    if state.resources.bandwidth_time_remaining_hz_s < summary["bandwidth_time_hz_s"]:
        reasons.append("bandwidth_time")
    if state.resources.dwell_remaining_s < summary["dwell_s"]:
        reasons.append("dwell")
    if state.resources.communication_remaining_mb < summary["communication_mb"]:
        reasons.append("communication")
    return not reasons, tuple(reasons)


def _observation_summary(world: PhysicalWorld, state: EpisodeState, action_ids: Sequence[int], rng: np.random.Generator) -> dict[str, Any]:
    actions = _selected_actions(world, action_ids)
    active = sum(action.kind.startswith("active") for action in actions)
    passive = sum(action.kind in {"passive_bistatic", "tdoa", "fdoa", "aoa"} for action in actions)
    mean_occupancy = float(np.mean([state.spectrum.occupancy_by_waveform.get(action.waveform_id, 0.0) for action in actions])) if actions else 1.0
    mean_reference = float(np.mean([state.spectrum.reference_quality_by_waveform.get(action.waveform_id, 1.0) for action in actions])) if actions else 0.0
    threat_signal = {
        ThreatMode.QUIET: 0.05,
        ThreatMode.BARRAGE: 0.75,
        ThreatMode.SPOT: 0.70,
        ThreatMode.SWEPT: 0.65,
        ThreatMode.DRFM: 0.85,
        ThreatMode.REFERENCE_CONTAMINATION: 0.72,
        ThreatMode.INTERMITTENT_DECEPTION: 0.55,
    }[state.adversary.mode]
    return {
        "epoch": state.epoch,
        "threat_indicator": float(np.clip(threat_signal + rng.normal(0.0, 0.08), 0.0, 1.0)),
        "active_measurement_count": active,
        "passive_measurement_count": passive,
        "spectrum_occupancy": mean_occupancy,
        "reference_quality": mean_reference,
        "connected_edge_count": len(state.communication_edges),
        "available_node_count": sum(_node_availability(node) for node in state.nodes),
        "target_count": sum(target.exists for target in state.targets),
        "resource_fraction": state.resources.energy_remaining_j / max(state.resources.energy_remaining_j + sum(action.energy_cost_j for action in actions), 1e-30),
    }


def generate_dynamic_episode(
    world: PhysicalWorld,
    policy: Callable[[EpisodeState, PhysicalWorld], Sequence[int]],
    *,
    config: EpisodeConfig | None = None,
    seed: int | None = None,
    action_reactive_adversary: bool = True,
) -> EpisodeTrajectory:
    config = config or EpisodeConfig()
    config.validate()
    base_seed = world.seed if seed is None else int(seed)
    state = initial_episode_state(world, config, seed=base_seed)
    states = [state]
    actions: list[tuple[int, ...]] = []
    adversary_actions: list[str] = []
    observations: list[Mapping[str, Any]] = []
    events: list[DynamicEvent] = []
    for epoch in range(config.horizon_steps):
        dynamic_world = episode_state_to_world(world, state)
        selected = tuple(sorted(set(int(value) for value in policy(state, dynamic_world))))
        feasible, reasons = action_is_dynamically_feasible(world, state, selected)
        if not feasible:
            selected = tuple(
                action.action_id
                for action in dynamic_world.actions
                if action.allowed and not action.kind.startswith("active")
            )[:1]
        rng = np.random.default_rng(base_seed * 65537 + epoch * 1009)
        next_state = transition_episode_state(
            world,
            state,
            selected,
            config=config,
            seed=base_seed * 1_000_003 + epoch,
            action_reactive_adversary=action_reactive_adversary,
        )
        observations.append({**_observation_summary(dynamic_world, next_state, selected, rng), "shield_reasons": reasons if not feasible else ()})
        actions.append(selected)
        adversary_actions.append(next_state.adversary.mode.value)
        events.extend(next_state.events_started)
        states.append(next_state)
        state = next_state
    trajectory = EpisodeTrajectory(tuple(states), tuple(actions), tuple(adversary_actions), tuple(observations), tuple(events), base_seed, world.family, state.generator_version)
    trajectory.validate()
    return trajectory


def default_event_scenarios() -> dict[str, EpisodeConfig]:
    """Registered scenario families used by smoke and server matrices."""

    return {
        "quiet_then_abrupt_attack": EpisodeConfig(event_schedule=((5, EventKind.ATTACK_ONSET.value, "adversary", 1.0, 4),)),
        "persistent_pressure": EpisodeConfig(event_schedule=((3, EventKind.ATTACK_ONSET.value, "adversary", 0.9, 10),), attack_duration_mean_steps=10.0),
        "intermittent_deception": EpisodeConfig(event_schedule=((3, EventKind.ATTACK_ONSET.value, "adversary", 0.7, 2), (8, EventKind.ATTACK_ONSET.value, "adversary", 0.7, 2))),
        "target_maneuver": EpisodeConfig(event_schedule=((6, EventKind.TARGET_MANEUVER.value, "t0", 0.8, 4),)),
        "node_loss_recovery": EpisodeConfig(event_schedule=((4, EventKind.NODE_LOSS.value, "n0", 1.0, 3), (8, EventKind.NODE_RECOVERY.value, "n0", 0.5, 2))),
        "spectrum_hole": EpisodeConfig(event_schedule=((5, EventKind.SPECTRUM_HOLE.value, "band", 0.8, 3),)),
        "passive_reference_degradation": EpisodeConfig(event_schedule=((4, EventKind.REFERENCE_DEGRADATION.value, "reference", 0.9, 5),)),
        "combined_stress": EpisodeConfig(event_schedule=((3, EventKind.COMBINED_STRESS.value, "system", 1.0, 6), (4, EventKind.NODE_LOSS.value, "n0", 1.0, 3), (5, EventKind.TARGET_MANEUVER.value, "t0", 0.8, 4), (6, EventKind.REFERENCE_DEGRADATION.value, "reference", 0.8, 4))),
        "bounded_multi_target": EpisodeConfig(event_schedule=((4, EventKind.TARGET_BIRTH.value, "t1", 0.8, 12),), max_targets=3),
    }
