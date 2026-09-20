from __future__ import annotations

"""Stable nuisance-marginal information and robust schedule objectives."""

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.linalg import solve_triangular

from .measurement import ActionFactor, StateLayout, build_action_factor, world_propagated_to_time
from .schema import Jammer, PhysicalWorld, RadarAction
from .effective_delay import action_receive_configurations


class _WorldCacheIdentity:
    """Hashable, strong-reference identity for an in-memory physical world.

    ``PhysicalWorld`` is frozen but contains dictionaries and is therefore not
    hashable.  A cache key made only from ``world_id`` and ``seed`` is still
    unsafe because controlled counterfactuals commonly use
    ``dataclasses.replace`` while intentionally preserving both fields.  This
    wrapper compares the underlying object by identity and keeps it alive for
    as long as the caller-owned cache keeps the key, preventing Python object-id
    reuse from aliasing a later world.

    The token is intentionally process-local: action factors and phase-space
    profiles are in-memory caches and are never persisted as scientific
    evidence.
    """

    __slots__ = ("world",)

    def __init__(self, world: PhysicalWorld) -> None:
        self.world = world

    def __hash__(self) -> int:
        return id(self.world)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _WorldCacheIdentity) and self.world is other.world


def physical_world_cache_identity(world: PhysicalWorld) -> object:
    """Return a hashable token that distinguishes same-id counterfactuals."""

    return _WorldCacheIdentity(world)


@dataclass(frozen=True)
class DecisionPolicy:
    risk_tail_probability: float = 0.25
    cvar_weight: float = 0.50
    task_d_opt_weight: float = 0.50
    resource_cost_mode: str = "normalized"
    energy_weight: float = 0.08
    emission_weight: float = 0.12
    communication_weight: float = 0.04
    latency_weight: float = 0.06
    energy_price_per_j: float = 0.01
    emission_price_per_j: float = 0.02
    communication_price_per_mb: float = 0.001

    def validate(self) -> None:
        if not 0.0 < self.risk_tail_probability <= 1.0:
            raise ValueError("risk_tail_probability must be in (0,1]")
        if not 0.0 <= self.cvar_weight <= 1.0:
            raise ValueError("cvar_weight must be in [0,1]")
        if not 0.0 <= self.task_d_opt_weight <= 1.0:
            raise ValueError("task_d_opt_weight must be in [0,1]")
        if self.resource_cost_mode not in {"normalized", "absolute"}:
            raise ValueError("resource_cost_mode must be 'normalized' or 'absolute'")
        weights = (self.energy_weight, self.emission_weight, self.communication_weight, self.latency_weight)
        prices = (self.energy_price_per_j, self.emission_price_per_j, self.communication_price_per_mb)
        if min(*weights, *prices) < 0.0:
            raise ValueError("resource weights and prices must be nonnegative")


@dataclass(frozen=True)
class ScheduleResourceUsage:
    dwell_s: float
    coherent_processing_s: float
    energy_j: float
    emission_exposure_j: float
    communication_mb: float
    decision_latency_s: float


@dataclass(frozen=True)
class ScheduleEvaluation:
    action_ids: tuple[int, ...]
    feasible: bool
    violations: tuple[str, ...]
    nominal_information_gain: float
    mean_information_gain: float
    cvar_information_gain: float
    worst_case_information_gain: float
    robust_information_score: float
    decision_cost: float
    robust_utility: float
    position_rmse_bound_m: float
    velocity_rmse_bound_m_s: float
    missing_information_trace: float
    scenario_information_gains: tuple[float, ...]
    energy_j: float
    emission_exposure_j: float
    communication_mb: float
    decision_latency_s: float
    nominal_d_opt_gain: float = 0.0
    nominal_a_opt_gain: float = 0.0
    scenario_d_opt_gains: tuple[float, ...] = ()
    scenario_a_opt_gains: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def action_factor_cache_key(
    world: PhysicalWorld,
    action_id: int,
    scenario_key: str,
    geometry_time_s: float | None = None,
) -> tuple[object, ...]:
    """Return a cache key covering every physical dependency of a factor.

    An :class:`ActionFactor` depends on the *world* (geometry, waveforms,
    surface, atmosphere, spectrum, timing configuration and RNG seed), on the
    action, on the adversary scenario, and - for sequential schedules - on the
    absolute sensing time at which the world is propagated.  Keying only on
    ``(action_id, scenario)`` silently returns another world's factor whenever a
    cache dict is shared across worlds, which the planner and the dataset
    generator both do.  ``world_id`` and ``seed`` alone are also insufficient:
    a controlled counterfactual produced with ``dataclasses.replace`` normally
    preserves both.  The process-local identity token therefore participates in
    the key and holds a strong reference to the exact world object.
    """

    return (
        physical_world_cache_identity(world),
        str(world.world_id),
        int(world.seed),
        int(action_id),
        str(scenario_key),
        None if geometry_time_s is None else round(float(geometry_time_s), 12),
    )


def _sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)


def _spd_eigendecomposition(matrix: np.ndarray, *, name: str, floor: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    symmetric = _sym(matrix)
    if not np.all(np.isfinite(symmetric)):
        bad = np.argwhere(~np.isfinite(symmetric))
        raise ValueError(f"{name} contains {bad.shape[0]} non-finite entries; first indices={bad[:8].tolist()}")
    normalization = float(np.max(np.abs(symmetric))) if symmetric.size else 0.0
    if normalization == 0.0:
        normalized = symmetric
        normalization = 1.0
    else:
        normalized = symmetric / normalization
    # Normalize first, then avoid the divide-and-conquer driver that exhibited
    # non-convergence on valid 26x26 covariance matrices under NumPy 2.3/Linux.
    # EVR is the primary symmetric solver; classical QR (EV) is the independent
    # fallback. A positive scalar preserves eigenvectors, inertia, condition
    # number, and PSD status.
    failures: list[str] = []
    values_normalized: np.ndarray | None = None
    vectors: np.ndarray | None = None
    for driver in ("evr", "ev"):
        try:
            candidate_values, candidate_vectors = scipy_eigh(
                normalized,
                driver=driver,
                check_finite=False,
                overwrite_a=False,
            )
            residual = float(
                np.linalg.norm(normalized @ candidate_vectors - candidate_vectors * candidate_values, ord="fro")
                / max(1.0, float(np.linalg.norm(normalized, ord="fro")))
            )
            orthogonality = float(
                np.linalg.norm(candidate_vectors.T @ candidate_vectors - np.eye(candidate_vectors.shape[1]), ord="fro")
            )
            if not np.all(np.isfinite(candidate_values)) or not np.all(np.isfinite(candidate_vectors)):
                raise np.linalg.LinAlgError("non-finite eigensystem")
            if residual > 1e-8 or orthogonality > 1e-8:
                raise np.linalg.LinAlgError(
                    f"unacceptable eigensystem residual={residual}, orthogonality={orthogonality}"
                )
            values_normalized, vectors = candidate_values, candidate_vectors
            break
        except np.linalg.LinAlgError as exc:
            failures.append(f"{driver}:{exc!r}")
    if values_normalized is None or vectors is None:
        raise np.linalg.LinAlgError(f"{name} symmetric eigensolvers failed: {'; '.join(failures)}")
    values = values_normalized * normalization
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -1e-8 * scale:
        raise ValueError(f"{name} is not positive semidefinite: min eigenvalue={float(np.min(values))}")
    return np.maximum(values, floor * scale), vectors


def stable_inverse_spd(matrix: np.ndarray, *, name: str) -> np.ndarray:
    symmetric = _sym(matrix)
    if not np.all(np.isfinite(symmetric)):
        bad = np.argwhere(~np.isfinite(symmetric))
        raise ValueError(f"{name} contains {bad.shape[0]} non-finite entries; first indices={bad[:8].tolist()}")
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        values, vectors = _spd_eigendecomposition(symmetric, name=name)
        return _sym((vectors * (1.0 / values)) @ vectors.T)
    inverse_factor = solve_triangular(factor, np.eye(factor.shape[0]), lower=True, check_finite=False)
    return _sym(inverse_factor.T @ inverse_factor)


def stable_logdet_spd(matrix: np.ndarray, *, name: str) -> float:
    symmetric = _sym(matrix)
    if not np.all(np.isfinite(symmetric)):
        bad = np.argwhere(~np.isfinite(symmetric))
        raise ValueError(f"{name} contains {bad.shape[0]} non-finite entries; first indices={bad[:8].tolist()}")
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        values, _ = _spd_eigendecomposition(symmetric, name=name)
        return float(np.log(values).sum())
    return float(2.0 * np.log(np.diag(factor)).sum())


def schur_effective_information(joint_information: np.ndarray, target_dim: int = 6) -> tuple[np.ndarray, np.ndarray]:
    joint = _sym(joint_information)
    jxx = joint[:target_dim, :target_dim]
    jxn = joint[:target_dim, target_dim:]
    jnn = _sym(joint[target_dim:, target_dim:])
    if jnn.size == 0:
        return jxx, np.zeros_like(jxx)
    # Form J_xn J_nn^{-1} J_nx as W.T @ W.  Besides avoiding an explicit
    # inverse, the whitened Gram form preserves the mathematical PSD structure
    # to backward precision when SI-normalized information spans many scales.
    try:
        factor = np.linalg.cholesky(jnn)
        whitened = solve_triangular(
            factor,
            jxn.T,
            lower=True,
            check_finite=False,
        )
    except np.linalg.LinAlgError:
        values, vectors = _spd_eigendecomposition(jnn, name="nuisance_information")
        whitened = (vectors.T @ jxn.T) / np.sqrt(values)[:, None]
    loss = _sym(whitened.T @ whitened)
    effective = _sym(jxx - loss)
    _spd_eigendecomposition(effective, name="Schur_effective_information")
    return effective, loss


def target_prior_standard_deviations_physical(world: PhysicalWorld) -> np.ndarray:
    """Return registered anisotropic target prior standard deviations in SI."""

    provenance = world.external_channel_provenance

    def axes(vector_key: str, scalar_key: str, default: float) -> np.ndarray:
        if vector_key in provenance:
            values = np.asarray(provenance[vector_key], dtype=float).reshape(-1)
            if values.shape != (3,):
                raise ValueError(f"{vector_key} must contain three axis values")
        else:
            values = np.full(3, float(provenance.get(scalar_key, default)))
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{vector_key} prior scales must be finite and positive")
        return values

    position = axes("position_prior_std_m_xyz", "position_prior_std_m", 300.0)
    velocity = axes(
        "velocity_prior_std_m_s_xyz",
        "velocity_prior_std_m_s",
        40.0,
    )
    return np.concatenate((position, velocity))


def prior_covariance_physical(
    world: PhysicalWorld,
    layout: StateLayout | None = None,
) -> np.ndarray:
    """Return the registered joint target/nuisance prior covariance in SI units.

    This is the single source of truth for both the graph prior and the
    simulator draw that separates the controller's prior reference from hidden
    truth.  Keeping those two uses on the same covariance prevents a subtle
    oracle initialization in which the graph advertises uncertainty while its
    prior mean is exactly the generated truth state.
    """

    layout = layout or StateLayout.from_world(world)
    stds = target_prior_standard_deviations_physical(world).tolist()
    nodes = {n.node_id: n for n in world.nodes}
    stds += [nodes[node_id].clock_bias_std_s for node_id in layout.node_ids]
    stds += [nodes[node_id].frequency_bias_std_hz for node_id in layout.node_ids]
    configurations = {item.configuration_id: item for item in world.receive_chain_configurations}
    stds += [configurations[configuration_id].prior_std_s for configuration_id in layout.receive_configuration_ids]
    if layout.receive_configuration_ids:
        stds += [nodes[node_id].calibration_path_truth_std_s for node_id in layout.node_ids]
    physical_covariance = np.diag(np.square(np.asarray(stds, dtype=float)))
    clock_indices = [layout.clock_index(node_id) for node_id in layout.node_ids]
    frequency_indices = [layout.frequency_index(node_id) for node_id in layout.node_ids]
    if world.clock_common_mode_std_s > 0.0:
        physical_covariance[np.ix_(clock_indices, clock_indices)] += world.clock_common_mode_std_s**2
    if world.frequency_common_mode_std_hz > 0.0:
        physical_covariance[np.ix_(frequency_indices, frequency_indices)] += world.frequency_common_mode_std_hz**2
    for left_index, left_id in enumerate(layout.receive_configuration_ids):
        left = configurations[left_id]
        left_state_index = layout.effective_delay_offset + left_index
        for right_index, right_id in enumerate(layout.receive_configuration_ids[left_index + 1 :], start=left_index + 1):
            right = configurations[right_id]
            if left.transfer_group_id != right.transfer_group_id:
                continue
            right_state_index = layout.effective_delay_offset + right_index
            shared_covariance = left.common_delay_std_s * right.common_delay_std_s
            physical_covariance[left_state_index, right_state_index] = shared_covariance
            physical_covariance[right_state_index, left_state_index] = shared_covariance
    return 0.5 * (physical_covariance + physical_covariance.T)


def prior_information_normalized(world: PhysicalWorld, layout: StateLayout | None = None) -> np.ndarray:
    layout = layout or StateLayout.from_world(world)
    scales = layout.scales()
    physical_covariance = prior_covariance_physical(world, layout)
    normalized_covariance = physical_covariance / np.outer(scales, scales)
    if (
        not layout.receive_configuration_ids
        and world.time_reference_profile == "legacy_uncorrelated"
    ):
        return np.diag(1.0 / np.diag(normalized_covariance))
    return stable_inverse_spd(normalized_covariance, name="correlated_timing_and_configuration_joint_prior_covariance")


def aggregate_joint_information(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    jammer: Jammer | None,
    factor_cache: dict[tuple[object, ...], ActionFactor] | None = None,
) -> tuple[np.ndarray, list[ActionFactor]]:
    layout = StateLayout.from_world(world)
    joint = prior_information_normalized(world, layout)
    factors: list[ActionFactor] = []
    scenario_key = "nominal" if jammer is None else jammer.jammer_id
    for action in actions:
        key = action_factor_cache_key(world, action.action_id, scenario_key)
        if factor_cache is not None and key in factor_cache:
            factor = factor_cache[key]
        else:
            factor = build_action_factor(world, action, jammer)
            if factor_cache is not None:
                factor_cache[key] = factor
        joint = joint + factor.observed_information_normalized
        factors.append(factor)
    return _sym(joint), factors


def _frequency_interval(world: PhysicalWorld, action: RadarAction) -> tuple[float, float]:
    waveform = world.waveform(action.waveform_id)
    return waveform.carrier_hz - waveform.bandwidth_hz / 2.0, waveform.carrier_hz + waveform.bandwidth_hz / 2.0


def _slot_transport_delay(world: PhysicalWorld, slot_actions: Sequence[RadarAction]) -> float:
    communication_by_node: dict[str, float] = {}
    for action in slot_actions:
        allocations = action.metadata.get("communication_mb_by_node", {})
        if isinstance(allocations, dict):
            for node_id, value in allocations.items():
                communication_by_node[str(node_id)] = communication_by_node.get(str(node_id), 0.0) + float(value)
    delays = []
    for node_id, megabytes in communication_by_node.items():
        node = world.node(node_id)
        serialization_s = 8.0 * megabytes / max(node.communication_capacity_mbps, 1e-12)
        delays.append(node.latency_s + serialization_s)
    return float(max(delays) if delays else 0.0)


def schedule_action_times(world: PhysicalWorld, actions: Sequence[RadarAction]) -> tuple[dict[int, float], float]:
    """Return absolute sensing mid-times and terminal network-decision time.

    ``resource_slot`` is a physical time index.  Empty slots therefore retain
    their duration instead of being compressed out of the schedule.  Receiver
    serialization and node latency affect decision availability, while sensing
    in later slots proceeds on the declared slot clock.
    """

    all_actions_by_slot: dict[int, list[RadarAction]] = {}
    for action in world.actions:
        all_actions_by_slot.setdefault(action.resource_slot, []).append(action)
    slot_starts: dict[int, float] = {}
    elapsed = 0.0
    for slot in sorted(all_actions_by_slot):
        slot_starts[slot] = elapsed
        elapsed += max(
            world.waveform(action.waveform_id).coherent_processing_interval_s
            for action in all_actions_by_slot[slot]
        )

    selected_by_slot: dict[int, list[RadarAction]] = {}
    for action in actions:
        selected_by_slot.setdefault(action.resource_slot, []).append(action)
    action_times: dict[int, float] = {}
    completion_times: list[float] = []
    for slot, slot_actions in sorted(selected_by_slot.items()):
        start_time = slot_starts.get(slot, 0.0)
        transport_delay = _slot_transport_delay(world, slot_actions)
        for action in slot_actions:
            duration = world.waveform(action.waveform_id).coherent_processing_interval_s
            action_times[action.action_id] = start_time + 0.5 * duration
            completion_times.append(start_time + duration + transport_delay)
    return action_times, float(max(completion_times, default=0.0))


def schedule_resource_usage(world: PhysicalWorld, actions: Sequence[RadarAction]) -> ScheduleResourceUsage:
    """Aggregate sensing, energy, emission, communication, and decision latency."""

    dwell = float(sum(world.waveform(action.waveform_id).dwell_s for action in actions))
    energy = float(sum(action.energy_cost_j for action in actions))
    emission = float(sum(action.emission_exposure_j for action in actions))
    communication = float(sum(action.communication_cost_mb for action in actions))
    actions_by_slot: dict[int, list[RadarAction]] = {}
    for action in actions:
        actions_by_slot.setdefault(action.resource_slot, []).append(action)
    coherent_processing = float(sum(
        max(world.waveform(action.waveform_id).coherent_processing_interval_s for action in slot_actions)
        for slot_actions in actions_by_slot.values()
    ))
    _, decision_latency = schedule_action_times(world, actions)
    return ScheduleResourceUsage(
        dwell_s=dwell,
        coherent_processing_s=coherent_processing,
        energy_j=energy,
        emission_exposure_j=emission,
        communication_mb=communication,
        decision_latency_s=decision_latency,
    )


def decision_cost_for_actions(world: PhysicalWorld, actions: Sequence[RadarAction], policy: DecisionPolicy) -> float:
    """Dimensionless resource cost aligned with declared world budgets."""

    policy.validate()
    usage = schedule_resource_usage(world, actions)
    if policy.resource_cost_mode == "absolute":
        return float(
            policy.energy_price_per_j * usage.energy_j
            + policy.emission_price_per_j * usage.emission_exposure_j
            + policy.communication_price_per_mb * usage.communication_mb
        )
    return float(
        policy.energy_weight * usage.energy_j / world.max_total_energy_j
        + policy.emission_weight * usage.emission_exposure_j / world.max_total_emission_j
        + policy.communication_weight * usage.communication_mb / world.max_total_communication_mb
        + policy.latency_weight * usage.decision_latency_s / world.max_decision_latency_s
    )


def normalized_state_transition(layout: StateLayout, dt_s: float) -> np.ndarray:
    """Constant-velocity state transition in normalized coordinates."""

    dt = max(0.0, float(dt_s))
    transition = np.eye(layout.dimension, dtype=float)
    scales = layout.scales()
    transition[:3, 3:6] = np.eye(3) * dt * scales[3] / scales[0]
    return transition


def normalized_process_covariance(world: PhysicalWorld, layout: StateLayout, dt_s: float) -> np.ndarray:
    """Piecewise-constant random-acceleration covariance.

    ``TargetState.acceleration_std_m_s2`` is a per-epoch acceleration standard
    deviation, not a continuous-time spectral density.  The corresponding
    discrete constant-acceleration model uses ``G Q_a G.T`` with
    ``G=[dt^2/2, dt].T``.  Keeping this interpretation explicit avoids mixing
    the dimensionally different continuous-white-noise ``dt^3/3`` model with
    the schema's acceleration units.
    """

    dt = max(0.0, float(dt_s))
    covariance = np.zeros((layout.dimension, layout.dimension), dtype=float)
    if dt == 0.0:
        return covariance
    spectral_density = float(world.target.acceleration_std_m_s2) ** 2
    physical = np.zeros((6, 6), dtype=float)
    for axis in range(3):
        physical[axis, axis] = spectral_density * dt**4 / 4.0
        physical[axis, axis + 3] = spectral_density * dt**3 / 2.0
        physical[axis + 3, axis] = physical[axis, axis + 3]
        physical[axis + 3, axis + 3] = spectral_density * dt**2
    scales = layout.scales()[:6]
    covariance[:6, :6] = physical / np.outer(scales, scales)
    nodes = {node.node_id: node for node in world.nodes}
    all_scales = layout.scales()
    for node_id in layout.node_ids:
        node = nodes[node_id]
        clock_index = layout.clock_index(node_id)
        frequency_index = layout.frequency_index(node_id)
        covariance[clock_index, clock_index] = (
            node.clock_random_walk_std_s_sqrt_s**2 * dt / all_scales[clock_index] ** 2
        )
        covariance[frequency_index, frequency_index] = (
            node.frequency_random_walk_std_hz_sqrt_s**2 * dt / all_scales[frequency_index] ** 2
        )
    clock_indices = [layout.clock_index(node_id) for node_id in layout.node_ids]
    frequency_indices = [layout.frequency_index(node_id) for node_id in layout.node_ids]
    if world.clock_common_random_walk_std_s_sqrt_s > 0.0:
        common_clock = world.clock_common_random_walk_std_s_sqrt_s**2 * dt
        covariance[np.ix_(clock_indices, clock_indices)] += common_clock / np.outer(
            all_scales[clock_indices], all_scales[clock_indices]
        )
    if world.frequency_common_random_walk_std_hz_sqrt_s > 0.0:
        common_frequency = world.frequency_common_random_walk_std_hz_sqrt_s**2 * dt
        covariance[np.ix_(frequency_indices, frequency_indices)] += common_frequency / np.outer(
            all_scales[frequency_indices], all_scales[frequency_indices]
        )
    configurations = {item.configuration_id: item for item in world.receive_chain_configurations}
    for configuration_id in layout.receive_configuration_ids:
        configuration = configurations[configuration_id]
        state_index = layout.effective_delay_index(configuration_id)
        covariance[state_index, state_index] = (
            configuration.delay_random_walk_std_s_sqrt_s**2
            * dt
            / all_scales[state_index] ** 2
        )
    return covariance


def propagate_normalized_covariance(
    covariance: np.ndarray, world: PhysicalWorld, layout: StateLayout, dt_s: float
) -> np.ndarray:
    transition = normalized_state_transition(layout, dt_s)
    result = transition @ covariance @ transition.T + normalized_process_covariance(world, layout, dt_s)
    return _sym(result)


def schedule_feasibility(world: PhysicalWorld, actions: Sequence[RadarAction]) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    if len(actions) > world.max_schedule_size:
        violations.append("schedule_size_budget")
    if len({a.action_id for a in actions}) != len(actions):
        violations.append("duplicate_action")
    actions_by_slot: dict[int, list[RadarAction]] = {}
    for action in actions:
        actions_by_slot.setdefault(action.resource_slot, []).append(action)
    slot_durations = {
        slot: max(world.waveform(action.waveform_id).coherent_processing_interval_s for action in slot_actions)
        for slot, slot_actions in actions_by_slot.items()
    }
    usage = schedule_resource_usage(world, actions)
    if usage.dwell_s > world.max_total_dwell_s + 1e-12:
        violations.append("total_dwell_budget")
    if usage.coherent_processing_s > world.max_total_coherent_processing_s + 1e-12:
        violations.append("total_coherent_processing_time_budget")
    if usage.energy_j > world.max_total_energy_j + 1e-12:
        violations.append("total_energy_budget")
    if usage.emission_exposure_j > world.max_total_emission_j + 1e-12:
        violations.append("total_emission_budget")
    if usage.communication_mb > world.max_total_communication_mb + 1e-12:
        violations.append("total_communication_budget")
    if usage.decision_latency_s > world.max_decision_latency_s + 1e-12:
        violations.append("decision_latency_budget")
    active_by_slot: dict[int, list[RadarAction]] = {}
    node_energy: dict[str, float] = {}
    for action in actions:
        if action.kind.startswith("active_"):
            active_by_slot.setdefault(action.resource_slot, []).append(action)
        energy_by_node = action.metadata.get("energy_j_by_node", {})
        if not isinstance(energy_by_node, dict):
            violations.append(f"invalid_energy_allocation:{action.action_id}")
        else:
            for node_id, value in energy_by_node.items():
                node_energy[str(node_id)] = node_energy.get(str(node_id), 0.0) + float(value)
    for node_id, value in node_energy.items():
        if value > world.node(node_id).max_energy_j + 1e-12:
            violations.append(f"node_energy_budget:{node_id}")
    for slot, slot_actions in actions_by_slot.items():
        duration = slot_durations[slot]
        rx_actions: dict[str, list[RadarAction]] = {}
        tx_actions: dict[str, list[RadarAction]] = {}
        communication_by_node: dict[str, float] = {}
        for action in slot_actions:
            if action.tx_node_id is not None and action.kind.startswith("active_"):
                tx_actions.setdefault(action.tx_node_id, []).append(action)
            for node_id in action.rx_node_ids:
                rx_actions.setdefault(node_id, []).append(action)
            communication_allocations = action.metadata.get("communication_mb_by_node", {})
            if not isinstance(communication_allocations, dict):
                violations.append(f"invalid_communication_allocation:{action.action_id}")
            else:
                for node_id, value in communication_allocations.items():
                    communication_by_node[str(node_id)] = communication_by_node.get(str(node_id), 0.0) + float(value)
        for node_id, receive_actions in rx_actions.items():
            node = world.node(node_id)
            if len(receive_actions) > node.simultaneous_receive_channels:
                violations.append(f"receive_channel_limit:{slot}:{node_id}")
            conflicting = {
                (tx_action.action_id, rx_action.action_id)
                for tx_action in tx_actions.get(node_id, [])
                for rx_action in receive_actions
                if tx_action.action_id != rx_action.action_id
            }
            if conflicting and not node.full_duplex_capable:
                violations.append(f"cross_action_half_duplex:{slot}:{node_id}")
        for node_id, communication_mb in communication_by_node.items():
            node = world.node(node_id)
            capacity_mb = node.communication_capacity_mbps * duration / 8.0
            if communication_mb > capacity_mb + 1e-12:
                violations.append(f"communication_capacity:{slot}:{node_id}")
        if world.effective_delay_model_enabled and world.calibration_resource_mode == "exclusive_receive_chain":
            configuration_roles: dict[str, set[str]] = {}
            for action in slot_actions:
                role = "calibration" if action.kind.startswith("calibration") else "sensing"
                for configuration in action_receive_configurations(world, action):
                    configuration_roles.setdefault(configuration.configuration_id, set()).add(role)
            for configuration_id, roles in configuration_roles.items():
                if roles == {"calibration", "sensing"}:
                    violations.append(f"calibration_sensing_receive_chain_conflict:{slot}:{configuration_id}")
    for slot, active in active_by_slot.items():
        if len(active) > world.max_concurrent_active:
            violations.append(f"concurrent_active_limit:{slot}")
        for i, left in enumerate(active):
            for right in active[i + 1 :]:
                if left.tx_node_id == right.tx_node_id:
                    violations.append(f"same_transmitter_half_duplex:{slot}:{left.tx_node_id}")
                l0, l1 = _frequency_interval(world, left); r0, r1 = _frequency_interval(world, right)
                if min(l1, r1) > max(l0, r0):
                    violations.append(f"cochannel_active_conflict:{slot}:{left.action_id}:{right.action_id}")
    return not violations, tuple(sorted(set(violations)))


def lower_tail_cvar(values: Iterable[float], probability: float) -> float:
    arr = np.sort(np.asarray(list(values), dtype=float))
    if arr.size == 0:
        raise ValueError("CVaR needs at least one scenario")
    count = max(1, int(math.ceil(float(probability) * arr.size)))
    return float(arr[:count].mean())


def _physical_covariance(effective_information_normalized: np.ndarray, layout: StateLayout) -> np.ndarray:
    covariance_normalized = stable_inverse_spd(effective_information_normalized, name="effective_target_information")
    target_scales = layout.scales()[: layout.target_dim]
    return np.diag(target_scales) @ covariance_normalized @ np.diag(target_scales)


def position_task_information_and_covariance(joint_information: np.ndarray, layout: StateLayout) -> tuple[np.ndarray, np.ndarray]:
    """Position-task information after marginalizing velocity and sensor nuisance."""

    effective_position, _ = schur_effective_information(joint_information, 3)
    covariance_normalized = stable_inverse_spd(effective_position, name="effective_position_task_information")
    scales = layout.scales()[:3]
    covariance_physical = np.diag(scales) @ covariance_normalized @ np.diag(scales)
    return effective_position, covariance_physical


def position_task_gain(joint_information: np.ndarray, prior_information: np.ndarray, layout: StateLayout, d_opt_weight: float) -> tuple[float, np.ndarray]:
    """Blend task-volume (D-opt) and position-MSE (A-opt) risk reduction."""

    posterior_effective, posterior_covariance = position_task_information_and_covariance(joint_information, layout)
    prior_effective, prior_covariance = position_task_information_and_covariance(prior_information, layout)
    d_gain_per_dimension = (stable_logdet_spd(posterior_effective, name="posterior_position_task_information") - stable_logdet_spd(prior_effective, name="prior_position_task_information")) / 3.0
    a_gain = math.log(max(float(np.trace(prior_covariance)), 1e-300) / max(float(np.trace(posterior_covariance)), 1e-300))
    weight = float(d_opt_weight)
    return weight * d_gain_per_dimension + (1.0 - weight) * a_gain, posterior_covariance


def _schedule_terminal_statistics(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    jammer: Jammer | None,
    policy: DecisionPolicy,
    factor_cache: dict[tuple[object, ...], ActionFactor] | None,
) -> tuple[float, np.ndarray, float, float, float]:
    """Sequential terminal covariance and task gain for one adversary scenario."""

    layout = StateLayout.from_world(world)
    prior_information = prior_information_normalized(world, layout)
    initial_covariance = stable_inverse_spd(prior_information, name="schedule_initial_information")
    action_times, terminal_time = schedule_action_times(world, actions)
    actions_by_time: dict[float, list[RadarAction]] = {}
    for action in actions:
        actions_by_time.setdefault(action_times[action.action_id], []).append(action)
    covariance = initial_covariance.copy()
    last_time = 0.0
    missing_information = 0.0
    scenario_key = "nominal" if jammer is None else jammer.jammer_id
    for measurement_time in sorted(actions_by_time):
        covariance = propagate_normalized_covariance(covariance, world, layout, measurement_time - last_time)
        information = stable_inverse_spd(covariance, name="predicted_schedule_covariance")
        for action in actions_by_time[measurement_time]:
            key = action_factor_cache_key(
                world, action.action_id, scenario_key, float(measurement_time)
            )
            if factor_cache is not None and key in factor_cache:
                factor = factor_cache[key]
            else:
                measurement_world = world_propagated_to_time(world, measurement_time)
                factor = build_action_factor(measurement_world, action, jammer)
                if factor_cache is not None:
                    factor_cache[key] = factor
            information += factor.observed_information_normalized
            missing_information += float(np.trace(factor.missing_information_normalized[:6, :6]))
        covariance = stable_inverse_spd(information, name="posterior_schedule_information")
        last_time = measurement_time
    covariance = propagate_normalized_covariance(covariance, world, layout, terminal_time - last_time)
    prior_terminal = propagate_normalized_covariance(initial_covariance, world, layout, terminal_time)
    scales = layout.scales()[:3]
    posterior_position = np.diag(scales) @ covariance[:3, :3] @ np.diag(scales)
    prior_position = np.diag(scales) @ prior_terminal[:3, :3] @ np.diag(scales)
    d_gain = (stable_logdet_spd(prior_position, name="terminal_prior_position_covariance") - stable_logdet_spd(posterior_position, name="terminal_posterior_position_covariance")) / 3.0
    a_gain = math.log(max(float(np.trace(prior_position)), 1e-300) / max(float(np.trace(posterior_position)), 1e-300))
    gain = policy.task_d_opt_weight * d_gain + (1.0 - policy.task_d_opt_weight) * a_gain
    return float(gain), covariance, float(missing_information), float(d_gain), float(a_gain)


def evaluate_schedule(
    world: PhysicalWorld,
    action_ids: Iterable[int],
    *,
    adversary_scenarios: Sequence[Jammer | None] | None = None,
    policy: DecisionPolicy | None = None,
    factor_cache: dict[tuple[object, ...], ActionFactor] | None = None,
) -> ScheduleEvaluation:
    """Evaluate terminal tracking utility with sequential sensing and latency."""

    world.validate()
    policy = policy or DecisionPolicy()
    policy.validate()
    by_id = {action.action_id: action for action in world.actions}
    ids = tuple(sorted(dict.fromkeys(int(value) for value in action_ids)))
    unknown = [value for value in ids if value not in by_id]
    if unknown:
        raise ValueError(f"unknown action identifiers: {unknown}")
    actions = [by_id[value] for value in ids]
    feasible, violations = schedule_feasibility(world, actions)
    scenarios = list(adversary_scenarios) if adversary_scenarios is not None else [None, *world.jammers]
    if not scenarios:
        scenarios = [None]
    gains: list[float] = []
    d_opt_gains: list[float] = []
    a_opt_gains: list[float] = []
    nominal_covariance: np.ndarray | None = None
    nominal_missing = 0.0
    for scenario_index, jammer in enumerate(scenarios):
        gain, covariance, missing, d_opt_gain, a_opt_gain = _schedule_terminal_statistics(
            world,
            actions,
            jammer,
            policy,
            factor_cache,
        )
        gains.append(gain)
        d_opt_gains.append(d_opt_gain)
        a_opt_gains.append(a_opt_gain)
        if scenario_index == 0:
            nominal_covariance = covariance
            nominal_missing = missing
    mean_gain = float(np.mean(gains))
    cvar_gain = lower_tail_cvar(gains, policy.risk_tail_probability)
    robust_info = (1.0 - policy.cvar_weight) * mean_gain + policy.cvar_weight * cvar_gain
    usage = schedule_resource_usage(world, actions)
    decision_cost = decision_cost_for_actions(world, actions, policy)
    utility = robust_info - decision_cost if feasible else -math.inf
    if nominal_covariance is None:
        layout = StateLayout.from_world(world)
        nominal_covariance = stable_inverse_spd(prior_information_normalized(world, layout), name="empty_schedule_prior")
    layout = StateLayout.from_world(world)
    target_scales = layout.scales()[:6]
    target_covariance = np.diag(target_scales) @ nominal_covariance[:6, :6] @ np.diag(target_scales)
    position_rmse = math.sqrt(max(0.0, float(np.trace(target_covariance[:3, :3]))))
    velocity_rmse = math.sqrt(max(0.0, float(np.trace(target_covariance[3:6, 3:6]))))
    return ScheduleEvaluation(
        ids,
        feasible,
        violations,
        gains[0],
        mean_gain,
        cvar_gain,
        float(min(gains)),
        robust_info,
        decision_cost,
        utility,
        position_rmse,
        velocity_rmse,
        nominal_missing,
        tuple(float(value) for value in gains),
        usage.energy_j,
        usage.emission_exposure_j,
        usage.communication_mb,
        usage.decision_latency_s,
        d_opt_gains[0],
        a_opt_gains[0],
        tuple(float(value) for value in d_opt_gains),
        tuple(float(value) for value in a_opt_gains),
    )
