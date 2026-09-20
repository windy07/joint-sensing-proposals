from __future__ import annotations

"""Graph-native candidate generation, exact reranking, and safety shielding."""

from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import combinations, product
import json
import math
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import gammaincinv, ndtri, roots_genlaguerre

from wideband_info_geometry.experiments_v69.v69_operational_information import (
    bures_wasserstein_distance_m,
    d_optimal_radius_m,
    information_gain_nats,
)
from wideband_info_geometry.physics_v58.adversary import canonical_adversary_scenarios
from wideband_info_geometry.physics_v58.dynamic_episode import (
    EpisodeState,
    action_is_dynamically_feasible,
)
from wideband_info_geometry.physics_v58.dynamic_factor_graph import RobustKernel
from wideband_info_geometry.physics_v58.information import (
    lower_tail_cvar,
    schedule_feasibility,
    schedule_resource_usage,
)
from wideband_info_geometry.physics_v58.measurement import StateLayout, build_action_factor
from wideband_info_geometry.physics_v58.effective_delay import (
    automatic_free_calibration_actions,
)
from wideband_info_geometry.physics_v58.schema import Jammer, PhysicalWorld, RadarAction
from wideband_info_geometry.physics_v58.v65_safe_autonomy import (
    CampaignSafetyShield,
    CandidateLibrary,
    MissionIR,
)

from .availability import DistributedAvailabilityManager
from .belief import GraphBeliefEngine, make_operational_action_factor
from .clock_sentinel_design import design_clock_sentinel_sequence
from .innovation_health import (
    DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS,
    InnovationHealthMonitor,
)
from .nonlocal_task_risk import posterior_predictive_schedule_risk
from .planning_detection_probability import PlanningDetectionProbability
from .schedule_update_probability_calibration import (
    ScheduleUpdateProbabilityCalibration,
)
from .contracts import (
    CandidateScore,
    DecisionRecord,
    FactorInformationMode,
    ProposalBackend,
    RuntimeConfig,
    SequentialPolicyMode,
    to_builtin,
)
from .world import (
    CONTROLLER_PROJECTION_VERSION,
    HIDDEN_ADVERSARY_FIELDS,
    HIDDEN_TARGET_FIELDS,
)


CHANCE_CONSTRAINT_PROBABILITY_CONDITIONING = (
    "declared_nominal_no_jammer_condition_x_registered_target_scattering_prior"
)
STRESS_ENSEMBLE_PROBABILITY_ROLE = (
    "robust_objective_and_sensitivity_quadrature_only"
)


def _bounded_agent_hint_quota(
    maximum_candidates: int,
    already_retained: int,
    available_hints: int,
) -> tuple[int, int]:
    """Split the remaining finite library between learned hints and guards.

    Mandatory no-op, sentinel, or identifiability schedules are retained before
    this function is called, so they already count toward the guard half of the
    finite library.  At least half of the original library budget is assigned to
    those retained schedules plus the subsequent source-native role, joint,
    calibration, and geometry guards whenever a learned hint is available.  The
    remaining slots form an explicit learned quota.  Counting the already
    retained guards prevents double reservation and keeps the quota aligned with
    the coordinator's learned-hint budget, while exact reranking remains final.
    """

    maximum = int(maximum_candidates)
    retained = int(already_retained)
    hints = int(available_hints)
    if maximum <= 0 or retained < 0 or hints < 0:
        raise ValueError("candidate and hint quota inputs must be nonnegative")
    if hints == 0 or retained >= maximum:
        return 0, max(0, maximum - retained)
    analytic_guard_target = min(
        maximum - 1,
        max(2, int(math.ceil(0.5 * maximum))),
    )
    analytic_guard_reserve = max(0, analytic_guard_target - retained)
    learned_quota = min(
        hints,
        max(0, maximum - retained - analytic_guard_reserve),
    )
    return int(learned_quota), int(analytic_guard_reserve)


def _velocity_independent_verification_rows(
    layout: StateLayout,
    factor: Any,
    *,
    include_clock_only_rows: bool,
) -> np.ndarray:
    r"""Return normalized rows that identify position without target velocity.

    A Doppler row may have a nonzero derivative with respect to position, but
    it also depends on the unknown target velocity.  Projecting that row onto
    the first three columns and then taking a rank therefore does not certify
    position observability.  This helper retains only rows whose normalized
    target-velocity derivative vanishes to numerical precision.  Clock-only
    calibration rows are optionally retained for the joint position/clock
    certificate even though they carry no target-position derivative.
    """

    normalized = (
        np.asarray(factor.jacobian_physical, dtype=float)
        * layout.scales()[None, :]
    )
    retained: list[np.ndarray] = []
    for row in normalized:
        row_norm = float(np.linalg.norm(row))
        tolerance = max(1.0e-15, 512.0 * np.finfo(float).eps * row_norm)
        velocity_norm = float(np.linalg.norm(row[3:6]))
        position_norm = float(np.linalg.norm(row[:3]))
        clock_norm = float(
            np.linalg.norm(row[layout.clock_offset : layout.frequency_offset])
        )
        if velocity_norm > tolerance:
            continue
        if position_norm > tolerance or (
            include_clock_only_rows and clock_norm > tolerance
        ):
            retained.append(row)
    if not retained:
        return np.zeros((0, layout.dimension), dtype=float)
    return np.vstack(retained)


def _clock_support_node_ids(
    layout: StateLayout,
    factor: Any,
) -> tuple[str, ...]:
    """Return clock-state nodes touched by one structural measurement factor."""

    clock_rows = np.asarray(factor.jacobian_physical, dtype=float)[
        :, layout.clock_offset : layout.frequency_offset
    ]
    if clock_rows.size == 0:
        return ()
    scale = max(float(np.max(np.abs(clock_rows))), 1.0)
    tolerance = 1.0e-12 * scale
    return tuple(
        str(layout.node_ids[index])
        for index in range(clock_rows.shape[1])
        if bool(np.any(np.abs(clock_rows[:, index]) > tolerance))
    )


def _differential_clock_verification_certificate(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    *,
    factor_by_action: Mapping[int, Any] | None = None,
    require_position_separating_rank: bool = False,
) -> dict[str, Any]:
    r"""Certify a target / differential-clock verification experiment.

    The certificate is structural and controller-visible.  It uses normalized
    Jacobian columns, removes the unobservable common-clock gauge, and requires
    a schedule to combine: (i) clock-invariant active-monostatic delay, (ii) a
    clock-sensitive cross-node timing observation, and (iii) explicit time
    calibration.  A rank gain beyond the three target-position dimensions is
    required so the schedule can discriminate at least one differential-clock
    direction instead of merely adding another target measurement.
    """

    layout = StateLayout.from_world(world)
    factors = []
    for action in actions:
        factor = (
            factor_by_action.get(int(action.action_id))
            if factor_by_action is not None
            else None
        )
        if factor is None:
            factor = build_action_factor(world, action, None)
        factors.append((action, factor))
    if factors:
        if require_position_separating_rank:
            row_blocks = [
                _velocity_independent_verification_rows(
                    layout,
                    factor,
                    include_clock_only_rows=True,
                )
                for _, factor in factors
            ]
            row_blocks = [block for block in row_blocks if block.shape[0] > 0]
            normalized_rows = (
                np.vstack(row_blocks)
                if row_blocks
                else np.zeros((0, layout.dimension), dtype=float)
            )
        else:
            normalized_rows = np.vstack(
                [
                    np.asarray(factor.jacobian_physical, dtype=float)
                    * layout.scales()[None, :]
                    for _, factor in factors
                ]
            )
    else:
        normalized_rows = np.zeros((0, layout.dimension), dtype=float)
    target_rows = normalized_rows[:, :3]
    clock_rows = normalized_rows[
        :, layout.clock_offset : layout.frequency_offset
    ]
    node_count = len(layout.node_ids)
    if node_count > 1:
        centering = np.eye(node_count) - np.ones((node_count, node_count)) / node_count
        left, _, _ = np.linalg.svd(centering, full_matrices=True)
        contrast_basis = left[:, : node_count - 1]
        clock_contrast_rows = clock_rows @ contrast_basis
    else:
        clock_contrast_rows = np.zeros((normalized_rows.shape[0], 0), dtype=float)
    joint_rows = np.hstack([target_rows, clock_contrast_rows])
    target_rank = int(np.linalg.matrix_rank(target_rows)) if target_rows.size else 0
    clock_rank = (
        int(np.linalg.matrix_rank(clock_contrast_rows))
        if clock_contrast_rows.size
        else 0
    )
    joint_rank = int(np.linalg.matrix_rank(joint_rows)) if joint_rows.size else 0

    anchor_ids: list[int] = []
    clock_sensitive_ids: list[int] = []
    calibration_ids: list[int] = []
    clock_support_by_action: dict[int, tuple[str, ...]] = {}
    for action, factor in factors:
        names = tuple(str(value) for value in factor.measurement_names)
        action_id = int(action.action_id)
        clock_support_by_action[action_id] = _clock_support_node_ids(
            layout, factor
        )
        if action.kind == "active_monostatic" and "monostatic_delay_s" in names:
            anchor_ids.append(action_id)
        if action.kind in {"active_bistatic", "tdoa"} and any(
            name in {"bistatic_delay_s", "tdoa_s"} for name in names
        ):
            clock_sensitive_ids.append(action_id)
        if action.kind == "calibration_time" and any(
            name.endswith("_s")
            and ("calibration" in name or "clock" in name or "time" in name)
            for name in names
        ):
            calibration_ids.append(action_id)

    connected_records = []
    for calibration_id in calibration_ids:
        calibration_support = set(clock_support_by_action[calibration_id])
        for sensing_id in clock_sensitive_ids:
            overlap = sorted(
                calibration_support.intersection(
                    clock_support_by_action[sensing_id]
                )
            )
            if overlap:
                connected_records.append(
                    {
                        "calibration_action_id": int(calibration_id),
                        "clock_sensitive_action_id": int(sensing_id),
                        "shared_clock_node_ids": overlap,
                    }
                )
    clock_calibration_connected = bool(connected_records)

    complete = bool(
        anchor_ids
        and clock_sensitive_ids
        and calibration_ids
        and clock_calibration_connected
        and target_rank >= 3
        and clock_rank >= 1
        and joint_rank >= target_rank + 1
    )
    return {
        "status": "passed",
        "differential_clock_identification_complete": complete,
        "clock_invariant_monostatic_anchor_action_ids": sorted(anchor_ids),
        "clock_sensitive_cross_node_action_ids": sorted(clock_sensitive_ids),
        "explicit_time_calibration_action_ids": sorted(calibration_ids),
        "clock_support_node_ids_by_action": {
            str(action_id): list(clock_support_by_action[action_id])
            for action_id in sorted(clock_support_by_action)
        },
        "clock_calibration_cross_node_connected": bool(
            clock_calibration_connected
        ),
        "clock_calibration_cross_node_overlap_records": connected_records,
        "target_position_structural_rank": target_rank,
        "differential_clock_contrast_rank": clock_rank,
        "target_clock_joint_structural_rank": joint_rank,
        "target_clock_separation_rank_gain": int(joint_rank - target_rank),
        "common_clock_gauge_removed": True,
        "normalized_jacobian_coordinates": True,
        "position_separating_rank_required": bool(
            require_position_separating_rank
        ),
        "target_velocity_sensitive_rows_excluded": bool(
            require_position_separating_rank
        ),
        "action_id_specific_rule_used": False,
        "truth_or_future_event_read": False,
        "certificate_definition": (
            "rank3_velocity_independent_target_plus_nonzero_differential_clock_"
            "rank_with_"
            "monostatic_anchor_and_clock_connected_cross_node_timing_"
            "plus_explicit_time_calibration"
            if require_position_separating_rank
            else
            "legacy_rank3_target_plus_nonzero_differential_clock_rank_with_"
            "monostatic_anchor_and_clock_connected_cross_node_timing_"
            "plus_explicit_time_calibration"
        ),
    }


def _realized_innovation_verification_certificate(
    world: PhysicalWorld,
    selected_action_ids: Sequence[int],
    observations: Sequence[Any],
    *,
    epoch: int,
    alarm_hypotheses: Sequence[str],
    minimum_sensing_actions: int,
    require_position_separating_rank: bool = False,
) -> dict[str, Any]:
    r"""Certify the *arrived and accepted* verification experiment.

    A planned rank certificate is not evidence that the corresponding factors
    reached the authoritative posterior.  This receipt therefore uses only
    committed action identifiers, observations whose transport arrival is no
    later than the current epoch, and receiver-admitted detections.  Generic
    model-health verification requires rank-three target geometry and an
    absolute monostatic range anchor.  The differential-clock hypothesis also
    requires a connected, admitted clock-calibration experiment.
    """

    if int(minimum_sensing_actions) < 1:
        raise ValueError("minimum realized verification sensing count must be positive")
    by_id = {int(action.action_id): action for action in world.actions}
    selected = {int(value) for value in selected_action_ids}

    def field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    accepted_ids = []
    rejected_or_delayed_ids = []
    for observation in observations:
        action_id = int(field(observation, "action_id", -1))
        if action_id not in selected or action_id not in by_id:
            continue
        arrival_epoch = int(field(observation, "arrival_epoch", epoch))
        factor_id = field(observation, "factor_id", None)
        accepted = bool(
            field(observation, "detected", False)
            and arrival_epoch <= int(epoch)
            and factor_id is not None
        )
        if accepted:
            accepted_ids.append(action_id)
        else:
            rejected_or_delayed_ids.append(action_id)
    accepted_ids = sorted(set(accepted_ids))
    accepted_actions = tuple(by_id[action_id] for action_id in accepted_ids)
    layout = StateLayout.from_world(world)
    accepted_factors = {
        int(action.action_id): build_action_factor(world, action, None)
        for action in accepted_actions
    }
    target_rows = []
    sensing_ids = []
    anchor_ids = []
    for action in accepted_actions:
        if action.kind.startswith("calibration"):
            continue
        sensing_ids.append(int(action.action_id))
        factor = accepted_factors[int(action.action_id)]
        names = tuple(str(value) for value in factor.measurement_names)
        primary_indices = [
            index
            for index, name in enumerate(names)
            if "calibration" not in name
        ]
        if primary_indices and not require_position_separating_rank:
            rows = np.asarray(factor.jacobian_physical, dtype=float)[
                primary_indices, :3
            ]
            target_rows.append(rows * layout.scales()[None, :3])
        if (
            action.kind == "active_monostatic"
            and "monostatic_delay_s" in names
        ):
            anchor_ids.append(int(action.action_id))
    if require_position_separating_rank:
        (
            _,
            position_separating_action_count,
            position_separating_singular_values,
            target_rank,
            position_separating_tolerance,
        ) = _position_separating_observation_geometry(
            world,
            accepted_actions,
            factor_by_action=accepted_factors,
        )
    else:
        stacked_target = (
            np.vstack(target_rows)
            if target_rows
            else np.zeros((0, 3), dtype=float)
        )
        target_rank = (
            int(np.linalg.matrix_rank(stacked_target))
            if stacked_target.size
            else 0
        )
        position_separating_action_count = len(sensing_ids)
        position_separating_singular_values = np.zeros(0, dtype=float)
        position_separating_tolerance = 0.0
    generic_complete = bool(
        len(sensing_ids) >= int(minimum_sensing_actions)
        and target_rank >= 3
        and anchor_ids
    )
    clock_certificate = _differential_clock_verification_certificate(
        world,
        accepted_actions,
        factor_by_action=accepted_factors,
        require_position_separating_rank=require_position_separating_rank,
    )
    clock_required = bool(
        DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS in set(alarm_hypotheses)
    )
    complete = bool(
        generic_complete
        and (
            not clock_required
            or clock_certificate["differential_clock_identification_complete"]
        )
    )
    return {
        "status": "passed",
        "realized_verification_evidence_complete": complete,
        "generic_rank3_anchor_evidence_complete": generic_complete,
        "differential_clock_evidence_required": clock_required,
        "differential_clock_evidence_complete": bool(
            clock_certificate["differential_clock_identification_complete"]
        ),
        "accepted_committed_action_ids": accepted_ids,
        "rejected_or_not_yet_arrived_committed_action_ids": sorted(
            set(rejected_or_delayed_ids)
        ),
        "accepted_sensing_action_ids": sorted(sensing_ids),
        "accepted_absolute_range_anchor_action_ids": sorted(anchor_ids),
        "accepted_target_position_structural_rank": int(target_rank),
        "accepted_position_separating_action_count": int(
            position_separating_action_count
        ),
        "accepted_position_separating_singular_values": [
            float(value) for value in position_separating_singular_values
        ],
        "accepted_position_separating_rank_tolerance": float(
            position_separating_tolerance
        ),
        "position_separating_rank_required": bool(
            require_position_separating_rank
        ),
        "target_velocity_sensitive_rows_excluded": bool(
            require_position_separating_rank
        ),
        "minimum_accepted_sensing_actions": int(minimum_sensing_actions),
        "differential_clock_certificate": clock_certificate,
        "uses_only_arrived_observations": True,
        "action_id_specific_rule_used": False,
        "truth_or_future_event_read": False,
    }


def _innovation_verification_selection_rows(
    rows: Sequence[CandidateScore],
    *,
    retain_admissible_set: bool = False,
    minimum_sensing_actions: int = 2,
    probability_indifference_band: float = 0.05,
    geometry_floor_fraction: float = 0.25,
    require_absolute_range_anchor_when_available: bool = False,
    require_differential_clock_identification_when_available: bool = False,
    require_position_separating_rank: bool = False,
) -> tuple[tuple[CandidateScore, ...], dict[str, Any]]:
    r"""Retain causally admissible innovation-verification candidates.

    Every row has already passed physical/resource checks and exact graph
    reranking.  During a causal innovation alarm, local log-determinant gain is
    not by itself a reliable mode-discrimination objective.  We therefore use

    ``P(full-rank correct update) * sigma_min(G)``,

    where ``G`` stacks unit-normalized position-measurement gradients.  This is
    dimensionless, action-ID agnostic, and rewards both directional diversity
    and the probability that the discriminating measurements are actually
    delivered and associated.  Frozen predecessor runs retain only the exact
    maximum of this score.  The opt-in R31 rule instead retains every row in a
    preregistered probability/geometry indifference region so that the exact
    nonlocal-risk selector, rather than this local diagnostic, resolves the
    remaining ambiguity.

    If at least one row in the admissible pool has a structurally verified
    active-monostatic delay observation, R31 can require that absolute-range
    anchor.  The certificate is produced from the implemented measurement
    names and action kind, never an action identifier or simulator truth.
    """

    if minimum_sensing_actions < 1:
        raise ValueError("minimum verification sensing actions must be positive")
    if not 0.0 <= probability_indifference_band <= 1.0:
        raise ValueError("verification probability band must lie in [0,1]")
    if not 0.0 < geometry_floor_fraction <= 1.0:
        raise ValueError("verification geometry floor must lie in (0,1]")
    original = tuple(rows)
    geometry_rank_key = (
        "position_separating_geometry_rank"
        if require_position_separating_rank
        else "position_observation_geometry_rank"
    )
    geometry_singular_key = (
        "position_separating_geometry_singular_values"
        if require_position_separating_rank
        else "position_observation_geometry_singular_values"
    )
    clock_identification_available = any(
        bool(
            row.metadata.get("differential_clock_verification", {}).get(
                "differential_clock_identification_complete", False
            )
        )
        and int(row.metadata.get(geometry_rank_key, 0)) >= 3
        and len(row.metadata.get(geometry_singular_key, ())) >= 3
        for row in original
    )
    use_joint_clock_probability = bool(
        retain_admissible_set
        and require_differential_clock_identification_when_available
        and clock_identification_available
        and require_position_separating_rank
    )
    probability_key = (
        "differential_clock_verification_probability_lcb"
        if use_joint_clock_probability
        else (
            "position_separating_update_probability_lcb"
            if require_position_separating_rank
            else "expected_full_rank_position_update_probability"
        )
    )
    scored: list[
        tuple[CandidateScore, float, float, float, bool, bool, int]
    ] = []
    for row in original:
        metadata = row.metadata
        singular_values = tuple(
            float(value)
            for value in metadata.get(
                geometry_singular_key, ()
            )
        )
        rank = int(metadata.get(geometry_rank_key, 0))
        probability = float(metadata.get(probability_key, 0.0))
        if not row.action_ids or rank < 3 or len(singular_values) < 3:
            continue
        minimum_singular_value = float(min(singular_values))
        score = probability * minimum_singular_value
        if not all(
            math.isfinite(value)
            for value in (probability, minimum_singular_value, score)
        ):
            raise RuntimeError("innovation verification geometry is nonfinite")
        scored.append(
            (
                row,
                score,
                probability,
                minimum_singular_value,
                bool(metadata.get("active_absolute_range_anchor_present", False)),
                bool(
                    metadata.get(
                        "differential_clock_verification", {}
                    ).get("differential_clock_identification_complete", False)
                ),
                int(metadata.get("sensing_action_count", 0)),
            )
        )
    if not scored:
        return original, {
            "status": "fallback_no_full_rank_candidate",
            "applied": False,
            "candidate_indices": [int(row.candidate_index) for row in original],
            "truth_or_future_event_read": False,
        }
    best = max(item[1] for item in scored)
    tolerance = max(1.0, abs(best)) * 128.0 * np.finfo(float).eps
    anchor_available = any(item[4] for item in scored)
    anchor_required = bool(
        retain_admissible_set
        and require_absolute_range_anchor_when_available
        and anchor_available
    )
    clock_identification_required = bool(
        retain_admissible_set
        and require_differential_clock_identification_when_available
        and clock_identification_available
    )
    if retain_admissible_set:
        pool = tuple(scored)
        if clock_identification_required:
            pool = tuple(item for item in pool if item[5])
        if anchor_required:
            pool = tuple(item for item in pool if item[4])
        maximum_probability = max(item[2] for item in pool)
        maximum_minimum_singular = max(item[3] for item in pool)
        probability_floor = max(
            0.0, maximum_probability - probability_indifference_band
        )
        geometry_floor = geometry_floor_fraction * maximum_minimum_singular
        selected_records = tuple(
            item
            for item in pool
            if item[2] + tolerance >= probability_floor
            and item[3] + tolerance >= geometry_floor
            and item[6] >= minimum_sensing_actions
        )
        if not selected_records:
            # The source rows have already passed the exact physical/resource
            # gates.  Falling back to the best row inside the same causal pool
            # is preferable to silently admitting a row outside that pool.
            selected_records = (
                max(pool, key=lambda item: (item[1], item[2], item[3])),
            )
        selected = tuple(item[0] for item in selected_records)
    else:
        maximum_probability = max(item[2] for item in scored)
        maximum_minimum_singular = max(item[3] for item in scored)
        probability_floor = maximum_probability
        geometry_floor = maximum_minimum_singular
        selected = tuple(item[0] for item in scored if best - item[1] <= tolerance)
    return selected, {
        "status": "passed",
        "applied": True,
        "criterion": (
            f"{probability_key}_times_minimum_unit_gradient_singular_value"
        ),
        "best_verification_score": float(best),
        "numerical_tolerance": float(tolerance),
        "selection_rule": (
            "r31_probability_geometry_admissible_set_then_exact_nonlocal_risk"
            if retain_admissible_set
            else "legacy_single_best_probabilistic_E_optimal_geometry"
        ),
        "retained_admissible_set": bool(retain_admissible_set),
        "minimum_sensing_actions": int(minimum_sensing_actions),
        "maximum_full_rank_update_probability": float(maximum_probability),
        "probability_indifference_band": float(probability_indifference_band),
        "probability_floor": float(probability_floor),
        "maximum_minimum_unit_gradient_singular_value": float(
            maximum_minimum_singular
        ),
        "geometry_floor_fraction": float(geometry_floor_fraction),
        "geometry_floor": float(geometry_floor),
        "absolute_range_anchor_available": bool(anchor_available),
        "absolute_range_anchor_required": bool(anchor_required),
        "absolute_range_anchor_definition": (
            "active_monostatic_action_with_monostatic_delay_s_measurement"
        ),
        "differential_clock_identification_available": bool(
            clock_identification_available
        ),
        "differential_clock_identification_required": bool(
            clock_identification_required
        ),
        "differential_clock_identification_definition": (
            "clock_invariant_monostatic_anchor_plus_clock_sensitive_cross_node_"
            "timing_plus_explicit_time_calibration_with_joint_structural_rank"
        ),
        "position_separating_rank_required": bool(
            require_position_separating_rank
        ),
        "position_geometry_rank_key": geometry_rank_key,
        "position_geometry_singular_value_key": geometry_singular_key,
        "verification_probability_key": probability_key,
        "joint_clock_delivery_probability_used": bool(
            use_joint_clock_probability
        ),
        "candidate_indices": [int(row.candidate_index) for row in selected],
        "all_scored_candidates": [
            {
                "candidate_index": int(row.candidate_index),
                "verification_score": float(score),
                "expected_full_rank_position_update_probability": float(probability),
                "minimum_unit_gradient_singular_value": float(minimum_singular),
                "active_absolute_range_anchor_present": bool(anchor),
                "differential_clock_identification_complete": bool(
                    clock_identification_complete
                ),
                "sensing_action_count": int(sensing_count),
            }
            for (
                row,
                score,
                probability,
                minimum_singular,
                anchor,
                clock_identification_complete,
                sensing_count,
            ) in scored
        ],
        "action_id_specific_rule_used": False,
        "truth_or_future_event_read": False,
        "exact_graph_rerank_rows_only": True,
    }


def _innovation_verification_was_committed(
    *,
    innovation_health_active: bool,
    verification_audit: Mapping[str, Any],
    selected_candidate: int,
    selected_action_ids: Sequence[int],
    schedule_available: bool,
    fail_closed: bool,
) -> bool:
    """Record verification only after the selected RF schedule is committed."""

    verification_indices = {
        int(value) for value in verification_audit.get("candidate_indices", ())
    }
    return bool(
        innovation_health_active
        and verification_audit.get("applied", False)
        and int(selected_candidate) in verification_indices
        and bool(selected_action_ids)
        and schedule_available
        and not fail_closed
    )


def _clock_sentinel_selection_rows(
    rows: Sequence[CandidateScore],
    requirement: Mapping[str, Any],
) -> tuple[tuple[CandidateScore, ...], dict[str, Any]]:
    """Restrict an already admissible exact domain to a registered probe.

    The requirement is constructed before receiver outcomes from the epoch are
    observed.  Every input row has already survived physical, availability,
    transport and track-update admissibility checks.  If no such row contains
    the registered probe, normal mission selection continues and the missed
    sentinel is reported rather than bypassing a safety or task constraint.
    """

    original = tuple(rows)
    scheduled = bool(requirement.get("scheduled_this_epoch", False))
    required_ids = {
        int(value) for value in requirement.get("required_action_ids", ())
    }
    requirement_provenance = {
        key: requirement.get(key)
        for key in (
            "enabled",
            "scheduled_this_epoch",
            "scheduled_by_periodic_interval",
            "scheduled_by_causal_anchor_continuity_loss",
            "coordinated_with_active_innovation_alarm",
            "suspended_by_active_innovation_alarm",
            "previous_clock_invariant_action_ids",
            "previous_clock_invariant_action_ids_still_available",
            "scheduled_clock_edge_action_id",
            "scheduled_clock_probe_action_id",
            "scheduled_clock_probe_action_kind",
            "scheduled_clock_probe_target_independent",
            "scheduled_monostatic_anchor_action_id",
            "scheduled_monostatic_anchor_action_ids",
            "requested_distinct_monostatic_anchor_count",
            "selected_distinct_monostatic_anchor_count",
            "monostatic_anchor_required",
            "design_frozen_on_first_controller_visible_world_snapshot",
            "event_schedule_or_realized_clock_state_read",
            "receiver_residual_read",
        )
    }
    if not scheduled:
        return original, {
            **requirement_provenance,
            "status": "not_scheduled",
            "applied": False,
            "candidate_indices": [int(row.candidate_index) for row in original],
            "required_action_ids": sorted(required_ids),
            "truth_or_future_event_read": False,
        }
    if not required_ids:
        return original, {
            **requirement_provenance,
            "status": "scheduled_but_no_feasible_registered_probe",
            "applied": False,
            "candidate_indices": [int(row.candidate_index) for row in original],
            "required_action_ids": [],
            "truth_or_future_event_read": False,
        }
    selected = tuple(
        row for row in original if required_ids.issubset(set(row.action_ids))
    )
    if not selected:
        return original, {
            **requirement_provenance,
            "status": "registered_probe_not_mission_admissible",
            "applied": False,
            "candidate_indices": [int(row.candidate_index) for row in original],
            "required_action_ids": sorted(required_ids),
            "truth_or_future_event_read": False,
        }
    return selected, {
        **requirement_provenance,
        "status": "passed",
        "applied": True,
        "selection_rule": (
            "registered_truth_isolated_timing_probe_and_monostatic_anchor_"
            "then_exact_rerank_within_the_constrained_domain"
        ),
        "candidate_indices": [int(row.candidate_index) for row in selected],
        "required_action_ids": sorted(required_ids),
        "truth_or_future_event_read": False,
        "exact_graph_rerank_rows_only": True,
        "safety_or_track_constraint_bypassed": False,
    }


def _clock_sentinel_trigger(
    *,
    enabled: bool,
    epoch: int,
    start_epoch: int,
    interval_epochs: int,
    innovation_health_active: bool,
    previous_clock_invariant_action_ids: Sequence[int],
    available_action_ids: set[int],
) -> dict[str, Any]:
    """Construct a causal sentinel trigger from the current filtration.

    Periodic probing and innovation-health verification are complementary.
    An immediate trigger is also raised when every action that supplied the
    latest accepted clock-invariant update has become unavailable.  These
    inputs are controller-visible before the decision; neither simulator truth
    nor the registered event schedule is consulted.
    """

    prior_ids = {int(value) for value in previous_clock_invariant_action_ids}
    periodic_due = bool(
        enabled
        and int(epoch) >= int(start_epoch)
        and (int(epoch) - int(start_epoch)) % int(interval_epochs) == 0
    )
    continuity_loss = bool(
        enabled and prior_ids and prior_ids.isdisjoint(available_action_ids)
    )
    scheduled = bool(periodic_due or continuity_loss)
    return {
        "scheduled": scheduled,
        "periodic_due": periodic_due,
        "causal_anchor_continuity_loss": continuity_loss,
        "previous_clock_invariant_action_ids": sorted(prior_ids),
        "previous_clock_invariant_action_ids_still_available": sorted(
            prior_ids.intersection(available_action_ids)
        ),
        "coordinated_with_active_innovation_alarm": bool(
            scheduled and innovation_health_active
        ),
        "suspended_by_active_innovation_alarm": False,
        "truth_or_future_event_read": False,
    }


def _monostatic_anchor_identity_nodes(action: RadarAction) -> tuple[str, ...]:
    """Return physical sensor identities, excluding transport/fusion relays.

    ``DistributedAvailabilityManager.factor_participants`` deliberately adds
    the fusion node because every factor must be transported there. That set
    is a provenance/transport set, not an independent-sensor set. Using it to
    test anchor diversity makes every pair appear to share the fusion node and
    prevents redundant monostatic acquisition by construction.
    """

    if action.kind != "active_monostatic":
        return ()
    physical = {
        str(value)
        for value in (action.tx_node_id, *action.rx_node_ids)
        if value is not None
    }
    return tuple(sorted(physical))


def _conditioned_probability_and_stress_mean(
    values: Sequence[float],
    scenario_weights: Sequence[float],
    condition_mask: Sequence[bool],
) -> tuple[float, float]:
    r"""Separate a conditional event probability from a stress-set mean.

    ``scenario_weights`` are design weights for the robust scenario quadrature.
    They are not an empirical occurrence law for jammer states.  Conditioning
    on the declared no-jammer environment first, and only then averaging over
    the registered target-scattering prior, yields the probability used by the
    track-update chance constraint.  The all-scenario mean remains a useful
    robustness diagnostic but is not frequency-calibratable against nominal
    episodes.
    """

    probabilities = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(scenario_weights, dtype=float).reshape(-1)
    mask = np.asarray(condition_mask, dtype=bool).reshape(-1)
    if probabilities.size == 0 or probabilities.shape != weights.shape:
        raise ValueError("scenario probabilities and weights are empty or misaligned")
    if mask.shape != probabilities.shape or not np.any(mask):
        raise ValueError("chance-constraint condition selects no planning scenario")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
        or float(np.sum(weights[mask])) <= 0.0
    ):
        raise ValueError("scenario probabilities or weights are invalid")
    normalized = weights / float(np.sum(weights))
    conditioned_weights = normalized[mask]
    conditioned_weights /= float(np.sum(conditioned_weights))
    conditioned = float(np.dot(conditioned_weights, probabilities[mask]))
    stress_mean = float(np.dot(normalized, probabilities))
    return conditioned, stress_mean


def _track_update_constrained_selection_rows(
    exact_rows: Sequence[CandidateScore],
    *,
    phase: str,
    required_probability: float,
    probability_key_override: str | None = None,
    infeasible_indifference_band: float = 0.0,
    tolerance: float = 1.0e-12,
) -> tuple[tuple[CandidateScore, ...], dict[str, Any]]:
    """Apply the track-update chance constraint before utility maximization.

    If the registered probability threshold is infeasible inside the exact
    shortlist, candidates within the registered minimum-important probability
    band of the best forecast form one admissibility class.  The downstream
    exact/sequential selector then resolves the task-information/resource
    tradeoff inside that class.  A zero band reproduces the historical strict
    lexicographic rule.
    """

    rows = tuple(exact_rows)
    if not rows:
        raise ValueError("track-update constrained selection requires exact rows")
    if (
        not math.isfinite(infeasible_indifference_band)
        or not 0.0 <= infeasible_indifference_band <= 1.0
    ):
        raise ValueError("infeasible probability indifference band is invalid")
    acquisition_phase = str(phase) in {"acquisition", "reacquisition"}
    probability_key = str(
        probability_key_override
        or (
            "expected_full_rank_position_update_probability"
            if acquisition_phase
            else "expected_track_maintaining_update_probability"
        )
    )

    structural_rows = rows
    structural_fallback = False
    if acquisition_phase:
        structural_rows = tuple(
            row
            for row in rows
            if int(row.metadata.get("position_observation_geometry_rank", 0)) >= 3
            and int(row.metadata.get("sensing_action_count", 0)) >= 2
        )
        if not structural_rows:
            structural_rows = rows
            structural_fallback = True

    probabilities = {
        int(row.candidate_index): float(row.metadata.get(probability_key, 0.0))
        for row in structural_rows
    }
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 + tolerance
           for value in probabilities.values()):
        raise RuntimeError("track-update probability is outside [0, 1]")
    satisfying = tuple(
        row
        for row in structural_rows
        if probabilities[int(row.candidate_index)] + tolerance
        >= float(required_probability)
    )
    if satisfying:
        selection_rows = satisfying
        fallback_to_minimum_shortfall = False
    else:
        maximum_probability = max(probabilities.values())
        selection_rows = tuple(
            row
            for row in structural_rows
            if probabilities[int(row.candidate_index)]
            >= maximum_probability - infeasible_indifference_band - tolerance
        )
        fallback_to_minimum_shortfall = True
    maximum_probability = max(probabilities.values())
    minimum_shortfall = max(0.0, float(required_probability) - maximum_probability)
    return selection_rows, {
        "probability_key": probability_key,
        "required_probability": float(required_probability),
        "maximum_available_probability": float(maximum_probability),
        "minimum_probability_shortfall": float(minimum_shortfall),
        "constraint_satisfied": bool(satisfying),
        "constraint_satisfying_candidate_indices": [
            int(row.candidate_index) for row in satisfying
        ],
        "selection_candidate_indices": [
            int(row.candidate_index) for row in selection_rows
        ],
        "infeasible_probability_indifference_band": float(
            infeasible_indifference_band
        ),
        "infeasible_candidates_inside_indifference_class": (
            len(selection_rows) if fallback_to_minimum_shortfall else 0
        ),
        "probability_comparison_numerical_tolerance": float(tolerance),
        "indifference_band_applied": bool(
            fallback_to_minimum_shortfall
            and infeasible_indifference_band > 0.0
        ),
        "fallback_to_minimum_shortfall": fallback_to_minimum_shortfall,
        "fallback_to_full_infeasible_set": False,
        "acquisition_geometry_structural_fallback": structural_fallback,
        "hard_constraint_before_exact_objective_when_feasible": True,
        "minimum_shortfall_class_before_exact_objective_when_infeasible": (
            infeasible_indifference_band == 0.0
        ),
        "mission_indifference_class_before_exact_objective_when_infeasible": (
            infeasible_indifference_band > 0.0
        ),
        "lexicographic_constraint_before_exact_objective": True,
    }


def _erasure_robust_position_selection_rows(
    exact_rows: Sequence[CandidateScore],
    *,
    phase: str,
    enabled: bool,
    maximum_jacobian_condition_number: float,
    tolerance: float = 1.0e-12,
) -> tuple[tuple[CandidateScore, ...], dict[str, Any]]:
    """Retain locally stable acquisition schedules after the chance gate.

    The update-probability contract answers whether a usable update arrives.
    This independent contract answers whether the arrived directional frame
    remains stably invertible after any registered sensing-action erasure.  It
    is deliberately restricted to acquisition and reacquisition; a confirmed
    track may coast and is governed by the existing continuity contract.
    """

    rows = tuple(exact_rows)
    if not rows:
        raise ValueError("erasure-robust selection requires exact rows")
    if (
        not math.isfinite(maximum_jacobian_condition_number)
        or maximum_jacobian_condition_number < 1.0
    ):
        raise ValueError("erasure-robust Jacobian condition limit is invalid")
    acquisition_phase = str(phase) in {"acquisition", "reacquisition"}
    applied = bool(enabled and acquisition_phase)
    ratio_threshold = 1.0 / float(maximum_jacobian_condition_number)

    def ratio(row: CandidateScore) -> float:
        certificate = row.metadata.get(
            "erasure_robust_position_frame_certificate", {}
        )
        if not isinstance(certificate, Mapping):
            return 0.0
        value = float(certificate.get("minimum_singular_value_ratio", 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("erasure-robust frame ratio is invalid")
        return value

    ratios = {int(row.candidate_index): ratio(row) for row in rows}
    if not applied:
        return rows, {
            "status": "disabled" if not enabled else "phase_not_applicable",
            "applied": False,
            "phase": str(phase),
            "maximum_jacobian_condition_number": float(
                maximum_jacobian_condition_number
            ),
            "maximum_normal_matrix_condition_number": float(
                maximum_jacobian_condition_number**2
            ),
            "minimum_singular_value_ratio": float(ratio_threshold),
            "constraint_satisfied": None,
            "selection_candidate_indices": [
                int(row.candidate_index) for row in rows
            ],
            "fallback_to_maximum_frame_ratio": False,
            "truth_or_future_event_read": False,
        }

    satisfying = tuple(
        row
        for row in rows
        if ratios[int(row.candidate_index)] + tolerance >= ratio_threshold
        and bool(
            row.metadata.get(
                "erasure_robust_position_frame_certificate", {}
            ).get("full_rank_after_all_evaluated_erasures", False)
        )
    )
    if satisfying:
        selection_rows = satisfying
        fallback = False
        maximum_ratio = max(ratios.values())
    else:
        maximum_ratio = max(ratios.values())
        selection_rows = tuple(
            row
            for row in rows
            if ratios[int(row.candidate_index)] >= maximum_ratio - tolerance
        )
        fallback = True
    return selection_rows, {
        "status": "passed" if satisfying else "threshold_infeasible",
        "applied": True,
        "phase": str(phase),
        "maximum_jacobian_condition_number": float(
            maximum_jacobian_condition_number
        ),
        "maximum_normal_matrix_condition_number": float(
            maximum_jacobian_condition_number**2
        ),
        "minimum_singular_value_ratio": float(ratio_threshold),
        "maximum_available_singular_value_ratio": float(maximum_ratio),
        "constraint_satisfied": bool(satisfying),
        "constraint_satisfying_candidate_indices": [
            int(row.candidate_index) for row in satisfying
        ],
        "selection_candidate_indices": [
            int(row.candidate_index) for row in selection_rows
        ],
        "candidate_minimum_singular_value_ratio": {
            str(index): float(value) for index, value in sorted(ratios.items())
        },
        "fallback_to_maximum_frame_ratio": bool(fallback),
        "fallback_to_unrestricted_exact_set": False,
        "applied_after_track_update_probability_constraint": True,
        "local_directional_stability_not_global_identifiability_claim": True,
        "truth_or_future_event_read": False,
    }


def _effective_sequential_policy_mode(
    config: RuntimeConfig, epoch: int
) -> SequentialPolicyMode:
    if int(epoch) < config.sequential_policy_activation_epoch:
        return SequentialPolicyMode.ROBUST_CVAR
    return config.sequential_policy_mode


def _registered_scenario_weights(rows: Sequence[CandidateScore]) -> np.ndarray:
    """Return the common controller-prior weights carried by exact rows."""

    if not rows:
        raise ValueError("sequential game requires at least one exact row")
    scenario_count = len(rows[0].scenario_ids)
    reference = np.asarray(
        rows[0].metadata.get("scenario_probability_weights", ()), dtype=float
    ).reshape(-1)
    if (
        reference.shape != (scenario_count,)
        or np.any(~np.isfinite(reference))
        or np.any(reference < 0.0)
        or float(np.sum(reference)) <= 0.0
    ):
        raise RuntimeError("exact rows omit a valid registered scenario prior")
    reference = reference / float(np.sum(reference))
    for row in rows[1:]:
        candidate = np.asarray(
            row.metadata.get("scenario_probability_weights", ()), dtype=float
        ).reshape(-1)
        if (
            candidate.shape != reference.shape
            or np.any(~np.isfinite(candidate))
            or np.any(candidate < 0.0)
            or float(np.sum(candidate)) <= 0.0
            or not np.allclose(
                candidate / float(np.sum(candidate)),
                reference,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        ):
            raise RuntimeError("registered scenario prior changed across candidates")
    return reference


def _belief_conditioned_payoff_matrix(
    rows: Sequence[CandidateScore],
    mission: MissionIR,
    registered_weights: np.ndarray,
) -> np.ndarray:
    """Lift the exact one-step objective into a scenario-conditioned payoff.

    The exact row already contains the registered-prior robust objective and the
    current switching charge. We remove that charge because the tree carries
    switching as state, then add only a zero-prior-mean scenario deviation.
    Thus the registered-prior one-step value after root switching is exactly the
    original ``row.exact_score``.
    """

    prior = np.asarray(registered_weights, dtype=float).reshape(-1)
    payoff_rows: list[np.ndarray] = []
    for row in rows:
        scenario_information = (
            float(mission.position_weight)
            * np.asarray(row.scenario_position_gains_nats, dtype=float)
            + float(mission.velocity_weight)
            * np.asarray(row.scenario_velocity_gains_nats, dtype=float)
        )
        if scenario_information.shape != prior.shape or np.any(
            ~np.isfinite(scenario_information)
        ):
            raise RuntimeError("scenario-conditioned exact information is invalid")
        centered = scenario_information - float(scenario_information @ prior)
        base_without_current_switch = float(row.exact_score + row.switching_cost)
        payoff_rows.append(base_without_current_switch + centered)
    payoff = np.asarray(payoff_rows, dtype=float)
    if np.any(~np.isfinite(payoff)):
        raise RuntimeError("sequential-game payoff is non-finite")
    return payoff


def _normalized_switching_cost_matrix(
    rows: Sequence[CandidateScore],
    current_action_ids: Sequence[int],
    *,
    switching_cost_weight: float,
    max_schedule_size: int,
) -> tuple[np.ndarray, int]:
    """Build the same bounded switching law used by exact reranking."""

    signatures = [set(row.action_ids) for row in rows]
    current_signature = set(int(item) for item in current_action_ids)
    previous_row = next(
        (
            index
            for index, signature in enumerate(signatures)
            if signature == current_signature
        ),
        len(rows),
    )
    denominator = max(1, int(max_schedule_size))
    switching_costs = np.empty((len(rows) + 1, len(rows)), dtype=float)
    for previous_index in range(len(rows) + 1):
        previous_signature = (
            signatures[previous_index]
            if previous_index < len(rows)
            else current_signature
        )
        for action_index, signature in enumerate(signatures):
            switching_fraction = min(
                1.0,
                len(previous_signature.symmetric_difference(signature))
                / denominator,
            )
            switching_costs[previous_index, action_index] = (
                float(switching_cost_weight) * switching_fraction
            )
    return switching_costs, int(previous_row)


def _action_role(action: RadarAction) -> str:
    if action.kind.startswith("active"):
        return "active_transmitter"
    if action.kind == "passive_bistatic":
        return "illuminator_reference_processor"
    if action.kind.startswith("calibration"):
        return "calibration_agent"
    return "passive_receiver"


def _scenario_identifier(jammer: Jammer | None) -> str:
    return "nominal" if jammer is None else f"{jammer.kind}:{jammer.jammer_id}"


def _robust_scenarios(world: PhysicalWorld, maximum: int) -> tuple[Jammer | None, ...]:
    bank = canonical_adversary_scenarios(world)
    if maximum >= len(bank):
        return tuple(bank)
    selected: list[Jammer | None] = [None]
    noise = next(
        (item for item in bank if item is not None and item.kind in {"barrage", "spot", "swept"}),
        None,
    )
    deception = next(
        (item for item in reversed(bank) if item is not None and item.kind == "drfm_false_target"),
        None,
    )
    for item in (noise, deception):
        if item is not None and item not in selected and len(selected) < maximum:
            selected.append(item)
    for item in bank:
        if item not in selected and len(selected) < maximum:
            selected.append(item)
    return tuple(selected[:maximum])


@dataclass(frozen=True)
class _JointPlanningScenario:
    """One controller-prior quadrature point, never a simulator-truth sample."""

    world: PhysicalWorld
    jammer: Jammer | None
    probability_weight: float
    scenario_id: str
    rcs_m2: float
    class_log_rcs_z: float
    swerling_texture: float


@dataclass(frozen=True)
class _CandidateScoringContext:
    """Cycle-local, read-only primitives shared by offline exact candidates."""

    source_world: PhysicalWorld
    action_ids: frozenset[int]
    scenarios: tuple[_JointPlanningScenario, ...]
    scenario_factor_by_action: tuple[Mapping[int, Any], ...]
    scenario_automatic_calibration_by_action: tuple[
        Mapping[int, tuple[tuple[RadarAction, Any], ...]], ...
    ]
    scenario_planning_probability_by_action: tuple[Mapping[int, Any], ...]
    structural_factor_by_action: Mapping[int, Any]


def _weighted_lower_tail_mean(
    values: Sequence[float],
    weights: Sequence[float],
    probability: float,
) -> float:
    """Weighted lower-tail CVaR for a finite quadrature distribution."""

    if len(values) != len(weights) or not values:
        raise ValueError("weighted lower-tail mean requires aligned nonempty inputs")
    alpha = float(probability)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("lower-tail probability must lie in (0, 1]")
    pairs = sorted(
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if float(weight) > 0.0
    )
    total = sum(weight for _, weight in pairs)
    if total <= 0.0:
        raise ValueError("quadrature weights must have positive mass")
    remaining = alpha
    integral = 0.0
    for value, raw_weight in pairs:
        weight = raw_weight / total
        admitted = min(weight, remaining)
        integral += admitted * value
        remaining -= admitted
        if remaining <= 1.0e-15:
            break
    if remaining > 1.0e-12:
        raise RuntimeError("lower-tail quadrature did not cover requested mass")
    return float(integral / alpha)


def _registered_scattering_quadrature(
    world: PhysicalWorld,
    order: int,
) -> tuple[tuple[float, float, float, float], ...]:
    r"""Return ``(rcs, weight, z_class, texture)`` prior quadrature points.

    The registered target-class mean RCS is lognormal.  Conditional on that
    class mean, Swerling I/II texture is Gamma(1,1) and Swerling III/IV texture
    is Gamma(2,1/2), both with unit mean.  Gauss-Hermite and generalized
    Gauss-Laguerre rules therefore integrate the declared prior without reading
    the realized target RCS.  The action likelihood is evaluated with Swerling
    case 0 at each point so no pseudo-random controller-side realization is
    introduced on top of the quadrature.
    """

    provenance = dict(world.external_channel_provenance)
    median = float(provenance.get("target_rcs_prior_median_m2", world.target.mean_rcs_m2))
    log_std = float(provenance.get("target_rcs_prior_log_std", 0.0))
    swerling = str(
        provenance.get("target_swerling_prior_case", world.target.swerling_case)
    )
    if median <= 0.0 or log_std < 0.0 or not math.isfinite(log_std):
        raise ValueError("registered target scattering prior is invalid")
    if order == 1:
        normal_points = ((0.0, 1.0),)
    else:
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        normal_points = tuple(
            (float(math.sqrt(2.0) * node), float(weight / math.sqrt(math.pi)))
            for node, weight in zip(nodes, weights)
        )
    if swerling == "0":
        texture_points = ((1.0, 1.0),)
    else:
        shape = 1.0 if swerling in {"I", "II"} else 2.0
        nodes, weights = roots_genlaguerre(order, shape - 1.0)
        normalization = math.gamma(shape)
        texture_points = tuple(
            (float(node / shape), float(weight / normalization))
            for node, weight in zip(nodes, weights)
        )
    points = []
    for (z_value, class_weight), (texture, texture_weight) in product(
        normal_points,
        texture_points,
    ):
        rcs = median * math.exp(log_std * z_value) * texture
        points.append(
            (
                float(max(rcs, 1.0e-8)),
                float(class_weight * texture_weight),
                float(z_value),
                float(texture),
            )
        )
    total = sum(item[1] for item in points)
    if total <= 0.0:
        raise RuntimeError("scattering quadrature has no probability mass")
    return tuple((rcs, weight / total, z, texture) for rcs, weight, z, texture in points)


def _scattering_texture_group_count(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
) -> int:
    """Count controller-visible carrier groups carrying target scattering."""

    carrier_groups = {
        int(round(float(world.waveform(action.waveform_id).carrier_hz) / 1.0e6))
        for action in actions
        if not action.kind.startswith("calibration")
    }
    return max(1, len(carrier_groups))


def _carrier_diverse_clock_verification_actions(
    world: PhysicalWorld,
    clock_sensitive_actions: Sequence[RadarAction],
    calibration_actions: Sequence[RadarAction],
    layout: StateLayout,
    factor_by_action: Mapping[int, Any],
) -> tuple[RadarAction, ...]:
    """Retain one calibration-connected timing action per carrier group.

    Input order is the proposal ranking.  The first connected action in each
    declared carrier group is retained; if no calibration edge overlaps that
    group, the first group action remains as a diagnostic fallback.
    """

    actions_by_carrier: dict[int, list[RadarAction]] = {}
    for action in clock_sensitive_actions:
        carrier_group = int(
            round(float(world.waveform(action.waveform_id).carrier_hz) / 1.0e6)
        )
        actions_by_carrier.setdefault(carrier_group, []).append(action)
    selected: list[RadarAction] = []
    for carrier_group in sorted(actions_by_carrier):
        group_actions = actions_by_carrier[carrier_group]
        connected = [
            action
            for action in group_actions
            if any(
                set(
                    _clock_support_node_ids(
                        layout,
                        factor_by_action[action.action_id],
                    )
                ).intersection(
                    _clock_support_node_ids(
                        layout,
                        factor_by_action[calibration.action_id],
                    )
                )
                for calibration in calibration_actions
            )
        ]
        selected.append((connected or group_actions)[0])
    return tuple(selected)


def _registered_scattering_support_lower_bound(
    world: PhysicalWorld,
    outage_probability: float,
    *,
    texture_group_count: int = 1,
) -> tuple[float, float, dict[str, Any]]:
    r"""Return a rigorous lower-support point for registered RCS uncertainty.

    Let the target-class mean be lognormal and the scan texture be Gamma with
    unit mean.  Splitting ``alpha`` equally across the independent uncertain
    factors and applying the union bound gives

    ``P(sigma >= sigma_floor) >= 1 - alpha``.

    Since the implemented correct-update probability is monotone in received
    target power, ``(1-alpha) P(E | sigma_floor)`` is a conservative lower
    bound for any verification event ``E``.  This bound is used only by the
    opt-in verification contract, not by the robust information objective.
    """

    alpha = float(outage_probability)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("scattering outage probability must lie in (0,1)")
    group_count = int(texture_group_count)
    if group_count < 1:
        raise ValueError("scattering texture group count must be positive")
    provenance = dict(world.external_channel_provenance)
    median = float(
        provenance.get("target_rcs_prior_median_m2", world.target.mean_rcs_m2)
    )
    log_std = float(provenance.get("target_rcs_prior_log_std", 0.0))
    swerling = str(
        provenance.get("target_swerling_prior_case", world.target.swerling_case)
    )
    if median <= 0.0 or log_std < 0.0 or not math.isfinite(log_std):
        raise ValueError("registered target scattering prior is invalid")
    uncertain_component_count = int(log_std > 0.0) + (
        group_count if swerling != "0" else 0
    )
    if uncertain_component_count == 0:
        return median, 1.0, {
            "status": "deterministic_scattering",
            "outage_probability": 0.0,
            "certified_retained_probability_mass": 1.0,
            "rcs_floor_m2": float(median),
            "union_bound_used": False,
        }
    component_tail = alpha / float(uncertain_component_count)
    class_multiplier = (
        math.exp(log_std * float(ndtri(component_tail)))
        if log_std > 0.0
        else 1.0
    )
    if swerling == "0":
        texture = 1.0
        texture_shape = None
    else:
        texture_shape = 1.0 if swerling in {"I", "II"} else 2.0
        texture = float(
            gammaincinv(texture_shape, component_tail) / texture_shape
        )
    floor = float(max(median * class_multiplier * texture, 1.0e-12))
    return floor, float(1.0 - alpha), {
        "status": "passed",
        "outage_probability": alpha,
        "certified_retained_probability_mass": float(1.0 - alpha),
        "uncertain_component_count": int(uncertain_component_count),
        "component_lower_tail_probability": float(component_tail),
        "target_class_median_rcs_m2": float(median),
        "target_class_log_standard_deviation": float(log_std),
        "target_class_lower_multiplier": float(class_multiplier),
        "swerling_case": swerling,
        "scan_texture_group_count": int(
            group_count if swerling != "0" else 0
        ),
        "scan_texture_group_definition": (
            "declared_carrier_frequency_in_integer_megahertz"
        ),
        "scan_texture_cross_group_correlation_model": (
            "independent_group_proxy_not_full_frequency_correlation_model"
        ),
        "swerling_texture_shape": texture_shape,
        "swerling_texture_lower_quantile": float(texture),
        "rcs_floor_m2": floor,
        "union_bound_used": bool(uncertain_component_count > 1),
        "truth_realization_read": False,
        "bound_semantics": (
            "P(event)>=retained_mass*P(event|registered_rcs_floor)"
        ),
    }


def _common_support_event_probability_lower_bound(
    conditional_event_probability: float,
    retained_probability_mass: float,
) -> float:
    r"""Apply one common latent-support bound to one joint event.

    If ``A`` is the registered scattering-support event and the verification
    event probability is monotone above the support floor, then

    ``P(E) >= P(A) P(E | sigma_floor)``.

    ``A`` is common to every action in one scan.  Its probability must
    therefore be multiplied once after the conditional multi-action event has
    been formed, never once per action.
    """

    conditional = float(conditional_event_probability)
    retained = float(retained_probability_mass)
    if not math.isfinite(conditional) or not 0.0 <= conditional <= 1.0:
        raise ValueError("conditional event probability must lie in [0,1]")
    if not math.isfinite(retained) or not 0.0 <= retained <= 1.0:
        raise ValueError("retained probability mass must lie in [0,1]")
    return float(np.clip(retained * conditional, 0.0, 1.0))


def _joint_planning_scenarios(
    world: PhysicalWorld,
    config: RuntimeConfig,
) -> tuple[_JointPlanningScenario, ...]:
    jammers = _robust_scenarios(world, config.maximum_robust_scenarios)
    scattering = (
        _registered_scattering_quadrature(world, config.scattering_quadrature_order)
        if config.scattering_prior_marginalization
        else ((float(world.target.mean_rcs_m2), 1.0, 0.0, 1.0),)
    )
    jammer_weight = 1.0 / len(jammers)
    scenarios: list[_JointPlanningScenario] = []
    for jammer, (rcs, scatter_weight, z_value, texture) in product(
        jammers,
        scattering,
    ):
        scenario_world = replace(
            world,
            target=replace(world.target, mean_rcs_m2=rcs, swerling_case="0"),
            external_channel_provenance={
                **dict(world.external_channel_provenance),
                "planning_scattering_marginalization": bool(
                    config.scattering_prior_marginalization
                ),
                "planning_scattering_quadrature_order": int(
                    config.scattering_quadrature_order
                ),
                "planning_scattering_source": (
                    "registered_target_class_prior_not_realized_truth"
                ),
            },
        )
        scenario_world.validate()
        scenarios.append(
            _JointPlanningScenario(
                world=scenario_world,
                jammer=jammer,
                probability_weight=float(jammer_weight * scatter_weight),
                scenario_id=(
                    f"{_scenario_identifier(jammer)}|rcs={rcs:.9g}m2|"
                    f"z={z_value:.6g}|texture={texture:.6g}"
                ),
                rcs_m2=float(rcs),
                class_log_rcs_z=float(z_value),
                swerling_texture=float(texture),
            )
        )
    total = sum(item.probability_weight for item in scenarios)
    return tuple(replace(item, probability_weight=item.probability_weight / total) for item in scenarios)


def _schedule_update_probability(
    action_probabilities: Mapping[int, float],
    schedule: Sequence[int],
) -> float:
    miss = 1.0
    for action_id in schedule:
        miss *= 1.0 - float(np.clip(action_probabilities.get(int(action_id), 0.0), 0.0, 1.0))
    return float(np.clip(1.0 - miss, 0.0, 1.0))


def _structural_position_observation_geometry(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    *,
    factor_by_action: Mapping[int, Any] | None = None,
) -> tuple[int, np.ndarray, int, float]:
    """Return sensing count, singular values, structural rank, and tolerance.

    Structural observability and detection probability are separate contracts.
    The exact controller marginalizes detection over scattering and interference
    scenarios, so one nominal low-detection realization must not delete a valid
    measurement Jacobian. Detectability is enforced independently by the
    marginalized correct-update probability gate.
    """

    position_rows: list[np.ndarray] = []
    sensing_action_count = 0
    for action in actions:
        if action.kind.startswith("calibration"):
            continue
        factor = (
            factor_by_action[action.action_id]
            if factor_by_action is not None
            else build_action_factor(world, action, None)
        )
        sensing_action_count += 1
        primary_rows = len(factor.measurement_names)
        for gradient in np.asarray(
            factor.jacobian_physical[:primary_rows, :3], dtype=float
        ):
            norm = float(np.linalg.norm(gradient))
            if norm > 1.0e-15:
                position_rows.append(gradient / norm)
    if not position_rows:
        return sensing_action_count, np.zeros(0, dtype=float), 0, 1.0e-9
    singular_values = np.linalg.svd(np.vstack(position_rows), compute_uv=False)
    tolerance = max(float(singular_values[0]) * 1.0e-3, 1.0e-9)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return sensing_action_count, singular_values, rank, tolerance


def _position_separating_observation_geometry(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    *,
    factor_by_action: Mapping[int, Any] | None = None,
) -> tuple[int, int, np.ndarray, int, float]:
    r"""Return geometry that identifies position independently of velocity.

    The returned counts are the number of sensing actions and the number that
    contribute at least one velocity-independent position row.  This is a
    necessary local identifiability certificate for a six-dimensional
    position/velocity target state; it is not an oracle guarantee of global
    unimodality or successful association.
    """

    layout = StateLayout.from_world(world)
    position_rows: list[np.ndarray] = []
    sensing_action_count = 0
    contributing_action_count = 0
    for action in actions:
        if action.kind.startswith("calibration"):
            continue
        sensing_action_count += 1
        factor = (
            factor_by_action.get(int(action.action_id))
            if factor_by_action is not None
            else None
        )
        if factor is None:
            factor = build_action_factor(world, action, None)
        rows = _velocity_independent_verification_rows(
            layout,
            factor,
            include_clock_only_rows=False,
        )
        action_contributed = False
        for gradient in rows[:, :3]:
            norm = float(np.linalg.norm(gradient))
            if norm > 1.0e-15:
                position_rows.append(gradient / norm)
                action_contributed = True
        contributing_action_count += int(action_contributed)
    if not position_rows:
        return (
            sensing_action_count,
            contributing_action_count,
            np.zeros(0, dtype=float),
            0,
            1.0e-9,
        )
    singular_values = np.linalg.svd(np.vstack(position_rows), compute_uv=False)
    tolerance = max(float(singular_values[0]) * 1.0e-3, 1.0e-9)
    rank = int(np.count_nonzero(singular_values > tolerance))
    return (
        sensing_action_count,
        contributing_action_count,
        singular_values,
        rank,
        tolerance,
    )


def _erasure_robust_position_frame_certificate(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    *,
    maximum_erasures: int = 1,
    factor_by_action: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    r"""Certify a directional position frame after bounded action erasures.

    Let ``U_E`` contain the unit-normalized, velocity-independent position
    Jacobian rows that remain after erasing the sensing-action set ``E``.  For
    every perturbation ``delta_x``, ``||U_E delta_x|| >= sigma_min(U_E)
    ||delta_x||``.  Therefore the worst local inversion amplification is bounded
    by ``1 / sigma_min(U_E)`` and the normal-matrix condition number is the
    squared Jacobian condition number.  This exact finite linear-algebra result
    is only a local directional certificate; it does not assert global
    unimodality, correct association, or realized packet delivery.
    """

    if not isinstance(maximum_erasures, int) or maximum_erasures < 1:
        raise ValueError("maximum erasures must be a positive integer")
    layout = StateLayout.from_world(world)
    sensing_action_ids: list[int] = []
    rows_by_action: dict[int, np.ndarray] = {}
    for action in actions:
        if action.kind.startswith("calibration"):
            continue
        action_id = int(action.action_id)
        sensing_action_ids.append(action_id)
        factor = (
            factor_by_action.get(action_id)
            if factor_by_action is not None
            else None
        )
        if factor is None:
            factor = build_action_factor(world, action, None)
        action_rows: list[np.ndarray] = []
        for gradient in _velocity_independent_verification_rows(
            layout,
            factor,
            include_clock_only_rows=False,
        )[:, :3]:
            norm = float(np.linalg.norm(gradient))
            if norm > 1.0e-15:
                action_rows.append(np.asarray(gradient, dtype=float) / norm)
        rows_by_action[action_id] = (
            np.vstack(action_rows)
            if action_rows
            else np.zeros((0, 3), dtype=float)
        )

    evaluated: list[dict[str, Any]] = []
    maximum_order = min(maximum_erasures, len(sensing_action_ids))
    for order in range(1, maximum_order + 1):
        for erased in combinations(sensing_action_ids, order):
            erased_set = set(erased)
            remaining_blocks = [
                rows_by_action[action_id]
                for action_id in sensing_action_ids
                if action_id not in erased_set and rows_by_action[action_id].size
            ]
            matrix = (
                np.vstack(remaining_blocks)
                if remaining_blocks
                else np.zeros((0, 3), dtype=float)
            )
            if matrix.shape[0] == 0:
                singular_values = np.zeros(0, dtype=float)
                tolerance = 1.0e-15
                rank = 0
                ratio = 0.0
            else:
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                leading = float(singular_values[0])
                tolerance = max(
                    np.finfo(float).eps
                    * max(matrix.shape)
                    * max(leading, 1.0)
                    * 16.0,
                    1.0e-15,
                )
                rank = int(np.count_nonzero(singular_values > tolerance))
                ratio = (
                    float(singular_values[-1] / leading)
                    if rank >= 3 and singular_values.size >= 3 and leading > 0.0
                    else 0.0
                )
            evaluated.append(
                {
                    "erased_action_ids": [int(value) for value in erased],
                    "remaining_position_row_count": int(matrix.shape[0]),
                    "rank": int(rank),
                    "rank_tolerance": float(tolerance),
                    "singular_values": [
                        float(value) for value in singular_values
                    ],
                    "singular_value_ratio": float(ratio),
                    "jacobian_condition_number": (
                        float(1.0 / ratio) if ratio > 0.0 else None
                    ),
                    "normal_matrix_condition_number": (
                        float(1.0 / ratio**2) if ratio > 0.0 else None
                    ),
                    "frame_lower_bound": (
                        float(singular_values[-1] ** 2)
                        if rank >= 3 and singular_values.size >= 3
                        else 0.0
                    ),
                    "frame_upper_bound": (
                        float(singular_values[0] ** 2)
                        if singular_values.size
                        else 0.0
                    ),
                }
            )

    worst = (
        min(
            evaluated,
            key=lambda item: (
                float(item["singular_value_ratio"]),
                tuple(item["erased_action_ids"]),
            ),
        )
        if evaluated
        else None
    )
    minimum_frame = (
        min(
            evaluated,
            key=lambda item: (
                float(item["frame_lower_bound"]),
                tuple(item["erased_action_ids"]),
            ),
        )
        if evaluated
        else None
    )
    minimum_ratio = float(worst["singular_value_ratio"]) if worst else 0.0
    full_rank = bool(evaluated) and all(int(item["rank"]) >= 3 for item in evaluated)
    return {
        "status": "passed" if full_rank else "rank_deficient_after_erasure",
        "maximum_erasures": int(maximum_erasures),
        "sensing_action_ids": list(sensing_action_ids),
        "sensing_action_count": len(sensing_action_ids),
        "contributing_action_ids": [
            action_id
            for action_id in sensing_action_ids
            if rows_by_action[action_id].size
        ],
        "evaluated_erasure_pattern_count": len(evaluated),
        "evaluated_erasure_patterns": evaluated,
        "full_rank_after_all_evaluated_erasures": bool(full_rank),
        "worst_erased_action_ids": (
            list(worst["erased_action_ids"]) if worst else []
        ),
        "worst_reduced_rank": int(worst["rank"]) if worst else 0,
        "worst_singular_values": (
            list(worst["singular_values"]) if worst else []
        ),
        "minimum_singular_value_ratio": float(minimum_ratio),
        "worst_jacobian_condition_number": (
            float(1.0 / minimum_ratio) if minimum_ratio > 0.0 else None
        ),
        "worst_normal_matrix_condition_number": (
            float(1.0 / minimum_ratio**2) if minimum_ratio > 0.0 else None
        ),
        "minimum_frame_lower_bound": (
            float(minimum_frame["frame_lower_bound"])
            if minimum_frame
            else 0.0
        ),
        "minimum_frame_erased_action_ids": (
            list(minimum_frame["erased_action_ids"])
            if minimum_frame
            else []
        ),
        "rows_are_velocity_independent": True,
        "rows_are_unit_normalized": True,
        "certificate_scope": (
            "local_directional_stability_after_bounded_sensing_action_erasure"
        ),
        "global_nonlinear_identifiability_claimed": False,
        "realized_delivery_or_detection_read": False,
        "truth_or_future_event_read": False,
    }


def _full_rank_position_update_probability(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    correct_probability_by_action: Mapping[int, float],
    *,
    factor_by_action: Mapping[int, Any] | None = None,
) -> float:
    """Probability that the correctly received action subset is 3-D observable.

    The Bernoulli independence is the same finite-action approximation already
    used by ``_schedule_update_probability``. Unlike the any-update probability,
    this event requires at least two sensing actions and structural position
    rank three. It is therefore the relevant continuity event during acquisition
    and reacquisition.
    """

    sensing = tuple(
        action for action in actions if not action.kind.startswith("calibration")
    )
    if len(sensing) < 2:
        return 0.0
    probability = 0.0
    for delivered_count in range(len(sensing) + 1):
        for delivered in combinations(sensing, delivered_count):
            delivered_ids = {action.action_id for action in delivered}
            outcome_probability = 1.0
            for action in sensing:
                p = float(
                    np.clip(
                        correct_probability_by_action.get(action.action_id, 0.0),
                        0.0,
                        1.0,
                    )
                )
                outcome_probability *= p if action.action_id in delivered_ids else 1.0 - p
            if outcome_probability <= 0.0 or len(delivered) < 2:
                continue
            _, _, rank, _ = _structural_position_observation_geometry(
                world,
                delivered,
                factor_by_action=factor_by_action,
            )
            if rank >= 3:
                probability += outcome_probability
    return float(np.clip(probability, 0.0, 1.0))


def _position_separating_update_probability(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    usable_probability_by_action: Mapping[int, float],
    *,
    factor_by_action: Mapping[int, Any] | None = None,
    minimum_sensing_actions: int = 2,
) -> float:
    """Probability of an arrived rank-three velocity-independent update."""

    sensing = tuple(
        action for action in actions if not action.kind.startswith("calibration")
    )
    if len(sensing) < minimum_sensing_actions:
        return 0.0
    probability = 0.0
    for delivered_count in range(len(sensing) + 1):
        for delivered in combinations(sensing, delivered_count):
            delivered_ids = {int(action.action_id) for action in delivered}
            outcome_probability = 1.0
            for action in sensing:
                p = float(
                    np.clip(
                        usable_probability_by_action.get(action.action_id, 0.0),
                        0.0,
                        1.0,
                    )
                )
                outcome_probability *= (
                    p if action.action_id in delivered_ids else 1.0 - p
                )
            if (
                outcome_probability <= 0.0
                or len(delivered) < minimum_sensing_actions
            ):
                continue
            _, _, _, rank, _ = _position_separating_observation_geometry(
                world,
                delivered,
                factor_by_action=factor_by_action,
            )
            if rank >= 3:
                probability += outcome_probability
    return float(np.clip(probability, 0.0, 1.0))


def _differential_clock_verification_probability(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    usable_probability_by_action: Mapping[int, float],
    *,
    factor_by_action: Mapping[int, Any] | None = None,
) -> float:
    """Probability that arrived factors satisfy the joint clock certificate."""

    action_tuple = tuple(actions)
    probability = 0.0
    for delivered_count in range(len(action_tuple) + 1):
        for delivered in combinations(action_tuple, delivered_count):
            delivered_ids = {int(action.action_id) for action in delivered}
            outcome_probability = 1.0
            for action in action_tuple:
                p = float(
                    np.clip(
                        usable_probability_by_action.get(action.action_id, 0.0),
                        0.0,
                        1.0,
                    )
                )
                outcome_probability *= (
                    p if action.action_id in delivered_ids else 1.0 - p
                )
            if outcome_probability <= 0.0:
                continue
            delivered_factors = None
            if factor_by_action is not None:
                delivered_factors = {
                    int(action.action_id): factor_by_action[int(action.action_id)]
                    for action in delivered
                    if int(action.action_id) in factor_by_action
                }
            certificate = _differential_clock_verification_certificate(
                world,
                delivered,
                factor_by_action=delivered_factors,
                require_position_separating_rank=True,
            )
            if certificate["differential_clock_identification_complete"]:
                probability += outcome_probability
    return float(np.clip(probability, 0.0, 1.0))


class GraphCandidateGenerator:
    """Bound the action domain without creating a second belief state."""

    def __init__(
        self,
        config: RuntimeConfig,
        mission: MissionIR,
        planning_probability: PlanningDetectionProbability,
    ) -> None:
        self.config = config
        self.mission = mission
        self.planning_probability = planning_probability
        self.last_generation_audit: dict[str, Any] = {}
        self.last_schedule_update_probability: dict[tuple[int, ...], float] = {}
        self.last_clock_sentinel_requirement: dict[str, Any] = {}
        self._clock_sentinel_design_by_world_id: dict[str, Any] = {}
        self.learned_backend = None
        if config.proposal_backend is ProposalBackend.V71_FACTOR_GNN:
            from wideband_info_geometry.learning.v71_graph_policy import (
                FactorGraphPolicyBackend,
            )

            self.learned_backend = FactorGraphPolicyBackend(
                str(config.proposal_checkpoint),
                device=config.proposal_device,
            )

    def build(
        self,
        world: PhysicalWorld,
        state: EpisodeState,
        availability: DistributedAvailabilityManager,
        *,
        candidate_hints: Sequence[Mapping[str, Any]] = (),
        innovation_health_active: bool = False,
        innovation_verification_active: bool = False,
        innovation_verification_hypotheses: Sequence[str] = (),
        previous_clock_invariant_action_ids: Sequence[int] = (),
    ) -> tuple[CandidateLibrary, dict[tuple[int, ...], float]]:
        """Build the bounded candidate library.

        ``candidate_hints`` are proposal-only schedules emitted by inspectable
        node Agents.  They may expand or reprioritize the bounded library, but
        they never bypass static, dynamic, transport, exact-graph, or campaign
        safety checks.  With an empty hint sequence this method is byte-for-byte
        equivalent in decision semantics to the pre-Part-1 analytic generator.
        """
        by_id = {action.action_id: action for action in world.actions}
        clock_hypothesis_active = bool(
            innovation_verification_active
            and DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS
            in {str(value) for value in innovation_verification_hypotheses}
        )
        available = [
            by_id[action_id]
            for action_id in availability.latest_record.available_action_ids
            if action_id in by_id
        ]
        available = [
            action
            for action in available
            if action.allowed
            and action.kind not in self.mission.forbidden_action_kinds
            and action.action_id not in self.mission.forbidden_action_ids
            and schedule_feasibility(world, (action,))[0]
        ]
        cheap_scores: dict[int, float] = {}
        correct_update_probability_by_action: dict[int, float] = {}
        detection_delivery_probability_by_action: dict[int, float] = {}
        delivery_probability_by_action: dict[int, float] = {}
        verification_conditional_probability_by_action: dict[int, float] = {}
        planning_probability_by_action: dict[str, Any] = {}
        factor_by_action: dict[int, Any] = {}
        proposal_factor_world = world
        verification_lower_world = None
        verification_retained_mass = 1.0
        verification_texture_group_count = 1
        verification_scattering_support_audit = None
        if self.config.innovation_verification_require_position_separating_rank:
            prior_median_rcs_m2 = float(
                world.external_channel_provenance.get(
                    "target_rcs_prior_median_m2",
                    world.target.mean_rcs_m2,
                )
            )
            proposal_factor_world = replace(
                world,
                target=replace(
                    world.target,
                    mean_rcs_m2=prior_median_rcs_m2,
                    swerling_case="0",
                ),
            )
            (
                verification_rcs_floor_m2,
                verification_retained_mass,
                verification_scattering_support_audit,
            ) = _registered_scattering_support_lower_bound(
                world,
                self.config.innovation_verification_scattering_outage_probability,
                texture_group_count=min(
                    _scattering_texture_group_count(world, available),
                    int(world.max_schedule_size),
                ),
            )
            verification_texture_group_count = int(
                verification_scattering_support_audit[
                    "scan_texture_group_count"
                ]
            )
            verification_lower_world = replace(
                world,
                target=replace(
                    world.target,
                    mean_rcs_m2=float(verification_rcs_floor_m2),
                    swerling_case="0",
                ),
            )
            proposal_factor_world.validate()
            verification_lower_world.validate()
        for action in available:
            factor = build_action_factor(proposal_factor_world, action, None)
            factor_by_action[action.action_id] = factor
            usage = schedule_resource_usage(world, (action,))
            normalized_cost = (
                usage.energy_j / max(world.max_total_energy_j, 1e-12)
                + usage.emission_exposure_j / max(world.max_total_emission_j, 1e-12)
                + usage.communication_mb / max(world.max_total_communication_mb, 1e-12)
            )
            planning_information = (
                self.planning_probability.expected_observed_information(
                    proposal_factor_world, action, factor
                )
            )
            probability_record = self.planning_probability.evaluate(action, factor)
            planning_probability_by_action[str(action.action_id)] = (
                probability_record.to_mapping()
            )
            cheap_scores[action.action_id] = float(
                np.trace(planning_information) - 0.02 * normalized_cost
            )
            delivery = self._delivery_probability(availability, action.action_id)
            delivery_probability_by_action[action.action_id] = float(
                np.clip(delivery, 0.0, 1.0)
            )
            if action.kind.startswith("calibration"):
                detection_delivery_probability_by_action[action.action_id] = 0.0
                correct_update_probability_by_action[action.action_id] = 0.0
                verification_conditional_probability_by_action[
                    action.action_id
                ] = float(
                    delivery_probability_by_action[action.action_id]
                )
            else:
                detection_delivery_probability_by_action[action.action_id] = float(
                    np.clip(
                        delivery * probability_record.planning_success_probability,
                        0.0,
                        1.0,
                    )
                )
                correct_update_probability_by_action[action.action_id] = float(
                    np.clip(
                        delivery
                        * probability_record.planning_success_probability
                        * factor.link_budget.correct_association_probability,
                        0.0,
                        1.0,
                    )
                )
                if verification_lower_world is not None:
                    lower_factor = build_action_factor(
                        verification_lower_world,
                        action,
                        None,
                    )
                    lower_record = self.planning_probability.evaluate(
                        action,
                        lower_factor,
                    )
                    verification_conditional_probability_by_action[
                        action.action_id
                    ] = float(
                        np.clip(
                            delivery
                            * lower_record.planning_success_probability
                            * lower_factor.link_budget.correct_association_probability,
                            0.0,
                            1.0,
                        )
                    )
                else:
                    verification_conditional_probability_by_action[
                        action.action_id
                    ] = correct_update_probability_by_action[action.action_id]
        if self.learned_backend is not None and available:
            learned_scores = self.learned_backend.score_actions(
                world,
                episode_id=str(state.episode_id),
                epoch=int(state.epoch),
            )
            missing = sorted(
                action.action_id
                for action in available
                if action.action_id not in learned_scores
            )
            if missing:
                raise RuntimeError(
                    f"V71 factor-GNN omitted available actions {missing}"
                )
            cheap_scores = {
                action.action_id: float(learned_scores[action.action_id])
                for action in available
            }
        ranked = sorted(
            available,
            key=lambda action: (-cheap_scores[action.action_id], action.action_id),
        )
        selected_actions: list[RadarAction] = []
        for role in (
            "active_transmitter",
            "passive_receiver",
            "illuminator_reference_processor",
            "calibration_agent",
        ):
            match = next((action for action in ranked if _action_role(action) == role), None)
            if match is not None and match not in selected_actions:
                selected_actions.append(match)
        verification_anchor_actions = sorted(
            (
                action
                for action in ranked
                if action.kind == "active_monostatic"
                and "monostatic_delay_s"
                in tuple(
                    str(name)
                    for name in factor_by_action[
                        action.action_id
                    ].measurement_names
                )
            ),
            key=lambda action: (
                -verification_conditional_probability_by_action.get(
                    action.action_id,
                    0.0,
                ),
                -cheap_scores[action.action_id],
                action.action_id,
            ),
        )
        verification_angle_actions = sorted(
            (action for action in ranked if action.kind == "aoa"),
            key=lambda action: (
                -verification_conditional_probability_by_action.get(
                    action.action_id,
                    0.0,
                ),
                -cheap_scores[action.action_id],
                action.action_id,
            ),
        )
        clock_sensitive_actions = [
            action
            for action in ranked
            if action.kind in {"active_bistatic", "tdoa"}
            and any(
                str(name) in {"bistatic_delay_s", "tdoa_s"}
                for name in factor_by_action[action.action_id].measurement_names
            )
        ]
        time_calibration_actions = [
            action for action in ranked if action.kind == "calibration_time"
        ]
        layout = StateLayout.from_world(world)
        clock_sensitive_node_ids = {
            node_id
            for action in clock_sensitive_actions
            for node_id in _clock_support_node_ids(
                layout, factor_by_action[action.action_id]
            )
        }
        connected_time_calibration_actions = [
            action
            for action in time_calibration_actions
            if clock_sensitive_node_ids.intersection(
                _clock_support_node_ids(
                    layout, factor_by_action[action.action_id]
                )
            )
        ]
        carrier_diverse_clock_actions: list[RadarAction] = []
        if self.config.innovation_verification_require_position_separating_rank:
            carrier_diverse_clock_actions = list(
                _carrier_diverse_clock_verification_actions(
                    world,
                    clock_sensitive_actions,
                    connected_time_calibration_actions,
                    layout,
                    factor_by_action,
                )
            )
        available_action_ids = {int(action.action_id) for action in available}
        sentinel_design = None
        sentinel_trigger = _clock_sentinel_trigger(
            enabled=bool(self.config.clock_sentinel_enabled),
            epoch=int(state.epoch),
            start_epoch=int(self.config.clock_sentinel_start_epoch),
            interval_epochs=int(self.config.clock_sentinel_interval_epochs),
            innovation_health_active=bool(innovation_health_active),
            previous_clock_invariant_action_ids=(
                previous_clock_invariant_action_ids
            ),
            available_action_ids=available_action_ids,
        )
        sentinel_scheduled = bool(sentinel_trigger["scheduled"])
        sentinel_design_error = None
        if self.config.clock_sentinel_enabled:
            sentinel_design = self._clock_sentinel_design_by_world_id.get(
                str(world.world_id)
            )
            if sentinel_design is None:
                information_weight_by_action_id: dict[int, float] = {}
                for action in clock_sensitive_actions + time_calibration_actions:
                    factor = factor_by_action[action.action_id]
                    timing_indices = [
                        index
                        for index, name in enumerate(factor.measurement_names)
                        if str(name) in {"bistatic_delay_s", "tdoa_s"}
                        or (
                            str(name).endswith("_s")
                            and (
                                "calibration" in str(name)
                                or "clock" in str(name)
                                or "effective_delay" in str(name)
                            )
                        )
                    ]
                    if not timing_indices:
                        continue
                    timing_variance = min(
                        float(factor.covariance[index, index])
                        for index in timing_indices
                    )
                    usable_probability = float(
                        delivery_probability_by_action.get(action.action_id, 0.0)
                        if action.kind == "calibration_time"
                        else (
                            verification_conditional_probability_by_action.get(
                                action.action_id, 0.0
                            )
                            if self.config.innovation_verification_require_position_separating_rank
                            else correct_update_probability_by_action.get(
                                action.action_id, 0.0
                            )
                        )
                    )
                    if (
                        math.isfinite(timing_variance)
                        and timing_variance > 0.0
                        and usable_probability > 0.0
                    ):
                        information_weight_by_action_id[action.action_id] = (
                            usable_probability / timing_variance
                        )
                try:
                    sentinel_design = design_clock_sentinel_sequence(
                        clock_sensitive_actions + time_calibration_actions,
                        maximum_unique_edges=(
                            self.config.clock_sentinel_maximum_unique_edges
                        ),
                        planning_quality_by_action_id=cheap_scores,
                        information_weight_by_action_id=(
                            information_weight_by_action_id
                        ),
                    )
                    self._clock_sentinel_design_by_world_id[
                        str(world.world_id)
                    ] = sentinel_design
                except ValueError as error:
                    sentinel_design_error = str(error)
        sentinel_probe_action = None
        sentinel_anchor_actions: list[RadarAction] = []
        sentinel_anchor_selection_rows: list[dict[str, Any]] = []
        sentinel_anchor_target_count = int(
            0
            if (
                not sentinel_scheduled
                or not self.config.clock_sentinel_require_monostatic_anchor
            )
            else 2
            if (
                not self.config.innovation_verification_require_position_separating_rank
                and (
                innovation_health_active
                or sentinel_trigger["causal_anchor_continuity_loss"]
                )
            )
            else 1
        )
        if sentinel_scheduled and sentinel_design is not None:
            sequence = tuple(int(value) for value in sentinel_design.action_ids)
            sequence_offset = (
                int(state.epoch) - self.config.clock_sentinel_start_epoch
            ) // self.config.clock_sentinel_interval_epochs
            for offset in range(len(sequence)):
                candidate_id = sequence[(sequence_offset + offset) % len(sequence)]
                if candidate_id in available_action_ids:
                    sentinel_probe_action = by_id[candidate_id]
                    break
            if (
                sentinel_probe_action is not None
                and sentinel_anchor_target_count > 0
            ):
                used_anchor_nodes: set[str] = set()
                for anchor in verification_anchor_actions:
                    anchor_nodes = set(
                        _monostatic_anchor_identity_nodes(anchor)
                    )
                    if not anchor_nodes:
                        sentinel_anchor_selection_rows.append(
                            {
                                "action_id": int(anchor.action_id),
                                "verification_scattering_lower_probability": float(
                                    verification_conditional_probability_by_action.get(
                                        anchor.action_id, 0.0
                                    )
                                ),
                                "physical_anchor_node_ids": [],
                                "status": "rejected_not_monostatic_identity",
                            }
                        )
                        continue
                    overlap = used_anchor_nodes.intersection(anchor_nodes)
                    if overlap:
                        sentinel_anchor_selection_rows.append(
                            {
                                "action_id": int(anchor.action_id),
                                "verification_scattering_lower_probability": float(
                                    verification_conditional_probability_by_action.get(
                                        anchor.action_id, 0.0
                                    )
                                ),
                                "physical_anchor_node_ids": sorted(anchor_nodes),
                                "status": "rejected_duplicate_physical_anchor_node",
                                "overlapping_physical_anchor_node_ids": sorted(
                                    overlap
                                ),
                            }
                        )
                        continue
                    trial = tuple(
                        sorted(
                            (
                                sentinel_probe_action,
                                *sentinel_anchor_actions,
                                anchor,
                            ),
                            key=lambda item: int(item.action_id),
                        )
                    )
                    feasible, static_reasons = schedule_feasibility(world, trial)
                    dynamic_feasible, dynamic_reasons = action_is_dynamically_feasible(
                        self._base_world_for_dynamic_check(world),
                        state,
                        tuple(int(item.action_id) for item in trial),
                    )
                    (
                        operational_feasible,
                        operational_reasons,
                        _,
                    ) = (
                        availability.schedule_operational_feasibility(trial)
                    )
                    sentinel_anchor_selection_rows.append(
                        {
                            "action_id": int(anchor.action_id),
                            "verification_scattering_lower_probability": float(
                                verification_conditional_probability_by_action.get(
                                    anchor.action_id, 0.0
                                )
                            ),
                            "physical_anchor_node_ids": sorted(anchor_nodes),
                            "trial_action_ids": [
                                int(item.action_id) for item in trial
                            ],
                            "static_feasible": bool(feasible),
                            "static_reasons": list(static_reasons),
                            "dynamic_feasible": bool(dynamic_feasible),
                            "dynamic_reasons": list(dynamic_reasons),
                            "operational_feasible": bool(
                                operational_feasible
                            ),
                            "operational_reasons": list(
                                operational_reasons
                            ),
                            "status": (
                                "selected"
                                if feasible
                                and dynamic_feasible
                                and operational_feasible
                                else "rejected_feasibility"
                            ),
                        }
                    )
                    if feasible and dynamic_feasible and operational_feasible:
                        sentinel_anchor_actions.append(anchor)
                        used_anchor_nodes.update(anchor_nodes)
                        if (
                            len(sentinel_anchor_actions)
                            >= sentinel_anchor_target_count
                        ):
                            break
        sentinel_required_action_ids: tuple[int, ...] = ()
        if sentinel_probe_action is not None:
            if self.config.clock_sentinel_require_monostatic_anchor:
                if sentinel_anchor_actions:
                    sentinel_required_action_ids = tuple(
                        sorted(
                            {
                                int(sentinel_probe_action.action_id),
                                *(
                                    int(action.action_id)
                                    for action in sentinel_anchor_actions
                                ),
                            }
                        )
                    )
            else:
                sentinel_required_action_ids = (
                    int(sentinel_probe_action.action_id),
                )
        self.last_clock_sentinel_requirement = {
            "enabled": bool(self.config.clock_sentinel_enabled),
            "scheduled_this_epoch": bool(sentinel_scheduled),
            "scheduled_by_periodic_interval": bool(
                sentinel_trigger["periodic_due"]
            ),
            "scheduled_by_causal_anchor_continuity_loss": bool(
                sentinel_trigger["causal_anchor_continuity_loss"]
            ),
            "coordinated_with_active_innovation_alarm": bool(
                sentinel_trigger["coordinated_with_active_innovation_alarm"]
            ),
            "suspended_by_active_innovation_alarm": False,
            "previous_clock_invariant_action_ids": list(
                sentinel_trigger["previous_clock_invariant_action_ids"]
            ),
            "previous_clock_invariant_action_ids_still_available": list(
                sentinel_trigger[
                    "previous_clock_invariant_action_ids_still_available"
                ]
            ),
            "required_action_ids": list(sentinel_required_action_ids),
            "scheduled_clock_edge_action_id": (
                int(sentinel_probe_action.action_id)
                if sentinel_probe_action is not None
                else None
            ),
            "scheduled_clock_probe_action_id": (
                int(sentinel_probe_action.action_id)
                if sentinel_probe_action is not None
                else None
            ),
            "scheduled_clock_probe_action_kind": (
                str(sentinel_probe_action.kind)
                if sentinel_probe_action is not None
                else None
            ),
            "scheduled_clock_probe_target_independent": bool(
                sentinel_probe_action is not None
                and sentinel_probe_action.kind == "calibration_time"
            ),
            "scheduled_monostatic_anchor_action_id": (
                int(sentinel_anchor_actions[0].action_id)
                if sentinel_anchor_actions
                else None
            ),
            "scheduled_monostatic_anchor_action_ids": [
                int(action.action_id) for action in sentinel_anchor_actions
            ],
            "requested_distinct_monostatic_anchor_count": int(
                sentinel_anchor_target_count
            ),
            "selected_distinct_monostatic_anchor_count": int(
                len(sentinel_anchor_actions)
            ),
            "monostatic_anchor_identity_definition": (
                "physical_active_monostatic_tx_rx_node_excluding_fusion_and_"
                "transport_relays"
            ),
            "monostatic_anchor_selection_rows": list(
                sentinel_anchor_selection_rows
            ),
            "monostatic_anchor_ranking_criterion": (
                "registered_scattering_support_lower_probability_then_"
                "planning_information_then_action_id"
                if self.config.innovation_verification_require_position_separating_rank
                else "legacy_planning_information_then_action_id"
            ),
            "verification_scattering_support_lower_bound": (
                verification_scattering_support_audit
            ),
            "proposal_factor_scattering_realization": (
                "deterministic_registered_prior_median"
                if self.config.innovation_verification_require_position_separating_rank
                else "legacy_controller_action_indexed_sample"
            ),
            "monostatic_anchor_required": bool(
                self.config.clock_sentinel_require_monostatic_anchor
            ),
            "design": (
                sentinel_design.to_mapping()
                if sentinel_design is not None
                else None
            ),
            "design_error": sentinel_design_error,
            "design_frozen_on_first_controller_visible_world_snapshot": True,
            "event_schedule_or_realized_clock_state_read": False,
            "receiver_residual_read": False,
            "truth_or_future_event_read": False,
        }
        if innovation_verification_active and verification_anchor_actions:
            anchor_action = verification_anchor_actions[0]
            if anchor_action not in selected_actions:
                selected_actions.append(anchor_action)
        if clock_hypothesis_active:
            preferred_time_calibration = (
                connected_time_calibration_actions[:1]
                or time_calibration_actions[:1]
            )
            for diagnostic_action in (
                clock_sensitive_actions[:2]
                + carrier_diverse_clock_actions
                + preferred_time_calibration
            ):
                if diagnostic_action not in selected_actions:
                    selected_actions.append(diagnostic_action)
        for sentinel_action in (
            sentinel_probe_action,
            *sentinel_anchor_actions,
        ):
            if (
                sentinel_action is not None
                and sentinel_action not in selected_actions
            ):
                selected_actions.append(sentinel_action)
        if (
            innovation_verification_active
            and self.config.innovation_verification_require_position_separating_rank
        ):
            # One absolute-range row plus an azimuth/elevation pair can provide
            # three velocity-independent position directions.  Keep two
            # controller-visible angle alternatives in the bounded action pool
            # so a clock probe and calibration action do not crowd this
            # complementary geometry out of a four-action schedule.
            for angle_action in verification_angle_actions[:2]:
                if angle_action not in selected_actions:
                    selected_actions.append(angle_action)
        for action in ranked[: max(6, self.config.maximum_candidates)]:
            if action not in selected_actions:
                selected_actions.append(action)
        candidates: list[tuple[int, ...]] = []
        hint_scores: dict[tuple[int, ...], float] = {}
        if self.config.include_noop_candidate:
            candidates.append(())
        candidates.extend((action.action_id,) for action in selected_actions)

        # Agent schedules are merely bounded-domain hints.  Revalidate every
        # hinted action against the exact same controller-visible world and
        # operational snapshot used by the analytic generator.  Invalid, stale,
        # or unreachable hints are silently excluded from the executable domain
        # and remain visible in the Part-1 trace as rejected proposals.
        available_ids = set(available_action_ids)
        for raw_hint in candidate_hints:
            if not isinstance(raw_hint, Mapping):
                continue
            raw_ids = raw_hint.get("action_ids", ())
            if not isinstance(raw_ids, (list, tuple, set)):
                continue
            try:
                schedule = tuple(sorted({int(value) for value in raw_ids}))
                hint_score = float(raw_hint.get("hint_score", 0.0))
            except (TypeError, ValueError):
                continue
            if not schedule or not math.isfinite(hint_score):
                continue
            if len(schedule) > world.max_schedule_size:
                continue
            if any(action_id not in by_id or action_id not in available_ids for action_id in schedule):
                continue
            actions = tuple(by_id[action_id] for action_id in schedule)
            feasible, _ = schedule_feasibility(world, actions)
            dynamic_feasible, _ = action_is_dynamically_feasible(
                self._base_world_for_dynamic_check(world),
                state,
                schedule,
            )
            operational_feasible, _, _ = availability.schedule_operational_feasibility(
                actions
            )
            if not (feasible and dynamic_feasible and operational_feasible):
                continue
            candidates.append(schedule)
            hint_scores[schedule] = max(hint_scores.get(schedule, -math.inf), hint_score)
        # Enumerate every feasible subset of the bounded action pool.  The pool
        # is already bounded; with the operational schedule-size cap this
        # remains a small finite domain while
        # avoiding the old, arbitrary "top seven pairs plus one triple" hole.
        maximum_size = min(int(world.max_schedule_size), len(selected_actions))
        for size in range(2, maximum_size + 1):
            for action_subset in combinations(selected_actions, size):
                schedule = tuple(sorted(action.action_id for action in action_subset))
                feasible, _ = schedule_feasibility(world, action_subset)
                dynamic_feasible, _ = action_is_dynamically_feasible(
                    self._base_world_for_dynamic_check(world),
                    state,
                    schedule,
                )
                operational_feasible, _, _ = availability.schedule_operational_feasibility(
                    action_subset
                )
                if feasible and dynamic_feasible and operational_feasible:
                    candidates.append(schedule)
        unique: list[tuple[int, ...]] = []
        for schedule in candidates:
            if schedule not in unique:
                unique.append(schedule)
        schedule_scores = {
            schedule: float(
                sum(cheap_scores.get(action_id, 0.0) for action_id in schedule)
                + hint_scores.get(schedule, 0.0)
            )
            for schedule in unique
        }
        unique.sort(
            key=lambda schedule: (
                -int(schedule in hint_scores),
                -schedule_scores[schedule],
                len(schedule),
                schedule,
            )
        )
        update_probability = {
            schedule: _schedule_update_probability(
                correct_update_probability_by_action,
                schedule,
            )
            for schedule in unique
        }
        detection_probability = {
            schedule: _schedule_update_probability(
                detection_delivery_probability_by_action,
                schedule,
            )
            for schedule in unique
        }
        role_sets = {
            schedule: frozenset(_action_role(by_id[action_id]) for action_id in schedule)
            for schedule in unique
        }
        geometry_by_schedule = {
            schedule: _structural_position_observation_geometry(
                world,
                tuple(by_id[action_id] for action_id in schedule),
                factor_by_action=factor_by_action,
            )
            for schedule in unique
        }
        full_rank_update_probability = {
            schedule: _full_rank_position_update_probability(
                world,
                tuple(by_id[action_id] for action_id in schedule),
                correct_update_probability_by_action,
                factor_by_action=factor_by_action,
            )
            for schedule in unique
        }
        position_separating_geometry_by_schedule = {
            schedule: _position_separating_observation_geometry(
                world,
                tuple(by_id[action_id] for action_id in schedule),
                factor_by_action=factor_by_action,
            )
            for schedule in unique
        }
        if self.config.innovation_verification_require_position_separating_rank:
            position_separating_update_probability_by_schedule = {
                schedule: _common_support_event_probability_lower_bound(
                    _position_separating_update_probability(
                        world,
                        tuple(by_id[action_id] for action_id in schedule),
                        verification_conditional_probability_by_action,
                        factor_by_action=factor_by_action,
                    ),
                    verification_retained_mass,
                )
                for schedule in unique
            }
        else:
            position_separating_update_probability_by_schedule = (
                full_rank_update_probability
            )
        differential_clock_certificate_by_schedule = {
            schedule: _differential_clock_verification_certificate(
                world,
                tuple(by_id[action_id] for action_id in schedule),
                factor_by_action=factor_by_action,
                require_position_separating_rank=(
                    self.config.innovation_verification_require_position_separating_rank
                ),
            )
            for schedule in unique
        }
        differential_clock_verification_probability_by_schedule = (
            {
                schedule: _common_support_event_probability_lower_bound(
                    _differential_clock_verification_probability(
                        world,
                        tuple(by_id[action_id] for action_id in schedule),
                        verification_conditional_probability_by_action,
                        factor_by_action=factor_by_action,
                    ),
                    verification_retained_mass,
                )
                for schedule in unique
                if differential_clock_certificate_by_schedule[schedule][
                    "differential_clock_identification_complete"
                ]
            }
            if self.config.innovation_verification_require_position_separating_rank
            else {schedule: 0.0 for schedule in unique}
        )
        if self.config.innovation_verification_require_position_separating_rank:
            differential_clock_verification_probability_by_schedule = {
                schedule: float(
                    differential_clock_verification_probability_by_schedule.get(
                        schedule, 0.0
                    )
                )
                for schedule in unique
            }
        verification_geometry_by_schedule = (
            position_separating_geometry_by_schedule
            if self.config.innovation_verification_require_position_separating_rank
            else geometry_by_schedule
        )
        verification_update_probability_by_schedule = (
            position_separating_update_probability_by_schedule
            if self.config.innovation_verification_require_position_separating_rank
            else full_rank_update_probability
        )
        verification_sensing_count_by_schedule = {
            schedule: int(values[0])
            for schedule, values in verification_geometry_by_schedule.items()
        }
        verification_geometry_rank_by_schedule = {
            schedule: int(
                values[3]
                if self.config.innovation_verification_require_position_separating_rank
                else values[2]
            )
            for schedule, values in verification_geometry_by_schedule.items()
        }
        verification_minimum_singular_value_by_schedule = {
            schedule: float(
                min(
                    values[2]
                    if self.config.innovation_verification_require_position_separating_rank
                    else values[1]
                )
            )
            if len(
                values[2]
                if self.config.innovation_verification_require_position_separating_rank
                else values[1]
            )
            else 0.0
            for schedule, values in verification_geometry_by_schedule.items()
        }
        required_sentinel_ids = set(sentinel_required_action_ids)
        sentinel_candidates = sorted(
            (
                schedule
                for schedule in unique
                if sentinel_scheduled
                and required_sentinel_ids
                and required_sentinel_ids.issubset(set(schedule))
            ),
            key=lambda schedule: (
                -verification_update_probability_by_schedule[schedule],
                -update_probability[schedule],
                -schedule_scores[schedule],
                len(schedule),
                schedule,
            ),
        )
        self.last_clock_sentinel_requirement = {
            **self.last_clock_sentinel_requirement,
            "precoverage_feasible_candidate_count": int(
                len(sentinel_candidates)
            ),
            "precoverage_feasible_candidate_schedules": [
                list(schedule) for schedule in sentinel_candidates
            ],
        }

        selected_schedules: list[tuple[int, ...]] = []

        def retain(schedule: tuple[int, ...] | None) -> None:
            if schedule is not None and schedule not in selected_schedules:
                selected_schedules.append(schedule)

        def best(predicate, *, key):
            rows = [schedule for schedule in unique if predicate(schedule)]
            return max(rows, key=key) if rows else None

        if self.config.include_noop_candidate and () in unique:
            retain(())
        # The sentinel is a pre-registered experiment constraint, not a cheap
        # proposal score.  Protect exactly one feasible schedule so bounded
        # truncation cannot remove the scheduled probe.  Exact graph and risk
        # reranking still select within the surviving sentinel domain.
        if not clock_hypothesis_active:
            for schedule in sentinel_candidates[:1]:
                retain(schedule)
        differential_clock_candidates = sorted(
            (
                schedule
                for schedule in unique
                if clock_hypothesis_active
                and differential_clock_certificate_by_schedule[schedule][
                    "differential_clock_identification_complete"
                ]
                and verification_update_probability_by_schedule[schedule] > 0.0
            ),
            key=lambda schedule: (
                -differential_clock_verification_probability_by_schedule[
                    schedule
                ],
                -verification_update_probability_by_schedule[schedule],
                -int(
                    differential_clock_certificate_by_schedule[schedule][
                        "target_clock_separation_rank_gain"
                    ]
                ),
                -schedule_scores[schedule],
                schedule,
            ),
        )
        sentinel_anchor_ids = {
            int(action.action_id) for action in sentinel_anchor_actions
        }
        sentinel_design_probe_ids = {
            int(value)
            for value in (
                sentinel_design.action_ids if sentinel_design is not None else ()
            )
        }
        joint_clock_sentinel_candidates = sorted(
            (
                schedule
                for schedule in unique
                if sentinel_scheduled
                and sentinel_anchor_ids
                and sentinel_anchor_ids.issubset(set(schedule))
                and sentinel_design_probe_ids.intersection(schedule)
                and differential_clock_certificate_by_schedule[schedule][
                    "differential_clock_identification_complete"
                ]
                and verification_update_probability_by_schedule[schedule] > 0.0
            ),
            key=lambda schedule: (
                -differential_clock_verification_probability_by_schedule[
                    schedule
                ],
                -verification_update_probability_by_schedule[schedule],
                -int(
                    differential_clock_certificate_by_schedule[schedule][
                        "target_clock_separation_rank_gain"
                    ]
                ),
                -schedule_scores[schedule],
                schedule,
            ),
        )
        self.last_clock_sentinel_requirement = {
            **self.last_clock_sentinel_requirement,
            "precoverage_joint_clock_sentinel_candidate_count": int(
                len(joint_clock_sentinel_candidates)
            ),
            "precoverage_joint_clock_sentinel_candidate_schedules": [
                list(schedule) for schedule in joint_clock_sentinel_candidates
            ],
            "joint_candidate_definition": (
                "one_physical_monostatic_absolute_range_anchor_plus_"
                "velocity_independent_position_complement_plus_clock_sensitive_"
                "cross_node_timing_plus_connected_time_calibration_with_joint_"
                "clock_identifiability_and_position_separating_rank3"
                if self.config.innovation_verification_require_position_separating_rank
                else
                "two_distinct_physical_monostatic_anchors_plus_one_registered_"
                "target_independent_clock_probe_plus_joint_structural_clock_"
                "identifiability_and_full_rank_position_update"
            ),
        }
        if clock_hypothesis_active:
            protected_sentinel_candidates = (
                joint_clock_sentinel_candidates or sentinel_candidates
            )
            for schedule in protected_sentinel_candidates[:1]:
                retain(schedule)
            self.last_clock_sentinel_requirement = {
                **self.last_clock_sentinel_requirement,
                "bounded_library_protected_sentinel_schedule": (
                    list(protected_sentinel_candidates[0])
                    if protected_sentinel_candidates
                    else None
                ),
                "bounded_library_sentinel_retention_mode": (
                    "joint_clock_identifiable_redundant_anchor_schedule"
                    if joint_clock_sentinel_candidates
                    else "fallback_registered_probe_and_anchor_schedule"
                ),
                "joint_clock_identifiability_prioritized_during_clock_alarm": bool(
                    joint_clock_sentinel_candidates
                ),
            }
        # A single structurally complete schedule must survive bounded-library
        # truncation. It is still only an admissible experiment: exact graph and
        # Bayes-risk reranking remains the final authority.
        for schedule in differential_clock_candidates[:1]:
            retain(schedule)
        # Agent proposals receive a checkpoint-independent bounded quota, not
        # authority.  Mandatory sentinels are already retained.  Half of the
        # total finite library remains reserved for the source-native analytic
        # role, joint, calibration, and geometry guards below.
        ranked_agent_hints = sorted(
            (item for item in unique if item in hint_scores),
            key=lambda item: (-hint_scores[item], -schedule_scores[item], item),
        )
        agent_hint_quota, analytic_guard_reserve = _bounded_agent_hint_quota(
            self.config.maximum_candidates,
            len(selected_schedules),
            len(ranked_agent_hints),
        )
        for schedule in ranked_agent_hints[:agent_hint_quota]:
            retain(schedule)
        verification_anchor_ids = {
            int(action.action_id) for action in verification_anchor_actions
        }
        verification_anchor_candidates = sorted(
            (
                schedule
                for schedule in unique
                if innovation_verification_active
                and verification_anchor_ids.intersection(schedule)
                and verification_sensing_count_by_schedule[schedule] >= 2
                and verification_geometry_rank_by_schedule[schedule] >= 3
                and verification_update_probability_by_schedule[schedule] > 0.0
            ),
            key=lambda schedule: (
                -verification_update_probability_by_schedule[schedule],
                -verification_minimum_singular_value_by_schedule[schedule],
                -update_probability[schedule],
                -schedule_scores[schedule],
                schedule,
            ),
        )
        # One anchor-bearing schedule is sufficient to expose the missing
        # absolute-range experiment to the exact selector. Protecting several
        # near-duplicate anchor schedules can crowd out complementary passive
        # or calibration alternatives in the bounded library.
        for schedule in verification_anchor_candidates[:1]:
            retain(schedule)
        for role in (
            "active_transmitter",
            "passive_receiver",
            "illuminator_reference_processor",
            "calibration_agent",
        ):
            retain(
                best(
                    lambda schedule, role=role: len(schedule) == 1
                    and role in role_sets[schedule],
                    key=lambda schedule: (
                        schedule_scores[schedule],
                        update_probability[schedule],
                        tuple(-value for value in schedule),
                    ),
                )
            )
        retain(
            best(
                lambda schedule: "active_transmitter" in role_sets[schedule]
                and bool(
                    role_sets[schedule]
                    & {"passive_receiver", "illuminator_reference_processor"}
                ),
                key=lambda schedule: (
                    update_probability[schedule],
                    schedule_scores[schedule],
                    tuple(-value for value in schedule),
                ),
            )
        )
        retain(
            best(
                lambda schedule: "calibration_agent" in role_sets[schedule]
                and len(role_sets[schedule]) > 1,
                key=lambda schedule: (
                    schedule_scores[schedule],
                    update_probability[schedule],
                    tuple(-value for value in schedule),
                ),
            )
        )
        # Protect structurally complementary acquisition schedules from being
        # crowded out by high-SNR but collinear single-view actions. The final
        # decision still comes from exact finite-outcome graph reranking.
        geometry_candidates = sorted(
            (
                schedule
                for schedule in unique
                if verification_sensing_count_by_schedule[schedule] >= 2
                and verification_geometry_rank_by_schedule[schedule] >= 3
                and verification_update_probability_by_schedule[schedule] > 0.0
            ),
            key=lambda schedule: (
                -verification_update_probability_by_schedule[schedule],
                -verification_minimum_singular_value_by_schedule[schedule],
                -update_probability[schedule],
                -schedule_scores[schedule],
                schedule,
            ),
        )
        for schedule in geometry_candidates[:3]:
            retain(schedule)
        retain(
            best(
                lambda schedule: bool(schedule),
                key=lambda schedule: (
                    update_probability[schedule],
                    detection_probability[schedule],
                    schedule_scores[schedule],
                    tuple(-value for value in schedule),
                ),
            )
        )
        for schedule in unique:
            retain(schedule)
        unique = selected_schedules[: self.config.maximum_candidates]
        retained_sentinel_candidates = [
            schedule
            for schedule in unique
            if required_sentinel_ids
            and required_sentinel_ids.issubset(set(schedule))
        ]
        self.last_clock_sentinel_requirement = {
            **self.last_clock_sentinel_requirement,
            "retained_candidate_count": int(
                len(retained_sentinel_candidates)
            ),
            "retained_candidate_schedules": [
                list(schedule) for schedule in retained_sentinel_candidates
            ],
        }
        self.last_schedule_update_probability = {
            schedule: float(update_probability[schedule]) for schedule in unique
        }
        self.last_generation_audit = {
            "precoverage_feasible_candidate_count": int(len(update_probability)),
            "retained_candidate_count": int(len(unique)),
            "bounded_action_pool_size": int(len(selected_actions)),
            "innovation_verification_active_at_generation": bool(
                innovation_verification_active
            ),
            "innovation_verification_hypotheses": sorted(
                {str(value) for value in innovation_verification_hypotheses}
            ),
            "differential_clock_hypothesis_active_at_generation": bool(
                clock_hypothesis_active
            ),
            "available_active_absolute_range_anchor_action_ids": sorted(
                verification_anchor_ids
            ),
            "available_position_angle_action_ids": [
                int(action.action_id) for action in verification_angle_actions
            ],
            "protected_position_angle_action_ids": [
                int(action.action_id)
                for action in verification_angle_actions[:2]
                if action in selected_actions
            ],
            "protected_active_absolute_range_anchor_candidate_count": int(
                min(1, len(verification_anchor_candidates))
            ),
            "active_absolute_range_anchor_generation_uses_truth": False,
            "available_clock_sensitive_cross_node_action_ids": sorted(
                int(action.action_id) for action in clock_sensitive_actions
            ),
            "protected_carrier_diverse_clock_action_ids": [
                int(action.action_id)
                for action in carrier_diverse_clock_actions
                if action in selected_actions
            ],
            "protected_carrier_diverse_clock_actions": [
                {
                    "action_id": int(action.action_id),
                    "carrier_hz": float(
                        world.waveform(action.waveform_id).carrier_hz
                    ),
                    "clock_support_node_ids": sorted(
                        _clock_support_node_ids(
                            layout,
                            factor_by_action[action.action_id],
                        )
                    ),
                }
                for action in carrier_diverse_clock_actions
                if action in selected_actions
            ],
            "carrier_diverse_clock_action_protection_uses_truth": False,
            "available_time_calibration_action_ids": sorted(
                int(action.action_id) for action in time_calibration_actions
            ),
            "clock_connected_time_calibration_action_ids": sorted(
                int(action.action_id)
                for action in connected_time_calibration_actions
            ),
            "protected_differential_clock_identification_candidate_count": int(
                min(1, len(differential_clock_candidates))
            ),
            "differential_clock_identification_by_candidate": {
                ",".join(map(str, schedule)): dict(
                    differential_clock_certificate_by_schedule[schedule]
                )
                for schedule in unique
            },
            "differential_clock_candidate_generation_uses_truth": False,
            "clock_sentinel": dict(self.last_clock_sentinel_requirement),
            "all_feasible_subsets_enumerated_within_bounded_action_pool": True,
            "candidate_retention_policy": (
                "noop_then_one_clock_sentinel_schedule_then_one_"
                "differential_clock_identification_schedule_then_"
                "bounded_agent_hint_quota_with_half_library_analytic_guard_"
                "reserve_then_one_verification_range_anchor_then_"
                "role_singles_then_active_passive_"
                "joint_then_calibration_joint_then_three_position_identifiable_"
                "geometry_schedules_"
                "then_max_correct_update_probability_then_proposal_score"
            ),
            "available_agent_hint_count": int(len(ranked_agent_hints)),
            "retained_agent_hint_quota": int(agent_hint_quota),
            "analytic_guard_reserve": int(analytic_guard_reserve),
            "correct_update_probability_by_candidate": {
                ",".join(map(str, schedule)): float(update_probability[schedule])
                for schedule in unique
            },
            "detection_delivery_probability_by_candidate": {
                ",".join(map(str, schedule)): float(detection_probability[schedule])
                for schedule in unique
            },
            "correct_update_probability_definition": (
                "1-product_a(1-P_delivery,a*P_planning_success,a*"
                "P_correct_association,a)"
            ),
            "planning_detection_probability": self.planning_probability.receipt,
            "planning_probability_by_action": planning_probability_by_action,
            "planning_probability_action_count": int(
                len(planning_probability_by_action)
            ),
            "planning_probability_frozen_AOA_application_count": int(
                sum(
                    bool(record.get("frozen_model_applied"))
                    for record in planning_probability_by_action.values()
                )
            ),
            "realized_target_scattering_read": False,
            "structural_position_geometry_by_candidate": {
                ",".join(map(str, schedule)): {
                    "sensing_action_count": int(geometry_by_schedule[schedule][0]),
                    "singular_values": [
                        float(value) for value in geometry_by_schedule[schedule][1]
                    ],
                    "rank": int(geometry_by_schedule[schedule][2]),
                    "rank_tolerance": float(geometry_by_schedule[schedule][3]),
                }
                for schedule in unique
            },
            "structural_geometry_uses_truth": False,
            "full_rank_position_update_probability_by_candidate": {
                ",".join(map(str, schedule)): float(
                    full_rank_update_probability[schedule]
                )
                for schedule in unique
            },
            "full_rank_position_update_probability_contract": (
                "independent_per_action_correct_delivery_and_association_"
                "with_at_least_two_sensing_actions_and_structural_rank3"
            ),
            "position_separating_geometry_required": bool(
                self.config.innovation_verification_require_position_separating_rank
            ),
            "position_separating_geometry_by_candidate": {
                ",".join(map(str, schedule)): {
                    "sensing_action_count": int(
                        position_separating_geometry_by_schedule[schedule][0]
                    ),
                    "contributing_action_count": int(
                        position_separating_geometry_by_schedule[schedule][1]
                    ),
                    "singular_values": [
                        float(value)
                        for value in position_separating_geometry_by_schedule[
                            schedule
                        ][2]
                    ],
                    "rank": int(
                        position_separating_geometry_by_schedule[schedule][3]
                    ),
                    "rank_tolerance": float(
                        position_separating_geometry_by_schedule[schedule][4]
                    ),
                }
                for schedule in unique
            },
            "position_separating_update_probability_by_candidate": {
                ",".join(map(str, schedule)): float(
                    position_separating_update_probability_by_schedule[schedule]
                )
                for schedule in unique
            },
            "differential_clock_verification_probability_by_candidate": {
                ",".join(map(str, schedule)): float(
                    differential_clock_verification_probability_by_schedule[
                        schedule
                    ]
                )
                for schedule in unique
            },
            "verification_probability_uses_registered_scattering_support_lower_bound": bool(
                self.config.innovation_verification_require_position_separating_rank
            ),
            "verification_scattering_support_lower_bound": (
                verification_scattering_support_audit
            ),
            "verification_usable_probability_conditional_on_rcs_floor_by_action": {
                str(action_id): float(probability)
                for action_id, probability in sorted(
                    verification_conditional_probability_by_action.items()
                )
            },
            "verification_scattering_retained_mass_applied_once_per_schedule": float(
                verification_retained_mass
            ),
            "verification_scattering_texture_group_count_used_for_bounded_action_pool": int(
                verification_texture_group_count
            ),
            "position_separating_geometry_excludes_target_velocity_sensitive_rows": True,
            "position_separating_geometry_is_necessary_not_global_sufficiency": True,
        }
        roles = tuple(
            tuple(sorted({_action_role(by_id[action_id]) for action_id in schedule}))
            for schedule in unique
        )
        payload = json.dumps(
            {
                "world_id": world.world_id,
                "epoch": state.epoch,
                "fusion_node": availability.fusion_node_id,
                "schedules": unique,
            },
            sort_keys=True,
        )
        library = CandidateLibrary(
            schedules=tuple(unique or [()]),
            generation_world_id=world.world_id,
            generation_hash=hashlib.sha256(payload.encode()).hexdigest(),
            role_by_candidate=roles or ((),),
            exact_bounded_domain=True,
        )
        library.validate(world)
        return library, schedule_scores

    @staticmethod
    def _base_world_for_dynamic_check(world: PhysicalWorld) -> PhysicalWorld:
        # Episode dynamic feasibility needs the same stable action identifiers.
        # The projected dynamic world is otherwise an admissible base record.
        return world

    @staticmethod
    def _delivery_probability(
        availability: DistributedAvailabilityManager,
        action_id: int,
    ) -> float:
        return float(np.clip(availability.delivery_probability(action_id), 0.0, 1.0))


def _position_credible_ellipsoid_certificate(
    covariance_m2: np.ndarray,
    *,
    credible_probability: float,
    maximum_semi_axis_m: float | None,
) -> dict[str, Any]:
    """Certify the weakest posterior position direction without reading truth.

    For a ``d``-dimensional Gaussian position marginal, the probability-``p``
    ellipsoid is ``e.T @ P^-1 @ e <= chi2_d(p)``.  Its largest semi-axis is
    ``sqrt(chi2_d(p) * lambda_max(P))``.  The D-optimal radius alone is the
    geometric mean of marginal standard deviations and can therefore hide one
    poorly observed direction; this certificate cannot.
    """

    covariance = np.asarray(covariance_m2, dtype=float)
    enabled = maximum_semi_axis_m is not None
    audit: dict[str, Any] = {
        "schema": "position-credible-ellipsoid-certificate/v1",
        "enabled": bool(enabled),
        "credible_probability": float(credible_probability),
        "maximum_allowed_semi_axis_m": (
            None if maximum_semi_axis_m is None else float(maximum_semi_axis_m)
        ),
        "truth_or_future_event_read": False,
        "posterior_source": "sole_authoritative_graph_position_marginal",
    }
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or covariance.shape[0] < 1
        or not np.all(np.isfinite(covariance))
    ):
        return {
            **audit,
            "status": "failed_invalid_covariance",
            "satisfied": False,
            "dimension": int(covariance.shape[0]) if covariance.ndim == 2 else 0,
        }
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    numerical_tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -numerical_tolerance:
        return {
            **audit,
            "status": "failed_non_psd_covariance",
            "satisfied": False,
            "dimension": int(covariance.shape[0]),
            "covariance_eigenvalues_m2": eigenvalues.tolist(),
        }
    eigenvalues = np.maximum(eigenvalues, 0.0)
    dimension = int(covariance.shape[0])
    chi_square_quantile = float(
        2.0 * gammaincinv(0.5 * dimension, float(credible_probability))
    )
    maximum_eigenvalue_m2 = float(np.max(eigenvalues))
    maximum_axis_standard_deviation_m = math.sqrt(maximum_eigenvalue_m2)
    maximum_credible_semi_axis_m = math.sqrt(
        chi_square_quantile * maximum_eigenvalue_m2
    )
    satisfied = bool(
        not enabled
        or maximum_credible_semi_axis_m <= float(maximum_semi_axis_m)
    )
    return {
        **audit,
        "status": "passed" if satisfied else "failed_directional_uncertainty",
        "satisfied": satisfied,
        "dimension": dimension,
        "chi_square_quantile": chi_square_quantile,
        "covariance_eigenvalues_m2": eigenvalues.tolist(),
        "maximum_covariance_eigenvalue_m2": maximum_eigenvalue_m2,
        "maximum_axis_standard_deviation_m": maximum_axis_standard_deviation_m,
        "maximum_credible_semi_axis_m": maximum_credible_semi_axis_m,
        "mathematical_definition": (
            "sqrt(chi2_dimension(credible_probability)*lambda_max(P_position))"
        ),
    }


class GraphSafeDecisionEngine:
    """Proposal -> exact graph counterfactual -> campaign shield -> action."""

    def __init__(
        self,
        base_world: PhysicalWorld,
        config: RuntimeConfig,
        availability: DistributedAvailabilityManager,
        belief: GraphBeliefEngine,
        mission: MissionIR,
        multi_target_engine=None,
    ) -> None:
        config.validate()
        mission.validate()
        self.base_world = base_world
        self.config = config
        self.availability = availability
        self.belief = belief
        self.mission = mission
        self.multi_target_engine = multi_target_engine
        self.planning_probability = PlanningDetectionProbability(config)
        self.schedule_update_probability_calibration = (
            ScheduleUpdateProbabilityCalibration.from_checkpoint(
                config.schedule_update_probability_calibration_checkpoint_path,
                str(
                    config.schedule_update_probability_calibration_checkpoint_sha256
                ),
            )
            if config.schedule_update_probability_calibration_enabled
            else ScheduleUpdateProbabilityCalibration.disabled()
        )
        self.generator = GraphCandidateGenerator(
            config, mission, self.planning_probability
        )
        self._initial_position_d_radius_m: float | None = None
        self._target_detection_epochs: list[int] = []
        self._target_associated_update_epochs: list[int] = []
        self._clock_invariant_associated_update_epochs: list[int] = []
        self._latest_clock_invariant_associated_action_ids: tuple[int, ...] = ()
        self._track_confirmed_once = False
        self._legacy_reconfirmation_pending = False
        self._legacy_reconfirmation_alarm_epoch: int | None = None
        self._legacy_reconfirmation_completed_epoch: int | None = None
        self._innovation_health = InnovationHealthMonitor(
            enabled=config.innovation_health_enabled,
            episode_alpha=config.innovation_health_episode_alpha,
            minimum_detected_actions_to_clear=(
                config.innovation_health_min_detected_actions_to_clear
            ),
            minimum_consistent_verification_epochs_to_clear=(
                config.innovation_health_min_consistent_verification_epochs_to_clear
            ),
        )
        self._last_selection_was_innovation_verification = False
        self._last_innovation_verification_world: PhysicalWorld | None = None
        self._last_innovation_verification_action_ids: tuple[int, ...] = ()
        self._last_innovation_verification_hypotheses: tuple[str, ...] = ()
        self.threat_belief: np.ndarray | None = None
        self.threat_scenario_ids: tuple[str, ...] | None = None
        self.last_game_model = None
        self.previous_game_candidate = -1
        self.game_history: list[dict[str, Any]] = []
        self._v73_adaptive_controller = None
        self._v73_gate_checkpoint_audit = None
        if self.config.v73_adaptive_enabled:
            from .v73_adaptive_belief_risk import (
                AdaptiveMissionConstrainedController,
                load_monotone_gate_checkpoint,
            )

            gate = None
            if self.config.v73_gate_checkpoint is not None:
                gate, self._v73_gate_checkpoint_audit = load_monotone_gate_checkpoint(
                    self.config.v73_gate_checkpoint,
                    self.config.v73_gate_checkpoint_sha256,
                )
            self._v73_adaptive_controller = AdaptiveMissionConstrainedController(
                gate=gate,
                mission_tolerance=self.config.v73_mission_tolerance,
                confidence_z=self.config.v73_confidence_z,
                max_certificate_relative_se=self.config.v73_max_certificate_relative_se,
                hard_mission_gate=self.config.v73_hard_mission_gate,
                reference_noninferiority_gate=(
                    self.config.v73_reference_noninferiority_gate
                ),
                reference_mission_loss_margin=(
                    self.config.v73_reference_mission_loss_margin
                ),
                reference_track_update_margin=(
                    self.config.v73_reference_track_update_margin
                ),
                reference_fallback_on_risk_uncertainty=(
                    self.config.v73_reference_fallback_on_risk_uncertainty
                ),
            )

    def innovation_health_snapshot(self) -> dict[str, Any]:
        """Return the causal model-health state available before a decision.

        The runtime uses this read-only view to configure a wider receiver
        search support.  It contains only arrived innovation statistics and
        exposes neither simulator truth nor future events.
        """

        return dict(self._innovation_health.snapshot())

    def _track_phase_contract(self, state: EpisodeState) -> dict[str, Any]:
        """Derive the causal track phase from arrived factors and graph uncertainty."""

        position_covariance_m2 = np.asarray(
            self.belief.record().position_covariance_m2,
            dtype=float,
        )
        current_radius = float(d_optimal_radius_m(position_covariance_m2))
        directional_confirmation = _position_credible_ellipsoid_certificate(
            position_covariance_m2,
            credible_probability=(
                self.mission.confirmation_position_credible_probability
            ),
            maximum_semi_axis_m=(
                self.mission.confirmation_max_position_ellipsoid_semi_axis_m
            ),
        )
        if self._initial_position_d_radius_m is None:
            self._initial_position_d_radius_m = max(current_radius, 1.0e-12)
        initial_radius = max(self._initial_position_d_radius_m, 1.0e-12)
        raw_detection_epochs = tuple(sorted(set(self._target_detection_epochs)))
        associated_update_epochs = tuple(
            sorted(set(self._target_associated_update_epochs))
        )
        associated_evidence_required = bool(
            self.config.track_phase_requires_associated_update_evidence
        )
        distinct_epochs = (
            associated_update_epochs
            if associated_evidence_required
            else raw_detection_epochs
        )
        last_detection_epoch = distinct_epochs[-1] if distinct_epochs else None
        age_steps = (
            int(state.epoch) + 1
            if last_detection_epoch is None
            else max(0, int(state.epoch) - int(last_detection_epoch))
        )
        confirmation_radius = (
            self.mission.confirmation_radius_fraction_of_initial * initial_radius
        )
        loss_radius = self.mission.loss_radius_fraction_of_initial * initial_radius
        confirmation_evidence = (
            len(distinct_epochs)
            >= self.mission.confirmation_min_distinct_detection_epochs
            and current_radius <= confirmation_radius
            and bool(directional_confirmation["satisfied"])
        )
        if confirmation_evidence:
            self._track_confirmed_once = True
        innovation_health = self._innovation_health.snapshot()
        lost_by_posterior_or_coast = bool(
            self._track_confirmed_once
            and (
                current_radius > loss_radius
                or age_steps > self.mission.maximum_coast_steps
            )
        )
        preserve_confirmed_track = bool(
            self.config.innovation_alarm_preserves_confirmed_track
        )
        clock_robust_reporting_enabled = bool(
            self.config.innovation_alarm_clock_robust_track_reporting_enabled
        )
        active_alarm_hypotheses = tuple(
            str(item)
            for item in innovation_health.get("active_alarm_hypotheses", ())
        )
        clock_only_alarm = bool(
            bool(innovation_health.get("active", False))
            and set(active_alarm_hypotheses)
            == {DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS}
        )
        clock_invariant_update_epochs = tuple(
            sorted(
                set(
                    int(item)
                    for item in getattr(
                        self, "_clock_invariant_associated_update_epochs", ()
                    )
                )
            )
        )
        last_clock_invariant_update_epoch = (
            clock_invariant_update_epochs[-1]
            if clock_invariant_update_epochs
            else None
        )
        clock_invariant_update_age_steps = (
            int(state.epoch) + 1
            if last_clock_invariant_update_epoch is None
            else max(
                0,
                int(state.epoch) - int(last_clock_invariant_update_epoch),
            )
        )
        raw_first_alarm_epoch = innovation_health.get("first_alarm_epoch")
        first_alarm_epoch = (
            int(raw_first_alarm_epoch)
            if raw_first_alarm_epoch is not None
            else None
        )
        clock_invariant_anchor_is_post_alarm = bool(
            last_clock_invariant_update_epoch is not None
            and (
                first_alarm_epoch is None
                or int(last_clock_invariant_update_epoch)
                >= int(first_alarm_epoch)
            )
        )
        clock_robust_track_report_certificate = bool(
            self._track_confirmed_once
            and bool(innovation_health.get("active", False))
            and preserve_confirmed_track
            and clock_robust_reporting_enabled
            and clock_only_alarm
            and not lost_by_posterior_or_coast
            and clock_invariant_anchor_is_post_alarm
            and clock_invariant_update_age_steps
            <= int(self.mission.maximum_coast_steps)
            and current_radius <= confirmation_radius
            and bool(directional_confirmation["satisfied"])
        )
        legacy_reconfirmation_pending = bool(
            getattr(self, "_legacy_reconfirmation_pending", False)
        )
        legacy_reconfirmation_alarm_epoch = getattr(
            self, "_legacy_reconfirmation_alarm_epoch", None
        )
        if (
            self._track_confirmed_once
            and bool(innovation_health["active"])
            and not preserve_confirmed_track
        ):
            legacy_reconfirmation_pending = True
            if legacy_reconfirmation_alarm_epoch is None:
                raw_alarm_epoch = innovation_health.get("first_alarm_epoch")
                legacy_reconfirmation_alarm_epoch = (
                    int(raw_alarm_epoch)
                    if raw_alarm_epoch is not None
                    else max(0, int(state.epoch) - 1)
                )
            self._legacy_reconfirmation_pending = True
            self._legacy_reconfirmation_alarm_epoch = (
                legacy_reconfirmation_alarm_epoch
            )
        post_alarm_reconfirmation_epochs = tuple(
            epoch
            for epoch in distinct_epochs
            if legacy_reconfirmation_alarm_epoch is not None
            and int(epoch) > int(legacy_reconfirmation_alarm_epoch)
        )
        legacy_reconfirmation_required_epochs = int(
            self.mission.confirmation_min_distinct_detection_epochs
        )
        legacy_reconfirmation_evidence = bool(
            legacy_reconfirmation_pending
            and not bool(innovation_health["active"])
            and len(post_alarm_reconfirmation_epochs)
            >= legacy_reconfirmation_required_epochs
            and current_radius <= confirmation_radius
            and bool(directional_confirmation["satisfied"])
        )
        if legacy_reconfirmation_evidence:
            legacy_reconfirmation_pending = False
            self._legacy_reconfirmation_pending = False
            self._legacy_reconfirmation_completed_epoch = int(state.epoch)
        if not self._track_confirmed_once:
            phase = "acquisition"
            track_status = "unconfirmed"
            sensing_mode = "acquisition"
            reported_track_exists = False
            required = self.mission.minimum_update_probability_acquisition
        elif lost_by_posterior_or_coast:
            phase = "reacquisition"
            track_status = "lost"
            sensing_mode = "reacquisition"
            reported_track_exists = False
            required = self.mission.minimum_update_probability_reacquisition
        elif bool(innovation_health["active"]):
            phase = "reacquisition"
            track_status = (
                "confirmed_clock_robust_track_under_verification"
                if clock_robust_track_report_certificate
                else "confirmed_track_memory_under_verification"
                if preserve_confirmed_track
                else "verification_pending"
            )
            sensing_mode = (
                "verification" if preserve_confirmed_track else "reacquisition"
            )
            # A generic innovation alarm can invalidate the local posterior,
            # so abstention remains the default.  A clock-only alarm is more
            # specific: a recent JPDA-associated active-monostatic update is
            # invariant to a static network clock-offset gauge in the current
            # physical model.  If that anchor is fresh and the sole graph
            # posterior remains inside the registered confirmation radius, the
            # controller may keep reporting the track while still executing
            # verification sensing.  No truth or scheduled event is read.
            reported_track_exists = bool(
                clock_robust_track_report_certificate
            )
            required = self.mission.minimum_update_probability_reacquisition
        elif legacy_reconfirmation_pending:
            phase = "reacquisition"
            track_status = "reconfirmation_pending_after_model_health_alarm"
            sensing_mode = "reacquisition"
            reported_track_exists = False
            required = self.mission.minimum_update_probability_reacquisition
        else:
            phase = "confirmed_tracking"
            track_status = "confirmed_tracking"
            sensing_mode = "tracking"
            reported_track_exists = True
            required = self.mission.minimum_update_probability_confirmed
        return {
            "phase": phase,
            "phase_semantics": (
                "legacy_selector_phase_for_backward_compatibility"
            ),
            "track_status": track_status,
            "sensing_mode": sensing_mode,
            "reported_track_exists": bool(reported_track_exists),
            "required_correct_update_probability": float(required),
            "initial_position_d_radius_m": float(initial_radius),
            "current_position_d_radius_m": float(current_radius),
            "confirmation_radius_m": float(confirmation_radius),
            "position_directional_confirmation_certificate": (
                directional_confirmation
            ),
            "loss_radius_m": float(loss_radius),
            "distinct_arrived_detection_epochs": list(distinct_epochs),
            "last_arrived_detection_epoch": last_detection_epoch,
            "raw_arrived_detection_epochs": list(raw_detection_epochs),
            "jpda_associated_update_epochs": list(associated_update_epochs),
            "confirmation_evidence_source": (
                "jpda_joint_map_associated_noncalibration_update"
                if associated_evidence_required
                else "raw_receiver_detection"
            ),
            "associated_update_evidence_required": associated_evidence_required,
            "coast_age_steps": int(age_steps),
            "confirmed_once": bool(self._track_confirmed_once),
            "innovation_health": innovation_health,
            "innovation_alarm_forced_reacquisition": bool(
                (innovation_health["active"] or legacy_reconfirmation_pending)
                and self._track_confirmed_once
                and not preserve_confirmed_track
            ),
            "innovation_alarm_requested_verification": bool(
                innovation_health["active"]
                and self._track_confirmed_once
                and preserve_confirmed_track
                and not lost_by_posterior_or_coast
            ),
            "innovation_alarm_preserves_confirmed_track": preserve_confirmed_track,
            "innovation_alarm_preserves_track_memory": preserve_confirmed_track,
            "clock_robust_track_reporting_enabled": (
                clock_robust_reporting_enabled
            ),
            "clock_robust_track_report_certificate": bool(
                clock_robust_track_report_certificate
            ),
            "clock_robust_track_report_certificate_components": {
                "clock_only_alarm": bool(clock_only_alarm),
                "active_alarm_hypotheses": list(active_alarm_hypotheses),
                "posterior_inside_confirmation_radius": bool(
                    current_radius <= confirmation_radius
                ),
                "recent_clock_invariant_associated_update": bool(
                    last_clock_invariant_update_epoch is not None
                    and clock_invariant_update_age_steps
                    <= int(self.mission.maximum_coast_steps)
                ),
                "clock_invariant_anchor_is_post_alarm": bool(
                    clock_invariant_anchor_is_post_alarm
                ),
                "last_clock_invariant_associated_update_epoch": (
                    last_clock_invariant_update_epoch
                ),
                "clock_invariant_associated_update_age_steps": int(
                    clock_invariant_update_age_steps
                ),
                "maximum_clock_invariant_anchor_age_steps": int(
                    self.mission.maximum_coast_steps
                ),
                "lost_by_posterior_or_coast": bool(
                    lost_by_posterior_or_coast
                ),
                "truth_or_future_event_read": False,
            },
            "legacy_reconfirmation_pending": bool(
                legacy_reconfirmation_pending
            ),
            "legacy_reconfirmation_alarm_epoch": (
                legacy_reconfirmation_alarm_epoch
            ),
            "legacy_reconfirmation_required_distinct_update_epochs": int(
                legacy_reconfirmation_required_epochs
            ),
            "legacy_reconfirmation_observed_update_epochs": list(
                post_alarm_reconfirmation_epochs
            ),
            "legacy_reconfirmation_evidence_satisfied": bool(
                legacy_reconfirmation_evidence
            ),
            "legacy_reconfirmation_completed_epoch": getattr(
                self, "_legacy_reconfirmation_completed_epoch", None
            ),
            "legacy_reconfirmation_uses_truth": False,
            "innovation_alarm_output_abstention": bool(
                innovation_health["active"]
                and self._track_confirmed_once
                and not lost_by_posterior_or_coast
                and not clock_robust_track_report_certificate
            ),
            "reported_track_policy": (
                "report_clock_robust_track_while_continuing_verification"
                if clock_robust_track_report_certificate
                else
                "abstain_during_model_health_alarm_until_causal_verification_clears"
                if preserve_confirmed_track and innovation_health["active"]
                else "legacy_identity_requires_post_alarm_associated_reconfirmation"
                if legacy_reconfirmation_pending
                else "legacy_phase_based_report"
            ),
            "track_loss_evidence": {
                "posterior_radius_exceeds_loss_radius": bool(
                    current_radius > loss_radius
                ),
                "coast_age_exceeds_limit": bool(
                    age_steps > self.mission.maximum_coast_steps
                ),
                "loss_declared": bool(lost_by_posterior_or_coast),
            },
            "truth_or_future_event_read": False,
        }

    def _stratified_shortlist(
        self,
        world: PhysicalWorld,
        library: CandidateLibrary,
        proposal_scores: np.ndarray,
        safe_indices: Sequence[int],
    ) -> list[int]:
        """Retain physical role and track-update strata before exact reranking."""

        capacity = min(self.config.shortlist_size, len(safe_indices))
        if capacity <= 0:
            return []
        selected: list[int] = []

        def retain(index: int | None) -> None:
            if index is not None and index in safe_indices and index not in selected:
                if len(selected) < capacity:
                    selected.append(index)

        def best(predicate, key):
            rows = [index for index in safe_indices if predicate(index)]
            return max(rows, key=key) if rows else None

        noop = next(
            (index for index in safe_indices if not library.schedules[index]),
            None,
        )
        retain(noop)
        erasure_robust_retained_index: int | None = None
        erasure_robust_ratios: dict[int, float] = {}
        if self.config.erasure_robust_position_geometry_enabled:
            by_id = {int(action.action_id): action for action in world.actions}
            for index in safe_indices:
                action_ids = library.schedules[index]
                if not action_ids:
                    continue
                certificate = _erasure_robust_position_frame_certificate(
                    world,
                    tuple(by_id[int(action_id)] for action_id in action_ids),
                    maximum_erasures=(
                        self.config.erasure_robust_position_geometry_max_erasures
                    ),
                )
                erasure_robust_ratios[int(index)] = float(
                    certificate["minimum_singular_value_ratio"]
                )
            if erasure_robust_ratios:
                erasure_robust_retained_index = max(
                    erasure_robust_ratios,
                    key=lambda index: (
                        erasure_robust_ratios[index],
                        proposal_scores[index],
                        -index,
                    ),
                )
                retain(erasure_robust_retained_index)
        self._last_erasure_robust_shortlist_retention = {
            "enabled": bool(
                self.config.erasure_robust_position_geometry_enabled
            ),
            "retained_candidate_index": erasure_robust_retained_index,
            "retained_minimum_singular_value_ratio": (
                float(erasure_robust_ratios[erasure_robust_retained_index])
                if erasure_robust_retained_index is not None
                else None
            ),
            "evaluated_safe_candidate_count": len(erasure_robust_ratios),
            "candidate_minimum_singular_value_ratio": {
                str(index): float(value)
                for index, value in sorted(erasure_robust_ratios.items())
            },
            "truth_or_future_event_read": False,
        }
        sentinel_requirement = self.generator.last_clock_sentinel_requirement
        sentinel_required_ids = {
            int(value)
            for value in sentinel_requirement.get("required_action_ids", ())
        }
        if bool(sentinel_requirement.get("scheduled_this_epoch", False)) and (
            sentinel_required_ids
        ):
            retain(
                best(
                    lambda index: sentinel_required_ids.issubset(
                        set(library.schedules[index])
                    ),
                    lambda index: (
                        proposal_scores[index],
                        -len(library.schedules[index]),
                        -index,
                    ),
                )
            )
        role_sets = [set(roles) for roles in library.role_by_candidate]
        update = self.generator.last_schedule_update_probability
        for role in (
            "active_transmitter",
            "passive_receiver",
            "illuminator_reference_processor",
            "calibration_agent",
        ):
            retain(
                best(
                    lambda index, role=role: role in role_sets[index],
                    lambda index: (
                        proposal_scores[index],
                        update.get(library.schedules[index], 0.0),
                        -index,
                    ),
                )
            )
        retain(
            best(
                lambda index: "active_transmitter" in role_sets[index]
                and bool(
                    role_sets[index]
                    & {"passive_receiver", "illuminator_reference_processor"}
                ),
                lambda index: (
                    update.get(library.schedules[index], 0.0),
                    proposal_scores[index],
                    -index,
                ),
            )
        )
        retain(
            best(
                lambda index: bool(library.schedules[index]),
                lambda index: (
                    update.get(library.schedules[index], 0.0),
                    proposal_scores[index],
                    -index,
                ),
            )
        )
        for index in sorted(
            safe_indices,
            key=lambda item: (-proposal_scores[item], item),
        ):
            retain(index)
        return selected

    @staticmethod
    def _v73_digest(value: Any) -> str:
        raw=json.dumps(to_builtin(value),sort_keys=True,separators=(",",":"),default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    def _v73_controller_state(self, state: EpisodeState, exact_rows: Sequence[CandidateScore],
                              library: CandidateLibrary):
        from .v73_adaptive_belief_risk import ControllerVisibleState
        belief_record=self.belief.record()
        belief_payload=asdict(belief_record)
        posterior_digest=self._v73_digest(belief_payload)
        availability_record=self.availability.latest_record
        availability_digest=self._v73_digest(asdict(availability_record))
        tracks=tuple(getattr(self.multi_target_engine,"tracks",()) or ())
        existences=[float(item.existence_probability) for item in tracks]
        if not existences: existences=[1.0]
        def binary_entropy(probability:float)->float:
            p=float(np.clip(probability,1.0e-12,1.0-1.0e-12))
            return float(-p*math.log(p)-(1.0-p)*math.log(1.0-p))
        cardinality_entropy=float(sum(binary_entropy(value) for value in existences))
        association_values=[]
        ambiguous=0
        history=tuple(getattr(self.multi_target_engine,"history",()) or ())
        if history:
            for action in history[-1].get("actions",()):
                if action.get("status")!="passed": continue
                association_values.append(float(action.get("preassociation_mean_normalized_entropy",0.0)))
                ambiguous+=int(action.get("preassociation_ambiguous_track_count",0))
        association_entropy=float(np.mean(association_values)) if association_values else 0.0
        position_radius=d_optimal_radius_m(np.asarray(belief_record.position_covariance_m2,float))
        target_scale=max(float(self.config.nonlocal_task_risk_cutoff_m),1.0)
        target_uncertainty=(position_radius/target_scale)**2
        joint_covariance=np.asarray(self.belief.current_state_covariance(),float)
        nuisance_block=joint_covariance[6:,6:] if joint_covariance.shape[0]>6 else np.zeros((0,0))
        nuisance_uncertainty=(float(np.trace(nuisance_block))/max(nuisance_block.shape[0],1)
                              if nuisance_block.size else 0.0)
        telemetry=availability_record.communication_contract.get("node_transport_telemetry",{})
        node_health=[]
        for values in telemetry.values():
            load=min(1.0,max(0.0,float(values.get("compute_load_fraction",0.0))))
            loss=min(1.0,max(0.0,float(values.get("packet_loss_probability",0.0))))
            node_health.append(1.0-max(load,loss))
        deliveries=availability_record.communication_contract.get("action_delivery_probability",{})
        link_health=[float(value) for value in deliveries.values()]
        energy_fraction=float(state.resources.energy_remaining_j/max(self.base_world.max_total_energy_j,1.0e-12))
        emission_fraction=float(state.resources.emission_remaining_j/max(self.base_world.max_total_emission_j,1.0e-12))
        return ControllerVisibleState(
            cycle_index=int(state.epoch),decision_time_s=float(state.epoch*self.config.dt_s),
            existence_mean=float(np.mean(existences)),existence_min=float(min(existences)),
            cardinality_entropy_nats=cardinality_entropy,
            association_entropy_nats=association_entropy,
            mixture_entropy_nats=association_entropy,
            multimodality_index=float(ambiguous/max(len(existences),1)),
            target_uncertainty_normalized=float(max(0.0,target_uncertainty)),
            nuisance_uncertainty_normalized=float(max(0.0,nuisance_uncertainty)),
            normalized_innovation_sq=max(
                1.0,
                float(
                    self._innovation_health.snapshot()[
                        "latest_maximum_threshold_ratio"
                    ]
                ),
            ),
            newest_factor_age_s=0.0 if belief_record.accepted_factor_ids else float((state.epoch+1)*self.config.dt_s),
            node_health_min=float(min(node_health,default=1.0)),
            link_health_min=float(min(link_health,default=1.0)),
            resource_remaining_fraction=float(np.clip(energy_fraction,0.0,1.0)),
            emission_remaining_fraction=float(np.clip(emission_fraction,0.0,1.0)),
            candidate_count=len(exact_rows),posterior_digest=posterior_digest,
            availability_digest=availability_digest,
            extra_observables={
                "safe_candidate_fraction":len(exact_rows)/max(len(library.schedules),1),
                "accepted_factor_log_count":math.log1p(len(belief_record.accepted_factor_ids)),
            },
        )

    def _v73_select(self, state: EpisodeState, exact_rows: Sequence[CandidateScore],
                    library: CandidateLibrary):
        from .v73_adaptive_belief_risk import CandidateCertificate
        if self._v73_adaptive_controller is None: raise RuntimeError("V73 controller was not initialized")
        if abs(self.config.nonlocal_task_risk_weight)>1.0e-15:
            raise RuntimeError("V73 risk would be double counted inside the information score")
        noop_rows=[row for row in exact_rows if not row.action_ids]
        if len(noop_rows)!=1: raise RuntimeError("V73 exact shortlist requires exactly one explicit no-op")
        cutoff=max(float(self.config.nonlocal_task_risk_cutoff_m),1.0e-12)
        def mission_loss(row:CandidateScore)->float:
            radius=d_optimal_radius_m(np.asarray(row.post_position_covariance_m2,float))
            transport=float(row.worst_operational_bures_degradation_m)/cutoff
            return float(self.mission.position_weight*radius/cutoff
                         +row.normalized_resource_cost
                         +row.switching_cost
                         +0.10*self.mission.tail_risk_weight*transport)
        noop_loss=mission_loss(noop_rows[0])
        physical_rows=[row for row in exact_rows if row.action_ids]
        task_reference=(
            max(
                physical_rows,
                key=lambda row:(
                    float(row.exact_score),
                    float(
                        row.metadata.get(
                            "calibrated_track_update_probability_point",
                            row.metadata.get(
                                "expected_track_maintaining_update_probability",
                                0.0,
                            ),
                        )
                    ),
                    -int(row.candidate_index),
                ),
            )
            if physical_rows
            else None
        )
        task_reference_index=(
            int(task_reference.candidate_index)
            if task_reference is not None
            else None
        )
        certificates=[]
        candidate_index_by_id={}
        for row in exact_rows:
            action_id=f"candidate:{int(row.candidate_index)}"
            candidate_index_by_id[action_id]=int(row.candidate_index)
            trace_digest=self._v73_digest(asdict(row))
            if not row.action_ids:
                certificates.append(CandidateCertificate(action_id,True,True,0.0,0.0,0.0,0.0,0.0,
                                                         physical_trace_digest=trace_digest,
                                                         track_update_probability=0.0))
                continue
            risk=row.metadata.get("nonlocal_task_risk")
            if not isinstance(risk,Mapping) or risk.get("status")!="passed":
                raise RuntimeError("V73 candidate omitted a passed posterior-predictive set-risk certificate")
            scale=max(float(risk["risk_normalization_power_m2"]),1.0e-12)
            usage=schedule_resource_usage(
                self.base_world,
                tuple(next(action for action in self.base_world.actions if action.action_id==aid) for aid in row.action_ids),
            )
            certificates.append(CandidateCertificate(
                action_id,True,False,float(row.exact_score),
                float(risk["paired_robust_risk_reduction_mean_m2"])/scale,
                float(risk["paired_robust_risk_reduction_standard_error_m2"])/scale,
                mission_loss(row)-noop_loss,0.0,
                resource_cost_normalized=float(row.normalized_resource_cost),
                emission_cost_normalized=float(usage.emission_exposure_j/max(self.base_world.max_total_emission_j,1.0e-12)),
                physical_trace_digest=trace_digest,
                metadata={
                    "candidate_index":int(row.candidate_index),
                    "set_risk_normalization_power_m2":scale,
                    "task_reference_definition":(
                        "maximum exact local information objective within the "
                        "same mission-admissible frozen candidate set"
                    ),
                },
                is_task_reference=(
                    task_reference_index is not None
                    and int(row.candidate_index)==task_reference_index
                ),
                track_update_probability=float(
                    row.metadata.get(
                        "calibrated_track_update_probability_lcb_95_one_sided",
                        row.metadata.get(
                            "expected_track_maintaining_update_probability",
                            0.0,
                        ),
                    )
                ),
            ))
        controller_state=self._v73_controller_state(state,exact_rows,library)
        decision=self._v73_adaptive_controller.decide(controller_state,certificates)
        proposed=candidate_index_by_id[decision.selected_action_id]
        selected_certificate_trace=next(
            row
            for row in decision.candidate_traces
            if row["action_id"]==decision.selected_action_id
        )
        metadata={
            "mode":"V73_monotone_adaptive_belief_risk","selected_certificate_id":decision.selected_action_id,
            "adaptive_lambda":decision.lambda_t,"dual_before":decision.nu_before,
            "no_op_reason":decision.no_op_reason,"fallback_used":decision.fallback_used,
            "fallback_reason":decision.fallback_reason,
            "pre_shield_selected_risk_certificate_evidence_ok":bool(
                selected_certificate_trace["evidence_ok"]
            ),
            "pre_shield_selected_risk_objective_used":bool(
                selected_certificate_trace["evidence_ok"]
                and decision.fallback_reason is None
            ),
            "task_reference_fallback_due_risk_uncertainty":bool(
                decision.fallback_reason
                == "risk_evidence_insufficient_fallback_to_task_reference"
            ),
            "gate_trace":dict(decision.gate_trace),"candidate_traces":[dict(row) for row in decision.candidate_traces],
            "posterior_digest":controller_state.posterior_digest,
            "availability_digest":controller_state.availability_digest,
            "candidate_universe_digest":library.generation_hash,
            "task_reference_candidate_id":(
                None
                if task_reference_index is None
                else f"candidate:{task_reference_index}"
            ),
            "task_reference_exact_score":(
                None if task_reference is None else float(task_reference.exact_score)
            ),
            "task_reference_available":task_reference is not None,
            "physical_task_candidate_count":len(physical_rows),
            "reference_noninferiority_gate":bool(
                self.config.v73_reference_noninferiority_gate
            ),
            "reference_fallback_on_risk_uncertainty":bool(
                self.config.v73_reference_fallback_on_risk_uncertainty
            ),
            "reference_mission_loss_margin":float(
                self.config.v73_reference_mission_loss_margin
            ),
            "reference_track_update_margin":float(
                self.config.v73_reference_track_update_margin
            ),
            "controller_state":asdict(controller_state),
            "gate_checkpoint_audit":self._v73_gate_checkpoint_audit,
            "truth_visible_to_controller":False,"exact_certificate_final_selector":True,
        }
        return proposed,metadata,{"state":controller_state,"certificates":tuple(certificates),
                                  "decision":decision,"index_by_id":candidate_index_by_id,
                                  "metadata":metadata}

    @staticmethod
    def _v73_certificate_rows(
        exact_rows: Sequence[CandidateScore],
        mission_admissible_rows: Sequence[CandidateScore],
    ) -> tuple[CandidateScore, ...]:
        """Carry one abstention certificate across mission admissibility gates.

        Acquisition and reacquisition rules intentionally reject a no-op because
        it cannot supply a rank-three position update.  The adaptive controller
        nevertheless needs the same schedule as a fail-closed rejection option.
        Non-empty actions therefore remain mission-admissible, while exactly one
        no-op is reintroduced only as the controller's abstention certificate.
        """

        noop_rows = tuple(row for row in exact_rows if not row.action_ids)
        if len(noop_rows) != 1:
            raise RuntimeError(
                "V73 exact shortlist requires exactly one explicit no-op"
            )
        noop = noop_rows[0]
        rows: list[CandidateScore] = [noop]
        seen = {int(noop.candidate_index)}
        for row in mission_admissible_rows:
            candidate_index = int(row.candidate_index)
            if not row.action_ids or candidate_index in seen:
                continue
            rows.append(row)
            seen.add(candidate_index)
        return tuple(rows)

    @staticmethod
    def _validate_controller_contract(
        world: PhysicalWorld,
        state: EpisodeState,
    ) -> None:
        """Reject any attempt to call the decision layer with simulator truth."""

        provenance = dict(world.external_channel_provenance)
        violations: list[str] = []
        if provenance.get("state_source") != "authoritative_DynamicFactorGraph_posterior":
            violations.append("controller_world_missing_authoritative_graph_source")
        if provenance.get("independent_controller_covariance") is not False:
            violations.append("independent_controller_covariance_not_forbidden")
        if provenance.get("oracle_target_or_nuisance_state_used") is not False:
            violations.append("oracle_target_or_nuisance_contract_not_false")
        if world.jammers:
            violations.append("hidden_jammer_present_in_controller_world")
        if state.generator_version != CONTROLLER_PROJECTION_VERSION:
            violations.append("controller_state_not_sanitized_projection")
        if state.events_active or state.events_started:
            violations.append("future_or_hidden_events_present_in_controller_state")
        if state.adversary.mode.value != "quiet":
            violations.append("hidden_adversary_mode_present_in_controller_state")
        # Field-level hidden-state gate.  The decision layer refuses to run on a
        # projection that still carries hidden target manoeuvre/scattering truth
        # or hidden adversary resources, independently of whether the current
        # scoring code happens to read those fields.
        target = state.primary_target
        for name, expected in HIDDEN_TARGET_FIELDS.items():
            if name == "acceleration_m_s2":
                continue
            if getattr(target, name) != expected:
                violations.append(f"hidden_target_field_present:{name}")
        for name, expected in HIDDEN_ADVERSARY_FIELDS.items():
            if getattr(state.adversary, name) != expected:
                violations.append(f"hidden_adversary_field_present:{name}")
        if dict(state.adversary.adaptation_memory):
            violations.append("hidden_adversary_field_present:adaptation_memory")
        if violations:
            raise RuntimeError(
                "decision controller contract failed closed: " + ", ".join(violations)
            )

    def _forecast_factors(
        self,
        world: PhysicalWorld,
        actions: Sequence[RadarAction],
        jammer: Jammer | None,
        *,
        epoch: int,
        candidate_index: int,
        scenario_index: int,
        arrival_epoch_by_action: Mapping[int, int] | None = None,
    ) -> tuple:
        if arrival_epoch_by_action is None:
            feasible, reasons, audit = self.availability.schedule_operational_feasibility(
                actions
            )
            if not feasible:
                raise RuntimeError(
                    "forecast factors require an operationally feasible schedule: "
                    + ", ".join(reasons)
                )
            raw_arrivals = audit.get("action_arrival_epoch", {})
            if not isinstance(raw_arrivals, Mapping):
                raise RuntimeError("schedule transport audit omitted action arrivals")
            arrival_epoch_by_action = {
                action.action_id: int(raw_arrivals[str(action.action_id)])
                for action in actions
            }
        x0 = self.belief.current_state_vector()
        factors = []
        for action in actions:
            action_factor = build_action_factor(world, action, jammer)
            factor = make_operational_action_factor(
                world,
                action,
                action_factor,
                epoch=epoch,
                arrival_epoch=int(arrival_epoch_by_action[action.action_id]),
                participants=self.availability.factor_participants(action),
                mode=FactorInformationMode.EXPECTED_OBSERVED,
                linearization_state_normalized=x0,
                measurement_residual=None,
                provenance_id=f"forecast:{epoch}:{candidate_index}:{scenario_index}:{action.action_id}",
                factor_id=f"forecast_factor:{epoch}:{candidate_index}:{scenario_index}:{action.action_id}",
                fidelity_level="R0_expected_operational_information",
                robust_kernel=RobustKernel.NONE,
                eigenvalue_floor=self.config.information_eigenvalue_floor,
                communication_snapshot_epoch=epoch,
                information_scale=self.availability.delivery_probability(
                    action.action_id
                ),
            )
            if factor is not None:
                factors.append(factor)
        return tuple(factors)

    @staticmethod
    def _positive_semidefinite_residual(
        matrix: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        tolerance = 1.0e-9 * max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) < -tolerance:
            raise RuntimeError(f"{name} is materially indefinite")
        return 0.5 * (
            (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            + ((eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T).T
        )

    def _prepare_candidate_scoring_context(
        self,
        world: PhysicalWorld,
        action_ids: Sequence[int],
    ) -> _CandidateScoringContext:
        """Precompute immutable physics primitives for one frozen label cycle.

        This context is deliberately narrower than the live decision path.  It
        contains no graph fork, posterior, transport arrival, candidate index,
        or random state; those remain candidate-local and are evaluated in the
        original serial order.
        """

        requested = frozenset(int(action_id) for action_id in action_ids)
        by_id = {int(action.action_id): action for action in world.actions}
        unknown = sorted(requested - set(by_id))
        if unknown:
            raise ValueError(f"candidate scoring context has unknown actions: {unknown}")
        actions = tuple(
            action
            for action in world.actions
            if int(action.action_id) in requested
        )
        scenarios = _joint_planning_scenarios(world, self.config)
        scenario_factors: list[Mapping[int, Any]] = []
        scenario_calibrations: list[
            Mapping[int, tuple[tuple[RadarAction, Any], ...]]
        ] = []
        scenario_probabilities: list[Mapping[int, Any]] = []
        for scenario in scenarios:
            factors: dict[int, Any] = {}
            calibrations: dict[int, tuple[tuple[RadarAction, Any], ...]] = {}
            probabilities: dict[int, Any] = {}
            for action in actions:
                action_id = int(action.action_id)
                factor = build_action_factor(
                    scenario.world,
                    action,
                    scenario.jammer,
                )
                factors[action_id] = factor
                if not action.kind.startswith("calibration"):
                    probabilities[action_id] = self.planning_probability.evaluate(
                        action,
                        factor,
                    )
                calibrations[action_id] = tuple(
                    (
                        calibration_action,
                        build_action_factor(
                            scenario.world,
                            calibration_action,
                            None,
                        ),
                    )
                    for calibration_action in automatic_free_calibration_actions(
                        scenario.world,
                        action,
                    )
                )
            scenario_factors.append(MappingProxyType(factors))
            scenario_calibrations.append(MappingProxyType(calibrations))
            scenario_probabilities.append(MappingProxyType(probabilities))
        structural_factors = MappingProxyType(
            {
                int(action.action_id): build_action_factor(world, action, None)
                for action in actions
            }
        )
        return _CandidateScoringContext(
            source_world=world,
            action_ids=requested,
            scenarios=scenarios,
            scenario_factor_by_action=tuple(scenario_factors),
            scenario_automatic_calibration_by_action=tuple(
                scenario_calibrations
            ),
            scenario_planning_probability_by_action=tuple(
                scenario_probabilities
            ),
            structural_factor_by_action=structural_factors,
        )

    def _action_outcome_alternatives(
        self,
        world: PhysicalWorld,
        action: RadarAction,
        jammer: Jammer | None,
        *,
        epoch: int,
        candidate_index: int,
        scenario_index: int,
        arrival_epoch: int,
        action_factor: Any | None = None,
        automatic_calibration_factors: Sequence[
            tuple[RadarAction, Any]
        ] | None = None,
        planning_probability_record: Any | None = None,
    ) -> tuple[tuple[float, tuple[Any, ...], str], ...]:
        r"""Enumerate packet/detection outcomes for one selected action.

        For a sensing action there are three mutually exclusive outcomes:
        packet loss, delivered miss, and delivered detection.  Simultaneous free
        calibration rows are present in both delivered outcomes, while the target
        factor appears only after a detection.  This avoids both replacing a
        finite mixture by one average Fisher matrix and applying ``P_D`` twice.
        """

        if action_factor is None:
            action_factor = build_action_factor(world, action, jammer)
        delivery = float(
            np.clip(self.availability.delivery_probability(action.action_id), 0.0, 1.0)
        )
        participants = self.availability.factor_participants(action)
        x0 = self.belief.current_state_vector()

        def make_factor(
            source_action: RadarAction,
            source_factor,
            *,
            mode: FactorInformationMode,
            suffix: str,
        ):
            return make_operational_action_factor(
                world,
                source_action,
                source_factor,
                epoch=epoch,
                arrival_epoch=arrival_epoch,
                participants=participants,
                mode=mode,
                linearization_state_normalized=x0,
                measurement_residual=None,
                provenance_id=(
                    f"forecast_outcome:{epoch}:{candidate_index}:{scenario_index}:"
                    f"{action.action_id}:{suffix}"
                ),
                factor_id=(
                    f"forecast_outcome_factor:{epoch}:{candidate_index}:"
                    f"{scenario_index}:{action.action_id}:{suffix}"
                ),
                fidelity_level="R0_conditional_finite_outcome_information",
                robust_kernel=RobustKernel.NONE,
                eigenvalue_floor=self.config.information_eigenvalue_floor,
                communication_snapshot_epoch=epoch,
                information_scale=1.0,
            )

        if action.kind.startswith("calibration"):
            calibration = make_factor(
                action,
                action_factor,
                mode=FactorInformationMode.CONDITIONAL_DETECTED,
                suffix="delivered_calibration",
            )
            alternatives = []
            if delivery < 1.0:
                alternatives.append((1.0 - delivery, (), "packet_loss"))
            if delivery > 0.0:
                alternatives.append(
                    (
                        delivery,
                        tuple(item for item in (calibration,) if item is not None),
                        "delivered_calibration",
                    )
                )
            return tuple(alternatives or [(1.0, (), "zero_delivery")])

        calibration_factors = []
        calibration_information = np.zeros_like(
            action_factor.observed_information_normalized
        )
        calibration_primitives = automatic_calibration_factors
        if calibration_primitives is None:
            calibration_primitives = tuple(
                (
                    calibration_action,
                    build_action_factor(world, calibration_action, None),
                )
                for calibration_action in automatic_free_calibration_actions(
                    world,
                    action,
                )
            )
        for calibration_index, (
            calibration_action,
            calibration_action_factor,
        ) in enumerate(calibration_primitives):
            calibration_information += (
                calibration_action_factor.observed_information_normalized
            )
            calibration_factor = make_factor(
                calibration_action,
                calibration_action_factor,
                mode=FactorInformationMode.CONDITIONAL_DETECTED,
                suffix=f"automatic_calibration_{calibration_index}",
            )
            if calibration_factor is not None:
                calibration_factors.append(calibration_factor)
        conditional = action_factor.conditional_detected_information_normalized
        if conditional is None:
            raise RuntimeError("sensing ActionFactor omitted conditional information")
        target_conditional = self._positive_semidefinite_residual(
            np.asarray(conditional, dtype=float) - calibration_information,
            name="conditional target information after calibration separation",
        )
        target_action_factor = replace(
            action_factor,
            conditional_detected_information_normalized=target_conditional,
            metadata={
                **dict(action_factor.metadata),
                "automatic_calibration_separated_for_finite_outcome_mixture": True,
            },
        )
        target_factor = make_factor(
            action,
            target_action_factor,
            mode=FactorInformationMode.CONDITIONAL_DETECTED,
            suffix="delivered_detection_target",
        )
        probability_record = planning_probability_record
        if probability_record is None:
            probability_record = self.planning_probability.evaluate(
                action, action_factor
            )
        detected_probability = probability_record.planning_success_probability
        alternatives: list[tuple[float, tuple[Any, ...], str]] = []
        if delivery < 1.0:
            alternatives.append((1.0 - delivery, (), "packet_loss"))
        delivered_miss = delivery * (1.0 - detected_probability)
        if delivered_miss > 0.0:
            alternatives.append(
                (delivered_miss, tuple(calibration_factors), "delivered_miss")
            )
        delivered_detection = delivery * detected_probability
        if delivered_detection > 0.0:
            alternatives.append(
                (
                    delivered_detection,
                    tuple(
                        [
                            *calibration_factors,
                            *(item for item in (target_factor,) if item is not None),
                        ]
                    ),
                    "delivered_detection",
                )
            )
        total = sum(item[0] for item in alternatives)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
            raise RuntimeError("action outcome probabilities do not sum to one")
        return tuple(alternatives)

    def _finite_outcome_schedule_moments(
        self,
        scenario: _JointPlanningScenario,
        actions: Sequence[RadarAction],
        *,
        state: EpisodeState,
        candidate_index: int,
        scenario_index: int,
        evaluation_epoch: int,
        arrival_epoch_by_action: Mapping[int, int],
        communication_template,
        prior_position_covariance_m2: np.ndarray,
        prior_velocity_covariance_m2_s2: np.ndarray,
        precomputed_factor_by_action: Mapping[int, Any] | None = None,
        precomputed_automatic_calibration_by_action: Mapping[
            int, Sequence[tuple[RadarAction, Any]]
        ] | None = None,
        precomputed_planning_probability_by_action: Mapping[int, Any] | None = None,
        reuse_identical_counterfactuals: bool = False,
    ) -> dict[str, Any]:
        alternatives = [
            self._action_outcome_alternatives(
                scenario.world,
                action,
                scenario.jammer,
                epoch=state.epoch,
                candidate_index=candidate_index,
                scenario_index=scenario_index,
                arrival_epoch=int(arrival_epoch_by_action[action.action_id]),
                action_factor=(
                    precomputed_factor_by_action.get(int(action.action_id))
                    if precomputed_factor_by_action is not None
                    else None
                ),
                automatic_calibration_factors=(
                    precomputed_automatic_calibration_by_action.get(
                        int(action.action_id)
                    )
                    if precomputed_automatic_calibration_by_action is not None
                    else None
                ),
                planning_probability_record=(
                    precomputed_planning_probability_by_action.get(
                        int(action.action_id)
                    )
                    if precomputed_planning_probability_by_action is not None
                    else None
                ),
            )
            for action in actions
        ]
        outcome_product = product(*alternatives) if alternatives else [()]
        expected_position = np.zeros_like(prior_position_covariance_m2, dtype=float)
        expected_velocity = np.zeros_like(prior_velocity_covariance_m2_s2, dtype=float)
        expected_position_gain = 0.0
        expected_velocity_gain = 0.0
        probability_total = 0.0
        outcome_count = 0
        counterfactual_by_factor_identity: dict[tuple[int, ...], Any] = {}
        for outcome in outcome_product:
            probability = float(np.prod([item[0] for item in outcome])) if outcome else 1.0
            if probability <= 0.0:
                continue
            factors = tuple(
                factor
                for _, factor_group, _ in outcome
                for factor in factor_group
            )
            factor_identity = tuple(id(factor) for factor in factors)
            result = (
                counterfactual_by_factor_identity.get(factor_identity)
                if reuse_identical_counterfactuals
                else None
            )
            if result is None:
                result = self.belief.counterfactual_at_epoch(
                    factors,
                    evaluation_epoch=evaluation_epoch,
                    reference_world=scenario.world,
                    communication_template=communication_template,
                )
                if reuse_identical_counterfactuals:
                    counterfactual_by_factor_identity[factor_identity] = result
            if result.status != "passed":
                raise RuntimeError("finite-outcome counterfactual graph failed closed")
            expected_position += probability * result.position_covariance_m2
            expected_velocity += probability * result.velocity_covariance_m2_s2
            expected_position_gain += probability * information_gain_nats(
                result.position_covariance_m2,
                prior_position_covariance_m2,
            )
            expected_velocity_gain += probability * information_gain_nats(
                result.velocity_covariance_m2_s2,
                prior_velocity_covariance_m2_s2,
            )
            probability_total += probability
            outcome_count += 1
        if not math.isclose(probability_total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise RuntimeError("joint schedule outcome probabilities do not sum to one")
        correct_update_probability_by_action: dict[int, float] = {}
        verification_usable_probability_by_action: dict[int, float] = {}
        scenario_factor_by_action: dict[int, Any] = {}
        planning_probability_by_action: dict[str, Any] = {}
        for action in actions:
            action_id = int(action.action_id)
            factor = (
                precomputed_factor_by_action.get(action_id)
                if precomputed_factor_by_action is not None
                else None
            )
            if factor is None:
                factor = build_action_factor(
                    scenario.world,
                    action,
                    scenario.jammer,
                )
            scenario_factor_by_action[action_id] = factor
            if action.kind.startswith("calibration"):
                correct_update_probability_by_action[action_id] = 0.0
                verification_usable_probability_by_action[action_id] = float(
                    np.clip(
                        self.availability.delivery_probability(action_id),
                        0.0,
                        1.0,
                    )
                )
                continue
            probability_record = (
                precomputed_planning_probability_by_action.get(action_id)
                if precomputed_planning_probability_by_action is not None
                else None
            )
            if probability_record is None:
                probability_record = self.planning_probability.evaluate(action, factor)
            planning_probability_by_action[str(action_id)] = (
                probability_record.to_mapping()
            )
            correct_update_probability_by_action[action_id] = float(
                np.clip(
                    self.availability.delivery_probability(action_id)
                    * probability_record.planning_success_probability
                    * factor.link_budget.correct_association_probability,
                    0.0,
                    1.0,
                )
            )
            verification_usable_probability_by_action[action_id] = (
                correct_update_probability_by_action[action_id]
            )
        correct_update_probability = _schedule_update_probability(
            correct_update_probability_by_action,
            tuple(action.action_id for action in actions),
        )
        full_rank_position_update_probability = (
            _full_rank_position_update_probability(
                scenario.world,
                actions,
                correct_update_probability_by_action,
                factor_by_action=scenario_factor_by_action,
            )
        )
        if self.config.innovation_verification_require_position_separating_rank:
            position_separating_update_probability = (
                _position_separating_update_probability(
                    scenario.world,
                    actions,
                    verification_usable_probability_by_action,
                    factor_by_action=scenario_factor_by_action,
                )
            )
            differential_clock_verification_probability = (
                _differential_clock_verification_probability(
                    scenario.world,
                    actions,
                    verification_usable_probability_by_action,
                    factor_by_action=scenario_factor_by_action,
                )
            )
        else:
            position_separating_update_probability = (
                full_rank_position_update_probability
            )
            differential_clock_verification_probability = 0.0
        return {
            "position_covariance_m2": 0.5 * (expected_position + expected_position.T),
            "velocity_covariance_m2_s2": 0.5 * (expected_velocity + expected_velocity.T),
            "position_information_gain_nats": float(expected_position_gain),
            "velocity_information_gain_nats": float(expected_velocity_gain),
            "correct_update_probability": float(correct_update_probability),
            "full_rank_position_update_probability": float(
                full_rank_position_update_probability
            ),
            "position_separating_update_probability": float(
                position_separating_update_probability
            ),
            "differential_clock_verification_probability": float(
                differential_clock_verification_probability
            ),
            "outcome_count": int(outcome_count),
            "outcome_probability_sum": float(probability_total),
            "planning_probability_by_action": planning_probability_by_action,
            "finite_outcome_semantics": (
                "packet_delivery_x_detection_with_association_inside_conditional_factor"
            ),
        }

    def _verification_scattering_probability_lower_bound(
        self,
        world: PhysicalWorld,
        actions: Sequence[RadarAction],
    ) -> dict[str, Any]:
        """Evaluate the opt-in verification event at a certified RCS floor."""

        rcs_floor_m2, retained_mass, support_audit = (
            _registered_scattering_support_lower_bound(
                world,
                self.config.innovation_verification_scattering_outage_probability,
                texture_group_count=_scattering_texture_group_count(
                    world,
                    actions,
                ),
            )
        )
        lower_world = replace(
            world,
            target=replace(
                world.target,
                mean_rcs_m2=float(rcs_floor_m2),
                swerling_case="0",
            ),
            external_channel_provenance={
                **dict(world.external_channel_provenance),
                "verification_scattering_support_lower_bound": True,
                "verification_scattering_rcs_floor_m2": float(rcs_floor_m2),
                "verification_scattering_truth_realization_read": False,
            },
        )
        lower_world.validate()
        factors: dict[int, Any] = {}
        usable_probability_by_action: dict[int, float] = {}
        action_receipts: dict[str, Any] = {}
        for action in actions:
            factor = build_action_factor(lower_world, action, None)
            factors[int(action.action_id)] = factor
            delivery_probability = float(
                np.clip(
                    self.availability.delivery_probability(action.action_id),
                    0.0,
                    1.0,
                )
            )
            if action.kind.startswith("calibration"):
                usable_probability = delivery_probability
                action_receipts[str(action.action_id)] = {
                    "action_kind": str(action.kind),
                    "transport_delivery_probability": delivery_probability,
                    "conditional_usable_probability_at_rcs_floor": (
                        usable_probability
                    ),
                    "calibration_detection_probability_assumed": 1.0,
                }
            else:
                probability_record = self.planning_probability.evaluate(
                    action,
                    factor,
                )
                usable_probability = float(
                    np.clip(
                        delivery_probability
                        * probability_record.planning_success_probability
                        * factor.link_budget.correct_association_probability,
                        0.0,
                        1.0,
                    )
                )
                action_receipts[str(action.action_id)] = {
                    "action_kind": str(action.kind),
                    "transport_delivery_probability": delivery_probability,
                    "planning_success_probability_at_rcs_floor": float(
                        probability_record.planning_success_probability
                    ),
                    "correct_association_probability_at_rcs_floor": float(
                        factor.link_budget.correct_association_probability
                    ),
                    "conditional_usable_probability_at_rcs_floor": (
                        usable_probability
                    ),
                    "rcs_after_declared_aspect_model_m2": float(
                        factor.link_budget.rcs_realization_m2
                    ),
                }
            usable_probability_by_action[int(action.action_id)] = (
                usable_probability
            )
        conditional_position = _position_separating_update_probability(
            lower_world,
            actions,
            usable_probability_by_action,
            factor_by_action=factors,
        )
        conditional_clock = _differential_clock_verification_probability(
            lower_world,
            actions,
            usable_probability_by_action,
            factor_by_action=factors,
        )
        return {
            "position_separating_update_probability_lcb": float(
                _common_support_event_probability_lower_bound(
                    conditional_position,
                    retained_mass,
                )
            ),
            "differential_clock_verification_probability_lcb": float(
                _common_support_event_probability_lower_bound(
                    conditional_clock,
                    retained_mass,
                )
            ),
            "conditional_position_separating_probability_at_rcs_floor": float(
                conditional_position
            ),
            "conditional_differential_clock_probability_at_rcs_floor": float(
                conditional_clock
            ),
            "support_lower_bound": support_audit,
            "action_probability_receipts": action_receipts,
            "nominal_no_jammer_condition": True,
            "received_target_power_monotonicity_required": True,
            "truth_or_realized_scattering_read": False,
        }

    def _score_candidate(
        self,
        world: PhysicalWorld,
        state: EpisodeState,
        library: CandidateLibrary,
        candidate_index: int,
        proposal_score: float,
        evaluation_epoch: int,
        baseline_position_covariance_m2: np.ndarray,
        baseline_velocity_covariance_m2_s2: np.ndarray,
        communication_template,
        transport_audit: Mapping[str, Any],
        track_phase: Mapping[str, Any],
        *,
        scoring_context: _CandidateScoringContext | None = None,
    ) -> CandidateScore:
        action_ids = library.schedules[candidate_index]
        by_id = {action.action_id: action for action in world.actions}
        actions = tuple(by_id[action_id] for action_id in action_ids)
        raw_arrivals = transport_audit.get("action_arrival_epoch", {})
        if not isinstance(raw_arrivals, Mapping):
            raise RuntimeError("candidate transport audit omitted action arrivals")
        arrival_epoch_by_action = {
            action.action_id: int(raw_arrivals[str(action.action_id)])
            for action in actions
        }
        prior_position = baseline_position_covariance_m2
        prior_velocity = baseline_velocity_covariance_m2_s2
        if scoring_context is None:
            scenarios = _joint_planning_scenarios(world, self.config)
        else:
            if scoring_context.source_world is not world:
                raise ValueError("candidate scoring context belongs to another world")
            missing = sorted(set(int(value) for value in action_ids) - scoring_context.action_ids)
            if missing:
                raise ValueError(
                    f"candidate scoring context omitted action ids: {missing}"
                )
            scenarios = scoring_context.scenarios
        position_gains: list[float] = []
        velocity_gains: list[float] = []
        position_covariances: list[np.ndarray] = []
        velocity_covariances: list[np.ndarray] = []
        scenario_ids: list[str] = []
        scenario_weights: list[float] = []
        correct_update_probabilities: list[float] = []
        full_rank_position_update_probabilities: list[float] = []
        position_separating_update_probabilities: list[float] = []
        differential_clock_verification_probabilities: list[float] = []
        outcome_counts: list[int] = []
        scenario_planning_probabilities: list[dict[str, Any]] = []
        for scenario_index, scenario in enumerate(scenarios):
            precomputed_factors = (
                scoring_context.scenario_factor_by_action[scenario_index]
                if scoring_context is not None
                else None
            )
            precomputed_calibrations = (
                scoring_context.scenario_automatic_calibration_by_action[
                    scenario_index
                ]
                if scoring_context is not None
                else None
            )
            precomputed_probabilities = (
                scoring_context.scenario_planning_probability_by_action[
                    scenario_index
                ]
                if scoring_context is not None
                else None
            )
            moments = self._finite_outcome_schedule_moments(
                scenario,
                actions,
                state=state,
                candidate_index=candidate_index,
                scenario_index=scenario_index,
                evaluation_epoch=evaluation_epoch,
                arrival_epoch_by_action=arrival_epoch_by_action,
                communication_template=communication_template,
                prior_position_covariance_m2=prior_position,
                prior_velocity_covariance_m2_s2=prior_velocity,
                precomputed_factor_by_action=precomputed_factors,
                precomputed_automatic_calibration_by_action=(
                    precomputed_calibrations
                ),
                precomputed_planning_probability_by_action=(
                    precomputed_probabilities
                ),
                reuse_identical_counterfactuals=scoring_context is not None,
            )
            position_covariances.append(moments["position_covariance_m2"])
            velocity_covariances.append(moments["velocity_covariance_m2_s2"])
            position_gains.append(moments["position_information_gain_nats"])
            velocity_gains.append(moments["velocity_information_gain_nats"])
            correct_update_probabilities.append(
                moments["correct_update_probability"]
            )
            full_rank_position_update_probabilities.append(
                moments["full_rank_position_update_probability"]
            )
            position_separating_update_probabilities.append(
                moments["position_separating_update_probability"]
            )
            differential_clock_verification_probabilities.append(
                moments["differential_clock_verification_probability"]
            )
            outcome_counts.append(moments["outcome_count"])
            scenario_planning_probabilities.append(
                dict(moments["planning_probability_by_action"])
            )
            scenario_ids.append(scenario.scenario_id)
            scenario_weights.append(float(scenario.probability_weight))
        weights = np.asarray(scenario_weights, dtype=float)
        weights /= float(np.sum(weights))
        nominal_mask = np.asarray(
            [scenario.jammer is None for scenario in scenarios],
            dtype=bool,
        )
        nominal_weights = weights[nominal_mask]
        nominal_weights /= float(np.sum(nominal_weights))
        nominal_position = sum(
            weight * covariance
            for weight, covariance in zip(
                nominal_weights,
                [
                    covariance
                    for covariance, keep in zip(position_covariances, nominal_mask)
                    if keep
                ],
            )
        )
        nominal_velocity = sum(
            weight * covariance
            for weight, covariance in zip(
                nominal_weights,
                [
                    covariance
                    for covariance, keep in zip(velocity_covariances, nominal_mask)
                    if keep
                ],
            )
        )
        lower_position = _weighted_lower_tail_mean(
            position_gains,
            scenario_weights,
            self.config.lower_tail_probability,
        )
        lower_velocity = _weighted_lower_tail_mean(
            velocity_gains,
            scenario_weights,
            self.config.lower_tail_probability,
        )
        mean_position = float(np.dot(weights, np.asarray(position_gains, dtype=float)))
        mean_velocity = float(np.dot(weights, np.asarray(velocity_gains, dtype=float)))
        (
            expected_correct_update_probability,
            stress_ensemble_mean_correct_update_probability,
        ) = _conditioned_probability_and_stress_mean(
            correct_update_probabilities,
            scenario_weights,
            nominal_mask,
        )
        (
            expected_full_rank_position_update_probability,
            stress_ensemble_mean_full_rank_position_update_probability,
        ) = _conditioned_probability_and_stress_mean(
            full_rank_position_update_probabilities,
            scenario_weights,
            nominal_mask,
        )
        (
            expected_position_separating_update_probability,
            stress_ensemble_mean_position_separating_update_probability,
        ) = _conditioned_probability_and_stress_mean(
            position_separating_update_probabilities,
            scenario_weights,
            nominal_mask,
        )
        (
            expected_differential_clock_verification_probability,
            stress_ensemble_mean_differential_clock_verification_probability,
        ) = _conditioned_probability_and_stress_mean(
            differential_clock_verification_probabilities,
            scenario_weights,
            nominal_mask,
        )
        tail_weight = float(np.clip(self.mission.tail_risk_weight, 0.0, 1.0))
        robust_position = (1.0 - tail_weight) * mean_position + tail_weight * lower_position
        robust_velocity = (1.0 - tail_weight) * mean_velocity + tail_weight * lower_velocity
        prior_d_radius = d_optimal_radius_m(prior_position)
        nominal_d_radius = d_optimal_radius_m(nominal_position)
        d_improvement_fraction = (
            0.0
            if not math.isfinite(prior_d_radius) or prior_d_radius <= 0.0
            else max(0.0, prior_d_radius - nominal_d_radius) / prior_d_radius
        )
        degradation = max(
            (
                bures_wasserstein_distance_m(nominal_position, covariance)
                for covariance in position_covariances[1:]
            ),
            default=0.0,
        )
        usage = schedule_resource_usage(world, actions)
        (
            sensing_action_count,
            position_geometry_singular_values,
            position_geometry_rank,
            geometry_tolerance,
        ) = _structural_position_observation_geometry(
            world,
            actions,
            factor_by_action=(
                scoring_context.structural_factor_by_action
                if scoring_context is not None
                else None
            ),
        )
        (
            _,
            position_separating_action_count,
            position_separating_singular_values,
            position_separating_rank,
            position_separating_tolerance,
        ) = _position_separating_observation_geometry(
            world,
            actions,
            factor_by_action=(
                scoring_context.structural_factor_by_action
                if scoring_context is not None
                else None
            ),
        )
        erasure_robust_position_frame = (
            _erasure_robust_position_frame_certificate(
                world,
                actions,
                maximum_erasures=(
                    self.config.erasure_robust_position_geometry_max_erasures
                ),
                factor_by_action=(
                    scoring_context.structural_factor_by_action
                    if scoring_context is not None
                    else None
                ),
            )
        )
        erasure_robust_ratio_threshold = 1.0 / float(
            self.config.erasure_robust_position_geometry_max_jacobian_condition_number
        )
        absolute_range_anchor_records: list[dict[str, Any]] = []
        for action in actions:
            if action.kind != "active_monostatic":
                continue
            structural_factor = (
                scoring_context.structural_factor_by_action.get(
                    int(action.action_id)
                )
                if scoring_context is not None
                else None
            )
            if structural_factor is None:
                structural_factor = build_action_factor(world, action, None)
            measurement_names = tuple(str(name) for name in structural_factor.measurement_names)
            if "monostatic_delay_s" in measurement_names:
                absolute_range_anchor_records.append(
                    {
                        "action_id": int(action.action_id),
                        "action_kind": str(action.kind),
                        "measurement_name": "monostatic_delay_s",
                        "two_way_time_of_flight_model": True,
                    }
                )
        differential_clock_verification = (
            _differential_clock_verification_certificate(
                world,
                actions,
                require_position_separating_rank=(
                    self.config.innovation_verification_require_position_separating_rank
                ),
                factor_by_action=(
                    scoring_context.structural_factor_by_action
                    if scoring_context is not None
                    else None
                ),
            )
        )
        verification_scattering_lower_bound = (
            self._verification_scattering_probability_lower_bound(
                world,
                actions,
            )
            if self.config.innovation_verification_require_position_separating_rank
            else None
        )
        position_separating_probability_lcb = float(
            verification_scattering_lower_bound[
                "position_separating_update_probability_lcb"
            ]
            if verification_scattering_lower_bound is not None
            else expected_position_separating_update_probability
        )
        differential_clock_verification_probability_lcb = float(
            verification_scattering_lower_bound[
                "differential_clock_verification_probability_lcb"
            ]
            if verification_scattering_lower_bound is not None
            else expected_differential_clock_verification_probability
        )
        switching_fraction = min(
            1.0,
            len(set(action_ids).symmetric_difference(state.resources.last_action_ids))
            / max(1, world.max_schedule_size),
        )
        switching = self.mission.switching_cost_weight * switching_fraction
        bandwidth_time = float(
            sum(
                world.waveform(action.waveform_id).bandwidth_hz
                * world.waveform(action.waveform_id).dwell_s
                for action in actions
            )
        )

        def fraction(numerator: float, denominator: float) -> float:
            if numerator <= 0.0:
                return 0.0
            if denominator <= 0.0:
                return math.inf
            return float(numerator / denominator)

        resource_consumption_fractions = {
            "energy_above_reserve": fraction(
                usage.energy_j,
                state.resources.energy_remaining_j
                - state.resources.reserve_energy_j,
            ),
            "emission_above_reserve": fraction(
                usage.emission_exposure_j,
                state.resources.emission_remaining_j
                - state.resources.reserve_emission_j,
            ),
            "bandwidth_time": fraction(
                bandwidth_time,
                state.resources.bandwidth_time_remaining_hz_s,
            ),
            "dwell": fraction(
                usage.dwell_s,
                state.resources.dwell_remaining_s,
            ),
            "communication": fraction(
                usage.communication_mb,
                state.resources.communication_remaining_mb,
            ),
        }
        future_resource_consumption = max(
            resource_consumption_fractions.values(), default=0.0
        )
        # Hard feasibility is enforced by CampaignSafetyShield.  The objective
        # therefore uses an opportunity cost, not a second copy of the hard
        # budget.  The maximum fraction identifies the resource bottleneck and
        # avoids triple-counting one action merely because it consumes energy,
        # emission and communication together.
        resource_cost = (
            self.mission.future_resource_weight
            * future_resource_consumption
        )
        transport_penalty = degradation / max(prior_d_radius, 1e-9)
        required_update_probability = float(
            track_phase["required_correct_update_probability"]
        )
        native_track_maintaining_update_probability = (
            position_separating_probability_lcb
            if (
                self.config.innovation_verification_require_position_separating_rank
                and str(track_phase["phase"]) in {"acquisition", "reacquisition"}
            )
            else (
                expected_full_rank_position_update_probability
                if str(track_phase["phase"]) in {"acquisition", "reacquisition"}
                else expected_correct_update_probability
            )
        )
        update_probability_calibration = (
            self.schedule_update_probability_calibration.evaluate(
                str(track_phase["phase"]),
                native_track_maintaining_update_probability,
            )
        )
        track_maintaining_update_probability = float(
            update_probability_calibration.objective_probability
        )
        admissibility_track_update_probability = float(
            update_probability_calibration.admissibility_probability
        )
        update_probability_shortfall = max(
            0.0,
            required_update_probability - track_maintaining_update_probability,
        )
        track_continuity_penalty = (
            self.mission.track_update_shortfall_weight
            * update_probability_shortfall
        )
        exact_score = (
            self.mission.position_weight * robust_position
            + self.mission.velocity_weight * robust_velocity
            + 0.10 * d_improvement_fraction
            - 0.10 * self.mission.tail_risk_weight * transport_penalty
            - resource_cost
            - switching
            - track_continuity_penalty
        )
        multi_target_information_gain = (
            0.0
            if self.multi_target_engine is None
            else float(
                self.multi_target_engine.expected_information_gain(
                    world,
                    action_ids,
                    planning_probability=self.planning_probability,
                )
            )
        )
        exact_score += (
            self.config.multi_target_information_weight
            * multi_target_information_gain
        )
        nonlocal_task_risk = None
        if self.config.nonlocal_task_risk_enabled:
            if self.multi_target_engine is None:
                raise RuntimeError(
                    "nonlocal task-risk reranking requires the controller JPDA posterior"
                )
            nonlocal_task_risk = posterior_predictive_schedule_risk(
                self.multi_target_engine,
                world,
                action_ids,
                epoch=state.epoch,
                delivery_probability={
                    action.action_id: self.availability.delivery_probability(
                        action.action_id
                    )
                    for action in actions
                },
                scenarios=tuple(
                    dict.fromkeys(scenario.jammer for scenario in scenarios)
                ),
                replicates=self.config.nonlocal_task_risk_replicates,
                seed=self.config.seed,
                cutoff_m=self.config.nonlocal_task_risk_cutoff_m,
                alpha=self.config.nonlocal_task_risk_alpha,
                tail_probability=self.config.lower_tail_probability,
                tail_weight=tail_weight,
                confidence_z=self.config.nonlocal_task_risk_confidence_z,
                planning_probability=self.planning_probability,
            )
            if nonlocal_task_risk.get("status") != "passed":
                raise RuntimeError("nonlocal task-risk evaluator failed closed")
            exact_score += self.config.nonlocal_task_risk_weight * float(
                nonlocal_task_risk["risk_reduction_lcb_fraction"]
            )
        return CandidateScore(
            candidate_index=candidate_index,
            action_ids=action_ids,
            proposal_score=float(proposal_score),
            exact_score=float(exact_score),
            position_information_gain_nats=mean_position,
            velocity_information_gain_nats=mean_velocity,
            lower_tail_position_gain_nats=float(lower_position),
            lower_tail_velocity_gain_nats=float(lower_velocity),
            nominal_d_radius_improvement_fraction=float(d_improvement_fraction),
            worst_operational_bures_degradation_m=float(degradation),
            normalized_resource_cost=float(resource_cost),
            switching_cost=float(switching),
            scenario_position_gains_nats=tuple(float(value) for value in position_gains),
            scenario_velocity_gains_nats=tuple(float(value) for value in velocity_gains),
            scenario_ids=tuple(scenario_ids),
            post_position_covariance_m2=nominal_position,
            post_velocity_covariance_m2_s2=nominal_velocity,
            status="passed",
            metadata={
                "authoritative_belief_backend": "DynamicFactorGraph",
                "exact_rerank_bypassed": False,
                "V69_operational_units": {
                    "information_gain": "nat_per_axis",
                    "Bures_degradation": "m",
                    "D_radius": "m",
                },
                "factor_detection_association_double_counted": False,
                "finite_detection_transport_outcomes_marginalized": True,
                "expected_fisher_substitution_used_for_detection_mixture": False,
                "scenario_probability_weights": [
                    float(value) for value in scenario_weights
                ],
                "scenario_weight_semantics": (
                    "equal_jammer_stress_design_x_registered_target_scattering_"
                    "quadrature_not_jammer_occurrence_probability"
                ),
                "scenario_probability_weights_are_occurrence_calibrated": False,
                "scenario_rcs_m2": [float(item.rcs_m2) for item in scenarios],
                "scenario_outcome_counts": [int(value) for value in outcome_counts],
                "planning_detection_probability": self.planning_probability.receipt,
                "scenario_planning_probability_by_action": (
                    scenario_planning_probabilities
                ),
                "scattering_prior_marginalization": bool(
                    self.config.scattering_prior_marginalization
                ),
                "scattering_quadrature_order": int(
                    self.config.scattering_quadrature_order
                ),
                "scattering_source": (
                    "registered_target_class_prior_not_realized_truth"
                ),
                "expected_correct_track_update_probability": float(
                    expected_correct_update_probability
                ),
                "expected_full_rank_position_update_probability": float(
                    expected_full_rank_position_update_probability
                ),
                "expected_position_separating_update_probability": float(
                    expected_position_separating_update_probability
                ),
                "position_separating_update_probability_lcb": float(
                    position_separating_probability_lcb
                ),
                "expected_differential_clock_verification_probability": float(
                    expected_differential_clock_verification_probability
                ),
                "differential_clock_verification_probability_lcb": float(
                    differential_clock_verification_probability_lcb
                ),
                "verification_scattering_probability_lower_bound": (
                    verification_scattering_lower_bound
                ),
                "expected_track_maintaining_update_probability": float(
                    native_track_maintaining_update_probability
                ),
                "calibrated_track_update_probability_point": float(
                    track_maintaining_update_probability
                ),
                "calibrated_track_update_probability_lcb_95_one_sided": float(
                    admissibility_track_update_probability
                ),
                "schedule_update_probability_calibration": (
                    update_probability_calibration.receipt()
                ),
                "chance_constraint_probability_conditioning": (
                    CHANCE_CONSTRAINT_PROBABILITY_CONDITIONING
                ),
                "chance_constraint_probability_uses_equal_jammer_stress_weights": (
                    False
                ),
                "chance_constraint_condition_scenario_ids": [
                    scenario_id
                    for scenario_id, keep in zip(scenario_ids, nominal_mask)
                    if keep
                ],
                "stress_ensemble_mean_correct_track_update_probability": float(
                    stress_ensemble_mean_correct_update_probability
                ),
                "stress_ensemble_mean_full_rank_position_update_probability": float(
                    stress_ensemble_mean_full_rank_position_update_probability
                ),
                "stress_ensemble_mean_position_separating_update_probability": float(
                    stress_ensemble_mean_position_separating_update_probability
                ),
                "stress_ensemble_mean_differential_clock_verification_probability": float(
                    stress_ensemble_mean_differential_clock_verification_probability
                ),
                "stress_ensemble_probability_is_calibrated_occurrence_probability": (
                    False
                ),
                "stress_ensemble_probability_role": (
                    STRESS_ENSEMBLE_PROBABILITY_ROLE
                ),
                "track_maintaining_update_event": (
                    "correctly_received_sensing_subset_with_velocity_independent_"
                    "position_rank3"
                    if (
                        self.config.innovation_verification_require_position_separating_rank
                        and str(track_phase["phase"])
                        in {"acquisition", "reacquisition"}
                    )
                    else (
                        "at_least_two_correctly_received_sensing_actions_with_"
                        "structural_position_rank3"
                        if str(track_phase["phase"])
                        in {"acquisition", "reacquisition"}
                        else "at_least_one_correctly_received_associated_update"
                    )
                ),
                "scenario_correct_track_update_probabilities": [
                    float(value) for value in correct_update_probabilities
                ],
                "scenario_full_rank_position_update_probabilities": [
                    float(value)
                    for value in full_rank_position_update_probabilities
                ],
                "scenario_position_separating_update_probabilities": [
                    float(value)
                    for value in position_separating_update_probabilities
                ],
                "scenario_differential_clock_verification_probabilities": [
                    float(value)
                    for value in differential_clock_verification_probabilities
                ],
                "track_phase_contract": dict(track_phase),
                "required_correct_track_update_probability": float(
                    required_update_probability
                ),
                "correct_track_update_probability_shortfall": float(
                    update_probability_shortfall
                ),
                "track_continuity_penalty": float(track_continuity_penalty),
                "track_continuity_penalty_definition": (
                    "weight*max(0,required_probability-calibrated_point_probability)"
                ),
                "sensing_action_count": int(sensing_action_count),
                "position_observation_geometry_rank": int(
                    position_geometry_rank
                ),
                "position_observation_geometry_singular_values": [
                    float(value) for value in position_geometry_singular_values
                ],
                "position_observation_geometry_rank_tolerance": float(
                    geometry_tolerance
                ),
                "position_observation_geometry_uses_truth": False,
                "position_observation_geometry_is_structural": True,
                "position_separating_geometry_required": bool(
                    self.config.innovation_verification_require_position_separating_rank
                ),
                "position_separating_action_count": int(
                    position_separating_action_count
                ),
                "position_separating_geometry_rank": int(
                    position_separating_rank
                ),
                "position_separating_geometry_singular_values": [
                    float(value) for value in position_separating_singular_values
                ],
                "position_separating_geometry_rank_tolerance": float(
                    position_separating_tolerance
                ),
                "position_separating_geometry_excludes_target_velocity_sensitive_rows": True,
                "position_separating_geometry_is_necessary_not_global_sufficiency": True,
                "erasure_robust_position_frame_certificate": dict(
                    erasure_robust_position_frame
                ),
                "erasure_robust_position_geometry_enabled": bool(
                    self.config.erasure_robust_position_geometry_enabled
                ),
                "erasure_robust_position_geometry_max_jacobian_condition_number": float(
                    self.config.erasure_robust_position_geometry_max_jacobian_condition_number
                ),
                "erasure_robust_position_geometry_minimum_singular_value_ratio": float(
                    erasure_robust_ratio_threshold
                ),
                "erasure_robust_position_geometry_threshold_satisfied": bool(
                    erasure_robust_position_frame[
                        "full_rank_after_all_evaluated_erasures"
                    ]
                    and float(
                        erasure_robust_position_frame[
                            "minimum_singular_value_ratio"
                        ]
                    )
                    >= erasure_robust_ratio_threshold - 1.0e-12
                ),
                "active_absolute_range_anchor_present": bool(
                    absolute_range_anchor_records
                ),
                "active_absolute_range_anchor_records": (
                    absolute_range_anchor_records
                ),
                "active_absolute_range_anchor_uses_truth": False,
                "active_absolute_range_anchor_definition": (
                    "active_monostatic_action_with_monostatic_delay_s_measurement"
                ),
                "differential_clock_verification": dict(
                    differential_clock_verification
                ),
                "detectability_is_enforced_by_marginal_correct_update_probability": True,
                "counterfactual_evaluation_epoch": int(evaluation_epoch),
                "action_arrival_epoch": {
                    str(action.action_id): int(
                        arrival_epoch_by_action[action.action_id]
                    )
                    for action in actions
                },
                "joint_schedule_transport": dict(transport_audit),
                "multi_target_expected_information_gain_nats": float(
                    multi_target_information_gain
                ),
                "multi_target_information_weight": float(
                    self.config.multi_target_information_weight
                ),
                "nonlocal_task_risk": nonlocal_task_risk,
                "nonlocal_task_risk_enabled": bool(
                    self.config.nonlocal_task_risk_enabled
                ),
                "nonlocal_task_risk_weight": float(
                    self.config.nonlocal_task_risk_weight
                ),
                "final_selector_estimand": (
                    "finite_outcome_robust_graph_information_plus_track_continuity_"
                    "plus_posterior_predictive_set_risk"
                    if self.config.nonlocal_task_risk_enabled
                    else "finite_outcome_robust_graph_information_plus_track_continuity"
                ),
                "final_selector_estimand_components": (
                    [
                        "finite_outcome_robust_graph_information",
                        "track_continuity",
                        "posterior_predictive_set_risk",
                    ]
                    if self.config.nonlocal_task_risk_enabled
                    else [
                        "finite_outcome_robust_graph_information",
                        "track_continuity",
                    ]
                ),
                "expected_transport_delivery_probability": {
                    str(action.action_id): self.availability.delivery_probability(
                        action.action_id
                    )
                    for action in actions
                },
                "future_resource_consumption_fraction_of_current_slack": float(
                    future_resource_consumption
                ),
                "normalized_resource_opportunity_cost": float(resource_cost),
                "resource_opportunity_cost_definition": (
                    "future_resource_weight*max(action_usage/current_usable_slack)"
                ),
                "resource_consumption_fraction_by_dimension": (
                    resource_consumption_fractions
                ),
                "future_resource_model_is_conservative_max_fraction": True,
                "switching_fraction_of_schedule_capacity": float(
                    switching_fraction
                ),
                "switching_cost_definition": (
                    "switching_weight*min(1,symmetric_difference/max_schedule_size)"
                ),
            },
        )

    def _sequential_game_proposal(
        self,
        rows: Sequence[CandidateScore],
        state: EpisodeState,
        *,
        remaining_epochs: int,
    ) -> tuple[int, dict[str, Any]]:
        if not rows:
            return -1, {"status": "not_applicable"}
        scenario_ids = rows[0].scenario_ids
        if any(row.scenario_ids != scenario_ids for row in rows):
            raise RuntimeError("sequential-game scenario ordering changed across candidates")
        registered_weights = _registered_scenario_weights(rows)
        payoff = _belief_conditioned_payoff_matrix(
            rows,
            self.mission,
            registered_weights,
        )
        scenario_discrimination_by_candidate = (
            np.ptp(payoff, axis=1)
            if payoff.shape[1] > 1
            else np.zeros(payoff.shape[0], dtype=float)
        )
        payoff_scale = max(1.0, float(np.max(np.abs(payoff))))
        scenario_discrimination_tolerance = float(
            128.0 * np.finfo(float).eps * payoff_scale
        )
        scenario_discrimination_maximum = float(
            np.max(scenario_discrimination_by_candidate)
        )
        action_resource_costs = np.asarray(
            [
                float(
                    row.metadata.get(
                        "future_resource_consumption_fraction_of_current_slack",
                        0.0,
                    )
                )
                for row in rows
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(action_resource_costs)) or np.any(
            action_resource_costs < 0.0
        ):
            raise RuntimeError("sequential candidate resource model is invalid")
        switching_costs, previous_row = _normalized_switching_cost_matrix(
            rows,
            state.resources.last_action_ids,
            switching_cost_weight=self.mission.switching_cost_weight,
            max_schedule_size=self.base_world.max_schedule_size,
        )
        from wideband_info_geometry.physics_v58.cognitive_ew import (
            build_hidden_tactic_model,
        )

        model = build_hidden_tactic_model(payoff, scenario_ids)
        if (
            self.threat_belief is None
            or self.threat_belief.size != len(scenario_ids)
            or self.threat_scenario_ids != tuple(scenario_ids)
        ):
            self.threat_belief = registered_weights.copy()
            self.threat_scenario_ids = tuple(scenario_ids)
        safety_mask = np.ones(len(rows), dtype=bool)
        horizon = max(
            1,
            min(int(self.config.sequential_game_horizon), int(remaining_epochs) + 1),
        )
        effective_policy_mode = _effective_sequential_policy_mode(
            self.config, int(state.epoch)
        )
        if effective_policy_mode is SequentialPolicyMode.BELIEF_MYOPIC:
            feasible = action_resource_costs <= 1.0 + 1.0e-12
            if not np.any(feasible):
                raise RuntimeError("belief-myopic policy has no resource-feasible action")
            feasible_rows = np.flatnonzero(feasible)
            root_values = (
                payoff[feasible_rows] @ self.threat_belief
                - switching_costs[previous_row, feasible_rows]
            )
            selected_row = int(feasible_rows[int(np.argmax(root_values))])
            policy_metadata = {
                "solver_kind": "exact_belief_myopic_control",
                "horizon": 1,
                "root_action_values": root_values.tolist(),
                "resource_transition_uses_candidate_physical_consumption": True,
            }
        elif effective_policy_mode is SequentialPolicyMode.BELIEF_POMCP:
            from wideband_info_geometry.physics_v58.stochastic_game import (
                OnlineBeliefTreeSearchConfig,
                scalable_belief_control,
            )

            result = scalable_belief_control(
                payoff,
                model,
                self.threat_belief,
                horizon=horizon,
                discount=self.config.sequential_game_discount,
                safety_mask=safety_mask,
                solver="auto",
                exact_horizon_limit=2,
                online_config=OnlineBeliefTreeSearchConfig(
                    simulations=self.config.sequential_game_simulations,
                    evaluation_rollouts_per_action=(
                        self.config.sequential_game_evaluation_rollouts
                    ),
                    seed=self.config.seed * 1009 + int(state.epoch),
                ),
                action_resource_costs=action_resource_costs,
                initial_resource_fraction=1.0,
                switching_costs=switching_costs,
                previous_action=previous_row,
            )
            selected_row = int(result["selected_action"])
            policy_metadata = result
        elif effective_policy_mode is SequentialPolicyMode.INDEPENDENT_RL:
            from wideband_info_geometry.physics_v58.stochastic_game import (
                IndependentRLConfig,
                train_independent_belief_q_policy,
            )

            policy = train_independent_belief_q_policy(
                payoff,
                model,
                config=IndependentRLConfig(
                    episodes=self.config.independent_rl_training_episodes,
                    steps_per_episode=max(8, horizon * 3),
                    seed=self.config.seed * 1013 + int(state.epoch),
                ),
                training_episode_ids=(
                    f"runtime_train:{self.base_world.world_id}:{state.epoch}",
                ),
                action_resource_costs=action_resource_costs,
                initial_resource_fraction=1.0,
                switching_costs=switching_costs,
            )
            selected_row = policy.select(
                self.threat_belief,
                1.0,
                previous_row,
                safety_mask,
            )
            policy_metadata = {
                "solver_kind": "independently_trained_recurrent_belief_Q_policy",
                "training_episode_ids": list(policy.training_episode_ids),
                "training_episode_overlap_with_runtime": False,
                "rewards_seen": int(policy.rewards_seen),
                "shield_interventions_during_training": int(
                    policy.shield_interventions
                ),
                "resource_transition_uses_candidate_physical_consumption": True,
                "reward_and_switching_cost_share_physical_utility_units": True,
                "reward_transform": policy.reward_transform,
                "reward_positive_scale": float(policy.reward_positive_scale),
                "per_step_centering_that_changes_resource_limited_horizon": False,
            }
        else:
            selected_row = int(
                np.argmax([row.exact_score for row in rows])
            )
            policy_metadata = {"solver_kind": "robust_CVaR_myopic_control"}
        self.last_game_model = model
        self.previous_game_candidate = int(selected_row)
        metadata = {
            "status": "passed",
            "policy_mode": effective_policy_mode.value,
            "configured_policy_mode": self.config.sequential_policy_mode.value,
            "policy_activation_epoch": int(
                self.config.sequential_policy_activation_epoch
            ),
            "matched_common_prefix_active": bool(
                int(state.epoch) < self.config.sequential_policy_activation_epoch
            ),
            "selected_row": int(selected_row),
            "selected_candidate_index": int(rows[selected_row].candidate_index),
            "tactic_ids": list(scenario_ids),
            "prior_threat_belief": self.threat_belief.tolist(),
            "registered_scenario_probability_weights": registered_weights.tolist(),
            "payoff_matrix": payoff.tolist(),
            "payoff_scenario_discrimination_range_by_candidate": (
                scenario_discrimination_by_candidate.tolist()
            ),
            "payoff_scenario_discrimination_range_max": (
                scenario_discrimination_maximum
            ),
            "payoff_scenario_discrimination_tolerance": (
                scenario_discrimination_tolerance
            ),
            "scenario_discriminating_game": bool(
                scenario_discrimination_maximum
                > scenario_discrimination_tolerance
            ),
            "action_resource_consumption_fraction_of_current_slack": (
                action_resource_costs.tolist()
            ),
            "switching_cost_matrix": switching_costs.tolist(),
            "root_previous_action_row": int(previous_row),
            "future_resource_transition_is_action_dependent": True,
            "hidden_tactic_truth_read_by_policy": False,
            "exact_graph_rows_form_registered_prior_immediate_payoff": True,
            "scenario_deviation_is_centered_under_registered_prior": True,
            "switching_cost_matches_exact_rerank_normalization": True,
            "policy": policy_metadata,
        }
        self.game_history.append(metadata)
        return int(rows[selected_row].candidate_index), metadata

    def _latest_jpda_associated_update_audit(self, epoch: int,
            receipt_ids: Sequence[str] = ()) -> dict[str, Any]:
        """Read only the arrived JPDA assignment record for ``epoch``.

        The multi-target engine has already processed the committed
        observations when this method is called by the runtime kernel.  We use
        only action identifiers and receiver-visible joint MAP assignments;
        the offline truth metrics and capture-audit fields in the same record
        are deliberately not inspected.
        """

        audit: dict[str, Any] = {
            "epoch": int(epoch),
            "status": "unavailable",
            "associated_noncalibration_update": False,
            "eligible_action_count": 0,
            "associated_action_count": 0,
            "associated_track_assignment_count": 0,
            "associated_action_ids": [],
            "associated_action_kinds": [],
            "clock_invariant_associated_update": False,
            "clock_invariant_associated_action_ids": [],
            "clock_invariant_associated_action_kinds": [],
            "source": "arrived_jpda_joint_map_assignments",
            "truth_or_future_event_read": False,
            "offline_truth_metrics_read": False,
        }
        engine = self.multi_target_engine
        if engine is None or not bool(getattr(engine, "enabled", False)):
            audit["status"] = "multi_target_engine_unavailable"
            return audit
        history = getattr(engine, "history", None)
        if not isinstance(history, list) or not history:
            audit["status"] = "jpda_history_unavailable"
            return audit
        record = history[-1]
        if not isinstance(record, Mapping):
            audit["status"] = "jpda_history_record_invalid"
            return audit
        if record.get("status") != "passed":
            audit["status"] = "jpda_receiver_batch_not_passed"
            return audit
        try:
            record_epoch = int(record.get("epoch", -1))
        except (TypeError, ValueError):
            audit["status"] = "jpda_history_epoch_invalid"
            return audit
        if record_epoch != int(epoch):
            audit["status"] = "jpda_history_epoch_mismatch"
            audit["latest_record_epoch"] = record_epoch
            return audit
        actions = record.get("actions", ())
        if not isinstance(actions, (list, tuple)):
            audit["status"] = "jpda_action_records_invalid"
            return audit
        action_kind_by_id = {
            int(action.action_id): str(action.kind)
            for action in self.base_world.actions
        }
        associated_action_ids: list[int] = []
        associated_action_kinds: list[str] = []
        clock_invariant_action_ids: list[int] = []
        clock_invariant_action_kinds: list[str] = []
        associated_assignment_count = 0
        eligible_action_count = 0
        for item in actions:
            if not isinstance(item, Mapping):
                continue
            if (item.get("status") != "passed"
                    or item.get("receiver_admission_validated") is not True
                    or item.get("admission_epoch") != int(epoch)
                    or item.get("receipt_id") not in receipt_ids
                    or item.get("receipt_id") not in getattr(
                        engine, "_consumed_return_set_receipts", set())):
                continue
            try:
                action_id = int(item.get("action_id"))
            except (TypeError, ValueError):
                continue
            action_kind = action_kind_by_id.get(action_id, "")
            if action_kind.startswith("calibration"):
                continue
            eligible_action_count += 1
            assignments = item.get("joint_assignment_MAP", ())
            if not isinstance(assignments, (list, tuple)):
                continue
            nonmiss_count = 0
            for assignment in assignments:
                try:
                    nonmiss_count += int(int(assignment) >= 0)
                except (TypeError, ValueError):
                    continue
            if nonmiss_count > 0:
                associated_action_ids.append(action_id)
                associated_action_kinds.append(action_kind)
                if action_kind == "active_monostatic":
                    clock_invariant_action_ids.append(action_id)
                    clock_invariant_action_kinds.append(action_kind)
                associated_assignment_count += nonmiss_count
        audit.update(
            {
                "status": "passed",
                "associated_noncalibration_update": bool(
                    associated_assignment_count > 0
                ),
                "eligible_action_count": int(eligible_action_count),
                "associated_action_count": int(len(associated_action_ids)),
                "associated_track_assignment_count": int(
                    associated_assignment_count
                ),
                "associated_action_ids": associated_action_ids,
                "associated_action_kinds": associated_action_kinds,
                "clock_invariant_associated_update": bool(
                    clock_invariant_action_ids
                ),
                "clock_invariant_associated_action_ids": (
                    clock_invariant_action_ids
                ),
                "clock_invariant_associated_action_kinds": (
                    clock_invariant_action_kinds
                ),
            }
        )
        return audit

    def update_after_observations(
        self,
        received_batch: Any,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        """Update causal controller memory and return its traceable receipt."""

        from .receiver_admission import validate_receiver_batch, ReceiverAdmissionError
        batch = validate_receiver_batch(received_batch, epoch=epoch,
            recipient_id=self.availability.fusion_node_id)
        observations = batch.observations
        consumed = getattr(self, "_consumed_observation_receipts", set())
        if (any(receipt in consumed for receipt in batch.receipt_ids)
                or getattr(self, "_last_observation_receiver_batch", None) is batch):
            raise ReceiverAdmissionError("UNKNOWN_FEEDBACK_RECEIPT_ALREADY_CONSUMED")
        self._consumed_observation_receipts = consumed | set(batch.receipt_ids)
        self._last_observation_receiver_batch = batch

        action_kind_by_id = {
            action.action_id: action.kind for action in self.base_world.actions
        }
        verification_evidence = (
            _realized_innovation_verification_certificate(
                self._last_innovation_verification_world,
                self._last_innovation_verification_action_ids,
                observations,
                epoch=int(epoch),
                alarm_hypotheses=(
                    self._last_innovation_verification_hypotheses
                ),
                minimum_sensing_actions=(
                    self.config.innovation_health_min_detected_actions_to_clear
                ),
                require_position_separating_rank=(
                    self.config.innovation_verification_require_position_separating_rank
                ),
            )
            if (
                self._last_selection_was_innovation_verification
                and self._last_innovation_verification_world is not None
            )
            else {
                "status": "not_committed",
                "realized_verification_evidence_complete": False,
                "truth_or_future_event_read": False,
            }
        )
        health_update = self._innovation_health.update(
            observations,
            epoch=int(epoch),
            action_kind_by_id=action_kind_by_id,
            verification_committed=(
                self._last_selection_was_innovation_verification
            ),
            verification_evidence_complete=bool(
                verification_evidence.get(
                    "realized_verification_evidence_complete", False
                )
            ),
            verification_evidence_receipt=verification_evidence,
        )
        self._last_selection_was_innovation_verification = False
        self._last_innovation_verification_world = None
        self._last_innovation_verification_action_ids = ()
        self._last_innovation_verification_hypotheses = ()
        detected_epochs = {
            int(item.observation_epoch)
            for item in observations
            if bool(item.detected)
            and not action_kind_by_id.get(int(item.action_id), "").startswith(
                "calibration"
            )
        }
        self._target_detection_epochs.extend(sorted(detected_epochs))
        association_audit = self._latest_jpda_associated_update_audit(
            int(epoch), batch.receipt_ids)
        if bool(association_audit["associated_noncalibration_update"]):
            self._target_associated_update_epochs.append(int(epoch))
        if bool(association_audit["clock_invariant_associated_update"]):
            self._clock_invariant_associated_update_epochs.append(int(epoch))
            self._latest_clock_invariant_associated_action_ids = tuple(
                int(value)
                for value in association_audit[
                    "clock_invariant_associated_action_ids"
                ]
            )
        observation_update_audit: dict[str, Any] = {
            "innovation_health_after_observation": dict(health_update),
            "track_confirmation_association_audit": dict(association_audit),
            "truth_or_future_event_read": False,
        }
        if self.game_history:
            self.game_history[-1].update(observation_update_audit)
        if (
            not observations
            or self.config.sequential_policy_mode is SequentialPolicyMode.ROBUST_CVAR
            or self.last_game_model is None
            or self.threat_belief is None
            or self.previous_game_candidate < 0
        ):
            if not observations:
                observation_update_audit["threat_belief_measurement_update"] = "skipped_no_packet"
            return observation_update_audit
        detected = [bool(item.detected) for item in observations]
        false_responsibilities = [
            float(item.metadata.get("association_responsibilities", [1.0, 0.0])[1])
            for item in observations
            if isinstance(item.metadata.get("association_responsibilities"), (list, tuple))
            and len(item.metadata.get("association_responsibilities")) == 2
        ]
        normalized_innovations: list[float] = []
        for item in observations:
            if item.residual is None:
                continue
            innovation = item.metadata.get("innovation_covariance")
            if innovation is None:
                continue
            residual = np.asarray(item.residual, dtype=float)
            covariance = np.asarray(innovation, dtype=float)
            normalized_innovations.append(
                float(residual @ np.linalg.solve(covariance, residual))
                / max(residual.size, 1)
            )
        if false_responsibilities and max(false_responsibilities) > 0.55:
            observation_index = 3
        elif observations and not any(detected):
            observation_index = 1
        elif normalized_innovations and float(np.mean(normalized_innovations)) > 3.0:
            observation_index = 2
        else:
            observation_index = 0
        from wideband_info_geometry.physics_v58.cognitive_ew import (
            update_tactic_belief,
        )

        prior = self.threat_belief.copy()
        self.threat_belief = update_tactic_belief(
            prior,
            self.last_game_model,
            self.previous_game_candidate,
            observation_index,
        )
        if self.game_history:
            self.game_history[-1]["observable_threat_indicator"] = int(
                observation_index
            )
            self.game_history[-1]["posterior_threat_belief"] = (
                self.threat_belief.tolist()
            )
        observation_update_audit.update(
            {
                "observable_threat_indicator": int(observation_index),
                "posterior_threat_belief": self.threat_belief.tolist(),
            }
        )
        return observation_update_audit

    def select(
        self,
        world: PhysicalWorld,
        state: EpisodeState,
        *,
        remaining_epochs: int,
        candidate_hints: Sequence[Mapping[str, Any]] = (),
        candidate_hint_context: Mapping[str, Any] | None = None,
    ) -> DecisionRecord:
        start = time.perf_counter()
        self._validate_controller_contract(world, state)
        innovation_health_before_decision = self._innovation_health.snapshot()
        library, proposal_by_schedule = self.generator.build(
            world,
            state,
            self.availability,
            candidate_hints=candidate_hints,
            innovation_health_active=bool(
                innovation_health_before_decision["active"]
            ),
            innovation_verification_active=bool(
                innovation_health_before_decision["active"]
                and self.config.innovation_verification_diverse_candidate_set_enabled
            ),
            innovation_verification_hypotheses=tuple(
                str(value)
                for value in innovation_health_before_decision.get(
                    "active_alarm_hypotheses", ()
                )
            ),
            previous_clock_invariant_action_ids=(
                self._latest_clock_invariant_associated_action_ids
            ),
        )
        shield = CampaignSafetyShield(self.base_world, library, self.mission)
        safe_mask_array = shield.safe_mask(state, remaining_epochs=remaining_epochs)
        safe_indices = [int(index) for index in np.flatnonzero(safe_mask_array)]
        proposal_scores = np.asarray(
            [proposal_by_schedule.get(schedule, 0.0) for schedule in library.schedules],
            dtype=float,
        )
        shortlist = self._stratified_shortlist(
            world,
            library,
            proposal_scores,
            safe_indices,
        )
        communication_template = self.availability.snapshot(state.epoch)
        by_id = {action.action_id: action for action in world.actions}
        transport_by_candidate: dict[int, dict[str, Any]] = {}
        operational_shortlist: list[int] = []
        for candidate_index in shortlist:
            actions = tuple(
                by_id[action_id]
                for action_id in library.schedules[candidate_index]
            )
            feasible, _, audit = self.availability.schedule_operational_feasibility(
                actions
            )
            if feasible:
                operational_shortlist.append(candidate_index)
                transport_by_candidate[candidate_index] = audit
        shortlist = operational_shortlist
        evaluation_epoch = max(
            (
                int(arrival)
                for audit in transport_by_candidate.values()
                for arrival in audit.get("action_arrival_epoch", {}).values()
            ),
            default=state.epoch,
        )
        baseline = self.belief.counterfactual_at_epoch(
            (),
            evaluation_epoch=evaluation_epoch,
            reference_world=world,
            communication_template=communication_template,
        )
        if baseline.status != "passed":
            raise RuntimeError("future no-action graph baseline failed closed")
        track_phase = self._track_phase_contract(state)
        exact_rows = tuple(
            self._score_candidate(
                world,
                state,
                library,
                candidate_index,
                float(proposal_scores[candidate_index]),
                evaluation_epoch,
                baseline.position_covariance_m2,
                baseline.velocity_covariance_m2_s2,
                communication_template,
                transport_by_candidate[candidate_index],
                track_phase,
            )
            for candidate_index in shortlist
        )
        required_update_probability = float(
            track_phase["required_correct_update_probability"]
        )
        phase = str(track_phase["phase"])
        selection_rows, track_update_selection_audit = (
            _track_update_constrained_selection_rows(
                exact_rows,
                phase=phase,
                required_probability=required_update_probability,
                probability_key_override=(
                    "calibrated_track_update_probability_lcb_95_one_sided"
                    if self.config.schedule_update_probability_calibration_enabled
                    else None
                ),
                infeasible_indifference_band=(
                    self.mission.track_update_probability_indifference_band
                ),
            )
        )
        admissible_indices = set(
            track_update_selection_audit[
                "constraint_satisfying_candidate_indices"
            ]
        )
        admissible_rows = tuple(
            row for row in exact_rows if int(row.candidate_index) in admissible_indices
        )
        admissibility_rule = (
            "hard_mission_minimum_track_update_probability_when_feasible_then_"
            "registered_probability_indifference_class_when_infeasible_then_"
            "optional_erasure_robust_directional_frame_then_"
            "exact_task_information_risk_resource_objective"
        )
        (
            policy_selection_rows,
            erasure_robust_position_selection_audit,
        ) = _erasure_robust_position_selection_rows(
            selection_rows,
            phase=phase,
            enabled=self.config.erasure_robust_position_geometry_enabled,
            maximum_jacobian_condition_number=(
                self.config.erasure_robust_position_geometry_max_jacobian_condition_number
            ),
        )
        innovation_verification_audit = {
            "status": "not_active",
            "applied": False,
            "candidate_indices": [
                int(row.candidate_index) for row in selection_rows
            ],
            "truth_or_future_event_read": False,
        }
        if bool(innovation_health_before_decision["active"]):
            policy_selection_rows, innovation_verification_audit = (
                _innovation_verification_selection_rows(
                    selection_rows,
                    retain_admissible_set=(
                        self.config.innovation_verification_diverse_candidate_set_enabled
                    ),
                    minimum_sensing_actions=(
                        self.config.innovation_health_min_detected_actions_to_clear
                    ),
                    probability_indifference_band=(
                        self.config.innovation_verification_probability_indifference_band
                    ),
                    geometry_floor_fraction=(
                        self.config.innovation_verification_geometry_floor_fraction
                    ),
                    require_absolute_range_anchor_when_available=(
                        self.config.innovation_verification_require_absolute_range_anchor
                    ),
                    require_differential_clock_identification_when_available=(
                        self.config.innovation_verification_require_differential_clock_identification
                        and DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS
                        in set(
                            innovation_health_before_decision.get(
                                "active_alarm_hypotheses", ()
                            )
                        )
                    ),
                    require_position_separating_rank=(
                        self.config.innovation_verification_require_position_separating_rank
                    ),
                )
            )
        policy_selection_rows, clock_sentinel_audit = (
            _clock_sentinel_selection_rows(
                policy_selection_rows,
                self.generator.last_clock_sentinel_requirement,
            )
        )
        v73_runtime=None
        if self.config.v73_adaptive_enabled:
            certificate_rows = self._v73_certificate_rows(
                exact_rows,
                policy_selection_rows,
            )
            proposed,sequential_policy,v73_runtime=self._v73_select(
                state,
                certificate_rows,
                library,
            )
        else:
            proposed, sequential_policy = self._sequential_game_proposal(
                policy_selection_rows,
                state,
                remaining_epochs=remaining_epochs,
            )
        shield_decision = shield.apply(
            proposed,
            state,
            remaining_epochs=remaining_epochs,
        )
        wall_clock_latency = time.perf_counter() - start
        # The *modeled* decision latency is the only quantity permitted to gate
        # execution.  Host wall-clock time is not part of the modeled world:
        # letting it decide whether the shield substitutes its backup candidate
        # makes the executed action -- and therefore the action-conditioned world
        # transition p(s_{k+1} | s_k, a_k, ...) -- a function of CPU load, so the
        # same seed yields different physics on different machines.  Wall clock
        # is retained below purely as an observability field.
        modeled_compute_cost_s = (
            self.config.modeled_compute_base_s
            + self.config.modeled_compute_per_candidate_s * len(exact_rows)
            + self.config.modeled_compute_per_scenario_s
            * sum(len(row.scenario_ids) for row in exact_rows)
        )
        latency = float(modeled_compute_cost_s)
        deadline = min(self.config.decision_deadline_s, self.mission.maximum_decision_latency_s)
        deadline_miss = latency > deadline
        selected_candidate = shield_decision.selected_candidate
        selected_ids = shield_decision.selected_action_ids
        shield_reasons = list(shield_decision.reasons)
        shield_intervened = shield_decision.intervened
        if deadline_miss and shield_decision.backup_candidate >= 0:
            selected_candidate = (next((row.candidate_index for row in exact_rows if not row.action_ids),-1)
                                  if self.config.v73_adaptive_enabled else shield_decision.backup_candidate)
            selected_ids = library.schedules[selected_candidate]
            shield_reasons.append("decision_deadline_miss_backup")
            shield_intervened = True
        if self.config.v73_adaptive_enabled:
            exact_indices={int(row.candidate_index) for row in exact_rows}
            if selected_candidate not in exact_indices:
                noop_index=next((int(row.candidate_index) for row in exact_rows if not row.action_ids),-1)
                if noop_index<0: raise RuntimeError("V73 safety fallback has no exact no-op certificate")
                selected_candidate=noop_index; selected_ids=()
                shield_reasons.append("v73_noncertified_shield_backup_replaced_by_exact_noop")
                shield_intervened=True
        unavailable = [
            action_id
            for action_id in selected_ids
            if not self.availability.action_available(action_id)
        ]
        selected_actions = tuple(
            next(action for action in world.actions if action.action_id == action_id)
            for action_id in selected_ids
        )
        schedule_available, schedule_reasons, schedule_transport = (
            self.availability.schedule_operational_feasibility(selected_actions)
        )
        fail_closed = False
        if unavailable or not schedule_available:
            selected_candidate = -1
            selected_ids = ()
            if unavailable:
                shield_reasons.append(
                    "selected_action_became_operationally_unavailable"
                )
            shield_reasons.extend(schedule_reasons)
            shield_intervened = True
            fail_closed = True
            self.availability.clear_committed_schedule()
        else:
            try:
                schedule_transport = self.availability.commit_selected_schedule(
                    selected_actions,
                    epoch=state.epoch,
                )
            except RuntimeError as error:
                selected_candidate = -1
                selected_ids = ()
                schedule_available = False
                shield_reasons.append(
                    "selected_schedule_arrival_contract_failed_closed"
                )
                shield_reasons.append(str(error))
                shield_intervened = True
                fail_closed = True
                self.availability.clear_committed_schedule()
        sentinel_required_ids = {
            int(value)
            for value in clock_sentinel_audit.get("required_action_ids", ())
        }
        clock_sentinel_audit = {
            **clock_sentinel_audit,
            "selected_candidate": int(selected_candidate),
            "selected_action_ids": [int(value) for value in selected_ids],
            "committed": bool(
                clock_sentinel_audit.get("applied", False)
                and sentinel_required_ids.issubset(set(selected_ids))
                and schedule_available
                and not fail_closed
            ),
        }
        self._last_selection_was_innovation_verification = (
            _innovation_verification_was_committed(
                innovation_health_active=bool(
                    innovation_health_before_decision["active"]
                ),
                verification_audit=innovation_verification_audit,
                selected_candidate=int(selected_candidate),
                selected_action_ids=selected_ids,
                schedule_available=bool(schedule_available),
                fail_closed=bool(fail_closed),
            )
        )
        if self._last_selection_was_innovation_verification:
            self._last_innovation_verification_world = world
            self._last_innovation_verification_action_ids = tuple(
                int(value) for value in selected_ids
            )
            self._last_innovation_verification_hypotheses = tuple(
                str(value)
                for value in innovation_health_before_decision.get(
                    "active_alarm_hypotheses", ()
                )
            )
        else:
            self._last_innovation_verification_world = None
            self._last_innovation_verification_action_ids = ()
            self._last_innovation_verification_hypotheses = ()
        status = (
            "passed"
            if not unavailable
            and schedule_available
            and (not selected_ids or exact_rows)
            else "failed_closed"
        )
        v73_metadata=None
        if v73_runtime is not None:
            final_index=(selected_candidate if selected_candidate>=0 else
                         next(row.candidate_index for row in exact_rows if not row.action_ids))
            final_id=f"candidate:{int(final_index)}"
            committed_decision=replace(v73_runtime["decision"],selected_action_id=final_id)
            physical_certificate_count=sum(
                bool(not certificate.is_noop and certificate.feasible)
                for certificate in v73_runtime["certificates"]
            )
            reference_empty_domain=bool(
                self.config.v73_reference_noninferiority_gate
                and physical_certificate_count == 0
            )
            dual_after=self._v73_adaptive_controller.commit_certificate(
                committed_decision,v73_runtime["certificates"])
            v73_metadata={**v73_runtime["metadata"],"selected_after_campaign_shield":final_id,
                          "dual_after":float(dual_after),
                          "dual_update_count":0 if reference_empty_domain else 1,
                          "dual_update_skipped_reason":(
                              "no_physical_task_candidate_available"
                              if reference_empty_domain
                              else None
                          ),
                          "physical_action_committed":bool(selected_ids)}
        return DecisionRecord(
            epoch=state.epoch,
            candidate_library=library.schedules,
            safe_mask=tuple(bool(value) for value in safe_mask_array),
            selected_candidate=int(selected_candidate),
            selected_action_ids=tuple(int(value) for value in selected_ids),
            proposed_candidate=int(proposed),
            shield_intervened=bool(shield_intervened),
            shield_reasons=tuple(shield_reasons),
            exact_rerank_rows=exact_rows,
            decision_latency_s=float(latency),
            deadline_miss=bool(deadline_miss),
            fail_closed=bool(fail_closed),
            status=status,
            metadata={
                "candidate_generation_hash": library.generation_hash,
                "proposal_role": "bounded_shortlist_only",
                "proposal_backend": self.config.proposal_backend.value,
                "candidate_domain_exactness": (
                    "exact_physics_rerank_within_generated_bounded_library"
                ),
                "global_combinatorial_optimum_claimed": False,
                "exact_graph_rerank_required": self.config.exact_rerank_required,
                "exact_rerank_candidate_indices": shortlist,
                "exact_shortlist_retention_policy": (
                    "noop_then_optional_max_erasure_robust_frame_then_physical_"
                    "role_strata_and_max_correct_update_probability_before_"
                    "proposal_score_fill"
                ),
                "erasure_robust_shortlist_retention": dict(
                    getattr(
                        self,
                        "_last_erasure_robust_shortlist_retention",
                        {
                            "enabled": False,
                            "retained_candidate_index": None,
                            "truth_or_future_event_read": False,
                        },
                    )
                ),
                "candidate_generation_audit": dict(
                    self.generator.last_generation_audit
                ),
                "track_phase_contract": dict(track_phase),
                "track_update_admissibility_rule": admissibility_rule,
                "track_update_admissible_candidate_indices": [
                    int(row.candidate_index) for row in admissible_rows
                ],
                "track_update_admissibility_applied": bool(admissible_rows),
                "track_update_admissibility_fallback_to_full_exact_set": False,
                "track_update_admissibility_fallback_to_minimum_shortfall": bool(
                    track_update_selection_audit["fallback_to_minimum_shortfall"]
                ),
                "track_update_selection_candidate_indices": list(
                    track_update_selection_audit["selection_candidate_indices"]
                ),
                "track_update_constraint_audit": dict(
                    track_update_selection_audit
                ),
                "erasure_robust_position_geometry_selection": dict(
                    erasure_robust_position_selection_audit
                ),
                "erasure_robust_position_geometry_applied": bool(
                    erasure_robust_position_selection_audit["applied"]
                ),
                "erasure_robust_position_geometry_selection_candidate_indices": list(
                    erasure_robust_position_selection_audit[
                        "selection_candidate_indices"
                    ]
                ),
                "innovation_health_before_decision": dict(
                    innovation_health_before_decision
                ),
                "innovation_verification_selection": {
                    **dict(innovation_verification_audit),
                    "committed": bool(
                        self._last_selection_was_innovation_verification
                    ),
                    "selected_candidate": int(selected_candidate),
                },
                "clock_sentinel_selection": dict(clock_sentinel_audit),
                "safe_candidate_count": len(safe_indices),
                "fusion_node_id": self.availability.fusion_node_id,
                "counterfactual_evaluation_epoch": int(evaluation_epoch),
                "delayed_observations_scored_on_future_graph_branch": True,
                "mission": asdict(self.mission),
                "modeled_compute_cost_s": float(modeled_compute_cost_s),
                "modeled_decision_latency_s": float(modeled_compute_cost_s),
                "wall_clock_decision_latency_s": float(wall_clock_latency),
                "wall_clock_latency_used_as_resource_cost": False,
                "wall_clock_latency_used_as_execution_gate": False,
                "decision_deadline_gate": "modeled_decision_latency_s",
                "controller_contract_verified": True,
                "joint_schedule_transport_verified": bool(schedule_available),
                "joint_schedule_transport": schedule_transport,
                "sequential_policy": sequential_policy,
                "forced_noop_due_empty_feasible_action_set": bool(
                    library.schedules == ((),)
                    and not self.config.include_noop_candidate
                ),
                "admissible_action_correspondence_closed_by_emergency_noop": True,
                "v73_adaptive_selector": v73_metadata,
                "agent_candidate_hints": to_builtin(tuple(candidate_hints)),
                "agent_candidate_hint_context": to_builtin(
                    dict(candidate_hint_context or {})
                ),
                "agent_hints_are_proposal_only": True,
                "agent_hints_can_bypass_exact_rerank": False,
                "agent_hints_can_bypass_campaign_shield": False,
            },
        )

    def v73_adaptive_decision(self, controller_state, candidate_certificates, *,
                              controller=None, safety_gate=None, commit_dual=False):
        """Select one action from exact certificates on frozen forks of the sole belief.

        This method accepts no truth.  A veto may only replace the proposal with
        the already-certified explicit no-op.  Dual state changes only when
        ``commit_dual`` is explicitly true after the unique action commit.
        """
        from wideband_info_geometry.operational_runtime.v73_adaptive_belief_risk import (
            AdaptiveMissionConstrainedController,
        )
        ctl = controller or getattr(self, "_v73_adaptive_controller", None)
        if ctl is None:
            ctl = AdaptiveMissionConstrainedController()
            try:
                self._v73_adaptive_controller = ctl
            except Exception:
                pass
        decision = ctl.decide(controller_state, candidate_certificates)
        if safety_gate is not None and safety_gate(decision.selected_action_id, decision) is False:
            noop = next((c for c in candidate_certificates if c.is_noop and c.feasible), None)
            if noop is None:
                raise RuntimeError("safety veto without feasible explicit no-op")
            decision = ctl.decide(controller_state, [noop])
        if commit_dual:
            ctl.commit_certificate(decision, candidate_certificates)
        return decision
