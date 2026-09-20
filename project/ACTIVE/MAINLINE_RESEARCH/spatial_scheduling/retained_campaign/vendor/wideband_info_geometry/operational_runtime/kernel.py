from __future__ import annotations

"""Single executable world-observation-inference-availability-decision kernel."""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np

from wideband_info_geometry.physics_v58.dynamic_episode import EpisodeConfig
from wideband_info_geometry.physics_v58.measurement import StateLayout
from wideband_info_geometry.physics_v58.schema import PhysicalWorld
from wideband_info_geometry.physics_v58.v65_safe_autonomy import MissionIR

from .availability import DistributedAvailabilityManager
from .belief import GraphBeliefEngine
from .contracts import (
    DecisionRecord,
    FusionMode,
    RuntimeConfig,
    RuntimeStepRecord,
    to_builtin,
)
from .decision import GraphSafeDecisionEngine
from .innovation_health import DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS
from .observation import ObservationEngine, loewner_upper_envelope
from .multitarget_v71 import MultiTargetAssociationEngine
from .measurement_return_sets import (
    UnlabelledReturnSet, capture_measurement_return_set,
)
from .receiver_admission import ReceiverAdmissionQueue, CapturePayloadFailure
from .world import DynamicWorldEngine


def _mahalanobis_step_sigma(
    delta: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Root-mean-square posterior-sigma length of a normalized state step."""

    vector = np.asarray(delta, dtype=float).reshape(-1)
    matrix = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    inverse = np.linalg.pinv(matrix, rcond=1.0e-12, hermitian=True)
    quadratic = max(float(vector @ inverse @ vector), 0.0)
    return float(math.sqrt(quadratic / max(vector.size, 1)))


def _line_search_scales(minimum: float) -> tuple[float, ...]:
    values: list[float] = []
    scale = 1.0
    lower = float(minimum)
    while scale >= lower - 1.0e-15:
        values.append(max(scale, lower))
        if scale <= lower:
            break
        scale *= 0.5
        if scale < lower:
            scale = lower
    return tuple(dict.fromkeys(values))


def _causal_reacquisition_covariance_envelope(
    current_covariance: np.ndarray,
    registered_initial_covariance: np.ndarray,
    *,
    floor_scale: float,
    registered_change_covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build truth-free search support that dominates current and initial P.

    The returned matrix is a receiver/acquisition search support only.  It is
    never installed as the DynamicFactorGraph covariance and therefore does
    not create a second posterior authority.
    """

    current = np.asarray(current_covariance, dtype=float)
    initial = np.asarray(registered_initial_covariance, dtype=float)
    if (
        current.shape != initial.shape
        or current.ndim != 2
        or current.shape[0] != current.shape[1]
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(initial))
    ):
        raise ValueError(
            "causal reacquisition covariance inputs must be equal finite square matrices"
        )
    scale = float(floor_scale)
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(
            "causal reacquisition covariance floor scale must be at least one"
        )
    current = 0.5 * (current + current.T)
    initial = 0.5 * (initial + initial.T)
    magnitude = max(
        float(np.max(np.abs(current))),
        float(np.max(np.abs(initial))),
        1.0,
    )
    tolerance = magnitude * 1.0e-10
    if min(
        float(np.min(np.linalg.eigvalsh(current))),
        float(np.min(np.linalg.eigvalsh(initial))),
    ) < -tolerance:
        raise ValueError("causal reacquisition covariance input is not PSD")
    change = (
        np.zeros_like(initial)
        if registered_change_covariance is None
        else np.asarray(registered_change_covariance, dtype=float)
    )
    if (
        change.shape != initial.shape
        or not np.all(np.isfinite(change))
    ):
        raise ValueError(
            "registered change covariance must match the finite initial covariance"
        )
    change = 0.5 * (change + change.T)
    if float(np.min(np.linalg.eigvalsh(change))) < -tolerance:
        raise ValueError("registered change covariance is not PSD")
    registered_floor = initial * scale**2 + change
    envelope = loewner_upper_envelope(current, registered_floor)
    current_margin = float(
        np.min(np.linalg.eigvalsh(envelope - current))
    )
    floor_margin = float(
        np.min(np.linalg.eigvalsh(envelope - registered_floor))
    )
    if min(current_margin, floor_margin) < -10.0 * tolerance:
        raise RuntimeError("Loewner search covariance construction failed")
    return envelope, {
        "schema": "causal_innovation_reacquisition_covariance/v1",
        "status": "passed",
        "floor_scale_standard_deviations": scale,
        "current_covariance_trace": float(np.trace(current)),
        "registered_initial_covariance_trace": float(np.trace(initial)),
        "registered_floor_covariance_trace": float(np.trace(registered_floor)),
        "registered_change_covariance_trace": float(np.trace(change)),
        "registered_change_covariance_applied": bool(
            registered_change_covariance is not None
        ),
        "search_covariance_trace": float(np.trace(envelope)),
        "minimum_search_minus_current_eigenvalue": current_margin,
        "minimum_search_minus_registered_floor_eigenvalue": floor_margin,
        "search_covariance_is_authoritative_posterior": False,
        "single_authoritative_dynamic_factor_graph_preserved": True,
        "truth_or_future_event_read": False,
    }


def _registered_network_clock_step_covariance_normalized(
    layout: StateLayout,
    *,
    rms_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Map a registered zero-mean network clock change into graph units.

    For ``n`` clocks, ``P = I - 11^T/n`` projects out the unobservable common
    time origin.  ``Q = sigma^2 n/(n-1) P`` has expected per-node RMS ``sigma``
    on that differential subspace.  No event time, realization, or truth state
    enters this construction.
    """

    sigma = float(rms_s)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("registered network clock-step RMS must be positive")
    node_ids = tuple(layout.node_ids)
    count = len(node_ids)
    if count < 2:
        raise ValueError("differential clock search requires at least two nodes")
    projector = np.eye(count) - np.ones((count, count), dtype=float) / count
    physical_block = sigma**2 * count / (count - 1) * projector
    scales = np.asarray(layout.scales(), dtype=float)
    covariance = np.zeros((layout.dimension, layout.dimension), dtype=float)
    indices = np.asarray(
        [layout.clock_index(node_id) for node_id in node_ids],
        dtype=int,
    )
    covariance[np.ix_(indices, indices)] = physical_block / np.outer(
        scales[indices], scales[indices]
    )
    return covariance, {
        "schema": "registered_differential_clock_change_covariance/v1",
        "status": "passed",
        "registered_clock_step_rms_s": sigma,
        "node_count": count,
        "common_clock_mode_projected_out": True,
        "differential_subspace_rank": count - 1,
        "normalized_covariance_trace": float(np.trace(covariance)),
        "used_only_after_causal_differential_clock_alarm": True,
        "event_epoch_or_realization_read": False,
        "truth_or_future_event_read": False,
    }


class OperationalFusionRuntime:
    """The only supported operational execution path.

    Runtime ownership is deliberately singular:

    * ``DynamicWorldEngine`` owns simulator truth;
    * ``GraphBeliefEngine`` owns all online state uncertainty;
    * ``DistributedAvailabilityManager`` owns action/factor reachability;
    * ``GraphSafeDecisionEngine`` may propose, but exact graph reranking and the
      campaign shield determine execution;
    * ``ObservationEngine`` is the only adapter allowed to create realized
      measurement factors.
    """

    def __init__(
        self,
        base_world: PhysicalWorld,
        runtime_config: RuntimeConfig,
        episode_config: EpisodeConfig,
        mission: MissionIR,
        coordination_layer: Any | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        base_world.validate()
        runtime_config.validate()
        episode_config.validate()
        mission.validate()
        if runtime_config.horizon_steps > episode_config.horizon_steps:
            raise ValueError("runtime horizon cannot exceed episode horizon")
        self.base_world = base_world
        self.config = runtime_config
        self.episode_config = episode_config
        self.mission = mission
        # Optional Part-1 layer.  Its only decision-side output is a bounded set
        # of proposal hints; the authoritative posterior, exact reranker, safety
        # shield, observation engine, and world transition remain unchanged.
        self.coordination_layer = coordination_layer
        self.world = DynamicWorldEngine(base_world, runtime_config, episode_config)
        self.availability = DistributedAvailabilityManager(base_world, runtime_config)
        self.belief = GraphBeliefEngine(base_world, runtime_config)
        self.observation = ObservationEngine(runtime_config, self.availability)
        if (
            runtime_config.nonlocal_task_risk_enabled
            and runtime_config.nonlocal_task_risk_closed_world_support
            and (
                episode_config.target_birth_probability > 0.0
                or any(
                    len(event) >= 2 and str(event[1]) == "target_birth"
                    for event in episode_config.event_schedule
                )
            )
        ):
            raise ValueError(
                "the closed-world nonlocal-risk contract conflicts with the "
                "declared episode birth process"
            )
        self.multitarget = MultiTargetAssociationEngine(
            base_world,
            runtime_config,
            episode_config,
        )
        self.decision = GraphSafeDecisionEngine(
            base_world,
            runtime_config,
            self.availability,
            self.belief,
            mission,
            self.multitarget,
        )
        self.steps: list[RuntimeStepRecord] = []
        self.leakage_audits: list[Mapping[str, Any]] = []
        self.world_event_history: list[Mapping[str, Any]] = []
        self.mixture_iteration_history: list[Mapping[str, Any]] = []
        self._registered_initial_state_covariance_normalized: np.ndarray | None = None
        self._registered_initial_position_covariance_m2: np.ndarray | None = None
        self._installed_clock_change_process_support_alarm_episode_count = 0
        self._progress_callback = progress_callback
        # A failure discovered after an action commitment cannot undo that
        # physical event. Finish its one transition, then forbid new decisions.
        self._terminal_runtime_failure: dict[str, Any] | None = None
        if coordination_layer is not None:
            raise ValueError(
                "UNKNOWN_LOCAL_RECIPIENT_ADMISSION: Phase11 receiver revision "
                "requires an explicit local-node capture receipt extension before "
                "the coordination path can execute; full asynchronous scope remains open"
            )
        capture_bound = max(1, len(base_world.actions) * self.config.horizon_steps)
        self._receiver_queue = ReceiverAdmissionQueue(
            self.availability.fusion_node_id, lag_epochs=self.config.fixed_lag_steps,
            max_pending=capture_bound, max_consumed=capture_bound,
        )
        self.receiver_admission_history: list[dict[str, Any]] = []
        self._offline_return_set_capture_history: list[dict[str, Any]] = []

    def set_progress_callback(
        self,
        callback: Callable[[Mapping[str, Any]], None] | None,
    ) -> None:
        """Register a non-authoritative, truth-free runtime progress sink."""

        self._progress_callback = callback

    def _emit_progress(self, stage: str, *, epoch: int, **values: Any) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(
            {
                "schema": "unified_operational_runtime_progress/v1",
                "epoch": int(epoch),
                "stage": str(stage),
                "completed_epoch_count": len(self.steps),
                "truth_state_exposed": False,
                **values,
            }
        )

    def step(self) -> RuntimeStepRecord:
        if self._terminal_runtime_failure is not None or (
            self.steps and self.steps[-1].status != "passed"
        ):
            raise RuntimeError("runtime terminated after a failed step; no further decision is permitted")
        epoch = self.world.epoch
        if epoch >= self.config.horizon_steps:
            raise RuntimeError("runtime horizon has already been exhausted")
        self._emit_progress("epoch_started", epoch=epoch)

        # Availability is an observable property of the current operational
        # world.  It is registered before graph solution so delayed factors that
        # arrive at this epoch can be admitted under the correct topology.
        operational_reference = self.world.operational_reference_world()
        availability = self.availability.evaluate(
            operational_reference,
            self.world.operational_network_state(),
        )
        communication = self.availability.snapshot(epoch)
        innovation_health_before_transition = (
            self.decision.innovation_health_snapshot()
        )
        clock_change_process_covariance = None
        clock_change_process_support_audit: dict[str, Any] = {
            "schema": "causal_clock_change_process_support/v1",
            "status": "disabled",
            "enabled": bool(
                self.config.innovation_health_clock_change_process_support_enabled
            ),
            "epoch": int(epoch),
            "alarm_episode_count": int(
                innovation_health_before_transition.get("alarm_episode_count", 0)
            ),
            "installed_alarm_episode_count_before": int(
                self._installed_clock_change_process_support_alarm_episode_count
            ),
            "authoritative_dynamic_factor_graph_transition_updated": False,
            "common_clock_mode_projected_out": False,
            "one_installation_per_alarm_episode": True,
            "event_epoch_or_realization_read": False,
            "truth_or_future_event_read": False,
        }
        install_clock_change_process_support = False
        if self.config.innovation_health_clock_change_process_support_enabled:
            alarm_active = bool(
                innovation_health_before_transition.get("active", False)
            )
            alarm_hypotheses = set(
                innovation_health_before_transition.get(
                    "active_alarm_hypotheses", ()
                )
            )
            alarm_episode_count = int(
                innovation_health_before_transition.get("alarm_episode_count", 0)
            )
            if epoch == 0:
                process_status = "not_applicable_initial_epoch"
            elif not alarm_active:
                process_status = "no_active_causal_alarm"
            elif DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS not in alarm_hypotheses:
                process_status = "active_alarm_not_differential_clock"
            elif (
                alarm_episode_count
                <= self._installed_clock_change_process_support_alarm_episode_count
            ):
                process_status = "already_installed_for_alarm_episode"
            else:
                (
                    clock_change_process_covariance,
                    registered_process_audit,
                ) = _registered_network_clock_step_covariance_normalized(
                    self.belief.layout,
                    rms_s=(
                        self.config.innovation_health_reacquisition_clock_step_rms_s
                    ),
                )
                clock_change_process_support_audit = {
                    **clock_change_process_support_audit,
                    **registered_process_audit,
                    "schema": "causal_clock_change_process_support/v1",
                    "status": "installed",
                    "enabled": True,
                    "epoch": int(epoch),
                    "alarm_episode_count": alarm_episode_count,
                    "installed_alarm_episode_count_before": int(
                        self._installed_clock_change_process_support_alarm_episode_count
                    ),
                    "authoritative_dynamic_factor_graph_transition_updated": True,
                    "one_installation_per_alarm_episode": True,
                }
                install_clock_change_process_support = True
                process_status = "installed"
            if not install_clock_change_process_support:
                clock_change_process_support_audit = {
                    **clock_change_process_support_audit,
                    "status": process_status,
                    "active_alarm_hypotheses": sorted(alarm_hypotheses),
                }
        # Inspect arrivals before graph preparation and primary synchronization.
        pre_batch = self._receiver_queue.prepare(epoch=epoch)
        pre_admission = pre_batch.audit()
        self.receiver_admission_history.append(pre_admission)
        if pre_batch.status == "passed":
            if pre_batch.admitted_sets:
                raise RuntimeError("pre-selection timely capture cannot preexist commitment")
            self._receiver_queue.commit(pre_batch)
        self.belief.prepare_epoch(
            epoch,
            operational_reference,
            communication,
            dt_s=self.config.dt_s,
            process_covariance_addition_normalized=(
                clock_change_process_covariance
            ),
            process_covariance_addition_audit=(
                clock_change_process_support_audit
            ),
        )
        if install_clock_change_process_support:
            self._installed_clock_change_process_support_alarm_episode_count = int(
                clock_change_process_support_audit["alarm_episode_count"]
            )
        self.multitarget.prepare_epoch(operational_reference, epoch=epoch)
        prior_belief = self.belief.record()
        if self._registered_initial_state_covariance_normalized is None:
            self._registered_initial_state_covariance_normalized = (
                self.belief.current_state_covariance().copy()
            )
            self._registered_initial_position_covariance_m2 = np.asarray(
                prior_belief.position_covariance_m2, dtype=float
            ).copy()
        resources_before = self.world.resource_snapshot(self.world.state)
        self._emit_progress(
            "availability_and_prior_ready",
            epoch=epoch,
            availability_status=availability.status,
            prior_belief_status=prior_belief.status,
        )

        # No RF action may be selected from an invalid posterior.  Previously a
        # failed graph solution was recorded only after ``decision.select`` and
        # ``observation.observe`` had already run, so an indefinite/singular
        # belief could still trigger a physical transmission.  The causal safety
        # invariant is
        #
        #     belief_status != passed  =>  a_k = no-op.
        #
        # We advance the simulator with the no-op solely to represent passage of
        # time, record the failure, and let ``run`` stop at this epoch.
        if (availability.status != "passed" or prior_belief.status != "passed"
                or pre_batch.status != "passed"):
            reason = (
                str(pre_batch.reason)
                if pre_batch.status != "passed"
                else
                "pre_decision_availability_failed"
                if availability.status != "passed"
                else "pre_decision_authoritative_belief_failed"
            )
            if pre_batch.status != "passed":
                self.multitarget.record_receiver_failure(epoch=epoch, reason=reason)
                self.multitarget.attach_offline_metrics(self.world.state.targets)
                self._terminal_runtime_failure = {
                    "status": reason, "component": "receiver_admission", "epoch": epoch,
                    "failure_stage": "before_selection", "further_decisions_allowed": False,
                    "physical_RF_action_executed": False,
                }
            decision = DecisionRecord(
                epoch=epoch,
                candidate_library=((),),
                safe_mask=(True,),
                selected_candidate=0,
                selected_action_ids=(),
                proposed_candidate=-1,
                shield_intervened=True,
                shield_reasons=(reason,),
                exact_rerank_rows=(),
                decision_latency_s=0.0,
                deadline_miss=False,
                fail_closed=True,
                status="failed_closed",
                metadata={
                    "controller_contract_verified": False,
                    "belief_failure_prevented_action": True,
                    "physical_RF_action_executed": False,
                    "failure_reason": reason,
                    "receiver_admission": pre_admission,
                    "modeled_compute_cost_s": 0.0,
                    "innovation_clock_change_process_support": dict(
                        clock_change_process_support_audit
                    ),
                },
            )
            next_state = self.world.transition((), compute_cost_s=0.0)
            self.world_event_history.append(
                {
                    "epoch": int(next_state.epoch),
                    "events_started": tuple(next_state.events_started),
                    "events_active": tuple(next_state.events_active),
                }
            )
            record = RuntimeStepRecord(
                epoch=epoch,
                world_id=self.base_world.world_id,
                availability=availability,
                prior_belief=prior_belief,
                decision=decision,
                observations=(),
                posterior_belief=prior_belief,
                next_epoch=next_state.epoch,
                resources_before=resources_before,
                resources_after=self.world.resource_snapshot(next_state),
                status="failed_closed",
            )
            self.steps.append(record)
            self._emit_progress(
                "epoch_failed_closed",
                epoch=epoch,
                failure_reason=reason,
            )
            return record

        innovation_health_before_views = innovation_health_before_transition
        innovation_reacquisition_active = bool(
            self.config.innovation_health_reacquisition_search_enabled
            and innovation_health_before_views.get("active", False)
        )
        position_search_covariance = None
        observation_search_covariance_normalized = None
        registered_clock_change_covariance_normalized = None
        registered_clock_change_audit: dict[str, Any] = {
            "schema": "registered_differential_clock_change_covariance/v1",
            "status": "not_required",
            "used_only_after_causal_differential_clock_alarm": True,
            "event_epoch_or_realization_read": False,
            "truth_or_future_event_read": False,
        }
        command_search_audit: dict[str, Any] = {
            "schema": "causal_innovation_reacquisition_command_support/v1",
            "status": "not_active",
            "active": False,
            "truth_or_future_event_read": False,
            "single_authoritative_dynamic_factor_graph_preserved": True,
        }
        if innovation_reacquisition_active:
            if (
                self._registered_initial_position_covariance_m2 is None
                or self._registered_initial_state_covariance_normalized is None
            ):
                raise RuntimeError(
                    "initial covariance was not registered before reacquisition"
                )
            if DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS in set(
                innovation_health_before_views.get(
                    "active_alarm_hypotheses", ()
                )
            ):
                (
                    registered_clock_change_covariance_normalized,
                    registered_clock_change_audit,
                ) = _registered_network_clock_step_covariance_normalized(
                    self.belief.layout,
                    rms_s=(
                        self.config.innovation_health_reacquisition_clock_step_rms_s
                    ),
                )
            position_search_covariance, covariance_audit = (
                _causal_reacquisition_covariance_envelope(
                    prior_belief.position_covariance_m2,
                    self._registered_initial_position_covariance_m2,
                    floor_scale=(
                        self.config.innovation_health_reacquisition_covariance_floor_scale
                    ),
                )
            )
            (
                observation_search_covariance_normalized,
                observation_covariance_audit,
            ) = _causal_reacquisition_covariance_envelope(
                self.belief.current_state_covariance(),
                self._registered_initial_state_covariance_normalized,
                floor_scale=(
                    self.config.innovation_health_reacquisition_covariance_floor_scale
                ),
                registered_change_covariance=(
                    registered_clock_change_covariance_normalized
                ),
            )
            command_search_audit = {
                **command_search_audit,
                **covariance_audit,
                "status": "active",
                "active": True,
                "epoch": int(epoch),
                "trigger": "arrived_innovation_health_alarm",
                "health_snapshot": dict(innovation_health_before_views),
                "controller_mean_source": (
                    "authoritative_dynamic_factor_graph_posterior_mean"
                ),
                "receiver_command_support_only": True,
                "registered_clock_change_support": dict(
                    registered_clock_change_audit
                ),
                "full_state_observation_search_covariance": {
                    **observation_covariance_audit,
                    "coordinates": "normalized_dynamic_factor_graph_state",
                    "used_only_for_receiver_search_and_consistency_gating": True,
                    "installed_as_authoritative_posterior": False,
                },
            }
            self._emit_progress(
                "innovation_reacquisition_command_support_ready",
                epoch=epoch,
                search_covariance_trace_m2=float(
                    np.trace(position_search_covariance)
                ),
            )
        views = self.world.views(
            self.belief,
            position_search_covariance_m2=position_search_covariance,
            position_search_covariance_source=(
                "innovation_alarm_loewner_envelope_of_current_and_"
                "registered_initial_graph_covariance"
                if innovation_reacquisition_active
                else None
            ),
        )
        self.leakage_audits.append(dict(views.leakage_audit))
        target_scales = self.belief.layout.scales()[:6]
        target_covariance_normalized = np.asarray(
            self.belief.current_state_covariance(), dtype=float
        )[:6, :6]
        target_covariance_physical = (
            np.diag(target_scales)
            @ target_covariance_normalized
            @ np.diag(target_scales)
        )
        primary_authority_sync = (
            self.multitarget.synchronize_primary_from_authoritative_graph(
                views.controller_world,
                target_covariance_physical,
                epoch=epoch,
            )
        )

        coordination_bundle: Mapping[str, Any] = {}
        if self.coordination_layer is not None:
            coordination_bundle = self.coordination_layer.begin_epoch(
                epoch=epoch,
                base_world=self.base_world,
                truth_state=self.world.state,
                views=views,
                availability_manager=self.availability,
                availability_record=availability,
                prior_belief=prior_belief,
                runtime_config=self.config,
                episode_config=self.episode_config,
            )
            if not isinstance(coordination_bundle, Mapping):
                raise TypeError("coordination layer must return a mapping")

        decision = self.decision.select(
            views.controller_world,
            views.controller_state,
            remaining_epochs=self.config.horizon_steps - epoch - 1,
            candidate_hints=tuple(
                coordination_bundle.get("candidate_hints", ())
            ),
            candidate_hint_context=coordination_bundle.get(
                "candidate_hint_context", {}
            ),
        )
        decision = replace(
            decision,
            metadata={
                **dict(decision.metadata),
                "multi_target_primary_authority_sync": dict(
                    primary_authority_sync
                ),
            },
        )
        self._emit_progress(
            "decision_completed",
            epoch=epoch,
            selected_action_count=len(decision.selected_action_ids),
            decision_status=decision.status,
        )
        if self.coordination_layer is not None:
            self.coordination_layer.after_decision(
                epoch=epoch,
                decision=decision,
                controller_world=views.controller_world,
                controller_state=views.controller_state,
                availability_manager=self.availability,
                authoritative_belief_after_decision=self.belief.record(),
                simulator_epoch_after_decision=int(self.world.state.epoch),
            )
        observations, factors = self.observation.observe(
            views.truth_world,
            views.controller_world,
            decision.selected_action_ids,
            self.belief,
            epoch=epoch,
            belief_covariance_override_normalized=(
                observation_search_covariance_normalized
            ),
        )
        for transport, capture_record, capture_factors in self.observation.take_capture_dispatches():
            try:
                if self.multitarget.enabled:
                    captured = capture_measurement_return_set(
                        views.controller_world, views.truth_world,
                        self.world.state.targets, int(capture_record.action_id),
                        (capture_record,), epoch=epoch,
                        observation_master_seed=self.config.observation_master_seed,
                        controller_reference_mean=(
                            self.multitarget.tracks[0].mean.copy()
                            if self.multitarget.tracks else None),
                        controller_reference_track_id=(
                            self.multitarget.tracks[0].track_id
                            if self.multitarget.tracks else None),
                    )
                    payload = captured.payload
                    self._offline_return_set_capture_history.append({
                        "epoch": epoch, "action_id": int(capture_record.action_id),
                        "offline_capture_audit": dict(captured.offline_capture_audit),
                    })
                else:
                    payload = UnlabelledReturnSet(int(capture_record.action_id), epoch,
                        tuple(capture_record.measurement_names),
                        (() if capture_record.value is None else
                         (tuple(float(v) for v in capture_record.value),)),
                        str(capture_record.provenance_id))
            except (ValueError, TypeError) as exc:
                payload = CapturePayloadFailure(f"{type(exc).__name__}:{exc}")
                self._offline_return_set_capture_history.append({
                    "epoch": epoch, "action_id": int(capture_record.action_id),
                    "offline_capture_failure": payload.offline_reason,
                })
            self._receiver_queue.enqueue(transport, payload, capture_record,
                tuple(capture_factors))
        received_batch = self._receiver_queue.prepare(epoch=epoch)
        self.receiver_admission_history.append(received_batch.audit())
        received_observations = received_batch.observations
        factors = received_batch.factors
        self._emit_progress(
            "observations_realized",
            epoch=epoch,
            observation_count=len(observations),
            admitted_factor_count=len(factors),
        )
        observation_linearization_state = self.belief.current_state_vector().copy()
        observation_prior_covariance = (
            self.belief.current_state_covariance().copy()
        )
        active_records = [
            record_item
            for record_item in received_observations
            if record_item.factor_id is not None
            and record_item.value is not None
            and record_item.residual is not None
            and record_item.arrival_epoch == epoch
            and record_item.metadata.get("mixture_likelihood_contract")
            is not None
        ]
        for factor in factors:
            self.belief.add_factor(factor)
        registered_epoch_acquisition = bool(
            self.config.nonlinear_acquisition_enabled
            and epoch <= self.config.nonlinear_acquisition_max_epoch
        )
        health_triggered_acquisition = bool(
            self.config.nonlinear_acquisition_enabled
            and innovation_reacquisition_active
        )
        acquisition_audit: dict[str, Any] = {
            "enabled": bool(self.config.nonlinear_acquisition_enabled),
            "status": "outside_registered_epoch",
            "epoch": int(epoch),
            "maximum_registered_epoch": int(
                self.config.nonlinear_acquisition_max_epoch
            ),
            "truth_state_used": False,
            "registered_epoch_triggered": registered_epoch_acquisition,
            "innovation_health_triggered": health_triggered_acquisition,
            "trigger": "none",
        }
        if registered_epoch_acquisition or health_triggered_acquisition:
            self._emit_progress(
                "nonlinear_acquisition_started",
                epoch=epoch,
                active_observation_count=len(active_records),
            )
            acquisition_covariance = observation_prior_covariance
            acquisition_covariance_audit: dict[str, Any] | None = None
            acquisition_sigma_override = None
            acquisition_trigger = "registered_initial_epoch"
            if health_triggered_acquisition:
                if self._registered_initial_state_covariance_normalized is None:
                    raise RuntimeError(
                        "initial state covariance was not registered before reacquisition"
                    )
                (
                    acquisition_covariance,
                    acquisition_covariance_audit,
                ) = _causal_reacquisition_covariance_envelope(
                    observation_prior_covariance,
                    self._registered_initial_state_covariance_normalized,
                    floor_scale=(
                        self.config.innovation_health_reacquisition_covariance_floor_scale
                    ),
                    registered_change_covariance=(
                        registered_clock_change_covariance_normalized
                    ),
                )
                acquisition_sigma_override = float(
                    self.config.innovation_health_reacquisition_sigma
                )
                acquisition_trigger = "arrived_innovation_health_alarm"
            _, acquisition_replacements, acquisition_audit = (
                self.observation.select_nonlinear_acquisition_hypothesis(
                    views.controller_world,
                    active_records,
                    initial_state_normalized=observation_linearization_state,
                    prior_covariance_normalized=acquisition_covariance,
                    sigma_override=acquisition_sigma_override,
                )
            )
            acquisition_audit = {
                **dict(acquisition_audit),
                "epoch": int(epoch),
                "maximum_registered_epoch": int(
                    self.config.nonlinear_acquisition_max_epoch
                ),
                "registered_epoch_triggered": registered_epoch_acquisition,
                "innovation_health_triggered": health_triggered_acquisition,
                "trigger": acquisition_trigger,
                "search_covariance_audit": acquisition_covariance_audit,
                "search_covariance_is_authoritative_posterior": bool(
                    not health_triggered_acquisition
                ),
                "search_covariance_source": (
                    "loewner_envelope_of_current_and_registered_initial_graph_covariance"
                    if health_triggered_acquisition
                    else "authoritative_current_graph_covariance"
                ),
                "search_covariance_mutated_live_graph_covariance": False,
            }
            for replacement in acquisition_replacements:
                self.belief.replace_factor(replacement)
            self._emit_progress(
                "nonlinear_acquisition_completed",
                epoch=epoch,
                acquisition_status=str(acquisition_audit.get("status")),
                replacement_factor_count=len(acquisition_replacements),
            )
        decision_metadata = {
            **dict(decision.metadata),
            "innovation_clock_change_process_support": dict(
                clock_change_process_support_audit
            ),
        }
        if innovation_reacquisition_active:
            decision_metadata = {
                    **decision_metadata,
                    "innovation_reacquisition_command_support": dict(
                        command_search_audit
                    ),
                    "innovation_reacquisition_nonlinear_search": dict(
                        acquisition_audit
                    ),
            }
        decision = replace(decision, metadata=decision_metadata)
        posterior_solution = self.belief.solve()
        self._emit_progress(
            "initial_posterior_solve_completed",
            epoch=epoch,
            posterior_status=posterior_solution.status,
        )
        if (
            self.config.likelihood_update_mode.value
            != "legacy_fisher_pseudofactor"
        ):
            iteration_rows: list[dict[str, Any]] = []
            for iteration in range(1, self.config.mixture_relinearization_iterations):
                previous_state = self.belief.current_state_vector().copy()
                previous_covariance = self.belief.current_state_covariance().copy()
                if not active_records:
                    break
                old_factors = {
                    str(record_item.factor_id): self.belief.factor(
                        str(record_item.factor_id)
                    )
                    for record_item in active_records
                }
                accepted = False
                accepted_relative_step = math.inf
                attempts: list[dict[str, Any]] = []
                for step_scale in _line_search_scales(
                    self.config.mixture_line_search_min_scale
                ):
                    replacements = []
                    current_audits = []
                    for record_item in active_records:
                        replacement, audit = (
                            self.observation.relinearize_mixture_factor(
                                views.controller_world,
                                record_item,
                                initial_state_normalized=(
                                    observation_linearization_state
                                ),
                                current_state_normalized=previous_state,
                                current_covariance_normalized=previous_covariance,
                                score_step_scale=step_scale,
                            )
                        )
                        replacements.append(replacement)
                        current_audits.append(audit)
                    current_log_likelihood = float(
                        sum(float(item["log_likelihood"]) for item in current_audits)
                    )
                    attempt: dict[str, Any] = {
                        "score_step_scale": float(step_scale),
                        "current_local_log_likelihood": current_log_likelihood,
                    }
                    try:
                        for replacement in replacements:
                            self.belief.replace_factor(replacement)
                        candidate_solution = self.belief.solve()
                        candidate_state = self.belief.current_state_vector().copy()
                        candidate_audits = []
                        for record_item in active_records:
                            _, audit = self.observation.relinearize_mixture_factor(
                                views.controller_world,
                                record_item,
                                initial_state_normalized=(
                                    observation_linearization_state
                                ),
                                current_state_normalized=candidate_state,
                                # Keep P fixed while comparing the local
                                # observed likelihood along this Fisher step.
                                current_covariance_normalized=previous_covariance,
                                score_step_scale=1.0,
                            )
                            candidate_audits.append(audit)
                        candidate_log_likelihood = float(
                            sum(
                                float(item["log_likelihood"])
                                for item in candidate_audits
                            )
                        )
                        delta = candidate_state - previous_state
                        mahalanobis_sigma = _mahalanobis_step_sigma(
                            delta,
                            previous_covariance,
                        )
                        relative_step = float(
                            np.linalg.norm(delta)
                            / max(np.linalg.norm(previous_state), 1.0)
                        )
                        likelihood_monotone = bool(
                            candidate_log_likelihood
                            + self.config.mixture_log_likelihood_tolerance
                            >= current_log_likelihood
                        )
                        trust_region_passed = bool(
                            mahalanobis_sigma
                            <= self.config.mixture_trust_region_sigma
                        )
                        solution_passed = candidate_solution.status == "passed"
                        accepted = bool(
                            solution_passed
                            and likelihood_monotone
                            and trust_region_passed
                        )
                        attempt.update(
                            {
                                "candidate_local_log_likelihood": (
                                    candidate_log_likelihood
                                ),
                                "local_log_likelihood_change": (
                                    candidate_log_likelihood
                                    - current_log_likelihood
                                ),
                                "mahalanobis_step_sigma": mahalanobis_sigma,
                                "relative_state_step": relative_step,
                                "candidate_graph_status": candidate_solution.status,
                                "likelihood_monotone": likelihood_monotone,
                                "trust_region_passed": trust_region_passed,
                                "accepted": accepted,
                            }
                        )
                        attempts.append(attempt)
                        if accepted:
                            posterior_solution = candidate_solution
                            accepted_relative_step = relative_step
                            break
                    except Exception as error:
                        attempt.update(
                            {
                                "accepted": False,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        attempts.append(attempt)
                    for old_factor in old_factors.values():
                        self.belief.replace_factor(old_factor)
                    posterior_solution = self.belief.solve()
                iteration_rows.append(
                    {
                        "iteration": iteration + 1,
                        "accepted": accepted,
                        "accepted_relative_state_step": (
                            accepted_relative_step if accepted else None
                        ),
                        "attempts": attempts,
                        "all_rejected_attempts_rolled_back": bool(not accepted),
                    }
                )
                self._emit_progress(
                    "mixture_relinearization_iteration_completed",
                    epoch=epoch,
                    iteration=iteration + 1,
                    accepted=accepted,
                )
                if not accepted:
                    break
                if (
                    accepted_relative_step
                    <= self.config.mixture_relinearization_tolerance
                ):
                    break
            self.mixture_iteration_history.append(
                {
                    "epoch": epoch,
                    "nonlinear_acquisition": acquisition_audit,
                    "iterations": iteration_rows,
                }
            )
        posterior_belief = self.belief.record(posterior_solution)
        multi_target_update = (self.multitarget.update_received(
            views.controller_world, received_batch.admitted_sets, epoch=epoch,
        ) if received_batch.status == "passed" else self.multitarget.record_receiver_failure(
            epoch=epoch, reason=received_batch.reason or "UNKNOWN_RECEIVER_ADMISSION"))
        self.multitarget.attach_offline_metrics(self.world.state.targets)
        if not isinstance(decision.metadata, dict):
            raise RuntimeError(
                "decision metadata must remain mutable until the causal "
                "post-observation receipt is serialized"
            )
        multi_target_status = (str(multi_target_update.get("status", "UNKNOWN_MISSING_STATUS"))
            if received_batch.status == "passed" else
            str(received_batch.reason or "UNKNOWN_RECEIVER_ADMISSION"))
        multi_target_update_valid = multi_target_status in {"passed", "disabled"}
        post_observation_health = {
            "status": "passed" if multi_target_update_valid else multi_target_status,
            "component": "multi_target_update",
            "epoch": int(epoch),
            "failure_stage": None if multi_target_update_valid else "after_action_commit_and_observation",
            "committed_action_ids": list(decision.selected_action_ids),
            "physical_commit_rolled_back": False,
            "authoritative_graph_status": str(posterior_belief.status),
            "further_decisions_allowed": bool(multi_target_update_valid),
            "world_transition_completed": False,
            # Do not copy the auxiliary update's truth-only offline metrics
            # into controller-visible decision metadata.
            "failed_action_updates": [
                {"action_id": item.get("action_id"), "status": str(item.get("status")),
                 "reason": item.get("reason")}
                for item in multi_target_update.get("actions", ())
                if item.get("status") not in {"passed", "not_target_sensitive"}
            ],
        }
        decision.metadata["multi_target_update_status"] = multi_target_status
        decision.metadata["receiver_admission"] = received_batch.audit()
        decision.metadata["post_observation_runtime_health"] = post_observation_health
        if not multi_target_update_valid:
            self._terminal_runtime_failure = dict(post_observation_health)
            decision.metadata["decision_observation_memory_update"] = {
                "status": "skipped_runtime_quarantined",
                "reason": multi_target_status,
                "observation_records_preserved": True,
            }
        else:
            self._receiver_queue.commit(received_batch)
            decision_observation_update = self.decision.update_after_observations(
                received_batch,
                epoch=epoch,
            )
            decision.metadata.update(decision_observation_update)

        posterior_controller_world = self.world.controller_world(
            self.belief,
            operational_reference,
        )
        compute_cost = float(decision.metadata["modeled_compute_cost_s"])
        previous_world_epoch = int(self.world.state.epoch)
        next_state = self.world.transition(
            decision.selected_action_ids,
            compute_cost_s=compute_cost,
            committed_actions=tuple(
                action
                for action in views.controller_world.actions
                if action.action_id in decision.selected_action_ids
            ),
        )
        if int(next_state.epoch) != previous_world_epoch + 1:
            raise RuntimeError("world transition did not advance exactly once")
        post_observation_health["world_transition_completed"] = True
        post_observation_health["next_epoch"] = int(next_state.epoch)
        if self._terminal_runtime_failure is not None:
            self._terminal_runtime_failure = dict(post_observation_health)
        self.world_event_history.append(
            {
                "epoch": int(next_state.epoch),
                "events_started": tuple(next_state.events_started),
                "events_active": tuple(next_state.events_active),
            }
        )
        resources_after = self.world.resource_snapshot(next_state)
        component_statuses = (
            availability.status,
            prior_belief.status,
            decision.status,
            posterior_belief.status,
            "passed" if multi_target_update_valid else multi_target_status,
        )
        status = "passed" if all(value == "passed" for value in component_statuses) and next_state.resources.feasible else "failed_closed"
        post_observation_health["further_decisions_allowed"] = status == "passed"
        decision.metadata["runtime_step_status"] = status
        record = RuntimeStepRecord(
            epoch=epoch,
            world_id=self.base_world.world_id,
            availability=availability,
            prior_belief=prior_belief,
            decision=decision,
            observations=observations,
            posterior_belief=posterior_belief,
            next_epoch=next_state.epoch,
            resources_before=resources_before,
            resources_after=resources_after,
            status=status,
        )
        self.steps.append(record)
        self._emit_progress(
            "epoch_completed",
            epoch=epoch,
            runtime_step_status=status,
            next_epoch=int(next_state.epoch),
        )
        if self.coordination_layer is not None:
            self.coordination_layer.finish_epoch(
                epoch=epoch,
                base_world=self.base_world,
                truth_state_before=views.truth_world,
                episode_state_before=views.controller_state,
                simulator_state_before_epoch=previous_world_epoch,
                controller_world_before=views.controller_world,
                posterior_controller_world=posterior_controller_world,
                prior_belief=prior_belief,
                posterior_belief=posterior_belief,
                decision=decision,
                observations=observations,
                next_state=next_state,
                resources_before=resources_before,
                resources_after=resources_after,
                leakage_audit=views.leakage_audit,
                multi_target_report=self.multitarget.report(),
                availability_record=availability,
            )
        return record

    def architecture_audit(self) -> dict[str, Any]:
        """Evaluate the single-authority contracts against live objects."""

        availability_input_audit = self.availability.input_contract_audit()
        checks = {
            "decision_uses_authoritative_belief_object": self.decision.belief is self.belief,
            "decision_uses_authoritative_availability_object": self.decision.availability is self.availability,
            "observation_uses_authoritative_availability_object": self.observation.availability is self.availability,
            "decision_uses_controller_multitarget_posterior": (
                self.decision.multi_target_engine is self.multitarget
            ),
            "multitarget_truth_identity_excluded_from_controller": not bool(
                self.multitarget.report().get(
                    "truth_identity_exposed_to_controller",
                    False,
                )
            ),
            "truth_episode_state_passed_to_availability": bool(
                availability_input_audit["forbidden_episode_state_accepted"]
            ),
            "availability_accepted_non_contract_state": bool(
                availability_input_audit["accepted_non_contract_type"]
            ),
            "out_of_band_fusion_enabled": (
                self.config.fusion_mode is not FusionMode.FUSION_COMPONENT_ONLY
            ),
            "exact_rerank_required": bool(self.config.exact_rerank_required),
            "all_fixed_lag_messages_valid": all(
                item.status in {"passed", "no_expired_variables"}
                for item in self.belief.fixed_lag_marginalizations
            ),
            "selected_actions_respected_availability": all(
                set(step.decision.selected_action_ids).issubset(
                    set(step.availability.available_action_ids)
                )
                for step in self.steps
            ),
            "controller_truth_leakage_flags_clear": all(
                not bool(audit.get("truth_target_read_by_decision_engine"))
                and not bool(audit.get("truth_jammer_read_by_decision_engine"))
                and not bool(
                    audit.get("truth_clock_frequency_delay_read_by_decision_engine")
                )
                and not bool(audit.get("truth_object_identity_shared_with_controller"))
                and bool(audit.get("controller_world_graph_source_verified"))
                and bool(
                    audit.get("controller_world_independent_covariance_forbidden")
                )
                and bool(audit.get("controller_world_oracle_state_forbidden"))
                and bool(audit.get("controller_hidden_jammer_excluded"))
                and bool(audit.get("controller_hidden_adversary_sanitized"))
                and bool(
                    audit.get("controller_hidden_target_mode_and_scattering_sanitized")
                )
                and not bool(audit.get("controller_prior_reference_is_truth_object"))
                and bool(
                    audit.get(
                        "controller_prior_reference_draw_nonzero_for_offline_audit"
                    )
                )
                and bool(audit.get("controller_beam_commands_from_graph_mean"))
                and bool(
                    audit.get(
                        "controller_planning_beam_gain_integrates_graph_covariance"
                    )
                )
                and bool(
                    audit.get(
                        "controller_scattering_uses_registered_class_prior"
                    )
                )
                and bool(
                    audit.get(
                        "controller_external_channel_truth_overrides_absent_or_registered"
                    )
                )
                and audit.get("structural_hidden_field_audit", {}).get("status")
                == "passed"
                for audit in self.leakage_audits
            ),
            "structural_hidden_field_violations": sorted(
                {
                    violation
                    for audit in self.leakage_audits
                    for violation in audit.get(
                        "structural_hidden_field_audit", {}
                    ).get("hidden_field_violations", ())
                }
            ),
            "all_decisions_verified_controller_contract": all(
                bool(step.decision.metadata.get("controller_contract_verified"))
                for step in self.steps
            ),
            "coordination_layer_can_commit_actions": False,
            "coordination_layer_output_is_candidate_hint_only": (
                self.coordination_layer is None
                or all(
                    bool(step.decision.metadata.get("agent_hints_are_proposal_only"))
                    and not bool(
                        step.decision.metadata.get(
                            "agent_hints_can_bypass_exact_rerank", True
                        )
                    )
                    and not bool(
                        step.decision.metadata.get(
                            "agent_hints_can_bypass_campaign_shield", True
                        )
                    )
                    for step in self.steps
                    if step.status == "passed"
                )
            ),
        }
        passed = (
            checks["decision_uses_authoritative_belief_object"]
            and checks["decision_uses_authoritative_availability_object"]
            and checks["observation_uses_authoritative_availability_object"]
            and checks["decision_uses_controller_multitarget_posterior"]
            and checks["multitarget_truth_identity_excluded_from_controller"]
            and not checks["truth_episode_state_passed_to_availability"]
            and not checks["availability_accepted_non_contract_state"]
            and not checks["out_of_band_fusion_enabled"]
            and checks["exact_rerank_required"]
            and checks["all_fixed_lag_messages_valid"]
            and checks["selected_actions_respected_availability"]
            and checks["controller_truth_leakage_flags_clear"]
            and checks["all_decisions_verified_controller_contract"]
            and not checks["coordination_layer_can_commit_actions"]
            and checks["coordination_layer_output_is_candidate_hint_only"]
        )
        return {
            "checks": checks,
            "availability_input_contract": availability_input_audit,
            "status": "passed" if passed else "failed_closed",
        }

    def run(self, *, steps: int | None = None) -> dict[str, Any]:
        count = self.config.horizon_steps if steps is None else int(steps)
        if count <= 0 or count > self.config.horizon_steps:
            raise ValueError("requested run length is outside the configured horizon")
        started = time.perf_counter()
        while (
            len(self.steps) < count
            and self._terminal_runtime_failure is None
            and (not self.steps or self.steps[-1].status == "passed")
        ):
            record = self.step()
            if record.status != "passed":
                break
        elapsed = time.perf_counter() - started
        statuses = [step.status for step in self.steps]
        fail_closed_steps = [step.epoch for step in self.steps if step.status != "passed"]
        selected_actions = [
            list(step.decision.selected_action_ids) for step in self.steps
        ]
        architecture_audit = self.architecture_audit()
        report = {
            "schema": "unified_operational_runtime/v1",
            "architecture": {
                "conceptual_cycle": [
                    "dynamic_world",
                    "arriving_observations",
                    "authoritative_joint_inference",
                    "distributed_operational_availability",
                    "exact_graph_counterfactual_safe_decision",
                    "selected_action_observation_capture",
                    "action_conditioned_world_transition",
                ],
                "causal_step_order": [
                    "observable_topology_snapshot",
                    "fuse_all_factors_arrived_by_epoch",
                    "construct_leakage_free_controller_view",
                    "availability_bounded_candidate_proposal",
                    "exact_graph_rerank_and_campaign_shield",
                    "capture_selected_observations",
                    "fuse_same_epoch_returns_or_queue_delayed_returns",
                    "action_conditioned_world_transition",
                ],
                "authoritative_online_belief": "DynamicFactorGraph",
                "independent_controller_covariance_exists": False,
                "candidate_proposal_can_bypass_exact_rerank": False,
                "selected_action_can_bypass_campaign_shield": False,
                "truth_state_available_to_decision": False,
                "realized_factor_detection_or_association_reweighted_twice": False,
                "legacy_parallel_execution_backends_authoritative": False,
                "nonlinear_global_MAP_or_full_smoothing_claim_allowed": False,
            },
            "architecture_audit": architecture_audit,
            "world": {
                "world_id": self.base_world.world_id,
                "seed": self.base_world.seed,
                "family": self.base_world.family,
                "node_count": len(self.base_world.nodes),
                "action_count": len(self.base_world.actions),
                "maximum_schedule_size": self.base_world.max_schedule_size,
                "effective_delay_profile": self.base_world.effective_delay_profile,
                "time_reference_profile": self.base_world.time_reference_profile,
            },
            "runtime_config": asdict(self.config),
            "episode_config": asdict(self.episode_config),
            "mission": asdict(self.mission),
            "steps": self.steps,
            "leakage_audits": self.leakage_audits,
            "sequential_game_history": self.decision.game_history,
            "multi_target_posterior": self.multitarget.report(),
            "world_event_history": self.world_event_history,
            "mixture_iteration_history": self.mixture_iteration_history,
            "terminal_runtime_failure": self._terminal_runtime_failure,
            "receiver_admission_history": self.receiver_admission_history,
            "offline_return_set_capture_history": self._offline_return_set_capture_history,
            "summary": {
                "requested_steps": count,
                "completed_steps": len(self.steps),
                "selected_action_ids_by_epoch": selected_actions,
                "fail_closed_steps": fail_closed_steps,
                "wall_clock_s": float(elapsed),
                "status": (
                    "passed"
                    if all(value == "passed" for value in statuses)
                    and len(self.steps) == count
                    and architecture_audit["status"] == "passed"
                    else "failed_closed"
                ),
            },
            "runtime_environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
        }
        if self.coordination_layer is not None:
            report["inspectable_episode"] = self.coordination_layer.build_trace(
                runtime=self
            )
        return to_builtin(report)

    def write_report(
        self,
        path: str | Path,
        *,
        steps: int | None = None,
    ) -> dict[str, Any]:
        report = self.run(steps=steps)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return report
