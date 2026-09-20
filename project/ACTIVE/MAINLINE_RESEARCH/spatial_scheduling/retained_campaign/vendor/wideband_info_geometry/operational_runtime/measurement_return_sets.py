from __future__ import annotations

"""Unlabelled measurement packets and the capture-only source generator.

The generator may read simulated targets and draw capture randomness.  The
payload contains only the complete measurement set and its declared schema;
generator labels and audit parameters are returned separately.  Possessing a
payload is not proof of receipt: the real receiver requires a separately
validated per-recipient admission object.
"""

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.dynamic_episode import TargetDynamicState
from wideband_info_geometry.physics_v58.measurement import build_action_factor
from wideband_info_geometry.physics_v58.schema import PhysicalWorld

from .contracts import ObservationRecord
from .covariance_coordinates import CovarianceModelError, gaussian_draw
from .planning_detection_probability import PlanningDetectionProbability


class ReturnSetContractError(ValueError):
    """A return-set schema or use is unsupported, rather than a missed target."""


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ReturnSetContractError(name + "_MUST_BE_INTEGER")
    result = int(value)
    if result < 0:
        raise ReturnSetContractError(name + "_MUST_BE_NONNEGATIVE")
    return result


@dataclass(frozen=True)
class UnlabelledReturnSet:
    action_id: int
    observation_epoch: int
    measurement_names: tuple[str, ...]
    measurements: tuple[tuple[float, ...], ...]
    provenance_id: str
    complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _strict_integer(self.action_id, "ACTION_ID"))
        object.__setattr__(self, "observation_epoch", _strict_integer(self.observation_epoch, "OBSERVATION_EPOCH"))
        if self.complete is not True:
            raise ReturnSetContractError("INCOMPLETE_RETURN_SET")
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ReturnSetContractError("RETURN_SET_PROVENANCE_REQUIRED")
        if not isinstance(self.measurement_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.measurement_names
        ) or len(set(self.measurement_names)) != len(self.measurement_names):
            raise ReturnSetContractError("RETURN_SET_MEASUREMENT_SCHEMA")
        if not isinstance(self.measurements, tuple):
            raise ReturnSetContractError("RETURN_SET_MUST_BE_IMMUTABLE")
        for row in self.measurements:
            if not isinstance(row, tuple) or len(row) != len(self.measurement_names):
                raise ReturnSetContractError("RETURN_SET_ROW_SCHEMA")
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   or not math.isfinite(float(value)) for value in row):
                raise ReturnSetContractError("RETURN_SET_FINITE_REAL_VALUES_REQUIRED")
        if not self.measurement_names and self.measurements:
            raise ReturnSetContractError("NONTARGET_SCHEMA_CANNOT_CONTAIN_RETURNS")


@dataclass(frozen=True)
class CaptureResult:
    payload: UnlabelledReturnSet
    offline_capture_audit: Mapping[str, Any]


@dataclass(frozen=True)
class HypotheticalReturnSet:
    """A planning-branch draw, never a substitute for a real receipt."""

    payload: UnlabelledReturnSet
    predictive_context_id: str

    def __post_init__(self) -> None:
        if type(self.payload) is not UnlabelledReturnSet:
            raise ReturnSetContractError("HYPOTHETICAL_PAYLOAD_TYPE")
        if not isinstance(self.predictive_context_id, str) or not self.predictive_context_id:
            raise ReturnSetContractError("HYPOTHETICAL_CONTEXT_REQUIRED")


def _stable_seed(*parts: object) -> int:
    value = 2166136261
    for byte in "|".join(str(part) for part in parts).encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _target_world(
    world: PhysicalWorld,
    state: np.ndarray | TargetDynamicState,
    *,
    target_id: str,
    mean_rcs_m2: float | None = None,
) -> PhysicalWorld:
    if isinstance(state, TargetDynamicState):
        position = tuple(float(value) for value in state.position_m)
        velocity = tuple(float(value) for value in state.velocity_m_s)
        rcs = float(state.mean_rcs_m2)
    else:
        vector = np.asarray(state, dtype=float).reshape(6)
        position = tuple(float(value) for value in vector[:3])
        velocity = tuple(float(value) for value in vector[3:])
        rcs = float(world.target.mean_rcs_m2 if mean_rcs_m2 is None else mean_rcs_m2)
    return replace(
        world,
        target=replace(
            world.target,
            target_id=target_id,
            position_m=position,
            velocity_m_s=velocity,
            mean_rcs_m2=max(rcs, 1.0e-4),
        ),
    )


def _target_rows(factor) -> np.ndarray:
    jacobian = np.asarray(factor.jacobian_physical, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] < 6 or not np.all(np.isfinite(jacobian)):
        raise CovarianceModelError("TARGET_JACOBIAN_SCHEMA")
    # Factory structural zeros identify absent target dependence. A raw numeric
    # norm threshold has incompatible units and can discard delay sensitivity.
    return np.flatnonzero(np.any(jacobian[:, :6] != 0.0, axis=1))


def capture_measurement_return_set(
    controller_world: PhysicalWorld,
    truth_world: PhysicalWorld,
    truth_targets: Sequence[TargetDynamicState],
    action_id: int,
    primary_observations: Sequence[ObservationRecord],
    *,
    epoch: int,
    observation_master_seed: int,
    planning_probability: PlanningDetectionProbability | None = None,
    controller_reference_mean: np.ndarray | None = None,
    controller_reference_track_id: str | None = None,
    provenance_id: str | None = None,
) -> CaptureResult:
    """Generate a complete unlabelled set once, strictly on the capture side.

    The row-reference factor and the original primary/secondary/clutter factors
    remain explicit constructor calls.  Random keys and draw/shuffle order are
    the original source keys.  For a multi-action compatibility comparison the
    caller must supply the original action's controller reference mean/schema;
    arbitrary degenerate row-schema equivalence is not assumed.
    """

    action_id = _strict_integer(action_id, "ACTION_ID")
    epoch = _strict_integer(epoch, "CAPTURE_EPOCH")
    observation_master_seed = _strict_integer(observation_master_seed, "CAPTURE_SEED")
    action = next(item for item in controller_world.actions if item.action_id == action_id)
    reference_mean = (np.r_[controller_world.target.position_m, controller_world.target.velocity_m_s]
                      if controller_reference_mean is None else controller_reference_mean)
    reference_id = (controller_world.target.target_id if controller_reference_track_id is None
                    else controller_reference_track_id)
    row_factor = build_action_factor(
        _target_world(controller_world, reference_mean, target_id=reference_id), action, None)
    rows = _target_rows(row_factor)
    names = tuple(row_factor.measurement_names[index] for index in rows)
    matching = [record for record in primary_observations if record.action_id == action_id]
    if provenance_id is None:
        provenance_id = (matching[0].provenance_id if matching
                         else f"capture_return_set:{truth_world.world_id}:{epoch}:{action_id}")
    if rows.size == 0:
        return CaptureResult(
            UnlabelledReturnSet(action_id, epoch, (), (), provenance_id),
            {"status": "not_target_sensitive", "primary_runtime": 0,
             "secondary_truth_capture": 0, "clutter": 0,
             "truth_labels_removed_before_association": True})

    measurements: list[np.ndarray] = []
    source_counts = {"primary_runtime": 0, "secondary_truth_capture": 0, "clutter": 0}
    for record in primary_observations:
        if (
            record.action_id == action.action_id
            and record.detected
            and record.value is not None
            and rows.size > 0
            and len(record.value) > int(np.max(rows))
        ):
            value = np.asarray(record.value, dtype=float)
            measurements.append(value[rows].copy())
            source_counts["primary_runtime"] += 1
            break
    jammer = truth_world.jammers[0] if truth_world.jammers else None
    primary_id = truth_world.target.target_id
    primary_truth_factor = build_action_factor(truth_world, action, jammer)
    expected_false_returns = float(primary_truth_factor.link_budget.expected_false_returns)
    association_search_cells = int(primary_truth_factor.link_budget.association_search_cells)
    reference_factor = primary_truth_factor
    for target in truth_targets:
        if not target.exists or target.target_id == primary_id:
            continue
        target_world = _target_world(truth_world, target, target_id=target.target_id)
        factor = build_action_factor(target_world, action, jammer)
        target_rows = _target_rows(factor)
        if target_rows.size != rows.size or target_rows.size == 0:
            continue
        rng = np.random.default_rng(_stable_seed(
            observation_master_seed, epoch, action.action_id, target.target_id))
        expected_false_returns = max(expected_false_returns, float(factor.link_budget.expected_false_returns))
        reference_factor = factor
        detection_probability = (
            float(factor.link_budget.detection_probability)
            if planning_probability is None
            else planning_probability.evaluate(action, factor).planning_success_probability)
        if rng.random() <= detection_probability:
            covariance = np.asarray(factor.covariance, dtype=float)[np.ix_(target_rows, target_rows)]
            draw = gaussian_draw(rng, np.asarray(factor.prediction, dtype=float)[target_rows], covariance)
            measurements.append(draw)
            source_counts["secondary_truth_capture"] += 1
    if reference_factor is not None:
        rng = np.random.default_rng(_stable_seed(observation_master_seed, epoch, action.action_id, "clutter"))
        clutter_count = min(int(rng.poisson(min(expected_false_returns, 2.0))), 3)
        reference_rows = _target_rows(reference_factor)
        covariance = np.asarray(reference_factor.covariance, dtype=float)[np.ix_(reference_rows, reference_rows)]
        center = (np.asarray(measurements[0], dtype=float) if measurements
                  else np.asarray(reference_factor.prediction, dtype=float)[reference_rows])
        for _ in range(clutter_count):
            measurements.append(gaussian_draw(rng, center, 16.0 * covariance))
            source_counts["clutter"] += 1
    rng = np.random.default_rng(_stable_seed(observation_master_seed, epoch, action.action_id, "shuffle"))
    rng.shuffle(measurements)
    payload = UnlabelledReturnSet(
        action_id, epoch, names,
        tuple(tuple(float(value) for value in np.asarray(row, dtype=float)) for row in measurements),
        provenance_id)
    return CaptureResult(payload, {
        **source_counts, "expected_false_returns": float(expected_false_returns),
        "association_search_cells": int(association_search_cells),
        "truth_labels_removed_before_association": True})
