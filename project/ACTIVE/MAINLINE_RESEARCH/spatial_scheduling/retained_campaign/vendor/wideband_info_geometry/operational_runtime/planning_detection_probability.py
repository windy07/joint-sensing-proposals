from __future__ import annotations

"""Truth-isolated controller probability for finite detection outcomes.

The link budget's probability is the nominal temporal target-cell event.  The
R18p6 frozen model additionally represents the spatial AOA acceptance event.
This adapter changes only controller forecasts for AOA actions.  It never
changes the observation kernel, simulator random draw, or realized factor.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wideband_info_geometry.experiments_v75_part2.aoa_joint_probability import (
    JointAOAProbabilityModel,
    load_joint_aoa_probability_model,
)
from wideband_info_geometry.physics_v58.effective_delay import (
    automatic_free_calibration_actions,
)
from wideband_info_geometry.physics_v58.measurement import build_action_factor
from wideband_info_geometry.physics_v58.schema import PhysicalWorld, RadarAction

from .contracts import PlanningDetectionProbabilityMode, RuntimeConfig


@dataclass(frozen=True)
class PlanningProbabilityRecord:
    action_id: int
    action_kind: str
    nominal_temporal_probability: float
    planning_success_probability: float
    probability_semantics: str
    frozen_model_applied: bool
    frozen_model_sha256: str | None
    calibrated_temporal_probability: float | None = None
    spatial_given_temporal_probability: float | None = None
    commanded_angular_coverage_frechet_lower: float | None = None
    commanded_delay_gate_coverage_frechet_lower: float | None = None
    commanded_joint_support_coverage_frechet_lower: float | None = None
    commanded_support_coverage_applied_to_planning_probability: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "action_kind": self.action_kind,
            "nominal_temporal_probability": float(
                self.nominal_temporal_probability
            ),
            "planning_success_probability": float(
                self.planning_success_probability
            ),
            "probability_semantics": self.probability_semantics,
            "frozen_model_applied": bool(self.frozen_model_applied),
            "frozen_model_sha256": self.frozen_model_sha256,
            "calibrated_temporal_probability": (
                None
                if self.calibrated_temporal_probability is None
                else float(self.calibrated_temporal_probability)
            ),
            "spatial_given_temporal_probability": (
                None
                if self.spatial_given_temporal_probability is None
                else float(self.spatial_given_temporal_probability)
            ),
            "commanded_angular_coverage_frechet_lower": (
                None
                if self.commanded_angular_coverage_frechet_lower is None
                else float(self.commanded_angular_coverage_frechet_lower)
            ),
            "commanded_delay_gate_coverage_frechet_lower": (
                None
                if self.commanded_delay_gate_coverage_frechet_lower is None
                else float(self.commanded_delay_gate_coverage_frechet_lower)
            ),
            "commanded_joint_support_coverage_frechet_lower": (
                None
                if self.commanded_joint_support_coverage_frechet_lower is None
                else float(self.commanded_joint_support_coverage_frechet_lower)
            ),
            "commanded_support_coverage_applied_to_planning_probability": bool(
                self.commanded_support_coverage_applied_to_planning_probability
            ),
        }


def _frechet_intersection_lower_bound(
    values: Mapping[str, Any] | None,
) -> float | None:
    if not values:
        return None
    probabilities = np.asarray(
        [float(value) for value in values.values()], dtype=float
    )
    if (
        probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("commanded support coverage probabilities are invalid")
    return float(
        np.clip(
            float(np.sum(probabilities)) - (probabilities.size - 1),
            0.0,
            1.0,
        )
    )


def _commanded_support_coverage_bounds(
    action: RadarAction,
) -> tuple[float | None, float | None, float | None]:
    angular = _frechet_intersection_lower_bound(
        action.metadata.get(
            "commanded_beam_angular_coverage_probability_by_node"
        )
    )
    delay = _frechet_intersection_lower_bound(
        action.metadata.get(
            "planning_capture_gate_coverage_probability_by_receiver"
        )
    )
    if angular is None or delay is None:
        joint = None
    else:
        joint = float(np.clip(angular + delay - 1.0, 0.0, 1.0))
    return angular, delay, joint


class PlanningDetectionProbability:
    """Hash-bound planning probability with a narrow AOA applicability scope."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.mode = config.planning_detection_probability_mode
        self.model: JointAOAProbabilityModel | None = None
        self.model_receipt: Mapping[str, Any] | None = None
        if self.mode is PlanningDetectionProbabilityMode.FROZEN_JOINT_AOA:
            path = Path(str(config.joint_aoa_probability_model_path)).expanduser()
            model, receipt = load_joint_aoa_probability_model(
                path.resolve(),
                expected_sha256=str(config.joint_aoa_probability_model_sha256),
            )
            self.model = model
            self.model_receipt = receipt

    @property
    def receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "frozen_model_loaded": self.model is not None,
            "frozen_model_receipt": (
                None if self.model_receipt is None else dict(self.model_receipt)
            ),
            "applicable_action_kinds": ["aoa"],
            "non_aoa_actions_retain_native_probability": True,
            "changes_realized_observation_kernel": False,
            "truth_fields_used": False,
            "records_commanded_support_coverage_bounds": True,
            "commanded_support_coverage_applied_to_planning_probability": False,
        }

    def evaluate(self, action: RadarAction, action_factor: Any) -> PlanningProbabilityRecord:
        nominal = float(
            np.clip(action_factor.link_budget.detection_probability, 0.0, 1.0)
        )
        angular_coverage, delay_coverage, joint_support_coverage = (
            _commanded_support_coverage_bounds(action)
        )
        if action.kind.startswith("calibration"):
            return PlanningProbabilityRecord(
                action_id=action.action_id,
                action_kind=action.kind,
                nominal_temporal_probability=1.0,
                planning_success_probability=1.0,
                probability_semantics="delivered_calibration_not_target_gated",
                frozen_model_applied=False,
                frozen_model_sha256=None,
                commanded_angular_coverage_frechet_lower=angular_coverage,
                commanded_delay_gate_coverage_frechet_lower=delay_coverage,
                commanded_joint_support_coverage_frechet_lower=(
                    joint_support_coverage
                ),
            )
        if self.model is None or action.kind != "aoa":
            return PlanningProbabilityRecord(
                action_id=action.action_id,
                action_kind=action.kind,
                nominal_temporal_probability=nominal,
                planning_success_probability=nominal,
                probability_semantics="native_link_budget_detection_probability",
                frozen_model_applied=False,
                frozen_model_sha256=None,
                commanded_angular_coverage_frechet_lower=angular_coverage,
                commanded_delay_gate_coverage_frechet_lower=delay_coverage,
                commanded_joint_support_coverage_frechet_lower=(
                    joint_support_coverage
                ),
            )
        components = self.model.predict(nominal)
        model_hash = str(dict(self.model_receipt or {}).get("sha256", ""))
        return PlanningProbabilityRecord(
            action_id=action.action_id,
            action_kind=action.kind,
            nominal_temporal_probability=nominal,
            planning_success_probability=float(
                components["joint_aoa_success_probability"]
            ),
            probability_semantics=(
                "frozen_P_temporal_CFAR_times_P_spatial_AOA_accept_given_temporal"
            ),
            frozen_model_applied=True,
            frozen_model_sha256=model_hash,
            calibrated_temporal_probability=float(
                components["calibrated_temporal_cfar_probability"]
            ),
            spatial_given_temporal_probability=float(
                components["spatial_acceptance_probability_given_temporal"]
            ),
            commanded_angular_coverage_frechet_lower=angular_coverage,
            commanded_delay_gate_coverage_frechet_lower=delay_coverage,
            commanded_joint_support_coverage_frechet_lower=(
                joint_support_coverage
            ),
        )

    def expected_observed_information(
        self,
        world: PhysicalWorld,
        action: RadarAction,
        action_factor: Any,
    ) -> np.ndarray:
        """Return expected information under the selected planning event law.

        Automatic calibration information is not target-detection gated.  It is
        therefore separated before replacing the AOA success probability.
        """

        record = self.evaluate(action, action_factor)
        observed = np.asarray(
            action_factor.observed_information_normalized, dtype=float
        )
        if not record.frozen_model_applied:
            return observed.copy()
        conditional = action_factor.conditional_detected_information_normalized
        if conditional is None:
            raise RuntimeError("AOA factor omitted conditional detected information")
        calibration_information = np.zeros_like(observed)
        for calibration_action in automatic_free_calibration_actions(world, action):
            calibration_information += np.asarray(
                build_action_factor(
                    world, calibration_action, None
                ).observed_information_normalized,
                dtype=float,
            )
        target_conditional = 0.5 * (
            np.asarray(conditional, dtype=float)
            - calibration_information
            + (
                np.asarray(conditional, dtype=float)
                - calibration_information
            ).T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(target_conditional)
        tolerance = 1.0e-9 * max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) < -tolerance:
            raise RuntimeError(
                "AOA conditional target information is materially indefinite"
            )
        target_conditional = (
            eigenvectors * np.maximum(eigenvalues, 0.0)
        ) @ eigenvectors.T
        result = calibration_information + (
            record.planning_success_probability * target_conditional
        )
        return 0.5 * (result + result.T)
