from __future__ import annotations

"""Hash-bound online adapter for the R30 schedule-update calibration.

The adapter consumes only a frozen development checkpoint.  It never observes
the realized update event.  A point multiplier is available to the soft
planning objective, whereas the hard chance-admissibility path receives the
registered one-sided lower confidence multiplier.  Unsupported phases and
out-of-support native probabilities remain explicit and receive zero
admissibility probability.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_SCHEMA = "v75-r30-schedule-update-calibration-checkpoint-1.1"
MODEL_TYPE = "phase_specific_common_cause_success_multiplier"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibratedScheduleUpdateProbability:
    phase: str
    estimand: str | None
    native_probability: float
    objective_probability: float
    admissibility_probability: float
    point_multiplier: float
    admissibility_multiplier: float
    calibration_enabled: bool
    runtime_phase_supported: bool
    native_probability_inside_training_support: bool
    objective_calibration_applied: bool
    admissibility_calibration_applied: bool
    reason: str
    checkpoint_sha256: str | None

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


class ScheduleUpdateProbabilityCalibration:
    """Immutable phase-specific common-cause probability calibration."""

    def __init__(
        self,
        *,
        enabled: bool,
        checkpoint: Mapping[str, Any] | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.checkpoint = dict(checkpoint or {})
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def disabled(cls) -> "ScheduleUpdateProbabilityCalibration":
        return cls(enabled=False)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        expected_sha256: str,
    ) -> "ScheduleUpdateProbabilityCalibration":
        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        expected = str(expected_sha256).strip().lower()
        observed = _sha256(checkpoint_path)
        if observed != expected:
            raise ValueError(
                "schedule-update calibration checkpoint SHA-256 mismatch"
            )
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        cls._validate_checkpoint(payload)
        return cls(
            enabled=True,
            checkpoint=payload,
            checkpoint_sha256=observed,
        )

    @staticmethod
    def _validate_checkpoint(payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported schedule-update calibration schema")
        if not str(payload.get("status", "")).startswith("PASS_"):
            raise ValueError("schedule-update calibration checkpoint did not pass")
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError("unsupported schedule-update calibration model")
        if payload.get("mathematical_map") != "q_e(p)=lambda_e*p":
            raise ValueError("schedule-update calibration map is not registered")
        boundary = payload.get("claim_boundary", {})
        if not isinstance(boundary, Mapping) or not bool(
            boundary.get("development_calibration_only", False)
        ):
            raise ValueError("calibration checkpoint omits its development boundary")
        if any(
            bool(boundary.get(key, True))
            for key in (
                "fresh_world_tracking_benefit_supported",
                "population_calibration_supported",
                "hardware_or_field_claim_allowed",
            )
        ):
            raise ValueError("calibration checkpoint exceeds its evidence scope")
        phase_map = payload.get("estimand_by_track_phase")
        estimands = payload.get("estimands")
        enabled_phases = payload.get("runtime_enabled_phases")
        blocked_phases = payload.get("runtime_blocked_phases")
        if not all(
            isinstance(value, Mapping)
            for value in (phase_map, estimands)
        ) or not all(
            isinstance(value, list)
            for value in (enabled_phases, blocked_phases)
        ):
            raise ValueError("calibration phase contract is incomplete")
        phases = set(str(value) for value in phase_map)
        if set(map(str, enabled_phases)) | set(map(str, blocked_phases)) != phases:
            raise ValueError("calibration enabled/blocked phase partition is invalid")
        if set(map(str, enabled_phases)) & set(map(str, blocked_phases)):
            raise ValueError("calibration phase partition overlaps")
        for phase, estimand_name in phase_map.items():
            record = estimands.get(str(estimand_name))
            if not isinstance(record, Mapping):
                raise ValueError(f"calibration estimand is missing for phase {phase}")
            runtime_enabled = bool(record.get("runtime_enabled", False))
            if runtime_enabled != (str(phase) in set(map(str, enabled_phases))):
                raise ValueError("calibration phase and estimand authorization disagree")
            point = float(record.get("common_cause_success_multiplier_point", math.nan))
            lower = float(
                record.get(
                    "common_cause_success_multiplier_lcb_95_one_sided",
                    math.nan,
                )
            )
            support = record.get("native_probability_training_range")
            if (
                not math.isfinite(point)
                or not math.isfinite(lower)
                or not 0.0 <= lower <= point <= 1.0
                or not isinstance(support, list)
                or len(support) != 2
                or not 0.0 <= float(support[0]) <= float(support[1]) <= 1.0
            ):
                raise ValueError("calibration multiplier or support is invalid")

    def evaluate(
        self,
        phase: str,
        native_probability: float,
    ) -> CalibratedScheduleUpdateProbability:
        phase = str(phase)
        native = float(native_probability)
        if not math.isfinite(native) or not 0.0 <= native <= 1.0:
            raise ValueError("native schedule-update probability is outside [0,1]")
        if not self.enabled:
            return CalibratedScheduleUpdateProbability(
                phase=phase,
                estimand=None,
                native_probability=native,
                objective_probability=native,
                admissibility_probability=native,
                point_multiplier=1.0,
                admissibility_multiplier=1.0,
                calibration_enabled=False,
                runtime_phase_supported=True,
                native_probability_inside_training_support=True,
                objective_calibration_applied=False,
                admissibility_calibration_applied=False,
                reason="calibration_disabled_identity",
                checkpoint_sha256=None,
            )
        phase_map = self.checkpoint["estimand_by_track_phase"]
        if phase not in phase_map:
            raise ValueError(f"unknown track phase for calibration: {phase}")
        estimand = str(phase_map[phase])
        record = self.checkpoint["estimands"][estimand]
        supported = bool(record["runtime_enabled"])
        point = float(record["common_cause_success_multiplier_point"])
        lower = float(
            record["common_cause_success_multiplier_lcb_95_one_sided"]
        )
        support_low, support_high = (
            float(value) for value in record["native_probability_training_range"]
        )
        tolerance = 128.0 * math.ulp(1.0)
        inside_support = bool(
            support_low - tolerance <= native <= support_high + tolerance
        )
        objective_applied = bool(supported and inside_support)
        admissibility_applied = bool(supported and inside_support)
        return CalibratedScheduleUpdateProbability(
            phase=phase,
            estimand=estimand,
            native_probability=native,
            objective_probability=(point * native if objective_applied else native),
            admissibility_probability=(lower * native if admissibility_applied else 0.0),
            point_multiplier=point,
            admissibility_multiplier=lower,
            calibration_enabled=True,
            runtime_phase_supported=supported,
            native_probability_inside_training_support=inside_support,
            objective_calibration_applied=objective_applied,
            admissibility_calibration_applied=admissibility_applied,
            reason=(
                "phase_supported_inside_training_probability_range"
                if admissibility_applied
                else (
                    "phase_supported_but_native_probability_outside_training_range"
                    if supported
                    else "phase_not_authorized_by_frozen_checkpoint"
                )
            ),
            checkpoint_sha256=self.checkpoint_sha256,
        )
