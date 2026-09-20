from __future__ import annotations

"""Anytime-valid, truth-isolated innovation health for the radar controller."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS = (
    "differential_clock_or_cross_node_timing_inconsistency"
)


def _causal_alarm_hypotheses(tests: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Infer bounded nuisance hypotheses from arrived receiver tests only.

    A target-dependent cross-node return requires a consistent monostatic
    anchor in the same arrived epoch, because target-state error is otherwise a
    competing explanation.  A registered ``calibration_time`` transfer has no
    target-state Jacobian, so its alarm alone identifies a bounded cross-node
    timing inconsistency.  Neither path asserts that oscillator error is the
    unique physical cause; receiver-chain and calibration-path changes remain
    inside the hypothesis.
    """

    target_dependent_clock_alarm = any(
        bool(item.get("alarm", False))
        and str(item.get("action_kind", "")) in {"active_bistatic", "tdoa"}
        for item in tests
    )
    target_independent_time_transfer_alarm = any(
        bool(item.get("alarm", False))
        and str(item.get("action_kind", "")) == "calibration_time"
        for item in tests
    )
    monostatic_consistent = any(
        not bool(item.get("alarm", False))
        and str(item.get("action_kind", "")) == "active_monostatic"
        and bool(item.get("detected_after_receiver_gate", False))
        for item in tests
    )
    return (
        (DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS,)
        if target_independent_time_transfer_alarm
        or (target_dependent_clock_alarm and monostatic_consistent)
        else ()
    )


def anytime_epoch_alpha(alpha: float, epoch: int) -> float:
    r"""Spend an episode error budget over an unbounded sequence of epochs.

    ``alpha_t = alpha / ((t + 1)(t + 2))`` and the telescoping identity
    ``1/((t+1)(t+2)) = 1/(t+1) - 1/(t+2)`` imply
    ``sum_t alpha_t = alpha``.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("episode alpha must lie in (0,1)")
    if int(epoch) != epoch or int(epoch) < 0:
        raise ValueError("epoch must be a nonnegative integer")
    t = int(epoch)
    return float(alpha) / float((t + 1) * (t + 2))


def laurent_massart_upper_threshold(
    dimension: int,
    *,
    alpha: float,
    simultaneous_tests: int,
) -> float:
    r"""Return a Bonferroni/Laurent--Massart chi-square upper threshold."""

    d = int(dimension)
    m = int(simultaneous_tests)
    if d <= 0:
        raise ValueError("innovation dimension must be positive")
    if m <= 0:
        raise ValueError("simultaneous test count must be positive")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("test alpha must lie in (0,1)")
    x = math.log(float(m) / float(alpha))
    return float(d + 2.0 * math.sqrt(d * x) + 2.0 * x)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(observation: Any) -> Mapping[str, Any]:
    value = _field(observation, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def receiver_innovation_test(
    observation: Any,
) -> tuple[float, int, str] | None:
    """Extract one causal NIS test, or ``None`` when it is undefined."""

    metadata = _metadata(observation)
    receiver_norm = metadata.get("r2_whitened_residual_norm")
    if receiver_norm is not None:
        norm = float(receiver_norm)
        if not math.isfinite(norm):
            return None
        covariance = metadata.get("r2_residual_gate_covariance")
        if covariance is None:
            dimension = len(_field(observation, "measurement_names", ()) or ())
        else:
            matrix = np.asarray(covariance, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("R2 residual-gate covariance must be square")
            dimension = int(matrix.shape[0])
        return float(norm * norm), dimension, "receiver_R2_pre_admission_gate"

    residual = _field(observation, "residual")
    covariance = metadata.get("innovation_covariance")
    if residual is None or covariance is None:
        return None
    vector = np.asarray(residual, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("innovation covariance and residual dimensions disagree")
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    value = float(vector @ np.linalg.solve(matrix, vector))
    return value, int(vector.size), "admitted_residual_and_predictive_covariance"


def receiver_statistic_is_nonfinite(observation: Any) -> bool:
    value = _metadata(observation).get("r2_whitened_residual_norm")
    return value is not None and not math.isfinite(float(value))


@dataclass
class InnovationHealthMonitor:
    """Stateful causal health flag consumed by the next decision cycle."""

    enabled: bool = False
    episode_alpha: float = 0.01
    minimum_detected_actions_to_clear: int = 2
    minimum_consistent_verification_epochs_to_clear: int = 1
    active: bool = False
    first_alarm_epoch: int | None = None
    latest_epoch: int | None = None
    latest_maximum_threshold_ratio: float = 0.0
    alarm_count: int = 0
    # Number of inactive-to-active alarm transitions.  ``alarm_count`` above
    # remains the historical count of alarmed epochs; this separate counter is
    # the causal key for interventions that must occur once per alarm episode.
    alarm_episode_count: int = 0
    nonfinite_omission_count: int = 0
    consecutive_consistent_verification_epochs: int = 0
    active_alarm_hypotheses: tuple[str, ...] = ()
    last_alarm_hypotheses: tuple[str, ...] = ()
    last_verification_evidence_complete: bool = False
    last_verification_evidence_receipt: dict[str, Any] = field(default_factory=dict)
    last_verified_clear_epoch: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("innovation health enabled flag must be boolean")
        if not 0.0 < float(self.episode_alpha) < 1.0:
            raise ValueError("innovation health episode alpha must lie in (0,1)")
        if int(self.minimum_detected_actions_to_clear) < 1:
            raise ValueError("innovation health clear count must be positive")
        if int(self.minimum_consistent_verification_epochs_to_clear) < 1:
            raise ValueError(
                "innovation health consistent verification epoch count must be positive"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "active": bool(self.active),
            "episode_alpha": float(self.episode_alpha),
            "minimum_detected_actions_to_clear": int(
                self.minimum_detected_actions_to_clear
            ),
            "minimum_consistent_verification_epochs_to_clear": int(
                self.minimum_consistent_verification_epochs_to_clear
            ),
            "consecutive_consistent_verification_epochs": int(
                self.consecutive_consistent_verification_epochs
            ),
            "first_alarm_epoch": self.first_alarm_epoch,
            "latest_epoch": self.latest_epoch,
            "latest_maximum_threshold_ratio": float(
                self.latest_maximum_threshold_ratio
            ),
            "alarm_count": int(self.alarm_count),
            "alarm_episode_count": int(self.alarm_episode_count),
            "nonfinite_omission_count": int(self.nonfinite_omission_count),
            "active_alarm_hypotheses": list(self.active_alarm_hypotheses),
            "last_alarm_hypotheses": list(self.last_alarm_hypotheses),
            "last_verification_evidence_complete": bool(
                self.last_verification_evidence_complete
            ),
            "last_verification_evidence_receipt": dict(
                self.last_verification_evidence_receipt
            ),
            "last_verified_clear_epoch": self.last_verified_clear_epoch,
            "differential_clock_timing_inconsistency_suspected": bool(
                DIFFERENTIAL_CLOCK_TIMING_HYPOTHESIS
                in self.active_alarm_hypotheses
            ),
            "alarm_hypothesis_semantics": (
                "causal_receiver_model_discrimination_not_truth_or_unique_cause"
            ),
            "history_count": len(self.history),
            "truth_or_future_event_read": False,
            "time_uniform_false_alarm_bound": (
                "alpha_t=alpha/((t+1)(t+2)); Bonferroni across receiver tests; "
                "Laurent-Massart chi-square upper bound"
            ),
        }

    def update(
        self,
        observations: Sequence[Any],
        *,
        epoch: int,
        action_kind_by_id: Mapping[int, str],
        verification_committed: bool,
        verification_evidence_complete: bool = False,
        verification_evidence_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("innovation health epoch must be nonnegative")
        if self.latest_epoch is not None and epoch <= self.latest_epoch:
            raise ValueError("innovation health epochs must be strictly increasing")
        active_before = bool(self.active)
        if not self.enabled:
            self.latest_epoch = epoch
            row = {
                **self.snapshot(),
                "epoch": epoch,
                "status": "disabled",
                "active_before": active_before,
                "active_after": False,
                "alarm": False,
                "verification_committed": bool(verification_committed),
                "verification_evidence_complete": bool(
                    verification_evidence_complete
                ),
                "verification_evidence_receipt": dict(
                    verification_evidence_receipt or {}
                ),
                "tests": [],
            }
            self.history.append(row)
            return row

        innovation_inputs = []
        nonfinite = 0
        for observation in observations:
            action_id = int(_field(observation, "action_id", -1))
            kind = str(action_kind_by_id.get(action_id, ""))
            # A time-transfer pilot is a direct nuisance observation and must
            # participate in health monitoring.  Frequency calibration targets
            # a different nuisance and remains outside this timing alarm.
            if kind.startswith("calibration") and kind != "calibration_time":
                continue
            if receiver_statistic_is_nonfinite(observation):
                nonfinite += 1
                continue
            test = receiver_innovation_test(observation)
            if test is not None:
                innovation_inputs.append((observation, test))
        epoch_alpha = anytime_epoch_alpha(self.episode_alpha, epoch)
        tests = []
        for observation, (statistic, dimension, source) in innovation_inputs:
            action_id = int(_field(observation, "action_id", -1))
            action_kind = str(action_kind_by_id.get(action_id, ""))
            threshold = laurent_massart_upper_threshold(
                dimension,
                alpha=epoch_alpha,
                simultaneous_tests=len(innovation_inputs),
            )
            tests.append(
                {
                    "action_id": action_id,
                    "action_kind": action_kind,
                    "measurement_names": [
                        str(value)
                        for value in (_field(observation, "measurement_names", ()) or ())
                    ],
                    "innovation_chi_square": float(statistic),
                    "degrees_of_freedom": int(dimension),
                    "threshold": float(threshold),
                    "threshold_ratio": float(statistic / max(threshold, 1.0e-15)),
                    "alarm": bool(statistic > threshold),
                    "detected_after_receiver_gate": bool(
                        _field(observation, "detected", False)
                    ),
                    "target_sensing_observation": not action_kind.startswith(
                        "calibration"
                    ),
                    "target_independent_timing_probe": bool(
                        action_kind == "calibration_time"
                    ),
                    "source": source,
                }
            )
        alarm = any(bool(item["alarm"]) for item in tests)
        admitted_detection_count = sum(
            bool(item["detected_after_receiver_gate"])
            and bool(item["target_sensing_observation"])
            for item in tests
        )
        admitted_time_calibration_count = sum(
            bool(item["detected_after_receiver_gate"])
            and bool(item["target_independent_timing_probe"])
            for item in tests
        )
        cleared = False
        alarm_hypotheses = _causal_alarm_hypotheses(tests) if alarm else ()
        cleared_alarm_hypotheses: tuple[str, ...] = ()
        if alarm:
            if not active_before:
                self.alarm_episode_count += 1
            self.active = True
            self.active_alarm_hypotheses = tuple(
                sorted(
                    {
                        *self.active_alarm_hypotheses,
                        *alarm_hypotheses,
                    }
                )
            )
            self.last_alarm_hypotheses = self.active_alarm_hypotheses
            self.consecutive_consistent_verification_epochs = 0
            self.last_verification_evidence_complete = False
            self.last_verification_evidence_receipt = {}
            self.alarm_count += 1
            if self.first_alarm_epoch is None:
                self.first_alarm_epoch = epoch
        elif self.active:
            verification_consistent = bool(
                verification_committed
                and verification_evidence_complete
                and admitted_detection_count
                >= int(self.minimum_detected_actions_to_clear)
            )
            self.last_verification_evidence_complete = bool(
                verification_evidence_complete
            )
            self.last_verification_evidence_receipt = dict(
                verification_evidence_receipt or {}
            )
            if verification_consistent:
                self.consecutive_consistent_verification_epochs += 1
            else:
                self.consecutive_consistent_verification_epochs = 0
            if (
                self.consecutive_consistent_verification_epochs
                >= int(self.minimum_consistent_verification_epochs_to_clear)
            ):
                cleared_alarm_hypotheses = self.active_alarm_hypotheses
                self.active = False
                self.active_alarm_hypotheses = ()
                self.last_verified_clear_epoch = epoch
                cleared = True
        else:
            self.consecutive_consistent_verification_epochs = 0
            self.active_alarm_hypotheses = ()
        maximum_ratio = max(
            (float(item["threshold_ratio"]) for item in tests), default=0.0
        )
        self.latest_epoch = epoch
        self.latest_maximum_threshold_ratio = maximum_ratio
        self.nonfinite_omission_count += nonfinite
        row = {
            **self.snapshot(),
            "epoch": epoch,
            "status": "passed",
            "epoch_alpha_spend": float(epoch_alpha),
            "active_before": active_before,
            "active_after": bool(self.active),
            "alarm": bool(alarm),
            "alarm_hypotheses_this_epoch": list(alarm_hypotheses),
            "cleared_alarm_hypotheses": list(cleared_alarm_hypotheses),
            "cleared_after_verified_consistent_update": bool(cleared),
            "verification_committed": bool(verification_committed),
            "verification_evidence_complete": bool(
                verification_evidence_complete
            ),
            "verification_evidence_receipt": dict(
                verification_evidence_receipt or {}
            ),
            "verification_epoch_consistent": bool(
                verification_committed
                and verification_evidence_complete
                and not alarm
                and admitted_detection_count
                >= int(self.minimum_detected_actions_to_clear)
            ),
            "admitted_target_sensing_detection_count": int(
                admitted_detection_count
            ),
            "admitted_time_calibration_detection_count": int(
                admitted_time_calibration_count
            ),
            "time_calibration_innovation_test_count": int(
                sum(
                    bool(item["target_independent_timing_probe"])
                    for item in tests
                )
            ),
            "time_calibration_not_counted_as_target_detection": True,
            "nonfinite_receiver_statistic_omission_count_this_epoch": int(
                nonfinite
            ),
            "tests": tests,
        }
        self.history.append(row)
        return row
