from __future__ import annotations

"""Development-only calibration of the AOA detector event chain.

The analytic link budget predicts a temporal target-cell detection event.  The
R2 AOA extractor accepts a measurement only after that event and a second
spatial/profile event.  This module keeps those events distinct and calibrates
their probabilities without changing the electromagnetic link budget:

    P(AOA accepted) = P(temporal CFAR) P(spatial acceptance | temporal CFAR).

Both factors are monotone logistic functions of the controller-observable
nominal temporal probability.  Slopes are constrained to be nonnegative, so
the fitted joint probability is monotone and cannot exceed its calibrated
temporal factor.  Fitting is permitted only on development observations.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


MODEL_SCHEMA = (
    "v75-r18p6-development-only-two-stage-joint-aoa-probability-1.0"
)
PROBABILITY_EPSILON = 1.0e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    raise ValueError(f"cannot parse Boolean value {value!r}")


def _clip_probability(value: object) -> float:
    probability = _finite(value, "probability")
    return float(
        np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    )


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p) - math.log1p(-p)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-min(value, 745.0))
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(max(value, -745.0))
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class JointAOAProbabilityModel:
    temporal_intercept: float
    temporal_slope: float
    spatial_given_temporal_intercept: float
    spatial_given_temporal_slope: float
    development_world_count: int
    development_temporal_pass_count: int
    source_report_sha256: str
    regularization_l2: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JointAOAProbabilityModel":
        if payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported joint AOA probability model schema")
        if payload.get("status") != "PASS":
            raise ValueError("joint AOA probability model did not pass its fit checks")
        if payload.get("source_split") != "development":
            raise ValueError("joint AOA probability model is not development-only")
        if payload.get("confirmation_statistics_used") is not False:
            raise ValueError("joint AOA probability fit lacks confirmation isolation")
        temporal = payload.get("temporal_cfar_calibration")
        spatial = payload.get("spatial_acceptance_given_temporal_calibration")
        if not isinstance(temporal, Mapping) or not isinstance(spatial, Mapping):
            raise ValueError("joint AOA probability model is missing an event factor")
        result = cls(
            temporal_intercept=_finite(temporal.get("intercept"), "temporal intercept"),
            temporal_slope=_finite(temporal.get("slope"), "temporal slope"),
            spatial_given_temporal_intercept=_finite(
                spatial.get("intercept"), "spatial intercept"
            ),
            spatial_given_temporal_slope=_finite(
                spatial.get("slope"), "spatial slope"
            ),
            development_world_count=int(payload.get("development_world_count", 0)),
            development_temporal_pass_count=int(
                payload.get("development_temporal_pass_count", 0)
            ),
            source_report_sha256=str(payload.get("source_report_sha256", "")),
            regularization_l2=_finite(
                payload.get("regularization_l2", 0.0), "regularization"
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.temporal_slope < 0.0 or self.spatial_given_temporal_slope < 0.0:
            raise ValueError("joint AOA probability slopes must be nonnegative")
        if self.development_world_count <= 0:
            raise ValueError("joint AOA probability model needs development worlds")
        if self.development_temporal_pass_count <= 0:
            raise ValueError("spatial calibration needs temporal CFAR passes")
        if self.regularization_l2 < 0.0:
            raise ValueError("regularization must be nonnegative")
        digest = self.source_report_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("joint AOA model source report needs a SHA-256 digest")

    def predict(self, nominal_temporal_probability: float) -> dict[str, float]:
        log_odds = _logit(nominal_temporal_probability)
        temporal = _sigmoid(
            self.temporal_intercept + self.temporal_slope * log_odds
        )
        spatial_given_temporal = _sigmoid(
            self.spatial_given_temporal_intercept
            + self.spatial_given_temporal_slope * log_odds
        )
        joint = temporal * spatial_given_temporal
        if not (
            0.0 <= joint <= temporal <= 1.0
            and 0.0 <= spatial_given_temporal <= 1.0
        ):
            raise RuntimeError("joint AOA probability violated event-chain bounds")
        return {
            "nominal_temporal_target_cell_probability": float(
                nominal_temporal_probability
            ),
            "calibrated_temporal_cfar_probability": float(temporal),
            "spatial_acceptance_probability_given_temporal": float(
                spatial_given_temporal
            ),
            "joint_aoa_success_probability": float(joint),
        }


def load_joint_aoa_probability_model(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[JointAOAProbabilityModel, dict[str, Any]]:
    observed = sha256_file(path)
    if observed != expected_sha256.lower():
        raise ValueError("joint AOA probability model hash mismatch")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    model = JointAOAProbabilityModel.from_mapping(payload)
    return model, {
        "path": str(path.resolve()),
        "sha256": observed,
        "schema": MODEL_SCHEMA,
        "source_split": "development",
        "source_report_sha256": model.source_report_sha256,
        "confirmation_statistics_used": False,
    }


def _fit_monotone_logistic(
    log_odds: np.ndarray,
    outcome: np.ndarray,
    *,
    regularization_l2: float,
) -> tuple[float, float, dict[str, Any]]:
    x = np.column_stack([np.ones(log_odds.size, dtype=float), log_odds])
    y = np.asarray(outcome, dtype=float)
    if x.shape[0] < 2 or np.unique(y).size < 2:
        raise ValueError("monotone logistic calibration needs both event outcomes")
    prevalence = float(np.clip(np.mean(y), 1.0e-6, 1.0 - 1.0e-6))
    initial = np.asarray([_logit(prevalence), 0.1], dtype=float)

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = np.clip(x @ coefficient, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-score))
        loss = float(
            np.sum(np.logaddexp(0.0, score) - y * score)
            + 0.5 * regularization_l2 * coefficient[1] ** 2
        )
        gradient = x.T @ (probability - y)
        gradient[1] += regularization_l2 * coefficient[1]
        return loss, gradient

    result = minimize(
        lambda coefficient: objective(coefficient),
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None)),
        options={"ftol": 1.0e-12, "gtol": 1.0e-9, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"monotone logistic calibration failed: {result.message}")
    intercept, slope = (float(value) for value in result.x)
    return intercept, slope, {
        "optimizer": "L-BFGS-B",
        "converged": True,
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "slope_lower_bound": 0.0,
    }


def probability_metrics(probability: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    p = np.clip(np.asarray(probability, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    y = np.asarray(outcome, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probability metrics require aligned nonempty vectors")
    index = np.minimum(np.floor(10.0 * p).astype(int), 9)
    bins: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for bin_index in range(10):
        mask = index == bin_index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        mean_prediction = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))
        weighted_gap += count * abs(mean_prediction - observed)
        bins.append(
            {
                "lower": bin_index / 10.0,
                "upper": (bin_index + 1) / 10.0,
                "count": count,
                "mean_prediction": mean_prediction,
                "observed_rate": observed,
            }
        )
    return {
        "sample_count": int(p.size),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_probability": float(np.mean(y)),
        "brier_score": float(np.mean(np.square(p - y))),
        "log_loss": float(
            -np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))
        ),
        "expected_calibration_error": float(weighted_gap / p.size),
        "bins": bins,
    }


def fit_development_joint_aoa_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_report_sha256: str,
    minimum_worlds: int = 60,
    minimum_temporal_passes: int = 30,
    regularization_l2: float = 0.1,
) -> dict[str, Any]:
    failed: list[str] = []
    world_ids = {str(row.get("world_id", "")) for row in rows}
    if len(world_ids) < minimum_worlds:
        failed.append("insufficient_independent_development_worlds")
    if any(
        not str(row.get("scenario", "")).startswith("development_")
        for row in rows
    ):
        failed.append("source_rows_are_not_exclusively_development")
    if regularization_l2 < 0.0 or not math.isfinite(regularization_l2):
        failed.append("invalid_regularization")
    nominal = np.asarray(
        [
            _clip_probability(
                row.get(
                    "predicted_nominal_temporal_target_cell_detection_probability"
                )
            )
            for row in rows
        ],
        dtype=float,
    )
    temporal = np.asarray(
        [_boolean(row.get("temporal_cfar_passed")) for row in rows], dtype=float
    )
    joint = np.asarray([_boolean(row.get("detected")) for row in rows], dtype=float)
    if np.any(joint > temporal):
        failed.append("joint_detection_occurred_without_temporal_cfar")
    temporal_mask = temporal > 0.5
    temporal_pass_count = int(np.count_nonzero(temporal_mask))
    if temporal_pass_count < minimum_temporal_passes:
        failed.append("insufficient_temporal_passes_for_spatial_calibration")
    digest = source_report_sha256.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        failed.append("invalid_source_report_sha256")
    if failed:
        return {
            "schema": MODEL_SCHEMA,
            "status": "FAIL",
            "source_split": "development",
            "confirmation_statistics_used": False,
            "failed_checks": failed,
            "development_world_count": len(world_ids),
            "development_temporal_pass_count": temporal_pass_count,
        }

    log_odds = np.asarray([_logit(value) for value in nominal], dtype=float)
    temporal_intercept, temporal_slope, temporal_optimizer = (
        _fit_monotone_logistic(
            log_odds,
            temporal,
            regularization_l2=regularization_l2,
        )
    )
    spatial_intercept, spatial_slope, spatial_optimizer = (
        _fit_monotone_logistic(
            log_odds[temporal_mask],
            joint[temporal_mask],
            regularization_l2=regularization_l2,
        )
    )
    model = JointAOAProbabilityModel(
        temporal_intercept=temporal_intercept,
        temporal_slope=temporal_slope,
        spatial_given_temporal_intercept=spatial_intercept,
        spatial_given_temporal_slope=spatial_slope,
        development_world_count=len(world_ids),
        development_temporal_pass_count=temporal_pass_count,
        source_report_sha256=digest,
        regularization_l2=regularization_l2,
    )
    model.validate()
    predictions = [model.predict(float(value)) for value in nominal]
    temporal_probability = np.asarray(
        [item["calibrated_temporal_cfar_probability"] for item in predictions]
    )
    joint_probability = np.asarray(
        [item["joint_aoa_success_probability"] for item in predictions]
    )
    monotone_grid = np.asarray(
        [model.predict(float(value))["joint_aoa_success_probability"] for value in np.linspace(1e-6, 1.0 - 1e-6, 1001)]
    )
    return {
        "schema": MODEL_SCHEMA,
        "status": "PASS",
        "source_split": "development",
        "source_report_sha256": digest,
        "confirmation_statistics_used": False,
        "feature": "logit_of_analytic_nominal_temporal_target_cell_probability",
        "event_factorization": (
            "P(joint_AOA_accept)=P(temporal_CFAR)*"
            "P(spatial_profile_accept|temporal_CFAR)"
        ),
        "regularization_l2": float(regularization_l2),
        "development_world_count": len(world_ids),
        "development_temporal_pass_count": temporal_pass_count,
        "development_joint_accept_count": int(np.sum(joint)),
        "temporal_cfar_calibration": {
            "intercept": temporal_intercept,
            "slope": temporal_slope,
            "optimizer": temporal_optimizer,
            "development_metrics": probability_metrics(
                temporal_probability, temporal
            ),
        },
        "spatial_acceptance_given_temporal_calibration": {
            "intercept": spatial_intercept,
            "slope": spatial_slope,
            "optimizer": spatial_optimizer,
            "development_metrics": probability_metrics(
                np.asarray(
                    [
                        predictions[index][
                            "spatial_acceptance_probability_given_temporal"
                        ]
                        for index in np.flatnonzero(temporal_mask)
                    ]
                ),
                joint[temporal_mask],
            ),
        },
        "development_joint_metrics": probability_metrics(
            joint_probability, joint
        ),
        "development_nominal_temporal_probability_against_joint_outcome": (
            probability_metrics(nominal, joint)
        ),
        "invariants": {
            "temporal_slope_nonnegative": temporal_slope >= 0.0,
            "spatial_slope_nonnegative": spatial_slope >= 0.0,
            "joint_probability_monotone_on_dense_grid": bool(
                np.all(np.diff(monotone_grid) >= -1.0e-14)
            ),
            "joint_probability_never_exceeds_calibrated_temporal_probability": True,
            "truth_fields_used_as_model_features": False,
            "confirmation_statistics_used": False,
        },
        "failed_checks": [],
    }
