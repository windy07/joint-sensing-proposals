from __future__ import annotations

"""Realized detection/association likelihood factors for the V71 runtime.

The target component is a local Gaussian posterior-predictive density.  The
non-target component is either the registered Gaussian deception model or a
detector-conditioned density supplied by the observation extractor.  In both
cases responsibilities, score, and Louis observed information are derivatives
of one declared local mixture objective.  The resulting canonical factor is a
local representation; it is not a globally exact nonlinear-mixture MAP factor.
"""

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.dynamic_factor_graph import (
    DynamicFactor,
    FactorKind,
    RobustKernel,
    VariableKey,
    VariableKind,
)
from wideband_info_geometry.physics_v58.measurement import (
    ActionFactor,
    StateLayout,
    false_return_normalized_jacobian,
)
from wideband_info_geometry.physics_v58.schema import PhysicalWorld, RadarAction

from .belief import make_joint_information_factor, selected_action_information
from .contracts import FactorInformationMode, LikelihoodUpdateMode


@dataclass(frozen=True)
class MixtureScoreResult:
    state_score_normalized: np.ndarray
    louis_observed_information_normalized: np.ndarray
    responsibilities: tuple[float, float]
    log_likelihood: float
    component_log_likelihoods: tuple[float, float]
    component_priors: tuple[float, float]
    primary_rows: int
    association_model: str
    false_component_density_model: str
    false_component_log_density: float | None
    parameter_dependent_components: tuple[bool, bool]
    target_dependent_components: tuple[bool, bool]
    nuisance_dependent_components: tuple[bool, bool]
    louis_raw_minimum_eigenvalue: float
    louis_negative_eigenvalue_mass: float


def _psd_projection(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    values, vectors = np.linalg.eigh(symmetric)
    projected = (vectors * np.maximum(values, 0.0)) @ vectors.T
    return (
        0.5 * (projected + projected.T),
        float(np.min(values)),
        float(np.sum(np.maximum(-values, 0.0))),
    )


def _log_gaussian(residual: np.ndarray, covariance: np.ndarray) -> float:
    value = np.asarray(residual, dtype=float).reshape(-1)
    matrix = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    lower = np.linalg.cholesky(matrix)
    whitened = np.linalg.solve(lower, value)
    return float(
        -0.5 * value.size * math.log(2.0 * math.pi)
        - np.sum(np.log(np.diag(lower)))
        - 0.5 * whitened @ whitened
    )


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def mixture_score(
    world: PhysicalWorld,
    factor: ActionFactor,
    residual: np.ndarray,
    *,
    primary_rows: int,
    predictive_measurement_covariance: np.ndarray | None = None,
    predictive_state_covariance_normalized: np.ndarray | None = None,
    false_component_residual: np.ndarray | None = None,
    false_component_log_density: float | None = None,
    false_component_state_independent: bool = False,
    false_component_density_model: str | None = None,
    association_model_override: str | None = None,
    association_component_priors: Sequence[float] | None = None,
) -> MixtureScoreResult:
    """Differentiate a local belief-marginalized association likelihood.

    With fixed state covariance ``P`` and predicted mean ``m``, component ``k``
    has innovation covariance ``S_k = H_k P H_k' + R_k``.  Both association
    responsibilities and derivatives with respect to ``m`` must use ``S_k``;
    mixing ``S_k`` responsibilities with ``R_k`` scores is not the derivative
    of any single declared likelihood.
    """

    layout = StateLayout.from_world(world)
    value = np.asarray(residual, dtype=float).reshape(-1)
    if value.shape != factor.prediction.shape or not np.all(np.isfinite(value)):
        raise ValueError("mixture residual is invalid")
    primary = int(primary_rows)
    if primary <= 0 or primary > value.size:
        raise ValueError("mixture primary-row count is invalid")

    covariance = np.asarray(factor.covariance, dtype=float)
    primary_covariance = covariance[:primary, :primary]
    primary_residual = value[:primary]
    association_model = str(
        association_model_override
        if association_model_override is not None
        else factor.metadata.get("association_model", "unregistered")
    )
    use_declared_false_density = false_component_log_density is not None
    if use_declared_false_density:
        if not false_component_state_independent:
            raise ValueError(
                "an externally declared false density must state that it is state independent"
            )
        if not math.isfinite(float(false_component_log_density)):
            raise ValueError("false-component log density must be finite")
        if not false_component_density_model:
            raise ValueError("false-component density model must be named")
    correct_probability = float(
        np.clip(factor.link_budget.correct_association_probability, 0.0, 1.0)
    )
    false_probability = 1.0 - correct_probability
    displacement_sigma = np.asarray(
        factor.metadata.get("false_target_displacement_sigma", [0.0] * value.size),
        dtype=float,
    ).reshape(-1)[:primary]
    primary_lower = np.linalg.cholesky(primary_covariance)
    false_offset = primary_lower @ displacement_sigma
    false_scale = float(
        factor.metadata.get("false_return_covariance_scale", 1.0)
    )
    false_scale = max(false_scale, 1.0)
    false_covariance = (
        primary_covariance
        if association_model == "displaced_Gaussian_deception_mixture"
        else false_scale * primary_covariance
    )

    if false_component_residual is None:
        false_residual = primary_residual - false_offset
    else:
        supplied_false_residual = np.asarray(
            false_component_residual, dtype=float
        ).reshape(-1)
        if supplied_false_residual.size not in (primary, value.size):
            raise ValueError("false-component residual has wrong dimension")
        false_residual = supplied_false_residual[:primary]
        if not np.all(np.isfinite(false_residual)):
            raise ValueError("false-component residual is not finite")
    component_residuals = (primary_residual, false_residual)
    component_covariances = (primary_covariance, false_covariance)
    h_normalized = factor.jacobian_physical * layout.scales()[None, :]
    h_true = h_normalized[:primary]
    h_false = (
        np.zeros_like(h_true)
        if use_declared_false_density
        else false_return_normalized_jacobian(factor, layout)[:primary]
    )
    if predictive_state_covariance_normalized is not None:
        if predictive_measurement_covariance is not None:
            raise ValueError(
                "provide state or measurement predictive covariance, not both"
            )
        state_covariance = np.asarray(
            predictive_state_covariance_normalized, dtype=float
        )
        if state_covariance.shape != (layout.dimension, layout.dimension):
            raise ValueError("predictive state covariance has wrong shape")
        state_prediction_covariances = (
            h_true @ state_covariance @ h_true.T,
            h_false @ state_covariance @ h_false.T,
        )
    elif predictive_measurement_covariance is None:
        state_prediction_covariances = (
            np.zeros_like(primary_covariance),
            np.zeros_like(primary_covariance),
        )
    else:
        predictive = np.asarray(predictive_measurement_covariance, dtype=float)
        if predictive.shape != covariance.shape:
            raise ValueError("predictive measurement covariance has wrong shape")
        # Backward-compatible API for callers that only carry H_true P H_true'.
        # Operational V71 passes the state covariance so the false component
        # receives its own nuisance-only predictive covariance.
        primary_predictive = predictive[:primary, :primary]
        state_prediction_covariances = (primary_predictive, primary_predictive)
    association_covariances = tuple(
        prediction + measurement
        for prediction, measurement in zip(
            state_prediction_covariances, component_covariances
        )
    )
    if association_component_priors is None:
        raw_priors = np.asarray(
            [correct_probability, false_probability], dtype=float
        )
    else:
        raw_priors = np.asarray(
            association_component_priors, dtype=float
        ).reshape(-1)
        if (
            raw_priors.shape != (2,)
            or not np.all(np.isfinite(raw_priors))
            or np.any(raw_priors < 0.0)
            or float(np.sum(raw_priors)) <= 0.0
        ):
            raise ValueError(
                "association component priors must be two finite, "
                "nonnegative values with positive total mass"
            )
        raw_priors = raw_priors / float(np.sum(raw_priors))
    priors = np.maximum(raw_priors, 1e-300)
    priors = priors / float(np.sum(priors))
    component_logs = np.asarray(
        [
            math.log(priors[0])
            + _log_gaussian(component_residuals[0], association_covariances[0]),
            math.log(priors[1])
            + (
                float(false_component_log_density)
                if use_declared_false_density
                else _log_gaussian(
                    component_residuals[1], association_covariances[1]
                )
            ),
        ],
        dtype=float,
    )
    normalization = _logsumexp(component_logs)
    responsibilities = np.exp(component_logs - normalization)

    # The false return is fixed with respect to the target state, but delay and
    # Doppler produced by that return still pass through receiver clock and
    # frequency offsets.  Component-specific Jacobians are therefore required:
    # H_false has zero target columns and retains the nuisance columns.  For a
    # two-component latent association model, Louis' identity gives
    #
    # J_obs(y) = E[J_complete | y] - Cov[s_complete | y].
    #
    # The raw local Hessian may be indefinite away from a mode.  The canonical
    # Gaussian factor uses its PSD part; the trust-region line search separately
    # enforces monotone likelihood and atomically rolls rejected factors back.
    component_state_scores = (
        h_true.T
        @ np.linalg.solve(association_covariances[0], component_residuals[0]),
        (
            np.zeros(layout.dimension, dtype=float)
            if use_declared_false_density
            else h_false.T
            @ np.linalg.solve(
                association_covariances[1], component_residuals[1]
            )
        ),
    )
    primary_state_score = sum(
        float(weight) * score
        for weight, score in zip(responsibilities, component_state_scores)
    )
    component_information = (
        h_true.T @ np.linalg.solve(association_covariances[0], h_true),
        (
            np.zeros((layout.dimension, layout.dimension), dtype=float)
            if use_declared_false_density
            else h_false.T @ np.linalg.solve(association_covariances[1], h_false)
        ),
    )
    expected_complete_information = sum(
        float(weight) * information
        for weight, information in zip(responsibilities, component_information)
    )
    score_second_moment = sum(
        float(weight) * np.outer(score, score)
        for weight, score in zip(responsibilities, component_state_scores)
    )
    louis_information = (
        expected_complete_information
        - score_second_moment
        + np.outer(primary_state_score, primary_state_score)
    )
    state_score = np.asarray(primary_state_score, dtype=float)
    if primary < value.size:
        calibration_covariance = covariance[primary:, primary:]
        h_calibration = h_normalized[primary:]
        state_score = state_score + h_calibration.T @ np.linalg.solve(
            calibration_covariance, value[primary:]
        )
        louis_information = louis_information + h_calibration.T @ np.linalg.solve(
            calibration_covariance, h_calibration
        )
    louis_psd, louis_minimum, louis_negative_mass = _psd_projection(
        louis_information
    )
    false_parameter_dependent = bool(
        not use_declared_false_density and np.linalg.norm(h_false) > 1.0e-14
    )
    false_nuisance_dependent = bool(
        h_false.shape[1] > layout.target_dim
        and np.linalg.norm(h_false[:, layout.target_dim :]) > 1.0e-14
    )
    return MixtureScoreResult(
        state_score_normalized=np.asarray(state_score, dtype=float),
        louis_observed_information_normalized=louis_psd,
        responsibilities=(
            float(responsibilities[0]),
            float(responsibilities[1]),
        ),
        log_likelihood=float(normalization),
        component_log_likelihoods=(
            float(component_logs[0]),
            float(component_logs[1]),
        ),
        component_priors=(float(priors[0]), float(priors[1])),
        primary_rows=primary,
        association_model=association_model,
        false_component_density_model=(
            str(false_component_density_model)
            if use_declared_false_density
            else "Gaussian_component"
        ),
        false_component_log_density=(
            float(false_component_log_density)
            if use_declared_false_density
            else None
        ),
        parameter_dependent_components=(True, false_parameter_dependent),
        target_dependent_components=(True, False),
        nuisance_dependent_components=(True, false_nuisance_dependent),
        louis_raw_minimum_eigenvalue=louis_minimum,
        louis_negative_eigenvalue_mass=louis_negative_mass,
    )


def make_operational_mixture_factor(
    world: PhysicalWorld,
    action: RadarAction,
    action_factor: ActionFactor,
    *,
    epoch: int,
    arrival_epoch: int,
    participants: Sequence[str],
    linearization_state_normalized: np.ndarray,
    measurement_residual: np.ndarray,
    primary_rows: int,
    provenance_id: str,
    factor_id: str,
    fidelity_level: str,
    update_mode: LikelihoodUpdateMode,
    sandwich_temperature: float,
    score_step_scale: float = 1.0,
    robust_kernel: RobustKernel,
    eigenvalue_floor: float,
    communication_snapshot_epoch: int | None = None,
    predictive_measurement_covariance: np.ndarray | None = None,
    predictive_state_covariance_normalized: np.ndarray | None = None,
    false_component_residual: np.ndarray | None = None,
    false_component_log_density: float | None = None,
    false_component_state_independent: bool = False,
    false_component_density_model: str | None = None,
    association_model_override: str | None = None,
    association_component_priors: Sequence[float] | None = None,
) -> tuple[DynamicFactor | None, MixtureScoreResult]:
    """Build a canonical factor from an exact local mixture derivative."""

    if update_mode is LikelihoodUpdateMode.LEGACY_FISHER_PSEUDOFACTOR:
        raise ValueError("legacy factors must use make_operational_action_factor")
    result = mixture_score(
        world,
        action_factor,
        measurement_residual,
        primary_rows=primary_rows,
        predictive_measurement_covariance=predictive_measurement_covariance,
        predictive_state_covariance_normalized=(
            predictive_state_covariance_normalized
        ),
        false_component_residual=false_component_residual,
        false_component_log_density=false_component_log_density,
        false_component_state_independent=false_component_state_independent,
        false_component_density_model=false_component_density_model,
        association_model_override=association_model_override,
        association_component_priors=association_component_priors,
    )
    x0 = np.asarray(linearization_state_normalized, dtype=float).reshape(-1)
    use_louis_information = (
        update_mode is LikelihoodUpdateMode.MIXTURE_SCORE_LOUIS_PSD
    )
    information = (
        result.louis_observed_information_normalized
        if use_louis_information
        else selected_action_information(
            world,
            action,
            action_factor,
            FactorInformationMode.CONDITIONAL_DETECTED,
        )
    )
    temperature = (
        float(sandwich_temperature)
        if update_mode is LikelihoodUpdateMode.MIXTURE_SCORE_SANDWICH
        else 1.0
    )
    information = information / temperature
    step_scale = float(score_step_scale)
    if not 0.0 < step_scale <= 1.0 or not math.isfinite(step_scale):
        raise ValueError("mixture Fisher score-step scale must lie in (0, 1]")
    score = step_scale * result.state_score_normalized / temperature
    eta = information @ x0 + score
    dynamic = make_joint_information_factor(
        factor_id=factor_id,
        kind=(
            FactorKind.CALIBRATION
            if action.kind.startswith("calibration")
            else FactorKind.ASSOCIATION
        ),
        epoch=epoch,
        target_key=VariableKey(VariableKind.TARGET, "primary", int(epoch)),
        nuisance_key=VariableKey(VariableKind.NUISANCE, "network", int(epoch)),
        information=information,
        information_vector=eta,
        participants=participants,
        provenance_ids=(provenance_id,),
        fidelity_level=fidelity_level,
        robust_kernel=robust_kernel,
        eigenvalue_floor=eigenvalue_floor,
        metadata={
            "action_id": int(action.action_id),
            "action_kind": action.kind,
            "measurement_names": list(action_factor.measurement_names),
            "information_mode": FactorInformationMode.CONDITIONAL_DETECTED.value,
            "factor_semantics": (
                "local_predictive_mixture_score_with_PSD_Louis_observed_information"
                if use_louis_information
                else "local_predictive_mixture_score_with_expected_Fisher_curvature"
            ),
            "exact_local_observed_data_mixture_score": True,
            # The local score is exact for the declared mixture at x0.  Its
            # canonical Gaussian container uses expected Fisher curvature, so
            # it is not the globally exact nonlinear likelihood function.
            "exact_observed_data_conditional_mixture_likelihood_factor": False,
            "exact_local_predictive_mixture_mean_score": True,
            "exact_local_Fisher_scoring_factor": not use_louis_information,
            "exact_local_Louis_observed_information_before_PSD_projection": (
                use_louis_information
            ),
            "global_nonlinear_MAP_claimed": False,
            "frozen_linearization": True,
            "score_and_curvature_share_predictive_mixture_objective": True,
            "curvature_semantics": (
                "PSD_part_of_sample_Louis_observed_information"
                if use_louis_information
                else "conditional_expected_Fisher_not_sample_observed_Hessian"
            ),
            "association_responsibilities": list(result.responsibilities),
            "association_responsibilities_use_predictive_innovation_covariance": True,
            "association_model": result.association_model,
            "mixture_log_likelihood": result.log_likelihood,
            "component_log_likelihoods": list(result.component_log_likelihoods),
            "association_component_priors": list(result.component_priors),
            "association_component_priors_frozen_at_capture": bool(
                association_component_priors is not None
            ),
            "association_component_priors_state_derivative_excluded": bool(
                association_component_priors is not None
            ),
            "false_component_density_model": result.false_component_density_model,
            "false_component_log_density": result.false_component_log_density,
            "primary_measurement_rows": int(primary_rows),
            "sandwich_temperature": float(temperature),
            "score_step_scale": float(step_scale),
            "false_component_target_state_derivative": False,
            "false_component_nuisance_state_derivative": bool(
                result.nuisance_dependent_components[1]
            ),
            "louis_raw_minimum_eigenvalue": (
                result.louis_raw_minimum_eigenvalue
            ),
            "louis_negative_eigenvalue_mass": (
                result.louis_negative_eigenvalue_mass
            ),
            "communication_snapshot_epoch": int(
                epoch
                if communication_snapshot_epoch is None
                else communication_snapshot_epoch
            ),
        },
    )
    if dynamic is None:
        return None, result
    return (
        DynamicFactor(
            factor_id=dynamic.factor_id,
            kind=dynamic.kind,
            observation_epoch=dynamic.observation_epoch,
            arrival_epoch=int(arrival_epoch),
            variables=dynamic.variables,
            jacobians=dynamic.jacobians,
            measurement=dynamic.measurement,
            covariance=dynamic.covariance,
            participants=dynamic.participants,
            provenance_ids=dynamic.provenance_ids,
            correlation_group=dynamic.correlation_group,
            robust_kernel=dynamic.robust_kernel,
            robust_scale=dynamic.robust_scale,
            association_probabilities=(1.0,),
            missing=dynamic.missing,
            fidelity_level=dynamic.fidelity_level,
            metadata=dynamic.metadata,
        ),
        result,
    )


def mixture_contract_report(result: MixtureScoreResult) -> dict[str, Any]:
    responsibilities = np.asarray(result.responsibilities, dtype=float)
    louis_information = np.asarray(
        result.louis_observed_information_normalized, dtype=float
    )
    louis_eigenvalues = np.linalg.eigvalsh(
        0.5 * (louis_information + louis_information.T)
    )
    passed = bool(
        np.isclose(np.sum(responsibilities), 1.0, atol=1e-12)
        and np.all(np.isfinite(responsibilities))
        and np.all(responsibilities >= 0.0)
        and np.all(np.isfinite(result.state_score_normalized))
        and np.all(np.isfinite(louis_information))
        and float(np.min(louis_eigenvalues)) >= -1.0e-9
    )
    return {
        "responsibilities_sum_to_one": bool(
            np.isclose(np.sum(responsibilities), 1.0, atol=1e-12)
        ),
        "responsibilities_finite_nonnegative": bool(
            np.all(np.isfinite(responsibilities))
            and np.all(responsibilities >= 0.0)
        ),
        "score_finite": bool(np.all(np.isfinite(result.state_score_normalized))),
        "association_model": result.association_model,
        "association_component_priors": list(result.component_priors),
        "component_log_likelihoods": list(result.component_log_likelihoods),
        "false_component_density_model": result.false_component_density_model,
        "false_component_log_density": result.false_component_log_density,
        "parameter_dependent_components": list(
            result.parameter_dependent_components
        ),
        "target_dependent_components": list(result.target_dependent_components),
        "nuisance_dependent_components": list(
            result.nuisance_dependent_components
        ),
        "false_component_target_state_derivative": False,
        "false_component_nuisance_state_derivative": bool(
            result.nuisance_dependent_components[1]
        ),
        "louis_PSD_projection_finite": bool(np.all(np.isfinite(louis_information))),
        "louis_PSD_minimum_eigenvalue": float(np.min(louis_eigenvalues)),
        "louis_raw_minimum_eigenvalue": result.louis_raw_minimum_eigenvalue,
        "louis_negative_eigenvalue_mass": result.louis_negative_eigenvalue_mass,
        "status": "passed" if passed else "failed_closed",
    }
