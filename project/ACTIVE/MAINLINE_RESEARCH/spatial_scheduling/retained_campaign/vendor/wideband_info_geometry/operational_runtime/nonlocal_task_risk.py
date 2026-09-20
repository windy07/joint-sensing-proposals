from __future__ import annotations

"""Posterior-predictive finite-set risk for bounded radar schedules.

This module is an evaluator, not a second belief filter.  It freezes the
controller-owned Bernoulli/JPDA track set, draws hypothetical target sets from
that belief, generates action-conditioned measurements with the project's
physical measurement model, and updates isolated planning forks.  Simulator
truth is never an input.

The terminal loss is the expected squared GOSPA distance over a deterministic
cubature representation of each Gaussian Bernoulli component.  The Bayes
action is restricted to subsets of the posterior track means.  Consequently,
the reported quantity is an exact risk for the declared finite cubature and
report-action support, not an exact continuous-RFS Bayes risk.
"""

from dataclasses import dataclass, replace
import hashlib
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.dynamic_episode import (
    ManeuverMode,
    TargetDynamicState,
)
from wideband_info_geometry.physics_v58.measurement import build_action_factor
from wideband_info_geometry.physics_v58.schema import Jammer, PhysicalWorld

from .contracts import ObservationRecord
from .planning_detection_probability import PlanningDetectionProbability
from .multitarget_v71 import (
    BernoulliKinematicTrack,
    MultiTargetAssociationEngine,
    _stable_seed,
    _target_rows,
    _target_world,
)
from .covariance_coordinates import CovarianceModelError, checked_covariance, gaussian_draw
from .measurement_return_sets import (
    HypotheticalReturnSet, capture_measurement_return_set,
)


@dataclass(frozen=True)
class FiniteSetRisk:
    power_risk_m2: float
    gospa_risk_m: float
    report_track_ids: tuple[str, ...]
    truth_support_size: int
    report_support_size: int
    probability_mass: float
    scope: str = (
        "independent Bernoulli tracks; positive-weight Gaussian cubature; "
        "Bayes report restricted to subsets of posterior track means"
    )


@dataclass(frozen=True)
class PosteriorPredictiveReplicate:
    prior_realized_power_loss_m2: float
    posterior_realized_power_loss_m2: float
    posterior_self_power_risk_m2: float
    posterior_report_track_ids: tuple[str, ...]


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _track_digest(tracks: Sequence[BernoulliKinematicTrack]) -> str:
    digest = hashlib.sha256()
    for track in tracks:
        digest.update(track.track_id.encode("utf-8"))
        digest.update(np.float64(track.existence_probability).tobytes())
        digest.update(np.asarray(track.mean, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(track.covariance, dtype="<f8").tobytes(order="C"))
        digest.update(int(track.age).to_bytes(8, "little", signed=True))
        digest.update(int(track.missed_epochs).to_bytes(8, "little", signed=True))
        digest.update(track.origin.encode("utf-8"))
    return digest.hexdigest()


def _engine_digest(engine: MultiTargetAssociationEngine) -> str:
    digest = hashlib.sha256(_track_digest(engine.tracks).encode("ascii"))
    digest.update(str(engine.receiver_id).encode("utf-8"))
    digest.update(str(engine._hypothetical_context_id).encode("utf-8"))
    for receipt_id in sorted(engine._consumed_return_set_receipts):
        digest.update(("receipt:" + receipt_id).encode("utf-8"))
    for provenance_id in sorted(engine._consumed_return_set_provenance_ids):
        digest.update(("provenance:" + provenance_id).encode("utf-8"))
    for birth in engine._tentative_births:
        digest.update(np.asarray(birth.mean, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(birth.covariance, dtype="<f8").tobytes(order="C"))
        digest.update(np.float64(birth.existence_probability).tobytes())
        digest.update(int(birth.last_epoch).to_bytes(8, "little", signed=True))
        for source in sorted(birth.sources):
            digest.update(int(source[0]).to_bytes(8, "little", signed=True))
            digest.update(int(source[1]).to_bytes(8, "little", signed=True))
    digest.update(int(engine._next_track_id).to_bytes(8, "little", signed=True))
    digest.update(
        int(-1 if engine._prepared_epoch is None else engine._prepared_epoch).to_bytes(
            8,
            "little",
            signed=True,
        )
    )
    return digest.hexdigest()


def _minimum_assignment_power(cost: np.ndarray) -> float:
    rows, columns = cost.shape
    if min(rows, columns) == 0:
        return 0.0
    if rows <= columns:
        return min(
            sum(float(cost[row, column]) for row, column in enumerate(selection))
            for selection in itertools.permutations(range(columns), rows)
        )
    return min(
        sum(float(cost[row, column]) for column, row in enumerate(selection))
        for selection in itertools.permutations(range(rows), columns)
    )


def gospa_power(
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    cutoff_m: float,
    alpha: float = 2.0,
) -> float:
    r"""Return :math:`d_2^{(c,2)}(X,\hat X)^2` in square metres.

    Only ``alpha=2`` is accepted because the localization/cardinality
    decomposition used here is the standard GOSPA decomposition for that
    value.  Silently extending the same expression to arbitrary alpha would
    change the assignment problem.
    """

    if cutoff_m <= 0.0 or not math.isfinite(cutoff_m):
        raise ValueError("GOSPA cutoff must be finite and positive")
    if not math.isclose(float(alpha), 2.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("the operational finite-set risk requires GOSPA alpha=2")
    left = np.asarray(truth, dtype=float)
    right = np.asarray(estimate, dtype=float)
    if left.size == 0:
        left = np.empty((0, 3), dtype=float)
    if right.size == 0:
        right = np.empty((0, 3), dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        raise ValueError("finite-set points must have matching [count, dimension] shapes")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("finite-set points must be finite")
    if left.shape[0] and right.shape[0]:
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        localization = _minimum_assignment_power(
            np.minimum(float(cutoff_m) ** 2, squared)
        )
    else:
        localization = 0.0
    cardinality = (
        float(cutoff_m) ** 2
        * abs(int(left.shape[0]) - int(right.shape[0]))
        / 2.0
    )
    return float(localization + cardinality)


def _positive_cubature(mean: np.ndarray, covariance: np.ndarray) -> tuple[tuple[np.ndarray, float], ...]:
    center = np.asarray(mean, dtype=float).reshape(3)
    # Position cubature remains defined in canonical metres with its original
    # eigendirection rule; numerical validation is not a new cubature law.
    covariance = checked_covariance(np.asarray(covariance, dtype=float).reshape(3, 3))
    values, vectors = np.linalg.eigh(covariance)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    dimension = center.size
    points: list[tuple[np.ndarray, float]] = [(center, 0.5)]
    scale = math.sqrt(2.0 * dimension)
    for column in range(dimension):
        delta = scale * root[:, column]
        weight = 0.25 / dimension
        points.append((center + delta, weight))
        points.append((center - delta, weight))
    mass = sum(weight for _, weight in points)
    if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("positive cubature weights do not sum to one")
    return tuple(points)


def finite_set_cubature_risk(
    tracks: Sequence[BernoulliKinematicTrack],
    *,
    cutoff_m: float,
    alpha: float = 2.0,
    maximum_tracks: int = 4,
) -> FiniteSetRisk:
    """Evaluate a bounded posterior Bayes set risk without truth labels."""

    frozen = tuple(tracks)
    if len(frozen) > maximum_tracks:
        raise ValueError(
            f"finite-set cubature supports at most {maximum_tracks} tracks; "
            f"received {len(frozen)}"
        )
    supports: list[tuple[tuple[np.ndarray | None, float], ...]] = []
    for track in frozen:
        existence = float(track.existence_probability)
        if not 0.0 <= existence <= 1.0 or not math.isfinite(existence):
            raise ValueError("track existence probability is invalid")
        position = np.asarray(track.mean, dtype=float)[:3]
        covariance = np.asarray(track.covariance, dtype=float)[:3, :3]
        present = tuple(
            (point, existence * weight)
            for point, weight in _positive_cubature(position, covariance)
        )
        supports.append(((None, 1.0 - existence), *present))

    truth_support: list[tuple[np.ndarray, float]] = []
    for joint in itertools.product(*supports) if supports else [()]:
        probability = math.prod(float(item[1]) for item in joint)
        if probability <= 0.0:
            continue
        points = [item[0] for item in joint if item[0] is not None]
        truth_support.append(
            (
                np.asarray(points, dtype=float).reshape((-1, 3)),
                probability,
            )
        )
    probability_mass = float(sum(weight for _, weight in truth_support))
    if not math.isclose(probability_mass, 1.0, rel_tol=0.0, abs_tol=2.0e-10):
        raise RuntimeError(
            f"finite-set truth support has probability mass {probability_mass:.16g}"
        )

    reports: list[tuple[tuple[str, ...], np.ndarray]] = []
    for mask in range(1 << len(frozen)):
        selected = [index for index in range(len(frozen)) if mask & (1 << index)]
        reports.append(
            (
                tuple(frozen[index].track_id for index in selected),
                np.asarray(
                    [np.asarray(frozen[index].mean, dtype=float)[:3] for index in selected],
                    dtype=float,
                ).reshape((-1, 3)),
            )
        )
    risks = []
    for track_ids, report in reports:
        power = sum(
            probability
            * gospa_power(truth, report, cutoff_m=cutoff_m, alpha=alpha)
            for truth, probability in truth_support
        )
        risks.append((float(power), track_ids))
    best_power, best_report = min(risks, key=lambda item: (item[0], item[1]))
    return FiniteSetRisk(
        power_risk_m2=best_power,
        gospa_risk_m=math.sqrt(max(best_power, 0.0)),
        report_track_ids=best_report,
        truth_support_size=len(truth_support),
        report_support_size=len(reports),
        probability_mass=probability_mass,
    )


def _planning_fork(
    engine: MultiTargetAssociationEngine,
    *,
    seed: int,
) -> MultiTargetAssociationEngine:
    branch = object.__new__(MultiTargetAssociationEngine)
    branch.config = replace(engine.config, seed=int(seed & 0x7FFFFFFF))
    branch.receiver_id = engine.receiver_id
    branch._hypothetical_context_id = f"posterior_predictive:{seed}"
    branch._consumed_return_set_receipts = set(engine._consumed_return_set_receipts)
    branch._consumed_return_set_provenance_ids = set(engine._consumed_return_set_provenance_ids)
    branch.enabled = bool(engine.enabled)
    branch.tracks = [
        replace(
            track,
            mean=np.asarray(track.mean, dtype=float).copy(),
            covariance=np.asarray(track.covariance, dtype=float).copy(),
        )
        for track in engine.tracks
    ]
    branch._bootstrapped = bool(engine._bootstrapped)
    branch._next_track_id = int(engine._next_track_id)
    branch._tentative_births = [
        replace(
            birth,
            mean=np.asarray(birth.mean, dtype=float).copy(),
            covariance=np.asarray(birth.covariance, dtype=float).copy(),
            sources=set(birth.sources),
        )
        for birth in engine._tentative_births
    ]
    branch._birth_promotions_by_epoch = dict(engine._birth_promotions_by_epoch)
    branch.history = []
    branch._previous_truth_assignment = {}
    branch._id_switches = 0
    branch._prepared_epoch = engine._prepared_epoch
    branch._birth_model = engine._birth_model
    branch._latest_primary_authority_sync = dict(engine._latest_primary_authority_sync)
    branch._primary_authority_sync_history = [dict(row) for row in engine._primary_authority_sync_history]
    return branch


def _sample_predictive_targets(
    tracks: Sequence[BernoulliKinematicTrack],
    world: PhysicalWorld,
    *,
    seed: int,
) -> tuple[TargetDynamicState, ...]:
    targets: list[TargetDynamicState] = []
    for index, track in enumerate(tracks):
        rng = np.random.default_rng(_stable_u64(seed, "latent_track", index))
        exists = bool(rng.random() < float(track.existence_probability))
        covariance = checked_covariance(np.asarray(track.covariance, dtype=float), require_spd=True)
        state = gaussian_draw(rng, np.asarray(track.mean, dtype=float), covariance)
        targets.append(
            TargetDynamicState(
                target_id=f"posterior_predictive_track_{index}",
                position_m=state[:3],
                velocity_m_s=state[3:6],
                acceleration_m_s2=np.zeros(3, dtype=float),
                existence_probability=float(track.existence_probability),
                exists=exists,
                maneuver_mode=ManeuverMode.CONSTANT_VELOCITY,
                mode_age_steps=0,
                mode_duration_steps=1,
                mean_rcs_m2=float(world.target.mean_rcs_m2),
                aspect_state_rad=0.0,
                frequency_scattering_slope=0.0,
                age_steps=int(track.age),
            )
        )
    return tuple(targets)


def _primary_observation(
    truth_world: PhysicalWorld,
    target: TargetDynamicState,
    action_id: int,
    *,
    epoch: int,
    seed: int,
    jammer: Jammer | None,
    planning_probability: PlanningDetectionProbability,
) -> tuple[ObservationRecord, ...]:
    if not target.exists:
        return ()
    action = next(item for item in truth_world.actions if item.action_id == action_id)
    factor = build_action_factor(truth_world, action, jammer)
    rows = _target_rows(factor)
    if rows.size == 0:
        return ()
    rng = np.random.default_rng(
        _stable_seed(seed, epoch, action_id, "primary_posterior_predictive")
    )
    success_probability = planning_probability.evaluate(
        action, factor
    ).planning_success_probability
    if rng.random() > success_probability:
        return ()
    covariance = checked_covariance(
        np.asarray(factor.covariance, dtype=float)[np.ix_(rows, rows)],
        require_spd=True,
    )
    noise = gaussian_draw(rng, np.zeros(rows.size), covariance)
    prediction = np.asarray(factor.prediction, dtype=float)
    value = prediction.copy()
    value[rows] += noise
    return (
        ObservationRecord(
            epoch=int(epoch),
            action_id=int(action_id),
            requested_fidelity="posterior_predictive",
            used_fidelity="posterior_predictive_R1_sufficient_statistic",
            detected=True,
            measurement_names=tuple(factor.measurement_names),
            value=value,
            prediction=prediction,
            residual=value - prediction,
            covariance=np.asarray(factor.covariance, dtype=float),
            observation_epoch=int(epoch),
            arrival_epoch=int(epoch),
            provenance_id=f"posterior_predictive:{seed}:{epoch}:{action_id}",
            factor_id=None,
            fallback_reason=None,
            metadata={
                "source": "frozen_controller_belief_posterior_predictive_draw",
                "simulator_truth_used": False,
            },
        ),
    )


def _one_replicate(
    engine: MultiTargetAssociationEngine,
    world: PhysicalWorld,
    action_ids: Sequence[int],
    prior_tracks: Sequence[BernoulliKinematicTrack],
    prior_report_track_ids: Sequence[str],
    *,
    epoch: int,
    replicate_seed: int,
    delivery_probability: Mapping[int, float],
    jammer: Jammer | None,
    planning_probability: PlanningDetectionProbability,
    cutoff_m: float,
    alpha: float,
) -> PosteriorPredictiveReplicate:
    branch = _planning_fork(engine, seed=replicate_seed)
    targets = _sample_predictive_targets(
        prior_tracks,
        world,
        seed=replicate_seed,
    )
    primary = targets[0]
    truth_world = _target_world(
        world,
        primary,
        target_id=primary.target_id,
    )
    truth_world = replace(
        truth_world,
        jammers=() if jammer is None else (jammer,),
    )
    for action_id in sorted(set(int(value) for value in action_ids)):
        delivery = float(delivery_probability.get(action_id, 1.0))
        if not 0.0 <= delivery <= 1.0 or not math.isfinite(delivery):
            raise ValueError("action delivery probability must lie in [0,1]")
        delivery_rng = np.random.default_rng(
            _stable_u64(replicate_seed, "delivery", action_id)
        )
        if delivery_rng.random() > delivery:
            continue
        observations = _primary_observation(
            truth_world,
            primary,
            action_id,
            epoch=epoch,
            seed=int(branch.config.seed),
            jammer=jammer,
            planning_probability=planning_probability,
        )
        captured = capture_measurement_return_set(
            world,
            truth_world,
            targets,
            action_id,
            observations,
            epoch=epoch,
            observation_master_seed=branch.config.observation_master_seed,
            planning_probability=planning_probability,
            controller_reference_mean=branch.tracks[0].mean,
            controller_reference_track_id=branch.tracks[0].track_id,
            provenance_id=f"posterior_predictive_return_set:{replicate_seed}:{epoch}:{action_id}",
        )
        update = branch.update_hypothetical_return_set(
            world,
            HypotheticalReturnSet(captured.payload, branch._hypothetical_context_id),
            epoch=epoch,
            planning_probability=planning_probability,
        )
        if str(update.get("status", "UNKNOWN_MISSING_STATUS")).startswith("UNKNOWN"):
            raise CovarianceModelError("PREDICTIVE_UPDATE_UNKNOWN:" + str(update.get("reason")))
    posterior = finite_set_cubature_risk(
        branch.finite_set_risk_tracks(),
        cutoff_m=cutoff_m,
        alpha=alpha,
    )
    truth_set = np.asarray(
        [target.position_m for target in targets if target.exists],
        dtype=float,
    ).reshape((-1, 3))
    prior_by_id = {track.track_id: track for track in prior_tracks}
    posterior_by_id = {
        track.track_id: track for track in branch.finite_set_risk_tracks()
    }
    prior_report = np.asarray(
        [prior_by_id[track_id].mean[:3] for track_id in prior_report_track_ids],
        dtype=float,
    ).reshape((-1, 3))
    posterior_report = np.asarray(
        [
            posterior_by_id[track_id].mean[:3]
            for track_id in posterior.report_track_ids
        ],
        dtype=float,
    ).reshape((-1, 3))
    return PosteriorPredictiveReplicate(
        prior_realized_power_loss_m2=gospa_power(
            truth_set,
            prior_report,
            cutoff_m=cutoff_m,
            alpha=alpha,
        ),
        posterior_realized_power_loss_m2=gospa_power(
            truth_set,
            posterior_report,
            cutoff_m=cutoff_m,
            alpha=alpha,
        ),
        posterior_self_power_risk_m2=posterior.power_risk_m2,
        posterior_report_track_ids=posterior.report_track_ids,
    )


def _upper_tail_mean(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("tail mean requires observations")
    if not 0.0 < probability <= 1.0:
        raise ValueError("tail probability must lie in (0,1]")
    ordered = sorted((float(value) for value in values), reverse=True)
    count = max(1, int(math.ceil(probability * len(ordered))))
    return float(np.mean(ordered[:count]))


def _lower_tail_mean(values: Sequence[float], probability: float) -> float:
    """Return the mean of the least favorable lower tail of a gain sample."""

    if not values:
        raise ValueError("tail mean requires observations")
    if not 0.0 < probability <= 1.0:
        raise ValueError("tail probability must lie in (0,1]")
    ordered = sorted(float(value) for value in values)
    count = max(1, int(math.ceil(probability * len(ordered))))
    return float(np.mean(ordered[:count]))


def posterior_predictive_schedule_risk(
    engine: MultiTargetAssociationEngine,
    world: PhysicalWorld,
    action_ids: Sequence[int],
    *,
    epoch: int,
    delivery_probability: Mapping[int, float],
    scenarios: Sequence[Jammer | None],
    replicates: int,
    seed: int,
    cutoff_m: float,
    alpha: float = 2.0,
    tail_probability: float = 0.25,
    tail_weight: float = 0.5,
    confidence_z: float = 1.96,
    planning_probability: PlanningDetectionProbability | None = None,
) -> dict[str, Any]:
    """Estimate one-step preposterior set risk with common random numbers.

    Random streams depend on the world/epoch/replicate/scenario/action, never on
    candidate index.  Candidates that share an action therefore share its
    hypothetical physical draws.  The live JPDA engine is digest-checked before
    and after evaluation.
    """

    planning_probability = planning_probability or PlanningDetectionProbability(
        engine.config
    )
    if not engine.enabled:
        raise ValueError("nonlocal task risk requires an enabled multi-target posterior")
    if not engine.tracks:
        raise ValueError("nonlocal task risk requires at least one posterior track")
    if replicates < 2:
        raise ValueError("at least two posterior-predictive replicates are required")
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail weight must lie in [0,1]")
    if confidence_z < 0.0 or not math.isfinite(confidence_z):
        raise ValueError("confidence multiplier must be finite and nonnegative")
    scenario_bank = tuple(scenarios) or (None,)
    before = _engine_digest(engine)
    risk_tracks = engine.finite_set_risk_tracks()
    prior = finite_set_cubature_risk(
        risk_tracks,
        cutoff_m=cutoff_m,
        alpha=alpha,
    )
    normalization_power_m2 = float(cutoff_m) ** 2 / float(alpha)
    zero_information_schedule = not action_ids or all(
        float(delivery_probability.get(int(action_id), 1.0)) <= 0.0
        for action_id in action_ids
    )
    if zero_information_schedule:
        return {
            "status": "passed",
            "estimator": "posterior_predictive_JPDA_plus_restricted_cubature_GOSPA",
            "prior_power_risk_m2": prior.power_risk_m2,
            "posterior_power_risk_m2": prior.power_risk_m2,
            "posterior_power_risk_standard_error_m2": 0.0,
            "posterior_power_risk_conservative_standard_error_m2": 0.0,
            "posterior_power_risk_upper_confidence_m2": prior.power_risk_m2,
            "posterior_self_power_risk_m2": prior.power_risk_m2,
            "self_risk_calibration_gap_m2": 0.0,
            "risk_reduction_fraction": 0.0,
            "risk_reduction_lcb_fraction": 0.0,
            "risk_normalization_power_m2": normalization_power_m2,
            "risk_normalization_contract": "GOSPA_cardinality_penalty_c_power_over_alpha",
            "gospa_cutoff_m": float(cutoff_m),
            "gospa_alpha": float(alpha),
            "scenario_rows": [],
            "replicates_per_scenario": 0,
            "noop_exact_identity": not action_ids,
            "zero_delivery_exact_identity": bool(action_ids),
            "live_track_digest_before": before,
            "live_track_digest_after": before,
            "live_posterior_mutated": False,
            "simulator_truth_used": False,
            "confirmed_track_count": len(engine.tracks),
            "tentative_birth_component_count": len(risk_tracks) - len(engine.tracks),
            "declared_birth_model_used": engine._birth_model is not None,
            "planning_detection_probability": planning_probability.receipt,
            "continuous_RFS_exact_claim_allowed": False,
            "scope": prior.scope,
        }

    scenario_rows: list[dict[str, Any]] = []
    paired_improvements_by_scenario: list[list[float]] = []
    posterior_losses_by_scenario: list[list[float]] = []
    prior_losses_by_scenario: list[list[float]] = []
    all_draws: list[float] = []
    for jammer in scenario_bank:
        draws: list[PosteriorPredictiveReplicate] = []
        for replicate in range(replicates):
            # The latent target set and standard-normal/uniform variates are
            # shared across registered adversary scenarios.  Scenario physics
            # changes the transform of those variates, not their identities.
            replicate_seed = _stable_u64(
                seed,
                world.world_id,
                epoch,
                replicate,
                "posterior_predictive_task_risk",
            )
            result = _one_replicate(
                engine,
                world,
                action_ids,
                risk_tracks,
                prior.report_track_ids,
                epoch=epoch,
                replicate_seed=replicate_seed,
                delivery_probability=delivery_probability,
                jammer=jammer,
                planning_probability=planning_probability,
                cutoff_m=cutoff_m,
                alpha=alpha,
            )
            draws.append(result)
        posterior_losses = [
            item.posterior_realized_power_loss_m2 for item in draws
        ]
        prior_losses = [item.prior_realized_power_loss_m2 for item in draws]
        paired_improvements = [
            prior_loss - posterior_loss
            for prior_loss, posterior_loss in zip(prior_losses, posterior_losses)
        ]
        posterior_losses_by_scenario.append(posterior_losses)
        prior_losses_by_scenario.append(prior_losses)
        paired_improvements_by_scenario.append(paired_improvements)
        self_risks = [item.posterior_self_power_risk_m2 for item in draws]
        mean = float(np.mean(posterior_losses))
        standard_error = float(
            np.std(posterior_losses, ddof=1) / math.sqrt(len(posterior_losses))
        )
        paired_standard_error = float(
            np.std(paired_improvements, ddof=1)
            / math.sqrt(len(paired_improvements))
        )
        scenario_rows.append(
            {
                "scenario": "nominal"
                if jammer is None
                else f"{jammer.kind}:{jammer.jammer_id}",
                "mean_power_risk_m2": mean,
                "standard_error_m2": standard_error,
                "paired_risk_reduction_standard_error_m2": paired_standard_error,
                "paired_risk_reduction_mean_m2": float(
                    np.mean(paired_improvements)
                ),
                "sampled_prior_power_loss_mean_m2": float(np.mean(prior_losses)),
                "posterior_self_power_risk_mean_m2": float(np.mean(self_risks)),
                "self_risk_calibration_gap_m2": float(np.mean(self_risks)) - mean,
                "upper_tail_power_risk_m2": _upper_tail_mean(
                    posterior_losses,
                    tail_probability,
                ),
            }
        )
        all_draws.extend(posterior_losses)

    # Build the risk-reduction estimator from paired losses on exactly the
    # same latent finite set.  This avoids subtracting a deterministic
    # cubature prior risk from a Monte Carlo posterior loss.  Scenario draws
    # share random numbers, so the robust functional is formed per replicate
    # before estimating its sampling uncertainty.
    reference_prior_losses = np.asarray(prior_losses_by_scenario[0], dtype=float)
    for scenario_prior_losses in prior_losses_by_scenario[1:]:
        if not np.array_equal(
            reference_prior_losses,
            np.asarray(scenario_prior_losses, dtype=float),
        ):
            raise RuntimeError("common-random-number prior losses changed by scenario")
    robust_posterior_losses = []
    robust_paired_improvements = []
    for replicate in range(replicates):
        posterior_losses = [
            scenario[replicate] for scenario in posterior_losses_by_scenario
        ]
        robust_posterior = (
            (1.0 - tail_weight) * float(np.mean(posterior_losses))
            + tail_weight * _upper_tail_mean(posterior_losses, tail_probability)
        )
        robust_posterior_losses.append(robust_posterior)
        robust_paired_improvements.append(
            float(reference_prior_losses[replicate]) - robust_posterior
        )
        gains = [
            scenario[replicate]
            for scenario in paired_improvements_by_scenario
        ]
        gain_functional = (
            (1.0 - tail_weight) * float(np.mean(gains))
            + tail_weight * _lower_tail_mean(gains, tail_probability)
        )
        if not math.isclose(
            robust_paired_improvements[-1],
            gain_functional,
            rel_tol=1.0e-12,
            abs_tol=1.0e-8,
        ):
            raise RuntimeError("paired gain and robust posterior risk are inconsistent")
    robust_risk = float(np.mean(robust_posterior_losses))
    robust_risk_standard_error = float(
        np.std(robust_posterior_losses, ddof=1)
        / math.sqrt(len(robust_posterior_losses))
    )
    robust_improvement = float(np.mean(robust_paired_improvements))
    robust_improvement_standard_error = float(
        np.std(robust_paired_improvements, ddof=1)
        / math.sqrt(len(robust_paired_improvements))
    )
    improvement_lcb = (
        robust_improvement - confidence_z * robust_improvement_standard_error
    )
    pooled_raw_draw_se = float(
        np.std(all_draws, ddof=1) / math.sqrt(len(all_draws))
    )
    conservative_se = max(
        robust_risk_standard_error,
        robust_improvement_standard_error,
    )
    upper_confidence = robust_risk + confidence_z * robust_risk_standard_error
    scale = max(normalization_power_m2, 1.0e-12)
    after = _engine_digest(engine)
    if before != after:
        raise RuntimeError("posterior-predictive risk evaluation mutated the live track set")
    return {
        "status": "passed",
        "estimator": "posterior_predictive_JPDA_plus_restricted_cubature_GOSPA",
        "prior_power_risk_m2": prior.power_risk_m2,
        "prior_gospa_risk_m": prior.gospa_risk_m,
        "posterior_power_risk_m2": robust_risk,
        "posterior_power_risk_standard_error_m2": robust_risk_standard_error,
        "pooled_raw_posterior_draw_standard_error_m2": pooled_raw_draw_se,
        "posterior_power_risk_conservative_standard_error_m2": conservative_se,
        "posterior_power_risk_upper_confidence_m2": upper_confidence,
        "paired_robust_risk_reduction_mean_m2": robust_improvement,
        "paired_robust_risk_reduction_standard_error_m2": (
            robust_improvement_standard_error
        ),
        "paired_robust_risk_reduction_lower_confidence_m2": improvement_lcb,
        "sampled_prior_power_loss_mean_m2": float(
            np.mean(reference_prior_losses)
        ),
        "paired_risk_identity_error_m2": float(
            np.mean(reference_prior_losses) - robust_risk - robust_improvement
        ),
        "posterior_self_power_risk_m2": float(
            np.mean(
                [row["posterior_self_power_risk_mean_m2"] for row in scenario_rows]
            )
        ),
        "self_risk_calibration_gap_m2": float(
            np.mean([row["self_risk_calibration_gap_m2"] for row in scenario_rows])
        ),
        "risk_reduction_fraction": float(robust_improvement / scale),
        "risk_reduction_lcb_fraction": float(improvement_lcb / scale),
        "risk_normalization_power_m2": normalization_power_m2,
        "risk_normalization_contract": "GOSPA_cardinality_penalty_c_power_over_alpha",
        "gospa_cutoff_m": float(cutoff_m),
        "gospa_alpha": float(alpha),
        "scenario_rows": scenario_rows,
        "replicates_per_scenario": int(replicates),
        "common_random_number_key_excludes_candidate_index": True,
        "live_track_digest_before": before,
        "live_track_digest_after": after,
        "live_posterior_mutated": False,
        "simulator_truth_used": False,
        "hypothetical_latents_sampled_from_frozen_controller_belief": True,
        "confirmed_track_count": len(engine.tracks),
        "tentative_birth_component_count": len(risk_tracks) - len(engine.tracks),
        "declared_birth_model_used": engine._birth_model is not None,
        "planning_detection_probability": planning_probability.receipt,
        "primary_estimand": (
            "E_frozen-belief,physical-observation[GOSPA^2(latent target set, "
            "Bayes report from the implemented JPDA posterior)]"
        ),
        "posterior_self_risk_used_for_selection": False,
        "cubature_prior_subtracted_from_monte_carlo_posterior": False,
        "continuous_RFS_exact_claim_allowed": False,
        "scope": prior.scope,
    }
