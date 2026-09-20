from __future__ import annotations

"""Leakage-free finite target-set tracking for the operational runtime.

The simulator uses target truth only to produce an *unlabelled* measurement
set.  The controller owns Bernoulli tracks, performs either a nearest-neighbour
negative control or JPDA, and exposes only its posterior track set to planning.
Truth identifiers are consumed solely by the offline metric accumulator.
"""

from dataclasses import dataclass, field, replace
import copy
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .covariance_coordinates import (
    CovarianceModelError, checked_covariance, covariance_solve,
    covariance_inverse, covariance_logdet, covariance_quadratic, gaussian_log_density,
    gaussian_draw, linear_gaussian_update, information_covariance_update,
)

from wideband_info_geometry.extensions_v33.rfs_jpda_multitarget import (
    enumerate_assignments,
)
from wideband_info_geometry.physics_v58.dynamic_episode import (
    EpisodeConfig,
    TargetDynamicState,
)
from wideband_info_geometry.physics_v58.information import prior_covariance_physical
from wideband_info_geometry.physics_v58.measurement import StateLayout, build_action_factor
from wideband_info_geometry.physics_v58.schema import PhysicalWorld, RadarAction

from .contracts import (
    MultiTargetAssociationMode,
    ObservationRecord,
    RuntimeConfig,
    to_builtin,
)
from .planning_detection_probability import PlanningDetectionProbability
from .availability import choose_fusion_node
from .measurement_return_sets import (
    UnlabelledReturnSet, HypotheticalReturnSet, ReturnSetContractError,
    _strict_integer, _stable_seed, _target_world, _target_rows,
)


@dataclass
class BernoulliKinematicTrack:
    track_id: str
    mean: np.ndarray
    covariance: np.ndarray
    existence_probability: float
    age: int = 0
    missed_epochs: int = 0
    origin: str = "confirmed_or_bootstrap_track"


@dataclass(frozen=True)
class ControllerBirthModel:
    probability_per_epoch: float
    death_probability_per_epoch: float
    offset_mean_m: tuple[float, float, float]
    offset_std_m: tuple[float, float, float]
    velocity_std_m_s: float
    maximum_targets: int

    @classmethod
    def from_episode(cls, episode: EpisodeConfig) -> "ControllerBirthModel":
        return cls(
            probability_per_epoch=float(episode.target_birth_probability),
            death_probability_per_epoch=float(episode.target_death_probability),
            offset_mean_m=tuple(float(value) for value in episode.target_birth_offset_mean_m),
            offset_std_m=tuple(float(value) for value in episode.target_birth_offset_std_m),
            velocity_std_m_s=float(episode.target_birth_velocity_std_m_s),
            maximum_targets=int(episode.max_targets),
        )


@dataclass
class TentativeBirth:
    mean: np.ndarray
    covariance: np.ndarray
    sources: set[tuple[int, int]] = field(default_factory=set)
    last_epoch: int = 0
    existence_probability: float = 0.25


@dataclass(frozen=True)
class JPDAHypothesisPosterior:
    beta: np.ndarray
    missed: np.ndarray
    normalization_error: float
    map_assignment: tuple[int, ...]
    map_probability: float
    joint_normalized_entropy: float


def _wrap(names: Sequence[str], residual: np.ndarray, coordinate_multipliers=None) -> np.ndarray:
    out = np.asarray(residual, dtype=float).copy()
    units = np.ones(len(names)) if coordinate_multipliers is None else np.asarray(coordinate_multipliers, dtype=float)
    if out.shape != (len(names),) or units.shape != out.shape or not np.all(np.isfinite(out)) or not np.all(np.isfinite(units)) or np.any(units<=0):
        raise CovarianceModelError("RESIDUAL_COORDINATE_CONTRACT")
    for index, name in enumerate(names):
        if name in {"azimuth_rad", "elevation_rad"}:
            period_half = math.pi * units[index]
            out[index] = (out[index] + period_half) % (2.0 * period_half) - period_half
    return out








def _log_gaussian(residual: np.ndarray, covariance: np.ndarray) -> float:
    return gaussian_log_density(residual, covariance)


def _jpda_probabilities(
    measurements: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    innovation_covariances: Sequence[np.ndarray],
    names: Sequence[str],
    existences: Sequence[float],
    detection_probabilities: Sequence[float],
    clutter_density: float,
    gate_sigma: float = 5.0,
    *,
    coordinate_multipliers=None,
) -> JPDAHypothesisPosterior:
    if clutter_density <= 0.0 or not math.isfinite(clutter_density):
        raise ValueError("JPDA clutter intensity must be finite and positive")
    if gate_sigma <= 0.0 or not math.isfinite(gate_sigma):
        raise ValueError("JPDA gate radius must be finite and positive")
    target_count = len(predictions)
    measurement_count = len(measurements)
    if not (len(innovation_covariances)==len(existences)==len(detection_probabilities)==target_count):
        raise CovarianceModelError("ASSOCIATION_TARGET_SCHEMA")
    for value in (*measurements, *predictions):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
            raise CovarianceModelError("ASSOCIATION_MEASUREMENT_SCHEMA")
    covariances = [checked_covariance(c, require_spd=True) for c in innovation_covariances]
    if any(c.shape != (len(names), len(names)) for c in covariances):
        raise CovarianceModelError("ASSOCIATION_COVARIANCE_SCHEMA")
    assignments = list(enumerate_assignments(target_count, measurement_count))
    log_weights: list[float] = []
    for assignment in assignments:
        used = {index for index in assignment if index >= 0}
        log_weight = (measurement_count - len(used)) * math.log(clutter_density)
        for target_index, measurement_index in enumerate(assignment):
            existence = float(np.clip(existences[target_index], 1.0e-9, 1.0))
            pd = float(np.clip(detection_probabilities[target_index], 1.0e-9, 1.0 - 1.0e-9))
            if measurement_index < 0:
                log_weight += math.log(max(1.0 - existence * pd, 1.0e-300))
                continue
            residual = _wrap(
                names,
                np.asarray(measurements[measurement_index]) - predictions[target_index],
                coordinate_multipliers,
            )
            covariance = covariances[target_index]
            mahalanobis_squared = covariance_quadratic(residual, covariance)
            if mahalanobis_squared > float(gate_sigma) ** 2:
                log_weight = -math.inf
                break
            log_weight += (
                math.log(existence)
                + math.log(pd)
                + _log_gaussian(residual, innovation_covariances[target_index])
            )
        log_weights.append(log_weight)
    maximum = max(log_weights, default=0.0)
    weights = np.exp(np.asarray(log_weights, dtype=float) - maximum)
    weights /= max(float(weights.sum()), 1.0e-300)
    beta = np.zeros((target_count, measurement_count), dtype=float)
    missed = np.zeros(target_count, dtype=float)
    for weight, assignment in zip(weights, assignments):
        for target_index, measurement_index in enumerate(assignment):
            if measurement_index < 0:
                missed[target_index] += weight
            else:
                beta[target_index, measurement_index] += weight
    normalization_error = float(
        np.max(np.abs(beta.sum(axis=1) + missed - 1.0))
    ) if target_count else 0.0
    map_index = int(np.argmax(weights)) if weights.size else 0
    positive_weights = weights[weights > 0.0]
    entropy = -float(np.sum(positive_weights * np.log(positive_weights)))
    entropy_normalizer = math.log(max(len(assignments), 2))
    return JPDAHypothesisPosterior(
        beta=beta,
        missed=missed,
        normalization_error=normalization_error,
        map_assignment=(assignments[map_index] if assignments else ()),
        map_probability=(float(weights[map_index]) if weights.size else 1.0),
        joint_normalized_entropy=float(entropy / entropy_normalizer),
    )


def _ellipsoidal_gate_volume(
    innovation_covariance: np.ndarray,
    gate_sigma: float,
) -> float:
    """Lebesgue volume of a Mahalanobis gate in measurement coordinates."""

    covariance = checked_covariance(innovation_covariance, require_spd=True)
    dimension = int(covariance.shape[0])
    logdet = covariance_logdet(covariance)
    if dimension <= 0 or gate_sigma <= 0 or not math.isfinite(gate_sigma):
        raise ValueError("innovation covariance does not define a gate volume")
    log_unit_ball = (
        0.5 * dimension * math.log(math.pi)
        - math.lgamma(0.5 * dimension + 1.0)
    )
    log_volume = (
        log_unit_ball
        + dimension * math.log(float(gate_sigma))
        + 0.5 * float(logdet)
    )
    if not -700.0 < log_volume < 700.0:
        raise CovarianceModelError("GATE_VOLUME_NUMERIC_RANGE")
    return float(math.exp(log_volume))


def _physical_clutter_intensity(
    expected_false_returns: float,
    innovation_covariances: Sequence[np.ndarray],
    *,
    gate_sigma: float,
    fallback_density: float,
) -> tuple[float, float]:
    """Convert an expected false-return count to a dimensioned intensity."""

    if not innovation_covariances or not math.isfinite(expected_false_returns) or expected_false_returns < 0:
        raise CovarianceModelError("CLUTTER_EXPECTATION_OR_DOMAIN")
    volumes = [
        _ellipsoidal_gate_volume(covariance, gate_sigma)
        for covariance in innovation_covariances
    ]
    representative_volume = float(np.exp(np.mean(np.log(volumes))))
    expected = max(float(expected_false_returns), 0.0)
    if expected > 0.0:
        intensity = expected / representative_volume
    else:
        intensity = float(fallback_density)
    if not math.isfinite(intensity) or intensity <= 0:
        raise CovarianceModelError("CLUTTER_INTENSITY_NUMERIC_RANGE")
    return float(intensity), representative_volume


def _nearest_neighbor_probabilities(
    measurements: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    innovation_covariances: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    coordinate_multipliers=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(predictions) != len(innovation_covariances):
        raise CovarianceModelError("NEAREST_NEIGHBOR_TARGET_SCHEMA")
    for value in (*measurements, *predictions):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
            raise CovarianceModelError("NEAREST_NEIGHBOR_MEASUREMENT_SCHEMA")
    beta = np.zeros((len(predictions), len(measurements)), dtype=float)
    missed = np.ones(len(predictions), dtype=float)
    pairs: list[tuple[float, int, int]] = []
    for target_index, (prediction, covariance) in enumerate(
        zip(predictions, innovation_covariances)
    ):
        for measurement_index, measurement in enumerate(measurements):
            residual = _wrap(names, np.asarray(measurement) - prediction, coordinate_multipliers)
            pairs.append(
                (covariance_quadratic(residual, covariance), target_index, measurement_index)
            )
    assigned_targets: set[int] = set()
    assigned_measurements: set[int] = set()
    for distance, target_index, measurement_index in sorted(pairs):
        if distance > 25.0:
            continue
        if target_index in assigned_targets or measurement_index in assigned_measurements:
            continue
        beta[target_index, measurement_index] = 1.0
        missed[target_index] = 0.0
        assigned_targets.add(target_index)
        assigned_measurements.add(measurement_index)
    return beta, missed, 0.0


def _bernoulli_existence_update(
    prior_existence: float,
    detection_probability: float,
    association_probability: float,
    missed_probability: float,
) -> float:
    """JPDA Bernoulli existence update from marginal association events.

    A MAP assignment may be used to keep kinematic means separated under high
    association entropy, but it is not a replacement for the marginalized
    existence posterior.  Hardening the existence event to the MAP branch
    prematurely deletes plausible tracks and breaks the Bernoulli semantics.
    """

    prior = float(np.clip(prior_existence, 0.0, 1.0))
    pd = float(np.clip(detection_probability, 0.0, 1.0))
    associated = float(np.clip(association_probability, 0.0, 1.0))
    missed = float(np.clip(missed_probability, 0.0, 1.0))
    total = associated + missed
    if total <= 0.0:
        raise ValueError("association and missed probabilities have zero mass")
    associated /= total
    missed /= total
    missed_existence = prior * (1.0 - pd) / max(1.0 - prior * pd, 1.0e-12)
    return float(np.clip(associated + missed * missed_existence, 0.0, 0.999))


def _minimum_position_assignment(
    truth_positions: Sequence[np.ndarray],
    track_positions: Sequence[np.ndarray],
) -> tuple[tuple[int, int], ...]:
    if not truth_positions or not track_positions:
        return ()
    truth_count = len(truth_positions)
    track_count = len(track_positions)
    best_cost = math.inf
    best: tuple[tuple[int, int], ...] = ()
    if truth_count <= track_count:
        for chosen in itertools.permutations(range(track_count), truth_count):
            pairs = tuple((index, chosen[index]) for index in range(truth_count))
            cost = sum(
                float(np.linalg.norm(truth_positions[left] - track_positions[right]))
                for left, right in pairs
            )
            if cost < best_cost:
                best_cost, best = cost, pairs
    else:
        for chosen in itertools.permutations(range(truth_count), track_count):
            pairs = tuple((chosen[index], index) for index in range(track_count))
            cost = sum(
                float(np.linalg.norm(truth_positions[left] - track_positions[right]))
                for left, right in pairs
            )
            if cost < best_cost:
                best_cost, best = cost, pairs
    return best


def _poisson_binomial_cardinality_probabilities(
    existence_probabilities: Sequence[float],
) -> np.ndarray:
    """Exact cardinality law of independent Bernoulli RFS components."""

    distribution = np.asarray([1.0], dtype=float)
    for value in existence_probabilities:
        probability = float(np.clip(value, 0.0, 1.0))
        updated = np.zeros(distribution.size + 1, dtype=float)
        updated[:-1] += (1.0 - probability) * distribution
        updated[1:] += probability * distribution
        distribution = updated
    total = float(distribution.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("invalid Poisson-binomial cardinality distribution")
    return distribution / total


def _cardinality_ranked_probability_score(
    distribution: np.ndarray, truth_cardinality: int
) -> float:
    """Return the discrete ranked probability score for target cardinality.

    The score compares the complete posterior cardinality CDF with the
    realized-cardinality step CDF.  It is proper for the distribution and
    remains sensitive to how far misplaced posterior mass lies from truth.
    """

    pmf = np.asarray(distribution, dtype=float).reshape(-1)
    if pmf.size == 0 or not np.all(np.isfinite(pmf)) or np.any(pmf < -1.0e-12):
        raise ValueError("invalid cardinality distribution for RPS")
    truth = int(truth_cardinality)
    if truth < 0:
        raise ValueError("truth cardinality must be nonnegative")
    support = max(pmf.size, truth + 2)
    padded = np.zeros(support, dtype=float)
    padded[: pmf.size] = np.clip(pmf, 0.0, None)
    total = float(padded.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("invalid cardinality distribution for RPS")
    padded /= total
    posterior_cdf = np.cumsum(padded)
    observed_cdf = (np.arange(support) >= truth).astype(float)
    return float(np.sum(np.square(posterior_cdf - observed_cdf)))


class MultiTargetAssociationEngine:
    """Controller-owned Bernoulli track set with offline truth-only metrics."""

    def __init__(
        self,
        base_world: PhysicalWorld,
        config: RuntimeConfig,
        episode_config: EpisodeConfig | None = None,
    ) -> None:
        self.config = config
        self.receiver_id = choose_fusion_node(base_world, config.fusion_node_id)
        self._hypothetical_context_id: str | None = None
        self.enabled = (
            config.multi_target_association_mode
            is not MultiTargetAssociationMode.DISABLED
        )
        layout = StateLayout.from_world(base_world)
        covariance = prior_covariance_physical(base_world, layout)[:6, :6]
        mean = np.r_[base_world.target.position_m, base_world.target.velocity_m_s]
        self.tracks: list[BernoulliKinematicTrack] = [
            BernoulliKinematicTrack(
                track_id="controller_track_0",
                mean=np.asarray(mean, dtype=float),
                covariance=checked_covariance(covariance),
                existence_probability=0.95,
            )
        ]
        self._bootstrapped = False
        self._next_track_id = 1
        self._tentative_births: list[TentativeBirth] = []
        self._birth_promotions_by_epoch: dict[int, int] = {}
        self.history: list[dict[str, Any]] = []
        self._consumed_return_set_receipts: set[str] = set()
        self._consumed_return_set_provenance_ids: set[str] = set()
        self._previous_truth_assignment: dict[str, str] = {}
        self._id_switches = 0
        self._prepared_epoch: int | None = None
        self._latest_primary_authority_sync: dict[str, Any] = {
            "status": "not_applied",
            "truth_or_future_event_read": False,
        }
        self._primary_authority_sync_history: list[dict[str, Any]] = []
        self._birth_model = (
            ControllerBirthModel.from_episode(episode_config)
            if config.multi_target_birth_prior_enabled
            and episode_config is not None
            else None
        )
        if config.multi_target_birth_prior_enabled:
            if episode_config is None:
                raise ValueError(
                    "the controller birth prior requires an episode transition model"
                )
            if self._birth_model is None or self._birth_model.probability_per_epoch <= 0.0:
                raise ValueError(
                    "the enabled controller birth prior requires a positive declared hazard"
                )
            if min(self._birth_model.maximum_targets, config.multi_target_count) < 2:
                raise ValueError(
                    "the controller birth prior requires room for at least two targets"
                )

    def _bootstrap(self, controller_world: PhysicalWorld) -> None:
        if self._bootstrapped:
            return
        self.tracks[0].mean = np.r_[
            controller_world.target.position_m,
            controller_world.target.velocity_m_s,
        ].astype(float)
        self._bootstrapped = True

    def _predict(self) -> None:
        dt = float(self.config.dt_s)
        transition = np.block(
            [
                [np.eye(3), dt * np.eye(3)],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        )
        acceleration_map = np.vstack(
            [0.5 * dt * dt * np.eye(3), dt * np.eye(3)]
        )
        process = acceleration_map @ (4.0 * np.eye(3)) @ acceleration_map.T
        predictions = [(transition @ track.mean, checked_covariance(
            transition @ track.covariance @ transition.T + process
        )) for track in self.tracks]
        for track, (mean, covariance) in zip(self.tracks, predictions):
            track.mean = mean
            track.covariance = covariance
            if self._birth_model is not None:
                track.existence_probability *= (
                    1.0 - self._birth_model.death_probability_per_epoch
                )
            track.age += 1

    def _inject_declared_birth_prior(self, epoch: int) -> bool:
        model = self._birth_model
        maximum = min(
            self.config.multi_target_count,
            model.maximum_targets if model is not None else 0,
        )
        if model is None or len(self.tracks) >= maximum:
            return False
        track_id = f"controller_birth_prior_epoch_{int(epoch)}"
        if any(track.track_id == track_id for track in self.tracks):
            return False
        anchor = self.tracks[0]
        mean = np.asarray(anchor.mean, dtype=float).copy()
        mean[:3] += np.asarray(model.offset_mean_m, dtype=float)
        covariance = np.asarray(anchor.covariance, dtype=float).copy()
        covariance[:3, :3] += np.diag(np.square(model.offset_std_m))
        covariance[3:, 3:] += (
            float(model.velocity_std_m_s) ** 2 * np.eye(3)
        )
        self.tracks.append(
            BernoulliKinematicTrack(
                track_id=track_id,
                mean=mean,
                covariance=checked_covariance(covariance),
                existence_probability=float(model.probability_per_epoch),
                origin="declared_stochastic_birth_prior",
            )
        )
        return True

    def prepare_epoch(
        self,
        controller_world: PhysicalWorld,
        *,
        epoch: int,
    ) -> Mapping[str, Any]:
        """Advance the controller posterior before it is exposed to planning."""

        if not self.enabled:
            return {"status": "disabled"}
        if self._prepared_epoch == int(epoch):
            return {"status": "passed", "already_prepared": True}
        if self._prepared_epoch is not None and int(epoch) != self._prepared_epoch + 1:
            raise RuntimeError("multi-target prediction epochs are not consecutive")
        self._bootstrap(controller_world)
        if int(epoch) > 0:
            self._predict()
        birth_injected = (
            self._inject_declared_birth_prior(int(epoch))
            if int(epoch) > 0
            else False
        )
        self._prepared_epoch = int(epoch)
        return {
            "status": "passed",
            "epoch": int(epoch),
            "prediction_applied_before_decision": int(epoch) > 0,
            "declared_birth_prior_injected": birth_injected,
            "forced_event_schedule_read_by_controller": False,
        }

    def synchronize_primary_from_authoritative_graph(
        self,
        controller_world: PhysicalWorld,
        covariance_physical: np.ndarray,
        *,
        epoch: int,
    ) -> Mapping[str, Any]:
        """Use the sole graph posterior as the primary JPDA kinematic prior.

        The Bernoulli layer retains existence, cardinality, and association
        uncertainty. It must not maintain a second independently drifting
        kinematic posterior for the same primary target, because that posterior
        would silently determine validation gates and nonlocal planning risk.
        The input world is the controller projection of the authoritative
        DynamicFactorGraph; simulator truth and event schedules are absent.
        """

        if not self.enabled:
            audit = {
                "status": "disabled",
                "epoch": int(epoch),
                "truth_or_future_event_read": False,
            }
            self._latest_primary_authority_sync = audit
            return audit
        if self._prepared_epoch != int(epoch):
            raise RuntimeError(
                "primary graph synchronization requires the matching prepared epoch"
            )
        if not self.tracks:
            raise RuntimeError("primary graph synchronization has no Bernoulli track")
        covariance = np.asarray(covariance_physical, dtype=float)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError(
                "authoritative primary covariance must be a finite 6x6 matrix"
            )
        covariance = checked_covariance(covariance)
        primary = self.tracks[0]
        prior_mean = np.asarray(primary.mean, dtype=float).copy()
        prior_covariance = np.asarray(primary.covariance, dtype=float).copy()
        graph_mean = np.r_[
            controller_world.target.position_m,
            controller_world.target.velocity_m_s,
        ].astype(float)
        if graph_mean.shape != (6,) or not np.all(np.isfinite(graph_mean)):
            raise ValueError("authoritative primary mean must be finite")
        primary.mean = graph_mean
        primary.covariance = covariance
        audit = {
            "status": "passed",
            "epoch": int(epoch),
            "track_id": str(primary.track_id),
            "source": "authoritative_dynamic_factor_graph_controller_projection",
            "mean_replacement_norm_physical": float(
                np.linalg.norm(graph_mean - prior_mean)
            ),
            "covariance_replacement_frobenius_physical": float(
                np.linalg.norm(covariance - prior_covariance, ord="fro")
            ),
            "existence_probability_preserved": True,
            "secondary_track_posteriors_preserved": True,
            "single_authoritative_primary_kinematic_posterior": True,
            "truth_or_future_event_read": False,
        }
        self._latest_primary_authority_sync = audit
        self._primary_authority_sync_history.append(dict(audit))
        return audit

    @staticmethod
    def _action_by_id(world: PhysicalWorld, action_id: int) -> RadarAction:
        return next(action for action in world.actions if action.action_id == action_id)


    def _spawn_from_unassigned(
        self,
        measurements: Sequence[np.ndarray],
        beta: np.ndarray,
        names: Sequence[str],
        models: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        *,
        epoch: int,
        action_id: int,
    ) -> tuple[int, int]:
        if len(self.tracks) >= self.config.multi_target_count or not models:
            return 0, len(self._tentative_births)
        self._tentative_births = [
            item
            for item in self._tentative_births
            if int(epoch) - int(item.last_epoch)
            <= max(
                1,
                int(self.config.jpda_birth_confirmation_min_distinct_epochs) - 1,
            )
        ]
        association_mass = beta.sum(axis=0) if beta.size else np.zeros(len(measurements))
        promoted = 0
        source = (int(epoch), int(action_id))
        for measurement_index in np.argsort(association_mass):
            unassigned_probability = float(
                np.clip(1.0 - association_mass[measurement_index], 0.0, 1.0)
            )
            if (
                unassigned_probability
                < self.config.jpda_birth_unassigned_probability_min
            ):
                break
            if len(self.tracks) >= self.config.multi_target_count:
                break
            prediction, h, covariance, innovation_covariance = models[0]
            residual = _wrap(names, measurements[int(measurement_index)] - prediction)
            anchor = self.tracks[0]
            gain, conditional, _ = linear_gaussian_update(anchor.covariance, h, covariance)
            mean = anchor.mean + gain @ residual
            covariance_new = checked_covariance(4.0 * conditional)
            best_index = None
            best_distance = math.inf
            for index, tentative in enumerate(self._tentative_births):
                difference = mean - tentative.mean
                joint_covariance = checked_covariance(covariance_new + tentative.covariance, require_spd=True)
                distance = covariance_quadratic(difference, joint_covariance)
                if distance < best_distance:
                    best_index, best_distance = index, distance
            if best_index is None or best_distance > 16.0:
                self._tentative_births.append(
                    TentativeBirth(
                        mean=mean,
                        covariance=covariance_new,
                        sources={source},
                        last_epoch=int(epoch),
                        existence_probability=float(
                            np.clip(
                                0.20 + 0.30 * unassigned_probability,
                                0.20,
                                0.49,
                            )
                        ),
                    )
                )
                continue
            tentative = self._tentative_births[best_index]
            if source in tentative.sources:
                continue
            # Covariance intersection avoids assuming independence between
            # two selected actions that may share a receiver or waveform.
            inverse_left = covariance_inverse(tentative.covariance)
            inverse_right = covariance_inverse(covariance_new)
            fused_covariance = checked_covariance(covariance_inverse(0.5 * inverse_left + 0.5 * inverse_right))
            fused_mean = fused_covariance @ (
                0.5 * inverse_left @ tentative.mean
                + 0.5 * inverse_right @ mean
            )
            tentative.mean = fused_mean
            tentative.covariance = fused_covariance
            tentative.sources.add(source)
            tentative.last_epoch = int(epoch)
            evidence_existence = float(
                np.clip(0.20 + 0.30 * unassigned_probability, 0.20, 0.49)
            )
            tentative.existence_probability = float(
                min(
                    0.49,
                    1.0
                    - (1.0 - tentative.existence_probability)
                    * (1.0 - evidence_existence),
                )
            )
            distinct_epochs = {source_epoch for source_epoch, _ in tentative.sources}
            if len(distinct_epochs) < int(
                self.config.jpda_birth_confirmation_min_distinct_epochs
            ):
                continue
            if self._birth_promotions_by_epoch.get(int(epoch), 0) >= int(
                self.config.jpda_max_confirmed_births_per_epoch
            ):
                continue
            self.tracks.append(
                BernoulliKinematicTrack(
                    track_id=f"controller_track_{self._next_track_id}",
                    mean=tentative.mean.copy(),
                    covariance=tentative.covariance.copy(),
                    existence_probability=float(
                        np.clip(
                            max(
                                tentative.existence_probability,
                                0.50 + 0.20 * unassigned_probability,
                            ),
                            0.51,
                            0.70,
                        )
                    ),
                    origin="confirmed_measurement_driven_birth",
                )
            )
            self._next_track_id += 1
            promoted += 1
            self._birth_promotions_by_epoch[int(epoch)] = (
                self._birth_promotions_by_epoch.get(int(epoch), 0) + 1
            )
            self._tentative_births.pop(best_index)
            if len(self.tracks) >= self.config.multi_target_count:
                break
        maximum_tentatives = max(2, 2 * self.config.multi_target_count)
        self._tentative_births = self._tentative_births[:maximum_tentatives]
        return promoted, len(self._tentative_births)

    def finite_set_risk_tracks(
        self,
        maximum_tracks: int = 4,
    ) -> tuple[BernoulliKinematicTrack, ...]:
        """Return bounded controller-only Bernoulli support for task risk."""

        if maximum_tracks <= 0:
            raise ValueError("finite-set risk support bound must be positive")
        tracks = list(self.tracks[:maximum_tracks])
        remaining = max(
            0,
            min(self.config.multi_target_count, maximum_tracks) - len(tracks),
        )
        tentative = sorted(
            self._tentative_births,
            key=lambda item: (
                -float(item.existence_probability),
                tuple(sorted(item.sources)),
            ),
        )[:remaining]
        for index, item in enumerate(tentative):
            tracks.append(
                BernoulliKinematicTrack(
                    track_id=f"controller_tentative_birth_{index}",
                    mean=np.asarray(item.mean, dtype=float).copy(),
                    covariance=np.asarray(item.covariance, dtype=float).copy(),
                    existence_probability=float(item.existence_probability),
                    origin="unconfirmed_measurement_driven_birth",
                )
            )
        return tuple(tracks)

    def _update_return_set_action(
        self, controller_world, payload, *, epoch, planning_probability=None,
    ) -> dict[str, Any]:
        # Numerical failure may occur after another track was updated. Preserve
        # the complete action-local controller state instead of returning a
        # partially changed belief or an invented zero-risk success.
        saved = copy.deepcopy((self.tracks, self._tentative_births,
                               self._birth_promotions_by_epoch, self._next_track_id))
        try:
            return self._update_return_set_action_impl(controller_world, payload, epoch=epoch,
                planning_probability=planning_probability)
        except CovarianceModelError as error:
            (self.tracks, self._tentative_births,
             self._birth_promotions_by_epoch, self._next_track_id) = saved
            return {"action_id":int(payload.action_id), "status":"UNKNOWN_NUMERIC_MODEL",
                    "reason":str(error), "controller_state_rolled_back":True,
                    "risk":None, "unit_numeric_contract":"normalized_covariance_v1"}
        except ReturnSetContractError as error:
            (self.tracks, self._tentative_births,
             self._birth_promotions_by_epoch, self._next_track_id) = saved
            return {"action_id": int(payload.action_id), "status": "UNKNOWN_RETURN_SET_MODEL",
                    "reason": str(error), "controller_state_rolled_back": True, "risk": None}

    def _update_return_set_action_impl(
        self,
        controller_world: PhysicalWorld,
        payload: UnlabelledReturnSet,
        *,
        epoch: int,
        planning_probability: PlanningDetectionProbability | None = None,
    ) -> dict[str, Any]:
        if type(payload) is not UnlabelledReturnSet:
            raise ReturnSetContractError("RETURN_SET_PAYLOAD_TYPE")
        if payload.observation_epoch != _strict_integer(epoch, "UPDATE_EPOCH"):
            raise ReturnSetContractError("LATE_OR_FUTURE_RETURN_SET_UNSUPPORTED")
        action_id = payload.action_id
        action = self._action_by_id(controller_world, action_id)
        if not payload.measurement_names:
            return {"action_id": action_id, "status": "not_target_sensitive",
                    "return_set_provenance_id": payload.provenance_id}
        models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        names: tuple[str, ...] = ()
        detection_probabilities: list[float] = []
        controller_expected_false_returns: list[float] = []
        for track in self.tracks:
            track_world = _target_world(
                controller_world,
                track.mean,
                target_id=track.track_id,
            )
            factor = build_action_factor(track_world, action, None)
            rows = _target_rows(factor)
            if rows.size == 0:
                raise ReturnSetContractError("CAPTURE_CONTROLLER_TARGET_SCHEMA_CHANGED")
            row_names = tuple(factor.measurement_names[index] for index in rows)
            if names and row_names != names:
                raise ReturnSetContractError("TARGET_SPECIFIC_MEASUREMENT_SCHEMA_CHANGED")
            names = row_names
            prediction = np.asarray(factor.prediction, dtype=float)[rows]
            h = np.asarray(factor.jacobian_physical, dtype=float)[rows, :6]
            covariance = np.asarray(factor.covariance, dtype=float)[np.ix_(rows, rows)]
            innovation_covariance = checked_covariance(h @ track.covariance @ h.T + covariance, require_spd=True)
            models.append((prediction, h, covariance, innovation_covariance))
            controller_expected_false_returns.append(float(factor.link_budget.expected_false_returns))
            detection_probabilities.append(
                float(factor.link_budget.detection_probability)
                if planning_probability is None
                else planning_probability.evaluate(
                    action, factor
                ).planning_success_probability
            )
        if names != payload.measurement_names:
            raise ReturnSetContractError("CAPTURE_CONTROLLER_MEASUREMENT_SCHEMA_CHANGED")
        measurements = [np.asarray(row, dtype=float) for row in payload.measurements]
        predictions = [model[0] for model in models]
        innovations = [model[3] for model in models]
        clutter_intensity, clutter_gate_volume = _physical_clutter_intensity(
            max(controller_expected_false_returns, default=0.0),
            innovations,
            gate_sigma=self.config.jpda_gate_sigma,
            fallback_density=self.config.jpda_clutter_density,
        )
        diagnostic = _jpda_probabilities(
            measurements,
            predictions,
            innovations,
            names,
            [track.existence_probability for track in self.tracks],
            detection_probabilities,
            clutter_intensity,
            self.config.jpda_gate_sigma,
        )
        diagnostic_beta = diagnostic.beta
        diagnostic_missed = diagnostic.missed
        diagnostic_normalization_error = diagnostic.normalization_error
        if self.config.multi_target_association_mode is MultiTargetAssociationMode.JPDA:
            beta = diagnostic_beta
            missed = diagnostic_missed
            normalization_error = diagnostic_normalization_error
        else:
            beta, missed, normalization_error = _nearest_neighbor_probabilities(
                measurements,
                predictions,
                innovations,
                names,
            )
        diagnostic_hypotheses = np.c_[diagnostic_beta, diagnostic_missed]
        diagnostic_entropy = []
        for probabilities in diagnostic_hypotheses:
            positive = probabilities[probabilities > 0.0]
            entropy = -float(np.sum(positive * np.log(positive)))
            normalizer = math.log(max(probabilities.size, 2))
            diagnostic_entropy.append(entropy / normalizer)
        ambiguous_tracks = sum(
            float(np.max(probabilities)) < 0.90
            for probabilities in diagnostic_hypotheses
            if probabilities.size > 1
        )
        original_tracks = list(self.tracks)
        coalescence_controlled_tracks = 0
        for target_index, (track, model) in enumerate(zip(original_tracks, models)):
            prediction, h, measurement_covariance, innovation_covariance = model
            gain, conditional_covariance, _ = linear_gaussian_update(
                track.covariance, h, measurement_covariance)
            weighted_residual = np.zeros(len(names), dtype=float)
            residual_spread = np.zeros((len(names), len(names)), dtype=float)
            for measurement_index, measurement in enumerate(measurements):
                residual = _wrap(names, measurement - prediction)
                weighted_residual += beta[target_index, measurement_index] * residual
                residual_spread += beta[target_index, measurement_index] * np.outer(
                    residual, residual
                )
            association_probability = float(1.0 - missed[target_index])
            spread = checked_covariance(
                residual_spread - np.outer(weighted_residual, weighted_residual),
            )
            use_joint_map_mean = bool(
                self.config.multi_target_association_mode
                is MultiTargetAssociationMode.JPDA
                and diagnostic_entropy[target_index]
                >= self.config.jpda_coalescence_entropy_threshold
            )
            map_measurement = (
                diagnostic.map_assignment[target_index]
                if target_index < len(diagnostic.map_assignment)
                else -1
            )
            selected_residual = weighted_residual
            if use_joint_map_mean:
                coalescence_controlled_tracks += 1
                selected_residual = (
                    _wrap(names, measurements[map_measurement] - prediction)
                    if map_measurement >= 0
                    else np.zeros(len(names), dtype=float)
                )
            selected_delta = gain @ selected_residual
            # Preserve the complete JPDA marginal second moment around the
            # selected mean.  Under high entropy the mean follows the globally
            # consistent MAP assignment, while covariance still accounts for
            # every joint-association branch instead of becoming overconfident.
            covariance_new = (
                missed[target_index]
                * (
                    track.covariance
                    + np.outer(selected_delta, selected_delta)
                )
            )
            for measurement_index, measurement in enumerate(measurements):
                residual = _wrap(names, measurement - prediction)
                component_delta = gain @ residual
                difference = component_delta - selected_delta
                covariance_new += beta[target_index, measurement_index] * (
                    conditional_covariance + np.outer(difference, difference)
                )
            track.mean = track.mean + selected_delta
            track.covariance = checked_covariance(covariance_new)
            pd = float(np.clip(detection_probabilities[target_index], 0.0, 1.0))
            existence_association_probability = association_probability
            existence_missed_probability = float(missed[target_index])
            track.existence_probability = _bernoulli_existence_update(
                track.existence_probability,
                pd,
                existence_association_probability,
                existence_missed_probability,
            )
            track.missed_epochs = (
                0
                if existence_association_probability >= 0.20
                else track.missed_epochs + 1
            )
        spawned, tentative_count = self._spawn_from_unassigned(
            measurements,
            beta,
            names,
            models,
            epoch=epoch,
            action_id=action_id,
        )
        retained = [
            track
            for track in self.tracks
            if track.existence_probability >= 0.05 or track.age <= 1
        ][: self.config.multi_target_count]
        if not retained and original_tracks:
            fallback = max(
                original_tracks,
                key=lambda item: item.existence_probability,
            )
            fallback.existence_probability = 0.05
            retained = [fallback]
        self.tracks = retained
        return {
            "action_id": action_id,
            "status": "passed",
            "measurement_count": len(measurements),
            "association_mode": self.config.multi_target_association_mode.value,
            "association_normalization_error": normalization_error,
            "preassociation_JPDA_normalization_error": (
                diagnostic_normalization_error
            ),
            "preassociation_mean_normalized_entropy": float(
                np.mean(diagnostic_entropy)
            )
            if diagnostic_entropy
            else 0.0,
            "preassociation_ambiguous_track_count": int(ambiguous_tracks),
            "joint_assignment_MAP": list(diagnostic.map_assignment),
            "joint_assignment_MAP_probability": diagnostic.map_probability,
            "joint_assignment_normalized_entropy": (
                diagnostic.joint_normalized_entropy
            ),
            "coalescence_controlled_track_count": int(
                coalescence_controlled_tracks
            ),
            "coalescence_MAP_mean_preserves_marginal_existence_update": True,
            "dimensionally_normalized_clutter_intensity": clutter_intensity,
            "association_gate_volume_measurement_units": clutter_gate_volume,
            "association_gate_sigma": self.config.jpda_gate_sigma,
            "association_gate_applied_to_joint_hypotheses": True,
            "clutter_intensity_source": (
                "controller_model_expected_false_returns_per_gate_volume"
                if max(controller_expected_false_returns, default=0.0) > 0.0
                else "explicit_fallback_density"
            ),
            "clutter_model_expected_false_returns": max(controller_expected_false_returns, default=0.0),
            "generator_capture_audit_used_for_clutter_parameter": False,
            "unit_numeric_contract": "normalized_covariance_v1",
            "spawned_tracks": spawned,
            "tentative_birth_count": int(tentative_count),
            "birth_confirmation_contract": (
                "registered_minimum_distinct_epochs_with_covariance_intersection"
            ),
            "birth_confirmation_minimum_distinct_epochs": int(
                self.config.jpda_birth_confirmation_min_distinct_epochs
            ),
            "receiver_return_set": {
                "provenance_id": payload.provenance_id,
                "observation_epoch": payload.observation_epoch,
                "complete": payload.complete,
                "measurement_count": len(payload.measurements),
            },
            "capture_audit_excluded_from_receiver": True,
            "beta": beta.tolist(),
            "missed_probability": missed.tolist(),
        }

    def _offline_metrics(
        self,
        truth_targets: Sequence[TargetDynamicState],
    ) -> dict[str, Any]:
        truths = [target for target in truth_targets if target.exists]
        estimates = [track for track in self.tracks if track.existence_probability >= 0.5]
        existence_probabilities = [
            float(track.existence_probability) for track in self.tracks
        ]
        cardinality_distribution = _poisson_binomial_cardinality_probabilities(
            existence_probabilities
        )
        truth_cardinality = len(truths)
        truth_probability = (
            float(cardinality_distribution[truth_cardinality])
            if truth_cardinality < cardinality_distribution.size
            else 0.0
        )
        expected_cardinality = float(sum(existence_probabilities))
        truth_positions = [np.asarray(target.position_m, dtype=float) for target in truths]
        estimate_positions = [track.mean[:3] for track in estimates]
        pairs = _minimum_position_assignment(truth_positions, estimate_positions)
        cutoff = 1000.0
        order = 2.0
        localization = sum(
            min(
                float(np.linalg.norm(truth_positions[left] - estimate_positions[right])),
                cutoff,
            ) ** order
            for left, right in pairs
        )
        cardinality = cutoff**order * abs(len(truths) - len(estimates))
        ospa = (
            (localization + cardinality) / max(len(truths), len(estimates), 1)
        ) ** (1.0 / order)
        gospa_power = (
            localization
            + cutoff**order * abs(len(truths) - len(estimates)) / 2.0
        )
        gospa = gospa_power ** (1.0 / order)
        current_assignment = {
            truths[left].target_id: estimates[right].track_id for left, right in pairs
        }
        for truth_id, track_id in current_assignment.items():
            previous = self._previous_truth_assignment.get(truth_id)
            if previous is not None and previous != track_id:
                self._id_switches += 1
        self._previous_truth_assignment = current_assignment
        return {
            "truth_target_count": len(truths),
            "estimated_target_count": len(estimates),
            "cardinality_error": len(estimates) - len(truths),
            "expected_cardinality": expected_cardinality,
            "expected_cardinality_error": expected_cardinality - truth_cardinality,
            "cardinality_log_score_nats": float(
                -math.log(max(truth_probability, 1.0e-300))
            ),
            "cardinality_ranked_probability_score": (
                _cardinality_ranked_probability_score(
                    cardinality_distribution, truth_cardinality
                )
            ),
            "cardinality_distribution": cardinality_distribution.tolist(),
            "ospa_position_m": float(ospa),
            "gospa_position_m": float(gospa),
            "gospa_power_m2": float(gospa_power),
            "gospa_cutoff_m": cutoff,
            "gospa_order": order,
            "gospa_alpha": 2.0,
            "cumulative_id_switches": int(self._id_switches),
            "truth_identity_used_only_for_offline_metrics": True,
            "truth_identity_exposed_to_controller": False,
        }

    def update(self, *args, **kwargs):
        """The old truth-generating receiver entry is intentionally unsupported."""
        raise ReturnSetContractError("LEGACY_TRUTH_CAPTURE_UPDATE_DISABLED_USE_RECEIPTS")

    def _update_action(self, *args, **kwargs):
        raise ReturnSetContractError("LEGACY_TRUTH_CAPTURE_ACTION_UPDATE_DISABLED")

    def _measurement_set(self, *args, **kwargs):
        raise ReturnSetContractError("CAPTURE_GENERATION_REMOVED_FROM_RECEIVER")

    def _append_receiver_record(self, *, epoch, status, actions, reason=None,
                                batch_rolled_back=False, receipt_ids=()):
        record = {
            "epoch": epoch,
            "status": status,
            "reason": reason,
            "association_mode": self.config.multi_target_association_mode.value,
            "track_count": len(self.tracks),
            "tentative_birth_count": len(self._tentative_births),
            "tracks": [
                {"track_id": track.track_id, "mean": track.mean.tolist(),
                 "covariance_trace": float(np.trace(track.covariance)),
                 "existence_probability": track.existence_probability,
                 "age": track.age, "missed_epochs": track.missed_epochs,
                 "origin": track.origin}
                for track in self.tracks
            ],
            "controller_preparation": {
                "status": "passed" if self._prepared_epoch == epoch else "not_prepared",
                "already_prepared": self._prepared_epoch == epoch,
                "authoritative_primary_kinematic_sync": dict(self._latest_primary_authority_sync),
            },
            "declared_birth_model_used": self._birth_model is not None,
            "forced_event_schedule_read_by_controller": False,
            "actions": actions,
            "receipt_ids": list(receipt_ids),
            "batch_controller_state_rolled_back": bool(batch_rolled_back),
            "receiver_update_kind": (
                "UNKNOWN_NO_UPDATE" if status != "passed" else
                "NO_PACKET" if not actions else "RECEIVED_RETURN_SETS"
            ),
            "capture_generation_inside_receiver": False,
            "truth_targets_inside_receiver": False,
            "metrics": None,
            "offline_metrics_status": "not_attached",
        }
        self.history.append(record)
        return record

    def update_received(self, controller_world, admitted_sets, *, epoch):
        """Apply one fully validated current-epoch batch, without capture or truth.

        An empty batch is no new observation.  A received complete empty set is
        passed to JPDA and may update existence.  They are different events.
        Epoch prediction and authoritative-primary synchronization belong to
        the kernel before this receiver entry.
        """
        if not self.enabled:
            return {"status": "disabled"}
        from .receiver_admission import validate_admitted_return_set

        actions = []
        receipt_ids = []
        payloads = []
        try:
            epoch = _strict_integer(epoch, "RECEIVER_EPOCH")
            if not isinstance(admitted_sets, (tuple, list)):
                raise ReturnSetContractError("RECEIVER_BATCH_MUST_BE_FINITE_SEQUENCE")
            if len(admitted_sets) > len(controller_world.actions):
                raise ReturnSetContractError("RECEIVER_BATCH_EXCEEDS_DECLARED_ACTION_DOMAIN")
            seen_receipts, seen_provenance, seen_actions = set(), set(), set()
            # All receipts are checked before any track or prediction mutation.
            for item in admitted_sets:
                payload = validate_admitted_return_set(
                    item, epoch=epoch, recipient_id=self.receiver_id)
                if type(payload) is not UnlabelledReturnSet:
                    raise ReturnSetContractError("ADMITTED_PAYLOAD_TYPE")
                if payload.observation_epoch != epoch:
                    raise ReturnSetContractError("UNKNOWN_LATE_OR_FUTURE_ASSOCIATION")
                receipt_id = item.receipt_id
                if not isinstance(receipt_id, str) or not receipt_id:
                    raise ReturnSetContractError("RECEIPT_ID_REQUIRED")
                if receipt_id in seen_receipts or receipt_id in self._consumed_return_set_receipts:
                    raise ReturnSetContractError("RECEIPT_ALREADY_CONSUMED")
                if payload.provenance_id in seen_provenance or payload.provenance_id in self._consumed_return_set_provenance_ids:
                    raise ReturnSetContractError("CAPTURE_PROVENANCE_ALREADY_CONSUMED")
                if payload.action_id in seen_actions:
                    raise ReturnSetContractError("DUPLICATE_ACTION_IN_RECEIVER_BATCH")
                if not any(action.action_id == payload.action_id for action in controller_world.actions):
                    raise ReturnSetContractError("RECEIVED_ACTION_NOT_IN_CONTROLLER_DOMAIN")
                seen_receipts.add(receipt_id)
                seen_provenance.add(payload.provenance_id)
                seen_actions.add(payload.action_id)
                receipt_ids.append(receipt_id)
                payloads.append(payload)
            if self._prepared_epoch != epoch:
                raise ReturnSetContractError("RECEIVER_EPOCH_NOT_PREPARED")
            if self._hypothetical_context_id is not None:
                raise ReturnSetContractError("REAL_RECEIPT_CANNOT_UPDATE_PREDICTIVE_BRANCH")
        except (ValueError, TypeError) as error:
            return self._append_receiver_record(
                epoch=epoch, status="UNKNOWN_RECEIVER_ADMISSION", actions=[],
                reason=str(error), receipt_ids=receipt_ids)

        saved = copy.deepcopy((self.tracks, self._tentative_births,
                               self._birth_promotions_by_epoch, self._next_track_id))
        failure = None
        try:
            for item, payload in zip(admitted_sets, payloads):
                action = self._update_return_set_action(controller_world, payload, epoch=epoch)
                action.update({"receipt_id": item.receipt_id,
                               "recipient_id": item.recipient_id,
                               "admission_epoch": item.admission_epoch,
                               "observation_epoch": payload.observation_epoch,
                               "return_set_provenance_id": payload.provenance_id,
                               "receiver_admission_validated": True})
                actions.append(action)
                if action.get("status") not in {"passed", "not_target_sensitive"}:
                    failure = str(action.get("status", "UNKNOWN_MISSING_UPDATE_STATUS"))
                    break
        except BaseException:
            # No partial batch survives even an unexpected exception.  Such an
            # exception is re-raised; it is not relabelled numeric success.
            (self.tracks, self._tentative_births,
             self._birth_promotions_by_epoch, self._next_track_id) = saved
            raise
        if failure is not None:
            (self.tracks, self._tentative_births,
             self._birth_promotions_by_epoch, self._next_track_id) = saved
            for action in actions:
                action["attempted_status_before_batch_rollback"] = action["status"]
                action["status"] = "ROLLED_BACK_BATCH"
                action["batch_controller_state_rolled_back"] = True
                action["risk"] = None
            return self._append_receiver_record(
                epoch=epoch, status=failure, actions=actions, reason="RECEIVER_BATCH_ROLLED_BACK",
                batch_rolled_back=True, receipt_ids=receipt_ids)
        self._consumed_return_set_receipts.update(receipt_ids)
        self._consumed_return_set_provenance_ids.update(p.provenance_id for p in payloads)
        return self._append_receiver_record(
            epoch=epoch, status="passed", actions=actions, receipt_ids=receipt_ids)

    def update_hypothetical_return_set(self, controller_world, hypothetical, *, epoch,
                                       planning_probability=None):
        """Isolated planning-branch entry; it cannot authorize a real receipt."""
        if type(hypothetical) is not HypotheticalReturnSet:
            raise ReturnSetContractError("HYPOTHETICAL_RETURN_SET_TYPE_REQUIRED")
        if self._hypothetical_context_id is None or hypothetical.predictive_context_id != self._hypothetical_context_id:
            raise ReturnSetContractError("HYPOTHETICAL_UPDATE_REQUIRES_MATCHING_PLANNING_BRANCH")
        result = self._update_return_set_action(
            controller_world, hypothetical.payload, epoch=epoch,
            planning_probability=planning_probability)
        result.update({"hypothetical_update": True,
                       "predictive_context_id": hypothetical.predictive_context_id,
                       "real_receiver_receipt": False})
        return result

    def record_receiver_failure(self, *, epoch, reason):
        """Record failed admission without a measurement or receipt consumption."""
        epoch = _strict_integer(epoch, "RECEIVER_FAILURE_EPOCH")
        if not isinstance(reason, str) or not reason:
            raise ReturnSetContractError("RECEIVER_FAILURE_REASON_REQUIRED")
        if not self.enabled:
            return {"status": "disabled"}
        return self._append_receiver_record(
            epoch=epoch, status=reason if reason.startswith("UNKNOWN") else "UNKNOWN_RECEIVER_ADMISSION",
            reason=reason, actions=[], receipt_ids=())

    def attach_offline_metrics(self, truth_targets):
        """Attach simulator-only metrics to the last recorded receiver epoch.

        This calls only the existing offline metric accumulator.  It never
        updates tracks, receipt ledgers, controller memories or decisions.
        """
        if not self.enabled:
            return {"status": "disabled"}
        if not self.history:
            raise RuntimeError("offline metrics require a completed receiver record")
        record = self.history[-1]
        if record.get("offline_metrics_status") != "not_attached" or record.get("metrics") is not None:
            raise RuntimeError("offline metrics already attached to this receiver record")
        record["metrics"] = self._offline_metrics(truth_targets)
        record["offline_metrics_status"] = "attached_simulator_only"
        return record["metrics"]


    def expected_information_gain(
        self,
        controller_world: PhysicalWorld,
        action_ids: Sequence[int],
        *,
        planning_probability: PlanningDetectionProbability | None = None,
    ) -> float:
        if not self.enabled or not action_ids:
            return 0.0
        by_id = {action.action_id: action for action in controller_world.actions}
        total = 0.0
        for track in self.finite_set_risk_tracks():
            information = np.zeros((6, 6), dtype=float)
            for action_id in action_ids:
                action = by_id[int(action_id)]
                track_world = _target_world(
                    controller_world,
                    track.mean,
                    target_id=track.track_id,
                )
                factor = build_action_factor(track_world, action, None)
                rows = _target_rows(factor)
                if rows.size == 0:
                    continue
                h = np.asarray(factor.jacobian_physical, dtype=float)[rows, :6]
                covariance = np.asarray(factor.covariance, dtype=float)[np.ix_(rows, rows)]
                detection_probability = (
                    float(factor.link_budget.detection_probability)
                    if planning_probability is None
                    else planning_probability.evaluate(
                        action, factor
                    ).planning_success_probability
                )
                retention = (
                    detection_probability
                    * factor.link_budget.correct_association_probability
                )
                information += retention * h.T @ covariance_solve(covariance, h)
            prior = checked_covariance(track.covariance, require_spd=True)
            posterior = information_covariance_update(prior, information)
            logdet_prior = covariance_logdet(prior)
            logdet_post = covariance_logdet(posterior)
            total += (track.existence_probability * 0.5 * max(0.0, float(logdet_prior - logdet_post)))
        return float(total)

    def report(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "status": "disabled",
                "association_mode": self.config.multi_target_association_mode.value,
            }
        qualified_records = [record for record in self.history if record.get("status") == "passed"]
        invalid_records = [record for record in self.history if record.get("status") != "passed"]
        # These are realized offline estimate errors, including failed-update
        # epochs. Preserve legacy unconditional aggregation: filtering UNKNOWN
        # epochs would hide exactly the failures this report must expose.
        metric_records = [record for record in self.history if isinstance(record.get("metrics"), Mapping)]
        metrics = [record["metrics"] for record in metric_records]
        ospa = [float(item["ospa_position_m"]) for item in metrics]
        gospa = [float(item["gospa_position_m"]) for item in metrics]
        cardinality = [float(item["cardinality_error"]) for item in metrics]
        expected_cardinality_errors = [
            float(item["expected_cardinality_error"]) for item in metrics
        ]
        cardinality_log_scores = [
            float(item["cardinality_log_score_nats"]) for item in metrics
        ]
        cardinality_ranked_probability_scores = [
            float(item["cardinality_ranked_probability_score"])
            for item in metrics
            if item.get("cardinality_ranked_probability_score") is not None
        ]
        normalization_errors = [
            float(action.get("association_normalization_error", 0.0))
            for record in self.history
            for action in record["actions"]
            if action.get("status") == "passed"
        ]
        preassociation_entropies = [
            float(action.get("preassociation_mean_normalized_entropy", 0.0))
            for record in self.history
            for action in record["actions"]
            if action.get("status") == "passed"
        ]
        ambiguous_track_events = sum(
            int(action.get("preassociation_ambiguous_track_count", 0))
            for record in self.history
            for action in record["actions"]
            if action.get("status") == "passed"
        )
        coalescence_controlled_track_events = sum(
            int(action.get("coalescence_controlled_track_count", 0))
            for record in self.history
            for action in record["actions"]
            if action.get("status") == "passed"
        )
        physically_normalized_clutter_actions = [
            action
            for record in self.history
            for action in record["actions"]
            if action.get("status") == "passed"
            and action.get("clutter_intensity_source")
            == "controller_model_expected_false_returns_per_gate_volume"
        ]
        updated_actions = [
            action
            for record in self.history
            for action in record["actions"]
            if action.get("status") == "passed"
        ]
        return to_builtin(
            {
                "status": (
                    str(invalid_records[0].get("status", "UNKNOWN_MISSING_STATUS")) if invalid_records
                    else "passed" if self.history else "not_evaluated"
                ),
                "association_mode": self.config.multi_target_association_mode.value,
                "epochs": len(self.history),
                "qualified_metric_epoch_count": sum(record.get("status") == "passed" for record in metric_records),
                "offline_metric_epoch_count": len(metric_records),
                "offline_metric_missing_epoch_count": len(self.history) - len(metric_records),
                "invalid_or_unknown_epoch_count": len(invalid_records),
                "successful_action_update_count": len(updated_actions),
                "metric_population": "all_recorded_epochs_with_attached_offline_metrics_including_unknown",
                "offline_metrics_complete": len(metric_records) == len(self.history),
                "receiver_delivery_support": "current_epoch_complete_return_sets_only",
                "consumed_return_set_receipt_count": len(self._consumed_return_set_receipts),
                "failed_epoch_statuses": [str(record.get("status")) for record in invalid_records],
                "mean_ospa_position_m": float(np.mean(ospa)) if ospa else None,
                "mean_gospa_position_m": (
                    float(np.mean(gospa)) if gospa else None
                ),
                "cardinality_rmse": float(np.sqrt(np.mean(np.square(cardinality))))
                if cardinality
                else None,
                "expected_cardinality_rmse": float(
                    np.sqrt(np.mean(np.square(expected_cardinality_errors)))
                )
                if expected_cardinality_errors
                else None,
                "mean_cardinality_log_score_nats": float(
                    np.mean(cardinality_log_scores)
                )
                if cardinality_log_scores
                else None,
                "mean_cardinality_ranked_probability_score": float(
                    np.mean(cardinality_ranked_probability_scores)
                )
                if cardinality_ranked_probability_scores
                else None,
                "cumulative_id_switches": int(self._id_switches),
                "maximum_association_normalization_error": max(
                    normalization_errors,
                    default=None,
                ),
                "mean_preassociation_normalized_entropy": float(
                    np.mean(preassociation_entropies)
                )
                if preassociation_entropies
                else None,
                "preassociation_ambiguous_track_events": int(
                    ambiguous_track_events
                ),
                "coalescence_controlled_track_events": int(
                    coalescence_controlled_track_events
                ),
                "physically_normalized_clutter_action_count": len(
                    physically_normalized_clutter_actions
                ),
                "all_physical_clutter_intensities_finite_positive": all(
                    math.isfinite(
                        float(action["dimensionally_normalized_clutter_intensity"])
                    )
                    and float(action["dimensionally_normalized_clutter_intensity"])
                    > 0.0
                    and math.isfinite(
                        float(action["association_gate_volume_measurement_units"])
                    )
                    and float(action["association_gate_volume_measurement_units"])
                    > 0.0
                    for action in physically_normalized_clutter_actions
                ) if physically_normalized_clutter_actions else None,
                "coalescence_MAP_mean_preserves_marginal_existence_update": all(
                    action.get(
                        "coalescence_MAP_mean_preserves_marginal_existence_update"
                    )
                    is True
                    for action in updated_actions
                ) if updated_actions else None,
                "truth_identity_exposed_to_controller": False,
                "primary_kinematic_authority": (
                    "authoritative_dynamic_factor_graph_controller_projection"
                ),
                "latest_primary_authority_sync": dict(
                    self._latest_primary_authority_sync
                ),
                "primary_authority_sync_history": list(
                    self._primary_authority_sync_history
                ),
                "prediction_applied_before_decision": all(
                    bool(record["controller_preparation"].get("already_prepared", False))
                    for record in self.history
                ) if self.history else None,
                "declared_birth_model": (
                    None
                    if self._birth_model is None
                    else {
                        "probability_per_epoch": self._birth_model.probability_per_epoch,
                        "death_probability_per_epoch": (
                            self._birth_model.death_probability_per_epoch
                        ),
                        "offset_mean_m": self._birth_model.offset_mean_m,
                        "offset_std_m": self._birth_model.offset_std_m,
                        "velocity_std_m_s": self._birth_model.velocity_std_m_s,
                        "maximum_targets": self._birth_model.maximum_targets,
                        "event_schedule_consumed": False,
                    }
                ),
                "final_tentative_birth_components": [
                    {
                        "existence_probability": item.existence_probability,
                        "distinct_sources": len(item.sources),
                        "last_epoch": item.last_epoch,
                    }
                    for item in self._tentative_births
                ],
                "history": self.history,
            }
        )
