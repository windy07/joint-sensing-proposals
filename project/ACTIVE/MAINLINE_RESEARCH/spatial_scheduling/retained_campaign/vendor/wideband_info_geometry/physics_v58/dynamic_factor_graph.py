from __future__ import annotations

"""Dynamic linearized factor/hypergraph with asynchronous provenance-aware fusion.

This dependency-light fallback is intentionally explicit.  It keeps the
communication graph, observation hypergraph and statistical provenance graph
separate; supports delayed/out-of-sequence factors within a fixed lag; uses
IRLS robust factors and soft-association weights; conservatively shares weight
among known-correlated descendants; rejects duplicate provenance; and exposes
clock/phase gauge diagnostics.  Small cases are solved by exact batch Gaussian
elimination and checked against incremental reassembly.  A production GTSAM or
other sparse backend can consume the same typed factor records without changing
radar likelihood semantics.
"""

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence
import weakref

import numpy as np

from wideband_info_geometry.small_symmetric_linalg import (
    platform_cholesky,
    platform_matmul,
    platform_norm,
    platform_singular_values,
    platform_solve,
    platform_svd,
    platform_symmetric_eigh,
    platform_symmetric_eigvalsh,
    platform_symmetric_pinv,
)

from .distributed_fusion import covariance_intersection
from .measurement import ActionFactor, StateLayout
from .schema import PhysicalWorld, RadarAction
from .gaussian_history import GaussianHistory, compose_cut, condition_history


# Issuance is process-local, not part of the published Gaussian graph.  Keeping
# it outside graph.__dict__ makes staging a read-only operation on graph state.
# These registries are accidental-misuse guards, not a Python security sandbox.
_ISSUED_SOLUTIONS = weakref.WeakKeyDictionary()
_STAGED_MARGINALIZATIONS = weakref.WeakKeyDictionary()


def _frozen_digest(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, Enum):
            return {"enum": type(item).__qualname__, "value": encode(item.value)}
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object arrays cannot bind a graph model")
            return {"dtype": item.dtype.str, "shape": list(item.shape),
                    "bytes": hashlib.sha256(item.tobytes(order="C")).hexdigest()}
        if isinstance(item, np.generic):
            return encode(item.item())
        if is_dataclass(item):
            return {"type": type(item).__qualname__,
                    "fields": {f.name: encode(getattr(item, f.name)) for f in fields(item)}}
        if isinstance(item, Mapping):
            return sorted([[encode(k), encode(v)] for k, v in item.items()],
                          key=lambda pair: json.dumps(pair[0], sort_keys=True))
        if isinstance(item, (tuple, list)):
            return [encode(v) for v in item]
        if isinstance(item, (set, frozenset)):
            return sorted((encode(v) for v in item), key=lambda v: json.dumps(v, sort_keys=True))
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite": repr(item)}
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError("unsupported graph digest value: " + type(item).__qualname__)
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _immutable_float_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    # A bytes backing store cannot be made writable with setflags(write=True).
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _spectral_threshold(eigenvalues: np.ndarray, relative: float) -> float:
    """Scale-relative eigentolerance for information/covariance matrices."""

    values = np.asarray(eigenvalues, dtype=float)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return max(scale * float(relative), np.finfo(float).tiny)


def _relative_residual(numerator: np.ndarray, reference: np.ndarray) -> float:
    return float(
        platform_norm(np.asarray(numerator, dtype=float))
        / max(
            platform_norm(np.asarray(reference, dtype=float)),
            np.finfo(float).tiny,
        )
    )


def _equilibrated_spd_solve(
    matrix: np.ndarray,
    right: np.ndarray,
    *,
    refinement_steps: int = 3,
) -> np.ndarray:
    """Solve an SPD system after symmetric diagonal equilibration.

    The joint radar information matrix mixes target kinematics with clock and
    delay nuisance coordinates.  It can therefore be positive definite while
    having a large unit-induced condition number.  Equilibration preserves the
    quadratic exactly and iterative refinement limits numerical drift when a
    fixed-lag Schur message is formed repeatedly.
    """

    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    rhs = np.asarray(right, dtype=float)
    diagonal = np.diag(symmetric)
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise np.linalg.LinAlgError("SPD equilibration requires a positive diagonal")
    scale = np.sqrt(diagonal)
    equilibrated = symmetric / scale[:, None] / scale[None, :]
    cholesky = platform_cholesky(equilibrated)

    def solve_once(value: np.ndarray) -> np.ndarray:
        scaled_value = (
            value / scale
            if value.ndim == 1
            else value / scale[:, None]
        )
        solved = platform_solve(
            cholesky.T,
            platform_solve(cholesky, scaled_value),
        )
        return solved / scale if value.ndim == 1 else solved / scale[:, None]

    solution = solve_once(rhs)
    for _ in range(max(0, int(refinement_steps))):
        residual = rhs - platform_matmul(symmetric, solution)
        correction = solve_once(residual)
        candidate = solution + correction
        if not np.all(np.isfinite(candidate)):
            break
        solution = candidate
    return solution


def _equilibrated_spd_inverse(matrix: np.ndarray) -> np.ndarray:
    r"""Return an SPD inverse as an explicit covariance-root product.

    With ``A = D L L.T D`` after symmetric diagonal equilibration,

    ``A^-1 = (D^-1 L^-T) (D^-1 L^-T).T``.

    Forming the two triangular solves column by column and then symmetrizing can
    leave negative eigenvalues when the radar graph contains weak clock or delay
    modes.  The root product below represents the same inverse but is positive
    semidefinite by construction.  This is important for later Bayesian
    covariance contractions, where a spurious negative mode can turn a
    nonnegative information gain into a platform-dependent failure.
    """

    symmetric = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    diagonal = np.diag(symmetric)
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise np.linalg.LinAlgError("SPD equilibration requires a positive diagonal")
    scale = np.sqrt(diagonal)
    equilibrated = symmetric / scale[:, None] / scale[None, :]
    cholesky = platform_cholesky(equilibrated)
    inverse_lower = platform_solve(
        cholesky,
        np.eye(cholesky.shape[0], dtype=float),
    )
    covariance_root = inverse_lower.T / scale[:, None]
    inverse = platform_matmul(covariance_root, covariance_root.T)
    if not np.all(np.isfinite(inverse)):
        raise np.linalg.LinAlgError("equilibrated SPD inverse is non-finite")
    return 0.5 * (inverse + inverse.T)


class VariableKind(str, Enum):
    TARGET = "target_state"
    NUISANCE = "synchronization_calibration_nuisance"
    ADVERSARY = "adversary_belief_logit"
    RESOURCE = "campaign_resource"
    TOPOLOGY = "topology_health"
    ASSOCIATION = "association_latent"


class FactorKind(str, Enum):
    PRIOR = "prior"
    TRANSITION = "transition"
    MEASUREMENT = "measurement"
    ASSOCIATION = "association"
    SYNCHRONIZATION = "synchronization"
    CALIBRATION = "calibration"
    TOPOLOGY = "topology"
    ADVERSARY = "adversary"
    RESOURCE = "resource"


class RobustKernel(str, Enum):
    NONE = "none"
    HUBER = "huber"
    CAUCHY = "cauchy"


@dataclass(frozen=True, order=True)
class VariableKey:
    kind: VariableKind
    entity_id: str
    epoch: int

    def token(self) -> str:
        return f"{self.kind.value}:{self.entity_id}:{self.epoch}"


@dataclass(frozen=True)
class DynamicVariable:
    key: VariableKey
    dimension: int
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    units: tuple[str, ...]
    gauge_group: str | None = None
    anchored: bool = False
    prior_enabled: bool = True

    def __post_init__(self) -> None:
        mean = np.asarray(self.prior_mean, dtype=float).reshape(-1)
        covariance = np.asarray(self.prior_covariance, dtype=float)
        if self.dimension <= 0 or mean.size != self.dimension or covariance.shape != (self.dimension, self.dimension):
            raise ValueError("dynamic variable dimensions disagree")
        if len(self.units) != self.dimension:
            raise ValueError("dynamic variable units length disagrees")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("dynamic variable prior must be finite")
        if float(platform_symmetric_eigvalsh(0.5 * (covariance + covariance.T)).min()) < -1e-10:
            raise ValueError("dynamic variable prior covariance must be PSD")
        if not isinstance(self.prior_enabled, bool):
            raise ValueError("dynamic variable prior_enabled flag must be boolean")
        object.__setattr__(self, "prior_mean", mean)
        object.__setattr__(self, "prior_covariance", 0.5 * (covariance + covariance.T))


@dataclass(frozen=True)
class DynamicFactor:
    factor_id: str
    kind: FactorKind
    observation_epoch: int
    arrival_epoch: int
    variables: tuple[VariableKey, ...]
    jacobians: tuple[np.ndarray, ...]
    measurement: np.ndarray
    covariance: np.ndarray
    participants: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    correlation_group: str | None = None
    robust_kernel: RobustKernel = RobustKernel.NONE
    robust_scale: float = 2.5
    association_probabilities: tuple[float, ...] = (1.0,)
    missing: bool = False
    fidelity_level: str = "R0_analytic_factor"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        measurement = np.asarray(self.measurement, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        jacobians = tuple(np.asarray(item, dtype=float) for item in self.jacobians)
        if not self.factor_id or self.observation_epoch < 0 or self.arrival_epoch < self.observation_epoch:
            raise ValueError("factor identity/epochs are invalid")
        if not self.variables or len(self.variables) != len(jacobians):
            raise ValueError("factor variables and Jacobians disagree")
        if covariance.shape != (measurement.size, measurement.size):
            raise ValueError("factor covariance dimension disagrees")
        if any(jacobian.shape[0] != measurement.size for jacobian in jacobians):
            raise ValueError("factor Jacobian row dimension disagrees")
        if not self.provenance_ids:
            raise ValueError("every factor requires provenance")
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("factor covariance must be finite")
        if float(platform_symmetric_eigvalsh(covariance).min()) <= 0.0:
            raise ValueError("factor covariance must be positive definite")
        probabilities = np.asarray(self.association_probabilities, dtype=float)
        if np.any(probabilities < 0.0) or not np.isfinite(probabilities).all() or float(probabilities.sum()) <= 0.0:
            raise ValueError("association probabilities must be finite and nonnegative")
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "jacobians", jacobians)
        object.__setattr__(self, "association_probabilities", tuple(float(value) for value in probabilities / probabilities.sum()))


@dataclass(frozen=True)
class FusionRejection:
    factor_id: str
    reason: str


@dataclass(frozen=True)
class GraphSolution:
    epoch: int
    estimates: Mapping[str, np.ndarray]
    covariances: Mapping[str, np.ndarray]
    joint_information: np.ndarray
    joint_covariance: np.ndarray
    variable_slices: Mapping[str, tuple[int, int]]
    accepted_factor_ids: tuple[str, ...]
    rejected_factors: tuple[FusionRejection, ...]
    robust_weights: Mapping[str, float]
    provenance_weight_sums: Mapping[str, float]
    gauge_diagnostics: Mapping[str, Any]
    atlas_diagnostics: Mapping[str, Any]
    definable_rank_strata: Mapping[str, Any]
    interaction_cumulant_diagnostics: Mapping[str, Any]
    exact_batch_incremental_residual: float
    status: str


@dataclass(frozen=True)
class FixedLagMarginalization:
    eliminated_epoch: int
    boundary_keys: tuple[VariableKey, ...]
    information: np.ndarray
    information_vector: np.ndarray
    absorbed_factor_ids: tuple[str, ...]
    expired_unfused_factor_ids: tuple[str, ...]
    status: str
    #: Factors that touched the eliminated epoch and had *not yet arrived* when
    #: it was summarized.  Their evidence is irrecoverably lost: the variables
    #: they constrain no longer exist.  This is a strict subset of
    #: ``expired_unfused_factor_ids``; the remainder were legitimately rejected
    #: (duplicate provenance, cross-partition unavailability, ...).
    expired_undelivered_factor_ids: tuple[str, ...] = ()
    absorbed_provenance_roots: tuple[str, ...] = ()
    absorbed_correlation_groups: tuple[str, ...] = ()
    combined_weights: tuple[tuple[str, float], ...] = ()
    eliminated_keys: tuple[VariableKey, ...] = ()
    removed_factor_ids: tuple[str, ...] = ()
    arrival_epoch: int = 0
    source_model_digest: str = ""
    source_solution_digest: str = ""
    stage_token: str = ""
    inherited_anchor_keys: tuple[VariableKey, ...] = ()
    retained_history: GaussianHistory | None = None

    @property
    def separator_provenance_id(self) -> str:
        return "marginal_evidence:" + self.stage_token

    @property
    def separator_metadata(self) -> dict[str, Any]:
        return {"marginalization_stage_token": self.stage_token,
                "provenance_roots": self.absorbed_provenance_roots,
                "absorbed_correlation_groups": self.absorbed_correlation_groups,
                "absorbed_factor_ids": self.absorbed_factor_ids,
                "frozen_combined_weights": self.combined_weights,
                "source_model_digest": self.source_model_digest,
                "source_solution_digest": self.source_solution_digest,
                "fixed_lag_exact_for_frozen_linearization": True,
                "global_nonlinear_remarginalization_claimed": False,
                "retained_history_digest":_frozen_digest(self.retained_history),
                "retained_history_fidelity":"proper frozen joint Gaussian conditional only",
                "marginalization_generated_epoch": self.arrival_epoch}


@dataclass(frozen=True)
class CommunicationSnapshot:
    epoch: int
    node_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def components(self) -> tuple[tuple[str, ...], ...]:
        adjacency = {node: set() for node in self.node_ids}
        for left, right in self.edges:
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining = set(self.node_ids)
        components: list[tuple[str, ...]] = []
        while remaining:
            root = min(remaining)
            stack = [root]
            component: set[str] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(adjacency[node].difference(component))
            remaining.difference_update(component)
            components.append(tuple(sorted(component)))
        return tuple(sorted(components))

    def participants_connected(self, participants: Sequence[str]) -> bool:
        requested = set(participants)
        if len(requested) <= 1:
            return True
        return any(requested.issubset(set(component)) for component in self.components())


class DynamicFactorGraph:
    """Fixed-lag dynamic graph with exact small-world Gaussian control."""

    def __init__(self, *, fixed_lag_steps: int = 6, correlation_weight_policy: str = "equal_share",
                 retained_history_dimension_cap: int = 2048, historical_acquisition_cap: int = 4096,
                 retained_history_array_bytes_cap: int = 64 * 1024**2) -> None:
        if fixed_lag_steps <= 0:
            raise ValueError("fixed_lag_steps must be positive")
        if correlation_weight_policy not in {"equal_share", "reject_secondary"}:
            raise ValueError("unsupported correlation weight policy")
        self.fixed_lag_steps = int(fixed_lag_steps)
        self.correlation_weight_policy = correlation_weight_policy
        self.variables: dict[VariableKey, DynamicVariable] = {}
        self.factors: list[DynamicFactor] = []
        self.communication: dict[int, CommunicationSnapshot] = {}
        self.observation_hypergraph: dict[int, set[tuple[str, ...]]] = {}
        self.provenance_graph: dict[str, set[str]] = {}
        self._factor_ids: set[str] = set()
        self._represented_root_owners: dict[str, str] = {}
        self._represented_group_owners: dict[str, str] = {}
        self._owned_separator_digests: dict[str, str] = {}
        if (type(retained_history_dimension_cap) is not int or retained_history_dimension_cap <= 0
                or type(historical_acquisition_cap) is not int or historical_acquisition_cap <= 0
                or type(retained_history_array_bytes_cap) is not int or retained_history_array_bytes_cap <= 0):
            raise ValueError("retained-history capacities must be positive explicit integers")
        self.retained_history_dimension_cap = retained_history_dimension_cap
        self.historical_acquisition_cap = historical_acquisition_cap
        self.retained_history_array_bytes_cap = retained_history_array_bytes_cap
        self._retained_history: GaussianHistory | None = None
        self._historical_acquisitions: dict[str, Mapping[str, Any]] = {}
        self._joint_state = None
        self._joint_budget = None

    def _model_digest(self) -> str:
        base = _frozen_digest((self.fixed_lag_steps, self.correlation_weight_policy,
            self.variables, self.factors, self.communication, self.observation_hypergraph,
            self.provenance_graph, self._factor_ids, self._represented_root_owners,
            self._represented_group_owners, self._owned_separator_digests,
            self.retained_history_dimension_cap, self.historical_acquisition_cap,
            self.retained_history_array_bytes_cap,
            self._retained_history, self._historical_acquisitions))
        if self._joint_state is None:
            return base
        from .joint_association import state_digest
        return _frozen_digest((base, state_digest(self._joint_state)))

    def _require_gaussian_mode(self):
        if self._joint_state is not None:
            raise ValueError("Joint return-set posterior requires the explicit joint API; Gaussian projection is not authority")

    def enable_joint_returns(self, **kwargs):
        from .joint_association import enable
        return enable(self, **kwargs)

    def prepare_joint_batch(self, batch):
        from .joint_association import prepare_batch
        return prepare_batch(self, batch)

    def validate_joint_stage(self, stage, batch):
        from .joint_association import validate_stage
        return validate_stage(self, stage, batch)

    def joint_predict(self, **kwargs):
        from .joint_association import predict
        return predict(self, **kwargs)

    def joint_cut(self, **kwargs):
        from .joint_association import cut
        return cut(self, **kwargs)

    def joint_snapshot(self, **kwargs):
        from .joint_association import snapshot
        return snapshot(self, **kwargs)

    def _check_history_capacity(self, history_dimension: int, boundary_dimension: int) -> None:
        # Bound stored float64 m/T/C, including growth of the current boundary.
        # This is not a peak-RAM or descriptor/provenance-storage certificate.
        payload = 8 * history_dimension * (1 + boundary_dimension + history_dimension)
        if (history_dimension > self.retained_history_dimension_cap
                or payload > self.retained_history_array_bytes_cap):
            raise ValueError("retained history numeric storage capacity exceeded before allocation")

    def retained_history_snapshot(self) -> GaussianHistory | None:
        """A detached conditional kernel; it contains no current marginal.

        The numerical arrays are immutable, but copying also detaches the
        descriptive variable specifications from arbitrary caller edits.
        """
        self._require_gaussian_mode()
        return copy.deepcopy(self._retained_history)

    def add_historical_factor(self, factor: DynamicFactor, *, current_epoch: int) -> Mapping[str, Any]:
        """Atomically assimilate one arrived frozen Gaussian historical channel.

        The induced factor contains only NEW acquisition evidence. Its old
        conditioning dependencies are recorded separately, not multiplied as
        additional provenance roots. This does not implement mixture JPDA or
        infer an unprovided correlation/receipt likelihood.
        """
        self._require_gaussian_mode()
        if (type(current_epoch) is not int or current_epoch < 0
                or not self.variables or current_epoch != max(k.epoch for k in self.variables)):
            raise ValueError("historical assimilation epoch must equal the current graph epoch")
        if self._retained_history is None:
            raise ValueError("no retained historical conditional kernel")
        if factor.arrival_epoch > current_epoch or factor.missing:
            raise ValueError("historical payload has not arrived or is marked missing")
        if factor.kind is not FactorKind.MEASUREMENT:
            raise ValueError("historical admission is for a newly acquired measurement channel")
        if factor.robust_kernel is not RobustKernel.NONE or factor.association_probabilities != (1.0,):
            raise ValueError("historical channel needs an explicit joint Gaussian likelihood, not projected association/robust weights")
        if factor.correlation_group is not None:
            raise ValueError("correlated historical channel requires an explicit joint noise model")
        if factor.metadata.get("conditional_noise_independent_given_scope") is not True:
            raise ValueError("historical likelihood must declare its conditional noise model")
        if len(self._historical_acquisitions) >= self.historical_acquisition_cap:
            raise ValueError("historical acquisition capacity exceeded before publication")
        if factor.factor_id in self._historical_acquisitions:
            raise ValueError("historical acquisition already consumed")
        if factor.factor_id in self._factor_ids:
            raise ValueError("historical acquisition identity conflicts with an existing graph factor")
        self.validate_owned_separators()
        reason = self.evidence_admission_reason(factor)
        if reason is not None:
            raise ValueError("historical evidence rejected: " + reason)
        new_roots = self._fusion_provenance_lineages(factor)
        existing_roots = set().union(*(self._fusion_provenance_lineages(f) for f in self.factors))
        if new_roots.intersection(existing_roots):
            raise ValueError("historical acquisition reuses already represented evidence")
        if factor.participants:
            snapshot_epoch = factor.metadata.get("communication_snapshot_epoch")
            if (type(snapshot_epoch) is not int or snapshot_epoch != factor.observation_epoch
                    or snapshot_epoch not in self.communication):
                raise ValueError("historical channel communication context is unavailable")
            snapshot = self.communication[snapshot_epoch]
            if (not set(factor.participants).issubset(snapshot.node_ids)
                    or not snapshot.participants_connected(factor.participants)):
                raise ValueError("historical channel participants were not connected")
        for variable in self._retained_history.boundary_variables:
            live = self.variables.get(variable.key)
            if live is None or live.dimension != variable.dimension or live.units != variable.units:
                raise ValueError("retained history boundary schema no longer matches the graph")
        original_digest = self._model_digest()
        history_keys = {v.key for v in self._retained_history.history_variables}
        expanded_boundary = {v.key for v in self._retained_history.boundary_variables}.union(
            set(factor.variables).difference(history_keys))
        self._check_history_capacity(self._retained_history.dimension,
            sum(self.variables[key].dimension for key in expanded_boundary))
        conditioned = condition_history(self._retained_history, factor, self.variables)
        kernel = conditioned.history
        keys = tuple(v.key for v in kernel.boundary_variables)
        if not set(keys).issubset({v.key for v in self._active_variables(current_epoch)}):
            raise ValueError("induced boundary is outside the active graph window")
        jacobians = []
        offset = 0
        for variable in kernel.boundary_variables:
            jacobians.append(conditioned.effective_jacobian[:,offset:offset+variable.dimension])
            offset += variable.dimension
        induced_id = "historical_induced:" + factor.factor_id
        if induced_id in self._factor_ids:
            raise ValueError("historical induced factor identity already exists")
        # A zero effective H is still retained: it can update the historical
        # kernel while adding no information to the current boundary marginal.
        induced = DynamicFactor(
            factor_id=induced_id, kind=FactorKind.MEASUREMENT,
            observation_epoch=min(k.epoch for k in keys), arrival_epoch=current_epoch,
            variables=keys, jacobians=tuple(jacobians),
            measurement=conditioned.effective_measurement,
            covariance=conditioned.effective_covariance, participants=(),
            provenance_ids=factor.provenance_ids, correlation_group=None,
            robust_kernel=RobustKernel.NONE, association_probabilities=(1.0,), missing=False,
            fidelity_level="frozen_Gaussian_historical_conditional_likelihood",
            metadata={"provenance_roots":tuple(sorted(new_roots)),
                "historical_original_factor_id":factor.factor_id,
                "historical_original_factor_digest":_frozen_digest(factor),
                "original_observation_epoch":factor.observation_epoch,
                "original_arrival_epoch":factor.arrival_epoch,
                "conditioning_history_digest":_frozen_digest(self._retained_history),
                "conditioning_acquisition_ids":self._retained_history.conditioning_acquisition_ids,
                "conditioning_dependency_ids":self._retained_history.conditioning_dependency_ids,
                "conditioning_dependencies_are_not_new_evidence":True,
                "historical_noise_model":"independent conditional on declared joint scope",
                "original_participants":factor.participants})
        receipt = {"factor_id":factor.factor_id,"induced_factor_id":induced_id,
                   "capture_epoch":factor.observation_epoch,"arrival_epoch":factor.arrival_epoch,
                   "admitted_epoch":current_epoch,"added_evidence_ids":tuple(sorted(new_roots)),
                   "before_history_digest":_frozen_digest(self._retained_history),
                   "after_history_digest":_frozen_digest(kernel),
                   "history_dimension":kernel.dimension,
                   "history_array_payload_bytes":kernel.array_payload_bytes,
                   "scope":"same frozen proper joint Gaussian law; no full association/physical Bayes claim"}
        new_factors = [*self.factors,induced]
        ids = {*self._factor_ids,induced_id}
        provenance = {k:set(v) for k,v in self.provenance_graph.items()}
        for identity in induced.provenance_ids:
            provenance.setdefault(identity,set()).add(induced_id)
        acquisitions = {**self._historical_acquisitions,factor.factor_id:receipt}
        # The updated conditional and this induced likelihood form one joint
        # checkpoint. Protect its likelihood before greedy factor ordering:
        # a later-added earlier-sorted descendant must not replace it while
        # leaving the historical conditional conditioned on the original y.
        root_owners = {**self._represented_root_owners,
                       **{root:induced_id for root in new_roots}}
        owned_digests = {**self._owned_separator_digests,
                         induced_id:_frozen_digest(induced)}
        if self._model_digest() != original_digest:
            raise RuntimeError("graph changed while staging historical assimilation")
        self.factors, self._factor_ids, self.provenance_graph = new_factors, ids, provenance
        self._retained_history, self._historical_acquisitions = kernel, acquisitions
        self._represented_root_owners = root_owners
        self._owned_separator_digests = owned_digests
        _STAGED_MARGINALIZATIONS.pop(self,None)
        _ISSUED_SOLUTIONS.pop(self,None)
        return dict(receipt)

    def solution_is_current(self, solution: GraphSolution | None) -> bool:
        """Whether this exact graph issued the unchanged current solution.

        A deep-copied graph must solve on its own before staging elimination;
        copying a numerical solution does not copy authority to publish a
        separator. Health is checked separately by the staging contract.
        """
        issued = _ISSUED_SOLUTIONS.get(self)
        return bool(issued is not None and issued[0] is solution
                    and issued[1] == self._model_digest()
                    and issued[2] == _frozen_digest(solution))

    def fork(self) -> "DynamicFactorGraph":
        """Return an isolated graph branch for counterfactual evaluation.

        The runtime uses branches only for bounded what-if checks.  A branch
        never mutates the authoritative online belief and preserves factor,
        communication, hyperedge, and provenance semantics exactly.
        """

        if self._joint_state is not None:
            self._joint_budget.charge('posterior_forks')
            return copy.deepcopy(self, {id(self._joint_budget): self._joint_budget})
        return copy.deepcopy(self)

    def add_communication_snapshot(self, snapshot: CommunicationSnapshot) -> None:
        self._require_gaussian_mode()
        if snapshot.epoch in self.communication:
            raise ValueError(f"communication snapshot already exists for epoch {snapshot.epoch}")
        self.communication[snapshot.epoch] = snapshot

    def add_observation_hyperedge(self, epoch: int, participants: Iterable[str]) -> None:
        self._require_gaussian_mode()
        edge = tuple(sorted(set(str(value) for value in participants)))
        if not edge:
            raise ValueError("observation hyperedge cannot be empty")
        self.observation_hypergraph.setdefault(int(epoch), set()).add(edge)

    def add_variable(self, variable: DynamicVariable) -> None:
        self._require_gaussian_mode()
        if (self._retained_history is not None and variable.key in
                {v.key for v in self._retained_history.history_variables}):
            raise ValueError("retained historical coordinates cannot be reintroduced as live variables")
        if variable.key in self.variables:
            existing = self.variables[variable.key]
            if existing.dimension != variable.dimension:
                raise ValueError("variable redefinition changed dimension")
            return
        self.variables[variable.key] = variable

    def add_factor(self, factor: DynamicFactor) -> None:
        self._require_gaussian_mode()
        if "marginalization_stage_token" in factor.metadata:
            raise ValueError("a staged separator must use commit_marginalization")
        if factor.factor_id in self._factor_ids:
            raise ValueError(f"factor identifier already exists: {factor.factor_id}")
        for key, jacobian in zip(factor.variables, factor.jacobians):
            if key not in self.variables:
                raise KeyError(f"factor {factor.factor_id} references unknown variable {key.token()}")
            if jacobian.shape[1] != self.variables[key].dimension:
                raise ValueError(f"factor {factor.factor_id} Jacobian dimension disagrees with {key.token()}")
        self.factors.append(factor)
        self._factor_ids.add(factor.factor_id)
        for provenance in factor.provenance_ids:
            self.provenance_graph.setdefault(provenance, set()).add(factor.factor_id)
        if factor.participants:
            self.add_observation_hyperedge(factor.observation_epoch, factor.participants)

    def replace_factor(self, factor: DynamicFactor) -> None:
        """Atomically replace one frozen local likelihood linearization."""

        self._require_gaussian_mode()
        if factor.factor_id not in self._factor_ids:
            raise KeyError(f"factor identifier does not exist: {factor.factor_id}")
        if factor.factor_id in self._owned_separator_digests:
            raise ValueError("an owned separator can change only through a new marginalization")
        index = next(
            position
            for position, existing in enumerate(self.factors)
            if existing.factor_id == factor.factor_id
        )
        previous = self.factors[index]
        if (
            factor.variables != previous.variables
            or factor.observation_epoch != previous.observation_epoch
            or factor.arrival_epoch != previous.arrival_epoch
            or factor.participants != previous.participants
            or factor.provenance_ids != previous.provenance_ids
            or self._fusion_provenance_lineages(factor) != self._fusion_provenance_lineages(previous)
            or factor.correlation_group != previous.correlation_group
            or "marginalization_stage_token" in factor.metadata
        ):
            raise ValueError(
                "factor relinearization changed identity, topology, timing or provenance"
            )
        for key, jacobian in zip(factor.variables, factor.jacobians):
            if key not in self.variables or jacobian.shape[1] != self.variables[key].dimension:
                raise ValueError("replacement factor variable dimension is invalid")
        self.factors[index] = factor

    def factor(self, factor_id: str) -> DynamicFactor:
        """Return the current immutable factor registered under ``factor_id``."""

        if factor_id not in self._factor_ids:
            raise KeyError(f"factor identifier does not exist: {factor_id}")
        return next(item for item in self.factors if item.factor_id == factor_id)

    def marginalize_epoch_to_boundary(
        self,
        *,
        eliminate_epoch: int,
        boundary_keys: Sequence[VariableKey],
        solved_epoch: int,
        solution: GraphSolution,
        arrival_epoch: int | None = None,
    ) -> FixedLagMarginalization:
        """Stage a generalized separator without changing the published graph.

        This API formerly deleted old factors immediately. Callers must now build
        the canonical factor and call commit_marginalization. The dependence
        closure preserves the same accepted, frozen weighted quadratic; it does
        not upgrade correlation heuristics or linearizations to exact Bayes.
        """
        self._require_gaussian_mode()
        for value in (eliminate_epoch, solved_epoch):
            if type(value) is not int or value < 0:
                raise ValueError("marginalization epochs must be nonnegative integers")
        arrival = solved_epoch if arrival_epoch is None else arrival_epoch
        if type(arrival) is not int or arrival < solved_epoch:
            raise ValueError("separator arrival cannot precede its source solve")
        issued = _ISSUED_SOLUTIONS.get(self)
        source_digest = self._model_digest()
        if (not self.solution_is_current(solution)
                or solution.epoch != solved_epoch or solution.status != "passed"):
            raise ValueError("marginalization requires the unchanged latest graph-issued healthy solution")
        epoch = eliminate_epoch
        requested = tuple(boundary_keys)
        if (not requested or len(set(requested)) != len(requested)
                or any(key not in self.variables or key.epoch > solved_epoch for key in requested)):
            raise ValueError("boundary must have unique existing nonfuture variables")
        old_keys = tuple(sorted((key for key in self.variables if key.epoch == epoch), key=lambda k:k.token()))
        old_set = set(old_keys)
        if sum(self.variables[key].dimension for key in old_keys) + (
                0 if self._retained_history is None else self._retained_history.dimension
                ) > self.retained_history_dimension_cap:
            raise ValueError("retained history capacity exceeded before Schur assembly")
        if old_set.intersection(requested):
            raise ValueError("boundary must not contain eliminated variables")
        if not old_keys:
            dim = sum(self.variables[key].dimension for key in requested)
            return FixedLagMarginalization(epoch, requested, np.zeros((dim,dim)),
                np.zeros(dim), (), (), "no_expired_variables")
        accepted_ids = set(solution.accepted_factor_ids)
        accepted = [factor for factor in self.factors if factor.factor_id in accepted_ids]
        absorbed_ids = {f.factor_id for f in accepted if old_set.intersection(f.variables)}
        def groups(factor):
            result = {factor.correlation_group} if factor.correlation_group else set()
            if factor.factor_id in self._owned_separator_digests:
                result.update(factor.metadata.get("absorbed_correlation_groups", ()))
            return result
        # Correlated accepted factors across the cut are absorbed together, with
        # their whole non-old scope retained. Iterate to a fixed point.
        while True:
            absorbed = [f for f in accepted if f.factor_id in absorbed_ids]
            roots = set().union(*(self._fusion_provenance_lineages(f) for f in absorbed))
            group_roots = set().union(*(groups(f) for f in absorbed))
            extension = {f.factor_id for f in accepted if
                roots.intersection(self._fusion_provenance_lineages(f)) or group_roots.intersection(groups(f))}
            if extension.issubset(absorbed_ids):
                break
            absorbed_ids.update(extension)
        boundary = tuple(sorted(set(requested).union(
            *(set(f.variables).difference(old_set) for f in absorbed)), key=lambda k:k.token()))
        history_boundary = set(boundary).union(() if self._retained_history is None else
            {v.key for v in self._retained_history.boundary_variables}.difference(old_set))
        self._check_history_capacity(
            sum(self.variables[key].dimension for key in old_keys) +
                (0 if self._retained_history is None else self._retained_history.dimension),
            sum(self.variables[key].dimension for key in history_boundary))
        local_variables = [self.variables[key] for key in sorted(old_set.union(boundary), key=lambda k:k.token())]
        counts = {}
        for factor in accepted:
            if factor.correlation_group:
                counts[factor.correlation_group] = counts.get(factor.correlation_group,0)+1
        combined = {}
        for factor in accepted:
            correlation = 1.0
            if factor.correlation_group:
                if self.correlation_weight_policy == "equal_share":
                    correlation = 1.0/counts[factor.correlation_group]
                else:
                    first = min(f.factor_id for f in accepted if f.correlation_group == factor.correlation_group)
                    correlation = float(factor.factor_id == first)
            weight = self._association_weight(factor)*correlation*float(solution.robust_weights.get(factor.factor_id,1.0))
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("invalid frozen combined factor weight")
            combined[factor.factor_id] = weight
        square_root, square_root_value = self._square_root_system(
            local_variables, absorbed, solution.robust_weights,
            combined_weights=combined, prior_keys=old_set)
        local_index, _ = self._index(local_variables)
        old_indices = np.asarray([i for key in old_keys for i in range(local_index[key].start,local_index[key].stop)],dtype=int)
        boundary_indices = np.asarray([i for key in boundary for i in range(local_index[key].start,local_index[key].stop)],dtype=int)
        old_rows = square_root[:, old_indices]
        boundary_rows = square_root[:, boundary_indices]
        # The separator and retained conditional must come from this SAME
        # frozen quadratic. A singular old block cannot supply the proper
        # conditional supported by this implementation; fail before publishing
        # either object instead of silently discarding historical support.
        local_information = platform_matmul(square_root.T, square_root)
        local_vector = platform_matmul(square_root.T, square_root_value)
        old_information = local_information[np.ix_(old_indices, old_indices)]
        old_boundary_information = local_information[
            np.ix_(old_indices, boundary_indices)
        ]
        boundary_old_information = local_information[
            np.ix_(boundary_indices, old_indices)
        ]
        boundary_information = local_information[
            np.ix_(boundary_indices, boundary_indices)
        ]
        old_vector = local_vector[old_indices]
        boundary_vector = local_vector[boundary_indices]
        try:
            solved = _equilibrated_spd_solve(
                old_information,
                np.column_stack((old_boundary_information, old_vector,
                                 np.eye(len(old_indices)))),
                refinement_steps=5,
            )
            boundary_dimension = len(boundary_indices)
            raw_message_information = boundary_information - platform_matmul(
                boundary_old_information,
                solved[:, :boundary_dimension],
            )
            message_information = 0.5 * (
                raw_message_information + raw_message_information.T
            )
            message_vector = boundary_vector - platform_matmul(
                boundary_old_information,
                solved[:, boundary_dimension],
            )
        except np.linalg.LinAlgError as error:
            raise ValueError("singular eliminated information needs an explicit supported conditional history") from error
        if self._retained_history is not None:
            for variable in self._retained_history.boundary_variables:
                live = self.variables.get(variable.key)
                if live is None or live.dimension != variable.dimension or live.units != variable.units:
                    raise ValueError("retained history boundary schema no longer matches the graph")
        retained_history = compose_cut(
            self._retained_history,
            tuple(self.variables[key] for key in old_keys),
            tuple(self.variables[key] for key in boundary),
            solved[:, boundary_dimension], -solved[:, :boundary_dimension],
            solved[:, boundary_dimension+1:],
            dimension_cap=self.retained_history_dimension_cap)
        message_eigenvalues, message_eigenvectors = platform_symmetric_eigh(message_information)
        message_tolerance = _spectral_threshold(message_eigenvalues, 1e-10)
        if float(message_eigenvalues.min()) < -message_tolerance:
            raise RuntimeError(
                "fixed-lag Schur message is materially indefinite"
            )
        # Clip only roundoff-scale negatives.  The reconstruction is an exact
        # no-op to the declared tolerance and avoids feeding a -0 eigenvalue to
        # later Cholesky/rank logic.
        if float(message_eigenvalues.min()) < 0.0:
            message_information = platform_matmul(
                message_eigenvectors * np.maximum(message_eigenvalues, 0.0),
                message_eigenvectors.T,
            )
            message_information = 0.5 * (
                message_information + message_information.T
            )
        message_rank_threshold = _spectral_threshold(
            message_eigenvalues,
            1e-12,
        )
        message_range_basis = message_eigenvectors[
            :, message_eigenvalues > message_rank_threshold
        ]
        message_rank = int(message_range_basis.shape[1])
        projected_message_vector = (
            platform_matmul(
                message_range_basis,
                platform_matmul(message_range_basis.T, message_vector),
            )
            if message_rank
            else np.zeros_like(message_vector)
        )
        message_range_residual = _relative_residual(
            message_vector - projected_message_vector,
            message_vector,
        )
        if message_range_residual > 1e-8:
            raise RuntimeError(
                "fixed-lag Schur information vector lies outside the message information range"
            )
        touching = [f for f in self.factors if old_set.intersection(f.variables)]
        touching_ids = {f.factor_id for f in touching}
        removed = touching_ids.union(absorbed_ids)
        expired = tuple(sorted(touching_ids.difference(absorbed_ids)))
        undelivered = tuple(sorted(f.factor_id for f in touching if
            f.factor_id not in absorbed_ids and f.arrival_epoch > solved_epoch and not f.missing))
        roots = set().union(*(self._fusion_provenance_lineages(f) for f in absorbed))
        for key in old_keys:
            variable = self.variables[key]
            if variable.prior_enabled:
                roots.add("implicit_prior:" + key.token() + ":" + _frozen_digest(variable))
        retained_history = replace(retained_history, conditioning_dependency_ids=tuple(sorted(
            set(retained_history.conditioning_dependency_ids).union(roots))))
        # A full-rank JOINT separator suffices to certify an inherited anchor.
        # Positive diagonal/principal blocks do not: relative-only constraints
        # can have positive diagonal blocks and a joint gauge null direction.
        # For singular separators preserve existing flags without inventing an
        # anchor; a more precise nullspace/anchor certificate remains separate.
        anchored_groups = {self.variables[k].gauge_group for k in old_keys
                           if self.variables[k].anchored and self.variables[k].gauge_group}
        inherit = []
        eigenvalues = platform_symmetric_eigvalsh(message_information)
        threshold = _spectral_threshold(eigenvalues, 1e-10)
        if float(eigenvalues.min()) > threshold:
            for group in sorted(anchored_groups):
                candidates = [k for k in boundary if self.variables[k].gauge_group == group]
                if candidates:
                    inherit.append(candidates[0])
        status = "passed" if (np.all(np.isfinite(message_information))
            and np.all(np.isfinite(message_vector)) and float(eigenvalues.min()) >= -threshold) else "failed_closed"
        token = _frozen_digest((source_digest, issued[2], epoch, boundary, arrival,
                               tuple(sorted(removed)), message_information, message_vector))
        result = FixedLagMarginalization(
            eliminated_epoch=epoch, boundary_keys=boundary,
            information=_immutable_float_array(message_information),
            information_vector=_immutable_float_array(message_vector),
            absorbed_factor_ids=tuple(sorted(absorbed_ids)), expired_unfused_factor_ids=expired,
            status=status, expired_undelivered_factor_ids=undelivered,
            absorbed_provenance_roots=tuple(sorted(roots)),
            absorbed_correlation_groups=tuple(sorted(group_roots)),
            combined_weights=tuple(sorted((k,combined[k]) for k in absorbed_ids)),
            eliminated_keys=old_keys, removed_factor_ids=tuple(sorted(removed)),
            arrival_epoch=arrival, source_model_digest=source_digest,
            source_solution_digest=issued[2], stage_token=token,
            inherited_anchor_keys=tuple(inherit), retained_history=retained_history)
        _STAGED_MARGINALIZATIONS[self] = (result, _frozen_digest(result), solution)
        return result

    def commit_marginalization(self, result: FixedLagMarginalization,
                               factor: DynamicFactor | None) -> None:
        """Atomically install exactly the issued canonical separator and ancestry.

        A failed validation leaves every published graph field unchanged. A zero
        information result may use factor=None; its roots remain owned tombstones.
        """
        self._require_gaussian_mode()
        staged = _STAGED_MARGINALIZATIONS.get(self)
        issued = _ISSUED_SOLUTIONS.get(self)
        if (staged is None or staged[0] is not result or staged[1] != _frozen_digest(result)
                or result.status != "passed" or self._model_digest() != result.source_model_digest
                or issued is None or issued[0] is not staged[2]
                or _frozen_digest(staged[2]) != result.source_solution_digest):
            raise ValueError("unknown, altered, consumed or stale marginalization result")
        if result.expired_undelivered_factor_ids:
            raise ValueError("undelivered evidence expired before marginalization commit")
        if set(result.eliminated_keys).intersection(result.boundary_keys):
            raise ValueError("separator boundary overlaps eliminated variables")
        roots = set(result.absorbed_provenance_roots)
        groups = set(result.absorbed_correlation_groups)
        if factor is None:
            if np.any(result.information != 0.0) or np.any(result.information_vector != 0.0):
                raise ValueError("nonzero marginal information requires a factor")
            owner_id = "zero_separator:" + result.stage_token
        else:
            if (factor.factor_id in self._factor_ids or factor.variables != result.boundary_keys
                    or factor.kind is not FactorKind.PRIOR
                    or factor.observation_epoch != min(k.epoch for k in result.boundary_keys)
                    or factor.arrival_epoch != result.arrival_epoch
                    or factor.provenance_ids != (result.separator_provenance_id,)
                    or factor.participants or factor.missing or factor.correlation_group is not None
                    or factor.robust_kernel is not RobustKernel.NONE
                    or factor.association_probabilities != (1.0,)
                    or factor.metadata.get("canonical_information_factor") is not True):
                raise ValueError("separator identity, scope, timing or weighting does not match stage")
            for key, expected in result.separator_metadata.items():
                if _frozen_digest(factor.metadata.get(key)) != _frozen_digest(expected):
                    raise ValueError("separator metadata/roots do not match stage: " + key)
            if (set(factor.metadata.get("provenance_roots",())) != roots
                    or not np.array_equal(factor.covariance,np.eye(factor.measurement.size))
                    or len(factor.jacobians) != len(result.boundary_keys)
                    or _frozen_digest(factor.metadata.get("boundary_units")) != _frozen_digest(
                        tuple(tuple(self.variables[key].units) for key in result.boundary_keys))):
                raise ValueError("separator must use canonical identity covariance and exact lineage")
            for key,jac in zip(result.boundary_keys,factor.jacobians):
                if jac.shape != (factor.measurement.size,self.variables[key].dimension):
                    raise ValueError("separator Jacobian dimension changed")
            H = np.concatenate(factor.jacobians,axis=1)
            J = platform_matmul(H.T,H)
            eta = platform_matmul(H.T,factor.measurement)
            if (not np.all(np.isfinite(H)) or not np.all(np.isfinite(factor.measurement))
                    or _relative_residual(J-result.information,result.information) > 1e-8
                    or _relative_residual(eta-result.information_vector,result.information_vector) > 1e-8):
                raise ValueError("actual canonical factor differs from staged information")
            owner_id = factor.factor_id
        removed = set(result.removed_factor_ids)
        if not removed.issubset(self._factor_ids):
            raise ValueError("marginalization removal set changed")
        new_factors = [f for f in self.factors if f.factor_id not in removed]
        if factor is not None:
            new_factors.append(factor)
        new_variables = {k:v for k,v in self.variables.items() if k not in result.eliminated_keys}
        for key in result.inherited_anchor_keys:
            new_variables[key] = replace(new_variables[key], anchored=True)
        if any(not set(f.variables).issubset(new_variables) for f in new_factors):
            raise ValueError("commit would leave dangling factor variables")
        root_owners = {r:o for r,o in self._represented_root_owners.items() if o not in removed}
        group_owners = {r:o for r,o in self._represented_group_owners.items() if o not in removed}
        if any(r in root_owners for r in roots) or any(g in group_owners for g in groups):
            raise ValueError("represented evidence ownership was not included in closure")
        root_owners.update({r:owner_id for r in roots})
        group_owners.update({g:owner_id for g in groups})
        if factor is not None:
            root_owners[result.separator_provenance_id] = owner_id
        separator_digests = {k:v for k,v in self._owned_separator_digests.items() if k not in removed}
        if factor is not None:
            # Detach caller arrays/metadata; tampering with the builder object
            # after commit cannot change the published separator.
            new_factors[-1] = copy.deepcopy(factor)
            separator_digests[owner_id] = _frozen_digest(new_factors[-1])
        provenance = {}
        for f in new_factors:
            for identity in f.provenance_ids:
                provenance.setdefault(identity,set()).add(f.factor_id)
        hypergraph = {k:set(v) for k,v in self.observation_hypergraph.items()
                      if k != result.eliminated_epoch}
        new_factor_ids = {f.factor_id for f in new_factors}
        if result.retained_history is None:
            raise ValueError("a nonempty cut requires its jointly staged conditional history")
        new_history = copy.deepcopy(result.retained_history)
        if not {v.key for v in new_history.boundary_variables}.issubset(new_variables):
            raise ValueError("committed history would have a dangling current boundary")
        # Everything that may validate/allocate/raise has completed above.
        self.variables, self.factors = new_variables, new_factors
        self._factor_ids = new_factor_ids
        self.provenance_graph, self.observation_hypergraph = provenance, hypergraph
        self._represented_root_owners, self._represented_group_owners = root_owners, group_owners
        self._owned_separator_digests = separator_digests
        self._retained_history = new_history
        _STAGED_MARGINALIZATIONS.pop(self,None)
        _ISSUED_SOLUTIONS.pop(self,None)

    def add_transition(
        self,
        previous: VariableKey,
        current: VariableKey,
        transition: np.ndarray,
        process_covariance: np.ndarray,
        *,
        transition_offset: np.ndarray | None = None,
        factor_id: str | None = None,
        provenance_id: str | None = None,
    ) -> None:
        matrix = np.asarray(transition, dtype=float)
        covariance = np.asarray(process_covariance, dtype=float)
        if previous not in self.variables or current not in self.variables:
            raise KeyError("transition variables must exist")
        if matrix.shape != (self.variables[current].dimension, self.variables[previous].dimension):
            raise ValueError("transition matrix dimension disagrees")
        dimension = self.variables[current].dimension
        offset = (
            np.zeros(dimension, dtype=float)
            if transition_offset is None
            else np.asarray(transition_offset, dtype=float).reshape(-1)
        )
        if offset.shape != (dimension,) or not np.all(np.isfinite(offset)):
            raise ValueError("transition offset dimension or values are invalid")
        self.add_factor(DynamicFactor(
            factor_id=factor_id or f"transition:{previous.token()}->{current.token()}",
            kind=FactorKind.TRANSITION,
            observation_epoch=current.epoch,
            arrival_epoch=current.epoch,
            variables=(previous, current),
            jacobians=(-matrix, np.eye(dimension)),
            measurement=offset,
            covariance=covariance,
            participants=(),
            provenance_ids=(provenance_id or f"process:{previous.token()}:{current.token()}",),
            correlation_group=None,
            robust_kernel=RobustKernel.NONE,
            fidelity_level="analytic_dynamic_transition",
            metadata={
                "transition_semantics": "x_current = F x_previous + offset + process_noise",
                "offset_is_normalized_state": True,
            },
        ))

    def _active_variables(self, current_epoch: int) -> list[DynamicVariable]:
        minimum = max(0, int(current_epoch) - self.fixed_lag_steps + 1)
        return sorted(
            [variable for variable in self.variables.values() if minimum <= variable.key.epoch <= current_epoch],
            key=lambda item: item.key,
        )

    def _factor_availability(self, factor: DynamicFactor, current_epoch: int, active_keys: set[VariableKey]) -> str | None:
        if factor.metadata.get("zero_local_mixture_information_tombstone") is True:
            return "zero_local_mixture_information_state_independent_false_peak"
        if factor.missing:
            return "measurement_marked_missing"
        if factor.arrival_epoch > current_epoch:
            return "not_arrived"
        if factor.observation_epoch < current_epoch - self.fixed_lag_steps + 1:
            return "outside_fixed_lag"
        if not set(factor.variables).issubset(active_keys):
            return "variable_outside_active_window"
        if factor.participants:
            declared_snapshot_epoch = factor.metadata.get("communication_snapshot_epoch")
            if declared_snapshot_epoch is not None:
                if not isinstance(declared_snapshot_epoch, (int, np.integer)):
                    return "invalid_declared_communication_snapshot_epoch"
                snapshot_epoch = int(declared_snapshot_epoch)
                # Capture-time topology is part of the factor provenance.  A
                # missing explicitly declared snapshot cannot be replaced by an
                # arrival- or solve-time graph: doing so changes whether the
                # physical participants could have formed the likelihood when
                # the data were acquired.
                snapshot = self.communication.get(snapshot_epoch)
                if snapshot is None:
                    return "declared_communication_snapshot_missing"
            else:
                snapshot_epoch = int(factor.arrival_epoch)
                snapshot = self.communication.get(snapshot_epoch)
                if snapshot is None and factor.arrival_epoch != factor.observation_epoch:
                    snapshot = self.communication.get(factor.observation_epoch)
            if snapshot is None:
                return "communication_snapshot_missing"
            if not snapshot.participants_connected(factor.participants):
                return "cross_partition_factor_unavailable"
            registered = self.observation_hypergraph.get(factor.observation_epoch, set())
            edge = tuple(sorted(set(factor.participants)))
            if edge not in registered:
                return "observation_hyperedge_not_registered"
        return None

    @staticmethod
    def _association_weight(factor: DynamicFactor) -> float:
        probabilities = np.asarray(factor.association_probabilities, dtype=float)
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
        maximum_entropy = math.log(max(probabilities.size, 1))
        confidence = float(np.max(probabilities))
        entropy_retention = 1.0 if maximum_entropy == 0.0 else max(0.0, 1.0 - entropy / maximum_entropy)
        return float(np.clip(confidence * entropy_retention + (1.0 - confidence) / max(probabilities.size, 1), 0.0, 1.0))

    @staticmethod
    def _robust_weight(kernel: RobustKernel, whitened_norm: float, scale: float) -> float:
        if kernel is RobustKernel.NONE:
            return 1.0
        ratio = float(whitened_norm) / max(float(scale), 1e-12)
        if kernel is RobustKernel.HUBER:
            return 1.0 if ratio <= 1.0 else 1.0 / ratio
        return 1.0 / (1.0 + ratio * ratio)

    def validate_owned_separators(self) -> None:
        """Check published separators before either solve or fast scoring.

        Ownership is graph state, not a consequence of greedy factor order.
        A missing or externally edited separator invalidates that state.
        """
        present = {factor.factor_id: factor for factor in self.factors}
        for identity, digest in self._owned_separator_digests.items():
            if identity not in present or _frozen_digest(present[identity]) != digest:
                raise RuntimeError("owned separator identity/content was modified")

    def evidence_admission_reason(self, factor: DynamicFactor) -> str | None:
        """Shared ownership gate for actual and hypothetical factors.

        Call ``validate_owned_separators`` once before examining a batch.
        This gate does not grant arrival, topology, or numerical validity.
        New members of an absorbed correlation group require reconstruction of
        its joint separator; independently adding them would change frozen
        combination weights without reconstructing the absorbed factors.
        """
        owned_digest = self._owned_separator_digests.get(factor.factor_id)
        if owned_digest is not None and _frozen_digest(factor) != owned_digest:
            return "owned_separator_identity_conflict"
        roots = self._fusion_provenance_lineages(factor)
        if any(owner != factor.factor_id
               for root, owner in self._represented_root_owners.items()
               if root in roots):
            return "evidence_already_represented_by_separator"
        if (factor.correlation_group is not None
                and factor.correlation_group in self._represented_group_owners
                and self._represented_group_owners[factor.correlation_group] != factor.factor_id):
            return "correlation_group_requires_joint_separator_replacement"
        return None

    def _accepted_factors(self, current_epoch: int, active_keys: set[VariableKey]) -> tuple[list[DynamicFactor], list[FusionRejection]]:
        self.validate_owned_separators()
        accepted: list[DynamicFactor] = []
        rejected: list[FusionRejection] = []
        seen_exact: set[tuple[str, ...]] = set()
        accepted_provenance: list[tuple[set[str], str | None]] = []
        for factor in sorted(self.factors, key=lambda item: (item.arrival_epoch, item.observation_epoch, item.factor_id)):
            reason = self._factor_availability(factor, current_epoch, active_keys)
            if reason is not None:
                if reason not in {"not_arrived", "variable_outside_active_window"}:
                    rejected.append(FusionRejection(factor.factor_id, reason))
                continue
            reason = self.evidence_admission_reason(factor)
            if reason is not None:
                rejected.append(FusionRejection(factor.factor_id, reason))
                continue
            provenance_key = tuple(sorted(set(factor.provenance_ids)))
            if provenance_key in seen_exact:
                rejected.append(FusionRejection(factor.factor_id, "duplicate_correlated_provenance"))
                continue
            provenance_set = self._fusion_provenance_lineages(factor)
            overlapping = [
                (roots, group)
                for roots, group in accepted_provenance
                if roots.intersection(provenance_set)
            ]
            # Multiple descendants of one raw capture are admissible only when
            # the producer explicitly declares the same ``provenance_roots``
            # lineage and the factors share a correlation group.  We do *not*
            # infer fusion ancestry by truncating colon-delimited identifiers:
            # strings such as ``forecast:k:candidate:scenario:action`` encode
            # different hypothetical measurements whose common prefix is not a
            # common random sample.  The equal-share policy caps the aggregate
            # information from an explicitly shared root.  An unregistered
            # overlap is rejected before it can inflate the normal equations.
            if any(
                factor.correlation_group is None
                or group is None
                or group != factor.correlation_group
                for _, group in overlapping
            ):
                rejected.append(FusionRejection(factor.factor_id, "overlapping_raw_provenance_without_correlation_group"))
                continue
            seen_exact.add(provenance_key)
            accepted.append(factor)
            accepted_provenance.append((provenance_set, factor.correlation_group))
        return accepted, rejected

    @staticmethod
    def _fusion_provenance_lineages(factor: DynamicFactor) -> set[str]:
        """Return evidence identities used by the fusion double-count gate.

        Full provenance identifiers are always identities.  Additional shared
        raw-capture lineages are honored only when the factor producer declares
        ``metadata['provenance_roots']`` explicitly.  This makes statistical
        dependence a typed contract rather than a filename-style heuristic.
        """

        if any(type(value) is not str or not value for value in factor.provenance_ids):
            raise ValueError("evidence identities must be nonempty strings")
        identities = set(factor.provenance_ids)
        declared = factor.metadata.get("provenance_roots")
        if declared is not None:
            if not isinstance(declared, (list, tuple, set, frozenset)) or any(
                    type(value) is not str or not value for value in declared):
                raise ValueError("explicit evidence roots must be nonempty strings")
            identities.update(declared)
        return identities

    def _index(self, variables: Sequence[DynamicVariable]) -> tuple[dict[VariableKey, slice], int]:
        index: dict[VariableKey, slice] = {}
        cursor = 0
        for variable in variables:
            index[variable.key] = slice(cursor, cursor + variable.dimension)
            cursor += variable.dimension
        return index, cursor

    @staticmethod
    def _stable_inverse(
        matrix: np.ndarray,
        *,
        pseudoinverse_rcond: float = 1e-12,
    ) -> np.ndarray:
        """Invert an SPD matrix without discarding small valid modes.

        Cholesky is attempted first.  This is essential for normalized radar
        information matrices whose condition number can exceed the default
        pseudoinverse cutoff while remaining strictly identifiable.  Singular
        gauge systems retain the previous spectral-pseudoinverse semantics.
        """

        symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        try:
            inverse = _equilibrated_spd_inverse(symmetric)
            if np.all(np.isfinite(inverse)):
                return inverse
        except np.linalg.LinAlgError:
            pass
        eigenvalues, eigenvectors = platform_symmetric_eigh(symmetric)
        threshold = _spectral_threshold(eigenvalues, pseudoinverse_rcond)
        if float(eigenvalues.min()) < -threshold:
            raise np.linalg.LinAlgError(
                "matrix is materially indefinite; a spectral pseudoinverse "
                "would hide negative information/covariance"
            )
        retained = eigenvalues > threshold
        if not np.any(retained):
            return np.zeros_like(symmetric)
        covariance_root = eigenvectors[:, retained] / np.sqrt(
            eigenvalues[retained]
        )
        inverse = platform_matmul(covariance_root, covariance_root.T)
        return 0.5 * (inverse + inverse.T)

    @classmethod
    def _stable_solve(cls, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        right = np.asarray(vector, dtype=float)
        try:
            result = _equilibrated_spd_solve(symmetric, right)
            if np.all(np.isfinite(result)):
                return result
        except np.linalg.LinAlgError:
            pass
        return platform_matmul(cls._stable_inverse(symmetric), right)

    @staticmethod
    def _whiten(
        covariance: np.ndarray,
        jacobian: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Whiten one Gaussian residual without first forming normal equations."""

        symmetric = 0.5 * (
            np.asarray(covariance, dtype=float)
            + np.asarray(covariance, dtype=float).T
        )
        matrix = np.asarray(jacobian, dtype=float)
        value = np.asarray(measurement, dtype=float)
        try:
            lower = platform_cholesky(symmetric)
            return platform_solve(lower, matrix), platform_solve(lower, value)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = platform_symmetric_eigh(symmetric)
            threshold = _spectral_threshold(eigenvalues, 1e-12)
            if float(eigenvalues.min()) < -threshold:
                raise np.linalg.LinAlgError(
                    "cannot whiten a materially indefinite covariance"
                )
            retained = eigenvalues > threshold
            if not np.any(retained):
                return (
                    np.zeros((0, matrix.shape[1]), dtype=float),
                    np.zeros(0, dtype=float),
                )
            whitener = (
                eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
            ).T
            return (
                platform_matmul(whitener, matrix),
                platform_matmul(whitener, value),
            )

    def _square_root_system(
        self,
        variables: Sequence[DynamicVariable],
        factors: Sequence[DynamicFactor],
        robust_weights: Mapping[str, float],
        *,
        combined_weights: Mapping[str, float] | None = None,
        prior_keys: set[VariableKey] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble whitened residual rows for numerically stable elimination."""

        index, dimension = self._index(variables)
        rows: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for variable in variables:
            if not variable.prior_enabled or (prior_keys is not None and variable.key not in prior_keys):
                continue
            local = np.zeros((variable.dimension, dimension), dtype=float)
            local[:, index[variable.key]] = np.eye(variable.dimension)
            whitened, measurement = self._whiten(
                variable.prior_covariance,
                local,
                variable.prior_mean,
            )
            rows.append(whitened)
            values.append(measurement)

        group_counts: dict[str, int] = {}
        for factor in factors:
            if factor.correlation_group:
                group_counts[factor.correlation_group] = (
                    group_counts.get(factor.correlation_group, 0) + 1
                )
        for factor in factors:
            association = self._association_weight(factor)
            correlation = 1.0
            if factor.correlation_group:
                count = group_counts[factor.correlation_group]
                if self.correlation_weight_policy == "equal_share":
                    correlation = 1.0 / count
                else:
                    first = min(
                        item.factor_id
                        for item in factors
                        if item.correlation_group == factor.correlation_group
                    )
                    correlation = 1.0 if factor.factor_id == first else 0.0
            weight = (
                association
                * correlation
                * float(robust_weights.get(factor.factor_id, 1.0))
            )
            if combined_weights is not None:
                weight = float(combined_weights[factor.factor_id])
            if weight <= 0.0:
                continue
            local = np.zeros((factor.measurement.size, dimension), dtype=float)
            for key, jacobian in zip(factor.variables, factor.jacobians):
                local[:, index[key]] = jacobian
            whitened, measurement = self._whiten(
                factor.covariance,
                local,
                factor.measurement,
            )
            scale = math.sqrt(weight)
            rows.append(scale * whitened)
            values.append(scale * measurement)
        if not rows:
            return np.zeros((0, dimension), dtype=float), np.zeros(0, dtype=float)
        return np.vstack(rows), np.concatenate(values)

    def _assemble(
        self,
        current_epoch: int,
        variables: Sequence[DynamicVariable],
        factors: Sequence[DynamicFactor],
        robust_weights: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        index, dimension = self._index(variables)
        information = np.zeros((dimension, dimension), dtype=float)
        vector = np.zeros(dimension, dtype=float)
        # Explicit priors.  Singular gauge priors are allowed; anchored variables
        # use their full inverse while gauge variables use a pseudoinverse.
        for variable in variables:
            if not variable.prior_enabled:
                continue
            sl = index[variable.key]
            covariance = variable.prior_covariance
            inverse = self._stable_inverse(covariance)
            information[sl, sl] += inverse
            vector[sl] += platform_matmul(inverse, variable.prior_mean)
        group_counts: dict[str, int] = {}
        for factor in factors:
            if factor.correlation_group:
                group_counts[factor.correlation_group] = group_counts.get(factor.correlation_group, 0) + 1
        provenance_weight_sums: dict[str, float] = {}
        for factor in factors:
            association = self._association_weight(factor)
            correlation = 1.0
            if factor.correlation_group:
                count = group_counts[factor.correlation_group]
                correlation = (1.0 / count) if self.correlation_weight_policy == "equal_share" else (1.0 if factor.factor_id == min(item.factor_id for item in factors if item.correlation_group == factor.correlation_group) else 0.0)
            weight = association * correlation * float(robust_weights.get(factor.factor_id, 1.0))
            if weight <= 0.0:
                continue
            inverse_covariance = self._stable_inverse(factor.covariance) * weight
            local_h = np.zeros((factor.measurement.size, dimension), dtype=float)
            for key, jacobian in zip(factor.variables, factor.jacobians):
                local_h[:, index[key]] = jacobian
            information += platform_matmul(
                platform_matmul(local_h.T, inverse_covariance), local_h
            )
            vector += platform_matmul(
                platform_matmul(local_h.T, inverse_covariance),
                factor.measurement,
            )
            for provenance in factor.provenance_ids:
                provenance_weight_sums[provenance] = provenance_weight_sums.get(provenance, 0.0) + weight
        information = 0.5 * (information + information.T)
        return information, vector, provenance_weight_sums

    @staticmethod
    def _positive_semidefinite_part(matrix: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        eigenvalues, eigenvectors = platform_symmetric_eigh(symmetric)
        tolerance = _spectral_threshold(eigenvalues, 1e-10)
        if float(eigenvalues.min()) < -tolerance:
            raise np.linalg.LinAlgError(
                "materially indefinite matrix cannot be converted to a PSD diagnostic"
            )
        return platform_matmul(
            eigenvectors * np.maximum(eigenvalues, 0.0), eigenvectors.T
        )

    @staticmethod
    def _provenance_roots(factor: DynamicFactor) -> set[str]:
        """Return declared or hierarchical provenance lineages.

        The first token (for example ``capture``) is only a record type and is
        not evidence of common statistical origin. Explicit roots take
        precedence; otherwise only the terminal sample identifier is removed.
        """

        declared = factor.metadata.get("provenance_roots")
        if isinstance(declared, (list, tuple, set)) and declared:
            return {str(value) for value in declared if str(value)}
        roots: set[str] = set()
        for value in factor.provenance_ids:
            parts = str(value).split(":")
            roots.add(":".join(parts[:-1]) if len(parts) >= 3 else str(value))
        return roots

    @staticmethod
    def _factor_relation_score(left: DynamicFactor, right: DynamicFactor) -> float:
        """Score intrinsic overlap between two local likelihood reductions."""

        shared_variables = len(set(left.variables).intersection(right.variables))
        shared_participants = len(set(left.participants).intersection(right.participants))
        left_roots = DynamicFactorGraph._provenance_roots(left)
        right_roots = DynamicFactorGraph._provenance_roots(right)
        shared_roots = len(left_roots.intersection(right_roots))
        same_correlation_group = bool(
            left.correlation_group
            and left.correlation_group == right.correlation_group
        )
        return float(
            4.0 * shared_variables
            + shared_participants
            + 2.0 * shared_roots
            + 3.0 * float(same_correlation_group)
        )

    @classmethod
    def _factor_scope_charts(
        cls,
        factors: Sequence[DynamicFactor],
        *,
        maximum_charts: int,
    ) -> list[tuple[DynamicFactor, ...]]:
        """Build deterministic overlapping charts from factor-scope stars."""

        ordered = tuple(
            sorted(
                factors,
                key=lambda item: (
                    item.observation_epoch,
                    item.arrival_epoch,
                    item.factor_id,
                ),
            )
        )
        if len(ordered) <= 1:
            return [ordered]
        target_size = min(
            len(ordered),
            max(2, int(math.ceil(math.sqrt(len(ordered)))) + 1),
        )
        candidates: dict[tuple[str, ...], tuple[DynamicFactor, ...]] = {}
        for seed_index, seed in enumerate(ordered):
            neighbors = [
                (
                    cls._factor_relation_score(seed, candidate),
                    -abs(seed.observation_epoch - candidate.observation_epoch),
                    candidate.factor_id,
                    candidate,
                )
                for candidate in ordered
                if candidate.factor_id != seed.factor_id
            ]
            ranked = sorted(
                neighbors,
                key=lambda item: (-item[0], -item[1], item[2]),
            )
            selected = [seed, *(item[3] for item in ranked[: target_size - 1])]
            chart = tuple(sorted(selected, key=lambda item: item.factor_id))
            candidates[tuple(item.factor_id for item in chart)] = chart
        if len(candidates) == 1:
            # Two-factor and fully symmetric cases still need a nontrivial
            # overlap test. Singleton charts overlap on their variable scopes.
            candidates = {
                (factor.factor_id,): (factor,) for factor in ordered
            }
        available = list(candidates.values())
        selected_charts: list[tuple[DynamicFactor, ...]] = []
        covered: set[str] = set()
        while available and len(selected_charts) < max(1, int(maximum_charts)):
            best = max(
                available,
                key=lambda chart: (
                    len({item.factor_id for item in chart}.difference(covered)),
                    len(chart),
                    tuple(reversed(tuple(item.factor_id for item in chart))),
                ),
            )
            selected_charts.append(best)
            covered.update(item.factor_id for item in best)
            available.remove(best)
            if len(covered) == len(ordered) and len(selected_charts) >= min(2, len(candidates)):
                break
        uncovered = [factor for factor in ordered if factor.factor_id not in covered]
        for factor in uncovered:
            if not selected_charts:
                selected_charts.append((factor,))
                continue
            destination = max(
                range(len(selected_charts)),
                key=lambda chart_index: (
                    max(
                        cls._factor_relation_score(factor, neighbor)
                        for neighbor in selected_charts[chart_index]
                    ),
                    -len(selected_charts[chart_index]),
                    tuple(
                        reversed(
                            tuple(
                                neighbor.factor_id
                                for neighbor in selected_charts[chart_index]
                            )
                        )
                    ),
                ),
            )
            selected_charts[destination] = tuple(
                sorted((*selected_charts[destination], factor), key=lambda item: item.factor_id)
            )
            covered.add(factor.factor_id)
        return sorted(
            selected_charts,
            key=lambda chart: tuple(item.factor_id for item in chart),
        )

    def _implicit_atlas_gluing(
        self,
        current_epoch: int,
        variables: Sequence[DynamicVariable],
        factors: Sequence[DynamicFactor],
        robust_weights: Mapping[str, float],
        global_information: np.ndarray,
        global_estimate: np.ndarray,
        base_covariance: np.ndarray,
        *,
        maximum_charts: int = 8,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Check local reductions without changing the global posterior.

        Charts are intrinsic factor-scope neighborhoods. Transition checks use
        only variables present in both local reductions. The nerve of chart
        overlaps must cover the accepted factors and stay connected. Residual
        disagreement on globally identifiable directions is reported as a
        positive-semidefinite *diagnostic* covariance correction.

        The exact global Gaussian posterior covariance is the inverse of the
        assembled global information matrix. Adding dispersion between
        partial-chart estimators to that covariance would count a diagnostic
        disagreement as a second stochastic uncertainty source. It would also
        make ``diagnostics=True`` change NEES, coverage, CRLB comparisons and
        downstream decisions while leaving the posterior mean unchanged. The
        proposed inflation is therefore retained in the audit record but is
        never substituted for the authoritative covariance.
        """

        ordered = tuple(sorted(
            factors,
            key=lambda item: (item.observation_epoch, item.arrival_epoch, item.factor_id),
        ))
        if len(ordered) < 2:
            return base_covariance, {
                "chart_count": 1,
                "construction": "single_global_chart_insufficient_factors_for_overlap",
                "factor_coverage_fraction": 1.0,
                "overlap_nerve_component_count": 1,
                "overlap_nerve_connected": True,
                "overlap_edge_fraction": 1.0,
                "minimum_overlap_variable_fraction": 1.0,
                "partition_of_unity_weights": [1.0],
                "factor_scope_local_reductions": True,
                "maximum_overlap_residual": 0.0,
                "median_overlap_residual": 0.0,
                "compatible_overlap_fraction": 1.0,
                "between_chart_covariance_trace": 0.0,
                "covariance_inflation_trace": 0.0,
                "local_to_global_gluing_finite": True,
            }
        charts = self._factor_scope_charts(ordered, maximum_charts=maximum_charts)
        chart_estimates: list[np.ndarray] = []
        chart_factor_ids: list[tuple[str, ...]] = []
        chart_ranks: list[int] = []
        chart_supports: list[tuple[VariableKey, ...]] = []
        chart_covariances: list[np.ndarray] = []
        global_eigenvalues, global_eigenvectors = platform_symmetric_eigh(
            0.5 * (global_information + global_information.T)
        )
        global_threshold = max(
            float(np.max(np.abs(global_eigenvalues))), 1.0
        ) * 1e-10
        identifiable = global_eigenvectors[:, global_eigenvalues > global_threshold]
        projector = (
            platform_matmul(identifiable, identifiable.T)
            if identifiable.size
            else np.zeros_like(global_information)
        )
        variable_index, _ = self._index(variables)
        prior_information, _, _ = self._assemble(current_epoch, variables, (), {})
        factor_coverage = {factor.factor_id: 0 for factor in ordered}
        chart_support_indices: list[np.ndarray] = []
        chart_confidences: list[float] = []
        for local_factors in charts:
            local_information, local_vector, _ = self._assemble(
                current_epoch,
                variables,
                local_factors,
                robust_weights,
            )
            # Every chart contains the registered global prior, so its local
            # information is positive definite even when the additional radar
            # rows are rank deficient.  Use the same equilibrated SPD inverse
            # as the authoritative solve.  A generic SVD pseudoinverse is both
            # unnecessary and has failed to converge for valid radar matrices
            # spanning clock, delay, Doppler and position units.
            local_covariance = self._stable_inverse(local_information)
            local_estimate = platform_matmul(local_covariance, local_vector)
            local_observed = self._positive_semidefinite_part(
                local_information - prior_information
            )
            local_eigenvalues = platform_symmetric_eigvalsh(local_observed)
            local_threshold = max(
                float(np.max(np.abs(local_eigenvalues))), 1.0
            ) * 1e-10
            local_rank = int(np.count_nonzero(local_eigenvalues > local_threshold))
            support = tuple(sorted(
                {key for factor in local_factors for key in factor.variables},
                key=lambda key: key.token(),
            ))
            support_indices = np.asarray([
                location
                for key in support
                for location in range(variable_index[key].start, variable_index[key].stop)
            ], dtype=int)
            chart_ranks.append(local_rank)
            chart_estimates.append(local_estimate)
            chart_factor_ids.append(tuple(factor.factor_id for factor in local_factors))
            chart_supports.append(support)
            chart_support_indices.append(support_indices)
            chart_covariances.append(local_covariance)
            chart_confidences.append(local_rank / max(support_indices.size, 1))
            for factor in local_factors:
                factor_coverage[factor.factor_id] += 1
        projected_rows: list[np.ndarray] = []
        for value, support_indices in zip(chart_estimates, chart_support_indices):
            supported_difference = np.zeros_like(global_estimate)
            supported_difference[support_indices] = (
                value[support_indices] - global_estimate[support_indices]
            )
            projected_rows.append(
                platform_matmul(projector, supported_difference)
            )
        projected = np.vstack(projected_rows)
        overlap_residuals: list[float] = []
        overlap_variable_fractions: list[float] = []
        nerve = np.zeros((len(chart_estimates), len(chart_estimates)), dtype=bool)
        for left in range(len(chart_estimates)):
            for right in range(left + 1, len(chart_estimates)):
                shared = tuple(sorted(
                    set(chart_supports[left]).intersection(chart_supports[right]),
                    key=lambda key: key.token(),
                ))
                if not shared:
                    continue
                nerve[left, right] = True
                nerve[right, left] = True
                overlap_indices = np.asarray([
                    location
                    for key in shared
                    for location in range(variable_index[key].start, variable_index[key].stop)
                ], dtype=int)
                difference = (
                    chart_estimates[left][overlap_indices]
                    - chart_estimates[right][overlap_indices]
                )
                overlap_covariance = (
                    chart_covariances[left][np.ix_(overlap_indices, overlap_indices)]
                    + chart_covariances[right][np.ix_(overlap_indices, overlap_indices)]
                )
                overlap_precision = self._stable_inverse(
                    overlap_covariance
                    + 1e-12 * np.eye(overlap_indices.size)
                )
                overlap_residuals.append(
                    math.sqrt(
                        max(
                            float(
                                platform_matmul(
                                    platform_matmul(difference, overlap_precision),
                                    difference,
                                )
                            ),
                            0.0,
                        )
                        / max(overlap_indices.size, 1)
                    )
                )
                overlap_variable_fractions.append(
                    overlap_indices.size
                    / max(
                        min(
                            chart_support_indices[left].size,
                            chart_support_indices[right].size,
                        ),
                        1,
                    )
                )
        visited: set[int] = set()
        nerve_component_count = 0
        for root in range(len(chart_estimates)):
            if root in visited:
                continue
            nerve_component_count += 1
            stack = [root]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(
                    int(value) for value in np.flatnonzero(nerve[node])
                    if int(value) not in visited
                )
        confidence = np.asarray(chart_confidences, dtype=float)
        if float(np.sum(confidence)) <= 0.0:
            confidence = np.ones_like(confidence)
        partition_weights = confidence / float(np.sum(confidence))
        between = sum(
            float(weight) * np.outer(row, row)
            for weight, row in zip(partition_weights, projected)
        )
        between = self._positive_semidefinite_part(between)
        diagnostic_glued = self._positive_semidefinite_part(
            base_covariance + between
        )
        coverage_fraction = float(
            np.mean([count > 0 for count in factor_coverage.values()])
        )
        total_chart_pairs = len(chart_estimates) * (len(chart_estimates) - 1) // 2
        overlap_edge_fraction = float(np.sum(nerve) / 2.0) / max(total_chart_pairs, 1)
        nerve_connected = nerve_component_count == 1
        finite = bool(
            np.all(np.isfinite(diagnostic_glued))
            and np.all(np.isfinite(projected))
            and all(math.isfinite(value) for value in overlap_residuals)
            and coverage_fraction == 1.0
            and nerve_connected
        )
        return base_covariance, {
            "chart_count": len(chart_estimates),
            "construction": "intrinsic_overlapping_factor_scope_local_reductions",
            "chart_ranks": chart_ranks,
            "chart_factor_counts": [len(value) for value in chart_factor_ids],
            "chart_support_dimensions": [int(value.size) for value in chart_support_indices],
            "factor_coverage_fraction": coverage_fraction,
            "minimum_factor_chart_multiplicity": min(factor_coverage.values(), default=0),
            "overlap_nerve_component_count": nerve_component_count,
            "overlap_nerve_connected": nerve_connected,
            "overlap_edge_fraction": overlap_edge_fraction,
            "minimum_overlap_variable_fraction": min(overlap_variable_fractions, default=0.0),
            "median_overlap_variable_fraction": float(np.median(overlap_variable_fractions)) if overlap_variable_fractions else 0.0,
            "partition_of_unity_weights": partition_weights.tolist(),
            "factor_scope_local_reductions": True,
            "maximum_overlap_residual": max(overlap_residuals, default=0.0),
            "median_overlap_residual": float(np.median(overlap_residuals))
            if overlap_residuals
            else 0.0,
            "compatible_overlap_fraction": float(
                np.mean([value <= 3.0 for value in overlap_residuals])
            )
            if overlap_residuals
            else 1.0,
            "between_chart_covariance_trace": float(np.trace(between)),
            "covariance_inflation_trace": float(
                np.trace(diagnostic_glued - base_covariance)
            ),
            "base_global_posterior_covariance_trace": float(
                np.trace(base_covariance)
            ),
            "diagnostic_glued_covariance_trace": float(
                np.trace(diagnostic_glued)
            ),
            "authoritative_covariance_source": (
                "inverse_of_global_assembled_information"
            ),
            "between_chart_inflation_is_diagnostic_only": True,
            "between_chart_inflation_applied_to_authoritative_covariance": False,
            "local_to_global_gluing_finite": finite,
            "global_mean_retained": True,
        }

    def _definable_rank_strata(
        self,
        observed_information: np.ndarray,
        total_information: np.ndarray,
        *,
        variables: Sequence[DynamicVariable],
        factors: Sequence[DynamicFactor],
        robust_weights: Mapping[str, float],
        structural_gauge_allowance: int,
        maximum_observable_rank: int,
    ) -> dict[str, Any]:
        """Classify tame rank-defect strata through vanishing singular minors."""

        observed = 0.5 * (observed_information + observed_information.T)
        total = 0.5 * (total_information + total_information.T)
        singular_values = platform_singular_values(observed)
        scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1e-300)
        tolerances = (1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
        ranks = {
            f"relative_tolerance_{tolerance:g}": int(
                np.count_nonzero(singular_values > scale * tolerance)
            )
            for tolerance in tolerances
        }
        variable_index, ambient_dimension = self._index(variables)
        local_bases: list[np.ndarray] = []
        local_factors: list[DynamicFactor] = []
        local_ranks: list[int] = []
        for factor in factors:
            if factor.missing or float(robust_weights.get(factor.factor_id, 1.0)) <= 1e-12:
                continue
            local_h = np.zeros((factor.measurement.size, ambient_dimension), dtype=float)
            for key, jacobian in zip(factor.variables, factor.jacobians):
                local_h[:, variable_index[key]] += jacobian
            chol = platform_cholesky(
                factor.covariance + 1e-12 * np.eye(factor.measurement.size)
            )
            whitened = platform_solve(chol, local_h)
            _, local_singular_values, right = platform_svd(
                whitened, full_matrices=False
            )
            local_scale = max(
                float(local_singular_values[0]) if local_singular_values.size else 0.0,
                1e-300,
            )
            local_rank = int(np.count_nonzero(local_singular_values > local_scale * 1e-6))
            if local_rank <= 0:
                continue
            local_bases.append(np.asarray(right[:local_rank], dtype=float))
            local_factors.append(factor)
            local_ranks.append(local_rank)

        overlap_scores: list[float] = []
        structured_overlap_weight = 0.0
        total_overlap_weight = 0.0
        special_reason_counts = {
            "shared_variable_scope": 0,
            "shared_nuisance_variable": 0,
            "shared_participant": 0,
            "shared_provenance_root": 0,
            "shared_correlation_group": 0,
            "identical_factor_scope": 0,
        }
        for left_index in range(len(local_bases)):
            for right_index in range(left_index + 1, len(local_bases)):
                left_basis = local_bases[left_index]
                right_basis = local_bases[right_index]
                denominator = max(1, min(left_basis.shape[0], right_basis.shape[0]))
                overlap = float(
                    platform_norm(
                        platform_matmul(left_basis, right_basis.T)
                    ) ** 2
                    / denominator
                )
                overlap = float(np.clip(overlap, 0.0, 1.0))
                overlap_scores.append(overlap)
                total_overlap_weight += overlap

                left = local_factors[left_index]
                right_factor = local_factors[right_index]
                left_variables = set(left.variables)
                right_variables = set(right_factor.variables)
                shared_variables = left_variables.intersection(right_variables)
                left_roots = self._provenance_roots(left)
                right_roots = self._provenance_roots(right_factor)
                reasons = {
                    "shared_variable_scope": bool(shared_variables),
                    "shared_nuisance_variable": any(
                        key.kind != VariableKind.TARGET for key in shared_variables
                    ),
                    "shared_participant": bool(
                        set(left.participants).intersection(right_factor.participants)
                    ),
                    "shared_provenance_root": bool(left_roots.intersection(right_roots)),
                    "shared_correlation_group": bool(
                        left.correlation_group
                        and left.correlation_group == right_factor.correlation_group
                    ),
                    "identical_factor_scope": left_variables == right_variables,
                }
                for reason, present in reasons.items():
                    special_reason_counts[reason] += int(present)
                nuisance_indices = np.asarray(
                    [
                        location
                        for key in shared_variables
                        if key.kind != VariableKind.TARGET
                        for location in range(
                            variable_index[key].start,
                            variable_index[key].stop,
                        )
                    ],
                    dtype=int,
                )
                nuisance_overlap = 0.0
                if nuisance_indices.size:
                    nuisance_overlap = float(
                        platform_norm(
                            platform_matmul(
                                left_basis[:, nuisance_indices],
                                right_basis[:, nuisance_indices].T,
                            )
                        )
                        ** 2
                        / denominator
                    )
                strong_relation = bool(
                    reasons["shared_participant"]
                    or reasons["shared_provenance_root"]
                    or reasons["shared_correlation_group"]
                )
                structured_overlap_weight += min(
                    overlap, overlap if strong_relation else nuisance_overlap
                )

        total_eigenvalues = platform_symmetric_eigvalsh(total)
        total_scale = max(float(np.max(np.abs(total_eigenvalues))), 1.0)
        total_rank = int(np.count_nonzero(total_eigenvalues > total_scale * 1e-10))
        generic_expected_rank = min(
            max(0, int(total_information.shape[0]) - int(structural_gauge_allowance)),
            max(0, int(maximum_observable_rank)),
            sum(local_ranks),
        )
        loose_rank = ranks["relative_tolerance_0.001"]
        atypical_defect = max(0, generic_expected_rank - loose_rank)
        adjacent_rank_pairs = list(zip(ranks.values(), tuple(ranks.values())[1:]))
        rank_persistence = (
            float(np.mean([left == right for left, right in adjacent_rank_pairs]))
            if adjacent_rank_pairs
            else 1.0
        )
        special_structure_fraction = (
            float(np.clip(structured_overlap_weight / total_overlap_weight, 0.0, 1.0))
            if total_overlap_weight > 1e-12
            else (1.0 if atypical_defect == 0 else 0.0)
        )
        unexplained_defect = float(atypical_defect) * (1.0 - special_structure_fraction)
        generic_attainment = (
            float(np.clip(loose_rank / generic_expected_rank, 0.0, 1.0))
            if generic_expected_rank > 0
            else 1.0
        )
        return {
            "ambient_dimension": int(total_information.shape[0]),
            "measurement_rank_by_tolerance": ranks,
            "total_information_rank": total_rank,
            "structural_gauge_allowance": int(structural_gauge_allowance),
            "expected_rank_after_gauge": generic_expected_rank,
            "generic_expected_rank": generic_expected_rank,
            "local_measurement_ranks": local_ranks,
            "generic_rank_attainment_fraction": generic_attainment,
            "atypical_intersection_defect": atypical_defect,
            "special_structure_explained_fraction": special_structure_fraction,
            "unexplained_atypical_defect": unexplained_defect,
            "special_structure_reason_counts": special_reason_counts,
            "grassmann_pair_overlap_scores": overlap_scores,
            "grassmann_transversality": float(
                1.0 - np.mean(overlap_scores) if overlap_scores else 1.0
            ),
            "rank_stratum_persistence": rank_persistence,
            "special_structure_certificate_finite": bool(
                math.isfinite(special_structure_fraction)
                and math.isfinite(unexplained_defect)
                and math.isfinite(rank_persistence)
            ),
            "stratum_id": f"measurement_rank_{loose_rank}_of_{generic_expected_rank}",
            "rank_stable_across_tolerances": len(set(ranks.values())) == 1,
            "largest_measurement_singular_value": float(scale),
            "smallest_nonzero_measurement_singular_value": float(
                singular_values[loose_rank - 1] if loose_rank else 0.0
            ),
            "interpretation": "finite numerical stratum of the analytic observation-map rank locus",
        }

    @staticmethod
    def _interaction_cumulant_diagnostics(
        factors: Sequence[DynamicFactor],
    ) -> dict[str, Any]:
        """Measure interaction-history cycles without treating them as independent."""

        adjacency: dict[str, set[str]] = {}

        def connect(left: str, right: str) -> None:
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

        provenance_adjacency: dict[str, set[str]] = {}

        def connect_provenance(left: str, right: str) -> None:
            provenance_adjacency.setdefault(left, set()).add(right)
            provenance_adjacency.setdefault(right, set()).add(left)

        factor_by_id = {factor.factor_id: factor for factor in factors}
        for factor in factors:
            factor_node = f"factor:{factor.factor_id}"
            adjacency.setdefault(factor_node, set())
            provenance_adjacency.setdefault(factor_node, set())
            for provenance in factor.provenance_ids:
                provenance_node = f"provenance:{provenance}"
                connect(factor_node, provenance_node)
                connect_provenance(factor_node, provenance_node)
            for participant in factor.participants:
                connect(factor_node, f"participant:{participant}")

        factor_adjacency = {factor.factor_id: set() for factor in factors}
        ordered_factor_ids = sorted(factor_by_id)
        for left_index, left_id in enumerate(ordered_factor_ids):
            left = factor_by_id[left_id]
            for right_id in ordered_factor_ids[left_index + 1 :]:
                right = factor_by_id[right_id]
                left_roots = DynamicFactorGraph._provenance_roots(left)
                right_roots = DynamicFactorGraph._provenance_roots(right)
                connected = bool(
                    set(left.variables).intersection(right.variables)
                    or set(left.participants).intersection(right.participants)
                    or left_roots.intersection(right_roots)
                    or (
                        left.correlation_group
                        and left.correlation_group == right.correlation_group
                    )
                )
                if connected:
                    factor_adjacency[left_id].add(right_id)
                    factor_adjacency[right_id].add(left_id)

        connected_pairs = sum(len(value) for value in factor_adjacency.values()) // 2
        connected_triples = 0
        tree_triples = 0
        cyclic_triples = 0
        for first_index, first in enumerate(ordered_factor_ids):
            for second_index in range(first_index + 1, len(ordered_factor_ids)):
                second = ordered_factor_ids[second_index]
                for third in ordered_factor_ids[second_index + 1 :]:
                    induced_edges = int(second in factor_adjacency[first])
                    induced_edges += int(third in factor_adjacency[first])
                    induced_edges += int(third in factor_adjacency[second])
                    if induced_edges >= 2:
                        connected_triples += 1
                        tree_triples += int(induced_edges == 2)
                        cyclic_triples += int(induced_edges == 3)

        def cycle_summary(graph: Mapping[str, set[str]]) -> tuple[int, int, int]:
            visited: set[str] = set()
            components = 0
            for root in graph:
                if root in visited:
                    continue
                components += 1
                stack = [root]
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    stack.extend(graph[node].difference(visited))
            edges = sum(len(neighbors) for neighbors in graph.values()) // 2
            circuit_rank = max(0, edges - len(graph) + components)
            return components, edges, circuit_rank

        components, edges, circuit_rank = cycle_summary(adjacency)
        provenance_components, provenance_edges, provenance_circuit_rank = cycle_summary(
            provenance_adjacency
        )
        factor_count = len(ordered_factor_ids)
        possible_pairs = math.comb(factor_count, 2) if factor_count >= 2 else 0
        possible_triples = math.comb(factor_count, 3) if factor_count >= 3 else 0
        return {
            "incidence_vertex_count": len(adjacency),
            "incidence_edge_count": edges,
            "elementary_connected_component_count": components,
            "incidence_circuit_rank": circuit_rank,
            "provenance_vertex_count": len(provenance_adjacency),
            "provenance_edge_count": provenance_edges,
            "provenance_component_count": provenance_components,
            "provenance_circuit_rank": provenance_circuit_rank,
            "interaction_cumulant_order_upper_bound": provenance_circuit_rank + 1,
            "connected_pair_motif_count": connected_pairs,
            "connected_pair_motif_density": connected_pairs / max(possible_pairs, 1),
            "connected_triple_motif_count": connected_triples,
            "connected_triple_motif_density": connected_triples / max(possible_triples, 1),
            "tree_connected_triple_count": tree_triples,
            "cyclic_connected_triple_count": cyclic_triples,
            "connected_cumulant_surrogate_orders_reported": [2, 3],
            "cumulant_truncation_order": 3,
            "higher_order_dependence_explicitly_reported": True,
            "cycle_count_treated_as_independent_information": False,
            "cycle_rank_is_not_cumulant": True,
            "interpretation": "connected motifs summarize pair/triple interaction closure; circuit rank is only a topology diagnostic",
        }

    def solve(
        self,
        current_epoch: int,
        *,
        irls_iterations: int = 4,
        diagnostics: bool = True,
    ) -> GraphSolution:
        """Solve the active fixed-lag graph.

        ``diagnostics=False`` retains the same normal equations, robust IRLS,
        provenance accounting, gauge audit, estimates, and covariance while
        skipping the expensive atlas/rank/cumulant research diagnostics.  It is
        the operational path used for real-time decisions; the default remains
        the full audited path used by the existing V65/V68 closure tests.
        """

        self._require_gaussian_mode()
        variables = self._active_variables(current_epoch)
        if not variables:
            raise ValueError("no active variables at requested epoch")
        active_keys = {variable.key for variable in variables}
        factors, rejected = self._accepted_factors(current_epoch, active_keys)
        weights = {factor.factor_id: 1.0 for factor in factors}
        estimate: np.ndarray | None = None
        provenance_sums: dict[str, float] = {}
        information = np.zeros((1, 1))
        vector = np.zeros(1)
        for _ in range(max(1, irls_iterations)):
            information, vector, provenance_sums = self._assemble(current_epoch, variables, factors, weights)
            estimate = self._stable_solve(information, vector)
            index, _ = self._index(variables)
            for factor in factors:
                prediction = np.zeros(factor.measurement.size, dtype=float)
                for key, jacobian in zip(factor.variables, factor.jacobians):
                    prediction += platform_matmul(
                        jacobian, estimate[index[key]]
                    )
                residual = factor.measurement - prediction
                chol = platform_cholesky(
                    factor.covariance
                    + 1e-12 * np.eye(factor.measurement.size)
                )
                norm = platform_norm(platform_solve(chol, residual))
                weights[factor.factor_id] = self._robust_weight(factor.robust_kernel, norm, factor.robust_scale)
        # The weights computed by the final IRLS residual evaluation must be
        # used in the returned estimate.  Without this reassembly, the report
        # exposed one set of robust weights while the state/covariance still
        # corresponded to the preceding iterate.
        information, vector, provenance_sums = self._assemble(
            current_epoch, variables, factors, weights
        )
        estimate = self._stable_solve(information, vector)
        assert estimate is not None
        base_covariance = self._stable_inverse(information)
        index, _ = self._index(variables)
        estimates: dict[str, np.ndarray] = {}
        slices: dict[str, tuple[int, int]] = {}
        for variable in variables:
            sl = index[variable.key]
            token = variable.key.token()
            estimates[token] = estimate[sl].copy()
            slices[token] = (int(sl.start), int(sl.stop))
        # Exact arrival-order control: compare the batch equations with one
        # independently ordered full reassembly.  Earlier implementations also
        # assembled every prefix, but discarded every prefix result; retaining
        # only the identical final call removes quadratic work without changing
        # the audited normal equations or floating-point accumulation order.
        arrival_ordered_factors = tuple(
            sorted(factors, key=lambda item: (item.arrival_epoch, item.factor_id))
        )
        incremental_information, incremental_vector, _ = self._assemble(
            current_epoch,
            variables,
            arrival_ordered_factors,
            weights,
        )
        incremental_estimate = self._stable_solve(
            incremental_information, incremental_vector
        )
        residual = float(
            platform_norm(incremental_estimate - estimate)
            / max(platform_norm(estimate), 1.0)
        )
        eigenvalues = platform_symmetric_eigvalsh(information)
        threshold = _spectral_threshold(eigenvalues, 1e-10)
        nullity = int(np.count_nonzero(eigenvalues <= threshold))
        gauge_groups: dict[str, dict[str, Any]] = {}
        for variable in variables:
            if variable.gauge_group:
                record = gauge_groups.setdefault(
                    variable.gauge_group,
                    {"variables": [], "anchored": False, "indices": []},
                )
                record["variables"].append(variable.key.token())
                record["anchored"] = bool(record["anchored"] or variable.anchored)
                sl = index[variable.key]
                record["indices"].extend(range(int(sl.start), int(sl.stop)))
        for record in gauge_groups.values():
            group_indices = np.asarray(record.pop("indices"), dtype=int)
            block = information[np.ix_(group_indices, group_indices)]
            block_eigenvalues = platform_symmetric_eigvalsh(0.5 * (block + block.T))
            block_threshold = _spectral_threshold(block_eigenvalues, 1e-10)
            record["information_nullity"] = int(np.count_nonzero(block_eigenvalues <= block_threshold))
            record["minimum_eigenvalue"] = float(block_eigenvalues.min())
            # An anchored gauge may coexist with other unobservable nuisance
            # directions, so full block rank is not required.  What is required
            # is either an explicit anchor in this same group or a null
            # direction in this same group; a nullity elsewhere in the graph no
            # longer satisfies the clock/phase gauge audit.
            record["gauge_condition_satisfied"] = bool(
                record["anchored"] or record["information_nullity"] > 0
            )
        gauge = {
            "information_nullity": nullity,
            "minimum_eigenvalue": float(eigenvalues.min()),
            "groups": gauge_groups,
            "unanchored_groups": sorted(group for group, record in gauge_groups.items() if not record["anchored"]),
            "gauge_preserved": all(record["gauge_condition_satisfied"] for record in gauge_groups.values()),
        }
        if diagnostics:
            covariance, atlas = self._implicit_atlas_gluing(
                current_epoch,
                variables,
                factors,
                weights,
                information,
                estimate,
                base_covariance,
            )
            prior_information, _, _ = self._assemble(
                current_epoch, variables, (), {}
            )
            rank_strata = self._definable_rank_strata(
                information - prior_information,
                information,
                variables=variables,
                factors=factors,
                robust_weights=weights,
                structural_gauge_allowance=len(gauge["unanchored_groups"]),
                maximum_observable_rank=sum(
                    factor.measurement.size for factor in factors
                ),
            )
            interaction_cumulants = self._interaction_cumulant_diagnostics(factors)
        else:
            covariance = base_covariance
            atlas = {
                "operational_diagnostics_skipped": True,
                "local_to_global_gluing_finite": True,
                "reason": "normal_equations_and_covariance_only",
            }
            rank_strata = {
                "operational_diagnostics_skipped": True,
                "stratum_id": "not_evaluated_in_operational_path",
            }
            interaction_cumulants = {
                "operational_diagnostics_skipped": True,
                "cycle_count_treated_as_independent_information": False,
                "cycle_rank_is_not_cumulant": True,
            }
        covariances: dict[str, np.ndarray] = {}
        for variable in variables:
            sl = index[variable.key]
            covariances[variable.key.token()] = covariance[sl, sl].copy()
        covariance_eigenvalues = platform_symmetric_eigvalsh(
            0.5 * (covariance + covariance.T)
        )
        status = "passed" if (
            residual < 1e-10
            and all(value <= 1.0 + 1e-9 for value in provenance_sums.values())
            and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in weights.values())
            and float(eigenvalues.min()) >= -threshold
            and gauge["gauge_preserved"]
            and atlas["local_to_global_gluing_finite"]
            and float(covariance_eigenvalues.min()) >= -1e-9
        ) else "failed"
        result = GraphSolution(
            epoch=int(current_epoch),
            estimates=estimates,
            covariances=covariances,
            joint_information=information,
            joint_covariance=covariance,
            variable_slices=slices,
            accepted_factor_ids=tuple(factor.factor_id for factor in factors),
            rejected_factors=tuple(rejected),
            robust_weights=weights,
            provenance_weight_sums=provenance_sums,
            gauge_diagnostics=gauge,
            atlas_diagnostics=atlas,
            definable_rank_strata=rank_strata,
            interaction_cumulant_diagnostics=interaction_cumulants,
            exact_batch_incremental_residual=residual,
            status=status,
        )
        _ISSUED_SOLUTIONS[self] = (result, self._model_digest(), _frozen_digest(result))
        return result

    def v73_authority_digest(self):
        """Deterministic live-posterior digest used to detect candidate-score mutation."""
        self._require_gaussian_mode()
        from wideband_info_geometry.operational_runtime.v73_runtime_cycle import canonical_digest
        return canonical_digest(self)

    def v73_frozen_counterfactual_fork(self):
        """Return an isolated fork; the live graph remains the sole posterior authority."""
        self._require_gaussian_mode()
        import copy
        fork = copy.deepcopy(self)
        if fork is self:
            raise RuntimeError("counterfactual fork aliases live DynamicFactorGraph")
        return fork

    def v73_validate_controller_observables(self, observables):
        """Reject truth/family/seed/future/test features before controller use."""
        from wideband_info_geometry.operational_runtime.v73_adaptive_belief_risk import reject_forbidden_features
        reject_forbidden_features(observables)
        return observables


def target_variable(
    epoch: int,
    *,
    entity_id: str = "primary",
    prior_mean: np.ndarray | None = None,
    prior_covariance: np.ndarray | None = None,
    anchored: bool = True,
    prior_enabled: bool = True,
) -> DynamicVariable:
    covariance = np.asarray(prior_covariance if prior_covariance is not None else np.diag([1.0] * 6), dtype=float)
    mean = np.zeros(6, dtype=float) if prior_mean is None else np.asarray(prior_mean, dtype=float).reshape(6)
    return DynamicVariable(
        key=VariableKey(VariableKind.TARGET, str(entity_id), int(epoch)),
        dimension=6,
        prior_mean=mean,
        prior_covariance=covariance,
        units=("normalized_m", "normalized_m", "normalized_m", "normalized_m_s", "normalized_m_s", "normalized_m_s"),
        gauge_group=None,
        anchored=anchored,
        prior_enabled=prior_enabled,
    )


def nuisance_variable(
    layout: StateLayout,
    epoch: int,
    *,
    prior_mean: np.ndarray | None = None,
    prior_covariance: np.ndarray | None = None,
    anchor_clock: bool = False,
    prior_enabled: bool = True,
) -> DynamicVariable:
    dimension = layout.dimension - layout.target_dim
    covariance = np.asarray(
        prior_covariance if prior_covariance is not None else np.eye(dimension) * 1e6,
        dtype=float,
    )
    if prior_covariance is None and anchor_clock and dimension:
        covariance[0, 0] = 1.0
    mean = (
        np.zeros(dimension, dtype=float)
        if prior_mean is None
        else np.asarray(prior_mean, dtype=float).reshape(dimension)
    )
    units = tuple(["normalized_clock_or_frequency_or_delay"] * dimension)
    return DynamicVariable(
        key=VariableKey(VariableKind.NUISANCE, "network", int(epoch)),
        dimension=dimension,
        prior_mean=mean,
        prior_covariance=covariance,
        units=units,
        gauge_group="network_time_phase_reference",
        anchored=anchor_clock,
        prior_enabled=prior_enabled,
    )


def make_action_dynamic_factor(
    world: PhysicalWorld,
    action: RadarAction,
    action_factor: ActionFactor,
    *,
    epoch: int,
    arrival_epoch: int | None = None,
    measurement_residual: np.ndarray | None = None,
    measurement_value: np.ndarray | None = None,
    linearization_state_normalized: np.ndarray | None = None,
    factor_id: str | None = None,
    communication_snapshot_epoch: int | None = None,
    provenance_id: str | None = None,
    correlation_group: str | None = None,
    robust_kernel: RobustKernel = RobustKernel.HUBER,
    association_probabilities: Sequence[float] | None = None,
    target_entity_id: str = "primary",
    missing: bool = False,
    fidelity_level: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DynamicFactor:
    layout = StateLayout.from_world(world)
    scales = layout.scales()
    normalized_h = action_factor.jacobian_physical * scales[None, :]
    target_key = VariableKey(VariableKind.TARGET, str(target_entity_id), int(epoch))
    nuisance_key = VariableKey(VariableKind.NUISANCE, "network", int(epoch))
    row_count = len(action_factor.measurement_names)
    if measurement_value is not None and measurement_residual is not None:
        raise ValueError("provide measurement_value or measurement_residual, not both")
    if measurement_value is not None:
        value = np.asarray(measurement_value, dtype=float).reshape(-1)
        if value.shape != (row_count,):
            raise ValueError("measurement value dimension disagrees with ActionFactor")
        residual = value - np.asarray(action_factor.prediction, dtype=float)
    elif measurement_residual is not None:
        residual = np.asarray(measurement_residual, dtype=float).reshape(-1)
    else:
        residual = np.zeros(row_count, dtype=float)
    if residual.shape != (row_count,) or not np.all(np.isfinite(residual)):
        raise ValueError("measurement residual dimension or values are invalid")
    if linearization_state_normalized is not None:
        linearization = np.asarray(linearization_state_normalized, dtype=float).reshape(-1)
        if linearization.shape != (layout.dimension,) or not np.all(np.isfinite(linearization)):
            raise ValueError("linearization state dimension or values are invalid")
        # For h(theta) linearized at theta_0,
        #   z ~= h(theta_0) + H(theta-theta_0)
        # is represented in the graph as H theta ~= z-h(theta_0)+H theta_0.
        measurement = residual + platform_matmul(normalized_h, linearization)
        factor_semantics = "absolute_normalized_state_linearization"
    else:
        measurement = residual
        factor_semantics = "legacy_local_correction_state"
    participants = tuple(sorted({node for node in (action.tx_node_id, action.reference_node_id, *action.rx_node_ids) if node}))
    probabilities = association_probabilities or (
        float(action_factor.link_budget.correct_association_probability),
        float(1.0 - action_factor.link_budget.correct_association_probability),
    )
    kind = FactorKind.CALIBRATION if action.kind.startswith("calibration") else FactorKind.MEASUREMENT
    arrival = int(epoch if arrival_epoch is None else arrival_epoch)
    extra_metadata = dict(metadata or {})
    return DynamicFactor(
        factor_id=(
            str(factor_id)
            if factor_id is not None
            else f"measurement:{epoch}:{action.action_id}:{arrival}"
        ),
        kind=kind,
        observation_epoch=int(epoch),
        arrival_epoch=arrival,
        variables=(target_key, nuisance_key),
        jacobians=(normalized_h[:, :6], normalized_h[:, 6:]),
        measurement=measurement,
        covariance=action_factor.covariance,
        participants=participants,
        provenance_ids=(provenance_id or f"capture:{world.world_id}:{epoch}:{action.action_id}",),
        correlation_group=correlation_group,
        robust_kernel=robust_kernel,
        robust_scale=2.5,
        association_probabilities=tuple(float(value) for value in probabilities),
        missing=missing,
        fidelity_level=(
            str(fidelity_level)
            if fidelity_level is not None
            else str(action_factor.metadata.get("model_level", "R0_analytic_factor"))
        ),
        metadata={
            "action_id": action.action_id,
            "measurement_names": action_factor.measurement_names,
            "SI_units": action_factor.metadata.get("measurement_units", []),
            "factor_provenance": "same_likelihood_as_V58_ActionFactor",
            "state_semantics": factor_semantics,
            "communication_snapshot_epoch": int(
                arrival if communication_snapshot_epoch is None else communication_snapshot_epoch
            ),
            **extra_metadata,
        },
    )


def add_epoch_variables_and_transitions(
    graph: DynamicFactorGraph,
    layout: StateLayout,
    epoch: int,
    *,
    target_transition: np.ndarray | None = None,
    target_process_covariance: np.ndarray | None = None,
    nuisance_process_covariance: np.ndarray | None = None,
    target_transition_offset: np.ndarray | None = None,
    nuisance_transition_offset: np.ndarray | None = None,
    anchor_clock_at_epoch_zero: bool = True,
) -> None:
    target = target_variable(
        epoch,
        prior_covariance=np.eye(6) * (1.0 if epoch == 0 else 1e8),
        anchored=epoch == 0,
        prior_enabled=epoch == 0,
    )
    nuisance = nuisance_variable(
        layout,
        epoch,
        anchor_clock=anchor_clock_at_epoch_zero and epoch == 0,
        prior_enabled=epoch == 0,
    )
    graph.add_variable(target)
    graph.add_variable(nuisance)
    if epoch > 0:
        previous_target = VariableKey(VariableKind.TARGET, "primary", epoch - 1)
        previous_nuisance = VariableKey(VariableKind.NUISANCE, "network", epoch - 1)
        transition = np.eye(6) if target_transition is None else np.asarray(target_transition, dtype=float)
        graph.add_transition(
            previous_target,
            target.key,
            transition,
            np.asarray(target_process_covariance if target_process_covariance is not None else np.eye(6) * 0.02, dtype=float),
            transition_offset=target_transition_offset,
        )
        graph.add_transition(
            previous_nuisance,
            nuisance.key,
            np.eye(nuisance.dimension),
            np.asarray(nuisance_process_covariance if nuisance_process_covariance is not None else np.eye(nuisance.dimension) * 0.01, dtype=float),
            transition_offset=nuisance_transition_offset,
        )


def conservative_summary_fusion(
    means: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    *,
    grid_size: int = 101,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    if len(means) != len(covariances) or not means:
        raise ValueError("summary fusion requires equally sized nonempty inputs")
    fused_mean = np.asarray(means[0], dtype=float)
    fused_covariance = np.asarray(covariances[0], dtype=float)
    weights: list[float] = [1.0]
    for mean, covariance in zip(means[1:], covariances[1:]):
        result = covariance_intersection(fused_mean, fused_covariance, np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float), grid_size=grid_size)
        fused_mean = result.mean
        fused_covariance = result.covariance
        weights.append(float(result.weight_first))
    return fused_mean, fused_covariance, tuple(weights)


def naive_double_count_information(factor: DynamicFactor, copies: int = 2) -> np.ndarray:
    """Deliberate negative control; never used by the production solver."""

    if copies < 1:
        raise ValueError("copies must be positive")
    h = np.hstack(factor.jacobians)
    information = platform_matmul(
        platform_matmul(
            h.T,
            platform_symmetric_pinv(factor.covariance, rcond=1e-12),
        ),
        h,
    )
    return float(copies) * information


def graph_semantics_report(graph: DynamicFactorGraph) -> dict[str, Any]:
    communication_edges = sum(len(snapshot.edges) for snapshot in graph.communication.values())
    observation_edges = sum(len(edges) for edges in graph.observation_hypergraph.values())
    provenance_edges = sum(len(factors) for factors in graph.provenance_graph.values())
    return {
        "communication_graph": {
            "snapshot_count": len(graph.communication),
            "edge_count": communication_edges,
            "meaning": "packet/message availability only",
        },
        "observation_hypergraph": {
            "epoch_count": len(graph.observation_hypergraph),
            "hyperedge_count": observation_edges,
            "meaning": "physical participants required by each likelihood factor",
        },
        "statistical_provenance_graph": {
            "root_count": len(graph.provenance_graph),
            "factor_incidence_count": provenance_edges,
            "meaning": "raw-evidence ancestry and correlation control",
        },
        "separate_graph_objects": True,
    }
