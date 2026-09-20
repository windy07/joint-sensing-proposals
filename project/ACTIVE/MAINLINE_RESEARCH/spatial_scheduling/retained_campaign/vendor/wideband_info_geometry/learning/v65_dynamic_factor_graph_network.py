from __future__ import annotations

"""Typed variable/factor/hypergraph amortizer with U(1) transport.

The production role is proposal/amortized inference only.  Candidate factors are
connected to the physical target, nuisance and participating station variables;
incidences carry factor/edge type, time, provenance, uncertainty and masks.
Station pooling is permutation invariant.  Coherent complex messages are
parallel-transported by the registered U(1) edge connection before invariant
readout.  Exact Schur/risk/safety reranking remains final.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wideband_info_geometry.physics_v58.measurement import ActionFactor
from wideband_info_geometry.physics_v58.information import schedule_action_times
from wideband_info_geometry.physics_v58.schema import PhysicalWorld, RadarAction


class GraphVariant(str, Enum):
    DEEPSETS = "deepsets"
    REAL_FACTOR_GRAPH = "real_factor_graph"
    COMPLEX_U1 = "complex_U1_transport_factor_graph"
    BROKEN_EDGE_TYPE = "broken_edge_type_negative_control"
    BROKEN_TRANSPORT = "broken_U1_transport_negative_control"
    SHUFFLED_TOPOLOGY = "shuffled_topology_negative_control"


@dataclass(frozen=True)
class IncidenceEdge:
    variable_index: int
    factor_index: int
    edge_type: int
    timestamp: int
    provenance_hash: float
    uncertainty: float
    mask: float
    transport_phase_rad: float


@dataclass(frozen=True)
class FactorGraphSample:
    sample_id: str
    episode_id: str
    epoch: int
    variable_features: np.ndarray
    factor_features: np.ndarray
    incidences: tuple[IncidenceEdge, ...]
    variable_types: np.ndarray
    factor_types: np.ndarray
    candidate_action_ids: tuple[int, ...]
    candidate_factor_indices: tuple[int, ...]
    exact_utility_labels: np.ndarray
    split: str
    feature_names: tuple[str, ...]
    label_provenance: str = "exact_physics_label_not_input_feature"

    def __post_init__(self) -> None:
        variable = np.asarray(self.variable_features, dtype=float)
        factor = np.asarray(self.factor_features, dtype=float)
        labels = np.asarray(self.exact_utility_labels, dtype=float).reshape(-1)
        if variable.ndim != 2 or factor.ndim != 2:
            raise ValueError("factor graph features must be matrices")
        if labels.size != len(self.candidate_action_ids) or len(self.candidate_factor_indices) != labels.size:
            raise ValueError("factor graph candidate dimensions disagree")
        if any(index < 0 or index >= factor.shape[0] for index in self.candidate_factor_indices):
            raise ValueError("candidate factor index is invalid")
        if np.asarray(self.variable_types).shape != (variable.shape[0],) or np.asarray(self.factor_types).shape != (factor.shape[0],):
            raise ValueError("factor graph type arrays disagree")
        if not np.all(np.isfinite(variable)) or not np.all(np.isfinite(factor)) or not np.all(np.isfinite(labels)):
            raise ValueError("factor graph sample contains non-finite values")
        for edge in self.incidences:
            if not 0 <= edge.variable_index < variable.shape[0] or not 0 <= edge.factor_index < factor.shape[0]:
                raise ValueError("incidence edge index is invalid")
            if not 0.0 <= edge.mask <= 1.0 or edge.uncertainty < 0.0:
                raise ValueError("incidence edge mask/uncertainty is invalid")
        object.__setattr__(self, "variable_features", variable)
        object.__setattr__(self, "factor_features", factor)
        object.__setattr__(self, "variable_types", np.asarray(self.variable_types, dtype=int))
        object.__setattr__(self, "factor_types", np.asarray(self.factor_types, dtype=int))
        object.__setattr__(self, "exact_utility_labels", labels)


@dataclass(frozen=True)
class ProposalResult:
    sample_id: str
    variant: GraphVariant
    scores: np.ndarray
    ranking: tuple[int, ...]
    top_k_action_ids: tuple[int, ...]
    recall_at_k: float
    proposal_regret: float
    post_rerank_regret: float
    exact_reranked_action_id: int
    gauge_equivariance_residual: float | None
    station_permutation_residual: float


@dataclass(frozen=True)
class AmortizerConfig:
    hidden_dim: int = 24
    rounds: int = 3
    ridge: float = 1e-3
    seed: int = 6510
    top_k: int = 3


@dataclass
class FactorGraphAmortizer:
    variant: GraphVariant
    config: AmortizerConfig = field(default_factory=AmortizerConfig)
    readout_weights: np.ndarray | None = None
    feature_dimension: int | None = None
    training_episode_ids: tuple[str, ...] = ()

    @property
    def fitted(self) -> bool:
        return self.readout_weights is not None

    def _random_matrices(self, sample: FactorGraphSample) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        dv = sample.variable_features.shape[1]
        df = sample.factor_features.shape[1]
        h = self.config.hidden_dim
        variable_projection = rng.normal(0.0, 1.0 / math.sqrt(max(dv, 1)), (dv, h))
        factor_projection = rng.normal(0.0, 1.0 / math.sqrt(max(df, 1)), (df, h))
        edge_projection = rng.normal(0.0, 1.0 / math.sqrt(6.0), (6, h))
        update_projection = rng.normal(0.0, 1.0 / math.sqrt(2.0 * h), (2 * h, h))
        return variable_projection, factor_projection, edge_projection, update_projection

    @staticmethod
    def _edge_features(edge: IncidenceEdge, *, broken_edge_type: bool) -> np.ndarray:
        edge_type = 0.0 if broken_edge_type else float(edge.edge_type)
        return np.asarray(
            [
                edge_type / 8.0,
                edge.timestamp / 32.0,
                edge.provenance_hash,
                math.log1p(edge.uncertainty),
                edge.mask,
                math.sin(edge.transport_phase_rad),
            ],
            dtype=float,
        )

    def _embeddings(
        self,
        sample: FactorGraphSample,
        *,
        gauge_phase: np.ndarray | None = None,
        station_permutation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        sample.__post_init__()
        variable_features = sample.variable_features.copy()
        incidences = list(sample.incidences)
        variable_types = sample.variable_types.copy()
        if station_permutation is not None:
            permutation = np.asarray(station_permutation, dtype=int)
            if sorted(permutation.tolist()) != list(range(variable_features.shape[0])):
                raise ValueError("variable permutation must be bijective")
            inverse = np.empty_like(permutation)
            inverse[permutation] = np.arange(permutation.size)
            variable_features = variable_features[permutation]
            variable_types = variable_types[permutation]
            incidences = [
                IncidenceEdge(
                    variable_index=int(inverse[edge.variable_index]),
                    factor_index=edge.factor_index,
                    edge_type=edge.edge_type,
                    timestamp=edge.timestamp,
                    provenance_hash=edge.provenance_hash,
                    uncertainty=edge.uncertainty,
                    mask=edge.mask,
                    transport_phase_rad=edge.transport_phase_rad,
                )
                for edge in incidences
            ]
        vp, fp, ep, up = self._random_matrices(sample)
        variable_hidden = np.tanh(variable_features @ vp)
        factor_hidden = np.tanh(sample.factor_features @ fp)
        complex_variable: np.ndarray | None = None
        if self.variant in {GraphVariant.COMPLEX_U1, GraphVariant.BROKEN_TRANSPORT}:
            phase = np.zeros(variable_hidden.shape[0], dtype=float) if gauge_phase is None else np.asarray(gauge_phase, dtype=float)
            if phase.shape != (variable_hidden.shape[0],):
                raise ValueError("gauge phase dimension disagrees")
            complex_variable = variable_hidden.astype(np.complex128) * np.exp(1j * phase[:, None])
        broken_edge_type = self.variant is GraphVariant.BROKEN_EDGE_TYPE
        shuffled_topology = self.variant is GraphVariant.SHUFFLED_TOPOLOGY
        if shuffled_topology:
            rng = np.random.default_rng(self.config.seed + sample.epoch)
            shuffled_variables = rng.permutation(variable_hidden.shape[0])
            incidences = [
                IncidenceEdge(
                    variable_index=int(shuffled_variables[edge.variable_index]),
                    factor_index=edge.factor_index,
                    edge_type=edge.edge_type,
                    timestamp=edge.timestamp,
                    provenance_hash=edge.provenance_hash,
                    uncertainty=edge.uncertainty,
                    mask=edge.mask,
                    transport_phase_rad=edge.transport_phase_rad,
                )
                for edge in incidences
            ]
        if self.variant is GraphVariant.DEEPSETS:
            pooled = np.r_[np.mean(variable_hidden, axis=0), np.std(variable_hidden, axis=0)]
            factor_hidden = np.tanh(factor_hidden + pooled[: self.config.hidden_dim][None, :])
            return variable_hidden, factor_hidden, complex_variable
        for _ in range(self.config.rounds):
            factor_messages = np.zeros_like(factor_hidden)
            factor_counts = np.zeros((factor_hidden.shape[0], 1))
            complex_factor = np.zeros_like(factor_hidden, dtype=np.complex128) if complex_variable is not None else None
            for edge in incidences:
                if edge.mask <= 0.0:
                    continue
                edge_vector = self._edge_features(edge, broken_edge_type=broken_edge_type)
                # The U(1) connection phase is a gauge-dependent transport, not
                # an ordinary scalar edge feature.  Feeding sin(phi) into a real
                # MLP would silently break equivariance even when complex
                # transport is otherwise correct.  Complex variants therefore
                # use the connection only in the parallel-transport operator.
                if complex_variable is not None:
                    edge_vector[-1] = 0.0
                edge_hidden = np.tanh(edge_vector @ ep)
                attenuation = edge.mask / (1.0 + edge.uncertainty)
                if complex_variable is not None:
                    transport = 1.0 + 0.0j if self.variant is GraphVariant.BROKEN_TRANSPORT else np.exp(1j * edge.transport_phase_rad)
                    assert complex_factor is not None
                    complex_factor[edge.factor_index] += attenuation * transport * complex_variable[edge.variable_index] * (1.0 + 0.1 * edge_hidden)
                else:
                    factor_messages[edge.factor_index] += attenuation * variable_hidden[edge.variable_index] * (1.0 + 0.1 * edge_hidden)
                factor_counts[edge.factor_index] += attenuation
            factor_counts = np.maximum(factor_counts, 1.0)
            if complex_factor is not None:
                # Invariant factor readout retains magnitude and phase-consistent
                # real part after transport.  It is unaffected by a common gauge
                # when edge phases are transformed consistently.
                invariant = np.abs(complex_factor) / factor_counts
                factor_messages = invariant
            else:
                factor_messages /= factor_counts
            factor_hidden = np.tanh(np.c_[factor_hidden, factor_messages] @ up)
            variable_messages = np.zeros_like(variable_hidden)
            variable_counts = np.zeros((variable_hidden.shape[0], 1))
            for edge in incidences:
                if edge.mask <= 0.0:
                    continue
                attenuation = edge.mask / (1.0 + edge.uncertainty)
                variable_messages[edge.variable_index] += attenuation * factor_hidden[edge.factor_index]
                variable_counts[edge.variable_index] += attenuation
            variable_hidden = np.tanh(np.c_[variable_hidden, variable_messages / np.maximum(variable_counts, 1.0)] @ up)
            if complex_variable is not None:
                phase = np.angle(complex_variable[:, :1])
                complex_variable = variable_hidden.astype(np.complex128) * np.exp(1j * phase)
        return variable_hidden, factor_hidden, complex_variable

    def candidate_design(self, sample: FactorGraphSample) -> np.ndarray:
        variable_hidden, factor_hidden, _ = self._embeddings(sample)
        pooled_variable = np.r_[np.mean(variable_hidden, axis=0), np.std(variable_hidden, axis=0)]
        designs = []
        for factor_index in sample.candidate_factor_indices:
            designs.append(np.r_[1.0, factor_hidden[factor_index], pooled_variable, sample.factor_features[factor_index]])
        design = np.vstack(designs)
        if self.feature_dimension is not None and design.shape[1] != self.feature_dimension:
            raise ValueError("amortizer design dimension changed")
        return design

    def fit(self, samples: Sequence[FactorGraphSample]) -> "FactorGraphAmortizer":
        if not samples or any(sample.split != "train" for sample in samples):
            raise ValueError("factor-graph amortizer fitting accepts train samples only")
        design = np.vstack([self.candidate_design(sample) for sample in samples])
        labels = np.concatenate([sample.exact_utility_labels for sample in samples])
        self.feature_dimension = design.shape[1]
        gram = design.T @ design + self.config.ridge * np.eye(design.shape[1])
        self.readout_weights = np.linalg.solve(gram, design.T @ labels)
        self.training_episode_ids = tuple(sorted({sample.episode_id for sample in samples}))
        return self

    def score(self, sample: FactorGraphSample) -> np.ndarray:
        if self.readout_weights is None:
            raise RuntimeError("factor-graph amortizer is not fitted")
        return self.candidate_design(sample) @ self.readout_weights

    def propose(self, sample: FactorGraphSample, *, top_k: int | None = None) -> ProposalResult:
        scores = self.score(sample)
        k = min(int(top_k or self.config.top_k), len(scores))
        ranking = tuple(int(value) for value in np.argsort(-scores))
        proposed_indices = ranking[:k]
        proposed_actions = tuple(sample.candidate_action_ids[index] for index in proposed_indices)
        exact_best_index = int(np.argmax(sample.exact_utility_labels))
        recall = float(exact_best_index in proposed_indices)
        proposal_best = int(proposed_indices[0])
        reranked = max(proposed_indices, key=lambda index: sample.exact_utility_labels[index])
        optimum = float(sample.exact_utility_labels[exact_best_index])
        gauge_residual = self.gauge_equivariance_residual(sample) if self.variant in {GraphVariant.COMPLEX_U1, GraphVariant.BROKEN_TRANSPORT} else None
        permutation = np.arange(sample.variable_features.shape[0])[::-1]
        original = self.candidate_design(sample)
        _, permuted_factor, _ = self._embeddings(sample, station_permutation=permutation)
        # Recreate candidate design with invariant pooled variables under the
        # explicit permutation path.
        permuted_variable, permuted_factor, _ = self._embeddings(sample, station_permutation=permutation)
        pooled = np.r_[np.mean(permuted_variable, axis=0), np.std(permuted_variable, axis=0)]
        permuted_design = np.vstack([
            np.r_[1.0, permuted_factor[index], pooled, sample.factor_features[index]]
            for index in sample.candidate_factor_indices
        ])
        permutation_residual = float(np.linalg.norm(original - permuted_design) / max(float(np.linalg.norm(original)), 1.0))
        return ProposalResult(
            sample_id=sample.sample_id,
            variant=self.variant,
            scores=scores,
            ranking=ranking,
            top_k_action_ids=proposed_actions,
            recall_at_k=recall,
            proposal_regret=optimum - float(sample.exact_utility_labels[proposal_best]),
            post_rerank_regret=optimum - float(sample.exact_utility_labels[reranked]),
            exact_reranked_action_id=sample.candidate_action_ids[reranked],
            gauge_equivariance_residual=gauge_residual,
            station_permutation_residual=permutation_residual,
        )

    def gauge_equivariance_residual(self, sample: FactorGraphSample) -> float:
        if self.variant not in {GraphVariant.COMPLEX_U1, GraphVariant.BROKEN_TRANSPORT}:
            raise ValueError("gauge test applies only to complex variants")
        rng = np.random.default_rng(self.config.seed + 77)
        gauge = rng.uniform(-math.pi, math.pi, sample.variable_features.shape[0])
        # Gauge-transform connection phases phi_ij -> phi_ij - chi_i.  Factor
        # nodes use the fixed reference frame in this bipartite representation.
        transformed_edges = tuple(
            IncidenceEdge(
                variable_index=edge.variable_index,
                factor_index=edge.factor_index,
                edge_type=edge.edge_type,
                timestamp=edge.timestamp,
                provenance_hash=edge.provenance_hash,
                uncertainty=edge.uncertainty,
                mask=edge.mask,
                transport_phase_rad=edge.transport_phase_rad - gauge[edge.variable_index],
            )
            for edge in sample.incidences
        )
        transformed = FactorGraphSample(
            sample_id=sample.sample_id + ":gauge",
            episode_id=sample.episode_id,
            epoch=sample.epoch,
            variable_features=sample.variable_features,
            factor_features=sample.factor_features,
            incidences=transformed_edges,
            variable_types=sample.variable_types,
            factor_types=sample.factor_types,
            candidate_action_ids=sample.candidate_action_ids,
            candidate_factor_indices=sample.candidate_factor_indices,
            exact_utility_labels=sample.exact_utility_labels,
            split=sample.split,
            feature_names=sample.feature_names,
        )
        _, original_factor, _ = self._embeddings(sample, gauge_phase=np.zeros(sample.variable_features.shape[0]))
        _, transformed_factor, _ = self._embeddings(transformed, gauge_phase=gauge)
        return float(np.linalg.norm(original_factor - transformed_factor) / max(float(np.linalg.norm(original_factor)), 1.0))


def _provenance_hash(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def build_action_factor_graph_sample(
    world: PhysicalWorld,
    action_factors: Sequence[tuple[RadarAction, ActionFactor]],
    exact_utilities: Sequence[float],
    *,
    episode_id: str,
    epoch: int,
    split: str,
    station_phase_by_node: Mapping[str, float] | None = None,
) -> FactorGraphSample:
    if len(action_factors) != len(exact_utilities) or not action_factors:
        raise ValueError("action factor graph requires aligned factors/utilities")
    station_phase_by_node = station_phase_by_node or {}
    # Variables: target, nuisance, then physical stations.  The first two carry
    # no arbitrary station ordering; station pooling is permutation invariant.
    variable_features: list[np.ndarray] = [
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    variable_types = [0, 1]
    node_index: dict[str, int] = {}
    role_code = {"active": 0.0, "passive": 0.25, "hybrid": 0.5, "illuminator": 0.75, "calibration": 1.0}
    for node in world.nodes:
        node_index[node.node_id] = len(variable_features)
        variable_features.append(
            np.asarray(
                [
                    0.0,
                    0.0,
                    role_code[node.role],
                    node.position_m[0] / 10_000.0,
                    node.position_m[1] / 10_000.0,
                    node.position_m[2] / 1_000.0,
                    node.clock_bias_std_s / 1e-6,
                    node.frequency_bias_std_hz / 1e3,
                ],
                dtype=float,
            )
        )
        variable_types.append(2)
    factor_features: list[np.ndarray] = []
    factor_types: list[int] = []
    incidences: list[IncidenceEdge] = []
    kind_order = {
        "active_monostatic": 0,
        "active_bistatic": 1,
        "passive_bistatic": 2,
        "tdoa": 3,
        "fdoa": 4,
        "aoa": 5,
        "calibration_time": 6,
        "calibration_frequency": 7,
    }
    names = (
        "factor_type",
        "log_signal_power",
        "log_disturbance_power",
        "detection_probability",
        "association_probability",
        "reference_purity",
        "information_log_trace",
        "information_logdet_proxy",
        "energy_cost",
        "emission_cost",
        "communication_cost",
        "bandwidth",
        "carrier",
        "dwell",
        "sensing_mid_time_fraction",
        "decision_ready_time_fraction",
    )
    for factor_index, (action, factor) in enumerate(action_factors):
        waveform = world.waveform(action.waveform_id)
        action_times, decision_ready_time = schedule_action_times(world, (action,))
        time_scale = max(float(world.max_decision_latency_s), 1.0e-12)
        disturbance = (
            factor.link_budget.thermal_noise_power_w
            + factor.link_budget.receiver_impairment_noise_w
            + factor.link_budget.clutter_power_w
            + factor.link_budget.jammer_power_w
        )
        target_block = factor.observed_information_normalized[:6, :6]
        eigenvalues = np.maximum(np.linalg.eigvalsh(0.5 * (target_block + target_block.T)), 0.0)
        factor_features.append(
            np.asarray(
                [
                    kind_order[action.kind] / 7.0,
                    math.log1p(max(factor.link_budget.signal_power_w, 0.0) * 1e12),
                    math.log1p(max(disturbance, 0.0) * 1e12),
                    factor.link_budget.detection_probability,
                    factor.link_budget.correct_association_probability,
                    factor.link_budget.reference_signal_purity,
                    math.log1p(max(float(np.trace(target_block)), 0.0)),
                    float(np.sum(np.log1p(eigenvalues))),
                    action.energy_cost_j / max(world.max_total_energy_j, 1e-30),
                    action.emission_exposure_j / max(world.max_total_emission_j, 1e-30),
                    action.communication_cost_mb / max(world.max_total_communication_mb, 1e-30),
                    waveform.bandwidth_hz / 1e9,
                    waveform.carrier_hz / 10e9,
                    waveform.dwell_s,
                    action_times[action.action_id] / time_scale,
                    decision_ready_time / time_scale,
                ],
                dtype=float,
            )
        )
        factor_types.append(kind_order[action.kind])
        provenance = f"{episode_id}:{epoch}:{action.action_id}"
        uncertainty = float(np.trace(factor.covariance))
        # Target and nuisance incidences.
        incidences.append(IncidenceEdge(0, factor_index, 0, epoch, _provenance_hash(provenance), uncertainty, 1.0, 0.0))
        incidences.append(IncidenceEdge(1, factor_index, 1, epoch, _provenance_hash(provenance), uncertainty, 1.0, 0.0))
        participants = sorted({node for node in (action.tx_node_id, action.reference_node_id, *action.rx_node_ids) if node})
        for role_index, node_id in enumerate(participants):
            phase = float(station_phase_by_node.get(node_id, 0.0))
            incidences.append(IncidenceEdge(node_index[node_id], factor_index, 2 + role_index, epoch, _provenance_hash(provenance), uncertainty, 1.0, phase))
    return FactorGraphSample(
        sample_id=f"{episode_id}:{epoch}",
        episode_id=episode_id,
        epoch=epoch,
        variable_features=np.vstack(variable_features),
        factor_features=np.vstack(factor_features),
        incidences=tuple(incidences),
        variable_types=np.asarray(variable_types),
        factor_types=np.asarray(factor_types),
        candidate_action_ids=tuple(action.action_id for action, _ in action_factors),
        candidate_factor_indices=tuple(range(len(action_factors))),
        exact_utility_labels=np.asarray(exact_utilities, dtype=float),
        split=split,
        feature_names=names,
    )


def compare_graph_variants(
    train_samples: Sequence[FactorGraphSample],
    test_samples: Sequence[FactorGraphSample],
    *,
    config: AmortizerConfig | None = None,
) -> dict[str, Any]:
    config = config or AmortizerConfig()
    variants = tuple(GraphVariant)
    rows: dict[str, Any] = {}
    for variant in variants:
        model = FactorGraphAmortizer(variant, config).fit(train_samples)
        results = [model.propose(sample) for sample in test_samples]
        rows[variant.value] = {
            "mean_recall_at_k": float(np.mean([result.recall_at_k for result in results])),
            "mean_proposal_regret": float(np.mean([result.proposal_regret for result in results])),
            "mean_post_rerank_regret": float(np.mean([result.post_rerank_regret for result in results])),
            "mean_station_permutation_residual": float(np.mean([result.station_permutation_residual for result in results])),
            "mean_gauge_equivariance_residual": (
                float(np.mean([result.gauge_equivariance_residual for result in results if result.gauge_equivariance_residual is not None]))
                if any(result.gauge_equivariance_residual is not None for result in results)
                else None
            ),
            "training_episode_ids": model.training_episode_ids,
            "role": "proposal_only_exact_physics_rerank_final",
        }
    return rows
