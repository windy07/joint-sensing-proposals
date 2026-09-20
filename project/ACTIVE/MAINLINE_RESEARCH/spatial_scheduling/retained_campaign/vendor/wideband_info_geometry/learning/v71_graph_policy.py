from __future__ import annotations

"""End-to-end trainable bipartite factor-graph proposal policy.

The neural network learns messages on variable--factor incidence edges. It is
permutation equivariant in station/factor order and never receives exact
utility, oracle actions, hidden target truth, or test statistics as inputs.
The output is proposal-only: the operational dynamic factor graph recomputes
every shortlisted schedule and remains the final selector.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import NormalDist
import time
from typing import Any, Mapping, Sequence

import numpy as np

from wideband_info_geometry.learning.v65_dynamic_factor_graph_network import (
    FactorGraphSample,
    build_action_factor_graph_sample,
)
from wideband_info_geometry.physics_v58.information import (
    action_factor_cache_key,
    evaluate_schedule,
    schedule_action_times,
    schedule_feasibility,
)
from wideband_info_geometry.physics_v58.measurement import (
    build_action_factor,
    world_propagated_to_time,
)
from wideband_info_geometry.physics_v58.schema import PhysicalWorld
from wideband_info_geometry.physics_v58.world import (
    WORLD_GENERATOR_VERSION,
    generate_physical_world,
)


@dataclass(frozen=True)
class V71GraphTrainingConfig:
    hidden_dim: int = 384
    depth: int = 4
    epochs: int = 32
    batch_size: int = 64
    learning_rate: float = 1.5e-3
    weight_decay: float = 1.0e-4
    top_k: int = 11
    listwise_temperature: float = 0.35
    ranking_margin: float = 0.20
    validation_fraction: float = 0.15
    validation_holdout_families: tuple[str, ...] = ("jammed_corridor",)
    calibration_fraction: float = 0.50
    calibration_split_seed: int = 71072
    physics_rank_fusion_min_neural_weight: float = 0.0
    physics_rank_fusion_max_neural_weight: float = 0.50
    physics_rank_fusion_grid_points: int = 5
    calibration_minimum_recall_at_k: float = 0.80
    calibration_minimum_decision_equivalent_wilson_95_lower: float = 0.94
    calibration_maximum_mean_regret: float = 0.02
    calibration_maximum_p99_regret: float = 0.02
    calibration_maximum_exact_evaluation_fraction: float = 0.20
    calibration_confidence: float = 0.95
    calibration_bootstrap_replicates: int = 2000
    minimum_neural_mean_regret_improvement: float = 0.0
    shortlist_tail_cvar_level: float = 0.90
    shortlist_tail_loss_weight: float = 0.25
    seed: int = 71071

    def validate(self) -> None:
        if min(
            self.hidden_dim,
            self.depth,
            self.epochs,
            self.batch_size,
            self.top_k,
        ) <= 0:
            raise ValueError("V71 graph-training counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("V71 graph-training optimizer settings are invalid")
        if self.listwise_temperature <= 0.0 or self.ranking_margin < 0.0:
            raise ValueError("V71 graph-ranking settings are invalid")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("V71 validation_fraction must lie in (0, 0.5)")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("V71 calibration_fraction must lie in (0, 1)")
        if any(not str(family).strip() for family in self.validation_holdout_families):
            raise ValueError("V71 validation holdout families must be nonempty strings")
        if len(set(self.validation_holdout_families)) != len(
            self.validation_holdout_families
        ):
            raise ValueError("V71 validation holdout families must be unique")
        if not (
            0.0
            <= self.physics_rank_fusion_min_neural_weight
            <= self.physics_rank_fusion_max_neural_weight
            <= 1.0
        ):
            raise ValueError("physics-rank fusion bounds must lie in [0, 1]")
        if not math.isclose(
            self.physics_rank_fusion_min_neural_weight, 0.0, abs_tol=1.0e-12
        ):
            raise ValueError("R8.3 calibration must include the physics-only fallback")
        if self.physics_rank_fusion_max_neural_weight > 0.50:
            raise ValueError("R8.3 neural rank-fusion contribution cannot exceed 0.50")
        if self.physics_rank_fusion_grid_points <= 0:
            raise ValueError("physics-rank fusion grid must be nonempty")
        if not 0.0 < self.calibration_minimum_recall_at_k <= 1.0:
            raise ValueError("calibration recall threshold must lie in (0, 1]")
        if not (
            0.0
            < self.calibration_minimum_decision_equivalent_wilson_95_lower
            <= 1.0
        ):
            raise ValueError("calibration Wilson threshold must lie in (0, 1]")
        if min(
            self.calibration_maximum_mean_regret,
            self.calibration_maximum_p99_regret,
            self.calibration_maximum_exact_evaluation_fraction,
        ) < 0.0:
            raise ValueError("calibration risk and compute thresholds must be nonnegative")
        if self.calibration_maximum_exact_evaluation_fraction > 1.0:
            raise ValueError("calibration exact-compute threshold cannot exceed one")
        if not 0.5 < self.calibration_confidence < 1.0:
            raise ValueError("calibration confidence must lie in (0.5, 1)")
        if self.calibration_bootstrap_replicates <= 0:
            raise ValueError("calibration bootstrap count must be positive")
        if self.minimum_neural_mean_regret_improvement < 0.0:
            raise ValueError("minimum neural improvement must be nonnegative")
        if not 0.0 < self.shortlist_tail_cvar_level < 1.0:
            raise ValueError("shortlist tail CVaR level must lie in (0, 1)")
        if not 0.0 <= self.shortlist_tail_loss_weight <= 1.0:
            raise ValueError("shortlist tail loss weight must lie in [0, 1]")


class CalibrationGateError(RuntimeError):
    """Raised before held-out test access when no deployable policy calibrates."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def _torch():
    import torch
    from torch import nn

    return torch, nn


def _model_class():
    torch, nn = _torch()

    class FactorGraphProposalNet(nn.Module):
        """Typed bipartite message passing with graph-conditioned scores."""

        def __init__(
            self,
            variable_dim: int,
            factor_dim: int,
            edge_dim: int,
            hidden_dim: int,
            depth: int,
        ) -> None:
            super().__init__()
            self.variable_encoder = nn.Sequential(
                nn.Linear(variable_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.factor_encoder = nn.Sequential(
                nn.Linear(factor_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.variable_type = nn.Embedding(3, hidden_dim)
            self.factor_type = nn.Embedding(8, hidden_dim)
            self.layers = nn.ModuleList()
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "v_to_f": nn.Sequential(
                                nn.Linear(2 * hidden_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim),
                            ),
                            "factor_update": nn.Sequential(
                                nn.Linear(2 * hidden_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim),
                            ),
                            "f_to_v": nn.Sequential(
                                nn.Linear(2 * hidden_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim),
                            ),
                            "variable_update": nn.Sequential(
                                nn.Linear(2 * hidden_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, hidden_dim),
                            ),
                            "factor_norm": nn.LayerNorm(hidden_dim),
                            "variable_norm": nn.LayerNorm(hidden_dim),
                        }
                    )
                )
            self.score = nn.Sequential(
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        @staticmethod
        def _scatter_mean(values, index, count: int):
            output = values.new_zeros((count, values.shape[-1]))
            output.index_add_(0, index, values)
            denominator = values.new_zeros((count, 1))
            denominator.index_add_(0, index, values.new_ones((values.shape[0], 1)))
            return output / denominator.clamp_min(1.0)

        def forward(
            self,
            variable_features,
            factor_features,
            edge_features,
            variable_types,
            factor_types,
            edge_variable,
            edge_factor,
            variable_graph,
            candidate_factor,
            candidate_graph,
            target_variable,
            nuisance_variable,
            graph_count: int,
        ):
            variable = self.variable_encoder(variable_features) + self.variable_type(
                variable_types
            )
            factor = self.factor_encoder(factor_features) + self.factor_type(factor_types)
            edge = self.edge_encoder(edge_features)
            for layer in self.layers:
                to_factor = layer["v_to_f"](
                    torch.cat((variable[edge_variable], edge), dim=-1)
                )
                factor_message = self._scatter_mean(
                    to_factor, edge_factor, factor.shape[0]
                )
                factor = layer["factor_norm"](
                    factor
                    + layer["factor_update"](
                        torch.cat((factor, factor_message), dim=-1)
                    )
                )
                to_variable = layer["f_to_v"](
                    torch.cat((factor[edge_factor], edge), dim=-1)
                )
                variable_message = self._scatter_mean(
                    to_variable, edge_variable, variable.shape[0]
                )
                variable = layer["variable_norm"](
                    variable
                    + layer["variable_update"](
                        torch.cat((variable, variable_message), dim=-1)
                    )
                )
            graph_context = self._scatter_mean(variable, variable_graph, graph_count)
            candidate_context = torch.cat(
                (
                    factor[candidate_factor],
                    graph_context[candidate_graph],
                    variable[target_variable[candidate_graph]],
                    variable[nuisance_variable[candidate_graph]],
                ),
                dim=-1,
            )
            return self.score(candidate_context).squeeze(-1)

    return FactorGraphProposalNet


def world_action_sample(
    world: PhysicalWorld,
    *,
    episode_id: str,
    epoch: int,
    split: str,
    exact_labels: bool,
) -> FactorGraphSample:
    feasible_actions = tuple(
        action
        for action in world.actions
        if action.allowed and schedule_feasibility(world, (action,))[0]
    )
    if not feasible_actions:
        raise ValueError(
            f"world {world.world_id} has no statically feasible sensing action"
        )
    # Exact singleton labels evaluate each action at its physical sensing
    # mid-time. The proposal features must use the same causal geometry. A
    # current-epoch factor can be arbitrarily stale for a later resource slot.
    sensing_times, _ = schedule_action_times(world, feasible_actions)
    factor_cache: dict[tuple[object, ...], Any] = {}
    factor_rows = []
    for action in feasible_actions:
        sensing_time = sensing_times[action.action_id]
        factor = build_action_factor(
            world_propagated_to_time(world, sensing_time),
            action,
            None,
        )
        factor_cache[
            action_factor_cache_key(
                world,
                action.action_id,
                "nominal",
                sensing_time,
            )
        ] = factor
        factor_rows.append((action, factor))
    factors = tuple(factor_rows)
    labels = (
        tuple(
            float(
                evaluate_schedule(
                    world,
                    (action.action_id,),
                    factor_cache=factor_cache,
                ).robust_utility
            )
            for action, _ in factors
        )
        if exact_labels
        else tuple(0.0 for _ in factors)
    )
    return build_action_factor_graph_sample(
        world,
        factors,
        labels,
        episode_id=episode_id,
        epoch=epoch,
        split=split,
    )


def _sample_task(payload: tuple[Any, ...]) -> tuple[tuple[int, str, int], FactorGraphSample]:
    (
        seed,
        family,
        world_index,
        actions_per_world,
        node_count,
        budget,
        effective_delay_profile,
        time_reference_profile,
        heldout,
    ) = payload
    identity = (int(seed), str(family), int(world_index))
    try:
        world = generate_physical_world(
            int(seed),
            str(family),
            int(world_index),
            actions_per_world=int(actions_per_world),
            node_count=int(node_count),
            budget=int(budget),
            effective_delay_profile=str(effective_delay_profile),
            time_reference_profile=str(time_reference_profile),
        )
        split = "test" if str(family) in heldout else "train"
        sample = world_action_sample(
            world,
            episode_id=f"v71:{seed}:{family}:{world_index}",
            epoch=0,
            split=split,
            exact_labels=True,
        )
    except BaseException as error:
        raise RuntimeError(
            "V71 graph-sample generation failed for "
            f"seed={identity[0]}, family={identity[1]}, world_index={identity[2]}"
        ) from error
    return identity, sample


def generate_graph_training_samples(
    *,
    seeds: Sequence[int],
    families: Sequence[str],
    worlds_per_family: int,
    actions_per_world: int,
    node_count: int,
    budget: int,
    effective_delay_profile: str,
    time_reference_profile: str,
    heldout_families: Sequence[str],
    workers: int = 1,
) -> tuple[list[FactorGraphSample], list[FactorGraphSample]]:
    heldout = tuple(sorted(set(str(value) for value in heldout_families)))
    tasks = [
        (
            int(seed),
            str(family),
            int(world_index),
            int(actions_per_world),
            int(node_count),
            int(budget),
            str(effective_delay_profile),
            str(time_reference_profile),
            heldout,
        )
        for seed in seeds
        for family in families
        for world_index in range(int(worlds_per_family))
    ]
    if int(workers) > 1:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            rows = list(executor.map(_sample_task, tasks, chunksize=1))
    else:
        rows = [_sample_task(task) for task in tasks]
    rows.sort(key=lambda item: item[0])
    train = [sample for _, sample in rows if sample.split == "train"]
    test = [sample for _, sample in rows if sample.split == "test"]
    if not train or not test:
        raise ValueError("V71 graph training requires nonempty train and held-out sets")
    return train, test


def _pack_samples(samples: Sequence[FactorGraphSample]) -> dict[str, Any]:
    variable_features: list[np.ndarray] = []
    factor_features: list[np.ndarray] = []
    edge_features: list[list[float]] = []
    variable_types: list[np.ndarray] = []
    factor_types: list[np.ndarray] = []
    edge_variable: list[int] = []
    edge_factor: list[int] = []
    variable_graph: list[int] = []
    candidate_factor: list[int] = []
    candidate_graph: list[int] = []
    target_variable: list[int] = []
    nuisance_variable: list[int] = []
    labels: list[np.ndarray] = []
    groups: list[slice] = []
    variable_offset = factor_offset = candidate_offset = 0
    for graph_index, sample in enumerate(samples):
        variable_features.append(np.asarray(sample.variable_features, dtype=np.float32))
        factor_features.append(np.asarray(sample.factor_features, dtype=np.float32))
        variable_types.append(np.asarray(sample.variable_types, dtype=np.int64))
        factor_types.append(np.asarray(sample.factor_types, dtype=np.int64))
        variable_graph.extend([graph_index] * sample.variable_features.shape[0])
        target_variable.append(variable_offset)
        nuisance_variable.append(variable_offset + 1)
        for edge in sample.incidences:
            edge_variable.append(variable_offset + edge.variable_index)
            edge_factor.append(factor_offset + edge.factor_index)
            edge_features.append(
                [
                    float(edge.edge_type) / 8.0,
                    math.log1p(max(float(edge.uncertainty), 0.0)),
                    float(edge.mask),
                    float(edge.transport_phase_rad) / math.pi,
                ]
            )
        local_candidates = [
            factor_offset + int(index) for index in sample.candidate_factor_indices
        ]
        candidate_factor.extend(local_candidates)
        candidate_graph.extend([graph_index] * len(local_candidates))
        local_labels = np.asarray(sample.exact_utility_labels, dtype=np.float32)
        labels.append(local_labels)
        groups.append(slice(candidate_offset, candidate_offset + local_labels.size))
        candidate_offset += local_labels.size
        variable_offset += sample.variable_features.shape[0]
        factor_offset += sample.factor_features.shape[0]
    return {
        "variable_features": np.vstack(variable_features),
        "factor_features": np.vstack(factor_features),
        "edge_features": np.asarray(edge_features, dtype=np.float32),
        "variable_types": np.concatenate(variable_types),
        "factor_types": np.concatenate(factor_types),
        "edge_variable": np.asarray(edge_variable, dtype=np.int64),
        "edge_factor": np.asarray(edge_factor, dtype=np.int64),
        "variable_graph": np.asarray(variable_graph, dtype=np.int64),
        "candidate_factor": np.asarray(candidate_factor, dtype=np.int64),
        "candidate_graph": np.asarray(candidate_graph, dtype=np.int64),
        "target_variable": np.asarray(target_variable, dtype=np.int64),
        "nuisance_variable": np.asarray(nuisance_variable, dtype=np.int64),
        "labels": np.concatenate(labels),
        "groups": groups,
        "graph_count": len(samples),
    }


def _feature_statistics(samples: Sequence[FactorGraphSample]) -> dict[str, np.ndarray]:
    packed = _pack_samples(samples)
    result: dict[str, np.ndarray] = {}
    for name in ("variable_features", "factor_features", "edge_features"):
        values = packed[name]
        result[f"{name}_mean"] = values.mean(axis=0).astype(np.float32)
        result[f"{name}_std"] = np.maximum(values.std(axis=0), 1.0e-6).astype(
            np.float32
        )
    return result


def _tensor_batch(
    samples: Sequence[FactorGraphSample],
    statistics: Mapping[str, np.ndarray],
    *,
    device: str,
) -> dict[str, Any]:
    torch, _ = _torch()
    packed = _pack_samples(samples)
    for name in ("variable_features", "factor_features", "edge_features"):
        packed[name] = (
            packed[name] - statistics[f"{name}_mean"]
        ) / statistics[f"{name}_std"]
    for name in ("variable_features", "factor_features", "edge_features", "labels"):
        packed[name] = torch.as_tensor(packed[name], dtype=torch.float32, device=device)
    for name in (
        "variable_types",
        "factor_types",
        "edge_variable",
        "edge_factor",
        "variable_graph",
        "candidate_factor",
        "candidate_graph",
        "target_variable",
        "nuisance_variable",
    ):
        packed[name] = torch.as_tensor(packed[name], dtype=torch.long, device=device)
    return packed


def _forward(model, batch: Mapping[str, Any]):
    return model(
        batch["variable_features"],
        batch["factor_features"],
        batch["edge_features"],
        batch["variable_types"],
        batch["factor_types"],
        batch["edge_variable"],
        batch["edge_factor"],
        batch["variable_graph"],
        batch["candidate_factor"],
        batch["candidate_graph"],
        batch["target_variable"],
        batch["nuisance_variable"],
        int(batch["graph_count"]),
    )


def _ranking_loss(prediction, target, groups: Sequence[slice], cfg):
    torch, _ = _torch()
    regression = torch.nn.functional.smooth_l1_loss(prediction, target)
    widths = tuple(group.stop - group.start for group in groups)
    if widths and len(set(widths)) == 1 and widths[0] > 1:
        width = widths[0]
        local_prediction = prediction.reshape(len(groups), width)
        local_target = target.reshape(len(groups), width)
        target_distribution = torch.softmax(
            local_target / cfg.listwise_temperature,
            dim=1,
        )
        listwise = -(
            target_distribution * torch.log_softmax(local_prediction, dim=1)
        ).sum(dim=1).mean()
        best = torch.argmax(local_target, dim=1)
        best_score = local_prediction.gather(1, best[:, None])
        difference = best_score - local_prediction
        non_best = torch.arange(width, device=prediction.device)[None, :] != best[:, None]
        margin = torch.relu(cfg.ranking_margin - difference)[non_best].mean()
    else:
        listwise_terms = []
        margin_terms = []
        for group in groups:
            local_prediction = prediction[group]
            local_target = target[group]
            target_distribution = torch.softmax(
                local_target / cfg.listwise_temperature,
                dim=0,
            )
            listwise_terms.append(
                -(
                    target_distribution
                    * torch.log_softmax(local_prediction, dim=0)
                ).sum()
            )
            best = torch.argmax(local_target)
            difference = local_prediction[best] - local_prediction
            mask = (
                torch.arange(local_prediction.numel(), device=prediction.device)
                != best
            )
            if mask.any():
                margin_terms.append(
                    torch.relu(cfg.ranking_margin - difference[mask]).mean()
                )
        listwise = torch.stack(listwise_terms).mean()
        margin = (
            torch.stack(margin_terms).mean() if margin_terms else 0.0 * regression
        )
    # The top-k hinge addresses the deployed object directly: whether the true
    # best action survives the bounded exact-evaluation shortlist.  Its severity
    # is measured in registered physical-utility tolerance units.  Empirical
    # upper-tail averaging is a coherent CVaR surrogate over graph episodes.
    shortlist_terms = []
    tolerance = max(float(cfg.calibration_maximum_p99_regret), 1.0e-12)
    for group in groups:
        local_prediction = prediction[group]
        local_target = target[group]
        best = torch.argmax(local_target)
        width = int(local_prediction.numel())
        shortlist_width = min(int(cfg.top_k), width)
        kth_score = torch.topk(local_prediction, shortlist_width).values[-1]
        inclusion_margin = local_prediction[best] - kth_score
        physical_span = torch.clamp(
            local_target[best] - torch.min(local_target), min=0.0
        )
        severity = 1.0 + torch.log1p(physical_span / tolerance)
        shortlist_terms.append(
            torch.relu(cfg.ranking_margin - inclusion_margin) * severity
        )
    if shortlist_terms:
        per_graph_shortlist_loss = torch.stack(shortlist_terms)
        tail_count = max(
            1,
            int(
                math.ceil(
                    (1.0 - float(cfg.shortlist_tail_cvar_level))
                    * per_graph_shortlist_loss.numel()
                )
            ),
        )
        shortlist_tail_cvar = torch.topk(
            per_graph_shortlist_loss, tail_count
        ).values.mean()
    else:
        shortlist_tail_cvar = 0.0 * regression
    base_weight = 1.0 - float(cfg.shortlist_tail_loss_weight)
    loss = base_weight * (
        0.30 * regression + 0.50 * listwise + 0.20 * margin
    ) + float(cfg.shortlist_tail_loss_weight) * shortlist_tail_cvar
    return loss, {
        "regression_loss": regression,
        "listwise_loss": listwise,
        "margin_loss": margin,
        "shortlist_tail_cvar_loss": shortlist_tail_cvar,
    }


def _physics_proxy_scores(sample: FactorGraphSample) -> np.ndarray:
    """Observable information-cost prior used only for rank fusion.

    Columns are fixed by ``build_action_factor_graph_sample``.  The proxy uses
    no exact utility label: it rewards target information, detection,
    association and reference purity while charging normalized mission costs.
    It is deliberately incomplete, leaving geometry/topology interactions to
    the trainable factor graph network and exact graph reranker.
    """

    features = np.asarray(sample.factor_features, dtype=float)[
        np.asarray(sample.candidate_factor_indices, dtype=int)
    ]
    if features.shape[1] not in {14, 16}:
        raise ValueError(
            "V71 physics proxy requires the legacy 14-column or R8.4 "
            "16-column time-aligned schema"
        )
    score = (
        0.35 * features[:, 6]
        + 0.65 * features[:, 7]
        + 0.10 * features[:, 3]
        + 0.08 * features[:, 4]
        + 0.04 * features[:, 5]
        - 0.08 * features[:, 8]
        - 0.12 * features[:, 9]
        - 0.04 * features[:, 10]
    )
    if features.shape[1] == 16:
        score -= 0.06 * features[:, 15]
    return score


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("V71 rank fusion requires finite candidate scores")
    if vector.size <= 1:
        return np.ones_like(vector)
    # Midranks preserve permutation equivariance when candidates have equal
    # scores. Assigning stable but distinct ranks to ties would leak the input
    # row order into the deployed policy.
    order = np.argsort(vector, kind="mergesort")
    sorted_values = vector[order]
    ordinal = np.empty(vector.size, dtype=float)
    start = 0
    while start < vector.size:
        stop = start + 1
        while stop < vector.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ordinal[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return ordinal / float(vector.size - 1)


def _fused_rank_scores(
    neural_scores: np.ndarray,
    samples: Sequence[FactorGraphSample],
    groups: Sequence[slice],
    *,
    neural_weight: float,
) -> np.ndarray:
    if len(samples) != len(groups):
        raise ValueError("V71 rank fusion requires one group per graph")
    fused = np.empty_like(np.asarray(neural_scores, dtype=float))
    for sample, group in zip(samples, groups):
        neural = _rank_percentiles(np.asarray(neural_scores)[group])
        physics = _rank_percentiles(_physics_proxy_scores(sample))
        if neural.size != physics.size:
            raise ValueError("V71 neural and physics candidate domains disagree")
        fused[group] = neural_weight * neural + (1.0 - neural_weight) * physics
    return fused


def _wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 0.0
    proportion = float(successes) / float(trials)
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return float((centre - radius) / denominator)


def _ranking_outcomes(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[slice],
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    recalls: list[float] = []
    top1_hits: list[float] = []
    regrets: list[float] = []
    reranked: list[float] = []
    normalized_regrets: list[float] = []
    normalized_reranked: list[float] = []
    utility_ranges: list[float] = []
    decision_equivalent_hits: list[float] = []
    exact_fractions: list[float] = []
    # This tolerance is inherited from the registered V59--V65 hybrid exact-
    # rerank contract.  Robust utility is expressed consistently as information
    # gain in nats minus dimensionless normalized mission costs, so an absolute
    # indifference band has stable operational meaning across worlds.  In
    # contrast, division by a within-world range is discontinuous as all
    # candidates become decision-equivalent; those relative metrics remain
    # diagnostics below, but are not the sole deployment criterion.
    decision_indifference_tolerance = 0.02
    for group in groups:
        local_scores = scores[group]
        local_labels = labels[group]
        optimum = int(np.argmax(local_labels))
        ranking = np.argsort(-local_scores)
        shortlist = ranking[: min(int(top_k), ranking.size)]
        recalls.append(float(optimum in shortlist))
        top1_hits.append(float(optimum == int(ranking[0])))
        proposal_regret = max(
            float(local_labels[optimum] - local_labels[int(ranking[0])]), 0.0
        )
        rerank_regret = max(
            float(local_labels[optimum] - np.max(local_labels[shortlist])), 0.0
        )
        utility_range = float(np.ptp(local_labels))
        scale = max(utility_range, 1.0e-12)
        regrets.append(proposal_regret)
        reranked.append(rerank_regret)
        normalized_regrets.append(proposal_regret / scale)
        normalized_reranked.append(rerank_regret / scale)
        utility_ranges.append(utility_range)
        decision_equivalent_hits.append(
            float(rerank_regret <= decision_indifference_tolerance)
        )
        exact_fractions.append(float(shortlist.size) / max(float(ranking.size), 1.0))
    return {
        "recall": np.asarray(recalls, dtype=float),
        "top1_hit": np.asarray(top1_hits, dtype=float),
        "proposal_regret": np.asarray(regrets, dtype=float),
        "rerank_regret": np.asarray(reranked, dtype=float),
        "normalized_proposal_regret": np.asarray(normalized_regrets, dtype=float),
        "normalized_rerank_regret": np.asarray(normalized_reranked, dtype=float),
        "utility_range": np.asarray(utility_ranges, dtype=float),
        "decision_equivalent_hit": np.asarray(
            decision_equivalent_hits, dtype=float
        ),
        "exact_fraction": np.asarray(exact_fractions, dtype=float),
        "decision_indifference_tolerance": np.asarray(
            [decision_indifference_tolerance], dtype=float
        ),
    }


def _metrics_from_outcomes(
    outcomes: Mapping[str, np.ndarray],
) -> dict[str, float | int | bool]:
    recalls = np.asarray(outcomes["recall"], dtype=float)
    top1_hits = np.asarray(outcomes["top1_hit"], dtype=float)
    regrets = np.asarray(outcomes["proposal_regret"], dtype=float)
    reranked = np.asarray(outcomes["rerank_regret"], dtype=float)
    normalized_regrets = np.asarray(
        outcomes["normalized_proposal_regret"], dtype=float
    )
    normalized_reranked = np.asarray(
        outcomes["normalized_rerank_regret"], dtype=float
    )
    utility_ranges = np.asarray(outcomes["utility_range"], dtype=float)
    decision_equivalent_hits = np.asarray(
        outcomes["decision_equivalent_hit"], dtype=float
    )
    exact_fractions = np.asarray(outcomes["exact_fraction"], dtype=float)
    if recalls.size <= 0:
        raise ValueError("V71 ranking metrics require at least one graph")
    decision_indifference_tolerance = float(
        np.asarray(outcomes["decision_indifference_tolerance"], dtype=float)[0]
    )
    recall_successes = int(np.sum(recalls))
    return {
        "recall_at_k": float(np.mean(recalls)),
        "recall_at_k_wilson_95_lower": _wilson_lower(
            recall_successes, len(recalls)
        ),
        "recall_successes": recall_successes,
        "world_count": int(len(recalls)),
        "top1_hit_rate": float(np.mean(top1_hits)),
        "proposal_mean_regret": float(np.mean(regrets)),
        "proposal_normalized_mean_regret": float(np.mean(normalized_regrets)),
        "post_exact_rerank_mean_regret": float(np.mean(reranked)),
        "post_exact_rerank_p95_regret": float(np.quantile(reranked, 0.95)),
        "post_exact_rerank_p99_regret": float(np.quantile(reranked, 0.99)),
        "post_exact_rerank_normalized_mean_regret": float(
            np.mean(normalized_reranked)
        ),
        "post_exact_rerank_normalized_p99_regret": float(
            np.quantile(normalized_reranked, 0.99)
        ),
        "post_exact_rerank_decision_equivalent_recall_at_k": float(
            np.mean(decision_equivalent_hits)
        ),
        "post_exact_rerank_decision_equivalent_wilson_95_lower": _wilson_lower(
            int(sum(decision_equivalent_hits)), len(decision_equivalent_hits)
        ),
        "registered_absolute_utility_indifference_tolerance": float(
            decision_indifference_tolerance
        ),
        "candidate_utility_range_median": float(np.median(utility_ranges)),
        "candidate_utility_range_p01": float(np.quantile(utility_ranges, 0.01)),
        "relative_regret_is_diagnostic_not_primary_gate": True,
        "mean_exact_evaluation_fraction": float(np.mean(exact_fractions)),
        "exact_rerank_returns_best_member_of_shortlist": True,
    }


def _ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[slice],
    *,
    top_k: int,
) -> dict[str, float | int | bool]:
    return _metrics_from_outcomes(
        _ranking_outcomes(scores, labels, groups, top_k=top_k)
    )


def _per_family_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[slice],
    samples: Sequence[FactorGraphSample],
    *,
    top_k: int,
) -> dict[str, dict[str, float | int | bool]]:
    if len(samples) != len(groups):
        raise ValueError("V71 family metrics require one sample per score group")
    outcomes = _ranking_outcomes(scores, labels, groups, top_k=top_k)
    family_names = [(_episode_family(sample) or "unidentified") for sample in samples]
    return {
        family: _metrics_from_outcomes(
            _outcome_subset(
                outcomes,
                [
                    index
                    for index, value in enumerate(family_names)
                    if value == family
                ],
            )
        )
        for family in sorted(set(family_names))
    }


def _episode_family(sample: FactorGraphSample) -> str | None:
    """Recover the generated-world family without consulting labels or truth."""

    fields = str(sample.episode_id).split(":")
    if len(fields) >= 4 and fields[0] == "v71" and fields[2]:
        return fields[2]
    return None


def _validation_partition(
    samples: Sequence[FactorGraphSample],
    *,
    fraction: float,
    holdout_families: Sequence[str] = (),
) -> tuple[list[FactorGraphSample], list[FactorGraphSample]]:
    """Create a family-disjoint validation set, with a fixture-only fallback."""

    if len(samples) < 2:
        raise ValueError("V71 requires at least two training-family graphs")
    requested = {str(family) for family in holdout_families}
    family_by_episode = {
        sample.episode_id: _episode_family(sample) for sample in samples
    }
    validation_ids = {
        sample.episode_id
        for sample in samples
        if family_by_episode[sample.episode_id] in requested
    }
    if validation_ids and len(validation_ids) < len(samples):
        fit = [sample for sample in samples if sample.episode_id not in validation_ids]
        validation = [
            replace(sample, split="validation")
            for sample in samples
            if sample.episode_id in validation_ids
        ]
        return fit, validation

    # Generic unit fixtures do not carry generated-world family identities. Keep
    # their split deterministic, but production gates explicitly reject it as a
    # substitute for family-disjoint model selection.
    ranked = sorted(
        samples,
        key=lambda sample: hashlib.sha256(sample.episode_id.encode("utf-8")).digest(),
    )
    validation_count = max(1, int(round(len(ranked) * float(fraction))))
    validation_count = min(validation_count, len(ranked) - 1)
    validation_ids = {sample.episode_id for sample in ranked[:validation_count]}
    fit = [sample for sample in samples if sample.episode_id not in validation_ids]
    validation = [
        replace(sample, split="validation")
        for sample in samples
        if sample.episode_id in validation_ids
    ]
    if not fit or not validation:
        raise RuntimeError("V71 deterministic validation split is empty")
    return fit, validation


def _episode_seed(sample: FactorGraphSample) -> int | None:
    fields = str(sample.episode_id).split(":")
    if len(fields) >= 4 and fields[0] == "v71":
        try:
            return int(fields[1])
        except ValueError:
            return None
    return None


def _selection_calibration_partition(
    samples: Sequence[FactorGraphSample],
    *,
    calibration_fraction: float,
    split_seed: int,
) -> tuple[list[FactorGraphSample], list[FactorGraphSample]]:
    """Split validation seeds before any checkpoint or policy is evaluated.

    All worlds generated from one seed remain in one role.  This prevents a
    common latent scene draw from coupling checkpoint selection to policy
    calibration.  Generic fixtures without generated-world IDs are grouped by
    episode and remain deterministic, but production reports expose that mode.
    """

    if len(samples) < 2:
        raise ValueError("R8.3 requires at least two validation-pool graphs")
    group_by_episode: dict[str, str] = {}
    for sample in samples:
        seed = _episode_seed(sample)
        group_by_episode[sample.episode_id] = (
            f"seed:{seed}" if seed is not None else f"episode:{sample.episode_id}"
        )
    groups = sorted(
        set(group_by_episode.values()),
        key=lambda value: hashlib.sha256(
            f"{int(split_seed)}:{value}".encode("utf-8")
        ).digest(),
    )
    if len(groups) < 2:
        raise ValueError(
            "R8.3 model-selection/calibration split requires at least two "
            "independent seed or episode groups"
        )
    calibration_count = max(1, int(round(len(groups) * calibration_fraction)))
    calibration_count = min(calibration_count, len(groups) - 1)
    calibration_groups = set(groups[:calibration_count])
    model_selection = [
        replace(sample, split="model_selection")
        for sample in samples
        if group_by_episode[sample.episode_id] not in calibration_groups
    ]
    calibration = [
        replace(sample, split="calibration")
        for sample in samples
        if group_by_episode[sample.episode_id] in calibration_groups
    ]
    if not model_selection or not calibration:
        raise RuntimeError("R8.3 model-selection/calibration partition is empty")
    return model_selection, calibration


def _validation_selection_key(
    metrics: Mapping[str, Any],
    *,
    neural_weight: float,
    training_loss: float | None = None,
) -> tuple[float, ...]:
    """Registered lexicographic selection rule in physical utility units."""

    key = (
        float(metrics["post_exact_rerank_decision_equivalent_wilson_95_lower"]),
        float(metrics["post_exact_rerank_decision_equivalent_recall_at_k"]),
        -float(metrics["post_exact_rerank_p99_regret"]),
        -float(metrics["post_exact_rerank_mean_regret"]),
        float(metrics["recall_at_k_wilson_95_lower"]),
        float(metrics["recall_at_k"]),
        -float(metrics["proposal_mean_regret"]),
        -float(neural_weight),
    )
    return key if training_loss is None else key + (-float(training_loss),)


def _checkpoint_selection_key(
    metrics: Mapping[str, Any],
    *,
    training_loss: float,
    cfg: V71GraphTrainingConfig,
) -> tuple[float, ...]:
    """Choose a checkpoint on model-selection data, before policy calibration."""

    feasible = (
        float(metrics["recall_at_k"]) >= cfg.calibration_minimum_recall_at_k
        and float(
            metrics["post_exact_rerank_decision_equivalent_wilson_95_lower"]
        )
        >= cfg.calibration_minimum_decision_equivalent_wilson_95_lower
        and float(metrics["post_exact_rerank_mean_regret"])
        <= cfg.calibration_maximum_mean_regret
        and float(metrics["post_exact_rerank_p99_regret"])
        <= cfg.calibration_maximum_p99_regret
        and float(metrics["mean_exact_evaluation_fraction"])
        <= cfg.calibration_maximum_exact_evaluation_fraction
    )
    return (
        float(feasible),
        -float(metrics["post_exact_rerank_p99_regret"]),
        -float(metrics["post_exact_rerank_mean_regret"]),
        float(metrics["post_exact_rerank_decision_equivalent_wilson_95_lower"]),
        float(metrics["recall_at_k"]),
        -float(training_loss),
    )


def _outcome_subset(
    outcomes: Mapping[str, np.ndarray], indices: Sequence[int]
) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=int)
    return {
        key: (
            np.asarray(value, dtype=float)
            if key == "decision_indifference_tolerance"
            else np.asarray(value, dtype=float)[selected]
        )
        for key, value in outcomes.items()
    }


def _bootstrap_upper_bound(
    values: np.ndarray,
    *,
    statistic: str,
    confidence: float,
    replicates: int,
    seed: int,
) -> float:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size <= 0 or not np.all(np.isfinite(vector)):
        raise ValueError("R8.3 bootstrap requires finite nonempty values")
    generator = np.random.default_rng(int(seed))
    sample_indices = generator.integers(
        0, vector.size, size=(int(replicates), vector.size)
    )
    resamples = vector[sample_indices]
    if statistic == "mean":
        estimates = np.mean(resamples, axis=1)
    elif statistic == "p99":
        estimates = np.quantile(resamples, 0.99, axis=1)
    else:
        raise ValueError(f"unknown bootstrap statistic {statistic!r}")
    return float(np.quantile(estimates, float(confidence)))


def _calibration_assessment(
    outcomes: Mapping[str, np.ndarray],
    samples: Sequence[FactorGraphSample],
    *,
    cfg: V71GraphTrainingConfig,
    simultaneous_comparisons: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = _metrics_from_outcomes(outcomes)
    comparison_count = max(int(simultaneous_comparisons), 1)
    family_names = [(_episode_family(sample) or "unidentified") for sample in samples]
    family_count = max(len(set(family_names)), 1)
    # Each fusion candidate is assessed on the aggregate and every family
    # stratum, with three stochastic bounds (Wilson, mean-regret UCB, and p99
    # UCB). Count all of them in the simultaneous error budget.
    assessment_strata = family_count + 1
    inferential_bounds_per_stratum = 3
    paired_comparison_count = max(comparison_count - 1, 0)
    simultaneous_bound_count = (
        comparison_count * assessment_strata * inferential_bounds_per_stratum
        + paired_comparison_count
    )
    adjusted_confidence = 1.0 - (
        (1.0 - float(cfg.calibration_confidence))
        / float(simultaneous_bound_count)
    )
    adjusted_confidence = min(adjusted_confidence, 1.0 - 1.0e-12)
    z = NormalDist().inv_cdf(1.0 - (1.0 - adjusted_confidence) / 2.0)

    def assess(local: Mapping[str, np.ndarray], seed_offset: int) -> dict[str, Any]:
        local_metrics = _metrics_from_outcomes(local)
        decision_hits = np.asarray(local["decision_equivalent_hit"], dtype=float)
        mean_upper = _bootstrap_upper_bound(
            np.asarray(local["rerank_regret"], dtype=float),
            statistic="mean",
            confidence=adjusted_confidence,
            replicates=cfg.calibration_bootstrap_replicates,
            seed=bootstrap_seed + seed_offset,
        )
        p99_upper = _bootstrap_upper_bound(
            np.asarray(local["rerank_regret"], dtype=float),
            statistic="p99",
            confidence=adjusted_confidence,
            replicates=cfg.calibration_bootstrap_replicates,
            seed=bootstrap_seed + seed_offset + 1,
        )
        simultaneous_wilson = _wilson_lower(
            int(np.sum(decision_hits)), int(decision_hits.size), z=z
        )
        checks = {
            "strict_recall_at_k": float(local_metrics["recall_at_k"])
            >= cfg.calibration_minimum_recall_at_k,
            "simultaneous_decision_equivalent_wilson_lower": simultaneous_wilson
            >= cfg.calibration_minimum_decision_equivalent_wilson_95_lower,
            "bootstrap_mean_regret_upper": mean_upper
            <= cfg.calibration_maximum_mean_regret,
            "bootstrap_p99_regret_upper": p99_upper
            <= cfg.calibration_maximum_p99_regret,
            "exact_compute_fraction": float(
                local_metrics["mean_exact_evaluation_fraction"]
            )
            <= cfg.calibration_maximum_exact_evaluation_fraction,
        }
        return {
            "metrics": local_metrics,
            "simultaneous_confidence": float(adjusted_confidence),
            "simultaneous_wilson_z": float(z),
            "simultaneous_decision_equivalent_wilson_lower": float(
                simultaneous_wilson
            ),
            "bootstrap_mean_regret_upper": float(mean_upper),
            "bootstrap_p99_regret_upper": float(p99_upper),
            "checks": checks,
            "feasible": bool(all(checks.values())),
        }

    aggregate = assess(outcomes, 0)
    per_family: dict[str, Any] = {}
    for family_index, family in enumerate(sorted(set(family_names)), start=1):
        indices = [
            index for index, value in enumerate(family_names) if value == family
        ]
        per_family[family] = assess(
            _outcome_subset(outcomes, indices), 2 * family_index
        )
    family_feasible = bool(
        per_family and all(row["feasible"] for row in per_family.values())
    )
    return {
        "metrics": metrics,
        "aggregate": aggregate,
        "per_family": per_family,
        "all_family_strata_feasible": family_feasible,
        "feasible": bool(aggregate["feasible"] and family_feasible),
        "simultaneous_bound_count": int(simultaneous_bound_count),
        "paired_comparison_count": int(paired_comparison_count),
        "assessment_strata_count": int(assessment_strata),
        "inferential_bounds_per_stratum": int(inferential_bounds_per_stratum),
        "multiplicity_control": (
            "Joint Bonferroni over fusion candidates, aggregate-plus-family "
            "Wilson/mean/p99 bounds, and all neural-versus-physics paired "
            "comparisons"
        ),
    }


def _calibrate_rank_fusion(
    neural_scores: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[slice],
    samples: Sequence[FactorGraphSample],
    *,
    cfg: V71GraphTrainingConfig,
) -> tuple[float, dict[str, Any]]:
    grid = np.linspace(
        cfg.physics_rank_fusion_min_neural_weight,
        cfg.physics_rank_fusion_max_neural_weight,
        cfg.physics_rank_fusion_grid_points,
    )
    if not np.any(np.isclose(grid, 0.0, atol=1.0e-12)):
        raise RuntimeError("R8.3 rank-fusion grid omitted the physics fallback")
    comparison_count = int(grid.size)
    rows: list[dict[str, Any]] = []
    physics_outcomes: dict[str, np.ndarray] | None = None
    for index, alpha in enumerate(grid):
        fused = _fused_rank_scores(
            neural_scores, samples, groups, neural_weight=float(alpha)
        )
        outcomes = _ranking_outcomes(fused, labels, groups, top_k=cfg.top_k)
        if math.isclose(float(alpha), 0.0, abs_tol=1.0e-12):
            physics_outcomes = outcomes
        assessment = _calibration_assessment(
            outcomes,
            samples,
            cfg=cfg,
            simultaneous_comparisons=comparison_count,
            bootstrap_seed=(
                int(cfg.calibration_split_seed) + 1000 * int(index + 1)
            ),
        )
        rows.append(
            {
                "neural_weight": float(alpha),
                "assessment": assessment,
                "outcomes": outcomes,
            }
        )
    if physics_outcomes is None:
        raise RuntimeError("R8.3 physics fallback calibration was not evaluated")

    joint_bound_count = int(rows[0]["assessment"]["simultaneous_bound_count"])
    if any(
        int(row["assessment"]["simultaneous_bound_count"]) != joint_bound_count
        for row in rows
    ):
        raise RuntimeError("R8.3 calibration multiplicity accounting drifted")
    adjusted_confidence = 1.0 - (
        (1.0 - cfg.calibration_confidence) / max(float(joint_bound_count), 1.0)
    )
    physics_regret = np.asarray(physics_outcomes["rerank_regret"], dtype=float)
    for index, row in enumerate(rows):
        alpha = float(row["neural_weight"])
        candidate_regret = np.asarray(
            row["outcomes"]["rerank_regret"], dtype=float
        )
        if math.isclose(alpha, 0.0, abs_tol=1.0e-12):
            delta_upper = 0.0
            improves = False
        else:
            delta_upper = _bootstrap_upper_bound(
                candidate_regret - physics_regret,
                statistic="mean",
                confidence=adjusted_confidence,
                replicates=cfg.calibration_bootstrap_replicates,
                seed=int(cfg.calibration_split_seed) + 50000 + index,
            )
            improves = delta_upper < -float(
                cfg.minimum_neural_mean_regret_improvement
            )
        row["paired_mean_regret_delta_vs_physics_upper"] = float(delta_upper)
        row["statistically_improves_physics"] = bool(improves)

    physics_row = next(
        row
        for row in rows
        if math.isclose(float(row["neural_weight"]), 0.0, abs_tol=1.0e-12)
    )
    serializable_rows = [
        {key: value for key, value in row.items() if key != "outcomes"}
        for row in rows
    ]
    thresholds = {
        "minimum_recall_at_k": cfg.calibration_minimum_recall_at_k,
        "minimum_decision_equivalent_wilson_95_lower": (
            cfg.calibration_minimum_decision_equivalent_wilson_95_lower
        ),
        "maximum_mean_regret": cfg.calibration_maximum_mean_regret,
        "maximum_p99_regret": cfg.calibration_maximum_p99_regret,
        "maximum_exact_evaluation_fraction": (
            cfg.calibration_maximum_exact_evaluation_fraction
        ),
        "confidence": cfg.calibration_confidence,
        "bootstrap_replicates": cfg.calibration_bootstrap_replicates,
        "joint_simultaneous_bound_count": joint_bound_count,
        "joint_adjusted_confidence": float(adjusted_confidence),
    }

    def failure_diagnostics(reason_code: str) -> dict[str, Any]:
        return {
            "schema": "v71_independent_rank_fusion_calibration/v1",
            "status": "failed",
            "reason_code": reason_code,
            "checkpoint_was_frozen_before_calibration": True,
            "heldout_test_metrics_read": False,
            "thresholds": thresholds,
            "candidates": serializable_rows,
            "physics_fallback_feasible": bool(
                physics_row["assessment"]["feasible"]
            ),
        }

    feasible = [row for row in rows if row["assessment"]["feasible"]]
    if not feasible:
        raise CalibrationGateError(
            "R8.3 calibration found no policy satisfying the preregistered "
            "recall, tail-regret, and exact-compute constraints",
            diagnostics=failure_diagnostics("no_candidate_policy_feasible"),
        )
    improved = [
        row
        for row in feasible
        if float(row["neural_weight"]) > 0.0
        and bool(row["statistically_improves_physics"])
    ]
    if improved:
        selected = min(improved, key=lambda row: float(row["neural_weight"]))
        rationale = "smallest_feasible_statistically_certified_neural_increment"
    elif physics_row["assessment"]["feasible"]:
        selected = physics_row
        rationale = "physics_fallback_feasible_no_certified_neural_increment"
    else:
        raise CalibrationGateError(
            "R8.3 physics fallback failed calibration and no feasible neural "
            "increment was statistically certified over it",
            diagnostics=failure_diagnostics(
                "physics_fallback_infeasible_without_certified_neural_increment"
            ),
        )

    report = {
        "schema": "v71_independent_rank_fusion_calibration/v1",
        "status": "passed",
        "selected_neural_weight": float(selected["neural_weight"]),
        "selection_rationale": rationale,
        "neural_increment_certified": bool(
            selected["statistically_improves_physics"]
        ),
        "checkpoint_was_frozen_before_calibration": True,
        "heldout_test_metrics_read": False,
        "thresholds": thresholds,
        "candidates": serializable_rows,
        "selected_assessment": selected["assessment"],
    }
    return float(selected["neural_weight"]), report


def _predict_samples(
    model,
    samples: Sequence[FactorGraphSample],
    statistics: Mapping[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    torch, _ = _torch()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[slice] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _tensor_batch(
                samples[start : start + batch_size], statistics, device=device
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                prediction = _forward(model, batch)
            scores.append(prediction.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            for group in batch["groups"]:
                width = group.stop - group.start
                groups.append(slice(cursor, cursor + width))
                cursor += width
    return np.concatenate(scores), np.concatenate(labels), groups


def train_graph_policy(
    train_samples: Sequence[FactorGraphSample],
    test_samples: Sequence[FactorGraphSample],
    *,
    output: str | Path,
    config: V71GraphTrainingConfig | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    cfg = config or V71GraphTrainingConfig()
    cfg.validate()
    torch, _ = _torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V71 graph training requested CUDA but CUDA is unavailable")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    fit_samples, validation_pool = _validation_partition(
        train_samples,
        fraction=cfg.validation_fraction,
        holdout_families=cfg.validation_holdout_families,
    )
    model_selection_samples, calibration_samples = _selection_calibration_partition(
        validation_pool,
        calibration_fraction=cfg.calibration_fraction,
        split_seed=cfg.calibration_split_seed,
    )
    fit_ids = {sample.episode_id for sample in fit_samples}
    model_selection_ids = {
        sample.episode_id for sample in model_selection_samples
    }
    calibration_ids = {sample.episode_id for sample in calibration_samples}
    test_ids = {sample.episode_id for sample in test_samples}
    model_selection_seeds = sorted(
        {
            seed
            for seed in (_episode_seed(sample) for sample in model_selection_samples)
            if seed is not None
        }
    )
    calibration_seeds = sorted(
        {
            seed
            for seed in (_episode_seed(sample) for sample in calibration_samples)
            if seed is not None
        }
    )
    role_ids = (fit_ids, model_selection_ids, calibration_ids, test_ids)
    if any(
        role_ids[left] & role_ids[right]
        for left in range(len(role_ids))
        for right in range(left + 1, len(role_ids))
    ):
        raise ValueError(
            "R8.3 fit, model-selection, calibration, and OOD-test episodes "
            "must be pairwise disjoint"
        )
    statistics = _feature_statistics(fit_samples)
    probe = _pack_samples(fit_samples[:1])
    Model = _model_class()
    model = Model(
        probe["variable_features"].shape[1],
        probe["factor_features"].shape[1],
        probe["edge_features"].shape[1],
        cfg.hidden_dim,
        cfg.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = np.random.default_rng(cfg.seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    peak_memory = 0
    best_state = None
    best_selection_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_model_selection_metrics: dict[str, Any] = {}
    for epoch in range(cfg.epochs):
        ordering = generator.permutation(len(fit_samples))
        losses: list[dict[str, float]] = []
        model.train()
        for start in range(0, len(ordering), cfg.batch_size):
            indices = ordering[start : start + cfg.batch_size]
            batch = _tensor_batch(
                [fit_samples[int(index)] for index in indices],
                statistics,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                prediction = _forward(model, batch)
                loss, components = _ranking_loss(
                    prediction, batch["labels"], batch["groups"], cfg
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{
                        key: float(value.detach().cpu())
                        for key, value in components.items()
                    },
                }
            )
            if device == "cuda":
                peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        (
            model_selection_scores,
            model_selection_labels,
            model_selection_groups,
        ) = _predict_samples(
            model,
            model_selection_samples,
            statistics,
            device=device,
            batch_size=cfg.batch_size,
        )
        model_selection_metrics = _ranking_metrics(
            model_selection_scores,
            model_selection_labels,
            model_selection_groups,
            top_k=cfg.top_k,
        )
        mean_training_loss = float(np.mean([item["loss"] for item in losses]))
        selection_key = _checkpoint_selection_key(
            model_selection_metrics,
            training_loss=mean_training_loss,
            cfg=cfg,
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch + 1
            best_model_selection_metrics = dict(model_selection_metrics)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": float(epoch + 1),
                "training_loss": mean_training_loss,
                "regression_loss": float(
                    np.mean([item["regression_loss"] for item in losses])
                ),
                "listwise_loss": float(
                    np.mean([item["listwise_loss"] for item in losses])
                ),
                "margin_loss": float(
                    np.mean([item["margin_loss"] for item in losses])
                ),
                "shortlist_tail_cvar_loss": float(
                    np.mean(
                        [item["shortlist_tail_cvar_loss"] for item in losses]
                    )
                ),
                **{
                    f"model_selection_{key}": value
                    for key, value in model_selection_metrics.items()
                },
            }
        )

    if best_state is None:
        raise RuntimeError("R8.3 training did not produce a model-selection checkpoint")
    model.load_state_dict(best_state)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_selection_checkpoint_path = destination.with_suffix(".model_selection.pt")
    model_selection_checkpoint = {
        "schema": "v71_frozen_model_selection_checkpoint/v2",
        "state_dict": model.state_dict(),
        "variable_dim": int(probe["variable_features"].shape[1]),
        "factor_dim": int(probe["factor_features"].shape[1]),
        "edge_dim": int(probe["edge_features"].shape[1]),
        "hidden_dim": int(cfg.hidden_dim),
        "depth": int(cfg.depth),
        "feature_statistics": statistics,
        "factor_feature_names": tuple(fit_samples[0].feature_names),
        "world_generator_version": WORLD_GENERATOR_VERSION,
        "proposal_factor_geometry_time_aligned_with_exact_singleton_labels": True,
        "config": asdict(cfg),
        "selected_epoch": int(best_epoch),
        "model_selection_metrics": best_model_selection_metrics,
        "model_selection_episode_ids": tuple(
            sample.episode_id for sample in model_selection_samples
        ),
        "calibration_or_test_metrics_read": False,
    }
    model_selection_temporary = model_selection_checkpoint_path.with_name(
        f".{model_selection_checkpoint_path.name}.tmp"
    )
    torch.save(model_selection_checkpoint, model_selection_temporary)
    os.replace(model_selection_temporary, model_selection_checkpoint_path)
    model_selection_checkpoint_sha256 = hashlib.sha256(
        model_selection_checkpoint_path.read_bytes()
    ).hexdigest()
    (
        calibration_neural_scores,
        calibration_labels,
        calibration_groups,
    ) = _predict_samples(
        model,
        calibration_samples,
        statistics,
        device=device,
        batch_size=cfg.batch_size,
    )
    best_rank_fusion_alpha, calibration_report = _calibrate_rank_fusion(
        calibration_neural_scores,
        calibration_labels,
        calibration_groups,
        calibration_samples,
        cfg=cfg,
    )
    # Held-out test tensors are evaluated only after the checkpoint and policy
    # calibration are frozen.  No test metric can flow back into either choice.
    test_neural_scores, test_labels, test_groups = _predict_samples(
        model,
        test_samples,
        statistics,
        device=device,
        batch_size=cfg.batch_size,
    )
    test_fused_scores = _fused_rank_scores(
        test_neural_scores,
        test_samples,
        test_groups,
        neural_weight=best_rank_fusion_alpha,
    )
    test_physics_scores = _fused_rank_scores(
        test_neural_scores,
        test_samples,
        test_groups,
        neural_weight=0.0,
    )
    final_metrics = _ranking_metrics(
        test_fused_scores,
        test_labels,
        test_groups,
        top_k=cfg.top_k,
    )
    final_metrics["selected_epoch"] = float(best_epoch)
    neural_only_final_metrics = _ranking_metrics(
        test_neural_scores,
        test_labels,
        test_groups,
        top_k=cfg.top_k,
    )
    physics_only_final_metrics = _ranking_metrics(
        test_physics_scores,
        test_labels,
        test_groups,
        top_k=cfg.top_k,
    )
    heldout_family_metrics = _per_family_ranking_metrics(
        test_fused_scores,
        test_labels,
        test_groups,
        test_samples,
        top_k=cfg.top_k,
    )
    neural_only_heldout_family_metrics = _per_family_ranking_metrics(
        test_neural_scores,
        test_labels,
        test_groups,
        test_samples,
        top_k=cfg.top_k,
    )
    physics_only_heldout_family_metrics = _per_family_ranking_metrics(
        test_physics_scores,
        test_labels,
        test_groups,
        test_samples,
        top_k=cfg.top_k,
    )

    checkpoint = {
        "schema": "v71_trainable_factor_graph_policy_checkpoint/v7",
        "state_dict": model.state_dict(),
        "variable_dim": int(probe["variable_features"].shape[1]),
        "factor_dim": int(probe["factor_features"].shape[1]),
        "edge_dim": int(probe["edge_features"].shape[1]),
        "hidden_dim": int(cfg.hidden_dim),
        "depth": int(cfg.depth),
        "feature_statistics": statistics,
        "factor_feature_names": tuple(fit_samples[0].feature_names),
        "config": asdict(cfg),
        "train_episode_ids": tuple(sample.episode_id for sample in fit_samples),
        "validation_episode_ids": tuple(
            sample.episode_id for sample in model_selection_samples
        )
        + tuple(sample.episode_id for sample in calibration_samples),
        "model_selection_episode_ids": tuple(
            sample.episode_id for sample in model_selection_samples
        ),
        "calibration_episode_ids": tuple(
            sample.episode_id for sample in calibration_samples
        ),
        "heldout_episode_ids": tuple(sample.episode_id for sample in test_samples),
        "role": "proposal_only_exact_dynamic_graph_rerank_final",
        "model_kind": "trainable_bipartite_factor_graph_message_passing",
        "rank_fusion_neural_weight": float(best_rank_fusion_alpha),
        "rank_fusion_calibration": calibration_report,
        "model_selection_checkpoint_sha256": model_selection_checkpoint_sha256,
        "physics_proxy_definition": (
            "rank(0.35*information_log_trace+0.65*information_logdet_proxy+"
            "0.10*Pd+0.08*Passoc+0.04*reference_purity-0.08*energy-"
            "0.12*emission-0.04*communication-0.06*decision_ready_time)"
        ),
        "proposal_factor_geometry_time_aligned_with_exact_singleton_labels": True,
        "proposal_factor_terminal_latency_observable": True,
        "factor_feature_names": list(fit_samples[0].feature_names),
        "world_generator_version": WORLD_GENERATOR_VERSION,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
    report = {
        "schema": "v71_factor_graph_policy_training/v7",
        "status": "passed",
        "device": device,
        "cuda_used": bool(device == "cuda"),
        "mixed_precision": "bfloat16" if device == "cuda" else "disabled",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_cuda_memory_bytes": int(peak_memory),
        "wall_clock_s": float(time.perf_counter() - started),
        "train_graph_count": len(fit_samples),
        "validation_graph_count": len(validation_pool),
        "model_selection_graph_count": len(model_selection_samples),
        "calibration_graph_count": len(calibration_samples),
        "model_selection_seed_count": len(model_selection_seeds),
        "calibration_seed_count": len(calibration_seeds),
        "model_selection_seeds": model_selection_seeds,
        "calibration_seeds": calibration_seeds,
        "heldout_graph_count": len(test_samples),
        "train_candidate_count": int(
            sum(len(sample.candidate_action_ids) for sample in fit_samples)
        ),
        "heldout_candidate_count": int(
            sum(len(sample.candidate_action_ids) for sample in test_samples)
        ),
        "heldout_family_split": True,
        "validation_family_split": bool(
            all(_episode_family(sample) is not None for sample in fit_samples)
            and all(
                _episode_family(sample) is not None
                for sample in validation_pool
            )
            and not (
                {_episode_family(sample) for sample in fit_samples}
                & {_episode_family(sample) for sample in validation_pool}
            )
        ),
        "fit_families": sorted(
            {
                family
                for family in (_episode_family(sample) for sample in fit_samples)
                if family is not None
            }
        ),
        "validation_families": sorted(
            {
                family
                for family in (
                    _episode_family(sample) for sample in validation_pool
                )
                if family is not None
            }
        ),
        "model_selection_families": sorted(
            {
                family
                for family in (
                    _episode_family(sample) for sample in model_selection_samples
                )
                if family is not None
            }
        ),
        "calibration_families": sorted(
            {
                family
                for family in (
                    _episode_family(sample) for sample in calibration_samples
                )
                if family is not None
            }
        ),
        "heldout_test_families": sorted(
            {
                family
                for family in (_episode_family(sample) for sample in test_samples)
                if family is not None
            }
        ),
        "validation_fit_family_overlap": bool(
            {
                family
                for family in (_episode_family(sample) for sample in fit_samples)
                if family is not None
            }
            & {
                family
                for family in (
                    _episode_family(sample) for sample in validation_pool
                )
                if family is not None
            }
        ),
        "training_episode_overlap": False,
        "train_validation_episode_overlap": False,
        "validation_test_episode_overlap": False,
        "model_selection_calibration_episode_overlap": False,
        "calibration_test_episode_overlap": False,
        "model_selection_calibration_seed_overlap": bool(
            {
                seed
                for seed in (_episode_seed(sample) for sample in model_selection_samples)
                if seed is not None
            }
            & {
                seed
                for seed in (_episode_seed(sample) for sample in calibration_samples)
                if seed is not None
            }
        ),
        "model_selection_split": "model_selection",
        "policy_calibration_split": "calibration",
        "checkpoint_selected_on_model_selection_only": True,
        "rank_fusion_selected_on_calibration_only": True,
        "calibration_heldout_test_metrics_read": False,
        "checkpoint_and_policy_frozen_before_test": True,
        "heldout_test_evaluations": 1,
        "selected_epoch": int(best_epoch),
        "training_config": asdict(cfg),
        "training_config_sha256": hashlib.sha256(
            json.dumps(
                asdict(cfg), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "selection_validation_metrics": best_model_selection_metrics,
        "model_selection_metrics": best_model_selection_metrics,
        "rank_fusion_calibration": calibration_report,
        "final_metrics": final_metrics,
        "heldout_family_metrics": heldout_family_metrics,
        "neural_only_final_metrics": neural_only_final_metrics,
        "neural_only_heldout_family_metrics": neural_only_heldout_family_metrics,
        "physics_only_final_metrics": physics_only_final_metrics,
        "physics_only_heldout_family_metrics": (
            physics_only_heldout_family_metrics
        ),
        "deployed_rank_fusion_neural_weight": float(best_rank_fusion_alpha),
        "rank_fusion_selected_on_validation_only": True,
        "model_selection_uses_absolute_utility_not_range_normalized_regret": True,
        "fusion_policy_uses_simultaneous_finite_sample_risk_control": True,
        "fusion_policy_prefers_smallest_certified_neural_increment": True,
        "neural_increment_certified": bool(
            calibration_report["neural_increment_certified"]
        ),
        "physics_proxy_uses_exact_labels_or_test_statistics": False,
        "tail_training_objective": {
            "kind": "empirical_CVaR_of_top_k_inclusion_margin_weighted_by_physical_utility_span",
            "cvar_level": float(cfg.shortlist_tail_cvar_level),
            "loss_weight": float(cfg.shortlist_tail_loss_weight),
            "physical_utility_tolerance": float(
                cfg.calibration_maximum_p99_regret
            ),
        },
        "history": history,
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "model_selection_checkpoint": str(model_selection_checkpoint_path.resolve()),
        "model_selection_checkpoint_sha256": model_selection_checkpoint_sha256,
        "exact_labels_not_input_features": True,
        "trainable_message_passing": True,
        "permutation_equivariant_incidence_aggregation": True,
        "final_selector": "exact_dynamic_factor_graph_counterfactual_rerank",
        "proposal_factor_geometry_time_aligned_with_exact_singleton_labels": True,
        "proposal_factor_terminal_latency_observable": True,
        "factor_feature_names": list(fit_samples[0].feature_names),
        "world_generator_version": WORLD_GENERATOR_VERSION,
    }
    report_path = destination.with_suffix(".training.json")
    report_temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(report_temporary, report_path)
    return report


class FactorGraphPolicyBackend:
    """Lazy checkpoint adapter used by the operational candidate generator."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        torch, _ = _torch()
        self.device = device
        payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        if payload.get("schema") != "v71_trainable_factor_graph_policy_checkpoint/v7":
            raise ValueError("unsupported V71 graph-policy checkpoint")
        if payload.get("world_generator_version") != WORLD_GENERATOR_VERSION:
            raise ValueError(
                "V71 graph-policy checkpoint was trained under a different "
                "world-generator contract"
            )
        if not payload.get(
            "proposal_factor_geometry_time_aligned_with_exact_singleton_labels"
        ):
            raise ValueError(
                "V71 graph-policy checkpoint lacks the time-aligned proposal "
                "geometry contract"
            )
        Model = _model_class()
        self.model = Model(
            int(payload["variable_dim"]),
            int(payload["factor_dim"]),
            int(payload["edge_dim"]),
            int(payload["hidden_dim"]),
            int(payload["depth"]),
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.statistics = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload["feature_statistics"].items()
        }
        self.training_episode_ids = tuple(payload.get("train_episode_ids", ()))
        self.factor_feature_names = tuple(payload.get("factor_feature_names", ()))
        if self.factor_feature_names[-2:] != (
            "sensing_mid_time_fraction",
            "decision_ready_time_fraction",
        ):
            raise ValueError(
                "V71 graph-policy checkpoint lacks the terminal-time feature schema"
            )
        self.rank_fusion_neural_weight = float(
            payload.get("rank_fusion_neural_weight", 1.0)
        )

    def score_actions(
        self,
        world: PhysicalWorld,
        *,
        episode_id: str,
        epoch: int,
    ) -> dict[int, float]:
        # A constrained finite-horizon controller must have a nonempty action
        # correspondence.  The decision layer closes an exhausted feasible set
        # with an emergency no-op; the proposal network therefore has nothing
        # to score in this state.  This is distinct from offering no-op as a
        # normal competing action.
        if not any(
            action.allowed and schedule_feasibility(world, (action,))[0]
            for action in world.actions
        ):
            return {}
        torch, _ = _torch()
        sample = world_action_sample(
            world,
            episode_id=episode_id,
            epoch=epoch,
            split="operational",
            exact_labels=False,
        )
        if tuple(sample.feature_names) != self.factor_feature_names:
            raise ValueError(
                "V71 checkpoint factor-feature contract does not match the "
                "time-aligned operational sample"
            )
        batch = _tensor_batch([sample], self.statistics, device=self.device)
        self.model.eval()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"
        ):
            neural_scores = _forward(self.model, batch).float().cpu().numpy()
        scores = _fused_rank_scores(
            neural_scores,
            (sample,),
            (slice(0, len(sample.candidate_action_ids)),),
            neural_weight=self.rank_fusion_neural_weight,
        )
        return {
            int(action_id): float(score)
            for action_id, score in zip(sample.candidate_action_ids, scores)
        }


def checkpoint_metadata(path: str | Path) -> Mapping[str, Any]:
    torch, _ = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "feature_statistics"}
    }

