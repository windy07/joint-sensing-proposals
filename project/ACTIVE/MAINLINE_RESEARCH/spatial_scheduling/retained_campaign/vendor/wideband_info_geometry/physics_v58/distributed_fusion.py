from __future__ import annotations

"""Provenance-aware distributed information fusion for radar networks."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .distributed_graph import (
    CommunicationGraph,
    build_communication_graph,
    build_observation_hypergraph,
    communication_observation_contract,
    consensus_iterate,
    topology_report,
)
from .information import (
    prior_information_normalized,
    schur_effective_information,
    stable_inverse_spd,
)
from .measurement import StateLayout, build_action_factor
from .schema import PhysicalWorld, RadarAction


@dataclass(frozen=True)
class CovarianceIntersectionResult:
    covariance: np.ndarray
    mean: np.ndarray | None
    weights: tuple[float, ...]
    objective_logdet: float
    solver: str


def _sym(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    return 0.5 * (value + value.T)


def covariance_intersection(
    covariances: Sequence[np.ndarray],
    means: Sequence[np.ndarray] | None = None,
) -> CovarianceIntersectionResult:
    """Fuse estimates without assuming unavailable cross-covariances.

    For weights on the probability simplex,

        P_CI^{-1} = sum_i w_i P_i^{-1}.

    The log-determinant objective chooses the smallest conservative ellipsoid
    within this family.  CI is intentionally not used when full provenance and
    cross-correlation are available, because exact information fusion is then
    more informative.
    """

    if not covariances:
        raise ValueError("covariance intersection requires at least one covariance")
    matrices = [_sym(np.asarray(value, dtype=float)) for value in covariances]
    shape = matrices[0].shape
    if shape[0] != shape[1] or any(value.shape != shape for value in matrices):
        raise ValueError("all covariance matrices must share one square shape")
    for index, matrix in enumerate(matrices):
        if not np.all(np.isfinite(matrix)) or np.linalg.eigvalsh(matrix)[0] <= 0.0:
            raise ValueError(f"covariance {index} must be finite positive definite")
    if means is not None:
        mean_values = [np.asarray(value, dtype=float).reshape(-1) for value in means]
        if len(mean_values) != len(matrices) or any(value.size != shape[0] for value in mean_values):
            raise ValueError("CI means must match covariance dimensions")
    else:
        mean_values = []
    information = [stable_inverse_spd(value, name=f"ci_covariance_{index}") for index, value in enumerate(matrices)]

    def objective(weights: np.ndarray) -> float:
        fused_information = sum(float(weight) * item for weight, item in zip(weights, information))
        sign, logdet_information = np.linalg.slogdet(_sym(fused_information))
        if sign <= 0.0 or not np.isfinite(logdet_information):
            return float("inf")
        return float(-logdet_information)

    count = len(matrices)
    initial = np.full(count, 1.0 / count, dtype=float)
    solver = "uniform_fallback"
    weights = initial
    if count > 1:
        try:
            from scipy.optimize import minimize

            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * count,
                constraints=[{"type": "eq", "fun": lambda value: float(np.sum(value) - 1.0)}],
                options={"ftol": 1e-12, "maxiter": 400},
            )
            if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
                raise RuntimeError(str(result.message))
            weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
            weights /= max(float(np.sum(weights)), 1e-300)
            solver = "scipy_slsqp_logdet"
        except Exception:
            weights = initial
    fused_information = _sym(sum(float(weight) * item for weight, item in zip(weights, information)))
    covariance = stable_inverse_spd(fused_information, name="covariance_intersection_information")
    mean = None
    if mean_values:
        information_vector = sum(
            float(weight) * info @ value
            for weight, info, value in zip(weights, information, mean_values)
        )
        mean = covariance @ information_vector
    return CovarianceIntersectionResult(
        covariance,
        mean,
        tuple(float(value) for value in weights),
        float(np.linalg.slogdet(covariance)[1]),
        solver,
    )


def action_processing_owner(action: RadarAction) -> str:
    if action.kind.startswith("active_") and action.tx_node_id is not None:
        return str(action.tx_node_id)
    if action.kind.startswith("calibration"):
        return str(action.rx_node_ids[0])
    if action.reference_node_id is not None:
        return str(action.reference_node_id)
    return str(action.rx_node_ids[0])


def _physical_position_covariance(joint_information: np.ndarray, layout: StateLayout) -> np.ndarray:
    effective, _ = schur_effective_information(_sym(joint_information), 3)
    normalized_covariance = stable_inverse_spd(effective, name="distributed_position_information")
    scales = layout.scales()[:3]
    return _sym(np.diag(scales) @ normalized_covariance @ np.diag(scales))


def _minimum_eigenvalue(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(_sym(matrix))[0])


def provenance_aware_information_consensus(
    prior_information: np.ndarray,
    increments_by_node: np.ndarray,
    graph: CommunicationGraph,
    *,
    iterations: int = 80,
) -> dict[str, object]:
    """Average unique factor increments, then restore their global sum.

    The prior is never transmitted as a new measurement.  This is the crucial
    provenance distinction from naively adding local posteriors, which counts
    the common prior once per node.
    """

    increments = np.asarray(increments_by_node, dtype=float)
    if increments.shape[0] != len(graph.node_ids):
        raise ValueError("local increment count must match communication nodes")
    central = _sym(np.asarray(prior_information, dtype=float) + np.sum(increments, axis=0))
    initial = np.asarray([prior_information + len(graph.node_ids) * value for value in increments])
    consensus = consensus_iterate(increments, graph.adjacency, iterations)
    reconstructed = np.asarray([prior_information + len(graph.node_ids) * value for value in consensus])
    denominator = max(float(np.linalg.norm(central)), 1e-300)
    initial_residuals = [float(np.linalg.norm(item - central) / denominator) for item in initial]
    final_residuals = [float(np.linalg.norm(item - central) / denominator) for item in reconstructed]
    return {
        "iterations": int(iterations),
        "initial_max_relative_residual": max(initial_residuals, default=0.0),
        "final_max_relative_residual": max(final_residuals, default=0.0),
        "final_mean_relative_residual": float(np.mean(final_residuals)) if final_residuals else 0.0,
        "converged_to_global_information": max(final_residuals, default=0.0) < 1e-6,
        "prior_transmitted_as_unique_measurement": False,
        "factor_provenance_preserved": True,
    }


def distributed_fusion_report(
    world: PhysicalWorld,
    action_ids: Iterable[int],
    *,
    communication: CommunicationGraph | None = None,
    consensus_iterations: int = 120,
) -> dict[str, object]:
    world.validate()
    graph = communication or build_communication_graph(world)
    graph.validate()
    by_id = {action.action_id: action for action in world.actions}
    ids = tuple(sorted(dict.fromkeys(int(value) for value in action_ids)))
    unknown = sorted(set(ids).difference(by_id))
    if unknown:
        raise ValueError(f"distributed fusion references unknown actions: {unknown}")
    actions = [by_id[action_id] for action_id in ids]
    layout = StateLayout.from_world(world)
    prior = prior_information_normalized(world, layout)
    node_index = {node_id: index for index, node_id in enumerate(graph.node_ids)}
    increments = np.zeros((len(graph.node_ids), layout.dimension, layout.dimension), dtype=float)
    factor_owner: dict[int, str] = {}
    for action in actions:
        owner = action_processing_owner(action)
        factor_owner[action.action_id] = owner
        increments[node_index[owner]] += build_action_factor(world, action).observed_information_normalized
    central_joint = _sym(prior + np.sum(increments, axis=0))
    central_covariance = _physical_position_covariance(central_joint, layout)
    local_covariances = [
        _physical_position_covariance(prior + increment, layout)
        for increment in increments
    ]
    ci = covariance_intersection(local_covariances)
    naive_information = _sym(sum(stable_inverse_spd(value, name=f"naive_local_{index}") for index, value in enumerate(local_covariances)))
    naive_covariance = stable_inverse_spd(naive_information, name="naive_distributed_information")
    consensus = provenance_aware_information_consensus(
        prior,
        increments,
        graph,
        iterations=consensus_iterations,
    )
    observation = build_observation_hypergraph(world, actions)
    operational = communication_observation_contract(graph, observation)
    ci_minus_central = _minimum_eigenvalue(ci.covariance - central_covariance)
    naive_minus_central = _minimum_eigenvalue(naive_covariance - central_covariance)
    central_trace = float(np.trace(central_covariance))
    return {
        "stage": "v64_distributed_factor_graph_fusion",
        "status": "passed" if ci_minus_central >= -1e-7 and not consensus["prior_transmitted_as_unique_measurement"] else "failed_closed",
        "action_ids": list(ids),
        "factor_owner": {str(key): value for key, value in factor_owner.items()},
        "local_estimate_count": len(local_covariances),
        "central_position_covariance_trace_m2": central_trace,
        "covariance_intersection_trace_m2": float(np.trace(ci.covariance)),
        "naive_posterior_sum_trace_m2": float(np.trace(naive_covariance)),
        "ci_to_central_trace_ratio": float(np.trace(ci.covariance) / max(central_trace, 1e-300)),
        "naive_to_central_trace_ratio": float(np.trace(naive_covariance) / max(central_trace, 1e-300)),
        "ci_minus_central_min_eigenvalue_m2": ci_minus_central,
        "naive_minus_central_min_eigenvalue_m2": naive_minus_central,
        "naive_common_prior_double_count_detected": naive_minus_central < -1e-9,
        "covariance_intersection_conservative_against_central": ci_minus_central >= -1e-7,
        "ci_weights": list(ci.weights),
        "ci_solver": ci.solver,
        "provenance_aware_consensus": consensus,
        "communication_topology": topology_report(graph),
        "communication_observation_contract": operational,
        "scientific_interpretation": {
            "known_factor_provenance": "sum unique likelihood increments and count the common prior once",
            "unknown_cross_correlation": "use covariance intersection rather than naive posterior addition",
            "disconnected_network": "global fusion is unavailable until a communication path or delayed store-and-forward exists",
        },
    }
