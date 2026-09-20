"""Truth-isolated graph design for differential-clock probes.

The designer uses only controller-visible action definitions, availability and
optional planning-quality scores.  It does not inspect a registered event,
event epoch, realized clock state, observation residual or simulator truth.
The returned actions are still proposal constraints: physical feasibility,
exact graph reranking and the campaign shield remain authoritative.

A registered time-transfer pilot is target independent and therefore remains
observable when a clock error moves a target return outside its acquisition
gate.  Such ``calibration_time`` actions are preferred whenever they exist.
Target-dependent bistatic/TDOA returns are retained only as an explicit
fallback for worlds without a dedicated timing-transfer channel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


CLOCK_SENSITIVE_KINDS = {"active_bistatic", "tdoa", "calibration_time"}
TARGET_INDEPENDENT_CLOCK_PROBE_KINDS = {"calibration_time"}


def clock_edge(action: Any) -> tuple[str, str] | None:
    """Return the undirected node pair in a cross-node timing equation."""

    kind = str(getattr(action, "kind", ""))
    receivers = tuple(str(value) for value in getattr(action, "rx_node_ids", ()))
    pair: tuple[str, str] | None = None
    if kind == "active_bistatic" and receivers:
        transmitter = str(getattr(action, "tx_node_id", ""))
        pair = (transmitter, receivers[0])
    elif kind == "tdoa" and len(receivers) >= 2:
        pair = (receivers[0], receivers[1])
    elif kind == "calibration_time" and receivers:
        reference = str(getattr(action, "reference_node_id", ""))
        pair = (receivers[0], reference)
    if pair is None or not pair[0] or not pair[1] or pair[0] == pair[1]:
        return None
    return tuple(sorted(pair))


def _action_id(action: Any) -> int:
    return int(getattr(action, "action_id"))


def _action_cost(action: Any) -> float:
    values = (
        float(getattr(action, "energy_cost_j", 0.0)),
        float(getattr(action, "emission_exposure_j", 0.0)),
        float(getattr(action, "communication_cost_mb", 0.0)),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("clock probe action has invalid resource cost")
    return float(sum(values))


def _incidence(edges: Sequence[tuple[str, str]], nodes: Sequence[str]) -> np.ndarray:
    index = {str(node): offset for offset, node in enumerate(nodes)}
    rows: list[np.ndarray] = []
    for left, right in edges:
        if left not in index or right not in index:
            raise ValueError("clock edge references a node outside the support")
        row = np.zeros(len(nodes), dtype=float)
        row[index[left]] = -1.0
        row[index[right]] = 1.0
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(nodes)), dtype=float)


def _graph_objective(
    edges: Sequence[tuple[str, str]],
    nodes: Sequence[str],
    weights: Sequence[float] | None = None,
) -> tuple[int, float, float]:
    incidence = _incidence(edges, nodes)
    if not len(edges):
        return 0, 0.0, -math.inf
    if weights is None:
        weight_vector = np.ones(len(edges), dtype=float)
    else:
        weight_vector = np.asarray(weights, dtype=float)
        if weight_vector.shape != (len(edges),):
            raise ValueError("clock-edge weights must match the edge sequence")
        if not np.all(np.isfinite(weight_vector)) or np.any(weight_vector <= 0.0):
            raise ValueError("clock-edge weights must be finite and positive")
    laplacian = incidence.T @ (weight_vector[:, None] * incidence)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    tolerance = max(float(eigenvalues[-1]), 1.0) * 1.0e-10
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(len(positive))
    algebraic_connectivity = (
        float(max(eigenvalues[1], 0.0)) if len(nodes) >= 2 else 0.0
    )
    log_pseudodeterminant = (
        float(np.sum(np.log(positive))) if len(positive) else -math.inf
    )
    return rank, algebraic_connectivity, log_pseudodeterminant


def _pairwise_clock_crlb_summary(
    edges: Sequence[tuple[str, str]],
    nodes: Sequence[str],
    information_weights_s_inv2: Sequence[float],
) -> dict[str, float | None]:
    """Return gauge-invariant pairwise clock-offset bounds in seconds.

    For a timing equation with derivative ``e_i - e_j`` and variance
    ``sigma_tau^2``, the Fisher contribution is
    ``p_usable (e_i-e_j)(e_i-e_j)^T / sigma_tau^2``.  Absolute clock offset
    remains a gauge.  Pairwise differences are estimable when the weighted
    graph has rank ``n-1`` and their covariance lower bound is evaluated with
    the Laplacian Moore-Penrose inverse.
    """

    incidence = _incidence(edges, nodes)
    weights = np.asarray(information_weights_s_inv2, dtype=float)
    if weights.shape != (len(edges),) or np.any(~np.isfinite(weights)) or np.any(
        weights <= 0.0
    ):
        raise ValueError("raw clock information weights must be finite and positive")
    laplacian = incidence.T @ (weights[:, None] * incidence)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    tolerance = max(float(eigenvalues[-1]), 1.0) * 1.0e-10
    positive = eigenvalues > tolerance
    if int(np.count_nonzero(positive)) != max(len(nodes) - 1, 0):
        return {
            "maximum_pairwise_clock_difference_crlb_std_s": None,
            "median_pairwise_clock_difference_crlb_std_s": None,
        }
    inverse_eigenvalues = np.zeros_like(eigenvalues)
    inverse_eigenvalues[positive] = 1.0 / eigenvalues[positive]
    pseudoinverse = (
        eigenvectors * inverse_eigenvalues[None, :]
    ) @ eigenvectors.T
    standard_deviations: list[float] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            contrast = np.zeros(len(nodes), dtype=float)
            contrast[left] = 1.0
            contrast[right] = -1.0
            variance = float(contrast @ pseudoinverse @ contrast)
            standard_deviations.append(math.sqrt(max(variance, 0.0)))
    return {
        "maximum_pairwise_clock_difference_crlb_std_s": max(
            standard_deviations
        ),
        "median_pairwise_clock_difference_crlb_std_s": median(
            standard_deviations
        ),
    }


@dataclass(frozen=True)
class ClockSentinelDesign:
    action_ids: tuple[int, ...]
    edge_sequence: tuple[tuple[str, str], ...]
    supported_node_ids: tuple[str, ...]
    differential_clock_rank: int
    full_supported_subspace_rank: bool
    unweighted_algebraic_connectivity: float
    normalized_information_weights: tuple[float, ...]
    weighted_algebraic_connectivity: float
    information_weights_s_inv2: tuple[float, ...]
    expected_clock_laplacian_algebraic_connectivity_s_inv2: float
    maximum_pairwise_clock_difference_crlb_std_s: float | None
    median_pairwise_clock_difference_crlb_std_s: float | None
    probe_action_kinds: tuple[str, ...]
    target_independent_calibration_available: bool
    target_dependent_fallback_used: bool
    truth_or_event_information_used: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "action_ids": list(self.action_ids),
            "edge_sequence": [list(edge) for edge in self.edge_sequence],
            "supported_node_ids": list(self.supported_node_ids),
            "differential_clock_rank": int(self.differential_clock_rank),
            "full_supported_subspace_rank": bool(
                self.full_supported_subspace_rank
            ),
            "unweighted_algebraic_connectivity": float(
                self.unweighted_algebraic_connectivity
            ),
            "normalized_information_weights": [
                float(value) for value in self.normalized_information_weights
            ],
            "weighted_algebraic_connectivity": float(
                self.weighted_algebraic_connectivity
            ),
            "information_weights_s_inv2": [
                float(value) for value in self.information_weights_s_inv2
            ],
            "expected_clock_laplacian_algebraic_connectivity_s_inv2": float(
                self.expected_clock_laplacian_algebraic_connectivity_s_inv2
            ),
            "maximum_pairwise_clock_difference_crlb_std_s": (
                None
                if self.maximum_pairwise_clock_difference_crlb_std_s is None
                else float(self.maximum_pairwise_clock_difference_crlb_std_s)
            ),
            "median_pairwise_clock_difference_crlb_std_s": (
                None
                if self.median_pairwise_clock_difference_crlb_std_s is None
                else float(self.median_pairwise_clock_difference_crlb_std_s)
            ),
            "probe_action_kinds": list(self.probe_action_kinds),
            "target_independent_calibration_available": bool(
                self.target_independent_calibration_available
            ),
            "target_dependent_fallback_used": bool(
                self.target_dependent_fallback_used
            ),
            "probe_preference": (
                "dedicated_target_independent_time_transfer_before_"
                "target_dependent_return_fallback"
            ),
            "clock_information_model": (
                "expected_independent_gaussian_pairwise_delay_fisher_"
                "p_usable_over_variance"
            ),
            "clock_crlb_scope": (
                "gauge_invariant_pairwise_offset_bound_not_realized_task_error"
            ),
            "truth_or_event_information_used": False,
            "design_objective": (
                "greedy_rank_then_algebraic_connectivity_then_log_"
                "pseudodeterminant_with_planning_quality_and_resource_ties"
            ),
        }


def design_clock_sentinel_sequence(
    actions: Sequence[Any],
    *,
    maximum_unique_edges: int,
    planning_quality_by_action_id: Mapping[int, float] | None = None,
    information_weight_by_action_id: Mapping[int, float] | None = None,
) -> ClockSentinelDesign:
    """Choose a deterministic E-optimal-style sequence of timing edges.

    One representative action is retained per undirected clock edge.  The
    representative maximizes a supplied controller-visible planning quality;
    resource cost and action id break ties.  Edges are then added greedily to
    increase differential-clock rank, algebraic connectivity and the
    Laplacian pseudodeterminant, in that order.
    """

    if maximum_unique_edges < 1:
        raise ValueError("maximum_unique_edges must be positive")
    quality = dict(planning_quality_by_action_id or {})
    information_weight = dict(information_weight_by_action_id or {})
    eligible_actions = [action for action in actions if clock_edge(action) is not None]
    target_independent_actions = [
        action
        for action in eligible_actions
        if str(getattr(action, "kind", ""))
        in TARGET_INDEPENDENT_CLOCK_PROBE_KINDS
    ]
    design_actions = target_independent_actions or eligible_actions
    representatives: dict[tuple[str, str], Any] = {}
    for action in design_actions:
        edge = clock_edge(action)
        if edge is None:
            continue
        action_id = _action_id(action)
        score = float(quality.get(action_id, 0.0))
        edge_weight = float(information_weight.get(action_id, 1.0))
        if not math.isfinite(score):
            raise ValueError("clock probe planning quality must be finite")
        if not math.isfinite(edge_weight) or edge_weight <= 0.0:
            continue
        incumbent = representatives.get(edge)
        if incumbent is None:
            representatives[edge] = action
            continue
        incumbent_id = _action_id(incumbent)
        incumbent_score = float(quality.get(incumbent_id, 0.0))
        incumbent_weight = float(information_weight.get(incumbent_id, 1.0))
        if (edge_weight, score, -_action_cost(action), -action_id) > (
            incumbent_weight,
            incumbent_score,
            -_action_cost(incumbent),
            -incumbent_id,
        ):
            representatives[edge] = action
    if not representatives:
        raise ValueError("no clock-sensitive action is available")

    supported_nodes = tuple(
        sorted({node for edge in representatives for node in edge})
    )
    maximum_weight = max(
        float(information_weight.get(_action_id(action), 1.0))
        for action in representatives.values()
    )
    normalized_weight_by_edge = {
        edge: float(
            information_weight.get(_action_id(action), 1.0) / maximum_weight
        )
        for edge, action in representatives.items()
    }
    remaining = sorted(representatives)
    chosen_edges: list[tuple[str, str]] = []
    chosen_actions: list[Any] = []
    while remaining and len(chosen_edges) < maximum_unique_edges:
        def key(edge: tuple[str, str]) -> tuple[Any, ...]:
            action = representatives[edge]
            action_id = _action_id(action)
            rank, connectivity, log_pdet = _graph_objective(
                chosen_edges + [edge],
                supported_nodes,
                [
                    normalized_weight_by_edge[item]
                    for item in chosen_edges + [edge]
                ],
            )
            return (
                rank,
                connectivity,
                log_pdet,
                float(quality.get(action_id, 0.0)),
                -_action_cost(action),
                -action_id,
            )

        selected_edge = max(remaining, key=key)
        chosen_edges.append(selected_edge)
        chosen_actions.append(representatives[selected_edge])
        remaining.remove(selected_edge)

    normalized_weights = tuple(
        normalized_weight_by_edge[edge] for edge in chosen_edges
    )
    raw_information_weights = tuple(
        float(information_weight.get(_action_id(action), 1.0))
        for action in chosen_actions
    )
    rank, weighted_connectivity, _ = _graph_objective(
        chosen_edges, supported_nodes, normalized_weights
    )
    _, raw_weighted_connectivity, _ = _graph_objective(
        chosen_edges, supported_nodes, raw_information_weights
    )
    _, unweighted_connectivity, _ = _graph_objective(
        chosen_edges, supported_nodes
    )
    pairwise_crlb = _pairwise_clock_crlb_summary(
        chosen_edges,
        supported_nodes,
        raw_information_weights,
    )
    return ClockSentinelDesign(
        action_ids=tuple(_action_id(action) for action in chosen_actions),
        edge_sequence=tuple(chosen_edges),
        supported_node_ids=supported_nodes,
        differential_clock_rank=rank,
        full_supported_subspace_rank=(rank == len(supported_nodes) - 1),
        unweighted_algebraic_connectivity=unweighted_connectivity,
        normalized_information_weights=normalized_weights,
        weighted_algebraic_connectivity=weighted_connectivity,
        information_weights_s_inv2=raw_information_weights,
        expected_clock_laplacian_algebraic_connectivity_s_inv2=(
            raw_weighted_connectivity
        ),
        maximum_pairwise_clock_difference_crlb_std_s=pairwise_crlb[
            "maximum_pairwise_clock_difference_crlb_std_s"
        ],
        median_pairwise_clock_difference_crlb_std_s=pairwise_crlb[
            "median_pairwise_clock_difference_crlb_std_s"
        ],
        probe_action_kinds=tuple(
            str(getattr(action, "kind", "")) for action in chosen_actions
        ),
        target_independent_calibration_available=bool(
            target_independent_actions
        ),
        target_dependent_fallback_used=not bool(target_independent_actions),
    )
