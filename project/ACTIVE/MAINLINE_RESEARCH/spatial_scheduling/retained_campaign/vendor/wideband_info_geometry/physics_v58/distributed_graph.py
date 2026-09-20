from __future__ import annotations

"""Communication and observation topology for distributed radar inference.

The communication graph answers whether information can be exchanged before
the decision deadline.  The observation hypergraph answers which physical
nodes jointly define each likelihood factor.  Keeping these structures
separate prevents a common modelling error: a geometrically informative
multistatic factor is not operationally available when its participants cannot
communicate within the planning horizon.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from wideband_info_geometry.small_symmetric_linalg import symmetric_eigh

from .schema import PhysicalWorld, RadarAction


@dataclass(frozen=True)
class CommunicationGraph:
    node_ids: tuple[str, ...]
    adjacency: np.ndarray
    latency_s: np.ndarray
    capacity_mbps: np.ndarray
    reliability: np.ndarray
    range_scale_m: float
    minimum_reliability: float

    def validate(self) -> None:
        n = len(self.node_ids)
        for name, value in (
            ("adjacency", self.adjacency),
            ("latency_s", self.latency_s),
            ("capacity_mbps", self.capacity_mbps),
            ("reliability", self.reliability),
        ):
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite {n} by {n} matrix")
            if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
                raise ValueError(f"{name} must be symmetric")
        if np.any(self.adjacency < 0.0) or np.any(self.reliability < 0.0):
            raise ValueError("graph weights and reliability must be nonnegative")
        if np.any(np.diag(self.adjacency) != 0.0):
            raise ValueError("communication adjacency must have a zero diagonal")


@dataclass(frozen=True)
class ObservationHypergraph:
    node_ids: tuple[str, ...]
    action_ids: tuple[int, ...]
    incidence: np.ndarray
    participant_ids: tuple[tuple[str, ...], ...]

    def validate(self) -> None:
        expected = (len(self.node_ids), len(self.action_ids))
        if self.incidence.shape != expected:
            raise ValueError(f"observation incidence must have shape {expected}")
        if not np.all(np.isin(self.incidence, [0.0, 1.0])):
            raise ValueError("observation incidence must be binary")
        if self.action_ids and np.any(np.sum(self.incidence, axis=0) < 1.0):
            raise ValueError("every observation factor must involve at least one node")


def build_communication_graph(
    world: PhysicalWorld,
    *,
    range_scale_m: float = 3_000.0,
    minimum_reliability: float = 0.25,
    outage_by_node: dict[str, float] | None = None,
) -> CommunicationGraph:
    """Construct a declared-resource communication graph.

    The range law is an explicit engineering transport model rather than an RF
    propagation claim.  Capacity and latency come from the node schema; the
    geometric reliability term creates reproducible link-loss and partition
    stress without silently asserting an ideal all-to-all backhaul.
    """

    if range_scale_m <= 0.0 or not 0.0 <= minimum_reliability < 1.0:
        raise ValueError("invalid communication range or reliability threshold")
    nodes = tuple(world.nodes)
    node_ids = tuple(node.node_id for node in nodes)
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=float)
    latency = np.zeros((n, n), dtype=float)
    capacity = np.zeros((n, n), dtype=float)
    reliability = np.eye(n, dtype=float)
    outages = {str(key): float(value) for key, value in (outage_by_node or {}).items()}
    max_capacity = max((node.communication_capacity_mbps for node in nodes), default=1.0)
    for i, left in enumerate(nodes):
        for j in range(i + 1, n):
            right = nodes[j]
            distance = float(np.linalg.norm(np.asarray(left.position_m) - np.asarray(right.position_m)))
            geometric = math.exp(-((distance / range_scale_m) ** 2))
            node_survival = (1.0 - outages.get(left.node_id, 0.0)) * (1.0 - outages.get(right.node_id, 0.0))
            link_reliability = float(np.clip(geometric * node_survival, 0.0, 1.0))
            link_capacity = min(left.communication_capacity_mbps, right.communication_capacity_mbps)
            link_latency = left.latency_s + right.latency_s + distance / 299_792_458.0
            reliability[i, j] = reliability[j, i] = link_reliability
            capacity[i, j] = capacity[j, i] = link_capacity
            latency[i, j] = latency[j, i] = link_latency
            if link_reliability >= minimum_reliability and link_latency <= world.max_decision_latency_s:
                capacity_factor = link_capacity / max(max_capacity, 1e-12)
                latency_factor = 1.0 / (1.0 + link_latency / max(world.max_decision_latency_s, 1e-12))
                adjacency[i, j] = adjacency[j, i] = link_reliability * capacity_factor * latency_factor
    graph = CommunicationGraph(
        node_ids,
        adjacency,
        latency,
        capacity,
        reliability,
        float(range_scale_m),
        float(minimum_reliability),
    )
    graph.validate()
    return graph


def action_participants(action: RadarAction) -> tuple[str, ...]:
    values = [action.tx_node_id, action.reference_node_id, *action.rx_node_ids]
    return tuple(sorted({str(value) for value in values if value is not None}))


def build_observation_hypergraph(
    world: PhysicalWorld,
    actions: Iterable[RadarAction] | None = None,
) -> ObservationHypergraph:
    selected = tuple(actions if actions is not None else world.actions)
    node_ids = tuple(node.node_id for node in world.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    incidence = np.zeros((len(node_ids), len(selected)), dtype=float)
    participants: list[tuple[str, ...]] = []
    for column, action in enumerate(selected):
        ids = action_participants(action)
        participants.append(ids)
        for node_id in ids:
            incidence[node_index[node_id], column] = 1.0
    graph = ObservationHypergraph(
        node_ids,
        tuple(action.action_id for action in selected),
        incidence,
        tuple(participants),
    )
    graph.validate()
    return graph


def connected_components(adjacency: np.ndarray, *, tolerance: float = 0.0) -> tuple[tuple[int, ...], ...]:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    unseen = set(range(matrix.shape[0]))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            neighbours = np.flatnonzero(matrix[current] > tolerance)
            stack.extend(int(value) for value in neighbours if int(value) not in component)
        unseen.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def weighted_laplacian(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=float)
    return np.diag(np.sum(matrix, axis=1)) - matrix


def metropolis_consensus_matrix(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=float)
    n = matrix.shape[0]
    binary_degree = np.sum(matrix > 0.0, axis=1)
    weights = np.zeros_like(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] <= 0.0:
                continue
            value = 1.0 / (1.0 + max(binary_degree[i], binary_degree[j]))
            weights[i, j] = weights[j, i] = value
    np.fill_diagonal(weights, 1.0 - np.sum(weights, axis=1))
    return weights


def consensus_iterate(values: np.ndarray, adjacency: np.ndarray, iterations: int) -> np.ndarray:
    if iterations < 0:
        raise ValueError("consensus iteration count must be nonnegative")
    state = np.asarray(values, dtype=float).copy()
    if state.shape[0] != np.asarray(adjacency).shape[0]:
        raise ValueError("the first value dimension must match graph nodes")
    weights = metropolis_consensus_matrix(adjacency)
    for _ in range(iterations):
        state = np.tensordot(weights, state, axes=(1, 0))
    return state


def _bridge_count(adjacency: np.ndarray) -> int:
    matrix = np.asarray(adjacency, dtype=float)
    baseline = len(connected_components(matrix))
    count = 0
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            if matrix[i, j] <= 0.0:
                continue
            modified = matrix.copy()
            modified[i, j] = modified[j, i] = 0.0
            count += int(len(connected_components(modified)) > baseline)
    return count


def topology_report(graph: CommunicationGraph) -> dict[str, object]:
    graph.validate()
    adjacency = graph.adjacency
    n = len(graph.node_ids)
    components = connected_components(adjacency)
    laplacian = weighted_laplacian(adjacency)
    eigenvalues = symmetric_eigh(0.5 * (laplacian + laplacian.T))[0] if n else np.zeros(0)
    algebraic_connectivity = float(eigenvalues[1]) if n > 1 else 0.0
    edge_count = int(np.count_nonzero(np.triu(adjacency > 0.0, 1)))
    cycle_rank = edge_count - n + len(components)
    consensus = metropolis_consensus_matrix(adjacency)
    consensus_eigenvalues = np.sort(
        np.abs(symmetric_eigh(0.5 * (consensus + consensus.T))[0])
    )
    contraction = float(consensus_eigenvalues[-2]) if n > 1 else 0.0
    return {
        "node_count": n,
        "edge_count": edge_count,
        "component_count": len(components),
        "components": [[graph.node_ids[index] for index in component] for component in components],
        "betti_0": len(components),
        "cycle_rank_betti_1": int(cycle_rank),
        "bridge_edge_count": _bridge_count(adjacency),
        "algebraic_connectivity": algebraic_connectivity,
        "consensus_contraction_factor": contraction,
        "connected": len(components) == 1,
        "global_phase_or_consensus_identifiable": len(components) == 1,
    }


def communication_observation_contract(
    communication: CommunicationGraph,
    observation: ObservationHypergraph,
) -> dict[str, object]:
    communication.validate()
    observation.validate()
    if communication.node_ids != observation.node_ids:
        raise ValueError("communication and observation graphs must use the same node order")
    components = connected_components(communication.adjacency)
    component_by_node = {
        communication.node_ids[index]: component_index
        for component_index, component in enumerate(components)
        for index in component
    }
    disconnected_actions: list[int] = []
    for action_id, participants in zip(observation.action_ids, observation.participant_ids):
        if len({component_by_node[node_id] for node_id in participants}) > 1:
            disconnected_actions.append(int(action_id))
    topology = topology_report(communication)
    return {
        "communication_topology": topology,
        "observation_factor_count": len(observation.action_ids),
        "observation_hyperedge_orders": [len(value) for value in observation.participant_ids],
        "higher_order_observation_factor_count": sum(len(value) > 2 for value in observation.participant_ids),
        "cross_partition_action_ids": disconnected_actions,
        "cross_partition_factor_count": len(disconnected_actions),
        "all_observation_factors_operationally_fusable": len(disconnected_actions) == 0,
        "contract": "communication_graph_and_observation_hypergraph_are_distinct",
    }


def induced_schedule_graph_features(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    communication: CommunicationGraph | None = None,
) -> dict[str, float | int | bool]:
    graph = communication or build_communication_graph(world)
    selected_nodes = sorted({node_id for action in actions for node_id in action_participants(action)})
    index = {node_id: i for i, node_id in enumerate(graph.node_ids)}
    indices = [index[node_id] for node_id in selected_nodes]
    induced = graph.adjacency[np.ix_(indices, indices)] if indices else np.zeros((0, 0))
    components = connected_components(induced) if indices else ()
    positive_edges = induced[np.triu_indices_from(induced, 1)] if indices else np.zeros(0)
    positive_edges = positive_edges[positive_edges > 0.0]
    return {
        "agent_node_count": len(selected_nodes),
        "agent_component_count": len(components),
        "agent_network_connected": bool(selected_nodes and len(components) == 1),
        "agent_link_weight_min": float(np.min(positive_edges)) if positive_edges.size else 0.0,
        "agent_link_weight_mean": float(np.mean(positive_edges)) if positive_edges.size else 0.0,
        "observation_hyperedge_order_mean": float(np.mean([len(action_participants(action)) for action in actions])) if actions else 0.0,
    }


def temporal_communication_resilience_report(
    world: PhysicalWorld,
    actions: Sequence[RadarAction],
    *,
    failed_node_id: str | None = None,
    range_scale_m: float = 3_000.0,
    minimum_reliability: float = 0.25,
) -> dict[str, object]:
    """Evaluate an abrupt node outage and subsequent network recovery.

    This is a topology/availability experiment.  It does not pretend that a
    disconnected multistatic likelihood was observed: any factor whose
    participants span components is explicitly blocked during the outage.
    """

    if not world.nodes:
        raise ValueError("temporal communication resilience requires network nodes")
    failed = failed_node_id or max(
        world.nodes,
        key=lambda node: (node.communication_capacity_mbps, node.node_id),
    ).node_id
    if failed not in {node.node_id for node in world.nodes}:
        raise ValueError("failed node is not present in the world")
    observation = build_observation_hypergraph(world, actions)
    graph_kwargs = {
        "range_scale_m": float(range_scale_m),
        "minimum_reliability": float(minimum_reliability),
    }
    healthy = build_communication_graph(world, **graph_kwargs)
    outage = build_communication_graph(world, outage_by_node={failed: 1.0}, **graph_kwargs)
    recovered = build_communication_graph(world, **graph_kwargs)
    phases = []
    for epoch, name, graph in (
        (0, "healthy", healthy),
        (1, "abrupt_node_outage", outage),
        (2, "recovered", recovered),
    ):
        contract = communication_observation_contract(graph, observation)
        phases.append({
            "epoch": epoch,
            "network_mode": name,
            "topology": topology_report(graph),
            "cross_partition_action_ids": contract["cross_partition_action_ids"],
            "operationally_fusable_action_count": len(observation.action_ids)
            - int(contract["cross_partition_factor_count"]),
            "all_observation_factors_operationally_fusable": contract[
                "all_observation_factors_operationally_fusable"
            ],
        })
    healthy_components = int(phases[0]["topology"]["component_count"])
    outage_components = int(phases[1]["topology"]["component_count"])
    recovery_matches = phases[0]["topology"] == phases[2]["topology"]
    blocked = list(phases[1]["cross_partition_action_ids"])
    return {
        "stage": "v64_temporal_communication_resilience",
        "status": "passed"
        if outage_components > healthy_components and bool(blocked) and recovery_matches
        else "failed_closed",
        "failed_node_id": failed,
        "timeline": phases,
        "abrupt_outage_changes_topology": outage_components > healthy_components,
        "cross_partition_factors_blocked_during_outage": bool(blocked),
        "network_recovery_restores_declared_topology": recovery_matches,
        "operational_contract": "unavailable_cross_partition_factors_are_not_fused",
    }
