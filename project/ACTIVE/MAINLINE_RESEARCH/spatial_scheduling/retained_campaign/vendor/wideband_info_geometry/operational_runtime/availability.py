from __future__ import annotations

"""Operational communication availability for the unified runtime."""

from dataclasses import dataclass, replace
import heapq
import math
from typing import Iterable, Mapping

import numpy as np

from wideband_info_geometry.physics_v58.distributed_graph import (
    CommunicationGraph,
    action_participants,
    build_communication_graph,
    build_observation_hypergraph,
    communication_observation_contract,
    connected_components,
    topology_report,
)
from wideband_info_geometry.physics_v58.dynamic_factor_graph import CommunicationSnapshot
from wideband_info_geometry.physics_v58.schema import PhysicalWorld, RadarAction

from .contracts import AvailabilityRecord, OperationalNetworkState, RuntimeConfig


def choose_fusion_node(world: PhysicalWorld, requested: str | None = None) -> str:
    node_ids = {node.node_id for node in world.nodes}
    if requested is not None:
        if requested not in node_ids:
            raise ValueError(f"configured fusion node {requested!r} is not present")
        return requested
    for role in ("hybrid", "calibration", "active", "passive", "illuminator"):
        match = next((node.node_id for node in world.nodes if node.role == role), None)
        if match is not None:
            return match
    raise ValueError("world has no radar nodes")


def communication_graph_from_episode(
    world: PhysicalWorld,
    state: OperationalNetworkState,
) -> CommunicationGraph:
    """Intersect the engineering link graph with the realized episode topology."""

    base = build_communication_graph(world)
    permitted = {
        tuple(sorted((str(left), str(right))))
        for left, right in state.communication_edges
    }
    adjacency = np.asarray(base.adjacency, dtype=float).copy()
    for i, left in enumerate(base.node_ids):
        for j in range(i + 1, len(base.node_ids)):
            if tuple(sorted((left, base.node_ids[j]))) not in permitted:
                adjacency[i, j] = adjacency[j, i] = 0.0
    graph = replace(base, adjacency=adjacency)
    graph.validate()
    return graph


@dataclass(frozen=True)
class TransportPath:
    """One store-and-forward route for a declared data product.

    ``payload_mb`` is the action product generated at ``source``.  On every
    traversed link the serialization delay is ``8 payload_mb / C_mbps``; link
    engineering latency is then added.  This gives the causal arrival time of a
    finite packet rather than the zero-payload shortest-path latency.
    """

    source: str
    destination: str
    node_path: tuple[str, ...]
    edge_path: tuple[tuple[str, str], ...]
    payload_mb: float
    engineering_latency_s: float
    serialization_latency_s: float
    total_latency_s: float
    bottleneck_capacity_mbps: float
    background_latency_s: float = 0.0


def _shortest_transport_path(
    graph: CommunicationGraph,
    source: str,
    destination: str,
    *,
    payload_mb: float = 0.0,
    background_delay_s_by_node: Mapping[str, float] | None = None,
) -> TransportPath | None:
    r"""Return the minimum finite-payload store-and-forward route.

    For path :math:`p` and payload :math:`M`, the route metric is

    .. math::

       T_p(M)=\sum_{e\in p}\left(\ell_e+\frac{8M}{C_e}\right).

    This is exact for sequential store-and-forward transmission of one action
    product and is a conservative upper bound when cut-through forwarding is
    available.  A disconnected pair returns ``None``.
    """

    payload = float(payload_mb)
    background = {
        str(node_id): float(value)
        for node_id, value in (background_delay_s_by_node or {}).items()
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in background.values()
    ):
        raise ValueError("background transport delay must be finite and nonnegative")
    if not math.isfinite(payload) or payload < 0.0:
        raise ValueError("communication payload must be finite and nonnegative")
    if source == destination:
        node = next(index for index, value in enumerate(graph.node_ids) if value == source)
        local_capacity = max(float(np.max(graph.capacity_mbps[node])), 1e12)
        return TransportPath(
            source,
            destination,
            (source,),
            (),
            payload,
            0.0,
            0.0,
            0.0,
            local_capacity,
            0.0,
        )
    index = {node_id: i for i, node_id in enumerate(graph.node_ids)}
    if source not in index or destination not in index:
        return None
    start = index[source]
    goal = index[destination]
    distances = [math.inf] * len(graph.node_ids)
    bottlenecks = [0.0] * len(graph.node_ids)
    predecessors: list[int | None] = [None] * len(graph.node_ids)
    distances[start] = 0.0
    bottlenecks[start] = math.inf
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node] + 1e-15:
            continue
        if node == goal:
            reverse_indices = [goal]
            cursor = goal
            while cursor != start:
                previous = predecessors[cursor]
                if previous is None:
                    return None
                reverse_indices.append(previous)
                cursor = previous
            route_indices = tuple(reversed(reverse_indices))
            node_path = tuple(graph.node_ids[value] for value in route_indices)
            edge_path = tuple(
                tuple(sorted((node_path[position], node_path[position + 1])))
                for position in range(len(node_path) - 1)
            )
            engineering = float(
                sum(
                    graph.latency_s[route_indices[position], route_indices[position + 1]]
                    for position in range(len(route_indices) - 1)
                )
            )
            serialization = float(
                sum(
                    8.0
                    * payload
                    / max(
                        float(
                            graph.capacity_mbps[
                                route_indices[position], route_indices[position + 1]
                            ]
                        ),
                        1e-30,
                    )
                    for position in range(len(route_indices) - 1)
                )
            )
            background_latency = float(
                sum(
                    background.get(node_path[position], 0.0)
                    for position in range(len(node_path) - 1)
                )
            )
            return TransportPath(
                source,
                destination,
                node_path,
                edge_path,
                payload,
                engineering,
                serialization,
                engineering + serialization + background_latency,
                float(bottlenecks[node]),
                background_latency,
            )
        for neighbour in np.flatnonzero(graph.adjacency[node] > 0.0):
            neighbour = int(neighbour)
            capacity = float(graph.capacity_mbps[node, neighbour])
            candidate = (
                distance
                + float(graph.latency_s[node, neighbour])
                + 8.0 * payload / max(capacity, 1e-30)
                + background.get(graph.node_ids[node], 0.0)
            )
            bottleneck = min(
                bottlenecks[node],
                capacity,
            )
            if candidate < distances[neighbour] - 1e-15 or (
                abs(candidate - distances[neighbour]) <= 1e-15
                and bottleneck > bottlenecks[neighbour]
            ):
                distances[neighbour] = candidate
                bottlenecks[neighbour] = bottleneck
                predecessors[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return None


def _shortest_transport(
    graph: CommunicationGraph,
    source: str,
    destination: str,
) -> tuple[float, float]:
    """Backward-compatible zero-payload shortest-transport query."""

    route = _shortest_transport_path(
        graph, source, destination, payload_mb=0.0
    )
    if route is None:
        return math.inf, 0.0
    return route.total_latency_s, route.bottleneck_capacity_mbps


class DistributedAvailabilityManager:
    """Make communication reachability an explicit precondition for inference."""

    def __init__(self, base_world: PhysicalWorld, config: RuntimeConfig) -> None:
        config.validate()
        self.base_world = base_world
        self.config = config
        self.fusion_node_id = choose_fusion_node(base_world, config.fusion_node_id)
        self._latest_graph: CommunicationGraph | None = None
        self._latest_record: AvailabilityRecord | None = None
        self._latest_routes_by_action: dict[int, tuple[TransportPath, ...]] = {}
        self._background_delay_s_by_node: dict[str, float] = {}
        self._node_transport_telemetry: dict[str, dict[str, float]] = {}
        self._committed_schedule_epoch: int | None = None
        self._committed_arrival_epoch_by_action: dict[int, int] = {}
        self._committed_transport_latency_by_action: dict[int, float] = {}
        self._committed_delivery_probability_by_action: dict[int, float] = {}
        self._latest_deadline_s = min(
            float(base_world.max_decision_latency_s),
            float(config.decision_deadline_s),
        )
        self._accepted_state_type_names: set[str] = set()
        self._rejected_state_type_names: list[str] = []

    @property
    def latest_graph(self) -> CommunicationGraph:
        if self._latest_graph is None:
            raise RuntimeError("availability has not been evaluated")
        return self._latest_graph

    @property
    def latest_record(self) -> AvailabilityRecord:
        if self._latest_record is None:
            raise RuntimeError("availability has not been evaluated")
        return self._latest_record

    def input_contract_audit(self) -> dict[str, object]:
        """Report the runtime-enforced topology-input boundary.

        This is live instrumentation rather than a static declaration.  The
        manager records every accepted input type and every rejected attempt,
        allowing the orchestration report to prove that simulator ``EpisodeState``
        was not admitted into operational availability.
        """

        accepted = tuple(sorted(self._accepted_state_type_names))
        rejected = tuple(self._rejected_state_type_names)
        return {
            "required_type": OperationalNetworkState.__name__,
            "accepted_type_names": accepted,
            "rejected_type_names": rejected,
            "accepted_non_contract_type": any(
                name != OperationalNetworkState.__name__ for name in accepted
            ),
            "forbidden_episode_state_accepted": "EpisodeState" in accepted,
            "rejected_attempt_count": len(rejected),
        }

    @staticmethod
    def _payload_by_source(action: RadarAction) -> tuple[dict[str, float], tuple[str, ...]]:
        """Return declared action data products without inventing a source.

        The schedule schema already declares ``communication_mb_by_node``.  Its
        sum must equal ``communication_cost_mb``; otherwise routing the aggregate
        cost from every participant would duplicate traffic, while assigning it
        to an arbitrary node would fabricate a transport model.  Such actions
        therefore fail closed.
        """

        raw = action.metadata.get("communication_mb_by_node", {})
        if not isinstance(raw, dict):
            return {}, ("invalid_communication_allocation",)
        payloads: dict[str, float] = {}
        reasons: list[str] = []
        for node_id, value in raw.items():
            payload = float(value)
            if not math.isfinite(payload) or payload < 0.0:
                reasons.append("invalid_communication_payload")
                continue
            if payload > 0.0:
                payloads[str(node_id)] = payloads.get(str(node_id), 0.0) + payload
        declared = float(action.communication_cost_mb)
        allocated = float(sum(payloads.values()))
        tolerance = max(1e-12, 1e-9 * max(abs(declared), 1.0))
        if abs(allocated - declared) > tolerance:
            reasons.append("communication_allocation_does_not_sum_to_action_cost")
        return payloads, tuple(sorted(set(reasons)))

    def _routes_for_action(
        self,
        graph: CommunicationGraph,
        action: RadarAction,
    ) -> tuple[tuple[TransportPath, ...], tuple[str, ...]]:
        payloads, payload_reasons = self._payload_by_source(action)
        reasons = list(payload_reasons)
        participants = tuple(
            sorted(set((*action_participants(action), self.fusion_node_id)))
        )
        participant_set = set(participants)
        unknown_payload_sources = sorted(set(payloads).difference(participant_set))
        if unknown_payload_sources:
            reasons.append("communication_payload_source_not_action_participant")
        routes: list[TransportPath] = []
        if reasons:
            return (), tuple(sorted(set(reasons)))
        for participant in participants:
            route = _shortest_transport_path(
                graph,
                participant,
                self.fusion_node_id,
                payload_mb=payloads.get(participant, 0.0),
                background_delay_s_by_node=self._background_delay_s_by_node,
            )
            if route is None:
                reasons.append("no_finite_transport_path")
                continue
            routes.append(route)
        return tuple(routes), tuple(sorted(set(reasons)))

    def evaluate(
        self,
        dynamic_world: PhysicalWorld,
        state: OperationalNetworkState,
        actions: Iterable[RadarAction] | None = None,
    ) -> AvailabilityRecord:
        if not isinstance(state, OperationalNetworkState):
            self._rejected_state_type_names.append(type(state).__name__)
            raise TypeError(
                "distributed availability accepts only OperationalNetworkState; "
                "passing simulator EpisodeState is forbidden"
            )
        self._accepted_state_type_names.add(type(state).__name__)
        state.validate()
        self._background_delay_s_by_node = {
            str(node_id): int(delay_steps) * float(self.config.dt_s)
            for node_id, delay_steps in state.node_packet_delay_steps.items()
        }
        node_ids = {node.node_id for node in dynamic_world.nodes}
        telemetry_ids = (
            set(state.node_packet_delay_steps)
            | set(state.node_queue_megabytes)
            | set(state.node_compute_load_fraction)
            | set(state.node_packet_loss_probability)
        )
        unknown_telemetry = sorted(telemetry_ids.difference(node_ids))
        if unknown_telemetry:
            raise ValueError(
                "operational network telemetry names unknown nodes: "
                + ", ".join(unknown_telemetry)
            )
        self._node_transport_telemetry = {
            node_id: {
                "packet_delay_steps": float(
                    state.node_packet_delay_steps.get(node_id, 0)
                ),
                "background_delay_s": float(
                    self._background_delay_s_by_node.get(node_id, 0.0)
                ),
                "queue_megabytes": float(
                    state.node_queue_megabytes.get(node_id, 0.0)
                ),
                "compute_load_fraction": float(
                    state.node_compute_load_fraction.get(node_id, 0.0)
                ),
                "packet_loss_probability": float(
                    state.node_packet_loss_probability.get(node_id, 0.0)
                ),
            }
            for node_id in sorted(node_ids)
        }
        graph = communication_graph_from_episode(dynamic_world, state)
        selected = tuple(actions if actions is not None else dynamic_world.actions)
        components = connected_components(graph.adjacency)
        component_by_node = {
            graph.node_ids[node_index]: component_index
            for component_index, component in enumerate(components)
            for node_index in component
        }
        fusion_component = component_by_node[self.fusion_node_id]
        fusion_nodes = tuple(
            node_id
            for node_id in graph.node_ids
            if component_by_node[node_id] == fusion_component
        )
        available: list[int] = []
        blocked: dict[int, tuple[str, ...]] = {}
        transport_latency: dict[int, float] = {}
        arrival_epoch_by_action: dict[int, int] = {}
        routes_by_action: dict[int, tuple[TransportPath, ...]] = {}
        for action in selected:
            reasons: list[str] = []
            if not action.allowed:
                reasons.append("action_disabled_by_dynamic_world")
            participants = tuple(sorted(set((*action_participants(action), self.fusion_node_id))))
            if any(node_id not in component_by_node for node_id in participants):
                reasons.append("unknown_participant")
            elif any(component_by_node[node_id] != fusion_component for node_id in participants):
                reasons.append("participant_not_reachable_from_fusion_node")
            maximum_latency = 0.0
            if not reasons:
                routes, route_reasons = self._routes_for_action(graph, action)
                routes_by_action[action.action_id] = routes
                reasons.extend(route_reasons)
                if routes:
                    maximum_latency = max(route.total_latency_s for route in routes)
                if not reasons and maximum_latency > min(
                    dynamic_world.max_decision_latency_s,
                    self.config.decision_deadline_s,
                ):
                    reasons.append("transport_latency_exceeds_decision_deadline")
            transport_latency[action.action_id] = float(maximum_latency)
            if math.isfinite(maximum_latency):
                if maximum_latency <= self.config.same_epoch_fusion_window_s:
                    arrival_epoch = int(state.epoch)
                else:
                    residual = maximum_latency - self.config.same_epoch_fusion_window_s
                    arrival_epoch = int(state.epoch) + max(
                        1,
                        int(math.ceil(residual / self.config.dt_s)),
                    )
            else:
                arrival_epoch = int(state.epoch) + self.config.fixed_lag_steps
            arrival_epoch_by_action[action.action_id] = arrival_epoch
            if not reasons and arrival_epoch - int(state.epoch) >= self.config.fixed_lag_steps:
                reasons.append("transport_arrival_exceeds_active_graph_window")
            if not reasons and arrival_epoch >= self.config.horizon_steps:
                reasons.append("observation_arrives_after_operational_horizon")
            if reasons:
                blocked[action.action_id] = tuple(sorted(set(reasons)))
            else:
                available.append(action.action_id)
        observation = build_observation_hypergraph(dynamic_world, selected)
        contract = communication_observation_contract(graph, observation)
        contract = {
            **contract,
            "fusion_node_id": self.fusion_node_id,
            "fusion_component_nodes": list(fusion_nodes),
            "topology": topology_report(graph),
            "availability_rule": "all likelihood participants and the fusion node must share one deadline-feasible component",
            "transport_model": "finite_payload_store_and_forward",
            "action_routes": {
                str(action_id): [
                    {
                        "source": route.source,
                        "destination": route.destination,
                        "node_path": list(route.node_path),
                        "payload_mb": route.payload_mb,
                        "engineering_latency_s": route.engineering_latency_s,
                        "serialization_latency_s": route.serialization_latency_s,
                        "background_latency_s": route.background_latency_s,
                        "total_latency_s": route.total_latency_s,
                        "bottleneck_capacity_mbps": route.bottleneck_capacity_mbps,
                    }
                    for route in routes
                ]
                for action_id, routes in sorted(routes_by_action.items())
            },
            "action_delivery_probability": {
                str(action_id): self._delivery_probability_for_routes(routes)
                for action_id, routes in sorted(routes_by_action.items())
            },
            "node_transport_telemetry": self._node_transport_telemetry,
        }
        record = AvailabilityRecord(
            epoch=state.epoch,
            fusion_node_id=self.fusion_node_id,
            component_by_node=component_by_node,
            fusion_component_nodes=fusion_nodes,
            available_action_ids=tuple(sorted(available)),
            blocked_action_reasons=blocked,
            action_transport_latency_s=transport_latency,
            action_arrival_epoch=arrival_epoch_by_action,
            communication_contract=contract,
            status="passed" if self.fusion_node_id in fusion_nodes else "failed_closed",
        )
        self._latest_graph = graph
        self._latest_record = record
        self._latest_routes_by_action = routes_by_action
        # A selected schedule belongs to one communication snapshot.  A new
        # availability evaluation invalidates the previous schedule-specific
        # queueing plan before any observation can query it.
        self._committed_schedule_epoch = None
        self._committed_arrival_epoch_by_action = {}
        self._committed_transport_latency_by_action = {}
        self._committed_delivery_probability_by_action = {}
        self._latest_deadline_s = min(
            float(dynamic_world.max_decision_latency_s),
            float(self.config.decision_deadline_s),
        )
        return record

    def snapshot(self, epoch: int) -> CommunicationSnapshot:
        graph = self.latest_graph
        edges = tuple(
            (graph.node_ids[i], graph.node_ids[j])
            for i in range(len(graph.node_ids))
            for j in range(i + 1, len(graph.node_ids))
            if graph.adjacency[i, j] > 0.0
        )
        return CommunicationSnapshot(int(epoch), graph.node_ids, edges)

    def factor_participants(self, action: RadarAction) -> tuple[str, ...]:
        return tuple(sorted(set((*action_participants(action), self.fusion_node_id))))

    def action_available(self, action_id: int) -> bool:
        return int(action_id) in set(self.latest_record.available_action_ids)

    def route_control_message(
        self,
        source: str,
        destination: str,
        *,
        payload_mb: float,
    ) -> dict[str, object]:
        """Route one finite control/proposal message on the live graph.

        This is an observability adapter for the Part-1 Agent layer.  It reads
        only the already-evaluated operational communication graph and node
        telemetry.  It does not mutate action availability, the world, or the
        authoritative posterior.
        """

        route = _shortest_transport_path(
            self.latest_graph,
            str(source),
            str(destination),
            payload_mb=float(payload_mb),
            background_delay_s_by_node=self._background_delay_s_by_node,
        )
        if route is None:
            return {
                "status": "unreachable",
                "source": str(source),
                "destination": str(destination),
                "payload_mb": float(payload_mb),
                "delivery_probability": 0.0,
            }
        probability = 1.0
        for sender in route.node_path[:-1]:
            loss = float(
                self._node_transport_telemetry.get(sender, {}).get(
                    "packet_loss_probability", 0.0
                )
            )
            probability *= 1.0 - float(np.clip(loss, 0.0, 1.0))
        return {
            "status": "routed",
            "source": route.source,
            "destination": route.destination,
            "node_path": tuple(route.node_path),
            "edge_path": tuple(route.edge_path),
            "payload_mb": float(route.payload_mb),
            "engineering_latency_s": float(route.engineering_latency_s),
            "serialization_latency_s": float(route.serialization_latency_s),
            "background_latency_s": float(route.background_latency_s),
            "total_latency_s": float(route.total_latency_s),
            "bottleneck_capacity_mbps": float(route.bottleneck_capacity_mbps),
            "delivery_probability": float(np.clip(probability, 0.0, 1.0)),
            "mutates_operational_state": False,
        }

    def _delivery_probability_for_routes(
        self,
        routes: Iterable[TransportPath],
    ) -> float:
        probability = 1.0
        for route in routes:
            if route.payload_mb <= 0.0:
                continue
            for sender in route.node_path[:-1]:
                loss = float(
                    self._node_transport_telemetry.get(sender, {}).get(
                        "packet_loss_probability", 0.0
                    )
                )
                probability *= 1.0 - float(np.clip(loss, 0.0, 1.0))
        return float(np.clip(probability, 0.0, 1.0))

    def _delivery_probability_from_routes(self, action_id: int) -> float:
        return self._delivery_probability_for_routes(
            self._latest_routes_by_action.get(int(action_id), ())
        )

    def delivery_probability(self, action_id: int, *, epoch: int | None = None) -> float:
        """Return the committed all-products delivery probability for an action."""

        token = int(action_id)
        if (
            epoch is not None
            and self._committed_schedule_epoch == int(epoch)
            and token in self._committed_delivery_probability_by_action
        ):
            return float(self._committed_delivery_probability_by_action[token])
        return self._delivery_probability_from_routes(token)

    def schedule_operational_feasibility(
        self,
        actions: Iterable[RadarAction],
    ) -> tuple[bool, tuple[str, ...], dict[str, object]]:
        """Check joint routed traffic, not only per-action reachability.

        Individual packets may each fit a link while their simultaneous sum does
        not.  For every undirected link ``e`` used by the selected routes, the
        finite-deadline capacity condition is

        ``8 * sum(M_{a,s}: e in p_{a,s}) / T_deadline <= C_e``.

        We also compute a conservative aggregate store-and-forward completion
        time by charging every route the total queued payload on each traversed
        link.  This is exact for serial service in the route order and fail-safe
        for arbitrary work-conserving scheduling.
        """

        selected = tuple(actions)
        reasons: list[str] = []
        unavailable = [
            action.action_id
            for action in selected
            if not self.action_available(action.action_id)
        ]
        if unavailable:
            reasons.append("schedule_contains_individually_unavailable_action")
        edge_payload_mb: dict[tuple[str, str], float] = {}
        route_records: list[tuple[int, TransportPath]] = []
        for action in selected:
            routes = self._latest_routes_by_action.get(action.action_id, ())
            if action.action_id not in self._latest_routes_by_action:
                reasons.append("schedule_route_missing")
                continue
            route_records.extend((action.action_id, route) for route in routes)
            for route in routes:
                if route.payload_mb <= 0.0:
                    continue
                for edge in route.edge_path:
                    edge_payload_mb[edge] = edge_payload_mb.get(edge, 0.0) + route.payload_mb

        node_index = {
            node_id: index for index, node_id in enumerate(self.latest_graph.node_ids)
        }
        deadline = float(self._latest_deadline_s)
        edge_load_mbps: dict[str, float] = {}
        edge_capacity_mbps: dict[str, float] = {}
        for edge, payload in edge_payload_mb.items():
            left, right = edge
            i, j = node_index[left], node_index[right]
            capacity = float(self.latest_graph.capacity_mbps[i, j])
            offered = 8.0 * payload / max(deadline, 1e-30)
            token = f"{left}<->{right}"
            edge_load_mbps[token] = offered
            edge_capacity_mbps[token] = capacity
            if offered > capacity + 1e-12:
                reasons.append(f"shared_link_capacity_exceeded:{token}")

        aggregate_route_latency: dict[str, float] = {}
        action_aggregate_latency: dict[int, float] = {
            action.action_id: 0.0 for action in selected
        }
        action_delivery_probability = {
            action.action_id: self._delivery_probability_from_routes(action.action_id)
            for action in selected
        }
        for action_id, route in route_records:
            engineering = route.engineering_latency_s + route.background_latency_s
            serialization = 0.0
            for edge in route.edge_path:
                left, right = edge
                i, j = node_index[left], node_index[right]
                serialization += 8.0 * edge_payload_mb.get(edge, 0.0) / max(
                    float(self.latest_graph.capacity_mbps[i, j]), 1e-30
                )
            total = engineering + serialization
            token = f"{route.source}->{route.destination}"
            aggregate_route_latency[token] = max(
                aggregate_route_latency.get(token, 0.0), total
            )
            action_aggregate_latency[action_id] = max(
                action_aggregate_latency.get(action_id, 0.0), total
            )
            if total > deadline + 1e-12:
                reasons.append(f"shared_link_queue_exceeds_deadline:{token}")

        capture_epoch = int(self.latest_record.epoch)
        action_arrival_epoch: dict[int, int] = {}
        for action in selected:
            latency = float(action_aggregate_latency.get(action.action_id, 0.0))
            if latency <= self.config.same_epoch_fusion_window_s:
                arrival = capture_epoch
            else:
                residual = latency - self.config.same_epoch_fusion_window_s
                arrival = capture_epoch + max(
                    1,
                    int(math.ceil(residual / self.config.dt_s)),
                )
            action_arrival_epoch[action.action_id] = int(arrival)
            if arrival - capture_epoch >= self.config.fixed_lag_steps:
                reasons.append(
                    f"shared_transport_arrival_exceeds_active_graph_window:{action.action_id}"
                )
            if arrival >= self.config.horizon_steps:
                reasons.append(
                    f"shared_transport_arrives_after_operational_horizon:{action.action_id}"
                )

        unique_reasons = tuple(sorted(set(reasons)))
        return (
            not unique_reasons,
            unique_reasons,
            {
                "deadline_s": deadline,
                "edge_payload_mb": {
                    f"{left}<->{right}": value
                    for (left, right), value in sorted(edge_payload_mb.items())
                },
                "edge_load_mbps": edge_load_mbps,
                "edge_capacity_mbps": edge_capacity_mbps,
                "aggregate_route_latency_s": aggregate_route_latency,
                "action_aggregate_transport_latency_s": {
                    str(action_id): float(value)
                    for action_id, value in sorted(action_aggregate_latency.items())
                },
                "action_arrival_epoch": {
                    str(action_id): int(value)
                    for action_id, value in sorted(action_arrival_epoch.items())
                },
                "action_delivery_probability": {
                    str(action_id): float(value)
                    for action_id, value in sorted(action_delivery_probability.items())
                },
                "transport_model": "finite_payload_store_and_forward_with_shared_link_queue",
                "delivery_model": "independent_per_transmission_node_packet_loss",
            },
        )

    def commit_selected_schedule(
        self,
        actions: Iterable[RadarAction],
        *,
        epoch: int,
    ) -> dict[str, object]:
        """Freeze the selected schedule's shared-queue arrival contract.

        Candidate evaluation is side-effect free.  Only after the safety layer
        has selected a schedule do we commit the joint routed traffic plan used
        by observation timestamps.  This prevents individually feasible
        packets from being fused in the capture epoch when shared-link
        serialization pushes their actual arrival into a later epoch.
        """

        selected = tuple(actions)
        if int(epoch) != int(self.latest_record.epoch):
            raise RuntimeError("selected schedule epoch disagrees with availability snapshot")
        feasible, reasons, audit = self.schedule_operational_feasibility(selected)
        if not feasible:
            raise RuntimeError(
                "cannot commit an operationally infeasible selected schedule: "
                + ", ".join(reasons)
            )
        raw_arrivals = audit.get("action_arrival_epoch", {})
        raw_latencies = audit.get("action_aggregate_transport_latency_s", {})
        raw_probabilities = audit.get("action_delivery_probability", {})
        if (
            not isinstance(raw_arrivals, dict)
            or not isinstance(raw_latencies, dict)
            or not isinstance(raw_probabilities, dict)
        ):
            raise RuntimeError("schedule transport audit omitted arrival semantics")
        self._committed_schedule_epoch = int(epoch)
        self._committed_arrival_epoch_by_action = {
            int(action_id): int(raw_arrivals[str(action_id)])
            for action_id in (action.action_id for action in selected)
        }
        self._committed_transport_latency_by_action = {
            int(action_id): float(raw_latencies[str(action_id)])
            for action_id in (action.action_id for action in selected)
        }
        self._committed_delivery_probability_by_action = {
            int(action_id): float(raw_probabilities[str(action_id)])
            for action_id in (action.action_id for action in selected)
        }
        return audit

    def clear_committed_schedule(self) -> None:
        self._committed_schedule_epoch = None
        self._committed_arrival_epoch_by_action = {}
        self._committed_transport_latency_by_action = {}
        self._committed_delivery_probability_by_action = {}

    def arrival_epoch(self, action_id: int, epoch: int) -> int:
        if (
            self._committed_schedule_epoch == int(epoch)
            and int(action_id) in self._committed_arrival_epoch_by_action
        ):
            return int(self._committed_arrival_epoch_by_action[int(action_id)])
        recorded = self.latest_record.action_arrival_epoch.get(int(action_id))
        if recorded is not None:
            return int(recorded)
        return int(epoch) + self.config.fixed_lag_steps
