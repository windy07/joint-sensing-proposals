from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schema import (
    DynamicInformationGraph,
    EdgeKind,
    FORBIDDEN_RUNTIME_KEYS,
    GraphEdge,
    GraphHyperedge,
    GraphNode,
    NodeKind,
    canonical_hash,
)


class LeakageError(ValueError):
    """Raised when truth, future information, or test-only labels enter the Agent view."""


class CausalityError(ValueError):
    """Raised when an item has not arrived by the graph decision time."""


@dataclass(frozen=True, slots=True)
class BuilderContract:
    schema_version: str = "iclr2027.info_graph.v1"
    require_units: bool = True
    reject_unknown_truth_aliases: bool = True
    time_tolerance_s: float = 1e-12


class InformationGraphBuilder:
    """Build the canonical Agent graph from arrived, Agent-allowed records.

    The builder intentionally accepts plain mappings so it can sit on the
    authoritative factor/message stores without copying their implementation.
    It performs a deny-list leakage scan and an arrival-time check before any
    feature conversion.
    """

    def __init__(self, contract: BuilderContract | None = None) -> None:
        self.contract=contract or BuilderContract()

    def build(
        self,
        snapshot: Mapping[str, Any],
        *,
        episode_id: str,
        cycle_index: int,
        now_s: float,
        system_posterior_ref: str | None,
    ) -> DynamicInformationGraph:
        self._assert_no_leakage(snapshot)
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        hyperedges: list[GraphHyperedge] = []

        for record in snapshot.get("nodes", ()):
            node=self._node(record, now_s)
            nodes.append(node)
        node_ids={n.node_id for n in nodes}

        # Provenance roots are first-class nodes.  The same root id shared by
        # multiple factors/messages prevents the model from treating copies as
        # statistically independent evidence.
        for record in snapshot.get("provenance_roots", ()):
            root_id=str(record["root_id"])
            if root_id in node_ids:
                continue
            timestamp=float(record.get("timestamp_s", now_s))
            self._assert_arrived(timestamp, now_s, root_id)
            nodes.append(
                GraphNode(
                    node_id=root_id,
                    kind=NodeKind.PROVENANCE_ROOT,
                    role=str(record.get("role", "shared_source")),
                    timestamp_s=timestamp,
                    owner_id=record.get("owner_id"),
                    features=self._numeric(record.get("features", {})),
                    units=self._units(record.get("units", {}), record.get("features", {})),
                    categorical=self._categorical(record.get("categorical", {})),
                )
            )
            node_ids.add(root_id)

        for record in snapshot.get("edges", ()):
            edges.append(self._edge(record, now_s))

        for record in snapshot.get("messages", ()):
            message_id=str(record["message_id"])
            timestamp=float(record.get("timestamp_s", record.get("sent_at_s", now_s)))
            arrived=float(record["arrived_at_s"])
            self._assert_arrived(arrived, now_s, message_id)
            if message_id not in node_ids:
                features=dict(record.get("features", {}))
                features.setdefault("age_s", max(0.0, now_s-arrived))
                features.setdefault("delay_s", max(0.0, arrived-timestamp))
                features.setdefault("compression_ratio", float(record.get("compression_ratio", 1.0)))
                nodes.append(
                    GraphNode(
                        node_id=message_id,
                        kind=NodeKind.MESSAGE,
                        role=str(record.get("role", "arrived_message")),
                        timestamp_s=arrived,
                        owner_id=record.get("sender_id"),
                        features=self._numeric(features),
                        units=self._units(record.get("units", {}), features),
                        categorical=self._categorical(
                            {
                                **record.get("categorical", {}),
                                "sender_id": str(record.get("sender_id", "unknown")),
                                "receiver_id": str(record.get("receiver_id", "unknown")),
                            }
                        ),
                    )
                )
                node_ids.add(message_id)
            sender=str(record["sender_id"])
            receiver=str(record["receiver_id"])
            if sender in node_ids:
                edges.append(GraphEdge(sender, message_id, EdgeKind.COMMUNICATES, True, arrived))
            if receiver in node_ids:
                edges.append(GraphEdge(message_id, receiver, EdgeKind.ARRIVED_MESSAGE, True, arrived))
            root=record.get("provenance_root_id")
            if root is not None and str(root) in node_ids:
                edges.append(GraphEdge(str(root), message_id, EdgeKind.PROVENANCE, True, arrived))

        for record in snapshot.get("hyperedges", ()):
            timestamp=float(record.get("timestamp_s", now_s))
            self._assert_arrived(timestamp, now_s, str(record.get("hyperedge_id", "hyperedge")))
            hyperedges.append(
                GraphHyperedge(
                    hyperedge_id=str(record["hyperedge_id"]),
                    kind=str(record["kind"]),
                    member_ids=tuple(str(v) for v in record["member_ids"]),
                    timestamp_s=timestamp,
                    provenance_root_id=(
                        None if record.get("provenance_root_id") is None else str(record["provenance_root_id"])
                    ),
                    features=self._numeric(record.get("features", {})),
                )
            )

        graph=DynamicInformationGraph(
            graph_id=str(snapshot.get("graph_id") or canonical_hash({
                "episode_id":episode_id,"cycle_index":cycle_index,"now_s":now_s,
                "node_ids":sorted(n.node_id for n in nodes),
            })[:24]),
            episode_id=episode_id,
            cycle_index=cycle_index,
            now_s=float(now_s),
            nodes=tuple(nodes),
            edges=tuple(edges),
            hyperedges=tuple(hyperedges),
            system_posterior_ref=system_posterior_ref,
            schema_version=self.contract.schema_version,
            metadata=dict(snapshot.get("metadata", {})),
        )
        graph.validate()
        return graph

    def _node(self, record: Mapping[str, Any], now_s: float) -> GraphNode:
        timestamp=float(record.get("timestamp_s", now_s))
        arrived=float(record.get("arrived_at_s", timestamp))
        self._assert_arrived(arrived, now_s, str(record.get("node_id", "node")))
        features=dict(record.get("features", {}))
        features.setdefault("age_s", max(0.0, now_s-arrived))
        return GraphNode(
            node_id=str(record["node_id"]),
            kind=NodeKind(str(record["kind"])),
            role=str(record.get("role", "unspecified")),
            timestamp_s=arrived,
            owner_id=None if record.get("owner_id") is None else str(record["owner_id"]),
            features=self._numeric(features),
            units=self._units(record.get("units", {}), features),
            categorical=self._categorical(record.get("categorical", {})),
        )

    def _edge(self, record: Mapping[str, Any], now_s: float) -> GraphEdge:
        timestamp=float(record.get("timestamp_s", now_s))
        arrived=float(record.get("arrived_at_s", timestamp))
        self._assert_arrived(arrived, now_s, f"{record.get('source_id')}->{record.get('target_id')}")
        features=dict(record.get("features", {}))
        features.setdefault("age_s", max(0.0, now_s-arrived))
        return GraphEdge(
            source_id=str(record["source_id"]),
            target_id=str(record["target_id"]),
            kind=EdgeKind(str(record["kind"])),
            directed=bool(record.get("directed", True)),
            timestamp_s=arrived,
            features=self._numeric(features),
            units=self._units(record.get("units", {}), features),
        )

    def _assert_arrived(self, arrived_at_s: float, now_s: float, item_id: str) -> None:
        if arrived_at_s > now_s + self.contract.time_tolerance_s:
            raise CausalityError(f"{item_id} arrives at {arrived_at_s}, after decision time {now_s}")

    def _assert_no_leakage(self, value: Any, path: str = "snapshot") -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_l=str(key).lower()
                if (
                    key_l in FORBIDDEN_RUNTIME_KEYS
                    or key_l.startswith("truth_")
                    or key_l.endswith("_truth")
                    or (self.contract.reject_unknown_truth_aliases and "groundtruth" in key_l.replace("_", ""))
                ):
                    raise LeakageError(f"forbidden Agent-view field at {path}.{key}")
                self._assert_no_leakage(nested, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                self._assert_no_leakage(nested, f"{path}[{index}]")

    @staticmethod
    def _numeric(values: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, bool):
                out[str(key)]=float(value)
            elif isinstance(value, (int, float)):
                out[str(key)]=float(value)
        return out

    def _units(self, values: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, str]:
        result={str(k):str(v) for k,v in values.items() if k in features}
        if self.contract.require_units:
            # Dimensionless is explicit rather than silently assumed.
            for key in features:
                result.setdefault(str(key), "1")
        return result

    @staticmethod
    def _categorical(values: Mapping[str, Any]) -> dict[str, str]:
        return {str(k):str(v) for k,v in values.items()}
