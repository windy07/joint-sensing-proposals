from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence


class NodeKind(str, Enum):
    SENSOR = "sensor"
    TRACK_VARIABLE = "track_variable"
    NUISANCE_VARIABLE = "nuisance_variable"
    FACTOR = "factor"
    MESSAGE = "message"
    PROVENANCE_ROOT = "provenance_root"
    ACTION = "action"
    RESOURCE = "resource"
    COALITION = "coalition"


class EdgeKind(str, Enum):
    OBSERVES = "observes"
    CONDITIONS = "conditions"
    ARRIVED_MESSAGE = "arrived_message"
    PROVENANCE = "provenance"
    SHARES_NUISANCE = "shares_nuisance"
    OWNS_ACTION = "owns_action"
    CONSUMES_RESOURCE = "consumes_resource"
    COMMUNICATES = "communicates"
    TEMPORAL = "temporal"
    INCIDENCE = "incidence"


FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "truth",
        "ground_truth",
        "target_truth",
        "true_state",
        "true_origin",
        "origin_truth",
        "future_measurement",
        "future_arrival",
        "test_label",
        "confirmation_metric",
        "oracle_utility",
        "oracle_rank",
        "heldout_label",
    }
)
SUPPORTED_INFORMATION_GRAPH_SCHEMA_VERSIONS = frozenset(
    {"iclr2027.info_graph.v1", "iclr2027.info_graph.v2"}
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are forbidden in canonical graph records")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    kind: NodeKind
    role: str
    timestamp_s: float
    owner_id: str | None = None
    features: Mapping[str, float] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    categorical: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be nonempty")
        if not math.isfinite(self.timestamp_s):
            raise ValueError(f"node {self.node_id}: timestamp_s must be finite")
        overlap = set(self.features) & FORBIDDEN_RUNTIME_KEYS
        if overlap:
            raise ValueError(f"node {self.node_id}: forbidden runtime features {sorted(overlap)}")
        for key, value in self.features.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"node {self.node_id}: feature {key!r} must be finite numeric")
        unknown_units = set(self.units) - set(self.features)
        if unknown_units:
            raise ValueError(f"node {self.node_id}: units without features {sorted(unknown_units)}")


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    kind: EdgeKind
    directed: bool = True
    timestamp_s: float = 0.0
    features: Mapping[str, float] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)

    def validate(self, node_ids: set[str]) -> None:
        if self.source_id not in node_ids or self.target_id not in node_ids:
            raise ValueError(f"edge endpoint missing: {self.source_id!r}->{self.target_id!r}")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("edge timestamp must be finite")
        for key, value in self.features.items():
            if key in FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"edge contains forbidden feature {key!r}")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"edge feature {key!r} must be finite numeric")
        if set(self.units) - set(self.features):
            raise ValueError("edge units must refer to present numeric features")


@dataclass(frozen=True, slots=True)
class GraphHyperedge:
    hyperedge_id: str
    kind: str
    member_ids: tuple[str, ...]
    timestamp_s: float
    provenance_root_id: str | None = None
    features: Mapping[str, float] = field(default_factory=dict)

    def validate(self, node_ids: set[str]) -> None:
        if len(self.member_ids) < 2:
            raise ValueError("a hyperedge must contain at least two members")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError(f"hyperedge {self.hyperedge_id}: duplicate members")
        missing = set(self.member_ids) - node_ids
        if missing:
            raise ValueError(f"hyperedge {self.hyperedge_id}: missing members {sorted(missing)}")
        if self.provenance_root_id is not None and self.provenance_root_id not in node_ids:
            raise ValueError(f"hyperedge {self.hyperedge_id}: missing provenance root")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("hyperedge timestamp must be finite")
        for key, value in self.features.items():
            if key in FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"hyperedge contains forbidden feature {key!r}")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"hyperedge feature {key!r} must be finite numeric")


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    candidate_id: str
    family: str
    action_ids: tuple[str, ...]
    owner_ids: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    resource_costs: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    protected: bool = False

    def canonical_key(self) -> str:
        semantic_payload = {
            k: v
            for k, v in self.payload.items()
            if k not in {"candidate_id", "score", "model_score", "rank", "trace_id"}
        }
        return canonical_hash(
            {
                "family": self.family,
                "action_ids": sorted(self.action_ids),
                "owner_ids": sorted(self.owner_ids),
                "payload": semantic_payload,
            }
        )

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if not self.family:
            raise ValueError("candidate family must be nonempty")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError(f"candidate {self.candidate_id}: duplicate action ids")
        for key, value in self.resource_costs.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"candidate {self.candidate_id}: invalid resource {key}")
        forbidden = _find_forbidden_keys(self.payload) | _find_forbidden_keys(self.metadata)
        if forbidden:
            raise ValueError(f"candidate {self.candidate_id}: forbidden keys {sorted(forbidden)}")


@dataclass(frozen=True, slots=True)
class DynamicInformationGraph:
    graph_id: str
    episode_id: str
    cycle_index: int
    now_s: float
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    hyperedges: tuple[GraphHyperedge, ...] = ()
    system_posterior_ref: str | None = None
    schema_version: str = "iclr2027.info_graph.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version not in SUPPORTED_INFORMATION_GRAPH_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported information graph schema: {self.schema_version}"
            )
        if self.cycle_index < 0:
            raise ValueError("cycle_index must be nonnegative")
        if not math.isfinite(self.now_s):
            raise ValueError("now_s must be finite")
        ids=[n.node_id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("graph node ids must be unique")
        node_ids=set(ids)
        for node in self.nodes:
            node.validate()
            if node.timestamp_s > self.now_s + 1e-12:
                raise ValueError(f"future node {node.node_id} at {node.timestamp_s}>{self.now_s}")
        for edge in self.edges:
            edge.validate(node_ids)
            if edge.timestamp_s > self.now_s + 1e-12:
                raise ValueError("future edge is forbidden")
        for hyperedge in self.hyperedges:
            hyperedge.validate(node_ids)
            if hyperedge.timestamp_s > self.now_s + 1e-12:
                raise ValueError("future hyperedge is forbidden")
        forbidden=_find_forbidden_keys(self.metadata)
        if forbidden:
            raise ValueError(f"graph metadata contains forbidden keys {sorted(forbidden)}")

    def canonical_record(self) -> Mapping[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "episode_id": self.episode_id,
            "cycle_index": self.cycle_index,
            "now_s": self.now_s,
            "system_posterior_ref": self.system_posterior_ref,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "kind": n.kind.value,
                    "role": n.role,
                    "timestamp_s": n.timestamp_s,
                    "owner_id": n.owner_id,
                    "features": dict(n.features),
                    "units": dict(n.units),
                    "categorical": dict(n.categorical),
                }
                for n in sorted(self.nodes, key=lambda x: x.node_id)
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "kind": e.kind.value,
                    "directed": e.directed,
                    "timestamp_s": e.timestamp_s,
                    "features": dict(e.features),
                    "units": dict(e.units),
                }
                for e in sorted(
                    self.edges,
                    key=lambda x: (x.kind.value, x.source_id, x.target_id, x.timestamp_s),
                )
            ],
            "hyperedges": [
                {
                    "hyperedge_id": h.hyperedge_id,
                    "kind": h.kind,
                    "member_ids": sorted(h.member_ids),
                    "timestamp_s": h.timestamp_s,
                    "provenance_root_id": h.provenance_root_id,
                    "features": dict(h.features),
                }
                for h in sorted(self.hyperedges, key=lambda x: x.hyperedge_id)
            ],
            "metadata": dict(self.metadata),
        }

    def canonical_hash(self) -> str:
        return canonical_hash(self.canonical_record())

    def permuted(self, node_order: Sequence[str]) -> "DynamicInformationGraph":
        by_id={n.node_id:n for n in self.nodes}
        if set(node_order)!=set(by_id) or len(node_order)!=len(by_id):
            raise ValueError("node_order must be a permutation of all node ids")
        return replace(self, nodes=tuple(by_id[i] for i in node_order))


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_s=str(key).lower()
            if key_s in FORBIDDEN_RUNTIME_KEYS or key_s.startswith("truth_") or key_s.endswith("_truth"):
                found.add(str(key))
            found |= _find_forbidden_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found |= _find_forbidden_keys(nested)
    return found


def deduplicate_candidates(candidates: Iterable[ActionCandidate]) -> list[ActionCandidate]:
    result: list[ActionCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate.validate()
        key=candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
