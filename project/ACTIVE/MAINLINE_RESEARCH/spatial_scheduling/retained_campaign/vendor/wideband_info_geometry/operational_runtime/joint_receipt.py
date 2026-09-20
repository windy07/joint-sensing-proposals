from __future__ import annotations

"""Receiver-local receipts and atomic publication for the new joint-set mode.

This module owns no posterior.  Capture adapters issue transports from an
actual committed capture and its actual delivery draw.  The graph owns all
likelihood branches.  Legacy receiver_admission contracts are not relaxed.

Opaque issuance prevents accidental fabrication, not hostile Python access.
The runtime is single-threaded: prepare/validate may allocate, but graph
publication must be exactly one prebuilt ``_joint_state`` reference.  The
coordinator invokes no publisher callback after the commit point.  A budget
ledger is external and its charges are never rolled back with inference state.
"""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping
import weakref

from .joint_return_set import FrozenAffineReturnLaw, JointReturnSetPayload


class JointReceiptError(ValueError):
    """Unsupported or invalid receiver transaction; never a missed detection."""


_TRANSPORT_SEAL = object()
_RECEIPT_SEAL = object()
_BATCH_SEAL = object()
_PLAN_SEAL = object()
_TRANSPORTS = weakref.WeakValueDictionary()
_RECEIPTS = weakref.WeakValueDictionary()
_BATCHES = weakref.WeakValueDictionary()
_PLANS = weakref.WeakValueDictionary()


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
        raise JointReceiptError("UNKNOWN_" + name + "_INTEGER")
    return value


def _positive(value: Any, name: str) -> int:
    if _integer(value, name) == 0:
        raise JointReceiptError("UNKNOWN_" + name + "_POSITIVE")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise JointReceiptError("UNKNOWN_" + name + "_IDENTIFIER")
    return value


def _digest(value: Any) -> str:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise JointReceiptError("UNKNOWN_CONTEXT_DIGEST")
    return value


def _object_digest(value: Any) -> str:
    # The two exact source types own a read-only string digest property.
    return _digest(value.digest)


def _immutable_tree(value: Any, *, depth: int = 0, remaining=None, byte_budget=None) -> Any:
    """Validate and serialize finite immutable typed data without repr fallbacks."""
    if remaining is None:
        remaining = [200000]
    remaining[0] -= 1
    if remaining[0] < 0 or depth > 32:
        raise JointReceiptError("UNKNOWN_IMMUTABLE_DATA_CAPACITY")
    if byte_budget is not None:
        # Conservative serialized-size charge BEFORE constructing the tree or
        # JSON string. Six bytes covers every escaped input character. Integer
        # bit_length/3 bounds its decimal digits. Container overhead is charged
        # per node as well as the explicit dataclass keys below.
        charge = 32
        if type(value) is str:
            charge += 6 * len(value)
        elif type(value) is int:
            charge += value.bit_length() // 3 + 2
        if is_dataclass(value) and not isinstance(value, type):
            charge += 6 * (len(type(value).__module__) + len(type(value).__qualname__)
                           + sum(len(f.name) for f in fields(value)))
        if isinstance(value, Enum):
            charge += 6 * (len(type(value).__module__) + len(type(value).__qualname__))
        byte_budget[0] -= charge
        if byte_budget[0] < 0:
            raise JointReceiptError("UNKNOWN_CONTEXT_OR_PAYLOAD_BYTE_CAPACITY")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise JointReceiptError("UNKNOWN_NONFINITE_CONTEXT_OR_PAYLOAD")
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__module__ + "." + type(value).__qualname__,
                "value": _immutable_tree(value.value, depth=depth + 1,
                    remaining=remaining, byte_budget=byte_budget)}
    if type(value) is tuple:
        return [_immutable_tree(v, depth=depth + 1, remaining=remaining,
                    byte_budget=byte_budget) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise JointReceiptError("UNKNOWN_MUTABLE_CONTEXT_OR_PAYLOAD")
        return {"type": type(value).__module__ + "." + type(value).__qualname__,
                "fields": {f.name: _immutable_tree(getattr(value, f.name),
                    depth=depth + 1, remaining=remaining, byte_budget=byte_budget)
                    for f in fields(value)}}
    raise JointReceiptError("UNKNOWN_MUTABLE_OR_UNSUPPORTED_CONTEXT_DATA")


def _content_digest(value: Any, byte_cap: int) -> str:
    encoded = json.dumps(_immutable_tree(value, byte_budget=[byte_cap]), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > byte_cap:
        raise JointReceiptError("UNKNOWN_CONTEXT_OR_PAYLOAD_BYTE_CAPACITY")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _RegisteredContext:
    law: Any
    law_digest: str
    content_digest: str
    action_id: int
    observation_epoch: int


class JointContextRegistry:
    """Capture-time law registry, constructed with exact trusted source types.

    Register before the capture generator is called, then issue its transport.
    Registration proves a frozen binding, not physical measurability by itself.
    The registered law and payload types must contain only immutable typed data.
    Context dependencies are not newly acquired observation provenance roots.
    """

    def __init__(self, *, max_contexts: int,
                 law_type: type = FrozenAffineReturnLaw,
                 payload_type: type = JointReturnSetPayload,
                 max_context_bytes: int = 1048576,
                 max_payload_bytes: int = 1048576, max_return_count: int = 4):
        if law_type is not FrozenAffineReturnLaw or payload_type is not JointReturnSetPayload:
            raise JointReceiptError("UNKNOWN_CONTEXT_SOURCE_TYPES")
        self.law_type = law_type
        self.payload_type = payload_type
        self.max_contexts = _positive(max_contexts, "CONTEXT_CAPACITY")
        self.max_context_bytes = _positive(max_context_bytes, "CONTEXT_BYTES")
        self.max_payload_bytes = _positive(max_payload_bytes, "PAYLOAD_BYTES")
        self.max_return_count = _positive(max_return_count, "RETURN_COUNT")
        self._contexts: dict[tuple[int, int], _RegisteredContext] = {}
        self._token = object()

    def register(self, law: Any, *, action_id: int, observation_epoch: int) -> str:
        key = (_integer(action_id, "ACTION"), _integer(observation_epoch, "CAPTURE_EPOCH"))
        if type(law) is not self.law_type:
            raise JointReceiptError("UNKNOWN_UNREGISTERED_LAW_TYPE")
        content = _content_digest(law, self.max_context_bytes)
        digest = _object_digest(law)
        for name, expected in (("action_id", key[0]), ("observation_epoch", key[1])):
            if _integer(getattr(law, name), "LAW_CAPTURE_IDENTITY") != expected:
                raise JointReceiptError("UNKNOWN_LAW_CAPTURE_IDENTITY")
        previous = self._contexts.get(key)
        if previous is not None:
            if previous.law_digest != digest or previous.content_digest != content:
                raise JointReceiptError("UNKNOWN_CAPTURE_CONTEXT_ALREADY_FROZEN")
            return digest
        if len(self._contexts) >= self.max_contexts:
            raise JointReceiptError("UNKNOWN_CONTEXT_CAPACITY")
        self._contexts[key] = _RegisteredContext(law, digest, content, *key)
        return digest

    def _get(self, action_id: int, observation_epoch: int, law_digest: str) -> _RegisteredContext:
        context = self._contexts.get((action_id, observation_epoch))
        if context is None or context.law_digest != law_digest:
            raise JointReceiptError("UNKNOWN_UNREGISTERED_CAPTURE_CONTEXT")
        if (_content_digest(context.law, self.max_context_bytes) != context.content_digest
                or _object_digest(context.law) != context.law_digest):
            raise JointReceiptError("UNKNOWN_REGISTERED_CONTEXT_CHANGED")
        return context

    def _payload_digest(self, payload: Any, event: "JointTransportEvent") -> str:
        if type(payload) is not self.payload_type:
            raise JointReceiptError("UNKNOWN_JOINT_PAYLOAD_TYPE")
        for name, expected in (("action_id", event.action_id),
                               ("observation_epoch", event.observation_epoch),
                               ("provenance_id", event.capture_id),
                               ("law_digest", event.law_digest)):
            if getattr(payload, name, None) != expected:
                raise JointReceiptError("UNKNOWN_JOINT_PAYLOAD_IDENTITY")
        _integer(payload.action_id, "PAYLOAD_ACTION")
        _integer(payload.observation_epoch, "PAYLOAD_EPOCH")
        rows = payload.measurements
        if type(rows) is not tuple or len(rows) > self.max_return_count:
            raise JointReceiptError("UNKNOWN_RETURN_COUNT_OR_SCHEMA")
        if hasattr(payload, "complete") and payload.complete is not True:
            raise JointReceiptError("UNKNOWN_INCOMPLETE_RETURN_SET")
        # This is invoked only for due packets, never on loss/future paths.
        content = _content_digest(payload, self.max_payload_bytes)
        context = self._get(event.action_id, event.observation_epoch, event.law_digest)
        context.law.validate_payload(payload)
        _object_digest(payload)
        return content


@dataclass(frozen=True)
class JointTransportEvent:
    capture_id: str
    action_id: int
    observation_epoch: int
    recipient_id: str
    arrival_epoch: int
    delivered: bool
    law_digest: str
    _registry_token: object
    _seal: object


def issue_joint_transport_event(*, registry: JointContextRegistry, capture_id: str,
        action_id: int, observation_epoch: int, recipient_id: str, arrival_epoch: int,
        delivered: bool, law_digest: str) -> JointTransportEvent:
    """Trusted new-mode capture adapter only; not an invented delivery receipt."""
    if type(registry) is not JointContextRegistry:
        raise JointReceiptError("UNKNOWN_CONTEXT_REGISTRY")
    _identifier(capture_id, "CAPTURE")
    _identifier(recipient_id, "RECIPIENT")
    _integer(action_id, "ACTION")
    _integer(observation_epoch, "CAPTURE_EPOCH")
    _integer(arrival_epoch, "ARRIVAL_EPOCH")
    _digest(law_digest)
    if type(delivered) is not bool or arrival_epoch < observation_epoch:
        raise JointReceiptError("UNKNOWN_TRANSPORT_CONTRACT")
    registry._get(action_id, observation_epoch, law_digest)
    event = JointTransportEvent(capture_id, action_id, observation_epoch,
        recipient_id, arrival_epoch, delivered, law_digest, registry._token, _TRANSPORT_SEAL)
    _TRANSPORTS[id(event)] = event
    return event


@dataclass(frozen=True)
class JointReceivedSet:
    payload: Any
    recipient_id: str
    admission_epoch: int
    arrival_epoch: int
    receipt_id: str
    law_digest: str
    _seal: object


@dataclass(frozen=True)
class JointBatch:
    current_epoch: int
    recipient_id: str
    receipts: tuple[JointReceivedSet, ...]
    status: str
    reason: str | None
    duplicate_count: int
    queue_version: int
    _contexts: tuple[_RegisteredContext, ...]
    _queue_token: object
    _owner_ref: Any
    _seal: object

    @property
    def epoch(self) -> int:
        return self.current_epoch

    def law_for(self, receipt: JointReceivedSet) -> Any:
        if (_BATCHES.get(id(self)) is not self or self._seal is not _BATCH_SEAL
                or self.status != "passed"):
            raise JointReceiptError("UNKNOWN_UNISSUED_OR_FAILED_BATCH")
        for item, context in zip(self.receipts, self._contexts):
            if item is receipt:
                if (_RECEIPTS.get(id(item)) is not item or item._seal is not _RECEIPT_SEAL
                        or _object_digest(context.law) != context.law_digest):
                    raise JointReceiptError("UNKNOWN_RECEIPT_CONTEXT")
                return context.law
        raise JointReceiptError("UNKNOWN_RECEIPT_NOT_IN_BATCH")

    def audit(self) -> dict[str, Any]:
        return {"schema": "joint_set_receiver_batch/v1", "epoch": self.current_epoch,
            "recipient_id": self.recipient_id, "status": self.status,
            "reason": self.reason, "received_set_count": len(self.receipts),
            "duplicate_count": self.duplicate_count,
            "receipt_ids": [r.receipt_id for r in self.receipts],
            "law_digests": [r.law_digest for r in self.receipts],
            "update_kind": ("UNKNOWN_NO_UPDATE" if self.status != "passed"
                else "NO_PACKET" if not self.receipts else "RECEIVED_JOINT_RETURN_SETS"),
            "late_likelihood_support_owned_by_graph": True,
            "wire_serialization_claimed": False}


@dataclass(frozen=True)
class _Pending:
    event: JointTransportEvent
    payload: Any


@dataclass(frozen=True)
class _Prepared:
    batch: JointBatch
    due_indices: tuple[int, ...]
    new_consumed: Mapping[str, str]
    due_digests: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _QueueState:
    pending: tuple[_Pending, ...]
    consumed: Mapping[str, str]
    version: int
    prepared: _Prepared | None
    last_epoch: int = -1


@dataclass(frozen=True)
class _QueuePublication:
    owner: Any
    batch: JointBatch
    expected_state: _QueueState
    next_state: _QueueState
    _seal: object


class JointReceiptQueue:
    """Bounded receiver inbox; no payload is inspected before actual arrival.

    Late arrival is admissible here.  The graph must separately validate its
    capture-time history and joint law.  A late unsupported likelihood is
    UNKNOWN, never a no-packet or missed-detection event.
    """

    def __init__(self, recipient_id: str, registry: JointContextRegistry, *,
                 max_pending: int, max_consumed: int):
        self.recipient_id = _identifier(recipient_id, "RECIPIENT")
        if type(registry) is not JointContextRegistry:
            raise JointReceiptError("UNKNOWN_CONTEXT_REGISTRY")
        self._registry = registry
        self.max_pending = _positive(max_pending, "PENDING_CAPACITY")
        self.max_consumed = _positive(max_consumed, "CONSUMED_CAPACITY")
        self._token = object()
        self._state = _QueueState((), MappingProxyType({}), 0, None)

    def enqueue(self, event: JointTransportEvent, payload: Any) -> None:
        if (type(event) is not JointTransportEvent or event._seal is not _TRANSPORT_SEAL
                or _TRANSPORTS.get(id(event)) is not event
                or event._registry_token is not self._registry._token):
            raise JointReceiptError("UNKNOWN_UNISSUED_JOINT_TRANSPORT")
        if event.recipient_id != self.recipient_id:
            raise JointReceiptError("UNKNOWN_WRONG_RECIPIENT")
        state = self._state
        if state.prepared is not None:
            raise JointReceiptError("UNKNOWN_UNCOMMITTED_JOINT_BATCH")
        if not event.delivered:
            return
        if len(state.pending) >= self.max_pending:
            raise JointReceiptError("UNKNOWN_PENDING_CAPACITY")
        # Hold a capture-owned immutable packet reference.  Do not call its
        # methods, deepcopy, hash, or validate it on a future/lost path.  A
        # malformed packet becomes receiver-visible only when it is due.
        self._state = _QueueState(state.pending + (_Pending(event, payload),),
            state.consumed, state.version + 1, None, state.last_epoch)

    def prepare(self, *, epoch: int) -> JointBatch:
        _integer(epoch, "RECEIVER_EPOCH")
        state = self._state
        if state.prepared is not None:
            raise JointReceiptError("UNKNOWN_UNCOMMITTED_JOINT_BATCH")
        if epoch < state.last_epoch:
            raise JointReceiptError("UNKNOWN_RECEIVER_TIME_REVERSAL")
        due = tuple(i for i, item in enumerate(state.pending) if item.event.arrival_epoch <= epoch)
        receipts, contexts, digests = [], [], []
        staged: dict[str, str] = {}
        duplicate_count = 0
        reason = None
        try:
            for index in due:
                item = state.pending[index]
                event = item.event
                context = self._registry._get(event.action_id, event.observation_epoch, event.law_digest)
                digest = self._registry._payload_digest(item.payload, event)
                digests.append((index, digest))
                previous = staged.get(event.capture_id, state.consumed.get(event.capture_id))
                if previous is not None:
                    if previous != digest:
                        raise JointReceiptError("UNKNOWN_CAPTURE_PROVENANCE_CONFLICT")
                    duplicate_count += 1
                    continue
                if len(state.consumed) + len(staged) >= self.max_consumed:
                    raise JointReceiptError("UNKNOWN_CONSUMED_CAPACITY")
                staged[event.capture_id] = digest
                receipt = JointReceivedSet(item.payload, self.recipient_id, epoch,
                    event.arrival_epoch, "joint_received:" + self.recipient_id + ":"
                    + event.capture_id + ":" + digest, event.law_digest, _RECEIPT_SEAL)
                _RECEIPTS[id(receipt)] = receipt
                receipts.append(receipt)
                contexts.append(context)
        except (ValueError, AttributeError, TypeError, OverflowError) as error:
            reason = str(error) if isinstance(error, JointReceiptError) else "UNKNOWN_RECEIVED_PAYLOAD_SCHEMA"
            receipts, contexts, staged, digests = [], [], {}, []
            duplicate_count = 0
        batch = JointBatch(epoch, self.recipient_id, tuple(receipts),
            "passed" if reason is None else "UNKNOWN", reason, duplicate_count,
            state.version + 1, tuple(contexts), self._token, weakref.ref(self), _BATCH_SEAL)
        _BATCHES[id(batch)] = batch
        prepared = _Prepared(batch, due, MappingProxyType(staged), tuple(digests))
        self._state = _QueueState(state.pending, state.consumed, state.version + 1, prepared, epoch)
        return batch

    def _stage_commit(self, batch: JointBatch) -> _QueuePublication:
        state = self._state
        prepared = state.prepared
        if (type(batch) is not JointBatch or batch._seal is not _BATCH_SEAL
                or _BATCHES.get(id(batch)) is not batch or batch._queue_token is not self._token
                or prepared is None or prepared.batch is not batch
                or batch.queue_version != state.version or batch.recipient_id != self.recipient_id):
            raise JointReceiptError("UNKNOWN_JOINT_BATCH_OWNERSHIP_OR_VERSION")
        if batch.status != "passed":
            raise JointReceiptError(batch.reason or "UNKNOWN_JOINT_BATCH")
        for index, old_digest in prepared.due_digests:
            item = state.pending[index]
            self._registry._get(item.event.action_id, item.event.observation_epoch, item.event.law_digest)
            if self._registry._payload_digest(item.payload, item.event) != old_digest:
                raise JointReceiptError("UNKNOWN_RECEIVED_PAYLOAD_CHANGED_AFTER_PREPARE")
        consumed = dict(state.consumed)
        consumed.update(prepared.new_consumed)
        removed = set(prepared.due_indices)
        pending = tuple(item for i, item in enumerate(state.pending) if i not in removed)
        next_state = _QueueState(pending, MappingProxyType(consumed), state.version + 1, None,
            state.last_epoch)
        plan = _QueuePublication(self, batch, state, next_state, _PLAN_SEAL)
        _PLANS[id(plan)] = plan
        return plan

    def abort(self, batch: JointBatch) -> None:
        state = self._state
        if state.prepared is None or state.prepared.batch is not batch:
            raise JointReceiptError("UNKNOWN_JOINT_BATCH_OWNERSHIP")
        self._state = _QueueState(state.pending, state.consumed, state.version + 1, None,
            state.last_epoch)


def validate_joint_batch(batch: JointBatch, *, epoch: int, recipient_id: str) -> JointBatch:
    """Graph-side validation of issued, arrived new-mode receipt identity."""
    _integer(epoch, "RECEIVER_EPOCH")
    _identifier(recipient_id, "RECIPIENT")
    if (type(batch) is not JointBatch or batch._seal is not _BATCH_SEAL
            or _BATCHES.get(id(batch)) is not batch or batch.status != "passed"
            or batch.current_epoch != epoch or batch.recipient_id != recipient_id):
        raise JointReceiptError("UNKNOWN_UNISSUED_OR_MISMATCHED_JOINT_BATCH")
    owner = batch._owner_ref()
    if (type(owner) is not JointReceiptQueue or owner._token is not batch._queue_token
            or owner._state.prepared is None or owner._state.prepared.batch is not batch
            or owner._state.version != batch.queue_version):
        raise JointReceiptError("UNKNOWN_JOINT_BATCH_ALREADY_CONSUMED_OR_ABORTED")
    for receipt in batch.receipts:
        if (type(receipt) is not JointReceivedSet or receipt._seal is not _RECEIPT_SEAL
                or _RECEIPTS.get(id(receipt)) is not receipt
                or receipt.recipient_id != recipient_id or receipt.admission_epoch != epoch
                or not receipt.payload.observation_epoch <= receipt.arrival_epoch <= epoch):
            raise JointReceiptError("UNKNOWN_JOINT_RECEIPT_IDENTITY_OR_ARRIVAL")
        batch.law_for(receipt)
    return batch


def commit_joint_batch(queue: JointReceiptQueue, graph: Any, batch: JointBatch) -> Any:
    """Stage both owners, then atomically publish two prebuilt state references.

    Required graph protocol:
      prepare_joint_batch(batch) -> graph-issued opaque stage;
      validate_joint_stage(stage, batch) -> EXACT dict {'_joint_state': state}.
    Both methods are read-only with respect to published graph/queue state.
    Validation binds stage owner, old state/version and complete batch/context
    evidence.  The sole publication target is a normal instance _joint_state
    field, not a descriptor. No publish method or user callback is invoked.

    Prepare/validation failure leaves the queue prepared for explicit abort.
    Publication exceptions restore both old references; computation charges
    remain spent.  This is in-process single-thread atomicity, not durable crash
    recovery and not protection against arbitrary malicious Python methods.
    """
    if type(queue) is not JointReceiptQueue:
        raise JointReceiptError("UNKNOWN_JOINT_QUEUE_TYPE")
    validate_joint_batch(batch, epoch=batch.current_epoch, recipient_id=queue.recipient_id)
    queue_plan = queue._stage_commit(batch)
    namespace = object.__getattribute__(graph, "__dict__")
    if type(namespace) is not dict or "_joint_state" not in namespace:
        raise JointReceiptError("UNKNOWN_GRAPH_JOINT_PUBLICATION_FIELD")
    for cls in type(graph).__mro__:
        descriptor = vars(cls).get("_joint_state")
        if descriptor is not None and hasattr(descriptor, "__set__"):
            raise JointReceiptError("UNKNOWN_GRAPH_JOINT_PUBLICATION_DESCRIPTOR")
    old_graph_state = namespace["_joint_state"]
    old_queue_state = queue._state
    try:
        stage = graph.prepare_joint_batch(batch)
        if (object.__getattribute__(graph, "__dict__")["_joint_state"] is not old_graph_state
                or queue._state is not old_queue_state):
            raise JointReceiptError("UNKNOWN_PUBLISHED_STATE_CHANGED_DURING_STAGING")
        # Recheck receiver digests after graph staging, before either publishes.
        queue_plan = queue._stage_commit(batch)
        if (queue_plan.owner is not queue or queue_plan.batch is not batch
                or queue_plan._seal is not _PLAN_SEAL or _PLANS.get(id(queue_plan)) is not queue_plan
                or queue_plan.expected_state is not old_queue_state):
            raise JointReceiptError("UNKNOWN_QUEUE_PUBLICATION_PLAN")
        # The graph owner/version check is last, after all receiver hashing.
        publication = graph.validate_joint_stage(stage, batch)
        if type(publication) is not dict or set(publication) != {"_joint_state"}:
            raise JointReceiptError("UNKNOWN_GRAPH_JOINT_PUBLICATION_MAPPING")
        if (object.__getattribute__(graph, "__dict__")["_joint_state"] is not old_graph_state
                or queue._state is not old_queue_state):
            raise JointReceiptError("UNKNOWN_PUBLISHED_STATE_CHANGED_DURING_VALIDATION")
        next_graph_state = publication["_joint_state"]
        next_queue_state = queue_plan.next_state
        # Commit point: no numeric work, allocation, hash, validation or callback.
        object.__setattr__(graph, "_joint_state", next_graph_state)
        object.__setattr__(queue, "_state", next_queue_state)
    except BaseException:
        object.__setattr__(graph, "_joint_state", old_graph_state)
        object.__setattr__(queue, "_state", old_queue_state)
        raise
    return stage
