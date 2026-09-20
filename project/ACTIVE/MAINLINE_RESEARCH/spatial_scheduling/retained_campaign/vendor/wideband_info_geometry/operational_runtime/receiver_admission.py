from __future__ import annotations

"""Receiver-local arrival boundary, separate from capture and posterior authority.

The opaque seals prevent accidental raw-record or metadata-based admission; they
are not a security boundary against Python code with access to this module.
Pending payloads are private to the runtime queue, never inserted into the graph.
This first revision explicitly rejects late association, rather than condition a
capture-time likelihood on a graph posterior that may already include it.
"""

from dataclasses import dataclass, field
import copy
import hashlib
import json
import weakref
from typing import Any

from .contracts import to_builtin
from .measurement_return_sets import UnlabelledReturnSet


_CAPTURE_SEAL = object()
_RECEIVER_SEAL = object()
_ISSUED_TRANSPORTS = weakref.WeakValueDictionary()
_ISSUED_RECEIPTS = weakref.WeakValueDictionary()
_ISSUED_BATCHES = weakref.WeakValueDictionary()


class ReceiverAdmissionError(ValueError):
    pass


def _epoch(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ReceiverAdmissionError("UNKNOWN_NONINTEGER_EPOCH_OR_BOUND")
    return value


def _identifier(value: Any) -> str:
    if type(value) is not str or not value:
        raise ReceiverAdmissionError("UNKNOWN_IDENTIFIER")
    return value


@dataclass(frozen=True)
class CaptureTransportEvent:
    capture_id: str
    action_id: int
    observation_epoch: int
    recipient_id: str
    arrival_epoch: int
    delivered: bool
    factor_ids: tuple[str, ...] = ()
    factor_provenance: tuple[tuple[str, ...], ...] = ()
    _seal: object = field(default=None, repr=False, compare=False)


def issue_capture_transport_event(*, capture_id: str, action_id: int,
        observation_epoch: int, recipient_id: str, arrival_epoch: int,
        delivered: bool, factor_ids: tuple[str, ...] = (),
        factor_provenance: tuple[tuple[str, ...], ...] = ()) -> CaptureTransportEvent:
    """Capture adapter only: arguments come from the actual transport draw."""
    _identifier(capture_id)
    _identifier(recipient_id)
    _epoch(action_id)
    _epoch(observation_epoch)
    _epoch(arrival_epoch)
    if type(delivered) is not bool or arrival_epoch < observation_epoch:
        raise ReceiverAdmissionError("UNKNOWN_TRANSPORT_CONTRACT")
    if (type(factor_ids) is not tuple or type(factor_provenance) is not tuple
            or len(factor_ids) != len(factor_provenance)
            or any(type(p) is not tuple for p in factor_provenance)):
        raise ReceiverAdmissionError("UNKNOWN_FACTOR_DISPATCH_CONTRACT")
    for name in (*factor_ids, *(p for row in factor_provenance for p in row)):
        _identifier(name)
    result = CaptureTransportEvent(str(capture_id), int(action_id),
        int(observation_epoch), str(recipient_id), int(arrival_epoch),
        bool(delivered), factor_ids, factor_provenance, _CAPTURE_SEAL)
    _ISSUED_TRANSPORTS[id(result)] = result
    return result


@dataclass(frozen=True)
class AdmittedReturnSet:
    payload: UnlabelledReturnSet
    recipient_id: str
    admission_epoch: int
    receipt_id: str
    _seal: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CapturePayloadFailure:
    """Capture-side failure, visible to a recipient only after actual arrival."""
    offline_reason: str


def validate_admitted_return_set(item: Any, *, epoch: int, recipient_id: str) -> UnlabelledReturnSet:
    _epoch(epoch)
    _identifier(recipient_id)
    if (not isinstance(item, AdmittedReturnSet) or item._seal is not _RECEIVER_SEAL
            or _ISSUED_RECEIPTS.get(id(item)) is not item):
        raise ReceiverAdmissionError("UNKNOWN_UNISSUED_RECEIPT")
    if item.admission_epoch != int(epoch):
        raise ReceiverAdmissionError("UNKNOWN_RECEIPT_EPOCH")
    if item.recipient_id != recipient_id:
        raise ReceiverAdmissionError("UNKNOWN_WRONG_RECIPIENT")
    if item.payload.observation_epoch != int(epoch):
        raise ReceiverAdmissionError("UNKNOWN_LATE_ASSOCIATION")
    if not item.payload.complete:
        raise ReceiverAdmissionError("UNKNOWN_INCOMPLETE_RETURN_SET")
    return item.payload


@dataclass(frozen=True)
class ReceiverBatch:
    epoch: int
    recipient_id: str
    admitted_sets: tuple[AdmittedReturnSet, ...]
    observations: tuple[Any, ...]
    factors: tuple[Any, ...]
    status: str
    reason: str | None
    receipt_ids: tuple[str, ...]
    _queue_token: object = field(repr=False, compare=False)
    _batch_id: int = field(repr=False)
    _seal: object = field(default=None, repr=False, compare=False)

    def audit(self) -> dict[str, Any]:
        # No pending/lost payload hashes or values appear in a receiver audit.
        return {"schema": "receiver_admission/v1", "epoch": self.epoch,
            "recipient_id": self.recipient_id, "status": self.status,
            "reason": self.reason, "received_capture_count": len(self.admitted_sets),
            "received_factor_count": len(self.factors),
            "receipt_ids": list(self.receipt_ids),
            "late_association_supported": False,
            "physical_wire_serialization_claimed": False}


def validate_receiver_batch(batch: Any, *, epoch: int, recipient_id: str) -> ReceiverBatch:
    _epoch(epoch)
    _identifier(recipient_id)
    if (not isinstance(batch, ReceiverBatch) or batch._seal is not _RECEIVER_SEAL
            or _ISSUED_BATCHES.get(id(batch)) is not batch):
        raise ReceiverAdmissionError("UNKNOWN_UNISSUED_RECEIVER_BATCH")
    if batch.epoch != int(epoch) or batch.status != "passed":
        raise ReceiverAdmissionError(batch.reason or "UNKNOWN_RECEIVER_BATCH")
    if batch.recipient_id != recipient_id:
        raise ReceiverAdmissionError("UNKNOWN_WRONG_RECIPIENT")
    for item in batch.admitted_sets:
        validate_admitted_return_set(item, epoch=epoch, recipient_id=recipient_id)
    return batch


@dataclass(frozen=True)
class _PendingCapture:
    transport: CaptureTransportEvent
    payload: Any
    observation: Any
    factors: tuple[Any, ...]


def _received_digest(item: _PendingCapture) -> str:
    body = {"payload": to_builtin(item.payload),
        "observation": to_builtin(item.observation),
        "factors": to_builtin(item.factors)}
    serialized = json.dumps(body, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ReceiverAdmissionQueue:
    """Finite receiver-local inbox; successful batches require explicit commit.

    A complete empty return set is an admitted measurement event. An empty inbox
    is NO_PACKET and is never converted to a missed-detection likelihood. Bounds
    fail closed instead of silently evicting duplicate/provenance history.
    """

    def __init__(self, recipient_id: str, *, lag_epochs: int,
            max_pending: int, max_consumed: int):
        _identifier(recipient_id)
        for bound in (lag_epochs, max_pending, max_consumed):
            _epoch(bound)
        if min(lag_epochs, max_pending, max_consumed) < 1:
            raise ValueError("receiver queue bounds must be positive")
        self.recipient_id = str(recipient_id)
        self.lag_epochs = int(lag_epochs)
        self.max_pending = int(max_pending)
        self.max_consumed = int(max_consumed)
        self._pending: list[_PendingCapture] = []
        self._consumed: dict[str, str] = {}
        self._token = object()
        self._next_batch = 0
        self._prepared: tuple[ReceiverBatch, tuple[int, ...], dict[str, str]] | None = None

    def enqueue(self, transport: CaptureTransportEvent, payload: UnlabelledReturnSet,
            observation: Any, factors: tuple[Any, ...]) -> None:
        if (not isinstance(transport, CaptureTransportEvent) or transport._seal is not _CAPTURE_SEAL
                or _ISSUED_TRANSPORTS.get(id(transport)) is not transport):
            raise ReceiverAdmissionError("UNKNOWN_UNISSUED_TRANSPORT_EVENT")
        if transport.recipient_id != self.recipient_id:
            raise ReceiverAdmissionError("UNKNOWN_WRONG_RECIPIENT")
        if self._prepared is not None:
            raise ReceiverAdmissionError("UNKNOWN_UNCOMMITTED_RECEIVER_BATCH")
        if not transport.delivered:
            return
        if len(self._pending) >= self.max_pending:
            raise ReceiverAdmissionError("UNKNOWN_RECEIVER_CAPACITY")
        # Do not hash/validate unavailable payload contents here. Their values,
        # including malformed fields, cannot influence the receiver before arrival.
        self._pending.append(_PendingCapture(transport, copy.deepcopy(payload),
            copy.deepcopy(observation), copy.deepcopy(tuple(factors))))

    def prepare(self, *, epoch: int) -> ReceiverBatch:
        _epoch(epoch)
        if self._prepared is not None:
            raise ReceiverAdmissionError("UNKNOWN_UNCOMMITTED_RECEIVER_BATCH")
        due = [(i, item) for i, item in enumerate(self._pending)
            if item.transport.arrival_epoch <= int(epoch)]
        admitted: list[AdmittedReturnSet] = []
        observations: list[Any] = []
        factors: list[Any] = []
        staged: dict[str, str] = {}
        reason = None
        for _, item in due:
            event = item.transport
            payload = item.payload
            if isinstance(payload, CapturePayloadFailure):
                reason = "UNKNOWN_RECEIVED_CAPTURE_PAYLOAD"
                break
            if not isinstance(payload, UnlabelledReturnSet) or not payload.complete:
                reason = "UNKNOWN_INCOMPLETE_RETURN_SET"
                break
            if payload.action_id != event.action_id or payload.observation_epoch != event.observation_epoch:
                reason = "UNKNOWN_CAPTURE_IDENTITY"
                break
            observation = item.observation
            if (payload.provenance_id != event.capture_id
                    or getattr(observation, "provenance_id", None) != event.capture_id
                    or getattr(observation, "action_id", None) != event.action_id
                    or getattr(observation, "observation_epoch", None) != event.observation_epoch
                    or getattr(observation, "arrival_epoch", None) != event.arrival_epoch
                    or tuple(getattr(f, "factor_id", None) for f in item.factors) != event.factor_ids
                    or tuple(tuple(getattr(f, "provenance_ids", ())) for f in item.factors) != event.factor_provenance
                    or any(getattr(f, "observation_epoch", None) != event.observation_epoch
                        or getattr(f, "arrival_epoch", None) != event.arrival_epoch for f in item.factors)):
                reason = "UNKNOWN_CAPTURE_DISPATCH_BINDING"
                break
            try:
                digest = _received_digest(item)
            except (TypeError, ValueError, OverflowError):
                reason = "UNKNOWN_INVALID_RECEIVED_PAYLOAD"
                break
            previous = staged.get(event.capture_id, self._consumed.get(event.capture_id))
            if previous is not None:
                if digest != previous:
                    reason = "UNKNOWN_RECEIPT_CONFLICT"
                    break
                continue
            if int(epoch) - event.observation_epoch >= self.lag_epochs:
                reason = "UNKNOWN_EXPIRED_CAPTURE"
                break
            if event.observation_epoch != int(epoch):
                reason = "UNKNOWN_LATE_ASSOCIATION"
                break
            if len(self._consumed) + len(staged) >= self.max_consumed:
                reason = "UNKNOWN_RECEIPT_CAPACITY"
                break
            staged[event.capture_id] = digest
            receipt_id = f"received:{self.recipient_id}:{event.capture_id}:{digest}"
            admitted_item = AdmittedReturnSet(payload, self.recipient_id,
                int(epoch), receipt_id, _RECEIVER_SEAL)
            _ISSUED_RECEIPTS[id(admitted_item)] = admitted_item
            admitted.append(admitted_item)
            observations.append(item.observation)
            factors.extend(item.factors)
        # Atomic precheck: no partial received payloads escape on failure.
        if reason is not None:
            admitted, observations, factors, staged = [], [], [], {}
        batch = ReceiverBatch(int(epoch), self.recipient_id, tuple(admitted),
            tuple(observations), tuple(factors), "passed" if reason is None else "UNKNOWN",
            reason, tuple(a.receipt_id for a in admitted), self._token,
            self._next_batch, _RECEIVER_SEAL)
        self._next_batch += 1
        _ISSUED_BATCHES[id(batch)] = batch
        self._prepared = (batch, tuple(i for i, _ in due), staged)
        return batch

    def commit(self, batch: ReceiverBatch) -> None:
        if self._prepared is None or self._prepared[0] is not batch or batch._queue_token is not self._token:
            raise ReceiverAdmissionError("UNKNOWN_BATCH_OWNERSHIP")
        validate_receiver_batch(batch, epoch=batch.epoch, recipient_id=self.recipient_id)
        _, indices, staged = self._prepared
        removed = set(indices)
        self._pending = [item for i, item in enumerate(self._pending) if i not in removed]
        self._consumed.update(staged)
        self._prepared = None

    def abort(self, batch: ReceiverBatch) -> None:
        if self._prepared is None or self._prepared[0] is not batch:
            raise ReceiverAdmissionError("UNKNOWN_BATCH_OWNERSHIP")
        self._prepared = None

    def reject(self, *, epoch: int, reason: str) -> ReceiverBatch:
        """Represent a capture contract failure after a physical commitment."""
        _epoch(epoch)
        _identifier(reason)
        if self._prepared is not None:
            raise ReceiverAdmissionError("UNKNOWN_UNCOMMITTED_RECEIVER_BATCH")
        batch = ReceiverBatch(epoch, self.recipient_id, (), (), (), "UNKNOWN",
            reason, (), self._token, self._next_batch, _RECEIVER_SEAL)
        self._next_batch += 1
        _ISSUED_BATCHES[id(batch)] = batch
        self._prepared = (batch, (), {})
        return batch
