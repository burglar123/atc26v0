"""V4D ST-Spec cross-batch draft payload mailbox scaffold.

The mailbox is intentionally a local diagnostic data structure. Draft and target
model runners currently live in separate Python processes, so this module does
not pretend that a normal Python object can transport payloads between them.
V4D uses the mailbox to make routing intent explicit, detect warmup/missing
payloads by ``home_batch_id`` and ``seq_id``, and produce JSON-serializable
trace context. Full end-to-end two-batch execution still requires a real
cross-process mailbox transport, target-side consume wiring, and KV/state
synchronization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from nano_pearl.pearl_engine.pearl_protocol import PearlDraftMessage


JsonDict = dict[str, Any]


class STSpecMailboxError(RuntimeError):
    """Mailbox routing failure with structured diagnostic context."""

    def __init__(self, message: str, *, kind: str, context: JsonDict | None = None):
        self.kind = str(kind)
        self.context = _jsonable(context or {})
        super().__init__(f"{message}; mailbox_error_kind={self.kind}; context={self.context}")


@dataclass(frozen=True, order=True)
class MailboxKey:
    home_batch_id: int | str | None
    seq_id: int

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MailboxPayloadLifecycle:
    payload_id: str
    home_batch_id: int | str | None
    seq_id: int
    source_plan_id: int | None
    source_draft_home_batch_id: int | str | None
    consumed_by_plan_id: int | None = None
    invalidated_by_plan_id: int | None = None
    consumed_token_count: int = 0
    invalidated_token_count: int = 0
    lifecycle_state: str = "available"

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MailboxPayload:
    plan_id: int | None
    producer_role: str | None
    producer_home_batch_id: int | str | None
    target_home_batch_id: int | str | None
    draft_home_batch_id: int | str | None
    seq_id: int
    request_id: Any
    home_batch_id: int | str | None
    gamma: int
    layout_kind: str
    protocol_version: int
    draft_token_ids: list[int] = field(default_factory=list)
    per_seq_length: int = 0
    offset: int = 0
    logical_step: int | None = None
    producer_actual_exec_seq_ids: list[int] = field(default_factory=list)
    producer_draft_message_seq_ids: list[int] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @property
    def key(self) -> MailboxKey:
        return MailboxKey(self.home_batch_id, int(self.seq_id))

    @property
    def payload_id(self) -> str:
        payload_id = self.metadata.get("payload_id") if isinstance(self.metadata, dict) else None
        if payload_id is not None:
            return str(payload_id)
        return f"{self.home_batch_id}:{self.seq_id}:{self.offset}:{self.per_seq_length}"

    def to_dict(self) -> JsonDict:
        data = _jsonable(asdict(self))
        data["payload_id"] = self.payload_id
        return data


@dataclass(frozen=True)
class MailboxGetResult:
    home_batch_id: int | str | None
    requested_seq_ids: list[int]
    payloads: list[MailboxPayload]
    missing_seq_ids: list[int]
    plan_id: int | None = None
    consumer_role: str | None = None

    @property
    def hit_count(self) -> int:
        return len(self.payloads)

    @property
    def miss_count(self) -> int:
        return len(self.missing_seq_ids)

    @property
    def success(self) -> bool:
        return self.miss_count == 0

    def to_trace_dict(self) -> JsonDict:
        return {
            "home_batch_id": self.home_batch_id,
            "requested_seq_ids": list(self.requested_seq_ids),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "missing_seq_ids": list(self.missing_seq_ids),
            "payload_seq_ids": [payload.seq_id for payload in self.payloads],
            "plan_id": self.plan_id,
            "consumer_role": self.consumer_role,
        }


class STSpecPayloadMailbox:
    """In-process mailbox keyed by ``(home_batch_id, seq_id)``.

    Duplicate puts fail explicitly by default so real-probe diagnostics cannot
    silently overwrite a draft payload. ``overwrite=True`` can be used by future
    transport code if a replay/update policy is deliberately chosen.
    """

    def __init__(self, *, overwrite: bool = False):
        self.overwrite = bool(overwrite)
        self._payloads: dict[MailboxKey, MailboxPayload] = {}
        self._lifecycle_by_key: dict[MailboxKey, MailboxPayloadLifecycle] = {}
        self._lifecycle_by_payload_id: dict[str, MailboxPayloadLifecycle] = {}
        self._put_count = 0
        self._duplicate_put_count = 0
        self._pop_count = 0

    def put_payloads(
        self,
        home_batch_id: int | str | None,
        payloads: Iterable[MailboxPayload],
        *,
        plan_id: int | None = None,
        producer_role: str | None = None,
    ) -> list[MailboxKey]:
        keys: list[MailboxKey] = []
        for payload in payloads:
            if payload.home_batch_id != home_batch_id:
                raise STSpecMailboxError(
                    "ST-Spec mailbox payload home_batch_id does not match put home_batch_id",
                    kind="mailbox_home_batch_mismatch",
                    context={
                        "put_home_batch_id": home_batch_id,
                        "payload_home_batch_id": payload.home_batch_id,
                        "seq_id": payload.seq_id,
                        "plan_id": plan_id,
                        "producer_role": producer_role,
                    },
                )
            key = payload.key
            if key in self._payloads and not self.overwrite:
                self._duplicate_put_count += 1
                raise STSpecMailboxError(
                    "Duplicate ST-Spec mailbox payload put",
                    kind="duplicate_put",
                    context={
                        "home_batch_id": home_batch_id,
                        "seq_id": payload.seq_id,
                        "plan_id": plan_id,
                        "producer_role": producer_role,
                    },
                )
            self._payloads[key] = payload
            lifecycle = MailboxPayloadLifecycle(
                payload_id=payload.payload_id,
                home_batch_id=payload.home_batch_id,
                seq_id=int(payload.seq_id),
                source_plan_id=payload.plan_id,
                source_draft_home_batch_id=payload.draft_home_batch_id,
                lifecycle_state="available",
            )
            self._lifecycle_by_key[key] = lifecycle
            self._lifecycle_by_payload_id[payload.payload_id] = lifecycle
            self._put_count += 1
            keys.append(key)
        return keys

    def get_payloads(
        self,
        home_batch_id: int | str | None,
        seq_ids: Iterable[int],
        *,
        plan_id: int | None = None,
        consumer_role: str | None = None,
    ) -> MailboxGetResult:
        requested = [int(seq_id) for seq_id in seq_ids]
        payloads: list[MailboxPayload] = []
        missing: list[int] = []
        for seq_id in requested:
            key = MailboxKey(home_batch_id, seq_id)
            payload = self._payloads.get(key)
            lifecycle = self._lifecycle_by_key.get(key)
            if payload is None or lifecycle is None or lifecycle.lifecycle_state != "available":
                missing.append(seq_id)
            else:
                payloads.append(payload)
        return MailboxGetResult(home_batch_id, requested, payloads, missing, plan_id, consumer_role)

    def has_payloads(self, home_batch_id: int | str | None, seq_ids: Iterable[int]) -> bool:
        return self.get_payloads(home_batch_id, seq_ids).success

    def peek_payloads(self, home_batch_id: int | str | None, seq_ids: Iterable[int]) -> MailboxGetResult:
        return self.get_payloads(home_batch_id, seq_ids)

    def pop_payloads(
        self,
        home_batch_id: int | str | None,
        seq_ids: Iterable[int],
        *,
        plan_id: int | None = None,
        consumer_role: str | None = None,
    ) -> MailboxGetResult:
        result = self.get_payloads(home_batch_id, seq_ids, plan_id=plan_id, consumer_role=consumer_role)
        if result.success:
            for seq_id in result.requested_seq_ids:
                key = MailboxKey(home_batch_id, seq_id)
                payload = self._payloads.pop(key, None)
                lifecycle = self._lifecycle_by_key.pop(key, None)
                if payload is not None:
                    self._lifecycle_by_payload_id.pop(payload.payload_id, None)
                elif lifecycle is not None:
                    self._lifecycle_by_payload_id.pop(lifecycle.payload_id, None)
                self._pop_count += 1
        return result

    def clear_finished(self, seq_ids: Iterable[int]) -> int:
        finished = {int(seq_id) for seq_id in seq_ids}
        keys = [key for key in self._payloads if key.seq_id in finished]
        for key in keys:
            payload = self._payloads.pop(key, None)
            lifecycle = self._lifecycle_by_key.pop(key, None)
            if payload is not None:
                self._lifecycle_by_payload_id.pop(payload.payload_id, None)
            elif lifecycle is not None:
                self._lifecycle_by_payload_id.pop(lifecycle.payload_id, None)
        return len(keys)

    def available_home_batch_ids(self) -> list[int | str | None]:
        return sorted(
            {key.home_batch_id for key, lifecycle in self._lifecycle_by_key.items() if lifecycle.lifecycle_state == "available"},
            key=lambda value: str(value),
        )

    def available_seq_ids_by_batch(self) -> dict[str, list[int]]:
        by_batch: dict[str, list[int]] = {}
        for key, lifecycle in self._lifecycle_by_key.items():
            if lifecycle.lifecycle_state == "available":
                by_batch.setdefault(str(key.home_batch_id), []).append(int(key.seq_id))
        return {batch: sorted(seq_ids) for batch, seq_ids in sorted(by_batch.items())}

    def lifecycle_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for lifecycle in self._lifecycle_by_payload_id.values():
            counts[lifecycle.lifecycle_state] = counts.get(lifecycle.lifecycle_state, 0) + 1
        return counts

    def lifecycle_snapshot(self, payload_ids: Iterable[str] | None = None) -> dict[str, JsonDict]:
        if payload_ids is None:
            items = self._lifecycle_by_payload_id.items()
        else:
            requested = [str(payload_id) for payload_id in payload_ids]
            items = ((payload_id, self._lifecycle_by_payload_id.get(payload_id)) for payload_id in requested)
        snapshot: dict[str, JsonDict] = {}
        for payload_id, lifecycle in items:
            if lifecycle is None:
                snapshot[str(payload_id)] = {"payload_id": str(payload_id), "lifecycle_state": "missing"}
            else:
                snapshot[str(payload_id)] = lifecycle.to_dict()
        return snapshot

    def restore_lifecycle_snapshot(self, snapshot: dict[str, JsonDict]) -> bool:
        try:
            for payload_id, state in snapshot.items():
                key = None
                for candidate, lifecycle in self._lifecycle_by_key.items():
                    if lifecycle.payload_id == payload_id:
                        key = candidate
                        break
                if state.get("lifecycle_state") == "missing":
                    if key is not None:
                        lifecycle = self._lifecycle_by_key.pop(key)
                        self._lifecycle_by_payload_id.pop(lifecycle.payload_id, None)
                    continue
                lifecycle = MailboxPayloadLifecycle(
                    payload_id=str(state.get("payload_id", payload_id)),
                    home_batch_id=state.get("home_batch_id"),
                    seq_id=int(state.get("seq_id")),
                    source_plan_id=state.get("source_plan_id"),
                    source_draft_home_batch_id=state.get("source_draft_home_batch_id"),
                    consumed_by_plan_id=state.get("consumed_by_plan_id"),
                    invalidated_by_plan_id=state.get("invalidated_by_plan_id"),
                    consumed_token_count=int(state.get("consumed_token_count") or 0),
                    invalidated_token_count=int(state.get("invalidated_token_count") or 0),
                    lifecycle_state=str(state.get("lifecycle_state", "available")),
                )
                if key is None:
                    key = MailboxKey(lifecycle.home_batch_id, lifecycle.seq_id)
                self._lifecycle_by_key[key] = lifecycle
                self._lifecycle_by_payload_id[lifecycle.payload_id] = lifecycle
            return True
        except Exception:
            return False

    def apply_payload_lifecycle(
        self,
        *,
        plan_id: int | None,
        target_home_batch_id: int | str | None,
        consumed_token_count_by_payload_id: dict[str, int],
        invalidated_token_count_by_payload_id: dict[str, int],
    ) -> dict[str, JsonDict]:
        payload_ids = sorted(set(consumed_token_count_by_payload_id) | set(invalidated_token_count_by_payload_id), key=str)
        updated: dict[str, JsonDict] = {}
        for payload_id in payload_ids:
            lifecycle = self._lifecycle_by_payload_id.get(str(payload_id))
            if lifecycle is None:
                raise STSpecMailboxError(
                    "Mailbox payload lifecycle missing during consume/invalidate",
                    kind="mailbox_payload_lifecycle_missing",
                    context={"payload_id": payload_id, "plan_id": plan_id},
                )
            if lifecycle.home_batch_id != target_home_batch_id:
                raise STSpecMailboxError(
                    "Mailbox payload lifecycle home_batch_id mismatch during consume/invalidate",
                    kind="mailbox_payload_lifecycle_home_batch_mismatch",
                    context={
                        "payload_id": payload_id,
                        "payload_home_batch_id": lifecycle.home_batch_id,
                        "target_home_batch_id": target_home_batch_id,
                        "plan_id": plan_id,
                    },
                )
            if lifecycle.lifecycle_state != "available":
                duplicate = MailboxPayloadLifecycle(
                    **{
                        **lifecycle.to_dict(),
                        "lifecycle_state": "duplicate_consume_error",
                    }
                )
                self._lifecycle_by_payload_id[str(payload_id)] = duplicate
                self._lifecycle_by_key[MailboxKey(duplicate.home_batch_id, duplicate.seq_id)] = duplicate
                raise STSpecMailboxError(
                    "Duplicate mailbox payload consume/invalidate",
                    kind="duplicate_consume_error",
                    context={"payload_id": payload_id, "plan_id": plan_id, "state": lifecycle.lifecycle_state},
                )
            consumed = int(consumed_token_count_by_payload_id.get(str(payload_id), 0) or 0)
            invalidated = int(invalidated_token_count_by_payload_id.get(str(payload_id), 0) or 0)
            new_state = "invalidated" if invalidated > 0 else "consumed"
            new_lifecycle = MailboxPayloadLifecycle(
                payload_id=lifecycle.payload_id,
                home_batch_id=lifecycle.home_batch_id,
                seq_id=lifecycle.seq_id,
                source_plan_id=lifecycle.source_plan_id,
                source_draft_home_batch_id=lifecycle.source_draft_home_batch_id,
                consumed_by_plan_id=plan_id if consumed > 0 else lifecycle.consumed_by_plan_id,
                invalidated_by_plan_id=plan_id if invalidated > 0 else lifecycle.invalidated_by_plan_id,
                consumed_token_count=consumed,
                invalidated_token_count=invalidated,
                lifecycle_state=new_state,
            )
            self._lifecycle_by_payload_id[str(payload_id)] = new_lifecycle
            self._lifecycle_by_key[MailboxKey(new_lifecycle.home_batch_id, new_lifecycle.seq_id)] = new_lifecycle
            updated[str(payload_id)] = new_lifecycle.to_dict()
        return updated

    def stats(self) -> JsonDict:
        return {
            "payload_count": len(self._payloads),
            "put_count": self._put_count,
            "pop_count": self._pop_count,
            "duplicate_put_count": self._duplicate_put_count,
            "lifecycle_counts": self.lifecycle_counts(),
            "available_home_batch_ids": self.available_home_batch_ids(),
            "available_seq_ids_by_batch": self.available_seq_ids_by_batch(),
        }


def payloads_from_draft_message(
    message: PearlDraftMessage,
    *,
    home_batch_id: int | str | None,
    target_home_batch_id: int | str | None = None,
    draft_home_batch_id: int | str | None = None,
    producer_home_batch_id: int | str | None = None,
    producer_role: str | None = None,
    logical_step: int | None = None,
    metadata: JsonDict | None = None,
) -> list[MailboxPayload]:
    """Convert a variable-offset draft envelope into per-sequence mailbox rows."""
    payloads: list[MailboxPayload] = []
    metadata = dict(metadata or {})
    for idx, seq_id in enumerate(message.seq_ids):
        offset = int(message.draft_offsets[idx])
        length = int(message.per_seq_draft_lengths[idx])
        request_id = message.request_ids[idx] if idx < len(message.request_ids) else None
        payloads.append(
            MailboxPayload(
                plan_id=message.plan_id,
                producer_role=producer_role or message.runner_role,
                producer_home_batch_id=producer_home_batch_id,
                target_home_batch_id=target_home_batch_id,
                draft_home_batch_id=draft_home_batch_id,
                seq_id=int(seq_id),
                request_id=request_id,
                home_batch_id=home_batch_id,
                gamma=int(message.gamma),
                layout_kind=message.layout_kind,
                protocol_version=int(message.protocol_version),
                draft_token_ids=list(message.draft_token_ids[offset : offset + length]),
                per_seq_length=length,
                offset=offset,
                logical_step=logical_step,
                producer_actual_exec_seq_ids=list(message.actual_exec_seq_ids),
                producer_draft_message_seq_ids=list(message.seq_ids),
                metadata=metadata,
            )
        )
    return payloads


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    return value
