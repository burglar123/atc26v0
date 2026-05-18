"""V4E ST-Spec mailbox transport envelopes.

V4E makes cross-process mailbox handoff visible as an explicit envelope with
validated sequence/offset metadata. The current implementation is intentionally a
scaffold: it can encode/decode JSON-safe payload envelopes and insert decoded
payloads into an in-process target mailbox, while runtime integration may still
report warmup scheduling or target consume wiring as later-stage blockers.
Final two-batch execution still requires: (1) cross-process envelope transport,
(2) pipeline warmup scheduling, (3) target verification input construction from
mailbox payloads, and (4) target/draft KV/state synchronization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable

from nano_pearl.pearl_engine.pearl_protocol import build_offsets, validate_offsets
from nano_pearl.pearl_engine.stspec_mailbox import MailboxPayload, STSpecPayloadMailbox


JsonDict = dict[str, Any]


class MailboxTransportMode(str, Enum):
    DIAGNOSTIC_ONLY = "diagnostic_only"
    EXISTING_BROADCAST = "existing_broadcast"
    SHARED_MEMORY = "shared_memory"
    MANAGER_QUEUE = "manager_queue"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True)
class MailboxTransportStatus:
    attempted: bool
    success: bool
    transport_mode: str
    payload_available: bool
    error: str | None = None
    error_kind: str | None = None
    seq_ids: list[int] = field(default_factory=list)
    home_batch_id: int | str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_trace_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MailboxTransportEnvelope:
    transport_version: int
    transport_mode: str
    plan_id: int | None
    producer_rank: int | None
    producer_role: str | None
    producer_home_batch_id: int | str | None
    target_home_batch_id: int | str | None
    draft_home_batch_id: int | str | None
    layout_kind: str
    protocol_version: int
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_lengths: list[int]
    offsets: list[int]
    total_tokens: int
    draft_token_ids: list[int] = field(default_factory=list)
    payload_metadata: JsonDict = field(default_factory=dict)
    produced_for_home_batch_id: int | str | None = None
    logical_step: int | None = None
    source_plan_signature_hash: str | None = None
    payload_available: bool = True

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


def normalize_transport_mode(mode: str | MailboxTransportMode) -> MailboxTransportMode:
    if isinstance(mode, MailboxTransportMode):
        return mode
    try:
        return MailboxTransportMode(str(mode))
    except ValueError as exc:
        raise ValueError(
            f"Invalid ST-Spec mailbox transport mode={mode!r}; expected "
            f"{[item.value for item in MailboxTransportMode]}"
        ) from exc


def encode_mailbox_transport_envelope(
    *,
    payloads: Iterable[MailboxPayload],
    transport_mode: str | MailboxTransportMode = MailboxTransportMode.DIAGNOSTIC_ONLY,
    transport_version: int = 1,
    plan_id: int | None = None,
    producer_rank: int | None = None,
    producer_role: str | None = None,
    producer_home_batch_id: int | str | None = None,
    target_home_batch_id: int | str | None = None,
    draft_home_batch_id: int | str | None = None,
    produced_for_home_batch_id: int | str | None = None,
    logical_step: int | None = None,
    source_plan_signature_hash: str | None = None,
    payload_available: bool = True,
    payload_metadata: JsonDict | None = None,
) -> MailboxTransportEnvelope:
    payload_list = list(payloads)
    mode = normalize_transport_mode(transport_mode)
    seq_ids = [int(payload.seq_id) for payload in payload_list]
    request_ids = [payload.request_id for payload in payload_list]
    lengths = [int(payload.per_seq_length) for payload in payload_list]
    offsets = build_offsets(lengths)
    draft_token_ids: list[int] = []
    if payload_available:
        for payload in payload_list:
            draft_token_ids.extend(int(token) for token in payload.draft_token_ids)
    first = payload_list[0] if payload_list else None
    envelope = MailboxTransportEnvelope(
        transport_version=int(transport_version),
        transport_mode=mode.value,
        plan_id=plan_id if plan_id is not None else (first.plan_id if first else None),
        producer_rank=producer_rank,
        producer_role=producer_role if producer_role is not None else (first.producer_role if first else None),
        producer_home_batch_id=producer_home_batch_id if producer_home_batch_id is not None else (first.producer_home_batch_id if first else None),
        target_home_batch_id=target_home_batch_id if target_home_batch_id is not None else (first.target_home_batch_id if first else None),
        draft_home_batch_id=draft_home_batch_id if draft_home_batch_id is not None else (first.draft_home_batch_id if first else None),
        layout_kind=first.layout_kind if first else "variable_offsets",
        protocol_version=int(first.protocol_version if first else 1),
        seq_ids=seq_ids,
        request_ids=request_ids,
        per_seq_lengths=lengths,
        offsets=offsets,
        total_tokens=sum(lengths),
        draft_token_ids=draft_token_ids,
        payload_metadata=dict(payload_metadata or {}),
        produced_for_home_batch_id=produced_for_home_batch_id if produced_for_home_batch_id is not None else (first.home_batch_id if first else None),
        logical_step=logical_step if logical_step is not None else (first.logical_step if first else None),
        source_plan_signature_hash=source_plan_signature_hash,
        payload_available=bool(payload_available),
    )
    validate_mailbox_transport_envelope(envelope)
    return envelope


def decode_mailbox_transport_envelope(payload: MailboxTransportEnvelope | str | JsonDict) -> MailboxTransportEnvelope:
    if isinstance(payload, MailboxTransportEnvelope):
        envelope = payload
    else:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
        envelope = MailboxTransportEnvelope(**data)
    validate_mailbox_transport_envelope(envelope)
    return envelope


def validate_mailbox_transport_envelope(envelope: MailboxTransportEnvelope) -> None:
    if int(envelope.transport_version) != 1:
        raise NotImplementedError(
            f"Unsupported ST-Spec mailbox transport_version={envelope.transport_version!r}; only version 1 is implemented."
        )
    normalize_transport_mode(envelope.transport_mode)
    if len(envelope.seq_ids) != len(envelope.per_seq_lengths):
        raise RuntimeError(
            "Invalid mailbox transport envelope; len(seq_ids) != len(per_seq_lengths): "
            f"seq_ids={envelope.seq_ids}, per_seq_lengths={envelope.per_seq_lengths}"
        )
    if envelope.request_ids and len(envelope.request_ids) != len(envelope.seq_ids):
        raise RuntimeError(
            "Invalid mailbox transport envelope; len(request_ids) != len(seq_ids): "
            f"request_ids={envelope.request_ids}, seq_ids={envelope.seq_ids}"
        )
    if len(set(envelope.seq_ids)) != len(envelope.seq_ids):
        raise RuntimeError(f"Invalid mailbox transport envelope; duplicate seq_ids={envelope.seq_ids}")
    validate_offsets(
        [int(length) for length in envelope.per_seq_lengths],
        [int(offset) for offset in envelope.offsets],
        int(envelope.total_tokens),
        message_type="mailbox_transport",
        layout_kind=envelope.layout_kind,
        seq_ids=[int(seq_id) for seq_id in envelope.seq_ids],
    )
    if envelope.payload_available and len(envelope.draft_token_ids) != int(envelope.total_tokens):
        raise RuntimeError(
            "Invalid mailbox transport envelope; draft_token_ids length does not match total_tokens: "
            f"payload_length={len(envelope.draft_token_ids)}, total_tokens={envelope.total_tokens}, "
            f"seq_ids={envelope.seq_ids}, offsets={envelope.offsets}"
        )


def envelope_to_mailbox_payloads(envelope: MailboxTransportEnvelope) -> list[MailboxPayload]:
    validate_mailbox_transport_envelope(envelope)
    payloads: list[MailboxPayload] = []
    for idx, seq_id in enumerate(envelope.seq_ids):
        offset = int(envelope.offsets[idx])
        length = int(envelope.per_seq_lengths[idx])
        request_id = envelope.request_ids[idx] if idx < len(envelope.request_ids) else None
        token_ids = list(envelope.draft_token_ids[offset : offset + length]) if envelope.payload_available else []
        payloads.append(
            MailboxPayload(
                plan_id=envelope.plan_id,
                producer_role=envelope.producer_role,
                producer_home_batch_id=envelope.producer_home_batch_id,
                target_home_batch_id=envelope.target_home_batch_id,
                draft_home_batch_id=envelope.draft_home_batch_id,
                seq_id=int(seq_id),
                request_id=request_id,
                home_batch_id=envelope.produced_for_home_batch_id,
                gamma=int(envelope.payload_metadata.get("gamma", 0) or 0),
                layout_kind=envelope.layout_kind,
                protocol_version=int(envelope.protocol_version),
                draft_token_ids=token_ids,
                per_seq_length=length,
                offset=offset,
                logical_step=envelope.logical_step,
                producer_actual_exec_seq_ids=list(envelope.seq_ids),
                producer_draft_message_seq_ids=list(envelope.seq_ids),
                metadata={
                    **dict(envelope.payload_metadata or {}),
                    "mailbox_transport_mode": envelope.transport_mode,
                    "mailbox_transport_digest": envelope.digest(),
                    "mailbox_transport_payload_available": envelope.payload_available,
                },
            )
        )
    return payloads


def insert_transport_envelope_into_mailbox(
    mailbox: STSpecPayloadMailbox,
    envelope: MailboxTransportEnvelope,
    *,
    consumer_role: str | None = None,
) -> int:
    payloads = envelope_to_mailbox_payloads(envelope)
    mailbox.put_payloads(
        envelope.produced_for_home_batch_id,
        payloads,
        plan_id=envelope.plan_id,
        producer_role=envelope.producer_role or consumer_role,
    )
    return len(payloads)


def classify_mailbox_miss(
    *,
    target_home_batch_id: int | str | None,
    available_home_batch_ids: Iterable[int | str | None],
    allow_warmup_miss: bool = False,
) -> tuple[str, str, bool]:
    """Return (error_kind, next_required_feature, warmup_miss)."""
    warmup_miss = target_home_batch_id not in set(available_home_batch_ids)
    if warmup_miss:
        if allow_warmup_miss:
            return "mailbox_warmup_skip_not_implemented", "pipeline_warmup_schedule", True
        return "mailbox_warmup_miss", "pipeline_warmup_schedule", True
    return "mailbox_missing_payload", "mailbox_payload_tensor_transport", False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, Enum):
        return value.value
    return value
