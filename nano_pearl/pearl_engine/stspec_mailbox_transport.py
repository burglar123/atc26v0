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


@dataclass(frozen=True)
class MailboxPayloadTensorEnvelope:
    """JSON-traceable V4G token payload envelope.

    The probe represents the payload as CPU/list token ids. Future backends can
    replace this with a real tensor side channel while preserving the same
    sequence/offset/home-batch validation contract.
    """

    payload_tensor_ids: list[int]
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_lengths: list[int]
    offsets: list[int]
    total_tokens: int
    home_batch_id: int | str | None
    source_plan_id: int | None
    source_runner_role: str | None
    source_draft_home_batch_id: int | str | None
    target_home_batch_id: int | str | None
    gamma: int
    layout_kind: str
    protocol_version: int
    logical_step: int | None = None
    payload_device: str = "cpu"
    payload_dtype: str = "int64"
    payload_shape: list[int] = field(default_factory=list)
    is_tensor_payload_available: bool = True

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TargetForwardFromMailboxInput:
    """V4H target-side forward input assembled from mailbox payloads.

    This object is intentionally JSON-serializable and explicit about the fact
    that position/KV metadata may still be placeholders.  Runtime code can use
    it for guarded real-probe validation without falling back to the legacy
    scheduled sequence set.
    """

    plan_id: int | None
    target_home_batch_id: int | str | None
    seq_ids: list[int]
    request_ids: list[Any]
    input_token_ids: list[int]
    per_seq_lengths: list[int]
    offsets: list[int]
    total_tokens: int
    gamma: int
    positions: list[int | None]
    kv_slot_ids: list[int | None]
    source_mailbox_payload_ids: list[str]
    source_draft_plan_id: int | None
    layout_kind: str = "variable_offsets"
    protocol_version: int = 1
    metadata: JsonDict = field(default_factory=dict)

    @property
    def input_shape(self) -> list[int]:
        return [int(self.total_tokens)]

    @property
    def attention_metadata_placeholders(self) -> dict[str, Any]:
        return dict(self.metadata.get("attention_metadata_placeholders", {}))

    @property
    def kv_positions_placeholders(self) -> dict[str, Any]:
        return dict(self.metadata.get("kv_positions_placeholders", {}))

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


# Backward-compatible alias for the V4G public helper/tests.
VerificationInputMetadata = TargetForwardFromMailboxInput


@dataclass(frozen=True)
class KVStateSyncStatus:
    attempted: bool
    success: bool
    missing_seq_ids: list[int] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


class TargetForwardMailboxError(RuntimeError):
    """Structured V4H target-forward/mailbox probe failure."""

    def __init__(self, message: str, *, next_required_feature: str, error_kind: str | None = None):
        self.next_required_feature = str(next_required_feature)
        self.error_kind = error_kind or type(self).__name__
        super().__init__(f"{message}; next_required_feature={self.next_required_feature}")


def encode_payload_tensor_envelope_from_payloads(
    payloads: Iterable[MailboxPayload],
    *,
    home_batch_id: int | str | None,
    source_plan_id: int | None = None,
    source_runner_role: str | None = None,
    source_draft_home_batch_id: int | str | None = None,
    target_home_batch_id: int | str | None = None,
    gamma: int | None = None,
    logical_step: int | None = None,
    payload_device: str = "cpu",
    payload_dtype: str = "int64",
) -> MailboxPayloadTensorEnvelope:
    payload_list = list(payloads)
    seq_ids = [int(payload.seq_id) for payload in payload_list]
    request_ids = [payload.request_id for payload in payload_list]
    per_seq_lengths = [int(payload.per_seq_length) for payload in payload_list]
    offsets = build_offsets(per_seq_lengths)
    token_ids: list[int] = []
    for payload in payload_list:
        token_ids.extend(int(token) for token in payload.draft_token_ids)
    first = payload_list[0] if payload_list else None
    envelope = MailboxPayloadTensorEnvelope(
        payload_tensor_ids=token_ids,
        seq_ids=seq_ids,
        request_ids=request_ids,
        per_seq_lengths=per_seq_lengths,
        offsets=offsets,
        total_tokens=sum(per_seq_lengths),
        home_batch_id=home_batch_id,
        source_plan_id=source_plan_id if source_plan_id is not None else (first.plan_id if first else None),
        source_runner_role=source_runner_role if source_runner_role is not None else (first.producer_role if first else None),
        source_draft_home_batch_id=source_draft_home_batch_id if source_draft_home_batch_id is not None else (first.draft_home_batch_id if first else None),
        target_home_batch_id=target_home_batch_id if target_home_batch_id is not None else (first.target_home_batch_id if first else None),
        gamma=int(gamma if gamma is not None else (first.gamma if first else 0)),
        layout_kind=first.layout_kind if first else "variable_offsets",
        protocol_version=int(first.protocol_version if first else 1),
        logical_step=logical_step if logical_step is not None else (first.logical_step if first else None),
        payload_device=payload_device,
        payload_dtype=payload_dtype,
        payload_shape=[len(token_ids)],
        is_tensor_payload_available=True,
    )
    validate_payload_tensor_envelope(envelope)
    return envelope


def decode_payload_tensor_envelope(payload: MailboxPayloadTensorEnvelope | str | JsonDict) -> MailboxPayloadTensorEnvelope:
    if isinstance(payload, MailboxPayloadTensorEnvelope):
        envelope = payload
    else:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
        envelope = MailboxPayloadTensorEnvelope(**data)
    validate_payload_tensor_envelope(envelope)
    return envelope


def validate_payload_tensor_envelope(envelope: MailboxPayloadTensorEnvelope) -> None:
    if envelope.layout_kind != "variable_offsets":
        raise RuntimeError(f"Mailbox payload tensor envelope requires variable_offsets layout, got {envelope.layout_kind!r}")
    if len(envelope.seq_ids) != len(envelope.per_seq_lengths):
        raise RuntimeError("Invalid mailbox payload tensor envelope; len(seq_ids) != len(per_seq_lengths)")
    if envelope.request_ids and len(envelope.request_ids) != len(envelope.seq_ids):
        raise RuntimeError("Invalid mailbox payload tensor envelope; len(request_ids) != len(seq_ids)")
    if len(set(envelope.seq_ids)) != len(envelope.seq_ids):
        raise RuntimeError(f"Invalid mailbox payload tensor envelope; duplicate seq_ids={envelope.seq_ids}")
    validate_offsets(
        [int(length) for length in envelope.per_seq_lengths],
        [int(offset) for offset in envelope.offsets],
        int(envelope.total_tokens),
        message_type="mailbox_payload_tensor",
        layout_kind=envelope.layout_kind,
        seq_ids=[int(seq_id) for seq_id in envelope.seq_ids],
    )
    if envelope.is_tensor_payload_available and len(envelope.payload_tensor_ids) != int(envelope.total_tokens):
        raise RuntimeError(
            "Invalid mailbox payload tensor envelope; payload_tensor_ids length does not match total_tokens: "
            f"payload_length={len(envelope.payload_tensor_ids)}, total_tokens={envelope.total_tokens}"
        )


def payload_tensor_envelope_to_mailbox_payloads(envelope: MailboxPayloadTensorEnvelope) -> list[MailboxPayload]:
    validate_payload_tensor_envelope(envelope)
    payloads: list[MailboxPayload] = []
    for idx, seq_id in enumerate(envelope.seq_ids):
        offset = int(envelope.offsets[idx])
        length = int(envelope.per_seq_lengths[idx])
        payloads.append(
            MailboxPayload(
                plan_id=envelope.source_plan_id,
                producer_role=envelope.source_runner_role,
                producer_home_batch_id=envelope.source_draft_home_batch_id,
                target_home_batch_id=envelope.target_home_batch_id,
                draft_home_batch_id=envelope.source_draft_home_batch_id,
                seq_id=int(seq_id),
                request_id=envelope.request_ids[idx] if idx < len(envelope.request_ids) else None,
                home_batch_id=envelope.home_batch_id,
                gamma=int(envelope.gamma),
                layout_kind=envelope.layout_kind,
                protocol_version=int(envelope.protocol_version),
                draft_token_ids=list(envelope.payload_tensor_ids[offset : offset + length]),
                per_seq_length=length,
                offset=offset,
                logical_step=envelope.logical_step,
                producer_actual_exec_seq_ids=list(envelope.seq_ids),
                producer_draft_message_seq_ids=list(envelope.seq_ids),
                metadata={
                    "mailbox_payload_tensor_digest": envelope.digest(),
                    "payload_device": envelope.payload_device,
                    "payload_dtype": envelope.payload_dtype,
                    "payload_shape": list(envelope.payload_shape),
                    "is_tensor_payload_available": envelope.is_tensor_payload_available,
                },
            )
        )
    return payloads


def _seq_request_id(seq: Any) -> Any:
    return getattr(seq, "request_id", None)


def validate_target_forward_from_mailbox_input(
    verification_input: TargetForwardFromMailboxInput,
    *,
    actual_target_exec_seq_ids: Iterable[int],
    target_scheduler_seq_ids: Iterable[int] | None = None,
    scheduled_seq_ids: Iterable[int] | None = None,
    target_home_batch_id: int | str | None = None,
) -> None:
    seq_ids = [int(seq_id) for seq_id in verification_input.seq_ids]
    actual_ids = [int(seq_id) for seq_id in actual_target_exec_seq_ids]
    if scheduled_seq_ids is not None:
        scheduled_ids = [int(seq_id) for seq_id in scheduled_seq_ids]
        if scheduled_ids != actual_ids and seq_ids == scheduled_ids:
            raise RuntimeError(
                "Illegal legacy fallback detected: mailbox target forward input uses scheduled full batch "
                f"instead of actual target exec seq ids; scheduled_seq_ids={scheduled_ids}, actual_target_exec_seq_ids={actual_ids}"
            )
    if seq_ids != actual_ids:
        raise RuntimeError(
            "Target forward mailbox input seq ids must exactly match actual_target_exec_seq_ids: "
            f"input_seq_ids={seq_ids}, actual_target_exec_seq_ids={actual_ids}"
        )
    if len(set(seq_ids)) != len(seq_ids):
        raise RuntimeError(f"Target forward mailbox input has duplicate seq_ids={seq_ids}")
    if target_home_batch_id is not None and verification_input.target_home_batch_id != target_home_batch_id:
        raise RuntimeError(
            "Target forward mailbox input home_batch_id mismatch: "
            f"input_home_batch_id={verification_input.target_home_batch_id}, target_home_batch_id={target_home_batch_id}"
        )
    if target_scheduler_seq_ids is not None:
        scheduler_ids = {int(seq_id) for seq_id in target_scheduler_seq_ids}
        missing = [seq_id for seq_id in seq_ids if seq_id not in scheduler_ids]
        if missing:
            raise RuntimeError(
                "Target forward mailbox input references seq ids missing from target scheduler state: "
                f"missing_seq_ids={missing}, scheduler_seq_ids={sorted(scheduler_ids)}"
            )
    validate_offsets(
        [int(length) for length in verification_input.per_seq_lengths],
        [int(offset) for offset in verification_input.offsets],
        int(verification_input.total_tokens),
        message_type="target_forward_from_mailbox_input",
        layout_kind=verification_input.layout_kind,
        seq_ids=seq_ids,
    )
    if len(verification_input.input_token_ids) != int(verification_input.total_tokens):
        raise RuntimeError(
            "Target forward mailbox input token length does not match total_tokens: "
            f"token_count={len(verification_input.input_token_ids)}, total_tokens={verification_input.total_tokens}"
        )
    if verification_input.layout_kind != "variable_offsets":
        raise RuntimeError(
            f"Target forward mailbox input requires variable_offsets layout, got {verification_input.layout_kind!r}"
        )
    if len(verification_input.positions) not in (0, int(verification_input.total_tokens)):
        raise RuntimeError("Target forward mailbox input positions must be empty or match total_tokens")
    if len(verification_input.kv_slot_ids) not in (0, int(verification_input.total_tokens)):
        raise RuntimeError("Target forward mailbox input kv_slot_ids must be empty or match total_tokens")


def build_verification_input_from_mailbox_payload(
    payloads: Iterable[MailboxPayload],
    seqs: Iterable[Any],
    step_plan: Any,
    gamma: int,
) -> VerificationInputMetadata:
    payload_list = list(payloads)
    seq_list = list(seqs)
    expected_seq_ids = [int(seq.seq_id) for seq in seq_list]
    payload_seq_ids = [int(payload.seq_id) for payload in payload_list]
    if payload_seq_ids != expected_seq_ids:
        raise RuntimeError(
            "Mailbox verification input seq mismatch: "
            f"expected_seq_ids={expected_seq_ids}, payload_seq_ids={payload_seq_ids}"
        )
    if len(set(payload_seq_ids)) != len(payload_seq_ids):
        raise RuntimeError(f"Mailbox verification input duplicate seq_ids={payload_seq_ids}")
    home_batch_ids = {payload.home_batch_id for payload in payload_list}
    target_home_batch_id = getattr(step_plan, "target_home_batch_id", None)
    if home_batch_ids != {target_home_batch_id}:
        raise RuntimeError(
            "Mailbox verification input home batch mismatch: "
            f"target_home_batch_id={target_home_batch_id}, payload_home_batch_ids={home_batch_ids}"
        )
    lengths = [int(payload.per_seq_length) for payload in payload_list]
    offsets = build_offsets(lengths)
    input_token_ids: list[int] = []
    positions: list[int | None] = []
    kv_slot_ids: list[int | None] = []
    for payload, seq in zip(payload_list, seq_list):
        if len(payload.draft_token_ids) != int(payload.per_seq_length):
            raise RuntimeError(
                "Mailbox verification input payload length mismatch: "
                f"seq_id={payload.seq_id}, token_count={len(payload.draft_token_ids)}, per_seq_length={payload.per_seq_length}"
            )
        input_token_ids.extend(int(token) for token in payload.draft_token_ids)
        seq_len = len(seq) if hasattr(seq, "__len__") else None
        start_pos = None if seq_len is None else max(0, int(seq_len) - int(payload.per_seq_length))
        for idx in range(int(payload.per_seq_length)):
            position = None if start_pos is None else start_pos + idx
            positions.append(position)
            slot_id = None
            if position is not None and hasattr(seq, "token_to_slot"):
                try:
                    slot_id = int(seq.token_to_slot(position))
                except Exception:
                    slot_id = None
            kv_slot_ids.append(slot_id)
    validate_offsets(lengths, offsets, len(input_token_ids), message_type="verification_input_from_mailbox", layout_kind="variable_offsets", seq_ids=payload_seq_ids)
    plan_id = getattr(step_plan, "plan_id", None)
    scheduled_seq_ids = getattr(step_plan, "scheduled_seq_ids", None)
    source_ids = [
        f"{payload.home_batch_id}:{payload.seq_id}:{payload.offset}:{payload.per_seq_length}"
        for payload in payload_list
    ]
    metadata = {
        "attention_metadata_placeholders": {"requires_attention_metadata_wiring": True, "gamma": int(gamma)},
        "kv_positions_placeholders": {
            "requires_kv_position_wiring": any(slot_id is None for slot_id in kv_slot_ids),
            "target_home_batch_id": target_home_batch_id,
        },
        "scheduled_seq_ids": list(scheduled_seq_ids) if scheduled_seq_ids is not None else None,
        "actual_target_exec_seq_ids": list(getattr(step_plan, "actual_target_exec_seq_ids", expected_seq_ids)),
        "target_forward_from_mailbox_input_scaffold_version": "v4h",
    }
    verification_input = TargetForwardFromMailboxInput(
        plan_id=plan_id,
        target_home_batch_id=target_home_batch_id,
        seq_ids=payload_seq_ids,
        request_ids=[payload.request_id if payload.request_id is not None else _seq_request_id(seq) for payload, seq in zip(payload_list, seq_list)],
        input_token_ids=input_token_ids,
        per_seq_lengths=lengths,
        offsets=offsets,
        total_tokens=len(input_token_ids),
        gamma=int(gamma),
        positions=positions,
        kv_slot_ids=kv_slot_ids,
        source_mailbox_payload_ids=source_ids,
        source_draft_plan_id=payload_list[0].plan_id if payload_list else None,
        layout_kind="variable_offsets",
        protocol_version=int(payload_list[0].protocol_version if payload_list else 1),
        metadata=metadata,
    )
    validate_target_forward_from_mailbox_input(
        verification_input,
        actual_target_exec_seq_ids=expected_seq_ids,
        target_scheduler_seq_ids=expected_seq_ids,
        scheduled_seq_ids=scheduled_seq_ids,
        target_home_batch_id=target_home_batch_id,
    )
    return verification_input


def validate_kv_state_sync_for_mailbox_forward(
    verification_input: TargetForwardFromMailboxInput,
    exec_seqs: Iterable[Any],
) -> KVStateSyncStatus:
    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    missing = [int(seq_id) for seq_id in verification_input.seq_ids if int(seq_id) not in seq_by_id]
    if missing:
        return KVStateSyncStatus(
            attempted=True,
            success=False,
            missing_seq_ids=missing,
            error="target forward from mailbox input requires KV/state synchronization",
            error_kind="missing_target_kv_state_for_mailbox_seq",
        )
    if any(slot_id is None for slot_id in verification_input.kv_slot_ids):
        return KVStateSyncStatus(
            attempted=True,
            success=False,
            missing_seq_ids=[],
            error="target forward from mailbox input requires KV/state synchronization",
            error_kind="kv_position_mapping_not_implemented",
        )
    return KVStateSyncStatus(
        attempted=True,
        success=False,
        missing_seq_ids=[],
        error="target forward from mailbox input requires KV/state synchronization",
        error_kind="mailbox_kv_sync_plan_required",
    )


def map_target_forward_output_rows_to_seq_offsets(
    verification_input: TargetForwardFromMailboxInput,
    output_shape: Iterable[int] | None = None,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if output_shape is not None:
        shape = list(output_shape)
        output_rows = int(shape[0]) if shape else 0
        if output_rows != int(verification_input.total_tokens):
            raise RuntimeError(
                "Target forward mailbox output row count does not match input tokens: "
                f"output_shape={shape}, total_tokens={verification_input.total_tokens}; "
                "next_required_feature=target_forward_output_normalization"
            )
    for seq_id, offset, length in zip(verification_input.seq_ids, verification_input.offsets, verification_input.per_seq_lengths):
        row_start = int(offset)
        row_end = row_start + int(length)
        rows.append({
            "seq_id": int(seq_id),
            "offset": int(offset),
            "length": int(length),
            "row_start": row_start,
            "row_end": row_end,
            "token_range": [row_start, row_end],
        })
    return rows


def interpret_target_forward_from_mailbox_output(
    verification_input: TargetForwardFromMailboxInput,
    output_shape: Iterable[int] | None = None,
) -> list[JsonDict]:
    """V4K metadata-level output interpretation scaffold.

    This validates that output rows map exactly onto mailbox token offsets and
    returns the per-seq row ranges.  Accept/reject application remains guarded
    by the caller's mailbox_verify_apply_path boundary.
    """

    return map_target_forward_output_rows_to_seq_offsets(verification_input, output_shape)
