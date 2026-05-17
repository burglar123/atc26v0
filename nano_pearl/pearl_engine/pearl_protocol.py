"""Explicit PEARL verification protocol envelopes.

V4B keeps the NCCL tensor transport and verification semantics legacy-equivalent,
but wraps the implicit ``gamma * len(seqs)`` layout in versioned metadata so later
ST-Spec work can introduce variable layouts deliberately instead of relying on
hidden positional assumptions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Iterable


class PearlProtocolVersion(IntEnum):
    LEGACY_EXPLICIT = 1


class PearlMessageType(str, Enum):
    DRAFT_TOKENS = "draft_tokens"
    VERIFY_RESULT = "verify_result"


class PearlLayoutKind(str, Enum):
    LEGACY_FIXED = "legacy_fixed"
    VARIABLE_OFFSETS = "variable_offsets"


@dataclass(frozen=True)
class PearlPackedLayout:
    per_seq_lengths: list[int]
    offsets: list[int]
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PearlDraftMessage:
    protocol_version: int
    message_type: str
    layout_kind: str
    plan_id: int | None
    gamma: int
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_draft_lengths: list[int]
    draft_offsets: list[int]
    total_draft_tokens: int
    draft_token_ids: list[int]
    next_round_input: list[int]
    next_round_offsets: list[int]
    scheduled_seq_ids: list[int] = field(default_factory=list)
    actual_exec_seq_ids: list[int] = field(default_factory=list)
    target_batch_seq_ids: list[int] = field(default_factory=list)
    draft_home_batch_seq_ids: list[int] = field(default_factory=list)
    runner_role: str | None = None

    def to_trace_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def digest(self) -> str:
        return _digest(self.to_trace_dict())


@dataclass(frozen=True)
class PearlVerifyResultMessage:
    protocol_version: int
    message_type: str
    layout_kind: str
    plan_id: int | None
    gamma: int
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_accepted_lengths: list[int]
    accepted_offsets: list[int]
    total_accepted_tokens: int
    acc: list[int | bool]
    rollout: list[int]
    revised_token_ids: list[int]
    finish_flags: list[int | bool]
    payload_rows: list[list[int | bool]]
    scheduled_seq_ids: list[int] = field(default_factory=list)
    actual_exec_seq_ids: list[int] = field(default_factory=list)
    target_batch_seq_ids: list[int] = field(default_factory=list)
    draft_home_batch_seq_ids: list[int] = field(default_factory=list)
    runner_role: str | None = None

    def to_trace_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def digest(self) -> str:
        return _digest(self.to_trace_dict())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def normalize_layout_kind(layout_kind: str | PearlLayoutKind) -> PearlLayoutKind:
    if isinstance(layout_kind, PearlLayoutKind):
        return layout_kind
    try:
        return PearlLayoutKind(layout_kind)
    except ValueError as exc:
        raise ValueError(
            f"Invalid PEARL protocol layout={layout_kind!r}; expected one of "
            f"{[item.value for item in PearlLayoutKind]}"
        ) from exc


def ensure_supported_protocol(protocol_version: int) -> None:
    if int(protocol_version) != int(PearlProtocolVersion.LEGACY_EXPLICIT):
        raise NotImplementedError(
            f"PEARL protocol_version={protocol_version!r} is unsupported; only version 1 is implemented."
        )


def ensure_legacy_fixed_layout(layout_kind: str | PearlLayoutKind) -> None:
    layout = normalize_layout_kind(layout_kind)
    if layout is PearlLayoutKind.VARIABLE_OFFSETS:
        raise NotImplementedError("variable_offsets PEARL protocol is reserved for V4C.")
    if layout is not PearlLayoutKind.LEGACY_FIXED:
        raise NotImplementedError(f"PEARL protocol layout {layout.value!r} is not implemented.")


def build_offsets(lengths: list[int]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for length in lengths:
        if int(length) < 0:
            raise ValueError(f"Negative PEARL layout length: {length}")
        offsets.append(cursor)
        cursor += int(length)
    return offsets


def validate_offsets(lengths: list[int], offsets: list[int], total: int) -> None:
    expected = build_offsets([int(length) for length in lengths])
    if list(offsets) != expected or int(total) != sum(int(length) for length in lengths):
        raise RuntimeError(
            "Invalid PEARL protocol offsets: "
            f"per_seq_lengths={lengths}, offsets={offsets}, expected_offsets={expected}, "
            f"total={total}, expected_total={sum(int(length) for length in lengths)}"
        )


def validate_seq_alignment(
    message: PearlDraftMessage | PearlVerifyResultMessage,
    expected_seq_ids: Iterable[int],
) -> None:
    expected = list(expected_seq_ids)
    if list(message.seq_ids) != expected:
        raise RuntimeError(
            "PEARL protocol seq alignment failed: "
            f"plan_id={message.plan_id}, message_type={message.message_type}, "
            f"layout_kind={message.layout_kind}, expected_seq_ids={expected}, "
            f"message_seq_ids={message.seq_ids}, per_seq_lengths={_message_lengths(message)}, "
            f"offsets={_message_offsets(message)}, total_payload_length={_message_total(message)}, "
            f"gamma={message.gamma}"
        )


def validate_legacy_fixed_layout(
    message: PearlDraftMessage | PearlVerifyResultMessage,
    expected_seq_ids: Iterable[int],
    gamma: int,
) -> None:
    ensure_supported_protocol(message.protocol_version)
    ensure_legacy_fixed_layout(message.layout_kind)
    validate_seq_alignment(message, expected_seq_ids)
    lengths = _message_lengths(message)
    offsets = _message_offsets(message)
    total = _message_total(message)
    validate_offsets(lengths, offsets, total)
    payload_length = _payload_length(message)
    if payload_length != total:
        raise RuntimeError(
            "Invalid PEARL protocol payload length: "
            f"plan_id={message.plan_id}, message_type={message.message_type}, "
            f"layout_kind={message.layout_kind}, expected_seq_ids={list(expected_seq_ids)}, "
            f"message_seq_ids={message.seq_ids}, per_seq_lengths={lengths}, offsets={offsets}, "
            f"total_payload_length={payload_length}, expected_total={total}, gamma={gamma}"
        )
    if isinstance(message, PearlDraftMessage):
        expected_next_total = int(gamma) * len(message.seq_ids)
        if len(message.next_round_input) != expected_next_total:
            raise RuntimeError(
                "Invalid PEARL draft next_round_input length: "
                f"plan_id={message.plan_id}, message_type={message.message_type}, "
                f"layout_kind={message.layout_kind}, expected_seq_ids={list(expected_seq_ids)}, "
                f"message_seq_ids={message.seq_ids}, per_seq_lengths={lengths}, offsets={offsets}, "
                f"total_payload_length={len(message.next_round_input)}, gamma={gamma}"
            )


def _message_lengths(message: PearlDraftMessage | PearlVerifyResultMessage) -> list[int]:
    if isinstance(message, PearlDraftMessage):
        return list(message.per_seq_draft_lengths)
    return list(message.per_seq_accepted_lengths)


def _message_offsets(message: PearlDraftMessage | PearlVerifyResultMessage) -> list[int]:
    if isinstance(message, PearlDraftMessage):
        return list(message.draft_offsets)
    return list(message.accepted_offsets)


def _message_total(message: PearlDraftMessage | PearlVerifyResultMessage) -> int:
    if isinstance(message, PearlDraftMessage):
        return int(message.total_draft_tokens)
    return int(message.total_accepted_tokens)


def _payload_length(message: PearlDraftMessage | PearlVerifyResultMessage) -> int:
    if isinstance(message, PearlDraftMessage):
        return len(message.draft_token_ids)
    return int(sum(int(length) for length in message.per_seq_accepted_lengths))


def encode_legacy_draft_message(
    *,
    seqs: list[Any],
    gamma: int,
    draft_token_ids: list[int],
    next_round_input: list[int],
    plan_id: int | None = None,
    runner_role: str | None = None,
    scheduled_seq_ids: list[int] | None = None,
    actual_exec_seq_ids: list[int] | None = None,
    target_batch_seq_ids: list[int] | None = None,
    draft_home_batch_seq_ids: list[int] | None = None,
    protocol_version: int = int(PearlProtocolVersion.LEGACY_EXPLICIT),
    layout_kind: str | PearlLayoutKind = PearlLayoutKind.LEGACY_FIXED,
) -> PearlDraftMessage:
    ensure_supported_protocol(protocol_version)
    ensure_legacy_fixed_layout(layout_kind)
    seq_ids = [int(seq.seq_id) for seq in seqs]
    request_ids = [seq.request_id for seq in seqs]
    per_seq_lengths = [1 if seq.pre_verify else int(gamma) for seq in seqs]
    offsets = build_offsets(per_seq_lengths)
    total = sum(per_seq_lengths)
    message = PearlDraftMessage(
        protocol_version=int(protocol_version),
        message_type=PearlMessageType.DRAFT_TOKENS.value,
        layout_kind=PearlLayoutKind.LEGACY_FIXED.value,
        plan_id=plan_id,
        gamma=int(gamma),
        seq_ids=seq_ids,
        request_ids=request_ids,
        per_seq_draft_lengths=per_seq_lengths,
        draft_offsets=offsets,
        total_draft_tokens=total,
        draft_token_ids=list(draft_token_ids),
        next_round_input=list(next_round_input),
        next_round_offsets=build_offsets([int(gamma)] * len(seq_ids)),
        scheduled_seq_ids=list(scheduled_seq_ids or seq_ids),
        actual_exec_seq_ids=list(actual_exec_seq_ids or seq_ids),
        target_batch_seq_ids=list(target_batch_seq_ids or []),
        draft_home_batch_seq_ids=list(draft_home_batch_seq_ids or []),
        runner_role=runner_role,
    )
    validate_legacy_fixed_layout(message, seq_ids, gamma)
    return message


def decode_legacy_draft_message(message: PearlDraftMessage) -> tuple[list[int], list[int]]:
    validate_legacy_fixed_layout(message, message.seq_ids, message.gamma)
    return list(message.draft_token_ids), list(message.next_round_input)


def encode_legacy_verify_result(
    *,
    seqs: list[Any],
    gamma: int,
    acc: list[int | bool],
    rollout: list[int],
    revise_token: list[int],
    finish: list[int | bool],
    per_seq_accepted_lengths: list[int] | None = None,
    plan_id: int | None = None,
    runner_role: str | None = None,
    scheduled_seq_ids: list[int] | None = None,
    actual_exec_seq_ids: list[int] | None = None,
    target_batch_seq_ids: list[int] | None = None,
    draft_home_batch_seq_ids: list[int] | None = None,
    protocol_version: int = int(PearlProtocolVersion.LEGACY_EXPLICIT),
    layout_kind: str | PearlLayoutKind = PearlLayoutKind.LEGACY_FIXED,
) -> PearlVerifyResultMessage:
    ensure_supported_protocol(protocol_version)
    ensure_legacy_fixed_layout(layout_kind)
    seq_ids = [int(seq.seq_id) for seq in seqs]
    request_ids = [seq.request_id for seq in seqs]
    if per_seq_accepted_lengths is None:
        per_seq_accepted_lengths = [
            _accepted_len_for_legacy(seq, bool(acc[idx]), int(rollout[idx]), int(gamma))
            for idx, seq in enumerate(seqs)
        ]
    offsets = build_offsets([int(length) for length in per_seq_accepted_lengths])
    total = sum(int(length) for length in per_seq_accepted_lengths)
    message = PearlVerifyResultMessage(
        protocol_version=int(protocol_version),
        message_type=PearlMessageType.VERIFY_RESULT.value,
        layout_kind=PearlLayoutKind.LEGACY_FIXED.value,
        plan_id=plan_id,
        gamma=int(gamma),
        seq_ids=seq_ids,
        request_ids=request_ids,
        per_seq_accepted_lengths=[int(length) for length in per_seq_accepted_lengths],
        accepted_offsets=offsets,
        total_accepted_tokens=total,
        acc=list(acc),
        rollout=[int(value) for value in rollout],
        revised_token_ids=[int(value) for value in revise_token],
        finish_flags=list(finish),
        payload_rows=[list(acc), [int(v) for v in rollout], [int(v) for v in revise_token], list(finish)],
        scheduled_seq_ids=list(scheduled_seq_ids or seq_ids),
        actual_exec_seq_ids=list(actual_exec_seq_ids or seq_ids),
        target_batch_seq_ids=list(target_batch_seq_ids or []),
        draft_home_batch_seq_ids=list(draft_home_batch_seq_ids or []),
        runner_role=runner_role,
    )
    validate_legacy_fixed_layout(message, seq_ids, gamma)
    return message


def decode_legacy_verify_result(message: PearlVerifyResultMessage) -> tuple[list[Any], list[int], list[int], list[Any]]:
    validate_legacy_fixed_layout(message, message.seq_ids, message.gamma)
    return (
        list(message.acc),
        list(message.rollout),
        list(message.revised_token_ids),
        list(message.finish_flags),
    )


def _accepted_len_for_legacy(seq: Any, accepted: bool, rollout: int, gamma: int) -> int:
    if getattr(seq, "pre_verify", False):
        return 1 if accepted else 0
    return int(gamma) if accepted else int(gamma) - int(rollout)


def encode_variable_draft_message(*args: Any, **kwargs: Any) -> PearlDraftMessage:
    raise NotImplementedError("variable_offsets PEARL protocol is reserved for V4C.")


def decode_variable_draft_message(*args: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
    raise NotImplementedError("variable_offsets PEARL protocol is reserved for V4C.")
