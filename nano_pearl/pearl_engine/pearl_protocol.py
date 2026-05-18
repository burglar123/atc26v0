"""Explicit PEARL verification protocol envelopes.

V4B introduced a versioned sidecar around the historical ``gamma * len(seqs)``
transport. V4C adds a real ``variable_offsets`` envelope that can describe
arbitrary/non-contiguous sequence sets and per-sequence payload lengths while
keeping default runtime behavior on ``legacy_fixed`` unchanged.
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
    if layout is not PearlLayoutKind.LEGACY_FIXED:
        raise NotImplementedError(f"PEARL protocol layout {layout.value!r} is not legacy_fixed.")


def build_offsets(lengths: list[int]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for length in lengths:
        if int(length) < 0:
            raise ValueError(f"Negative PEARL layout length: {length}")
        offsets.append(cursor)
        cursor += int(length)
    return offsets


def _layout_error(
    message_type: str,
    layout_kind: str,
    seq_ids: list[int],
    per_seq_lengths: list[int],
    offsets: list[int],
    total: int,
    detail: str,
) -> RuntimeError:
    return RuntimeError(
        f"{detail}: message_type={message_type}, layout_kind={layout_kind}, "
        f"seq_ids={seq_ids}, per_seq_lengths={per_seq_lengths}, offsets={offsets}, total={total}"
    )


def validate_offsets(
    lengths: list[int],
    offsets: list[int],
    total: int,
    *,
    message_type: str = "unknown",
    layout_kind: str = "unknown",
    seq_ids: list[int] | None = None,
) -> None:
    seq_ids = list(seq_ids or [])
    lengths = [int(length) for length in lengths]
    offsets = [int(offset) for offset in offsets]
    if offsets and offsets[0] != 0:
        raise _layout_error(message_type, layout_kind, seq_ids, lengths, offsets, total, "Invalid PEARL protocol offsets; first offset must be 0")
    if any(offsets[idx] > offsets[idx + 1] for idx in range(len(offsets) - 1)):
        raise _layout_error(message_type, layout_kind, seq_ids, lengths, offsets, total, "Invalid PEARL protocol offsets; offsets must be monotonic")
    expected = build_offsets(lengths)
    expected_total = sum(lengths)
    if offsets != expected or int(total) != expected_total:
        raise _layout_error(
            message_type,
            layout_kind,
            seq_ids,
            lengths,
            offsets,
            total,
            f"Invalid PEARL protocol offsets; expected_offsets={expected}, expected_total={expected_total}",
        )


def _validate_common_layout(message: PearlDraftMessage | PearlVerifyResultMessage, *, allow_duplicate_seq_ids: bool = False) -> None:
    ensure_supported_protocol(message.protocol_version)
    lengths = _message_lengths(message)
    offsets = _message_offsets(message)
    total = _message_total(message)
    if len(message.seq_ids) != len(lengths):
        raise _layout_error(message.message_type, message.layout_kind, message.seq_ids, lengths, offsets, total, "Invalid PEARL protocol layout; len(seq_ids) != len(per_seq_lengths)")
    if message.request_ids and len(message.request_ids) != len(message.seq_ids):
        raise _layout_error(message.message_type, message.layout_kind, message.seq_ids, lengths, offsets, total, "Invalid PEARL protocol layout; len(request_ids) != len(seq_ids)")
    if not allow_duplicate_seq_ids and len(set(message.seq_ids)) != len(message.seq_ids):
        raise _layout_error(message.message_type, message.layout_kind, message.seq_ids, lengths, offsets, total, "Invalid PEARL protocol layout; duplicate seq_ids are not allowed")
    validate_offsets(lengths, offsets, total, message_type=message.message_type, layout_kind=message.layout_kind, seq_ids=message.seq_ids)
    payload_length = _payload_length(message)
    if payload_length != total:
        raise _layout_error(message.message_type, message.layout_kind, message.seq_ids, lengths, offsets, total, f"Invalid PEARL protocol payload length; payload_length={payload_length}")


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
    ensure_legacy_fixed_layout(message.layout_kind)
    _validate_common_layout(message)
    validate_seq_alignment(message, expected_seq_ids)
    if isinstance(message, PearlDraftMessage):
        expected_next_total = int(gamma) * len(message.seq_ids)
        if len(message.next_round_input) != expected_next_total:
            raise _layout_error(message.message_type, message.layout_kind, message.seq_ids, _message_lengths(message), _message_offsets(message), _message_total(message), f"Invalid PEARL draft next_round_input length; next_round_length={len(message.next_round_input)}, expected_next_round_length={expected_next_total}")


def validate_variable_offsets_layout(
    message: PearlDraftMessage | PearlVerifyResultMessage,
    expected_seq_ids: Iterable[int] | None = None,
    *,
    allow_duplicate_seq_ids: bool = False,
) -> None:
    if normalize_layout_kind(message.layout_kind) is not PearlLayoutKind.VARIABLE_OFFSETS:
        raise NotImplementedError(f"PEARL protocol layout {message.layout_kind!r} is not variable_offsets.")
    _validate_common_layout(message, allow_duplicate_seq_ids=allow_duplicate_seq_ids)
    if expected_seq_ids is not None:
        validate_seq_alignment(message, expected_seq_ids)
    if isinstance(message, PearlDraftMessage):
        validate_offsets(
            [int(message.gamma)] * len(message.seq_ids),
            message.next_round_offsets,
            len(message.next_round_input),
            message_type=message.message_type,
            layout_kind=message.layout_kind,
            seq_ids=message.seq_ids,
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


def _seq_ids(seqs: list[Any]) -> list[int]:
    return [int(seq.seq_id) for seq in seqs]


def _request_ids(seqs: list[Any]) -> list[Any]:
    return [seq.request_id for seq in seqs]


def _draft_lengths(seqs: list[Any], gamma: int, per_seq_lengths: list[int] | None = None) -> list[int]:
    if per_seq_lengths is not None:
        return [int(length) for length in per_seq_lengths]
    return [1 if getattr(seq, "pre_verify", False) else int(gamma) for seq in seqs]


def _accepted_len_for_legacy(seq: Any, accepted: bool, rollout: int, gamma: int) -> int:
    if getattr(seq, "pre_verify", False):
        return 1 if accepted else 0
    return int(gamma) if accepted else int(gamma) - int(rollout)


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
    seq_ids = _seq_ids(seqs)
    per_seq_lengths = _draft_lengths(seqs, gamma)
    message = PearlDraftMessage(
        protocol_version=int(protocol_version),
        message_type=PearlMessageType.DRAFT_TOKENS.value,
        layout_kind=PearlLayoutKind.LEGACY_FIXED.value,
        plan_id=plan_id,
        gamma=int(gamma),
        seq_ids=seq_ids,
        request_ids=_request_ids(seqs),
        per_seq_draft_lengths=per_seq_lengths,
        draft_offsets=build_offsets(per_seq_lengths),
        total_draft_tokens=sum(per_seq_lengths),
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


def encode_variable_draft_message(
    *,
    seqs: list[Any],
    gamma: int,
    draft_token_ids: list[int],
    next_round_input: list[int],
    per_seq_draft_lengths: list[int] | None = None,
    next_round_lengths: list[int] | None = None,
    plan_id: int | None = None,
    runner_role: str | None = None,
    scheduled_seq_ids: list[int] | None = None,
    actual_exec_seq_ids: list[int] | None = None,
    target_batch_seq_ids: list[int] | None = None,
    draft_home_batch_seq_ids: list[int] | None = None,
    protocol_version: int = int(PearlProtocolVersion.LEGACY_EXPLICIT),
    layout_kind: str | PearlLayoutKind = PearlLayoutKind.VARIABLE_OFFSETS,
) -> PearlDraftMessage:
    ensure_supported_protocol(protocol_version)
    if normalize_layout_kind(layout_kind) is not PearlLayoutKind.VARIABLE_OFFSETS:
        raise NotImplementedError(f"encode_variable_draft_message requires variable_offsets, got {layout_kind!r}.")
    seq_ids = _seq_ids(seqs)
    per_seq_lengths = _draft_lengths(seqs, gamma, per_seq_draft_lengths)
    if next_round_lengths is None:
        next_round_lengths = [int(gamma)] * len(seq_ids)
    message = PearlDraftMessage(
        protocol_version=int(protocol_version),
        message_type=PearlMessageType.DRAFT_TOKENS.value,
        layout_kind=PearlLayoutKind.VARIABLE_OFFSETS.value,
        plan_id=plan_id,
        gamma=int(gamma),
        seq_ids=seq_ids,
        request_ids=_request_ids(seqs),
        per_seq_draft_lengths=per_seq_lengths,
        draft_offsets=build_offsets(per_seq_lengths),
        total_draft_tokens=sum(per_seq_lengths),
        draft_token_ids=list(draft_token_ids),
        next_round_input=list(next_round_input),
        next_round_offsets=build_offsets([int(length) for length in next_round_lengths]),
        scheduled_seq_ids=list(scheduled_seq_ids or seq_ids),
        actual_exec_seq_ids=list(actual_exec_seq_ids or seq_ids),
        target_batch_seq_ids=list(target_batch_seq_ids or []),
        draft_home_batch_seq_ids=list(draft_home_batch_seq_ids or []),
        runner_role=runner_role,
    )
    validate_variable_offsets_layout(message, seq_ids)
    return message


def decode_variable_draft_message(message: PearlDraftMessage) -> tuple[list[int], list[int]]:
    validate_variable_offsets_layout(message, message.seq_ids)
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
    seq_ids = _seq_ids(seqs)
    if per_seq_accepted_lengths is None:
        per_seq_accepted_lengths = [
            _accepted_len_for_legacy(seq, bool(acc[idx]), int(rollout[idx]), int(gamma))
            for idx, seq in enumerate(seqs)
        ]
    per_seq_accepted_lengths = [int(length) for length in per_seq_accepted_lengths]
    message = PearlVerifyResultMessage(
        protocol_version=int(protocol_version),
        message_type=PearlMessageType.VERIFY_RESULT.value,
        layout_kind=PearlLayoutKind.LEGACY_FIXED.value,
        plan_id=plan_id,
        gamma=int(gamma),
        seq_ids=seq_ids,
        request_ids=_request_ids(seqs),
        per_seq_accepted_lengths=per_seq_accepted_lengths,
        accepted_offsets=build_offsets(per_seq_accepted_lengths),
        total_accepted_tokens=sum(per_seq_accepted_lengths),
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


def encode_variable_verify_result(
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
    layout_kind: str | PearlLayoutKind = PearlLayoutKind.VARIABLE_OFFSETS,
) -> PearlVerifyResultMessage:
    ensure_supported_protocol(protocol_version)
    if normalize_layout_kind(layout_kind) is not PearlLayoutKind.VARIABLE_OFFSETS:
        raise NotImplementedError(f"encode_variable_verify_result requires variable_offsets, got {layout_kind!r}.")
    seq_ids = _seq_ids(seqs)
    if per_seq_accepted_lengths is None:
        per_seq_accepted_lengths = [
            _accepted_len_for_legacy(seq, bool(acc[idx]), int(rollout[idx]), int(gamma))
            for idx, seq in enumerate(seqs)
        ]
    per_seq_accepted_lengths = [int(length) for length in per_seq_accepted_lengths]
    message = PearlVerifyResultMessage(
        protocol_version=int(protocol_version),
        message_type=PearlMessageType.VERIFY_RESULT.value,
        layout_kind=PearlLayoutKind.VARIABLE_OFFSETS.value,
        plan_id=plan_id,
        gamma=int(gamma),
        seq_ids=seq_ids,
        request_ids=_request_ids(seqs),
        per_seq_accepted_lengths=per_seq_accepted_lengths,
        accepted_offsets=build_offsets(per_seq_accepted_lengths),
        total_accepted_tokens=sum(per_seq_accepted_lengths),
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
    validate_variable_offsets_layout(message, seq_ids)
    return message


def decode_variable_verify_result(message: PearlVerifyResultMessage) -> tuple[list[Any], list[int], list[int], list[Any]]:
    validate_variable_offsets_layout(message, message.seq_ids)
    return (
        list(message.acc),
        list(message.rollout),
        list(message.revised_token_ids),
        list(message.finish_flags),
    )
