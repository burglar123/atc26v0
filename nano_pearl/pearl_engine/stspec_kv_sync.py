"""V4I ST-Spec mailbox KV/state synchronization probe helpers.

The helpers in this module deliberately build metadata-first synchronization
plans.  They make the target-side positions and KV slot/block intent explicit
for mailbox verification payloads without committing sequence/KV mutations.
Runtime code can then either stop after metadata validation or opt into a
separate guarded forward attempt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable

from nano_pearl.pearl_engine.pearl_protocol import validate_offsets


JsonDict = dict[str, Any]


class MailboxKVSyncMode(str, Enum):
    METADATA_ONLY = "metadata_only"
    GUARDED_FORWARD = "guarded_forward"
    NO_COMMIT_PROBE = "no_commit_probe"


@dataclass(frozen=True)
class MailboxKVSyncPlan:
    plan_id: int | None
    target_home_batch_id: int | str | None
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_lengths: list[int]
    offsets: list[int]
    total_tokens: int
    gamma: int
    current_seq_lengths: dict[int, int]
    current_kv_positions: dict[int, int]
    mailbox_token_positions: list[int]
    kv_slot_ids: list[int | None]
    block_ids: list[int | None]
    append_start_positions: dict[int, int]
    append_end_positions: dict[int, int]
    can_append_variable_offsets: bool
    requires_kv_append: bool
    requires_position_mapping: bool
    state_sync_mode: str
    mailbox_forward_commit_disabled: bool = True
    error_kind: str | None = None
    error_message: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MailboxKVSyncResult:
    attempted: bool
    success: bool
    plan: MailboxKVSyncPlan | None = None
    missing_seq_ids: list[int] = field(default_factory=list)
    error_kind: str | None = None
    error_message: str | None = None
    mutation_attempted: bool = False
    mutation_committed: bool = False
    mutation_rollback_success: bool = True

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


def normalize_kv_sync_mode(mode: str | MailboxKVSyncMode | None) -> MailboxKVSyncMode:
    if mode is None:
        return MailboxKVSyncMode.METADATA_ONLY
    if isinstance(mode, MailboxKVSyncMode):
        return mode
    try:
        return MailboxKVSyncMode(str(mode))
    except ValueError as exc:
        raise ValueError(
            f"Invalid stspec_kv_sync_mode={mode!r}; expected {[item.value for item in MailboxKVSyncMode]}"
        ) from exc


def build_mailbox_kv_sync_plan(
    verification_input: Any,
    exec_seqs: Iterable[Any],
    *,
    state_sync_mode: str | MailboxKVSyncMode = MailboxKVSyncMode.METADATA_ONLY,
    max_model_len: int | None = None,
    mailbox_forward_commit_disabled: bool = True,
) -> MailboxKVSyncPlan:
    mode = normalize_kv_sync_mode(state_sync_mode)
    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    seq_ids = [int(seq_id) for seq_id in verification_input.seq_ids]
    missing_seq_ids = [seq_id for seq_id in seq_ids if seq_id not in seq_by_id]
    if missing_seq_ids:
        plan = _base_error_plan(
            verification_input,
            mode=mode,
            error_kind="missing_target_kv_state_for_mailbox_seq",
            error_message="missing target Sequence state for mailbox KV sync plan",
            mailbox_forward_commit_disabled=mailbox_forward_commit_disabled,
        )
        return MailboxKVSyncPlan(**{**plan.to_dict(), "metadata": {**plan.metadata, "missing_seq_ids": missing_seq_ids}})

    validate_offsets(
        [int(length) for length in verification_input.per_seq_lengths],
        [int(offset) for offset in verification_input.offsets],
        int(verification_input.total_tokens),
        message_type="mailbox_kv_sync_plan",
        layout_kind=getattr(verification_input, "layout_kind", "variable_offsets"),
        seq_ids=seq_ids,
    )

    current_seq_lengths: dict[int, int] = {}
    current_kv_positions: dict[int, int] = {}
    append_start_positions: dict[int, int] = {}
    append_end_positions: dict[int, int] = {}
    mailbox_token_positions: list[int] = []
    kv_slot_ids: list[int | None] = []
    block_ids: list[int | None] = []
    requires_position_mapping = False
    error_kind = None
    error_message = None

    input_positions = list(getattr(verification_input, "positions", []) or [])
    input_kv_slots = list(getattr(verification_input, "kv_slot_ids", []) or [])

    for idx, seq_id in enumerate(seq_ids):
        seq = seq_by_id[seq_id]
        length = int(verification_input.per_seq_lengths[idx])
        offset = int(verification_input.offsets[idx])
        current_len = len(seq) if hasattr(seq, "__len__") else int(getattr(seq, "num_tokens", 0) or 0)
        current_seq_lengths[seq_id] = int(current_len)
        current_kv_positions[seq_id] = max(0, int(current_len) - 1)
        start_pos = int(input_positions[offset]) if offset < len(input_positions) and input_positions[offset] is not None else max(0, int(current_len) - length)
        end_pos = start_pos + length
        append_start_positions[seq_id] = start_pos
        append_end_positions[seq_id] = end_pos
        if max_model_len is not None and end_pos > int(max_model_len):
            error_kind = "mailbox_kv_sync_exceeds_max_model_len"
            error_message = (
                f"mailbox KV sync plan exceeds max_model_len for seq_id={seq_id}: "
                f"append_end={end_pos}, max_model_len={max_model_len}"
            )
        for token_index in range(length):
            token_position = start_pos + token_index
            mailbox_token_positions.append(token_position)
            slot = input_kv_slots[offset + token_index] if offset + token_index < len(input_kv_slots) else None
            if slot is None and hasattr(seq, "token_to_slot"):
                try:
                    slot = int(seq.token_to_slot(token_position))
                except Exception:
                    slot = None
            if slot is None:
                requires_position_mapping = True
            kv_slot_ids.append(None if slot is None else int(slot))
            block_ids.append(None if slot is None else _slot_to_block_id(seq, int(slot)))

    if len(mailbox_token_positions) != int(verification_input.total_tokens):
        error_kind = "mailbox_kv_sync_position_count_mismatch"
        error_message = (
            "mailbox KV sync plan position count does not match total_tokens: "
            f"position_count={len(mailbox_token_positions)}, total_tokens={verification_input.total_tokens}"
        )

    plan = MailboxKVSyncPlan(
        plan_id=getattr(verification_input, "plan_id", None),
        target_home_batch_id=getattr(verification_input, "target_home_batch_id", None),
        seq_ids=seq_ids,
        request_ids=list(getattr(verification_input, "request_ids", []) or []),
        per_seq_lengths=[int(length) for length in verification_input.per_seq_lengths],
        offsets=[int(offset) for offset in verification_input.offsets],
        total_tokens=int(verification_input.total_tokens),
        gamma=int(getattr(verification_input, "gamma", 0) or 0),
        current_seq_lengths=current_seq_lengths,
        current_kv_positions=current_kv_positions,
        mailbox_token_positions=mailbox_token_positions,
        kv_slot_ids=kv_slot_ids,
        block_ids=block_ids,
        append_start_positions=append_start_positions,
        append_end_positions=append_end_positions,
        can_append_variable_offsets=error_kind is None,
        requires_kv_append=True,
        requires_position_mapping=requires_position_mapping,
        state_sync_mode=mode.value,
        mailbox_forward_commit_disabled=bool(mailbox_forward_commit_disabled),
        error_kind=error_kind,
        error_message=error_message,
        metadata={
            "layout_kind": getattr(verification_input, "layout_kind", "variable_offsets"),
            "protocol_version": int(getattr(verification_input, "protocol_version", 1) or 1),
            "source_mailbox_payload_ids": list(getattr(verification_input, "source_mailbox_payload_ids", []) or []),
            "metadata_only_no_state_mutation": True,
        },
    )
    validate_mailbox_kv_sync_plan(plan, expected_seq_ids=seq_ids)
    return plan


def validate_mailbox_kv_sync_plan(
    plan: MailboxKVSyncPlan,
    *,
    expected_seq_ids: Iterable[int] | None = None,
) -> None:
    if expected_seq_ids is not None:
        expected = [int(seq_id) for seq_id in expected_seq_ids]
        if list(plan.seq_ids) != expected:
            raise RuntimeError(
                "Mailbox KV sync plan seq mismatch: "
                f"plan_seq_ids={plan.seq_ids}, expected_seq_ids={expected}"
            )
    if len(set(plan.seq_ids)) != len(plan.seq_ids):
        raise RuntimeError(f"Mailbox KV sync plan duplicate seq_ids={plan.seq_ids}")
    validate_offsets(
        [int(length) for length in plan.per_seq_lengths],
        [int(offset) for offset in plan.offsets],
        int(plan.total_tokens),
        message_type="mailbox_kv_sync_plan",
        layout_kind="variable_offsets",
        seq_ids=[int(seq_id) for seq_id in plan.seq_ids],
    )
    if len(plan.mailbox_token_positions) != int(plan.total_tokens):
        raise RuntimeError("Mailbox KV sync plan position_ids length must equal total_tokens")
    if len(plan.kv_slot_ids) not in (0, int(plan.total_tokens)):
        raise RuntimeError("Mailbox KV sync plan kv_slot_ids length must be empty or equal total_tokens")
    for seq_id, offset, length in zip(plan.seq_ids, plan.offsets, plan.per_seq_lengths):
        start = int(plan.append_start_positions[int(seq_id)])
        end = int(plan.append_end_positions[int(seq_id)])
        if end - start != int(length):
            raise RuntimeError(
                "Mailbox KV sync plan append positions do not match per_seq_length: "
                f"seq_id={seq_id}, start={start}, end={end}, length={length}"
            )
        expected_positions = list(range(start, end))
        actual_positions = plan.mailbox_token_positions[int(offset) : int(offset) + int(length)]
        if actual_positions != expected_positions:
            raise RuntimeError(
                "Mailbox KV sync plan position ids are not contiguous for seq: "
                f"seq_id={seq_id}, expected={expected_positions}, actual={actual_positions}"
            )
    if plan.error_kind:
        raise RuntimeError(f"Mailbox KV sync plan error: {plan.error_kind}: {plan.error_message}")
    normalize_kv_sync_mode(plan.state_sync_mode)


def apply_mailbox_kv_sync_plan_probe(
    plan: MailboxKVSyncPlan,
    *,
    commit_enabled: bool = False,
    forward_backend_available: bool = True,
) -> MailboxKVSyncResult:
    if plan.error_kind:
        return MailboxKVSyncResult(
            attempted=True,
            success=False,
            plan=plan,
            missing_seq_ids=[int(seq_id) for seq_id in plan.metadata.get("missing_seq_ids", [])],
            error_kind=plan.error_kind,
            error_message=plan.error_message,
            mutation_attempted=False,
            mutation_committed=False,
            mutation_rollback_success=True,
        )
    try:
        validate_mailbox_kv_sync_plan(plan, expected_seq_ids=plan.seq_ids)
    except Exception as exc:
        return MailboxKVSyncResult(
            attempted=True,
            success=False,
            plan=plan,
            error_kind=type(exc).__name__,
            error_message=str(exc),
            mutation_attempted=False,
            mutation_committed=False,
            mutation_rollback_success=True,
        )
    mode = normalize_kv_sync_mode(plan.state_sync_mode)
    if mode in {MailboxKVSyncMode.GUARDED_FORWARD, MailboxKVSyncMode.NO_COMMIT_PROBE} and not forward_backend_available:
        return MailboxKVSyncResult(
            attempted=True,
            success=False,
            plan=plan,
            error_kind="target_forward_backend_unavailable",
            error_message="guarded target forward backend is unavailable for mailbox KV sync probe",
            mutation_attempted=False,
            mutation_committed=False,
            mutation_rollback_success=True,
        )
    mutation_attempted = bool(commit_enabled and not plan.mailbox_forward_commit_disabled)
    return MailboxKVSyncResult(
        attempted=True,
        success=True,
        plan=plan,
        mutation_attempted=mutation_attempted,
        mutation_committed=False,
        mutation_rollback_success=True,
    )


def _slot_to_block_id(seq: Any, slot: int) -> int | None:
    block_size = int(getattr(seq, "block_size", 0) or 0)
    if block_size > 0:
        return slot // block_size
    block_table = getattr(seq, "block_table", None)
    if block_table:
        for block_id in block_table:
            if block_id is not None and int(block_id) >= 0:
                # Without the runtime block size, expose that a block mapping is
                # available rather than inventing a precise slot->block relation.
                return int(block_id)
    return None


def _base_error_plan(
    verification_input: Any,
    *,
    mode: MailboxKVSyncMode,
    error_kind: str,
    error_message: str,
    mailbox_forward_commit_disabled: bool,
) -> MailboxKVSyncPlan:
    seq_ids = [int(seq_id) for seq_id in verification_input.seq_ids]
    return MailboxKVSyncPlan(
        plan_id=getattr(verification_input, "plan_id", None),
        target_home_batch_id=getattr(verification_input, "target_home_batch_id", None),
        seq_ids=seq_ids,
        request_ids=list(getattr(verification_input, "request_ids", []) or []),
        per_seq_lengths=[int(length) for length in verification_input.per_seq_lengths],
        offsets=[int(offset) for offset in verification_input.offsets],
        total_tokens=int(verification_input.total_tokens),
        gamma=int(getattr(verification_input, "gamma", 0) or 0),
        current_seq_lengths={},
        current_kv_positions={},
        mailbox_token_positions=[],
        kv_slot_ids=[],
        block_ids=[],
        append_start_positions={},
        append_end_positions={},
        can_append_variable_offsets=False,
        requires_kv_append=True,
        requires_position_mapping=True,
        state_sync_mode=mode.value,
        mailbox_forward_commit_disabled=bool(mailbox_forward_commit_disabled),
        error_kind=error_kind,
        error_message=error_message,
        metadata={"metadata_only_no_state_mutation": True},
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, Enum):
        return value.value
    return value
