"""V4L no-commit mailbox verification apply probe helpers.

These helpers turn mailbox target-forward interpretation metadata into a
per-sequence verification result and a no-commit apply plan.  They deliberately
validate state and payload intent without mutating Sequence, KV cache, scheduler,
or mailbox state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


JsonDict = dict[str, Any]


class MailboxVerifyApplyError(RuntimeError):
    def __init__(self, message: str, *, next_required_feature: str, error_kind: str | None = None):
        self.next_required_feature = str(next_required_feature)
        self.error_kind = error_kind or type(self).__name__
        super().__init__(f"{message}; next_required_feature={self.next_required_feature}")


@dataclass(frozen=True)
class MailboxVerifyResult:
    plan_id: int | None
    target_home_batch_id: int | str | None
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_lengths: list[int]
    offsets: list[int]
    drafted_token_ids_by_seq: dict[int, list[int]]
    target_token_ids_by_seq: dict[int, list[int]]
    accepted_lengths_by_seq: dict[int, int]
    rejected_seq_ids: list[int]
    rejected_token_positions_by_seq: dict[int, list[int]]
    invalidated_mailbox_payload_ids: list[str]
    mailbox_payload_ids: list[str]
    total_accepted_tokens: int
    total_rejected_tokens: int
    output_owner_rank: int | None = None
    no_commit: bool = True
    metadata_only: bool = False
    next_required_feature: str | None = None
    positions_by_seq: dict[int, list[int]] = field(default_factory=dict)
    kv_slot_ids_by_seq: dict[int, list[int]] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MailboxVerifyApplyPlan:
    seq_ids: list[int]
    accepted_token_ids_by_seq: dict[int, list[int]]
    rejected_token_ids_by_seq: dict[int, list[int]]
    accepted_lengths_by_seq: dict[int, int]
    append_positions_by_seq: dict[int, list[int]]
    sequence_state_before: dict[int, JsonDict]
    would_finish_seq_ids: list[int]
    would_continue_seq_ids: list[int]
    mailbox_payloads_to_consume: list[str]
    mailbox_payloads_to_invalidate: list[str]
    state_mutation_allowed: bool = False
    no_commit: bool = True

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MailboxVerifyApplyProbeResult:
    attempted: bool
    success: bool
    plan: MailboxVerifyApplyPlan | None = None
    error_kind: str | None = None
    error_message: str | None = None
    state_mutation_attempted: bool = False
    state_mutation_committed: bool = False
    state_mutation_rollback_success: bool = True
    next_required_feature: str | None = None

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MailboxVerifyCommitPlan:
    plan_id: int | None
    target_home_batch_id: int | str | None
    seq_ids: list[int]
    request_ids: list[Any]
    accepted_lengths_by_seq: dict[int, int]
    accepted_token_ids_by_seq: dict[int, list[int]]
    rejected_seq_ids: list[int]
    rejected_token_ids_by_seq: dict[int, list[int]]
    invalidated_payload_ids: list[str]
    mailbox_payloads_to_consume: list[str]
    mailbox_payloads_to_invalidate: list[str]
    sequence_state_before: dict[int, JsonDict]
    sequence_state_after_expected: dict[int, JsonDict]
    kv_state_before: JsonDict
    kv_state_after_expected: JsonDict
    commit_allowed: bool
    commit_mode: str = "guarded_probe"
    rollback_required: bool = False
    rollback_success: bool = True

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MailboxVerifyCommitResult:
    attempted: bool
    success: bool
    plan: MailboxVerifyCommitPlan | None = None
    error_kind: str | None = None
    error_message: str | None = None
    next_required_feature: str | None = None
    sequence_state_commit_attempted: bool = False
    sequence_state_commit_success: bool = False
    sequence_state_before: dict[int, JsonDict] = field(default_factory=dict)
    sequence_state_after: dict[int, JsonDict] = field(default_factory=dict)
    kv_commit_plan_built: bool = False
    kv_commit_attempted: bool = False
    kv_commit_success: bool = False
    kv_commit_shadow_only: bool = False
    kv_commit_error: str | None = None
    kv_commit_error_kind: str | None = None
    kv_commit_rollback_attempted: bool = False
    kv_commit_rollback_success: bool = True
    kv_commit_skipped_non_owner: bool = False
    mailbox_payload_consume_plan_built: bool = False
    mailbox_payload_consume_attempted: bool = False
    mailbox_payload_consume_success: bool = False
    mailbox_payload_consume_error: str | None = None
    mailbox_payload_consume_error_kind: str | None = None
    mailbox_payload_consumed_payload_ids: list[str] = field(default_factory=list)
    mailbox_payload_consumed_token_count: int = 0
    mailbox_payload_invalidate_attempted: bool = False
    mailbox_payload_invalidate_success: bool = False
    mailbox_payload_invalidate_error: str | None = None
    mailbox_payload_invalidated_payload_ids: list[str] = field(default_factory=list)
    mailbox_payload_invalidated_token_count: int = 0
    mailbox_payload_lifecycle_before: dict[str, JsonDict] = field(default_factory=dict)
    mailbox_payload_lifecycle_after: dict[str, JsonDict] = field(default_factory=dict)
    mailbox_payload_duplicate_consume_detected: bool = False
    mailbox_payload_consume_rollback_attempted: bool = False
    mailbox_payload_consume_rollback_success: bool = True
    mailbox_payload_consume_skipped_non_owner: bool = False
    mailbox_payload_invalidate_skipped_non_owner: bool = False
    next_pipeline_step_attempted: bool = False
    next_pipeline_step_success: bool = False
    next_pipeline_step_error: str | None = None
    next_pipeline_step_error_kind: str | None = None
    next_pipeline_plan_id: int | None = None
    next_pipeline_target_home_batch_id: int | str | None = None
    next_pipeline_draft_home_batch_id: int | str | None = None
    next_pipeline_actual_target_seq_ids: list[int] = field(default_factory=list)
    next_pipeline_actual_draft_seq_ids: list[int] = field(default_factory=list)
    previous_committed_plan_id: int | None = None
    previous_consumed_payload_ids: list[str] = field(default_factory=list)
    previous_invalidated_payload_ids: list[str] = field(default_factory=list)
    duplicate_payload_consume_after_continue: bool = False
    pipeline_state_after_commit_valid: bool = False
    scheduler_state_after_commit_valid: bool = False
    breadth_only_step_count: int = 0
    current_pipeline_step: int = 0
    current_plan_id: int | None = None
    next_plan_id: int | None = None
    previous_target_home_batch_id: int | str | None = None
    previous_draft_home_batch_id: int | str | None = None
    current_target_home_batch_id: int | str | None = None
    current_draft_home_batch_id: int | str | None = None
    active_seq_ids_before_second_step: list[int] = field(default_factory=list)
    active_seq_ids_after_second_step: list[int] = field(default_factory=list)
    committed_seq_ids: list[int] = field(default_factory=list)
    consumed_payload_ids: list[str] = field(default_factory=list)
    invalidated_payload_ids: list[str] = field(default_factory=list)
    available_mailbox_payload_ids: list[str] = field(default_factory=list)
    pending_mailbox_payload_ids: list[str] = field(default_factory=list)
    second_step_state_check_attempted: bool = False
    second_step_state_check_success: bool = False
    second_step_state_error: str | None = None
    second_step_state_error_kind: str | None = None
    repeated_verify_after_commit_detected: bool = False
    scheduler_state_after_second_step_valid: bool = False
    sequence_state_after_second_step_valid: bool = False
    mailbox_state_after_second_step_valid: bool = False
    request_completion_check_attempted: bool = False
    request_completion_check_success: bool = False
    request_completion_reason: str | None = None
    active_seq_ids_at_completion_check: list[int] = field(default_factory=list)
    finished_seq_ids_at_completion_check: list[int] = field(default_factory=list)
    unfinished_seq_ids_at_completion_check: list[int] = field(default_factory=list)
    max_tokens_reached_seq_ids: list[int] = field(default_factory=list)
    eos_reached_seq_ids: list[int] = field(default_factory=list)
    mailbox_pending_payload_ids_at_completion: list[str] = field(default_factory=list)
    mailbox_consumed_payload_ids_at_completion: list[str] = field(default_factory=list)
    scheduler_active_seq_ids_at_completion: list[int] = field(default_factory=list)
    sequence_state_completion_valid: bool = False
    scheduler_state_completion_valid: bool = False
    mailbox_state_completion_valid: bool = False
    request_completion_error: str | None = None
    request_completion_error_kind: str | None = None
    result_finalization_attempted: bool = False
    result_finalization_success: bool = False
    result_finalization_error: str | None = None
    second_step_rollback_attempted: bool = False
    second_step_rollback_success: bool = True
    breadth_only_completed: bool = False
    breadth_only_completion_reason: str | None = None
    next_pipeline_step_skipped_non_owner: bool = False
    rollback_attempted: bool = False
    rollback_success: bool = True
    skipped_non_owner: bool = False
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MailboxKVCommitPlan:
    plan_id: int | None
    target_home_batch_id: int | str | None
    seq_ids: list[int]
    request_ids: list[Any]
    accepted_lengths_by_seq: dict[int, int]
    accepted_token_ids_by_seq: dict[int, list[int]]
    append_start_positions_by_seq: dict[int, int]
    append_end_positions_by_seq: dict[int, int]
    kv_positions_by_seq: dict[int, list[int]]
    kv_slots_or_blocks_by_seq: dict[int, list[int]]
    sequence_length_before_by_seq: dict[int, int]
    sequence_length_after_by_seq: dict[int, int]
    kv_length_before_by_seq: dict[int, int]
    kv_length_after_by_seq: dict[int, int]
    commit_allowed: bool
    commit_mode: str = "shadow_only"
    rollback_required: bool = False
    rollback_success: bool = True
    error_kind: str | None = None
    error_message: str | None = None

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MailboxKVCommitResult:
    attempted: bool
    success: bool
    plan: MailboxKVCommitPlan | None = None
    shadow_only: bool = False
    skipped_non_owner: bool = False
    error_kind: str | None = None
    error_message: str | None = None
    next_required_feature: str | None = None
    rollback_attempted: bool = False
    rollback_success: bool = True
    kv_state_before: dict[int, JsonDict] = field(default_factory=dict)
    kv_state_after: dict[int, JsonDict] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MailboxPayloadConsumePlan:
    plan_id: int | None
    target_home_batch_id: int | str | None
    seq_ids: list[int]
    consumed_payload_ids: list[str]
    invalidated_payload_ids: list[str]
    consumed_token_count_by_seq: dict[int, int]
    invalidated_token_count_by_seq: dict[int, int]
    accepted_lengths_by_seq: dict[int, int]
    rejected_seq_ids: list[int]
    mailbox_state_before: dict[str, JsonDict]
    mailbox_state_after_expected: dict[str, JsonDict]
    rollback_required: bool = False
    rollback_success: bool = True

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class MailboxPayloadConsumeResult:
    attempted: bool
    success: bool
    plan: MailboxPayloadConsumePlan | None = None
    error_kind: str | None = None
    error_message: str | None = None
    next_required_feature: str | None = None
    consumed_payload_ids: list[str] = field(default_factory=list)
    invalidated_payload_ids: list[str] = field(default_factory=list)
    consumed_token_count: int = 0
    invalidated_token_count: int = 0
    mailbox_state_before: dict[str, JsonDict] = field(default_factory=dict)
    mailbox_state_after: dict[str, JsonDict] = field(default_factory=dict)
    duplicate_consume_detected: bool = False
    rollback_attempted: bool = False
    rollback_success: bool = True
    skipped_non_owner: bool = False
    invalidate_skipped_non_owner: bool = False
    next_pipeline_step_attempted: bool = False
    next_pipeline_step_success: bool = False
    next_pipeline_step_error: str | None = None
    next_pipeline_step_error_kind: str | None = None
    next_pipeline_plan_id: int | None = None
    next_pipeline_target_home_batch_id: int | str | None = None
    next_pipeline_draft_home_batch_id: int | str | None = None
    next_pipeline_actual_target_seq_ids: list[int] = field(default_factory=list)
    next_pipeline_actual_draft_seq_ids: list[int] = field(default_factory=list)
    previous_committed_plan_id: int | None = None
    previous_consumed_payload_ids: list[str] = field(default_factory=list)
    previous_invalidated_payload_ids: list[str] = field(default_factory=list)
    duplicate_payload_consume_after_continue: bool = False
    pipeline_state_after_commit_valid: bool = False
    scheduler_state_after_commit_valid: bool = False
    breadth_only_step_count: int = 0
    current_pipeline_step: int = 0
    current_plan_id: int | None = None
    next_plan_id: int | None = None
    previous_target_home_batch_id: int | str | None = None
    previous_draft_home_batch_id: int | str | None = None
    current_target_home_batch_id: int | str | None = None
    current_draft_home_batch_id: int | str | None = None
    active_seq_ids_before_second_step: list[int] = field(default_factory=list)
    active_seq_ids_after_second_step: list[int] = field(default_factory=list)
    committed_seq_ids: list[int] = field(default_factory=list)
    consumed_payload_ids: list[str] = field(default_factory=list)
    invalidated_payload_ids: list[str] = field(default_factory=list)
    available_mailbox_payload_ids: list[str] = field(default_factory=list)
    pending_mailbox_payload_ids: list[str] = field(default_factory=list)
    second_step_state_check_attempted: bool = False
    second_step_state_check_success: bool = False
    second_step_state_error: str | None = None
    second_step_state_error_kind: str | None = None
    repeated_verify_after_commit_detected: bool = False
    scheduler_state_after_second_step_valid: bool = False
    sequence_state_after_second_step_valid: bool = False
    mailbox_state_after_second_step_valid: bool = False
    request_completion_check_attempted: bool = False
    request_completion_check_success: bool = False
    request_completion_reason: str | None = None
    active_seq_ids_at_completion_check: list[int] = field(default_factory=list)
    finished_seq_ids_at_completion_check: list[int] = field(default_factory=list)
    unfinished_seq_ids_at_completion_check: list[int] = field(default_factory=list)
    max_tokens_reached_seq_ids: list[int] = field(default_factory=list)
    eos_reached_seq_ids: list[int] = field(default_factory=list)
    mailbox_pending_payload_ids_at_completion: list[str] = field(default_factory=list)
    mailbox_consumed_payload_ids_at_completion: list[str] = field(default_factory=list)
    scheduler_active_seq_ids_at_completion: list[int] = field(default_factory=list)
    sequence_state_completion_valid: bool = False
    scheduler_state_completion_valid: bool = False
    mailbox_state_completion_valid: bool = False
    request_completion_error: str | None = None
    request_completion_error_kind: str | None = None
    result_finalization_attempted: bool = False
    result_finalization_success: bool = False
    result_finalization_error: str | None = None
    second_step_rollback_attempted: bool = False
    second_step_rollback_success: bool = True
    breadth_only_completed: bool = False
    breadth_only_completion_reason: str | None = None
    next_pipeline_step_skipped_non_owner: bool = False

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


def build_mailbox_verify_result(
    verification_input: Any,
    *,
    target_token_ids: Iterable[int] | None = None,
    output_owner_rank: int | None = None,
    metadata_only: bool = False,
) -> MailboxVerifyResult:
    seq_ids = [int(seq_id) for seq_id in verification_input.seq_ids]
    lengths = [int(length) for length in verification_input.per_seq_lengths]
    offsets = [int(offset) for offset in verification_input.offsets]
    drafted_flat = [int(token) for token in verification_input.input_token_ids]
    target_flat = [int(token) for token in target_token_ids] if target_token_ids is not None else []
    if len(drafted_flat) != int(verification_input.total_tokens):
        raise MailboxVerifyApplyError(
            "mailbox verify result draft token length mismatch",
            next_required_feature="mailbox_verify_apply_plan_validation",
            error_kind="mailbox_verify_draft_length_mismatch",
        )
    if target_token_ids is not None and len(target_flat) != len(drafted_flat):
        raise MailboxVerifyApplyError(
            "mailbox verify target token length mismatch",
            next_required_feature="mailbox_verify_token_decision_backend",
            error_kind="mailbox_verify_target_length_mismatch",
        )

    drafted_by_seq: dict[int, list[int]] = {}
    target_by_seq: dict[int, list[int]] = {}
    positions_by_seq: dict[int, list[int]] = {}
    kv_slot_ids_by_seq: dict[int, list[int]] = {}
    positions_flat = [int(pos) for pos in (getattr(verification_input, "positions", []) or [])]
    kv_slot_ids_flat = [int(slot) for slot in (getattr(verification_input, "kv_slot_ids", []) or [])]
    accepted: dict[int, int] = {}
    rejected_seq_ids: list[int] = []
    rejected_positions: dict[int, list[int]] = {}
    invalidated_payload_ids: list[str] = []
    source_payload_ids = list(getattr(verification_input, "source_mailbox_payload_ids", []) or [])
    for idx, seq_id in enumerate(seq_ids):
        offset = offsets[idx]
        length = lengths[idx]
        drafted = drafted_flat[offset : offset + length]
        target = target_flat[offset : offset + length] if target_token_ids is not None else []
        drafted_by_seq[seq_id] = drafted
        target_by_seq[seq_id] = target
        positions_by_seq[seq_id] = positions_flat[offset : offset + length] if positions_flat else []
        kv_slot_ids_by_seq[seq_id] = kv_slot_ids_flat[offset : offset + length] if kv_slot_ids_flat else []
        if metadata_only:
            accepted_len = 0
        else:
            accepted_len = _accepted_prefix_length(drafted, target)
        accepted[seq_id] = accepted_len
        if accepted_len < length:
            rejected_seq_ids.append(seq_id)
            rejected_positions[seq_id] = list(range(accepted_len, length))
            if idx < len(source_payload_ids):
                invalidated_payload_ids.append(str(source_payload_ids[idx]))
        else:
            rejected_positions[seq_id] = []
    total_accepted = sum(accepted.values())
    total_rejected = sum(lengths) - total_accepted
    return MailboxVerifyResult(
        plan_id=getattr(verification_input, "plan_id", None),
        target_home_batch_id=getattr(verification_input, "target_home_batch_id", None),
        seq_ids=seq_ids,
        request_ids=list(getattr(verification_input, "request_ids", []) or []),
        per_seq_lengths=lengths,
        offsets=offsets,
        drafted_token_ids_by_seq=drafted_by_seq,
        target_token_ids_by_seq=target_by_seq,
        accepted_lengths_by_seq=accepted,
        rejected_seq_ids=rejected_seq_ids,
        rejected_token_positions_by_seq=rejected_positions,
        invalidated_mailbox_payload_ids=invalidated_payload_ids,
        mailbox_payload_ids=[str(payload_id) for payload_id in source_payload_ids],
        total_accepted_tokens=total_accepted,
        total_rejected_tokens=total_rejected,
        output_owner_rank=output_owner_rank,
        no_commit=True,
        metadata_only=metadata_only,
        next_required_feature="mailbox_verify_token_decision_backend" if metadata_only else None,
        positions_by_seq=positions_by_seq,
        kv_slot_ids_by_seq=kv_slot_ids_by_seq,
    )


def extract_target_token_ids_from_logits(logits: Any, total_tokens: int) -> list[int] | None:
    """Extract argmax token ids from tensor-like logits when safely available."""

    if logits is None or not hasattr(logits, "argmax"):
        return None
    try:
        token_ids = logits.argmax(dim=-1)
    except TypeError:
        try:
            token_ids = logits.argmax(axis=-1)
        except Exception:
            return None
    except Exception:
        return None
    try:
        values = token_ids.tolist()
    except Exception:
        return None
    if isinstance(values, int):
        values = [values]
    flat = [int(value) for value in values]
    if len(flat) != int(total_tokens):
        return None
    return flat


def build_mailbox_verify_apply_plan(
    verify_result: MailboxVerifyResult,
    exec_seqs: Iterable[Any],
    step_plan: Any,
    *,
    max_model_len: int | None = None,
    state_mutation_allowed: bool = False,
) -> MailboxVerifyApplyPlan:
    seq_list = list(exec_seqs)
    seq_ids = [int(seq_id) for seq_id in verify_result.seq_ids]
    expected = [int(seq_id) for seq_id in getattr(step_plan, "actual_target_exec_seq_ids", seq_ids)]
    if seq_ids != expected:
        raise MailboxVerifyApplyError(
            f"mailbox verify apply seq ids mismatch: seq_ids={seq_ids}, expected={expected}",
            next_required_feature="mailbox_verify_apply_plan_validation",
            error_kind="mailbox_verify_apply_seq_mismatch",
        )
    if verify_result.target_home_batch_id != getattr(step_plan, "target_home_batch_id", None):
        raise MailboxVerifyApplyError(
            "mailbox verify apply home_batch_id mismatch",
            next_required_feature="mailbox_verify_apply_plan_validation",
            error_kind="mailbox_verify_apply_home_batch_mismatch",
        )
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    if set(seq_by_id) != set(seq_ids):
        raise MailboxVerifyApplyError(
            f"mailbox verify apply missing Sequence state: seq_ids={seq_ids}, exec_seq_ids={sorted(seq_by_id)}",
            next_required_feature="mailbox_verify_apply_plan_validation",
            error_kind="mailbox_verify_apply_missing_sequence_state",
        )
    accepted_tokens: dict[int, list[int]] = {}
    rejected_tokens: dict[int, list[int]] = {}
    append_positions: dict[int, list[int]] = {}
    state_before: dict[int, JsonDict] = {}
    would_finish: list[int] = []
    would_continue: list[int] = []
    payloads_to_consume: list[str] = []
    payloads_to_invalidate: list[str] = list(verify_result.invalidated_mailbox_payload_ids)
    for seq_id, length in zip(seq_ids, verify_result.per_seq_lengths):
        accepted_len = int(verify_result.accepted_lengths_by_seq[seq_id])
        if accepted_len < 0 or accepted_len > int(length):
            raise MailboxVerifyApplyError(
                f"accepted length out of range for seq_id={seq_id}: accepted_len={accepted_len}, length={length}",
                next_required_feature="mailbox_verify_apply_plan_validation",
                error_kind="mailbox_verify_accepted_length_out_of_range",
            )
        seq = seq_by_id[seq_id]
        current_len = len(seq) if hasattr(seq, "__len__") else int(getattr(seq, "num_tokens", 0) or 0)
        end_len = current_len + accepted_len
        if max_model_len is not None and end_len > int(max_model_len):
            raise MailboxVerifyApplyError(
                f"mailbox verify apply would exceed max_model_len for seq_id={seq_id}: end_len={end_len}, max_model_len={max_model_len}",
                next_required_feature="mailbox_verify_apply_plan_validation",
                error_kind="mailbox_verify_apply_exceeds_max_model_len",
            )
        drafted = list(verify_result.drafted_token_ids_by_seq[seq_id])
        accepted_tokens[seq_id] = drafted[:accepted_len]
        rejected_tokens[seq_id] = drafted[accepted_len:]
        append_positions[seq_id] = list(range(current_len, end_len))
        state_before[seq_id] = _sequence_snapshot(seq)
        if accepted_len == int(length):
            payloads_to_consume.append(_payload_id_for_seq(verify_result, seq_id))
        else:
            would_continue.append(seq_id)
        if bool(getattr(seq, "is_finished", False)):
            would_finish.append(seq_id)
    return MailboxVerifyApplyPlan(
        seq_ids=seq_ids,
        accepted_token_ids_by_seq=accepted_tokens,
        rejected_token_ids_by_seq=rejected_tokens,
        accepted_lengths_by_seq=dict(verify_result.accepted_lengths_by_seq),
        append_positions_by_seq=append_positions,
        sequence_state_before=state_before,
        would_finish_seq_ids=would_finish,
        would_continue_seq_ids=would_continue,
        mailbox_payloads_to_consume=payloads_to_consume,
        mailbox_payloads_to_invalidate=payloads_to_invalidate,
        state_mutation_allowed=bool(state_mutation_allowed),
        no_commit=not state_mutation_allowed,
    )


def run_mailbox_verify_apply_no_commit_probe(
    apply_plan: MailboxVerifyApplyPlan,
    exec_seqs: Iterable[Any],
) -> MailboxVerifyApplyProbeResult:
    before = {int(seq.seq_id): _sequence_snapshot(seq) for seq in exec_seqs}
    try:
        if apply_plan.state_mutation_allowed:
            raise MailboxVerifyApplyError(
                "mailbox verify no-commit probe received mutation-enabled plan",
                next_required_feature="mailbox_verify_state_mutation_guard",
                error_kind="mailbox_verify_mutation_enabled_in_no_commit",
            )
        for seq_id, accepted_len in apply_plan.accepted_lengths_by_seq.items():
            if accepted_len != len(apply_plan.accepted_token_ids_by_seq[int(seq_id)]):
                raise MailboxVerifyApplyError(
                    "mailbox verify apply plan accepted token length mismatch",
                    next_required_feature="mailbox_verify_apply_plan_validation",
                    error_kind="mailbox_verify_apply_plan_length_mismatch",
                )
        after = {int(seq.seq_id): _sequence_snapshot(seq) for seq in exec_seqs}
        if after != before:
            raise MailboxVerifyApplyError(
                "mailbox verify no-commit probe detected Sequence state mutation",
                next_required_feature="mailbox_verify_state_mutation_guard",
                error_kind="mailbox_verify_state_mutation_detected",
            )
        return MailboxVerifyApplyProbeResult(attempted=True, success=True, plan=apply_plan)
    except MailboxVerifyApplyError as exc:
        return MailboxVerifyApplyProbeResult(
            attempted=True,
            success=False,
            plan=apply_plan,
            error_kind=exc.error_kind,
            error_message=str(exc),
            state_mutation_attempted=False,
            state_mutation_committed=False,
            state_mutation_rollback_success=True,
            next_required_feature=exc.next_required_feature,
        )



def build_mailbox_verify_commit_plan(
    verify_result: MailboxVerifyResult,
    apply_plan: MailboxVerifyApplyPlan,
    exec_seqs: Iterable[Any],
    step_plan: Any,
    *,
    commit_allowed: bool,
    commit_mode: str = "guarded_probe",
) -> MailboxVerifyCommitPlan:
    """Build a guarded sequence-level commit plan from a verified apply plan."""

    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    seq_ids = [int(seq_id) for seq_id in apply_plan.seq_ids]
    expected = [int(seq_id) for seq_id in getattr(step_plan, "actual_target_exec_seq_ids", seq_ids)]
    if seq_ids != expected:
        raise MailboxVerifyApplyError(
            f"mailbox verify commit seq ids mismatch: seq_ids={seq_ids}, expected={expected}",
            next_required_feature="mailbox_commit_rollback_validation",
            error_kind="mailbox_verify_commit_seq_mismatch",
        )
    if verify_result.target_home_batch_id != getattr(step_plan, "target_home_batch_id", None):
        raise MailboxVerifyApplyError(
            "mailbox verify commit home_batch_id mismatch",
            next_required_feature="mailbox_commit_rollback_validation",
            error_kind="mailbox_verify_commit_home_batch_mismatch",
        )
    if set(seq_by_id) != set(seq_ids):
        raise MailboxVerifyApplyError(
            f"mailbox verify commit missing Sequence state: seq_ids={seq_ids}, exec_seq_ids={sorted(seq_by_id)}",
            next_required_feature="mailbox_commit_rollback_validation",
            error_kind="mailbox_verify_commit_missing_sequence_state",
        )

    state_before: dict[int, JsonDict] = {}
    state_after_expected: dict[int, JsonDict] = {}
    rejected_seq_ids: list[int] = []
    invalidated_payload_ids = list(apply_plan.mailbox_payloads_to_invalidate)
    for seq_id, drafted_len in zip(seq_ids, verify_result.per_seq_lengths):
        seq = seq_by_id[seq_id]
        home_batch_id = getattr(seq, "home_batch_id", None)
        if home_batch_id != verify_result.target_home_batch_id:
            raise MailboxVerifyApplyError(
                f"mailbox verify commit Sequence home_batch_id mismatch for seq_id={seq_id}: "
                f"sequence_home_batch_id={home_batch_id}, target_home_batch_id={verify_result.target_home_batch_id}",
                next_required_feature="mailbox_commit_rollback_validation",
                error_kind="mailbox_verify_commit_sequence_home_batch_mismatch",
            )
        accepted_len = int(apply_plan.accepted_lengths_by_seq[seq_id])
        if accepted_len < 0 or accepted_len > int(drafted_len):
            raise MailboxVerifyApplyError(
                f"accepted length out of range for commit seq_id={seq_id}: accepted_len={accepted_len}, drafted_len={drafted_len}",
                next_required_feature="mailbox_commit_rollback_validation",
                error_kind="mailbox_verify_commit_accepted_length_out_of_range",
            )
        accepted_tokens = list(apply_plan.accepted_token_ids_by_seq[seq_id])
        rejected_tokens = list(apply_plan.rejected_token_ids_by_seq.get(seq_id, []))
        if len(accepted_tokens) != accepted_len:
            raise MailboxVerifyApplyError(
                f"mailbox verify commit accepted token length mismatch for seq_id={seq_id}",
                next_required_feature="mailbox_commit_rollback_validation",
                error_kind="mailbox_verify_commit_accepted_token_length_mismatch",
            )
        current_len = len(seq) if hasattr(seq, "__len__") else int(getattr(seq, "num_tokens", 0) or 0)
        append_positions = list(apply_plan.append_positions_by_seq.get(seq_id, []))
        expected_positions = list(range(current_len, current_len + accepted_len))
        if append_positions != expected_positions:
            raise MailboxVerifyApplyError(
                f"mailbox verify commit append position mismatch for seq_id={seq_id}: "
                f"append_positions={append_positions}, expected={expected_positions}",
                next_required_feature="mailbox_commit_rollback_validation",
                error_kind="mailbox_verify_commit_append_position_mismatch",
            )
        before = _sequence_snapshot(seq)
        after = dict(before)
        token_ids = list(before.get("token_ids") or []) + accepted_tokens
        trace_stats = json.loads(json.dumps(before.get("trace_stats", {}) or {}, default=str))
        if accepted_len:
            trace_stats["accepted_tokens"] = int(trace_stats.get("accepted_tokens") or 0) + accepted_len
        if rejected_tokens:
            trace_stats["invalidated_predraft_tokens"] = int(trace_stats.get("invalidated_predraft_tokens") or 0) + len(rejected_tokens)
        after.update(
            {
                "token_ids": token_ids,
                "num_tokens": int(before.get("num_tokens") or 0) + accepted_len,
                "last_token": token_ids[-1] if token_ids else before.get("last_token"),
                "output_token_count": max(len(token_ids) - int(before.get("num_prompt_tokens") or 0), 0),
                "trace_stats": trace_stats,
            }
        )
        state_before[seq_id] = before
        state_after_expected[seq_id] = after
        if rejected_tokens:
            rejected_seq_ids.append(seq_id)
    return MailboxVerifyCommitPlan(
        plan_id=getattr(step_plan, "plan_id", verify_result.plan_id),
        target_home_batch_id=verify_result.target_home_batch_id,
        seq_ids=seq_ids,
        request_ids=list(verify_result.request_ids),
        accepted_lengths_by_seq=dict(apply_plan.accepted_lengths_by_seq),
        accepted_token_ids_by_seq=dict(apply_plan.accepted_token_ids_by_seq),
        rejected_seq_ids=rejected_seq_ids,
        rejected_token_ids_by_seq=dict(apply_plan.rejected_token_ids_by_seq),
        invalidated_payload_ids=invalidated_payload_ids,
        mailbox_payloads_to_consume=list(apply_plan.mailbox_payloads_to_consume),
        mailbox_payloads_to_invalidate=list(apply_plan.mailbox_payloads_to_invalidate),
        sequence_state_before=state_before,
        sequence_state_after_expected=state_after_expected,
        kv_state_before={"kv_commit_available": False},
        kv_state_after_expected={"next_required_feature": "mailbox_payload_consume_invalidate", "commit_mode": "shadow_only"},
        commit_allowed=bool(commit_allowed),
        commit_mode=str(commit_mode),
    )



def build_mailbox_kv_commit_plan(
    verify_result: MailboxVerifyResult,
    commit_plan: MailboxVerifyCommitPlan,
    exec_seqs: Iterable[Any],
    step_plan: Any,
    *,
    max_model_len: int | None = None,
    commit_allowed: bool,
    commit_mode: str = "shadow_only",
) -> MailboxKVCommitPlan:
    """Build a guarded KV shadow commit plan for accepted mailbox tokens."""

    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    seq_ids = [int(seq_id) for seq_id in commit_plan.seq_ids]
    expected = [int(seq_id) for seq_id in getattr(step_plan, "actual_target_exec_seq_ids", seq_ids)]
    if seq_ids != expected:
        raise MailboxVerifyApplyError(
            f"mailbox KV commit seq ids mismatch: seq_ids={seq_ids}, expected={expected}",
            next_required_feature="kv_commit_rollback_validation",
            error_kind="mailbox_kv_commit_seq_mismatch",
        )
    if verify_result.target_home_batch_id != getattr(step_plan, "target_home_batch_id", None):
        raise MailboxVerifyApplyError(
            "mailbox KV commit home_batch_id mismatch",
            next_required_feature="kv_commit_rollback_validation",
            error_kind="mailbox_kv_commit_home_batch_mismatch",
        )
    if set(seq_by_id) != set(seq_ids):
        raise MailboxVerifyApplyError(
            f"mailbox KV commit missing Sequence state: seq_ids={seq_ids}, exec_seq_ids={sorted(seq_by_id)}",
            next_required_feature="kv_commit_rollback_validation",
            error_kind="mailbox_kv_commit_missing_sequence_state",
        )

    append_start: dict[int, int] = {}
    append_end: dict[int, int] = {}
    kv_positions: dict[int, list[int]] = {}
    kv_slots_or_blocks: dict[int, list[int]] = {}
    seq_len_before: dict[int, int] = {}
    seq_len_after: dict[int, int] = {}
    kv_len_before: dict[int, int] = {}
    kv_len_after: dict[int, int] = {}
    for seq_id, drafted_len in zip(seq_ids, verify_result.per_seq_lengths):
        seq = seq_by_id[seq_id]
        if getattr(seq, "home_batch_id", None) != verify_result.target_home_batch_id:
            raise MailboxVerifyApplyError(
                f"mailbox KV commit Sequence home_batch_id mismatch for seq_id={seq_id}",
                next_required_feature="kv_commit_rollback_validation",
                error_kind="mailbox_kv_commit_sequence_home_batch_mismatch",
            )
        accepted_len = int(commit_plan.accepted_lengths_by_seq[seq_id])
        if accepted_len < 0 or accepted_len > int(drafted_len):
            raise MailboxVerifyApplyError(
                f"accepted length out of range for KV commit seq_id={seq_id}: accepted_len={accepted_len}, drafted_len={drafted_len}",
                next_required_feature="kv_commit_rollback_validation",
                error_kind="mailbox_kv_commit_accepted_length_out_of_range",
            )
        before_len = len(seq) if hasattr(seq, "__len__") else int(getattr(seq, "num_tokens", 0) or 0)
        start = _sequence_output_length(seq)
        end = start + accepted_len
        total_end = before_len + accepted_len
        if max_model_len is not None and total_end > int(max_model_len):
            raise MailboxVerifyApplyError(
                f"mailbox KV commit would exceed max_model_len for seq_id={seq_id}: end_len={total_end}, max_model_len={max_model_len}",
                next_required_feature="kv_commit_rollback_validation",
                error_kind="mailbox_kv_commit_exceeds_max_model_len",
            )
        positions = list(verify_result.positions_by_seq.get(seq_id, []))[:accepted_len]
        if accepted_len and len(positions) != accepted_len:
            positions = list(range(before_len, total_end))
        slots = list(verify_result.kv_slot_ids_by_seq.get(seq_id, []))[:accepted_len]
        if accepted_len and len(slots) != accepted_len:
            block_table = list(getattr(seq, "block_table", []) or [])
            if not block_table:
                raise MailboxVerifyApplyError(
                    f"mailbox KV commit has no slot/block mapping for seq_id={seq_id}",
                    next_required_feature="kv_slot_mapping_after_mailbox_verify",
                    error_kind="mailbox_kv_commit_missing_slot_mapping",
                )
            slots = block_table
        append_start[seq_id] = start
        append_end[seq_id] = end
        kv_positions[seq_id] = positions
        kv_slots_or_blocks[seq_id] = slots
        seq_len_before[seq_id] = before_len
        seq_len_after[seq_id] = total_end
        kv_before = _kv_shadow_length(seq, default=before_len)
        kv_len_before[seq_id] = kv_before
        kv_len_after[seq_id] = kv_before + accepted_len
    return MailboxKVCommitPlan(
        plan_id=commit_plan.plan_id,
        target_home_batch_id=commit_plan.target_home_batch_id,
        seq_ids=seq_ids,
        request_ids=list(commit_plan.request_ids),
        accepted_lengths_by_seq=dict(commit_plan.accepted_lengths_by_seq),
        accepted_token_ids_by_seq=dict(commit_plan.accepted_token_ids_by_seq),
        append_start_positions_by_seq=append_start,
        append_end_positions_by_seq=append_end,
        kv_positions_by_seq=kv_positions,
        kv_slots_or_blocks_by_seq=kv_slots_or_blocks,
        sequence_length_before_by_seq=seq_len_before,
        sequence_length_after_by_seq=seq_len_after,
        kv_length_before_by_seq=kv_len_before,
        kv_length_after_by_seq=kv_len_after,
        commit_allowed=bool(commit_allowed),
        commit_mode=str(commit_mode),
    )


def run_mailbox_kv_commit_probe(
    kv_commit_plan: MailboxKVCommitPlan,
    exec_seqs: Iterable[Any],
    *,
    current_rank: int | None = None,
    output_owner_rank: int | None = None,
    commit_enabled: bool = True,
) -> MailboxKVCommitResult:
    """Apply a rollback-safe shadow KV commit for mailbox accepted tokens."""

    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    before = {int(seq.seq_id): _kv_shadow_snapshot(seq) for seq in seq_list}
    if output_owner_rank is not None and current_rank is not None and int(current_rank) != int(output_owner_rank):
        return MailboxKVCommitResult(
            attempted=False,
            success=True,
            plan=kv_commit_plan,
            shadow_only=True,
            skipped_non_owner=True,
            kv_state_before=before,
            kv_state_after=before,
        )
    if not commit_enabled or not kv_commit_plan.commit_allowed:
        return MailboxKVCommitResult(
            attempted=False,
            success=True,
            plan=kv_commit_plan,
            shadow_only=True,
            kv_state_before=before,
            kv_state_after=before,
        )

    mutated_seq_ids: list[int] = []
    try:
        for seq_id in kv_commit_plan.seq_ids:
            if seq_id not in seq_by_id:
                raise MailboxVerifyApplyError(
                    f"mailbox KV commit missing Sequence during commit: seq_id={seq_id}",
                    next_required_feature="kv_commit_rollback_validation",
                    error_kind="mailbox_kv_commit_missing_sequence_state",
                )
            seq = seq_by_id[seq_id]
            expected_seq_len = int(kv_commit_plan.sequence_length_after_by_seq[seq_id])
            current_seq_len = len(seq) if hasattr(seq, "__len__") else int(getattr(seq, "num_tokens", 0) or 0)
            if current_seq_len != expected_seq_len:
                raise MailboxVerifyApplyError(
                    f"mailbox KV commit sequence length mismatch for seq_id={seq_id}: current={current_seq_len}, expected={expected_seq_len}",
                    next_required_feature="kv_commit_rollback_validation",
                    error_kind="mailbox_kv_commit_sequence_length_mismatch",
                )
            if bool(getattr(seq, "fail_kv_commit", False)):
                raise MailboxVerifyApplyError(
                    f"injected mailbox KV commit failure for seq_id={seq_id}",
                    next_required_feature="kv_commit_rollback_validation",
                    error_kind="mailbox_kv_commit_injected_failure",
                )
            accepted_len = int(kv_commit_plan.accepted_lengths_by_seq[seq_id])
            if accepted_len:
                positions = list(kv_commit_plan.kv_positions_by_seq.get(seq_id, []))
                slots = list(kv_commit_plan.kv_slots_or_blocks_by_seq.get(seq_id, []))
                if len(positions) != accepted_len or not slots:
                    raise MailboxVerifyApplyError(
                        f"mailbox KV commit append mapping invalid for seq_id={seq_id}",
                        next_required_feature="kv_slot_mapping_after_mailbox_verify",
                        error_kind="mailbox_kv_commit_invalid_slot_mapping",
                    )
            _set_kv_shadow_state(
                seq,
                {
                    "plan_id": kv_commit_plan.plan_id,
                    "target_home_batch_id": kv_commit_plan.target_home_batch_id,
                    "seq_id": seq_id,
                    "shadow_only": True,
                    "length": int(kv_commit_plan.kv_length_after_by_seq[seq_id]),
                    "append_start_position": int(kv_commit_plan.append_start_positions_by_seq[seq_id]),
                    "append_end_position": int(kv_commit_plan.append_end_positions_by_seq[seq_id]),
                    "positions": list(kv_commit_plan.kv_positions_by_seq.get(seq_id, [])),
                    "slots_or_blocks": list(kv_commit_plan.kv_slots_or_blocks_by_seq.get(seq_id, [])),
                },
            )
            mutated_seq_ids.append(seq_id)
        after = {int(seq.seq_id): _kv_shadow_snapshot(seq) for seq in seq_list}
        return MailboxKVCommitResult(
            attempted=True,
            success=True,
            plan=kv_commit_plan,
            shadow_only=True,
            next_required_feature="mailbox_payload_consume_invalidate",
            kv_state_before=before,
            kv_state_after=after,
        )
    except MailboxVerifyApplyError as exc:
        rollback_attempted = bool(mutated_seq_ids)
        rollback_success = True
        if rollback_attempted:
            rollback_success = _rollback_kv_shadow(seq_by_id, before, mutated_seq_ids)
        after = {int(seq.seq_id): _kv_shadow_snapshot(seq) for seq in seq_list}
        return MailboxKVCommitResult(
            attempted=True,
            success=False,
            plan=kv_commit_plan,
            shadow_only=True,
            error_kind=exc.error_kind,
            error_message=str(exc),
            next_required_feature=exc.next_required_feature if rollback_success else "kv_commit_rollback_validation",
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
            kv_state_before=before,
            kv_state_after=after,
        )
    except Exception as exc:
        rollback_attempted = bool(mutated_seq_ids)
        rollback_success = True
        if rollback_attempted:
            rollback_success = _rollback_kv_shadow(seq_by_id, before, mutated_seq_ids)
        after = {int(seq.seq_id): _kv_shadow_snapshot(seq) for seq in seq_list}
        return MailboxKVCommitResult(
            attempted=True,
            success=False,
            plan=kv_commit_plan,
            shadow_only=True,
            error_kind=type(exc).__name__,
            error_message=f"mailbox KV commit unexpected error: {exc}",
            next_required_feature="kv_commit_rollback_validation",
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
            kv_state_before=before,
            kv_state_after=after,
        )


def build_mailbox_payload_consume_plan(
    commit_plan: MailboxVerifyCommitPlan,
    mailbox: Any,
) -> MailboxPayloadConsumePlan:
    payload_ids = sorted(
        set(str(payload_id) for payload_id in commit_plan.mailbox_payloads_to_consume)
        | set(str(payload_id) for payload_id in commit_plan.mailbox_payloads_to_invalidate),
        key=str,
    )
    state_before = mailbox.lifecycle_snapshot(payload_ids) if hasattr(mailbox, "lifecycle_snapshot") else {}
    consumed_by_seq: dict[int, int] = {}
    invalidated_by_seq: dict[int, int] = {}
    consumed_payload_ids: list[str] = []
    invalidated_payload_ids: list[str] = []
    state_after: dict[str, JsonDict] = {}
    for seq_id in commit_plan.seq_ids:
        accepted_len = int(commit_plan.accepted_lengths_by_seq.get(seq_id, 0) or 0)
        rejected_len = len(commit_plan.rejected_token_ids_by_seq.get(seq_id, []))
        payload_id = _payload_id_for_commit_plan(commit_plan, seq_id)
        lifecycle = state_before.get(payload_id)
        if lifecycle is None or lifecycle.get("lifecycle_state") == "missing":
            raise MailboxVerifyApplyError(
                f"mailbox payload consume plan missing lifecycle for payload_id={payload_id}",
                next_required_feature="mailbox_payload_consume_rollback_validation",
                error_kind="mailbox_payload_consume_missing_lifecycle",
            )
        if lifecycle.get("home_batch_id") != commit_plan.target_home_batch_id:
            raise MailboxVerifyApplyError(
                f"mailbox payload consume plan home_batch_id mismatch for payload_id={payload_id}",
                next_required_feature="mailbox_payload_consume_rollback_validation",
                error_kind="mailbox_payload_consume_home_batch_mismatch",
            )
        if int(lifecycle.get("seq_id")) != int(seq_id):
            raise MailboxVerifyApplyError(
                f"mailbox payload consume plan seq_id mismatch for payload_id={payload_id}",
                next_required_feature="mailbox_payload_consume_rollback_validation",
                error_kind="mailbox_payload_consume_seq_mismatch",
            )
        if lifecycle.get("lifecycle_state") != "available":
            raise MailboxVerifyApplyError(
                f"mailbox payload duplicate consume detected for payload_id={payload_id}",
                next_required_feature="mailbox_payload_consume_rollback_validation",
                error_kind="duplicate_consume_error",
            )
        consumed_by_seq[seq_id] = accepted_len
        invalidated_by_seq[seq_id] = rejected_len
        if accepted_len > 0:
            consumed_payload_ids.append(payload_id)
        if rejected_len > 0:
            invalidated_payload_ids.append(payload_id)
        after = dict(lifecycle)
        after["consumed_by_plan_id"] = commit_plan.plan_id if accepted_len > 0 else lifecycle.get("consumed_by_plan_id")
        after["invalidated_by_plan_id"] = commit_plan.plan_id if rejected_len > 0 else lifecycle.get("invalidated_by_plan_id")
        after["consumed_token_count"] = accepted_len
        after["invalidated_token_count"] = rejected_len
        after["lifecycle_state"] = "invalidated" if rejected_len > 0 else "consumed"
        state_after[payload_id] = after
    return MailboxPayloadConsumePlan(
        plan_id=commit_plan.plan_id,
        target_home_batch_id=commit_plan.target_home_batch_id,
        seq_ids=list(commit_plan.seq_ids),
        consumed_payload_ids=consumed_payload_ids,
        invalidated_payload_ids=invalidated_payload_ids,
        consumed_token_count_by_seq=consumed_by_seq,
        invalidated_token_count_by_seq=invalidated_by_seq,
        accepted_lengths_by_seq=dict(commit_plan.accepted_lengths_by_seq),
        rejected_seq_ids=list(commit_plan.rejected_seq_ids),
        mailbox_state_before=state_before,
        mailbox_state_after_expected=state_after,
    )


def run_mailbox_payload_consume_probe(
    consume_plan: MailboxPayloadConsumePlan,
    mailbox: Any,
    *,
    current_rank: int | None = None,
    output_owner_rank: int | None = None,
    next_pipeline_plan_id: int | None = None,
    continue_after_commit: bool = False,
    continuation_context: JsonDict | None = None,
) -> MailboxPayloadConsumeResult:
    payload_ids = sorted(
        set(consume_plan.consumed_payload_ids) | set(consume_plan.invalidated_payload_ids),
        key=str,
    )
    before = mailbox.lifecycle_snapshot(payload_ids) if hasattr(mailbox, "lifecycle_snapshot") else dict(consume_plan.mailbox_state_before)
    if output_owner_rank is not None and current_rank is not None and int(current_rank) != int(output_owner_rank):
        return MailboxPayloadConsumeResult(
            attempted=False,
            success=True,
            plan=consume_plan,
            mailbox_state_before=before,
            mailbox_state_after=before,
            skipped_non_owner=True,
            invalidate_skipped_non_owner=True,
            next_pipeline_step_skipped_non_owner=True,
        )
    consumed_by_payload: dict[str, int] = {}
    invalidated_by_payload: dict[str, int] = {}
    for seq_id in consume_plan.seq_ids:
        payload_id = _payload_id_for_consume_plan(consume_plan, seq_id)
        consumed_by_payload[payload_id] = int(consume_plan.consumed_token_count_by_seq.get(seq_id, 0) or 0)
        invalidated_by_payload[payload_id] = int(consume_plan.invalidated_token_count_by_seq.get(seq_id, 0) or 0)
    try:
        if not hasattr(mailbox, "apply_payload_lifecycle"):
            raise MailboxVerifyApplyError(
                "mailbox payload lifecycle backend unavailable",
                next_required_feature="mailbox_payload_consume_rollback_validation",
                error_kind="mailbox_payload_lifecycle_backend_unavailable",
            )
        mailbox.apply_payload_lifecycle(
            plan_id=consume_plan.plan_id,
            target_home_batch_id=consume_plan.target_home_batch_id,
            consumed_token_count_by_payload_id=consumed_by_payload,
            invalidated_token_count_by_payload_id=invalidated_by_payload,
        )
        after = mailbox.lifecycle_snapshot(payload_ids) if hasattr(mailbox, "lifecycle_snapshot") else {}
        if after != consume_plan.mailbox_state_after_expected:
            raise MailboxVerifyApplyError(
                "mailbox payload lifecycle post-state mismatch",
                next_required_feature="mailbox_payload_consume_rollback_validation",
                error_kind="mailbox_payload_lifecycle_post_state_mismatch",
            )
        continuation = _build_next_pipeline_continuation_metadata(
            consume_plan,
            after,
            next_pipeline_plan_id=next_pipeline_plan_id,
            continue_after_commit=continue_after_commit,
            continuation_context=continuation_context,
        )
        return MailboxPayloadConsumeResult(
            attempted=True,
            success=True,
            plan=consume_plan,
            next_required_feature=continuation["next_required_feature"],
            consumed_payload_ids=list(consume_plan.consumed_payload_ids),
            invalidated_payload_ids=list(consume_plan.invalidated_payload_ids),
            consumed_token_count=sum(consume_plan.consumed_token_count_by_seq.values()),
            invalidated_token_count=sum(consume_plan.invalidated_token_count_by_seq.values()),
            mailbox_state_before=before,
            mailbox_state_after=after,
            next_pipeline_step_attempted=bool(continuation["next_pipeline_step_attempted"]),
            next_pipeline_step_success=bool(continuation["next_pipeline_step_success"]),
            next_pipeline_step_error=continuation.get("next_pipeline_step_error"),
            next_pipeline_step_error_kind=continuation.get("next_pipeline_step_error_kind"),
            next_pipeline_plan_id=continuation.get("next_pipeline_plan_id"),
            next_pipeline_target_home_batch_id=continuation.get("next_pipeline_target_home_batch_id"),
            next_pipeline_draft_home_batch_id=continuation.get("next_pipeline_draft_home_batch_id"),
            next_pipeline_actual_target_seq_ids=list(continuation.get("next_pipeline_actual_target_seq_ids") or []),
            next_pipeline_actual_draft_seq_ids=list(continuation.get("next_pipeline_actual_draft_seq_ids") or []),
            previous_committed_plan_id=continuation.get("previous_committed_plan_id"),
            previous_consumed_payload_ids=list(continuation.get("previous_consumed_payload_ids") or []),
            previous_invalidated_payload_ids=list(continuation.get("previous_invalidated_payload_ids") or []),
            duplicate_payload_consume_after_continue=bool(continuation.get("duplicate_payload_consume_after_continue")),
            pipeline_state_after_commit_valid=bool(continuation.get("pipeline_state_after_commit_valid")),
            scheduler_state_after_commit_valid=bool(continuation.get("scheduler_state_after_commit_valid")),
            breadth_only_step_count=int(continuation.get("breadth_only_step_count") or 0),
            current_pipeline_step=int(continuation.get("current_pipeline_step") or 0),
            current_plan_id=continuation.get("current_plan_id"),
            next_plan_id=continuation.get("next_plan_id"),
            previous_target_home_batch_id=continuation.get("previous_target_home_batch_id"),
            previous_draft_home_batch_id=continuation.get("previous_draft_home_batch_id"),
            current_target_home_batch_id=continuation.get("current_target_home_batch_id"),
            current_draft_home_batch_id=continuation.get("current_draft_home_batch_id"),
            active_seq_ids_before_second_step=list(continuation.get("active_seq_ids_before_second_step") or []),
            active_seq_ids_after_second_step=list(continuation.get("active_seq_ids_after_second_step") or []),
            committed_seq_ids=list(continuation.get("committed_seq_ids") or []),
            available_mailbox_payload_ids=list(continuation.get("available_mailbox_payload_ids") or []),
            pending_mailbox_payload_ids=list(continuation.get("pending_mailbox_payload_ids") or []),
            second_step_state_check_attempted=bool(continuation.get("second_step_state_check_attempted")),
            second_step_state_check_success=bool(continuation.get("second_step_state_check_success")),
            second_step_state_error=continuation.get("second_step_state_error"),
            second_step_state_error_kind=continuation.get("second_step_state_error_kind"),
            repeated_verify_after_commit_detected=bool(continuation.get("repeated_verify_after_commit_detected")),
            scheduler_state_after_second_step_valid=bool(continuation.get("scheduler_state_after_second_step_valid")),
            sequence_state_after_second_step_valid=bool(continuation.get("sequence_state_after_second_step_valid")),
            mailbox_state_after_second_step_valid=bool(continuation.get("mailbox_state_after_second_step_valid")),
            request_completion_check_attempted=bool(continuation.get("request_completion_check_attempted")),
            request_completion_check_success=bool(continuation.get("request_completion_check_success")),
            second_step_rollback_attempted=bool(continuation.get("second_step_rollback_attempted")),
            second_step_rollback_success=bool(continuation.get("second_step_rollback_success", True)),
            request_completion_reason=continuation.get("request_completion_reason"),
            active_seq_ids_at_completion_check=list(continuation.get("active_seq_ids_at_completion_check") or []),
            finished_seq_ids_at_completion_check=list(continuation.get("finished_seq_ids_at_completion_check") or []),
            unfinished_seq_ids_at_completion_check=list(continuation.get("unfinished_seq_ids_at_completion_check") or []),
            max_tokens_reached_seq_ids=list(continuation.get("max_tokens_reached_seq_ids") or []),
            eos_reached_seq_ids=list(continuation.get("eos_reached_seq_ids") or []),
            mailbox_pending_payload_ids_at_completion=list(continuation.get("mailbox_pending_payload_ids_at_completion") or []),
            mailbox_consumed_payload_ids_at_completion=list(continuation.get("mailbox_consumed_payload_ids_at_completion") or []),
            scheduler_active_seq_ids_at_completion=list(continuation.get("scheduler_active_seq_ids_at_completion") or []),
            sequence_state_completion_valid=bool(continuation.get("sequence_state_completion_valid")),
            scheduler_state_completion_valid=bool(continuation.get("scheduler_state_completion_valid")),
            mailbox_state_completion_valid=bool(continuation.get("mailbox_state_completion_valid")),
            request_completion_error=continuation.get("request_completion_error"),
            request_completion_error_kind=continuation.get("request_completion_error_kind"),
            result_finalization_attempted=bool(continuation.get("result_finalization_attempted")),
            result_finalization_success=bool(continuation.get("result_finalization_success")),
            result_finalization_error=continuation.get("result_finalization_error"),
            breadth_only_completed=bool(continuation.get("breadth_only_completed")),
            breadth_only_completion_reason=continuation.get("breadth_only_completion_reason"),
        )
    except Exception as exc:
        rollback_attempted = True
        rollback_success = False
        if hasattr(mailbox, "restore_lifecycle_snapshot"):
            rollback_success = bool(mailbox.restore_lifecycle_snapshot(before))
        after = mailbox.lifecycle_snapshot(payload_ids) if hasattr(mailbox, "lifecycle_snapshot") else {}
        error_kind = getattr(exc, "error_kind", getattr(exc, "kind", type(exc).__name__))
        error_message = str(exc)
        duplicate = str(error_kind) == "duplicate_consume_error"
        return MailboxPayloadConsumeResult(
            attempted=True,
            success=False,
            plan=consume_plan,
            error_kind=str(error_kind),
            error_message=error_message,
            next_required_feature="mailbox_payload_consume_rollback_validation",
            consumed_payload_ids=list(consume_plan.consumed_payload_ids),
            invalidated_payload_ids=list(consume_plan.invalidated_payload_ids),
            consumed_token_count=sum(consume_plan.consumed_token_count_by_seq.values()),
            invalidated_token_count=sum(consume_plan.invalidated_token_count_by_seq.values()),
            mailbox_state_before=before,
            mailbox_state_after=after,
            duplicate_consume_detected=duplicate,
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
        )

def run_mailbox_verify_commit_probe(
    commit_plan: MailboxVerifyCommitPlan,
    exec_seqs: Iterable[Any],
    *,
    current_rank: int | None = None,
    output_owner_rank: int | None = None,
    eos_token_id: int | list[int] | None = None,
    commit_enabled: bool = True,
    kv_commit_plan: MailboxKVCommitPlan | None = None,
    payload_consume_plan: MailboxPayloadConsumePlan | None = None,
    mailbox: Any | None = None,
    next_pipeline_plan_id: int | None = None,
    continue_after_commit: bool = False,
    continuation_context: JsonDict | None = None,
) -> MailboxVerifyCommitResult:
    """Apply a guarded sequence-only commit with rollback on any failure.

    The V4N path performs a shadow-only KV metadata commit when a
    ``MailboxKVCommitPlan`` is supplied; after Sequence/KV shadow commit success
    the result points to mailbox payload consume/invalidate.
    """

    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    total_accepted = sum(len(tokens) for tokens in commit_plan.accepted_token_ids_by_seq.values())
    total_rejected = sum(len(tokens) for tokens in commit_plan.rejected_token_ids_by_seq.values())
    before = {int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list}
    if output_owner_rank is not None and current_rank is not None and int(current_rank) != int(output_owner_rank):
        return MailboxVerifyCommitResult(
            attempted=False,
            success=True,
            plan=commit_plan,
            skipped_non_owner=True,
            kv_commit_skipped_non_owner=True,
            next_pipeline_step_skipped_non_owner=True,
            sequence_state_before=before,
            sequence_state_after=before,
            total_accepted_tokens=total_accepted,
            total_rejected_tokens=total_rejected,
        )
    if not commit_enabled or not commit_plan.commit_allowed:
        return MailboxVerifyCommitResult(
            attempted=False,
            success=True,
            plan=commit_plan,
            sequence_state_before=before,
            sequence_state_after=before,
            total_accepted_tokens=total_accepted,
            total_rejected_tokens=total_rejected,
        )

    mutated_seq_ids: list[int] = []
    rollback_attempted = False
    rollback_success = True
    try:
        for seq_id in commit_plan.seq_ids:
            if seq_id not in seq_by_id:
                raise MailboxVerifyApplyError(
                    f"mailbox verify commit missing Sequence during commit: seq_id={seq_id}",
                    next_required_feature="mailbox_commit_rollback_validation",
                    error_kind="mailbox_verify_commit_missing_sequence_state",
                )
            seq = seq_by_id[seq_id]
            snapshot = commit_plan.sequence_state_before.get(seq_id)
            if snapshot is None:
                raise MailboxVerifyApplyError(
                    f"mailbox verify commit missing snapshot: seq_id={seq_id}",
                    next_required_feature="mailbox_commit_rollback_validation",
                    error_kind="mailbox_verify_commit_missing_snapshot",
                )
            if _sequence_snapshot(seq) != snapshot:
                raise MailboxVerifyApplyError(
                    f"mailbox verify commit stale Sequence snapshot: seq_id={seq_id}",
                    next_required_feature="mailbox_commit_rollback_validation",
                    error_kind="mailbox_verify_commit_stale_sequence_snapshot",
                )
            accepted_tokens = list(commit_plan.accepted_token_ids_by_seq.get(seq_id, []))
            if accepted_tokens:
                if seq_id not in mutated_seq_ids:
                    mutated_seq_ids.append(seq_id)
                for token in accepted_tokens:
                    if not hasattr(seq, "append_token"):
                        raise MailboxVerifyApplyError(
                            f"mailbox verify commit Sequence cannot append token: seq_id={seq_id}",
                            next_required_feature="mailbox_commit_rollback_validation",
                            error_kind="mailbox_verify_commit_append_unavailable",
                        )
                    seq.append_token(int(token))
                    if eos_token_id is not None and _is_eos_token(int(token), eos_token_id) and not bool(getattr(seq, "ignore_eos", False)):
                        if hasattr(seq, "mark_finished"):
                            seq.mark_finished(record_finish_ts=False)
                        else:
                            raise MailboxVerifyApplyError(
                                f"mailbox verify commit EOS handling unavailable: seq_id={seq_id}",
                                next_required_feature="mailbox_commit_rollback_validation",
                                error_kind="mailbox_verify_commit_eos_handling_unavailable",
                            )
            rejected_len = len(commit_plan.rejected_token_ids_by_seq.get(seq_id, []))
            if rejected_len and hasattr(seq, "record_invalidated_predraft"):
                if seq_id not in mutated_seq_ids:
                    mutated_seq_ids.append(seq_id)
                seq.record_invalidated_predraft(rejected_len)
            accepted_len = len(accepted_tokens)
            if accepted_len and hasattr(seq, "record_accepted"):
                if seq_id not in mutated_seq_ids:
                    mutated_seq_ids.append(seq_id)
                seq.record_accepted(accepted_len)
        after = {int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list}
        for seq_id in commit_plan.seq_ids:
            if not _snapshots_equal(after.get(seq_id), commit_plan.sequence_state_after_expected.get(seq_id), ignore_volatile_timestamps=True):
                raise MailboxVerifyApplyError(
                    f"mailbox verify commit post-state mismatch: seq_id={seq_id}",
                    next_required_feature="mailbox_commit_rollback_validation",
                    error_kind="mailbox_verify_commit_post_state_mismatch",
                )
        kv_result: MailboxKVCommitResult | None = None
        if kv_commit_plan is not None:
            kv_result = run_mailbox_kv_commit_probe(
                kv_commit_plan,
                seq_list,
                current_rank=current_rank,
                output_owner_rank=output_owner_rank,
                commit_enabled=commit_enabled,
            )
            if not kv_result.success:
                kv_rollback_success = kv_result.rollback_success
                seq_rollback_success = _rollback_sequences(seq_by_id, commit_plan.sequence_state_before, commit_plan.seq_ids)
                after_rollback = {int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list}
                combined_rollback_success = bool(kv_rollback_success and seq_rollback_success)
                return MailboxVerifyCommitResult(
                    attempted=True,
                    success=False,
                    plan=commit_plan,
                    error_kind=kv_result.error_kind,
                    error_message=kv_result.error_message,
                    next_required_feature=kv_result.next_required_feature if combined_rollback_success else "kv_commit_rollback_validation",
                    sequence_state_commit_attempted=True,
                    sequence_state_commit_success=False,
                    sequence_state_before=before,
                    sequence_state_after=after_rollback,
                    kv_commit_plan_built=True,
                    kv_commit_attempted=kv_result.attempted,
                    kv_commit_success=False,
                    kv_commit_shadow_only=kv_result.shadow_only,
                    kv_commit_error=kv_result.error_message,
                    kv_commit_error_kind=kv_result.error_kind,
                    kv_commit_rollback_attempted=bool(kv_result.rollback_attempted),
                    kv_commit_rollback_success=bool(kv_result.rollback_success),
                    kv_commit_skipped_non_owner=bool(kv_result.skipped_non_owner),
                    mailbox_payload_consume_attempted=False,
                    mailbox_payload_consume_success=False,
                    mailbox_payload_invalidate_attempted=False,
                    mailbox_payload_invalidate_success=False,
                    rollback_attempted=True,
                    rollback_success=combined_rollback_success,
                    total_accepted_tokens=total_accepted,
                    total_rejected_tokens=total_rejected,
                )
        payload_result: MailboxPayloadConsumeResult | None = None
        if payload_consume_plan is not None and mailbox is not None:
            payload_result = run_mailbox_payload_consume_probe(
                payload_consume_plan,
                mailbox,
                current_rank=current_rank,
                output_owner_rank=output_owner_rank,
                next_pipeline_plan_id=next_pipeline_plan_id,
                continue_after_commit=continue_after_commit,
                continuation_context=continuation_context,
            )
            if not payload_result.success:
                kv_rollback_success = True
                if kv_result is not None and kv_result.kv_state_before:
                    kv_rollback_success = _rollback_kv_shadow(seq_by_id, kv_result.kv_state_before, commit_plan.seq_ids)
                seq_rollback_success = _rollback_sequences(seq_by_id, commit_plan.sequence_state_before, commit_plan.seq_ids)
                after_rollback = {int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list}
                combined_rollback_success = bool(payload_result.rollback_success and kv_rollback_success and seq_rollback_success)
                return MailboxVerifyCommitResult(
                    attempted=True,
                    success=False,
                    plan=commit_plan,
                    error_kind=payload_result.error_kind,
                    error_message=payload_result.error_message,
                    next_required_feature=payload_result.next_required_feature if combined_rollback_success else "mailbox_payload_consume_rollback_validation",
                    sequence_state_commit_attempted=True,
                    sequence_state_commit_success=False,
                    sequence_state_before=before,
                    sequence_state_after=after_rollback,
                    kv_commit_plan_built=kv_commit_plan is not None,
                    kv_commit_attempted=bool(kv_result.attempted) if kv_result is not None else False,
                    kv_commit_success=bool(kv_result.success) if kv_result is not None else False,
                    kv_commit_shadow_only=bool(kv_result.shadow_only) if kv_result is not None else False,
                    kv_commit_error=kv_result.error_message if kv_result is not None else None,
                    kv_commit_error_kind=kv_result.error_kind if kv_result is not None else None,
                    kv_commit_rollback_attempted=True,
                    kv_commit_rollback_success=kv_rollback_success,
                    kv_commit_skipped_non_owner=bool(kv_result.skipped_non_owner) if kv_result is not None else False,
                    mailbox_payload_consume_plan_built=True,
                    mailbox_payload_consume_attempted=payload_result.attempted,
                    mailbox_payload_consume_success=False,
                    mailbox_payload_consume_error=payload_result.error_message,
                    mailbox_payload_consume_error_kind=payload_result.error_kind,
                    mailbox_payload_consumed_payload_ids=payload_result.consumed_payload_ids,
                    mailbox_payload_consumed_token_count=payload_result.consumed_token_count,
                    mailbox_payload_invalidate_attempted=payload_result.attempted,
                    mailbox_payload_invalidate_success=False,
                    mailbox_payload_invalidate_error=payload_result.error_message,
                    mailbox_payload_invalidated_payload_ids=payload_result.invalidated_payload_ids,
                    mailbox_payload_invalidated_token_count=payload_result.invalidated_token_count,
                    mailbox_payload_lifecycle_before=payload_result.mailbox_state_before,
                    mailbox_payload_lifecycle_after=payload_result.mailbox_state_after,
                    mailbox_payload_duplicate_consume_detected=payload_result.duplicate_consume_detected,
                    mailbox_payload_consume_rollback_attempted=payload_result.rollback_attempted,
                    mailbox_payload_consume_rollback_success=payload_result.rollback_success,
                    mailbox_payload_consume_skipped_non_owner=payload_result.skipped_non_owner,
                    mailbox_payload_invalidate_skipped_non_owner=payload_result.invalidate_skipped_non_owner,
                    next_pipeline_step_attempted=payload_result.next_pipeline_step_attempted,
                    next_pipeline_step_success=payload_result.next_pipeline_step_success,
                    next_pipeline_step_error=payload_result.next_pipeline_step_error,
                    next_pipeline_step_error_kind=payload_result.next_pipeline_step_error_kind,
                    next_pipeline_plan_id=payload_result.next_pipeline_plan_id,
                    next_pipeline_target_home_batch_id=payload_result.next_pipeline_target_home_batch_id,
                    next_pipeline_draft_home_batch_id=payload_result.next_pipeline_draft_home_batch_id,
                    next_pipeline_actual_target_seq_ids=payload_result.next_pipeline_actual_target_seq_ids,
                    next_pipeline_actual_draft_seq_ids=payload_result.next_pipeline_actual_draft_seq_ids,
                    previous_committed_plan_id=payload_result.previous_committed_plan_id,
                    previous_consumed_payload_ids=payload_result.previous_consumed_payload_ids,
                    previous_invalidated_payload_ids=payload_result.previous_invalidated_payload_ids,
                    duplicate_payload_consume_after_continue=payload_result.duplicate_payload_consume_after_continue,
                    pipeline_state_after_commit_valid=payload_result.pipeline_state_after_commit_valid,
                    scheduler_state_after_commit_valid=payload_result.scheduler_state_after_commit_valid,
                    breadth_only_step_count=payload_result.breadth_only_step_count,
                    current_pipeline_step=payload_result.current_pipeline_step,
                    current_plan_id=payload_result.current_plan_id,
                    next_plan_id=payload_result.next_plan_id,
                    previous_target_home_batch_id=payload_result.previous_target_home_batch_id,
                    previous_draft_home_batch_id=payload_result.previous_draft_home_batch_id,
                    current_target_home_batch_id=payload_result.current_target_home_batch_id,
                    current_draft_home_batch_id=payload_result.current_draft_home_batch_id,
                    active_seq_ids_before_second_step=payload_result.active_seq_ids_before_second_step,
                    active_seq_ids_after_second_step=payload_result.active_seq_ids_after_second_step,
                    committed_seq_ids=payload_result.committed_seq_ids,
                    consumed_payload_ids=payload_result.consumed_payload_ids,
                    invalidated_payload_ids=payload_result.invalidated_payload_ids,
                    available_mailbox_payload_ids=payload_result.available_mailbox_payload_ids,
                    pending_mailbox_payload_ids=payload_result.pending_mailbox_payload_ids,
                    second_step_state_check_attempted=payload_result.second_step_state_check_attempted,
                    second_step_state_check_success=payload_result.second_step_state_check_success,
                    second_step_state_error=payload_result.second_step_state_error,
                    second_step_state_error_kind=payload_result.second_step_state_error_kind,
                    repeated_verify_after_commit_detected=payload_result.repeated_verify_after_commit_detected,
                    scheduler_state_after_second_step_valid=payload_result.scheduler_state_after_second_step_valid,
                    sequence_state_after_second_step_valid=payload_result.sequence_state_after_second_step_valid,
                    mailbox_state_after_second_step_valid=payload_result.mailbox_state_after_second_step_valid,
                    request_completion_check_attempted=payload_result.request_completion_check_attempted,
                    request_completion_check_success=payload_result.request_completion_check_success,
                    second_step_rollback_attempted=payload_result.second_step_rollback_attempted,
                    second_step_rollback_success=payload_result.second_step_rollback_success,
                    breadth_only_completed=payload_result.breadth_only_completed,
                    breadth_only_completion_reason=payload_result.breadth_only_completion_reason,
                    next_pipeline_step_skipped_non_owner=payload_result.next_pipeline_step_skipped_non_owner,
                    rollback_attempted=True,
                    rollback_success=combined_rollback_success,
                    total_accepted_tokens=total_accepted,
                    total_rejected_tokens=total_rejected,
                )
        return MailboxVerifyCommitResult(
            attempted=True,
            success=True,
            plan=commit_plan,
            next_required_feature=(
                payload_result.next_required_feature
                if payload_result is not None
                else (kv_result.next_required_feature if kv_result is not None else "kv_append_backend_after_mailbox_verify")
            ),
            sequence_state_commit_attempted=True,
            sequence_state_commit_success=True,
            sequence_state_before=before,
            sequence_state_after={int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list},
            kv_commit_plan_built=kv_commit_plan is not None,
            kv_commit_attempted=bool(kv_result.attempted) if kv_result is not None else False,
            kv_commit_success=bool(kv_result.success) if kv_result is not None else False,
            kv_commit_shadow_only=bool(kv_result.shadow_only) if kv_result is not None else False,
            kv_commit_error=kv_result.error_message if kv_result is not None else "KV append backend is not implemented for mailbox verify commit probe",
            kv_commit_error_kind=kv_result.error_kind if kv_result is not None else "kv_append_backend_after_mailbox_verify",
            kv_commit_rollback_attempted=bool(kv_result.rollback_attempted) if kv_result is not None else False,
            kv_commit_rollback_success=bool(kv_result.rollback_success) if kv_result is not None else True,
            kv_commit_skipped_non_owner=bool(kv_result.skipped_non_owner) if kv_result is not None else False,
            mailbox_payload_consume_plan_built=payload_consume_plan is not None,
            mailbox_payload_consume_attempted=bool(payload_result.attempted) if payload_result is not None else False,
            mailbox_payload_consume_success=bool(payload_result.success) if payload_result is not None else False,
            mailbox_payload_consume_error=payload_result.error_message if payload_result is not None else None,
            mailbox_payload_consume_error_kind=payload_result.error_kind if payload_result is not None else None,
            mailbox_payload_consumed_payload_ids=payload_result.consumed_payload_ids if payload_result is not None else [],
            mailbox_payload_consumed_token_count=payload_result.consumed_token_count if payload_result is not None else 0,
            mailbox_payload_invalidate_attempted=bool(payload_result.attempted) if payload_result is not None else False,
            mailbox_payload_invalidate_success=bool(payload_result.success) if payload_result is not None else False,
            mailbox_payload_invalidate_error=payload_result.error_message if payload_result is not None else None,
            mailbox_payload_invalidated_payload_ids=payload_result.invalidated_payload_ids if payload_result is not None else [],
            mailbox_payload_invalidated_token_count=payload_result.invalidated_token_count if payload_result is not None else 0,
            mailbox_payload_lifecycle_before=payload_result.mailbox_state_before if payload_result is not None else {},
            mailbox_payload_lifecycle_after=payload_result.mailbox_state_after if payload_result is not None else {},
            mailbox_payload_duplicate_consume_detected=bool(payload_result.duplicate_consume_detected) if payload_result is not None else False,
            mailbox_payload_consume_rollback_attempted=bool(payload_result.rollback_attempted) if payload_result is not None else False,
            mailbox_payload_consume_rollback_success=bool(payload_result.rollback_success) if payload_result is not None else True,
            mailbox_payload_consume_skipped_non_owner=bool(payload_result.skipped_non_owner) if payload_result is not None else False,
            mailbox_payload_invalidate_skipped_non_owner=bool(payload_result.invalidate_skipped_non_owner) if payload_result is not None else False,
            next_pipeline_step_attempted=bool(payload_result.next_pipeline_step_attempted) if payload_result is not None else False,
            next_pipeline_step_success=bool(payload_result.next_pipeline_step_success) if payload_result is not None else False,
            next_pipeline_step_error=payload_result.next_pipeline_step_error if payload_result is not None else None,
            next_pipeline_step_error_kind=payload_result.next_pipeline_step_error_kind if payload_result is not None else None,
            next_pipeline_plan_id=payload_result.next_pipeline_plan_id if payload_result is not None else None,
            next_pipeline_target_home_batch_id=payload_result.next_pipeline_target_home_batch_id if payload_result is not None else None,
            next_pipeline_draft_home_batch_id=payload_result.next_pipeline_draft_home_batch_id if payload_result is not None else None,
            next_pipeline_actual_target_seq_ids=payload_result.next_pipeline_actual_target_seq_ids if payload_result is not None else [],
            next_pipeline_actual_draft_seq_ids=payload_result.next_pipeline_actual_draft_seq_ids if payload_result is not None else [],
            previous_committed_plan_id=payload_result.previous_committed_plan_id if payload_result is not None else None,
            previous_consumed_payload_ids=payload_result.previous_consumed_payload_ids if payload_result is not None else [],
            previous_invalidated_payload_ids=payload_result.previous_invalidated_payload_ids if payload_result is not None else [],
            duplicate_payload_consume_after_continue=bool(payload_result.duplicate_payload_consume_after_continue) if payload_result is not None else False,
            pipeline_state_after_commit_valid=bool(payload_result.pipeline_state_after_commit_valid) if payload_result is not None else False,
            scheduler_state_after_commit_valid=bool(payload_result.scheduler_state_after_commit_valid) if payload_result is not None else False,
            breadth_only_step_count=payload_result.breadth_only_step_count if payload_result is not None else 0,
            current_pipeline_step=payload_result.current_pipeline_step if payload_result is not None else 0,
            current_plan_id=payload_result.current_plan_id if payload_result is not None else None,
            next_plan_id=payload_result.next_plan_id if payload_result is not None else None,
            previous_target_home_batch_id=payload_result.previous_target_home_batch_id if payload_result is not None else None,
            previous_draft_home_batch_id=payload_result.previous_draft_home_batch_id if payload_result is not None else None,
            current_target_home_batch_id=payload_result.current_target_home_batch_id if payload_result is not None else None,
            current_draft_home_batch_id=payload_result.current_draft_home_batch_id if payload_result is not None else None,
            active_seq_ids_before_second_step=payload_result.active_seq_ids_before_second_step if payload_result is not None else [],
            active_seq_ids_after_second_step=payload_result.active_seq_ids_after_second_step if payload_result is not None else [],
            committed_seq_ids=payload_result.committed_seq_ids if payload_result is not None else [],
            consumed_payload_ids=payload_result.consumed_payload_ids if payload_result is not None else [],
            invalidated_payload_ids=payload_result.invalidated_payload_ids if payload_result is not None else [],
            available_mailbox_payload_ids=payload_result.available_mailbox_payload_ids if payload_result is not None else [],
            pending_mailbox_payload_ids=payload_result.pending_mailbox_payload_ids if payload_result is not None else [],
            second_step_state_check_attempted=bool(payload_result.second_step_state_check_attempted) if payload_result is not None else False,
            second_step_state_check_success=bool(payload_result.second_step_state_check_success) if payload_result is not None else False,
            second_step_state_error=payload_result.second_step_state_error if payload_result is not None else None,
            second_step_state_error_kind=payload_result.second_step_state_error_kind if payload_result is not None else None,
            repeated_verify_after_commit_detected=bool(payload_result.repeated_verify_after_commit_detected) if payload_result is not None else False,
            scheduler_state_after_second_step_valid=bool(payload_result.scheduler_state_after_second_step_valid) if payload_result is not None else False,
            sequence_state_after_second_step_valid=bool(payload_result.sequence_state_after_second_step_valid) if payload_result is not None else False,
            mailbox_state_after_second_step_valid=bool(payload_result.mailbox_state_after_second_step_valid) if payload_result is not None else False,
            request_completion_check_attempted=bool(payload_result.request_completion_check_attempted) if payload_result is not None else False,
            request_completion_check_success=bool(payload_result.request_completion_check_success) if payload_result is not None else False,
            second_step_rollback_attempted=bool(payload_result.second_step_rollback_attempted) if payload_result is not None else False,
            second_step_rollback_success=bool(payload_result.second_step_rollback_success) if payload_result is not None else True,
            breadth_only_completed=bool(payload_result.breadth_only_completed) if payload_result is not None else False,
            breadth_only_completion_reason=payload_result.breadth_only_completion_reason if payload_result is not None else None,
            next_pipeline_step_skipped_non_owner=bool(payload_result.next_pipeline_step_skipped_non_owner) if payload_result is not None else False,
            rollback_attempted=False,
            rollback_success=True,
            total_accepted_tokens=total_accepted,
            total_rejected_tokens=total_rejected,
        )
    except MailboxVerifyApplyError as exc:
        rollback_attempted = bool(mutated_seq_ids)
        if rollback_attempted:
            rollback_success = _rollback_sequences(seq_by_id, commit_plan.sequence_state_before, mutated_seq_ids)
        after = {int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list}
        next_feature = exc.next_required_feature if rollback_success else "mailbox_commit_rollback_validation"
        return MailboxVerifyCommitResult(
            attempted=True,
            success=False,
            plan=commit_plan,
            error_kind=exc.error_kind,
            error_message=str(exc),
            next_required_feature=next_feature,
            sequence_state_commit_attempted=True,
            sequence_state_commit_success=False,
            sequence_state_before=before,
            sequence_state_after=after,
            kv_commit_attempted=False,
            kv_commit_success=False,
            mailbox_payload_consume_attempted=False,
            mailbox_payload_invalidate_attempted=False,
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
            total_accepted_tokens=total_accepted,
            total_rejected_tokens=total_rejected,
        )
    except Exception as exc:
        rollback_attempted = bool(mutated_seq_ids)
        if rollback_attempted:
            rollback_success = _rollback_sequences(seq_by_id, commit_plan.sequence_state_before, mutated_seq_ids)
        after = {int(seq.seq_id): _sequence_snapshot(seq) for seq in seq_list}
        return MailboxVerifyCommitResult(
            attempted=True,
            success=False,
            plan=commit_plan,
            error_kind=type(exc).__name__,
            error_message=f"mailbox verify commit unexpected error: {exc}",
            next_required_feature="mailbox_commit_rollback_validation",
            sequence_state_commit_attempted=True,
            sequence_state_commit_success=False,
            sequence_state_before=before,
            sequence_state_after=after,
            kv_commit_attempted=False,
            kv_commit_success=False,
            mailbox_payload_consume_attempted=False,
            mailbox_payload_invalidate_attempted=False,
            rollback_attempted=rollback_attempted,
            rollback_success=rollback_success,
            total_accepted_tokens=total_accepted,
            total_rejected_tokens=total_rejected,
        )



def _build_next_pipeline_continuation_metadata(
    consume_plan: MailboxPayloadConsumePlan,
    mailbox_state_after: dict[str, JsonDict],
    *,
    next_pipeline_plan_id: int | None,
    continue_after_commit: bool,
    continuation_context: JsonDict | None,
) -> JsonDict:
    context = dict(continuation_context or {})
    duplicate_after_continue = any(
        str(row.get("lifecycle_state")) == "duplicate_consume_error"
        for row in mailbox_state_after.values()
        if isinstance(row, dict)
    )
    next_plan_id = int(next_pipeline_plan_id) if next_pipeline_plan_id is not None else int(consume_plan.plan_id or 0) + 1
    current_target = consume_plan.target_home_batch_id
    if isinstance(current_target, int):
        next_target = 1 - current_target if current_target in (0, 1) else current_target
        next_draft = current_target
    else:
        next_target = context.get("next_target_home_batch_id", current_target)
        next_draft = context.get("next_draft_home_batch_id", current_target)
    committed_seq_ids = [int(seq_id) for seq_id in consume_plan.seq_ids]
    active_seq_ids = [int(seq_id) for seq_id in context.get("active_seq_ids", consume_plan.seq_ids) or []]
    scheduler_active_seq_ids = [int(seq_id) for seq_id in context.get("scheduler_active_seq_ids", active_seq_ids) or []]
    available_payload_ids = sorted(
        [str(payload_id) for payload_id, row in mailbox_state_after.items() if isinstance(row, dict) and str(row.get("lifecycle_state")) == "available"],
        key=str,
    )
    context_pending_payload_ids = [str(payload_id) for payload_id in context.get("pending_mailbox_payload_ids", [])]
    metadata: JsonDict = {
        "next_pipeline_step_attempted": True,
        "next_pipeline_step_success": False,
        "next_pipeline_step_error": "next pipeline step after mailbox commit is not implemented",
        "next_pipeline_step_error_kind": "next_pipeline_step_after_mailbox_commit",
        "next_pipeline_plan_id": next_plan_id,
        "next_pipeline_target_home_batch_id": next_target,
        "next_pipeline_draft_home_batch_id": next_draft,
        "next_pipeline_actual_target_seq_ids": [],
        "next_pipeline_actual_draft_seq_ids": [],
        "previous_committed_plan_id": consume_plan.plan_id,
        "previous_consumed_payload_ids": list(consume_plan.consumed_payload_ids),
        "previous_invalidated_payload_ids": list(consume_plan.invalidated_payload_ids),
        "consumed_payload_ids": list(consume_plan.consumed_payload_ids),
        "invalidated_payload_ids": list(consume_plan.invalidated_payload_ids),
        "duplicate_payload_consume_after_continue": duplicate_after_continue,
        "pipeline_state_after_commit_valid": not duplicate_after_continue,
        "scheduler_state_after_commit_valid": True,
        "second_step_state_check_attempted": bool(continue_after_commit),
        "second_step_state_check_success": False,
        "second_step_state_error": None,
        "second_step_state_error_kind": None,
        "current_pipeline_step": 1,
        "current_plan_id": consume_plan.plan_id,
        "next_plan_id": next_plan_id,
        "previous_target_home_batch_id": consume_plan.target_home_batch_id,
        "previous_draft_home_batch_id": context.get("previous_draft_home_batch_id", next_draft),
        "current_target_home_batch_id": next_target,
        "current_draft_home_batch_id": next_draft,
        "active_seq_ids_before_second_step": list(active_seq_ids),
        "active_seq_ids_after_second_step": list(active_seq_ids),
        "committed_seq_ids": committed_seq_ids,
        "available_mailbox_payload_ids": available_payload_ids,
        "pending_mailbox_payload_ids": available_payload_ids,
        "repeated_verify_after_commit_detected": False,
        "scheduler_state_after_second_step_valid": True,
        "sequence_state_after_second_step_valid": True,
        "mailbox_state_after_second_step_valid": not duplicate_after_continue,
        "request_completion_check_attempted": False,
        "request_completion_check_success": False,
        "request_completion_reason": None,
        "active_seq_ids_at_completion_check": [],
        "finished_seq_ids_at_completion_check": [],
        "unfinished_seq_ids_at_completion_check": [],
        "max_tokens_reached_seq_ids": [],
        "eos_reached_seq_ids": [],
        "mailbox_pending_payload_ids_at_completion": [],
        "mailbox_consumed_payload_ids_at_completion": list(consume_plan.consumed_payload_ids),
        "scheduler_active_seq_ids_at_completion": list(scheduler_active_seq_ids),
        "sequence_state_completion_valid": True,
        "scheduler_state_completion_valid": True,
        "mailbox_state_completion_valid": not duplicate_after_continue,
        "request_completion_error": None,
        "request_completion_error_kind": None,
        "result_finalization_attempted": False,
        "result_finalization_success": False,
        "result_finalization_error": None,
        "second_step_rollback_attempted": False,
        "second_step_rollback_success": True,
        "breadth_only_step_count": 1,
        "breadth_only_completed": False,
        "breadth_only_completion_reason": None,
        "next_required_feature": "next_pipeline_step_after_mailbox_commit",
    }
    if not continue_after_commit:
        return metadata
    if sorted(active_seq_ids) != sorted(scheduler_active_seq_ids):
        metadata.update(
            {
                "second_step_state_check_success": False,
                "second_step_state_error": "active seq ids mismatch with scheduler state after second-step continuation",
                "second_step_state_error_kind": "scheduler_state_after_second_step",
                "scheduler_state_after_second_step_valid": False,
                "next_pipeline_step_error_kind": "scheduler_state_after_second_step",
                "next_required_feature": "scheduler_state_after_breadth_only_completion",
                "breadth_only_step_count": 2,
            }
        )
        return metadata
    if not active_seq_ids:
        pending_payload_ids = sorted(
            set([payload_id for payload_id in available_payload_ids if payload_id not in set(consume_plan.consumed_payload_ids)] + context_pending_payload_ids),
            key=str,
        )
        metadata["pending_mailbox_payload_ids"] = list(pending_payload_ids)
        metadata["request_completion_check_attempted"] = True
        metadata["request_completion_check_success"] = True
        metadata.update(
            {
                "next_pipeline_step_success": True,
                "next_pipeline_step_error": None,
                "next_pipeline_step_error_kind": None,
                "second_step_state_check_success": True,
                "breadth_only_step_count": 2,
                "current_pipeline_step": 2,
                "active_seq_ids_after_second_step": [],
                "breadth_only_completed": len(pending_payload_ids) == 0,
                "breadth_only_completion_reason": "all_requests_finished" if len(pending_payload_ids) == 0 else "pending_mailbox_payload_after_second_step",
                "request_completion_reason": "all_requests_finished" if len(pending_payload_ids) == 0 else "mailbox_pending_payloads",
                "mailbox_pending_payload_ids_at_completion": list(pending_payload_ids),
                "active_seq_ids_at_completion_check": [],
                "finished_seq_ids_at_completion_check": list(committed_seq_ids),
                "unfinished_seq_ids_at_completion_check": [],
                "result_finalization_attempted": len(pending_payload_ids) == 0,
                "result_finalization_success": len(pending_payload_ids) == 0,
                "next_required_feature": "end_to_end_breadth_only_completion" if len(pending_payload_ids) == 0 else "mailbox_drain_after_breadth_only_completion",
            }
        )
        return metadata
    committed = set(int(seq_id) for seq_id in consume_plan.seq_ids)
    repeated_verify_after_commit = any(seq_id in committed for seq_id in active_seq_ids)
    metadata.update(
        {
            "next_pipeline_step_success": True,
            "next_pipeline_step_error": "second-step state validated; continuation requires request completion drain",
            "next_pipeline_step_error_kind": "active_request_continuation_after_breadth_only_step",
            "next_pipeline_actual_target_seq_ids": [seq_id for seq_id in active_seq_ids if seq_id not in committed],
            "next_pipeline_actual_draft_seq_ids": list(active_seq_ids),
            "second_step_state_check_success": True,
            "current_pipeline_step": 2,
            "active_seq_ids_after_second_step": [seq_id for seq_id in active_seq_ids if seq_id not in committed],
            "repeated_verify_after_commit_detected": repeated_verify_after_commit,
            "request_completion_check_attempted": True,
            "request_completion_check_success": True,
            "request_completion_reason": "active_requests_remaining",
            "active_seq_ids_at_completion_check": list(active_seq_ids),
            "finished_seq_ids_at_completion_check": [],
            "unfinished_seq_ids_at_completion_check": list(active_seq_ids),
            "mailbox_pending_payload_ids_at_completion": list(available_payload_ids),
            "result_finalization_attempted": False,
            "breadth_only_step_count": 2,
            "breadth_only_completed": False,
            "breadth_only_completion_reason": "second_step_metadata_built",
            "next_required_feature": "active_request_continuation_after_breadth_only_step" if any(seq_id for seq_id in active_seq_ids if seq_id not in committed) else "result_finalization_after_breadth_only_completion",
        }
    )
    return metadata

def _payload_id_for_commit_plan(commit_plan: MailboxVerifyCommitPlan, seq_id: int) -> str:
    seq_id = int(seq_id)
    for payload_id in list(commit_plan.mailbox_payloads_to_consume) + list(commit_plan.mailbox_payloads_to_invalidate):
        text = str(payload_id)
        parts = text.split(":")
        if len(parts) >= 2:
            try:
                if int(parts[1]) == seq_id:
                    return text
            except Exception:
                pass
    return f"{commit_plan.target_home_batch_id}:{seq_id}"


def _payload_id_for_consume_plan(consume_plan: MailboxPayloadConsumePlan, seq_id: int) -> str:
    seq_id = int(seq_id)
    for payload_id in list(consume_plan.consumed_payload_ids) + list(consume_plan.invalidated_payload_ids):
        text = str(payload_id)
        parts = text.split(":")
        if len(parts) >= 2:
            try:
                if int(parts[1]) == seq_id:
                    return text
            except Exception:
                pass
    for payload_id, state in consume_plan.mailbox_state_before.items():
        try:
            if int(state.get("seq_id")) == seq_id:
                return str(payload_id)
        except Exception:
            pass
    return f"{consume_plan.target_home_batch_id}:{seq_id}"


def _sequence_output_length(seq: Any) -> int:
    token_ids = list(getattr(seq, "token_ids", []) or [])
    num_tokens = int(getattr(seq, "num_tokens", len(token_ids)) or 0)
    num_prompt_tokens = int(getattr(seq, "num_prompt_tokens", 0) or 0)
    return max(num_tokens - num_prompt_tokens, 0)


def _kv_shadow_length(seq: Any, *, default: int = 0) -> int:
    state = getattr(seq, "stspec_mailbox_kv_shadow", None)
    if isinstance(state, dict) and "length" in state:
        try:
            return int(state.get("length") or 0)
        except Exception:
            return int(default)
    return int(default)


def _kv_shadow_snapshot(seq: Any) -> JsonDict:
    state = getattr(seq, "stspec_mailbox_kv_shadow", None)
    if isinstance(state, dict):
        state = json.loads(json.dumps(state, default=str))
    else:
        state = None
    return {
        "seq_id": int(getattr(seq, "seq_id")),
        "shadow_state": state,
        "shadow_length": _kv_shadow_length(seq, default=int(getattr(seq, "num_tokens", 0) or 0)),
    }


def _set_kv_shadow_state(seq: Any, state: JsonDict | None) -> None:
    if state is None:
        if hasattr(seq, "stspec_mailbox_kv_shadow"):
            delattr(seq, "stspec_mailbox_kv_shadow")
        return
    setattr(seq, "stspec_mailbox_kv_shadow", json.loads(json.dumps(state, default=str)))


def _rollback_kv_shadow(seq_by_id: dict[int, Any], snapshots: dict[int, JsonDict], seq_ids: Iterable[int]) -> bool:
    ok = True
    for seq_id in seq_ids:
        seq = seq_by_id.get(int(seq_id))
        snapshot = snapshots.get(int(seq_id))
        if seq is None or snapshot is None:
            ok = False
            continue
        try:
            _set_kv_shadow_state(seq, snapshot.get("shadow_state"))
            ok = ok and (_kv_shadow_snapshot(seq) == snapshot)
        except Exception:
            ok = False
    return ok


def _snapshots_equal(left: JsonDict | None, right: JsonDict | None, *, ignore_volatile_timestamps: bool = False) -> bool:
    if left is None or right is None:
        return left == right
    if not ignore_volatile_timestamps:
        return left == right
    left_cmp = dict(left)
    right_cmp = dict(right)
    for key in ("first_token_ts", "finish_ts"):
        left_cmp.pop(key, None)
        right_cmp.pop(key, None)
    return left_cmp == right_cmp


def _rollback_sequences(seq_by_id: dict[int, Any], snapshots: dict[int, JsonDict], seq_ids: Iterable[int]) -> bool:
    ok = True
    for seq_id in seq_ids:
        seq = seq_by_id.get(int(seq_id))
        snapshot = snapshots.get(int(seq_id))
        if seq is None or snapshot is None:
            ok = False
            continue
        try:
            _restore_sequence_snapshot(seq, snapshot)
            ok = ok and (_sequence_snapshot(seq) == snapshot)
        except Exception:
            ok = False
    return ok


def _restore_sequence_snapshot(seq: Any, snapshot: JsonDict) -> None:
    for attr in (
        "token_ids",
        "num_tokens",
        "last_token",
        "first_token_ts",
        "finish_ts",
        "trace_stats",
    ):
        if attr in snapshot and hasattr(seq, attr):
            value = snapshot[attr]
            if attr == "token_ids":
                value = list(value or [])
            elif attr == "trace_stats":
                value = json.loads(json.dumps(value, default=str))
            setattr(seq, attr, value)
    if "status_value" in snapshot and hasattr(seq, "status"):
        current_status = getattr(seq, "status")
        enum_type = type(current_status)
        try:
            setattr(seq, "status", enum_type[str(snapshot["status_value"])])
        except Exception:
            pass


def _is_eos_token(token_id: int, eos_token_id: int | list[int]) -> bool:
    if isinstance(eos_token_id, int):
        return int(token_id) == int(eos_token_id)
    return int(token_id) in {int(token) for token in eos_token_id}

def _accepted_prefix_length(drafted: list[int], target: list[int]) -> int:
    accepted = 0
    for draft_token, target_token in zip(drafted, target):
        if int(draft_token) != int(target_token):
            break
        accepted += 1
    return accepted


def _payload_id_for_seq(verify_result: MailboxVerifyResult, seq_id: int) -> str:
    try:
        idx = verify_result.seq_ids.index(int(seq_id))
        source_ids = list(getattr(verify_result, "mailbox_payload_ids", []) or [])
        if idx < len(source_ids):
            return str(source_ids[idx])
    except ValueError:
        pass
    return f"{verify_result.target_home_batch_id}:{seq_id}"


def _sequence_snapshot(seq: Any) -> JsonDict:
    token_ids = list(getattr(seq, "token_ids", []) or [])
    status = getattr(seq, "status", None)
    status_value = getattr(status, "name", status)
    num_prompt_tokens = int(getattr(seq, "num_prompt_tokens", 0) or 0)
    return {
        "seq_id": int(getattr(seq, "seq_id")),
        "request_id": getattr(seq, "request_id", None),
        "token_ids": token_ids,
        "num_tokens": int(getattr(seq, "num_tokens", len(token_ids)) or 0),
        "num_prompt_tokens": num_prompt_tokens,
        "output_token_count": max(len(token_ids) - num_prompt_tokens, 0),
        "last_token": getattr(seq, "last_token", token_ids[-1] if token_ids else None),
        "block_table": list(getattr(seq, "block_table", []) or []),
        "is_finished": bool(getattr(seq, "is_finished", False)),
        "status_value": status_value,
        "first_token_ts": getattr(seq, "first_token_ts", None),
        "finish_ts": getattr(seq, "finish_ts", None),
        "home_batch_id": getattr(seq, "home_batch_id", None),
        "trace_stats": json.loads(json.dumps(getattr(seq, "trace_stats", {}) or {}, default=str)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    return value
