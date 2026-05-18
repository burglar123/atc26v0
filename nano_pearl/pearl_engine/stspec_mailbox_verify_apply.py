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
    return {
        "seq_id": int(getattr(seq, "seq_id")),
        "request_id": getattr(seq, "request_id", None),
        "token_ids": token_ids,
        "num_tokens": int(getattr(seq, "num_tokens", len(token_ids)) or 0),
        "last_token": getattr(seq, "last_token", token_ids[-1] if token_ids else None),
        "block_table": list(getattr(seq, "block_table", []) or []),
        "is_finished": bool(getattr(seq, "is_finished", False)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    return value
