#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


TAKEOVER_SOURCE = "phase1h5e3_takeover_lane"
FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"
PARTIAL_ACTION = "discard_partial_no_mutation"
REJECT_ACTION = "discard_reject_no_mutation"
SKIPPED_ACTION = "skipped_invalid_no_mutation"
ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]
VALID_SKIP_REASONS = {
    "duplicate_result",
    "not_validated_result_transfer",
    "missing_proposal_id",
    "missing_local_seq",
    "seq_id_mismatch",
    "invalid_proposal_len",
    "invalid_to_verify_len",
    "invalid_base_pre_verify",
    "invalid_result_metadata",
    "target_action_mismatch",
    "target_token_count_mismatch",
    "target_rollback_failed",
    "target_mutation_detected",
    "target_checkpoint_failed",
    "missing_proposal_token_payload",
    "seq_finished_before_sync_apply_dry_run",
    "span_invalidated_before_sync_apply_dry_run",
    "seq_pre_verify_before_sync_apply_dry_run",
    "draft_base_not_reached",
    "unexpected_draft_base_overshot",
    "skipped_invalid",
}


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError(
        "Unsupported trace format. Expected a raw list or a dict with "
        "'traces', 'records', or 'trace_records'."
    )


def is_dual_record(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    )


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def as_int_set(value: Any) -> set[int]:
    return set(as_int_list(value))


def dict_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def first_int_set(record: dict[str, Any], *keys: str) -> set[int]:
    for key in keys:
        values = as_int_set(record.get(key))
        if values:
            return values
    return set()


def first_int_list(record: dict[str, Any], *keys: str) -> list[int]:
    for key in keys:
        values = as_int_list(record.get(key))
        if values:
            return values
    return []


def first_mapping(record: dict[str, Any], *keys: str) -> dict:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def step_key(record: dict[str, Any]) -> tuple[str, int]:
    step_id = record.get("eager_sync_apply_step_id")
    if step_id is not None:
        return ("step", int_value(step_id, -1))
    step_id = record.get("step_id")
    if step_id is not None:
        return ("step", int_value(step_id, -1))
    return ("plan", int_value(record.get("eager_sync_apply_plan_id", record.get("plan_id")), -1))


def proposal_seq_map(record: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    proposal_ids = first_int_list(
        record,
        "eager_sync_apply_dry_run_candidate_proposal_ids",
        "eager_sync_apply_candidate_proposal_ids",
        "eager_result_transfer_validated_proposal_ids",
        "eager_result_validated_proposal_ids",
    )
    seq_ids = (
        as_int_list(record.get("eager_sync_apply_dry_run_candidate_seq_ids"))
        or as_int_list(record.get("eager_sync_apply_candidate_seq_ids"))
        or as_int_list(record.get("eager_result_transfer_received_seq_ids"))
        or as_int_list(record.get("eager_result_received_seq_ids"))
    )
    for proposal_id, seq_id in zip(proposal_ids, seq_ids):
        mapping[int(proposal_id)] = int(seq_id)
    return mapping


def sync_active(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("eager_sync_apply_dry_run_enabled", False))
        or record.get("eager_sync_apply_dry_run_source") == TAKEOVER_SOURCE
        or bool(first_int_set(record, "eager_sync_apply_dry_run_candidate_proposal_ids", "eager_sync_apply_candidate_proposal_ids"))
        or bool(first_int_set(record, "eager_sync_apply_dry_run_executed_proposal_ids", "eager_sync_apply_executed_proposal_ids"))
        or bool(first_int_set(record, "eager_sync_apply_dry_run_skipped_proposal_ids", "eager_sync_apply_skipped_proposal_ids"))
        or int_value(record.get("eager_tokens_sync_apply_dry_run"), 0) > 0
    )


def expected_actions(verify_result: str, accept_len: int) -> set[str]:
    if verify_result == "full_accept":
        return {FULL_ACCEPT_ACTION}
    if verify_result == "partial_accept":
        return {PARTIAL_ACTION}
    if verify_result == "reject_at_first_token":
        actions = {REJECT_ACTION}
        if accept_len == 0:
            actions.add(PARTIAL_ACTION)
        return actions
    if verify_result == "skipped_invalid":
        return {SKIPPED_ACTION}
    return set()


def validate_action_result(
    *,
    idx: int,
    proposal_id: int,
    verify_result: str,
    action: str,
    accept_len: int,
    proposal_len: int,
    append_tokens: int,
    discarded_tokens: int,
) -> list[str]:
    errors: list[str] = []
    if proposal_len <= 0:
        errors.append(f"record[{idx}] proposal_id={proposal_id} invalid proposal_len={proposal_len}")
        return errors
    if not (0 <= accept_len <= proposal_len):
        errors.append(f"record[{idx}] proposal_id={proposal_id} accept_len out of range")
    allowed_actions = expected_actions(verify_result, accept_len)
    if action not in allowed_actions:
        errors.append(f"record[{idx}] proposal_id={proposal_id} bad sync action={action!r}")
    if verify_result == "full_accept":
        if accept_len != proposal_len:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} accept_len mismatch")
        if append_tokens != proposal_len:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} append token mismatch")
        if discarded_tokens != 0:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} discarded tokens must be zero")
    elif verify_result == "partial_accept":
        if not (0 < accept_len < proposal_len):
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} accept_len mismatch")
        if append_tokens != 0:
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} must append zero tokens")
        if discarded_tokens != proposal_len:
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} discarded token mismatch")
    elif verify_result == "reject_at_first_token":
        if accept_len != 0:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} accept_len must be zero")
        if append_tokens != 0:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} must append zero tokens")
        if discarded_tokens != proposal_len:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} discarded token mismatch")
    elif verify_result != "skipped_invalid":
        errors.append(f"record[{idx}] proposal_id={proposal_id} unknown verify_result={verify_result!r}")
    return errors


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_sync_apply_enabled = 0
    sync_apply_active_records = 0
    candidate_ids_seen: set[int] = set()
    executed_ids_seen: set[int] = set()
    skipped_ids_seen: set[int] = set()
    validated_result_ids_seen: set[int] = set()
    consistent_ids_seen: set[int] = set()
    inconsistent_ids_seen: set[int] = set()
    repeated_steps_by_proposal: dict[int, set[tuple[str, int]]] = defaultdict(set)
    action_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    full_accept_count = 0
    discard_count = 0
    appended_token_count = 0
    rollback_ok_count = 0
    rollback_failure_count = 0
    mutation_remaining_count = 0
    checkpoint_bad_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    missing_unexpected_count = 0
    sync_tokens_total = 0
    full_accept_tokens_total = 0
    discarded_tokens_total = 0
    append_tokens_total = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        target_eager = as_int_set(record.get("target_eager_set"))
        if target_eager:
            real_target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_actual = [
            field for field in ALWAYS_ZERO_COUNTER_FIELDS if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_verified_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must stay zero: {nonzero_actual}")

        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(
                f"record[{idx}] unexpected missing buffered proposals: {sorted(missing_unexpected)}"
            )

        sync_enabled = bool(record.get("enable_eager_sync_apply_dry_run", False))
        active = sync_active(record)
        if not sync_enabled:
            if active:
                errors.append(f"record[{idx}] sync apply fields populated while disabled")
            continue

        records_with_sync_apply_enabled += 1
        if not bool(record.get("enable_eager_result_transfer_dry_run", False)):
            errors.append(f"record[{idx}] sync apply dry-run must imply result transfer dry-run")
        if not active and first_int_set(
            record,
            "eager_result_transfer_validated_proposal_ids",
            "eager_result_validated_proposal_ids",
        ):
            active = True
        if not active:
            continue

        sync_apply_active_records += 1
        source = record.get("eager_sync_apply_dry_run_source")
        is_takeover = source == TAKEOVER_SOURCE or bool(
            record.get("eager_sync_apply_dry_run_from_result_transfer_proposal_ids")
        )
        candidate_ids = first_int_set(
            record,
            "eager_sync_apply_dry_run_candidate_proposal_ids",
            "eager_sync_apply_candidate_proposal_ids",
        )
        candidate_seq_ids = (
            as_int_list(record.get("eager_sync_apply_dry_run_candidate_seq_ids"))
            or as_int_list(record.get("eager_sync_apply_candidate_seq_ids"))
        )
        from_result_ids = first_int_set(
            record,
            "eager_sync_apply_dry_run_from_result_transfer_proposal_ids",
            "eager_result_transfer_validated_proposal_ids",
            "eager_result_validated_proposal_ids",
        )
        from_result_seq_ids = (
            as_int_list(record.get("eager_sync_apply_dry_run_from_result_transfer_seq_ids"))
            or as_int_list(record.get("eager_result_transfer_received_seq_ids"))
            or as_int_list(record.get("eager_result_received_seq_ids"))
        )
        executed_ids = first_int_set(
            record,
            "eager_sync_apply_dry_run_executed_proposal_ids",
            "eager_sync_apply_executed_proposal_ids",
        )
        skipped_ids = first_int_set(
            record,
            "eager_sync_apply_dry_run_skipped_proposal_ids",
            "eager_sync_apply_skipped_proposal_ids",
        )
        skip_reasons = first_mapping(
            record,
            "eager_sync_apply_dry_run_skip_reason_by_proposal_id",
            "eager_sync_apply_skip_reason_by_proposal_id",
        )
        proposal_to_seq = proposal_seq_map(record)
        validated_result_ids_seen.update(from_result_ids)
        candidate_ids_seen.update(candidate_ids)
        executed_ids_seen.update(executed_ids)
        skipped_ids_seen.update(skipped_ids)
        consistent_ids = as_int_set(record.get("eager_sync_apply_dry_run_consistent_proposal_ids"))
        inconsistent_ids = as_int_set(record.get("eager_sync_apply_dry_run_inconsistent_proposal_ids"))
        consistent_ids_seen.update(consistent_ids)
        inconsistent_ids_seen.update(inconsistent_ids)

        if is_takeover and source != TAKEOVER_SOURCE:
            errors.append(f"record[{idx}] takeover sync apply row missing source={TAKEOVER_SOURCE!r}")
        if from_result_ids and candidate_ids != from_result_ids:
            errors.append(
                f"record[{idx}] sync candidates must equal validated result-transfer ids: "
                f"candidates={sorted(candidate_ids)}, validated={sorted(from_result_ids)}"
            )
        if from_result_seq_ids and sorted(candidate_seq_ids) != sorted(from_result_seq_ids):
            errors.append(
                f"record[{idx}] sync candidate seq ids must equal validated result-transfer seq ids"
            )
        classified = executed_ids | skipped_ids
        if candidate_ids and classified != candidate_ids:
            errors.append(
                f"record[{idx}] sync candidates must be executed or skipped: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_ids)}"
            )
        if executed_ids & skipped_ids:
            errors.append(f"record[{idx}] proposal ids both executed and skipped: {sorted(executed_ids & skipped_ids)}")
        duplicate_ids = as_int_set(record.get("eager_sync_apply_dry_run_duplicate_proposal_ids"))
        missing_local_ids = as_int_set(record.get("eager_sync_apply_dry_run_missing_local_proposal_ids"))
        if duplicate_ids:
            errors.append(f"record[{idx}] duplicate sync apply results: {sorted(duplicate_ids)}")
        if missing_local_ids:
            errors.append(f"record[{idx}] sync apply missing local proposals/seqs: {sorted(missing_local_ids)}")
        if inconsistent_ids:
            errors.append(f"record[{idx}] inconsistent sync apply proposals: {sorted(inconsistent_ids)}")

        target_action_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_target_action_by_proposal_id")
        draft_action_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_draft_action_by_proposal_id")
        target_result_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_target_verify_result_by_proposal_id")
        draft_result_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_draft_verify_result_by_proposal_id")
        target_accept_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_target_accept_len_by_proposal_id")
        draft_accept_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_draft_accept_len_by_proposal_id")
        action_match_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_action_match_by_proposal_id")
        accept_match_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_accept_len_match_by_proposal_id")
        result_match_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_result_match_by_proposal_id")
        append_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_append_tokens_by_proposal_id")
        discarded_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_discarded_tokens_by_proposal_id")
        rollback_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_rollback_ok_by_proposal_id")
        mutation_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_mutation_detected_by_proposal_id")
        checkpoint_by_proposal = first_mapping(record, "eager_sync_apply_dry_run_checkpoint_failed_by_proposal_id")
        len_before_by_seq = first_mapping(
            record,
            "eager_sync_apply_dry_run_sequence_len_before_by_seq_id",
            "draft_sync_apply_len_before_by_seq_id",
        )
        len_after_by_seq = first_mapping(
            record,
            "eager_sync_apply_dry_run_sequence_len_after_by_seq_id",
            "draft_sync_apply_len_after_restore_by_seq_id",
        )
        pre_before_by_seq = first_mapping(record, "eager_sync_apply_dry_run_pre_verify_before_by_seq_id")
        pre_after_by_seq = first_mapping(record, "eager_sync_apply_dry_run_pre_verify_after_by_seq_id")
        status_before_by_seq = first_mapping(
            record,
            "eager_sync_apply_dry_run_status_before_by_seq_id",
            "draft_sync_apply_status_before_by_seq_id",
        )
        status_after_by_seq = first_mapping(
            record,
            "eager_sync_apply_dry_run_status_after_by_seq_id",
            "draft_sync_apply_status_after_restore_by_seq_id",
        )
        old_action_by_seq = first_mapping(record, "eager_sync_apply_action_by_seq_id")
        old_accept_by_seq = first_mapping(record, "eager_sync_apply_accepted_len_by_seq_id")
        old_full_by_seq = first_mapping(record, "eager_sync_apply_full_accept_by_seq_id")
        gamma = int_value(record.get("normal_gamma"), 0)

        row_expected_tokens = 0
        row_full_tokens = 0
        row_discarded_tokens = 0
        row_append_tokens = 0
        current_step_key = step_key(record)
        for proposal_id in sorted(executed_ids):
            repeated_steps_by_proposal[proposal_id].add(current_step_key)
            seq_id = proposal_to_seq.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing proposal-to-seq mapping")
                continue
            if target_action_by_proposal:
                target_action = str(dict_get(target_action_by_proposal, proposal_id, ""))
                draft_action = str(dict_get(draft_action_by_proposal, proposal_id, ""))
                verify_result = str(dict_get(draft_result_by_proposal, proposal_id, ""))
                target_result = str(dict_get(target_result_by_proposal, proposal_id, ""))
                accept_len = int_value(dict_get(draft_accept_by_proposal, proposal_id), -1)
                target_accept = int_value(dict_get(target_accept_by_proposal, proposal_id), -1)
                append_tokens = int_value(dict_get(append_by_proposal, proposal_id), 0)
                discarded_tokens = int_value(dict_get(discarded_by_proposal, proposal_id), 0)
                proposal_len = append_tokens + discarded_tokens
                if proposal_len <= 0:
                    proposal_len = gamma
                if target_action != draft_action:
                    errors.append(f"record[{idx}] proposal_id={proposal_id} target/draft action mismatch")
                if target_result != verify_result:
                    errors.append(f"record[{idx}] proposal_id={proposal_id} target/draft verify_result mismatch")
                if target_accept != accept_len:
                    errors.append(f"record[{idx}] proposal_id={proposal_id} target/draft accept_len mismatch")
                for mapping_name, mapping in (
                    ("action", action_match_by_proposal),
                    ("accept_len", accept_match_by_proposal),
                    ("result", result_match_by_proposal),
                ):
                    if dict_get(mapping, proposal_id, True) is not True:
                        errors.append(f"record[{idx}] proposal_id={proposal_id} {mapping_name} match flag is false")
                errors.extend(
                    validate_action_result(
                        idx=idx,
                        proposal_id=proposal_id,
                        verify_result=verify_result,
                        action=draft_action,
                        accept_len=accept_len,
                        proposal_len=proposal_len,
                        append_tokens=append_tokens,
                        discarded_tokens=discarded_tokens,
                    )
                )
            else:
                action = str(dict_get(old_action_by_seq, seq_id, ""))
                accept_len = int_value(dict_get(old_accept_by_seq, seq_id), -1)
                full_accept = bool(dict_get(old_full_by_seq, seq_id, False))
                verify_result = "full_accept" if full_accept else ("partial_accept" if accept_len > 0 else "reject_at_first_token")
                proposal_len = gamma
                append_tokens = gamma if full_accept else 0
                discarded_tokens = 0 if full_accept else gamma
                errors.extend(
                    validate_action_result(
                        idx=idx,
                        proposal_id=proposal_id,
                        verify_result=verify_result,
                        action=action,
                        accept_len=accept_len,
                        proposal_len=proposal_len,
                        append_tokens=append_tokens,
                        discarded_tokens=discarded_tokens,
                    )
                )

            action_counts[str(dict_get(draft_action_by_proposal, proposal_id, dict_get(old_action_by_seq, seq_id, "")))] += 1
            row_expected_tokens += int(proposal_len)
            row_full_tokens += int(append_tokens)
            row_discarded_tokens += int(discarded_tokens)
            row_append_tokens += int(append_tokens)
            if append_tokens:
                full_accept_count += 1
            else:
                discard_count += 1
            rollback_ok = dict_get(rollback_by_proposal, proposal_id, None)
            mutation = dict_get(mutation_by_proposal, proposal_id, None)
            checkpoint_failed = dict_get(checkpoint_by_proposal, proposal_id, None)
            if rollback_ok is None:
                rollback_ok = dict_get(record.get("draft_sync_apply_rollback_ok_by_seq_id", {}), seq_id)
            if mutation is None:
                mutation = dict_get(record.get("draft_sync_apply_mutation_remaining_by_seq_id", {}), seq_id)
            if checkpoint_failed is None:
                checkpoint_ok = dict_get(record.get("draft_sync_apply_checkpoint_ok_by_seq_id", {}), seq_id)
                checkpoint_failed = checkpoint_ok is not True
            if rollback_ok is not True:
                rollback_failure_count += 1
                errors.append(f"record[{idx}] rollback failed for proposal_id={proposal_id}")
            else:
                rollback_ok_count += 1
            if mutation is True:
                mutation_remaining_count += 1
                errors.append(f"record[{idx}] mutation remained for proposal_id={proposal_id}")
            if checkpoint_failed is True:
                checkpoint_bad_count += 1
                errors.append(f"record[{idx}] checkpoint failed for proposal_id={proposal_id}")
            len_before = int_value(dict_get(len_before_by_seq, seq_id), -1)
            len_after = int_value(dict_get(len_after_by_seq, seq_id), -1)
            if len_before >= 0 and len_after >= 0 and len_before != len_after:
                errors.append(f"record[{idx}] sequence len not restored for seq_id={seq_id}")
            pre_before = dict_get(pre_before_by_seq, seq_id)
            pre_after = dict_get(pre_after_by_seq, seq_id)
            if pre_before is not None and pre_after is not None and bool(pre_before) != bool(pre_after):
                errors.append(f"record[{idx}] pre_verify not restored for seq_id={seq_id}")
            status_before = dict_get(status_before_by_seq, seq_id)
            status_after = dict_get(status_after_by_seq, seq_id)
            if status_before is not None and status_after is not None and status_before != status_after:
                errors.append(f"record[{idx}] status not restored for seq_id={seq_id}")

        for proposal_id in sorted(skipped_ids):
            reason = str(dict_get(skip_reasons, proposal_id, ""))
            skip_reason_counts[reason] += 1
            if reason not in VALID_SKIP_REASONS:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad reason={reason!r}")

        tokens = int_value(record.get("eager_tokens_sync_apply_dry_run"), 0)
        full_tokens = int_value(record.get("eager_tokens_sync_apply_dry_run_full_accept"), 0)
        discarded_tokens_counter = int_value(record.get("eager_tokens_sync_apply_dry_run_discarded"), 0)
        append_tokens_counter = int_value(record.get("eager_sync_apply_dry_run_append_tokens"), row_append_tokens)
        if executed_ids and tokens != row_expected_tokens:
            errors.append(
                f"record[{idx}] sync token total mismatch: got={tokens}, expected={row_expected_tokens}"
            )
        if executed_ids and full_tokens != row_full_tokens:
            errors.append(
                f"record[{idx}] sync full-accept token mismatch: got={full_tokens}, expected={row_full_tokens}"
            )
        if executed_ids and discarded_tokens_counter != row_discarded_tokens:
            errors.append(
                f"record[{idx}] sync discarded token mismatch: got={discarded_tokens_counter}, expected={row_discarded_tokens}"
            )
        if executed_ids and append_tokens_counter != row_append_tokens:
            errors.append(
                f"record[{idx}] sync append token mismatch: got={append_tokens_counter}, expected={row_append_tokens}"
            )
        sync_tokens_total += tokens
        full_accept_tokens_total += full_tokens
        discarded_tokens_total += discarded_tokens_counter
        append_tokens_total += append_tokens_counter

    repeated_sync_apply_proposal_ids = sorted(
        proposal_id for proposal_id, keys in repeated_steps_by_proposal.items() if len(keys) > 1
    )
    if repeated_sync_apply_proposal_ids:
        errors.append(
            "proposals sync-applied in more than one step/plan: "
            f"{repeated_sync_apply_proposal_ids}"
        )
    missing_sync_ids = sorted(validated_result_ids_seen - executed_ids_seen - skipped_ids_seen)
    if missing_sync_ids:
        errors.append(f"validated result-transfer proposals not sync-applied or skipped: {missing_sync_ids}")

    summary = {
        "total_trace_records": len(records),
        "records_with_sync_apply_dry_run_enabled": records_with_sync_apply_enabled,
        "sync_apply_active_records": sync_apply_active_records,
        "sync_apply_candidate_proposal_count": len(candidate_ids_seen),
        "sync_apply_executed_proposal_count": len(executed_ids_seen),
        "sync_apply_skipped_proposal_count": len(skipped_ids_seen),
        "validated_result_proposal_count": len(validated_result_ids_seen),
        "consistent_proposal_count": len(consistent_ids_seen),
        "inconsistent_proposal_count": len(inconsistent_ids_seen),
        "full_accept_sync_apply_count": full_accept_count,
        "discard_partial_reject_count": discard_count,
        "appended_token_count_in_sync_dry_run": append_tokens_total,
        "rollback_ok_count": rollback_ok_count,
        "rollback_failure_count": rollback_failure_count,
        "mutation_remaining_count": mutation_remaining_count,
        "checkpoint_bad_count": checkpoint_bad_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "repeated_sync_apply_proposal_ids": repeated_sync_apply_proposal_ids,
        "missing_sync_apply_for_validated_proposal_ids": missing_sync_ids,
        "sync_apply_action_counts": dict(action_counts),
        "skip_reason_counts": dict(skip_reason_counts),
        "eager_tokens_sync_apply_dry_run": sync_tokens_total,
        "eager_tokens_sync_apply_dry_run_full_accept": full_accept_tokens_total,
        "eager_tokens_sync_apply_dry_run_discarded": discarded_tokens_total,
        "eager_sync_apply_dry_run_append_tokens": append_tokens_total,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_sync_apply_dry_run_enabled",
        "sync_apply_active_records",
        "sync_apply_candidate_proposal_count",
        "sync_apply_executed_proposal_count",
        "sync_apply_skipped_proposal_count",
        "validated_result_proposal_count",
        "consistent_proposal_count",
        "inconsistent_proposal_count",
        "full_accept_sync_apply_count",
        "discard_partial_reject_count",
        "appended_token_count_in_sync_dry_run",
        "rollback_ok_count",
        "rollback_failure_count",
        "mutation_remaining_count",
        "checkpoint_bad_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "missing_buffered_proposal_unexpected_count",
        "repeated_sync_apply_proposal_ids",
        "missing_sync_apply_for_validated_proposal_ids",
        "sync_apply_action_counts",
        "skip_reason_counts",
        "eager_tokens_sync_apply_dry_run",
        "eager_tokens_sync_apply_dry_run_full_accept",
        "eager_tokens_sync_apply_dry_run_discarded",
        "eager_sync_apply_dry_run_append_tokens",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "runner_role": "dual_draft",
        "plan_id": 10,
        "step_id": 4,
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_result_transfer_dry_run": False,
        "enable_eager_sync_apply_dry_run": False,
        "eager_sync_apply_dry_run_enabled": False,
        "eager_sync_apply_dry_run_source": None,
        "eager_tokens_sync_apply_dry_run": 0,
        "eager_tokens_sync_apply_dry_run_full_accept": 0,
        "eager_tokens_sync_apply_dry_run_discarded": 0,
        "eager_sync_apply_dry_run_append_tokens": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_sync_record(
    proposal_id: int = 101,
    seq_id: int = 7,
    verify_result: str = "full_accept",
    accept_len: int = 4,
    step_id: int = 4,
) -> dict[str, Any]:
    record = synthetic_base_record()
    action = {
        "full_accept": FULL_ACCEPT_ACTION,
        "partial_accept": PARTIAL_ACTION,
        "reject_at_first_token": REJECT_ACTION,
        "skipped_invalid": SKIPPED_ACTION,
    }[verify_result]
    append_tokens = 4 if verify_result == "full_accept" else 0
    discarded_tokens = 0 if verify_result == "full_accept" else 4
    executed_ids = [] if verify_result == "skipped_invalid" else [proposal_id]
    skipped_ids = [proposal_id] if verify_result == "skipped_invalid" else []
    record.update(
        {
            "plan_id": 20 + step_id,
            "step_id": step_id,
            "enable_eager_result_transfer_dry_run": True,
            "enable_eager_sync_apply_dry_run": True,
            "eager_sync_apply_dry_run_enabled": True,
            "eager_sync_apply_dry_run_source": TAKEOVER_SOURCE,
            "eager_result_transfer_validated_proposal_ids": [proposal_id],
            "eager_result_transfer_received_proposal_ids": [proposal_id],
            "eager_result_transfer_received_seq_ids": [seq_id],
            "eager_result_transfer_action_by_proposal_id": {str(proposal_id): action},
            "eager_result_transfer_verify_result_by_proposal_id": {str(proposal_id): verify_result},
            "eager_result_transfer_accept_len_by_proposal_id": {str(proposal_id): accept_len},
            "eager_result_transfer_validation_reason_by_proposal_id": {str(proposal_id): "ok"},
            "eager_sync_apply_step_id": step_id,
            "eager_sync_apply_plan_id": 20 + step_id,
            "eager_sync_apply_dry_run_candidate_proposal_ids": [proposal_id],
            "eager_sync_apply_dry_run_candidate_seq_ids": [seq_id],
            "eager_sync_apply_dry_run_from_result_transfer_proposal_ids": [proposal_id],
            "eager_sync_apply_dry_run_from_result_transfer_seq_ids": [seq_id],
            "eager_sync_apply_dry_run_executed_proposal_ids": executed_ids,
            "eager_sync_apply_dry_run_executed_seq_ids": [] if not executed_ids else [seq_id],
            "eager_sync_apply_dry_run_skipped_proposal_ids": skipped_ids,
            "eager_sync_apply_dry_run_skip_reason_by_proposal_id": (
                {str(proposal_id): "skipped_invalid"} if skipped_ids else {}
            ),
            "eager_sync_apply_dry_run_target_action_by_proposal_id": {str(proposal_id): action},
            "eager_sync_apply_dry_run_draft_action_by_proposal_id": {str(proposal_id): action},
            "eager_sync_apply_dry_run_target_verify_result_by_proposal_id": {str(proposal_id): verify_result},
            "eager_sync_apply_dry_run_draft_verify_result_by_proposal_id": {str(proposal_id): verify_result},
            "eager_sync_apply_dry_run_target_accept_len_by_proposal_id": {str(proposal_id): accept_len},
            "eager_sync_apply_dry_run_draft_accept_len_by_proposal_id": {str(proposal_id): accept_len},
            "eager_sync_apply_dry_run_action_match_by_proposal_id": {str(proposal_id): True},
            "eager_sync_apply_dry_run_accept_len_match_by_proposal_id": {str(proposal_id): True},
            "eager_sync_apply_dry_run_result_match_by_proposal_id": {str(proposal_id): True},
            "eager_sync_apply_dry_run_append_tokens_by_proposal_id": {str(proposal_id): append_tokens},
            "eager_sync_apply_dry_run_discarded_tokens_by_proposal_id": {str(proposal_id): discarded_tokens},
            "eager_sync_apply_dry_run_rollback_ok_by_proposal_id": {str(proposal_id): True},
            "eager_sync_apply_dry_run_mutation_detected_by_proposal_id": {str(proposal_id): False},
            "eager_sync_apply_dry_run_checkpoint_failed_by_proposal_id": {str(proposal_id): False},
            "eager_sync_apply_dry_run_sequence_len_before_by_seq_id": {str(seq_id): 12},
            "eager_sync_apply_dry_run_sequence_len_after_by_seq_id": {str(seq_id): 12},
            "eager_sync_apply_dry_run_pre_verify_before_by_seq_id": {str(seq_id): False},
            "eager_sync_apply_dry_run_pre_verify_after_by_seq_id": {str(seq_id): False},
            "eager_sync_apply_dry_run_status_before_by_seq_id": {str(seq_id): "RUNNING"},
            "eager_sync_apply_dry_run_status_after_by_seq_id": {str(seq_id): "RUNNING"},
            "eager_sync_apply_dry_run_consistent_proposal_ids": [proposal_id],
            "eager_sync_apply_dry_run_inconsistent_proposal_ids": [],
            "eager_sync_apply_dry_run_missing_local_proposal_ids": [],
            "eager_sync_apply_dry_run_duplicate_proposal_ids": [],
            "eager_tokens_sync_apply_dry_run": 0 if skipped_ids else 4,
            "eager_tokens_sync_apply_dry_run_full_accept": append_tokens if executed_ids else 0,
            "eager_tokens_sync_apply_dry_run_discarded": discarded_tokens if executed_ids else 0,
            "eager_sync_apply_dry_run_append_tokens": append_tokens if executed_ids else 0,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_sync_record(101, 7, "full_accept", 4),
        synthetic_sync_record(102, 8, "partial_accept", 2, step_id=5),
        synthetic_sync_record(103, 9, "reject_at_first_token", 0, step_id=6),
        synthetic_sync_record(104, 10, "skipped_invalid", 0, step_id=7),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic sync apply records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_sync_apply_dry_run_draft_action_by_proposal_id"] = {"101": PARTIAL_ACTION}
    invalid[1]["eager_sync_apply_dry_run_action_match_by_proposal_id"] = {"101": False}
    errors, _ = validate_records(invalid)
    assert any("action mismatch" in error or "match flag" in error for error in errors), (
        "checker missed target/draft action mismatch"
    )

    invalid = deepcopy(valid_records)
    invalid[1]["eager_sync_apply_dry_run_mutation_detected_by_proposal_id"] = {"101": True}
    errors, _ = validate_records(invalid)
    assert any("mutation remained" in error for error in errors), "checker missed mutation"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[1]["target_eager_set"] = [7]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_sync_apply_dry_run_executed_proposal_ids"] = []
    invalid[1]["eager_sync_apply_dry_run_executed_seq_ids"] = []
    errors, _ = validate_records(invalid)
    assert any("must be executed or skipped" in error for error in errors), (
        "checker missed missing sync apply"
    )

    invalid = deepcopy(valid_records)
    invalid[1]["eager_sync_apply_step_id"] = 8
    invalid.append(deepcopy(valid_records[1]))
    invalid[-1]["eager_sync_apply_step_id"] = 9
    errors, _ = validate_records(invalid)
    assert any("more than one step" in error for error in errors), (
        "checker missed repeated sync apply"
    )

    invalid = deepcopy(valid_records)
    invalid[1]["missing_buffered_proposal_unexpected_seq_ids"] = [7]
    errors, _ = validate_records(invalid)
    assert any("unexpected missing buffered" in error for error in errors), (
        "checker missed unexpected missing normal proposal"
    )

    print("Synthetic eager sync apply dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H eager sync apply dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Eager sync apply dry-run trace checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
