#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]

VALID_SKIP_REASONS = {
    "base_mismatch_before_apply_dry_run",
    "seq_pre_verify_before_apply_dry_run",
    "invalid_proposal_len",
    "invalid_to_verify_len",
    "missing_verify_result",
    "seq_finished_before_apply_dry_run",
    "span_invalidated_before_apply_dry_run",
    "not_verify_executed",
    "ready_proposal_not_found",
    "routed_seq_mismatch",
    "seq_not_found",
    "skipped_invalid",
}

FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"
PARTIAL_ACTION = "discard_partial_no_mutation"
REJECT_ACTION = "discard_reject_no_mutation"
SKIPPED_ACTION = "skipped_invalid_no_mutation"
TAKEOVER_SOURCE = "phase1h5e3_takeover_lane"


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


def as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(item) for item in value}


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


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


def first_mapping(record: dict[str, Any], *keys: str) -> dict:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def proposal_seq_map(record: dict[str, Any]) -> dict[int, int]:
    for key in (
        "eager_apply_dry_run_seq_id_by_proposal_id",
        "eager_verify_dry_run_seq_id_by_proposal_id",
    ):
        mapping = record.get(key)
        if isinstance(mapping, dict) and mapping:
            return {int(proposal_id): int(seq_id) for proposal_id, seq_id in mapping.items()}

    proposal_ids = (
        as_int_list(record.get("target_eager_verify_proposal_ids_dry_run"))
        or as_int_list(record.get("scheduled_target_eager_proposal_ids_dry_run"))
        or as_int_list(record.get("eager_scheduled_proposal_ids"))
    )
    seq_ids = (
        as_int_list(record.get("target_eager_verify_seq_ids_dry_run"))
        or as_int_list(record.get("scheduled_target_eager_seq_ids_dry_run"))
        or as_int_list(record.get("eager_scheduled_seq_ids"))
    )
    return {proposal_id: seq_id for proposal_id, seq_id in zip(proposal_ids, seq_ids)}


def apply_step_key(record: dict[str, Any]) -> tuple[str, int]:
    step_id = record.get("eager_apply_dry_run_step_id")
    if step_id is not None:
        return ("step", int_value(step_id, -1))
    step_id = record.get("step_id")
    if step_id is not None:
        return ("step", int_value(step_id, -1))
    return ("plan", int_value(record.get("eager_apply_dry_run_plan_id", record.get("plan_id")), -1))


def is_apply_execution_row(
    record: dict[str, Any],
    candidate_ids: set[int],
    executed_ids: set[int],
    skipped_ids: set[int],
    dry_run_tokens: int,
    action_by_proposal: dict,
) -> bool:
    return (
        bool(record.get("eager_apply_dry_run_enabled", False))
        or record.get("eager_apply_dry_run_source") == TAKEOVER_SOURCE
        or bool(candidate_ids)
        or bool(executed_ids)
        or bool(skipped_ids)
        or int(dry_run_tokens) > 0
        or bool(action_by_proposal)
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_apply_enabled = 0
    apply_active_records = 0
    apply_candidate_ids_seen: set[int] = set()
    apply_executed_ids_seen: set[int] = set()
    apply_skipped_ids_seen: set[int] = set()
    verify_executed_ids_seen: set[int] = set()
    full_accept_apply_count = 0
    discard_count = 0
    appended_token_count = 0
    rollback_ok_count = 0
    rollback_failure_count = 0
    mutation_remaining_count = 0
    checkpoint_bad_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    missing_unexpected_count = 0
    action_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    apply_tokens = 0
    repeated_apply_steps_by_proposal: dict[int, set[tuple[str, int]]] = defaultdict(set)

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        apply_enabled = bool(record.get("enable_eager_apply_dry_run", False))
        verify_enabled = bool(record.get("enable_eager_verify_dry_run", False))
        apply_source = record.get("eager_apply_dry_run_source")
        target_eager = as_int_set(record.get("target_eager_set"))
        target_home = as_int_set(record.get("target_home_set"))
        target_normal = as_int_set(record.get("target_normal_verify_seq_ids"))
        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        seq_by_proposal = proposal_seq_map(record)
        verify_executed_ids = first_int_set(
            record,
            "eager_verify_dry_run_executed_proposal_ids",
            "eager_verify_executed_proposal_ids",
        )
        candidate_ids = first_int_set(
            record,
            "eager_apply_dry_run_candidate_proposal_ids",
            "eager_apply_candidate_proposal_ids",
        )
        executed_ids = first_int_set(
            record,
            "eager_apply_dry_run_executed_proposal_ids",
            "eager_apply_executed_proposal_ids",
        )
        executed_seq_ids = first_int_set(
            record,
            "eager_apply_dry_run_executed_seq_ids",
            "eager_apply_executed_seq_ids",
        )
        skipped_ids = first_int_set(
            record,
            "eager_apply_dry_run_skipped_proposal_ids",
            "eager_apply_skipped_proposal_ids",
        )
        skip_reasons = first_mapping(
            record,
            "eager_apply_dry_run_skip_reason_by_proposal_id",
            "eager_apply_skip_reason_by_proposal_id",
        )
        action_by_proposal = first_mapping(record, "eager_apply_dry_run_action_by_proposal_id")
        action_by_seq = first_mapping(record, "eager_apply_action_by_seq_id")
        verify_result_by_proposal = first_mapping(record, "eager_apply_dry_run_verify_result_by_proposal_id")
        accept_len_by_proposal = first_mapping(
            record,
            "eager_apply_dry_run_accept_len_by_proposal_id",
        )
        accepted_len_by_seq = first_mapping(record, "eager_apply_accepted_len_by_seq_id")
        full_accept_by_seq = first_mapping(record, "eager_apply_full_accept_by_seq_id")
        proposal_len_by_id = first_mapping(
            record,
            "eager_apply_dry_run_proposal_len_by_proposal_id",
            "eager_verify_dry_run_proposal_len_by_proposal_id",
            "eager_schedule_proposal_len_by_proposal_id",
        )
        append_tokens_by_proposal = first_mapping(record, "eager_apply_dry_run_append_tokens_by_proposal_id")
        discarded_tokens_by_proposal = first_mapping(record, "eager_apply_dry_run_discarded_tokens_by_proposal_id")
        rollback_ok_by_proposal = first_mapping(record, "eager_apply_dry_run_rollback_ok_by_proposal_id")
        mutation_by_proposal = first_mapping(record, "eager_apply_dry_run_mutation_detected_by_proposal_id")
        checkpoint_failed_by_proposal = first_mapping(record, "eager_apply_dry_run_checkpoint_failed_by_proposal_id")
        len_before_by_seq = first_mapping(
            record,
            "eager_apply_dry_run_sequence_len_before_by_seq_id",
            "eager_apply_current_len_before_by_seq_id",
        )
        len_after_by_seq = first_mapping(
            record,
            "eager_apply_dry_run_sequence_len_after_by_seq_id",
            "eager_apply_current_len_after_rollback_by_seq_id",
        )
        len_after_apply_by_seq = first_mapping(record, "eager_apply_current_len_after_simulated_apply_by_seq_id")
        pre_before_by_seq = first_mapping(
            record,
            "eager_apply_dry_run_pre_verify_before_by_seq_id",
            "eager_apply_pre_verify_before_by_seq_id",
        )
        pre_after_by_seq = first_mapping(
            record,
            "eager_apply_dry_run_pre_verify_after_by_seq_id",
            "eager_apply_pre_verify_after_rollback_by_seq_id",
        )
        status_before_by_seq = first_mapping(
            record,
            "eager_apply_dry_run_status_before_by_seq_id",
            "eager_apply_status_before_by_seq_id",
        )
        status_after_by_seq = first_mapping(
            record,
            "eager_apply_dry_run_status_after_by_seq_id",
            "eager_apply_status_after_rollback_by_seq_id",
        )
        checkpoint_ok_by_seq = first_mapping(record, "eager_apply_checkpoint_ok_by_seq_id")
        rollback_ok_by_seq = first_mapping(record, "eager_apply_rollback_ok_by_seq_id")
        mutation_by_seq = first_mapping(record, "eager_apply_mutation_remaining_by_seq_id")
        appended_by_seq = first_mapping(record, "eager_apply_dry_run_appended_token_count_by_seq_id")
        gamma = int_value(record.get("normal_gamma"), None)
        dry_run_tokens = int_value(record.get("eager_tokens_apply_dry_run"), 0)
        full_accept_tokens = int_value(record.get("eager_tokens_apply_dry_run_full_accept"), 0)
        discarded_tokens = int_value(record.get("eager_tokens_apply_dry_run_discarded"), 0)
        append_tokens = int_value(record.get("eager_apply_dry_run_append_tokens"), 0)
        rollback_failure_row_count = int_value(record.get("eager_apply_dry_run_rollback_failure_count"), 0)
        apply_execution_row = is_apply_execution_row(
            record,
            candidate_ids,
            executed_ids,
            skipped_ids,
            dry_run_tokens,
            action_by_proposal,
        )

        if target_eager:
            real_target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_actual = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_verified_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must stay zero: {nonzero_actual}")

        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(
                f"record[{idx}] missing normal proposals outside eager takeover: "
                f"{sorted(missing_unexpected)}"
            )

        if not apply_enabled:
            if (
                apply_execution_row
                or full_accept_tokens
                or discarded_tokens
                or append_tokens
                or rollback_failure_row_count
            ):
                errors.append(f"record[{idx}] eager apply dry-run fields populated while disabled")
            continue

        records_with_apply_enabled += 1
        if not verify_enabled:
            errors.append(f"record[{idx}] apply dry-run must imply verify dry-run")
        if apply_execution_row:
            apply_active_records += 1
        else:
            continue

        if apply_source == TAKEOVER_SOURCE:
            if not verify_executed_ids:
                errors.append(f"record[{idx}] takeover apply dry-run missing verify-executed ids")
            if candidate_ids != verify_executed_ids:
                errors.append(
                    f"record[{idx}] apply candidates must equal verify executed proposals: "
                    f"candidates={sorted(candidate_ids)}, verify={sorted(verify_executed_ids)}"
                )
        elif apply_source not in (None, ""):
            errors.append(f"record[{idx}] unknown eager apply dry-run source={apply_source!r}")
        elif verify_executed_ids and not candidate_ids:
            errors.append(f"record[{idx}] verify executed proposals exist but apply candidates are empty")

        if not candidate_ids <= verify_executed_ids and verify_executed_ids:
            errors.append(
                f"record[{idx}] apply candidates must come from verify executed proposals: "
                f"extra={sorted(candidate_ids - verify_executed_ids)}"
            )

        verify_executed_ids_seen.update(verify_executed_ids)
        apply_candidate_ids_seen.update(candidate_ids)
        apply_executed_ids_seen.update(executed_ids)
        apply_skipped_ids_seen.update(skipped_ids)
        apply_tokens += dry_run_tokens
        appended_token_count += append_tokens
        rollback_failure_count += rollback_failure_row_count

        classified = executed_ids | skipped_ids
        if (candidate_ids or classified) and classified != candidate_ids:
            errors.append(
                f"record[{idx}] apply candidates must be executed or skipped: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_ids)}"
            )
        if executed_ids & skipped_ids:
            errors.append(f"record[{idx}] proposal ids both executed and skipped: {sorted(executed_ids & skipped_ids)}")
        if dry_run_tokens and not candidate_ids:
            errors.append(f"record[{idx}] eager_tokens_apply_dry_run set without candidates")

        expected_total_tokens = 0
        expected_full_tokens = 0
        expected_discarded_tokens = 0
        expected_append_tokens = 0

        for proposal_id in sorted(skipped_ids):
            reason = dict_get(skip_reasons, proposal_id)
            skip_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing skip reason")
            elif reason not in VALID_SKIP_REASONS:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad reason={reason!r}")

        for proposal_id in sorted(candidate_ids):
            seq_id = seq_by_proposal.get(proposal_id)
            proposal_len = int_value(dict_get(proposal_len_by_id, proposal_id), gamma or -1)
            if proposal_len > 0:
                expected_total_tokens += proposal_len
            if seq_id is None:
                errors.append(f"record[{idx}] proposal_id={proposal_id} missing proposal->seq mapping")
                continue
            if target_home and seq_id not in target_home:
                errors.append(f"record[{idx}] apply seq_id={seq_id} not in target_home_set")
            if seq_id in target_normal:
                errors.append(f"record[{idx}] apply seq_id={seq_id} still in target_normal_verify_seq_ids")

            accept_len = int_value(
                dict_get(accept_len_by_proposal, proposal_id, dict_get(accepted_len_by_seq, seq_id)),
                -1,
            )
            verify_result = str(dict_get(verify_result_by_proposal, proposal_id, ""))
            if not verify_result:
                if gamma is not None and accept_len == gamma:
                    verify_result = "full_accept"
                elif accept_len > 0:
                    verify_result = "partial_accept"
                elif accept_len == 0:
                    verify_result = "reject_at_first_token"
            action = str(dict_get(action_by_proposal, proposal_id, dict_get(action_by_seq, seq_id, "")))
            action_counts[action] += 1
            append_count = int_value(
                dict_get(append_tokens_by_proposal, proposal_id, dict_get(appended_by_seq, seq_id)),
                0,
            )
            discarded_count = int_value(dict_get(discarded_tokens_by_proposal, proposal_id), 0)
            expected_discarded_for_proposal = 0

            if gamma is not None and proposal_len != gamma:
                errors.append(f"record[{idx}] proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and not (0 <= accept_len <= gamma):
                errors.append(f"record[{idx}] accepted_len out of range for proposal_id={proposal_id}: {accept_len}")

            if proposal_id in executed_ids:
                repeated_apply_steps_by_proposal[proposal_id].add(apply_step_key(record))
                if seq_id not in executed_seq_ids:
                    errors.append(f"record[{idx}] executed proposal_id={proposal_id} seq_id={seq_id} missing executed seq")
                if verify_executed_ids and proposal_id not in verify_executed_ids:
                    errors.append(f"record[{idx}] executed proposal_id={proposal_id} was not verify-dry-run executed")

                len_before = int_value(dict_get(len_before_by_seq, seq_id), -1)
                len_after = int_value(dict_get(len_after_by_seq, seq_id), -1)
                len_after_apply = int_value(dict_get(len_after_apply_by_seq, seq_id), len_after)
                if dict_get(pre_before_by_seq, seq_id) != dict_get(pre_after_by_seq, seq_id):
                    errors.append(f"record[{idx}] pre_verify not restored for seq_id={seq_id}")
                if dict_get(status_before_by_seq, seq_id) != dict_get(status_after_by_seq, seq_id):
                    errors.append(f"record[{idx}] status not restored for seq_id={seq_id}")
                if len_before != len_after:
                    errors.append(f"record[{idx}] sequence len not restored for seq_id={seq_id}")
                checkpoint_failed = bool(dict_get(checkpoint_failed_by_proposal, proposal_id, False))
                if checkpoint_failed:
                    checkpoint_bad_count += 1
                    errors.append(f"record[{idx}] checkpoint failed for proposal_id={proposal_id}")
                if checkpoint_ok_by_seq and dict_get(checkpoint_ok_by_seq, seq_id) is not True:
                    checkpoint_bad_count += 1
                    errors.append(f"record[{idx}] checkpoint failed for executed seq_id={seq_id}")
                rollback_ok = dict_get(rollback_ok_by_proposal, proposal_id, dict_get(rollback_ok_by_seq, seq_id))
                if rollback_ok is not True:
                    rollback_failure_count += 1
                    errors.append(f"record[{idx}] rollback failed for proposal_id={proposal_id}")
                else:
                    rollback_ok_count += 1
                mutation_detected = bool(
                    dict_get(mutation_by_proposal, proposal_id, dict_get(mutation_by_seq, seq_id, False))
                )
                if mutation_detected:
                    mutation_remaining_count += 1
                    errors.append(f"record[{idx}] mutation remained after rollback for proposal_id={proposal_id}")

                if verify_result == "full_accept":
                    full_accept_apply_count += 1
                    expected_full_tokens += max(0, proposal_len)
                    expected_append_tokens += max(0, proposal_len)
                    if action != FULL_ACCEPT_ACTION:
                        errors.append(f"record[{idx}] full-accept proposal_id={proposal_id} bad action={action!r}")
                    if gamma is not None and append_count != gamma:
                        errors.append(f"record[{idx}] full-accept proposal_id={proposal_id} append_count != gamma")
                    if gamma is not None and len_after_apply != len_before + gamma:
                        errors.append(f"record[{idx}] full-accept proposal_id={proposal_id} simulated apply length mismatch")
                elif verify_result == "partial_accept":
                    discard_count += 1
                    expected_discarded_for_proposal = max(0, proposal_len)
                    expected_discarded_tokens += expected_discarded_for_proposal
                    if action != PARTIAL_ACTION:
                        errors.append(f"record[{idx}] partial proposal_id={proposal_id} bad action={action!r}")
                    if append_count != 0:
                        errors.append(f"record[{idx}] partial proposal_id={proposal_id} must append zero tokens")
                    if len_after_apply != len_before:
                        errors.append(f"record[{idx}] partial proposal_id={proposal_id} mutated during discard action")
                elif verify_result == "reject_at_first_token":
                    discard_count += 1
                    expected_discarded_for_proposal = max(0, proposal_len)
                    expected_discarded_tokens += expected_discarded_for_proposal
                    if action != REJECT_ACTION:
                        errors.append(f"record[{idx}] rejected proposal_id={proposal_id} bad action={action!r}")
                    if append_count != 0:
                        errors.append(f"record[{idx}] rejected proposal_id={proposal_id} must append zero tokens")
                    if len_after_apply != len_before:
                        errors.append(f"record[{idx}] rejected proposal_id={proposal_id} mutated during discard action")
                elif verify_result == "skipped_invalid":
                    expected_discarded_for_proposal = max(0, proposal_len)
                    expected_discarded_tokens += expected_discarded_for_proposal
                    if action != SKIPPED_ACTION:
                        errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad action={action!r}")
                else:
                    errors.append(f"record[{idx}] proposal_id={proposal_id} unknown verify result={verify_result!r}")
            elif proposal_id in skipped_ids:
                expected_discarded_for_proposal = max(0, proposal_len)
                expected_discarded_tokens += expected_discarded_for_proposal
                if action and action != SKIPPED_ACTION:
                    errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad action={action!r}")

            if discarded_count != expected_discarded_for_proposal:
                errors.append(
                    f"record[{idx}] proposal_id={proposal_id} discarded token count mismatch: "
                    f"{discarded_count} != {expected_discarded_for_proposal}"
                )

        if expected_total_tokens and dry_run_tokens != expected_total_tokens:
            errors.append(
                f"record[{idx}] eager_tokens_apply_dry_run must equal candidate proposal lengths: "
                f"{dry_run_tokens} != {expected_total_tokens}"
            )
        if full_accept_tokens != expected_full_tokens:
            errors.append(
                f"record[{idx}] eager_tokens_apply_dry_run_full_accept mismatch: "
                f"{full_accept_tokens} != {expected_full_tokens}"
            )
        if discarded_tokens != expected_discarded_tokens:
            errors.append(
                f"record[{idx}] eager_tokens_apply_dry_run_discarded mismatch: "
                f"{discarded_tokens} != {expected_discarded_tokens}"
            )
        if append_tokens != expected_append_tokens:
            errors.append(
                f"record[{idx}] eager_apply_dry_run_append_tokens mismatch: "
                f"{append_tokens} != {expected_append_tokens}"
            )

    repeated_apply_proposal_ids = sorted(
        proposal_id
        for proposal_id, step_keys in repeated_apply_steps_by_proposal.items()
        if len(step_keys) > 1
    )
    if repeated_apply_proposal_ids:
        errors.append(
            "proposal ids apply-dry-run executed in multiple unique steps/plans: "
            f"{repeated_apply_proposal_ids}"
        )
    missing_apply_for_verified = sorted(verify_executed_ids_seen - apply_candidate_ids_seen)
    if records_with_apply_enabled and missing_apply_for_verified:
        errors.append(
            "verify-executed proposal ids missing apply candidates: "
            f"{missing_apply_for_verified}"
        )

    summary = {
        "total_trace_records": len(records),
        "records_with_apply_dry_run_enabled": records_with_apply_enabled,
        "apply_active_records": apply_active_records,
        "apply_candidate_proposal_count": len(apply_candidate_ids_seen),
        "apply_executed_proposal_count": len(apply_executed_ids_seen),
        "apply_skipped_proposal_count": len(apply_skipped_ids_seen),
        "full_accept_apply_count": full_accept_apply_count,
        "discard_partial_reject_count": discard_count,
        "appended_token_count_in_dry_run": appended_token_count,
        "rollback_ok_count": rollback_ok_count,
        "rollback_failure_count": rollback_failure_count,
        "mutation_remaining_count": mutation_remaining_count,
        "checkpoint_bad_count": checkpoint_bad_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "repeated_apply_proposal_ids": repeated_apply_proposal_ids,
        "missing_apply_for_verified_proposal_ids": missing_apply_for_verified,
        "apply_action_counts": dict(action_counts),
        "skip_reason_counts": dict(skip_reason_counts),
        "eager_tokens_apply_dry_run": apply_tokens,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_apply_dry_run_enabled",
        "apply_active_records",
        "apply_candidate_proposal_count",
        "apply_executed_proposal_count",
        "apply_skipped_proposal_count",
        "full_accept_apply_count",
        "discard_partial_reject_count",
        "appended_token_count_in_dry_run",
        "rollback_ok_count",
        "rollback_failure_count",
        "mutation_remaining_count",
        "checkpoint_bad_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "missing_buffered_proposal_unexpected_count",
        "repeated_apply_proposal_ids",
        "missing_apply_for_verified_proposal_ids",
        "apply_action_counts",
        "skip_reason_counts",
        "eager_tokens_apply_dry_run",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "enable_eager_transfer_dry_run": False,
        "enable_eager_schedule_dry_run": False,
        "enable_eager_lane_exclusion_dry_run": False,
        "enable_eager_verify_dry_run": False,
        "enable_eager_apply_dry_run": False,
        "eager_apply_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_home_set": [],
        "target_normal_verify_seq_ids": [],
        "target_eager_set": [],
        "target_eager_verify_seq_ids_dry_run": [],
        "target_eager_verify_proposal_ids_dry_run": [],
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "eager_apply_dry_run_candidate_proposal_ids": [],
        "eager_apply_dry_run_candidate_seq_ids": [],
        "eager_apply_dry_run_executed_proposal_ids": [],
        "eager_apply_dry_run_executed_seq_ids": [],
        "eager_apply_dry_run_skipped_proposal_ids": [],
        "eager_apply_dry_run_skip_reason_by_proposal_id": {},
        "eager_tokens_apply_dry_run": 0,
        "eager_tokens_apply_dry_run_full_accept": 0,
        "eager_tokens_apply_dry_run_discarded": 0,
        "eager_apply_dry_run_append_tokens": 0,
        "eager_apply_dry_run_rollback_failure_count": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_apply_record(accepted_len: int = 4, *, proposal_id: int = 101, step_id: int = 7) -> dict[str, Any]:
    record = synthetic_base_record()
    gamma = 4
    full_accept = accepted_len == gamma
    result = "full_accept" if full_accept else "partial_accept" if accepted_len > 0 else "reject_at_first_token"
    action = FULL_ACCEPT_ACTION if full_accept else (PARTIAL_ACTION if accepted_len > 0 else REJECT_ACTION)
    appended = gamma if full_accept else 0
    discarded = 0 if full_accept else gamma
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_lane_exclusion_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "enable_eager_apply_dry_run": True,
            "eager_apply_dry_run_enabled": True,
            "eager_apply_dry_run_source": TAKEOVER_SOURCE,
            "eager_apply_dry_run_step_id": step_id,
            "eager_apply_dry_run_plan_id": 11,
            "target_home_set": [3, 4],
            "target_normal_verify_seq_ids": [4],
            "target_eager_verify_seq_ids_dry_run": [3],
            "target_eager_verify_proposal_ids_dry_run": [proposal_id],
            "eager_verify_dry_run_executed_proposal_ids": [proposal_id],
            "eager_verify_dry_run_executed_seq_ids": [3],
            "eager_verify_dry_run_seq_id_by_proposal_id": {str(proposal_id): 3},
            "eager_apply_dry_run_candidate_proposal_ids": [proposal_id],
            "eager_apply_dry_run_candidate_seq_ids": [3],
            "eager_apply_dry_run_executed_proposal_ids": [proposal_id],
            "eager_apply_dry_run_executed_seq_ids": [3],
            "eager_apply_dry_run_from_verify_proposal_ids": [proposal_id],
            "eager_apply_dry_run_verify_result_by_proposal_id": {str(proposal_id): result},
            "eager_apply_dry_run_accept_len_by_proposal_id": {str(proposal_id): accepted_len},
            "eager_apply_dry_run_seq_id_by_proposal_id": {str(proposal_id): 3},
            "eager_apply_dry_run_proposal_len_by_proposal_id": {str(proposal_id): gamma},
            "eager_apply_dry_run_action_by_proposal_id": {str(proposal_id): action},
            "eager_apply_dry_run_full_accept_proposal_ids": [proposal_id] if full_accept else [],
            "eager_apply_dry_run_discarded_proposal_ids": [] if full_accept else [proposal_id],
            "eager_apply_dry_run_append_tokens_by_proposal_id": {str(proposal_id): appended},
            "eager_apply_dry_run_discarded_tokens_by_proposal_id": {str(proposal_id): discarded},
            "eager_apply_dry_run_rollback_ok_by_proposal_id": {str(proposal_id): True},
            "eager_apply_dry_run_mutation_detected_by_proposal_id": {str(proposal_id): False},
            "eager_apply_dry_run_checkpoint_failed_by_proposal_id": {str(proposal_id): False},
            "eager_apply_dry_run_sequence_len_before_by_seq_id": {"3": 12},
            "eager_apply_dry_run_sequence_len_after_by_seq_id": {"3": 12},
            "eager_apply_dry_run_pre_verify_before_by_seq_id": {"3": False},
            "eager_apply_dry_run_pre_verify_after_by_seq_id": {"3": False},
            "eager_apply_dry_run_status_before_by_seq_id": {"3": "RUNNING"},
            "eager_apply_dry_run_status_after_by_seq_id": {"3": "RUNNING"},
            "eager_apply_current_len_after_simulated_apply_by_seq_id": {"3": 12 + appended},
            "eager_tokens_apply_dry_run": gamma,
            "eager_tokens_apply_dry_run_full_accept": gamma if full_accept else 0,
            "eager_tokens_apply_dry_run_discarded": discarded,
            "eager_apply_dry_run_append_tokens": appended,
            "eager_apply_dry_run_rollback_failure_count": 0,
        }
    )
    return record


def synthetic_apply_skip_record(reason: str) -> dict[str, Any]:
    record = synthetic_apply_record(4)
    record.update(
        {
            "eager_apply_dry_run_executed_proposal_ids": [],
            "eager_apply_dry_run_executed_seq_ids": [],
            "eager_apply_dry_run_skipped_proposal_ids": [101],
            "eager_apply_dry_run_skip_reason_by_proposal_id": {"101": reason},
            "eager_apply_dry_run_verify_result_by_proposal_id": {"101": "skipped_invalid"},
            "eager_apply_dry_run_action_by_proposal_id": {"101": SKIPPED_ACTION},
            "eager_apply_dry_run_full_accept_proposal_ids": [],
            "eager_apply_dry_run_discarded_proposal_ids": [101],
            "eager_apply_dry_run_append_tokens_by_proposal_id": {"101": 0},
            "eager_apply_dry_run_discarded_tokens_by_proposal_id": {"101": 4},
            "eager_tokens_apply_dry_run_full_accept": 0,
            "eager_tokens_apply_dry_run_discarded": 4,
            "eager_apply_dry_run_append_tokens": 0,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_apply_record(4),
        synthetic_apply_record(2, proposal_id=102),
        synthetic_apply_record(0, proposal_id=103),
        synthetic_apply_skip_record("base_mismatch_before_apply_dry_run"),
        synthetic_apply_skip_record("seq_pre_verify_before_apply_dry_run"),
        synthetic_apply_skip_record("missing_verify_result"),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager apply records failed: {errors}"

    repeated_same_step = [synthetic_apply_record(4), synthetic_apply_record(4)]
    errors, _ = validate_records(repeated_same_step)
    assert not errors, f"same-step repeated apply rows should pass: {errors}"

    invalid = [synthetic_apply_record(4), synthetic_apply_record(4, step_id=8)]
    errors, _ = validate_records(invalid)
    assert any("multiple unique steps" in error for error in errors), "checker missed repeated apply"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_apply_dry_run_sequence_len_after_by_seq_id"] = {"3": 16}
    errors, _ = validate_records(invalid)
    assert any("sequence len not restored" in error for error in errors), "checker missed rollback len failure"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_apply_dry_run_mutation_detected_by_proposal_id"] = {"101": True}
    errors, _ = validate_records(invalid)
    assert any("mutation remained" in error for error in errors), "checker missed remaining mutation"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_apply_dry_run_rollback_ok_by_proposal_id"] = {"101": False}
    errors, _ = validate_records(invalid)
    assert any("rollback failed" in error for error in errors), "checker missed rollback failure"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_apply_dry_run_action_by_proposal_id"] = {"101": PARTIAL_ACTION}
    errors, _ = validate_records(invalid)
    assert any("bad action" in error for error in errors), "checker missed full-accept action mismatch"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_apply_dry_run_append_tokens_by_proposal_id"] = {"102": 1}
    errors, _ = validate_records(invalid)
    assert any("must append zero" in error for error in errors), "checker missed partial append"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[1]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_apply_dry_run_accept_len_by_proposal_id"] = {"101": 5}
    errors, _ = validate_records(invalid)
    assert any("accepted_len out of range" in error for error in errors), "checker missed accepted_len range"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_apply_dry_run_candidate_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("candidates must equal verify executed" in error for error in errors), (
        "checker missed candidate mismatch"
    )

    print("Synthetic eager apply dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H eager apply dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        if args.trace is None:
            return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nEager apply dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
