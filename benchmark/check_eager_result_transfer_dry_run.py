#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


TAKEOVER_SOURCE = "phase1h5e3_takeover_lane"
LEGACY_SOURCE = "scheduled_target_eager_lane"
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


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


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


def step_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int_value(record.get("eager_result_transfer_plan_id"), int_value(record.get("plan_id"), -1)),
        int_value(record.get("eager_result_transfer_step_id"), int_value(record.get("step_id"), -1)),
    )


def result_transfer_active(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("eager_result_transfer_dry_run_enabled", False))
        or record.get("eager_result_transfer_dry_run_source") == TAKEOVER_SOURCE
        or bool(as_int_set(record.get("eager_result_transfer_sent_proposal_ids")))
        or bool(as_int_set(record.get("eager_result_sent_proposal_ids")))
        or bool(as_int_set(record.get("eager_result_transfer_received_proposal_ids")))
        or bool(as_int_set(record.get("eager_result_received_proposal_ids")))
        or int_value(record.get("eager_tokens_result_transfer_dry_run"), 0) > 0
        or int_value(record.get("eager_tokens_result_transfer_sent"), 0) > 0
        or int_value(record.get("eager_tokens_result_transfer_received"), 0) > 0
        or bool(record.get("eager_result_transfer_zero_result_step", False))
        or bool(record.get("eager_result_zero_result_step", False))
    )


def validate_result_mapping(
    *,
    idx: int,
    proposal_id: int,
    gamma: int | None,
    verify_result: str,
    action: str,
    accept_len: int,
    proposal_len: int,
    append_tokens: int,
    discarded_tokens: int,
    rollback_ok: bool,
    mutation_detected: bool,
    checkpoint_failed: bool,
) -> list[str]:
    errors: list[str] = []
    if gamma is not None and proposal_len != gamma:
        errors.append(f"record[{idx}] proposal_id={proposal_id} proposal_len != gamma")
    if not (0 <= accept_len <= proposal_len):
        errors.append(f"record[{idx}] proposal_id={proposal_id} accept_len out of range")
    if verify_result == "full_accept":
        if accept_len != proposal_len:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} accept_len mismatch")
        if action != FULL_ACCEPT_ACTION:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} bad action={action!r}")
        if append_tokens != proposal_len:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} append token mismatch")
        if discarded_tokens != 0:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} discarded tokens must be zero")
        if rollback_ok is not True:
            errors.append(f"record[{idx}] full_accept proposal_id={proposal_id} rollback not ok")
    elif verify_result == "partial_accept":
        if not (0 < accept_len < proposal_len):
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} accept_len mismatch")
        if action != PARTIAL_ACTION:
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} bad action={action!r}")
        if append_tokens != 0:
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} must append zero tokens")
        if discarded_tokens != proposal_len:
            errors.append(f"record[{idx}] partial proposal_id={proposal_id} discarded token mismatch")
    elif verify_result == "reject_at_first_token":
        if accept_len != 0:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} accept_len must be zero")
        if action not in {REJECT_ACTION, PARTIAL_ACTION}:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} bad action={action!r}")
        if append_tokens != 0:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} must append zero tokens")
        if discarded_tokens != proposal_len:
            errors.append(f"record[{idx}] reject proposal_id={proposal_id} discarded token mismatch")
    elif verify_result == "skipped_invalid":
        if action != SKIPPED_ACTION:
            errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad action={action!r}")
    else:
        errors.append(f"record[{idx}] proposal_id={proposal_id} unknown verify result={verify_result!r}")
    if mutation_detected:
        errors.append(f"record[{idx}] proposal_id={proposal_id} mutation detected in transferred result")
    if checkpoint_failed:
        errors.append(f"record[{idx}] proposal_id={proposal_id} checkpoint failed in transferred result")
    return errors


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_result_transfer_enabled = 0
    result_transfer_active_records = 0
    zero_result_steps: set[tuple[int, int]] = set()
    sent_by_step: dict[tuple[int, int], set[int]] = defaultdict(set)
    received_by_step: dict[tuple[int, int], set[int]] = defaultdict(set)
    sent_count_by_step: dict[tuple[int, int], int] = {}
    received_count_by_step: dict[tuple[int, int], int] = {}
    validated_count_by_step: dict[tuple[int, int], int] = {}
    sent_ids_seen: set[int] = set()
    received_ids_seen: set[int] = set()
    validated_ids_seen: set[int] = set()
    invalid_ids_seen: set[int] = set()
    duplicate_ids_seen: set[int] = set()
    action_counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    validation_reason_counts: Counter[str] = Counter()
    tokens_sent = 0
    tokens_received = 0
    tokens_validated = 0
    tokens_invalid = 0
    tokens_full_accept = 0
    tokens_discarded = 0
    missing_unexpected_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    draft_mutation_detected_count = 0
    draft_checkpoint_failure_count = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        result_enabled = bool(record.get("enable_eager_result_transfer_dry_run", False))
        apply_enabled = bool(record.get("enable_eager_apply_dry_run", False))
        source = record.get("eager_result_transfer_dry_run_source")
        active = result_transfer_active(record)
        target_eager = as_int_set(record.get("target_eager_set"))
        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        sent_ids = first_int_set(
            record,
            "eager_result_transfer_sent_proposal_ids",
            "eager_result_sent_proposal_ids",
        )
        received_ids = first_int_set(
            record,
            "eager_result_transfer_received_proposal_ids",
            "eager_result_received_proposal_ids",
        )
        validated_ids = first_int_set(
            record,
            "eager_result_transfer_validated_proposal_ids",
            "eager_result_validated_proposal_ids",
        )
        invalid_ids = first_int_set(
            record,
            "eager_result_transfer_invalid_proposal_ids",
            "eager_result_invalid_proposal_ids",
        )
        duplicate_ids = as_int_set(record.get("eager_result_transfer_duplicate_proposal_ids"))
        apply_executed_ids = as_int_set(record.get("eager_apply_dry_run_executed_proposal_ids"))
        action_by_proposal = first_mapping(
            record,
            "eager_result_transfer_action_by_proposal_id",
            "eager_apply_dry_run_action_by_proposal_id",
        )
        verify_result_by_proposal = first_mapping(
            record,
            "eager_result_transfer_verify_result_by_proposal_id",
            "eager_apply_dry_run_verify_result_by_proposal_id",
        )
        accept_len_by_proposal = first_mapping(
            record,
            "eager_result_transfer_accept_len_by_proposal_id",
            "eager_apply_dry_run_accept_len_by_proposal_id",
        )
        proposal_len_by_proposal = first_mapping(
            record,
            "eager_apply_dry_run_proposal_len_by_proposal_id",
            "eager_result_received_proposal_len_by_proposal_id",
            "eager_result_sent_proposal_len_by_proposal_id",
        )
        append_tokens_by_proposal = first_mapping(
            record,
            "eager_result_transfer_append_tokens_by_proposal_id",
            "eager_apply_dry_run_append_tokens_by_proposal_id",
        )
        discarded_tokens_by_proposal = first_mapping(
            record,
            "eager_result_transfer_discarded_tokens_by_proposal_id",
            "eager_apply_dry_run_discarded_tokens_by_proposal_id",
        )
        rollback_ok_by_proposal = first_mapping(
            record,
            "eager_result_transfer_rollback_ok_by_proposal_id",
            "eager_apply_dry_run_rollback_ok_by_proposal_id",
        )
        mutation_by_proposal = first_mapping(
            record,
            "eager_result_transfer_mutation_detected_by_proposal_id",
            "eager_apply_dry_run_mutation_detected_by_proposal_id",
        )
        checkpoint_by_proposal = first_mapping(
            record,
            "eager_result_transfer_checkpoint_failed_by_proposal_id",
            "eager_apply_dry_run_checkpoint_failed_by_proposal_id",
        )
        validation_reasons = first_mapping(
            record,
            "eager_result_transfer_validation_reason_by_proposal_id",
            "eager_result_validation_reason_by_proposal_id",
        )
        sent_count = int_value(record.get("eager_result_transfer_sent_count"), len(sent_ids))
        received_count = int_value(record.get("eager_result_transfer_received_count"), len(received_ids))
        validated_count = int_value(record.get("eager_result_transfer_validated_result_count"), len(validated_ids))
        invalid_count = int_value(record.get("eager_result_transfer_invalid_result_count"), len(invalid_ids))
        sent_token_count = int_value(record.get("eager_tokens_result_transfer_sent"), 0)
        received_token_count = int_value(record.get("eager_tokens_result_transfer_received"), 0)
        validated_token_count = int_value(record.get("eager_tokens_result_transfer_validated"), 0)
        invalid_token_count = int_value(record.get("eager_tokens_result_transfer_invalid"), 0)
        full_accept_token_count = int_value(record.get("eager_tokens_result_transfer_full_accept"), 0)
        discarded_token_count = int_value(record.get("eager_tokens_result_transfer_discarded"), 0)
        dry_run_token_count = int_value(record.get("eager_tokens_result_transfer_dry_run"), 0)
        gamma = int_value(record.get("normal_gamma"), None)

        if target_eager:
            real_target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty, got {sorted(target_eager)}")
        nonzero_actual = [
            field for field in ALWAYS_ZERO_COUNTER_FIELDS if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_verified_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must stay zero: {nonzero_actual}")
        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(f"record[{idx}] missing normal proposals outside eager takeover: {sorted(missing_unexpected)}")

        if not result_enabled:
            if active or sent_token_count or received_token_count or validated_token_count or invalid_token_count:
                errors.append(f"record[{idx}] eager result transfer fields populated while disabled")
            continue

        records_with_result_transfer_enabled += 1
        if not apply_enabled:
            errors.append(f"record[{idx}] result transfer dry-run must imply apply dry-run")
        if not active:
            continue

        result_transfer_active_records += 1
        key = step_key(record)
        runner_role = str(record.get("runner_role", ""))
        is_sender = bool(sent_ids or sent_count or sent_token_count) or (
            "verify" in runner_role and not received_ids and not received_token_count
        )
        is_receiver = bool(received_ids or received_count or received_token_count or validated_ids or invalid_ids) or (
            "draft" in runner_role and not sent_ids and not sent_token_count
        )
        if bool(record.get("eager_result_transfer_zero_result_step", False)) or bool(
            record.get("eager_result_zero_result_step", False)
        ):
            zero_result_steps.add(key)
            sent_by_step.setdefault(key, set())
            received_by_step.setdefault(key, set())

        if source not in (TAKEOVER_SOURCE, LEGACY_SOURCE, None, ""):
            errors.append(f"record[{idx}] unknown result transfer source={source!r}")
        if source == TAKEOVER_SOURCE and sent_ids and sent_ids != apply_executed_ids:
            errors.append(
                f"record[{idx}] sent result ids must equal apply executed ids: "
                f"sent={sorted(sent_ids)}, apply={sorted(apply_executed_ids)}"
            )

        if is_sender:
            sent_by_step[key].update(sent_ids)
            sent_count_by_step[key] = sent_count
            sent_ids_seen.update(sent_ids)
            tokens_sent += sent_token_count
            if dry_run_token_count and dry_run_token_count != sent_token_count:
                errors.append(f"record[{idx}] dry-run token count must match sent tokens")
        if is_receiver:
            received_by_step[key].update(received_ids)
            received_count_by_step[key] = received_count
            validated_count_by_step[key] = validated_count
            received_ids_seen.update(received_ids)
            validated_ids_seen.update(validated_ids)
            invalid_ids_seen.update(invalid_ids)
            duplicate_ids_seen.update(duplicate_ids)
            tokens_received += received_token_count
            tokens_validated += validated_token_count
            tokens_invalid += invalid_token_count

        tokens_full_accept += full_accept_token_count
        tokens_discarded += discarded_token_count

        if received_ids and validated_ids != received_ids:
            errors.append(
                f"record[{idx}] validated result ids must equal received ids: "
                f"validated={sorted(validated_ids)}, received={sorted(received_ids)}"
            )
        if invalid_ids:
            errors.append(f"record[{idx}] invalid result ids must be empty: {sorted(invalid_ids)}")
        if duplicate_ids:
            errors.append(f"record[{idx}] duplicate result ids must be empty: {sorted(duplicate_ids)}")
        if received_count and received_count != len(received_ids):
            errors.append(f"record[{idx}] received count mismatch")
        if validated_count and validated_count != len(validated_ids):
            errors.append(f"record[{idx}] validated count mismatch")
        if invalid_count:
            errors.append(f"record[{idx}] invalid result count must be zero")
        if bool(record.get("eager_result_transfer_draft_mutation_detected", False)):
            draft_mutation_detected_count += 1
            errors.append(f"record[{idx}] draft mutation detected")
        if bool(record.get("eager_result_transfer_draft_checkpoint_failed", False)):
            draft_checkpoint_failure_count += 1
            errors.append(f"record[{idx}] draft checkpoint failed")

        for proposal_id in sorted(received_ids | sent_ids):
            reason = dict_get(validation_reasons, proposal_id)
            if proposal_id in received_ids:
                validation_reason_counts[str(reason)] += 1
                if reason != "ok":
                    errors.append(f"record[{idx}] received proposal_id={proposal_id} validation reason={reason!r}")
            proposal_len = int_value(dict_get(proposal_len_by_proposal, proposal_id), gamma or -1)
            verify_result = str(dict_get(verify_result_by_proposal, proposal_id, ""))
            action = str(dict_get(action_by_proposal, proposal_id, ""))
            accept_len = int_value(dict_get(accept_len_by_proposal, proposal_id), -1)
            append_tokens = int_value(dict_get(append_tokens_by_proposal, proposal_id), 0)
            discarded_tokens = int_value(dict_get(discarded_tokens_by_proposal, proposal_id), 0)
            rollback_ok = bool_value(dict_get(rollback_ok_by_proposal, proposal_id), False)
            mutation_detected = bool_value(dict_get(mutation_by_proposal, proposal_id), False)
            checkpoint_failed = bool_value(dict_get(checkpoint_by_proposal, proposal_id), False)
            action_counts[action] += 1
            result_counts[verify_result] += 1
            errors.extend(
                validate_result_mapping(
                    idx=idx,
                    proposal_id=proposal_id,
                    gamma=gamma,
                    verify_result=verify_result,
                    action=action,
                    accept_len=accept_len,
                    proposal_len=proposal_len,
                    append_tokens=append_tokens,
                    discarded_tokens=discarded_tokens,
                    rollback_ok=rollback_ok,
                    mutation_detected=mutation_detected,
                    checkpoint_failed=checkpoint_failed,
                )
            )

        for seq_id, checkpoint_ok in first_mapping(record, "eager_result_draft_checkpoint_ok_by_seq_id").items():
            if checkpoint_ok is not True:
                draft_checkpoint_failure_count += 1
                errors.append(f"record[{idx}] draft checkpoint failed for seq_id={seq_id}")
        for seq_id, mutation in first_mapping(record, "eager_result_draft_mutation_detected_by_seq_id").items():
            if mutation is True:
                draft_mutation_detected_count += 1
                errors.append(f"record[{idx}] draft mutation detected for seq_id={seq_id}")

    bad_steps = 0
    for key in sorted(set(sent_by_step) | set(received_by_step)):
        sent = sent_by_step.get(key, set())
        received = received_by_step.get(key, set())
        if sent != received:
            bad_steps += 1
            errors.append(f"step{key} sent/received result ids mismatch: sent={sorted(sent)}, received={sorted(received)}")
        sent_count = sent_count_by_step.get(key)
        received_count = received_count_by_step.get(key)
        validated_count = validated_count_by_step.get(key)
        if sent_count is not None and received_count is not None and sent_count != received_count:
            bad_steps += 1
            errors.append(f"step{key} sent/received count mismatch: sent={sent_count}, received={received_count}")
        if received_count is not None and validated_count is not None and received_count != validated_count:
            bad_steps += 1
            errors.append(f"step{key} received/validated count mismatch: received={received_count}, validated={validated_count}")

    summary = {
        "total_trace_records": len(records),
        "records_with_result_transfer_dry_run_enabled": records_with_result_transfer_enabled,
        "result_transfer_active_records": result_transfer_active_records,
        "zero_result_steps": len(zero_result_steps),
        "sent_result_count": len(sent_ids_seen),
        "received_result_count": len(received_ids_seen),
        "validated_result_count": len(validated_ids_seen),
        "invalid_result_count": len(invalid_ids_seen),
        "duplicate_result_count": len(duplicate_ids_seen),
        "bad_sent_received_step_count": bad_steps,
        "eager_tokens_result_transfer_dry_run": tokens_sent or tokens_received,
        "eager_tokens_result_transfer_full_accept": tokens_full_accept,
        "eager_tokens_result_transfer_discarded": tokens_discarded,
        "eager_tokens_result_transfer_sent": tokens_sent,
        "eager_tokens_result_transfer_received": tokens_received,
        "eager_tokens_result_transfer_validated": tokens_validated,
        "eager_tokens_result_transfer_invalid": tokens_invalid,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "draft_mutation_detected_count": draft_mutation_detected_count,
        "draft_checkpoint_failure_count": draft_checkpoint_failure_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "result_counts": dict(result_counts),
        "action_counts": dict(action_counts),
        "validation_reason_counts": dict(validation_reason_counts),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_result_transfer_dry_run_enabled",
        "result_transfer_active_records",
        "zero_result_steps",
        "sent_result_count",
        "received_result_count",
        "validated_result_count",
        "invalid_result_count",
        "duplicate_result_count",
        "bad_sent_received_step_count",
        "eager_tokens_result_transfer_dry_run",
        "eager_tokens_result_transfer_full_accept",
        "eager_tokens_result_transfer_discarded",
        "eager_tokens_result_transfer_sent",
        "eager_tokens_result_transfer_received",
        "eager_tokens_result_transfer_validated",
        "eager_tokens_result_transfer_invalid",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "draft_mutation_detected_count",
        "draft_checkpoint_failure_count",
        "missing_buffered_proposal_unexpected_count",
        "result_counts",
        "action_counts",
        "validation_reason_counts",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "runner_role": "dual_verify",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "enable_eager_transfer_dry_run": False,
        "enable_eager_schedule_dry_run": False,
        "enable_eager_lane_exclusion_dry_run": False,
        "enable_eager_verify_dry_run": False,
        "enable_eager_apply_dry_run": False,
        "enable_eager_result_transfer_dry_run": False,
        "eager_result_transfer_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_eager_set": [],
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "eager_tokens_result_transfer_dry_run": 0,
        "eager_tokens_result_transfer_full_accept": 0,
        "eager_tokens_result_transfer_discarded": 0,
        "eager_tokens_result_transfer_sent": 0,
        "eager_tokens_result_transfer_received": 0,
        "eager_tokens_result_transfer_validated": 0,
        "eager_tokens_result_transfer_invalid": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_transfer_record(
    *,
    receiver: bool,
    proposal_id: int = 101,
    seq_id: int = 3,
    accepted_len: int = 4,
    step_id: int = 7,
) -> dict[str, Any]:
    record = synthetic_base_record()
    gamma = 4
    if accepted_len == gamma:
        verify_result = "full_accept"
        action = FULL_ACCEPT_ACTION
        append_tokens = gamma
        discarded_tokens = 0
    elif accepted_len == 0:
        verify_result = "reject_at_first_token"
        action = REJECT_ACTION
        append_tokens = 0
        discarded_tokens = gamma
    else:
        verify_result = "partial_accept"
        action = PARTIAL_ACTION
        append_tokens = 0
        discarded_tokens = gamma
    common = {
        "enable_eager_plan_dry_run": True,
        "enable_eager_draft_dry_run": True,
        "enable_eager_promotion_dry_run": True,
        "enable_eager_transfer_dry_run": True,
        "enable_eager_schedule_dry_run": True,
        "enable_eager_lane_exclusion_dry_run": True,
        "enable_eager_verify_dry_run": True,
        "enable_eager_apply_dry_run": True,
        "enable_eager_result_transfer_dry_run": True,
        "eager_result_transfer_dry_run_enabled": True,
        "eager_result_transfer_dry_run_source": TAKEOVER_SOURCE,
        "eager_result_transfer_plan_id": 11,
        "eager_result_transfer_step_id": step_id,
        "eager_result_transfer_payload_len": 31,
        "eager_result_transfer_action_by_proposal_id": {str(proposal_id): action},
        "eager_result_transfer_verify_result_by_proposal_id": {str(proposal_id): verify_result},
        "eager_result_transfer_accept_len_by_proposal_id": {str(proposal_id): accepted_len},
        "eager_result_transfer_append_tokens_by_proposal_id": {str(proposal_id): append_tokens},
        "eager_result_transfer_discarded_tokens_by_proposal_id": {str(proposal_id): discarded_tokens},
        "eager_result_transfer_rollback_ok_by_proposal_id": {str(proposal_id): True},
        "eager_result_transfer_mutation_detected_by_proposal_id": {str(proposal_id): False},
        "eager_result_transfer_checkpoint_failed_by_proposal_id": {str(proposal_id): False},
        "eager_apply_dry_run_executed_proposal_ids": [proposal_id],
        "eager_apply_dry_run_proposal_len_by_proposal_id": {str(proposal_id): gamma},
        "eager_tokens_result_transfer_dry_run": gamma,
        "eager_tokens_result_transfer_full_accept": gamma if verify_result == "full_accept" else 0,
        "eager_tokens_result_transfer_discarded": discarded_tokens,
    }
    record.update(common)
    if receiver:
        record.update(
            {
                "runner_role": "dual_draft",
                "eager_result_transfer_received_proposal_ids": [proposal_id],
                "eager_result_transfer_received_seq_ids": [seq_id],
                "eager_result_transfer_received_count": 1,
                "eager_result_transfer_received_result_count": 1,
                "eager_result_transfer_validated_proposal_ids": [proposal_id],
                "eager_result_transfer_validated_result_count": 1,
                "eager_result_transfer_invalid_proposal_ids": [],
                "eager_result_transfer_invalid_result_count": 0,
                "eager_result_transfer_validation_reason_by_proposal_id": {str(proposal_id): "ok"},
                "eager_result_transfer_duplicate_proposal_ids": [],
                "eager_result_received_proposal_ids": [proposal_id],
                "eager_result_received_seq_ids": [seq_id],
                "eager_result_validated_proposal_ids": [proposal_id],
                "eager_result_invalid_proposal_ids": [],
                "eager_result_draft_checkpoint_ok_by_seq_id": {str(seq_id): True},
                "eager_result_draft_mutation_detected_by_seq_id": {str(seq_id): False},
                "eager_tokens_result_transfer_received": gamma,
                "eager_tokens_result_transfer_validated": gamma,
            }
        )
    else:
        record.update(
            {
                "runner_role": "dual_verify",
                "eager_result_transfer_sent_proposal_ids": [proposal_id],
                "eager_result_transfer_sent_seq_ids": [seq_id],
                "eager_result_transfer_sent_count": 1,
                "eager_result_transfer_sent_result_count": 1,
                "eager_result_sent_proposal_ids": [proposal_id],
                "eager_result_sent_seq_ids": [seq_id],
                "eager_tokens_result_transfer_sent": gamma,
            }
        )
    return record


def synthetic_zero_record(*, receiver: bool, step_id: int = 8) -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "runner_role": "dual_draft" if receiver else "dual_verify",
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_lane_exclusion_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "enable_eager_apply_dry_run": True,
            "enable_eager_result_transfer_dry_run": True,
            "eager_result_transfer_dry_run_enabled": True,
            "eager_result_transfer_dry_run_source": TAKEOVER_SOURCE,
            "eager_result_transfer_plan_id": 12,
            "eager_result_transfer_step_id": step_id,
            "eager_result_transfer_zero_result_step": True,
            "eager_result_zero_result_step": True,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_transfer_record(receiver=False, proposal_id=101, accepted_len=4),
        synthetic_transfer_record(receiver=True, proposal_id=101, accepted_len=4),
        synthetic_transfer_record(receiver=False, proposal_id=102, accepted_len=2, step_id=9),
        synthetic_transfer_record(receiver=True, proposal_id=102, accepted_len=2, step_id=9),
        synthetic_transfer_record(receiver=False, proposal_id=103, accepted_len=0, step_id=10),
        synthetic_transfer_record(receiver=True, proposal_id=103, accepted_len=0, step_id=10),
        synthetic_zero_record(receiver=False),
        synthetic_zero_record(receiver=True),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager result transfer records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_transfer_validation_reason_by_proposal_id"] = {"101": "bad_apply_action"}
    errors, _ = validate_records(invalid)
    assert any("validation reason" in error for error in errors), "checker missed invalid validation reason"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_transfer_received_proposal_ids"] = [999]
    invalid[2]["eager_result_received_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("sent/received result ids mismatch" in error for error in errors), "checker missed sent/receive mismatch"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_transfer_duplicate_proposal_ids"] = [101]
    errors, _ = validate_records(invalid)
    assert any("duplicate result ids" in error for error in errors), "checker missed duplicate result"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_result_transfer_sent_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("sent result ids must equal apply executed" in error for error in errors), (
        "checker missed apply-source mismatch"
    )

    invalid = deepcopy(valid_records)
    invalid[1]["eager_result_transfer_action_by_proposal_id"] = {"101": PARTIAL_ACTION}
    errors, _ = validate_records(invalid)
    assert any("bad action" in error for error in errors), "checker missed full-accept action mismatch"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_transfer_draft_mutation_detected"] = True
    errors, _ = validate_records(invalid)
    assert any("draft mutation detected" in error for error in errors), "checker missed draft mutation"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    print("Synthetic eager result transfer dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H eager result transfer dry-run traces.")
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
    print("\nEager result transfer dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
