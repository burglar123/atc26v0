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
    "not_scheduled_proposal",
    "not_scheduled_seq",
    "intersects_target_home",
    "intersects_adjusted_draft_home",
    "invalid_lane",
    "invalid_state",
    "invalid_base_pre_verify",
    "invalid_proposal_len",
    "invalid_to_verify_len",
    "invalid_proposal_token_len",
    "invalid_to_verify_token_len",
    "ready_proposal_not_found",
    "routed_seq_mismatch",
    "seq_not_in_target_home",
    "seq_still_in_target_normal_verify",
    "invalid_ready_proposal_state",
    "takeover_not_routed",
    "takeover_routed_in_different_step",
    "verify_dry_run_already_executed",
    "seq_not_found",
    "seq_finished_before_verify",
    "seq_span_invalidated_before_verify",
    "seq_not_running",
    "seq_returned_pre_verify_before_verify",
    "base_mismatch_before_verify",
    "base_not_reached_before_verify",
    "base_overshot_before_verify",
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


def proposal_seq_map(record: dict[str, Any]) -> dict[int, int]:
    explicit = record.get("eager_verify_dry_run_seq_id_by_proposal_id", {})
    if isinstance(explicit, dict) and explicit:
        return {int(proposal_id): int(seq_id) for proposal_id, seq_id in explicit.items()}

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


def verify_step_key(record: dict[str, Any]) -> tuple[str, int]:
    step_id = record.get("eager_verify_dry_run_step_id")
    if step_id is not None:
        return ("step", int_value(step_id, -1))
    step_id = record.get("step_id")
    if step_id is not None:
        return ("step", int_value(step_id, -1))
    return ("plan", int_value(record.get("eager_verify_dry_run_plan_id", record.get("plan_id")), -1))


def is_takeover_routing_row(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("target_eager_verify_proposal_ids_dry_run"))
        or bool(record.get("target_eager_verify_seq_ids_dry_run"))
    )


def is_phase1h5f_source(record: dict[str, Any]) -> bool:
    return record.get("eager_verify_dry_run_source") == "phase1h5e3_takeover_lane"


def is_verify_execution_row(
    record: dict[str, Any],
    candidate_proposal_ids: set[int],
    candidate_seq_ids: set[int],
    executed_proposal_ids: set[int],
    executed_seq_ids: set[int],
    skipped_proposal_ids: set[int],
    dry_run_tokens: int,
    result_by_proposal: dict,
) -> bool:
    return (
        bool(record.get("eager_verify_dry_run_enabled", False))
        or is_phase1h5f_source(record)
        or bool(candidate_proposal_ids)
        or bool(candidate_seq_ids)
        or bool(executed_proposal_ids)
        or bool(executed_seq_ids)
        or bool(skipped_proposal_ids)
        or int(dry_run_tokens) > 0
        or bool(result_by_proposal)
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    verify_records = 0
    verify_active_records = 0
    scheduled_proposal_ids_seen: set[int] = set()
    executed_proposal_ids_seen: set[int] = set()
    executed_seq_ids_seen: set[int] = set()
    skipped_proposal_ids_seen: set[int] = set()
    skip_reason_counts: Counter[str] = Counter()
    accepted_len_distribution: Counter[int] = Counter()
    full_accept_count = 0
    reject_first_count = 0
    partial_accept_count = 0
    checkpoint_failure_count = 0
    mutation_detected_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    missing_unexpected_count = 0
    verify_tokens = 0
    target_eager_verify_seq_ids_dry_run_count = 0
    repeated_verify_steps_by_proposal: dict[int, set[tuple[str, int]]] = defaultdict(set)
    takeover_proposal_ids_seen: set[int] = set()
    takeover_seq_ids_by_proposal_id: dict[int, int] = {}
    verify_executed_or_skipped_proposal_ids_seen: set[int] = set()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        verify_enabled = bool(record.get("enable_eager_verify_dry_run", False))
        schedule_enabled = bool(record.get("enable_eager_schedule_dry_run", False))
        lane_enabled = bool(record.get("enable_eager_lane_exclusion_dry_run", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        target_home = as_int_set(record.get("target_home_set"))
        target_normal = as_int_set(record.get("target_normal_verify_seq_ids"))
        takeover_seq_ids = as_int_set(record.get("target_eager_verify_seq_ids_dry_run"))
        takeover_proposal_ids = as_int_set(record.get("target_eager_verify_proposal_ids_dry_run"))
        missing_allowed = as_int_set(record.get("missing_buffered_proposal_allowed_by_eager_seq_ids"))
        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        target_eager_verify_seq_ids_dry_run_count += len(takeover_seq_ids)

        candidate_proposal_ids = first_int_set(
            record,
            "eager_verify_dry_run_candidate_proposal_ids",
            "eager_verify_candidate_proposal_ids",
        )
        candidate_seq_ids = first_int_set(
            record,
            "eager_verify_dry_run_candidate_seq_ids",
            "eager_verify_candidate_seq_ids",
        )
        executed_proposal_ids = first_int_set(
            record,
            "eager_verify_dry_run_executed_proposal_ids",
            "eager_verify_executed_proposal_ids",
        )
        executed_seq_ids = first_int_set(
            record,
            "eager_verify_dry_run_executed_seq_ids",
            "eager_verify_executed_seq_ids",
        )
        skipped_proposal_ids = first_int_set(
            record,
            "eager_verify_dry_run_skipped_proposal_ids",
            "eager_verify_skipped_proposal_ids",
        )
        skip_reasons = first_mapping(
            record,
            "eager_verify_dry_run_skip_reason_by_proposal_id",
            "eager_verify_skip_reason_by_proposal_id",
        )
        seq_by_proposal = proposal_seq_map(record)
        accepted_len_by_proposal = first_mapping(
            record,
            "eager_verify_dry_run_accept_len_by_proposal_id",
        )
        result_by_proposal = first_mapping(record, "eager_verify_dry_run_result_by_proposal_id")
        reject_position_by_proposal = first_mapping(
            record,
            "eager_verify_dry_run_reject_position_by_proposal_id",
        )
        accepted_len_by_seq = first_mapping(record, "eager_verify_accepted_len_by_seq_id")
        full_accept_by_seq = first_mapping(record, "eager_verify_full_accept_by_seq_id")
        invalidated_len_by_seq = first_mapping(record, "eager_verify_invalidated_len_by_seq_id")
        reject_position_by_seq = first_mapping(record, "eager_verify_reject_position_by_seq_id")
        base_match_by_seq = first_mapping(record, "eager_verify_base_match_by_seq_id")
        seq_pre_verify_by_seq = first_mapping(record, "eager_verify_seq_pre_verify_by_seq_id")
        checkpoint_ok_by_seq = first_mapping(record, "eager_verify_checkpoint_ok_by_seq_id")
        mutation_by_seq = first_mapping(record, "eager_verify_mutation_detected_by_seq_id")
        proposal_len_by_id = first_mapping(
            record,
            "eager_verify_dry_run_proposal_len_by_proposal_id",
            "eager_schedule_proposal_len_by_proposal_id",
        )
        to_verify_len_by_id = first_mapping(
            record,
            "eager_verify_dry_run_to_verify_len_by_proposal_id",
            "eager_schedule_to_verify_len_by_proposal_id",
        )
        len_before_by_seq = first_mapping(
            record,
            "eager_verify_dry_run_sequence_len_before_by_seq_id",
        )
        len_after_by_seq = first_mapping(
            record,
            "eager_verify_dry_run_sequence_len_after_by_seq_id",
        )
        gamma = int_value(record.get("normal_gamma"), None)
        dry_run_tokens = int_value(record.get("eager_tokens_verify_dry_run"), 0)
        full_accept_tokens = int_value(record.get("eager_tokens_verify_dry_run_full_accept"), 0)
        rejected_tokens = int_value(record.get("eager_tokens_verify_dry_run_rejected"), 0)
        partial_tokens = int_value(record.get("eager_tokens_verify_dry_run_partial_accept"), 0)
        takeover_routing_row = is_takeover_routing_row(record)
        verify_execution_row = is_verify_execution_row(
            record,
            candidate_proposal_ids,
            candidate_seq_ids,
            executed_proposal_ids,
            executed_seq_ids,
            skipped_proposal_ids,
            dry_run_tokens,
            result_by_proposal,
        )
        phase1h5f_source = is_phase1h5f_source(record)

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
            errors.append(f"record[{idx}] actual eager verification counters must stay zero: {nonzero_actual}")

        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(
                f"record[{idx}] missing normal proposals outside eager takeover: "
                f"{sorted(missing_unexpected)}"
            )

        if takeover_routing_row:
            takeover_proposal_ids_seen.update(takeover_proposal_ids)
            for proposal_id, seq_id in seq_by_proposal.items():
                if proposal_id in takeover_proposal_ids:
                    existing_seq_id = takeover_seq_ids_by_proposal_id.setdefault(proposal_id, seq_id)
                    if existing_seq_id != seq_id:
                        errors.append(
                            f"record[{idx}] takeover proposal_id={proposal_id} maps to multiple seq ids: "
                            f"{existing_seq_id} and {seq_id}"
                        )
            if not takeover_seq_ids.issubset(target_home):
                errors.append(
                    f"record[{idx}] target eager verify seqs must be in target_home_set: "
                    f"extra={sorted(takeover_seq_ids - target_home)}"
                )
            if takeover_seq_ids & target_normal:
                errors.append(
                    f"record[{idx}] eager takeover seqs must be absent from target_normal_verify_seq_ids: "
                    f"{sorted(takeover_seq_ids & target_normal)}"
                )
            if missing_allowed and not missing_allowed.issubset(takeover_seq_ids):
                errors.append(
                    f"record[{idx}] missing proposals allowed by eager must be takeover seqs: "
                    f"extra={sorted(missing_allowed - takeover_seq_ids)}"
                )

        if not verify_enabled:
            if verify_execution_row:
                errors.append(f"record[{idx}] eager verify dry-run fields populated while disabled")
            continue

        verify_records += 1
        if verify_execution_row:
            verify_active_records += 1
        if not schedule_enabled:
            errors.append(f"record[{idx}] verify dry-run must imply schedule dry-run")
        if (takeover_routing_row or phase1h5f_source) and not lane_enabled:
            errors.append(f"record[{idx}] takeover verify dry-run must imply lane exclusion dry-run")

        scheduled_proposal_ids_seen.update(seq_by_proposal)

        if not verify_execution_row:
            continue

        executed_proposal_ids_seen.update(executed_proposal_ids)
        executed_seq_ids_seen.update(executed_seq_ids)
        skipped_proposal_ids_seen.update(skipped_proposal_ids)
        verify_executed_or_skipped_proposal_ids_seen.update(executed_proposal_ids | skipped_proposal_ids)
        verify_tokens += dry_run_tokens

        if takeover_routing_row:
            if candidate_proposal_ids != takeover_proposal_ids:
                errors.append(
                    f"record[{idx}] takeover verify candidates must equal target eager proposals: "
                    f"candidates={sorted(candidate_proposal_ids)}, takeover={sorted(takeover_proposal_ids)}"
                )
            if candidate_seq_ids != takeover_seq_ids:
                errors.append(
                    f"record[{idx}] takeover verify candidate seqs must equal target eager seqs: "
                    f"candidates={sorted(candidate_seq_ids)}, takeover={sorted(takeover_seq_ids)}"
                )
        elif phase1h5f_source and not candidate_proposal_ids:
            errors.append(f"record[{idx}] active takeover verify row missing candidate proposal ids")

        if phase1h5f_source or takeover_routing_row:
            if target_home and not executed_seq_ids.issubset(target_home):
                errors.append(
                    f"record[{idx}] executed takeover seqs must be in target_home_set: "
                    f"extra={sorted(executed_seq_ids - target_home)}"
                )
            if executed_seq_ids & target_normal:
                errors.append(
                    f"record[{idx}] executed takeover seqs must be absent from target_normal_verify_seq_ids: "
                    f"{sorted(executed_seq_ids & target_normal)}"
                )

        classified = executed_proposal_ids | skipped_proposal_ids
        if (candidate_proposal_ids or classified) and classified != candidate_proposal_ids:
            errors.append(
                f"record[{idx}] verify candidates must be executed or skipped: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_proposal_ids)}"
            )
        if executed_proposal_ids & skipped_proposal_ids:
            errors.append(
                f"record[{idx}] proposal ids both executed and skipped: "
                f"{sorted(executed_proposal_ids & skipped_proposal_ids)}"
            )
        if dry_run_tokens and not executed_proposal_ids:
            errors.append(f"record[{idx}] eager_tokens_verify_dry_run set without executed proposals")
        if executed_proposal_ids and dry_run_tokens <= 0:
            errors.append(f"record[{idx}] executed verify proposals require positive dry-run token count")
        if full_accept_tokens + rejected_tokens + partial_tokens != dry_run_tokens:
            errors.append(
                f"record[{idx}] full_accept + rejected + partial token counters must equal total: "
                f"{full_accept_tokens} + {rejected_tokens} + {partial_tokens} != {dry_run_tokens}"
            )

        expected_dry_run_tokens = 0
        for proposal_id in sorted(executed_proposal_ids):
            step_key = verify_step_key(record)
            repeated_verify_steps_by_proposal[proposal_id].add(step_key)
            seq_id = seq_by_proposal.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing proposal->seq mapping")
                continue
            if proposal_id not in candidate_proposal_ids:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing from verify candidates")
            if seq_id not in executed_seq_ids:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} seq_id={seq_id} missing executed seq")
            expected_takeover_seq_id = takeover_seq_ids_by_proposal_id.get(proposal_id)
            if expected_takeover_seq_id is not None and expected_takeover_seq_id != seq_id:
                errors.append(
                    f"record[{idx}] executed proposal_id={proposal_id} seq_id={seq_id} "
                    f"does not match routed takeover seq_id={expected_takeover_seq_id}"
                )
            if takeover_routing_row:
                if seq_id not in takeover_seq_ids:
                    errors.append(f"record[{idx}] executed takeover proposal_id={proposal_id} seq_id={seq_id} not in takeover lane")
                if seq_id not in target_home:
                    errors.append(f"record[{idx}] executed takeover seq_id={seq_id} not in target_home_set")
                if seq_id in target_normal:
                    errors.append(f"record[{idx}] executed takeover seq_id={seq_id} still in target_normal_verify_seq_ids")
            if dict_get(base_match_by_seq, seq_id) is not True:
                errors.append(f"record[{idx}] executed seq_id={seq_id} lacks base_len match")
            if dict_get(seq_pre_verify_by_seq, seq_id) is not False:
                errors.append(f"record[{idx}] executed seq_id={seq_id} has pre_verify=true")

            proposal_len = int_value(dict_get(proposal_len_by_id, proposal_id), gamma or -1)
            to_verify_len = int_value(dict_get(to_verify_len_by_id, proposal_id), gamma or -1)
            if gamma is not None and proposal_len != gamma:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and to_verify_len != gamma:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} to_verify_len != gamma")
            if proposal_len > 0:
                expected_dry_run_tokens += proposal_len

            accepted_len = int_value(
                dict_get(accepted_len_by_proposal, proposal_id, dict_get(accepted_len_by_seq, seq_id)),
                -1,
            )
            invalidated_len = int_value(dict_get(invalidated_len_by_seq, seq_id), -1)
            reject_position = int_value(
                dict_get(reject_position_by_proposal, proposal_id, dict_get(reject_position_by_seq, seq_id)),
                -2,
            )
            result = dict_get(result_by_proposal, proposal_id)
            full_accept = bool(dict_get(full_accept_by_seq, seq_id, accepted_len == gamma))
            if gamma is not None and not (0 <= accepted_len <= gamma):
                errors.append(f"record[{idx}] accepted_len out of range for proposal_id={proposal_id}: {accepted_len}")
            if gamma is not None and invalidated_len != -1 and not (0 <= invalidated_len <= gamma):
                errors.append(f"record[{idx}] invalidated_len out of range for seq_id={seq_id}: {invalidated_len}")
            if gamma is not None and full_accept != (accepted_len == gamma):
                errors.append(
                    f"record[{idx}] full_accept mismatch for proposal_id={proposal_id}: "
                    f"full_accept={full_accept}, accepted_len={accepted_len}, gamma={gamma}"
                )
            if result == "full_accept":
                if gamma is not None and accepted_len != gamma:
                    errors.append(f"record[{idx}] full_accept result has accepted_len={accepted_len}")
            elif result == "reject_at_first_token":
                if accepted_len != 0:
                    errors.append(f"record[{idx}] reject_at_first_token result has accepted_len={accepted_len}")
            elif result == "partial_accept":
                if gamma is not None and not (0 < accepted_len < gamma):
                    errors.append(f"record[{idx}] partial_accept result has accepted_len={accepted_len}")
            elif result not in (None, "skipped_invalid"):
                errors.append(f"record[{idx}] unknown verify result={result!r} for proposal_id={proposal_id}")

            if accepted_len == gamma:
                full_accept_count += 1
                if reject_position != -1:
                    errors.append(f"record[{idx}] full accept proposal_id={proposal_id} must have reject_position=-1")
            elif accepted_len == 0:
                reject_first_count += 1
                if reject_position != accepted_len:
                    errors.append(f"record[{idx}] reject_position must equal accepted_len for proposal_id={proposal_id}")
            else:
                partial_accept_count += 1
                if reject_position != accepted_len:
                    errors.append(f"record[{idx}] reject_position must equal accepted_len for proposal_id={proposal_id}")
            accepted_len_distribution[accepted_len] += 1

            if dict_get(checkpoint_ok_by_seq, seq_id) is not True:
                checkpoint_failure_count += 1
                errors.append(f"record[{idx}] checkpoint failed for executed seq_id={seq_id}")
            if dict_get(mutation_by_seq, seq_id) is True:
                mutation_detected_count += 1
                errors.append(f"record[{idx}] mutation detected for executed seq_id={seq_id}")
            before_len = dict_get(len_before_by_seq, seq_id)
            after_len = dict_get(len_after_by_seq, seq_id)
            if before_len is not None and after_len is not None and int_value(before_len, -1) != int_value(after_len, -2):
                mutation_detected_count += 1
                errors.append(
                    f"record[{idx}] sequence len changed for executed seq_id={seq_id}: "
                    f"{before_len} -> {after_len}"
                )

        if expected_dry_run_tokens and dry_run_tokens != expected_dry_run_tokens:
            errors.append(
                f"record[{idx}] eager_tokens_verify_dry_run must equal executed proposal lengths: "
                f"{dry_run_tokens} != {expected_dry_run_tokens}"
            )

        for proposal_id in sorted(skipped_proposal_ids):
            reason = dict_get(skip_reasons, proposal_id)
            skip_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing skip reason")
            elif reason not in VALID_SKIP_REASONS:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad reason={reason!r}")

        if bool(record.get("eager_verify_dry_run_mutation_detected", False)):
            mutation_detected_count += 1
            errors.append(f"record[{idx}] eager verify dry-run mutation_detected flag is set")
        if bool(record.get("eager_verify_dry_run_checkpoint_failed", False)):
            checkpoint_failure_count += 1
            errors.append(f"record[{idx}] eager verify dry-run checkpoint_failed flag is set")

    repeated_verify_proposal_ids = sorted(
        proposal_id
        for proposal_id, step_keys in repeated_verify_steps_by_proposal.items()
        if len(step_keys) > 1
    )
    if repeated_verify_proposal_ids:
        errors.append(
            "proposal ids verify-dry-run executed in multiple unique steps/plans: "
            f"{repeated_verify_proposal_ids}"
        )
    missing_verify_for_takeover_proposal_ids = []
    if verify_records:
        missing_verify_for_takeover_proposal_ids = sorted(
            takeover_proposal_ids_seen - verify_executed_or_skipped_proposal_ids_seen
        )
        if missing_verify_for_takeover_proposal_ids:
            errors.append(
                "takeover proposal ids never verify-executed or skipped: "
                f"{missing_verify_for_takeover_proposal_ids}"
            )

    summary = {
        "total_trace_records": len(records),
        "records_with_eager_verify_dry_run_enabled": verify_records,
        "verify_active_records": verify_active_records,
        "scheduled_proposal_count": len(scheduled_proposal_ids_seen),
        "takeover_proposal_count": len(takeover_proposal_ids_seen),
        "target_eager_verify_seq_ids_dry_run_count": target_eager_verify_seq_ids_dry_run_count,
        "verify_dry_run_executed_proposal_count": len(executed_proposal_ids_seen),
        "verify_dry_run_skipped_proposal_count": len(skipped_proposal_ids_seen),
        "skip_reason_counts": dict(skip_reason_counts),
        "unique_verified_dry_run_proposal_ids": len(executed_proposal_ids_seen),
        "unique_verified_dry_run_seq_ids": len(executed_seq_ids_seen),
        "accepted_len_distribution": dict(accepted_len_distribution),
        "full_accept_count": full_accept_count,
        "reject_first_count": reject_first_count,
        "partial_accept_count": partial_accept_count,
        "checkpoint_failure_count": checkpoint_failure_count,
        "mutation_detected_count": mutation_detected_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "repeated_verify_proposal_ids": repeated_verify_proposal_ids,
        "missing_verify_for_takeover_proposal_ids": missing_verify_for_takeover_proposal_ids,
        "eager_tokens_verify_dry_run": verify_tokens,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_eager_verify_dry_run_enabled",
        "verify_active_records",
        "scheduled_proposal_count",
        "takeover_proposal_count",
        "target_eager_verify_seq_ids_dry_run_count",
        "verify_dry_run_executed_proposal_count",
        "verify_dry_run_skipped_proposal_count",
        "skip_reason_counts",
        "unique_verified_dry_run_proposal_ids",
        "unique_verified_dry_run_seq_ids",
        "accepted_len_distribution",
        "full_accept_count",
        "reject_first_count",
        "partial_accept_count",
        "checkpoint_failure_count",
        "mutation_detected_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "missing_buffered_proposal_unexpected_count",
        "repeated_verify_proposal_ids",
        "missing_verify_for_takeover_proposal_ids",
        "eager_tokens_verify_dry_run",
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
        "eager_verify_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_home_set": [],
        "target_normal_verify_seq_ids": [],
        "target_eager_set": [],
        "target_eager_verify_seq_ids_dry_run": [],
        "target_eager_verify_proposal_ids_dry_run": [],
        "missing_buffered_proposal_allowed_by_eager_seq_ids": [],
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "eager_verify_dry_run_candidate_proposal_ids": [],
        "eager_verify_dry_run_candidate_seq_ids": [],
        "eager_verify_dry_run_skipped_proposal_ids": [],
        "eager_verify_dry_run_skip_reason_by_proposal_id": {},
        "eager_verify_dry_run_executed_proposal_ids": [],
        "eager_verify_dry_run_executed_seq_ids": [],
        "eager_tokens_verify_dry_run": 0,
        "eager_tokens_verify_dry_run_full_accept": 0,
        "eager_tokens_verify_dry_run_rejected": 0,
        "eager_tokens_verify_dry_run_partial_accept": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_passive_takeover_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_lane_exclusion_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "eager_verify_dry_run_enabled": False,
            "target_home_set": [3, 8],
            "target_normal_verify_seq_ids": [8],
            "target_eager_verify_seq_ids_dry_run": [3],
            "target_eager_verify_proposal_ids_dry_run": [101],
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [3],
        }
    )
    return record


def synthetic_takeover_record(accepted_len: int = 4, *, step_id: int = 7) -> dict[str, Any]:
    record = synthetic_base_record()
    gamma = 4
    full_accept = accepted_len == gamma
    partial = 0 < accepted_len < gamma
    rejected_first = accepted_len == 0
    result = "full_accept" if full_accept else "partial_accept" if partial else "reject_at_first_token"
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_lane_exclusion_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "eager_verify_dry_run_enabled": True,
            "eager_verify_dry_run_source": "phase1h5e3_takeover_lane",
            "eager_verify_dry_run_step_id": step_id,
            "eager_verify_dry_run_plan_id": 11,
            "target_home_set": [3, 4],
            "target_normal_verify_seq_ids": [4],
            "target_eager_verify_seq_ids_dry_run": [3],
            "target_eager_verify_proposal_ids_dry_run": [101],
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [3],
            "eager_verify_dry_run_candidate_proposal_ids": [101],
            "eager_verify_dry_run_candidate_seq_ids": [3],
            "eager_verify_dry_run_executed_proposal_ids": [101],
            "eager_verify_dry_run_executed_seq_ids": [3],
            "eager_verify_dry_run_seq_id_by_proposal_id": {"101": 3},
            "eager_verify_dry_run_proposal_len_by_proposal_id": {"101": gamma},
            "eager_verify_dry_run_to_verify_len_by_proposal_id": {"101": gamma},
            "eager_verify_dry_run_base_len_by_proposal_id": {"101": 12},
            "eager_verify_dry_run_takeover_step_by_proposal_id": {"101": step_id},
            "eager_verify_dry_run_accept_len_by_proposal_id": {"101": accepted_len},
            "eager_verify_dry_run_result_by_proposal_id": {"101": result},
            "eager_verify_dry_run_reject_position_by_proposal_id": {"101": -1 if full_accept else accepted_len},
            "eager_verify_dry_run_full_accept_proposal_ids": [101] if full_accept else [],
            "eager_verify_dry_run_rejected_proposal_ids": [101] if rejected_first else [],
            "eager_verify_dry_run_partial_accept_proposal_ids": [101] if partial else [],
            "eager_verify_dry_run_sequence_len_before_by_seq_id": {"3": 12},
            "eager_verify_dry_run_sequence_len_after_by_seq_id": {"3": 12},
            "eager_verify_accepted_len_by_seq_id": {"3": accepted_len},
            "eager_verify_full_accept_by_seq_id": {"3": full_accept},
            "eager_verify_reject_position_by_seq_id": {"3": -1 if full_accept else accepted_len},
            "eager_verify_invalidated_len_by_seq_id": {"3": 0 if full_accept else gamma - accepted_len},
            "eager_verify_revised_token_by_seq_id": {"3": -1 if full_accept else 200},
            "eager_verify_base_len_by_seq_id": {"3": 12},
            "eager_verify_current_len_by_seq_id": {"3": 12},
            "eager_verify_base_match_by_seq_id": {"3": True},
            "eager_verify_seq_pre_verify_by_seq_id": {"3": False},
            "eager_verify_seq_status_before_by_seq_id": {"3": "RUNNING"},
            "eager_verify_seq_status_after_by_seq_id": {"3": "RUNNING"},
            "eager_verify_mutation_detected_by_seq_id": {"3": False},
            "eager_verify_checkpoint_ok_by_seq_id": {"3": True},
            "eager_tokens_verify_dry_run": gamma,
            "eager_tokens_verify_dry_run_full_accept": gamma if full_accept else 0,
            "eager_tokens_verify_dry_run_rejected": gamma if rejected_first else 0,
            "eager_tokens_verify_dry_run_partial_accept": gamma if partial else 0,
        }
    )
    return record


def synthetic_old_verify_record(accepted_len: int = 4) -> dict[str, Any]:
    record = synthetic_base_record()
    full_accept = accepted_len == 4
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "eager_verify_dry_run_enabled": True,
            "scheduled_target_eager_proposal_ids_dry_run": [201],
            "scheduled_target_eager_seq_ids_dry_run": [5],
            "eager_verify_candidate_proposal_ids": [201],
            "eager_verify_candidate_seq_ids": [5],
            "eager_verify_executed_proposal_ids": [201],
            "eager_verify_executed_seq_ids": [5],
            "eager_schedule_proposal_len_by_proposal_id": {"201": 4},
            "eager_schedule_to_verify_len_by_proposal_id": {"201": 4},
            "eager_verify_accepted_len_by_seq_id": {"5": accepted_len},
            "eager_verify_full_accept_by_seq_id": {"5": full_accept},
            "eager_verify_reject_position_by_seq_id": {"5": -1 if full_accept else accepted_len},
            "eager_verify_invalidated_len_by_seq_id": {"5": 0 if full_accept else 4 - accepted_len},
            "eager_verify_base_match_by_seq_id": {"5": True},
            "eager_verify_seq_pre_verify_by_seq_id": {"5": False},
            "eager_verify_checkpoint_ok_by_seq_id": {"5": True},
            "eager_verify_mutation_detected_by_seq_id": {"5": False},
            "eager_tokens_verify_dry_run": 4,
            "eager_tokens_verify_dry_run_full_accept": 4 if full_accept else 0,
            "eager_tokens_verify_dry_run_rejected": 0 if full_accept else 4,
            "eager_tokens_verify_dry_run_partial_accept": 0,
        }
    )
    return record


def synthetic_verify_skip_record(reason: str) -> dict[str, Any]:
    record = synthetic_takeover_record()
    record.update(
        {
            "eager_verify_dry_run_executed_proposal_ids": [],
            "eager_verify_dry_run_executed_seq_ids": [],
            "eager_verify_dry_run_skipped_proposal_ids": [101],
            "eager_verify_dry_run_skip_reason_by_proposal_id": {"101": reason},
            "eager_verify_dry_run_accept_len_by_proposal_id": {},
            "eager_verify_dry_run_result_by_proposal_id": {"101": "skipped_invalid"},
            "eager_verify_accepted_len_by_seq_id": {},
            "eager_verify_full_accept_by_seq_id": {},
            "eager_verify_checkpoint_ok_by_seq_id": {},
            "eager_verify_mutation_detected_by_seq_id": {},
            "eager_tokens_verify_dry_run": 0,
            "eager_tokens_verify_dry_run_full_accept": 0,
            "eager_tokens_verify_dry_run_rejected": 0,
            "eager_tokens_verify_dry_run_partial_accept": 0,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_old_verify_record(4),
        synthetic_passive_takeover_record(),
        synthetic_takeover_record(4),
        synthetic_takeover_record(2),
        synthetic_takeover_record(0),
        synthetic_verify_skip_record("base_overshot_before_verify"),
        synthetic_verify_skip_record("seq_returned_pre_verify_before_verify"),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager verify records failed: {errors}"

    errors, _ = validate_records([synthetic_passive_takeover_record()])
    assert not any("takeover verify candidates must equal" in error for error in errors), (
        "passive takeover rows must not require verify candidates"
    )
    assert any("never verify-executed or skipped" in error for error in errors), (
        "checker missed takeover proposal with no verify execution or skip"
    )

    repeated_same_step = [synthetic_takeover_record(4), synthetic_takeover_record(4)]
    errors, _ = validate_records(repeated_same_step)
    assert not errors, f"same-step repeated trace rows should pass: {errors}"

    invalid = [synthetic_takeover_record(4), synthetic_takeover_record(4, step_id=8)]
    errors, _ = validate_records(invalid)
    assert any("multiple unique steps" in error for error in errors), "checker missed repeated verify"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_verify_full_accept_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("full_accept mismatch" in error for error in errors), "checker missed full_accept mismatch"

    invalid = deepcopy(valid_records)
    invalid[3]["target_normal_verify_seq_ids"] = [3, 4]
    errors, _ = validate_records(invalid)
    assert any("target_normal_verify_seq_ids" in error for error in errors), "checker missed target-normal overlap"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_verify_dry_run_mutation_detected"] = True
    errors, _ = validate_records(invalid)
    assert any("mutation_detected" in error for error in errors), "checker missed mutation flag"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_verify_checkpoint_ok_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("checkpoint failed" in error for error in errors), "checker missed checkpoint failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager verification counters" in error for error in errors), (
        "checker missed actual eager verified counter"
    )

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_verify_dry_run_accept_len_by_proposal_id"] = {"101": 5}
    invalid[3]["eager_verify_accepted_len_by_seq_id"] = {"3": 5}
    errors, _ = validate_records(invalid)
    assert any("accepted_len out of range" in error for error in errors), "checker missed accepted_len range"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_verify_dry_run_candidate_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("candidates must equal target eager proposals" in error for error in errors), (
        "checker missed takeover candidate proposal mismatch"
    )

    print("Synthetic eager verify dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H eager verify dry-run traces.")
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
    print("\nEager verify dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
