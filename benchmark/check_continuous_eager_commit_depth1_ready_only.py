#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


CONTINUOUS_SOURCE = "continuous_shadow"
CONTINUOUS_COMMIT_SOURCE = "continuous_depth1_ready_only"
ONE_SHOT_PARENT_SOURCE = "phase1h6a_one_shot_commit"
FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"
CONTINUOUS_TIMING_FIELDS = (
    "continuous_eager_commit_decision_broadcast_time_ms",
    "continuous_eager_result_transfer_time_ms",
    "continuous_eager_sync_apply_dry_run_time_ms",
    "continuous_eager_sync_apply_time_ms",
    "continuous_eager_commit_time_ms",
    "continuous_eager_real_commit_time_ms",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import (  # noqa: E402
    as_int_list,
    as_int_set,
    dict_get,
    int_value,
    is_dual_record,
    load_trace,
    synthetic_commit_record,
    validate_records as validate_one_shot_records,
)
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting  # noqa: E402


def as_int_map(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, int] = {}
    for key, item in value.items():
        try:
            result[int(key)] = int(item)
        except Exception:
            continue
    return result


def as_str_map(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, str] = {}
    for key, item in value.items():
        try:
            result[int(key)] = str(item)
        except Exception:
            continue
    return result


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def commit_active(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("continuous_eager_commit_enabled", False))
        or bool(record.get("enable_continuous_eager_commit_depth1_ready_only", False))
        or bool(as_int_set(record.get("continuous_eager_commit_candidate_proposal_ids")))
        or bool(as_int_set(record.get("continuous_eager_real_committed_proposal_ids")))
        or bool(as_int_set(record.get("continuous_eager_real_commit_skipped_proposal_ids")))
        or int_value(record.get("continuous_eager_tokens_committed"), 0) > 0
    )


def step_plan_key(record: dict[str, Any]) -> tuple[int, int]:
    plan_id = int_value(record.get("continuous_eager_commit_plan_id"), int_value(record.get("plan_id"), -1))
    step_id = int_value(record.get("continuous_eager_commit_step_id"), int_value(record.get("step_id"), -1))
    return plan_id, step_id


def combined_accounting_errors(
    accounting: dict[str, Any],
    *,
    one_shot_tokens: int,
    continuous_tokens: int,
    legal_partial_prefix_total_recovered_token_count: int = 0,
) -> list[str]:
    combined_tokens = int_value(accounting.get("combined_real_committed_token_count"), 0)
    depth2_tokens = int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
    depth3_tokens = int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0)
    depth4_tokens = int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0)
    lower_bound = one_shot_tokens + continuous_tokens
    expected_with_known_higher_depth = (
        lower_bound
        + depth2_tokens
        + depth3_tokens
        + int(legal_partial_prefix_total_recovered_token_count)
    )
    depth4_commit_enabled = (
        bool(accounting.get("enable_rolling_continuous_depth4_commit_ready_only", False))
        or bool(accounting.get("rolling_depth4_commit_enabled", False))
    )
    if depth4_tokens > 0:
        if not depth4_commit_enabled:
            return ["depth4 real committed tokens present while depth4 commit flag is disabled"]
        expected_with_known_higher_depth += depth4_tokens
    higher_depth_tokens_present = (
        depth2_tokens > 0
        or depth3_tokens > 0
        or depth4_tokens > 0
        or int(legal_partial_prefix_total_recovered_token_count) > 0
    )
    higher_depth_enabled_or_present = (
        higher_depth_tokens_present
        or bool(accounting.get("rolling_depth2_commit_enabled", False))
        or bool(accounting.get("rolling_depth3_commit_enabled", False))
        or int_value(accounting.get("rolling_depth3_real_commit_count"), 0) > 0
        or depth4_commit_enabled
    )

    if higher_depth_tokens_present:
        if combined_tokens != expected_with_known_higher_depth:
            return [
                "combined real committed token count mismatch "
                "(expected one-shot + depth1 + known rolling depth2/depth3/depth4 + legal partial recovery tokens)"
            ]
    elif higher_depth_enabled_or_present:
        if combined_tokens < lower_bound:
            return ["combined real committed token count is below one-shot + depth1 lower bound"]
    elif combined_tokens != lower_bound:
        return ["combined real committed token count mismatch"]
    return []


def partial_recovery_accounting_errors(
    accounting: dict[str, Any],
    *,
    descendant_committed_after_partial_count: int = 0,
) -> tuple[list[str], int]:
    errors: list[str] = []
    enabled = bool(accounting.get("partial_prefix_recovery_enabled", False)) or bool(
        accounting.get("enable_rolling_continuous_partial_prefix_recovery", False)
    )
    success_count = int_value(accounting.get("partial_prefix_recovery_success_count"), 0)
    partial_accepted = int_value(accounting.get("partial_prefix_accepted_token_count"), 0)
    partial_revised = int_value(accounting.get("partial_prefix_revised_token_count"), 0)
    partial_total = int_value(accounting.get("partial_prefix_total_recovered_token_count"), 0)
    len_mismatch = int_value(accounting.get("partial_recovery_target_draft_length_mismatch_count"), 0)
    token_mismatch = int_value(accounting.get("partial_recovery_target_draft_token_mismatch_count"), 0)

    if partial_total and not enabled:
        errors.append("partial recovery tokens present while partial-prefix recovery is disabled")
    if partial_total < 0:
        errors.append("partial recovery total token count must be nonnegative")
    if partial_total and success_count <= 0:
        errors.append("partial recovery tokens require successful partial recovery evidence")
    if partial_total != partial_accepted + partial_revised:
        errors.append("partial recovery total tokens must equal accepted prefix plus revised tokens")
    if partial_revised and partial_revised != success_count:
        errors.append("partial recovery revised token count must equal successful recovery count")
    if len_mismatch:
        errors.append("partial recovery target/draft length mismatch count must be zero")
    if token_mismatch:
        errors.append("partial recovery target/draft token mismatch count must be zero")
    if descendant_committed_after_partial_count:
        errors.append("descendant committed after partial recovery")

    legal_partial_total = 0
    if (
        enabled
        and success_count > 0
        and partial_total >= 0
        and partial_total == partial_accepted + partial_revised
        and len_mismatch == 0
        and token_mismatch == 0
        and descendant_committed_after_partial_count == 0
    ):
        legal_partial_total = partial_total

    if legal_partial_total:
        combined_accepted = int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0)
        combined_revised = int_value(accounting.get("combined_actual_revised_token_increment_sum"), 0)
        combined_output = int_value(accounting.get("combined_actual_output_token_increment_sum"), 0)
        combined_verified = int_value(accounting.get("combined_actual_verified_token_increment_sum"), 0)
        combined_tokens = int_value(accounting.get("combined_real_committed_token_count"), 0)
        if combined_output != combined_accepted + combined_revised:
            errors.append("combined output increment must equal accepted plus revised increments in partial recovery mode")
        if combined_revised != partial_revised:
            errors.append("combined revised increment must equal partial revised token count")
        if combined_output != combined_tokens:
            errors.append("combined output increment must equal combined real committed token count")
        if combined_verified != combined_tokens:
            errors.append("combined verified increment must equal combined real committed token count")

    return errors, legal_partial_total


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    one_shot_errors, one_shot_summary = validate_one_shot_records(records)
    errors.extend(f"one-shot commit checker: {error}" for error in one_shot_errors)
    accounting = aggregate_performance_accounting(records, {})

    one_shot_committed_ids: set[int] = set()
    ready_shadow_ids: set[int] = set()
    not_ready_ids: set[int] = set()
    full_accept_ids: set[int] = set()
    partial_reject_ids: set[int] = set()
    parent_by_id: dict[int, int] = {}
    parent_source_by_id: dict[int, str] = {}
    depth_by_id: dict[int, int] = {}
    token_by_id: dict[int, int] = {}

    records_with_commit_enabled = 0
    commit_active_records = 0
    committed_ids_seen: set[int] = set()
    committed_sides_by_id: dict[int, set[str]] = defaultdict(set)
    committed_steps_by_side: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    skipped_ids_seen: set[int] = set()
    skip_reason_by_id: dict[int, str] = {}
    committed_but_not_ready: set[int] = set()
    committed_depth_not_one: set[int] = set()
    committed_bad_parent: set[int] = set()
    committed_non_full_accept: set[int] = set()
    committed_not_ready: set[int] = set()
    repeated_commit_ids: set[int] = set()
    duplicate_seq_depth_events: set[tuple[str, int, int, int, int]] = set()
    seen_seq_depth: set[tuple[str, int, int, int, int]] = set()
    missing_unexpected_count = 0
    real_target_eager_nonempty_count = 0
    depth2_real_commit_count = 0
    depth4_commit_enabled = False
    depth4_real_commit_count = 0
    depth_gt4_real_commit_count = 0
    timing_sums = {field: 0.0 for field in CONTINUOUS_TIMING_FIELDS}
    timing_negative_fields: set[str] = set()
    result_transfer_protocols: set[str] = set()
    result_transfer_payload_len_units = 0
    result_transfer_payload_len_units_before_compact = 0
    descendant_committed_after_partial_count = 0

    for record in records:
        if not is_dual_record(record):
            continue
        one_shot_committed_ids.update(as_int_set(record.get("eager_committed_proposal_ids")))
        ready_shadow_ids.update(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
        not_ready_ids.update(as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids")))
        full_accept_ids.update(as_int_set(record.get("continuous_eager_full_accept_proposal_ids")))
        partial_reject_ids.update(as_int_set(record.get("continuous_eager_partial_reject_proposal_ids")))
        parent_by_id.update(as_int_map(record.get("continuous_eager_parent_proposal_id_by_proposal_id")))
        parent_source_by_id.update(as_str_map(record.get("continuous_eager_parent_source_by_proposal_id")))
        depth_by_id.update(as_int_map(record.get("continuous_eager_chain_depth_by_proposal_id")))
        token_by_id.update(as_int_map(record.get("continuous_eager_candidate_token_count_by_proposal_id")))

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue
        if as_int_set(record.get("target_eager_set")):
            real_target_eager_nonempty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty")
        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(f"record[{idx}] unexpected missing buffered proposals: {sorted(missing_unexpected)}")
        for field in CONTINUOUS_TIMING_FIELDS:
            if field in record:
                value = float_value(record.get(field), 0.0)
                if value < 0.0:
                    timing_negative_fields.add(field)
                    errors.append(f"record[{idx}] {field} must be nonnegative")
                else:
                    timing_sums[field] += value
        protocol = record.get("continuous_eager_result_transfer_protocol")
        if protocol:
            result_transfer_protocols.add(str(protocol))
            if str(protocol) != "compact_v1":
                errors.append(f"record[{idx}] unexpected continuous result-transfer protocol {protocol!r}")
        payload_len = int_value(record.get("continuous_eager_result_transfer_payload_len_units"), 0)
        payload_before = int_value(record.get("continuous_eager_result_transfer_payload_len_units_before_compact"), 0)
        if payload_len < 0 or payload_before < 0:
            errors.append(f"record[{idx}] continuous result-transfer payload lengths must be nonnegative")
        if payload_before and payload_len > payload_before:
            errors.append(f"record[{idx}] compact continuous result-transfer payload grew")
        result_transfer_payload_len_units += max(0, payload_len)
        result_transfer_payload_len_units_before_compact += max(0, payload_before)
        descendant_committed_after_partial_count += int_value(
            record.get("descendant_committed_after_partial_count"),
            0,
        )

        enabled = bool(record.get("enable_continuous_eager_commit_depth1_ready_only", False))
        if enabled:
            records_with_commit_enabled += 1
        if not commit_active(record):
            continue
        commit_active_records += 1
        if not enabled:
            errors.append(f"record[{idx}] continuous real commit active while flag disabled")
        if record.get("continuous_eager_commit_source") not in {None, CONTINUOUS_COMMIT_SOURCE}:
            errors.append(f"record[{idx}] bad continuous commit source {record.get('continuous_eager_commit_source')!r}")

        side = str(record.get("continuous_eager_commit_side") or "")
        plan_id, step_id = step_plan_key(record)
        committed_ids = as_int_set(record.get("continuous_eager_real_committed_proposal_ids"))
        committed_seq_ids = as_int_list(record.get("continuous_eager_real_committed_seq_ids"))
        pid_to_seq = dict(zip(as_int_list(record.get("continuous_eager_real_committed_proposal_ids")), committed_seq_ids))
        token_count_by_id = as_int_map(record.get("continuous_eager_real_committed_token_count_by_proposal_id"))
        accept_len_by_id = as_int_map(record.get("continuous_eager_real_committed_accept_len_by_proposal_id"))
        action_by_id = as_str_map(record.get("continuous_eager_real_commit_action_by_proposal_id"))
        result_by_id = as_str_map(record.get("continuous_eager_real_commit_verify_result_by_proposal_id"))
        precondition_ok_by_id = record.get("continuous_eager_real_commit_precondition_ok_by_proposal_id") or {}
        failed_by_id = record.get("continuous_eager_real_commit_precondition_failed_by_proposal_id") or {}
        skip_reason_map = as_str_map(record.get("continuous_eager_real_commit_skip_reason_by_proposal_id"))
        skipped = as_int_set(record.get("continuous_eager_real_commit_skipped_proposal_ids"))
        skipped_ids_seen.update(skipped)
        for proposal_id, reason in skip_reason_map.items():
            skip_reason_by_id.setdefault(proposal_id, reason)
        depth2_real_commit_count += int_value(record.get("continuous_depth2_real_commit_count"), 0)

        if bool(record.get("enable_rolling_continuous_depth4_commit_ready_only", False)) or bool(
            record.get("rolling_depth4_commit_enabled", False)
        ):
            depth4_commit_enabled = True
        depth4_real_commit_count += int_value(record.get("rolling_depth4_real_commit_count"), 0)
        depth_gt4_real_commit_count += int_value(record.get("rolling_depth_gt4_real_commit_count"), 0)

        if committed_ids & as_int_set(record.get("eager_committed_proposal_ids")):
            errors.append(f"record[{idx}] continuous proposal appeared in one-shot commit fields")
        if committed_ids & as_int_set(record.get("lane_exclusion_applied_proposal_ids")):
            errors.append(f"record[{idx}] continuous proposal affected lane exclusion")
        if committed_ids & as_int_set(record.get("target_eager_verify_proposal_ids_dry_run")):
            errors.append(f"record[{idx}] continuous proposal entered target takeover")

        before_map = as_int_map(record.get("continuous_eager_target_seq_len_before_by_seq_id"))
        after_map = as_int_map(record.get("continuous_eager_target_seq_len_after_by_seq_id"))
        len_match_map = record.get("continuous_eager_target_draft_len_match_by_seq_id") or {}
        token_match_map = record.get("continuous_eager_target_draft_token_match_by_seq_id") or {}

        for proposal_id in committed_ids:
            seq_id = int(pid_to_seq.get(proposal_id, -1))
            committed_ids_seen.add(proposal_id)
            committed_sides_by_id[proposal_id].add(side)
            committed_steps_by_side[(side, proposal_id)].add((plan_id, step_id))
            depth = int(dict_get(
                record.get("continuous_eager_chain_depth_by_proposal_id"),
                proposal_id,
                depth_by_id.get(proposal_id, 0),
            ))
            parent_id = int(dict_get(
                record.get("continuous_eager_parent_proposal_id_by_proposal_id"),
                proposal_id,
                parent_by_id.get(proposal_id, -1),
            ))
            parent_source = str(dict_get(
                record.get("continuous_eager_parent_source_by_proposal_id"),
                proposal_id,
                parent_source_by_id.get(proposal_id, ""),
            ))
            token_count = int(token_count_by_id.get(proposal_id, token_by_id.get(proposal_id, 0)))
            accept_len = int(accept_len_by_id.get(proposal_id, -1))
            if proposal_id not in ready_shadow_ids:
                committed_but_not_ready.add(proposal_id)
            if depth != 1:
                committed_depth_not_one.add(proposal_id)
            if parent_source != ONE_SHOT_PARENT_SOURCE or parent_id not in one_shot_committed_ids:
                committed_bad_parent.add(proposal_id)
            if proposal_id in partial_reject_ids or proposal_id in not_ready_ids:
                committed_not_ready.add(proposal_id)
            if result_by_id.get(proposal_id) != "full_accept" or action_by_id.get(proposal_id) != FULL_ACCEPT_ACTION:
                committed_non_full_accept.add(proposal_id)
            if token_count <= 0 or accept_len != token_count:
                errors.append(f"record[{idx}] committed proposal {proposal_id} has bad token/accept count")
            if precondition_ok_by_id and not bool(dict_get(precondition_ok_by_id, proposal_id, False)):
                errors.append(f"record[{idx}] committed proposal {proposal_id} has precondition_ok=false")
            if bool(dict_get(failed_by_id, proposal_id, False)):
                errors.append(f"record[{idx}] committed proposal {proposal_id} has failed precondition")
            if seq_id not in before_map or seq_id not in after_map:
                errors.append(f"record[{idx}] committed seq {seq_id} missing length checkpoint")
            elif int(after_map[seq_id]) - int(before_map[seq_id]) != token_count:
                errors.append(f"record[{idx}] committed seq {seq_id} length delta != token count")
            if len_match_map and not bool(dict_get(len_match_map, seq_id, False)):
                errors.append(f"record[{idx}] committed seq {seq_id} target/draft length mismatch")
            if token_match_map and not bool(dict_get(token_match_map, seq_id, False)):
                errors.append(f"record[{idx}] committed seq {seq_id} target/draft token mismatch")
            seq_depth_event = (side, plan_id, step_id, seq_id, depth)
            if seq_depth_event in seen_seq_depth:
                duplicate_seq_depth_events.add(seq_depth_event)
            seen_seq_depth.add(seq_depth_event)

    for (_side, proposal_id), steps in committed_steps_by_side.items():
        if len(steps) > 1:
            repeated_commit_ids.add(proposal_id)

    missing_target = {proposal_id for proposal_id, sides in committed_sides_by_id.items() if "target" not in sides}
    missing_draft = {proposal_id for proposal_id, sides in committed_sides_by_id.items() if "draft" not in sides}
    for proposal_id in skipped_ids_seen:
        if proposal_id not in skip_reason_by_id:
            errors.append(f"skipped continuous proposal {proposal_id} lacks reason")
    if committed_but_not_ready:
        errors.append(f"continuous committed but not shadow-ready: {sorted(committed_but_not_ready)}")
    if committed_depth_not_one:
        errors.append(f"continuous committed non-depth-1 proposals: {sorted(committed_depth_not_one)}")
    if committed_bad_parent:
        errors.append(f"continuous committed proposals with bad parent: {sorted(committed_bad_parent)}")
    if committed_non_full_accept:
        errors.append(f"continuous committed non-full-accept proposals: {sorted(committed_non_full_accept)}")
    if committed_not_ready:
        errors.append(f"continuous not-ready proposals were committed: {sorted(committed_not_ready)}")
    if repeated_commit_ids:
        errors.append(f"repeated continuous commit proposal ids: {sorted(repeated_commit_ids)}")
    if duplicate_seq_depth_events:
        errors.append(f"duplicate continuous seq/depth commit events: {sorted(duplicate_seq_depth_events)}")
    if missing_target:
        errors.append(f"missing target-side continuous commit records: {sorted(missing_target)}")
    if missing_draft:
        errors.append(f"missing draft-side continuous commit records: {sorted(missing_draft)}")
    if depth2_real_commit_count:
        errors.append("continuous depth-2 real commit count must be zero")
    if depth4_real_commit_count and not depth4_commit_enabled:
        errors.append("rolling depth4 real commit count must be zero")
    if depth_gt4_real_commit_count:
        errors.append("rolling depth>4 real commit count must be zero")
    partial_errors, legal_partial_total = partial_recovery_accounting_errors(
        accounting,
        descendant_committed_after_partial_count=descendant_committed_after_partial_count,
    )
    errors.extend(partial_errors)

    continuous_tokens = int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
    if continuous_tokens != int_value(accounting.get("continuous_target_actual_verified_token_increment_sum"), 0):
        errors.append("continuous tokens != target continuous verified increment sum")
    if continuous_tokens != int_value(accounting.get("continuous_target_actual_accepted_token_increment_sum"), 0):
        errors.append("continuous tokens != target continuous accepted increment sum")
    if continuous_tokens != int_value(accounting.get("continuous_draft_actual_verified_token_increment_sum"), 0):
        errors.append("continuous tokens != draft continuous verified increment sum")
    if continuous_tokens != int_value(accounting.get("continuous_draft_actual_accepted_token_increment_sum"), 0):
        errors.append("continuous tokens != draft continuous accepted increment sum")
    for field in (
        "continuous_target_actual_rejected_token_increment_sum",
        "continuous_target_actual_invalidated_token_increment_sum",
        "continuous_draft_actual_rejected_token_increment_sum",
        "continuous_draft_actual_invalidated_token_increment_sum",
    ):
        if int_value(accounting.get(field), 0) != 0:
            errors.append(f"{field} must be zero")
    one_shot_tokens = int_value(one_shot_summary.get("committed_token_count"), 0)
    depth4_tokens = int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0)
    expected_combined = (
        one_shot_tokens
        + continuous_tokens
        + int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0)
    )
    if depth4_tokens > 0 and depth4_commit_enabled:
        expected_combined += depth4_tokens
    expected_combined += legal_partial_total
    errors.extend(
        combined_accounting_errors(
            accounting,
            one_shot_tokens=one_shot_tokens,
            continuous_tokens=continuous_tokens,
            legal_partial_prefix_total_recovered_token_count=legal_partial_total,
        )
    )

    summary = {
        "total_trace_records": len(records),
        "records_with_continuous_commit_enabled": records_with_commit_enabled,
        "continuous_commit_active_records": commit_active_records,
        "one_shot_committed_token_count": one_shot_tokens,
        "continuous_shadow_ready_proposal_count": len(ready_shadow_ids),
        "continuous_real_committed_proposal_count": len(committed_ids_seen),
        "continuous_real_committed_token_count": continuous_tokens,
        "rolling_depth2_real_committed_token_count": accounting.get("rolling_depth2_real_committed_token_count", 0),
        "rolling_depth3_real_committed_token_count": accounting.get("rolling_depth3_real_committed_token_count", 0),
        "rolling_depth4_real_committed_token_count": depth4_tokens,
        "rolling_depth4_real_commit_count": depth4_real_commit_count,
        "rolling_depth_gt4_real_commit_count": depth_gt4_real_commit_count,
        "depth4_commit_enabled": depth4_commit_enabled,
        "partial_prefix_recovery_enabled": accounting.get("partial_prefix_recovery_enabled", False),
        "partial_prefix_recovery_attempt_count": accounting.get("partial_prefix_recovery_attempt_count", 0),
        "partial_prefix_recovery_success_count": accounting.get("partial_prefix_recovery_success_count", 0),
        "partial_prefix_accepted_token_count": accounting.get("partial_prefix_accepted_token_count", 0),
        "partial_prefix_revised_token_count": accounting.get("partial_prefix_revised_token_count", 0),
        "partial_prefix_total_recovered_token_count": accounting.get("partial_prefix_total_recovered_token_count", 0),
        "partial_recovery_cascade_discard_count": accounting.get("partial_recovery_cascade_discard_count", 0),
        "descendant_committed_after_partial_count": descendant_committed_after_partial_count,
        "combined_actual_verified_token_increment_sum": accounting.get(
            "combined_actual_verified_token_increment_sum", 0
        ),
        "combined_actual_accepted_token_increment_sum": accounting.get(
            "combined_actual_accepted_token_increment_sum", 0
        ),
        "combined_actual_revised_token_increment_sum": accounting.get(
            "combined_actual_revised_token_increment_sum", 0
        ),
        "combined_actual_output_token_increment_sum": accounting.get(
            "combined_actual_output_token_increment_sum", 0
        ),
        "expected_combined_real_committed_token_count": expected_combined,
        "combined_real_committed_token_count": accounting.get("combined_real_committed_token_count", 0),
        "continuous_target_actual_verified_token_increment_sum": accounting.get(
            "continuous_target_actual_verified_token_increment_sum", 0
        ),
        "continuous_target_actual_accepted_token_increment_sum": accounting.get(
            "continuous_target_actual_accepted_token_increment_sum", 0
        ),
        "continuous_draft_actual_verified_token_increment_sum": accounting.get(
            "continuous_draft_actual_verified_token_increment_sum", 0
        ),
        "continuous_draft_actual_accepted_token_increment_sum": accounting.get(
            "continuous_draft_actual_accepted_token_increment_sum", 0
        ),
        "continuous_real_commit_skip_reason_counts": dict(Counter(skip_reason_by_id.values())),
        "continuous_depth2_real_commit_count": depth2_real_commit_count,
        "continuous_timing_sums_ms": dict(timing_sums),
        "continuous_timing_negative_fields": sorted(timing_negative_fields),
        "continuous_result_transfer_protocols": sorted(result_transfer_protocols),
        "continuous_result_transfer_payload_len_units": result_transfer_payload_len_units,
        "continuous_result_transfer_payload_len_units_before_compact": result_transfer_payload_len_units_before_compact,
        "repeated_continuous_commit_proposal_ids": sorted(repeated_commit_ids),
        "continuous_committed_but_not_shadow_ready_ids": sorted(committed_but_not_ready),
        "continuous_committed_non_full_accept_ids": sorted(committed_non_full_accept),
        "missing_target_side_continuous_commit_ids": sorted(missing_target),
        "missing_draft_side_continuous_commit_ids": sorted(missing_draft),
        "real_target_eager_non_empty_count": real_target_eager_nonempty_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_continuous_commit_enabled",
        "continuous_commit_active_records",
        "one_shot_committed_token_count",
        "continuous_shadow_ready_proposal_count",
        "continuous_real_committed_proposal_count",
        "continuous_real_committed_token_count",
        "rolling_depth2_real_committed_token_count",
        "rolling_depth3_real_committed_token_count",
        "rolling_depth4_real_committed_token_count",
        "rolling_depth4_real_commit_count",
        "rolling_depth_gt4_real_commit_count",
        "depth4_commit_enabled",
        "partial_prefix_recovery_enabled",
        "partial_prefix_recovery_attempt_count",
        "partial_prefix_recovery_success_count",
        "partial_prefix_accepted_token_count",
        "partial_prefix_revised_token_count",
        "partial_prefix_total_recovered_token_count",
        "partial_recovery_cascade_discard_count",
        "descendant_committed_after_partial_count",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "combined_actual_revised_token_increment_sum",
        "combined_actual_output_token_increment_sum",
        "expected_combined_real_committed_token_count",
        "combined_real_committed_token_count",
        "continuous_target_actual_verified_token_increment_sum",
        "continuous_target_actual_accepted_token_increment_sum",
        "continuous_draft_actual_verified_token_increment_sum",
        "continuous_draft_actual_accepted_token_increment_sum",
        "continuous_real_commit_skip_reason_counts",
        "continuous_depth2_real_commit_count",
        "continuous_timing_sums_ms",
        "continuous_timing_negative_fields",
        "continuous_result_transfer_protocols",
        "continuous_result_transfer_payload_len_units",
        "continuous_result_transfer_payload_len_units_before_compact",
        "repeated_continuous_commit_proposal_ids",
        "continuous_committed_but_not_shadow_ready_ids",
        "continuous_committed_non_full_accept_ids",
        "missing_target_side_continuous_commit_ids",
        "missing_draft_side_continuous_commit_ids",
        "real_target_eager_non_empty_count",
        "missing_buffered_proposal_unexpected_count",
    ):
        print(f"{key}={summary.get(key)}")


def synthetic_continuous_commit_record(side: str, *, proposal_id: int = 900000601) -> dict[str, Any]:
    seq_id = 12
    record = synthetic_commit_record(6, seq_id, side, 7, 27)
    record.update(
        {
            "enable_continuous_eager_dry_run": True,
            "enable_continuous_eager_verify_apply_dry_run": True,
            "enable_continuous_eager_commit_depth1_ready_only": True,
            "continuous_eager_dry_run_enabled": True,
            "continuous_eager_source": CONTINUOUS_SOURCE,
            "continuous_eager_execution_stage": "verify_apply_dry_run",
            "continuous_shadow_stage": "verify_apply_dry_run",
            "continuous_eager_candidate_proposal_ids": [proposal_id],
            "continuous_eager_candidate_seq_ids": [seq_id],
            "continuous_eager_candidate_token_count_by_proposal_id": {str(proposal_id): 4},
            "continuous_eager_parent_proposal_id_by_proposal_id": {str(proposal_id): 6},
            "continuous_eager_parent_source_by_proposal_id": {str(proposal_id): ONE_SHOT_PARENT_SOURCE},
            "continuous_eager_chain_depth_by_proposal_id": {str(proposal_id): 1},
            "continuous_eager_root_proposal_id_by_proposal_id": {str(proposal_id): 6},
            "continuous_eager_verified_proposal_ids": [proposal_id],
            "continuous_eager_full_accept_proposal_ids": [proposal_id],
            "continuous_eager_verify_result_by_proposal_id": {str(proposal_id): "full_accept"},
            "continuous_eager_accept_len_by_proposal_id": {str(proposal_id): 4},
            "continuous_eager_apply_dry_run_executed_proposal_ids": [proposal_id],
            "continuous_eager_apply_action_by_proposal_id": {str(proposal_id): FULL_ACCEPT_ACTION},
            "continuous_eager_apply_rollback_ok_by_proposal_id": {str(proposal_id): True},
            "continuous_eager_apply_mutation_detected_by_proposal_id": {str(proposal_id): False},
            "continuous_eager_apply_checkpoint_failed_by_proposal_id": {str(proposal_id): False},
            "continuous_eager_result_transfer_validated_proposal_ids": [proposal_id],
            "continuous_eager_sync_apply_executed_proposal_ids": [proposal_id],
            "continuous_eager_sync_apply_action_match_by_proposal_id": {str(proposal_id): True},
            "continuous_eager_sync_apply_result_match_by_proposal_id": {str(proposal_id): True},
            "continuous_eager_sync_apply_accept_len_match_by_proposal_id": {str(proposal_id): True},
            "continuous_eager_sync_apply_rollback_ok_by_proposal_id": {str(proposal_id): True},
            "continuous_eager_sync_apply_mutation_detected_by_proposal_id": {str(proposal_id): False},
            "continuous_eager_sync_apply_checkpoint_failed_by_proposal_id": {str(proposal_id): False},
            "continuous_eager_commit_ready_shadow_proposal_ids": [proposal_id],
            "continuous_eager_commit_ready_shadow_seq_ids": [seq_id],
            "continuous_eager_commit_ready_shadow_token_count_by_proposal_id": {str(proposal_id): 4},
            "continuous_eager_commit_enabled": True,
            "continuous_eager_commit_source": CONTINUOUS_COMMIT_SOURCE,
            "continuous_eager_commit_side": side,
            "continuous_eager_commit_step_id": 7,
            "continuous_eager_commit_plan_id": 27,
            "continuous_eager_commit_candidate_proposal_ids": [proposal_id],
            "continuous_eager_commit_candidate_seq_ids": [seq_id],
            "continuous_eager_commit_ready_source_proposal_ids": [proposal_id],
            "continuous_eager_real_committed_proposal_ids": [proposal_id],
            "continuous_eager_real_committed_seq_ids": [seq_id],
            "continuous_eager_real_committed_token_count_by_proposal_id": {str(proposal_id): 4},
            "continuous_eager_real_committed_accept_len_by_proposal_id": {str(proposal_id): 4},
            "continuous_eager_real_commit_action_by_proposal_id": {str(proposal_id): FULL_ACCEPT_ACTION},
            "continuous_eager_real_commit_verify_result_by_proposal_id": {str(proposal_id): "full_accept"},
            "continuous_eager_real_commit_precondition_ok_by_proposal_id": {str(proposal_id): True},
            "continuous_eager_real_commit_precondition_failed_by_proposal_id": {str(proposal_id): False},
            "continuous_eager_target_seq_len_before_by_seq_id": {str(seq_id): 24},
            "continuous_eager_target_seq_len_after_by_seq_id": {str(seq_id): 28},
            "continuous_eager_draft_seq_len_before_by_seq_id": {str(seq_id): 24},
            "continuous_eager_draft_seq_len_after_by_seq_id": {str(seq_id): 28},
            "continuous_eager_target_draft_len_match_by_seq_id": {str(seq_id): True},
            "continuous_eager_target_draft_token_match_by_seq_id": {str(seq_id): True},
            "continuous_eager_tokens_verified": 4,
            "continuous_eager_tokens_accepted": 4,
            "continuous_eager_tokens_committed": 4,
            "continuous_eager_tokens_rejected": 0,
            "continuous_eager_tokens_invalidated": 0,
            "continuous_eager_real_commit_count": 1,
            "continuous_eager_real_committed_proposal_count": 1,
            "continuous_eager_real_committed_token_count": 4,
            "continuous_depth2_real_commit_count": 0,
        }
    )
    return record


def retokenize_synthetic_record(record: dict[str, Any], *, one_shot_tokens: int, continuous_tokens: int) -> None:
    record["normal_gamma"] = one_shot_tokens
    one_shot_id = int(record["eager_committed_proposal_ids"][0])
    one_shot_seq = str(record["eager_committed_seq_ids"][0])
    record["eager_committed_token_count_by_proposal_id"] = {str(one_shot_id): one_shot_tokens}
    record["eager_committed_accept_len_by_proposal_id"] = {str(one_shot_id): one_shot_tokens}
    record["eager_tokens_committed"] = one_shot_tokens
    record["eager_tokens_committed_full_accept"] = one_shot_tokens
    record["eager_tokens_verified"] = one_shot_tokens
    record["eager_tokens_accepted"] = one_shot_tokens
    one_shot_target_before = int(record["eager_commit_target_seq_len_before_by_seq_id"][one_shot_seq])
    one_shot_draft_before = int(record["eager_commit_draft_seq_len_before_by_seq_id"][one_shot_seq])
    record["eager_commit_target_seq_len_after_by_seq_id"] = {one_shot_seq: one_shot_target_before + one_shot_tokens}
    record["eager_commit_draft_seq_len_after_by_seq_id"] = {one_shot_seq: one_shot_draft_before + one_shot_tokens}

    continuous_id = int(record["continuous_eager_real_committed_proposal_ids"][0])
    continuous_seq = str(record["continuous_eager_real_committed_seq_ids"][0])
    record["continuous_eager_candidate_token_count_by_proposal_id"] = {str(continuous_id): continuous_tokens}
    record["continuous_eager_accept_len_by_proposal_id"] = {str(continuous_id): continuous_tokens}
    record["continuous_eager_commit_ready_shadow_token_count_by_proposal_id"] = {str(continuous_id): continuous_tokens}
    record["continuous_eager_real_committed_token_count_by_proposal_id"] = {str(continuous_id): continuous_tokens}
    record["continuous_eager_real_committed_accept_len_by_proposal_id"] = {str(continuous_id): continuous_tokens}
    record["continuous_eager_tokens_verified"] = continuous_tokens
    record["continuous_eager_tokens_accepted"] = continuous_tokens
    record["continuous_eager_tokens_committed"] = continuous_tokens
    record["continuous_eager_real_committed_token_count"] = continuous_tokens
    continuous_target_before = int(record["continuous_eager_target_seq_len_before_by_seq_id"][continuous_seq])
    continuous_draft_before = int(record["continuous_eager_draft_seq_len_before_by_seq_id"][continuous_seq])
    record["continuous_eager_target_seq_len_after_by_seq_id"] = {
        continuous_seq: continuous_target_before + continuous_tokens
    }
    record["continuous_eager_draft_seq_len_after_by_seq_id"] = {
        continuous_seq: continuous_draft_before + continuous_tokens
    }


def add_higher_depth_commit_accounting(record: dict[str, Any], *, depth2_tokens: int, depth3_tokens: int) -> None:
    side = str(record["continuous_eager_commit_side"])
    seq_id = int(record["continuous_eager_real_committed_seq_ids"][0])
    depth2_id = 900000602
    depth3_id = 900000603
    record.update(
        {
            "enable_rolling_continuous_depth2_commit_ready_only": True,
            "rolling_depth2_commit_enabled": True,
            "rolling_depth2_commit_side": side,
            "rolling_depth2_commit_plan_id": 37,
            "rolling_depth2_commit_step_id": 17,
            "rolling_depth2_real_committed_proposal_ids": [depth2_id],
            "rolling_depth2_real_committed_seq_ids": [seq_id],
            "rolling_depth2_real_committed_token_count_by_proposal_id": {str(depth2_id): depth2_tokens},
            "rolling_depth2_tokens_verified": depth2_tokens,
            "rolling_depth2_tokens_accepted": depth2_tokens,
            "rolling_depth2_tokens_rejected": 0,
            "rolling_depth2_tokens_invalidated": 0,
            "enable_rolling_continuous_depth3_commit_ready_only": True,
            "rolling_depth3_commit_enabled": True,
            "rolling_depth3_commit_side": side,
            "rolling_depth3_commit_plan_id": 38,
            "rolling_depth3_commit_step_id": 18,
            "rolling_depth3_real_committed_proposal_ids": [depth3_id],
            "rolling_depth3_real_committed_seq_ids": [seq_id],
            "rolling_depth3_real_committed_token_count_by_proposal_id": {str(depth3_id): depth3_tokens},
            "rolling_depth3_tokens_verified": depth3_tokens,
            "rolling_depth3_tokens_accepted": depth3_tokens,
            "rolling_depth3_tokens_rejected": 0,
            "rolling_depth3_tokens_invalidated": 0,
            "rolling_depth3_real_commit_count": 1,
        }
    )


def add_depth4_commit_accounting(record: dict[str, Any], *, depth4_tokens: int, enabled: bool = True) -> None:
    side = str(record.get("continuous_eager_commit_side") or record.get("continuous_eager_commit_side", ""))
    seq_id = int(record.get("continuous_eager_real_committed_seq_ids", [0])[0]) if record.get(
        "continuous_eager_real_committed_seq_ids"
    ) else 12
    depth4_id = 900000604
    has_commit = enabled and depth4_tokens > 0
    record["enable_rolling_continuous_depth4_commit_ready_only"] = enabled
    record["rolling_depth4_commit_enabled"] = enabled
    record["rolling_depth4_commit_side"] = side
    record["rolling_depth4_commit_plan_id"] = 39
    record["rolling_depth4_commit_step_id"] = 19
    record["rolling_depth4_real_committed_proposal_ids"] = [depth4_id] if has_commit else []
    record["rolling_depth4_real_committed_seq_ids"] = [seq_id] if has_commit else []
    record["rolling_depth4_real_committed_token_count_by_proposal_id"] = (
        {str(depth4_id): depth4_tokens} if has_commit else {}
    )
    record["rolling_depth4_tokens_verified"] = depth4_tokens
    record["rolling_depth4_tokens_accepted"] = depth4_tokens
    record["rolling_depth4_tokens_rejected"] = 0
    record["rolling_depth4_tokens_invalidated"] = 0
    record["rolling_depth4_real_committed_token_count"] = depth4_tokens if has_commit else 0
    record["rolling_depth4_real_commit_count"] = 1 if has_commit else 0
    record["rolling_depth_gt4_real_commit_count"] = 0


def add_partial_recovery_accounting(
    record: dict[str, Any],
    *,
    enabled: bool = True,
    accepted_tokens: int = 1,
    revised_tokens: int = 1,
    total_tokens: int | None = None,
    len_match: bool = True,
    token_match: bool = True,
    descendant_committed_after_partial_count: int = 0,
) -> None:
    proposal_id = 900000605
    seq_id = int(record.get("continuous_eager_real_committed_seq_ids", [12])[0]) if record.get(
        "continuous_eager_real_committed_seq_ids"
    ) else 12
    total = accepted_tokens + revised_tokens if total_tokens is None else int(total_tokens)
    record["partial_prefix_recovery_enabled"] = bool(enabled)
    record["enable_rolling_continuous_partial_prefix_recovery"] = bool(enabled)
    record["partial_prefix_recovery_attempt_count"] = 1
    record["partial_prefix_recovery_success_count"] = 1
    record["partial_prefix_recovery_skip_reason_counts"] = {}
    record["partial_prefix_recovered_proposal_ids"] = [proposal_id]
    record["partial_prefix_recovered_seq_ids"] = [seq_id]
    record["partial_prefix_recovered_depth_by_proposal_id"] = {str(proposal_id): 1}
    record["partial_prefix_accepted_len_by_proposal_id"] = {str(proposal_id): accepted_tokens}
    record["partial_prefix_reject_index_by_proposal_id"] = {str(proposal_id): accepted_tokens}
    record["partial_prefix_revised_token_count_by_proposal_id"] = {str(proposal_id): revised_tokens}
    record["partial_prefix_committed_token_count_by_proposal_id"] = {str(proposal_id): total}
    record["partial_prefix_recovery_frontier_before_by_seq_id"] = {str(seq_id): 44}
    record["partial_prefix_recovery_frontier_after_by_seq_id"] = {str(seq_id): 44 + total}
    record["partial_prefix_descendant_cascade_discard_count_by_proposal_id"] = {str(proposal_id): 0}
    record["partial_prefix_recovery_normal_release_seq_ids"] = [seq_id]
    record["partial_recovery_target_seq_len_before_by_seq_id"] = {str(seq_id): 44}
    record["partial_recovery_target_seq_len_after_by_seq_id"] = {str(seq_id): 44 + total}
    record["partial_recovery_draft_seq_len_before_by_seq_id"] = {str(seq_id): 44}
    record["partial_recovery_draft_seq_len_after_by_seq_id"] = {str(seq_id): 44 + total}
    record["partial_recovery_target_draft_len_match_by_seq_id"] = {str(seq_id): bool(len_match)}
    record["partial_recovery_target_draft_token_match_by_seq_id"] = {str(seq_id): bool(token_match)}
    record["partial_recovery_cascade_discarded_descendant_proposal_ids"] = []
    record["partial_recovery_cascade_discarded_descendant_depth_by_proposal_id"] = {}
    record["partial_recovery_cascade_discarded_descendant_reason_by_proposal_id"] = {}
    record["descendant_committed_after_partial_count"] = int(descendant_committed_after_partial_count)


def run_synthetic_tests() -> None:
    valid = [
        synthetic_continuous_commit_record("target"),
        synthetic_continuous_commit_record("draft"),
    ]
    errors, summary = validate_records(valid)
    assert not errors, f"valid continuous depth-1 commit synthetic failed: {errors}"
    assert summary["continuous_real_committed_token_count"] == 4
    assert summary["combined_real_committed_token_count"] == 8

    higher_depth_valid = deepcopy(valid)
    for record in higher_depth_valid:
        retokenize_synthetic_record(record, one_shot_tokens=12, continuous_tokens=8)
        add_higher_depth_commit_accounting(record, depth2_tokens=8, depth3_tokens=8)
    errors, summary = validate_records(higher_depth_valid)
    assert not errors, f"depth1 checker rejected legal higher-depth combined accounting: {errors}"
    assert summary["one_shot_committed_token_count"] == 12
    assert summary["continuous_real_committed_token_count"] == 8
    assert summary["rolling_depth2_real_committed_token_count"] == 8
    assert summary["rolling_depth3_real_committed_token_count"] == 8
    assert summary["combined_real_committed_token_count"] == 36

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_chain_depth_by_proposal_id"] = {"900000601": 2}
    errors, _summary = validate_records(invalid)
    assert any("non-depth-1" in error for error in errors), "missed depth-2 continuous commit"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_real_commit_verify_result_by_proposal_id"] = {"900000601": "partial_accept"}
    errors, _summary = validate_records(invalid)
    assert any("non-full-accept" in error for error in errors), "missed committed partial continuous proposal"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_real_committed_proposal_ids"] = []
    errors, _summary = validate_records(invalid)
    assert any("missing target-side" in error for error in errors), "missed missing target-side continuous commit"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_tokens_verified"] = 0
    errors, _summary = validate_records(invalid)
    assert any("target continuous verified" in error for error in errors), "missed continuous counter mismatch"

    # —— depth4 synthetic tests ——

    # A. depth4 disabled, combined=36, pass
    depth4_disabled = deepcopy(higher_depth_valid)
    for record in depth4_disabled:
        add_depth4_commit_accounting(record, depth4_tokens=0, enabled=False)
    errors, summary = validate_records(depth4_disabled)
    assert not errors, f"depth4 disabled synthetic failed: {errors}"
    assert summary["one_shot_committed_token_count"] == 12
    assert summary["continuous_real_committed_token_count"] == 8
    assert summary["rolling_depth4_real_committed_token_count"] == 0
    assert summary["depth4_commit_enabled"] is False
    assert summary["combined_real_committed_token_count"] == 36
    assert summary["expected_combined_real_committed_token_count"] == 36

    # B. legal depth4 commit, combined=44, pass
    depth4_legal = deepcopy(higher_depth_valid)
    for record in depth4_legal:
        add_depth4_commit_accounting(record, depth4_tokens=8, enabled=True)
    errors, summary = validate_records(depth4_legal)
    assert not errors, f"legal depth4 commit synthetic failed: {errors}"
    assert summary["one_shot_committed_token_count"] == 12
    assert summary["continuous_real_committed_token_count"] == 8
    assert summary["rolling_depth4_real_committed_token_count"] == 8
    assert summary["rolling_depth4_real_commit_count"] == 2  # both target and draft records
    assert summary["depth4_commit_enabled"] is True
    assert summary["rolling_depth_gt4_real_commit_count"] == 0
    assert summary["combined_real_committed_token_count"] == 44
    assert summary["expected_combined_real_committed_token_count"] == 44

    # C. illegal depth4 commit (tokens present, flag disabled), fail
    depth4_illegal = deepcopy(higher_depth_valid)
    for i, record in enumerate(depth4_illegal):
        # Inject depth4 committed evidence WITHOUT enabling the flag
        record["enable_rolling_continuous_depth4_commit_ready_only"] = False
        record["rolling_depth4_commit_enabled"] = False
        record["rolling_depth4_real_committed_proposal_ids"] = [900000604]
        record["rolling_depth4_real_committed_seq_ids"] = [12]
        record["rolling_depth4_real_committed_token_count_by_proposal_id"] = {"900000604": 8}
        record["rolling_depth4_real_committed_token_count"] = 8
        record["rolling_depth4_tokens_verified"] = 8
        record["rolling_depth4_tokens_accepted"] = 8
        record["rolling_depth4_tokens_rejected"] = 0
        record["rolling_depth4_tokens_invalidated"] = 0
        record["rolling_depth4_real_commit_count"] = 1 if i == 0 else 0  # count once only
        record["rolling_depth4_commit_side"] = record.get("continuous_eager_commit_side", "target")
        record["rolling_depth4_commit_plan_id"] = 39
        record["rolling_depth4_commit_step_id"] = 19
        record["rolling_depth_gt4_real_commit_count"] = 0
    errors, _summary = validate_records(depth4_illegal)
    assert errors, "illegal depth4 commit should fail"
    assert any("depth4" in error.lower() for error in errors), f"wrong error for illegal depth4: {errors}"

    # Verify expected_combined matches actual combined for legal depth4
    assert summary["expected_combined_real_committed_token_count"] == 44
    assert summary["combined_real_committed_token_count"] == 44

    # D. legal partial-prefix recovery, combined=44+2, accepted excludes revised token.
    partial_legal = deepcopy(depth4_legal)
    for record in partial_legal:
        add_partial_recovery_accounting(record, enabled=True, accepted_tokens=1, revised_tokens=1)
    errors, summary = validate_records(partial_legal)
    assert not errors, f"legal partial recovery synthetic failed: {errors}"
    assert summary["partial_prefix_recovery_enabled"] is True
    assert summary["partial_prefix_recovery_success_count"] == 1
    assert summary["partial_prefix_accepted_token_count"] == 1
    assert summary["partial_prefix_revised_token_count"] == 1
    assert summary["partial_prefix_total_recovered_token_count"] == 2
    assert summary["combined_real_committed_token_count"] == 46
    assert summary["expected_combined_real_committed_token_count"] == 46
    assert summary["combined_actual_accepted_token_increment_sum"] == 45
    assert summary["combined_actual_revised_token_increment_sum"] == 1
    assert summary["combined_actual_output_token_increment_sum"] == 46

    # E. partial tokens while flag disabled, fail.
    partial_disabled = deepcopy(depth4_legal)
    for record in partial_disabled:
        add_partial_recovery_accounting(record, enabled=False, accepted_tokens=1, revised_tokens=1)
    errors, _summary = validate_records(partial_disabled)
    assert errors, "partial tokens while disabled should fail"
    assert any("partial-prefix recovery is disabled" in error for error in errors), errors

    # F/G. combined accounting mismatch is still caught if a summary under- or over-counts partial tokens.
    partial_accounting = aggregate_performance_accounting(partial_legal, {})
    bad_under = dict(partial_accounting)
    bad_under["combined_real_committed_token_count"] = 44
    errors = combined_accounting_errors(
        bad_under,
        one_shot_tokens=12,
        continuous_tokens=8,
        legal_partial_prefix_total_recovered_token_count=2,
    )
    assert errors, "combined missing partial tokens should fail"
    bad_over = dict(partial_accounting)
    bad_over["combined_real_committed_token_count"] = 48
    errors = combined_accounting_errors(
        bad_over,
        one_shot_tokens=12,
        continuous_tokens=8,
        legal_partial_prefix_total_recovered_token_count=2,
    )
    assert errors, "combined double-counting partial tokens should fail"

    # H. revised token accounting mismatch, fail.
    bad_revised = dict(partial_accounting)
    bad_revised["combined_actual_revised_token_increment_sum"] = 0
    errors, _legal_partial = partial_recovery_accounting_errors(bad_revised)
    assert any("revised increment" in error for error in errors), errors

    # I. descendant commit after partial recovery, fail.
    partial_descendant = deepcopy(depth4_legal)
    for i, record in enumerate(partial_descendant):
        add_partial_recovery_accounting(
            record,
            enabled=True,
            accepted_tokens=1,
            revised_tokens=1,
            descendant_committed_after_partial_count=1 if i == 0 else 0,
        )
    errors, _summary = validate_records(partial_descendant)
    assert any("descendant committed after partial recovery" in error for error in errors), errors

    # J. partial target/draft mismatch, fail.
    partial_mismatch = deepcopy(depth4_legal)
    for record in partial_mismatch:
        add_partial_recovery_accounting(record, enabled=True, accepted_tokens=1, revised_tokens=1, len_match=False)
    errors, _summary = validate_records(partial_mismatch)
    assert any("target/draft length mismatch" in error for error in errors), errors

    # E. depth_gt4 > 0, fail
    depth4_gt4 = deepcopy(higher_depth_valid)
    for record in depth4_gt4:
        add_depth4_commit_accounting(record, depth4_tokens=8, enabled=True)
        record["rolling_depth_gt4_real_commit_count"] = 1
    errors, _summary = validate_records(depth4_gt4)
    assert errors, "depth_gt4 should fail"
    assert any("depth>4" in error for error in errors), f"wrong error for depth_gt4: {errors}"

    print("Synthetic continuous eager depth-1 commit checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-7c real continuous depth-1 eager commit traces.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0
    errors, summary = validate_records(load_trace(args.trace))
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
