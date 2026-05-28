#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROLLING_COMMIT_SOURCE = "rolling_depth2_ready_only"
ROLLING_ACTION = "append_full_accept_real_commit"


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import (  # noqa: E402
    as_int_list,
    as_int_set,
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


def as_bool_map(value: Any) -> dict[int, bool]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, bool] = {}
    for key, item in value.items():
        try:
            result[int(key)] = bool(item)
        except Exception:
            continue
    return result


def commit_active(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("rolling_depth2_commit_enabled", False))
        or bool(record.get("enable_rolling_continuous_depth2_commit_ready_only", False))
        or bool(as_int_set(record.get("rolling_depth2_commit_candidate_proposal_ids")))
        or bool(as_int_set(record.get("rolling_depth2_real_committed_proposal_ids")))
        or bool(as_int_set(record.get("rolling_depth2_real_commit_skipped_proposal_ids")))
        or int_value(record.get("rolling_depth2_tokens_committed"), 0) > 0
    )


def step_plan_key(record: dict[str, Any]) -> tuple[int, int]:
    plan_id = int_value(record.get("rolling_depth2_commit_plan_id"), int_value(record.get("plan_id"), -1))
    step_id = int_value(record.get("rolling_depth2_commit_step_id"), int_value(record.get("step_id"), -1))
    return plan_id, step_id


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    one_shot_errors, _one_shot_summary = validate_one_shot_records(records)
    errors.extend(f"one-shot commit checker: {error}" for error in one_shot_errors)
    accounting = aggregate_performance_accounting(records, {})

    ready_ids: set[int] = set()
    invalidated_ids: set[int] = set()
    cascade_ids: set[int] = set()
    parent_full_ids: set[int] = set()
    parent_by_id: dict[int, int] = {}
    depth_by_id: dict[int, int] = {}
    root_by_id: dict[int, int] = {}
    status_by_id: dict[int, str] = {}
    status_reason_by_id: dict[int, str] = {}

    for record in records:
        if not is_dual_record(record):
            continue
        ready_ids.update(as_int_set(record.get("rolling_child_ready_after_parent_full_accept_proposal_ids")))
        invalidated_ids.update(as_int_set(record.get("rolling_child_invalidated_proposal_ids")))
        cascade_ids.update(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("rolling_parent_full_accept_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("continuous_eager_real_committed_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
        parent_by_id.update(as_int_map(record.get("rolling_chain_parent_by_proposal_id")))
        parent_by_id.update(as_int_map(record.get("rolling_depth2_real_commit_parent_by_proposal_id")))
        depth_by_id.update(as_int_map(record.get("rolling_chain_depth_by_proposal_id")))
        depth_by_id.update(as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id")))
        root_by_id.update(as_int_map(record.get("rolling_chain_root_by_proposal_id")))
        root_by_id.update(as_int_map(record.get("rolling_depth2_real_commit_root_by_proposal_id")))
        status_by_id.update(as_str_map(record.get("rolling_chain_status_by_proposal_id")))
        status_reason_by_id.update(as_str_map(record.get("rolling_chain_status_reason_by_proposal_id")))

    records_with_enabled = 0
    active_records = 0
    committed_ids_seen: set[int] = set()
    committed_sides_by_id: dict[int, set[str]] = defaultdict(set)
    committed_steps_by_side: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    repeated_commit_ids: set[int] = set()
    duplicate_seq_depth_events: set[tuple[str, int, int, int, int]] = set()
    seen_seq_depth_events: set[tuple[str, int, int, int, int]] = set()
    skipped_ids_seen: set[int] = set()
    skip_reason_by_id: dict[int, str] = {}
    committed_without_ready: set[int] = set()
    committed_without_parent: set[int] = set()
    committed_invalidated: set[int] = set()
    committed_cascade: set[int] = set()
    committed_depth_bad: set[int] = set()
    committed_non_full_accept: set[int] = set()
    depth3_real_commit_count = 0
    depth_gt2_real_commit_count = 0
    missing_unexpected_count = 0
    normal_lane_conflict_count = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue
        if as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")):
            missing_unexpected_count += len(as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")))
            errors.append(f"record[{idx}] unexpected missing buffered proposal")
        if as_int_set(record.get("rolling_normal_lane_conflict_seq_ids")):
            normal_lane_conflict_count += len(as_int_set(record.get("rolling_normal_lane_conflict_seq_ids")))
            errors.append(f"record[{idx}] rolling normal lane conflict present")
        if int_value(record.get("rolling_depth3_real_commit_count"), 0):
            depth3_real_commit_count += int_value(record.get("rolling_depth3_real_commit_count"), 0)
            errors.append(f"record[{idx}] rolling depth-3 real commit count must be zero")
        if int_value(record.get("rolling_depth_gt2_real_commit_count"), 0):
            depth_gt2_real_commit_count += int_value(record.get("rolling_depth_gt2_real_commit_count"), 0)
            errors.append(f"record[{idx}] rolling depth>2 real commit count must be zero")

        enabled = bool(record.get("enable_rolling_continuous_depth2_commit_ready_only", False))
        if enabled:
            records_with_enabled += 1
        if not commit_active(record):
            continue
        active_records += 1
        if not enabled:
            errors.append(f"record[{idx}] rolling depth-2 commit active while flag disabled")
        if record.get("rolling_depth2_commit_source") not in {None, ROLLING_COMMIT_SOURCE}:
            errors.append(f"record[{idx}] bad rolling depth-2 commit source {record.get('rolling_depth2_commit_source')!r}")

        side = str(record.get("rolling_depth2_commit_side") or "")
        plan_id, step_id = step_plan_key(record)
        committed_ids = as_int_set(record.get("rolling_depth2_real_committed_proposal_ids"))
        committed_seq_ids = as_int_list(record.get("rolling_depth2_real_committed_seq_ids"))
        pid_to_seq = dict(zip(as_int_list(record.get("rolling_depth2_real_committed_proposal_ids")), committed_seq_ids))
        token_count_by_id = as_int_map(record.get("rolling_depth2_real_committed_token_count_by_proposal_id"))
        accept_len_by_id = as_int_map(record.get("rolling_depth2_real_committed_accept_len_by_proposal_id"))
        action_by_id = as_str_map(record.get("rolling_depth2_real_commit_action_by_proposal_id"))
        result_by_id = as_str_map(record.get("rolling_depth2_real_commit_verify_result_by_proposal_id"))
        parent_commit_by_id = as_int_map(record.get("rolling_depth2_real_commit_parent_by_proposal_id"))
        depth_commit_by_id = as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id"))
        precondition_ok_by_id = as_bool_map(record.get("rolling_depth2_commit_precondition_ok_by_proposal_id"))
        precondition_failed_by_id = as_bool_map(record.get("rolling_depth2_commit_precondition_failed_by_proposal_id"))
        skip_reason_map = as_str_map(record.get("rolling_depth2_real_commit_skip_reason_by_proposal_id"))
        skipped = as_int_set(record.get("rolling_depth2_real_commit_skipped_proposal_ids"))
        candidate_ids = as_int_set(record.get("rolling_depth2_commit_candidate_proposal_ids"))
        skipped_ids_seen.update(skipped)
        for proposal_id, reason in skip_reason_map.items():
            skip_reason_by_id.setdefault(proposal_id, reason)
        if candidate_ids - committed_ids - skipped:
            errors.append(f"record[{idx}] rolling depth-2 candidates missing commit/skip classification")

        before_map = as_int_map(record.get("rolling_depth2_target_seq_len_before_by_seq_id"))
        after_map = as_int_map(record.get("rolling_depth2_target_seq_len_after_by_seq_id"))
        len_match_map = as_bool_map(record.get("rolling_depth2_target_draft_len_match_by_seq_id"))
        token_match_map = as_bool_map(record.get("rolling_depth2_target_draft_token_match_by_seq_id"))

        for proposal_id in committed_ids:
            seq_id = int(pid_to_seq.get(proposal_id, -1))
            token_count = int(token_count_by_id.get(proposal_id, 0))
            accept_len = int(accept_len_by_id.get(proposal_id, -1))
            depth = int(depth_commit_by_id.get(proposal_id, depth_by_id.get(proposal_id, 0)))
            parent_id = int(parent_commit_by_id.get(proposal_id, parent_by_id.get(proposal_id, -1)))
            if proposal_id in committed_ids_seen and side in committed_sides_by_id[proposal_id]:
                repeated_commit_ids.add(proposal_id)
            committed_ids_seen.add(proposal_id)
            committed_sides_by_id[proposal_id].add(side)
            event = (side, plan_id, step_id, seq_id, depth)
            if event in seen_seq_depth_events:
                duplicate_seq_depth_events.add(event)
            seen_seq_depth_events.add(event)
            committed_steps_by_side[(side, proposal_id)].add((plan_id, step_id))

            if proposal_id not in ready_ids:
                committed_without_ready.add(proposal_id)
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} was not ready shadow")
            if parent_id not in parent_full_ids:
                committed_without_parent.add(proposal_id)
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} lacks full-accept parent")
            if proposal_id in invalidated_ids:
                committed_invalidated.add(proposal_id)
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} was invalidated")
            if proposal_id in cascade_ids:
                committed_cascade.add(proposal_id)
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} was cascade-discarded")
            if depth != 2:
                committed_depth_bad.add(proposal_id)
                errors.append(f"record[{idx}] committed rolling proposal {proposal_id} depth is {depth}, expected 2")
            if result_by_id.get(proposal_id) != "full_accept":
                committed_non_full_accept.add(proposal_id)
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} is not full_accept")
            if action_by_id.get(proposal_id) != ROLLING_ACTION:
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} has bad action")
            if accept_len != token_count or token_count <= 0:
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} token/accept mismatch")
            if precondition_ok_by_id.get(proposal_id) is not True:
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} missing precondition ok")
            if precondition_failed_by_id.get(proposal_id):
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} has failed precondition")
            if int(after_map.get(seq_id, before_map.get(seq_id, -1))) - int(before_map.get(seq_id, -1)) != token_count:
                errors.append(f"record[{idx}] committed rolling depth-2 seq {seq_id} length delta mismatch")
            if len_match_map.get(seq_id) is not True:
                errors.append(f"record[{idx}] committed rolling depth-2 seq {seq_id} target/draft len mismatch")
            if token_match_map and token_match_map.get(seq_id) is not True:
                errors.append(f"record[{idx}] committed rolling depth-2 seq {seq_id} token mismatch")
            if status_by_id.get(proposal_id) not in {None, "CHILD_READY_AFTER_PARENT_FULL_ACCEPT"}:
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} has bad chain status")
            if status_reason_by_id.get(proposal_id) == "parent_verify_pending":
                errors.append(f"record[{idx}] committed rolling depth-2 child {proposal_id} still parent-pending")

        for proposal_id, reason in skip_reason_map.items():
            if proposal_id in ready_ids and not reason:
                errors.append(f"record[{idx}] skipped ready rolling child {proposal_id} missing reason")

    for (side, proposal_id), steps in committed_steps_by_side.items():
        if len(steps) > 1:
            repeated_commit_ids.add(proposal_id)
            errors.append(f"rolling depth-2 proposal {proposal_id} committed multiple times on {side}")
    for proposal_id, sides in committed_sides_by_id.items():
        if "target" not in sides:
            errors.append(f"rolling depth-2 proposal {proposal_id} missing target-side commit record")
        if "draft" not in sides:
            errors.append(f"rolling depth-2 proposal {proposal_id} missing draft-side commit record")
    if duplicate_seq_depth_events:
        errors.append(f"duplicate rolling seq/depth commit events: {sorted(duplicate_seq_depth_events)}")

    rolling_tokens = int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
    if rolling_tokens:
        if rolling_tokens != int_value(accounting.get("rolling_depth2_target_actual_verified_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal target verified increment sum")
        if rolling_tokens != int_value(accounting.get("rolling_depth2_target_actual_accepted_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal target accepted increment sum")
        if rolling_tokens != int_value(accounting.get("rolling_depth2_draft_actual_verified_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal draft verified increment sum")
        if rolling_tokens != int_value(accounting.get("rolling_depth2_draft_actual_accepted_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal draft accepted increment sum")
    for field in (
        "rolling_depth2_target_actual_rejected_token_increment_sum",
        "rolling_depth2_target_actual_invalidated_token_increment_sum",
        "rolling_depth2_draft_actual_rejected_token_increment_sum",
        "rolling_depth2_draft_actual_invalidated_token_increment_sum",
    ):
        if int_value(accounting.get(field), 0) != 0:
            errors.append(f"{field} must remain zero")
    combined_expected = (
        int_value(accounting.get("eager_committed_token_count"), 0)
        + int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
        + rolling_tokens
    )
    if int_value(accounting.get("combined_real_committed_token_count"), 0) != combined_expected:
        errors.append("combined real committed token count does not include one-shot + depth1 + rolling depth2")

    summary = {
        "total_trace_records": len(records),
        "records_with_rolling_depth2_commit_enabled": records_with_enabled,
        "rolling_depth2_commit_active_records": active_records,
        "rolling_depth2_real_committed_proposal_count": len(committed_ids_seen),
        "rolling_depth2_real_committed_token_count": rolling_tokens,
        "rolling_depth2_target_actual_verified_token_increment_sum": accounting.get(
            "rolling_depth2_target_actual_verified_token_increment_sum", 0
        ),
        "rolling_depth2_draft_actual_verified_token_increment_sum": accounting.get(
            "rolling_depth2_draft_actual_verified_token_increment_sum", 0
        ),
        "combined_real_committed_token_count": accounting.get("combined_real_committed_token_count", 0),
        "rolling_depth2_commit_skip_reason_counts": dict(Counter(skip_reason_by_id.values())),
        "rolling_depth2_repeated_commit_proposal_ids": sorted(repeated_commit_ids),
        "rolling_depth2_duplicate_seq_depth_event_count": len(duplicate_seq_depth_events),
        "rolling_depth2_committed_without_ready_shadow_ids": sorted(committed_without_ready),
        "rolling_depth2_committed_without_parent_full_accept_ids": sorted(committed_without_parent),
        "rolling_depth2_committed_invalidated_child_ids": sorted(committed_invalidated),
        "rolling_depth2_committed_cascade_discarded_child_ids": sorted(committed_cascade),
        "rolling_depth2_committed_non_full_accept_ids": sorted(committed_non_full_accept),
        "rolling_depth3_real_commit_count": depth3_real_commit_count,
        "rolling_depth_gt2_real_commit_count": depth_gt2_real_commit_count,
        "rolling_normal_lane_conflict_count": normal_lane_conflict_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_rolling_depth2_commit_enabled",
        "rolling_depth2_commit_active_records",
        "rolling_depth2_real_committed_proposal_count",
        "rolling_depth2_real_committed_token_count",
        "rolling_depth2_target_actual_verified_token_increment_sum",
        "rolling_depth2_draft_actual_verified_token_increment_sum",
        "combined_real_committed_token_count",
        "rolling_depth2_commit_skip_reason_counts",
        "rolling_depth2_repeated_commit_proposal_ids",
        "rolling_depth2_duplicate_seq_depth_event_count",
        "rolling_depth2_committed_without_ready_shadow_ids",
        "rolling_depth2_committed_without_parent_full_accept_ids",
        "rolling_depth2_committed_invalidated_child_ids",
        "rolling_depth2_committed_cascade_discarded_child_ids",
        "rolling_depth2_committed_non_full_accept_ids",
        "rolling_depth3_real_commit_count",
        "rolling_depth_gt2_real_commit_count",
        "rolling_normal_lane_conflict_count",
        "missing_buffered_proposal_unexpected_count",
    ):
        print(f"{key}={summary.get(key)}")


def synthetic_records() -> list[dict[str, Any]]:
    one_shot_target = synthetic_commit_record(101, 7, "target", 10, 30)
    one_shot_draft = synthetic_commit_record(101, 7, "draft", 10, 30)
    parent_id = 900000101
    child_id = 900000102
    seq_id = 7

    def rolling_record(side: str) -> dict[str, Any]:
        return {
            **deepcopy(one_shot_target if side == "target" else one_shot_draft),
            "enable_rolling_continuous_eager_dry_run": True,
            "rolling_continuous_eager_dry_run_enabled": True,
            "rolling_continuous_stage": "overlap_dry_run",
            "rolling_continuous_source": "rolling_continuous_shadow",
            "enable_rolling_continuous_depth2_commit_ready_only": True,
            "rolling_chain_parent_by_proposal_id": {str(parent_id): 101, str(child_id): parent_id},
            "rolling_chain_root_by_proposal_id": {str(parent_id): 101, str(child_id): 101},
            "rolling_chain_depth_by_proposal_id": {str(parent_id): 1, str(child_id): 2},
            "rolling_chain_status_by_proposal_id": {
                str(parent_id): "PARENT_FULL_ACCEPT",
                str(child_id): "CHILD_READY_AFTER_PARENT_FULL_ACCEPT",
            },
            "rolling_chain_status_reason_by_proposal_id": {str(child_id): "parent_full_accept"},
            "rolling_parent_full_accept_proposal_ids": [parent_id],
            "continuous_eager_real_committed_proposal_ids": [parent_id],
            "rolling_child_generated_proposal_ids": [child_id],
            "rolling_child_ready_after_parent_full_accept_proposal_ids": [child_id],
            "rolling_child_ready_shadow_proposal_count": 1,
            "rolling_child_ready_shadow_token_count": 4,
            "rolling_depth2_commit_enabled": True,
            "rolling_depth2_commit_source": ROLLING_COMMIT_SOURCE,
            "rolling_depth2_commit_side": side,
            "rolling_depth2_commit_step_id": 12,
            "rolling_depth2_commit_plan_id": 32,
            "rolling_depth2_commit_candidate_proposal_ids": [child_id],
            "rolling_depth2_commit_candidate_seq_ids": [seq_id],
            "rolling_depth2_commit_ready_source_proposal_ids": [child_id],
            "rolling_depth2_commit_parent_by_proposal_id": {str(child_id): parent_id},
            "rolling_depth2_commit_precondition_ok_by_proposal_id": {str(child_id): True},
            "rolling_depth2_commit_precondition_failed_by_proposal_id": {str(child_id): False},
            "rolling_depth2_real_committed_proposal_ids": [child_id],
            "rolling_depth2_real_committed_seq_ids": [seq_id],
            "rolling_depth2_real_committed_token_count_by_proposal_id": {str(child_id): 4},
            "rolling_depth2_real_committed_accept_len_by_proposal_id": {str(child_id): 4},
            "rolling_depth2_real_commit_action_by_proposal_id": {str(child_id): ROLLING_ACTION},
            "rolling_depth2_real_commit_verify_result_by_proposal_id": {str(child_id): "full_accept"},
            "rolling_depth2_real_commit_parent_by_proposal_id": {str(child_id): parent_id},
            "rolling_depth2_real_commit_root_by_proposal_id": {str(child_id): 101},
            "rolling_depth2_real_commit_depth_by_proposal_id": {str(child_id): 2},
            "rolling_depth2_target_seq_len_before_by_seq_id": {str(seq_id): 12},
            "rolling_depth2_target_seq_len_after_by_seq_id": {str(seq_id): 16},
            "rolling_depth2_draft_seq_len_before_by_seq_id": {str(seq_id): 12},
            "rolling_depth2_draft_seq_len_after_by_seq_id": {str(seq_id): 16},
            "rolling_depth2_target_draft_len_match_by_seq_id": {str(seq_id): True},
            "rolling_depth2_target_draft_token_match_by_seq_id": {str(seq_id): True},
            "rolling_depth2_tokens_verified": 4,
            "rolling_depth2_tokens_accepted": 4,
            "rolling_depth2_tokens_committed": 4,
            "rolling_depth2_tokens_rejected": 0,
            "rolling_depth2_tokens_invalidated": 0,
            "rolling_depth2_real_committed_proposal_count": 1,
            "rolling_depth2_real_committed_token_count": 4,
            "rolling_depth2_real_commit_count": 1,
            "rolling_depth3_real_commit_count": 0,
            "rolling_depth_gt2_real_commit_count": 0,
            "rolling_normal_lane_conflict_count": 0,
        }

    return [rolling_record("target"), rolling_record("draft")]


def run_synthetic() -> None:
    valid = synthetic_records()
    errors, summary = validate_records(valid)
    if errors:
        raise SystemExit(f"synthetic valid rolling depth2 commit failed: {errors}\nsummary={summary}")

    invalid_not_ready = deepcopy(valid)
    for record in invalid_not_ready:
        record["rolling_child_ready_after_parent_full_accept_proposal_ids"] = []
    errors, _summary = validate_records(invalid_not_ready)
    if not errors:
        raise SystemExit("synthetic committed but not ready shadow should fail")

    invalid_parent = deepcopy(valid)
    for record in invalid_parent:
        record["rolling_parent_full_accept_proposal_ids"] = []
        record["continuous_eager_real_committed_proposal_ids"] = []
    errors, _summary = validate_records(invalid_parent)
    if not errors:
        raise SystemExit("synthetic committed without parent full accept should fail")

    invalid_depth = deepcopy(valid)
    for record in invalid_depth:
        record["rolling_chain_depth_by_proposal_id"]["900000102"] = 3
        record["rolling_depth2_real_commit_depth_by_proposal_id"]["900000102"] = 3
        record["rolling_depth3_real_commit_count"] = 1
        record["rolling_depth_gt2_real_commit_count"] = 1
    errors, _summary = validate_records(invalid_depth)
    if not errors:
        raise SystemExit("synthetic depth>2 commit should fail")

    print("Synthetic rolling continuous depth-2 commit checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 1H-8b rolling depth-2 ready-only real commit.")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        run_synthetic()
        return
    if not args.trace:
        raise SystemExit("trace path is required unless --synthetic is used")
    records = load_trace(Path(args.trace))
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
