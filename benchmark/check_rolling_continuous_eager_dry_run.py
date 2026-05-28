#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROLLING_SOURCE = "rolling_continuous_shadow"
ROLLING_STAGE = "overlap_dry_run"


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records", "events", "iterations", "batches"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise SystemExit(f"{path} does not contain trace records")


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def as_int_set(value: Any) -> set[int]:
    return set(as_int_list(value))


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


def as_list_map(value: Any) -> dict[int, list[int]]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, list[int]] = {}
    for key, item in value.items():
        try:
            result[int(key)] = as_int_list(item)
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


def rolling_row(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("enable_rolling_continuous_eager_dry_run", False))
        or bool(record.get("rolling_continuous_eager_dry_run_enabled", False))
        or bool(as_int_set(record.get("draft_rolling_eager_draft_proposal_ids")))
        or bool(as_int_set(record.get("target_rolling_eager_verify_proposal_ids")))
        or bool(as_int_set(record.get("rolling_child_generated_proposal_ids")))
        or bool(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_enabled = 0
    active_records = 0
    target_verify_ids_seen: set[int] = set()
    draft_child_ids_seen: set[int] = set()
    ready_child_ids_seen: set[int] = set()
    invalidated_child_ids_seen: set[int] = set()
    cascade_ids_seen: set[int] = set()
    same_seq_overlap_count = 0
    normal_lane_conflict_count = 0
    depth2_real_commit_count = 0
    depth_gt1_real_commit_count = 0
    child_verified_without_parent_count = 0
    child_committed_without_parent_count = 0
    drafted_without_parent_count = 0
    duplicate_child_count = 0
    frontier_mismatch_count = 0
    max_depth_observed = 0
    candidate_token_count = 0
    ready_token_count = 0
    reason_counter: Counter[str] = Counter()

    for idx, record in enumerate(records):
        enabled = bool(record.get("enable_rolling_continuous_eager_dry_run", False))
        if enabled:
            records_with_enabled += 1
        if not rolling_row(record):
            continue
        active_records += 1
        if not enabled:
            errors.append(f"record[{idx}] has rolling fields while rolling dry-run flag is disabled")
        if record.get("rolling_continuous_eager_dry_run_enabled") and record.get("rolling_continuous_source") != ROLLING_SOURCE:
            errors.append(f"record[{idx}] bad rolling source {record.get('rolling_continuous_source')!r}")
        if record.get("rolling_continuous_eager_dry_run_enabled") and record.get("rolling_continuous_stage") != ROLLING_STAGE:
            errors.append(f"record[{idx}] bad rolling stage {record.get('rolling_continuous_stage')!r}")
        if as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")):
            errors.append(f"record[{idx}] unexpected missing normal proposal while rolling dry-run is active")

        target_ids = as_int_list(record.get("target_rolling_eager_verify_proposal_ids"))
        target_seq_ids = as_int_list(record.get("target_rolling_eager_verify_seq_ids"))
        child_ids = as_int_list(record.get("draft_rolling_eager_draft_proposal_ids"))
        child_seq_ids = as_int_list(record.get("draft_rolling_eager_draft_seq_ids"))
        generated_ids = as_int_set(record.get("rolling_child_generated_proposal_ids"))
        ready_ids = as_int_set(record.get("rolling_child_ready_after_parent_full_accept_proposal_ids"))
        invalidated_ids = as_int_set(record.get("rolling_child_invalidated_proposal_ids"))
        cascade_ids = as_int_set(record.get("rolling_cascade_discarded_proposal_ids"))
        parent_by_id = as_int_map(record.get("rolling_chain_parent_by_proposal_id"))
        children_by_id = as_list_map(record.get("rolling_chain_children_by_proposal_id"))
        root_by_id = as_int_map(record.get("rolling_chain_root_by_proposal_id"))
        depth_by_id = as_int_map(record.get("rolling_chain_depth_by_proposal_id"))
        status_by_id = as_str_map(record.get("rolling_chain_status_by_proposal_id"))
        status_reason_by_id = as_str_map(record.get("rolling_chain_status_reason_by_proposal_id"))
        invalid_reason_by_id = as_str_map(record.get("rolling_child_invalidated_reason_by_proposal_id"))
        cascade_reason_by_id = as_str_map(record.get("rolling_cascade_discard_reason_by_proposal_id"))
        full_accept_parent_ids = as_int_set(record.get("rolling_parent_full_accept_proposal_ids"))
        partial_parent_ids = as_int_set(record.get("rolling_parent_partial_reject_proposal_ids"))
        max_depth = int_value(record.get("max_rolling_continuous_depth"), 0)
        observed_depth = max(
            int_value(record.get("rolling_max_depth_observed"), 0),
            int_value(record.get("max_rolling_continuous_depth_observed"), 0),
        )
        max_depth_observed = max(max_depth_observed, observed_depth)

        target_verify_ids_seen.update(target_ids)
        draft_child_ids_seen.update(child_ids)
        ready_child_ids_seen.update(ready_ids)
        invalidated_child_ids_seen.update(invalidated_ids)
        cascade_ids_seen.update(cascade_ids)
        same_seq_overlap_count += int_value(record.get("rolling_same_seq_overlap_count"), 0)
        normal_lane_conflict_count += int_value(record.get("rolling_normal_lane_conflict_count"), 0)
        depth2_real_commit_count += int_value(record.get("rolling_depth2_real_commit_count"), 0)
        depth_gt1_real_commit_count += int_value(record.get("rolling_depth_gt1_real_commit_count"), 0)
        child_verified_without_parent_count += int_value(
            record.get("rolling_child_verified_without_parent_full_accept_count"), 0
        )
        child_committed_without_parent_count += int_value(
            record.get("rolling_child_committed_without_parent_full_accept_count"), 0
        )
        drafted_without_parent_count += int_value(record.get("rolling_child_drafted_without_valid_parent_count"), 0)
        duplicate_child_count += int_value(record.get("rolling_duplicate_child_count"), 0)
        frontier_mismatch_count += int_value(record.get("rolling_frontier_mismatch_count"), 0)
        candidate_token_count += int_value(record.get("rolling_child_candidate_token_count"), 0)
        ready_token_count += int_value(record.get("rolling_child_ready_shadow_token_count"), 0)
        reason_counter.update(invalid_reason_by_id.values())

        if len(target_ids) != len(target_seq_ids):
            errors.append(f"record[{idx}] target rolling proposal/seq length mismatch")
        if len(child_ids) != len(child_seq_ids):
            errors.append(f"record[{idx}] draft rolling proposal/seq length mismatch")
        if set(target_ids) & set(child_ids):
            errors.append(f"record[{idx}] same rolling proposal is both target-verified and draft-drafted")
        computed_overlap = set(target_seq_ids) & set(child_seq_ids)
        if int_value(record.get("rolling_same_seq_overlap_count"), 0) != len(computed_overlap):
            errors.append(f"record[{idx}] rolling same-seq overlap count mismatch")
        if set(as_int_list(record.get("rolling_same_seq_overlap_seq_ids"))) != computed_overlap:
            errors.append(f"record[{idx}] rolling same-seq overlap seq ids mismatch")
        if as_int_set(record.get("rolling_normal_lane_conflict_seq_ids")):
            errors.append(f"record[{idx}] rolling seqs conflict with normal lanes")
        if max_depth and observed_depth > max_depth:
            errors.append(f"record[{idx}] rolling observed depth exceeds configured max")
        if as_int_set(record.get("continuous_eager_real_committed_proposal_ids")) & generated_ids:
            errors.append(f"record[{idx}] rolling child appeared in continuous real commit fields")
        if generated_ids & as_int_set(record.get("lane_exclusion_applied_proposal_ids")):
            errors.append(f"record[{idx}] rolling child entered lane exclusion")
        if generated_ids & as_int_set(record.get("target_eager_verify_proposal_ids_dry_run")):
            errors.append(f"record[{idx}] rolling child entered one-shot target takeover")
        if depth2_real_commit_count or depth_gt1_real_commit_count:
            errors.append(f"record[{idx}] rolling depth>1 real commit count must stay zero")

        seq_by_proposal: dict[int, int] = dict(zip(target_ids, target_seq_ids))
        seq_by_proposal.update(dict(zip(child_ids, child_seq_ids)))
        for child_id in generated_ids:
            parent_id = parent_by_id.get(child_id)
            if parent_id is None:
                errors.append(f"record[{idx}] rolling child {child_id} missing parent")
                continue
            if parent_id not in parent_by_id and parent_id not in target_ids:
                errors.append(f"record[{idx}] rolling child {child_id} parent {parent_id} is not in chain")
            if seq_by_proposal.get(parent_id) is not None and seq_by_proposal.get(child_id) != seq_by_proposal.get(parent_id):
                errors.append(f"record[{idx}] rolling child {child_id} seq differs from parent {parent_id}")
            child_depth = depth_by_id.get(child_id, 0)
            parent_depth = depth_by_id.get(parent_id, 0)
            if child_depth != parent_depth + 1:
                errors.append(f"record[{idx}] rolling child {child_id} depth is not parent depth + 1")
            if root_by_id.get(child_id) != root_by_id.get(parent_id):
                errors.append(f"record[{idx}] rolling child {child_id} root does not match parent")
            if child_id not in children_by_id.get(parent_id, []):
                errors.append(f"record[{idx}] rolling child {child_id} missing from parent children map")
            if not status_by_id.get(child_id):
                errors.append(f"record[{idx}] rolling child {child_id} missing status")

        if ready_ids & invalidated_ids:
            errors.append(f"record[{idx}] rolling child cannot be both ready and invalidated")
        for child_id in ready_ids:
            parent_id = parent_by_id.get(child_id, -1)
            if parent_id not in full_accept_parent_ids:
                errors.append(f"record[{idx}] rolling child {child_id} ready without full-accept parent")
        for child_id in invalidated_ids:
            if child_id not in invalid_reason_by_id:
                errors.append(f"record[{idx}] rolling invalidated child {child_id} missing reason")
            parent_id = parent_by_id.get(child_id, -1)
            if parent_id in full_accept_parent_ids and child_id in generated_ids:
                errors.append(f"record[{idx}] rolling child {child_id} invalidated despite full-accept parent")
            if parent_id in partial_parent_ids and invalid_reason_by_id.get(child_id) not in {
                "parent_partial_accept",
                "parent_rejected",
                "parent_not_full_accept",
            }:
                errors.append(f"record[{idx}] rolling child {child_id} has bad parent-failure reason")
        for child_id in cascade_ids:
            if child_id not in cascade_reason_by_id:
                errors.append(f"record[{idx}] cascade-discarded child {child_id} missing reason")
            if child_id in ready_ids:
                errors.append(f"record[{idx}] cascade-discarded child {child_id} is also ready")
        if int_value(record.get("rolling_cascade_discard_count"), 0) != len(cascade_ids):
            errors.append(f"record[{idx}] cascade discard count/list mismatch")
        for child_id, reason in status_reason_by_id.items():
            if reason:
                reason_counter[reason] += 0

    summary = {
        "total_trace_records": len(records),
        "records_with_rolling_enabled": records_with_enabled,
        "rolling_active_records": active_records,
        "target_rolling_verify_proposal_count": len(target_verify_ids_seen),
        "rolling_child_candidate_proposal_count": len(draft_child_ids_seen),
        "rolling_child_candidate_token_count": candidate_token_count,
        "rolling_child_ready_shadow_proposal_count": len(ready_child_ids_seen),
        "rolling_child_ready_shadow_token_count": ready_token_count,
        "rolling_child_invalidated_count": len(invalidated_child_ids_seen),
        "rolling_cascade_discard_count": len(cascade_ids_seen),
        "rolling_same_seq_overlap_count": same_seq_overlap_count,
        "rolling_normal_lane_conflict_count": normal_lane_conflict_count,
        "rolling_depth2_real_commit_count": depth2_real_commit_count,
        "rolling_depth_gt1_real_commit_count": depth_gt1_real_commit_count,
        "rolling_child_verified_without_parent_full_accept_count": child_verified_without_parent_count,
        "rolling_child_committed_without_parent_full_accept_count": child_committed_without_parent_count,
        "rolling_child_drafted_without_valid_parent_count": drafted_without_parent_count,
        "rolling_duplicate_child_count": duplicate_child_count,
        "rolling_frontier_mismatch_count": frontier_mismatch_count,
        "rolling_max_depth_observed": max_depth_observed,
        "rolling_drop_reason_counts": dict(sorted(reason_counter.items())),
    }
    if depth2_real_commit_count:
        errors.append("rolling_depth2_real_commit_count must be zero")
    if depth_gt1_real_commit_count:
        errors.append("rolling_depth_gt1_real_commit_count must be zero")
    if child_verified_without_parent_count:
        errors.append("rolling child verified without full-accept parent")
    if child_committed_without_parent_count:
        errors.append("rolling child committed without full-accept parent")
    if normal_lane_conflict_count:
        errors.append("rolling normal lane conflicts must be zero")
    if records_with_enabled and not draft_child_ids_seen:
        errors.append("rolling dry-run enabled but no rolling child candidates were generated")
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_rolling_enabled",
        "rolling_active_records",
        "target_rolling_verify_proposal_count",
        "rolling_child_candidate_proposal_count",
        "rolling_child_candidate_token_count",
        "rolling_child_ready_shadow_proposal_count",
        "rolling_child_ready_shadow_token_count",
        "rolling_child_invalidated_count",
        "rolling_cascade_discard_count",
        "rolling_same_seq_overlap_count",
        "rolling_normal_lane_conflict_count",
        "rolling_depth2_real_commit_count",
        "rolling_depth_gt1_real_commit_count",
        "rolling_max_depth_observed",
        "rolling_drop_reason_counts",
    ):
        print(f"{key} = {summary.get(key)}")


def synthetic_valid_record() -> dict[str, Any]:
    return {
        "enable_rolling_continuous_eager_dry_run": True,
        "rolling_continuous_eager_dry_run_enabled": True,
        "rolling_continuous_source": ROLLING_SOURCE,
        "rolling_continuous_stage": ROLLING_STAGE,
        "max_rolling_continuous_depth": 2,
        "rolling_max_depth_observed": 2,
        "target_rolling_eager_verify_proposal_ids": [101],
        "target_rolling_eager_verify_seq_ids": [7],
        "draft_rolling_eager_draft_proposal_ids": [102],
        "draft_rolling_eager_draft_seq_ids": [7],
        "rolling_same_seq_overlap_count": 1,
        "rolling_same_seq_overlap_seq_ids": [7],
        "rolling_normal_lane_conflict_count": 0,
        "rolling_normal_lane_conflict_seq_ids": [],
        "rolling_chain_proposal_ids": [101, 102],
        "rolling_chain_parent_by_proposal_id": {"101": 1, "102": 101},
        "rolling_chain_children_by_proposal_id": {"101": [102]},
        "rolling_chain_root_by_proposal_id": {"101": 1, "102": 1},
        "rolling_chain_depth_by_proposal_id": {"101": 1, "102": 2},
        "rolling_chain_status_by_proposal_id": {
            "101": "PARENT_FULL_ACCEPT",
            "102": "CHILD_READY_AFTER_PARENT_FULL_ACCEPT",
        },
        "rolling_parent_verified_proposal_ids": [101],
        "rolling_parent_full_accept_proposal_ids": [101],
        "rolling_parent_partial_reject_proposal_ids": [],
        "rolling_child_generated_proposal_ids": [102],
        "rolling_child_ready_after_parent_full_accept_proposal_ids": [102],
        "rolling_child_invalidated_proposal_ids": [],
        "rolling_cascade_discarded_proposal_ids": [],
        "rolling_cascade_discard_count": 0,
        "rolling_child_candidate_token_count": 4,
        "rolling_child_ready_shadow_token_count": 4,
        "rolling_depth2_real_commit_count": 0,
        "rolling_depth_gt1_real_commit_count": 0,
    }


def run_synthetic() -> None:
    valid = synthetic_valid_record()
    errors, _summary = validate_records([valid])
    if errors:
        raise SystemExit(f"valid synthetic rolling record failed: {errors}")

    cascade = deepcopy(valid)
    cascade["rolling_chain_status_by_proposal_id"] = {
        "101": "PARENT_NOT_FULL_ACCEPT",
        "102": "CHILD_INVALIDATED_PARENT_REJECT",
    }
    cascade["rolling_parent_full_accept_proposal_ids"] = []
    cascade["rolling_parent_partial_reject_proposal_ids"] = [101]
    cascade["rolling_child_ready_after_parent_full_accept_proposal_ids"] = []
    cascade["rolling_child_invalidated_proposal_ids"] = [102]
    cascade["rolling_child_invalidated_reason_by_proposal_id"] = {"102": "parent_partial_accept"}
    cascade["rolling_cascade_discarded_proposal_ids"] = [102]
    cascade["rolling_cascade_discard_reason_by_proposal_id"] = {"102": "parent_partial_accept"}
    cascade["rolling_cascade_discard_count"] = 1
    cascade["rolling_child_ready_shadow_token_count"] = 0
    errors, _summary = validate_records([cascade])
    if errors:
        raise SystemExit(f"cascade synthetic rolling record failed: {errors}")

    missing_parent = deepcopy(valid)
    missing_parent["rolling_chain_parent_by_proposal_id"] = {"101": 1}
    errors, _summary = validate_records([missing_parent])
    if not errors:
        raise SystemExit("synthetic missing parent should fail")

    depth2_commit = deepcopy(valid)
    depth2_commit["rolling_depth2_real_commit_count"] = 1
    errors, _summary = validate_records([depth2_commit])
    if not errors:
        raise SystemExit("synthetic depth2 real commit should fail")
    print("Synthetic rolling continuous eager dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-8a rolling continuous overlap dry-run traces.")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        run_synthetic()
        return 0
    if not args.trace:
        parser.error("trace is required unless --synthetic is used")
    errors, summary = validate_records(load_trace(Path(args.trace)))
    print_summary(summary)
    if errors:
        print("\nErrors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
