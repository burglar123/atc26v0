#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


DEPTH3_STAGE = "depth3_shadow_dry_run"
DEPTH3_PARENT_FAILURE_REASONS = {
    "parent_depth2_not_committed",
    "parent_depth2_precondition_failed",
    "parent_depth2_skipped",
    "parent_depth2_not_full_accept",
    "parent_depth2_invalidated",
    "parent_depth2_cascade_discarded",
    "parent_depth2_pending",
}
DEPTH3_ACTIVE_LIST_FIELDS = (
    "rolling_depth3_child_generated_proposal_ids",
    "rolling_depth3_child_ready_shadow_proposal_ids",
    "rolling_depth3_child_invalidated_proposal_ids",
    "rolling_depth3_parent_depth2_real_committed_proposal_ids",
    "rolling_depth3_parent_depth2_full_accept_proposal_ids",
    "rolling_depth3_parent_depth2_skipped_proposal_ids",
    "rolling_depth3_parent_depth2_invalidated_proposal_ids",
    "rolling_depth3_parent_resolution_pending_proposal_ids",
    "rolling_depth3_committed_without_parent_depth2_commit_ids",
    "rolling_depth3_committed_without_ready_shadow_ids",
    "rolling_depth3_committed_invalidated_child_ids",
    "rolling_depth3_committed_cascade_discarded_child_ids",
    "rolling_depth3_duplicate_child_ids",
    "rolling_depth3_child_generation_skipped_proposal_ids",
)
DEPTH3_ACTIVE_MAP_FIELDS = (
    "rolling_depth3_child_parent_by_proposal_id",
    "rolling_depth3_child_root_by_proposal_id",
    "rolling_depth3_child_depth_by_proposal_id",
    "rolling_depth3_child_token_count_by_proposal_id",
    "rolling_depth3_child_base_len_by_proposal_id",
    "rolling_depth3_child_status_by_proposal_id",
    "rolling_depth3_child_status_reason_by_proposal_id",
    "rolling_depth3_child_invalidated_reason_by_proposal_id",
    "rolling_depth3_child_generation_skipped_parent_by_proposal_id",
    "rolling_depth3_child_generation_skip_reason_by_proposal_id",
    "rolling_depth3_child_generation_skip_reason_counts",
    "rolling_depth3_drop_reason_counts",
)
DEPTH3_ACTIVE_COUNT_FIELDS = (
    "rolling_depth3_child_candidate_proposal_count",
    "rolling_depth3_child_candidate_token_count",
    "rolling_depth3_child_ready_shadow_proposal_count",
    "rolling_depth3_child_ready_shadow_token_count",
    "rolling_depth3_child_invalidated_count",
    "rolling_depth3_same_seq_overlap_count",
    "rolling_depth3_normal_lane_conflict_count",
    "rolling_depth3_real_commit_count",
    "rolling_depth4_real_commit_count",
    "rolling_depth_gt3_real_commit_count",
    "rolling_depth3_parent_resolution_pending_count",
    "rolling_depth3_frontier_mismatch_count",
    "rolling_depth3_max_depth_observed",
)
DEPTH3_ALLOWED_SKIP_REASONS = DEPTH3_PARENT_FAILURE_REASONS | {
    "max_depth_exceeded",
    "parent_depth2_depth_mismatch",
    "parent_depth2_chain_missing",
    "parent_depth2_missing_shadow_proposal",
    "parent_depth2_bad_action",
    "parent_depth2_token_mismatch",
    "parent_depth2_len_mismatch",
    "duplicate_child",
    "seq_not_found",
    "seq_pre_verify",
    "frontier_mismatch",
    "invalid_depth3_token_span",
    "parent_depth2_finished",
}


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


def active_depth3_fields(record: dict[str, Any]) -> bool:
    if bool(record.get("enable_rolling_continuous_depth3_shadow_dry_run", False)):
        return True
    if bool(record.get("rolling_depth3_shadow_enabled", False)):
        return True
    if record.get("rolling_depth3_shadow_stage") or record.get("rolling_depth3_shadow_source"):
        return True
    for field in DEPTH3_ACTIVE_LIST_FIELDS:
        if as_int_set(record.get(field)):
            return True
    for field in DEPTH3_ACTIVE_MAP_FIELDS:
        value = record.get(field)
        if isinstance(value, dict) and value:
            return True
    for field in DEPTH3_ACTIVE_COUNT_FIELDS:
        if int_value(record.get(field), 0) != 0:
            return True
    return False


def token_count(proposal_id: int, token_by_id: dict[int, int], gamma: int) -> int:
    value = int(token_by_id.get(proposal_id, 0))
    return value if value > 0 else max(0, gamma)


def merge_int_map(target: dict[int, int], source: dict[int, int]) -> None:
    for key, value in source.items():
        target.setdefault(key, value)


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    enabled_records = 0
    active_records = 0
    gamma = 0
    generated_ids: set[int] = set()
    ready_ids: set[int] = set()
    invalidated_ids: set[int] = set()
    skipped_child_ids: set[int] = set()
    parent_committed_ids: set[int] = set()
    parent_full_ids: set[int] = set()
    parent_skipped_ids: set[int] = set()
    parent_invalidated_ids: set[int] = set()
    parent_pending_ids: set[int] = set()
    depth2_committed_ids: set[int] = set()
    parent_by_child: dict[int, int] = {}
    root_by_child: dict[int, int] = {}
    depth_by_child: dict[int, int] = {}
    token_by_child: dict[int, int] = {}
    seq_by_child: dict[int, int] = {}
    seq_by_parent: dict[int, int] = {}
    root_by_parent: dict[int, int] = {}
    depth_by_parent: dict[int, int] = {}
    result_by_parent: dict[int, str] = {}
    action_by_parent: dict[int, str] = {}
    precondition_ok_by_parent: dict[int, bool] = {}
    invalid_reason_by_child: dict[int, str] = {}
    skip_reason_by_child: dict[int, str] = {}
    status_by_child: dict[int, str] = {}
    drop_reason_counter: Counter[str] = Counter()
    normal_lane_conflict_count = 0
    missing_unexpected_count = 0
    same_seq_overlap_count = 0
    depth3_real_commit_count = 0
    depth_gt3_real_commit_count = 0
    max_depth_observed = 0
    commit_enabled_records = 0

    for idx, record in enumerate(records):
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        enabled = bool(record.get("enable_rolling_continuous_depth3_shadow_dry_run", False))
        commit_enabled = bool(record.get("enable_rolling_continuous_depth3_commit_ready_only", False)) or bool(
            record.get("rolling_depth3_commit_enabled", False)
        )
        if commit_enabled:
            commit_enabled_records += 1
        if enabled:
            enabled_records += 1
        active = active_depth3_fields(record)
        if not active:
            continue
        active_records += 1
        if not enabled:
            errors.append(f"record[{idx}] has depth3 shadow fields while flag is disabled")
        if record.get("rolling_depth3_shadow_enabled") and record.get("rolling_depth3_shadow_stage") != DEPTH3_STAGE:
            errors.append(f"record[{idx}] bad rolling depth3 stage {record.get('rolling_depth3_shadow_stage')!r}")
        if as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")):
            missing_unexpected_count += len(as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")))
            errors.append(f"record[{idx}] unexpected missing buffered proposal")
        if as_int_set(record.get("rolling_depth3_normal_lane_conflict_seq_ids")):
            normal_lane_conflict_count += len(as_int_set(record.get("rolling_depth3_normal_lane_conflict_seq_ids")))
            errors.append(f"record[{idx}] rolling depth3 normal lane conflict present")
        normal_lane_conflict_count += int_value(record.get("rolling_depth3_normal_lane_conflict_count"), 0)
        same_seq_overlap_count += int_value(record.get("rolling_depth3_same_seq_overlap_count"), 0)
        depth3_real_commit_count += int_value(record.get("rolling_depth3_real_commit_count"), 0)
        depth_gt3_real_commit_count += int_value(record.get("rolling_depth_gt3_real_commit_count"), 0)
        max_depth_observed = max(
            max_depth_observed,
            int_value(record.get("rolling_depth3_max_depth_observed"), 0),
            int_value(record.get("max_rolling_continuous_depth_observed"), 0),
        )
        if int_value(record.get("rolling_depth3_real_commit_count"), 0) and not commit_enabled:
            errors.append(f"record[{idx}] rolling depth3 real commit count must be zero")
        if int_value(record.get("rolling_depth4_real_commit_count"), 0):
            errors.append(f"record[{idx}] rolling depth4 real commit count must be zero")
        if int_value(record.get("rolling_depth_gt3_real_commit_count"), 0):
            errors.append(f"record[{idx}] rolling depth>3 real commit count must be zero")

        child_ids = as_int_set(record.get("rolling_depth3_child_generated_proposal_ids"))
        child_seq_ids = as_int_list(record.get("rolling_depth3_child_generated_seq_ids"))
        generated_ids.update(child_ids)
        ready_ids.update(as_int_set(record.get("rolling_depth3_child_ready_shadow_proposal_ids")))
        invalidated_ids.update(as_int_set(record.get("rolling_depth3_child_invalidated_proposal_ids")))
        skipped_record_ids = as_int_set(record.get("rolling_depth3_child_generation_skipped_proposal_ids"))
        skipped_child_ids.update(skipped_record_ids)
        parent_committed_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_real_committed_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_full_accept_proposal_ids")))
        parent_skipped_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_skipped_proposal_ids")))
        parent_invalidated_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_invalidated_proposal_ids")))
        parent_pending_ids.update(as_int_set(record.get("rolling_depth3_parent_resolution_pending_proposal_ids")))
        depth2_ids = as_int_set(record.get("rolling_depth2_real_committed_proposal_ids"))
        depth2_committed_ids.update(depth2_ids)
        merge_int_map(parent_by_child, as_int_map(record.get("rolling_depth3_child_parent_by_proposal_id")))
        merge_int_map(parent_by_child, as_int_map(record.get("rolling_depth3_commit_parent_by_proposal_id")))
        merge_int_map(parent_by_child, as_int_map(record.get("rolling_depth3_real_commit_parent_by_proposal_id")))
        merge_int_map(root_by_child, as_int_map(record.get("rolling_depth3_child_root_by_proposal_id")))
        merge_int_map(root_by_child, as_int_map(record.get("rolling_depth3_real_commit_root_by_proposal_id")))
        merge_int_map(depth_by_child, as_int_map(record.get("rolling_depth3_child_depth_by_proposal_id")))
        merge_int_map(depth_by_child, as_int_map(record.get("rolling_depth3_real_commit_depth_by_proposal_id")))
        merge_int_map(token_by_child, as_int_map(record.get("rolling_depth3_child_token_count_by_proposal_id")))
        merge_int_map(token_by_child, as_int_map(record.get("rolling_depth3_real_committed_token_count_by_proposal_id")))
        for proposal_id, status in as_str_map(record.get("rolling_depth3_child_status_by_proposal_id")).items():
            status_by_child.setdefault(proposal_id, status)
        generation_skipped_parent_by_child = as_int_map(
            record.get("rolling_depth3_child_generation_skipped_parent_by_proposal_id")
        )
        for child_id, parent_id in generation_skipped_parent_by_child.items():
            parent_by_child.setdefault(child_id, parent_id)
        merge_int_map(root_by_parent, as_int_map(record.get("rolling_depth2_real_commit_root_by_proposal_id")))
        merge_int_map(depth_by_parent, as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id")))
        for proposal_id, result in as_str_map(record.get("rolling_depth2_real_commit_verify_result_by_proposal_id")).items():
            result_by_parent.setdefault(proposal_id, result)
            if result == "full_accept" and proposal_id in depth2_ids:
                parent_full_ids.add(proposal_id)
        for proposal_id, action in as_str_map(record.get("rolling_depth2_real_commit_action_by_proposal_id")).items():
            action_by_parent.setdefault(proposal_id, action)
        for proposal_id, ok in as_bool_map(record.get("rolling_depth2_commit_precondition_ok_by_proposal_id")).items():
            if ok or proposal_id not in precondition_ok_by_parent:
                precondition_ok_by_parent[proposal_id] = ok
        for proposal_id, seq_id in zip(as_int_list(record.get("rolling_depth2_real_committed_proposal_ids")), as_int_list(record.get("rolling_depth2_real_committed_seq_ids"))):
            seq_by_parent.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(as_int_list(record.get("rolling_depth2_commit_candidate_proposal_ids")), as_int_list(record.get("rolling_depth2_commit_candidate_seq_ids"))):
            seq_by_parent.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(as_int_list(record.get("draft_rolling_eager_draft_proposal_ids")), as_int_list(record.get("draft_rolling_eager_draft_seq_ids"))):
            seq_by_parent.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(as_int_list(record.get("rolling_depth3_child_generated_proposal_ids")), child_seq_ids):
            seq_by_child.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(as_int_list(record.get("rolling_depth3_real_committed_proposal_ids")), as_int_list(record.get("rolling_depth3_real_committed_seq_ids"))):
            seq_by_child.setdefault(proposal_id, seq_id)
        for proposal_id, reason in as_str_map(record.get("rolling_depth3_child_invalidated_reason_by_proposal_id")).items():
            invalid_reason_by_child.setdefault(proposal_id, reason)
            if reason:
                drop_reason_counter[reason] += 1
        for proposal_id, reason in as_str_map(record.get("rolling_depth3_child_generation_skip_reason_by_proposal_id")).items():
            skip_reason_by_child.setdefault(proposal_id, reason)
            if reason:
                drop_reason_counter[reason] += 1

        if child_ids and int_value(record.get("rolling_depth3_child_candidate_proposal_count"), len(child_ids)) != len(child_ids):
            errors.append(f"record[{idx}] depth3 candidate count/list mismatch")
        expected_candidate_tokens = sum(token_count(proposal_id, as_int_map(record.get("rolling_depth3_child_token_count_by_proposal_id")), gamma) for proposal_id in child_ids)
        if child_ids and int_value(record.get("rolling_depth3_child_candidate_token_count"), expected_candidate_tokens) != expected_candidate_tokens:
            errors.append(f"record[{idx}] depth3 candidate token count mismatch")
        ready_record_ids = as_int_set(record.get("rolling_depth3_child_ready_shadow_proposal_ids"))
        expected_ready_tokens = sum(token_count(proposal_id, as_int_map(record.get("rolling_depth3_child_token_count_by_proposal_id")), gamma) for proposal_id in ready_record_ids)
        if ready_record_ids and int_value(record.get("rolling_depth3_child_ready_shadow_token_count"), expected_ready_tokens) != expected_ready_tokens:
            errors.append(f"record[{idx}] depth3 ready token count mismatch")
        duplicate_ids = as_int_set(record.get("rolling_depth3_duplicate_child_ids"))
        if duplicate_ids:
            errors.append(f"record[{idx}] duplicate rolling depth3 children present: {sorted(duplicate_ids)}")

    parent_committed_ids.update(depth2_committed_ids)
    parent_resolved_ids = parent_committed_ids | parent_skipped_ids | parent_invalidated_ids
    committed_depth3_children = (generated_ids | ready_ids | invalidated_ids | skipped_child_ids) & depth2_committed_ids
    if committed_depth3_children:
        errors.append(f"rolling depth3 children appeared in real-committed ids: {sorted(committed_depth3_children)}")
    for child_id in sorted(generated_ids | ready_ids | invalidated_ids | skipped_child_ids):
        parent_id = parent_by_child.get(child_id)
        if parent_id is None:
            errors.append(f"rolling depth3 child {child_id} missing parent")
            continue
        if depth_by_parent.get(parent_id, 2) != 2:
            errors.append(f"rolling depth3 child {child_id} parent {parent_id} depth is not 2")
        if depth_by_child.get(child_id, 0) != 3:
            errors.append(f"rolling depth3 child {child_id} depth is not 3")
        if parent_id in root_by_parent and child_id in root_by_child and root_by_parent[parent_id] != root_by_child[child_id]:
            errors.append(f"rolling depth3 child {child_id} root does not match parent {parent_id}")
        if parent_id in seq_by_parent and child_id in seq_by_child and seq_by_parent[parent_id] != seq_by_child[child_id]:
            errors.append(f"rolling depth3 child {child_id} seq differs from parent {parent_id}")

    if ready_ids & invalidated_ids:
        errors.append("rolling depth3 child cannot be both ready and invalidated")
    for child_id in sorted(ready_ids):
        parent_id = parent_by_child.get(child_id)
        if parent_id is None:
            continue
        if parent_id not in parent_committed_ids or parent_id not in parent_full_ids:
            errors.append(f"rolling depth3 child {child_id} ready without committed/full-accept depth2 parent")
        if parent_id in parent_skipped_ids:
            errors.append(f"rolling depth3 child {child_id} ready despite skipped depth2 parent {parent_id}")
        if parent_id in parent_invalidated_ids:
            errors.append(f"rolling depth3 child {child_id} ready despite invalidated depth2 parent {parent_id}")
        if result_by_parent.get(parent_id, "full_accept") != "full_accept":
            errors.append(f"rolling depth3 child {child_id} ready without full_accept parent result")
        if action_by_parent.get(parent_id, "append_full_accept_real_commit") != "append_full_accept_real_commit":
            errors.append(f"rolling depth3 child {child_id} ready without append real-commit parent action")
        if precondition_ok_by_parent.get(parent_id, True) is not True:
            errors.append(f"rolling depth3 child {child_id} ready despite failed parent precondition")
        status = status_by_child.get(child_id)
        if status and status != "DEPTH3_READY_AFTER_PARENT_DEPTH2_COMMIT":
            errors.append(f"rolling depth3 ready child {child_id} has unexpected status {status!r}")

    for child_id in sorted(generated_ids - ready_ids - invalidated_ids):
        parent_id = parent_by_child.get(child_id)
        if parent_id in parent_resolved_ids:
            errors.append(f"rolling depth3 child {child_id} unresolved after parent {parent_id} resolved")

    for child_id in sorted(invalidated_ids):
        parent_id = parent_by_child.get(child_id)
        reason = invalid_reason_by_child.get(child_id, "")
        if not reason:
            errors.append(f"rolling depth3 invalidated child {child_id} missing reason")
        elif reason not in DEPTH3_PARENT_FAILURE_REASONS and reason not in {
            "max_depth_exceeded",
            "parent_depth2_depth_mismatch",
            "parent_depth2_chain_missing",
            "parent_depth2_bad_action",
            "parent_depth2_token_mismatch",
            "parent_depth2_len_mismatch",
            "duplicate_child",
            "seq_not_found",
            "seq_pre_verify",
            "frontier_mismatch",
            "invalid_depth3_token_span",
            "parent_depth2_finished",
        }:
            errors.append(f"rolling depth3 child {child_id} has bad invalidation reason {reason!r}")
        if parent_id in parent_committed_ids and parent_id in parent_full_ids and reason in DEPTH3_PARENT_FAILURE_REASONS:
            errors.append(f"rolling depth3 child {child_id} invalidated by parent failure despite committed parent")
        if reason == "parent_depth2_pending" and parent_id in parent_resolved_ids:
            errors.append(f"rolling depth3 child {child_id} has pending reason after parent resolved")

    for child_id in sorted(skipped_child_ids):
        reason = skip_reason_by_child.get(child_id, "")
        if not reason:
            errors.append(f"rolling depth3 skipped child {child_id} missing reason")
        elif reason not in DEPTH3_ALLOWED_SKIP_REASONS:
            errors.append(f"rolling depth3 skipped child {child_id} has bad reason {reason!r}")
        if child_id in ready_ids:
            errors.append(f"rolling depth3 skipped child {child_id} is also ready")

    for parent_id in sorted(parent_skipped_ids | parent_invalidated_ids):
        children = [child_id for child_id, p_id in parent_by_child.items() if p_id == parent_id]
        for child_id in children:
            if child_id in ready_ids:
                errors.append(f"rolling depth3 child {child_id} ready despite failed parent {parent_id}")
            if child_id in generated_ids and child_id not in invalidated_ids:
                errors.append(f"rolling depth3 child {child_id} not invalidated despite failed parent {parent_id}")

    candidate_token_count = sum(token_count(proposal_id, token_by_child, gamma) for proposal_id in generated_ids)
    ready_token_count = sum(token_count(proposal_id, token_by_child, gamma) for proposal_id in ready_ids)
    pending_child_count = len(generated_ids - ready_ids - invalidated_ids)
    if len(ready_ids) + len(invalidated_ids) + pending_child_count > len(generated_ids):
        errors.append("rolling depth3 ready + invalidated + pending exceeds candidates")
    if max_depth_observed > 3:
        errors.append("rolling depth3 observed max depth exceeds 3")
    if enabled_records and parent_committed_ids and not generated_ids and not skipped_child_ids and not parent_pending_ids:
        errors.append("rolling depth3 shadow enabled with committed depth2 parents but no generated children or skip reasons")
    if depth3_real_commit_count and not commit_enabled_records:
        errors.append("rolling_depth3_real_commit_count must be zero")
    if depth_gt3_real_commit_count:
        errors.append("rolling_depth_gt3_real_commit_count must be zero")
    if normal_lane_conflict_count:
        errors.append("rolling_depth3_normal_lane_conflict_count must be zero")
    if missing_unexpected_count:
        errors.append("missing_buffered_proposal_unexpected_count must be zero")

    summary = {
        "total_trace_records": len(records),
        "records_with_rolling_depth3_shadow_enabled": enabled_records,
        "rolling_depth3_shadow_active_records": active_records,
        "rolling_depth3_child_candidate_proposal_count": len(generated_ids),
        "rolling_depth3_child_candidate_token_count": candidate_token_count,
        "rolling_depth3_child_ready_shadow_proposal_count": len(ready_ids),
        "rolling_depth3_child_ready_shadow_token_count": ready_token_count,
        "rolling_depth3_child_invalidated_count": len(invalidated_ids),
        "rolling_depth3_parent_resolution_pending_count": len(parent_pending_ids),
        "rolling_depth3_same_seq_overlap_count": same_seq_overlap_count,
        "rolling_depth3_normal_lane_conflict_count": normal_lane_conflict_count,
        "rolling_depth3_real_commit_count": depth3_real_commit_count,
        "rolling_depth_gt3_real_commit_count": depth_gt3_real_commit_count,
        "rolling_depth3_drop_reason_counts": dict(sorted(drop_reason_counter.items())),
        "rolling_depth3_max_depth_observed": max_depth_observed,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_rolling_depth3_shadow_enabled",
        "rolling_depth3_shadow_active_records",
        "rolling_depth3_child_candidate_proposal_count",
        "rolling_depth3_child_candidate_token_count",
        "rolling_depth3_child_ready_shadow_proposal_count",
        "rolling_depth3_child_ready_shadow_token_count",
        "rolling_depth3_child_invalidated_count",
        "rolling_depth3_parent_resolution_pending_count",
        "rolling_depth3_same_seq_overlap_count",
        "rolling_depth3_normal_lane_conflict_count",
        "rolling_depth3_real_commit_count",
        "rolling_depth_gt3_real_commit_count",
        "rolling_depth3_drop_reason_counts",
        "rolling_depth3_max_depth_observed",
    ):
        print(f"{key} = {summary.get(key)}")


def synthetic_good_record() -> dict[str, Any]:
    parent_id = 900000102
    child_id = 900000103
    root_id = 101
    seq_id = 7
    return {
        "normal_gamma": 4,
        "enable_rolling_continuous_depth3_shadow_dry_run": True,
        "rolling_depth3_shadow_enabled": True,
        "rolling_depth3_shadow_stage": DEPTH3_STAGE,
        "rolling_depth3_shadow_source": "rolling_depth3_shadow",
        "max_rolling_continuous_depth_observed": 3,
        "rolling_depth2_real_committed_proposal_ids": [parent_id],
        "rolling_depth2_real_committed_seq_ids": [seq_id],
        "rolling_depth2_real_committed_token_count_by_proposal_id": {str(parent_id): 4},
        "rolling_depth2_real_committed_accept_len_by_proposal_id": {str(parent_id): 4},
        "rolling_depth2_real_commit_action_by_proposal_id": {str(parent_id): "append_full_accept_real_commit"},
        "rolling_depth2_real_commit_verify_result_by_proposal_id": {str(parent_id): "full_accept"},
        "rolling_depth2_real_commit_parent_by_proposal_id": {str(parent_id): 900000101},
        "rolling_depth2_real_commit_root_by_proposal_id": {str(parent_id): root_id},
        "rolling_depth2_real_commit_depth_by_proposal_id": {str(parent_id): 2},
        "rolling_depth2_commit_precondition_ok_by_proposal_id": {str(parent_id): True},
        "rolling_depth3_parent_depth2_real_committed_proposal_ids": [parent_id],
        "rolling_depth3_parent_depth2_full_accept_proposal_ids": [parent_id],
        "rolling_depth3_parent_depth2_skipped_proposal_ids": [],
        "rolling_depth3_parent_depth2_invalidated_proposal_ids": [],
        "rolling_depth3_parent_resolution_pending_proposal_ids": [],
        "rolling_depth3_child_generated_proposal_ids": [child_id],
        "rolling_depth3_child_generated_seq_ids": [seq_id],
        "rolling_depth3_child_parent_by_proposal_id": {str(child_id): parent_id},
        "rolling_depth3_child_root_by_proposal_id": {str(child_id): root_id},
        "rolling_depth3_child_depth_by_proposal_id": {str(child_id): 3},
        "rolling_depth3_child_token_count_by_proposal_id": {str(child_id): 4},
        "rolling_depth3_child_base_len_by_proposal_id": {str(child_id): 16},
        "rolling_depth3_child_ready_shadow_proposal_ids": [child_id],
        "rolling_depth3_child_ready_shadow_seq_ids": [seq_id],
        "rolling_depth3_child_invalidated_proposal_ids": [],
        "rolling_depth3_child_invalidated_reason_by_proposal_id": {},
        "rolling_depth3_child_candidate_proposal_count": 1,
        "rolling_depth3_child_candidate_token_count": 4,
        "rolling_depth3_child_ready_shadow_proposal_count": 1,
        "rolling_depth3_child_ready_shadow_token_count": 4,
        "rolling_depth3_child_invalidated_count": 0,
        "rolling_depth3_parent_resolution_pending_count": 0,
        "rolling_depth3_same_seq_overlap_count": 1,
        "rolling_depth3_same_seq_overlap_seq_ids": [seq_id],
        "rolling_depth3_normal_lane_conflict_count": 0,
        "rolling_depth3_normal_lane_conflict_seq_ids": [],
        "rolling_depth3_real_commit_count": 0,
        "rolling_depth_gt3_real_commit_count": 0,
        "rolling_depth3_max_depth_observed": 3,
        "missing_buffered_proposal_unexpected_seq_ids": [],
    }


def run_synthetic() -> None:
    good = synthetic_good_record()
    errors, summary = validate_records([good])
    if errors:
        raise SystemExit(f"synthetic depth3 good case failed: {errors}\nsummary={summary}")

    parent_failure = deepcopy(good)
    parent_id = 900000102
    child_id = 900000103
    parent_failure["rolling_depth2_real_committed_proposal_ids"] = []
    parent_failure["rolling_depth2_real_committed_seq_ids"] = []
    parent_failure["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    parent_failure["rolling_depth3_parent_depth2_full_accept_proposal_ids"] = []
    parent_failure["rolling_depth3_parent_depth2_skipped_proposal_ids"] = [parent_id]
    parent_failure["rolling_depth3_child_ready_shadow_proposal_ids"] = []
    parent_failure["rolling_depth3_child_ready_shadow_seq_ids"] = []
    parent_failure["rolling_depth3_child_ready_shadow_proposal_count"] = 0
    parent_failure["rolling_depth3_child_ready_shadow_token_count"] = 0
    parent_failure["rolling_depth3_child_invalidated_proposal_ids"] = [child_id]
    parent_failure["rolling_depth3_child_invalidated_reason_by_proposal_id"] = {
        str(child_id): "parent_depth2_skipped"
    }
    parent_failure["rolling_depth3_child_invalidated_count"] = 1
    errors, _summary = validate_records([parent_failure])
    if errors:
        raise SystemExit(f"synthetic depth3 parent failure should pass: {errors}")

    ready_without_parent = deepcopy(good)
    ready_without_parent["rolling_depth2_real_committed_proposal_ids"] = []
    ready_without_parent["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    ready_without_parent["rolling_depth3_parent_depth2_full_accept_proposal_ids"] = []
    errors, _summary = validate_records([ready_without_parent])
    if not errors:
        raise SystemExit("synthetic depth3 ready without depth2 commit should fail")

    real_commit = deepcopy(good)
    real_commit["rolling_depth3_real_commit_count"] = 1
    errors, _summary = validate_records([real_commit])
    if not errors:
        raise SystemExit("synthetic depth3 real commit should fail")

    legal_real_commit = deepcopy(good)
    legal_real_commit["enable_rolling_continuous_depth3_commit_ready_only"] = True
    legal_real_commit["rolling_depth3_commit_enabled"] = True
    legal_real_commit["rolling_depth3_real_commit_count"] = 1
    legal_real_commit["rolling_depth3_real_committed_proposal_ids"] = [child_id]
    legal_real_commit["rolling_depth3_real_committed_seq_ids"] = [7]
    legal_real_commit["rolling_depth3_real_committed_token_count_by_proposal_id"] = {str(child_id): 4}
    legal_real_commit["rolling_depth3_real_commit_parent_by_proposal_id"] = {str(child_id): parent_id}
    legal_real_commit["rolling_depth3_real_commit_depth_by_proposal_id"] = {str(child_id): 3}
    errors, _summary = validate_records([legal_real_commit])
    if errors:
        raise SystemExit(f"synthetic legal depth3 real commit should pass shadow checker: {errors}")

    commit_record = deepcopy(good)
    ready_record = deepcopy(good)
    commit_record["rolling_depth3_child_generated_proposal_ids"] = []
    commit_record["rolling_depth3_child_generated_seq_ids"] = []
    commit_record["rolling_depth3_child_ready_shadow_proposal_ids"] = []
    commit_record["rolling_depth3_child_ready_shadow_seq_ids"] = []
    commit_record["rolling_depth3_child_candidate_proposal_count"] = 0
    commit_record["rolling_depth3_child_candidate_token_count"] = 0
    commit_record["rolling_depth3_child_ready_shadow_proposal_count"] = 0
    commit_record["rolling_depth3_child_ready_shadow_token_count"] = 0
    ready_record["rolling_depth2_real_committed_proposal_ids"] = []
    ready_record["rolling_depth2_real_committed_seq_ids"] = []
    ready_record["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    ready_record["rolling_depth3_parent_depth2_full_accept_proposal_ids"] = []
    errors, _summary = validate_records([commit_record, ready_record])
    if errors:
        raise SystemExit(f"synthetic distributed depth3 evidence should pass: {errors}")

    errors, summary = validate_records([good, deepcopy(good)])
    if errors:
        raise SystemExit(f"synthetic duplicate side rows should pass: {errors}")
    if summary.get("rolling_depth3_child_ready_shadow_token_count") != 4:
        raise SystemExit("synthetic duplicate side rows should dedupe ready tokens")

    print("Synthetic rolling continuous depth-3 shadow checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-8c rolling depth-3 shadow dry-run traces.")
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
