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
ROLLING_DEPTH2_COMMIT_ACTION = "append_full_accept_real_commit"
UNRESOLVED_VERIFY_RESULTS = {"", "unknown", "not_executed"}
PARENT_FAILURE_REASONS = {
    "parent_partial_accept",
    "parent_rejected",
    "parent_not_full_accept",
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


def rolling_row(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("enable_rolling_continuous_eager_dry_run", False))
        or bool(record.get("rolling_continuous_eager_dry_run_enabled", False))
        or bool(record.get("rolling_depth2_commit_enabled", False))
        or bool(record.get("enable_rolling_continuous_depth2_commit_ready_only", False))
        or bool(as_int_set(record.get("draft_rolling_eager_draft_proposal_ids")))
        or bool(as_int_set(record.get("target_rolling_eager_verify_proposal_ids")))
        or bool(as_int_set(record.get("rolling_child_generated_proposal_ids")))
        or bool(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
        or bool(as_int_set(record.get("rolling_depth2_commit_candidate_proposal_ids")))
        or bool(as_int_set(record.get("rolling_depth2_real_committed_proposal_ids")))
    )


def merge_int_map(
    target: dict[int, int],
    incoming: dict[int, int],
    *,
    conflicts: list[str],
    field_name: str,
    record_idx: int,
) -> None:
    for key, value in incoming.items():
        old = target.get(key)
        if old is not None and old != value:
            conflicts.append(
                f"record[{record_idx}] {field_name} conflict for proposal {key}: {old} vs {value}"
            )
            continue
        target[key] = value


def add_reason(reason_by_id: dict[int, set[str]], proposal_id: int, reason: str | None) -> None:
    if reason:
        reason_by_id.setdefault(int(proposal_id), set()).add(str(reason))


def proposal_token_count(proposal_id: int, token_count_by_id: dict[int, int], gamma: int) -> int:
    token_count = int(token_count_by_id.get(proposal_id, 0))
    if token_count > 0:
        return token_count
    return max(0, gamma)


def collect_global_lifecycle(records: list[dict[str, Any]]) -> dict[str, Any]:
    parent_full_ids: set[int] = set()
    parent_partial_ids: set[int] = set()
    ready_child_ids: set[int] = set()
    invalidated_child_ids: set[int] = set()
    cascade_ids: set[int] = set()
    generated_child_ids: set[int] = set()
    target_verify_ids: set[int] = set()
    draft_child_ids: set[int] = set()
    depth2_committed_child_ids: set[int] = set()
    parent_by_child: dict[int, int] = {}
    children_by_parent: dict[int, set[int]] = {}
    depth_by_id: dict[int, int] = {}
    root_by_id: dict[int, int] = {}
    seq_by_id: dict[int, int] = {}
    status_by_id: dict[int, str] = {}
    status_reason_by_id: dict[int, set[str]] = {}
    invalid_reason_by_id: dict[int, set[str]] = {}
    cascade_reason_by_id: dict[int, set[str]] = {}
    token_count_by_id: dict[int, int] = {}
    depth2_parent_by_child: dict[int, int] = {}
    depth2_depth_by_id: dict[int, int] = {}
    depth2_action_by_id: dict[int, str] = {}
    depth2_result_by_id: dict[int, str] = {}
    depth2_token_by_id: dict[int, int] = {}
    depth2_accept_len_by_id: dict[int, int] = {}
    depth2_precondition_ok_by_id: dict[int, bool] = {}
    conflicts: list[str] = []
    gamma = 0
    depth2_commit_enabled = False

    for idx, record in enumerate(records):
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        depth2_commit_enabled = depth2_commit_enabled or bool(record.get("rolling_depth2_commit_enabled", False)) or bool(
            record.get("enable_rolling_continuous_depth2_commit_ready_only", False)
        )

        target_ids = as_int_list(record.get("target_rolling_eager_verify_proposal_ids"))
        target_seq_ids = as_int_list(record.get("target_rolling_eager_verify_seq_ids"))
        child_ids = as_int_list(record.get("draft_rolling_eager_draft_proposal_ids"))
        child_seq_ids = as_int_list(record.get("draft_rolling_eager_draft_seq_ids"))
        target_verify_ids.update(target_ids)
        draft_child_ids.update(child_ids)
        for proposal_id, seq_id in zip(target_ids, target_seq_ids):
            seq_by_id.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(child_ids, child_seq_ids):
            seq_by_id.setdefault(proposal_id, seq_id)

        generated_ids = as_int_set(record.get("rolling_child_generated_proposal_ids"))
        generated_child_ids.update(generated_ids)
        ready_child_ids.update(as_int_set(record.get("rolling_child_ready_after_parent_full_accept_proposal_ids")))
        invalidated_child_ids.update(as_int_set(record.get("rolling_child_invalidated_proposal_ids")))
        cascade_ids.update(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
        depth2_committed_child_ids.update(as_int_set(record.get("rolling_depth2_real_committed_proposal_ids")))

        parent_full_ids.update(as_int_set(record.get("rolling_parent_full_accept_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("continuous_eager_full_accept_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("continuous_eager_real_committed_proposal_ids")))
        parent_partial_ids.update(as_int_set(record.get("rolling_parent_partial_reject_proposal_ids")))
        parent_partial_ids.update(as_int_set(record.get("continuous_eager_partial_reject_proposal_ids")))

        verify_result_by_id = as_str_map(record.get("continuous_eager_verify_result_by_proposal_id"))
        verify_result_by_id.update(as_str_map(record.get("continuous_eager_sync_apply_draft_verify_result_by_proposal_id")))
        verify_result_by_id.update(as_str_map(record.get("continuous_eager_sync_apply_target_verify_result_by_proposal_id")))
        verify_result_by_id.update(as_str_map(record.get("continuous_eager_real_commit_verify_result_by_proposal_id")))
        for proposal_id, verify_result in verify_result_by_id.items():
            if verify_result == "full_accept":
                parent_full_ids.add(proposal_id)
            elif verify_result not in UNRESOLVED_VERIFY_RESULTS:
                parent_partial_ids.add(proposal_id)

        merge_int_map(
            parent_by_child,
            as_int_map(record.get("rolling_chain_parent_by_proposal_id")),
            conflicts=conflicts,
            field_name="rolling_chain_parent_by_proposal_id",
            record_idx=idx,
        )
        merge_int_map(
            parent_by_child,
            as_int_map(record.get("rolling_depth2_commit_parent_by_proposal_id")),
            conflicts=conflicts,
            field_name="rolling_depth2_commit_parent_by_proposal_id",
            record_idx=idx,
        )
        depth2_parent = as_int_map(record.get("rolling_depth2_real_commit_parent_by_proposal_id"))
        merge_int_map(
            parent_by_child,
            depth2_parent,
            conflicts=conflicts,
            field_name="rolling_depth2_real_commit_parent_by_proposal_id",
            record_idx=idx,
        )
        merge_int_map(
            depth2_parent_by_child,
            depth2_parent,
            conflicts=conflicts,
            field_name="rolling_depth2_real_commit_parent_by_proposal_id",
            record_idx=idx,
        )
        merge_int_map(
            depth_by_id,
            as_int_map(record.get("rolling_chain_depth_by_proposal_id")),
            conflicts=conflicts,
            field_name="rolling_chain_depth_by_proposal_id",
            record_idx=idx,
        )
        depth2_depth = as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id"))
        merge_int_map(
            depth_by_id,
            depth2_depth,
            conflicts=conflicts,
            field_name="rolling_depth2_real_commit_depth_by_proposal_id",
            record_idx=idx,
        )
        merge_int_map(
            depth2_depth_by_id,
            depth2_depth,
            conflicts=conflicts,
            field_name="rolling_depth2_real_commit_depth_by_proposal_id",
            record_idx=idx,
        )
        merge_int_map(
            root_by_id,
            as_int_map(record.get("rolling_chain_root_by_proposal_id")),
            conflicts=conflicts,
            field_name="rolling_chain_root_by_proposal_id",
            record_idx=idx,
        )
        merge_int_map(
            root_by_id,
            as_int_map(record.get("rolling_depth2_real_commit_root_by_proposal_id")),
            conflicts=conflicts,
            field_name="rolling_depth2_real_commit_root_by_proposal_id",
            record_idx=idx,
        )
        for parent_id, children in as_list_map(record.get("rolling_chain_children_by_proposal_id")).items():
            children_by_parent.setdefault(parent_id, set()).update(children)
        for proposal_id, status in as_str_map(record.get("rolling_chain_status_by_proposal_id")).items():
            status_by_id[proposal_id] = status
        for proposal_id, reason in as_str_map(record.get("rolling_chain_status_reason_by_proposal_id")).items():
            add_reason(status_reason_by_id, proposal_id, reason)
        for proposal_id, reason in as_str_map(record.get("rolling_child_invalidated_reason_by_proposal_id")).items():
            add_reason(invalid_reason_by_id, proposal_id, reason)
        for proposal_id, reason in as_str_map(record.get("rolling_cascade_discard_reason_by_proposal_id")).items():
            add_reason(cascade_reason_by_id, proposal_id, reason)

        for token_map_key in (
            "rolling_child_token_count_by_proposal_id",
            "rolling_depth2_real_committed_token_count_by_proposal_id",
        ):
            for proposal_id, token_count in as_int_map(record.get(token_map_key)).items():
                if token_count > 0:
                    token_count_by_id.setdefault(proposal_id, token_count)
        for proposal_id in generated_ids | set(child_ids):
            if gamma > 0:
                token_count_by_id.setdefault(proposal_id, gamma)

        for proposal_id, action in as_str_map(record.get("rolling_depth2_real_commit_action_by_proposal_id")).items():
            depth2_action_by_id[proposal_id] = action
        for proposal_id, result in as_str_map(record.get("rolling_depth2_real_commit_verify_result_by_proposal_id")).items():
            depth2_result_by_id[proposal_id] = result
        for proposal_id, token_count in as_int_map(record.get("rolling_depth2_real_committed_token_count_by_proposal_id")).items():
            depth2_token_by_id[proposal_id] = token_count
        for proposal_id, accept_len in as_int_map(record.get("rolling_depth2_real_committed_accept_len_by_proposal_id")).items():
            depth2_accept_len_by_id[proposal_id] = accept_len
        for proposal_id, ok in as_bool_map(record.get("rolling_depth2_commit_precondition_ok_by_proposal_id")).items():
            if ok or proposal_id not in depth2_precondition_ok_by_id:
                depth2_precondition_ok_by_id[proposal_id] = ok

    for proposal_id in generated_child_ids | draft_child_ids | ready_child_ids | depth2_committed_child_ids:
        if gamma > 0:
            token_count_by_id.setdefault(proposal_id, gamma)

    for child_id in sorted(depth2_committed_child_ids):
        parent_id = depth2_parent_by_child.get(child_id, parent_by_child.get(child_id))
        if parent_id is None:
            continue
        token_count = depth2_token_by_id.get(child_id, token_count_by_id.get(child_id, 0))
        accept_len = depth2_accept_len_by_id.get(child_id, token_count)
        depth = depth2_depth_by_id.get(child_id, depth_by_id.get(child_id, 0))
        if (
            child_id in ready_child_ids
            and child_id not in invalidated_child_ids
            and child_id not in cascade_ids
            and depth == 2
            and token_count > 0
            and accept_len == token_count
            and depth2_result_by_id.get(child_id) == "full_accept"
            and depth2_action_by_id.get(child_id) == ROLLING_DEPTH2_COMMIT_ACTION
            and depth2_precondition_ok_by_id.get(child_id, True) is True
        ):
            parent_full_ids.add(parent_id)

    return {
        "parent_full_ids": parent_full_ids,
        "parent_partial_ids": parent_partial_ids,
        "parent_resolved_ids": set(parent_full_ids) | set(parent_partial_ids),
        "ready_child_ids": ready_child_ids,
        "invalidated_child_ids": invalidated_child_ids,
        "cascade_ids": cascade_ids,
        "generated_child_ids": generated_child_ids,
        "target_verify_ids": target_verify_ids,
        "draft_child_ids": draft_child_ids,
        "depth2_committed_child_ids": depth2_committed_child_ids,
        "parent_by_child": parent_by_child,
        "children_by_parent": children_by_parent,
        "depth_by_id": depth_by_id,
        "root_by_id": root_by_id,
        "seq_by_id": seq_by_id,
        "status_by_id": status_by_id,
        "status_reason_by_id": status_reason_by_id,
        "invalid_reason_by_id": invalid_reason_by_id,
        "cascade_reason_by_id": cascade_reason_by_id,
        "token_count_by_id": token_count_by_id,
        "gamma": gamma,
        "depth2_commit_enabled": depth2_commit_enabled,
        "conflicts": conflicts,
    }


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    lifecycle = collect_global_lifecycle(records)
    errors.extend(lifecycle["conflicts"])
    parent_full_ids: set[int] = lifecycle["parent_full_ids"]
    parent_partial_ids: set[int] = lifecycle["parent_partial_ids"]
    parent_resolved_ids: set[int] = lifecycle["parent_resolved_ids"]
    ready_child_ids_seen: set[int] = lifecycle["ready_child_ids"]
    invalidated_child_ids_seen: set[int] = lifecycle["invalidated_child_ids"]
    cascade_ids_seen: set[int] = lifecycle["cascade_ids"]
    generated_child_ids_seen: set[int] = lifecycle["generated_child_ids"]
    target_verify_ids_seen: set[int] = lifecycle["target_verify_ids"]
    draft_child_ids_seen: set[int] = lifecycle["draft_child_ids"]
    depth2_committed_child_ids: set[int] = lifecycle["depth2_committed_child_ids"]
    parent_by_id: dict[int, int] = lifecycle["parent_by_child"]
    children_by_id: dict[int, set[int]] = lifecycle["children_by_parent"]
    root_by_id: dict[int, int] = lifecycle["root_by_id"]
    depth_by_id: dict[int, int] = lifecycle["depth_by_id"]
    seq_by_proposal: dict[int, int] = lifecycle["seq_by_id"]
    status_by_id: dict[int, str] = lifecycle["status_by_id"]
    status_reason_by_id: dict[int, set[str]] = lifecycle["status_reason_by_id"]
    invalid_reason_by_id: dict[int, set[str]] = lifecycle["invalid_reason_by_id"]
    cascade_reason_by_id: dict[int, set[str]] = lifecycle["cascade_reason_by_id"]
    token_count_by_id: dict[int, int] = lifecycle["token_count_by_id"]
    gamma: int = lifecycle["gamma"]
    global_depth2_commit_enabled: bool = lifecycle["depth2_commit_enabled"]
    records_with_enabled = 0
    active_records = 0
    same_seq_overlap_count = 0
    normal_lane_conflict_count = 0
    missing_unexpected_count = 0
    depth2_real_commit_count = 0
    depth_gt1_real_commit_count = 0
    depth3_real_commit_count = 0
    depth_gt2_real_commit_count = 0
    child_verified_without_parent_count = 0
    child_committed_without_parent_count = 0
    drafted_without_parent_count = 0
    duplicate_child_count = 0
    frontier_mismatch_count = 0
    max_depth_observed = 0
    reason_counter: Counter[str] = Counter()
    for reasons in invalid_reason_by_id.values():
        reason_counter.update(reasons)

    for idx, record in enumerate(records):
        rolling_enabled = bool(record.get("enable_rolling_continuous_eager_dry_run", False))
        depth2_commit_enabled = bool(record.get("rolling_depth2_commit_enabled", False)) or bool(
            record.get("enable_rolling_continuous_depth2_commit_ready_only", False)
        )
        if rolling_enabled:
            records_with_enabled += 1
        if not rolling_row(record):
            continue
        active_records += 1
        if not rolling_enabled and not depth2_commit_enabled:
            errors.append(f"record[{idx}] has rolling fields while rolling flags are disabled")
        if record.get("rolling_continuous_eager_dry_run_enabled") and record.get("rolling_continuous_source") != ROLLING_SOURCE:
            errors.append(f"record[{idx}] bad rolling source {record.get('rolling_continuous_source')!r}")
        if record.get("rolling_continuous_eager_dry_run_enabled") and record.get("rolling_continuous_stage") != ROLLING_STAGE:
            errors.append(f"record[{idx}] bad rolling stage {record.get('rolling_continuous_stage')!r}")
        if as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")):
            missing_unexpected_count += len(as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")))
            errors.append(f"record[{idx}] unexpected missing normal proposal while rolling dry-run is active")

        target_ids = as_int_list(record.get("target_rolling_eager_verify_proposal_ids"))
        target_seq_ids = as_int_list(record.get("target_rolling_eager_verify_seq_ids"))
        child_ids = as_int_list(record.get("draft_rolling_eager_draft_proposal_ids"))
        child_seq_ids = as_int_list(record.get("draft_rolling_eager_draft_seq_ids"))
        generated_ids = as_int_set(record.get("rolling_child_generated_proposal_ids"))
        ready_ids = as_int_set(record.get("rolling_child_ready_after_parent_full_accept_proposal_ids"))
        invalidated_ids = as_int_set(record.get("rolling_child_invalidated_proposal_ids"))
        cascade_ids = as_int_set(record.get("rolling_cascade_discarded_proposal_ids"))
        local_invalid_reason_by_id = as_str_map(record.get("rolling_child_invalidated_reason_by_proposal_id"))
        local_cascade_reason_by_id = as_str_map(record.get("rolling_cascade_discard_reason_by_proposal_id"))
        max_depth = int_value(record.get("max_rolling_continuous_depth"), 0)
        observed_depth = max(
            int_value(record.get("rolling_max_depth_observed"), 0),
            int_value(record.get("max_rolling_continuous_depth_observed"), 0),
        )
        max_depth_observed = max(max_depth_observed, observed_depth)

        same_seq_overlap_count += int_value(record.get("rolling_same_seq_overlap_count"), 0)
        normal_lane_conflict_count += int_value(record.get("rolling_normal_lane_conflict_count"), 0)
        record_depth2_count = int_value(record.get("rolling_depth2_real_commit_count"), 0)
        record_depth_gt1_count = int_value(record.get("rolling_depth_gt1_real_commit_count"), 0)
        record_depth3_count = int_value(record.get("rolling_depth3_real_commit_count"), 0)
        record_depth_gt2_count = int_value(record.get("rolling_depth_gt2_real_commit_count"), 0)
        depth2_real_commit_count += record_depth2_count
        depth_gt1_real_commit_count += record_depth_gt1_count
        depth3_real_commit_count += record_depth3_count
        depth_gt2_real_commit_count += record_depth_gt2_count
        child_verified_without_parent_count += int_value(
            record.get("rolling_child_verified_without_parent_full_accept_count"), 0
        )
        child_committed_without_parent_count += int_value(
            record.get("rolling_child_committed_without_parent_full_accept_count"), 0
        )
        drafted_without_parent_count += int_value(record.get("rolling_child_drafted_without_valid_parent_count"), 0)
        duplicate_child_count += int_value(record.get("rolling_duplicate_child_count"), 0)
        frontier_mismatch_count += int_value(record.get("rolling_frontier_mismatch_count"), 0)

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
        if record_depth2_count and not depth2_commit_enabled:
            errors.append(f"record[{idx}] rolling depth-2 real commit requires depth2 commit flag")
        if record_depth3_count:
            errors.append(f"record[{idx}] rolling depth-3 real commit count must stay zero")
        if record_depth_gt2_count:
            errors.append(f"record[{idx}] rolling depth>2 real commit count must stay zero")
        if record_depth_gt1_count and not (
            depth2_commit_enabled
            and record_depth_gt1_count == record_depth2_count
            and record_depth3_count == 0
            and record_depth_gt2_count == 0
        ):
            errors.append(f"record[{idx}] rolling depth>1 real commit count is not pure enabled depth-2 commit")

        for child_id in generated_ids:
            parent_id = parent_by_id.get(child_id)
            if parent_id is None:
                errors.append(f"record[{idx}] rolling child {child_id} missing parent")
                continue
            if parent_id not in parent_by_id and parent_id not in target_verify_ids_seen:
                errors.append(f"record[{idx}] rolling child {child_id} parent {parent_id} is not in chain")
            if seq_by_proposal.get(parent_id) is not None and seq_by_proposal.get(child_id) != seq_by_proposal.get(parent_id):
                errors.append(f"record[{idx}] rolling child {child_id} seq differs from parent {parent_id}")
            child_depth = depth_by_id.get(child_id, 0)
            parent_depth = depth_by_id.get(parent_id, 0)
            if child_depth != parent_depth + 1:
                errors.append(f"record[{idx}] rolling child {child_id} depth is not parent depth + 1")
            if child_id in root_by_id and parent_id in root_by_id and root_by_id.get(child_id) != root_by_id.get(parent_id):
                errors.append(f"record[{idx}] rolling child {child_id} root does not match parent")
            if parent_id in children_by_id and child_id not in children_by_id.get(parent_id, set()):
                errors.append(f"record[{idx}] rolling child {child_id} missing from parent children map")
            if not status_by_id.get(child_id):
                errors.append(f"record[{idx}] rolling child {child_id} missing status")
            child_reasons = set(status_reason_by_id.get(child_id, set())) | set(invalid_reason_by_id.get(child_id, set()))
            if parent_id in parent_full_ids:
                if child_id not in ready_child_ids_seen:
                    errors.append(f"record[{idx}] rolling child {child_id} not ready despite full-accept parent {parent_id}")
                if child_id in invalidated_child_ids_seen:
                    errors.append(f"record[{idx}] rolling child {child_id} invalidated despite full-accept parent {parent_id}")
                if child_id in cascade_ids_seen:
                    errors.append(f"record[{idx}] rolling child {child_id} cascade-discarded despite full-accept parent {parent_id}")
                if "parent_verify_pending" in child_reasons:
                    errors.append(f"record[{idx}] rolling child {child_id} pending despite full-accept parent {parent_id}")
            elif parent_id in parent_partial_ids:
                if child_id not in invalidated_child_ids_seen:
                    errors.append(f"record[{idx}] rolling child {child_id} not invalidated despite not-full-accept parent {parent_id}")
                if not (child_reasons & PARENT_FAILURE_REASONS):
                    errors.append(f"record[{idx}] rolling child {child_id} has bad not-full-accept reason {sorted(child_reasons)!r}")
            elif "parent_verify_pending" in child_reasons and parent_id in parent_resolved_ids:
                errors.append(f"record[{idx}] rolling child {child_id} pending after resolved parent {parent_id}")

        if ready_ids & invalidated_ids:
            errors.append(f"record[{idx}] rolling child cannot be both ready and invalidated")
        for child_id in ready_ids:
            parent_id = parent_by_id.get(child_id, -1)
            if parent_id not in parent_full_ids:
                errors.append(f"record[{idx}] rolling child {child_id} ready without full-accept parent")
        for child_id in invalidated_ids:
            if child_id not in local_invalid_reason_by_id:
                errors.append(f"record[{idx}] rolling invalidated child {child_id} missing reason")
            parent_id = parent_by_id.get(child_id, -1)
            if parent_id in parent_full_ids and child_id in generated_child_ids_seen:
                errors.append(f"record[{idx}] rolling child {child_id} invalidated despite full-accept parent")
            if parent_id in parent_partial_ids and local_invalid_reason_by_id.get(child_id) not in PARENT_FAILURE_REASONS:
                errors.append(f"record[{idx}] rolling child {child_id} has bad parent-failure reason")
        for child_id in cascade_ids:
            if child_id not in local_cascade_reason_by_id:
                errors.append(f"record[{idx}] cascade-discarded child {child_id} missing reason")
            if child_id in ready_ids:
                errors.append(f"record[{idx}] cascade-discarded child {child_id} is also ready")
        if int_value(record.get("rolling_cascade_discard_count"), 0) != len(cascade_ids):
            errors.append(f"record[{idx}] cascade discard count/list mismatch")

    if ready_child_ids_seen & invalidated_child_ids_seen:
        errors.append("rolling child cannot be both globally ready and invalidated")
    if ready_child_ids_seen & cascade_ids_seen:
        errors.append("rolling child cannot be both globally ready and cascade-discarded")
    for child_id in sorted(ready_child_ids_seen):
        parent_id = parent_by_id.get(child_id)
        if parent_id is None:
            errors.append(f"rolling ready child {child_id} missing parent")
        elif parent_id not in parent_full_ids:
            errors.append(f"rolling child {child_id} ready without full-accept parent")
    for child_id in sorted(generated_child_ids_seen):
        parent_id = parent_by_id.get(child_id)
        if parent_id is None:
            continue
        child_reasons = set(status_reason_by_id.get(child_id, set())) | set(invalid_reason_by_id.get(child_id, set()))
        if parent_id in parent_full_ids:
            if child_id not in ready_child_ids_seen:
                errors.append(f"rolling child {child_id} not ready despite full-accept parent {parent_id}")
            if child_id in invalidated_child_ids_seen:
                errors.append(f"rolling child {child_id} invalidated despite full-accept parent {parent_id}")
            if child_id in cascade_ids_seen:
                errors.append(f"rolling child {child_id} cascade-discarded despite full-accept parent {parent_id}")
            if "parent_verify_pending" in child_reasons:
                errors.append(f"rolling child {child_id} pending despite full-accept parent {parent_id}")
        elif parent_id in parent_partial_ids:
            if child_id not in invalidated_child_ids_seen:
                errors.append(f"rolling child {child_id} not invalidated despite not-full-accept parent {parent_id}")
            if not (child_reasons & PARENT_FAILURE_REASONS):
                errors.append(f"rolling child {child_id} has bad not-full-accept reason {sorted(child_reasons)!r}")
        elif "parent_verify_pending" in child_reasons and parent_id in parent_resolved_ids:
            errors.append(f"rolling child {child_id} pending after resolved parent {parent_id}")

    candidate_token_count = sum(
        proposal_token_count(proposal_id, token_count_by_id, gamma)
        for proposal_id in draft_child_ids_seen
    )
    ready_token_count = sum(
        proposal_token_count(proposal_id, token_count_by_id, gamma)
        for proposal_id in ready_child_ids_seen
    )
    depth2_committed_token_count = sum(
        proposal_token_count(proposal_id, token_count_by_id, gamma)
        for proposal_id in depth2_committed_child_ids
    )

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
        "rolling_depth2_real_committed_proposal_count": len(depth2_committed_child_ids),
        "rolling_depth2_real_committed_token_count": depth2_committed_token_count,
        "rolling_depth_gt1_real_commit_count": depth_gt1_real_commit_count,
        "rolling_depth3_real_commit_count": depth3_real_commit_count,
        "rolling_depth_gt2_real_commit_count": depth_gt2_real_commit_count,
        "rolling_child_verified_without_parent_full_accept_count": child_verified_without_parent_count,
        "rolling_child_committed_without_parent_full_accept_count": child_committed_without_parent_count,
        "rolling_child_drafted_without_valid_parent_count": drafted_without_parent_count,
        "rolling_duplicate_child_count": duplicate_child_count,
        "rolling_frontier_mismatch_count": frontier_mismatch_count,
        "rolling_max_depth_observed": max_depth_observed,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "rolling_drop_reason_counts": dict(sorted(reason_counter.items())),
    }
    if depth2_real_commit_count and not global_depth2_commit_enabled:
        errors.append("rolling_depth2_real_commit_count must be zero")
    if depth_gt1_real_commit_count and not (
        global_depth2_commit_enabled
        and depth_gt1_real_commit_count == depth2_real_commit_count
        and depth3_real_commit_count == 0
        and depth_gt2_real_commit_count == 0
    ):
        errors.append("rolling_depth_gt1_real_commit_count is not pure enabled depth-2 commit")
    if depth3_real_commit_count:
        errors.append("rolling_depth3_real_commit_count must be zero")
    if depth_gt2_real_commit_count:
        errors.append("rolling_depth_gt2_real_commit_count must be zero")
    if child_verified_without_parent_count:
        errors.append("rolling child verified without full-accept parent")
    if child_committed_without_parent_count:
        errors.append("rolling child committed without full-accept parent")
    if normal_lane_conflict_count:
        errors.append("rolling normal lane conflicts must be zero")
    if missing_unexpected_count:
        errors.append("unexpected missing normal proposal count must be zero")
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
        "rolling_depth2_real_committed_proposal_count",
        "rolling_depth2_real_committed_token_count",
        "rolling_depth_gt1_real_commit_count",
        "rolling_depth3_real_commit_count",
        "rolling_depth_gt2_real_commit_count",
        "rolling_max_depth_observed",
        "missing_buffered_proposal_unexpected_count",
        "rolling_drop_reason_counts",
    ):
        print(f"{key} = {summary.get(key)}")


def synthetic_valid_record() -> dict[str, Any]:
    return {
        "enable_rolling_continuous_eager_dry_run": True,
        "rolling_continuous_eager_dry_run_enabled": True,
        "rolling_continuous_source": ROLLING_SOURCE,
        "rolling_continuous_stage": ROLLING_STAGE,
        "normal_gamma": 4,
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

    duplicate_side_rows = [deepcopy(valid), deepcopy(valid)]
    errors, summary = validate_records(duplicate_side_rows)
    if errors:
        raise SystemExit(f"duplicate-side synthetic rolling records failed: {errors}")
    if summary.get("rolling_child_ready_shadow_token_count") != 4:
        raise SystemExit("duplicate-side synthetic ready tokens should be counted once")

    distributed_parent = deepcopy(valid)
    distributed_parent["rolling_child_ready_after_parent_full_accept_proposal_ids"] = []
    distributed_parent["rolling_child_ready_shadow_token_count"] = 0
    distributed_ready = deepcopy(valid)
    distributed_ready["rolling_parent_full_accept_proposal_ids"] = []
    errors, _summary = validate_records([distributed_parent, distributed_ready])
    if errors:
        raise SystemExit(f"distributed parent/ready synthetic rolling records failed: {errors}")

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

    missing_full_parent = deepcopy(valid)
    missing_full_parent["rolling_parent_full_accept_proposal_ids"] = []
    errors, _summary = validate_records([missing_full_parent])
    if not errors:
        raise SystemExit("synthetic ready child without global full parent should fail")

    depth2_commit = deepcopy(valid)
    depth2_commit["rolling_depth2_real_commit_count"] = 1
    errors, _summary = validate_records([depth2_commit])
    if not errors:
        raise SystemExit("synthetic depth2 real commit should fail")
    enabled_depth2_side_count = deepcopy(valid)
    enabled_depth2_side_count["enable_rolling_continuous_depth2_commit_ready_only"] = True
    enabled_depth2_side_count["rolling_depth2_commit_enabled"] = True
    enabled_depth2_side_count["rolling_depth2_real_commit_count"] = 1
    enabled_depth2_side_count["rolling_depth_gt1_real_commit_count"] = 1
    errors, _summary = validate_records([enabled_depth2_side_count])
    if errors:
        raise SystemExit(f"synthetic enabled depth2 side count should pass: {errors}")
    resolved_parent_pending = deepcopy(valid)
    resolved_parent_pending["rolling_child_ready_after_parent_full_accept_proposal_ids"] = []
    resolved_parent_pending["rolling_child_invalidated_proposal_ids"] = [102]
    resolved_parent_pending["rolling_child_invalidated_reason_by_proposal_id"] = {"102": "parent_verify_pending"}
    resolved_parent_pending["rolling_chain_status_by_proposal_id"] = {
        "101": "PARENT_FULL_ACCEPT",
        "102": "CHILD_DROPPED",
    }
    resolved_parent_pending["continuous_eager_commit_ready_shadow_proposal_ids"] = [101]
    errors, _summary = validate_records([resolved_parent_pending])
    if not errors:
        raise SystemExit("synthetic resolved parent with pending child should fail")
    partial_parent_pending = deepcopy(valid)
    partial_parent_pending["rolling_parent_full_accept_proposal_ids"] = []
    partial_parent_pending["rolling_parent_partial_reject_proposal_ids"] = [101]
    partial_parent_pending["rolling_child_ready_after_parent_full_accept_proposal_ids"] = []
    partial_parent_pending["rolling_child_invalidated_proposal_ids"] = [102]
    partial_parent_pending["rolling_child_invalidated_reason_by_proposal_id"] = {"102": "parent_verify_pending"}
    partial_parent_pending["continuous_eager_verify_result_by_proposal_id"] = {"101": "partial_accept"}
    errors, _summary = validate_records([partial_parent_pending])
    if not errors:
        raise SystemExit("synthetic partial parent with pending child should fail")
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
