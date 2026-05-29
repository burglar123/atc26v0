#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROLLING_DEPTH3_COMMIT_SOURCE = "rolling_depth3_ready_only"
ROLLING_ACTION = "append_full_accept_real_commit"


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_performance_accounting import aggregate_performance_accounting  # noqa: E402


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
    for key, value_item in value.items():
        try:
            result[int(key)] = int(value_item)
        except Exception:
            continue
    return result


def as_bool_map(value: Any) -> dict[int, bool]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, bool] = {}
    for key, value_item in value.items():
        try:
            result[int(key)] = bool(value_item)
        except Exception:
            continue
    return result


def as_str_map(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, str] = {}
    for key, value_item in value.items():
        try:
            result[int(key)] = str(value_item)
        except Exception:
            continue
    return result


def token_count(proposal_id: int, token_by_id: dict[int, int], gamma: int) -> int:
    value = int(token_by_id.get(proposal_id, 0))
    return value if value > 0 else max(0, gamma)


def commit_active(record: dict[str, Any]) -> bool:
    if bool(record.get("enable_rolling_continuous_depth3_commit_ready_only", False)):
        return True
    if bool(record.get("rolling_depth3_commit_enabled", False)):
        return True
    fields = (
        "rolling_depth3_commit_candidate_proposal_ids",
        "rolling_depth3_real_committed_proposal_ids",
        "rolling_depth3_real_commit_skipped_proposal_ids",
        "rolling_depth3_real_commit_duplicate_proposal_ids",
        "rolling_depth3_real_commit_duplicate_seq_ids",
        "rolling_depth3_committed_without_ready_shadow_ids",
        "rolling_depth3_committed_without_parent_depth2_commit_ids",
        "rolling_depth3_committed_invalidated_child_ids",
        "rolling_depth3_committed_cascade_discarded_child_ids",
        "rolling_depth3_committed_non_full_accept_ids",
    )
    if any(as_int_set(record.get(field)) for field in fields):
        return True
    count_fields = (
        "rolling_depth3_real_commit_count",
        "rolling_depth3_real_committed_token_count",
        "rolling_depth4_real_commit_count",
        "rolling_depth_gt3_real_commit_count",
    )
    return any(int_value(record.get(field), 0) != 0 for field in count_fields)


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    accounting = aggregate_performance_accounting(records, {})
    gamma = 0
    commit_enabled_records = 0
    active_records = 0
    ready_ids: set[int] = set()
    generated_ids: set[int] = set()
    invalidated_ids: set[int] = set()
    cascade_ids: set[int] = set()
    parent_committed_ids: set[int] = set()
    parent_full_ids: set[int] = set()
    parent_skipped_ids: set[int] = set()
    parent_invalidated_ids: set[int] = set()
    parent_pending_ids: set[int] = set()
    committed_ids: set[int] = set()
    skipped_ids: set[int] = set()
    candidate_ids: set[int] = set()
    parent_by_child: dict[int, int] = {}
    root_by_child: dict[int, int] = {}
    depth_by_child: dict[int, int] = {}
    seq_by_child: dict[int, int] = {}
    token_by_child: dict[int, int] = {}
    commit_token_by_id: dict[int, int] = {}
    commit_accept_by_id: dict[int, int] = {}
    commit_parent_by_id: dict[int, int] = {}
    commit_root_by_id: dict[int, int] = {}
    commit_depth_by_id: dict[int, int] = {}
    commit_action_by_id: dict[int, str] = {}
    commit_result_by_id: dict[int, str] = {}
    precondition_ok_by_id: dict[int, bool] = {}
    parent_depth_by_id: dict[int, int] = {}
    parent_action_by_id: dict[int, str] = {}
    parent_result_by_id: dict[int, str] = {}
    parent_seq_by_id: dict[int, int] = {}
    normal_lane_conflict_count = 0
    missing_unexpected_count = 0
    real_commit_count = 0
    depth4_real_commit_count = 0
    depth_gt3_real_commit_count = 0
    depth4_commit_enabled = False
    target_verified_sum = 0
    target_accepted_sum = 0
    target_rejected_sum = 0
    target_invalidated_sum = 0
    draft_verified_sum = 0
    draft_accepted_sum = 0
    draft_rejected_sum = 0
    draft_invalidated_sum = 0
    counted_records: set[tuple[str, int, int]] = set()
    duplicate_proposal_ids: set[int] = set()
    duplicate_seq_ids: set[int] = set()
    without_ready_ids: set[int] = set()
    without_parent_ids: set[int] = set()
    committed_invalidated_ids: set[int] = set()
    committed_cascade_ids: set[int] = set()
    committed_non_full_ids: set[int] = set()
    skip_reason_by_id: dict[int, str] = {}

    for idx, record in enumerate(records):
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        enabled = bool(record.get("enable_rolling_continuous_depth3_commit_ready_only", False))
        enabled = enabled or bool(record.get("rolling_depth3_commit_enabled", False))
        if enabled:
            commit_enabled_records += 1
        depth4_commit_enabled = depth4_commit_enabled or bool(
            record.get("enable_rolling_continuous_depth4_commit_ready_only", False)
        ) or bool(record.get("rolling_depth4_commit_enabled", False))
        active = commit_active(record)
        if active:
            active_records += 1
        if active and not enabled:
            errors.append(f"record[{idx}] rolling depth3 commit active while flag disabled")
        if record.get("rolling_depth3_commit_source") not in {None, ROLLING_DEPTH3_COMMIT_SOURCE}:
            errors.append(f"record[{idx}] bad rolling depth3 commit source {record.get('rolling_depth3_commit_source')!r}")
        if as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")):
            missing_unexpected_count += len(as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")))
            errors.append(f"record[{idx}] unexpected missing buffered proposal")
        if as_int_set(record.get("rolling_depth3_normal_lane_conflict_seq_ids")):
            normal_lane_conflict_count += len(as_int_set(record.get("rolling_depth3_normal_lane_conflict_seq_ids")))
            errors.append(f"record[{idx}] rolling depth3 normal lane conflict present")
        normal_lane_conflict_count += int_value(record.get("rolling_depth3_normal_lane_conflict_count"), 0)
        depth4_real_commit_count += int_value(record.get("rolling_depth4_real_commit_count"), 0)
        depth_gt3_real_commit_count += int_value(record.get("rolling_depth_gt3_real_commit_count"), 0)
        if int_value(record.get("rolling_depth4_real_commit_count"), 0) and not depth4_commit_enabled:
            errors.append(f"record[{idx}] rolling depth4 real commit count must be zero")
        if int_value(record.get("rolling_depth_gt3_real_commit_count"), 0) and not depth4_commit_enabled:
            errors.append(f"record[{idx}] rolling depth>3 real commit count must be zero")
        if int_value(record.get("rolling_depth3_real_commit_count"), 0) and not enabled:
            errors.append(f"record[{idx}] depth3 real commit count requires commit flag")

        generated_ids.update(as_int_set(record.get("rolling_depth3_child_generated_proposal_ids")))
        ready_ids.update(as_int_set(record.get("rolling_depth3_child_ready_shadow_proposal_ids")))
        invalidated_ids.update(as_int_set(record.get("rolling_depth3_child_invalidated_proposal_ids")))
        cascade_ids.update(as_int_set(record.get("rolling_depth3_committed_cascade_discarded_child_ids")))
        parent_committed_ids.update(as_int_set(record.get("rolling_depth2_real_committed_proposal_ids")))
        parent_committed_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_real_committed_proposal_ids")))
        parent_full_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_full_accept_proposal_ids")))
        parent_skipped_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_skipped_proposal_ids")))
        parent_invalidated_ids.update(as_int_set(record.get("rolling_depth3_parent_depth2_invalidated_proposal_ids")))
        parent_pending_ids.update(as_int_set(record.get("rolling_depth3_parent_resolution_pending_proposal_ids")))
        parent_invalidated_ids.update(as_int_set(record.get("rolling_child_invalidated_proposal_ids")))
        parent_invalidated_ids.update(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
        committed_record_ids = as_int_set(record.get("rolling_depth3_real_committed_proposal_ids"))
        committed_ids.update(committed_record_ids)
        candidate_ids.update(as_int_set(record.get("rolling_depth3_commit_candidate_proposal_ids")))
        skipped_ids.update(as_int_set(record.get("rolling_depth3_real_commit_skipped_proposal_ids")))
        duplicate_proposal_ids.update(as_int_set(record.get("rolling_depth3_real_commit_duplicate_proposal_ids")))
        duplicate_seq_ids.update(as_int_set(record.get("rolling_depth3_real_commit_duplicate_seq_ids")))
        without_ready_ids.update(as_int_set(record.get("rolling_depth3_committed_without_ready_shadow_ids")))
        without_parent_ids.update(as_int_set(record.get("rolling_depth3_committed_without_parent_depth2_commit_ids")))
        committed_invalidated_ids.update(as_int_set(record.get("rolling_depth3_committed_invalidated_child_ids")))
        committed_cascade_ids.update(as_int_set(record.get("rolling_depth3_committed_cascade_discarded_child_ids")))
        committed_non_full_ids.update(as_int_set(record.get("rolling_depth3_committed_non_full_accept_ids")))

        parent_by_child.update(as_int_map(record.get("rolling_depth3_child_parent_by_proposal_id")))
        root_by_child.update(as_int_map(record.get("rolling_depth3_child_root_by_proposal_id")))
        depth_by_child.update(as_int_map(record.get("rolling_depth3_child_depth_by_proposal_id")))
        token_by_child.update(
            {
                proposal_id: count
                for proposal_id, count in as_int_map(record.get("rolling_depth3_child_token_count_by_proposal_id")).items()
                if count > 0
            }
        )
        for proposal_id, seq_id in zip(
            as_int_list(record.get("rolling_depth3_child_generated_proposal_ids")),
            as_int_list(record.get("rolling_depth3_child_generated_seq_ids")),
        ):
            seq_by_child.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(
            as_int_list(record.get("rolling_depth3_child_ready_shadow_proposal_ids")),
            as_int_list(record.get("rolling_depth3_child_ready_shadow_seq_ids")),
        ):
            seq_by_child.setdefault(proposal_id, seq_id)
        for proposal_id, seq_id in zip(
            as_int_list(record.get("rolling_depth2_real_committed_proposal_ids")),
            as_int_list(record.get("rolling_depth2_real_committed_seq_ids")),
        ):
            parent_seq_by_id.setdefault(proposal_id, seq_id)
        parent_depth_by_id.update(as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id")))
        for proposal_id, action in as_str_map(record.get("rolling_depth2_real_commit_action_by_proposal_id")).items():
            parent_action_by_id.setdefault(proposal_id, action)
        for proposal_id, result in as_str_map(record.get("rolling_depth2_real_commit_verify_result_by_proposal_id")).items():
            parent_result_by_id.setdefault(proposal_id, result)
            if result == "full_accept":
                parent_full_ids.add(proposal_id)

        commit_token_by_id.update(as_int_map(record.get("rolling_depth3_real_committed_token_count_by_proposal_id")))
        commit_accept_by_id.update(as_int_map(record.get("rolling_depth3_real_committed_accept_len_by_proposal_id")))
        commit_parent_by_id.update(as_int_map(record.get("rolling_depth3_real_commit_parent_by_proposal_id")))
        commit_root_by_id.update(as_int_map(record.get("rolling_depth3_real_commit_root_by_proposal_id")))
        commit_depth_by_id.update(as_int_map(record.get("rolling_depth3_real_commit_depth_by_proposal_id")))
        for proposal_id, action in as_str_map(record.get("rolling_depth3_real_commit_action_by_proposal_id")).items():
            commit_action_by_id.setdefault(proposal_id, action)
        for proposal_id, result in as_str_map(record.get("rolling_depth3_real_commit_verify_result_by_proposal_id")).items():
            commit_result_by_id.setdefault(proposal_id, result)
        for proposal_id, ok in as_bool_map(record.get("rolling_depth3_commit_precondition_ok_by_proposal_id")).items():
            if ok or proposal_id not in precondition_ok_by_id:
                precondition_ok_by_id[proposal_id] = ok
        for proposal_id, reason in as_str_map(record.get("rolling_depth3_real_commit_skip_reason_by_proposal_id")).items():
            skip_reason_by_id.setdefault(proposal_id, reason)

        record_count = int_value(record.get("rolling_depth3_real_commit_count"), len(committed_record_ids))
        real_commit_count += record_count
        if committed_record_ids and record_count != len(committed_record_ids):
            errors.append(f"record[{idx}] depth3 committed count/list mismatch")
        expected_record_tokens = sum(
            token_count(
                proposal_id,
                as_int_map(record.get("rolling_depth3_real_committed_token_count_by_proposal_id")),
                gamma,
            )
            for proposal_id in committed_record_ids
        )
        if committed_record_ids and int_value(
            record.get("rolling_depth3_real_committed_token_count"),
            expected_record_tokens,
        ) != expected_record_tokens:
            errors.append(f"record[{idx}] depth3 committed token count mismatch")

        side = str(record.get("rolling_depth3_commit_side") or "")
        if committed_record_ids and side in {"target", "draft"}:
            plan_id = int_value(record.get("rolling_depth3_commit_plan_id"), int_value(record.get("plan_id"), -1))
            step_id = int_value(record.get("rolling_depth3_commit_step_id"), int_value(record.get("step_id"), -1))
            record_key = (side, plan_id, step_id)
            if record_key not in counted_records:
                counted_records.add(record_key)
                if side == "target":
                    target_verified_sum += int_value(record.get("rolling_depth3_tokens_verified"), expected_record_tokens)
                    target_accepted_sum += int_value(record.get("rolling_depth3_tokens_accepted"), expected_record_tokens)
                    target_rejected_sum += int_value(record.get("rolling_depth3_tokens_rejected"), 0)
                    target_invalidated_sum += int_value(record.get("rolling_depth3_tokens_invalidated"), 0)
                else:
                    draft_verified_sum += int_value(record.get("rolling_depth3_tokens_verified"), expected_record_tokens)
                    draft_accepted_sum += int_value(record.get("rolling_depth3_tokens_accepted"), expected_record_tokens)
                    draft_rejected_sum += int_value(record.get("rolling_depth3_tokens_rejected"), 0)
                    draft_invalidated_sum += int_value(record.get("rolling_depth3_tokens_invalidated"), 0)

        for field in ("rolling_depth3_target_draft_len_match_by_seq_id", "rolling_depth3_target_draft_token_match_by_seq_id"):
            for seq_id, ok in as_bool_map(record.get(field)).items():
                if not ok:
                    errors.append(f"record[{idx}] {field} false for seq {seq_id}")

    if committed_ids and not commit_enabled_records:
        errors.append("rolling depth3 commits require depth3 commit flag")
    if duplicate_proposal_ids:
        errors.append(f"duplicate rolling depth3 proposal commits: {sorted(duplicate_proposal_ids)}")
    if duplicate_seq_ids:
        errors.append(f"duplicate rolling depth3 seq/depth commits: {sorted(duplicate_seq_ids)}")
    if without_ready_ids:
        errors.append(f"rolling depth3 commits without ready shadow: {sorted(without_ready_ids)}")
    if without_parent_ids:
        errors.append(f"rolling depth3 commits without depth2 parent commit: {sorted(without_parent_ids)}")
    if committed_invalidated_ids:
        errors.append(f"rolling depth3 invalidated children committed: {sorted(committed_invalidated_ids)}")
    if committed_cascade_ids:
        errors.append(f"rolling depth3 cascade-discarded children committed: {sorted(committed_cascade_ids)}")
    if committed_non_full_ids:
        errors.append(f"rolling depth3 non-full-accept children committed: {sorted(committed_non_full_ids)}")

    for proposal_id in sorted(committed_ids):
        parent_id = commit_parent_by_id.get(proposal_id, parent_by_child.get(proposal_id))
        if proposal_id not in ready_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} was not ready shadow")
        if proposal_id not in generated_ids and proposal_id not in ready_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} was not generated/ready")
        if proposal_id in invalidated_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} was invalidated")
        if proposal_id in cascade_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} was cascade-discarded")
        if parent_id is None:
            errors.append(f"rolling depth3 committed proposal {proposal_id} missing parent")
            continue
        if parent_id not in parent_committed_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} not committed")
        if parent_id in parent_skipped_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} skipped")
        if parent_id in parent_invalidated_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} invalidated")
        if parent_id in parent_pending_ids:
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} pending")
        if parent_depth_by_id.get(parent_id, 2) != 2:
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} depth is not 2")
        if parent_result_by_id.get(parent_id, "full_accept") != "full_accept":
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} not full_accept")
        if parent_action_by_id.get(parent_id, ROLLING_ACTION) != ROLLING_ACTION:
            errors.append(f"rolling depth3 committed proposal {proposal_id} parent {parent_id} action is not real commit")
        if commit_depth_by_id.get(proposal_id, depth_by_child.get(proposal_id, 0)) != 3:
            errors.append(f"rolling depth3 committed proposal {proposal_id} depth is not 3")
        if depth_by_child.get(proposal_id, 3) != 3:
            errors.append(f"rolling depth3 committed proposal {proposal_id} child depth is not 3")
        if proposal_id in root_by_child and proposal_id in commit_root_by_id:
            if root_by_child[proposal_id] != commit_root_by_id[proposal_id]:
                errors.append(f"rolling depth3 committed proposal {proposal_id} root mismatch")
        if parent_id in parent_seq_by_id and proposal_id in seq_by_child:
            if parent_seq_by_id[parent_id] != seq_by_child[proposal_id]:
                errors.append(f"rolling depth3 committed proposal {proposal_id} seq differs from parent")
        if commit_action_by_id.get(proposal_id, ROLLING_ACTION) != ROLLING_ACTION:
            errors.append(f"rolling depth3 committed proposal {proposal_id} action is not append real commit")
        if commit_result_by_id.get(proposal_id, "full_accept") != "full_accept":
            errors.append(f"rolling depth3 committed proposal {proposal_id} result is not full_accept")
        if precondition_ok_by_id.get(proposal_id, True) is not True:
            errors.append(f"rolling depth3 committed proposal {proposal_id} failed precondition")
        token_value = token_count(proposal_id, commit_token_by_id or token_by_child, gamma)
        accept_value = commit_accept_by_id.get(proposal_id, token_value)
        if token_value <= 0 or accept_value != token_value:
            errors.append(f"rolling depth3 committed proposal {proposal_id} token/accept mismatch")

    committed_token_count = sum(
        token_count(proposal_id, commit_token_by_id or token_by_child, gamma)
        for proposal_id in committed_ids
    )
    if committed_token_count != target_verified_sum:
        errors.append("rolling depth3 committed tokens must equal target verified increment sum")
    if committed_token_count != target_accepted_sum:
        errors.append("rolling depth3 committed tokens must equal target accepted increment sum")
    if committed_token_count != draft_verified_sum:
        errors.append("rolling depth3 committed tokens must equal draft verified increment sum")
    if committed_token_count != draft_accepted_sum:
        errors.append("rolling depth3 committed tokens must equal draft accepted increment sum")
    if target_rejected_sum or target_invalidated_sum or draft_rejected_sum or draft_invalidated_sum:
        errors.append("rolling depth3 rejected/invalidated increments must be zero")
    if depth4_real_commit_count and not depth4_commit_enabled:
        errors.append("rolling_depth4_real_commit_count must be zero")
    if depth_gt3_real_commit_count and not depth4_commit_enabled:
        errors.append("rolling_depth_gt3_real_commit_count must be zero")
    if normal_lane_conflict_count:
        errors.append("rolling_depth3_normal_lane_conflict_count must be zero")
    if missing_unexpected_count:
        errors.append("missing_buffered_proposal_unexpected_count must be zero")
    combined_expected = (
        int_value(accounting.get("eager_committed_token_count"), 0)
        + int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
        + committed_token_count
        + int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0)
    )
    if int_value(accounting.get("combined_real_committed_token_count"), 0) != combined_expected:
        errors.append("combined real committed token count must equal one-shot + depth1 + depth2 + depth3 + optional depth4")

    summary = {
        "total_trace_records": len(records),
        "records_with_rolling_depth3_commit_enabled": commit_enabled_records,
        "rolling_depth3_commit_active_records": active_records,
        "rolling_depth3_commit_candidate_proposal_count": len(candidate_ids),
        "rolling_depth3_child_ready_shadow_proposal_count": len(ready_ids),
        "rolling_depth3_real_committed_proposal_count": len(committed_ids),
        "rolling_depth3_real_committed_token_count": committed_token_count,
        "combined_real_committed_token_count": accounting.get("combined_real_committed_token_count", 0),
        "rolling_depth3_real_commit_count": real_commit_count,
        "rolling_depth4_real_commit_count": depth4_real_commit_count,
        "rolling_depth_gt3_real_commit_count": depth_gt3_real_commit_count,
        "rolling_depth3_target_actual_verified_token_increment_sum": target_verified_sum,
        "rolling_depth3_target_actual_accepted_token_increment_sum": target_accepted_sum,
        "rolling_depth3_draft_actual_verified_token_increment_sum": draft_verified_sum,
        "rolling_depth3_draft_actual_accepted_token_increment_sum": draft_accepted_sum,
        "rolling_depth3_normal_lane_conflict_count": normal_lane_conflict_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "rolling_depth3_real_commit_skip_reason_counts": dict(Counter(skip_reason_by_id.values())),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_rolling_depth3_commit_enabled",
        "rolling_depth3_commit_active_records",
        "rolling_depth3_commit_candidate_proposal_count",
        "rolling_depth3_child_ready_shadow_proposal_count",
        "rolling_depth3_real_committed_proposal_count",
        "rolling_depth3_real_committed_token_count",
        "combined_real_committed_token_count",
        "rolling_depth3_real_commit_count",
        "rolling_depth4_real_commit_count",
        "rolling_depth_gt3_real_commit_count",
        "rolling_depth3_target_actual_verified_token_increment_sum",
        "rolling_depth3_target_actual_accepted_token_increment_sum",
        "rolling_depth3_draft_actual_verified_token_increment_sum",
        "rolling_depth3_draft_actual_accepted_token_increment_sum",
        "rolling_depth3_normal_lane_conflict_count",
        "missing_buffered_proposal_unexpected_count",
        "rolling_depth3_real_commit_skip_reason_counts",
    ):
        print(f"{key} = {summary.get(key)}")


def synthetic_good_record() -> dict[str, Any]:
    parent_id = 900000102
    child_id = 900000103
    seq_id = 7
    root_id = 101
    return {
        "normal_gamma": 4,
        "enable_rolling_continuous_depth3_shadow_dry_run": True,
        "enable_rolling_continuous_depth3_commit_ready_only": True,
        "rolling_depth3_commit_enabled": True,
        "rolling_depth3_commit_source": ROLLING_DEPTH3_COMMIT_SOURCE,
        "rolling_depth3_commit_side": "target",
        "rolling_depth3_commit_plan_id": 1,
        "rolling_depth3_commit_step_id": 2,
        "rolling_depth2_real_committed_proposal_ids": [parent_id],
        "rolling_depth2_real_committed_seq_ids": [seq_id],
        "rolling_depth2_real_commit_depth_by_proposal_id": {str(parent_id): 2},
        "rolling_depth2_real_commit_action_by_proposal_id": {str(parent_id): ROLLING_ACTION},
        "rolling_depth2_real_commit_verify_result_by_proposal_id": {str(parent_id): "full_accept"},
        "rolling_depth3_parent_depth2_real_committed_proposal_ids": [parent_id],
        "rolling_depth3_parent_depth2_full_accept_proposal_ids": [parent_id],
        "rolling_depth3_child_generated_proposal_ids": [child_id],
        "rolling_depth3_child_generated_seq_ids": [seq_id],
        "rolling_depth3_child_ready_shadow_proposal_ids": [child_id],
        "rolling_depth3_child_ready_shadow_seq_ids": [seq_id],
        "rolling_depth3_child_parent_by_proposal_id": {str(child_id): parent_id},
        "rolling_depth3_child_root_by_proposal_id": {str(child_id): root_id},
        "rolling_depth3_child_depth_by_proposal_id": {str(child_id): 3},
        "rolling_depth3_child_token_count_by_proposal_id": {str(child_id): 4},
        "rolling_depth3_commit_candidate_proposal_ids": [child_id],
        "rolling_depth3_commit_candidate_seq_ids": [seq_id],
        "rolling_depth3_commit_ready_source_proposal_ids": [child_id],
        "rolling_depth3_commit_parent_by_proposal_id": {str(child_id): parent_id},
        "rolling_depth3_commit_precondition_ok_by_proposal_id": {str(child_id): True},
        "rolling_depth3_real_committed_proposal_ids": [child_id],
        "rolling_depth3_real_committed_seq_ids": [seq_id],
        "rolling_depth3_real_committed_token_count_by_proposal_id": {str(child_id): 4},
        "rolling_depth3_real_committed_accept_len_by_proposal_id": {str(child_id): 4},
        "rolling_depth3_real_commit_action_by_proposal_id": {str(child_id): ROLLING_ACTION},
        "rolling_depth3_real_commit_verify_result_by_proposal_id": {str(child_id): "full_accept"},
        "rolling_depth3_real_commit_parent_by_proposal_id": {str(child_id): parent_id},
        "rolling_depth3_real_commit_root_by_proposal_id": {str(child_id): root_id},
        "rolling_depth3_real_commit_depth_by_proposal_id": {str(child_id): 3},
        "rolling_depth3_tokens_verified": 4,
        "rolling_depth3_tokens_accepted": 4,
        "rolling_depth3_tokens_committed": 4,
        "rolling_depth3_tokens_rejected": 0,
        "rolling_depth3_tokens_invalidated": 0,
        "rolling_depth3_real_committed_proposal_count": 1,
        "rolling_depth3_real_committed_token_count": 4,
        "rolling_depth3_real_commit_count": 1,
        "rolling_depth4_real_commit_count": 0,
        "rolling_depth_gt3_real_commit_count": 0,
        "rolling_depth3_normal_lane_conflict_count": 0,
        "missing_buffered_proposal_unexpected_seq_ids": [],
    }


def run_synthetic() -> None:
    target = synthetic_good_record()
    draft = deepcopy(target)
    draft["rolling_depth3_commit_side"] = "draft"
    errors, summary = validate_records([target, draft])
    if errors:
        raise SystemExit(f"synthetic depth3 commit good case failed: {errors}\nsummary={summary}")

    no_ready = deepcopy(target)
    no_ready["rolling_depth3_child_ready_shadow_proposal_ids"] = []
    errors, _summary = validate_records([no_ready, {**draft, "rolling_depth3_child_ready_shadow_proposal_ids": []}])
    if not errors:
        raise SystemExit("synthetic depth3 commit without ready shadow should fail")

    no_parent = deepcopy(target)
    no_parent["rolling_depth2_real_committed_proposal_ids"] = []
    no_parent["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    no_parent_draft = deepcopy(draft)
    no_parent_draft["rolling_depth2_real_committed_proposal_ids"] = []
    no_parent_draft["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    errors, _summary = validate_records([no_parent, no_parent_draft])
    if not errors:
        raise SystemExit("synthetic depth3 commit without parent should fail")

    invalidated = deepcopy(target)
    invalidated["rolling_depth3_child_invalidated_proposal_ids"] = [900000103]
    errors, _summary = validate_records([invalidated, draft])
    if not errors:
        raise SystemExit("synthetic invalidated depth3 child commit should fail")

    depth4 = deepcopy(target)
    depth4["rolling_depth4_real_commit_count"] = 1
    depth4["rolling_depth_gt3_real_commit_count"] = 1
    errors, _summary = validate_records([depth4, draft])
    if not errors:
        raise SystemExit("synthetic depth4 commit should fail")

    parent_record = deepcopy(target)
    ready_record = deepcopy(target)
    commit_record = deepcopy(target)
    parent_record["rolling_depth3_child_ready_shadow_proposal_ids"] = []
    parent_record["rolling_depth3_real_committed_proposal_ids"] = []
    parent_record["rolling_depth3_real_commit_count"] = 0
    parent_record["rolling_depth3_tokens_verified"] = 0
    parent_record["rolling_depth3_tokens_accepted"] = 0
    ready_record["rolling_depth2_real_committed_proposal_ids"] = []
    ready_record["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    ready_record["rolling_depth3_real_committed_proposal_ids"] = []
    ready_record["rolling_depth3_real_commit_count"] = 0
    ready_record["rolling_depth3_tokens_verified"] = 0
    ready_record["rolling_depth3_tokens_accepted"] = 0
    commit_record["rolling_depth2_real_committed_proposal_ids"] = []
    commit_record["rolling_depth3_parent_depth2_real_committed_proposal_ids"] = []
    errors, _summary = validate_records([parent_record, ready_record, commit_record, {**commit_record, "rolling_depth3_commit_side": "draft"}])
    if errors:
        raise SystemExit(f"synthetic distributed depth3 commit evidence should pass: {errors}")

    errors, summary = validate_records([target, deepcopy(target), draft, deepcopy(draft)])
    if errors:
        raise SystemExit(f"synthetic duplicate side rows should pass: {errors}")
    if summary.get("rolling_depth3_real_committed_token_count") != 4:
        raise SystemExit("synthetic duplicate side rows should dedupe committed tokens")

    print("Synthetic rolling continuous depth-3 commit checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-8d rolling depth-3 real commit traces.")
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
