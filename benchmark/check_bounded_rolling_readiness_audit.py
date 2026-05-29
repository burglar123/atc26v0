#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    float_value,
    int_value,
    load_json,
)
from benchmark.bounded_rolling_chain_parser import (  # noqa: E402
    parse_legacy_rolling_chain,
    summarize_registry,
)


ONE_SHOT_ACTION = "append_full_accept_then_rollback"
ROLLING_ACTION = "append_full_accept_real_commit"
MAX_AUDITED_REAL_DEPTH = 4

GENERIC_PARITY_FIELD_PAIRS = (
    ("one_shot_committed_token_count", "generic_one_shot_committed_token_count"),
    ("depth1_committed_token_count", "generic_depth1_committed_token_count"),
    ("depth2_committed_token_count", "generic_depth2_committed_token_count"),
    ("depth3_committed_token_count", "generic_depth3_committed_token_count"),
    ("depth4_committed_token_count", "generic_depth4_committed_token_count"),
    ("partial_prefix_recovery_enabled", "generic_partial_prefix_recovery_enabled"),
    ("partial_prefix_recovery_success_count", "generic_partial_prefix_recovery_success_count"),
    ("partial_prefix_accepted_token_count", "generic_partial_prefix_accepted_token_count"),
    ("partial_prefix_revised_token_count", "generic_partial_prefix_revised_token_count"),
    ("partial_prefix_total_recovered_token_count", "generic_partial_prefix_total_recovered_token_count"),
    ("combined_real_committed_token_count", "generic_combined_real_committed_token_count"),
    ("max_observed_depth", "generic_max_observed_depth"),
    ("max_real_committed_depth", "generic_max_real_committed_depth"),
    ("depth4_real_commit_count", "generic_depth4_real_commit_count"),
    ("depth_gt3_real_commit_count", "generic_depth_gt3_real_commit_count"),
    ("depth_gt4_real_commit_count", "generic_depth_gt4_real_commit_count"),
    ("normal_lane_conflict_count", "generic_normal_lane_conflict_count"),
    ("missing_buffered_proposal_unexpected_count", "generic_missing_buffered_proposal_unexpected_count"),
    ("duplicate_commit_count", "generic_duplicate_commit_count"),
    ("invalid_committed_child_count", "generic_invalid_committed_child_count"),
    ("cascade_committed_child_count", "generic_cascade_committed_child_count"),
    ("parent_missing_committed_child_count", "generic_parent_missing_committed_child_count"),
    ("combined_accounting_ok", "generic_combined_accounting_ok"),
    ("target_draft_accounting_ok", "generic_target_draft_accounting_ok"),
)


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


def merge_first(target: dict[int, int], source: dict[int, int]) -> None:
    for key, value in source.items():
        target.setdefault(key, value)


def merge_positive(target: dict[int, int], source: dict[int, int]) -> None:
    for key, value in source.items():
        if value > 0:
            target.setdefault(key, value)


def zip_map(ids: list[int], values: list[int]) -> dict[int, int]:
    return {proposal_id: value for proposal_id, value in zip(ids, values)}


def add_seq_map(
    seq_by_depth: dict[int, dict[int, int]],
    depth: int,
    proposal_ids: list[int],
    seq_ids: list[int],
) -> None:
    for proposal_id, seq_id in zip(proposal_ids, seq_ids):
        seq_by_depth[depth].setdefault(proposal_id, seq_id)


def add_side_events(
    events: dict[tuple[int, str, int], set[tuple[int, int]]],
    record: dict[str, Any],
    *,
    depth: int,
    side_field: str,
    plan_field: str,
    step_field: str,
    id_field: str,
) -> None:
    side = str(record.get(side_field) or "")
    if side not in {"target", "draft"}:
        return
    plan_id = int_value(record.get(plan_field), int_value(record.get("plan_id"), -1))
    step_id = int_value(record.get(step_field), int_value(record.get("step_id"), -1))
    for proposal_id in as_int_set(record.get(id_field)):
        events[(depth, side, proposal_id)].add((plan_id, step_id))


def bool_false_count(value: Any) -> int:
    return sum(1 for item in as_bool_map(value).values() if item is False)


def dict_field_false_ids(value: Any) -> set[int]:
    return {key for key, item in as_bool_map(value).items() if item is False}


def proposal_tokens(ids: set[int], token_by_depth: dict[int, dict[int, int]], depth: int, gamma: int) -> int:
    return sum(max(0, int(token_by_depth[depth].get(proposal_id, gamma))) for proposal_id in ids)


def collect_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    committed_by_depth: dict[int, set[int]] = defaultdict(set)
    token_by_depth: dict[int, dict[int, int]] = defaultdict(dict)
    seq_by_depth: dict[int, dict[int, int]] = defaultdict(dict)
    parent_by_depth: dict[int, dict[int, int]] = defaultdict(dict)
    root_by_depth: dict[int, dict[int, int]] = defaultdict(dict)
    declared_depth_by_id: dict[int, int] = {}
    action_by_depth: dict[int, dict[int, str]] = defaultdict(dict)
    result_by_depth: dict[int, dict[int, str]] = defaultdict(dict)
    accept_by_depth: dict[int, dict[int, int]] = defaultdict(dict)
    ready_by_depth: dict[int, set[int]] = defaultdict(set)
    generated_by_depth: dict[int, set[int]] = defaultdict(set)
    invalidated_ids: set[int] = set()
    cascade_ids: set[int] = set()
    stale_expired_ids: set[int] = set()
    frontier_mismatch_ids: set[int] = set()
    side_events: dict[tuple[int, str, int], set[tuple[int, int]]] = defaultdict(set)
    seq_depth_events: set[tuple[int, str, int, int, int]] = set()
    duplicate_seq_depth_events: set[tuple[int, str, int, int, int]] = set()
    explicit_duplicate_ids: set[int] = set()
    explicit_duplicate_seq_ids: set[int] = set()
    explicit_without_ready: set[int] = set()
    explicit_without_parent: set[int] = set()
    explicit_invalid_committed: set[int] = set()
    explicit_cascade_committed: set[int] = set()
    explicit_non_full: set[int] = set()
    higher_depth_commit_ids: set[int] = set()
    length_mismatch_count = 0
    token_mismatch_count = 0
    normal_lane_conflict_count = 0
    missing_unexpected_count = 0
    depth4_real_commit_count = 0
    depth_gt3_real_commit_count = 0
    depth_gt4_real_commit_count = 0
    max_configured_depth = 0
    max_observed_depth = 0
    gamma = 0
    flags = {
        "one_shot_commit_enabled": False,
        "continuous_depth1_commit_enabled": False,
        "rolling_depth2_commit_enabled": False,
        "rolling_depth3_shadow_enabled": False,
        "rolling_depth3_commit_enabled": False,
        "rolling_depth4_shadow_enabled": False,
        "rolling_depth4_commit_enabled": False,
    }

    for record in records:
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        max_configured_depth = max(max_configured_depth, int_value(record.get("max_rolling_continuous_depth"), 0))
        max_observed_depth = max(
            max_observed_depth,
            int_value(record.get("rolling_max_depth_observed"), 0),
            int_value(record.get("max_rolling_continuous_depth_observed"), 0),
            int_value(record.get("rolling_depth3_max_depth_observed"), 0),
            int_value(record.get("rolling_depth4_max_depth_observed"), 0),
        )
        flags["one_shot_commit_enabled"] = flags["one_shot_commit_enabled"] or bool(
            record.get("enable_eager_commit_ready_only", False)
        ) or bool(record.get("eager_commit_enabled", False))
        flags["continuous_depth1_commit_enabled"] = flags["continuous_depth1_commit_enabled"] or bool(
            record.get("enable_continuous_eager_commit_depth1_ready_only", False)
        ) or bool(record.get("continuous_eager_commit_enabled", False))
        flags["rolling_depth2_commit_enabled"] = flags["rolling_depth2_commit_enabled"] or bool(
            record.get("enable_rolling_continuous_depth2_commit_ready_only", False)
        ) or bool(record.get("rolling_depth2_commit_enabled", False))
        flags["rolling_depth3_shadow_enabled"] = flags["rolling_depth3_shadow_enabled"] or bool(
            record.get("enable_rolling_continuous_depth3_shadow_dry_run", False)
        ) or bool(record.get("rolling_depth3_shadow_enabled", False))
        flags["rolling_depth3_commit_enabled"] = flags["rolling_depth3_commit_enabled"] or bool(
            record.get("enable_rolling_continuous_depth3_commit_ready_only", False)
        ) or bool(record.get("rolling_depth3_commit_enabled", False))
        flags["rolling_depth4_shadow_enabled"] = flags["rolling_depth4_shadow_enabled"] or bool(
            record.get("enable_rolling_continuous_depth4_shadow_dry_run", False)
        ) or bool(record.get("rolling_depth4_shadow_enabled", False))
        flags["rolling_depth4_commit_enabled"] = flags["rolling_depth4_commit_enabled"] or bool(
            record.get("enable_rolling_continuous_depth4_commit_ready_only", False)
        ) or bool(record.get("rolling_depth4_commit_enabled", False))

        missing_unexpected_count += int_value(record.get("missing_buffered_proposal_unexpected_count"), 0)
        missing_unexpected_count += len(as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids")))
        normal_lane_conflict_count += int_value(record.get("rolling_normal_lane_conflict_count"), 0)
        normal_lane_conflict_count += len(as_int_set(record.get("rolling_normal_lane_conflict_seq_ids")))
        normal_lane_conflict_count += int_value(record.get("rolling_depth3_normal_lane_conflict_count"), 0)
        normal_lane_conflict_count += len(as_int_set(record.get("rolling_depth3_normal_lane_conflict_seq_ids")))
        normal_lane_conflict_count += int_value(record.get("rolling_depth4_normal_lane_conflict_count"), 0)
        normal_lane_conflict_count += len(as_int_set(record.get("rolling_depth4_normal_lane_conflict_seq_ids")))
        depth4_real_commit_count += int_value(record.get("rolling_depth4_real_commit_count"), 0)
        depth_gt3_real_commit_count += int_value(record.get("rolling_depth_gt3_real_commit_count"), 0)
        depth_gt4_real_commit_count += int_value(record.get("rolling_depth_gt4_real_commit_count"), 0)
        higher_depth_commit_ids.update(as_int_set(record.get("rolling_depth_gt3_real_committed_proposal_ids")))
        higher_depth_commit_ids.update(as_int_set(record.get("rolling_depth_gt4_real_committed_proposal_ids")))
        length_mismatch_count += bool_false_count(record.get("eager_commit_target_draft_len_match_by_seq_id"))
        length_mismatch_count += bool_false_count(record.get("continuous_eager_target_draft_len_match_by_seq_id"))
        length_mismatch_count += bool_false_count(record.get("rolling_depth2_target_draft_len_match_by_seq_id"))
        length_mismatch_count += bool_false_count(record.get("rolling_depth3_target_draft_len_match_by_seq_id"))
        length_mismatch_count += bool_false_count(record.get("rolling_depth4_target_draft_len_match_by_seq_id"))
        token_mismatch_count += bool_false_count(record.get("eager_commit_target_draft_token_match_by_seq_id"))
        token_mismatch_count += bool_false_count(record.get("continuous_eager_target_draft_token_match_by_seq_id"))
        token_mismatch_count += bool_false_count(record.get("rolling_depth2_target_draft_token_match_by_seq_id"))
        token_mismatch_count += bool_false_count(record.get("rolling_depth3_target_draft_token_match_by_seq_id"))
        token_mismatch_count += bool_false_count(record.get("rolling_depth4_target_draft_token_match_by_seq_id"))

        eager_ids = as_int_list(record.get("eager_committed_proposal_ids"))
        committed_by_depth[0].update(eager_ids)
        add_seq_map(seq_by_depth, 0, eager_ids, as_int_list(record.get("eager_committed_seq_ids")))
        merge_positive(token_by_depth[0], as_int_map(record.get("eager_committed_token_count_by_proposal_id")))
        merge_positive(accept_by_depth[0], as_int_map(record.get("eager_committed_accept_len_by_proposal_id")))
        for proposal_id in eager_ids:
            declared_depth_by_id.setdefault(proposal_id, 0)
            root_by_depth[0].setdefault(proposal_id, proposal_id)
        action_by_depth[0].update(as_str_map(record.get("eager_committed_action_by_proposal_id")))
        result_by_depth[0].update(as_str_map(record.get("eager_committed_verify_result_by_proposal_id")))
        explicit_duplicate_ids.update(as_int_set(record.get("eager_commit_duplicate_proposal_ids")))
        explicit_duplicate_seq_ids.update(as_int_set(record.get("eager_commit_duplicate_seq_ids")))
        explicit_non_full.update(
            proposal_id
            for proposal_id, result in as_str_map(record.get("eager_committed_verify_result_by_proposal_id")).items()
            if result != "full_accept"
        )
        add_side_events(
            side_events,
            record,
            depth=0,
            side_field="eager_commit_side",
            plan_field="eager_commit_plan_id",
            step_field="eager_commit_step_id",
            id_field="eager_committed_proposal_ids",
        )

        depth1_ids = as_int_list(record.get("continuous_eager_real_committed_proposal_ids"))
        committed_by_depth[1].update(depth1_ids)
        ready_by_depth[1].update(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
        generated_by_depth[1].update(as_int_set(record.get("continuous_eager_candidate_proposal_ids")))
        add_seq_map(seq_by_depth, 1, depth1_ids, as_int_list(record.get("continuous_eager_real_committed_seq_ids")))
        add_seq_map(
            seq_by_depth,
            1,
            as_int_list(record.get("continuous_eager_candidate_proposal_ids")),
            as_int_list(record.get("continuous_eager_candidate_seq_ids")),
        )
        merge_positive(token_by_depth[1], as_int_map(record.get("continuous_eager_real_committed_token_count_by_proposal_id")))
        merge_positive(token_by_depth[1], as_int_map(record.get("continuous_eager_candidate_token_count_by_proposal_id")))
        merge_positive(accept_by_depth[1], as_int_map(record.get("continuous_eager_real_committed_accept_len_by_proposal_id")))
        merge_first(parent_by_depth[1], as_int_map(record.get("continuous_eager_parent_proposal_id_by_proposal_id")))
        merge_first(root_by_depth[1], as_int_map(record.get("continuous_eager_root_proposal_id_by_proposal_id")))
        for proposal_id, depth in as_int_map(record.get("continuous_eager_chain_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
        action_by_depth[1].update(as_str_map(record.get("continuous_eager_real_commit_action_by_proposal_id")))
        result_by_depth[1].update(as_str_map(record.get("continuous_eager_real_commit_verify_result_by_proposal_id")))
        invalidated_ids.update(as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids")))
        invalidated_ids.update(as_int_set(record.get("continuous_eager_partial_reject_proposal_ids")))
        frontier_mismatch_ids.update(as_int_set(record.get("continuous_eager_frontier_mismatch_proposal_ids")))
        frontier_mismatch_ids.update(as_int_set(record.get("continuous_eager_true_frontier_mismatch_proposal_ids")))
        add_side_events(
            side_events,
            record,
            depth=1,
            side_field="continuous_eager_commit_side",
            plan_field="continuous_eager_commit_plan_id",
            step_field="continuous_eager_commit_step_id",
            id_field="continuous_eager_real_committed_proposal_ids",
        )

        depth2_ids = as_int_list(record.get("rolling_depth2_real_committed_proposal_ids"))
        committed_by_depth[2].update(depth2_ids)
        ready_by_depth[2].update(as_int_set(record.get("rolling_child_ready_after_parent_full_accept_proposal_ids")))
        generated_by_depth[2].update(as_int_set(record.get("rolling_child_generated_proposal_ids")))
        add_seq_map(seq_by_depth, 2, depth2_ids, as_int_list(record.get("rolling_depth2_real_committed_seq_ids")))
        add_seq_map(
            seq_by_depth,
            2,
            as_int_list(record.get("rolling_child_generated_proposal_ids")),
            as_int_list(record.get("draft_rolling_eager_draft_seq_ids")),
        )
        merge_positive(token_by_depth[2], as_int_map(record.get("rolling_depth2_real_committed_token_count_by_proposal_id")))
        merge_positive(accept_by_depth[2], as_int_map(record.get("rolling_depth2_real_committed_accept_len_by_proposal_id")))
        merge_first(parent_by_depth[2], as_int_map(record.get("rolling_chain_parent_by_proposal_id")))
        merge_first(parent_by_depth[2], as_int_map(record.get("rolling_depth2_real_commit_parent_by_proposal_id")))
        merge_first(root_by_depth[2], as_int_map(record.get("rolling_chain_root_by_proposal_id")))
        merge_first(root_by_depth[2], as_int_map(record.get("rolling_depth2_real_commit_root_by_proposal_id")))
        for proposal_id, depth in as_int_map(record.get("rolling_chain_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
        for proposal_id, depth in as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
        action_by_depth[2].update(as_str_map(record.get("rolling_depth2_real_commit_action_by_proposal_id")))
        result_by_depth[2].update(as_str_map(record.get("rolling_depth2_real_commit_verify_result_by_proposal_id")))
        invalidated_ids.update(as_int_set(record.get("rolling_child_invalidated_proposal_ids")))
        cascade_ids.update(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
        add_side_events(
            side_events,
            record,
            depth=2,
            side_field="rolling_depth2_commit_side",
            plan_field="rolling_depth2_commit_plan_id",
            step_field="rolling_depth2_commit_step_id",
            id_field="rolling_depth2_real_committed_proposal_ids",
        )

        depth3_ids = as_int_list(record.get("rolling_depth3_real_committed_proposal_ids"))
        committed_by_depth[3].update(depth3_ids)
        ready_by_depth[3].update(as_int_set(record.get("rolling_depth3_child_ready_shadow_proposal_ids")))
        generated_by_depth[3].update(as_int_set(record.get("rolling_depth3_child_generated_proposal_ids")))
        add_seq_map(seq_by_depth, 3, depth3_ids, as_int_list(record.get("rolling_depth3_real_committed_seq_ids")))
        add_seq_map(
            seq_by_depth,
            3,
            as_int_list(record.get("rolling_depth3_child_generated_proposal_ids")),
            as_int_list(record.get("rolling_depth3_child_generated_seq_ids")),
        )
        merge_positive(token_by_depth[3], as_int_map(record.get("rolling_depth3_real_committed_token_count_by_proposal_id")))
        merge_positive(token_by_depth[3], as_int_map(record.get("rolling_depth3_child_token_count_by_proposal_id")))
        merge_positive(accept_by_depth[3], as_int_map(record.get("rolling_depth3_real_committed_accept_len_by_proposal_id")))
        merge_first(parent_by_depth[3], as_int_map(record.get("rolling_depth3_child_parent_by_proposal_id")))
        merge_first(parent_by_depth[3], as_int_map(record.get("rolling_depth3_commit_parent_by_proposal_id")))
        merge_first(parent_by_depth[3], as_int_map(record.get("rolling_depth3_real_commit_parent_by_proposal_id")))
        merge_first(root_by_depth[3], as_int_map(record.get("rolling_depth3_child_root_by_proposal_id")))
        merge_first(root_by_depth[3], as_int_map(record.get("rolling_depth3_real_commit_root_by_proposal_id")))
        for proposal_id, depth in as_int_map(record.get("rolling_depth3_child_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
        for proposal_id, depth in as_int_map(record.get("rolling_depth3_real_commit_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
            if depth > MAX_AUDITED_REAL_DEPTH and proposal_id in depth3_ids:
                higher_depth_commit_ids.add(proposal_id)
        action_by_depth[3].update(as_str_map(record.get("rolling_depth3_real_commit_action_by_proposal_id")))
        result_by_depth[3].update(as_str_map(record.get("rolling_depth3_real_commit_verify_result_by_proposal_id")))
        invalidated_ids.update(as_int_set(record.get("rolling_depth3_child_invalidated_proposal_ids")))
        cascade_ids.update(as_int_set(record.get("rolling_depth3_committed_cascade_discarded_child_ids")))
        explicit_duplicate_ids.update(as_int_set(record.get("rolling_depth3_real_commit_duplicate_proposal_ids")))
        explicit_duplicate_seq_ids.update(as_int_set(record.get("rolling_depth3_real_commit_duplicate_seq_ids")))
        explicit_without_ready.update(as_int_set(record.get("rolling_depth3_committed_without_ready_shadow_ids")))
        explicit_without_parent.update(as_int_set(record.get("rolling_depth3_committed_without_parent_depth2_commit_ids")))
        explicit_invalid_committed.update(as_int_set(record.get("rolling_depth3_committed_invalidated_child_ids")))
        explicit_cascade_committed.update(as_int_set(record.get("rolling_depth3_committed_cascade_discarded_child_ids")))
        explicit_non_full.update(as_int_set(record.get("rolling_depth3_committed_non_full_accept_ids")))
        stale_expired_ids.update(dict_field_false_ids(record.get("rolling_depth3_target_draft_len_match_by_seq_id")))
        add_side_events(
            side_events,
            record,
            depth=3,
            side_field="rolling_depth3_commit_side",
            plan_field="rolling_depth3_commit_plan_id",
            step_field="rolling_depth3_commit_step_id",
            id_field="rolling_depth3_real_committed_proposal_ids",
        )

        depth4_ids = as_int_list(record.get("rolling_depth4_real_committed_proposal_ids"))
        committed_by_depth[4].update(depth4_ids)
        ready_by_depth[4].update(as_int_set(record.get("rolling_depth4_child_ready_shadow_proposal_ids")))
        generated_by_depth[4].update(as_int_set(record.get("rolling_depth4_child_generated_proposal_ids")))
        add_seq_map(seq_by_depth, 4, depth4_ids, as_int_list(record.get("rolling_depth4_real_committed_seq_ids")))
        add_seq_map(
            seq_by_depth,
            4,
            as_int_list(record.get("rolling_depth4_child_generated_proposal_ids")),
            as_int_list(record.get("rolling_depth4_child_generated_seq_ids")),
        )
        add_seq_map(
            seq_by_depth,
            4,
            as_int_list(record.get("rolling_depth4_child_ready_shadow_proposal_ids")),
            as_int_list(record.get("rolling_depth4_child_ready_shadow_seq_ids")),
        )
        merge_positive(token_by_depth[4], as_int_map(record.get("rolling_depth4_real_committed_token_count_by_proposal_id")))
        merge_positive(token_by_depth[4], as_int_map(record.get("rolling_depth4_child_token_count_by_proposal_id")))
        merge_positive(accept_by_depth[4], as_int_map(record.get("rolling_depth4_real_committed_accept_len_by_proposal_id")))
        merge_first(parent_by_depth[4], as_int_map(record.get("rolling_depth4_child_parent_by_proposal_id")))
        merge_first(parent_by_depth[4], as_int_map(record.get("rolling_depth4_commit_parent_by_proposal_id")))
        merge_first(parent_by_depth[4], as_int_map(record.get("rolling_depth4_real_commit_parent_by_proposal_id")))
        merge_first(root_by_depth[4], as_int_map(record.get("rolling_depth4_child_root_by_proposal_id")))
        merge_first(root_by_depth[4], as_int_map(record.get("rolling_depth4_real_commit_root_by_proposal_id")))
        for proposal_id, depth in as_int_map(record.get("rolling_depth4_child_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
        for proposal_id, depth in as_int_map(record.get("rolling_depth4_real_commit_depth_by_proposal_id")).items():
            declared_depth_by_id.setdefault(proposal_id, depth)
            if depth > MAX_AUDITED_REAL_DEPTH and proposal_id in depth4_ids:
                higher_depth_commit_ids.add(proposal_id)
        action_by_depth[4].update(as_str_map(record.get("rolling_depth4_real_commit_action_by_proposal_id")))
        result_by_depth[4].update(as_str_map(record.get("rolling_depth4_real_commit_verify_result_by_proposal_id")))
        invalidated_ids.update(as_int_set(record.get("rolling_depth4_child_invalidated_proposal_ids")))
        cascade_ids.update(as_int_set(record.get("rolling_depth4_committed_cascade_discarded_child_ids")))
        explicit_duplicate_ids.update(as_int_set(record.get("rolling_depth4_real_commit_duplicate_proposal_ids")))
        explicit_duplicate_seq_ids.update(as_int_set(record.get("rolling_depth4_real_commit_duplicate_seq_ids")))
        explicit_without_ready.update(as_int_set(record.get("rolling_depth4_committed_without_ready_shadow_ids")))
        explicit_without_parent.update(as_int_set(record.get("rolling_depth4_committed_without_parent_depth3_commit_ids")))
        explicit_invalid_committed.update(as_int_set(record.get("rolling_depth4_committed_invalidated_child_ids")))
        explicit_cascade_committed.update(as_int_set(record.get("rolling_depth4_committed_cascade_discarded_child_ids")))
        explicit_non_full.update(as_int_set(record.get("rolling_depth4_committed_non_full_accept_ids")))
        stale_expired_ids.update(dict_field_false_ids(record.get("rolling_depth4_target_draft_len_match_by_seq_id")))
        add_side_events(
            side_events,
            record,
            depth=4,
            side_field="rolling_depth4_commit_side",
            plan_field="rolling_depth4_commit_plan_id",
            step_field="rolling_depth4_commit_step_id",
            id_field="rolling_depth4_real_committed_proposal_ids",
        )

        for field in (
            "ready_eager_proposal_stale_ids",
            "ready_eager_proposal_expired_ids",
            "stale_lane_exclusion_decision_ids",
            "expired_lane_exclusion_decision_ids",
        ):
            stale_expired_ids.update(as_int_set(record.get(field)))

    for depth, ids in committed_by_depth.items():
        for proposal_id in ids:
            declared_depth_by_id.setdefault(proposal_id, depth)
            max_observed_depth = max(max_observed_depth, depth)

    for (depth, side, proposal_id), steps in side_events.items():
        if len(steps) > 1:
            explicit_duplicate_ids.add(proposal_id)
        for plan_id, step_id in steps:
            seq_id = seq_by_depth[depth].get(proposal_id, -1)
            event = (depth, side, plan_id, step_id, seq_id)
            if event in seq_depth_events:
                duplicate_seq_depth_events.add(event)
            seq_depth_events.add(event)

    return {
        "committed_by_depth": committed_by_depth,
        "token_by_depth": token_by_depth,
        "seq_by_depth": seq_by_depth,
        "parent_by_depth": parent_by_depth,
        "root_by_depth": root_by_depth,
        "declared_depth_by_id": declared_depth_by_id,
        "action_by_depth": action_by_depth,
        "result_by_depth": result_by_depth,
        "accept_by_depth": accept_by_depth,
        "ready_by_depth": ready_by_depth,
        "generated_by_depth": generated_by_depth,
        "invalidated_ids": invalidated_ids,
        "cascade_ids": cascade_ids,
        "stale_expired_ids": stale_expired_ids,
        "frontier_mismatch_ids": frontier_mismatch_ids,
        "explicit_duplicate_ids": explicit_duplicate_ids,
        "explicit_duplicate_seq_ids": explicit_duplicate_seq_ids,
        "duplicate_seq_depth_events": duplicate_seq_depth_events,
        "explicit_without_ready": explicit_without_ready,
        "explicit_without_parent": explicit_without_parent,
        "explicit_invalid_committed": explicit_invalid_committed,
        "explicit_cascade_committed": explicit_cascade_committed,
        "explicit_non_full": explicit_non_full,
        "higher_depth_commit_ids": higher_depth_commit_ids,
        "length_mismatch_count": length_mismatch_count,
        "token_mismatch_count": token_mismatch_count,
        "normal_lane_conflict_count": normal_lane_conflict_count,
        "missing_unexpected_count": missing_unexpected_count,
        "depth4_real_commit_count": depth4_real_commit_count,
        "depth_gt3_real_commit_count": depth_gt3_real_commit_count,
        "depth_gt4_real_commit_count": depth_gt4_real_commit_count,
        "max_configured_depth": max_configured_depth,
        "max_observed_depth": max_observed_depth,
        "gamma": gamma,
        "flags": flags,
    }


def validate_accounting_by_depth(accounting: dict[str, Any], errors: list[str]) -> tuple[bool, bool]:
    depth_specs = (
        (
            "one-shot",
            "eager_committed_token_count",
            "target_actual_eager_verified_token_increment_sum",
            "target_actual_eager_accepted_token_increment_sum",
            "target_actual_eager_rejected_token_increment_sum",
            "target_actual_eager_invalidated_token_increment_sum",
            "draft_actual_eager_verified_token_increment_sum",
            "draft_actual_eager_accepted_token_increment_sum",
            None,
            None,
        ),
        (
            "depth1",
            "continuous_eager_real_committed_token_count",
            "continuous_target_actual_verified_token_increment_sum",
            "continuous_target_actual_accepted_token_increment_sum",
            "continuous_target_actual_rejected_token_increment_sum",
            "continuous_target_actual_invalidated_token_increment_sum",
            "continuous_draft_actual_verified_token_increment_sum",
            "continuous_draft_actual_accepted_token_increment_sum",
            "continuous_draft_actual_rejected_token_increment_sum",
            "continuous_draft_actual_invalidated_token_increment_sum",
        ),
        (
            "depth2",
            "rolling_depth2_real_committed_token_count",
            "rolling_depth2_target_actual_verified_token_increment_sum",
            "rolling_depth2_target_actual_accepted_token_increment_sum",
            "rolling_depth2_target_actual_rejected_token_increment_sum",
            "rolling_depth2_target_actual_invalidated_token_increment_sum",
            "rolling_depth2_draft_actual_verified_token_increment_sum",
            "rolling_depth2_draft_actual_accepted_token_increment_sum",
            "rolling_depth2_draft_actual_rejected_token_increment_sum",
            "rolling_depth2_draft_actual_invalidated_token_increment_sum",
        ),
        (
            "depth3",
            "rolling_depth3_real_committed_token_count",
            "rolling_depth3_target_actual_verified_token_increment_sum",
            "rolling_depth3_target_actual_accepted_token_increment_sum",
            "rolling_depth3_target_actual_rejected_token_increment_sum",
            "rolling_depth3_target_actual_invalidated_token_increment_sum",
            "rolling_depth3_draft_actual_verified_token_increment_sum",
            "rolling_depth3_draft_actual_accepted_token_increment_sum",
            "rolling_depth3_draft_actual_rejected_token_increment_sum",
            "rolling_depth3_draft_actual_invalidated_token_increment_sum",
        ),
        (
            "depth4",
            "rolling_depth4_real_committed_token_count",
            "rolling_depth4_target_actual_verified_token_increment_sum",
            "rolling_depth4_target_actual_accepted_token_increment_sum",
            "rolling_depth4_target_actual_rejected_token_increment_sum",
            "rolling_depth4_target_actual_invalidated_token_increment_sum",
            "rolling_depth4_draft_actual_verified_token_increment_sum",
            "rolling_depth4_draft_actual_accepted_token_increment_sum",
            "rolling_depth4_draft_actual_rejected_token_increment_sum",
            "rolling_depth4_draft_actual_invalidated_token_increment_sum",
        ),
    )
    target_draft_ok = True
    for label, token_key, target_verified, target_accepted, target_rejected, target_invalidated, draft_verified, draft_accepted, draft_rejected, draft_invalidated in depth_specs:
        tokens = int_value(accounting.get(token_key), 0)
        checks = (
            (target_verified, tokens, "target verified"),
            (target_accepted, tokens, "target accepted"),
            (draft_verified, tokens, "draft verified"),
            (draft_accepted, tokens, "draft accepted"),
            (target_rejected, 0, "target rejected"),
            (target_invalidated, 0, "target invalidated"),
            (draft_rejected, 0, "draft rejected"),
            (draft_invalidated, 0, "draft invalidated"),
        )
        for key, expected, description in checks:
            if key is None:
                continue
            if int_value(accounting.get(key), 0) != expected:
                target_draft_ok = False
                errors.append(f"{label} {description} increment mismatch")

    combined_expected = (
        int_value(accounting.get("eager_committed_token_count"), 0)
        + int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0)
        + int_value(accounting.get("partial_prefix_total_recovered_token_count"), 0)
    )
    combined_ok = int_value(accounting.get("combined_real_committed_token_count"), 0) == combined_expected
    combined_ok = combined_ok and int_value(accounting.get("combined_actual_verified_token_increment_sum"), 0) == combined_expected
    partial_revised = int_value(accounting.get("partial_prefix_revised_token_count"), 0)
    if partial_revised:
        combined_ok = combined_ok and int_value(accounting.get("combined_actual_output_token_increment_sum"), 0) == combined_expected
        combined_ok = combined_ok and int_value(
            accounting.get("combined_actual_accepted_token_increment_sum"), 0
        ) == combined_expected - partial_revised
        combined_ok = combined_ok and int_value(
            accounting.get("combined_actual_revised_token_increment_sum"), 0
        ) == partial_revised
    else:
        combined_ok = combined_ok and int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0) == combined_expected
    if not combined_ok:
        errors.append("combined accounting does not equal full-accept commits plus partial recovery output")
    return combined_ok, target_draft_ok


def validate_chain(trace: dict[str, Any], errors: list[str]) -> dict[str, int]:
    committed_by_depth: dict[int, set[int]] = trace["committed_by_depth"]
    token_by_depth: dict[int, dict[int, int]] = trace["token_by_depth"]
    seq_by_depth: dict[int, dict[int, int]] = trace["seq_by_depth"]
    parent_by_depth: dict[int, dict[int, int]] = trace["parent_by_depth"]
    root_by_depth: dict[int, dict[int, int]] = trace["root_by_depth"]
    declared_depth_by_id: dict[int, int] = trace["declared_depth_by_id"]
    action_by_depth: dict[int, dict[int, str]] = trace["action_by_depth"]
    result_by_depth: dict[int, dict[int, str]] = trace["result_by_depth"]
    accept_by_depth: dict[int, dict[int, int]] = trace["accept_by_depth"]
    ready_by_depth: dict[int, set[int]] = trace["ready_by_depth"]
    invalidated_ids: set[int] = trace["invalidated_ids"]
    cascade_ids: set[int] = trace["cascade_ids"]
    stale_expired_ids: set[int] = trace["stale_expired_ids"]
    frontier_mismatch_ids: set[int] = trace["frontier_mismatch_ids"]
    explicit_without_ready: set[int] = trace["explicit_without_ready"]
    explicit_without_parent: set[int] = trace["explicit_without_parent"]
    explicit_invalid_committed: set[int] = trace["explicit_invalid_committed"]
    explicit_cascade_committed: set[int] = trace["explicit_cascade_committed"]
    explicit_non_full: set[int] = trace["explicit_non_full"]
    gamma = int(trace["gamma"])

    invalid_committed_ids: set[int] = set(explicit_invalid_committed)
    cascade_committed_ids: set[int] = set(explicit_cascade_committed)
    parent_missing_ids: set[int] = set(explicit_without_parent)
    generated_only_ids: set[int] = set(explicit_without_ready)
    non_full_ids: set[int] = set(explicit_non_full)
    stale_committed_ids: set[int] = set()

    for depth in range(0, MAX_AUDITED_REAL_DEPTH + 1):
        expected_action = ROLLING_ACTION if depth >= 2 else ONE_SHOT_ACTION
        for proposal_id in sorted(committed_by_depth[depth]):
            declared_depth = declared_depth_by_id.get(proposal_id, depth)
            if declared_depth != depth:
                errors.append(f"committed proposal {proposal_id} declared depth {declared_depth}, expected {depth}")
            result = result_by_depth[depth].get(proposal_id)
            if result is not None and result != "full_accept":
                non_full_ids.add(proposal_id)
            action = action_by_depth[depth].get(proposal_id)
            if action is not None and action != expected_action:
                non_full_ids.add(proposal_id)
            token_count = token_by_depth[depth].get(proposal_id, gamma)
            accept_len = accept_by_depth[depth].get(proposal_id, token_count)
            if token_count <= 0 or accept_len != token_count:
                errors.append(f"committed proposal {proposal_id} depth {depth} token/accept mismatch")
            if proposal_id in invalidated_ids:
                invalid_committed_ids.add(proposal_id)
            if proposal_id in cascade_ids:
                cascade_committed_ids.add(proposal_id)
            if proposal_id in stale_expired_ids or proposal_id in frontier_mismatch_ids:
                stale_committed_ids.add(proposal_id)
            if depth >= 1 and proposal_id not in ready_by_depth[depth]:
                generated_only_ids.add(proposal_id)
            if depth == 0:
                continue
            parent_id = parent_by_depth[depth].get(proposal_id)
            if parent_id is None:
                parent_missing_ids.add(proposal_id)
                continue
            if parent_id not in committed_by_depth[depth - 1]:
                parent_missing_ids.add(proposal_id)
            parent_depth = declared_depth_by_id.get(parent_id, depth - 1)
            if parent_depth != depth - 1:
                errors.append(f"proposal {proposal_id} parent {parent_id} depth {parent_depth}, expected {depth - 1}")
            parent_seq = seq_by_depth[depth - 1].get(parent_id)
            child_seq = seq_by_depth[depth].get(proposal_id)
            if parent_seq is not None and child_seq is not None and parent_seq != child_seq:
                errors.append(f"proposal {proposal_id} seq {child_seq} differs from parent {parent_id} seq {parent_seq}")
            parent_root = root_by_depth[depth - 1].get(parent_id, parent_id if depth - 1 == 0 else None)
            child_root = root_by_depth[depth].get(proposal_id)
            if parent_root is not None and child_root is not None and parent_root != child_root:
                errors.append(f"proposal {proposal_id} root {child_root} differs from parent {parent_id} root {parent_root}")

    if invalid_committed_ids:
        errors.append(f"invalidated committed proposal ids: {sorted(invalid_committed_ids)}")
    if cascade_committed_ids:
        errors.append(f"cascade-discarded committed proposal ids: {sorted(cascade_committed_ids)}")
    if stale_committed_ids:
        errors.append(f"stale/expired/frontier-mismatched committed proposal ids: {sorted(stale_committed_ids)}")
    if parent_missing_ids:
        errors.append(f"parent-missing committed proposal ids: {sorted(parent_missing_ids)}")
    if generated_only_ids:
        errors.append(f"committed proposals without ready shadow: {sorted(generated_only_ids)}")
    if non_full_ids:
        errors.append(f"non-full-accept committed proposal ids: {sorted(non_full_ids)}")

    return {
        "invalid_committed_child_count": len(invalid_committed_ids),
        "cascade_committed_child_count": len(cascade_committed_ids),
        "parent_missing_committed_child_count": len(parent_missing_ids),
        "generated_only_committed_child_count": len(generated_only_ids),
        "non_full_accept_committed_count": len(non_full_ids),
        "stale_or_frontier_committed_child_count": len(stale_committed_ids),
    }


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    result_payload = result_payload or {}
    accounting = aggregate_performance_accounting(records, result_payload)
    trace = collect_trace(records)
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if isinstance(result_args, dict):
        trace["max_configured_depth"] = max(
            int_value(trace.get("max_configured_depth"), 0),
            int_value(result_args.get("max_rolling_continuous_depth"), 0),
        )
    combined_ok, target_draft_ok = validate_accounting_by_depth(accounting, errors)
    chain_summary = validate_chain(trace, errors)

    committed_by_depth: dict[int, set[int]] = trace["committed_by_depth"]
    token_by_depth: dict[int, dict[int, int]] = trace["token_by_depth"]
    gamma = int(trace["gamma"])
    derived_tokens = {
        depth: proposal_tokens(committed_by_depth[depth], token_by_depth, depth, gamma)
        for depth in range(0, MAX_AUDITED_REAL_DEPTH + 1)
    }
    accounting_token_keys = {
        0: "eager_committed_token_count",
        1: "continuous_eager_real_committed_token_count",
        2: "rolling_depth2_real_committed_token_count",
        3: "rolling_depth3_real_committed_token_count",
        4: "rolling_depth4_real_committed_token_count",
    }
    for depth, key in accounting_token_keys.items():
        accounting_tokens = int_value(accounting.get(key), 0)
        if committed_by_depth[depth] or accounting_tokens:
            if accounting_tokens != derived_tokens[depth]:
                errors.append(f"depth {depth} committed token aggregate mismatch")

    duplicate_commit_count = (
        len(trace["explicit_duplicate_ids"])
        + len(trace["explicit_duplicate_seq_ids"])
        + len(trace["duplicate_seq_depth_events"])
    )
    if duplicate_commit_count:
        errors.append("duplicate commit evidence present")
    if committed_by_depth[3] and not bool(trace["flags"].get("rolling_depth3_commit_enabled", False)):
        errors.append("depth3 real commit appears while depth3 commit flag is disabled")
    if committed_by_depth[4]:
        if not bool(trace["flags"].get("rolling_depth4_commit_enabled", False)):
            errors.append("depth4 real commit appears while depth4 commit flag is disabled")
        if not bool(trace["flags"].get("rolling_depth4_shadow_enabled", False)):
            errors.append("depth4 real commit appears while depth4 shadow flag is disabled")
    if trace["higher_depth_commit_ids"]:
        errors.append(f"depth>4 committed proposal ids present: {sorted(trace['higher_depth_commit_ids'])}")
    if trace["length_mismatch_count"]:
        errors.append("target/draft length mismatch evidence present")
    if trace["token_mismatch_count"]:
        errors.append("target/draft token mismatch evidence present")
    if trace["normal_lane_conflict_count"]:
        errors.append("normal lane conflict evidence present")
    if trace["missing_unexpected_count"]:
        errors.append("unexpected missing buffered proposal evidence present")
    if trace["depth4_real_commit_count"] and not bool(trace["flags"].get("rolling_depth4_commit_enabled", False)):
        errors.append("rolling depth4 real commit count requires depth4 commit flag")
    if trace["depth_gt3_real_commit_count"] and not bool(trace["flags"].get("rolling_depth4_commit_enabled", False)):
        errors.append("rolling depth>3 real commit count must be zero")
    if trace["depth_gt4_real_commit_count"]:
        errors.append("rolling depth>4 real commit count must be zero")

    max_real_depth = max((depth for depth, ids in committed_by_depth.items() if ids), default=0)
    max_real_depth = max(
        max_real_depth,
        3 if int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0) else 0,
        4 if int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0) else 0,
        2 if int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0) else 0,
        1 if int_value(accounting.get("continuous_eager_real_committed_token_count"), 0) else 0,
    )
    if max_real_depth > MAX_AUDITED_REAL_DEPTH:
        errors.append(f"max real committed depth {max_real_depth} exceeds {MAX_AUDITED_REAL_DEPTH}")

    summary = {
        "total_trace_records": len(records),
        **trace["flags"],
        "max_configured_depth": trace["max_configured_depth"],
        "max_observed_depth": trace["max_observed_depth"],
        "max_real_committed_depth": max_real_depth,
        "depth_gt3_real_commit_count": trace["depth_gt3_real_commit_count"],
        "depth_gt4_real_commit_count": trace["depth_gt4_real_commit_count"],
        "depth4_real_commit_count": trace["depth4_real_commit_count"],
        "depth_gt3_committed_proposal_count": len(trace["higher_depth_commit_ids"]),
        "one_shot_committed_proposal_count": len(committed_by_depth[0]),
        "one_shot_committed_token_count": int_value(accounting.get("eager_committed_token_count"), 0),
        "depth1_committed_proposal_count": len(committed_by_depth[1]),
        "depth1_committed_token_count": int_value(accounting.get("continuous_eager_real_committed_token_count"), 0),
        "depth2_committed_proposal_count": len(committed_by_depth[2]),
        "depth2_committed_token_count": int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0),
        "depth3_committed_proposal_count": len(committed_by_depth[3]),
        "depth3_committed_token_count": int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0),
        "depth4_committed_proposal_count": len(committed_by_depth[4]),
        "depth4_committed_token_count": int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0),
        "partial_prefix_recovery_enabled": bool(accounting.get("partial_prefix_recovery_enabled", False)),
        "partial_prefix_recovery_success_count": int_value(
            accounting.get("partial_prefix_recovery_success_count"), 0
        ),
        "partial_prefix_accepted_token_count": int_value(
            accounting.get("partial_prefix_accepted_token_count"), 0
        ),
        "partial_prefix_revised_token_count": int_value(
            accounting.get("partial_prefix_revised_token_count"), 0
        ),
        "partial_prefix_total_recovered_token_count": int_value(
            accounting.get("partial_prefix_total_recovered_token_count"), 0
        ),
        "combined_real_committed_token_count": int_value(accounting.get("combined_real_committed_token_count"), 0),
        "combined_actual_verified_token_increment_sum": int_value(
            accounting.get("combined_actual_verified_token_increment_sum"),
            0,
        ),
        "combined_actual_accepted_token_increment_sum": int_value(
            accounting.get("combined_actual_accepted_token_increment_sum"),
            0,
        ),
        "combined_actual_revised_token_increment_sum": int_value(
            accounting.get("combined_actual_revised_token_increment_sum"),
            0,
        ),
        "combined_actual_output_token_increment_sum": int_value(
            accounting.get("combined_actual_output_token_increment_sum"),
            0,
        ),
        "combined_accounting_ok": combined_ok,
        "target_draft_accounting_ok": target_draft_ok and trace["length_mismatch_count"] == 0 and trace["token_mismatch_count"] == 0,
        "normal_lane_conflict_count": trace["normal_lane_conflict_count"],
        "missing_buffered_proposal_unexpected_count": trace["missing_unexpected_count"],
        "duplicate_commit_count": duplicate_commit_count,
        "target_draft_length_mismatch_count": trace["length_mismatch_count"],
        "target_draft_token_mismatch_count": trace["token_mismatch_count"],
        "performance_warnings": accounting.get("performance_warnings", []),
        **chain_summary,
    }
    return errors, summary


def generic_parity_errors(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None,
    legacy_summary: dict[str, Any],
) -> list[str]:
    accounting = aggregate_performance_accounting(records, result_payload or {})
    registry = parse_legacy_rolling_chain(records)
    generic_summary = summarize_registry(registry, accounting=accounting)
    errors: list[str] = []
    for legacy_key, generic_key in GENERIC_PARITY_FIELD_PAIRS:
        legacy_value = legacy_summary.get(legacy_key)
        generic_value = generic_summary.get(generic_key)
        if legacy_value != generic_value:
            errors.append(
                f"generic parity mismatch for {legacy_key}: "
                f"legacy={legacy_value!r} generic={generic_value!r}"
            )
    return errors


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "one_shot_commit_enabled",
        "continuous_depth1_commit_enabled",
        "rolling_depth2_commit_enabled",
        "rolling_depth3_shadow_enabled",
        "rolling_depth3_commit_enabled",
        "rolling_depth4_shadow_enabled",
        "rolling_depth4_commit_enabled",
        "max_configured_depth",
        "max_observed_depth",
        "max_real_committed_depth",
        "depth_gt3_real_commit_count",
        "depth_gt4_real_commit_count",
        "depth4_real_commit_count",
        "depth_gt3_committed_proposal_count",
        "one_shot_committed_proposal_count",
        "one_shot_committed_token_count",
        "depth1_committed_proposal_count",
        "depth1_committed_token_count",
        "depth2_committed_proposal_count",
        "depth2_committed_token_count",
        "depth3_committed_proposal_count",
        "depth3_committed_token_count",
        "depth4_committed_proposal_count",
        "depth4_committed_token_count",
        "partial_prefix_recovery_enabled",
        "partial_prefix_recovery_success_count",
        "partial_prefix_accepted_token_count",
        "partial_prefix_revised_token_count",
        "partial_prefix_total_recovered_token_count",
        "combined_real_committed_token_count",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "combined_actual_revised_token_increment_sum",
        "combined_actual_output_token_increment_sum",
        "combined_accounting_ok",
        "target_draft_accounting_ok",
        "normal_lane_conflict_count",
        "missing_buffered_proposal_unexpected_count",
        "duplicate_commit_count",
        "invalid_committed_child_count",
        "cascade_committed_child_count",
        "parent_missing_committed_child_count",
        "generated_only_committed_child_count",
        "non_full_accept_committed_count",
        "stale_or_frontier_committed_child_count",
        "target_draft_length_mismatch_count",
        "target_draft_token_mismatch_count",
        "performance_warnings",
    ):
        print(f"{key}={summary.get(key)}")


def synthetic_full_chain_record(side: str) -> dict[str, Any]:
    seq_id = 7
    p0 = 900000100
    p1 = 900000101
    p2 = 900000102
    p3 = 900000103
    return {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "max_rolling_continuous_depth": 3,
        "rolling_max_depth_observed": 3,
        "max_rolling_continuous_depth_observed": 3,
        "enable_eager_commit_ready_only": True,
        "enable_eager_commit_readiness_dry_run": True,
        "eager_commit_enabled": True,
        "eager_commit_source": "phase1h5e3_takeover_lane",
        "eager_commit_side": side,
        "eager_commit_plan_id": 1,
        "eager_commit_step_id": 1,
        "eager_commit_ready_proposal_ids": [p0],
        "eager_commit_ready_seq_ids": [seq_id],
        "eager_commit_candidate_proposal_ids": [p0],
        "eager_commit_candidate_seq_ids": [seq_id],
        "eager_commit_from_readiness_proposal_ids": [p0],
        "eager_committed_proposal_ids": [p0],
        "eager_committed_seq_ids": [seq_id],
        "eager_committed_token_count_by_proposal_id": {str(p0): 4},
        "eager_committed_accept_len_by_proposal_id": {str(p0): 4},
        "eager_committed_action_by_proposal_id": {str(p0): ONE_SHOT_ACTION},
        "eager_committed_verify_result_by_proposal_id": {str(p0): "full_accept"},
        "eager_commit_precondition_ok_by_proposal_id": {str(p0): True},
        "eager_commit_precondition_failed_by_proposal_id": {str(p0): False},
        "eager_commit_target_seq_len_before_by_seq_id": {str(seq_id): 20},
        "eager_commit_target_seq_len_after_by_seq_id": {str(seq_id): 24},
        "eager_commit_draft_seq_len_before_by_seq_id": {str(seq_id): 20},
        "eager_commit_draft_seq_len_after_by_seq_id": {str(seq_id): 24},
        "eager_tokens_verified": 4,
        "eager_tokens_accepted": 4,
        "eager_tokens_rejected": 0,
        "eager_tokens_invalidated": 0,
        "eager_commit_target_draft_len_match_by_seq_id": {str(seq_id): True},
        "eager_commit_target_draft_token_match_by_seq_id": {str(seq_id): True},
        "enable_continuous_eager_commit_depth1_ready_only": True,
        "continuous_eager_commit_enabled": True,
        "continuous_eager_commit_side": side,
        "continuous_eager_commit_plan_id": 2,
        "continuous_eager_commit_step_id": 2,
        "continuous_eager_candidate_proposal_ids": [p1],
        "continuous_eager_candidate_seq_ids": [seq_id],
        "continuous_eager_candidate_token_count_by_proposal_id": {str(p1): 4},
        "continuous_eager_parent_proposal_id_by_proposal_id": {str(p1): p0},
        "continuous_eager_chain_depth_by_proposal_id": {str(p1): 1},
        "continuous_eager_root_proposal_id_by_proposal_id": {str(p1): p0},
        "continuous_eager_commit_ready_shadow_proposal_ids": [p1],
        "continuous_eager_full_accept_proposal_ids": [p1],
        "continuous_eager_real_committed_proposal_ids": [p1],
        "continuous_eager_real_committed_seq_ids": [seq_id],
        "continuous_eager_real_committed_token_count_by_proposal_id": {str(p1): 4},
        "continuous_eager_real_committed_accept_len_by_proposal_id": {str(p1): 4},
        "continuous_eager_real_commit_action_by_proposal_id": {str(p1): ONE_SHOT_ACTION},
        "continuous_eager_real_commit_verify_result_by_proposal_id": {str(p1): "full_accept"},
        "continuous_eager_tokens_verified": 4,
        "continuous_eager_tokens_accepted": 4,
        "continuous_eager_tokens_rejected": 0,
        "continuous_eager_tokens_invalidated": 0,
        "continuous_eager_target_draft_len_match_by_seq_id": {str(seq_id): True},
        "continuous_eager_target_draft_token_match_by_seq_id": {str(seq_id): True},
        "enable_rolling_continuous_depth2_commit_ready_only": True,
        "rolling_depth2_commit_enabled": True,
        "rolling_depth2_commit_side": side,
        "rolling_depth2_commit_plan_id": 3,
        "rolling_depth2_commit_step_id": 3,
        "rolling_chain_parent_by_proposal_id": {str(p2): p1},
        "rolling_chain_root_by_proposal_id": {str(p2): p0},
        "rolling_chain_depth_by_proposal_id": {str(p2): 2},
        "rolling_child_generated_proposal_ids": [p2],
        "rolling_child_ready_after_parent_full_accept_proposal_ids": [p2],
        "rolling_parent_full_accept_proposal_ids": [p1],
        "rolling_depth2_real_committed_proposal_ids": [p2],
        "rolling_depth2_real_committed_seq_ids": [seq_id],
        "rolling_depth2_real_committed_token_count_by_proposal_id": {str(p2): 4},
        "rolling_depth2_real_committed_accept_len_by_proposal_id": {str(p2): 4},
        "rolling_depth2_real_commit_action_by_proposal_id": {str(p2): ROLLING_ACTION},
        "rolling_depth2_real_commit_verify_result_by_proposal_id": {str(p2): "full_accept"},
        "rolling_depth2_real_commit_parent_by_proposal_id": {str(p2): p1},
        "rolling_depth2_real_commit_root_by_proposal_id": {str(p2): p0},
        "rolling_depth2_real_commit_depth_by_proposal_id": {str(p2): 2},
        "rolling_depth2_tokens_verified": 4,
        "rolling_depth2_tokens_accepted": 4,
        "rolling_depth2_tokens_rejected": 0,
        "rolling_depth2_tokens_invalidated": 0,
        "rolling_depth2_target_draft_len_match_by_seq_id": {str(seq_id): True},
        "rolling_depth2_target_draft_token_match_by_seq_id": {str(seq_id): True},
        "enable_rolling_continuous_depth3_shadow_dry_run": True,
        "enable_rolling_continuous_depth3_commit_ready_only": True,
        "rolling_depth3_shadow_enabled": True,
        "rolling_depth3_commit_enabled": True,
        "rolling_depth3_commit_side": side,
        "rolling_depth3_commit_plan_id": 4,
        "rolling_depth3_commit_step_id": 4,
        "rolling_depth3_parent_depth2_real_committed_proposal_ids": [p2],
        "rolling_depth3_parent_depth2_full_accept_proposal_ids": [p2],
        "rolling_depth3_child_generated_proposal_ids": [p3],
        "rolling_depth3_child_generated_seq_ids": [seq_id],
        "rolling_depth3_child_parent_by_proposal_id": {str(p3): p2},
        "rolling_depth3_child_root_by_proposal_id": {str(p3): p0},
        "rolling_depth3_child_depth_by_proposal_id": {str(p3): 3},
        "rolling_depth3_child_token_count_by_proposal_id": {str(p3): 4},
        "rolling_depth3_child_ready_shadow_proposal_ids": [p3],
        "rolling_depth3_child_ready_shadow_seq_ids": [seq_id],
        "rolling_depth3_real_committed_proposal_ids": [p3],
        "rolling_depth3_real_committed_seq_ids": [seq_id],
        "rolling_depth3_real_committed_token_count_by_proposal_id": {str(p3): 4},
        "rolling_depth3_real_committed_accept_len_by_proposal_id": {str(p3): 4},
        "rolling_depth3_real_commit_action_by_proposal_id": {str(p3): ROLLING_ACTION},
        "rolling_depth3_real_commit_verify_result_by_proposal_id": {str(p3): "full_accept"},
        "rolling_depth3_real_commit_parent_by_proposal_id": {str(p3): p2},
        "rolling_depth3_real_commit_root_by_proposal_id": {str(p3): p0},
        "rolling_depth3_real_commit_depth_by_proposal_id": {str(p3): 3},
        "rolling_depth3_tokens_verified": 4,
        "rolling_depth3_tokens_accepted": 4,
        "rolling_depth3_tokens_rejected": 0,
        "rolling_depth3_tokens_invalidated": 0,
        "rolling_depth3_real_commit_count": 1,
        "rolling_depth4_real_commit_count": 0,
        "rolling_depth_gt3_real_commit_count": 0,
        "rolling_normal_lane_conflict_count": 0,
        "rolling_depth3_normal_lane_conflict_count": 0,
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "rolling_depth3_target_draft_len_match_by_seq_id": {str(seq_id): True},
        "rolling_depth3_target_draft_token_match_by_seq_id": {str(seq_id): True},
    }


def synthetic_result_payload() -> dict[str, Any]:
    return {
        "args": {"execution_mode": "dual_batch_pearl", "decode_ready": True},
        "metrics": {
            "engine_elapsed_s": 1.0,
            "overall": {
                "total_output_tokens": 64,
                "goodput_tokens_per_s": 64.0,
                "mean_tpot_ms": 15.625,
            },
        },
        "traces": [
            {
                "request_id": "synthetic",
                "num_output_tokens": 4,
                "decode_start_ts": 0.0,
                "finish_ts": 0.064,
                "decode_elapsed_ms": 64.0,
                "observed_tpot_ms": 16.0,
            }
        ],
    }


def run_synthetic(check_generic_parity: bool = False) -> None:
    records = [synthetic_full_chain_record("target"), synthetic_full_chain_record("draft")]
    result_payload = synthetic_result_payload()
    errors, summary = validate_records(records, result_payload)
    if errors:
        raise SystemExit(f"synthetic full bounded chain failed: {errors}\nsummary={summary}")
    if summary["combined_real_committed_token_count"] != 16:
        raise SystemExit("synthetic combined token count mismatch")
    if summary["max_real_committed_depth"] != 3:
        raise SystemExit("synthetic max real depth mismatch")
    if check_generic_parity:
        parity_errors = generic_parity_errors(records, result_payload, summary)
        if parity_errors:
            raise SystemExit(f"synthetic full bounded chain generic parity failed: {parity_errors}")
        bad_summary = dict(summary)
        bad_summary["combined_real_committed_token_count"] += 1
        bad_parity_errors = generic_parity_errors(records, result_payload, bad_summary)
        if not bad_parity_errors:
            raise SystemExit("synthetic bad generic parity case should fail")

    lower_mode = deepcopy(records)
    for record in lower_mode:
        for field in (
            "enable_rolling_continuous_depth3_commit_ready_only",
            "rolling_depth3_commit_enabled",
        ):
            record[field] = False
        for field in (
            "rolling_depth3_real_committed_proposal_ids",
            "rolling_depth3_real_committed_seq_ids",
        ):
            record[field] = []
        for field in (
            "rolling_depth3_real_committed_token_count_by_proposal_id",
            "rolling_depth3_real_committed_accept_len_by_proposal_id",
            "rolling_depth3_real_commit_action_by_proposal_id",
            "rolling_depth3_real_commit_verify_result_by_proposal_id",
            "rolling_depth3_real_commit_parent_by_proposal_id",
            "rolling_depth3_real_commit_root_by_proposal_id",
            "rolling_depth3_real_commit_depth_by_proposal_id",
        ):
            record[field] = {}
        record["rolling_depth3_tokens_verified"] = 0
        record["rolling_depth3_tokens_accepted"] = 0
        record["rolling_depth3_real_commit_count"] = 0
    errors, summary = validate_records(lower_mode, result_payload)
    if errors:
        raise SystemExit(f"synthetic lower-mode depth3 shadow failed: {errors}\nsummary={summary}")
    if summary["max_real_committed_depth"] != 2:
        raise SystemExit("synthetic lower-mode max real depth mismatch")
    if check_generic_parity:
        parity_errors = generic_parity_errors(lower_mode, result_payload, summary)
        if parity_errors:
            raise SystemExit(f"synthetic lower-mode generic parity failed: {parity_errors}")

    invalid_parent = deepcopy(records)
    for record in invalid_parent:
        record["rolling_depth3_real_commit_parent_by_proposal_id"] = {"900000103": 12345}
        record["rolling_depth3_child_parent_by_proposal_id"] = {"900000103": 12345}
    errors, _summary = validate_records(invalid_parent, result_payload)
    if not errors or not any("parent-missing" in error for error in errors):
        raise SystemExit("synthetic missing parent should fail")

    invalid_counter = deepcopy(records)
    invalid_counter[0]["rolling_depth3_tokens_verified"] = 0
    errors, _summary = validate_records(invalid_counter, result_payload)
    if not errors or not any("depth3 target verified" in error for error in errors):
        raise SystemExit("synthetic depth3 target counter mismatch should fail")

    depth4 = deepcopy(records)
    depth4[0]["rolling_depth4_real_commit_count"] = 1
    depth4[0]["rolling_depth_gt3_real_commit_count"] = 1
    errors, _summary = validate_records(depth4, result_payload)
    if not errors or not any("depth4" in error for error in errors):
        raise SystemExit("synthetic depth4 real commit should fail")

    print("Synthetic bounded rolling readiness audit checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit bounded N=3 rolling readiness and real-commit invariants.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--check-generic-parity", action="store_true")
    args = parser.parse_args()
    if args.synthetic or args.trace is None:
        run_synthetic(check_generic_parity=args.check_generic_parity)
        return 0
    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result else {}
    errors, summary = validate_records(records, result_payload)
    if args.check_generic_parity:
        errors.extend(generic_parity_errors(records, result_payload, summary))
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
