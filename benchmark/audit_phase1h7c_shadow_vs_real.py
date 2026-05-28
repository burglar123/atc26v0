#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import (  # noqa: E402
    as_int_list,
    as_int_set,
    dict_get,
    int_value,
)
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    TIMING_FIELDS,
    aggregate_performance_accounting,
    load_json,
    load_trace,
)


ONE_SHOT_PARENT_SOURCE = "phase1h6a_one_shot_commit"
CONTINUOUS_SOURCE = "continuous_shadow"
FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"

TIMING_COMPONENT_FIELDS = {
    "continuous_decision_broadcast_time_ms": [
        "continuous_eager_commit_decision_broadcast_time_ms",
        "continuous_eager_decision_broadcast_time_ms",
    ],
    "continuous_result_transfer_time_ms": [
        "continuous_eager_result_transfer_time_ms",
    ],
    "continuous_sync_apply_time_ms": [
        "continuous_eager_sync_apply_dry_run_time_ms",
        "continuous_eager_sync_apply_time_ms",
    ],
    "continuous_real_commit_time_ms": [
        "continuous_eager_commit_time_ms",
        "continuous_eager_real_commit_time_ms",
    ],
    "continuous_verify_apply_dry_run_time_ms": [
        "continuous_eager_verify_apply_dry_run_time_ms",
        "continuous_eager_verify_dry_run_time_ms",
        "continuous_eager_apply_dry_run_time_ms",
        "continuous_eager_overhead_time_ms",
    ],
    "one_shot_real_commit_time_ms": [
        "eager_commit_time_ms",
    ],
    "total_eager_overhead_time_ms": [
        "total_eager_overhead_time_ms",
        *TIMING_FIELDS,
    ],
}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


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


def step_plan(record: dict[str, Any], step_key: str | None = None, plan_key: str | None = None) -> tuple[int, int]:
    step = int_value(record.get(step_key or ""), int_value(record.get("step_id"), -1))
    plan = int_value(record.get(plan_key or ""), int_value(record.get("plan_id"), -1))
    return step, plan


def sorted_ints(values: set[int] | list[int]) -> list[int]:
    return sorted(int(value) for value in values)


def first_non_empty_reason(*reasons: Any) -> str:
    for reason in reasons:
        if reason:
            return str(reason)
    return ""


def update_reason_map(target: dict[int, str], source: Any) -> None:
    for proposal_id, reason in as_str_map(source).items():
        if reason and proposal_id not in target:
            target[proposal_id] = reason


def zip_ids(ids: list[int], seq_ids: list[int]) -> dict[int, int]:
    return {proposal_id: seq_id for proposal_id, seq_id in zip(ids, seq_ids)}


def unique_token_sum(ids: set[int], token_by_id: dict[int, int], gamma: int = 0) -> int:
    return sum(int(token_by_id.get(proposal_id, gamma if gamma > 0 else 0)) for proposal_id in ids)


def extract_one_shot(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: set[int] = set()
    committed: set[int] = set()
    skipped: set[int] = set()
    not_ready: set[int] = set()
    seq_by_id: dict[int, int] = {}
    candidate_seq_by_id: dict[int, int] = {}
    token_by_id: dict[int, int] = {}
    reason_by_id: dict[int, str] = {}
    verify_result_by_id: dict[int, str] = {}
    action_by_id: dict[int, str] = {}
    steps_by_id: dict[int, set[tuple[int, int]]] = defaultdict(set)
    sides_by_id: dict[int, set[str]] = defaultdict(set)
    gamma = 0

    for record in records:
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        candidate_ids = as_int_list(record.get("eager_commit_candidate_proposal_ids"))
        candidate_ids += as_int_list(record.get("eager_commit_readiness_candidate_proposal_ids"))
        candidate_seq_ids = as_int_list(record.get("eager_commit_candidate_seq_ids"))
        candidates.update(candidate_ids)
        candidate_seq_by_id.update(zip_ids(candidate_ids, candidate_seq_ids))

        committed_ids = as_int_list(record.get("eager_committed_proposal_ids"))
        committed_seq_ids = as_int_list(record.get("eager_committed_seq_ids"))
        committed.update(committed_ids)
        seq_by_id.update(zip_ids(committed_ids, committed_seq_ids))
        for proposal_id, seq_id in zip_ids(candidate_ids, candidate_seq_ids).items():
            seq_by_id.setdefault(proposal_id, seq_id)

        skipped_ids = as_int_set(record.get("eager_commit_skipped_proposal_ids"))
        not_ready_ids = as_int_set(record.get("eager_commit_not_ready_proposal_ids"))
        skipped.update(skipped_ids)
        skipped.update(not_ready_ids)
        not_ready.update(not_ready_ids)

        for field in (
            "eager_committed_token_count_by_proposal_id",
            "eager_commit_ready_token_count_by_proposal_id",
            "eager_committed_accept_len_by_proposal_id",
            "eager_commit_readiness_token_count_by_proposal_id",
        ):
            for proposal_id, token_count in as_int_map(record.get(field)).items():
                if token_count > 0:
                    token_by_id.setdefault(proposal_id, token_count)

        update_reason_map(reason_by_id, record.get("eager_commit_skip_reason_by_proposal_id"))
        update_reason_map(reason_by_id, record.get("eager_commit_not_ready_reason_by_proposal_id"))
        update_reason_map(reason_by_id, record.get("eager_commit_precondition_failure_reason_by_proposal_id"))

        for proposal_id, result in as_str_map(record.get("eager_committed_verify_result_by_proposal_id")).items():
            verify_result_by_id.setdefault(proposal_id, result)
        for proposal_id, action in as_str_map(record.get("eager_committed_action_by_proposal_id")).items():
            action_by_id.setdefault(proposal_id, action)

        side = str(record.get("eager_commit_side") or "")
        step, plan = step_plan(record, "eager_commit_step_id", "eager_commit_plan_id")
        for proposal_id in committed_ids:
            steps_by_id[proposal_id].add((step, plan))
            if side:
                sides_by_id[proposal_id].add(side)

    seq_to_committed: dict[int, set[int]] = defaultdict(set)
    seq_to_candidates: dict[int, set[int]] = defaultdict(set)
    for proposal_id in committed:
        if proposal_id in seq_by_id:
            seq_to_committed[seq_by_id[proposal_id]].add(proposal_id)
    for proposal_id in candidates:
        seq_id = seq_by_id.get(proposal_id, candidate_seq_by_id.get(proposal_id))
        if seq_id is not None:
            seq_to_candidates[seq_id].add(proposal_id)

    return {
        "candidate_ids": candidates,
        "committed_ids": committed,
        "skipped_ids": skipped,
        "not_ready_ids": not_ready,
        "seq_by_id": seq_by_id,
        "token_by_id": token_by_id,
        "reason_by_id": reason_by_id,
        "verify_result_by_id": verify_result_by_id,
        "action_by_id": action_by_id,
        "steps_by_id": {key: sorted(value) for key, value in steps_by_id.items()},
        "sides_by_id": {key: sorted(value) for key, value in sides_by_id.items()},
        "seq_to_committed": {key: set(value) for key, value in seq_to_committed.items()},
        "seq_to_candidates": {key: set(value) for key, value in seq_to_candidates.items()},
        "gamma": gamma,
    }


def extract_continuous(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: set[int] = set()
    verified: set[int] = set()
    full_accept: set[int] = set()
    ready_shadow: set[int] = set()
    not_ready: set[int] = set()
    real_committed: set[int] = set()
    real_skipped: set[int] = set()
    seq_by_id: dict[int, int] = {}
    token_by_id: dict[int, int] = {}
    ready_token_by_id: dict[int, int] = {}
    real_token_by_id: dict[int, int] = {}
    parent_by_id: dict[int, int] = {}
    parent_source_by_id: dict[int, str] = {}
    depth_by_id: dict[int, int] = {}
    verify_result_by_id: dict[int, str] = {}
    accept_len_by_id: dict[int, int] = {}
    not_ready_reason_by_id: dict[int, str] = {}
    real_skip_reason_by_id: dict[int, str] = {}
    steps_by_id: dict[int, set[tuple[int, int]]] = defaultdict(set)
    sides_by_id: dict[int, set[str]] = defaultdict(set)
    true_frontier_mismatch_ids: set[int] = set()
    parent_shadow_not_committed_ids: set[int] = set()
    parent_not_ready_ids: set[int] = set()
    duplicate_ids: set[int] = set()
    lane_overlap: set[int] = set()
    takeover_overlap: set[int] = set()
    depth2_real_commit_count = 0
    real_commit_count_fields = 0
    gamma = 0

    for record in records:
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        candidate_ids = as_int_list(record.get("continuous_eager_candidate_proposal_ids"))
        candidate_seq_ids = as_int_list(record.get("continuous_eager_candidate_seq_ids"))
        candidates.update(candidate_ids)
        seq_by_id.update(zip_ids(candidate_ids, candidate_seq_ids))
        verified.update(as_int_set(record.get("continuous_eager_verified_proposal_ids")))
        verified.update(as_int_set(record.get("continuous_eager_verify_dry_run_executed_proposal_ids")))
        full_accept.update(as_int_set(record.get("continuous_eager_full_accept_proposal_ids")))
        ready_shadow.update(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
        not_ready.update(as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids")))
        real_committed_ids = as_int_set(record.get("continuous_eager_real_committed_proposal_ids"))
        real_committed.update(real_committed_ids)
        real_skipped.update(as_int_set(record.get("continuous_eager_real_commit_skipped_proposal_ids")))

        for field, target in (
            ("continuous_eager_candidate_token_count_by_proposal_id", token_by_id),
            ("continuous_eager_commit_ready_shadow_token_count_by_proposal_id", ready_token_by_id),
            ("continuous_eager_real_committed_token_count_by_proposal_id", real_token_by_id),
        ):
            for proposal_id, token_count in as_int_map(record.get(field)).items():
                if token_count > 0:
                    target.setdefault(proposal_id, token_count)
        for proposal_id, parent_id in as_int_map(record.get("continuous_eager_parent_proposal_id_by_proposal_id")).items():
            parent_by_id.setdefault(proposal_id, parent_id)
        for proposal_id, source in as_str_map(record.get("continuous_eager_parent_source_by_proposal_id")).items():
            parent_source_by_id.setdefault(proposal_id, source)
        for proposal_id, depth in as_int_map(record.get("continuous_eager_chain_depth_by_proposal_id")).items():
            if depth > 0:
                depth_by_id.setdefault(proposal_id, depth)
        for proposal_id, result in as_str_map(record.get("continuous_eager_verify_result_by_proposal_id")).items():
            verify_result_by_id.setdefault(proposal_id, result)
        for proposal_id, accept_len in as_int_map(record.get("continuous_eager_accept_len_by_proposal_id")).items():
            accept_len_by_id.setdefault(proposal_id, accept_len)

        update_reason_map(not_ready_reason_by_id, record.get("continuous_eager_not_ready_shadow_reason_by_proposal_id"))
        update_reason_map(real_skip_reason_by_id, record.get("continuous_eager_real_commit_skip_reason_by_proposal_id"))

        true_frontier_mismatch_ids.update(as_int_set(record.get("continuous_eager_true_frontier_mismatch_proposal_ids")))
        parent_shadow_not_committed_ids.update(
            as_int_set(record.get("continuous_eager_parent_shadow_not_committed_proposal_ids"))
        )
        parent_not_ready_ids.update(as_int_set(record.get("continuous_eager_parent_not_ready_proposal_ids")))
        duplicate_ids.update(as_int_set(record.get("continuous_eager_duplicate_proposal_ids")))
        lane_overlap.update(real_committed_ids & as_int_set(record.get("lane_exclusion_applied_proposal_ids")))
        takeover_overlap.update(real_committed_ids & as_int_set(record.get("target_eager_verify_proposal_ids_dry_run")))
        depth2_real_commit_count += int_value(record.get("continuous_depth2_real_commit_count"), 0)
        real_commit_count_fields += int_value(record.get("continuous_eager_real_commit_count"), 0)

        side = str(record.get("continuous_eager_commit_side") or "")
        step, plan = step_plan(record, "continuous_eager_commit_step_id", "continuous_eager_commit_plan_id")
        for proposal_id in real_committed_ids:
            steps_by_id[proposal_id].add((step, plan))
            if side:
                sides_by_id[proposal_id].add(side)

    parent_to_candidates: dict[int, set[int]] = defaultdict(set)
    for proposal_id, parent_id in parent_by_id.items():
        parent_to_candidates[parent_id].add(proposal_id)

    return {
        "candidate_ids": candidates,
        "verified_ids": verified,
        "full_accept_ids": full_accept,
        "ready_shadow_ids": ready_shadow,
        "not_ready_ids": not_ready,
        "real_committed_ids": real_committed,
        "real_skipped_ids": real_skipped,
        "seq_by_id": seq_by_id,
        "token_by_id": token_by_id,
        "ready_token_by_id": ready_token_by_id,
        "real_token_by_id": real_token_by_id,
        "parent_by_id": parent_by_id,
        "parent_source_by_id": parent_source_by_id,
        "parent_to_candidates": {key: set(value) for key, value in parent_to_candidates.items()},
        "depth_by_id": depth_by_id,
        "verify_result_by_id": verify_result_by_id,
        "accept_len_by_id": accept_len_by_id,
        "not_ready_reason_by_id": not_ready_reason_by_id,
        "real_skip_reason_by_id": real_skip_reason_by_id,
        "true_frontier_mismatch_ids": true_frontier_mismatch_ids,
        "parent_shadow_not_committed_ids": parent_shadow_not_committed_ids,
        "parent_not_ready_ids": parent_not_ready_ids,
        "duplicate_ids": duplicate_ids,
        "lane_overlap": lane_overlap,
        "takeover_overlap": takeover_overlap,
        "depth2_real_commit_count": depth2_real_commit_count,
        "real_commit_count_fields": real_commit_count_fields,
        "steps_by_id": {key: sorted(value) for key, value in steps_by_id.items()},
        "sides_by_id": {key: sorted(value) for key, value in sides_by_id.items()},
        "gamma": gamma,
    }


def reason_for_shadow_only_one_shot(
    proposal_id: int,
    shadow_one_shot: dict[str, Any],
    real_one_shot: dict[str, Any],
    real_continuous: dict[str, Any],
) -> str:
    seq_id = shadow_one_shot["seq_by_id"].get(proposal_id)
    if proposal_id in real_one_shot["committed_ids"]:
        return "committed_in_real"
    if proposal_id in real_one_shot["skipped_ids"] or proposal_id in real_one_shot["not_ready_ids"]:
        return first_non_empty_reason(
            real_one_shot["reason_by_id"].get(proposal_id),
            "skipped_without_reason",
        )
    if proposal_id in real_one_shot["candidate_ids"]:
        return first_non_empty_reason(
            real_one_shot["reason_by_id"].get(proposal_id),
            "candidate_not_committed_without_reason",
        )
    if seq_id is not None and real_continuous["real_committed_ids"]:
        real_cont_seq_ids = {
            real_continuous["seq_by_id"].get(pid)
            for pid in real_continuous["real_committed_ids"]
        }
        if seq_id in real_cont_seq_ids:
            return "replaced_or_advanced_by_continuous_commit"
    if seq_id is not None and seq_id in real_one_shot["seq_to_committed"]:
        return "changed_proposal_id_same_seq_committed"
    if seq_id is not None and seq_id in real_one_shot["seq_to_candidates"]:
        return "changed_proposal_id_same_seq_not_committed"
    return "absent"


def parent_status(parent_id: int, one_shot: dict[str, Any]) -> str:
    if parent_id in one_shot["committed_ids"]:
        return "one_shot_committed"
    if parent_id in one_shot["skipped_ids"] or parent_id in one_shot["not_ready_ids"]:
        return first_non_empty_reason(one_shot["reason_by_id"].get(parent_id), "one_shot_skipped_without_reason")
    if parent_id in one_shot["candidate_ids"]:
        return "one_shot_candidate_not_committed"
    return "one_shot_absent_or_run_local_id_mismatch"


def compare_one_shot(shadow: dict[str, Any], real: dict[str, Any], real_continuous: dict[str, Any]) -> dict[str, Any]:
    shadow_committed = set(shadow["committed_ids"])
    real_committed = set(real["committed_ids"])
    common = shadow_committed & real_committed
    shadow_only = shadow_committed - real_committed
    real_only = real_committed - shadow_committed
    reason_by_id = {
        proposal_id: reason_for_shadow_only_one_shot(proposal_id, shadow, real, real_continuous)
        for proposal_id in shadow_only
    }
    seq_shadow = {shadow["seq_by_id"].get(proposal_id) for proposal_id in shadow_only}
    seq_real = {real["seq_by_id"].get(proposal_id) for proposal_id in real_only}
    seq_shadow.discard(None)
    seq_real.discard(None)
    gamma = max(int(shadow.get("gamma", 0)), int(real.get("gamma", 0)))
    return {
        "shadow_one_shot_committed_count": len(shadow_committed),
        "real_one_shot_committed_count": len(real_committed),
        "shadow_one_shot_committed_tokens": unique_token_sum(shadow_committed, shadow["token_by_id"], gamma),
        "real_one_shot_committed_tokens": unique_token_sum(real_committed, real["token_by_id"], gamma),
        "one_shot_committed_token_gap": (
            unique_token_sum(shadow_committed, shadow["token_by_id"], gamma)
            - unique_token_sum(real_committed, real["token_by_id"], gamma)
        ),
        "common_one_shot_committed_proposal_ids": sorted_ints(common),
        "shadow_only_one_shot_committed_proposal_ids": sorted_ints(shadow_only),
        "real_only_one_shot_committed_proposal_ids": sorted_ints(real_only),
        "shadow_only_one_shot_committed_seq_ids": sorted_ints(set(seq_shadow)),
        "real_only_one_shot_committed_seq_ids": sorted_ints(set(seq_real)),
        "one_shot_gap_reason_by_proposal_id": {str(key): value for key, value in sorted(reason_by_id.items())},
        "one_shot_gap_reason_counts": dict(Counter(reason_by_id.values())),
        "shadow_only_one_shot_examples": [
            {
                "proposal_id": proposal_id,
                "seq_id": shadow["seq_by_id"].get(proposal_id),
                "token_count": shadow["token_by_id"].get(proposal_id, gamma),
                "shadow_steps": shadow["steps_by_id"].get(proposal_id, []),
                "real_reason": reason_by_id.get(proposal_id, "unknown"),
            }
            for proposal_id in sorted(shadow_only)[:16]
        ],
    }


def compare_continuous(
    shadow: dict[str, Any],
    real: dict[str, Any],
    real_one_shot: dict[str, Any],
) -> dict[str, Any]:
    shadow_ready = set(shadow["ready_shadow_ids"])
    real_ready = set(real["ready_shadow_ids"])
    real_committed = set(real["real_committed_ids"])
    shadow_parents = {
        shadow["parent_by_id"].get(proposal_id)
        for proposal_id in shadow["candidate_ids"] | shadow_ready
        if shadow["parent_by_id"].get(proposal_id) is not None
    }
    real_parents = {
        real["parent_by_id"].get(proposal_id)
        for proposal_id in real["candidate_ids"] | real_ready | real_committed
        if real["parent_by_id"].get(proposal_id) is not None
    }
    real_shadow_ready_not_committed = real_ready - real_committed
    real_committed_not_ready = real_committed - real_ready
    direct_shadow_ready_not_real_committed = shadow_ready - real_committed
    direct_real_committed_not_shadow_ready = real_committed - shadow_ready
    reason_by_id: dict[int, str] = {}
    for proposal_id in real_shadow_ready_not_committed:
        reason_by_id[proposal_id] = first_non_empty_reason(
            real["real_skip_reason_by_id"].get(proposal_id),
            real["not_ready_reason_by_id"].get(proposal_id),
            "shadow_ready_not_real_committed_without_reason",
        )
    parent_status_by_id = {
        parent_id: parent_status(parent_id, real_one_shot)
        for parent_id in sorted(shadow_parents - real_parents)
    }
    gamma = max(int(shadow.get("gamma", 0)), int(real.get("gamma", 0)))
    return {
        "shadow_continuous_candidate_tokens": unique_token_sum(shadow["candidate_ids"], shadow["token_by_id"], gamma),
        "shadow_continuous_ready_tokens": unique_token_sum(shadow_ready, shadow["ready_token_by_id"], gamma),
        "real_continuous_candidate_tokens": unique_token_sum(real["candidate_ids"], real["token_by_id"], gamma),
        "real_continuous_shadow_ready_tokens": unique_token_sum(real_ready, real["ready_token_by_id"], gamma),
        "real_continuous_committed_tokens": unique_token_sum(real_committed, real["real_token_by_id"], gamma),
        "continuous_shadow_to_real_gap": (
            unique_token_sum(shadow_ready, shadow["ready_token_by_id"], gamma)
            - unique_token_sum(real_committed, real["real_token_by_id"], gamma)
        ),
        "common_continuous_parent_one_shot_proposal_ids": sorted_ints(shadow_parents & real_parents),
        "shadow_only_continuous_parent_proposal_ids": sorted_ints(shadow_parents - real_parents),
        "real_only_continuous_parent_proposal_ids": sorted_ints(real_parents - shadow_parents),
        "shadow_ready_but_not_real_committed_ids": sorted_ints(direct_shadow_ready_not_real_committed),
        "real_shadow_ready_but_not_real_committed_ids": sorted_ints(real_shadow_ready_not_committed),
        "real_committed_but_not_shadow_ready_ids": sorted_ints(real_committed_not_ready),
        "real_committed_but_not_shadow_trace_ready_ids_direct": sorted_ints(direct_real_committed_not_shadow_ready),
        "continuous_real_commit_skip_reason_counts": dict(Counter(real["real_skip_reason_by_id"].values())),
        "continuous_not_ready_reason_counts": dict(Counter(real["not_ready_reason_by_id"].values())),
        "continuous_gap_reason_by_proposal_id": {str(key): value for key, value in sorted(reason_by_id.items())},
        "continuous_gap_reason_counts": dict(Counter(reason_by_id.values())),
        "shadow_only_parent_status_in_real_by_parent_id": {
            str(key): value for key, value in parent_status_by_id.items()
        },
        "shadow_ready_but_not_real_examples": [
            {
                "proposal_id": proposal_id,
                "seq_id": real["seq_by_id"].get(proposal_id),
                "parent_proposal_id": real["parent_by_id"].get(proposal_id),
                "depth": real["depth_by_id"].get(proposal_id),
                "verify_result": real["verify_result_by_id"].get(proposal_id),
                "reason": reason_by_id.get(proposal_id, "unknown"),
            }
            for proposal_id in sorted(real_shadow_ready_not_committed)[:16]
        ],
        "continuous_true_frontier_mismatch_count": len(real["true_frontier_mismatch_ids"]),
        "continuous_parent_shadow_not_committed_count": len(real["parent_shadow_not_committed_ids"]),
        "continuous_parent_shadow_not_ready_count": len(real["parent_not_ready_ids"]),
        "continuous_duplicate_count": len(real["duplicate_ids"]),
        "continuous_lane_overlap_ids": sorted_ints(real["lane_overlap"]),
        "continuous_takeover_overlap_ids": sorted_ints(real["takeover_overlap"]),
        "continuous_depth2_real_commit_count": real["depth2_real_commit_count"],
    }


def aggregate_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    available_field_values: dict[str, list[float]] = defaultdict(list)
    missing_fields: list[str] = []
    for component, fields in TIMING_COMPONENT_FIELDS.items():
        seen_events: set[tuple[str, int, int, str, float]] = set()
        for record in records:
            step = int_value(record.get("step_id"), -1)
            plan = int_value(record.get("plan_id"), -1)
            side = str(
                record.get("continuous_eager_commit_side")
                or record.get("eager_commit_side")
                or record.get("runner_role")
                or ""
            )
            for field in fields:
                if field not in record:
                    continue
                try:
                    value = float(record.get(field) or 0.0)
                except Exception:
                    continue
                if value < 0:
                    continue
                event_key = (field, step, plan, side, value)
                if event_key in seen_events:
                    continue
                seen_events.add(event_key)
                available_field_values[component].append(value)
        if not available_field_values.get(component):
            missing_fields.extend(fields)

    timing_summary: dict[str, Any] = {}
    for component in TIMING_COMPONENT_FIELDS:
        values = available_field_values.get(component, [])
        if values:
            timing_summary[component] = {
                "available": True,
                "count": len(values),
                "sum_ms": sum(values),
                "min_ms": min(values),
                "median_ms": statistics.median(values),
                "max_ms": max(values),
            }
        else:
            timing_summary[component] = {
                "available": False,
                "count": 0,
                "sum_ms": 0.0,
                "min_ms": 0.0,
                "median_ms": 0.0,
                "max_ms": 0.0,
            }
    continuous_components = [
        "continuous_decision_broadcast_time_ms",
        "continuous_result_transfer_time_ms",
        "continuous_sync_apply_time_ms",
        "continuous_real_commit_time_ms",
        "continuous_verify_apply_dry_run_time_ms",
    ]
    timing_summary["total_continuous_overhead_time_ms"] = sum(
        float(timing_summary[name]["sum_ms"])
        for name in continuous_components
        if timing_summary[name]["available"]
    )
    timing_summary["timing_available"] = any(item["available"] for item in timing_summary.values() if isinstance(item, dict))
    timing_summary["missing_timing_fields"] = sorted(set(missing_fields))
    timing_summary["timing_notes"] = (
        ["timing fields are partially available"]
        if timing_summary["timing_available"]
        else ["timing unavailable for audited components; no timing values were synthesized"]
    )
    return timing_summary


def likely_reasons(one_shot: dict[str, Any], continuous: dict[str, Any], token_gap: int) -> list[str]:
    reasons: list[str] = []
    if token_gap == 0:
        reasons.append("shadow and real token totals match for the compared traces")
    else:
        reasons.append(
            "shadow trace is counterfactual while real continuous commit mutates sequence frontiers, so later scheduling and proposal identity can legitimately diverge"
        )
    if one_shot.get("one_shot_committed_token_gap", 0):
        counts = one_shot.get("one_shot_gap_reason_counts", {})
        if counts:
            reasons.append(f"one-shot gap attribution: {counts}")
    if continuous.get("continuous_shadow_to_real_gap", 0):
        reasons.append(
            "continuous shadow-to-real gap is the difference between 7b shadow-ready continuous tokens and 7c real continuous commits"
        )
    if continuous.get("continuous_parent_shadow_not_committed_count", 0):
        reasons.append("some deeper continuous opportunities are blocked because parent shadow proposals are not real committed")
    if continuous.get("continuous_true_frontier_mismatch_count", 0):
        reasons.append("true continuous frontier mismatches are present and should be inspected")
    return reasons


def build_audit(
    shadow_records: list[dict[str, Any]],
    real_records: list[dict[str, Any]],
    shadow_result: dict[str, Any] | None = None,
    real_result: dict[str, Any] | None = None,
    *,
    shadow_case: str = "shadow",
    real_case: str = "real",
) -> dict[str, Any]:
    shadow_result = shadow_result or {}
    real_result = real_result or {}
    shadow_accounting = aggregate_performance_accounting(shadow_records, shadow_result)
    real_accounting = aggregate_performance_accounting(real_records, real_result)
    shadow_one_shot = extract_one_shot(shadow_records)
    real_one_shot = extract_one_shot(real_records)
    shadow_continuous = extract_continuous(shadow_records)
    real_continuous = extract_continuous(real_records)
    one_shot = compare_one_shot(shadow_one_shot, real_one_shot, real_continuous)
    continuous = compare_continuous(shadow_continuous, real_continuous, real_one_shot)

    shadow_combined_estimated = int_value(
        shadow_accounting.get("combined_one_shot_plus_continuous_shadow_token_count"),
        one_shot["shadow_one_shot_committed_tokens"] + continuous["shadow_continuous_ready_tokens"],
    )
    real_combined_committed = int_value(
        real_accounting.get("combined_real_committed_token_count"),
        one_shot["real_one_shot_committed_tokens"] + continuous["real_continuous_committed_tokens"],
    )
    token_gap = shadow_combined_estimated - real_combined_committed
    one_shot_gap = one_shot["shadow_one_shot_committed_tokens"] - one_shot["real_one_shot_committed_tokens"]
    continuous_gap = continuous["shadow_continuous_ready_tokens"] - continuous["real_continuous_committed_tokens"]
    timing = aggregate_timing(real_records)

    unresolved: list[str] = []
    if one_shot.get("shadow_only_one_shot_committed_proposal_ids") and not one_shot.get("one_shot_gap_reason_counts"):
        unresolved.append("shadow-only one-shot proposals had no explicit real-trace reason")
    if continuous.get("real_committed_but_not_shadow_ready_ids"):
        unresolved.append("real continuous commits not found in real shadow-ready source require checker inspection")
    if not timing.get("timing_available"):
        unresolved.append("scoped timing fields are not instrumented for audited components")

    conclusions = {
        "likely_reasons": likely_reasons(one_shot, continuous, token_gap),
        "unresolved_questions": unresolved,
        "recommended_next_action": (
            "Inspect one_shot_gap_reason_counts and shadow_only_one_shot_examples first; if most are absent or same-seq shifted, compare scheduler/home-set evolution after continuous commits."
            if token_gap
            else "No token attribution gap detected; continue with performance-oriented diagnosis."
        ),
    }

    return {
        "shadow_case": shadow_case,
        "real_case": real_case,
        "shadow_combined_estimated_tokens": shadow_combined_estimated,
        "real_combined_committed_tokens": real_combined_committed,
        "token_gap": token_gap,
        "one_shot_gap": one_shot_gap,
        "continuous_gap": continuous_gap,
        "shadow_accounting": {
            "one_shot_committed_tokens": shadow_accounting.get("eager_committed_token_count", 0),
            "continuous_ready_shadow_tokens": shadow_accounting.get(
                "continuous_eager_commit_ready_shadow_token_count",
                0,
            ),
            "combined_estimated_tokens": shadow_accounting.get(
                "combined_one_shot_plus_continuous_shadow_token_count",
                0,
            ),
            "continuous_drop_reason_counts": shadow_accounting.get(
                "continuous_eager_drop_reason_counts",
                {},
            ),
        },
        "real_accounting": {
            "one_shot_committed_tokens": real_accounting.get("eager_committed_token_count", 0),
            "continuous_shadow_ready_tokens": real_accounting.get(
                "continuous_eager_commit_ready_shadow_token_count",
                0,
            ),
            "continuous_real_committed_tokens": real_accounting.get(
                "continuous_eager_real_committed_token_count",
                0,
            ),
            "combined_real_committed_tokens": real_accounting.get("combined_real_committed_token_count", 0),
            "continuous_real_commit_skip_reason_counts": real_accounting.get(
                "continuous_eager_real_commit_skip_reason_counts",
                {},
            ),
        },
        "one_shot": one_shot,
        "continuous": continuous,
        "timing": timing,
        "conclusions": conclusions,
    }


def print_report(audit: dict[str, Any]) -> None:
    print(f"shadow_case={audit['shadow_case']}")
    print(f"real_case={audit['real_case']}")
    for key in (
        "shadow_combined_estimated_tokens",
        "real_combined_committed_tokens",
        "token_gap",
        "one_shot_gap",
        "continuous_gap",
    ):
        print(f"{key}={audit.get(key)}")

    print("one_shot:")
    one_shot = audit["one_shot"]
    for key in (
        "shadow_one_shot_committed_tokens",
        "real_one_shot_committed_tokens",
        "one_shot_committed_token_gap",
        "one_shot_gap_reason_counts",
        "shadow_only_one_shot_committed_proposal_ids",
        "shadow_only_one_shot_committed_seq_ids",
    ):
        print(f"  {key}={one_shot.get(key)}")

    print("continuous:")
    continuous = audit["continuous"]
    for key in (
        "shadow_continuous_ready_tokens",
        "real_continuous_shadow_ready_tokens",
        "real_continuous_committed_tokens",
        "continuous_shadow_to_real_gap",
        "continuous_gap_reason_counts",
        "continuous_real_commit_skip_reason_counts",
        "continuous_true_frontier_mismatch_count",
        "continuous_parent_shadow_not_committed_count",
        "continuous_depth2_real_commit_count",
    ):
        print(f"  {key}={continuous.get(key)}")

    print("timing:")
    timing = audit["timing"]
    print(f"  timing_available={timing.get('timing_available')}")
    print(f"  total_continuous_overhead_time_ms={timing.get('total_continuous_overhead_time_ms')}")
    for component in (
        "continuous_decision_broadcast_time_ms",
        "continuous_result_transfer_time_ms",
        "continuous_sync_apply_time_ms",
        "continuous_real_commit_time_ms",
        "continuous_verify_apply_dry_run_time_ms",
        "one_shot_real_commit_time_ms",
    ):
        value = timing.get(component, {})
        print(f"  {component}={value.get('sum_ms', 0.0)} available={value.get('available', False)}")
    if timing.get("missing_timing_fields"):
        print(f"  missing_timing_fields={timing['missing_timing_fields']}")

    print("conclusions:")
    for reason in audit["conclusions"].get("likely_reasons", []):
        print(f"  likely_reason={reason}")
    for question in audit["conclusions"].get("unresolved_questions", []):
        print(f"  unresolved={question}")
    print(f"  recommended_next_action={audit['conclusions'].get('recommended_next_action')}")


def synthetic_one_shot_record(
    proposal_id: int,
    seq_id: int,
    *,
    committed: bool,
    reason: str = "",
    side: str = "target",
    step: int = 1,
    plan: int = 11,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "enable_eager_commit_ready_only": True,
        "enable_eager_commit_readiness_dry_run": True,
        "eager_commit_enabled": True,
        "eager_commit_source": "phase1h5e3_takeover_lane",
        "eager_commit_candidate_proposal_ids": [proposal_id],
        "eager_commit_candidate_seq_ids": [seq_id],
        "eager_commit_ready_proposal_ids": [proposal_id],
        "eager_commit_side": side,
        "eager_commit_step_id": step,
        "eager_commit_plan_id": plan,
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "target_eager_set": [],
    }
    if committed:
        record.update(
            {
                "eager_committed_proposal_ids": [proposal_id],
                "eager_committed_seq_ids": [seq_id],
                "eager_committed_token_count_by_proposal_id": {str(proposal_id): 4},
                "eager_committed_accept_len_by_proposal_id": {str(proposal_id): 4},
                "eager_committed_action_by_proposal_id": {str(proposal_id): FULL_ACCEPT_ACTION},
                "eager_committed_verify_result_by_proposal_id": {str(proposal_id): "full_accept"},
                "eager_commit_precondition_ok_by_proposal_id": {str(proposal_id): True},
                "eager_commit_precondition_failed_by_proposal_id": {str(proposal_id): False},
                "eager_tokens_committed": 4,
                "eager_tokens_verified": 4,
                "eager_tokens_accepted": 4,
                "eager_tokens_rejected": 0,
                "eager_tokens_invalidated": 0,
            }
        )
    else:
        record.update(
            {
                "eager_committed_proposal_ids": [],
                "eager_commit_skipped_proposal_ids": [proposal_id],
                "eager_commit_skip_reason_by_proposal_id": {str(proposal_id): reason or "not_full_accept"},
            }
        )
    return record


def synthetic_continuous_record(
    proposal_id: int,
    parent_id: int,
    seq_id: int,
    *,
    ready: bool,
    committed: bool,
    reason: str = "",
    side: str = "target",
    step: int = 2,
    plan: int = 12,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "enable_continuous_eager_dry_run": True,
        "enable_continuous_eager_verify_apply_dry_run": True,
        "continuous_eager_dry_run_enabled": True,
        "continuous_eager_source": CONTINUOUS_SOURCE,
        "continuous_eager_execution_stage": "verify_apply_dry_run",
        "continuous_shadow_stage": "verify_apply_dry_run",
        "continuous_eager_candidate_proposal_ids": [proposal_id],
        "continuous_eager_candidate_seq_ids": [seq_id],
        "continuous_eager_candidate_token_count_by_proposal_id": {str(proposal_id): 4},
        "continuous_eager_parent_proposal_id_by_proposal_id": {str(proposal_id): parent_id},
        "continuous_eager_parent_source_by_proposal_id": {str(proposal_id): ONE_SHOT_PARENT_SOURCE},
        "continuous_eager_chain_depth_by_proposal_id": {str(proposal_id): 1},
        "continuous_eager_verified_proposal_ids": [proposal_id],
        "continuous_eager_full_accept_proposal_ids": [proposal_id] if ready else [],
        "continuous_eager_verify_result_by_proposal_id": {
            str(proposal_id): "full_accept" if ready else "partial_accept"
        },
        "continuous_eager_accept_len_by_proposal_id": {str(proposal_id): 4 if ready else 2},
        "continuous_eager_commit_ready_shadow_proposal_ids": [proposal_id] if ready else [],
        "continuous_eager_commit_ready_shadow_token_count_by_proposal_id": (
            {str(proposal_id): 4} if ready else {}
        ),
        "continuous_eager_not_ready_shadow_proposal_ids": [] if ready else [proposal_id],
        "continuous_eager_not_ready_shadow_reason_by_proposal_id": (
            {} if ready else {str(proposal_id): reason or "continuous_not_full_accept"}
        ),
        "continuous_eager_real_commit_count": 0,
        "continuous_depth2_real_commit_count": 0,
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "target_eager_set": [],
    }
    if committed:
        record.update(
            {
                "enable_continuous_eager_commit_depth1_ready_only": True,
                "continuous_eager_commit_enabled": True,
                "continuous_eager_commit_source": "continuous_depth1_ready_only",
                "continuous_eager_commit_side": side,
                "continuous_eager_commit_step_id": step,
                "continuous_eager_commit_plan_id": plan,
                "continuous_eager_real_committed_proposal_ids": [proposal_id],
                "continuous_eager_real_committed_seq_ids": [seq_id],
                "continuous_eager_real_committed_token_count_by_proposal_id": {str(proposal_id): 4},
                "continuous_eager_tokens_verified": 4,
                "continuous_eager_tokens_accepted": 4,
                "continuous_eager_tokens_rejected": 0,
                "continuous_eager_tokens_invalidated": 0,
                "continuous_eager_real_commit_count": 1,
            }
        )
    return record


def run_synthetic() -> None:
    shadow_records = [
        synthetic_one_shot_record(1, 7, committed=True),
        synthetic_one_shot_record(2, 9, committed=True, step=3, plan=13),
        synthetic_continuous_record(9001, 1, 7, ready=True, committed=False),
        synthetic_continuous_record(9002, 2, 9, ready=True, committed=False),
    ]
    real_records = [
        synthetic_one_shot_record(1, 7, committed=True),
        synthetic_one_shot_record(2, 9, committed=False, reason="frontier_mismatch", step=3, plan=13),
        synthetic_continuous_record(9001, 1, 7, ready=True, committed=True),
        synthetic_continuous_record(9002, 2, 9, ready=False, committed=False, reason="continuous_not_full_accept"),
    ]
    shadow_result = {"metrics": {"overall": {"total_output_tokens": 64}}}
    real_result = {"metrics": {"overall": {"total_output_tokens": 64}}}
    audit = build_audit(
        shadow_records,
        real_records,
        shadow_result,
        real_result,
        shadow_case="synthetic_shadow",
        real_case="synthetic_real",
    )
    assert audit["shadow_combined_estimated_tokens"] == 16, audit
    assert audit["real_combined_committed_tokens"] == 8, audit
    assert audit["token_gap"] == 8, audit
    assert audit["one_shot_gap"] == 4, audit
    assert audit["continuous_gap"] == 4, audit
    assert audit["one_shot"]["one_shot_gap_reason_counts"].get("frontier_mismatch") == 1, audit
    print("Synthetic 7c shadow-vs-real audit passed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Phase 1H-7b shadow continuous potential against Phase 1H-7c real continuous commits."
    )
    parser.add_argument("--shadow-trace", type=Path)
    parser.add_argument("--shadow-result", type=Path)
    parser.add_argument("--real-trace", type=Path)
    parser.add_argument("--real-result", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
        return 0

    if args.shadow_trace is None or args.real_trace is None:
        parser.error("--shadow-trace and --real-trace are required unless --synthetic is used")

    shadow_records = load_trace(args.shadow_trace)
    real_records = load_trace(args.real_trace)
    shadow_result = load_json(args.shadow_result) if args.shadow_result else {}
    real_result = load_json(args.real_result) if args.real_result else {}
    audit = build_audit(
        shadow_records,
        real_records,
        shadow_result,
        real_result,
        shadow_case=str(args.shadow_trace),
        real_case=str(args.real_trace),
    )
    print_report(audit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
