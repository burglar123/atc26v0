#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import int_value  # noqa: E402
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    load_json,
    trace_payload_to_records,
)


def load_trace(path: Path) -> list[dict[str, Any]]:
    return trace_payload_to_records(json.loads(path.read_text(encoding="utf-8")))


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


def as_depth_int_lists(value: Any) -> dict[int, list[int]]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, list[int]] = {}
    for key, items in value.items():
        try:
            depth = int(key)
        except Exception:
            continue
        result[depth] = as_int_list(items)
    return result


def as_depth_int_map(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, int] = {}
    for key, item in value.items():
        try:
            result[int(key)] = int(item)
        except Exception:
            continue
    return result


def merge_depth_lists(records: list[dict[str, Any]], field: str) -> dict[int, list[int]]:
    merged: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for record in records:
        for depth, proposal_ids in as_depth_int_lists(record.get(field)).items():
            for proposal_id in proposal_ids:
                if proposal_id in seen[depth]:
                    continue
                seen[depth].add(proposal_id)
                merged[depth].append(proposal_id)
    return {depth: values for depth, values in sorted(merged.items())}


def merge_int_map(records: list[dict[str, Any]], *fields: str) -> dict[int, int]:
    merged: dict[int, int] = {}
    for record in records:
        for field in fields:
            for key, value in as_int_map(record.get(field)).items():
                merged.setdefault(key, value)
    return merged


def max_record_int(records: list[dict[str, Any]], *fields: str) -> int:
    value = 0
    for record in records:
        for field in fields:
            value = max(value, int_value(record.get(field), 0))
    return value


def sum_depth_values(value: Any) -> int:
    return sum(int_value(item, 0) for item in (value or {}).values()) if isinstance(value, dict) else 0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def merge_depth_counts(records: list[dict[str, Any]], *fields: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for record in records:
        for field in fields:
            for depth, value in as_depth_int_map(record.get(field)).items():
                key = str(int(depth))
                merged[key] = max(int(merged.get(key, 0)), int(value))
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def count_depth_lists(value: dict[int, list[int]]) -> dict[str, int]:
    return {
        str(int(depth)): len(set(int(item) for item in items))
        for depth, items in sorted(value.items())
    }


def depth_list_token_counts(
    records: list[dict[str, Any]],
    depth_lists: dict[int, list[int]],
    *token_fields: str,
) -> dict[str, int]:
    token_by_id = merge_int_map(
        records,
        *(token_fields or ("generic_rolling_token_count_by_proposal_id",)),
    )
    counts: dict[str, int] = {}
    for depth, proposal_ids in sorted(depth_lists.items()):
        total = 0
        for proposal_id in set(int(item) for item in proposal_ids):
            total += int(token_by_id.get(proposal_id, 0))
        if total:
            counts[str(int(depth))] = int(total)
    return counts


def record_step_key(record: dict[str, Any], index: int) -> tuple[int, int]:
    return (
        int_value(record.get("step_id"), int_value(record.get("eager_commit_step_id"), index)),
        int_value(record.get("plan_id"), int_value(record.get("eager_commit_plan_id"), -1)),
    )


def record_max_depth(record: dict[str, Any], *fields: str) -> int:
    depths: set[int] = set()
    for field in fields:
        depths.update(as_depth_int_lists(record.get(field)).keys())
        depths.update(as_depth_int_map(record.get(field)).keys())
    return max(depths or {0})


def stop_reasons_by_depth(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_depth: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if not (
            bool(record.get("unified_generic_rolling_enabled", False))
            or bool(record.get("enable_unified_generic_rolling_runtime", False))
        ):
            continue
        reasons = record.get("generic_full_continuous_stop_reason_counts")
        if not isinstance(reasons, dict):
            continue
        depth = max(
            int_value(record.get("unified_generic_max_observed_depth"), 0),
            record_max_depth(
                record,
                "generic_rolling_candidate_proposal_ids_by_depth",
                "generic_rolling_ready_proposal_ids_by_depth",
                "generic_rolling_real_committed_proposal_ids_by_depth",
                "unified_generic_depth_candidate_token_counts",
                "unified_generic_depth_ready_token_counts",
                "unified_generic_depth_commit_token_counts",
            ),
        )
        if depth <= 0:
            depth = 1
        for reason, count in reasons.items():
            by_depth[str(int(depth))][str(reason)] += int_value(count, 0)
    return {
        depth: dict(sorted(counter.items()))
        for depth, counter in sorted(by_depth.items(), key=lambda item: int(item[0]))
    }


def reason_count_by_depth(reason_by_depth: dict[str, dict[str, int]], reason: str) -> dict[str, int]:
    return {
        depth: int(counts.get(reason, 0))
        for depth, counts in reason_by_depth.items()
        if int(counts.get(reason, 0)) > 0
    }


def utilization_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[tuple[int, int], dict[str, int]] = {}
    for index, record in enumerate(records):
        if not (
            bool(record.get("unified_generic_rolling_enabled", False))
            or bool(record.get("enable_unified_generic_rolling_runtime", False))
        ):
            continue
        key = record_step_key(record, index)
        stat = by_step.setdefault(
            key,
            {
                "candidate_seqs": 0,
                "ready_seqs": 0,
                "committed_seqs": 0,
                "candidate_tokens": 0,
                "committed_tokens": 0,
                "candidate_depth": 0,
                "committed_depth": 0,
            },
        )
        candidate_seq_count = sum(
            len(set(items))
            for items in as_depth_int_lists(record.get("generic_rolling_candidate_seq_ids_by_depth")).values()
        )
        ready_seq_count = sum(
            len(set(items))
            for items in as_depth_int_lists(record.get("generic_rolling_ready_seq_ids_by_depth")).values()
        )
        committed_seq_count = sum(
            len(set(items))
            for items in as_depth_int_lists(record.get("generic_rolling_real_committed_seq_ids_by_depth")).values()
        )
        candidate_tokens = sum_depth_values(
            record.get("unified_generic_depth_candidate_token_counts")
            or record.get("generic_full_continuous_depth_candidate_token_counts")
        )
        committed_tokens = sum_depth_values(
            record.get("unified_generic_depth_commit_token_counts")
            or record.get("generic_full_continuous_depth_commit_token_counts")
        )
        stat["candidate_seqs"] = max(stat["candidate_seqs"], int(candidate_seq_count))
        stat["ready_seqs"] = max(stat["ready_seqs"], int(ready_seq_count))
        stat["committed_seqs"] = max(stat["committed_seqs"], int(committed_seq_count))
        stat["candidate_tokens"] = max(stat["candidate_tokens"], int(candidate_tokens))
        stat["committed_tokens"] = max(stat["committed_tokens"], int(committed_tokens))
        stat["candidate_depth"] = max(
            stat["candidate_depth"],
            record_max_depth(
                record,
                "generic_rolling_candidate_proposal_ids_by_depth",
                "unified_generic_depth_candidate_token_counts",
            ),
        )
        stat["committed_depth"] = max(
            stat["committed_depth"],
            record_max_depth(
                record,
                "generic_rolling_real_committed_proposal_ids_by_depth",
                "unified_generic_depth_commit_token_counts",
            ),
        )
    num_steps = len(by_step)
    values = list(by_step.values())
    return {
        "num_steps": num_steps,
        "steps_with_any_unified_candidate": sum(1 for item in values if item["candidate_seqs"] > 0),
        "steps_with_any_unified_commit": sum(1 for item in values if item["committed_seqs"] > 0),
        "avg_candidate_seqs_per_step": (
            sum(item["candidate_seqs"] for item in values) / num_steps if num_steps else 0.0
        ),
        "avg_ready_seqs_per_step": (
            sum(item["ready_seqs"] for item in values) / num_steps if num_steps else 0.0
        ),
        "avg_committed_seqs_per_step": (
            sum(item["committed_seqs"] for item in values) / num_steps if num_steps else 0.0
        ),
        "avg_candidate_tokens_per_step": (
            sum(item["candidate_tokens"] for item in values) / num_steps if num_steps else 0.0
        ),
        "avg_committed_tokens_per_step": (
            sum(item["committed_tokens"] for item in values) / num_steps if num_steps else 0.0
        ),
        "max_candidate_depth_per_step": max((item["candidate_depth"] for item in values), default=0),
        "max_committed_depth_per_step": max((item["committed_depth"] for item in values), default=0),
    }


def collect_descendants(parent_by_id: dict[int, int], root_id: int) -> set[int]:
    children: dict[int, set[int]] = defaultdict(set)
    for child_id, parent_id in parent_by_id.items():
        children[int(parent_id)].add(int(child_id))
    descendants: set[int] = set()
    stack = list(children.get(int(root_id), set()))
    while stack:
        proposal_id = stack.pop()
        if proposal_id in descendants:
            continue
        descendants.add(proposal_id)
        stack.extend(children.get(proposal_id, set()))
    return descendants


def build_summary(records: list[dict[str, Any]], result_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result_payload = result_payload or {}
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(result_args, dict):
        result_args = {}
    accounting = aggregate_performance_accounting(records, result_payload)
    unified_enabled = any(
        bool(record.get("unified_generic_rolling_enabled", False))
        or bool(record.get("enable_unified_generic_rolling_runtime", False))
        for record in records
    ) or bool(result_args.get("enable_unified_generic_rolling_runtime", False))
    committed_by_depth = merge_depth_lists(records, "generic_rolling_real_committed_proposal_ids_by_depth")
    candidate_by_depth = merge_depth_lists(records, "generic_rolling_candidate_proposal_ids_by_depth")
    ready_by_depth = merge_depth_lists(records, "generic_rolling_ready_proposal_ids_by_depth")
    parent_by_id = merge_int_map(
        records,
        "generic_rolling_parent_by_proposal_id",
        "generic_rolling_real_commit_parent_by_proposal_id",
    )
    depth_by_id = merge_int_map(
        records,
        "generic_rolling_depth_by_proposal_id",
        "generic_rolling_real_commit_depth_by_proposal_id",
    )
    committed_depth_by_id: dict[int, int] = {}
    duplicate_counter: Counter[int] = Counter()
    for depth, proposal_ids in committed_by_depth.items():
        for proposal_id in proposal_ids:
            duplicate_counter[proposal_id] += 1
            committed_depth_by_id.setdefault(proposal_id, int(depth_by_id.get(proposal_id, depth)))

    committed_ids = set(committed_depth_by_id)
    missing_parent_ids: list[int] = []
    for proposal_id, depth in sorted(committed_depth_by_id.items()):
        if depth == 1:
            continue
        parent_id = parent_by_id.get(proposal_id)
        if parent_id is None or parent_id not in committed_depth_by_id:
            missing_parent_ids.append(proposal_id)
            continue
        if int(committed_depth_by_id[parent_id]) != depth - 1:
            missing_parent_ids.append(proposal_id)

    partial_ids = set()
    for record in records:
        partial_ids.update(as_int_list(record.get("partial_prefix_recovered_proposal_ids")))
    descendant_after_partial: set[int] = set()
    for proposal_id in partial_ids:
        descendant_after_partial.update(collect_descendants(parent_by_id, proposal_id) & committed_ids)

    configured_max = max(
        max_record_int(records, "unified_generic_max_depth", "generic_full_continuous_max_depth", "generic_rolling_max_depth"),
        int_value(result_args.get("max_rolling_continuous_depth"), 0),
    )
    max_observed = max(
        max_record_int(records, "unified_generic_max_observed_depth", "generic_full_continuous_max_observed_depth"),
        max([*candidate_by_depth.keys(), *ready_by_depth.keys(), *committed_by_depth.keys(), 0]),
    )
    max_real = max(
        max_record_int(records, "unified_generic_max_real_committed_depth", "generic_full_continuous_max_real_committed_depth"),
        max(committed_by_depth.keys() or [0]),
    )
    depth_commit_counts = {}
    for record in records:
        for field in ("unified_generic_depth_commit_token_counts", "generic_full_continuous_depth_commit_token_counts"):
            for depth, token_count in as_depth_int_map(record.get(field)).items():
                depth_commit_counts[str(depth)] = max(int(depth_commit_counts.get(str(depth), 0)), int(token_count))
    candidate_seq_counts = count_depth_lists(
        merge_depth_lists(records, "generic_rolling_candidate_seq_ids_by_depth")
    )
    ready_seq_counts = count_depth_lists(merge_depth_lists(records, "generic_rolling_ready_seq_ids_by_depth"))
    committed_seq_counts = count_depth_lists(
        merge_depth_lists(records, "generic_rolling_real_committed_seq_ids_by_depth")
    )
    candidate_token_counts = depth_list_token_counts(
        records,
        candidate_by_depth,
        "generic_rolling_token_count_by_proposal_id",
    ) or merge_depth_counts(
        records,
        "unified_generic_depth_candidate_token_counts",
        "generic_full_continuous_depth_candidate_token_counts",
    )
    ready_token_counts = depth_list_token_counts(
        records,
        ready_by_depth,
        "generic_rolling_token_count_by_proposal_id",
    ) or merge_depth_counts(
        records,
        "unified_generic_depth_ready_token_counts",
        "generic_full_continuous_depth_ready_token_counts",
    )
    committed_token_counts = depth_list_token_counts(
        records,
        committed_by_depth,
        "generic_rolling_real_committed_token_count_by_proposal_id",
        "generic_rolling_token_count_by_proposal_id",
    ) or merge_depth_counts(
        records,
        "unified_generic_depth_commit_token_counts",
        "generic_full_continuous_depth_commit_token_counts",
        "generic_rolling_real_committed_token_count_by_depth",
    )
    if not depth_commit_counts:
        depth_commit_counts = {str(depth): int(value) for depth, value in committed_token_counts.items()}
    commit_share_by_depth = {
        depth: safe_div(float(committed_token_counts.get(depth, 0)), float(candidate_token_counts.get(depth, 0)))
        for depth in sorted(set(candidate_token_counts) | set(committed_token_counts), key=int)
    }
    reason_by_depth = stop_reasons_by_depth(records)
    utilization = utilization_summary(records)
    unified_total_output = max_record_int(records, "unified_generic_total_output_token_count")
    if unified_enabled and unified_total_output > 0:
        total_full = max_record_int(records, "unified_generic_total_full_commit_token_count")
        total_partial = max_record_int(records, "unified_generic_total_partial_recovered_token_count")
        total_revised = max_record_int(records, "unified_generic_total_revised_token_count")
        total_output = unified_total_output
    else:
        total_full = max_record_int(records, "generic_full_continuous_total_full_commit_token_count")
        total_partial = max_record_int(
            records,
            "generic_full_continuous_total_partial_recovered_token_count",
        )
        total_revised = max_record_int(records, "generic_full_continuous_total_revised_token_count")
        total_output = max_record_int(records, "generic_full_continuous_total_output_token_count")
    if total_full == 0 and depth_commit_counts:
        total_full = sum(int(value) for value in depth_commit_counts.values())
    if total_output == 0:
        total_output = total_full + total_partial

    return {
        "unified_generic_rolling_enabled": bool(unified_enabled),
        "configured_max_depth": configured_max,
        "max_observed_depth": max_observed,
        "max_real_committed_depth": max_real,
        "candidate_depths": sorted(candidate_by_depth),
        "ready_depths": sorted(ready_by_depth),
        "committed_depths": sorted(committed_by_depth),
        "depth_commit_token_counts": dict(sorted(depth_commit_counts.items(), key=lambda item: int(item[0]))),
        "unified_candidate_seq_count_by_depth": candidate_seq_counts,
        "unified_ready_seq_count_by_depth": ready_seq_counts,
        "unified_committed_seq_count_by_depth": committed_seq_counts,
        "unified_candidate_token_count_by_depth": candidate_token_counts,
        "unified_ready_token_count_by_depth": ready_token_counts,
        "unified_committed_token_count_by_depth": committed_token_counts,
        "unified_commit_share_by_depth": commit_share_by_depth,
        "unified_parent_not_full_accept_count_by_depth": reason_count_by_depth(
            reason_by_depth,
            "parent_not_full_accept",
        ),
        "unified_stop_reason_counts_by_depth": reason_by_depth,
        "unified_no_eligible_parent_count_by_depth": reason_count_by_depth(
            reason_by_depth,
            "no_eligible_parent",
        ),
        "unified_no_eligible_ready_child_count_by_depth": reason_count_by_depth(
            reason_by_depth,
            "no_eligible_ready_child",
        ),
        "unified_sequence_finished_count_by_depth": reason_count_by_depth(
            reason_by_depth,
            "sequence_finished",
        ),
        **utilization,
        "total_full_commit_token_count": total_full,
        "total_partial_recovered_token_count": total_partial,
        "total_revised_token_count": total_revised,
        "total_output_token_count": total_output,
        "combined_real_committed_token_count": int_value(accounting.get("combined_real_committed_token_count"), 0),
        "partial_prefix_accepted_token_count": int_value(accounting.get("partial_prefix_accepted_token_count"), 0),
        "partial_prefix_revised_token_count": int_value(accounting.get("partial_prefix_revised_token_count"), 0),
        "partial_prefix_total_recovered_token_count": int_value(
            accounting.get("partial_prefix_total_recovered_token_count"), 0
        ),
        "normal_lane_conflict_count": max(
            max_record_int(records, "unified_generic_normal_lane_conflict_count"),
            int_value(accounting.get("generic_full_continuous_normal_lane_conflict_count"), 0),
        ),
        "target_draft_mismatch_count": max(
            max_record_int(records, "unified_generic_target_draft_mismatch_count"),
            int_value(accounting.get("generic_full_continuous_target_draft_mismatch_count"), 0),
        ),
        "parity_ok": all(
            bool(record.get("unified_generic_parity_ok", record.get("generic_full_continuous_parity_ok", True)))
            for record in records
            if bool(record.get("unified_generic_rolling_enabled", False))
            or bool(record.get("enable_unified_generic_rolling_runtime", False))
        ),
        "duplicate_committed_proposal_ids": sorted(
            proposal_id for proposal_id, count in duplicate_counter.items() if count > 1
        ),
        "missing_parent_commit_proposal_ids": sorted(set(missing_parent_ids)),
        "descendant_committed_after_partial_ids": sorted(descendant_after_partial),
        "depth_gt_max_commit_ids": sorted(
            proposal_id
            for proposal_id, depth in committed_depth_by_id.items()
            if configured_max and int(depth) > configured_max
        ),
    }


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    summary = build_summary(records, result_payload)
    errors: list[str] = []
    if not summary["unified_generic_rolling_enabled"]:
        errors.append("unified generic rolling runtime must be enabled")
        return errors, summary

    all_depths = set(summary["candidate_depths"]) | set(summary["ready_depths"]) | set(summary["committed_depths"])
    if not all_depths:
        errors.append("unified generic runtime produced no depth-indexed activity")
    elif min(all_depths) != 1:
        errors.append(f"unified generic depths must start at 1, got min_depth={min(all_depths)}")
    if 0 in all_depths:
        errors.append("unified generic runtime must not emit depth 0 nodes")

    max_depth = int_value(summary["configured_max_depth"], 0)
    max_observed = int_value(summary["max_observed_depth"], 0)
    max_real = int_value(summary["max_real_committed_depth"], 0)
    if max_depth <= 0:
        errors.append("configured max depth must be positive")
    if max_observed > max_depth:
        errors.append("max observed depth exceeds configured max depth")
    if max_real > max_depth:
        errors.append("max real committed depth exceeds configured max depth")
    if max_depth > 4 and max_real <= 4:
        errors.append("max real committed depth must exceed 4 when configured max depth exceeds 4")

    total_full = int_value(summary["total_full_commit_token_count"], 0)
    total_partial = int_value(summary["total_partial_recovered_token_count"], 0)
    total_revised = int_value(summary["total_revised_token_count"], 0)
    total_output = int_value(summary["total_output_token_count"], 0)
    if total_output != total_full + total_partial:
        errors.append("unified total output must equal full commits plus partial recovered tokens")
    if total_output != int_value(summary["combined_real_committed_token_count"], 0):
        errors.append("unified total output must equal combined real committed token count")
    if total_partial != int_value(summary["partial_prefix_total_recovered_token_count"], 0):
        errors.append("unified partial recovered total must match partial-prefix total")
    if total_revised != int_value(summary["partial_prefix_revised_token_count"], 0):
        errors.append("unified revised total must match partial-prefix revised token count")
    if total_partial:
        accepted = int_value(summary["partial_prefix_accepted_token_count"], 0)
        if total_partial != accepted + total_revised:
            errors.append("partial recovered total must equal accepted prefix plus revised tokens")

    if int_value(summary["normal_lane_conflict_count"], 0) != 0:
        errors.append("normal lane conflict count must be zero")
    if int_value(summary["target_draft_mismatch_count"], 0) != 0:
        errors.append("target/draft mismatch count must be zero")
    if not bool(summary["parity_ok"]):
        errors.append("unified generic parity flag must be true")
    if summary["duplicate_committed_proposal_ids"]:
        errors.append(f"duplicate proposal commits: {summary['duplicate_committed_proposal_ids']}")
    if summary["missing_parent_commit_proposal_ids"]:
        errors.append(f"missing parent commits: {summary['missing_parent_commit_proposal_ids']}")
    if summary["descendant_committed_after_partial_ids"]:
        errors.append(
            f"descendants committed after partial recovery: {summary['descendant_committed_after_partial_ids']}"
        )
    if summary["depth_gt_max_commit_ids"]:
        errors.append(f"commits beyond max depth: {summary['depth_gt_max_commit_ids']}")
    if sum_depth_values(summary["depth_commit_token_counts"]) != total_full:
        errors.append("depth commit token counts must sum to total full commit tokens")
    return errors, summary


def synthetic_payload(total_output_tokens: int) -> dict[str, Any]:
    return {
        "args": {
            "enable_unified_generic_rolling_runtime": True,
            "enable_full_continuous_eager": True,
            "enable_generic_rolling_runtime_loop": True,
            "enable_generic_rolling_apply_path": True,
            "max_rolling_continuous_depth": 6,
        },
        "metrics": {"total_output_tokens": int(total_output_tokens)},
    }


def synthetic_records() -> list[dict[str, Any]]:
    ids_by_depth = {depth: [1000 + depth] for depth in range(1, 7)}
    parent_by_id = {1000 + depth: 999 + depth for depth in range(2, 7)}
    depth_by_id = {1000 + depth: depth for depth in range(1, 7)}
    token_by_id = {1000 + depth: 4 for depth in range(1, 7)}
    full = sum(token_by_id.values())
    partial = 3
    revised = 1
    return [
        {
            "normal_gamma": 4,
            "unified_generic_rolling_enabled": True,
            "enable_unified_generic_rolling_runtime": True,
            "generic_full_continuous_enabled": True,
            "enable_full_continuous_eager": True,
            "generic_rolling_runtime_enabled": True,
            "enable_generic_rolling_runtime_loop": True,
            "generic_rolling_apply_path_enabled": True,
            "enable_generic_rolling_apply_path": True,
            "unified_generic_max_depth": 6,
            "unified_generic_max_observed_depth": 6,
            "unified_generic_max_real_committed_depth": 6,
            "generic_full_continuous_max_depth": 6,
            "generic_full_continuous_max_observed_depth": 6,
            "generic_full_continuous_max_real_committed_depth": 6,
            "generic_rolling_candidate_proposal_ids_by_depth": {
                str(depth): list(ids) for depth, ids in ids_by_depth.items()
            },
            "generic_rolling_candidate_seq_ids_by_depth": {str(depth): [7] for depth in ids_by_depth},
            "generic_rolling_ready_proposal_ids_by_depth": {
                str(depth): list(ids) for depth, ids in ids_by_depth.items()
            },
            "generic_rolling_ready_seq_ids_by_depth": {str(depth): [7] for depth in ids_by_depth},
            "generic_rolling_real_committed_proposal_ids_by_depth": {
                str(depth): list(ids) for depth, ids in ids_by_depth.items()
            },
            "generic_rolling_real_committed_seq_ids_by_depth": {str(depth): [7] for depth in ids_by_depth},
            "generic_rolling_parent_by_proposal_id": {str(k): v for k, v in parent_by_id.items()},
            "generic_rolling_real_commit_parent_by_proposal_id": {str(k): v for k, v in parent_by_id.items()},
            "generic_rolling_root_by_proposal_id": {str(1000 + depth): 1001 for depth in range(1, 7)},
            "generic_rolling_real_commit_root_by_proposal_id": {
                str(1000 + depth): 1001 for depth in range(1, 7)
            },
            "generic_rolling_depth_by_proposal_id": {str(k): v for k, v in depth_by_id.items()},
            "generic_rolling_real_commit_depth_by_proposal_id": {str(k): v for k, v in depth_by_id.items()},
            "generic_rolling_token_count_by_proposal_id": {str(k): v for k, v in token_by_id.items()},
            "generic_rolling_real_committed_token_count_by_proposal_id": {
                str(k): v for k, v in token_by_id.items()
            },
            "generic_rolling_real_committed_accept_len_by_proposal_id": {
                str(k): v for k, v in token_by_id.items()
            },
            "generic_rolling_real_commit_action_by_proposal_id": {
                str(k): "append_full_accept_real_commit" for k in token_by_id
            },
            "generic_rolling_real_commit_verify_result_by_proposal_id": {
                str(k): "full_accept" for k in token_by_id
            },
            "generic_rolling_real_committed_token_count_by_depth": {
                str(depth): 4 for depth in ids_by_depth
            },
            "generic_rolling_real_committed_proposal_count_by_depth": {
                str(depth): 1 for depth in ids_by_depth
            },
            "generic_full_continuous_depth_commit_token_counts": {
                str(depth): 4 for depth in ids_by_depth
            },
            "generic_full_continuous_depth_candidate_token_counts": {
                str(depth): 4 for depth in ids_by_depth
            },
            "generic_full_continuous_depth_ready_token_counts": {
                str(depth): 4 for depth in ids_by_depth
            },
            "unified_generic_depth_commit_token_counts": {str(depth): 4 for depth in ids_by_depth},
            "unified_generic_depth_candidate_token_counts": {str(depth): 4 for depth in ids_by_depth},
            "unified_generic_depth_ready_token_counts": {str(depth): 4 for depth in ids_by_depth},
            "generic_full_continuous_total_full_commit_token_count": full,
            "generic_full_continuous_total_partial_recovered_token_count": partial,
            "generic_full_continuous_total_revised_token_count": revised,
            "generic_full_continuous_total_output_token_count": 1244,
            "unified_generic_total_full_commit_token_count": full,
            "unified_generic_total_partial_recovered_token_count": partial,
            "unified_generic_total_revised_token_count": revised,
            "unified_generic_total_output_token_count": full + partial,
            "combined_real_committed_token_count": 1244,
            "partial_prefix_recovery_enabled": True,
            "partial_prefix_recovered_proposal_ids": [2002],
            "partial_prefix_recovered_seq_ids": [8],
            "partial_prefix_recovered_depth_by_proposal_id": {"2002": 2},
            "partial_prefix_accepted_len_by_proposal_id": {"2002": 2},
            "partial_prefix_revised_token_count_by_proposal_id": {"2002": revised},
            "partial_prefix_committed_token_count_by_proposal_id": {"2002": partial},
            "generic_full_continuous_depth_partial_recovered_token_counts": {"2": partial},
            "generic_full_continuous_depth_revised_token_counts": {"2": revised},
            "unified_generic_depth_partial_recovered_token_counts": {"2": partial},
            "unified_generic_depth_revised_token_counts": {"2": revised},
            "generic_full_continuous_stop_reason_counts": {"max_depth_reached": 1},
            "generic_full_continuous_normal_lane_conflict_count": 0,
            "generic_full_continuous_target_draft_mismatch_count": 0,
            "generic_full_continuous_depth_gt_max_real_commit_count": 0,
            "generic_full_continuous_parity_ok": True,
            "unified_generic_normal_lane_conflict_count": 0,
            "unified_generic_target_draft_mismatch_count": 0,
            "unified_generic_parity_ok": True,
        }
    ]


def run_synthetic_tests() -> None:
    records = synthetic_records()
    payload = synthetic_payload(total_output_tokens=27)
    errors, summary = validate_records(records, payload)
    assert not errors, f"valid unified synthetic failed: {errors}\nsummary={summary}"
    assert summary["combined_real_committed_token_count"] == 27
    assert summary["unified_candidate_seq_count_by_depth"]["1"] == 1
    assert summary["unified_committed_token_count_by_depth"]["6"] == 4
    assert summary["unified_commit_share_by_depth"]["1"] == 1.0
    assert summary["num_steps"] == 1
    assert summary["steps_with_any_unified_candidate"] == 1
    assert summary["steps_with_any_unified_commit"] == 1

    bad_depth = [dict(records[0])]
    bad_depth[0]["generic_rolling_real_committed_proposal_ids_by_depth"] = {"2": [1002]}
    errors, _summary = validate_records(bad_depth, payload)
    assert errors, "synthetic missing depth1 should fail"

    bad_parent = [dict(records[0])]
    bad_parent[0]["generic_rolling_parent_by_proposal_id"] = {"1003": 999999}
    bad_parent[0]["generic_rolling_real_commit_parent_by_proposal_id"] = {"1003": 999999}
    errors, _summary = validate_records(bad_parent, payload)
    assert errors, "synthetic missing parent should fail"

    bad_output = [dict(records[0])]
    bad_output[0]["unified_generic_total_output_token_count"] = 26
    bad_output[0]["generic_full_continuous_total_output_token_count"] = 26
    errors, _summary = validate_records(bad_output, payload)
    assert errors, "synthetic output mismatch should fail"
    print("synthetic unified generic rolling runtime checks passed")


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "unified_generic_rolling_enabled",
        "configured_max_depth",
        "max_observed_depth",
        "max_real_committed_depth",
        "candidate_depths",
        "ready_depths",
        "committed_depths",
        "unified_candidate_seq_count_by_depth",
        "unified_ready_seq_count_by_depth",
        "unified_committed_seq_count_by_depth",
        "unified_candidate_token_count_by_depth",
        "unified_ready_token_count_by_depth",
        "unified_committed_token_count_by_depth",
        "unified_commit_share_by_depth",
        "unified_parent_not_full_accept_count_by_depth",
        "unified_stop_reason_counts_by_depth",
        "unified_no_eligible_parent_count_by_depth",
        "unified_no_eligible_ready_child_count_by_depth",
        "unified_sequence_finished_count_by_depth",
        "num_steps",
        "steps_with_any_unified_candidate",
        "steps_with_any_unified_commit",
        "avg_candidate_seqs_per_step",
        "avg_ready_seqs_per_step",
        "avg_committed_seqs_per_step",
        "avg_candidate_tokens_per_step",
        "avg_committed_tokens_per_step",
        "max_candidate_depth_per_step",
        "max_committed_depth_per_step",
        "total_full_commit_token_count",
        "total_partial_recovered_token_count",
        "total_revised_token_count",
        "total_output_token_count",
        "combined_real_committed_token_count",
        "normal_lane_conflict_count",
        "target_draft_mismatch_count",
        "parity_ok",
    ):
        print(f"{key}: {summary.get(key)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8x unified generic rolling runtime traces.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0
    result_payload = load_json(args.result) if args.result is not None else {}
    records = load_trace(args.trace)
    errors, summary = validate_records(records, result_payload)
    print_summary(summary)
    if errors:
        print("unified_generic_rolling_runtime=fail")
        for error in errors:
            print(f"- {error}")
        return 1
    print("unified_generic_rolling_runtime=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
