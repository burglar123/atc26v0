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


def as_int_list_map(value: Any) -> dict[int, list[int]]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, list[int]] = {}
    for key, items in value.items():
        try:
            proposal_id = int(key)
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        result[proposal_id] = as_int_list(items)
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


def merge_bool_map(records: list[dict[str, Any]], *fields: str) -> dict[int, bool]:
    merged: dict[int, bool] = {}
    for record in records:
        for field in fields:
            for key, value in as_bool_map(record.get(field)).items():
                merged.setdefault(key, value)
    return merged


def merge_str_map(records: list[dict[str, Any]], field: str) -> dict[int, str]:
    merged: dict[int, str] = {}
    for record in records:
        for key, value in as_str_map(record.get(field)).items():
            merged.setdefault(key, value)
    return merged


def merge_int_list_map(records: list[dict[str, Any]], field: str) -> dict[int, list[int]]:
    merged: dict[int, list[int]] = {}
    for record in records:
        for key, value in as_int_list_map(record.get(field)).items():
            merged.setdefault(key, value)
    return merged


def max_record_int(records: list[dict[str, Any]], *fields: str) -> int:
    value = 0
    for record in records:
        for field in fields:
            value = max(value, int_value(record.get(field), 0))
    return value


def any_record_bool(records: list[dict[str, Any]], field: str) -> bool:
    return any(bool(record.get(field, False)) for record in records)


def aggregate_shape(records: list[dict[str, Any]], field: str) -> list[int]:
    rows = 0
    width = 0
    saw_width = False
    for record in records:
        shape = as_int_list(record.get(field))
        if not shape:
            continue
        rows += int(shape[0])
        if len(shape) > 1:
            width = max(width, int(shape[1]))
            saw_width = True
    return [rows, width] if saw_width else ([rows] if rows else [])


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


def sum_record_int(records: list[dict[str, Any]], field: str) -> int:
    return sum(int_value(record.get(field), 0) for record in records)


def has_trace_field(records: list[dict[str, Any]], field: str) -> bool:
    return any(field in record for record in records)


def sum_depth_counts(records: list[dict[str, Any]], *fields: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for record in records:
        for field in fields:
            for depth, value in as_depth_int_map(record.get(field)).items():
                key = str(int(depth))
                merged[key] = int(merged.get(key, 0)) + int(value)
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def sum_counter_fields(records: list[dict[str, Any]], *fields: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for field in fields:
            value = record.get(field)
            if not isinstance(value, dict):
                continue
            for key, count in value.items():
                counter[str(key)] += int_value(count, 0)
    return dict(sorted(counter.items()))


def merge_int_lists(records: list[dict[str, Any]], field: str) -> list[int]:
    values: set[int] = set()
    for record in records:
        values.update(as_int_list(record.get(field)))
    return sorted(values)


def collect_dict_examples(records: list[dict[str, Any]], field: str, *, limit: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        value = record.get(field)
        if not isinstance(value, list):
            continue
        for item in value:
            if len(examples) >= limit:
                return examples
            if isinstance(item, dict):
                examples.append(dict(item))
    return examples


def add_depth_count_maps(*maps: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for value in maps:
        if not isinstance(value, dict):
            continue
        for depth, count in value.items():
            try:
                depth_key = str(int(depth))
            except Exception:
                continue
            combined[depth_key] = int(combined.get(depth_key, 0)) + int_value(count, 0)
    return dict(sorted(combined.items(), key=lambda item: int(item[0])))


def normal_buffer_consumer_role(event_type: str, reason: str) -> str:
    reason = str(reason or "")
    event_type = str(event_type or "")
    if reason == "target_normal_verify_consumed":
        return "target_normal_verify"
    if reason == "draft_apply_verify_consumed":
        return "draft_apply_verify"
    if reason in {"eager_owned", "unified_ready_child_owned", "owned_by_eager", "owned_by_unified_ready_child"}:
        return "owned_lane"
    if reason in {"sequence_finished", "request_finished", "cached_admission_completed"}:
        return "terminal"
    if event_type == "store":
        return "store"
    if reason in {"discard_inactive", "stale_base", "stale_discard"}:
        return "stale"
    return reason or event_type or "unknown"


def classify_normal_buffer_event(event_type: str, reason: str) -> str:
    reason = str(reason or "")
    event_type = str(event_type or "")
    if event_type == "store":
        return "store"
    if event_type == "receive":
        return "receive"
    if reason == "target_normal_verify_consumed":
        return "target_normal_verify_consume"
    if reason == "draft_apply_verify_consumed":
        return "draft_apply_verify_consume"
    if reason in {"eager_owned", "unified_ready_child_owned", "owned_by_eager", "owned_by_unified_ready_child"}:
        return "owned_lane_skip"
    if reason in {"sequence_finished", "request_finished", "cached_admission_completed"}:
        return "terminal_discard"
    if reason in {"discard_inactive", "stale_base", "stale_discard"}:
        return "stale_discard"
    if event_type in {"discard", "consume"}:
        return "unexpected_discard"
    return str(event_type or "unknown")


def normal_proposal_buffer_lifecycle_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    events_by_key: dict[tuple[int, str, int, int, int, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        events = record.get("normal_proposal_buffer_event_history")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type") or "")
            reason = str(event.get("reason") or "")
            role = str(event.get("consumer_role") or normal_buffer_consumer_role(event_type, reason))
            seq_id = int_value(event.get("seq_id"), -1)
            request_id = str(event.get("request_id") or "")
            proposal_id = int_value(event.get("proposal_id"), -1)
            plan_id = int_value(event.get("plan_id"), -1)
            step_id = int_value(event.get("step_id"), int_value(event.get("dual_step_id"), -1))
            key = (seq_id, request_id, proposal_id, plan_id, step_id, event_type, role)
            if key in events_by_key:
                duplicate_count += 1
                continue
            copied = dict(event)
            copied["consumer_role"] = role
            copied["classification"] = classify_normal_buffer_event(event_type, reason)
            events_by_key[key] = copied

    legal_consume = Counter()
    illegal_examples: list[dict[str, Any]] = []
    ordered_by_seq: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events_by_key.values():
        classification = str(event.get("classification") or "")
        reason = str(event.get("reason") or "")
        if classification in {
            "target_normal_verify_consume",
            "draft_apply_verify_consume",
            "owned_lane_skip",
            "terminal_discard",
            "stale_discard",
        }:
            legal_consume[reason or classification] += 1
        seq_key = (
            int_value(event.get("seq_id"), -1),
            str(event.get("request_id") or ""),
            int_value(event.get("proposal_id"), -1),
        )
        ordered_by_seq[seq_key].append(event)

    for seq_key, events in ordered_by_seq.items():
        sorted_events = sorted(
            events,
            key=lambda item: (
                int_value(item.get("step_id"), int_value(item.get("dual_step_id"), -1)),
                int_value(item.get("plan_id"), -1),
                str(item.get("event_type") or ""),
            ),
        )
        saw_store = False
        saw_legal_consume = False
        for event in sorted_events:
            classification = str(event.get("classification") or "")
            if classification == "store":
                saw_store = True
                continue
            if classification in {
                "target_normal_verify_consume",
                "draft_apply_verify_consume",
                "owned_lane_skip",
                "terminal_discard",
                "stale_discard",
            }:
                saw_legal_consume = True
                continue
            if classification == "unexpected_discard" and saw_store and not saw_legal_consume:
                if len(illegal_examples) < 8:
                    illegal_examples.append(
                        {
                            "seq_id": int(seq_key[0]),
                            "request_id": seq_key[1],
                            "proposal_id": int(seq_key[2]),
                            "event": dict(event),
                            "events": [dict(item) for item in sorted_events[:8]],
                        }
                    )

    for record in records:
        details = record.get("target_normal_verify_missing_buffer_details")
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict) or not bool(detail.get("was_buffer_discarded", False)):
                continue
            reason = str(detail.get("discard_reason") or "")
            classification = classify_normal_buffer_event("discard", reason)
            if classification in {
                "target_normal_verify_consume",
                "draft_apply_verify_consume",
                "owned_lane_skip",
                "terminal_discard",
                "stale_discard",
            } or bool(detail.get("was_sequence_finished", False)) or bool(detail.get("was_eager_owned", False)) or bool(
                detail.get("was_unified_ready_child_owned", False)
            ):
                legal_consume[reason or classification] += 1
                continue

    return {
        "normal_proposal_buffer_illegal_discard_count": len(illegal_examples),
        "normal_proposal_buffer_legal_consume_count_by_reason": dict(sorted(legal_consume.items())),
        "normal_proposal_buffer_event_order_violation_examples": illegal_examples,
        "normal_proposal_buffer_event_dedup_count": int(duplicate_count),
    }


def sum_depth_reason_counts(records: list[dict[str, Any]], *fields: str) -> dict[str, dict[str, int]]:
    by_depth: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for field in fields:
            value = record.get(field)
            if not isinstance(value, dict):
                continue
            for raw_depth, raw_counts in value.items():
                try:
                    depth = str(int(raw_depth))
                except Exception:
                    continue
                if not isinstance(raw_counts, dict):
                    continue
                for reason, count in raw_counts.items():
                    by_depth[depth][str(reason)] += int_value(count, 0)
    return {
        depth: dict(sorted(counter.items()))
        for depth, counter in sorted(by_depth.items(), key=lambda item: int(item[0]))
    }


def sum_nested_int_hist(records: list[dict[str, Any]], *fields: str) -> dict[str, dict[str, int]]:
    return sum_depth_reason_counts(records, *fields)


def nested_hist_total(hist_by_depth: dict[str, dict[str, int]]) -> int:
    return sum(int_value(count, 0) for hist in hist_by_depth.values() for count in hist.values())


def nested_hist_nonzero_accept_count(hist_by_depth: dict[str, dict[str, int]]) -> int:
    total = 0
    for hist in hist_by_depth.values():
        for bucket, count in hist.items():
            try:
                accepted_len = int(bucket)
            except Exception:
                continue
            if accepted_len > 0:
                total += int_value(count, 0)
    return total


def nested_hist_reject_count(hist_by_depth: dict[str, dict[str, int]]) -> int:
    return sum(int_value(hist.get("0"), 0) for hist in hist_by_depth.values())


def nested_hist_all_first_token_reject(hist_by_depth: dict[str, dict[str, int]]) -> bool:
    total = nested_hist_total(hist_by_depth)
    return total > 0 and nested_hist_nonzero_accept_count(hist_by_depth) == 0


def nested_hist_full_accept_count(hist_by_depth: dict[str, dict[str, int]], gamma: int) -> int:
    if gamma <= 0:
        return 0
    return sum(int_value(hist.get(str(int(gamma))), 0) for hist in hist_by_depth.values())


def nested_hist_partial_accept_count(hist_by_depth: dict[str, dict[str, int]], gamma: int) -> int:
    total = 0
    for hist in hist_by_depth.values():
        for bucket, count in hist.items():
            try:
                accepted_len = int(bucket)
            except Exception:
                continue
            if 0 < accepted_len < int(gamma):
                total += int_value(count, 0)
    return total


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


def derive_raw_outcome_fallback(
    records: list[dict[str, Any]],
    committed_by_depth: dict[int, list[int]],
) -> dict[str, Any]:
    result_by_id: dict[int, str] = {}
    accept_by_id = merge_int_map(records, "generic_rolling_real_committed_accept_len_by_proposal_id")
    token_by_id = merge_int_map(
        records,
        "generic_rolling_real_committed_token_count_by_proposal_id",
        "generic_rolling_token_count_by_proposal_id",
    )
    for record in records:
        value = record.get("generic_rolling_real_commit_verify_result_by_proposal_id")
        if not isinstance(value, dict):
            continue
        for raw_id, raw_result in value.items():
            try:
                proposal_id = int(raw_id)
            except Exception:
                continue
            result_by_id.setdefault(proposal_id, str(raw_result))

    verified: dict[str, int] = {}
    full: dict[str, int] = {}
    partial: dict[str, int] = {}
    reject: dict[str, int] = {}
    invalidated: dict[str, int] = {}
    accepted_hist: dict[str, dict[str, int]] = {}
    revised: dict[str, int] = {}
    eligible: dict[str, int] = {}
    for depth, proposal_ids in committed_by_depth.items():
        depth_key = str(int(depth))
        for proposal_id in set(proposal_ids):
            result = result_by_id.get(int(proposal_id), "full_accept")
            verified[depth_key] = int(verified.get(depth_key, 0)) + 1
            accepted_len = int(accept_by_id.get(int(proposal_id), token_by_id.get(int(proposal_id), 0)))
            accepted_hist.setdefault(depth_key, {})
            accepted_key = str(int(accepted_len))
            accepted_hist[depth_key][accepted_key] = int(accepted_hist[depth_key].get(accepted_key, 0)) + 1
            if result == "full_accept":
                full[depth_key] = int(full.get(depth_key, 0)) + 1
            elif result == "partial_accept":
                partial[depth_key] = int(partial.get(depth_key, 0)) + 1
                eligible[depth_key] = int(eligible.get(depth_key, 0)) + 1
            elif result in {"reject", "rejected"}:
                reject[depth_key] = int(reject.get(depth_key, 0)) + 1
            elif result == "invalidated":
                invalidated[depth_key] = int(invalidated.get(depth_key, 0)) + 1
    return {
        "unified_raw_verified_proposal_count_by_depth": dict(sorted(verified.items(), key=lambda item: int(item[0]))),
        "unified_raw_full_accept_proposal_count_by_depth": dict(sorted(full.items(), key=lambda item: int(item[0]))),
        "unified_raw_partial_accept_proposal_count_by_depth": dict(sorted(partial.items(), key=lambda item: int(item[0]))),
        "unified_raw_reject_proposal_count_by_depth": dict(sorted(reject.items(), key=lambda item: int(item[0]))),
        "unified_raw_invalidated_proposal_count_by_depth": dict(sorted(invalidated.items(), key=lambda item: int(item[0]))),
        "unified_raw_accepted_len_hist_by_depth": {
            depth: dict(sorted(hist.items(), key=lambda item: int(item[0])))
            for depth, hist in sorted(accepted_hist.items(), key=lambda item: int(item[0]))
        },
        "unified_raw_revised_token_count_by_depth": dict(sorted(revised.items(), key=lambda item: int(item[0]))),
        "unified_raw_partial_recovery_eligible_count_by_depth": dict(sorted(eligible.items(), key=lambda item: int(item[0]))),
        "unified_raw_partial_recovery_ineligible_reason_counts_by_depth": {},
    }


def raw_and_budget_summary(
    records: list[dict[str, Any]],
    candidate_by_depth: dict[int, list[int]],
    committed_by_depth: dict[int, list[int]],
) -> dict[str, Any]:
    fallback = derive_raw_outcome_fallback(records, committed_by_depth)
    raw_fields = [
        "unified_raw_verified_proposal_count_by_depth",
        "unified_raw_full_accept_proposal_count_by_depth",
        "unified_raw_partial_accept_proposal_count_by_depth",
        "unified_raw_reject_proposal_count_by_depth",
        "unified_raw_invalidated_proposal_count_by_depth",
        "unified_raw_revised_token_count_by_depth",
        "unified_raw_partial_recovery_eligible_count_by_depth",
        "unified_raw_candidate_proposal_count_by_depth",
        "unified_raw_committed_proposal_count_by_depth",
        "unified_raw_full_commit_proposal_count_by_depth",
        "unified_raw_partial_recovery_applied_proposal_count_by_depth",
        "unified_raw_reject_revised_correction_applied_proposal_count_by_depth",
        "unified_raw_no_mutation_reject_proposal_count_by_depth",
    ]
    summary: dict[str, Any] = {}
    for field in raw_fields:
        if field == "unified_raw_candidate_proposal_count_by_depth":
            traced_candidate = sum_depth_counts(records, field) if has_trace_field(records, field) else {}
            derived_candidate = count_depth_lists(candidate_by_depth)
            summary[field] = traced_candidate or derived_candidate
            summary["unified_raw_candidate_proposal_count_available"] = bool(
                summary[field] or traced_candidate or derived_candidate
            )
        elif has_trace_field(records, field):
            summary[field] = sum_depth_counts(records, field)
        elif field == "unified_raw_committed_proposal_count_by_depth":
            summary[field] = count_depth_lists(committed_by_depth)
        else:
            summary[field] = fallback.get(field, {})
    summary.setdefault("unified_raw_candidate_proposal_count_available", False)
    summary["unified_raw_accepted_len_hist_by_depth"] = (
        sum_nested_int_hist(records, "unified_raw_accepted_len_hist_by_depth")
        if has_trace_field(records, "unified_raw_accepted_len_hist_by_depth")
        else fallback["unified_raw_accepted_len_hist_by_depth"]
    )
    summary["unified_raw_partial_recovery_ineligible_reason_counts_by_depth"] = (
        sum_depth_reason_counts(records, "unified_raw_partial_recovery_ineligible_reason_counts_by_depth")
        if has_trace_field(records, "unified_raw_partial_recovery_ineligible_reason_counts_by_depth")
        else fallback["unified_raw_partial_recovery_ineligible_reason_counts_by_depth"]
    )
    split_committed: dict[str, int] = {}
    for field in (
        "unified_raw_full_commit_proposal_count_by_depth",
        "unified_raw_partial_recovery_applied_proposal_count_by_depth",
        "unified_raw_reject_revised_correction_applied_proposal_count_by_depth",
    ):
        for depth, count in summary.get(field, {}).items():
            split_committed[str(depth)] = int(split_committed.get(str(depth), 0)) + int_value(count, 0)
    if split_committed:
        summary["unified_raw_committed_proposal_count_by_depth"] = dict(
            sorted(split_committed.items(), key=lambda item: int(item[0]))
        )
    verified = summary["unified_raw_verified_proposal_count_by_depth"]
    committed = summary["unified_raw_committed_proposal_count_by_depth"]
    summary["unified_raw_verified_to_committed_ratio_by_depth"] = {
        depth: safe_div(float(committed.get(depth, 0)), float(verified.get(depth, 0)))
        for depth in sorted(set(verified) | set(committed), key=int)
    }
    candidate = summary["unified_raw_candidate_proposal_count_by_depth"]
    invalidated = summary["unified_raw_invalidated_proposal_count_by_depth"]
    wasted = {
        depth: max(0, int_value(candidate.get(depth), 0) - int_value(committed.get(depth), 0))
        for depth in sorted(set(candidate) | set(committed), key=int)
    }
    summary["unified_candidate_waste_ratio_by_depth"] = ratio_by_depth(wasted, candidate)
    summary["unified_invalidated_candidate_ratio_by_depth"] = ratio_by_depth(invalidated, candidate)
    summary["unified_verified_candidate_ratio_by_depth"] = ratio_by_depth(verified, candidate)
    summary["unified_committed_candidate_ratio_by_depth"] = ratio_by_depth(committed, candidate)
    sources = sorted(
        {
            str(record.get("unified_raw_verification_source"))
            for record in records
            if record.get("unified_raw_verification_source") is not None
        }
    )
    summary["unified_raw_target_verification_available"] = any(
        bool(record.get("unified_raw_target_verification_available", False))
        for record in records
    )
    summary["unified_raw_verification_source"] = (
        ",".join(sources)
        if sources
        else ("trace_field" if has_trace_field(records, "unified_raw_verified_proposal_count_by_depth") else "legacy_committed_result_fallback")
    )
    num_steps = max(
        len({record_step_key(record, index) for index, record in enumerate(records) if (
            bool(record.get("unified_generic_rolling_enabled", False))
            or bool(record.get("enable_unified_generic_rolling_runtime", False))
        )}),
        0,
    )
    summary.update(
        {
            "unified_candidate_budget_tokens_per_step": max_record_int(records, "unified_candidate_budget_tokens_per_step"),
            "unified_candidate_budget_used_tokens_per_step": max_record_int(records, "unified_candidate_budget_used_tokens_per_step"),
            "unified_candidate_budget_saturated_step_count": sum_record_int(records, "unified_candidate_budget_saturated_step_count"),
            "unified_commit_budget_tokens_per_step": max_record_int(records, "unified_commit_budget_tokens_per_step"),
            "unified_commit_budget_used_tokens_per_step": max_record_int(records, "unified_commit_budget_used_tokens_per_step"),
            "unified_commit_budget_saturated_step_count": sum_record_int(records, "unified_commit_budget_saturated_step_count"),
            "unified_ready_but_not_committed_token_count": sum_record_int(records, "unified_ready_but_not_committed_token_count"),
            "unified_ready_but_not_committed_reason_counts": sum_counter_fields(records, "unified_ready_but_not_committed_reason_counts"),
            "unified_ready_but_not_committed_by_depth": sum_depth_counts(records, "unified_ready_but_not_committed_by_depth"),
            "unified_commit_limited_by_token_budget_count": sum_record_int(records, "unified_commit_limited_by_token_budget_count"),
            "unified_commit_limited_by_seq_budget_count": sum_record_int(records, "unified_commit_limited_by_seq_budget_count"),
            "unified_commit_limited_by_no_ready_parent_count": sum_record_int(records, "unified_commit_limited_by_no_ready_parent_count"),
            "unified_commit_limited_by_parent_not_full_accept_count": sum_record_int(records, "unified_commit_limited_by_parent_not_full_accept_count"),
            "unified_cascade_discard_count": len(merge_int_lists(records, "unified_cascade_discard_proposal_ids")),
            "unified_cascade_discard_proposal_ids": merge_int_lists(records, "unified_cascade_discard_proposal_ids"),
            "unified_cascade_discard_parent_proposal_ids": merge_int_lists(
                records,
                "unified_cascade_discard_parent_proposal_ids",
            ),
            "unified_cascade_discard_reason_counts": sum_counter_fields(
                records,
                "unified_cascade_discard_reason_counts",
            ),
            "unified_invalidated_due_to_parent_not_full_accept_count_by_depth": sum_depth_counts(
                records,
                "unified_invalidated_due_to_parent_not_full_accept_count_by_depth",
            ),
            "unified_invalidated_due_to_parent_not_full_accept_proposal_ids_by_depth": merge_depth_lists(
                records,
                "unified_invalidated_due_to_parent_not_full_accept_proposal_ids_by_depth",
            ),
            "unified_no_candidate_step_count": (
                sum_record_int(records, "unified_no_candidate_step_count")
                if has_trace_field(records, "unified_no_candidate_step_count")
                else max(0, num_steps - int_value(utilization_summary(records).get("steps_with_any_unified_candidate"), 0))
            ),
            "unified_no_commit_step_count": (
                sum_record_int(records, "unified_no_commit_step_count")
                if has_trace_field(records, "unified_no_commit_step_count")
                else max(0, num_steps - int_value(utilization_summary(records).get("steps_with_any_unified_commit"), 0))
            ),
            "unified_candidate_step_reason_counts": sum_counter_fields(records, "unified_candidate_step_reason_counts"),
            "unified_no_commit_step_reason_counts": sum_counter_fields(records, "unified_no_commit_step_reason_counts"),
            "unified_active_seq_count_by_step": sum_counter_fields(records, "unified_active_seq_count_by_step"),
            "unified_ready_parent_count_by_step": sum_counter_fields(records, "unified_ready_parent_count_by_step"),
            "unified_committed_seq_count_by_step": sum_counter_fields(records, "unified_committed_seq_count_by_step"),
        }
    )
    if summary["unified_no_candidate_step_count"] and not summary["unified_candidate_step_reason_counts"]:
        summary["unified_candidate_step_reason_counts"] = {"unknown_legacy_trace": summary["unified_no_candidate_step_count"]}
    if summary["unified_no_commit_step_count"] and not summary["unified_no_commit_step_reason_counts"]:
        summary["unified_no_commit_step_reason_counts"] = {"unknown_legacy_trace": summary["unified_no_commit_step_count"]}
    return summary


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


def ratio_by_depth(
    numerator_by_depth: dict[str, int],
    denominator_by_depth: dict[str, int],
) -> dict[str, float]:
    return {
        depth: safe_div(float(numerator_by_depth.get(depth, 0)), float(denominator_by_depth.get(depth, 0)))
        for depth in sorted(set(numerator_by_depth) | set(denominator_by_depth), key=int)
    }


def single_child_ahead_summary(
    records: list[dict[str, Any]],
    result_args: dict[str, Any],
    candidate_by_depth: dict[int, list[int]],
    parent_by_id: dict[int, int],
    depth_by_id: dict[int, int],
) -> dict[str, Any]:
    configured_limit = max(
        max_record_int(records, "unified_max_unverified_depth_ahead"),
        int_value(result_args.get("unified_generic_max_unverified_depth_ahead"), 0),
    )
    single_child_enabled = any_record_bool(records, "unified_single_child_ahead_enabled") or configured_limit == 1
    root_by_id = merge_int_map(
        records,
        "unified_candidate_root_proposal_id_by_proposal_id",
        "generic_rolling_root_by_proposal_id",
        "generic_rolling_real_commit_root_by_proposal_id",
    )
    created_step_by_id = merge_int_map(
        records,
        "unified_candidate_created_step_by_proposal_id",
        "generic_rolling_source_dual_step_id_by_proposal_id",
    )
    parent_by_id = {
        **parent_by_id,
        **merge_int_map(records, "unified_candidate_parent_proposal_id_by_proposal_id"),
    }
    depth_by_id = {
        **depth_by_id,
        **merge_int_map(records, "unified_candidate_depth_by_proposal_id"),
    }
    committed_by_depth = merge_depth_lists(records, "generic_rolling_real_committed_proposal_ids_by_depth")
    committed_depth_by_id: dict[int, int] = {}
    for committed_depth, committed_ids in committed_by_depth.items():
        for committed_id in committed_ids:
            committed_depth_by_id.setdefault(int(committed_id), int(committed_depth))
    committed_accept_by_id = merge_int_map(records, "generic_rolling_real_committed_accept_len_by_proposal_id")
    committed_token_by_id = merge_int_map(
        records,
        "generic_rolling_real_committed_token_count_by_proposal_id",
        "generic_rolling_token_count_by_proposal_id",
    )
    committed_action_by_id = merge_str_map(records, "generic_rolling_real_commit_action_by_proposal_id")
    committed_result_by_id = merge_str_map(records, "generic_rolling_real_commit_verify_result_by_proposal_id")
    gamma = max_record_int(records, "normal_gamma")

    registry_depth_by_id = merge_int_map(records, "unified_proposal_registry_depth_by_proposal_id")
    registry_full_by_id = merge_bool_map(records, "unified_proposal_registry_full_accept_by_proposal_id")
    registry_partial_by_id = merge_bool_map(records, "unified_proposal_registry_partial_accept_by_proposal_id")
    registry_reject_by_id = merge_bool_map(records, "unified_proposal_registry_reject_by_proposal_id")
    registry_invalidated_by_id = merge_bool_map(records, "unified_proposal_registry_invalidated_by_proposal_id")
    registry_target_inflight_by_id = merge_bool_map(
        records,
        "unified_proposal_registry_target_verify_inflight_by_proposal_id",
    )
    registry_applied_full_by_id = merge_bool_map(
        records,
        "unified_proposal_registry_applied_full_commit_by_proposal_id",
    )
    registry_applied_partial_by_id = merge_bool_map(
        records,
        "unified_proposal_registry_applied_partial_recovery_by_proposal_id",
    )
    registry_no_mutation_reject_by_id = merge_bool_map(
        records,
        "unified_proposal_registry_no_mutation_reject_by_proposal_id",
    )
    registered_trace_by_depth = sum_depth_counts(records, "unified_full_accept_parent_registered_count_by_depth")
    registry_size_trace_by_depth = merge_depth_counts(records, "unified_full_accept_parent_registry_size_by_depth")
    selected_trace_by_depth = sum_depth_counts(records, "unified_full_accept_parent_selected_for_child_count_by_depth")
    parent_not_selected_reason_counts = sum_depth_reason_counts(
        records,
        "unified_full_accept_parent_not_selected_reason_counts_by_depth",
    )
    full_accept_without_child_reason_counts = sum_depth_reason_counts(
        records,
        "unified_full_accept_without_child_reason_counts_by_depth",
    )
    depth2_block_reason_counts: Counter[str] = Counter()
    for record in records:
        for reason, count in (record.get("unified_depth2_generation_block_reason_counts") or {}).items():
            depth2_block_reason_counts[str(reason)] += int_value(count, 0)
    frontier_mismatch_by_delta: Counter[str] = Counter()
    for record in records:
        for delta, count in (
            record.get("unified_full_accept_parent_frontier_mismatch_count_by_delta") or {}
        ).items():
            frontier_mismatch_by_delta[str(delta)] += int_value(count, 0)
    expected_frontier_source_counts: Counter[str] = Counter()
    for record in records:
        for source, count in (
            record.get("unified_full_accept_parent_expected_frontier_len_source_counts") or {}
        ).items():
            expected_frontier_source_counts[str(source)] += int_value(count, 0)
    child_full_accept_examples: list[dict[str, Any]] = []
    for record in records:
        for example in record.get("unified_child_generated_from_full_accept_parent_examples") or []:
            if len(child_full_accept_examples) >= 8:
                break
            if isinstance(example, dict):
                child_full_accept_examples.append(example)
    frontier_mismatch_examples: list[dict[str, Any]] = []
    for record in records:
        for example in record.get("unified_full_accept_parent_frontier_mismatch_examples") or []:
            if len(frontier_mismatch_examples) >= 8:
                break
            if isinstance(example, dict):
                frontier_mismatch_examples.append(example)

    registered_from_registry_by_depth: Counter[str] = Counter()
    for proposal_id, depth in registry_depth_by_id.items():
        if (
            int(depth) >= 1
            and registry_full_by_id.get(proposal_id, False)
            and registry_applied_full_by_id.get(proposal_id, False)
            and not registry_invalidated_by_id.get(proposal_id, False)
        ):
            registered_from_registry_by_depth[str(int(depth))] += 1

    child_parent_lookup_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_lookup_found_by_proposal_id",
    )
    child_parent_allowed_by_id = merge_bool_map(records, "unified_child_generation_allowed_by_proposal_id")
    child_parent_full_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_full_accept_by_proposal_id",
    )
    child_parent_partial_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_partial_accept_by_proposal_id",
    )
    child_parent_reject_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_reject_by_proposal_id",
    )
    child_parent_invalidated_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_invalidated_by_proposal_id",
    )
    child_parent_applied_full_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_applied_full_commit_by_proposal_id",
    )
    child_parent_applied_partial_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_applied_partial_recovery_by_proposal_id",
    )
    child_parent_no_mutation_reject_by_id = merge_bool_map(
        records,
        "unified_child_generation_parent_no_mutation_reject_by_proposal_id",
    )
    child_parent_state_by_id = merge_str_map(records, "unified_child_generation_parent_state_by_proposal_id")
    child_generation_block_by_id = merge_str_map(records, "unified_child_generation_block_reason_by_proposal_id")
    child_generation_mode_by_id = merge_str_map(records, "unified_child_generation_mode_by_proposal_id")
    child_generated_before_parent_result_by_id = merge_bool_map(
        records,
        "unified_child_generated_before_parent_result_by_proposal_id",
        "unified_proposal_registry_child_generated_before_parent_result_by_proposal_id",
    )
    child_pending_parent_result_by_id = merge_bool_map(
        records,
        "unified_child_pending_parent_result_by_proposal_id",
        "unified_proposal_registry_child_pending_parent_result_by_proposal_id",
    )
    child_promoted_after_parent_full_by_id = merge_bool_map(
        records,
        "unified_child_promoted_after_parent_full_accept_by_proposal_id",
        "unified_proposal_registry_child_promoted_after_parent_full_accept_by_proposal_id",
    )
    child_invalidated_after_parent_non_full_by_id = merge_bool_map(
        records,
        "unified_child_invalidated_after_parent_non_full_by_proposal_id",
        "unified_proposal_registry_child_invalidated_after_parent_non_full_by_proposal_id",
    )
    inflight_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_generated_from_inflight_parent_count_by_depth",
    )
    pending_trace_by_depth = sum_depth_counts(records, "unified_child_pending_parent_result_count_by_depth")
    promoted_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_promoted_after_parent_full_accept_count_by_depth",
    )
    promoted_to_ready_trace_by_depth = sum_depth_counts(records, "unified_child_promoted_to_ready_count_by_depth")
    ready_for_target_trace_by_depth = sum_depth_counts(records, "unified_child_ready_for_target_verify_count_by_depth")
    ready_not_scheduled_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_ready_but_not_scheduled_count_by_depth",
    )
    ready_not_scheduled_reason_counts = sum_depth_reason_counts(
        records,
        "unified_child_ready_not_scheduled_reason_counts_by_depth",
    )
    ready_lane_excluded_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_ready_seq_normal_lane_excluded_count_by_depth",
    )
    ready_lane_conflict_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_ready_seq_normal_lane_conflict_count_by_depth",
    )
    stale_base_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_stale_base_count_by_depth",
    )
    ready_lane_conflict_examples: list[dict[str, Any]] = []
    stale_base_examples: list[dict[str, Any]] = []
    for record in records:
        for example in record.get("unified_child_ready_seq_normal_lane_conflict_examples") or []:
            if len(ready_lane_conflict_examples) >= 8:
                break
            if isinstance(example, dict):
                ready_lane_conflict_examples.append(example)
        for example in record.get("unified_child_stale_base_examples") or []:
            if len(stale_base_examples) >= 8:
                break
            if isinstance(example, dict):
                stale_base_examples.append(example)
    scheduled_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_scheduled_for_target_verify_count_by_depth",
    )
    target_verify_inflight_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_target_verify_inflight_count_by_depth",
    )
    scheduled_state_by_depth = sum_depth_reason_counts(
        records,
        "unified_child_scheduled_state_by_depth",
    )
    duplicate_schedule_skip_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_duplicate_schedule_skip_count_by_depth",
    )
    schedule_state_error_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_schedule_state_error_count_by_depth",
    )
    schedule_state_error_examples: list[dict[str, Any]] = []
    for record in records:
        for example in record.get("unified_child_schedule_state_error_examples") or []:
            if len(schedule_state_error_examples) >= 8:
                break
            if isinstance(example, dict):
                schedule_state_error_examples.append(example)
    ready_owner_seq_ids = merge_int_lists(records, "unified_ready_child_lane_owner_seq_ids")
    ready_owner_request_ids: dict[str, str] = {}
    ready_exclusion_mismatch_examples: list[dict[str, Any]] = []
    for record in records:
        request_ids = record.get("unified_ready_child_lane_owner_request_ids")
        if isinstance(request_ids, dict):
            for seq_id, request_id in request_ids.items():
                try:
                    ready_owner_request_ids[str(int(seq_id))] = str(request_id)
                except Exception:
                    continue
        for example in record.get("unified_ready_child_normal_verify_exclusion_mismatch_examples") or []:
            if len(ready_exclusion_mismatch_examples) >= 8:
                break
            if isinstance(example, dict):
                ready_exclusion_mismatch_examples.append(example)
    ready_excluded_from_draft = merge_int_lists(
        records,
        "unified_ready_child_excluded_from_normal_draft_seq_ids",
    )
    ready_excluded_from_target_normal = merge_int_lists(
        records,
        "unified_ready_child_excluded_from_target_normal_verify_seq_ids",
    )
    ready_excluded_from_target_normal_before_filter = merge_int_lists(
        records,
        "unified_ready_child_excluded_from_target_normal_verify_before_filter_seq_ids",
    )
    ready_excluded_from_target_normal_after_filter = merge_int_lists(
        records,
        "unified_ready_child_excluded_from_target_normal_verify_after_filter_seq_ids",
    )
    ready_remaining_in_target_normal = merge_int_lists(
        records,
        "unified_ready_child_remaining_in_target_normal_verify_seq_ids",
    )
    ready_missing_normal_allowed = merge_int_lists(
        records,
        "unified_ready_child_missing_normal_proposal_allowed_seq_ids",
    )
    missing_buffered_allowed_unified = merge_int_lists(
        records,
        "missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids",
    )
    ready_exclusion_mismatch_count = sum_record_int(
        records,
        "unified_ready_child_normal_verify_exclusion_mismatch_count",
    )
    target_verified_after_promotion_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_target_verified_after_promotion_count_by_depth",
    )
    invalidated_after_non_full_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_invalidated_after_parent_non_full_count_by_depth",
    )
    verified_after_parent_full_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_verified_after_parent_full_accept_count_by_depth",
    )
    verified_before_parent_result_trace_by_depth = sum_depth_counts(
        records,
        "unified_child_verified_before_parent_result_count_by_depth",
    )

    candidate_depth_by_id: dict[int, int] = {}
    candidate_ids: set[int] = set()
    for depth, proposal_ids in candidate_by_depth.items():
        for proposal_id in proposal_ids:
            candidate_ids.add(int(proposal_id))
            candidate_depth_by_id.setdefault(int(proposal_id), int(depth))
            depth_by_id.setdefault(int(proposal_id), int(depth))

    for index, record in enumerate(records):
        step_id = int_value(record.get("step_id"), int_value(record.get("eager_commit_step_id"), index))
        for depth, proposal_ids in as_depth_int_lists(record.get("generic_rolling_candidate_proposal_ids_by_depth")).items():
            for proposal_id in proposal_ids:
                candidate_ids.add(int(proposal_id))
                candidate_depth_by_id.setdefault(int(proposal_id), int(depth))
                depth_by_id.setdefault(int(proposal_id), int(depth))
                created_step_by_id.setdefault(int(proposal_id), int(step_id))

    def root_for(proposal_id: int) -> int:
        if proposal_id in root_by_id:
            return int(root_by_id[proposal_id])
        seen: set[int] = set()
        cursor = int(proposal_id)
        while cursor in parent_by_id and cursor not in seen:
            seen.add(cursor)
            parent = int(parent_by_id[cursor])
            if parent < 0:
                break
            cursor = parent
        return int(cursor)

    by_chain_step: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for proposal_id in sorted(candidate_ids):
        depth = int(depth_by_id.get(proposal_id, candidate_depth_by_id.get(proposal_id, 0)))
        if depth <= 0:
            continue
        step_id = int(created_step_by_id.get(proposal_id, -1))
        root_id = root_for(int(proposal_id))
        by_chain_step[(root_id, step_id)].append((int(depth), int(proposal_id)))

    ahead_by_chain: dict[int, int] = {}
    violation_examples: list[dict[str, Any]] = []
    grandchild_examples: list[dict[str, Any]] = []
    grandchild_count = 0
    for (root_id, step_id), depth_ids in sorted(by_chain_step.items()):
        depths = [depth for depth, _proposal_id in depth_ids]
        if not depths:
            continue
        ahead = max(1, max(depths) - min(depths))
        ahead_by_chain[root_id] = max(int(ahead_by_chain.get(root_id, 0)), int(ahead))
        proposal_ids = [proposal_id for _depth, proposal_id in sorted(depth_ids)]
        if configured_limit > 0 and ahead > configured_limit:
            violation_examples.append(
                {
                    "root_id": int(root_id),
                    "source_step_id": int(step_id),
                    "depths": sorted(set(int(depth) for depth in depths)),
                    "proposal_ids": proposal_ids[:8],
                }
            )
        min_depth = min(depths)
        for depth, proposal_id in sorted(depth_ids):
            if int(depth) - int(min_depth) >= 2:
                grandchild_count += 1
                if len(grandchild_examples) < 8:
                    grandchild_examples.append(
                        {
                            "root_id": int(root_id),
                            "source_step_id": int(step_id),
                            "depth": int(depth),
                            "proposal_id": int(proposal_id),
                            "earliest_depth_in_step": int(min_depth),
                        }
                    )

    before_parent_verified_by_depth: Counter[str] = Counter()
    after_parent_verified_by_depth: Counter[str] = Counter()
    full_parent_by_depth: Counter[str] = Counter()
    inflight_parent_by_depth: Counter[str] = Counter()
    non_full_parent_by_depth: Counter[str] = Counter()
    unverified_parent_by_depth: Counter[str] = Counter()
    partial_parent_by_depth: Counter[str] = Counter()
    reject_parent_by_depth: Counter[str] = Counter()
    invalidated_parent_by_depth: Counter[str] = Counter()
    missing_parent_by_depth: Counter[str] = Counter()
    guard_violation_ids: set[int] = set()
    guard_examples: list[dict[str, Any]] = []
    for proposal_id in sorted(candidate_ids):
        parent_id = parent_by_id.get(int(proposal_id))
        if parent_id is None or int(parent_id) < 0:
            continue
        depth = int(depth_by_id.get(int(proposal_id), 0))
        if depth <= 0:
            continue
        child_step = int(created_step_by_id.get(int(proposal_id), -1))
        parent_step = int(created_step_by_id.get(int(parent_id), -1))
        if parent_step >= child_step >= 0:
            before_parent_verified_by_depth[str(depth)] += 1
        elif parent_step >= 0 and child_step > parent_step:
            after_parent_verified_by_depth[str(depth)] += 1

        if depth <= 1:
            continue
        parent_id = int(parent_id)
        parent_committed = parent_id in committed_depth_by_id
        parent_token_len = int(committed_token_by_id.get(parent_id, gamma or 0))
        parent_accept_len = int(committed_accept_by_id.get(parent_id, parent_token_len))
        inferred_full = bool(
            parent_committed
            and int(committed_depth_by_id.get(parent_id, depth - 1)) == depth - 1
            and parent_token_len > 0
            and parent_accept_len == parent_token_len
            and str(committed_result_by_id.get(parent_id, "full_accept")) == "full_accept"
            and str(committed_action_by_id.get(parent_id, "append_full_accept_real_commit"))
            == "append_full_accept_real_commit"
        )
        lookup_known = (
            child_parent_lookup_by_id.get(proposal_id)
            if proposal_id in child_parent_lookup_by_id
            else bool(parent_id in registry_depth_by_id or parent_committed)
        )
        parent_full = (
            child_parent_full_by_id.get(proposal_id)
            if proposal_id in child_parent_full_by_id
            else bool(registry_full_by_id.get(parent_id, inferred_full))
        )
        parent_applied_full = (
            child_parent_applied_full_by_id.get(proposal_id)
            if proposal_id in child_parent_applied_full_by_id
            else bool(registry_applied_full_by_id.get(parent_id, inferred_full))
        )
        parent_partial = (
            child_parent_partial_by_id.get(proposal_id)
            if proposal_id in child_parent_partial_by_id
            else bool(registry_partial_by_id.get(parent_id, False))
        )
        parent_reject = (
            child_parent_reject_by_id.get(proposal_id)
            if proposal_id in child_parent_reject_by_id
            else bool(registry_reject_by_id.get(parent_id, False))
        )
        parent_invalidated = (
            child_parent_invalidated_by_id.get(proposal_id)
            if proposal_id in child_parent_invalidated_by_id
            else bool(registry_invalidated_by_id.get(parent_id, False))
        )
        parent_applied_partial = (
            child_parent_applied_partial_by_id.get(proposal_id)
            if proposal_id in child_parent_applied_partial_by_id
            else bool(registry_applied_partial_by_id.get(parent_id, False))
        )
        parent_no_mutation_reject = (
            child_parent_no_mutation_reject_by_id.get(proposal_id)
            if proposal_id in child_parent_no_mutation_reject_by_id
            else bool(registry_no_mutation_reject_by_id.get(parent_id, False))
        )
        parent_target_inflight = bool(registry_target_inflight_by_id.get(parent_id, False))
        inflight_generated = bool(
            child_generated_before_parent_result_by_id.get(proposal_id, False)
            or child_pending_parent_result_by_id.get(proposal_id, False)
            or child_generation_mode_by_id.get(proposal_id, "") == "inflight_parent_speculative"
            or child_parent_state_by_id.get(proposal_id, "") == "target_verify_inflight"
            or parent_target_inflight
        )
        default_allowed = bool(lookup_known and parent_full and parent_applied_full and not parent_invalidated)
        if inflight_generated:
            if parent_invalidated or parent_partial or parent_applied_partial or parent_reject or parent_no_mutation_reject:
                allowed = bool(child_invalidated_after_parent_non_full_by_id.get(proposal_id, False))
            elif parent_full and parent_applied_full:
                allowed = bool(
                    child_promoted_after_parent_full_by_id.get(proposal_id, False)
                    or child_pending_parent_result_by_id.get(proposal_id, False)
                    or default_allowed
                )
            else:
                allowed = True
        else:
            allowed = (
                child_parent_allowed_by_id.get(proposal_id)
                if proposal_id in child_parent_allowed_by_id
                else default_allowed
            )
        depth_key = str(depth)
        if inflight_generated:
            inflight_parent_by_depth[depth_key] += 1
        elif lookup_known and parent_full and parent_applied_full and not parent_invalidated:
            full_parent_by_depth[depth_key] += 1
        if inflight_generated:
            pass
        elif not lookup_known:
            missing_parent_by_depth[depth_key] += 1
            unverified_parent_by_depth[depth_key] += 1
        elif parent_invalidated:
            invalidated_parent_by_depth[depth_key] += 1
        elif parent_partial or parent_applied_partial:
            partial_parent_by_depth[depth_key] += 1
        elif parent_reject or parent_no_mutation_reject:
            reject_parent_by_depth[depth_key] += 1
        elif not (parent_full and parent_applied_full):
            non_full_parent_by_depth[depth_key] += 1
        if not allowed:
            guard_violation_ids.add(int(proposal_id))
            if len(guard_examples) < 8:
                guard_examples.append(
                    {
                        "child_proposal_id": int(proposal_id),
                        "child_depth": int(depth),
                        "parent_proposal_id": int(parent_id),
                        "parent_lookup_found": bool(lookup_known),
                        "parent_state": str(child_parent_state_by_id.get(proposal_id, "")),
                        "child_generation_mode": str(child_generation_mode_by_id.get(proposal_id, "")),
                        "block_reason": str(child_generation_block_by_id.get(proposal_id, "")),
                        "parent_full_accept": bool(parent_full),
                        "parent_applied_full_commit": bool(parent_applied_full),
                        "child_pending_parent_result": bool(
                            child_pending_parent_result_by_id.get(proposal_id, False)
                        ),
                        "child_promoted_after_parent_full_accept": bool(
                            child_promoted_after_parent_full_by_id.get(proposal_id, False)
                        ),
                        "child_invalidated_after_parent_non_full": bool(
                            child_invalidated_after_parent_non_full_by_id.get(proposal_id, False)
                        ),
                    }
                )

    trace_violation_count = sum_record_int(records, "unified_single_child_ahead_violation_count")
    trace_grandchild_count = sum_record_int(records, "unified_generated_grandchild_before_parent_verified_count")
    derived_violation_count = len(violation_examples) if configured_limit > 0 else 0
    parent_guard_violation_count = len(guard_violation_ids)
    selected_from_children_by_parent_depth: Counter[str] = Counter()
    for depth_key, count in full_parent_by_depth.items():
        parent_depth = int(depth_key) - 1
        if parent_depth >= 1:
            selected_from_children_by_parent_depth[str(parent_depth)] += int(count)
    registered_by_depth = {
        str(depth): int(count)
        for depth, count in (registered_trace_by_depth or registered_from_registry_by_depth).items()
    }
    registry_size_by_depth = {
        str(depth): int(count)
        for depth, count in (registry_size_trace_by_depth or registered_from_registry_by_depth).items()
    }
    selected_parent_by_depth = {
        str(depth): int(count)
        for depth, count in (selected_trace_by_depth or selected_from_children_by_parent_depth).items()
    }
    return {
        "unified_single_child_ahead_enabled": bool(single_child_enabled),
        "unified_max_unverified_depth_ahead": int(configured_limit),
        "unified_unverified_depth_ahead_max_observed": max(
            max((int(value) for value in ahead_by_chain.values()), default=0),
            max_record_int(records, "unified_unverified_depth_ahead_max_observed"),
        ),
        "unified_unverified_depth_ahead_by_chain": {
            str(root_id): int(value) for root_id, value in sorted(ahead_by_chain.items())
        },
        "unified_single_child_ahead_violation_count": max(int(trace_violation_count), int(derived_violation_count)),
        "unified_single_child_ahead_violation_examples": violation_examples[:8],
        "unified_generated_grandchild_before_parent_verified_count": max(
            int(trace_grandchild_count),
            int(grandchild_count if configured_limit > 0 else 0),
        ),
        "unified_generated_grandchild_before_parent_verified_examples": grandchild_examples[:8],
        "unified_candidate_depth_created_before_parent_verified_count_by_depth": dict(
            sorted(before_parent_verified_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_candidate_depth_created_after_parent_verified_count_by_depth": dict(
            sorted(after_parent_verified_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_generated_from_full_accept_parent_count_by_depth": dict(
            sorted(full_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_generated_from_inflight_parent_count_by_depth": dict(
            sorted(
                (inflight_trace_by_depth or inflight_parent_by_depth).items(),
                key=lambda item: int(item[0]),
            )
        ),
        "unified_child_pending_parent_result_count_by_depth": dict(
            sorted(pending_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_promoted_after_parent_full_accept_count_by_depth": dict(
            sorted(promoted_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_promoted_to_ready_count_by_depth": dict(
            sorted(promoted_to_ready_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_ready_for_target_verify_count_by_depth": dict(
            sorted(ready_for_target_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_ready_but_not_scheduled_count_by_depth": dict(
            sorted(ready_not_scheduled_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_ready_not_scheduled_reason_counts_by_depth": {
            str(depth): dict(sorted(reasons.items()))
            for depth, reasons in sorted(ready_not_scheduled_reason_counts.items(), key=lambda item: int(item[0]))
        },
        "unified_child_ready_seq_normal_lane_excluded_count_by_depth": dict(
            sorted(ready_lane_excluded_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_ready_seq_normal_lane_conflict_count_by_depth": dict(
            sorted(ready_lane_conflict_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_ready_seq_normal_lane_conflict_examples": ready_lane_conflict_examples[:8],
        "unified_child_stale_base_count_by_depth": dict(
            sorted(stale_base_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_stale_base_examples": stale_base_examples[:8],
        "unified_child_scheduled_for_target_verify_count_by_depth": dict(
            sorted(scheduled_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_target_verify_inflight_count_by_depth": dict(
            sorted(target_verify_inflight_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_scheduled_state_by_depth": {
            str(depth): dict(sorted(states.items()))
            for depth, states in sorted(scheduled_state_by_depth.items(), key=lambda item: int(item[0]))
        },
        "unified_child_duplicate_schedule_skip_count_by_depth": dict(
            sorted(duplicate_schedule_skip_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_schedule_state_error_count_by_depth": dict(
            sorted(schedule_state_error_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_schedule_state_error_examples": schedule_state_error_examples[:8],
        "unified_ready_child_lane_owner_seq_ids": list(ready_owner_seq_ids),
        "unified_ready_child_lane_owner_request_ids": dict(sorted(ready_owner_request_ids.items())),
        "unified_ready_child_excluded_from_normal_draft_seq_ids": list(ready_excluded_from_draft),
        "unified_ready_child_excluded_from_target_normal_verify_seq_ids": list(
            ready_excluded_from_target_normal
        ),
        "unified_ready_child_excluded_from_target_normal_verify_before_filter_seq_ids": list(
            ready_excluded_from_target_normal_before_filter
        ),
        "unified_ready_child_excluded_from_target_normal_verify_after_filter_seq_ids": list(
            ready_excluded_from_target_normal_after_filter
        ),
        "unified_ready_child_remaining_in_target_normal_verify_seq_ids": list(
            ready_remaining_in_target_normal
        ),
        "unified_ready_child_normal_verify_exclusion_mismatch_count": int(
            ready_exclusion_mismatch_count
        ),
        "unified_ready_child_normal_verify_exclusion_mismatch_examples": (
            ready_exclusion_mismatch_examples[:8]
        ),
        "unified_ready_child_missing_normal_proposal_allowed_seq_ids": list(
            ready_missing_normal_allowed
        ),
        "missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids": list(
            missing_buffered_allowed_unified
        ),
        "unified_child_target_verified_after_promotion_count_by_depth": dict(
            sorted(target_verified_after_promotion_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_invalidated_after_parent_non_full_count_by_depth": dict(
            sorted(invalidated_after_non_full_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_verified_after_parent_full_accept_count_by_depth": dict(
            sorted(verified_after_parent_full_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_verified_before_parent_result_count_by_depth": dict(
            sorted(verified_before_parent_result_trace_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_verified_before_parent_full_accept_violation_count": sum_record_int(
            records,
            "unified_child_verified_before_parent_full_accept_violation_count",
        ),
        "unified_child_generated_in_same_burst_as_grandchild_violation_count": sum_record_int(
            records,
            "unified_child_generated_in_same_burst_as_grandchild_violation_count",
        ),
        "unified_depth2_generated_while_depth1_verifying_count": sum_record_int(
            records,
            "unified_depth2_generated_while_depth1_verifying_count",
        ),
        "unified_depth2_ready_after_depth1_full_accept_count": sum_record_int(
            records,
            "unified_depth2_ready_after_depth1_full_accept_count",
        ),
        "unified_depth2_scheduled_after_depth1_full_accept_count": sum_record_int(
            records,
            "unified_depth2_scheduled_after_depth1_full_accept_count",
        ),
        "unified_depth2_verified_after_depth1_full_accept_count": sum_record_int(
            records,
            "unified_depth2_verified_after_depth1_full_accept_count",
        ),
        "unified_depth3_generated_while_depth2_verifying_count": sum_record_int(
            records,
            "unified_depth3_generated_while_depth2_verifying_count",
        ),
        "unified_child_generated_from_non_full_parent_count_by_depth": dict(
            sorted(non_full_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_generated_from_unverified_parent_count_by_depth": dict(
            sorted(unverified_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_generated_from_partial_parent_count_by_depth": dict(
            sorted(partial_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_generated_from_reject_parent_count_by_depth": dict(
            sorted(reject_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_generated_from_invalidated_parent_count_by_depth": dict(
            sorted(invalidated_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_parent_outcome_missing_count_by_depth": dict(
            sorted(missing_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_parent_full_accept_guard_violation_count": int(parent_guard_violation_count),
        "unified_child_parent_full_accept_guard_examples": guard_examples[:8],
        "unified_full_accept_parent_registered_count_by_depth": dict(
            sorted(registered_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_full_accept_parent_registry_size_by_depth": dict(
            sorted(registry_size_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_full_accept_parent_selected_for_child_count_by_depth": dict(
            sorted(selected_parent_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_full_accept_parent_not_selected_reason_counts_by_depth": {
            str(depth): dict(sorted(reasons.items()))
            for depth, reasons in sorted(parent_not_selected_reason_counts.items(), key=lambda item: int(item[0]))
        },
        "unified_child_generated_from_full_accept_parent_examples": child_full_accept_examples[:8],
        "unified_depth2_generation_block_reason_counts": dict(sorted(depth2_block_reason_counts.items())),
        "unified_full_accept_without_child_reason_counts_by_depth": {
            str(depth): dict(sorted(reasons.items()))
            for depth, reasons in sorted(full_accept_without_child_reason_counts.items(), key=lambda item: int(item[0]))
        },
        "unified_full_accept_parent_frontier_mismatch_count_by_delta": dict(
            sorted(frontier_mismatch_by_delta.items(), key=lambda item: int(item[0]))
        ),
        "unified_full_accept_parent_frontier_mismatch_examples": frontier_mismatch_examples[:8],
        "unified_full_accept_parent_active_seq_missing_count": sum_record_int(
            records,
            "unified_full_accept_parent_active_seq_missing_count",
        ),
        "unified_full_accept_parent_request_id_mismatch_count": sum_record_int(
            records,
            "unified_full_accept_parent_request_id_mismatch_count",
        ),
        "unified_full_accept_parent_seq_id_mismatch_count": sum_record_int(
            records,
            "unified_full_accept_parent_seq_id_mismatch_count",
        ),
        "unified_full_accept_parent_expected_frontier_len_source_counts": dict(
            sorted(expected_frontier_source_counts.items())
        ),
    }


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
    diagnostics = raw_and_budget_summary(records, candidate_by_depth, committed_by_depth)
    child_depth_verified_by_depth = {
        str(depth): int(count)
        for depth, count in diagnostics.get("unified_raw_verified_proposal_count_by_depth", {}).items()
        if int(depth) > 1
    }
    child_depth_invalidated_parent_not_full_by_depth = sum_depth_counts(
        records,
        "unified_invalidated_due_to_parent_not_full_accept_count_by_depth",
    )
    child_depth_invalidated_parent_not_full_by_depth = {
        str(depth): int(count)
        for depth, count in child_depth_invalidated_parent_not_full_by_depth.items()
        if int(depth) > 1
    }
    single_child_diagnostics = single_child_ahead_summary(
        records,
        result_args,
        candidate_by_depth,
        parent_by_id,
        depth_by_id,
    )
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

    proposal_tokens_by_id = merge_int_list_map(records, "generic_rolling_proposal_token_ids_by_proposal_id")
    to_verify_tokens_by_id = merge_int_list_map(records, "generic_rolling_to_be_verified_token_ids_by_proposal_id")
    to_verify_equals_by_id = merge_bool_map(records, "generic_rolling_to_verify_equals_proposal_by_proposal_id")
    to_verify_mismatch_ids: set[int] = {
        int(proposal_id) for proposal_id, is_equal in to_verify_equals_by_id.items() if not bool(is_equal)
    }
    for proposal_id in set(proposal_tokens_by_id) & set(to_verify_tokens_by_id):
        if proposal_tokens_by_id[proposal_id] != to_verify_tokens_by_id[proposal_id]:
            to_verify_mismatch_ids.add(int(proposal_id))

    target_seq_len_before = merge_int_map(
        records,
        "unified_generic_target_verify_seq_len_before_temp_append_by_proposal_id",
    )
    target_seq_len_after_rollback = merge_int_map(
        records,
        "unified_generic_target_verify_seq_len_after_rollback_by_proposal_id",
    )
    target_checkpoint_restored = merge_bool_map(
        records,
        "unified_generic_target_verify_checkpoint_restored_by_proposal_id",
    )
    target_input_ids = merge_int_list_map(
        records,
        "unified_generic_target_verify_input_ids_by_proposal_id",
    )
    target_next_round = merge_int_list_map(
        records,
        "unified_generic_target_verify_next_round_input_by_proposal_id",
    )
    target_original_proposals = merge_int_list_map(
        records,
        "unified_generic_target_verify_original_proposal_token_ids_by_proposal_id",
    )
    target_input_equals = merge_bool_map(
        records,
        "unified_generic_target_verify_input_equals_next_round_by_proposal_id",
    )
    target_next_round_equals = merge_bool_map(
        records,
        "unified_generic_target_verify_next_round_equals_original_proposal_by_proposal_id",
    )
    target_checkpoint_failed_ids = sorted(
        int(proposal_id) for proposal_id, restored in target_checkpoint_restored.items() if not bool(restored)
    )
    target_rollback_len_mismatch_ids = sorted(
        int(proposal_id)
        for proposal_id in set(target_seq_len_before) & set(target_seq_len_after_rollback)
        if int(target_seq_len_before[proposal_id]) != int(target_seq_len_after_rollback[proposal_id])
    )
    target_input_mismatch_ids = {
        int(proposal_id) for proposal_id, is_equal in target_input_equals.items() if not bool(is_equal)
    }
    for proposal_id in set(target_input_ids) & set(target_next_round):
        if target_input_ids[proposal_id] != target_next_round[proposal_id]:
            target_input_mismatch_ids.add(int(proposal_id))
    target_next_round_mismatch_ids = {
        int(proposal_id) for proposal_id, is_equal in target_next_round_equals.items() if not bool(is_equal)
    }
    for proposal_id in set(target_next_round) & set(target_original_proposals):
        if target_next_round[proposal_id] != target_original_proposals[proposal_id]:
            target_next_round_mismatch_ids.add(int(proposal_id))
    target_input_shape = aggregate_shape(records, "unified_generic_target_verify_input_ids_shape")
    target_logits_shape = aggregate_shape(records, "unified_generic_target_verify_logits_shape")
    current_window_hist = sum_nested_int_hist(
        records,
        "unified_generic_target_verify_current_window_accept_hist_by_depth",
    )
    proposal_window_shadow_hist = sum_nested_int_hist(
        records,
        "unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth",
    )
    current_mapping_hist = sum_nested_int_hist(
        records,
        "unified_generic_target_verify_current_mapping_accept_hist_by_depth",
    )
    shifted_mapping_hist = sum_nested_int_hist(
        records,
        "unified_generic_target_verify_shifted_mapping_accept_hist_by_depth",
    )
    proposal_window_verify_enabled = any_record_bool(
        records,
        "unified_generic_proposal_window_verify_enabled",
    ) or bool(result_args.get("enable_unified_generic_proposal_window_verify", False))
    gamma = max_record_int(records, "normal_gamma")
    sampled_current_to_verify = merge_int_list_map(
        records,
        "unified_generic_target_verify_current_to_be_verified_by_proposal_id",
    )
    sampled_proposal_to_verify = merge_int_list_map(
        records,
        "unified_generic_target_verify_proposal_window_to_be_verified_by_proposal_id",
    )
    current_first_token_probs = {}
    proposal_first_token_probs = {}
    for record in records:
        for raw_id, raw_value in (
            record.get("unified_generic_target_verify_current_window_first_token_prob_by_proposal_id") or {}
        ).items():
            try:
                current_first_token_probs.setdefault(int(raw_id), float(raw_value))
            except Exception:
                continue
        for raw_id, raw_value in (
            record.get("unified_generic_target_verify_proposal_window_first_token_prob_by_proposal_id") or {}
        ).items():
            try:
                proposal_first_token_probs.setdefault(int(raw_id), float(raw_value))
            except Exception:
                continue
    top5_tokens = merge_int_list_map(
        records,
        "unified_generic_target_verify_first_position_target_top5_tokens_by_proposal_id",
    )
    frontier_checkpoint_restored = merge_bool_map(
        records,
        "unified_generic_target_verify_frontier_checkpoint_restored_by_proposal_id",
    )
    frontier_block_table_match = merge_bool_map(
        records,
        "unified_generic_target_verify_frontier_block_table_match_by_proposal_id",
    )
    logits_owner_trace_present = has_trace_field(records, "unified_generic_target_verify_logits_owner")
    logits_owner_records = [
        record for record in records if bool(record.get("unified_generic_target_verify_logits_owner", False))
    ]
    logits_non_owner_records = [
        record
        for record in records
        if "unified_generic_target_verify_logits_owner" in record
        and not bool(record.get("unified_generic_target_verify_logits_owner", False))
    ]
    owner_target_input_shape = aggregate_shape(logits_owner_records, "unified_generic_target_verify_input_ids_shape")
    owner_target_logits_shape = aggregate_shape(logits_owner_records, "unified_generic_target_verify_logits_shape")
    non_owner_raw_result_count = sum(
        sum(int_value(value, 0) for value in as_depth_int_map(record.get(field)).values())
        for record in logits_non_owner_records
        for field in (
            "unified_raw_verified_proposal_count_by_depth",
            "unified_raw_full_accept_proposal_count_by_depth",
            "unified_raw_partial_accept_proposal_count_by_depth",
            "unified_raw_reject_proposal_count_by_depth",
            "unified_raw_invalidated_proposal_count_by_depth",
        )
    )
    buffer_lifecycle = normal_proposal_buffer_lifecycle_summary(records)

    explicit_full_by_depth = merge_depth_counts(
        records,
        "unified_full_commit_token_count_by_depth",
        "unified_generic_full_commit_token_count_by_depth",
    )
    action_by_id = merge_str_map(records, "generic_rolling_real_commit_action_by_proposal_id")
    result_by_id = merge_str_map(records, "generic_rolling_real_commit_verify_result_by_proposal_id")
    committed_token_by_id = merge_int_map(
        records,
        "generic_rolling_real_committed_token_count_by_proposal_id",
        "generic_rolling_token_count_by_proposal_id",
    )
    derived_full_by_depth: dict[str, int] = {}
    if action_by_id or result_by_id:
        for depth, proposal_ids in committed_by_depth.items():
            depth_key = str(int(depth))
            for proposal_id in set(int(item) for item in proposal_ids):
                action = str(action_by_id.get(proposal_id, ""))
                result = str(result_by_id.get(proposal_id, ""))
                is_partial = "partial" in action or "partial" in result or "revised" in action
                is_full = "full" in action or result == "full_accept" or (not action and not result)
                if not is_full or is_partial:
                    continue
                derived_full_by_depth[depth_key] = int(derived_full_by_depth.get(depth_key, 0)) + int(
                    committed_token_by_id.get(proposal_id, 0)
                )
        derived_full_by_depth = {
            depth: int(count)
            for depth, count in sorted(derived_full_by_depth.items(), key=lambda item: int(item[0]))
            if int(count) > 0
        }
    depth_commit_sum = sum_depth_values(depth_commit_counts)
    if explicit_full_by_depth:
        full_commit_token_count_by_depth = explicit_full_by_depth
    elif derived_full_by_depth:
        full_commit_token_count_by_depth = derived_full_by_depth
    elif depth_commit_sum == int(total_full) or int(total_partial) == 0:
        full_commit_token_count_by_depth = dict(depth_commit_counts)
    else:
        full_commit_token_count_by_depth = {}

    partial_recovered_by_depth = merge_depth_counts(
        records,
        "unified_partial_recovered_token_count_by_depth",
        "unified_generic_depth_partial_recovered_token_counts",
        "generic_full_continuous_depth_partial_recovered_token_counts",
    )
    partial_revised_by_depth = merge_depth_counts(
        records,
        "unified_partial_revised_token_count_by_depth",
        "unified_generic_depth_revised_token_counts",
        "generic_full_continuous_depth_revised_token_counts",
    )
    explicit_output_by_depth = merge_depth_counts(
        records,
        "unified_total_output_token_count_by_depth",
        "unified_generic_depth_output_token_counts",
    )
    total_output_by_depth = explicit_output_by_depth or add_depth_count_maps(
        full_commit_token_count_by_depth,
        partial_recovered_by_depth,
    )
    actual_verified_proposal_count = sum(
        int_value(value, 0)
        for value in diagnostics.get("unified_raw_verified_proposal_count_by_depth", {}).values()
    )
    total_candidate_proposal_count = sum(
        int_value(value, 0)
        for value in diagnostics.get("unified_raw_candidate_proposal_count_by_depth", {}).values()
    )
    denominator_source = (
        "unified_raw_verified_proposal_count_by_depth"
        if actual_verified_proposal_count > 0
        else "unified_generic_target_verify_num_proposals"
    )
    denominator_count = (
        int(actual_verified_proposal_count)
        if actual_verified_proposal_count > 0
        else sum_record_int(records, "unified_generic_target_verify_num_proposals")
    )

    return {
        "unified_generic_rolling_enabled": bool(unified_enabled),
        "normal_gamma": max_record_int(records, "normal_gamma"),
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
        "unified_child_depth_verified_count_by_depth": dict(
            sorted(child_depth_verified_by_depth.items(), key=lambda item: int(item[0]))
        ),
        "unified_child_depth_invalidated_parent_not_full_count_by_depth": dict(
            sorted(child_depth_invalidated_parent_not_full_by_depth.items(), key=lambda item: int(item[0]))
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
        **diagnostics,
        **single_child_diagnostics,
        "total_full_commit_token_count": total_full,
        "total_partial_recovered_token_count": total_partial,
        "total_revised_token_count": total_revised,
        "total_output_token_count": total_output,
        "unified_full_commit_token_count_by_depth": dict(full_commit_token_count_by_depth),
        "unified_total_full_commit_token_count": int(total_full),
        "unified_partial_recovered_token_count_by_depth": dict(partial_recovered_by_depth),
        "unified_total_partial_recovered_token_count": int(total_partial),
        "unified_partial_revised_token_count_by_depth": dict(partial_revised_by_depth),
        "unified_total_output_token_count_by_depth": dict(total_output_by_depth),
        "unified_total_output_token_count": int(total_output),
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
        "generic_rolling_to_verify_equals_proposal_all": not bool(to_verify_mismatch_ids),
        "generic_rolling_to_verify_mismatch_proposal_ids": sorted(to_verify_mismatch_ids),
        "normal_proposal_generated_seq_ids": merge_int_lists(records, "normal_proposal_generated_seq_ids"),
        "normal_proposal_generated_request_ids": {
            str(seq_id): request_id
            for seq_id, request_id in sorted(
                merge_str_map(records, "normal_proposal_generated_request_ids").items()
            )
        },
        "normal_proposal_transfer_sent_seq_ids": merge_int_lists(
            records,
            "normal_proposal_transfer_sent_seq_ids",
        ),
        "normal_proposal_transfer_received_seq_ids": merge_int_lists(
            records,
            "normal_proposal_transfer_received_seq_ids",
        ),
        "dual_proposal_buffer_store_seq_ids": merge_int_lists(records, "dual_proposal_buffer_store_seq_ids"),
        "dual_proposal_buffer_discard_seq_ids": merge_int_lists(
            records,
            "dual_proposal_buffer_discard_seq_ids",
        ),
        "dual_proposal_buffer_available_seq_ids_before_target_verify": merge_int_lists(
            records,
            "dual_proposal_buffer_available_seq_ids_before_target_verify",
        ),
        "target_normal_verify_seq_ids_before_buffer_filter": merge_int_lists(
            records,
            "target_normal_verify_seq_ids_before_buffer_filter",
        ),
        "target_normal_verify_seq_ids_after_buffer_filter": merge_int_lists(
            records,
            "target_normal_verify_seq_ids_after_buffer_filter",
        ),
        "target_normal_verify_missing_buffer_seq_ids": merge_int_lists(
            records,
            "target_normal_verify_missing_buffer_seq_ids",
        ),
        "target_normal_verify_deferred_missing_buffer_seq_ids": merge_int_lists(
            records,
            "target_normal_verify_deferred_missing_buffer_seq_ids",
        ),
        "target_normal_verify_missing_buffer_request_ids": {
            str(seq_id): request_id
            for seq_id, request_id in sorted(
                merge_str_map(records, "target_normal_verify_missing_buffer_request_ids").items()
            )
        },
        "target_normal_verify_missing_buffer_reason_by_seq_id": {
            str(seq_id): reason
            for seq_id, reason in sorted(
                merge_str_map(records, "target_normal_verify_missing_buffer_reason_by_seq_id").items()
            )
        },
        "target_normal_verify_deferred_missing_buffer_reason_counts": sum_counter_fields(
            records,
            "target_normal_verify_deferred_missing_buffer_reason_counts",
        ),
        "target_normal_verify_missing_buffer_details": collect_dict_examples(
            records,
            "target_normal_verify_missing_buffer_details",
        ),
        "normal_proposal_buffer_event_history": collect_dict_examples(
            records,
            "normal_proposal_buffer_event_history",
        ),
        "unified_ready_child_owner_cleared_seq_ids": merge_int_lists(
            records,
            "unified_ready_child_owner_cleared_seq_ids",
        ),
        "unified_ready_child_owner_clear_reason_by_seq_id": {
            str(seq_id): reason
            for seq_id, reason in sorted(
                merge_str_map(records, "unified_ready_child_owner_clear_reason_by_seq_id").items()
            )
        },
        **buffer_lifecycle,
        "unified_generic_target_verify_temp_append_used": any_record_bool(
            records,
            "unified_generic_target_verify_temp_append_used",
        ),
        "unified_generic_target_verify_num_proposals": sum_record_int(
            records,
            "unified_generic_target_verify_num_proposals",
        ),
        "unified_generic_target_verify_num_to_verify_tokens": sum_record_int(
            records,
            "unified_generic_target_verify_num_to_verify_tokens",
        ),
        "unified_generic_target_verify_actual_verified_proposal_count": int(actual_verified_proposal_count),
        "unified_generic_target_verify_total_candidate_proposal_count": int(total_candidate_proposal_count),
        "unified_generic_target_verify_deferred_or_invalidated_proposal_count": max(
            0,
            int(total_candidate_proposal_count) - int(actual_verified_proposal_count),
        ),
        "unified_generic_target_verify_token_count_denominator_source": str(denominator_source),
        "unified_generic_target_verify_token_count_denominator_count": int(denominator_count),
        "unified_generic_target_verify_input_ids_shape": target_input_shape,
        "unified_generic_target_verify_logits_shape": target_logits_shape,
        "unified_generic_target_verify_logits_rows_per_proposal": max_record_int(
            records,
            "unified_generic_target_verify_logits_rows_per_proposal",
        ),
        "unified_generic_target_verify_logits_owner": any_record_bool(
            records,
            "unified_generic_target_verify_logits_owner",
        ),
        "unified_generic_target_verify_logits_owner_record_count": len(logits_owner_records),
        "unified_generic_target_verify_owner_num_proposals": sum_record_int(
            logits_owner_records,
            "unified_generic_target_verify_num_proposals",
        ),
        "unified_generic_target_verify_owner_num_to_verify_tokens": sum_record_int(
            logits_owner_records,
            "unified_generic_target_verify_num_to_verify_tokens",
        ),
        "unified_generic_target_verify_owner_input_ids_shape": owner_target_input_shape,
        "unified_generic_target_verify_owner_logits_shape": owner_target_logits_shape,
        "unified_generic_target_verify_owner_uses_shifted_logits": any_record_bool(
            logits_owner_records,
            "unified_generic_target_verify_uses_shifted_logits",
        ),
        "unified_generic_target_verify_owner_frontier_logits_available": any_record_bool(
            logits_owner_records,
            "unified_generic_target_verify_frontier_logits_available",
        ),
        "unified_generic_target_verify_owner_appended_logits_available": any_record_bool(
            logits_owner_records,
            "unified_generic_target_verify_appended_logits_available",
        ),
        "unified_generic_target_verify_non_owner_record_count": len(logits_non_owner_records),
        "unified_generic_target_verify_non_owner_frontier_none_allowed_count": sum(
            1
            for record in logits_non_owner_records
            if bool(record.get("unified_generic_target_verify_frontier_logits_none_allowed", False))
        ),
        "unified_generic_target_verify_non_owner_raw_result_count": int(non_owner_raw_result_count),
        "unified_generic_target_verify_logits_owner_trace_present": bool(logits_owner_trace_present),
        "unified_generic_target_verify_checkpoint_failed_proposal_ids": target_checkpoint_failed_ids,
        "unified_generic_target_verify_rollback_len_mismatch_proposal_ids": target_rollback_len_mismatch_ids,
        "unified_generic_target_verify_input_mismatch_proposal_ids": sorted(target_input_mismatch_ids),
        "unified_generic_target_verify_next_round_mismatch_proposal_ids": sorted(target_next_round_mismatch_ids),
        "unified_generic_target_verify_sampled_input_ids_by_proposal_id": {
            str(proposal_id): list(tokens)
            for proposal_id, tokens in sorted(target_input_ids.items())
        },
        "unified_generic_target_verify_sampled_next_round_input_by_proposal_id": {
            str(proposal_id): list(tokens)
            for proposal_id, tokens in sorted(target_next_round.items())
        },
        "unified_generic_target_verify_sampled_original_proposal_token_ids_by_proposal_id": {
            str(proposal_id): list(tokens)
            for proposal_id, tokens in sorted(target_original_proposals.items())
        },
        "unified_generic_proposal_window_verify_enabled": bool(proposal_window_verify_enabled),
        "unified_generic_target_verify_current_window_accept_hist_by_depth": current_window_hist,
        "unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth": proposal_window_shadow_hist,
        "unified_generic_target_verify_current_window_all_first_token_reject": (
            nested_hist_all_first_token_reject(current_window_hist)
        ),
        "unified_generic_target_verify_proposal_window_shadow_all_first_token_reject": (
            nested_hist_all_first_token_reject(proposal_window_shadow_hist)
        ),
        "unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept": (
            nested_hist_nonzero_accept_count(proposal_window_shadow_hist) > 0
        ),
        "unified_generic_target_verify_proposal_window_shadow_full_accept_count": (
            nested_hist_full_accept_count(proposal_window_shadow_hist, gamma)
        ),
        "unified_generic_target_verify_proposal_window_shadow_partial_accept_count": (
            nested_hist_partial_accept_count(proposal_window_shadow_hist, gamma)
        ),
        "unified_generic_target_verify_proposal_window_shadow_reject_count": (
            nested_hist_reject_count(proposal_window_shadow_hist)
        ),
        "unified_generic_target_verify_uses_shifted_logits": any_record_bool(
            records,
            "unified_generic_target_verify_uses_shifted_logits",
        ),
        "unified_generic_target_verify_frontier_logits_available": any_record_bool(
            records,
            "unified_generic_target_verify_frontier_logits_available",
        ),
        "unified_generic_target_verify_current_mapping_accept_hist_by_depth": current_mapping_hist,
        "unified_generic_target_verify_shifted_mapping_accept_hist_by_depth": shifted_mapping_hist,
        "unified_generic_target_verify_shifted_mapping_has_nonzero_accept": (
            nested_hist_nonzero_accept_count(shifted_mapping_hist) > 0
        ),
        "unified_generic_target_verify_frontier_checkpoint_failed_proposal_ids": sorted(
            int(proposal_id)
            for proposal_id, restored in frontier_checkpoint_restored.items()
            if not bool(restored)
        ),
        "unified_generic_target_verify_frontier_block_table_mismatch_proposal_ids": sorted(
            int(proposal_id)
            for proposal_id, matches in frontier_block_table_match.items()
            if not bool(matches)
        ),
        "unified_generic_target_verify_sampled_current_to_be_verified_by_proposal_id": {
            str(proposal_id): list(tokens)
            for proposal_id, tokens in sorted(sampled_current_to_verify.items())
        },
        "unified_generic_target_verify_sampled_proposal_window_to_be_verified_by_proposal_id": {
            str(proposal_id): list(tokens)
            for proposal_id, tokens in sorted(sampled_proposal_to_verify.items())
        },
        "unified_generic_target_verify_sampled_current_window_first_token_prob_by_proposal_id": {
            str(proposal_id): float(value)
            for proposal_id, value in sorted(current_first_token_probs.items())
        },
        "unified_generic_target_verify_sampled_proposal_window_first_token_prob_by_proposal_id": {
            str(proposal_id): float(value)
            for proposal_id, value in sorted(proposal_first_token_probs.items())
        },
        "unified_generic_target_verify_sampled_first_position_target_top5_tokens_by_proposal_id": {
            str(proposal_id): list(tokens)
            for proposal_id, tokens in sorted(top5_tokens.items())
        },
    }


def validate_normal_proposal_buffer_filter(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    new_trace_fields = {
        "dual_proposal_buffer_available_seq_ids_before_target_verify",
        "target_normal_verify_seq_ids_before_buffer_filter",
        "target_normal_verify_seq_ids_after_buffer_filter",
        "target_normal_verify_missing_buffer_seq_ids",
        "target_normal_verify_deferred_missing_buffer_seq_ids",
        "target_normal_verify_missing_buffer_reason_by_seq_id",
    }
    for index, record in enumerate(records):
        if not any(field in record for field in new_trace_fields):
            continue
        before = as_int_list(
            record.get("target_normal_verify_seq_ids_before_buffer_filter")
            if "target_normal_verify_seq_ids_before_buffer_filter" in record
            else record.get("target_normal_verify_seq_ids")
        )
        after = as_int_list(
            record.get("target_normal_verify_seq_ids_after_buffer_filter")
            if "target_normal_verify_seq_ids_after_buffer_filter" in record
            else record.get("target_normal_verify_seq_ids")
        )
        available = set(as_int_list(record.get("dual_proposal_buffer_available_seq_ids_before_target_verify")))
        eager_owned = set(as_int_list(record.get("target_eager_verify_seq_ids_dry_run")))
        eager_owned.update(as_int_list(record.get("excluded_from_target_normal_verify_for_eager_dry_run")))
        eager_owned.update(as_int_list(record.get("missing_buffered_proposal_allowed_by_eager_seq_ids")))
        unified_owned = set(as_int_list(record.get("unified_ready_child_lane_owner_seq_ids")))
        unified_owned.update(as_int_list(record.get("unified_ready_child_excluded_from_target_normal_verify_seq_ids")))
        unified_owned.update(as_int_list(record.get("unified_ready_child_missing_normal_proposal_allowed_seq_ids")))
        unified_owned.update(as_int_list(record.get("missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids")))
        owned = eager_owned | unified_owned
        reason_by_seq_id = as_str_map(record.get("target_normal_verify_missing_buffer_reason_by_seq_id"))
        fallback_pending = {
            seq_id
            for seq_id, reason in reason_by_seq_id.items()
            if str(reason) == "fallback_pending_receive"
        }

        after_set = set(after)
        owned_after = sorted(after_set & owned)
        if owned_after:
            errors.append(
                "target normal verify after buffer filter retained owned seqs: "
                f"record_index={index}, seq_ids={owned_after}"
            )
        missing_after = sorted(seq_id for seq_id in after if seq_id not in available and seq_id not in fallback_pending)
        if missing_after:
            errors.append(
                "target normal verify after buffer filter includes seqs without buffered proposals: "
                f"record_index={index}, after={after}, available={sorted(available)}, missing={missing_after}"
            )

        missing_before = as_int_list(record.get("target_normal_verify_missing_buffer_seq_ids"))
        if missing_before:
            missing_without_reason = sorted(seq_id for seq_id in missing_before if not reason_by_seq_id.get(seq_id))
            if missing_without_reason:
                errors.append(
                    "target normal verify missing-buffer seqs require explicit reasons: "
                    f"record_index={index}, missing={missing_without_reason}"
                )
            deferred = set(as_int_list(record.get("target_normal_verify_deferred_missing_buffer_seq_ids")))
            deferred_without_reason = sorted(
                seq_id
                for seq_id in deferred
                if str(reason_by_seq_id.get(seq_id, "")) != "missing_normal_proposal_deferred"
            )
            if deferred_without_reason:
                errors.append(
                    "target normal verify deferred missing-buffer seqs require missing_normal_proposal_deferred reason: "
                    f"record_index={index}, missing={deferred_without_reason}"
                )
            if deferred & after_set:
                errors.append(
                    "target normal verify after buffer filter retained deferred missing-buffer seqs: "
                    f"record_index={index}, seq_ids={sorted(deferred & after_set)}"
                )
            reason_counts = record.get("target_normal_verify_deferred_missing_buffer_reason_counts")
            if deferred and not isinstance(reason_counts, dict):
                errors.append(
                    "target normal verify deferred missing-buffer seqs require reason counts: "
                    f"record_index={index}, deferred={sorted(deferred)}"
                )

        if before and "target_normal_verify_seq_ids_after_buffer_filter" in record:
            missing_not_available = sorted(seq_id for seq_id in before if seq_id not in available)
            unexplained = [
                seq_id
                for seq_id in missing_not_available
                if seq_id not in owned and seq_id not in fallback_pending and not reason_by_seq_id.get(seq_id)
            ]
            if unexplained:
                errors.append(
                    "target normal verify before buffer filter had missing seqs without deferral/ownership reason: "
                    f"record_index={index}, seq_ids={unexplained}"
                )
    return errors


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    summary = build_summary(records, result_payload)
    errors: list[str] = []
    errors.extend(validate_normal_proposal_buffer_filter(records))
    if not summary["unified_generic_rolling_enabled"]:
        errors.append("unified generic rolling runtime must be enabled")
        return errors, summary
    raw_target_field_present = any(
        "unified_raw_target_verification_available" in record
        or "unified_raw_verification_source" in record
        for record in records
    )
    raw_source = str(summary.get("unified_raw_verification_source") or "")
    if raw_target_field_present and (
        not bool(summary.get("unified_raw_target_verification_available"))
        or "draft_commit_decision_no_target_verify" in raw_source
    ):
        errors.append("strict unified target verification must be available")

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
    raw_verified_total = sum(int_value(value, 0) for value in summary["unified_raw_verified_proposal_count_by_depth"].values())
    raw_full_total = sum(int_value(value, 0) for value in summary["unified_raw_full_accept_proposal_count_by_depth"].values())
    temp_append_field_present = has_trace_field(records, "unified_generic_target_verify_temp_append_used")
    if raw_verified_total > 0 and temp_append_field_present:
        if not bool(summary.get("unified_generic_target_verify_temp_append_used", False)):
            errors.append("unified generic target verification must use temporary append")
        if int_value(summary.get("unified_generic_target_verify_num_proposals"), 0) <= 0:
            errors.append("temporary target verification proposal count must be positive")
        if summary.get("unified_generic_target_verify_checkpoint_failed_proposal_ids"):
            errors.append(
                "temporary target verification checkpoint restore failed: "
                f"{summary['unified_generic_target_verify_checkpoint_failed_proposal_ids']}"
            )
        if summary.get("unified_generic_target_verify_rollback_len_mismatch_proposal_ids"):
            errors.append(
                "temporary target verification rollback length mismatch: "
                f"{summary['unified_generic_target_verify_rollback_len_mismatch_proposal_ids']}"
            )
        if summary.get("unified_generic_target_verify_input_mismatch_proposal_ids"):
            errors.append(
                "temporary target verification input_ids must equal next_round_input for sampled proposals: "
                f"{summary['unified_generic_target_verify_input_mismatch_proposal_ids']}"
            )
        if summary.get("unified_generic_target_verify_next_round_mismatch_proposal_ids"):
            errors.append(
                "temporary target verification next_round_input must equal original proposal tokens for sampled proposals: "
                f"{summary['unified_generic_target_verify_next_round_mismatch_proposal_ids']}"
            )
        gamma = int_value(summary.get("normal_gamma"), 0)
        owner_trace_present = bool(summary.get("unified_generic_target_verify_logits_owner_trace_present", False))
        num_proposals = int_value(
            summary.get(
                "unified_generic_target_verify_owner_num_proposals"
                if owner_trace_present
                else "unified_generic_target_verify_num_proposals"
            ),
            0,
        )
        num_to_verify = int_value(
            summary.get(
                "unified_generic_target_verify_owner_num_to_verify_tokens"
                if owner_trace_present
                else "unified_generic_target_verify_num_to_verify_tokens"
            ),
            0,
        )
        input_shape = as_int_list(
            summary.get(
                "unified_generic_target_verify_owner_input_ids_shape"
                if owner_trace_present
                else "unified_generic_target_verify_input_ids_shape"
            )
        )
        logits_shape = as_int_list(
            summary.get(
                "unified_generic_target_verify_owner_logits_shape"
                if owner_trace_present
                else "unified_generic_target_verify_logits_shape"
            )
        )
        logits_rows = logits_shape[0] if logits_shape else 0
        input_rows = input_shape[0] if input_shape else 0
        denominator_count = int_value(
            summary.get("unified_generic_target_verify_token_count_denominator_count"),
            num_proposals,
        )
        if gamma > 0 and denominator_count > 0 and num_to_verify != gamma * denominator_count:
            errors.append(
                "temporary target verification to-verify token count must equal gamma * actual verified proposals: "
                f"tokens={num_to_verify}, gamma={gamma}, denominator={denominator_count}, "
                f"source={summary.get('unified_generic_target_verify_token_count_denominator_source')}"
            )
        if logits_rows != num_to_verify:
            errors.append("temporary target verification logits rows must equal num_to_verify_tokens")
        if input_rows != logits_rows:
            errors.append("temporary target verification input rows must equal logits rows")
        rows_per_proposal = int_value(summary.get("unified_generic_target_verify_logits_rows_per_proposal"), 0)
        if gamma > 0 and rows_per_proposal and rows_per_proposal != gamma:
            errors.append("temporary target verification logits rows per proposal must equal gamma")
        if bool(summary.get("unified_generic_proposal_window_verify_enabled", False)):
            proposal_shadow_hist = summary.get(
                "unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth",
                {},
            )
            actual_hist = summary.get("unified_raw_accepted_len_hist_by_depth", {})
            if proposal_shadow_hist and actual_hist != proposal_shadow_hist:
                errors.append(
                    "proposal-window verify enabled but actual accepted_len hist does not match proposal-window shadow hist"
                )
        shifted_trace_present = has_trace_field(records, "unified_generic_target_verify_uses_shifted_logits")
        if shifted_trace_present:
            owner_trace_present = bool(summary.get("unified_generic_target_verify_logits_owner_trace_present", False))
            owner_num_proposals = int_value(summary.get("unified_generic_target_verify_owner_num_proposals"), 0)
            if owner_trace_present:
                if int_value(summary.get("unified_generic_target_verify_logits_owner_record_count"), 0) <= 0:
                    errors.append("unified generic target verification must have a logits owner record")
                if owner_num_proposals > 0:
                    if not bool(summary.get("unified_generic_target_verify_owner_uses_shifted_logits", False)):
                        errors.append("unified generic target verification owner must use shifted logits")
                    if not bool(
                        summary.get("unified_generic_target_verify_owner_frontier_logits_available", False)
                    ):
                        errors.append("unified generic target verification owner must have frontier logits")
                    if not bool(
                        summary.get("unified_generic_target_verify_owner_appended_logits_available", False)
                    ):
                        errors.append("unified generic target verification owner must have appended logits")
                non_owner_count = int_value(
                    summary.get("unified_generic_target_verify_non_owner_record_count"),
                    0,
                )
                non_owner_none_allowed = int_value(
                    summary.get("unified_generic_target_verify_non_owner_frontier_none_allowed_count"),
                    0,
                )
                if non_owner_count and non_owner_none_allowed < non_owner_count:
                    errors.append("non-owner target TP records must allow missing frontier logits")
                if int_value(summary.get("unified_generic_target_verify_non_owner_raw_result_count"), 0) > 0:
                    errors.append("non-owner target TP ranks must not record raw verify result counters")
            else:
                if not bool(summary.get("unified_generic_target_verify_uses_shifted_logits", False)):
                    errors.append("unified generic target verification must use shifted logits")
                if not bool(summary.get("unified_generic_target_verify_frontier_logits_available", False)):
                    errors.append("unified generic target verification must have frontier logits")
            if summary.get("unified_generic_target_verify_frontier_checkpoint_failed_proposal_ids"):
                errors.append(
                    "frontier no-apply forward changed sequence checkpoint: "
                    f"{summary['unified_generic_target_verify_frontier_checkpoint_failed_proposal_ids']}"
                )
            if summary.get("unified_generic_target_verify_frontier_block_table_mismatch_proposal_ids"):
                errors.append(
                    "frontier no-apply forward changed block table: "
                    f"{summary['unified_generic_target_verify_frontier_block_table_mismatch_proposal_ids']}"
                )
            shifted_hist = summary.get("unified_generic_target_verify_shifted_mapping_accept_hist_by_depth", {})
            actual_hist = summary.get("unified_raw_accepted_len_hist_by_depth", {})
            if shifted_hist and actual_hist != shifted_hist:
                errors.append("unified raw accepted_len hist must match shifted mapping hist")
    target_verified_reject_only = bool(
        summary.get("unified_raw_target_verification_available")
        and raw_verified_total > 0
        and raw_full_total == 0
    )
    single_child_ahead_enabled = bool(summary.get("unified_single_child_ahead_enabled", False))
    if single_child_ahead_enabled:
        ahead_limit = int_value(summary.get("unified_max_unverified_depth_ahead"), 0)
        observed_ahead = int_value(summary.get("unified_unverified_depth_ahead_max_observed"), 0)
        if ahead_limit <= 0:
            errors.append("single-child-ahead mode requires positive max unverified depth ahead")
        if ahead_limit > 0 and observed_ahead > ahead_limit:
            errors.append(
                "single-child-ahead observed unverified depth ahead exceeds configured limit: "
                f"observed={observed_ahead} limit={ahead_limit}"
            )
        if int_value(summary.get("unified_single_child_ahead_violation_count"), 0) > 0:
            errors.append(
                "single-child-ahead violation examples: "
                f"{summary.get('unified_single_child_ahead_violation_examples')}"
            )
        if int_value(summary.get("unified_generated_grandchild_before_parent_verified_count"), 0) > 0:
            errors.append(
                "generated grandchild before parent verification examples: "
                f"{summary.get('unified_generated_grandchild_before_parent_verified_examples')}"
            )
        if int_value(summary.get("unified_child_verified_before_parent_full_accept_violation_count"), 0) > 0:
            errors.append(
                "single-child target verified child before parent full accept: "
                f"{summary.get('unified_child_verified_before_parent_result_count_by_depth')}"
            )
        if int_value(summary.get("unified_child_generated_in_same_burst_as_grandchild_violation_count"), 0) > 0:
            errors.append("single-child generated grandchild in the same speculative burst")
        if int_value(summary.get("unified_child_parent_full_accept_guard_violation_count"), 0) > 0:
            errors.append(
                "single-child parent full-accept guard violation examples: "
                f"{summary.get('unified_child_parent_full_accept_guard_examples')}"
            )
        child_parent_not_full = summary.get("unified_child_depth_invalidated_parent_not_full_count_by_depth", {})
        if any(int_value(value, 0) > 0 for value in child_parent_not_full.values()):
            errors.append(
                "single-child child depth invalidated as parent_not_full_accept: "
                f"{child_parent_not_full}"
            )
        child_full_parent = summary.get("unified_child_generated_from_full_accept_parent_count_by_depth", {})
        child_inflight_parent = summary.get("unified_child_generated_from_inflight_parent_count_by_depth", {})
        candidate_counts = summary.get("unified_raw_candidate_proposal_count_by_depth", {})
        missing_provenance_depths = [
            str(depth)
            for depth, count in sorted(candidate_counts.items(), key=lambda item: int(item[0]))
            if int(depth) > 1
            and int_value(count, 0) > 0
            and int_value(child_full_parent.get(str(depth)), 0) <= 0
            and int_value(child_inflight_parent.get(str(depth)), 0) <= 0
        ]
        if missing_provenance_depths:
            errors.append(
                "single-child depth>1 candidates require full-accepted or in-flight parent provenance: "
                f"depths={missing_provenance_depths}"
            )
        promoted_by_depth = summary.get("unified_child_promoted_after_parent_full_accept_count_by_depth", {})
        verified_after_parent_full = summary.get("unified_child_verified_after_parent_full_accept_count_by_depth", {})
        target_verified_after_promotion = summary.get(
            "unified_child_target_verified_after_promotion_count_by_depth",
            {},
        )
        ready_not_scheduled_reasons = summary.get(
            "unified_child_ready_not_scheduled_reason_counts_by_depth",
            {},
        )
        ready_for_target_by_depth = summary.get("unified_child_ready_for_target_verify_count_by_depth", {})
        scheduled_after_promotion = summary.get("unified_child_scheduled_for_target_verify_count_by_depth", {})
        ready_lane_excluded = summary.get(
            "unified_child_ready_seq_normal_lane_excluded_count_by_depth",
            {},
        )
        ready_lane_conflict = summary.get(
            "unified_child_ready_seq_normal_lane_conflict_count_by_depth",
            {},
        )
        schedule_state_errors = summary.get(
            "unified_child_schedule_state_error_count_by_depth",
            {},
        )
        if any(int_value(count, 0) > 0 for count in schedule_state_errors.values()):
            errors.append(
                "single-child promoted-child scheduling saw invalid proposal state: "
                f"counts={schedule_state_errors}, examples="
                f"{summary.get('unified_child_schedule_state_error_examples', [])}"
            )
        owner_target_remaining = summary.get(
            "unified_ready_child_remaining_in_target_normal_verify_seq_ids",
            [],
        )
        owner_exclusion_mismatch_count = int_value(
            summary.get("unified_ready_child_normal_verify_exclusion_mismatch_count"),
            0,
        )
        if owner_target_remaining or owner_exclusion_mismatch_count > 0:
            errors.append(
                "single-child ready-child owners must be excluded from target normal verify: "
                f"remaining={owner_target_remaining}, mismatch_count={owner_exclusion_mismatch_count}, "
                f"examples={summary.get('unified_ready_child_normal_verify_exclusion_mismatch_examples', [])}"
            )
        for index, record in enumerate(records):
            owner_seq_ids = set(as_int_list(record.get("unified_ready_child_lane_owner_seq_ids")))
            unexpected_missing = as_int_list(record.get("missing_buffered_proposal_unexpected_seq_ids"))
            if unexpected_missing:
                errors.append(
                    "single-child target normal verify is missing non-owned buffered proposals: "
                    f"record_index={index}, missing={unexpected_missing}"
                )
            if not owner_seq_ids:
                continue
            draft_overlap = sorted(
                owner_seq_ids & set(as_int_list(record.get("actual_draft_home_set_for_normal_draft")))
            )
            target_overlap = sorted(
                owner_seq_ids & set(as_int_list(record.get("target_normal_verify_seq_ids")))
            )
            if draft_overlap or target_overlap:
                errors.append(
                    "single-child ready-child owner remained in a normal lane: "
                    f"record_index={index}, draft_overlap={draft_overlap}, target_overlap={target_overlap}, "
                    f"owners={sorted(owner_seq_ids)}"
                )
            allowed_by_unified = set(
                as_int_list(record.get("missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids"))
            )
            if allowed_by_unified & set(as_int_list(record.get("target_normal_verify_seq_ids"))):
                errors.append(
                    "single-child unified-ready missing-proposal allowance must not leave seq in target normal verify: "
                    f"record_index={index}, overlap="
                    f"{sorted(allowed_by_unified & set(as_int_list(record.get('target_normal_verify_seq_ids'))))}"
                )
        disappeared_promoted_depths: list[str] = []
        for depth, count in sorted(promoted_by_depth.items(), key=lambda item: int(item[0])):
            if int_value(count, 0) <= 0:
                continue
            if (
                int_value(verified_after_parent_full.get(str(depth)), 0) <= 0
                and int_value(target_verified_after_promotion.get(str(depth)), 0) <= 0
                and not ready_not_scheduled_reasons.get(str(depth))
            ):
                disappeared_promoted_depths.append(str(depth))
        if disappeared_promoted_depths:
            errors.append(
                "single-child promoted children were not target verified and lack ready-not-scheduled reasons: "
                f"depths={disappeared_promoted_depths}"
            )
        unscheduled_ready_depths: list[str] = []
        stale_only_depths: list[str] = []
        non_stale_block_reasons = {
            "budget_exhausted",
            "no_target_slot",
            "sequence_finished",
            "no_active_sequence",
            "max_depth_reached",
            "lane_conflict",
            "not_in_target_home_set",
        }
        for depth, count in sorted(ready_for_target_by_depth.items(), key=lambda item: int(item[0])):
            depth_key = str(depth)
            if int_value(count, 0) <= 0 or int_value(scheduled_after_promotion.get(depth_key), 0) > 0:
                continue
            reasons = ready_not_scheduled_reasons.get(depth_key, {}) or {}
            non_stale_count = sum(
                int_value(reason_count, 0)
                for reason, reason_count in reasons.items()
                if str(reason) in non_stale_block_reasons
            )
            stale_count = int_value(reasons.get("stale_base"), 0)
            if non_stale_count <= 0:
                unscheduled_ready_depths.append(depth_key)
            if (
                stale_count > 0
                and stale_count >= int_value(count, 0)
                and int_value(scheduled_after_promotion.get(depth_key), 0) <= 0
                and non_stale_count <= 0
            ):
                stale_only_depths.append(depth_key)
        if unscheduled_ready_depths:
            errors.append(
                "single-child ready children were not scheduled for target verification and lack non-stale block reasons: "
                f"depths={unscheduled_ready_depths}, reasons={ready_not_scheduled_reasons}"
            )
        if stale_only_depths:
            errors.append(
                "single-child ready children were dropped as stale_base instead of being owned/scheduled: "
                f"depths={stale_only_depths}, lane_excluded={ready_lane_excluded}, lane_conflict={ready_lane_conflict}"
            )
        full_accept_by_depth = summary.get("unified_raw_full_accept_proposal_count_by_depth", {})
        full_child_by_depth = summary.get("unified_child_generated_from_full_accept_parent_count_by_depth", {})
        inflight_child_by_depth = summary.get("unified_child_generated_from_inflight_parent_count_by_depth", {})
        selected_parent_by_depth = summary.get("unified_full_accept_parent_selected_for_child_count_by_depth", {})
        not_selected_reasons = summary.get("unified_full_accept_without_child_reason_counts_by_depth", {})
        unexplained_full_depths: list[str] = []
        for depth, count in sorted(full_accept_by_depth.items(), key=lambda item: int(item[0])):
            depth_int = int(depth)
            if depth_int >= max_depth or int_value(count, 0) <= 0:
                continue
            child_depth = str(depth_int + 1)
            if (
                int_value(full_child_by_depth.get(child_depth), 0) <= 0
                and int_value(inflight_child_by_depth.get(child_depth), 0) <= 0
                and int_value(selected_parent_by_depth.get(str(depth_int)), 0) <= 0
                and not not_selected_reasons.get(str(depth_int))
            ):
                unexplained_full_depths.append(str(depth_int))
        if unexplained_full_depths:
            errors.append(
                "single-child full-accepted parents did not produce continuation children or block reasons: "
                f"depths={unexplained_full_depths}"
            )
    if max_depth > 4 and max_real <= 4 and not target_verified_reject_only and not single_child_ahead_enabled:
        errors.append("max real committed depth must exceed 4 when configured max depth exceeds 4")

    split_committed: dict[str, int] = {}
    for field in (
        "unified_raw_full_commit_proposal_count_by_depth",
        "unified_raw_partial_recovery_applied_proposal_count_by_depth",
        "unified_raw_reject_revised_correction_applied_proposal_count_by_depth",
    ):
        for depth, count in summary.get(field, {}).items():
            split_committed[str(depth)] = int(split_committed.get(str(depth), 0)) + int_value(count, 0)
    if split_committed and split_committed != summary.get("unified_raw_committed_proposal_count_by_depth", {}):
        errors.append("raw committed proposal count must equal applied full/partial/reject-correction counts")
    reject_total = sum(int_value(value, 0) for value in summary["unified_raw_reject_proposal_count_by_depth"].values())
    no_mutation_reject_total = sum(
        int_value(value, 0) for value in summary.get("unified_raw_no_mutation_reject_proposal_count_by_depth", {}).values()
    )
    reject_correction_total = sum(
        int_value(value, 0)
        for value in summary.get("unified_raw_reject_revised_correction_applied_proposal_count_by_depth", {}).values()
    )
    if reject_total and no_mutation_reject_total + reject_correction_total > reject_total:
        errors.append("reject apply counters must not exceed raw reject outcomes")

    total_full = int_value(summary["total_full_commit_token_count"], 0)
    total_partial = int_value(summary["total_partial_recovered_token_count"], 0)
    total_revised = int_value(summary["total_revised_token_count"], 0)
    total_output = int_value(summary["total_output_token_count"], 0)
    alias_total_full = int_value(summary.get("unified_total_full_commit_token_count"), total_full)
    alias_total_partial = int_value(summary.get("unified_total_partial_recovered_token_count"), total_partial)
    alias_total_output = int_value(summary.get("unified_total_output_token_count"), total_output)
    if alias_total_output != int_value(summary["combined_real_committed_token_count"], 0):
        errors.append("unified total output must equal combined real committed token count")
    if alias_total_partial != int_value(summary["partial_prefix_total_recovered_token_count"], 0):
        errors.append("unified partial recovered total must match partial-prefix total")
    if total_revised != int_value(summary["partial_prefix_revised_token_count"], 0):
        errors.append("unified revised total must match partial-prefix revised token count")
    if alias_total_partial:
        accepted = int_value(summary["partial_prefix_accepted_token_count"], 0)
        if alias_total_partial != accepted + total_revised:
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
    if int_value(summary.get("normal_proposal_buffer_illegal_discard_count"), 0) > 0:
        errors.append(
            "normal proposal buffer had illegal discard order violations: "
            f"{summary.get('normal_proposal_buffer_event_order_violation_examples', [])}"
        )
    if alias_total_output != alias_total_full + alias_total_partial:
        errors.append("unified total output must equal full commits plus partial recovered tokens")
    if sum_depth_values(summary.get("unified_full_commit_token_count_by_depth")) != alias_total_full:
        errors.append("full-only depth token counts must sum to total full commit tokens")
    if sum_depth_values(summary.get("unified_total_output_token_count_by_depth")) != alias_total_output:
        errors.append("depth total output token counts must sum to total output tokens")
    return errors, summary


def synthetic_payload(total_output_tokens: int) -> dict[str, Any]:
    return {
        "args": {
            "enable_unified_generic_rolling_runtime": True,
            "enable_full_continuous_eager": True,
            "enable_generic_rolling_runtime_loop": True,
            "enable_generic_rolling_apply_path": True,
            "max_rolling_continuous_depth": 6,
            "unified_generic_max_unverified_depth_ahead": 0,
        },
        "metrics": {"total_output_tokens": int(total_output_tokens)},
    }


def add_synthetic_temp_append_trace(
    record: dict[str, Any],
    proposal_tokens_by_id: dict[int, list[int]],
    *,
    seq_len_base: int = 32,
    sample_limit: int = 8,
) -> None:
    proposal_ids = sorted(int(proposal_id) for proposal_id in proposal_tokens_by_id)
    sampled_ids = proposal_ids[:sample_limit]
    depth_by_id = as_int_map(record.get("generic_rolling_depth_by_proposal_id")) or {
        proposal_id: 1 for proposal_id in proposal_ids
    }
    before = {proposal_id: seq_len_base + index * 4 for index, proposal_id in enumerate(proposal_ids)}
    after = {proposal_id: before[proposal_id] + 4 for proposal_id in proposal_ids}
    record["unified_generic_target_verify_temp_append_used"] = True
    record["unified_generic_target_verify_seq_len_before_temp_append_by_proposal_id"] = {
        str(proposal_id): int(before[proposal_id]) for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_seq_len_after_temp_append_by_proposal_id"] = {
        str(proposal_id): int(after[proposal_id]) for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_seq_len_after_rollback_by_proposal_id"] = {
        str(proposal_id): int(before[proposal_id]) for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_checkpoint_restored_by_proposal_id"] = {
        str(proposal_id): True for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_input_ids_shape"] = [4 * len(proposal_ids)]
    record["unified_generic_target_verify_logits_shape"] = [4 * len(proposal_ids), 32000]
    record["unified_generic_target_verify_num_proposals"] = len(proposal_ids)
    record["unified_generic_target_verify_num_to_verify_tokens"] = 4 * len(proposal_ids)
    record["unified_generic_target_verify_logits_rows_per_proposal"] = 4
    record["unified_generic_target_verify_input_ids_by_proposal_id"] = {
        str(proposal_id): list(proposal_tokens_by_id[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_positions_by_proposal_id"] = {
        str(proposal_id): list(range(after[proposal_id] - 4, after[proposal_id]))
        for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_context_lens_by_proposal_id"] = {
        str(proposal_id): list(range(after[proposal_id] - 3, after[proposal_id] + 1))
        for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_next_round_input_by_proposal_id"] = {
        str(proposal_id): list(proposal_tokens_by_id[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_legacy_to_be_verified_by_proposal_id"] = {
        str(proposal_id): [900 + index, 901 + index, 902 + index, proposal_tokens_by_id[proposal_id][0]]
        for index, proposal_id in enumerate(sampled_ids)
    }
    record["unified_generic_target_verify_original_proposal_token_ids_by_proposal_id"] = {
        str(proposal_id): list(proposal_tokens_by_id[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_input_equals_next_round_by_proposal_id"] = {
        str(proposal_id): True for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_next_round_equals_original_proposal_by_proposal_id"] = {
        str(proposal_id): True for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_pre_verify_before_by_proposal_id"] = {
        str(proposal_id): False for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_pre_verify_after_rollback_by_proposal_id"] = {
        str(proposal_id): False for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_base_len_by_proposal_id"] = {
        str(proposal_id): int(before[proposal_id]) for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_parent_proposal_id_by_proposal_id"] = {
        str(proposal_id): -1 if proposal_id == proposal_ids[0] else proposal_id - 1
        for proposal_id in proposal_ids
    }
    current_to_verify = {
        proposal_id: [900 + index, 901 + index, 902 + index, proposal_tokens_by_id[proposal_id][0]]
        for index, proposal_id in enumerate(proposal_ids)
    }
    current_accept_by_id = {proposal_id: 4 for proposal_id in proposal_ids}
    proposal_accept_by_id = {proposal_id: 4 for proposal_id in proposal_ids}
    current_hist: dict[str, dict[str, int]] = {}
    proposal_hist: dict[str, dict[str, int]] = {}
    current_mapping_hist: dict[str, dict[str, int]] = {}
    for proposal_id in proposal_ids:
        depth = str(int(depth_by_id.get(proposal_id, 1)))
        current_hist.setdefault(depth, {})
        proposal_hist.setdefault(depth, {})
        current_mapping_hist.setdefault(depth, {})
        current_hist[depth]["4"] = int(current_hist[depth].get("4", 0)) + 1
        proposal_hist[depth]["4"] = int(proposal_hist[depth].get("4", 0)) + 1
        current_mapping_hist[depth]["0"] = int(current_mapping_hist[depth].get("0", 0)) + 1
    record["unified_generic_proposal_window_verify_enabled"] = False
    record["unified_generic_target_verify_logits_owner"] = True
    record["unified_generic_target_verify_appended_logits_available"] = True
    record["unified_generic_target_verify_frontier_logits_none_allowed"] = False
    record["unified_generic_target_verify_rank"] = 1
    record["unified_generic_target_verify_tp_local_rank"] = 0
    record["unified_generic_target_verify_target_master_rank"] = 1
    record["unified_generic_target_verify_current_to_be_verified_by_proposal_id"] = {
        str(proposal_id): list(current_to_verify[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_proposal_window_to_be_verified_by_proposal_id"] = {
        str(proposal_id): list(proposal_tokens_by_id[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_current_window_equals_proposal_by_proposal_id"] = {
        str(proposal_id): current_to_verify[proposal_id] == proposal_tokens_by_id[proposal_id]
        for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_proposal_window_equals_input_by_proposal_id"] = {
        str(proposal_id): True for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_current_window_accepted_len_by_proposal_id"] = {
        str(proposal_id): int(current_accept_by_id[proposal_id]) for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_proposal_window_shadow_accepted_len_by_proposal_id"] = {
        str(proposal_id): int(proposal_accept_by_id[proposal_id]) for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_current_window_accept_hist_by_depth"] = current_hist
    record["unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth"] = proposal_hist
    record["unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept"] = True
    record["unified_generic_target_verify_uses_shifted_logits"] = True
    record["unified_generic_target_verify_frontier_logits_available"] = True
    record["unified_generic_target_verify_frontier_checkpoint_restored_by_proposal_id"] = {
        str(proposal_id): True for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_frontier_block_table_match_by_proposal_id"] = {
        str(proposal_id): True for proposal_id in proposal_ids
    }
    record["unified_generic_target_verify_current_mapping_accept_hist_by_depth"] = current_mapping_hist
    record["unified_generic_target_verify_shifted_mapping_accept_hist_by_depth"] = proposal_hist
    record["unified_raw_current_verify_token_ids_by_proposal_id"] = {
        str(proposal_id): list(proposal_tokens_by_id[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_raw_precondition_token_ids_by_proposal_id"] = {
        str(proposal_id): list(current_to_verify[proposal_id]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_current_window_first_token_prob_by_proposal_id"] = {
        str(proposal_id): 0.05 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_proposal_window_first_token_prob_by_proposal_id"] = {
        str(proposal_id): 0.85 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_current_window_first_token_id_by_proposal_id"] = {
        str(proposal_id): int(current_to_verify[proposal_id][0]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_proposal_window_first_token_id_by_proposal_id"] = {
        str(proposal_id): int(proposal_tokens_by_id[proposal_id][0]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_first_position_target_argmax_token_by_proposal_id"] = {
        str(proposal_id): int(proposal_tokens_by_id[proposal_id][0]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_first_position_target_top5_tokens_by_proposal_id"] = {
        str(proposal_id): [
            int(proposal_tokens_by_id[proposal_id][0]),
            int(proposal_tokens_by_id[proposal_id][1]),
            int(current_to_verify[proposal_id][0]),
            42,
            43,
        ]
        for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_first_position_target_top5_probs_by_proposal_id"] = {
        str(proposal_id): [0.85, 0.05, 0.03, 0.02, 0.01] for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_frontier_top_token_by_proposal_id"] = {
        str(proposal_id): int(proposal_tokens_by_id[proposal_id][0]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_frontier_top_prob_by_proposal_id"] = {
        str(proposal_id): 0.85 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_frontier_prob_for_p0_by_proposal_id"] = {
        str(proposal_id): 0.85 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_appended_row0_top_token_by_proposal_id"] = {
        str(proposal_id): int(proposal_tokens_by_id[proposal_id][1]) for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_appended_row0_top_prob_by_proposal_id"] = {
        str(proposal_id): 0.80 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_appended_row0_prob_for_p0_by_proposal_id"] = {
        str(proposal_id): 0.01 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_appended_row0_prob_for_p1_by_proposal_id"] = {
        str(proposal_id): 0.80 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_shifted_prob_for_p0_by_proposal_id"] = {
        str(proposal_id): 0.85 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_shifted_prob_for_p1_by_proposal_id"] = {
        str(proposal_id): 0.80 for proposal_id in sampled_ids
    }
    record["unified_generic_target_verify_shifted_accepted_len_by_proposal_id"] = {
        str(proposal_id): int(proposal_accept_by_id[proposal_id]) for proposal_id in sampled_ids
    }


def synthetic_records() -> list[dict[str, Any]]:
    ids_by_depth = {depth: [1000 + depth] for depth in range(1, 7)}
    parent_by_id = {1000 + depth: 999 + depth for depth in range(2, 7)}
    depth_by_id = {1000 + depth: depth for depth in range(1, 7)}
    token_by_id = {1000 + depth: 4 for depth in range(1, 7)}
    proposal_tokens_by_id = {
        1000 + depth: [depth * 10 + offset for offset in range(4)]
        for depth in range(1, 7)
    }
    full = sum(token_by_id.values())
    partial = 3
    revised = 1
    record = {
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
            "generic_rolling_proposal_token_ids_by_proposal_id": {
                str(k): list(v) for k, v in proposal_tokens_by_id.items()
            },
            "generic_rolling_to_be_verified_token_ids_by_proposal_id": {
                str(k): list(v) for k, v in proposal_tokens_by_id.items()
            },
            "generic_rolling_frontier_tail_token_ids_by_proposal_id": {
                str(k): [max(0, token_id - 4) for token_id in v]
                for k, v in proposal_tokens_by_id.items()
            },
            "generic_rolling_to_verify_equals_proposal_by_proposal_id": {
                str(k): True for k in proposal_tokens_by_id
            },
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
            "unified_raw_target_verification_available": True,
            "unified_raw_verification_source": "target_verify_result",
            "unified_raw_verified_proposal_count_by_depth": {
                str(depth): 1 for depth in ids_by_depth
            },
            "unified_raw_full_accept_proposal_count_by_depth": {
                str(depth): 1 for depth in ids_by_depth
            },
            "unified_raw_partial_accept_proposal_count_by_depth": {},
            "unified_raw_reject_proposal_count_by_depth": {},
            "unified_raw_invalidated_proposal_count_by_depth": {},
            "unified_raw_accepted_len_hist_by_depth": {
                str(depth): {"4": 1} for depth in ids_by_depth
            },
            "unified_raw_revised_token_count_by_depth": {},
            "unified_raw_partial_recovery_eligible_count_by_depth": {},
            "unified_raw_partial_recovery_ineligible_reason_counts_by_depth": {},
            "unified_raw_candidate_proposal_count_by_depth": {
                str(depth): 1 for depth in ids_by_depth
            },
            "unified_raw_committed_proposal_count_by_depth": {
                str(depth): 1 for depth in ids_by_depth
            },
            "unified_raw_full_commit_proposal_count_by_depth": {
                str(depth): 1 for depth in ids_by_depth
            },
            "unified_raw_partial_recovery_applied_proposal_count_by_depth": {},
            "unified_raw_reject_revised_correction_applied_proposal_count_by_depth": {},
            "unified_raw_no_mutation_reject_proposal_count_by_depth": {},
            "generic_full_continuous_stop_reason_counts": {"max_depth_reached": 1},
            "generic_full_continuous_normal_lane_conflict_count": 0,
            "generic_full_continuous_target_draft_mismatch_count": 0,
            "generic_full_continuous_depth_gt_max_real_commit_count": 0,
            "generic_full_continuous_parity_ok": True,
            "unified_generic_normal_lane_conflict_count": 0,
            "unified_generic_target_draft_mismatch_count": 0,
            "unified_generic_parity_ok": True,
        }
    add_synthetic_temp_append_trace(record, proposal_tokens_by_id)
    return [record]


def synthetic_reject_partial_records() -> list[dict[str, Any]]:
    proposal_ids = list(range(3001, 3181))
    partial_id = proposal_ids[-1]
    token_by_id = {proposal_id: 4 for proposal_id in proposal_ids}
    proposal_tokens_by_id = {
        proposal_id: [proposal_id, proposal_id + 1, proposal_id + 2, proposal_id + 3]
        for proposal_id in proposal_ids
    }
    record = {
            "normal_gamma": 4,
            "unified_generic_rolling_enabled": True,
            "enable_unified_generic_rolling_runtime": True,
            "generic_full_continuous_enabled": True,
            "enable_full_continuous_eager": True,
            "generic_rolling_runtime_enabled": True,
            "enable_generic_rolling_runtime_loop": True,
            "generic_rolling_apply_path_enabled": True,
            "enable_generic_rolling_apply_path": True,
            "unified_generic_max_depth": 8,
            "unified_generic_max_observed_depth": 1,
            "unified_generic_max_real_committed_depth": 0,
            "generic_full_continuous_max_depth": 8,
            "generic_full_continuous_max_observed_depth": 1,
            "generic_full_continuous_max_real_committed_depth": 0,
            "generic_rolling_candidate_proposal_ids_by_depth": {"1": proposal_ids},
            "generic_rolling_candidate_seq_ids_by_depth": {"1": list(range(4001, 4181))},
            "generic_rolling_ready_proposal_ids_by_depth": {"1": proposal_ids},
            "generic_rolling_ready_seq_ids_by_depth": {"1": list(range(4001, 4181))},
            "generic_rolling_real_committed_proposal_ids_by_depth": {},
            "generic_rolling_real_committed_seq_ids_by_depth": {},
            "generic_rolling_depth_by_proposal_id": {str(proposal_id): 1 for proposal_id in proposal_ids},
            "generic_rolling_token_count_by_proposal_id": {
                str(proposal_id): token_count for proposal_id, token_count in token_by_id.items()
            },
            "generic_full_continuous_depth_candidate_token_counts": {"1": 720},
            "generic_full_continuous_depth_ready_token_counts": {"1": 720},
            "unified_generic_depth_candidate_token_counts": {"1": 720},
            "unified_generic_depth_ready_token_counts": {"1": 720},
            "generic_full_continuous_total_full_commit_token_count": 0,
            "generic_full_continuous_total_partial_recovered_token_count": 2,
            "generic_full_continuous_total_revised_token_count": 1,
            "generic_full_continuous_total_output_token_count": 2,
            "unified_generic_total_full_commit_token_count": 0,
            "unified_generic_total_partial_recovered_token_count": 2,
            "unified_generic_total_revised_token_count": 1,
            "unified_generic_total_output_token_count": 2,
            "combined_real_committed_token_count": 2,
            "partial_prefix_recovery_enabled": True,
            "partial_prefix_recovered_proposal_ids": [partial_id],
            "partial_prefix_recovered_seq_ids": [4180],
            "partial_prefix_recovered_depth_by_proposal_id": {str(partial_id): 1},
            "partial_prefix_accepted_len_by_proposal_id": {str(partial_id): 1},
            "partial_prefix_revised_token_count_by_proposal_id": {str(partial_id): 1},
            "partial_prefix_committed_token_count_by_proposal_id": {str(partial_id): 2},
            "generic_full_continuous_depth_partial_recovered_token_counts": {"1": 2},
            "generic_full_continuous_depth_revised_token_counts": {"1": 1},
            "unified_generic_depth_partial_recovered_token_counts": {"1": 2},
            "unified_generic_depth_revised_token_counts": {"1": 1},
            "unified_raw_target_verification_available": True,
            "unified_raw_verification_source": "target_verify_result",
            "unified_raw_verified_proposal_count_by_depth": {"1": 180},
            "unified_raw_full_accept_proposal_count_by_depth": {},
            "unified_raw_partial_accept_proposal_count_by_depth": {"1": 1},
            "unified_raw_reject_proposal_count_by_depth": {"1": 179},
            "unified_raw_invalidated_proposal_count_by_depth": {},
            "unified_raw_accepted_len_hist_by_depth": {"1": {"0": 179, "1": 1}},
            "unified_raw_revised_token_count_by_depth": {"1": 180},
            "unified_raw_partial_recovery_eligible_count_by_depth": {"1": 1},
            "unified_raw_partial_recovery_ineligible_reason_counts_by_depth": {
                "1": {"partial_recovery_not_selected": 179}
            },
            "unified_raw_candidate_proposal_count_by_depth": {"1": 180},
            "unified_raw_committed_proposal_count_by_depth": {"1": 1},
            "unified_raw_full_commit_proposal_count_by_depth": {},
            "unified_raw_partial_recovery_applied_proposal_count_by_depth": {"1": 1},
            "unified_raw_reject_revised_correction_applied_proposal_count_by_depth": {},
            "unified_raw_no_mutation_reject_proposal_count_by_depth": {"1": 179},
            "generic_full_continuous_stop_reason_counts": {"parent_not_full_accept": 179},
            "generic_full_continuous_normal_lane_conflict_count": 0,
            "generic_full_continuous_target_draft_mismatch_count": 0,
            "generic_full_continuous_depth_gt_max_real_commit_count": 0,
            "generic_full_continuous_parity_ok": True,
            "unified_generic_normal_lane_conflict_count": 0,
            "unified_generic_target_draft_mismatch_count": 0,
            "unified_generic_parity_ok": True,
        }
    add_synthetic_temp_append_trace(record, proposal_tokens_by_id)
    shifted_accept_by_id = {proposal_id: 0 for proposal_id in proposal_ids}
    shifted_accept_by_id[partial_id] = 1
    shifted_accept_hist = {"1": {"0": len(proposal_ids) - 1, "1": 1}}
    record["unified_generic_target_verify_shifted_mapping_accept_hist_by_depth"] = {
        depth: dict(hist) for depth, hist in shifted_accept_hist.items()
    }
    record["unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth"] = {
        depth: dict(hist) for depth, hist in shifted_accept_hist.items()
    }
    record["unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept"] = True
    record["unified_generic_target_verify_proposal_window_shadow_accepted_len_by_proposal_id"] = {
        str(proposal_id): int(accepted_len) for proposal_id, accepted_len in shifted_accept_by_id.items()
    }
    record["unified_generic_target_verify_shifted_accepted_len_by_proposal_id"] = {
        str(proposal_id): int(accepted_len) for proposal_id, accepted_len in shifted_accept_by_id.items()
    }
    return [record]


def synthetic_single_child_payload(*, max_depth: int = 8, max_unverified_ahead: int = 1) -> dict[str, Any]:
    payload = synthetic_payload(total_output_tokens=0)
    payload["args"]["max_rolling_continuous_depth"] = int(max_depth)
    payload["args"]["unified_generic_max_unverified_depth_ahead"] = int(max_unverified_ahead)
    return payload


def synthetic_single_child_records(
    steps: list[list[int]],
    *,
    max_depth: int = 8,
    max_unverified_ahead: int = 1,
    committed_depths: set[int] | None = None,
    stop_reason: str = "target_verify_pending",
    parent_outcomes_by_depth: dict[int, str] | None = None,
    invalidated_parent_not_full_depths: set[int] | None = None,
    full_accept_without_child_reasons_by_depth: dict[int, str] | None = None,
    inflight_child_depths: set[int] | None = None,
    promoted_child_depths: set[int] | None = None,
    scheduled_child_depths: set[int] | None = None,
    target_verified_after_promotion_depths: set[int] | None = None,
    invalidated_after_non_full_child_depths: set[int] | None = None,
    ready_not_scheduled_reasons_by_depth: dict[int, str] | None = None,
    verified_before_parent_full_accept_violation: bool = False,
) -> list[dict[str, Any]]:
    committed_depths = set(committed_depths or set())
    parent_outcomes_by_depth = dict(parent_outcomes_by_depth or {})
    invalidated_parent_not_full_depths = set(invalidated_parent_not_full_depths or set())
    full_accept_without_child_reasons_by_depth = dict(full_accept_without_child_reasons_by_depth or {})
    inflight_child_depths = set(inflight_child_depths or set())
    promoted_child_depths = set(promoted_child_depths or set())
    scheduled_child_depths = set(scheduled_child_depths or set())
    target_verified_after_promotion_depths = set(target_verified_after_promotion_depths or set())
    invalidated_after_non_full_child_depths = set(invalidated_after_non_full_child_depths or set())
    ready_not_scheduled_reasons_by_depth = dict(ready_not_scheduled_reasons_by_depth or {})
    target_inflight_depths = {int(depth) - 1 for depth in inflight_child_depths if int(depth) > 1}
    gamma = 4
    proposal_id_by_depth = {depth: 9000 + depth for depths in steps for depth in depths}
    if committed_depths:
        for depth in committed_depths:
            proposal_id_by_depth.setdefault(int(depth), 9000 + int(depth))
    root_id = proposal_id_by_depth.get(1, 9001)
    parent_by_id = {
        proposal_id_by_depth[depth]: proposal_id_by_depth[depth - 1]
        for depth in proposal_id_by_depth
        if depth > 1 and (depth - 1) in proposal_id_by_depth
    }
    depth_by_id = {proposal_id: depth for depth, proposal_id in proposal_id_by_depth.items()}
    total_full = int(len(committed_depths) * gamma)
    records: list[dict[str, Any]] = []
    registry_ids = set(proposal_id_by_depth.values())
    registry_seq_by_id = {proposal_id: 7 for proposal_id in registry_ids}
    registry_root_by_id = {proposal_id: int(root_id) for proposal_id in registry_ids}
    registry_depth_by_id = {proposal_id: int(depth_by_id[proposal_id]) for proposal_id in registry_ids}
    registry_parent_by_id = {proposal_id: int(parent_by_id.get(proposal_id, -1)) for proposal_id in registry_ids}
    registry_created_step_by_id = {
        proposal_id: max(0, int(depth_by_id[proposal_id]) - 1) for proposal_id in registry_ids
    }
    registry_verified_step_by_id: dict[int, int] = {}
    registry_accepted_by_id = {proposal_id: -1 for proposal_id in registry_ids}
    registry_proposal_len_by_id = {proposal_id: gamma for proposal_id in registry_ids}
    registry_base_len_by_id = {proposal_id: 100 + int(depth_by_id[proposal_id] - 1) * gamma for proposal_id in registry_ids}
    registry_output_by_id = {proposal_id: 0 for proposal_id in registry_ids}
    registry_full_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_partial_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_reject_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_invalidated_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_applied_full_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_applied_partial_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_no_mutation_reject_by_id = {proposal_id: False for proposal_id in registry_ids}
    registry_target_inflight_by_id = {
        proposal_id: int(depth_by_id[proposal_id]) in target_inflight_depths
        for proposal_id in registry_ids
    }
    registry_invalidation_reason_by_id = {proposal_id: "" for proposal_id in registry_ids}
    for depth, proposal_id in proposal_id_by_depth.items():
        outcome = parent_outcomes_by_depth.get(int(depth))
        if outcome is None and int(depth) in committed_depths:
            outcome = "full"
        if outcome is None:
            continue
        if outcome == "missing":
            for mapping in (
                registry_seq_by_id,
                registry_root_by_id,
                registry_depth_by_id,
                registry_parent_by_id,
                registry_created_step_by_id,
                registry_accepted_by_id,
                registry_proposal_len_by_id,
                registry_base_len_by_id,
                registry_output_by_id,
                registry_full_by_id,
                registry_partial_by_id,
                registry_reject_by_id,
                registry_invalidated_by_id,
                registry_applied_full_by_id,
                registry_applied_partial_by_id,
                registry_no_mutation_reject_by_id,
                registry_target_inflight_by_id,
                registry_invalidation_reason_by_id,
            ):
                mapping.pop(proposal_id, None)
            continue
        registry_verified_step_by_id[proposal_id] = max(0, int(depth))
        if outcome == "full":
            registry_accepted_by_id[proposal_id] = gamma
            registry_output_by_id[proposal_id] = gamma
            registry_full_by_id[proposal_id] = True
            registry_applied_full_by_id[proposal_id] = True
        elif outcome in {"partial", "partial_recovery"}:
            registry_accepted_by_id[proposal_id] = 2
            registry_output_by_id[proposal_id] = 3 if outcome == "partial_recovery" else 0
            registry_partial_by_id[proposal_id] = True
            registry_applied_partial_by_id[proposal_id] = outcome == "partial_recovery"
        elif outcome == "reject":
            registry_accepted_by_id[proposal_id] = 0
            registry_reject_by_id[proposal_id] = True
            registry_no_mutation_reject_by_id[proposal_id] = True
        elif outcome == "invalidated":
            registry_accepted_by_id[proposal_id] = 0
            registry_invalidated_by_id[proposal_id] = True
            registry_invalidation_reason_by_id[proposal_id] = "invalidated_ancestor"
        elif outcome == "inflight":
            registry_target_inflight_by_id[proposal_id] = True

    for step_index, depths in enumerate(steps):
        candidate_by_depth = {str(depth): [proposal_id_by_depth[depth]] for depth in depths}
        ready_depths = [
            depth
            for depth in depths
            if int(depth) not in inflight_child_depths or int(depth) in promoted_child_depths
        ]
        ready_by_depth = {str(depth): [proposal_id_by_depth[depth]] for depth in ready_depths}
        committed_by_depth = {
            str(depth): [proposal_id_by_depth[depth]]
            for depth in depths
            if int(depth) in committed_depths
        }
        ids_in_record = [proposal_id_by_depth[depth] for depth in depths]
        committed_ids = [proposal_id_by_depth[depth] for depth in depths if int(depth) in committed_depths]
        token_by_id = {proposal_id: gamma for proposal_id in ids_in_record}
        token_by_id.update({proposal_id_by_depth[depth]: gamma for depth in committed_depths})
        source_step_by_id = {proposal_id: step_index for proposal_id in ids_in_record}
        record = {
            "step_id": step_index,
            "plan_id": 1,
            "normal_gamma": gamma,
            "unified_generic_rolling_enabled": True,
            "enable_unified_generic_rolling_runtime": True,
            "generic_full_continuous_enabled": True,
            "enable_full_continuous_eager": True,
            "generic_rolling_runtime_enabled": True,
            "enable_generic_rolling_runtime_loop": True,
            "generic_rolling_apply_path_enabled": True,
            "enable_generic_rolling_apply_path": True,
            "unified_single_child_ahead_enabled": bool(max_unverified_ahead == 1),
            "unified_max_unverified_depth_ahead": int(max_unverified_ahead),
            "unified_generic_max_depth": int(max_depth),
            "unified_generic_max_observed_depth": max(depths or [0]),
            "unified_generic_max_real_committed_depth": max(committed_depths or {0}),
            "generic_full_continuous_max_depth": int(max_depth),
            "generic_full_continuous_max_observed_depth": max(depths or [0]),
            "generic_full_continuous_max_real_committed_depth": max(committed_depths or {0}),
            "generic_rolling_candidate_proposal_ids_by_depth": candidate_by_depth,
            "generic_rolling_candidate_seq_ids_by_depth": {str(depth): [7] for depth in depths},
            "generic_rolling_ready_proposal_ids_by_depth": ready_by_depth,
            "generic_rolling_ready_seq_ids_by_depth": {str(depth): [7] for depth in ready_depths},
            "generic_rolling_real_committed_proposal_ids_by_depth": committed_by_depth,
            "generic_rolling_real_committed_seq_ids_by_depth": {
                str(depth): [7] for depth in depths if int(depth) in committed_depths
            },
            "generic_rolling_parent_by_proposal_id": {
                str(proposal_id): parent_id for proposal_id, parent_id in parent_by_id.items()
            },
            "generic_rolling_real_commit_parent_by_proposal_id": {
                str(proposal_id): parent_id
                for proposal_id, parent_id in parent_by_id.items()
                if proposal_id in committed_ids
            },
            "generic_rolling_root_by_proposal_id": {
                str(proposal_id): int(root_id) for proposal_id in proposal_id_by_depth.values()
            },
            "generic_rolling_real_commit_root_by_proposal_id": {
                str(proposal_id): int(root_id) for proposal_id in committed_ids
            },
            "generic_rolling_depth_by_proposal_id": {
                str(proposal_id): depth for proposal_id, depth in depth_by_id.items()
            },
            "generic_rolling_real_commit_depth_by_proposal_id": {
                str(proposal_id): depth_by_id[proposal_id] for proposal_id in committed_ids
            },
            "generic_rolling_source_dual_step_id_by_proposal_id": {
                str(proposal_id): step for proposal_id, step in source_step_by_id.items()
            },
            "generic_rolling_token_count_by_proposal_id": {
                str(proposal_id): token_count for proposal_id, token_count in token_by_id.items()
            },
            "generic_rolling_proposal_token_ids_by_proposal_id": {
                str(proposal_id): [proposal_id, proposal_id + 1, proposal_id + 2, proposal_id + 3]
                for proposal_id in token_by_id
            },
            "generic_rolling_to_be_verified_token_ids_by_proposal_id": {
                str(proposal_id): [proposal_id, proposal_id + 1, proposal_id + 2, proposal_id + 3]
                for proposal_id in token_by_id
            },
            "generic_rolling_to_verify_equals_proposal_by_proposal_id": {
                str(proposal_id): True for proposal_id in token_by_id
            },
            "generic_rolling_real_committed_token_count_by_proposal_id": {
                str(proposal_id): gamma for proposal_id in committed_ids
            },
            "generic_rolling_real_committed_accept_len_by_proposal_id": {
                str(proposal_id): gamma for proposal_id in committed_ids
            },
            "generic_rolling_real_commit_action_by_proposal_id": {
                str(proposal_id): "append_full_accept_real_commit" for proposal_id in committed_ids
            },
            "generic_rolling_real_commit_verify_result_by_proposal_id": {
                str(proposal_id): "full_accept" for proposal_id in committed_ids
            },
            "generic_rolling_real_committed_token_count_by_depth": {
                str(depth): gamma for depth in depths if int(depth) in committed_depths
            },
            "generic_rolling_real_committed_proposal_count_by_depth": {
                str(depth): 1 for depth in depths if int(depth) in committed_depths
            },
            "generic_full_continuous_depth_candidate_token_counts": {
                str(depth): gamma for depth in depths
            },
            "generic_full_continuous_depth_ready_token_counts": {str(depth): gamma for depth in ready_depths},
            "generic_full_continuous_depth_commit_token_counts": {
                str(depth): gamma for depth in committed_depths
            },
            "unified_generic_depth_candidate_token_counts": {str(depth): gamma for depth in depths},
            "unified_generic_depth_ready_token_counts": {str(depth): gamma for depth in ready_depths},
            "unified_generic_depth_commit_token_counts": {
                str(depth): gamma for depth in committed_depths
            },
            "generic_full_continuous_total_full_commit_token_count": total_full,
            "generic_full_continuous_total_partial_recovered_token_count": 0,
            "generic_full_continuous_total_revised_token_count": 0,
            "generic_full_continuous_total_output_token_count": total_full,
            "unified_generic_total_full_commit_token_count": total_full,
            "unified_generic_total_partial_recovered_token_count": 0,
            "unified_generic_total_revised_token_count": 0,
            "unified_generic_total_output_token_count": total_full,
            "combined_real_committed_token_count": total_full,
            "partial_prefix_recovery_enabled": True,
            "partial_prefix_recovered_proposal_ids": [],
            "partial_prefix_accepted_len_by_proposal_id": {},
            "partial_prefix_revised_token_count_by_proposal_id": {},
            "partial_prefix_committed_token_count_by_proposal_id": {},
            "generic_full_continuous_stop_reason_counts": {str(stop_reason): 1},
            "generic_full_continuous_normal_lane_conflict_count": 0,
            "generic_full_continuous_target_draft_mismatch_count": 0,
            "generic_full_continuous_depth_gt_max_real_commit_count": 0,
            "generic_full_continuous_parity_ok": True,
            "unified_generic_normal_lane_conflict_count": 0,
            "unified_generic_target_draft_mismatch_count": 0,
            "unified_generic_parity_ok": True,
            "unified_single_child_parent_full_accept_guard_enabled": bool(max_unverified_ahead == 1),
            "unified_proposal_registry_seq_id_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_seq_by_id.items()
            },
            "unified_proposal_registry_root_id_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_root_by_id.items()
            },
            "unified_proposal_registry_depth_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_depth_by_id.items()
            },
            "unified_proposal_registry_parent_proposal_id_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_parent_by_id.items()
            },
            "unified_proposal_registry_created_step_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_created_step_by_id.items()
            },
            "unified_proposal_registry_verified_step_by_proposal_id": {
                str(proposal_id): registry_verified_step_by_id.get(proposal_id, -1)
                for proposal_id in registry_seq_by_id
            },
            "unified_proposal_registry_accepted_len_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_accepted_by_id.items()
            },
            "unified_proposal_registry_proposal_len_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_proposal_len_by_id.items()
            },
            "unified_proposal_registry_base_len_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_base_len_by_id.items()
            },
            "unified_proposal_registry_output_tokens_committed_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_output_by_id.items()
            },
            "unified_proposal_registry_full_accept_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_full_by_id.items()
            },
            "unified_proposal_registry_partial_accept_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_partial_by_id.items()
            },
            "unified_proposal_registry_reject_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_reject_by_id.items()
            },
            "unified_proposal_registry_invalidated_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_invalidated_by_id.items()
            },
            "unified_proposal_registry_applied_full_commit_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_applied_full_by_id.items()
            },
            "unified_proposal_registry_applied_partial_recovery_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_applied_partial_by_id.items()
            },
            "unified_proposal_registry_no_mutation_reject_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_no_mutation_reject_by_id.items()
            },
            "unified_proposal_registry_target_verify_inflight_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_target_inflight_by_id.items()
            },
            "unified_proposal_registry_invalidation_reason_by_proposal_id": {
                str(proposal_id): value for proposal_id, value in registry_invalidation_reason_by_id.items()
            },
        }
        child_depths = [int(depth) for depth in depths if int(depth) > 1 and int(depth) in proposal_id_by_depth]
        if child_depths:
            child_ids = [proposal_id_by_depth[depth] for depth in child_depths]
            parent_ids = {child_id: parent_by_id[child_id] for child_id in child_ids}
            record["unified_child_generation_depth_by_proposal_id"] = {
                str(child_id): depth_by_id[child_id] for child_id in child_ids
            }
            record["unified_child_generation_seq_id_by_proposal_id"] = {str(child_id): 7 for child_id in child_ids}
            record["unified_child_generation_root_id_by_proposal_id"] = {
                str(child_id): int(root_id) for child_id in child_ids
            }
            record["unified_child_generation_created_step_by_proposal_id"] = {
                str(child_id): step_index for child_id in child_ids
            }
            record["unified_child_generation_parent_proposal_id_by_proposal_id"] = {
                str(child_id): parent_id for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_lookup_found_by_proposal_id"] = {
                str(child_id): parent_id in registry_depth_by_id for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_depth_by_proposal_id"] = {
                str(child_id): registry_depth_by_id.get(parent_id, -1) for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_created_step_by_proposal_id"] = {
                str(child_id): registry_created_step_by_id.get(parent_id, -1)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_verified_step_by_proposal_id"] = {
                str(child_id): registry_verified_step_by_id.get(parent_id, -1)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_accepted_len_by_proposal_id"] = {
                str(child_id): registry_accepted_by_id.get(parent_id, -1)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_proposal_len_by_proposal_id"] = {
                str(child_id): registry_proposal_len_by_id.get(parent_id, gamma)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_full_accept_by_proposal_id"] = {
                str(child_id): registry_full_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_partial_accept_by_proposal_id"] = {
                str(child_id): registry_partial_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_reject_by_proposal_id"] = {
                str(child_id): registry_reject_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_invalidated_by_proposal_id"] = {
                str(child_id): registry_invalidated_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_applied_full_commit_by_proposal_id"] = {
                str(child_id): registry_applied_full_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_applied_partial_recovery_by_proposal_id"] = {
                str(child_id): registry_applied_partial_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_no_mutation_reject_by_proposal_id"] = {
                str(child_id): registry_no_mutation_reject_by_id.get(parent_id, False)
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_mode_by_proposal_id"] = {
                str(child_id): (
                    "inflight_parent_speculative"
                    if depth_by_id[child_id] in inflight_child_depths
                    else ""
                )
                for child_id in child_ids
            }
            record["unified_child_generated_before_parent_result_by_proposal_id"] = {
                str(child_id): depth_by_id[child_id] in inflight_child_depths
                for child_id in child_ids
            }
            record["unified_child_pending_parent_result_by_proposal_id"] = {
                str(child_id): (
                    depth_by_id[child_id] in inflight_child_depths
                    and depth_by_id[child_id] not in promoted_child_depths
                    and depth_by_id[child_id] not in invalidated_after_non_full_child_depths
                )
                for child_id in child_ids
            }
            record["unified_child_promoted_after_parent_full_accept_by_proposal_id"] = {
                str(child_id): depth_by_id[child_id] in promoted_child_depths
                for child_id in child_ids
            }
            record["unified_child_invalidated_after_parent_non_full_by_proposal_id"] = {
                str(child_id): depth_by_id[child_id] in invalidated_after_non_full_child_depths
                for child_id in child_ids
            }
            record["unified_child_generation_allowed_by_proposal_id"] = {
                str(child_id): (
                    bool(depth_by_id[child_id] in invalidated_after_non_full_child_depths)
                    if (
                        depth_by_id[child_id] in inflight_child_depths
                        and (
                            registry_partial_by_id.get(parent_id, False)
                            or registry_reject_by_id.get(parent_id, False)
                            or registry_invalidated_by_id.get(parent_id, False)
                            or registry_applied_partial_by_id.get(parent_id, False)
                            or registry_no_mutation_reject_by_id.get(parent_id, False)
                        )
                    )
                    else bool(depth_by_id[child_id] in inflight_child_depths)
                    or bool(
                        parent_id in registry_depth_by_id
                        and registry_full_by_id.get(parent_id, False)
                        and registry_applied_full_by_id.get(parent_id, False)
                        and not registry_invalidated_by_id.get(parent_id, False)
                    )
                )
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_parent_state_by_proposal_id"] = {
                str(child_id): (
                    "target_verify_inflight"
                    if depth_by_id[child_id] in inflight_child_depths
                    else "missing"
                    if parent_id not in registry_depth_by_id
                    else "full_accept_applied"
                    if registry_full_by_id.get(parent_id, False)
                    and registry_applied_full_by_id.get(parent_id, False)
                    and not registry_invalidated_by_id.get(parent_id, False)
                    else "parent_invalidated"
                    if registry_invalidated_by_id.get(parent_id, False)
                    else "parent_partial_accept"
                    if registry_partial_by_id.get(parent_id, False)
                    else "parent_reject"
                    if registry_reject_by_id.get(parent_id, False)
                    else "parent_not_full_accept"
                )
                for child_id, parent_id in parent_ids.items()
            }
            record["unified_child_generation_block_reason_by_proposal_id"] = {
                str(child_id): (
                    ""
                    if record["unified_child_generation_allowed_by_proposal_id"][str(child_id)]
                    else record["unified_child_generation_parent_state_by_proposal_id"][str(child_id)]
                )
                for child_id in child_ids
            }
        inflight_depths = {
            str(depth): 1 for depth in depths if int(depth) in inflight_child_depths
        }
        if inflight_depths:
            record["unified_child_generated_from_inflight_parent_count_by_depth"] = inflight_depths
            record["unified_child_pending_parent_result_count_by_depth"] = inflight_depths
            if 2 in [int(depth) for depth in depths if int(depth) in inflight_child_depths]:
                record["unified_depth2_generated_while_depth1_verifying_count"] = 1
            if 3 in [int(depth) for depth in depths if int(depth) in inflight_child_depths]:
                record["unified_depth3_generated_while_depth2_verifying_count"] = 1
        promoted_depths = {
            str(depth): 1 for depth in depths if int(depth) in promoted_child_depths
        }
        if promoted_depths:
            record["unified_child_promoted_after_parent_full_accept_count_by_depth"] = promoted_depths
            record["unified_child_promoted_to_ready_count_by_depth"] = promoted_depths
            record["unified_child_ready_for_target_verify_count_by_depth"] = promoted_depths
            if 2 in [int(depth) for depth in depths if int(depth) in promoted_child_depths]:
                record["unified_depth2_ready_after_depth1_full_accept_count"] = 1
        scheduled_depths = {
            str(depth): 1 for depth in depths if int(depth) in scheduled_child_depths
        }
        if scheduled_depths:
            record["unified_child_scheduled_for_target_verify_count_by_depth"] = scheduled_depths
            record["unified_child_target_verify_inflight_count_by_depth"] = scheduled_depths
            record["unified_child_scheduled_state_by_depth"] = {
                str(depth): {"READY_TO_VERIFY": 1}
                for depth in depths
                if int(depth) in scheduled_child_depths
            }
            if 2 in [int(depth) for depth in depths if int(depth) in scheduled_child_depths]:
                record["unified_depth2_scheduled_after_depth1_full_accept_count"] = 1
        target_verified_after_promotion = {
            str(depth): 1 for depth in depths if int(depth) in target_verified_after_promotion_depths
        }
        if target_verified_after_promotion:
            record["unified_child_target_verified_after_promotion_count_by_depth"] = (
                target_verified_after_promotion
            )
            record["unified_child_verified_after_parent_full_accept_count_by_depth"] = (
                target_verified_after_promotion
            )
            if 2 in [int(depth) for depth in depths if int(depth) in target_verified_after_promotion_depths]:
                record["unified_depth2_verified_after_depth1_full_accept_count"] = 1
        ready_not_scheduled_depths = {
            str(depth): 1
            for depth in depths
            if int(depth) in ready_not_scheduled_reasons_by_depth
        }
        if ready_not_scheduled_depths:
            record["unified_child_ready_but_not_scheduled_count_by_depth"] = ready_not_scheduled_depths
            record["unified_child_ready_not_scheduled_reason_counts_by_depth"] = {
                str(depth): {str(ready_not_scheduled_reasons_by_depth[int(depth)]): 1}
                for depth in depths
                if int(depth) in ready_not_scheduled_reasons_by_depth
            }
        invalidated_after_non_full_depths = {
            str(depth): 1 for depth in depths if int(depth) in invalidated_after_non_full_child_depths
        }
        if invalidated_after_non_full_depths:
            record["unified_child_invalidated_after_parent_non_full_count_by_depth"] = (
                invalidated_after_non_full_depths
            )
        if verified_before_parent_full_accept_violation:
            record["unified_child_verified_before_parent_full_accept_violation_count"] = 1
            record["unified_child_verified_before_parent_result_count_by_depth"] = {
                str(depth): 1 for depth in depths if int(depth) > 1
            }
        invalidated_depths = {
            str(depth): 1 for depth in depths if int(depth) in invalidated_parent_not_full_depths
        }
        if invalidated_depths:
            record["unified_invalidated_due_to_parent_not_full_accept_count_by_depth"] = invalidated_depths
        full_accept_without_child_reasons = {
            str(depth): {str(reason): 1}
            for depth, reason in full_accept_without_child_reasons_by_depth.items()
            if int(depth) in depths and str(reason)
        }
        if full_accept_without_child_reasons:
            record["unified_full_accept_without_child_reason_counts_by_depth"] = (
                full_accept_without_child_reasons
            )
            record["unified_full_accept_parent_not_selected_reason_counts_by_depth"] = (
                full_accept_without_child_reasons
            )
            if "1" in full_accept_without_child_reasons:
                record["unified_depth2_generation_block_reason_counts"] = dict(
                    full_accept_without_child_reasons["1"]
                )
        records.append(record)
    return records


NON_OWNER_TARGET_VERIFY_DETAIL_PREFIXES = (
    "unified_generic_target_verify_current_window_",
    "unified_generic_target_verify_current_mapping_",
    "unified_generic_target_verify_proposal_window_",
    "unified_generic_target_verify_first_position_",
    "unified_generic_target_verify_frontier_",
    "unified_generic_target_verify_appended_",
    "unified_generic_target_verify_shifted_",
)


def synthetic_target_tp_worker_record(owner_record: dict[str, Any], *, rank: int, bad_raw: bool = False) -> dict[str, Any]:
    worker = json.loads(json.dumps(owner_record))
    for field in list(worker):
        if field.startswith("unified_raw_") and field not in {
            "unified_raw_target_verification_available",
            "unified_raw_verification_source",
        }:
            worker.pop(field, None)
            continue
        if any(field.startswith(prefix) for prefix in NON_OWNER_TARGET_VERIFY_DETAIL_PREFIXES):
            worker.pop(field, None)
    worker["unified_generic_target_verify_logits_owner"] = False
    worker["unified_generic_target_verify_frontier_logits_available"] = False
    worker["unified_generic_target_verify_appended_logits_available"] = False
    worker["unified_generic_target_verify_frontier_logits_none_allowed"] = True
    worker["unified_generic_target_verify_uses_shifted_logits"] = False
    worker["unified_generic_target_verify_rank"] = int(rank)
    worker["unified_generic_target_verify_tp_local_rank"] = int(rank - 1)
    worker["unified_generic_target_verify_target_master_rank"] = 1
    worker["unified_generic_target_verify_logits_shape"] = [0, 0]
    worker["unified_generic_target_verify_logits_rows_per_proposal"] = 0
    if bad_raw:
        worker["unified_raw_verified_proposal_count_by_depth"] = {"1": 1}
        worker["unified_raw_full_accept_proposal_count_by_depth"] = {"1": 1}
        worker["unified_raw_accepted_len_hist_by_depth"] = {"1": {"4": 1}}
    return worker


def run_synthetic_tests() -> None:
    records = synthetic_records()
    payload = synthetic_payload(total_output_tokens=27)
    errors, summary = validate_records(records, payload)
    assert not errors, f"valid unified synthetic failed: {errors}\nsummary={summary}"
    assert summary["combined_real_committed_token_count"] == 27
    assert summary["unified_candidate_seq_count_by_depth"]["1"] == 1
    assert summary["unified_committed_token_count_by_depth"]["6"] == 4
    assert summary["unified_commit_share_by_depth"]["1"] == 1.0
    assert summary["unified_raw_candidate_proposal_count_by_depth"]["1"] == 1
    assert summary["unified_raw_committed_proposal_count_by_depth"]["6"] == 1
    assert summary["unified_raw_verified_to_committed_ratio_by_depth"]["1"] == 1.0
    assert summary["unified_total_full_commit_token_count"] == 24
    assert summary["unified_total_partial_recovered_token_count"] == 3
    assert summary["unified_total_output_token_count"] == 27
    assert summary["unified_full_commit_token_count_by_depth"]["6"] == 4
    assert summary["unified_total_output_token_count_by_depth"]["2"] == 7
    assert summary["num_steps"] == 1
    assert summary["steps_with_any_unified_candidate"] == 1
    assert summary["steps_with_any_unified_commit"] == 1
    assert summary["unified_generic_target_verify_temp_append_used"] is True
    assert summary["unified_generic_target_verify_input_mismatch_proposal_ids"] == []
    assert summary["unified_generic_target_verify_next_round_mismatch_proposal_ids"] == []
    assert summary["unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept"] is True
    assert summary["unified_generic_target_verify_uses_shifted_logits"] is True
    assert summary["unified_generic_target_verify_frontier_logits_available"] is True
    assert summary["unified_generic_target_verify_current_mapping_accept_hist_by_depth"]["1"]["0"] == 1
    assert summary["unified_generic_target_verify_shifted_mapping_accept_hist_by_depth"]["1"]["4"] == 1
    sampled_inputs = summary["unified_generic_target_verify_sampled_input_ids_by_proposal_id"]
    sampled_precondition = records[0]["unified_raw_precondition_token_ids_by_proposal_id"]
    sampled_current_verify = records[0]["unified_raw_current_verify_token_ids_by_proposal_id"]
    assert sampled_current_verify["1001"] == sampled_inputs["1001"]
    assert sampled_precondition["1001"] != sampled_current_verify["1001"]
    assert records[0]["unified_generic_target_verify_appended_row0_top_token_by_proposal_id"]["1001"] == (
        sampled_inputs["1001"][1]
    )

    extra_candidates_not_verified = [json.loads(json.dumps(records[0]))]
    extra_candidates_not_verified[0]["unified_generic_target_verify_num_proposals"] = 12
    extra_candidates_not_verified[0]["unified_raw_candidate_proposal_count_by_depth"] = {"1": 12}
    errors, extra_candidate_summary = validate_records(extra_candidates_not_verified, payload)
    assert not errors, (
        "target verify token denominator should use actual verified proposals, not all candidates: "
        f"{errors}\nsummary={extra_candidate_summary}"
    )
    assert extra_candidate_summary["unified_generic_target_verify_actual_verified_proposal_count"] == 6
    assert (
        extra_candidate_summary["unified_generic_target_verify_token_count_denominator_source"]
        == "unified_raw_verified_proposal_count_by_depth"
    )

    wrong_verified_token_count = [json.loads(json.dumps(extra_candidates_not_verified[0]))]
    wrong_verified_token_count[0]["unified_generic_target_verify_num_to_verify_tokens"] = 48
    wrong_verified_token_count[0]["unified_generic_target_verify_input_ids_shape"] = [48]
    wrong_verified_token_count[0]["unified_generic_target_verify_logits_shape"] = [48, 32000]
    errors, _summary = validate_records(wrong_verified_token_count, payload)
    assert any("actual verified proposals" in error for error in errors), (
        "wrong target verify token count should still fail against actual verified proposals"
    )

    missing_temp_append = [json.loads(json.dumps(records[0]))]
    missing_temp_append[0]["unified_generic_target_verify_temp_append_used"] = False
    errors, _summary = validate_records(missing_temp_append, payload)
    assert any("temporary append" in error for error in errors), "missing temporary append should fail"

    input_not_proposal = [json.loads(json.dumps(records[0]))]
    input_not_proposal[0]["unified_generic_target_verify_next_round_input_by_proposal_id"]["1001"] = [1, 2, 3, 4]
    input_not_proposal[0]["unified_generic_target_verify_next_round_equals_original_proposal_by_proposal_id"]["1001"] = False
    errors, _summary = validate_records(input_not_proposal, payload)
    assert any("next_round_input" in error for error in errors), "next_round/proposal mismatch should fail"

    checkpoint_not_restored = [json.loads(json.dumps(records[0]))]
    checkpoint_not_restored[0]["unified_generic_target_verify_seq_len_after_rollback_by_proposal_id"]["1001"] += 1
    checkpoint_not_restored[0]["unified_generic_target_verify_checkpoint_restored_by_proposal_id"]["1001"] = False
    errors, _summary = validate_records(checkpoint_not_restored, payload)
    assert any("checkpoint" in error or "rollback length" in error for error in errors), (
        "checkpoint restore failure should fail"
    )

    current_reject_shadow_accept = [json.loads(json.dumps(records[0]))]
    current_reject_shadow_accept[0]["unified_generic_target_verify_current_window_accept_hist_by_depth"] = {
        "1": {"0": 1}
    }
    current_reject_shadow_accept[0]["unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth"] = {
        "1": {"2": 1}
    }
    errors, shadow_summary = validate_records(current_reject_shadow_accept, payload)
    assert not errors, f"shadow nonzero diagnostic should pass: {errors}\nsummary={shadow_summary}"
    assert shadow_summary["unified_generic_target_verify_current_window_all_first_token_reject"] is True
    assert shadow_summary["unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept"] is True
    assert shadow_summary["unified_generic_target_verify_proposal_window_shadow_partial_accept_count"] == 1

    both_windows_reject = [json.loads(json.dumps(records[0]))]
    both_windows_reject[0]["unified_generic_target_verify_current_window_accept_hist_by_depth"] = {"1": {"0": 1}}
    both_windows_reject[0]["unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth"] = {
        "1": {"0": 1}
    }
    both_windows_reject[0]["unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept"] = False
    errors, both_reject_summary = validate_records(both_windows_reject, payload)
    assert not errors, f"both windows reject diagnostic should pass: {errors}\nsummary={both_reject_summary}"
    assert both_reject_summary["unified_generic_target_verify_current_window_all_first_token_reject"] is True
    assert both_reject_summary["unified_generic_target_verify_proposal_window_shadow_all_first_token_reject"] is True

    guarded_proposal_window = [json.loads(json.dumps(records[0]))]
    guarded_proposal_window[0]["unified_generic_proposal_window_verify_enabled"] = True
    guarded_proposal_window[0]["unified_raw_accepted_len_hist_by_depth"] = {
        "1": {"4": 1},
        "2": {"4": 1},
        "3": {"4": 1},
        "4": {"4": 1},
        "5": {"4": 1},
        "6": {"4": 1},
    }
    guarded_proposal_window[0]["unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth"] = (
        guarded_proposal_window[0]["unified_raw_accepted_len_hist_by_depth"]
    )
    errors, guarded_summary = validate_records(guarded_proposal_window, payload)
    assert not errors, f"guarded proposal-window synthetic should pass: {errors}\nsummary={guarded_summary}"
    assert guarded_summary["unified_generic_proposal_window_verify_enabled"] is True

    sampled_missing = [json.loads(json.dumps(records[0]))]
    for field in (
        "unified_generic_target_verify_current_to_be_verified_by_proposal_id",
        "unified_generic_target_verify_proposal_window_to_be_verified_by_proposal_id",
        "unified_generic_target_verify_first_position_target_top5_tokens_by_proposal_id",
    ):
        sampled_missing[0].pop(field, None)
    errors, _summary = validate_records(sampled_missing, payload)
    assert not errors, f"missing sampled debug fields should not fail: {errors}"

    ready_owner_base = json.loads(json.dumps(synthetic_reject_partial_records()[0]))
    ready_owner_payload = synthetic_payload(total_output_tokens=2)
    ready_owner_ok = [json.loads(json.dumps(ready_owner_base))]
    ready_owner_ok[0].update(
        {
            "target_home_set": [1, 2],
            "target_normal_verify_seq_ids": [2],
            "draft_home_set": [0],
            "original_draft_home_set": [0, 1],
            "actual_draft_home_set_for_normal_draft": [0],
            "unified_ready_child_lane_owner_seq_ids": [1],
            "unified_ready_child_lane_owner_request_ids": {"1": "req-1"},
            "unified_ready_child_excluded_from_normal_draft_seq_ids": [1],
            "unified_ready_child_excluded_from_target_normal_verify_seq_ids": [1],
            "unified_ready_child_remaining_in_target_normal_verify_seq_ids": [],
            "unified_ready_child_normal_verify_exclusion_mismatch_count": 0,
            "unified_ready_child_missing_normal_proposal_allowed_seq_ids": [1],
            "missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids": [1],
            "missing_buffered_proposal_unexpected_seq_ids": [],
            "unified_single_child_ahead_enabled": True,
            "unified_max_unverified_depth_ahead": 1,
        }
    )
    errors, ready_owner_ok_summary = validate_records(ready_owner_ok, ready_owner_payload)
    assert not errors, f"ready-child owner excluded from both normal lanes should pass: {errors}\nsummary={ready_owner_ok_summary}"

    ready_owner_target_mismatch = [json.loads(json.dumps(ready_owner_ok[0]))]
    ready_owner_target_mismatch[0]["target_normal_verify_seq_ids"] = [1, 2]
    ready_owner_target_mismatch[0]["unified_ready_child_remaining_in_target_normal_verify_seq_ids"] = [1]
    ready_owner_target_mismatch[0]["unified_ready_child_normal_verify_exclusion_mismatch_count"] = 1
    ready_owner_target_mismatch[0]["unified_ready_child_normal_verify_exclusion_mismatch_examples"] = [
        {"seq_id": 1, "child_proposal_id": 1002, "child_depth": 2}
    ]
    errors, _summary = validate_records(ready_owner_target_mismatch, ready_owner_payload)
    assert any("target normal verify" in error or "normal lane" in error for error in errors), (
        "ready-child owner still in target normal verify should fail"
    )

    ready_owner_missing_allowed_but_targeted = [json.loads(json.dumps(ready_owner_ok[0]))]
    ready_owner_missing_allowed_but_targeted[0]["target_normal_verify_seq_ids"] = [1, 2]
    ready_owner_missing_allowed_but_targeted[0]["missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids"] = [1]
    errors, _summary = validate_records(ready_owner_missing_allowed_but_targeted, ready_owner_payload)
    assert any("target normal verify" in error for error in errors), (
        "ready-child owner missing a normal proposal while still targeted should fail"
    )

    non_owned_missing_buffer = [json.loads(json.dumps(ready_owner_base))]
    non_owned_missing_buffer[0]["target_home_set"] = [9]
    non_owned_missing_buffer[0]["target_normal_verify_seq_ids"] = [9]
    non_owned_missing_buffer[0]["missing_buffered_proposal_unexpected_seq_ids"] = [9]
    non_owned_missing_buffer[0]["unified_single_child_ahead_enabled"] = True
    non_owned_missing_buffer[0]["unified_max_unverified_depth_ahead"] = 1
    errors, _summary = validate_records(non_owned_missing_buffer, ready_owner_payload)
    assert any("missing non-owned buffered proposals" in error for error in errors), (
        "non-owned target-normal seq missing buffer should fail"
    )

    def normal_buffer_filter_record(updates: dict[str, Any]) -> list[dict[str, Any]]:
        record = json.loads(json.dumps(ready_owner_base))
        record.update(
            {
                "target_home_set": [2],
                "target_normal_verify_seq_ids": [2],
                "target_normal_verify_seq_ids_before_buffer_filter": [2],
                "target_normal_verify_seq_ids_after_buffer_filter": [2],
                "dual_proposal_buffer_available_seq_ids_before_target_verify": [2],
                "target_normal_verify_missing_buffer_seq_ids": [],
                "target_normal_verify_deferred_missing_buffer_seq_ids": [],
                "target_normal_verify_missing_buffer_reason_by_seq_id": {},
                "target_normal_verify_deferred_missing_buffer_reason_counts": {},
                "missing_buffered_proposal_unexpected_seq_ids": [],
                "unified_single_child_ahead_enabled": True,
                "unified_max_unverified_depth_ahead": 1,
            }
        )
        record.update(updates)
        return [record]

    buffer_hit = normal_buffer_filter_record({})
    errors, buffer_hit_summary = validate_records(buffer_hit, ready_owner_payload)
    assert not errors, f"target normal verify seq with buffered proposal should pass: {errors}\nsummary={buffer_hit_summary}"

    missing_deferred = normal_buffer_filter_record(
        {
            "target_home_set": [9],
            "target_normal_verify_seq_ids": [],
            "target_normal_verify_seq_ids_before_buffer_filter": [9],
            "target_normal_verify_seq_ids_after_buffer_filter": [],
            "dual_proposal_buffer_available_seq_ids_before_target_verify": [],
            "target_normal_verify_missing_buffer_seq_ids": [9],
            "target_normal_verify_deferred_missing_buffer_seq_ids": [9],
            "target_normal_verify_missing_buffer_reason_by_seq_id": {"9": "missing_normal_proposal_deferred"},
            "target_normal_verify_deferred_missing_buffer_reason_counts": {
                "missing_normal_proposal_deferred": 1
            },
        }
    )
    errors, missing_deferred_summary = validate_records(missing_deferred, ready_owner_payload)
    assert not errors, (
        "target normal verify seq missing buffer should pass when deferred with explicit reason: "
        f"{errors}\nsummary={missing_deferred_summary}"
    )

    missing_still_targeted = normal_buffer_filter_record(
        {
            "target_home_set": [9],
            "target_normal_verify_seq_ids": [9],
            "target_normal_verify_seq_ids_before_buffer_filter": [9],
            "target_normal_verify_seq_ids_after_buffer_filter": [9],
            "dual_proposal_buffer_available_seq_ids_before_target_verify": [],
            "target_normal_verify_missing_buffer_seq_ids": [9],
            "target_normal_verify_deferred_missing_buffer_seq_ids": [9],
            "target_normal_verify_missing_buffer_reason_by_seq_id": {"9": "missing_normal_proposal_deferred"},
            "target_normal_verify_deferred_missing_buffer_reason_counts": {
                "missing_normal_proposal_deferred": 1
            },
        }
    )
    errors, _summary = validate_records(missing_still_targeted, ready_owner_payload)
    assert any("after buffer filter includes seqs without buffered proposals" in error for error in errors), (
        "target normal verify seq missing buffer and not owned must not remain after filter"
    )

    unified_owner_missing_removed = normal_buffer_filter_record(
        {
            "target_home_set": [1],
            "target_normal_verify_seq_ids": [],
            "target_normal_verify_seq_ids_before_buffer_filter": [1],
            "target_normal_verify_seq_ids_after_buffer_filter": [],
            "dual_proposal_buffer_available_seq_ids_before_target_verify": [],
            "target_normal_verify_missing_buffer_seq_ids": [1],
            "target_normal_verify_missing_buffer_reason_by_seq_id": {"1": "unified_ready_child_owned"},
            "target_normal_verify_deferred_missing_buffer_reason_counts": {"unified_ready_child_owned": 1},
            "unified_ready_child_lane_owner_seq_ids": [1],
            "unified_ready_child_excluded_from_target_normal_verify_seq_ids": [1],
            "unified_ready_child_missing_normal_proposal_allowed_seq_ids": [1],
            "missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids": [1],
        }
    )
    errors, unified_owner_removed_summary = validate_records(unified_owner_missing_removed, ready_owner_payload)
    assert not errors, (
        "missing buffer due unified ready-child ownership should pass only when removed from target normal verify: "
        f"{errors}\nsummary={unified_owner_removed_summary}"
    )

    raw_target_home_filtered = normal_buffer_filter_record(
        {
            "target_home_set": [1, 2],
            "target_normal_verify_seq_ids": [2],
            "target_normal_verify_seq_ids_before_buffer_filter": [1, 2],
            "target_normal_verify_seq_ids_after_buffer_filter": [2],
            "dual_proposal_buffer_available_seq_ids_before_target_verify": [2],
            "target_normal_verify_missing_buffer_seq_ids": [1],
            "target_normal_verify_deferred_missing_buffer_seq_ids": [1],
            "target_normal_verify_missing_buffer_reason_by_seq_id": {"1": "missing_normal_proposal_deferred"},
            "target_normal_verify_deferred_missing_buffer_reason_counts": {
                "missing_normal_proposal_deferred": 1
            },
        }
    )
    errors, raw_target_home_summary = validate_records(raw_target_home_filtered, ready_owner_payload)
    assert not errors, (
        "raw target_home_set seq without buffered proposal should be filtered/deferred with reason: "
        f"{errors}\nsummary={raw_target_home_summary}"
    )

    discarded_before_verify = normal_buffer_filter_record(
        {
            "target_home_set": [3],
            "target_normal_verify_seq_ids": [],
            "target_normal_verify_seq_ids_before_buffer_filter": [3],
            "target_normal_verify_seq_ids_after_buffer_filter": [],
            "dual_proposal_buffer_available_seq_ids_before_target_verify": [],
            "target_normal_verify_missing_buffer_seq_ids": [3],
            "target_normal_verify_deferred_missing_buffer_seq_ids": [3],
            "target_normal_verify_missing_buffer_reason_by_seq_id": {"3": "missing_normal_proposal_deferred"},
            "target_normal_verify_deferred_missing_buffer_reason_counts": {
                "missing_normal_proposal_deferred": 1
            },
            "target_normal_verify_missing_buffer_details": [
                {
                    "seq_id": 3,
                    "request_id": "req-3",
                    "proposal_id": 303,
                    "plan_id": 12,
                    "dual_step_id": 8,
                    "was_buffer_discarded": True,
                    "discard_reason": "manual_drop",
                    "was_sequence_finished": False,
                    "was_eager_owned": False,
                    "was_unified_ready_child_owned": False,
                }
            ],
            "normal_proposal_buffer_event_history": [
                {
                    "step_id": 7,
                    "plan_id": 11,
                    "seq_id": 3,
                    "request_id": "req-3",
                    "proposal_id": 303,
                    "event_type": "store",
                    "reason": "target_received_normal_proposal_store",
                    "buffer_size_after": 1,
                },
                {
                    "step_id": 8,
                    "plan_id": 12,
                    "seq_id": 3,
                    "request_id": "req-3",
                    "proposal_id": 303,
                    "event_type": "discard",
                    "reason": "manual_drop",
                    "buffer_size_after": 0,
                },
            ],
        }
    )
    errors, _summary = validate_records(discarded_before_verify, ready_owner_payload)
    assert any("discard" in error for error in errors), (
        "buffer store then discard before target verify should fail without terminal/owned reason"
    )

    terminal_discarded_before_verify = json.loads(json.dumps(discarded_before_verify))
    terminal_discarded_before_verify[0]["target_normal_verify_missing_buffer_details"][0].update(
        {
            "discard_reason": "sequence_finished",
            "was_sequence_finished": True,
        }
    )
    terminal_discarded_before_verify[0]["normal_proposal_buffer_event_history"][1][
        "reason"
    ] = "sequence_finished"
    errors, terminal_discard_summary = validate_records(terminal_discarded_before_verify, ready_owner_payload)
    assert not errors, (
        "buffer discard before target verify should pass with terminal sequence-finished reason: "
        f"{errors}\nsummary={terminal_discard_summary}"
    )

    target_consume_before_verify = json.loads(json.dumps(discarded_before_verify))
    target_consume_before_verify[0]["target_normal_verify_missing_buffer_details"][0].update(
        {
            "discard_reason": "target_normal_verify_consumed",
            "was_sequence_finished": False,
        }
    )
    target_consume_before_verify[0]["normal_proposal_buffer_event_history"] = [
        {
            "step_id": 7,
            "plan_id": 11,
            "seq_id": 3,
            "request_id": "req-3",
            "proposal_id": 303,
            "event_type": "store",
            "reason": "target_received_normal_proposal_store",
            "buffer_size_after": 1,
        },
        {
            "step_id": 8,
            "plan_id": 12,
            "seq_id": 3,
            "request_id": "req-3",
            "proposal_id": 303,
            "event_type": "consume",
            "reason": "target_normal_verify_consumed",
            "buffer_size_after": 0,
        },
    ]
    errors, target_consume_summary = validate_records(target_consume_before_verify, ready_owner_payload)
    assert not errors, (
        "target normal verify consume should be legal, not an illegal discard: "
        f"{errors}\nsummary={target_consume_summary}"
    )
    assert target_consume_summary["normal_proposal_buffer_legal_consume_count_by_reason"][
        "target_normal_verify_consumed"
    ] >= 1

    draft_apply_consume = json.loads(json.dumps(target_consume_before_verify))
    draft_apply_consume[0]["target_normal_verify_missing_buffer_details"][0][
        "discard_reason"
    ] = "draft_apply_verify_consumed"
    draft_apply_consume[0]["normal_proposal_buffer_event_history"][1]["reason"] = "draft_apply_verify_consumed"
    errors, draft_apply_summary = validate_records(draft_apply_consume, ready_owner_payload)
    assert not errors, (
        "draft apply verify consume should be legal, not an illegal discard: "
        f"{errors}\nsummary={draft_apply_summary}"
    )
    assert draft_apply_summary["normal_proposal_buffer_legal_consume_count_by_reason"][
        "draft_apply_verify_consumed"
    ] >= 1

    ready_owner_scheduled = [json.loads(json.dumps(ready_owner_ok[0]))]
    ready_owner_scheduled[0]["unified_child_scheduled_for_target_verify_count_by_depth"] = {"2": 1}
    ready_owner_scheduled[0]["unified_child_target_verify_inflight_count_by_depth"] = {"2": 1}
    ready_owner_scheduled[0]["unified_child_scheduled_state_by_depth"] = {"2": {"READY_TO_VERIFY": 1}}
    errors, ready_owner_scheduled_summary = validate_records(ready_owner_scheduled, ready_owner_payload)
    assert not errors, (
        "ready-child owner routed to promoted child scheduling should pass: "
        f"{errors}\nsummary={ready_owner_scheduled_summary}"
    )

    reject_partial_records = synthetic_reject_partial_records()
    errors, reject_partial_summary = validate_records(
        reject_partial_records,
        synthetic_payload(total_output_tokens=2),
    )
    assert not errors, f"reject/partial target synthetic failed: {errors}\nsummary={reject_partial_summary}"
    assert reject_partial_summary["max_real_committed_depth"] == 0
    assert reject_partial_summary["unified_raw_reject_proposal_count_by_depth"]["1"] == 179
    assert reject_partial_summary["unified_raw_no_mutation_reject_proposal_count_by_depth"]["1"] == 179
    assert reject_partial_summary["unified_raw_revised_token_count_by_depth"]["1"] == 180
    assert reject_partial_summary["unified_raw_partial_recovery_applied_proposal_count_by_depth"]["1"] == 1
    assert reject_partial_summary["unified_raw_full_commit_proposal_count_by_depth"] == {}
    assert reject_partial_summary["unified_raw_reject_revised_correction_applied_proposal_count_by_depth"] == {}
    assert reject_partial_summary["partial_prefix_revised_token_count"] == 1
    assert reject_partial_summary["total_partial_recovered_token_count"] == 2
    assert reject_partial_summary["unified_generic_target_verify_temp_append_used"] is True

    target_tp1_owner = synthetic_records()
    errors, tp1_summary = validate_records(target_tp1_owner, payload)
    assert not errors, f"target_tp=1 owner logits synthetic failed: {errors}\nsummary={tp1_summary}"
    assert tp1_summary["unified_generic_target_verify_logits_owner_record_count"] == 1
    assert tp1_summary["unified_generic_target_verify_owner_frontier_logits_available"] is True
    assert tp1_summary["unified_generic_target_verify_owner_appended_logits_available"] is True

    target_tp3_records = synthetic_records()
    target_tp3_records.extend(
        [
            synthetic_target_tp_worker_record(target_tp3_records[0], rank=2),
            synthetic_target_tp_worker_record(target_tp3_records[0], rank=3),
        ]
    )
    errors, tp3_summary = validate_records(target_tp3_records, payload)
    assert not errors, f"target_tp=3 owner/workers synthetic failed: {errors}\nsummary={tp3_summary}"
    assert tp3_summary["unified_generic_target_verify_logits_owner_record_count"] == 1
    assert tp3_summary["unified_generic_target_verify_non_owner_record_count"] == 2
    assert tp3_summary["unified_generic_target_verify_non_owner_frontier_none_allowed_count"] == 2

    target_tp3_owner_missing = synthetic_records()
    target_tp3_owner_missing[0]["unified_generic_target_verify_frontier_logits_available"] = False
    target_tp3_owner_missing[0]["unified_generic_target_verify_appended_logits_available"] = False
    target_tp3_owner_missing.extend(
        [
            synthetic_target_tp_worker_record(target_tp3_owner_missing[0], rank=2),
            synthetic_target_tp_worker_record(target_tp3_owner_missing[0], rank=3),
        ]
    )
    errors, _summary = validate_records(target_tp3_owner_missing, payload)
    assert any("owner must have frontier logits" in error for error in errors), (
        "target_tp=3 owner missing frontier logits should fail"
    )

    target_tp3_bad_worker = synthetic_records()
    target_tp3_bad_worker.append(synthetic_target_tp_worker_record(target_tp3_bad_worker[0], rank=2, bad_raw=True))
    errors, _summary = validate_records(target_tp3_bad_worker, payload)
    assert any("non-owner target TP ranks" in error for error in errors), (
        "worker raw result counters should fail"
    )

    zero_proposal_owner = synthetic_records()
    for record in zero_proposal_owner:
        for field in list(record):
            if field.startswith("unified_raw_") and field not in {
                "unified_raw_target_verification_available",
                "unified_raw_verification_source",
            }:
                record.pop(field, None)
            elif field.startswith("unified_generic_target_verify_") and field not in {
                "unified_generic_target_verify_logits_owner",
                "unified_generic_target_verify_frontier_logits_available",
                "unified_generic_target_verify_appended_logits_available",
                "unified_generic_target_verify_frontier_logits_none_allowed",
                "unified_generic_target_verify_rank",
                "unified_generic_target_verify_tp_local_rank",
                "unified_generic_target_verify_target_master_rank",
                "unified_generic_target_verify_num_proposals",
                "unified_generic_target_verify_num_to_verify_tokens",
            }:
                record.pop(field, None)
        record["unified_generic_target_verify_logits_owner"] = True
        record["unified_generic_target_verify_frontier_logits_available"] = False
        record["unified_generic_target_verify_appended_logits_available"] = False
        record["unified_generic_target_verify_frontier_logits_none_allowed"] = False
        record["unified_generic_target_verify_num_proposals"] = 0
        record["unified_generic_target_verify_num_to_verify_tokens"] = 0
    zero_proposal_owner.extend(
        [
            synthetic_target_tp_worker_record(zero_proposal_owner[0], rank=2),
            synthetic_target_tp_worker_record(zero_proposal_owner[0], rank=3),
        ]
    )
    for worker in zero_proposal_owner[1:]:
        worker["unified_generic_target_verify_num_proposals"] = 0
        worker["unified_generic_target_verify_num_to_verify_tokens"] = 0
    errors, zero_summary = validate_records(zero_proposal_owner, payload)
    assert not errors, f"target_tp=3 zero proposal synthetic failed: {errors}\nsummary={zero_summary}"
    assert zero_summary["unified_generic_target_verify_owner_num_proposals"] == 0

    single_child_pass = synthetic_single_child_records([[1], [2], [3]], committed_depths={1, 2})
    errors, single_child_summary = validate_records(single_child_pass, synthetic_single_child_payload())
    assert not errors, f"single-child-ahead pass synthetic failed: {errors}\nsummary={single_child_summary}"
    assert single_child_summary["unified_single_child_ahead_enabled"] is True
    assert single_child_summary["unified_unverified_depth_ahead_max_observed"] <= 1
    assert single_child_summary["unified_child_generated_from_full_accept_parent_count_by_depth"]["2"] == 1
    assert single_child_summary["unified_raw_candidate_proposal_count_available"] is True

    inflight_child_pass = synthetic_single_child_records([[1, 2]], inflight_child_depths={2})
    errors, inflight_child_summary = validate_records(inflight_child_pass, synthetic_single_child_payload())
    assert not errors, f"in-flight child generation synthetic failed: {errors}\nsummary={inflight_child_summary}"
    assert inflight_child_summary["unified_child_generated_from_inflight_parent_count_by_depth"]["2"] == 1
    assert inflight_child_summary["unified_depth2_generated_while_depth1_verifying_count"] == 1

    promoted_child_scheduled = synthetic_single_child_records(
        [[1], [2]],
        committed_depths={1},
        promoted_child_depths={2},
        scheduled_child_depths={2},
        target_verified_after_promotion_depths={2},
    )
    errors, promoted_scheduled_summary = validate_records(
        promoted_child_scheduled,
        synthetic_single_child_payload(),
    )
    assert not errors, (
        "promoted child scheduled for target verification should pass: "
        f"errors={errors}\nsummary={promoted_scheduled_summary}"
    )
    assert promoted_scheduled_summary["unified_child_scheduled_for_target_verify_count_by_depth"]["2"] == 1
    assert promoted_scheduled_summary["unified_child_target_verified_after_promotion_count_by_depth"]["2"] == 1

    promoted_child_disappeared = synthetic_single_child_records(
        [[1], [2]],
        committed_depths={1},
        promoted_child_depths={2},
    )
    errors, promoted_disappeared_summary = validate_records(
        promoted_child_disappeared,
        synthetic_single_child_payload(),
    )
    assert any("promoted children were not target verified" in error for error in errors), (
        "promoted child without scheduling or reason should fail: "
        f"errors={errors}\nsummary={promoted_disappeared_summary}"
    )

    promoted_child_not_scheduled_reason = synthetic_single_child_records(
        [[1], [2]],
        committed_depths={1},
        promoted_child_depths={2},
        ready_not_scheduled_reasons_by_depth={2: "stale_base"},
    )
    errors, promoted_reason_summary = validate_records(
        promoted_child_not_scheduled_reason,
        synthetic_single_child_payload(),
    )
    assert any("dropped as stale_base" in error for error in errors), (
        "promoted child without verification must fail when the only scheduling reason is stale_base: "
        f"errors={errors}\nsummary={promoted_reason_summary}"
    )

    promoted_child_terminal_reason = synthetic_single_child_records(
        [[1], [2]],
        committed_depths={1},
        promoted_child_depths={2},
        ready_not_scheduled_reasons_by_depth={2: "sequence_finished"},
    )
    errors, promoted_terminal_summary = validate_records(
        promoted_child_terminal_reason,
        synthetic_single_child_payload(),
    )
    assert not errors, (
        "promoted child without verification should pass with a non-stale terminal scheduling reason: "
        f"errors={errors}\nsummary={promoted_terminal_summary}"
    )

    premature_child_verify = synthetic_single_child_records(
        [[1, 2]],
        inflight_child_depths={2},
        verified_before_parent_full_accept_violation=True,
    )
    errors, premature_child_summary = validate_records(premature_child_verify, synthetic_single_child_payload())
    assert any("target verified child before parent full accept" in error for error in errors), (
        "target verification before parent full accept should fail: "
        f"errors={errors}\nsummary={premature_child_summary}"
    )

    partial_parent_invalidates_child = synthetic_single_child_records(
        [[1, 2]],
        parent_outcomes_by_depth={1: "partial"},
        inflight_child_depths={2},
        invalidated_after_non_full_child_depths={2},
    )
    errors, partial_invalidated_summary = validate_records(
        partial_parent_invalidates_child,
        synthetic_single_child_payload(),
    )
    assert not errors, (
        "in-flight child invalidated after partial parent should pass: "
        f"errors={errors}\nsummary={partial_invalidated_summary}"
    )

    aggressive_under_flag = synthetic_single_child_records([[1, 2, 3, 4]])
    errors, aggressive_summary = validate_records(aggressive_under_flag, synthetic_single_child_payload())
    assert any("single-child-ahead" in error or "grandchild" in error for error in errors), (
        f"aggressive burst under single-child flag should fail: errors={errors}\nsummary={aggressive_summary}"
    )

    parent_partial_one_child = synthetic_single_child_records(
        [[1, 2]],
        stop_reason="partial_recovery_selected",
        parent_outcomes_by_depth={1: "partial"},
    )
    errors, partial_one_child_summary = validate_records(parent_partial_one_child, synthetic_single_child_payload())
    assert any("parent full-accept guard" in error for error in errors), (
        f"child from verified partial parent should fail: errors={errors}\nsummary={partial_one_child_summary}"
    )

    parent_reject_one_child = synthetic_single_child_records(
        [[1, 2]],
        stop_reason="parent_not_full_accept",
        parent_outcomes_by_depth={1: "reject"},
    )
    errors, reject_one_child_summary = validate_records(parent_reject_one_child, synthetic_single_child_payload())
    assert any("parent full-accept guard" in error for error in errors), (
        f"child from verified reject parent should fail: errors={errors}\nsummary={reject_one_child_summary}"
    )

    parent_invalidated_one_child = synthetic_single_child_records(
        [[1, 2]],
        stop_reason="parent_not_full_accept",
        parent_outcomes_by_depth={1: "invalidated"},
    )
    errors, invalidated_one_child_summary = validate_records(
        parent_invalidated_one_child,
        synthetic_single_child_payload(),
    )
    assert any("parent full-accept guard" in error for error in errors), (
        "child from invalidated parent should fail: "
        f"errors={errors}\nsummary={invalidated_one_child_summary}"
    )

    parent_missing_one_child = synthetic_single_child_records(
        [[1, 2]],
        stop_reason="parent_not_full_accept",
        parent_outcomes_by_depth={1: "missing"},
    )
    errors, missing_one_child_summary = validate_records(parent_missing_one_child, synthetic_single_child_payload())
    assert any("parent full-accept guard" in error for error in errors), (
        f"child from missing parent outcome should fail: errors={errors}\nsummary={missing_one_child_summary}"
    )

    full_parent_later_invalidated = synthetic_single_child_records(
        [[1, 2]],
        committed_depths={1},
        stop_reason="parent_not_full_accept",
        invalidated_parent_not_full_depths={2},
    )
    errors, invalidated_child_summary = validate_records(full_parent_later_invalidated, synthetic_single_child_payload())
    assert any("invalidated as parent_not_full_accept" in error for error in errors), (
        "child generated from full parent but invalidated as parent_not_full_accept should fail: "
        f"errors={errors}\nsummary={invalidated_child_summary}"
    )

    partial_recovery_restart_depth1 = synthetic_single_child_records(
        [[1], [1]],
        parent_outcomes_by_depth={1: "partial_recovery"},
        stop_reason="partial_recovery_selected",
    )
    errors, restart_summary = validate_records(partial_recovery_restart_depth1, synthetic_single_child_payload())
    assert not errors, f"partial recovery restart depth1 synthetic should pass: {errors}\nsummary={restart_summary}"

    parent_full_next_child = synthetic_single_child_records(
        [[1], [2]],
        committed_depths={1, 2},
        full_accept_without_child_reasons_by_depth={2: "no_active_sequence"},
    )
    errors, full_next_summary = validate_records(parent_full_next_child, synthetic_single_child_payload())
    assert not errors, f"parent full accept next child should pass: {errors}\nsummary={full_next_summary}"
    assert full_next_summary["max_real_committed_depth"] == 2
    assert full_next_summary["unified_full_accept_parent_registered_count_by_depth"]["1"] == 1
    assert full_next_summary["unified_full_accept_parent_selected_for_child_count_by_depth"]["1"] == 1

    full_parent_no_child_no_reason = synthetic_single_child_records([[1]], committed_depths={1})
    errors, no_child_no_reason_summary = validate_records(
        full_parent_no_child_no_reason,
        synthetic_single_child_payload(),
    )
    assert any("did not produce continuation children or block reasons" in error for error in errors), (
        "full-accepted parent without child or reason should fail: "
        f"errors={errors}\nsummary={no_child_no_reason_summary}"
    )

    full_parent_no_child_with_reason = synthetic_single_child_records(
        [[1]],
        committed_depths={1},
        full_accept_without_child_reasons_by_depth={1: "no_active_sequence"},
    )
    errors, no_child_reason_summary = validate_records(
        full_parent_no_child_with_reason,
        synthetic_single_child_payload(),
    )
    assert not errors, (
        "full-accepted parent without child should pass with explicit reason: "
        f"{errors}\nsummary={no_child_reason_summary}"
    )
    assert no_child_reason_summary["unified_full_accept_parent_registered_count_by_depth"]["1"] == 1
    assert (
        no_child_reason_summary["unified_full_accept_without_child_reason_counts_by_depth"]["1"][
            "no_active_sequence"
        ]
        == 1
    )
    assert no_child_reason_summary["unified_depth2_generation_block_reason_counts"]["no_active_sequence"] == 1

    default_aggressive = synthetic_single_child_records(
        [[1, 2, 3, 4, 5]],
        max_unverified_ahead=0,
        committed_depths={1, 2, 3, 4, 5},
    )
    errors, default_aggressive_summary = validate_records(
        default_aggressive,
        synthetic_single_child_payload(max_unverified_ahead=0),
    )
    assert not errors, f"default aggressive compatibility should pass: {errors}\nsummary={default_aggressive_summary}"
    assert default_aggressive_summary["unified_single_child_ahead_enabled"] is False

    raw_candidate_missing = synthetic_single_child_records([[1]])
    errors, raw_candidate_summary = validate_records(raw_candidate_missing, synthetic_single_child_payload())
    assert not errors, f"raw candidate fallback synthetic should pass: {errors}\nsummary={raw_candidate_summary}"
    assert raw_candidate_summary["unified_raw_candidate_proposal_count_by_depth"]["1"] == 1
    assert raw_candidate_summary["unified_raw_candidate_proposal_count_available"] is True

    finished_one_child = synthetic_single_child_records([[1]], stop_reason="sequence_finished")
    errors, finished_summary = validate_records(finished_one_child, synthetic_single_child_payload())
    assert not errors, f"sequence finished single-child synthetic should pass: {errors}\nsummary={finished_summary}"
    assert finished_summary["unified_sequence_finished_count_by_depth"]

    missing_target_verify = [dict(records[0])]
    missing_target_verify[0]["unified_raw_target_verification_available"] = False
    missing_target_verify[0]["unified_raw_verification_source"] = "draft_commit_decision_no_target_verify"
    errors, _summary = validate_records(missing_target_verify, payload)
    assert any("target verification" in error for error in errors), "missing target verification should fail"

    old_trace_fallback = [dict(records[0])]
    for field in list(old_trace_fallback[0]):
        if field.startswith("unified_raw_"):
            old_trace_fallback[0].pop(field, None)
    errors, _summary = validate_records(old_trace_fallback, payload)
    assert not errors, f"old trace fallback should pass: {errors}"

    old_target_trace_without_temp_append = [json.loads(json.dumps(records[0]))]
    for field in list(old_target_trace_without_temp_append[0]):
        if field.startswith("unified_generic_target_verify_"):
            old_target_trace_without_temp_append[0].pop(field, None)
    errors, _summary = validate_records(old_target_trace_without_temp_append, payload)
    assert not errors, f"old target trace without temp append fields should pass: {errors}"

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
        "unified_full_commit_token_count_by_depth",
        "unified_total_full_commit_token_count",
        "unified_partial_recovered_token_count_by_depth",
        "unified_total_partial_recovered_token_count",
        "unified_partial_revised_token_count_by_depth",
        "unified_total_output_token_count_by_depth",
        "unified_total_output_token_count",
        "unified_commit_share_by_depth",
        "unified_raw_target_verification_available",
        "unified_raw_verification_source",
        "unified_raw_verified_proposal_count_by_depth",
        "unified_raw_full_accept_proposal_count_by_depth",
        "unified_raw_partial_accept_proposal_count_by_depth",
        "unified_raw_reject_proposal_count_by_depth",
        "unified_raw_invalidated_proposal_count_by_depth",
        "unified_raw_accepted_len_hist_by_depth",
        "unified_raw_revised_token_count_by_depth",
        "unified_raw_partial_recovery_eligible_count_by_depth",
        "unified_raw_partial_recovery_ineligible_reason_counts_by_depth",
        "unified_raw_candidate_proposal_count_by_depth",
        "unified_raw_candidate_proposal_count_available",
        "unified_raw_committed_proposal_count_by_depth",
        "unified_raw_full_commit_proposal_count_by_depth",
        "unified_raw_partial_recovery_applied_proposal_count_by_depth",
        "unified_raw_reject_revised_correction_applied_proposal_count_by_depth",
        "unified_raw_no_mutation_reject_proposal_count_by_depth",
        "unified_raw_verified_to_committed_ratio_by_depth",
        "unified_candidate_waste_ratio_by_depth",
        "unified_invalidated_candidate_ratio_by_depth",
        "unified_verified_candidate_ratio_by_depth",
        "unified_committed_candidate_ratio_by_depth",
        "unified_single_child_ahead_enabled",
        "unified_max_unverified_depth_ahead",
        "unified_unverified_depth_ahead_max_observed",
        "unified_unverified_depth_ahead_by_chain",
        "unified_single_child_ahead_violation_count",
        "unified_single_child_ahead_violation_examples",
        "unified_generated_grandchild_before_parent_verified_count",
        "unified_generated_grandchild_before_parent_verified_examples",
        "unified_candidate_depth_created_before_parent_verified_count_by_depth",
        "unified_candidate_depth_created_after_parent_verified_count_by_depth",
        "unified_child_generated_from_full_accept_parent_count_by_depth",
        "unified_child_generated_from_inflight_parent_count_by_depth",
        "unified_child_pending_parent_result_count_by_depth",
        "unified_child_promoted_after_parent_full_accept_count_by_depth",
        "unified_child_promoted_to_ready_count_by_depth",
        "unified_child_ready_for_target_verify_count_by_depth",
        "unified_child_ready_but_not_scheduled_count_by_depth",
        "unified_child_ready_not_scheduled_reason_counts_by_depth",
        "unified_child_ready_seq_normal_lane_excluded_count_by_depth",
        "unified_child_ready_seq_normal_lane_conflict_count_by_depth",
        "unified_child_ready_seq_normal_lane_conflict_examples",
        "unified_child_stale_base_count_by_depth",
        "unified_child_stale_base_examples",
        "unified_child_scheduled_for_target_verify_count_by_depth",
        "unified_child_target_verify_inflight_count_by_depth",
        "unified_child_scheduled_state_by_depth",
        "unified_child_duplicate_schedule_skip_count_by_depth",
        "unified_child_schedule_state_error_count_by_depth",
        "unified_child_schedule_state_error_examples",
        "unified_ready_child_lane_owner_seq_ids",
        "unified_ready_child_lane_owner_request_ids",
        "unified_ready_child_excluded_from_normal_draft_seq_ids",
        "unified_ready_child_excluded_from_target_normal_verify_before_filter_seq_ids",
        "unified_ready_child_excluded_from_target_normal_verify_after_filter_seq_ids",
        "unified_ready_child_excluded_from_target_normal_verify_seq_ids",
        "unified_ready_child_remaining_in_target_normal_verify_seq_ids",
        "unified_ready_child_normal_verify_exclusion_mismatch_count",
        "unified_ready_child_normal_verify_exclusion_mismatch_examples",
        "unified_ready_child_missing_normal_proposal_allowed_seq_ids",
        "missing_buffered_proposal_allowed_by_unified_ready_child_seq_ids",
        "unified_child_target_verified_after_promotion_count_by_depth",
        "unified_child_invalidated_after_parent_non_full_count_by_depth",
        "unified_child_verified_after_parent_full_accept_count_by_depth",
        "unified_child_verified_before_parent_result_count_by_depth",
        "unified_child_verified_before_parent_full_accept_violation_count",
        "unified_child_generated_in_same_burst_as_grandchild_violation_count",
        "unified_depth2_generated_while_depth1_verifying_count",
        "unified_depth2_ready_after_depth1_full_accept_count",
        "unified_depth2_scheduled_after_depth1_full_accept_count",
        "unified_depth2_verified_after_depth1_full_accept_count",
        "unified_depth3_generated_while_depth2_verifying_count",
        "unified_child_generated_from_non_full_parent_count_by_depth",
        "unified_child_generated_from_unverified_parent_count_by_depth",
        "unified_child_generated_from_partial_parent_count_by_depth",
        "unified_child_generated_from_reject_parent_count_by_depth",
        "unified_child_generated_from_invalidated_parent_count_by_depth",
        "unified_child_parent_outcome_missing_count_by_depth",
        "unified_child_parent_full_accept_guard_violation_count",
        "unified_child_parent_full_accept_guard_examples",
        "unified_child_depth_verified_count_by_depth",
        "unified_child_depth_invalidated_parent_not_full_count_by_depth",
        "unified_full_accept_parent_registered_count_by_depth",
        "unified_full_accept_parent_registry_size_by_depth",
        "unified_full_accept_parent_selected_for_child_count_by_depth",
        "unified_full_accept_parent_not_selected_reason_counts_by_depth",
        "unified_child_generated_from_full_accept_parent_examples",
        "unified_depth2_generation_block_reason_counts",
        "unified_full_accept_without_child_reason_counts_by_depth",
        "unified_full_accept_parent_frontier_mismatch_count_by_delta",
        "unified_full_accept_parent_frontier_mismatch_examples",
        "unified_full_accept_parent_active_seq_missing_count",
        "unified_full_accept_parent_request_id_mismatch_count",
        "unified_full_accept_parent_seq_id_mismatch_count",
        "unified_full_accept_parent_expected_frontier_len_source_counts",
        "generic_rolling_to_verify_equals_proposal_all",
        "generic_rolling_to_verify_mismatch_proposal_ids",
        "normal_proposal_generated_seq_ids",
        "normal_proposal_transfer_sent_seq_ids",
        "normal_proposal_transfer_received_seq_ids",
        "dual_proposal_buffer_store_seq_ids",
        "dual_proposal_buffer_discard_seq_ids",
        "dual_proposal_buffer_available_seq_ids_before_target_verify",
        "target_normal_verify_seq_ids_before_buffer_filter",
        "target_normal_verify_seq_ids_after_buffer_filter",
        "target_normal_verify_missing_buffer_seq_ids",
        "target_normal_verify_deferred_missing_buffer_seq_ids",
        "target_normal_verify_missing_buffer_reason_by_seq_id",
        "target_normal_verify_deferred_missing_buffer_reason_counts",
        "target_normal_verify_missing_buffer_details",
        "normal_proposal_buffer_illegal_discard_count",
        "normal_proposal_buffer_legal_consume_count_by_reason",
        "normal_proposal_buffer_event_order_violation_examples",
        "normal_proposal_buffer_event_dedup_count",
        "unified_ready_child_owner_cleared_seq_ids",
        "unified_ready_child_owner_clear_reason_by_seq_id",
        "unified_generic_target_verify_temp_append_used",
        "unified_generic_target_verify_num_proposals",
        "unified_generic_target_verify_num_to_verify_tokens",
        "unified_generic_target_verify_actual_verified_proposal_count",
        "unified_generic_target_verify_total_candidate_proposal_count",
        "unified_generic_target_verify_deferred_or_invalidated_proposal_count",
        "unified_generic_target_verify_token_count_denominator_source",
        "unified_generic_target_verify_token_count_denominator_count",
        "unified_generic_target_verify_input_ids_shape",
        "unified_generic_target_verify_logits_shape",
        "unified_generic_target_verify_logits_rows_per_proposal",
        "unified_generic_target_verify_logits_owner",
        "unified_generic_target_verify_logits_owner_record_count",
        "unified_generic_target_verify_owner_num_proposals",
        "unified_generic_target_verify_owner_num_to_verify_tokens",
        "unified_generic_target_verify_owner_input_ids_shape",
        "unified_generic_target_verify_owner_logits_shape",
        "unified_generic_target_verify_owner_uses_shifted_logits",
        "unified_generic_target_verify_owner_frontier_logits_available",
        "unified_generic_target_verify_owner_appended_logits_available",
        "unified_generic_target_verify_non_owner_record_count",
        "unified_generic_target_verify_non_owner_frontier_none_allowed_count",
        "unified_generic_target_verify_non_owner_raw_result_count",
        "unified_generic_target_verify_logits_owner_trace_present",
        "unified_generic_target_verify_checkpoint_failed_proposal_ids",
        "unified_generic_target_verify_rollback_len_mismatch_proposal_ids",
        "unified_generic_target_verify_input_mismatch_proposal_ids",
        "unified_generic_target_verify_next_round_mismatch_proposal_ids",
        "unified_generic_target_verify_sampled_input_ids_by_proposal_id",
        "unified_generic_target_verify_sampled_next_round_input_by_proposal_id",
        "unified_generic_target_verify_sampled_original_proposal_token_ids_by_proposal_id",
        "unified_generic_proposal_window_verify_enabled",
        "unified_generic_target_verify_current_window_accept_hist_by_depth",
        "unified_generic_target_verify_proposal_window_shadow_accept_hist_by_depth",
        "unified_generic_target_verify_current_window_all_first_token_reject",
        "unified_generic_target_verify_proposal_window_shadow_all_first_token_reject",
        "unified_generic_target_verify_proposal_window_shadow_has_nonzero_accept",
        "unified_generic_target_verify_proposal_window_shadow_full_accept_count",
        "unified_generic_target_verify_proposal_window_shadow_partial_accept_count",
        "unified_generic_target_verify_proposal_window_shadow_reject_count",
        "unified_generic_target_verify_uses_shifted_logits",
        "unified_generic_target_verify_frontier_logits_available",
        "unified_generic_target_verify_current_mapping_accept_hist_by_depth",
        "unified_generic_target_verify_shifted_mapping_accept_hist_by_depth",
        "unified_generic_target_verify_shifted_mapping_has_nonzero_accept",
        "unified_generic_target_verify_frontier_checkpoint_failed_proposal_ids",
        "unified_generic_target_verify_frontier_block_table_mismatch_proposal_ids",
        "unified_generic_target_verify_sampled_current_to_be_verified_by_proposal_id",
        "unified_generic_target_verify_sampled_proposal_window_to_be_verified_by_proposal_id",
        "unified_generic_target_verify_sampled_current_window_first_token_prob_by_proposal_id",
        "unified_generic_target_verify_sampled_proposal_window_first_token_prob_by_proposal_id",
        "unified_generic_target_verify_sampled_first_position_target_top5_tokens_by_proposal_id",
        "unified_candidate_budget_tokens_per_step",
        "unified_candidate_budget_used_tokens_per_step",
        "unified_candidate_budget_saturated_step_count",
        "unified_commit_budget_tokens_per_step",
        "unified_commit_budget_used_tokens_per_step",
        "unified_commit_budget_saturated_step_count",
        "unified_ready_but_not_committed_token_count",
        "unified_ready_but_not_committed_reason_counts",
        "unified_ready_but_not_committed_by_depth",
        "unified_commit_limited_by_token_budget_count",
        "unified_commit_limited_by_seq_budget_count",
        "unified_commit_limited_by_no_ready_parent_count",
        "unified_commit_limited_by_parent_not_full_accept_count",
        "unified_cascade_discard_count",
        "unified_cascade_discard_proposal_ids",
        "unified_cascade_discard_parent_proposal_ids",
        "unified_cascade_discard_reason_counts",
        "unified_invalidated_due_to_parent_not_full_accept_count_by_depth",
        "unified_invalidated_due_to_parent_not_full_accept_proposal_ids_by_depth",
        "unified_no_candidate_step_count",
        "unified_no_commit_step_count",
        "unified_candidate_step_reason_counts",
        "unified_no_commit_step_reason_counts",
        "unified_active_seq_count_by_step",
        "unified_ready_parent_count_by_step",
        "unified_committed_seq_count_by_step",
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
