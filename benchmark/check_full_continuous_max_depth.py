#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import as_int_map, as_int_set, int_value  # noqa: E402
from benchmark.check_bounded_rolling_readiness_audit import load_trace, synthetic_result_payload  # noqa: E402
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402


FULL_CONTINUOUS_INT_FIELDS = (
    "generic_full_continuous_max_depth",
    "generic_full_continuous_max_observed_depth",
    "generic_full_continuous_max_real_committed_depth",
    "generic_full_continuous_total_full_commit_token_count",
    "generic_full_continuous_total_partial_recovered_token_count",
    "generic_full_continuous_total_revised_token_count",
    "generic_full_continuous_total_output_token_count",
    "generic_full_continuous_depth_gt_max_real_commit_count",
    "generic_full_continuous_normal_lane_conflict_count",
    "generic_full_continuous_target_draft_mismatch_count",
)


def _bool_any(records: list[dict[str, Any]], *fields: str) -> bool:
    return any(bool(record.get(field, False)) for record in records for field in fields)


def _max_int(records: list[dict[str, Any]], field: str, default: int = 0) -> int:
    values = [int_value(record.get(field), default) for record in records if field in record]
    return max(values) if values else default


def _merge_int_map(records: list[dict[str, Any]], field: str) -> dict[int, int]:
    merged: dict[int, int] = {}
    for record in records:
        for key, value in as_int_map(record.get(field)).items():
            merged.setdefault(int(key), int(value))
    return merged


def _merge_int_set(records: list[dict[str, Any]], field: str) -> set[int]:
    merged: set[int] = set()
    for record in records:
        merged.update(as_int_set(record.get(field)))
    return merged


def _partial_recovery_depth_counts(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    recovered_ids = _merge_int_set(records, "partial_prefix_recovered_proposal_ids")
    depth_by_id = _merge_int_map(records, "partial_prefix_recovered_depth_by_proposal_id")
    recovered_by_id = _merge_int_map(records, "partial_prefix_committed_token_count_by_proposal_id")
    accepted_by_id = _merge_int_map(records, "partial_prefix_accepted_len_by_proposal_id")
    revised_by_id = _merge_int_map(records, "partial_prefix_revised_token_count_by_proposal_id")
    partial_counts: dict[str, int] = {}
    revised_counts: dict[str, int] = {}
    for proposal_id in sorted(recovered_ids):
        if proposal_id not in depth_by_id:
            continue
        depth = str(int(depth_by_id[proposal_id]))
        revised = int(revised_by_id.get(proposal_id, 0))
        recovered = int(
            recovered_by_id.get(
                proposal_id,
                int(accepted_by_id.get(proposal_id, 0)) + revised,
            )
        )
        partial_counts[depth] = int(partial_counts.get(depth, 0)) + recovered
        revised_counts[depth] = int(revised_counts.get(depth, 0)) + revised
    return (
        dict(sorted(partial_counts.items(), key=lambda item: int(item[0]))),
        dict(sorted(revised_counts.items(), key=lambda item: int(item[0]))),
    )


def _merge_depth_counts(records: list[dict[str, Any]], *fields: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for record in records:
        for field in fields:
            raw = record.get(field)
            if not isinstance(raw, dict):
                continue
            for key, value in raw.items():
                try:
                    depth = str(int(key))
                except Exception:
                    continue
                merged[depth] = max(int_value(value, 0), int_value(merged.get(depth), 0))
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def _merge_reason_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for record in records:
        raw = record.get(field)
        if not isinstance(raw, dict):
            continue
        for key, value in raw.items():
            merged[str(key)] = max(int_value(value, 0), int_value(merged.get(str(key)), 0))
    return dict(sorted(merged.items()))


def build_summary(records: list[dict[str, Any]], result_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result_payload = result_payload or {}
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(result_args, dict):
        result_args = {}
    accounting = aggregate_performance_accounting(records, result_payload)
    unified_authority = bool(
        _bool_any(records, "unified_generic_rolling_enabled", "enable_unified_generic_rolling_runtime")
        or bool(result_args.get("enable_unified_generic_rolling_runtime", False))
        or accounting.get("unified_generic_rolling_enabled", False)
        or accounting.get("enable_unified_generic_rolling_runtime", False)
    )
    generic_authority = bool(
        unified_authority
        or accounting.get("generic_accounting_mode", False)
        or accounting.get("unified_generic_rolling_enabled", False)
        or accounting.get("enable_unified_generic_rolling_runtime", False)
        or accounting.get("generic_full_continuous_enabled", False)
        or int_value(accounting.get("generic_full_continuous_total_output_token_count"), 0) > 0
        or int_value(accounting.get("unified_generic_total_output_token_count"), 0) > 0
    )
    accounting_partial_total = int_value(accounting.get("partial_prefix_total_recovered_token_count"), 0)
    accounting_partial_revised = int_value(accounting.get("partial_prefix_revised_token_count"), 0)
    partial_depth_counts, revised_depth_counts = _partial_recovery_depth_counts(records)

    summary: dict[str, Any] = {
        "generic_full_continuous_enabled": _bool_any(
            records,
            "generic_full_continuous_enabled",
            "enable_full_continuous_eager",
            "unified_generic_rolling_enabled",
            "enable_unified_generic_rolling_runtime",
        )
        or bool(result_args.get("enable_full_continuous_eager", False))
        or bool(result_args.get("enable_unified_generic_rolling_runtime", False)),
        "generic_rolling_runtime_enabled": _bool_any(
            records,
            "generic_rolling_runtime_enabled",
            "enable_generic_rolling_runtime_loop",
        )
        or bool(result_args.get("enable_generic_rolling_runtime_loop", False)),
        "generic_rolling_apply_path_enabled": _bool_any(
            records,
            "generic_rolling_apply_path_enabled",
            "enable_generic_rolling_apply_path",
        )
        or bool(result_args.get("enable_generic_rolling_apply_path", False)),
        "generic_full_continuous_parity_ok": all(
            bool(record.get("generic_full_continuous_parity_ok", True))
            for record in records
            if bool(record.get("generic_full_continuous_enabled", False))
            or bool(record.get("enable_full_continuous_eager", False))
            or bool(record.get("unified_generic_rolling_enabled", False))
            or bool(record.get("enable_unified_generic_rolling_runtime", False))
        ),
        "one_shot_committed_token_count": int_value(accounting.get("eager_committed_token_count"), 0)
        or _max_int(records, "eager_committed_token_count"),
        "combined_real_committed_token_count": (
            int_value(accounting.get("combined_real_committed_token_count"), 0)
            if generic_authority
            else max(
                int_value(accounting.get("combined_real_committed_token_count"), 0),
                _max_int(records, "combined_real_committed_token_count"),
            )
        ),
        "partial_prefix_accepted_token_count": int_value(
            accounting.get("partial_prefix_accepted_token_count"), 0
        )
        or _max_int(records, "partial_prefix_accepted_token_count"),
        "partial_prefix_revised_token_count": int_value(
            accounting.get("partial_prefix_revised_token_count"), 0
        )
        or _max_int(records, "partial_prefix_revised_token_count"),
        "partial_prefix_total_recovered_token_count": int_value(
            accounting.get("partial_prefix_total_recovered_token_count"), 0
        )
        or _max_int(records, "partial_prefix_total_recovered_token_count"),
        "descendant_committed_after_partial_count": int_value(
            accounting.get("descendant_committed_after_partial_count"), 0
        )
        or _max_int(records, "descendant_committed_after_partial_count"),
    }
    for field in FULL_CONTINUOUS_INT_FIELDS:
        summary[field] = _max_int(records, field)
    summary["generic_full_continuous_max_depth"] = max(
        int_value(summary.get("generic_full_continuous_max_depth"), 0),
        _max_int(records, "unified_generic_max_depth"),
        int_value(result_args.get("max_rolling_continuous_depth"), 0),
    )
    summary["generic_full_continuous_max_observed_depth"] = max(
        int_value(summary.get("generic_full_continuous_max_observed_depth"), 0),
        _max_int(records, "unified_generic_max_observed_depth"),
    )
    summary["generic_full_continuous_max_real_committed_depth"] = max(
        int_value(summary.get("generic_full_continuous_max_real_committed_depth"), 0),
        _max_int(records, "unified_generic_max_real_committed_depth"),
    )
    for generic_field, unified_field in (
        (
            "generic_full_continuous_total_full_commit_token_count",
            "unified_generic_total_full_commit_token_count",
        ),
        (
            "generic_full_continuous_total_partial_recovered_token_count",
            "unified_generic_total_partial_recovered_token_count",
        ),
        (
            "generic_full_continuous_total_revised_token_count",
            "unified_generic_total_revised_token_count",
        ),
        (
            "generic_full_continuous_total_output_token_count",
            "unified_generic_total_output_token_count",
        ),
        (
            "generic_full_continuous_depth_gt_max_real_commit_count",
            "unified_generic_depth_gt_max_real_commit_count",
        ),
        (
            "generic_full_continuous_normal_lane_conflict_count",
            "unified_generic_normal_lane_conflict_count",
        ),
        (
            "generic_full_continuous_target_draft_mismatch_count",
            "unified_generic_target_draft_mismatch_count",
        ),
    ):
        if unified_authority and "_total_" in generic_field:
            summary[generic_field] = int_value(accounting.get(generic_field), 0)
        else:
            summary[generic_field] = max(
                int_value(summary.get(generic_field), 0),
                _max_int(records, unified_field),
                int_value(accounting.get(generic_field), 0),
            )
    if accounting_partial_total:
        summary["generic_full_continuous_total_partial_recovered_token_count"] = accounting_partial_total
    if accounting_partial_revised:
        summary["generic_full_continuous_total_revised_token_count"] = accounting_partial_revised

    summary["generic_full_continuous_depth_commit_token_counts"] = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_commit_token_counts",
        "unified_generic_depth_commit_token_counts",
    )
    summary["generic_full_continuous_depth_commit_proposal_counts"] = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_commit_proposal_counts",
    )
    summary["generic_full_continuous_depth_candidate_token_counts"] = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_candidate_token_counts",
        "unified_generic_depth_candidate_token_counts",
    )
    summary["generic_full_continuous_depth_ready_token_counts"] = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_ready_token_counts",
        "unified_generic_depth_ready_token_counts",
    )
    fallback_partial_depth_counts = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_partial_recovered_token_counts",
        "unified_generic_depth_partial_recovered_token_counts",
    )
    fallback_revised_depth_counts = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_revised_token_counts",
        "unified_generic_depth_revised_token_counts",
    )
    summary["generic_full_continuous_depth_partial_recovered_token_counts"] = (
        partial_depth_counts or fallback_partial_depth_counts
    )
    summary["generic_full_continuous_depth_revised_token_counts"] = (
        revised_depth_counts or fallback_revised_depth_counts
    )
    summary["generic_full_continuous_depth_cascade_discard_counts"] = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_cascade_discard_counts",
    )
    summary["generic_full_continuous_stop_reason_counts"] = _merge_reason_counts(
        records,
        "generic_full_continuous_stop_reason_counts",
    )
    summary["unified_raw_target_verification_available"] = any(
        bool(record.get("unified_raw_target_verification_available", False)) for record in records
    )
    summary["unified_raw_verified_proposal_count_by_depth"] = _merge_depth_counts(
        records,
        "unified_raw_verified_proposal_count_by_depth",
    )
    summary["unified_raw_full_accept_proposal_count_by_depth"] = _merge_depth_counts(
        records,
        "unified_raw_full_accept_proposal_count_by_depth",
    )
    summary["unified_raw_invalidated_proposal_count_by_depth"] = _merge_depth_counts(
        records,
        "unified_raw_invalidated_proposal_count_by_depth",
    )
    return summary


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    *,
    require_depth_gt4_activity: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    summary = build_summary(records, result_payload)
    errors: list[str] = []
    enabled = bool(summary.get("generic_full_continuous_enabled", False))

    if not enabled:
        leaked = [
            field
            for field in FULL_CONTINUOUS_INT_FIELDS
            if field != "generic_full_continuous_max_depth" and int_value(summary.get(field), 0) != 0
        ]
        if leaked:
            errors.append(f"generic full continuous fields nonzero while disabled: {leaked}")
        return errors, summary

    if not summary.get("generic_rolling_runtime_enabled", False):
        errors.append("full continuous mode requires generic rolling runtime loop")
    if not summary.get("generic_rolling_apply_path_enabled", False):
        errors.append("full continuous mode requires generic rolling apply path")

    max_depth = int_value(summary.get("generic_full_continuous_max_depth"), 0)
    max_observed = int_value(summary.get("generic_full_continuous_max_observed_depth"), 0)
    max_real = int_value(summary.get("generic_full_continuous_max_real_committed_depth"), 0)
    if not (4 <= max_depth <= 100):
        errors.append("full continuous max depth must be configurable in range 4..100")
    if max_observed > max_depth:
        errors.append("max observed depth must not exceed configured max depth")
    if max_real > max_depth:
        errors.append("max real committed depth must not exceed configured max depth")
    raw_verified_total = sum(
        int_value(value, 0)
        for value in (summary.get("unified_raw_verified_proposal_count_by_depth") or {}).values()
    )
    raw_full_total = sum(
        int_value(value, 0)
        for value in (summary.get("unified_raw_full_accept_proposal_count_by_depth") or {}).values()
    )
    target_verified_no_full_accept = bool(
        summary.get("unified_raw_target_verification_available")
        and raw_verified_total > 0
        and raw_full_total == 0
    )
    if int_value(summary.get("generic_full_continuous_depth_gt_max_real_commit_count"), 0) != 0:
        errors.append("depth_gt_max real commit count must remain zero")

    depth_commit_counts = summary.get("generic_full_continuous_depth_commit_token_counts", {})
    depth_partial_counts = summary.get("generic_full_continuous_depth_partial_recovered_token_counts", {})
    depth_revised_counts = summary.get("generic_full_continuous_depth_revised_token_counts", {})
    full_sum = sum(int_value(value, 0) for value in depth_commit_counts.values())
    partial_sum = sum(int_value(value, 0) for value in depth_partial_counts.values())
    revised_sum = sum(int_value(value, 0) for value in depth_revised_counts.values())
    total_full = int_value(summary.get("generic_full_continuous_total_full_commit_token_count"), 0)
    total_partial = int_value(summary.get("generic_full_continuous_total_partial_recovered_token_count"), 0)
    total_revised = int_value(summary.get("generic_full_continuous_total_revised_token_count"), 0)
    total_output = int_value(summary.get("generic_full_continuous_total_output_token_count"), 0)
    if full_sum != total_full:
        errors.append(f"depth-indexed full commit sum mismatch: sum={full_sum} total={total_full}")
    if partial_sum != total_partial:
        errors.append(f"depth-indexed partial recovery sum mismatch: sum={partial_sum} total={total_partial}")
    if revised_sum != total_revised:
        errors.append(f"depth-indexed revised token sum mismatch: sum={revised_sum} total={total_revised}")
    if total_output != total_full + total_partial:
        errors.append("full continuous output tokens must equal full commits plus partial recovered tokens")

    partial_accepted = int_value(summary.get("partial_prefix_accepted_token_count"), 0)
    partial_revised = int_value(summary.get("partial_prefix_revised_token_count"), 0)
    partial_total = int_value(summary.get("partial_prefix_total_recovered_token_count"), 0)
    if partial_total and partial_total != partial_accepted + partial_revised:
        errors.append("partial recovered tokens must equal accepted prefix plus revised tokens")
    if total_partial and total_partial != partial_total:
        errors.append("full continuous partial total must match partial-prefix recovery total")
    if total_revised and total_revised != partial_revised:
        errors.append("full continuous revised total must match partial-prefix revised token count")

    expected_combined = total_output
    if int_value(summary.get("combined_real_committed_token_count"), 0) != expected_combined:
        errors.append(
            "combined real committed/output tokens must equal full continuous output"
        )
    if int_value(summary.get("generic_full_continuous_normal_lane_conflict_count"), 0) != 0:
        errors.append("full continuous normal lane conflict count must remain zero")
    if int_value(summary.get("generic_full_continuous_target_draft_mismatch_count"), 0) != 0:
        errors.append("full continuous target/draft mismatch count must remain zero")
    if int_value(summary.get("descendant_committed_after_partial_count"), 0) != 0:
        errors.append("descendant committed after partial recovery must remain zero")
    if max_observed < max_depth and not summary.get("generic_full_continuous_stop_reason_counts"):
        errors.append("stop reason counts must explain chains that stop before max depth")
    if not bool(summary.get("generic_full_continuous_parity_ok", False)):
        errors.append("generic_full_continuous_parity_ok must be true")
    if require_depth_gt4_activity:
        activity_depths: set[int] = set()
        for field in (
            "generic_full_continuous_depth_commit_token_counts",
            "generic_full_continuous_depth_candidate_token_counts",
            "generic_full_continuous_depth_ready_token_counts",
            "generic_full_continuous_depth_partial_recovered_token_counts",
            "generic_full_continuous_depth_revised_token_counts",
            "unified_raw_verified_proposal_count_by_depth",
            "unified_raw_invalidated_proposal_count_by_depth",
        ):
            raw = summary.get(field)
            if not isinstance(raw, dict):
                continue
            for key, value in raw.items():
                try:
                    depth = int(key)
                    count = int(value)
                except Exception:
                    continue
                if depth > 4 and count > 0:
                    activity_depths.add(depth)
        if max(activity_depths or {0}) <= 4 and max_observed <= 4 and not target_verified_no_full_accept:
            errors.append("full continuous mode must show real depth>4 candidate/ready/commit activity")

    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "generic_full_continuous_enabled",
        "generic_rolling_runtime_enabled",
        "generic_rolling_apply_path_enabled",
        "generic_full_continuous_max_depth",
        "generic_full_continuous_max_observed_depth",
        "generic_full_continuous_max_real_committed_depth",
        "generic_full_continuous_depth_commit_token_counts",
        "generic_full_continuous_depth_candidate_token_counts",
        "generic_full_continuous_depth_ready_token_counts",
        "generic_full_continuous_depth_partial_recovered_token_counts",
        "generic_full_continuous_depth_revised_token_counts",
        "generic_full_continuous_stop_reason_counts",
        "generic_full_continuous_total_full_commit_token_count",
        "generic_full_continuous_total_partial_recovered_token_count",
        "generic_full_continuous_total_revised_token_count",
        "generic_full_continuous_total_output_token_count",
        "generic_full_continuous_depth_gt_max_real_commit_count",
        "unified_raw_target_verification_available",
        "unified_raw_verified_proposal_count_by_depth",
        "unified_raw_full_accept_proposal_count_by_depth",
        "unified_raw_invalidated_proposal_count_by_depth",
        "generic_full_continuous_normal_lane_conflict_count",
        "generic_full_continuous_target_draft_mismatch_count",
        "generic_full_continuous_parity_ok",
        "partial_prefix_accepted_token_count",
        "partial_prefix_revised_token_count",
        "partial_prefix_total_recovered_token_count",
        "combined_real_committed_token_count",
    ):
        print(f"{key}={summary.get(key)}")


def _base_record(
    *,
    max_depth: int = 100,
    max_observed: int = 4,
    max_real: int = 4,
    depth_commit_counts: dict[str, int] | None = None,
    partial_counts: dict[str, int] | None = None,
    revised_counts: dict[str, int] | None = None,
    stop_reasons: dict[str, int] | None = None,
    one_shot: int = 12,
    combined: int | None = None,
    normal_conflict: int = 0,
    target_draft_mismatch: int = 0,
    depth_gt_max: int = 0,
    parity_ok: bool = True,
) -> dict[str, Any]:
    depth_commit_counts = depth_commit_counts or {"0": one_shot, "1": 8, "2": 8, "3": 8, "4": 8}
    if int(one_shot) > 0 and "0" not in depth_commit_counts:
        depth_commit_counts = {"0": int(one_shot), **depth_commit_counts}
    partial_counts = partial_counts or {}
    revised_counts = revised_counts or {}
    total_full = sum(depth_commit_counts.values())
    total_partial = sum(partial_counts.values())
    total_revised = sum(revised_counts.values())
    if combined is None:
        combined = total_full + total_partial
    return {
        "generic_full_continuous_enabled": True,
        "enable_full_continuous_eager": True,
        "generic_rolling_runtime_enabled": True,
        "enable_generic_rolling_runtime_loop": True,
        "generic_rolling_apply_path_enabled": True,
        "enable_generic_rolling_apply_path": True,
        "generic_full_continuous_max_depth": int(max_depth),
        "generic_full_continuous_max_observed_depth": int(max_observed),
        "generic_full_continuous_max_real_committed_depth": int(max_real),
        "generic_full_continuous_depth_commit_token_counts": dict(depth_commit_counts),
        "generic_full_continuous_depth_commit_proposal_counts": {
            depth: max(1, tokens // 4) for depth, tokens in depth_commit_counts.items() if tokens > 0
        },
        "generic_full_continuous_depth_candidate_token_counts": dict(depth_commit_counts),
        "generic_full_continuous_depth_ready_token_counts": dict(depth_commit_counts),
        "generic_full_continuous_depth_partial_recovered_token_counts": dict(partial_counts),
        "generic_full_continuous_depth_revised_token_counts": dict(revised_counts),
        "generic_full_continuous_depth_cascade_discard_counts": {depth: 1 for depth in partial_counts},
        "generic_full_continuous_stop_reason_counts": stop_reasons or {"no_eligible_ready_child": 1},
        "generic_full_continuous_total_full_commit_token_count": int(total_full),
        "generic_full_continuous_total_partial_recovered_token_count": int(total_partial),
        "generic_full_continuous_total_revised_token_count": int(total_revised),
        "generic_full_continuous_total_output_token_count": int(total_full + total_partial),
        "generic_full_continuous_depth_gt_max_real_commit_count": int(depth_gt_max),
        "generic_full_continuous_normal_lane_conflict_count": int(normal_conflict),
        "generic_full_continuous_target_draft_mismatch_count": int(target_draft_mismatch),
        "generic_full_continuous_parity_ok": bool(parity_ok),
        "eager_committed_token_count": int(one_shot),
        "combined_real_committed_token_count": int(combined),
        "partial_prefix_accepted_token_count": max(0, int(total_partial - total_revised)),
        "partial_prefix_revised_token_count": int(total_revised),
        "partial_prefix_total_recovered_token_count": int(total_partial),
        "descendant_committed_after_partial_count": 0,
    }


def _assert_pass(name: str, records: list[dict[str, Any]]) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic {name} failed: {errors}\nsummary={summary}")


def _assert_fail(name: str, records: list[dict[str, Any]], needle: str) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if not errors:
        raise SystemExit(f"synthetic {name} should fail\nsummary={summary}")
    if not any(needle in error for error in errors):
        raise SystemExit(f"synthetic {name} failed for wrong reason: {errors}\nsummary={summary}")


def run_synthetic() -> None:
    _assert_pass("max_depth=4 regression", [_base_record(max_depth=4, stop_reasons={"max_depth_reached": 1})])

    depth10 = {"1": 8, "2": 8, "3": 8, "4": 8, "5": 4, "10": 4}
    _assert_pass(
        "max_depth=100 full accept chain",
        [_base_record(max_depth=100, max_observed=10, max_real=10, depth_commit_counts=depth10)],
    )

    unified_stale_legacy = [
        _base_record(
            max_depth=8,
            max_observed=8,
            max_real=8,
            depth_commit_counts={str(depth): 8 for depth in range(1, 9)},
            stop_reasons={"max_depth_reached": 1},
            one_shot=0,
            combined=1244,
        )
    ]
    unified_stale_legacy[0].update(
        {
            "unified_generic_rolling_enabled": True,
            "enable_unified_generic_rolling_runtime": True,
            "unified_generic_max_depth": 8,
            "unified_generic_max_observed_depth": 8,
            "unified_generic_max_real_committed_depth": 8,
            "unified_generic_total_full_commit_token_count": 64,
            "unified_generic_total_partial_recovered_token_count": 0,
            "unified_generic_total_revised_token_count": 0,
            "unified_generic_total_output_token_count": 64,
            "unified_generic_depth_commit_token_counts": {str(depth): 8 for depth in range(1, 9)},
            "unified_generic_normal_lane_conflict_count": 0,
            "unified_generic_target_draft_mismatch_count": 0,
            "unified_generic_parity_ok": True,
        }
    )
    errors, summary = validate_records(unified_stale_legacy, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic unified stale legacy combined failed: {errors}\nsummary={summary}")
    if summary.get("combined_real_committed_token_count") != 64:
        raise SystemExit(f"synthetic unified stale legacy combined was not normalized: {summary}")

    depth100 = dict(depth10)
    depth100["100"] = 4
    _assert_pass(
        "stop at max_depth=100",
        [
            _base_record(
                max_depth=100,
                max_observed=100,
                max_real=100,
                depth_commit_counts=depth100,
                stop_reasons={"max_depth_reached": 1},
            )
        ],
    )

    _assert_pass(
        "partial at depth greater than 4",
        [
            _base_record(
                max_depth=100,
                max_observed=6,
                max_real=4,
                partial_counts={"5": 2},
                revised_counts={"5": 1},
                stop_reasons={"partial_recovery": 1},
            )
        ],
    )

    derived_partial_depth = [
        _base_record(
            max_depth=100,
            max_observed=6,
            max_real=4,
            partial_counts={"6": 2},
            revised_counts={"6": 1},
            stop_reasons={"partial_recovery": 2},
            combined=49,
        )
    ]
    derived_record = derived_partial_depth[0]
    derived_record["partial_prefix_recovered_proposal_ids"] = [980000601, 980000602]
    derived_record["partial_prefix_recovered_depth_by_proposal_id"] = {
        "980000601": 6,
        "980000602": 6,
    }
    derived_record["partial_prefix_accepted_len_by_proposal_id"] = {
        "980000601": 1,
        "980000602": 2,
    }
    derived_record["partial_prefix_revised_token_count_by_proposal_id"] = {
        "980000601": 1,
        "980000602": 1,
    }
    derived_record["partial_prefix_committed_token_count_by_proposal_id"] = {
        "980000601": 2,
        "980000602": 3,
    }
    derived_record["partial_prefix_accepted_token_count"] = 3
    derived_record["partial_prefix_revised_token_count"] = 2
    derived_record["partial_prefix_total_recovered_token_count"] = 5
    derived_record["generic_full_continuous_total_partial_recovered_token_count"] = 5
    derived_record["generic_full_continuous_total_revised_token_count"] = 2
    derived_record["generic_full_continuous_total_output_token_count"] = 49
    errors, summary = validate_records(derived_partial_depth, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic proposal-derived partial depth failed: {errors}\nsummary={summary}")
    if summary.get("generic_full_continuous_depth_partial_recovered_token_counts") != {"6": 5}:
        raise SystemExit(f"synthetic proposal-derived partial depth summary mismatch: {summary}")
    if summary.get("generic_full_continuous_depth_revised_token_counts") != {"6": 2}:
        raise SystemExit(f"synthetic proposal-derived revised depth summary mismatch: {summary}")

    _assert_pass(
        "reject at depth greater than 4",
        [
            _base_record(
                max_depth=100,
                max_observed=6,
                max_real=4,
                partial_counts={"5": 1},
                revised_counts={"5": 1},
                stop_reasons={"reject_recovery": 1},
            )
        ],
    )

    _assert_pass(
        "missing revised token fallback",
        [_base_record(max_depth=100, max_observed=5, max_real=4, stop_reasons={"partial_recovery_missing_revised_token": 1})],
    )

    bad_mismatch = [_base_record(max_depth=100, target_draft_mismatch=1, parity_ok=False)]
    _assert_fail("target/draft mismatch", bad_mismatch, "target/draft mismatch")

    bad_conflict = [_base_record(max_depth=100, normal_conflict=1, parity_ok=False)]
    _assert_fail("normal lane conflict", bad_conflict, "normal lane conflict")

    bad_depth = [_base_record(max_depth=100, max_observed=101, max_real=101, depth_gt_max=1, parity_ok=False)]
    _assert_fail("depth beyond max", bad_depth, "max observed depth")

    bad_output_total = [_base_record(max_depth=100, partial_counts={"5": 2}, revised_counts={"5": 1})]
    bad_output_total[0]["generic_full_continuous_total_output_token_count"] = 45
    _assert_fail("bad output total", bad_output_total, "output tokens")

    bad_descendant = [_base_record(max_depth=100, partial_counts={"5": 2}, revised_counts={"5": 1})]
    bad_descendant[0]["descendant_committed_after_partial_count"] = 1
    _assert_fail("descendant committed after partial", bad_descendant, "descendant committed")

    disabled_leak = [deepcopy(_base_record())]
    disabled_leak[0]["generic_full_continuous_enabled"] = False
    disabled_leak[0]["enable_full_continuous_eager"] = False
    _assert_fail("disabled full continuous leakage", disabled_leak, "nonzero while disabled")

    errors, summary = validate_records(
        [_base_record(max_depth=100, max_observed=4, max_real=4)],
        synthetic_result_payload(),
        require_depth_gt4_activity=True,
    )
    if not any("depth>4" in error for error in errors):
        raise SystemExit(f"synthetic require depth>4 should fail\nerrors={errors}\nsummary={summary}")

    target_reject_early = [
        _base_record(
            max_depth=8,
            max_observed=1,
            max_real=0,
            depth_commit_counts={"1": 0},
            one_shot=0,
            combined=0,
            stop_reasons={"parent_not_full_accept": 1},
        )
    ]
    target_reject_early[0].update(
        {
            "unified_generic_rolling_enabled": True,
            "enable_unified_generic_rolling_runtime": True,
            "unified_generic_max_depth": 8,
            "unified_generic_max_observed_depth": 1,
            "unified_generic_max_real_committed_depth": 0,
            "unified_raw_target_verification_available": True,
            "unified_raw_verified_proposal_count_by_depth": {"1": 180},
            "unified_raw_full_accept_proposal_count_by_depth": {},
            "unified_raw_reject_proposal_count_by_depth": {"1": 180},
            "generic_full_continuous_depth_candidate_token_counts": {"1": 720},
            "generic_full_continuous_depth_ready_token_counts": {"1": 720},
            "generic_full_continuous_parity_ok": True,
            "generic_full_continuous_total_full_commit_token_count": 0,
            "generic_full_continuous_total_output_token_count": 0,
        }
    )
    errors, summary = validate_records(
        target_reject_early,
        synthetic_result_payload(),
        require_depth_gt4_activity=True,
    )
    if errors:
        raise SystemExit(f"synthetic target-verified early reject should pass: {errors}\nsummary={summary}")

    errors, summary = validate_records(
        [_base_record(max_depth=100, max_observed=5, max_real=5, depth_commit_counts={"1": 8, "2": 8, "3": 8, "4": 8, "5": 4})],
        synthetic_result_payload(),
        require_depth_gt4_activity=True,
    )
    if errors:
        raise SystemExit(f"synthetic require depth>4 should pass: {errors}\nsummary={summary}")

    print("Synthetic full continuous max-depth checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8u full continuous max-depth trace fields.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--require-depth-gt4-activity", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic()
        return 0

    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result is not None else {}
    errors, summary = validate_records(
        records,
        result_payload,
        require_depth_gt4_activity=bool(args.require_depth_gt4_activity),
    )
    print_summary(summary)
    if errors:
        print("check_status=fail")
        print(json.dumps({"errors": errors}, indent=2, sort_keys=True))
        return 1
    print("check_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
