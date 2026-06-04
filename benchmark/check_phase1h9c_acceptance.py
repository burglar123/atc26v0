#!/usr/bin/env python3
"""Phase 1H-9C acceptance checker.

This is the dedicated hard gate for Phase 1H-9C.  It checks the
single-child-ahead unified rolling invariants and reports accounting scopes
separately.  Historical full-continuous/eager accounting checkers remain useful
diagnostics, but they intentionally are not 9C hard gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import int_value  # noqa: E402
from benchmark.check_unified_generic_rolling_runtime import (  # noqa: E402
    build_summary as build_unified_summary,
    load_json,
    load_trace,
    synthetic_single_child_payload,
    synthetic_single_child_records,
)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "enabled"}
    return bool(value)


def float_value(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("traces", "requests", "request_traces", "request_summaries"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def result_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    overall = metrics.get("overall") if isinstance(metrics.get("overall"), dict) else {}
    merged = dict(metrics)
    merged.update({f"overall.{key}": value for key, value in overall.items()})
    for key, value in overall.items():
        merged.setdefault(key, value)
    return merged


def infer_total_output_tokens(payload: dict[str, Any]) -> int:
    metrics = result_metrics(payload)
    value = first_present(
        metrics,
        "overall.total_output_tokens",
        "total_output_tokens",
        "num_output_tokens",
    )
    if value is not None:
        return int_value(value, 0)
    total = 0
    for row in result_rows(payload):
        row_value = first_present(
            row,
            "num_decode_output_tokens",
            "num_output_tokens",
            "num_completion_tokens",
            "completion_tokens",
            "output_tokens",
            "num_generated_tokens",
            "num_tokens",
        )
        if isinstance(row_value, list):
            total += len(row_value)
        else:
            total += int_value(row_value, 0)
    return total


def cached_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metrics = payload.get("metrics") if isinstance(payload, dict) else {}
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if (
                key.startswith("cached_admission_")
                or key.startswith("cached_cache_build_")
                or key.startswith("cached_kv_")
                or key == "cached_prefill_mode"
            ):
                summary[key] = value
        nested = metrics.get("cached_admission")
        if isinstance(nested, dict):
            summary.update(nested)
    top = payload.get("cached_admission") if isinstance(payload, dict) else {}
    if isinstance(top, dict):
        summary.update(top)
    return summary


def sum_depth_values(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(int_value(item, 0) for item in value.values())


def has_positive_depth(value: Any, depth: int) -> bool:
    if not isinstance(value, dict):
        return False
    return int_value(value.get(str(depth), value.get(depth)), 0) > 0


def positive_depths(value: Any) -> list[int]:
    if not isinstance(value, dict):
        return []
    depths: list[int] = []
    for key, item in value.items():
        try:
            depth = int(key)
        except Exception:
            continue
        if int_value(item, 0) > 0:
            depths.append(depth)
    return sorted(depths)


def all_zero_depth_counts(summary: dict[str, Any], field: str) -> bool:
    raw = summary.get(field)
    if not isinstance(raw, dict):
        return True
    return all(int_value(value, 0) == 0 for value in raw.values())


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def add_warning(warnings: list[str], condition: bool, message: str) -> None:
    if not condition:
        warnings.append(message)


def validate_result_sanity(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    total_output_tokens = infer_total_output_tokens(payload)
    rows = result_rows(payload)
    arrival_after_finish = 0
    elapsed_mismatch = 0
    tpot_mismatch = 0
    for row in rows:
        arrival_ts = float_value(row.get("arrival_ts"))
        finish_ts = float_value(first_present(row, "finish_ts", "finished_ts", "end_ts", "end_time"))
        decode_start_ts = float_value(
            first_present(row, "decode_start_ts", "decoding_start_ts", "first_decode_ts", "first_token_ts")
        )
        decode_elapsed_ms = float_value(row.get("decode_elapsed_ms"))
        observed_tpot_ms = float_value(row.get("observed_tpot_ms"))
        tokens = int_value(
            first_present(
                row,
                "num_decode_output_tokens",
                "num_output_tokens",
                "num_completion_tokens",
                "completion_tokens",
                "output_tokens",
            ),
            0,
        )
        if arrival_ts is not None and finish_ts is not None and arrival_ts > finish_ts:
            arrival_after_finish += 1
        if finish_ts is not None and decode_start_ts is not None and decode_elapsed_ms is not None:
            expected = (finish_ts - decode_start_ts) * 1000.0
            if abs(expected - decode_elapsed_ms) > 1e-3:
                elapsed_mismatch += 1
        if tokens > 0 and decode_elapsed_ms is not None and observed_tpot_ms is not None:
            expected = decode_elapsed_ms / tokens
            if abs(expected - observed_tpot_ms) > 1e-3:
                tpot_mismatch += 1

    add_error(errors, total_output_tokens > 0, "result total_output_tokens must be > 0")
    add_error(errors, arrival_after_finish == 0, "result has arrival_ts > finish_ts rows")
    add_error(errors, elapsed_mismatch == 0, "result has decode_elapsed_ms timestamp mismatches")
    add_error(errors, tpot_mismatch == 0, "result has observed_tpot_ms arithmetic mismatches")
    return {
        "total_output_tokens": total_output_tokens,
        "result_row_count": len(rows),
        "arrival_after_finish": arrival_after_finish,
        "elapsed_mismatch": elapsed_mismatch,
        "tpot_mismatch": tpot_mismatch,
    }


def validate_cached_admission(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    summary = cached_summary(payload)
    rows = result_rows(payload)
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    enabled = bool_value(summary.get("cached_admission_enabled")) or bool_value(
        args.get("enable_cached_admission")
    ) or bool_value(
        args.get("cached_admission_enabled")
    )
    total_requests = int_value(summary.get("cached_admission_total_requests"), len(rows))
    total_arrived = int_value(summary.get("cached_admission_total_arrived"), len(rows))
    total_admitted = int_value(summary.get("cached_admission_total_admitted"), 0)
    total_completed = int_value(summary.get("cached_admission_total_completed"), 0)

    add_error(errors, enabled, "cached_admission_enabled must be true")
    add_error(
        errors,
        total_arrived == total_admitted == total_completed,
        "cached admission must have total_arrived == total_admitted == total_completed",
    )
    if total_requests:
        add_error(errors, total_arrived <= total_requests, "cached admission total_arrived exceeds total_requests")

    cached_kv_num_requests = int_value(summary.get("cached_kv_num_requests"), -1)
    if cached_kv_num_requests >= 0 and total_requests:
        add_error(
            errors,
            cached_kv_num_requests == total_requests
            or summary.get("cached_kv_num_requests_explanation") is not None,
            "cached KV build request count is incomplete",
        )
    for row in rows:
        request_id = row.get("request_id")
        if "cached_kv_ready" in row:
            add_error(errors, bool_value(row.get("cached_kv_ready")), f"request_id={request_id} cached_kv_ready=false")
        if "cached_kv_materialized" in row:
            add_error(
                errors,
                bool_value(row.get("cached_kv_materialized")),
                f"request_id={request_id} cached_kv_materialized=false",
            )
    for key, value in summary.items():
        if key.endswith("_cache_build_complete") or key.endswith("_cached_kv_ready"):
            add_error(errors, bool_value(value), f"{key} is false")

    return {
        "cached_admission_enabled": enabled,
        "cached_admission_total_requests": total_requests,
        "cached_admission_total_arrived": total_arrived,
        "cached_admission_total_admitted": total_admitted,
        "cached_admission_total_completed": total_completed,
        "cached_kv_num_requests": cached_kv_num_requests,
    }


def validate_unified_9c(
    summary: dict[str, Any],
    result_info: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    require_depth: int,
    tiny_output_threshold: int,
) -> None:
    add_error(errors, bool_value(summary.get("unified_generic_rolling_enabled")), "unified generic rolling must be enabled")
    add_error(errors, bool_value(summary.get("unified_single_child_ahead_enabled")), "single-child-ahead must be enabled")
    add_error(
        errors,
        int_value(summary.get("unified_max_unverified_depth_ahead"), -1) == 1,
        "unified_max_unverified_depth_ahead must be 1",
    )

    add_error(
        errors,
        int_value(summary.get("unified_unverified_depth_ahead_max_observed"), 0) <= 1,
        "unverified depth ahead max observed must be <= 1",
    )
    for field, message in (
        ("unified_single_child_ahead_violation_count", "single-child ahead violation count must be 0"),
        (
            "unified_generated_grandchild_before_parent_verified_count",
            "generated-grandchild-before-parent-verified count must be 0",
        ),
        (
            "unified_child_generated_in_same_burst_as_grandchild_violation_count",
            "same-burst child/grandchild violation count must be 0",
        ),
    ):
        add_error(errors, int_value(summary.get(field), 0) == 0, message)

    add_error(
        errors,
        bool_value(summary.get("unified_raw_target_verification_available")),
        "target verification must be available",
    )
    add_error(
        errors,
        str(summary.get("unified_raw_verification_source") or "") == "target_verify_result",
        "unified_raw_verification_source must be target_verify_result",
    )
    add_error(
        errors,
        bool_value(summary.get("unified_generic_target_verify_owner_uses_shifted_logits")),
        "target verify owner must use shifted logits",
    )
    for field in (
        "unified_generic_target_verify_checkpoint_failed_proposal_ids",
        "unified_generic_target_verify_rollback_len_mismatch_proposal_ids",
        "unified_generic_target_verify_input_mismatch_proposal_ids",
        "unified_generic_target_verify_next_round_mismatch_proposal_ids",
        "unified_generic_target_verify_frontier_checkpoint_failed_proposal_ids",
        "unified_generic_target_verify_frontier_block_table_mismatch_proposal_ids",
    ):
        add_error(errors, not summary.get(field), f"{field} must be empty")

    candidate_depths = set(int_value(depth) for depth in summary.get("candidate_depths", []) or [])
    ready_depths = set(int_value(depth) for depth in summary.get("ready_depths", []) or [])
    committed_depths = set(int_value(depth) for depth in summary.get("committed_depths", []) or [])
    max_real = int_value(summary.get("max_real_committed_depth"), 0)
    total_output_tokens = int_value(result_info.get("total_output_tokens"), 0)
    add_error(errors, 2 in candidate_depths, "candidate_depths must include depth 2")
    add_error(errors, 2 in ready_depths, "ready_depths must include depth 2")
    if 2 not in committed_depths:
        if total_output_tokens <= tiny_output_threshold:
            warnings.append("committed_depths does not include depth 2; treated as tiny workload")
        else:
            errors.append("committed_depths must include depth 2 for normal 9C runs")
    add_error(errors, max_real >= require_depth, f"max_real_committed_depth must be >= {require_depth}")

    for field, message in (
        (
            "unified_child_verified_before_parent_full_accept_violation_count",
            "child verified before parent full accept violation count must be 0",
        ),
        (
            "unified_child_parent_full_accept_guard_violation_count",
            "child parent full-accept guard violation count must be 0",
        ),
    ):
        add_error(errors, int_value(summary.get(field), 0) == 0, message)
    for field in (
        "unified_child_generated_from_non_full_parent_count_by_depth",
        "unified_child_generated_from_unverified_parent_count_by_depth",
        "unified_child_generated_from_partial_parent_count_by_depth",
        "unified_child_generated_from_reject_parent_count_by_depth",
        "unified_child_generated_from_invalidated_parent_count_by_depth",
    ):
        add_error(errors, all_zero_depth_counts(summary, field), f"{field} must be empty/zero")
    add_error(
        errors,
        has_positive_depth(summary.get("unified_child_target_verified_after_promotion_count_by_depth"), 2),
        "depth2 child target-verified-after-promotion count must be > 0",
    )

    add_error(
        errors,
        int_value(summary.get("normal_proposal_buffer_illegal_discard_count"), 0) == 0,
        "normal proposal buffer illegal discard count must be 0",
    )
    add_error(
        errors,
        int_value(summary.get("unified_ready_child_normal_verify_exclusion_mismatch_count"), 0) == 0,
        "ready-child normal verify exclusion mismatch count must be 0",
    )
    add_error(
        errors,
        sum_depth_values(summary.get("unified_child_schedule_state_error_count_by_depth")) == 0,
        "unified child schedule state error count must be 0",
    )


def validate_accounting_scope(summary: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    event_full = int_value(summary.get("unified_applied_event_total_full_token_count"), 0)
    event_partial = int_value(summary.get("unified_applied_event_total_partial_recovered_token_count"), 0)
    event_revised = int_value(summary.get("unified_applied_event_total_revised_token_count"), 0)
    event_output = int_value(summary.get("unified_applied_event_total_output_token_count"), 0)
    event_available = bool(
        event_output
        or summary.get("unified_applied_event_total_output_token_count_by_depth")
        or summary.get("unified_applied_event_full_token_count_by_depth")
    )
    if event_available:
        add_error(
            errors,
            sum_depth_values(summary.get("unified_applied_event_full_token_count_by_depth")) == event_full,
            "proposal-event full depth sum must match total",
        )
        add_error(
            errors,
            sum_depth_values(summary.get("unified_applied_event_partial_recovered_token_count_by_depth"))
            == event_partial,
            "proposal-event partial depth sum must match total",
        )
        add_error(
            errors,
            sum_depth_values(summary.get("unified_applied_event_partial_revised_token_count_by_depth")) == event_revised,
            "proposal-event revised depth sum must match total",
        )
        add_error(
            errors,
            sum_depth_values(summary.get("unified_applied_event_total_output_token_count_by_depth")) == event_output,
            "proposal-event output depth sum must match total",
        )
        add_error(errors, event_full + event_partial == event_output, "proposal-event full + partial must equal output")

    final_output = int_value(summary.get("unified_final_output_token_count"), 0)
    final_by_depth = summary.get("unified_final_total_output_token_count_by_depth")
    final_by_depth_available = isinstance(final_by_depth, dict) and bool(final_by_depth)
    if final_by_depth_available:
        add_error(errors, sum_depth_values(final_by_depth) == final_output, "final-output depth sum must match total")

    return {
        "proposal_event_accounting_available": event_available,
        "proposal_event_output_token_count": event_output,
        "final_output_token_count": final_output,
        "final_output_by_depth_available": final_by_depth_available,
        "cross_scope_comparison_skipped_reason": (
            None if final_by_depth_available else "final_output_by_depth_not_instrumented"
        ),
    }


def performance_info(result_payload: dict[str, Any], unified_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = result_metrics(result_payload)
    return {
        "goodput_tokens_per_s": first_present(metrics, "goodput_tokens_per_s", "overall.goodput_tokens_per_s"),
        "wall_decode_tokens_per_s": first_present(
            metrics,
            "wall_decode_tokens_per_s",
            "overall.wall_decode_tokens_per_s",
        ),
        "mean_tpot_ms": first_present(metrics, "mean_tpot_ms", "overall.mean_tpot_ms"),
        "max_real_committed_depth": unified_summary.get("max_real_committed_depth"),
        "candidate_counts_by_depth": unified_summary.get("unified_raw_candidate_proposal_count_by_depth"),
        "verified_counts_by_depth": unified_summary.get("unified_raw_verified_proposal_count_by_depth"),
        "committed_counts_by_depth": unified_summary.get("unified_raw_committed_proposal_count_by_depth"),
        "candidate_waste_ratio_by_depth": unified_summary.get("unified_candidate_waste_ratio_by_depth"),
        "total_control_payload_len_units": unified_summary.get("total_control_payload_len_units"),
        "total_control_payload_bytes": unified_summary.get("total_control_payload_bytes"),
        "zero_candidate_meta_broadcast_count": unified_summary.get("zero_candidate_meta_broadcast_count"),
    }


def validate_phase1h9c(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any],
    *,
    require_depth: int,
    tiny_output_threshold: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    unified_summary = build_unified_summary(records, result_payload)
    result_info = validate_result_sanity(result_payload, errors)
    cached_info = validate_cached_admission(result_payload, errors)
    validate_unified_9c(
        unified_summary,
        result_info,
        errors,
        warnings,
        require_depth=require_depth,
        tiny_output_threshold=tiny_output_threshold,
    )
    accounting_info = validate_accounting_scope(unified_summary, errors)
    summary = {
        **result_info,
        **cached_info,
        **performance_info(result_payload, unified_summary),
        **accounting_info,
        "candidate_depths": unified_summary.get("candidate_depths"),
        "ready_depths": unified_summary.get("ready_depths"),
        "committed_depths": unified_summary.get("committed_depths"),
        "unified_unverified_depth_ahead_max_observed": unified_summary.get(
            "unified_unverified_depth_ahead_max_observed"
        ),
        "unified_child_target_verified_after_promotion_count_by_depth": unified_summary.get(
            "unified_child_target_verified_after_promotion_count_by_depth"
        ),
        "normal_proposal_buffer_illegal_discard_count": unified_summary.get(
            "normal_proposal_buffer_illegal_discard_count"
        ),
        "unified_ready_child_normal_verify_exclusion_mismatch_count": unified_summary.get(
            "unified_ready_child_normal_verify_exclusion_mismatch_count"
        ),
        "unified_child_schedule_state_error_count_by_depth": unified_summary.get(
            "unified_child_schedule_state_error_count_by_depth"
        ),
    }
    return errors, warnings, summary


def print_table(summary: dict[str, Any], warnings: list[str], errors: list[str]) -> None:
    rows = [
        ("total_output_tokens", summary.get("total_output_tokens")),
        ("goodput_tokens_per_s", summary.get("goodput_tokens_per_s")),
        ("wall_decode_tokens_per_s", summary.get("wall_decode_tokens_per_s")),
        ("mean_tpot_ms", summary.get("mean_tpot_ms")),
        ("max_real_committed_depth", summary.get("max_real_committed_depth")),
        ("candidate_depths", summary.get("candidate_depths")),
        ("ready_depths", summary.get("ready_depths")),
        ("committed_depths", summary.get("committed_depths")),
        ("candidate_counts_by_depth", summary.get("candidate_counts_by_depth")),
        ("verified_counts_by_depth", summary.get("verified_counts_by_depth")),
        ("committed_counts_by_depth", summary.get("committed_counts_by_depth")),
        ("waste_ratio_by_depth", summary.get("candidate_waste_ratio_by_depth")),
        ("child_verified_after_promotion", summary.get("unified_child_target_verified_after_promotion_count_by_depth")),
        ("proposal_event_output", summary.get("proposal_event_output_token_count")),
        ("final_output", summary.get("final_output_token_count")),
        ("final_by_depth_available", summary.get("final_output_by_depth_available")),
        ("cross_scope_skip_reason", summary.get("cross_scope_comparison_skipped_reason")),
        ("total_control_payload_len_units", summary.get("total_control_payload_len_units")),
        ("total_control_payload_bytes", summary.get("total_control_payload_bytes")),
        ("zero_candidate_meta_broadcast_count", summary.get("zero_candidate_meta_broadcast_count")),
    ]
    print("Phase 1H-9C acceptance summary")
    print("--------------------------------")
    for key, value in rows:
        print(f"{key:40} {value}")
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nerrors:")
        for error in errors:
            print(f"- {error}")
    print(f"\nphase1h9c_acceptance={'fail' if errors else 'pass'}")


def synthetic_case() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = synthetic_single_child_records(
        [[1], [2]],
        committed_depths={1, 2},
        parent_outcomes_by_depth={1: "full", 2: "full"},
        promoted_child_depths={2},
        scheduled_child_depths={2},
        target_verified_after_promotion_depths={2},
    )
    records[0]["unified_generic_target_verify_owner_uses_shifted_logits"] = True
    records[0]["unified_generic_target_verify_logits_owner"] = True
    records[0]["unified_generic_target_verify_uses_shifted_logits"] = True
    records[0]["unified_generic_target_verify_temp_append_used"] = True
    records[0]["unified_raw_target_verification_available"] = True
    records[0]["unified_raw_verification_source"] = "target_verify_result"
    payload = synthetic_single_child_payload()
    payload["metrics"]["overall"] = {
        "total_output_tokens": 8,
        "goodput_tokens_per_s": 100.0,
        "wall_decode_tokens_per_s": 120.0,
        "mean_tpot_ms": 10.0,
    }
    payload["cached_admission"] = {
        "cached_admission_enabled": True,
        "cached_admission_total_requests": 1,
        "cached_admission_total_arrived": 1,
        "cached_admission_total_admitted": 1,
        "cached_admission_total_completed": 1,
        "cached_kv_num_requests": 1,
    }
    payload["traces"] = [
        {
            "request_id": "r0",
            "arrival_ts": 1.0,
            "decode_start_ts": 1.1,
            "finish_ts": 1.2,
            "decode_elapsed_ms": 100.0,
            "observed_tpot_ms": 12.5,
            "num_decode_output_tokens": 8,
            "cached_kv_ready": True,
            "cached_kv_materialized": True,
        }
    ]
    return records, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--require-depth", type=int, default=2)
    parser.add_argument("--tiny-output-threshold", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic:
        records, result_payload = synthetic_case()
    else:
        if args.trace is None or args.result is None:
            raise SystemExit("TRACE and RESULT paths are required unless --synthetic is set")
        records = load_trace(args.trace)
        result_payload = load_json(args.result)
    errors, warnings, summary = validate_phase1h9c(
        records,
        result_payload,
        require_depth=max(1, int(args.require_depth)),
        tiny_output_threshold=max(0, int(args.tiny_output_threshold)),
    )
    print_table(summary, warnings, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
