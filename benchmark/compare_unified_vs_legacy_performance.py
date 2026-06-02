#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
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
    trace_payload_to_records,
)
from benchmark.check_unified_generic_rolling_runtime import build_summary as build_unified_summary  # noqa: E402


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_trace(path: Path) -> list[dict[str, Any]]:
    return trace_payload_to_records(json.loads(path.read_text(encoding="utf-8")))


def first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def result_rows(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result_payload.get("traces", []) if isinstance(result_payload, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def row_float(row: dict[str, Any], *keys: str) -> float | None:
    value = first_present(row, tuple(keys))
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def queue_wait_ms(row: dict[str, Any]) -> float | None:
    explicit = row_float(
        row,
        "queue_wait_ms",
        "admission_queue_wait_ms",
        "decode_queue_wait_ms",
        "queue_wait_time_ms",
    )
    if explicit is not None:
        return explicit
    arrival_ts = row_float(row, "arrival_ts")
    decode_start_ts = row_float(
        row,
        "decode_start_ts",
        "decoding_start_ts",
        "first_decode_ts",
        "first_token_ts",
        "start_decode_ts",
    )
    if arrival_ts is None or decode_start_ts is None:
        return None
    return max(0.0, (decode_start_ts - arrival_ts) * 1000.0)


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = sorted(value.items(), key=lambda item: str(item[0]))[:4]
        return "{" + ", ".join(f"{key}:{format_value(item)}" for key, item in items) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value[:6]) + (", ..." if len(value) > 6 else "") + "]"
    if value is None:
        return ""
    return str(value)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def table_delta(legacy_value: Any, unified_value: Any) -> tuple[str, str]:
    if not is_number(legacy_value) or not is_number(unified_value):
        return "", ""
    delta = float(unified_value) - float(legacy_value)
    ratio = safe_div(float(unified_value), float(legacy_value))
    return format_value(delta), format_value(ratio)


def print_compare_table(title: str, rows: list[tuple[str, Any, Any]]) -> None:
    rendered: list[tuple[str, str, str, str, str]] = []
    for metric, legacy_value, unified_value in rows:
        delta, ratio = table_delta(legacy_value, unified_value)
        rendered.append(
            (
                metric,
                format_value(legacy_value),
                format_value(unified_value),
                delta,
                ratio,
            )
        )
    headers = ("metric", "legacy", "unified", "delta", "ratio")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered)) if rendered else len(headers[index])
        for index in range(len(headers))
    ]
    print(f"\n== {title} ==")
    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rendered:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(row))))


def print_text_table(title: str, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    rendered = [tuple(format_value(item) for item in row) for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered)) if rendered else len(headers[index])
        for index in range(len(headers))
    ]
    print(f"\n== {title} ==")
    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rendered:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(row))))


def cached_summary(result_payload: dict[str, Any], accounting: dict[str, Any]) -> dict[str, Any]:
    cached = result_payload.get("cached_admission", {}) if isinstance(result_payload, dict) else {}
    cached = cached if isinstance(cached, dict) else {}
    metrics = result_payload.get("metrics", {}) if isinstance(result_payload, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "cached_admission_total_requests": int_value(cached.get("cached_admission_total_requests"), 0),
        "cached_admission_arrived_count": int_value(
            first_present(cached, ("cached_admission_arrived_count", "cached_admission_total_arrived")),
            0,
        ),
        "cached_admission_admitted_count": int_value(
            first_present(cached, ("cached_admission_admitted_count", "cached_admission_total_admitted")),
            0,
        ),
        "cached_admission_completed_count": int_value(
            first_present(cached, ("cached_admission_completed_count", "cached_admission_total_completed")),
            0,
        ),
        "cached_admission_peak_active": int_value(cached.get("cached_admission_peak_active"), 0),
        "cached_admission_queue_wait_mean_ms": float_value(
            first_present(cached, ("cached_admission_queue_wait_mean_ms", "cached_admission_mean_queue_wait_ms")),
            0.0,
        ),
        "cached_admission_queue_wait_p50_ms": float_value(
            first_present(cached, ("cached_admission_queue_wait_p50_ms", "cached_admission_p50_queue_wait_ms")),
            0.0,
        ),
        "cached_admission_queue_wait_p90_ms": float_value(
            first_present(cached, ("cached_admission_queue_wait_p90_ms", "cached_admission_p90_queue_wait_ms")),
            0.0,
        ),
        "cached_admission_queue_wait_p99_ms": float_value(
            first_present(cached, ("cached_admission_queue_wait_p99_ms", "cached_admission_p99_queue_wait_ms")),
            0.0,
        ),
        "cached_admission_decode_only_elapsed_s": float_value(
            first_present(
                metrics,
                ("cached_admission_decode_only_elapsed_s",),
                cached.get("cached_admission_decode_only_elapsed_s"),
            ),
            float_value(accounting.get("cached_admission_decode_only_elapsed_s"), 0.0),
        ),
        "cached_cache_build_elapsed_s": float_value(cached.get("cached_cache_build_elapsed_s"), 0.0),
    }


def summarize_case(result_path: Path, trace_path: Path) -> dict[str, Any]:
    result_payload = load_json(result_path)
    records = load_trace(trace_path)
    accounting = aggregate_performance_accounting(records, result_payload)
    unified_summary = build_unified_summary(records, result_payload)
    rows = result_rows(result_payload)
    decode_elapsed = [
        value for row in rows if (value := row_float(row, "decode_elapsed_ms")) is not None
    ]
    observed_tpot = [
        value for row in rows if (value := row_float(row, "observed_tpot_ms")) is not None
    ]
    queue_waits = [value for row in rows if (value := queue_wait_ms(row)) is not None]
    cached = cached_summary(result_payload, accounting)
    total_output_tokens = int_value(accounting.get("total_output_tokens"), 0)
    engine_elapsed_s = float_value(accounting.get("engine_elapsed_s"), 0.0)
    decode_only_elapsed_s = float_value(cached.get("cached_admission_decode_only_elapsed_s"), 0.0)
    accounting.setdefault("wall_engine_tokens_per_s", safe_div(total_output_tokens, engine_elapsed_s))
    accounting.setdefault("wall_decode_tokens_per_s", safe_div(total_output_tokens, decode_only_elapsed_s))
    return {
        "result": result_payload,
        "records": records,
        "accounting": accounting,
        "unified": unified_summary,
        "cached": cached,
        "derived": {
            "decode_elapsed_ms_median": median(decode_elapsed),
            "decode_elapsed_ms_p90": percentile(decode_elapsed, 0.90),
            "observed_tpot_ms_median": median(observed_tpot),
            "observed_tpot_ms_p90": percentile(observed_tpot, 0.90),
            "queue_wait_ms_mean": statistics.mean(queue_waits) if queue_waits else 0.0,
            "queue_wait_ms_p90": percentile(queue_waits, 0.90),
        },
    }


def value(case: dict[str, Any], key: str) -> Any:
    for section in ("accounting", "unified", "cached", "derived"):
        mapping = case.get(section)
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return None


def top_stop_reasons(case: dict[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    by_depth = value(case, "unified_stop_reason_counts_by_depth")
    if not isinstance(by_depth, dict):
        return {}
    for reasons in by_depth.values():
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            totals[str(reason)] = totals.get(str(reason), 0) + int_value(count, 0)
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:6])


def sum_int_map(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(int_value(item, 0) for item in value.values() if not isinstance(item, dict))


def top_counts(value: Any, limit: int = 6) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    totals: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            for nested_key, nested_value in item.items():
                totals[str(nested_key)] = totals.get(str(nested_key), 0) + int_value(nested_value, 0)
        else:
            totals[str(key)] = totals.get(str(key), 0) + int_value(item, 0)
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit])


def metric_definition_rows(legacy: dict[str, Any], unified: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    keys = (
        "goodput_metric_numerator_tokens",
        "goodput_metric_denominator_s",
        "goodput_metric_request_filter",
        "goodput_metric_includes_queue_wait",
        "goodput_metric_includes_decode_only",
        "goodput_metric_includes_cache_build",
        "goodput_metric_definition_version",
    )
    return [(key, value(legacy, key), value(unified, key)) for key in keys]


def diagnosis_rows(legacy: dict[str, Any], unified: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    legacy_goodput = float_value(value(legacy, "goodput_tokens_per_s"), 0.0)
    unified_goodput = float_value(value(unified, "goodput_tokens_per_s"), 0.0)
    legacy_wall_decode = float_value(value(legacy, "wall_decode_tokens_per_s"), 0.0)
    unified_wall_decode = float_value(value(unified, "wall_decode_tokens_per_s"), 0.0)
    unified_commit_share = float_value(value(unified, "generic_committed_token_share_of_output"), 0.0)
    candidate_per_step = float_value(value(unified, "avg_candidate_tokens_per_step"), 0.0)
    committed_per_step = float_value(value(unified, "avg_committed_tokens_per_step"), 0.0)
    commit_steps = int_value(value(unified, "steps_with_any_unified_commit"), 0)
    num_steps = int_value(value(unified, "num_steps"), 0)
    zero_generic_ms = float_value(value(unified, "zero_generic_full_continuous_broadcast_time_ms"), 0.0)
    total_control_units = int_value(value(unified, "total_control_payload_len_units"), 0)
    raw_target_available = bool(value(unified, "unified_raw_target_verification_available"))
    raw_source = value(unified, "unified_raw_verification_source")
    raw_partial = sum_int_map(value(unified, "unified_raw_partial_accept_proposal_count_by_depth"))
    raw_reject = sum_int_map(value(unified, "unified_raw_reject_proposal_count_by_depth"))
    raw_invalidated = sum_int_map(value(unified, "unified_raw_invalidated_proposal_count_by_depth"))
    raw_full = sum_int_map(value(unified, "unified_raw_full_accept_proposal_count_by_depth"))
    raw_verified = sum_int_map(value(unified, "unified_raw_verified_proposal_count_by_depth"))
    ineligible = top_counts(value(unified, "unified_raw_partial_recovery_ineligible_reason_counts_by_depth"))
    commit_budget = int_value(value(unified, "unified_commit_budget_tokens_per_step"), 0)
    commit_budget_used = int_value(value(unified, "unified_commit_budget_used_tokens_per_step"), 0)
    commit_budget_saturated = int_value(value(unified, "unified_commit_budget_saturated_step_count"), 0)
    no_commit_reasons = top_counts(value(unified, "unified_no_commit_step_reason_counts"))
    candidate_reasons = top_counts(value(unified, "unified_candidate_step_reason_counts"))
    if not raw_target_available:
        rows.append(
            (
                "raw verification",
                f"target raw verification unavailable; source={raw_source}",
                "Next action: fix outcome accounting or add a real target-verify trace point before interpreting full accepts as target model agreement.",
            )
        )
    elif raw_partial or raw_reject or raw_invalidated:
        rows.append(
            (
                "raw verification",
                f"partial={raw_partial}, reject={raw_reject}, invalidated={raw_invalidated}",
                f"Partial recovery did not materialize because ineligible/filtered reasons were {format_value(ineligible)}.",
            )
        )
    else:
        rows.append(
            (
                "raw verification",
                f"all raw verified proposals were full accepts ({raw_full}/{raw_verified})",
                "Next action: inspect proposal/verification identity, tokenizer alignment, and whether the path is reusing target tokens instead of draft tokens.",
            )
        )
    if commit_budget and commit_budget_used >= commit_budget and commit_budget_saturated:
        rows.append(
            (
                "budget saturation",
                f"commit used {commit_budget_used}/{commit_budget} tokens on {commit_budget_saturated} step(s)",
                "The flat tokens/depth pattern is consistent with budget saturation; increase budget only after raw outcome accounting is trustworthy.",
            )
        )
    elif commit_budget:
        rows.append(
            (
                "budget saturation",
                f"commit used {commit_budget_used}/{commit_budget} tokens",
                "The flat tokens/depth pattern is not explained by the exposed commit budget alone.",
            )
        )
    if no_commit_reasons or candidate_reasons:
        rows.append(
            (
                "step coverage",
                f"no_commit={format_value(no_commit_reasons)}, no_candidate={format_value(candidate_reasons)}",
                "Use these top reasons to choose between no-ready-parent fixes, active-seq scheduling, or limit tuning.",
            )
        )
    if unified_goodput > legacy_goodput and unified_wall_decode < legacy_wall_decode:
        rows.append(
            (
                "metric definition",
                "goodput improved while decode-only wall throughput regressed",
                "Treat goodput as SLO-filtered numerator over its configured denominator; use wall_decode_tokens_per_s for decode wall throughput.",
            )
        )
    if unified_commit_share < 0.05:
        rows.append(
            (
                "runtime utilization",
                f"unified committed share is {unified_commit_share:.4g}",
                "Next runtime work should improve candidate readiness and commit utilization before tuning communication.",
            )
        )
    if candidate_per_step <= 0.0 or committed_per_step <= 0.0:
        rows.append(
            (
                "scheduling budget",
                f"avg candidate/committed tokens per step = {candidate_per_step:.4g}/{committed_per_step:.4g}",
                "Check active budget, per-depth eligibility, and parent readiness gates.",
            )
        )
    elif num_steps and safe_div(commit_steps, num_steps) < 0.25:
        rows.append(
            (
                "commit cadence",
                f"{commit_steps}/{num_steps} unified steps committed any token",
                "Focus on why ready children do not become globally committed each cached step.",
            )
        )
    if total_control_units > 0 or zero_generic_ms > 0.0:
        rows.append(
            (
                "control plane",
                f"control payload units={total_control_units}, zero generic broadcast ms={zero_generic_ms:.4g}",
                "If utilization is already healthy, inspect compact-v2/meta skip and broadcast payload shape.",
            )
        )
    stop_reasons = top_stop_reasons(unified)
    if stop_reasons:
        rows.append(
            (
                "stop reasons",
                format_value(stop_reasons),
                "Use the top stop reason to choose between eligibility, budget, and sequence-finish fixes.",
            )
        )
    if not rows:
        rows.append(
            (
                "summary",
                "no obvious single bottleneck from the available diagnostics",
                "Compare wall throughput and commit-share trends across more runs before changing runtime behavior.",
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare unified generic rolling performance against legacy.")
    parser.add_argument("--legacy-result", required=True, type=Path)
    parser.add_argument("--legacy-trace", required=True, type=Path)
    parser.add_argument("--unified-result", required=True, type=Path)
    parser.add_argument("--unified-trace", required=True, type=Path)
    args = parser.parse_args()

    legacy = summarize_case(args.legacy_result, args.legacy_trace)
    unified = summarize_case(args.unified_result, args.unified_trace)

    print_compare_table(
        "General",
        [
            ("total_output_tokens", value(legacy, "total_output_tokens"), value(unified, "total_output_tokens")),
            ("engine_elapsed_s", value(legacy, "engine_elapsed_s"), value(unified, "engine_elapsed_s")),
            (
                "cached_admission_decode_only_elapsed_s",
                value(legacy, "cached_admission_decode_only_elapsed_s"),
                value(unified, "cached_admission_decode_only_elapsed_s"),
            ),
            ("goodput_tokens_per_s", value(legacy, "goodput_tokens_per_s"), value(unified, "goodput_tokens_per_s")),
            ("wall_decode_tokens_per_s", value(legacy, "wall_decode_tokens_per_s"), value(unified, "wall_decode_tokens_per_s")),
            ("wall_engine_tokens_per_s", value(legacy, "wall_engine_tokens_per_s"), value(unified, "wall_engine_tokens_per_s")),
            ("mean_tpot_ms", value(legacy, "mean_tpot_ms"), value(unified, "mean_tpot_ms")),
            ("decode_elapsed_ms_median", value(legacy, "decode_elapsed_ms_median"), value(unified, "decode_elapsed_ms_median")),
            ("decode_elapsed_ms_p90", value(legacy, "decode_elapsed_ms_p90"), value(unified, "decode_elapsed_ms_p90")),
            ("observed_tpot_ms_median", value(legacy, "observed_tpot_ms_median"), value(unified, "observed_tpot_ms_median")),
            ("observed_tpot_ms_p90", value(legacy, "observed_tpot_ms_p90"), value(unified, "observed_tpot_ms_p90")),
            ("queue_wait_ms_mean", value(legacy, "queue_wait_ms_mean"), value(unified, "queue_wait_ms_mean")),
            ("queue_wait_ms_p90", value(legacy, "queue_wait_ms_p90"), value(unified, "queue_wait_ms_p90")),
        ],
    )
    print_compare_table("Goodput Metric Definition", metric_definition_rows(legacy, unified))
    print_compare_table(
        "Speculative/Generic",
        [
            ("unified_generic_rolling_enabled", value(legacy, "unified_generic_rolling_enabled"), value(unified, "unified_generic_rolling_enabled")),
            ("combined_real_committed_token_count", value(legacy, "combined_real_committed_token_count"), value(unified, "combined_real_committed_token_count")),
            ("generic_full_commit_tokens", value(legacy, "generic_full_commit_tokens"), value(unified, "generic_full_commit_tokens")),
            ("generic_partial_recovered_tokens", value(legacy, "generic_partial_recovered_tokens"), value(unified, "generic_partial_recovered_tokens")),
            ("generic_total_output_tokens", value(legacy, "generic_total_output_tokens"), value(unified, "generic_total_output_tokens")),
            ("generic_committed_token_share_of_output", value(legacy, "generic_committed_token_share_of_output"), value(unified, "generic_committed_token_share_of_output")),
            ("max_real_committed_depth", value(legacy, "max_real_committed_depth"), value(unified, "max_real_committed_depth")),
            ("candidate_depths", value(legacy, "candidate_depths"), value(unified, "candidate_depths")),
            ("ready_depths", value(legacy, "ready_depths"), value(unified, "ready_depths")),
            ("committed_depths", value(legacy, "committed_depths"), value(unified, "committed_depths")),
            ("unified_candidate_token_count_by_depth", value(legacy, "unified_candidate_token_count_by_depth"), value(unified, "unified_candidate_token_count_by_depth")),
            ("unified_ready_token_count_by_depth", value(legacy, "unified_ready_token_count_by_depth"), value(unified, "unified_ready_token_count_by_depth")),
            ("unified_committed_token_count_by_depth", value(legacy, "unified_committed_token_count_by_depth"), value(unified, "unified_committed_token_count_by_depth")),
            ("unified_commit_share_by_depth", value(legacy, "unified_commit_share_by_depth"), value(unified, "unified_commit_share_by_depth")),
            ("num_steps", value(legacy, "num_steps"), value(unified, "num_steps")),
            ("steps_with_any_unified_candidate", value(legacy, "steps_with_any_unified_candidate"), value(unified, "steps_with_any_unified_candidate")),
            ("steps_with_any_unified_commit", value(legacy, "steps_with_any_unified_commit"), value(unified, "steps_with_any_unified_commit")),
            ("avg_candidate_tokens_per_step", value(legacy, "avg_candidate_tokens_per_step"), value(unified, "avg_candidate_tokens_per_step")),
            ("avg_committed_tokens_per_step", value(legacy, "avg_committed_tokens_per_step"), value(unified, "avg_committed_tokens_per_step")),
            ("unified_stop_reason_counts_by_depth", value(legacy, "unified_stop_reason_counts_by_depth"), value(unified, "unified_stop_reason_counts_by_depth")),
        ],
    )
    print_compare_table(
        "Control Plane",
        [
            ("timing_available", value(legacy, "timing_available"), value(unified, "timing_available")),
            ("payload_bytes_available", value(legacy, "payload_bytes_available"), value(unified, "payload_bytes_available")),
            ("total_control_payload_bytes", value(legacy, "total_control_payload_bytes"), value(unified, "total_control_payload_bytes")),
            ("total_control_payload_len_units", value(legacy, "total_control_payload_len_units"), value(unified, "total_control_payload_len_units")),
            ("total_control_payload_units_per_total_output_token", value(legacy, "total_control_payload_units_per_total_output_token"), value(unified, "total_control_payload_units_per_total_output_token")),
            ("zero_candidate_payload_broadcast_skipped_count", value(legacy, "zero_candidate_payload_broadcast_skipped_count"), value(unified, "zero_candidate_payload_broadcast_skipped_count")),
            ("zero_stage_fast_path_payload_build_skipped_count", value(legacy, "zero_stage_fast_path_payload_build_skipped_count"), value(unified, "zero_stage_fast_path_payload_build_skipped_count")),
            ("zero_generic_full_continuous_broadcast_time_ms", value(legacy, "zero_generic_full_continuous_broadcast_time_ms"), value(unified, "zero_generic_full_continuous_broadcast_time_ms")),
            ("total_eager_overhead_time_ms", value(legacy, "total_eager_overhead_time_ms"), value(unified, "total_eager_overhead_time_ms")),
            ("eager_transfer_time_ms", value(legacy, "eager_transfer_time_ms"), value(unified, "eager_transfer_time_ms")),
        ],
    )
    print_compare_table(
        "Unified Raw/Budget Diagnosis",
        [
            ("unified_raw_target_verification_available", value(legacy, "unified_raw_target_verification_available"), value(unified, "unified_raw_target_verification_available")),
            ("unified_raw_verification_source", value(legacy, "unified_raw_verification_source"), value(unified, "unified_raw_verification_source")),
            ("unified_raw_verified_proposal_count_by_depth", value(legacy, "unified_raw_verified_proposal_count_by_depth"), value(unified, "unified_raw_verified_proposal_count_by_depth")),
            ("unified_raw_full_accept_proposal_count_by_depth", value(legacy, "unified_raw_full_accept_proposal_count_by_depth"), value(unified, "unified_raw_full_accept_proposal_count_by_depth")),
            ("unified_raw_partial_accept_proposal_count_by_depth", value(legacy, "unified_raw_partial_accept_proposal_count_by_depth"), value(unified, "unified_raw_partial_accept_proposal_count_by_depth")),
            ("unified_raw_reject_proposal_count_by_depth", value(legacy, "unified_raw_reject_proposal_count_by_depth"), value(unified, "unified_raw_reject_proposal_count_by_depth")),
            ("unified_raw_invalidated_proposal_count_by_depth", value(legacy, "unified_raw_invalidated_proposal_count_by_depth"), value(unified, "unified_raw_invalidated_proposal_count_by_depth")),
            ("unified_raw_accepted_len_hist_by_depth", value(legacy, "unified_raw_accepted_len_hist_by_depth"), value(unified, "unified_raw_accepted_len_hist_by_depth")),
            ("unified_raw_revised_token_count_by_depth", value(legacy, "unified_raw_revised_token_count_by_depth"), value(unified, "unified_raw_revised_token_count_by_depth")),
            ("unified_raw_partial_recovery_eligible_count_by_depth", value(legacy, "unified_raw_partial_recovery_eligible_count_by_depth"), value(unified, "unified_raw_partial_recovery_eligible_count_by_depth")),
            ("unified_raw_partial_recovery_ineligible_reason_counts_by_depth", value(legacy, "unified_raw_partial_recovery_ineligible_reason_counts_by_depth"), value(unified, "unified_raw_partial_recovery_ineligible_reason_counts_by_depth")),
            ("unified_raw_candidate_proposal_count_by_depth", value(legacy, "unified_raw_candidate_proposal_count_by_depth"), value(unified, "unified_raw_candidate_proposal_count_by_depth")),
            ("unified_raw_committed_proposal_count_by_depth", value(legacy, "unified_raw_committed_proposal_count_by_depth"), value(unified, "unified_raw_committed_proposal_count_by_depth")),
            ("unified_raw_verified_to_committed_ratio_by_depth", value(legacy, "unified_raw_verified_to_committed_ratio_by_depth"), value(unified, "unified_raw_verified_to_committed_ratio_by_depth")),
            ("unified_candidate_waste_ratio_by_depth", value(legacy, "unified_candidate_waste_ratio_by_depth"), value(unified, "unified_candidate_waste_ratio_by_depth")),
            ("unified_invalidated_candidate_ratio_by_depth", value(legacy, "unified_invalidated_candidate_ratio_by_depth"), value(unified, "unified_invalidated_candidate_ratio_by_depth")),
            ("unified_verified_candidate_ratio_by_depth", value(legacy, "unified_verified_candidate_ratio_by_depth"), value(unified, "unified_verified_candidate_ratio_by_depth")),
            ("unified_committed_candidate_ratio_by_depth", value(legacy, "unified_committed_candidate_ratio_by_depth"), value(unified, "unified_committed_candidate_ratio_by_depth")),
            ("unified_candidate_budget_tokens_per_step", value(legacy, "unified_candidate_budget_tokens_per_step"), value(unified, "unified_candidate_budget_tokens_per_step")),
            ("unified_candidate_budget_used_tokens_per_step", value(legacy, "unified_candidate_budget_used_tokens_per_step"), value(unified, "unified_candidate_budget_used_tokens_per_step")),
            ("unified_candidate_budget_saturated_step_count", value(legacy, "unified_candidate_budget_saturated_step_count"), value(unified, "unified_candidate_budget_saturated_step_count")),
            ("unified_commit_budget_tokens_per_step", value(legacy, "unified_commit_budget_tokens_per_step"), value(unified, "unified_commit_budget_tokens_per_step")),
            ("unified_commit_budget_used_tokens_per_step", value(legacy, "unified_commit_budget_used_tokens_per_step"), value(unified, "unified_commit_budget_used_tokens_per_step")),
            ("unified_commit_budget_saturated_step_count", value(legacy, "unified_commit_budget_saturated_step_count"), value(unified, "unified_commit_budget_saturated_step_count")),
            ("unified_ready_but_not_committed_token_count", value(legacy, "unified_ready_but_not_committed_token_count"), value(unified, "unified_ready_but_not_committed_token_count")),
            ("unified_ready_but_not_committed_reason_counts", value(legacy, "unified_ready_but_not_committed_reason_counts"), value(unified, "unified_ready_but_not_committed_reason_counts")),
            ("unified_ready_but_not_committed_by_depth", value(legacy, "unified_ready_but_not_committed_by_depth"), value(unified, "unified_ready_but_not_committed_by_depth")),
            ("unified_commit_limited_by_token_budget_count", value(legacy, "unified_commit_limited_by_token_budget_count"), value(unified, "unified_commit_limited_by_token_budget_count")),
            ("unified_commit_limited_by_seq_budget_count", value(legacy, "unified_commit_limited_by_seq_budget_count"), value(unified, "unified_commit_limited_by_seq_budget_count")),
            ("unified_commit_limited_by_no_ready_parent_count", value(legacy, "unified_commit_limited_by_no_ready_parent_count"), value(unified, "unified_commit_limited_by_no_ready_parent_count")),
            ("unified_commit_limited_by_parent_not_full_accept_count", value(legacy, "unified_commit_limited_by_parent_not_full_accept_count"), value(unified, "unified_commit_limited_by_parent_not_full_accept_count")),
            ("unified_no_candidate_step_count", value(legacy, "unified_no_candidate_step_count"), value(unified, "unified_no_candidate_step_count")),
            ("unified_no_commit_step_count", value(legacy, "unified_no_commit_step_count"), value(unified, "unified_no_commit_step_count")),
            ("unified_candidate_step_reason_counts", value(legacy, "unified_candidate_step_reason_counts"), value(unified, "unified_candidate_step_reason_counts")),
            ("unified_no_commit_step_reason_counts", value(legacy, "unified_no_commit_step_reason_counts"), value(unified, "unified_no_commit_step_reason_counts")),
            ("unified_active_seq_count_by_step", value(legacy, "unified_active_seq_count_by_step"), value(unified, "unified_active_seq_count_by_step")),
            ("unified_ready_parent_count_by_step", value(legacy, "unified_ready_parent_count_by_step"), value(unified, "unified_ready_parent_count_by_step")),
            ("unified_committed_seq_count_by_step", value(legacy, "unified_committed_seq_count_by_step"), value(unified, "unified_committed_seq_count_by_step")),
            ("unified_single_child_ahead_enabled", value(legacy, "unified_single_child_ahead_enabled"), value(unified, "unified_single_child_ahead_enabled")),
            ("unified_max_unverified_depth_ahead", value(legacy, "unified_max_unverified_depth_ahead"), value(unified, "unified_max_unverified_depth_ahead")),
            ("unified_unverified_depth_ahead_max_observed", value(legacy, "unified_unverified_depth_ahead_max_observed"), value(unified, "unified_unverified_depth_ahead_max_observed")),
            ("unified_unverified_depth_ahead_by_chain", value(legacy, "unified_unverified_depth_ahead_by_chain"), value(unified, "unified_unverified_depth_ahead_by_chain")),
            ("unified_single_child_ahead_violation_count", value(legacy, "unified_single_child_ahead_violation_count"), value(unified, "unified_single_child_ahead_violation_count")),
            ("unified_generated_grandchild_before_parent_verified_count", value(legacy, "unified_generated_grandchild_before_parent_verified_count"), value(unified, "unified_generated_grandchild_before_parent_verified_count")),
            ("unified_candidate_depth_created_before_parent_verified_count_by_depth", value(legacy, "unified_candidate_depth_created_before_parent_verified_count_by_depth"), value(unified, "unified_candidate_depth_created_before_parent_verified_count_by_depth")),
            ("unified_candidate_depth_created_after_parent_verified_count_by_depth", value(legacy, "unified_candidate_depth_created_after_parent_verified_count_by_depth"), value(unified, "unified_candidate_depth_created_after_parent_verified_count_by_depth")),
        ],
    )
    print_compare_table(
        "Cached Admission",
        [
            ("cached_admission_total_requests", value(legacy, "cached_admission_total_requests"), value(unified, "cached_admission_total_requests")),
            ("cached_admission_arrived_count", value(legacy, "cached_admission_arrived_count"), value(unified, "cached_admission_arrived_count")),
            ("cached_admission_admitted_count", value(legacy, "cached_admission_admitted_count"), value(unified, "cached_admission_admitted_count")),
            ("cached_admission_completed_count", value(legacy, "cached_admission_completed_count"), value(unified, "cached_admission_completed_count")),
            ("cached_admission_peak_active", value(legacy, "cached_admission_peak_active"), value(unified, "cached_admission_peak_active")),
            ("cached_admission_queue_wait_mean_ms", value(legacy, "cached_admission_queue_wait_mean_ms"), value(unified, "cached_admission_queue_wait_mean_ms")),
            ("cached_admission_queue_wait_p50_ms", value(legacy, "cached_admission_queue_wait_p50_ms"), value(unified, "cached_admission_queue_wait_p50_ms")),
            ("cached_admission_queue_wait_p90_ms", value(legacy, "cached_admission_queue_wait_p90_ms"), value(unified, "cached_admission_queue_wait_p90_ms")),
            ("cached_admission_queue_wait_p99_ms", value(legacy, "cached_admission_queue_wait_p99_ms"), value(unified, "cached_admission_queue_wait_p99_ms")),
            ("cached_admission_decode_only_elapsed_s", value(legacy, "cached_admission_decode_only_elapsed_s"), value(unified, "cached_admission_decode_only_elapsed_s")),
            ("cached_cache_build_elapsed_s", value(legacy, "cached_cache_build_elapsed_s"), value(unified, "cached_cache_build_elapsed_s")),
        ],
    )
    print_text_table(
        "Diagnosis/Recommendation",
        ("topic", "finding", "recommendation"),
        diagnosis_rows(legacy, unified),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
