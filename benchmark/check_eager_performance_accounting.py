#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import validate_records as validate_commit_records


TAKEOVER_SOURCE = "phase1h5e3_takeover_lane"
TIMING_FIELDS = [
    "eager_plan_time_ms",
    "eager_draft_time_ms",
    "eager_transfer_time_ms",
    "eager_verify_dry_run_time_ms",
    "eager_apply_dry_run_time_ms",
    "eager_result_transfer_time_ms",
    "eager_sync_apply_dry_run_time_ms",
    "eager_commit_readiness_time_ms",
    "eager_commit_time_ms",
]
PROPOSAL_LEN_MAP_KEYS = [
    "eager_commit_ready_token_count_by_proposal_id",
    "eager_committed_token_count_by_proposal_id",
    "eager_apply_dry_run_proposal_len_by_proposal_id",
    "eager_verify_dry_run_proposal_len_by_proposal_id",
    "eager_result_sent_proposal_len_by_proposal_id",
    "eager_result_received_proposal_len_by_proposal_id",
    "eager_transfer_proposal_len_by_proposal_id",
    "eager_schedule_proposal_len_by_proposal_id",
]
DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD = 0.01
DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD = 128.0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return trace_payload_to_records(data)


def trace_payload_to_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records", "events", "iterations", "batches"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    return []


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def dict_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def step_plan_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int_value(record.get("step_id"), int_value(record.get("eager_commit_step_id"), -1)),
        int_value(record.get("plan_id"), int_value(record.get("eager_commit_plan_id"), -1)),
    )


def proposal_len(proposal_id: int, proposal_len_by_id: dict[int, int], gamma: int) -> int:
    value = proposal_len_by_id.get(proposal_id, 0)
    if value > 0:
        return value
    return max(0, gamma)


def sum_proposal_lens(proposal_ids: set[int], proposal_len_by_id: dict[int, int], gamma: int) -> int:
    return sum(proposal_len(proposal_id, proposal_len_by_id, gamma) for proposal_id in proposal_ids)


def result_metrics(result_payload: dict[str, Any]) -> dict[str, Any]:
    args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    metrics = result_payload.get("metrics", {}) if isinstance(result_payload, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    return {
        "execution_mode": args.get("execution_mode") or metrics.get("execution_mode"),
        "decode_ready_mode": args.get("decode_ready", metrics.get("decode_ready_mode")),
        "total_output_tokens": int_value(overall.get("total_output_tokens"), 0),
        "engine_elapsed_s": float_value(metrics.get("engine_elapsed_s"), 0.0),
        "goodput_tokens_per_s": float_value(overall.get("goodput_tokens_per_s"), 0.0),
        "mean_tpot_ms": float_value(overall.get("mean_tpot_ms"), 0.0),
    }


def add_derived_metrics(accounting: dict[str, Any]) -> dict[str, Any]:
    total_output_tokens = int_value(accounting.get("total_output_tokens"), 0)
    candidate_tokens = int_value(accounting.get("eager_candidate_token_count"), 0)
    ready_tokens = int_value(accounting.get("eager_ready_token_count"), 0)
    committed_tokens = int_value(accounting.get("eager_committed_token_count"), 0)
    suppressed_slots = int_value(accounting.get("normal_draft_token_slots_suppressed"), 0)
    replaced_slots = int_value(accounting.get("target_normal_verify_token_slots_replaced_by_eager"), 0)
    proposal_payload_units = int_value(accounting.get("eager_proposal_transfer_payload_len_units"), 0)
    result_payload_units = int_value(accounting.get("eager_result_transfer_payload_len_units"), 0)
    accounting.update(
        {
            "committed_token_share_of_output": safe_div(committed_tokens, total_output_tokens),
            "candidate_token_share_of_output": safe_div(candidate_tokens, total_output_tokens),
            "suppressed_slots_per_committed_token": safe_div(suppressed_slots, committed_tokens),
            "replaced_slots_per_committed_token": safe_div(replaced_slots, committed_tokens),
            "proposal_payload_len_units_per_committed_token": safe_div(
                proposal_payload_units,
                committed_tokens,
            ),
            "result_payload_len_units_per_committed_token": safe_div(
                result_payload_units,
                committed_tokens,
            ),
            "committed_tokens_per_candidate_token": safe_div(committed_tokens, candidate_tokens),
            "ready_tokens_per_candidate_token": safe_div(ready_tokens, candidate_tokens),
            "committed_tokens_per_ready_token": safe_div(committed_tokens, ready_tokens),
        }
    )
    return accounting


def performance_warnings(
    accounting: dict[str, Any],
    *,
    low_committed_share_threshold: float = DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD,
    high_payload_len_per_committed_token_threshold: float = (
        DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD
    ),
    baseline_goodput_tokens_per_s: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    committed_tokens = int_value(accounting.get("eager_committed_token_count"), 0)
    committed_share = float_value(accounting.get("committed_token_share_of_output"), 0.0)
    if committed_tokens <= 0:
        warnings.append("no_committed_tokens")
    elif committed_share < low_committed_share_threshold:
        warnings.append(
            f"committed_token_share_below_{low_committed_share_threshold:g}"
        )
    if baseline_goodput_tokens_per_s is not None and baseline_goodput_tokens_per_s > 0:
        eager_goodput = float_value(accounting.get("goodput_tokens_per_s"), 0.0)
        if eager_goodput < baseline_goodput_tokens_per_s:
            warnings.append("eager_goodput_below_baseline")
    if not bool(accounting.get("timing_available", False)):
        warnings.append("timing_unavailable")
    if not bool(accounting.get("payload_bytes_available", False)):
        warnings.append("payload_bytes_unavailable")
    if (
        committed_tokens > 0
        and float_value(accounting.get("proposal_payload_len_units_per_committed_token"), 0.0)
        > high_payload_len_per_committed_token_threshold
    ):
        warnings.append("proposal_payload_len_units_high_per_committed_token")
    if (
        committed_tokens > 0
        and int_value(accounting.get("normal_draft_token_slots_suppressed"), 0) > committed_tokens
    ):
        warnings.append("suppressed_slots_exceed_committed_tokens")
    return warnings


def validate_result_sanity(result_payload: dict[str, Any]) -> list[str]:
    rows = result_payload.get("traces", []) if isinstance(result_payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    errors: list[str] = []
    arrival_after_finish = 0
    elapsed_mismatch = 0
    tpot_mismatch = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        arrival_ts = float_value(row.get("arrival_ts"), float("nan"))
        finish_ts = first_present(row, ["finish_ts", "finished_ts", "end_ts", "end_time"])
        finish_ts = float_value(finish_ts, float("nan"))
        if arrival_ts == arrival_ts and finish_ts == finish_ts and arrival_ts > finish_ts:
            arrival_after_finish += 1
        decode_start_ts = first_present(
            row,
            ["decode_start_ts", "decoding_start_ts", "first_decode_ts", "first_token_ts", "start_decode_ts"],
        )
        decode_elapsed_ms = row.get("decode_elapsed_ms")
        if decode_start_ts is not None and decode_elapsed_ms is not None and finish_ts == finish_ts:
            expected = (finish_ts - float_value(decode_start_ts, finish_ts)) * 1000.0
            if abs(expected - float_value(decode_elapsed_ms)) > 1e-3:
                elapsed_mismatch += 1
        tokens = int_value(
            first_present(
                row,
                [
                    "num_decode_output_tokens",
                    "num_output_tokens",
                    "num_completion_tokens",
                    "completion_tokens",
                    "output_tokens",
                    "num_generated_tokens",
                    "num_tokens",
                ],
            ),
            0,
        )
        observed_tpot_ms = row.get("observed_tpot_ms")
        if tokens > 0 and decode_elapsed_ms is not None and observed_tpot_ms is not None:
            expected = float_value(decode_elapsed_ms) / tokens
            if abs(expected - float_value(observed_tpot_ms)) > 1e-3:
                tpot_mismatch += 1
    if arrival_after_finish:
        errors.append(f"result JSON has arrival_ts > finish_ts rows: {arrival_after_finish}")
    if elapsed_mismatch:
        errors.append(f"result JSON has decode_elapsed_ms timestamp mismatches: {elapsed_mismatch}")
    if tpot_mismatch:
        errors.append(f"result JSON has observed_tpot_ms arithmetic mismatches: {tpot_mismatch}")
    return errors


def aggregate_performance_accounting(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_payload = result_payload or {}
    _commit_errors, commit_summary = validate_commit_records(records)
    gamma = 0
    proposal_len_by_id: dict[int, int] = {}
    candidate_ids: set[int] = set()
    ready_ids: set[int] = set()
    committed_ids: set[int] = set()
    skipped_ids: set[int] = set()
    skip_reason_by_id: dict[int, str] = {}
    lane_applied_ids: set[int] = set()
    lane_applied_seq_fallback_events: set[tuple[int, int, int]] = set()
    takeover_ids: set[int] = set()
    takeover_seq_fallback_events: set[tuple[int, int, int]] = set()
    missing_allowed_events: set[tuple[int, int, int]] = set()
    proposal_transfer_steps: set[tuple[int, int]] = set()
    result_transfer_steps: set[tuple[int, int]] = set()
    lane_sync_steps: set[tuple[int, int]] = set()
    zero_result_transfer_steps: set[tuple[int, int]] = set()
    proposal_transfer_payload_len_units = 0
    result_transfer_payload_len_units = 0
    proposal_transfer_payload_bytes = 0
    result_transfer_payload_bytes = 0
    negative_payload_field_count = 0
    payload_bytes_available = False
    timing_sums = {key: 0.0 for key in TIMING_FIELDS}
    timing_available = False

    counted_payload_len_events: set[tuple[str, int, int, int]] = set()
    counted_payload_byte_events: set[tuple[str, int, int, int]] = set()
    counted_timing_events: set[tuple[str, int, int, float]] = set()

    for record in records:
        gamma = max(gamma, int_value(record.get("normal_gamma"), 0))
        key = step_plan_key(record)
        for map_key in PROPOSAL_LEN_MAP_KEYS:
            for proposal_id, token_count in as_int_map(record.get(map_key)).items():
                if token_count > 0:
                    proposal_len_by_id.setdefault(proposal_id, token_count)

        candidate_ids.update(as_int_set(record.get("eager_commit_candidate_proposal_ids")))
        candidate_ids.update(as_int_set(record.get("eager_commit_readiness_candidate_proposal_ids")))
        ready_ids.update(as_int_set(record.get("eager_commit_ready_proposal_ids")))
        ready_ids.update(as_int_set(record.get("eager_commit_from_readiness_proposal_ids")))
        committed_ids.update(as_int_set(record.get("eager_committed_proposal_ids")))
        skipped_ids.update(as_int_set(record.get("eager_commit_skipped_proposal_ids")))
        skipped_ids.update(as_int_set(record.get("eager_commit_not_ready_proposal_ids")))

        reason_map = record.get("eager_commit_skip_reason_by_proposal_id")
        if not isinstance(reason_map, dict):
            reason_map = record.get("eager_commit_not_ready_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    skip_reason_by_id.setdefault(proposal_id, str(reason))

        applied_ids = as_int_list(record.get("lane_exclusion_applied_proposal_ids"))
        lane_applied_ids.update(applied_ids)
        if applied_ids:
            for proposal_id in applied_ids:
                event_key = ("lane", key[0], key[1], proposal_id)
                if event_key not in counted_payload_len_events:
                    counted_payload_len_events.add(event_key)
        else:
            for seq_id in as_int_list(record.get("lane_exclusion_applied_seq_ids")):
                lane_applied_seq_fallback_events.add((key[0], key[1], seq_id))

        takeover_proposals = as_int_list(record.get("target_eager_verify_proposal_ids_dry_run"))
        takeover_ids.update(takeover_proposals)
        if not takeover_proposals:
            for seq_id in as_int_list(record.get("target_eager_verify_seq_ids_dry_run")):
                takeover_seq_fallback_events.add((key[0], key[1], seq_id))
        for seq_id in as_int_list(record.get("missing_buffered_proposal_allowed_by_eager_seq_ids")):
            missing_allowed_events.add((key[0], key[1], seq_id))

        if record.get("ready_eager_proposal_synced_ids"):
            lane_sync_steps.add(key)

        for field_name, step_set in (
            ("eager_transfer_payload_len", proposal_transfer_steps),
            ("eager_result_transfer_payload_len", result_transfer_steps),
        ):
            if field_name in record:
                step_set.add(key)
                payload_len = int_value(record.get(field_name), 0)
                if payload_len < 0:
                    negative_payload_field_count += 1
                event_key = (field_name, key[0], key[1], payload_len)
                if payload_len > 0 and event_key not in counted_payload_len_events:
                    counted_payload_len_events.add(event_key)
                    if field_name == "eager_transfer_payload_len":
                        proposal_transfer_payload_len_units += payload_len
                    else:
                        result_transfer_payload_len_units += payload_len

        for field_name in ("eager_transfer_payload_bytes", "eager_result_transfer_payload_bytes"):
            if field_name in record:
                payload_bytes_available = True
                payload_bytes = int_value(record.get(field_name), 0)
                if payload_bytes < 0:
                    negative_payload_field_count += 1
                event_key = (field_name, key[0], key[1], payload_bytes)
                if payload_bytes >= 0 and event_key not in counted_payload_byte_events:
                    counted_payload_byte_events.add(event_key)
                    if field_name == "eager_transfer_payload_bytes":
                        proposal_transfer_payload_bytes += payload_bytes
                    else:
                        result_transfer_payload_bytes += payload_bytes

        if bool(record.get("eager_result_transfer_zero_result_step", False)):
            zero_result_transfer_steps.add(key)

        for field_name in TIMING_FIELDS:
            if field_name in record:
                timing_available = True
                value = float_value(record.get(field_name), 0.0)
                event_key = (field_name, key[0], key[1], value)
                if event_key not in counted_timing_events:
                    counted_timing_events.add(event_key)
                    timing_sums[field_name] += value

    committed_token_count = int_value(commit_summary.get("committed_token_count"), 0)
    committed_proposal_count = int_value(commit_summary.get("committed_proposal_count"), len(committed_ids))
    candidate_token_count = sum_proposal_lens(candidate_ids, proposal_len_by_id, gamma)
    ready_token_count = sum_proposal_lens(ready_ids, proposal_len_by_id, gamma)
    lane_token_slots = sum_proposal_lens(lane_applied_ids, proposal_len_by_id, gamma)
    if not lane_applied_ids:
        lane_token_slots = len(lane_applied_seq_fallback_events) * max(0, gamma)
    takeover_token_slots = sum_proposal_lens(takeover_ids, proposal_len_by_id, gamma)
    if not takeover_ids:
        takeover_token_slots = len(takeover_seq_fallback_events) * max(0, gamma)

    candidate_count = len(candidate_ids)
    ready_count = len(ready_ids)
    skipped_count = len(skipped_ids)
    commit_rate_by_proposal = committed_proposal_count / candidate_count if candidate_count else 0.0
    commit_rate_by_token = committed_token_count / candidate_token_count if candidate_token_count else 0.0
    eager_full_accept_rate = ready_count / candidate_count if candidate_count else 0.0
    eager_partial_reject_rate = skipped_count / candidate_count if candidate_count else 0.0

    timing_summary = dict(timing_sums)
    timing_summary["total_eager_overhead_time_ms"] = sum(timing_sums.values()) if timing_available else 0.0
    embedded_accounting = (
        result_payload.get("eager_performance_accounting", {})
        if isinstance(result_payload, dict)
        else {}
    )
    accounting_summary_time_ms = (
        float_value(embedded_accounting.get("eager_accounting_summary_time_ms"), 0.0)
        if isinstance(embedded_accounting, dict)
        else 0.0
    )

    accounting = {
        "accounting_available": True,
        **result_metrics(result_payload),
        "timing_available": bool(timing_available),
        "missing_timing_reason": None if timing_available else "not_instrumented",
        "payload_bytes_available": bool(payload_bytes_available),
        "missing_payload_bytes_reason": None if payload_bytes_available else "payload_byte_fields_not_instrumented",
        "eager_candidate_proposal_count": candidate_count,
        "eager_candidate_token_count": candidate_token_count,
        "eager_ready_proposal_count": ready_count,
        "eager_ready_token_count": ready_token_count,
        "eager_committed_proposal_count": committed_proposal_count,
        "eager_committed_token_count": committed_token_count,
        "eager_skipped_proposal_count": skipped_count,
        "eager_skip_reason_counts": dict(Counter(skip_reason_by_id.values())),
        "eager_commit_rate_by_proposal": commit_rate_by_proposal,
        "eager_commit_rate_by_token": commit_rate_by_token,
        "eager_full_accept_rate": eager_full_accept_rate,
        "eager_partial_reject_rate": eager_partial_reject_rate,
        "normal_draft_seq_excluded_count": len(lane_applied_ids) or len(lane_applied_seq_fallback_events),
        "normal_draft_token_slots_suppressed": lane_token_slots,
        "normal_proposal_missing_allowed_by_eager_count": len(missing_allowed_events),
        "target_normal_verify_seq_excluded_by_eager_count": len(takeover_ids) or len(takeover_seq_fallback_events),
        "target_normal_verify_token_slots_replaced_by_eager": takeover_token_slots,
        "eager_verified_token_count": commit_summary.get("target_actual_eager_verified_token_increment_sum", 0),
        "eager_accepted_token_count": commit_summary.get("target_actual_eager_accepted_token_increment_sum", 0),
        "eager_rejected_token_count": commit_summary.get("target_actual_eager_rejected_token_increment_sum", 0),
        "eager_invalidated_token_count": commit_summary.get("target_actual_eager_invalidated_token_increment_sum", 0),
        "target_actual_eager_verified_token_increment_sum": commit_summary.get(
            "target_actual_eager_verified_token_increment_sum", 0
        ),
        "target_actual_eager_accepted_token_increment_sum": commit_summary.get(
            "target_actual_eager_accepted_token_increment_sum", 0
        ),
        "target_actual_eager_rejected_token_increment_sum": commit_summary.get(
            "target_actual_eager_rejected_token_increment_sum", 0
        ),
        "target_actual_eager_invalidated_token_increment_sum": commit_summary.get(
            "target_actual_eager_invalidated_token_increment_sum", 0
        ),
        "draft_actual_eager_verified_token_increment_sum": commit_summary.get(
            "draft_actual_eager_verified_token_increment_sum", 0
        ),
        "draft_actual_eager_accepted_token_increment_sum": commit_summary.get(
            "draft_actual_eager_accepted_token_increment_sum", 0
        ),
        "eager_proposal_transfer_steps": len(proposal_transfer_steps),
        "eager_proposal_transfer_payload_bytes": proposal_transfer_payload_bytes,
        "eager_proposal_transfer_payload_len_units": proposal_transfer_payload_len_units,
        "eager_result_transfer_steps": len(result_transfer_steps),
        "eager_result_transfer_payload_bytes": result_transfer_payload_bytes,
        "eager_result_transfer_payload_len_units": result_transfer_payload_len_units,
        "negative_payload_field_count": negative_payload_field_count,
        "eager_lane_exclusion_sync_steps": len(lane_sync_steps),
        "eager_commit_check_active_records": commit_summary.get("commit_active_records", 0),
        "eager_commit_metadata_bytes": 0,
        "extra_collective_count_by_type": {},
        "zero_result_transfer_steps": len(zero_result_transfer_steps),
        "eager_accounting_summary_time_ms": accounting_summary_time_ms,
        "repeated_commit_proposal_ids": commit_summary.get("repeated_commit_proposal_ids", []),
        "missing_buffered_proposal_unexpected_count": commit_summary.get(
            "missing_buffered_proposal_unexpected_count", 0
        ),
        "real_target_eager_non_empty_count": commit_summary.get("real_target_eager_non_empty_count", 0),
        "commit_checker_error_count": 0,
        **timing_summary,
    }
    accounting = add_derived_metrics(accounting)
    accounting["performance_warnings"] = performance_warnings(accounting)
    return accounting


def validate_accounting(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    *,
    strict_performance: bool = False,
    low_committed_share_threshold: float = DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD,
    high_payload_len_per_committed_token_threshold: float = (
        DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD
    ),
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    commit_errors, _commit_summary = validate_commit_records(records)
    errors.extend(f"commit checker: {error}" for error in commit_errors)
    accounting = aggregate_performance_accounting(records, result_payload)
    accounting["commit_checker_error_count"] = len(commit_errors)

    committed_tokens = int_value(accounting.get("eager_committed_token_count"), 0)
    committed_proposals = int_value(accounting.get("eager_committed_proposal_count"), 0)
    candidate_tokens = int_value(accounting.get("eager_candidate_token_count"), 0)
    candidate_proposals = int_value(accounting.get("eager_candidate_proposal_count"), 0)
    skipped_proposals = int_value(accounting.get("eager_skipped_proposal_count"), 0)

    if committed_tokens != int_value(accounting.get("target_actual_eager_verified_token_increment_sum"), 0):
        errors.append("committed tokens must equal target actual eager verified increment sum")
    if committed_tokens != int_value(accounting.get("target_actual_eager_accepted_token_increment_sum"), 0):
        errors.append("committed tokens must equal target actual eager accepted increment sum")
    if committed_proposals > candidate_proposals:
        errors.append("committed proposal count exceeds candidate proposal count")
    if candidate_tokens and committed_tokens > candidate_tokens:
        errors.append("committed token count exceeds candidate token count")
    for field in ("eager_commit_rate_by_token", "eager_commit_rate_by_proposal"):
        value = float_value(accounting.get(field), -1.0)
        if not (0.0 <= value <= 1.0):
            errors.append(f"{field} must be in [0, 1], got {value}")
    if candidate_proposals and skipped_proposals + committed_proposals > candidate_proposals:
        errors.append("skipped plus committed proposals exceeds candidate proposal count")
    if skipped_proposals and not accounting.get("eager_skip_reason_counts"):
        errors.append("skipped proposals require skip reason counts")
    if committed_tokens and int_value(accounting.get("normal_draft_token_slots_suppressed"), 0) < committed_tokens:
        errors.append("normal draft token slots suppressed must cover committed tokens")
    if committed_tokens and int_value(accounting.get("target_normal_verify_token_slots_replaced_by_eager"), 0) < committed_tokens:
        errors.append("target normal verify token slots replaced by eager must cover committed tokens")
    for field in (
        "eager_proposal_transfer_payload_bytes",
        "eager_result_transfer_payload_bytes",
        "eager_proposal_transfer_payload_len_units",
        "eager_result_transfer_payload_len_units",
    ):
        if int_value(accounting.get(field), 0) < 0:
            errors.append(f"{field} must be nonnegative")
    if int_value(accounting.get("negative_payload_field_count"), 0) != 0:
        errors.append("payload length/byte fields must be nonnegative")
    if accounting.get("timing_available"):
        for field in TIMING_FIELDS + ["total_eager_overhead_time_ms"]:
            if float_value(accounting.get(field), 0.0) < 0.0:
                errors.append(f"{field} must be nonnegative")
    elif accounting.get("missing_timing_reason") != "not_instrumented":
        errors.append("missing timing must be explicitly marked not_instrumented")
    if float_value(accounting.get("eager_accounting_summary_time_ms"), 0.0) < 0.0:
        errors.append("eager_accounting_summary_time_ms must be nonnegative")
    if int_value(accounting.get("missing_buffered_proposal_unexpected_count"), 0) != 0:
        errors.append("unexpected missing normal proposal count must be zero")
    if accounting.get("repeated_commit_proposal_ids"):
        errors.append(f"duplicate commit proposal ids present: {accounting['repeated_commit_proposal_ids']}")
    if result_payload:
        errors.extend(validate_result_sanity(result_payload))
    warnings = performance_warnings(
        accounting,
        low_committed_share_threshold=low_committed_share_threshold,
        high_payload_len_per_committed_token_threshold=high_payload_len_per_committed_token_threshold,
    )
    accounting["performance_warnings"] = warnings
    if strict_performance and warnings:
        errors.append(f"strict performance warnings present: {warnings}")
    return errors, accounting


def print_summary(summary: dict[str, Any]) -> None:
    keys = [
        "execution_mode",
        "decode_ready_mode",
        "total_output_tokens",
        "engine_elapsed_s",
        "goodput_tokens_per_s",
        "mean_tpot_ms",
        "eager_candidate_proposal_count",
        "eager_candidate_token_count",
        "eager_ready_proposal_count",
        "eager_ready_token_count",
        "eager_committed_proposal_count",
        "eager_committed_token_count",
        "eager_skipped_proposal_count",
        "eager_skip_reason_counts",
        "eager_commit_rate_by_proposal",
        "eager_commit_rate_by_token",
        "eager_full_accept_rate",
        "eager_partial_reject_rate",
        "committed_token_share_of_output",
        "candidate_token_share_of_output",
        "suppressed_slots_per_committed_token",
        "replaced_slots_per_committed_token",
        "proposal_payload_len_units_per_committed_token",
        "result_payload_len_units_per_committed_token",
        "committed_tokens_per_candidate_token",
        "ready_tokens_per_candidate_token",
        "committed_tokens_per_ready_token",
        "normal_draft_seq_excluded_count",
        "normal_draft_token_slots_suppressed",
        "normal_proposal_missing_allowed_by_eager_count",
        "target_normal_verify_seq_excluded_by_eager_count",
        "target_normal_verify_token_slots_replaced_by_eager",
        "target_actual_eager_verified_token_increment_sum",
        "target_actual_eager_accepted_token_increment_sum",
        "eager_proposal_transfer_payload_bytes",
        "eager_proposal_transfer_payload_len_units",
        "eager_result_transfer_payload_bytes",
        "eager_result_transfer_payload_len_units",
        "eager_accounting_summary_time_ms",
        "timing_available",
        "missing_timing_reason",
        *TIMING_FIELDS,
        "total_eager_overhead_time_ms",
        "performance_warnings",
        "repeated_commit_proposal_ids",
        "missing_buffered_proposal_unexpected_count",
    ]
    for key in keys:
        print(f"{key}={summary.get(key)}")


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
                "arrival_ts": 1.0,
                "decode_start_ts": 1.0,
                "finish_ts": 1.064,
                "decode_elapsed_ms": 64.0,
                "observed_tpot_ms": 16.0,
                "num_output_tokens": 4,
            }
        ],
    }


def synthetic_records() -> list[dict[str, Any]]:
    base = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_commit_readiness_dry_run": True,
        "enable_eager_commit_ready_only": True,
        "eager_commit_enabled": True,
        "eager_commit_source": TAKEOVER_SOURCE,
        "eager_commit_candidate_proposal_ids": [101, 102],
        "eager_commit_candidate_seq_ids": [7, 8],
        "eager_commit_ready_proposal_ids": [101],
        "eager_commit_ready_seq_ids": [7],
        "eager_commit_not_ready_proposal_ids": [102],
        "eager_commit_not_ready_seq_ids": [8],
        "eager_commit_not_ready_reason_by_proposal_id": {"102": "not_full_accept"},
        "eager_commit_from_readiness_proposal_ids": [101],
        "eager_commit_skipped_proposal_ids": [102],
        "eager_commit_skip_reason_by_proposal_id": {"102": "not_full_accept"},
        "eager_commit_candidate_count": 2,
        "eager_commit_committed_count": 1,
        "eager_commit_skipped_count": 1,
        "eager_commit_duplicate_proposal_ids": [],
        "eager_commit_duplicate_seq_ids": [],
        "eager_committed_proposal_ids": [101],
        "eager_committed_seq_ids": [7],
        "eager_committed_token_count_by_proposal_id": {"101": 4},
        "eager_committed_accept_len_by_proposal_id": {"101": 4},
        "eager_committed_action_by_proposal_id": {"101": "append_full_accept_then_rollback"},
        "eager_committed_verify_result_by_proposal_id": {"101": "full_accept"},
        "eager_commit_precondition_ok_by_proposal_id": {"101": True},
        "eager_commit_precondition_failed_by_proposal_id": {"101": False},
        "eager_commit_target_seq_len_before_by_seq_id": {"7": 20},
        "eager_commit_target_seq_len_after_by_seq_id": {"7": 24},
        "eager_commit_draft_seq_len_before_by_seq_id": {"7": 20},
        "eager_commit_draft_seq_len_after_by_seq_id": {"7": 24},
        "eager_commit_target_draft_len_match_by_seq_id": {"7": True},
        "eager_commit_target_draft_token_match_by_seq_id": {"7": True},
        "eager_tokens_committed": 4,
        "eager_tokens_committed_full_accept": 4,
        "eager_tokens_verified": 4,
        "eager_tokens_accepted": 4,
        "eager_tokens_rejected": 0,
        "eager_tokens_invalidated": 0,
        "eager_apply_dry_run_proposal_len_by_proposal_id": {"101": 4, "102": 4},
        "lane_exclusion_applied_proposal_ids": [101, 102],
        "lane_exclusion_applied_seq_ids": [7, 8],
        "target_eager_verify_proposal_ids_dry_run": [101, 102],
        "target_eager_verify_seq_ids_dry_run": [7, 8],
        "missing_buffered_proposal_allowed_by_eager_seq_ids": [7, 8],
        "eager_transfer_payload_len": 12,
        "eager_result_transfer_payload_len": 8,
        "eager_result_transfer_zero_result_step": False,
    }
    target = dict(base, eager_commit_side="target", step_id=11, plan_id=21)
    draft = dict(base, eager_commit_side="draft", step_id=11, plan_id=21)
    return [target, draft]


def run_synthetic_tests() -> None:
    records = synthetic_records()
    errors, summary = validate_accounting(records, synthetic_result_payload())
    assert not errors, f"valid performance accounting synthetic failed: {errors}"
    assert summary["eager_candidate_proposal_count"] == 2
    assert summary["eager_candidate_token_count"] == 8
    assert summary["eager_committed_token_count"] == 4
    assert summary["normal_draft_token_slots_suppressed"] == 8
    assert summary["target_normal_verify_token_slots_replaced_by_eager"] == 8
    assert summary["committed_token_share_of_output"] == 4 / 64
    assert summary["suppressed_slots_per_committed_token"] == 2.0
    assert summary["proposal_payload_len_units_per_committed_token"] == 3.0
    assert summary["timing_available"] is False
    assert summary["missing_timing_reason"] == "not_instrumented"

    invalid = deepcopy(records)
    invalid[0]["eager_tokens_verified"] = 0
    errors, _ = validate_accounting(invalid, synthetic_result_payload())
    assert any("target actual eager verified" in error for error in errors), "missed target counter mismatch"

    invalid = deepcopy(records)
    invalid[0]["lane_exclusion_applied_proposal_ids"] = []
    invalid[1]["lane_exclusion_applied_proposal_ids"] = []
    invalid[0]["lane_exclusion_applied_seq_ids"] = []
    invalid[1]["lane_exclusion_applied_seq_ids"] = []
    errors, _ = validate_accounting(invalid, synthetic_result_payload())
    assert any("normal draft token slots suppressed" in error for error in errors), "missed suppressed-slot mismatch"

    invalid = deepcopy(records)
    invalid[0]["eager_result_transfer_payload_len"] = -1
    errors, _ = validate_accounting(invalid, synthetic_result_payload())
    assert any("payload" in error for error in errors), "missed negative payload accounting"

    invalid_result = synthetic_result_payload()
    invalid_result["traces"][0]["observed_tpot_ms"] = 999.0
    errors, _ = validate_accounting(records, invalid_result)
    assert any("observed_tpot" in error for error in errors), "missed result JSON sanity error"

    errors, summary = validate_accounting(
        records,
        synthetic_result_payload(),
        strict_performance=True,
        low_committed_share_threshold=0.5,
    )
    assert any("strict performance warnings" in error for error in errors), "missed strict performance warning"
    assert "committed_token_share_below_0.5" in summary["performance_warnings"]

    print("Synthetic eager performance accounting checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-6c eager performance accounting consistency.")
    parser.add_argument("trace", nargs="?", type=Path, help="Engine trace JSON.")
    parser.add_argument("result", nargs="?", type=Path, help="Optional eval result JSON.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checks.")
    parser.add_argument("--strict-performance", action="store_true", help="Fail when diagnostic performance warnings fire.")
    parser.add_argument("--low-committed-share-threshold", type=float, default=DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD)
    parser.add_argument(
        "--high-payload-len-per-committed-token-threshold",
        type=float,
        default=DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD,
    )
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result else {}
    errors, summary = validate_accounting(
        records,
        result_payload,
        strict_performance=args.strict_performance,
        low_committed_share_threshold=args.low_committed_share_threshold,
        high_payload_len_per_committed_token_threshold=args.high_payload_len_per_committed_token_threshold,
    )
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Eager performance accounting checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
