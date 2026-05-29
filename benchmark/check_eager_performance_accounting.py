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
from benchmark.bounded_rolling_chain_parser import (
    parse_legacy_rolling_chain,
    summarize_registry,
)


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
    "continuous_eager_overhead_time_ms",
    "continuous_eager_commit_decision_broadcast_time_ms",
    "continuous_eager_result_transfer_time_ms",
    "continuous_eager_sync_apply_dry_run_time_ms",
    "continuous_eager_sync_apply_time_ms",
    "continuous_eager_commit_time_ms",
    "continuous_eager_real_commit_time_ms",
    "rolling_depth2_commit_decision_broadcast_time_ms",
    "rolling_depth2_commit_time_ms",
    "rolling_depth3_shadow_generation_time_ms",
    "rolling_depth4_shadow_generation_time_ms",
    "rolling_depth3_commit_decision_broadcast_time_ms",
    "rolling_depth3_commit_time_ms",
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
    "rolling_depth2_real_committed_token_count_by_proposal_id",
    "rolling_depth3_real_committed_token_count_by_proposal_id",
    "rolling_depth4_child_token_count_by_proposal_id",
]
DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD = 0.01
DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD = 128.0

GENERIC_ACCOUNTING_FIELD_PAIRS = (
    ("eager_committed_token_count", "generic_one_shot_committed_token_count"),
    ("continuous_eager_real_committed_token_count", "generic_depth1_committed_token_count"),
    ("rolling_depth2_real_committed_token_count", "generic_depth2_committed_token_count"),
    ("rolling_depth3_real_committed_token_count", "generic_depth3_committed_token_count"),
    ("combined_real_committed_token_count", "generic_combined_real_committed_token_count"),
    ("rolling_depth_gt3_real_commit_count", "generic_depth_gt3_real_commit_count"),
    ("rolling_depth4_real_commit_count", "generic_depth4_real_commit_count"),
    ("rolling_depth_gt4_real_commit_count", "generic_depth_gt4_real_commit_count"),
)


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
    if (
        int_value(accounting.get("continuous_eager_real_committed_token_count"), 0) > 0
        and float_value(accounting.get("continuous_eager_commit_decision_broadcast_time_ms"), 0.0) <= 0.0
        and float_value(accounting.get("continuous_eager_result_transfer_time_ms"), 0.0) <= 0.0
        and float_value(accounting.get("continuous_eager_sync_apply_dry_run_time_ms"), 0.0) <= 0.0
    ):
        warnings.append("continuous_control_plane_timing_unavailable")
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
    continuous_candidate_ids: set[int] = set()
    continuous_ready_shadow_ids: set[int] = set()
    continuous_verified_ids: set[int] = set()
    continuous_full_accept_ids: set[int] = set()
    continuous_token_by_id: dict[int, int] = {}
    continuous_ready_token_by_id: dict[int, int] = {}
    continuous_verified_token_by_id: dict[int, int] = {}
    continuous_full_accept_token_by_id: dict[int, int] = {}
    continuous_drop_reason_by_id: dict[int, str] = {}
    continuous_chain_depth_by_id: dict[int, int] = {}
    continuous_duplicate_ids: set[int] = set()
    continuous_frontier_mismatch_ids: set[int] = set()
    continuous_true_frontier_mismatch_ids: set[int] = set()
    continuous_parent_shadow_not_committed_ids: set[int] = set()
    continuous_parent_not_ready_ids: set[int] = set()
    continuous_real_commit_enabled = False
    continuous_real_committed_ids: set[int] = set()
    continuous_real_committed_token_by_id: dict[int, int] = {}
    continuous_real_commit_skip_reason_by_id: dict[int, str] = {}
    continuous_target_verified_sum = 0
    continuous_target_accepted_sum = 0
    continuous_target_rejected_sum = 0
    continuous_target_invalidated_sum = 0
    continuous_draft_verified_sum = 0
    continuous_draft_accepted_sum = 0
    continuous_draft_rejected_sum = 0
    continuous_draft_invalidated_sum = 0
    continuous_depth2_real_commit_count = 0
    continuous_result_transfer_payload_len_units = 0
    continuous_commit_decision_payload_len_units = 0
    continuous_result_transfer_payload_len_units_before_compact = 0
    continuous_result_transfer_protocols: set[str] = set()
    continuous_zero_result_fast_path_count = 0
    continuous_zero_decision_fast_path_count = 0
    continuous_sync_apply_zero_steps = 0
    continuous_verify_apply_zero_candidate_steps = 0
    rolling_child_candidate_ids: set[int] = set()
    rolling_child_ready_ids: set[int] = set()
    rolling_child_invalidated_ids: set[int] = set()
    rolling_child_token_by_id: dict[int, int] = {}
    rolling_drop_reason_by_id: dict[int, str] = {}
    rolling_parent_full_accept_ids: set[int] = set()
    rolling_parent_partial_reject_ids: set[int] = set()
    rolling_same_seq_overlap_count = 0
    rolling_normal_lane_conflict_count = 0
    rolling_cascade_discard_ids: set[int] = set()
    rolling_depth2_real_commit_count = 0
    rolling_depth_gt1_real_commit_count = 0
    rolling_depth2_commit_enabled = False
    rolling_depth2_real_committed_ids: set[int] = set()
    rolling_depth2_real_committed_token_by_id: dict[int, int] = {}
    rolling_depth2_real_commit_skip_reason_by_id: dict[int, str] = {}
    rolling_depth2_target_verified_sum = 0
    rolling_depth2_target_accepted_sum = 0
    rolling_depth2_target_rejected_sum = 0
    rolling_depth2_target_invalidated_sum = 0
    rolling_depth2_draft_verified_sum = 0
    rolling_depth2_draft_accepted_sum = 0
    rolling_depth2_draft_rejected_sum = 0
    rolling_depth2_draft_invalidated_sum = 0
    rolling_depth3_real_commit_count = 0
    rolling_depth3_commit_enabled = False
    rolling_depth3_real_committed_ids: set[int] = set()
    rolling_depth3_real_committed_token_by_id: dict[int, int] = {}
    rolling_depth3_real_commit_skip_reason_by_id: dict[int, str] = {}
    rolling_depth3_target_verified_sum = 0
    rolling_depth3_target_accepted_sum = 0
    rolling_depth3_target_rejected_sum = 0
    rolling_depth3_target_invalidated_sum = 0
    rolling_depth3_draft_verified_sum = 0
    rolling_depth3_draft_accepted_sum = 0
    rolling_depth3_draft_rejected_sum = 0
    rolling_depth3_draft_invalidated_sum = 0
    rolling_depth4_real_commit_count = 0
    rolling_depth_gt2_real_commit_count = 0
    rolling_depth3_shadow_enabled = False
    rolling_depth3_child_candidate_ids: set[int] = set()
    rolling_depth3_child_ready_ids: set[int] = set()
    rolling_depth3_child_invalidated_ids: set[int] = set()
    rolling_depth3_child_token_by_id: dict[int, int] = {}
    rolling_depth3_parent_pending_ids: set[int] = set()
    rolling_depth3_same_seq_overlap_count = 0
    rolling_depth3_normal_lane_conflict_count = 0
    rolling_depth_gt3_real_commit_count = 0
    rolling_depth_gt4_real_commit_count = 0
    rolling_depth3_drop_reason_by_id: dict[int, str] = {}
    rolling_depth3_max_depth_observed = 0
    rolling_depth4_shadow_enabled = False
    rolling_depth4_child_candidate_ids: set[int] = set()
    rolling_depth4_child_ready_ids: set[int] = set()
    rolling_depth4_child_invalidated_ids: set[int] = set()
    rolling_depth4_child_token_by_id: dict[int, int] = {}
    rolling_depth4_parent_pending_ids: set[int] = set()
    rolling_depth4_same_seq_overlap_count = 0
    rolling_depth4_normal_lane_conflict_count = 0
    rolling_depth4_drop_reason_by_id: dict[int, str] = {}
    rolling_depth4_max_depth_observed = 0
    rolling_depth2_commit_decision_payload_len_units = 0
    rolling_depth3_commit_decision_payload_len_units = 0
    rolling_max_depth_observed = 0
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
    counted_continuous_commit_events: set[tuple[str, int, int, int]] = set()
    counted_continuous_commit_records: set[tuple[str, int, int]] = set()
    counted_rolling_depth2_commit_events: set[tuple[str, int, int, int]] = set()
    counted_rolling_depth2_commit_records: set[tuple[str, int, int]] = set()
    counted_rolling_depth3_commit_events: set[tuple[str, int, int, int]] = set()
    counted_rolling_depth3_commit_records: set[tuple[str, int, int]] = set()

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

        continuous_candidate_ids.update(
            as_int_set(record.get("continuous_eager_candidate_proposal_ids"))
        )
        continuous_ready_shadow_ids.update(
            as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids"))
        )
        continuous_verified_ids.update(as_int_set(record.get("continuous_eager_verified_proposal_ids")))
        continuous_verified_ids.update(
            as_int_set(record.get("continuous_eager_verify_dry_run_executed_proposal_ids"))
        )
        continuous_full_accept_ids.update(as_int_set(record.get("continuous_eager_full_accept_proposal_ids")))
        for proposal_id, token_count in as_int_map(
            record.get("continuous_eager_candidate_token_count_by_proposal_id")
        ).items():
            if token_count > 0:
                continuous_token_by_id.setdefault(proposal_id, token_count)
        for proposal_id, token_count in as_int_map(
            record.get("continuous_eager_commit_ready_shadow_token_count_by_proposal_id")
        ).items():
            if token_count > 0:
                continuous_ready_token_by_id.setdefault(proposal_id, token_count)
        for proposal_id in as_int_set(record.get("continuous_eager_verified_proposal_ids")):
            continuous_verified_token_by_id.setdefault(
                proposal_id,
                continuous_token_by_id.get(proposal_id, max(0, gamma)),
            )
        for proposal_id in as_int_set(record.get("continuous_eager_full_accept_proposal_ids")):
            continuous_full_accept_token_by_id.setdefault(
                proposal_id,
                continuous_token_by_id.get(proposal_id, max(0, gamma)),
            )
        for proposal_id, depth in as_int_map(
            record.get("continuous_eager_chain_depth_by_proposal_id")
        ).items():
            if depth > 0:
                continuous_chain_depth_by_id.setdefault(proposal_id, depth)
        reason_map = record.get("continuous_eager_not_ready_shadow_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    continuous_drop_reason_by_id.setdefault(proposal_id, str(reason))
        continuous_duplicate_ids.update(as_int_set(record.get("continuous_eager_duplicate_proposal_ids")))
        continuous_frontier_mismatch_ids.update(
            as_int_set(record.get("continuous_eager_frontier_mismatch_proposal_ids"))
        )
        continuous_true_frontier_mismatch_ids.update(
            as_int_set(record.get("continuous_eager_true_frontier_mismatch_proposal_ids"))
        )
        continuous_parent_shadow_not_committed_ids.update(
            as_int_set(record.get("continuous_eager_parent_shadow_not_committed_proposal_ids"))
        )
        continuous_parent_not_ready_ids.update(
            as_int_set(record.get("continuous_eager_parent_not_ready_proposal_ids"))
        )
        if bool(record.get("enable_continuous_eager_commit_depth1_ready_only", False)):
            continuous_real_commit_enabled = True
        continuous_depth2_real_commit_count += int_value(record.get("continuous_depth2_real_commit_count"), 0)
        continuous_committed_ids = as_int_set(record.get("continuous_eager_real_committed_proposal_ids"))
        continuous_real_committed_ids.update(continuous_committed_ids)
        continuous_commit_token_by_id = as_int_map(
            record.get("continuous_eager_real_committed_token_count_by_proposal_id")
        )
        for proposal_id in continuous_committed_ids:
            token_count = int(continuous_commit_token_by_id.get(proposal_id, continuous_token_by_id.get(proposal_id, max(0, gamma))))
            if token_count > 0:
                continuous_real_committed_token_by_id.setdefault(proposal_id, token_count)
        continuous_skip_map = record.get("continuous_eager_real_commit_skip_reason_by_proposal_id")
        if isinstance(continuous_skip_map, dict):
            for raw_proposal_id, reason in continuous_skip_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    continuous_real_commit_skip_reason_by_id.setdefault(proposal_id, str(reason))
        continuous_side = str(record.get("continuous_eager_commit_side") or "")
        if continuous_committed_ids and continuous_side in {"target", "draft"}:
            commit_plan = int_value(record.get("continuous_eager_commit_plan_id"), key[0])
            commit_step = int_value(record.get("continuous_eager_commit_step_id"), key[1])
            record_event = (continuous_side, commit_plan, commit_step)
            record_token_sum = 0
            for proposal_id in continuous_committed_ids:
                event = (continuous_side, commit_plan, commit_step, proposal_id)
                if event in counted_continuous_commit_events:
                    continue
                counted_continuous_commit_events.add(event)
                token_count = int(continuous_commit_token_by_id.get(proposal_id, continuous_token_by_id.get(proposal_id, max(0, gamma))))
                record_token_sum += token_count
            if record_event not in counted_continuous_commit_records:
                counted_continuous_commit_records.add(record_event)
                if continuous_side == "target":
                    continuous_target_verified_sum += int_value(
                        record.get("continuous_eager_tokens_verified"),
                        record_token_sum,
                    )
                    continuous_target_accepted_sum += int_value(
                        record.get("continuous_eager_tokens_accepted"),
                        record_token_sum,
                    )
                    continuous_target_rejected_sum += int_value(record.get("continuous_eager_tokens_rejected"), 0)
                    continuous_target_invalidated_sum += int_value(record.get("continuous_eager_tokens_invalidated"), 0)
                else:
                    continuous_draft_verified_sum += int_value(
                        record.get("continuous_eager_tokens_verified"),
                        record_token_sum,
                    )
                    continuous_draft_accepted_sum += int_value(
                        record.get("continuous_eager_tokens_accepted"),
                        record_token_sum,
                    )
                    continuous_draft_rejected_sum += int_value(record.get("continuous_eager_tokens_rejected"), 0)
                    continuous_draft_invalidated_sum += int_value(record.get("continuous_eager_tokens_invalidated"), 0)
        continuous_result_transfer_payload_len_units += max(
            0,
            int_value(record.get("continuous_eager_result_transfer_payload_len_units"), 0),
        )
        continuous_result_transfer_payload_len_units_before_compact += max(
            0,
            int_value(record.get("continuous_eager_result_transfer_payload_len_units_before_compact"), 0),
        )
        protocol = record.get("continuous_eager_result_transfer_protocol")
        if protocol:
            continuous_result_transfer_protocols.add(str(protocol))
        continuous_zero_result_fast_path_count += int_value(record.get("continuous_zero_result_fast_path_count"), 0)
        continuous_zero_decision_fast_path_count += int_value(record.get("continuous_zero_decision_fast_path_count"), 0)
        continuous_sync_apply_zero_steps += int_value(record.get("continuous_eager_sync_apply_zero_steps"), 0)
        continuous_verify_apply_zero_candidate_steps += int_value(
            record.get("continuous_eager_verify_apply_zero_candidate_steps"),
            0,
        )
        rolling_child_candidate_ids.update(as_int_set(record.get("rolling_child_generated_proposal_ids")))
        rolling_child_ready_ids.update(
            as_int_set(record.get("rolling_child_ready_after_parent_full_accept_proposal_ids"))
        )
        rolling_child_invalidated_ids.update(as_int_set(record.get("rolling_child_invalidated_proposal_ids")))
        rolling_parent_full_accept_ids.update(as_int_set(record.get("rolling_parent_full_accept_proposal_ids")))
        rolling_parent_partial_reject_ids.update(as_int_set(record.get("rolling_parent_partial_reject_proposal_ids")))
        for proposal_id in as_int_set(record.get("rolling_child_generated_proposal_ids")):
            rolling_child_token_by_id.setdefault(proposal_id, max(0, gamma))
        reason_map = record.get("rolling_child_invalidated_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                rolling_drop_reason_by_id.setdefault(proposal_id, str(reason))
        rolling_same_seq_overlap_count += int_value(record.get("rolling_same_seq_overlap_count"), 0)
        rolling_normal_lane_conflict_count += int_value(record.get("rolling_normal_lane_conflict_count"), 0)
        rolling_cascade_discard_ids.update(as_int_set(record.get("rolling_cascade_discarded_proposal_ids")))
        rolling_depth2_real_commit_count += int_value(record.get("rolling_depth2_real_commit_count"), 0)
        rolling_depth_gt1_real_commit_count += int_value(record.get("rolling_depth_gt1_real_commit_count"), 0)
        if bool(record.get("enable_rolling_continuous_depth2_commit_ready_only", False)):
            rolling_depth2_commit_enabled = True
        rolling_depth3_real_commit_count += int_value(record.get("rolling_depth3_real_commit_count"), 0)
        rolling_depth_gt2_real_commit_count += int_value(record.get("rolling_depth_gt2_real_commit_count"), 0)
        if bool(record.get("enable_rolling_continuous_depth3_shadow_dry_run", False)):
            rolling_depth3_shadow_enabled = True
        if bool(record.get("enable_rolling_continuous_depth3_commit_ready_only", False)) or bool(
            record.get("rolling_depth3_commit_enabled", False)
        ):
            rolling_depth3_commit_enabled = True
        rolling_depth4_real_commit_count += int_value(record.get("rolling_depth4_real_commit_count"), 0)
        rolling_depth_gt3_real_commit_count += int_value(record.get("rolling_depth_gt3_real_commit_count"), 0)
        rolling_depth_gt4_real_commit_count += int_value(record.get("rolling_depth_gt4_real_commit_count"), 0)
        if bool(record.get("enable_rolling_continuous_depth4_shadow_dry_run", False)) or bool(
            record.get("rolling_depth4_shadow_enabled", False)
        ):
            rolling_depth4_shadow_enabled = True
        rolling_depth3_child_candidate_ids.update(
            as_int_set(record.get("rolling_depth3_child_generated_proposal_ids"))
        )
        rolling_depth3_child_ready_ids.update(
            as_int_set(record.get("rolling_depth3_child_ready_shadow_proposal_ids"))
        )
        rolling_depth3_child_invalidated_ids.update(
            as_int_set(record.get("rolling_depth3_child_invalidated_proposal_ids"))
        )
        rolling_depth3_parent_pending_ids.update(
            as_int_set(record.get("rolling_depth3_parent_resolution_pending_proposal_ids"))
        )
        rolling_depth3_same_seq_overlap_count += int_value(record.get("rolling_depth3_same_seq_overlap_count"), 0)
        rolling_depth3_normal_lane_conflict_count += int_value(
            record.get("rolling_depth3_normal_lane_conflict_count"),
            0,
        )
        rolling_depth3_max_depth_observed = max(
            rolling_depth3_max_depth_observed,
            int_value(record.get("rolling_depth3_max_depth_observed"), 0),
        )
        for proposal_id, token_count in as_int_map(
            record.get("rolling_depth3_child_token_count_by_proposal_id")
        ).items():
            if token_count > 0:
                rolling_depth3_child_token_by_id.setdefault(proposal_id, token_count)
        for proposal_id in as_int_set(record.get("rolling_depth3_child_generated_proposal_ids")):
            rolling_depth3_child_token_by_id.setdefault(proposal_id, max(0, gamma))
        reason_map = record.get("rolling_depth3_child_invalidated_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    rolling_depth3_drop_reason_by_id.setdefault(proposal_id, str(reason))

        rolling_depth4_child_candidate_ids.update(
            as_int_set(record.get("rolling_depth4_child_generated_proposal_ids"))
        )
        rolling_depth4_child_ready_ids.update(
            as_int_set(record.get("rolling_depth4_child_ready_shadow_proposal_ids"))
        )
        rolling_depth4_child_invalidated_ids.update(
            as_int_set(record.get("rolling_depth4_child_invalidated_proposal_ids"))
        )
        rolling_depth4_parent_pending_ids.update(
            as_int_set(record.get("rolling_depth4_parent_resolution_pending_proposal_ids"))
        )
        rolling_depth4_same_seq_overlap_count += int_value(record.get("rolling_depth4_same_seq_overlap_count"), 0)
        rolling_depth4_normal_lane_conflict_count += int_value(
            record.get("rolling_depth4_normal_lane_conflict_count"),
            0,
        )
        rolling_depth4_max_depth_observed = max(
            rolling_depth4_max_depth_observed,
            int_value(record.get("rolling_depth4_max_depth_observed"), 0),
        )
        for proposal_id, token_count in as_int_map(
            record.get("rolling_depth4_child_token_count_by_proposal_id")
        ).items():
            if token_count > 0:
                rolling_depth4_child_token_by_id.setdefault(proposal_id, token_count)
        for proposal_id in as_int_set(record.get("rolling_depth4_child_generated_proposal_ids")):
            rolling_depth4_child_token_by_id.setdefault(proposal_id, max(0, gamma))
        reason_map = record.get("rolling_depth4_child_invalidated_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                rolling_depth4_drop_reason_by_id.setdefault(proposal_id, str(reason))
        reason_map = record.get("rolling_depth4_child_generation_skip_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                rolling_depth4_drop_reason_by_id.setdefault(proposal_id, str(reason))
        skip_reason_map = record.get("rolling_depth3_child_generation_skip_reason_by_proposal_id")
        if isinstance(skip_reason_map, dict):
            for raw_proposal_id, reason in skip_reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    rolling_depth3_drop_reason_by_id.setdefault(proposal_id, str(reason))
        rolling_depth2_committed_ids = as_int_set(record.get("rolling_depth2_real_committed_proposal_ids"))
        rolling_depth2_real_committed_ids.update(rolling_depth2_committed_ids)
        rolling_depth2_commit_token_by_id = as_int_map(
            record.get("rolling_depth2_real_committed_token_count_by_proposal_id")
        )
        for proposal_id in rolling_depth2_committed_ids:
            token_count = int(
                rolling_depth2_commit_token_by_id.get(
                    proposal_id,
                    rolling_child_token_by_id.get(proposal_id, max(0, gamma)),
                )
            )
            if token_count > 0:
                rolling_depth2_real_committed_token_by_id.setdefault(proposal_id, token_count)
        reason_map = record.get("rolling_depth2_real_commit_skip_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    rolling_depth2_real_commit_skip_reason_by_id.setdefault(proposal_id, str(reason))
        rolling_side = str(record.get("rolling_depth2_commit_side") or "")
        if rolling_depth2_committed_ids and rolling_side in {"target", "draft"}:
            commit_plan = int_value(record.get("rolling_depth2_commit_plan_id"), key[0])
            commit_step = int_value(record.get("rolling_depth2_commit_step_id"), key[1])
            record_event = (rolling_side, commit_plan, commit_step)
            record_token_sum = 0
            for proposal_id in rolling_depth2_committed_ids:
                event = (rolling_side, commit_plan, commit_step, proposal_id)
                if event in counted_rolling_depth2_commit_events:
                    continue
                counted_rolling_depth2_commit_events.add(event)
                token_count = int(
                    rolling_depth2_commit_token_by_id.get(
                        proposal_id,
                        rolling_child_token_by_id.get(proposal_id, max(0, gamma)),
                    )
                )
                record_token_sum += token_count
            if record_event not in counted_rolling_depth2_commit_records:
                counted_rolling_depth2_commit_records.add(record_event)
                if rolling_side == "target":
                    rolling_depth2_target_verified_sum += int_value(
                        record.get("rolling_depth2_tokens_verified"),
                        record_token_sum,
                    )
                    rolling_depth2_target_accepted_sum += int_value(
                        record.get("rolling_depth2_tokens_accepted"),
                        record_token_sum,
                    )
                    rolling_depth2_target_rejected_sum += int_value(record.get("rolling_depth2_tokens_rejected"), 0)
                    rolling_depth2_target_invalidated_sum += int_value(record.get("rolling_depth2_tokens_invalidated"), 0)
                else:
                    rolling_depth2_draft_verified_sum += int_value(
                        record.get("rolling_depth2_tokens_verified"),
                        record_token_sum,
                    )
                    rolling_depth2_draft_accepted_sum += int_value(
                        record.get("rolling_depth2_tokens_accepted"),
                        record_token_sum,
                    )
                    rolling_depth2_draft_rejected_sum += int_value(record.get("rolling_depth2_tokens_rejected"), 0)
                    rolling_depth2_draft_invalidated_sum += int_value(record.get("rolling_depth2_tokens_invalidated"), 0)
        if "rolling_depth2_commit_decision_broadcast_payload_len_units" in record:
            payload_len = int_value(record.get("rolling_depth2_commit_decision_broadcast_payload_len_units"), 0)
            if payload_len < 0:
                negative_payload_field_count += 1
            event_key = (
                "rolling_depth2_commit_decision_broadcast_payload_len_units",
                key[0],
                key[1],
                payload_len,
            )
            if payload_len > 0 and event_key not in counted_payload_len_events:
                counted_payload_len_events.add(event_key)
                rolling_depth2_commit_decision_payload_len_units += payload_len
        rolling_depth3_committed_ids = as_int_set(record.get("rolling_depth3_real_committed_proposal_ids"))
        rolling_depth3_real_committed_ids.update(rolling_depth3_committed_ids)
        rolling_depth3_commit_token_by_id = as_int_map(
            record.get("rolling_depth3_real_committed_token_count_by_proposal_id")
        )
        for proposal_id in rolling_depth3_committed_ids:
            token_count = int(
                rolling_depth3_commit_token_by_id.get(
                    proposal_id,
                    rolling_depth3_child_token_by_id.get(proposal_id, max(0, gamma)),
                )
            )
            if token_count > 0:
                rolling_depth3_real_committed_token_by_id.setdefault(proposal_id, token_count)
        reason_map = record.get("rolling_depth3_real_commit_skip_reason_by_proposal_id")
        if isinstance(reason_map, dict):
            for raw_proposal_id, reason in reason_map.items():
                try:
                    proposal_id = int(raw_proposal_id)
                except Exception:
                    continue
                if reason:
                    rolling_depth3_real_commit_skip_reason_by_id.setdefault(proposal_id, str(reason))
        rolling3_side = str(record.get("rolling_depth3_commit_side") or "")
        if rolling_depth3_committed_ids and rolling3_side in {"target", "draft"}:
            commit_plan = int_value(record.get("rolling_depth3_commit_plan_id"), key[0])
            commit_step = int_value(record.get("rolling_depth3_commit_step_id"), key[1])
            record_event = (rolling3_side, commit_plan, commit_step)
            record_token_sum = 0
            for proposal_id in rolling_depth3_committed_ids:
                event = (rolling3_side, commit_plan, commit_step, proposal_id)
                if event in counted_rolling_depth3_commit_events:
                    continue
                counted_rolling_depth3_commit_events.add(event)
                token_count = int(
                    rolling_depth3_commit_token_by_id.get(
                        proposal_id,
                        rolling_depth3_child_token_by_id.get(proposal_id, max(0, gamma)),
                    )
                )
                record_token_sum += token_count
            if record_event not in counted_rolling_depth3_commit_records:
                counted_rolling_depth3_commit_records.add(record_event)
                if rolling3_side == "target":
                    rolling_depth3_target_verified_sum += int_value(
                        record.get("rolling_depth3_tokens_verified"),
                        record_token_sum,
                    )
                    rolling_depth3_target_accepted_sum += int_value(
                        record.get("rolling_depth3_tokens_accepted"),
                        record_token_sum,
                    )
                    rolling_depth3_target_rejected_sum += int_value(record.get("rolling_depth3_tokens_rejected"), 0)
                    rolling_depth3_target_invalidated_sum += int_value(record.get("rolling_depth3_tokens_invalidated"), 0)
                else:
                    rolling_depth3_draft_verified_sum += int_value(
                        record.get("rolling_depth3_tokens_verified"),
                        record_token_sum,
                    )
                    rolling_depth3_draft_accepted_sum += int_value(
                        record.get("rolling_depth3_tokens_accepted"),
                        record_token_sum,
                    )
                    rolling_depth3_draft_rejected_sum += int_value(record.get("rolling_depth3_tokens_rejected"), 0)
                    rolling_depth3_draft_invalidated_sum += int_value(record.get("rolling_depth3_tokens_invalidated"), 0)
        if "rolling_depth3_commit_decision_broadcast_payload_len_units" in record:
            payload_len = int_value(record.get("rolling_depth3_commit_decision_broadcast_payload_len_units"), 0)
            if payload_len < 0:
                negative_payload_field_count += 1
            event_key = (
                "rolling_depth3_commit_decision_broadcast_payload_len_units",
                key[0],
                key[1],
                payload_len,
            )
            if payload_len > 0 and event_key not in counted_payload_len_events:
                counted_payload_len_events.add(event_key)
                rolling_depth3_commit_decision_payload_len_units += payload_len
        rolling_max_depth_observed = max(
            rolling_max_depth_observed,
            int_value(record.get("rolling_max_depth_observed"), 0),
            int_value(record.get("max_rolling_continuous_depth_observed"), 0),
        )
        if "continuous_eager_commit_decision_broadcast_payload_len_units" in record:
            payload_len = int_value(record.get("continuous_eager_commit_decision_broadcast_payload_len_units"), 0)
            if payload_len < 0:
                negative_payload_field_count += 1
            event_key = (
                "continuous_eager_commit_decision_broadcast_payload_len_units",
                key[0],
                key[1],
                payload_len,
            )
            if payload_len > 0 and event_key not in counted_payload_len_events:
                counted_payload_len_events.add(event_key)
                continuous_commit_decision_payload_len_units += payload_len

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
    continuous_candidate_token_count = sum(
        int(continuous_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in continuous_candidate_ids
    )
    continuous_ready_shadow_token_count = sum(
        int(continuous_ready_token_by_id.get(proposal_id, continuous_token_by_id.get(proposal_id, max(0, gamma))))
        for proposal_id in continuous_ready_shadow_ids
    )
    continuous_verified_token_count = sum(
        int(continuous_verified_token_by_id.get(proposal_id, continuous_token_by_id.get(proposal_id, max(0, gamma))))
        for proposal_id in continuous_verified_ids
    )
    continuous_full_accept_token_count = sum(
        int(continuous_full_accept_token_by_id.get(proposal_id, continuous_token_by_id.get(proposal_id, max(0, gamma))))
        for proposal_id in continuous_full_accept_ids
    )
    continuous_real_committed_token_count = sum(
        int(continuous_real_committed_token_by_id.get(proposal_id, continuous_token_by_id.get(proposal_id, max(0, gamma))))
        for proposal_id in continuous_real_committed_ids
    )
    rolling_child_candidate_token_count = sum(
        int(rolling_child_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in rolling_child_candidate_ids
    )
    rolling_child_ready_shadow_token_count = sum(
        int(rolling_child_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in rolling_child_ready_ids
    )
    rolling_depth2_real_committed_token_count = sum(
        int(rolling_depth2_real_committed_token_by_id.get(proposal_id, rolling_child_token_by_id.get(proposal_id, max(0, gamma))))
        for proposal_id in rolling_depth2_real_committed_ids
    )
    rolling_depth3_child_candidate_token_count = sum(
        int(rolling_depth3_child_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in rolling_depth3_child_candidate_ids
    )
    rolling_depth3_child_ready_shadow_token_count = sum(
        int(rolling_depth3_child_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in rolling_depth3_child_ready_ids
    )
    rolling_depth3_real_committed_token_count = sum(
        int(
            rolling_depth3_real_committed_token_by_id.get(
                proposal_id,
                rolling_depth3_child_token_by_id.get(proposal_id, max(0, gamma)),
            )
        )
        for proposal_id in rolling_depth3_real_committed_ids
    )
    rolling_depth4_child_candidate_token_count = sum(
        int(rolling_depth4_child_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in rolling_depth4_child_candidate_ids
    )
    rolling_depth4_child_ready_shadow_token_count = sum(
        int(rolling_depth4_child_token_by_id.get(proposal_id, max(0, gamma)))
        for proposal_id in rolling_depth4_child_ready_ids
    )
    continuous_chain_distribution = Counter(
        str(depth)
        for proposal_id, depth in continuous_chain_depth_by_id.items()
        if proposal_id in continuous_candidate_ids
    )
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
        "continuous_eager_candidate_proposal_count": len(continuous_candidate_ids),
        "continuous_eager_candidate_token_count": continuous_candidate_token_count,
        "continuous_eager_verified_proposal_count": len(continuous_verified_ids),
        "continuous_eager_verified_token_count": continuous_verified_token_count,
        "continuous_eager_full_accept_proposal_count": len(continuous_full_accept_ids),
        "continuous_eager_full_accept_token_count": continuous_full_accept_token_count,
        "continuous_eager_commit_ready_shadow_proposal_count": len(continuous_ready_shadow_ids),
        "continuous_eager_commit_ready_shadow_token_count": continuous_ready_shadow_token_count,
        "continuous_eager_estimated_committed_token_share_of_output": safe_div(
            continuous_ready_shadow_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "combined_one_shot_plus_continuous_shadow_token_count": (
            committed_token_count + continuous_ready_shadow_token_count
        ),
        "combined_estimated_token_share_of_output": safe_div(
            committed_token_count + continuous_ready_shadow_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "continuous_eager_chain_length_distribution": dict(continuous_chain_distribution),
        "continuous_eager_drop_reason_counts": dict(Counter(continuous_drop_reason_by_id.values())),
        "continuous_eager_duplicate_count": len(continuous_duplicate_ids),
        "continuous_eager_frontier_mismatch_count": len(continuous_frontier_mismatch_ids),
        "continuous_true_frontier_mismatch_count": len(continuous_true_frontier_mismatch_ids),
        "continuous_parent_shadow_not_committed_count": len(continuous_parent_shadow_not_committed_ids),
        "continuous_parent_shadow_not_ready_count": len(continuous_parent_not_ready_ids),
        "continuous_eager_real_commit_enabled": bool(continuous_real_commit_enabled),
        "continuous_eager_real_commit_count": len(continuous_real_committed_ids),
        "continuous_eager_real_committed_proposal_count": len(continuous_real_committed_ids),
        "continuous_eager_real_committed_token_count": continuous_real_committed_token_count,
        "continuous_real_committed_token_share_of_output": safe_div(
            continuous_real_committed_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "continuous_shadow_ready_but_not_committed_count": len(
            continuous_ready_shadow_ids - continuous_real_committed_ids
        ),
        "continuous_commit_depth1_rate_by_token": safe_div(
            continuous_real_committed_token_count,
            continuous_ready_shadow_token_count,
        ),
        "continuous_commit_depth1_rate_by_proposal": safe_div(
            len(continuous_real_committed_ids),
            len(continuous_ready_shadow_ids),
        ),
        "continuous_target_actual_verified_token_increment_sum": continuous_target_verified_sum,
        "continuous_target_actual_accepted_token_increment_sum": continuous_target_accepted_sum,
        "continuous_target_actual_rejected_token_increment_sum": continuous_target_rejected_sum,
        "continuous_target_actual_invalidated_token_increment_sum": continuous_target_invalidated_sum,
        "continuous_draft_actual_verified_token_increment_sum": continuous_draft_verified_sum,
        "continuous_draft_actual_accepted_token_increment_sum": continuous_draft_accepted_sum,
        "continuous_draft_actual_rejected_token_increment_sum": continuous_draft_rejected_sum,
        "continuous_draft_actual_invalidated_token_increment_sum": continuous_draft_invalidated_sum,
        "continuous_eager_real_commit_skip_reason_counts": dict(Counter(continuous_real_commit_skip_reason_by_id.values())),
        "continuous_depth2_real_commit_count": continuous_depth2_real_commit_count,
        "rolling_depth2_commit_enabled": bool(rolling_depth2_commit_enabled),
        "rolling_depth2_real_committed_proposal_count": len(rolling_depth2_real_committed_ids),
        "rolling_depth2_real_committed_token_count": rolling_depth2_real_committed_token_count,
        "rolling_depth2_real_committed_token_share_of_output": safe_div(
            rolling_depth2_real_committed_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "rolling_depth2_commit_rate_by_token": safe_div(
            rolling_depth2_real_committed_token_count,
            rolling_child_ready_shadow_token_count,
        ),
        "rolling_depth2_commit_rate_by_proposal": safe_div(
            len(rolling_depth2_real_committed_ids),
            len(rolling_child_ready_ids),
        ),
        "rolling_depth2_target_actual_verified_token_increment_sum": rolling_depth2_target_verified_sum,
        "rolling_depth2_target_actual_accepted_token_increment_sum": rolling_depth2_target_accepted_sum,
        "rolling_depth2_target_actual_rejected_token_increment_sum": rolling_depth2_target_rejected_sum,
        "rolling_depth2_target_actual_invalidated_token_increment_sum": rolling_depth2_target_invalidated_sum,
        "rolling_depth2_draft_actual_verified_token_increment_sum": rolling_depth2_draft_verified_sum,
        "rolling_depth2_draft_actual_accepted_token_increment_sum": rolling_depth2_draft_accepted_sum,
        "rolling_depth2_draft_actual_rejected_token_increment_sum": rolling_depth2_draft_rejected_sum,
        "rolling_depth2_draft_actual_invalidated_token_increment_sum": rolling_depth2_draft_invalidated_sum,
        "rolling_depth2_real_commit_skip_reason_counts": dict(Counter(rolling_depth2_real_commit_skip_reason_by_id.values())),
        "rolling_depth3_commit_enabled": bool(rolling_depth3_commit_enabled),
        "rolling_depth3_real_commit_count": rolling_depth3_real_commit_count,
        "rolling_depth3_real_committed_proposal_count": len(rolling_depth3_real_committed_ids),
        "rolling_depth3_real_committed_token_count": rolling_depth3_real_committed_token_count,
        "rolling_depth3_real_committed_token_share_of_output": safe_div(
            rolling_depth3_real_committed_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "rolling_depth3_commit_rate_by_token": safe_div(
            rolling_depth3_real_committed_token_count,
            rolling_depth3_child_ready_shadow_token_count,
        ),
        "rolling_depth3_commit_rate_by_proposal": safe_div(
            len(rolling_depth3_real_committed_ids),
            len(rolling_depth3_child_ready_ids),
        ),
        "rolling_depth3_target_actual_verified_token_increment_sum": rolling_depth3_target_verified_sum,
        "rolling_depth3_target_actual_accepted_token_increment_sum": rolling_depth3_target_accepted_sum,
        "rolling_depth3_target_actual_rejected_token_increment_sum": rolling_depth3_target_rejected_sum,
        "rolling_depth3_target_actual_invalidated_token_increment_sum": rolling_depth3_target_invalidated_sum,
        "rolling_depth3_draft_actual_verified_token_increment_sum": rolling_depth3_draft_verified_sum,
        "rolling_depth3_draft_actual_accepted_token_increment_sum": rolling_depth3_draft_accepted_sum,
        "rolling_depth3_draft_actual_rejected_token_increment_sum": rolling_depth3_draft_rejected_sum,
        "rolling_depth3_draft_actual_invalidated_token_increment_sum": rolling_depth3_draft_invalidated_sum,
        "rolling_depth3_real_commit_skip_reason_counts": dict(
            Counter(rolling_depth3_real_commit_skip_reason_by_id.values())
        ),
        "rolling_depth4_real_commit_count": rolling_depth4_real_commit_count,
        "rolling_depth_gt2_real_commit_count": rolling_depth_gt2_real_commit_count,
        "rolling_depth2_commit_decision_broadcast_payload_len_units": rolling_depth2_commit_decision_payload_len_units,
        "rolling_depth3_shadow_enabled": bool(rolling_depth3_shadow_enabled),
        "rolling_depth3_child_candidate_proposal_count": len(rolling_depth3_child_candidate_ids),
        "rolling_depth3_child_candidate_token_count": rolling_depth3_child_candidate_token_count,
        "rolling_depth3_child_ready_shadow_proposal_count": len(rolling_depth3_child_ready_ids),
        "rolling_depth3_child_ready_shadow_token_count": rolling_depth3_child_ready_shadow_token_count,
        "rolling_depth3_child_invalidated_count": len(rolling_depth3_child_invalidated_ids),
        "rolling_depth3_parent_resolution_pending_count": len(rolling_depth3_parent_pending_ids),
        "rolling_depth3_same_seq_overlap_count": rolling_depth3_same_seq_overlap_count,
        "rolling_depth3_normal_lane_conflict_count": rolling_depth3_normal_lane_conflict_count,
        "rolling_depth_gt3_real_commit_count": rolling_depth_gt3_real_commit_count,
        "rolling_depth3_drop_reason_counts": dict(Counter(rolling_depth3_drop_reason_by_id.values())),
        "rolling_depth3_max_depth_observed": rolling_depth3_max_depth_observed,
        "rolling_depth3_estimated_future_token_count": rolling_depth3_child_ready_shadow_token_count,
        "rolling_depth3_estimated_future_token_share_of_output": safe_div(
            rolling_depth3_child_ready_shadow_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "rolling_depth4_shadow_enabled": bool(rolling_depth4_shadow_enabled),
        "rolling_depth4_child_candidate_proposal_count": len(rolling_depth4_child_candidate_ids),
        "rolling_depth4_child_candidate_token_count": rolling_depth4_child_candidate_token_count,
        "rolling_depth4_child_ready_shadow_proposal_count": len(rolling_depth4_child_ready_ids),
        "rolling_depth4_child_ready_shadow_token_count": rolling_depth4_child_ready_shadow_token_count,
        "rolling_depth4_child_invalidated_count": len(rolling_depth4_child_invalidated_ids),
        "rolling_depth4_parent_resolution_pending_count": len(rolling_depth4_parent_pending_ids),
        "rolling_depth4_same_seq_overlap_count": rolling_depth4_same_seq_overlap_count,
        "rolling_depth4_normal_lane_conflict_count": rolling_depth4_normal_lane_conflict_count,
        "rolling_depth4_real_committed_token_count": 0,
        "rolling_depth_gt4_real_commit_count": rolling_depth_gt4_real_commit_count,
        "rolling_depth4_drop_reason_counts": dict(Counter(rolling_depth4_drop_reason_by_id.values())),
        "rolling_depth4_max_depth_observed": rolling_depth4_max_depth_observed,
        "rolling_depth4_estimated_future_token_count": rolling_depth4_child_ready_shadow_token_count,
        "rolling_depth4_estimated_future_token_share_of_output": safe_div(
            rolling_depth4_child_ready_shadow_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "combined_real_committed_token_count": (
            committed_token_count
            + continuous_real_committed_token_count
            + rolling_depth2_real_committed_token_count
            + rolling_depth3_real_committed_token_count
        ),
        "combined_real_committed_token_share_of_output": safe_div(
            committed_token_count
            + continuous_real_committed_token_count
            + rolling_depth2_real_committed_token_count
            + rolling_depth3_real_committed_token_count,
            int_value(result_metrics(result_payload).get("total_output_tokens"), 0),
        ),
        "combined_actual_verified_token_increment_sum": (
            int_value(commit_summary.get("target_actual_eager_verified_token_increment_sum"), 0)
            + continuous_target_verified_sum
            + rolling_depth2_target_verified_sum
            + rolling_depth3_target_verified_sum
        ),
        "combined_actual_accepted_token_increment_sum": (
            int_value(commit_summary.get("target_actual_eager_accepted_token_increment_sum"), 0)
            + continuous_target_accepted_sum
            + rolling_depth2_target_accepted_sum
            + rolling_depth3_target_accepted_sum
        ),
        "continuous_eager_payload_len_units_per_ready_token": 0.0,
        "continuous_eager_result_transfer_protocol": (
            sorted(continuous_result_transfer_protocols)[0]
            if len(continuous_result_transfer_protocols) == 1
            else None
        ),
        "continuous_eager_result_transfer_protocols": sorted(continuous_result_transfer_protocols),
        "continuous_eager_result_transfer_payload_len_units_before_compact": (
            continuous_result_transfer_payload_len_units_before_compact
        ),
        "continuous_eager_commit_decision_broadcast_payload_len_units": continuous_commit_decision_payload_len_units,
        "continuous_eager_result_transfer_payload_len_units": continuous_result_transfer_payload_len_units,
        "rolling_depth3_commit_decision_broadcast_payload_len_units": rolling_depth3_commit_decision_payload_len_units,
        "continuous_zero_result_fast_path_count": continuous_zero_result_fast_path_count,
        "continuous_zero_decision_fast_path_count": continuous_zero_decision_fast_path_count,
        "continuous_eager_sync_apply_zero_steps": continuous_sync_apply_zero_steps,
        "continuous_eager_verify_apply_zero_candidate_steps": continuous_verify_apply_zero_candidate_steps,
        "rolling_child_candidate_proposal_count": len(rolling_child_candidate_ids),
        "rolling_child_candidate_token_count": rolling_child_candidate_token_count,
        "rolling_child_ready_shadow_proposal_count": len(rolling_child_ready_ids),
        "rolling_child_ready_shadow_token_count": rolling_child_ready_shadow_token_count,
        "rolling_child_invalidated_count": len(rolling_child_invalidated_ids),
        "rolling_cascade_discard_count": len(rolling_cascade_discard_ids),
        "rolling_parent_full_accept_count": len(rolling_parent_full_accept_ids),
        "rolling_parent_partial_reject_count": len(rolling_parent_partial_reject_ids),
        "rolling_parent_resolution_pending_count": max(
            0,
            len(rolling_child_candidate_ids) - len(rolling_parent_full_accept_ids) - len(rolling_parent_partial_reject_ids),
        ),
        "rolling_same_seq_overlap_count": rolling_same_seq_overlap_count,
        "rolling_normal_lane_conflict_count": rolling_normal_lane_conflict_count,
        "rolling_depth2_real_commit_count": rolling_depth2_real_commit_count,
        "rolling_depth_gt1_real_commit_count": rolling_depth_gt1_real_commit_count,
        "rolling_max_depth_observed": rolling_max_depth_observed,
        "rolling_drop_reason_counts": dict(Counter(rolling_drop_reason_by_id.values())),
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


def generic_chain_accounting_errors(records: list[dict[str, Any]], accounting: dict[str, Any]) -> list[str]:
    registry = parse_legacy_rolling_chain(records)
    generic_summary = summarize_registry(registry, accounting=accounting)
    errors: list[str] = []
    for accounting_key, generic_key in GENERIC_ACCOUNTING_FIELD_PAIRS:
        accounting_value = int_value(accounting.get(accounting_key), 0)
        generic_value = int_value(generic_summary.get(generic_key), 0)
        if accounting_value != generic_value:
            errors.append(
                f"generic chain mismatch for {accounting_key}: "
                f"accounting={accounting_value} generic={generic_value}"
            )
    accounting_conflict_count = int_value(accounting.get("rolling_normal_lane_conflict_count"), 0) + int_value(
        accounting.get("rolling_depth3_normal_lane_conflict_count"), 0
    ) + int_value(
        accounting.get("rolling_depth4_normal_lane_conflict_count"), 0
    )
    if accounting_conflict_count != int_value(generic_summary.get("generic_normal_lane_conflict_count"), 0):
        errors.append(
            "generic chain mismatch for normal lane conflict count: "
            f"accounting={accounting_conflict_count} "
            f"generic={generic_summary.get('generic_normal_lane_conflict_count')}"
        )
    if not generic_summary.get("generic_combined_accounting_ok", False):
        errors.append("generic chain combined accounting check failed")
    if not generic_summary.get("generic_target_draft_accounting_ok", False):
        errors.append("generic chain target/draft accounting check failed")
    return errors


def validate_accounting(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    *,
    strict_performance: bool = False,
    check_generic_chain: bool = False,
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
    if check_generic_chain:
        errors.extend(generic_chain_accounting_errors(records, accounting))

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
    continuous_real_tokens = int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
    continuous_real_proposals = int_value(accounting.get("continuous_eager_real_committed_proposal_count"), 0)
    if (
        not bool(accounting.get("continuous_eager_real_commit_enabled", False))
        and int_value(accounting.get("continuous_eager_real_commit_count"), 0) != 0
    ):
        errors.append("continuous eager shadow dry-run must not real-commit proposals")
    if continuous_real_tokens:
        if continuous_real_tokens != int_value(accounting.get("continuous_target_actual_verified_token_increment_sum"), 0):
            errors.append("continuous committed tokens must equal target continuous verified increment sum")
        if continuous_real_tokens != int_value(accounting.get("continuous_target_actual_accepted_token_increment_sum"), 0):
            errors.append("continuous committed tokens must equal target continuous accepted increment sum")
        if continuous_real_tokens != int_value(accounting.get("continuous_draft_actual_verified_token_increment_sum"), 0):
            errors.append("continuous committed tokens must equal draft continuous verified increment sum")
        if continuous_real_tokens != int_value(accounting.get("continuous_draft_actual_accepted_token_increment_sum"), 0):
            errors.append("continuous committed tokens must equal draft continuous accepted increment sum")
    if int_value(accounting.get("continuous_target_actual_rejected_token_increment_sum"), 0) != 0:
        errors.append("continuous target rejected increments must remain zero")
    if int_value(accounting.get("continuous_target_actual_invalidated_token_increment_sum"), 0) != 0:
        errors.append("continuous target invalidated increments must remain zero")
    if int_value(accounting.get("continuous_draft_actual_rejected_token_increment_sum"), 0) != 0:
        errors.append("continuous draft rejected increments must remain zero")
    if int_value(accounting.get("continuous_draft_actual_invalidated_token_increment_sum"), 0) != 0:
        errors.append("continuous draft invalidated increments must remain zero")
    if int_value(accounting.get("continuous_depth2_real_commit_count"), 0) != 0:
        errors.append("continuous depth-2 real commit count must remain zero")
    rolling_depth2_tokens = int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
    rolling_depth2_proposals = int_value(accounting.get("rolling_depth2_real_committed_proposal_count"), 0)
    if rolling_depth2_tokens:
        if not bool(accounting.get("rolling_depth2_commit_enabled", False)):
            errors.append("rolling depth-2 committed tokens require rolling depth-2 commit enabled")
        if rolling_depth2_tokens != int_value(accounting.get("rolling_depth2_target_actual_verified_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal target verified increment sum")
        if rolling_depth2_tokens != int_value(accounting.get("rolling_depth2_target_actual_accepted_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal target accepted increment sum")
        if rolling_depth2_tokens != int_value(accounting.get("rolling_depth2_draft_actual_verified_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal draft verified increment sum")
        if rolling_depth2_tokens != int_value(accounting.get("rolling_depth2_draft_actual_accepted_token_increment_sum"), 0):
            errors.append("rolling depth-2 committed tokens must equal draft accepted increment sum")
    if rolling_depth2_proposals > int_value(accounting.get("rolling_child_ready_shadow_proposal_count"), 0):
        errors.append("rolling depth-2 committed proposal count exceeds ready shadow proposal count")
    if rolling_depth2_tokens > int_value(accounting.get("rolling_child_ready_shadow_token_count"), 0):
        errors.append("rolling depth-2 committed tokens exceed ready shadow tokens")
    if int_value(accounting.get("rolling_depth2_target_actual_rejected_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-2 target rejected increments must remain zero")
    if int_value(accounting.get("rolling_depth2_target_actual_invalidated_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-2 target invalidated increments must remain zero")
    if int_value(accounting.get("rolling_depth2_draft_actual_rejected_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-2 draft rejected increments must remain zero")
    if int_value(accounting.get("rolling_depth2_draft_actual_invalidated_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-2 draft invalidated increments must remain zero")
    rolling_depth3_tokens = int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0)
    rolling_depth3_proposals = int_value(accounting.get("rolling_depth3_real_committed_proposal_count"), 0)
    if int_value(accounting.get("rolling_depth3_real_commit_count"), 0) != 0 and not bool(
        accounting.get("rolling_depth3_commit_enabled", False)
    ):
        errors.append("rolling depth-3 real commit count requires depth-3 commit enabled")
    if rolling_depth3_tokens:
        if not bool(accounting.get("rolling_depth3_commit_enabled", False)):
            errors.append("rolling depth-3 committed tokens require rolling depth-3 commit enabled")
        if rolling_depth3_tokens != int_value(accounting.get("rolling_depth3_target_actual_verified_token_increment_sum"), 0):
            errors.append("rolling depth-3 committed tokens must equal target verified increment sum")
        if rolling_depth3_tokens != int_value(accounting.get("rolling_depth3_target_actual_accepted_token_increment_sum"), 0):
            errors.append("rolling depth-3 committed tokens must equal target accepted increment sum")
        if rolling_depth3_tokens != int_value(accounting.get("rolling_depth3_draft_actual_verified_token_increment_sum"), 0):
            errors.append("rolling depth-3 committed tokens must equal draft verified increment sum")
        if rolling_depth3_tokens != int_value(accounting.get("rolling_depth3_draft_actual_accepted_token_increment_sum"), 0):
            errors.append("rolling depth-3 committed tokens must equal draft accepted increment sum")
    if rolling_depth3_proposals > int_value(accounting.get("rolling_depth3_child_ready_shadow_proposal_count"), 0):
        errors.append("rolling depth-3 committed proposal count exceeds ready shadow proposal count")
    if rolling_depth3_tokens > int_value(accounting.get("rolling_depth3_child_ready_shadow_token_count"), 0):
        errors.append("rolling depth-3 committed tokens exceed ready shadow tokens")
    if int_value(accounting.get("rolling_depth3_target_actual_rejected_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-3 target rejected increments must remain zero")
    if int_value(accounting.get("rolling_depth3_target_actual_invalidated_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-3 target invalidated increments must remain zero")
    if int_value(accounting.get("rolling_depth3_draft_actual_rejected_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-3 draft rejected increments must remain zero")
    if int_value(accounting.get("rolling_depth3_draft_actual_invalidated_token_increment_sum"), 0) != 0:
        errors.append("rolling depth-3 draft invalidated increments must remain zero")
    if int_value(accounting.get("rolling_depth4_real_commit_count"), 0) != 0:
        errors.append("rolling depth-4 real commit count must remain zero")
    if int_value(accounting.get("rolling_depth_gt2_real_commit_count"), 0) != 0 and not bool(
        accounting.get("rolling_depth3_commit_enabled", False)
    ):
        errors.append("rolling depth>2 real commit count must remain zero")
    if int_value(accounting.get("rolling_depth_gt3_real_commit_count"), 0) != 0:
        errors.append("rolling depth>3 real commit count must remain zero")
    if int_value(accounting.get("rolling_depth_gt4_real_commit_count"), 0) != 0:
        errors.append("rolling depth>4 real commit count must remain zero")
    if int_value(accounting.get("rolling_depth3_child_ready_shadow_token_count"), 0) > int_value(
        accounting.get("rolling_depth3_child_candidate_token_count"),
        0,
    ):
        errors.append("rolling depth-3 ready shadow tokens exceed candidate tokens")
    if int_value(accounting.get("rolling_depth3_child_ready_shadow_proposal_count"), 0) > int_value(
        accounting.get("rolling_depth3_child_candidate_proposal_count"),
        0,
    ):
        errors.append("rolling depth-3 ready shadow proposals exceed candidate proposals")
    if int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0) != 0:
        errors.append("rolling depth-4 real committed tokens must remain zero")
    if int_value(accounting.get("rolling_depth4_child_ready_shadow_token_count"), 0) > int_value(
        accounting.get("rolling_depth4_child_candidate_token_count"),
        0,
    ):
        errors.append("rolling depth-4 ready shadow tokens exceed candidate tokens")
    if int_value(accounting.get("rolling_depth4_child_ready_shadow_proposal_count"), 0) > int_value(
        accounting.get("rolling_depth4_child_candidate_proposal_count"),
        0,
    ):
        errors.append("rolling depth-4 ready shadow proposals exceed candidate proposals")
    if int_value(accounting.get("rolling_depth3_child_invalidated_count"), 0) > int_value(
        accounting.get("rolling_depth3_child_candidate_proposal_count"),
        0,
    ):
        errors.append("rolling depth-3 invalidated proposals exceed candidate proposals")
    if int_value(accounting.get("rolling_depth3_normal_lane_conflict_count"), 0) != 0:
        errors.append("rolling depth-3 normal lane conflict count must be zero")
    if int_value(accounting.get("rolling_depth3_max_depth_observed"), 0) > 3:
        errors.append("rolling depth-3 observed max depth exceeds 3")
    if continuous_real_proposals > int_value(accounting.get("continuous_eager_commit_ready_shadow_proposal_count"), 0):
        errors.append("continuous committed proposal count exceeds shadow-ready proposal count")
    if continuous_real_tokens > int_value(accounting.get("continuous_eager_commit_ready_shadow_token_count"), 0):
        errors.append("continuous committed tokens exceed shadow-ready tokens")
    if int_value(accounting.get("continuous_eager_commit_ready_shadow_token_count"), 0) > int_value(
        accounting.get("continuous_eager_candidate_token_count"),
        0,
    ):
        errors.append("continuous shadow ready tokens exceed continuous candidate tokens")
    if int_value(accounting.get("continuous_eager_verified_token_count"), 0) > int_value(
        accounting.get("continuous_eager_candidate_token_count"),
        0,
    ):
        errors.append("continuous verified tokens exceed continuous candidate tokens")
    if int_value(accounting.get("continuous_eager_commit_ready_shadow_token_count"), 0) > int_value(
        accounting.get("continuous_eager_verified_token_count"),
        0,
    ):
        errors.append("continuous shadow ready tokens exceed continuous verified tokens")
    if committed_tokens and int_value(accounting.get("normal_draft_token_slots_suppressed"), 0) < committed_tokens:
        errors.append("normal draft token slots suppressed must cover committed tokens")
    if committed_tokens and int_value(accounting.get("target_normal_verify_token_slots_replaced_by_eager"), 0) < committed_tokens:
        errors.append("target normal verify token slots replaced by eager must cover committed tokens")
    for field in (
        "eager_proposal_transfer_payload_bytes",
        "eager_result_transfer_payload_bytes",
        "eager_proposal_transfer_payload_len_units",
        "eager_result_transfer_payload_len_units",
        "continuous_eager_commit_decision_broadcast_payload_len_units",
        "continuous_eager_result_transfer_payload_len_units",
        "continuous_eager_result_transfer_payload_len_units_before_compact",
        "continuous_zero_result_fast_path_count",
        "continuous_zero_decision_fast_path_count",
        "continuous_eager_sync_apply_zero_steps",
        "continuous_eager_verify_apply_zero_candidate_steps",
        "rolling_child_candidate_token_count",
        "rolling_child_ready_shadow_token_count",
        "rolling_child_invalidated_count",
        "rolling_cascade_discard_count",
        "rolling_parent_full_accept_count",
        "rolling_parent_partial_reject_count",
        "rolling_parent_resolution_pending_count",
        "rolling_same_seq_overlap_count",
        "rolling_normal_lane_conflict_count",
        "rolling_depth2_real_commit_count",
        "rolling_depth_gt1_real_commit_count",
        "rolling_depth2_real_committed_proposal_count",
        "rolling_depth2_real_committed_token_count",
        "rolling_depth2_target_actual_verified_token_increment_sum",
        "rolling_depth2_target_actual_accepted_token_increment_sum",
        "rolling_depth2_target_actual_rejected_token_increment_sum",
        "rolling_depth2_target_actual_invalidated_token_increment_sum",
        "rolling_depth2_draft_actual_verified_token_increment_sum",
        "rolling_depth2_draft_actual_accepted_token_increment_sum",
        "rolling_depth2_draft_actual_rejected_token_increment_sum",
        "rolling_depth2_draft_actual_invalidated_token_increment_sum",
        "rolling_depth3_real_commit_count",
        "rolling_depth3_real_committed_proposal_count",
        "rolling_depth3_real_committed_token_count",
        "rolling_depth3_target_actual_verified_token_increment_sum",
        "rolling_depth3_target_actual_accepted_token_increment_sum",
        "rolling_depth3_target_actual_rejected_token_increment_sum",
        "rolling_depth3_target_actual_invalidated_token_increment_sum",
        "rolling_depth3_draft_actual_verified_token_increment_sum",
        "rolling_depth3_draft_actual_accepted_token_increment_sum",
        "rolling_depth3_draft_actual_rejected_token_increment_sum",
        "rolling_depth3_draft_actual_invalidated_token_increment_sum",
        "rolling_depth4_real_commit_count",
        "rolling_depth_gt2_real_commit_count",
        "rolling_depth3_child_candidate_proposal_count",
        "rolling_depth3_child_candidate_token_count",
        "rolling_depth3_child_ready_shadow_proposal_count",
        "rolling_depth3_child_ready_shadow_token_count",
        "rolling_depth3_child_invalidated_count",
        "rolling_depth3_parent_resolution_pending_count",
        "rolling_depth3_same_seq_overlap_count",
        "rolling_depth3_normal_lane_conflict_count",
        "rolling_depth_gt3_real_commit_count",
        "rolling_depth3_max_depth_observed",
        "rolling_depth3_estimated_future_token_count",
        "rolling_depth4_child_candidate_proposal_count",
        "rolling_depth4_child_candidate_token_count",
        "rolling_depth4_child_ready_shadow_proposal_count",
        "rolling_depth4_child_ready_shadow_token_count",
        "rolling_depth4_child_invalidated_count",
        "rolling_depth4_parent_resolution_pending_count",
        "rolling_depth4_same_seq_overlap_count",
        "rolling_depth4_normal_lane_conflict_count",
        "rolling_depth4_real_committed_token_count",
        "rolling_depth_gt4_real_commit_count",
        "rolling_depth4_max_depth_observed",
        "rolling_depth4_estimated_future_token_count",
        "rolling_depth2_commit_decision_broadcast_payload_len_units",
        "rolling_depth3_commit_decision_broadcast_payload_len_units",
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
        "continuous_eager_candidate_proposal_count",
        "continuous_eager_candidate_token_count",
        "continuous_eager_verified_proposal_count",
        "continuous_eager_verified_token_count",
        "continuous_eager_full_accept_proposal_count",
        "continuous_eager_full_accept_token_count",
        "continuous_eager_commit_ready_shadow_proposal_count",
        "continuous_eager_commit_ready_shadow_token_count",
        "continuous_eager_estimated_committed_token_share_of_output",
        "combined_one_shot_plus_continuous_shadow_token_count",
        "combined_estimated_token_share_of_output",
        "continuous_eager_chain_length_distribution",
        "continuous_eager_drop_reason_counts",
        "continuous_eager_duplicate_count",
        "continuous_eager_frontier_mismatch_count",
        "continuous_true_frontier_mismatch_count",
        "continuous_parent_shadow_not_committed_count",
        "continuous_parent_shadow_not_ready_count",
        "continuous_eager_real_commit_enabled",
        "continuous_eager_real_commit_count",
        "continuous_eager_real_committed_proposal_count",
        "continuous_eager_real_committed_token_count",
        "continuous_real_committed_token_share_of_output",
        "continuous_shadow_ready_but_not_committed_count",
        "continuous_commit_depth1_rate_by_token",
        "continuous_commit_depth1_rate_by_proposal",
        "continuous_target_actual_verified_token_increment_sum",
        "continuous_target_actual_accepted_token_increment_sum",
        "continuous_target_actual_rejected_token_increment_sum",
        "continuous_target_actual_invalidated_token_increment_sum",
        "continuous_draft_actual_verified_token_increment_sum",
        "continuous_draft_actual_accepted_token_increment_sum",
        "continuous_draft_actual_rejected_token_increment_sum",
        "continuous_draft_actual_invalidated_token_increment_sum",
        "continuous_eager_real_commit_skip_reason_counts",
        "continuous_depth2_real_commit_count",
        "rolling_depth2_commit_enabled",
        "rolling_depth2_real_committed_proposal_count",
        "rolling_depth2_real_committed_token_count",
        "rolling_depth2_real_committed_token_share_of_output",
        "rolling_depth2_commit_rate_by_token",
        "rolling_depth2_commit_rate_by_proposal",
        "rolling_depth2_target_actual_verified_token_increment_sum",
        "rolling_depth2_target_actual_accepted_token_increment_sum",
        "rolling_depth2_target_actual_rejected_token_increment_sum",
        "rolling_depth2_target_actual_invalidated_token_increment_sum",
        "rolling_depth2_draft_actual_verified_token_increment_sum",
        "rolling_depth2_draft_actual_accepted_token_increment_sum",
        "rolling_depth2_draft_actual_rejected_token_increment_sum",
        "rolling_depth2_draft_actual_invalidated_token_increment_sum",
        "rolling_depth2_real_commit_skip_reason_counts",
        "rolling_depth3_commit_enabled",
        "rolling_depth3_real_commit_count",
        "rolling_depth3_real_committed_proposal_count",
        "rolling_depth3_real_committed_token_count",
        "rolling_depth3_real_committed_token_share_of_output",
        "rolling_depth3_commit_rate_by_token",
        "rolling_depth3_commit_rate_by_proposal",
        "rolling_depth3_target_actual_verified_token_increment_sum",
        "rolling_depth3_target_actual_accepted_token_increment_sum",
        "rolling_depth3_target_actual_rejected_token_increment_sum",
        "rolling_depth3_target_actual_invalidated_token_increment_sum",
        "rolling_depth3_draft_actual_verified_token_increment_sum",
        "rolling_depth3_draft_actual_accepted_token_increment_sum",
        "rolling_depth3_draft_actual_rejected_token_increment_sum",
        "rolling_depth3_draft_actual_invalidated_token_increment_sum",
        "rolling_depth3_real_commit_skip_reason_counts",
        "rolling_depth4_real_commit_count",
        "rolling_depth_gt2_real_commit_count",
        "rolling_depth2_commit_decision_broadcast_payload_len_units",
        "rolling_depth3_commit_decision_broadcast_payload_len_units",
        "rolling_depth3_shadow_enabled",
        "rolling_depth3_child_candidate_proposal_count",
        "rolling_depth3_child_candidate_token_count",
        "rolling_depth3_child_ready_shadow_proposal_count",
        "rolling_depth3_child_ready_shadow_token_count",
        "rolling_depth3_child_invalidated_count",
        "rolling_depth3_parent_resolution_pending_count",
        "rolling_depth3_same_seq_overlap_count",
        "rolling_depth3_normal_lane_conflict_count",
        "rolling_depth_gt3_real_commit_count",
        "rolling_depth3_drop_reason_counts",
        "rolling_depth3_max_depth_observed",
        "rolling_depth3_estimated_future_token_count",
        "rolling_depth3_estimated_future_token_share_of_output",
        "rolling_depth4_shadow_enabled",
        "rolling_depth4_child_candidate_proposal_count",
        "rolling_depth4_child_candidate_token_count",
        "rolling_depth4_child_ready_shadow_proposal_count",
        "rolling_depth4_child_ready_shadow_token_count",
        "rolling_depth4_child_invalidated_count",
        "rolling_depth4_parent_resolution_pending_count",
        "rolling_depth4_same_seq_overlap_count",
        "rolling_depth4_normal_lane_conflict_count",
        "rolling_depth4_real_committed_token_count",
        "rolling_depth_gt4_real_commit_count",
        "rolling_depth4_drop_reason_counts",
        "rolling_depth4_max_depth_observed",
        "rolling_depth4_estimated_future_token_count",
        "rolling_depth4_estimated_future_token_share_of_output",
        "combined_real_committed_token_count",
        "combined_real_committed_token_share_of_output",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "continuous_eager_result_transfer_protocol",
        "continuous_eager_result_transfer_protocols",
        "continuous_eager_result_transfer_payload_len_units_before_compact",
        "continuous_eager_commit_decision_broadcast_payload_len_units",
        "continuous_eager_result_transfer_payload_len_units",
        "continuous_zero_result_fast_path_count",
        "continuous_zero_decision_fast_path_count",
        "continuous_eager_sync_apply_zero_steps",
        "continuous_eager_verify_apply_zero_candidate_steps",
        "rolling_child_candidate_proposal_count",
        "rolling_child_candidate_token_count",
        "rolling_child_ready_shadow_proposal_count",
        "rolling_child_ready_shadow_token_count",
        "rolling_child_invalidated_count",
        "rolling_cascade_discard_count",
        "rolling_parent_full_accept_count",
        "rolling_parent_partial_reject_count",
        "rolling_parent_resolution_pending_count",
        "rolling_same_seq_overlap_count",
        "rolling_normal_lane_conflict_count",
        "rolling_depth2_real_commit_count",
        "rolling_depth_gt1_real_commit_count",
        "rolling_max_depth_observed",
        "rolling_drop_reason_counts",
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
        "enable_continuous_eager_dry_run": True,
        "enable_continuous_eager_verify_apply_dry_run": True,
        "enable_continuous_eager_commit_depth1_ready_only": True,
        "continuous_eager_candidate_proposal_ids": [900000101],
        "continuous_eager_candidate_seq_ids": [7],
        "continuous_eager_candidate_token_count_by_proposal_id": {"900000101": 4},
        "continuous_eager_chain_depth_by_proposal_id": {"900000101": 1},
        "continuous_eager_parent_proposal_id_by_proposal_id": {"900000101": 101},
        "continuous_eager_parent_source_by_proposal_id": {"900000101": "phase1h6a_one_shot_commit"},
        "continuous_eager_verified_proposal_ids": [900000101],
        "continuous_eager_full_accept_proposal_ids": [900000101],
        "continuous_eager_commit_ready_shadow_proposal_ids": [900000101],
        "continuous_eager_commit_ready_shadow_token_count_by_proposal_id": {"900000101": 4},
        "continuous_eager_real_committed_proposal_ids": [900000101],
        "continuous_eager_real_committed_seq_ids": [7],
        "continuous_eager_real_committed_token_count_by_proposal_id": {"900000101": 4},
        "continuous_eager_tokens_verified": 4,
        "continuous_eager_tokens_accepted": 4,
        "continuous_eager_tokens_rejected": 0,
        "continuous_eager_tokens_invalidated": 0,
        "continuous_eager_real_commit_count": 1,
        "continuous_depth2_real_commit_count": 0,
        "enable_rolling_continuous_depth3_shadow_dry_run": True,
        "enable_rolling_continuous_depth3_commit_ready_only": True,
        "rolling_depth3_shadow_enabled": True,
        "rolling_depth3_commit_enabled": True,
        "rolling_depth3_child_generated_proposal_ids": [900000103],
        "rolling_depth3_child_generated_seq_ids": [7],
        "rolling_depth3_child_ready_shadow_proposal_ids": [900000103],
        "rolling_depth3_child_ready_shadow_seq_ids": [7],
        "rolling_depth3_child_parent_by_proposal_id": {"900000103": 900000102},
        "rolling_depth3_child_depth_by_proposal_id": {"900000103": 3},
        "rolling_depth3_child_token_count_by_proposal_id": {"900000103": 4},
        "rolling_depth3_real_committed_proposal_ids": [900000103],
        "rolling_depth3_real_committed_seq_ids": [7],
        "rolling_depth3_real_committed_token_count_by_proposal_id": {"900000103": 4},
        "rolling_depth3_real_committed_accept_len_by_proposal_id": {"900000103": 4},
        "rolling_depth3_real_commit_action_by_proposal_id": {"900000103": "append_full_accept_real_commit"},
        "rolling_depth3_real_commit_verify_result_by_proposal_id": {"900000103": "full_accept"},
        "rolling_depth3_real_commit_parent_by_proposal_id": {"900000103": 900000102},
        "rolling_depth3_real_commit_depth_by_proposal_id": {"900000103": 3},
        "rolling_depth3_tokens_verified": 4,
        "rolling_depth3_tokens_accepted": 4,
        "rolling_depth3_tokens_rejected": 0,
        "rolling_depth3_tokens_invalidated": 0,
        "rolling_depth3_real_commit_count": 1,
        "rolling_depth3_real_committed_proposal_count": 1,
        "rolling_depth3_real_committed_token_count": 4,
        "rolling_depth4_real_commit_count": 0,
        "rolling_depth_gt3_real_commit_count": 0,
    }
    target = dict(
        base,
        eager_commit_side="target",
        continuous_eager_commit_side="target",
        rolling_depth3_commit_side="target",
        step_id=11,
        plan_id=21,
    )
    draft = dict(
        base,
        eager_commit_side="draft",
        continuous_eager_commit_side="draft",
        rolling_depth3_commit_side="draft",
        step_id=11,
        plan_id=21,
    )
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
    assert summary["continuous_eager_real_committed_token_count"] == 4
    assert summary["continuous_target_actual_verified_token_increment_sum"] == 4
    assert summary["continuous_draft_actual_verified_token_increment_sum"] == 4
    assert summary["rolling_depth3_real_committed_token_count"] == 4
    assert summary["rolling_depth3_target_actual_verified_token_increment_sum"] == 4
    assert summary["rolling_depth3_draft_actual_verified_token_increment_sum"] == 4
    assert summary["combined_real_committed_token_count"] == 12
    assert summary["timing_available"] is False
    assert summary["missing_timing_reason"] == "not_instrumented"

    invalid = deepcopy(records)
    invalid[0]["eager_tokens_verified"] = 0
    errors, _ = validate_accounting(invalid, synthetic_result_payload())
    assert any("target actual eager verified" in error for error in errors), "missed target counter mismatch"

    invalid = deepcopy(records)
    invalid[0]["continuous_eager_tokens_verified"] = 0
    errors, _ = validate_accounting(invalid, synthetic_result_payload())
    assert any("target continuous verified" in error for error in errors), "missed continuous counter mismatch"

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
        check_generic_chain=True,
    )
    assert not errors, f"valid generic-chain performance accounting synthetic failed: {errors}"

    bad_accounting = aggregate_performance_accounting(records, synthetic_result_payload())
    bad_accounting["combined_real_committed_token_count"] += 1
    errors = generic_chain_accounting_errors(records, bad_accounting)
    assert any("combined" in error for error in errors), "missed generic-chain combined mismatch"

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
    parser.add_argument("--check-generic-chain", action="store_true", help="Cross-check committed-depth accounting with generic chain parser.")
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
        check_generic_chain=args.check_generic_chain,
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
