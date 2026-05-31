#!/usr/bin/env python3
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
from benchmark.check_eager_performance_accounting import load_json, trace_payload_to_records  # noqa: E402


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "pass"}
    return bool(value)


def float_value(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, (str, bytes)):
        return []
    try:
        return [int(item) for item in value]
    except Exception:
        return []


def load_trace_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        return trace_payload_to_records(payload)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def result_rows(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("traces", "requests", "request_traces", "request_summaries"):
        rows = result_payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def result_summary(result_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = result_payload.get("metrics", {}) if isinstance(result_payload, dict) else {}
    cached = result_payload.get("cached_admission", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(cached, dict):
        cached = {}
    nested = metrics.get("cached_admission", {})
    if not isinstance(nested, dict):
        nested = {}
    args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(args, dict):
        args = {}
    summary = dict(nested)
    summary.update({key: value for key, value in metrics.items() if str(key).startswith("cached_")})
    summary.update(cached)
    if "cached_admission_enabled" not in summary:
        summary["cached_admission_enabled"] = bool(args.get("cached_admission") or args.get("enable_cached_admission"))
    if "cached_admission_max_active" not in summary:
        summary["cached_admission_max_active"] = int_value(
            args.get("max_active_cached_seqs") or args.get("cached_admission_max_active"),
            0,
        )
    if "cached_admission_policy" not in summary:
        summary["cached_admission_policy"] = args.get("cached_admission_policy", "fifo")
    if "execution_mode" not in summary and args.get("execution_mode") is not None:
        summary["execution_mode"] = args.get("execution_mode")
    return summary


def trace_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for record in records:
        for key, value in record.items():
            if (
                str(key).startswith("cached_admission_")
                or str(key).startswith("cached_prefill_")
                or key
                in {
                    "raw_target_normal_verify_seq_ids_before_buffer_filter",
                    "target_normal_verify_seq_ids_after_buffer_filter",
                    "proposal_buffer_hit_seq_ids",
                    "proposal_buffer_miss_seq_ids",
                    "missing_buffered_proposal_unexpected_seq_ids",
                }
            ):
                if key not in summary or value not in (None, [], {}, ""):
                    summary[key] = value
    return summary


def validate(records: list[dict[str, Any]], result_payload: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    rows = result_rows(result_payload)
    summary = result_summary(result_payload)
    summary.update({key: value for key, value in trace_summary(records).items() if key not in summary})
    enabled = bool_value(summary.get("cached_admission_enabled"))

    if not enabled:
        leaked = [
            key for key, value in summary.items()
            if key.startswith("cached_admission_")
            and key not in {
                "cached_admission_enabled",
                "cached_admission_policy",
                "cached_admission_mode",
                "cached_admission_max_active",
                "cached_admission_total_requests",
            }
            and value not in (None, False, 0, 0.0, [], {})
        ]
        if leaked:
            errors.append(f"cached-admission fields nonzero while disabled: {sorted(leaked)}")
        return errors, summary

    max_active = int_value(summary.get("cached_admission_max_active"), 0)
    total_requests = int_value(summary.get("cached_admission_total_requests"), len(rows))
    total_arrived = int_value(summary.get("cached_admission_total_arrived"), 0)
    total_admitted = int_value(summary.get("cached_admission_total_admitted"), 0)
    total_completed = int_value(summary.get("cached_admission_total_completed"), 0)
    peak_active = int_value(summary.get("cached_admission_peak_active"), 0)

    if total_arrived > total_requests:
        errors.append("cached_admission_total_arrived must not exceed total_requests")
    if total_admitted > total_arrived:
        errors.append("cached_admission_total_admitted must not exceed total_arrived")
    if total_completed > total_admitted:
        errors.append("cached_admission_total_completed must not exceed total_admitted")
    if max_active > 0 and peak_active > max_active:
        errors.append("cached admission peak_active must not exceed max_active")
    if not bool_value(summary.get("cached_admission_prefill_compute_skipped")):
        errors.append("cached admission enabled requires cached_admission_prefill_compute_skipped=true")
    if summary.get("cached_admission_decode_only_elapsed_s") is None:
        errors.append("cached admission enabled requires decode-only elapsed time")

    cached_step_records = [
        record for record in records
        if "cached_admission_step" in record and str(record.get("runner_role")) in {"verify", "aggregate", "None"}
    ]
    if not cached_step_records:
        cached_step_records = [record for record in records if "cached_admission_step" in record]

    seen_admitted: set[str] = set()
    seen_completed: set[str] = set()
    for record in cached_step_records:
        for raw_id in record.get("cached_admission_admitted_request_ids", []) or []:
            request_id = str(raw_id)
            if request_id in seen_admitted:
                errors.append(f"request admitted more than once: {request_id}")
            seen_admitted.add(request_id)
        for raw_id in record.get("cached_admission_completed_request_ids", []) or []:
            seen_completed.add(str(raw_id))
        active_count = int_value(record.get("cached_admission_active_count"), 0)
        if max_active > 0 and active_count > max_active:
            errors.append("cached admission trace active_count exceeds max_active")

    for row in rows:
        request_id = str(row.get("request_id"))
        arrival_ts = float_value(row.get("arrival_ts"))
        admission_ts = float_value(row.get("admission_ts"), float_value(row.get("admit_ts")))
        decode_start_ts = float_value(row.get("decode_start_ts"))
        finish_ts = float_value(row.get("finish_ts"))
        queue_wait_ms = float_value(row.get("queue_wait_ms"))
        status = row.get("cached_admission_status")

        if admission_ts is not None and arrival_ts is not None and admission_ts < arrival_ts:
            errors.append(f"request {request_id} admitted before arrival")
        if queue_wait_ms is not None and queue_wait_ms < -1e-6:
            errors.append(f"request {request_id} has negative queue_wait_ms")
        if decode_start_ts is not None and admission_ts is not None and decode_start_ts < admission_ts:
            errors.append(f"request {request_id} decode_start_ts before admission_ts")
        if finish_ts is not None and decode_start_ts is not None and finish_ts < decode_start_ts:
            errors.append(f"request {request_id} finish_ts before decode_start_ts")
        if status == "completed" and admission_ts is None:
            errors.append(f"request {request_id} completed without admission")

    if seen_completed and not seen_completed.issubset(seen_admitted):
        missing = sorted(seen_completed - seen_admitted)
        errors.append(f"completed request was never admitted: {missing}")

    for idx, record in enumerate(records):
        execution_mode = str(record.get("execution_mode") or summary.get("execution_mode") or "")
        if execution_mode != "dual_batch_pearl":
            continue
        target_normal = set(int_list(record.get("target_normal_verify_seq_ids")))
        priming = set(int_list(record.get("cached_admission_draft_priming_seq_ids")))
        newly_admitted = set(int_list(record.get("cached_admission_newly_admitted_seq_ids")))
        filtered = set(int_list(record.get("cached_admission_unprimed_target_filtered_seq_ids")))
        primed = set(int_list(record.get("cached_admission_primed_seq_ids")))
        missing_after_filter = set(
            int_list(record.get("cached_admission_missing_proposal_after_filter_seq_ids"))
        )
        raw_target_before_filter = set(
            int_list(record.get("raw_target_normal_verify_seq_ids_before_buffer_filter"))
        )
        target_after_filter = int_list(record.get("target_normal_verify_seq_ids_after_buffer_filter"))
        target_filtered_missing = set(
            int_list(record.get("cached_admission_target_filtered_missing_proposal_seq_ids"))
        )
        target_buffer_hits = set(int_list(record.get("cached_admission_target_buffer_hit_seq_ids")))
        target_buffer_misses = set(int_list(record.get("cached_admission_target_buffer_miss_seq_ids")))
        missing_buffered_unexpected = set(
            int_list(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        )
        proposal_miss = set(int_list(record.get("proposal_buffer_miss_seq_ids")))
        allowed_missing = set(
            int_list(record.get("missing_buffered_proposal_allowed_by_eager_seq_ids"))
        )
        fallback_pending = set(int_list(record.get("fallback_pending_receive_seq_ids")))
        actual_draft = set(
            int_list(record.get("actual_draft_home_set_for_normal_draft") or record.get("draft_home_set"))
        )
        actual_draft_ordered = int_list(
            record.get("actual_draft_home_set_for_normal_draft") or record.get("draft_home_set")
        )
        sent_proposals = int_list(record.get("dual_proposal_sent_seq_ids"))
        expected_receive = int_list(record.get("dual_proposal_expected_receive_seq_ids"))
        received_proposals = int_list(record.get("dual_proposal_received_seq_ids"))

        if priming & target_normal:
            errors.append(
                f"record[{idx}] cached-admission priming seqs entered target verify: "
                f"{sorted(priming & target_normal)}"
            )
        if missing_after_filter:
            errors.append(
                f"record[{idx}] cached-admission missing proposal after filter: "
                f"{sorted(missing_after_filter)}"
            )
        if missing_buffered_unexpected:
            errors.append(
                f"record[{idx}] unexpected missing buffered proposals after target filter: "
                f"{sorted(missing_buffered_unexpected)}"
            )
        if target_filtered_missing & target_normal:
            errors.append(
                f"record[{idx}] filtered target proposal misses remained in target verify: "
                f"{sorted(target_filtered_missing & target_normal)}"
            )
        if target_after_filter and target_after_filter != int_list(record.get("target_normal_verify_seq_ids")):
            errors.append(
                f"record[{idx}] target verify after buffer filter must match final target verify: "
                f"after={target_after_filter}, final={int_list(record.get('target_normal_verify_seq_ids'))}"
            )
        target_miss_still_selected = (target_buffer_misses & target_normal) - allowed_missing - fallback_pending
        if target_miss_still_selected:
            errors.append(
                f"record[{idx}] target normal verify kept proposal-buffer misses: "
                f"{sorted(target_miss_still_selected)}"
            )
        if target_buffer_hits:
            target_without_buffer_hit = target_normal - target_buffer_hits - allowed_missing - fallback_pending
            if target_without_buffer_hit:
                errors.append(
                    f"record[{idx}] target normal verify must be backed by proposal-buffer hits: "
                    f"target_without_hit={sorted(target_without_buffer_hit)}, "
                    f"raw_target={sorted(raw_target_before_filter)}"
                )
        unexpected_missing = (proposal_miss & target_normal) - allowed_missing - fallback_pending
        if unexpected_missing:
            errors.append(
                f"record[{idx}] target normal verify missing buffered proposals after cached filter: "
                f"{sorted(unexpected_missing)}"
            )
        if filtered and not filtered <= (priming | newly_admitted):
            errors.append(
                f"record[{idx}] filtered cached-admission seqs are not newly admitted or priming: "
                f"{sorted(filtered - (priming | newly_admitted))}"
            )
        if priming and not priming <= actual_draft:
            errors.append(
                f"record[{idx}] cached-admission priming seqs not routed to normal draft: "
                f"{sorted(priming - actual_draft)}"
            )
        if primed and primed & target_normal:
            errors.append(
                f"record[{idx}] primed seqs should not target-verify in the same priming record: "
                f"{sorted(primed & target_normal)}"
            )
        if sent_proposals and sent_proposals != actual_draft_ordered:
            errors.append(
                f"record[{idx}] dual proposal sent seqs must match actual draft seqs: "
                f"sent={sent_proposals}, actual={actual_draft_ordered}"
            )
        if expected_receive and expected_receive != actual_draft_ordered:
            errors.append(
                f"record[{idx}] dual proposal expected receive seqs must match actual draft seqs: "
                f"expected={expected_receive}, actual={actual_draft_ordered}"
            )
        if received_proposals and expected_receive and received_proposals != expected_receive:
            errors.append(
                f"record[{idx}] dual proposal received seqs must match expected receive seqs: "
                f"received={received_proposals}, expected={expected_receive}"
            )

    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    fields = (
        "cached_admission_enabled",
        "cached_admission_policy",
        "cached_admission_mode",
        "cached_admission_max_active",
        "cached_admission_total_requests",
        "cached_admission_total_arrived",
        "cached_admission_total_admitted",
        "cached_admission_total_completed",
        "cached_admission_peak_active",
        "cached_admission_mean_queue_wait_ms",
        "cached_admission_p90_queue_wait_ms",
        "cached_admission_prefill_compute_skipped",
        "cached_admission_decode_only_elapsed_s",
        "cached_admission_newly_admitted_seq_ids",
        "cached_admission_draft_priming_seq_ids",
        "cached_admission_primed_seq_ids",
        "cached_admission_unprimed_target_filtered_seq_ids",
        "cached_admission_missing_proposal_after_filter_seq_ids",
        "cached_admission_filtered_draft_seq_ids",
        "raw_target_normal_verify_seq_ids_before_buffer_filter",
        "cached_admission_target_filtered_missing_proposal_seq_ids",
        "cached_admission_target_buffer_hit_seq_ids",
        "cached_admission_target_buffer_miss_seq_ids",
        "target_normal_verify_seq_ids_after_buffer_filter",
        "proposal_buffer_hit_seq_ids",
        "proposal_buffer_miss_seq_ids",
        "missing_buffered_proposal_unexpected_seq_ids",
        "dual_proposal_sent_seq_ids",
        "dual_proposal_expected_receive_seq_ids",
        "dual_proposal_received_seq_ids",
    )
    for field in fields:
        print(f"{field} = {summary.get(field)}")


def synthetic_payload(
    *,
    enabled: bool = True,
    bad_admit_before_arrival: bool = False,
    negative_wait: bool = False,
    exceed_cap: bool = False,
    duplicate_admit: bool = False,
    completed_without_admit: bool = False,
    prefill_not_skipped: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return [], {"args": {"cached_admission": False}, "cached_admission": {"cached_admission_enabled": False}}

    rows = [
        {
            "request_id": "r0",
            "arrival_ts": 10.0,
            "admission_ts": 10.01,
            "decode_start_ts": 10.01,
            "finish_ts": 10.05,
            "queue_wait_ms": 10.0,
            "cached_admission_status": "completed",
        },
        {
            "request_id": "r1",
            "arrival_ts": 10.02,
            "admission_ts": 10.06,
            "decode_start_ts": 10.06,
            "finish_ts": 10.10,
            "queue_wait_ms": 40.0,
            "cached_admission_status": "completed",
        },
    ]
    if bad_admit_before_arrival:
        rows[0]["admission_ts"] = 9.9
    if negative_wait:
        rows[0]["queue_wait_ms"] = -1.0
    if completed_without_admit:
        rows[1].pop("admission_ts")

    admitted = ["r0", "r1"]
    if duplicate_admit:
        admitted.append("r0")
    records = [
        {
            "cached_admission_enabled": True,
            "cached_admission_step": 0,
            "cached_admission_admitted_request_ids": admitted,
            "cached_admission_completed_request_ids": ["r0", "r1"],
            "cached_admission_active_count": 3 if exceed_cap else 2,
        }
    ]
    result = {
        "args": {"cached_admission": True},
        "cached_admission": {
            "cached_admission_enabled": True,
            "cached_admission_policy": "fifo",
            "cached_admission_mode": "in_memory_kv",
            "cached_admission_max_active": 2,
            "cached_admission_total_requests": 2,
            "cached_admission_total_arrived": 2,
            "cached_admission_total_admitted": 2,
            "cached_admission_total_completed": 2,
            "cached_admission_peak_active": 3 if exceed_cap else 2,
            "cached_admission_mean_queue_wait_ms": 25.0,
            "cached_admission_p50_queue_wait_ms": 25.0,
            "cached_admission_p90_queue_wait_ms": 37.0,
            "cached_admission_p99_queue_wait_ms": 39.7,
            "cached_admission_prefill_compute_skipped": not prefill_not_skipped,
            "cached_admission_decode_only_elapsed_s": 0.1,
        },
        "traces": rows,
    }
    return records, result


def synthetic_dual_payload(
    *,
    execution_mode: str = "dual_batch_pearl",
    newly_admitted_seq_ids: list[int] | None = None,
    priming_seq_ids: list[int] | None = None,
    primed_seq_ids: list[int] | None = None,
    filtered_seq_ids: list[int] | None = None,
    target_normal_verify_seq_ids: list[int] | None = None,
    actual_draft_seq_ids: list[int] | None = None,
    proposal_hit_seq_ids: list[int] | None = None,
    proposal_miss_seq_ids: list[int] | None = None,
    missing_after_filter_seq_ids: list[int] | None = None,
    missing_buffered_unexpected_seq_ids: list[int] | None = None,
    raw_target_before_filter_seq_ids: list[int] | None = None,
    target_filtered_missing_seq_ids: list[int] | None = None,
    target_buffer_hit_seq_ids: list[int] | None = None,
    target_buffer_miss_seq_ids: list[int] | None = None,
    target_after_filter_seq_ids: list[int] | None = None,
    original_draft_seq_ids: list[int] | None = None,
    sent_proposal_seq_ids: list[int] | None = None,
    expected_receive_seq_ids: list[int] | None = None,
    received_proposal_seq_ids: list[int] | None = None,
    filtered_draft_seq_ids: list[int] | None = None,
    fallback_pending_receive_seq_ids: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, result = synthetic_payload()
    result.setdefault("args", {})["execution_mode"] = execution_mode
    records.append(
        {
            "execution_mode": execution_mode,
            "cached_admission_enabled": True,
            "runner_role": "verify",
            "target_normal_verify_seq_ids": target_normal_verify_seq_ids or [],
            "original_draft_home_set": original_draft_seq_ids or actual_draft_seq_ids or [],
            "actual_draft_home_set_for_normal_draft": actual_draft_seq_ids or [],
            "cached_admission_filtered_draft_seq_ids": filtered_draft_seq_ids or [],
            "cached_admission_newly_admitted_seq_ids": newly_admitted_seq_ids or [],
            "cached_admission_draft_priming_seq_ids": priming_seq_ids or [],
            "cached_admission_primed_seq_ids": primed_seq_ids or [],
            "cached_admission_unprimed_target_filtered_seq_ids": filtered_seq_ids or [],
            "cached_admission_missing_proposal_after_filter_seq_ids": (
                missing_after_filter_seq_ids or []
            ),
            "proposal_buffer_hit_seq_ids": proposal_hit_seq_ids or [],
            "proposal_buffer_miss_seq_ids": proposal_miss_seq_ids or [],
            "raw_target_normal_verify_seq_ids_before_buffer_filter": (
                raw_target_before_filter_seq_ids or []
            ),
            "cached_admission_target_filtered_missing_proposal_seq_ids": (
                target_filtered_missing_seq_ids or []
            ),
            "cached_admission_target_buffer_hit_seq_ids": target_buffer_hit_seq_ids or [],
            "cached_admission_target_buffer_miss_seq_ids": target_buffer_miss_seq_ids or [],
            "target_normal_verify_seq_ids_after_buffer_filter": target_after_filter_seq_ids or [],
            "dual_proposal_sent_seq_ids": sent_proposal_seq_ids or [],
            "dual_proposal_expected_receive_seq_ids": expected_receive_seq_ids or [],
            "dual_proposal_received_seq_ids": received_proposal_seq_ids or [],
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [],
            "missing_buffered_proposal_unexpected_seq_ids": (
                missing_buffered_unexpected_seq_ids or []
            ),
            "fallback_pending_receive_seq_ids": fallback_pending_receive_seq_ids or [],
        }
    )
    return records, result


def run_synthetic() -> int:
    cases = [
        ("disabled", synthetic_payload(enabled=False), False),
        ("fifo", synthetic_payload(), False),
        ("bad_admit_before_arrival", synthetic_payload(bad_admit_before_arrival=True), True),
        ("negative_wait", synthetic_payload(negative_wait=True), True),
        ("exceed_cap", synthetic_payload(exceed_cap=True), True),
        ("duplicate_admit", synthetic_payload(duplicate_admit=True), True),
        ("completed_without_admit", synthetic_payload(completed_without_admit=True), True),
        ("prefill_not_skipped", synthetic_payload(prefill_not_skipped=True), True),
        (
            "dual_new_admit_filtered_until_primed",
            synthetic_dual_payload(
                newly_admitted_seq_ids=[6],
                priming_seq_ids=[6],
                primed_seq_ids=[6],
                filtered_seq_ids=[6],
                target_normal_verify_seq_ids=[4],
                actual_draft_seq_ids=[6],
                proposal_hit_seq_ids=[4],
            ),
            False,
        ),
        (
            "dual_new_admit_unprimed_target_verify",
            synthetic_dual_payload(
                newly_admitted_seq_ids=[6],
                priming_seq_ids=[6],
                target_normal_verify_seq_ids=[4, 6],
                actual_draft_seq_ids=[],
                proposal_hit_seq_ids=[4],
                proposal_miss_seq_ids=[6],
                missing_after_filter_seq_ids=[6],
            ),
            True,
        ),
        (
            "dual_target_verify_after_buffered",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[6],
                actual_draft_seq_ids=[],
                proposal_hit_seq_ids=[6],
            ),
            False,
        ),
        (
            "dual_target_filter_missing_proposal_good",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[6],
                actual_draft_seq_ids=[5],
                proposal_hit_seq_ids=[6],
                raw_target_before_filter_seq_ids=[4, 6],
                target_filtered_missing_seq_ids=[4],
                target_buffer_hit_seq_ids=[6],
                target_buffer_miss_seq_ids=[4],
                target_after_filter_seq_ids=[6],
            ),
            False,
        ),
        (
            "dual_target_filter_missing_proposal_bad_final_keeps_miss",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[4, 6],
                actual_draft_seq_ids=[5],
                proposal_hit_seq_ids=[6],
                proposal_miss_seq_ids=[4],
                missing_buffered_unexpected_seq_ids=[4],
                raw_target_before_filter_seq_ids=[4, 6],
                target_filtered_missing_seq_ids=[4],
                target_buffer_hit_seq_ids=[6],
                target_buffer_miss_seq_ids=[4],
                target_after_filter_seq_ids=[4, 6],
            ),
            True,
        ),
        (
            "dual_target_filter_all_missing_good",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[],
                actual_draft_seq_ids=[5],
                raw_target_before_filter_seq_ids=[4],
                target_filtered_missing_seq_ids=[4],
                target_buffer_miss_seq_ids=[4],
                target_after_filter_seq_ids=[],
            ),
            False,
        ),
        (
            "dual_fallback_same_batch_missing_proposal_allowed",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[4],
                actual_draft_seq_ids=[4],
                proposal_miss_seq_ids=[4],
                raw_target_before_filter_seq_ids=[4],
                target_after_filter_seq_ids=[4],
                fallback_pending_receive_seq_ids=[4],
            ),
            False,
        ),
        (
            "non_dual_no_priming_requirement",
            synthetic_dual_payload(
                execution_mode="parallel_pearl",
                newly_admitted_seq_ids=[6],
                priming_seq_ids=[6],
                target_normal_verify_seq_ids=[6],
                proposal_miss_seq_ids=[6],
                missing_after_filter_seq_ids=[6],
            ),
            False,
        ),
        (
            "dual_sender_receiver_actual_draft_alignment_bad",
            synthetic_dual_payload(
                original_draft_seq_ids=[5, 7],
                actual_draft_seq_ids=[7],
                sent_proposal_seq_ids=[5, 7],
                expected_receive_seq_ids=[7],
                received_proposal_seq_ids=[5, 7],
                filtered_draft_seq_ids=[5],
            ),
            True,
        ),
        (
            "dual_sender_receiver_actual_draft_alignment_good",
            synthetic_dual_payload(
                original_draft_seq_ids=[5, 7],
                actual_draft_seq_ids=[7],
                sent_proposal_seq_ids=[7],
                expected_receive_seq_ids=[7],
                received_proposal_seq_ids=[7],
                filtered_draft_seq_ids=[5],
            ),
            False,
        ),
    ]
    for name, (records, result), should_fail in cases:
        errors, _ = validate(records, result)
        failed = bool(errors)
        if failed != should_fail:
            print(f"synthetic case {name} expected fail={should_fail}, got errors={errors}")
            return 1
    print("Synthetic cached admission checks passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H cached admission trace/result fields.")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("result", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        return run_synthetic()
    if not args.trace or not args.result:
        parser.error("TRACE and RESULT are required unless --synthetic is used")

    records = load_trace_records(Path(args.trace))
    result_payload = load_json(Path(args.result))
    errors, summary = validate(records, result_payload)
    print_summary(summary)
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nCached admission check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
