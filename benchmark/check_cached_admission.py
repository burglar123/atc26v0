#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


class CachedAdmissionError(AssertionError):
    pass


SUPPORTED_IN_MEMORY_EXECUTION_MODES = {
    "ar",
    "serialized_pearl",
    "parallel_pearl",
    "dual_batch_pearl",
}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CachedAdmissionError(f"{path} must contain a JSON object")
    return data


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "pass", "enabled"}
    return bool(value)


def float_value(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def is_enabled(payload: dict[str, Any]) -> bool:
    for container_name in ("cached_admission", "metrics", "args"):
        container = payload.get(container_name)
        if isinstance(container, dict) and bool_value(container.get("cached_admission_enabled")):
            return True
        if isinstance(container, dict) and bool_value(container.get("enable_cached_admission")):
            return True
    return False


def cached_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        summary.update(
            {
                key: value
                for key, value in metrics.items()
                if key.startswith("cached_admission_")
                or key.startswith("cached_cache_build_")
                or key.startswith("cached_kv_")
                or key == "cached_prefill_mode"
            }
        )
        nested = metrics.get("cached_admission")
        if isinstance(nested, dict):
            summary.update(nested)
    top = payload.get("cached_admission")
    if isinstance(top, dict):
        summary.update(top)
    return summary


def request_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("traces")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = payload.get("requests")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def trace_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("cached_admission_trace")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    return []


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def running_signature_key(signature: Any) -> tuple[Any, ...] | None:
    if not isinstance(signature, dict):
        return None
    seq_ids = tuple(int_value(value) for value in signature.get("seq_ids", []) or [])
    request_ids = tuple(str(value) for value in signature.get("request_ids", []) or [])
    return (
        int_value(signature.get("count"), len(seq_ids)),
        seq_ids,
        request_ids,
        int_value(signature.get("sum_seq_ids"), sum(seq_ids)),
        int_value(
            signature.get("weighted_sum_seq_ids"),
            sum((idx + 1) * seq_id for idx, seq_id in enumerate(seq_ids)),
        ),
    )


def validate_running_signature_event(event: dict[str, Any], errors: list[str]) -> None:
    if bool_value(event.get("cached_admission_running_divergence_detected")):
        errors.append("cached admission trace reports running-set divergence")
    local_signature = event.get("cached_admission_running_signature")
    local_key = running_signature_key(local_signature)
    if local_key is not None:
        active_count = int_value(event.get("cached_admission_active_count"), local_key[0])
        require(
            active_count == local_key[0],
            errors,
            "cached admission active_count does not match running signature count",
        )
        running_seq_ids = tuple(int_value(value) for value in event.get("cached_admission_running_seq_ids", []) or [])
        if running_seq_ids:
            require(
                tuple(sorted(running_seq_ids)) == tuple(sorted(local_key[1])),
                errors,
                "cached admission running seq ids do not match running signature",
            )
    reports = event.get("cached_admission_running_signature_by_rank")
    if not isinstance(reports, list) or not reports:
        return
    signature_keys: list[tuple[Any, ...]] = []
    for report in reports:
        if not isinstance(report, dict):
            errors.append("cached admission rank running-signature report must be an object")
            continue
        signature = report.get("cached_admission_running_signature")
        key = running_signature_key(signature)
        if key is None:
            errors.append("cached admission rank report missing running signature")
            continue
        signature_keys.append(key)
        running_count = int_value(report.get("running_count"), key[0])
        require(
            running_count == key[0],
            errors,
            "cached admission rank running_count does not match running signature count",
        )
    if signature_keys:
        first = signature_keys[0]
        for key in signature_keys[1:]:
            require(
                key == first,
                errors,
                "cached admission running signatures diverge across ranks",
            )


def request_id_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value}


def rank_request_sets(value: Any) -> dict[str, set[str]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(rank): request_id_set(request_ids)
        for rank, request_ids in value.items()
    }


def validate_completed_sync_event(event: dict[str, Any], errors: list[str]) -> None:
    divergence_count = int_value(event.get("cached_admission_completed_divergence_count"), 0)
    divergence_allowed = bool_value(event.get("cached_admission_completed_divergence_allowed"))
    if divergence_count and not divergence_allowed:
        errors.append("cached admission completed-set divergence is not marked allowed")
    global_completed = request_id_set(event.get("cached_admission_globally_completed_request_ids"))
    pending_completion = request_id_set(event.get("cached_admission_pending_completion_request_ids"))
    divergent_ids = request_id_set(event.get("cached_admission_completed_divergent_request_ids"))
    require(
        not (global_completed & pending_completion),
        errors,
        "cached admission globally completed and pending completion sets overlap",
    )
    if divergence_count:
        require(
            divergent_ids == pending_completion,
            errors,
            "cached admission divergent completion ids must match pending completion ids",
        )
    removed_by_rank = rank_request_sets(event.get("cached_admission_removed_request_ids_by_rank"))
    if not removed_by_rank:
        if bool_value(event.get("cached_admission_completion_sync")):
            errors.append("cached admission completion sync event missing removed ids by rank")
        return
    expected_removed = None
    for rank, removed in sorted(removed_by_rank.items()):
        if expected_removed is None:
            expected_removed = set(removed)
        require(
            removed == expected_removed,
            errors,
            f"cached admission removed request ids differ for rank={rank}",
        )
    if expected_removed is not None:
        require(
            expected_removed == global_completed,
            errors,
            "cached admission removed ids by rank must equal globally completed ids",
        )


def result_execution_mode(payload: dict[str, Any]) -> str | None:
    for container_name in ("args", "metrics"):
        container = payload.get(container_name)
        if isinstance(container, dict) and container.get("execution_mode") is not None:
            return str(container.get("execution_mode"))
    return None


def validate_disabled(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = cached_summary(payload)
    require(
        not bool_value(summary.get("cached_admission_enabled")),
        errors,
        "disabled result has cached_admission_enabled=true in run-level fields",
    )
    for row in request_rows(payload):
        require(
            not bool_value(row.get("cached_admission_enabled")),
            errors,
            f"request_id={row.get('request_id')} has cached_admission_enabled=true while mode is disabled",
        )
    for event in trace_events(payload):
        require(
            not bool_value(event.get("cached_admission_enabled")),
            errors,
            "trace event has cached_admission_enabled=true while mode is disabled",
        )
    return errors


def validate_enabled(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = cached_summary(payload)
    rows = request_rows(payload)
    events = trace_events(payload)

    total_requests = int_value(summary.get("cached_admission_total_requests"), len(rows))
    total_arrived = int_value(summary.get("cached_admission_total_arrived"), len(rows))
    total_admitted = int_value(summary.get("cached_admission_total_admitted"), 0)
    total_completed = int_value(summary.get("cached_admission_total_completed"), 0)
    cap = int_value(summary.get("cached_admission_max_active"), 0)
    peak_active = int_value(summary.get("cached_admission_peak_active"), 0)
    mode = summary.get("cached_prefill_mode")

    require(total_arrived <= total_requests, errors, "total_arrived exceeds total_requests")
    require(total_admitted <= total_arrived, errors, "total_admitted exceeds total_arrived")
    require(total_completed <= total_admitted, errors, "total_completed exceeds total_admitted")
    require(total_admitted == total_arrived, errors, "cached admission did not admit all arrived requests")
    require(total_completed == total_admitted, errors, "cached admission did not complete all admitted requests")
    require(cap > 0, errors, "cached_admission_max_active must be positive when enabled")
    require(peak_active <= cap, errors, "cached_admission_peak_active exceeds cap")
    require(
        bool_value(summary.get("cached_admission_prefill_compute_skipped")),
        errors,
        "cached_admission_prefill_compute_skipped must be true",
    )
    require(
        bool_value(summary.get("cached_admission_enabled")),
        errors,
        "cached_admission_enabled run-level field must be true",
    )
    require(
        mode is not None,
        errors,
        "cached_prefill_mode is missing from run-level fields",
    )
    require(
        mode in {"metadata_only", "in_memory_kv"},
        errors,
        f"unsupported cached_prefill_mode={mode!r}",
    )

    if mode == "in_memory_kv":
        execution_mode = result_execution_mode(payload)
        if execution_mode is not None:
            require(
                execution_mode in SUPPORTED_IN_MEMORY_EXECUTION_MODES,
                errors,
                (
                    "cached-prefill-mode=in_memory_kv is not yet supported "
                    f"for execution_mode={execution_mode}"
                ),
            )
        require(
            summary.get("cached_cache_build_elapsed_s") is not None,
            errors,
            "cached_cache_build_elapsed_s must exist for in_memory_kv",
        )
        cached_kv_num_requests = int_value(summary.get("cached_kv_num_requests"), -1)
        require(
            cached_kv_num_requests == total_requests
            or summary.get("cached_kv_num_requests_explanation") is not None,
            errors,
            "cached_kv_num_requests must equal total_requests or explain filtered count",
        )

    admitted: set[str] = set()
    completed: set[str] = set()
    materialized: set[str] = set()
    for row in rows:
        request_id = str(row.get("request_id"))
        arrival_ts = float_value(row.get("arrival_ts"))
        admission_ts = float_value(first_present(row, "admission_ts", "admit_ts"))
        decode_start_ts = float_value(row.get("decode_start_ts"))
        finish_ts = float_value(first_present(row, "finish_ts", "finished_ts", "end_ts", "end_time"))
        queue_wait_ms = float_value(row.get("queue_wait_ms"))
        status = row.get("cached_admission_status")
        is_completed = status == "completed" or finish_ts is not None

        require(admission_ts is not None, errors, f"request_id={request_id} missing admission_ts")
        if admission_ts is not None:
            require(request_id not in admitted, errors, f"duplicate admission for request_id={request_id}")
            admitted.add(request_id)
        if is_completed:
            require(admission_ts is not None, errors, f"completed request_id={request_id} has no admission")
            completed.add(request_id)
        if mode == "in_memory_kv":
            materialized_value = bool_value(row.get("cached_kv_materialized"))
            admitted_or_completed = admission_ts is not None or is_completed
            if admitted_or_completed:
                require(
                    bool_value(row.get("cached_kv_ready")),
                    errors,
                    f"request_id={request_id} cached_kv_ready must be true for in_memory_kv",
                )
                require(
                    materialized_value,
                    errors,
                    f"request_id={request_id} cached_kv_materialized must be true for in_memory_kv",
                )
            if materialized_value:
                require(
                    admission_ts is not None,
                    errors,
                    f"request_id={request_id} materialized without admission",
                )
                require(
                    request_id not in materialized,
                    errors,
                    f"duplicate materialization for request_id={request_id}",
                )
                materialized.add(request_id)
        if arrival_ts is not None and admission_ts is not None:
            require(
                admission_ts + 1e-9 >= arrival_ts,
                errors,
                f"request_id={request_id} admitted before arrival",
            )
        require(
            queue_wait_ms is not None and queue_wait_ms >= 0,
            errors,
            f"request_id={request_id} has negative or missing queue_wait_ms",
        )
        if admission_ts is not None and decode_start_ts is not None:
            require(
                decode_start_ts + 1e-9 >= admission_ts,
                errors,
                f"request_id={request_id} decode_start_ts is before admission_ts",
            )
        if finish_ts is not None and decode_start_ts is not None:
            require(
                finish_ts + 1e-9 >= decode_start_ts,
                errors,
                f"request_id={request_id} finish_ts is before decode_start_ts",
            )
        require(
            bool_value(row.get("cached_prefill_skipped")),
            errors,
            f"request_id={request_id} cached_prefill_skipped must be true",
        )
        require(
            row.get("cached_prefill_mode") is not None,
            errors,
            f"request_id={request_id} cached_prefill_mode is missing",
        )

    event_admitted: set[str] = set()
    event_completed: set[str] = set()
    for event in events:
        active_count = int_value(event.get("cached_admission_active_count"), 0)
        require(active_count <= cap, errors, "trace active_count exceeds cap")
        validate_running_signature_event(event, errors)
        validate_completed_sync_event(event, errors)
        for request_id in event.get("cached_admission_admitted_request_ids", []) or []:
            request_id = str(request_id)
            require(request_id not in event_admitted, errors, f"duplicate event admission for {request_id}")
            event_admitted.add(request_id)
        for request_id in event.get("cached_admission_completed_request_ids", []) or []:
            event_completed.add(str(request_id))
    for request_id in event_completed:
        require(
            request_id in admitted or request_id in event_admitted,
            errors,
            f"event completed request_id={request_id} without admission",
        )

    return errors


def validate_payload(payload: dict[str, Any]) -> None:
    errors = validate_enabled(payload) if is_enabled(payload) else validate_disabled(payload)
    if errors:
        raise CachedAdmissionError("\n".join(f"- {error}" for error in errors))


def valid_enabled_payload() -> dict[str, Any]:
    rows = [
        {
            "request_id": "r0",
            "arrival_ts": 100.0,
            "admission_ts": 100.0,
            "decode_start_ts": 100.0,
            "finish_ts": 101.0,
            "queue_wait_ms": 0.0,
            "cached_admission_status": "completed",
            "cached_admission_enabled": True,
            "cached_prefill_skipped": True,
            "cached_prefill_mode": "metadata_only",
        },
        {
            "request_id": "r1",
            "arrival_ts": 100.5,
            "admission_ts": 101.0,
            "decode_start_ts": 101.0,
            "finish_ts": 102.0,
            "queue_wait_ms": 500.0,
            "cached_admission_status": "completed",
            "cached_admission_enabled": True,
            "cached_prefill_skipped": True,
            "cached_prefill_mode": "metadata_only",
        },
    ]
    return {
        "args": {"enable_cached_admission": True},
        "cached_admission": {
            "cached_admission_enabled": True,
            "cached_prefill_mode": "metadata_only",
            "cached_admission_policy": "fifo",
            "cached_admission_max_active": 1,
            "cached_admission_total_requests": 2,
            "cached_admission_total_arrived": 2,
            "cached_admission_total_admitted": 2,
            "cached_admission_total_completed": 2,
            "cached_admission_peak_active": 1,
            "cached_admission_mean_queue_wait_ms": 250.0,
            "cached_admission_p50_queue_wait_ms": 250.0,
            "cached_admission_p90_queue_wait_ms": 450.0,
            "cached_admission_p99_queue_wait_ms": 495.0,
            "cached_admission_prefill_compute_skipped": True,
            "cached_admission_decode_only_elapsed_s": 2.0,
        },
        "cached_admission_trace": [
            {
                "cached_admission_enabled": True,
                "cached_admission_step": 0,
                "cached_admission_arrived_request_ids": ["r0"],
                "cached_admission_admitted_request_ids": ["r0"],
                "cached_admission_active_request_ids": ["r0"],
                "cached_admission_completed_request_ids": [],
                "cached_admission_pending_count": 1,
                "cached_admission_active_count": 1,
                "cached_admission_queue_wait_ms_by_request": {"r0": 0.0},
            },
            {
                "cached_admission_enabled": True,
                "cached_admission_step": 1,
                "cached_admission_arrived_request_ids": ["r1"],
                "cached_admission_admitted_request_ids": ["r1"],
                "cached_admission_active_request_ids": ["r1"],
                "cached_admission_completed_request_ids": ["r0"],
                "cached_admission_pending_count": 0,
                "cached_admission_active_count": 1,
                "cached_admission_queue_wait_ms_by_request": {"r0": 0.0, "r1": 500.0},
            },
        ],
        "traces": rows,
    }


def cached_running_signature(seq_ids: list[int], request_ids: list[str]) -> dict[str, Any]:
    sorted_seq_ids = sorted(int(seq_id) for seq_id in seq_ids)
    sorted_request_ids = sorted(str(request_id) for request_id in request_ids)
    return {
        "count": len(sorted_seq_ids),
        "seq_ids": sorted_seq_ids,
        "request_ids": sorted_request_ids,
        "sum_seq_ids": sum(sorted_seq_ids),
        "weighted_sum_seq_ids": sum((idx + 1) * seq_id for idx, seq_id in enumerate(sorted_seq_ids)),
    }


def cached_running_rank_report(
    rank: int,
    seq_ids: list[int],
    request_ids: list[str],
    *,
    materialized_count: int = 1,
    pending_count: int = 1,
    completed_request_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "global_rank": int(rank),
        "runner_role": "draft" if rank == 0 else "verify",
        "group_name": "synthetic",
        "materialized_count": int(materialized_count),
        "pending_count": int(pending_count),
        "running_count": len(seq_ids),
        "running_request_ids": sorted(str(request_id) for request_id in request_ids),
        "running_seq_ids": sorted(int(seq_id) for seq_id in seq_ids),
        "admitted_this_step_request_ids": [],
        "cached_step": 0,
        "plan_id": 1,
        "dual_step_id": None,
        "enable_unified_generic_rolling_runtime": True,
        "cached_admission_running_signature": cached_running_signature(seq_ids, request_ids),
        "cached_admission_completed_request_ids": [
            str(request_id) for request_id in (completed_request_ids or [])
        ],
    }


def completion_sync_event(
    *,
    local_completed_by_rank: dict[str, list[str]],
    globally_completed: list[str],
    pending_completion: list[str],
    authority: str = "intersection",
    removed_by_rank: dict[str, list[str]] | None = None,
    divergence_allowed: bool = True,
) -> dict[str, Any]:
    removed_by_rank = removed_by_rank if removed_by_rank is not None else {
        str(rank): list(globally_completed) for rank in range(4)
    }
    divergent_ids = sorted(str(request_id) for request_id in pending_completion)
    return {
        "trace_record_type": "cached_admission_step",
        "cached_admission_completion_sync": True,
        "cached_admission_enabled": True,
        "cached_prefill_mode": "in_memory_kv",
        "cached_admission_step": 99,
        "cached_admission_arrived_request_ids": [],
        "cached_admission_admitted_request_ids": [],
        "cached_admission_active_request_ids": [],
        "cached_admission_completed_request_ids": list(globally_completed),
        "cached_admission_pending_count": 0,
        "cached_admission_active_count": 0,
        "cached_admission_running_request_ids": [],
        "cached_admission_running_seq_ids": [],
        "cached_admission_running_signature": cached_running_signature([], []),
        "cached_admission_running_signature_by_rank": [
            cached_running_rank_report(rank, [], [], completed_request_ids=globally_completed)
            for rank in range(4)
        ],
        "cached_admission_running_divergence_detected": False,
        "cached_admission_unified_generic_enabled": True,
        "cached_admission_local_completed_request_ids_by_rank": {
            str(rank): [str(request_id) for request_id in request_ids]
            for rank, request_ids in local_completed_by_rank.items()
        },
        "cached_admission_globally_completed_request_ids": [
            str(request_id) for request_id in globally_completed
        ],
        "cached_admission_pending_completion_request_ids": [
            str(request_id) for request_id in pending_completion
        ],
        "cached_admission_completed_authority": authority,
        "cached_admission_completed_divergence_count": len(divergent_ids),
        "cached_admission_completed_divergent_request_ids": divergent_ids,
        "cached_admission_completed_divergence_allowed": bool(divergence_allowed),
        "cached_admission_removed_request_ids": [
            str(request_id) for request_id in globally_completed
        ],
        "cached_admission_removed_request_ids_by_rank": {
            str(rank): [str(request_id) for request_id in request_ids]
            for rank, request_ids in removed_by_rank.items()
        },
        "cached_admission_completed_reason_by_request": {
            str(request_id): "target_verified_finish" for request_id in globally_completed
        },
        "cached_admission_queue_wait_ms_by_request": {},
    }


def attach_matching_running_reports(payload: dict[str, Any]) -> dict[str, Any]:
    for event in payload.get("cached_admission_trace", []) or []:
        active_request_ids = [str(request_id) for request_id in event.get("cached_admission_active_request_ids", [])]
        seq_ids = list(range(len(active_request_ids)))
        signature = cached_running_signature(seq_ids, active_request_ids)
        event["cached_admission_running_request_ids"] = list(active_request_ids)
        event["cached_admission_running_seq_ids"] = list(seq_ids)
        event["cached_admission_running_signature"] = signature
        event["cached_admission_running_signature_by_rank"] = [
            cached_running_rank_report(rank, seq_ids, active_request_ids)
            for rank in range(4)
        ]
        event["cached_admission_running_divergence_detected"] = False
        event["cached_admission_unified_generic_enabled"] = True
    return payload


def valid_in_memory_payload() -> dict[str, Any]:
    payload = copy.deepcopy(valid_enabled_payload())
    payload["args"]["execution_mode"] = "parallel_pearl"
    payload["cached_admission"].update(
        {
            "cached_prefill_mode": "in_memory_kv",
            "cached_cache_build_elapsed_s": 0.25,
            "cached_cache_build_batch_size": 2,
            "cached_kv_total_cpu_bytes": 4096,
            "cached_kv_num_requests": 2,
            "cached_kv_avg_blocks_per_request": 1.0,
            "cached_kv_max_blocks_per_request": 1,
        }
    )
    for row in payload["traces"]:
        row["cached_prefill_mode"] = "in_memory_kv"
        row["cached_kv_ready"] = True
        row["cached_kv_materialized"] = True
    for event in payload["cached_admission_trace"]:
        event["cached_prefill_mode"] = "in_memory_kv"
    return attach_matching_running_reports(payload)


def disabled_payload() -> dict[str, Any]:
    return {
        "args": {"enable_cached_admission": False},
        "cached_admission": {"cached_admission_enabled": False},
        "traces": [{"request_id": "r0", "arrival_ts": 100.0, "finish_ts": 101.0}],
    }


def run_synthetic() -> int:
    cases: list[tuple[str, dict[str, Any], bool]] = [
        ("disabled mode pass", disabled_payload(), True),
        ("FIFO admission pass", valid_enabled_payload(), True),
        ("in_memory_kv summary schema pass", valid_in_memory_payload(), True),
    ]

    payload = valid_enabled_payload()
    payload["traces"][0]["admission_ts"] = 99.0
    cases.append(("admission before arrival fail", payload, False))

    payload = valid_enabled_payload()
    payload["traces"][0]["queue_wait_ms"] = -1.0
    cases.append(("negative queue wait fail", payload, False))

    payload = valid_enabled_payload()
    payload["cached_admission"]["cached_admission_peak_active"] = 2
    cases.append(("active count exceeds cap fail", payload, False))

    payload = valid_enabled_payload()
    payload["traces"][1]["request_id"] = "r0"
    cases.append(("duplicate admission fail", payload, False))

    payload = valid_enabled_payload()
    payload["traces"][0].pop("admission_ts")
    cases.append(("completed without admission fail", payload, False))

    payload = valid_enabled_payload()
    payload["traces"][0]["cached_prefill_skipped"] = False
    cases.append(("cached enabled but prefill skipped false fail", payload, False))

    payload = valid_in_memory_payload()
    payload["traces"][0]["cached_kv_ready"] = False
    cases.append(("in_memory_kv missing cached_kv_ready fail", payload, False))

    payload = valid_in_memory_payload()
    payload["traces"][0]["cached_kv_materialized"] = False
    cases.append(("in_memory_kv missing cached_kv_materialized fail", payload, False))

    payload = valid_in_memory_payload()
    duplicate = copy.deepcopy(payload["traces"][0])
    payload["traces"].append(duplicate)
    payload["cached_admission"]["cached_admission_total_requests"] = 3
    payload["cached_admission"]["cached_kv_num_requests"] = 3
    cases.append(("duplicate materialization fail", payload, False))

    payload = valid_in_memory_payload()
    payload["traces"][0].pop("admission_ts")
    payload["traces"][0].pop("admit_ts", None)
    cases.append(("materialized without admission fail", payload, False))

    payload = valid_in_memory_payload()
    payload["args"]["execution_mode"] = "unsupported_mode"
    cases.append(("unsupported in_memory_kv execution_mode fail", payload, False))

    payload = valid_in_memory_payload()
    event = payload["cached_admission_trace"][0]
    event["cached_admission_running_signature_by_rank"] = [
        cached_running_rank_report(0, [1], ["r0"], materialized_count=4, pending_count=28),
        cached_running_rank_report(1, [1, 2], ["r0", "r1"], materialized_count=4, pending_count=28),
    ]
    cases.append(("running count divergence fail", payload, False))

    payload = valid_in_memory_payload()
    event = payload["cached_admission_trace"][0]
    event["cached_admission_running_signature_by_rank"] = [
        cached_running_rank_report(0, [1], ["r0"], materialized_count=4, pending_count=28),
        cached_running_rank_report(1, [2], ["r1"], materialized_count=4, pending_count=28),
    ]
    cases.append(("running seq id divergence fail", payload, False))

    payload = valid_in_memory_payload()
    cases.append(("running signature all ranks pass", payload, True))

    payload = valid_in_memory_payload()
    payload["cached_admission_trace"].append(
        completion_sync_event(
            local_completed_by_rank={"0": ["r0"], "1": [], "2": [], "3": []},
            globally_completed=[],
            pending_completion=["r0"],
            authority="intersection",
            removed_by_rank={"0": [], "1": [], "2": [], "3": []},
        )
    )
    cases.append(("local completed draft-only pending pass", payload, True))

    payload = valid_in_memory_payload()
    payload["cached_admission_trace"].append(
        completion_sync_event(
            local_completed_by_rank={"0": ["r0"], "1": ["r0"], "2": ["r0"], "3": ["r0"]},
            globally_completed=["r0"],
            pending_completion=[],
            authority="intersection",
        )
    )
    cases.append(("local completed all ranks global removal pass", payload, True))

    payload = valid_in_memory_payload()
    payload["cached_admission_trace"].append(
        completion_sync_event(
            local_completed_by_rank={"0": [], "1": ["r0"], "2": [], "3": []},
            globally_completed=["r0"],
            pending_completion=[],
            authority="target_master",
        )
    )
    cases.append(("target master authority removal pass", payload, True))

    payload = valid_in_memory_payload()
    payload["cached_admission_trace"].append(
        completion_sync_event(
            local_completed_by_rank={"0": ["r0"], "1": ["r1"], "2": [], "3": []},
            globally_completed=["r0"],
            pending_completion=[],
            authority="target_master",
            removed_by_rank={"0": ["r0"], "1": ["r1"], "2": ["r0"], "3": ["r0"]},
        )
    )
    cases.append(("different removed request ids fail", payload, False))

    failures = 0
    for name, payload, expect_ok in cases:
        try:
            validate_payload(copy.deepcopy(payload))
            ok = True
            detail = ""
        except CachedAdmissionError as exc:
            ok = False
            detail = str(exc).splitlines()[0] if str(exc) else ""
        status = "PASS" if ok == expect_ok else "FAIL"
        print(f"{status}: {name}")
        if status == "FAIL":
            failures += 1
            if detail:
                print(f"  detail: {detail}")
    if failures:
        return 1
    print("synthetic_cached_admission=pass")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cached-admission result metadata.")
    parser.add_argument("paths", nargs="*", help="Result JSON path, or engine trace plus result JSON.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Accepted for runner compatibility.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic:
        return run_synthetic()
    if not args.paths:
        raise SystemExit("result JSON path is required unless --synthetic is set")
    result_path = Path(args.paths[-1])
    payload = load_json(result_path)
    try:
        validate_payload(payload)
    except CachedAdmissionError as exc:
        print("cached_admission_check=fail")
        print(str(exc))
        return 1
    print("cached_admission_check=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
