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
                if key.startswith("cached_admission_") or key == "cached_prefill_mode"
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

    require(total_arrived <= total_requests, errors, "total_arrived exceeds total_requests")
    require(total_admitted <= total_arrived, errors, "total_admitted exceeds total_arrived")
    require(total_completed <= total_admitted, errors, "total_completed exceeds total_admitted")
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
        summary.get("cached_prefill_mode") is not None,
        errors,
        "cached_prefill_mode is missing from run-level fields",
    )

    admitted: set[str] = set()
    completed: set[str] = set()
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
