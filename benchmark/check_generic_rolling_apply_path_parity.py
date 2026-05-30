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

from benchmark.bounded_rolling_chain_parser import int_value  # noqa: E402
from benchmark.check_bounded_rolling_readiness_audit import load_trace, synthetic_result_payload  # noqa: E402
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402
from benchmark.check_generic_rolling_runtime_parity import (  # noqa: E402
    _apply_generic_fields,
    _full_depth4_records,
    _partial_records,
    build_summary as build_runtime_summary,
)


GENERIC_APPLY_INT_FIELDS = (
    "generic_rolling_apply_node_count",
    "generic_rolling_apply_full_commit_token_count",
    "generic_rolling_apply_partial_recovered_token_count",
    "generic_rolling_apply_revised_token_count",
    "generic_rolling_apply_output_token_count",
    "generic_rolling_apply_cascade_discard_count",
    "generic_rolling_apply_depth_gt4_count",
    "generic_rolling_apply_normal_lane_conflict_count",
    "generic_rolling_apply_target_draft_mismatch_count",
)


def _bool_any(records: list[dict[str, Any]], *fields: str) -> bool:
    return any(bool(record.get(field, False)) for record in records for field in fields)


def _max_int(records: list[dict[str, Any]], field: str, default: int = 0) -> int:
    values = [int_value(record.get(field), default) for record in records if field in record]
    return max(values) if values else default


def _apply_depths(records: list[dict[str, Any]]) -> list[int]:
    depths: set[int] = set()
    for record in records:
        raw = record.get("generic_rolling_apply_depths")
        if not isinstance(raw, list):
            continue
        for item in raw:
            try:
                depths.add(int(item))
            except Exception:
                continue
    return sorted(depths)


def _all_apply_parity_records_ok(records: list[dict[str, Any]]) -> bool:
    values = [
        bool(record.get("generic_rolling_apply_parity_ok"))
        for record in records
        if bool(record.get("generic_rolling_apply_path_enabled", False))
        or bool(record.get("enable_generic_rolling_apply_path", False))
    ]
    return all(values) if values else True


def build_summary(records: list[dict[str, Any]], result_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result_payload = result_payload or {}
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(result_args, dict):
        result_args = {}
    accounting = aggregate_performance_accounting(records, result_payload)
    runtime = build_runtime_summary(records, result_payload)
    enabled = _bool_any(records, "generic_rolling_apply_path_enabled", "enable_generic_rolling_apply_path") or bool(
        result_args.get("enable_generic_rolling_apply_path", False)
    )

    summary: dict[str, Any] = {
        **runtime,
        "generic_rolling_apply_path_enabled": bool(enabled),
        "generic_rolling_apply_depths": _apply_depths(records),
        "generic_rolling_apply_parity_ok": _all_apply_parity_records_ok(records),
        "partial_prefix_accepted_token_count": int_value(
            accounting.get("partial_prefix_accepted_token_count"), 0
        ),
        "partial_prefix_revised_token_count": int_value(
            accounting.get("partial_prefix_revised_token_count"), 0
        ),
        "partial_prefix_total_recovered_token_count": int_value(
            accounting.get("partial_prefix_total_recovered_token_count"), 0
        ),
        "descendant_committed_after_partial_count": int_value(
            accounting.get("descendant_committed_after_partial_count"), 0
        ),
    }
    for field in GENERIC_APPLY_INT_FIELDS:
        summary[field] = _max_int(records, field)

    summary["expected_generic_rolling_apply_full_commit_token_count"] = int_value(
        runtime.get("generic_rolling_full_commit_token_count"), 0
    )
    summary["expected_generic_rolling_apply_partial_recovered_token_count"] = int_value(
        runtime.get("generic_rolling_partial_recovered_token_count"), 0
    )
    summary["expected_generic_rolling_apply_revised_token_count"] = int_value(
        runtime.get("generic_rolling_revised_token_count"), 0
    )
    summary["expected_generic_rolling_apply_output_token_count"] = int_value(
        runtime.get("generic_rolling_output_token_count"), 0
    )
    summary["expected_generic_rolling_apply_depth_gt4_count"] = int_value(
        summary.get("depth_gt4_real_commit_count"), 0
    )
    summary["expected_generic_rolling_apply_normal_lane_conflict_count"] = int_value(
        runtime.get("generic_rolling_normal_lane_conflict_count"), 0
    )
    summary["expected_generic_rolling_apply_target_draft_mismatch_count"] = int_value(
        runtime.get("generic_rolling_target_draft_mismatch_count"), 0
    )
    summary["combined_real_committed_token_count"] = int_value(
        accounting.get("combined_real_committed_token_count"),
        int_value(runtime.get("combined_real_committed_token_count"), 0),
    )
    return summary


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    summary = build_summary(records, result_payload)
    errors: list[str] = []

    if summary["generic_rolling_apply_path_enabled"]:
        if not summary.get("generic_rolling_runtime_enabled", False):
            errors.append("generic apply path requires generic rolling runtime loop")
        if int_value(summary.get("generic_rolling_max_depth"), 0) != 4:
            errors.append("generic rolling apply path requires max depth 4")
        apply_depths = summary.get("generic_rolling_apply_depths") or []
        if any(int(depth) > 4 or int(depth) < 2 for depth in apply_depths):
            errors.append(f"generic rolling apply depths must stay within 2..4: {apply_depths}")

        comparisons = (
            (
                "generic_rolling_apply_full_commit_token_count",
                "expected_generic_rolling_apply_full_commit_token_count",
            ),
            (
                "generic_rolling_apply_partial_recovered_token_count",
                "expected_generic_rolling_apply_partial_recovered_token_count",
            ),
            (
                "generic_rolling_apply_revised_token_count",
                "expected_generic_rolling_apply_revised_token_count",
            ),
            (
                "generic_rolling_apply_output_token_count",
                "expected_generic_rolling_apply_output_token_count",
            ),
            ("generic_rolling_apply_depth_gt4_count", "expected_generic_rolling_apply_depth_gt4_count"),
            (
                "generic_rolling_apply_normal_lane_conflict_count",
                "expected_generic_rolling_apply_normal_lane_conflict_count",
            ),
            (
                "generic_rolling_apply_target_draft_mismatch_count",
                "expected_generic_rolling_apply_target_draft_mismatch_count",
            ),
        )
        for actual_key, expected_key in comparisons:
            if summary[actual_key] != summary[expected_key]:
                errors.append(
                    f"{actual_key} mismatch: actual={summary[actual_key]} expected={summary[expected_key]}"
                )
        if summary["generic_rolling_apply_output_token_count"] != summary["combined_real_committed_token_count"]:
            errors.append(
                "generic rolling apply output token count must match combined real committed/output progress"
            )
        if int_value(summary.get("generic_rolling_apply_depth_gt4_count"), 0) != 0:
            errors.append("generic rolling apply depth_gt4 count must remain zero")
        if int_value(summary.get("generic_rolling_apply_normal_lane_conflict_count"), 0) != 0:
            errors.append("generic rolling apply normal lane conflict count must remain zero")
        if int_value(summary.get("generic_rolling_apply_target_draft_mismatch_count"), 0) != 0:
            errors.append("generic rolling apply target/draft mismatch count must remain zero")
        if int_value(summary.get("descendant_committed_after_partial_count"), 0) != 0:
            errors.append("descendant committed after partial recovery must remain zero")
        partial_total = int_value(summary.get("partial_prefix_total_recovered_token_count"), 0)
        partial_parts = int_value(summary.get("partial_prefix_accepted_token_count"), 0) + int_value(
            summary.get("partial_prefix_revised_token_count"), 0
        )
        if partial_total and partial_total != partial_parts:
            errors.append("partial recovered total must equal accepted prefix plus revised tokens")
        if not bool(summary.get("generic_rolling_apply_parity_ok", False)):
            errors.append("generic_rolling_apply_parity_ok must be true when apply path is enabled")
    else:
        leaked_fields = [
            field
            for field in GENERIC_APPLY_INT_FIELDS
            if int_value(summary.get(field), 0) != 0
        ]
        if leaked_fields:
            errors.append(f"generic rolling apply fields nonzero while disabled: {leaked_fields}")

    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "generic_rolling_apply_path_enabled",
        "generic_rolling_runtime_enabled",
        "generic_rolling_max_depth",
        "generic_rolling_apply_depths",
        "generic_rolling_apply_node_count",
        "generic_rolling_apply_full_commit_token_count",
        "expected_generic_rolling_apply_full_commit_token_count",
        "generic_rolling_apply_partial_recovered_token_count",
        "expected_generic_rolling_apply_partial_recovered_token_count",
        "generic_rolling_apply_revised_token_count",
        "expected_generic_rolling_apply_revised_token_count",
        "generic_rolling_apply_output_token_count",
        "expected_generic_rolling_apply_output_token_count",
        "combined_real_committed_token_count",
        "generic_rolling_apply_cascade_discard_count",
        "generic_rolling_apply_depth_gt4_count",
        "generic_rolling_apply_normal_lane_conflict_count",
        "generic_rolling_apply_target_draft_mismatch_count",
        "generic_rolling_apply_parity_ok",
        "partial_prefix_accepted_token_count",
        "partial_prefix_revised_token_count",
        "partial_prefix_total_recovered_token_count",
        "descendant_committed_after_partial_count",
    ):
        print(f"{key}={summary.get(key)}")


def _apply_generic_apply_fields(
    records: list[dict[str, Any]],
    *,
    enabled: bool,
    full_tokens: int = 44,
    partial_tokens: int = 0,
    revised_tokens: int = 0,
    output_tokens: int = 44,
    depths: list[int] | None = None,
    depth_gt4: int = 0,
    target_draft_mismatch: int = 0,
    normal_conflict: int = 0,
    parity_ok: bool = True,
) -> None:
    for record in records:
        record["generic_rolling_apply_path_enabled"] = bool(enabled)
        record["enable_generic_rolling_apply_path"] = bool(enabled)
        record["generic_rolling_apply_depths"] = list(depths if depths is not None else [2, 3, 4]) if enabled else []
        record["generic_rolling_apply_node_count"] = 4 if enabled else 0
        record["generic_rolling_apply_full_commit_token_count"] = int(full_tokens) if enabled else 0
        record["generic_rolling_apply_partial_recovered_token_count"] = int(partial_tokens) if enabled else 0
        record["generic_rolling_apply_revised_token_count"] = int(revised_tokens) if enabled else 0
        record["generic_rolling_apply_output_token_count"] = int(output_tokens) if enabled else 0
        record["generic_rolling_apply_cascade_discard_count"] = 0
        record["generic_rolling_apply_depth_gt4_count"] = int(depth_gt4) if enabled else 0
        record["generic_rolling_apply_normal_lane_conflict_count"] = int(normal_conflict) if enabled else 0
        record["generic_rolling_apply_target_draft_mismatch_count"] = (
            int(target_draft_mismatch) if enabled else 0
        )
        record["generic_rolling_apply_parity_ok"] = bool(parity_ok)


def _assert_pass(name: str, records: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic {name} failed: {errors}\nsummary={summary}")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SystemExit(f"synthetic {name} {key} mismatch: expected {value}, got {summary.get(key)}")


def _assert_fail(name: str, records: list[dict[str, Any]], needle: str) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if not errors:
        raise SystemExit(f"synthetic {name} should fail\nsummary={summary}")
    if not any(needle in error for error in errors):
        raise SystemExit(f"synthetic {name} failed for wrong reason: {errors}\nsummary={summary}")


def run_synthetic() -> None:
    disabled = _full_depth4_records()
    _apply_generic_fields(disabled, enabled=False)
    _apply_generic_apply_fields(disabled, enabled=False)
    _assert_pass(
        "disabled legacy apply",
        disabled,
        {"generic_rolling_apply_path_enabled": False, "combined_real_committed_token_count": 44},
    )

    full_accept = _full_depth4_records()
    _apply_generic_fields(full_accept, enabled=True)
    _apply_generic_apply_fields(full_accept, enabled=True)
    _assert_pass(
        "generic apply full accept",
        full_accept,
        {
            "generic_rolling_apply_path_enabled": True,
            "generic_rolling_apply_output_token_count": 44,
            "combined_real_committed_token_count": 44,
        },
    )

    partial = _partial_records()
    _apply_generic_fields(partial, enabled=True, partial_tokens=2, revised_tokens=1, output_tokens=46)
    _apply_generic_apply_fields(partial, enabled=True, partial_tokens=2, revised_tokens=1, output_tokens=46)
    _assert_pass(
        "generic apply partial recovery",
        partial,
        {
            "generic_rolling_apply_partial_recovered_token_count": 2,
            "generic_rolling_apply_revised_token_count": 1,
            "generic_rolling_apply_output_token_count": 46,
        },
    )

    bad_output = deepcopy(partial)
    _apply_generic_apply_fields(bad_output, enabled=True, partial_tokens=2, revised_tokens=1, output_tokens=45)
    _assert_fail("bad apply output mismatch", bad_output, "generic_rolling_apply_output_token_count")

    bad_depth = deepcopy(full_accept)
    _apply_generic_apply_fields(bad_depth, enabled=True, depths=[2, 3, 5], depth_gt4=1, parity_ok=False)
    for record in bad_depth:
        record["rolling_depth_gt4_real_commit_count"] = 1
    _assert_fail("bad apply depth gt4", bad_depth, "depth_gt4")

    bad_max_depth = deepcopy(full_accept)
    _apply_generic_apply_fields(bad_max_depth, enabled=True)
    for record in bad_max_depth:
        record["generic_rolling_max_depth"] = 5
    _assert_fail("bad apply max depth", bad_max_depth, "max depth 4")

    leaked_disabled = _full_depth4_records()
    _apply_generic_fields(leaked_disabled, enabled=False)
    _apply_generic_apply_fields(leaked_disabled, enabled=False)
    for record in leaked_disabled:
        record["generic_rolling_apply_output_token_count"] = 44
    _assert_fail("disabled apply leakage", leaked_disabled, "nonzero while disabled")

    print("Synthetic generic rolling apply-path parity checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8t generic rolling apply-path parity fields.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic()
        return 0

    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result is not None else {}
    errors, summary = validate_records(records, result_payload)
    print_summary(summary)
    if errors:
        print("check_status=fail")
        print(json.dumps({"errors": errors}, indent=2, sort_keys=True))
        return 1
    print("check_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
