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

from benchmark.bounded_rolling_chain_parser import int_value, parse_legacy_rolling_chain, summarize_registry  # noqa: E402
from benchmark.check_bounded_rolling_readiness_audit import load_trace, synthetic_result_payload  # noqa: E402
from benchmark.check_eager_partial_prefix_recovery import _apply_partial_recovery  # noqa: E402
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402
from benchmark.check_generic_bounded_rolling_chain import add_depth4_shadow, synthetic_records  # noqa: E402
from benchmark.check_rolling_continuous_depth4_commit_ready_only import add_depth4_commit  # noqa: E402


FULL_COMMIT_TOKEN_FIELDS = (
    "eager_committed_token_count",
    "continuous_eager_real_committed_token_count",
    "rolling_depth2_real_committed_token_count",
    "rolling_depth3_real_committed_token_count",
    "rolling_depth4_real_committed_token_count",
)

GENERIC_RUNTIME_INT_FIELDS = (
    "generic_rolling_max_depth",
    "generic_rolling_node_count",
    "generic_rolling_max_observed_depth",
    "generic_rolling_max_real_committed_depth",
    "generic_rolling_full_commit_token_count",
    "generic_rolling_partial_recovered_token_count",
    "generic_rolling_revised_token_count",
    "generic_rolling_output_token_count",
    "generic_rolling_descendant_cascade_discard_count",
    "generic_rolling_normal_lane_conflict_count",
    "generic_rolling_target_draft_mismatch_count",
)


def _bool_any(records: list[dict[str, Any]], *fields: str) -> bool:
    return any(bool(record.get(field, False)) for record in records for field in fields)


def _max_int(records: list[dict[str, Any]], field: str, default: int = 0) -> int:
    values = [int_value(record.get(field), default) for record in records if field in record]
    return max(values) if values else default


def _all_generic_parity_records_ok(records: list[dict[str, Any]]) -> bool:
    values = [
        bool(record.get("generic_rolling_parity_ok"))
        for record in records
        if bool(record.get("generic_rolling_runtime_enabled", False))
        or bool(record.get("enable_generic_rolling_runtime_loop", False))
    ]
    return all(values) if values else True


def _runtime_generic_summary(records: list[dict[str, Any]], result_payload: dict[str, Any]) -> dict[str, Any]:
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(result_args, dict):
        result_args = {}
    enabled = _bool_any(records, "generic_rolling_runtime_enabled", "enable_generic_rolling_runtime_loop") or bool(
        result_args.get("enable_generic_rolling_runtime_loop", False)
    )
    summary: dict[str, Any] = {"generic_rolling_runtime_enabled": bool(enabled)}
    for field in GENERIC_RUNTIME_INT_FIELDS:
        summary[field] = _max_int(records, field)
    summary["generic_rolling_parity_ok"] = _all_generic_parity_records_ok(records)
    return summary


def build_summary(records: list[dict[str, Any]], result_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result_payload = result_payload or {}
    accounting = aggregate_performance_accounting(records, result_payload)
    registry = parse_legacy_rolling_chain(records)
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if isinstance(result_args, dict):
        registry.max_configured_depth = max(
            registry.max_configured_depth,
            int_value(result_args.get("max_rolling_continuous_depth"), 0),
        )
    generic_chain = summarize_registry(registry, accounting=accounting)
    runtime = _runtime_generic_summary(records, result_payload)

    full_commit_token_count = sum(int_value(accounting.get(field), 0) for field in FULL_COMMIT_TOKEN_FIELDS)
    partial_total = int_value(accounting.get("partial_prefix_total_recovered_token_count"), 0)
    partial_revised = int_value(accounting.get("partial_prefix_revised_token_count"), 0)
    expected_output = int_value(accounting.get("combined_real_committed_token_count"), 0)
    target_draft_mismatch_count = (
        int_value(generic_chain.get("generic_target_draft_length_mismatch_count"), 0)
        + int_value(generic_chain.get("generic_target_draft_token_mismatch_count"), 0)
        + int_value(accounting.get("partial_recovery_target_draft_length_mismatch_count"), 0)
        + int_value(accounting.get("partial_recovery_target_draft_token_mismatch_count"), 0)
    )
    summary = {
        **runtime,
        "generic_full_continuous_enabled": _bool_any(
            records,
            "generic_full_continuous_enabled",
            "enable_full_continuous_eager",
        )
        or bool(result_args.get("enable_full_continuous_eager", False)),
        "expected_generic_rolling_max_observed_depth": int_value(generic_chain.get("generic_max_observed_depth"), 0),
        "expected_generic_rolling_max_real_committed_depth": int_value(
            generic_chain.get("generic_max_real_committed_depth"), 0
        ),
        "expected_generic_rolling_full_commit_token_count": full_commit_token_count,
        "expected_generic_rolling_partial_recovered_token_count": partial_total,
        "expected_generic_rolling_revised_token_count": partial_revised,
        "expected_generic_rolling_output_token_count": expected_output,
        "expected_generic_rolling_normal_lane_conflict_count": int_value(
            generic_chain.get("generic_normal_lane_conflict_count"), 0
        ),
        "expected_generic_rolling_target_draft_mismatch_count": target_draft_mismatch_count,
        "combined_real_committed_token_count": expected_output,
        "combined_actual_verified_token_increment_sum": int_value(
            accounting.get("combined_actual_verified_token_increment_sum"), 0
        ),
        "combined_actual_accepted_token_increment_sum": int_value(
            accounting.get("combined_actual_accepted_token_increment_sum"), 0
        ),
        "combined_actual_revised_token_increment_sum": int_value(
            accounting.get("combined_actual_revised_token_increment_sum"), 0
        ),
        "combined_actual_output_token_increment_sum": int_value(
            accounting.get("combined_actual_output_token_increment_sum"), 0
        ),
        "partial_prefix_recovery_enabled": bool(accounting.get("partial_prefix_recovery_enabled", False)),
        "partial_prefix_recovery_success_count": int_value(accounting.get("partial_prefix_recovery_success_count"), 0),
        "partial_prefix_total_recovered_token_count": partial_total,
        "partial_prefix_revised_token_count": partial_revised,
        "depth_gt4_real_commit_count": int_value(generic_chain.get("generic_depth_gt4_real_commit_count"), 0),
    }
    max_depth = int_value(summary.get("generic_rolling_max_depth"), 0)
    max_depth_ok = bool(
        max_depth == 4
        or (bool(summary.get("generic_full_continuous_enabled", False)) and 4 <= max_depth <= 100)
    )
    summary["expected_generic_rolling_parity_ok"] = bool(
        max_depth_ok
        and summary["expected_generic_rolling_max_observed_depth"] <= max_depth
        and summary["expected_generic_rolling_max_real_committed_depth"] <= max_depth
        and summary["depth_gt4_real_commit_count"] == 0
        and summary["expected_generic_rolling_normal_lane_conflict_count"] == 0
        and summary["expected_generic_rolling_target_draft_mismatch_count"] == 0
    )
    return summary


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    summary = build_summary(records, result_payload)
    errors: list[str] = []

    if summary["depth_gt4_real_commit_count"] != 0:
        errors.append("depth_gt4 real commit count must remain zero")
    if summary["generic_rolling_runtime_enabled"]:
        if summary.get("generic_full_continuous_enabled", False):
            if not (4 <= summary["generic_rolling_max_depth"] <= 100):
                errors.append("generic rolling full continuous mode requires max depth in 4..100")
        elif summary["generic_rolling_max_depth"] != 4:
            errors.append("generic rolling runtime parity mode requires max depth 4")
        comparisons = (
            ("generic_rolling_max_observed_depth", "expected_generic_rolling_max_observed_depth"),
            ("generic_rolling_max_real_committed_depth", "expected_generic_rolling_max_real_committed_depth"),
            ("generic_rolling_full_commit_token_count", "expected_generic_rolling_full_commit_token_count"),
            ("generic_rolling_partial_recovered_token_count", "expected_generic_rolling_partial_recovered_token_count"),
            ("generic_rolling_revised_token_count", "expected_generic_rolling_revised_token_count"),
            ("generic_rolling_output_token_count", "expected_generic_rolling_output_token_count"),
            ("generic_rolling_normal_lane_conflict_count", "expected_generic_rolling_normal_lane_conflict_count"),
            ("generic_rolling_target_draft_mismatch_count", "expected_generic_rolling_target_draft_mismatch_count"),
        )
        for actual_key, expected_key in comparisons:
            if summary[actual_key] != summary[expected_key]:
                errors.append(
                    f"{actual_key} mismatch: actual={summary[actual_key]} expected={summary[expected_key]}"
                )
        if bool(summary["generic_rolling_parity_ok"]) != bool(summary["expected_generic_rolling_parity_ok"]):
            errors.append(
                "generic_rolling_parity_ok mismatch: "
                f"actual={summary['generic_rolling_parity_ok']} "
                f"expected={summary['expected_generic_rolling_parity_ok']}"
            )
    else:
        leaked_fields = [
            field
            for field in GENERIC_RUNTIME_INT_FIELDS
            if field != "generic_rolling_max_depth" and int_value(summary.get(field), 0) != 0
        ]
        if leaked_fields:
            errors.append(f"generic rolling runtime fields nonzero while disabled: {leaked_fields}")

    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "generic_rolling_runtime_enabled",
        "generic_rolling_max_depth",
        "generic_rolling_node_count",
        "generic_rolling_max_observed_depth",
        "expected_generic_rolling_max_observed_depth",
        "generic_rolling_max_real_committed_depth",
        "expected_generic_rolling_max_real_committed_depth",
        "generic_rolling_full_commit_token_count",
        "expected_generic_rolling_full_commit_token_count",
        "generic_rolling_partial_recovered_token_count",
        "expected_generic_rolling_partial_recovered_token_count",
        "generic_rolling_revised_token_count",
        "expected_generic_rolling_revised_token_count",
        "generic_rolling_output_token_count",
        "expected_generic_rolling_output_token_count",
        "generic_rolling_descendant_cascade_discard_count",
        "generic_rolling_normal_lane_conflict_count",
        "generic_rolling_target_draft_mismatch_count",
        "generic_rolling_parity_ok",
        "expected_generic_rolling_parity_ok",
        "combined_real_committed_token_count",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "combined_actual_revised_token_increment_sum",
        "combined_actual_output_token_increment_sum",
        "partial_prefix_recovery_enabled",
        "partial_prefix_recovery_success_count",
        "partial_prefix_total_recovered_token_count",
        "partial_prefix_revised_token_count",
        "depth_gt4_real_commit_count",
    ):
        print(f"{key}={summary.get(key)}")


def _full_depth4_records() -> list[dict[str, Any]]:
    records = synthetic_records()
    for record in records:
        add_depth4_shadow(record)
        add_depth4_commit(record)
    return records


def _apply_generic_fields(
    records: list[dict[str, Any]],
    *,
    enabled: bool,
    full_tokens: int = 44,
    partial_tokens: int = 0,
    revised_tokens: int = 0,
    output_tokens: int = 44,
    max_observed_depth: int = 4,
    max_real_depth: int = 4,
    target_draft_mismatch: int = 0,
    normal_conflict: int = 0,
    parity_ok: bool = True,
) -> None:
    for record in records:
        record["generic_rolling_runtime_enabled"] = bool(enabled)
        record["enable_generic_rolling_runtime_loop"] = bool(enabled)
        record["generic_rolling_max_depth"] = 4
        record["generic_rolling_node_count"] = 5 if enabled else 0
        record["generic_rolling_max_observed_depth"] = int(max_observed_depth) if enabled else 0
        record["generic_rolling_max_real_committed_depth"] = int(max_real_depth) if enabled else 0
        record["generic_rolling_full_commit_token_count"] = int(full_tokens) if enabled else 0
        record["generic_rolling_partial_recovered_token_count"] = int(partial_tokens) if enabled else 0
        record["generic_rolling_revised_token_count"] = int(revised_tokens) if enabled else 0
        record["generic_rolling_output_token_count"] = int(output_tokens) if enabled else 0
        record["generic_rolling_descendant_cascade_discard_count"] = 0
        record["generic_rolling_normal_lane_conflict_count"] = int(normal_conflict) if enabled else 0
        record["generic_rolling_target_draft_mismatch_count"] = int(target_draft_mismatch) if enabled else 0
        record["generic_rolling_parity_ok"] = bool(parity_ok)


def _partial_records() -> list[dict[str, Any]]:
    records = _full_depth4_records()
    _apply_partial_recovery(
        records,
        proposal_id=900000105,
        seq_id=8,
        depth=3,
        accepted_len=1,
        revised_count=1,
        frontier_before=44,
        descendants=(900000106,),
    )
    return records


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
    _assert_pass(
        "disabled legacy full accept",
        disabled,
        {
            "generic_rolling_runtime_enabled": False,
            "combined_real_committed_token_count": 44,
        },
    )

    full_accept = _full_depth4_records()
    _apply_generic_fields(full_accept, enabled=True)
    _assert_pass(
        "generic full accept parity",
        full_accept,
        {
            "generic_rolling_runtime_enabled": True,
            "generic_rolling_output_token_count": 44,
            "expected_generic_rolling_output_token_count": 44,
            "generic_rolling_parity_ok": True,
        },
    )

    partial = _partial_records()
    _apply_generic_fields(partial, enabled=True, partial_tokens=2, revised_tokens=1, output_tokens=46)
    _assert_pass(
        "generic partial recovery parity",
        partial,
        {
            "generic_rolling_partial_recovered_token_count": 2,
            "generic_rolling_revised_token_count": 1,
            "generic_rolling_output_token_count": 46,
            "expected_generic_rolling_output_token_count": 46,
        },
    )

    bad_output = deepcopy(partial)
    _apply_generic_fields(bad_output, enabled=True, partial_tokens=2, revised_tokens=1, output_tokens=45)
    _assert_fail("bad output mismatch", bad_output, "generic_rolling_output_token_count mismatch")

    bad_mismatch = deepcopy(partial)
    _apply_generic_fields(
        bad_mismatch,
        enabled=True,
        partial_tokens=2,
        revised_tokens=1,
        output_tokens=46,
        target_draft_mismatch=1,
        parity_ok=False,
    )
    _assert_fail("bad target/draft mismatch", bad_mismatch, "target_draft_mismatch")

    bad_depth = _full_depth4_records()
    _apply_generic_fields(bad_depth, enabled=True, max_observed_depth=5, max_real_depth=5, parity_ok=False)
    for record in bad_depth:
        record["rolling_depth_gt4_real_commit_count"] = 1
    _assert_fail("bad depth gt4", bad_depth, "depth_gt4")

    leaked_disabled = _full_depth4_records()
    _apply_generic_fields(leaked_disabled, enabled=False)
    for record in leaked_disabled:
        record["generic_rolling_output_token_count"] = 44
    _assert_fail("disabled generic leakage", leaked_disabled, "nonzero while disabled")

    print("Synthetic generic rolling runtime parity checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8s generic rolling runtime parity fields.")
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
