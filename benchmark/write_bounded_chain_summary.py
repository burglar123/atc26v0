#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import (  # noqa: E402
    int_value,
    parse_legacy_rolling_chain,
    summarize_registry,
)
from benchmark.check_bounded_rolling_readiness_audit import (  # noqa: E402
    GENERIC_PARITY_FIELD_PAIRS,
    load_trace,
    synthetic_result_payload,
    validate_records as validate_legacy_records,
)
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    load_json,
)
from benchmark.check_generic_bounded_rolling_chain import (  # noqa: E402
    add_depth4_shadow,
    clear_depth3_commit,
    synthetic_records,
)


ALIAS_FROM_GENERIC = {
    "one_shot_committed_proposal_count": "generic_one_shot_committed_proposal_count",
    "one_shot_committed_token_count": "generic_one_shot_committed_token_count",
    "depth1_committed_proposal_count": "generic_depth1_committed_proposal_count",
    "depth1_committed_token_count": "generic_depth1_committed_token_count",
    "depth2_committed_proposal_count": "generic_depth2_committed_proposal_count",
    "depth2_committed_token_count": "generic_depth2_committed_token_count",
    "depth3_committed_proposal_count": "generic_depth3_committed_proposal_count",
    "depth3_committed_token_count": "generic_depth3_committed_token_count",
    "depth4_committed_proposal_count": "generic_depth4_committed_proposal_count",
    "depth4_committed_token_count": "generic_depth4_committed_token_count",
    "depth4_shadow_generated_proposal_count": "generic_depth4_shadow_generated_proposal_count",
    "depth4_shadow_generated_token_count": "generic_depth4_shadow_generated_token_count",
    "depth4_shadow_ready_proposal_count": "generic_depth4_shadow_ready_proposal_count",
    "depth4_shadow_ready_token_count": "generic_depth4_shadow_ready_token_count",
    "depth4_shadow_invalidated_count": "generic_depth4_shadow_invalidated_count",
    "combined_real_committed_token_count": "generic_combined_real_committed_token_count",
    "max_observed_depth": "generic_max_observed_depth",
    "max_real_committed_depth": "generic_max_real_committed_depth",
    "depth4_real_commit_count": "generic_depth4_real_commit_count",
    "depth_gt3_real_commit_count": "generic_depth_gt3_real_commit_count",
    "depth_gt4_real_commit_count": "generic_depth_gt4_real_commit_count",
    "depth_gt3_committed_proposal_count": "generic_depth_gt3_committed_proposal_count",
    "normal_lane_conflict_count": "generic_normal_lane_conflict_count",
    "missing_buffered_proposal_unexpected_count": "generic_missing_buffered_proposal_unexpected_count",
    "duplicate_commit_count": "generic_duplicate_commit_count",
    "invalid_committed_child_count": "generic_invalid_committed_child_count",
    "cascade_committed_child_count": "generic_cascade_committed_child_count",
    "parent_missing_committed_child_count": "generic_parent_missing_committed_child_count",
    "generated_only_committed_child_count": "generic_generated_only_committed_child_count",
    "non_full_accept_committed_count": "generic_non_full_accept_committed_count",
    "stale_or_frontier_committed_child_count": "generic_stale_or_frontier_committed_child_count",
    "target_draft_length_mismatch_count": "generic_target_draft_length_mismatch_count",
    "target_draft_token_mismatch_count": "generic_target_draft_token_mismatch_count",
    "combined_accounting_ok": "generic_combined_accounting_ok",
    "target_draft_accounting_ok": "generic_target_draft_accounting_ok",
}


def generic_legacy_parity_ok(legacy_summary: dict[str, Any], generic_summary: dict[str, Any]) -> bool:
    return all(
        legacy_summary.get(legacy_key) == generic_summary.get(generic_key)
        for legacy_key, generic_key in GENERIC_PARITY_FIELD_PAIRS
    )


def build_chain_summary(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    *,
    case_name: str | None = None,
) -> dict[str, Any]:
    result_payload = result_payload or {}
    accounting = aggregate_performance_accounting(records, result_payload)
    registry = parse_legacy_rolling_chain(records)
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if isinstance(result_args, dict):
        registry.max_configured_depth = max(
            registry.max_configured_depth,
            int_value(result_args.get("max_rolling_continuous_depth"), 0),
        )
    generic_summary = summarize_registry(registry, accounting=accounting)
    legacy_errors, legacy_summary = validate_legacy_records(records, result_payload)
    parity_ok = generic_legacy_parity_ok(legacy_summary, generic_summary)
    generic_chain_accounting_ok = bool(generic_summary.get("generic_combined_accounting_ok", False)) and bool(
        generic_summary.get("generic_target_draft_accounting_ok", False)
    )

    summary: dict[str, Any] = {
        "case_name": case_name,
        "total_trace_records": len(records),
        "generic_summary_source": "bounded_rolling_chain_parser.summarize_registry",
        **registry.flags,
        "max_configured_depth": registry.max_configured_depth,
        **generic_summary,
    }
    for alias_key, generic_key in ALIAS_FROM_GENERIC.items():
        summary[alias_key] = generic_summary.get(generic_key)
    summary.update(
        {
            "combined_actual_verified_token_increment_sum": int_value(
                accounting.get("combined_actual_verified_token_increment_sum"),
                0,
            ),
            "combined_actual_accepted_token_increment_sum": int_value(
                accounting.get("combined_actual_accepted_token_increment_sum"),
                0,
            ),
            "performance_warnings": accounting.get("performance_warnings", []),
            "legacy_generic_parity_ok": parity_ok,
            "generic_chain_accounting_ok": generic_chain_accounting_ok,
            "legacy_audit_error_count": len(legacy_errors),
            "legacy_audit_errors": legacy_errors,
        }
    )
    return summary


def write_chain_summary(summary: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)


def print_compact(summary: dict[str, Any]) -> None:
    for key in (
        "case_name",
        "one_shot_committed_token_count",
        "depth1_committed_token_count",
        "depth2_committed_token_count",
        "depth3_committed_token_count",
        "depth4_committed_token_count",
        "depth4_shadow_generated_token_count",
        "depth4_shadow_ready_token_count",
        "combined_real_committed_token_count",
        "max_observed_depth",
        "max_real_committed_depth",
        "depth_gt3_real_commit_count",
        "depth_gt4_real_commit_count",
        "depth4_real_commit_count",
        "normal_lane_conflict_count",
        "combined_accounting_ok",
        "target_draft_accounting_ok",
        "legacy_generic_parity_ok",
        "generic_chain_accounting_ok",
    ):
        print(f"{key}={summary.get(key)}")


def assert_summary(name: str, summary: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SystemExit(f"synthetic {name} {key} mismatch: expected {value}, got {summary.get(key)}")


def run_synthetic() -> None:
    records = synthetic_records()
    summary = build_chain_summary(records, synthetic_result_payload(), case_name="synthetic_good_n3")
    assert_summary(
        "good N=3",
        summary,
        {
            "one_shot_committed_token_count": 12,
            "depth1_committed_token_count": 8,
            "depth2_committed_token_count": 8,
            "depth3_committed_token_count": 8,
            "combined_real_committed_token_count": 36,
            "max_real_committed_depth": 3,
            "depth_gt3_real_commit_count": 0,
            "combined_accounting_ok": True,
            "target_draft_accounting_ok": True,
            "legacy_generic_parity_ok": True,
            "generic_chain_accounting_ok": True,
        },
    )

    shadow_only = deepcopy(records)
    for record in shadow_only:
        clear_depth3_commit(record)
    summary = build_chain_summary(
        shadow_only,
        synthetic_result_payload(),
        case_name="synthetic_depth3_shadow_only",
    )
    assert_summary(
        "depth3 shadow only",
        summary,
        {
            "depth3_committed_token_count": 0,
            "max_observed_depth": 3,
            "max_real_committed_depth": 2,
            "depth_gt3_real_commit_count": 0,
            "combined_accounting_ok": True,
            "target_draft_accounting_ok": True,
            "legacy_generic_parity_ok": True,
            "generic_chain_accounting_ok": True,
        },
    )

    depth4_shadow = deepcopy(records)
    for record in depth4_shadow:
        add_depth4_shadow(record)
    summary = build_chain_summary(
        depth4_shadow,
        synthetic_result_payload(),
        case_name="synthetic_depth4_shadow",
    )
    assert_summary(
        "depth4 shadow",
        summary,
        {
            "depth4_shadow_generated_token_count": 8,
            "depth4_shadow_ready_token_count": 8,
            "combined_real_committed_token_count": 36,
            "max_observed_depth": 4,
            "max_real_committed_depth": 3,
            "depth4_real_commit_count": 0,
            "combined_accounting_ok": True,
            "target_draft_accounting_ok": True,
            "legacy_generic_parity_ok": True,
            "generic_chain_accounting_ok": True,
        },
    )
    print("Synthetic bounded chain summary checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a bounded rolling chain summary JSON from legacy trace fields.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case-name")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic()
        return 0
    if args.result is None:
        raise SystemExit("RESULT path is required outside --synthetic mode")

    records = load_trace(args.trace)
    result_payload = load_json(args.result)
    summary = build_chain_summary(records, result_payload, case_name=args.case_name)
    if args.out is not None:
        write_chain_summary(summary, args.out)
        print(f"wrote_chain_summary={args.out}")
    print_compact(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
