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

from benchmark.check_bounded_rolling_readiness_audit import (  # noqa: E402
    load_trace,
    synthetic_result_payload,
    validate_records as validate_legacy_records,
)
from benchmark.bounded_rolling_chain_parser import parse_legacy_rolling_chain, summarize_registry  # noqa: E402
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402
from benchmark.check_generic_bounded_rolling_chain import (  # noqa: E402
    synthetic_records,
    validate_records as validate_generic_records,
)


CORE_FIELD_PAIRS = (
    ("one_shot_committed_token_count", "generic_one_shot_committed_token_count"),
    ("depth1_committed_token_count", "generic_depth1_committed_token_count"),
    ("depth2_committed_token_count", "generic_depth2_committed_token_count"),
    ("depth3_committed_token_count", "generic_depth3_committed_token_count"),
    ("combined_real_committed_token_count", "generic_combined_real_committed_token_count"),
    ("max_real_committed_depth", "generic_max_real_committed_depth"),
    ("max_observed_depth", "generic_max_observed_depth"),
    ("depth4_real_commit_count", "generic_depth4_real_commit_count"),
    ("depth_gt3_real_commit_count", "generic_depth_gt3_real_commit_count"),
    ("normal_lane_conflict_count", "generic_normal_lane_conflict_count"),
    ("missing_buffered_proposal_unexpected_count", "generic_missing_buffered_proposal_unexpected_count"),
    ("duplicate_commit_count", "generic_duplicate_commit_count"),
    ("invalid_committed_child_count", "generic_invalid_committed_child_count"),
    ("cascade_committed_child_count", "generic_cascade_committed_child_count"),
    ("parent_missing_committed_child_count", "generic_parent_missing_committed_child_count"),
    ("combined_accounting_ok", "generic_combined_accounting_ok"),
    ("target_draft_accounting_ok", "generic_target_draft_accounting_ok"),
)


def compare_summaries(legacy: dict[str, Any], generic: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for legacy_key, generic_key in CORE_FIELD_PAIRS:
        legacy_value = legacy.get(legacy_key)
        generic_value = generic.get(generic_key)
        if legacy_value != generic_value:
            mismatches.append(f"{legacy_key}: legacy={legacy_value!r} generic={generic_value!r}")
    return mismatches


def compare_chain_summary(
    legacy: dict[str, Any],
    generic_checker: dict[str, Any],
    chain_summary: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for legacy_key, _generic_key in CORE_FIELD_PAIRS:
        legacy_value = legacy.get(legacy_key)
        chain_value = chain_summary.get(legacy_key)
        if legacy_value != chain_value:
            mismatches.append(
                f"{legacy_key}: legacy={legacy_value!r} chain_summary={chain_value!r}"
            )
        checker_value = generic_checker.get(legacy_key)
        if checker_value != chain_value:
            mismatches.append(
                f"{legacy_key}: generic_checker={checker_value!r} chain_summary={chain_value!r}"
            )
    return mismatches


def print_comparison(
    legacy: dict[str, Any],
    generic: dict[str, Any],
    mismatches: list[str],
    chain_summary: dict[str, Any] | None = None,
) -> None:
    print("legacy/generic bounded audit comparison")
    print("generic_summary_source=bounded_rolling_chain_parser.summarize_registry")
    for legacy_key, generic_key in CORE_FIELD_PAIRS:
        line = f"{legacy_key}: legacy={legacy.get(legacy_key)} generic={generic.get(generic_key)}"
        if chain_summary is not None:
            line += f" chain_summary={chain_summary.get(legacy_key)}"
        print(line)
    if mismatches:
        print("Mismatches:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
    else:
        print("Parity: pass")


def run_compare(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any],
    *,
    chain_summary: dict[str, Any] | None = None,
) -> int:
    legacy_errors, legacy = validate_legacy_records(records, result_payload)
    accounting = aggregate_performance_accounting(records, result_payload)
    generic = summarize_registry(parse_legacy_rolling_chain(records), accounting=accounting)
    generic_errors, generic_checker = validate_generic_records(records, result_payload)
    mismatches = compare_summaries(legacy, generic)
    if chain_summary is not None:
        mismatches.extend(compare_chain_summary(legacy, generic_checker, chain_summary))
        if chain_summary.get("legacy_generic_parity_ok") is not True:
            mismatches.append("chain_summary legacy_generic_parity_ok is not true")
        if chain_summary.get("generic_chain_accounting_ok") is not True:
            mismatches.append("chain_summary generic_chain_accounting_ok is not true")
    print_comparison(legacy, generic, mismatches, chain_summary)

    errors: list[str] = []
    if legacy_errors:
        errors.append(f"legacy audit errors: {legacy_errors}")
    if generic_errors:
        errors.append(f"generic checker errors: {generic_errors}")
    errors.extend(mismatches)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


def run_synthetic() -> None:
    records = synthetic_records()
    legacy_errors, legacy = validate_legacy_records(records, synthetic_result_payload())
    accounting = aggregate_performance_accounting(records, synthetic_result_payload())
    generic = summarize_registry(parse_legacy_rolling_chain(records), accounting=accounting)
    generic_errors, generic_checker = validate_generic_records(records, synthetic_result_payload())
    if legacy_errors:
        raise SystemExit(f"synthetic legacy audit failed: {legacy_errors}")
    if generic_errors:
        raise SystemExit(f"synthetic generic checker failed: {generic_errors}")
    mismatches = compare_summaries(legacy, generic)
    if mismatches:
        raise SystemExit(f"synthetic parity failed: {mismatches}")

    bad_generic = deepcopy(generic)
    bad_generic["generic_depth3_committed_token_count"] = int(
        bad_generic["generic_depth3_committed_token_count"]
    ) + 1
    bad_mismatches = compare_summaries(legacy, bad_generic)
    if not bad_mismatches:
        raise SystemExit("synthetic parity mismatch case should fail")

    chain_summary = {
        **legacy,
        "legacy_generic_parity_ok": True,
        "generic_chain_accounting_ok": True,
    }
    chain_mismatches = compare_chain_summary(legacy, generic_checker, chain_summary)
    if chain_mismatches:
        raise SystemExit(f"synthetic chain summary parity failed: {chain_mismatches}")
    bad_chain = dict(chain_summary)
    bad_chain["depth3_committed_token_count"] += 1
    if not compare_chain_summary(legacy, generic_checker, bad_chain):
        raise SystemExit("synthetic chain summary mismatch case should fail")

    print("Synthetic generic vs legacy bounded audit comparison checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare legacy bounded audit output with generic checker output.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--chain-summary", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic()
        return 0

    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result else {}
    chain_summary = None
    if args.chain_summary is not None:
        data = json.loads(args.chain_summary.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{args.chain_summary} does not contain a JSON object")
        chain_summary = data
    return run_compare(records, result_payload, chain_summary=chain_summary)


if __name__ == "__main__":
    raise SystemExit(main())
