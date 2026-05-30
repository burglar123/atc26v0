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

from benchmark.bounded_rolling_chain_parser import (  # noqa: E402
    GENERIC_ROLLING_ACTION,
    RollingProposalNode,
    int_value,
    parse_legacy_rolling_chain,
)
from benchmark.check_bounded_rolling_readiness_audit import (  # noqa: E402
    load_trace,
    synthetic_result_payload,
)
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    load_json,
)
from benchmark.check_generic_bounded_rolling_chain import (  # noqa: E402
    add_depth4_shadow,
    synthetic_records,
)
from benchmark.partial_recovery_checker_utils import (  # noqa: E402
    assert_partial_recovery_checker_cases,
    partial_recovery_accounting_errors,
    partial_recovery_descendant_commit_count,
    partial_recovery_print_fields,
    partial_recovery_summary_fields,
)


def _depth4_shadow_enabled(records: list[dict[str, Any]]) -> bool:
    return any(
        bool(record.get("enable_rolling_continuous_depth4_shadow_dry_run", False))
        or bool(record.get("rolling_depth4_shadow_enabled", False))
        for record in records
    )


def _depth4_commit_enabled(records: list[dict[str, Any]]) -> bool:
    return any(
        bool(record.get("enable_rolling_continuous_depth4_commit_ready_only", False))
        or bool(record.get("rolling_depth4_commit_enabled", False))
        for record in records
    )


def _node_parent_root(parent: RollingProposalNode) -> int | None:
    return parent.root_id if parent.root_id is not None else parent.proposal_id


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    result_payload = result_payload or {}
    registry = parse_legacy_rolling_chain(records)
    accounting = aggregate_performance_accounting(records, result_payload)
    errors: list[str] = []

    enabled = _depth4_shadow_enabled(records)
    commit_enabled = _depth4_commit_enabled(records)
    generated_ids = set(registry.generated_by_depth[4])
    ready_ids = set(registry.ready_by_depth[4])
    invalidated_ids = set(registry.invalidated_by_depth[4])
    observed_ids = generated_ids | ready_ids | invalidated_ids

    if observed_ids and not enabled:
        errors.append("depth4 shadow evidence appears while depth4 shadow flag is disabled")
    if enabled and not registry.flags.get("rolling_depth3_commit_enabled", False):
        errors.append("depth4 shadow requires depth3 commit to be enabled")
    if registry.depth4_real_commit_count and not commit_enabled:
        errors.append("rolling depth4 real commit count must remain zero")
    if registry.depth_gt4_real_commit_count:
        errors.append("rolling depth>4 real commit count must remain zero")
    if registry.depth_gt3_real_commit_count and not commit_enabled:
        errors.append("rolling depth>3 real commit count must remain zero")
    if registry.higher_depth_commit_ids:
        errors.append(f"depth>4 real committed proposal ids present: {sorted(registry.higher_depth_commit_ids)}")
    if int_value(accounting.get("rolling_depth4_real_commit_count"), 0) != 0 and not commit_enabled:
        errors.append("accounting reports depth4 real commit count")
    if int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0) != 0 and not commit_enabled:
        errors.append("accounting reports depth4 real committed tokens")

    for proposal_id in sorted(observed_ids):
        node = registry.nodes_by_id.get(proposal_id)
        if node is None:
            errors.append(f"depth4 child {proposal_id} missing generic node")
            continue
        if node.depth != 4:
            errors.append(f"depth4 child {proposal_id} has depth {node.depth}")
        if proposal_id in ready_ids and proposal_id not in generated_ids:
            errors.append(f"depth4 child {proposal_id} ready without generated evidence")
        if proposal_id in ready_ids and proposal_id in invalidated_ids:
            errors.append(f"depth4 child {proposal_id} is both ready and invalidated")
        if node.parent_id is None:
            errors.append(f"depth4 child {proposal_id} missing parent")
            continue
        parent = registry.nodes_by_id.get(node.parent_id)
        if parent is None:
            errors.append(f"depth4 child {proposal_id} parent {node.parent_id} missing")
            continue
        if parent.depth != 3:
            errors.append(f"depth4 child {proposal_id} parent depth {parent.depth}, expected 3")
        if not parent.committed:
            errors.append(f"depth4 child {proposal_id} parent {node.parent_id} was not real committed")
        if parent.verify_result is not None and parent.verify_result != "full_accept":
            errors.append(f"depth4 child {proposal_id} parent {node.parent_id} was not full accept")
        if parent.action is not None and parent.action != GENERIC_ROLLING_ACTION:
            errors.append(f"depth4 child {proposal_id} parent {node.parent_id} action is not real commit")
        if parent.seq_id is not None and node.seq_id is not None and parent.seq_id != node.seq_id:
            errors.append(f"depth4 child {proposal_id} seq differs from depth3 parent")
        parent_root = _node_parent_root(parent)
        if node.root_id is not None and parent_root is not None and node.root_id != parent_root:
            errors.append(f"depth4 child {proposal_id} root differs from depth3 parent root")

    candidate_tokens = int_value(accounting.get("rolling_depth4_child_candidate_token_count"), 0)
    ready_tokens = int_value(accounting.get("rolling_depth4_child_ready_shadow_token_count"), 0)
    if ready_tokens > candidate_tokens:
        errors.append("depth4 ready shadow tokens exceed candidate tokens")
    if int_value(accounting.get("rolling_depth4_child_ready_shadow_proposal_count"), 0) > int_value(
        accounting.get("rolling_depth4_child_candidate_proposal_count"), 0
    ):
        errors.append("depth4 ready shadow proposal count exceeds candidate proposal count")
    if registry.normal_lane_conflict_count:
        errors.append("normal lane conflict evidence present")

    lower_depth_sum = (
        int_value(accounting.get("eager_committed_token_count"), 0)
        + int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0)
    )
    expected_combined = lower_depth_sum + (
        int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0) if commit_enabled else 0
    )
    descendant_committed_after_partial_count = partial_recovery_descendant_commit_count(records)
    partial_errors, legal_partial_total = partial_recovery_accounting_errors(
        accounting,
        descendant_committed_after_partial_count=descendant_committed_after_partial_count,
    )
    errors.extend(partial_errors)
    expected_combined += legal_partial_total
    combined = int_value(accounting.get("combined_real_committed_token_count"), 0)
    if combined != expected_combined:
        errors.append(
            "combined real committed tokens must include real commits plus legal partial recovery output: "
            f"combined={combined} expected={expected_combined}"
        )

    max_observed_depth = registry.max_observed_depth
    max_real_committed_depth = registry.max_real_committed_depth
    summary = {
        "depth4_shadow_enabled": enabled,
        "depth4_commit_enabled": commit_enabled,
        "rolling_depth4_child_candidate_proposal_count": int_value(
            accounting.get("rolling_depth4_child_candidate_proposal_count"), 0
        ),
        "rolling_depth4_child_candidate_token_count": candidate_tokens,
        "rolling_depth4_child_ready_shadow_proposal_count": int_value(
            accounting.get("rolling_depth4_child_ready_shadow_proposal_count"), 0
        ),
        "rolling_depth4_child_ready_shadow_token_count": ready_tokens,
        "rolling_depth4_child_invalidated_count": int_value(
            accounting.get("rolling_depth4_child_invalidated_count"), 0
        ),
        "rolling_depth4_real_commit_count": registry.depth4_real_commit_count,
        "rolling_depth4_real_committed_token_count": int_value(
            accounting.get("rolling_depth4_real_committed_token_count"), 0
        ),
        "rolling_depth_gt4_real_commit_count": registry.depth_gt4_real_commit_count,
        "rolling_depth_gt3_real_commit_count": registry.depth_gt3_real_commit_count,
        "combined_real_committed_token_count": combined,
        "expected_combined_real_committed_token_count": expected_combined,
        **partial_recovery_summary_fields(
            accounting,
            descendant_committed_after_partial_count=descendant_committed_after_partial_count,
        ),
        "max_observed_depth": max_observed_depth,
        "max_real_committed_depth": max_real_committed_depth,
        "normal_lane_conflict_count": registry.normal_lane_conflict_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "depth4_shadow_enabled",
        "depth4_commit_enabled",
        "rolling_depth4_child_candidate_proposal_count",
        "rolling_depth4_child_candidate_token_count",
        "rolling_depth4_child_ready_shadow_proposal_count",
        "rolling_depth4_child_ready_shadow_token_count",
        "rolling_depth4_child_invalidated_count",
        "rolling_depth4_real_commit_count",
        "rolling_depth4_real_committed_token_count",
        "rolling_depth_gt4_real_commit_count",
        "rolling_depth_gt3_real_commit_count",
        *partial_recovery_print_fields(),
        "max_observed_depth",
        "max_real_committed_depth",
        "normal_lane_conflict_count",
    ):
        print(f"{key}={summary.get(key)}")


def assert_pass(name: str, records: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic {name} failed: {errors}\nsummary={summary}")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SystemExit(f"synthetic {name} {key} mismatch: expected {value}, got {summary.get(key)}")


def assert_fail(name: str, records: list[dict[str, Any]], needle: str) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if not errors:
        raise SystemExit(f"synthetic {name} should fail\nsummary={summary}")
    if not any(needle in error for error in errors):
        raise SystemExit(f"synthetic {name} failed for wrong reason: {errors}\nsummary={summary}")


def run_synthetic() -> None:
    no_depth4 = synthetic_records()
    assert_pass(
        "backward compatible no depth4",
        no_depth4,
        {
            "depth4_shadow_enabled": False,
            "rolling_depth4_child_candidate_token_count": 0,
            "rolling_depth4_real_commit_count": 0,
            "max_observed_depth": 3,
            "max_real_committed_depth": 3,
        },
    )

    good = deepcopy(no_depth4)
    for record in good:
        add_depth4_shadow(record)
    assert_pass(
        "good depth4 shadow",
        good,
        {
            "depth4_shadow_enabled": True,
            "rolling_depth4_child_candidate_token_count": 8,
            "rolling_depth4_child_ready_shadow_token_count": 8,
            "rolling_depth4_real_commit_count": 0,
            "rolling_depth4_real_committed_token_count": 0,
            "combined_real_committed_token_count": 36,
            "max_observed_depth": 4,
            "max_real_committed_depth": 3,
        },
    )

    bad_real = deepcopy(good)
    bad_real[0]["rolling_depth4_real_commit_count"] = 1
    bad_real[0]["rolling_depth_gt3_real_commit_count"] = 1
    assert_fail("bad depth4 real commit", bad_real, "depth4 real commit")

    bad_parent = deepcopy(good)
    for record in bad_parent:
        record["rolling_depth4_child_parent_by_proposal_id"] = {"900000104": 12345}
    assert_fail("bad depth4 parent", bad_parent, "parent")

    disabled_with_evidence = deepcopy(good)
    for record in disabled_with_evidence:
        record["enable_rolling_continuous_depth4_shadow_dry_run"] = False
        record["rolling_depth4_shadow_enabled"] = False
    assert_fail("depth4 evidence disabled", disabled_with_evidence, "flag")

    bad_ready_accounting = deepcopy(good)
    for record in bad_ready_accounting:
        record["rolling_depth4_child_ready_shadow_proposal_ids"] = [900000104, 900000105]
        record["rolling_depth4_child_ready_shadow_seq_ids"] = [7, 7]
        record["rolling_depth4_child_token_count_by_proposal_id"]["900000105"] = 8
    assert_fail("bad depth4 ready accounting", bad_ready_accounting, "ready")

    # legal depth4 commit with commit flag enabled — should pass
    legal_d4_commit = deepcopy(good)
    p4 = 900000104
    for i, record in enumerate(legal_d4_commit):
        side = record.get("rolling_depth3_commit_side", "target")
        seq_id = 7
        record["enable_rolling_continuous_depth4_commit_ready_only"] = True
        record["rolling_depth4_commit_enabled"] = True
        record["rolling_depth4_commit_source"] = "rolling_depth4_ready_only"
        record["rolling_depth4_commit_side"] = side
        record["rolling_depth4_commit_plan_id"] = 42
        record["rolling_depth4_commit_step_id"] = 22
        record["rolling_depth4_real_committed_proposal_ids"] = [p4]
        record["rolling_depth4_real_committed_seq_ids"] = [seq_id]
        record["rolling_depth4_real_committed_token_count_by_proposal_id"] = {str(p4): 8}
        record["rolling_depth4_real_committed_token_count"] = 8
        record["rolling_depth4_tokens_verified"] = 8
        record["rolling_depth4_tokens_accepted"] = 8
        record["rolling_depth4_tokens_rejected"] = 0
        record["rolling_depth4_tokens_invalidated"] = 0
        record["rolling_depth4_real_commit_count"] = 1 if i == 0 else 0
        record["rolling_depth_gt3_real_commit_count"] = 1 if i == 0 else 0
        record["rolling_depth_gt4_real_commit_count"] = 0
    assert_pass(
        "legal depth4 commit passes shadow checker",
        legal_d4_commit,
        {
            "depth4_shadow_enabled": True,
            "depth4_commit_enabled": True,
            "rolling_depth4_real_commit_count": 1,
            "rolling_depth4_real_committed_token_count": 8,
            "rolling_depth_gt4_real_commit_count": 0,
            "combined_real_committed_token_count": 44,
            "max_real_committed_depth": 4,
        },
    )

    def _make_partial_recovery_records() -> list[dict[str, Any]]:
        records = deepcopy(good)
        for i, record in enumerate(records):
            side = record.get("rolling_depth3_commit_side", "target")
            seq_id = 7
            record["enable_rolling_continuous_depth4_commit_ready_only"] = True
            record["rolling_depth4_commit_enabled"] = True
            record["rolling_depth4_commit_source"] = "rolling_depth4_ready_only"
            record["rolling_depth4_commit_side"] = side
            record["rolling_depth4_commit_plan_id"] = 42
            record["rolling_depth4_commit_step_id"] = 22
            record["rolling_depth4_real_committed_proposal_ids"] = [p4]
            record["rolling_depth4_real_committed_seq_ids"] = [seq_id]
            record["rolling_depth4_real_committed_token_count_by_proposal_id"] = {str(p4): 8}
            record["rolling_depth4_real_committed_token_count"] = 8
            record["rolling_depth4_tokens_verified"] = 8
            record["rolling_depth4_tokens_accepted"] = 8
            record["rolling_depth4_tokens_rejected"] = 0
            record["rolling_depth4_tokens_invalidated"] = 0
            record["rolling_depth4_real_commit_count"] = 1 if i == 0 else 0
            record["rolling_depth_gt3_real_commit_count"] = 1 if i == 0 else 0
            record["rolling_depth_gt4_real_commit_count"] = 0
        return records

    assert_partial_recovery_checker_cases(
        "rolling depth4 shadow dry-run",
        _make_partial_recovery_records,
        lambda records: validate_records(records, synthetic_result_payload()),
        expected_full_accept_combined_token_count=44,
    )

    # depth5 (depth_gt4) — always fail
    bad_d5 = deepcopy(good)
    bad_d5[0]["rolling_depth_gt4_real_commit_count"] = 1
    assert_fail("bad depth_gt4", bad_d5, "depth>4")

    print("Synthetic rolling depth4 shadow dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8o rolling depth4 shadow dry-run traces.")
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
