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


def _enabled(records: list[dict[str, Any]], *fields: str) -> bool:
    return any(bool(record.get(field, False)) for record in records for field in fields)


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

    shadow_enabled = _enabled(
        records,
        "enable_rolling_continuous_depth4_shadow_dry_run",
        "rolling_depth4_shadow_enabled",
    )
    commit_enabled = _enabled(
        records,
        "enable_rolling_continuous_depth4_commit_ready_only",
        "rolling_depth4_commit_enabled",
    )
    committed_ids = set(registry.committed_by_depth[4])
    ready_ids = set(registry.ready_by_depth[4])
    generated_ids = set(registry.generated_by_depth[4])
    invalidated_ids = set(registry.invalidated_by_depth[4])
    cascade_ids = set(registry.cascade_by_depth[4])

    if commit_enabled and not shadow_enabled:
        errors.append("depth4 commit requires depth4 shadow flag")
    if committed_ids and not commit_enabled:
        errors.append("depth4 committed proposals appear while commit flag is disabled")
    if committed_ids and not shadow_enabled:
        errors.append("depth4 committed proposals appear while shadow flag is disabled")
    if registry.depth_gt4_real_commit_count:
        errors.append("rolling depth>4 real commit count must remain zero")
    if registry.higher_depth_commit_ids:
        errors.append(f"depth>4 real committed proposal ids present: {sorted(registry.higher_depth_commit_ids)}")
    if registry.normal_lane_conflict_count:
        errors.append("normal lane conflict evidence present")
    if registry.missing_buffered_proposal_unexpected_count:
        errors.append("unexpected missing buffered proposal evidence present")

    for proposal_id in sorted(committed_ids):
        node = registry.nodes_by_id.get(proposal_id)
        if node is None:
            errors.append(f"depth4 committed proposal {proposal_id} missing generic node")
            continue
        if node.depth != 4:
            errors.append(f"depth4 committed proposal {proposal_id} has depth {node.depth}")
        if proposal_id not in ready_ids:
            errors.append(f"depth4 committed proposal {proposal_id} was not ready shadow")
        if proposal_id not in generated_ids:
            errors.append(f"depth4 committed proposal {proposal_id} was not generated")
        if proposal_id in invalidated_ids or node.invalidated:
            errors.append(f"depth4 committed proposal {proposal_id} was invalidated")
        if proposal_id in cascade_ids or node.cascade_discarded:
            errors.append(f"depth4 committed proposal {proposal_id} was cascade-discarded")
        if node.stale_or_frontier_mismatch:
            errors.append(f"depth4 committed proposal {proposal_id} was stale/frontier-mismatched")
        if node.verify_result is not None and node.verify_result != "full_accept":
            errors.append(f"depth4 committed proposal {proposal_id} result is not full_accept")
        if node.action is not None and node.action != GENERIC_ROLLING_ACTION:
            errors.append(f"depth4 committed proposal {proposal_id} action is not real commit")
        if node.accept_len is not None and node.accept_len != node.token_count:
            errors.append(f"depth4 committed proposal {proposal_id} accept length is partial")
        if node.token_count <= 0:
            errors.append(f"depth4 committed proposal {proposal_id} token count must be positive")
        if node.parent_id is None:
            errors.append(f"depth4 committed proposal {proposal_id} missing depth3 parent")
            continue
        parent = registry.nodes_by_id.get(node.parent_id)
        if parent is None:
            errors.append(f"depth4 committed proposal {proposal_id} parent {node.parent_id} missing")
            continue
        if parent.depth != 3:
            errors.append(f"depth4 committed proposal {proposal_id} parent depth {parent.depth}, expected 3")
        if not parent.committed:
            errors.append(f"depth4 committed proposal {proposal_id} parent {node.parent_id} was not committed")
        if parent.verify_result is not None and parent.verify_result != "full_accept":
            errors.append(f"depth4 committed proposal {proposal_id} parent {node.parent_id} was not full_accept")
        if parent.action is not None and parent.action != GENERIC_ROLLING_ACTION:
            errors.append(f"depth4 committed proposal {proposal_id} parent {node.parent_id} action is not real commit")
        if parent.seq_id is not None and node.seq_id is not None and parent.seq_id != node.seq_id:
            errors.append(f"depth4 committed proposal {proposal_id} seq differs from depth3 parent")
        parent_root = _node_parent_root(parent)
        if node.root_id is not None and parent_root is not None and node.root_id != parent_root:
            errors.append(f"depth4 committed proposal {proposal_id} root differs from depth3 parent root")

    depth4_tokens = int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0)
    depth4_proposals = int_value(accounting.get("rolling_depth4_real_committed_proposal_count"), 0)
    if depth4_proposals != len(committed_ids):
        errors.append("depth4 committed proposal count mismatch")
    if depth4_tokens != sum(max(0, registry.token_by_depth[4].get(proposal_id, registry.gamma)) for proposal_id in committed_ids):
        errors.append("depth4 committed token count mismatch")
    if depth4_tokens:
        for key in (
            "rolling_depth4_target_actual_verified_token_increment_sum",
            "rolling_depth4_target_actual_accepted_token_increment_sum",
            "rolling_depth4_draft_actual_verified_token_increment_sum",
            "rolling_depth4_draft_actual_accepted_token_increment_sum",
        ):
            if int_value(accounting.get(key), 0) != depth4_tokens:
                errors.append(f"{key} must equal depth4 committed tokens")
        for key in (
            "rolling_depth4_target_actual_rejected_token_increment_sum",
            "rolling_depth4_target_actual_invalidated_token_increment_sum",
            "rolling_depth4_draft_actual_rejected_token_increment_sum",
            "rolling_depth4_draft_actual_invalidated_token_increment_sum",
        ):
            if int_value(accounting.get(key), 0) != 0:
                errors.append(f"{key} must remain zero")
    if committed_ids and int_value(accounting.get("rolling_depth4_real_commit_count"), 0) <= 0:
        errors.append("depth4 real commit count missing despite committed proposals")
    if depth4_proposals > int_value(accounting.get("rolling_depth4_child_ready_shadow_proposal_count"), 0):
        errors.append("depth4 committed proposal count exceeds ready shadow proposal count")
    if depth4_tokens > int_value(accounting.get("rolling_depth4_child_ready_shadow_token_count"), 0):
        errors.append("depth4 committed tokens exceed ready shadow tokens")

    expected_combined = (
        int_value(accounting.get("eager_committed_token_count"), 0)
        + int_value(accounting.get("continuous_eager_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0)
        + int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0)
        + depth4_tokens
    )
    descendant_committed_after_partial_count = partial_recovery_descendant_commit_count(records)
    partial_errors, legal_partial_total = partial_recovery_accounting_errors(
        accounting,
        descendant_committed_after_partial_count=descendant_committed_after_partial_count,
    )
    errors.extend(partial_errors)
    expected_combined += legal_partial_total
    if int_value(accounting.get("combined_real_committed_token_count"), 0) != expected_combined:
        errors.append("combined accounting must include depth4 committed tokens and legal partial recovery tokens exactly once")
    if int_value(accounting.get("combined_actual_verified_token_increment_sum"), 0) != expected_combined:
        errors.append("combined verified accounting must include depth4 committed tokens and legal partial recovery output exactly once")
    if legal_partial_total:
        expected_accepted = expected_combined - int_value(accounting.get("partial_prefix_revised_token_count"), 0)
        if int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0) != expected_accepted:
            errors.append("combined accepted accounting must exclude revised partial-recovery tokens")
    elif int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0) != expected_combined:
        errors.append("combined accepted accounting must include depth4 committed tokens exactly once")

    summary = {
        "depth4_shadow_enabled": shadow_enabled,
        "depth4_commit_enabled": commit_enabled,
        "rolling_depth4_commit_candidate_proposal_count": len(
            set().union(
                *[
                    set(int(item) for item in record.get("rolling_depth4_commit_candidate_proposal_ids", []) if isinstance(item, int))
                    for record in records
                ]
            )
        ) if records else 0,
        "rolling_depth4_child_ready_shadow_proposal_count": len(ready_ids),
        "rolling_depth4_real_committed_proposal_count": len(committed_ids),
        "rolling_depth4_real_committed_token_count": depth4_tokens,
        "rolling_depth4_real_commit_count": registry.depth4_real_commit_count,
        "rolling_depth_gt4_real_commit_count": registry.depth_gt4_real_commit_count,
        "combined_real_committed_token_count": int_value(accounting.get("combined_real_committed_token_count"), 0),
        "expected_combined_real_committed_token_count": expected_combined,
        **partial_recovery_summary_fields(
            accounting,
            descendant_committed_after_partial_count=descendant_committed_after_partial_count,
        ),
        "max_observed_depth": registry.max_observed_depth,
        "max_real_committed_depth": registry.max_real_committed_depth,
        "normal_lane_conflict_count": registry.normal_lane_conflict_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "depth4_shadow_enabled",
        "depth4_commit_enabled",
        "rolling_depth4_commit_candidate_proposal_count",
        "rolling_depth4_child_ready_shadow_proposal_count",
        "rolling_depth4_real_committed_proposal_count",
        "rolling_depth4_real_committed_token_count",
        "rolling_depth4_real_commit_count",
        "rolling_depth_gt4_real_commit_count",
        *partial_recovery_print_fields(),
        "max_observed_depth",
        "max_real_committed_depth",
        "normal_lane_conflict_count",
    ):
        print(f"{key}={summary.get(key)}")


def add_depth4_commit(record: dict[str, Any], *, token_count: int = 8) -> None:
    seq_id = 7
    p0 = 900000100
    p3 = 900000103
    p4 = 900000104
    record["enable_rolling_continuous_depth4_commit_ready_only"] = True
    record["rolling_depth4_commit_enabled"] = True
    record["rolling_depth4_commit_source"] = "rolling_depth4_ready_only"
    record["rolling_depth4_commit_side"] = record.get("rolling_depth4_commit_side") or record.get("rolling_depth3_commit_side") or "target"
    record["rolling_depth4_commit_plan_id"] = 1
    record["rolling_depth4_commit_step_id"] = 2
    record["rolling_depth4_commit_candidate_proposal_ids"] = [p4]
    record["rolling_depth4_commit_candidate_seq_ids"] = [seq_id]
    record["rolling_depth4_commit_ready_source_proposal_ids"] = [p4]
    record["rolling_depth4_commit_parent_by_proposal_id"] = {str(p4): p3}
    record["rolling_depth4_commit_precondition_ok_by_proposal_id"] = {str(p4): True}
    record["rolling_depth4_commit_precondition_failed_by_proposal_id"] = {str(p4): False}
    record["rolling_depth4_commit_precondition_failure_reason_by_proposal_id"] = {}
    record["rolling_depth4_real_committed_proposal_ids"] = [p4]
    record["rolling_depth4_real_committed_seq_ids"] = [seq_id]
    record["rolling_depth4_real_committed_token_count_by_proposal_id"] = {str(p4): token_count}
    record["rolling_depth4_real_committed_accept_len_by_proposal_id"] = {str(p4): token_count}
    record["rolling_depth4_real_commit_action_by_proposal_id"] = {str(p4): GENERIC_ROLLING_ACTION}
    record["rolling_depth4_real_commit_verify_result_by_proposal_id"] = {str(p4): "full_accept"}
    record["rolling_depth4_real_commit_parent_by_proposal_id"] = {str(p4): p3}
    record["rolling_depth4_real_commit_root_by_proposal_id"] = {str(p4): p0}
    record["rolling_depth4_real_commit_depth_by_proposal_id"] = {str(p4): 4}
    record["rolling_depth4_target_seq_len_before_by_seq_id"] = {str(seq_id): 36}
    record["rolling_depth4_target_seq_len_after_by_seq_id"] = {str(seq_id): 44}
    record["rolling_depth4_draft_seq_len_before_by_seq_id"] = {str(seq_id): 36}
    record["rolling_depth4_draft_seq_len_after_by_seq_id"] = {str(seq_id): 44}
    record["rolling_depth4_target_draft_len_match_by_seq_id"] = {str(seq_id): True}
    record["rolling_depth4_target_draft_token_match_by_seq_id"] = {str(seq_id): True}
    record["rolling_depth4_tokens_verified"] = token_count
    record["rolling_depth4_tokens_accepted"] = token_count
    record["rolling_depth4_tokens_committed"] = token_count
    record["rolling_depth4_tokens_rejected"] = 0
    record["rolling_depth4_tokens_invalidated"] = 0
    record["rolling_depth4_real_committed_proposal_count"] = 1
    record["rolling_depth4_real_committed_token_count"] = token_count
    record["rolling_depth4_real_commit_count"] = 1
    record["rolling_depth4_real_commit_skip_reason_counts"] = {}
    record["rolling_depth_gt4_real_commit_count"] = 0


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
    records = synthetic_records()
    shadow_only = deepcopy(records)
    for record in shadow_only:
        add_depth4_shadow(record)
    assert_pass(
        "depth4 shadow only",
        shadow_only,
        {
            "depth4_shadow_enabled": True,
            "depth4_commit_enabled": False,
            "rolling_depth4_real_committed_token_count": 0,
            "combined_real_committed_token_count": 36,
            "max_observed_depth": 4,
            "max_real_committed_depth": 3,
        },
    )

    good = deepcopy(shadow_only)
    for record in good:
        add_depth4_commit(record)
    good[1]["rolling_depth4_commit_side"] = "draft"
    assert_pass(
        "good depth4 commit",
        good,
        {
            "depth4_shadow_enabled": True,
            "depth4_commit_enabled": True,
            "rolling_depth4_real_committed_token_count": 8,
            "combined_real_committed_token_count": 44,
            "max_observed_depth": 4,
            "max_real_committed_depth": 4,
        },
    )

    def _make_partial_recovery_records() -> list[dict[str, Any]]:
        records = deepcopy(shadow_only)
        for record in records:
            add_depth4_commit(record)
        records[1]["rolling_depth4_commit_side"] = "draft"
        return records

    assert_partial_recovery_checker_cases(
        "rolling depth4 commit ready-only",
        _make_partial_recovery_records,
        lambda records: validate_records(records, synthetic_result_payload()),
        expected_full_accept_combined_token_count=44,
    )

    bad_parent = deepcopy(good)
    for record in bad_parent:
        record["rolling_depth4_real_commit_parent_by_proposal_id"] = {"900000104": 12345}
        record["rolling_depth4_child_parent_by_proposal_id"] = {"900000104": 12345}
    assert_fail("bad parent", bad_parent, "parent")

    generated_only = deepcopy(good)
    for record in generated_only:
        record["rolling_depth4_child_ready_shadow_proposal_ids"] = []
    assert_fail("generated-only commit", generated_only, "ready shadow")

    bad_accounting = deepcopy(good)
    bad_accounting[0]["rolling_depth4_tokens_verified"] = 7
    assert_fail("bad accounting", bad_accounting, "must equal depth4 committed tokens")

    bad_higher = deepcopy(good)
    bad_higher[0]["rolling_depth_gt4_real_commit_count"] = 1
    assert_fail("bad higher depth", bad_higher, "depth>4")

    disabled = deepcopy(good)
    for record in disabled:
        record["enable_rolling_continuous_depth4_commit_ready_only"] = False
        record["rolling_depth4_commit_enabled"] = False
    assert_fail("commit disabled", disabled, "commit flag")

    print("Synthetic rolling depth4 commit ready-only checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8p rolling depth4 real commit traces.")
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
