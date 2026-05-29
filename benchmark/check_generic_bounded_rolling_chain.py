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
    GENERIC_ONE_SHOT_ACTION,
    GENERIC_ROLLING_ACTION,
    MAX_LEGACY_REAL_DEPTH,
    RollingChainRegistry,
    committed_token_counts_by_depth,
    int_value,
    parse_legacy_rolling_chain,
    registry_safety_issue_sets,
    summarize_registry,
)
from benchmark.check_bounded_rolling_readiness_audit import (  # noqa: E402
    synthetic_full_chain_record,
    synthetic_result_payload,
)
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    load_json,
)


ONE_SHOT_ACTION = GENERIC_ONE_SHOT_ACTION
ROLLING_ACTION = GENERIC_ROLLING_ACTION

ACCOUNTING_TOKEN_KEYS = {
    0: "eager_committed_token_count",
    1: "continuous_eager_real_committed_token_count",
    2: "rolling_depth2_real_committed_token_count",
    3: "rolling_depth3_real_committed_token_count",
}

DEPTH_INCREMENT_SPECS = {
    0: {
        "label": "one-shot",
        "target_verified": "target_actual_eager_verified_token_increment_sum",
        "target_accepted": "target_actual_eager_accepted_token_increment_sum",
        "target_rejected": "target_actual_eager_rejected_token_increment_sum",
        "target_invalidated": "target_actual_eager_invalidated_token_increment_sum",
        "draft_verified": "draft_actual_eager_verified_token_increment_sum",
        "draft_accepted": "draft_actual_eager_accepted_token_increment_sum",
        "draft_rejected": None,
        "draft_invalidated": None,
    },
    1: {
        "label": "depth1",
        "target_verified": "continuous_target_actual_verified_token_increment_sum",
        "target_accepted": "continuous_target_actual_accepted_token_increment_sum",
        "target_rejected": "continuous_target_actual_rejected_token_increment_sum",
        "target_invalidated": "continuous_target_actual_invalidated_token_increment_sum",
        "draft_verified": "continuous_draft_actual_verified_token_increment_sum",
        "draft_accepted": "continuous_draft_actual_accepted_token_increment_sum",
        "draft_rejected": "continuous_draft_actual_rejected_token_increment_sum",
        "draft_invalidated": "continuous_draft_actual_invalidated_token_increment_sum",
    },
    2: {
        "label": "depth2",
        "target_verified": "rolling_depth2_target_actual_verified_token_increment_sum",
        "target_accepted": "rolling_depth2_target_actual_accepted_token_increment_sum",
        "target_rejected": "rolling_depth2_target_actual_rejected_token_increment_sum",
        "target_invalidated": "rolling_depth2_target_actual_invalidated_token_increment_sum",
        "draft_verified": "rolling_depth2_draft_actual_verified_token_increment_sum",
        "draft_accepted": "rolling_depth2_draft_actual_accepted_token_increment_sum",
        "draft_rejected": "rolling_depth2_draft_actual_rejected_token_increment_sum",
        "draft_invalidated": "rolling_depth2_draft_actual_invalidated_token_increment_sum",
    },
    3: {
        "label": "depth3",
        "target_verified": "rolling_depth3_target_actual_verified_token_increment_sum",
        "target_accepted": "rolling_depth3_target_actual_accepted_token_increment_sum",
        "target_rejected": "rolling_depth3_target_actual_rejected_token_increment_sum",
        "target_invalidated": "rolling_depth3_target_actual_invalidated_token_increment_sum",
        "draft_verified": "rolling_depth3_draft_actual_verified_token_increment_sum",
        "draft_accepted": "rolling_depth3_draft_actual_accepted_token_increment_sum",
        "draft_rejected": "rolling_depth3_draft_actual_rejected_token_increment_sum",
        "draft_invalidated": "rolling_depth3_draft_actual_invalidated_token_increment_sum",
    },
}


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records", "events", "iterations", "batches"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise SystemExit(f"{path} does not contain trace records")


def validate_accounting(
    registry: RollingChainRegistry,
    accounting: dict[str, Any],
    errors: list[str],
) -> tuple[dict[int, int], bool, bool]:
    token_by_depth = committed_token_counts_by_depth(registry, MAX_LEGACY_REAL_DEPTH)
    combined_from_registry = sum(token_by_depth.values())
    combined_ok = True
    target_draft_ok = True

    for depth, key in ACCOUNTING_TOKEN_KEYS.items():
        accounting_tokens = int_value(accounting.get(key), 0)
        if registry.committed_by_depth[depth] or accounting_tokens:
            if accounting_tokens != token_by_depth[depth]:
                combined_ok = False
                errors.append(
                    f"depth {depth} committed token mismatch: "
                    f"generic={token_by_depth[depth]} accounting={accounting_tokens}"
                )

    accounting_combined = int_value(accounting.get("combined_real_committed_token_count"), 0)
    accounting_verified = int_value(accounting.get("combined_actual_verified_token_increment_sum"), 0)
    accounting_accepted = int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0)
    if accounting_combined != combined_from_registry:
        combined_ok = False
        errors.append(
            f"combined real committed token mismatch: "
            f"generic={combined_from_registry} accounting={accounting_combined}"
        )
    if accounting_verified != combined_from_registry:
        combined_ok = False
        errors.append(
            f"combined actual verified token mismatch: "
            f"generic={combined_from_registry} accounting={accounting_verified}"
        )
    if accounting_accepted != combined_from_registry:
        combined_ok = False
        errors.append(
            f"combined actual accepted token mismatch: "
            f"generic={combined_from_registry} accounting={accounting_accepted}"
        )

    for depth, spec in DEPTH_INCREMENT_SPECS.items():
        tokens = token_by_depth[depth]
        label = str(spec["label"])
        for key_name in ("target_verified", "target_accepted", "draft_verified", "draft_accepted"):
            key = spec[key_name]
            if key is None:
                continue
            value = int_value(accounting.get(key), 0)
            if value != tokens:
                target_draft_ok = False
                errors.append(f"{label} {key_name.replace('_', ' ')} increment mismatch")
        for key_name in ("target_rejected", "target_invalidated", "draft_rejected", "draft_invalidated"):
            key = spec[key_name]
            if key is None:
                continue
            value = int_value(accounting.get(key), 0)
            if value != 0:
                target_draft_ok = False
                errors.append(f"{label} {key_name.replace('_', ' ')} increment must be zero")

    if registry.target_draft_length_mismatch_count:
        target_draft_ok = False
        errors.append("target/draft length mismatch evidence present")
    if registry.target_draft_token_mismatch_count:
        target_draft_ok = False
        errors.append("target/draft token mismatch evidence present")

    return token_by_depth, combined_ok, target_draft_ok


def validate_chain(registry: RollingChainRegistry, errors: list[str]) -> dict[str, int]:
    issue_sets = registry_safety_issue_sets(registry, max_depth=MAX_LEGACY_REAL_DEPTH)
    invalid_committed_ids = set(issue_sets["invalid_committed_ids"])
    cascade_committed_ids = set(issue_sets["cascade_committed_ids"])
    parent_missing_ids = set(issue_sets["parent_missing_ids"])
    generated_only_ids = set(issue_sets["generated_only_ids"])
    non_full_ids = set(issue_sets["non_full_ids"])
    stale_committed_ids = set(issue_sets["stale_committed_ids"])

    for depth in range(0, MAX_LEGACY_REAL_DEPTH + 1):
        expected_action = ROLLING_ACTION if depth >= 2 else ONE_SHOT_ACTION
        for proposal_id in sorted(registry.committed_by_depth[depth]):
            node = registry.nodes_by_id.get(proposal_id)
            if node is None:
                errors.append(f"committed proposal {proposal_id} missing generic node")
                continue
            declared_depth = registry.declared_depth_by_id.get(proposal_id, node.depth)
            if declared_depth != depth or node.depth != depth:
                errors.append(
                    f"committed proposal {proposal_id} depth mismatch: "
                    f"node={node.depth} declared={declared_depth} expected={depth}"
                )
            if node.verify_result is not None and node.verify_result != "full_accept":
                non_full_ids.add(proposal_id)
            if node.action is not None and node.action != expected_action:
                non_full_ids.add(proposal_id)
            token_count = max(0, int(node.token_count))
            accept_len = node.accept_len if node.accept_len is not None else token_count
            if token_count <= 0 or accept_len != token_count:
                errors.append(f"committed proposal {proposal_id} depth {depth} token/accept mismatch")
            if depth == 0:
                continue
            if node.parent_id is None:
                continue
            parent = registry.nodes_by_id.get(node.parent_id)
            if parent is None:
                continue
            if parent.depth != depth - 1:
                errors.append(f"proposal {proposal_id} parent {node.parent_id} depth {parent.depth}, expected {depth - 1}")
            if parent.seq_id is not None and node.seq_id is not None and parent.seq_id != node.seq_id:
                errors.append(f"proposal {proposal_id} seq {node.seq_id} differs from parent {node.parent_id} seq {parent.seq_id}")
            parent_root = parent.root_id if parent.root_id is not None else parent.proposal_id
            if node.root_id is not None and parent_root is not None and node.root_id != parent_root:
                errors.append(f"proposal {proposal_id} root {node.root_id} differs from parent {node.parent_id} root {parent_root}")

    if registry.higher_depth_commit_ids:
        errors.append(f"depth>3 committed proposal ids present: {sorted(registry.higher_depth_commit_ids)}")
    if registry.depth4_real_commit_count:
        errors.append("rolling depth4 real commit count must be zero")
    if registry.depth_gt3_real_commit_count:
        errors.append("rolling depth>3 real commit count must be zero")
    if registry.normal_lane_conflict_count:
        errors.append("normal lane conflict evidence present")
    if registry.missing_buffered_proposal_unexpected_count:
        errors.append("unexpected missing buffered proposal evidence present")
    if registry.committed_by_depth[3] and not registry.flags.get("rolling_depth3_commit_enabled", False):
        errors.append("depth3 real commit appears while depth3 commit flag is disabled")

    duplicate_commit_count = int(issue_sets["duplicate_commit_count"])
    if duplicate_commit_count:
        errors.append("duplicate commit evidence present")
    if invalid_committed_ids:
        errors.append(f"invalidated committed proposal ids: {sorted(invalid_committed_ids)}")
    if cascade_committed_ids:
        errors.append(f"cascade-discarded committed proposal ids: {sorted(cascade_committed_ids)}")
    if stale_committed_ids:
        errors.append(f"stale/expired/frontier-mismatched committed proposal ids: {sorted(stale_committed_ids)}")
    if parent_missing_ids:
        errors.append(f"parent-missing committed proposal ids: {sorted(parent_missing_ids)}")
    if generated_only_ids:
        errors.append(f"committed proposals without ready shadow: {sorted(generated_only_ids)}")
    if non_full_ids:
        errors.append(f"non-full-accept committed proposal ids: {sorted(non_full_ids)}")

    return {
        "duplicate_commit_count": duplicate_commit_count,
        "invalid_committed_child_count": len(invalid_committed_ids),
        "cascade_committed_child_count": len(cascade_committed_ids),
        "parent_missing_committed_child_count": len(parent_missing_ids),
        "generated_only_committed_child_count": len(generated_only_ids),
        "non_full_accept_committed_count": len(non_full_ids),
        "stale_or_frontier_committed_child_count": len(stale_committed_ids),
    }


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    *,
    strict_performance: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    result_payload = result_payload or {}
    registry = parse_legacy_rolling_chain(records)
    accounting = aggregate_performance_accounting(records, result_payload)
    result_args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if isinstance(result_args, dict):
        registry.max_configured_depth = max(
            registry.max_configured_depth,
            int_value(result_args.get("max_rolling_continuous_depth"), 0),
        )

    token_by_depth, combined_ok, target_draft_ok = validate_accounting(registry, accounting, errors)
    chain_summary = validate_chain(registry, errors)
    generic_summary = summarize_registry(registry, accounting=accounting)

    max_real_depth = int(generic_summary["generic_max_real_committed_depth"])
    if max_real_depth > MAX_LEGACY_REAL_DEPTH:
        errors.append(f"max real committed depth {max_real_depth} exceeds {MAX_LEGACY_REAL_DEPTH}")

    performance_warnings = accounting.get("performance_warnings", [])
    if strict_performance and performance_warnings:
        errors.append(f"performance warnings present under --strict-performance: {performance_warnings}")

    metrics = result_payload.get("metrics", {}) if isinstance(result_payload, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    if not isinstance(overall, dict):
        overall = {}

    summary = {
        "total_trace_records": len(records),
        **registry.flags,
        "max_configured_depth": registry.max_configured_depth,
        "max_observed_depth": generic_summary["generic_max_observed_depth"],
        "max_real_committed_depth": max_real_depth,
        "depth_gt3_real_commit_count": generic_summary["generic_depth_gt3_real_commit_count"],
        "depth4_real_commit_count": generic_summary["generic_depth4_real_commit_count"],
        "depth_gt3_committed_proposal_count": generic_summary["generic_depth_gt3_committed_proposal_count"],
        "one_shot_committed_proposal_count": generic_summary["generic_one_shot_committed_proposal_count"],
        "one_shot_committed_token_count": token_by_depth[0],
        "depth1_committed_proposal_count": generic_summary["generic_depth1_committed_proposal_count"],
        "depth1_committed_token_count": token_by_depth[1],
        "depth2_committed_proposal_count": generic_summary["generic_depth2_committed_proposal_count"],
        "depth2_committed_token_count": token_by_depth[2],
        "depth3_committed_proposal_count": generic_summary["generic_depth3_committed_proposal_count"],
        "depth3_committed_token_count": token_by_depth[3],
        "combined_real_committed_token_count": generic_summary["generic_combined_real_committed_token_count"],
        "accounting_combined_real_committed_token_count": int_value(
            accounting.get("combined_real_committed_token_count"), 0
        ),
        "combined_actual_verified_token_increment_sum": int_value(
            accounting.get("combined_actual_verified_token_increment_sum"), 0
        ),
        "combined_actual_accepted_token_increment_sum": int_value(
            accounting.get("combined_actual_accepted_token_increment_sum"), 0
        ),
        "combined_accounting_ok": combined_ok and bool(generic_summary["generic_combined_accounting_ok"]),
        "target_draft_accounting_ok": target_draft_ok and bool(generic_summary["generic_target_draft_accounting_ok"]),
        "normal_lane_conflict_count": generic_summary["generic_normal_lane_conflict_count"],
        "missing_buffered_proposal_unexpected_count": generic_summary[
            "generic_missing_buffered_proposal_unexpected_count"
        ],
        "target_draft_length_mismatch_count": generic_summary["generic_target_draft_length_mismatch_count"],
        "target_draft_token_mismatch_count": generic_summary["generic_target_draft_token_mismatch_count"],
        "total_output_tokens": int_value(overall.get("total_output_tokens"), 0),
        "goodput_tokens_per_s": overall.get("goodput_tokens_per_s"),
        "mean_tpot_ms": overall.get("mean_tpot_ms"),
        "performance_warnings": performance_warnings,
        **chain_summary,
    }
    return errors, summary


def summarize_generic_chain(
    trace_records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
    *,
    strict_performance: bool = False,
) -> dict[str, Any]:
    errors, summary = validate_records(
        trace_records,
        result_payload,
        strict_performance=strict_performance,
    )
    return {
        **summary,
        "generic_error_count": len(errors),
        "generic_errors": errors,
    }


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "one_shot_commit_enabled",
        "continuous_depth1_commit_enabled",
        "rolling_depth2_commit_enabled",
        "rolling_depth3_shadow_enabled",
        "rolling_depth3_commit_enabled",
        "max_configured_depth",
        "max_observed_depth",
        "max_real_committed_depth",
        "depth_gt3_real_commit_count",
        "depth4_real_commit_count",
        "depth_gt3_committed_proposal_count",
        "one_shot_committed_proposal_count",
        "one_shot_committed_token_count",
        "depth1_committed_proposal_count",
        "depth1_committed_token_count",
        "depth2_committed_proposal_count",
        "depth2_committed_token_count",
        "depth3_committed_proposal_count",
        "depth3_committed_token_count",
        "combined_real_committed_token_count",
        "accounting_combined_real_committed_token_count",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "combined_accounting_ok",
        "target_draft_accounting_ok",
        "normal_lane_conflict_count",
        "missing_buffered_proposal_unexpected_count",
        "duplicate_commit_count",
        "invalid_committed_child_count",
        "cascade_committed_child_count",
        "parent_missing_committed_child_count",
        "generated_only_committed_child_count",
        "non_full_accept_committed_count",
        "stale_or_frontier_committed_child_count",
        "target_draft_length_mismatch_count",
        "target_draft_token_mismatch_count",
        "total_output_tokens",
        "goodput_tokens_per_s",
        "mean_tpot_ms",
        "performance_warnings",
    ):
        print(f"{key}={summary.get(key)}")


def retokenize_full_chain(records: list[dict[str, Any]], *, p0: int = 12, p1: int = 8, p2: int = 8, p3: int = 8) -> None:
    proposal_ids = (900000100, 900000101, 900000102, 900000103)
    token_fields = (
        ("eager_committed_token_count_by_proposal_id", p0),
        ("eager_committed_accept_len_by_proposal_id", p0),
        ("continuous_eager_candidate_token_count_by_proposal_id", p1),
        ("continuous_eager_real_committed_token_count_by_proposal_id", p1),
        ("continuous_eager_real_committed_accept_len_by_proposal_id", p1),
        ("rolling_depth2_real_committed_token_count_by_proposal_id", p2),
        ("rolling_depth2_real_committed_accept_len_by_proposal_id", p2),
        ("rolling_depth3_child_token_count_by_proposal_id", p3),
        ("rolling_depth3_real_committed_token_count_by_proposal_id", p3),
        ("rolling_depth3_real_committed_accept_len_by_proposal_id", p3),
    )
    id_by_field = {
        "eager": proposal_ids[0],
        "continuous": proposal_ids[1],
        "rolling_depth2": proposal_ids[2],
        "rolling_depth3": proposal_ids[3],
    }
    for record in records:
        for field, tokens in token_fields:
            if field.startswith("eager"):
                proposal_id = id_by_field["eager"]
            elif field.startswith("continuous"):
                proposal_id = id_by_field["continuous"]
            elif field.startswith("rolling_depth2"):
                proposal_id = id_by_field["rolling_depth2"]
            else:
                proposal_id = id_by_field["rolling_depth3"]
            record[field] = {str(proposal_id): tokens}
        record["eager_tokens_verified"] = p0
        record["eager_tokens_accepted"] = p0
        record["continuous_eager_tokens_verified"] = p1
        record["continuous_eager_tokens_accepted"] = p1
        record["rolling_depth2_tokens_verified"] = p2
        record["rolling_depth2_tokens_accepted"] = p2
        record["rolling_depth3_tokens_verified"] = p3
        record["rolling_depth3_tokens_accepted"] = p3


def synthetic_records() -> list[dict[str, Any]]:
    records = [synthetic_full_chain_record("target"), synthetic_full_chain_record("draft")]
    retokenize_full_chain(records)
    return records


def clear_depth3_commit(record: dict[str, Any]) -> None:
    for field in (
        "enable_rolling_continuous_depth3_commit_ready_only",
        "rolling_depth3_commit_enabled",
    ):
        record[field] = False
    for field in (
        "rolling_depth3_real_committed_proposal_ids",
        "rolling_depth3_real_committed_seq_ids",
    ):
        record[field] = []
    for field in (
        "rolling_depth3_real_committed_token_count_by_proposal_id",
        "rolling_depth3_real_committed_accept_len_by_proposal_id",
        "rolling_depth3_real_commit_action_by_proposal_id",
        "rolling_depth3_real_commit_verify_result_by_proposal_id",
        "rolling_depth3_real_commit_parent_by_proposal_id",
        "rolling_depth3_real_commit_root_by_proposal_id",
        "rolling_depth3_real_commit_depth_by_proposal_id",
    ):
        record[field] = {}
    record["rolling_depth3_tokens_verified"] = 0
    record["rolling_depth3_tokens_accepted"] = 0
    record["rolling_depth3_tokens_rejected"] = 0
    record["rolling_depth3_tokens_invalidated"] = 0
    record["rolling_depth3_real_commit_count"] = 0


def clear_depth2_and_depth3(record: dict[str, Any]) -> None:
    for field in (
        "enable_rolling_continuous_depth2_commit_ready_only",
        "rolling_depth2_commit_enabled",
        "enable_rolling_continuous_depth3_shadow_dry_run",
        "rolling_depth3_shadow_enabled",
    ):
        record[field] = False
    for field in (
        "rolling_child_generated_proposal_ids",
        "rolling_child_candidate_proposal_ids",
        "rolling_child_ready_after_parent_full_accept_proposal_ids",
        "rolling_child_ready_shadow_proposal_ids",
        "rolling_depth2_real_committed_proposal_ids",
        "rolling_depth2_real_committed_seq_ids",
        "rolling_depth3_child_generated_proposal_ids",
        "rolling_depth3_child_ready_shadow_proposal_ids",
        "rolling_depth3_child_ready_shadow_seq_ids",
        "rolling_depth3_real_committed_proposal_ids",
        "rolling_depth3_real_committed_seq_ids",
    ):
        record[field] = []
    for field in (
        "rolling_chain_parent_by_proposal_id",
        "rolling_chain_root_by_proposal_id",
        "rolling_chain_depth_by_proposal_id",
        "rolling_child_parent_by_proposal_id",
        "rolling_child_root_by_proposal_id",
        "rolling_child_depth_by_proposal_id",
        "rolling_child_token_count_by_proposal_id",
        "rolling_depth2_real_committed_token_count_by_proposal_id",
        "rolling_depth2_real_committed_accept_len_by_proposal_id",
        "rolling_depth2_real_commit_action_by_proposal_id",
        "rolling_depth2_real_commit_verify_result_by_proposal_id",
        "rolling_depth2_real_commit_parent_by_proposal_id",
        "rolling_depth2_real_commit_root_by_proposal_id",
        "rolling_depth2_real_commit_depth_by_proposal_id",
        "rolling_depth3_child_parent_by_proposal_id",
        "rolling_depth3_child_root_by_proposal_id",
        "rolling_depth3_child_depth_by_proposal_id",
        "rolling_depth3_child_token_count_by_proposal_id",
        "rolling_depth3_real_committed_token_count_by_proposal_id",
        "rolling_depth3_real_committed_accept_len_by_proposal_id",
        "rolling_depth3_real_commit_action_by_proposal_id",
        "rolling_depth3_real_commit_verify_result_by_proposal_id",
        "rolling_depth3_real_commit_parent_by_proposal_id",
        "rolling_depth3_real_commit_root_by_proposal_id",
        "rolling_depth3_real_commit_depth_by_proposal_id",
    ):
        record[field] = {}
    for field in (
        "rolling_depth2_tokens_verified",
        "rolling_depth2_tokens_accepted",
        "rolling_depth2_tokens_rejected",
        "rolling_depth2_tokens_invalidated",
        "rolling_depth2_real_commit_count",
        "rolling_depth3_tokens_verified",
        "rolling_depth3_tokens_accepted",
        "rolling_depth3_tokens_rejected",
        "rolling_depth3_tokens_invalidated",
        "rolling_depth3_real_commit_count",
        "rolling_depth3_same_seq_overlap_count",
    ):
        record[field] = 0


def assert_synthetic_pass(name: str, records: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic {name} failed: {errors}\nsummary={summary}")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SystemExit(f"synthetic {name} {key} mismatch: expected {value}, got {summary.get(key)}")


def assert_synthetic_fail(name: str, records: list[dict[str, Any]], needle: str) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if not errors:
        raise SystemExit(f"synthetic {name} should fail\nsummary={summary}")
    if not any(needle in error for error in errors):
        raise SystemExit(f"synthetic {name} failed for wrong reason: {errors}\nsummary={summary}")


def run_synthetic() -> None:
    records = synthetic_records()
    assert_synthetic_pass(
        "good N=3",
        records,
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
        },
    )

    lower_depth = deepcopy(records)
    for record in lower_depth:
        clear_depth2_and_depth3(record)
    assert_synthetic_pass(
        "lower depth",
        lower_depth,
        {
            "one_shot_committed_token_count": 12,
            "depth1_committed_token_count": 8,
            "depth2_committed_token_count": 0,
            "depth3_committed_token_count": 0,
            "combined_real_committed_token_count": 20,
            "max_real_committed_depth": 1,
            "combined_accounting_ok": True,
            "target_draft_accounting_ok": True,
        },
    )

    shadow_only = deepcopy(records)
    for record in shadow_only:
        clear_depth3_commit(record)
    assert_synthetic_pass(
        "depth3 shadow only",
        shadow_only,
        {
            "depth3_committed_token_count": 0,
            "combined_real_committed_token_count": 28,
            "max_observed_depth": 3,
            "max_real_committed_depth": 2,
            "combined_accounting_ok": True,
            "target_draft_accounting_ok": True,
        },
    )

    bad_parent = deepcopy(records)
    for record in bad_parent:
        record["rolling_depth3_child_parent_by_proposal_id"] = {"900000103": 12345}
        record["rolling_depth3_real_commit_parent_by_proposal_id"] = {"900000103": 12345}
    assert_synthetic_fail("bad parent", bad_parent, "parent-missing")

    bad_higher_depth = deepcopy(records)
    bad_higher_depth[0]["rolling_depth4_real_commit_count"] = 1
    bad_higher_depth[0]["rolling_depth_gt3_real_commit_count"] = 1
    assert_synthetic_fail("bad higher depth", bad_higher_depth, "depth4")

    bad_accounting = deepcopy(records)
    bad_accounting[0]["rolling_depth3_tokens_verified"] = 7
    assert_synthetic_fail("bad accounting", bad_accounting, "depth3 target verified")

    duplicate_side_rows = deepcopy(records)
    duplicate_side_rows.append(deepcopy(records[0]))
    duplicate_side_rows.append(deepcopy(records[1]))
    assert_synthetic_pass(
        "duplicate side rows",
        duplicate_side_rows,
        {
            "combined_real_committed_token_count": 36,
            "duplicate_commit_count": 0,
            "combined_accounting_ok": True,
            "target_draft_accounting_ok": True,
        },
    )

    print("Synthetic generic bounded rolling chain checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate bounded N=3 rolling chains through the generic legacy parser.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--strict-performance", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic()
        return 0

    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result else {}
    errors, summary = validate_records(records, result_payload, strict_performance=args.strict_performance)
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
