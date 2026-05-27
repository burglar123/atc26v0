#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import validate_records as validate_commit_records
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting


CONTINUOUS_SOURCE = "continuous_shadow"
ONE_SHOT_PARENT_SOURCE = "phase1h6a_one_shot_commit"


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


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def as_int_set(value: Any) -> set[int]:
    return set(as_int_list(value))


def as_int_map(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, int] = {}
    for key, item in value.items():
        try:
            result[int(key)] = int(item)
        except Exception:
            continue
    return result


def as_str_map(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, str] = {}
    for key, item in value.items():
        try:
            result[int(key)] = str(item)
        except Exception:
            continue
    return result


def step_plan_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int_value(record.get("step_id"), int_value(record.get("eager_commit_step_id"), -1)),
        int_value(record.get("plan_id"), int_value(record.get("eager_commit_plan_id"), -1)),
    )


def continuous_row(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("enable_continuous_eager_dry_run", False))
        or bool(record.get("continuous_eager_dry_run_enabled", False))
        or bool(as_int_set(record.get("continuous_eager_candidate_proposal_ids")))
        or bool(as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids")))
        or bool(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    commit_errors, commit_summary = validate_commit_records(records)
    errors.extend(f"one-shot commit checker: {error}" for error in commit_errors)

    one_shot_committed_ids: set[int] = set()
    one_shot_ready_ids: set[int] = set()
    one_shot_committed_ids_by_record: dict[int, set[int]] = {}
    for idx, record in enumerate(records):
        committed = as_int_set(record.get("eager_committed_proposal_ids"))
        ready = as_int_set(record.get("eager_commit_ready_proposal_ids")) | as_int_set(
            record.get("eager_commit_from_readiness_proposal_ids")
        )
        one_shot_committed_ids.update(committed)
        one_shot_ready_ids.update(ready)
        one_shot_committed_ids_by_record[idx] = committed

    records_with_enabled = 0
    active_records = 0
    candidate_ids_seen: set[int] = set()
    candidate_token_by_id: dict[int, int] = {}
    ready_shadow_ids_seen: set[int] = set()
    not_ready_ids_seen: set[int] = set()
    verified_ids_seen: set[int] = set()
    full_accept_ids_seen: set[int] = set()
    duplicate_ids_seen: set[int] = set()
    frontier_mismatch_ids_seen: set[int] = set()
    real_commit_count = 0
    mutation_detected_count = 0
    missing_unexpected_count = 0
    chain_depths: dict[int, int] = {}
    drop_reason_by_id: dict[int, str] = {}
    step_seq_depth_seen: dict[tuple[int, int, int], set[int]] = defaultdict(set)

    for idx, record in enumerate(records):
        enabled = bool(record.get("enable_continuous_eager_dry_run", False))
        if enabled:
            records_with_enabled += 1
        if not continuous_row(record):
            continue
        if not enabled:
            errors.append(f"record[{idx}] has continuous eager fields while flag is disabled")
        active = bool(record.get("continuous_eager_dry_run_enabled", False))
        if active:
            active_records += 1
        if active and record.get("continuous_eager_source") != CONTINUOUS_SOURCE:
            errors.append(f"record[{idx}] continuous source is not {CONTINUOUS_SOURCE!r}")

        candidate_ids = as_int_list(record.get("continuous_eager_candidate_proposal_ids"))
        candidate_seq_ids = as_int_list(record.get("continuous_eager_candidate_seq_ids"))
        candidate_id_set = set(candidate_ids)
        not_ready_ids = as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids"))
        ready_shadow_ids = as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids"))
        verified_ids = as_int_set(record.get("continuous_eager_verified_proposal_ids"))
        full_accept_ids = as_int_set(record.get("continuous_eager_full_accept_proposal_ids"))
        token_by_id = as_int_map(record.get("continuous_eager_candidate_token_count_by_proposal_id"))
        parent_by_id = as_int_map(record.get("continuous_eager_parent_proposal_id_by_proposal_id"))
        root_by_id = as_int_map(record.get("continuous_eager_root_proposal_id_by_proposal_id"))
        depth_by_id = as_int_map(record.get("continuous_eager_chain_depth_by_proposal_id"))
        reason_by_id = as_str_map(record.get("continuous_eager_not_ready_shadow_reason_by_proposal_id"))
        parent_source_by_id = as_str_map(record.get("continuous_eager_parent_source_by_proposal_id"))
        max_depth = int_value(record.get("max_continuous_eager_chain_depth"), 0)
        step, _plan = step_plan_key(record)

        if len(candidate_ids) != len(candidate_seq_ids):
            errors.append(f"record[{idx}] candidate proposal/seq length mismatch")
        for proposal_id in candidate_ids:
            if proposal_id not in token_by_id or int(token_by_id.get(proposal_id, 0)) <= 0:
                errors.append(f"record[{idx}] candidate {proposal_id} missing positive token count")
            if proposal_id not in parent_by_id:
                errors.append(f"record[{idx}] candidate {proposal_id} missing parent proposal id")
            if proposal_id not in root_by_id:
                errors.append(f"record[{idx}] candidate {proposal_id} missing root proposal id")
            depth = int(depth_by_id.get(proposal_id, 0))
            if depth <= 0:
                errors.append(f"record[{idx}] candidate {proposal_id} missing positive chain depth")
            if max_depth > 0 and depth > max_depth:
                errors.append(f"record[{idx}] candidate {proposal_id} exceeds max chain depth")
            parent_id = int(parent_by_id.get(proposal_id, -1))
            if depth == 1 and parent_id not in one_shot_committed_ids and parent_id not in one_shot_ready_ids:
                errors.append(
                    f"record[{idx}] candidate {proposal_id} parent {parent_id} is not one-shot committed/ready"
                )

        for proposal_id, seq_id in zip(candidate_ids, candidate_seq_ids):
            depth = int(depth_by_id.get(proposal_id, 0))
            event_key = (step, int(seq_id), depth)
            step_seq_depth_seen[event_key].add(int(proposal_id))

        for proposal_id in not_ready_ids:
            if not reason_by_id.get(proposal_id):
                errors.append(f"record[{idx}] not-ready continuous proposal {proposal_id} lacks reason")
        if ready_shadow_ids - candidate_id_set:
            errors.append(
                f"record[{idx}] shadow-ready ids are not candidates: {sorted(ready_shadow_ids - candidate_id_set)}"
            )
        if full_accept_ids - verified_ids:
            errors.append(
                f"record[{idx}] full-accept ids are not verified: {sorted(full_accept_ids - verified_ids)}"
            )
        if full_accept_ids - ready_shadow_ids:
            errors.append(
                f"record[{idx}] full-accept ids are not shadow-ready: {sorted(full_accept_ids - ready_shadow_ids)}"
            )
        continuous_ids = (
            candidate_id_set
            | not_ready_ids
            | ready_shadow_ids
            | verified_ids
            | full_accept_ids
        )
        if continuous_ids & one_shot_committed_ids_by_record.get(idx, set()):
            errors.append(f"record[{idx}] continuous ids were real committed in one-shot field")
        if continuous_ids & as_int_set(record.get("lane_exclusion_applied_proposal_ids")):
            errors.append(f"record[{idx}] continuous ids affected lane exclusion")
        if continuous_ids & as_int_set(record.get("target_eager_verify_proposal_ids_dry_run")):
            errors.append(f"record[{idx}] continuous ids entered target takeover lane")
        if int_value(record.get("continuous_eager_real_commit_count"), 0) != 0:
            errors.append(f"record[{idx}] continuous eager real commit count is nonzero")
        if int_value(record.get("missing_buffered_proposal_unexpected_count"), 0) != 0:
            errors.append(f"record[{idx}] aggregate missing buffered proposal unexpected count is nonzero")
        if record.get("missing_buffered_proposal_unexpected_seq_ids"):
            errors.append(f"record[{idx}] unexpected missing buffered proposals present")

        candidate_ids_seen.update(candidate_id_set)
        ready_shadow_ids_seen.update(ready_shadow_ids)
        not_ready_ids_seen.update(not_ready_ids)
        verified_ids_seen.update(verified_ids)
        full_accept_ids_seen.update(full_accept_ids)
        duplicate_ids_seen.update(as_int_set(record.get("continuous_eager_duplicate_proposal_ids")))
        frontier_mismatch_ids_seen.update(
            as_int_set(record.get("continuous_eager_frontier_mismatch_proposal_ids"))
        )
        real_commit_count += int_value(record.get("continuous_eager_real_commit_count"), 0)
        mutation_detected_count += int_value(record.get("continuous_eager_mutation_detected_count"), 0)
        missing_unexpected_count += len(as_int_list(record.get("missing_buffered_proposal_unexpected_seq_ids")))
        for proposal_id, token_count in token_by_id.items():
            if proposal_id in candidate_id_set and token_count > 0:
                candidate_token_by_id.setdefault(proposal_id, token_count)
        for proposal_id, depth in depth_by_id.items():
            if proposal_id in candidate_id_set and depth > 0:
                chain_depths.setdefault(proposal_id, depth)
        for proposal_id, reason in reason_by_id.items():
            drop_reason_by_id.setdefault(proposal_id, reason)
        for proposal_id, source in parent_source_by_id.items():
            if proposal_id in candidate_id_set and source not in {
                ONE_SHOT_PARENT_SOURCE,
                CONTINUOUS_SOURCE,
            }:
                errors.append(f"record[{idx}] continuous proposal {proposal_id} has bad parent source {source!r}")

    repeated_seq_depth = {
        key: sorted(proposal_ids)
        for key, proposal_ids in step_seq_depth_seen.items()
        if len(proposal_ids) > 1
    }
    if repeated_seq_depth:
        errors.append(f"duplicate continuous candidates for same step/seq/depth: {repeated_seq_depth}")

    accounting = aggregate_performance_accounting(records, {})
    candidate_tokens = sum(candidate_token_by_id.values())
    ready_shadow_tokens = int_value(accounting.get("continuous_eager_commit_ready_shadow_token_count"), 0)
    chain_distribution = Counter(str(depth) for depth in chain_depths.values())
    summary = {
        "total_trace_records": len(records),
        "records_with_continuous_eager_enabled": records_with_enabled,
        "continuous_active_records": active_records,
        "one_shot_committed_proposal_count": int_value(commit_summary.get("committed_proposal_count"), 0),
        "one_shot_committed_token_count": int_value(commit_summary.get("committed_token_count"), 0),
        "continuous_candidate_proposal_count": len(candidate_ids_seen),
        "continuous_candidate_token_count": candidate_tokens,
        "continuous_verified_proposal_count": len(verified_ids_seen),
        "continuous_full_accept_proposal_count": len(full_accept_ids_seen),
        "continuous_commit_ready_shadow_proposal_count": len(ready_shadow_ids_seen),
        "continuous_commit_ready_shadow_token_count": ready_shadow_tokens,
        "continuous_not_ready_shadow_proposal_count": len(not_ready_ids_seen),
        "continuous_chain_length_distribution": dict(chain_distribution),
        "continuous_drop_reason_counts": dict(Counter(drop_reason_by_id.values())),
        "continuous_duplicate_count": len(duplicate_ids_seen),
        "continuous_frontier_mismatch_count": len(frontier_mismatch_ids_seen),
        "continuous_mutation_detected_count": mutation_detected_count,
        "continuous_real_commit_count": real_commit_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "combined_one_shot_plus_continuous_shadow_token_count": accounting.get(
            "combined_one_shot_plus_continuous_shadow_token_count",
            0,
        ),
        "combined_estimated_token_share_of_output": accounting.get(
            "combined_estimated_token_share_of_output",
            0.0,
        ),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_continuous_eager_enabled",
        "continuous_active_records",
        "one_shot_committed_proposal_count",
        "one_shot_committed_token_count",
        "continuous_candidate_proposal_count",
        "continuous_candidate_token_count",
        "continuous_verified_proposal_count",
        "continuous_full_accept_proposal_count",
        "continuous_commit_ready_shadow_proposal_count",
        "continuous_commit_ready_shadow_token_count",
        "continuous_not_ready_shadow_proposal_count",
        "continuous_chain_length_distribution",
        "continuous_drop_reason_counts",
        "continuous_duplicate_count",
        "continuous_frontier_mismatch_count",
        "continuous_mutation_detected_count",
        "continuous_real_commit_count",
        "missing_buffered_proposal_unexpected_count",
        "combined_one_shot_plus_continuous_shadow_token_count",
        "combined_estimated_token_share_of_output",
    ):
        print(f"{key}={summary.get(key)}")


def synthetic_base_record() -> dict[str, Any]:
    return {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_commit_readiness_dry_run": True,
        "enable_eager_commit_ready_only": True,
        "eager_commit_enabled": True,
        "eager_commit_source": "phase1h5e3_takeover_lane",
        "eager_commit_side": "target",
        "eager_commit_step_id": 7,
        "eager_commit_plan_id": 27,
        "eager_commit_candidate_proposal_ids": [6],
        "eager_commit_ready_proposal_ids": [6],
        "eager_commit_from_readiness_proposal_ids": [6],
        "eager_committed_proposal_ids": [6],
        "eager_committed_seq_ids": [12],
        "eager_committed_token_count_by_proposal_id": {"6": 4},
        "eager_committed_accept_len_by_proposal_id": {"6": 4},
        "eager_committed_action_by_proposal_id": {"6": "append_full_accept_then_rollback"},
        "eager_committed_verify_result_by_proposal_id": {"6": "full_accept"},
        "eager_commit_precondition_ok_by_proposal_id": {"6": True},
        "eager_commit_precondition_failed_by_proposal_id": {"6": False},
        "eager_commit_target_seq_len_before_by_seq_id": {"12": 20},
        "eager_commit_target_seq_len_after_by_seq_id": {"12": 24},
        "eager_commit_draft_seq_len_before_by_seq_id": {"12": 20},
        "eager_commit_draft_seq_len_after_by_seq_id": {"12": 24},
        "eager_commit_target_draft_len_match_by_seq_id": {"12": True},
        "eager_commit_target_draft_token_match_by_seq_id": {"12": True},
        "eager_commit_candidate_count": 1,
        "eager_commit_committed_count": 1,
        "eager_commit_skipped_count": 0,
        "eager_tokens_committed": 4,
        "eager_tokens_committed_full_accept": 4,
        "eager_tokens_verified": 4,
        "eager_tokens_accepted": 4,
        "eager_tokens_rejected": 0,
        "eager_tokens_invalidated": 0,
        "step_id": 7,
        "plan_id": 27,
    }


def synthetic_continuous_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_continuous_eager_dry_run": True,
            "continuous_eager_dry_run_enabled": True,
            "continuous_eager_source": CONTINUOUS_SOURCE,
            "continuous_eager_parent_source": ONE_SHOT_PARENT_SOURCE,
            "max_continuous_eager_chain_depth": 2,
            "continuous_eager_candidate_proposal_ids": [900000601],
            "continuous_eager_candidate_seq_ids": [12],
            "continuous_eager_candidate_token_count_by_proposal_id": {"900000601": 4},
            "continuous_eager_parent_proposal_id_by_proposal_id": {
                "900000601": 6,
                "900000602": 900000601,
            },
            "continuous_eager_chain_depth_by_proposal_id": {
                "900000601": 1,
                "900000602": 2,
            },
            "continuous_eager_root_proposal_id_by_proposal_id": {
                "900000601": 6,
                "900000602": 6,
            },
            "continuous_eager_parent_source_by_proposal_id": {
                "900000601": ONE_SHOT_PARENT_SOURCE,
                "900000602": CONTINUOUS_SOURCE,
            },
            "continuous_eager_not_ready_shadow_proposal_ids": [900000601, 900000602],
            "continuous_eager_not_ready_shadow_reason_by_proposal_id": {
                "900000601": "shadow_verify_not_executed",
                "900000602": "parent_shadow_not_committed",
            },
            "continuous_eager_frontier_mismatch_proposal_ids": [900000602],
            "continuous_eager_candidate_proposal_count": 1,
            "continuous_eager_candidate_token_count": 4,
            "continuous_eager_commit_ready_shadow_proposal_count": 0,
            "continuous_eager_commit_ready_shadow_token_count": 0,
            "continuous_eager_real_commit_count": 0,
            "continuous_eager_mutation_detected_count": 0,
        }
    )
    return record


def run_synthetic_tests() -> None:
    draft_record = synthetic_base_record()
    draft_record["eager_commit_side"] = "draft"
    valid = [synthetic_continuous_record(), draft_record]
    errors, summary = validate_records(valid)
    assert not errors, f"valid continuous synthetic failed: {errors}"
    assert summary["continuous_candidate_token_count"] == 4
    assert summary["continuous_real_commit_count"] == 0
    assert summary["continuous_drop_reason_counts"]["shadow_verify_not_executed"] == 1

    invalid = deepcopy(valid)
    invalid[0]["enable_continuous_eager_dry_run"] = False
    errors, _summary = validate_records(invalid)
    assert any("flag is disabled" in error for error in errors), "missed disabled-flag continuous fields"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_real_commit_count"] = 1
    errors, _summary = validate_records(invalid)
    assert any("real commit count is nonzero" in error for error in errors), "missed real continuous commit"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = {}
    errors, _summary = validate_records(invalid)
    assert any("lacks reason" in error for error in errors), "missed missing not-ready reason"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_candidate_proposal_ids"] = [900000601, 900000699]
    invalid[0]["continuous_eager_candidate_seq_ids"] = [12, 12]
    invalid[0]["continuous_eager_candidate_token_count_by_proposal_id"]["900000699"] = 4
    invalid[0]["continuous_eager_parent_proposal_id_by_proposal_id"]["900000699"] = 6
    invalid[0]["continuous_eager_root_proposal_id_by_proposal_id"]["900000699"] = 6
    invalid[0]["continuous_eager_chain_depth_by_proposal_id"]["900000699"] = 1
    invalid[0]["continuous_eager_parent_source_by_proposal_id"]["900000699"] = ONE_SHOT_PARENT_SOURCE
    errors, _summary = validate_records(invalid)
    assert any("duplicate continuous candidates" in error for error in errors), "missed duplicate seq/depth"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_chain_depth_by_proposal_id"]["900000601"] = 3
    errors, _summary = validate_records(invalid)
    assert any("exceeds max chain depth" in error for error in errors), "missed excessive chain depth"

    print("Synthetic continuous eager dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-7a continuous eager shadow dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    errors, summary = validate_records(load_trace(args.trace))
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Continuous eager dry-run checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
