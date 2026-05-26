#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]

FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"
PARTIAL_ACTION = "discard_partial_no_mutation"
REJECT_ACTION = "discard_reject_no_mutation"

VALID_SKIP_REASONS = {
    "missing_proposal_id",
    "missing_local_seq",
    "invalid_proposal_len",
    "invalid_to_verify_len",
    "invalid_base_pre_verify",
    "invalid_result_metadata",
    "seq_finished_before_sync_apply_dry_run",
    "span_invalidated_before_sync_apply_dry_run",
    "seq_pre_verify_before_sync_apply_dry_run",
    "draft_base_not_reached",
    "draft_base_ahead_requires_real_lane_exclusion",
    "unexpected_draft_base_overshot",
}


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError(
        "Unsupported trace format. Expected a raw list or a dict with "
        "'traces', 'records', or 'trace_records'."
    )


def is_dual_record(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    )


def as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(item) for item in value}


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def dict_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def proposal_seq_map(record: dict[str, Any]) -> dict[int, int]:
    proposal_ids = (
        as_int_list(record.get("scheduled_target_eager_proposal_ids_dry_run"))
        or as_int_list(record.get("eager_scheduled_proposal_ids"))
        or as_int_list(record.get("eager_result_received_proposal_ids"))
        or as_int_list(record.get("eager_result_sent_proposal_ids"))
    )
    seq_ids = (
        as_int_list(record.get("scheduled_target_eager_seq_ids_dry_run"))
        or as_int_list(record.get("eager_scheduled_seq_ids"))
        or as_int_list(record.get("eager_result_received_seq_ids"))
        or as_int_list(record.get("eager_result_sent_seq_ids"))
    )
    return {proposal_id: seq_id for proposal_id, seq_id in zip(proposal_ids, seq_ids)}


def expected_action(accepted_len: int, full_accept: bool) -> str:
    if full_accept:
        return FULL_ACCEPT_ACTION
    return PARTIAL_ACTION if accepted_len > 0 else REJECT_ACTION


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_sync_enabled = 0
    candidate_ids_seen: set[int] = set()
    executed_ids_seen: set[int] = set()
    skipped_ids_seen: set[int] = set()
    validated_result_ids_seen: set[int] = set()
    full_accept_count = 0
    discard_count = 0
    target_rollback_ok_count = 0
    draft_rollback_ok_count = 0
    target_mutation_remaining_count = 0
    draft_mutation_remaining_count = 0
    expected_conflict_count = 0
    unexpected_overshot_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    action_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        sync_enabled = bool(record.get("enable_eager_sync_apply_dry_run", False))
        sync_active = bool(record.get("eager_sync_apply_dry_run_enabled", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        candidate_ids = as_int_set(record.get("eager_sync_apply_candidate_proposal_ids"))
        executed_ids = as_int_set(record.get("eager_sync_apply_executed_proposal_ids"))
        skipped_ids = as_int_set(record.get("eager_sync_apply_skipped_proposal_ids"))
        skip_reasons = record.get("eager_sync_apply_skip_reason_by_proposal_id", {})
        action_by_seq = record.get("eager_sync_apply_action_by_seq_id", {})
        accepted_len_by_seq = record.get("eager_sync_apply_accepted_len_by_seq_id", {})
        full_accept_by_seq = record.get("eager_sync_apply_full_accept_by_seq_id", {})
        proposal_to_seq = proposal_seq_map(record)
        gamma = int_value(record.get("normal_gamma"), None)
        runner_role = str(record.get("runner_role", ""))
        is_target_side = "verify" in runner_role
        is_draft_side = "draft" in runner_role
        validated_result_ids = as_int_set(record.get("eager_result_validated_proposal_ids"))
        sent_result_ids = as_int_set(record.get("eager_result_sent_proposal_ids"))
        received_result_ids = as_int_set(record.get("eager_result_received_proposal_ids"))
        allowed_result_ids = validated_result_ids or received_result_ids or sent_result_ids
        sync_tokens = int_value(record.get("eager_tokens_sync_apply_dry_run"), 0)
        target_tokens = int_value(record.get("eager_tokens_sync_apply_dry_run_target_side"), 0)
        draft_tokens = int_value(record.get("eager_tokens_sync_apply_dry_run_draft_side"), 0)

        if target_eager:
            real_target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_actual = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_verified_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must stay zero: {nonzero_actual}")

        if not sync_enabled:
            populated = (
                sync_active
                or candidate_ids
                or executed_ids
                or skipped_ids
                or sync_tokens
                or target_tokens
                or draft_tokens
                or int_value(record.get("eager_tokens_sync_apply_dry_run_full_accept"), 0)
                or int_value(record.get("eager_tokens_sync_apply_dry_run_discarded"), 0)
            )
            if populated:
                errors.append(f"record[{idx}] sync apply fields populated while disabled")
            continue

        records_with_sync_enabled += 1
        validated_result_ids_seen.update(validated_result_ids)
        if not bool(record.get("enable_eager_result_transfer_dry_run", False)):
            errors.append(f"record[{idx}] sync apply dry-run must imply result transfer dry-run")
        if not sync_active and (candidate_ids or executed_ids or skipped_ids or sync_tokens):
            errors.append(f"record[{idx}] sync apply data populated without active sync flag")
        if candidate_ids and allowed_result_ids and not candidate_ids <= allowed_result_ids:
            errors.append(
                f"record[{idx}] sync candidates must come from result-transfer ids: "
                f"extra={sorted(candidate_ids - allowed_result_ids)}"
            )

        candidate_ids_seen.update(candidate_ids)
        executed_ids_seen.update(executed_ids)
        skipped_ids_seen.update(skipped_ids)

        classified = executed_ids | skipped_ids
        if candidate_ids and classified != candidate_ids:
            errors.append(
                f"record[{idx}] sync candidates must be executed or skipped: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_ids)}"
            )
        if executed_ids & skipped_ids:
            errors.append(f"record[{idx}] proposal ids both executed and skipped: {sorted(executed_ids & skipped_ids)}")
        if executed_ids and sync_tokens <= 0:
            errors.append(f"record[{idx}] executed sync apply proposals require positive dry-run token count")
        if is_target_side and draft_tokens:
            errors.append(f"record[{idx}] target-side sync row must not count draft-side tokens")
        if is_draft_side and target_tokens:
            errors.append(f"record[{idx}] draft-side sync row must not count target-side tokens")

        for proposal_id in sorted(skipped_ids):
            reason = dict_get(skip_reasons, proposal_id)
            skip_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing skip reason")
            elif reason not in VALID_SKIP_REASONS:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad reason={reason!r}")
            if reason == "unexpected_draft_base_overshot":
                unexpected_overshot_count += 1

        for proposal_id in sorted(executed_ids):
            seq_id = proposal_to_seq.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing proposal-to-seq mapping")
                continue
            accepted_len = int_value(dict_get(accepted_len_by_seq, seq_id), -1)
            full_accept = bool(dict_get(full_accept_by_seq, seq_id, False))
            action = str(dict_get(action_by_seq, seq_id, ""))
            action_counts[action] += 1
            if gamma is not None and not (0 <= accepted_len <= gamma):
                errors.append(f"record[{idx}] accepted_len out of range for seq_id={seq_id}: {accepted_len}")
            if gamma is not None and full_accept != (accepted_len == gamma):
                errors.append(f"record[{idx}] full_accept mismatch for seq_id={seq_id}")
            expected = expected_action(accepted_len, full_accept)
            if action != expected:
                errors.append(f"record[{idx}] seq_id={seq_id} bad sync action={action!r}; expected={expected!r}")
            if full_accept:
                full_accept_count += 1
            else:
                discard_count += 1

            if is_target_side:
                checkpoint_ok = dict_get(record.get("target_sync_apply_checkpoint_ok_by_seq_id", {}), seq_id)
                rollback_ok = dict_get(record.get("target_sync_apply_rollback_ok_by_seq_id", {}), seq_id)
                mutation = dict_get(record.get("target_sync_apply_mutation_remaining_by_seq_id", {}), seq_id)
                len_before = int_value(dict_get(record.get("target_sync_apply_len_before_by_seq_id", {}), seq_id), -1)
                len_after_sim = int_value(dict_get(record.get("target_sync_apply_len_after_simulated_by_seq_id", {}), seq_id), -1)
                len_after_restore = int_value(dict_get(record.get("target_sync_apply_len_after_restore_by_seq_id", {}), seq_id), -1)
                if checkpoint_ok is not True:
                    errors.append(f"record[{idx}] target checkpoint failed for seq_id={seq_id}")
                if rollback_ok is not True:
                    errors.append(f"record[{idx}] target rollback failed for seq_id={seq_id}")
                else:
                    target_rollback_ok_count += 1
                if mutation is True:
                    target_mutation_remaining_count += 1
                    errors.append(f"record[{idx}] target mutation remained for seq_id={seq_id}")
                if len_after_restore != len_before:
                    errors.append(f"record[{idx}] target len not restored for seq_id={seq_id}")
                if full_accept and gamma is not None and len_after_sim != len_before + gamma:
                    errors.append(f"record[{idx}] target full-accept simulated len mismatch for seq_id={seq_id}")
                if not full_accept and len_after_sim != len_before:
                    errors.append(f"record[{idx}] target discard action mutated len for seq_id={seq_id}")

            if is_draft_side:
                checkpoint_ok = dict_get(record.get("draft_sync_apply_checkpoint_ok_by_seq_id", {}), seq_id)
                rollback_ok = dict_get(record.get("draft_sync_apply_rollback_ok_by_seq_id", {}), seq_id)
                mutation = dict_get(record.get("draft_sync_apply_mutation_remaining_by_seq_id", {}), seq_id)
                len_before = int_value(dict_get(record.get("draft_sync_apply_len_before_by_seq_id", {}), seq_id), -1)
                base_len = int_value(dict_get(record.get("draft_sync_apply_base_len_by_seq_id", {}), seq_id), -1)
                len_after_sim = int_value(dict_get(record.get("draft_sync_apply_len_after_simulated_by_seq_id", {}), seq_id), -1)
                len_after_restore = int_value(dict_get(record.get("draft_sync_apply_len_after_restore_by_seq_id", {}), seq_id), -1)
                expected_conflict = bool(
                    dict_get(record.get("draft_sync_apply_expected_normal_draft_conflict_by_seq_id", {}), seq_id, False)
                )
                original_draft_home = bool(
                    dict_get(record.get("draft_sync_apply_original_draft_home_intersection_by_seq_id", {}), seq_id, False)
                )
                adjusted_excluded = bool(
                    dict_get(record.get("draft_sync_apply_adjusted_draft_home_exclusion_by_seq_id", {}), seq_id, False)
                )
                if expected_conflict:
                    expected_conflict_count += 1
                    if not (original_draft_home or adjusted_excluded):
                        errors.append(f"record[{idx}] expected draft conflict lacks dry-run exclusion evidence")
                if len_before > base_len and not expected_conflict:
                    unexpected_overshot_count += 1
                    errors.append(f"record[{idx}] draft base overshot without expected conflict for seq_id={seq_id}")
                if len_before < base_len:
                    errors.append(f"record[{idx}] draft executed before base reached for seq_id={seq_id}")
                if checkpoint_ok is not True:
                    errors.append(f"record[{idx}] draft checkpoint failed for seq_id={seq_id}")
                if rollback_ok is not True:
                    errors.append(f"record[{idx}] draft rollback failed for seq_id={seq_id}")
                else:
                    draft_rollback_ok_count += 1
                if mutation is True:
                    draft_mutation_remaining_count += 1
                    errors.append(f"record[{idx}] draft mutation remained for seq_id={seq_id}")
                if len_after_restore != len_before:
                    errors.append(f"record[{idx}] draft len not restored for seq_id={seq_id}")
                expected_full_len = base_len + (gamma if gamma is not None else 0)
                if full_accept and gamma is not None and len_after_sim != expected_full_len:
                    errors.append(f"record[{idx}] draft full-accept simulated len mismatch for seq_id={seq_id}")
                if not full_accept and len_after_sim != len_before:
                    errors.append(f"record[{idx}] draft discard action mutated len for seq_id={seq_id}")

    summary = {
        "total_trace_records": len(records),
        "records_with_sync_apply_dry_run_enabled": records_with_sync_enabled,
        "sync_apply_candidate_proposal_count": len(candidate_ids_seen),
        "sync_apply_executed_proposal_count": len(executed_ids_seen),
        "sync_apply_skipped_proposal_count": len(skipped_ids_seen),
        "validated_result_proposal_count": len(validated_result_ids_seen),
        "full_accept_count": full_accept_count,
        "partial_reject_discard_count": discard_count,
        "target_rollback_ok_count": target_rollback_ok_count,
        "draft_rollback_ok_count": draft_rollback_ok_count,
        "target_mutation_remaining_count": target_mutation_remaining_count,
        "draft_mutation_remaining_count": draft_mutation_remaining_count,
        "expected_normal_draft_conflict_count": expected_conflict_count,
        "unexpected_draft_base_overshot_count": unexpected_overshot_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "sync_apply_action_counts": dict(action_counts),
        "skip_reason_counts": dict(skip_reason_counts),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_sync_apply_dry_run_enabled",
        "sync_apply_candidate_proposal_count",
        "sync_apply_executed_proposal_count",
        "sync_apply_skipped_proposal_count",
        "validated_result_proposal_count",
        "full_accept_count",
        "partial_reject_discard_count",
        "target_rollback_ok_count",
        "draft_rollback_ok_count",
        "target_mutation_remaining_count",
        "draft_mutation_remaining_count",
        "expected_normal_draft_conflict_count",
        "unexpected_draft_base_overshot_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "sync_apply_action_counts",
        "skip_reason_counts",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "runner_role": "dual_draft",
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_result_transfer_dry_run": False,
        "enable_eager_sync_apply_dry_run": False,
        "eager_sync_apply_dry_run_enabled": False,
        "eager_sync_apply_candidate_proposal_ids": [],
        "eager_sync_apply_candidate_seq_ids": [],
        "eager_sync_apply_executed_proposal_ids": [],
        "eager_sync_apply_executed_seq_ids": [],
        "eager_sync_apply_skipped_proposal_ids": [],
        "eager_tokens_sync_apply_dry_run": 0,
        "eager_tokens_sync_apply_dry_run_full_accept": 0,
        "eager_tokens_sync_apply_dry_run_discarded": 0,
        "eager_tokens_sync_apply_dry_run_target_side": 0,
        "eager_tokens_sync_apply_dry_run_draft_side": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_sync_record(
    side: str = "draft",
    accepted_len: int = 4,
    len_before: int = 12,
    base_len: int = 12,
    expected_conflict: bool = False,
) -> dict[str, Any]:
    record = synthetic_base_record()
    full_accept = accepted_len == 4
    action = expected_action(accepted_len, full_accept)
    simulated_len = base_len + 4 if full_accept else len_before
    side_prefix = "draft" if side == "draft" else "target"
    record.update(
        {
            "runner_role": "dual_draft" if side == "draft" else "dual_verify",
            "enable_eager_result_transfer_dry_run": True,
            "enable_eager_sync_apply_dry_run": True,
            "eager_sync_apply_dry_run_enabled": True,
            "eager_sync_apply_candidate_proposal_ids": [101],
            "eager_sync_apply_candidate_seq_ids": [3],
            "eager_sync_apply_executed_proposal_ids": [101],
            "eager_sync_apply_executed_seq_ids": [3],
            "eager_sync_apply_action_by_seq_id": {"3": action},
            "eager_sync_apply_accepted_len_by_seq_id": {"3": accepted_len},
            "eager_sync_apply_full_accept_by_seq_id": {"3": full_accept},
            "eager_tokens_sync_apply_dry_run": 4,
            "eager_tokens_sync_apply_dry_run_full_accept": 4 if full_accept else 0,
            "eager_tokens_sync_apply_dry_run_discarded": 0 if full_accept else 4,
            f"eager_tokens_sync_apply_dry_run_{side_prefix}_side": 4,
            "eager_result_validated_proposal_ids": [101],
            "eager_result_received_proposal_ids": [101],
            "eager_result_received_seq_ids": [3],
            "eager_result_sent_proposal_ids": [101],
            "eager_result_sent_seq_ids": [3],
            "scheduled_target_eager_proposal_ids_dry_run": [101],
            "scheduled_target_eager_seq_ids_dry_run": [3],
        }
    )
    if side == "draft":
        record.update(
            {
                "draft_sync_apply_checkpoint_ok_by_seq_id": {"3": True},
                "draft_sync_apply_rollback_ok_by_seq_id": {"3": True},
                "draft_sync_apply_mutation_remaining_by_seq_id": {"3": False},
                "draft_sync_apply_len_before_by_seq_id": {"3": len_before},
                "draft_sync_apply_base_len_by_seq_id": {"3": base_len},
                "draft_sync_apply_len_after_simulated_by_seq_id": {"3": simulated_len},
                "draft_sync_apply_len_after_restore_by_seq_id": {"3": len_before},
                "draft_sync_apply_status_before_by_seq_id": {"3": "RUNNING"},
                "draft_sync_apply_status_after_restore_by_seq_id": {"3": "RUNNING"},
                "draft_sync_apply_expected_normal_draft_conflict_by_seq_id": {"3": expected_conflict},
                "draft_sync_apply_original_draft_home_intersection_by_seq_id": {"3": expected_conflict},
                "draft_sync_apply_adjusted_draft_home_exclusion_by_seq_id": {"3": expected_conflict},
            }
        )
    else:
        record.update(
            {
                "target_sync_apply_checkpoint_ok_by_seq_id": {"3": True},
                "target_sync_apply_rollback_ok_by_seq_id": {"3": True},
                "target_sync_apply_mutation_remaining_by_seq_id": {"3": False},
                "target_sync_apply_len_before_by_seq_id": {"3": len_before},
                "target_sync_apply_len_after_simulated_by_seq_id": {"3": len_before + 4 if full_accept else len_before},
                "target_sync_apply_len_after_restore_by_seq_id": {"3": len_before},
                "target_sync_apply_status_before_by_seq_id": {"3": "RUNNING"},
                "target_sync_apply_status_after_restore_by_seq_id": {"3": "RUNNING"},
            }
        )
    return record


def synthetic_skip_record(reason: str) -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_result_transfer_dry_run": True,
            "enable_eager_sync_apply_dry_run": True,
            "eager_sync_apply_dry_run_enabled": True,
            "eager_sync_apply_candidate_proposal_ids": [101],
            "eager_sync_apply_candidate_seq_ids": [3],
            "eager_sync_apply_skipped_proposal_ids": [101],
            "eager_sync_apply_skip_reason_by_proposal_id": {"101": reason},
            "eager_result_validated_proposal_ids": [101],
            "eager_result_received_proposal_ids": [101],
            "eager_result_received_seq_ids": [3],
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_sync_record("target", 4),
        synthetic_sync_record("draft", 4),
        synthetic_sync_record("draft", 2),
        synthetic_sync_record("draft", 0),
        synthetic_sync_record("draft", 4, len_before=16, base_len=12, expected_conflict=True),
        synthetic_skip_record("missing_proposal_id"),
        synthetic_skip_record("unexpected_draft_base_overshot"),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic sync apply records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[2]["draft_sync_apply_mutation_remaining_by_seq_id"] = {"3": True}
    errors, _ = validate_records(invalid)
    assert any("draft mutation remained" in error for error in errors), "checker missed draft mutation"

    invalid = deepcopy(valid_records)
    invalid[1]["target_sync_apply_rollback_ok_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("target rollback failed" in error for error in errors), "checker missed target rollback failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_sync_apply_action_by_seq_id"] = {"3": PARTIAL_ACTION}
    errors, _ = validate_records(invalid)
    assert any("bad sync action" in error for error in errors), "checker missed action mismatch"

    invalid = deepcopy(valid_records)
    invalid[5]["draft_sync_apply_expected_normal_draft_conflict_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("draft base overshot" in error for error in errors), "checker missed unexpected overshot"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    print("Synthetic eager sync apply dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5d eager sync apply dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Eager sync apply dry-run trace checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
