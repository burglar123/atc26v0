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

VALID_SKIP_REASONS = {
    "base_mismatch_before_apply_dry_run",
    "seq_pre_verify_before_apply_dry_run",
    "invalid_proposal_len",
    "invalid_to_verify_len",
    "missing_verify_result",
    "seq_finished_before_apply_dry_run",
    "span_invalidated_before_apply_dry_run",
}

FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"
PARTIAL_ACTION = "discard_partial_no_mutation"
REJECT_ACTION = "discard_reject_no_mutation"


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


def scheduled_proposal_seq_map(record: dict[str, Any]) -> dict[int, int]:
    proposal_ids = (
        as_int_list(record.get("scheduled_target_eager_proposal_ids_dry_run"))
        or as_int_list(record.get("eager_scheduled_proposal_ids"))
    )
    seq_ids = (
        as_int_list(record.get("scheduled_target_eager_seq_ids_dry_run"))
        or as_int_list(record.get("eager_scheduled_seq_ids"))
    )
    return {proposal_id: seq_id for proposal_id, seq_id in zip(proposal_ids, seq_ids)}


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_apply_enabled = 0
    apply_candidate_ids_seen: set[int] = set()
    apply_executed_ids_seen: set[int] = set()
    apply_skipped_ids_seen: set[int] = set()
    full_accept_apply_count = 0
    discard_count = 0
    appended_token_count = 0
    rollback_ok_count = 0
    rollback_failure_count = 0
    mutation_remaining_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    action_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    apply_tokens = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        apply_enabled = bool(record.get("enable_eager_apply_dry_run", False))
        apply_active = bool(record.get("eager_apply_dry_run_enabled", False))
        verify_enabled = bool(record.get("enable_eager_verify_dry_run", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        scheduled_map = scheduled_proposal_seq_map(record)
        verify_executed_ids = as_int_set(record.get("eager_verify_executed_proposal_ids"))
        candidate_ids = as_int_set(record.get("eager_apply_candidate_proposal_ids"))
        executed_ids = as_int_set(record.get("eager_apply_executed_proposal_ids"))
        executed_seq_ids = as_int_set(record.get("eager_apply_executed_seq_ids"))
        skipped_ids = as_int_set(record.get("eager_apply_skipped_proposal_ids"))
        skip_reasons = record.get("eager_apply_skip_reason_by_proposal_id", {})
        action_by_seq = record.get("eager_apply_action_by_seq_id", {})
        accepted_len_by_seq = record.get("eager_apply_accepted_len_by_seq_id", {})
        full_accept_by_seq = record.get("eager_apply_full_accept_by_seq_id", {})
        base_len_by_seq = record.get("eager_apply_base_len_by_seq_id", {})
        len_before_by_seq = record.get("eager_apply_current_len_before_by_seq_id", {})
        len_after_apply_by_seq = record.get("eager_apply_current_len_after_simulated_apply_by_seq_id", {})
        len_after_rollback_by_seq = record.get("eager_apply_current_len_after_rollback_by_seq_id", {})
        pre_before_by_seq = record.get("eager_apply_pre_verify_before_by_seq_id", {})
        pre_after_rollback_by_seq = record.get("eager_apply_pre_verify_after_rollback_by_seq_id", {})
        status_before_by_seq = record.get("eager_apply_status_before_by_seq_id", {})
        status_after_rollback_by_seq = record.get("eager_apply_status_after_rollback_by_seq_id", {})
        checkpoint_ok_by_seq = record.get("eager_apply_checkpoint_ok_by_seq_id", {})
        rollback_ok_by_seq = record.get("eager_apply_rollback_ok_by_seq_id", {})
        mutation_by_seq = record.get("eager_apply_mutation_remaining_by_seq_id", {})
        appended_by_seq = record.get("eager_apply_dry_run_appended_token_count_by_seq_id", {})
        proposal_len_by_id = record.get("eager_schedule_proposal_len_by_proposal_id", {})
        gamma = int_value(record.get("normal_gamma"), None)
        dry_run_tokens = int_value(record.get("eager_tokens_apply_dry_run"), 0)
        full_accept_tokens = int_value(record.get("eager_tokens_apply_dry_run_full_accept"), 0)
        discarded_tokens = int_value(record.get("eager_tokens_apply_dry_run_discarded"), 0)
        append_tokens = int_value(record.get("eager_apply_dry_run_append_tokens"), 0)
        rollback_failure_row_count = int_value(record.get("eager_apply_dry_run_rollback_failure_count"), 0)

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

        if not apply_enabled:
            if (
                apply_active
                or candidate_ids
                or executed_ids
                or skipped_ids
                or dry_run_tokens
                or full_accept_tokens
                or discarded_tokens
                or append_tokens
                or rollback_failure_row_count
            ):
                errors.append(f"record[{idx}] eager apply dry-run fields populated while disabled")
            continue

        records_with_apply_enabled += 1
        if not verify_enabled:
            errors.append(f"record[{idx}] apply dry-run must imply verify dry-run")
        if verify_executed_ids and not candidate_ids:
            errors.append(f"record[{idx}] verify executed proposals exist but apply candidates are empty")
        if not candidate_ids <= verify_executed_ids:
            errors.append(
                f"record[{idx}] apply candidates must come from verify executed proposals: "
                f"extra={sorted(candidate_ids - verify_executed_ids)}"
            )

        apply_candidate_ids_seen.update(candidate_ids)
        apply_executed_ids_seen.update(executed_ids)
        apply_skipped_ids_seen.update(skipped_ids)
        apply_tokens += dry_run_tokens
        appended_token_count += append_tokens
        rollback_failure_count += rollback_failure_row_count

        classified = executed_ids | skipped_ids
        if candidate_ids and classified != candidate_ids:
            errors.append(
                f"record[{idx}] apply candidates must be executed or skipped: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_ids)}"
            )
        if executed_ids & skipped_ids:
            errors.append(f"record[{idx}] proposal ids both executed and skipped: {sorted(executed_ids & skipped_ids)}")
        if dry_run_tokens and not executed_ids:
            errors.append(f"record[{idx}] eager_tokens_apply_dry_run set without executed proposals")
        if executed_ids and dry_run_tokens <= 0:
            errors.append(f"record[{idx}] executed apply proposals require positive dry-run token count")
        if full_accept_tokens + discarded_tokens != dry_run_tokens:
            errors.append(
                f"record[{idx}] full_accept + discarded apply counters must equal total: "
                f"{full_accept_tokens} + {discarded_tokens} != {dry_run_tokens}"
            )

        for proposal_id in sorted(skipped_ids):
            reason = dict_get(skip_reasons, proposal_id)
            skip_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing skip reason")
            elif reason not in VALID_SKIP_REASONS:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad reason={reason!r}")

        for proposal_id in sorted(executed_ids):
            seq_id = scheduled_map.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing scheduled mapping")
                continue
            if seq_id not in executed_seq_ids:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} seq_id={seq_id} missing executed seq")
            if proposal_id not in verify_executed_ids:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} was not verify-dry-run executed")
            if gamma is not None and int_value(dict_get(proposal_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} proposal_len != gamma")

            accepted_len = int_value(dict_get(accepted_len_by_seq, seq_id), -1)
            full_accept = bool(dict_get(full_accept_by_seq, seq_id, False))
            action = str(dict_get(action_by_seq, seq_id, ""))
            action_counts[action] += 1
            base_len = int_value(dict_get(base_len_by_seq, seq_id), -1)
            len_before = int_value(dict_get(len_before_by_seq, seq_id), -1)
            len_after_apply = int_value(dict_get(len_after_apply_by_seq, seq_id), -1)
            len_after_rollback = int_value(dict_get(len_after_rollback_by_seq, seq_id), -1)
            appended_count = int_value(dict_get(appended_by_seq, seq_id), -1)

            if gamma is not None and not (0 <= accepted_len <= gamma):
                errors.append(f"record[{idx}] accepted_len out of range for seq_id={seq_id}: {accepted_len}")
            if gamma is not None and full_accept != (accepted_len == gamma):
                errors.append(
                    f"record[{idx}] full_accept mismatch for seq_id={seq_id}: "
                    f"full_accept={full_accept}, accepted_len={accepted_len}, gamma={gamma}"
                )
            if dict_get(pre_before_by_seq, seq_id) is not False:
                errors.append(f"record[{idx}] executed seq_id={seq_id} pre_verify before apply is not false")
            if dict_get(pre_after_rollback_by_seq, seq_id) is not False:
                errors.append(f"record[{idx}] executed seq_id={seq_id} pre_verify not restored after rollback")
            if dict_get(status_before_by_seq, seq_id) != dict_get(status_after_rollback_by_seq, seq_id):
                errors.append(f"record[{idx}] executed seq_id={seq_id} status not restored after rollback")
            if base_len != len_before:
                errors.append(f"record[{idx}] executed seq_id={seq_id} len_before must equal base_len")
            if len_after_rollback != len_before:
                errors.append(f"record[{idx}] executed seq_id={seq_id} len not restored after rollback")
            if dict_get(checkpoint_ok_by_seq, seq_id) is not True:
                errors.append(f"record[{idx}] checkpoint failed for executed seq_id={seq_id}")
            if dict_get(rollback_ok_by_seq, seq_id) is not True:
                errors.append(f"record[{idx}] rollback failed for executed seq_id={seq_id}")
            else:
                rollback_ok_count += 1
            if dict_get(mutation_by_seq, seq_id) is True:
                mutation_remaining_count += 1
                errors.append(f"record[{idx}] mutation remained after rollback for seq_id={seq_id}")

            if full_accept:
                full_accept_apply_count += 1
                if action != FULL_ACCEPT_ACTION:
                    errors.append(f"record[{idx}] full-accept seq_id={seq_id} bad action={action!r}")
                if gamma is not None and appended_count != gamma:
                    errors.append(f"record[{idx}] full-accept seq_id={seq_id} appended_count != gamma")
                if gamma is not None and len_after_apply != len_before + gamma:
                    errors.append(f"record[{idx}] full-accept seq_id={seq_id} simulated apply length mismatch")
            else:
                discard_count += 1
                expected_action = PARTIAL_ACTION if accepted_len > 0 else REJECT_ACTION
                if action != expected_action:
                    errors.append(
                        f"record[{idx}] rejected/partial seq_id={seq_id} bad action={action!r}; "
                        f"expected={expected_action!r}"
                    )
                if appended_count != 0:
                    errors.append(f"record[{idx}] rejected/partial seq_id={seq_id} must append zero tokens")
                if len_after_apply != len_before:
                    errors.append(f"record[{idx}] rejected/partial seq_id={seq_id} mutated during discard action")

    summary = {
        "total_trace_records": len(records),
        "records_with_apply_dry_run_enabled": records_with_apply_enabled,
        "apply_candidate_proposal_count": len(apply_candidate_ids_seen),
        "apply_executed_proposal_count": len(apply_executed_ids_seen),
        "apply_skipped_proposal_count": len(apply_skipped_ids_seen),
        "full_accept_apply_count": full_accept_apply_count,
        "discard_partial_reject_count": discard_count,
        "appended_token_count_in_dry_run": appended_token_count,
        "rollback_ok_count": rollback_ok_count,
        "rollback_failure_count": rollback_failure_count,
        "mutation_remaining_count": mutation_remaining_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "apply_action_counts": dict(action_counts),
        "skip_reason_counts": dict(skip_reason_counts),
        "eager_tokens_apply_dry_run": apply_tokens,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_apply_dry_run_enabled",
        "apply_candidate_proposal_count",
        "apply_executed_proposal_count",
        "apply_skipped_proposal_count",
        "full_accept_apply_count",
        "discard_partial_reject_count",
        "appended_token_count_in_dry_run",
        "rollback_ok_count",
        "rollback_failure_count",
        "mutation_remaining_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "apply_action_counts",
        "skip_reason_counts",
        "eager_tokens_apply_dry_run",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "enable_eager_transfer_dry_run": False,
        "enable_eager_schedule_dry_run": False,
        "enable_eager_verify_dry_run": False,
        "enable_eager_apply_dry_run": False,
        "eager_apply_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_eager_set": [],
        "scheduled_target_eager_set_dry_run": [],
        "scheduled_target_eager_proposal_ids_dry_run": [],
        "scheduled_target_eager_seq_ids_dry_run": [],
        "eager_apply_candidate_proposal_ids": [],
        "eager_apply_candidate_seq_ids": [],
        "eager_apply_executed_proposal_ids": [],
        "eager_apply_executed_seq_ids": [],
        "eager_apply_skipped_proposal_ids": [],
        "eager_apply_skip_reason_by_proposal_id": {},
        "eager_tokens_apply_dry_run": 0,
        "eager_tokens_apply_dry_run_full_accept": 0,
        "eager_tokens_apply_dry_run_discarded": 0,
        "eager_apply_dry_run_append_tokens": 0,
        "eager_apply_dry_run_rollback_failure_count": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_verify_record(accepted_len: int = 4) -> dict[str, Any]:
    record = synthetic_base_record()
    full_accept = accepted_len == 4
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "scheduled_target_eager_set_dry_run": [3],
            "scheduled_target_eager_proposal_ids_dry_run": [101],
            "scheduled_target_eager_seq_ids_dry_run": [3],
            "eager_scheduled_proposal_ids": [101],
            "eager_scheduled_seq_ids": [3],
            "eager_schedule_proposal_len_by_proposal_id": {"101": 4},
            "eager_verify_executed_proposal_ids": [101],
            "eager_verify_executed_seq_ids": [3],
            "eager_verify_accepted_len_by_seq_id": {"3": accepted_len},
            "eager_verify_full_accept_by_seq_id": {"3": full_accept},
        }
    )
    return record


def synthetic_apply_record(accepted_len: int = 4) -> dict[str, Any]:
    record = synthetic_verify_record(accepted_len)
    full_accept = accepted_len == 4
    action = FULL_ACCEPT_ACTION if full_accept else (PARTIAL_ACTION if accepted_len > 0 else REJECT_ACTION)
    appended = 4 if full_accept else 0
    record.update(
        {
            "enable_eager_apply_dry_run": True,
            "eager_apply_dry_run_enabled": True,
            "eager_apply_dry_run_step_id": 7,
            "eager_apply_dry_run_plan_id": 11,
            "eager_apply_candidate_proposal_ids": [101],
            "eager_apply_candidate_seq_ids": [3],
            "eager_apply_executed_proposal_ids": [101],
            "eager_apply_executed_seq_ids": [3],
            "eager_apply_action_by_seq_id": {"3": action},
            "eager_apply_accepted_len_by_seq_id": {"3": accepted_len},
            "eager_apply_full_accept_by_seq_id": {"3": full_accept},
            "eager_apply_base_len_by_seq_id": {"3": 12},
            "eager_apply_current_len_before_by_seq_id": {"3": 12},
            "eager_apply_current_len_after_simulated_apply_by_seq_id": {"3": 12 + appended},
            "eager_apply_current_len_after_rollback_by_seq_id": {"3": 12},
            "eager_apply_pre_verify_before_by_seq_id": {"3": False},
            "eager_apply_pre_verify_after_simulated_apply_by_seq_id": {"3": False},
            "eager_apply_pre_verify_after_rollback_by_seq_id": {"3": False},
            "eager_apply_status_before_by_seq_id": {"3": "RUNNING"},
            "eager_apply_status_after_simulated_apply_by_seq_id": {"3": "RUNNING"},
            "eager_apply_status_after_rollback_by_seq_id": {"3": "RUNNING"},
            "eager_apply_checkpoint_ok_by_seq_id": {"3": True},
            "eager_apply_rollback_ok_by_seq_id": {"3": True},
            "eager_apply_mutation_remaining_by_seq_id": {"3": False},
            "eager_apply_dry_run_appended_token_count_by_seq_id": {"3": appended},
            "eager_tokens_apply_dry_run": 4,
            "eager_tokens_apply_dry_run_full_accept": 4 if full_accept else 0,
            "eager_tokens_apply_dry_run_discarded": 0 if full_accept else 4,
            "eager_apply_dry_run_append_tokens": appended,
            "eager_apply_dry_run_rollback_failure_count": 0,
        }
    )
    return record


def synthetic_apply_skip_record(reason: str) -> dict[str, Any]:
    record = synthetic_verify_record(4)
    record.update(
        {
            "enable_eager_apply_dry_run": True,
            "eager_apply_dry_run_enabled": True,
            "eager_apply_candidate_proposal_ids": [101],
            "eager_apply_candidate_seq_ids": [3],
            "eager_apply_skipped_proposal_ids": [101],
            "eager_apply_skip_reason_by_proposal_id": {"101": reason},
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_verify_record(4),
        synthetic_apply_record(4),
        synthetic_apply_record(2),
        synthetic_apply_record(0),
        synthetic_apply_skip_record("base_mismatch_before_apply_dry_run"),
        synthetic_apply_skip_record("seq_pre_verify_before_apply_dry_run"),
        synthetic_apply_skip_record("missing_verify_result"),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager apply records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_apply_current_len_after_rollback_by_seq_id"] = {"3": 16}
    errors, _ = validate_records(invalid)
    assert any("len not restored" in error for error in errors), "checker missed rollback len failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_apply_mutation_remaining_by_seq_id"] = {"3": True}
    errors, _ = validate_records(invalid)
    assert any("mutation remained" in error for error in errors), "checker missed remaining mutation"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_apply_rollback_ok_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("rollback failed" in error for error in errors), "checker missed rollback failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_apply_action_by_seq_id"] = {"3": PARTIAL_ACTION}
    errors, _ = validate_records(invalid)
    assert any("bad action" in error for error in errors), "checker missed full-accept action mismatch"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_apply_dry_run_appended_token_count_by_seq_id"] = {"3": 1}
    errors, _ = validate_records(invalid)
    assert any("append zero" in error for error in errors), "checker missed partial mutation"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_apply_accepted_len_by_seq_id"] = {"3": 5}
    errors, _ = validate_records(invalid)
    assert any("accepted_len out of range" in error for error in errors), "checker missed accepted_len range"

    print("Synthetic eager apply dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5b eager apply dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        if args.trace is None:
            return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nEager apply dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
