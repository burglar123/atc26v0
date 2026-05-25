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
    "not_scheduled_proposal",
    "not_scheduled_seq",
    "intersects_target_home",
    "intersects_adjusted_draft_home",
    "invalid_lane",
    "invalid_state",
    "invalid_base_pre_verify",
    "invalid_proposal_len",
    "invalid_to_verify_len",
    "invalid_proposal_token_len",
    "seq_not_found",
    "seq_finished_before_verify",
    "seq_span_invalidated_before_verify",
    "seq_returned_pre_verify_before_verify",
    "base_mismatch_before_verify",
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
    verify_records = 0
    scheduled_proposal_ids_seen: set[int] = set()
    executed_proposal_ids_seen: set[int] = set()
    executed_seq_ids_seen: set[int] = set()
    skipped_proposal_ids_seen: set[int] = set()
    skip_reason_counts: Counter[str] = Counter()
    accepted_len_distribution: Counter[int] = Counter()
    full_accept_count = 0
    reject_partial_count = 0
    checkpoint_failure_count = 0
    mutation_detected_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    verify_tokens = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        verify_enabled = bool(record.get("enable_eager_verify_dry_run", False))
        verify_active = bool(record.get("eager_verify_dry_run_enabled", False))
        schedule_enabled = bool(record.get("enable_eager_schedule_dry_run", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        scheduled_seq_ids = (
            as_int_set(record.get("scheduled_target_eager_set_dry_run"))
            or as_int_set(record.get("target_eager_set_dry_run"))
        )
        scheduled_map = scheduled_proposal_seq_map(record)
        scheduled_proposal_ids = set(scheduled_map)
        candidate_proposal_ids = as_int_set(record.get("eager_verify_candidate_proposal_ids"))
        candidate_seq_ids = as_int_set(record.get("eager_verify_candidate_seq_ids"))
        executed_proposal_ids = as_int_set(record.get("eager_verify_executed_proposal_ids"))
        executed_seq_ids = as_int_set(record.get("eager_verify_executed_seq_ids"))
        skipped_proposal_ids = as_int_set(record.get("eager_verify_skipped_proposal_ids"))
        skip_reasons = record.get("eager_verify_skip_reason_by_proposal_id", {})
        accepted_len_by_seq = record.get("eager_verify_accepted_len_by_seq_id", {})
        full_accept_by_seq = record.get("eager_verify_full_accept_by_seq_id", {})
        invalidated_len_by_seq = record.get("eager_verify_invalidated_len_by_seq_id", {})
        reject_position_by_seq = record.get("eager_verify_reject_position_by_seq_id", {})
        base_match_by_seq = record.get("eager_verify_base_match_by_seq_id", {})
        seq_pre_verify_by_seq = record.get("eager_verify_seq_pre_verify_by_seq_id", {})
        checkpoint_ok_by_seq = record.get("eager_verify_checkpoint_ok_by_seq_id", {})
        mutation_by_seq = record.get("eager_verify_mutation_detected_by_seq_id", {})
        proposal_len_by_id = record.get("eager_schedule_proposal_len_by_proposal_id", {})
        to_verify_len_by_id = record.get("eager_schedule_to_verify_len_by_proposal_id", {})
        gamma = int_value(record.get("normal_gamma"), None)
        dry_run_tokens = int_value(record.get("eager_tokens_verify_dry_run"), 0)
        full_accept_tokens = int_value(record.get("eager_tokens_verify_dry_run_full_accept"), 0)
        rejected_tokens = int_value(record.get("eager_tokens_verify_dry_run_rejected"), 0)

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
            errors.append(f"record[{idx}] actual eager verification counters must stay zero: {nonzero_actual}")

        if not verify_enabled:
            if verify_active or candidate_proposal_ids or executed_proposal_ids or skipped_proposal_ids or dry_run_tokens:
                errors.append(f"record[{idx}] eager verify dry-run fields populated while disabled")
            continue

        verify_records += 1
        if not schedule_enabled:
            errors.append(f"record[{idx}] verify dry-run must imply schedule dry-run")

        scheduled_proposal_ids_seen.update(scheduled_proposal_ids)
        executed_proposal_ids_seen.update(executed_proposal_ids)
        executed_seq_ids_seen.update(executed_seq_ids)
        skipped_proposal_ids_seen.update(skipped_proposal_ids)
        verify_tokens += dry_run_tokens

        classified = executed_proposal_ids | skipped_proposal_ids
        if candidate_proposal_ids and classified != candidate_proposal_ids:
            errors.append(
                f"record[{idx}] verify candidates must be executed or skipped: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_proposal_ids)}"
            )
        if executed_proposal_ids & skipped_proposal_ids:
            errors.append(
                f"record[{idx}] proposal ids both executed and skipped: "
                f"{sorted(executed_proposal_ids & skipped_proposal_ids)}"
            )
        if scheduled_seq_ids and not candidate_proposal_ids:
            errors.append(f"record[{idx}] scheduled eager set present but no verify candidates were traced")

        if dry_run_tokens and not executed_proposal_ids:
            errors.append(f"record[{idx}] eager_tokens_verify_dry_run set without executed proposals")
        if executed_proposal_ids and dry_run_tokens <= 0:
            errors.append(f"record[{idx}] executed verify proposals require positive dry-run token count")
        if full_accept_tokens + rejected_tokens != dry_run_tokens:
            errors.append(
                f"record[{idx}] full_accept + rejected dry-run token counters must equal total: "
                f"{full_accept_tokens} + {rejected_tokens} != {dry_run_tokens}"
            )

        for proposal_id in sorted(skipped_proposal_ids):
            reason = dict_get(skip_reasons, proposal_id)
            skip_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing skip reason")
            elif reason not in VALID_SKIP_REASONS:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} bad reason={reason!r}")

        for proposal_id in sorted(executed_proposal_ids):
            seq_id = scheduled_map.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing scheduled proposal mapping")
                continue
            if proposal_id not in candidate_proposal_ids:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} missing from verify candidates")
            if seq_id not in executed_seq_ids:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} seq_id={seq_id} missing executed seq")
            if dict_get(base_match_by_seq, seq_id) is not True:
                errors.append(f"record[{idx}] executed seq_id={seq_id} lacks base_len match")
            if dict_get(seq_pre_verify_by_seq, seq_id) is not False:
                errors.append(f"record[{idx}] executed seq_id={seq_id} has pre_verify=true")
            if gamma is not None and int_value(dict_get(proposal_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and int_value(dict_get(to_verify_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] executed proposal_id={proposal_id} to_verify_len != gamma")

            accepted_len = int_value(dict_get(accepted_len_by_seq, seq_id), -1)
            invalidated_len = int_value(dict_get(invalidated_len_by_seq, seq_id), -1)
            reject_position = int_value(dict_get(reject_position_by_seq, seq_id), -2)
            full_accept = bool(dict_get(full_accept_by_seq, seq_id, False))
            if gamma is not None and not (0 <= accepted_len <= gamma):
                errors.append(f"record[{idx}] accepted_len out of range for seq_id={seq_id}: {accepted_len}")
            if gamma is not None and not (0 <= invalidated_len <= gamma):
                errors.append(f"record[{idx}] invalidated_len out of range for seq_id={seq_id}: {invalidated_len}")
            if gamma is not None and full_accept != (accepted_len == gamma):
                errors.append(
                    f"record[{idx}] full_accept mismatch for seq_id={seq_id}: "
                    f"full_accept={full_accept}, accepted_len={accepted_len}, gamma={gamma}"
                )
            if full_accept:
                full_accept_count += 1
                if reject_position != -1:
                    errors.append(f"record[{idx}] full accept seq_id={seq_id} must have reject_position=-1")
            else:
                reject_partial_count += 1
                if reject_position != accepted_len:
                    errors.append(
                        f"record[{idx}] reject_position must equal accepted_len for rejected seq_id={seq_id}"
                    )
            accepted_len_distribution[accepted_len] += 1

            if dict_get(checkpoint_ok_by_seq, seq_id) is not True:
                checkpoint_failure_count += 1
                errors.append(f"record[{idx}] checkpoint failed for executed seq_id={seq_id}")
            if dict_get(mutation_by_seq, seq_id) is True:
                mutation_detected_count += 1
                errors.append(f"record[{idx}] mutation detected for executed seq_id={seq_id}")

    summary = {
        "total_trace_records": len(records),
        "records_with_eager_verify_dry_run_enabled": verify_records,
        "scheduled_proposal_count": len(scheduled_proposal_ids_seen),
        "verify_dry_run_executed_proposal_count": len(executed_proposal_ids_seen),
        "verify_dry_run_skipped_proposal_count": len(skipped_proposal_ids_seen),
        "skip_reason_counts": dict(skip_reason_counts),
        "unique_verified_dry_run_proposal_ids": len(executed_proposal_ids_seen),
        "unique_verified_dry_run_seq_ids": len(executed_seq_ids_seen),
        "accepted_len_distribution": dict(accepted_len_distribution),
        "full_accept_count": full_accept_count,
        "reject_partial_count": reject_partial_count,
        "checkpoint_failure_count": checkpoint_failure_count,
        "mutation_detected_count": mutation_detected_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "eager_tokens_verify_dry_run": verify_tokens,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_eager_verify_dry_run_enabled",
        "scheduled_proposal_count",
        "verify_dry_run_executed_proposal_count",
        "verify_dry_run_skipped_proposal_count",
        "skip_reason_counts",
        "unique_verified_dry_run_proposal_ids",
        "unique_verified_dry_run_seq_ids",
        "accepted_len_distribution",
        "full_accept_count",
        "reject_partial_count",
        "checkpoint_failure_count",
        "mutation_detected_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "eager_tokens_verify_dry_run",
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
        "eager_verify_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_eager_set": [],
        "scheduled_target_eager_set_dry_run": [],
        "scheduled_target_eager_proposal_ids_dry_run": [],
        "scheduled_target_eager_seq_ids_dry_run": [],
        "eager_verify_candidate_proposal_ids": [],
        "eager_verify_candidate_seq_ids": [],
        "eager_verify_skipped_proposal_ids": [],
        "eager_verify_skip_reason_by_proposal_id": {},
        "eager_verify_executed_proposal_ids": [],
        "eager_verify_executed_seq_ids": [],
        "eager_tokens_verify_dry_run": 0,
        "eager_tokens_verify_dry_run_full_accept": 0,
        "eager_tokens_verify_dry_run_rejected": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_schedule_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "scheduled_target_eager_set_dry_run": [3],
            "scheduled_target_eager_proposal_ids_dry_run": [101],
            "scheduled_target_eager_seq_ids_dry_run": [3],
            "eager_scheduled_proposal_ids": [101],
            "eager_scheduled_seq_ids": [3],
            "eager_schedule_proposal_len_by_proposal_id": {"101": 4},
            "eager_schedule_to_verify_len_by_proposal_id": {"101": 4},
        }
    )
    return record


def synthetic_verify_record(accepted_len: int = 4) -> dict[str, Any]:
    record = synthetic_schedule_record()
    full_accept = accepted_len == 4
    record.update(
        {
            "enable_eager_verify_dry_run": True,
            "eager_verify_dry_run_enabled": True,
            "eager_verify_dry_run_step_id": 7,
            "eager_verify_dry_run_plan_id": 11,
            "eager_verify_candidate_proposal_ids": [101],
            "eager_verify_candidate_seq_ids": [3],
            "eager_verify_executed_proposal_ids": [101],
            "eager_verify_executed_seq_ids": [3],
            "eager_verify_accepted_len_by_seq_id": {"3": accepted_len},
            "eager_verify_full_accept_by_seq_id": {"3": full_accept},
            "eager_verify_reject_position_by_seq_id": {"3": -1 if full_accept else accepted_len},
            "eager_verify_invalidated_len_by_seq_id": {"3": 0 if full_accept else 4 - accepted_len},
            "eager_verify_revised_token_by_seq_id": {"3": -1 if full_accept else 200},
            "eager_verify_base_len_by_seq_id": {"3": 12},
            "eager_verify_current_len_by_seq_id": {"3": 12},
            "eager_verify_base_match_by_seq_id": {"3": True},
            "eager_verify_seq_pre_verify_by_seq_id": {"3": False},
            "eager_verify_seq_status_before_by_seq_id": {"3": "RUNNING"},
            "eager_verify_seq_status_after_by_seq_id": {"3": "RUNNING"},
            "eager_verify_mutation_detected_by_seq_id": {"3": False},
            "eager_verify_checkpoint_ok_by_seq_id": {"3": True},
            "eager_tokens_verify_dry_run": 4,
            "eager_tokens_verify_dry_run_full_accept": 4 if full_accept else 0,
            "eager_tokens_verify_dry_run_rejected": 0 if full_accept else 4,
        }
    )
    return record


def synthetic_verify_skip_record(reason: str) -> dict[str, Any]:
    record = synthetic_schedule_record()
    record.update(
        {
            "enable_eager_verify_dry_run": True,
            "eager_verify_dry_run_enabled": True,
            "eager_verify_candidate_proposal_ids": [101],
            "eager_verify_candidate_seq_ids": [3],
            "eager_verify_skipped_proposal_ids": [101],
            "eager_verify_skip_reason_by_proposal_id": {"101": reason},
            "eager_verify_base_len_by_seq_id": {"3": 12},
            "eager_verify_current_len_by_seq_id": {"3": 11 if reason == "base_mismatch_before_verify" else 12},
            "eager_verify_base_match_by_seq_id": {"3": reason != "base_mismatch_before_verify"},
            "eager_verify_seq_pre_verify_by_seq_id": {"3": reason == "seq_returned_pre_verify_before_verify"},
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_schedule_record(),
        synthetic_verify_record(4),
        synthetic_verify_record(2),
        synthetic_verify_skip_record("base_mismatch_before_verify"),
        synthetic_verify_skip_record("seq_returned_pre_verify_before_verify"),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager verify records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_verify_full_accept_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("full_accept mismatch" in error for error in errors), "checker missed full_accept mismatch"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_verify_mutation_detected_by_seq_id"] = {"3": True}
    errors, _ = validate_records(invalid)
    assert any("mutation detected" in error for error in errors), "checker missed mutation detection"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_verify_checkpoint_ok_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("checkpoint failed" in error for error in errors), "checker missed checkpoint failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager verification counters" in error for error in errors), (
        "checker missed actual eager verified counter"
    )

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_verify_accepted_len_by_seq_id"] = {"3": 5}
    errors, _ = validate_records(invalid)
    assert any("accepted_len out of range" in error for error in errors), "checker missed accepted_len range"

    print("Synthetic eager verify dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5a eager verify dry-run traces.")
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
    print("\nEager verify dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
