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

EXPECTED_CLEAR_REASON = "phase1h4c_schedule_dry_run_no_verify_yet"


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


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_schedule_enabled = 0
    ready_proposal_ids: set[int] = set()
    scheduled_proposal_ids_seen: set[int] = set()
    scheduled_seq_ids_seen: set[int] = set()
    scheduled_target_eager_records = 0
    target_eager_set_dry_run_count = 0
    target_home_intersection_count = 0
    draft_home_intersection_count = 0
    base_mismatch_count = 0
    pre_verify_scheduled_count = 0
    verified_counter_rows = 0
    eager_tokens_scheduled = 0
    skip_reason_counts: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        schedule_enabled = bool(record.get("enable_eager_schedule_dry_run", False))
        schedule_active = bool(record.get("eager_schedule_dry_run_enabled", False))
        plan_enabled = bool(record.get("enable_eager_plan_dry_run", False))
        draft_enabled = bool(record.get("enable_eager_draft_dry_run", False))
        promotion_enabled = bool(record.get("enable_eager_promotion_dry_run", False))
        transfer_enabled = bool(record.get("enable_eager_transfer_dry_run", False))
        phase = record.get("plan_phase")
        target_home = as_int_set(record.get("target_home_set"))
        draft_home = as_int_set(record.get("draft_home_set"))
        target_eager = as_int_set(record.get("target_eager_set"))
        target_eager_dry_run = as_int_set(record.get("target_eager_set_dry_run"))
        candidate_proposal_ids = as_int_set(record.get("eager_schedule_candidate_proposal_ids"))
        scheduled_proposal_ids = as_int_set(record.get("eager_scheduled_proposal_ids"))
        scheduled_seq_ids = as_int_set(record.get("eager_scheduled_seq_ids"))
        skipped_proposal_ids = as_int_set(record.get("eager_schedule_skipped_proposal_ids"))
        skip_reasons = record.get("eager_schedule_skip_reason_by_proposal_id", {})
        proposal_id_by_seq = record.get("eager_schedule_proposal_id_by_seq_id", {})
        base_match_by_seq = record.get("eager_schedule_base_match_by_seq_id", {})
        seq_pre_verify_by_seq = record.get("eager_schedule_seq_pre_verify_by_seq_id", {})
        base_pre_verify_by_proposal = record.get("eager_schedule_base_pre_verify_by_proposal_id", {})
        proposal_len_by_proposal = record.get("eager_schedule_proposal_len_by_proposal_id", {})
        to_verify_len_by_proposal = record.get("eager_schedule_to_verify_len_by_proposal_id", {})
        clear_reason_by_proposal = record.get("eager_schedule_clear_reason_by_proposal_id", {})
        gamma = int_value(record.get("normal_gamma"), None)
        schedule_candidates_tokens = int_value(record.get("eager_tokens_schedule_candidates"), 0)
        schedule_tokens = int_value(record.get("eager_tokens_scheduled_dry_run"), 0)

        if target_eager:
            errors.append(f"record[{idx}] target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_verified = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_verified:
            verified_counter_rows += 1
            errors.append(f"record[{idx}] eager verification counters must stay zero: {nonzero_verified}")

        if schedule_enabled:
            records_with_schedule_enabled += 1
            if not (plan_enabled and draft_enabled and promotion_enabled and transfer_enabled):
                errors.append(
                    f"record[{idx}] schedule dry-run must imply plan/draft/promotion/transfer dry-runs"
                )
        else:
            if (
                schedule_active
                or target_eager_dry_run
                or candidate_proposal_ids
                or scheduled_proposal_ids
                or scheduled_seq_ids
                or skipped_proposal_ids
                or schedule_candidates_tokens
                or schedule_tokens
            ):
                errors.append(f"record[{idx}] schedule dry-run fields populated while schedule is disabled")
            continue

        ready_proposal_ids.update(candidate_proposal_ids)
        scheduled_proposal_ids_seen.update(scheduled_proposal_ids)
        scheduled_seq_ids_seen.update(scheduled_seq_ids)
        eager_tokens_scheduled += schedule_tokens

        for proposal_id in skipped_proposal_ids:
            reason = dict_get(skip_reasons, proposal_id)
            skip_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing skip reason")

        missing_classification = candidate_proposal_ids - scheduled_proposal_ids - skipped_proposal_ids
        if missing_classification:
            errors.append(
                f"record[{idx}] candidate proposal ids missing scheduled/skipped classification: "
                f"{sorted(missing_classification)}"
            )
        overlap = scheduled_proposal_ids & skipped_proposal_ids
        if overlap:
            errors.append(f"record[{idx}] proposal ids both scheduled and skipped: {sorted(overlap)}")

        if target_eager_dry_run:
            scheduled_target_eager_records += 1
            target_eager_set_dry_run_count += len(target_eager_dry_run)
            if not schedule_active:
                errors.append(f"record[{idx}] target_eager_set_dry_run populated without active schedule row")
            if phase != "steady":
                errors.append(f"record[{idx}] target_eager_set_dry_run populated in non-steady phase={phase!r}")
            if schedule_tokens <= 0:
                errors.append(f"record[{idx}] scheduled dry-run seqs require positive scheduled token count")

        if target_eager_dry_run != scheduled_seq_ids:
            errors.append(
                f"record[{idx}] target_eager_set_dry_run must match eager_scheduled_seq_ids: "
                f"target={sorted(target_eager_dry_run)}, scheduled={sorted(scheduled_seq_ids)}"
            )
        if scheduled_seq_ids and not scheduled_proposal_ids:
            errors.append(f"record[{idx}] scheduled seqs missing scheduled proposal ids")
        if scheduled_proposal_ids and not scheduled_seq_ids:
            errors.append(f"record[{idx}] scheduled proposals missing scheduled seq ids")
        if schedule_tokens and not scheduled_proposal_ids:
            errors.append(f"record[{idx}] eager_tokens_scheduled_dry_run set without scheduled proposals")
        if schedule_candidates_tokens and not candidate_proposal_ids:
            errors.append(f"record[{idx}] schedule candidate tokens set without candidate proposals")

        target_intersection = target_eager_dry_run & target_home
        draft_intersection = target_eager_dry_run & draft_home
        if target_intersection or bool(record.get("eager_schedule_intersects_target_home", False)):
            target_home_intersection_count += len(target_intersection) or 1
            errors.append(
                f"record[{idx}] target_eager_set_dry_run intersects target_home_set: "
                f"{sorted(target_intersection)}"
            )
        if draft_intersection or bool(record.get("eager_schedule_intersects_draft_home", False)):
            draft_home_intersection_count += len(draft_intersection) or 1
            errors.append(
                f"record[{idx}] target_eager_set_dry_run intersects draft_home_set: "
                f"{sorted(draft_intersection)}"
            )

        for seq_id in sorted(scheduled_seq_ids):
            proposal_id = dict_get(proposal_id_by_seq, seq_id)
            if proposal_id is None:
                errors.append(f"record[{idx}] scheduled seq_id={seq_id} missing proposal id")
                continue
            proposal_id = int(proposal_id)
            if proposal_id not in scheduled_proposal_ids:
                errors.append(
                    f"record[{idx}] scheduled seq_id={seq_id} proposal_id={proposal_id} "
                    "not in eager_scheduled_proposal_ids"
                )
            if proposal_id not in candidate_proposal_ids:
                errors.append(
                    f"record[{idx}] scheduled proposal_id={proposal_id} not in schedule candidates"
                )
            if dict_get(base_match_by_seq, seq_id) is not True:
                base_mismatch_count += 1
                errors.append(f"record[{idx}] scheduled seq_id={seq_id} lacks base_len match")
            if dict_get(seq_pre_verify_by_seq, seq_id) is not False:
                pre_verify_scheduled_count += 1
                errors.append(f"record[{idx}] scheduled seq_id={seq_id} has pre_verify=true")
            if dict_get(base_pre_verify_by_proposal, proposal_id) is not False:
                errors.append(f"record[{idx}] scheduled proposal_id={proposal_id} base_pre_verify is not false")
            if gamma is not None and int_value(dict_get(proposal_len_by_proposal, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] scheduled proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and int_value(dict_get(to_verify_len_by_proposal, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] scheduled proposal_id={proposal_id} to_verify_len != gamma")
            clear_reason = dict_get(clear_reason_by_proposal, proposal_id)
            if clear_reason not in (None, EXPECTED_CLEAR_REASON):
                errors.append(
                    f"record[{idx}] scheduled proposal_id={proposal_id} has bad clear reason={clear_reason!r}"
                )

        for proposal_id in sorted(scheduled_proposal_ids):
            clear_reason = dict_get(clear_reason_by_proposal, proposal_id)
            if clear_reason not in (None, EXPECTED_CLEAR_REASON):
                errors.append(
                    f"record[{idx}] scheduled proposal_id={proposal_id} has bad clear reason={clear_reason!r}"
                )

    summary = {
        "total_trace_records": len(records),
        "records_with_schedule_dry_run_enabled": records_with_schedule_enabled,
        "ready_proposal_count": len(ready_proposal_ids),
        "scheduled_target_eager_records": scheduled_target_eager_records,
        "unique_scheduled_proposal_ids": len(scheduled_proposal_ids_seen),
        "unique_scheduled_seq_ids": len(scheduled_seq_ids_seen),
        "target_eager_set_dry_run_count": target_eager_set_dry_run_count,
        "target_home_intersection_count": target_home_intersection_count,
        "draft_home_intersection_count": draft_home_intersection_count,
        "base_mismatch_count": base_mismatch_count,
        "pre_verify_scheduled_count": pre_verify_scheduled_count,
        "eager_tokens_scheduled_dry_run": eager_tokens_scheduled,
        "eager_verified_counter_rows": verified_counter_rows,
        "schedule_skip_reason_counts": dict(skip_reason_counts),
        "pending_coverage_self_test_status": "not_run",
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_schedule_dry_run_enabled",
        "ready_proposal_count",
        "scheduled_target_eager_records",
        "unique_scheduled_proposal_ids",
        "unique_scheduled_seq_ids",
        "target_eager_set_dry_run_count",
        "target_home_intersection_count",
        "draft_home_intersection_count",
        "base_mismatch_count",
        "pre_verify_scheduled_count",
        "eager_tokens_scheduled_dry_run",
        "eager_verified_counter_rows",
        "schedule_skip_reason_counts",
        "pending_coverage_self_test_status",
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
        "eager_schedule_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_home_set": [1],
        "draft_home_set": [2],
        "target_eager_set": [],
        "target_eager_set_dry_run": [],
        "draft_eager_set": [],
        "eager_schedule_candidate_proposal_ids": [],
        "eager_schedule_candidate_seq_ids": [],
        "eager_scheduled_proposal_ids": [],
        "eager_scheduled_seq_ids": [],
        "eager_schedule_skipped_proposal_ids": [],
        "eager_schedule_skip_reason_by_proposal_id": {},
        "eager_schedule_clear_reason_by_proposal_id": {},
        "eager_schedule_proposal_id_by_seq_id": {},
        "eager_schedule_base_len_by_seq_id": {},
        "eager_schedule_current_len_by_seq_id": {},
        "eager_schedule_base_match_by_seq_id": {},
        "eager_schedule_seq_pre_verify_by_seq_id": {},
        "eager_schedule_base_pre_verify_by_proposal_id": {},
        "eager_schedule_proposal_len_by_proposal_id": {},
        "eager_schedule_to_verify_len_by_proposal_id": {},
        "eager_schedule_intersects_target_home": False,
        "eager_schedule_intersects_draft_home": False,
        "eager_tokens_schedule_candidates": 0,
        "eager_tokens_scheduled_dry_run": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_transfer_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "eager_transfer_dry_run_enabled": True,
            "eager_tokens_generated": 4,
            "eager_tokens_promoted": 4,
            "eager_tokens_discarded": 0,
            "eager_tokens_transferred": 4,
            "eager_tokens_transfer_validated": 4,
            "eager_tokens_transfer_pending": 0,
            "eager_tokens_transfer_dropped": 0,
            "eager_transfer_validated_proposal_ids": [101],
            "eager_transfer_pending_proposal_ids": [],
            "eager_transfer_dropped_proposal_ids": [],
            "eager_transfer_base_len_by_seq_id": {"3": 12},
            "eager_transfer_current_len_by_seq_id": {"3": 12},
            "eager_transfer_base_match_by_seq_id": {"3": True},
            "eager_transfer_base_pre_verify_by_seq_id": {"3": False},
            "eager_transfer_classification_by_proposal_id": {"101": "base_reached"},
        }
    )
    return record


def synthetic_pending_record() -> dict[str, Any]:
    record = synthetic_transfer_record()
    record.update(
        {
            "eager_transfer_validated_proposal_ids": [],
            "eager_transfer_pending_proposal_ids": [101],
            "eager_transfer_base_len_by_seq_id": {"3": 12},
            "eager_transfer_current_len_by_seq_id": {"3": 11},
            "eager_transfer_base_match_by_seq_id": {"3": False},
            "eager_transfer_classification_by_proposal_id": {"101": "pending_base_not_reached"},
            "eager_tokens_transfer_validated": 0,
            "eager_tokens_transfer_pending": 4,
        }
    )
    return record


def synthetic_schedule_record() -> dict[str, Any]:
    record = synthetic_transfer_record()
    record.update(
        {
            "enable_eager_schedule_dry_run": True,
            "eager_schedule_dry_run_enabled": True,
            "target_home_set": [1],
            "draft_home_set": [2],
            "target_eager_set_dry_run": [3],
            "eager_schedule_candidate_proposal_ids": [101],
            "eager_schedule_candidate_seq_ids": [3],
            "eager_scheduled_proposal_ids": [101],
            "eager_scheduled_seq_ids": [3],
            "eager_schedule_skipped_proposal_ids": [],
            "eager_schedule_skip_reason_by_proposal_id": {},
            "eager_schedule_clear_reason_by_proposal_id": {
                "101": EXPECTED_CLEAR_REASON,
            },
            "eager_schedule_proposal_id_by_seq_id": {"3": 101},
            "eager_schedule_base_len_by_seq_id": {"3": 12},
            "eager_schedule_current_len_by_seq_id": {"3": 12},
            "eager_schedule_base_match_by_seq_id": {"3": True},
            "eager_schedule_seq_pre_verify_by_seq_id": {"3": False},
            "eager_schedule_base_pre_verify_by_proposal_id": {"101": False},
            "eager_schedule_proposal_len_by_proposal_id": {"101": 4},
            "eager_schedule_to_verify_len_by_proposal_id": {"101": 4},
            "eager_tokens_schedule_candidates": 4,
            "eager_tokens_scheduled_dry_run": 4,
        }
    )
    return record


def synthetic_schedule_skip_record(reason: str = "base_not_reached") -> dict[str, Any]:
    record = synthetic_schedule_record()
    record.update(
        {
            "target_eager_set_dry_run": [],
            "eager_scheduled_proposal_ids": [],
            "eager_scheduled_seq_ids": [],
            "eager_schedule_skipped_proposal_ids": [101],
            "eager_schedule_skip_reason_by_proposal_id": {"101": reason},
            "eager_tokens_schedule_candidates": 4,
            "eager_tokens_scheduled_dry_run": 0,
        }
    )
    return record


def synthetic_transfer_drop_record(reason: str, current_len: int = 11) -> dict[str, Any]:
    record = synthetic_transfer_record()
    record.update(
        {
            "eager_transfer_validated_proposal_ids": [],
            "eager_transfer_pending_proposal_ids": [],
            "eager_transfer_dropped_proposal_ids": [101],
            "eager_transfer_drop_reason_by_proposal_id": {"101": reason},
            "eager_transfer_base_len_by_seq_id": {"3": 12},
            "eager_transfer_current_len_by_seq_id": {"3": current_len},
            "eager_transfer_base_match_by_seq_id": {"3": current_len == 12},
            "eager_transfer_classification_by_proposal_id": {"101": reason},
            "eager_tokens_transfer_validated": 0,
            "eager_tokens_transfer_dropped": 4,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_transfer_record(),
        synthetic_pending_record(),
        synthetic_transfer_drop_record("seq_returned_pre_verify_before_base"),
        synthetic_transfer_drop_record("seq_span_invalidated_before_base"),
        synthetic_transfer_drop_record("seq_finished_before_base"),
        synthetic_transfer_drop_record("base_overshot_or_stale", current_len=13),
        synthetic_schedule_skip_record("base_not_reached"),
        synthetic_schedule_record(),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager schedule records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[-1]["target_home_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("intersects target_home_set" in error for error in errors), (
        "checker missed target_home intersection"
    )

    invalid = deepcopy(valid_records)
    invalid[-1]["draft_home_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("intersects draft_home_set" in error for error in errors), (
        "checker missed draft_home intersection"
    )

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_schedule_base_match_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("base_len match" in error for error in errors), "checker missed base mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_schedule_seq_pre_verify_by_seq_id"] = {"3": True}
    errors, _ = validate_records(invalid)
    assert any("pre_verify=true" in error for error in errors), "checker missed pre_verify schedule"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_schedule_proposal_len_by_proposal_id"] = {"101": 3}
    errors, _ = validate_records(invalid)
    assert any("proposal_len != gamma" in error for error in errors), (
        "checker missed proposal_len mismatch"
    )

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_schedule_to_verify_len_by_proposal_id"] = {"101": 3}
    errors, _ = validate_records(invalid)
    assert any("to_verify_len != gamma" in error for error in errors), (
        "checker missed to_verify_len mismatch"
    )

    invalid = deepcopy(valid_records)
    invalid[-1]["enable_eager_schedule_dry_run"] = False
    errors, _ = validate_records(invalid)
    assert any("schedule dry-run fields populated while schedule is disabled" in error for error in errors), (
        "checker missed schedule fields while disabled"
    )

    print("Synthetic eager schedule dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-4c eager schedule dry-run traces.")
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
    print("\nEager schedule dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
