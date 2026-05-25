#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_promoted",
    "eager_tokens_discarded",
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]


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


def is_dual_record(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    draft_enabled_records = 0
    eager_draft_seq_count = 0
    unique_draft_seq_ids: set[int] = set()
    total_generated = 0
    rollback_ok_count = 0
    rollback_failure_count = 0
    target_eager_non_empty_count = 0
    promoted_or_verified_rows = 0
    selected_by_phase: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        enable_plan_dry_run = bool(record.get("enable_eager_plan_dry_run", False))
        enable_draft_dry_run = bool(record.get("enable_eager_draft_dry_run", False))
        draft_dry_run_enabled = bool(record.get("eager_draft_dry_run_enabled", False))
        phase = record.get("plan_phase")
        target_home = as_int_set(record.get("target_home_set"))
        target_eager = as_int_set(record.get("target_eager_set"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        draft_seq_ids = as_int_set(record.get("eager_draft_seq_ids"))

        if draft_dry_run_enabled:
            draft_enabled_records += 1
        if target_eager:
            target_eager_non_empty_count += 1

        generated = int_value(record.get("eager_tokens_generated"), 0)
        dry_run_generated = int_value(record.get("eager_dry_run_tokens_generated"), 0)
        total_generated += generated
        nonzero_always_zero = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_always_zero:
            promoted_or_verified_rows += 1
            errors.append(f"record[{idx}] eager execution counters must stay zero: {nonzero_always_zero}")

        if target_eager:
            errors.append(f"record[{idx}] target_eager_set must remain empty, got {sorted(target_eager)}")
        if (enable_plan_dry_run or enable_draft_dry_run) and draft_eager:
            if phase != "steady":
                errors.append(f"record[{idx}] selected eager in non-steady phase={phase!r}")
            if not draft_eager <= target_home:
                errors.append(
                    f"record[{idx}] draft_eager_set must be subset of target_home_set: "
                    f"extra={sorted(draft_eager - target_home)}"
                )

        if not enable_plan_dry_run and not enable_draft_dry_run:
            if draft_eager:
                errors.append(f"record[{idx}] default run has non-empty draft_eager_set: {sorted(draft_eager)}")
            if generated or dry_run_generated:
                errors.append(f"record[{idx}] default run generated eager tokens")
            continue

        if enable_plan_dry_run and not enable_draft_dry_run:
            if generated or dry_run_generated:
                errors.append(f"record[{idx}] plan dry-run generated eager tokens")
            continue

        if enable_draft_dry_run and not enable_plan_dry_run:
            errors.append(f"record[{idx}] draft dry-run must imply plan dry-run")

        if not draft_dry_run_enabled:
            if generated or dry_run_generated:
                errors.append(f"record[{idx}] generated eager tokens outside an eager draft record")
            continue

        eager_draft_seq_count += len(draft_seq_ids)
        unique_draft_seq_ids.update(draft_seq_ids)
        if draft_seq_ids:
            selected_by_phase[str(phase)] += len(draft_seq_ids)

        if draft_seq_ids and generated <= 0:
            errors.append(f"record[{idx}] eager draft record has selected seqs but generated no eager tokens")
        if dry_run_generated != generated:
            errors.append(
                f"record[{idx}] eager_dry_run_tokens_generated must equal eager_tokens_generated: "
                f"{dry_run_generated} != {generated}"
            )
        proposal_id_list = as_int_list(record.get("eager_draft_proposal_ids"))
        if len(proposal_id_list) != len(draft_seq_ids):
            errors.append(
                f"record[{idx}] eager_draft_proposal_ids length must match eager_draft_seq_ids"
            )
        if len(set(proposal_id_list)) != len(proposal_id_list):
            errors.append(f"record[{idx}] eager_draft_proposal_ids must be unique")
        if int_value(record.get("eager_buffer_size_after"), 0) != 0:
            errors.append(f"record[{idx}] eager_buffer_size_after must be 0")
        if not draft_seq_ids <= draft_eager:
            errors.append(
                f"record[{idx}] eager_draft_seq_ids must be subset of draft_eager_set: "
                f"extra={sorted(draft_seq_ids - draft_eager)}"
            )

        gamma = int_value(record.get("normal_gamma"), None)
        budgets = record.get("budgets", {})
        base_pre_verify = record.get("eager_draft_base_pre_verify_by_seq_id") or record.get(
            "eager_base_pre_verify_by_seq_id",
            {},
        )
        proposal_len_by_seq_id = record.get("eager_draft_proposal_len_by_seq_id", {})
        to_verify_len_by_seq_id = record.get("eager_draft_to_verify_len_by_seq_id", {})
        rollback_ok_by_seq_id = record.get("eager_draft_rollback_ok_by_seq_id", {})
        discard_reason_by_seq_id = record.get("eager_draft_discard_reason_by_seq_id", {})
        if gamma is not None and generated != len(draft_seq_ids) * gamma:
            errors.append(
                f"record[{idx}] eager_tokens_generated must equal eager_draft_seq_count * gamma: "
                f"{generated} != {len(draft_seq_ids) * gamma}"
            )

        for seq_id in sorted(draft_seq_ids):
            if dict_get(base_pre_verify, seq_id) is not False:
                errors.append(f"record[{idx}] eager draft seq_id={seq_id} is not post_verify")

            budget = dict_get(budgets, seq_id, {})
            budget_gamma = int_value(
                budget.get("eager_gamma") if isinstance(budget, dict) else None,
                gamma,
            )
            expected_gamma = gamma if gamma is not None else budget_gamma
            if expected_gamma is None:
                errors.append(f"record[{idx}] cannot infer gamma for eager draft seq_id={seq_id}")
                continue
            if budget_gamma != expected_gamma:
                errors.append(
                    f"record[{idx}] eager draft seq_id={seq_id} eager_gamma mismatch: "
                    f"expected={expected_gamma}, got={budget_gamma}"
                )
            if int_value(dict_get(to_verify_len_by_seq_id, seq_id), -1) != expected_gamma:
                errors.append(
                    f"record[{idx}] eager draft seq_id={seq_id} to_be_verified length mismatch"
                )
            if int_value(dict_get(proposal_len_by_seq_id, seq_id), -1) != expected_gamma:
                errors.append(f"record[{idx}] eager draft seq_id={seq_id} proposal length mismatch")
            if dict_get(rollback_ok_by_seq_id, seq_id) is True:
                rollback_ok_count += 1
            else:
                rollback_failure_count += 1
                errors.append(f"record[{idx}] eager draft seq_id={seq_id} rollback_ok is not true")
            if dict_get(discard_reason_by_seq_id, seq_id) != "phase1h2_dry_run_rollback":
                errors.append(f"record[{idx}] eager draft seq_id={seq_id} missing dry-run discard reason")

    summary = {
        "total_trace_records": len(records),
        "records_with_eager_draft_dry_run_enabled": draft_enabled_records,
        "eager_draft_seq_count": eager_draft_seq_count,
        "unique_eager_draft_seq_ids": sorted(unique_draft_seq_ids),
        "eager_tokens_generated": total_generated,
        "rollback_ok_count": rollback_ok_count,
        "rollback_failure_count": rollback_failure_count,
        "target_eager_non_empty_count": target_eager_non_empty_count,
        "promoted_verified_counter_rows": promoted_or_verified_rows,
        "selected_by_plan_phase": dict(selected_by_phase),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total_trace_records={summary['total_trace_records']}")
    print(f"records_with_eager_draft_dry_run_enabled={summary['records_with_eager_draft_dry_run_enabled']}")
    print(f"eager_draft_seq_count={summary['eager_draft_seq_count']}")
    print(f"unique_eager_draft_seq_ids={summary['unique_eager_draft_seq_ids']}")
    print(f"eager_tokens_generated={summary['eager_tokens_generated']}")
    print(f"rollback_ok_count={summary['rollback_ok_count']}")
    print(f"rollback_failure_count={summary['rollback_failure_count']}")
    print(f"target_eager_non_empty_count={summary['target_eager_non_empty_count']}")
    print(f"promoted_verified_counter_rows={summary['promoted_verified_counter_rows']}")
    print(f"selected_by_plan_phase={summary['selected_by_plan_phase']}")


def synthetic_default_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "eager_draft_dry_run_enabled": False,
        "target_home_set": [1],
        "draft_home_set": [2],
        "target_eager_set": [],
        "draft_eager_set": [],
        "eager_draft_seq_ids": [],
        "normal_gamma": 4,
        "budgets": {"1": {"normal_gamma": 4, "eager_gamma": 0}},
        "eager_tokens_generated": 0,
        "eager_dry_run_tokens_generated": 0,
        "eager_buffer_size_after": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_plan_record() -> dict[str, Any]:
    record = synthetic_default_record()
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "target_home_set": [1, 3],
            "draft_eager_set": [1],
            "eager_new_selected_set": [1],
            "eager_selected_seq_ids": [1],
            "budgets": {
                "1": {"normal_gamma": 4, "eager_gamma": 4},
                "3": {"normal_gamma": 4, "eager_gamma": 0},
            },
            "eager_base_pre_verify_by_seq_id": {"1": False},
        }
    )
    return record


def synthetic_draft_record() -> dict[str, Any]:
    record = synthetic_plan_record()
    record.update(
        {
            "enable_eager_draft_dry_run": True,
            "eager_draft_dry_run_enabled": True,
            "eager_draft_seq_ids": [1],
            "eager_draft_proposal_ids": [101],
            "eager_draft_base_len_by_seq_id": {"1": 12},
            "eager_draft_base_pre_verify_by_seq_id": {"1": False},
            "eager_draft_to_verify_len_by_seq_id": {"1": 4},
            "eager_draft_proposal_len_by_seq_id": {"1": 4},
            "eager_draft_rollback_seq_ids": [1],
            "eager_draft_rollback_ok_by_seq_id": {"1": True},
            "eager_draft_discard_reason_by_seq_id": {"1": "phase1h2_dry_run_rollback"},
            "eager_tokens_generated": 4,
            "eager_dry_run_tokens_generated": 4,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_default_record(),
        synthetic_plan_record(),
        synthetic_draft_record(),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager draft dry-run trace failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [1]
    errors, _ = validate_records(invalid)
    assert any("target_eager_set" in error for error in errors), "checker missed target_eager_set"

    invalid = deepcopy(valid_records)
    invalid[2]["plan_phase"] = "fallback"
    errors, _ = validate_records(invalid)
    assert any("non-steady" in error for error in errors), "checker missed fallback eager draft"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_draft_base_pre_verify_by_seq_id"] = {"1": True}
    errors, _ = validate_records(invalid)
    assert any("not post_verify" in error for error in errors), "checker missed pre_verify eager draft"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_draft_rollback_ok_by_seq_id"] = {"1": False}
    errors, _ = validate_records(invalid)
    assert any("rollback_ok" in error for error in errors), "checker missed rollback failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_draft_to_verify_len_by_seq_id"] = {"1": 3}
    errors, _ = validate_records(invalid)
    assert any("to_be_verified length" in error for error in errors), "checker missed to_verify length mismatch"

    invalid = deepcopy(valid_records)
    invalid[2]["budgets"]["1"]["eager_gamma"] = 2
    errors, _ = validate_records(invalid)
    assert any("eager_gamma mismatch" in error for error in errors), "checker missed eager_gamma mismatch"

    print("Synthetic eager draft dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-2 eager draft dry-run traces.")
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
    print("\nEager draft dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
