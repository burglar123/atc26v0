#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


EAGER_COUNTER_FIELDS = [
    "eager_tokens_generated",
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


def dict_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def nonzero_eager_counter_fields(record: dict[str, Any]) -> list[str]:
    return [
        field
        for field in EAGER_COUNTER_FIELDS
        if int_value(record.get(field), 0) != 0
    ]


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    candidate_count = 0
    selected_count = 0
    unique_selected: set[int] = set()
    selected_by_phase: Counter[str] = Counter()
    skipped_pre_verify_count = 0
    bad_overlap_count = 0
    nonzero_counter_rows = 0
    dry_run_records = 0

    for idx, record in enumerate(records):
        is_dual = (
            record.get("execution_mode") == "dual_batch_pearl"
            and record.get("dual_batch_enabled") is True
        )
        if not is_dual:
            continue

        dry_run = bool(record.get("enable_eager_plan_dry_run", False))
        if dry_run:
            dry_run_records += 1
        phase = record.get("plan_phase")
        target_home = as_int_set(record.get("target_home_set"))
        draft_home = as_int_set(record.get("draft_home_set"))
        target_eager = as_int_set(record.get("target_eager_set"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        eager_new_selected = as_int_set(record.get("eager_new_selected_set"))
        eager_continuing = as_int_set(record.get("eager_continuing_set"))
        eager_selected = as_int_set(record.get("eager_selected_seq_ids"))

        candidate_count += len(record.get("eager_candidate_seq_ids") or [])
        selected_count += len(draft_eager)
        unique_selected.update(draft_eager)
        if draft_eager:
            selected_by_phase[str(phase)] += len(draft_eager)
        skipped_pre_verify_count += len(record.get("eager_skipped_pre_verify_seq_ids") or [])

        nonzero_fields = nonzero_eager_counter_fields(record)
        if nonzero_fields:
            nonzero_counter_rows += 1
            errors.append(f"record[{idx}] nonzero eager counters: {nonzero_fields}")

        if not dry_run:
            if draft_eager:
                errors.append(f"record[{idx}] dry-run disabled but draft_eager_set is non-empty: {sorted(draft_eager)}")
            if target_eager:
                errors.append(f"record[{idx}] dry-run disabled but target_eager_set is non-empty: {sorted(target_eager)}")
            continue

        if target_eager:
            errors.append(f"record[{idx}] dry-run requires empty target_eager_set, got {sorted(target_eager)}")
        if eager_continuing:
            errors.append(f"record[{idx}] dry-run requires empty eager_continuing_set, got {sorted(eager_continuing)}")
        if not draft_eager <= target_home:
            bad_overlap_count += 1
            errors.append(
                f"record[{idx}] draft_eager_set must be subset of target_home_set: "
                f"extra={sorted(draft_eager - target_home)}"
            )
        if draft_eager & draft_home:
            bad_overlap_count += 1
            errors.append(
                f"record[{idx}] draft_eager_set intersects draft_home_set: {sorted(draft_eager & draft_home)}"
            )
        if eager_new_selected != draft_eager:
            errors.append(
                f"record[{idx}] eager_new_selected_set must equal draft_eager_set: "
                f"eager_new_selected={sorted(eager_new_selected)}, draft_eager={sorted(draft_eager)}"
            )
        if eager_selected != draft_eager:
            errors.append(
                f"record[{idx}] eager_selected_seq_ids must equal draft_eager_set: "
                f"eager_selected={sorted(eager_selected)}, draft_eager={sorted(draft_eager)}"
            )
        if draft_eager and phase != "steady":
            errors.append(f"record[{idx}] selected eager in non-steady phase={phase!r}")

        budgets = record.get("budgets", {})
        normal_gamma = int_value(record.get("normal_gamma"), None)
        eager_budget_by_seq_id = record.get("eager_budget_by_seq_id", {})
        base_pre_verify_by_seq_id = record.get("eager_base_pre_verify_by_seq_id", {})

        for seq_id in sorted(draft_eager):
            base_pre_verify = dict_get(base_pre_verify_by_seq_id, seq_id)
            if base_pre_verify is not False:
                errors.append(
                    f"record[{idx}] selected seq_id={seq_id} is not traced as post_verify: "
                    f"eager_base_pre_verify={base_pre_verify!r}"
                )

            budget = dict_get(budgets, seq_id, {})
            budget_normal_gamma = int_value(budget.get("normal_gamma") if isinstance(budget, dict) else None, normal_gamma)
            budget_eager_gamma = int_value(budget.get("eager_gamma") if isinstance(budget, dict) else None, None)
            eager_budget = int_value(dict_get(eager_budget_by_seq_id, seq_id), None)
            expected_gamma = normal_gamma if normal_gamma is not None else budget_normal_gamma
            if expected_gamma is None:
                errors.append(f"record[{idx}] cannot infer gamma for selected seq_id={seq_id}")
                continue
            if budget_normal_gamma != expected_gamma or budget_eager_gamma != expected_gamma or eager_budget != expected_gamma:
                errors.append(
                    f"record[{idx}] selected seq_id={seq_id} eager gamma mismatch: "
                    f"normal_gamma={budget_normal_gamma}, eager_gamma={budget_eager_gamma}, "
                    f"eager_budget={eager_budget}, expected={expected_gamma}"
                )

        if draft_eager and not bool(record.get("eager_post_verify_only", False)):
            errors.append(f"record[{idx}] selected eager but eager_post_verify_only is not true")
        if draft_eager and not bool(record.get("eager_gamma_equals_global_gamma", False)):
            errors.append(f"record[{idx}] selected eager but eager_gamma_equals_global_gamma is not true")

    summary = {
        "total_trace_records": len(records),
        "records_with_dry_run_enabled": dry_run_records,
        "eager_candidate_count": candidate_count,
        "selected_eager_count": selected_count,
        "unique_selected_eager_seq_ids": sorted(unique_selected),
        "selected_by_plan_phase": dict(selected_by_phase),
        "skipped_pre_verify_count": skipped_pre_verify_count,
        "bad_overlap_count": bad_overlap_count,
        "nonzero_eager_counter_rows": nonzero_counter_rows,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total_trace_records={summary['total_trace_records']}")
    print(f"records_with_dry_run_enabled={summary['records_with_dry_run_enabled']}")
    print(f"eager_candidate_count={summary['eager_candidate_count']}")
    print(f"selected_eager_count={summary['selected_eager_count']}")
    print(f"unique_selected_eager_seq_ids={summary['unique_selected_eager_seq_ids']}")
    print(f"selected_by_plan_phase={summary['selected_by_plan_phase']}")
    print(f"skipped_pre_verify_count={summary['skipped_pre_verify_count']}")
    print(f"bad_overlap_count={summary['bad_overlap_count']}")
    print(f"nonzero_eager_counter_rows={summary['nonzero_eager_counter_rows']}")


def synthetic_records() -> list[dict[str, Any]]:
    base = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": True,
        "target_home_set": [1, 2, 3],
        "draft_home_set": [4, 5],
        "target_eager_set": [],
        "draft_eager_set": [2],
        "eager_new_selected_set": [2],
        "eager_continuing_set": [],
        "eager_selected_seq_ids": [2],
        "normal_gamma": 4,
        "budgets": {
            "1": {"normal_gamma": 4, "eager_gamma": 0},
            "2": {"normal_gamma": 4, "eager_gamma": 4},
            "3": {"normal_gamma": 4, "eager_gamma": 0},
        },
        "eager_candidate_seq_ids": [1, 2, 3],
        "eager_candidate_reject_reason_by_seq_id": {"1": "pre_verify", "3": "non_tight"},
        "eager_budget_by_seq_id": {"2": 4},
        "eager_total_budget": 4,
        "eager_post_verify_only": True,
        "eager_gamma_equals_global_gamma": True,
        "eager_pre_verify_candidate_count": 1,
        "eager_post_verify_candidate_count": 2,
        "eager_skipped_pre_verify_seq_ids": [1],
        "eager_skipped_non_tight_seq_ids": [3],
        "eager_base_pre_verify_by_seq_id": {"2": False},
    }
    for field in EAGER_COUNTER_FIELDS:
        base[field] = 0

    default = deepcopy(base)
    default["enable_eager_plan_dry_run"] = False
    default["draft_eager_set"] = []
    default["eager_new_selected_set"] = []
    default["eager_selected_seq_ids"] = []
    default["eager_budget_by_seq_id"] = {}
    default["eager_total_budget"] = 0
    default["eager_base_pre_verify_by_seq_id"] = {}
    return [default, base]


def run_synthetic_tests() -> None:
    records = synthetic_records()
    errors, _ = validate_records(records)
    assert not errors, f"valid synthetic dry-run trace failed: {errors}"

    invalid = deepcopy(records)
    invalid[1]["plan_phase"] = "fallback"
    errors, _ = validate_records(invalid)
    assert any("non-steady" in error for error in errors), "checker missed fallback selection"

    invalid = deepcopy(records)
    invalid[1]["target_eager_set"] = [2]
    errors, _ = validate_records(invalid)
    assert any("target_eager_set" in error for error in errors), "checker missed target_eager_set"

    invalid = deepcopy(records)
    invalid[1]["eager_base_pre_verify_by_seq_id"] = {"2": True}
    errors, _ = validate_records(invalid)
    assert any("post_verify" in error for error in errors), "checker missed pre_verify selection"

    invalid = deepcopy(records)
    invalid[1]["eager_tokens_generated"] = 1
    errors, _ = validate_records(invalid)
    assert any("nonzero eager counters" in error for error in errors), "checker missed nonzero counter"
    print("Synthetic eager plan dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-1 eager plan dry-run traces.")
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
    print("\nEager plan dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
