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

DISCARD_REASONS = {
    "parent_rejected",
    "parent_partial_accept",
    "parent_finished",
    "parent_pre_verify_after_apply",
    "base_len_mismatch",
    "missing_parent_result",
    "selected_seq_not_running",
    "selected_seq_pre_verify",
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
    seq_ids = as_int_list(record.get("eager_draft_seq_ids"))
    proposal_ids = as_int_list(record.get("eager_draft_proposal_ids"))
    return {
        proposal_id: seq_id
        for proposal_id, seq_id in zip(proposal_ids, seq_ids)
    }


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    promotion_records = 0
    generated_tokens = 0
    promoted_tokens = 0
    discarded_tokens = 0
    unique_promoted_proposals: set[int] = set()
    unique_discarded_proposals: set[int] = set()
    promotion_reason_counts: Counter[str] = Counter()
    discard_reason_counts: Counter[str] = Counter()
    base_mismatch_count = 0
    target_eager_non_empty_count = 0
    verified_counter_rows = 0
    selected_by_phase: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        plan_enabled = bool(record.get("enable_eager_plan_dry_run", False))
        draft_enabled = bool(record.get("enable_eager_draft_dry_run", False))
        promotion_enabled = bool(record.get("enable_eager_promotion_dry_run", False))
        transfer_enabled = bool(record.get("enable_eager_transfer_dry_run", False))
        promotion_active = bool(record.get("eager_promotion_dry_run_enabled", False))
        phase = record.get("plan_phase")
        target_home = as_int_set(record.get("target_home_set"))
        target_eager = as_int_set(record.get("target_eager_set"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        draft_seq_ids = as_int_set(record.get("eager_draft_seq_ids"))
        promoted_seq_ids = as_int_set(record.get("eager_promoted_seq_ids"))
        discarded_seq_ids = as_int_set(record.get("eager_discarded_seq_ids"))
        promoted_proposal_ids = as_int_set(record.get("eager_promoted_proposal_ids"))
        discarded_proposal_ids = as_int_set(record.get("eager_discarded_proposal_ids"))

        if target_eager:
            target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_verified = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_verified:
            verified_counter_rows += 1
            errors.append(f"record[{idx}] eager verified/accepted/rejected counters must stay zero: {nonzero_verified}")

        generated = int_value(record.get("eager_tokens_generated"), 0)
        promoted = int_value(record.get("eager_tokens_promoted"), 0)
        discarded = int_value(record.get("eager_tokens_discarded"), 0)
        transferred = int_value(record.get("eager_tokens_transferred"), 0)
        transfer_pending = int_value(record.get("eager_tokens_transfer_pending"), 0)
        transfer_validated = int_value(record.get("eager_tokens_transfer_validated"), 0)
        transfer_dropped = int_value(record.get("eager_tokens_transfer_dropped"), 0)
        generated_tokens += generated
        promoted_tokens += promoted
        discarded_tokens += discarded

        if not plan_enabled and not draft_enabled and not promotion_enabled:
            if draft_eager or target_eager:
                errors.append(f"record[{idx}] default run has non-empty eager sets")
            if generated or promoted or discarded or transferred or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] default run has nonzero eager counters")
            continue

        if plan_enabled and not draft_enabled and not promotion_enabled:
            if generated or promoted or discarded or transferred or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] plan dry-run must not generate/promote/discard eager tokens")
            continue

        if draft_enabled and not promotion_enabled:
            if promoted or discarded:
                errors.append(f"record[{idx}] draft dry-run must not promote/discard eager tokens")
            if transferred or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] draft dry-run must not transfer eager tokens")
            if draft_seq_ids and generated <= 0:
                errors.append(f"record[{idx}] draft dry-run selected eager but generated no eager tokens")
            continue

        if promotion_enabled:
            if not plan_enabled or not draft_enabled:
                errors.append(f"record[{idx}] promotion dry-run must imply plan and draft dry-run")
            if promotion_active:
                promotion_records += 1
            if (transferred or transfer_pending or transfer_validated or transfer_dropped) and not transfer_enabled:
                errors.append(f"record[{idx}] transfer counters require transfer dry-run")

        if draft_eager:
            selected_by_phase[str(phase)] += len(draft_eager)
            if phase != "steady":
                errors.append(f"record[{idx}] selected eager in non-steady phase={phase!r}")
            if not draft_eager <= target_home:
                errors.append(
                    f"record[{idx}] draft_eager_set must be subset of target_home_set: "
                    f"extra={sorted(draft_eager - target_home)}"
                )

        if not promotion_active:
            if promoted or discarded:
                errors.append(f"record[{idx}] promoted/discarded tokens outside active promotion dry-run row")
            continue

        proposal_by_seq = {
            seq_id: proposal_id
            for proposal_id, seq_id in proposal_seq_map(record).items()
        }
        unique_promoted_proposals.update(promoted_proposal_ids)
        unique_discarded_proposals.update(discarded_proposal_ids)

        if promoted + discarded != generated:
            errors.append(
                f"record[{idx}] eager_tokens_promoted + eager_tokens_discarded must equal "
                f"eager_tokens_generated: {promoted} + {discarded} != {generated}"
            )
        if int_value(record.get("eager_buffer_size_after"), 0) != 0:
            errors.append(f"record[{idx}] eager_buffer_size_after must be 0")
        if promoted_seq_ids & discarded_seq_ids:
            errors.append(f"record[{idx}] seq ids both promoted and discarded: {sorted(promoted_seq_ids & discarded_seq_ids)}")
        if promoted_proposal_ids & discarded_proposal_ids:
            errors.append(
                f"record[{idx}] proposal ids both promoted and discarded: "
                f"{sorted(promoted_proposal_ids & discarded_proposal_ids)}"
            )

        parent_full_accept = record.get("eager_parent_full_accept_by_seq_id", {})
        base_match = record.get("eager_promotion_base_match_by_seq_id", {})
        base_pre_verify = record.get("eager_draft_base_pre_verify_by_seq_id") or record.get(
            "eager_base_pre_verify_by_seq_id",
            {},
        )
        promotion_reasons = record.get("eager_promotion_reason_by_seq_id", {})
        discard_reasons = record.get("eager_discard_reason_by_seq_id", {})

        for seq_id in sorted(promoted_seq_ids):
            if dict_get(parent_full_accept, seq_id) is not True:
                errors.append(f"record[{idx}] promoted seq_id={seq_id} without parent_full_accept=true")
            if dict_get(base_match, seq_id) is not True:
                base_mismatch_count += 1
                errors.append(f"record[{idx}] promoted seq_id={seq_id} without base_len match")
            if dict_get(base_pre_verify, seq_id) is not False:
                errors.append(f"record[{idx}] promoted seq_id={seq_id} was pre_verify at selection")
            reason = dict_get(promotion_reasons, seq_id)
            promotion_reason_counts[str(reason)] += 1
            if reason != "parent_normal_full_accept":
                errors.append(f"record[{idx}] promoted seq_id={seq_id} has bad reason={reason!r}")
            if seq_id not in proposal_by_seq:
                errors.append(f"record[{idx}] promoted seq_id={seq_id} missing draft proposal id")

        for seq_id in sorted(discarded_seq_ids):
            reason = dict_get(discard_reasons, seq_id)
            discard_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] discarded seq_id={seq_id} missing discard reason")
            elif reason not in DISCARD_REASONS:
                errors.append(f"record[{idx}] discarded seq_id={seq_id} has invalid reason={reason!r}")
            if dict_get(base_match, seq_id) is False:
                base_mismatch_count += 1
            if seq_id not in proposal_by_seq:
                errors.append(f"record[{idx}] discarded seq_id={seq_id} missing draft proposal id")

        classified_proposal_ids = promoted_proposal_ids | discarded_proposal_ids
        draft_proposal_ids = set(proposal_by_seq.values())
        if classified_proposal_ids != draft_proposal_ids:
            errors.append(
                f"record[{idx}] promoted/discarded proposal ids must classify every draft proposal: "
                f"classified={sorted(classified_proposal_ids)}, draft={sorted(draft_proposal_ids)}"
            )

    summary = {
        "total_trace_records": len(records),
        "records_with_promotion_dry_run_enabled": promotion_records,
        "eager_generated_tokens": generated_tokens,
        "eager_promoted_tokens": promoted_tokens,
        "eager_discarded_tokens": discarded_tokens,
        "unique_promoted_proposal_ids": sorted(unique_promoted_proposals),
        "unique_discarded_proposal_ids": sorted(unique_discarded_proposals),
        "promotion_reason_counts": dict(promotion_reason_counts),
        "discard_reason_counts": dict(discard_reason_counts),
        "base_mismatch_count": base_mismatch_count,
        "target_eager_non_empty_count": target_eager_non_empty_count,
        "verified_counter_rows": verified_counter_rows,
        "selected_by_plan_phase": dict(selected_by_phase),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total_trace_records={summary['total_trace_records']}")
    print(f"records_with_promotion_dry_run_enabled={summary['records_with_promotion_dry_run_enabled']}")
    print(f"eager_generated_tokens={summary['eager_generated_tokens']}")
    print(f"eager_promoted_tokens={summary['eager_promoted_tokens']}")
    print(f"eager_discarded_tokens={summary['eager_discarded_tokens']}")
    print(f"unique_promoted_proposal_ids={summary['unique_promoted_proposal_ids']}")
    print(f"unique_discarded_proposal_ids={summary['unique_discarded_proposal_ids']}")
    print(f"promotion_reason_counts={summary['promotion_reason_counts']}")
    print(f"discard_reason_counts={summary['discard_reason_counts']}")
    print(f"base_mismatch_count={summary['base_mismatch_count']}")
    print(f"target_eager_non_empty_count={summary['target_eager_non_empty_count']}")
    print(f"verified_counter_rows={summary['verified_counter_rows']}")
    print(f"selected_by_plan_phase={summary['selected_by_plan_phase']}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "eager_promotion_dry_run_enabled": False,
        "target_home_set": [1],
        "draft_home_set": [2],
        "target_eager_set": [],
        "draft_eager_set": [],
        "eager_draft_seq_ids": [],
        "eager_draft_proposal_ids": [],
        "normal_gamma": 4,
        "budgets": {"1": {"normal_gamma": 4, "eager_gamma": 0}},
        "eager_tokens_generated": 0,
        "eager_tokens_promoted": 0,
        "eager_tokens_discarded": 0,
        "eager_buffer_size_after": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_plan_record() -> dict[str, Any]:
    record = synthetic_base_record()
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
            "eager_tokens_generated": 4,
        }
    )
    return record


def synthetic_promotion_record(reason: str = "promote") -> dict[str, Any]:
    record = synthetic_draft_record()
    record.update(
        {
            "enable_eager_promotion_dry_run": True,
            "eager_promotion_dry_run_enabled": True,
            "eager_parent_seq_ids": [1],
            "eager_parent_accepted_len_by_seq_id": {"1": 4},
            "eager_parent_invalidated_len_by_seq_id": {"1": 0},
            "eager_parent_full_accept_by_seq_id": {"1": True},
            "eager_parent_finished_by_seq_id": {"1": False},
            "eager_promotion_base_len_by_seq_id": {"1": 12},
            "eager_promotion_current_len_by_seq_id": {"1": 12},
            "eager_promotion_base_match_by_seq_id": {"1": True},
            "eager_promoted_seq_ids": [1],
            "eager_promoted_proposal_ids": [101],
            "eager_discarded_seq_ids": [],
            "eager_discarded_proposal_ids": [],
            "eager_promotion_reason_by_seq_id": {"1": "parent_normal_full_accept"},
            "eager_discard_reason_by_seq_id": {},
            "eager_tokens_promoted": 4,
            "eager_tokens_discarded": 0,
        }
    )
    if reason == "reject":
        record.update(
            {
                "eager_parent_accepted_len_by_seq_id": {"1": 0},
                "eager_parent_invalidated_len_by_seq_id": {"1": 4},
                "eager_parent_full_accept_by_seq_id": {"1": False},
                "eager_promoted_seq_ids": [],
                "eager_promoted_proposal_ids": [],
                "eager_discarded_seq_ids": [1],
                "eager_discarded_proposal_ids": [101],
                "eager_promotion_reason_by_seq_id": {},
                "eager_discard_reason_by_seq_id": {"1": "parent_rejected"},
                "eager_tokens_promoted": 0,
                "eager_tokens_discarded": 4,
            }
        )
    elif reason == "finished":
        record.update(
            {
                "eager_parent_full_accept_by_seq_id": {"1": False},
                "eager_parent_finished_by_seq_id": {"1": True},
                "eager_promoted_seq_ids": [],
                "eager_promoted_proposal_ids": [],
                "eager_discarded_seq_ids": [1],
                "eager_discarded_proposal_ids": [101],
                "eager_promotion_reason_by_seq_id": {},
                "eager_discard_reason_by_seq_id": {"1": "parent_finished"},
                "eager_tokens_promoted": 0,
                "eager_tokens_discarded": 4,
            }
        )
    elif reason == "base_mismatch":
        record.update(
            {
                "eager_parent_full_accept_by_seq_id": {"1": False},
                "eager_promotion_current_len_by_seq_id": {"1": 13},
                "eager_promotion_base_match_by_seq_id": {"1": False},
                "eager_promoted_seq_ids": [],
                "eager_promoted_proposal_ids": [],
                "eager_discarded_seq_ids": [1],
                "eager_discarded_proposal_ids": [101],
                "eager_promotion_reason_by_seq_id": {},
                "eager_discard_reason_by_seq_id": {"1": "base_len_mismatch"},
                "eager_tokens_promoted": 0,
                "eager_tokens_discarded": 4,
            }
        )
    elif reason == "pre_verify":
        record.update(
            {
                "eager_draft_base_pre_verify_by_seq_id": {"1": True},
                "eager_parent_full_accept_by_seq_id": {"1": False},
                "eager_promoted_seq_ids": [],
                "eager_promoted_proposal_ids": [],
                "eager_discarded_seq_ids": [1],
                "eager_discarded_proposal_ids": [101],
                "eager_promotion_reason_by_seq_id": {},
                "eager_discard_reason_by_seq_id": {"1": "selected_seq_pre_verify"},
                "eager_tokens_promoted": 0,
                "eager_tokens_discarded": 4,
            }
        )
    return record


def run_synthetic_tests() -> None:
    valid_records = [
        synthetic_base_record(),
        synthetic_plan_record(),
        synthetic_draft_record(),
        synthetic_promotion_record(),
        synthetic_promotion_record("reject"),
        synthetic_promotion_record("finished"),
        synthetic_promotion_record("base_mismatch"),
        synthetic_promotion_record("pre_verify"),
    ]
    errors, summary = validate_records(valid_records)
    assert not errors, f"valid synthetic promotion dry-run records failed: {errors}"
    assert summary["eager_promoted_tokens"] + summary["eager_discarded_tokens"] == 20

    invalid = deepcopy(valid_records)
    invalid[3]["eager_parent_full_accept_by_seq_id"] = {"1": False}
    errors, _ = validate_records(invalid)
    assert any("parent_full_accept" in error for error in errors), "checker missed bad promotion accept state"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_promotion_base_match_by_seq_id"] = {"1": False}
    errors, _ = validate_records(invalid)
    assert any("base_len match" in error for error in errors), "checker missed promoted base mismatch"

    invalid = deepcopy(valid_records)
    invalid[4]["eager_discard_reason_by_seq_id"] = {}
    errors, _ = validate_records(invalid)
    assert any("missing discard reason" in error for error in errors), "checker missed missing discard reason"

    invalid = deepcopy(valid_records)
    invalid[3]["eager_tokens_discarded"] = 1
    errors, _ = validate_records(invalid)
    assert any("must equal" in error for error in errors), "checker missed token accounting mismatch"

    print("Synthetic eager promotion dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-3 eager promotion dry-run traces.")
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
    print("\nEager promotion dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
