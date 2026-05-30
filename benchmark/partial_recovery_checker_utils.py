#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def partial_recovery_descendant_commit_count(records: list[dict[str, Any]]) -> int:
    return sum(
        int_value(record.get("descendant_committed_after_partial_count"), 0)
        for record in records
        if isinstance(record, dict)
    )


def partial_recovery_accounting_errors(
    accounting: dict[str, Any],
    *,
    descendant_committed_after_partial_count: int = 0,
) -> tuple[list[str], int]:
    errors: list[str] = []
    enabled = bool(accounting.get("partial_prefix_recovery_enabled", False)) or bool(
        accounting.get("enable_rolling_continuous_partial_prefix_recovery", False)
    )
    success_count = int_value(accounting.get("partial_prefix_recovery_success_count"), 0)
    partial_accepted = int_value(accounting.get("partial_prefix_accepted_token_count"), 0)
    partial_revised = int_value(accounting.get("partial_prefix_revised_token_count"), 0)
    partial_total = int_value(accounting.get("partial_prefix_total_recovered_token_count"), 0)
    len_mismatch = int_value(accounting.get("partial_recovery_target_draft_length_mismatch_count"), 0)
    token_mismatch = int_value(accounting.get("partial_recovery_target_draft_token_mismatch_count"), 0)

    if partial_total and not enabled:
        errors.append("partial recovery tokens present while partial-prefix recovery is disabled")
    if partial_total < 0:
        errors.append("partial recovery total token count must be nonnegative")
    if partial_total and success_count <= 0:
        errors.append("partial recovery tokens require successful partial recovery evidence")
    if partial_total != partial_accepted + partial_revised:
        errors.append("partial recovery total tokens must equal accepted prefix plus revised tokens")
    if partial_revised and partial_revised != success_count:
        errors.append("partial recovery revised token count must equal successful recovery count")
    if len_mismatch:
        errors.append("partial recovery target/draft length mismatch count must be zero")
    if token_mismatch:
        errors.append("partial recovery target/draft token mismatch count must be zero")
    if descendant_committed_after_partial_count:
        errors.append("descendant committed after partial recovery")

    legal_partial_total = 0
    if (
        enabled
        and success_count > 0
        and partial_total >= 0
        and partial_total == partial_accepted + partial_revised
        and len_mismatch == 0
        and token_mismatch == 0
        and descendant_committed_after_partial_count == 0
    ):
        legal_partial_total = partial_total

    if legal_partial_total:
        combined_accepted = int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0)
        combined_revised = int_value(accounting.get("combined_actual_revised_token_increment_sum"), 0)
        combined_output = int_value(accounting.get("combined_actual_output_token_increment_sum"), 0)
        combined_verified = int_value(accounting.get("combined_actual_verified_token_increment_sum"), 0)
        combined_tokens = int_value(accounting.get("combined_real_committed_token_count"), 0)
        if combined_output != combined_accepted + combined_revised:
            errors.append("combined output increment must equal accepted plus revised increments in partial recovery mode")
        if combined_revised != partial_revised:
            errors.append("combined revised increment must equal partial revised token count")
        if combined_output != combined_tokens:
            errors.append("combined output increment must equal combined real committed token count")
        if combined_verified != combined_tokens:
            errors.append("combined verified increment must equal combined real committed token count")

    return errors, legal_partial_total


def partial_recovery_summary_fields(
    accounting: dict[str, Any],
    *,
    descendant_committed_after_partial_count: int = 0,
) -> dict[str, Any]:
    return {
        "partial_prefix_recovery_enabled": bool(accounting.get("partial_prefix_recovery_enabled", False)),
        "partial_prefix_recovery_success_count": int_value(
            accounting.get("partial_prefix_recovery_success_count"),
            0,
        ),
        "partial_prefix_accepted_token_count": int_value(
            accounting.get("partial_prefix_accepted_token_count"),
            0,
        ),
        "partial_prefix_revised_token_count": int_value(
            accounting.get("partial_prefix_revised_token_count"),
            0,
        ),
        "partial_prefix_total_recovered_token_count": int_value(
            accounting.get("partial_prefix_total_recovered_token_count"),
            0,
        ),
        "descendant_committed_after_partial_count": int(descendant_committed_after_partial_count),
        "combined_actual_verified_token_increment_sum": int_value(
            accounting.get("combined_actual_verified_token_increment_sum"),
            0,
        ),
        "combined_actual_accepted_token_increment_sum": int_value(
            accounting.get("combined_actual_accepted_token_increment_sum"),
            0,
        ),
        "combined_actual_revised_token_increment_sum": int_value(
            accounting.get("combined_actual_revised_token_increment_sum"),
            0,
        ),
        "combined_actual_output_token_increment_sum": int_value(
            accounting.get("combined_actual_output_token_increment_sum"),
            0,
        ),
    }


def partial_recovery_print_fields() -> tuple[str, ...]:
    return (
        "partial_prefix_recovery_enabled",
        "partial_prefix_recovery_success_count",
        "partial_prefix_accepted_token_count",
        "partial_prefix_revised_token_count",
        "partial_prefix_total_recovered_token_count",
        "descendant_committed_after_partial_count",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "combined_actual_revised_token_increment_sum",
        "combined_actual_output_token_increment_sum",
        "expected_combined_real_committed_token_count",
        "combined_real_committed_token_count",
    )


def add_synthetic_partial_recovery_fields(
    record: dict[str, Any],
    *,
    enabled: bool = True,
    accepted_tokens: int = 1,
    revised_tokens: int = 1,
    proposal_id: int = 900000605,
    seq_id: int = 7,
    depth: int = 1,
    frontier_before: int = 44,
    len_match: bool = True,
    token_match: bool = True,
    descendant_committed_after_partial_count: int = 0,
) -> None:
    total = int(accepted_tokens) + int(revised_tokens)
    record["partial_prefix_recovery_enabled"] = bool(enabled)
    record["enable_rolling_continuous_partial_prefix_recovery"] = bool(enabled)
    record["partial_prefix_recovery_attempt_count"] = 1
    record["partial_prefix_recovery_success_count"] = 1
    record["partial_prefix_recovery_skip_reason_counts"] = {}
    record["partial_prefix_recovered_proposal_ids"] = [int(proposal_id)]
    record["partial_prefix_recovered_seq_ids"] = [int(seq_id)]
    record["partial_prefix_recovered_depth_by_proposal_id"] = {str(proposal_id): int(depth)}
    record["partial_prefix_accepted_len_by_proposal_id"] = {str(proposal_id): int(accepted_tokens)}
    record["partial_prefix_reject_index_by_proposal_id"] = {str(proposal_id): int(accepted_tokens)}
    record["partial_prefix_revised_token_count_by_proposal_id"] = {str(proposal_id): int(revised_tokens)}
    record["partial_prefix_committed_token_count_by_proposal_id"] = {str(proposal_id): total}
    record["partial_prefix_recovery_frontier_before_by_seq_id"] = {str(seq_id): int(frontier_before)}
    record["partial_prefix_recovery_frontier_after_by_seq_id"] = {str(seq_id): int(frontier_before) + total}
    record["partial_prefix_descendant_cascade_discard_count_by_proposal_id"] = {str(proposal_id): 0}
    record["partial_prefix_recovery_normal_release_seq_ids"] = [int(seq_id)]
    record["partial_recovery_target_seq_len_before_by_seq_id"] = {str(seq_id): int(frontier_before)}
    record["partial_recovery_target_seq_len_after_by_seq_id"] = {str(seq_id): int(frontier_before) + total}
    record["partial_recovery_draft_seq_len_before_by_seq_id"] = {str(seq_id): int(frontier_before)}
    record["partial_recovery_draft_seq_len_after_by_seq_id"] = {str(seq_id): int(frontier_before) + total}
    record["partial_recovery_target_draft_len_match_by_seq_id"] = {str(seq_id): bool(len_match)}
    record["partial_recovery_target_draft_token_match_by_seq_id"] = {str(seq_id): bool(token_match)}
    record["partial_recovery_cascade_discarded_descendant_proposal_ids"] = []
    record["partial_recovery_cascade_discarded_descendant_depth_by_proposal_id"] = {}
    record["partial_recovery_cascade_discarded_descendant_reason_by_proposal_id"] = {}
    record["descendant_committed_after_partial_count"] = int(descendant_committed_after_partial_count)


def assert_partial_recovery_checker_cases(
    checker_name: str,
    make_records: Callable[[], list[dict[str, Any]]],
    validate_records: Callable[[list[dict[str, Any]]], tuple[list[str], dict[str, Any]]],
    *,
    expected_full_accept_combined_token_count: int | None = 44,
) -> None:
    base_records = make_records()
    errors, summary = validate_records(base_records)
    if errors:
        raise SystemExit(f"synthetic {checker_name} full-accept baseline failed: {errors}\nsummary={summary}")
    base_combined = int_value(summary.get("combined_real_committed_token_count"), 0)
    if expected_full_accept_combined_token_count is not None and base_combined != expected_full_accept_combined_token_count:
        raise SystemExit(
            f"synthetic {checker_name} full-accept baseline combined mismatch: "
            f"expected {expected_full_accept_combined_token_count}, got {base_combined}"
        )
    if int_value(summary.get("partial_prefix_total_recovered_token_count"), 0) != 0:
        raise SystemExit(f"synthetic {checker_name} full-accept baseline should not include partial recovery")

    legal = make_records()
    for record in legal:
        add_synthetic_partial_recovery_fields(record)
    errors, summary = validate_records(legal)
    if errors:
        raise SystemExit(f"synthetic {checker_name} legal partial recovery failed: {errors}\nsummary={summary}")
    if summary.get("partial_prefix_recovery_enabled") is not True:
        raise SystemExit(f"synthetic {checker_name} legal partial recovery should be enabled")
    if int_value(summary.get("partial_prefix_recovery_success_count"), 0) != 1:
        raise SystemExit(f"synthetic {checker_name} legal partial recovery success count should be 1")
    if int_value(summary.get("partial_prefix_accepted_token_count"), 0) != 1:
        raise SystemExit(f"synthetic {checker_name} legal partial accepted tokens should be 1")
    if int_value(summary.get("partial_prefix_revised_token_count"), 0) != 1:
        raise SystemExit(f"synthetic {checker_name} legal partial revised tokens should be 1")
    if int_value(summary.get("partial_prefix_total_recovered_token_count"), 0) != 2:
        raise SystemExit(f"synthetic {checker_name} legal partial total should be 2")
    if int_value(summary.get("combined_real_committed_token_count"), 0) != base_combined + 2:
        raise SystemExit(
            f"synthetic {checker_name} legal partial combined mismatch: "
            f"expected {base_combined + 2}, got {summary.get('combined_real_committed_token_count')}"
        )
    if int_value(summary.get("combined_actual_accepted_token_increment_sum"), 0) != base_combined + 1:
        raise SystemExit(f"synthetic {checker_name} legal partial accepted increment mismatch")
    if int_value(summary.get("combined_actual_revised_token_increment_sum"), 0) != 1:
        raise SystemExit(f"synthetic {checker_name} legal partial revised increment mismatch")
    if int_value(summary.get("combined_actual_output_token_increment_sum"), 0) != base_combined + 2:
        raise SystemExit(f"synthetic {checker_name} legal partial output increment mismatch")

    disabled = make_records()
    for record in disabled:
        add_synthetic_partial_recovery_fields(record, enabled=False)
    errors, summary = validate_records(disabled)
    if not errors:
        raise SystemExit(f"synthetic {checker_name} partial tokens while disabled should fail\nsummary={summary}")
    if not any("partial recovery tokens present" in error for error in errors):
        raise SystemExit(f"synthetic {checker_name} disabled partial failed for wrong reason: {errors}")

    missing_partial = make_records()
    for record in missing_partial:
        add_synthetic_partial_recovery_fields(record)
        record["partial_prefix_committed_token_count_by_proposal_id"] = {"900000605": 0}
    errors, summary = validate_records(missing_partial)
    if not errors:
        raise SystemExit(f"synthetic {checker_name} missing partial combined evidence should fail\nsummary={summary}")
    if not any("partial recovery total tokens" in error or "combined real committed" in error for error in errors):
        raise SystemExit(f"synthetic {checker_name} missing partial failed for wrong reason: {errors}")

    double_partial = make_records()
    for record in double_partial:
        add_synthetic_partial_recovery_fields(record)
        record["partial_prefix_committed_token_count_by_proposal_id"] = {"900000605": 4}
    errors, summary = validate_records(double_partial)
    if not errors:
        raise SystemExit(f"synthetic {checker_name} double-counted partial evidence should fail\nsummary={summary}")
    if not any("partial recovery total tokens" in error or "combined real committed" in error for error in errors):
        raise SystemExit(f"synthetic {checker_name} double partial failed for wrong reason: {errors}")

    descendant = make_records()
    for record in descendant:
        add_synthetic_partial_recovery_fields(record, descendant_committed_after_partial_count=1)
    errors, summary = validate_records(descendant)
    if not errors:
        raise SystemExit(f"synthetic {checker_name} descendant-after-partial should fail\nsummary={summary}")
    if not any("descendant committed after partial" in error for error in errors):
        raise SystemExit(f"synthetic {checker_name} descendant failure reported wrong reason: {errors}")

    mismatch = make_records()
    for record in mismatch:
        add_synthetic_partial_recovery_fields(record, len_match=False)
    errors, summary = validate_records(mismatch)
    if not errors:
        raise SystemExit(f"synthetic {checker_name} partial target/draft mismatch should fail\nsummary={summary}")
    if not any("target/draft length mismatch" in error for error in errors):
        raise SystemExit(f"synthetic {checker_name} mismatch failure reported wrong reason: {errors}")

    bad_revised = {
        "partial_prefix_recovery_enabled": True,
        "partial_prefix_recovery_success_count": 1,
        "partial_prefix_accepted_token_count": 1,
        "partial_prefix_revised_token_count": 1,
        "partial_prefix_total_recovered_token_count": 2,
        "combined_real_committed_token_count": base_combined + 2,
        "combined_actual_verified_token_increment_sum": base_combined + 2,
        "combined_actual_accepted_token_increment_sum": base_combined + 1,
        "combined_actual_revised_token_increment_sum": 1,
        "combined_actual_output_token_increment_sum": base_combined + 2,
    }
    errors, legal_partial_total = partial_recovery_accounting_errors(bad_revised)
    if errors or legal_partial_total != 2:
        raise SystemExit(f"synthetic {checker_name} direct legal partial accounting failed: {errors}")
    broken_revised = deepcopy(bad_revised)
    broken_revised["combined_actual_revised_token_increment_sum"] = 0
    errors, _legal_partial_total = partial_recovery_accounting_errors(broken_revised)
    if not any("combined revised increment" in error for error in errors):
        raise SystemExit(f"synthetic {checker_name} bad revised accounting should fail: {errors}")
