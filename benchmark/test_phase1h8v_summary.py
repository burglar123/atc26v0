#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy

from summarize_phase1h8u_full_continuous import compact_row, strict_errors


BASELINE_CASE = "baseline_8t_generic_apply_depth4"
FULL_BASELINE_CASE = "full_continuous_depth100_baseline"
FULL_PARTIAL_CASE = "full_continuous_depth100_partial_recovery"
FULL_STRESS_CASE = "full_continuous_depth100_stress"
FULL_MUST_EXCEED_CASE = "full_continuous_depth100_must_exceed4"


def baseline_row() -> dict[str, object]:
    return {
        "case_name": BASELINE_CASE,
        "check_status": "pass",
        "generic_full_continuous_enabled": False,
        "combined_real_committed_token_count": 44,
        "max_observed_depth": 4,
        "max_real_committed_depth": 4,
        "normal_lane_conflict_count": 0,
        "target_draft_length_mismatch_count": 0,
        "target_draft_token_mismatch_count": 0,
        "combined_accounting_ok": True,
        "generic_chain_accounting_ok": True,
    }


def full_continuous_row(
    case_name: str,
    *,
    output_tokens: int = 484,
    partial_tokens: int = 0,
    revised_tokens: int = 0,
    max_observed: int = 60,
    max_real: int = 60,
    depth_gt_max: int = 0,
) -> dict[str, object]:
    return {
        "case_name": case_name,
        "check_status": "pass",
        "generic_full_continuous_enabled": True,
        "generic_full_continuous_max_depth": 100,
        "generic_full_continuous_max_observed_depth": max_observed,
        "generic_full_continuous_max_real_committed_depth": max_real,
        "generic_full_continuous_depth_commit_token_counts": {
            "0": 12,
            "1": 8,
            "2": 8,
            "3": 8,
            "4": 8,
            "5": 8,
        },
        "generic_full_continuous_depth_candidate_token_counts": {"5": 8},
        "generic_full_continuous_depth_ready_token_counts": {"5": 8},
        "generic_full_continuous_total_full_commit_token_count": 484,
        "generic_full_continuous_total_partial_recovered_token_count": partial_tokens,
        "generic_full_continuous_total_revised_token_count": revised_tokens,
        "generic_full_continuous_total_output_token_count": output_tokens,
        "combined_real_committed_token_count": output_tokens,
        "combined_actual_accepted_token_increment_sum": output_tokens - revised_tokens,
        "combined_actual_revised_token_increment_sum": revised_tokens,
        "generic_full_continuous_depth_gt_max_real_commit_count": depth_gt_max,
        "generic_full_continuous_normal_lane_conflict_count": 0,
        "generic_full_continuous_target_draft_mismatch_count": 0,
        "target_draft_token_mismatch_count": 0,
        "combined_accounting_ok": True,
        "generic_chain_accounting_ok": True,
        "partial_prefix_total_recovered_token_count": partial_tokens,
        "partial_prefix_revised_token_count": revised_tokens,
    }


def all_rows() -> list[dict[str, object]]:
    return [
        baseline_row(),
        full_continuous_row(FULL_BASELINE_CASE),
        full_continuous_row(FULL_PARTIAL_CASE, output_tokens=486, partial_tokens=2, revised_tokens=1),
        full_continuous_row(FULL_STRESS_CASE, output_tokens=486, partial_tokens=2, revised_tokens=1),
        full_continuous_row(FULL_MUST_EXCEED_CASE),
    ]


def test_compact_baseline_row() -> None:
    row = compact_row(baseline_row())
    assert row["case"] == BASELINE_CASE
    assert row["combined"] == 44
    assert row["full_continuous"] is False
    assert row["depth_gt4_activity"] is False


def test_compact_full_continuous_row() -> None:
    row = compact_row(full_continuous_row(FULL_BASELINE_CASE))
    assert row["case"] == FULL_BASELINE_CASE
    assert row["full_continuous"] is True
    assert row["max_real"] == 60
    assert row["depth_gt4_activity"] is True
    assert row["output_tokens"] == row["combined"]


def test_strict_passes_for_valid_rows() -> None:
    assert strict_errors(all_rows()) == []


def test_strict_fails_if_must_exceed_does_not_exceed_depth4() -> None:
    rows = all_rows()
    rows[-1] = full_continuous_row(FULL_MUST_EXCEED_CASE, max_observed=4, max_real=4)
    errors = strict_errors(rows)
    assert any("max_observed" in error for error in errors)
    assert any("max_real" in error for error in errors)


def test_strict_fails_if_output_differs_from_combined() -> None:
    rows = deepcopy(all_rows())
    rows[1]["combined_real_committed_token_count"] = 483
    errors = strict_errors(rows)
    assert any("output_tokens" in error for error in errors)


def test_strict_fails_if_depth_gt_max_is_nonzero() -> None:
    rows = all_rows()
    rows[1] = full_continuous_row(FULL_BASELINE_CASE, depth_gt_max=1)
    errors = strict_errors(rows)
    assert any("depth_gt_max" in error for error in errors)


def run_all_tests() -> None:
    test_compact_baseline_row()
    test_compact_full_continuous_row()
    test_strict_passes_for_valid_rows()
    test_strict_fails_if_must_exceed_does_not_exceed_depth4()
    test_strict_fails_if_output_differs_from_combined()
    test_strict_fails_if_depth_gt_max_is_nonzero()


if __name__ == "__main__":
    run_all_tests()
    print("Synthetic phase1h8v summary tests passed.")
