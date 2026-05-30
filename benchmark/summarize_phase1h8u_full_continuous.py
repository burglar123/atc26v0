#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_NAME = "phase1h8u_summary.json"
DEFAULT_CASES = (
    "baseline_8t_generic_apply_depth4",
    "full_continuous_depth100_baseline",
    "full_continuous_depth100_partial_recovery",
    "full_continuous_depth100_stress",
    "full_continuous_depth100_must_exceed4",
)


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "pass"}
    return bool(value)


def depth_map_has_gt4_activity(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key, raw_count in value.items():
        try:
            depth = int(key)
            count = int(raw_count)
        except Exception:
            continue
        if depth >= 5 and count > 0:
            return True
    return False


def has_depth_gt4_activity(row: dict[str, Any]) -> bool:
    max_observed = int_value(first_present(row, "generic_full_continuous_max_observed_depth", "max_observed_depth"))
    max_real = int_value(first_present(row, "generic_full_continuous_max_real_committed_depth", "max_real_committed_depth"))
    if max_observed > 4 or max_real > 4:
        return True
    return any(
        depth_map_has_gt4_activity(row.get(field))
        for field in (
            "generic_full_continuous_depth_commit_token_counts",
            "generic_full_continuous_depth_candidate_token_counts",
            "generic_full_continuous_depth_ready_token_counts",
        )
    )


def resolve_summary_path(root: Path | None, summary: Path | None) -> Path:
    if summary is not None:
        return summary
    if root is None:
        raise SystemExit("either --root or --summary is required")
    preferred = root / DEFAULT_SUMMARY_NAME
    if preferred.exists():
        return preferred
    candidates = sorted(root.glob("*summary*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"no summary JSON found under {root}")
    raise SystemExit(
        "multiple summary JSON files found; pass --summary explicitly: "
        + ", ".join(str(path) for path in candidates)
    )


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("rows", "cases", "summary"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise SystemExit(f"{path} does not contain a summary row list")


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    full_continuous = bool_value(
        first_present(row, "generic_full_continuous_enabled", "enable_full_continuous_eager", default=False)
    )
    combined = int_value(
        first_present(row, "combined_real_committed_tokens", "combined_real_committed_token_count")
    )
    output_tokens = int_value(
        first_present(
            row,
            "generic_full_continuous_total_output_token_count",
            "combined_actual_output_token_increment_sum",
            "combined_real_committed_tokens",
            "combined_real_committed_token_count",
        )
    )
    partial_tokens = int_value(
        first_present(
            row,
            "generic_full_continuous_total_partial_recovered_token_count",
            "partial_prefix_total_recovered_token_count",
        )
    )
    revised_tokens = int_value(
        first_present(
            row,
            "generic_full_continuous_total_revised_token_count",
            "partial_prefix_revised_token_count",
            "combined_actual_revised_token_increment_sum",
        )
    )
    target_draft_mismatch = int_value(
        first_present(
            row,
            "generic_full_continuous_target_draft_mismatch_count",
            "target_draft_mismatch_count",
            "target_draft_length_mismatch_count",
            "partial_recovery_target_draft_length_mismatch_count",
        )
    ) + int_value(first_present(row, "target_draft_token_mismatch_count", default=0))
    return {
        "case": first_present(row, "case_name", "case", default=""),
        "status": first_present(row, "check_status", "status", default="unknown"),
        "core_status": first_present(row, "core_checker_chain_status", "core_status", default=None),
        "full_continuous": full_continuous,
        "max_depth": int_value(
            first_present(row, "generic_full_continuous_max_depth", "generic_rolling_max_depth", "max_configured_depth")
        ),
        "max_observed": int_value(
            first_present(row, "generic_full_continuous_max_observed_depth", "max_observed_depth")
        ),
        "max_real": int_value(
            first_present(row, "generic_full_continuous_max_real_committed_depth", "max_real_committed_depth")
        ),
        "depth_gt_max": int_value(first_present(row, "generic_full_continuous_depth_gt_max_real_commit_count")),
        "depth_gt4_activity": has_depth_gt4_activity(row),
        "full_tokens": int_value(first_present(row, "generic_full_continuous_total_full_commit_token_count")),
        "partial_tokens": partial_tokens,
        "revised_tokens": revised_tokens,
        "output_tokens": output_tokens,
        "combined": combined,
        "accepted": int_value(first_present(row, "combined_actual_accepted_token_increment_sum")),
        "revised": int_value(first_present(row, "combined_actual_revised_token_increment_sum")),
        "normal_conflict": int_value(
            first_present(row, "generic_full_continuous_normal_lane_conflict_count", "normal_lane_conflict_count")
        ),
        "target_draft_mismatch": target_draft_mismatch,
        "accounting_ok": bool_value(
            first_present(row, "combined_accounting_ok", "generic_chain_accounting_ok", default=False)
        ),
        "generic_chain_ok": bool_value(
            first_present(
                row,
                "generic_chain_accounting_ok",
                "generic_full_continuous_parity_ok",
                "generic_rolling_apply_parity_ok",
                default=False,
            )
        ),
        "stop_reasons": first_present(row, "generic_full_continuous_stop_reason_counts", default={}) or {},
        "goodput": first_present(row, "goodput_tokens_per_s", default=None),
        "mean_tpot": first_present(row, "mean_tpot_ms", default=None),
        "warnings": first_present(row, "performance_warnings", default=[]) or [],
        "partial_prefix_total_recovered_token_count": int_value(
            first_present(row, "partial_prefix_total_recovered_token_count")
        ),
        "partial_prefix_revised_token_count": int_value(first_present(row, "partial_prefix_revised_token_count")),
    }


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_row(row) for row in rows]


def strict_errors(rows: list[dict[str, Any]]) -> list[str]:
    compact = compact_rows(rows)
    by_case = {str(row["case"]): row for row in compact}
    errors: list[str] = []

    def require(case: str) -> dict[str, Any] | None:
        row = by_case.get(case)
        if row is None:
            errors.append(f"{case}: missing summary row")
        return row

    baseline = require("baseline_8t_generic_apply_depth4")
    if baseline is not None:
        if baseline["status"] != "pass":
            errors.append("baseline_8t_generic_apply_depth4: check_status must be pass")
        if baseline["combined"] != 44:
            errors.append("baseline_8t_generic_apply_depth4: combined must be 44")
        if baseline["full_continuous"]:
            errors.append("baseline_8t_generic_apply_depth4: full_continuous must be false")
        if baseline["normal_conflict"] != 0:
            errors.append("baseline_8t_generic_apply_depth4: normal_conflict must be zero")
        if baseline["target_draft_mismatch"] != 0:
            errors.append("baseline_8t_generic_apply_depth4: target_draft_mismatch must be zero")

    for case in (
        "full_continuous_depth100_baseline",
        "full_continuous_depth100_partial_recovery",
        "full_continuous_depth100_stress",
        "full_continuous_depth100_must_exceed4",
    ):
        row = require(case)
        if row is None:
            continue
        if row["status"] != "pass":
            errors.append(f"{case}: check_status must be pass")
        if not row["full_continuous"]:
            errors.append(f"{case}: full_continuous must be true")
        if row["max_depth"] != 100:
            errors.append(f"{case}: max_depth must be 100")
        if row["max_observed"] <= 4:
            errors.append(f"{case}: max_observed must exceed 4")
        if row["max_real"] <= 4:
            errors.append(f"{case}: max_real must exceed 4")
        if not row["depth_gt4_activity"]:
            errors.append(f"{case}: depth_gt4_activity must be true")
        if row["output_tokens"] != row["combined"]:
            errors.append(f"{case}: output_tokens must equal combined")
        if row["depth_gt_max"] != 0:
            errors.append(f"{case}: depth_gt_max must be zero")
        if row["normal_conflict"] != 0:
            errors.append(f"{case}: normal_conflict must be zero")
        if row["target_draft_mismatch"] != 0:
            errors.append(f"{case}: target_draft_mismatch must be zero")
        if case == "full_continuous_depth100_baseline":
            if row["partial_tokens"] != 0:
                errors.append(f"{case}: partial_tokens must be zero")
            if row["revised_tokens"] != 0:
                errors.append(f"{case}: revised_tokens must be zero")
        else:
            partial_reference = row["partial_prefix_total_recovered_token_count"]
            if partial_reference and row["partial_tokens"] != partial_reference:
                errors.append(f"{case}: partial_tokens must match partial_prefix_total_recovered_token_count")
            revised_reference = row["partial_prefix_revised_token_count"]
            if revised_reference and row["revised_tokens"] != revised_reference:
                errors.append(f"{case}: revised_tokens must match partial_prefix_revised_token_count")

    return errors


def print_human(rows: list[dict[str, Any]]) -> None:
    compact = compact_rows(rows)
    fields = (
        "case",
        "status",
        "core_status",
        "full_continuous",
        "max_depth",
        "max_observed",
        "max_real",
        "depth_gt_max",
        "depth_gt4_activity",
        "full_tokens",
        "partial_tokens",
        "revised_tokens",
        "output_tokens",
        "combined",
        "accepted",
        "revised",
        "normal_conflict",
        "target_draft_mismatch",
        "accounting_ok",
        "generic_chain_ok",
        "stop_reasons",
        "goodput",
        "mean_tpot",
        "warnings",
    )
    widths = {
        field: max(len(field), *(len(json.dumps(row.get(field), sort_keys=True)) for row in compact))
        for field in fields
    }
    print(" ".join(field.ljust(widths[field]) for field in fields))
    for row in compact:
        print(" ".join(json.dumps(row.get(field), sort_keys=True).ljust(widths[field]) for field in fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 1H-8u full-continuous milestone results.")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = resolve_summary_path(args.root, args.summary)
    rows = load_summary_rows(summary_path)
    errors = strict_errors(rows) if args.strict else []
    compact = compact_rows(rows)
    if args.json:
        payload = {"summary_path": str(summary_path), "rows": compact, "errors": errors}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"summary_path={summary_path}")
        print_human(rows)
        if errors:
            print("strict_status=fail")
            for error in errors:
                print(f"- {error}")
        elif args.strict:
            print("strict_status=pass")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
