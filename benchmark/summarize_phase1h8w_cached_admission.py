#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import int_value  # noqa: E402
from benchmark.summarize_phase1h8u_full_continuous import bool_value, first_present  # noqa: E402


DEFAULT_SUMMARY_NAME = "phase1h8w_summary.json"


def float_value(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("rows", "cases", "summary"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise SystemExit(f"{path} does not contain a summary row list")


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
    raise SystemExit("multiple summary files found; pass --summary explicitly")


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": first_present(row, "case_name", "case", default=""),
        "status": first_present(row, "check_status", "status", default="unknown"),
        "cached_admission_enabled": bool_value(first_present(row, "cached_admission_enabled", default=False)),
        "policy": first_present(row, "cached_admission_policy", default=None),
        "max_active": int_value(first_present(row, "cached_admission_max_active")),
        "total_requests": int_value(first_present(row, "cached_admission_total_requests")),
        "arrived": int_value(first_present(row, "cached_admission_total_arrived")),
        "admitted": int_value(first_present(row, "cached_admission_total_admitted")),
        "completed": int_value(first_present(row, "cached_admission_total_completed")),
        "peak_active": int_value(first_present(row, "cached_admission_peak_active")),
        "mean_queue_wait_ms": float_value(first_present(row, "cached_admission_mean_queue_wait_ms"), 0.0),
        "p90_queue_wait_ms": float_value(first_present(row, "cached_admission_p90_queue_wait_ms"), 0.0),
        "decode_only_elapsed_s": float_value(first_present(row, "cached_admission_decode_only_elapsed_s"), 0.0),
        "full_continuous": bool_value(
            first_present(row, "generic_full_continuous_enabled", "enable_full_continuous_eager", default=False)
        ),
        "max_real_depth": int_value(
            first_present(row, "generic_full_continuous_max_real_committed_depth", "max_real_committed_depth")
        ),
        "output_tokens": int_value(
            first_present(
                row,
                "generic_full_continuous_total_output_token_count",
                "combined_actual_output_token_increment_sum",
                "combined_real_committed_tokens",
                "combined_real_committed_token_count",
            )
        ),
        "combined": int_value(
            first_present(row, "combined_real_committed_tokens", "combined_real_committed_token_count")
        ),
        "target_draft_mismatch": int_value(
            first_present(
                row,
                "generic_full_continuous_target_draft_mismatch_count",
                "target_draft_mismatch_count",
                "target_draft_length_mismatch_count",
                "partial_recovery_target_draft_length_mismatch_count",
            )
        ),
        "normal_conflict": int_value(
            first_present(row, "generic_full_continuous_normal_lane_conflict_count", "normal_lane_conflict_count")
        ),
        "accounting_ok": bool_value(
            first_present(row, "combined_accounting_ok", "generic_chain_accounting_ok", default=False)
        ),
        "goodput": first_present(row, "goodput_tokens_per_s", default=None),
        "mean_tpot": first_present(row, "mean_tpot_ms", default=None),
    }


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_row(row) for row in rows]


def print_human(rows: list[dict[str, Any]]) -> None:
    compact = compact_rows(rows)
    fields = (
        "case",
        "status",
        "cached_admission_enabled",
        "policy",
        "max_active",
        "total_requests",
        "arrived",
        "admitted",
        "completed",
        "peak_active",
        "mean_queue_wait_ms",
        "p90_queue_wait_ms",
        "decode_only_elapsed_s",
        "full_continuous",
        "max_real_depth",
        "output_tokens",
        "combined",
        "target_draft_mismatch",
        "normal_conflict",
        "accounting_ok",
        "goodput",
        "mean_tpot",
    )
    widths = {
        field: max(len(field), *(len(json.dumps(row.get(field), sort_keys=True)) for row in compact))
        for field in fields
    }
    print(" ".join(field.ljust(widths[field]) for field in fields))
    for row in compact:
        print(" ".join(json.dumps(row.get(field), sort_keys=True).ljust(widths[field]) for field in fields))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 1H-8w cached admission results.")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary_path = resolve_summary_path(args.root, args.summary)
    rows = load_summary_rows(summary_path)
    compact = compact_rows(rows)
    if args.json:
        print(json.dumps({"summary_path": str(summary_path), "rows": compact}, indent=2, sort_keys=True))
    else:
        print(f"summary_path={summary_path}")
        print_human(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
