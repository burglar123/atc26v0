#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("cases", "rows", "summary"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise SystemExit(f"{path} does not contain a phase1h8e summary list")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value[:3])
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = list(value.items())[:3]
        return ",".join(f"{key}:{item}" for key, item in items)
    return str(value)


def warnings_text(row: dict[str, Any]) -> str:
    warnings = row.get("performance_warnings") or []
    if not isinstance(warnings, list):
        return fmt(warnings)
    return ",".join(str(item) for item in warnings[:3])


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("case", "case_name"),
        ("status", "check_status"),
        ("goodput", "goodput_tokens_per_s"),
        ("mean_tpot", "mean_tpot_ms"),
        ("one", "one_shot_committed_tokens"),
        ("d1", "continuous_depth1_real_committed_tokens"),
        ("d2", "rolling_depth2_real_committed_tokens"),
        ("d3", "rolling_depth3_real_committed_tokens"),
        ("combined", "combined_real_committed_tokens"),
        ("max_depth", "max_real_committed_depth"),
        ("gt3", "depth_gt3_real_commit_count"),
        ("conflict", "normal_lane_conflict_count"),
        ("combined_ok", "combined_accounting_ok"),
        ("warnings", "performance_warnings"),
    ]
    rendered: list[list[str]] = []
    headers = [header for header, _key in columns]
    for row in rows:
        rendered_row: list[str] = []
        for _header, key in columns:
            rendered_row.append(warnings_text(row) if key == "performance_warnings" else fmt(row.get(key)))
        rendered.append(rendered_row)
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered)) if rendered else len(headers[index])
        for index in range(len(headers))
    ]
    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rendered:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(row))))


def strict_failures(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        case_name = str(row.get("case_name", "<unknown>"))
        if row.get("check_status") != "pass":
            failures.append(f"{case_name}: status={row.get('check_status')}")
        if row.get("combined_accounting_ok") is not True:
            failures.append(f"{case_name}: combined_accounting_ok is not true")
        if row.get("target_draft_accounting_ok") is not True:
            failures.append(f"{case_name}: target_draft_accounting_ok is not true")
        for key in (
            "depth_gt3_real_commit_count",
            "depth4_real_commit_count",
            "depth_gt3_committed_proposal_count",
            "normal_lane_conflict_count",
            "missing_buffered_proposal_unexpected_count",
            "duplicate_commit_count",
            "invalid_committed_child_count",
            "cascade_committed_child_count",
            "parent_missing_committed_child_count",
        ):
            try:
                value = int(row.get(key) or 0)
            except Exception:
                value = 0
            if value:
                failures.append(f"{case_name}: {key}={value}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Phase 1H-8e bounded-readiness regression summaries.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = load_rows(args.summary)
    print_table(rows)
    failures = strict_failures(rows)
    if args.strict and failures:
        print("\nStrict failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
