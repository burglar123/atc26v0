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

from benchmark.check_eager_performance_accounting import (
    aggregate_performance_accounting,
    load_json,
    load_trace,
)


TABLE_COLUMNS = [
    "case",
    "engine_elapsed_s",
    "total_output_tokens",
    "goodput_tokens_per_s",
    "mean_tpot_ms",
    "committed_tokens",
    "committed_proposals",
    "commit_rate_token",
    "commit_rate_proposal",
    "normal_draft_slots_suppressed",
    "target_verify_slots_replaced",
    "transfer_payload_bytes",
    "overhead_time_ms",
    "notes",
]


def result_embedded_accounting(result_payload: dict[str, Any]) -> dict[str, Any]:
    value = result_payload.get("eager_performance_accounting")
    return value if isinstance(value, dict) else {}


def accounting_from_inputs(
    result_path: Path,
    trace_path: Path | None,
) -> dict[str, Any]:
    result_payload = load_json(result_path)
    embedded = result_embedded_accounting(result_payload)
    if embedded:
        return {**embedded, "accounting_source": "result_json"}
    if trace_path is not None:
        accounting = aggregate_performance_accounting(load_trace(trace_path), result_payload)
        return {**accounting, "accounting_source": "trace"}
    accounting = aggregate_performance_accounting([], result_payload)
    return {**accounting, "accounting_source": "result_metrics_only"}


def compact_row(case_name: str, accounting: dict[str, Any]) -> dict[str, Any]:
    transfer_payload_bytes = int(accounting.get("eager_proposal_transfer_payload_bytes") or 0) + int(
        accounting.get("eager_result_transfer_payload_bytes") or 0
    )
    notes: list[str] = []
    if not accounting.get("payload_bytes_available", False):
        notes.append("payload_bytes_unavailable")
    if not accounting.get("timing_available", False):
        notes.append("timing_unavailable")
    if int(accounting.get("eager_committed_token_count") or 0) == 0:
        notes.append("no_committed_tokens")
    if accounting.get("accounting_source") == "result_metrics_only":
        notes.append("trace_accounting_missing")
    return {
        "case": case_name,
        "engine_elapsed_s": accounting.get("engine_elapsed_s", 0.0),
        "total_output_tokens": accounting.get("total_output_tokens", 0),
        "goodput_tokens_per_s": accounting.get("goodput_tokens_per_s", 0.0),
        "mean_tpot_ms": accounting.get("mean_tpot_ms", 0.0),
        "committed_tokens": accounting.get("eager_committed_token_count", 0),
        "committed_proposals": accounting.get("eager_committed_proposal_count", 0),
        "commit_rate_token": accounting.get("eager_commit_rate_by_token", 0.0),
        "commit_rate_proposal": accounting.get("eager_commit_rate_by_proposal", 0.0),
        "normal_draft_slots_suppressed": accounting.get("normal_draft_token_slots_suppressed", 0),
        "target_verify_slots_replaced": accounting.get("target_normal_verify_token_slots_replaced_by_eager", 0),
        "transfer_payload_bytes": transfer_payload_bytes,
        "overhead_time_ms": accounting.get("total_eager_overhead_time_ms", 0.0),
        "notes": ",".join(notes) if notes else "ok",
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    widths = {
        column: max(len(column), *(len(fmt(row.get(column, ""))) for row in rows))
        for column in TABLE_COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in TABLE_COLUMNS)
    print(header)
    print("  ".join("-" * widths[column] for column in TABLE_COLUMNS))
    for row in rows:
        print("  ".join(fmt(row.get(column, "")).ljust(widths[column]) for column in TABLE_COLUMNS))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Phase 1H-6c performance accounting summaries.")
    parser.add_argument("--baseline-json", required=True, type=Path)
    parser.add_argument("--eager-json", required=True, type=Path)
    parser.add_argument("--dryrun-json", type=Path)
    parser.add_argument("--baseline-trace", type=Path)
    parser.add_argument("--eager-trace", type=Path)
    parser.add_argument("--dryrun-trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cases = [
        ("baseline", args.baseline_json, args.baseline_trace),
        ("eager", args.eager_json, args.eager_trace),
    ]
    if args.dryrun_json:
        cases.append(("dryrun", args.dryrun_json, args.dryrun_trace))

    rows = [
        compact_row(case_name, accounting_from_inputs(result_path, trace_path))
        for case_name, result_path, trace_path in cases
    ]
    print_table(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump({"cases": rows}, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote comparison summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
