#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_performance_accounting import (
    add_derived_metrics,
    aggregate_performance_accounting,
    load_json,
    load_trace,
)


TABLE_COLUMNS = [
    "case",
    "engine_elapsed_s",
    "total_output_tokens",
    "goodput_tokens_per_s",
    "goodput_ratio_eager_vs_baseline",
    "mean_tpot_ms",
    "mean_tpot_ratio_eager_vs_baseline",
    "committed_tokens",
    "committed_proposals",
    "committed_token_share_of_output",
    "commit_rate_token",
    "commit_rate_proposal",
    "normal_draft_slots_suppressed",
    "target_verify_slots_replaced",
    "suppressed_slots_per_committed_token",
    "replaced_slots_per_committed_token",
    "transfer_payload_bytes",
    "proposal_payload_len_units_per_committed_token",
    "result_payload_len_units_per_committed_token",
    "overhead_time_ms",
    "eager_transfer_time_ms",
    "eager_result_transfer_time_ms",
    "eager_commit_time_ms",
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
    if trace_path is not None:
        accounting = aggregate_performance_accounting(load_trace(trace_path), result_payload)
        return {**accounting, "accounting_source": "trace"}
    embedded = result_embedded_accounting(result_payload)
    if embedded:
        return {**add_derived_metrics(dict(embedded)), "accounting_source": "result_json"}
    accounting = aggregate_performance_accounting([], result_payload)
    return {**accounting, "accounting_source": "result_metrics_only"}


def safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator or 0.0)
        denominator = float(denominator or 0.0)
    except Exception:
        return 0.0
    return numerator / denominator if denominator else 0.0


def compact_row(
    case_name: str,
    accounting: dict[str, Any],
    baseline_accounting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transfer_payload_bytes = int(accounting.get("eager_proposal_transfer_payload_bytes") or 0) + int(
        accounting.get("eager_result_transfer_payload_bytes") or 0
    )
    baseline_accounting = baseline_accounting or accounting
    goodput_ratio = safe_ratio(
        accounting.get("goodput_tokens_per_s"),
        baseline_accounting.get("goodput_tokens_per_s"),
    )
    tpot_ratio = safe_ratio(
        accounting.get("mean_tpot_ms"),
        baseline_accounting.get("mean_tpot_ms"),
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
        "goodput_ratio_eager_vs_baseline": goodput_ratio,
        "mean_tpot_ms": accounting.get("mean_tpot_ms", 0.0),
        "mean_tpot_ratio_eager_vs_baseline": tpot_ratio,
        "committed_tokens": accounting.get("eager_committed_token_count", 0),
        "committed_proposals": accounting.get("eager_committed_proposal_count", 0),
        "committed_token_share_of_output": accounting.get("committed_token_share_of_output", 0.0),
        "commit_rate_token": accounting.get("eager_commit_rate_by_token", 0.0),
        "commit_rate_proposal": accounting.get("eager_commit_rate_by_proposal", 0.0),
        "normal_draft_slots_suppressed": accounting.get("normal_draft_token_slots_suppressed", 0),
        "target_verify_slots_replaced": accounting.get("target_normal_verify_token_slots_replaced_by_eager", 0),
        "suppressed_slots_per_committed_token": accounting.get("suppressed_slots_per_committed_token", 0.0),
        "replaced_slots_per_committed_token": accounting.get("replaced_slots_per_committed_token", 0.0),
        "transfer_payload_bytes": transfer_payload_bytes,
        "proposal_payload_len_units_per_committed_token": accounting.get(
            "proposal_payload_len_units_per_committed_token",
            0.0,
        ),
        "result_payload_len_units_per_committed_token": accounting.get(
            "result_payload_len_units_per_committed_token",
            0.0,
        ),
        "overhead_time_ms": accounting.get("total_eager_overhead_time_ms", 0.0),
        "eager_transfer_time_ms": accounting.get("eager_transfer_time_ms", 0.0),
        "eager_result_transfer_time_ms": accounting.get("eager_result_transfer_time_ms", 0.0),
        "eager_commit_time_ms": accounting.get("eager_commit_time_ms", 0.0),
        "notes": ",".join(notes) if notes else "ok",
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def synthetic_payload(
    *,
    total_output_tokens: int,
    goodput: float,
    mean_tpot_ms: float,
    stale_accounting: bool = False,
) -> dict[str, Any]:
    payload = {
        "args": {"execution_mode": "dual_batch_pearl", "decode_ready": True},
        "metrics": {
            "engine_elapsed_s": total_output_tokens / max(goodput, 1e-9),
            "overall": {
                "total_output_tokens": total_output_tokens,
                "goodput_tokens_per_s": goodput,
                "mean_tpot_ms": mean_tpot_ms,
            },
        },
        "traces": [],
    }
    if stale_accounting:
        payload["eager_performance_accounting"] = {
            "accounting_available": True,
            "total_output_tokens": total_output_tokens,
            "engine_elapsed_s": payload["metrics"]["engine_elapsed_s"],
            "goodput_tokens_per_s": goodput,
            "mean_tpot_ms": mean_tpot_ms,
            "eager_committed_token_count": 20,
            "eager_committed_proposal_count": 5,
            "eager_candidate_token_count": 32,
            "eager_candidate_proposal_count": 8,
            "normal_draft_token_slots_suppressed": 32,
            "target_normal_verify_token_slots_replaced_by_eager": 32,
            "eager_proposal_transfer_payload_len_units": 5320,
            "eager_result_transfer_payload_len_units": 248,
            "committed_token_share_of_output": 0,
            "suppressed_slots_per_committed_token": 0,
            "replaced_slots_per_committed_token": 0,
            "proposal_payload_len_units_per_committed_token": 0,
            "result_payload_len_units_per_committed_token": 0,
            "payload_bytes_available": False,
            "timing_available": False,
            "total_eager_overhead_time_ms": 0.0,
        }
    return payload


def synthetic_trace_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    committed = [101, 102, 103, 104, 105]
    skipped = [201, 202, 203]
    candidate = committed + skipped
    for side in ("target", "draft"):
        for offset, proposal_id in enumerate(committed):
            seq_id = 10 + offset
            records.append(
                {
                    "execution_mode": "dual_batch_pearl",
                    "dual_batch_enabled": True,
                    "normal_gamma": 4,
                    "target_eager_set": [],
                    "enable_eager_commit_readiness_dry_run": True,
                    "enable_eager_commit_ready_only": True,
                    "eager_commit_enabled": True,
                    "eager_commit_source": "phase1h5e3_takeover_lane",
                    "eager_commit_side": side,
                    "step_id": 20 + offset,
                    "plan_id": 30 + offset,
                    "eager_commit_candidate_proposal_ids": candidate,
                    "eager_commit_candidate_seq_ids": [10, 11, 12, 13, 14, 20, 21, 22],
                    "eager_commit_ready_proposal_ids": committed,
                    "eager_commit_from_readiness_proposal_ids": committed,
                    "eager_commit_not_ready_proposal_ids": skipped,
                    "eager_commit_not_ready_reason_by_proposal_id": {
                        str(pid): "not_full_accept" for pid in skipped
                    },
                    "eager_commit_skipped_proposal_ids": skipped,
                    "eager_commit_skip_reason_by_proposal_id": {
                        str(pid): "not_full_accept" for pid in skipped
                    },
                    "eager_committed_proposal_ids": [proposal_id],
                    "eager_committed_seq_ids": [seq_id],
                    "eager_committed_token_count_by_proposal_id": {str(proposal_id): 4},
                    "eager_committed_accept_len_by_proposal_id": {str(proposal_id): 4},
                    "eager_committed_action_by_proposal_id": {
                        str(proposal_id): "append_full_accept_then_rollback"
                    },
                    "eager_committed_verify_result_by_proposal_id": {str(proposal_id): "full_accept"},
                    "eager_commit_precondition_ok_by_proposal_id": {str(proposal_id): True},
                    "eager_commit_precondition_failed_by_proposal_id": {str(proposal_id): False},
                    "eager_commit_target_seq_len_before_by_seq_id": {str(seq_id): 40},
                    "eager_commit_target_seq_len_after_by_seq_id": {str(seq_id): 44},
                    "eager_commit_draft_seq_len_before_by_seq_id": {str(seq_id): 40},
                    "eager_commit_draft_seq_len_after_by_seq_id": {str(seq_id): 44},
                    "eager_commit_target_draft_len_match_by_seq_id": {str(seq_id): True},
                    "eager_commit_target_draft_token_match_by_seq_id": {str(seq_id): True},
                    "eager_commit_candidate_count": 8,
                    "eager_commit_committed_count": 1,
                    "eager_commit_skipped_count": 3,
                    "eager_tokens_committed": 4,
                    "eager_tokens_committed_full_accept": 4,
                    "eager_tokens_verified": 4,
                    "eager_tokens_accepted": 4,
                    "eager_tokens_rejected": 0,
                    "eager_tokens_invalidated": 0,
                    "eager_apply_dry_run_proposal_len_by_proposal_id": {
                        str(pid): 4 for pid in candidate
                    },
                }
            )
    records.append(
        {
            "execution_mode": "dual_batch_pearl",
            "dual_batch_enabled": True,
            "normal_gamma": 4,
            "target_eager_set": [],
            "enable_eager_commit_readiness_dry_run": True,
            "enable_eager_commit_ready_only": True,
            "eager_commit_enabled": True,
            "eager_commit_source": "phase1h5e3_takeover_lane",
            "eager_commit_side": "target",
            "step_id": 99,
            "plan_id": 199,
            "eager_commit_candidate_proposal_ids": candidate,
            "eager_commit_ready_proposal_ids": committed,
            "eager_commit_not_ready_proposal_ids": skipped,
            "eager_committed_proposal_ids": [],
            "eager_committed_seq_ids": [],
            "eager_commit_skipped_proposal_ids": skipped,
            "eager_commit_skip_reason_by_proposal_id": {
                str(pid): "not_full_accept" for pid in skipped
            },
            "eager_apply_dry_run_proposal_len_by_proposal_id": {str(pid): 4 for pid in candidate},
            "lane_exclusion_applied_proposal_ids": candidate,
            "lane_exclusion_applied_seq_ids": [10, 11, 12, 13, 14, 20, 21, 22],
            "target_eager_verify_proposal_ids_dry_run": candidate,
            "target_eager_verify_seq_ids_dry_run": [10, 11, 12, 13, 14, 20, 21, 22],
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [10, 11, 12, 13, 14, 20, 21, 22],
            "eager_transfer_payload_len": 5320,
            "eager_result_transfer_payload_len": 248,
            "eager_tokens_verified": 0,
            "eager_tokens_accepted": 0,
            "eager_tokens_rejected": 0,
            "eager_tokens_invalidated": 0,
        }
    )
    return records


def run_synthetic_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline_path = root / "baseline.json"
        eager_path = root / "eager.json"
        eager_trace_path = root / "eager_trace.json"
        out_path = root / "compare.json"
        baseline_path.write_text(
            json.dumps(synthetic_payload(total_output_tokens=64, goodput=457.0, mean_tpot_ms=34.0)),
            encoding="utf-8",
        )
        eager_path.write_text(
            json.dumps(
                synthetic_payload(
                    total_output_tokens=64,
                    goodput=320.0,
                    mean_tpot_ms=42.0,
                    stale_accounting=True,
                )
            ),
            encoding="utf-8",
        )
        eager_trace_path.write_text(json.dumps(synthetic_trace_records()), encoding="utf-8")
        baseline_accounting = accounting_from_inputs(baseline_path, None)
        eager_accounting = accounting_from_inputs(eager_path, eager_trace_path)
        row = compact_row("eager", eager_accounting, baseline_accounting)
        assert row["committed_tokens"] == 20
        assert row["committed_proposals"] == 5
        assert row["committed_token_share_of_output"] == 20 / 64
        assert row["suppressed_slots_per_committed_token"] == 32 / 20
        assert row["replaced_slots_per_committed_token"] == 32 / 20
        assert row["proposal_payload_len_units_per_committed_token"] == 5320 / 20
        assert row["result_payload_len_units_per_committed_token"] == 248 / 20
        assert row["commit_rate_token"] == 20 / 32
        rows = [
            compact_row("baseline", baseline_accounting, baseline_accounting),
            row,
        ]
        out_path.write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
        saved = json.loads(out_path.read_text())
        saved_eager = saved["cases"][1]
        assert saved_eager["committed_token_share_of_output"] == 20 / 64
        assert saved_eager["proposal_payload_len_units_per_committed_token"] == 5320 / 20
    print("Synthetic Phase 1H-6c comparison checks passed.")


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
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--eager-json", type=Path)
    parser.add_argument("--dryrun-json", type=Path)
    parser.add_argument("--baseline-trace", type=Path)
    parser.add_argument("--eager-trace", type=Path)
    parser.add_argument("--dryrun-trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_tests()
        return 0
    missing = [
        flag
        for flag, value in (
            ("--baseline-json", args.baseline_json),
            ("--eager-json", args.eager_json),
        )
        if value is None
    ]
    if missing:
        print(f"ERROR: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2

    cases = [
        ("baseline", args.baseline_json, args.baseline_trace),
        ("eager", args.eager_json, args.eager_trace),
    ]
    if args.dryrun_json:
        cases.append(("dryrun", args.dryrun_json, args.dryrun_trace))

    accountings = [
        (case_name, accounting_from_inputs(result_path, trace_path))
        for case_name, result_path, trace_path in cases
    ]
    baseline_accounting = accountings[0][1]
    rows = [
        compact_row(case_name, accounting, baseline_accounting)
        for case_name, accounting in accountings
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
