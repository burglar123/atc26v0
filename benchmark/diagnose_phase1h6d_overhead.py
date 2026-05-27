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

from benchmark.check_eager_performance_accounting import (  # noqa: E402
    DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD,
    DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD,
    aggregate_performance_accounting,
    load_json,
    load_trace,
    performance_warnings,
    safe_div,
    synthetic_records,
    synthetic_result_payload,
)
from benchmark.compare_phase1h6c_performance import accounting_from_inputs  # noqa: E402


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def load_optional_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def build_diagnosis(
    baseline_accounting: dict[str, Any],
    eager_accounting: dict[str, Any],
    *,
    dryrun_accounting: dict[str, Any] | None = None,
    summary_payload: dict[str, Any] | None = None,
    low_committed_share_threshold: float = DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD,
    high_payload_len_per_committed_token_threshold: float = (
        DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD
    ),
) -> dict[str, Any]:
    baseline_goodput = float_value(baseline_accounting.get("goodput_tokens_per_s"), 0.0)
    eager_goodput = float_value(eager_accounting.get("goodput_tokens_per_s"), 0.0)
    baseline_tpot = float_value(baseline_accounting.get("mean_tpot_ms"), 0.0)
    eager_tpot = float_value(eager_accounting.get("mean_tpot_ms"), 0.0)
    committed_tokens = int_value(eager_accounting.get("eager_committed_token_count"), 0)
    candidate_tokens = int_value(eager_accounting.get("eager_candidate_token_count"), 0)
    ready_tokens = int_value(eager_accounting.get("eager_ready_token_count"), 0)

    warnings = performance_warnings(
        eager_accounting,
        baseline_goodput_tokens_per_s=baseline_goodput,
        low_committed_share_threshold=low_committed_share_threshold,
        high_payload_len_per_committed_token_threshold=high_payload_len_per_committed_token_threshold,
    )
    if (
        committed_tokens > 0
        and int_value(eager_accounting.get("target_normal_verify_token_slots_replaced_by_eager"), 0)
        > committed_tokens
    ):
        warnings.append("replaced_slots_exceed_committed_tokens")
    if (
        committed_tokens > 0
        and int_value(eager_accounting.get("eager_result_transfer_payload_len_units"), 0)
        and float_value(eager_accounting.get("result_payload_len_units_per_committed_token"), 0.0) > 16.0
    ):
        warnings.append("result_payload_len_units_high_per_committed_token")

    diagnosis = {
        "baseline": {
            "total_output_tokens": baseline_accounting.get("total_output_tokens", 0),
            "engine_elapsed_s": baseline_accounting.get("engine_elapsed_s", 0.0),
            "goodput_tokens_per_s": baseline_goodput,
            "mean_tpot_ms": baseline_tpot,
        },
        "eager": {
            "total_output_tokens": eager_accounting.get("total_output_tokens", 0),
            "engine_elapsed_s": eager_accounting.get("engine_elapsed_s", 0.0),
            "goodput_tokens_per_s": eager_goodput,
            "mean_tpot_ms": eager_tpot,
            "eager_committed_token_count": committed_tokens,
            "eager_committed_proposal_count": eager_accounting.get("eager_committed_proposal_count", 0),
            "eager_candidate_token_count": candidate_tokens,
            "eager_ready_token_count": ready_tokens,
            "normal_draft_token_slots_suppressed": eager_accounting.get(
                "normal_draft_token_slots_suppressed",
                0,
            ),
            "target_normal_verify_token_slots_replaced_by_eager": eager_accounting.get(
                "target_normal_verify_token_slots_replaced_by_eager",
                0,
            ),
            "eager_proposal_transfer_payload_len_units": eager_accounting.get(
                "eager_proposal_transfer_payload_len_units",
                0,
            ),
            "eager_result_transfer_payload_len_units": eager_accounting.get(
                "eager_result_transfer_payload_len_units",
                0,
            ),
            "timing_available": eager_accounting.get("timing_available", False),
            "eager_transfer_time_ms": eager_accounting.get("eager_transfer_time_ms", 0.0),
            "eager_result_transfer_time_ms": eager_accounting.get(
                "eager_result_transfer_time_ms",
                0.0,
            ),
            "eager_commit_readiness_time_ms": eager_accounting.get(
                "eager_commit_readiness_time_ms",
                0.0,
            ),
            "eager_commit_time_ms": eager_accounting.get("eager_commit_time_ms", 0.0),
            "eager_accounting_summary_time_ms": eager_accounting.get(
                "eager_accounting_summary_time_ms",
                0.0,
            ),
            "total_eager_overhead_time_ms": eager_accounting.get(
                "total_eager_overhead_time_ms",
                0.0,
            ),
            "payload_bytes_available": eager_accounting.get("payload_bytes_available", False),
        },
        "derived": {
            "committed_token_share_of_output": eager_accounting.get(
                "committed_token_share_of_output",
                0.0,
            ),
            "candidate_token_share_of_output": eager_accounting.get(
                "candidate_token_share_of_output",
                0.0,
            ),
            "suppressed_slots_per_committed_token": eager_accounting.get(
                "suppressed_slots_per_committed_token",
                0.0,
            ),
            "replaced_slots_per_committed_token": eager_accounting.get(
                "replaced_slots_per_committed_token",
                0.0,
            ),
            "proposal_payload_len_units_per_committed_token": eager_accounting.get(
                "proposal_payload_len_units_per_committed_token",
                0.0,
            ),
            "result_payload_len_units_per_committed_token": eager_accounting.get(
                "result_payload_len_units_per_committed_token",
                0.0,
            ),
            "goodput_delta_tokens_per_s": eager_goodput - baseline_goodput,
            "goodput_ratio_eager_vs_baseline": safe_div(eager_goodput, baseline_goodput),
            "mean_tpot_delta_ms": eager_tpot - baseline_tpot,
            "mean_tpot_ratio_eager_vs_baseline": safe_div(eager_tpot, baseline_tpot),
            "committed_tokens_per_candidate_token": safe_div(committed_tokens, candidate_tokens),
            "ready_tokens_per_candidate_token": safe_div(ready_tokens, candidate_tokens),
            "committed_tokens_per_ready_token": safe_div(committed_tokens, ready_tokens),
        },
        "funnel": {
            "candidate_proposals": eager_accounting.get("eager_candidate_proposal_count", 0),
            "candidate_tokens": candidate_tokens,
            "ready_proposals": eager_accounting.get("eager_ready_proposal_count", 0),
            "ready_tokens": ready_tokens,
            "committed_proposals": eager_accounting.get("eager_committed_proposal_count", 0),
            "committed_tokens": committed_tokens,
            "skipped_proposals": eager_accounting.get("eager_skipped_proposal_count", 0),
            "skip_reason_counts": eager_accounting.get("eager_skip_reason_counts", {}),
        },
        "warnings": sorted(set(warnings)),
        "dryrun": dryrun_accounting or {},
        "summary_json": summary_payload or {},
    }
    return diagnosis


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_report(diagnosis: dict[str, Any]) -> None:
    print("Phase 1H-6d overhead diagnosis")
    print("\nBaseline:")
    for key, value in diagnosis["baseline"].items():
        print(f"  {key}: {fmt(value)}")
    print("\nEager:")
    for key, value in diagnosis["eager"].items():
        print(f"  {key}: {fmt(value)}")
    print("\nDerived:")
    for key, value in diagnosis["derived"].items():
        print(f"  {key}: {fmt(value)}")
    print("\nFunnel:")
    for key, value in diagnosis["funnel"].items():
        print(f"  {key}: {value if isinstance(value, dict) else fmt(value)}")
    print("\nWarnings:")
    if diagnosis["warnings"]:
        for warning in diagnosis["warnings"]:
            print(f"  - {warning}")
    else:
        print("  none")


def synthetic_baseline_payload() -> dict[str, Any]:
    payload = synthetic_result_payload()
    payload["metrics"]["engine_elapsed_s"] = 0.14
    payload["metrics"]["overall"]["total_output_tokens"] = 64
    payload["metrics"]["overall"]["goodput_tokens_per_s"] = 457.0
    payload["metrics"]["overall"]["mean_tpot_ms"] = 34.0
    payload["eager_performance_accounting"] = aggregate_performance_accounting([], payload)
    return payload


def synthetic_eager_payload_and_trace() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = synthetic_result_payload()
    payload["metrics"]["engine_elapsed_s"] = 0.2
    payload["metrics"]["overall"]["total_output_tokens"] = 64
    payload["metrics"]["overall"]["goodput_tokens_per_s"] = 320.0
    payload["metrics"]["overall"]["mean_tpot_ms"] = 42.0
    records = synthetic_records()
    payload["eager_performance_accounting"] = aggregate_performance_accounting(records, payload)
    return payload, records


def run_synthetic_tests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline_payload = synthetic_baseline_payload()
        eager_payload, eager_records = synthetic_eager_payload_and_trace()
        baseline_path = root / "baseline.json"
        eager_path = root / "eager.json"
        trace_path = root / "eager_trace.json"
        out_path = root / "diagnosis.json"
        baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
        eager_path.write_text(json.dumps(eager_payload), encoding="utf-8")
        trace_path.write_text(json.dumps(eager_records), encoding="utf-8")

        baseline_accounting = accounting_from_inputs(baseline_path, None)
        eager_accounting = aggregate_performance_accounting(load_trace(trace_path), load_json(eager_path))
        diagnosis = build_diagnosis(baseline_accounting, eager_accounting)
        assert diagnosis["derived"]["goodput_ratio_eager_vs_baseline"] < 1.0
        assert "eager_goodput_below_baseline" in diagnosis["warnings"]
        assert diagnosis["derived"]["suppressed_slots_per_committed_token"] == 2.0
        out_path.write_text(json.dumps(diagnosis), encoding="utf-8")
        assert out_path.exists()
    print("Synthetic Phase 1H-6d diagnosis checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phase 1H-6d eager overhead attribution.")
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--eager-json", type=Path)
    parser.add_argument("--eager-trace", type=Path)
    parser.add_argument("--dryrun-trace", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--low-committed-share-threshold", type=float, default=DEFAULT_LOW_COMMITTED_SHARE_THRESHOLD)
    parser.add_argument(
        "--high-payload-len-per-committed-token-threshold",
        type=float,
        default=DEFAULT_HIGH_PAYLOAD_LEN_PER_COMMITTED_TOKEN_THRESHOLD,
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_tests()
        return 0
    missing = [
        name
        for name, value in (
            ("--baseline-json", args.baseline_json),
            ("--eager-json", args.eager_json),
            ("--eager-trace", args.eager_trace),
        )
        if value is None
    ]
    if missing:
        print(f"ERROR: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2

    baseline_accounting = accounting_from_inputs(args.baseline_json, None)
    eager_accounting = aggregate_performance_accounting(
        load_trace(args.eager_trace),
        load_json(args.eager_json),
    )
    dryrun_accounting = None
    if args.dryrun_trace:
        dryrun_accounting = aggregate_performance_accounting(load_trace(args.dryrun_trace), {})
    diagnosis = build_diagnosis(
        baseline_accounting,
        eager_accounting,
        dryrun_accounting=dryrun_accounting,
        summary_payload=load_optional_summary(args.summary_json),
        low_committed_share_threshold=args.low_committed_share_threshold,
        high_payload_len_per_committed_token_threshold=(
            args.high_payload_len_per_committed_token_threshold
        ),
    )
    print_report(diagnosis)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(diagnosis, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote diagnosis JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
