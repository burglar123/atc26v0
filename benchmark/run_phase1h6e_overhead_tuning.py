#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import load_trace  # noqa: E402
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    load_json,
)


@dataclass(frozen=True)
class TuningCase:
    name: str
    limit_requests: int
    eval_batch_size: int
    gamma: int
    max_eager_requests_per_step: int
    max_eager_tokens_per_step: int
    max_eager_tokens_per_request: int
    warmup_iters: int
    eager_trace_level: str
    description: str


CASE_PRESETS: dict[str, TuningCase] = {
    "baseline_current": TuningCase(
        name="baseline_current",
        limit_requests=32,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        eager_trace_level="full",
        description="Current known-good 6d eager commit configuration.",
    ),
    "trace_minimal": TuningCase(
        name="trace_minimal",
        limit_requests=32,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        eager_trace_level="minimal",
        description="Baseline config with minimal eager trace output.",
    ),
    "eager_pressure_moderate": TuningCase(
        name="eager_pressure_moderate",
        limit_requests=32,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=4,
        max_eager_tokens_per_step=16,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        eager_trace_level="summary",
        description="Moderately higher one-shot eager candidate pressure.",
    ),
    "gamma2_trace_minimal": TuningCase(
        name="gamma2_trace_minimal",
        limit_requests=32,
        eval_batch_size=32,
        gamma=2,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=4,
        max_eager_tokens_per_request=2,
        warmup_iters=1,
        eager_trace_level="minimal",
        description="Smaller gamma with matching eager limits and minimal trace.",
    ),
    "limit64_trace_minimal": TuningCase(
        name="limit64_trace_minimal",
        limit_requests=64,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        eager_trace_level="minimal",
        description="Larger request limit with minimal trace.",
    ),
}


REAL_COMMIT_CHECKERS = [
    ("benchmark/check_eager_commit_ready_only.py", "trace"),
    ("benchmark/check_multislo_result.py", "result"),
    ("benchmark/check_eager_performance_accounting.py", "trace_result"),
]


def format_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str], env: dict[str, str], print_only: bool) -> int:
    print(f"\n$ {format_cmd(command)}", flush=True)
    if print_only:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def maybe_add_int(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def apply_overrides(case: TuningCase, args: argparse.Namespace) -> TuningCase:
    updates: dict[str, Any] = {}
    for field in (
        "limit_requests",
        "eval_batch_size",
        "gamma",
        "max_eager_requests_per_step",
        "max_eager_tokens_per_step",
        "max_eager_tokens_per_request",
        "warmup_iters",
        "eager_trace_level",
    ):
        value = getattr(args, field)
        if value is not None:
            updates[field] = value
    if not updates:
        return case
    return replace(case, **updates)


def eval_command(
    args: argparse.Namespace,
    case: TuningCase,
    *,
    result_json: Path,
    engine_trace: Path,
    real_commit: bool,
) -> list[str]:
    command = [
        args.python,
        "benchmark/eval_multi_slo.py",
        "--draft-model",
        args.draft_model,
        "--target-model",
        args.target_model,
        "--draft-tp",
        str(args.draft_tp),
        "--target-tp",
        str(args.target_tp),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--execution-mode",
        "dual_batch_pearl",
        "--gamma",
        str(case.gamma),
        "--workload-in",
        args.workload_in,
        "--out",
        str(result_json),
        "--engine-trace-out",
        str(engine_trace),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-iters",
        str(case.warmup_iters),
        "--warmup-max-tokens",
        str(args.warmup_max_tokens),
        "--eager-trace-level",
        case.eager_trace_level,
    ]
    maybe_add_int(command, "--limit-requests", case.limit_requests)
    maybe_add_int(command, "--eval-batch-size", case.eval_batch_size)
    if args.decode_ready:
        command.append("--decode-ready")
    if args.ignore_eos:
        command.append("--ignore-eos")
    if real_commit:
        command.extend(
            [
                "--enable-eager-commit-ready-only",
                "--eager-policy",
                args.eager_policy,
                "--max-eager-requests-per-step",
                str(case.max_eager_requests_per_step),
                "--max-eager-tokens-per-step",
                str(case.max_eager_tokens_per_step),
                "--max-eager-tokens-per-request",
                str(case.max_eager_tokens_per_request),
            ]
        )
    for extra_arg in args.extra_eval_arg:
        command.append(extra_arg)
    return command


def checker_command(args: argparse.Namespace, checker: str, *paths: Path) -> list[str]:
    return [args.python, checker, *(str(path) for path in paths)]


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_accounting_summary(engine_trace: Path, result_json: Path) -> dict[str, Any]:
    records = load_trace(engine_trace) if engine_trace.exists() else []
    result_payload = load_json(result_json) if result_json.exists() else {}
    accounting = aggregate_performance_accounting(records, result_payload)
    skip_reasons = accounting.get("eager_skip_reason_counts", {})
    not_full_accept_count = int(skip_reasons.get("not_full_accept", 0)) if isinstance(skip_reasons, dict) else 0
    return {
        "goodput_tokens_per_s": accounting.get("goodput_tokens_per_s"),
        "mean_tpot_ms": accounting.get("mean_tpot_ms"),
        "engine_elapsed_s": accounting.get("engine_elapsed_s"),
        "total_output_tokens": accounting.get("total_output_tokens"),
        "candidate_tokens": accounting.get("eager_candidate_token_count"),
        "ready_tokens": accounting.get("eager_ready_token_count"),
        "committed_tokens": accounting.get("eager_committed_token_count"),
        "committed_proposal_count": accounting.get("eager_committed_proposal_count"),
        "committed_token_share_of_output": accounting.get("committed_token_share_of_output"),
        "commit_rate_by_token": accounting.get("eager_commit_rate_by_token"),
        "not_full_accept_count": not_full_accept_count,
        "normal_draft_token_slots_suppressed": accounting.get("normal_draft_token_slots_suppressed"),
        "target_normal_verify_token_slots_replaced_by_eager": accounting.get(
            "target_normal_verify_token_slots_replaced_by_eager"
        ),
        "proposal_payload_len_units_per_committed_token": accounting.get(
            "proposal_payload_len_units_per_committed_token"
        ),
        "result_payload_len_units_per_committed_token": accounting.get(
            "result_payload_len_units_per_committed_token"
        ),
        "timing_available": accounting.get("timing_available"),
        "eager_transfer_time_ms": accounting.get("eager_transfer_time_ms"),
        "eager_result_transfer_time_ms": accounting.get("eager_result_transfer_time_ms"),
        "eager_commit_readiness_time_ms": accounting.get("eager_commit_readiness_time_ms"),
        "eager_commit_time_ms": accounting.get("eager_commit_time_ms"),
        "eager_accounting_summary_time_ms": accounting.get("eager_accounting_summary_time_ms"),
        "total_eager_overhead_time_ms": accounting.get("total_eager_overhead_time_ms"),
        "performance_warnings": accounting.get("performance_warnings", []),
        "repeated_commit_proposal_ids": accounting.get("repeated_commit_proposal_ids", []),
        "missing_buffered_proposal_unexpected_count": accounting.get(
            "missing_buffered_proposal_unexpected_count"
        ),
    }


def run_real_case(
    args: argparse.Namespace,
    case: TuningCase,
    env: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)

    status = run_command(
        eval_command(args, case, result_json=result_json, engine_trace=engine_trace, real_commit=True),
        env,
        args.print_only,
    )
    check_status = "eval_failed" if status else "pass"
    if status == 0:
        for checker, input_kind in REAL_COMMIT_CHECKERS:
            if input_kind == "result":
                checker_status = run_command(checker_command(args, checker, result_json), env, args.print_only)
            elif input_kind == "trace_result":
                checker_status = run_command(checker_command(args, checker, engine_trace, result_json), env, args.print_only)
            else:
                checker_status = run_command(checker_command(args, checker, engine_trace), env, args.print_only)
            if checker_status != 0:
                status = checker_status
                check_status = f"failed:{checker}"
                break

    row: dict[str, Any] = {
        "case_name": case.name,
        "description": case.description,
        "eager_trace_level": case.eager_trace_level,
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "check_status": check_status,
    }
    if status == 0 and not args.print_only:
        row.update(load_accounting_summary(engine_trace, result_json))
    return status, row


def run_default_baseline(
    args: argparse.Namespace,
    reference_case: TuningCase,
    env: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / "default_baseline"
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)
    status = run_command(
        eval_command(args, reference_case, result_json=result_json, engine_trace=engine_trace, real_commit=False),
        env,
        args.print_only,
    )
    check_status = "eval_failed" if status else "pass"
    if status == 0:
        for checker, path in (
            ("benchmark/check_dual_batch_trace.py", engine_trace),
            ("benchmark/check_multislo_result.py", result_json),
        ):
            checker_status = run_command(checker_command(args, checker, path), env, args.print_only)
            if checker_status != 0:
                status = checker_status
                check_status = f"failed:{checker}"
                break
    row: dict[str, Any] = {
        "case_name": "default_baseline",
        "description": "Default dual_batch_pearl baseline without eager commit.",
        "eager_trace_level": reference_case.eager_trace_level,
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "check_status": check_status,
    }
    if status == 0 and not args.print_only:
        row.update(load_accounting_summary(engine_trace, result_json))
    return status, row


def maybe_run_diagnosis(
    args: argparse.Namespace,
    baseline_json: Path | None,
    row: dict[str, Any],
    env: dict[str, str],
) -> int:
    if baseline_json is None or args.print_only:
        return 0
    result_json = Path(str(row.get("result_json", "")))
    engine_trace = Path(str(row.get("engine_trace", "")))
    if not result_json.exists() or not engine_trace.exists():
        return 0
    out_path = result_json.parent / "phase1h6d_diagnosis.json"
    return run_command(
        [
            args.python,
            "benchmark/diagnose_phase1h6d_overhead.py",
            "--baseline-json",
            str(baseline_json),
            "--eager-json",
            str(result_json),
            "--eager-trace",
            str(engine_trace),
            "--out",
            str(out_path),
        ],
        env,
        args.print_only,
    )


def select_cases(raw_cases: str) -> list[str]:
    names = [item.strip() for item in raw_cases.split(",") if item.strip()]
    if not names:
        raise ValueError("--cases must name at least one preset")
    if names == ["all"]:
        return list(CASE_PRESETS)
    unknown = [name for name in names if name not in CASE_PRESETS]
    if unknown:
        known = ", ".join(sorted(CASE_PRESETS))
        raise ValueError(f"unknown case(s): {unknown}; known cases: {known}, all")
    return names


def print_case_list() -> None:
    for case in CASE_PRESETS.values():
        print(
            f"{case.name}: limit={case.limit_requests}, batch={case.eval_batch_size}, "
            f"gamma={case.gamma}, eager_requests={case.max_eager_requests_per_step}, "
            f"eager_tokens_step={case.max_eager_tokens_per_step}, "
            f"eager_tokens_request={case.max_eager_tokens_per_request}, "
            f"warmup={case.warmup_iters}, trace={case.eager_trace_level} -- {case.description}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 1H-6e overhead/trigger-rate tuning cases for guarded real one-shot eager commit."
        )
    )
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h6e_overhead_tuning")
    parser.add_argument("--cases", default="baseline_current,trace_minimal", help="Comma-separated preset names, or all.")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--eager-policy", default="tight_only")
    parser.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True)
    parser.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--decode-ready", dest="decode_ready", action="store_true", default=True)
    parser.add_argument("--no-decode-ready", dest="decode_ready", action="store_false")
    parser.add_argument("--run-default-baseline", action="store_true")
    parser.add_argument("--default-result-json", type=Path, default=None)
    parser.add_argument("--print-only", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--extra-eval-arg", action="append", default=[], help="Append one raw argument to eval_multi_slo.py.")
    parser.add_argument("--limit-requests", type=int, default=None, help="Override selected case request limit.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override selected case eval batch size.")
    parser.add_argument("--gamma", type=int, default=None, help="Override selected case gamma.")
    parser.add_argument("--max-eager-requests-per-step", type=int, default=None, help="Override selected case eager request limit.")
    parser.add_argument("--max-eager-tokens-per-step", type=int, default=None, help="Override selected case eager token-step limit.")
    parser.add_argument("--max-eager-tokens-per-request", type=int, default=None, help="Override selected case eager token-request limit.")
    parser.add_argument("--warmup-iters", type=int, default=None, help="Override selected case warmup iterations.")
    parser.add_argument(
        "--eager-trace-level",
        choices=["full", "summary", "minimal"],
        default=None,
        help="Override selected case eager trace level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print_case_list()
        return 0
    missing_required = [
        flag
        for flag, value in (
            ("--draft-model", args.draft_model),
            ("--target-model", args.target_model),
            ("--workload-in", args.workload_in),
        )
        if not value
    ]
    if missing_required:
        print(f"ERROR: missing required arguments: {', '.join(missing_required)}", file=sys.stderr)
        return 2
    try:
        case_names = select_cases(args.cases)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    rows: list[dict[str, Any]] = []
    status = 0
    baseline_json_for_diagnosis = args.default_result_json
    baseline_goodput: float | None = None

    if args.run_default_baseline:
        default_status, default_row = run_default_baseline(
            args,
            apply_overrides(CASE_PRESETS["baseline_current"], args),
            env,
        )
        rows.append(default_row)
        status = max(status, default_status)
        if default_status == 0 and not args.print_only:
            baseline_json_for_diagnosis = Path(str(default_row["result_json"]))
            baseline_goodput = float_value(default_row.get("goodput_tokens_per_s"), 0.0)
    elif args.default_result_json and args.default_result_json.exists():
        default_payload = load_json(args.default_result_json)
        metrics = aggregate_performance_accounting([], default_payload)
        baseline_goodput = float_value(metrics.get("goodput_tokens_per_s"), 0.0)

    for case_name in case_names:
        case_status, row = run_real_case(args, apply_overrides(CASE_PRESETS[case_name], args), env)
        rows.append(row)
        status = max(status, case_status)
        if case_status == 0 and baseline_json_for_diagnosis is not None:
            diag_status = maybe_run_diagnosis(args, baseline_json_for_diagnosis, row, env)
            if diag_status != 0:
                row["check_status"] = f"failed:benchmark/diagnose_phase1h6d_overhead.py"
                status = max(status, diag_status)

    if baseline_goodput is None:
        for row in rows:
            if row.get("case_name") == "baseline_current":
                baseline_goodput = float_value(row.get("goodput_tokens_per_s"), 0.0)
                break
    for row in rows:
        row_goodput = float_value(row.get("goodput_tokens_per_s"), 0.0)
        row["goodput_ratio_vs_default"] = (
            row_goodput / baseline_goodput if baseline_goodput and row_goodput else 0.0
        )
        notes = list(row.get("performance_warnings") or [])
        if baseline_json_for_diagnosis is None and row.get("case_name") != "default_baseline":
            notes.append("default_reference_unavailable")
        row["notes"] = sorted(set(str(note) for note in notes))

    if not args.print_only:
        summary_path = Path(args.out_root) / "phase1h6e_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"cases": rows}, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote Phase 1H-6e summary: {summary_path}")

    print("\nPhase 1H-6e overhead tuning summary:")
    for row in rows:
        print(
            f"- {row['case_name']}: {row['check_status']} "
            f"goodput={row.get('goodput_tokens_per_s', 'n/a')} "
            f"ratio={row.get('goodput_ratio_vs_default', 'n/a')} "
            f"committed_tokens={row.get('committed_tokens', 'n/a')} "
            f"share={row.get('committed_token_share_of_output', 'n/a')} "
            f"payload_units/token={row.get('proposal_payload_len_units_per_committed_token', 'n/a')} "
            f"timing={row.get('timing_available', 'n/a')}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
