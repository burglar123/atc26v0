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
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402


@dataclass(frozen=True)
class RegressionCase:
    name: str
    limit_requests: int
    eval_batch_size: int
    gamma: int
    max_eager_requests_per_step: int
    max_eager_tokens_per_step: int
    max_eager_tokens_per_request: int
    warmup_iters: int
    description: str


CASE_PRESETS: dict[str, RegressionCase] = {
    "baseline": RegressionCase(
        name="baseline",
        limit_requests=32,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        description="Known-good 1H-6a baseline.",
    ),
    "warmup_off": RegressionCase(
        name="warmup_off",
        limit_requests=32,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=0,
        description="Baseline without warmup.",
    ),
    "gamma2": RegressionCase(
        name="gamma2",
        limit_requests=32,
        eval_batch_size=32,
        gamma=2,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=4,
        max_eager_tokens_per_request=2,
        warmup_iters=1,
        description="Smaller gamma and matching eager-token limits.",
    ),
    "limit64": RegressionCase(
        name="limit64",
        limit_requests=64,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        description="Larger request limit.",
    ),
    "eager_pressure": RegressionCase(
        name="eager_pressure",
        limit_requests=32,
        eval_batch_size=32,
        gamma=4,
        max_eager_requests_per_step=4,
        max_eager_tokens_per_step=16,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        description="More eager pressure per step.",
    ),
    "batch16": RegressionCase(
        name="batch16",
        limit_requests=32,
        eval_batch_size=16,
        gamma=4,
        max_eager_requests_per_step=2,
        max_eager_tokens_per_step=8,
        max_eager_tokens_per_request=4,
        warmup_iters=1,
        description="Smaller eval batch.",
    ),
}


REAL_COMMIT_CHECKERS = [
    ("benchmark/check_eager_commit_ready_only.py", "trace"),
    ("benchmark/check_multislo_result.py", "result"),
    ("benchmark/check_eager_performance_accounting.py", "trace_result"),
]

DRY_RUN_CHECKERS = [
    "benchmark/check_eager_lane_exclusion_dry_run.py",
    "benchmark/check_eager_schedule_dry_run.py",
    "benchmark/check_eager_verify_dry_run.py",
    "benchmark/check_eager_apply_dry_run.py",
    "benchmark/check_eager_result_transfer_dry_run.py",
    "benchmark/check_eager_sync_apply_dry_run.py",
    "benchmark/check_eager_commit_readiness_dry_run.py",
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


def apply_overrides(case: RegressionCase, args: argparse.Namespace) -> RegressionCase:
    updates: dict[str, int] = {}
    for field in (
        "limit_requests",
        "eval_batch_size",
        "gamma",
        "max_eager_requests_per_step",
        "max_eager_tokens_per_step",
        "max_eager_tokens_per_request",
        "warmup_iters",
    ):
        value = getattr(args, field)
        if value is not None:
            updates[field] = int(value)
    if not updates:
        return case
    return replace(case, **updates)


def eval_command(
    args: argparse.Namespace,
    case: RegressionCase,
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
        "--eager-policy",
        args.eager_policy,
        "--max-eager-requests-per-step",
        str(case.max_eager_requests_per_step),
        "--max-eager-tokens-per-step",
        str(case.max_eager_tokens_per_step),
        "--max-eager-tokens-per-request",
        str(case.max_eager_tokens_per_request),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-iters",
        str(case.warmup_iters),
        "--warmup-max-tokens",
        str(args.warmup_max_tokens),
    ]
    maybe_add_int(command, "--limit-requests", case.limit_requests)
    maybe_add_int(command, "--eval-batch-size", case.eval_batch_size)
    if args.decode_ready:
        command.append("--decode-ready")
    if args.ignore_eos:
        command.append("--ignore-eos")
    if real_commit:
        command.append("--enable-eager-commit-ready-only")
    else:
        command.append("--enable-eager-commit-readiness-dry-run")
    for extra_arg in args.extra_eval_arg:
        command.append(extra_arg)
    return command


def checker_command(args: argparse.Namespace, checker: str, *paths: Path) -> list[str]:
    return [args.python, checker, *(str(path) for path in paths)]


def load_accounting_summary(engine_trace: Path, result_json: Path) -> dict[str, Any]:
    records = load_trace(engine_trace)
    result_payload = load_json(result_json) if result_json.exists() else {}
    accounting = aggregate_performance_accounting(records, result_payload)
    return {
        "committed_proposal_count": accounting.get("eager_committed_proposal_count"),
        "committed_token_count": accounting.get("eager_committed_token_count"),
        "candidate_proposal_count": accounting.get("eager_candidate_proposal_count"),
        "candidate_token_count": accounting.get("eager_candidate_token_count"),
        "commit_rate_by_proposal": accounting.get("eager_commit_rate_by_proposal"),
        "commit_rate_by_token": accounting.get("eager_commit_rate_by_token"),
        "target_actual_eager_verified_token_increment_sum": accounting.get(
            "target_actual_eager_verified_token_increment_sum"
        ),
        "target_actual_eager_accepted_token_increment_sum": accounting.get(
            "target_actual_eager_accepted_token_increment_sum"
        ),
        "skipped_proposal_count": accounting.get("eager_skipped_proposal_count"),
        "skip_reason_counts": accounting.get("eager_skip_reason_counts"),
        "repeated_commit_proposal_ids": accounting.get("repeated_commit_proposal_ids"),
        "missing_buffered_proposal_unexpected_count": accounting.get(
            "missing_buffered_proposal_unexpected_count"
        ),
        "engine_elapsed_s": accounting.get("engine_elapsed_s"),
        "goodput_tokens_per_s": accounting.get("goodput_tokens_per_s"),
        "mean_tpot_ms": accounting.get("mean_tpot_ms"),
        "normal_draft_token_slots_suppressed": accounting.get("normal_draft_token_slots_suppressed"),
        "target_normal_verify_token_slots_replaced_by_eager": accounting.get(
            "target_normal_verify_token_slots_replaced_by_eager"
        ),
        "eager_proposal_transfer_payload_bytes": accounting.get("eager_proposal_transfer_payload_bytes"),
        "eager_result_transfer_payload_bytes": accounting.get("eager_result_transfer_payload_bytes"),
        "eager_proposal_transfer_payload_len_units": accounting.get(
            "eager_proposal_transfer_payload_len_units"
        ),
        "eager_result_transfer_payload_len_units": accounting.get(
            "eager_result_transfer_payload_len_units"
        ),
        "timing_available": accounting.get("timing_available"),
        "total_eager_overhead_time_ms": accounting.get("total_eager_overhead_time_ms"),
    }


def run_real_case(
    args: argparse.Namespace,
    case: RegressionCase,
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
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "check_status": check_status,
    }
    if status == 0 and not args.print_only and engine_trace.exists():
        row.update(load_accounting_summary(engine_trace, result_json))
    return status, row


def run_dry_run_case(
    args: argparse.Namespace,
    case: RegressionCase,
    env: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / f"dryrun_{case.name}"
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)

    status = run_command(
        eval_command(args, case, result_json=result_json, engine_trace=engine_trace, real_commit=False),
        env,
        args.print_only,
    )
    check_status = "eval_failed" if status else "pass"
    if status == 0:
        for checker in DRY_RUN_CHECKERS:
            checker_status = run_command(checker_command(args, checker, engine_trace), env, args.print_only)
            if checker_status != 0:
                status = checker_status
                check_status = f"failed:{checker}"
                break
    row = {
        "case_name": f"dryrun_{case.name}",
        "description": "Dry-run regression chain without real eager commit.",
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "check_status": check_status,
    }
    return status, row


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
    for name, case in CASE_PRESETS.items():
        print(
            f"{name}: limit={case.limit_requests}, batch={case.eval_batch_size}, "
            f"gamma={case.gamma}, eager_requests={case.max_eager_requests_per_step}, "
            f"eager_tokens_step={case.max_eager_tokens_per_step}, "
            f"eager_tokens_request={case.max_eager_tokens_per_request}, "
            f"warmup={case.warmup_iters} -- {case.description}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 1H-6b local GPU regression cases for guarded real one-shot eager commit. "
            "Omitting --cuda-visible-devices preserves the caller's CUDA_VISIBLE_DEVICES."
        )
    )
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h6b_regression")
    parser.add_argument("--cases", default="baseline", help="Comma-separated preset names, or all.")
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
    parser.add_argument("--skip-dry-run", action="store_true", help="Do not run the separate dry-run checker chain.")
    parser.add_argument("--dry-run-case", default="baseline", help="Preset used for the separate dry-run chain.")
    parser.add_argument("--print-only", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--extra-eval-arg", action="append", default=[], help="Append one raw argument to eval_multi_slo.py.")

    # Optional overrides for all selected presets.
    parser.add_argument("--limit-requests", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=int, default=None)
    parser.add_argument("--max-eager-requests-per-step", type=int, default=None)
    parser.add_argument("--max-eager-tokens-per-step", type=int, default=None)
    parser.add_argument("--max-eager-tokens-per-request", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
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
        if args.dry_run_case not in CASE_PRESETS:
            raise ValueError(f"unknown --dry-run-case {args.dry_run_case!r}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    rows: list[dict[str, Any]] = []
    status = 0
    for case_name in case_names:
        case = apply_overrides(CASE_PRESETS[case_name], args)
        case_status, row = run_real_case(args, case, env)
        rows.append(row)
        status = max(status, case_status)

    if not args.skip_dry_run:
        dry_case = apply_overrides(CASE_PRESETS[args.dry_run_case], args)
        dry_status, dry_row = run_dry_run_case(args, dry_case, env)
        rows.append(dry_row)
        status = max(status, dry_status)

    if not args.print_only:
        summary_path = Path(args.out_root) / "phase1h6b_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"cases": rows}, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote Phase 1H-6b summary: {summary_path}")

    print("\nPhase 1H-6b summary:")
    for row in rows:
        print(
            f"- {row['case_name']}: {row['check_status']} "
            f"committed={row.get('committed_proposal_count', 'n/a')} "
            f"tokens={row.get('committed_token_count', 'n/a')} "
            f"target_verified={row.get('target_actual_eager_verified_token_increment_sum', 'n/a')} "
            f"rate_token={row.get('commit_rate_by_token', 'n/a')} "
            f"goodput={row.get('goodput_tokens_per_s', 'n/a')}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
