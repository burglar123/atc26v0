#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CachedKVCase:
    name: str
    execution_mode: str
    cached_prefill_mode: str | None
    limit_requests: int
    eval_batch_size: int
    max_tokens: int
    cached_admission_max_active: int
    cache_build_batch_size: int
    enable_full_continuous: bool
    expect_failure: bool
    description: str


CASE_PRESETS: dict[str, CachedKVCase] = {
    "disabled_baseline": CachedKVCase(
        name="disabled_baseline",
        execution_mode="dual_batch_pearl",
        cached_prefill_mode=None,
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=0,
        cache_build_batch_size=0,
        enable_full_continuous=False,
        expect_failure=False,
        description="Decode-ready dual-batch baseline with cached admission disabled.",
    ),
    "metadata_only_fifo_smoke": CachedKVCase(
        name="metadata_only_fifo_smoke",
        execution_mode="dual_batch_pearl",
        cached_prefill_mode="metadata_only",
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=2,
        cache_build_batch_size=0,
        enable_full_continuous=False,
        expect_failure=False,
        description="Existing metadata-only FIFO cached-admission regression.",
    ),
    "in_memory_kv_ar_smoke": CachedKVCase(
        name="in_memory_kv_ar_smoke",
        execution_mode="ar",
        cached_prefill_mode="in_memory_kv",
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=4,
        cache_build_batch_size=8,
        enable_full_continuous=False,
        expect_failure=False,
        description="Target-only in-memory cached KV admission for AR decode.",
    ),
    "in_memory_kv_serialized_smoke": CachedKVCase(
        name="in_memory_kv_serialized_smoke",
        execution_mode="serialized_pearl",
        cached_prefill_mode="in_memory_kv",
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=4,
        cache_build_batch_size=8,
        enable_full_continuous=False,
        expect_failure=False,
        description="Draft and target cached KV admission through serialized PEARL.",
    ),
    "in_memory_kv_parallel_smoke": CachedKVCase(
        name="in_memory_kv_parallel_smoke",
        execution_mode="parallel_pearl",
        cached_prefill_mode="in_memory_kv",
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=4,
        cache_build_batch_size=8,
        enable_full_continuous=False,
        expect_failure=False,
        description="Draft and target cached KV admission through parallel PEARL.",
    ),
    "in_memory_kv_dual_batch_full_continuous_smoke": CachedKVCase(
        name="in_memory_kv_dual_batch_full_continuous_smoke",
        execution_mode="dual_batch_pearl",
        cached_prefill_mode="in_memory_kv",
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=4,
        cache_build_batch_size=8,
        enable_full_continuous=True,
        expect_failure=False,
        description="In-memory cached KV admission through dual-batch full-continuous flags.",
    ),
    "in_memory_kv_max_active_cap": CachedKVCase(
        name="in_memory_kv_max_active_cap",
        execution_mode="parallel_pearl",
        cached_prefill_mode="in_memory_kv",
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=1,
        cache_build_batch_size=8,
        enable_full_continuous=False,
        expect_failure=False,
        description="In-memory cached KV FIFO admission with max_active=1.",
    ),
    "unsupported_mode_negative": CachedKVCase(
        name="unsupported_mode_negative",
        execution_mode="unsupported_mode",
        cached_prefill_mode="in_memory_kv",
        limit_requests=1,
        eval_batch_size=1,
        max_tokens=1,
        cached_admission_max_active=1,
        cache_build_batch_size=1,
        enable_full_continuous=False,
        expect_failure=True,
        description="Negative smoke for unsupported execution_mode fail-loud behavior.",
    ),
}


DEFAULT_CASES = (
    "disabled_baseline,metadata_only_fifo_smoke,in_memory_kv_ar_smoke,"
    "in_memory_kv_serialized_smoke,in_memory_kv_parallel_smoke,"
    "in_memory_kv_dual_batch_full_continuous_smoke,in_memory_kv_max_active_cap"
)


def format_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str], env: dict[str, str], print_only: bool) -> int:
    print(f"\n$ {format_cmd(command)}", flush=True)
    if print_only:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def add_full_continuous_flags(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--enable-eager-commit-ready-only",
            "--enable-continuous-eager-dry-run",
            "--enable-continuous-eager-verify-apply-dry-run",
            "--enable-continuous-eager-commit-depth1-ready-only",
            "--enable-rolling-continuous-eager-dry-run",
            "--enable-rolling-continuous-depth2-commit-ready-only",
            "--enable-rolling-continuous-depth3-shadow-dry-run",
            "--enable-rolling-continuous-depth3-commit-ready-only",
            "--enable-rolling-continuous-depth4-shadow-dry-run",
            "--enable-rolling-continuous-depth4-commit-ready-only",
            "--enable-generic-rolling-runtime-loop",
            "--enable-generic-rolling-apply-path",
            "--enable-full-continuous-eager",
            "--eager-policy",
            args.eager_policy,
            "--max-eager-requests-per-step",
            "2",
            "--max-eager-tokens-per-step",
            "8",
            "--max-eager-tokens-per-request",
            "4",
            "--max-continuous-eager-chain-depth",
            "1",
            "--max-continuous-eager-requests-per-step",
            "2",
            "--max-continuous-eager-tokens-per-step",
            "8",
            "--max-continuous-eager-tokens-per-request",
            "4",
            "--max-rolling-continuous-depth",
            "4",
            "--max-rolling-continuous-draft-children-per-step",
            "2",
            "--max-rolling-continuous-seqs-per-step",
            "2",
            "--eager-trace-level",
            "minimal",
        ]
    )


def eval_command(
    args: argparse.Namespace,
    case: CachedKVCase,
    result_json: Path,
    request_trace: Path,
    engine_trace: Path,
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
        case.execution_mode,
        "--gamma",
        str(args.gamma),
        "--workload-in",
        args.workload_in,
        "--out",
        str(result_json),
        "--trace-out",
        str(request_trace),
        "--engine-trace-out",
        str(engine_trace),
        "--decode-ready",
        "--ignore-eos",
        "--limit-requests",
        str(case.limit_requests),
        "--eval-batch-size",
        str(case.eval_batch_size),
        "--max-tokens",
        str(case.max_tokens),
        "--warmup-iters",
        "0",
    ]
    if case.cached_prefill_mode is not None:
        command.extend(
            [
                "--enable-cached-admission",
                "--cached-prefill-mode",
                case.cached_prefill_mode,
                "--cached-admission-policy",
                "fifo",
                "--cached-admission-max-active",
                str(case.cached_admission_max_active),
                "--cached-admission-arrival-field",
                "arrival_offset_sec",
            ]
        )
        if case.cached_prefill_mode == "in_memory_kv":
            command.extend(["--cache-build-batch-size", str(case.cache_build_batch_size)])
    if case.enable_full_continuous:
        add_full_continuous_flags(command, args)
    for extra_arg in args.extra_eval_arg:
        command.append(extra_arg)
    return command


def checker_commands(args: argparse.Namespace, case: CachedKVCase, result_json: Path, engine_trace: Path) -> list[list[str]]:
    commands = [
        [args.python, "benchmark/check_cached_admission.py", str(result_json)],
        [args.python, "benchmark/check_multislo_result.py", str(result_json)],
    ]
    if case.enable_full_continuous:
        commands.append([args.python, "benchmark/check_full_continuous_max_depth.py", str(engine_trace), str(result_json)])
    return commands


def load_case_summary(result_json: Path) -> dict[str, Any]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    cached = payload.get("cached_admission", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(cached, dict):
        cached = {}
    overall = metrics.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    return {
        "engine_elapsed_s": metrics.get("engine_elapsed_s"),
        "mean_tpot_ms": overall.get("mean_tpot_ms"),
        "cached_admission_enabled": cached.get("cached_admission_enabled", False),
        "cached_prefill_mode": cached.get("cached_prefill_mode"),
        "cached_admission_total_requests": cached.get("cached_admission_total_requests"),
        "cached_admission_total_admitted": cached.get("cached_admission_total_admitted"),
        "cached_admission_total_completed": cached.get("cached_admission_total_completed"),
        "cached_admission_peak_active": cached.get("cached_admission_peak_active"),
        "cached_cache_build_elapsed_s": cached.get("cached_cache_build_elapsed_s"),
        "cached_kv_num_requests": cached.get("cached_kv_num_requests"),
        "cached_kv_total_cpu_bytes": cached.get("cached_kv_total_cpu_bytes"),
    }


def run_case(args: argparse.Namespace, case: CachedKVCase, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    request_trace = case_dir / "request_trace.jsonl"
    engine_trace = case_dir / "engine_trace.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)

    status = run_command(eval_command(args, case, result_json, request_trace, engine_trace), env, args.print_only)
    check_status = "eval_failed" if status else "pass"
    failed_checker: str | None = None
    if case.expect_failure:
        if args.print_only:
            check_status = "print_only_expected_failure"
            status = 0
        else:
            check_status = "expected_failure" if status != 0 else "unexpected_success"
            status = 0 if check_status == "expected_failure" else 1
    elif status == 0 and args.run_checkers:
        for command in checker_commands(args, case, result_json, engine_trace):
            checker_status = run_command(command, env, args.print_only)
            if checker_status != 0:
                status = checker_status
                check_status = f"failed:{command[1]}"
                failed_checker = command[1]
                if args.strict:
                    break

    row: dict[str, Any] = {
        "case_name": case.name,
        "execution_mode": case.execution_mode,
        "cached_admission_enabled": case.cached_prefill_mode is not None,
        "cached_prefill_mode": case.cached_prefill_mode,
        "cached_admission_max_active": case.cached_admission_max_active if case.cached_prefill_mode else None,
        "cache_build_batch_size": case.cache_build_batch_size if case.cached_prefill_mode == "in_memory_kv" else None,
        "full_continuous_smoke": case.enable_full_continuous,
        "check_status": check_status,
        "failed_checker": failed_checker,
        "result_json": str(result_json),
        "request_trace": str(request_trace),
        "engine_trace": str(engine_trace),
        "description": case.description,
    }
    if status == 0 and not args.print_only and not case.expect_failure:
        row.update(load_case_summary(result_json))
    return status, row


def select_cases(raw_cases: str) -> list[str]:
    names = [item.strip() for item in raw_cases.split(",") if item.strip()]
    if names == ["all"]:
        return [name for name, case in CASE_PRESETS.items() if not case.expect_failure]
    unknown = [name for name in names if name not in CASE_PRESETS]
    if unknown:
        known = ", ".join(sorted(CASE_PRESETS))
        raise ValueError(f"unknown case(s): {unknown}; known cases: {known}, all")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1H-8w-b in-memory cached KV cases.")
    parser.add_argument("--draft-model", default=os.environ.get("PEARL_DRAFT_MODEL"))
    parser.add_argument("--target-model", default=os.environ.get("PEARL_TARGET_MODEL"))
    parser.add_argument(
        "--workload-in",
        default=os.environ.get(
            "PEARL_WORKLOAD_IN",
            "benchmark/workloads/sanity_rps4_n32_mix622_seed0.jsonl",
        ),
    )
    parser.add_argument("--out-root", default="results/multislo/phase1h8w_b_cached_kv")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--eager-policy", default="tight_only")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--run-checkers", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--extra-eval-arg", action="append", default=[])
    return parser.parse_args()


def validate_required_args(args: argparse.Namespace) -> None:
    missing = []
    for name in ("draft_model", "target_model", "workload_in"):
        if not getattr(args, name):
            missing.append(name.replace("_", "-"))
    if missing and not args.print_only:
        raise SystemExit(
            "missing required argument(s): "
            + ", ".join(f"--{name}" for name in missing)
            + ". You can also set PEARL_DRAFT_MODEL, PEARL_TARGET_MODEL, and PEARL_WORKLOAD_IN."
        )


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for name, case in CASE_PRESETS.items():
            print(f"{name}: {case.description}")
        return 0
    validate_required_args(args)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    rows: list[dict[str, Any]] = []
    exit_code = 0
    for case_name in select_cases(args.cases):
        status, row = run_case(args, CASE_PRESETS[case_name], env)
        rows.append(row)
        if status != 0:
            exit_code = status
            if args.strict or not args.keep_going:
                break
    summary_path = Path(args.out_root) / "phase1h8w_b_cached_kv_summary.json"
    if not args.print_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote summary: {summary_path}")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
