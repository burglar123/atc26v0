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
class CachedAdmissionCase:
    name: str
    enable_cached_admission: bool
    enable_full_continuous: bool
    limit_requests: int
    eval_batch_size: int
    max_tokens: int
    cached_admission_max_active: int
    description: str


CASE_PRESETS: dict[str, CachedAdmissionCase] = {
    "baseline_8v_no_cached_admission": CachedAdmissionCase(
        name="baseline_8v_no_cached_admission",
        enable_cached_admission=False,
        enable_full_continuous=False,
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=0,
        description="Decode-ready dual-batch baseline with cached admission disabled.",
    ),
    "cached_admission_metadata_only_fifo": CachedAdmissionCase(
        name="cached_admission_metadata_only_fifo",
        enable_cached_admission=True,
        enable_full_continuous=False,
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=2,
        description="Metadata-only cached admission using FIFO and a small active cap.",
    ),
    "cached_admission_full_continuous_smoke": CachedAdmissionCase(
        name="cached_admission_full_continuous_smoke",
        enable_cached_admission=True,
        enable_full_continuous=True,
        limit_requests=8,
        eval_batch_size=8,
        max_tokens=32,
        cached_admission_max_active=2,
        description="Cached admission through the dual-batch full-continuous smoke path.",
    ),
}


def format_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str], env: dict[str, str], print_only: bool) -> int:
    print(f"\n$ {format_cmd(command)}", flush=True)
    if print_only:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def add_eager_smoke_flags(command: list[str], args: argparse.Namespace) -> None:
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
    case: CachedAdmissionCase,
    result_json: Path,
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
        "dual_batch_pearl",
        "--gamma",
        "4",
        "--workload-in",
        args.workload_in,
        "--out",
        str(result_json),
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
    if case.enable_cached_admission:
        command.extend(
            [
                "--enable-cached-admission",
                "--cached-prefill-mode",
                "metadata_only",
                "--cached-admission-policy",
                "fifo",
                "--cached-admission-max-active",
                str(case.cached_admission_max_active),
                "--cached-admission-arrival-field",
                "arrival_offset_sec",
            ]
        )
    if case.enable_full_continuous:
        add_eager_smoke_flags(command, args)
        command.extend(
            [
                "--enable-generic-rolling-runtime-loop",
                "--enable-generic-rolling-apply-path",
                "--enable-full-continuous-eager",
            ]
        )
    for extra_arg in args.extra_eval_arg:
        command.append(extra_arg)
    return command


def checker_commands(args: argparse.Namespace, case: CachedAdmissionCase, result_json: Path, engine_trace: Path) -> list[list[str]]:
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
        "goodput_tokens_per_s": overall.get("goodput_tokens_per_s"),
        "cached_admission_enabled": cached.get("cached_admission_enabled", False),
        "cached_prefill_mode": cached.get("cached_prefill_mode"),
        "cached_admission_total_requests": cached.get("cached_admission_total_requests"),
        "cached_admission_total_arrived": cached.get("cached_admission_total_arrived"),
        "cached_admission_total_admitted": cached.get("cached_admission_total_admitted"),
        "cached_admission_total_completed": cached.get("cached_admission_total_completed"),
        "cached_admission_peak_active": cached.get("cached_admission_peak_active"),
        "cached_admission_mean_queue_wait_ms": cached.get("cached_admission_mean_queue_wait_ms"),
        "cached_admission_decode_only_elapsed_s": cached.get("cached_admission_decode_only_elapsed_s"),
    }


def run_case(args: argparse.Namespace, case: CachedAdmissionCase, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)

    status = run_command(eval_command(args, case, result_json, engine_trace), env, args.print_only)
    check_status = "eval_failed" if status else "pass"
    failed_checker: str | None = None
    if status == 0 and args.run_checkers:
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
        "cached_admission_enabled": case.enable_cached_admission,
        "cached_prefill_mode": "metadata_only" if case.enable_cached_admission else None,
        "cached_admission_policy": "fifo" if case.enable_cached_admission else None,
        "cached_admission_max_active": case.cached_admission_max_active if case.enable_cached_admission else None,
        "full_continuous_smoke": case.enable_full_continuous,
        "check_status": check_status,
        "failed_checker": failed_checker,
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "description": case.description,
    }
    if status == 0 and not args.print_only:
        row.update(load_case_summary(result_json))
    return status, row


def select_cases(raw_cases: str) -> list[str]:
    names = [item.strip() for item in raw_cases.split(",") if item.strip()]
    if names == ["all"]:
        return list(CASE_PRESETS)
    unknown = [name for name in names if name not in CASE_PRESETS]
    if unknown:
        known = ", ".join(sorted(CASE_PRESETS))
        raise ValueError(f"unknown case(s): {unknown}; known cases: {known}, all")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1H-8w cached-admission cases.")
    parser.add_argument("--draft-model", default=os.environ.get("PEARL_DRAFT_MODEL"))
    parser.add_argument("--target-model", default=os.environ.get("PEARL_TARGET_MODEL"))
    parser.add_argument(
        "--workload-in",
        default=os.environ.get(
            "PEARL_WORKLOAD_IN",
            "benchmark/workloads/sanity_rps4_n32_mix622_seed0.jsonl",
        ),
    )
    parser.add_argument("--out-root", default="results/multislo/phase1h8w_cached_admission")
    parser.add_argument(
        "--cases",
        default=(
            "baseline_8v_no_cached_admission,cached_admission_metadata_only_fifo,"
            "cached_admission_full_continuous_smoke"
        ),
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
    summary_path = Path(args.out_root) / "phase1h8w_cached_admission_summary.json"
    if not args.print_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote summary: {summary_path}")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
