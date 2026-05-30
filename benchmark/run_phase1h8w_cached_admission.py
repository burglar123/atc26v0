#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_cached_admission import result_summary as cached_result_summary  # noqa: E402
from benchmark.check_eager_performance_accounting import load_json  # noqa: E402
from benchmark.run_phase1h8q_partial_prefix_recovery import run_checker_chain, run_command  # noqa: E402
from benchmark.run_phase1h8u_full_continuous_max_depth import (  # noqa: E402
    CASE_PRESETS as FULL_CONTINUOUS_CASE_PRESETS,
    FullContinuousMaxDepthCase,
    eval_command_8u,
    load_case_summary as load_8u_case_summary,
)


CHECKERS = [
    ("benchmark/check_cached_admission.py", "trace_result", ()),
    ("benchmark/check_eager_partial_prefix_recovery.py", "trace_result", ()),
    ("benchmark/check_bounded_rolling_readiness_audit.py", "trace_result", ("--check-generic-parity",)),
    ("benchmark/check_generic_bounded_rolling_chain.py", "trace_result", ()),
    ("benchmark/check_generic_rolling_runtime_parity.py", "trace_result", ()),
    ("benchmark/check_generic_rolling_apply_path_parity.py", "trace_result", ()),
    ("benchmark/check_full_continuous_max_depth.py", "trace_result", ()),
    ("benchmark/check_eager_performance_accounting.py", "trace_result", ("--check-generic-chain",)),
    ("benchmark/check_multislo_result.py", "result", ()),
]


@dataclass(frozen=True)
class CachedAdmissionCase:
    name: str
    base_case: FullContinuousMaxDepthCase
    cached_admission_enabled: bool
    partial_recovery_expected: bool
    require_depth_gt4_activity: bool
    description: str


CASE_PRESETS: dict[str, CachedAdmissionCase] = {
    "baseline_8v_no_cached_admission": CachedAdmissionCase(
        "baseline_8v_no_cached_admission",
        FULL_CONTINUOUS_CASE_PRESETS["full_continuous_depth100_baseline"],
        False,
        False,
        False,
        "8v full-continuous depth100 baseline with cached admission disabled.",
    ),
    "cached_admission_decode_only_fifo": CachedAdmissionCase(
        "cached_admission_decode_only_fifo",
        FULL_CONTINUOUS_CASE_PRESETS["baseline_8t_generic_apply_depth4"],
        True,
        False,
        False,
        "Decode-only FIFO cached admission with conservative depth-4 apply path.",
    ),
    "cached_admission_full_continuous_depth100": CachedAdmissionCase(
        "cached_admission_full_continuous_depth100",
        FULL_CONTINUOUS_CASE_PRESETS["full_continuous_depth100_baseline"],
        True,
        False,
        True,
        "FIFO cached admission with full-continuous max_depth=100.",
    ),
    "cached_admission_full_continuous_partial_recovery": CachedAdmissionCase(
        "cached_admission_full_continuous_partial_recovery",
        FULL_CONTINUOUS_CASE_PRESETS["full_continuous_depth100_partial_recovery"],
        True,
        True,
        False,
        "FIFO cached admission with full-continuous depth100 and partial-prefix recovery enabled.",
    ),
}


def eval_command_8w(
    args: argparse.Namespace,
    case: CachedAdmissionCase,
    result_json: Path,
    engine_trace: Path,
) -> list[str]:
    command = eval_command_8u(args, case.base_case, result_json, engine_trace)
    if case.cached_admission_enabled:
        command.extend(
            [
                "--enable-cached-admission",
                "--decode-ready",
                "--cache-build-batch-size",
                str(args.cache_build_batch_size),
                "--cached-admission-policy",
                args.cached_admission_policy,
                "--cached-admission-mode",
                args.cached_admission_mode,
                "--cached-admission-max-active",
                str(args.cached_admission_max_active),
                "--cached-admission-arrival-field",
                args.cached_admission_arrival_field,
            ]
        )
    return command


def select_cases(raw_cases: str) -> list[str]:
    names = [item.strip() for item in raw_cases.split(",") if item.strip()]
    if names == ["all"]:
        return list(CASE_PRESETS)
    unknown = [name for name in names if name not in CASE_PRESETS]
    if unknown:
        known = ", ".join(sorted(CASE_PRESETS))
        raise ValueError(f"unknown case(s): {unknown}; known cases: {known}, all")
    return names


def load_case_summary(
    engine_trace: Path,
    result_json: Path,
    *,
    case_name: str,
    chain_summary_path: Path,
) -> dict[str, Any]:
    row = load_8u_case_summary(
        engine_trace,
        result_json,
        case_name=case_name,
        chain_summary_path=chain_summary_path,
    )
    result_payload = load_json(result_json) if result_json.exists() else {}
    cached = cached_result_summary(result_payload)
    row.update(cached)
    metrics = result_payload.get("metrics", {}) if isinstance(result_payload, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    if isinstance(overall, dict):
        row["goodput_tokens_per_s"] = overall.get("goodput_tokens_per_s")
        row["mean_tpot_ms"] = overall.get("mean_tpot_ms")
    return row


def run_case(args: argparse.Namespace, case: CachedAdmissionCase, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    chain_summary_path = case_dir / "chain_summary.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)

    status = run_command(eval_command_8w(args, case, result_json, engine_trace), env, args.print_only)
    check_status = "eval_failed" if status else "pass"
    checker_chain_status = "not_run:eval_failed" if status else "pass"
    checker_chain_failed_checker: str | None = None
    chain_passed = False

    if status == 0:
        checker_status, checker_chain_status, checker_chain_failed_checker = run_checker_chain(
            args,
            CHECKERS,
            engine_trace=engine_trace,
            result_json=result_json,
            env=env,
            label="core",
        )
        if checker_status != 0:
            status = checker_status
            check_status = checker_chain_status
        else:
            chain_passed = True

        if chain_passed and case.require_depth_gt4_activity:
            depth_status, depth_chain_status, depth_failed = run_checker_chain(
                args,
                [("benchmark/check_full_continuous_max_depth.py", "trace_result", ("--require-depth-gt4-activity",))],
                engine_trace=engine_trace,
                result_json=result_json,
                env=env,
                label="depth_gt4",
            )
            if depth_status != 0:
                status = depth_status
                check_status = depth_chain_status
                checker_chain_status = depth_chain_status
                checker_chain_failed_checker = depth_failed

    row: dict[str, Any] = {
        "case_name": case.name,
        "cached_admission_enabled": case.cached_admission_enabled,
        "cached_admission_policy": args.cached_admission_policy,
        "cached_admission_mode": args.cached_admission_mode if case.cached_admission_enabled else None,
        "cached_admission_max_active": args.cached_admission_max_active if case.cached_admission_enabled else 0,
        "partial_prefix_recovery_expected": case.partial_recovery_expected,
        "check_status": check_status,
        "core_checker_chain_status": checker_chain_status,
        "core_checker_chain_failed_checker": checker_chain_failed_checker,
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "description": case.description,
    }
    if chain_passed and not args.print_only:
        row.update(
            load_case_summary(
                engine_trace,
                result_json,
                case_name=case.name,
                chain_summary_path=chain_summary_path,
            )
        )
        if case.cached_admission_enabled and not bool(row.get("cached_admission_enabled", False)):
            status = 1
            row["check_status"] = "failed:cached_admission_summary"
            row["core_checker_chain_status"] = "failed:cached_admission_summary"
            row["core_checker_chain_failed_checker"] = "cached_admission_summary"
    return status, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1H-8w cached admission cases.")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h8w_cached_admission")
    parser.add_argument(
        "--cases",
        default=(
            "baseline_8v_no_cached_admission,cached_admission_decode_only_fifo,"
            "cached_admission_full_continuous_depth100,cached_admission_full_continuous_partial_recovery"
        ),
    )
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--decode-ready", action="store_true", default=True)
    parser.add_argument("--no-decode-ready", dest="decode_ready", action="store_false")
    parser.add_argument("--eager-policy", default="tight_only")
    parser.add_argument("--cache-build-batch-size", type=int, default=16)
    parser.add_argument("--cached-admission-max-active", type=int, default=16)
    parser.add_argument("--cached-admission-policy", choices=["fifo"], default="fifo")
    parser.add_argument("--cached-admission-mode", choices=["in_memory_kv"], default="in_memory_kv")
    parser.add_argument(
        "--cached-admission-arrival-field",
        choices=["arrival_offset_sec", "arrival_ts"],
        default="arrival_offset_sec",
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--run-legacy-depth-checkers", action="store_true")
    parser.add_argument("--strict-legacy-depth-checkers", action="store_true")
    parser.add_argument("--extra-eval-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for name, case in CASE_PRESETS.items():
            print(f"{name}: {case.description}")
        return 0
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
            if not args.keep_going:
                break
    summary_path = Path(args.out_root) / "phase1h8w_summary.json"
    if not args.print_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote summary: {summary_path}")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
