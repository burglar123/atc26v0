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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import load_trace  # noqa: E402
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402


@dataclass(frozen=True)
class RollingCase:
    name: str
    limit_requests: int
    eval_batch_size: int
    gamma: int
    max_eager_requests_per_step: int
    max_eager_tokens_per_step: int
    max_eager_tokens_per_request: int
    warmup_iters: int
    rolling_enabled: bool
    max_continuous_eager_requests_per_step: int
    max_continuous_eager_tokens_per_step: int
    max_continuous_eager_tokens_per_request: int
    max_rolling_continuous_depth: int
    max_rolling_continuous_draft_children_per_step: int
    max_rolling_continuous_seqs_per_step: int
    eager_trace_level: str
    description: str


CASE_PRESETS: dict[str, RollingCase] = {
    "baseline_7d_depth1": RollingCase(
        "baseline_7d_depth1", 32, 32, 4, 2, 8, 4, 1, False, 2, 8, 4, 2, 2, 2, "minimal",
        "Existing 7d one-shot plus guarded continuous depth-1 commit.",
    ),
    "rolling_depth2_shadow": RollingCase(
        "rolling_depth2_shadow", 32, 32, 4, 2, 8, 4, 1, True, 2, 8, 4, 2, 2, 2, "minimal",
        "Rolling overlap shadow child generation after depth-1 continuous parents.",
    ),
    "rolling_depth2_pressure_shadow": RollingCase(
        "rolling_depth2_pressure_shadow", 32, 32, 4, 4, 16, 4, 1, True, 4, 16, 4, 2, 4, 4, "summary",
        "Moderate eager pressure with rolling depth-2 shadow generation.",
    ),
    "rolling_gamma2_depth2_shadow": RollingCase(
        "rolling_gamma2_depth2_shadow", 32, 32, 2, 2, 4, 2, 1, True, 2, 4, 2, 2, 2, 2, "minimal",
        "Gamma-2 rolling depth-2 shadow generation.",
    ),
}


BASE_CHECKERS = [
    ("benchmark/check_eager_commit_ready_only.py", "trace"),
    ("benchmark/check_multislo_result.py", "result"),
    ("benchmark/check_eager_performance_accounting.py", "trace_result"),
    ("benchmark/check_continuous_eager_dry_run.py", "trace"),
    ("benchmark/check_continuous_eager_commit_depth1_ready_only.py", "trace"),
]


def format_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str], env: dict[str, str], print_only: bool) -> int:
    print(f"\n$ {format_cmd(command)}", flush=True)
    if print_only:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def eval_command(args: argparse.Namespace, case: RollingCase, result_json: Path, engine_trace: Path) -> list[str]:
    command = [
        args.python,
        "benchmark/eval_multi_slo.py",
        "--draft-model", args.draft_model,
        "--target-model", args.target_model,
        "--draft-tp", str(args.draft_tp),
        "--target-tp", str(args.target_tp),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--execution-mode", "dual_batch_pearl",
        "--gamma", str(case.gamma),
        "--workload-in", args.workload_in,
        "--out", str(result_json),
        "--engine-trace-out", str(engine_trace),
        "--enable-eager-commit-ready-only",
        "--enable-continuous-eager-dry-run",
        "--enable-continuous-eager-verify-apply-dry-run",
        "--enable-continuous-eager-commit-depth1-ready-only",
        "--eager-policy", args.eager_policy,
        "--max-eager-requests-per-step", str(case.max_eager_requests_per_step),
        "--max-eager-tokens-per-step", str(case.max_eager_tokens_per_step),
        "--max-eager-tokens-per-request", str(case.max_eager_tokens_per_request),
        "--max-continuous-eager-chain-depth", "1",
        "--max-continuous-eager-requests-per-step", str(case.max_continuous_eager_requests_per_step),
        "--max-continuous-eager-tokens-per-step", str(case.max_continuous_eager_tokens_per_step),
        "--max-continuous-eager-tokens-per-request", str(case.max_continuous_eager_tokens_per_request),
        "--eager-trace-level", case.eager_trace_level,
        "--max-tokens", str(args.max_tokens),
        "--warmup-iters", str(case.warmup_iters),
        "--warmup-max-tokens", str(args.warmup_max_tokens),
        "--limit-requests", str(case.limit_requests),
        "--eval-batch-size", str(case.eval_batch_size),
    ]
    if case.rolling_enabled:
        command.extend(
            [
                "--enable-rolling-continuous-eager-dry-run",
                "--max-rolling-continuous-depth", str(case.max_rolling_continuous_depth),
                "--max-rolling-continuous-draft-children-per-step",
                str(case.max_rolling_continuous_draft_children_per_step),
                "--max-rolling-continuous-seqs-per-step", str(case.max_rolling_continuous_seqs_per_step),
            ]
        )
    if args.decode_ready:
        command.append("--decode-ready")
    if args.ignore_eos:
        command.append("--ignore-eos")
    for extra_arg in args.extra_eval_arg:
        command.append(extra_arg)
    return command


def checker_command(args: argparse.Namespace, checker: str, *paths: Path) -> list[str]:
    return [args.python, checker, *(str(path) for path in paths)]


def load_case_summary(engine_trace: Path, result_json: Path) -> dict[str, Any]:
    records = load_trace(engine_trace) if engine_trace.exists() else []
    result_payload = load_json(result_json) if result_json.exists() else {}
    accounting = aggregate_performance_accounting(records, result_payload)
    return {
        "goodput_tokens_per_s": accounting.get("goodput_tokens_per_s"),
        "mean_tpot_ms": accounting.get("mean_tpot_ms"),
        "one_shot_committed_tokens": accounting.get("eager_committed_token_count"),
        "continuous_depth1_real_committed_tokens": accounting.get("continuous_eager_real_committed_token_count"),
        "combined_real_committed_tokens": accounting.get("combined_real_committed_token_count"),
        "rolling_child_candidate_tokens": accounting.get("rolling_child_candidate_token_count"),
        "rolling_child_ready_shadow_tokens": accounting.get("rolling_child_ready_shadow_token_count"),
        "rolling_child_invalidated_count": accounting.get("rolling_child_invalidated_count"),
        "rolling_cascade_discard_count": accounting.get("rolling_cascade_discard_count"),
        "rolling_same_seq_overlap_count": accounting.get("rolling_same_seq_overlap_count"),
        "rolling_normal_lane_conflict_count": accounting.get("rolling_normal_lane_conflict_count"),
        "rolling_depth2_real_commit_count": accounting.get("rolling_depth2_real_commit_count"),
        "rolling_depth_gt1_real_commit_count": accounting.get("rolling_depth_gt1_real_commit_count"),
        "rolling_max_depth_observed": accounting.get("rolling_max_depth_observed"),
        "rolling_drop_reason_counts": accounting.get("rolling_drop_reason_counts"),
    }


def run_case(args: argparse.Namespace, case: RollingCase, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)
    status = run_command(eval_command(args, case, result_json, engine_trace), env, args.print_only)
    check_status = "eval_failed" if status else "pass"
    if status == 0:
        for checker, input_kind in BASE_CHECKERS:
            if input_kind == "trace":
                checker_status = run_command(checker_command(args, checker, engine_trace), env, args.print_only)
            elif input_kind == "result":
                checker_status = run_command(checker_command(args, checker, result_json), env, args.print_only)
            else:
                checker_status = run_command(
                    checker_command(args, checker, engine_trace, result_json),
                    env,
                    args.print_only,
                )
            if checker_status != 0:
                status = checker_status
                check_status = f"failed:{checker}"
                break
    if status == 0 and case.rolling_enabled:
        checker_status = run_command(
            checker_command(args, "benchmark/check_rolling_continuous_eager_dry_run.py", engine_trace),
            env,
            args.print_only,
        )
        if checker_status != 0:
            status = checker_status
            check_status = "failed:benchmark/check_rolling_continuous_eager_dry_run.py"
    row: dict[str, Any] = {
        "case_name": case.name,
        "rolling_enabled": case.rolling_enabled,
        "check_status": check_status,
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "description": case.description,
    }
    if status == 0 and not args.print_only:
        row.update(load_case_summary(engine_trace, result_json))
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
    parser = argparse.ArgumentParser(description="Run Phase 1H-8a rolling continuous overlap dry-run cases.")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h8a_rolling_continuous_dry_run")
    parser.add_argument("--cases", default="baseline_7d_depth1,rolling_depth2_shadow")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--decode-ready", action="store_true", default=True)
    parser.add_argument("--no-decode-ready", dest="decode_ready", action="store_false")
    parser.add_argument("--eager-policy", default="tight_only")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--print-only", action="store_true")
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
    selected = select_cases(args.cases)
    rows = []
    final_status = 0
    for name in selected:
        status, row = run_case(args, CASE_PRESETS[name], env)
        rows.append(row)
        if status != 0:
            final_status = status
            break
    out_root = Path(args.out_root)
    if not args.print_only:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "phase1h8a_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print("\nPhase 1H-8a rolling continuous overlap summary:")
    for row in rows:
        print(
            f"{row['case_name']}: {row['check_status']} "
            f"combined={row.get('combined_real_committed_tokens', 'n/a')} "
            f"rolling_candidate={row.get('rolling_child_candidate_tokens', 'n/a')} "
            f"rolling_ready={row.get('rolling_child_ready_shadow_tokens', 'n/a')} "
            f"overlap={row.get('rolling_same_seq_overlap_count', 'n/a')} "
            f"depth2_commit={row.get('rolling_depth2_real_commit_count', 'n/a')}"
        )
    return final_status


if __name__ == "__main__":
    raise SystemExit(main())
