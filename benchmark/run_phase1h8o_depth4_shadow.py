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

from benchmark.check_bounded_rolling_readiness_audit import validate_records as validate_audit_records  # noqa: E402
from benchmark.check_eager_commit_ready_only import load_trace  # noqa: E402
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402
from benchmark.check_generic_bounded_rolling_chain import summarize_generic_chain  # noqa: E402
from benchmark.write_bounded_chain_summary import build_chain_summary, write_chain_summary  # noqa: E402


@dataclass(frozen=True)
class Depth4ShadowCase:
    name: str
    limit_requests: int
    eval_batch_size: int
    gamma: int
    max_eager_requests_per_step: int
    max_eager_tokens_per_step: int
    max_eager_tokens_per_request: int
    warmup_iters: int
    depth4_shadow_enabled: bool
    max_continuous_eager_requests_per_step: int
    max_continuous_eager_tokens_per_step: int
    max_continuous_eager_tokens_per_request: int
    max_rolling_continuous_depth: int
    max_rolling_continuous_draft_children_per_step: int
    max_rolling_continuous_seqs_per_step: int
    eager_trace_level: str
    description: str


CASE_PRESETS: dict[str, Depth4ShadowCase] = {
    "baseline_8n_depth3_commit": Depth4ShadowCase(
        "baseline_8n_depth3_commit", 32, 32, 4, 2, 8, 4, 1, False, 2, 8, 4, 3, 2, 2, "minimal",
        "Depth-3 commit baseline with depth-4 shadow disabled.",
    ),
    "rolling_depth4_shadow": Depth4ShadowCase(
        "rolling_depth4_shadow", 32, 32, 4, 2, 8, 4, 1, True, 2, 8, 4, 4, 2, 2, "minimal",
        "Depth-4 shadow dry-run after guarded depth-3 real commit.",
    ),
    "rolling_depth4_pressure_shadow": Depth4ShadowCase(
        "rolling_depth4_pressure_shadow", 32, 32, 4, 4, 16, 4, 1, True, 4, 16, 4, 4, 4, 4, "summary",
        "Moderate eager pressure with depth-4 shadow dry-run.",
    ),
    "rolling_gamma2_depth4_shadow": Depth4ShadowCase(
        "rolling_gamma2_depth4_shadow", 32, 32, 2, 2, 4, 2, 1, True, 2, 4, 2, 4, 2, 2, "minimal",
        "Gamma-2 depth-4 shadow dry-run.",
    ),
}


CHECKERS = [
    ("benchmark/check_eager_commit_ready_only.py", "trace", ()),
    ("benchmark/check_continuous_eager_dry_run.py", "trace", ()),
    ("benchmark/check_continuous_eager_commit_depth1_ready_only.py", "trace", ()),
    ("benchmark/check_rolling_continuous_eager_dry_run.py", "trace", ()),
    ("benchmark/check_rolling_continuous_depth2_commit_ready_only.py", "trace", ()),
    ("benchmark/check_rolling_continuous_depth3_shadow_dry_run.py", "trace", ()),
    ("benchmark/check_rolling_continuous_depth3_commit_ready_only.py", "trace", ()),
    ("benchmark/check_rolling_continuous_depth4_shadow_dry_run.py", "trace_result", ()),
    ("benchmark/check_bounded_rolling_readiness_audit.py", "trace_result", ("--check-generic-parity",)),
    ("benchmark/check_generic_bounded_rolling_chain.py", "trace_result", ()),
    ("benchmark/check_eager_performance_accounting.py", "trace_result", ("--check-generic-chain",)),
    ("benchmark/check_multislo_result.py", "result", ()),
]


def format_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: list[str], env: dict[str, str], print_only: bool) -> int:
    print(f"\n$ {format_cmd(command)}", flush=True)
    if print_only:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def eval_command(args: argparse.Namespace, case: Depth4ShadowCase, result_json: Path, engine_trace: Path) -> list[str]:
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
        "--enable-rolling-continuous-eager-dry-run",
        "--enable-rolling-continuous-depth2-commit-ready-only",
        "--enable-rolling-continuous-depth3-shadow-dry-run",
        "--enable-rolling-continuous-depth3-commit-ready-only",
        "--eager-policy", args.eager_policy,
        "--max-eager-requests-per-step", str(case.max_eager_requests_per_step),
        "--max-eager-tokens-per-step", str(case.max_eager_tokens_per_step),
        "--max-eager-tokens-per-request", str(case.max_eager_tokens_per_request),
        "--max-continuous-eager-chain-depth", "1",
        "--max-continuous-eager-requests-per-step", str(case.max_continuous_eager_requests_per_step),
        "--max-continuous-eager-tokens-per-step", str(case.max_continuous_eager_tokens_per_step),
        "--max-continuous-eager-tokens-per-request", str(case.max_continuous_eager_tokens_per_request),
        "--max-rolling-continuous-depth", str(case.max_rolling_continuous_depth),
        "--max-rolling-continuous-draft-children-per-step",
        str(case.max_rolling_continuous_draft_children_per_step),
        "--max-rolling-continuous-seqs-per-step", str(case.max_rolling_continuous_seqs_per_step),
        "--eager-trace-level", case.eager_trace_level,
        "--max-tokens", str(args.max_tokens),
        "--warmup-iters", str(case.warmup_iters),
        "--warmup-max-tokens", str(args.warmup_max_tokens),
        "--limit-requests", str(case.limit_requests),
        "--eval-batch-size", str(case.eval_batch_size),
    ]
    if case.depth4_shadow_enabled:
        command.append("--enable-rolling-continuous-depth4-shadow-dry-run")
    if args.decode_ready:
        command.append("--decode-ready")
    if args.ignore_eos:
        command.append("--ignore-eos")
    for extra_arg in args.extra_eval_arg:
        command.append(extra_arg)
    return command


def checker_command(args: argparse.Namespace, checker: str, *paths: Path, extra_args: tuple[str, ...] = ()) -> list[str]:
    return [args.python, checker, *(str(path) for path in paths), *extra_args]


def load_case_summary(engine_trace: Path, result_json: Path, *, case_name: str, chain_summary_path: Path) -> dict[str, Any]:
    records = load_trace(engine_trace) if engine_trace.exists() else []
    result_payload = load_json(result_json) if result_json.exists() else {}
    accounting = aggregate_performance_accounting(records, result_payload)
    audit_errors, audit = validate_audit_records(records, result_payload)
    generic = summarize_generic_chain(records, result_payload)
    chain_summary = build_chain_summary(records, result_payload, case_name=case_name)
    write_chain_summary(chain_summary, chain_summary_path)
    depth_token_parity_ok = all(
        generic.get(generic_key) == audit.get(legacy_key)
        for generic_key, legacy_key in (
            ("one_shot_committed_token_count", "one_shot_committed_token_count"),
            ("depth1_committed_token_count", "depth1_committed_token_count"),
            ("depth2_committed_token_count", "depth2_committed_token_count"),
            ("depth3_committed_token_count", "depth3_committed_token_count"),
        )
    )
    combined_parity_ok = (
        generic.get("combined_real_committed_token_count") == audit.get("combined_real_committed_token_count")
        == accounting.get("combined_real_committed_token_count")
    )
    safety_parity_ok = all(
        generic.get(field) == audit.get(field)
        for field in (
            "max_real_committed_depth",
            "max_observed_depth",
            "depth_gt3_real_commit_count",
            "depth_gt4_real_commit_count",
            "depth4_real_commit_count",
            "normal_lane_conflict_count",
            "missing_buffered_proposal_unexpected_count",
            "duplicate_commit_count",
            "invalid_committed_child_count",
            "cascade_committed_child_count",
            "parent_missing_committed_child_count",
            "combined_accounting_ok",
            "target_draft_accounting_ok",
        )
    )
    return {
        "goodput_tokens_per_s": accounting.get("goodput_tokens_per_s"),
        "mean_tpot_ms": accounting.get("mean_tpot_ms"),
        "total_output_tokens": accounting.get("total_output_tokens"),
        "engine_elapsed_s": accounting.get("engine_elapsed_s"),
        "one_shot_committed_tokens": accounting.get("eager_committed_token_count"),
        "continuous_depth1_real_committed_tokens": accounting.get("continuous_eager_real_committed_token_count"),
        "rolling_depth2_real_committed_tokens": accounting.get("rolling_depth2_real_committed_token_count"),
        "rolling_depth3_real_committed_tokens": accounting.get("rolling_depth3_real_committed_token_count"),
        "rolling_depth4_child_candidate_tokens": accounting.get("rolling_depth4_child_candidate_token_count"),
        "rolling_depth4_child_ready_shadow_tokens": accounting.get("rolling_depth4_child_ready_shadow_token_count"),
        "rolling_depth4_child_invalidated_count": accounting.get("rolling_depth4_child_invalidated_count"),
        "rolling_depth4_real_committed_tokens": accounting.get("rolling_depth4_real_committed_token_count"),
        "combined_real_committed_tokens": accounting.get("combined_real_committed_token_count"),
        "combined_real_committed_token_share": accounting.get("combined_real_committed_token_share_of_output"),
        "max_real_committed_depth": audit.get("max_real_committed_depth"),
        "max_observed_depth": audit.get("max_observed_depth"),
        "depth4_real_commit_count": audit.get("depth4_real_commit_count"),
        "depth_gt3_real_commit_count": audit.get("depth_gt3_real_commit_count"),
        "depth_gt4_real_commit_count": audit.get("depth_gt4_real_commit_count"),
        "normal_lane_conflict_count": audit.get("normal_lane_conflict_count"),
        "combined_accounting_ok": audit.get("combined_accounting_ok"),
        "target_draft_accounting_ok": audit.get("target_draft_accounting_ok"),
        "audit_error_count": len(audit_errors),
        "generic_error_count": generic.get("generic_error_count"),
        "generic_legacy_combined_parity_ok": combined_parity_ok,
        "generic_legacy_depth_token_parity_ok": depth_token_parity_ok,
        "generic_legacy_safety_parity_ok": safety_parity_ok,
        "legacy_generic_parity_ok": chain_summary.get("legacy_generic_parity_ok"),
        "generic_chain_accounting_ok": chain_summary.get("generic_chain_accounting_ok"),
        "chain_summary_written": chain_summary_path.exists(),
        "chain_summary_path": str(chain_summary_path),
        "rolling_depth4_drop_reason_counts": accounting.get("rolling_depth4_drop_reason_counts"),
        "rolling_depth4_parent_resolution_pending_count": accounting.get("rolling_depth4_parent_resolution_pending_count"),
        "rolling_depth4_shadow_generation_time_ms": accounting.get("rolling_depth4_shadow_generation_time_ms"),
        "performance_warnings": accounting.get("performance_warnings"),
    }


def run_case(args: argparse.Namespace, case: Depth4ShadowCase, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    chain_summary_path = case_dir / "chain_summary.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)
    status = run_command(eval_command(args, case, result_json, engine_trace), env, args.print_only)
    check_status = "eval_failed" if status else "pass"
    if status == 0:
        for checker, input_kind, extra_args in CHECKERS:
            if input_kind == "trace":
                command = checker_command(args, checker, engine_trace, extra_args=extra_args)
            elif input_kind == "result":
                command = checker_command(args, checker, result_json, extra_args=extra_args)
            else:
                command = checker_command(args, checker, engine_trace, result_json, extra_args=extra_args)
            checker_status = run_command(command, env, args.print_only)
            if checker_status != 0:
                status = checker_status
                check_status = f"failed:{checker}"
                break
    row: dict[str, Any] = {
        "case_name": case.name,
        "depth2_commit_enabled": True,
        "depth3_shadow_enabled": True,
        "depth3_commit_enabled": True,
        "depth4_shadow_enabled": case.depth4_shadow_enabled,
        "check_status": check_status,
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "description": case.description,
    }
    if status == 0 and not args.print_only:
        row.update(
            load_case_summary(
                engine_trace,
                result_json,
                case_name=case.name,
                chain_summary_path=chain_summary_path,
            )
        )
        parity_ok = all(
            bool(row.get(field))
            for field in (
                "generic_legacy_combined_parity_ok",
                "generic_legacy_depth_token_parity_ok",
                "generic_legacy_safety_parity_ok",
                "legacy_generic_parity_ok",
                "generic_chain_accounting_ok",
                "chain_summary_written",
            )
        )
        if not parity_ok:
            status = 1
            row["check_status"] = "failed:generic_legacy_parity"
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
    parser = argparse.ArgumentParser(description="Run Phase 1H-8o rolling depth4 shadow dry-run cases.")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h8o_depth4_shadow")
    parser.add_argument("--cases", default="baseline_8n_depth3_commit,rolling_depth4_shadow")
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
    parser.add_argument("--keep-going", action="store_true")
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
    summary_path = Path(args.out_root) / "phase1h8o_summary.json"
    if not args.print_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote summary: {summary_path}")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
