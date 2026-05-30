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

from benchmark.check_eager_performance_accounting import aggregate_performance_accounting, load_json  # noqa: E402
from benchmark.run_phase1h8p_depth4_commit import (  # noqa: E402
    CASE_PRESETS as DEPTH4_CASE_PRESETS,
    CHECKERS as DEPTH4_CHECKERS,
    Depth4CommitCase,
    checker_command,
    eval_command,
    format_cmd,
    load_case_summary as load_depth4_case_summary,
    load_trace,
    run_command,
)


@dataclass(frozen=True)
class PartialRecoveryCase:
    name: str
    base_case: Depth4CommitCase
    partial_recovery_enabled: bool
    description: str


CASE_PRESETS: dict[str, PartialRecoveryCase] = {
    "baseline_8p_depth4_commit": PartialRecoveryCase(
        "baseline_8p_depth4_commit",
        DEPTH4_CASE_PRESETS["rolling_depth4_commit"],
        False,
        "Depth-4 real commit baseline with partial-prefix recovery disabled.",
    ),
    "partial_recovery_enabled": PartialRecoveryCase(
        "partial_recovery_enabled",
        DEPTH4_CASE_PRESETS["rolling_depth4_commit"],
        True,
        "Depth-4 real commit with partial-prefix recovery enabled when revised-token evidence exists.",
    ),
    "partial_recovery_trace_validation": PartialRecoveryCase(
        "partial_recovery_trace_validation",
        DEPTH4_CASE_PRESETS["rolling_depth4_commit"],
        True,
        "Partial-prefix recovery enabled; this workload may report zero natural recovery events.",
    ),
}


LEGACY_DEPTH_CHECKERS = DEPTH4_CHECKERS[:9]

CORE_CHECKERS = [
    ("benchmark/check_eager_partial_prefix_recovery.py", "trace_result", ()),
    ("benchmark/check_bounded_rolling_readiness_audit.py", "trace_result", ("--check-generic-parity",)),
    ("benchmark/check_generic_bounded_rolling_chain.py", "trace_result", ()),
    ("benchmark/check_eager_performance_accounting.py", "trace_result", ("--check-generic-chain",)),
    ("benchmark/check_multislo_result.py", "result", ()),
]

CHECKERS = CORE_CHECKERS


def eval_command_8q(args: argparse.Namespace, case: PartialRecoveryCase, result_json: Path, engine_trace: Path) -> list[str]:
    command = eval_command(args, case.base_case, result_json, engine_trace)
    if case.partial_recovery_enabled:
        command.append("--enable-rolling-continuous-partial-prefix-recovery")
    return command


def load_case_summary(
    engine_trace: Path,
    result_json: Path,
    *,
    case_name: str,
    chain_summary_path: Path,
) -> dict[str, Any]:
    row = load_depth4_case_summary(
        engine_trace,
        result_json,
        case_name=case_name,
        chain_summary_path=chain_summary_path,
    )
    records = load_trace(engine_trace) if engine_trace.exists() else []
    result_payload = load_json(result_json) if result_json.exists() else {}
    accounting = aggregate_performance_accounting(records, result_payload)
    row.update(
        {
            "partial_prefix_recovery_enabled": accounting.get("partial_prefix_recovery_enabled"),
            "partial_prefix_recovery_attempt_count": accounting.get("partial_prefix_recovery_attempt_count"),
            "partial_prefix_recovery_success_count": accounting.get("partial_prefix_recovery_success_count"),
            "partial_prefix_recovery_skip_reason_counts": accounting.get(
                "partial_prefix_recovery_skip_reason_counts"
            ),
            "partial_prefix_recovered_proposal_count": accounting.get(
                "partial_prefix_recovered_proposal_count"
            ),
            "partial_prefix_recovered_seq_count": accounting.get("partial_prefix_recovered_seq_count"),
            "partial_prefix_accepted_token_count": accounting.get("partial_prefix_accepted_token_count"),
            "partial_prefix_revised_token_count": accounting.get("partial_prefix_revised_token_count"),
            "partial_prefix_total_recovered_token_count": accounting.get(
                "partial_prefix_total_recovered_token_count"
            ),
            "partial_recovery_cascade_discard_count": accounting.get("partial_recovery_cascade_discard_count"),
            "partial_recovery_target_draft_length_mismatch_count": accounting.get(
                "partial_recovery_target_draft_length_mismatch_count"
            ),
            "partial_recovery_target_draft_token_mismatch_count": accounting.get(
                "partial_recovery_target_draft_token_mismatch_count"
            ),
            "combined_actual_verified_token_increment_sum": accounting.get(
                "combined_actual_verified_token_increment_sum"
            ),
            "combined_actual_accepted_token_increment_sum": accounting.get(
                "combined_actual_accepted_token_increment_sum"
            ),
            "combined_actual_revised_token_increment_sum": accounting.get(
                "combined_actual_revised_token_increment_sum"
            ),
            "combined_actual_output_token_increment_sum": accounting.get(
                "combined_actual_output_token_increment_sum"
            ),
        }
    )
    return row


def run_checker_chain(
    args: argparse.Namespace,
    checkers: list[tuple[str, str, tuple[str, ...]]],
    *,
    engine_trace: Path,
    result_json: Path,
    env: dict[str, str],
    label: str,
) -> tuple[int, str, str | None]:
    for checker, input_kind, extra_args in checkers:
        if input_kind == "trace":
            command = checker_command(args, checker, engine_trace, extra_args=extra_args)
        elif input_kind == "result":
            command = checker_command(args, checker, result_json, extra_args=extra_args)
        else:
            command = checker_command(args, checker, engine_trace, result_json, extra_args=extra_args)
        print(f"{label}_checker={checker}", flush=True)
        checker_status = run_command(command, env, args.print_only)
        if checker_status != 0:
            return checker_status, f"failed:{checker}", checker
    return 0, "pass", None


def run_case(args: argparse.Namespace, case: PartialRecoveryCase, env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    chain_summary_path = case_dir / "chain_summary.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)
    status = run_command(eval_command_8q(args, case, result_json, engine_trace), env, args.print_only)
    check_status = "eval_failed" if status else "pass"
    core_checker_chain_status = "not_run:eval_failed" if status else "pass"
    core_checker_chain_failed_checker: str | None = None
    legacy_depth_checker_chain_ran = bool(args.run_legacy_depth_checkers or args.strict_legacy_depth_checkers)
    legacy_depth_checker_chain_status = "not_run"
    legacy_depth_checker_chain_failed_checker: str | None = None
    core_chain_passed = False
    if status == 0:
        core_status, core_checker_chain_status, core_checker_chain_failed_checker = run_checker_chain(
            args,
            CORE_CHECKERS,
            engine_trace=engine_trace,
            result_json=result_json,
            env=env,
            label="core",
        )
        if core_status != 0:
            status = core_status
            check_status = core_checker_chain_status
        else:
            core_chain_passed = True
        if core_chain_passed and legacy_depth_checker_chain_ran:
            legacy_status, legacy_depth_checker_chain_status, legacy_depth_checker_chain_failed_checker = (
                run_checker_chain(
                    args,
                    LEGACY_DEPTH_CHECKERS,
                    engine_trace=engine_trace,
                    result_json=result_json,
                    env=env,
                    label="legacy_depth",
                )
            )
            if legacy_status != 0 and args.strict_legacy_depth_checkers:
                status = legacy_status
                check_status = legacy_depth_checker_chain_status
    row: dict[str, Any] = {
        "case_name": case.name,
        "depth2_commit_enabled": True,
        "depth3_shadow_enabled": True,
        "depth3_commit_enabled": True,
        "depth4_shadow_enabled": case.base_case.depth4_shadow_enabled,
        "depth4_commit_enabled": case.base_case.depth4_commit_enabled,
        "partial_prefix_recovery_enabled": case.partial_recovery_enabled,
        "check_status": check_status,
        "core_checker_chain_status": core_checker_chain_status,
        "core_checker_chain_failed_checker": core_checker_chain_failed_checker,
        "legacy_depth_checker_chain_status": legacy_depth_checker_chain_status,
        "legacy_depth_checker_chain_failed_checker": legacy_depth_checker_chain_failed_checker,
        "legacy_depth_checker_chain_ran": legacy_depth_checker_chain_ran,
        "strict_legacy_depth_checkers": bool(args.strict_legacy_depth_checkers),
        "result_json": str(result_json),
        "engine_trace": str(engine_trace),
        "description": case.description,
    }
    if core_chain_passed and not args.print_only:
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
            row["core_checker_chain_status"] = "failed:generic_legacy_parity"
            row["core_checker_chain_failed_checker"] = "generic_legacy_parity"
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
    parser = argparse.ArgumentParser(description="Run Phase 1H-8q partial-prefix recovery cases.")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h8q_partial_recovery")
    parser.add_argument("--cases", default="baseline_8p_depth4_commit,partial_recovery_enabled")
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
    parser.add_argument(
        "--run-legacy-depth-checkers",
        action="store_true",
        help="Run old depth-specific checkers and report their status without failing cases unless strict mode is set.",
    )
    parser.add_argument(
        "--strict-legacy-depth-checkers",
        action="store_true",
        help="Run old depth-specific checkers and fail the case on legacy checker failure.",
    )
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
    summary_path = Path(args.out_root) / "phase1h8q_summary.json"
    if not args.print_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote summary: {summary_path}")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
