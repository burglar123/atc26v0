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
from benchmark.check_full_continuous_max_depth import build_summary as build_full_continuous_summary  # noqa: E402
from benchmark.run_phase1h8q_partial_prefix_recovery import (  # noqa: E402
    LEGACY_DEPTH_CHECKERS,
    load_trace,
    run_checker_chain,
    run_command,
)
from benchmark.run_phase1h8t_generic_apply_path_parity import (  # noqa: E402
    CASE_PRESETS as APPLY_CASE_PRESETS,
    GenericApplyPathParityCase,
    eval_command_8t,
    load_case_summary as load_8t_case_summary,
)


CORE_CHECKERS = [
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
class FullContinuousMaxDepthCase:
    name: str
    apply_case: GenericApplyPathParityCase
    full_continuous_enabled: bool
    max_depth: int
    description: str
    require_depth_gt4_activity: bool = False


CASE_PRESETS: dict[str, FullContinuousMaxDepthCase] = {
    "baseline_8t_generic_apply_depth4": FullContinuousMaxDepthCase(
        "baseline_8t_generic_apply_depth4",
        APPLY_CASE_PRESETS["generic_apply_baseline"],
        False,
        4,
        "8t generic apply-path depth-4 regression case.",
    ),
    "full_continuous_depth100_baseline": FullContinuousMaxDepthCase(
        "full_continuous_depth100_baseline",
        APPLY_CASE_PRESETS["generic_apply_baseline"],
        True,
        100,
        "Generic apply path with bounded full continuous mode at max_depth=100.",
    ),
    "full_continuous_depth100_partial_recovery": FullContinuousMaxDepthCase(
        "full_continuous_depth100_partial_recovery",
        APPLY_CASE_PRESETS["generic_apply_partial_recovery"],
        True,
        100,
        "Generic apply path with bounded full continuous mode and partial recovery enabled.",
    ),
    "full_continuous_depth100_stress": FullContinuousMaxDepthCase(
        "full_continuous_depth100_stress",
        APPLY_CASE_PRESETS["generic_apply_partial_recovery"],
        True,
        100,
        "Depth100 full continuous stress case using the partial-recovery workload preset.",
    ),
    "full_continuous_depth100_must_exceed4": FullContinuousMaxDepthCase(
        "full_continuous_depth100_must_exceed4",
        APPLY_CASE_PRESETS["generic_apply_baseline"],
        True,
        100,
        "Depth100 full continuous case that fails unless real depth>4 activity is observed.",
        True,
    ),
}


def eval_command_8u(
    args: argparse.Namespace,
    case: FullContinuousMaxDepthCase,
    result_json: Path,
    engine_trace: Path,
) -> list[str]:
    command = eval_command_8t(args, case.apply_case, result_json, engine_trace)
    if case.full_continuous_enabled:
        command.extend(
            [
                "--enable-full-continuous-eager",
                "--enable-generic-rolling-runtime-loop",
                "--enable-generic-rolling-apply-path",
                "--max-rolling-continuous-depth",
                str(case.max_depth),
            ]
        )
    return command


def load_case_summary(
    engine_trace: Path,
    result_json: Path,
    *,
    case_name: str,
    chain_summary_path: Path,
) -> dict[str, Any]:
    row = load_8t_case_summary(
        engine_trace,
        result_json,
        case_name=case_name,
        chain_summary_path=chain_summary_path,
    )
    records = load_trace(engine_trace) if engine_trace.exists() else []
    result_payload = load_json(result_json) if result_json.exists() else {}
    accounting = aggregate_performance_accounting(records, result_payload)
    full_continuous = build_full_continuous_summary(records, result_payload)
    row.update(
        {
            "generic_full_continuous_enabled": full_continuous.get("generic_full_continuous_enabled"),
            "generic_full_continuous_max_depth": full_continuous.get("generic_full_continuous_max_depth"),
            "generic_full_continuous_max_observed_depth": full_continuous.get(
                "generic_full_continuous_max_observed_depth"
            ),
            "generic_full_continuous_max_real_committed_depth": full_continuous.get(
                "generic_full_continuous_max_real_committed_depth"
            ),
            "generic_full_continuous_depth_commit_token_counts": full_continuous.get(
                "generic_full_continuous_depth_commit_token_counts"
            ),
            "generic_full_continuous_depth_partial_recovered_token_counts": full_continuous.get(
                "generic_full_continuous_depth_partial_recovered_token_counts"
            ),
            "generic_full_continuous_depth_revised_token_counts": full_continuous.get(
                "generic_full_continuous_depth_revised_token_counts"
            ),
            "generic_full_continuous_stop_reason_counts": full_continuous.get(
                "generic_full_continuous_stop_reason_counts"
            ),
            "generic_full_continuous_total_full_commit_token_count": full_continuous.get(
                "generic_full_continuous_total_full_commit_token_count"
            ),
            "generic_full_continuous_total_partial_recovered_token_count": full_continuous.get(
                "generic_full_continuous_total_partial_recovered_token_count"
            ),
            "generic_full_continuous_total_revised_token_count": full_continuous.get(
                "generic_full_continuous_total_revised_token_count"
            ),
            "generic_full_continuous_total_output_token_count": full_continuous.get(
                "generic_full_continuous_total_output_token_count"
            ),
            "generic_full_continuous_depth_gt_max_real_commit_count": full_continuous.get(
                "generic_full_continuous_depth_gt_max_real_commit_count"
            ),
            "generic_full_continuous_normal_lane_conflict_count": full_continuous.get(
                "generic_full_continuous_normal_lane_conflict_count"
            ),
            "generic_full_continuous_target_draft_mismatch_count": full_continuous.get(
                "generic_full_continuous_target_draft_mismatch_count"
            ),
            "generic_full_continuous_parity_ok": full_continuous.get("generic_full_continuous_parity_ok"),
            "partial_prefix_recovery_attempt_count": accounting.get("partial_prefix_recovery_attempt_count"),
            "partial_prefix_recovery_success_count": accounting.get("partial_prefix_recovery_success_count"),
            "partial_prefix_accepted_token_count": accounting.get("partial_prefix_accepted_token_count"),
            "partial_prefix_revised_token_count": accounting.get("partial_prefix_revised_token_count"),
            "partial_prefix_total_recovered_token_count": accounting.get(
                "partial_prefix_total_recovered_token_count"
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


def run_case(
    args: argparse.Namespace,
    case: FullContinuousMaxDepthCase,
    env: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    case_dir = Path(args.out_root) / case.name
    result_json = case_dir / "result.json"
    engine_trace = case_dir / "engine_trace.json"
    chain_summary_path = case_dir / "chain_summary.json"
    if not args.print_only:
        case_dir.mkdir(parents=True, exist_ok=True)

    status = run_command(eval_command_8u(args, case, result_json, engine_trace), env, args.print_only)
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
        if core_chain_passed and case.require_depth_gt4_activity:
            strict_status = run_command(
                [
                    args.python,
                    "benchmark/check_full_continuous_max_depth.py",
                    str(engine_trace),
                    str(result_json),
                    "--require-depth-gt4-activity",
                ],
                env,
                args.print_only,
            )
            if strict_status != 0:
                status = strict_status
                check_status = "failed:benchmark/check_full_continuous_max_depth.py:depth_gt4_activity"
                core_checker_chain_status = check_status
                core_checker_chain_failed_checker = "benchmark/check_full_continuous_max_depth.py"
                core_chain_passed = False
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
        "generic_rolling_runtime_enabled": case.apply_case.runtime_case.generic_runtime_enabled,
        "generic_rolling_apply_path_enabled": case.apply_case.generic_apply_enabled,
        "generic_full_continuous_enabled": case.full_continuous_enabled,
        "generic_full_continuous_max_depth": case.max_depth if case.full_continuous_enabled else 0,
        "require_depth_gt4_activity": bool(case.require_depth_gt4_activity),
        "partial_prefix_recovery_enabled": case.apply_case.runtime_case.base_case.partial_recovery_enabled,
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
        parity_fields = [
            "generic_legacy_combined_parity_ok",
            "generic_legacy_depth_token_parity_ok",
            "generic_legacy_safety_parity_ok",
            "legacy_generic_parity_ok",
            "generic_chain_accounting_ok",
            "chain_summary_written",
            "generic_rolling_parity_ok",
            "generic_rolling_apply_parity_ok",
        ]
        if case.full_continuous_enabled:
            parity_fields.append("generic_full_continuous_parity_ok")
        parity_ok = all(bool(row.get(field)) for field in parity_fields)
        if not parity_ok:
            status = 1
            row["check_status"] = "failed:full_continuous_parity"
            row["core_checker_chain_status"] = "failed:full_continuous_parity"
            row["core_checker_chain_failed_checker"] = "full_continuous_parity"
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
    parser = argparse.ArgumentParser(description="Run Phase 1H-8u full continuous max-depth cases.")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-model")
    parser.add_argument("--workload-in")
    parser.add_argument("--out-root", default="results/multislo/phase1h8u_full_continuous_depth100")
    parser.add_argument(
        "--cases",
        default=(
            "baseline_8t_generic_apply_depth4,full_continuous_depth100_baseline,"
            "full_continuous_depth100_partial_recovery,full_continuous_depth100_stress,"
            "full_continuous_depth100_must_exceed4"
        ),
    )
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
    summary_path = Path(args.out_root) / "phase1h8u_summary.json"
    if not args.print_only:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote summary: {summary_path}")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
