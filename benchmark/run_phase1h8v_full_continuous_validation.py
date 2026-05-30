#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.summarize_phase1h8u_full_continuous import (  # noqa: E402
    DEFAULT_CASES,
    compact_rows,
    load_summary_rows,
    resolve_summary_path,
    strict_errors,
)


CHECKER_CHAIN: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("check_eager_partial_prefix_recovery", "trace_result", ()),
    ("check_bounded_rolling_readiness_audit", "trace_result", ("--check-generic-parity",)),
    ("check_generic_bounded_rolling_chain", "trace_result", ()),
    ("check_generic_rolling_runtime_parity", "trace_result", ()),
    ("check_generic_rolling_apply_path_parity", "trace_result", ()),
    ("check_full_continuous_max_depth", "trace_result", ()),
    ("check_eager_performance_accounting", "trace_result", ("--check-generic-chain",)),
    ("check_multislo_result", "result", ()),
)


def case_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if names == ["all"]:
        return list(DEFAULT_CASES)
    return names


def checker_path(name: str) -> str:
    return f"benchmark/{name}.py"


def build_command(
    python: str,
    checker_name: str,
    mode: str,
    trace: Path,
    result: Path,
    extra: tuple[str, ...],
) -> list[str]:
    command = [python, checker_path(checker_name)]
    if mode == "trace_result":
        command.extend([str(trace), str(result)])
    elif mode == "result":
        command.append(str(result))
    else:
        raise ValueError(f"unknown checker mode: {mode}")
    command.extend(extra)
    return command


def format_command(command: list[str]) -> str:
    return " ".join(command)


def run_checker(command: list[str], *, verbose: bool) -> int:
    completed = subprocess.run(command, text=True, capture_output=not verbose)
    if verbose:
        return completed.returncode
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
    return completed.returncode


def run_summary(root: Path, *, strict: bool, summary_only: bool) -> int:
    summary_path = resolve_summary_path(root, None)
    rows = load_summary_rows(summary_path)
    compact = compact_rows(rows)
    errors = strict_errors(rows) if strict else []
    print(f"summary_path={summary_path}")
    for row in compact:
        print(
            "summary: "
            f"{row['case']} status={row['status']} "
            f"full_continuous={row['full_continuous']} "
            f"max_real={row['max_real']} output={row['output_tokens']} "
            f"combined={row['combined']} depth_gt4_activity={row['depth_gt4_activity']}"
        )
    if errors:
        print("summary_strict=fail")
        for error in errors:
            print(f"- {error}")
        return 1
    if strict or summary_only:
        print("summary_strict=pass" if strict else "summary_status=pass")
    return 0


def run_case(args: argparse.Namespace, case: str) -> int:
    case_dir = args.root / case
    trace = case_dir / "engine_trace.json"
    result = case_dir / "result.json"
    print(f"case: {case}")
    if not trace.exists() or not result.exists():
        print(f"  missing: trace={trace.exists()} result={result.exists()}")
        return 1

    exit_code = 0
    for checker_name, mode, extra in CHECKER_CHAIN:
        command = build_command(args.python, checker_name, mode, trace, result, extra)
        if args.print_commands:
            print(f"  {checker_name}: {format_command(command)}")
            continue
        status = run_checker(command, verbose=args.verbose)
        label = "pass" if status == 0 else f"fail({status})"
        print(f"  {checker_name}: {label}")
        if status != 0:
            exit_code = status
            if args.strict:
                return status

    if case == "full_continuous_depth100_must_exceed4":
        command = build_command(
            args.python,
            "check_full_continuous_max_depth",
            "trace_result",
            trace,
            result,
            ("--require-depth-gt4-activity",),
        )
        if args.print_commands:
            print(f"  check_full_continuous_max_depth: {format_command(command)}")
        else:
            status = run_checker(command, verbose=args.verbose)
            label = "pass" if status == 0 else f"fail({status})"
            print(f"  check_full_continuous_max_depth: {label} --require-depth-gt4-activity")
            if status != 0:
                exit_code = status
                if args.strict:
                    return status

    rows = load_summary_rows(resolve_summary_path(args.root, None))
    compact_by_case = {str(row["case"]): row for row in compact_rows(rows)}
    row = compact_by_case.get(case)
    if row is not None:
        print(
            f"  summary: pass, max_real={row['max_real']}, "
            f"output={row['output_tokens']}, combined={row['combined']}"
        )
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate existing Phase 1H-8u full-continuous outputs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/multislo/phase1h8u_full_continuous_depth100_real_depthgt4_min_fix3"),
    )
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--run-checkers", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summary_only or not (args.run_checkers or args.print_commands):
        return run_summary(args.root, strict=args.strict, summary_only=True)

    exit_code = 0
    if not args.print_commands:
        summary_status = run_summary(args.root, strict=args.strict, summary_only=False)
        if summary_status != 0:
            exit_code = summary_status
            if args.strict:
                return summary_status

    for case in case_names(args.cases):
        status = run_case(args, case)
        if status != 0:
            exit_code = status
            if args.strict:
                return status
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
