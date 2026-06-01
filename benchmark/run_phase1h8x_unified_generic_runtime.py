#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CASES = (
    "legacy_full_continuous_depth100",
    "unified_generic_depth4_baseline",
    "unified_generic_depth8_baseline",
    "unified_generic_depth100_baseline",
    "unified_generic_depth100_cached_admission",
    "unified_generic_depth100_partial_recovery",
    "unified_generic_depth100_cached_partial_recovery",
)

CHECKERS = (
    ("check_multislo_result", "result", ()),
    ("check_cached_admission", "trace_result", ()),
    ("check_unified_generic_rolling_runtime", "trace_result", ()),
    ("check_full_continuous_max_depth", "trace_result", ()),
    ("check_eager_partial_prefix_recovery", "trace_result", ()),
    ("check_eager_performance_accounting", "trace_result", ("--check-generic-chain",)),
)


def selected_cases(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return list(CASES) if names == ["all"] else names


def command_for(python: str, checker: str, mode: str, trace: Path, result: Path, extra: tuple[str, ...]) -> list[str]:
    command = [python, f"benchmark/{checker}.py"]
    if mode == "result":
        command.append(str(result))
    elif mode == "trace_result":
        command.extend([str(trace), str(result)])
    else:
        raise ValueError(f"unknown checker mode: {mode}")
    command.extend(extra)
    return command


def run(command: list[str], *, verbose: bool) -> int:
    completed = subprocess.run(command, text=True, capture_output=not verbose)
    if completed.returncode and not verbose:
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
    return completed.returncode


def validate_case(args: argparse.Namespace, case: str) -> int:
    case_dir = args.root / case
    trace = case_dir / "engine_trace.json"
    result = case_dir / "result.json"
    print(f"case: {case}")
    if not trace.exists() or not result.exists():
        print(f"  missing trace/result: trace={trace.exists()} result={result.exists()}")
        return 1
    exit_code = 0
    for checker, mode, extra in CHECKERS:
        if case.startswith("legacy_") and checker == "check_unified_generic_rolling_runtime":
            continue
        command = command_for(args.python, checker, mode, trace, result, extra)
        if args.print_commands:
            print("  " + " ".join(command))
            continue
        status = run(command, verbose=args.verbose)
        print(f"  {checker}: {'pass' if status == 0 else f'fail({status})'}")
        if status != 0:
            exit_code = status
            if args.strict:
                return status
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8x unified generic runtime output folders.")
    parser.add_argument("--root", type=Path, default=Path("/tmp/obs2/unified"))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    exit_code = 0
    for case in selected_cases(args.cases):
        status = validate_case(args, case)
        if status != 0:
            exit_code = status
            if args.strict:
                return status
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
