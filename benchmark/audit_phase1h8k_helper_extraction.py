#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


RUNTIME_FILE = Path("nano_pearl/pearl_engine/pearl_model_runner.py")
FORBIDDEN_FLAG_RE = re.compile(r"\benable_[a-zA-Z0-9_]*\b")
SUSPICIOUS_RUNTIME_RE = re.compile(r"\b(depth4|depth_?4|unbounded|partial)\b", re.IGNORECASE)
HELPER_DEF_RE = re.compile(r"^\+\s+def (_[a-zA-Z0-9_]+)\(")
ADDED_DEF_RE = re.compile(r"^\+\s+def ([a-zA-Z0-9_]+)\(")


def run_git_diff(repo_root: Path, path: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--", str(path)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or "git diff failed")
    return completed.stdout.splitlines()


def added_runtime_lines(diff_lines: list[str]) -> list[str]:
    return [
        line
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++") and line.strip() != "+"
    ]


def audit_runtime_diff(repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    diff_lines = run_git_diff(repo_root, RUNTIME_FILE)
    added_lines = added_runtime_lines(diff_lines)
    helper_names = sorted(
        {
            match.group(1)
            for line in added_lines
            for match in [HELPER_DEF_RE.match(line)]
            if match is not None
        }
    )
    added_function_names = sorted(
        {
            match.group(1)
            for line in added_lines
            for match in [ADDED_DEF_RE.match(line)]
            if match is not None
        }
    )
    non_helper_function_names = [
        name for name in added_function_names if not name.startswith("_")
    ]
    added_flags = sorted(
        {
            match.group(0)
            for line in added_lines
            for match in FORBIDDEN_FLAG_RE.finditer(line)
        }
    )
    suspicious_lines = [
        line[1:].strip()
        for line in added_lines
        if SUSPICIOUS_RUNTIME_RE.search(line)
    ]
    errors: list[str] = []
    if added_flags:
        errors.append(f"new runtime flag-looking tokens added: {added_flags}")
    if suspicious_lines:
        errors.append(f"suspicious depth4/unbounded/partial runtime additions: {suspicious_lines}")
    if non_helper_function_names:
        errors.append(f"non-internal runtime functions added: {non_helper_function_names}")
    summary = {
        "runtime_file": str(RUNTIME_FILE),
        "runtime_diff_line_count": len(diff_lines),
        "runtime_added_line_count": len(added_lines),
        "helper_names_added": helper_names,
        "runtime_function_names_added": added_function_names,
        "helper_name_style_ok": not non_helper_function_names,
        "new_runtime_flag_tokens": added_flags,
        "suspicious_runtime_additions": suspicious_lines,
    }
    return errors, summary


def load_trace_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records", "events", "iterations", "batches"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise SystemExit(f"{path} does not contain trace records")


def trace_key_set(records: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for record in records:
        keys.update(str(key) for key in record)
    return keys


def compare_trace_keys(before_trace: Path, after_trace: Path) -> tuple[list[str], dict[str, Any]]:
    before_keys = trace_key_set(load_trace_records(before_trace))
    after_keys = trace_key_set(load_trace_records(after_trace))
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    errors: list[str] = []
    if added or removed:
        errors.append(f"trace key-set mismatch: added={added} removed={removed}")
    return errors, {
        "before_trace": str(before_trace),
        "after_trace": str(after_trace),
        "before_trace_key_count": len(before_keys),
        "after_trace_key_count": len(after_keys),
        "trace_keys_added": added,
        "trace_keys_removed": removed,
    }


def print_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        print(f"{key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Best-effort audit for Phase 1H-8k/8l helper extraction.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--before-trace", type=Path)
    parser.add_argument("--after-trace", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    errors, summary = audit_runtime_diff(repo_root)
    if args.before_trace or args.after_trace:
        if not args.before_trace or not args.after_trace:
            raise SystemExit("--before-trace and --after-trace must be provided together")
        trace_errors, trace_summary = compare_trace_keys(args.before_trace, args.after_trace)
        errors.extend(trace_errors)
        summary.update(trace_summary)

    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase 1H-8k/8l helper extraction audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
