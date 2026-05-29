#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUNTIME_FILE = Path("nano_pearl/pearl_engine/pearl_model_runner.py")
REQUIRED_RUNTIME_RECORDS = ("RollingProposalCommitRecord", "RollingCommitTraceBundle")
FORBIDDEN_FLAG_RE = re.compile(r"\benable_[a-zA-Z0-9_]*\b")
SUSPICIOUS_RUNTIME_RE = re.compile(r"\b(depth4|depth_?4|unbounded|partial)\b", re.IGNORECASE)
ADDED_DEF_RE = re.compile(r"^\+\s+def ([a-zA-Z0-9_]+)\(")
ADDED_CLASS_RE = re.compile(r"^\+\s*class ([A-Za-z0-9_]+)(?:\(|:)")

CHAIN_SAFETY_FIELDS = (
    "max_observed_depth",
    "max_real_committed_depth",
    "depth4_real_commit_count",
    "depth_gt3_real_commit_count",
    "normal_lane_conflict_count",
    "missing_buffered_proposal_unexpected_count",
    "duplicate_commit_count",
    "invalid_committed_child_count",
    "cascade_committed_child_count",
    "parent_missing_committed_child_count",
    "combined_accounting_ok",
    "target_draft_accounting_ok",
    "legacy_generic_parity_ok",
    "generic_chain_accounting_ok",
)
CHAIN_TOKEN_FIELDS = (
    "one_shot_committed_token_count",
    "depth1_committed_token_count",
    "depth2_committed_token_count",
    "depth3_committed_token_count",
    "combined_real_committed_token_count",
)


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


def audit_runtime_skeleton(repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    runtime_path = repo_root / RUNTIME_FILE
    runtime_text = runtime_path.read_text(encoding="utf-8")
    diff_lines = run_git_diff(repo_root, RUNTIME_FILE)
    added_lines = added_runtime_lines(diff_lines)
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
    added_function_names = sorted(
        {
            match.group(1)
            for line in added_lines
            for match in [ADDED_DEF_RE.match(line)]
            if match is not None
        }
    )
    non_internal_functions = [name for name in added_function_names if not name.startswith("_")]
    added_class_names = sorted(
        {
            match.group(1)
            for line in added_lines
            for match in [ADDED_CLASS_RE.match(line)]
            if match is not None
        }
    )
    unexpected_classes = [
        name for name in added_class_names if name not in REQUIRED_RUNTIME_RECORDS
    ]
    missing_records = [name for name in REQUIRED_RUNTIME_RECORDS if name not in runtime_text]

    errors: list[str] = []
    if missing_records:
        errors.append(f"required runtime skeleton records missing: {missing_records}")
    if added_flags:
        errors.append(f"new runtime flag-looking tokens added: {added_flags}")
    if suspicious_lines:
        errors.append(f"suspicious depth4/unbounded/partial runtime additions: {suspicious_lines}")
    if non_internal_functions:
        errors.append(f"non-internal runtime functions added: {non_internal_functions}")
    if unexpected_classes:
        errors.append(f"unexpected runtime classes added: {unexpected_classes}")

    return errors, {
        "runtime_file": str(RUNTIME_FILE),
        "runtime_diff_line_count": len(diff_lines),
        "runtime_added_line_count": len(added_lines),
        "runtime_record_names_present": {
            name: name in runtime_text for name in REQUIRED_RUNTIME_RECORDS
        },
        "runtime_function_names_added": added_function_names,
        "runtime_class_names_added": added_class_names,
        "runtime_function_style_ok": not non_internal_functions,
        "runtime_class_scope_ok": not unexpected_classes,
        "new_runtime_flag_tokens": added_flags,
        "suspicious_runtime_additions": suspicious_lines,
    }


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


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} does not contain a JSON object")
    return data


def compare_chain_summaries(before_path: Path, after_path: Path) -> tuple[list[str], dict[str, Any]]:
    before = load_json_object(before_path)
    after = load_json_object(after_path)
    errors: list[str] = []
    safety_comparison: dict[str, dict[str, Any]] = {}
    token_comparison: dict[str, dict[str, Any]] = {}

    for field in CHAIN_SAFETY_FIELDS:
        before_value = before.get(field)
        after_value = after.get(field)
        safety_comparison[field] = {"before": before_value, "after": after_value}
        if before_value != after_value:
            errors.append(f"chain summary safety mismatch for {field}: before={before_value!r} after={after_value!r}")

    for field in CHAIN_TOKEN_FIELDS:
        before_value = before.get(field)
        after_value = after.get(field)
        token_comparison[field] = {"before": before_value, "after": after_value}
        try:
            before_int = int(before_value)
            after_int = int(after_value)
        except Exception:
            errors.append(f"chain summary token field {field} is not integer-like")
            continue
        if before_int < 0 or after_int < 0:
            errors.append(f"chain summary token field {field} must be non-negative")
        if (before_int == 0) != (after_int == 0):
            errors.append(
                f"chain summary token zero/nonzero state changed for {field}: "
                f"before={before_int} after={after_int}"
            )

    return errors, {
        "before_chain_summary": str(before_path),
        "after_chain_summary": str(after_path),
        "chain_summary_safety_comparison": safety_comparison,
        "chain_summary_token_comparison": token_comparison,
    }


def print_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        print(f"{key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Best-effort audit for Phase 1H-8m runtime skeleton.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--before-trace", type=Path)
    parser.add_argument("--after-trace", type=Path)
    parser.add_argument("--before-chain-summary", type=Path)
    parser.add_argument("--after-chain-summary", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    errors, summary = audit_runtime_skeleton(repo_root)

    if args.before_trace or args.after_trace:
        if not args.before_trace or not args.after_trace:
            raise SystemExit("--before-trace and --after-trace must be provided together")
        trace_errors, trace_summary = compare_trace_keys(args.before_trace, args.after_trace)
        errors.extend(trace_errors)
        summary.update(trace_summary)

    if args.before_chain_summary or args.after_chain_summary:
        if not args.before_chain_summary or not args.after_chain_summary:
            raise SystemExit("--before-chain-summary and --after-chain-summary must be provided together")
        chain_errors, chain_summary = compare_chain_summaries(
            args.before_chain_summary,
            args.after_chain_summary,
        )
        errors.extend(chain_errors)
        summary.update(chain_summary)

    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase 1H-8m runtime skeleton audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
