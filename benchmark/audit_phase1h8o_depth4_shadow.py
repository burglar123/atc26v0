#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNTIME_REL = Path("nano_pearl/pearl_engine/pearl_model_runner.py")
CONFIG_REL = Path("nano_pearl/pearl_config.py")
EVAL_REL = Path("benchmark/eval_multi_slo.py")


def load_trace_keys(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    records = data if isinstance(data, list) else data.get("records", [])
    keys: set[str] = set()
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                keys.update(str(key) for key in record)
    return keys


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def function_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def audit_repo(repo_root: Path) -> list[str]:
    errors: list[str] = []
    runtime_text = (repo_root / RUNTIME_REL).read_text()
    config_text = (repo_root / CONFIG_REL).read_text()
    eval_text = (repo_root / EVAL_REL).read_text()

    if "enable_rolling_continuous_depth4_shadow_dry_run" not in config_text:
        errors.append("depth4 shadow config flag missing")
    if "--enable-rolling-continuous-depth4-shadow-dry-run" not in eval_text:
        errors.append("depth4 shadow eval CLI flag missing")
    if "enable_rolling_continuous_depth4_commit" in runtime_text + config_text + eval_text:
        errors.append("depth4 commit flag must not exist in 8o")
    if "enable_rolling_continuous_depth5" in runtime_text + config_text + eval_text:
        errors.append("depth5 runtime flag must not exist in 8o")
    if "unbounded" in runtime_text.lower():
        errors.append("unexpected unbounded runtime reference present")
    if "partial_commit" in runtime_text or "partial commit" in runtime_text.lower():
        errors.append("unexpected partial commit runtime reference present")
    if "RollingProposalCommitRecord(" not in runtime_text:
        errors.append("RollingProposalCommitRecord construction missing")
    if "_run_rolling_depth4_shadow_dry_run" not in function_names(runtime_text):
        errors.append("depth4 shadow runtime helper missing")
    if "rolling_depth4_real_commit_count\"] = 0" not in runtime_text:
        errors.append("depth4 real commit hard-zero trace assignment missing")
    return errors


def audit_traces(before_trace: Path | None, after_trace: Path | None) -> list[str]:
    errors: list[str] = []
    if before_trace is None or after_trace is None:
        return errors
    before_keys = load_trace_keys(before_trace)
    after_keys = load_trace_keys(after_trace)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    unexpected_removed = removed
    expected_added_prefixes = ("rolling_depth4_", "enable_rolling_continuous_depth4_shadow_dry_run")
    unexpected_added = [
        key for key in added if not key.startswith(expected_added_prefixes)
    ]
    print(f"before_trace_key_count={len(before_keys)}")
    print(f"after_trace_key_count={len(after_keys)}")
    print(f"trace_keys_added={added}")
    print(f"trace_keys_removed={removed}")
    if unexpected_removed:
        errors.append(f"trace keys removed: {unexpected_removed}")
    if unexpected_added:
        errors.append(f"unexpected trace keys added: {unexpected_added}")
    return errors


def audit_chain_summaries(before_summary: Path | None, after_summary: Path | None) -> list[str]:
    errors: list[str] = []
    if before_summary is None or after_summary is None:
        return errors
    before = load_summary(before_summary)
    after = load_summary(after_summary)
    print(f"before_max_observed_depth={before.get('max_observed_depth')}")
    print(f"after_max_observed_depth={after.get('max_observed_depth')}")
    for field in (
        "depth4_real_commit_count",
        "depth_gt4_real_commit_count",
        "normal_lane_conflict_count",
        "combined_accounting_ok",
        "target_draft_accounting_ok",
        "legacy_generic_parity_ok",
        "generic_chain_accounting_ok",
    ):
        print(f"{field}: before={before.get(field)} after={after.get(field)}")
    if int(after.get("depth4_real_commit_count") or 0) != 0:
        errors.append("after chain summary reports depth4 real commit")
    if int(after.get("depth_gt4_real_commit_count") or 0) != 0:
        errors.append("after chain summary reports depth>4 real commit")
    if int(after.get("max_real_committed_depth") or 0) > 3:
        errors.append("after chain summary max real depth exceeds 3")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Best-effort Phase 1H-8o depth4 shadow audit.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--before-trace", type=Path)
    parser.add_argument("--after-trace", type=Path)
    parser.add_argument("--before-chain-summary", type=Path)
    parser.add_argument("--after-chain-summary", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    errors = audit_repo(repo_root)
    errors.extend(audit_traces(args.before_trace, args.after_trace))
    errors.extend(audit_chain_summaries(args.before_chain_summary, args.after_chain_summary))
    if errors:
        print(json.dumps({"check_status": "fail", "errors": errors}, indent=2, sort_keys=True))
        return 1
    print("check_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
