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


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    records = data.get("records", []) if isinstance(data, dict) else []
    return [record for record in records if isinstance(record, dict)]


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
    combined_text = "\n".join((runtime_text, config_text, eval_text))
    funcs = function_names(runtime_text)

    if "enable_rolling_continuous_depth4_shadow_dry_run" not in config_text:
        errors.append("depth4 shadow config flag missing")
    if "enable_rolling_continuous_depth4_commit_ready_only" not in config_text:
        errors.append("depth4 commit config flag missing")
    if "--enable-rolling-continuous-depth4-shadow-dry-run" not in eval_text:
        errors.append("depth4 shadow eval CLI flag missing")
    if "--enable-rolling-continuous-depth4-commit-ready-only" not in eval_text:
        errors.append("depth4 commit eval CLI flag missing")
    if "enable_rolling_continuous_depth5" in combined_text:
        errors.append("depth5 runtime flag must not exist in 8p")
    if "unbounded" in runtime_text.lower():
        errors.append("unexpected unbounded runtime reference present")
    if "partial_commit" in runtime_text or "partial commit" in runtime_text.lower():
        errors.append("unexpected partial commit runtime reference present")
    if "RollingProposalCommitRecord(" not in runtime_text:
        errors.append("RollingProposalCommitRecord construction missing")
    if "_run_rolling_depth4_shadow_dry_run" not in funcs:
        errors.append("depth4 shadow runtime helper missing")
    if "_run_rolling_depth4_commit_ready_only" not in funcs:
        errors.append("depth4 commit runtime helper missing")
    if "ROLLING_DEPTH4_COMMIT_MAGIC" not in runtime_text:
        errors.append("depth4 commit broadcast constants missing")
    if "rolling_depth_gt4_real_commit_count" not in runtime_text:
        errors.append("depth>4 real commit guard trace field missing")
    return errors


def audit_trace(path: Path | None) -> list[str]:
    errors: list[str] = []
    if path is None:
        return errors
    records = load_trace(path)
    depth4_commit_enabled = any(
        bool(record.get("enable_rolling_continuous_depth4_commit_ready_only", False))
        or bool(record.get("rolling_depth4_commit_enabled", False))
        for record in records
    )
    depth4_shadow_enabled = any(
        bool(record.get("enable_rolling_continuous_depth4_shadow_dry_run", False))
        or bool(record.get("rolling_depth4_shadow_enabled", False))
        for record in records
    )
    depth4_real_commit_count = sum(int(record.get("rolling_depth4_real_commit_count") or 0) for record in records)
    depth_gt4_real_commit_count = sum(int(record.get("rolling_depth_gt4_real_commit_count") or 0) for record in records)
    depth4_committed_ids: set[int] = set()
    for record in records:
        for proposal_id in record.get("rolling_depth4_real_committed_proposal_ids", []) or []:
            try:
                depth4_committed_ids.add(int(proposal_id))
            except (TypeError, ValueError):
                continue
    print(f"trace_depth4_shadow_enabled={depth4_shadow_enabled}")
    print(f"trace_depth4_commit_enabled={depth4_commit_enabled}")
    print(f"trace_depth4_real_commit_count={depth4_real_commit_count}")
    print(f"trace_depth4_unique_committed_proposal_count={len(depth4_committed_ids)}")
    print(f"trace_depth_gt4_real_commit_count={depth_gt4_real_commit_count}")
    if depth4_commit_enabled and not depth4_shadow_enabled:
        errors.append("trace enables depth4 commit without depth4 shadow")
    if depth4_committed_ids and not depth4_commit_enabled:
        errors.append("trace has depth4 committed proposals while commit flag is disabled")
    if depth_gt4_real_commit_count:
        errors.append("trace reports depth>4 real commit")
    return errors


def audit_chain_summary(path: Path | None) -> list[str]:
    errors: list[str] = []
    if path is None:
        return errors
    summary = load_summary(path)
    depth4_commit_enabled = bool(
        summary.get("depth4_commit_enabled", False)
        or summary.get("rolling_depth4_commit_enabled", False)
        or summary.get("enable_rolling_continuous_depth4_commit_ready_only", False)
    )
    depth4_tokens = int(summary.get("depth4_committed_token_count") or 0)
    if not depth4_tokens:
        depth4_tokens = int(summary.get("rolling_depth4_real_committed_token_count") or 0)
    max_real_depth = int(summary.get("max_real_committed_depth") or 0)
    depth_gt4 = int(summary.get("depth_gt4_real_commit_count") or 0)
    print(f"summary_depth4_commit_enabled={depth4_commit_enabled}")
    print(f"summary_depth4_committed_token_count={depth4_tokens}")
    print(f"summary_max_observed_depth={summary.get('max_observed_depth')}")
    print(f"summary_max_real_committed_depth={max_real_depth}")
    print(f"summary_depth_gt4_real_commit_count={depth_gt4}")
    if depth4_tokens and not depth4_commit_enabled:
        errors.append("chain summary has depth4 committed tokens while commit flag is disabled")
    if max_real_depth > 4:
        errors.append("chain summary max real committed depth exceeds 4")
    if max_real_depth == 4 and not depth4_commit_enabled:
        errors.append("chain summary max real depth is 4 while commit flag is disabled")
    if depth_gt4:
        errors.append("chain summary reports depth>4 real commit")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Best-effort Phase 1H-8p depth4 commit audit.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--chain-summary", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    errors = audit_repo(repo_root)
    errors.extend(audit_trace(args.trace))
    errors.extend(audit_chain_summary(args.chain_summary))
    if errors:
        print(json.dumps({"check_status": "fail", "errors": errors}, indent=2, sort_keys=True))
        return 1
    print("check_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
