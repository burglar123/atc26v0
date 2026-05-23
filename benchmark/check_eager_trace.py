#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(
        "Unsupported engine trace JSON format. Expected a raw list, or a dict "
        "with 'traces', 'records', or 'trace_records'."
    )


def as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(x) for x in value}


def budget_for(record: dict[str, Any], seq_id: int) -> dict[str, Any]:
    budgets = record.get("budgets")
    if not isinstance(budgets, dict):
        return {}
    value = budgets.get(str(seq_id), budgets.get(seq_id, {}))
    return value if isinstance(value, dict) else {}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python benchmark/check_eager_trace.py <engine_trace.json>")
        return 2

    records = load_trace(Path(sys.argv[1]))
    eager_enabled_records = [r for r in records if r.get("eager_trace_enabled") is True]
    candidate_count = sum(len(r.get("eager_candidate_seq_ids") or []) for r in records)
    selected_count = sum(len(r.get("eager_selected_seq_ids") or []) for r in records)
    selected_seq_ids = sorted({seq_id for r in records for seq_id in as_int_set(r.get("eager_selected_seq_ids"))})
    selected_by_phase = Counter()
    for record in records:
        if record.get("eager_selected_seq_ids"):
            selected_by_phase[record.get("plan_phase")] += len(record.get("eager_selected_seq_ids") or [])

    print(f"records={len(records)}")
    print(f"records_with_eager_trace_enabled={len(eager_enabled_records)}")
    print(f"eager_candidate_count={candidate_count}")
    print(f"eager_selected_count={selected_count}")
    print(f"selected_eager_seq_ids={selected_seq_ids}")
    print(f"selected_by_plan_phase={dict(selected_by_phase)}")

    errors = []
    for idx, record in enumerate(records):
        enabled = bool(record.get("eager_trace_enabled"))
        selected = as_int_set(record.get("eager_selected_seq_ids"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        target_home = as_int_set(record.get("target_home_set"))
        target_eager = as_int_set(record.get("target_eager_set"))
        budget_by_seq = record.get("eager_budget_by_seq_id") or {}
        max_requests = int(record.get("max_eager_requests_per_step") or 0)
        max_tokens_step = int(record.get("max_eager_tokens_per_step") or 0)
        max_tokens_request = int(record.get("max_eager_tokens_per_request") or 0)
        eager_total_budget = int(record.get("eager_total_budget") or 0)

        if selected and not enabled:
            errors.append(f"record[{idx}] selected eager seqs while eager_trace_enabled is false")
        if selected - target_home:
            errors.append(
                f"record[{idx}] selected eager seqs outside target_home_set: "
                f"{sorted(selected - target_home)}"
            )
        if target_eager:
            errors.append(f"record[{idx}] target_eager_set must be empty, got {sorted(target_eager)}")
        if draft_eager != selected:
            errors.append(
                f"record[{idx}] draft_eager_set must equal selected eager seqs: "
                f"draft_eager={sorted(draft_eager)}, selected={sorted(selected)}"
            )
        if len(selected) > max_requests:
            errors.append(f"record[{idx}] selected {len(selected)} eager seqs above cap {max_requests}")
        if eager_total_budget > max_tokens_step:
            errors.append(
                f"record[{idx}] eager_total_budget={eager_total_budget} above step cap {max_tokens_step}"
            )

        for key, value in budget_by_seq.items():
            seq_id = int(key)
            budget = int(value or 0)
            if budget and seq_id not in selected:
                errors.append(f"record[{idx}] eager budget set for non-selected seq_id={seq_id}")
            if budget > max_tokens_request:
                errors.append(
                    f"record[{idx}] eager budget for seq_id={seq_id} is {budget}, "
                    f"above per-request cap {max_tokens_request}"
                )

        normal_gamma = record.get("normal_gamma")
        for seq_id in as_int_set(record.get("target_home_set")) | as_int_set(record.get("draft_home_set")) | selected:
            budget = budget_for(record, seq_id)
            if not budget:
                continue
            if normal_gamma is not None and int(budget.get("normal_gamma", normal_gamma)) != int(normal_gamma):
                errors.append(f"record[{idx}] normal_gamma mismatch for seq_id={seq_id}")
            eager_gamma = int(budget.get("eager_gamma", 0) or 0)
            if eager_gamma and seq_id not in selected:
                errors.append(f"record[{idx}] non-selected seq_id={seq_id} has eager_gamma={eager_gamma}")
            if seq_id in selected and eager_gamma != int(budget_by_seq.get(str(seq_id), budget_by_seq.get(seq_id, 0)) or 0):
                errors.append(f"record[{idx}] budget eager_gamma differs from eager_budget_by_seq_id for seq_id={seq_id}")

        for field in (
            "eager_tokens_generated",
            "eager_tokens_verified",
            "eager_tokens_promoted",
            "eager_tokens_discarded",
        ):
            if int(record.get(field) or 0) != 0:
                errors.append(f"record[{idx}] {field} must stay 0 in Phase 1G-lite")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nEager trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
