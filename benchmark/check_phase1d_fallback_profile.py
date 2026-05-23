#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
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


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def check_budget_gamma(record: dict[str, Any]) -> list[str]:
    errors = []
    budgets = record.get("budgets")
    if not isinstance(budgets, dict):
        return errors
    normal_gamma = record.get("normal_gamma")
    selected = {int(seq_id) for seq_id in record.get("eager_selected_seq_ids") or []}
    eager_trace_enabled = bool(record.get("eager_trace_enabled"))
    seen_normal = set()
    for seq_id, budget in budgets.items():
        if not isinstance(budget, dict):
            continue
        eager_gamma = int(budget.get("eager_gamma", 0) or 0)
        if eager_gamma != 0 and (not eager_trace_enabled or int(seq_id) not in selected):
            errors.append(f"budget[{seq_id}] eager_gamma must be 0, got {eager_gamma}")
        if budget.get("normal_gamma") is not None:
            budget_gamma = int(budget.get("normal_gamma"))
            seen_normal.add(budget_gamma)
            if normal_gamma is not None and budget_gamma != int(normal_gamma):
                errors.append(
                    f"budget[{seq_id}] normal_gamma={budget_gamma} differs from "
                    f"record normal_gamma={normal_gamma}"
                )
    if len(seen_normal) > 1:
        errors.append(f"record has varying budget normal_gamma values: {sorted(seen_normal)}")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python benchmark/check_phase1d_fallback_profile.py <engine_trace.json>")
        return 2

    records = load_trace(Path(sys.argv[1]))
    dual_records = [
        record for record in records
        if record.get("execution_mode") == "dual_batch_pearl" and record.get("dual_batch_enabled") is True
    ]
    fallback_records = [record for record in dual_records if record.get("plan_phase") == "fallback"]
    fallback_reasons = Counter(record.get("fallback_reason") for record in fallback_records)
    missing_fallback_reason = sum(1 for record in fallback_records if not record.get("fallback_reason"))

    target_fractions = [
        fnum(record.get("target_fraction_of_active"))
        for record in dual_records
        if fnum(record.get("target_fraction_of_active")) is not None
    ]
    split_imbalances = [
        fnum(record.get("split_imbalance"))
        for record in dual_records
        if fnum(record.get("split_imbalance")) is not None
    ]

    print(f"dual_batch_records={len(dual_records)}")
    print(f"fallback_count={len(fallback_records)}")
    print(f"fallback_reasons={dict(fallback_reasons)}")
    print(f"missing_fallback_reason_count={missing_fallback_reason}")
    print(f"mean_target_fraction_of_active={mean(target_fractions) if target_fractions else 0.0:.3f}")
    print(f"mean_split_imbalance={mean(split_imbalances) if split_imbalances else 0.0:.3f}")

    errors = []
    if not dual_records:
        errors.append("no dual_batch_pearl records with dual_batch_enabled=True")

    for idx, record in enumerate(dual_records):
        if record.get("target_eager_set"):
            errors.append(f"dual_record[{idx}] has non-empty target_eager_set")
        if record.get("draft_eager_set"):
            errors.append(f"dual_record[{idx}] has non-empty draft_eager_set")
        for field in ("active_seq_count", "target_fraction_of_active", "split_imbalance"):
            if field not in record or record.get(field) is None:
                errors.append(f"dual_record[{idx}] missing {field}")
        for error in check_budget_gamma(record):
            errors.append(f"dual_record[{idx}] {error}")

    for idx, record in enumerate(fallback_records):
        if not record.get("fallback_reason"):
            errors.append(f"fallback_record[{idx}] missing fallback_reason")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nPhase 1D fallback profile check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
