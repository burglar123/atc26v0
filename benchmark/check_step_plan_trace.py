#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_trace(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            if isinstance(data.get(key), list):
                return data[key]
        if "requests" in data:
            raise ValueError(
                "No trace records found in engine trace JSON: found 'requests' but none of "
                "'traces', 'records', or 'trace_records'."
            )
    raise ValueError(
        "Unsupported engine trace JSON format. Expected one of: raw list, or dict with "
        "'traces', 'records', or 'trace_records'."
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark/check_step_plan_trace.py <engine_trace.json>")
        return 2

    path = Path(sys.argv[1])
    records = load_trace(path)

    execution_modes = sorted({r.get("execution_mode") for r in records if r.get("execution_mode") is not None})
    runner_roles = sorted({r.get("runner_role") for r in records if r.get("runner_role") is not None})
    with_plan_id = sum(1 for r in records if "plan_id" in r)
    with_draft_home = sum(1 for r in records if "draft_home_set" in r)
    with_target_home = sum(1 for r in records if "target_home_set" in r)

    print(f"trace_records={len(records)}")
    print(f"execution_modes={execution_modes}")
    print(f"runner_roles={runner_roles}")
    print(f"records_with_plan_id={with_plan_id}")
    print(f"records_with_draft_home_set={with_draft_home}")
    print(f"records_with_target_home_set={with_target_home}")

    errors = []
    for idx, r in enumerate(records):
        role = r.get("runner_role", "")
        if "plan_id" not in r:
            errors.append(f"record[{idx}] missing plan_id")
        if "scheduled_seq_ids" not in r:
            errors.append(f"record[{idx}] missing scheduled_seq_ids")
        if "draft_home_set" not in r or "target_home_set" not in r:
            errors.append(f"record[{idx}] missing role-specific plan home sets")
        if r.get("draft_eager_set"):
            errors.append(f"record[{idx}] has non-empty draft_eager_set in Phase 1A")
        if r.get("target_eager_set"):
            errors.append(f"record[{idx}] has non-empty target_eager_set in Phase 1A")

        if "draft" in role:
            if r.get("target_home_set"):
                errors.append(f"record[{idx}] draft role should have empty target_home_set")
        else:
            if r.get("draft_home_set"):
                errors.append(f"record[{idx}] verify/target role should have empty draft_home_set")

    print("plan_examples=")
    for r in records[:5]:
        print(json.dumps({
            "runner_role": r.get("runner_role"),
            "iteration_id": r.get("iteration_id"),
            "plan_id": r.get("plan_id"),
            "batch_id": r.get("batch_id"),
            "scheduled_seq_ids": r.get("scheduled_seq_ids"),
            "draft_home_set": r.get("draft_home_set"),
            "target_home_set": r.get("target_home_set"),
            "draft_eager_set": r.get("draft_eager_set"),
            "target_eager_set": r.get("target_eager_set"),
        }, ensure_ascii=False))

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("\nStepPlan trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
