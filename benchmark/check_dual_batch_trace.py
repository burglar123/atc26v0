#!/usr/bin/env python3
import json
import sys
from collections import Counter
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
        "Unsupported engine trace JSON format. Expected a raw list, or a dict with "
        "'traces', 'records', or 'trace_records'."
    )


def as_set(value):
    if not isinstance(value, list):
        return set()
    return {int(x) for x in value}


def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark/check_dual_batch_trace.py <engine_trace.json>")
        return 2

    path = Path(sys.argv[1])
    records = load_trace(path)
    execution_modes = sorted({r.get("execution_mode") for r in records if r.get("execution_mode") is not None})
    dual_records = [
        r for r in records
        if r.get("execution_mode") == "dual_batch_pearl" and r.get("dual_batch_enabled") is True
    ]
    phases = Counter(r.get("plan_phase") for r in dual_records)
    records_with_batch_ids = sum(
        1 for r in dual_records
        if "target_batch_id" in r and "draft_batch_id" in r
    )
    steady_same_batch = sum(
        1 for r in dual_records
        if r.get("plan_phase") == "steady" and r.get("target_batch_id") == r.get("draft_batch_id")
    )
    steady_intersections = sum(
        1 for r in dual_records
        if r.get("plan_phase") == "steady" and as_set(r.get("target_home_set")) & as_set(r.get("draft_home_set"))
    )
    non_empty_eager = sum(
        1 for r in dual_records
        if r.get("target_eager_set") or r.get("draft_eager_set")
    )
    unexpected_non_empty_eager = sum(
        1 for r in dual_records
        if (
            (r.get("target_eager_set") and not r.get("eager_execution_enabled"))
            or (r.get("draft_eager_set") and not (r.get("eager_trace_enabled") or r.get("eager_execution_enabled")))
        )
    )
    missing_plan_id = sum(1 for r in dual_records if "plan_id" not in r)
    resolved_mismatch = sum(
        1 for r in dual_records
        if r.get("resolved_seq_ids") != r.get("scheduled_seq_ids")
    )

    print(f"trace_records={len(records)}")
    print(f"execution_modes={execution_modes}")
    print(f"dual_batch_pearl_records={len(dual_records)}")
    print(f"plan_phase_counts={dict(phases)}")
    print(f"records_with_target_and_draft_batch_id_keys={records_with_batch_ids}")
    print(f"steady_records_with_same_batch_id={steady_same_batch}")
    print(f"steady_records_with_intersecting_home_sets={steady_intersections}")
    print(f"records_with_non_empty_eager_sets={non_empty_eager}")
    print(f"records_with_unexpected_non_empty_eager_sets={unexpected_non_empty_eager}")
    print(f"records_missing_plan_id={missing_plan_id}")
    print(f"records_with_resolved_seq_ids_mismatch={resolved_mismatch}")

    required_fields = [
        "execution_mode",
        "dual_batch_enabled",
        "plan_id",
        "step_id",
        "plan_phase",
        "target_batch_id",
        "draft_batch_id",
        "target_home_set",
        "draft_home_set",
        "target_eager_set",
        "draft_eager_set",
        "scheduled_seq_ids",
        "resolved_seq_ids",
        "home_batch_ids",
        "normal_gamma",
        "eager_gamma",
    ]
    errors = []
    if not dual_records:
        errors.append("no dual_batch_pearl records with dual_batch_enabled=True")

    for idx, record in enumerate(dual_records):
        missing = [field for field in required_fields if field not in record]
        if missing:
            errors.append(f"dual_record[{idx}] missing fields: {missing}")
        phase = record.get("plan_phase")
        if phase not in {"priming", "steady", "fallback"}:
            errors.append(f"dual_record[{idx}] invalid plan_phase={phase!r}")
        if record.get("target_eager_set") and not record.get("eager_execution_enabled"):
            errors.append(f"dual_record[{idx}] has non-empty target_eager_set")
        if record.get("draft_eager_set") and not (record.get("eager_trace_enabled") or record.get("eager_execution_enabled")):
            errors.append(f"dual_record[{idx}] has non-empty draft_eager_set")
        if record.get("eager_gamma", 0) != 0:
            errors.append(f"dual_record[{idx}] eager_gamma must be 0")
        if record.get("normal_gamma") is None:
            errors.append(f"dual_record[{idx}] missing normal_gamma")
        if record.get("resolved_seq_ids") != record.get("scheduled_seq_ids"):
            errors.append(
                f"dual_record[{idx}] resolved_seq_ids mismatch: "
                f"scheduled={record.get('scheduled_seq_ids')}, resolved={record.get('resolved_seq_ids')}"
            )
        if phase == "steady":
            if record.get("target_batch_id") == record.get("draft_batch_id"):
                errors.append(f"dual_record[{idx}] steady plan uses the same target/draft batch id")
            if as_set(record.get("target_home_set")) & as_set(record.get("draft_home_set")):
                errors.append(f"dual_record[{idx}] steady plan has intersecting target/draft home sets")

    print("dual_batch_plan_examples=")
    for record in dual_records[:5]:
        print(json.dumps({
            "runner_role": record.get("runner_role"),
            "plan_id": record.get("plan_id"),
            "step_id": record.get("step_id"),
            "plan_phase": record.get("plan_phase"),
            "target_batch_id": record.get("target_batch_id"),
            "draft_batch_id": record.get("draft_batch_id"),
            "target_home_set": record.get("target_home_set"),
            "draft_home_set": record.get("draft_home_set"),
            "scheduled_seq_ids": record.get("scheduled_seq_ids"),
            "resolved_seq_ids": record.get("resolved_seq_ids"),
            "buffered_proposal_seq_ids": record.get("buffered_proposal_seq_ids"),
        }, ensure_ascii=False))

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nDual-batch trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
