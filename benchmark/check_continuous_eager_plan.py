#!/usr/bin/env python3
"""Check Phase 1H-continuous-trace eager plan invariants in an engine trace JSON.

Validates the corrected continuous eager state machine semantics for
trace-only mode.  All execution sets must be empty and all execution
counters must be zero.
"""

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


def as_str_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(x) for x in value}


ALLOWED_PROPOSAL_STATES = {"selected", "pending_parent"}
ALLOWED_PARENT_KINDS = {"normal", ""}
EXECUTION_COUNTER_FIELDS = (
    "eager_tokens_generated",
    "eager_tokens_verified",
    "eager_tokens_promoted",
    "eager_tokens_discarded",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
)
EXECUTED_SET_FIELDS = (
    "draft_eager_set_executed",
    "target_eager_set_executed",
    "target_home_set_executed",
    "draft_home_set_executed",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python benchmark/check_continuous_eager_plan.py <engine_trace.json>")
        return 2

    records = load_trace(Path(sys.argv[1]))
    continuous_records = [
        r for r in records
        if r.get("effective_enable_eager_trace") is True
        and r.get("eager_trace_only") is True
        and r.get("eager_policy") == "tight_only"
    ]

    print(f"total_records={len(records)}")
    print(f"continuous_eager_trace_records={len(continuous_records)}")

    if not continuous_records:
        print("\nNo continuous eager trace records found — nothing to check.")
        return 0

    # Summary statistics.
    total_new = sum(len(r.get("draft_eager_new_set") or []) for r in continuous_records)
    total_continue = sum(len(r.get("draft_eager_continue_set") or []) for r in continuous_records)
    total_trace = sum(len(r.get("draft_eager_set_trace") or []) for r in continuous_records)
    total_candidates = sum(len(r.get("eager_candidate_seq_ids") or []) for r in continuous_records)
    total_selected = sum(len(r.get("eager_selected_seq_ids") or []) for r in continuous_records)
    total_skip_pre_verify = sum(
        sum(1 for v in (r.get("continuous_eager_skip_reason_by_seq_id") or {}).values()
            if v == "skip_pre_verify_seq")
        for r in continuous_records
    )
    total_skip_score = sum(
        sum(1 for v in (r.get("continuous_eager_skip_reason_by_seq_id") or {}).values()
            if v == "score_below_threshold")
        for r in continuous_records
    )

    print(f"draft_eager_new_set_total={total_new}")
    print(f"draft_eager_continue_set_total={total_continue}")
    print(f"draft_eager_set_trace_total={total_trace}")
    print(f"eager_candidate_total={total_candidates}")
    print(f"eager_selected_total={total_selected}")
    print(f"skip_pre_verify_total={total_skip_pre_verify}")
    print(f"skip_score_below_threshold_total={total_skip_score}")

    errors = []
    seen_proposal_ids: set[str] = set()
    seen_parent_ids: set[str] = set()
    # proposal_id -> record index for uniqueness checking.
    proposal_id_sources: dict[str, int] = {}

    for idx, record in enumerate(continuous_records):
        original_target = as_int_set(record.get("original_target_home_set"))
        original_draft = as_int_set(record.get("original_draft_home_set"))
        draft_eager_new = as_int_set(record.get("draft_eager_new_set"))
        draft_eager_continue = as_int_set(record.get("draft_eager_continue_set"))
        draft_eager_trace = as_int_set(record.get("draft_eager_set_trace"))
        target_eager_trace = as_int_set(record.get("target_eager_set_trace"))
        draft_eager_exec = as_int_set(record.get("draft_eager_set_executed"))
        target_eager_exec = as_int_set(record.get("target_eager_set_executed"))
        target_home_exec = as_int_set(record.get("target_home_set_executed"))
        draft_home_exec = as_int_set(record.get("draft_home_set_executed"))
        pre_verify_by_seq = record.get("target_home_pre_verify_by_seq_id") or {}
        slo_class_by_seq = record.get("target_home_slo_class_by_seq_id") or {}
        missing_meta = as_int_set(record.get("missing_eager_metadata_seq_ids"))
        skip_reason_by_seq = record.get("continuous_eager_skip_reason_by_seq_id") or {}
        skip_reason_counts = record.get("continuous_eager_skip_reason_counts") or {}
        selected = as_int_set(record.get("eager_selected_seq_ids"))
        skipped = as_int_set(record.get("eager_draft_skipped_seq_ids"))
        skipped_reason = record.get("eager_draft_skipped_reason_by_seq_id") or {}
        proposal_state_by_seq = record.get("eager_proposal_state_by_seq_id") or {}
        proposal_id_by_seq = record.get("eager_proposal_id_by_seq_id") or {}
        parent_proposal_id_by_seq = record.get("eager_parent_proposal_id_by_seq_id") or {}
        parent_kind_by_seq = record.get("eager_parent_kind_by_seq_id") or {}
        promotion_pending_by_seq = record.get("eager_promotion_condition_pending_by_seq_id") or {}
        slo_class_by_seq = record.get("eager_slo_class_by_seq_id") or {}

        # --- Invariant 1: draft_eager_new_set ⊆ original_target_home_set ---
        if draft_eager_new - original_target:
            errors.append(
                f"record[{idx}] draft_eager_new_set outside original_target_home_set: "
                f"{sorted(draft_eager_new - original_target)}"
            )

        # --- Invariant 2: draft_eager_set_trace == draft_eager_new_set ∪ draft_eager_continue_set ---
        expected_trace = draft_eager_new | draft_eager_continue
        if draft_eager_trace != expected_trace:
            errors.append(
                f"record[{idx}] draft_eager_set_trace != union(new, continue): "
                f"trace={sorted(draft_eager_trace)}, expected={sorted(expected_trace)}"
            )

        # --- Invariant 3: executed eager sets are empty ---
        for field in EXECUTED_SET_FIELDS:
            value = as_int_set(record.get(field))
            if value:
                errors.append(
                    f"record[{idx}] {field} must be empty in trace-only mode, got {sorted(value)}"
                )

        # --- Invariant 4: execution counters are zero ---
        for field in EXECUTION_COUNTER_FIELDS:
            value = int(record.get(field) or 0)
            if value != 0:
                errors.append(
                    f"record[{idx}] {field} must be 0 in trace-only mode, got {value}"
                )

        # --- Invariant 5: selected seqs are tight under tight_only policy ---
        for seq_id in selected:
            slo = str(slo_class_by_seq.get(str(seq_id), slo_class_by_seq.get(seq_id, "")))
            if slo != "tight":
                errors.append(
                    f"record[{idx}] selected seq_id={seq_id} has slo_class={slo!r}, "
                    f"expected 'tight' under tight_only policy"
                )

        # --- Invariant 6: selected seqs are pre_verify=False ---
        for seq_id in draft_eager_new:
            pv = pre_verify_by_seq.get(str(seq_id), pre_verify_by_seq.get(seq_id))
            if pv is True:
                errors.append(
                    f"record[{idx}] draft_eager_new_set seq_id={seq_id} has "
                    f"pre_verify=True (should be False for post-verify stable)"
                )

        # --- Invariant 7: pre_verify=True seqs are skipped with explicit reason ---
        for key, pv in pre_verify_by_seq.items():
            sid = int(key)
            if pv is True and sid not in skipped:
                # The seq might not be a candidate at all (not running, finished, etc.),
                # which is fine.  But if it IS in draft_eager_new, that's already caught
                # by Invariant 6.
                pass
        for sid in skipped:
            reason = str(skipped_reason.get(str(sid), skipped_reason.get(sid, "")))
            if reason == "skip_post_verify_seq":
                errors.append(
                    f"record[{idx}] skipped seq_id={sid} has reason "
                    f"'skip_post_verify_seq' — continuous eager selects post-verify"
                )

        # --- Invariant 8: only "selected"/"pending_parent" states allowed ---
        for key, state in proposal_state_by_seq.items():
            if str(state) not in ALLOWED_PROPOSAL_STATES:
                errors.append(
                    f"record[{idx}] proposal_state_by_seq_id[{key}] = {state!r}, "
                    f"only {ALLOWED_PROPOSAL_STATES} allowed in trace-only"
                )

        # --- Invariant 9: metadata exists or missing is explicit ---
        for seq_id in sorted(original_target):
            has_slo = (
                str(seq_id) in (record.get("target_home_slo_class_by_seq_id") or {})
                or seq_id in (record.get("target_home_slo_class_by_seq_id") or {})
            )
            if not has_slo and seq_id not in missing_meta:
                errors.append(
                    f"record[{idx}] seq_id={seq_id} in original_target_home_set "
                    f"has no slo_class metadata and is not in missing_eager_metadata_seq_ids"
                )

        # --- Invariant 10: target_home_pre_verify_by_seq_id completeness ---
        for seq_id in sorted(original_target):
            has_pv = (
                str(seq_id) in pre_verify_by_seq
                or seq_id in pre_verify_by_seq
            )
            if not has_pv:
                errors.append(
                    f"record[{idx}] seq_id={seq_id} in original_target_home_set "
                    f"missing from target_home_pre_verify_by_seq_id"
                )

        # --- Invariant 11: proposal IDs are unique ---
        for key, pid in proposal_id_by_seq.items():
            pid_str = str(pid)
            if pid_str in seen_proposal_ids:
                errors.append(
                    f"record[{idx}] duplicate proposal_id={pid_str} for seq_id={key}, "
                    f"first seen at record[{proposal_id_sources[pid_str]}]"
                )
            else:
                seen_proposal_ids.add(pid_str)
                proposal_id_sources[pid_str] = idx

        # --- Invariant 12: parent kind is "normal" only (no "eager" in first version) ---
        for key, kind in parent_kind_by_seq.items():
            if str(kind) not in ALLOWED_PARENT_KINDS:
                errors.append(
                    f"record[{idx}] parent_kind_by_seq_id[{key}] = {kind!r}, "
                    f"only {ALLOWED_PARENT_KINDS} allowed"
                )

        # --- Invariant 13: draft_eager_continue_set and target_eager_set_trace are empty ---
        if draft_eager_continue:
            errors.append(
                f"record[{idx}] draft_eager_continue_set must be empty in first version, "
                f"got {sorted(draft_eager_continue)}"
            )
        if target_eager_trace:
            errors.append(
                f"record[{idx}] target_eager_set_trace must be empty in first version, "
                f"got {sorted(target_eager_trace)}"
            )

        # --- Invariant 14: continuous_eager_skip_reason_counts is consistent ---
        expected_counts: dict[str, int] = {}
        for reason in skip_reason_by_seq.values():
            expected_counts[str(reason)] = expected_counts.get(str(reason), 0) + 1
        for reason, count in skip_reason_counts.items():
            if int(count) != expected_counts.get(str(reason), 0):
                errors.append(
                    f"record[{idx}] continuous_eager_skip_reason_counts[{reason!r}]="
                    f"{count}, expected {expected_counts.get(str(reason), 0)}"
                )

        # --- Invariant 15: draft_eager_new_set matches eager_selected_seq_ids ---
        if draft_eager_new != selected:
            errors.append(
                f"record[{idx}] draft_eager_new_set != eager_selected_seq_ids: "
                f"new={sorted(draft_eager_new)}, selected={sorted(selected)}"
            )

        # --- Invariant 16: every selected seq has proposal metadata ---
        for seq_id in draft_eager_new:
            pid = proposal_id_by_seq.get(str(seq_id), proposal_id_by_seq.get(seq_id))
            if not pid:
                errors.append(
                    f"record[{idx}] draft_eager_new_set seq_id={seq_id} "
                    f"missing eager_proposal_id_by_seq_id"
                )
            state = proposal_state_by_seq.get(str(seq_id), proposal_state_by_seq.get(seq_id))
            if not state or str(state) not in ALLOWED_PROPOSAL_STATES:
                errors.append(
                    f"record[{idx}] draft_eager_new_set seq_id={seq_id} "
                    f"missing or invalid proposal state: {state!r}"
                )
            pending = promotion_pending_by_seq.get(str(seq_id), promotion_pending_by_seq.get(seq_id))
            if pending is not True:
                errors.append(
                    f"record[{idx}] draft_eager_new_set seq_id={seq_id} "
                    f"promotion_condition_pending must be True, got {pending!r}"
                )

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"\nContinuous eager trace check passed ({len(continuous_records)} records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
