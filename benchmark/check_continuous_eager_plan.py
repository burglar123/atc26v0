#!/usr/bin/env python3
"""Check Phase 1H-continuous-trace eager plan invariants in an engine trace JSON.

Validates the corrected continuous eager state machine semantics for
trace-only mode.  All execution sets must be empty and all execution
counters must be zero.

Phase-aware: strict metadata/proposal checks for steady phase only;
non-steady (fallback/priming) records get safety-only checks.

Proposal-id aware: the same logical proposal is emitted by multiple
runner-role records.  The checker groups by proposal_id and validates
metadata consistency across occurrences.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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


def _get(record: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a field from a record, returning default if absent or None."""
    v = record.get(key)
    return v if v is not None else default


ALLOWED_PROPOSAL_STATES = {"selected", "pending_parent"}
ALLOWED_CONTINUOUS_PROPOSAL_STATES = {
    "selected", "pending_parent", "pending_parent_unknown",
    "ready", "discarded", "target_trace_ready", "continue_pending",
}
ALLOWED_PARENT_KINDS = {"normal", ""}
ALLOWED_CONTINUOUS_PARENT_KINDS = {"normal", "eager", ""}
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


def _safety_checks(record: dict[str, Any], idx: int, phase: str) -> list[str]:
    """Safety checks applicable to all phases (including fallback)."""
    errors: list[str] = []

    for field in EXECUTED_SET_FIELDS:
        value = as_int_set(record.get(field))
        if value:
            errors.append(
                f"record[{idx}] phase={phase} {field} must be empty in trace-only mode, "
                f"got {sorted(value)}"
            )

    for field in EXECUTION_COUNTER_FIELDS:
        value = int(record.get(field) or 0)
        if value != 0:
            errors.append(
                f"record[{idx}] phase={phase} {field} must be 0 in trace-only mode, got {value}"
            )

    return errors


def _steady_checks(
    record: dict[str, Any],
    idx: int,
    original_target: set[int],
    draft_eager_new: set[int],
    draft_eager_continue: set[int],
    draft_eager_trace: set[int],
    target_eager_trace: set[int],
    selected: set[int],
    skipped: set[int],
) -> list[str]:
    """Strict continuous eager checks for steady phase only."""
    errors: list[str] = []

    pre_verify_by_seq = record.get("target_home_pre_verify_by_seq_id") or {}
    slo_class_by_seq = record.get("eager_slo_class_by_seq_id") or {}
    target_slo = record.get("target_home_slo_class_by_seq_id") or {}
    missing_meta = as_int_set(record.get("missing_eager_metadata_seq_ids"))
    skip_reason_by_seq = record.get("continuous_eager_skip_reason_by_seq_id") or {}
    skip_reason_counts = record.get("continuous_eager_skip_reason_counts") or {}
    skipped_reason = record.get("eager_draft_skipped_reason_by_seq_id") or {}
    proposal_state_by_seq = record.get("eager_proposal_state_by_seq_id") or {}
    proposal_id_by_seq = record.get("eager_proposal_id_by_seq_id") or {}
    parent_kind_by_seq = record.get("eager_parent_kind_by_seq_id") or {}
    promotion_pending_by_seq = record.get("eager_promotion_condition_pending_by_seq_id") or {}

    # Invariant: draft_eager_new_set ⊆ original_target_home_set
    if draft_eager_new - original_target:
        errors.append(
            f"record[{idx}] draft_eager_new_set outside original_target_home_set: "
            f"{sorted(draft_eager_new - original_target)}"
        )

    # Invariant: draft_eager_set_trace == draft_eager_new_set ∪ draft_eager_continue_set
    expected_trace = draft_eager_new | draft_eager_continue
    if draft_eager_trace != expected_trace:
        errors.append(
            f"record[{idx}] draft_eager_set_trace != union(new, continue): "
            f"trace={sorted(draft_eager_trace)}, expected={sorted(expected_trace)}"
        )

    # Invariant: draft_eager_new_set matches eager_selected_seq_ids
    if draft_eager_new != selected:
        errors.append(
            f"record[{idx}] draft_eager_new_set != eager_selected_seq_ids: "
            f"new={sorted(draft_eager_new)}, selected={sorted(selected)}"
        )

    # Invariant: selected seqs are tight under tight_only policy
    for seq_id in selected:
        slo = str(slo_class_by_seq.get(str(seq_id), slo_class_by_seq.get(seq_id, "")))
        if slo != "tight":
            errors.append(
                f"record[{idx}] selected seq_id={seq_id} has slo_class={slo!r}, "
                f"expected 'tight' under tight_only policy"
            )

    # Invariant: selected seqs are pre_verify=False
    for seq_id in draft_eager_new:
        pv = pre_verify_by_seq.get(str(seq_id), pre_verify_by_seq.get(seq_id))
        if pv is True:
            errors.append(
                f"record[{idx}] draft_eager_new_set seq_id={seq_id} has "
                f"pre_verify=True (should be False for post-verify stable)"
            )

    # Invariant: no skip_post_verify_seq reason in continuous trace
    for sid in skipped:
        reason = str(skipped_reason.get(str(sid), skipped_reason.get(sid, "")))
        if reason == "skip_post_verify_seq":
            errors.append(
                f"record[{idx}] skipped seq_id={sid} has reason "
                f"'skip_post_verify_seq' — continuous eager selects post-verify"
            )

    # Invariant: only "selected"/"pending_parent" states allowed
    for key, state in proposal_state_by_seq.items():
        if str(state) not in ALLOWED_PROPOSAL_STATES:
            errors.append(
                f"record[{idx}] proposal_state_by_seq_id[{key}] = {state!r}, "
                f"only {ALLOWED_PROPOSAL_STATES} allowed in trace-only"
            )

    # Invariant: metadata exists or missing is explicit
    for seq_id in sorted(original_target):
        has_slo = (
            str(seq_id) in target_slo
            or seq_id in target_slo
        )
        if not has_slo and seq_id not in missing_meta:
            errors.append(
                f"record[{idx}] seq_id={seq_id} in original_target_home_set "
                f"has no slo_class metadata and is not in missing_eager_metadata_seq_ids"
            )

    # Invariant: target_home_pre_verify_by_seq_id completeness
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

    # Invariant: parent kind is "normal" only
    for key, kind in parent_kind_by_seq.items():
        if str(kind) not in ALLOWED_PARENT_KINDS:
            errors.append(
                f"record[{idx}] parent_kind_by_seq_id[{key}] = {kind!r}, "
                f"only {ALLOWED_PARENT_KINDS} allowed"
            )

    # Phase 1H-continuous-promotion-trace: draft_eager_continue_set and
    # target_eager_set_trace may now be non-empty — they are populated from
    # the trace state tracker (ready proposals → target_eager_set_trace,
    # urgent seqs in target_eager_set_trace → draft_eager_continue_set).
    # Validity is enforced by:
    #   - draft_eager_continue_set ⊆ target_eager_set_trace (checked below)
    #   - target_eager_set_trace ∩ exclusion views == ∅ (checked below)

    # Invariant: continuous_eager_skip_reason_counts is consistent
    expected_counts: dict[str, int] = {}
    for reason in skip_reason_by_seq.values():
        expected_counts[str(reason)] = expected_counts.get(str(reason), 0) + 1
    for reason, count in skip_reason_counts.items():
        if int(count) != expected_counts.get(str(reason), 0):
            errors.append(
                f"record[{idx}] continuous_eager_skip_reason_counts[{reason!r}]="
                f"{count}, expected {expected_counts.get(str(reason), 0)}"
            )

    # Invariant: every selected seq has complete proposal metadata
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

    # --- Phase 1H-continuous-promotion-trace checks ---

    # B1. target_eager_set_trace ∩ exclusion views == ∅
    target_home_after_excl = as_int_set(record.get("target_home_set_after_eager_exclusion_trace"))
    draft_home_after_excl = as_int_set(record.get("draft_home_set_after_eager_exclusion_trace"))
    _target_eager_trace_set = as_int_set(record.get("target_eager_set_trace"))
    if target_eager_trace and _target_eager_trace_set:
        overlap_target_home = _target_eager_trace_set & target_home_after_excl
        if overlap_target_home:
            errors.append(
                f"record[{idx}] target_eager_set_trace overlaps "
                f"target_home_set_after_eager_exclusion_trace: {sorted(overlap_target_home)}"
            )
        overlap_draft_home = _target_eager_trace_set & draft_home_after_excl
        if overlap_draft_home:
            errors.append(
                f"record[{idx}] target_eager_set_trace overlaps "
                f"draft_home_set_after_eager_exclusion_trace: {sorted(overlap_draft_home)}"
            )

    # B2. draft_eager_continue_set ⊆ target_eager_set_trace
    if draft_eager_continue and _target_eager_trace_set:
        continue_outside = draft_eager_continue - _target_eager_trace_set
        if continue_outside:
            errors.append(
                f"record[{idx}] draft_eager_continue_set outside target_eager_set_trace: "
                f"{sorted(continue_outside)}"
            )

    # B3. continuous_eager proposal states and parent kinds.
    cont_state_by_seq = record.get("continuous_eager_proposal_state_by_seq_id") or {}
    cont_parent_kind_by_seq = record.get("continuous_eager_parent_kind_by_seq_id") or {}
    for key, state in cont_state_by_seq.items():
        if str(state) not in ALLOWED_CONTINUOUS_PROPOSAL_STATES:
            errors.append(
                f"record[{idx}] continuous_eager_proposal_state_by_seq_id[{key}] = {state!r}, "
                f"only {ALLOWED_CONTINUOUS_PROPOSAL_STATES} allowed in trace-only"
            )
        if str(state) in ("verified", "applied"):
            errors.append(
                f"record[{idx}] continuous_eager_proposal_state_by_seq_id[{key}] = {state!r} "
                f"— no 'verified'/'applied' state allowed in trace-only"
            )
    for key, kind in cont_parent_kind_by_seq.items():
        if str(kind) not in ALLOWED_CONTINUOUS_PARENT_KINDS:
            errors.append(
                f"record[{idx}] continuous_eager_parent_kind_by_seq_id[{key}] = {kind!r}, "
                f"only {ALLOWED_CONTINUOUS_PARENT_KINDS} allowed"
            )

    # B4. Promotion/discard accounting.
    promoted_count = int(record.get("continuous_eager_trace_promoted_count") or 0)
    discarded_count = int(record.get("continuous_eager_trace_discarded_count") or 0)
    promoted_ids = as_int_set(record.get("continuous_eager_promoted_seq_ids"))
    discarded_ids = as_int_set(record.get("continuous_eager_discarded_seq_ids"))
    unknown_ids = as_int_set(record.get("continuous_eager_parent_acceptance_unknown_seq_ids"))
    checked_ids = as_int_set(record.get("continuous_eager_promotion_checked_seq_ids"))
    promoted_actual = len(promoted_ids)
    discarded_actual = len(discarded_ids)
    unknown_actual = len(unknown_ids)
    if promoted_actual != promoted_count:
        errors.append(
            f"record[{idx}] continuous_eager_trace_promoted_count={promoted_count} "
            f"!= len(promoted_seq_ids)={promoted_actual}"
        )
    if discarded_actual != discarded_count:
        errors.append(
            f"record[{idx}] continuous_eager_trace_discarded_count={discarded_count} "
            f"!= len(discarded_seq_ids)={discarded_actual}"
        )
    # Disjointness: a seq can't be both promoted and discarded.
    if promoted_ids & discarded_ids:
        errors.append(
            f"record[{idx}] seq_ids both promoted and discarded: "
            f"{sorted(promoted_ids & discarded_ids)}"
        )
    if promoted_ids & unknown_ids:
        errors.append(
            f"record[{idx}] seq_ids both promoted and unknown: "
            f"{sorted(promoted_ids & unknown_ids)}"
        )
    if discarded_ids & unknown_ids:
        errors.append(
            f"record[{idx}] seq_ids both discarded and unknown: "
            f"{sorted(discarded_ids & unknown_ids)}"
        )
    # Accounting: every checked seq should be in promoted ∪ discarded ∪ unknown.
    accounted = promoted_ids | discarded_ids | unknown_ids
    unaccounted = checked_ids - accounted
    if unaccounted:
        errors.append(
            f"record[{idx}] promotion_checked seq_ids not accounted for "
            f"(not in promoted/discarded/unknown): {sorted(unaccounted)}"
        )

    # B3 continued: chain_depth non-decreasing check (per-record, within record).
    cont_chain = record.get("continuous_eager_chain_depth_by_seq_id") or {}
    for key, depth in cont_chain.items():
        if int(depth) < 0:
            errors.append(
                f"record[{idx}] continuous_eager_chain_depth_by_seq_id[{key}] = {depth} "
                f"(must be >= 0)"
            )

    # B3 continued: parent_proposal_id exists when parent_kind="eager".
    cont_parent_id_by_seq = record.get("continuous_eager_parent_proposal_id_by_seq_id") or {}
    cont_proposal_id_by_seq = record.get("continuous_eager_proposal_id_by_seq_id") or {}
    all_cont_proposal_ids = set(str(v) for v in cont_proposal_id_by_seq.values())
    for key, parent_id in cont_parent_id_by_seq.items():
        parent_id_str = str(parent_id)
        kind = str(cont_parent_kind_by_seq.get(key, cont_parent_kind_by_seq.get(str(key), "")))
        if kind == "eager" and parent_id_str and parent_id_str not in all_cont_proposal_ids:
            # Parent might be from a different record — only flag if it looks malformed.
            if parent_id_str.strip():
                pass  # cross-record references validated in proposal_groups below

    return errors


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

    steady_records = [r for r in continuous_records if r.get("plan_phase") == "steady"]
    non_steady_records = [r for r in continuous_records if r.get("plan_phase") != "steady"]

    print(f"steady_records={len(steady_records)}")
    print(f"non_steady_records={len(non_steady_records)}")

    # --- Summary statistics (steady records only for semantic stats) ---
    total_new = sum(len(r.get("draft_eager_new_set") or []) for r in steady_records)
    total_continue = sum(len(r.get("draft_eager_continue_set") or []) for r in steady_records)
    total_trace = sum(len(r.get("draft_eager_set_trace") or []) for r in steady_records)
    total_candidates = sum(len(r.get("eager_candidate_seq_ids") or []) for r in steady_records)
    total_selected = sum(len(r.get("eager_selected_seq_ids") or []) for r in steady_records)

    print(f"\n--- Steady-phase summary ---")
    print(f"draft_eager_new_set_total={total_new}")
    print(f"draft_eager_continue_set_total={total_continue}")
    print(f"draft_eager_set_trace_total={total_trace}")
    print(f"eager_candidate_total={total_candidates}")
    print(f"eager_selected_total={total_selected}")

    # Skip reason counts across steady records.
    skip_counts: dict[str, int] = {}
    for r in steady_records:
        for reason in (r.get("continuous_eager_skip_reason_by_seq_id") or {}).values():
            skip_counts[str(reason)] = skip_counts.get(str(reason), 0) + 1
    if skip_counts:
        print(f"skip_reason_counts={dict(skip_counts)}")

    # --- Promotion-trace summary (steady records) ---
    total_target_eager_trace = sum(len(r.get("target_eager_set_trace") or []) for r in steady_records)
    total_cont_promoted = sum(int(r.get("continuous_eager_trace_promoted_count") or 0) for r in steady_records)
    total_cont_discarded = sum(int(r.get("continuous_eager_trace_discarded_count") or 0) for r in steady_records)
    total_cont_unknown = sum(len(r.get("continuous_eager_parent_acceptance_unknown_seq_ids") or []) for r in steady_records)
    total_cont_selected = sum(int(r.get("continuous_eager_trace_selected_count") or 0) for r in steady_records)

    # Chain depth distribution across steady records.
    chain_depths: list[int] = []
    for r in steady_records:
        for depth in (r.get("continuous_eager_chain_depth_by_seq_id") or {}).values():
            chain_depths.append(int(depth))

    print(f"\n--- Promotion-trace summary ---")
    print(f"target_eager_set_trace_total={total_target_eager_trace}")
    print(f"continuous_eager_trace_promoted_total={total_cont_promoted}")
    print(f"continuous_eager_trace_discarded_total={total_cont_discarded}")
    print(f"continuous_eager_parent_acceptance_unknown_total={total_cont_unknown}")
    if total_cont_selected > 0:
        print(f"promotion_rate={total_cont_promoted / total_cont_selected:.3f}")
    if total_target_eager_trace > 0 and total_cont_selected > 0:
        print(f"continuation_rate={total_continue / total_target_eager_trace:.3f}" if total_target_eager_trace > 0 else "continuation_rate=N/A")
    if chain_depths:
        print(f"chain_depth_min={min(chain_depths)}")
        print(f"chain_depth_max={max(chain_depths)}")
        print(f"chain_depth_avg={sum(chain_depths) / len(chain_depths):.2f}")
        from collections import Counter as _Counter
        _depth_dist = _Counter(chain_depths)
        print(f"chain_depth_distribution={dict(sorted(_depth_dist.items()))}")

    # --- Missing metadata by phase/role ---
    missing_by_key: dict[tuple[str, str], int] = Counter()
    for r in continuous_records:
        missing = r.get("missing_eager_metadata_seq_ids") or []
        if missing:
            role = r.get("runner_role", r.get("role", "unknown"))
            phase = r.get("plan_phase", "unknown")
            missing_by_key[(str(role), str(phase))] += len(missing)

    if missing_by_key:
        print(f"\n--- Missing metadata by (role, phase) ---")
        for (role, phase), count in missing_by_key.most_common():
            print(f"  ({role}, {phase}): {count}")

    errors: list[str] = []

    # --- Collect proposal ID occurrences across all continuous records ---
    # proposal_id -> list of (record_idx, seq_id_str, parent_id, parent_kind, state)
    proposal_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, record in enumerate(continuous_records):
        proposal_id_by_seq = record.get("eager_proposal_id_by_seq_id") or {}
        parent_id_by_seq = record.get("eager_parent_proposal_id_by_seq_id") or {}
        parent_kind_by_seq = record.get("eager_parent_kind_by_seq_id") or {}
        state_by_seq = record.get("eager_proposal_state_by_seq_id") or {}
        request_id_by_seq = record.get("target_home_request_id_by_seq_id") or {}
        plan_id = record.get("plan_id", -1)
        step_id = record.get("step_id", -1)

        for key, pid in proposal_id_by_seq.items():
            pid_str = str(pid)
            seq_id = int(key)
            proposal_groups[pid_str].append({
                "record_idx": idx,
                "seq_id": seq_id,
                "seq_id_str": str(seq_id),
                "parent_id": str(parent_id_by_seq.get(key, parent_id_by_seq.get(seq_id, ""))),
                "parent_kind": str(parent_kind_by_seq.get(key, parent_kind_by_seq.get(seq_id, ""))),
                "state": str(state_by_seq.get(key, state_by_seq.get(seq_id, ""))),
                "request_id": str(request_id_by_seq.get(str(seq_id), request_id_by_seq.get(seq_id, ""))),
                "plan_id": plan_id,
                "step_id": step_id,
            })

    # --- Check proposal ID consistency within groups ---
    repeated_count = sum(1 for g in proposal_groups.values() if len(g) > 1)
    inconsistent_count = 0
    if repeated_count > 0:
        print(f"\n--- Proposal ID diagnostics ---")
        print(f"total_unique_proposal_ids={len(proposal_groups)}")
        print(f"proposal_ids_seen_in_multiple_records={repeated_count}")
        for pid_str, occurrences in sorted(proposal_groups.items()):
            if len(occurrences) <= 1:
                continue
            # Check consistency.
            ref = occurrences[0]
            inconsistent = False
            reasons = []
            for occ in occurrences[1:]:
                if occ["seq_id"] != ref["seq_id"]:
                    inconsistent = True
                    reasons.append(f"seq_id mismatch: {occ['seq_id']} vs {ref['seq_id']}")
                if occ["parent_kind"] != ref["parent_kind"]:
                    inconsistent = True
                    reasons.append(f"parent_kind mismatch: {occ['parent_kind']!r} vs {ref['parent_kind']!r}")
                if occ["state"] != ref["state"]:
                    inconsistent = True
                    reasons.append(f"state mismatch: {occ['state']!r} vs {ref['state']!r}")
                if occ["parent_id"] != ref["parent_id"]:
                    inconsistent = True
                    reasons.append(f"parent_id mismatch: {occ['parent_id']!r} vs {ref['parent_id']!r}")
            if inconsistent:
                inconsistent_count += 1
                record_ids = [o["record_idx"] for o in occurrences]
                errors.append(
                    f"proposal_id={pid_str} inconsistent across records {record_ids}: "
                    + "; ".join(reasons)
                )

    # --- Collect continuous_eager proposal ID occurrences ---
    cont_proposal_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, record in enumerate(continuous_records):
        cont_pid_by_seq = record.get("continuous_eager_proposal_id_by_seq_id") or {}
        cont_parent_id_by_seq = record.get("continuous_eager_parent_proposal_id_by_seq_id") or {}
        cont_parent_kind_by_seq = record.get("continuous_eager_parent_kind_by_seq_id") or {}
        cont_state_by_seq = record.get("continuous_eager_proposal_state_by_seq_id") or {}
        plan_id = record.get("plan_id", -1)
        step_id = record.get("step_id", -1)

        for key, pid in cont_pid_by_seq.items():
            pid_str = str(pid)
            seq_id = int(key)
            cont_proposal_groups[pid_str].append({
                "record_idx": idx,
                "seq_id": seq_id,
                "seq_id_str": str(seq_id),
                "parent_id": str(cont_parent_id_by_seq.get(key, cont_parent_id_by_seq.get(seq_id, ""))),
                "parent_kind": str(cont_parent_kind_by_seq.get(key, cont_parent_kind_by_seq.get(seq_id, ""))),
                "state": str(cont_state_by_seq.get(key, cont_state_by_seq.get(seq_id, ""))),
                "plan_id": plan_id,
                "step_id": step_id,
            })

    # Check continuous proposal ID consistency.
    cont_repeated = sum(1 for g in cont_proposal_groups.values() if len(g) > 1)
    cont_inconsistent = 0
    if cont_repeated > 0:
        print(f"\n--- Continuous proposal ID diagnostics ---")
        print(f"total_unique_continuous_proposal_ids={len(cont_proposal_groups)}")
        print(f"continuous_proposal_ids_seen_in_multiple_records={cont_repeated}")
        for pid_str, occurrences in sorted(cont_proposal_groups.items()):
            if len(occurrences) <= 1:
                continue
            ref = occurrences[0]
            inconsistent = False
            reasons = []
            for occ in occurrences[1:]:
                if occ["seq_id"] != ref["seq_id"]:
                    inconsistent = True
                    reasons.append(f"seq_id mismatch: {occ['seq_id']} vs {ref['seq_id']}")
                if occ["parent_kind"] != ref["parent_kind"]:
                    inconsistent = True
                    reasons.append(f"parent_kind mismatch: {occ['parent_kind']!r} vs {ref['parent_kind']!r}")
                if occ["state"] != ref["state"]:
                    inconsistent = True
                    reasons.append(f"state mismatch: {occ['state']!r} vs {ref['state']!r}")
                if occ["parent_id"] != ref["parent_id"]:
                    inconsistent = True
                    reasons.append(f"parent_id mismatch: {occ['parent_id']!r} vs {ref['parent_id']!r}")
            if inconsistent:
                cont_inconsistent += 1
                record_ids = [o["record_idx"] for o in occurrences]
                errors.append(
                    f"continuous_proposal_id={pid_str} inconsistent across records {record_ids}: "
                    + "; ".join(reasons)
                )

    # --- Collect pending_parent_unknown warnings ---
    unknown_warnings: list[str] = []
    for r in steady_records:
        unknown_ids = r.get("continuous_eager_parent_acceptance_unknown_seq_ids") or []
        for sid in unknown_ids:
            unknown_warnings.append(
                f"record steady seq_id={sid} has pending_parent_unknown — "
                f"parent acceptance could not be determined"
            )
    if unknown_warnings:
        print(f"\n--- Warnings: pending_parent_unknown ({len(unknown_warnings)}) ---")
        for w in unknown_warnings[:20]:
            print(f"  WARNING: {w}")
        if len(unknown_warnings) > 20:
            print(f"  ... and {len(unknown_warnings) - 20} more")

    # --- Cross-record evidence-chain validation ---
    # Every seq in target_eager_set_trace must have a prior promotion/ready event.
    # Build an index: proposal_id -> set of records where it was promoted/readied.
    promoted_proposal_ids: set[str] = set()
    simulated_proposal_ids: set[str] = set()
    for r in continuous_records:
        promoted = r.get("continuous_eager_promoted_seq_ids") or []
        simulated = r.get("continuous_eager_promotion_simulated_seq_ids") or []
        pid_by_seq = r.get("continuous_eager_proposal_id_by_seq_id") or {}
        for sid in promoted:
            pid = pid_by_seq.get(str(sid), pid_by_seq.get(sid, ""))
            if pid:
                promoted_proposal_ids.add(str(pid))
        for sid in simulated:
            pid = pid_by_seq.get(str(sid), pid_by_seq.get(sid, ""))
            if pid:
                simulated_proposal_ids.add(str(pid))

    all_evidenced_ids = promoted_proposal_ids | simulated_proposal_ids

    # Also track proposals that were ever in "ready" state (from trace state).
    ready_proposal_ids: set[str] = set()
    for r in continuous_records:
        state_by_seq = r.get("continuous_eager_proposal_state_by_seq_id") or {}
        pid_by_seq = r.get("continuous_eager_proposal_id_by_seq_id") or {}
        for key, state in state_by_seq.items():
            if str(state) == "ready":
                pid = pid_by_seq.get(key, pid_by_seq.get(str(key), ""))
                if pid:
                    ready_proposal_ids.add(str(pid))
    all_evidenced_ids |= ready_proposal_ids

    # Validate: every seq in target_eager_set_trace must have evidence.
    evidence_errors = []
    for r in continuous_records:
        target_trace = r.get("target_eager_set_trace") or []
        if not target_trace:
            continue
        pid_by_seq = r.get("continuous_eager_proposal_id_by_seq_id") or {}
        for sid in target_trace:
            pid = pid_by_seq.get(str(sid), pid_by_seq.get(sid, ""))
            if pid and str(pid) not in all_evidenced_ids:
                evidence_errors.append(
                    f"seq_id={sid} proposal_id={pid} in target_eager_set_trace "
                    f"has no prior promotion/ready evidence"
                )
    if evidence_errors:
        print(f"\n--- Evidence-chain errors ({len(evidence_errors)}) ---")
        for e in evidence_errors[:20]:
            print(f"  EVIDENCE ERROR: {e}")
        if len(evidence_errors) > 20:
            print(f"  ... and {len(evidence_errors) - 20} more")
        errors.extend(evidence_errors)

    # Validate: if target_eager_set_trace_total > 0, at least some promotions
    # or simulated promotions must exist.
    if total_target_eager_trace > 0 and not all_evidenced_ids:
        errors.append(
            f"target_eager_set_trace_total={total_target_eager_trace} > 0 "
            f"but no promoted, simulated, or ready proposals found across all records"
        )

    # Validate: continue_pending proposals must have a parent that was promoted/ready.
    for r in continuous_records:
        state_by_seq = r.get("continuous_eager_proposal_state_by_seq_id") or {}
        parent_kind_by_seq = r.get("continuous_eager_parent_kind_by_seq_id") or {}
        parent_id_by_seq = r.get("continuous_eager_parent_proposal_id_by_seq_id") or {}
        for key, state in state_by_seq.items():
            if str(state) != "continue_pending":
                continue
            parent_kind = str(parent_kind_by_seq.get(key, parent_kind_by_seq.get(str(key), "")))
            if parent_kind != "eager":
                continue
            parent_pid = str(parent_id_by_seq.get(key, parent_id_by_seq.get(str(key), "")))
            if parent_pid and parent_pid not in all_evidenced_ids:
                errors.append(
                    f"continue_pending proposal {key} has parent_proposal_id={parent_pid} "
                    f"which was never promoted/readied"
                )

    # --- Per-record phase-aware validation ---
    for idx, record in enumerate(continuous_records):
        phase = record.get("plan_phase", "unknown")

        # --- Safety checks for ALL phases ---
        errors.extend(_safety_checks(record, idx, str(phase)))

        # --- Steady-phase strict checks ---
        if phase == "steady":
            original_target = as_int_set(record.get("original_target_home_set"))
            draft_eager_new = as_int_set(record.get("draft_eager_new_set"))
            draft_eager_continue = as_int_set(record.get("draft_eager_continue_set"))
            draft_eager_trace = as_int_set(record.get("draft_eager_set_trace"))
            target_eager_trace = as_int_set(record.get("target_eager_set_trace"))
            selected = as_int_set(record.get("eager_selected_seq_ids"))
            skipped = as_int_set(record.get("eager_draft_skipped_seq_ids"))

            errors.extend(_steady_checks(
                record, idx,
                original_target, draft_eager_new, draft_eager_continue,
                draft_eager_trace, target_eager_trace, selected, skipped,
            ))

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"\nContinuous eager trace check passed ({len(continuous_records)} records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
