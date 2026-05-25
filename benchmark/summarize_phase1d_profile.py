#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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
    raise ValueError(f"Unsupported trace format: {path}")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def isum(values: list[Any]) -> int:
    total = 0
    for value in values:
        try:
            total += int(value or 0)
        except Exception:
            pass
    return total


def first_present(records: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    for record in records:
        if record.get(key) is not None:
            return record.get(key)
    return default


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def is_draft_stage(record: dict[str, Any]) -> bool:
    return record.get("runner_role") in {"draft", "serialized_draft", "dual_draft"}


def is_verify_stage(record: dict[str, Any]) -> bool:
    return record.get("runner_role") in {"verify", "serialized_verify", "dual_verify"}


def group_key(idx: int, record: dict[str, Any]) -> tuple[str, Any]:
    if record.get("step_id") is not None:
        return ("step", record.get("step_id"))
    if record.get("plan_id") is not None:
        return ("plan", record.get("plan_id"), record.get("runner_role"))
    return ("record", idx)


def min_ts(values: list[Any]) -> float | None:
    nums = [fnum(value, None) for value in values]
    nums = [value for value in nums if value is not None]
    return min(nums) if nums else None


def max_ts(values: list[Any]) -> float | None:
    nums = [fnum(value, None) for value in values]
    nums = [value for value in nums if value is not None]
    return max(nums) if nums else None


def group_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    draft_records = [r for r in records if is_draft_stage(r)]
    verify_records = [r for r in records if is_verify_stage(r)]
    draft_start = min_ts([r.get("draft_start_ts") for r in draft_records])
    draft_end = max_ts([r.get("draft_end_ts") for r in draft_records])
    verify_start = min_ts([r.get("verify_start_ts") for r in verify_records])
    verify_end = max_ts([r.get("verify_end_ts") for r in verify_records])

    all_starts = (
        [r.get("step_start_ts") for r in records]
        + [draft_start, verify_start]
        + [r.get("total_iteration_start_ts") for r in records]
    )
    all_ends = (
        [r.get("step_end_ts") for r in records]
        + [draft_end, verify_end]
        + [r.get("total_iteration_end_ts") for r in records]
    )
    step_start = min_ts(all_starts)
    step_end = max_ts(all_ends)

    draft_time_ms = max(0.0, (draft_end - draft_start) * 1000) if draft_start is not None and draft_end is not None else 0.0
    verify_time_ms = max(0.0, (verify_end - verify_start) * 1000) if verify_start is not None and verify_end is not None else 0.0
    step_time_ms = max(0.0, (step_end - step_start) * 1000) if step_start is not None and step_end is not None else sum(fnum(r.get("step_time_ms")) for r in records)
    overlap_time_ms = 0.0
    if draft_start is not None and draft_end is not None and verify_start is not None and verify_end is not None:
        overlap_time_ms = max(0.0, min(draft_end, verify_end) - max(draft_start, verify_start)) * 1000
    overlap_ratio = None
    if min(draft_time_ms, verify_time_ms) > 0:
        overlap_ratio = overlap_time_ms / max(1e-9, min(draft_time_ms, verify_time_ms))

    phase = first_present(records, "plan_phase")
    fallback_reason = first_present(records, "fallback_reason")
    target_home_size = max([
        int(r.get("target_home_size") or len(r.get("target_home_set") or []))
        for r in records
    ], default=0)
    draft_home_size = max([
        int(r.get("draft_home_size") or len(r.get("draft_home_set") or []))
        for r in records
    ], default=0)
    active_batch_size = max([int(r.get("active_batch_size") or 0) for r in records], default=0)
    active_seq_count = max([int(r.get("active_seq_count") or 0) for r in records], default=0)
    active_batch_count = max([int(r.get("active_batch_count") or 0) for r in records], default=0)
    if not active_seq_count:
        active_seq_count = max(active_batch_size, target_home_size + draft_home_size)
    target_fraction_of_active = fnum(first_present(records, "target_fraction_of_active"), None)
    draft_fraction_of_active = fnum(first_present(records, "draft_fraction_of_active"), None)
    split_imbalance = fnum(first_present(records, "split_imbalance"), None)
    target_to_draft_size_ratio = fnum(first_present(records, "target_to_draft_size_ratio"), None)
    if target_fraction_of_active is None:
        target_fraction_of_active = target_home_size / max(1, active_seq_count)
    if draft_fraction_of_active is None:
        draft_fraction_of_active = draft_home_size / max(1, active_seq_count)
    if split_imbalance is None:
        split_imbalance = abs(target_home_size - draft_home_size) / max(1, target_home_size + draft_home_size)
    if target_to_draft_size_ratio is None:
        target_to_draft_size_ratio = target_home_size / max(1, draft_home_size)

    verify_token_records = verify_records if verify_records else records
    hit_seq_ids = set()
    miss_seq_ids = set()
    consumed_seq_ids = set()
    dropped_seq_ids = set()
    invalid_seq_ids = set()
    for record in records:
        hit_seq_ids.update(int(x) for x in record.get("proposal_buffer_hit_seq_ids") or [])
        miss_seq_ids.update(int(x) for x in record.get("proposal_buffer_miss_seq_ids") or [])
        consumed_seq_ids.update(int(x) for x in record.get("proposal_buffer_consumed_seq_ids") or [])
        dropped_seq_ids.update(int(x) for x in record.get("proposal_buffer_dropped_seq_ids") or [])
        invalid_seq_ids.update(int(x) for x in record.get("proposal_buffer_invalid_seq_ids") or [])

    draft_tokens_generated = isum([r.get("draft_tokens_generated") for r in draft_records])
    proposal_tokens_verified = isum([r.get("proposal_tokens_verified") for r in verify_token_records if is_verify_stage(r)])
    accepted_tokens = isum([r.get("accepted_tokens") for r in verify_token_records if is_verify_stage(r)])
    invalidated_predraft_tokens = isum([r.get("invalidated_predraft_tokens") for r in verify_token_records if is_verify_stage(r)])
    dropped_tokens = len(dropped_seq_ids) * int(first_present(records, "normal_gamma", 0) or 0)
    wasted_draft_tokens = invalidated_predraft_tokens + dropped_tokens
    eager_candidate_seq_ids = set()
    eager_selected_seq_ids = set()
    eager_slo_class_by_seq_id = {}
    for record in records:
        eager_candidate_seq_ids.update(int(x) for x in record.get("eager_candidate_seq_ids") or [])
        eager_selected_seq_ids.update(int(x) for x in record.get("eager_selected_seq_ids") or [])
        eager_slo_class_by_seq_id.update(record.get("eager_slo_class_by_seq_id") or {})
    eager_selected_slo_classes = [
        str(eager_slo_class_by_seq_id.get(str(seq_id), eager_slo_class_by_seq_id.get(seq_id, "unknown")))
        for seq_id in sorted(eager_selected_seq_ids)
    ]
    eager_tokens_generated = isum([r.get("eager_tokens_generated") for r in records])
    eager_tokens_promoted = isum([r.get("eager_tokens_promoted") for r in records])
    eager_tokens_discarded = isum([r.get("eager_tokens_discarded") for r in records])
    eager_tokens_verified = isum([r.get("eager_tokens_verified") for r in records])
    eager_tokens_accepted = isum([r.get("eager_tokens_accepted") for r in records])
    eager_tokens_rejected = isum([r.get("eager_tokens_rejected") for r in records])
    records_with_target_eager_set = 1 if any(r.get("target_eager_set") for r in records) else 0

    # Continuous eager trace-only metrics (Phase 1H-continuous-promotion-trace).
    ce_selected = isum([r.get("continuous_eager_trace_selected_count") for r in records])
    ce_promoted = isum([r.get("continuous_eager_trace_promoted_count") for r in records])
    ce_simulated = isum([r.get("continuous_eager_trace_simulated_promoted_total") for r in records])
    ce_discarded = isum([r.get("continuous_eager_trace_discarded_count") for r in records])
    ce_unknown = sum(len(r.get("continuous_eager_parent_acceptance_unknown_seq_ids") or []) for r in records)
    ce_target_trace = sum(len(r.get("target_eager_set_trace") or []) for r in records)
    ce_continue = sum(len(r.get("draft_eager_continue_set") or []) for r in records)
    ce_chain_depths = []
    for r in records:
        for d in (r.get("continuous_eager_chain_depth_by_seq_id") or {}).values():
            ce_chain_depths.append(int(d))

    # Split target/draft metadata: count ready target proposals.
    target_eager_ready_count = 0
    target_eager_total = 0
    for r in records:
        target_state = r.get("target_eager_proposal_state_by_seq_id") or {}
        for state in target_state.values():
            target_eager_total += 1
            if str(state) in ("ready", "target_trace_ready"):
                target_eager_ready_count += 1
    target_eager_without_ready = target_eager_total - target_eager_ready_count

    # Count continuation entries with/without valid target parent.
    cont_with_parent = 0
    cont_without_parent = 0
    for r in records:
        continue_set = r.get("draft_eager_continue_set") or []
        cont_parent = r.get("draft_eager_continue_parent_proposal_id_by_seq_id") or {}
        target_pid = r.get("target_eager_proposal_id_by_seq_id") or {}
        for sid in continue_set:
            cp = str(cont_parent.get(str(sid), cont_parent.get(sid, "")))
            tp = str(target_pid.get(str(sid), target_pid.get(sid, "")))
            if cp and tp and cp == tp:
                cont_with_parent += 1
            else:
                cont_without_parent += 1

    # Phase 1I-A scaffold metrics.
    scaffold_gen = isum([r.get("continuous_eager_scaffold_tokens_generated") for r in records])
    scaffold_prop_gen = isum([r.get("continuous_eager_scaffold_proposals_generated") for r in records])
    scaffold_sent = isum([r.get("continuous_eager_scaffold_proposals_sent") for r in records])
    scaffold_recv = isum([r.get("continuous_eager_scaffold_proposals_received") for r in records])
    scaffold_promoted = isum([r.get("continuous_eager_scaffold_proposals_promoted") for r in records])
    scaffold_discarded = isum([r.get("continuous_eager_scaffold_proposals_discarded") for r in records])
    scaffold_tokens_promoted = isum([r.get("continuous_eager_scaffold_tokens_promoted") for r in records])
    scaffold_tokens_discarded = isum([r.get("continuous_eager_scaffold_tokens_discarded") for r in records])
    scaffold_executed = sum(len(r.get("draft_eager_new_set_executed") or []) for r in records)
    scaffold_send_tokens = isum([r.get("continuous_eager_exec_send_token_count") for r in records])
    scaffold_recv_tokens = isum([r.get("continuous_eager_exec_receive_token_count") for r in records])
    scaffold_sent_seq_count = sum(len(r.get("continuous_eager_exec_sent_seq_ids") or []) for r in records)
    scaffold_received_seq_count = sum(len(r.get("continuous_eager_exec_received_seq_ids") or []) for r in records)
    scaffold_base_val_failures = sum(
        1 for r in records
        for ok in (r.get("continuous_eager_exec_base_validation_ok_by_seq_id") or {}).values()
        if not ok
    )
    scaffold_buf_max = max(
        [int(r.get("continuous_eager_exec_buffer_size_after") or 0) for r in records] + [0]
    )
    scaffold_unknown = sum(
        len(r.get("continuous_eager_exec_parent_unknown_seq_ids") or []) for r in records
    )
    scaffold_receive_val_failures = sum(
        1 for r in records
        if r.get("continuous_eager_exec_receive_validation_ok") is False
    )
    target_eager_set_executed_count = sum(
        len(r.get("target_eager_set_executed") or []) for r in records
    )
    continue_set_executed_count = sum(
        len(r.get("draft_eager_continue_set_executed") or []) for r in records
    )

    return {
        "mode": first_present(records, "execution_mode", "unknown"),
        "phase": phase,
        "fallback_reason": fallback_reason,
        "step_time_ms": step_time_ms,
        "draft_time_ms": draft_time_ms,
        "verify_time_ms": verify_time_ms,
        "overlap_time_ms": overlap_time_ms,
        "overlap_ratio": overlap_ratio,
        "exposed_draft_time_ms": max(0.0, draft_time_ms - overlap_time_ms),
        "pipeline_bubble_ms": max(0.0, step_time_ms - draft_time_ms - verify_time_ms + overlap_time_ms),
        "buffer_hits": len(hit_seq_ids),
        "buffer_misses": len(miss_seq_ids),
        "buffer_consumed": len(consumed_seq_ids),
        "buffer_dropped": len(dropped_seq_ids),
        "buffer_invalid": len(invalid_seq_ids),
        "draft_tokens_generated": draft_tokens_generated,
        "proposal_tokens_verified": proposal_tokens_verified,
        "accepted_tokens": accepted_tokens,
        "invalidated_predraft_tokens": invalidated_predraft_tokens,
        "wasted_draft_tokens": wasted_draft_tokens,
        "target_home_size": target_home_size,
        "draft_home_size": draft_home_size,
        "active_batch_size": active_batch_size,
        "active_seq_count": active_seq_count,
        "active_batch_count": active_batch_count,
        "target_fraction_of_active": target_fraction_of_active,
        "draft_fraction_of_active": draft_fraction_of_active,
        "split_imbalance": split_imbalance,
        "target_to_draft_size_ratio": target_to_draft_size_ratio,
        "eager_trace_enabled": any(bool(r.get("eager_trace_enabled")) for r in records),
        "eager_candidate_count": len(eager_candidate_seq_ids),
        "eager_selected_count": len(eager_selected_seq_ids),
        "eager_total_budget": max([int(r.get("eager_total_budget") or 0) for r in records], default=0),
        "eager_selected_slo_classes": eager_selected_slo_classes,
        "eager_tokens_generated": eager_tokens_generated,
        "eager_tokens_promoted": eager_tokens_promoted,
        "eager_tokens_discarded": eager_tokens_discarded,
        "eager_tokens_verified": eager_tokens_verified,
        "eager_tokens_accepted": eager_tokens_accepted,
        "eager_tokens_rejected": eager_tokens_rejected,
        "records_with_target_eager_set": records_with_target_eager_set,
        "ce_trace_selected": ce_selected,
        "ce_trace_promoted": ce_promoted,
        "ce_trace_simulated_promoted": ce_simulated,
        "ce_trace_discarded": ce_discarded,
        "ce_trace_unknown": ce_unknown,
        "ce_trace_target_set": ce_target_trace,
        "ce_trace_continue_set": ce_continue,
        "ce_trace_chain_depths": ce_chain_depths,
        "ce_trace_max_chain_depth": max(ce_chain_depths) if ce_chain_depths else 0,
        "ce_trace_mean_chain_depth": mean(ce_chain_depths) if ce_chain_depths else 0.0,
        "ce_trace_promotion_rate": ce_promoted / max(1, ce_selected),
        "ce_trace_continuation_rate": ce_continue / max(1, ce_target_trace),
        "target_eager_ready_count": target_eager_ready_count,
        "target_eager_total": target_eager_total,
        "target_eager_without_ready": target_eager_without_ready,
        "cont_with_parent": cont_with_parent,
        "cont_without_parent": cont_without_parent,
        # Phase 1I-A scaffold metrics.
        "scaffold_tokens_generated": scaffold_gen,
        "scaffold_proposals_generated": scaffold_prop_gen,
        "scaffold_proposals_sent": scaffold_sent,
        "scaffold_proposals_received": scaffold_recv,
        "scaffold_proposals_promoted": scaffold_promoted,
        "scaffold_proposals_discarded": scaffold_discarded,
        "scaffold_tokens_promoted": scaffold_tokens_promoted,
        "scaffold_tokens_discarded": scaffold_tokens_discarded,
        "scaffold_executed": scaffold_executed,
        "scaffold_promotion_rate": scaffold_promoted / max(1, scaffold_executed),
        "scaffold_generation_success_rate": scaffold_prop_gen / max(1, scaffold_executed),
        "scaffold_send_tokens": scaffold_send_tokens,
        "scaffold_recv_tokens": scaffold_recv_tokens,
        "scaffold_sent_seq_count": scaffold_sent_seq_count,
        "scaffold_received_seq_count": scaffold_received_seq_count,
        "scaffold_base_val_failures": scaffold_base_val_failures,
        "scaffold_buf_max": scaffold_buf_max,
        "scaffold_unknown": scaffold_unknown,
        "scaffold_receive_val_failures": scaffold_receive_val_failures,
        "target_eager_set_executed_count": target_eager_set_executed_count,
        "continue_set_executed_count": continue_set_executed_count,
    }


def summarize(path: Path) -> dict[str, Any]:
    all_records = load_trace(path)
    records = [r for r in all_records if not r.get("is_prefill")]
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[group_key(idx, record)].append(record)
    step_profiles = [group_profile(group) for group in groups.values()]

    modes = [p["mode"] for p in step_profiles if p["mode"]]
    mode = Counter(modes).most_common(1)[0][0] if modes else "unknown"
    total_step_time_ms = sum(p["step_time_ms"] for p in step_profiles)
    total_draft_time_ms = sum(p["draft_time_ms"] for p in step_profiles)
    total_verify_time_ms = sum(p["verify_time_ms"] for p in step_profiles)
    total_overlap_time_ms = sum(p["overlap_time_ms"] for p in step_profiles)
    ratios = [p["overlap_ratio"] for p in step_profiles if p["overlap_ratio"] is not None]
    phase_counts = Counter(p["phase"] for p in step_profiles if p["phase"] is not None)
    phase_time = defaultdict(float)
    for profile in step_profiles:
        if profile["phase"] is not None:
            phase_time[profile["phase"]] += profile["step_time_ms"]

    buffer_hits = sum(p["buffer_hits"] for p in step_profiles)
    buffer_misses = sum(p["buffer_misses"] for p in step_profiles)
    buffer_dropped = sum(p["buffer_dropped"] for p in step_profiles)
    draft_tokens_generated = sum(p["draft_tokens_generated"] for p in step_profiles)
    proposal_tokens_verified = sum(p["proposal_tokens_verified"] for p in step_profiles)
    accepted_tokens = sum(p["accepted_tokens"] for p in step_profiles)
    invalidated_predraft_tokens = sum(p["invalidated_predraft_tokens"] for p in step_profiles)
    wasted_draft_tokens = sum(p["wasted_draft_tokens"] for p in step_profiles)
    target_sizes = [p["target_home_size"] for p in step_profiles if p["target_home_size"]]
    draft_sizes = [p["draft_home_size"] for p in step_profiles if p["draft_home_size"]]
    active_seq_counts = [p["active_seq_count"] for p in step_profiles if p["active_seq_count"]]
    target_fractions = [p["target_fraction_of_active"] for p in step_profiles if p["active_seq_count"]]
    draft_fractions = [p["draft_fraction_of_active"] for p in step_profiles if p["active_seq_count"]]
    split_imbalances = [p["split_imbalance"] for p in step_profiles]
    target_to_draft_ratios = [p["target_to_draft_size_ratio"] for p in step_profiles]
    eager_selected_by_slo_class = Counter()
    eager_selected_by_plan_phase = Counter()
    for profile in step_profiles:
        eager_selected_by_slo_class.update(profile["eager_selected_slo_classes"])
        if profile["eager_selected_count"]:
            eager_selected_by_plan_phase[profile["phase"]] += profile["eager_selected_count"]
    total_eager_generated = sum(p["eager_tokens_generated"] for p in step_profiles)
    total_eager_promoted = sum(p["eager_tokens_promoted"] for p in step_profiles)
    total_eager_discarded = sum(p["eager_tokens_discarded"] for p in step_profiles)
    total_eager_verified = sum(p["eager_tokens_verified"] for p in step_profiles)
    total_eager_accepted = sum(p["eager_tokens_accepted"] for p in step_profiles)
    total_eager_rejected = sum(p["eager_tokens_rejected"] for p in step_profiles)
    records_with_target_eager_set = sum(p["records_with_target_eager_set"] for p in step_profiles)

    # Phase 1I-A scaffold aggregation.
    scaffold_gen = sum(p["scaffold_tokens_generated"] for p in step_profiles)
    scaffold_prop_gen = sum(p["scaffold_proposals_generated"] for p in step_profiles)
    scaffold_sent = sum(p["scaffold_proposals_sent"] for p in step_profiles)
    scaffold_recv = sum(p["scaffold_proposals_received"] for p in step_profiles)
    scaffold_promoted = sum(p["scaffold_proposals_promoted"] for p in step_profiles)
    scaffold_discarded = sum(p["scaffold_proposals_discarded"] for p in step_profiles)
    scaffold_executed = sum(p["scaffold_executed"] for p in step_profiles)
    scaffold_send_tokens = sum(p["scaffold_send_tokens"] for p in step_profiles)
    scaffold_recv_tokens = sum(p["scaffold_recv_tokens"] for p in step_profiles)
    scaffold_sent_seq_count = sum(p["scaffold_sent_seq_count"] for p in step_profiles)
    scaffold_received_seq_count = sum(p["scaffold_received_seq_count"] for p in step_profiles)
    scaffold_base_val_failures = sum(p["scaffold_base_val_failures"] for p in step_profiles)
    scaffold_buf_max = max([p["scaffold_buf_max"] for p in step_profiles] + [0])
    scaffold_unknown = sum(p["scaffold_unknown"] for p in step_profiles)
    scaffold_receive_val_failures = sum(p["scaffold_receive_val_failures"] for p in step_profiles)
    target_eager_set_executed_count = sum(p["target_eager_set_executed_count"] for p in step_profiles)
    continue_set_executed_count = sum(p["continue_set_executed_count"] for p in step_profiles)

    total_ce_selected = sum(p["ce_trace_selected"] for p in step_profiles)
    total_ce_promoted = sum(p["ce_trace_promoted"] for p in step_profiles)
    total_ce_simulated = sum(p["ce_trace_simulated_promoted"] for p in step_profiles)
    total_ce_discarded = sum(p["ce_trace_discarded"] for p in step_profiles)
    total_ce_unknown = sum(p["ce_trace_unknown"] for p in step_profiles)
    total_ce_target_set = sum(p["ce_trace_target_set"] for p in step_profiles)
    total_ce_continue_set = sum(p["ce_trace_continue_set"] for p in step_profiles)
    all_ce_chain_depths = []
    for p in step_profiles:
        all_ce_chain_depths.extend(p["ce_trace_chain_depths"])

    total_target_eager_ready = sum(p["target_eager_ready_count"] for p in step_profiles)
    total_target_eager = sum(p["target_eager_total"] for p in step_profiles)
    total_target_without_ready = sum(p["target_eager_without_ready"] for p in step_profiles)
    total_cont_with_parent = sum(p["cont_with_parent"] for p in step_profiles)
    total_cont_without_parent = sum(p["cont_without_parent"] for p in step_profiles)

    return {
        "path": str(path),
        "mode": mode,
        "num_records": len(records),
        "num_steps": len(step_profiles),
        "total_step_time_ms": total_step_time_ms,
        "total_draft_time_ms": total_draft_time_ms,
        "total_verify_time_ms": total_verify_time_ms,
        "total_overlap_time_ms": total_overlap_time_ms,
        "mean_overlap_ratio": mean(ratios) if ratios else 0.0,
        "total_exposed_draft_time_ms": sum(p["exposed_draft_time_ms"] for p in step_profiles),
        "total_pipeline_bubble_ms": sum(p["pipeline_bubble_ms"] for p in step_profiles),
        "phase_counts": dict(phase_counts),
        "phase_time": dict(phase_time),
        "proposal_buffer_hit_rate": buffer_hits / max(1, buffer_hits + buffer_misses),
        "proposal_buffer_miss_count": buffer_misses,
        "proposal_buffer_dropped_count": buffer_dropped,
        "draft_tokens_generated": draft_tokens_generated,
        "proposal_tokens_verified": proposal_tokens_verified,
        "accepted_tokens": accepted_tokens,
        "invalidated_predraft_tokens": invalidated_predraft_tokens,
        "draft_waste_rate": wasted_draft_tokens / max(1, draft_tokens_generated),
        "acceptance_rate": accepted_tokens / max(1, proposal_tokens_verified),
        "mean_target_home_size": mean(target_sizes) if target_sizes else 0.0,
        "mean_draft_home_size": mean(draft_sizes) if draft_sizes else 0.0,
        "mean_active_seq_count": mean(active_seq_counts) if active_seq_counts else 0.0,
        "mean_target_fraction_of_active": mean_or_zero(target_fractions),
        "mean_draft_fraction_of_active": mean_or_zero(draft_fractions),
        "mean_split_imbalance": mean_or_zero(split_imbalances),
        "mean_target_to_draft_size_ratio": mean_or_zero(target_to_draft_ratios),
        "avg_eager_candidates_per_step": mean_or_zero([p["eager_candidate_count"] for p in step_profiles]),
        "avg_eager_selected_per_step": mean_or_zero([p["eager_selected_count"] for p in step_profiles]),
        "total_eager_budget": sum(p["eager_total_budget"] for p in step_profiles),
        "eager_selected_by_slo_class": dict(eager_selected_by_slo_class),
        "eager_selected_by_plan_phase": dict(eager_selected_by_plan_phase),
        "total_eager_tokens_generated": total_eager_generated,
        "total_eager_tokens_promoted": total_eager_promoted,
        "total_eager_tokens_discarded": total_eager_discarded,
        "total_eager_tokens_verified": total_eager_verified,
        "total_eager_tokens_accepted": total_eager_accepted,
        "total_eager_tokens_rejected": total_eager_rejected,
        "eager_promotion_rate": total_eager_promoted / max(1, total_eager_generated),
        "eager_verification_acceptance_rate": total_eager_accepted / max(1, total_eager_verified),
        "eager_waste_rate": (total_eager_discarded + total_eager_rejected) / max(1, total_eager_generated),
        "records_with_target_eager_set": records_with_target_eager_set,
        "eager_benefit_proxy": total_eager_verified / max(1, proposal_tokens_verified + total_eager_verified),
        "total_ce_trace_selected": total_ce_selected,
        "total_ce_trace_promoted": total_ce_promoted,
        "total_ce_trace_simulated_promoted": total_ce_simulated,
        "total_ce_trace_discarded": total_ce_discarded,
        "total_ce_trace_unknown": total_ce_unknown,
        "total_ce_trace_target_set": total_ce_target_set,
        "total_ce_trace_continue_set": total_ce_continue_set,
        "ce_trace_max_chain_depth": max(all_ce_chain_depths) if all_ce_chain_depths else 0,
        "ce_trace_mean_chain_depth": mean(all_ce_chain_depths) if all_ce_chain_depths else 0.0,
        "ce_trace_promotion_rate": total_ce_promoted / max(1, total_ce_selected),
        "ce_trace_continuation_rate": total_ce_continue_set / max(1, total_ce_target_set),
        "total_target_eager_ready": total_target_eager_ready,
        "total_target_eager_without_ready": total_target_without_ready,
        "total_cont_with_parent": total_cont_with_parent,
        "total_cont_without_parent": total_cont_without_parent,
        # Phase 1I-A scaffold.
        "scaffold_tokens_generated": scaffold_gen,
        "scaffold_proposals_generated": scaffold_prop_gen,
        "scaffold_proposals_sent": scaffold_sent,
        "scaffold_proposals_received": scaffold_recv,
        "scaffold_proposals_promoted": scaffold_promoted,
        "scaffold_proposals_discarded": scaffold_discarded,
        "scaffold_executed": scaffold_executed,
        "scaffold_promotion_rate": scaffold_promoted / max(1, scaffold_executed),
        "scaffold_send_tokens": scaffold_send_tokens,
        "scaffold_recv_tokens": scaffold_recv_tokens,
        "scaffold_sent_seq_count": scaffold_sent_seq_count,
        "scaffold_received_seq_count": scaffold_received_seq_count,
        "scaffold_base_val_failures": scaffold_base_val_failures,
        "scaffold_buf_max": scaffold_buf_max,
        "scaffold_unknown": scaffold_unknown,
        "scaffold_receive_val_failures": scaffold_receive_val_failures,
        "target_eager_set_executed_count": target_eager_set_executed_count,
        "continue_set_executed_count": continue_set_executed_count,
        "steps": step_profiles,
    }


def fmt_ms(value: float) -> str:
    return f"{value:.2f}"


def print_summary(rows: list[dict[str, Any]]) -> None:
    headers = [
        "file", "mode", "records", "steps", "step_ms", "draft_ms", "verify_ms",
        "overlap_ms", "overlap", "exposed_draft", "bubble_ms", "hit_rate",
        "miss", "drop", "draft_tok", "verified", "accepted", "invalidated",
        "waste", "accept", "tgt_bs", "drf_bs", "active", "tgt_frac",
        "drf_frac", "imbalance", "tgt_drf",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join([
            Path(row["path"]).name,
            str(row["mode"]),
            str(row["num_records"]),
            str(row["num_steps"]),
            fmt_ms(row["total_step_time_ms"]),
            fmt_ms(row["total_draft_time_ms"]),
            fmt_ms(row["total_verify_time_ms"]),
            fmt_ms(row["total_overlap_time_ms"]),
            f"{row['mean_overlap_ratio']:.3f}",
            fmt_ms(row["total_exposed_draft_time_ms"]),
            fmt_ms(row["total_pipeline_bubble_ms"]),
            f"{row['proposal_buffer_hit_rate']:.3f}",
            str(row["proposal_buffer_miss_count"]),
            str(row["proposal_buffer_dropped_count"]),
            str(row["draft_tokens_generated"]),
            str(row["proposal_tokens_verified"]),
            str(row["accepted_tokens"]),
            str(row["invalidated_predraft_tokens"]),
            f"{row['draft_waste_rate']:.3f}",
            f"{row['acceptance_rate']:.3f}",
            f"{row['mean_target_home_size']:.2f}",
            f"{row['mean_draft_home_size']:.2f}",
            f"{row['mean_active_seq_count']:.2f}",
            f"{row['mean_target_fraction_of_active']:.3f}",
            f"{row['mean_draft_fraction_of_active']:.3f}",
            f"{row['mean_split_imbalance']:.3f}",
            f"{row['mean_target_to_draft_size_ratio']:.3f}",
        ]))


def phase_metrics(steps: list[dict[str, Any]]) -> dict[str, float]:
    if not steps:
        return {
            "mean_target_home_size": 0.0,
            "mean_draft_home_size": 0.0,
            "mean_active_seq_count": 0.0,
            "mean_target_fraction_of_active": 0.0,
            "mean_draft_fraction_of_active": 0.0,
            "mean_split_imbalance": 0.0,
            "mean_target_to_draft_size_ratio": 0.0,
        }
    return {
        "mean_target_home_size": mean_or_zero([s["target_home_size"] for s in steps]),
        "mean_draft_home_size": mean_or_zero([s["draft_home_size"] for s in steps]),
        "mean_active_seq_count": mean_or_zero([s["active_seq_count"] for s in steps]),
        "mean_target_fraction_of_active": mean_or_zero([s["target_fraction_of_active"] for s in steps]),
        "mean_draft_fraction_of_active": mean_or_zero([s["draft_fraction_of_active"] for s in steps]),
        "mean_split_imbalance": mean_or_zero([s["split_imbalance"] for s in steps]),
        "mean_target_to_draft_size_ratio": mean_or_zero([s["target_to_draft_size_ratio"] for s in steps]),
    }


def fallback_reason_summary(row: dict[str, Any]) -> dict[str, Any]:
    fallback_steps = [step for step in row["steps"] if step["phase"] == "fallback"]
    reason_counts = Counter(step.get("fallback_reason") or "missing_fallback_reason" for step in fallback_steps)
    reason_time: dict[str, float] = defaultdict(float)
    reason_target_sizes: dict[str, list[int]] = defaultdict(list)
    reason_draft_sizes: dict[str, list[int]] = defaultdict(list)
    total = max(1e-9, row["total_step_time_ms"])
    for step in fallback_steps:
        reason = step.get("fallback_reason") or "missing_fallback_reason"
        reason_time[reason] += step["step_time_ms"]
        reason_target_sizes[reason].append(step["target_home_size"])
        reason_draft_sizes[reason].append(step["draft_home_size"])
    return {
        "counts": dict(reason_counts),
        "time_ms": {key: round(value, 2) for key, value in reason_time.items()},
        "time_pct": {key: round(value / total, 3) for key, value in reason_time.items()},
        "mean_target_size": {
            key: round(mean_or_zero([float(v) for v in values]), 2)
            for key, values in reason_target_sizes.items()
        },
        "mean_draft_size": {
            key: round(mean_or_zero([float(v) for v in values]), 2)
            for key, values in reason_draft_sizes.items()
        },
    }


def dual_warnings(row: dict[str, Any]) -> list[str]:
    total = max(1e-9, row["total_step_time_ms"])
    fallback_time_pct = row["phase_time"].get("fallback", 0.0) / total
    steady_steps = [step for step in row["steps"] if step["phase"] == "steady"]
    steady_target_fraction = phase_metrics(steady_steps)["mean_target_fraction_of_active"]
    warnings = []
    if row["mean_overlap_ratio"] < 0.2:
        warnings.append("low overlap: dual-batch is serialized in practice")
    if fallback_time_pct > 0.10:
        warnings.append("non-trivial fallback time")
    if steady_steps and steady_target_fraction < 0.45:
        warnings.append("target batch splitting may reduce utilization")
    if row["mean_target_home_size"] < 4:
        warnings.append("small target batch may reduce utilization")
    if row["proposal_buffer_hit_rate"] < 0.8:
        warnings.append("proposal buffer miss rate is high")
    if row["draft_waste_rate"] > 0.5:
        warnings.append("draft waste is high")
    return warnings


def dual_bottleneck(row: dict[str, Any]) -> str:
    warnings = dual_warnings(row)
    if warnings:
        return warnings[0]
    return "no single bottleneck identified"


def print_dual_details(rows: list[dict[str, Any]]) -> None:
    dual_rows = [row for row in rows if row["mode"] == "dual_batch_pearl"]
    if not dual_rows:
        return
    print("\ndual_batch_pearl details")
    for row in dual_rows:
        steady_steps = [step for step in row["steps"] if step["phase"] == "steady"]
        steady_ratios = [step["overlap_ratio"] for step in steady_steps if step["overlap_ratio"] is not None]
        steady_hits = sum(step["buffer_hits"] for step in steady_steps)
        steady_misses = sum(step["buffer_misses"] for step in steady_steps)
        steady_hit_rate = steady_hits / max(1, steady_hits + steady_misses)
        phase_splits = {
            phase: phase_metrics([step for step in row["steps"] if step["phase"] == phase])
            for phase in ("priming", "steady", "fallback")
        }
        fallback_reasons = fallback_reason_summary(row)
        total = max(1e-9, row["total_step_time_ms"])
        fallback_pct = row["phase_time"].get("fallback", 0.0) / total
        priming_pct = row["phase_time"].get("priming", 0.0) / total
        rounded_phase_time = {key: round(value, 2) for key, value in row["phase_time"].items()}
        print(f"{Path(row['path']).name}:")
        print(f"  phase_counts={row['phase_counts']}")
        print(f"  phase_time_ms={rounded_phase_time}")
        print(f"  steady_overlap_ratio={mean(steady_ratios) if steady_ratios else 0.0:.3f}")
        print(f"  steady_buffer_hit_rate={steady_hit_rate:.3f}")
        print(f"  fallback_time_pct={fallback_pct:.3f}")
        print(f"  priming_time_pct={priming_pct:.3f}")
        print(f"  phase_batch_splits={phase_splits}")
        print(f"  fallback_reason_counts={fallback_reasons['counts']}")
        print(f"  fallback_reason_time_ms={fallback_reasons['time_ms']}")
        print(f"  fallback_reason_time_pct={fallback_reasons['time_pct']}")
        print(f"  fallback_reason_mean_target_size={fallback_reasons['mean_target_size']}")
        print(f"  fallback_reason_mean_draft_size={fallback_reasons['mean_draft_size']}")
        print(f"  avg_eager_candidates_per_step={row['avg_eager_candidates_per_step']:.3f}")
        print(f"  avg_eager_selected_per_step={row['avg_eager_selected_per_step']:.3f}")
        print(f"  total_eager_budget={row['total_eager_budget']}")
        print(f"  eager_selected_by_slo_class={row['eager_selected_by_slo_class']}")
        print(f"  eager_selected_by_plan_phase={row['eager_selected_by_plan_phase']}")
        print(f"  total_eager_tokens_generated={row['total_eager_tokens_generated']}")
        print(f"  total_eager_tokens_promoted={row['total_eager_tokens_promoted']}")
        print(f"  total_eager_tokens_discarded={row['total_eager_tokens_discarded']}")
        print(f"  total_eager_tokens_verified={row['total_eager_tokens_verified']}")
        print(f"  total_eager_tokens_accepted={row['total_eager_tokens_accepted']}")
        print(f"  eager_promotion_rate={row['eager_promotion_rate']:.3f}")
        print(f"  eager_verification_acceptance_rate={row['eager_verification_acceptance_rate']:.3f}")
        print(f"  eager_waste_rate={row['eager_waste_rate']:.3f}")
        print(f"  records_with_target_eager_set={row['records_with_target_eager_set']}")
        print(f"  eager_benefit_proxy={row['eager_benefit_proxy']:.3f}")
        print(f"  total_ce_trace_selected={row['total_ce_trace_selected']}")
        print(f"  total_ce_trace_promoted={row['total_ce_trace_promoted']}")
        print(f"  total_ce_trace_simulated_promoted={row['total_ce_trace_simulated_promoted']}")
        print(f"  total_ce_trace_discarded={row['total_ce_trace_discarded']}")
        print(f"  total_ce_trace_unknown={row['total_ce_trace_unknown']}")
        print(f"  total_ce_trace_target_set={row['total_ce_trace_target_set']}")
        print(f"  total_ce_trace_continue_set={row['total_ce_trace_continue_set']}")
        print(f"  ce_trace_max_chain_depth={row['ce_trace_max_chain_depth']}")
        print(f"  ce_trace_mean_chain_depth={row['ce_trace_mean_chain_depth']:.3f}")
        print(f"  ce_trace_promotion_rate={row['ce_trace_promotion_rate']:.3f}")
        print(f"  ce_trace_continuation_rate={row['ce_trace_continuation_rate']:.3f}")
        print(f"  total_target_eager_ready={row['total_target_eager_ready']}")
        print(f"  total_target_eager_without_ready={row['total_target_eager_without_ready']}")
        print(f"  total_cont_with_parent={row['total_cont_with_parent']}")
        print(f"  total_cont_without_parent={row['total_cont_without_parent']}")
        # Phase 1I-A scaffold metrics.
        scaffold_executed = row.get("scaffold_executed", 0)
        if scaffold_executed > 0:
            print(f"  scaffold_executed={scaffold_executed}")
            print(f"  scaffold_tokens_generated={row.get('scaffold_tokens_generated', 0)}")
            print(f"  scaffold_proposals_generated={row.get('scaffold_proposals_generated', 0)}")
            print(f"  scaffold_proposals_sent={row.get('scaffold_proposals_sent', 0)}")
            print(f"  scaffold_proposals_received={row.get('scaffold_proposals_received', 0)}")
            print(f"  scaffold_proposals_promoted={row.get('scaffold_proposals_promoted', 0)}")
            print(f"  scaffold_proposals_discarded={row.get('scaffold_proposals_discarded', 0)}")
            print(f"  scaffold_promotion_rate={row.get('scaffold_promotion_rate', 0.0):.3f}")
            print(f"  scaffold_send_tokens={row.get('scaffold_send_tokens', 0)}")
            print(f"  scaffold_recv_tokens={row.get('scaffold_recv_tokens', 0)}")
            print(f"  scaffold_sent_seq_count={row.get('scaffold_sent_seq_count', 0)}")
            print(f"  scaffold_received_seq_count={row.get('scaffold_received_seq_count', 0)}")
            print(f"  scaffold_base_val_failures={row.get('scaffold_base_val_failures', 0)}")
            print(f"  scaffold_buf_max={row.get('scaffold_buf_max', 0)}")
            print(f"  scaffold_unknown={row.get('scaffold_unknown', 0)}")
            print(f"  scaffold_receive_val_failures={row.get('scaffold_receive_val_failures', 0)}")
            print(f"  target_eager_set_executed_count={row.get('target_eager_set_executed_count', 0)}")
            print(f"  continue_set_executed_count={row.get('continue_set_executed_count', 0)}")
        print(f"  suspected_bottleneck={dual_bottleneck(row)}")
        print(f"  warnings={dual_warnings(row)}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python benchmark/summarize_phase1d_profile.py <engine_trace.json> [more_traces...]")
        return 2
    rows = [summarize(Path(arg)) for arg in sys.argv[1:]]
    print_summary(rows)
    print_dual_details(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
