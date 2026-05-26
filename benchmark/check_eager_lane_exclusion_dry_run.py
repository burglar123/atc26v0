#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any


ACTUAL_EAGER_COUNTERS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]

TERMINAL_READY_STATES = {"CONSUMED_APPLIED", "STALE", "EXPIRED", "INVALIDATED"}
ZERO_ONLY_DRY_RUN_FIELDS = [
    "eager_tokens_verify_dry_run",
    "eager_tokens_apply_dry_run",
    "eager_tokens_result_transfer_dry_run",
    "eager_tokens_sync_apply_dry_run",
]


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError("Unsupported trace format")


def is_dual_record(record: dict[str, Any]) -> bool:
    return record.get("execution_mode") == "dual_batch_pearl" and record.get("dual_batch_enabled") is True


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def as_int_set(value: Any) -> set[int]:
    return set(as_int_list(value))


def record_int_set(record: dict[str, Any], key: str, default: set[int] | None = None) -> set[int]:
    if key in record and isinstance(record.get(key), list):
        return as_int_set(record.get(key))
    return set() if default is None else set(default)


def as_str_map(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, str] = {}
    for key, item in value.items():
        try:
            result[int(key)] = str(item)
        except Exception:
            continue
    return result


def as_int_map(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, int] = {}
    for key, item in value.items():
        try:
            result[int(key)] = int(item)
        except Exception:
            continue
    return result


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _normal_metadata_present(record: dict[str, Any], key: str) -> bool:
    return key in record and isinstance(record.get(key), list) and bool(record.get(key))


def _proposal_lifecycle_key(
    record: dict[str, Any],
    proposal_id: int,
    step_by_id: dict[int, int],
) -> tuple[str, int]:
    if proposal_id in step_by_id:
        return ("step", int(step_by_id[proposal_id]))
    step_id = int_value(record.get("step_id"), -1)
    if step_id >= 0:
        return ("step", step_id)
    return ("plan", int_value(record.get("plan_id"), -1))


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_lane_enabled = 0
    ready_created_count = 0
    ready_synced_count = 0
    ready_seen_count = 0
    ready_in_target_count = 0
    ready_in_draft_count = 0
    ready_applied_count = 0
    ready_stale_count = 0
    ready_expired_count = 0
    ready_invalidated_count = 0
    lane_exclusion_applied_decision_count = 0
    excluded_from_actual_count = 0
    target_eager_verify_count = 0
    missing_buffered_allowed_count = 0
    missing_buffered_unexpected_count = 0
    sent_received_mismatch_count = 0
    excluded_expected_count = 0
    actual_eager_counter_rows = 0
    real_target_eager_nonempty_count = 0
    verify_or_apply_rows = 0
    original_draft_sizes: list[int] = []
    adjusted_draft_sizes: list[int] = []
    apply_keys_by_proposal_id: dict[int, set[tuple[str, int]]] = {}
    takeover_keys_by_proposal_id: dict[int, set[tuple[str, int]]] = {}
    apply_steps_by_proposal_id: dict[int, set[int]] = {}
    takeover_steps_by_proposal_id: dict[int, set[int]] = {}
    skip_reason_counts: Counter[str] = Counter()
    stale_reason_counts: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        lane_enabled = bool(record.get("enable_eager_lane_exclusion_dry_run", False))
        lane_active = bool(record.get("eager_lane_exclusion_dry_run_enabled", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        if target_eager:
            real_target_eager_nonempty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty: {sorted(target_eager)}")

        nonzero_actual = [
            field for field in ACTUAL_EAGER_COUNTERS if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must remain zero: {nonzero_actual}")

        verify_dry_run_allowed = (
            record.get("eager_verify_dry_run_source") == "phase1h5e3_takeover_lane"
        )
        forbidden_dry_run_fields = [
            "eager_apply_dry_run_enabled",
            "eager_result_transfer_dry_run_enabled",
            "eager_sync_apply_dry_run_enabled",
        ]
        if not verify_dry_run_allowed:
            forbidden_dry_run_fields.append("eager_verify_dry_run_enabled")
        forbidden_dry_run = [
            field for field in forbidden_dry_run_fields if bool(record.get(field, False))
        ]
        zero_only_fields = [
            field
            for field in ZERO_ONLY_DRY_RUN_FIELDS
            if field != "eager_tokens_verify_dry_run" or not verify_dry_run_allowed
        ]
        nonzero_forbidden_tokens = [
            field for field in zero_only_fields if int_value(record.get(field), 0) != 0
        ]
        if lane_enabled and (forbidden_dry_run or nonzero_forbidden_tokens):
            verify_or_apply_rows += 1
            errors.append(
                f"record[{idx}] Phase 1H lane exclusion must not run forbidden eager verify/apply/result transfer: "
                f"flags={forbidden_dry_run}, tokens={nonzero_forbidden_tokens}"
            )

        if not lane_enabled:
            if lane_active or as_int_set(record.get("lane_excluded_seq_ids")):
                errors.append(f"record[{idx}] lane-exclusion fields populated while lane dry-run disabled")
            continue

        records_with_lane_enabled += 1
        original = as_int_set(record.get("original_draft_home_set")) or as_int_set(record.get("draft_home_set"))
        target_home = as_int_set(record.get("target_home_set"))
        target_normal_verify = record_int_set(record, "target_normal_verify_seq_ids", target_home)
        target_eager_verify = as_int_set(record.get("target_eager_verify_seq_ids_dry_run"))
        target_eager_verify_proposal_ids = as_int_set(record.get("target_eager_verify_proposal_ids_dry_run"))
        target_eager_reason_by_seq = as_str_map(record.get("target_eager_verify_reason_by_seq_id_dry_run"))
        excluded_from_target_normal = as_int_set(
            record.get("excluded_from_target_normal_verify_for_eager_dry_run")
        )
        missing_allowed = as_int_set(record.get("missing_normal_proposal_allowed_seq_ids_dry_run")) or as_int_set(
            record.get("missing_buffered_proposal_allowed_by_eager_seq_ids")
        )
        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        missing_buffered = as_int_set(record.get("missing_buffered_proposal_seq_ids"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        actual = as_int_set(record.get("actual_draft_home_set_for_normal_draft")) or as_int_set(
            record.get("draft_home_set")
        )
        excluded = as_int_set(record.get("excluded_from_actual_draft_home_for_eager")) or as_int_set(
            record.get("lane_excluded_seq_ids")
        )
        expected_seq_ids = as_int_set(record.get("normal_proposal_expected_seq_ids_after_lane_exclusion")) or actual
        adjusted_expected = as_int_set(record.get("adjusted_normal_proposal_expected_seq_ids")) or actual
        sent_seq_ids = as_int_set(record.get("normal_proposal_sent_seq_ids_after_lane_exclusion"))
        received_seq_ids = as_int_set(record.get("normal_proposal_received_seq_ids_after_lane_exclusion"))
        fallback_same_batch = bool(record.get("fallback_same_batch", False)) or (
            record.get("plan_phase") == "fallback"
            and bool(target_home)
            and target_home == actual
            and target_normal_verify == target_home
        )
        fallback_pending_receive = as_int_set(record.get("fallback_pending_receive_seq_ids"))
        if fallback_same_batch and not fallback_pending_receive:
            fallback_pending_receive = missing_buffered & target_normal_verify
        fallback_received = as_int_set(record.get("fallback_received_seq_ids"))
        fallback_missing_after_receive = as_int_set(record.get("fallback_missing_after_receive_seq_ids"))
        effective_missing_unexpected = (
            missing_unexpected - fallback_pending_receive
            if fallback_same_batch
            else set(missing_unexpected)
        )
        original_draft_sizes.append(len(original))
        adjusted_draft_sizes.append(len(actual))

        pending_decisions = as_int_set(record.get("pending_lane_exclusion_decision_ids_before_plan"))
        applied_legacy_decisions = as_int_set(record.get("applied_lane_exclusion_decision_ids"))
        if pending_decisions or applied_legacy_decisions:
            errors.append(
                f"record[{idx}] 1H-5e3 must not carry pending LaneExclusionDecision state: "
                f"pending={sorted(pending_decisions)}, applied={sorted(applied_legacy_decisions)}"
            )

        created_ids = as_int_set(record.get("ready_eager_proposal_created_ids"))
        synced_ids = as_int_set(record.get("ready_eager_proposal_synced_ids")) or (
            as_int_set(record.get("ready_eager_proposal_sent_ids"))
            | as_int_set(record.get("ready_eager_proposal_received_ids"))
        )
        registry_ids = as_int_set(record.get("ready_eager_proposal_registry_ids_before_plan"))
        seen_ids = as_int_set(record.get("ready_eager_proposal_seen_by_scheduler_ids"))
        in_target_ids = as_int_set(record.get("ready_eager_proposal_in_target_home_ids"))
        in_draft_ids = as_int_set(record.get("ready_eager_proposal_in_draft_home_ids"))
        applied_ids = as_int_set(record.get("ready_eager_proposal_applied_ids")) or as_int_set(
            record.get("lane_exclusion_applied_proposal_ids")
        )
        stale_ids = as_int_set(record.get("ready_eager_proposal_stale_ids"))
        expired_ids = as_int_set(record.get("ready_eager_proposal_expired_ids"))
        invalidated_ids = as_int_set(record.get("ready_eager_proposal_invalidated_ids"))
        terminal_ids = applied_ids | stale_ids | expired_ids | invalidated_ids
        state_by_id = as_str_map(record.get("ready_eager_proposal_state_by_id"))
        skip_reason_by_id = as_str_map(record.get("ready_eager_proposal_skip_reason_by_id"))
        stale_reason_by_id = as_str_map(record.get("ready_eager_proposal_stale_reason_by_id"))
        apply_step_by_id = as_int_map(record.get("ready_eager_proposal_apply_step_by_id"))
        takeover_step_by_id = as_int_map(record.get("ready_eager_proposal_takeover_routed_step_by_id"))
        takeover_routed_ids = as_int_set(record.get("ready_eager_proposal_takeover_routed_ids"))
        pending_takeover_ids = (
            as_int_set(record.get("ready_eager_proposal_pending_takeover_ids"))
            | as_int_set(record.get("ready_eager_proposal_pending_takeover_proposal_ids"))
        )
        lane_applied_ids = as_int_set(record.get("lane_exclusion_applied_proposal_ids"))
        lane_applied_seq_ids = as_int_set(record.get("lane_exclusion_applied_seq_ids"))
        apply_reason_by_id = as_str_map(record.get("lane_exclusion_apply_reason_by_proposal_id"))

        ready_created_count += len(created_ids)
        ready_synced_count += len(synced_ids)
        ready_seen_count += len(seen_ids)
        ready_in_target_count += len(in_target_ids)
        ready_in_draft_count += len(in_draft_ids)
        ready_applied_count += len(applied_ids)
        ready_stale_count += len(stale_ids)
        ready_expired_count += len(expired_ids)
        ready_invalidated_count += len(invalidated_ids)
        lane_exclusion_applied_decision_count += len(lane_applied_ids)
        excluded_from_actual_count += len(excluded)
        target_eager_verify_count += len(target_eager_verify)
        missing_buffered_allowed_count += len(missing_allowed)
        missing_buffered_unexpected_count += len(effective_missing_unexpected)
        skip_reason_counts.update(skip_reason_by_id.values())
        stale_reason_counts.update(stale_reason_by_id.values())

        if target_normal_verify - target_home:
            errors.append(
                f"record[{idx}] target_normal_verify_seq_ids must be a subset of target_home_set: "
                f"extra={sorted(target_normal_verify - target_home)}"
            )
        if target_normal_verify & target_eager_verify:
            errors.append(
                f"record[{idx}] target normal verify overlaps eager takeover seqs: "
                f"{sorted(target_normal_verify & target_eager_verify)}"
            )
        if target_eager_verify - target_home:
            errors.append(
                f"record[{idx}] target eager takeover seqs must be in current target_home_set: "
                f"extra={sorted(target_eager_verify - target_home)}"
            )
        expected_target_normal = target_home - target_eager_verify
        if target_normal_verify != expected_target_normal:
            errors.append(
                f"record[{idx}] target_normal_verify_seq_ids must equal target_home minus eager takeover: "
                f"target_normal={sorted(target_normal_verify)}, expected={sorted(expected_target_normal)}"
            )
        if target_eager_verify and len(target_eager_verify_proposal_ids) != len(target_eager_verify):
            errors.append(
                f"record[{idx}] target eager verify seqs need matching proposal ids: "
                f"seqs={sorted(target_eager_verify)}, proposals={sorted(target_eager_verify_proposal_ids)}"
            )
        if target_eager_verify and not target_eager_verify <= set(target_eager_reason_by_seq):
            errors.append(
                f"record[{idx}] target eager verify seqs missing reason mapping: "
                f"missing={sorted(target_eager_verify - set(target_eager_reason_by_seq))}"
            )
        if draft_eager & target_eager_verify:
            errors.append(
                f"record[{idx}] target_eager_verify_seq_ids_dry_run must not be treated as draft_eager_set: "
                f"overlap={sorted(draft_eager & target_eager_verify)}"
            )
        if missing_allowed - target_eager_verify:
            errors.append(
                f"record[{idx}] allowed missing normal proposals must be eager takeover seqs: "
                f"extra={sorted(missing_allowed - target_eager_verify)}"
            )
        if effective_missing_unexpected:
            errors.append(
                f"record[{idx}] unexpected missing buffered normal proposals: "
                f"{sorted(effective_missing_unexpected)}"
            )
        expected_missing_split = missing_allowed | effective_missing_unexpected | fallback_pending_receive
        if missing_buffered and missing_buffered != expected_missing_split:
            errors.append(
                f"record[{idx}] missing buffered proposal split is inconsistent: "
                f"missing={sorted(missing_buffered)}, allowed={sorted(missing_allowed)}, "
                f"fallback_pending={sorted(fallback_pending_receive)}, "
                f"unexpected={sorted(effective_missing_unexpected)}"
            )
        if missing_allowed and not bool(record.get("missing_normal_proposal_allowed_by_eager_dry_run", False)):
            errors.append(f"record[{idx}] allowed eager-takeover missing proposals must set allow flag")
        if fallback_same_batch:
            fallback_coverage = fallback_received or received_seq_ids
            if fallback_missing_after_receive:
                errors.append(
                    f"record[{idx}] fallback same-batch missing proposals after receive: "
                    f"{sorted(fallback_missing_after_receive)}"
                )
            if fallback_coverage and not target_normal_verify <= fallback_coverage:
                errors.append(
                    f"record[{idx}] fallback same-batch received proposals do not cover target normal verify: "
                    f"target_normal={sorted(target_normal_verify)}, received={sorted(fallback_coverage)}"
                )

        if seen_ids and not bool(record.get("ready_eager_proposals_synchronized_before_plan", False)):
            errors.append(f"record[{idx}] scheduler saw ready proposals without pre-plan sync")
        if seen_ids and not seen_ids <= registry_ids:
            errors.append(
                f"record[{idx}] seen ready proposals must come from registry ids before plan: "
                f"seen={sorted(seen_ids)}, registry={sorted(registry_ids)}"
            )
        if in_target_ids - seen_ids:
            errors.append(f"record[{idx}] in-target ready proposal ids were not seen by scheduler")
        if in_draft_ids - seen_ids:
            errors.append(f"record[{idx}] in-draft ready proposal ids were not seen by scheduler")
        if applied_ids - in_draft_ids:
            errors.append(f"record[{idx}] applied proposals must be seen in candidate draft home")
        if applied_ids & (stale_ids | expired_ids | invalidated_ids):
            errors.append(f"record[{idx}] terminal stale/expired/invalidated proposals were also applied")

        for proposal_id in applied_ids:
            apply_key = _proposal_lifecycle_key(record, proposal_id, apply_step_by_id)
            apply_keys_by_proposal_id.setdefault(proposal_id, set()).add(apply_key)
            if apply_key[0] == "step" and apply_key[1] >= 0:
                apply_steps_by_proposal_id.setdefault(proposal_id, set()).add(apply_key[1])
            if state_by_id.get(proposal_id) != "CONSUMED_APPLIED":
                errors.append(
                    f"record[{idx}] applied proposal {proposal_id} must have state CONSUMED_APPLIED, "
                    f"got {state_by_id.get(proposal_id)!r}"
                )
            if apply_reason_by_id.get(proposal_id) != "ready_eager_proposal_available_for_draft_home":
                errors.append(f"record[{idx}] applied proposal {proposal_id} has missing/bad apply reason")
        for proposal_id in target_eager_verify_proposal_ids | takeover_routed_ids:
            takeover_key = _proposal_lifecycle_key(record, proposal_id, takeover_step_by_id)
            takeover_keys_by_proposal_id.setdefault(proposal_id, set()).add(takeover_key)
            if takeover_key[0] == "step" and takeover_key[1] >= 0:
                takeover_steps_by_proposal_id.setdefault(proposal_id, set()).add(takeover_key[1])
        for proposal_id in terminal_ids:
            if state_by_id.get(proposal_id) == "READY":
                errors.append(f"record[{idx}] proposal {proposal_id} is both READY and terminal")
        for proposal_id in in_target_ids:
            if proposal_id not in terminal_ids:
                if state_by_id.get(proposal_id) != "READY":
                    errors.append(f"record[{idx}] in-target proposal {proposal_id} should stay READY")
                if skip_reason_by_id.get(proposal_id) != "still_in_target_home":
                    errors.append(f"record[{idx}] in-target proposal {proposal_id} missing still_in_target_home")

        if lane_applied_ids:
            if lane_applied_ids != applied_ids:
                errors.append(
                    f"record[{idx}] lane applied ids must match ready applied ids: "
                    f"lane={sorted(lane_applied_ids)}, ready={sorted(applied_ids)}"
                )
            if excluded != lane_applied_seq_ids:
                errors.append(
                    f"record[{idx}] excluded seq ids must match lane applied seq ids: "
                    f"excluded={sorted(excluded)}, applied_seq={sorted(lane_applied_seq_ids)}"
                )
            expected_actual = original - excluded
            if actual != expected_actual:
                errors.append(
                    f"record[{idx}] actual draft home must equal original minus excluded: "
                    f"actual={sorted(actual)}, expected={sorted(expected_actual)}"
                )
            if expected_seq_ids != actual or adjusted_expected != actual:
                errors.append(f"record[{idx}] normal proposal expected seq ids must use adjusted set")
            if not bool(record.get("lane_exclusion_dry_run_done", False)):
                errors.append(f"record[{idx}] applied lane exclusion must mark lane_exclusion_dry_run_done")
            excluded_in_target = excluded & target_home
            excluded_not_in_target = excluded - target_home
            if not excluded_in_target <= target_eager_verify:
                errors.append(
                    f"record[{idx}] excluded seqs that are also in target_home must route to target eager verify: "
                    f"excluded_in_target={sorted(excluded_in_target)}, "
                    f"target_eager_verify={sorted(target_eager_verify)}"
                )
            if excluded_not_in_target & target_eager_verify:
                errors.append(
                    f"record[{idx}] draft-home-only exclusions must not route to target eager verify yet: "
                    f"excluded_not_in_target={sorted(excluded_not_in_target)}, "
                    f"target_eager_verify={sorted(target_eager_verify)}"
                )
            if not excluded_in_target <= excluded_from_target_normal:
                errors.append(
                    f"record[{idx}] target-home excluded seqs must be traced as excluded from target normal verify: "
                    f"excluded_in_target={sorted(excluded_in_target)}, traced={sorted(excluded_from_target_normal)}"
                )
            if excluded_not_in_target and not (applied_ids - takeover_routed_ids) <= pending_takeover_ids:
                errors.append(
                    f"record[{idx}] draft-home-only applied proposals must remain pending takeover: "
                    f"applied={sorted(applied_ids)}, routed={sorted(takeover_routed_ids)}, "
                    f"pending={sorted(pending_takeover_ids)}"
                )
        else:
            if excluded:
                errors.append(f"record[{idx}] excluded seq ids require an applied ready proposal")
            if actual != original:
                errors.append(
                    f"record[{idx}] no applied ready proposal must leave draft home unchanged: "
                    f"actual={sorted(actual)}, original={sorted(original)}"
                )

        if _normal_metadata_present(record, "normal_proposal_sent_seq_ids_after_lane_exclusion"):
            if sent_seq_ids != actual:
                sent_received_mismatch_count += 1
                errors.append(
                    f"record[{idx}] sent normal proposal seq ids must match adjusted set: "
                    f"sent={sorted(sent_seq_ids)}, expected={sorted(actual)}"
                )
        if _normal_metadata_present(record, "normal_proposal_received_seq_ids_after_lane_exclusion"):
            if received_seq_ids != actual:
                sent_received_mismatch_count += 1
                errors.append(
                    f"record[{idx}] received normal proposal seq ids must match adjusted set: "
                    f"received={sorted(received_seq_ids)}, expected={sorted(actual)}"
                )
        if sent_seq_ids and received_seq_ids and sent_seq_ids != received_seq_ids:
            sent_received_mismatch_count += 1
            errors.append(
                f"record[{idx}] sent/received normal proposal metadata mismatch: "
                f"sent={sorted(sent_seq_ids)}, received={sorted(received_seq_ids)}"
            )
        if excluded & expected_seq_ids:
            excluded_expected_count += len(excluded & expected_seq_ids)
            errors.append(f"record[{idx}] excluded seqs were still expected by normal proposal receive")
        if excluded & sent_seq_ids or excluded & received_seq_ids:
            errors.append(f"record[{idx}] excluded seqs appeared in normal proposal metadata")

        if in_draft_ids and not applied_ids:
            accounted = terminal_ids | set(skip_reason_by_id)
            if not in_draft_ids <= accounted:
                errors.append(
                    f"record[{idx}] in-draft ready proposals were neither applied nor explicitly explained: "
                    f"missing={sorted(in_draft_ids - accounted)}"
                )

    repeated_applied = [
        proposal_id
        for proposal_id, keys in apply_keys_by_proposal_id.items()
        if len(keys) > 1
    ]
    if repeated_applied:
        errors.append(f"ready proposals applied more than once: {sorted(repeated_applied)}")
    repeated_takeover = [
        proposal_id
        for proposal_id, keys in takeover_keys_by_proposal_id.items()
        if len(keys) > 1
    ]
    if repeated_takeover:
        errors.append(f"ready proposals takeover-routed more than once: {sorted(repeated_takeover)}")
    takeover_before_apply = []
    for proposal_id, takeover_steps in takeover_steps_by_proposal_id.items():
        apply_steps = apply_steps_by_proposal_id.get(proposal_id)
        if apply_steps and min(takeover_steps) < min(apply_steps):
            takeover_before_apply.append(proposal_id)
    if takeover_before_apply:
        errors.append(
            f"ready proposals takeover-routed before lane-exclusion apply: "
            f"{sorted(takeover_before_apply)}"
        )

    if records_with_lane_enabled:
        if ready_created_count == 0:
            errors.append("ready_eager_proposal_created_count must be > 0 for lane-exclusion validation")
        if ready_seen_count == 0:
            errors.append("ready_eager_proposal_seen_by_scheduler_count must be > 0")
        if ready_applied_count and not excluded_from_actual_count:
            errors.append("applied ready proposals must exclude at least one seq from actual draft home")
        if ready_applied_count and adjusted_draft_sizes and original_draft_sizes:
            if mean(adjusted_draft_sizes) >= mean(original_draft_sizes):
                errors.append(
                    "adjusted_draft_home_size_mean must be smaller than original_draft_home_size_mean "
                    "when proposals are applied"
                )

    coverage_caveat = ""
    if records_with_lane_enabled and ready_created_count and ready_seen_count and ready_applied_count == 0:
        explained_count = ready_stale_count + ready_expired_count + ready_invalidated_count + sum(skip_reason_counts.values())
        if explained_count > 0:
            coverage_caveat = (
                "pass with coverage caveat: ready proposals were seen, but none were cleanly applicable"
            )
        else:
            errors.append("ready proposals were seen but none applied and no stale/expired/skip reason was traced")

    summary = {
        "total_trace_records": len(records),
        "records_with_lane_exclusion_enabled": records_with_lane_enabled,
        "ready_eager_proposal_created_count": ready_created_count,
        "ready_eager_proposal_synced_count": ready_synced_count,
        "ready_eager_proposal_seen_by_scheduler_count": ready_seen_count,
        "ready_eager_proposal_in_target_home_count": ready_in_target_count,
        "ready_eager_proposal_in_draft_home_count": ready_in_draft_count,
        "ready_eager_proposal_applied_count": ready_applied_count,
        "ready_eager_proposal_stale_count": ready_stale_count,
        "ready_eager_proposal_expired_count": ready_expired_count,
        "ready_eager_proposal_invalidated_count": ready_invalidated_count,
        "lane_exclusion_applied_decision_count": lane_exclusion_applied_decision_count,
        "excluded_from_actual_draft_home_count": excluded_from_actual_count,
        "target_eager_verify_seq_ids_dry_run_count": target_eager_verify_count,
        "missing_buffered_proposal_allowed_by_eager_count": missing_buffered_allowed_count,
        "missing_buffered_proposal_unexpected_count": missing_buffered_unexpected_count,
        "repeated_apply_proposal_ids": sorted(repeated_applied),
        "repeated_takeover_proposal_ids": sorted(repeated_takeover),
        "original_draft_home_size_mean": mean(original_draft_sizes) if original_draft_sizes else 0.0,
        "adjusted_draft_home_size_mean": mean(adjusted_draft_sizes) if adjusted_draft_sizes else 0.0,
        "normal_proposal_sent_received_mismatch_count": sent_received_mismatch_count,
        "excluded_seqs_in_normal_expected_count": excluded_expected_count,
        "actual_eager_verified_counter_rows": actual_eager_counter_rows,
        "real_target_eager_nonempty_count": real_target_eager_nonempty_count,
        "forbidden_verify_apply_rows": verify_or_apply_rows,
        "ready_eager_proposal_skip_reason_counts": dict(skip_reason_counts),
        "ready_eager_proposal_stale_reason_counts": dict(stale_reason_counts),
        "coverage_caveat": coverage_caveat,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        print(f"{key}={value}")


def base_record(*, lane_enabled: bool = True) -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "step_id": 12,
        "normal_gamma": 4,
        "enable_eager_lane_exclusion_dry_run": lane_enabled,
        "eager_lane_exclusion_dry_run_enabled": lane_enabled,
        "target_home_set": [0, 2],
        "target_normal_verify_seq_ids": [0, 2],
        "target_eager_verify_seq_ids_dry_run": [],
        "target_eager_verify_proposal_ids_dry_run": [],
        "target_eager_verify_reason_by_seq_id_dry_run": {},
        "excluded_from_target_normal_verify_for_eager_dry_run": [],
        "missing_normal_proposal_allowed_by_eager_dry_run": False,
        "missing_normal_proposal_allowed_seq_ids_dry_run": [],
        "raw_target_home_set_for_normal_verify": [0, 2],
        "missing_buffered_proposal_seq_ids": [],
        "missing_buffered_proposal_allowed_by_eager_seq_ids": [],
        "missing_buffered_proposal_unexpected_seq_ids": [],
        "fallback_same_batch": False,
        "fallback_pending_receive_seq_ids": [],
        "fallback_received_seq_ids": [],
        "fallback_missing_after_receive_seq_ids": [],
        "draft_eager_set": [],
        "draft_home_set": [1, 3],
        "original_draft_home_set": [1, 3],
        "actual_draft_home_set_for_normal_draft": [1, 3],
        "target_eager_set": [],
        "target_eager_set_dry_run": [],
        "scheduled_target_eager_set_dry_run": [],
        "excluded_from_actual_draft_home_for_eager": [],
        "excluded_from_draft_home_for_eager_dry_run": [],
        "lane_excluded_seq_ids": [],
        "lane_exclusion_decision_available_before_draft": False,
        "lane_exclusion_deferred_until_next_step": False,
        "lane_exclusion_defer_reason": None,
        "pending_lane_exclusion_decision_ids_before_plan": [],
        "applied_lane_exclusion_decision_ids": [],
        "stale_lane_exclusion_decision_ids": [],
        "expired_lane_exclusion_decision_ids": [],
        "active_pending_lane_exclusion_decision_ids": [],
        "terminal_lane_exclusion_decision_ids": [],
        "touched_lane_exclusion_decision_ids": [],
        "lane_exclusion_decisions_synchronized_before_plan": lane_enabled,
        "lane_exclusion_decision_transfer_called": lane_enabled,
        "lane_exclusion_decision_num_decisions": 0,
        "lane_exclusion_decision_payload_len": 0,
        "lane_exclusion_decision_zero_decision": True,
        "lane_exclusion_dry_run_done": False,
        "lane_exclusion_dry_run_done_proposal_ids": [],
        "lane_exclusion_dry_run_done_seq_ids": [],
        "normal_proposal_expected_seq_ids_after_lane_exclusion": [1, 3],
        "normal_proposal_sent_seq_ids_after_lane_exclusion": [1, 3],
        "normal_proposal_received_seq_ids_after_lane_exclusion": [1, 3],
        "adjusted_normal_proposal_expected_seq_ids": [1, 3],
        "ready_eager_proposal_created_ids": [],
        "ready_eager_proposal_created_seq_ids": [],
        "ready_eager_proposal_synced_ids": [],
        "ready_eager_proposal_registry_ids_before_plan": [],
        "ready_eager_proposal_seen_by_scheduler_ids": [],
        "ready_eager_proposal_in_target_home_ids": [],
        "ready_eager_proposal_in_draft_home_ids": [],
        "ready_eager_proposal_applied_ids": [],
        "ready_eager_proposal_stale_ids": [],
        "ready_eager_proposal_expired_ids": [],
        "ready_eager_proposal_invalidated_ids": [],
        "ready_eager_proposal_state_by_id": {},
        "ready_eager_proposal_skip_reason_by_id": {},
        "ready_eager_proposal_stale_reason_by_id": {},
        "ready_eager_proposal_age_by_id": {},
        "ready_eager_proposal_seq_id_by_id": {},
        "ready_eager_proposal_base_len_by_id": {},
        "ready_eager_proposal_current_len_by_id": {},
        "ready_eager_proposal_current_pre_verify_by_id": {},
        "ready_eager_proposal_current_status_by_id": {},
        "ready_eager_proposal_apply_step_by_id": {},
        "ready_eager_proposal_takeover_routed_step_by_id": {},
        "ready_eager_proposal_takeover_routed_ids": [],
        "ready_eager_proposal_takeover_routed_seq_ids": [],
        "ready_eager_proposal_pending_takeover_ids": [],
        "ready_eager_proposal_pending_takeover_proposal_ids": [],
        "ready_eager_proposal_pending_takeover_seq_ids": [],
        "ready_eager_proposal_takeover_waiting_for_target_home_ids": [],
        "ready_eager_proposal_already_takeover_routed_ids": [],
        "repeated_takeover_proposal_ids": [],
        "ready_eager_proposals_synchronized_before_plan": lane_enabled,
        "ready_eager_proposal_transfer_called": lane_enabled,
        "ready_eager_proposal_sent_ids": [],
        "ready_eager_proposal_received_ids": [],
        "ready_eager_proposal_num_proposals": 0,
        "ready_eager_proposal_payload_len": 0,
        "ready_eager_proposal_zero_proposal": True,
        "lane_exclusion_applied_proposal_ids": [],
        "lane_exclusion_applied_seq_ids": [],
        "lane_exclusion_apply_reason_by_proposal_id": {},
    }
    for field in ACTUAL_EAGER_COUNTERS + ZERO_ONLY_DRY_RUN_FIELDS:
        record[field] = 0
    for field in (
        "eager_verify_dry_run_enabled",
        "eager_apply_dry_run_enabled",
        "eager_result_transfer_dry_run_enabled",
        "eager_sync_apply_dry_run_enabled",
    ):
        record[field] = False
    return record


def creation_record(proposal_id: int = 200, seq_id: int = 3) -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "ready_eager_proposal_created_ids": [proposal_id],
            "ready_eager_proposal_created_seq_ids": [seq_id],
            "lane_exclusion_deferred_until_next_step": True,
            "lane_exclusion_defer_reason": "ready_eager_proposal_created_after_plan",
        }
    )
    return record


def target_home_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "target_home_set": [2],
            "target_normal_verify_seq_ids": [2],
            "raw_target_home_set_for_normal_verify": [2],
            "draft_home_set": [3],
            "original_draft_home_set": [3],
            "actual_draft_home_set_for_normal_draft": [3],
            "normal_proposal_expected_seq_ids_after_lane_exclusion": [3],
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [3],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [3],
            "adjusted_normal_proposal_expected_seq_ids": [3],
            "ready_eager_proposal_synced_ids": [201],
            "ready_eager_proposal_registry_ids_before_plan": [201],
            "ready_eager_proposal_seen_by_scheduler_ids": [201],
            "ready_eager_proposal_in_target_home_ids": [201],
            "ready_eager_proposal_state_by_id": {"201": "READY"},
            "ready_eager_proposal_skip_reason_by_id": {"201": "still_in_target_home"},
        }
    )
    return record


def applied_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "target_home_set": [0],
            "target_normal_verify_seq_ids": [0],
            "raw_target_home_set_for_normal_verify": [0],
            "target_eager_verify_seq_ids_dry_run": [],
            "target_eager_verify_proposal_ids_dry_run": [],
            "target_eager_verify_reason_by_seq_id_dry_run": {},
            "excluded_from_target_normal_verify_for_eager_dry_run": [],
            "draft_home_set": [5],
            "original_draft_home_set": [3, 5],
            "actual_draft_home_set_for_normal_draft": [5],
            "excluded_from_actual_draft_home_for_eager": [3],
            "lane_excluded_seq_ids": [3],
            "lane_exclusion_decision_available_before_draft": True,
            "lane_exclusion_dry_run_done": True,
            "lane_exclusion_dry_run_done_proposal_ids": [202],
            "lane_exclusion_dry_run_done_seq_ids": [3],
            "normal_proposal_expected_seq_ids_after_lane_exclusion": [5],
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [5],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [5],
            "adjusted_normal_proposal_expected_seq_ids": [5],
            "ready_eager_proposal_synced_ids": [202],
            "ready_eager_proposal_registry_ids_before_plan": [202],
            "ready_eager_proposal_seen_by_scheduler_ids": [202],
            "ready_eager_proposal_in_draft_home_ids": [202],
            "ready_eager_proposal_applied_ids": [202],
            "ready_eager_proposal_state_by_id": {"202": "CONSUMED_APPLIED"},
            "ready_eager_proposal_age_by_id": {"202": 1},
            "ready_eager_proposal_seq_id_by_id": {"202": 3},
            "ready_eager_proposal_base_len_by_id": {"202": 8},
            "ready_eager_proposal_current_len_by_id": {"202": 8},
            "ready_eager_proposal_current_pre_verify_by_id": {"202": False},
            "ready_eager_proposal_current_status_by_id": {"202": "RUNNING"},
            "ready_eager_proposal_apply_step_by_id": {"202": 12},
            "ready_eager_proposal_takeover_routed_step_by_id": {},
            "ready_eager_proposal_takeover_routed_ids": [],
            "ready_eager_proposal_takeover_routed_seq_ids": [],
            "ready_eager_proposal_pending_takeover_ids": [202],
            "ready_eager_proposal_pending_takeover_proposal_ids": [202],
            "ready_eager_proposal_pending_takeover_seq_ids": [3],
            "ready_eager_proposal_takeover_waiting_for_target_home_ids": [202],
            "lane_exclusion_applied_proposal_ids": [202],
            "lane_exclusion_applied_seq_ids": [3],
            "lane_exclusion_apply_reason_by_proposal_id": {
                "202": "ready_eager_proposal_available_for_draft_home"
            },
        }
    )
    return record


def takeover_target_verify_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "step_id": 13,
            "target_home_set": [3, 5],
            "target_normal_verify_seq_ids": [5],
            "raw_target_home_set_for_normal_verify": [3, 5],
            "target_eager_verify_seq_ids_dry_run": [3],
            "target_eager_verify_proposal_ids_dry_run": [202],
            "target_eager_verify_reason_by_seq_id_dry_run": {
                "3": "ready_eager_takeover_for_target_normal_verify"
            },
            "excluded_from_target_normal_verify_for_eager_dry_run": [3],
            "missing_normal_proposal_allowed_by_eager_dry_run": True,
            "missing_normal_proposal_allowed_seq_ids_dry_run": [3],
            "missing_buffered_proposal_seq_ids": [3],
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [3],
            "missing_buffered_proposal_unexpected_seq_ids": [],
            "draft_home_set": [4, 6],
            "original_draft_home_set": [4, 6],
            "actual_draft_home_set_for_normal_draft": [4, 6],
            "normal_proposal_expected_seq_ids_after_lane_exclusion": [4, 6],
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [4, 6],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [4, 6],
            "adjusted_normal_proposal_expected_seq_ids": [4, 6],
            "ready_eager_proposal_state_by_id": {"202": "CONSUMED_APPLIED"},
            "ready_eager_proposal_apply_step_by_id": {"202": 12},
            "ready_eager_proposal_takeover_routed_step_by_id": {"202": 13},
            "ready_eager_proposal_takeover_routed_ids": [202],
            "ready_eager_proposal_takeover_routed_seq_ids": [3],
            "enable_eager_verify_dry_run": True,
            "eager_verify_dry_run_enabled": True,
            "eager_verify_dry_run_source": "phase1h5e3_takeover_lane",
            "eager_tokens_verify_dry_run": 4,
        }
    )
    return record


def stale_pre_verify_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "ready_eager_proposal_synced_ids": [203],
            "ready_eager_proposal_registry_ids_before_plan": [203],
            "ready_eager_proposal_seen_by_scheduler_ids": [203],
            "ready_eager_proposal_in_draft_home_ids": [203],
            "ready_eager_proposal_stale_ids": [203],
            "ready_eager_proposal_state_by_id": {"203": "STALE"},
            "ready_eager_proposal_stale_reason_by_id": {"203": "seq_returned_pre_verify"},
            "ready_eager_proposal_current_pre_verify_by_id": {"203": True},
        }
    )
    return record


def stale_base_overshot_record() -> dict[str, Any]:
    record = stale_pre_verify_record()
    record.update(
        {
            "ready_eager_proposal_synced_ids": [204],
            "ready_eager_proposal_registry_ids_before_plan": [204],
            "ready_eager_proposal_seen_by_scheduler_ids": [204],
            "ready_eager_proposal_in_draft_home_ids": [204],
            "ready_eager_proposal_stale_ids": [204],
            "ready_eager_proposal_state_by_id": {"204": "STALE"},
            "ready_eager_proposal_stale_reason_by_id": {"204": "base_overshot"},
            "ready_eager_proposal_base_len_by_id": {"204": 8},
            "ready_eager_proposal_current_len_by_id": {"204": 9},
            "ready_eager_proposal_current_pre_verify_by_id": {"204": False},
        }
    )
    return record


def expired_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "ready_eager_proposal_synced_ids": [205],
            "ready_eager_proposal_registry_ids_before_plan": [205],
            "ready_eager_proposal_seen_by_scheduler_ids": [205],
            "ready_eager_proposal_in_draft_home_ids": [205],
            "ready_eager_proposal_expired_ids": [205],
            "ready_eager_proposal_state_by_id": {"205": "EXPIRED"},
            "ready_eager_proposal_stale_reason_by_id": {"205": "expired"},
            "ready_eager_proposal_age_by_id": {"205": 4},
        }
    )
    return record


def fallback_same_batch_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "plan_phase": "fallback",
            "target_home_set": [6, 20],
            "target_normal_verify_seq_ids": [6, 20],
            "raw_target_home_set_for_normal_verify": [6, 20],
            "draft_home_set": [6, 20],
            "original_draft_home_set": [6, 20],
            "actual_draft_home_set_for_normal_draft": [6, 20],
            "normal_proposal_expected_seq_ids_after_lane_exclusion": [6, 20],
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [6, 20],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [6, 20],
            "adjusted_normal_proposal_expected_seq_ids": [6, 20],
            "missing_buffered_proposal_seq_ids": [6, 20],
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [],
            "missing_buffered_proposal_unexpected_seq_ids": [],
            "fallback_same_batch": True,
            "fallback_pending_receive_seq_ids": [6, 20],
            "fallback_received_seq_ids": [6, 20],
            "fallback_missing_after_receive_seq_ids": [],
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid = [
        base_record(lane_enabled=False),
        creation_record(),
        target_home_record(),
        applied_record(),
        deepcopy(applied_record()),
        takeover_target_verify_record(),
        fallback_same_batch_record(),
        stale_pre_verify_record(),
        stale_base_overshot_record(),
        expired_record(),
    ]
    errors, _ = validate_records(valid)
    assert not errors, f"valid synthetic lane-exclusion records failed: {errors}"

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["normal_proposal_received_seq_ids_after_lane_exclusion"] = [3, 5]
    errors, _ = validate_records(invalid)
    assert any("received normal proposal seq ids" in error for error in errors), (
        "checker missed target receive using original draft set"
    )

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["normal_proposal_expected_seq_ids_after_lane_exclusion"] = [3, 5]
    errors, _ = validate_records(invalid)
    assert any("excluded seqs were still expected" in error for error in errors), (
        "checker missed excluded seq expected by receive"
    )

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["ready_eager_proposal_state_by_id"] = {"202": "READY"}
    errors, _ = validate_records(invalid)
    assert any("CONSUMED_APPLIED" in error for error in errors), (
        "checker missed applied proposal with READY state"
    )

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["ready_eager_proposal_stale_ids"] = [202]
    errors, _ = validate_records(invalid)
    assert any("were also applied" in error for error in errors), "checker missed stale+applied proposal"

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["pending_lane_exclusion_decision_ids_before_plan"] = [202]
    errors, _ = validate_records(invalid)
    assert any("must not carry pending LaneExclusionDecision" in error for error in errors), (
        "checker missed legacy pending decision state"
    )

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = [creation_record(), deepcopy(applied_record())]
    invalid[1]["eager_verify_dry_run_enabled"] = True
    errors, _ = validate_records(invalid)
    assert any("forbidden eager verify/apply" in error for error in errors), (
        "checker missed forbidden eager verify dry-run"
    )

    invalid = [creation_record(), deepcopy(takeover_target_verify_record())]
    invalid[1]["missing_buffered_proposal_unexpected_seq_ids"] = [5]
    invalid[1]["missing_buffered_proposal_seq_ids"] = [3, 5]
    errors, _ = validate_records(invalid)
    assert any("unexpected missing buffered normal proposals" in error for error in errors), (
        "checker missed unexpected missing normal proposal"
    )

    invalid = [creation_record(), deepcopy(takeover_target_verify_record())]
    invalid[1]["draft_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("must not be treated as draft_eager_set" in error for error in errors), (
        "checker missed target eager takeover mixed with draft eager candidates"
    )

    invalid = [creation_record(), deepcopy(fallback_same_batch_record())]
    invalid[1]["fallback_received_seq_ids"] = [6]
    invalid[1]["fallback_missing_after_receive_seq_ids"] = [20]
    invalid[1]["normal_proposal_received_seq_ids_after_lane_exclusion"] = [6]
    errors, _ = validate_records(invalid)
    assert any("fallback same-batch missing proposals after receive" in error for error in errors), (
        "checker missed fallback same-batch receive coverage failure"
    )

    invalid = [creation_record(), applied_record(), deepcopy(applied_record())]
    invalid[2]["step_id"] = 13
    invalid[2]["ready_eager_proposal_apply_step_by_id"] = {"202": 13}
    invalid[2]["ready_eager_proposal_takeover_routed_step_by_id"] = {"202": 13}
    errors, _ = validate_records(invalid)
    assert any("applied more than once" in error for error in errors), (
        "checker missed repeated proposal application"
    )

    invalid = [creation_record(), takeover_target_verify_record(), deepcopy(takeover_target_verify_record())]
    invalid[2]["step_id"] = 14
    invalid[2]["ready_eager_proposal_takeover_routed_step_by_id"] = {"202": 14}
    errors, _ = validate_records(invalid)
    assert any("takeover-routed more than once" in error for error in errors), (
        "checker missed repeated proposal takeover routing"
    )

    print("Synthetic eager lane-exclusion dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5e3 eager lane-exclusion dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        if args.trace is None:
            return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nEager lane-exclusion dry-run trace check passed.")
    if summary.get("coverage_caveat"):
        print(summary["coverage_caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
