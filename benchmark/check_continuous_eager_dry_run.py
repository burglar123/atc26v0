#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_eager_commit_ready_only import validate_records as validate_commit_records
from benchmark.check_eager_performance_accounting import aggregate_performance_accounting


CONTINUOUS_SOURCE = "continuous_shadow"
ONE_SHOT_PARENT_SOURCE = "phase1h6a_one_shot_commit"


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records", "events", "iterations", "batches"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise SystemExit(f"{path} does not contain trace records")


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


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


def step_plan_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int_value(record.get("step_id"), int_value(record.get("eager_commit_step_id"), -1)),
        int_value(record.get("plan_id"), int_value(record.get("eager_commit_plan_id"), -1)),
    )


def continuous_row(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("enable_continuous_eager_dry_run", False))
        or bool(record.get("continuous_eager_dry_run_enabled", False))
        or bool(as_int_set(record.get("continuous_eager_candidate_proposal_ids")))
        or bool(as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids")))
        or bool(as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids")))
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    commit_errors, commit_summary = validate_commit_records(records)
    errors.extend(f"one-shot commit checker: {error}" for error in commit_errors)

    one_shot_committed_ids: set[int] = set()
    one_shot_ready_ids: set[int] = set()
    one_shot_committed_ids_by_record: dict[int, set[int]] = {}
    for idx, record in enumerate(records):
        committed = as_int_set(record.get("eager_committed_proposal_ids"))
        ready = as_int_set(record.get("eager_commit_ready_proposal_ids")) | as_int_set(
            record.get("eager_commit_from_readiness_proposal_ids")
        )
        one_shot_committed_ids.update(committed)
        one_shot_ready_ids.update(ready)
        one_shot_committed_ids_by_record[idx] = committed

    records_with_enabled = 0
    records_with_verify_apply_enabled = 0
    records_with_commit_enabled = 0
    active_records = 0
    candidate_ids_seen: set[int] = set()
    candidate_token_by_id: dict[int, int] = {}
    ready_shadow_ids_seen: set[int] = set()
    not_ready_ids_seen: set[int] = set()
    verified_ids_seen: set[int] = set()
    verify_skipped_ids_seen: set[int] = set()
    full_accept_ids_seen: set[int] = set()
    duplicate_ids_seen: set[int] = set()
    frontier_mismatch_ids_seen: set[int] = set()
    true_frontier_mismatch_ids_seen: set[int] = set()
    parent_shadow_not_committed_ids_seen: set[int] = set()
    real_commit_count = 0
    mutation_detected_count = 0
    missing_unexpected_count = 0
    sent_result_ids_seen: set[int] = set()
    received_result_ids_seen: set[int] = set()
    validated_result_ids_seen: set[int] = set()
    invalid_result_ids_seen: set[int] = set()
    sync_executed_ids_seen: set[int] = set()
    chain_depths: dict[int, int] = {}
    drop_reason_by_id: dict[int, str] = {}
    step_seq_depth_seen: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    result_transfer_protocols: set[str] = set()
    result_transfer_payload_len_units = 0
    result_transfer_payload_len_units_before_compact = 0
    zero_result_fast_path_count = 0

    for idx, record in enumerate(records):
        enabled = bool(record.get("enable_continuous_eager_dry_run", False))
        verify_apply_enabled = bool(record.get("enable_continuous_eager_verify_apply_dry_run", False))
        commit_enabled = bool(record.get("enable_continuous_eager_commit_depth1_ready_only", False))
        if enabled:
            records_with_enabled += 1
        if verify_apply_enabled:
            records_with_verify_apply_enabled += 1
        if commit_enabled:
            records_with_commit_enabled += 1
        if not continuous_row(record):
            continue
        if not enabled:
            errors.append(f"record[{idx}] has continuous eager fields while flag is disabled")
        active = bool(record.get("continuous_eager_dry_run_enabled", False))
        if active:
            active_records += 1
        if active and record.get("continuous_eager_source") != CONTINUOUS_SOURCE:
            errors.append(f"record[{idx}] continuous source is not {CONTINUOUS_SOURCE!r}")
        stage = record.get("continuous_eager_execution_stage") or record.get("continuous_shadow_stage")
        if verify_apply_enabled and active and stage != "verify_apply_dry_run":
            errors.append(f"record[{idx}] continuous verify/apply enabled but stage is {stage!r}")

        candidate_ids = as_int_list(record.get("continuous_eager_candidate_proposal_ids"))
        candidate_seq_ids = as_int_list(record.get("continuous_eager_candidate_seq_ids"))
        candidate_id_set = set(candidate_ids)
        not_ready_ids = as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids"))
        ready_shadow_ids = as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids"))
        verified_ids = as_int_set(record.get("continuous_eager_verified_proposal_ids"))
        verify_candidates = as_int_set(record.get("continuous_eager_verify_dry_run_candidate_proposal_ids"))
        verify_executed = as_int_set(record.get("continuous_eager_verify_dry_run_executed_proposal_ids"))
        verify_skipped = as_int_set(record.get("continuous_eager_verify_dry_run_skipped_proposal_ids"))
        full_accept_ids = as_int_set(record.get("continuous_eager_full_accept_proposal_ids"))
        apply_candidates = as_int_set(record.get("continuous_eager_apply_dry_run_candidate_proposal_ids"))
        apply_executed = as_int_set(record.get("continuous_eager_apply_dry_run_executed_proposal_ids"))
        token_by_id = as_int_map(record.get("continuous_eager_candidate_token_count_by_proposal_id"))
        parent_by_id = as_int_map(record.get("continuous_eager_parent_proposal_id_by_proposal_id"))
        root_by_id = as_int_map(record.get("continuous_eager_root_proposal_id_by_proposal_id"))
        depth_by_id = as_int_map(record.get("continuous_eager_chain_depth_by_proposal_id"))
        reason_by_id = as_str_map(record.get("continuous_eager_not_ready_shadow_reason_by_proposal_id"))
        parent_source_by_id = as_str_map(record.get("continuous_eager_parent_source_by_proposal_id"))
        verify_result_by_id = as_str_map(record.get("continuous_eager_verify_result_by_proposal_id"))
        accept_len_by_id = as_int_map(record.get("continuous_eager_accept_len_by_proposal_id"))
        apply_action_by_id = as_str_map(record.get("continuous_eager_apply_action_by_proposal_id"))
        apply_rollback_by_id = record.get("continuous_eager_apply_rollback_ok_by_proposal_id") or {}
        apply_mutation_by_id = record.get("continuous_eager_apply_mutation_detected_by_proposal_id") or {}
        apply_checkpoint_by_id = record.get("continuous_eager_apply_checkpoint_failed_by_proposal_id") or {}
        sync_action_match_by_id = record.get("continuous_eager_sync_apply_action_match_by_proposal_id") or {}
        sync_result_match_by_id = record.get("continuous_eager_sync_apply_result_match_by_proposal_id") or {}
        sync_accept_match_by_id = record.get("continuous_eager_sync_apply_accept_len_match_by_proposal_id") or {}
        sync_rollback_by_id = record.get("continuous_eager_sync_apply_rollback_ok_by_proposal_id") or {}
        sync_mutation_by_id = record.get("continuous_eager_sync_apply_mutation_detected_by_proposal_id") or {}
        sync_checkpoint_by_id = record.get("continuous_eager_sync_apply_checkpoint_failed_by_proposal_id") or {}
        max_depth = int_value(record.get("max_continuous_eager_chain_depth"), 0)
        observed_depth = int_value(record.get("max_continuous_depth_observed"), 0)
        step, _plan = step_plan_key(record)

        if len(candidate_ids) != len(candidate_seq_ids):
            errors.append(f"record[{idx}] candidate proposal/seq length mismatch")
        for proposal_id in candidate_ids:
            if proposal_id not in token_by_id or int(token_by_id.get(proposal_id, 0)) <= 0:
                errors.append(f"record[{idx}] candidate {proposal_id} missing positive token count")
            if proposal_id not in parent_by_id:
                errors.append(f"record[{idx}] candidate {proposal_id} missing parent proposal id")
            if proposal_id not in root_by_id:
                errors.append(f"record[{idx}] candidate {proposal_id} missing root proposal id")
            depth = int(depth_by_id.get(proposal_id, 0))
            if depth <= 0:
                errors.append(f"record[{idx}] candidate {proposal_id} missing positive chain depth")
            if max_depth > 0 and depth > max_depth:
                errors.append(f"record[{idx}] candidate {proposal_id} exceeds max chain depth")
            parent_id = int(parent_by_id.get(proposal_id, -1))
            if depth == 1 and parent_id not in one_shot_committed_ids and parent_id not in one_shot_ready_ids:
                errors.append(
                    f"record[{idx}] candidate {proposal_id} parent {parent_id} is not one-shot committed/ready"
                )
            if (
                verify_apply_enabled
                and verify_candidates
                and depth == 1
                and proposal_id in verify_candidates
                and proposal_id not in (verify_executed | verify_skipped)
            ):
                errors.append(f"record[{idx}] depth-1 candidate {proposal_id} was not verified or skipped")
        if max_depth > 0 and observed_depth > max_depth:
            errors.append(f"record[{idx}] observed depth {observed_depth} exceeds configured max {max_depth}")
        if verify_apply_enabled and "shadow_verify_not_executed" in reason_by_id.values():
            errors.append(f"record[{idx}] verify/apply stage still reports shadow_verify_not_executed")

        for proposal_id, seq_id in zip(candidate_ids, candidate_seq_ids):
            depth = int(depth_by_id.get(proposal_id, 0))
            event_key = (step, int(seq_id), depth)
            step_seq_depth_seen[event_key].add(int(proposal_id))

        for proposal_id in not_ready_ids:
            if not reason_by_id.get(proposal_id):
                errors.append(f"record[{idx}] not-ready continuous proposal {proposal_id} lacks reason")
        if ready_shadow_ids - candidate_id_set:
            errors.append(
                f"record[{idx}] shadow-ready ids are not candidates: {sorted(ready_shadow_ids - candidate_id_set)}"
            )
        if full_accept_ids - verified_ids:
            errors.append(
                f"record[{idx}] full-accept ids are not verified: {sorted(full_accept_ids - verified_ids)}"
            )
        if verify_apply_enabled and apply_executed - full_accept_ids:
            errors.append(f"record[{idx}] non-full-accept ids were apply-executed: {sorted(apply_executed - full_accept_ids)}")
        for proposal_id in full_accept_ids:
            if int(accept_len_by_id.get(proposal_id, -1)) != int(token_by_id.get(proposal_id, 0)):
                errors.append(f"record[{idx}] full-accept {proposal_id} accept_len/token_count mismatch")
            if apply_action_by_id.get(proposal_id) not in {None, "append_full_accept_then_rollback"}:
                errors.append(f"record[{idx}] full-accept {proposal_id} has bad apply action")
            if str(apply_rollback_by_id.get(str(proposal_id), apply_rollback_by_id.get(proposal_id, True))) == "False":
                errors.append(f"record[{idx}] full-accept {proposal_id} apply rollback failed")
            if bool(apply_mutation_by_id.get(str(proposal_id), apply_mutation_by_id.get(proposal_id, False))):
                errors.append(f"record[{idx}] full-accept {proposal_id} apply mutation detected")
            if bool(apply_checkpoint_by_id.get(str(proposal_id), apply_checkpoint_by_id.get(proposal_id, False))):
                errors.append(f"record[{idx}] full-accept {proposal_id} apply checkpoint failed")
        for proposal_id in ready_shadow_ids:
            for mapping, name in (
                (sync_action_match_by_id, "sync action"),
                (sync_result_match_by_id, "sync result"),
                (sync_accept_match_by_id, "sync accept_len"),
                (sync_rollback_by_id, "sync rollback"),
            ):
                if mapping and not bool(mapping.get(str(proposal_id), mapping.get(proposal_id, False))):
                    errors.append(f"record[{idx}] shadow-ready {proposal_id} failed {name} guard")
            if bool(sync_mutation_by_id.get(str(proposal_id), sync_mutation_by_id.get(proposal_id, False))):
                errors.append(f"record[{idx}] shadow-ready {proposal_id} sync mutation detected")
            if bool(sync_checkpoint_by_id.get(str(proposal_id), sync_checkpoint_by_id.get(proposal_id, False))):
                errors.append(f"record[{idx}] shadow-ready {proposal_id} sync checkpoint failed")
        continuous_ids = (
            candidate_id_set
            | not_ready_ids
            | ready_shadow_ids
            | verified_ids
            | full_accept_ids
        )
        if continuous_ids & one_shot_committed_ids_by_record.get(idx, set()):
            errors.append(f"record[{idx}] continuous ids were real committed in one-shot field")
        if continuous_ids & as_int_set(record.get("lane_exclusion_applied_proposal_ids")):
            errors.append(f"record[{idx}] continuous ids affected lane exclusion")
        if continuous_ids & as_int_set(record.get("target_eager_verify_proposal_ids_dry_run")):
            errors.append(f"record[{idx}] continuous ids entered target takeover lane")
        if not commit_enabled and int_value(record.get("continuous_eager_real_commit_count"), 0) != 0:
            errors.append(f"record[{idx}] continuous eager real commit count is nonzero")
        if int_value(record.get("continuous_depth2_real_commit_count"), 0) != 0:
            errors.append(f"record[{idx}] continuous depth-2 real commit count is nonzero")
        if int_value(record.get("missing_buffered_proposal_unexpected_count"), 0) != 0:
            errors.append(f"record[{idx}] aggregate missing buffered proposal unexpected count is nonzero")
        if record.get("missing_buffered_proposal_unexpected_seq_ids"):
            errors.append(f"record[{idx}] unexpected missing buffered proposals present")

        candidate_ids_seen.update(candidate_id_set)
        ready_shadow_ids_seen.update(ready_shadow_ids)
        not_ready_ids_seen.update(not_ready_ids)
        verified_ids_seen.update(verified_ids)
        verified_ids_seen.update(verify_executed)
        verify_skipped_ids_seen.update(verify_skipped)
        full_accept_ids_seen.update(full_accept_ids)
        sent_result_ids_seen.update(as_int_set(record.get("continuous_eager_result_transfer_sent_proposal_ids")))
        received_result_ids_seen.update(as_int_set(record.get("continuous_eager_result_transfer_received_proposal_ids")))
        validated_result_ids_seen.update(as_int_set(record.get("continuous_eager_result_transfer_validated_proposal_ids")))
        invalid_result_ids_seen.update(as_int_set(record.get("continuous_eager_result_transfer_invalid_proposal_ids")))
        protocol = record.get("continuous_eager_result_transfer_protocol")
        if protocol:
            result_transfer_protocols.add(str(protocol))
            if str(protocol) != "compact_v1":
                errors.append(f"record[{idx}] unexpected continuous result-transfer protocol {protocol!r}")
        compacted = record.get("continuous_eager_result_transfer_compacted")
        if protocol and compacted is not True:
            errors.append(f"record[{idx}] compact continuous result transfer missing compacted=true")
        payload_len = int_value(record.get("continuous_eager_result_transfer_payload_len_units"), 0)
        payload_before = int_value(
            record.get("continuous_eager_result_transfer_payload_len_units_before_compact"),
            0,
        )
        if payload_len < 0 or payload_before < 0:
            errors.append(f"record[{idx}] continuous result-transfer payload lengths must be nonnegative")
        if payload_before and payload_len > payload_before:
            errors.append(f"record[{idx}] compact payload larger than before-compact payload")
        result_transfer_payload_len_units += max(0, payload_len)
        result_transfer_payload_len_units_before_compact += max(0, payload_before)
        zero_result_fast_path_count += int_value(record.get("continuous_zero_result_fast_path_count"), 0)
        sent_count = int_value(record.get("continuous_eager_result_transfer_sent_count"), -1)
        received_count = int_value(record.get("continuous_eager_result_transfer_received_count"), -1)
        validated_count = int_value(record.get("continuous_eager_result_transfer_validated_count"), -1)
        if sent_count >= 0 and sent_count != len(as_int_set(record.get("continuous_eager_result_transfer_sent_proposal_ids"))):
            errors.append(f"record[{idx}] continuous sent count/list mismatch")
        if received_count >= 0 and received_count != len(as_int_set(record.get("continuous_eager_result_transfer_received_proposal_ids"))):
            errors.append(f"record[{idx}] continuous received count/list mismatch")
        if validated_count >= 0 and validated_count != len(as_int_set(record.get("continuous_eager_result_transfer_validated_proposal_ids"))):
            errors.append(f"record[{idx}] continuous validated count/list mismatch")
        sync_executed_ids_seen.update(as_int_set(record.get("continuous_eager_sync_apply_executed_proposal_ids")))
        duplicate_ids_seen.update(as_int_set(record.get("continuous_eager_duplicate_proposal_ids")))
        frontier_mismatch_ids_seen.update(
            as_int_set(record.get("continuous_eager_frontier_mismatch_proposal_ids"))
        )
        true_frontier_mismatch_ids_seen.update(
            as_int_set(record.get("continuous_eager_true_frontier_mismatch_proposal_ids"))
        )
        parent_shadow_not_committed_ids_seen.update(
            as_int_set(record.get("continuous_eager_parent_shadow_not_committed_proposal_ids"))
        )
        real_commit_count += int_value(record.get("continuous_eager_real_commit_count"), 0)
        mutation_detected_count += int_value(record.get("continuous_eager_mutation_detected_count"), 0)
        missing_unexpected_count += len(as_int_list(record.get("missing_buffered_proposal_unexpected_seq_ids")))
        for proposal_id, token_count in token_by_id.items():
            if proposal_id in candidate_id_set and token_count > 0:
                candidate_token_by_id.setdefault(proposal_id, token_count)
        for proposal_id, depth in depth_by_id.items():
            if proposal_id in candidate_id_set and depth > 0:
                chain_depths.setdefault(proposal_id, depth)
        for proposal_id, reason in reason_by_id.items():
            drop_reason_by_id.setdefault(proposal_id, reason)
        for proposal_id, source in parent_source_by_id.items():
            if proposal_id in candidate_id_set and source not in {
                ONE_SHOT_PARENT_SOURCE,
                CONTINUOUS_SOURCE,
            }:
                errors.append(f"record[{idx}] continuous proposal {proposal_id} has bad parent source {source!r}")

    repeated_seq_depth = {
        key: sorted(proposal_ids)
        for key, proposal_ids in step_seq_depth_seen.items()
        if len(proposal_ids) > 1
    }
    if repeated_seq_depth:
        errors.append(f"duplicate continuous candidates for same step/seq/depth: {repeated_seq_depth}")
    ready_without_full_accept = ready_shadow_ids_seen - full_accept_ids_seen
    if ready_without_full_accept:
        errors.append(f"continuous shadow-ready ids were not full-accept: {sorted(ready_without_full_accept)}")
    if records_with_verify_apply_enabled:
        depth1_candidates = {
            proposal_id for proposal_id, depth in chain_depths.items() if int(depth) == 1
        }
        missing_verify_or_skip = depth1_candidates - verified_ids_seen - verify_skipped_ids_seen
        if missing_verify_or_skip:
            errors.append(
                f"continuous depth-1 candidates not verified or skipped: {sorted(missing_verify_or_skip)}"
            )
        missing_result_receive = sent_result_ids_seen - received_result_ids_seen
        if missing_result_receive:
            errors.append(f"continuous result transfer sent but not received: {sorted(missing_result_receive)}")
        missing_result_validation = received_result_ids_seen - validated_result_ids_seen - invalid_result_ids_seen
        if missing_result_validation:
            errors.append(f"continuous result transfer missing validation: {sorted(missing_result_validation)}")
        if invalid_result_ids_seen:
            errors.append(f"continuous result transfer invalid ids: {sorted(invalid_result_ids_seen)}")
        missing_sync = validated_result_ids_seen - sync_executed_ids_seen
        if missing_sync:
            errors.append(f"continuous validated results not sync-applied: {sorted(missing_sync)}")
        if parent_shadow_not_committed_ids_seen & true_frontier_mismatch_ids_seen:
            errors.append("parent_shadow_not_committed counted as true frontier mismatch")

    accounting = aggregate_performance_accounting(records, {})
    candidate_tokens = sum(candidate_token_by_id.values())
    ready_shadow_tokens = int_value(accounting.get("continuous_eager_commit_ready_shadow_token_count"), 0)
    chain_distribution = Counter(str(depth) for depth in chain_depths.values())
    summary = {
        "total_trace_records": len(records),
        "records_with_continuous_eager_enabled": records_with_enabled,
        "records_with_continuous_verify_apply_enabled": records_with_verify_apply_enabled,
        "records_with_continuous_commit_enabled": records_with_commit_enabled,
        "continuous_active_records": active_records,
        "one_shot_committed_proposal_count": int_value(commit_summary.get("committed_proposal_count"), 0),
        "one_shot_committed_token_count": int_value(commit_summary.get("committed_token_count"), 0),
        "continuous_candidate_proposal_count": len(candidate_ids_seen),
        "continuous_candidate_token_count": candidate_tokens,
        "continuous_verified_proposal_count": len(verified_ids_seen),
        "continuous_full_accept_proposal_count": len(full_accept_ids_seen),
        "continuous_commit_ready_shadow_proposal_count": len(ready_shadow_ids_seen),
        "continuous_commit_ready_shadow_token_count": ready_shadow_tokens,
        "continuous_not_ready_shadow_proposal_count": len(not_ready_ids_seen),
        "continuous_chain_length_distribution": dict(chain_distribution),
        "continuous_drop_reason_counts": dict(Counter(drop_reason_by_id.values())),
        "continuous_duplicate_count": len(duplicate_ids_seen),
        "continuous_frontier_mismatch_count": len(frontier_mismatch_ids_seen),
        "continuous_true_frontier_mismatch_count": len(true_frontier_mismatch_ids_seen),
        "continuous_parent_shadow_not_committed_count": len(parent_shadow_not_committed_ids_seen),
        "continuous_mutation_detected_count": mutation_detected_count,
        "continuous_real_commit_count": real_commit_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
        "continuous_result_transfer_sent_count": len(sent_result_ids_seen),
        "continuous_result_transfer_received_count": len(received_result_ids_seen),
        "continuous_result_transfer_validated_count": len(validated_result_ids_seen),
        "continuous_result_transfer_protocols": sorted(result_transfer_protocols),
        "continuous_result_transfer_payload_len_units": result_transfer_payload_len_units,
        "continuous_result_transfer_payload_len_units_before_compact": result_transfer_payload_len_units_before_compact,
        "continuous_zero_result_fast_path_count": zero_result_fast_path_count,
        "continuous_sync_apply_executed_count": len(sync_executed_ids_seen),
        "combined_one_shot_plus_continuous_shadow_token_count": accounting.get(
            "combined_one_shot_plus_continuous_shadow_token_count",
            0,
        ),
        "combined_estimated_token_share_of_output": accounting.get(
            "combined_estimated_token_share_of_output",
            0.0,
        ),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_continuous_eager_enabled",
        "records_with_continuous_verify_apply_enabled",
        "records_with_continuous_commit_enabled",
        "continuous_active_records",
        "one_shot_committed_proposal_count",
        "one_shot_committed_token_count",
        "continuous_candidate_proposal_count",
        "continuous_candidate_token_count",
        "continuous_verified_proposal_count",
        "continuous_full_accept_proposal_count",
        "continuous_commit_ready_shadow_proposal_count",
        "continuous_commit_ready_shadow_token_count",
        "continuous_not_ready_shadow_proposal_count",
        "continuous_chain_length_distribution",
        "continuous_drop_reason_counts",
        "continuous_duplicate_count",
        "continuous_frontier_mismatch_count",
        "continuous_true_frontier_mismatch_count",
        "continuous_parent_shadow_not_committed_count",
        "continuous_mutation_detected_count",
        "continuous_real_commit_count",
        "missing_buffered_proposal_unexpected_count",
        "continuous_result_transfer_sent_count",
        "continuous_result_transfer_received_count",
        "continuous_result_transfer_validated_count",
        "continuous_result_transfer_protocols",
        "continuous_result_transfer_payload_len_units",
        "continuous_result_transfer_payload_len_units_before_compact",
        "continuous_zero_result_fast_path_count",
        "continuous_sync_apply_executed_count",
        "combined_one_shot_plus_continuous_shadow_token_count",
        "combined_estimated_token_share_of_output",
    ):
        print(f"{key}={summary.get(key)}")


def synthetic_base_record() -> dict[str, Any]:
    return {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_commit_readiness_dry_run": True,
        "enable_eager_commit_ready_only": True,
        "eager_commit_enabled": True,
        "eager_commit_source": "phase1h5e3_takeover_lane",
        "eager_commit_side": "target",
        "eager_commit_step_id": 7,
        "eager_commit_plan_id": 27,
        "eager_commit_candidate_proposal_ids": [6],
        "eager_commit_ready_proposal_ids": [6],
        "eager_commit_from_readiness_proposal_ids": [6],
        "eager_committed_proposal_ids": [6],
        "eager_committed_seq_ids": [12],
        "eager_committed_token_count_by_proposal_id": {"6": 4},
        "eager_committed_accept_len_by_proposal_id": {"6": 4},
        "eager_committed_action_by_proposal_id": {"6": "append_full_accept_then_rollback"},
        "eager_committed_verify_result_by_proposal_id": {"6": "full_accept"},
        "eager_commit_precondition_ok_by_proposal_id": {"6": True},
        "eager_commit_precondition_failed_by_proposal_id": {"6": False},
        "eager_commit_target_seq_len_before_by_seq_id": {"12": 20},
        "eager_commit_target_seq_len_after_by_seq_id": {"12": 24},
        "eager_commit_draft_seq_len_before_by_seq_id": {"12": 20},
        "eager_commit_draft_seq_len_after_by_seq_id": {"12": 24},
        "eager_commit_target_draft_len_match_by_seq_id": {"12": True},
        "eager_commit_target_draft_token_match_by_seq_id": {"12": True},
        "eager_commit_candidate_count": 1,
        "eager_commit_committed_count": 1,
        "eager_commit_skipped_count": 0,
        "eager_tokens_committed": 4,
        "eager_tokens_committed_full_accept": 4,
        "eager_tokens_verified": 4,
        "eager_tokens_accepted": 4,
        "eager_tokens_rejected": 0,
        "eager_tokens_invalidated": 0,
        "step_id": 7,
        "plan_id": 27,
    }


def synthetic_continuous_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_continuous_eager_dry_run": True,
            "continuous_eager_dry_run_enabled": True,
            "continuous_eager_source": CONTINUOUS_SOURCE,
            "continuous_eager_parent_source": ONE_SHOT_PARENT_SOURCE,
            "max_continuous_eager_chain_depth": 2,
            "continuous_eager_candidate_proposal_ids": [900000601],
            "continuous_eager_candidate_seq_ids": [12],
            "continuous_eager_candidate_token_count_by_proposal_id": {"900000601": 4},
            "continuous_eager_parent_proposal_id_by_proposal_id": {
                "900000601": 6,
                "900000602": 900000601,
            },
            "continuous_eager_chain_depth_by_proposal_id": {
                "900000601": 1,
                "900000602": 2,
            },
            "continuous_eager_root_proposal_id_by_proposal_id": {
                "900000601": 6,
                "900000602": 6,
            },
            "continuous_eager_parent_source_by_proposal_id": {
                "900000601": ONE_SHOT_PARENT_SOURCE,
                "900000602": CONTINUOUS_SOURCE,
            },
            "continuous_eager_not_ready_shadow_proposal_ids": [900000601, 900000602],
            "continuous_eager_not_ready_shadow_reason_by_proposal_id": {
                "900000601": "shadow_verify_not_executed",
                "900000602": "parent_shadow_not_committed",
            },
            "continuous_eager_parent_shadow_not_committed_proposal_ids": [900000602],
            "continuous_eager_true_frontier_mismatch_proposal_ids": [],
            "continuous_eager_frontier_mismatch_proposal_ids": [],
            "continuous_eager_candidate_proposal_count": 1,
            "continuous_eager_candidate_token_count": 4,
            "continuous_eager_commit_ready_shadow_proposal_count": 0,
            "continuous_eager_commit_ready_shadow_token_count": 0,
            "continuous_eager_real_commit_count": 0,
            "continuous_eager_mutation_detected_count": 0,
        }
    )
    return record


def synthetic_verify_apply_records() -> list[dict[str, Any]]:
    target = synthetic_base_record()
    target.update(
        {
            "enable_continuous_eager_dry_run": True,
            "enable_continuous_eager_verify_apply_dry_run": True,
            "continuous_eager_dry_run_enabled": True,
            "continuous_eager_source": CONTINUOUS_SOURCE,
            "continuous_eager_execution_stage": "verify_apply_dry_run",
            "continuous_shadow_stage": "verify_apply_dry_run",
            "max_continuous_eager_chain_depth": 1,
            "max_continuous_depth_configured": 1,
            "max_continuous_depth_observed": 1,
            "continuous_eager_candidate_proposal_ids": [900000601],
            "continuous_eager_candidate_seq_ids": [12],
            "continuous_eager_candidate_token_count_by_proposal_id": {"900000601": 4},
            "continuous_eager_parent_proposal_id_by_proposal_id": {"900000601": 6},
            "continuous_eager_chain_depth_by_proposal_id": {"900000601": 1},
            "continuous_eager_root_proposal_id_by_proposal_id": {"900000601": 6},
            "continuous_eager_parent_source_by_proposal_id": {"900000601": ONE_SHOT_PARENT_SOURCE},
            "continuous_eager_verify_dry_run_candidate_proposal_ids": [900000601],
            "continuous_eager_verify_dry_run_executed_proposal_ids": [900000601],
            "continuous_eager_verified_proposal_ids": [900000601],
            "continuous_eager_verify_result_by_proposal_id": {"900000601": "full_accept"},
            "continuous_eager_accept_len_by_proposal_id": {"900000601": 4},
            "continuous_eager_full_accept_proposal_ids": [900000601],
            "continuous_eager_apply_dry_run_candidate_proposal_ids": [900000601],
            "continuous_eager_apply_dry_run_executed_proposal_ids": [900000601],
            "continuous_eager_apply_action_by_proposal_id": {"900000601": "append_full_accept_then_rollback"},
            "continuous_eager_apply_rollback_ok_by_proposal_id": {"900000601": True},
            "continuous_eager_apply_mutation_detected_by_proposal_id": {"900000601": False},
            "continuous_eager_apply_checkpoint_failed_by_proposal_id": {"900000601": False},
            "continuous_eager_result_transfer_sent_proposal_ids": [900000601],
            "continuous_eager_result_transfer_protocol": "compact_v1",
            "continuous_eager_result_transfer_compacted": True,
            "continuous_eager_result_transfer_payload_len_units": 13,
            "continuous_eager_result_transfer_payload_len_units_before_compact": 31,
            "continuous_eager_result_transfer_sent_count": 1,
            "continuous_eager_real_commit_count": 0,
        }
    )
    draft = synthetic_base_record()
    draft["eager_commit_side"] = "draft"
    draft.update(
        {
            "enable_continuous_eager_dry_run": True,
            "enable_continuous_eager_verify_apply_dry_run": True,
            "continuous_eager_dry_run_enabled": True,
            "continuous_eager_source": CONTINUOUS_SOURCE,
            "continuous_eager_execution_stage": "verify_apply_dry_run",
            "continuous_shadow_stage": "verify_apply_dry_run",
            "max_continuous_eager_chain_depth": 1,
            "max_continuous_depth_configured": 1,
            "max_continuous_depth_observed": 1,
            "continuous_eager_candidate_proposal_ids": [900000601],
            "continuous_eager_candidate_seq_ids": [12],
            "continuous_eager_candidate_token_count_by_proposal_id": {"900000601": 4},
            "continuous_eager_parent_proposal_id_by_proposal_id": {"900000601": 6},
            "continuous_eager_chain_depth_by_proposal_id": {"900000601": 1},
            "continuous_eager_root_proposal_id_by_proposal_id": {"900000601": 6},
            "continuous_eager_parent_source_by_proposal_id": {"900000601": ONE_SHOT_PARENT_SOURCE},
            "continuous_eager_result_transfer_received_proposal_ids": [900000601],
            "continuous_eager_result_transfer_validated_proposal_ids": [900000601],
            "continuous_eager_result_transfer_protocol": "compact_v1",
            "continuous_eager_result_transfer_compacted": True,
            "continuous_eager_result_transfer_payload_len_units": 13,
            "continuous_eager_result_transfer_payload_len_units_before_compact": 31,
            "continuous_eager_result_transfer_received_count": 1,
            "continuous_eager_result_transfer_validated_count": 1,
            "continuous_eager_sync_apply_executed_proposal_ids": [900000601],
            "continuous_eager_sync_apply_action_match_by_proposal_id": {"900000601": True},
            "continuous_eager_sync_apply_result_match_by_proposal_id": {"900000601": True},
            "continuous_eager_sync_apply_accept_len_match_by_proposal_id": {"900000601": True},
            "continuous_eager_sync_apply_rollback_ok_by_proposal_id": {"900000601": True},
            "continuous_eager_sync_apply_mutation_detected_by_proposal_id": {"900000601": False},
            "continuous_eager_sync_apply_checkpoint_failed_by_proposal_id": {"900000601": False},
            "continuous_eager_commit_ready_shadow_proposal_ids": [900000601],
            "continuous_eager_commit_ready_shadow_seq_ids": [12],
            "continuous_eager_commit_ready_shadow_token_count_by_proposal_id": {"900000601": 4},
            "continuous_eager_real_commit_count": 0,
        }
    )
    return [target, draft]


def run_synthetic_tests() -> None:
    draft_record = synthetic_base_record()
    draft_record["eager_commit_side"] = "draft"
    valid = [synthetic_continuous_record(), draft_record]
    errors, summary = validate_records(valid)
    assert not errors, f"valid continuous synthetic failed: {errors}"
    assert summary["continuous_candidate_token_count"] == 4
    assert summary["continuous_real_commit_count"] == 0
    assert summary["continuous_drop_reason_counts"]["shadow_verify_not_executed"] == 1

    verify_valid = synthetic_verify_apply_records()
    errors, summary = validate_records(verify_valid)
    assert not errors, f"valid verify/apply synthetic failed: {errors}"
    assert summary["continuous_verified_proposal_count"] == 1
    assert summary["continuous_commit_ready_shadow_token_count"] == 4

    invalid = deepcopy(valid)
    invalid[0]["enable_continuous_eager_dry_run"] = False
    errors, _summary = validate_records(invalid)
    assert any("flag is disabled" in error for error in errors), "missed disabled-flag continuous fields"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_real_commit_count"] = 1
    errors, _summary = validate_records(invalid)
    assert any("real commit count is nonzero" in error for error in errors), "missed real continuous commit"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = {}
    errors, _summary = validate_records(invalid)
    assert any("lacks reason" in error for error in errors), "missed missing not-ready reason"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_candidate_proposal_ids"] = [900000601, 900000699]
    invalid[0]["continuous_eager_candidate_seq_ids"] = [12, 12]
    invalid[0]["continuous_eager_candidate_token_count_by_proposal_id"]["900000699"] = 4
    invalid[0]["continuous_eager_parent_proposal_id_by_proposal_id"]["900000699"] = 6
    invalid[0]["continuous_eager_root_proposal_id_by_proposal_id"]["900000699"] = 6
    invalid[0]["continuous_eager_chain_depth_by_proposal_id"]["900000699"] = 1
    invalid[0]["continuous_eager_parent_source_by_proposal_id"]["900000699"] = ONE_SHOT_PARENT_SOURCE
    errors, _summary = validate_records(invalid)
    assert any("duplicate continuous candidates" in error for error in errors), "missed duplicate seq/depth"

    invalid = deepcopy(valid)
    invalid[0]["continuous_eager_chain_depth_by_proposal_id"]["900000601"] = 3
    errors, _summary = validate_records(invalid)
    assert any("exceeds max chain depth" in error for error in errors), "missed excessive chain depth"

    invalid = deepcopy(verify_valid)
    invalid[0]["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = {
        "900000601": "shadow_verify_not_executed"
    }
    invalid[0]["continuous_eager_not_ready_shadow_proposal_ids"] = [900000601]
    errors, _summary = validate_records(invalid)
    assert any("shadow_verify_not_executed" in error for error in errors), "missed stale 7a not-executed reason"

    invalid = deepcopy(verify_valid)
    invalid[1]["continuous_eager_sync_apply_rollback_ok_by_proposal_id"] = {"900000601": False}
    errors, _summary = validate_records(invalid)
    assert any("sync rollback" in error for error in errors), "missed sync rollback failure"

    invalid = deepcopy(verify_valid)
    invalid[0]["continuous_eager_verify_dry_run_executed_proposal_ids"] = []
    invalid[0]["continuous_eager_verify_dry_run_skipped_proposal_ids"] = []
    invalid[0]["continuous_eager_verified_proposal_ids"] = []
    invalid[0]["continuous_eager_full_accept_proposal_ids"] = []
    invalid[1]["continuous_eager_commit_ready_shadow_proposal_ids"] = []
    invalid[1]["continuous_eager_commit_ready_shadow_seq_ids"] = []
    invalid[1]["continuous_eager_commit_ready_shadow_token_count_by_proposal_id"] = {}
    errors, _summary = validate_records(invalid)
    assert any("not verified or skipped" in error for error in errors), "missed unexecuted depth-1 7b candidate"

    print("Synthetic continuous eager dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-7a continuous eager shadow dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    errors, summary = validate_records(load_trace(args.trace))
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Continuous eager dry-run checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
