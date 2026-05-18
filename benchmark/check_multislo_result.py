#!/usr/bin/env python3
"""Validate multi-SLO result or request-trace files.

Examples:
  python benchmark/check_multislo_result.py results/multislo/*.json
  python benchmark/check_multislo_result.py results/multislo/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def infer_tokens(row: Dict[str, Any]) -> int:
    value = first_present(
        row,
        [
            "num_decode_output_tokens",
            "num_output_tokens",
            "num_completion_tokens",
            "completion_tokens",
            "output_tokens",
            "num_generated_tokens",
            "num_tokens",
        ],
    )
    if isinstance(value, list):
        return len(value)
    try:
        return int(value or 0)
    except Exception:
        return 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_file(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if path.suffix == ".jsonl":
        return {}, load_jsonl(path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {}, [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return {}, []
    for key in ("traces", "requests", "request_traces", "request_summaries"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return payload, [row for row in rows if isinstance(row, dict)]
    return payload, []


def quantiles(values: List[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    return min(values), statistics.median(values), max(values)


def fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def values_from_mapping(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def summarize_plan_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_plan_rows = 0
    plan_roles = set()
    effective_gamma_values = set()
    legacy_false = 0
    eager_true = 0
    non_null_home_batch_id = 0
    null_home_batch_id = 0
    raw_home_batch_id_values = set()
    raw_home_batch_id_counts: Counter[Any] = Counter()
    raw_target_home_batch_id_values = set()
    raw_draft_home_batch_id_values = set()
    raw_same_target_draft = 0
    raw_invalid_target_draft = 0
    plan_ids: List[int] = []
    plan_id_role_counts: Dict[tuple[Any, Any], int] = {}
    real_probe_attempted_rows = 0
    real_probe_applied_rows = 0
    real_probe_blocked_rows = 0
    real_probe_block_reasons: set[Any] = set()
    raw_actual_differs_from_scheduled = 0
    raw_protocol_alignment_false = 0
    raw_empty_actual_exec = 0
    actual_exec_fraction_values: List[float] = []
    protocol_layouts = set()
    protocol_versions = set()
    protocol_validation_error_count = 0
    raw_protocol_validation_false = 0
    raw_draft_message_count = 0
    raw_verify_result_message_count = 0
    raw_protocol_seq_alignment_errors = 0
    draft_message_total_tokens: List[float] = []
    verify_result_total_tokens: List[float] = []
    variable_offsets_seen_count = 0
    variable_offsets_validation_errors = 0
    cross_batch_routing_error_count = 0
    mailbox_put_count = 0
    mailbox_get_hit_count = 0
    mailbox_get_miss_count = 0
    mailbox_warmup_miss_count = 0
    mailbox_routing_error_count = 0
    mailbox_error_kinds = set()
    raw_mailbox_put_rows = 0
    raw_mailbox_get_rows = 0
    raw_mailbox_success_rows = 0
    raw_mailbox_missing_seq_count = 0
    mailbox_transport_send_count = 0
    mailbox_transport_recv_count = 0
    mailbox_transport_send_success_count = 0
    mailbox_transport_recv_success_count = 0
    mailbox_transport_error_kinds = set()
    mailbox_payload_tensor_transport_attempt_count = 0
    mailbox_payload_tensor_transport_success_count = 0
    mailbox_payload_tensor_transport_error_count = 0
    target_mailbox_insert_count = 0
    pipeline_phases_seen = set()
    warmup_draft_payload_produced_count = 0
    mailbox_warmup_skip_count = 0
    target_verify_skipped_for_warmup_count = 0
    target_consume_from_mailbox_attempt_count = 0
    target_consume_from_mailbox_success_count = 0
    target_consume_from_mailbox_error_count = 0
    verification_input_from_mailbox_attempt_count = 0
    verification_input_from_mailbox_success_count = 0
    verification_input_from_mailbox_error_count = 0
    target_forward_from_mailbox_input_built_count = 0
    target_forward_mailbox_context_build_count = 0
    target_forward_mailbox_context_success_count = 0
    target_forward_mailbox_context_error_count = 0
    target_forward_mailbox_slot_mapping_available_count = 0
    target_forward_mailbox_cannot_run_reasons = set()
    mailbox_kv_sync_plan_built_count = 0
    kv_state_sync_check_attempt_count = 0
    kv_state_sync_check_success_count = 0
    kv_state_sync_check_error_count = 0
    kv_state_sync_error_kinds = set()
    target_forward_from_mailbox_attempt_count = 0
    target_forward_from_mailbox_success_count = 0
    target_forward_from_mailbox_error_count = 0
    target_forward_output_normalization_attempt_count = 0
    target_forward_output_normalization_success_count = 0
    target_forward_output_normalization_error_count = 0
    target_forward_output_none_expected_count = 0
    target_forward_output_none_unexpected_count = 0
    target_forward_output_owner_ranks = set()
    target_tp_owner_rows = 0
    mailbox_payload_availability_rows = 0
    target_tp_skipped_non_owner_count = 0
    mailbox_payload_envelope_available_count = 0
    mailbox_payload_token_ids_available_count = 0
    mailbox_payload_tensor_available_count = 0
    mailbox_payload_missing_reasons = set()
    mailbox_missing_payload_count = 0
    mailbox_verify_apply_path_count = 0
    output_interpretation_skipped_non_owner_count = 0
    mailbox_verify_apply_skipped_non_owner_count = 0
    output_interpretation_attempt_count = 0
    output_interpretation_success_count = 0
    output_interpretation_error_count = 0
    mailbox_verify_apply_attempt_count = 0
    mailbox_verify_apply_success_count = 0
    mailbox_verify_apply_error_count = 0
    mailbox_forward_state_mutation_attempt_count = 0
    mailbox_forward_state_mutation_committed_count = 0
    mailbox_forward_state_mutation_rollback_success_count = 0
    illegal_legacy_fallback_count = 0
    raw_mailbox_transport_send_rows = 0
    raw_mailbox_transport_recv_rows = 0
    raw_target_consume_from_mailbox_rows = 0
    next_required_features = set()
    raw_variable_draft_message_count = 0
    raw_variable_verify_result_message_count = 0
    variable_draft_total_tokens: List[float] = []
    variable_verify_total_tokens: List[float] = []

    for row in rows:
        signature = row.get("plan_signature")
        if not isinstance(signature, dict):
            signature = {}
        has_plan = any(
            key in row
            for key in (
                "plan_signature",
                "plan_id",
                "plan_digest",
                "plan_runner_role",
                "effective_gamma_per_seq",
                "stspec_mailbox_enabled",
                "mailbox_put_attempted",
                "mailbox_get_attempted",
                "mailbox_transport_send_attempted",
                "mailbox_transport_recv_attempted",
                "target_consume_from_mailbox_attempted",
                "stspec_pipeline_phase",
                "mailbox_payload_tensor_transport_attempted",
            )
        )
        if not has_plan:
            continue

        raw_plan_rows += 1
        role = row.get("plan_runner_role") or signature.get("runner_role")
        if role is not None:
            plan_roles.add(role)

        plan_id = row.get("plan_id", signature.get("plan_id"))
        try:
            plan_id_int = int(plan_id)
        except Exception:
            plan_id_int = None
        if plan_id_int is not None:
            plan_ids.append(plan_id_int)
            key = (role, plan_id_int)
            plan_id_role_counts[key] = plan_id_role_counts.get(key, 0) + 1

        legacy_equivalent = row.get(
            "plan_legacy_equivalent", signature.get("legacy_equivalent")
        )
        if legacy_equivalent is False:
            legacy_false += 1

        gamma_values = values_from_mapping(row.get("effective_gamma_per_seq"))
        gamma_values += values_from_mapping(signature.get("effective_gamma_per_seq"))
        if row.get("effective_gamma") is not None:
            gamma_values.append(row.get("effective_gamma"))
        for value in gamma_values:
            try:
                effective_gamma_values.add(int(value))
            except Exception:
                pass

        eager_values = values_from_mapping(row.get("is_eager_per_seq"))
        eager_values += values_from_mapping(signature.get("is_eager_per_seq"))
        if row.get("is_eager") is not None:
            eager_values.append(row.get("is_eager"))
        if any(value is True for value in eager_values):
            eager_true += 1

        row_home_values = values_from_mapping(row.get("home_batch_id_per_seq"))
        signature_home_values = values_from_mapping(signature.get("home_batch_id_per_seq"))
        home_values = row_home_values if row_home_values else signature_home_values
        if row.get("home_batch_id") is not None:
            home_values.append(row.get("home_batch_id"))

        target_home_batch_id = row.get(
            "target_home_batch_id", signature.get("target_home_batch_id")
        )
        draft_home_batch_id = row.get(
            "draft_home_batch_id", signature.get("draft_home_batch_id")
        )
        if target_home_batch_id is not None:
            raw_target_home_batch_id_values.add(target_home_batch_id)
        if draft_home_batch_id is not None:
            raw_draft_home_batch_id_values.add(draft_home_batch_id)
        if target_home_batch_id is not None and draft_home_batch_id is not None:
            if target_home_batch_id == draft_home_batch_id:
                raw_same_target_draft += 1
            if target_home_batch_id not in {0, 1} or draft_home_batch_id not in {0, 1}:
                raw_invalid_target_draft += 1

        if any(value is not None for value in home_values):
            non_null_home_batch_id += 1
        if any(value is None for value in home_values) or not home_values:
            null_home_batch_id += 1
        for value in home_values:
            if value is not None:
                raw_home_batch_id_values.add(value)
                raw_home_batch_id_counts[value] += 1


        if bool(row.get("real_probe_attempted") or signature.get("real_probe_attempted")):
            real_probe_attempted_rows += 1
        if bool(row.get("real_probe_applied") or signature.get("real_probe_applied")):
            real_probe_applied_rows += 1
        if bool(row.get("real_probe_blocked") or signature.get("real_probe_blocked")):
            real_probe_blocked_rows += 1
        for reason_value in values_from_mapping(row.get("real_probe_block_reasons")):
            if reason_value:
                real_probe_block_reasons.add(reason_value)
        reason = row.get("real_probe_block_reason", signature.get("real_probe_block_reason"))
        if reason:
            real_probe_block_reasons.add(reason)

        actual_exec_seq_ids = row.get("actual_exec_seq_ids", signature.get("actual_exec_seq_ids"))
        scheduled_seq_ids = row.get("plan_scheduled_seq_ids") or row.get("scheduled_seq_ids") or signature.get("scheduled_seq_ids")
        if isinstance(actual_exec_seq_ids, list) and isinstance(scheduled_seq_ids, list):
            if actual_exec_seq_ids != scheduled_seq_ids:
                raw_actual_differs_from_scheduled += 1
            if not actual_exec_seq_ids:
                raw_empty_actual_exec += 1
        protocol_ok = row.get("protocol_alignment_ok", signature.get("protocol_alignment_ok"))
        if protocol_ok is False:
            raw_protocol_alignment_false += 1
        fraction = to_float(row.get("actual_exec_fraction", signature.get("actual_exec_fraction")))
        if fraction is not None:
            actual_exec_fraction_values.append(fraction)


        layout = row.get("pearl_protocol_layout")
        if layout is not None:
            protocol_layouts.add(layout)
        for value in values_from_mapping(row.get("pearl_protocol_layouts")):
            if value is not None:
                protocol_layouts.add(value)
        version = row.get("pearl_protocol_version")
        if version is not None:
            protocol_versions.add(version)
        if row.get("protocol_validation_error"):
            protocol_validation_error_count += 1
        protocol_validation_error_count += int(row.get("protocol_validation_error_count") or 0)
        if row.get("protocol_validation_ok") is False:
            raw_protocol_validation_false += 1
        if row.get("draft_message_seq_ids") is not None:
            raw_draft_message_count += 1
        raw_draft_message_count += int(row.get("draft_message_seen_count") or 0)
        if row.get("verify_result_seq_ids") is not None:
            raw_verify_result_message_count += 1
        raw_verify_result_message_count += int(row.get("verify_result_seen_count") or 0)
        if "seq alignment" in str(row.get("protocol_validation_error") or "").lower():
            raw_protocol_seq_alignment_errors += 1
        raw_protocol_seq_alignment_errors += int(row.get("protocol_seq_alignment_error_count") or 0)
        draft_total = to_float(row.get("draft_message_total_tokens"))
        if draft_total is not None:
            draft_message_total_tokens.append(draft_total)
        verify_total = to_float(row.get("verify_result_total_tokens"))
        if verify_total is not None:
            verify_result_total_tokens.append(verify_total)
        if row.get("variable_offsets_enabled") is True or row.get("pearl_protocol_layout") == "variable_offsets":
            variable_offsets_seen_count += 1
        variable_offsets_seen_count += int(row.get("variable_offsets_seen_count") or 0)
        if row.get("variable_offsets_validation_error"):
            variable_offsets_validation_errors += 1
        variable_offsets_validation_errors += int(row.get("variable_offsets_validation_error_count") or 0)
        if row.get("cross_batch_routing_error"):
            cross_batch_routing_error_count += 1
        cross_batch_routing_error_count += int(row.get("cross_batch_routing_error_count") or 0)
        mailbox_put_count += int(row.get("mailbox_put_count") or 0)
        mailbox_get_hit_count += int(row.get("mailbox_get_hit_count") or 0)
        mailbox_get_miss_count += int(row.get("mailbox_get_miss_count") or 0)
        mailbox_warmup_miss_count += int(row.get("mailbox_warmup_miss_count") or 0)
        mailbox_routing_error_count += int(row.get("mailbox_routing_error_count") or 0)
        if row.get("mailbox_put_attempted"):
            raw_mailbox_put_rows += 1
        if row.get("mailbox_get_attempted"):
            raw_mailbox_get_rows += 1
        if row.get("mailbox_get_success") or row.get("mailbox_put_success"):
            raw_mailbox_success_rows += 1
        if row.get("mailbox_warmup_miss"):
            mailbox_warmup_miss_count += 1
        if row.get("mailbox_routing_ok") is False or row.get("mailbox_error"):
            mailbox_routing_error_count += 1
        if row.get("mailbox_error_kind"):
            mailbox_error_kinds.add(row.get("mailbox_error_kind"))
        for value in values_from_mapping(row.get("mailbox_error_kinds")):
            if value:
                mailbox_error_kinds.add(value)
        raw_mailbox_missing_seq_count += len(row.get("mailbox_missing_seq_ids") or [])
        raw_mailbox_missing_seq_count += int(row.get("mailbox_missing_count") or 0)
        mailbox_transport_send_count += int(row.get("mailbox_transport_send_count") or 0)
        mailbox_transport_recv_count += int(row.get("mailbox_transport_recv_count") or 0)
        mailbox_transport_send_success_count += int(row.get("mailbox_transport_send_success_count") or 0)
        mailbox_transport_recv_success_count += int(row.get("mailbox_transport_recv_success_count") or 0)
        if row.get("mailbox_transport_send_attempted"):
            raw_mailbox_transport_send_rows += 1
            mailbox_transport_send_count += 1
        if row.get("mailbox_transport_recv_attempted"):
            raw_mailbox_transport_recv_rows += 1
            mailbox_transport_recv_count += 1
        if row.get("mailbox_transport_send_success"):
            mailbox_transport_send_success_count += 1
        if row.get("mailbox_transport_recv_success"):
            mailbox_transport_recv_success_count += 1
        if row.get("mailbox_transport_error_kind"):
            mailbox_transport_error_kinds.add(row.get("mailbox_transport_error_kind"))
        for value in values_from_mapping(row.get("mailbox_transport_error_kinds")):
            if value:
                mailbox_transport_error_kinds.add(value)
        if row.get("stspec_pipeline_phase"):
            pipeline_phases_seen.add(row.get("stspec_pipeline_phase"))
        for value in values_from_mapping(row.get("stspec_pipeline_phases")):
            if value:
                pipeline_phases_seen.add(value)
        warmup_draft_payload_produced_count += int(row.get("warmup_draft_payload_produced_count") or 0)
        if row.get("warmup_draft_payload_produced"):
            warmup_draft_payload_produced_count += 1
        mailbox_payload_tensor_transport_attempt_count += int(row.get("mailbox_payload_tensor_transport_attempt_count") or 0)
        mailbox_payload_tensor_transport_success_count += int(row.get("mailbox_payload_tensor_transport_success_count") or 0)
        mailbox_payload_tensor_transport_error_count += int(row.get("mailbox_payload_tensor_transport_error_count") or 0)
        target_mailbox_insert_count += int(row.get("target_mailbox_insert_count") or 0)
        if row.get("mailbox_payload_tensor_transport_attempted"):
            mailbox_payload_tensor_transport_attempt_count += 1
        if row.get("mailbox_payload_tensor_transport_success"):
            mailbox_payload_tensor_transport_success_count += 1
        if row.get("mailbox_payload_tensor_transport_error"):
            mailbox_payload_tensor_transport_error_count += 1
        mailbox_warmup_skip_count += int(row.get("mailbox_warmup_skip_count") or 0)
        target_verify_skipped_for_warmup_count += int(row.get("target_verify_skipped_for_warmup_count") or 0)
        target_consume_from_mailbox_attempt_count += int(row.get("target_consume_from_mailbox_attempt_count") or 0)
        target_consume_from_mailbox_success_count += int(row.get("target_consume_from_mailbox_success_count") or 0)
        target_consume_from_mailbox_error_count += int(row.get("target_consume_from_mailbox_error_count") or 0)
        verification_input_from_mailbox_attempt_count += int(row.get("verification_input_from_mailbox_attempt_count") or 0)
        verification_input_from_mailbox_success_count += int(row.get("verification_input_from_mailbox_success_count") or 0)
        verification_input_from_mailbox_error_count += int(row.get("verification_input_from_mailbox_error_count") or 0)
        target_forward_from_mailbox_input_built_count += int(row.get("target_forward_from_mailbox_input_built_count") or 0)
        target_forward_mailbox_context_build_count += int(row.get("target_forward_mailbox_context_build_count") or 0)
        target_forward_mailbox_context_success_count += int(row.get("target_forward_mailbox_context_success_count") or 0)
        target_forward_mailbox_context_error_count += int(row.get("target_forward_mailbox_context_error_count") or 0)
        target_forward_mailbox_slot_mapping_available_count += int(row.get("target_forward_mailbox_slot_mapping_available_count") or 0)
        for value in values_from_mapping(row.get("target_forward_mailbox_cannot_run_reasons")):
            if value:
                target_forward_mailbox_cannot_run_reasons.add(value)
        mailbox_kv_sync_plan_built_count += int(row.get("mailbox_kv_sync_plan_built_count") or 0)
        kv_state_sync_check_attempt_count += int(row.get("kv_state_sync_check_attempt_count") or 0)
        kv_state_sync_check_success_count += int(row.get("kv_state_sync_check_success_count") or 0)
        kv_state_sync_check_error_count += int(row.get("kv_state_sync_error_count") or row.get("kv_state_sync_check_error_count") or 0)
        for value in values_from_mapping(row.get("kv_state_sync_error_kinds")):
            if value:
                kv_state_sync_error_kinds.add(value)
        target_forward_from_mailbox_attempt_count += int(row.get("target_forward_from_mailbox_attempt_count") or 0)
        target_forward_from_mailbox_success_count += int(row.get("target_forward_from_mailbox_success_count") or 0)
        target_forward_from_mailbox_error_count += int(row.get("target_forward_from_mailbox_error_count") or 0)
        target_forward_output_normalization_attempt_count += int(row.get("target_forward_output_normalization_attempt_count") or 0)
        target_forward_output_normalization_success_count += int(row.get("target_forward_output_normalization_success_count") or 0)
        target_forward_output_normalization_error_count += int(row.get("target_forward_output_normalization_error_count") or 0)
        target_forward_output_none_expected_count += int(row.get("target_forward_output_none_expected_count") or 0)
        target_forward_output_none_unexpected_count += int(row.get("target_forward_output_none_unexpected_count") or 0)
        target_tp_owner_rows += int(row.get("target_tp_owner_rows") or 0)
        mailbox_payload_availability_rows += int(row.get("mailbox_payload_availability_rows") or 0)
        target_tp_skipped_non_owner_count += int(row.get("target_tp_skipped_non_owner_count") or 0)
        mailbox_payload_envelope_available_count += int(row.get("mailbox_payload_envelope_available_count") or 0)
        mailbox_payload_token_ids_available_count += int(row.get("mailbox_payload_token_ids_available_count") or 0)
        mailbox_payload_tensor_available_count += int(row.get("mailbox_payload_tensor_available_count") or 0)
        for value in values_from_mapping(row.get("mailbox_payload_missing_reasons")):
            if value:
                mailbox_payload_missing_reasons.add(value)
        mailbox_missing_payload_count += int(row.get("mailbox_missing_payload_count") or 0)
        mailbox_verify_apply_path_count += int(row.get("mailbox_verify_apply_path_count") or 0)
        output_interpretation_skipped_non_owner_count += int(row.get("output_interpretation_skipped_non_owner_count") or 0)
        mailbox_verify_apply_skipped_non_owner_count += int(row.get("mailbox_verify_apply_skipped_non_owner_count") or 0)
        for value in values_from_mapping(row.get("target_forward_output_owner_ranks") or row.get("target_tp_owner_ranks_seen")):
            if value is not None:
                target_forward_output_owner_ranks.add(value)
        output_interpretation_attempt_count += int(row.get("output_interpretation_attempt_count") or 0)
        output_interpretation_success_count += int(row.get("output_interpretation_success_count") or 0)
        output_interpretation_error_count += int(row.get("output_interpretation_error_count") or 0)
        mailbox_verify_apply_attempt_count += int(row.get("mailbox_verify_apply_attempt_count") or 0)
        mailbox_verify_apply_success_count += int(row.get("mailbox_verify_apply_success_count") or 0)
        mailbox_verify_apply_error_count += int(row.get("mailbox_verify_apply_error_count") or 0)
        mailbox_forward_state_mutation_attempt_count += int(row.get("mailbox_forward_state_mutation_attempt_count") or 0)
        mailbox_forward_state_mutation_committed_count += int(row.get("mailbox_forward_state_mutation_committed_count") or row.get("mailbox_forward_state_mutation_commit_count") or 0)
        mailbox_forward_state_mutation_rollback_success_count += int(row.get("mailbox_forward_state_mutation_rollback_success_count") or 0)
        illegal_legacy_fallback_count += int(row.get("illegal_legacy_fallback_count") or 0)
        if row.get("mailbox_warmup_skip"):
            mailbox_warmup_skip_count += 1
        if row.get("target_verify_skipped_for_warmup"):
            target_verify_skipped_for_warmup_count += 1
        if row.get("target_consume_from_mailbox_attempted"):
            raw_target_consume_from_mailbox_rows += 1
            target_consume_from_mailbox_attempt_count += 1
        if row.get("target_consume_from_mailbox_success"):
            target_consume_from_mailbox_success_count += 1
        if row.get("target_consume_from_mailbox_error"):
            target_consume_from_mailbox_error_count += 1
        if row.get("verification_input_from_mailbox_attempted"):
            verification_input_from_mailbox_attempt_count += 1
        if row.get("verification_input_from_mailbox_success"):
            verification_input_from_mailbox_success_count += 1
        if row.get("verification_input_from_mailbox_error"):
            verification_input_from_mailbox_error_count += 1
        if row.get("target_forward_from_mailbox_input_built"):
            target_forward_from_mailbox_input_built_count += 1
        if row.get("target_forward_mailbox_context_build_attempted"):
            target_forward_mailbox_context_build_count += 1
        if row.get("target_forward_mailbox_context_build_success"):
            target_forward_mailbox_context_success_count += 1
        if row.get("target_forward_mailbox_context_error"):
            target_forward_mailbox_context_error_count += 1
        if row.get("target_forward_mailbox_slot_mapping_available"):
            target_forward_mailbox_slot_mapping_available_count += 1
        if row.get("target_forward_mailbox_cannot_run_reason"):
            target_forward_mailbox_cannot_run_reasons.add(row.get("target_forward_mailbox_cannot_run_reason"))
        if row.get("mailbox_kv_sync_plan_built"):
            mailbox_kv_sync_plan_built_count += 1
        if row.get("kv_state_sync_error_kind"):
            kv_state_sync_error_kinds.add(row.get("kv_state_sync_error_kind"))
        if row.get("kv_state_sync_check_attempted"):
            kv_state_sync_check_attempt_count += 1
        if row.get("kv_state_sync_check_success"):
            kv_state_sync_check_success_count += 1
        if row.get("kv_state_sync_error"):
            kv_state_sync_check_error_count += 1
        if row.get("target_forward_from_mailbox_attempted"):
            target_forward_from_mailbox_attempt_count += 1
        if row.get("target_forward_from_mailbox_success"):
            target_forward_from_mailbox_success_count += 1
        if row.get("target_forward_from_mailbox_error"):
            target_forward_from_mailbox_error_count += 1
        if row.get("target_forward_output_normalization_attempted"):
            target_forward_output_normalization_attempt_count += 1
        if row.get("target_forward_output_normalization_success"):
            target_forward_output_normalization_success_count += 1
        if row.get("target_forward_output_normalization_error"):
            target_forward_output_normalization_error_count += 1
        if row.get("target_forward_output_none_expected"):
            target_forward_output_none_expected_count += 1
        if row.get("target_forward_output_none_unexpected"):
            target_forward_output_none_unexpected_count += 1
        if row.get("target_forward_output_owner_rank") is not None:
            target_forward_output_owner_ranks.add(row.get("target_forward_output_owner_rank"))
        if "target_tp_is_output_owner" in row:
            target_tp_owner_rows += 1
        if any(key in row for key in ("mailbox_payload_envelope_available", "mailbox_payload_token_ids_available", "mailbox_payload_tensor_available")):
            mailbox_payload_availability_rows += 1
        if row.get("target_tp_skipped_non_owner"):
            target_tp_skipped_non_owner_count += 1
        if row.get("mailbox_payload_envelope_available"):
            mailbox_payload_envelope_available_count += 1
        if row.get("mailbox_payload_token_ids_available"):
            mailbox_payload_token_ids_available_count += 1
        if row.get("mailbox_payload_tensor_available"):
            mailbox_payload_tensor_available_count += 1
        if row.get("mailbox_payload_missing_reason"):
            mailbox_payload_missing_reasons.add(row.get("mailbox_payload_missing_reason"))
        if row.get("mailbox_error_kind") == "mailbox_missing_payload":
            mailbox_missing_payload_count += 1
        if row.get("next_required_feature") == "mailbox_verify_apply_path":
            mailbox_verify_apply_path_count += 1
        if row.get("output_interpretation_skipped_non_owner"):
            output_interpretation_skipped_non_owner_count += 1
        if row.get("mailbox_verify_apply_skipped_non_owner"):
            mailbox_verify_apply_skipped_non_owner_count += 1
        if row.get("target_forward_from_mailbox_output_interpretation_attempted"):
            output_interpretation_attempt_count += 1
        if row.get("target_forward_from_mailbox_output_interpretation_success"):
            output_interpretation_success_count += 1
        if row.get("target_forward_from_mailbox_output_interpretation_error"):
            output_interpretation_error_count += 1
        if row.get("mailbox_verify_apply_attempted"):
            mailbox_verify_apply_attempt_count += 1
        if row.get("mailbox_verify_apply_success"):
            mailbox_verify_apply_success_count += 1
        if row.get("mailbox_verify_apply_error"):
            mailbox_verify_apply_error_count += 1
        if row.get("mailbox_forward_state_mutation_attempted"):
            mailbox_forward_state_mutation_attempt_count += 1
        if row.get("mailbox_forward_state_mutation_committed"):
            mailbox_forward_state_mutation_committed_count += 1
        if row.get("mailbox_forward_state_mutation_rollback_success"):
            mailbox_forward_state_mutation_rollback_success_count += 1
        if row.get("illegal_legacy_fallback"):
            illegal_legacy_fallback_count += 1
        if row.get("next_required_feature"):
            next_required_features.add(row.get("next_required_feature"))
        for value in values_from_mapping(row.get("next_required_features")):
            if value:
                next_required_features.add(value)
        if row.get("variable_draft_message_seq_ids") is not None:
            raw_variable_draft_message_count += 1
        raw_variable_draft_message_count += int(row.get("variable_draft_message_seen_count") or 0)
        if row.get("variable_verify_result_seq_ids") is not None:
            raw_variable_verify_result_message_count += 1
        raw_variable_verify_result_message_count += int(row.get("variable_verify_result_seen_count") or 0)
        variable_draft_total = to_float(row.get("variable_draft_message_total_tokens"))
        if variable_draft_total is not None:
            variable_draft_total_tokens.append(variable_draft_total)
        variable_verify_total = to_float(row.get("variable_verify_result_total_tokens"))
        if variable_verify_total is not None:
            variable_verify_total_tokens.append(variable_verify_total)

    draft_total_min, draft_total_med, draft_total_max = quantiles(draft_message_total_tokens)
    verify_total_min, verify_total_med, verify_total_max = quantiles(verify_result_total_tokens)
    variable_draft_min, variable_draft_med, variable_draft_max = quantiles(variable_draft_total_tokens)
    variable_verify_min, variable_verify_med, variable_verify_max = quantiles(variable_verify_total_tokens)

    duplicate_plan_ids_by_role = {
        f"{role}:{plan_id}": count
        for (role, plan_id), count in plan_id_role_counts.items()
        if count > 1
    }

    return {
        "raw_plan_traces": raw_plan_rows,
        "unique_plan_roles": sorted(plan_roles, key=str),
        "unique_effective_gamma_values": sorted(effective_gamma_values),
        "plan_legacy_equivalent_false": legacy_false,
        "plan_is_eager_true": eager_true,
        "plan_home_batch_id_non_null": non_null_home_batch_id,
        "plan_home_batch_id_null": null_home_batch_id,
        "raw_unique_home_batch_id_values": sorted(raw_home_batch_id_values, key=str),
        "raw_count_per_home_batch_id": dict(
            sorted(raw_home_batch_id_counts.items(), key=lambda kv: str(kv[0]))
        ),
        "raw_unique_target_home_batch_id_values": sorted(
            raw_target_home_batch_id_values, key=str
        ),
        "raw_unique_draft_home_batch_id_values": sorted(
            raw_draft_home_batch_id_values, key=str
        ),
        "raw_same_target_draft_home_batch_id": raw_same_target_draft,
        "raw_invalid_target_draft_home_batch_id": raw_invalid_target_draft,
        "plan_id_min": min(plan_ids) if plan_ids else None,
        "plan_id_max": max(plan_ids) if plan_ids else None,
        "duplicate_plan_id_by_role": duplicate_plan_ids_by_role,
        "real_probe_attempted_rows": real_probe_attempted_rows,
        "real_probe_applied_rows": real_probe_applied_rows,
        "real_probe_blocked_rows": real_probe_blocked_rows,
        "unique_real_probe_block_reasons": sorted(real_probe_block_reasons, key=str),
        "raw_actual_exec_differs_from_scheduled": raw_actual_differs_from_scheduled,
        "raw_protocol_alignment_false": raw_protocol_alignment_false,
        "raw_empty_actual_exec": raw_empty_actual_exec,
        "avg_actual_exec_fraction": (
            sum(actual_exec_fraction_values) / len(actual_exec_fraction_values)
            if actual_exec_fraction_values
            else None
        ),
        "protocol_layouts_seen": sorted(protocol_layouts, key=str),
        "unique_protocol_versions": sorted(protocol_versions, key=str),
        "protocol_validation_error_count": protocol_validation_error_count,
        "raw_protocol_validation_false": raw_protocol_validation_false,
        "raw_draft_message_count": raw_draft_message_count,
        "raw_verify_result_message_count": raw_verify_result_message_count,
        "raw_protocol_seq_alignment_errors": raw_protocol_seq_alignment_errors,
        "draft_message_total_token_summary": (draft_total_min, draft_total_med, draft_total_max),
        "verify_result_total_token_summary": (verify_total_min, verify_total_med, verify_total_max),
        "variable_offsets_seen_count": variable_offsets_seen_count,
        "variable_offsets_validation_errors": variable_offsets_validation_errors,
        "cross_batch_routing_error_count": cross_batch_routing_error_count,
        "mailbox_put_count": mailbox_put_count,
        "mailbox_get_hit_count": mailbox_get_hit_count,
        "mailbox_get_miss_count": mailbox_get_miss_count,
        "mailbox_warmup_miss_count": mailbox_warmup_miss_count,
        "mailbox_routing_error_count": mailbox_routing_error_count,
        "mailbox_error_kinds": sorted(mailbox_error_kinds, key=str),
        "raw_mailbox_put_rows": raw_mailbox_put_rows,
        "raw_mailbox_get_rows": raw_mailbox_get_rows,
        "raw_mailbox_success_rows": raw_mailbox_success_rows,
        "raw_mailbox_missing_seq_count": raw_mailbox_missing_seq_count,
        "mailbox_transport_send_count": mailbox_transport_send_count,
        "mailbox_transport_recv_count": mailbox_transport_recv_count,
        "mailbox_transport_send_success_count": mailbox_transport_send_success_count,
        "mailbox_transport_recv_success_count": mailbox_transport_recv_success_count,
        "mailbox_transport_error_kinds": sorted(mailbox_transport_error_kinds, key=str),
        "mailbox_payload_tensor_transport_attempt_count": mailbox_payload_tensor_transport_attempt_count,
        "mailbox_payload_tensor_transport_success_count": mailbox_payload_tensor_transport_success_count,
        "mailbox_payload_tensor_transport_error_count": mailbox_payload_tensor_transport_error_count,
        "target_mailbox_insert_count": target_mailbox_insert_count,
        "pipeline_phases_seen": sorted(pipeline_phases_seen, key=str),
        "warmup_draft_payload_produced_count": warmup_draft_payload_produced_count,
        "mailbox_warmup_skip_count": mailbox_warmup_skip_count,
        "target_verify_skipped_for_warmup_count": target_verify_skipped_for_warmup_count,
        "target_consume_from_mailbox_attempt_count": target_consume_from_mailbox_attempt_count,
        "target_consume_from_mailbox_success_count": target_consume_from_mailbox_success_count,
        "target_consume_from_mailbox_error_count": target_consume_from_mailbox_error_count,
        "verification_input_from_mailbox_attempt_count": verification_input_from_mailbox_attempt_count,
        "verification_input_from_mailbox_success_count": verification_input_from_mailbox_success_count,
        "verification_input_from_mailbox_error_count": verification_input_from_mailbox_error_count,
        "target_forward_from_mailbox_input_built_count": target_forward_from_mailbox_input_built_count,
        "target_forward_mailbox_context_build_count": target_forward_mailbox_context_build_count,
        "target_forward_mailbox_context_success_count": target_forward_mailbox_context_success_count,
        "target_forward_mailbox_context_error_count": target_forward_mailbox_context_error_count,
        "target_forward_mailbox_slot_mapping_available_count": target_forward_mailbox_slot_mapping_available_count,
        "target_forward_mailbox_cannot_run_reasons": sorted(target_forward_mailbox_cannot_run_reasons, key=str),
        "mailbox_kv_sync_plan_built_count": mailbox_kv_sync_plan_built_count,
        "kv_state_sync_check_attempt_count": kv_state_sync_check_attempt_count,
        "kv_state_sync_check_success_count": kv_state_sync_check_success_count,
        "kv_state_sync_check_error_count": kv_state_sync_check_error_count,
        "kv_state_sync_error_kinds": sorted(kv_state_sync_error_kinds, key=str),
        "target_forward_from_mailbox_attempt_count": target_forward_from_mailbox_attempt_count,
        "target_forward_from_mailbox_success_count": target_forward_from_mailbox_success_count,
        "target_forward_from_mailbox_error_count": target_forward_from_mailbox_error_count,
        "target_forward_output_normalization_attempt_count": target_forward_output_normalization_attempt_count,
        "target_forward_output_normalization_success_count": target_forward_output_normalization_success_count,
        "target_forward_output_normalization_error_count": target_forward_output_normalization_error_count,
        "target_forward_output_none_expected_count": target_forward_output_none_expected_count,
        "target_forward_output_none_unexpected_count": target_forward_output_none_unexpected_count,
        "target_forward_output_owner_ranks": sorted(target_forward_output_owner_ranks, key=str),
        "target_tp_owner_ranks_seen": sorted(target_forward_output_owner_ranks, key=str),
        "target_tp_owner_rows": target_tp_owner_rows,
        "mailbox_payload_availability_rows": mailbox_payload_availability_rows,
        "target_tp_skipped_non_owner_count": target_tp_skipped_non_owner_count,
        "mailbox_payload_envelope_available_count": mailbox_payload_envelope_available_count,
        "mailbox_payload_token_ids_available_count": mailbox_payload_token_ids_available_count,
        "mailbox_payload_tensor_available_count": mailbox_payload_tensor_available_count,
        "mailbox_payload_missing_reasons": sorted(mailbox_payload_missing_reasons, key=str),
        "mailbox_missing_payload_count": mailbox_missing_payload_count,
        "mailbox_verify_apply_path_count": mailbox_verify_apply_path_count,
        "output_interpretation_skipped_non_owner_count": output_interpretation_skipped_non_owner_count,
        "mailbox_verify_apply_skipped_non_owner_count": mailbox_verify_apply_skipped_non_owner_count,
        "output_interpretation_attempt_count": output_interpretation_attempt_count,
        "output_interpretation_success_count": output_interpretation_success_count,
        "output_interpretation_error_count": output_interpretation_error_count,
        "mailbox_verify_apply_attempt_count": mailbox_verify_apply_attempt_count,
        "mailbox_verify_apply_success_count": mailbox_verify_apply_success_count,
        "mailbox_verify_apply_error_count": mailbox_verify_apply_error_count,
        "mailbox_forward_state_mutation_attempt_count": mailbox_forward_state_mutation_attempt_count,
        "mailbox_forward_state_mutation_committed_count": mailbox_forward_state_mutation_committed_count,
        "mailbox_forward_state_mutation_commit_count": mailbox_forward_state_mutation_committed_count,
        "mailbox_forward_state_mutation_rollback_success_count": mailbox_forward_state_mutation_rollback_success_count,
        "illegal_legacy_fallback_count": illegal_legacy_fallback_count,
        "raw_mailbox_transport_send_rows": raw_mailbox_transport_send_rows,
        "raw_mailbox_transport_recv_rows": raw_mailbox_transport_recv_rows,
        "raw_target_consume_from_mailbox_rows": raw_target_consume_from_mailbox_rows,
        "next_required_features": sorted(next_required_features, key=str),
        "raw_variable_draft_message_count": raw_variable_draft_message_count,
        "raw_variable_verify_result_message_count": raw_variable_verify_result_message_count,
        "variable_draft_total_token_summary": (variable_draft_min, variable_draft_med, variable_draft_max),
        "variable_verify_total_token_summary": (variable_verify_min, variable_verify_med, variable_verify_max),
    }


def summarize(path: Path) -> int:
    payload, rows = load_file(path)
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}

    execution_mode = args.get("execution_mode") or metrics.get("execution_mode")
    decode_ready_mode = args.get("decode_ready")
    if decode_ready_mode is None:
        decode_ready_mode = metrics.get("decode_ready_mode")

    engine_elapsed_s = metrics.get("engine_elapsed_s")
    total_output_tokens = overall.get("total_output_tokens")
    goodput = overall.get("goodput_tokens_per_s")
    mean_tpot_ms = overall.get("mean_tpot_ms")

    decode_elapsed_values: List[float] = []
    observed_tpot_values: List[float] = []
    arrival_after_finish = 0
    long_fast = 0
    elapsed_mismatch = 0
    tpot_mismatch = 0
    rows_with_effective_gamma = 0
    effective_gamma_values = set()
    eager_true = 0
    non_null_home_batch_id = 0
    home_batch_id_values = set()
    home_batch_id_counts: Counter[Any] = Counter()
    rows_with_plan_ids = 0
    rows_with_two_batch_shadow = 0
    target_home_batch_id_values = set()
    draft_home_batch_id_values = set()
    target_batch_hit_sum = 0
    draft_home_batch_hit_sum = 0

    for row in rows:
        tokens = infer_tokens(row)
        decode_elapsed_ms = to_float(row.get("decode_elapsed_ms"))
        if decode_elapsed_ms is not None:
            decode_elapsed_values.append(decode_elapsed_ms)
        observed_tpot_ms = to_float(row.get("observed_tpot_ms"))
        if observed_tpot_ms is not None:
            observed_tpot_values.append(observed_tpot_ms)

        arrival_ts = to_float(row.get("arrival_ts"))
        finish_ts = to_float(first_present(row, ["finish_ts", "finished_ts", "end_ts", "end_time"]))
        decode_start_ts = to_float(first_present(row, ["decode_start_ts", "decoding_start_ts", "first_decode_ts", "first_token_ts", "start_decode_ts"]))
        if arrival_ts is not None and finish_ts is not None and arrival_ts > finish_ts:
            arrival_after_finish += 1
        if tokens > 200 and decode_elapsed_ms is not None and decode_elapsed_ms < 1000:
            long_fast += 1
        if finish_ts is not None and decode_start_ts is not None and decode_elapsed_ms is not None:
            expected = (finish_ts - decode_start_ts) * 1000.0
            if abs(expected - decode_elapsed_ms) > 1e-3:
                elapsed_mismatch += 1
        if tokens > 0 and decode_elapsed_ms is not None and observed_tpot_ms is not None:
            expected = decode_elapsed_ms / tokens
            if abs(expected - observed_tpot_ms) > 1e-3:
                tpot_mismatch += 1

        if row.get("effective_gamma") is not None:
            rows_with_effective_gamma += 1
            effective_gamma_values.add(row.get("effective_gamma"))
        if row.get("is_eager") is True:
            eager_true += 1
        if row.get("home_batch_id") is not None:
            non_null_home_batch_id += 1
            home_batch_id_values.add(row.get("home_batch_id"))
            home_batch_id_counts[row.get("home_batch_id")] += 1
        plan_ids = row.get("plan_ids")
        if isinstance(plan_ids, list) and plan_ids:
            rows_with_plan_ids += 1
        if row.get("plan_two_batch_shadow") is True:
            rows_with_two_batch_shadow += 1
        for value in row.get("target_home_batch_ids") or []:
            target_home_batch_id_values.add(value)
        for value in row.get("draft_home_batch_ids") or []:
            draft_home_batch_id_values.add(value)
        target_batch_hit_sum += int(row.get("target_batch_hit_count") or 0)
        draft_home_batch_hit_sum += int(row.get("draft_home_batch_hit_count") or 0)

    dec_min, dec_med, dec_max = quantiles(decode_elapsed_values)
    tpot_min, tpot_med, tpot_max = quantiles(observed_tpot_values)

    print(f"\n== {path} ==")
    print(f"execution_mode: {fmt(execution_mode)}")
    print(f"decode_ready_mode: {fmt(decode_ready_mode)}")
    print(f"engine_elapsed_s: {fmt(engine_elapsed_s)}")
    print(f"total_output_tokens: {fmt(total_output_tokens)}")
    print(f"goodput_tokens_per_s: {fmt(goodput)}")
    print(f"mean_tpot_ms: {fmt(mean_tpot_ms)}")
    print(f"decode_elapsed_ms min/median/max: {fmt(dec_min)} / {fmt(dec_med)} / {fmt(dec_max)}")
    print(f"observed_tpot_ms min/median/max: {fmt(tpot_min)} / {fmt(tpot_med)} / {fmt(tpot_max)}")
    print(f"arrival_ts > finish_ts rows: {arrival_after_finish}")
    print(f"num_decode_output_tokens > 200 and decode_elapsed_ms < 1000 rows: {long_fast}")
    print(f"decode_elapsed_ms timestamp mismatches: {elapsed_mismatch}")
    print(f"observed_tpot_ms arithmetic mismatches: {tpot_mismatch}")
    print(f"rows with effective_gamma: {rows_with_effective_gamma}")
    print(f"unique effective_gamma values: {sorted(effective_gamma_values, key=str)}")
    print(f"rows with is_eager=true: {eager_true}")
    print(f"rows with non-null home_batch_id: {non_null_home_batch_id}")
    print(f"unique home_batch_id values: {sorted(home_batch_id_values, key=str)}")
    print(f"count per home_batch_id: {dict(sorted(home_batch_id_counts.items(), key=lambda kv: str(kv[0])))}")
    print(f"rows with plan_ids: {rows_with_plan_ids}")
    print(f"rows with plan_two_batch_shadow: {rows_with_two_batch_shadow}")
    print(
        "unique target_home_batch_id values: "
        f"{sorted(target_home_batch_id_values, key=str)}"
    )
    print(
        "unique draft_home_batch_id values: "
        f"{sorted(draft_home_batch_id_values, key=str)}"
    )
    print(f"sum target_batch_hit_count: {target_batch_hit_sum}")
    print(f"sum draft_home_batch_hit_count: {draft_home_batch_hit_sum}")

    plan_summary = summarize_plan_rows(rows)
    print(f"raw plan traces: {plan_summary['raw_plan_traces']}")
    print(f"unique plan roles: {plan_summary['unique_plan_roles']}")
    print(
        "raw unique effective_gamma values: "
        f"{plan_summary['unique_effective_gamma_values']}"
    )
    print(
        "raw plan_legacy_equivalent=false rows: "
        f"{plan_summary['plan_legacy_equivalent_false']}"
    )
    print(f"raw plan rows with is_eager=true: {plan_summary['plan_is_eager_true']}")
    print(
        "raw plan rows with non-null home_batch_id: "
        f"{plan_summary['plan_home_batch_id_non_null']}"
    )
    print(
        "raw plan rows with null home_batch_id: "
        f"{plan_summary['plan_home_batch_id_null']}"
    )
    print(
        "raw unique home_batch_id values: "
        f"{plan_summary['raw_unique_home_batch_id_values']}"
    )
    print(
        "raw count per home_batch_id: "
        f"{plan_summary['raw_count_per_home_batch_id']}"
    )
    print(
        "raw unique target_home_batch_id values: "
        f"{plan_summary['raw_unique_target_home_batch_id_values']}"
    )
    print(
        "raw unique draft_home_batch_id values: "
        f"{plan_summary['raw_unique_draft_home_batch_id_values']}"
    )
    print(
        "raw plans with target_home_batch_id == draft_home_batch_id: "
        f"{plan_summary['raw_same_target_draft_home_batch_id']}"
    )
    print(
        "raw plans with target/draft ids outside {0,1}: "
        f"{plan_summary['raw_invalid_target_draft_home_batch_id']}"
    )
    print(
        "raw plan_id range: "
        f"{fmt(plan_summary['plan_id_min'])} / {fmt(plan_summary['plan_id_max'])}"
    )
    print(
        "duplicate plan_id by role: "
        f"{plan_summary['duplicate_plan_id_by_role']}"
    )
    print(f"real_probe_attempted rows: {plan_summary['real_probe_attempted_rows']}")
    print(f"real_probe_applied rows: {plan_summary['real_probe_applied_rows']}")
    print(f"real_probe_blocked rows: {plan_summary['real_probe_blocked_rows']}")
    print(
        "unique real_probe_block_reason values: "
        f"{plan_summary['unique_real_probe_block_reasons']}"
    )
    print(
        "raw rows with actual_exec_seq_ids != scheduled_seq_ids: "
        f"{plan_summary['raw_actual_exec_differs_from_scheduled']}"
    )
    print(
        "raw rows with protocol_alignment_ok=false: "
        f"{plan_summary['raw_protocol_alignment_false']}"
    )
    print(f"raw rows with empty actual_exec_seq_ids: {plan_summary['raw_empty_actual_exec']}")
    print(
        "average actual_exec_fraction: "
        f"{fmt(plan_summary['avg_actual_exec_fraction'])}"
    )
    print(f"protocol layouts seen: {plan_summary['protocol_layouts_seen']}")
    print(f"unique protocol versions: {plan_summary['unique_protocol_versions']}")
    print(f"protocol validation errors: {plan_summary['protocol_validation_error_count']}")
    print(f"rows with protocol_validation_ok=false: {plan_summary['raw_protocol_validation_false']}")
    print(f"raw draft message count: {plan_summary['raw_draft_message_count']}")
    print(f"raw verify result message count: {plan_summary['raw_verify_result_message_count']}")
    print(f"rows with protocol seq alignment errors: {plan_summary['raw_protocol_seq_alignment_errors']}")
    draft_min, draft_med, draft_max = plan_summary['draft_message_total_token_summary']
    verify_min, verify_med, verify_max = plan_summary['verify_result_total_token_summary']
    print(
        "draft message total tokens min/median/max: "
        f"{fmt(draft_min)} / {fmt(draft_med)} / {fmt(draft_max)}"
    )
    print(
        "verify result accepted tokens min/median/max: "
        f"{fmt(verify_min)} / {fmt(verify_med)} / {fmt(verify_max)}"
    )
    print(f"variable_offsets seen count: {plan_summary['variable_offsets_seen_count']}")
    print(f"variable_offsets validation errors: {plan_summary['variable_offsets_validation_errors']}")
    print(f"cross_batch_routing_error_count: {plan_summary['cross_batch_routing_error_count']}")
    print(f"mailbox put count: {plan_summary['mailbox_put_count']}")
    print(f"mailbox get hit count: {plan_summary['mailbox_get_hit_count']}")
    print(f"mailbox get miss count: {plan_summary['mailbox_get_miss_count']}")
    print(f"mailbox warmup miss count: {plan_summary['mailbox_warmup_miss_count']}")
    print(f"mailbox routing error count: {plan_summary['mailbox_routing_error_count']}")
    print(f"mailbox_error_kinds: {plan_summary['mailbox_error_kinds']}")
    print(f"raw mailbox put rows: {plan_summary['raw_mailbox_put_rows']}")
    print(f"raw mailbox get rows: {plan_summary['raw_mailbox_get_rows']}")
    print(f"raw mailbox success rows: {plan_summary['raw_mailbox_success_rows']}")
    print(f"raw mailbox missing seq count: {plan_summary['raw_mailbox_missing_seq_count']}")
    print(f"mailbox transport send count: {plan_summary['mailbox_transport_send_count']}")
    print(f"mailbox transport recv count: {plan_summary['mailbox_transport_recv_count']}")
    print(f"mailbox transport send success count: {plan_summary['mailbox_transport_send_success_count']}")
    print(f"mailbox transport recv success count: {plan_summary['mailbox_transport_recv_success_count']}")
    print(f"mailbox transport error kinds: {plan_summary['mailbox_transport_error_kinds']}")
    print(f"mailbox payload tensor transport attempts: {plan_summary['mailbox_payload_tensor_transport_attempt_count']}")
    print(f"mailbox payload tensor transport successes: {plan_summary['mailbox_payload_tensor_transport_success_count']}")
    print(f"mailbox payload tensor transport errors: {plan_summary['mailbox_payload_tensor_transport_error_count']}")
    print(f"target mailbox insert count: {plan_summary['target_mailbox_insert_count']}")
    print(f"pipeline phases seen: {plan_summary['pipeline_phases_seen']}")
    print(f"warmup draft payload produced count: {plan_summary['warmup_draft_payload_produced_count']}")
    print(f"warmup skip count: {plan_summary['mailbox_warmup_skip_count']}")
    print(f"target verify skipped for warmup count: {plan_summary['target_verify_skipped_for_warmup_count']}")
    print(f"target consume-from-mailbox attempts: {plan_summary['target_consume_from_mailbox_attempt_count']}")
    print(f"target consume-from-mailbox successes: {plan_summary['target_consume_from_mailbox_success_count']}")
    print(f"target consume-from-mailbox errors: {plan_summary['target_consume_from_mailbox_error_count']}")
    print(f"verification input from mailbox attempts: {plan_summary['verification_input_from_mailbox_attempt_count']}")
    print(f"verification input from mailbox successes: {plan_summary['verification_input_from_mailbox_success_count']}")
    print(f"verification input from mailbox errors: {plan_summary['verification_input_from_mailbox_error_count']}")
    print(f"target forward input built count: {plan_summary['target_forward_from_mailbox_input_built_count']}")
    print(f"target forward mailbox context build attempts: {plan_summary['target_forward_mailbox_context_build_count']}")
    print(f"target forward mailbox context successes: {plan_summary['target_forward_mailbox_context_success_count']}")
    print(f"target forward mailbox context errors: {plan_summary['target_forward_mailbox_context_error_count']}")
    print(f"target forward mailbox slot mapping available count: {plan_summary['target_forward_mailbox_slot_mapping_available_count']}")
    print(f"target forward mailbox cannot-run reasons: {plan_summary['target_forward_mailbox_cannot_run_reasons']}")
    print(f"KV sync plan built count: {plan_summary['mailbox_kv_sync_plan_built_count']}")
    print(f"KV sync check attempts: {plan_summary['kv_state_sync_check_attempt_count']}")
    print(f"KV sync check successes: {plan_summary['kv_state_sync_check_success_count']}")
    print(f"KV sync check errors: {plan_summary['kv_state_sync_check_error_count']}")
    print(f"KV sync error kinds: {plan_summary['kv_state_sync_error_kinds']}")
    print(f"target forward from mailbox attempts: {plan_summary['target_forward_from_mailbox_attempt_count']}")
    print(f"target forward from mailbox successes: {plan_summary['target_forward_from_mailbox_success_count']}")
    print(f"target forward from mailbox errors: {plan_summary['target_forward_from_mailbox_error_count']}")
    print(f"target forward output normalization attempts: {plan_summary['target_forward_output_normalization_attempt_count']}")
    print(f"target forward output normalization successes: {plan_summary['target_forward_output_normalization_success_count']}")
    print(f"target forward output normalization errors: {plan_summary['target_forward_output_normalization_error_count']}")
    print(f"target forward output owner ranks seen: {plan_summary['target_forward_output_owner_ranks']}")
    print(f"target TP owner rows: {plan_summary['target_tp_owner_rows']}")
    print(f"mailbox payload availability rows: {plan_summary['mailbox_payload_availability_rows']}")
    print(f"target TP non-owner skip count: {plan_summary['target_tp_skipped_non_owner_count']}")
    print(f"mailbox payload envelope available count: {plan_summary['mailbox_payload_envelope_available_count']}")
    print(f"mailbox payload token ids available count: {plan_summary['mailbox_payload_token_ids_available_count']}")
    print(f"mailbox payload tensor available count: {plan_summary['mailbox_payload_tensor_available_count']}")
    print(f"mailbox payload missing reasons: {plan_summary['mailbox_payload_missing_reasons']}")
    print(f"mailbox_missing_payload count: {plan_summary['mailbox_missing_payload_count']}")
    print(f"mailbox_verify_apply_path count: {plan_summary['mailbox_verify_apply_path_count']}")
    print(f"target forward output none expected count: {plan_summary['target_forward_output_none_expected_count']}")
    print(f"target forward output none unexpected count: {plan_summary['target_forward_output_none_unexpected_count']}")
    print(f"output interpretation skipped non-owner count: {plan_summary['output_interpretation_skipped_non_owner_count']}")
    print(f"mailbox verify apply skipped non-owner count: {plan_summary['mailbox_verify_apply_skipped_non_owner_count']}")
    print(f"output interpretation attempts: {plan_summary['output_interpretation_attempt_count']}")
    print(f"output interpretation successes: {plan_summary['output_interpretation_success_count']}")
    print(f"output interpretation errors: {plan_summary['output_interpretation_error_count']}")
    print(f"mailbox verify apply attempts: {plan_summary['mailbox_verify_apply_attempt_count']}")
    print(f"mailbox verify apply successes: {plan_summary['mailbox_verify_apply_success_count']}")
    print(f"mailbox verify apply errors: {plan_summary['mailbox_verify_apply_error_count']}")
    print(f"mailbox forward state mutation attempts: {plan_summary['mailbox_forward_state_mutation_attempt_count']}")
    print(f"mailbox forward state mutation commits: {plan_summary['mailbox_forward_state_mutation_committed_count']}")
    print(f"mailbox forward state mutation rollback successes: {plan_summary['mailbox_forward_state_mutation_rollback_success_count']}")
    print(f"illegal legacy fallback count: {plan_summary['illegal_legacy_fallback_count']}")
    print(f"raw rows with mailbox_transport_send_attempted: {plan_summary['raw_mailbox_transport_send_rows']}")
    print(f"raw rows with mailbox_transport_recv_attempted: {plan_summary['raw_mailbox_transport_recv_rows']}")
    print(f"raw rows with target_consume_from_mailbox_attempted: {plan_summary['raw_target_consume_from_mailbox_rows']}")
    print(f"next_required_features: {plan_summary['next_required_features']}")
    print(f"raw variable draft message count: {plan_summary['raw_variable_draft_message_count']}")
    print(f"raw variable verify result message count: {plan_summary['raw_variable_verify_result_message_count']}")
    variable_draft_min, variable_draft_med, variable_draft_max = plan_summary['variable_draft_total_token_summary']
    variable_verify_min, variable_verify_med, variable_verify_max = plan_summary['variable_verify_total_token_summary']
    print(
        "variable draft total tokens min/median/max: "
        f"{fmt(variable_draft_min)} / {fmt(variable_draft_med)} / {fmt(variable_draft_max)}"
    )
    print(
        "variable verify total tokens min/median/max: "
        f"{fmt(variable_verify_min)} / {fmt(variable_verify_med)} / {fmt(variable_verify_max)}"
    )

    anomalous = arrival_after_finish + elapsed_mismatch + tpot_mismatch
    if long_fast:
        print("WARNING: long/fast rows found; inspect whether timestamps reflect true wall-clock completion.")
    if anomalous:
        print("ERROR: timing invariants failed.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Result JSON or request trace JSONL files")
    args = parser.parse_args()

    status = 0
    for raw_path in args.paths:
        try:
            status = max(status, summarize(Path(raw_path)))
        except Exception as exc:
            print(f"ERROR: {raw_path}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
