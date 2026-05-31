#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import int_value  # noqa: E402
from benchmark.check_eager_performance_accounting import load_json, trace_payload_to_records  # noqa: E402


DUAL_COLLECTIVE_STAGE_ORDER = [
    "normal_proposal_transfer",
    "target_verify_result_transfer",
    "eager_transfer",
    "eager_result_transfer",
    "generic_full_continuous_stage",
]


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "pass"}
    return bool(value)


def float_value(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, (str, bytes)):
        return []
    try:
        return [int(item) for item in value]
    except Exception:
        return []


def bool_map(value: Any) -> dict[int, bool]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, bool] = {}
    for raw_key, raw_value in value.items():
        try:
            result[int(raw_key)] = bool_value(raw_value)
        except Exception:
            continue
    return result


def load_trace_records(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        return trace_payload_to_records(payload)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def result_rows(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("traces", "requests", "request_traces", "request_summaries"):
        rows = result_payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def result_summary(result_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = result_payload.get("metrics", {}) if isinstance(result_payload, dict) else {}
    cached = result_payload.get("cached_admission", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(cached, dict):
        cached = {}
    nested = metrics.get("cached_admission", {})
    if not isinstance(nested, dict):
        nested = {}
    args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    if not isinstance(args, dict):
        args = {}
    summary = dict(nested)
    summary.update({key: value for key, value in metrics.items() if str(key).startswith("cached_")})
    summary.update(cached)
    if "cached_admission_enabled" not in summary:
        summary["cached_admission_enabled"] = bool(args.get("cached_admission") or args.get("enable_cached_admission"))
    if "cached_admission_max_active" not in summary:
        summary["cached_admission_max_active"] = int_value(
            args.get("max_active_cached_seqs") or args.get("cached_admission_max_active"),
            0,
        )
    if "cached_admission_policy" not in summary:
        summary["cached_admission_policy"] = args.get("cached_admission_policy", "fifo")
    if "execution_mode" not in summary and args.get("execution_mode") is not None:
        summary["execution_mode"] = args.get("execution_mode")
    return summary


def trace_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for record in records:
        for key, value in record.items():
            if (
                str(key).startswith("cached_admission_")
                or str(key).startswith("cached_prefill_")
                or key
                in {
                    "raw_target_normal_verify_seq_ids_before_buffer_filter",
                    "target_normal_verify_seq_ids_after_buffer_filter",
                    "proposal_pre_verify_by_seq_id",
                    "target_seq_pre_verify_by_seq_id",
                    "pre_verify_mismatch_seq_ids",
                    "pre_verify_stale_proposal_discarded_seq_ids",
                    "pre_verify_redraft_required_seq_ids",
                    "warmup_mode",
                    "cached_admission_decode_loop_active",
                    "requires_framed_dual_verify_result_transfer",
                    "cached_active_stage_debug_enabled",
                    "dual_stage_rank",
                    "dual_stage_tp_local_rank",
                    "active_cached_stage_order",
                    "active_cached_verify_result_transfer_sent",
                    "active_cached_verify_result_transfer_received",
                    "active_cached_eager_transfer_sent",
                    "active_cached_eager_transfer_received",
                    "active_cached_eager_result_sent",
                    "active_cached_eager_result_received",
                    "target_verify_seq_ids_before_received_proposal_override",
                    "target_verify_seq_ids_from_received_proposals",
                    "target_verify_seq_ids_after_received_proposal_override",
                    "target_tp_verify_seq_agreement_ok",
                    "target_tp_verify_seq_agreement_signature",
                    "target_tp_buffer_seq_agreement_ok",
                    "target_tp_buffer_seq_agreement_signature",
                    "target_tp_buffer_seq_ids",
                    "target_candidate_seq_ids_before_buffer_hit_agreement",
                    "target_candidate_buffer_hit_seq_ids",
                    "target_candidate_buffer_miss_seq_ids",
                    "target_tp_candidate_buffer_agreement_ok",
                    "target_tp_candidate_buffer_agreement_signatures",
                    "target_candidate_buffer_hit_agreed_seq_ids",
                    "target_candidate_buffer_miss_agreed_seq_ids",
                    "dual_buffer_mutation_events",
                    "received_proposal_seq_ids",
                    "fallback_same_batch_received_seq_ids",
                    "fallback_same_batch_verify_seq_ids",
                    "local_actual_draft_home_set_for_normal_draft",
                    "normal_draft_transfer_synced_expected_seq_ids",
                    "normal_draft_transfer_sender_seq_ids",
                    "normal_proposal_transfer_called",
                    "normal_proposal_transfer_zero_payload",
                    "normal_proposal_transfer_role",
                    "normal_proposal_transfer_meta_len",
                    "normal_proposal_transfer_payload_len",
                    "normal_proposal_transfer_next_collective_stage",
                    "dual_collective_stage_order",
                    "normal_proposal_transfer_enter",
                    "normal_proposal_transfer_exit",
                    "target_verify_result_transfer_enter",
                    "target_verify_result_transfer_exit",
                    "target_verify_result_transfer_meta_len",
                    "target_verify_result_transfer_payload_len",
                    "target_verify_result_transfer_num_results",
                    "target_verify_result_transfer_seq_ids",
                    "target_verify_result_transfer_zero_result_step",
                    "verify_result_numel",
                    "eager_transfer_enter",
                    "eager_transfer_exit",
                    "eager_result_transfer_enter",
                    "eager_result_transfer_exit",
                    "generic_full_continuous_stage_enter",
                    "generic_full_continuous_stage_exit",
                    "old_verify_result_transfer_used",
                    "full_continuous_enabled",
                    "generic_full_continuous_enabled",
                    "plan_id",
                    "dual_step_id",
                    "normal_transfer_called",
                    "normal_transfer_meta_len",
                    "normal_transfer_payload_len",
                    "next_collective_stage",
                    "proposal_buffer_hit_seq_ids",
                    "proposal_buffer_miss_seq_ids",
                    "missing_buffered_proposal_unexpected_seq_ids",
                }
            ):
                if key not in summary or value not in (None, [], {}, ""):
                    summary[key] = value
    return summary


def stage_order(record: dict[str, Any]) -> list[str]:
    raw_order = record.get("dual_collective_stage_order")
    if isinstance(raw_order, list):
        return [str(item) for item in raw_order]
    return []


def stage_indices(record: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    enter: dict[str, int] = {}
    exit_: dict[str, int] = {}
    for index, item in enumerate(stage_order(record)):
        if ":" not in item:
            continue
        stage, event = item.rsplit(":", 1)
        if stage not in DUAL_COLLECTIVE_STAGE_ORDER:
            continue
        if event == "enter" and stage not in enter:
            enter[stage] = index
        elif event == "exit" and stage not in exit_:
            exit_[stage] = index
    return enter, exit_


def has_stage_trace(record: dict[str, Any]) -> bool:
    if stage_order(record):
        return True
    return any(
        f"{stage}_{event}" in record
        for stage in DUAL_COLLECTIVE_STAGE_ORDER
        for event in ("enter", "exit")
    )


def stage_called(record: dict[str, Any], stage: str) -> bool:
    enter, exit_ = stage_indices(record)
    return (
        bool_value(record.get(f"{stage}_enter"))
        or bool_value(record.get(f"{stage}_exit"))
        or stage in enter
        or stage in exit_
    )


def first_enter_stage(record: dict[str, Any]) -> str | None:
    enter, _ = stage_indices(record)
    if not enter:
        return None
    return min(enter.items(), key=lambda item: item[1])[0]


def role_family(record: dict[str, Any]) -> str:
    role = str(record.get("normal_proposal_transfer_role") or record.get("runner_role") or "")
    if "draft" in role:
        return "draft"
    if "verify" in role or "target" in role or "aggregate" in role:
        return "verify"
    return role


def validate_dual_collective_stage_order(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    grouped: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for idx, record in enumerate(records):
        if not has_stage_trace(record):
            continue
        try:
            step_id = int(record.get("dual_step_id", record.get("iteration_id", -1)))
        except Exception:
            step_id = -1
        try:
            plan_id = int(record.get("plan_id", -1))
        except Exception:
            plan_id = -1
        grouped.setdefault((step_id, plan_id), []).append((idx, record))

        enter, exit_ = stage_indices(record)
        for stage in DUAL_COLLECTIVE_STAGE_ORDER:
            enter_bool = bool_value(record.get(f"{stage}_enter"))
            exit_bool = bool_value(record.get(f"{stage}_exit"))
            called = enter_bool or exit_bool or stage in enter or stage in exit_
            if not called:
                continue
            if enter_bool != exit_bool:
                errors.append(f"record[{idx}] stage {stage} enter/exit flags disagree")
            if stage in enter and stage not in exit_:
                errors.append(f"record[{idx}] stage {stage} has enter without exit in stage order")
            if stage in exit_ and stage not in enter:
                errors.append(f"record[{idx}] stage {stage} has exit without enter in stage order")
            if stage in enter and stage in exit_ and enter[stage] > exit_[stage]:
                errors.append(f"record[{idx}] stage {stage} exits before enter")

        for left_index, left_stage in enumerate(DUAL_COLLECTIVE_STAGE_ORDER):
            for right_stage in DUAL_COLLECTIVE_STAGE_ORDER[left_index + 1:]:
                if left_stage in enter and right_stage in enter and enter[left_stage] > enter[right_stage]:
                    errors.append(
                        f"record[{idx}] stage order violation: {left_stage} enters after {right_stage}"
                    )

        cached = bool_value(record.get("cached_admission_enabled"))
        full = bool_value(
            record.get("full_continuous_enabled")
            or record.get("generic_full_continuous_enabled")
            or record.get("enable_full_continuous_eager")
        )
        if cached and full and stage_order(record):
            first_stage = first_enter_stage(record)
            if first_stage != "normal_proposal_transfer":
                errors.append(
                    f"record[{idx}] cached full-continuous first collective stage must be "
                    f"normal_proposal_transfer, got {first_stage}"
                )

    for (step_id, plan_id), indexed_records in grouped.items():
        cached_full = any(
            bool_value(record.get("cached_admission_enabled"))
            and bool_value(
                record.get("full_continuous_enabled")
                or record.get("generic_full_continuous_enabled")
                or record.get("enable_full_continuous_eager")
            )
            for _, record in indexed_records
        )
        decode_loop_active = any(
            bool_value(record.get("cached_admission_decode_loop_active"))
            for _, record in indexed_records
        )
        if not cached_full or not decode_loop_active:
            continue
        called_stages = {
            stage
            for _, record in indexed_records
            for stage in DUAL_COLLECTIVE_STAGE_ORDER
            if stage_called(record, stage)
        }
        reached_post_normal_stage = bool(
            called_stages
            & {
                "eager_transfer",
                "eager_result_transfer",
                "generic_full_continuous_stage",
            }
        )
        if reached_post_normal_stage and "target_verify_result_transfer" not in called_stages:
            errors.append(
                f"dual_step_id={step_id} plan_id={plan_id} cached full-continuous reached "
                "post-normal stages without zero-safe target verify result stage"
            )
    return errors


def _target_verify_final_seq_ids(record: dict[str, Any]) -> list[int]:
    transferred = int_list(record.get("target_verify_result_transfer_seq_ids"))
    if transferred:
        return transferred
    return int_list(record.get("target_verify_seq_ids_after_received_proposal_override"))


def validate_target_tp_verify_seq_agreement(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    grouped: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for idx, record in enumerate(records):
        if role_family(record) != "verify":
            continue
        if not (
            bool_value(record.get("cached_admission_enabled"))
            and bool_value(record.get("cached_admission_decode_loop_active"))
            and bool_value(record.get("requires_framed_dual_verify_result_transfer"))
        ):
            continue
        has_target_tp_fields = any(
            key in record
            for key in (
                "target_verify_seq_ids_after_received_proposal_override",
                "target_tp_verify_seq_agreement_ok",
                "target_verify_result_transfer_seq_ids",
            )
        )
        if not has_target_tp_fields:
            continue
        step_id = int_value(record.get("dual_step_id", record.get("iteration_id", -1)), -1)
        plan_id = int_value(record.get("plan_id", -1), -1)
        grouped.setdefault((step_id, plan_id), []).append((idx, record))

    for (step_id, plan_id), indexed_records in grouped.items():
        final_by_record = [
            (idx, _target_verify_final_seq_ids(record))
            for idx, record in indexed_records
        ]
        non_empty_finals = [seq_ids for _, seq_ids in final_by_record if seq_ids]
        if non_empty_finals:
            expected = non_empty_finals[0]
            for idx, seq_ids in final_by_record:
                if seq_ids != expected:
                    errors.append(
                        f"dual_step_id={step_id} plan_id={plan_id} target TP verify seq divergence: "
                        f"record[{idx}] seq_ids={seq_ids}, expected={expected}"
                    )

        for idx, record in indexed_records:
            if "target_tp_verify_seq_agreement_ok" in record and not bool_value(
                record.get("target_tp_verify_seq_agreement_ok")
            ):
                errors.append(
                    f"record[{idx}] target TP verify seq agreement flag is false"
                )
            signatures = record.get("target_tp_verify_seq_agreement_signature")
            if isinstance(signatures, list) and signatures:
                normalized = [
                    int_list(signature)
                    for signature in signatures
                    if isinstance(signature, (list, tuple))
                ]
                if normalized and any(signature != normalized[0] for signature in normalized):
                    errors.append(
                        f"record[{idx}] target TP verify seq agreement signatures diverge: "
                        f"{normalized}"
                    )

            if "target_tp_buffer_seq_agreement_ok" in record and not bool_value(
                record.get("target_tp_buffer_seq_agreement_ok")
            ):
                errors.append(
                    f"record[{idx}] target TP buffer seq agreement flag is false"
                )
            buffer_signatures = record.get("target_tp_buffer_seq_agreement_signature")
            if isinstance(buffer_signatures, list) and buffer_signatures:
                normalized_buffer = [
                    int_list(signature)
                    for signature in buffer_signatures
                    if isinstance(signature, (list, tuple))
                ]
                if normalized_buffer and any(
                    signature != normalized_buffer[0] for signature in normalized_buffer
                ):
                    errors.append(
                        f"record[{idx}] target TP buffer seq agreement signatures diverge: "
                        f"{normalized_buffer}"
                    )

            if "target_tp_candidate_buffer_agreement_ok" in record and not bool_value(
                record.get("target_tp_candidate_buffer_agreement_ok")
            ):
                errors.append(
                    f"record[{idx}] target TP candidate buffer agreement flag is false"
                )
            candidate_signatures = record.get("target_tp_candidate_buffer_agreement_signatures")
            if isinstance(candidate_signatures, list) and candidate_signatures:
                normalized_candidate = [
                    int_list(signature)
                    for signature in candidate_signatures
                    if isinstance(signature, (list, tuple))
                ]
                if normalized_candidate and any(
                    signature != normalized_candidate[0]
                    for signature in normalized_candidate
                ):
                    errors.append(
                        f"record[{idx}] target TP candidate buffer agreement signatures diverge: "
                        f"{normalized_candidate}"
                    )

            plan_phase = str(record.get("plan_phase") or "")
            received = int_list(record.get("received_proposal_seq_ids"))
            after_override = int_list(
                record.get("target_verify_seq_ids_after_received_proposal_override")
            )
            raw_candidates = int_list(
                record.get("target_candidate_seq_ids_before_buffer_hit_agreement")
            )
            candidate_hit = int_list(record.get("target_candidate_buffer_hit_seq_ids"))
            candidate_miss = int_list(record.get("target_candidate_buffer_miss_seq_ids"))
            agreed_hit = int_list(record.get("target_candidate_buffer_hit_agreed_seq_ids"))
            agreed_miss = int_list(record.get("target_candidate_buffer_miss_agreed_seq_ids"))
            if raw_candidates and (candidate_hit or candidate_miss):
                partition = sorted(set(candidate_hit) | set(candidate_miss))
                if partition != sorted(set(raw_candidates)):
                    errors.append(
                        f"record[{idx}] target candidate buffer hit/miss partition mismatch: "
                        f"candidates={raw_candidates}, hit={candidate_hit}, miss={candidate_miss}"
                    )
            if agreed_hit or agreed_miss:
                agreed_partition = sorted(set(agreed_hit) | set(agreed_miss))
                if raw_candidates and agreed_partition != sorted(set(raw_candidates)):
                    errors.append(
                        f"record[{idx}] target agreed candidate hit/miss partition mismatch: "
                        f"candidates={raw_candidates}, agreed_hit={agreed_hit}, "
                        f"agreed_miss={agreed_miss}"
                    )
                final_not_agreed_hit = sorted(set(after_override) - set(agreed_hit))
                if final_not_agreed_hit:
                    errors.append(
                        f"record[{idx}] final target verify seqs must come from agreed "
                        f"candidate buffer hits: final={after_override}, "
                        f"agreed_hit={agreed_hit}, agreed_miss={agreed_miss}"
                    )
            from_received = int_list(record.get("target_verify_seq_ids_from_received_proposals"))
            priming_received = int_list(record.get("cached_admission_priming_received_seq_ids"))
            priming_buffered = int_list(record.get("cached_admission_priming_buffered_seq_ids"))
            suppressed = int_list(
                record.get("cached_admission_priming_same_step_verify_suppressed_seq_ids")
            )
            same_batch_received = int_list(record.get("fallback_same_batch_received_seq_ids"))
            same_batch_verify = int_list(record.get("fallback_same_batch_verify_seq_ids"))
            if plan_phase == "fallback" and received:
                legal_verify = same_batch_verify or from_received
                if same_batch_received and legal_verify and same_batch_received != legal_verify:
                    errors.append(
                        f"record[{idx}] fallback same-batch received seqs must match verify seqs: "
                        f"same_batch_received={same_batch_received}, legal_verify={legal_verify}"
                    )
                partition = sorted(set(same_batch_received) | set(suppressed))
                if partition and partition != sorted(set(received)):
                    errors.append(
                        f"record[{idx}] fallback received proposals must split into same-batch "
                        f"or priming-suppressed proposals: received={received}, "
                        f"same_batch={same_batch_received}, suppressed={suppressed}"
                    )
                if from_received and from_received != legal_verify:
                    errors.append(
                        f"record[{idx}] fallback target verify seq ids from received proposals "
                        f"must match legal same-batch proposals: from_received={from_received}, "
                        f"legal_verify={legal_verify}"
                    )
                if after_override != legal_verify:
                    errors.append(
                        f"record[{idx}] fallback final target verify seq ids must match legal "
                        f"same-batch proposals: after_override={after_override}, "
                        f"legal_verify={legal_verify}, received={received}"
                    )
                suppressed_still_verified = sorted(set(suppressed) & set(after_override))
                if suppressed_still_verified:
                    errors.append(
                        f"record[{idx}] cached-admission priming proposals must not be "
                        f"same-step target verified: {suppressed_still_verified}"
                    )
                if priming_received and sorted(set(priming_received)) != sorted(set(priming_buffered)):
                    errors.append(
                        f"record[{idx}] priming received proposals must be buffered: "
                        f"received={priming_received}, buffered={priming_buffered}"
                    )

        buffer_by_record: list[tuple[int, list[int]]] = []
        for idx, record in indexed_records:
            buffer_seq_ids = int_list(record.get("target_tp_buffer_seq_ids"))
            if not buffer_seq_ids and "target_tp_buffer_seq_ids" not in record:
                buffer_seq_ids = int_list(record.get("buffered_proposal_seq_ids"))
            if buffer_seq_ids or "target_tp_buffer_seq_ids" in record or "buffered_proposal_seq_ids" in record:
                buffer_by_record.append((idx, sorted(set(buffer_seq_ids))))
        if buffer_by_record:
            expected_buffer = buffer_by_record[0][1]
            for idx, buffer_seq_ids in buffer_by_record:
                if buffer_seq_ids != expected_buffer:
                    errors.append(
                        f"dual_step_id={step_id} plan_id={plan_id} target TP buffer seq divergence: "
                        f"record[{idx}] buffer_seq_ids={buffer_seq_ids}, expected={expected_buffer}"
                    )
            final_verify_seq_ids = sorted(
                set(seq_id for _, seq_ids in final_by_record for seq_id in seq_ids)
            )
            if final_verify_seq_ids:
                missing_from_buffer = sorted(set(final_verify_seq_ids) - set(expected_buffer))
                if missing_from_buffer:
                    errors.append(
                        f"dual_step_id={step_id} plan_id={plan_id} target verify seqs missing "
                        f"from agreed buffer: seq_ids={missing_from_buffer}, buffer={expected_buffer}"
                    )

        candidate_by_record: list[tuple[int, list[int], list[int], list[int]]] = []
        for idx, record in indexed_records:
            raw_candidates = int_list(
                record.get("target_candidate_seq_ids_before_buffer_hit_agreement")
            )
            hit_seq_ids = int_list(record.get("target_candidate_buffer_hit_seq_ids"))
            miss_seq_ids = int_list(record.get("target_candidate_buffer_miss_seq_ids"))
            if raw_candidates or hit_seq_ids or miss_seq_ids:
                candidate_by_record.append(
                    (
                        idx,
                        list(raw_candidates),
                        list(hit_seq_ids),
                        list(miss_seq_ids),
                    )
                )
        if candidate_by_record:
            _, expected_candidates, expected_hit, expected_miss = candidate_by_record[0]
            for idx, candidates, hit_seq_ids, miss_seq_ids in candidate_by_record:
                if (
                    candidates != expected_candidates
                    or hit_seq_ids != expected_hit
                    or miss_seq_ids != expected_miss
                ):
                    errors.append(
                        f"dual_step_id={step_id} plan_id={plan_id} target TP candidate "
                        f"buffer hit/miss divergence: record[{idx}] candidates={candidates}, "
                        f"hit={hit_seq_ids}, miss={miss_seq_ids}, "
                        f"expected_candidates={expected_candidates}, "
                        f"expected_hit={expected_hit}, expected_miss={expected_miss}"
                    )
    return errors


def validate(records: list[dict[str, Any]], result_payload: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    rows = result_rows(result_payload)
    summary = result_summary(result_payload)
    summary.update({key: value for key, value in trace_summary(records).items() if key not in summary})
    enabled = bool_value(summary.get("cached_admission_enabled"))

    if not enabled:
        leaked = [
            key for key, value in summary.items()
            if key.startswith("cached_admission_")
            and key not in {
                "cached_admission_enabled",
                "cached_admission_policy",
                "cached_admission_mode",
                "cached_admission_max_active",
                "cached_admission_total_requests",
            }
            and value not in (None, False, 0, 0.0, [], {})
        ]
        if leaked:
            errors.append(f"cached-admission fields nonzero while disabled: {sorted(leaked)}")
        return errors, summary

    max_active = int_value(summary.get("cached_admission_max_active"), 0)
    total_requests = int_value(summary.get("cached_admission_total_requests"), len(rows))
    total_arrived = int_value(summary.get("cached_admission_total_arrived"), 0)
    total_admitted = int_value(summary.get("cached_admission_total_admitted"), 0)
    total_completed = int_value(summary.get("cached_admission_total_completed"), 0)
    peak_active = int_value(summary.get("cached_admission_peak_active"), 0)

    if total_arrived > total_requests:
        errors.append("cached_admission_total_arrived must not exceed total_requests")
    if total_admitted > total_arrived:
        errors.append("cached_admission_total_admitted must not exceed total_arrived")
    if total_completed > total_admitted:
        errors.append("cached_admission_total_completed must not exceed total_admitted")
    if max_active > 0 and peak_active > max_active:
        errors.append("cached admission peak_active must not exceed max_active")
    if not bool_value(summary.get("cached_admission_prefill_compute_skipped")):
        errors.append("cached admission enabled requires cached_admission_prefill_compute_skipped=true")
    if summary.get("cached_admission_decode_only_elapsed_s") is None:
        errors.append("cached admission enabled requires decode-only elapsed time")

    cached_step_records = [
        record for record in records
        if "cached_admission_step" in record and str(record.get("runner_role")) in {"verify", "aggregate", "None"}
    ]
    if not cached_step_records:
        cached_step_records = [record for record in records if "cached_admission_step" in record]

    seen_admitted: set[str] = set()
    seen_completed: set[str] = set()
    for record in cached_step_records:
        for raw_id in record.get("cached_admission_admitted_request_ids", []) or []:
            request_id = str(raw_id)
            if request_id in seen_admitted:
                errors.append(f"request admitted more than once: {request_id}")
            seen_admitted.add(request_id)
        for raw_id in record.get("cached_admission_completed_request_ids", []) or []:
            seen_completed.add(str(raw_id))
        active_count = int_value(record.get("cached_admission_active_count"), 0)
        if max_active > 0 and active_count > max_active:
            errors.append("cached admission trace active_count exceeds max_active")

    for row in rows:
        request_id = str(row.get("request_id"))
        arrival_ts = float_value(row.get("arrival_ts"))
        admission_ts = float_value(row.get("admission_ts"), float_value(row.get("admit_ts")))
        decode_start_ts = float_value(row.get("decode_start_ts"))
        finish_ts = float_value(row.get("finish_ts"))
        queue_wait_ms = float_value(row.get("queue_wait_ms"))
        status = row.get("cached_admission_status")

        if admission_ts is not None and arrival_ts is not None and admission_ts < arrival_ts:
            errors.append(f"request {request_id} admitted before arrival")
        if queue_wait_ms is not None and queue_wait_ms < -1e-6:
            errors.append(f"request {request_id} has negative queue_wait_ms")
        if decode_start_ts is not None and admission_ts is not None and decode_start_ts < admission_ts:
            errors.append(f"request {request_id} decode_start_ts before admission_ts")
        if finish_ts is not None and decode_start_ts is not None and finish_ts < decode_start_ts:
            errors.append(f"request {request_id} finish_ts before decode_start_ts")
        if status == "completed" and admission_ts is None:
            errors.append(f"request {request_id} completed without admission")

    if seen_completed and not seen_completed.issubset(seen_admitted):
        missing = sorted(seen_completed - seen_admitted)
        errors.append(f"completed request was never admitted: {missing}")

    errors.extend(validate_dual_collective_stage_order(records))
    errors.extend(validate_target_tp_verify_seq_agreement(records))

    for idx, record in enumerate(records):
        execution_mode = str(record.get("execution_mode") or summary.get("execution_mode") or "")
        if execution_mode != "dual_batch_pearl":
            continue
        cached_full_record = bool_value(record.get("cached_admission_enabled")) and bool_value(
            record.get("full_continuous_enabled")
            or record.get("generic_full_continuous_enabled")
            or record.get("enable_full_continuous_eager")
        )
        decode_loop_active = bool_value(record.get("cached_admission_decode_loop_active"))
        if (
            cached_full_record
            and decode_loop_active
            and bool_value(record.get("old_verify_result_transfer_used"))
        ):
            errors.append(
                f"record[{idx}] old verify-result transfer is illegal under active cached full-continuous"
            )
        target_normal = set(int_list(record.get("target_normal_verify_seq_ids")))
        priming = set(int_list(record.get("cached_admission_draft_priming_seq_ids")))
        newly_admitted = set(int_list(record.get("cached_admission_newly_admitted_seq_ids")))
        filtered = set(int_list(record.get("cached_admission_unprimed_target_filtered_seq_ids")))
        primed = set(int_list(record.get("cached_admission_primed_seq_ids")))
        missing_after_filter = set(
            int_list(record.get("cached_admission_missing_proposal_after_filter_seq_ids"))
        )
        raw_target_before_filter = set(
            int_list(record.get("raw_target_normal_verify_seq_ids_before_buffer_filter"))
        )
        target_after_filter = int_list(record.get("target_normal_verify_seq_ids_after_buffer_filter"))
        target_filtered_missing = set(
            int_list(record.get("cached_admission_target_filtered_missing_proposal_seq_ids"))
        )
        target_buffer_hits = set(int_list(record.get("cached_admission_target_buffer_hit_seq_ids")))
        target_buffer_misses = set(int_list(record.get("cached_admission_target_buffer_miss_seq_ids")))
        missing_buffered_unexpected = set(
            int_list(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        )
        proposal_miss = set(int_list(record.get("proposal_buffer_miss_seq_ids")))
        allowed_missing = set(
            int_list(record.get("missing_buffered_proposal_allowed_by_eager_seq_ids"))
        )
        fallback_pending = set(int_list(record.get("fallback_pending_receive_seq_ids")))
        proposal_pre_verify = bool_map(record.get("proposal_pre_verify_by_seq_id"))
        target_seq_pre_verify = bool_map(record.get("target_seq_pre_verify_by_seq_id"))
        pre_verify_mismatch = set(int_list(record.get("pre_verify_mismatch_seq_ids")))
        pre_verify_stale_discarded = set(
            int_list(record.get("pre_verify_stale_proposal_discarded_seq_ids"))
        )
        pre_verify_redraft_required = set(
            int_list(record.get("pre_verify_redraft_required_seq_ids"))
        )
        actual_draft = set(
            int_list(record.get("actual_draft_home_set_for_normal_draft") or record.get("draft_home_set"))
        )
        actual_draft_ordered = int_list(
            record.get("actual_draft_home_set_for_normal_draft") or record.get("draft_home_set")
        )
        sent_proposals = int_list(record.get("dual_proposal_sent_seq_ids"))
        expected_receive = int_list(record.get("dual_proposal_expected_receive_seq_ids"))
        received_proposals = int_list(record.get("dual_proposal_received_seq_ids"))
        transfer_called = bool_value(record.get("normal_proposal_transfer_called"))
        transfer_zero_payload = bool_value(record.get("normal_proposal_transfer_zero_payload"))
        transfer_role = str(record.get("normal_proposal_transfer_role") or record.get("runner_role") or "")
        transfer_payload_len = int_value(record.get("normal_proposal_transfer_payload_len"), 0)
        sender_seq_ids = int_list(record.get("normal_draft_transfer_sender_seq_ids"))
        if not sender_seq_ids:
            sender_seq_ids = int_list(record.get("normal_draft_transfer_synced_expected_seq_ids"))

        if priming & target_normal:
            errors.append(
                f"record[{idx}] cached-admission priming seqs entered target verify: "
                f"{sorted(priming & target_normal)}"
            )
        if missing_after_filter:
            errors.append(
                f"record[{idx}] cached-admission missing proposal after filter: "
                f"{sorted(missing_after_filter)}"
            )
        if missing_buffered_unexpected:
            errors.append(
                f"record[{idx}] unexpected missing buffered proposals after target filter: "
                f"{sorted(missing_buffered_unexpected)}"
            )
        if target_filtered_missing & target_normal:
            errors.append(
                f"record[{idx}] filtered target proposal misses remained in target verify: "
                f"{sorted(target_filtered_missing & target_normal)}"
            )
        if target_after_filter and target_after_filter != int_list(record.get("target_normal_verify_seq_ids")):
            errors.append(
                f"record[{idx}] target verify after buffer filter must match final target verify: "
                f"after={target_after_filter}, final={int_list(record.get('target_normal_verify_seq_ids'))}"
            )
        common_pre_verify_seq_ids = set(proposal_pre_verify) & set(target_seq_pre_verify)
        observed_pre_verify_mismatch = {
            seq_id
            for seq_id in common_pre_verify_seq_ids
            if bool(proposal_pre_verify[seq_id]) != bool(target_seq_pre_verify[seq_id])
        }
        if observed_pre_verify_mismatch and observed_pre_verify_mismatch != pre_verify_mismatch:
            errors.append(
                f"record[{idx}] pre_verify mismatch trace does not match observed proposal/target "
                f"state: observed={sorted(observed_pre_verify_mismatch)}, "
                f"traced={sorted(pre_verify_mismatch)}"
            )
        if pre_verify_mismatch & target_normal:
            errors.append(
                f"record[{idx}] stale pre_verify-mismatched proposals entered target verify: "
                f"{sorted(pre_verify_mismatch & target_normal)}"
            )
        if pre_verify_stale_discarded & target_normal:
            errors.append(
                f"record[{idx}] stale pre_verify proposals marked discarded still entered target verify: "
                f"{sorted(pre_verify_stale_discarded & target_normal)}"
            )
        if pre_verify_mismatch and not pre_verify_mismatch <= (
            pre_verify_stale_discarded | pre_verify_redraft_required
        ):
            errors.append(
                f"record[{idx}] pre_verify mismatches must be discarded or redraft-required: "
                f"{sorted(pre_verify_mismatch - (pre_verify_stale_discarded | pre_verify_redraft_required))}"
            )
        target_miss_still_selected = (target_buffer_misses & target_normal) - allowed_missing - fallback_pending
        if target_miss_still_selected:
            errors.append(
                f"record[{idx}] target normal verify kept proposal-buffer misses: "
                f"{sorted(target_miss_still_selected)}"
            )
        if target_buffer_hits:
            target_without_buffer_hit = target_normal - target_buffer_hits - allowed_missing - fallback_pending
            if target_without_buffer_hit:
                errors.append(
                    f"record[{idx}] target normal verify must be backed by proposal-buffer hits: "
                    f"target_without_hit={sorted(target_without_buffer_hit)}, "
                    f"raw_target={sorted(raw_target_before_filter)}"
                )
        unexpected_missing = (proposal_miss & target_normal) - allowed_missing - fallback_pending
        if unexpected_missing:
            errors.append(
                f"record[{idx}] target normal verify missing buffered proposals after cached filter: "
                f"{sorted(unexpected_missing)}"
            )
        if filtered and not filtered <= (priming | newly_admitted):
            errors.append(
                f"record[{idx}] filtered cached-admission seqs are not newly admitted or priming: "
                f"{sorted(filtered - (priming | newly_admitted))}"
            )
        if priming and not priming <= actual_draft:
            errors.append(
                f"record[{idx}] cached-admission priming seqs not routed to normal draft: "
                f"{sorted(priming - actual_draft)}"
            )
        if primed and primed & target_normal:
            errors.append(
                f"record[{idx}] primed seqs should not target-verify in the same priming record: "
                f"{sorted(primed & target_normal)}"
            )
        if transfer_called:
            is_draft_transfer_record = transfer_role == "draft" or "draft" in str(record.get("runner_role") or "")
            duplicate_sender = sorted({seq_id for seq_id in sender_seq_ids if sender_seq_ids.count(seq_id) > 1})
            if duplicate_sender:
                errors.append(
                    f"record[{idx}] normal proposal transfer sender seqs contain duplicates: "
                    f"{duplicate_sender}"
                )
            if transfer_zero_payload:
                if sender_seq_ids or sent_proposals or expected_receive or received_proposals:
                    errors.append(
                        f"record[{idx}] zero-payload normal proposal transfer must have empty seq ids: "
                        f"sender={sender_seq_ids}, sent={sent_proposals}, "
                        f"expected={expected_receive}, received={received_proposals}"
                    )
            else:
                if transfer_payload_len <= 0:
                    errors.append(
                        f"record[{idx}] nonzero normal proposal transfer must have payload_len > 0"
                    )
                if sender_seq_ids and sent_proposals != sender_seq_ids:
                    errors.append(
                        f"record[{idx}] dual proposal sent seqs must match sender seqs: "
                        f"sent={sent_proposals}, sender={sender_seq_ids}"
                    )
                if sender_seq_ids and expected_receive != sender_seq_ids:
                    errors.append(
                        f"record[{idx}] dual proposal expected receive seqs must match sender seqs: "
                        f"expected={expected_receive}, sender={sender_seq_ids}"
                    )
                if sender_seq_ids and received_proposals and received_proposals != sender_seq_ids:
                    errors.append(
                        f"record[{idx}] dual proposal received seqs must match sender seqs: "
                        f"received={received_proposals}, sender={sender_seq_ids}"
                    )
                if sender_seq_ids and not received_proposals and not is_draft_transfer_record:
                    errors.append(
                        f"record[{idx}] target normal proposal transfer must record received seqs: "
                        f"sender={sender_seq_ids}"
                    )

    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    fields = (
        "cached_admission_enabled",
        "cached_admission_policy",
        "cached_admission_mode",
        "cached_admission_max_active",
        "cached_admission_total_requests",
        "cached_admission_total_arrived",
        "cached_admission_total_admitted",
        "cached_admission_total_completed",
        "cached_admission_peak_active",
        "cached_admission_mean_queue_wait_ms",
        "cached_admission_p90_queue_wait_ms",
        "cached_admission_prefill_compute_skipped",
        "cached_admission_decode_only_elapsed_s",
        "cached_admission_newly_admitted_seq_ids",
        "cached_admission_draft_priming_seq_ids",
        "cached_admission_primed_seq_ids",
        "cached_admission_unprimed_target_filtered_seq_ids",
        "cached_admission_missing_proposal_after_filter_seq_ids",
        "cached_admission_filtered_draft_seq_ids",
        "raw_target_normal_verify_seq_ids_before_buffer_filter",
        "cached_admission_target_filtered_missing_proposal_seq_ids",
        "cached_admission_target_buffer_hit_seq_ids",
        "cached_admission_target_buffer_miss_seq_ids",
        "target_normal_verify_seq_ids_after_buffer_filter",
        "proposal_pre_verify_by_seq_id",
        "target_seq_pre_verify_by_seq_id",
        "pre_verify_mismatch_seq_ids",
        "pre_verify_stale_proposal_discarded_seq_ids",
        "pre_verify_redraft_required_seq_ids",
        "warmup_mode",
        "cached_admission_decode_loop_active",
        "requires_framed_dual_verify_result_transfer",
        "cached_active_stage_debug_enabled",
        "dual_stage_rank",
        "dual_stage_tp_local_rank",
        "active_cached_stage_order",
        "active_cached_verify_result_transfer_sent",
        "active_cached_verify_result_transfer_received",
        "active_cached_eager_transfer_sent",
        "active_cached_eager_transfer_received",
        "active_cached_eager_result_sent",
        "active_cached_eager_result_received",
        "target_verify_seq_ids_before_received_proposal_override",
        "target_verify_seq_ids_from_received_proposals",
        "target_verify_seq_ids_after_received_proposal_override",
        "target_tp_verify_seq_agreement_ok",
        "target_tp_verify_seq_agreement_signature",
        "target_tp_buffer_seq_agreement_ok",
        "target_tp_buffer_seq_agreement_signature",
        "target_tp_buffer_seq_ids",
        "target_candidate_seq_ids_before_buffer_hit_agreement",
        "target_candidate_buffer_hit_seq_ids",
        "target_candidate_buffer_miss_seq_ids",
        "target_tp_candidate_buffer_agreement_ok",
        "target_tp_candidate_buffer_agreement_signatures",
        "target_candidate_buffer_hit_agreed_seq_ids",
        "target_candidate_buffer_miss_agreed_seq_ids",
        "dual_buffer_mutation_events",
        "received_proposal_seq_ids",
        "cached_admission_priming_received_seq_ids",
        "cached_admission_priming_buffered_seq_ids",
        "cached_admission_priming_same_step_verify_suppressed_seq_ids",
        "fallback_same_batch_received_seq_ids",
        "fallback_same_batch_verify_seq_ids",
        "local_actual_draft_home_set_for_normal_draft",
        "normal_draft_transfer_synced_expected_seq_ids",
        "normal_draft_transfer_sender_seq_ids",
        "normal_proposal_transfer_called",
        "normal_proposal_transfer_zero_payload",
        "normal_proposal_transfer_role",
        "normal_proposal_transfer_meta_len",
        "normal_proposal_transfer_payload_len",
        "normal_proposal_transfer_next_collective_stage",
        "dual_collective_stage_order",
        "normal_proposal_transfer_enter",
        "normal_proposal_transfer_exit",
        "target_verify_result_transfer_enter",
        "target_verify_result_transfer_exit",
        "target_verify_result_transfer_meta_len",
        "target_verify_result_transfer_payload_len",
        "target_verify_result_transfer_num_results",
        "target_verify_result_transfer_seq_ids",
        "target_verify_result_transfer_zero_result_step",
        "verify_result_numel",
        "eager_transfer_enter",
        "eager_transfer_exit",
        "eager_result_transfer_enter",
        "eager_result_transfer_exit",
        "generic_full_continuous_stage_enter",
        "generic_full_continuous_stage_exit",
        "old_verify_result_transfer_used",
        "full_continuous_enabled",
        "generic_full_continuous_enabled",
        "plan_id",
        "dual_step_id",
        "normal_transfer_called",
        "normal_transfer_meta_len",
        "normal_transfer_payload_len",
        "next_collective_stage",
        "proposal_buffer_hit_seq_ids",
        "proposal_buffer_miss_seq_ids",
        "missing_buffered_proposal_unexpected_seq_ids",
        "dual_proposal_sent_seq_ids",
        "dual_proposal_expected_receive_seq_ids",
        "dual_proposal_received_seq_ids",
    )
    for field in fields:
        print(f"{field} = {summary.get(field)}")


def synthetic_payload(
    *,
    enabled: bool = True,
    bad_admit_before_arrival: bool = False,
    negative_wait: bool = False,
    exceed_cap: bool = False,
    duplicate_admit: bool = False,
    completed_without_admit: bool = False,
    prefill_not_skipped: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return [], {"args": {"cached_admission": False}, "cached_admission": {"cached_admission_enabled": False}}

    rows = [
        {
            "request_id": "r0",
            "arrival_ts": 10.0,
            "admission_ts": 10.01,
            "decode_start_ts": 10.01,
            "finish_ts": 10.05,
            "queue_wait_ms": 10.0,
            "cached_admission_status": "completed",
        },
        {
            "request_id": "r1",
            "arrival_ts": 10.02,
            "admission_ts": 10.06,
            "decode_start_ts": 10.06,
            "finish_ts": 10.10,
            "queue_wait_ms": 40.0,
            "cached_admission_status": "completed",
        },
    ]
    if bad_admit_before_arrival:
        rows[0]["admission_ts"] = 9.9
    if negative_wait:
        rows[0]["queue_wait_ms"] = -1.0
    if completed_without_admit:
        rows[1].pop("admission_ts")

    admitted = ["r0", "r1"]
    if duplicate_admit:
        admitted.append("r0")
    records = [
        {
            "cached_admission_enabled": True,
            "cached_admission_step": 0,
            "cached_admission_admitted_request_ids": admitted,
            "cached_admission_completed_request_ids": ["r0", "r1"],
            "cached_admission_active_count": 3 if exceed_cap else 2,
        }
    ]
    result = {
        "args": {"cached_admission": True},
        "cached_admission": {
            "cached_admission_enabled": True,
            "cached_admission_policy": "fifo",
            "cached_admission_mode": "in_memory_kv",
            "cached_admission_max_active": 2,
            "cached_admission_total_requests": 2,
            "cached_admission_total_arrived": 2,
            "cached_admission_total_admitted": 2,
            "cached_admission_total_completed": 2,
            "cached_admission_peak_active": 3 if exceed_cap else 2,
            "cached_admission_mean_queue_wait_ms": 25.0,
            "cached_admission_p50_queue_wait_ms": 25.0,
            "cached_admission_p90_queue_wait_ms": 37.0,
            "cached_admission_p99_queue_wait_ms": 39.7,
            "cached_admission_prefill_compute_skipped": not prefill_not_skipped,
            "cached_admission_decode_only_elapsed_s": 0.1,
        },
        "traces": rows,
    }
    return records, result


def synthetic_dual_payload(
    *,
    execution_mode: str = "dual_batch_pearl",
    newly_admitted_seq_ids: list[int] | None = None,
    priming_seq_ids: list[int] | None = None,
    primed_seq_ids: list[int] | None = None,
    filtered_seq_ids: list[int] | None = None,
    target_normal_verify_seq_ids: list[int] | None = None,
    actual_draft_seq_ids: list[int] | None = None,
    proposal_hit_seq_ids: list[int] | None = None,
    proposal_miss_seq_ids: list[int] | None = None,
    missing_after_filter_seq_ids: list[int] | None = None,
    missing_buffered_unexpected_seq_ids: list[int] | None = None,
    raw_target_before_filter_seq_ids: list[int] | None = None,
    target_filtered_missing_seq_ids: list[int] | None = None,
    target_buffer_hit_seq_ids: list[int] | None = None,
    target_buffer_miss_seq_ids: list[int] | None = None,
    target_after_filter_seq_ids: list[int] | None = None,
    proposal_pre_verify_by_seq_id: dict[int, bool] | None = None,
    target_seq_pre_verify_by_seq_id: dict[int, bool] | None = None,
    pre_verify_mismatch_seq_ids: list[int] | None = None,
    pre_verify_stale_proposal_discarded_seq_ids: list[int] | None = None,
    pre_verify_redraft_required_seq_ids: list[int] | None = None,
    warmup_mode: bool = False,
    cached_admission_decode_loop_active: bool = False,
    requires_framed_dual_verify_result_transfer: bool = False,
    original_draft_seq_ids: list[int] | None = None,
    sent_proposal_seq_ids: list[int] | None = None,
    expected_receive_seq_ids: list[int] | None = None,
    received_proposal_seq_ids: list[int] | None = None,
    synced_expected_receive_seq_ids: list[int] | None = None,
    sender_seq_ids: list[int] | None = None,
    normal_proposal_transfer_called: bool = False,
    normal_proposal_transfer_zero_payload: bool = False,
    normal_proposal_transfer_role: str = "verify",
    normal_proposal_transfer_payload_len: int | None = None,
    filtered_draft_seq_ids: list[int] | None = None,
    fallback_pending_receive_seq_ids: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, result = synthetic_payload()
    result.setdefault("args", {})["execution_mode"] = execution_mode
    records.append(
        {
            "execution_mode": execution_mode,
            "cached_admission_enabled": True,
            "runner_role": "verify",
            "target_normal_verify_seq_ids": target_normal_verify_seq_ids or [],
            "original_draft_home_set": original_draft_seq_ids or actual_draft_seq_ids or [],
            "actual_draft_home_set_for_normal_draft": actual_draft_seq_ids or [],
            "cached_admission_filtered_draft_seq_ids": filtered_draft_seq_ids or [],
            "cached_admission_newly_admitted_seq_ids": newly_admitted_seq_ids or [],
            "cached_admission_draft_priming_seq_ids": priming_seq_ids or [],
            "cached_admission_primed_seq_ids": primed_seq_ids or [],
            "cached_admission_unprimed_target_filtered_seq_ids": filtered_seq_ids or [],
            "cached_admission_missing_proposal_after_filter_seq_ids": (
                missing_after_filter_seq_ids or []
            ),
            "proposal_buffer_hit_seq_ids": proposal_hit_seq_ids or [],
            "proposal_buffer_miss_seq_ids": proposal_miss_seq_ids or [],
            "raw_target_normal_verify_seq_ids_before_buffer_filter": (
                raw_target_before_filter_seq_ids or []
            ),
            "cached_admission_target_filtered_missing_proposal_seq_ids": (
                target_filtered_missing_seq_ids or []
            ),
            "cached_admission_target_buffer_hit_seq_ids": target_buffer_hit_seq_ids or [],
            "cached_admission_target_buffer_miss_seq_ids": target_buffer_miss_seq_ids or [],
            "target_normal_verify_seq_ids_after_buffer_filter": target_after_filter_seq_ids or [],
            "proposal_pre_verify_by_seq_id": {
                str(seq_id): bool(value)
                for seq_id, value in sorted((proposal_pre_verify_by_seq_id or {}).items())
            },
            "target_seq_pre_verify_by_seq_id": {
                str(seq_id): bool(value)
                for seq_id, value in sorted((target_seq_pre_verify_by_seq_id or {}).items())
            },
            "pre_verify_mismatch_seq_ids": pre_verify_mismatch_seq_ids or [],
            "pre_verify_stale_proposal_discarded_seq_ids": (
                pre_verify_stale_proposal_discarded_seq_ids or []
            ),
            "pre_verify_redraft_required_seq_ids": pre_verify_redraft_required_seq_ids or [],
            "warmup_mode": bool(warmup_mode),
            "cached_admission_decode_loop_active": bool(cached_admission_decode_loop_active),
            "requires_framed_dual_verify_result_transfer": bool(
                requires_framed_dual_verify_result_transfer
            ),
            "local_actual_draft_home_set_for_normal_draft": actual_draft_seq_ids or [],
            "normal_draft_transfer_synced_expected_seq_ids": synced_expected_receive_seq_ids or [],
            "normal_draft_transfer_sender_seq_ids": sender_seq_ids or [],
            "dual_proposal_sent_seq_ids": sent_proposal_seq_ids or [],
            "dual_proposal_expected_receive_seq_ids": expected_receive_seq_ids or [],
            "dual_proposal_received_seq_ids": received_proposal_seq_ids or [],
            "normal_proposal_transfer_called": bool(normal_proposal_transfer_called),
            "normal_proposal_transfer_zero_payload": bool(normal_proposal_transfer_zero_payload),
            "normal_proposal_transfer_role": str(normal_proposal_transfer_role),
            "normal_proposal_transfer_meta_len": 5 if normal_proposal_transfer_called else 0,
            "normal_proposal_transfer_payload_len": (
                0
                if normal_proposal_transfer_zero_payload
                else (
                    int(normal_proposal_transfer_payload_len)
                    if normal_proposal_transfer_payload_len is not None
                    else (1 if normal_proposal_transfer_called else 0)
                )
            ),
            "normal_proposal_transfer_next_collective_stage": (
                "synthetic_next" if normal_proposal_transfer_called else None
            ),
            "dual_step_id": 0,
            "normal_transfer_called": bool(normal_proposal_transfer_called),
            "normal_transfer_meta_len": 5 if normal_proposal_transfer_called else 0,
            "normal_transfer_payload_len": (
                0
                if normal_proposal_transfer_zero_payload
                else (
                    int(normal_proposal_transfer_payload_len)
                    if normal_proposal_transfer_payload_len is not None
                    else (1 if normal_proposal_transfer_called else 0)
                )
            ),
            "next_collective_stage": (
                "synthetic_next" if normal_proposal_transfer_called else None
            ),
            "missing_buffered_proposal_allowed_by_eager_seq_ids": [],
            "missing_buffered_proposal_unexpected_seq_ids": (
                missing_buffered_unexpected_seq_ids or []
            ),
            "fallback_pending_receive_seq_ids": fallback_pending_receive_seq_ids or [],
        }
    )
    return records, result


def stage_trace_record(
    *,
    runner_role: str,
    dual_step_id: int = 0,
    plan_id: int = 0,
    plan_phase: str = "steady",
    full_continuous_enabled: bool = True,
    order: list[str] | None = None,
    verify_payload_len: int = 0,
    verify_num_results: int = 0,
    verify_seq_ids: list[int] | None = None,
    target_seq_ids_before_override: list[int] | None = None,
    target_seq_ids_from_received: list[int] | None = None,
    target_seq_ids_after_override: list[int] | None = None,
    target_tp_agreement_ok: bool = True,
    target_tp_agreement_signature: list[list[int]] | None = None,
    target_tp_buffer_agreement_ok: bool = True,
    target_tp_buffer_agreement_signature: list[list[int]] | None = None,
    target_tp_buffer_seq_ids: list[int] | None = None,
    buffered_proposal_seq_ids: list[int] | None = None,
    target_candidate_seq_ids_before_buffer_hit_agreement: list[int] | None = None,
    target_candidate_buffer_hit_seq_ids: list[int] | None = None,
    target_candidate_buffer_miss_seq_ids: list[int] | None = None,
    target_tp_candidate_buffer_agreement_ok: bool = True,
    target_tp_candidate_buffer_agreement_signatures: list[list[int]] | None = None,
    target_candidate_buffer_hit_agreed_seq_ids: list[int] | None = None,
    target_candidate_buffer_miss_agreed_seq_ids: list[int] | None = None,
    received_proposal_seq_ids: list[int] | None = None,
    cached_admission_draft_priming_seq_ids: list[int] | None = None,
    cached_admission_unprimed_target_filtered_seq_ids: list[int] | None = None,
    cached_admission_priming_received_seq_ids: list[int] | None = None,
    cached_admission_priming_buffered_seq_ids: list[int] | None = None,
    cached_admission_priming_same_step_verify_suppressed_seq_ids: list[int] | None = None,
    fallback_same_batch_received_seq_ids: list[int] | None = None,
    fallback_same_batch_verify_seq_ids: list[int] | None = None,
    actual_draft_seq_ids: list[int] | None = None,
    dual_stage_rank: int = 0,
    dual_stage_tp_local_rank: int = 0,
    normal_zero_payload: bool = True,
    old_verify_result_transfer_used: bool = False,
    cached_admission_decode_loop_active: bool = False,
    requires_framed_dual_verify_result_transfer: bool = False,
) -> dict[str, Any]:
    order = order or []
    enter_stages = {
        item.rsplit(":", 1)[0]
        for item in order
        if isinstance(item, str) and item.endswith(":enter") and ":" in item
    }
    exit_stages = {
        item.rsplit(":", 1)[0]
        for item in order
        if isinstance(item, str) and item.endswith(":exit") and ":" in item
    }
    record: dict[str, Any] = {
        "execution_mode": "dual_batch_pearl",
        "cached_admission_enabled": True,
        "cached_admission_decode_loop_active": bool(cached_admission_decode_loop_active),
        "requires_framed_dual_verify_result_transfer": bool(
            requires_framed_dual_verify_result_transfer
        ),
        "cached_active_stage_debug_enabled": False,
        "dual_stage_rank": int(dual_stage_rank),
        "dual_stage_tp_local_rank": int(dual_stage_tp_local_rank),
        "active_cached_stage_order": list(order),
        "active_cached_verify_result_transfer_sent": (
            runner_role == "target"
            and "target_verify_result_transfer" in enter_stages
        ),
        "active_cached_verify_result_transfer_received": (
            runner_role == "draft"
            and "target_verify_result_transfer" in enter_stages
        ),
        "active_cached_eager_transfer_sent": (
            runner_role == "draft"
            and "eager_transfer" in enter_stages
        ),
        "active_cached_eager_transfer_received": (
            runner_role == "target"
            and "eager_transfer" in enter_stages
        ),
        "active_cached_eager_result_sent": (
            runner_role == "target"
            and "eager_result_transfer" in enter_stages
        ),
        "active_cached_eager_result_received": (
            runner_role == "draft"
            and "eager_result_transfer" in enter_stages
        ),
        "target_verify_seq_ids_before_received_proposal_override": (
            target_seq_ids_before_override or []
        ),
        "target_verify_seq_ids_from_received_proposals": (
            target_seq_ids_from_received or []
        ),
        "target_verify_seq_ids_after_received_proposal_override": (
            target_seq_ids_after_override
            if target_seq_ids_after_override is not None
            else (verify_seq_ids or [])
        ),
        "target_tp_verify_seq_agreement_ok": bool(target_tp_agreement_ok),
        "target_tp_verify_seq_agreement_signature": target_tp_agreement_signature or [],
        "received_proposal_seq_ids": received_proposal_seq_ids or [],
        "cached_admission_draft_priming_seq_ids": (
            cached_admission_draft_priming_seq_ids or []
        ),
        "cached_admission_unprimed_target_filtered_seq_ids": (
            cached_admission_unprimed_target_filtered_seq_ids or []
        ),
        "cached_admission_priming_received_seq_ids": (
            cached_admission_priming_received_seq_ids or []
        ),
        "cached_admission_priming_buffered_seq_ids": (
            cached_admission_priming_buffered_seq_ids or []
        ),
        "cached_admission_priming_same_step_verify_suppressed_seq_ids": (
            cached_admission_priming_same_step_verify_suppressed_seq_ids or []
        ),
        "fallback_same_batch_received_seq_ids": fallback_same_batch_received_seq_ids or [],
        "fallback_same_batch_verify_seq_ids": fallback_same_batch_verify_seq_ids or [],
        "full_continuous_enabled": bool(full_continuous_enabled),
        "generic_full_continuous_enabled": bool(full_continuous_enabled),
        "enable_full_continuous_eager": bool(full_continuous_enabled),
        "runner_role": runner_role,
        "normal_proposal_transfer_role": "draft" if "draft" in runner_role else "verify",
        "dual_step_id": int(dual_step_id),
        "plan_id": int(plan_id),
        "plan_phase": str(plan_phase),
        "target_normal_verify_seq_ids": [],
        "actual_draft_home_set_for_normal_draft": actual_draft_seq_ids or [],
        "proposal_buffer_hit_seq_ids": [],
        "proposal_buffer_miss_seq_ids": [],
        "normal_proposal_transfer_called": "normal_proposal_transfer" in enter_stages,
        "normal_proposal_transfer_zero_payload": bool(normal_zero_payload),
        "normal_proposal_transfer_meta_len": 5 if "normal_proposal_transfer" in enter_stages else 0,
        "normal_proposal_transfer_payload_len": 0 if normal_zero_payload else 1,
        "normal_proposal_transfer_next_collective_stage": "target_verify_result_transfer",
        "dual_collective_stage_order": list(order),
        "target_verify_result_transfer_meta_len": (
            6 if "target_verify_result_transfer" in enter_stages else 0
        ),
        "target_verify_result_transfer_payload_len": int(verify_payload_len),
        "target_verify_result_transfer_num_results": int(verify_num_results),
        "target_verify_result_transfer_seq_ids": verify_seq_ids or [],
        "target_verify_result_transfer_zero_result_step": int(verify_num_results == 0),
        "verify_result_numel": int(4 * verify_num_results),
        "old_verify_result_transfer_used": bool(old_verify_result_transfer_used),
        "dual_proposal_sent_seq_ids": [],
        "dual_proposal_expected_receive_seq_ids": [],
        "dual_proposal_received_seq_ids": [],
        "missing_buffered_proposal_allowed_by_eager_seq_ids": [],
        "missing_buffered_proposal_unexpected_seq_ids": [],
    }
    if (
        target_tp_buffer_agreement_signature is not None
        or target_tp_buffer_seq_ids is not None
        or buffered_proposal_seq_ids is not None
        or not target_tp_buffer_agreement_ok
    ):
        record["target_tp_buffer_seq_agreement_ok"] = bool(target_tp_buffer_agreement_ok)
    if target_tp_buffer_agreement_signature is not None:
        record["target_tp_buffer_seq_agreement_signature"] = target_tp_buffer_agreement_signature
    if target_tp_buffer_seq_ids is not None:
        record["target_tp_buffer_seq_ids"] = target_tp_buffer_seq_ids
    if buffered_proposal_seq_ids is not None:
        record["buffered_proposal_seq_ids"] = buffered_proposal_seq_ids
    if (
        target_candidate_seq_ids_before_buffer_hit_agreement is not None
        or target_candidate_buffer_hit_seq_ids is not None
        or target_candidate_buffer_miss_seq_ids is not None
        or target_tp_candidate_buffer_agreement_signatures is not None
        or target_candidate_buffer_hit_agreed_seq_ids is not None
        or target_candidate_buffer_miss_agreed_seq_ids is not None
        or not target_tp_candidate_buffer_agreement_ok
    ):
        record["target_tp_candidate_buffer_agreement_ok"] = bool(
            target_tp_candidate_buffer_agreement_ok
        )
    if target_candidate_seq_ids_before_buffer_hit_agreement is not None:
        record["target_candidate_seq_ids_before_buffer_hit_agreement"] = (
            target_candidate_seq_ids_before_buffer_hit_agreement
        )
    if target_candidate_buffer_hit_seq_ids is not None:
        record["target_candidate_buffer_hit_seq_ids"] = target_candidate_buffer_hit_seq_ids
    if target_candidate_buffer_miss_seq_ids is not None:
        record["target_candidate_buffer_miss_seq_ids"] = target_candidate_buffer_miss_seq_ids
    if target_tp_candidate_buffer_agreement_signatures is not None:
        record["target_tp_candidate_buffer_agreement_signatures"] = (
            target_tp_candidate_buffer_agreement_signatures
        )
    if target_candidate_buffer_hit_agreed_seq_ids is not None:
        record["target_candidate_buffer_hit_agreed_seq_ids"] = (
            target_candidate_buffer_hit_agreed_seq_ids
        )
    if target_candidate_buffer_miss_agreed_seq_ids is not None:
        record["target_candidate_buffer_miss_agreed_seq_ids"] = (
            target_candidate_buffer_miss_agreed_seq_ids
        )
    for stage in DUAL_COLLECTIVE_STAGE_ORDER:
        record[f"{stage}_enter"] = stage in enter_stages
        record[f"{stage}_exit"] = stage in exit_stages
    return record


def synthetic_stage_payload(stage_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, result = synthetic_payload()
    result.setdefault("args", {})["execution_mode"] = "dual_batch_pearl"
    records.extend(stage_records)
    return records, result


def run_synthetic() -> int:
    legal_depth4_order = [
        "normal_proposal_transfer:enter",
        "normal_proposal_transfer:exit",
        "target_verify_result_transfer:enter",
        "target_verify_result_transfer:exit",
        "eager_transfer:enter",
        "eager_transfer:exit",
        "eager_result_transfer:enter",
        "eager_result_transfer:exit",
    ]
    legal_full_order = legal_depth4_order + [
        "generic_full_continuous_stage:enter",
        "generic_full_continuous_stage:exit",
    ]
    cases = [
        ("disabled", synthetic_payload(enabled=False), False),
        ("fifo", synthetic_payload(), False),
        ("bad_admit_before_arrival", synthetic_payload(bad_admit_before_arrival=True), True),
        ("negative_wait", synthetic_payload(negative_wait=True), True),
        ("exceed_cap", synthetic_payload(exceed_cap=True), True),
        ("duplicate_admit", synthetic_payload(duplicate_admit=True), True),
        ("completed_without_admit", synthetic_payload(completed_without_admit=True), True),
        ("prefill_not_skipped", synthetic_payload(prefill_not_skipped=True), True),
        (
            "dual_new_admit_filtered_until_primed",
            synthetic_dual_payload(
                newly_admitted_seq_ids=[6],
                priming_seq_ids=[6],
                primed_seq_ids=[6],
                filtered_seq_ids=[6],
                target_normal_verify_seq_ids=[4],
                actual_draft_seq_ids=[6],
                proposal_hit_seq_ids=[4],
            ),
            False,
        ),
        (
            "dual_new_admit_unprimed_target_verify",
            synthetic_dual_payload(
                newly_admitted_seq_ids=[6],
                priming_seq_ids=[6],
                target_normal_verify_seq_ids=[4, 6],
                actual_draft_seq_ids=[],
                proposal_hit_seq_ids=[4],
                proposal_miss_seq_ids=[6],
                missing_after_filter_seq_ids=[6],
            ),
            True,
        ),
        (
            "dual_target_verify_after_buffered",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[6],
                actual_draft_seq_ids=[],
                proposal_hit_seq_ids=[6],
            ),
            False,
        ),
        (
            "dual_target_filter_missing_proposal_good",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[6],
                actual_draft_seq_ids=[5],
                proposal_hit_seq_ids=[6],
                raw_target_before_filter_seq_ids=[4, 6],
                target_filtered_missing_seq_ids=[4],
                target_buffer_hit_seq_ids=[6],
                target_buffer_miss_seq_ids=[4],
                target_after_filter_seq_ids=[6],
            ),
            False,
        ),
        (
            "dual_target_filter_missing_proposal_bad_final_keeps_miss",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[4, 6],
                actual_draft_seq_ids=[5],
                proposal_hit_seq_ids=[6],
                proposal_miss_seq_ids=[4],
                missing_buffered_unexpected_seq_ids=[4],
                raw_target_before_filter_seq_ids=[4, 6],
                target_filtered_missing_seq_ids=[4],
                target_buffer_hit_seq_ids=[6],
                target_buffer_miss_seq_ids=[4],
                target_after_filter_seq_ids=[4, 6],
            ),
            True,
        ),
        (
            "dual_target_filter_all_missing_good",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[],
                actual_draft_seq_ids=[5],
                raw_target_before_filter_seq_ids=[4],
                target_filtered_missing_seq_ids=[4],
                target_buffer_miss_seq_ids=[4],
                target_after_filter_seq_ids=[],
            ),
            False,
        ),
        (
            "dual_fallback_same_batch_missing_proposal_allowed",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[4],
                actual_draft_seq_ids=[4],
                proposal_miss_seq_ids=[4],
                raw_target_before_filter_seq_ids=[4],
                target_after_filter_seq_ids=[4],
                fallback_pending_receive_seq_ids=[4],
            ),
            False,
        ),
        (
            "dual_transfer_sender_synced_expected_legal_divergent_local_draft",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[4, 6],
                proposal_hit_seq_ids=[4, 6],
                target_buffer_hit_seq_ids=[4, 6],
                actual_draft_seq_ids=[7],
                original_draft_seq_ids=[5, 7],
                sent_proposal_seq_ids=[5, 7],
                sender_seq_ids=[5, 7],
                expected_receive_seq_ids=[5, 7],
                received_proposal_seq_ids=[5, 7],
                normal_proposal_transfer_called=True,
            ),
            False,
        ),
        (
            "dual_transfer_sender_synced_expected_mismatch_bad",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[4, 6],
                proposal_hit_seq_ids=[4, 6],
                target_buffer_hit_seq_ids=[4, 6],
                actual_draft_seq_ids=[7],
                original_draft_seq_ids=[5, 7],
                sent_proposal_seq_ids=[5, 7],
                sender_seq_ids=[5, 7],
                expected_receive_seq_ids=[5, 7],
                received_proposal_seq_ids=[7],
                normal_proposal_transfer_called=True,
            ),
            True,
        ),
        (
            "dual_transfer_zero_proposal_step",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[],
                actual_draft_seq_ids=[],
                sent_proposal_seq_ids=[],
                sender_seq_ids=[],
                expected_receive_seq_ids=[],
                received_proposal_seq_ids=[],
                normal_proposal_transfer_called=True,
                normal_proposal_transfer_zero_payload=True,
            ),
            False,
        ),
        (
            "dual_draft_transfer_sent_only",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[],
                actual_draft_seq_ids=[5, 7],
                sent_proposal_seq_ids=[5, 7],
                sender_seq_ids=[5, 7],
                expected_receive_seq_ids=[5, 7],
                received_proposal_seq_ids=[],
                normal_proposal_transfer_called=True,
                normal_proposal_transfer_role="draft",
            ),
            False,
        ),
        (
            "dual_non_transfer_record_inherited_sender_seq_ids_ignored",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[6],
                proposal_hit_seq_ids=[6],
                target_buffer_hit_seq_ids=[6],
                sender_seq_ids=[32, 33],
                sent_proposal_seq_ids=[],
                expected_receive_seq_ids=[],
                received_proposal_seq_ids=[],
                normal_proposal_transfer_called=False,
            ),
            False,
        ),
        (
            "non_dual_no_priming_requirement",
            synthetic_dual_payload(
                execution_mode="parallel_pearl",
                newly_admitted_seq_ids=[6],
                priming_seq_ids=[6],
                target_normal_verify_seq_ids=[6],
                proposal_miss_seq_ids=[6],
                missing_after_filter_seq_ids=[6],
            ),
            False,
        ),
        (
            "dual_sender_receiver_actual_draft_alignment_bad",
            synthetic_dual_payload(
                original_draft_seq_ids=[5, 7],
                actual_draft_seq_ids=[7],
                sent_proposal_seq_ids=[5, 7],
                sender_seq_ids=[7],
                expected_receive_seq_ids=[7],
                received_proposal_seq_ids=[5, 7],
                filtered_draft_seq_ids=[5],
                normal_proposal_transfer_called=True,
            ),
            True,
        ),
        (
            "dual_sender_receiver_actual_draft_alignment_good",
            synthetic_dual_payload(
                original_draft_seq_ids=[5, 7],
                actual_draft_seq_ids=[7],
                sent_proposal_seq_ids=[7],
                sender_seq_ids=[7],
                expected_receive_seq_ids=[7],
                received_proposal_seq_ids=[7],
                filtered_draft_seq_ids=[5],
                normal_proposal_transfer_called=True,
            ),
            False,
        ),
        (
            "pre_verify_aligned_target_verify",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[0],
                proposal_hit_seq_ids=[0],
                target_buffer_hit_seq_ids=[0],
                target_after_filter_seq_ids=[0],
                proposal_pre_verify_by_seq_id={0: True},
                target_seq_pre_verify_by_seq_id={0: True},
            ),
            False,
        ),
        (
            "pre_verify_stale_proposal_discarded",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[],
                proposal_hit_seq_ids=[0],
                target_buffer_hit_seq_ids=[0],
                target_after_filter_seq_ids=[],
                proposal_pre_verify_by_seq_id={0: False},
                target_seq_pre_verify_by_seq_id={0: True},
                pre_verify_mismatch_seq_ids=[0],
                pre_verify_stale_proposal_discarded_seq_ids=[0],
                pre_verify_redraft_required_seq_ids=[0],
            ),
            False,
        ),
        (
            "pre_verify_stale_proposal_used_for_target_verify_bad",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[0],
                proposal_hit_seq_ids=[0],
                target_buffer_hit_seq_ids=[0],
                target_after_filter_seq_ids=[0],
                proposal_pre_verify_by_seq_id={0: False},
                target_seq_pre_verify_by_seq_id={0: True},
                pre_verify_mismatch_seq_ids=[0],
                pre_verify_stale_proposal_discarded_seq_ids=[0],
                pre_verify_redraft_required_seq_ids=[0],
            ),
            True,
        ),
        (
            "pre_verify_warmup_aligned_target_verify",
            synthetic_dual_payload(
                target_normal_verify_seq_ids=[0],
                proposal_hit_seq_ids=[0],
                target_buffer_hit_seq_ids=[0],
                target_after_filter_seq_ids=[0],
                proposal_pre_verify_by_seq_id={0: True},
                target_seq_pre_verify_by_seq_id={0: True},
                warmup_mode=True,
            ),
            False,
        ),
        (
            "cached_depth4_legal_stage_order",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft",
                        full_continuous_enabled=False,
                        order=legal_depth4_order,
                    ),
                    stage_trace_record(
                        runner_role="verify",
                        full_continuous_enabled=False,
                        order=legal_depth4_order,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_transfer_only_record",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="dual_draft_transfer",
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                        ],
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_warmup_old_verify_result_path_allowed",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft_apply_verify",
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                            "eager_transfer:enter",
                            "eager_transfer:exit",
                            "eager_result_transfer:enter",
                            "eager_result_transfer:exit",
                            "generic_full_continuous_stage:enter",
                            "generic_full_continuous_stage:exit",
                        ],
                        old_verify_result_transfer_used=True,
                        cached_admission_decode_loop_active=False,
                        requires_framed_dual_verify_result_transfer=False,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_legal_stage_order",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft",
                        order=legal_full_order,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="verify",
                        order=legal_full_order,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_tp_received_seq_agree",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_before_override=[4],
                        target_seq_ids_from_received=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        received_proposal_seq_ids=[4],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_before_override=[],
                        target_seq_ids_from_received=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        received_proposal_seq_ids=[4],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_tp_zero_seq_agree",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        received_proposal_seq_ids=[],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        received_proposal_seq_ids=[],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_tp_buffer_agree_45",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=62,
                        plan_id=77,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_agreement_signature=[[2, 9, 14], [2, 9, 14]],
                        target_tp_buffer_seq_ids=[4, 5],
                        buffered_proposal_seq_ids=[4, 5],
                        received_proposal_seq_ids=[5],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=62,
                        plan_id=77,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_agreement_signature=[[2, 9, 14], [2, 9, 14]],
                        target_tp_buffer_seq_ids=[4, 5],
                        buffered_proposal_seq_ids=[4, 5],
                        received_proposal_seq_ids=[5],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_tp_buffer_divergence_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=62,
                        plan_id=77,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_agreement_signature=[[2, 9, 14], [1, 5, 5]],
                        target_tp_buffer_seq_ids=[4, 5],
                        buffered_proposal_seq_ids=[4, 5],
                        received_proposal_seq_ids=[5],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=62,
                        plan_id=77,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_agreement_ok=False,
                        target_tp_buffer_agreement_signature=[[2, 9, 14], [1, 5, 5]],
                        target_tp_buffer_seq_ids=[5],
                        buffered_proposal_seq_ids=[5],
                        received_proposal_seq_ids=[5],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_target_tp_buffer_agree_missing_skip",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=63,
                        plan_id=78,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        target_tp_buffer_agreement_signature=[[1, 5, 5], [1, 5, 5]],
                        target_tp_buffer_seq_ids=[5],
                        buffered_proposal_seq_ids=[5],
                        received_proposal_seq_ids=[5],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=63,
                        plan_id=78,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        target_tp_buffer_agreement_signature=[[1, 5, 5], [1, 5, 5]],
                        target_tp_buffer_seq_ids=[5],
                        buffered_proposal_seq_ids=[5],
                        received_proposal_seq_ids=[5],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_tp_buffer_agree_verify_4",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=64,
                        plan_id=79,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_seq_ids=[4],
                        buffered_proposal_seq_ids=[4],
                        received_proposal_seq_ids=[],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=64,
                        plan_id=79,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_tp_buffer_seq_ids=[4],
                        buffered_proposal_seq_ids=[4],
                        received_proposal_seq_ids=[],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_candidate_buffer_hit_agree",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=65,
                        plan_id=80,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=65,
                        plan_id=80,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_candidate_buffer_hit_divergence_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=66,
                        plan_id=81,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [0, 0, 0]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 0, 0, 0, 1, 4, 4],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=66,
                        plan_id=81,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[1, 4, 4], [0, 0, 0]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[],
                        target_candidate_buffer_miss_seq_ids=[4],
                        target_tp_candidate_buffer_agreement_ok=False,
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 0, 0, 0, 1, 4, 4],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[],
                        target_candidate_buffer_miss_agreed_seq_ids=[4],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_target_candidate_buffer_miss_agree_skip",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=67,
                        plan_id=82,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[],
                        target_candidate_buffer_miss_seq_ids=[4],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 0, 0, 0, 1, 4, 4],
                            [1, 4, 4, 0, 0, 0, 1, 4, 4],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[],
                        target_candidate_buffer_miss_agreed_seq_ids=[4],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=67,
                        plan_id=82,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[],
                        target_candidate_buffer_miss_seq_ids=[4],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 0, 0, 0, 1, 4, 4],
                            [1, 4, 4, 0, 0, 0, 1, 4, 4],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[],
                        target_candidate_buffer_miss_agreed_seq_ids=[4],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_candidate_buffer_hit_agree_verify_4",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=68,
                        plan_id=83,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=68,
                        plan_id=83,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_candidate_agree_final_diverge_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=69,
                        plan_id=84,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [0, 0, 0]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=69,
                        plan_id=84,
                        plan_phase="steady",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_ok=False,
                        target_tp_agreement_signature=[[1, 4, 4], [0, 0, 0]],
                        target_candidate_seq_ids_before_buffer_hit_agreement=[4],
                        target_candidate_buffer_hit_seq_ids=[4],
                        target_candidate_buffer_miss_seq_ids=[],
                        target_tp_candidate_buffer_agreement_signatures=[
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                            [1, 4, 4, 1, 4, 4, 0, 0, 0],
                        ],
                        target_candidate_buffer_hit_agreed_seq_ids=[4],
                        target_candidate_buffer_miss_agreed_seq_ids=[],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_fallback_priming_received_suppressed",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=1,
                        plan_id=16,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_before_override=[4],
                        target_seq_ids_from_received=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        received_proposal_seq_ids=[4],
                        cached_admission_draft_priming_seq_ids=[4],
                        cached_admission_unprimed_target_filtered_seq_ids=[4],
                        cached_admission_priming_received_seq_ids=[4],
                        cached_admission_priming_buffered_seq_ids=[4],
                        cached_admission_priming_same_step_verify_suppressed_seq_ids=[4],
                        fallback_same_batch_received_seq_ids=[],
                        fallback_same_batch_verify_seq_ids=[],
                        actual_draft_seq_ids=[4],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=1,
                        plan_id=16,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_before_override=[],
                        target_seq_ids_from_received=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        received_proposal_seq_ids=[4],
                        cached_admission_draft_priming_seq_ids=[4],
                        cached_admission_unprimed_target_filtered_seq_ids=[4],
                        cached_admission_priming_received_seq_ids=[4],
                        cached_admission_priming_buffered_seq_ids=[4],
                        cached_admission_priming_same_step_verify_suppressed_seq_ids=[4],
                        fallback_same_batch_received_seq_ids=[],
                        fallback_same_batch_verify_seq_ids=[],
                        actual_draft_seq_ids=[4],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_fallback_priming_same_step_verify_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=1,
                        plan_id=16,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_before_override=[4],
                        target_seq_ids_from_received=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [1, 4, 4]],
                        received_proposal_seq_ids=[4],
                        cached_admission_draft_priming_seq_ids=[4],
                        cached_admission_unprimed_target_filtered_seq_ids=[4],
                        cached_admission_priming_received_seq_ids=[4],
                        cached_admission_priming_buffered_seq_ids=[4],
                        cached_admission_priming_same_step_verify_suppressed_seq_ids=[4],
                        fallback_same_batch_received_seq_ids=[],
                        fallback_same_batch_verify_seq_ids=[],
                        actual_draft_seq_ids=[4],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_fallback_legal_same_batch_received",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=1,
                        plan_id=16,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[5],
                        target_seq_ids_before_override=[5],
                        target_seq_ids_from_received=[5],
                        target_seq_ids_after_override=[5],
                        target_tp_agreement_signature=[[1, 5, 5], [1, 5, 5]],
                        received_proposal_seq_ids=[5],
                        fallback_same_batch_received_seq_ids=[5],
                        fallback_same_batch_verify_seq_ids=[5],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=1,
                        plan_id=16,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[5],
                        target_seq_ids_before_override=[],
                        target_seq_ids_from_received=[5],
                        target_seq_ids_after_override=[5],
                        target_tp_agreement_signature=[[1, 5, 5], [1, 5, 5]],
                        received_proposal_seq_ids=[5],
                        fallback_same_batch_received_seq_ids=[5],
                        fallback_same_batch_verify_seq_ids=[5],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_target_tp_seq_divergence_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=5,
                        verify_num_results=1,
                        verify_seq_ids=[4],
                        target_seq_ids_from_received=[4],
                        target_seq_ids_after_override=[4],
                        target_tp_agreement_signature=[[1, 4, 4], [0, 0, 0]],
                        received_proposal_seq_ids=[4],
                        dual_stage_rank=1,
                        dual_stage_tp_local_rank=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_from_received=[],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_ok=False,
                        target_tp_agreement_signature=[[1, 4, 4], [0, 0, 0]],
                        received_proposal_seq_ids=[],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_fallback_received_not_final_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="target",
                        dual_step_id=2,
                        plan_id=17,
                        plan_phase="fallback",
                        order=legal_full_order,
                        verify_payload_len=0,
                        verify_num_results=0,
                        verify_seq_ids=[],
                        target_seq_ids_from_received=[4],
                        target_seq_ids_after_override=[],
                        target_tp_agreement_signature=[[0, 0, 0], [0, 0, 0]],
                        received_proposal_seq_ids=[4],
                        dual_stage_rank=2,
                        dual_stage_tp_local_rank=1,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_divergent_first_stage_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft",
                        order=[
                            "target_verify_result_transfer:enter",
                            "target_verify_result_transfer:exit",
                            "eager_transfer:enter",
                            "eager_transfer:exit",
                        ],
                    ),
                    stage_trace_record(runner_role="verify", order=legal_full_order),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_missing_verify_stage_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft",
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                            "eager_transfer:enter",
                            "eager_transfer:exit",
                            "eager_result_transfer:enter",
                            "eager_result_transfer:exit",
                            "generic_full_continuous_stage:enter",
                            "generic_full_continuous_stage:exit",
                        ],
                    ),
                    stage_trace_record(
                        runner_role="verify",
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                            "eager_transfer:enter",
                            "eager_transfer:exit",
                            "eager_result_transfer:enter",
                            "eager_result_transfer:exit",
                            "generic_full_continuous_stage:enter",
                            "generic_full_continuous_stage:exit",
                        ],
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_old_verify_result_path_bad",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft_apply_verify",
                        order=legal_full_order,
                        old_verify_result_transfer_used=True,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            True,
        ),
        (
            "cached_full_continuous_tp1_mixed_records",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft_apply_verify",
                        dual_step_id=17,
                        plan_id=23,
                        order=legal_full_order,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="dual_draft_transfer",
                        dual_step_id=17,
                        plan_id=23,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                        ],
                    ),
                    stage_trace_record(
                        runner_role="dual_draft_transfer",
                        dual_step_id=17,
                        plan_id=23,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                        ],
                    ),
                ]
            ),
            False,
        ),
        (
            "cached_full_continuous_zero_target_verify_stage",
            synthetic_stage_payload(
                [
                    stage_trace_record(
                        runner_role="draft",
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                            "target_verify_result_transfer:enter",
                            "target_verify_result_transfer:exit",
                        ],
                        verify_payload_len=0,
                        verify_num_results=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                    stage_trace_record(
                        runner_role="verify",
                        order=[
                            "normal_proposal_transfer:enter",
                            "normal_proposal_transfer:exit",
                            "target_verify_result_transfer:enter",
                            "target_verify_result_transfer:exit",
                        ],
                        verify_payload_len=0,
                        verify_num_results=0,
                        cached_admission_decode_loop_active=True,
                        requires_framed_dual_verify_result_transfer=True,
                    ),
                ]
            ),
            False,
        ),
    ]
    for name, (records, result), should_fail in cases:
        errors, _ = validate(records, result)
        failed = bool(errors)
        if failed != should_fail:
            print(f"synthetic case {name} expected fail={should_fail}, got errors={errors}")
            return 1
    print("Synthetic cached admission checks passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H cached admission trace/result fields.")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("result", nargs="?")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        return run_synthetic()
    if not args.trace or not args.result:
        parser.error("TRACE and RESULT are required unless --synthetic is used")

    records = load_trace_records(Path(args.trace))
    result_payload = load_json(Path(args.result))
    errors, summary = validate(records, result_payload)
    print_summary(summary)
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nCached admission check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
