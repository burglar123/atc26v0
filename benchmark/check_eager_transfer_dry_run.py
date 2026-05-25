#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from nano_pearl.pearl_engine.dual_batch import (  # type: ignore  # noqa: E402
        EAGER_STATE_READY_TO_VERIFY,
        EagerProposal,
        deserialize_eager_transfer_payload,
        serialize_eager_transfer_payload,
    )
except ModuleNotFoundError:
    from benchmark.check_eager_scaffold import (  # type: ignore  # noqa: E402
        EAGER_STATE_READY_TO_VERIFY,
        EagerProposal,
        deserialize_eager_transfer_payload,
        serialize_eager_transfer_payload,
    )


ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]

DROP_REASONS = {
    "seq_not_found",
    "seq_not_found_later",
    "seq_not_running",
    "seq_finished",
    "seq_finished_before_base",
    "seq_pre_verify_before_base",
    "seq_returned_pre_verify_before_base",
    "base_overshot",
    "base_overshot_later",
    "base_overshot_or_stale",
    "base_pre_verify_not_supported",
    "proposal_len_mismatch",
    "to_verify_len_mismatch",
    "proposal_token_len_mismatch",
    "parent_dependency_invalidated",
}

PENDING_REASONS = {"pending_base_not_reached", "base_not_reached"}


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError(
        "Unsupported trace format. Expected a raw list or a dict with "
        "'traces', 'records', or 'trace_records'."
    )


def as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(item) for item in value}


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def dict_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def is_dual_record(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    )


def transfer_step_key(record: dict[str, Any]) -> tuple[int, int]:
    plan_id = int_value(record.get("eager_transfer_plan_id"), int_value(record.get("plan_id"), -1))
    step_id = int_value(record.get("eager_transfer_step_id"), int_value(record.get("step_id"), -1))
    return plan_id, step_id


def median_int(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    transfer_records = 0
    sent_proposal_ids: set[int] = set()
    received_proposal_ids: set[int] = set()
    validated_proposal_ids: set[int] = set()
    pending_proposal_ids: set[int] = set()
    ready_proposal_ids: set[int] = set()
    dropped_proposal_ids: set[int] = set()
    pending_reason_counts: Counter[str] = Counter()
    drop_reason_counts: Counter[str] = Counter()
    generated_tokens = 0
    promoted_tokens = 0
    discarded_tokens = 0
    transferred_tokens = 0
    transfer_pending_tokens = 0
    transfer_validated_tokens = 0
    transfer_dropped_tokens = 0
    target_eager_non_empty_count = 0
    verified_counter_rows = 0
    base_mismatch_count = 0
    base_overshot_count = 0
    bad_sent_received_steps = 0
    pending_base_deltas: list[int] = []
    transfer_by_step: dict[tuple[int, int], dict[str, Any]] = {}

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        plan_enabled = bool(record.get("enable_eager_plan_dry_run", False))
        draft_enabled = bool(record.get("enable_eager_draft_dry_run", False))
        promotion_enabled = bool(record.get("enable_eager_promotion_dry_run", False))
        transfer_enabled = bool(record.get("enable_eager_transfer_dry_run", False))
        transfer_active = bool(record.get("eager_transfer_dry_run_enabled", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        sent_ids = as_int_set(record.get("eager_transfer_sent_proposal_ids"))
        received_ids = as_int_set(record.get("eager_transfer_received_proposal_ids"))
        validated_ids = as_int_set(record.get("eager_transfer_validated_proposal_ids"))
        pending_ids = as_int_set(record.get("eager_transfer_pending_proposal_ids"))
        dropped_ids = as_int_set(record.get("eager_transfer_dropped_proposal_ids"))
        pending_ready_ids = as_int_set(record.get("eager_pending_ready_proposal_ids"))
        pending_dropped_ids = as_int_set(record.get("eager_pending_dropped_proposal_ids"))

        if target_eager:
            target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_verified = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_verified:
            verified_counter_rows += 1
            errors.append(f"record[{idx}] eager verification counters must stay zero: {nonzero_verified}")

        generated = int_value(record.get("eager_tokens_generated"), 0)
        promoted = int_value(record.get("eager_tokens_promoted"), 0)
        discarded = int_value(record.get("eager_tokens_discarded"), 0)
        transferred = int_value(record.get("eager_tokens_transferred"), 0)
        transfer_pending = int_value(record.get("eager_tokens_transfer_pending"), 0)
        transfer_validated = int_value(record.get("eager_tokens_transfer_validated"), 0)
        transfer_dropped = int_value(record.get("eager_tokens_transfer_dropped"), 0)
        generated_tokens += generated
        promoted_tokens += promoted
        discarded_tokens += discarded
        transferred_tokens += transferred
        transfer_pending_tokens += transfer_pending
        transfer_validated_tokens += transfer_validated
        transfer_dropped_tokens += transfer_dropped

        if not plan_enabled and not draft_enabled and not promotion_enabled and not transfer_enabled:
            if (
                draft_eager
                or generated
                or promoted
                or discarded
                or transferred
                or transfer_pending
                or transfer_validated
                or transfer_dropped
            ):
                errors.append(f"record[{idx}] default run has eager activity")
            continue

        if plan_enabled and not draft_enabled and not promotion_enabled and not transfer_enabled:
            if generated or promoted or discarded or transferred or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] plan dry-run has eager execution/transfer counters")
            continue

        if draft_enabled and not promotion_enabled and not transfer_enabled:
            if transferred or promoted or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] draft dry-run has promotion/transfer counters")
            continue

        if promotion_enabled and not transfer_enabled:
            if transferred or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] promotion dry-run has transfer counters")
            continue

        if transfer_enabled:
            if not (plan_enabled and draft_enabled and promotion_enabled):
                errors.append(f"record[{idx}] transfer dry-run must imply plan/draft/promotion dry-runs")
            if transfer_active:
                transfer_records += 1

        if not transfer_active:
            if transferred or transfer_pending or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] transfer counters outside active transfer dry-run row")
            continue

        key = transfer_step_key(record)
        step = transfer_by_step.setdefault(
            key,
            {
                "sent": set(),
                "received": set(),
                "validated": set(),
                "pending": set(),
                "dropped": set(),
                "num_proposals": 0,
                "promoted_tokens": 0,
                "transferred_tokens": 0,
            },
        )
        num_proposals = int_value(record.get("eager_transfer_num_proposals"), 0)
        if num_proposals != len(sent_ids) and sent_ids:
            errors.append(f"record[{idx}] eager_transfer_num_proposals does not match sent ids")
        if num_proposals != len(received_ids) and received_ids:
            errors.append(f"record[{idx}] eager_transfer_num_proposals does not match received ids")

        sent_proposal_ids.update(sent_ids)
        received_proposal_ids.update(received_ids)
        validated_proposal_ids.update(validated_ids)
        pending_proposal_ids.update(pending_ids)
        ready_proposal_ids.update(pending_ready_ids)
        dropped_proposal_ids.update(dropped_ids)
        dropped_proposal_ids.update(pending_dropped_ids)
        step["sent"].update(sent_ids)
        step["received"].update(received_ids)
        step["validated"].update(validated_ids)
        step["pending"].update(pending_ids)
        step["dropped"].update(dropped_ids)
        step["num_proposals"] = max(int(step["num_proposals"]), num_proposals)
        step["promoted_tokens"] += promoted
        step["transferred_tokens"] += transferred

        received_classified = validated_ids | pending_ids | dropped_ids
        if received_ids and received_classified != received_ids:
            errors.append(
                f"record[{idx}] validated+pending+dropped proposal ids must classify received ids: "
                f"classified={sorted(received_classified)}, received={sorted(received_ids)}"
            )
        overlapping = (validated_ids & pending_ids) | (validated_ids & dropped_ids) | (pending_ids & dropped_ids)
        if overlapping:
            errors.append(f"record[{idx}] proposal ids in multiple receive classes: {sorted(overlapping)}")

        if transferred and transferred != promoted:
            errors.append(
                f"record[{idx}] eager_tokens_transferred must equal eager_tokens_promoted on draft send rows: "
                f"{transferred} != {promoted}"
            )
        if int_value(record.get("draft_eager_buffer_size_after_transfer"), 0) != 0:
            errors.append(f"record[{idx}] draft_eager_buffer_size_after_transfer must be 0")
        if int_value(record.get("target_eager_buffer_size_after_clear"), 0) < 0:
            errors.append(f"record[{idx}] target eager buffer size cannot be negative")
        if int_value(record.get("eager_pending_buffer_size_after_clear"), 0) < 0:
            errors.append(f"record[{idx}] pending eager buffer size cannot be negative")
        if int_value(record.get("eager_pending_buffer_size_after_clear"), 0) > (
            int_value(record.get("eager_pending_buffer_size_before_update"), 0) + len(pending_ids)
        ):
            errors.append(
                f"record[{idx}] pending eager buffer grew by more than newly pending proposals"
            )

        base_match = record.get("eager_transfer_base_match_by_seq_id", {})
        base_pre_verify = record.get("eager_transfer_base_pre_verify_by_seq_id", {})
        current_len_by_seq = record.get("eager_transfer_current_len_by_seq_id", {})
        base_len_by_seq = record.get("eager_transfer_base_len_by_seq_id", {})
        base_delta_by_seq = record.get("eager_pending_base_delta_by_seq_id", {})
        proposal_len_by_id = record.get("eager_transfer_proposal_len_by_proposal_id", {})
        to_verify_len_by_id = record.get("eager_transfer_to_verify_len_by_proposal_id", {})
        drop_reasons = record.get("eager_transfer_drop_reason_by_proposal_id", {})
        pending_drop_reasons = record.get("eager_pending_drop_reason_by_proposal_id", {})
        pending_states = record.get("eager_pending_state_by_proposal_id", {})
        gamma = int_value(record.get("normal_gamma"), None)

        proposal_to_seq = {
            proposal_id: seq_id
            for proposal_id, seq_id in zip(
                as_int_list(record.get("eager_transfer_received_proposal_ids")),
                as_int_list(record.get("eager_transfer_received_seq_ids")),
            )
        }
        present_seq_ids = (
            as_int_set(record.get("scheduled_seq_ids"))
            | as_int_set(record.get("resolved_seq_ids"))
            | as_int_set(record.get("target_home_set"))
        )
        for proposal_id in sorted(validated_ids):
            seq_id = proposal_to_seq.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} missing seq id")
                continue
            if dict_get(base_match, seq_id) is not True:
                base_mismatch_count += 1
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} lacks base_len match")
            if dict_get(base_pre_verify, seq_id) is not False:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} base_pre_verify is not false")
            if gamma is not None and int_value(dict_get(proposal_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and int_value(dict_get(to_verify_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} to_verify_len != gamma")

        for proposal_id in sorted(pending_ids):
            seq_id = proposal_to_seq.get(proposal_id)
            reason = dict_get(pending_states, proposal_id, "pending_base_not_reached")
            pending_reason_counts[str(reason)] += 1
            if reason not in PENDING_REASONS:
                errors.append(f"record[{idx}] pending proposal_id={proposal_id} has unexpected status={reason!r}")
            if seq_id is None:
                errors.append(f"record[{idx}] pending proposal_id={proposal_id} missing seq id")
                continue
            current_len = int_value(dict_get(current_len_by_seq, seq_id), -1)
            base_len = int_value(dict_get(base_len_by_seq, seq_id), -1)
            delta = int_value(dict_get(base_delta_by_seq, seq_id), base_len - current_len)
            if delta > 0:
                pending_base_deltas.append(delta)
            if not (current_len >= 0 and base_len > current_len):
                errors.append(
                    f"record[{idx}] pending proposal_id={proposal_id} must have current_len < base_len: "
                    f"current_len={current_len}, base_len={base_len}"
                )
            if dict_get(base_pre_verify, seq_id) is not False:
                errors.append(f"record[{idx}] pending proposal_id={proposal_id} base_pre_verify is not false")
            if gamma is not None and int_value(dict_get(proposal_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] pending proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and int_value(dict_get(to_verify_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] pending proposal_id={proposal_id} to_verify_len != gamma")

        pending_ready_seq_ids = as_int_list(record.get("eager_pending_ready_seq_ids"))
        for proposal_id, seq_id in zip(as_int_list(record.get("eager_pending_ready_proposal_ids")), pending_ready_seq_ids):
            ready_proposal_ids.add(int(proposal_id))
            if dict_get(base_match, seq_id) is not True:
                base_mismatch_count += 1
                errors.append(f"record[{idx}] pending-ready proposal_id={proposal_id} lacks base_len match")
            if dict_get(base_pre_verify, seq_id) is not False:
                errors.append(f"record[{idx}] pending-ready proposal_id={proposal_id} base_pre_verify is not false")

        for proposal_id in sorted(dropped_ids):
            reason = dict_get(drop_reasons, proposal_id)
            drop_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] dropped proposal_id={proposal_id} missing drop reason")
            elif reason not in DROP_REASONS:
                errors.append(f"record[{idx}] dropped proposal_id={proposal_id} has unexpected reason={reason!r}")
            if reason in {"base_overshot", "base_overshot_later", "base_overshot_or_stale"}:
                base_overshot_count += 1
            seq_id = proposal_to_seq.get(proposal_id)
            if reason == "seq_not_running" and seq_id in present_seq_ids:
                errors.append(
                    f"record[{idx}] proposal_id={proposal_id} used seq_not_running for present seq_id={seq_id}"
                )
            if seq_id is not None and dict_get(base_match, seq_id) is False:
                current_len = int_value(dict_get(current_len_by_seq, seq_id), -1)
                base_len = int_value(dict_get(base_len_by_seq, seq_id), -1)
                if current_len > base_len:
                    base_overshot_count += 1
                elif reason not in {"base_overshot", "base_overshot_later", "base_overshot_or_stale"}:
                    base_mismatch_count += 1

        for proposal_id in sorted(pending_dropped_ids):
            reason = dict_get(pending_drop_reasons, proposal_id)
            drop_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] pending-dropped proposal_id={proposal_id} missing drop reason")
            elif reason not in DROP_REASONS:
                errors.append(f"record[{idx}] pending-dropped proposal_id={proposal_id} has unexpected reason={reason!r}")
            if reason in {"base_overshot", "base_overshot_later", "base_overshot_or_stale"}:
                base_overshot_count += 1

    for key, step in sorted(transfer_by_step.items()):
        sent = step["sent"]
        received = step["received"]
        classified = step["validated"] | step["pending"] | step["dropped"]
        if sent != received:
            bad_sent_received_steps += 1
            errors.append(
                f"transfer_step{key} received proposal ids must match sent proposal ids: "
                f"sent={sorted(sent)}, received={sorted(received)}"
            )
        if classified != received:
            errors.append(
                f"transfer_step{key} validated+pending+dropped proposal ids must classify received ids: "
                f"classified={sorted(classified)}, received={sorted(received)}"
            )
        if step["transferred_tokens"] and step["transferred_tokens"] != step["promoted_tokens"]:
            errors.append(
                f"transfer_step{key} transferred tokens must equal promoted tokens: "
                f"{step['transferred_tokens']} != {step['promoted_tokens']}"
            )

    summary = {
        "total_trace_records": len(records),
        "records_with_transfer_dry_run_enabled": transfer_records,
        "transfer_steps": len(transfer_by_step),
        "zero_proposal_transfer_steps": sum(1 for step in transfer_by_step.values() if int(step["num_proposals"]) == 0),
        "sent_proposal_count": len(sent_proposal_ids),
        "received_proposal_count": len(received_proposal_ids),
        "pending_proposal_count": len(pending_proposal_ids),
        "ready_validated_proposal_count": len(validated_proposal_ids | ready_proposal_ids),
        "dropped_proposal_count": len(dropped_proposal_ids),
        "pending_reason_counts": dict(pending_reason_counts),
        "drop_reason_counts": dict(drop_reason_counts),
        "base_delta_min": min(pending_base_deltas) if pending_base_deltas else None,
        "base_delta_median": median_int(pending_base_deltas),
        "base_delta_max": max(pending_base_deltas) if pending_base_deltas else None,
        "base_overshot_count": base_overshot_count,
        "eager_tokens_generated": generated_tokens,
        "eager_tokens_promoted": promoted_tokens,
        "eager_tokens_discarded": discarded_tokens,
        "eager_tokens_transferred": transferred_tokens,
        "eager_tokens_transfer_pending": transfer_pending_tokens,
        "eager_tokens_transfer_validated": transfer_validated_tokens,
        "eager_tokens_transfer_dropped": transfer_dropped_tokens,
        "target_eager_non_empty_count": target_eager_non_empty_count,
        "eager_verified_counter_rows": verified_counter_rows,
        "base_mismatch_count": base_mismatch_count,
        "bad_sent_received_step_count": bad_sent_received_steps,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_transfer_dry_run_enabled",
        "transfer_steps",
        "zero_proposal_transfer_steps",
        "sent_proposal_count",
        "received_proposal_count",
        "pending_proposal_count",
        "ready_validated_proposal_count",
        "dropped_proposal_count",
        "pending_reason_counts",
        "drop_reason_counts",
        "base_delta_min",
        "base_delta_median",
        "base_delta_max",
        "base_overshot_count",
        "eager_tokens_generated",
        "eager_tokens_promoted",
        "eager_tokens_discarded",
        "eager_tokens_transferred",
        "eager_tokens_transfer_pending",
        "eager_tokens_transfer_validated",
        "eager_tokens_transfer_dropped",
        "target_eager_non_empty_count",
        "eager_verified_counter_rows",
        "base_mismatch_count",
        "bad_sent_received_step_count",
    ):
        print(f"{key}={summary[key]}")


def make_proposal(proposal_id: int = 101, seq_id: int = 1) -> EagerProposal:
    return EagerProposal(
        proposal_id=proposal_id,
        seq_id=seq_id,
        request_id=f"req-{seq_id}",
        parent_kind="normal",
        parent_proposal_id=None,
        source_step_id=7,
        source_plan_id=11,
        home_batch_id=0,
        base_len=12,
        base_pre_verify=False,
        base_num_completion_tokens=8,
        proposal_token_ids=[201, 202, 203, 204],
        to_be_verified_token_ids=[111, 112, 113, 201],
        proposal_len=4,
        state=EAGER_STATE_READY_TO_VERIFY,
        valid=True,
    )


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "enable_eager_transfer_dry_run": False,
        "eager_transfer_dry_run_enabled": False,
        "target_home_set": [1],
        "draft_home_set": [2],
        "target_eager_set": [],
        "draft_eager_set": [],
        "normal_gamma": 4,
        "eager_tokens_generated": 0,
        "eager_tokens_promoted": 0,
        "eager_tokens_discarded": 0,
        "eager_tokens_transferred": 0,
        "eager_tokens_transfer_pending": 0,
        "eager_tokens_transfer_validated": 0,
        "eager_tokens_transfer_dropped": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_plan_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "target_home_set": [1],
            "draft_eager_set": [1],
        }
    )
    return record


def synthetic_draft_record() -> dict[str, Any]:
    record = synthetic_plan_record()
    record.update(
        {
            "enable_eager_draft_dry_run": True,
            "eager_tokens_generated": 4,
        }
    )
    return record


def synthetic_promotion_record() -> dict[str, Any]:
    record = synthetic_draft_record()
    record.update(
        {
            "enable_eager_promotion_dry_run": True,
            "eager_tokens_promoted": 4,
            "eager_tokens_discarded": 0,
        }
    )
    return record


def synthetic_transfer_record(num_proposals: int = 1) -> dict[str, Any]:
    record = synthetic_promotion_record()
    sent_ids = [101] if num_proposals else []
    seq_ids = [1] if num_proposals else []
    transferred = 4 if num_proposals else 0
    record.update(
        {
            "enable_eager_transfer_dry_run": True,
            "eager_transfer_dry_run_enabled": True,
            "eager_transfer_step_id": 7,
            "eager_transfer_plan_id": 11,
            "eager_transfer_num_proposals": num_proposals,
            "eager_transfer_payload_len": 20 if num_proposals else 0,
            "eager_transfer_sent_proposal_ids": sent_ids,
            "eager_transfer_sent_seq_ids": seq_ids,
            "eager_transfer_received_proposal_ids": sent_ids,
            "eager_transfer_received_seq_ids": seq_ids,
            "eager_transfer_validated_proposal_ids": sent_ids,
            "eager_transfer_pending_proposal_ids": [],
            "eager_transfer_dropped_proposal_ids": [],
            "eager_transfer_drop_reason_by_proposal_id": {},
            "eager_transfer_base_len_by_seq_id": {"1": 12} if num_proposals else {},
            "eager_transfer_base_pre_verify_by_seq_id": {"1": False} if num_proposals else {},
            "eager_transfer_current_len_by_seq_id": {"1": 12} if num_proposals else {},
            "eager_transfer_base_match_by_seq_id": {"1": True} if num_proposals else {},
            "eager_transfer_proposal_len_by_proposal_id": {"101": 4} if num_proposals else {},
            "eager_transfer_to_verify_len_by_proposal_id": {"101": 4} if num_proposals else {},
            "eager_pending_received_proposal_ids": [],
            "eager_pending_received_seq_ids": [],
            "eager_pending_base_not_reached_proposal_ids": [],
            "eager_pending_base_not_reached_seq_ids": [],
            "eager_pending_ready_proposal_ids": [],
            "eager_pending_ready_seq_ids": [],
            "eager_pending_dropped_proposal_ids": [],
            "eager_pending_drop_reason_by_proposal_id": {},
            "eager_pending_buffer_size_before_update": 0,
            "eager_pending_buffer_size_after_update": 0,
            "eager_pending_buffer_size_after_receive": num_proposals,
            "eager_pending_buffer_size_after_clear": 0,
            "eager_pending_current_len_by_seq_id": {"1": 12} if num_proposals else {},
            "eager_pending_base_len_by_seq_id": {"1": 12} if num_proposals else {},
            "eager_pending_base_delta_by_seq_id": {"1": 0} if num_proposals else {},
            "eager_pending_state_by_proposal_id": {"101": "base_reached"} if num_proposals else {},
            "draft_eager_buffer_size_after_transfer": 0,
            "target_eager_buffer_size_after_clear": 0,
            "eager_tokens_transferred": transferred,
            "eager_tokens_transfer_pending": 0,
            "eager_tokens_transfer_validated": transferred,
            "eager_tokens_transfer_dropped": 0,
        }
    )
    if not num_proposals:
        record["eager_tokens_generated"] = 0
        record["eager_tokens_promoted"] = 0
    return record


def synthetic_pending_receive_record() -> dict[str, Any]:
    record = synthetic_transfer_record(1)
    record.update(
        {
            "eager_transfer_validated_proposal_ids": [],
            "eager_transfer_pending_proposal_ids": [101],
            "eager_transfer_current_len_by_seq_id": {"1": 11},
            "eager_transfer_base_match_by_seq_id": {"1": False},
            "eager_pending_received_proposal_ids": [101],
            "eager_pending_received_seq_ids": [1],
            "eager_pending_base_not_reached_proposal_ids": [101],
            "eager_pending_base_not_reached_seq_ids": [1],
            "eager_pending_buffer_size_after_receive": 1,
            "eager_pending_buffer_size_after_clear": 1,
            "target_eager_buffer_size_after_clear": 1,
            "eager_pending_current_len_by_seq_id": {"1": 11},
            "eager_pending_base_len_by_seq_id": {"1": 12},
            "eager_pending_base_delta_by_seq_id": {"1": 1},
            "eager_pending_state_by_proposal_id": {"101": "pending_base_not_reached"},
            "eager_tokens_transfer_pending": 4,
            "eager_tokens_transfer_validated": 0,
        }
    )
    return record


def synthetic_pending_ready_record() -> dict[str, Any]:
    record = synthetic_transfer_record(0)
    record.update(
        {
            "eager_transfer_step_id": 8,
            "eager_transfer_plan_id": 12,
            "eager_pending_ready_proposal_ids": [101],
            "eager_pending_ready_seq_ids": [1],
            "eager_transfer_base_len_by_seq_id": {"1": 12},
            "eager_transfer_base_pre_verify_by_seq_id": {"1": False},
            "eager_transfer_current_len_by_seq_id": {"1": 12},
            "eager_transfer_base_match_by_seq_id": {"1": True},
            "eager_pending_current_len_by_seq_id": {"1": 12},
            "eager_pending_base_len_by_seq_id": {"1": 12},
            "eager_pending_base_delta_by_seq_id": {"1": 0},
            "eager_pending_state_by_proposal_id": {"101": "base_reached"},
            "eager_pending_buffer_size_before_update": 1,
            "eager_pending_buffer_size_after_update": 0,
            "eager_pending_buffer_size_after_receive": 0,
            "eager_pending_buffer_size_after_clear": 0,
            "target_eager_buffer_size_before_receive": 1,
            "eager_tokens_transfer_validated": 4,
        }
    )
    return record


def synthetic_pending_drop_record(reason: str, proposal_id: int, step_id: int, current_len: int) -> dict[str, Any]:
    record = synthetic_transfer_record(0)
    record.update(
        {
            "eager_transfer_step_id": step_id,
            "eager_transfer_plan_id": 20 + step_id,
            "eager_pending_dropped_proposal_ids": [proposal_id],
            "eager_pending_drop_reason_by_proposal_id": {str(proposal_id): reason},
            "eager_transfer_base_len_by_seq_id": {"1": 12},
            "eager_transfer_base_pre_verify_by_seq_id": {"1": False},
            "eager_transfer_current_len_by_seq_id": {"1": current_len},
            "eager_transfer_base_match_by_seq_id": {"1": current_len == 12},
            "eager_pending_current_len_by_seq_id": {"1": current_len},
            "eager_pending_base_len_by_seq_id": {"1": 12},
            "eager_pending_base_delta_by_seq_id": {"1": 12 - current_len},
            "eager_pending_state_by_proposal_id": {str(proposal_id): reason},
            "eager_pending_buffer_size_before_update": 1,
            "eager_pending_buffer_size_after_update": 0,
            "eager_pending_buffer_size_after_receive": 0,
            "eager_pending_buffer_size_after_clear": 0,
            "target_eager_buffer_size_before_receive": 1,
            "eager_tokens_transfer_dropped": 4,
        }
    )
    return record


def synthetic_split_transfer_records() -> list[dict[str, Any]]:
    draft = synthetic_transfer_record(1)
    target = synthetic_transfer_record(1)
    draft.update(
        {
            "runner_role": "dual_draft",
            "eager_transfer_received_proposal_ids": [],
            "eager_transfer_received_seq_ids": [],
            "eager_transfer_validated_proposal_ids": [],
            "eager_transfer_pending_proposal_ids": [],
            "eager_tokens_transfer_validated": 0,
            "eager_pending_buffer_size_after_receive": 0,
        }
    )
    target.update(
        {
            "runner_role": "dual_verify",
            "eager_tokens_generated": 0,
            "eager_tokens_promoted": 0,
            "eager_tokens_transferred": 0,
            "eager_transfer_sent_proposal_ids": [],
            "eager_transfer_sent_seq_ids": [],
        }
    )
    return [draft, target]


def run_synthetic_tests() -> None:
    proposal = make_proposal()
    meta, payload = serialize_eager_transfer_payload([proposal], gamma=4, plan_id=11, step_id=7)
    roundtrip = deserialize_eager_transfer_payload(meta, payload)
    assert len(roundtrip) == 1
    assert roundtrip[0].proposal_id == proposal.proposal_id
    assert roundtrip[0].proposal_token_ids == proposal.proposal_token_ids
    assert roundtrip[0].to_be_verified_token_ids == proposal.to_be_verified_token_ids

    zero_meta, zero_payload = serialize_eager_transfer_payload([], gamma=4, plan_id=11, step_id=7)
    assert zero_meta[0] == 0 and zero_meta[1] == 0 and zero_payload == []
    assert deserialize_eager_transfer_payload(zero_meta, zero_payload) == []

    bad_meta = list(meta)
    bad_meta[1] += 1
    try:
        deserialize_eager_transfer_payload(bad_meta, payload)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed eager transfer payload length was accepted")

    valid_records = [
        synthetic_base_record(),
        synthetic_plan_record(),
        synthetic_draft_record(),
        synthetic_promotion_record(),
        synthetic_transfer_record(0),
        synthetic_transfer_record(1),
        synthetic_pending_receive_record(),
        synthetic_pending_ready_record(),
        synthetic_pending_drop_record("seq_returned_pre_verify_before_base", 201, 9, 11),
        synthetic_pending_drop_record("seq_finished_before_base", 202, 10, 11),
        synthetic_pending_drop_record("base_overshot_later", 203, 11, 13),
        *synthetic_split_transfer_records(),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager transfer records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_received_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("received proposal ids" in error for error in errors), "checker missed sent/received mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_base_match_by_seq_id"] = {"1": False}
    errors, _ = validate_records(invalid)
    assert any("base_len match" in error for error in errors), "checker missed base_len mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_base_pre_verify_by_seq_id"] = {"1": True}
    errors, _ = validate_records(invalid)
    assert any("base_pre_verify" in error for error in errors), "checker missed base_pre_verify mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_validated_proposal_ids"] = []
    invalid[-1]["eager_transfer_dropped_proposal_ids"] = [101]
    invalid[-1]["eager_transfer_drop_reason_by_proposal_id"] = {"101": "seq_not_running"}
    invalid[-1]["eager_tokens_transfer_validated"] = 0
    invalid[-1]["eager_tokens_transfer_dropped"] = 4
    errors, _ = validate_records(invalid)
    assert any("seq_not_running" in error for error in errors), "checker missed seq_not_running for present seq"

    print("Synthetic eager transfer dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-4 eager transfer dry-run traces.")
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
    print("\nEager transfer dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
