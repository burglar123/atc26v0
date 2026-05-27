#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


TAKEOVER_SOURCE = "phase1h5e3_takeover_lane"
FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError("Unsupported trace format. Expected a raw list or trace-record dict.")


def is_dual_record(record: dict[str, Any]) -> bool:
    return record.get("execution_mode") == "dual_batch_pearl" and record.get("dual_batch_enabled") is True


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def as_int_set(value: Any) -> set[int]:
    return set(as_int_list(value))


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


def first_mapping(record: dict[str, Any], *keys: str) -> dict:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def commit_active(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("eager_commit_enabled", False))
        or record.get("eager_commit_source") == TAKEOVER_SOURCE
        or bool(as_int_set(record.get("eager_commit_candidate_proposal_ids")))
        or bool(as_int_set(record.get("eager_committed_proposal_ids")))
        or bool(as_int_set(record.get("eager_commit_skipped_proposal_ids")))
        or int_value(record.get("eager_tokens_committed"), 0) > 0
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_commit_enabled = 0
    commit_active_records = 0
    committed_ids_seen: set[int] = set()
    skipped_ids_seen: set[int] = set()
    candidate_ids_seen: set[int] = set()
    readiness_ready_ids_seen: set[int] = set()
    committed_token_by_id_seen: dict[int, int] = {}
    committed_sides_by_id: dict[int, set[str]] = defaultdict(set)
    repeated_steps_by_side: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    skip_reason_by_id_seen: dict[int, str] = {}
    committed_but_not_readiness_ready_ids: set[int] = set()
    committed_non_full_accept_ids: set[int] = set()
    missing_unexpected_count = 0
    real_target_eager_nonempty_count = 0
    max_actual_verified_per_record = 0
    max_actual_accepted_per_record = 0
    max_committed_tokens_per_record = 0
    counted_commit_events: set[tuple[str, int, int, int]] = set()
    counted_counter_records: set[tuple[str, int, int, tuple[int, ...]]] = set()
    target_actual_verified_sum = 0
    target_actual_accepted_sum = 0
    target_actual_rejected_sum = 0
    target_actual_invalidated_sum = 0
    draft_actual_verified_sum = 0
    draft_actual_accepted_sum = 0
    draft_actual_rejected_sum = 0
    draft_actual_invalidated_sum = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        target_eager = as_int_set(record.get("target_eager_set"))
        if target_eager:
            real_target_eager_nonempty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty: {sorted(target_eager)}")

        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(f"record[{idx}] unexpected missing buffered normal proposals: {sorted(missing_unexpected)}")

        enabled = bool(record.get("enable_eager_commit_ready_only", False))
        active = commit_active(record)
        if not enabled:
            if active:
                errors.append(f"record[{idx}] real eager commit fields populated while flag disabled")
            continue

        records_with_commit_enabled += 1
        if not bool(record.get("enable_eager_commit_readiness_dry_run", False)):
            errors.append(f"record[{idx}] eager commit-ready-only must imply commit-readiness dry-run")
        if not active:
            continue

        commit_active_records += 1
        if record.get("eager_commit_source") != TAKEOVER_SOURCE:
            errors.append(f"record[{idx}] eager commit source must be {TAKEOVER_SOURCE!r}")

        candidate_ids = as_int_set(record.get("eager_commit_candidate_proposal_ids"))
        committed_ids = as_int_set(record.get("eager_committed_proposal_ids"))
        skipped_ids = as_int_set(record.get("eager_commit_skipped_proposal_ids"))
        readiness_ids = as_int_set(record.get("eager_commit_ready_proposal_ids")) or as_int_set(
            record.get("eager_commit_from_readiness_proposal_ids")
        )
        not_ready_ids = as_int_set(record.get("eager_commit_not_ready_proposal_ids"))
        candidate_ids_seen.update(candidate_ids)
        committed_ids_seen.update(committed_ids)
        skipped_ids_seen.update(skipped_ids)
        readiness_ready_ids_seen.update(readiness_ids)
        if committed_ids & skipped_ids:
            errors.append(f"record[{idx}] proposals both committed and skipped: {sorted(committed_ids & skipped_ids)}")
        if candidate_ids and not committed_ids <= candidate_ids:
            errors.append(f"record[{idx}] committed proposal outside commit candidates: {sorted(committed_ids - candidate_ids)}")
        if readiness_ids and not committed_ids <= readiness_ids:
            outside_readiness = committed_ids - readiness_ids
            committed_but_not_readiness_ready_ids.update(outside_readiness)
            errors.append(
                f"record[{idx}] committed proposal outside readiness-ready ids: {sorted(outside_readiness)}"
            )
        if not_ready_ids & committed_ids:
            errors.append(f"record[{idx}] not-ready proposals committed: {sorted(not_ready_ids & committed_ids)}")
        if record.get("eager_commit_duplicate_proposal_ids"):
            errors.append(f"record[{idx}] duplicate committed proposal ids: {record['eager_commit_duplicate_proposal_ids']}")
        if record.get("eager_commit_duplicate_seq_ids"):
            errors.append(f"record[{idx}] duplicate committed seq ids: {record['eager_commit_duplicate_seq_ids']}")

        token_by_id = first_mapping(record, "eager_committed_token_count_by_proposal_id")
        accept_by_id = first_mapping(record, "eager_committed_accept_len_by_proposal_id")
        action_by_id = first_mapping(record, "eager_committed_action_by_proposal_id")
        result_by_id = first_mapping(record, "eager_committed_verify_result_by_proposal_id")
        precondition_ok_by_id = first_mapping(record, "eager_commit_precondition_ok_by_proposal_id")
        precondition_failed_by_id = first_mapping(record, "eager_commit_precondition_failed_by_proposal_id")
        reason_by_id = first_mapping(record, "eager_commit_skip_reason_by_proposal_id")
        target_before_by_seq = first_mapping(record, "eager_commit_target_seq_len_before_by_seq_id")
        target_after_by_seq = first_mapping(record, "eager_commit_target_seq_len_after_by_seq_id")
        draft_before_by_seq = first_mapping(record, "eager_commit_draft_seq_len_before_by_seq_id")
        draft_after_by_seq = first_mapping(record, "eager_commit_draft_seq_len_after_by_seq_id")
        len_match_by_seq = first_mapping(record, "eager_commit_target_draft_len_match_by_seq_id")
        token_match_by_seq = first_mapping(record, "eager_commit_target_draft_token_match_by_seq_id")
        seq_ids = as_int_list(record.get("eager_committed_seq_ids"))
        pid_to_seq = dict(zip(as_int_list(record.get("eager_committed_proposal_ids")), seq_ids))
        gamma = int_value(record.get("normal_gamma"), 0)
        side = str(record.get("eager_commit_side", "unknown"))
        step = int_value(record.get("eager_commit_step_id"), int_value(record.get("step_id"), -1))
        plan = int_value(record.get("eager_commit_plan_id"), int_value(record.get("plan_id"), -1))

        for proposal_id in sorted(committed_ids):
            token_count = int_value(dict_get(token_by_id, proposal_id), -1)
            accept_len = int_value(dict_get(accept_by_id, proposal_id), -1)
            action = str(dict_get(action_by_id, proposal_id, ""))
            verify_result = str(dict_get(result_by_id, proposal_id, ""))
            if dict_get(precondition_ok_by_id, proposal_id) is not True:
                errors.append(f"record[{idx}] committed proposal_id={proposal_id} lacks precondition_ok")
            if dict_get(precondition_failed_by_id, proposal_id) is True:
                errors.append(f"record[{idx}] committed proposal_id={proposal_id} has failed precondition")
            if action != FULL_ACCEPT_ACTION:
                errors.append(f"record[{idx}] committed proposal_id={proposal_id} bad action={action!r}")
            if verify_result != "full_accept":
                committed_non_full_accept_ids.add(proposal_id)
                errors.append(f"record[{idx}] committed proposal_id={proposal_id} is not full_accept")
            if gamma > 0 and token_count != gamma:
                errors.append(f"record[{idx}] committed proposal_id={proposal_id} token_count != gamma")
            if accept_len != token_count:
                errors.append(f"record[{idx}] committed proposal_id={proposal_id} accept_len/token_count mismatch")
            seq_id = pid_to_seq.get(proposal_id)
            if seq_id is not None:
                target_before = int_value(dict_get(target_before_by_seq, seq_id), -1)
                target_after = int_value(dict_get(target_after_by_seq, seq_id), -1)
                draft_before = int_value(dict_get(draft_before_by_seq, seq_id), -1)
                draft_after = int_value(dict_get(draft_after_by_seq, seq_id), -1)
                if target_before >= 0 and target_after - target_before != token_count:
                    errors.append(f"record[{idx}] target len delta mismatch for seq_id={seq_id}")
                if draft_before >= 0 and draft_after - draft_before != token_count:
                    errors.append(f"record[{idx}] draft len delta mismatch for seq_id={seq_id}")
                if dict_get(len_match_by_seq, seq_id) is not True:
                    errors.append(f"record[{idx}] target/draft length mismatch for seq_id={seq_id}")
                if dict_get(token_match_by_seq, seq_id) is False:
                    errors.append(f"record[{idx}] target/draft token mismatch for seq_id={seq_id}")
            previous_token_count = committed_token_by_id_seen.get(proposal_id)
            if previous_token_count is not None and previous_token_count != token_count:
                errors.append(
                    f"record[{idx}] committed token count changed for proposal_id={proposal_id}: "
                    f"old={previous_token_count}, new={token_count}"
                )
            committed_token_by_id_seen[proposal_id] = token_count
            committed_sides_by_id[proposal_id].add(side)
            repeated_steps_by_side[(side, proposal_id)].add((step, plan))

        for proposal_id in sorted(skipped_ids):
            reason = str(dict_get(reason_by_id, proposal_id, ""))
            if not reason:
                errors.append(f"record[{idx}] skipped proposal_id={proposal_id} missing reason")
            previous_reason = skip_reason_by_id_seen.get(proposal_id)
            if previous_reason is not None and previous_reason != reason:
                errors.append(
                    f"record[{idx}] skipped reason changed for proposal_id={proposal_id}: "
                    f"old={previous_reason!r}, new={reason!r}"
                )
            skip_reason_by_id_seen[proposal_id] = reason

        committed_token_sum = sum(int_value(dict_get(token_by_id, proposal_id), 0) for proposal_id in committed_ids)
        if int_value(record.get("eager_tokens_committed"), 0) != committed_token_sum:
            errors.append(f"record[{idx}] eager_tokens_committed mismatch")
        if int_value(record.get("eager_tokens_committed_full_accept"), 0) != committed_token_sum:
            errors.append(f"record[{idx}] eager_tokens_committed_full_accept mismatch")
        if int_value(record.get("eager_commit_committed_count"), 0) != len(committed_ids):
            errors.append(f"record[{idx}] committed count field mismatch")
        if int_value(record.get("eager_commit_candidate_count"), 0) != len(candidate_ids):
            errors.append(f"record[{idx}] candidate count field mismatch")
        if int_value(record.get("eager_tokens_verified"), 0) != committed_token_sum:
            errors.append(f"record[{idx}] actual eager verified counter mismatch")
        if int_value(record.get("eager_tokens_accepted"), 0) != committed_token_sum:
            errors.append(f"record[{idx}] actual eager accepted counter mismatch")
        if int_value(record.get("eager_tokens_rejected"), 0) != 0:
            errors.append(f"record[{idx}] actual eager rejected counter must remain zero")
        if int_value(record.get("eager_tokens_invalidated"), 0) != 0:
            errors.append(f"record[{idx}] actual eager invalidated counter must remain zero")
        max_actual_verified_per_record = max(
            max_actual_verified_per_record,
            int_value(record.get("eager_tokens_verified"), 0),
        )
        max_actual_accepted_per_record = max(
            max_actual_accepted_per_record,
            int_value(record.get("eager_tokens_accepted"), 0),
        )
        max_committed_tokens_per_record = max(
            max_committed_tokens_per_record,
            int_value(record.get("eager_tokens_committed"), 0),
        )
        counter_key = (side, step, plan, tuple(sorted(committed_ids)))
        if committed_ids and counter_key not in counted_counter_records:
            counted_counter_records.add(counter_key)
            if side == "target":
                target_actual_verified_sum += int_value(record.get("eager_tokens_verified"), 0)
                target_actual_accepted_sum += int_value(record.get("eager_tokens_accepted"), 0)
                target_actual_rejected_sum += int_value(record.get("eager_tokens_rejected"), 0)
                target_actual_invalidated_sum += int_value(record.get("eager_tokens_invalidated"), 0)
            elif side == "draft":
                draft_actual_verified_sum += int_value(record.get("eager_tokens_verified"), 0)
                draft_actual_accepted_sum += int_value(record.get("eager_tokens_accepted"), 0)
                draft_actual_rejected_sum += int_value(record.get("eager_tokens_rejected"), 0)
                draft_actual_invalidated_sum += int_value(record.get("eager_tokens_invalidated"), 0)
        for proposal_id in sorted(committed_ids):
            event_key = (side, proposal_id, step, plan)
            if event_key in counted_commit_events:
                continue
            counted_commit_events.add(event_key)

    repeated_commit_proposal_ids = sorted({
        proposal_id
        for (_side, proposal_id), step_keys in repeated_steps_by_side.items()
        if len(step_keys) > 1
    })
    if repeated_commit_proposal_ids:
        errors.append(f"proposal committed more than once on one side: {repeated_commit_proposal_ids}")
    if readiness_ready_ids_seen and not committed_ids_seen:
        errors.append(
            "eager commit-ready-only enabled and commit-ready proposals exist, "
            f"but no proposals were committed: ready_ids={sorted(readiness_ready_ids_seen)}"
        )
    if readiness_ready_ids_seen and not committed_ids_seen <= readiness_ready_ids_seen:
        outside_readiness = committed_ids_seen - readiness_ready_ids_seen
        committed_but_not_readiness_ready_ids.update(outside_readiness)
        errors.append(
            "committed proposals were not found in any readiness-ready source: "
            f"{sorted(outside_readiness)}"
        )
    committed_token_count = sum(int(value) for value in committed_token_by_id_seen.values())
    missing_target_side = sorted(
        proposal_id for proposal_id, sides in committed_sides_by_id.items() if "target" not in sides
    )
    missing_draft_side = sorted(
        proposal_id for proposal_id, sides in committed_sides_by_id.items() if "draft" not in sides
    )
    if missing_target_side:
        errors.append(f"committed proposals missing target-side commit event: {missing_target_side}")
    if missing_draft_side:
        errors.append(f"committed proposals missing draft-side commit event: {missing_draft_side}")
    if committed_token_count and target_actual_verified_sum != committed_token_count:
        errors.append(
            "target actual eager verified aggregate does not match committed_token_count: "
            f"target={target_actual_verified_sum}, committed={committed_token_count}"
        )
    if committed_token_count and target_actual_accepted_sum != committed_token_count:
        errors.append(
            "target actual eager accepted aggregate does not match committed_token_count: "
            f"target={target_actual_accepted_sum}, committed={committed_token_count}"
        )
    if committed_token_count and draft_actual_verified_sum != committed_token_count:
        errors.append(
            "draft actual eager verified aggregate does not match committed_token_count: "
            f"draft={draft_actual_verified_sum}, committed={committed_token_count}"
        )
    if committed_token_count and draft_actual_accepted_sum != committed_token_count:
        errors.append(
            "draft actual eager accepted aggregate does not match committed_token_count: "
            f"draft={draft_actual_accepted_sum}, committed={committed_token_count}"
        )
    skip_reason_counts = Counter(skip_reason_by_id_seen.values())

    summary = {
        "total_trace_records": len(records),
        "records_with_eager_commit_enabled": records_with_commit_enabled,
        "commit_active_records": commit_active_records,
        "commit_candidate_proposal_count": len(candidate_ids_seen),
        "commit_readiness_ready_proposal_count": len(readiness_ready_ids_seen),
        "committed_proposal_count": len(committed_ids_seen),
        "skipped_proposal_count": len(skipped_ids_seen),
        "committed_token_count": committed_token_count,
        "target_actual_eager_verified_token_increment_sum": target_actual_verified_sum,
        "target_actual_eager_accepted_token_increment_sum": target_actual_accepted_sum,
        "target_actual_eager_rejected_token_increment_sum": target_actual_rejected_sum,
        "target_actual_eager_invalidated_token_increment_sum": target_actual_invalidated_sum,
        "draft_actual_eager_verified_token_increment_sum": draft_actual_verified_sum,
        "draft_actual_eager_accepted_token_increment_sum": draft_actual_accepted_sum,
        "draft_actual_eager_rejected_token_increment_sum": draft_actual_rejected_sum,
        "draft_actual_eager_invalidated_token_increment_sum": draft_actual_invalidated_sum,
        "max_actual_eager_verified_tokens_per_record": max_actual_verified_per_record,
        "max_actual_eager_accepted_tokens_per_record": max_actual_accepted_per_record,
        "max_eager_tokens_committed_per_record": max_committed_tokens_per_record,
        "skip_reason_counts": dict(skip_reason_counts),
        "repeated_commit_proposal_ids": repeated_commit_proposal_ids,
        "committed_but_not_readiness_ready_proposal_ids": sorted(committed_but_not_readiness_ready_ids),
        "committed_non_full_accept_proposal_ids": sorted(committed_non_full_accept_ids),
        "missing_target_side_commit_proposal_ids": missing_target_side,
        "missing_draft_side_commit_proposal_ids": missing_draft_side,
        "real_target_eager_non_empty_count": real_target_eager_nonempty_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_eager_commit_enabled",
        "commit_active_records",
        "commit_candidate_proposal_count",
        "commit_readiness_ready_proposal_count",
        "committed_proposal_count",
        "skipped_proposal_count",
        "committed_token_count",
        "target_actual_eager_verified_token_increment_sum",
        "target_actual_eager_accepted_token_increment_sum",
        "target_actual_eager_rejected_token_increment_sum",
        "target_actual_eager_invalidated_token_increment_sum",
        "draft_actual_eager_verified_token_increment_sum",
        "draft_actual_eager_accepted_token_increment_sum",
        "draft_actual_eager_rejected_token_increment_sum",
        "draft_actual_eager_invalidated_token_increment_sum",
        "max_actual_eager_verified_tokens_per_record",
        "max_actual_eager_accepted_tokens_per_record",
        "max_eager_tokens_committed_per_record",
        "skip_reason_counts",
        "repeated_commit_proposal_ids",
        "committed_but_not_readiness_ready_proposal_ids",
        "committed_non_full_accept_proposal_ids",
        "missing_target_side_commit_proposal_ids",
        "missing_draft_side_commit_proposal_ids",
        "real_target_eager_non_empty_count",
        "missing_buffered_proposal_unexpected_count",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    return {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_commit_readiness_dry_run": False,
        "enable_eager_commit_ready_only": False,
        "eager_commit_enabled": False,
        "eager_commit_source": None,
        "eager_tokens_verified": 0,
        "eager_tokens_accepted": 0,
        "eager_tokens_rejected": 0,
        "eager_tokens_invalidated": 0,
    }


def synthetic_commit_record(
    proposal_id: int,
    seq_id: int,
    side: str,
    step: int,
    plan: int,
) -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_commit_readiness_dry_run": True,
            "enable_eager_commit_ready_only": True,
            "eager_commit_enabled": True,
            "eager_commit_source": TAKEOVER_SOURCE,
            "eager_commit_side": side,
            "eager_commit_step_id": step,
            "eager_commit_plan_id": plan,
            "eager_commit_ready_proposal_ids": [proposal_id],
            "eager_commit_ready_seq_ids": [seq_id],
            "eager_commit_not_ready_proposal_ids": [],
            "eager_commit_not_ready_seq_ids": [],
            "eager_commit_not_ready_reason_by_proposal_id": {},
            "eager_commit_candidate_proposal_ids": [proposal_id],
            "eager_commit_candidate_seq_ids": [seq_id],
            "eager_commit_from_readiness_proposal_ids": [proposal_id],
            "eager_committed_proposal_ids": [proposal_id],
            "eager_committed_seq_ids": [seq_id],
            "eager_committed_token_count_by_proposal_id": {str(proposal_id): 4},
            "eager_committed_accept_len_by_proposal_id": {str(proposal_id): 4},
            "eager_committed_action_by_proposal_id": {str(proposal_id): FULL_ACCEPT_ACTION},
            "eager_committed_verify_result_by_proposal_id": {str(proposal_id): "full_accept"},
            "eager_commit_skipped_proposal_ids": [],
            "eager_commit_skip_reason_by_proposal_id": {},
            "eager_commit_precondition_ok_by_proposal_id": {str(proposal_id): True},
            "eager_commit_precondition_failed_by_proposal_id": {str(proposal_id): False},
            "eager_commit_precondition_failure_reason_by_proposal_id": {},
            "eager_commit_target_seq_len_before_by_seq_id": {str(seq_id): 20},
            "eager_commit_target_seq_len_after_by_seq_id": {str(seq_id): 24},
            "eager_commit_draft_seq_len_before_by_seq_id": {str(seq_id): 20},
            "eager_commit_draft_seq_len_after_by_seq_id": {str(seq_id): 24},
            "eager_commit_target_draft_len_match_by_seq_id": {str(seq_id): True},
            "eager_commit_target_draft_token_match_by_seq_id": {str(seq_id): True},
            "eager_tokens_committed": 4,
            "eager_tokens_committed_full_accept": 4,
            "eager_commit_candidate_count": 1,
            "eager_commit_committed_count": 1,
            "eager_commit_skipped_count": 0,
            "eager_tokens_verified": 4,
            "eager_tokens_accepted": 4,
            "eager_tokens_rejected": 0,
            "eager_tokens_invalidated": 0,
        }
    )
    return record


def synthetic_skip_record() -> dict[str, Any]:
    record = synthetic_base_record()
    proposal_ids = [201, 202, 203]
    seq_ids = [301, 302, 303]
    record.update(
        {
            "enable_eager_commit_readiness_dry_run": True,
            "enable_eager_commit_ready_only": True,
            "eager_commit_enabled": True,
            "eager_commit_source": TAKEOVER_SOURCE,
            "eager_commit_side": "draft",
            "eager_commit_step_id": 30,
            "eager_commit_plan_id": 40,
            "eager_commit_ready_proposal_ids": [],
            "eager_commit_ready_seq_ids": [],
            "eager_commit_not_ready_proposal_ids": proposal_ids,
            "eager_commit_not_ready_seq_ids": seq_ids,
            "eager_commit_not_ready_reason_by_proposal_id": {
                str(proposal_id): "not_full_accept" for proposal_id in proposal_ids
            },
            "eager_commit_candidate_proposal_ids": proposal_ids,
            "eager_commit_candidate_seq_ids": seq_ids,
            "eager_commit_from_readiness_proposal_ids": [],
            "eager_committed_proposal_ids": [],
            "eager_committed_seq_ids": [],
            "eager_commit_skipped_proposal_ids": proposal_ids,
            "eager_commit_skip_reason_by_proposal_id": {
                str(proposal_id): "not_full_accept" for proposal_id in proposal_ids
            },
            "eager_commit_precondition_ok_by_proposal_id": {
                str(proposal_id): False for proposal_id in proposal_ids
            },
            "eager_commit_precondition_failed_by_proposal_id": {
                str(proposal_id): True for proposal_id in proposal_ids
            },
            "eager_commit_precondition_failure_reason_by_proposal_id": {
                str(proposal_id): "not_full_accept" for proposal_id in proposal_ids
            },
            "eager_tokens_committed": 0,
            "eager_tokens_committed_full_accept": 0,
            "eager_commit_candidate_count": 3,
            "eager_commit_committed_count": 0,
            "eager_commit_skipped_count": 3,
        }
    )
    return record


def synthetic_mirrored_records() -> list[dict[str, Any]]:
    records = [synthetic_base_record()]
    for offset, proposal_id in enumerate([101, 102, 103, 104, 105]):
        seq_id = 7 + offset
        step = 10 + offset
        plan = 20 + offset
        records.append(synthetic_commit_record(proposal_id, seq_id, "target", step, plan))
        records.append(synthetic_commit_record(proposal_id, seq_id, "draft", step, plan))
    records.append(synthetic_skip_record())
    return records


def run_synthetic_tests() -> None:
    valid_records = synthetic_mirrored_records()
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid mirrored synthetic commit-ready-only records failed: {errors}"
    _, summary = validate_records(valid_records)
    assert summary["committed_proposal_count"] == 5, "synthetic committed proposal count mismatch"
    assert summary["committed_token_count"] == 20, "synthetic committed token aggregate mismatch"
    assert summary["target_actual_eager_verified_token_increment_sum"] == 20, "synthetic target aggregate mismatch"
    assert summary["draft_actual_eager_verified_token_increment_sum"] == 20, "synthetic draft aggregate mismatch"

    invalid = deepcopy(valid_records)
    invalid[1]["enable_eager_commit_ready_only"] = False
    errors, _ = validate_records(invalid)
    assert any("flag disabled" in error for error in errors), "checker missed disabled commit fields"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_not_ready_proposal_ids"] = [201]
    invalid[1]["eager_commit_not_ready_reason_by_proposal_id"] = {"201": "not_full_accept"}
    invalid[1]["eager_commit_candidate_proposal_ids"] = [101, 201]
    invalid[1]["eager_commit_candidate_count"] = 2
    invalid[1]["eager_committed_proposal_ids"] = [101, 201]
    errors, _ = validate_records(invalid)
    assert any("not-ready proposals committed" in error for error in errors), "checker missed not-ready commit"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_committed_verify_result_by_proposal_id"] = {"101": "partial_accept"}
    errors, _ = validate_records(invalid)
    assert any("not full_accept" in error for error in errors), "checker missed partial commit"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_tokens_verified"] = 0
    errors, _ = validate_records(invalid)
    assert any("target actual eager verified aggregate" in error for error in errors), (
        "checker missed target aggregate mismatch"
    )

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_target_seq_len_after_by_seq_id"] = {"7": 23}
    errors, _ = validate_records(invalid)
    assert any("target len delta mismatch" in error for error in errors), "checker missed target len mismatch"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_duplicate_proposal_ids"] = [101]
    errors, _ = validate_records(invalid)
    assert any("duplicate committed proposal" in error for error in errors), "checker missed duplicate proposal"

    duplicate_passive = deepcopy(valid_records)
    duplicate_passive.append(deepcopy(valid_records[1]))
    errors, summary = validate_records(duplicate_passive)
    assert not errors, f"checker double-counted duplicate passive row: {errors}"
    assert summary["target_actual_eager_verified_token_increment_sum"] == 20, (
        "duplicate passive row changed target aggregate"
    )

    true_duplicate = deepcopy(valid_records)
    extra = deepcopy(valid_records[1])
    extra["eager_commit_step_id"] = 99
    extra["eager_commit_plan_id"] = 199
    true_duplicate.append(extra)
    errors, _ = validate_records(true_duplicate)
    assert any("more than once" in error for error in errors), "checker missed true duplicate commit"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 0
    errors, _ = validate_records(invalid)
    assert any("draft actual eager verified aggregate" in error for error in errors), (
        "checker missed draft aggregate mismatch"
    )

    print("Synthetic eager commit-ready-only checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-6a guarded eager commit-ready-only traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Eager commit-ready-only trace checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
