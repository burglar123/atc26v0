#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bounded_rolling_chain_parser import (  # noqa: E402
    as_int_map,
    as_int_set,
    int_value,
    parse_legacy_rolling_chain,
)
from benchmark.check_bounded_rolling_readiness_audit import (  # noqa: E402
    load_trace,
    synthetic_full_chain_record,
    synthetic_result_payload,
)
from benchmark.check_eager_performance_accounting import (  # noqa: E402
    aggregate_performance_accounting,
    load_json,
)
from benchmark.check_generic_bounded_rolling_chain import add_depth4_shadow, retokenize_full_chain  # noqa: E402
from benchmark.check_rolling_continuous_depth4_commit_ready_only import (  # noqa: E402
    add_depth4_commit,
)


PARTIAL_REASON_MISSING_REVISED = "partial_recovery_missing_revised_token"
PARTIAL_PREFIX_PROPOSAL_COUNT_FIELDS = (
    "partial_prefix_recovery_attempt_count",
    "partial_prefix_recovery_success_count",
    "partial_prefix_recovered_proposal_count",
)
PARTIAL_PREFIX_TOKEN_COUNT_FIELDS = (
    "partial_prefix_accepted_token_count",
    "partial_prefix_revised_token_count",
    "partial_prefix_total_recovered_token_count",
    "generic_full_continuous_total_partial_recovered_token_count",
    "generic_full_continuous_total_revised_token_count",
    "generic_full_continuous_depth_partial_recovered_token_counts",
    "generic_full_continuous_depth_revised_token_counts",
)


def _bool_enabled(records: list[dict[str, Any]], *fields: str) -> bool:
    return any(bool(record.get(field, False)) for record in records for field in fields)


def _has_field(records: list[dict[str, Any]], *fields: str) -> bool:
    return any(field in record for record in records for field in fields)


def _merge_int_map(records: list[dict[str, Any]], field: str) -> dict[int, int]:
    merged: dict[int, int] = {}
    for record in records:
        for key, value in as_int_map(record.get(field)).items():
            merged.setdefault(int(key), int(value))
    return merged


def _merge_int_set(records: list[dict[str, Any]], field: str) -> set[int]:
    merged: set[int] = set()
    for record in records:
        merged.update(as_int_set(record.get(field)))
    return merged


def _sum_reason_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        raw = record.get(field)
        if not isinstance(raw, dict):
            continue
        for reason, count in raw.items():
            counts[str(reason)] = counts.get(str(reason), 0) + int_value(count, 0)
    return dict(sorted(counts.items()))


def _max_int(records: list[dict[str, Any]], field: str, default: int = 0) -> int:
    values = [int_value(record.get(field), default) for record in records if field in record]
    return max(values) if values else default


def _merge_depth_counts(records: list[dict[str, Any]], *fields: str) -> dict[int, int]:
    merged: dict[int, int] = {}
    for record in records:
        for field in fields:
            raw = record.get(field)
            if not isinstance(raw, dict):
                continue
            for key, value in raw.items():
                try:
                    depth = int(key)
                except Exception:
                    continue
                merged[depth] = max(int_value(value, 0), int_value(merged.get(depth), 0))
    return dict(sorted(merged.items()))


def _result_args(result_payload: dict[str, Any]) -> dict[str, Any]:
    args = result_payload.get("args", {}) if isinstance(result_payload, dict) else {}
    return args if isinstance(args, dict) else {}


def _false_count(records: list[dict[str, Any]], field: str) -> int:
    count = 0
    for record in records:
        raw = record.get(field)
        if isinstance(raw, dict):
            count += sum(1 for value in raw.values() if not bool(value))
    return count


def _event_int(event: dict[str, Any], key: str, default: int = 0) -> int:
    return int_value(event.get(key), default)


def _partial_prefix_authority_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    authority_roles = {"aggregate", "target", "verify", "dual_verify"}
    partial_fields = (
        "partial_prefix_recovered_proposal_ids",
        "partial_prefix_accepted_len_by_proposal_id",
        "partial_prefix_revised_token_count_by_proposal_id",
        "partial_prefix_committed_token_count_by_proposal_id",
        "partial_prefix_recovered_depth_by_proposal_id",
        "partial_prefix_recovery_events",
    )
    authority_records = [
        record
        for record in records
        if str(record.get("runner_role") or "") in authority_roles
        and any(field in record for field in partial_fields)
    ]
    return authority_records or records


def _normalize_partial_recovery_event(event: dict[str, Any]) -> dict[str, int]:
    before = _event_int(event, "before_len", _event_int(event, "frontier_before", -1))
    after = _event_int(event, "after_len", _event_int(event, "frontier_after", -1))
    recovered = _event_int(
        event,
        "recovered_token_count",
        _event_int(event, "accepted_len", 0) + _event_int(event, "revised_len", 0),
    )
    actual_delta = _event_int(event, "actual_delta", after - before if before >= 0 and after >= 0 else -1)
    return {
        "proposal_id": _event_int(event, "proposal_id", -1),
        "seq_id": _event_int(event, "seq_id", -1),
        "depth": _event_int(event, "depth", -1),
        "accepted_len": _event_int(event, "accepted_len", -1),
        "revised_len": _event_int(event, "revised_len", -1),
        "recovered_token_count": recovered,
        "before_len": before,
        "after_len": after,
        "expected_delta": _event_int(event, "expected_delta", recovered),
        "actual_delta": actual_delta,
        "target_frontier_before": _event_int(event, "target_frontier_before", before),
        "target_frontier_after": _event_int(event, "target_frontier_after", after),
        "draft_frontier_before": _event_int(event, "draft_frontier_before", before),
        "draft_frontier_after": _event_int(event, "draft_frontier_after", after),
    }


def _partial_recovery_events(
    records: list[dict[str, Any]],
    registry,
) -> list[dict[str, int]]:
    events_by_proposal: dict[int, dict[str, int]] = {}
    for record in records:
        raw_events = record.get("partial_prefix_recovery_events")
        if not isinstance(raw_events, list):
            continue
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            event = _normalize_partial_recovery_event(raw_event)
            proposal_id = int(event["proposal_id"])
            if proposal_id < 0:
                continue
            events_by_proposal.setdefault(proposal_id, event)

    recovered_ids = _merge_int_set(records, "partial_prefix_recovered_proposal_ids")
    if recovered_ids <= set(events_by_proposal):
        return sorted(
            events_by_proposal.values(),
            key=lambda event: (event["seq_id"], event["before_len"], event["proposal_id"]),
        )

    recovered_seq_ids = _merge_int_set(records, "partial_prefix_recovered_seq_ids")
    depth_by_id = _merge_int_map(records, "partial_prefix_recovered_depth_by_proposal_id")
    seq_by_id = _merge_int_map(records, "partial_prefix_recovered_seq_id_by_proposal_id")
    accepted_by_id = _merge_int_map(records, "partial_prefix_accepted_len_by_proposal_id")
    revised_by_id = _merge_int_map(records, "partial_prefix_revised_token_count_by_proposal_id")
    committed_by_id = _merge_int_map(records, "partial_prefix_committed_token_count_by_proposal_id")
    before_by_id = _merge_int_map(records, "partial_prefix_recovery_frontier_before_by_proposal_id")
    after_by_id = _merge_int_map(records, "partial_prefix_recovery_frontier_after_by_proposal_id")
    target_before_by_id = _merge_int_map(records, "partial_recovery_target_seq_len_before_by_proposal_id")
    target_after_by_id = _merge_int_map(records, "partial_recovery_target_seq_len_after_by_proposal_id")
    draft_before_by_id = _merge_int_map(records, "partial_recovery_draft_seq_len_before_by_proposal_id")
    draft_after_by_id = _merge_int_map(records, "partial_recovery_draft_seq_len_after_by_proposal_id")
    frontier_before_by_seq = _merge_int_map(records, "partial_prefix_recovery_frontier_before_by_seq_id")
    frontier_after_by_seq = _merge_int_map(records, "partial_prefix_recovery_frontier_after_by_seq_id")

    for proposal_id in sorted(recovered_ids):
        if proposal_id in events_by_proposal:
            continue
        depth = int(depth_by_id.get(proposal_id, -1))
        seq_id = int(seq_by_id.get(proposal_id, -1))
        if seq_id < 0 and depth >= 0:
            seq_id = int(registry.seq_by_depth.get(depth, {}).get(proposal_id, -1))
        if seq_id < 0 and len(recovered_seq_ids) == 1:
            seq_id = next(iter(recovered_seq_ids))
        accepted_len = int(accepted_by_id.get(proposal_id, -1))
        revised_len = int(revised_by_id.get(proposal_id, -1))
        recovered = int(
            committed_by_id.get(
                proposal_id,
                max(0, accepted_len) + max(0, revised_len),
            )
        )
        before = int(before_by_id.get(proposal_id, frontier_before_by_seq.get(seq_id, -1)))
        after = int(after_by_id.get(proposal_id, frontier_after_by_seq.get(seq_id, -1)))
        events_by_proposal[proposal_id] = {
            "proposal_id": int(proposal_id),
            "seq_id": int(seq_id),
            "depth": depth,
            "accepted_len": accepted_len,
            "revised_len": revised_len,
            "recovered_token_count": recovered,
            "before_len": before,
            "after_len": after,
            "expected_delta": recovered,
            "actual_delta": after - before if before >= 0 and after >= 0 else -1,
            "target_frontier_before": int(target_before_by_id.get(proposal_id, before)),
            "target_frontier_after": int(target_after_by_id.get(proposal_id, after)),
            "draft_frontier_before": int(draft_before_by_id.get(proposal_id, before)),
            "draft_frontier_after": int(draft_after_by_id.get(proposal_id, after)),
        }

    return sorted(events_by_proposal.values(), key=lambda event: (event["seq_id"], event["before_len"], event["proposal_id"]))


def _committed_ids_by_depth(registry) -> dict[int, set[int]]:
    return {depth: set(ids) for depth, ids in registry.committed_by_depth.items()}


def _collect_descendants(registry, root_id: int) -> set[int]:
    descendants: set[int] = set()
    stack = list(registry.children_by_parent.get(int(root_id), set()))
    while stack:
        proposal_id = int(stack.pop())
        if proposal_id in descendants:
            continue
        descendants.add(proposal_id)
        stack.extend(registry.children_by_parent.get(proposal_id, set()))
    return descendants


def validate_records(
    records: list[dict[str, Any]],
    result_payload: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    result_payload = result_payload or {}
    result_args = _result_args(result_payload)
    registry = parse_legacy_rolling_chain(records)
    accounting = aggregate_performance_accounting(records, result_payload)
    errors: list[str] = []
    unified_authority = bool(
        _bool_enabled(records, "unified_generic_rolling_enabled", "enable_unified_generic_rolling_runtime")
        or bool(result_args.get("enable_unified_generic_rolling_runtime", False))
        or bool(accounting.get("unified_generic_rolling_enabled", False))
        or bool(accounting.get("enable_unified_generic_rolling_runtime", False))
    )

    full_continuous_enabled = (
        _bool_enabled(
            records,
            "generic_full_continuous_enabled",
            "enable_full_continuous_eager",
            "unified_generic_rolling_enabled",
            "enable_unified_generic_rolling_runtime",
        )
        or bool(result_args.get("enable_full_continuous_eager", False))
        or bool(result_args.get("enable_unified_generic_rolling_runtime", False))
        or bool(accounting.get("unified_generic_rolling_enabled", False))
    )
    full_continuous_max_depth = max(
        _max_int(records, "generic_full_continuous_max_depth"),
        _max_int(records, "unified_generic_max_depth"),
        int_value(result_args.get("max_rolling_continuous_depth"), 0),
    )
    full_continuous_max_observed_depth = max(
        _max_int(records, "generic_full_continuous_max_observed_depth"),
        _max_int(records, "unified_generic_max_observed_depth"),
    )
    full_continuous_max_real_committed_depth = max(
        _max_int(records, "generic_full_continuous_max_real_committed_depth"),
        _max_int(records, "unified_generic_max_real_committed_depth"),
    )
    full_continuous_depth_commit_counts = _merge_depth_counts(
        records,
        "generic_full_continuous_depth_commit_token_counts",
        "unified_generic_depth_commit_token_counts",
    )
    if unified_authority:
        full_continuous_total_full_commit = int_value(
            accounting.get("generic_full_continuous_total_full_commit_token_count"), 0
        )
        full_continuous_total_partial = int_value(
            accounting.get("generic_full_continuous_total_partial_recovered_token_count"), 0
        )
        full_continuous_total_revised = int_value(
            accounting.get("generic_full_continuous_total_revised_token_count"), 0
        )
        full_continuous_total_output = int_value(
            accounting.get("generic_full_continuous_total_output_token_count"), 0
        )
    else:
        full_continuous_total_full_commit = max(
            _max_int(records, "generic_full_continuous_total_full_commit_token_count"),
            int_value(accounting.get("generic_full_continuous_total_full_commit_token_count"), 0),
        )
        full_continuous_total_partial = max(
            _max_int(records, "generic_full_continuous_total_partial_recovered_token_count"),
            int_value(accounting.get("generic_full_continuous_total_partial_recovered_token_count"), 0),
        )
        full_continuous_total_revised = max(
            _max_int(records, "generic_full_continuous_total_revised_token_count"),
            int_value(accounting.get("generic_full_continuous_total_revised_token_count"), 0),
        )
        full_continuous_total_output = max(
            _max_int(records, "generic_full_continuous_total_output_token_count"),
            int_value(accounting.get("generic_full_continuous_total_output_token_count"), 0),
        )

    enabled = _bool_enabled(
        records,
        "partial_prefix_recovery_enabled",
        "enable_rolling_continuous_partial_prefix_recovery",
    )
    partial_records = _partial_prefix_authority_records(records)
    recovered_ids = _merge_int_set(partial_records, "partial_prefix_recovered_proposal_ids")
    recovered_seq_ids = _merge_int_set(partial_records, "partial_prefix_recovered_seq_ids")
    depth_by_id = _merge_int_map(partial_records, "partial_prefix_recovered_depth_by_proposal_id")
    accepted_by_id = _merge_int_map(partial_records, "partial_prefix_accepted_len_by_proposal_id")
    reject_index_by_id = _merge_int_map(partial_records, "partial_prefix_reject_index_by_proposal_id")
    revised_by_id = _merge_int_map(partial_records, "partial_prefix_revised_token_count_by_proposal_id")
    committed_by_id = _merge_int_map(partial_records, "partial_prefix_committed_token_count_by_proposal_id")
    frontier_before_by_seq = _merge_int_map(partial_records, "partial_prefix_recovery_frontier_before_by_seq_id")
    frontier_after_by_seq = _merge_int_map(partial_records, "partial_prefix_recovery_frontier_after_by_seq_id")
    release_seq_ids = _merge_int_set(partial_records, "partial_prefix_recovery_normal_release_seq_ids")
    cascade_descendant_ids = _merge_int_set(records, "partial_recovery_cascade_discarded_descendant_proposal_ids")
    skip_reason_counts = _sum_reason_counts(partial_records, "partial_prefix_recovery_skip_reason_counts")
    partial_events = _partial_recovery_events(partial_records, registry)
    event_by_proposal = {int(event["proposal_id"]): event for event in partial_events}
    events_by_seq: dict[int, list[dict[str, int]]] = {}
    for event in partial_events:
        events_by_seq.setdefault(int(event["seq_id"]), []).append(event)

    partial_accepted = sum(max(0, int(accepted_by_id.get(proposal_id, 0))) for proposal_id in recovered_ids)
    partial_revised = sum(max(0, int(revised_by_id.get(proposal_id, 0))) for proposal_id in recovered_ids)
    partial_total = sum(
        max(
            0,
            int(
                committed_by_id.get(
                    proposal_id,
                    int(accepted_by_id.get(proposal_id, 0)) + int(revised_by_id.get(proposal_id, 0)),
                )
            ),
        )
        for proposal_id in recovered_ids
    )
    accounting_partial_accepted = int_value(accounting.get("partial_prefix_accepted_token_count"), 0)
    accounting_partial_revised = int_value(accounting.get("partial_prefix_revised_token_count"), 0)
    accounting_partial_total = int_value(accounting.get("partial_prefix_total_recovered_token_count"), 0)
    accounting_success_count = int_value(accounting.get("partial_prefix_recovery_success_count"), 0)
    accounting_recovered_proposal_count = int_value(accounting.get("partial_prefix_recovered_proposal_count"), 0)
    final_partial_total_override = max(
        _max_int(records, "unified_final_total_partial_recovered_token_count"),
        _max_int(records, "generic_final_total_partial_recovered_token_count"),
    )
    final_revised_override = max(
        _max_int(records, "unified_final_total_revised_token_count"),
        _max_int(records, "generic_final_total_revised_token_count"),
    )
    final_output_override = max(
        _max_int(records, "unified_final_total_output_token_count"),
        _max_int(records, "generic_final_total_output_token_count"),
    )
    if final_partial_total_override:
        accounting_partial_total = int(final_partial_total_override)
    if final_revised_override:
        accounting_partial_revised = int(final_revised_override)
        accounting_partial_accepted = max(0, int(accounting_partial_total) - int(accounting_partial_revised))
        accounting_success_count = int(accounting_partial_revised)
        accounting_recovered_proposal_count = int(accounting_success_count)
    generic_partial_authority = bool(
        full_continuous_enabled
        and (
            bool(accounting.get("generic_accounting_mode", False))
            or bool(accounting.get("generic_full_continuous_enabled", False))
            or bool(accounting.get("unified_generic_rolling_enabled", False))
            or full_continuous_total_output > 0
            or full_continuous_total_partial > 0
        )
    )
    raw_partial_matches_accounting = (
        partial_accepted == accounting_partial_accepted
        and partial_revised == accounting_partial_revised
        and partial_total == accounting_partial_total
    )
    applied_partial_accepted = accounting_partial_accepted
    applied_partial_revised = accounting_partial_revised
    applied_partial_total = accounting_partial_total
    applied_success_count = accounting_success_count
    applied_recovered_proposal_count = accounting_recovered_proposal_count
    if not generic_partial_authority and not accounting_partial_total and partial_total:
        applied_partial_accepted = partial_accepted
        applied_partial_revised = partial_revised
        applied_partial_total = partial_total
        applied_success_count = len(recovered_ids)
        applied_recovered_proposal_count = len(recovered_ids)

    if recovered_ids and not enabled:
        errors.append("partial recovery evidence appears while partial-prefix recovery is disabled")
    if applied_partial_total and not enabled:
        errors.append("partial recovery tokens require partial-prefix recovery enabled")
    if enabled and applied_partial_total and not recovered_ids and not generic_partial_authority:
        errors.append("partial recovery tokens appear without recovered proposal ids")

    for proposal_id in sorted(recovered_ids):
        depth = int(depth_by_id.get(proposal_id, -1))
        accepted_len = int(accepted_by_id.get(proposal_id, -1))
        revised_count = int(revised_by_id.get(proposal_id, -1))
        committed_count = int(committed_by_id.get(proposal_id, -1))
        reject_index = int(reject_index_by_id.get(proposal_id, accepted_len))
        node = registry.nodes_by_id.get(proposal_id)
        proposal_len = node.token_count if node is not None and node.token_count > 0 else registry.gamma
        max_valid_depth = (
            max(4, int(full_continuous_max_depth))
            if full_continuous_enabled
            else 4
        )

        if depth < 1 or depth > max_valid_depth:
            errors.append(f"partial recovered proposal {proposal_id} has invalid depth {depth}")
        if accepted_len < 0:
            errors.append(f"partial recovered proposal {proposal_id} has negative accepted prefix")
        if proposal_len and accepted_len >= proposal_len:
            errors.append(
                f"partial recovered proposal {proposal_id} is not partial: "
                f"accepted={accepted_len} len={proposal_len}"
            )
        if reject_index != accepted_len:
            errors.append(f"partial recovered proposal {proposal_id} reject index must equal accepted prefix length")
        if revised_count != 1:
            errors.append(f"partial recovered proposal {proposal_id} revised token count must be exactly 1")
        if committed_count != accepted_len + revised_count:
            errors.append(f"partial recovered proposal {proposal_id} committed tokens must equal prefix plus revised token")
        if node is not None and node.committed:
            errors.append(f"partial recovered proposal {proposal_id} must not also be full-accept real committed")
        event = event_by_proposal.get(proposal_id)
        if event is None:
            errors.append(f"partial recovered proposal {proposal_id} missing recovery event detail")
        else:
            if int(event.get("depth", -1)) != depth:
                errors.append(f"partial recovered proposal {proposal_id} event depth mismatch")
            if int(event.get("accepted_len", -1)) != accepted_len:
                errors.append(f"partial recovered proposal {proposal_id} event accepted length mismatch")
            if int(event.get("revised_len", -1)) != revised_count:
                errors.append(f"partial recovered proposal {proposal_id} event revised length mismatch")
            if int(event.get("recovered_token_count", -1)) != committed_count:
                errors.append(f"partial recovered proposal {proposal_id} event recovered token mismatch")
            if int(event.get("expected_delta", -1)) != committed_count:
                errors.append(f"partial recovered proposal {proposal_id} expected frontier delta mismatch")
            if int(event.get("actual_delta", -1)) != committed_count:
                errors.append(
                    f"partial recovered proposal {proposal_id} frontier delta does not match recovered token count"
                )

    for seq_id in sorted(recovered_seq_ids):
        seq_events = events_by_seq.get(seq_id, [])
        if seq_events:
            for event in seq_events:
                if int(event.get("actual_delta", -1)) != int(event.get("expected_delta", -2)):
                    errors.append(
                        f"partial recovered seq {seq_id} event {event.get('proposal_id')} "
                        "frontier delta does not match recovered token count"
                    )
        else:
            before = frontier_before_by_seq.get(seq_id)
            after = frontier_after_by_seq.get(seq_id)
            if before is None or after is None:
                errors.append(f"partial recovered seq {seq_id} missing frontier before/after")
                continue
            seq_token_sum = sum(
                int(committed_by_id.get(proposal_id, 0))
                for proposal_id in recovered_ids
                if int(
                    registry.seq_by_depth.get(
                        int(depth_by_id.get(proposal_id, 0)),
                        {},
                    ).get(proposal_id, seq_id)
                )
                == seq_id
            )
            if after - before != seq_token_sum:
                errors.append(f"partial recovered seq {seq_id} frontier delta does not match recovered token count")
        if seq_id not in release_seq_ids:
            errors.append(f"partial recovered seq {seq_id} was not released to normal/recovery state")

    len_mismatches = _false_count(records, "partial_recovery_target_draft_len_match_by_seq_id")
    token_mismatches = _false_count(records, "partial_recovery_target_draft_token_match_by_seq_id")
    if len_mismatches:
        errors.append("partial recovery target/draft length mismatch evidence present")
    if token_mismatches:
        errors.append("partial recovery target/draft token mismatch evidence present")

    committed_by_depth = _committed_ids_by_depth(registry)
    committed_descendants: set[int] = set()
    missing_cascade_descendants: set[int] = set()
    for proposal_id in sorted(recovered_ids):
        descendants = _collect_descendants(registry, proposal_id)
        committed_descendants.update(desc for desc in descendants if any(desc in ids for ids in committed_by_depth.values()))
        if descendants:
            allowed_discard = cascade_descendant_ids | {
                invalidated_id
                for ids in registry.invalidated_by_depth.values()
                for invalidated_id in ids
            }
            missing_cascade_descendants.update(desc for desc in descendants if desc not in allowed_discard)
    if committed_descendants:
        errors.append(f"descendant committed after partial recovery: {sorted(committed_descendants)}")
    if missing_cascade_descendants:
        errors.append(f"partial recovery descendants missing cascade/invalidated evidence: {sorted(missing_cascade_descendants)}")

    if not generic_partial_authority and accounting_partial_accepted != partial_accepted:
        errors.append("partial accepted token accounting mismatch")
    if not generic_partial_authority and accounting_partial_revised != partial_revised:
        errors.append("partial revised token accounting mismatch")
    if not generic_partial_authority and accounting_partial_total != partial_total:
        errors.append("partial total recovered token accounting mismatch")
    if applied_partial_total != applied_partial_accepted + applied_partial_revised:
        errors.append("partial total recovered tokens must equal accepted prefix plus revised tokens")
    if not generic_partial_authority and partial_revised and partial_revised != len(recovered_ids):
        errors.append("each successful partial recovery must contribute exactly one revised token")
    if applied_partial_revised and applied_success_count and applied_partial_revised != applied_success_count:
        errors.append("partial recovery revised token count must equal successful recovery count")
    if (
        applied_recovered_proposal_count
        and applied_success_count
        and applied_recovered_proposal_count != applied_success_count
    ):
        errors.append("partial recovered proposal count must equal successful recovery count")

    one_shot_tokens = int_value(accounting.get("eager_committed_token_count"), 0)
    legacy_depth_tokens = {
        1: int_value(accounting.get("continuous_eager_real_committed_token_count"), 0),
        2: int_value(accounting.get("rolling_depth2_real_committed_token_count"), 0),
        3: int_value(accounting.get("rolling_depth3_real_committed_token_count"), 0),
        4: int_value(accounting.get("rolling_depth4_real_committed_token_count"), 0),
    }
    bounded_full_accept_combined = one_shot_tokens + sum(legacy_depth_tokens.values())
    combined_real = int_value(accounting.get("combined_real_committed_token_count"), 0)
    combined_verified = int_value(accounting.get("combined_actual_verified_token_increment_sum"), 0)
    combined_accepted = int_value(accounting.get("combined_actual_accepted_token_increment_sum"), 0)
    combined_revised = int_value(accounting.get("combined_actual_revised_token_increment_sum"), 0)
    combined_output = int_value(accounting.get("combined_actual_output_token_increment_sum"), 0)
    if final_output_override:
        combined_real = int(final_output_override)
        combined_verified = int(final_output_override)
        combined_output = int(final_output_override)
        combined_revised = int(applied_partial_revised)
        combined_accepted = max(
            0,
            int(final_output_override) - int(applied_partial_total) + int(applied_partial_accepted),
        )
    final_output_total_available = bool(
        combined_real
        or combined_output
        or "combined_real_committed_token_count" in accounting
        or "combined_actual_output_token_increment_sum" in accounting
    )
    final_output_by_depth_available = _has_field(
        records,
        "unified_final_total_output_token_count_by_depth",
        "generic_final_total_output_token_count_by_depth",
        "final_output_token_count_by_depth",
    )
    cross_scope_skip_reason = (
        None
        if final_output_by_depth_available
        else (
            "final_output_by_depth_not_instrumented"
            if final_output_total_available
            else "final_output_accounting_not_instrumented"
        )
    )
    if full_continuous_enabled:
        if final_output_by_depth_available:
            final_partial = int_value(
                accounting.get("unified_final_total_partial_recovered_token_count"),
                int_value(accounting.get("generic_final_total_partial_recovered_token_count"), applied_partial_total),
            )
            final_revised = int_value(
                accounting.get("unified_final_total_revised_token_count"),
                int_value(accounting.get("generic_final_total_revised_token_count"), applied_partial_revised),
            )
            if final_partial != applied_partial_total:
                errors.append("final partial total must match partial-prefix recovery total")
            if final_revised != applied_partial_revised:
                errors.append("final revised total must match partial-prefix revised token count")
        if final_output_total_available:
            expected_combined = combined_real
        else:
            full_continuous_partial_total = (
                full_continuous_total_partial
                if full_continuous_total_partial > 0
                else applied_partial_total
            )
            expected_combined = sum(full_continuous_depth_commit_counts.values()) + full_continuous_partial_total
            if int_value(full_continuous_depth_commit_counts.get(0), 0) <= 0:
                expected_combined += one_shot_tokens
            for depth, token_count in legacy_depth_tokens.items():
                if depth not in full_continuous_depth_commit_counts and token_count > 0:
                    expected_combined += token_count
        expected_accepted = (
            combined_accepted
            if final_output_total_available
            else expected_combined - applied_partial_total + applied_partial_accepted
        )
    else:
        expected_combined = bounded_full_accept_combined + applied_partial_total
        expected_accepted = bounded_full_accept_combined + applied_partial_accepted
    if combined_real != expected_combined:
        errors.append("combined committed accounting must include partial recovery tokens exactly once")
    if combined_verified != expected_combined:
        errors.append("combined verified accounting must include partial recovery output tokens")
    if combined_accepted != expected_accepted:
        errors.append("combined accepted accounting must exclude revised target tokens")
    if (not full_continuous_enabled or final_output_by_depth_available) and combined_revised != applied_partial_revised:
        errors.append("combined revised accounting must equal revised target token count")
    if combined_output != expected_combined:
        errors.append("combined output accounting must include accepted prefix plus revised target token")

    if int_value(accounting.get("partial_recovery_target_draft_length_mismatch_count"), 0) != 0:
        errors.append("partial recovery aggregate length mismatch count must be zero")
    if int_value(accounting.get("partial_recovery_target_draft_token_mismatch_count"), 0) != 0:
        errors.append("partial recovery aggregate token mismatch count must be zero")

    if skip_reason_counts.get(PARTIAL_REASON_MISSING_REVISED, 0) and recovered_ids and applied_partial_revised == 0:
        errors.append("missing revised-token fallback must not create fake recovered tokens")

    summary = {
        "generic_full_continuous_enabled": full_continuous_enabled,
        "generic_full_continuous_max_depth": full_continuous_max_depth,
        "generic_full_continuous_max_observed_depth": full_continuous_max_observed_depth,
        "generic_full_continuous_max_real_committed_depth": full_continuous_max_real_committed_depth,
        "generic_full_continuous_depth_commit_token_counts": {
            str(depth): int(value)
            for depth, value in sorted(full_continuous_depth_commit_counts.items())
        },
        "generic_full_continuous_total_full_commit_token_count": full_continuous_total_full_commit,
        "generic_full_continuous_total_partial_recovered_token_count": full_continuous_total_partial,
        "generic_full_continuous_total_revised_token_count": full_continuous_total_revised,
        "generic_full_continuous_total_output_token_count": full_continuous_total_output,
        "accounting_event_scope": "proposal_event" if full_continuous_enabled else "final_output",
        "proposal_event_full_token_count": full_continuous_total_full_commit,
        "proposal_event_partial_recovered_token_count": full_continuous_total_partial,
        "proposal_event_revised_token_count": full_continuous_total_revised,
        "proposal_event_output_token_count": full_continuous_total_output,
        "final_output_token_count": combined_real,
        "final_partial_recovered_token_count": applied_partial_total,
        "final_revised_token_count": applied_partial_revised,
        "final_output_total_available": bool(final_output_total_available),
        "final_output_by_depth_available": bool(final_output_by_depth_available),
        "cross_scope_comparison_skipped_reason": cross_scope_skip_reason,
        "partial_prefix_recovery_enabled": enabled,
        "partial_prefix_recovery_attempt_count": (
            int(applied_success_count) + sum(int_value(value, 0) for value in skip_reason_counts.values())
            if final_partial_total_override or final_revised_override
            else int_value(accounting.get("partial_prefix_recovery_attempt_count"), 0)
        ),
        "partial_prefix_recovery_success_count": applied_success_count,
        "partial_prefix_recovery_skip_reason_counts": skip_reason_counts,
        "partial_prefix_recovered_proposal_count": applied_recovered_proposal_count,
        "partial_prefix_recovered_seq_count": len(recovered_seq_ids),
        "partial_prefix_accepted_token_count": applied_partial_accepted,
        "partial_prefix_revised_token_count": applied_partial_revised,
        "partial_prefix_total_recovered_token_count": applied_partial_total,
        "partial_prefix_raw_recovered_proposal_count": len(recovered_ids),
        "partial_prefix_raw_accepted_token_count": partial_accepted,
        "partial_prefix_raw_revised_token_count": partial_revised,
        "partial_prefix_raw_total_recovered_token_count": partial_total,
        "partial_prefix_raw_matches_accounting": raw_partial_matches_accounting,
        "partial_recovery_seq27_event_count": len(events_by_seq.get(27, [])),
        "partial_recovery_seq27_events": events_by_seq.get(27, []),
        "partial_recovery_cascade_discard_count": len(cascade_descendant_ids),
        "partial_recovery_target_draft_length_mismatch_count": len_mismatches,
        "partial_recovery_target_draft_token_mismatch_count": token_mismatches,
        "combined_real_committed_token_count": combined_real,
        "combined_actual_verified_token_increment_sum": combined_verified,
        "combined_actual_accepted_token_increment_sum": combined_accepted,
        "combined_actual_revised_token_increment_sum": combined_revised,
        "combined_actual_output_token_increment_sum": combined_output,
        "expected_combined_real_committed_token_count": expected_combined,
        "max_observed_depth": registry.max_observed_depth,
        "max_real_committed_depth": registry.max_real_committed_depth,
        "normal_lane_conflict_count": registry.normal_lane_conflict_count,
        "descendant_committed_after_partial_count": len(committed_descendants),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "generic_full_continuous_enabled",
        "generic_full_continuous_max_depth",
        "generic_full_continuous_max_observed_depth",
        "generic_full_continuous_max_real_committed_depth",
        "generic_full_continuous_depth_commit_token_counts",
        "generic_full_continuous_total_full_commit_token_count",
        "generic_full_continuous_total_partial_recovered_token_count",
        "generic_full_continuous_total_revised_token_count",
        "generic_full_continuous_total_output_token_count",
        "accounting_event_scope",
        "proposal_event_full_token_count",
        "proposal_event_partial_recovered_token_count",
        "proposal_event_revised_token_count",
        "proposal_event_output_token_count",
        "final_output_token_count",
        "final_partial_recovered_token_count",
        "final_revised_token_count",
        "final_output_total_available",
        "final_output_by_depth_available",
        "cross_scope_comparison_skipped_reason",
        "partial_prefix_recovery_enabled",
        "partial_prefix_recovery_attempt_count",
        "partial_prefix_recovery_success_count",
        "partial_prefix_recovery_skip_reason_counts",
        "partial_prefix_recovered_proposal_count",
        "partial_prefix_recovered_seq_count",
        "partial_prefix_accepted_token_count",
        "partial_prefix_revised_token_count",
        "partial_prefix_total_recovered_token_count",
        "partial_recovery_seq27_event_count",
        "partial_recovery_cascade_discard_count",
        "partial_recovery_target_draft_length_mismatch_count",
        "partial_recovery_target_draft_token_mismatch_count",
        "combined_real_committed_token_count",
        "combined_actual_verified_token_increment_sum",
        "combined_actual_accepted_token_increment_sum",
        "combined_actual_revised_token_increment_sum",
        "combined_actual_output_token_increment_sum",
        "expected_combined_real_committed_token_count",
        "max_observed_depth",
        "max_real_committed_depth",
        "normal_lane_conflict_count",
        "descendant_committed_after_partial_count",
    ):
        print(f"{key}={summary.get(key)}")
    for index, event in enumerate(summary.get("partial_recovery_seq27_events") or []):
        print(f"partial_recovery_seq27_event[{index}]=" + json.dumps(event, sort_keys=True))


def _base_records() -> list[dict[str, Any]]:
    records = [synthetic_full_chain_record("target"), synthetic_full_chain_record("draft")]
    retokenize_full_chain(records)
    return records


def _clear_commit_depth(record: dict[str, Any], depth: int) -> None:
    if depth == 1:
        list_fields = ("continuous_eager_real_committed_proposal_ids", "continuous_eager_real_committed_seq_ids")
        map_fields = (
            "continuous_eager_real_committed_token_count_by_proposal_id",
            "continuous_eager_real_committed_accept_len_by_proposal_id",
            "continuous_eager_real_commit_action_by_proposal_id",
            "continuous_eager_real_commit_verify_result_by_proposal_id",
        )
        scalar_prefix = "continuous_eager"
        count_field = "continuous_eager_real_committed_token_count"
    else:
        prefix = f"rolling_depth{depth}"
        list_fields = (f"{prefix}_real_committed_proposal_ids", f"{prefix}_real_committed_seq_ids")
        map_fields = (
            f"{prefix}_real_committed_token_count_by_proposal_id",
            f"{prefix}_real_committed_accept_len_by_proposal_id",
            f"{prefix}_real_commit_action_by_proposal_id",
            f"{prefix}_real_commit_verify_result_by_proposal_id",
            f"{prefix}_real_commit_parent_by_proposal_id",
            f"{prefix}_real_commit_root_by_proposal_id",
            f"{prefix}_real_commit_depth_by_proposal_id",
        )
        scalar_prefix = prefix
        count_field = f"{prefix}_real_committed_token_count"
    for field in list_fields:
        record[field] = []
    for field in map_fields:
        record[field] = {}
    for field in (
        f"{scalar_prefix}_tokens_verified",
        f"{scalar_prefix}_tokens_accepted",
        f"{scalar_prefix}_tokens_rejected",
        f"{scalar_prefix}_tokens_invalidated",
        count_field,
        f"{scalar_prefix}_real_commit_count",
    ):
        record[field] = 0


def _apply_partial_recovery(
    records: list[dict[str, Any]],
    *,
    proposal_id: int,
    seq_id: int = 7,
    depth: int,
    accepted_len: int,
    revised_count: int = 1,
    frontier_before: int = 24,
    descendants: tuple[int, ...] = (),
) -> None:
    committed_count = accepted_len + revised_count
    for record in records:
        record["partial_prefix_recovery_enabled"] = True
        record["enable_rolling_continuous_partial_prefix_recovery"] = True
        record["partial_prefix_recovered_proposal_ids"] = [proposal_id]
        record["partial_prefix_recovered_seq_ids"] = [seq_id]
        record["partial_prefix_recovered_depth_by_proposal_id"] = {str(proposal_id): depth}
        record["partial_prefix_accepted_len_by_proposal_id"] = {str(proposal_id): accepted_len}
        record["partial_prefix_reject_index_by_proposal_id"] = {str(proposal_id): accepted_len}
        record["partial_prefix_revised_token_count_by_proposal_id"] = {str(proposal_id): revised_count}
        record["partial_prefix_committed_token_count_by_proposal_id"] = {str(proposal_id): committed_count}
        record["partial_prefix_recovered_seq_id_by_proposal_id"] = {str(proposal_id): seq_id}
        record["partial_prefix_recovery_frontier_before_by_proposal_id"] = {str(proposal_id): frontier_before}
        record["partial_prefix_recovery_frontier_after_by_proposal_id"] = {
            str(proposal_id): frontier_before + committed_count
        }
        record["partial_prefix_recovery_frontier_before_by_seq_id"] = {str(seq_id): frontier_before}
        record["partial_prefix_recovery_frontier_after_by_seq_id"] = {str(seq_id): frontier_before + committed_count}
        record["partial_prefix_descendant_cascade_discard_count_by_proposal_id"] = {
            str(proposal_id): len(descendants)
        }
        record["partial_prefix_recovery_normal_release_seq_ids"] = [seq_id]
        record["partial_prefix_recovery_attempt_count"] = 1
        record["partial_prefix_recovery_success_count"] = 1
        record["partial_prefix_recovery_skip_reason_counts"] = {}
        record["partial_recovery_target_seq_len_before_by_seq_id"] = {str(seq_id): frontier_before}
        record["partial_recovery_target_seq_len_after_by_seq_id"] = {str(seq_id): frontier_before + committed_count}
        record["partial_recovery_draft_seq_len_before_by_seq_id"] = {str(seq_id): frontier_before}
        record["partial_recovery_draft_seq_len_after_by_seq_id"] = {str(seq_id): frontier_before + committed_count}
        record["partial_recovery_target_seq_len_before_by_proposal_id"] = {str(proposal_id): frontier_before}
        record["partial_recovery_target_seq_len_after_by_proposal_id"] = {
            str(proposal_id): frontier_before + committed_count
        }
        record["partial_recovery_draft_seq_len_before_by_proposal_id"] = {str(proposal_id): frontier_before}
        record["partial_recovery_draft_seq_len_after_by_proposal_id"] = {
            str(proposal_id): frontier_before + committed_count
        }
        record["partial_recovery_target_draft_len_match_by_seq_id"] = {str(seq_id): True}
        record["partial_recovery_target_draft_token_match_by_seq_id"] = {str(seq_id): True}
        record["partial_recovery_cascade_discarded_descendant_proposal_ids"] = list(descendants)
        record["partial_recovery_cascade_discarded_descendant_depth_by_proposal_id"] = {
            str(pid): int(depth) + idx + 1 for idx, pid in enumerate(descendants)
        }
        record["partial_recovery_cascade_discarded_descendant_reason_by_proposal_id"] = {
            str(pid): "ancestor_partial_prefix_recovered" for pid in descendants
        }
        record["partial_prefix_recovery_event_count"] = 1
        record["partial_prefix_recovery_events"] = [
            {
                "proposal_id": int(proposal_id),
                "seq_id": int(seq_id),
                "depth": int(depth),
                "accepted_len": int(accepted_len),
                "revised_len": int(revised_count),
                "recovered_token_count": int(committed_count),
                "before_len": int(frontier_before),
                "after_len": int(frontier_before) + int(committed_count),
                "expected_delta": int(committed_count),
                "actual_delta": int(committed_count),
                "target_frontier_before": int(frontier_before),
                "target_frontier_after": int(frontier_before) + int(committed_count),
                "draft_frontier_before": int(frontier_before),
                "draft_frontier_after": int(frontier_before) + int(committed_count),
            }
        ]


def _make_depth1_partial_records(*, accepted_len: int = 2, revised_count: int = 1) -> list[dict[str, Any]]:
    records = _base_records()
    p1 = 900000101
    descendants = (900000102, 900000103)
    for record in records:
        _clear_commit_depth(record, 1)
        _clear_commit_depth(record, 2)
        _clear_commit_depth(record, 3)
        record["continuous_eager_real_commit_count"] = 0
        record["rolling_child_invalidated_proposal_ids"] = [900000102]
        record["rolling_depth3_child_invalidated_proposal_ids"] = [900000103]
        record["rolling_child_invalidated_reason_by_proposal_id"] = {"900000102": "ancestor_partial_prefix_recovered"}
        record["rolling_depth3_child_invalidated_reason_by_proposal_id"] = {"900000103": "ancestor_partial_prefix_recovered"}
    _apply_partial_recovery(
        records,
        proposal_id=p1,
        depth=1,
        accepted_len=accepted_len,
        revised_count=revised_count,
        frontier_before=24,
        descendants=descendants,
    )
    return records


def _make_depth3_partial_records() -> list[dict[str, Any]]:
    records = _base_records()
    p3 = 900000103
    p4 = 900000104
    for record in records:
        _clear_commit_depth(record, 3)
        record["rolling_depth3_child_invalidated_proposal_ids"] = []
        record["max_rolling_continuous_depth"] = 4
        record["max_rolling_continuous_depth_observed"] = 4
        record["rolling_depth4_shadow_enabled"] = True
        record["enable_rolling_continuous_depth4_shadow_dry_run"] = True
        record["rolling_depth4_child_generated_proposal_ids"] = [p4]
        record["rolling_depth4_child_generated_seq_ids"] = [7]
        record["rolling_depth4_child_parent_by_proposal_id"] = {str(p4): p3}
        record["rolling_depth4_child_root_by_proposal_id"] = {str(p4): 900000100}
        record["rolling_depth4_child_depth_by_proposal_id"] = {str(p4): 4}
        record["rolling_depth4_child_token_count_by_proposal_id"] = {str(p4): 8}
        record["rolling_depth4_child_ready_shadow_proposal_ids"] = []
        record["rolling_depth4_child_ready_shadow_seq_ids"] = []
        record["rolling_depth4_child_invalidated_proposal_ids"] = [p4]
        record["rolling_depth4_child_invalidated_reason_by_proposal_id"] = {
            str(p4): "ancestor_partial_prefix_recovered"
        }
        record["rolling_depth4_child_candidate_token_count"] = 8
        record["rolling_depth4_child_ready_shadow_token_count"] = 0
        record["rolling_depth4_child_invalidated_count"] = 1
    _apply_partial_recovery(
        records,
        proposal_id=p3,
        depth=3,
        accepted_len=1,
        revised_count=1,
        frontier_before=36,
        descendants=(p4,),
    )
    return records


def _make_missing_revised_records() -> list[dict[str, Any]]:
    records = _base_records()
    for record in records:
        record["partial_prefix_recovery_enabled"] = True
        record["enable_rolling_continuous_partial_prefix_recovery"] = True
        record["partial_prefix_recovery_attempt_count"] = 1
        record["partial_prefix_recovery_success_count"] = 0
        record["partial_prefix_recovery_skip_reason_counts"] = {PARTIAL_REASON_MISSING_REVISED: 1}
    return records


def _make_multi_event_seq27_records() -> list[dict[str, Any]]:
    records = _base_records()
    events = [
        {
            "proposal_id": 970000101,
            "seq_id": 27,
            "depth": 1,
            "accepted_len": 1,
            "revised_len": 1,
            "recovered_token_count": 2,
            "before_len": 100,
            "after_len": 102,
            "expected_delta": 2,
            "actual_delta": 2,
            "target_frontier_before": 100,
            "target_frontier_after": 102,
            "draft_frontier_before": 100,
            "draft_frontier_after": 102,
        },
        {
            "proposal_id": 970000202,
            "seq_id": 27,
            "depth": 2,
            "accepted_len": 2,
            "revised_len": 1,
            "recovered_token_count": 3,
            "before_len": 110,
            "after_len": 113,
            "expected_delta": 3,
            "actual_delta": 3,
            "target_frontier_before": 110,
            "target_frontier_after": 113,
            "draft_frontier_before": 110,
            "draft_frontier_after": 113,
        },
    ]
    for record in records:
        record["partial_prefix_recovery_enabled"] = True
        record["enable_rolling_continuous_partial_prefix_recovery"] = True
        record["partial_prefix_recovery_attempt_count"] = len(events)
        record["partial_prefix_recovery_success_count"] = len(events)
        record["partial_prefix_recovery_skip_reason_counts"] = {}
        record["partial_prefix_recovered_proposal_ids"] = [event["proposal_id"] for event in events]
        record["partial_prefix_recovered_seq_ids"] = [27]
        record["partial_prefix_recovered_depth_by_proposal_id"] = {
            str(event["proposal_id"]): event["depth"] for event in events
        }
        record["partial_prefix_recovered_seq_id_by_proposal_id"] = {
            str(event["proposal_id"]): event["seq_id"] for event in events
        }
        record["partial_prefix_accepted_len_by_proposal_id"] = {
            str(event["proposal_id"]): event["accepted_len"] for event in events
        }
        record["partial_prefix_reject_index_by_proposal_id"] = {
            str(event["proposal_id"]): event["accepted_len"] for event in events
        }
        record["partial_prefix_revised_token_count_by_proposal_id"] = {
            str(event["proposal_id"]): event["revised_len"] for event in events
        }
        record["partial_prefix_committed_token_count_by_proposal_id"] = {
            str(event["proposal_id"]): event["recovered_token_count"] for event in events
        }
        record["partial_prefix_recovery_frontier_before_by_proposal_id"] = {
            str(event["proposal_id"]): event["before_len"] for event in events
        }
        record["partial_prefix_recovery_frontier_after_by_proposal_id"] = {
            str(event["proposal_id"]): event["after_len"] for event in events
        }
        record["partial_recovery_target_seq_len_before_by_proposal_id"] = {
            str(event["proposal_id"]): event["target_frontier_before"] for event in events
        }
        record["partial_recovery_target_seq_len_after_by_proposal_id"] = {
            str(event["proposal_id"]): event["target_frontier_after"] for event in events
        }
        record["partial_recovery_draft_seq_len_before_by_proposal_id"] = {
            str(event["proposal_id"]): event["draft_frontier_before"] for event in events
        }
        record["partial_recovery_draft_seq_len_after_by_proposal_id"] = {
            str(event["proposal_id"]): event["draft_frontier_after"] for event in events
        }
        record["partial_prefix_recovery_frontier_before_by_seq_id"] = {"27": 100}
        record["partial_prefix_recovery_frontier_after_by_seq_id"] = {"27": 113}
        record["partial_recovery_target_seq_len_before_by_seq_id"] = {"27": 100}
        record["partial_recovery_target_seq_len_after_by_seq_id"] = {"27": 113}
        record["partial_recovery_draft_seq_len_before_by_seq_id"] = {"27": 100}
        record["partial_recovery_draft_seq_len_after_by_seq_id"] = {"27": 113}
        record["partial_prefix_descendant_cascade_discard_count_by_proposal_id"] = {
            str(event["proposal_id"]): 0 for event in events
        }
        record["partial_prefix_recovery_normal_release_seq_ids"] = [27]
        record["partial_recovery_target_draft_len_match_by_seq_id"] = {"27": True}
        record["partial_recovery_target_draft_token_match_by_seq_id"] = {"27": True}
        record["partial_recovery_cascade_discarded_descendant_proposal_ids"] = []
        record["partial_recovery_cascade_discarded_descendant_depth_by_proposal_id"] = {}
        record["partial_recovery_cascade_discarded_descendant_reason_by_proposal_id"] = {}
        record["partial_prefix_recovery_event_count"] = len(events)
        record["partial_prefix_recovery_events"] = list(events)
    return records


def _make_bounded_full_accept_records() -> list[dict[str, Any]]:
    records = _base_records()
    for record in records:
        add_depth4_shadow(record)
        add_depth4_commit(record)
    return records


def _make_bounded_partial_plus_full_records() -> list[dict[str, Any]]:
    records = _make_bounded_full_accept_records()
    _apply_partial_recovery(
        records,
        proposal_id=910000103,
        depth=3,
        accepted_len=1,
        revised_count=1,
        frontier_before=44,
    )
    return records


def _add_full_continuous_fields(
    records: list[dict[str, Any]],
    *,
    full_total: int = 484,
    partial_total: int = 0,
    revised_total: int = 0,
    output_total: int | None = None,
    include_generic_accounting: bool = True,
    tail_token_count: int = 440,
) -> None:
    output = full_total + partial_total if output_total is None else output_total
    depth_commit_counts = {
        "0": 12,
        "1": 8,
        "2": 8,
        "3": 8,
        "4": 8,
        "5": max(0, int(tail_token_count)),
    }
    for record in records:
        record["generic_full_continuous_enabled"] = True
        record["enable_full_continuous_eager"] = True
        record["generic_rolling_runtime_enabled"] = True
        record["enable_generic_rolling_runtime_loop"] = True
        record["generic_rolling_apply_path_enabled"] = True
        record["enable_generic_rolling_apply_path"] = True
        record["generic_full_continuous_max_depth"] = 100
        record["generic_full_continuous_max_observed_depth"] = 60
        record["generic_full_continuous_max_real_committed_depth"] = 60
        record["generic_full_continuous_depth_commit_token_counts"] = dict(depth_commit_counts)
        record["generic_full_continuous_total_full_commit_token_count"] = int(full_total)
        record["generic_full_continuous_total_partial_recovered_token_count"] = int(partial_total)
        record["generic_full_continuous_total_revised_token_count"] = int(revised_total)
        record["generic_full_continuous_total_output_token_count"] = int(output)
        record["generic_full_continuous_normal_lane_conflict_count"] = 0
        record["generic_full_continuous_target_draft_mismatch_count"] = 0
        record["generic_full_continuous_depth_gt_max_real_commit_count"] = 0
        record["generic_full_continuous_parity_ok"] = True
        if include_generic_accounting:
            record["generic_rolling_real_committed_proposal_ids_by_depth"] = {"5": [950000005]}
            record["generic_rolling_real_committed_token_count_by_depth"] = {"5": int(tail_token_count)}
            record["generic_rolling_real_committed_token_count_by_proposal_id"] = {
                "950000005": int(tail_token_count)
            }
            record["generic_rolling_real_commit_depth_by_proposal_id"] = {"950000005": 5}
            record["generic_rolling_real_commit_action_by_proposal_id"] = {"950000005": "commit"}
            record["generic_rolling_real_commit_verify_result_by_proposal_id"] = {
                "950000005": "full_accept"
            }


def _make_full_continuous_records(
    *,
    partial: bool = False,
    include_generic_accounting: bool = True,
    tail_token_count: int = 440,
    full_total: int = 484,
    partial_total_override: int | None = None,
    revised_total_override: int | None = None,
) -> list[dict[str, Any]]:
    records = _make_bounded_full_accept_records()
    partial_total = 0
    revised_total = 0
    if partial:
        _apply_partial_recovery(
            records,
            proposal_id=950000006,
            depth=6,
            accepted_len=1,
            revised_count=1,
            frontier_before=484,
        )
        partial_total = 2
        revised_total = 1
    if partial_total_override is not None:
        partial_total = int(partial_total_override)
    if revised_total_override is not None:
        revised_total = int(revised_total_override)
    _add_full_continuous_fields(
        records,
        full_total=full_total,
        partial_total=partial_total,
        revised_total=revised_total,
        include_generic_accounting=include_generic_accounting,
        tail_token_count=tail_token_count,
    )
    return records


def _add_raw_partial_prefix_evidence(
    records: list[dict[str, Any]],
    *,
    proposal_count: int,
    accepted_len: int,
    revised_count: int,
    depth: int,
    first_proposal_id: int = 970000000,
    first_seq_id: int = 7000,
) -> None:
    proposal_ids = [int(first_proposal_id) + index for index in range(int(proposal_count))]
    seq_ids = [int(first_seq_id) + index for index in range(int(proposal_count))]
    committed_count = int(accepted_len) + int(revised_count)
    for record in records:
        record["runner_role"] = "target"
        record["partial_prefix_recovery_enabled"] = True
        record["enable_rolling_continuous_partial_prefix_recovery"] = True
        record["partial_prefix_recovered_proposal_ids"] = list(proposal_ids)
        record["partial_prefix_recovered_seq_ids"] = list(seq_ids)
        record["partial_prefix_recovered_depth_by_proposal_id"] = {
            str(proposal_id): int(depth) for proposal_id in proposal_ids
        }
        record["partial_prefix_accepted_len_by_proposal_id"] = {
            str(proposal_id): int(accepted_len) for proposal_id in proposal_ids
        }
        record["partial_prefix_reject_index_by_proposal_id"] = {
            str(proposal_id): int(accepted_len) for proposal_id in proposal_ids
        }
        record["partial_prefix_revised_token_count_by_proposal_id"] = {
            str(proposal_id): int(revised_count) for proposal_id in proposal_ids
        }
        record["partial_prefix_committed_token_count_by_proposal_id"] = {
            str(proposal_id): int(committed_count) for proposal_id in proposal_ids
        }
        record["partial_prefix_recovered_seq_id_by_proposal_id"] = {
            str(proposal_id): int(seq_id) for proposal_id, seq_id in zip(proposal_ids, seq_ids)
        }
        before_by_proposal = {
            str(proposal_id): 1000 + index * committed_count
            for index, proposal_id in enumerate(proposal_ids)
        }
        after_by_proposal = {
            str(proposal_id): int(before) + int(committed_count)
            for proposal_id, before in before_by_proposal.items()
        }
        record["partial_prefix_recovery_frontier_before_by_proposal_id"] = dict(before_by_proposal)
        record["partial_prefix_recovery_frontier_after_by_proposal_id"] = dict(after_by_proposal)
        record["partial_recovery_target_seq_len_before_by_proposal_id"] = dict(before_by_proposal)
        record["partial_recovery_target_seq_len_after_by_proposal_id"] = dict(after_by_proposal)
        record["partial_recovery_draft_seq_len_before_by_proposal_id"] = dict(before_by_proposal)
        record["partial_recovery_draft_seq_len_after_by_proposal_id"] = dict(after_by_proposal)
        record["partial_prefix_recovery_frontier_before_by_seq_id"] = {
            str(seq_id): int(before_by_proposal[str(proposal_id)])
            for proposal_id, seq_id in zip(proposal_ids, seq_ids)
        }
        record["partial_prefix_recovery_frontier_after_by_seq_id"] = {
            str(seq_id): int(after_by_proposal[str(proposal_id)])
            for proposal_id, seq_id in zip(proposal_ids, seq_ids)
        }
        record["partial_prefix_recovery_normal_release_seq_ids"] = list(seq_ids)
        record["partial_recovery_target_draft_len_match_by_seq_id"] = {
            str(seq_id): True for seq_id in seq_ids
        }
        record["partial_recovery_target_draft_token_match_by_seq_id"] = {
            str(seq_id): True for seq_id in seq_ids
        }
        record["partial_prefix_recovery_attempt_count"] = int(proposal_count)
        record["partial_prefix_recovery_success_count"] = int(proposal_count)
        record["partial_prefix_recovery_skip_reason_counts"] = {}


def assert_pass(name: str, records: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if errors:
        raise SystemExit(f"synthetic {name} failed: {errors}\nsummary={summary}")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise SystemExit(f"synthetic {name} {key} mismatch: expected {value}, got {summary.get(key)}")


def assert_fail(name: str, records: list[dict[str, Any]], needle: str) -> None:
    errors, summary = validate_records(records, synthetic_result_payload())
    if not errors:
        raise SystemExit(f"synthetic {name} should fail\nsummary={summary}")
    if not any(needle in error for error in errors):
        raise SystemExit(f"synthetic {name} failed for wrong reason: {errors}\nsummary={summary}")


def run_synthetic() -> None:
    full_accept = _make_bounded_full_accept_records()
    assert_pass(
        "bounded 8q/8t baseline",
        full_accept,
        {
            "partial_prefix_recovery_success_count": 0,
            "combined_real_committed_token_count": 44,
            "max_real_committed_depth": 4,
        },
    )

    bounded_partial = _make_bounded_partial_plus_full_records()
    assert_pass(
        "bounded partial recovery",
        bounded_partial,
        {
            "partial_prefix_recovery_success_count": 1,
            "partial_prefix_accepted_token_count": 1,
            "partial_prefix_revised_token_count": 1,
            "partial_prefix_total_recovered_token_count": 2,
            "combined_real_committed_token_count": 46,
            "combined_actual_accepted_token_increment_sum": 45,
            "combined_actual_revised_token_increment_sum": 1,
            "combined_actual_output_token_increment_sum": 46,
        },
    )

    full_continuous_baseline = _make_full_continuous_records()
    assert_pass(
        "full continuous baseline depth60",
        full_continuous_baseline,
        {
            "generic_full_continuous_enabled": True,
            "generic_full_continuous_max_depth": 100,
            "generic_full_continuous_max_observed_depth": 60,
            "generic_full_continuous_max_real_committed_depth": 60,
            "generic_full_continuous_total_full_commit_token_count": 484,
            "generic_full_continuous_total_partial_recovered_token_count": 0,
            "generic_full_continuous_total_revised_token_count": 0,
            "generic_full_continuous_total_output_token_count": 484,
            "combined_real_committed_token_count": 484,
            "combined_actual_accepted_token_increment_sum": 484,
            "combined_actual_revised_token_increment_sum": 0,
            "combined_actual_output_token_increment_sum": 484,
        },
    )

    full_continuous_partial = _make_full_continuous_records(partial=True)
    assert_pass(
        "full continuous partial recovery",
        full_continuous_partial,
        {
            "generic_full_continuous_enabled": True,
            "partial_prefix_recovery_success_count": 1,
            "partial_prefix_accepted_token_count": 1,
            "partial_prefix_revised_token_count": 1,
            "partial_prefix_total_recovered_token_count": 2,
            "generic_full_continuous_total_full_commit_token_count": 484,
            "generic_full_continuous_total_partial_recovered_token_count": 2,
            "generic_full_continuous_total_revised_token_count": 1,
            "generic_full_continuous_total_output_token_count": 486,
            "combined_real_committed_token_count": 486,
            "combined_actual_accepted_token_increment_sum": 485,
            "combined_actual_revised_token_increment_sum": 1,
            "combined_actual_output_token_increment_sum": 486,
        },
    )

    full_continuous_raw_overcount = _make_full_continuous_records(
        partial=False,
        full_total=28,
        tail_token_count=0,
        partial_total_override=10,
        revised_total_override=3,
    )
    _add_raw_partial_prefix_evidence(
        full_continuous_raw_overcount,
        proposal_count=31,
        accepted_len=2,
        revised_count=1,
        depth=6,
    )
    for record in full_continuous_raw_overcount:
        record["unified_final_total_partial_recovered_token_count"] = 10
        record["unified_final_total_revised_token_count"] = 3
        record["unified_final_total_output_token_count"] = 38
    assert_pass(
        "full continuous raw partial evidence overcount",
        full_continuous_raw_overcount,
        {
            "generic_full_continuous_enabled": True,
            "generic_full_continuous_total_partial_recovered_token_count": 10,
            "generic_full_continuous_total_revised_token_count": 3,
            "generic_full_continuous_total_output_token_count": 38,
            "partial_prefix_recovery_attempt_count": 3,
            "partial_prefix_recovery_success_count": 3,
            "partial_prefix_recovered_proposal_count": 3,
            "partial_prefix_accepted_token_count": 7,
            "partial_prefix_revised_token_count": 3,
            "partial_prefix_total_recovered_token_count": 10,
            "partial_prefix_raw_recovered_proposal_count": 31,
            "partial_prefix_raw_total_recovered_token_count": 93,
            "partial_prefix_raw_matches_accounting": False,
            "combined_real_committed_token_count": 38,
            "combined_actual_accepted_token_increment_sum": 35,
            "combined_actual_revised_token_increment_sum": 3,
            "combined_actual_output_token_increment_sum": 38,
        },
    )

    unified_alias_partial = _make_full_continuous_records(partial=True)
    for record in unified_alias_partial:
        record.update(
            {
                "unified_generic_rolling_enabled": True,
                "enable_unified_generic_rolling_runtime": True,
                "unified_generic_total_full_commit_token_count": record[
                    "generic_full_continuous_total_full_commit_token_count"
                ],
                "unified_generic_total_partial_recovered_token_count": record[
                    "generic_full_continuous_total_partial_recovered_token_count"
                ],
                "unified_generic_total_revised_token_count": record[
                    "generic_full_continuous_total_revised_token_count"
                ],
                "unified_generic_total_output_token_count": record[
                    "generic_full_continuous_total_output_token_count"
                ],
                "unified_final_total_partial_recovered_token_count": record[
                    "generic_full_continuous_total_partial_recovered_token_count"
                ],
                "unified_final_total_revised_token_count": record[
                    "generic_full_continuous_total_revised_token_count"
                ],
                "unified_final_total_output_token_count": record[
                    "generic_full_continuous_total_output_token_count"
                ],
                "unified_generic_depth_commit_token_counts": dict(
                    record["generic_full_continuous_depth_commit_token_counts"]
                ),
                "combined_real_committed_token_count": 1244,
            }
        )
        for key in (
            "generic_full_continuous_total_full_commit_token_count",
            "generic_full_continuous_total_partial_recovered_token_count",
            "generic_full_continuous_total_revised_token_count",
            "generic_full_continuous_total_output_token_count",
            "generic_full_continuous_depth_commit_token_counts",
        ):
            record.pop(key, None)
    assert_pass(
        "unified alias partial recovery authority",
        unified_alias_partial,
        {
            "generic_full_continuous_total_output_token_count": 486,
            "combined_real_committed_token_count": 486,
            "combined_actual_accepted_token_increment_sum": 485,
            "combined_actual_revised_token_increment_sum": 1,
            "combined_actual_output_token_increment_sum": 486,
        },
    )

    full_continuous_total_authority = _make_full_continuous_records(include_generic_accounting=False)
    for record in full_continuous_total_authority:
        record["unified_final_total_full_commit_token_count"] = 484
        record["unified_final_total_partial_recovered_token_count"] = 0
        record["unified_final_total_revised_token_count"] = 0
        record["unified_final_total_output_token_count"] = 484
    assert_pass(
        "full continuous total-output authority without raw generic commits",
        full_continuous_total_authority,
        {
            "generic_full_continuous_total_output_token_count": 484,
            "combined_real_committed_token_count": 484,
            "combined_actual_accepted_token_increment_sum": 484,
            "combined_actual_revised_token_increment_sum": 0,
            "combined_actual_output_token_increment_sum": 484,
        },
    )

    full_continuous_stale_legacy_double_count = _make_full_continuous_records(tail_token_count=484)
    for record in full_continuous_stale_legacy_double_count:
        record["unified_final_total_full_commit_token_count"] = 484
        record["unified_final_total_partial_recovered_token_count"] = 0
        record["unified_final_total_revised_token_count"] = 0
        record["unified_final_total_output_token_count"] = 484
    assert_pass(
        "full continuous ignores stale legacy double count",
        full_continuous_stale_legacy_double_count,
        {
            "generic_full_continuous_total_output_token_count": 484,
            "combined_real_committed_token_count": 484,
            "combined_actual_accepted_token_increment_sum": 484,
            "combined_actual_revised_token_increment_sum": 0,
            "combined_actual_output_token_increment_sum": 484,
        },
    )

    full_continuous_bad_revised = _make_full_continuous_records(
        partial=False,
        partial_total_override=2,
        revised_total_override=1,
    )
    for record in full_continuous_bad_revised:
        record["unified_final_total_partial_recovered_token_count"] = 2
        record["unified_final_total_revised_token_count"] = 1
        record["unified_final_total_output_token_count"] = 486
    assert_fail(
        "full continuous bad revised accounting",
        full_continuous_bad_revised,
        "partial recovery tokens require",
    )

    depth1_partial = _make_depth1_partial_records()
    assert_pass(
        "depth1 partial recovery",
        depth1_partial,
        {
            "partial_prefix_recovery_success_count": 1,
            "partial_prefix_accepted_token_count": 2,
            "partial_prefix_revised_token_count": 1,
            "partial_prefix_total_recovered_token_count": 3,
            "combined_real_committed_token_count": 15,
        },
    )

    depth3_partial = _make_depth3_partial_records()
    assert_pass(
        "depth3 partial recovery",
        depth3_partial,
        {
            "partial_prefix_recovery_success_count": 1,
            "partial_prefix_accepted_token_count": 1,
            "partial_prefix_revised_token_count": 1,
            "partial_prefix_total_recovered_token_count": 2,
            "combined_real_committed_token_count": 30,
        },
    )

    first_token_reject = _make_depth1_partial_records(accepted_len=0, revised_count=1)
    assert_pass(
        "first-token reject recovery",
        first_token_reject,
        {
            "partial_prefix_recovery_success_count": 1,
            "partial_prefix_accepted_token_count": 0,
            "partial_prefix_revised_token_count": 1,
            "partial_prefix_total_recovered_token_count": 1,
        },
    )

    multi_seq27 = _make_multi_event_seq27_records()
    assert_pass(
        "multiple partial recoveries on seq 27",
        multi_seq27,
        {
            "partial_prefix_recovery_success_count": 2,
            "partial_prefix_accepted_token_count": 3,
            "partial_prefix_revised_token_count": 2,
            "partial_prefix_total_recovered_token_count": 5,
            "partial_recovery_seq27_event_count": 2,
        },
    )

    missing_revised = _make_missing_revised_records()
    assert_pass(
        "missing revised token fallback",
        missing_revised,
        {
            "partial_prefix_recovery_success_count": 0,
            "partial_prefix_total_recovered_token_count": 0,
            "combined_real_committed_token_count": 36,
        },
    )

    bad_descendant = _make_depth1_partial_records()
    for record in bad_descendant:
        record["rolling_depth2_real_committed_proposal_ids"] = [900000102]
        record["rolling_depth2_real_committed_token_count_by_proposal_id"] = {"900000102": 8}
    assert_fail("bad descendant commit", bad_descendant, "descendant committed")

    bad_accounting = _make_depth1_partial_records()
    for record in bad_accounting:
        record["partial_prefix_committed_token_count_by_proposal_id"] = {"900000101": 4}
    assert_fail("bad partial accounting", bad_accounting, "prefix plus revised")

    bad_sync = _make_depth1_partial_records()
    for record in bad_sync:
        record["partial_recovery_target_draft_len_match_by_seq_id"] = {"7": False}
    assert_fail("bad target/draft sync", bad_sync, "length mismatch")

    print("Synthetic eager partial-prefix recovery checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1H-8q eager partial-prefix recovery traces.")
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic()
        return 0

    records = load_trace(args.trace)
    result_payload = load_json(args.result) if args.result is not None else {}
    errors, summary = validate_records(records, result_payload)
    print_summary(summary)
    if errors:
        print("check_status=fail")
        print(json.dumps({"errors": errors}, indent=2, sort_keys=True))
        return 1
    print("check_status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
