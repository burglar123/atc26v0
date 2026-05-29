#!/usr/bin/env python3
"""Generic legacy trace parser for bounded rolling proposal chains.

The parser is read-only and consumes existing Phase 1H legacy trace fields.
It does not require or emit any runtime trace changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


MAX_LEGACY_REAL_DEPTH = 3


@dataclass
class RollingProposalNode:
    proposal_id: int
    seq_id: int | None
    depth: int
    parent_id: int | None
    root_id: int | None
    source: str
    status: str
    status_reason: str | None
    token_count: int
    accept_len: int | None
    verify_result: str | None
    action: str | None
    committed: bool
    ready_shadow: bool
    generated: bool
    invalidated: bool
    cascade_discarded: bool
    stale_or_frontier_mismatch: bool
    target_side_seen: bool
    draft_side_seen: bool


@dataclass
class RollingChainRegistry:
    nodes_by_id: dict[int, RollingProposalNode] = field(default_factory=dict)
    children_by_parent: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    committed_by_depth: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    ready_by_depth: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    generated_by_depth: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    invalidated_by_depth: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    cascade_by_depth: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    token_by_depth: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    seq_by_depth: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    parent_by_depth: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    root_by_depth: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    accept_len_by_depth: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    action_by_depth: dict[int, dict[int, str]] = field(default_factory=lambda: defaultdict(dict))
    verify_result_by_depth: dict[int, dict[int, str]] = field(default_factory=lambda: defaultdict(dict))
    declared_depth_by_id: dict[int, int] = field(default_factory=dict)
    max_observed_depth: int = 0
    max_real_committed_depth: int = 0
    max_configured_depth: int = 0
    gamma: int = 0
    flags: dict[str, bool] = field(default_factory=dict)
    depth4_real_commit_count: int = 0
    depth_gt3_real_commit_count: int = 0
    higher_depth_commit_ids: set[int] = field(default_factory=set)
    normal_lane_conflict_count: int = 0
    missing_buffered_proposal_unexpected_count: int = 0
    duplicate_commit_ids: set[int] = field(default_factory=set)
    duplicate_commit_seq_ids: set[int] = field(default_factory=set)
    duplicate_seq_depth_events: set[tuple[int, str, int, int, int]] = field(default_factory=set)
    committed_without_ready_ids: set[int] = field(default_factory=set)
    committed_without_parent_ids: set[int] = field(default_factory=set)
    invalid_committed_ids: set[int] = field(default_factory=set)
    cascade_committed_ids: set[int] = field(default_factory=set)
    non_full_accept_ids: set[int] = field(default_factory=set)
    stale_or_frontier_ids: set[int] = field(default_factory=set)
    target_draft_length_mismatch_count: int = 0
    target_draft_token_mismatch_count: int = 0
    target_side_seen_by_id: set[int] = field(default_factory=set)
    draft_side_seen_by_id: set[int] = field(default_factory=set)


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


def as_bool_map(value: Any) -> dict[int, bool]:
    if not isinstance(value, dict):
        return {}
    result: dict[int, bool] = {}
    for key, item in value.items():
        try:
            result[int(key)] = bool(item)
        except Exception:
            continue
    return result


def merge_first(target: dict[int, int], source: dict[int, int]) -> None:
    for key, value in source.items():
        target.setdefault(key, value)


def merge_positive(target: dict[int, int], source: dict[int, int]) -> None:
    for key, value in source.items():
        if value > 0:
            target.setdefault(key, value)


def merge_str_first(target: dict[int, str], source: dict[int, str]) -> None:
    for key, value in source.items():
        target.setdefault(key, value)


def add_seq_map(
    registry: RollingChainRegistry,
    depth: int,
    proposal_ids: list[int],
    seq_ids: list[int],
) -> None:
    for proposal_id, seq_id in zip(proposal_ids, seq_ids):
        registry.seq_by_depth[depth].setdefault(proposal_id, seq_id)


def add_id_set(
    registry: RollingChainRegistry,
    depth: int,
    ids: set[int],
    bucket: dict[int, set[int]],
) -> None:
    if not ids:
        return
    bucket[depth].update(ids)
    for proposal_id in ids:
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
        registry.max_observed_depth = max(registry.max_observed_depth, depth)


def collect_false_ids(value: Any) -> set[int]:
    return {key for key, item in as_bool_map(value).items() if item is False}


def false_count(value: Any) -> int:
    return len(collect_false_ids(value))


def mark_side_seen(
    registry: RollingChainRegistry,
    side_events: dict[tuple[int, str, int], set[tuple[int, int]]],
    record: dict[str, Any],
    *,
    depth: int,
    side_field: str,
    plan_field: str,
    step_field: str,
    id_field: str,
) -> None:
    side = str(record.get(side_field) or "")
    if side not in {"target", "draft"}:
        return
    plan_id = int_value(record.get(plan_field), int_value(record.get("plan_id"), -1))
    step_id = int_value(record.get(step_field), int_value(record.get("step_id"), -1))
    ids = as_int_set(record.get(id_field))
    if side == "target":
        registry.target_side_seen_by_id.update(ids)
    else:
        registry.draft_side_seen_by_id.update(ids)
    for proposal_id in ids:
        side_events[(depth, side, proposal_id)].add((plan_id, step_id))


def update_depth_maps_from_record(registry: RollingChainRegistry, record: dict[str, Any]) -> None:
    for proposal_id, depth in as_int_map(record.get("continuous_eager_chain_depth_by_proposal_id")).items():
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
    for proposal_id, depth in as_int_map(record.get("rolling_child_depth_by_proposal_id")).items():
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
    for proposal_id, depth in as_int_map(record.get("rolling_chain_depth_by_proposal_id")).items():
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
    for proposal_id, depth in as_int_map(record.get("rolling_depth2_real_commit_depth_by_proposal_id")).items():
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
    for proposal_id, depth in as_int_map(record.get("rolling_depth3_child_depth_by_proposal_id")).items():
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
    for proposal_id, depth in as_int_map(record.get("rolling_depth3_real_commit_depth_by_proposal_id")).items():
        registry.declared_depth_by_id.setdefault(proposal_id, depth)
        if depth > MAX_LEGACY_REAL_DEPTH:
            registry.higher_depth_commit_ids.add(proposal_id)


def parse_legacy_rolling_chain(records: list[dict[str, Any]]) -> RollingChainRegistry:
    registry = RollingChainRegistry(
        flags={
            "one_shot_commit_enabled": False,
            "continuous_depth1_commit_enabled": False,
            "rolling_depth2_commit_enabled": False,
            "rolling_depth3_shadow_enabled": False,
            "rolling_depth3_commit_enabled": False,
        }
    )
    side_events: dict[tuple[int, str, int], set[tuple[int, int]]] = defaultdict(set)
    seq_depth_events: set[tuple[int, str, int, int, int]] = set()
    stale_or_frontier_ids: set[int] = set()

    for record in records:
        registry.gamma = max(registry.gamma, int_value(record.get("normal_gamma"), 0))
        registry.max_configured_depth = max(
            registry.max_configured_depth,
            int_value(record.get("max_rolling_continuous_depth"), 0),
        )
        registry.max_observed_depth = max(
            registry.max_observed_depth,
            int_value(record.get("rolling_max_depth_observed"), 0),
            int_value(record.get("max_rolling_continuous_depth_observed"), 0),
            int_value(record.get("rolling_depth3_max_depth_observed"), 0),
        )
        registry.flags["one_shot_commit_enabled"] = registry.flags["one_shot_commit_enabled"] or bool(
            record.get("enable_eager_commit_ready_only", False)
        ) or bool(record.get("eager_commit_enabled", False))
        registry.flags["continuous_depth1_commit_enabled"] = registry.flags[
            "continuous_depth1_commit_enabled"
        ] or bool(record.get("enable_continuous_eager_commit_depth1_ready_only", False)) or bool(
            record.get("continuous_eager_commit_enabled", False)
        )
        registry.flags["rolling_depth2_commit_enabled"] = registry.flags["rolling_depth2_commit_enabled"] or bool(
            record.get("enable_rolling_continuous_depth2_commit_ready_only", False)
        ) or bool(record.get("rolling_depth2_commit_enabled", False))
        registry.flags["rolling_depth3_shadow_enabled"] = registry.flags["rolling_depth3_shadow_enabled"] or bool(
            record.get("enable_rolling_continuous_depth3_shadow_dry_run", False)
        ) or bool(record.get("rolling_depth3_shadow_enabled", False))
        registry.flags["rolling_depth3_commit_enabled"] = registry.flags["rolling_depth3_commit_enabled"] or bool(
            record.get("enable_rolling_continuous_depth3_commit_ready_only", False)
        ) or bool(record.get("rolling_depth3_commit_enabled", False))

        registry.missing_buffered_proposal_unexpected_count += int_value(
            record.get("missing_buffered_proposal_unexpected_count"), 0
        )
        registry.missing_buffered_proposal_unexpected_count += len(
            as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        )
        registry.normal_lane_conflict_count += int_value(record.get("rolling_normal_lane_conflict_count"), 0)
        registry.normal_lane_conflict_count += len(as_int_set(record.get("rolling_normal_lane_conflict_seq_ids")))
        registry.normal_lane_conflict_count += int_value(record.get("rolling_depth3_normal_lane_conflict_count"), 0)
        registry.normal_lane_conflict_count += len(
            as_int_set(record.get("rolling_depth3_normal_lane_conflict_seq_ids"))
        )
        registry.depth4_real_commit_count += int_value(record.get("rolling_depth4_real_commit_count"), 0)
        registry.depth_gt3_real_commit_count += int_value(record.get("rolling_depth_gt3_real_commit_count"), 0)
        registry.higher_depth_commit_ids.update(as_int_set(record.get("rolling_depth4_real_committed_proposal_ids")))
        registry.higher_depth_commit_ids.update(as_int_set(record.get("rolling_depth_gt3_real_committed_proposal_ids")))

        for field_name in (
            "eager_commit_target_draft_len_match_by_seq_id",
            "continuous_eager_target_draft_len_match_by_seq_id",
            "rolling_depth2_target_draft_len_match_by_seq_id",
            "rolling_depth3_target_draft_len_match_by_seq_id",
        ):
            false_ids = collect_false_ids(record.get(field_name))
            registry.target_draft_length_mismatch_count += len(false_ids)
            stale_or_frontier_ids.update(false_ids)
        for field_name in (
            "eager_commit_target_draft_token_match_by_seq_id",
            "continuous_eager_target_draft_token_match_by_seq_id",
            "rolling_depth2_target_draft_token_match_by_seq_id",
            "rolling_depth3_target_draft_token_match_by_seq_id",
        ):
            registry.target_draft_token_mismatch_count += false_count(record.get(field_name))

        eager_ids = as_int_list(record.get("eager_committed_proposal_ids"))
        add_id_set(registry, 0, set(eager_ids), registry.committed_by_depth)
        add_seq_map(registry, 0, eager_ids, as_int_list(record.get("eager_committed_seq_ids")))
        merge_positive(registry.token_by_depth[0], as_int_map(record.get("eager_committed_token_count_by_proposal_id")))
        merge_positive(registry.accept_len_by_depth[0], as_int_map(record.get("eager_committed_accept_len_by_proposal_id")))
        merge_str_first(registry.action_by_depth[0], as_str_map(record.get("eager_committed_action_by_proposal_id")))
        merge_str_first(
            registry.verify_result_by_depth[0],
            as_str_map(record.get("eager_committed_verify_result_by_proposal_id")),
        )
        for proposal_id in eager_ids:
            registry.root_by_depth[0].setdefault(proposal_id, proposal_id)
        registry.duplicate_commit_ids.update(as_int_set(record.get("eager_commit_duplicate_proposal_ids")))
        registry.duplicate_commit_seq_ids.update(as_int_set(record.get("eager_commit_duplicate_seq_ids")))
        mark_side_seen(
            registry,
            side_events,
            record,
            depth=0,
            side_field="eager_commit_side",
            plan_field="eager_commit_plan_id",
            step_field="eager_commit_step_id",
            id_field="eager_committed_proposal_ids",
        )

        depth1_committed = as_int_list(record.get("continuous_eager_real_committed_proposal_ids"))
        depth1_generated = as_int_set(record.get("continuous_eager_candidate_proposal_ids"))
        depth1_ready = as_int_set(record.get("continuous_eager_commit_ready_shadow_proposal_ids"))
        add_id_set(registry, 1, set(depth1_committed), registry.committed_by_depth)
        add_id_set(registry, 1, depth1_generated, registry.generated_by_depth)
        add_id_set(registry, 1, depth1_ready, registry.ready_by_depth)
        add_seq_map(registry, 1, depth1_committed, as_int_list(record.get("continuous_eager_real_committed_seq_ids")))
        add_seq_map(
            registry,
            1,
            as_int_list(record.get("continuous_eager_candidate_proposal_ids")),
            as_int_list(record.get("continuous_eager_candidate_seq_ids")),
        )
        merge_positive(
            registry.token_by_depth[1],
            as_int_map(record.get("continuous_eager_real_committed_token_count_by_proposal_id")),
        )
        merge_positive(
            registry.token_by_depth[1],
            as_int_map(record.get("continuous_eager_candidate_token_count_by_proposal_id")),
        )
        merge_positive(
            registry.accept_len_by_depth[1],
            as_int_map(record.get("continuous_eager_real_committed_accept_len_by_proposal_id")),
        )
        merge_first(registry.parent_by_depth[1], as_int_map(record.get("continuous_eager_parent_proposal_id_by_proposal_id")))
        merge_first(registry.root_by_depth[1], as_int_map(record.get("continuous_eager_root_proposal_id_by_proposal_id")))
        merge_str_first(
            registry.action_by_depth[1],
            as_str_map(record.get("continuous_eager_real_commit_action_by_proposal_id")),
        )
        merge_str_first(
            registry.verify_result_by_depth[1],
            as_str_map(record.get("continuous_eager_real_commit_verify_result_by_proposal_id")),
        )
        depth1_invalid = set()
        depth1_invalid.update(as_int_set(record.get("continuous_eager_not_ready_shadow_proposal_ids")))
        depth1_invalid.update(as_int_set(record.get("continuous_eager_partial_reject_proposal_ids")))
        add_id_set(registry, 1, depth1_invalid, registry.invalidated_by_depth)
        stale_or_frontier_ids.update(as_int_set(record.get("continuous_eager_frontier_mismatch_proposal_ids")))
        stale_or_frontier_ids.update(as_int_set(record.get("continuous_eager_true_frontier_mismatch_proposal_ids")))
        mark_side_seen(
            registry,
            side_events,
            record,
            depth=1,
            side_field="continuous_eager_commit_side",
            plan_field="continuous_eager_commit_plan_id",
            step_field="continuous_eager_commit_step_id",
            id_field="continuous_eager_real_committed_proposal_ids",
        )

        depth2_committed = as_int_list(record.get("rolling_depth2_real_committed_proposal_ids"))
        depth2_generated = set()
        for field_name in ("rolling_child_candidate_proposal_ids", "rolling_child_generated_proposal_ids"):
            depth2_generated.update(as_int_set(record.get(field_name)))
        depth2_ready = set()
        for field_name in ("rolling_child_ready_shadow_proposal_ids", "rolling_child_ready_after_parent_full_accept_proposal_ids"):
            depth2_ready.update(as_int_set(record.get(field_name)))
        add_id_set(registry, 2, set(depth2_committed), registry.committed_by_depth)
        add_id_set(registry, 2, depth2_generated, registry.generated_by_depth)
        add_id_set(registry, 2, depth2_ready, registry.ready_by_depth)
        add_seq_map(registry, 2, depth2_committed, as_int_list(record.get("rolling_depth2_real_committed_seq_ids")))
        add_seq_map(
            registry,
            2,
            as_int_list(record.get("rolling_child_generated_proposal_ids")),
            as_int_list(record.get("rolling_child_generated_seq_ids")),
        )
        add_seq_map(
            registry,
            2,
            as_int_list(record.get("rolling_child_candidate_proposal_ids")),
            as_int_list(record.get("rolling_child_candidate_seq_ids")),
        )
        add_seq_map(
            registry,
            2,
            as_int_list(record.get("rolling_child_generated_proposal_ids")),
            as_int_list(record.get("draft_rolling_eager_draft_seq_ids")),
        )
        merge_positive(
            registry.token_by_depth[2],
            as_int_map(record.get("rolling_child_token_count_by_proposal_id")),
        )
        merge_positive(
            registry.token_by_depth[2],
            as_int_map(record.get("rolling_depth2_real_committed_token_count_by_proposal_id")),
        )
        merge_positive(
            registry.accept_len_by_depth[2],
            as_int_map(record.get("rolling_depth2_real_committed_accept_len_by_proposal_id")),
        )
        for field_name in (
            "rolling_child_parent_by_proposal_id",
            "rolling_chain_parent_by_proposal_id",
            "rolling_depth2_real_commit_parent_by_proposal_id",
        ):
            merge_first(registry.parent_by_depth[2], as_int_map(record.get(field_name)))
        for field_name in (
            "rolling_child_root_by_proposal_id",
            "rolling_chain_root_by_proposal_id",
            "rolling_depth2_real_commit_root_by_proposal_id",
        ):
            merge_first(registry.root_by_depth[2], as_int_map(record.get(field_name)))
        merge_str_first(
            registry.action_by_depth[2],
            as_str_map(record.get("rolling_depth2_real_commit_action_by_proposal_id")),
        )
        merge_str_first(
            registry.verify_result_by_depth[2],
            as_str_map(record.get("rolling_depth2_real_commit_verify_result_by_proposal_id")),
        )
        add_id_set(
            registry,
            2,
            as_int_set(record.get("rolling_child_invalidated_proposal_ids")),
            registry.invalidated_by_depth,
        )
        add_id_set(
            registry,
            2,
            as_int_set(record.get("rolling_cascade_discarded_proposal_ids")),
            registry.cascade_by_depth,
        )
        registry.duplicate_commit_ids.update(as_int_set(record.get("rolling_depth2_real_commit_duplicate_proposal_ids")))
        registry.duplicate_commit_seq_ids.update(as_int_set(record.get("rolling_depth2_real_commit_duplicate_seq_ids")))
        registry.committed_without_ready_ids.update(as_int_set(record.get("rolling_depth2_committed_without_ready_shadow_ids")))
        registry.committed_without_parent_ids.update(
            as_int_set(record.get("rolling_depth2_committed_without_parent_depth1_commit_ids"))
        )
        registry.invalid_committed_ids.update(as_int_set(record.get("rolling_depth2_committed_invalidated_child_ids")))
        registry.cascade_committed_ids.update(as_int_set(record.get("rolling_depth2_committed_cascade_discarded_child_ids")))
        registry.non_full_accept_ids.update(as_int_set(record.get("rolling_depth2_committed_non_full_accept_ids")))
        mark_side_seen(
            registry,
            side_events,
            record,
            depth=2,
            side_field="rolling_depth2_commit_side",
            plan_field="rolling_depth2_commit_plan_id",
            step_field="rolling_depth2_commit_step_id",
            id_field="rolling_depth2_real_committed_proposal_ids",
        )

        depth3_committed = as_int_list(record.get("rolling_depth3_real_committed_proposal_ids"))
        depth3_generated = as_int_set(record.get("rolling_depth3_child_generated_proposal_ids"))
        depth3_ready = as_int_set(record.get("rolling_depth3_child_ready_shadow_proposal_ids"))
        add_id_set(registry, 3, set(depth3_committed), registry.committed_by_depth)
        add_id_set(registry, 3, depth3_generated, registry.generated_by_depth)
        add_id_set(registry, 3, depth3_ready, registry.ready_by_depth)
        add_seq_map(registry, 3, depth3_committed, as_int_list(record.get("rolling_depth3_real_committed_seq_ids")))
        add_seq_map(
            registry,
            3,
            as_int_list(record.get("rolling_depth3_child_generated_proposal_ids")),
            as_int_list(record.get("rolling_depth3_child_generated_seq_ids")),
        )
        merge_positive(
            registry.token_by_depth[3],
            as_int_map(record.get("rolling_depth3_child_token_count_by_proposal_id")),
        )
        merge_positive(
            registry.token_by_depth[3],
            as_int_map(record.get("rolling_depth3_real_committed_token_count_by_proposal_id")),
        )
        merge_positive(
            registry.accept_len_by_depth[3],
            as_int_map(record.get("rolling_depth3_real_committed_accept_len_by_proposal_id")),
        )
        for field_name in (
            "rolling_depth3_child_parent_by_proposal_id",
            "rolling_depth3_commit_parent_by_proposal_id",
            "rolling_depth3_real_commit_parent_by_proposal_id",
        ):
            merge_first(registry.parent_by_depth[3], as_int_map(record.get(field_name)))
        for field_name in (
            "rolling_depth3_child_root_by_proposal_id",
            "rolling_depth3_real_commit_root_by_proposal_id",
        ):
            merge_first(registry.root_by_depth[3], as_int_map(record.get(field_name)))
        merge_str_first(
            registry.action_by_depth[3],
            as_str_map(record.get("rolling_depth3_real_commit_action_by_proposal_id")),
        )
        merge_str_first(
            registry.verify_result_by_depth[3],
            as_str_map(record.get("rolling_depth3_real_commit_verify_result_by_proposal_id")),
        )
        add_id_set(
            registry,
            3,
            as_int_set(record.get("rolling_depth3_child_invalidated_proposal_ids")),
            registry.invalidated_by_depth,
        )
        for field_name in (
            "rolling_depth3_child_cascade_discarded_proposal_ids",
            "rolling_depth3_committed_cascade_discarded_child_ids",
        ):
            add_id_set(registry, 3, as_int_set(record.get(field_name)), registry.cascade_by_depth)
        registry.duplicate_commit_ids.update(as_int_set(record.get("rolling_depth3_real_commit_duplicate_proposal_ids")))
        registry.duplicate_commit_seq_ids.update(as_int_set(record.get("rolling_depth3_real_commit_duplicate_seq_ids")))
        registry.committed_without_ready_ids.update(as_int_set(record.get("rolling_depth3_committed_without_ready_shadow_ids")))
        registry.committed_without_parent_ids.update(
            as_int_set(record.get("rolling_depth3_committed_without_parent_depth2_commit_ids"))
        )
        registry.invalid_committed_ids.update(as_int_set(record.get("rolling_depth3_committed_invalidated_child_ids")))
        registry.cascade_committed_ids.update(as_int_set(record.get("rolling_depth3_committed_cascade_discarded_child_ids")))
        registry.non_full_accept_ids.update(as_int_set(record.get("rolling_depth3_committed_non_full_accept_ids")))
        mark_side_seen(
            registry,
            side_events,
            record,
            depth=3,
            side_field="rolling_depth3_commit_side",
            plan_field="rolling_depth3_commit_plan_id",
            step_field="rolling_depth3_commit_step_id",
            id_field="rolling_depth3_real_committed_proposal_ids",
        )

        for field_name in (
            "ready_eager_proposal_stale_ids",
            "ready_eager_proposal_expired_ids",
            "stale_lane_exclusion_decision_ids",
            "expired_lane_exclusion_decision_ids",
        ):
            stale_or_frontier_ids.update(as_int_set(record.get(field_name)))

        update_depth_maps_from_record(registry, record)

    for depth, ids in registry.committed_by_depth.items():
        if ids:
            registry.max_real_committed_depth = max(registry.max_real_committed_depth, depth)
        for proposal_id in ids:
            registry.declared_depth_by_id.setdefault(proposal_id, depth)
            registry.max_observed_depth = max(registry.max_observed_depth, depth)

    for (depth, side, proposal_id), steps in side_events.items():
        if len(steps) > 1:
            registry.duplicate_commit_ids.add(proposal_id)
        for plan_id, step_id in steps:
            seq_id = registry.seq_by_depth[depth].get(proposal_id, -1)
            event = (depth, side, plan_id, step_id, seq_id)
            if event in seq_depth_events:
                registry.duplicate_seq_depth_events.add(event)
            seq_depth_events.add(event)

    registry.stale_or_frontier_ids.update(stale_or_frontier_ids)
    build_nodes(registry)
    return registry


def build_nodes(registry: RollingChainRegistry) -> None:
    all_ids: set[int] = set(registry.declared_depth_by_id)
    for depth in range(0, MAX_LEGACY_REAL_DEPTH + 1):
        all_ids.update(registry.committed_by_depth[depth])
        all_ids.update(registry.ready_by_depth[depth])
        all_ids.update(registry.generated_by_depth[depth])
        all_ids.update(registry.invalidated_by_depth[depth])
        all_ids.update(registry.cascade_by_depth[depth])
        all_ids.update(registry.token_by_depth[depth])
        all_ids.update(registry.seq_by_depth[depth])
        all_ids.update(registry.parent_by_depth[depth])
        all_ids.update(registry.root_by_depth[depth])
        all_ids.update(registry.accept_len_by_depth[depth])
        all_ids.update(registry.action_by_depth[depth])
        all_ids.update(registry.verify_result_by_depth[depth])

    for proposal_id in sorted(all_ids):
        depth = registry.declared_depth_by_id.get(proposal_id)
        if depth is None:
            depth = inferred_depth_for_id(registry, proposal_id)
        committed = proposal_id in registry.committed_by_depth[depth]
        ready = proposal_id in registry.ready_by_depth[depth]
        generated = proposal_id in registry.generated_by_depth[depth]
        invalidated = proposal_id in registry.invalidated_by_depth[depth]
        cascade = proposal_id in registry.cascade_by_depth[depth]
        stale = proposal_id in registry.stale_or_frontier_ids
        status = node_status(committed, ready, generated, invalidated, cascade, stale)
        node = RollingProposalNode(
            proposal_id=proposal_id,
            seq_id=registry.seq_by_depth[depth].get(proposal_id),
            depth=depth,
            parent_id=registry.parent_by_depth[depth].get(proposal_id),
            root_id=registry.root_by_depth[depth].get(proposal_id),
            source=source_for_depth(depth),
            status=status,
            status_reason=status_reason(invalidated, cascade, stale),
            token_count=registry.token_by_depth[depth].get(proposal_id, registry.gamma),
            accept_len=registry.accept_len_by_depth[depth].get(proposal_id),
            verify_result=registry.verify_result_by_depth[depth].get(proposal_id),
            action=registry.action_by_depth[depth].get(proposal_id),
            committed=committed,
            ready_shadow=ready,
            generated=generated,
            invalidated=invalidated,
            cascade_discarded=cascade,
            stale_or_frontier_mismatch=stale,
            target_side_seen=proposal_id in registry.target_side_seen_by_id,
            draft_side_seen=proposal_id in registry.draft_side_seen_by_id,
        )
        registry.nodes_by_id[proposal_id] = node
        if node.parent_id is not None:
            registry.children_by_parent[node.parent_id].add(proposal_id)


def inferred_depth_for_id(registry: RollingChainRegistry, proposal_id: int) -> int:
    for depth in range(0, MAX_LEGACY_REAL_DEPTH + 1):
        if (
            proposal_id in registry.committed_by_depth[depth]
            or proposal_id in registry.ready_by_depth[depth]
            or proposal_id in registry.generated_by_depth[depth]
            or proposal_id in registry.invalidated_by_depth[depth]
            or proposal_id in registry.cascade_by_depth[depth]
            or proposal_id in registry.token_by_depth[depth]
            or proposal_id in registry.seq_by_depth[depth]
            or proposal_id in registry.parent_by_depth[depth]
            or proposal_id in registry.root_by_depth[depth]
            or proposal_id in registry.accept_len_by_depth[depth]
            or proposal_id in registry.action_by_depth[depth]
            or proposal_id in registry.verify_result_by_depth[depth]
        ):
            return depth
    return 0


def source_for_depth(depth: int) -> str:
    if depth == 0:
        return "one_shot_eager"
    if depth == 1:
        return "continuous_eager_depth1"
    if depth == 2:
        return "rolling_continuous_depth2"
    if depth == 3:
        return "rolling_continuous_depth3"
    return f"rolling_continuous_depth{depth}"


def node_status(
    committed: bool,
    ready: bool,
    generated: bool,
    invalidated: bool,
    cascade: bool,
    stale: bool,
) -> str:
    if committed:
        return "COMMITTED"
    if cascade:
        return "CASCADE_DISCARDED"
    if invalidated:
        return "INVALIDATED"
    if stale:
        return "STALE_OR_FRONTIER_MISMATCH"
    if ready:
        return "READY_SHADOW"
    if generated:
        return "GENERATED"
    return "OBSERVED"


def status_reason(invalidated: bool, cascade: bool, stale: bool) -> str | None:
    if cascade:
        return "cascade_discarded"
    if invalidated:
        return "invalidated"
    if stale:
        return "stale_or_frontier_mismatch"
    return None


def committed_token_count(registry: RollingChainRegistry, depth: int) -> int:
    return sum(
        max(0, int(registry.token_by_depth[depth].get(proposal_id, registry.gamma)))
        for proposal_id in registry.committed_by_depth[depth]
    )


def committed_token_counts_by_depth(registry: RollingChainRegistry, max_depth: int = MAX_LEGACY_REAL_DEPTH) -> dict[int, int]:
    return {depth: committed_token_count(registry, depth) for depth in range(0, max_depth + 1)}
