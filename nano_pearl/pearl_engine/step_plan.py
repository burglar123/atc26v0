from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _int_list(values: List[int]) -> List[int]:
    return [int(value) for value in values]


def _trace_int_value(value: Any, allow_none: bool = False) -> Any:
    if value is None and allow_none:
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return int(value)


def _trace_mapping(mapping: Dict[int, Any], value_fn) -> Dict[str, Any]:
    return {
        str(seq_id): value_fn(value)
        for seq_id, value in mapping.items()
    }


@dataclass
class RequestBudget:
    normal_gamma: int
    eager_gamma: int = 0

    def to_trace_dict(self) -> dict:
        return {
            "normal_gamma": int(self.normal_gamma),
            "eager_gamma": int(self.eager_gamma),
        }


@dataclass
class StepPlan:
    plan_id: int
    iteration_id: int
    execution_mode: str

    target_home_set: List[int] = field(default_factory=list)
    target_normal_verify_seq_ids: List[int] = field(default_factory=list)
    target_eager_verify_seq_ids_dry_run: List[int] = field(default_factory=list)
    target_eager_verify_proposal_ids_dry_run: List[int] = field(default_factory=list)
    target_eager_verify_reason_by_seq_id_dry_run: Dict[int, str] = field(default_factory=dict)
    excluded_from_target_normal_verify_for_eager_dry_run: List[int] = field(default_factory=list)
    missing_normal_proposal_allowed_by_eager_dry_run: bool = False
    missing_normal_proposal_allowed_seq_ids_dry_run: List[int] = field(default_factory=list)
    raw_target_home_set_for_normal_verify: List[int] = field(default_factory=list)
    missing_buffered_proposal_seq_ids: List[int] = field(default_factory=list)
    missing_buffered_proposal_allowed_by_eager_seq_ids: List[int] = field(default_factory=list)
    missing_buffered_proposal_unexpected_seq_ids: List[int] = field(default_factory=list)
    fallback_same_batch: bool = False
    fallback_pending_receive_seq_ids: List[int] = field(default_factory=list)
    fallback_received_seq_ids: List[int] = field(default_factory=list)
    fallback_missing_after_receive_seq_ids: List[int] = field(default_factory=list)
    target_eager_set: List[int] = field(default_factory=list)
    target_eager_set_dry_run: List[int] = field(default_factory=list)
    scheduled_target_eager_set_dry_run: List[int] = field(default_factory=list)
    scheduled_target_eager_proposal_ids_dry_run: List[int] = field(default_factory=list)
    scheduled_target_eager_seq_ids_dry_run: List[int] = field(default_factory=list)
    adjusted_draft_home_set_dry_run: List[int] = field(default_factory=list)
    excluded_from_draft_home_for_eager_dry_run: List[int] = field(default_factory=list)
    original_draft_home_set: List[int] = field(default_factory=list)
    actual_draft_home_set_for_normal_draft: List[int] = field(default_factory=list)
    lane_excluded_seq_ids: List[int] = field(default_factory=list)
    lane_exclusion_decision_available_before_draft: bool = False
    lane_exclusion_deferred_until_next_step: bool = False
    lane_exclusion_defer_reason: Optional[str] = None
    pending_lane_exclusion_decision_ids_before_plan: List[int] = field(default_factory=list)
    applied_lane_exclusion_decision_ids: List[int] = field(default_factory=list)
    stale_lane_exclusion_decision_ids: List[int] = field(default_factory=list)
    expired_lane_exclusion_decision_ids: List[int] = field(default_factory=list)
    lane_exclusion_drop_reason_by_decision_id: Dict[int, str] = field(default_factory=dict)
    active_pending_lane_exclusion_decision_ids: List[int] = field(default_factory=list)
    terminal_lane_exclusion_decision_ids: List[int] = field(default_factory=list)
    touched_lane_exclusion_decision_ids: List[int] = field(default_factory=list)
    lane_exclusion_source_step_by_decision_id: Dict[int, int] = field(default_factory=dict)
    lane_exclusion_created_step_by_decision_id: Dict[int, int] = field(default_factory=dict)
    lane_exclusion_applied_step_by_decision_id: Dict[int, int] = field(default_factory=dict)
    lane_exclusion_defer_reason_by_decision_id: Dict[int, str] = field(default_factory=dict)
    lane_exclusion_decisions_synchronized_before_plan: bool = False
    lane_exclusion_decision_transfer_called: bool = False
    lane_exclusion_decision_sent_proposal_ids: List[int] = field(default_factory=list)
    lane_exclusion_decision_received_proposal_ids: List[int] = field(default_factory=list)
    lane_exclusion_decision_sent_seq_ids: List[int] = field(default_factory=list)
    lane_exclusion_decision_received_seq_ids: List[int] = field(default_factory=list)
    lane_exclusion_decision_num_decisions: int = 0
    lane_exclusion_decision_payload_len: int = 0
    lane_exclusion_decision_zero_decision: bool = True
    lane_exclusion_decision_sync_plan_id: Optional[int] = None
    lane_exclusion_decision_sync_step_id: Optional[int] = None
    ready_eager_proposal_created_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_created_seq_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_synced_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_registry_ids_before_plan: List[int] = field(default_factory=list)
    ready_eager_proposal_seen_by_scheduler_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_in_target_home_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_in_draft_home_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_applied_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_stale_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_expired_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_invalidated_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_state_by_id: Dict[int, str] = field(default_factory=dict)
    ready_eager_proposal_skip_reason_by_id: Dict[int, str] = field(default_factory=dict)
    ready_eager_proposal_stale_reason_by_id: Dict[int, str] = field(default_factory=dict)
    ready_eager_proposal_age_by_id: Dict[int, int] = field(default_factory=dict)
    ready_eager_proposal_seq_id_by_id: Dict[int, int] = field(default_factory=dict)
    ready_eager_proposal_base_len_by_id: Dict[int, int] = field(default_factory=dict)
    ready_eager_proposal_current_len_by_id: Dict[int, int] = field(default_factory=dict)
    ready_eager_proposal_current_pre_verify_by_id: Dict[int, bool] = field(default_factory=dict)
    ready_eager_proposal_current_status_by_id: Dict[int, str] = field(default_factory=dict)
    ready_eager_proposal_apply_step_by_id: Dict[int, int] = field(default_factory=dict)
    ready_eager_proposal_takeover_routed_step_by_id: Dict[int, int] = field(default_factory=dict)
    ready_eager_proposal_takeover_routed_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_takeover_routed_seq_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_pending_takeover_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_pending_takeover_proposal_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_pending_takeover_seq_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_takeover_waiting_for_target_home_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_already_takeover_routed_ids: List[int] = field(default_factory=list)
    repeated_takeover_proposal_ids: List[int] = field(default_factory=list)
    ready_eager_proposals_synchronized_before_plan: bool = False
    ready_eager_proposal_transfer_called: bool = False
    ready_eager_proposal_sent_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_received_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_sent_seq_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_received_seq_ids: List[int] = field(default_factory=list)
    ready_eager_proposal_num_proposals: int = 0
    ready_eager_proposal_payload_len: int = 0
    ready_eager_proposal_zero_proposal: bool = True
    ready_eager_proposal_sync_plan_id: Optional[int] = None
    ready_eager_proposal_sync_step_id: Optional[int] = None
    lane_exclusion_applied_proposal_ids: List[int] = field(default_factory=list)
    lane_exclusion_applied_seq_ids: List[int] = field(default_factory=list)
    lane_exclusion_apply_reason_by_proposal_id: Dict[int, str] = field(default_factory=dict)
    normal_proposal_expected_seq_ids_after_lane_exclusion: List[int] = field(default_factory=list)
    adjusted_normal_proposal_expected_seq_ids: List[int] = field(default_factory=list)
    draft_home_set: List[int] = field(default_factory=list)
    draft_eager_set: List[int] = field(default_factory=list)

    budgets: Dict[int, RequestBudget] = field(default_factory=dict)

    target_batch_id: Optional[str | int] = None
    draft_batch_id: Optional[str | int] = None

    decode_ready_mode: bool = False
    is_prefill: bool = False
    plan_phase: Optional[str] = None
    dual_batch_enabled: bool = False
    home_batch_ids: Dict[int, int] = field(default_factory=dict)
    pending_proposal_seq_ids: List[int] = field(default_factory=list)
    buffered_proposal_seq_ids: List[int] = field(default_factory=list)
    normal_gamma: Optional[int] = None
    eager_gamma: int = 0
    eager_new_selected_set: List[int] = field(default_factory=list)
    eager_continuing_set: List[int] = field(default_factory=list)
    eager_active_seq_ids: List[int] = field(default_factory=list)
    eager_ready_seq_ids: List[int] = field(default_factory=list)
    eager_proposal_ids_by_seq_id: Dict[int, Any] = field(default_factory=dict)
    eager_parent_proposal_ids_by_seq_id: Dict[int, Any] = field(default_factory=dict)
    eager_base_len_by_seq_id: Dict[int, int] = field(default_factory=dict)
    eager_base_pre_verify_by_seq_id: Dict[int, bool] = field(default_factory=dict)
    eager_parent_kind_by_seq_id: Dict[int, str] = field(default_factory=dict)
    step_id: Optional[int] = None
    dual_batch_state: Optional[Any] = None
    fallback_reason: Optional[str] = None
    steady_step: bool = False
    priming_step: bool = False

    fallback_has_target_batch: bool = False
    fallback_has_draft_batch: bool = False
    fallback_active_batch_count: int = 0
    fallback_active_seq_count: int = 0
    fallback_pending_proposal_count: int = 0
    fallback_target_seq_count: int = 0
    fallback_draft_seq_count: int = 0
    fallback_buffer_hit_count: int = 0
    fallback_buffer_miss_count: int = 0

    active_seq_count: int = 0
    active_batch_count: int = 0
    target_fraction_of_active: float = 0.0
    draft_fraction_of_active: float = 0.0
    split_imbalance: float = 0.0
    target_to_draft_size_ratio: float = 0.0

    proposal_buffer_size_before: Optional[int] = None
    proposal_buffer_size_after: Optional[int] = None
    proposal_buffer_requested_seq_ids: List[int] = field(default_factory=list)
    proposal_buffer_hit_seq_ids: List[int] = field(default_factory=list)
    proposal_buffer_miss_seq_ids: List[int] = field(default_factory=list)
    proposal_buffer_consumed_seq_ids: List[int] = field(default_factory=list)
    proposal_buffer_dropped_seq_ids: List[int] = field(default_factory=list)
    proposal_buffer_invalid_seq_ids: List[int] = field(default_factory=list)
    proposal_buffer_hit_count: int = 0
    proposal_buffer_miss_count: int = 0
    proposal_buffer_consumed_count: int = 0
    proposal_buffer_dropped_count: int = 0
    proposal_buffer_invalid_count: int = 0
    cached_admission_newly_admitted_seq_ids: List[int] = field(default_factory=list)
    cached_admission_draft_priming_seq_ids: List[int] = field(default_factory=list)
    cached_admission_primed_seq_ids: List[int] = field(default_factory=list)
    cached_admission_unprimed_target_filtered_seq_ids: List[int] = field(default_factory=list)
    cached_admission_missing_proposal_after_filter_seq_ids: List[int] = field(default_factory=list)
    cached_admission_filtered_draft_seq_ids: List[int] = field(default_factory=list)
    dual_proposal_sent_seq_ids: List[int] = field(default_factory=list)
    dual_proposal_expected_receive_seq_ids: List[int] = field(default_factory=list)
    dual_proposal_received_seq_ids: List[int] = field(default_factory=list)
    raw_target_normal_verify_seq_ids_before_buffer_filter: List[int] = field(default_factory=list)
    cached_admission_target_filtered_missing_proposal_seq_ids: List[int] = field(default_factory=list)
    cached_admission_target_buffer_hit_seq_ids: List[int] = field(default_factory=list)
    cached_admission_target_buffer_miss_seq_ids: List[int] = field(default_factory=list)
    target_normal_verify_seq_ids_after_buffer_filter: List[int] = field(default_factory=list)
    local_actual_draft_home_set_for_normal_draft: List[int] = field(default_factory=list)
    normal_draft_transfer_synced_expected_seq_ids: List[int] = field(default_factory=list)
    normal_draft_transfer_sender_seq_ids: List[int] = field(default_factory=list)

    enable_eager_plan_dry_run: bool = False
    eager_policy: str = "none"
    eager_candidate_seq_ids: List[int] = field(default_factory=list)
    eager_candidate_reject_reason_by_seq_id: Dict[int, str] = field(default_factory=dict)
    eager_selected_seq_ids: List[int] = field(default_factory=list)
    eager_budget_by_seq_id: Dict[int, int] = field(default_factory=dict)
    eager_total_budget: int = 0
    eager_post_verify_only: bool = True
    eager_gamma_equals_global_gamma: bool = False
    eager_pre_verify_candidate_count: int = 0
    eager_post_verify_candidate_count: int = 0
    eager_skipped_pre_verify_seq_ids: List[int] = field(default_factory=list)
    eager_skipped_non_tight_seq_ids: List[int] = field(default_factory=list)
    eager_skipped_not_in_target_home_set_seq_ids: List[int] = field(default_factory=list)
    enable_eager_draft_dry_run: bool = False
    eager_draft_dry_run_enabled: bool = False
    eager_draft_seq_ids: List[int] = field(default_factory=list)
    eager_draft_proposal_ids: List[int] = field(default_factory=list)
    eager_draft_base_len_by_seq_id: Dict[int, int] = field(default_factory=dict)
    eager_draft_base_pre_verify_by_seq_id: Dict[int, bool] = field(default_factory=dict)
    eager_draft_to_verify_len_by_seq_id: Dict[int, int] = field(default_factory=dict)
    eager_draft_proposal_len_by_seq_id: Dict[int, int] = field(default_factory=dict)
    eager_draft_rollback_seq_ids: List[int] = field(default_factory=list)
    eager_draft_rollback_ok_by_seq_id: Dict[int, bool] = field(default_factory=dict)
    eager_draft_discard_reason_by_seq_id: Dict[int, str] = field(default_factory=dict)
    eager_dry_run_tokens_generated: int = 0
    enable_eager_promotion_dry_run: bool = False
    eager_promotion_dry_run_enabled: bool = False
    enable_eager_transfer_dry_run: bool = False
    eager_transfer_dry_run_enabled: bool = False
    enable_eager_schedule_dry_run: bool = False
    eager_schedule_dry_run_enabled: bool = False
    enable_eager_verify_dry_run: bool = False
    eager_verify_dry_run_enabled: bool = False
    enable_eager_apply_dry_run: bool = False
    eager_apply_dry_run_enabled: bool = False
    enable_eager_result_transfer_dry_run: bool = False
    eager_result_transfer_dry_run_enabled: bool = False
    enable_eager_sync_apply_dry_run: bool = False
    eager_sync_apply_dry_run_enabled: bool = False
    enable_eager_commit_readiness_dry_run: bool = False
    eager_commit_readiness_dry_run_enabled: bool = False
    enable_eager_commit_ready_only: bool = False
    eager_commit_enabled: bool = False
    enable_eager_lane_exclusion_dry_run: bool = False
    eager_lane_exclusion_dry_run_enabled: bool = False

    def all_seq_ids(self) -> List[int]:
        return list(self.target_home_set) + list(self.target_eager_set) + list(self.draft_home_set) + list(self.draft_eager_set)

    def role_seq_ids(self, runner_role: str) -> List[int]:
        if "draft" in runner_role:
            return list(self.draft_home_set) + list(self.draft_eager_set)
        return list(self.target_home_set) + list(self.target_eager_set)

    def target_normal_verify_ids(self) -> List[int]:
        if (
            self.target_normal_verify_seq_ids
            or self.raw_target_home_set_for_normal_verify
            or self.target_eager_verify_seq_ids_dry_run
            or self.excluded_from_target_normal_verify_for_eager_dry_run
        ):
            return list(self.target_normal_verify_seq_ids)
        return list(self.target_home_set)

    def is_eager_empty(self) -> bool:
        return not self.target_eager_set and not self.draft_eager_set

    def validate_phase1b(self, gamma: int, runner_role: str):
        assert self.is_eager_empty(), f"Phase 1B requires empty eager sets, got target_eager_set={self.target_eager_set}, draft_eager_set={self.draft_eager_set}"
        if "draft" in runner_role:
            assert not self.target_home_set, f"Draft role cannot own target_home_set: {self.target_home_set}"
        else:
            assert not self.draft_home_set, f"Verify/target role cannot own draft_home_set: {self.draft_home_set}"
        for seq_id, budget in self.budgets.items():
            assert int(budget.eager_gamma) == 0, f"Phase 1B eager_gamma must be 0 for seq_id={seq_id}, got {budget.eager_gamma}"
            assert int(budget.normal_gamma) == int(gamma), f"Phase 1B normal_gamma mismatch for seq_id={seq_id}: expected {gamma}, got {budget.normal_gamma}"

    def validate_phase1c(self):
        assert self.execution_mode == "dual_batch_pearl", (
            f"Phase 1C StepPlan requires execution_mode='dual_batch_pearl', got {self.execution_mode!r}"
        )
        assert self.dual_batch_enabled, "Phase 1C StepPlan must set dual_batch_enabled=True"
        assert self.plan_phase in {"priming", "steady", "fallback"}, (
            f"Invalid Phase 1C plan_phase={self.plan_phase!r}"
        )
        assert self.is_eager_empty(), (
            f"Phase 1C keeps eager sets empty, got target_eager_set={self.target_eager_set}, "
            f"draft_eager_set={self.draft_eager_set}"
        )
        if self.plan_phase == "steady":
            assert self.target_batch_id != self.draft_batch_id, (
                f"Steady dual-batch plan must use different batches, got target_batch_id={self.target_batch_id}, "
                f"draft_batch_id={self.draft_batch_id}"
            )
            assert set(self.target_home_set).isdisjoint(self.draft_home_set), (
                f"Steady dual-batch home sets must be disjoint, got target_home_set={self.target_home_set}, "
                f"draft_home_set={self.draft_home_set}"
            )
        for seq_id, budget in self.budgets.items():
            assert int(budget.eager_gamma) == 0, (
                f"Phase 1C eager_gamma must be 0 for seq_id={seq_id}, got {budget.eager_gamma}"
            )
            if self.normal_gamma is not None:
                assert int(budget.normal_gamma) == int(self.normal_gamma), (
                    f"Phase 1C normal_gamma mismatch for seq_id={seq_id}: "
                    f"expected {self.normal_gamma}, got {budget.normal_gamma}"
                )

    def _inferred_gamma_for_eager_validation(self) -> int | None:
        if self.normal_gamma is not None:
            return int(self.normal_gamma)
        budget_gammas = {
            int(budget.normal_gamma)
            for budget in self.budgets.values()
            if budget.normal_gamma is not None
        }
        if len(budget_gammas) == 1:
            return next(iter(budget_gammas))
        if self.eager_gamma:
            return int(self.eager_gamma)
        return None

    def validate_phase1h_eager_scaffold(
        self,
        enable_eager_execution: bool,
        enable_eager_plan_dry_run: bool = False,
        enable_eager_draft_dry_run: bool = False,
        enable_eager_promotion_dry_run: bool = False,
        enable_eager_transfer_dry_run: bool = False,
        enable_eager_schedule_dry_run: bool = False,
        enable_eager_verify_dry_run: bool = False,
        enable_eager_apply_dry_run: bool = False,
        enable_eager_result_transfer_dry_run: bool = False,
        enable_eager_sync_apply_dry_run: bool = False,
        enable_eager_commit_readiness_dry_run: bool = False,
        enable_eager_commit_ready_only: bool = False,
        enable_eager_lane_exclusion_dry_run: bool = False,
        global_gamma: int | None = None,
    ):
        target_home = set(int(seq_id) for seq_id in self.target_home_set)
        target_normal_verify = set(int(seq_id) for seq_id in self.target_normal_verify_ids())
        target_eager_verify = set(int(seq_id) for seq_id in self.target_eager_verify_seq_ids_dry_run)
        draft_home = set(int(seq_id) for seq_id in self.draft_home_set)
        target_eager = set(int(seq_id) for seq_id in self.target_eager_set)
        target_eager_dry_run = set(int(seq_id) for seq_id in self.target_eager_set_dry_run)
        scheduled_target_eager_dry_run = set(
            int(seq_id) for seq_id in self.scheduled_target_eager_set_dry_run
        )
        scheduled_target_eager_seq_ids = set(
            int(seq_id) for seq_id in self.scheduled_target_eager_seq_ids_dry_run
        )
        adjusted_draft_home_dry_run = set(
            int(seq_id) for seq_id in self.adjusted_draft_home_set_dry_run
        )
        excluded_from_draft_home = set(
            int(seq_id) for seq_id in self.excluded_from_draft_home_for_eager_dry_run
        )
        original_draft_home = set(int(seq_id) for seq_id in self.original_draft_home_set)
        actual_draft_home = set(int(seq_id) for seq_id in self.actual_draft_home_set_for_normal_draft)
        lane_excluded = set(int(seq_id) for seq_id in self.lane_excluded_seq_ids)
        draft_eager = set(int(seq_id) for seq_id in self.draft_eager_set)
        assert target_normal_verify <= target_home, (
            "target_normal_verify_seq_ids must be a subset of target_home_set: "
            f"extra={sorted(target_normal_verify - target_home)}"
        )
        assert not (target_normal_verify & target_eager_verify), (
            "target_normal_verify_seq_ids cannot overlap target_eager_verify_seq_ids_dry_run: "
            f"{sorted(target_normal_verify & target_eager_verify)}"
        )
        assert target_eager_verify <= target_home, (
            "target_eager_verify_seq_ids_dry_run must be routed only for current target_home_set: "
            f"extra={sorted(target_eager_verify - target_home)}"
        )
        assert not (draft_eager & target_eager_verify), (
            "draft_eager_set is a current target-home candidate set and must not overlap "
            f"one-shot target_eager_verify_seq_ids_dry_run: {sorted(draft_eager & target_eager_verify)}"
        )

        assert not (target_eager & target_home), (
            f"target_eager_set cannot overlap target_home_set: "
            f"{sorted(target_eager & target_home)}"
        )
        assert not (target_eager & draft_home), (
            f"target_eager_set cannot overlap draft_home_set: "
            f"{sorted(target_eager & draft_home)}"
        )
        assert not (draft_eager & draft_home), (
            f"draft_eager_set cannot overlap draft_home_set: "
            f"{sorted(draft_eager & draft_home)}"
        )
        assert not (target_eager_dry_run & target_home), (
            f"target_eager_set_dry_run cannot overlap target_home_set: "
            f"{sorted(target_eager_dry_run & target_home)}"
        )
        assert not (scheduled_target_eager_dry_run & target_home), (
            f"scheduled_target_eager_set_dry_run cannot overlap target_home_set: "
            f"{sorted(scheduled_target_eager_dry_run & target_home)}"
        )
        if self.adjusted_draft_home_set_dry_run:
            assert not (target_eager_dry_run & adjusted_draft_home_dry_run), (
                f"target_eager_set_dry_run cannot overlap adjusted_draft_home_set_dry_run: "
                f"{sorted(target_eager_dry_run & adjusted_draft_home_dry_run)}"
            )
            assert not (scheduled_target_eager_dry_run & adjusted_draft_home_dry_run), (
                "scheduled_target_eager_set_dry_run cannot overlap "
                f"adjusted_draft_home_set_dry_run: "
                f"{sorted(scheduled_target_eager_dry_run & adjusted_draft_home_dry_run)}"
            )
            adjustment_parent_home = original_draft_home if original_draft_home else draft_home
            assert adjusted_draft_home_dry_run == adjustment_parent_home - excluded_from_draft_home, (
                "adjusted_draft_home_set_dry_run must equal the original draft home set minus "
                f"excluded_from_draft_home_for_eager_dry_run: adjusted={sorted(adjusted_draft_home_dry_run)}, "
                f"expected={sorted(adjustment_parent_home - excluded_from_draft_home)}"
            )
        excluded_parent_home = original_draft_home if original_draft_home else draft_home
        assert excluded_from_draft_home <= excluded_parent_home, (
            "excluded_from_draft_home_for_eager_dry_run must be a subset of the original draft home set: "
            f"extra={sorted(excluded_from_draft_home - excluded_parent_home)}"
        )
        if enable_eager_lane_exclusion_dry_run and original_draft_home:
            assert actual_draft_home == draft_home, (
                "actual_draft_home_set_for_normal_draft must match current draft_home_set "
                f"after lane exclusion: actual={sorted(actual_draft_home)}, draft={sorted(draft_home)}"
            )
            assert lane_excluded <= original_draft_home, (
                f"lane_excluded_seq_ids must be a subset of original_draft_home_set: "
                f"extra={sorted(lane_excluded - original_draft_home)}"
            )
            assert draft_home == original_draft_home - lane_excluded, (
                "draft_home_set must equal original_draft_home_set minus lane_excluded_seq_ids "
                f"when lane exclusion is enabled: draft={sorted(draft_home)}, "
                f"expected={sorted(original_draft_home - lane_excluded)}"
            )

        eager_new_selected = set(int(seq_id) for seq_id in self.eager_new_selected_set)
        draft_eager_set_new = set(
            int(seq_id) for seq_id in getattr(self, "draft_eager_set_new", [])
        )
        represented_new = eager_new_selected | draft_eager_set_new
        assert represented_new <= target_home, (
            f"new eager selections must be a subset of target_home_set: "
            f"extra={sorted(represented_new - target_home)}"
        )

        eager_continuing = set(int(seq_id) for seq_id in self.eager_continuing_set)
        continuing_eager_set = set(
            int(seq_id) for seq_id in getattr(self, "continuing_eager_set", [])
        )
        represented_continuing = eager_continuing | continuing_eager_set
        assert represented_continuing <= target_eager, (
            f"continuing eager set must be a subset of target_eager_set: "
            f"extra={sorted(represented_continuing - target_eager)}"
        )

        gamma = int(global_gamma) if global_gamma is not None else self._inferred_gamma_for_eager_validation()
        if self.eager_gamma and gamma is not None:
            assert int(self.eager_gamma) in {0, int(gamma)}, (
                f"plan eager_gamma must be 0 or gamma={gamma}, got {self.eager_gamma}"
            )
        for seq_id, budget in self.budgets.items():
            eager_gamma = int(budget.eager_gamma)
            if gamma is None:
                assert eager_gamma == 0, (
                    f"cannot validate nonzero eager budget without gamma for seq_id={seq_id}: "
                    f"eager_gamma={eager_gamma}"
                )
            else:
                assert eager_gamma in {0, int(gamma)}, (
                    f"eager budget must be 0 or gamma={gamma} for seq_id={seq_id}, "
                    f"got {eager_gamma}"
                )

        eager_list_fields = [
            self.target_eager_set,
            self.target_eager_set_dry_run,
            self.scheduled_target_eager_set_dry_run,
            self.scheduled_target_eager_proposal_ids_dry_run,
            self.scheduled_target_eager_seq_ids_dry_run,
            self.adjusted_draft_home_set_dry_run,
            self.excluded_from_draft_home_for_eager_dry_run,
            self.draft_eager_set,
            self.eager_new_selected_set,
            self.eager_continuing_set,
            self.eager_active_seq_ids,
            self.eager_ready_seq_ids,
            self.eager_draft_seq_ids,
            self.eager_draft_proposal_ids,
            self.eager_draft_rollback_seq_ids,
            list(getattr(self, "draft_eager_set_new", [])),
            list(getattr(self, "continuing_eager_set", [])),
            self.ready_eager_proposal_created_ids,
            self.ready_eager_proposal_created_seq_ids,
            self.ready_eager_proposal_synced_ids,
            self.ready_eager_proposal_registry_ids_before_plan,
            self.ready_eager_proposal_seen_by_scheduler_ids,
            self.ready_eager_proposal_in_target_home_ids,
            self.ready_eager_proposal_in_draft_home_ids,
            self.ready_eager_proposal_applied_ids,
            self.ready_eager_proposal_stale_ids,
            self.ready_eager_proposal_expired_ids,
            self.ready_eager_proposal_invalidated_ids,
            self.ready_eager_proposal_takeover_routed_ids,
            self.ready_eager_proposal_takeover_routed_seq_ids,
            self.ready_eager_proposal_pending_takeover_ids,
            self.ready_eager_proposal_pending_takeover_proposal_ids,
            self.ready_eager_proposal_pending_takeover_seq_ids,
            self.ready_eager_proposal_takeover_waiting_for_target_home_ids,
            self.ready_eager_proposal_already_takeover_routed_ids,
            self.repeated_takeover_proposal_ids,
            self.target_eager_verify_seq_ids_dry_run,
            self.target_eager_verify_proposal_ids_dry_run,
            self.excluded_from_target_normal_verify_for_eager_dry_run,
            self.missing_normal_proposal_allowed_seq_ids_dry_run,
            self.missing_buffered_proposal_allowed_by_eager_seq_ids,
            self.missing_buffered_proposal_unexpected_seq_ids,
            self.raw_target_normal_verify_seq_ids_before_buffer_filter,
            self.cached_admission_target_filtered_missing_proposal_seq_ids,
            self.cached_admission_target_buffer_hit_seq_ids,
            self.cached_admission_target_buffer_miss_seq_ids,
            self.target_normal_verify_seq_ids_after_buffer_filter,
            self.local_actual_draft_home_set_for_normal_draft,
            self.normal_draft_transfer_synced_expected_seq_ids,
            self.normal_draft_transfer_sender_seq_ids,
            self.lane_exclusion_applied_proposal_ids,
            self.lane_exclusion_applied_seq_ids,
        ]
        eager_mapping_fields = [
            self.eager_proposal_ids_by_seq_id,
            self.eager_parent_proposal_ids_by_seq_id,
            self.eager_base_len_by_seq_id,
            self.eager_base_pre_verify_by_seq_id,
            self.eager_parent_kind_by_seq_id,
            self.eager_draft_base_len_by_seq_id,
            self.eager_draft_base_pre_verify_by_seq_id,
            self.eager_draft_to_verify_len_by_seq_id,
            self.eager_draft_proposal_len_by_seq_id,
            self.eager_draft_rollback_ok_by_seq_id,
            self.eager_draft_discard_reason_by_seq_id,
            self.ready_eager_proposal_state_by_id,
            self.ready_eager_proposal_skip_reason_by_id,
            self.ready_eager_proposal_stale_reason_by_id,
            self.ready_eager_proposal_age_by_id,
            self.ready_eager_proposal_seq_id_by_id,
            self.ready_eager_proposal_base_len_by_id,
            self.ready_eager_proposal_current_len_by_id,
            self.ready_eager_proposal_current_pre_verify_by_id,
            self.ready_eager_proposal_current_status_by_id,
            self.ready_eager_proposal_apply_step_by_id,
            self.ready_eager_proposal_takeover_routed_step_by_id,
            self.target_eager_verify_reason_by_seq_id_dry_run,
            self.lane_exclusion_apply_reason_by_proposal_id,
        ]
        has_eager_scaffold = any(eager_list_fields) or any(eager_mapping_fields)
        assert (
            bool(enable_eager_execution)
            or bool(enable_eager_plan_dry_run)
            or bool(enable_eager_draft_dry_run)
            or bool(enable_eager_promotion_dry_run)
            or bool(enable_eager_transfer_dry_run)
            or bool(enable_eager_schedule_dry_run)
            or bool(enable_eager_verify_dry_run)
            or bool(enable_eager_apply_dry_run)
            or bool(enable_eager_result_transfer_dry_run)
            or bool(enable_eager_sync_apply_dry_run)
            or bool(enable_eager_commit_readiness_dry_run)
            or bool(enable_eager_commit_ready_only)
            or bool(enable_eager_lane_exclusion_dry_run)
            or not has_eager_scaffold
        ), (
            "non-empty eager scaffold fields require eager trace/execution to be enabled"
        )
        if enable_eager_commit_ready_only:
            assert enable_eager_commit_readiness_dry_run, (
                "Phase 1H-6a eager commit-ready-only requires commit-readiness dry-run"
            )
        if enable_eager_commit_readiness_dry_run:
            assert enable_eager_sync_apply_dry_run, (
                "Phase 1H-5j commit-readiness dry-run requires eager sync apply dry-run"
            )
        if enable_eager_sync_apply_dry_run:
            assert enable_eager_result_transfer_dry_run, (
                "Phase 1H-5d sync apply dry-run requires eager result transfer dry-run"
            )
        if enable_eager_result_transfer_dry_run:
            assert enable_eager_apply_dry_run, (
                "Phase 1H-5c result transfer dry-run requires eager apply dry-run"
            )
        if enable_eager_apply_dry_run:
            assert enable_eager_verify_dry_run, "Phase 1H-5b apply dry-run requires eager verify dry-run"
        if enable_eager_verify_dry_run:
            assert enable_eager_schedule_dry_run, "Phase 1H-5a verify dry-run requires eager schedule dry-run"
        if enable_eager_lane_exclusion_dry_run:
            assert enable_eager_schedule_dry_run, (
                "Phase 1H-5e lane exclusion dry-run requires eager schedule dry-run"
            )
        if enable_eager_schedule_dry_run:
            assert enable_eager_transfer_dry_run, "Phase 1H-4c schedule dry-run requires eager transfer dry-run"
            assert target_eager_dry_run == scheduled_target_eager_dry_run == scheduled_target_eager_seq_ids, (
                "Phase 1H-4c requires scheduled target eager dry-run sets to match: "
                f"target_eager_set_dry_run={self.target_eager_set_dry_run}, "
                f"scheduled_target_eager_set_dry_run={self.scheduled_target_eager_set_dry_run}, "
                f"scheduled_target_eager_seq_ids_dry_run={self.scheduled_target_eager_seq_ids_dry_run}"
            )
        else:
            assert not (
                target_eager_dry_run
                or scheduled_target_eager_dry_run
                or scheduled_target_eager_seq_ids
                or self.scheduled_target_eager_proposal_ids_dry_run
                or excluded_from_draft_home
            ), (
                "target eager scheduling dry-run fields require Phase 1H-4c schedule dry-run"
            )
        if enable_eager_transfer_dry_run:
            assert enable_eager_promotion_dry_run, "Phase 1H-4 transfer dry-run requires eager promotion dry-run"
        if enable_eager_promotion_dry_run:
            assert enable_eager_draft_dry_run, "Phase 1H-3 promotion dry-run requires eager draft dry-run"
        if enable_eager_draft_dry_run:
            assert enable_eager_plan_dry_run, "Phase 1H-2 draft dry-run requires eager plan dry-run"

        if enable_eager_plan_dry_run:
            assert not target_eager, (
                f"Phase 1H dry-run requires empty target_eager_set, got {self.target_eager_set}"
            )
            assert draft_eager <= target_home, (
                f"Phase 1H dry-run requires draft_eager_set subset of target_home_set: "
                f"extra={sorted(draft_eager - target_home)}"
            )
            assert set(int(seq_id) for seq_id in self.eager_new_selected_set) == draft_eager, (
                f"Phase 1H dry-run requires eager_new_selected_set == draft_eager_set: "
                f"eager_new_selected_set={self.eager_new_selected_set}, draft_eager_set={self.draft_eager_set}"
            )
            assert not self.eager_continuing_set, (
                f"Phase 1H dry-run requires empty eager_continuing_set, got {self.eager_continuing_set}"
            )
            selected = sorted(draft_eager)
            for seq_id in selected:
                budget = self.budgets.get(seq_id)
                assert budget is not None, f"missing budget for selected eager seq_id={seq_id}"
                assert gamma is not None, f"missing global gamma for selected eager seq_id={seq_id}"
                assert int(budget.eager_gamma) == int(budget.normal_gamma) == int(gamma), (
                    f"selected eager seq_id={seq_id} must have eager_gamma == normal_gamma == "
                    f"global gamma={gamma}, got eager_gamma={budget.eager_gamma}, "
                    f"normal_gamma={budget.normal_gamma}"
                )
                assert int(self.eager_budget_by_seq_id.get(seq_id, 0)) == int(gamma), (
                    f"selected eager seq_id={seq_id} must have eager budget gamma={gamma}, "
                    f"got {self.eager_budget_by_seq_id.get(seq_id)}"
                )
                assert self.eager_base_pre_verify_by_seq_id.get(seq_id) is False, (
                    f"selected eager seq_id={seq_id} must be post_verify"
                )
            assert set(int(seq_id) for seq_id in self.eager_selected_seq_ids) == draft_eager, (
                f"Phase 1H-1 dry-run requires eager_selected_seq_ids == draft_eager_set: "
                f"eager_selected_seq_ids={self.eager_selected_seq_ids}, draft_eager_set={self.draft_eager_set}"
            )
            assert int(self.eager_total_budget) == len(selected) * (int(gamma) if gamma is not None else 0), (
                f"eager_total_budget mismatch: expected={len(selected) * (int(gamma) if gamma is not None else 0)}, "
                f"got={self.eager_total_budget}"
            )

    def to_trace_dict(self) -> dict:
        return {
            "plan_id": int(self.plan_id),
            "iteration_id": int(self.iteration_id),
            "execution_mode": self.execution_mode,
            "target_home_set": _int_list(self.target_home_set),
            "target_normal_verify_seq_ids": _int_list(
                self.target_normal_verify_ids()
            ),
            "target_eager_verify_seq_ids_dry_run": _int_list(
                self.target_eager_verify_seq_ids_dry_run
            ),
            "target_eager_verify_proposal_ids_dry_run": _int_list(
                self.target_eager_verify_proposal_ids_dry_run
            ),
            "target_eager_verify_reason_by_seq_id_dry_run": {
                str(seq_id): str(reason)
                for seq_id, reason in self.target_eager_verify_reason_by_seq_id_dry_run.items()
            },
            "excluded_from_target_normal_verify_for_eager_dry_run": _int_list(
                self.excluded_from_target_normal_verify_for_eager_dry_run
            ),
            "missing_normal_proposal_allowed_by_eager_dry_run": bool(
                self.missing_normal_proposal_allowed_by_eager_dry_run
            ),
            "missing_normal_proposal_allowed_seq_ids_dry_run": _int_list(
                self.missing_normal_proposal_allowed_seq_ids_dry_run
            ),
            "raw_target_home_set_for_normal_verify": _int_list(
                self.raw_target_home_set_for_normal_verify or self.target_home_set
            ),
            "missing_buffered_proposal_seq_ids": _int_list(
                self.missing_buffered_proposal_seq_ids
            ),
            "missing_buffered_proposal_allowed_by_eager_seq_ids": _int_list(
                self.missing_buffered_proposal_allowed_by_eager_seq_ids
            ),
            "missing_buffered_proposal_unexpected_seq_ids": _int_list(
                self.missing_buffered_proposal_unexpected_seq_ids
            ),
            "fallback_same_batch": bool(self.fallback_same_batch),
            "fallback_pending_receive_seq_ids": _int_list(
                self.fallback_pending_receive_seq_ids
            ),
            "fallback_received_seq_ids": _int_list(self.fallback_received_seq_ids),
            "fallback_missing_after_receive_seq_ids": _int_list(
                self.fallback_missing_after_receive_seq_ids
            ),
            "target_eager_set": _int_list(self.target_eager_set),
            "target_eager_set_dry_run": _int_list(self.target_eager_set_dry_run),
            "scheduled_target_eager_set_dry_run": _int_list(self.scheduled_target_eager_set_dry_run),
            "scheduled_target_eager_proposal_ids_dry_run": _int_list(
                self.scheduled_target_eager_proposal_ids_dry_run
            ),
            "scheduled_target_eager_seq_ids_dry_run": _int_list(
                self.scheduled_target_eager_seq_ids_dry_run
            ),
            "adjusted_draft_home_set_dry_run": _int_list(self.adjusted_draft_home_set_dry_run),
            "excluded_from_draft_home_for_eager_dry_run": _int_list(
                self.excluded_from_draft_home_for_eager_dry_run
            ),
            "original_draft_home_set": _int_list(self.original_draft_home_set),
            "actual_draft_home_set_for_normal_draft": _int_list(
                self.actual_draft_home_set_for_normal_draft
            ),
            "lane_excluded_seq_ids": _int_list(self.lane_excluded_seq_ids),
            "lane_exclusion_decision_available_before_draft": bool(
                self.lane_exclusion_decision_available_before_draft
            ),
            "lane_exclusion_deferred_until_next_step": bool(
                self.lane_exclusion_deferred_until_next_step
            ),
            "lane_exclusion_defer_reason": self.lane_exclusion_defer_reason,
            "pending_lane_exclusion_decision_ids_before_plan": _int_list(
                self.pending_lane_exclusion_decision_ids_before_plan
            ),
            "applied_lane_exclusion_decision_ids": _int_list(
                self.applied_lane_exclusion_decision_ids
            ),
            "stale_lane_exclusion_decision_ids": _int_list(
                self.stale_lane_exclusion_decision_ids
            ),
            "expired_lane_exclusion_decision_ids": _int_list(
                self.expired_lane_exclusion_decision_ids
            ),
            "lane_exclusion_drop_reason_by_decision_id": {
                str(decision_id): str(reason)
                for decision_id, reason in self.lane_exclusion_drop_reason_by_decision_id.items()
            },
            "active_pending_lane_exclusion_decision_ids": _int_list(
                self.active_pending_lane_exclusion_decision_ids
            ),
            "terminal_lane_exclusion_decision_ids": _int_list(
                self.terminal_lane_exclusion_decision_ids
            ),
            "touched_lane_exclusion_decision_ids": _int_list(
                self.touched_lane_exclusion_decision_ids
            ),
            "lane_exclusion_source_step_by_decision_id": _trace_mapping(
                self.lane_exclusion_source_step_by_decision_id,
                lambda value: int(value),
            ),
            "lane_exclusion_created_step_by_decision_id": _trace_mapping(
                self.lane_exclusion_created_step_by_decision_id,
                lambda value: int(value),
            ),
            "lane_exclusion_applied_step_by_decision_id": _trace_mapping(
                self.lane_exclusion_applied_step_by_decision_id,
                lambda value: int(value),
            ),
            "lane_exclusion_defer_reason_by_decision_id": {
                str(decision_id): str(reason)
                for decision_id, reason in self.lane_exclusion_defer_reason_by_decision_id.items()
            },
            "lane_exclusion_decisions_synchronized_before_plan": bool(
                self.lane_exclusion_decisions_synchronized_before_plan
            ),
            "lane_exclusion_decision_transfer_called": bool(
                self.lane_exclusion_decision_transfer_called
            ),
            "lane_exclusion_decision_sent_proposal_ids": _int_list(
                self.lane_exclusion_decision_sent_proposal_ids
            ),
            "lane_exclusion_decision_received_proposal_ids": _int_list(
                self.lane_exclusion_decision_received_proposal_ids
            ),
            "lane_exclusion_decision_sent_seq_ids": _int_list(
                self.lane_exclusion_decision_sent_seq_ids
            ),
            "lane_exclusion_decision_received_seq_ids": _int_list(
                self.lane_exclusion_decision_received_seq_ids
            ),
            "lane_exclusion_decision_num_decisions": int(
                self.lane_exclusion_decision_num_decisions
            ),
            "lane_exclusion_decision_payload_len": int(
                self.lane_exclusion_decision_payload_len
            ),
            "lane_exclusion_decision_zero_decision": bool(
                self.lane_exclusion_decision_zero_decision
            ),
            "lane_exclusion_decision_sync_plan_id": self.lane_exclusion_decision_sync_plan_id,
            "lane_exclusion_decision_sync_step_id": self.lane_exclusion_decision_sync_step_id,
            "ready_eager_proposal_created_ids": _int_list(
                self.ready_eager_proposal_created_ids
            ),
            "ready_eager_proposal_created_seq_ids": _int_list(
                self.ready_eager_proposal_created_seq_ids
            ),
            "ready_eager_proposal_synced_ids": _int_list(
                self.ready_eager_proposal_synced_ids
            ),
            "ready_eager_proposal_registry_ids_before_plan": _int_list(
                self.ready_eager_proposal_registry_ids_before_plan
            ),
            "ready_eager_proposal_seen_by_scheduler_ids": _int_list(
                self.ready_eager_proposal_seen_by_scheduler_ids
            ),
            "ready_eager_proposal_in_target_home_ids": _int_list(
                self.ready_eager_proposal_in_target_home_ids
            ),
            "ready_eager_proposal_in_draft_home_ids": _int_list(
                self.ready_eager_proposal_in_draft_home_ids
            ),
            "ready_eager_proposal_applied_ids": _int_list(
                self.ready_eager_proposal_applied_ids
            ),
            "ready_eager_proposal_stale_ids": _int_list(
                self.ready_eager_proposal_stale_ids
            ),
            "ready_eager_proposal_expired_ids": _int_list(
                self.ready_eager_proposal_expired_ids
            ),
            "ready_eager_proposal_invalidated_ids": _int_list(
                self.ready_eager_proposal_invalidated_ids
            ),
            "ready_eager_proposal_state_by_id": {
                str(proposal_id): str(state)
                for proposal_id, state in self.ready_eager_proposal_state_by_id.items()
            },
            "ready_eager_proposal_skip_reason_by_id": {
                str(proposal_id): str(reason)
                for proposal_id, reason in self.ready_eager_proposal_skip_reason_by_id.items()
            },
            "ready_eager_proposal_stale_reason_by_id": {
                str(proposal_id): str(reason)
                for proposal_id, reason in self.ready_eager_proposal_stale_reason_by_id.items()
            },
            "ready_eager_proposal_age_by_id": _trace_mapping(
                self.ready_eager_proposal_age_by_id,
                lambda value: int(value),
            ),
            "ready_eager_proposal_seq_id_by_id": _trace_mapping(
                self.ready_eager_proposal_seq_id_by_id,
                lambda value: int(value),
            ),
            "ready_eager_proposal_base_len_by_id": _trace_mapping(
                self.ready_eager_proposal_base_len_by_id,
                lambda value: int(value),
            ),
            "ready_eager_proposal_current_len_by_id": _trace_mapping(
                self.ready_eager_proposal_current_len_by_id,
                lambda value: int(value),
            ),
            "ready_eager_proposal_current_pre_verify_by_id": _trace_mapping(
                self.ready_eager_proposal_current_pre_verify_by_id,
                lambda value: bool(value),
            ),
            "ready_eager_proposal_current_status_by_id": {
                str(proposal_id): str(status)
                for proposal_id, status in self.ready_eager_proposal_current_status_by_id.items()
            },
            "ready_eager_proposal_apply_step_by_id": _trace_mapping(
                self.ready_eager_proposal_apply_step_by_id,
                lambda value: int(value),
            ),
            "ready_eager_proposal_takeover_routed_step_by_id": _trace_mapping(
                self.ready_eager_proposal_takeover_routed_step_by_id,
                lambda value: int(value),
            ),
            "ready_eager_proposal_takeover_routed_ids": _int_list(
                self.ready_eager_proposal_takeover_routed_ids
            ),
            "ready_eager_proposal_takeover_routed_seq_ids": _int_list(
                self.ready_eager_proposal_takeover_routed_seq_ids
            ),
            "ready_eager_proposal_pending_takeover_ids": _int_list(
                self.ready_eager_proposal_pending_takeover_ids
            ),
            "ready_eager_proposal_pending_takeover_proposal_ids": _int_list(
                self.ready_eager_proposal_pending_takeover_proposal_ids
            ),
            "ready_eager_proposal_pending_takeover_seq_ids": _int_list(
                self.ready_eager_proposal_pending_takeover_seq_ids
            ),
            "ready_eager_proposal_takeover_waiting_for_target_home_ids": _int_list(
                self.ready_eager_proposal_takeover_waiting_for_target_home_ids
            ),
            "ready_eager_proposal_already_takeover_routed_ids": _int_list(
                self.ready_eager_proposal_already_takeover_routed_ids
            ),
            "repeated_takeover_proposal_ids": _int_list(self.repeated_takeover_proposal_ids),
            "ready_eager_proposals_synchronized_before_plan": bool(
                self.ready_eager_proposals_synchronized_before_plan
            ),
            "ready_eager_proposal_transfer_called": bool(
                self.ready_eager_proposal_transfer_called
            ),
            "ready_eager_proposal_sent_ids": _int_list(
                self.ready_eager_proposal_sent_ids
            ),
            "ready_eager_proposal_received_ids": _int_list(
                self.ready_eager_proposal_received_ids
            ),
            "ready_eager_proposal_sent_seq_ids": _int_list(
                self.ready_eager_proposal_sent_seq_ids
            ),
            "ready_eager_proposal_received_seq_ids": _int_list(
                self.ready_eager_proposal_received_seq_ids
            ),
            "ready_eager_proposal_num_proposals": int(
                self.ready_eager_proposal_num_proposals
            ),
            "ready_eager_proposal_payload_len": int(
                self.ready_eager_proposal_payload_len
            ),
            "ready_eager_proposal_zero_proposal": bool(
                self.ready_eager_proposal_zero_proposal
            ),
            "ready_eager_proposal_sync_plan_id": self.ready_eager_proposal_sync_plan_id,
            "ready_eager_proposal_sync_step_id": self.ready_eager_proposal_sync_step_id,
            "lane_exclusion_applied_proposal_ids": _int_list(
                self.lane_exclusion_applied_proposal_ids
            ),
            "lane_exclusion_applied_seq_ids": _int_list(
                self.lane_exclusion_applied_seq_ids
            ),
            "lane_exclusion_apply_reason_by_proposal_id": {
                str(proposal_id): str(reason)
                for proposal_id, reason in self.lane_exclusion_apply_reason_by_proposal_id.items()
            },
            "normal_proposal_expected_seq_ids_after_lane_exclusion": _int_list(
                self.normal_proposal_expected_seq_ids_after_lane_exclusion
            ),
            "adjusted_normal_proposal_expected_seq_ids": _int_list(
                self.adjusted_normal_proposal_expected_seq_ids
            ),
            "draft_home_set": _int_list(self.draft_home_set),
            "draft_eager_set": _int_list(self.draft_eager_set),
            "budgets": {
                str(seq_id): budget.to_trace_dict()
                for seq_id, budget in self.budgets.items()
            },
            "target_batch_id": self.target_batch_id,
            "draft_batch_id": self.draft_batch_id,
            "decode_ready_mode": bool(self.decode_ready_mode),
            "is_prefill": bool(self.is_prefill),
            "plan_phase": self.plan_phase,
            "dual_batch_enabled": bool(self.dual_batch_enabled),
            "home_batch_ids": {
                str(seq_id): int(batch_id)
                for seq_id, batch_id in self.home_batch_ids.items()
            },
            "pending_proposal_seq_ids": [int(seq_id) for seq_id in self.pending_proposal_seq_ids],
            "buffered_proposal_seq_ids": [int(seq_id) for seq_id in self.buffered_proposal_seq_ids],
            "normal_gamma": None if self.normal_gamma is None else int(self.normal_gamma),
            "eager_gamma": int(self.eager_gamma),
            "eager_new_selected_set": _int_list(self.eager_new_selected_set),
            "eager_continuing_set": _int_list(self.eager_continuing_set),
            "eager_active_seq_ids": _int_list(self.eager_active_seq_ids),
            "eager_ready_seq_ids": _int_list(self.eager_ready_seq_ids),
            "eager_proposal_ids_by_seq_id": _trace_mapping(
                self.eager_proposal_ids_by_seq_id,
                lambda value: _trace_int_value(value, allow_none=True),
            ),
            "eager_parent_proposal_ids_by_seq_id": _trace_mapping(
                self.eager_parent_proposal_ids_by_seq_id,
                lambda value: _trace_int_value(value, allow_none=True),
            ),
            "eager_base_len_by_seq_id": _trace_mapping(
                self.eager_base_len_by_seq_id,
                lambda value: int(value),
            ),
            "eager_base_pre_verify_by_seq_id": _trace_mapping(
                self.eager_base_pre_verify_by_seq_id,
                lambda value: bool(value),
            ),
            "eager_parent_kind_by_seq_id": _trace_mapping(
                self.eager_parent_kind_by_seq_id,
                lambda value: str(value),
            ),
            "step_id": None if self.step_id is None else int(self.step_id),
            "fallback_reason": self.fallback_reason,
            "steady_step": bool(self.steady_step),
            "priming_step": bool(self.priming_step),
            "fallback_has_target_batch": bool(self.fallback_has_target_batch),
            "fallback_has_draft_batch": bool(self.fallback_has_draft_batch),
            "fallback_active_batch_count": int(self.fallback_active_batch_count),
            "fallback_active_seq_count": int(self.fallback_active_seq_count),
            "fallback_pending_proposal_count": int(self.fallback_pending_proposal_count),
            "fallback_target_seq_count": int(self.fallback_target_seq_count),
            "fallback_draft_seq_count": int(self.fallback_draft_seq_count),
            "fallback_buffer_hit_count": int(self.fallback_buffer_hit_count),
            "fallback_buffer_miss_count": int(self.fallback_buffer_miss_count),
            "active_seq_count": int(self.active_seq_count),
            "active_batch_count": int(self.active_batch_count),
            "target_fraction_of_active": float(self.target_fraction_of_active),
            "draft_fraction_of_active": float(self.draft_fraction_of_active),
            "split_imbalance": float(self.split_imbalance),
            "target_to_draft_size_ratio": float(self.target_to_draft_size_ratio),
            "proposal_buffer_size_before": self.proposal_buffer_size_before,
            "proposal_buffer_size_after": self.proposal_buffer_size_after,
            "proposal_buffer_requested_seq_ids": [int(seq_id) for seq_id in self.proposal_buffer_requested_seq_ids],
            "proposal_buffer_hit_seq_ids": [int(seq_id) for seq_id in self.proposal_buffer_hit_seq_ids],
            "proposal_buffer_miss_seq_ids": [int(seq_id) for seq_id in self.proposal_buffer_miss_seq_ids],
            "proposal_buffer_consumed_seq_ids": [int(seq_id) for seq_id in self.proposal_buffer_consumed_seq_ids],
            "proposal_buffer_dropped_seq_ids": [int(seq_id) for seq_id in self.proposal_buffer_dropped_seq_ids],
            "proposal_buffer_invalid_seq_ids": [int(seq_id) for seq_id in self.proposal_buffer_invalid_seq_ids],
            "proposal_buffer_hit_count": int(self.proposal_buffer_hit_count),
            "proposal_buffer_miss_count": int(self.proposal_buffer_miss_count),
            "proposal_buffer_consumed_count": int(self.proposal_buffer_consumed_count),
            "proposal_buffer_dropped_count": int(self.proposal_buffer_dropped_count),
            "proposal_buffer_invalid_count": int(self.proposal_buffer_invalid_count),
            "cached_admission_newly_admitted_seq_ids": _int_list(
                self.cached_admission_newly_admitted_seq_ids
            ),
            "cached_admission_draft_priming_seq_ids": _int_list(
                self.cached_admission_draft_priming_seq_ids
            ),
            "cached_admission_primed_seq_ids": _int_list(
                self.cached_admission_primed_seq_ids
            ),
            "cached_admission_unprimed_target_filtered_seq_ids": _int_list(
                self.cached_admission_unprimed_target_filtered_seq_ids
            ),
            "cached_admission_missing_proposal_after_filter_seq_ids": _int_list(
                self.cached_admission_missing_proposal_after_filter_seq_ids
            ),
            "cached_admission_filtered_draft_seq_ids": _int_list(
                self.cached_admission_filtered_draft_seq_ids
            ),
            "dual_proposal_sent_seq_ids": _int_list(self.dual_proposal_sent_seq_ids),
            "dual_proposal_expected_receive_seq_ids": _int_list(
                self.dual_proposal_expected_receive_seq_ids
            ),
            "dual_proposal_received_seq_ids": _int_list(self.dual_proposal_received_seq_ids),
            "raw_target_normal_verify_seq_ids_before_buffer_filter": _int_list(
                self.raw_target_normal_verify_seq_ids_before_buffer_filter
            ),
            "cached_admission_target_filtered_missing_proposal_seq_ids": _int_list(
                self.cached_admission_target_filtered_missing_proposal_seq_ids
            ),
            "cached_admission_target_buffer_hit_seq_ids": _int_list(
                self.cached_admission_target_buffer_hit_seq_ids
            ),
            "cached_admission_target_buffer_miss_seq_ids": _int_list(
                self.cached_admission_target_buffer_miss_seq_ids
            ),
            "target_normal_verify_seq_ids_after_buffer_filter": _int_list(
                self.target_normal_verify_seq_ids_after_buffer_filter
            ),
            "local_actual_draft_home_set_for_normal_draft": _int_list(
                self.local_actual_draft_home_set_for_normal_draft
            ),
            "normal_draft_transfer_synced_expected_seq_ids": _int_list(
                self.normal_draft_transfer_synced_expected_seq_ids
            ),
            "normal_draft_transfer_sender_seq_ids": _int_list(
                self.normal_draft_transfer_sender_seq_ids
            ),
            "enable_eager_plan_dry_run": bool(self.enable_eager_plan_dry_run),
            "eager_policy": self.eager_policy,
            "eager_candidate_seq_ids": _int_list(self.eager_candidate_seq_ids),
            "eager_candidate_reject_reason_by_seq_id": {
                str(seq_id): str(reason)
                for seq_id, reason in self.eager_candidate_reject_reason_by_seq_id.items()
            },
            "eager_selected_seq_ids": _int_list(self.eager_selected_seq_ids),
            "eager_budget_by_seq_id": _trace_mapping(
                self.eager_budget_by_seq_id,
                lambda value: int(value),
            ),
            "eager_total_budget": int(self.eager_total_budget),
            "eager_post_verify_only": bool(self.eager_post_verify_only),
            "eager_gamma_equals_global_gamma": bool(self.eager_gamma_equals_global_gamma),
            "eager_pre_verify_candidate_count": int(self.eager_pre_verify_candidate_count),
            "eager_post_verify_candidate_count": int(self.eager_post_verify_candidate_count),
            "eager_skipped_pre_verify_seq_ids": _int_list(self.eager_skipped_pre_verify_seq_ids),
            "eager_skipped_non_tight_seq_ids": _int_list(self.eager_skipped_non_tight_seq_ids),
            "eager_skipped_not_in_target_home_set_seq_ids": _int_list(
                self.eager_skipped_not_in_target_home_set_seq_ids
            ),
            "enable_eager_draft_dry_run": bool(self.enable_eager_draft_dry_run),
            "eager_draft_dry_run_enabled": bool(self.eager_draft_dry_run_enabled),
            "eager_draft_seq_ids": _int_list(self.eager_draft_seq_ids),
            "eager_draft_proposal_ids": _int_list(self.eager_draft_proposal_ids),
            "eager_draft_base_len_by_seq_id": _trace_mapping(
                self.eager_draft_base_len_by_seq_id,
                lambda value: int(value),
            ),
            "eager_draft_base_pre_verify_by_seq_id": _trace_mapping(
                self.eager_draft_base_pre_verify_by_seq_id,
                lambda value: bool(value),
            ),
            "eager_draft_to_verify_len_by_seq_id": _trace_mapping(
                self.eager_draft_to_verify_len_by_seq_id,
                lambda value: int(value),
            ),
            "eager_draft_proposal_len_by_seq_id": _trace_mapping(
                self.eager_draft_proposal_len_by_seq_id,
                lambda value: int(value),
            ),
            "eager_draft_rollback_seq_ids": _int_list(self.eager_draft_rollback_seq_ids),
            "eager_draft_rollback_ok_by_seq_id": _trace_mapping(
                self.eager_draft_rollback_ok_by_seq_id,
                lambda value: bool(value),
            ),
            "eager_draft_discard_reason_by_seq_id": {
                str(seq_id): str(reason)
                for seq_id, reason in self.eager_draft_discard_reason_by_seq_id.items()
            },
            "eager_dry_run_tokens_generated": int(self.eager_dry_run_tokens_generated),
            "enable_eager_promotion_dry_run": bool(self.enable_eager_promotion_dry_run),
            "eager_promotion_dry_run_enabled": bool(self.eager_promotion_dry_run_enabled),
            "enable_eager_transfer_dry_run": bool(self.enable_eager_transfer_dry_run),
            "eager_transfer_dry_run_enabled": bool(self.eager_transfer_dry_run_enabled),
            "enable_eager_schedule_dry_run": bool(self.enable_eager_schedule_dry_run),
            "eager_schedule_dry_run_enabled": bool(self.eager_schedule_dry_run_enabled),
            "enable_eager_verify_dry_run": bool(self.enable_eager_verify_dry_run),
            "eager_verify_dry_run_enabled": bool(self.eager_verify_dry_run_enabled),
            "enable_eager_apply_dry_run": bool(self.enable_eager_apply_dry_run),
            "eager_apply_dry_run_enabled": bool(self.eager_apply_dry_run_enabled),
            "enable_eager_result_transfer_dry_run": bool(self.enable_eager_result_transfer_dry_run),
            "eager_result_transfer_dry_run_enabled": bool(self.eager_result_transfer_dry_run_enabled),
            "enable_eager_sync_apply_dry_run": bool(self.enable_eager_sync_apply_dry_run),
            "eager_sync_apply_dry_run_enabled": bool(self.eager_sync_apply_dry_run_enabled),
            "enable_eager_commit_readiness_dry_run": bool(self.enable_eager_commit_readiness_dry_run),
            "eager_commit_readiness_dry_run_enabled": bool(self.eager_commit_readiness_dry_run_enabled),
            "enable_eager_commit_ready_only": bool(self.enable_eager_commit_ready_only),
            "eager_commit_enabled": bool(self.eager_commit_enabled),
            "enable_eager_lane_exclusion_dry_run": bool(self.enable_eager_lane_exclusion_dry_run),
            "eager_lane_exclusion_dry_run_enabled": bool(self.eager_lane_exclusion_dry_run_enabled),
        }
