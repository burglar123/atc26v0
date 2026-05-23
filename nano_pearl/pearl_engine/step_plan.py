from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    target_eager_set: List[int] = field(default_factory=list)
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

    eager_trace_enabled: bool = False
    enable_eager_execution: bool = False
    eager_execution_enabled: bool = False
    eager_policy: str = "none"
    eager_candidate_seq_ids: List[int] = field(default_factory=list)
    eager_selected_seq_ids: List[int] = field(default_factory=list)
    eager_score_by_seq_id: Dict[int, float] = field(default_factory=dict)
    eager_budget_by_seq_id: Dict[int, int] = field(default_factory=dict)
    eager_total_budget: int = 0
    eager_selection_reason_by_seq_id: Dict[int, str] = field(default_factory=dict)
    eager_slo_class_by_seq_id: Dict[int, str] = field(default_factory=dict)
    max_eager_requests_per_step: int = 0
    max_eager_tokens_per_step: int = 0
    max_eager_tokens_per_request: int = 0
    eager_tokens_generated: int = 0
    eager_tokens_verified: int = 0
    eager_tokens_accepted: int = 0
    eager_tokens_rejected: int = 0
    eager_tokens_invalidated: int = 0
    eager_tokens_promoted: int = 0
    eager_tokens_discarded: int = 0
    eager_waste_rate: Optional[float] = None
    eager_buffer_size_before: int = 0
    eager_buffer_size_after: int = 0
    eager_ready_seq_ids: List[int] = field(default_factory=list)
    eager_promoted_seq_ids: List[int] = field(default_factory=list)
    eager_discarded_seq_ids: List[int] = field(default_factory=list)
    eager_verified_seq_ids: List[int] = field(default_factory=list)
    eager_accepted_seq_ids: List[int] = field(default_factory=list)
    eager_rejected_seq_ids: List[int] = field(default_factory=list)
    eager_draft_start_ts: Optional[float] = None
    eager_draft_end_ts: Optional[float] = None
    eager_draft_time_ms: float = 0.0
    eager_verify_start_ts: Optional[float] = None
    eager_verify_end_ts: Optional[float] = None
    eager_verify_time_ms: float = 0.0

    proposal_buffer_size_before: Optional[int] = None
    proposal_buffer_size_after: Optional[int] = None
    proposal_buffer_keys_before_eager_selection: List[int] = field(default_factory=list)
    proposal_buffer_keys_after_eager_selection: List[int] = field(default_factory=list)
    proposal_buffer_keys_after_eager_draft: List[int] = field(default_factory=list)
    missing_normal_proposal_seq_ids: List[int] = field(default_factory=list)
    normal_proposal_refresh_seq_ids: List[int] = field(default_factory=list)
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

    def all_seq_ids(self) -> List[int]:
        return list(self.target_home_set) + list(self.target_eager_set) + list(self.draft_home_set) + list(self.draft_eager_set)

    def role_seq_ids(self, runner_role: str) -> List[int]:
        if "draft" in runner_role:
            return list(self.draft_home_set) + list(self.draft_eager_set)
        return list(self.target_home_set) + list(self.target_eager_set)

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
        assert not self.target_eager_set, (
            f"Phase 1C/1G-lite keeps target_eager_set empty, got target_eager_set={self.target_eager_set}"
        )
        if self.draft_eager_set:
            assert self.eager_trace_enabled or self.eager_execution_enabled, (
                f"draft_eager_set may be non-empty only in trace-only eager mode, got {self.draft_eager_set}"
            )
        if self.target_eager_set:
            assert self.eager_execution_enabled, (
                f"target_eager_set may be non-empty only when eager execution is enabled, got {self.target_eager_set}"
            )
            assert set(self.target_eager_set).isdisjoint(self.target_home_set), (
                f"target_eager_set must be disjoint from target_home_set, got target_eager_set={self.target_eager_set}, "
                f"target_home_set={self.target_home_set}"
            )
            assert set(self.target_eager_set).isdisjoint(self.draft_home_set), (
                f"target_eager_set must be disjoint from draft_home_set, got target_eager_set={self.target_eager_set}, "
                f"draft_home_set={self.draft_home_set}"
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
            if int(budget.eager_gamma) != 0:
                assert (
                    (self.eager_trace_enabled and int(seq_id) in set(self.eager_selected_seq_ids))
                    or (self.eager_execution_enabled and int(seq_id) in set(self.target_eager_set))
                ), (
                    f"Non-zero eager_gamma is trace-only and must belong to selected eager seqs, "
                    f"seq_id={seq_id}, eager_gamma={budget.eager_gamma}"
                )
            if self.normal_gamma is not None:
                assert int(budget.normal_gamma) == int(self.normal_gamma), (
                    f"Phase 1C normal_gamma mismatch for seq_id={seq_id}: "
                    f"expected {self.normal_gamma}, got {budget.normal_gamma}"
                )

    def validate_phase1h_eager_execution(self):
        assert self.execution_mode == "dual_batch_pearl", (
            f"Phase 1H eager execution requires execution_mode='dual_batch_pearl', got {self.execution_mode!r}"
        )
        assert self.dual_batch_enabled, "Phase 1H eager execution requires dual_batch_enabled=True"
        assert self.plan_phase in {"priming", "steady", "fallback"}, (
            f"Invalid Phase 1H plan_phase={self.plan_phase!r}"
        )
        assert self.enable_eager_execution and self.eager_execution_enabled, (
            "Phase 1H eager execution validator requires eager execution to be enabled"
        )
        if self.target_eager_set:
            assert self.plan_phase == "steady", (
                f"target_eager_set is allowed only in steady eager execution steps, "
                f"got plan_phase={self.plan_phase!r}, target_eager_set={self.target_eager_set}"
            )
        assert not (self.plan_phase == "priming" and self.target_eager_set), (
            f"Priming steps cannot have target_eager_set, got {self.target_eager_set}"
        )
        if self.draft_eager_set:
            assert self.eager_trace_enabled or self.eager_execution_enabled, (
                f"draft_eager_set may be non-empty only when eager trace or eager execution is enabled, "
                f"got {self.draft_eager_set}"
            )
        assert set(self.target_eager_set).isdisjoint(self.target_home_set), (
            f"target_eager_set must be disjoint from target_home_set, got target_eager_set={self.target_eager_set}, "
            f"target_home_set={self.target_home_set}"
        )
        assert set(self.target_eager_set).isdisjoint(self.draft_home_set), (
            f"target_eager_set must be disjoint from draft_home_set, got target_eager_set={self.target_eager_set}, "
            f"draft_home_set={self.draft_home_set}"
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
        allowed_eager_budget_seq_ids = set(self.eager_selected_seq_ids) | set(self.draft_eager_set) | set(self.target_eager_set)
        for seq_id, budget in self.budgets.items():
            if int(budget.eager_gamma) != 0:
                assert int(seq_id) in allowed_eager_budget_seq_ids, (
                    f"Non-zero eager_gamma must belong to selected or eager seqs, "
                    f"seq_id={seq_id}, eager_gamma={budget.eager_gamma}, "
                    f"allowed={sorted(allowed_eager_budget_seq_ids)}"
                )
            if self.normal_gamma is not None:
                assert int(budget.normal_gamma) == int(self.normal_gamma), (
                    f"Phase 1H normal_gamma mismatch for seq_id={seq_id}: "
                    f"expected {self.normal_gamma}, got {budget.normal_gamma}"
                )

    def to_trace_dict(self) -> dict:
        return {
            "plan_id": int(self.plan_id),
            "iteration_id": int(self.iteration_id),
            "execution_mode": self.execution_mode,
            "target_home_set": [int(seq_id) for seq_id in self.target_home_set],
            "target_eager_set": [int(seq_id) for seq_id in self.target_eager_set],
            "draft_home_set": [int(seq_id) for seq_id in self.draft_home_set],
            "draft_eager_set": [int(seq_id) for seq_id in self.draft_eager_set],
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
            "eager_trace_enabled": bool(self.eager_trace_enabled),
            "enable_eager_execution": bool(self.enable_eager_execution),
            "eager_execution_enabled": bool(self.eager_execution_enabled),
            "eager_policy": self.eager_policy,
            "eager_candidate_seq_ids": [int(seq_id) for seq_id in self.eager_candidate_seq_ids],
            "eager_selected_seq_ids": [int(seq_id) for seq_id in self.eager_selected_seq_ids],
            "eager_score_by_seq_id": {
                str(seq_id): float(score)
                for seq_id, score in self.eager_score_by_seq_id.items()
            },
            "eager_budget_by_seq_id": {
                str(seq_id): int(budget)
                for seq_id, budget in self.eager_budget_by_seq_id.items()
            },
            "eager_total_budget": int(self.eager_total_budget),
            "eager_selection_reason_by_seq_id": {
                str(seq_id): str(reason)
                for seq_id, reason in self.eager_selection_reason_by_seq_id.items()
            },
            "eager_slo_class_by_seq_id": {
                str(seq_id): str(slo_class)
                for seq_id, slo_class in self.eager_slo_class_by_seq_id.items()
            },
            "max_eager_requests_per_step": int(self.max_eager_requests_per_step),
            "max_eager_tokens_per_step": int(self.max_eager_tokens_per_step),
            "max_eager_tokens_per_request": int(self.max_eager_tokens_per_request),
            "eager_tokens_generated": int(self.eager_tokens_generated),
            "eager_tokens_verified": int(self.eager_tokens_verified),
            "eager_tokens_accepted": int(self.eager_tokens_accepted),
            "eager_tokens_rejected": int(self.eager_tokens_rejected),
            "eager_tokens_invalidated": int(self.eager_tokens_invalidated),
            "eager_tokens_promoted": int(self.eager_tokens_promoted),
            "eager_tokens_discarded": int(self.eager_tokens_discarded),
            "eager_waste_rate": self.eager_waste_rate,
            "eager_buffer_size_before": int(self.eager_buffer_size_before),
            "eager_buffer_size_after": int(self.eager_buffer_size_after),
            "eager_ready_seq_ids": [int(seq_id) for seq_id in self.eager_ready_seq_ids],
            "eager_promoted_seq_ids": [int(seq_id) for seq_id in self.eager_promoted_seq_ids],
            "eager_discarded_seq_ids": [int(seq_id) for seq_id in self.eager_discarded_seq_ids],
            "eager_verified_seq_ids": [int(seq_id) for seq_id in self.eager_verified_seq_ids],
            "eager_accepted_seq_ids": [int(seq_id) for seq_id in self.eager_accepted_seq_ids],
            "eager_rejected_seq_ids": [int(seq_id) for seq_id in self.eager_rejected_seq_ids],
            "eager_draft_start_ts": self.eager_draft_start_ts,
            "eager_draft_end_ts": self.eager_draft_end_ts,
            "eager_draft_time_ms": float(self.eager_draft_time_ms),
            "eager_verify_start_ts": self.eager_verify_start_ts,
            "eager_verify_end_ts": self.eager_verify_end_ts,
            "eager_verify_time_ms": float(self.eager_verify_time_ms),
            "proposal_buffer_size_before": self.proposal_buffer_size_before,
            "proposal_buffer_size_after": self.proposal_buffer_size_after,
            "proposal_buffer_keys_before_eager_selection": [
                int(seq_id) for seq_id in self.proposal_buffer_keys_before_eager_selection
            ],
            "proposal_buffer_keys_after_eager_selection": [
                int(seq_id) for seq_id in self.proposal_buffer_keys_after_eager_selection
            ],
            "proposal_buffer_keys_after_eager_draft": [
                int(seq_id) for seq_id in self.proposal_buffer_keys_after_eager_draft
            ],
            "missing_normal_proposal_seq_ids": [
                int(seq_id) for seq_id in self.missing_normal_proposal_seq_ids
            ],
            "normal_proposal_refresh_seq_ids": [
                int(seq_id) for seq_id in self.normal_proposal_refresh_seq_ids
            ],
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
        }
