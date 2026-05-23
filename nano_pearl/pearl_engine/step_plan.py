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
        }
