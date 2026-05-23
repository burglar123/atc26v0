from dataclasses import dataclass, field
from typing import Dict, List, Optional


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

    target_batch_id: Optional[str] = None
    draft_batch_id: Optional[str] = None

    decode_ready_mode: bool = False
    is_prefill: bool = False

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
        }
