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
