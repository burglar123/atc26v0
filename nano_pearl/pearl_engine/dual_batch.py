from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.pearl_engine.step_plan import RequestBudget, StepPlan


@dataclass
class BatchState:
    batch_id: int
    seq_ids: list[int] = field(default_factory=list)


@dataclass
class DualBatchPlanState:
    target_batch_id: Optional[int]
    draft_batch_id: Optional[int]
    step_id: int
    phase: str


@dataclass
class BufferedProposal:
    seq_id: int
    request_id: str | int
    home_batch_id: int
    proposal_token_ids: list[int]
    to_be_verified_token_ids: list[int]
    proposal_len: int
    pre_verify: bool
    plan_id: int
    valid: bool = True


class ProposalBuffer:
    """Small seq-id keyed proposal buffer for delayed dual-batch verification."""

    def __init__(self):
        self._proposals: Dict[int, BufferedProposal] = {}

    def clear(self) -> None:
        self._proposals.clear()

    def store(self, proposals: Iterable[BufferedProposal]) -> None:
        for proposal in proposals:
            if proposal.valid:
                self._proposals[int(proposal.seq_id)] = proposal

    def discard(self, seq_ids: Iterable[int]) -> None:
        for seq_id in seq_ids:
            self._proposals.pop(int(seq_id), None)

    def discard_inactive(self, active_seq_ids: Iterable[int]) -> None:
        active = {int(seq_id) for seq_id in active_seq_ids}
        for seq_id in list(self._proposals):
            if seq_id not in active:
                self._proposals.pop(seq_id, None)

    def get_many(self, seq_ids: Iterable[int]) -> list[BufferedProposal]:
        proposals = []
        for seq_id in seq_ids:
            proposal = self._proposals.get(int(seq_id))
            if proposal is not None and proposal.valid:
                proposals.append(proposal)
        return proposals

    def consume(self, seq_ids: Iterable[int]) -> list[BufferedProposal]:
        proposals = self.get_many(seq_ids)
        self.discard(seq_ids)
        return proposals

    def has_all(self, seq_ids: Iterable[int]) -> bool:
        return all(int(seq_id) in self._proposals and self._proposals[int(seq_id)].valid for seq_id in seq_ids)

    def pending_seq_ids(self) -> list[int]:
        return sorted(seq_id for seq_id, proposal in self._proposals.items() if proposal.valid)

    def pending_batch_ids(self) -> list[int]:
        return sorted({proposal.home_batch_id for proposal in self._proposals.values() if proposal.valid})


class DualBatchManager:
    """Assign sticky home batches and produce breadth-only A/B StepPlans."""

    def __init__(self, gamma: int):
        self.gamma = int(gamma)
        self.batches = {0: BatchState(0), 1: BatchState(1)}
        self.step_id = 0

    def reset(self) -> None:
        for batch in self.batches.values():
            batch.seq_ids.clear()
        self.step_id = 0

    def update_running(self, running_seqs: Iterable[Sequence]) -> None:
        running = list(running_seqs)
        active_seq_ids = {int(seq.seq_id) for seq in running}

        for batch in self.batches.values():
            batch.seq_ids = [seq_id for seq_id in batch.seq_ids if seq_id in active_seq_ids]

        unassigned = [seq for seq in running if getattr(seq, "home_batch_id", None) is None]
        if not self.batches[0].seq_ids and not self.batches[1].seq_ids and unassigned:
            for idx, seq in enumerate(sorted(unassigned, key=lambda s: s.seq_id)):
                self.assign(seq, idx % 2)
            return

        for seq in sorted(unassigned, key=lambda s: s.seq_id):
            self.assign(seq, self.smaller_batch_id())

        for seq in running:
            home_batch_id = getattr(seq, "home_batch_id", None)
            if home_batch_id in self.batches and seq.seq_id not in self.batches[home_batch_id].seq_ids:
                self.batches[home_batch_id].seq_ids.append(seq.seq_id)

    def assign(self, seq: Sequence, batch_id: int) -> None:
        batch_id = int(batch_id)
        seq.home_batch_id = batch_id
        if seq.seq_id not in self.batches[batch_id].seq_ids:
            self.batches[batch_id].seq_ids.append(seq.seq_id)

    def smaller_batch_id(self) -> int:
        len0 = len(self.batches[0].seq_ids)
        len1 = len(self.batches[1].seq_ids)
        if len0 <= len1:
            return 0
        return 1

    def batch_seq_ids(self, batch_id: Optional[int]) -> list[int]:
        if batch_id is None:
            return []
        return list(self.batches[int(batch_id)].seq_ids)

    def active_batch_ids(self) -> list[int]:
        return [batch_id for batch_id, batch in self.batches.items() if batch.seq_ids]

    def home_batch_ids(self) -> dict[int, int]:
        mapping = {}
        for batch_id, batch in self.batches.items():
            for seq_id in batch.seq_ids:
                mapping[int(seq_id)] = int(batch_id)
        return mapping

    def build_step_plan(
        self,
        plan_id: int,
        iteration_id: int,
        execution_mode: str,
        decode_ready_mode: bool,
        pending_proposal_seq_ids: list[int],
        pending_batch_ids: list[int],
    ) -> StepPlan:
        active = self.active_batch_ids()
        target_batch_id: Optional[int] = None
        draft_batch_id: Optional[int] = None
        phase = "fallback"

        if len(active) == 2:
            active_pending = [batch_id for batch_id in pending_batch_ids if batch_id in active]
            if active_pending:
                target_batch_id = active_pending[0]
                draft_batch_id = 1 - target_batch_id
                phase = "steady"
            else:
                draft_batch_id = 0
                phase = "priming"
        elif len(active) == 1:
            only_batch_id = active[0]
            if any(batch_id == only_batch_id for batch_id in pending_batch_ids):
                target_batch_id = only_batch_id
                draft_batch_id = None
            else:
                target_batch_id = only_batch_id
                draft_batch_id = only_batch_id
            phase = "fallback"

        target_home_set = self.batch_seq_ids(target_batch_id)
        draft_home_set = self.batch_seq_ids(draft_batch_id)
        involved_seq_ids = sorted(set(target_home_set) | set(draft_home_set))
        budgets = {
            seq_id: RequestBudget(normal_gamma=self.gamma, eager_gamma=0)
            for seq_id in involved_seq_ids
        }

        plan = StepPlan(
            plan_id=plan_id,
            iteration_id=iteration_id,
            execution_mode=execution_mode,
            target_home_set=target_home_set,
            target_eager_set=[],
            draft_home_set=draft_home_set,
            draft_eager_set=[],
            budgets=budgets,
            target_batch_id=target_batch_id,
            draft_batch_id=draft_batch_id,
            decode_ready_mode=decode_ready_mode,
            is_prefill=False,
            plan_phase=phase,
            dual_batch_enabled=True,
            home_batch_ids=self.home_batch_ids(),
            pending_proposal_seq_ids=list(pending_proposal_seq_ids),
            buffered_proposal_seq_ids=list(pending_proposal_seq_ids),
            normal_gamma=self.gamma,
            eager_gamma=0,
            step_id=self.step_id,
        )
        plan_state = DualBatchPlanState(
            target_batch_id=target_batch_id,
            draft_batch_id=draft_batch_id,
            step_id=self.step_id,
            phase=phase,
        )
        plan.validate_phase1c()
        self.step_id += 1
        plan.dual_batch_state = plan_state
        return plan
