from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from nano_pearl.pearl_engine.sequence import Sequence, SequenceStatus
from nano_pearl.pearl_engine.step_plan import RequestBudget, StepPlan


LANE_NORMAL = "normal"
LANE_EAGER = "eager"
PROPOSAL_LANES = {LANE_NORMAL, LANE_EAGER}

EAGER_STATE_DRAFTED_PENDING_PARENT = "DRAFTED_PENDING_PARENT"
EAGER_STATE_DRAFTED_DRY_RUN = "DRAFTED_DRY_RUN"
EAGER_STATE_PENDING_BASE_REACHED = "PENDING_BASE_REACHED"
EAGER_STATE_READY_TO_VERIFY = "READY_TO_VERIFY"
EAGER_STATE_READY_TO_VERIFY_DRY_RUN = "READY_TO_VERIFY_DRY_RUN"
EAGER_STATE_TRANSFERRED_DRY_RUN = "TRANSFERRED_DRY_RUN"
EAGER_STATE_SCHEDULED_DRY_RUN = "SCHEDULED_DRY_RUN"
EAGER_STATE_VERIFYING = "VERIFYING"
EAGER_STATE_CONSUMED = "CONSUMED"
EAGER_STATE_DISCARDED = "DISCARDED"
EAGER_PROPOSAL_STATES = {
    EAGER_STATE_DRAFTED_PENDING_PARENT,
    EAGER_STATE_DRAFTED_DRY_RUN,
    EAGER_STATE_PENDING_BASE_REACHED,
    EAGER_STATE_READY_TO_VERIFY,
    EAGER_STATE_READY_TO_VERIFY_DRY_RUN,
    EAGER_STATE_TRANSFERRED_DRY_RUN,
    EAGER_STATE_SCHEDULED_DRY_RUN,
    EAGER_STATE_VERIFYING,
    EAGER_STATE_CONSUMED,
    EAGER_STATE_DISCARDED,
}
EAGER_TRANSFER_META_LEN = 5
EAGER_TRANSFER_HEADER_LEN = 12
EAGER_PARENT_KIND_TO_INT = {LANE_NORMAL: 0, LANE_EAGER: 1}
EAGER_PARENT_KIND_FROM_INT = {value: key for key, value in EAGER_PARENT_KIND_TO_INT.items()}
LANE_EXCLUSION_STATE_PENDING = "PENDING"
LANE_EXCLUSION_STATE_APPLIED = "APPLIED"
LANE_EXCLUSION_STATE_STALE_DROPPED = "STALE_DROPPED"
LANE_EXCLUSION_STATE_EXPIRED = "EXPIRED"
LANE_EXCLUSION_STATES = {
    LANE_EXCLUSION_STATE_PENDING,
    LANE_EXCLUSION_STATE_APPLIED,
    LANE_EXCLUSION_STATE_STALE_DROPPED,
    LANE_EXCLUSION_STATE_EXPIRED,
}
LANE_EXCLUSION_DECISION_REASON_READY_EAGER_SCHEDULED = "ready_eager_scheduled"
LANE_EXCLUSION_TRANSFER_META_LEN = 5
LANE_EXCLUSION_TRANSFER_HEADER_LEN = 13
LANE_EXCLUSION_STATE_TO_INT = {
    LANE_EXCLUSION_STATE_PENDING: 0,
    LANE_EXCLUSION_STATE_APPLIED: 1,
    LANE_EXCLUSION_STATE_STALE_DROPPED: 2,
    LANE_EXCLUSION_STATE_EXPIRED: 3,
}
LANE_EXCLUSION_STATE_FROM_INT = {
    value: key for key, value in LANE_EXCLUSION_STATE_TO_INT.items()
}
READY_EAGER_STATE_READY = "READY"
READY_EAGER_STATE_CONSUMED_APPLIED = "CONSUMED_APPLIED"
READY_EAGER_STATE_STALE = "STALE"
READY_EAGER_STATE_EXPIRED = "EXPIRED"
READY_EAGER_STATE_INVALIDATED = "INVALIDATED"
READY_EAGER_PROPOSAL_STATES = {
    READY_EAGER_STATE_READY,
    READY_EAGER_STATE_CONSUMED_APPLIED,
    READY_EAGER_STATE_STALE,
    READY_EAGER_STATE_EXPIRED,
    READY_EAGER_STATE_INVALIDATED,
}
READY_EAGER_STATE_TO_INT = {
    READY_EAGER_STATE_READY: 0,
    READY_EAGER_STATE_CONSUMED_APPLIED: 1,
    READY_EAGER_STATE_STALE: 2,
    READY_EAGER_STATE_EXPIRED: 3,
    READY_EAGER_STATE_INVALIDATED: 4,
}
READY_EAGER_STATE_FROM_INT = {
    value: key for key, value in READY_EAGER_STATE_TO_INT.items()
}
READY_EAGER_TRANSFER_META_LEN = 5
READY_EAGER_TRANSFER_HEADER_LEN = 19
READY_EAGER_DEFAULT_MAX_AGE = 3
LANE_EXCLUSION_APPLY_REASON_READY_PROPOSAL = "ready_eager_proposal_available_for_draft_home"


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
    fallback_reason: Optional[str] = None


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


@dataclass
class EagerProposal:
    proposal_id: int
    seq_id: int
    request_id: str | int
    lane: str = LANE_EAGER
    parent_proposal_id: int | None = None
    parent_kind: str = LANE_NORMAL
    parent_step_id: int | None = None
    source_step_id: int = 0
    source_plan_id: int = 0
    home_batch_id: int = 0
    base_len: int = 0
    base_pre_verify: bool = True
    base_num_completion_tokens: int = 0
    proposal_token_ids: list[int] = field(default_factory=list)
    to_be_verified_token_ids: list[int] = field(default_factory=list)
    proposal_len: int = 0
    state: str = EAGER_STATE_DRAFTED_PENDING_PARENT
    valid: bool = True

    def __post_init__(self):
        self.proposal_id = int(self.proposal_id)
        self.seq_id = int(self.seq_id)
        if self.lane != LANE_EAGER:
            raise ValueError(f"EagerProposal lane must be {LANE_EAGER!r}, got {self.lane!r}")
        if self.parent_kind not in PROPOSAL_LANES:
            raise ValueError(f"Invalid eager parent_kind={self.parent_kind!r}")
        if self.state not in EAGER_PROPOSAL_STATES:
            raise ValueError(f"Invalid eager proposal state={self.state!r}")
        if self.parent_proposal_id is not None:
            self.parent_proposal_id = int(self.parent_proposal_id)
        if self.parent_step_id is not None:
            self.parent_step_id = int(self.parent_step_id)
        self.source_step_id = int(self.source_step_id)
        self.source_plan_id = int(self.source_plan_id)
        self.home_batch_id = int(self.home_batch_id)
        self.base_len = int(self.base_len)
        self.base_pre_verify = bool(self.base_pre_verify)
        self.base_num_completion_tokens = int(self.base_num_completion_tokens)
        self.proposal_token_ids = [int(token_id) for token_id in self.proposal_token_ids]
        self.to_be_verified_token_ids = [int(token_id) for token_id in self.to_be_verified_token_ids]
        if self.proposal_len == 0 and self.proposal_token_ids:
            self.proposal_len = len(self.proposal_token_ids)
        self.proposal_len = int(self.proposal_len)
        if self.proposal_token_ids and self.proposal_len != len(self.proposal_token_ids):
            raise ValueError(
                f"EagerProposal proposal_len={self.proposal_len} does not match "
                f"proposal_token_ids length={len(self.proposal_token_ids)}"
            )
        self.valid = bool(self.valid)


@dataclass
class ReadyEagerProposal:
    proposal_id: int
    seq_id: int
    request_id: str | int | None = None
    base_len: int = 0
    base_pre_verify: bool = True
    proposal_len: int = 0
    to_verify_len: int = 0
    to_be_verified_token_ids: list[int] = field(default_factory=list)
    proposal_token_ids: list[int] = field(default_factory=list)
    created_step_id: int = 0
    source_plan_id: int = 0
    source_step_id: int = 0
    parent_proposal_id: int | None = None
    parent_ready: bool = True
    ready: bool = True
    scheduled: bool = False
    seq_epoch: int | None = None
    tokens_digest: int | None = None
    state: str = READY_EAGER_STATE_READY
    max_ready_proposal_age: int = READY_EAGER_DEFAULT_MAX_AGE
    takeover_routed_step_id: int | None = None
    verify_dry_run_step_id: int | None = None

    def __post_init__(self):
        self.proposal_id = int(self.proposal_id)
        self.seq_id = int(self.seq_id)
        self.base_len = int(self.base_len)
        self.base_pre_verify = bool(self.base_pre_verify)
        self.proposal_len = int(self.proposal_len)
        self.to_verify_len = int(self.to_verify_len)
        self.to_be_verified_token_ids = [
            int(token_id) for token_id in self.to_be_verified_token_ids
        ]
        self.proposal_token_ids = [int(token_id) for token_id in self.proposal_token_ids]
        if self.to_verify_len == 0 and self.to_be_verified_token_ids:
            self.to_verify_len = len(self.to_be_verified_token_ids)
        self.created_step_id = int(self.created_step_id)
        self.source_plan_id = int(self.source_plan_id)
        self.source_step_id = int(self.source_step_id)
        if self.parent_proposal_id is not None:
            self.parent_proposal_id = int(self.parent_proposal_id)
        self.parent_ready = bool(self.parent_ready)
        self.ready = bool(self.ready)
        self.scheduled = bool(self.scheduled)
        if self.seq_epoch is not None:
            self.seq_epoch = int(self.seq_epoch)
        if self.tokens_digest is not None:
            self.tokens_digest = int(self.tokens_digest)
        if self.takeover_routed_step_id is not None:
            self.takeover_routed_step_id = int(self.takeover_routed_step_id)
        if self.verify_dry_run_step_id is not None:
            self.verify_dry_run_step_id = int(self.verify_dry_run_step_id)
        if self.state not in READY_EAGER_PROPOSAL_STATES:
            raise ValueError(f"invalid ready eager proposal state={self.state!r}")
        self.max_ready_proposal_age = int(self.max_ready_proposal_age)
        if self.max_ready_proposal_age < 0:
            raise ValueError(
                "max_ready_proposal_age must be non-negative, "
                f"got {self.max_ready_proposal_age}"
            )


@dataclass
class LaneExclusionApplyRecord:
    """StepPlan-local lane exclusion record; never pending across steps."""

    proposal_id: int
    seq_id: int
    apply_step_id: int
    apply_plan_id: int
    reason: str = LANE_EXCLUSION_APPLY_REASON_READY_PROPOSAL

    def __post_init__(self):
        self.proposal_id = int(self.proposal_id)
        self.seq_id = int(self.seq_id)
        self.apply_step_id = int(self.apply_step_id)
        self.apply_plan_id = int(self.apply_plan_id)
        self.reason = str(self.reason)


class ReadyEagerProposalRegistry:
    """Scheduler-owned ready eager proposal opportunities keyed by proposal id."""

    def __init__(self):
        self._proposals: Dict[int, ReadyEagerProposal] = {}
        self._outbox: Dict[int, ReadyEagerProposal] = {}

    def clear(self) -> None:
        self._proposals.clear()
        self._outbox.clear()

    def size(self) -> int:
        return len(self._proposals)

    def proposals(self) -> list[ReadyEagerProposal]:
        return sorted(
            self._proposals.values(),
            key=lambda proposal: (int(proposal.proposal_id), int(proposal.seq_id)),
        )

    def ready(self) -> list[ReadyEagerProposal]:
        return [
            proposal
            for proposal in self.proposals()
            if proposal.state == READY_EAGER_STATE_READY
        ]

    def ready_ids(self) -> list[int]:
        return [int(proposal.proposal_id) for proposal in self.ready()]

    def pending_takeover_for_target(self, target_seq_ids: Iterable[int]) -> list[ReadyEagerProposal]:
        target_seq_id_set = {int(seq_id) for seq_id in target_seq_ids}
        return sorted(
            [
                proposal
                for proposal in self._proposals.values()
                if proposal.state == READY_EAGER_STATE_CONSUMED_APPLIED
                and proposal.takeover_routed_step_id is None
                and int(proposal.seq_id) in target_seq_id_set
            ],
            key=lambda proposal: (int(proposal.seq_id), int(proposal.proposal_id)),
        )

    def get_ready_for_seq(self, seq_id: int) -> list[ReadyEagerProposal]:
        seq_id = int(seq_id)
        return [
            proposal
            for proposal in self.ready()
            if int(proposal.seq_id) == seq_id
        ]

    def upsert(self, proposal: ReadyEagerProposal, *, enqueue_for_sync: bool = False) -> bool:
        if not isinstance(proposal, ReadyEagerProposal):
            raise TypeError(
                "ReadyEagerProposalRegistry.upsert expected ReadyEagerProposal, "
                f"got {type(proposal).__name__}"
            )
        if proposal.state != READY_EAGER_STATE_READY:
            return False
        proposal_id = int(proposal.proposal_id)
        existing = self._proposals.get(proposal_id)
        if existing is not None and existing.state != READY_EAGER_STATE_READY:
            return False
        existed = existing is not None
        self._proposals[proposal_id] = proposal
        if enqueue_for_sync:
            self._outbox[proposal_id] = proposal
        return not existed

    def mark_state(self, proposal_id: int, state: str) -> ReadyEagerProposal | None:
        if state not in READY_EAGER_PROPOSAL_STATES:
            raise ValueError(f"invalid ready eager proposal state={state!r}")
        proposal = self._proposals.get(int(proposal_id))
        if proposal is not None:
            proposal.state = state
        self._outbox.pop(int(proposal_id), None)
        return proposal

    def mark_takeover_routed(self, proposal_id: int, step_id: int) -> ReadyEagerProposal | None:
        proposal = self._proposals.get(int(proposal_id))
        if proposal is not None:
            proposal.takeover_routed_step_id = int(step_id)
        return proposal

    def mark_verify_dry_run_executed(self, proposal_id: int, step_id: int) -> ReadyEagerProposal | None:
        proposal = self._proposals.get(int(proposal_id))
        if proposal is not None:
            proposal.verify_dry_run_step_id = int(step_id)
        return proposal

    def by_id(self, proposal_id: int) -> ReadyEagerProposal | None:
        return self._proposals.get(int(proposal_id))

    def pop_outbox(self) -> list[ReadyEagerProposal]:
        proposals = sorted(
            self._outbox.values(),
            key=lambda proposal: (int(proposal.proposal_id), int(proposal.seq_id)),
        )
        self._outbox.clear()
        return proposals


class ProposalBuffer:
    """Small seq-id keyed proposal buffer for delayed dual-batch verification."""

    def __init__(self):
        self._proposals: Dict[int, BufferedProposal] = {}

    def clear(self) -> None:
        self._proposals.clear()

    def size(self) -> int:
        return len(self._proposals)

    def store(self, proposals: Iterable[BufferedProposal]) -> None:
        for proposal in proposals:
            if proposal.valid:
                self._proposals[int(proposal.seq_id)] = proposal

    def discard(self, seq_ids: Iterable[int]) -> list[int]:
        dropped = []
        for seq_id in seq_ids:
            seq_id = int(seq_id)
            if self._proposals.pop(seq_id, None) is not None:
                dropped.append(seq_id)
        return dropped

    def discard_inactive(
        self,
        active_seq_ids: Iterable[int],
        pending_batch_ids: Iterable[int] | None = None,
        protected_seq_ids: Iterable[int] | None = None,
    ) -> list[int]:
        active = {int(seq_id) for seq_id in active_seq_ids}
        protected: set[int] = set()
        if pending_batch_ids is not None:
            pending = {int(b) for b in pending_batch_ids}
            for seq_id, proposal in self._proposals.items():
                if int(proposal.home_batch_id) in pending:
                    protected.add(int(seq_id))
        if protected_seq_ids is not None:
            protected.update(int(s) for s in protected_seq_ids)
        dropped = []
        for seq_id in list(self._proposals):
            if seq_id not in active and seq_id not in protected:
                self._proposals.pop(seq_id, None)
                dropped.append(seq_id)
        return dropped

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

    def inspect(self, seq_ids: Iterable[int]) -> dict:
        requested_seq_ids = [int(seq_id) for seq_id in seq_ids]
        hit_seq_ids = []
        miss_seq_ids = []
        invalid_seq_ids = []
        for seq_id in requested_seq_ids:
            proposal = self._proposals.get(seq_id)
            if proposal is None:
                miss_seq_ids.append(seq_id)
            elif proposal.valid:
                hit_seq_ids.append(seq_id)
            else:
                invalid_seq_ids.append(seq_id)
        return {
            "requested_seq_ids": requested_seq_ids,
            "hit_seq_ids": hit_seq_ids,
            "miss_seq_ids": miss_seq_ids,
            "invalid_seq_ids": invalid_seq_ids,
        }

    def has_all(self, seq_ids: Iterable[int]) -> bool:
        return all(int(seq_id) in self._proposals and self._proposals[int(seq_id)].valid for seq_id in seq_ids)

    def pending_seq_ids(self) -> list[int]:
        return sorted(seq_id for seq_id, proposal in self._proposals.items() if proposal.valid)

    def pending_batch_ids(self) -> list[int]:
        return sorted({proposal.home_batch_id for proposal in self._proposals.values() if proposal.valid})


class EagerProposalBuffer:
    """Proposal-id keyed buffer for future eager proposals.

    This buffer is intentionally separate from the seq-id keyed normal
    ProposalBuffer so dry-run eager lifecycle state cannot leak into normal
    dual-batch verification.
    """

    def __init__(self):
        self._proposals: Dict[int, EagerProposal] = {}
        self._discard_reasons: Dict[int, str] = {}

    def clear(self) -> None:
        self._proposals.clear()
        self._discard_reasons.clear()

    def remove(self, proposal_id: int) -> EagerProposal | None:
        proposal_id = int(proposal_id)
        self._discard_reasons.pop(proposal_id, None)
        return self._proposals.pop(proposal_id, None)

    def remove_many(self, proposal_ids: Iterable[int]) -> list[int]:
        removed = []
        for proposal_id in proposal_ids:
            if self.remove(int(proposal_id)) is not None:
                removed.append(int(proposal_id))
        return sorted(removed)

    def size(self) -> int:
        return sum(1 for proposal in self._proposals.values() if proposal.valid)

    def proposals(self) -> list[EagerProposal]:
        return [proposal for proposal in self._proposals.values()]

    def store(self, proposal: EagerProposal) -> None:
        if not isinstance(proposal, EagerProposal):
            raise TypeError(f"EagerProposalBuffer.store expected EagerProposal, got {type(proposal).__name__}")
        proposal_id = int(proposal.proposal_id)
        if proposal_id in self._proposals:
            raise ValueError(f"duplicate eager proposal_id={proposal_id}")
        if proposal.lane != LANE_EAGER:
            raise ValueError(f"eager proposal must use lane={LANE_EAGER!r}")
        self._proposals[proposal_id] = proposal

    def _get(self, proposal_id: int) -> EagerProposal:
        proposal_id = int(proposal_id)
        if proposal_id not in self._proposals:
            raise KeyError(f"unknown eager proposal_id={proposal_id}")
        return self._proposals[proposal_id]

    def mark_ready(self, proposal_id: int) -> EagerProposal:
        proposal = self._get(proposal_id)
        if proposal.state in {EAGER_STATE_DISCARDED, EAGER_STATE_CONSUMED}:
            raise ValueError(f"cannot mark eager proposal_id={proposal_id} ready from state={proposal.state}")
        proposal.state = EAGER_STATE_READY_TO_VERIFY
        proposal.valid = True
        return proposal

    def mark_consumed(self, proposal_id: int) -> EagerProposal:
        proposal = self._get(proposal_id)
        if proposal.state == EAGER_STATE_DISCARDED:
            raise ValueError(f"cannot consume discarded eager proposal_id={proposal_id}")
        proposal.state = EAGER_STATE_CONSUMED
        proposal.valid = False
        return proposal

    def discard(self, proposal_id: int, reason: str) -> EagerProposal:
        proposal = self._get(proposal_id)
        proposal.state = EAGER_STATE_DISCARDED
        proposal.valid = False
        self._discard_reasons[int(proposal_id)] = str(reason)
        return proposal

    def discard_by_seq_id(self, seq_id: int, reason: str) -> list[int]:
        seq_id = int(seq_id)
        discarded = []
        for proposal_id, proposal in self._proposals.items():
            if proposal.seq_id == seq_id and proposal.valid:
                self.discard(proposal_id, reason)
                discarded.append(proposal_id)
        return sorted(discarded)

    def get_ready_by_seq_ids(self, seq_ids: Iterable[int]) -> list[EagerProposal]:
        requested = [int(seq_id) for seq_id in seq_ids]
        requested_set = set(requested)
        ready = [
            proposal
            for proposal in self._proposals.values()
            if proposal.valid
            and proposal.state in {EAGER_STATE_READY_TO_VERIFY, EAGER_STATE_READY_TO_VERIFY_DRY_RUN}
            and proposal.seq_id in requested_set
        ]
        order = {seq_id: idx for idx, seq_id in enumerate(requested)}
        return sorted(ready, key=lambda proposal: (order.get(proposal.seq_id, len(order)), proposal.proposal_id))

    def pending_seq_ids(self) -> list[int]:
        return sorted({
            proposal.seq_id
            for proposal in self._proposals.values()
            if proposal.valid
            and proposal.state in {EAGER_STATE_DRAFTED_PENDING_PARENT, EAGER_STATE_PENDING_BASE_REACHED}
        })

    def ready_seq_ids(self) -> list[int]:
        return sorted({
            proposal.seq_id
            for proposal in self._proposals.values()
            if proposal.valid
            and proposal.state in {EAGER_STATE_READY_TO_VERIFY, EAGER_STATE_READY_TO_VERIFY_DRY_RUN}
        })

    def inspect(self) -> dict:
        state_counts: dict[str, int] = {state: 0 for state in EAGER_PROPOSAL_STATES}
        proposals = []
        for proposal_id, proposal in sorted(self._proposals.items()):
            state_counts[proposal.state] = state_counts.get(proposal.state, 0) + 1
            proposals.append(eager_proposal_to_trace_dict(proposal))
        return {
            "size": self.size(),
            "pending_seq_ids": self.pending_seq_ids(),
            "ready_seq_ids": self.ready_seq_ids(),
            "state_counts": state_counts,
            "discard_reasons": {str(k): v for k, v in sorted(self._discard_reasons.items())},
            "proposals": proposals,
        }


def eager_proposal_to_trace_dict(proposal: EagerProposal) -> dict[str, Any]:
    return {
        "proposal_id": int(proposal.proposal_id),
        "seq_id": int(proposal.seq_id),
        "request_id": proposal.request_id,
        "lane": proposal.lane,
        "parent_proposal_id": proposal.parent_proposal_id,
        "parent_kind": proposal.parent_kind,
        "parent_step_id": proposal.parent_step_id,
        "source_step_id": int(proposal.source_step_id),
        "source_plan_id": int(proposal.source_plan_id),
        "home_batch_id": int(proposal.home_batch_id),
        "base_len": int(proposal.base_len),
        "base_pre_verify": bool(proposal.base_pre_verify),
        "base_num_completion_tokens": int(proposal.base_num_completion_tokens),
        "proposal_token_ids": [int(token_id) for token_id in proposal.proposal_token_ids],
        "to_be_verified_token_ids": [int(token_id) for token_id in proposal.to_be_verified_token_ids],
        "proposal_len": int(proposal.proposal_len),
        "state": proposal.state,
        "valid": bool(proposal.valid),
    }


def serialize_eager_proposal_meta(
    proposals: Iterable[EagerProposal],
    gamma: int | None = None,
    plan_id: int | None = None,
) -> tuple[dict[str, Any], list[int]]:
    proposals = list(proposals)
    payload: list[int] = []
    proposal_lens = []
    to_verify_lens = []
    for proposal in proposals:
        proposal_lens.append(int(proposal.proposal_len))
        to_verify_lens.append(len(proposal.to_be_verified_token_ids))
        payload.extend(int(token_id) for token_id in proposal.to_be_verified_token_ids)
        payload.extend(int(token_id) for token_id in proposal.proposal_token_ids)

    inferred_gamma = gamma
    if inferred_gamma is None:
        inferred_gamma = proposals[0].proposal_len if proposals else 0
    inferred_plan_id = plan_id
    if inferred_plan_id is None:
        inferred_plan_id = proposals[0].source_plan_id if proposals else -1

    meta = {
        "num_proposals": len(proposals),
        "payload_length": len(payload),
        "gamma": int(inferred_gamma),
        "plan_id": int(inferred_plan_id),
        "proposal_ids": [int(proposal.proposal_id) for proposal in proposals],
        "seq_ids": [int(proposal.seq_id) for proposal in proposals],
        "request_ids": [proposal.request_id for proposal in proposals],
        "parent_ids": [proposal.parent_proposal_id for proposal in proposals],
        "parent_kinds": [proposal.parent_kind for proposal in proposals],
        "parent_step_ids": [proposal.parent_step_id for proposal in proposals],
        "source_step_ids": [int(proposal.source_step_id) for proposal in proposals],
        "source_plan_ids": [int(proposal.source_plan_id) for proposal in proposals],
        "home_batch_ids": [int(proposal.home_batch_id) for proposal in proposals],
        "base_lens": [int(proposal.base_len) for proposal in proposals],
        "base_pre_verify": [bool(proposal.base_pre_verify) for proposal in proposals],
        "base_num_completion_tokens": [int(proposal.base_num_completion_tokens) for proposal in proposals],
        "proposal_lens": proposal_lens,
        "to_verify_lens": to_verify_lens,
        "lane_kinds": [proposal.lane for proposal in proposals],
        "states": [proposal.state for proposal in proposals],
        "valid": [bool(proposal.valid) for proposal in proposals],
    }
    return meta, payload


def _meta_list(meta: dict[str, Any], key: str, n: int, default: Any = None) -> list[Any]:
    value = meta.get(key)
    if value is None:
        return [default for _ in range(n)]
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"eager meta field {key!r} must be a list of length {n}")
    return value


def deserialize_eager_proposal_meta(meta: dict[str, Any], payload: Iterable[int]) -> list[EagerProposal]:
    payload = [int(token_id) for token_id in payload]
    n = int(meta.get("num_proposals", 0))
    payload_length = int(meta.get("payload_length", 0))
    if payload_length != len(payload):
        raise ValueError(f"eager payload length mismatch: meta={payload_length}, payload={len(payload)}")

    proposal_ids = _meta_list(meta, "proposal_ids", n)
    seq_ids = _meta_list(meta, "seq_ids", n)
    request_ids = _meta_list(meta, "request_ids", n)
    parent_ids = _meta_list(meta, "parent_ids", n)
    parent_kinds = _meta_list(meta, "parent_kinds", n, LANE_NORMAL)
    parent_step_ids = _meta_list(meta, "parent_step_ids", n)
    source_step_ids = _meta_list(meta, "source_step_ids", n, 0)
    source_plan_ids = _meta_list(meta, "source_plan_ids", n, meta.get("plan_id", 0))
    home_batch_ids = _meta_list(meta, "home_batch_ids", n, 0)
    base_lens = _meta_list(meta, "base_lens", n, 0)
    base_pre_verify = _meta_list(meta, "base_pre_verify", n, True)
    base_num_completion_tokens = _meta_list(meta, "base_num_completion_tokens", n, 0)
    proposal_lens = _meta_list(meta, "proposal_lens", n, int(meta.get("gamma", 0)))
    to_verify_lens = _meta_list(meta, "to_verify_lens", n, 0)
    lane_kinds = _meta_list(meta, "lane_kinds", n, LANE_EAGER)
    states = _meta_list(meta, "states", n, EAGER_STATE_DRAFTED_PENDING_PARENT)
    valid_flags = _meta_list(meta, "valid", n, True)

    proposals = []
    offset = 0
    for idx in range(n):
        to_verify_len = int(to_verify_lens[idx])
        proposal_len = int(proposal_lens[idx])
        to_verify = payload[offset:offset + to_verify_len]
        offset += to_verify_len
        proposal_tokens = payload[offset:offset + proposal_len]
        offset += proposal_len
        if len(to_verify) != to_verify_len or len(proposal_tokens) != proposal_len:
            raise ValueError(f"eager payload ended while decoding proposal index={idx}")
        proposals.append(
            EagerProposal(
                proposal_id=int(proposal_ids[idx]),
                seq_id=int(seq_ids[idx]),
                request_id=request_ids[idx],
                lane=lane_kinds[idx],
                parent_proposal_id=parent_ids[idx],
                parent_kind=parent_kinds[idx],
                parent_step_id=parent_step_ids[idx],
                source_step_id=int(source_step_ids[idx]),
                source_plan_id=int(source_plan_ids[idx]),
                home_batch_id=int(home_batch_ids[idx]),
                base_len=int(base_lens[idx]),
                base_pre_verify=bool(base_pre_verify[idx]),
                base_num_completion_tokens=int(base_num_completion_tokens[idx]),
                proposal_token_ids=proposal_tokens,
                to_be_verified_token_ids=to_verify,
                proposal_len=proposal_len,
                state=states[idx],
                valid=bool(valid_flags[idx]),
            )
        )
    if offset != len(payload):
        raise ValueError(f"eager payload has trailing tokens: decoded={offset}, payload={len(payload)}")
    return proposals


def _encode_eager_parent_kind(kind: str) -> int:
    if kind not in EAGER_PARENT_KIND_TO_INT:
        raise ValueError(f"unknown eager parent kind={kind!r}")
    return EAGER_PARENT_KIND_TO_INT[kind]


def _decode_eager_parent_kind(value: int) -> str:
    value = int(value)
    if value not in EAGER_PARENT_KIND_FROM_INT:
        raise ValueError(f"unknown encoded eager parent kind={value}")
    return EAGER_PARENT_KIND_FROM_INT[value]


def serialize_eager_transfer_payload(
    proposals: Iterable[EagerProposal],
    gamma: int,
    plan_id: int,
    step_id: int | None,
) -> tuple[list[int], list[int]]:
    proposals = list(proposals)
    payload: list[int] = []
    for proposal in proposals:
        to_verify_len = len(proposal.to_be_verified_token_ids)
        proposal_len = int(proposal.proposal_len)
        payload.extend(
            [
                int(proposal.proposal_id),
                int(proposal.seq_id),
                int(proposal.home_batch_id),
                int(proposal.base_len),
                int(bool(proposal.base_pre_verify)),
                int(proposal.base_num_completion_tokens),
                proposal_len,
                to_verify_len,
                _encode_eager_parent_kind(proposal.parent_kind),
                -1 if proposal.parent_proposal_id is None else int(proposal.parent_proposal_id),
                int(proposal.source_plan_id),
                int(proposal.source_step_id),
            ]
        )
        payload.extend(int(token_id) for token_id in proposal.to_be_verified_token_ids)
        payload.extend(int(token_id) for token_id in proposal.proposal_token_ids)
    meta = [
        len(proposals),
        len(payload),
        int(gamma),
        int(plan_id),
        -1 if step_id is None else int(step_id),
    ]
    return meta, payload


def deserialize_eager_transfer_payload(
    meta: Iterable[int],
    payload: Iterable[int],
) -> list[EagerProposal]:
    meta_values = [int(value) for value in meta]
    if len(meta_values) != EAGER_TRANSFER_META_LEN:
        raise ValueError(
            f"eager transfer meta must have {EAGER_TRANSFER_META_LEN} values, got {len(meta_values)}"
        )
    num_proposals, payload_len, gamma, plan_id, step_id = meta_values
    payload_values = [int(value) for value in payload]
    if int(payload_len) != len(payload_values):
        raise ValueError(
            f"eager transfer payload length mismatch: meta={payload_len}, payload={len(payload_values)}"
        )

    proposals = []
    offset = 0
    for idx in range(num_proposals):
        header = payload_values[offset:offset + EAGER_TRANSFER_HEADER_LEN]
        if len(header) != EAGER_TRANSFER_HEADER_LEN:
            raise ValueError(f"eager transfer payload ended while reading header index={idx}")
        offset += EAGER_TRANSFER_HEADER_LEN
        (
            proposal_id,
            seq_id,
            home_batch_id,
            base_len,
            base_pre_verify,
            base_num_completion_tokens,
            proposal_len,
            to_verify_len,
            parent_kind_value,
            parent_proposal_id,
            source_plan_id,
            source_step_id,
        ) = header
        to_verify = payload_values[offset:offset + to_verify_len]
        offset += to_verify_len
        proposal_tokens = payload_values[offset:offset + proposal_len]
        offset += proposal_len
        if len(to_verify) != to_verify_len or len(proposal_tokens) != proposal_len:
            raise ValueError(f"eager transfer payload ended while reading tokens index={idx}")
        proposals.append(
            EagerProposal(
                proposal_id=proposal_id,
                seq_id=seq_id,
                request_id=seq_id,
                lane=LANE_EAGER,
                parent_proposal_id=None if parent_proposal_id < 0 else parent_proposal_id,
                parent_kind=_decode_eager_parent_kind(parent_kind_value),
                parent_step_id=None,
                source_step_id=source_step_id if source_step_id >= 0 else step_id,
                source_plan_id=source_plan_id if source_plan_id >= 0 else plan_id,
                home_batch_id=home_batch_id,
                base_len=base_len,
                base_pre_verify=bool(base_pre_verify),
                base_num_completion_tokens=base_num_completion_tokens,
                proposal_token_ids=proposal_tokens,
                to_be_verified_token_ids=to_verify,
                proposal_len=proposal_len if proposal_len else gamma,
                state=EAGER_STATE_READY_TO_VERIFY,
                valid=True,
            )
        )
    if offset != len(payload_values):
        raise ValueError(
            f"eager transfer payload has trailing values: consumed={offset}, payload={len(payload_values)}"
        )
    return proposals


def _numeric_request_id(request_id: str | int | None) -> int:
    try:
        if request_id is None:
            return -1
        return int(request_id)
    except Exception:
        return -1


def _proposal_tokens_digest(token_ids: Iterable[int]) -> int:
    digest = 0
    for token_id in token_ids:
        digest = (digest * 1315423911 + int(token_id) + 0x9E3779B9) & 0x7FFFFFFFFFFFFFFF
    return int(digest)


def _encode_ready_eager_state(state: str) -> int:
    if state not in READY_EAGER_STATE_TO_INT:
        raise ValueError(f"unknown ready eager proposal state={state!r}")
    return READY_EAGER_STATE_TO_INT[state]


def _decode_ready_eager_state(value: int) -> str:
    value = int(value)
    if value not in READY_EAGER_STATE_FROM_INT:
        raise ValueError(f"unknown encoded ready eager proposal state={value}")
    return READY_EAGER_STATE_FROM_INT[value]


def ready_eager_proposal_from_eager_proposal(
    proposal: EagerProposal,
    created_step_id: int,
    max_ready_proposal_age: int = READY_EAGER_DEFAULT_MAX_AGE,
    scheduled: bool = False,
) -> ReadyEagerProposal:
    proposal_token_ids = [int(token_id) for token_id in proposal.proposal_token_ids]
    to_be_verified_token_ids = [
        int(token_id) for token_id in proposal.to_be_verified_token_ids
    ]
    return ReadyEagerProposal(
        proposal_id=int(proposal.proposal_id),
        seq_id=int(proposal.seq_id),
        request_id=proposal.request_id,
        base_len=int(proposal.base_len),
        base_pre_verify=bool(proposal.base_pre_verify),
        proposal_len=int(proposal.proposal_len),
        to_verify_len=len(proposal.to_be_verified_token_ids),
        to_be_verified_token_ids=to_be_verified_token_ids,
        proposal_token_ids=proposal_token_ids,
        created_step_id=int(created_step_id),
        source_plan_id=int(proposal.source_plan_id),
        source_step_id=int(proposal.source_step_id),
        parent_proposal_id=proposal.parent_proposal_id,
        parent_ready=True,
        ready=True,
        scheduled=bool(scheduled),
        seq_epoch=None,
        tokens_digest=_proposal_tokens_digest(proposal_token_ids),
        state=READY_EAGER_STATE_READY,
        max_ready_proposal_age=int(max_ready_proposal_age),
    )


def serialize_ready_eager_proposals(
    proposals: Iterable[ReadyEagerProposal],
    plan_id: int,
    step_id: int | None,
) -> tuple[list[int], list[int]]:
    proposals = list(proposals)
    payload: list[int] = []
    for proposal in proposals:
        proposal_token_ids = [int(token_id) for token_id in proposal.proposal_token_ids]
        to_be_verified_token_ids = [
            int(token_id) for token_id in proposal.to_be_verified_token_ids
        ]
        if len(to_be_verified_token_ids) != int(proposal.to_verify_len):
            raise ValueError(
                "ready eager proposal to_verify_len does not match "
                f"token payload for proposal_id={proposal.proposal_id}: "
                f"to_verify_len={proposal.to_verify_len}, "
                f"tokens={len(to_be_verified_token_ids)}"
            )
        payload.extend(
            [
                int(proposal.proposal_id),
                int(proposal.seq_id),
                _numeric_request_id(proposal.request_id),
                int(proposal.source_plan_id),
                int(proposal.source_step_id),
                int(proposal.created_step_id),
                int(proposal.base_len),
                int(bool(proposal.base_pre_verify)),
                int(proposal.proposal_len),
                int(proposal.to_verify_len),
                len(proposal_token_ids),
                _encode_ready_eager_state(proposal.state),
                int(proposal.max_ready_proposal_age),
                -1 if proposal.parent_proposal_id is None else int(proposal.parent_proposal_id),
                int(bool(proposal.parent_ready)),
                int(bool(proposal.ready)),
                int(bool(proposal.scheduled)),
                -1 if proposal.seq_epoch is None else int(proposal.seq_epoch),
                -1 if proposal.tokens_digest is None else int(proposal.tokens_digest),
            ]
        )
        payload.extend(to_be_verified_token_ids)
        payload.extend(proposal_token_ids)
    meta = [
        len(proposals),
        len(payload),
        int(plan_id),
        -1 if step_id is None else int(step_id),
        READY_EAGER_TRANSFER_HEADER_LEN,
    ]
    return meta, payload


def deserialize_ready_eager_proposals(
    meta: Iterable[int],
    payload: Iterable[int],
) -> list[ReadyEagerProposal]:
    meta_values = [int(value) for value in meta]
    if len(meta_values) != READY_EAGER_TRANSFER_META_LEN:
        raise ValueError(
            "ready eager proposal transfer meta must have "
            f"{READY_EAGER_TRANSFER_META_LEN} values, got {len(meta_values)}"
        )
    num_proposals, payload_len, _plan_id, _step_id, header_len = meta_values
    if int(header_len) != READY_EAGER_TRANSFER_HEADER_LEN:
        raise ValueError(
            f"ready eager proposal transfer header length mismatch: expected="
            f"{READY_EAGER_TRANSFER_HEADER_LEN}, got {header_len}"
        )
    payload_values = [int(value) for value in payload]
    if int(payload_len) != len(payload_values):
        raise ValueError(
            f"ready eager proposal transfer payload length mismatch: "
            f"meta={payload_len}, payload={len(payload_values)}"
        )

    proposals: list[ReadyEagerProposal] = []
    offset = 0
    for idx in range(int(num_proposals)):
        header = payload_values[offset:offset + READY_EAGER_TRANSFER_HEADER_LEN]
        if len(header) != READY_EAGER_TRANSFER_HEADER_LEN:
            raise ValueError(f"ready eager proposal payload ended while reading header index={idx}")
        offset += READY_EAGER_TRANSFER_HEADER_LEN
        (
            proposal_id,
            seq_id,
            request_id,
            source_plan_id,
            source_step_id,
            created_step_id,
            base_len,
            base_pre_verify,
            proposal_len,
            to_verify_len,
            proposal_token_len,
            state_value,
            max_ready_proposal_age,
            parent_proposal_id,
            parent_ready,
            ready,
            scheduled,
            seq_epoch,
            tokens_digest,
        ) = header
        to_be_verified_token_ids = payload_values[offset:offset + to_verify_len]
        offset += to_verify_len
        proposal_token_ids = payload_values[offset:offset + proposal_token_len]
        offset += proposal_token_len
        if len(to_be_verified_token_ids) != to_verify_len:
            raise ValueError(
                f"ready eager proposal payload ended while reading verify tokens index={idx}"
            )
        if len(proposal_token_ids) != proposal_token_len:
            raise ValueError(f"ready eager proposal payload ended while reading tokens index={idx}")
        proposals.append(
            ReadyEagerProposal(
                proposal_id=proposal_id,
                seq_id=seq_id,
                request_id=None if request_id < 0 else request_id,
                base_len=base_len,
                base_pre_verify=bool(base_pre_verify),
                proposal_len=proposal_len,
                to_verify_len=to_verify_len,
                to_be_verified_token_ids=to_be_verified_token_ids,
                proposal_token_ids=proposal_token_ids,
                created_step_id=created_step_id,
                source_plan_id=source_plan_id,
                source_step_id=source_step_id,
                parent_proposal_id=None if parent_proposal_id < 0 else parent_proposal_id,
                parent_ready=bool(parent_ready),
                ready=bool(ready),
                scheduled=bool(scheduled),
                seq_epoch=None if seq_epoch < 0 else seq_epoch,
                tokens_digest=None if tokens_digest < 0 else tokens_digest,
                state=_decode_ready_eager_state(state_value),
                max_ready_proposal_age=max_ready_proposal_age,
            )
        )
    if offset != len(payload_values):
        raise ValueError(
            f"ready eager proposal transfer payload has trailing values: "
            f"consumed={offset}, payload={len(payload_values)}"
        )
    return proposals


class DualBatchManager:
    """Assign sticky home batches and produce breadth-only A/B StepPlans."""

    def __init__(self, gamma: int):
        self.gamma = int(gamma)
        self.batches = {0: BatchState(0), 1: BatchState(1)}
        self.step_id = 0
        self.ready_eager_proposals = ReadyEagerProposalRegistry()
        self._managed_seq_ids: set[int] = set()

    def reset(self) -> None:
        for batch in self.batches.values():
            batch.seq_ids.clear()
        self.step_id = 0
        self.ready_eager_proposals.clear()

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
        self._managed_seq_ids.add(int(seq.seq_id))
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

    @property
    def pending_batch_ids(self) -> list[int]:
        """All batch ids managed by this dual-batch instance.

        Returns all known batch ids (always [0, 1]) so that every proposal
        buffered by the dual-batch system is protected from ``discard_inactive``,
        regardless of per-rank ``scheduler.running`` state.
        """
        return sorted(self.batches.keys())

    @property
    def all_managed_seq_ids(self) -> list[int]:
        """Every seq_id ever assigned to a batch, monotonically growing.

        This set is identical across TP ranks because batch assignments are
        deterministic.  It is the correct TP-consistent guard for
        ``discard_inactive``: a proposal for a managed seq_id must never be
        evicted merely because that seq_id left the local ``scheduler.running``.
        """
        return sorted(self._managed_seq_ids)

    def home_batch_ids(self) -> dict[int, int]:
        mapping = {}
        for batch_id, batch in self.batches.items():
            for seq_id in batch.seq_ids:
                mapping[int(seq_id)] = int(batch_id)
        return mapping

    def emit_ready_eager_proposal(self, proposal: ReadyEagerProposal) -> bool:
        return self.ready_eager_proposals.upsert(proposal, enqueue_for_sync=True)

    def receive_ready_eager_proposals(self, proposals: Iterable[ReadyEagerProposal]) -> list[int]:
        stored = []
        for proposal in proposals:
            if self.ready_eager_proposals.upsert(proposal):
                stored.append(int(proposal.proposal_id))
        return sorted(stored)

    def pop_ready_eager_proposals_for_sync(self) -> list[ReadyEagerProposal]:
        return self.ready_eager_proposals.pop_outbox()

    def _sequence_status_name(self, seq: Sequence | None) -> str:
        if seq is None:
            return "MISSING"
        status = getattr(seq, "status", None)
        return status.name if isinstance(status, SequenceStatus) else str(status)

    def _mark_ready_proposal_terminal(
        self,
        proposal_id: int,
        state: str,
        ids: list[int],
        reason_by_id: dict[int, str],
        reason: str,
    ) -> None:
        self.ready_eager_proposals.mark_state(proposal_id, state)
        ids.append(int(proposal_id))
        reason_by_id[int(proposal_id)] = str(reason)

    def _refresh_ready_takeover_trace_fields(
        self,
        plan: StepPlan,
        routed_proposals: list[ReadyEagerProposal],
        routed_step_by_id: dict[int, int],
    ) -> None:
        routed_proposal_ids = [int(proposal.proposal_id) for proposal in routed_proposals]
        routed_seq_ids = [int(proposal.seq_id) for proposal in routed_proposals]
        pending_proposals = [
            proposal
            for proposal in self.ready_eager_proposals.proposals()
            if proposal.state == READY_EAGER_STATE_CONSUMED_APPLIED
            and proposal.takeover_routed_step_id is None
        ]
        target_seq_ids = {int(seq_id) for seq_id in plan.target_home_set}
        pending_proposal_ids = [int(proposal.proposal_id) for proposal in pending_proposals]
        plan.ready_eager_proposal_takeover_routed_ids = list(routed_proposal_ids)
        plan.ready_eager_proposal_takeover_routed_seq_ids = list(routed_seq_ids)
        plan.ready_eager_proposal_takeover_routed_step_by_id = dict(
            sorted((int(proposal_id), int(step_id)) for proposal_id, step_id in routed_step_by_id.items())
        )
        plan.ready_eager_proposal_pending_takeover_ids = list(pending_proposal_ids)
        plan.ready_eager_proposal_pending_takeover_proposal_ids = list(pending_proposal_ids)
        plan.ready_eager_proposal_pending_takeover_seq_ids = [
            int(proposal.seq_id) for proposal in pending_proposals
        ]
        plan.ready_eager_proposal_takeover_waiting_for_target_home_ids = [
            int(proposal.proposal_id)
            for proposal in pending_proposals
            if int(proposal.seq_id) not in target_seq_ids
        ]

    def _apply_ready_eager_proposals_to_plan(
        self,
        plan: StepPlan,
        running_seqs: Iterable[Sequence] | None,
        enable_eager_lane_exclusion_dry_run: bool,
    ) -> None:
        if not enable_eager_lane_exclusion_dry_run:
            return

        current_step_id = int(plan.step_id if plan.step_id is not None else self.step_id)
        original_draft_home = [int(seq_id) for seq_id in plan.draft_home_set]
        target_home_set = {int(seq_id) for seq_id in plan.target_home_set}
        draft_home_set = set(original_draft_home)
        seq_by_id = {int(seq.seq_id): seq for seq in (running_seqs or [])}
        ready_proposals = self.ready_eager_proposals.ready()
        ready_proposal_ids = [int(proposal.proposal_id) for proposal in ready_proposals]
        takeover_proposals = self.ready_eager_proposals.pending_takeover_for_target(plan.target_home_set)
        takeover_seq_ids = [int(proposal.seq_id) for proposal in takeover_proposals]
        takeover_proposal_ids = [int(proposal.proposal_id) for proposal in takeover_proposals]
        already_takeover_routed_ids = [
            int(proposal.proposal_id)
            for proposal in self.ready_eager_proposals.proposals()
            if proposal.state == READY_EAGER_STATE_CONSUMED_APPLIED
            and proposal.takeover_routed_step_id is not None
        ]
        takeover_routed_step_by_id: dict[int, int] = {}

        plan.enable_eager_lane_exclusion_dry_run = True
        plan.eager_lane_exclusion_dry_run_enabled = True
        plan.raw_target_home_set_for_normal_verify = [int(seq_id) for seq_id in plan.target_home_set]
        plan.target_eager_verify_seq_ids_dry_run = list(takeover_seq_ids)
        plan.target_eager_verify_proposal_ids_dry_run = list(takeover_proposal_ids)
        plan.target_eager_verify_reason_by_seq_id_dry_run = {
            int(proposal.seq_id): "ready_eager_takeover_for_target_normal_verify"
            for proposal in takeover_proposals
        }
        plan.excluded_from_target_normal_verify_for_eager_dry_run = list(takeover_seq_ids)
        plan.target_normal_verify_seq_ids = [
            int(seq_id) for seq_id in plan.target_home_set if int(seq_id) not in set(takeover_seq_ids)
        ]
        plan.missing_normal_proposal_allowed_by_eager_dry_run = bool(takeover_seq_ids)
        plan.missing_normal_proposal_allowed_seq_ids_dry_run = list(takeover_seq_ids)
        for proposal in takeover_proposals:
            self.ready_eager_proposals.mark_takeover_routed(proposal.proposal_id, current_step_id)
            takeover_routed_step_by_id[int(proposal.proposal_id)] = int(current_step_id)
        plan.original_draft_home_set = list(original_draft_home)
        plan.actual_draft_home_set_for_normal_draft = list(original_draft_home)
        plan.adjusted_draft_home_set_dry_run = list(original_draft_home)
        plan.excluded_from_draft_home_for_eager_dry_run = []
        plan.lane_excluded_seq_ids = []
        plan.pending_lane_exclusion_decision_ids_before_plan = []
        plan.applied_lane_exclusion_decision_ids = []
        plan.stale_lane_exclusion_decision_ids = []
        plan.expired_lane_exclusion_decision_ids = []
        plan.lane_exclusion_drop_reason_by_decision_id = {}
        plan.active_pending_lane_exclusion_decision_ids = []
        plan.terminal_lane_exclusion_decision_ids = []
        plan.touched_lane_exclusion_decision_ids = []
        plan.normal_proposal_expected_seq_ids_after_lane_exclusion = list(original_draft_home)
        plan.adjusted_normal_proposal_expected_seq_ids = list(original_draft_home)
        plan.lane_exclusion_decision_available_before_draft = False
        plan.lane_exclusion_deferred_until_next_step = False
        plan.lane_exclusion_defer_reason = None
        plan.lane_exclusion_source_step_by_decision_id = {}
        plan.lane_exclusion_created_step_by_decision_id = {}
        plan.lane_exclusion_applied_step_by_decision_id = {}
        plan.lane_exclusion_applied_proposal_ids = []
        plan.lane_exclusion_applied_seq_ids = []
        plan.lane_exclusion_apply_reason_by_proposal_id = {}
        plan.ready_eager_proposal_registry_ids_before_plan = list(ready_proposal_ids)
        plan.ready_eager_proposal_seen_by_scheduler_ids = list(ready_proposal_ids)
        plan.ready_eager_proposal_in_target_home_ids = []
        plan.ready_eager_proposal_in_draft_home_ids = []
        plan.ready_eager_proposal_applied_ids = []
        plan.ready_eager_proposal_stale_ids = []
        plan.ready_eager_proposal_expired_ids = []
        plan.ready_eager_proposal_invalidated_ids = []
        plan.ready_eager_proposal_state_by_id = {}
        plan.ready_eager_proposal_skip_reason_by_id = {}
        plan.ready_eager_proposal_stale_reason_by_id = {}
        plan.ready_eager_proposal_age_by_id = {}
        plan.ready_eager_proposal_seq_id_by_id = {}
        plan.ready_eager_proposal_base_len_by_id = {}
        plan.ready_eager_proposal_current_len_by_id = {}
        plan.ready_eager_proposal_current_pre_verify_by_id = {}
        plan.ready_eager_proposal_current_status_by_id = {}
        plan.ready_eager_proposal_apply_step_by_id = {}
        plan.ready_eager_proposal_already_takeover_routed_ids = sorted(already_takeover_routed_ids)
        plan.repeated_takeover_proposal_ids = []
        self._refresh_ready_takeover_trace_fields(plan, takeover_proposals, takeover_routed_step_by_id)

        if not ready_proposals:
            if takeover_proposals:
                plan.ready_eager_proposal_state_by_id = {
                    int(proposal.proposal_id): proposal.state
                    for proposal in takeover_proposals
                }
            self._refresh_ready_takeover_trace_fields(plan, takeover_proposals, takeover_routed_step_by_id)
            return

        applied_records: list[LaneExclusionApplyRecord] = []
        applied_seq_ids: set[int] = set()
        stale_ids: list[int] = []
        expired_ids: list[int] = []
        invalidated_ids: list[int] = []
        skip_reasons: dict[int, str] = {}
        stale_reasons: dict[int, str] = {}

        for proposal in ready_proposals:
            proposal_id = int(proposal.proposal_id)
            seq_id = int(proposal.seq_id)
            seq = seq_by_id.get(seq_id)
            age = max(0, int(current_step_id) - int(proposal.created_step_id))
            current_len = -1 if seq is None else int(len(seq))
            current_pre_verify = True if seq is None else bool(getattr(seq, "pre_verify", True))
            current_status = self._sequence_status_name(seq)
            plan.ready_eager_proposal_age_by_id[proposal_id] = age
            plan.ready_eager_proposal_seq_id_by_id[proposal_id] = seq_id
            plan.ready_eager_proposal_base_len_by_id[proposal_id] = int(proposal.base_len)
            plan.ready_eager_proposal_current_len_by_id[proposal_id] = current_len
            plan.ready_eager_proposal_current_pre_verify_by_id[proposal_id] = current_pre_verify
            plan.ready_eager_proposal_current_status_by_id[proposal_id] = current_status
            if seq_id in target_home_set:
                plan.ready_eager_proposal_in_target_home_ids.append(proposal_id)
            if seq_id in draft_home_set:
                plan.ready_eager_proposal_in_draft_home_ids.append(proposal_id)

            if age > int(proposal.max_ready_proposal_age):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_EXPIRED,
                    expired_ids,
                    stale_reasons,
                    "expired",
                )
                continue
            if plan.execution_mode != "dual_batch_pearl" or plan.plan_phase != "steady":
                skip_reasons[proposal_id] = "non_steady_plan"
                continue
            if seq_id in target_home_set:
                skip_reasons[proposal_id] = "still_in_target_home"
                continue
            if seq_id not in draft_home_set:
                skip_reasons[proposal_id] = "not_in_candidate_draft_home"
                continue
            if seq is None:
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "seq_not_found",
                )
                continue
            if getattr(seq, "status", None) != SequenceStatus.RUNNING:
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "seq_not_running",
                )
                continue
            if bool(getattr(seq, "is_finished", False)):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "seq_finished",
                )
                continue
            if bool(getattr(seq, "pre_verify", True)):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "seq_returned_pre_verify",
                )
                continue
            if bool(proposal.base_pre_verify):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "invalid_base_pre_verify",
                )
                continue
            if int(proposal.proposal_len) != int(self.gamma):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "invalid_proposal_len",
                )
                continue
            if int(proposal.to_verify_len) != int(self.gamma):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "invalid_to_verify_len",
                )
                continue
            if current_len < int(proposal.base_len):
                skip_reasons[proposal_id] = "wait_base_not_reached"
                continue
            if current_len > int(proposal.base_len):
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "base_overshot",
                )
                continue
            if seq_id in applied_seq_ids:
                self._mark_ready_proposal_terminal(
                    proposal_id,
                    READY_EAGER_STATE_STALE,
                    stale_ids,
                    stale_reasons,
                    "duplicate_ready_eager_proposal_for_seq",
                )
                continue

            self.ready_eager_proposals.mark_state(proposal_id, READY_EAGER_STATE_CONSUMED_APPLIED)
            applied_records.append(
                LaneExclusionApplyRecord(
                    proposal_id=proposal_id,
                    seq_id=seq_id,
                    apply_step_id=current_step_id,
                    apply_plan_id=int(plan.plan_id),
                )
            )
            applied_seq_ids.add(seq_id)

        if skip_reasons and not applied_records:
            plan.lane_exclusion_deferred_until_next_step = True
            plan.lane_exclusion_defer_reason = next(iter(skip_reasons.values()))

        plan.ready_eager_proposal_stale_ids = sorted(stale_ids)
        plan.ready_eager_proposal_expired_ids = sorted(expired_ids)
        plan.ready_eager_proposal_invalidated_ids = sorted(invalidated_ids)
        plan.ready_eager_proposal_skip_reason_by_id = {
            int(proposal_id): reason for proposal_id, reason in sorted(skip_reasons.items())
        }
        plan.ready_eager_proposal_stale_reason_by_id = {
            int(proposal_id): reason for proposal_id, reason in sorted(stale_reasons.items())
        }
        plan.ready_eager_proposal_state_by_id = {
            int(proposal_id): self.ready_eager_proposals._proposals[int(proposal_id)].state
            for proposal_id in ready_proposal_ids + takeover_proposal_ids
            if int(proposal_id) in self.ready_eager_proposals._proposals
        }

        if not applied_records:
            return

        applied_records = sorted(
            applied_records,
            key=lambda record: original_draft_home.index(int(record.seq_id))
            if int(record.seq_id) in draft_home_set
            else len(original_draft_home),
        )
        excluded_seq_ids = [int(record.seq_id) for record in applied_records]
        proposal_ids = [int(record.proposal_id) for record in applied_records]
        adjusted_draft_home = [
            int(seq_id) for seq_id in original_draft_home if int(seq_id) not in set(excluded_seq_ids)
        ]

        plan.draft_home_set = list(adjusted_draft_home)
        plan.actual_draft_home_set_for_normal_draft = list(adjusted_draft_home)
        plan.adjusted_draft_home_set_dry_run = list(adjusted_draft_home)
        plan.excluded_from_draft_home_for_eager_dry_run = list(excluded_seq_ids)
        plan.lane_excluded_seq_ids = list(excluded_seq_ids)
        plan.normal_proposal_expected_seq_ids_after_lane_exclusion = list(adjusted_draft_home)
        plan.adjusted_normal_proposal_expected_seq_ids = list(adjusted_draft_home)
        plan.lane_exclusion_decision_available_before_draft = True
        plan.lane_exclusion_deferred_until_next_step = False
        plan.lane_exclusion_defer_reason = None
        plan.ready_eager_proposal_applied_ids = list(proposal_ids)
        plan.ready_eager_proposal_apply_step_by_id = {
            int(record.proposal_id): int(record.apply_step_id) for record in applied_records
        }
        plan.lane_exclusion_applied_proposal_ids = list(proposal_ids)
        plan.lane_exclusion_applied_seq_ids = list(excluded_seq_ids)
        plan.lane_exclusion_apply_reason_by_proposal_id = {
            int(record.proposal_id): record.reason for record in applied_records
        }
        takeover_seq_id_set = set(int(seq_id) for seq_id in plan.target_eager_verify_seq_ids_dry_run)
        plan.excluded_from_target_normal_verify_for_eager_dry_run = list(
            plan.target_eager_verify_seq_ids_dry_run
        )
        plan.target_normal_verify_seq_ids = [
            int(seq_id)
            for seq_id in plan.raw_target_home_set_for_normal_verify or plan.target_home_set
            if int(seq_id) not in takeover_seq_id_set
        ]
        plan.missing_normal_proposal_allowed_by_eager_dry_run = bool(plan.target_eager_verify_seq_ids_dry_run)
        plan.missing_normal_proposal_allowed_seq_ids_dry_run = list(plan.target_eager_verify_seq_ids_dry_run)
        self._refresh_ready_takeover_trace_fields(plan, takeover_proposals, takeover_routed_step_by_id)
        plan.eager_schedule_dry_run_enabled = True
        plan.eager_lane_exclusion_proposal_ids = list(proposal_ids)
        plan.eager_lane_exclusion_seq_ids = list(excluded_seq_ids)
        plan.eager_lane_exclusion_reason_by_seq_id = {
            int(record.seq_id): record.reason for record in applied_records
        }
        plan.lane_exclusion_dry_run_done = True
        plan.lane_exclusion_dry_run_done_proposal_ids = list(proposal_ids)
        plan.lane_exclusion_dry_run_done_seq_ids = list(excluded_seq_ids)
        plan.ready_eager_proposal_state_by_id = {
            int(proposal_id): self.ready_eager_proposals._proposals[int(proposal_id)].state
            for proposal_id in ready_proposal_ids + takeover_proposal_ids
            if int(proposal_id) in self.ready_eager_proposals._proposals
        }

    def build_step_plan(
        self,
        plan_id: int,
        iteration_id: int,
        execution_mode: str,
        decode_ready_mode: bool,
        pending_proposal_seq_ids: list[int],
        pending_batch_ids: list[int],
        enable_eager_execution: bool = False,
        enable_eager_lane_exclusion_dry_run: bool = False,
        running_seqs: Iterable[Sequence] | None = None,
    ) -> StepPlan:
        active = self.active_batch_ids()
        target_batch_id: Optional[int] = None
        draft_batch_id: Optional[int] = None
        phase = "fallback"
        fallback_reason: Optional[str] = None
        active_batch_count = len(active)
        active_seq_count = sum(len(self.batches[batch_id].seq_ids) for batch_id in active)

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
                fallback_reason = "single_active_batch_with_buffered_proposals"
            else:
                target_batch_id = only_batch_id
                draft_batch_id = only_batch_id
                fallback_reason = "single_active_batch_without_buffered_proposals"
            phase = "fallback"
        elif len(active) == 0:
            phase = "fallback"
            fallback_reason = "no_active_batch"
        else:
            phase = "fallback"
            fallback_reason = "unknown_fallback_condition"

        target_home_set = self.batch_seq_ids(target_batch_id)
        draft_home_set = self.batch_seq_ids(draft_batch_id)
        involved_seq_ids = sorted(set(target_home_set) | set(draft_home_set))
        budgets = {
            seq_id: RequestBudget(normal_gamma=self.gamma, eager_gamma=0)
            for seq_id in involved_seq_ids
        }
        target_home_size = len(target_home_set)
        draft_home_size = len(draft_home_set)
        active_denominator = max(1, active_seq_count)
        split_denominator = max(1, target_home_size + draft_home_size)
        target_fraction_of_active = target_home_size / active_denominator
        draft_fraction_of_active = draft_home_size / active_denominator
        split_imbalance = abs(target_home_size - draft_home_size) / split_denominator
        target_to_draft_size_ratio = target_home_size / max(1, draft_home_size)

        if phase == "fallback" and fallback_reason is None:
            if target_batch_id is None and draft_batch_id is not None:
                fallback_reason = "empty_target_batch"
            elif draft_batch_id is None and target_batch_id is not None:
                fallback_reason = "empty_draft_batch"
            else:
                fallback_reason = "unknown_fallback_condition"

        plan = StepPlan(
            plan_id=plan_id,
            iteration_id=iteration_id,
            execution_mode=execution_mode,
            target_home_set=target_home_set,
            target_normal_verify_seq_ids=list(target_home_set),
            raw_target_home_set_for_normal_verify=list(target_home_set),
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
            fallback_reason=fallback_reason,
            steady_step=phase == "steady",
            priming_step=phase == "priming",
            fallback_has_target_batch=phase == "fallback" and target_batch_id is not None,
            fallback_has_draft_batch=phase == "fallback" and draft_batch_id is not None,
            fallback_active_batch_count=active_batch_count if phase == "fallback" else 0,
            fallback_active_seq_count=active_seq_count if phase == "fallback" else 0,
            fallback_pending_proposal_count=len(pending_proposal_seq_ids) if phase == "fallback" else 0,
            fallback_target_seq_count=target_home_size if phase == "fallback" else 0,
            fallback_draft_seq_count=draft_home_size if phase == "fallback" else 0,
            active_seq_count=active_seq_count,
            active_batch_count=active_batch_count,
            target_fraction_of_active=target_fraction_of_active,
            draft_fraction_of_active=draft_fraction_of_active,
            split_imbalance=split_imbalance,
            target_to_draft_size_ratio=target_to_draft_size_ratio,
        )
        plan_state = DualBatchPlanState(
            target_batch_id=target_batch_id,
            draft_batch_id=draft_batch_id,
            step_id=self.step_id,
            phase=phase,
            fallback_reason=fallback_reason,
        )
        self._apply_ready_eager_proposals_to_plan(
            plan,
            running_seqs=running_seqs,
            enable_eager_lane_exclusion_dry_run=enable_eager_lane_exclusion_dry_run,
        )
        if enable_eager_execution:
            plan.validate_phase1h_eager_scaffold(enable_eager_execution=True)
        else:
            plan.validate_phase1c()
        self.step_id += 1
        plan.dual_batch_state = plan_state
        return plan
