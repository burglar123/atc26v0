"""Legacy-equivalent ST-Spec step planning scaffold.

This module intentionally does *not* implement ST-Spec scheduling features yet.
It records the current Scheduler.schedule() decision in a structured StepPlan so
future work can add dual batches, eager paths, and SLO-aware allocation behind a
stable interface without changing today's runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nano_pearl.pearl_engine.sequence import Sequence


class PlanRole(str, Enum):
    """Logical role assigned to a request inside a StepPlan."""

    PREFILL = "prefill"
    DRAFT = "draft"
    TARGET = "target"


@dataclass(frozen=True)
class PlanRequest:
    seq_id: int
    request_id: Any
    role: PlanRole
    home_batch_id: int | str | None
    is_eager: bool
    effective_gamma: int
    draft_budget: int
    eager_budget: int
    slo_class: str | None
    slo_tpot_ms: float | None
    per_request_gamma: int | None


@dataclass(frozen=True)
class StepPlan:
    plan_id: int
    execution_mode: str
    decode_ready_mode: bool
    runner_role: str
    is_prefill: bool
    legacy_equivalent: bool
    scheduled_seq_ids: list[int] = field(default_factory=list)
    target_seq_ids: list[int] = field(default_factory=list)
    draft_home_seq_ids: list[int] = field(default_factory=list)
    eager_seq_ids: list[int] = field(default_factory=list)
    target_home_batch_id: int | None = None
    draft_home_batch_id: int | None = None
    target_batch_seq_ids: list[int] = field(default_factory=list)
    draft_home_batch_seq_ids: list[int] = field(default_factory=list)
    off_batch_seq_ids: list[int] = field(default_factory=list)
    plan_two_batch_shadow: bool = False
    two_batch_execution_enabled: bool = False
    two_batch_execution_dryrun: bool = True
    two_batch_execution_mode: str = "legacy"
    actual_target_exec_seq_ids: list[int] = field(default_factory=list)
    actual_draft_exec_seq_ids: list[int] = field(default_factory=list)
    dryrun_target_exec_seq_ids: list[int] = field(default_factory=list)
    dryrun_draft_exec_seq_ids: list[int] = field(default_factory=list)
    requests: list[PlanRequest] = field(default_factory=list)

    @property
    def effective_gamma_per_seq(self) -> dict[int, int]:
        return {request.seq_id: request.effective_gamma for request in self.requests}

    @property
    def home_batch_id_per_seq(self) -> dict[int, int | str | None]:
        return {request.seq_id: request.home_batch_id for request in self.requests}

    @property
    def is_eager_per_seq(self) -> dict[int, bool]:
        return {request.seq_id: request.is_eager for request in self.requests}

    @property
    def request_ids(self) -> list[Any]:
        return [request.request_id for request in self.requests]

    def signature(self) -> dict[str, Any]:
        """Return a compact JSON-serializable plan signature.

        The signature is diagnostic-only: it summarizes the legacy-equivalent
        decision already made by Scheduler.schedule() and does not participate
        in scheduling, verification, or KV-cache behavior. Per-sequence maps use
        string keys so the signature is stable under JSON serialization.
        """
        return {
            "plan_id": self.plan_id,
            "execution_mode": self.execution_mode,
            "decode_ready_mode": self.decode_ready_mode,
            "runner_role": self.runner_role,
            "is_prefill": self.is_prefill,
            "legacy_equivalent": self.legacy_equivalent,
            "target_home_batch_id": self.target_home_batch_id,
            "draft_home_batch_id": self.draft_home_batch_id,
            "target_batch_seq_ids": list(self.target_batch_seq_ids),
            "draft_home_batch_seq_ids": list(self.draft_home_batch_seq_ids),
            "off_batch_seq_ids": list(self.off_batch_seq_ids),
            "plan_two_batch_shadow": self.plan_two_batch_shadow,
            "two_batch_execution_enabled": self.two_batch_execution_enabled,
            "two_batch_execution_dryrun": self.two_batch_execution_dryrun,
            "two_batch_execution_mode": self.two_batch_execution_mode,
            "actual_target_exec_seq_ids": list(self.actual_target_exec_seq_ids),
            "actual_draft_exec_seq_ids": list(self.actual_draft_exec_seq_ids),
            "dryrun_target_exec_seq_ids": list(self.dryrun_target_exec_seq_ids),
            "dryrun_draft_exec_seq_ids": list(self.dryrun_draft_exec_seq_ids),
            "scheduled_seq_ids": list(self.scheduled_seq_ids),
            "request_ids": list(self.request_ids),
            "effective_gamma_per_seq": {
                str(seq_id): gamma
                for seq_id, gamma in self.effective_gamma_per_seq.items()
            },
            "home_batch_id_per_seq": {
                str(seq_id): home_batch_id
                for seq_id, home_batch_id in self.home_batch_id_per_seq.items()
            },
            "is_eager_per_seq": {
                str(seq_id): is_eager
                for seq_id, is_eager in self.is_eager_per_seq.items()
            },
        }

    def digest(self) -> str:
        return step_plan_digest(self)


def step_plan_signature(step_plan: StepPlan) -> dict[str, Any]:
    return step_plan.signature()


def step_plan_digest(step_plan: StepPlan) -> str:
    encoded = json.dumps(
        step_plan_signature(step_plan),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def role_from_runner(runner_role: str, is_prefill: bool) -> PlanRole:
    if is_prefill or runner_role.endswith("_prefill"):
        return PlanRole.PREFILL
    if "draft" in runner_role:
        return PlanRole.DRAFT
    return PlanRole.TARGET


def build_legacy_step_plan(
    *,
    plan_id: int,
    seqs: list[Sequence],
    is_prefill: bool,
    runner_role: str,
    execution_mode: str,
    decode_ready_mode: bool,
    default_gamma: int,
    target_home_batch_id: int | None = None,
    draft_home_batch_id: int | None = None,
    two_batch_execution_enabled: bool = False,
    two_batch_execution_dryrun: bool = True,
) -> StepPlan:
    """Wrap the existing scheduler output in a legacy-equivalent StepPlan.

    The supplied ``seqs`` are used as-is. No sequence ordering, batching,
    budgets, KV state, or verification layout is changed by this helper. V3C
    is a dry-run scaffold: actual execution still uses legacy scheduled seqs,
    while dryrun exec sets expose future verify(A) || draft(B) semantics.
    """
    scheduled_seq_ids = [seq.seq_id for seq in seqs]
    role = role_from_runner(runner_role, is_prefill)
    effective_gamma = max(int(default_gamma), 1)
    is_pearl_decode = (
        execution_mode in {"parallel_pearl", "serialized_pearl"}
        and not is_prefill
    )
    draft_budget = effective_gamma if is_pearl_decode else 1

    if "draft" in runner_role:
        draft_home_seq_ids = list(scheduled_seq_ids)
        target_seq_ids: list[int] = []
    else:
        draft_home_seq_ids = []
        target_seq_ids = list(scheduled_seq_ids)

    plan_two_batch_shadow = not is_prefill
    target_batch_seq_ids = [
        seq.seq_id for seq in seqs if seq.home_batch_id == target_home_batch_id
    ]
    draft_home_batch_seq_ids = [
        seq.seq_id for seq in seqs if seq.home_batch_id == draft_home_batch_id
    ]
    classified = set(target_batch_seq_ids) | set(draft_home_batch_seq_ids)
    off_batch_seq_ids = [seq.seq_id for seq in seqs if seq.seq_id not in classified]

    if two_batch_execution_enabled and not two_batch_execution_dryrun:
        raise NotImplementedError(
            "Real ST-Spec two-batch execution is not implemented in V3C."
        )
    two_batch_execution_mode = "dryrun" if two_batch_execution_enabled else "legacy"
    # V3C never filters execution. These actual sets intentionally mirror the
    # legacy scheduled batch for both runner families so model forward inputs are
    # unchanged; dryrun_* records the future filtered subsets only.
    actual_target_exec_seq_ids = list(scheduled_seq_ids)
    actual_draft_exec_seq_ids = list(scheduled_seq_ids)
    if is_prefill:
        dryrun_target_exec_seq_ids: list[int] = []
        dryrun_draft_exec_seq_ids: list[int] = []
    elif "draft" in runner_role:
        dryrun_target_exec_seq_ids = []
        dryrun_draft_exec_seq_ids = list(draft_home_batch_seq_ids)
    else:
        dryrun_target_exec_seq_ids = list(target_batch_seq_ids)
        dryrun_draft_exec_seq_ids = []

    requests = [
        PlanRequest(
            seq_id=seq.seq_id,
            request_id=seq.request_id,
            role=role,
            home_batch_id=seq.home_batch_id,
            is_eager=False,
            effective_gamma=effective_gamma,
            draft_budget=draft_budget,
            eager_budget=0,
            slo_class=seq.slo_class,
            slo_tpot_ms=seq.slo_tpot_ms,
            per_request_gamma=seq.per_request_gamma,
        )
        for seq in seqs
    ]

    return StepPlan(
        plan_id=plan_id,
        execution_mode=execution_mode,
        decode_ready_mode=decode_ready_mode,
        runner_role=runner_role,
        is_prefill=is_prefill,
        legacy_equivalent=True,
        scheduled_seq_ids=scheduled_seq_ids,
        target_seq_ids=target_seq_ids,
        draft_home_seq_ids=draft_home_seq_ids,
        eager_seq_ids=[],
        target_home_batch_id=target_home_batch_id,
        draft_home_batch_id=draft_home_batch_id,
        target_batch_seq_ids=target_batch_seq_ids,
        draft_home_batch_seq_ids=draft_home_batch_seq_ids,
        off_batch_seq_ids=off_batch_seq_ids,
        plan_two_batch_shadow=plan_two_batch_shadow,
        two_batch_execution_enabled=two_batch_execution_enabled,
        two_batch_execution_dryrun=two_batch_execution_dryrun,
        two_batch_execution_mode=two_batch_execution_mode,
        actual_target_exec_seq_ids=actual_target_exec_seq_ids,
        actual_draft_exec_seq_ids=actual_draft_exec_seq_ids,
        dryrun_target_exec_seq_ids=dryrun_target_exec_seq_ids,
        dryrun_draft_exec_seq_ids=dryrun_draft_exec_seq_ids,
        requests=requests,
    )
