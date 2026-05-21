"""Legacy-equivalent ST-Spec step planning scaffold.

This module intentionally keeps default scheduling legacy-equivalent. It records
the current Scheduler.schedule() decision in a structured StepPlan so future work
can add dual batches, eager paths, and SLO-aware allocation behind a stable
interface. V4A adds an explicitly gated real two-batch feasibility probe that may
fail fast when the current PEARL protocol cannot represent filtered draft/verify
subsets; final real execution still requires a later protocol refactor.
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
    # V4A probe metadata. The real-probe path is intentionally experimental:
    # it may expose PEARL protocol assumptions by failing fast before any
    # cross-runner communication instead of silently changing default behavior.
    stspec_probe_enabled: bool = False
    stspec_probe_fail_fast: bool = True
    stspec_probe_local_only: bool = False
    real_probe_attempted: bool = False
    real_probe_applied: bool = False
    real_probe_blocked: bool = False
    real_probe_block_reason: str | None = None
    protocol_alignment_ok: bool = True
    protocol_alignment_error: str | None = None
    actual_target_exec_seq_ids: list[int] = field(default_factory=list)
    actual_draft_exec_seq_ids: list[int] = field(default_factory=list)
    actual_exec_seq_ids: list[int] = field(default_factory=list)
    dryrun_target_exec_seq_ids: list[int] = field(default_factory=list)
    dryrun_draft_exec_seq_ids: list[int] = field(default_factory=list)
    filtered_out_seq_ids: list[int] = field(default_factory=list)
    actual_exec_fraction: float = 1.0
    stspec_pipeline_enabled: bool = False
    stspec_pipeline_phase: str = "disabled"
    stspec_pipeline_step: int = 0
    stspec_pipeline_warmup_done: bool = True
    stspec_warmup_target_home_batch_id: int | None = None
    stspec_warmup_draft_home_batch_id: int | None = None
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
            "stspec_probe_enabled": self.stspec_probe_enabled,
            "stspec_probe_fail_fast": self.stspec_probe_fail_fast,
            "stspec_probe_local_only": self.stspec_probe_local_only,
            "real_probe_attempted": self.real_probe_attempted,
            "real_probe_applied": self.real_probe_applied,
            "real_probe_blocked": self.real_probe_blocked,
            "real_probe_block_reason": self.real_probe_block_reason,
            "protocol_alignment_ok": self.protocol_alignment_ok,
            "protocol_alignment_error": self.protocol_alignment_error,
            "actual_target_exec_seq_ids": list(self.actual_target_exec_seq_ids),
            "actual_draft_exec_seq_ids": list(self.actual_draft_exec_seq_ids),
            "actual_exec_seq_ids": list(self.actual_exec_seq_ids),
            "dryrun_target_exec_seq_ids": list(self.dryrun_target_exec_seq_ids),
            "dryrun_draft_exec_seq_ids": list(self.dryrun_draft_exec_seq_ids),
            "filtered_out_seq_ids": list(self.filtered_out_seq_ids),
            "actual_exec_fraction": self.actual_exec_fraction,
            "stspec_pipeline_enabled": self.stspec_pipeline_enabled,
            "stspec_pipeline_phase": self.stspec_pipeline_phase,
            "stspec_pipeline_step": self.stspec_pipeline_step,
            "stspec_pipeline_warmup_done": self.stspec_pipeline_warmup_done,
            "stspec_warmup_target_home_batch_id": self.stspec_warmup_target_home_batch_id,
            "stspec_warmup_draft_home_batch_id": self.stspec_warmup_draft_home_batch_id,
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


def actual_exec_ids_for_runner(
    runner_role: str,
    actual_target_exec_seq_ids: list[int],
    actual_draft_exec_seq_ids: list[int],
) -> list[int]:
    if "draft" in runner_role:
        return list(actual_draft_exec_seq_ids)
    return list(actual_target_exec_seq_ids)


def select_exec_seqs_for_plan(
    seqs: list[Sequence],
    step_plan: StepPlan,
    runner_role: str,
) -> list[Sequence]:
    """Return the runner-local execution subset for a StepPlan.

    The helper is intentionally conservative for V4A: it never mutates
    ``scheduled_seq_ids``, preserves the original scheduler order, and raises a
    controlled diagnostic if the selected subset would be empty or invalid.
    """
    scheduled_seq_ids = [seq.seq_id for seq in seqs]
    if list(step_plan.scheduled_seq_ids) != scheduled_seq_ids:
        raise RuntimeError(
            "ST-Spec execution selection received seqs that do not match the StepPlan: "
            f"plan_id={step_plan.plan_id}, runner_role={runner_role}, "
            f"scheduled_seq_ids={step_plan.scheduled_seq_ids}, input_seq_ids={scheduled_seq_ids}"
        )

    actual_exec_seq_ids = actual_exec_ids_for_runner(
        runner_role,
        list(step_plan.actual_target_exec_seq_ids),
        list(step_plan.actual_draft_exec_seq_ids),
    )
    if step_plan.is_prefill:
        actual_exec_seq_ids = list(scheduled_seq_ids)

    if len(actual_exec_seq_ids) != len(set(actual_exec_seq_ids)):
        raise RuntimeError(
            "ST-Spec execution selection found duplicate actual exec ids: "
            f"plan_id={step_plan.plan_id}, runner_role={runner_role}, "
            f"actual_exec_seq_ids={actual_exec_seq_ids}"
        )
    missing = [seq_id for seq_id in actual_exec_seq_ids if seq_id not in set(scheduled_seq_ids)]
    if missing:
        raise RuntimeError(
            "ST-Spec execution selection found actual ids outside scheduled ids: "
            f"plan_id={step_plan.plan_id}, runner_role={runner_role}, "
            f"scheduled_seq_ids={scheduled_seq_ids}, actual_exec_seq_ids={actual_exec_seq_ids}, "
            f"missing={missing}"
        )
    if not actual_exec_seq_ids:
        raise RuntimeError(
            "ST-Spec execution selection produced an empty actual exec set; empty model "
            "forward is unsupported in V4A: "
            f"plan_id={step_plan.plan_id}, runner_role={runner_role}, "
            f"scheduled_seq_ids={scheduled_seq_ids}, target_batch_seq_ids={step_plan.target_batch_seq_ids}, "
            f"draft_home_batch_seq_ids={step_plan.draft_home_batch_seq_ids}"
        )

    actual_set = set(actual_exec_seq_ids)
    exec_seqs = [seq for seq in seqs if seq.seq_id in actual_set]
    if [seq.seq_id for seq in exec_seqs] != actual_exec_seq_ids:
        raise RuntimeError(
            "ST-Spec execution selection could not preserve actual exec order: "
            f"plan_id={step_plan.plan_id}, runner_role={runner_role}, "
            f"scheduled_seq_ids={scheduled_seq_ids}, actual_exec_seq_ids={actual_exec_seq_ids}, "
            f"selected_seq_ids={[seq.seq_id for seq in exec_seqs]}"
        )
    return exec_seqs


def stspec_protocol_alignment_error(
    step_plan: StepPlan,
    runner_role: str,
    gamma: int,
    layout_kind: str = "legacy_fixed",
) -> str | None:
    """Return a V4A protocol blocker message, or None if alignment is safe.

    Current PEARL verification packs draft and target messages by sequence index
    and by ``gamma * len(seqs)`` layout. The V4A real probe is allowed to fail
    before communication when filtered draft and verify subsets differ; a later
    protocol refactor is required for final real two-batch execution.
    """
    if (
        not step_plan.real_probe_attempted
        or step_plan.stspec_probe_local_only
        or step_plan.is_prefill
        or step_plan.execution_mode not in {"parallel_pearl", "serialized_pearl"}
    ):
        return None

    draft_ids = list(step_plan.actual_draft_exec_seq_ids)
    target_ids = list(step_plan.actual_target_exec_seq_ids)
    if draft_ids == target_ids:
        return None
    actual_ids = actual_exec_ids_for_runner(
        runner_role,
        list(step_plan.actual_target_exec_seq_ids),
        list(step_plan.actual_draft_exec_seq_ids),
    )
    if layout_kind == "variable_offsets":
        # V4C made the PEARL envelope capable of representing divergent draft
        # and target sequence sets. V4D moves the remaining real-probe blocker
        # to the explicit mailbox/routing preflight in pearl_model_runner.py so
        # failures identify warmup misses or missing cross-process transport
        # instead of a generic protocol-representation error.
        return None
    return (
        "ST-Spec real two-batch probe cannot proceed: draft exec seq ids != "
        "target verify seq ids under current legacy_fixed PEARL message layout; "
        f"plan_id={step_plan.plan_id}, runner_role={runner_role}, "
        f"scheduled_seq_ids={step_plan.scheduled_seq_ids}, "
        f"actual_exec_seq_ids={actual_ids}, "
        f"target_batch_seq_ids={step_plan.target_batch_seq_ids}, "
        f"draft_home_batch_seq_ids={step_plan.draft_home_batch_seq_ids}, "
        f"actual_target_exec_seq_ids={target_ids}, "
        f"actual_draft_exec_seq_ids={draft_ids}, "
        f"gamma={gamma}, execution_mode={step_plan.execution_mode}; "
        "variable_offsets PEARL protocol support is required for divergent draft/verify seq sets"
    )


def validate_stspec_protocol_alignment(
    step_plan: StepPlan,
    runner_role: str,
    gamma: int,
    layout_kind: str = "legacy_fixed",
) -> None:
    error = stspec_protocol_alignment_error(step_plan, runner_role, gamma, layout_kind)
    if error is not None:
        raise RuntimeError(error)


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
    stspec_two_batch_probe: bool = False,
    stspec_two_batch_probe_fail_fast: bool = True,
    stspec_two_batch_probe_local_only: bool = False,
    stspec_pipeline_enabled: bool = False,
    stspec_pipeline_phase: str = "disabled",
    stspec_pipeline_step: int = 0,
    stspec_pipeline_warmup_done: bool = True,
    stspec_warmup_target_home_batch_id: int | None = None,
    stspec_warmup_draft_home_batch_id: int | None = None,
    batch_lock_active: bool = False,
) -> StepPlan:
    """Wrap the existing scheduler output in a legacy-equivalent StepPlan.

    The supplied ``seqs`` are used as-is. No sequence ordering, batching,
    budgets, KV state, or verification layout is changed by default. V3C dry-run
    keeps actual execution on legacy scheduled seqs; V4A real-probe mode records
    filtered actual exec sets so runners can validate protocol alignment and
    fail fast with a precise diagnostic before distributed communication.

    V4AC: When batch_lock_active, the draft runner's actual exec seqs are
    aligned with the target batch so corrections propagate to draft generation.
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

    real_probe_attempted = (
        two_batch_execution_enabled
        and not two_batch_execution_dryrun
        and stspec_two_batch_probe
    )
    if two_batch_execution_enabled and not two_batch_execution_dryrun and not stspec_two_batch_probe:
        raise NotImplementedError(
            "Real ST-Spec two-batch execution is only available in explicit V4A "
            "probe mode; pass stspec_two_batch_probe=True."
        )

    if real_probe_attempted:
        two_batch_execution_mode = "real_probe"
    else:
        two_batch_execution_mode = "dryrun" if two_batch_execution_enabled else "legacy"

    if is_prefill:
        dryrun_target_exec_seq_ids: list[int] = []
        dryrun_draft_exec_seq_ids: list[int] = []
    elif "draft" in runner_role:
        dryrun_target_exec_seq_ids = []
        dryrun_draft_exec_seq_ids = list(draft_home_batch_seq_ids)
    else:
        dryrun_target_exec_seq_ids = list(target_batch_seq_ids)
        dryrun_draft_exec_seq_ids = []

    # V4A is a feasibility probe, not the final two-batch protocol. In legacy
    # and dry-run modes, actual execution remains the legacy scheduled batch.
    # In real_probe decode mode, the StepPlan records filtered actual sets so
    # runner-side validation can fail fast with a precise protocol diagnostic if
    # PEARL cannot safely represent different draft/verify subsets.
    actual_target_exec_seq_ids = list(scheduled_seq_ids)
    actual_draft_exec_seq_ids = list(scheduled_seq_ids)
    if real_probe_attempted and not is_prefill:
        actual_target_exec_seq_ids = list(target_batch_seq_ids)
        # V4AC: during batch lock, draft generates for the locked target batch
        if batch_lock_active:
            actual_draft_exec_seq_ids = list(target_batch_seq_ids)
        else:
            actual_draft_exec_seq_ids = list(draft_home_batch_seq_ids)

    actual_exec_seq_ids = actual_exec_ids_for_runner(
        runner_role,
        actual_target_exec_seq_ids,
        actual_draft_exec_seq_ids,
    )
    if is_prefill:
        actual_exec_seq_ids = list(scheduled_seq_ids)
    filtered_out_seq_ids = [seq_id for seq_id in scheduled_seq_ids if seq_id not in set(actual_exec_seq_ids)]
    actual_exec_fraction = (
        float(len(actual_exec_seq_ids)) / float(len(scheduled_seq_ids))
        if scheduled_seq_ids
        else 1.0
    )

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
        stspec_probe_enabled=stspec_two_batch_probe,
        stspec_probe_fail_fast=stspec_two_batch_probe_fail_fast,
        stspec_probe_local_only=stspec_two_batch_probe_local_only,
        real_probe_attempted=real_probe_attempted,
        real_probe_applied=real_probe_attempted and bool(filtered_out_seq_ids) and not stspec_two_batch_probe_local_only,
        real_probe_blocked=False,
        real_probe_block_reason=None,
        protocol_alignment_ok=True,
        protocol_alignment_error=None,
        actual_target_exec_seq_ids=actual_target_exec_seq_ids,
        actual_draft_exec_seq_ids=actual_draft_exec_seq_ids,
        actual_exec_seq_ids=actual_exec_seq_ids,
        dryrun_target_exec_seq_ids=dryrun_target_exec_seq_ids,
        dryrun_draft_exec_seq_ids=dryrun_draft_exec_seq_ids,
        filtered_out_seq_ids=filtered_out_seq_ids,
        actual_exec_fraction=actual_exec_fraction,
        stspec_pipeline_enabled=stspec_pipeline_enabled,
        stspec_pipeline_phase=stspec_pipeline_phase,
        stspec_pipeline_step=stspec_pipeline_step,
        stspec_pipeline_warmup_done=stspec_pipeline_warmup_done,
        stspec_warmup_target_home_batch_id=stspec_warmup_target_home_batch_id,
        stspec_warmup_draft_home_batch_id=stspec_warmup_draft_home_batch_id,
        requests=requests,
    )
