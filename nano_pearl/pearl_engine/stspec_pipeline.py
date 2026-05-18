"""V4F ST-Spec two-batch pipeline warmup scaffold.

The helper models the diagnostic phase rhythm for the guarded real probe only:
first decode cycle is ``warmup_draft_only`` (draft fills mailbox for the draft
home batch while target records a warmup skip), then later cycles are
``steady_state``. It does not implement final target verification input
construction or KV/state synchronization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class STSpecPipelinePhase(str, Enum):
    DISABLED = "disabled"
    WARMUP_DRAFT_ONLY = "warmup_draft_only"
    STEADY_STATE = "steady_state"
    DRAIN = "drain"


@dataclass(frozen=True)
class STSpecPipelineState:
    enabled: bool
    phase: str
    step: int
    warmup_done: bool
    warmup_target_home_batch_id: int | None = None
    warmup_draft_home_batch_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class STSpecPipelineController:
    """Small deterministic controller for V4F probe diagnostics."""

    def __init__(self, *, enabled: bool, warmup_enabled: bool = True, warmup_draft_only: bool = True):
        self.enabled = bool(enabled)
        self.warmup_enabled = bool(warmup_enabled)
        self.warmup_draft_only = bool(warmup_draft_only)
        self.step = 0
        self.warmup_done = not (self.enabled and self.warmup_enabled and self.warmup_draft_only)
        self.warmup_target_home_batch_id: int | None = None
        self.warmup_draft_home_batch_id: int | None = None

    def state_for_next_decode(self, target_home_batch_id: int | None, draft_home_batch_id: int | None) -> STSpecPipelineState:
        if not self.enabled:
            phase = STSpecPipelinePhase.DISABLED.value
        elif not self.warmup_done:
            phase = STSpecPipelinePhase.WARMUP_DRAFT_ONLY.value
            if self.warmup_target_home_batch_id is None:
                self.warmup_target_home_batch_id = target_home_batch_id
            if self.warmup_draft_home_batch_id is None:
                self.warmup_draft_home_batch_id = draft_home_batch_id
        else:
            phase = STSpecPipelinePhase.STEADY_STATE.value
        return STSpecPipelineState(
            enabled=self.enabled,
            phase=phase,
            step=self.step,
            warmup_done=self.warmup_done,
            warmup_target_home_batch_id=self.warmup_target_home_batch_id,
            warmup_draft_home_batch_id=self.warmup_draft_home_batch_id,
        )

    def advance_after_decode(self) -> None:
        if self.enabled and not self.warmup_done:
            self.warmup_done = True
        if self.enabled:
            self.step += 1

    def clear(self) -> None:
        self.step = 0
        self.warmup_done = not (self.enabled and self.warmup_enabled and self.warmup_draft_only)
        self.warmup_target_home_batch_id = None
        self.warmup_draft_home_batch_id = None


def should_skip_target_for_warmup(*, phase: str, runner_role: str, allow_warmup_miss: bool) -> bool:
    return phase == STSpecPipelinePhase.WARMUP_DRAFT_ONLY.value and "draft" not in runner_role and bool(allow_warmup_miss)


def target_home_batch_for_steady_after_warmup(state: STSpecPipelineState) -> int | None:
    return state.warmup_draft_home_batch_id
