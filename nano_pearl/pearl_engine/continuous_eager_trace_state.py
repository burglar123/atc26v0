"""Trace-only continuous eager state tracker.

Persists across steps in ModelRunnerBase.  Tracks per-seq proposal lifecycle:
pending_parent → ready → target_trace_ready → continue_pending → ...

No tokens, tensors, logits, or KV state are stored.  This is purely a
control-plane state machine for trace/diagnostic purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContinuousEagerTraceState:
    """Trace-only continuous eager proposal lifecycle tracker.

    Parallels EagerProposalBuffer but stores only metadata — no token ids,
    tensors, or KV state.  Used by the continuous eager trace path to model
    promotion/discard/continuation across steps.
    """

    # Per-seq latest state
    _latest_proposal_id: dict[int, str] = field(default_factory=dict)
    _latest_parent_proposal_id: dict[int, str] = field(default_factory=dict)
    _latest_parent_kind: dict[int, str] = field(default_factory=dict)
    _latest_state: dict[int, str] = field(default_factory=dict)
    _latest_step_id: dict[int, int] = field(default_factory=dict)
    _latest_plan_id: dict[int, int] = field(default_factory=dict)
    _latest_base_len: dict[int, int] = field(default_factory=dict)
    _chain_depth: dict[int, int] = field(default_factory=dict)

    # Global index sets
    _ready_seq_ids: set[int] = field(default_factory=set)
    _pending_seq_ids: set[int] = field(default_factory=set)
    _discarded_seq_ids: set[int] = field(default_factory=set)

    # Reasons
    _promotion_reason: dict[int, str] = field(default_factory=dict)
    _discard_reason: dict[int, str] = field(default_factory=dict)

    # Provenance
    _source_step_id: dict[int, int] = field(default_factory=dict)
    _source_plan_id: dict[int, int] = field(default_factory=dict)

    # History for chain tracking
    _history: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_new_pending(
        self,
        seq_id: int,
        proposal_id: str,
        parent_proposal_id: str,
        parent_kind: str,
        step_id: int,
        plan_id: int,
        base_len: int,
    ) -> None:
        """Record a new pending eager proposal (parent_kind='normal')."""
        self._latest_proposal_id[seq_id] = proposal_id
        self._latest_parent_proposal_id[seq_id] = parent_proposal_id
        self._latest_parent_kind[seq_id] = parent_kind
        self._latest_state[seq_id] = "pending_parent"
        self._latest_step_id[seq_id] = step_id
        self._latest_plan_id[seq_id] = plan_id
        self._latest_base_len[seq_id] = base_len
        self._chain_depth[seq_id] = 1
        self._source_step_id[seq_id] = step_id
        self._source_plan_id[seq_id] = plan_id
        self._pending_seq_ids.add(seq_id)
        self._ready_seq_ids.discard(seq_id)
        self._discarded_seq_ids.discard(seq_id)
        self._record_history(seq_id, "create_new_pending")

    def promote_pending(self, seq_id: int, proposal_id: str, reason: str) -> None:
        """Promote a pending proposal to ready."""
        self._latest_state[seq_id] = "ready"
        self._promotion_reason[seq_id] = reason
        self._pending_seq_ids.discard(seq_id)
        self._ready_seq_ids.add(seq_id)
        self._discarded_seq_ids.discard(seq_id)
        self._record_history(seq_id, "promote_pending")

    def discard_pending(self, seq_id: int, proposal_id: str, reason: str) -> None:
        """Discard a pending proposal."""
        self._latest_state[seq_id] = "discarded"
        self._discard_reason[seq_id] = reason
        self._pending_seq_ids.discard(seq_id)
        self._ready_seq_ids.discard(seq_id)
        self._discarded_seq_ids.add(seq_id)
        self._record_history(seq_id, "discard_pending")

    def mark_pending_unknown(self, seq_id: int, proposal_id: str, reason: str) -> None:
        """Mark a pending proposal as unknown (could not determine acceptance)."""
        self._latest_state[seq_id] = "pending_parent_unknown"
        self._discard_reason[seq_id] = reason
        self._pending_seq_ids.add(seq_id)
        self._record_history(seq_id, "mark_pending_unknown")

    def get_ready_seq_ids(self) -> set[int]:
        """Return seq_ids currently in 'ready' state."""
        return set(self._ready_seq_ids)

    def consume_ready_for_target_trace(self, seq_id: int) -> dict[str, Any] | None:
        """Transition a ready proposal to target_trace_ready.

        Returns metadata dict for populating target_eager_set_trace fields,
        or None if the seq_id is not in ready state.
        """
        if seq_id not in self._ready_seq_ids:
            return None
        self._latest_state[seq_id] = "target_trace_ready"
        self._ready_seq_ids.discard(seq_id)
        self._record_history(seq_id, "consume_ready_for_target_trace")
        return self.get_proposal_metadata(seq_id)

    def create_continue_pending(
        self,
        seq_id: int,
        proposal_id: str,
        parent_proposal_id: str,
        step_id: int,
        plan_id: int,
        base_len: int,
    ) -> None:
        """Create a continue_pending entry (parent_kind='eager')."""
        parent_depth = self._chain_depth.get(seq_id, 0)
        self._latest_proposal_id[seq_id] = proposal_id
        self._latest_parent_proposal_id[seq_id] = parent_proposal_id
        self._latest_parent_kind[seq_id] = "eager"
        self._latest_state[seq_id] = "continue_pending"
        self._latest_step_id[seq_id] = step_id
        self._latest_plan_id[seq_id] = plan_id
        self._latest_base_len[seq_id] = base_len
        self._chain_depth[seq_id] = parent_depth + 1
        self._source_step_id[seq_id] = step_id
        self._source_plan_id[seq_id] = plan_id
        self._pending_seq_ids.add(seq_id)
        self._ready_seq_ids.discard(seq_id)
        self._discarded_seq_ids.discard(seq_id)
        self._record_history(seq_id, "create_continue_pending")

    def get_pending_seq_ids(self) -> set[int]:
        """Return seq_ids with state 'pending_parent' or 'pending_parent_unknown'."""
        return {s for s in self._pending_seq_ids
                if self._latest_state.get(s) in ("pending_parent", "pending_parent_unknown")}

    def get_chain_depth(self, seq_id: int) -> int:
        """Return the chain depth for a seq_id (0 if unknown)."""
        return self._chain_depth.get(seq_id, 0)

    def get_proposal_metadata(self, seq_id: int) -> dict[str, Any] | None:
        """Return current proposal metadata for a seq_id, or None."""
        if seq_id not in self._latest_proposal_id:
            return None
        return {
            "seq_id": seq_id,
            "proposal_id": self._latest_proposal_id.get(seq_id, ""),
            "parent_proposal_id": self._latest_parent_proposal_id.get(seq_id, ""),
            "parent_kind": self._latest_parent_kind.get(seq_id, ""),
            "state": self._latest_state.get(seq_id, ""),
            "step_id": self._latest_step_id.get(seq_id, 0),
            "plan_id": self._latest_plan_id.get(seq_id, 0),
            "base_len": self._latest_base_len.get(seq_id, 0),
            "chain_depth": self._chain_depth.get(seq_id, 0),
            "promotion_reason": self._promotion_reason.get(seq_id, ""),
            "discard_reason": self._discard_reason.get(seq_id, ""),
            "source_step_id": self._source_step_id.get(seq_id, 0),
            "source_plan_id": self._source_plan_id.get(seq_id, 0),
        }

    def to_summary(self) -> dict[str, Any]:
        """Return a summary dict for trace/diagnostic output."""
        return {
            "total_entries": len(self._latest_proposal_id),
            "ready_count": len(self._ready_seq_ids),
            "pending_count": len(self._pending_seq_ids),
            "discarded_count": len(self._discarded_seq_ids),
            "chain_depth_by_seq": {
                str(k): v for k, v in sorted(self._chain_depth.items())
            },
            "state_by_seq": {
                str(k): self._latest_state.get(k, "")
                for k in sorted(self._latest_proposal_id)
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_history(self, seq_id: int, event: str) -> None:
        entry = {
            "event": event,
            "proposal_id": self._latest_proposal_id.get(seq_id, ""),
            "state": self._latest_state.get(seq_id, ""),
            "chain_depth": self._chain_depth.get(seq_id, 0),
        }
        self._history.setdefault(seq_id, []).append(entry)
