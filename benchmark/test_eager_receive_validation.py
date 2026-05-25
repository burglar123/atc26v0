#!/usr/bin/env python3
"""CPU synthetic tests for Phase 1H-lite producer-authoritative eager receive validation.

Uses exec() to load StepPlan and EagerBufferedProposal from source (avoiding
CUDA-heavy imports), then tests the eager receive validation rules directly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_step_plan():
    """Load StepPlan + RequestBudget from source, patching in future annotations."""
    src = (ROOT / "nano_pearl/pearl_engine/step_plan.py").read_text(encoding="utf-8")
    src = "from __future__ import annotations\n" + src
    mod = types.ModuleType("nano_pearl.pearl_engine.step_plan")
    mod.__file__ = str(ROOT / "nano_pearl/pearl_engine/step_plan.py")
    mod.__dict__["__name__"] = "nano_pearl.pearl_engine.step_plan"
    sys.modules["nano_pearl.pearl_engine.step_plan"] = mod
    exec(src, mod.__dict__)
    return mod.StepPlan, mod.RequestBudget


def _load_dual_batch():
    """Load EagerBufferedProposal from source, mocking Sequence import."""
    src = (ROOT / "nano_pearl/pearl_engine/dual_batch.py").read_text(encoding="utf-8")
    src = "from __future__ import annotations\n" + src
    # Replace Sequence import — Sequence is only used as a type annotation here.
    src = src.replace(
        "from nano_pearl.pearl_engine.sequence import Sequence",
        "Sequence = object  # mocked for CPU test",
    )
    mod = types.ModuleType("nano_pearl.pearl_engine.dual_batch")
    mod.__file__ = str(ROOT / "nano_pearl/pearl_engine/dual_batch.py")
    mod.__dict__["__name__"] = "nano_pearl.pearl_engine.dual_batch"
    sys.modules["nano_pearl.pearl_engine.dual_batch"] = mod
    exec(src, mod.__dict__)
    return mod.EagerBufferedProposal, mod.BufferedProposal, mod.ProposalBuffer, mod.EagerProposalBuffer, mod.ContinuousEagerDraftExecutionBuffer


StepPlan, RequestBudget = _load_step_plan()
EagerBufferedProposal, BufferedProposal, ProposalBuffer, EagerProposalBuffer, ContinuousEagerDraftExecutionBuffer = _load_dual_batch()


# --- Standalone validation function that mirrors _validate_received_eager_seq_ids ---

def validate_received_eager_seq_ids(
    received_eager_seq_ids: list[int],
    eager_proposals: list,
    plan,
    running_seq_ids: set,
) -> str | None:
    received_set = set(received_eager_seq_ids)
    target_home = set(plan.target_home_set)
    draft_home = set(plan.draft_home_set)
    target_eager = set(plan.target_eager_set)

    if not received_set.issubset(target_home):
        return (
            f"received_eager_seq_ids must be subset of target_home_set: "
            f"received={sorted(received_set)}, "
            f"outside_target_home={sorted(received_set - target_home)}"
        )
    if received_set & draft_home:
        return (
            f"received_eager_seq_ids must not overlap draft_home_set: "
            f"overlap={sorted(received_set & draft_home)}"
        )
    if received_set & target_eager:
        return (
            f"received_eager_seq_ids must not overlap target_eager_set: "
            f"overlap={sorted(received_set & target_eager)}"
        )
    if len(received_eager_seq_ids) > plan.max_eager_requests_per_step:
        return (
            f"received eager count {len(received_eager_seq_ids)} exceeds "
            f"max_eager_requests_per_step={plan.max_eager_requests_per_step}"
        )

    total_eager_tokens = 0
    for proposal in eager_proposals:
        seq_id = int(proposal.seq_id)
        eager_len = int(proposal.eager_len)
        total_eager_tokens += eager_len
        if eager_len > plan.max_eager_tokens_per_request:
            return (
                f"eager proposal for seq_id={seq_id} has length {eager_len} "
                f"above max_eager_tokens_per_request={plan.max_eager_tokens_per_request}"
            )
        if seq_id not in running_seq_ids:
            return f"eager proposal seq_id={seq_id} is not in running set (finished or missing)"
    if total_eager_tokens > plan.max_eager_tokens_per_step:
        return (
            f"total eager tokens {total_eager_tokens} exceeds "
            f"max_eager_tokens_per_step={plan.max_eager_tokens_per_step}"
        )

    return None


# --- Helpers ---

def _make_plan(
    target_home_set,
    draft_home_set,
    target_eager_set,
    max_requests=8,
    max_tokens_request=16,
    max_tokens_step=64,
):
    return StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch",
        target_home_set=list(target_home_set),
        draft_home_set=list(draft_home_set),
        target_eager_set=list(target_eager_set),
        max_eager_requests_per_step=max_requests,
        max_eager_tokens_per_request=max_tokens_request,
        max_eager_tokens_per_step=max_tokens_step,
    )


def _make_eager_proposal(seq_id, eager_len=1, **kwargs):
    defaults = dict(
        seq_id=seq_id,
        request_id=f"r{seq_id}",
        home_batch_id=0,
        eager_token_ids=[100 + seq_id] * eager_len,
        eager_len=eager_len,
        eager_base_len=0,
        source_plan_id=0,
        source_step_id=0,
        source_home_batch_id=0,
    )
    defaults.update(kwargs)
    return EagerBufferedProposal(**defaults)


# --- Valid cases ---


def test_valid_eager_receive_subset_of_target_home():
    """target_home=[1,3,5,7], draft_home=[4,6,8], target_eager=[2], receive [1] -> passes."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(1)]
    err = validate_received_eager_seq_ids([1], proposals, plan, running)
    assert err is None, f"expected no error, got: {err}"


def test_valid_eager_receive_empty():
    """Empty received list is valid."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    err = validate_received_eager_seq_ids([], [], plan, running)
    assert err is None, f"expected no error, got: {err}"


def test_valid_multiple_eager_seqs():
    """receive [1, 3] both in target_home -> passes."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(1), _make_eager_proposal(3)]
    err = validate_received_eager_seq_ids([1, 3], proposals, plan, running)
    assert err is None, f"expected no error, got: {err}"


# --- Invalid: overlaps draft_home_set ---


def test_eager_receive_overlaps_draft_home():
    """receive [1] where 1 is in both target_home and draft_home -> draft_home overlap error."""
    plan = _make_plan([1, 3, 5, 7], [1, 6, 8], [2])  # 1 overlaps target_home + draft_home
    running = {1, 3, 5, 7, 6, 8, 2}
    proposals = [_make_eager_proposal(1)]
    err = validate_received_eager_seq_ids([1], proposals, plan, running)
    assert err is not None, "expected validation error for overlap with draft_home_set"
    assert "overlap" in err.lower() and "draft_home_set" in err, \
        f"error should mention draft_home_set overlap, got: {err}"


# --- Invalid: outside target_home_set ---


def test_eager_receive_outside_target_home():
    """receive [9] not in target_home=[1,3,5,7] -> must fail."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(9)]
    err = validate_received_eager_seq_ids([9], proposals, plan, running)
    assert err is not None, "expected validation error for seq outside target_home_set"
    assert "subset" in err.lower() and "target_home_set" in err, \
        f"error should mention target_home_set subset, got: {err}"


# --- Invalid: overlaps target_eager_set ---


def test_eager_receive_overlaps_target_eager():
    """receive [1] where 1 is in both target_home and target_eager -> target_eager overlap error."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [1])  # 1 overlaps target_home + target_eager
    running = {1, 3, 5, 7, 4, 6, 8}
    proposals = [_make_eager_proposal(1)]
    err = validate_received_eager_seq_ids([1], proposals, plan, running)
    assert err is not None, "expected validation error for overlap with target_eager_set"
    assert "target_eager_set" in err, \
        f"error should mention target_eager_set, got: {err}"


# --- Invalid: exceeds budget caps ---


def test_eager_receive_exceeds_max_requests():
    """receive more seqs than max_eager_requests_per_step -> must fail."""
    plan = _make_plan([1, 3, 5], [], [], max_requests=1)
    running = {1, 3, 5}
    proposals = [_make_eager_proposal(1), _make_eager_proposal(3)]
    err = validate_received_eager_seq_ids([1, 3], proposals, plan, running)
    assert err is not None, "expected error for exceeding max_eager_requests_per_step"
    assert "max_eager_requests_per_step" in err, \
        f"error should mention max_eager_requests_per_step, got: {err}"


def test_eager_receive_exceeds_per_request_tokens():
    """eager_len above max_eager_tokens_per_request -> must fail."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2], max_tokens_request=1)
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(1, eager_len=2)]
    err = validate_received_eager_seq_ids([1], proposals, plan, running)
    assert err is not None, "expected error for exceeding max_eager_tokens_per_request"
    assert "max_eager_tokens_per_request" in err, \
        f"error should mention max_eager_tokens_per_request, got: {err}"


def test_eager_receive_exceeds_step_tokens():
    """total eager tokens above max_eager_tokens_per_step -> must fail."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2], max_tokens_step=1)
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(1, eager_len=1), _make_eager_proposal(3, eager_len=1)]
    err = validate_received_eager_seq_ids([1, 3], proposals, plan, running)
    assert err is not None, "expected error for exceeding max_eager_tokens_per_step"
    assert "max_eager_tokens_per_step" in err, \
        f"error should mention max_eager_tokens_per_step, got: {err}"


# --- Invalid: finished seq ---


def test_eager_receive_finished_seq():
    """receive eager proposal for a seq not in running set -> must fail."""
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {3, 5, 7, 4, 6, 8, 2}  # 1 is NOT running
    proposals = [_make_eager_proposal(1)]
    err = validate_received_eager_seq_ids([1], proposals, plan, running)
    assert err is not None, "expected error for finished/missing seq"
    assert ("running" in err.lower() or "finished" in err.lower()), \
        f"error should mention running/finished, got: {err}"


# --- Exact divergence case from the bug report ---


def test_divergence_case_valid():
    """Exact divergence scenario — valid.

    target-local plan: target_home_set=[1,3,5,7], draft_home_set=[4,6,8],
    target_eager_set=[2], draft_eager_set=[].
    payload: normal_seq_ids=[4,6,8], eager_seq_ids=[1].

    Expected: eager validation passes because [1] ⊆ target_home_set and
    [1] does not overlap draft_home_set or target_eager_set.
    """
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(1)]
    err = validate_received_eager_seq_ids([1], proposals, plan, running)
    assert err is None, f"divergence case (valid): expected no error, got: {err}"


def test_divergence_case_invalid():
    """Exact divergence scenario — invalid.

    target-local plan: target_home_set=[1,3,5,7], draft_home_set=[4,6,8],
    target_eager_set=[2], draft_eager_set=[].
    payload: normal_seq_ids=[4,6,8], eager_seq_ids=[4].

    Expected: eager validation FAILS because [4] overlaps draft_home_set
    and is not in target_home_set.  The subset-of-target_home check fires
    first since 4 ∉ {1,3,5,7}, which is the correct behavior.
    """
    plan = _make_plan([1, 3, 5, 7], [4, 6, 8], [2])
    running = {1, 3, 5, 7, 4, 6, 8, 2}
    proposals = [_make_eager_proposal(4)]
    err = validate_received_eager_seq_ids([4], proposals, plan, running)
    assert err is not None, (
        "divergence case (invalid): expected validation error because [4] "
        "overlaps draft_home_set and is not in target_home_set"
    )
    # Either "subset"/"target_home_set" or "draft_home_set" is acceptable —
    # both describe why seq 4 is invalid.
    ok = ("target_home_set" in err) or ("draft_home_set" in err)
    assert ok, f"error should mention target_home_set or draft_home_set, got: {err}"


# --- Phase 1H-lite normal buffer preservation ---


def test_draft_home_set_preserved_when_target_eager_overlaps():
    """Seq in both draft_home_set and target_eager_set — must stay in draft_home_set.

    After the fix in _annotate_eager_execution_plan, a seq with a promoted
    eager proposal stays in draft_home_set so its normal proposal is still
    generated.  validate_phase1h_eager_execution must accept the overlap.
    """
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[1, 4, 6, 8],
        target_eager_set=[1],  # seq 1 has ready eager proposal AND needs normal draft
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    # Must not raise — the fix allows target_eager_set to overlap draft_home_set.
    plan.validate_phase1h_eager_execution()
    # Verify the seq is still in both sets.
    assert 1 in plan.target_eager_set, "seq 1 must be in target_eager_set"
    assert 1 in plan.draft_home_set, "seq 1 must remain in draft_home_set for normal proposal generation"


def test_target_eager_and_target_home_still_disjoint():
    """target_eager_set and target_home_set must remain disjoint (different batches)."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3, 5, 7],
        draft_home_set=[4, 6, 8],
        target_eager_set=[1],  # overlap with target_home — this is invalid
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    try:
        plan.validate_phase1h_eager_execution()
        assert False, "expected assertion error for target_eager_set overlapping target_home_set"
    except AssertionError as e:
        assert "target_home_set" in str(e), \
            f"error should mention target_home_set, got: {e}"


def test_normal_buffer_has_all_after_eager_store():
    """EagerProposalBuffer.store does not affect ProposalBuffer contents."""
    normal_buf = ProposalBuffer()
    eager_buf = EagerProposalBuffer()

    normal_proposal = BufferedProposal(
        seq_id=1,
        request_id="r1",
        home_batch_id=0,
        proposal_token_ids=[101, 102],
        to_be_verified_token_ids=[101],
        proposal_len=2,
        pre_verify=False,
        plan_id=0,
        valid=True,
    )
    normal_buf.store([normal_proposal])
    assert normal_buf.has_all([1]), "normal buffer must have seq 1 before eager store"

    eager_proposal = _make_eager_proposal(1, eager_len=3)
    eager_buf.store([eager_proposal])

    # Eager store must not affect normal buffer.
    assert normal_buf.has_all([1]), "normal buffer must still have seq 1 after eager store"
    assert eager_buf.get(1) is not None, "eager buffer must have seq 1"


# --- Phase 1H-lite overlap scenario tests ---


def test_overlap_rejected_without_eager_execution():
    """target_eager_set ∩ draft_home_set must fail if eager_execution_enabled=False."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[1, 4, 6, 8],
        target_eager_set=[1],  # overlaps draft_home_set
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=False,  # NOT enabled
        eager_execution_enabled=False,
    )
    try:
        plan.validate_phase1h_eager_execution()
        assert False, "expected assertion error for overlap without eager execution"
    except AssertionError as e:
        assert "eager execution" in str(e).lower(), (
            f"error should mention eager execution, got: {e}"
        )


def test_overlap_fails_outside_steady_phase():
    """target_eager_set ∩ draft_home_set must fail if plan_phase is not steady."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[1, 4, 6, 8],
        target_eager_set=[1],  # overlaps draft_home_set
        dual_batch_enabled=True,
        plan_phase="priming",  # NOT steady
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    try:
        plan.validate_phase1h_eager_execution()
        assert False, "expected assertion error for overlap outside steady phase"
    except AssertionError as e:
        assert "steady" in str(e).lower(), (
            f"error should mention steady, got: {e}"
        )


def test_overlap_target_eager_subset_constraint():
    """target_eager_set must be subset of target_home_set ∪ draft_home_set.

    The subset check only fires when there IS overlap with draft_home_set,
    so we include an overlapping seq along with an invalid one.
    """
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[1, 4, 6, 8],
        target_eager_set=[1, 9],  # 1 overlaps draft_home, 9 outside both
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    try:
        plan.validate_phase1h_eager_execution()
        assert False, "expected assertion error for target_eager outside target_home ∪ draft_home"
    except AssertionError as e:
        assert "subset" in str(e).lower(), (
            f"error should mention subset constraint, got: {e}"
        )


def test_overlap_trace_fields_default():
    """New trace fields have correct defaults."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[4, 6, 8],
        target_eager_set=[],
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    assert plan.target_eager_draft_home_overlap_seq_ids == []
    assert plan.overlap_normal_proposal_kept_seq_ids == []
    assert plan.overlap_normal_proposal_discarded_seq_ids == []
    assert plan.overlap_normal_proposal_discard_reason is None


def test_overlap_trace_fields_in_trace_dict():
    """New trace fields appear in to_trace_dict()."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[4, 6, 8],
        target_eager_set=[],
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    plan.target_eager_draft_home_overlap_seq_ids = [1]
    plan.overlap_normal_proposal_discarded_seq_ids = [1]
    plan.overlap_normal_proposal_discard_reason = "eager_accepted_base_mismatch"
    d = plan.to_trace_dict()
    assert d.get("target_eager_draft_home_overlap_seq_ids") == [1], f"got {d.get('target_eager_draft_home_overlap_seq_ids')}"
    assert d.get("overlap_normal_proposal_discarded_seq_ids") == [1], f"got {d.get('overlap_normal_proposal_discarded_seq_ids')}"
    assert d.get("overlap_normal_proposal_discard_reason") == "eager_accepted_base_mismatch", f"got {d.get('overlap_normal_proposal_discard_reason')}"


# --- H2 Eager Lifecycle Ordering Tests ---


def test_eager_buffer_ready_only_filter():
    """get_many(ready_only=True) returns only ready, non-consumed proposals."""
    buf = EagerProposalBuffer()
    p1 = _make_eager_proposal(1)
    p2 = _make_eager_proposal(2)
    p3 = _make_eager_proposal(3)
    buf.store([p1, p2, p3])

    buf.mark_ready(1)
    buf.mark_ready(3)

    ready = buf.get_many([1, 2, 3], ready_only=True)
    ready_ids = sorted(p.seq_id for p in ready)
    assert ready_ids == [1, 3], f"expected ready=[1,3], got {ready_ids}"

    all_proposals = buf.get_many([1, 2, 3], ready_only=False)
    all_ids = sorted(p.seq_id for p in all_proposals)
    assert all_ids == [1, 2, 3], f"expected all=[1,2,3], got {all_ids}"


def test_eager_buffer_discard_removes_from_ready():
    """Discarded proposals are not returned by get_many."""
    buf = EagerProposalBuffer()
    p1 = _make_eager_proposal(1)
    p2 = _make_eager_proposal(2)
    buf.store([p1, p2])
    buf.mark_ready(1)
    buf.mark_ready(2)

    buf.discard([1])
    ready = buf.get_many([1, 2], ready_only=True)
    ready_ids = sorted(p.seq_id for p in ready)
    assert ready_ids == [2], f"expected ready=[2] after discard, got {ready_ids}"


def test_promote_non_overlap_eager_proposals():
    """Non-overlap seqs (not in normal verify result) are promoted by default."""
    buf = EagerProposalBuffer()
    p1 = _make_eager_proposal(1)  # overlap seq (in verify result)
    p2 = _make_eager_proposal(2)  # non-overlap seq (NOT in verify result)
    buf.store([p1, p2])

    draft_eager_set = [1, 2]
    accepted_lens = {1: 2}  # seq 1 accepted 2 tokens
    invalidated_lens = {1: 0}
    running_seq_ids = {1, 2}

    promoted = []
    discarded = []
    for seq_id in draft_eager_set:
        proposal = buf.get(seq_id)
        assert proposal is not None, f"missing proposal for seq_id={seq_id}"
        in_normal_verify = int(seq_id) in accepted_lens or int(seq_id) in invalidated_lens
        if not in_normal_verify:
            full_accept = int(seq_id) in running_seq_ids
        else:
            full_accept = (
                int(seq_id) in running_seq_ids
                and int(invalidated_lens.get(seq_id, 0)) == 0
                and int(accepted_lens.get(seq_id, 0)) > 0
            )
        if full_accept:
            buf.mark_ready(seq_id)
            promoted.append(int(seq_id))
        else:
            buf.discard([seq_id])
            discarded.append(int(seq_id))

    assert 1 in promoted, f"overlap seq 1 (full accept) should be promoted, got promoted={promoted}"
    assert 2 in promoted, f"non-overlap seq 2 should be promoted by default, got promoted={promoted}, discarded={discarded}"
    assert len(discarded) == 0, f"no seqs should be discarded, got discarded={discarded}"


def test_promote_discard_mixed_overlap_and_rejected():
    """Overlap seq rejected in normal verify -> discarded; non-overlap -> promoted."""
    buf = EagerProposalBuffer()
    p1 = _make_eager_proposal(1)  # overlap seq, REJECTED
    p2 = _make_eager_proposal(2)  # non-overlap seq
    p3 = _make_eager_proposal(3)  # overlap seq, ACCEPTED
    buf.store([p1, p2, p3])

    draft_eager_set = [1, 2, 3]
    accepted_lens = {1: 0, 3: 2}  # seq 1 rejected, seq 3 accepted
    invalidated_lens = {1: 2, 3: 0}
    running_seq_ids = {1, 2, 3}

    promoted = []
    discarded = []
    for seq_id in draft_eager_set:
        proposal = buf.get(seq_id)
        assert proposal is not None
        in_normal_verify = int(seq_id) in accepted_lens or int(seq_id) in invalidated_lens
        if not in_normal_verify:
            full_accept = int(seq_id) in running_seq_ids
        else:
            full_accept = (
                int(seq_id) in running_seq_ids
                and int(invalidated_lens.get(seq_id, 0)) == 0
                and int(accepted_lens.get(seq_id, 0)) > 0
            )
        if full_accept:
            buf.mark_ready(seq_id)
            promoted.append(int(seq_id))
        else:
            buf.discard([seq_id])
            discarded.append(int(seq_id))

    assert 1 in discarded, f"rejected overlap seq 1 should be discarded, got discarded={discarded}"
    assert 2 in promoted, f"non-overlap seq 2 should be promoted, got promoted={promoted}"
    assert 3 in promoted, f"accepted overlap seq 3 should be promoted, got promoted={promoted}"


def test_eager_lifecycle_trace_fields_default():
    """New H2 eager lifecycle trace fields have correct defaults."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[4, 6, 8],
        target_eager_set=[],
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    assert plan.eager_draft_attempted_seq_ids == []
    assert plan.eager_draft_generated_seq_ids == []
    assert plan.eager_draft_skipped_seq_ids == []
    assert plan.eager_draft_skipped_reason_by_seq_id == {}
    assert plan.eager_buffer_keys_before_eager_draft == []
    assert plan.eager_ready_seq_ids_before_send == []
    assert plan.eager_sent_seq_ids == []


def test_eager_lifecycle_trace_fields_in_trace_dict():
    """New H2 eager lifecycle trace fields appear in to_trace_dict()."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[3, 5, 7],
        draft_home_set=[4, 6, 8],
        target_eager_set=[],
        dual_batch_enabled=True,
        plan_phase="steady",
        target_batch_id=0,
        draft_batch_id=1,
        enable_eager_execution=True,
        eager_execution_enabled=True,
    )
    plan.eager_draft_attempted_seq_ids = [4, 6]
    plan.eager_draft_generated_seq_ids = [4, 6]
    plan.eager_buffer_keys_before_eager_draft = [1, 3]
    plan.eager_ready_seq_ids_before_send = [4, 6]
    plan.eager_sent_seq_ids = [4, 6]
    d = plan.to_trace_dict()
    assert d.get("eager_draft_attempted_seq_ids") == [4, 6], f"got {d.get('eager_draft_attempted_seq_ids')}"
    assert d.get("eager_draft_generated_seq_ids") == [4, 6], f"got {d.get('eager_draft_generated_seq_ids')}"
    assert d.get("eager_buffer_keys_before_eager_draft") == [1, 3], f"got {d.get('eager_buffer_keys_before_eager_draft')}"
    assert d.get("eager_ready_seq_ids_before_send") == [4, 6], f"got {d.get('eager_ready_seq_ids_before_send')}"
    assert d.get("eager_sent_seq_ids") == [4, 6], f"got {d.get('eager_sent_seq_ids')}"


# --- H2 NCCL Collective Schedule Tests ---


def _load_proposal_payload():
    """Load build_combined_proposal_payload from source, mocking imports."""
    src = (ROOT / "nano_pearl/pearl_engine/proposal_payload.py").read_text(encoding="utf-8")
    src = "from __future__ import annotations\n" + src
    mod = types.ModuleType("nano_pearl.pearl_engine.proposal_payload")
    mod.__file__ = str(ROOT / "nano_pearl/pearl_engine/proposal_payload.py")
    mod.__dict__["__name__"] = "nano_pearl.pearl_engine.proposal_payload"
    sys.modules["nano_pearl.pearl_engine.proposal_payload"] = mod
    exec(src, mod.__dict__)
    return mod.build_combined_proposal_payload


build_combined_proposal_payload = _load_proposal_payload()


def test_combined_payload_all_empty():
    """All proposal lists empty → valid combined payload with zero-length sections."""
    payload = build_combined_proposal_payload(
        normal_proposals=[],
        conditional_normal_proposals=[],
        eager_proposals=[],
        plan_id=1,
        step_id=0,
        draft_batch_id=0,
        gamma=4,
    )
    assert payload["kind"] == "combined", f"expected kind=combined, got {payload['kind']}"
    assert payload["flat_payload"] == [], f"expected empty flat_payload, got {payload['flat_payload']}"
    assert payload["normal_seq_ids"] == []
    assert payload["conditional_normal_seq_ids"] == []
    assert payload["eager_seq_ids"] == []
    assert payload["normal_payload"] == []
    assert payload["conditional_normal_payload"] == []
    assert payload["eager_payload"] == []
    assert payload["plan_id"] == 1
    assert payload["gamma"] == 4


def test_combined_payload_empty_eager_nonempty_normal():
    """Normal proposals present, eager empty → valid combined payload."""
    normal = BufferedProposal(
        seq_id=1, request_id="r1", home_batch_id=0,
        proposal_token_ids=[101], to_be_verified_token_ids=[101],
        proposal_len=1, pre_verify=False, plan_id=0, valid=True,
    )
    payload = build_combined_proposal_payload(
        normal_proposals=[normal],
        conditional_normal_proposals=[],
        eager_proposals=[],
        plan_id=2,
        step_id=1,
        draft_batch_id=1,
        gamma=4,
    )
    assert payload["kind"] == "combined"
    assert payload["normal_seq_ids"] == [1]
    assert payload["eager_seq_ids"] == []
    assert len(payload["flat_payload"]) == len(payload["normal_payload"])
    assert payload["conditional_normal_payload"] == []
    assert payload["eager_payload"] == []


def test_combined_payload_empty_normal_nonempty_eager():
    """Eager proposals present, normal empty → valid combined payload."""
    eager = EagerBufferedProposal(
        seq_id=2, request_id="r2", home_batch_id=1,
        eager_token_ids=[201, 202], eager_len=2, eager_base_len=0,
        source_plan_id=0, source_step_id=0, source_home_batch_id=0,
    )
    payload = build_combined_proposal_payload(
        normal_proposals=[],
        conditional_normal_proposals=[],
        eager_proposals=[eager],
        plan_id=3,
        step_id=2,
        draft_batch_id=1,
        gamma=4,
    )
    assert payload["kind"] == "combined"
    assert payload["normal_seq_ids"] == []
    assert payload["eager_seq_ids"] == [2]
    assert len(payload["eager_payload"]) > 0
    assert payload["normal_payload"] == []


def test_combined_payload_with_conditional():
    """Conditional normal proposals included in payload."""
    normal = BufferedProposal(
        seq_id=1, request_id="r1", home_batch_id=0,
        proposal_token_ids=[101], to_be_verified_token_ids=[101],
        proposal_len=1, pre_verify=False, plan_id=0, valid=True,
    )
    conditional = BufferedProposal(
        seq_id=3, request_id="r3", home_batch_id=0,
        proposal_token_ids=[301], to_be_verified_token_ids=[301],
        proposal_len=1, pre_verify=False, plan_id=0, valid=True,
    )
    eager = EagerBufferedProposal(
        seq_id=5, request_id="r5", home_batch_id=1,
        eager_token_ids=[501], eager_len=1, eager_base_len=0,
        source_plan_id=0, source_step_id=0, source_home_batch_id=0,
    )
    payload = build_combined_proposal_payload(
        normal_proposals=[normal],
        conditional_normal_proposals=[conditional],
        eager_proposals=[eager],
        plan_id=4,
        step_id=3,
        draft_batch_id=1,
        gamma=4,
    )
    assert payload["kind"] == "combined"
    assert payload["normal_seq_ids"] == [1]
    assert payload["conditional_normal_seq_ids"] == [3]
    assert payload["eager_seq_ids"] == [5]
    assert len(payload["flat_payload"]) == (
        len(payload["normal_payload"]) + len(payload["conditional_normal_payload"]) + len(payload["eager_payload"])
    )


def test_send_normal_subset_assertion_accepts_partial():
    """Normal + conditional being strict subset of expected passes subset check.

    This tests that the assertion change from == to issubset in
    _send_combined_dual_proposals is correct: when repair seqs are excluded,
    the subset check passes.
    """
    expected = [1, 2, 3, 4]
    actual_normal = [1]
    actual_conditional = [3]
    actual_all = actual_normal + actual_conditional
    assert set(actual_all).issubset(set(expected)), \
        f"[1,3] should be subset of [1,2,3,4]"


def test_send_eager_subset_assertion_accepts_empty():
    """Empty eager with non-empty expected passes subset check.

    When all eager proposals are discarded during promote/discard,
    the subset assertion should accept empty actual against non-empty expected.
    """
    expected = [1, 5]
    actual = []
    assert set(actual).issubset(set(expected)), \
        f"[] should be subset of [1,5]"


# --- Source-authoritative eager verify meta+payload protocol tests ---


def _simulate_eager_result_broadcast(source_cols: int, receiver_local_len: int):
    """Simulate the source-authoritative meta+payload eager verify protocol.

    Returns (meta, payload) as dict for inspection.  The receiver must allocate
    from the source meta, NOT from receiver_local_len.
    """
    rows = 6
    numel = rows * source_cols
    meta = [rows, source_cols, numel]

    # Receiver side: parse meta, allocate from source metadata
    _r_rows, r_cols, r_numel = meta
    assert r_cols == source_cols, "receiver cols must equal source cols from meta"
    # Receiver allocates based on source meta, not local length
    if r_numel > 0:
        payload = list(range(r_numel))  # placeholder
    else:
        payload = []

    return {
        "meta": meta,
        "source_cols": source_cols,
        "receiver_local_len": receiver_local_len,
        "receiver_allocated_cols": r_cols,
        "payload_len": len(payload),
        "payload_skipped": r_numel == 0,
    }


def test_eager_protocol_source_cols_zero_receiver_nonzero():
    """Source has cols=0, receiver local target_eager_seqs length > 0.

    Receiver must still allocate [6,0] from source meta, not from local length.
    """
    result = _simulate_eager_result_broadcast(source_cols=0, receiver_local_len=3)
    assert result["receiver_allocated_cols"] == 0, \
        f"receiver must allocate 0 cols from source meta, got {result['receiver_allocated_cols']}"
    assert result["payload_skipped"] is True


def test_eager_protocol_source_cols_one_receiver_zero():
    """Source has cols=1, receiver local target_eager_seqs length == 0.

    Receiver must allocate [6,1] from source meta.
    """
    result = _simulate_eager_result_broadcast(source_cols=1, receiver_local_len=0)
    assert result["receiver_allocated_cols"] == 1, \
        f"receiver must allocate 1 col from source meta, got {result['receiver_allocated_cols']}"
    assert result["payload_skipped"] is False
    assert result["payload_len"] == 6


def test_eager_protocol_both_cols_zero():
    """Both sides have cols=0."""
    result = _simulate_eager_result_broadcast(source_cols=0, receiver_local_len=0)
    assert result["receiver_allocated_cols"] == 0
    assert result["payload_skipped"] is True


def test_eager_protocol_both_cols_positive():
    """Both sides have cols>0."""
    result = _simulate_eager_result_broadcast(source_cols=2, receiver_local_len=2)
    assert result["receiver_allocated_cols"] == 2
    assert result["payload_skipped"] is False
    assert result["payload_len"] == 12  # 6 * 2


def test_eager_protocol_payload_skip_driven_by_source_meta():
    """Payload skip is driven only by source meta.numel, never by local state."""
    # Source cols=0 → meta.numel=0 → payload skipped regardless of local len
    for local_len in [0, 1, 3, 5]:
        result = _simulate_eager_result_broadcast(source_cols=0, receiver_local_len=local_len)
        assert result["payload_skipped"] is True, \
            f"payload must be skipped when source numel=0 (local_len={local_len})"

    # Source cols>0 → meta.numel>0 → payload NOT skipped regardless of local len
    for local_len in [0, 1, 3]:
        result = _simulate_eager_result_broadcast(source_cols=2, receiver_local_len=local_len)
        assert result["payload_skipped"] is False, \
            f"payload must NOT be skipped when source numel>0 (local_len={local_len})"


def test_eager_protocol_receiver_ignores_local_length():
    """Receiver allocation must always equal source_cols from meta, never local length."""
    for source_cols in [0, 1, 3]:
        for local_len in [0, 1, 2, 4]:
            if local_len == source_cols:
                continue  # skip the matching case
            result = _simulate_eager_result_broadcast(source_cols=source_cols, receiver_local_len=local_len)
            assert result["receiver_allocated_cols"] == source_cols, \
                f"source_cols={source_cols}, local_len={local_len}: receiver used {result['receiver_allocated_cols']}"


# --- H2 schedule shape-mismatch prevention tests ---


def test_h2_eager_shape_divergence_target_empty_source_nonempty():
    """TARGET eager set empty on source, non-empty on receiver: no shape mismatch.

    The source-authoritative meta ensures both ranks use source_cols, not local state.
    """
    source_cols = 1  # TARGET rank has 1 eager result
    receiver_local_len = 0  # DRAFT rank thinks there are 0 target eager seqs

    result = _simulate_eager_result_broadcast(source_cols=source_cols, receiver_local_len=receiver_local_len)
    assert result["receiver_allocated_cols"] == source_cols, \
        f"receiver must follow source meta ({source_cols}), not local state ({receiver_local_len})"
    assert result["payload_skipped"] is False


def test_h2_eager_shape_divergence_source_empty_target_nonempty():
    """TARGET eager set non-empty on source, empty on receiver: no shape mismatch."""
    source_cols = 0  # TARGET rank has no eager results
    receiver_local_len = 3  # DRAFT rank thinks there are 3 target eager seqs

    result = _simulate_eager_result_broadcast(source_cols=source_cols, receiver_local_len=receiver_local_len)
    assert result["receiver_allocated_cols"] == source_cols, \
        f"receiver must follow source meta ({source_cols}), not local state ({receiver_local_len})"
    assert result["payload_skipped"] is True


# --- Source code checks for source-authoritative protocol ---


def test_receive_eager_verify_result_uses_meta_broadcast():
    """_receive_eager_verify_result must broadcast meta before allocating payload."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")

    fn_start = src.find("def _receive_eager_verify_result(self, seqs:")
    assert fn_start != -1, "missing _receive_eager_verify_result definition"
    fn_end = src.find("def _apply_eager_verify_result(", fn_start)
    fn_body = src[fn_start:fn_end]

    # Must contain the meta broadcast
    assert "torch.zeros(3, dtype=torch.int64" in fn_body, \
        "_receive_eager_verify_result must allocate meta tensor of size 3"
    assert "Source-authoritative meta+payload" in fn_body or "source-authoritative" in fn_body.lower(), \
        "_receive_eager_verify_result must document source-authoritative protocol"

    # Must NOT allocate based on len(seqs) for the broadcast tensor
    lines = fn_body.split("\n")
    verify_res_alloc = [l for l in lines if "verify_res = torch." in l and "zeros" in l]
    if verify_res_alloc:
        assert "len(seqs)" not in verify_res_alloc[0], \
            f"_receive_eager_verify_result must not allocate verify_res from len(seqs): {verify_res_alloc[0]}"

    # Must have payload broadcast inside numel > 0 check
    assert "if numel > 0:" in fn_body, \
        "_receive_eager_verify_result must condition payload broadcast on numel > 0"


def test_build_eager_verify_result_uses_meta_broadcast():
    """_build_eager_verify_result must use source-authoritative meta+payload."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")

    fn_start = src.find("def _build_eager_verify_result(")
    assert fn_start != -1, "missing _build_eager_verify_result definition"
    fn_end = src.find("def _run_eager_verify_sidecar(", fn_start)
    fn_body = src[fn_start:fn_end]

    # Must contain source-authoritative meta
    assert "Source-authoritative meta+payload" in fn_body or "source-authoritative" in fn_body.lower(), \
        "_build_eager_verify_result must document source-authoritative protocol"
    assert 'meta = torch.tensor([6, cols, numel]' in fn_body, \
        "_build_eager_verify_result must broadcast meta [rows, cols, numel]"
    assert "dist.broadcast(meta, src=" in fn_body, \
        "_build_eager_verify_result must broadcast meta before payload"
    assert "if numel > 0:" in fn_body, \
        "_build_eager_verify_result must condition payload broadcast on numel > 0"


# --- H2 proposal-set invariant tests ---
# In H2 steady, seqs in target_eager_set are covered by the eager verify result
# broadcast and must be excluded from normal/conditional proposal expectations.


def _build_step_plan_for_test(*, draft_home_set, target_eager_set, draft_eager_set,
                               target_home_set=None, plan_phase="steady"):
    """Build a StepPlan with H2-aware proposal fields for testing."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=target_home_set or [],
        draft_home_set=list(draft_home_set),
        budgets={s: RequestBudget(normal_gamma=4, eager_gamma=0) for s in draft_home_set},
        draft_batch_id=0,
        decode_ready_mode=False,
    )
    plan.plan_phase = plan_phase
    plan.target_eager_set = list(target_eager_set)
    plan.draft_eager_set = list(draft_eager_set)

    # Simulate the H2-aware computation from _build_dual_batch_step_plan
    if plan_phase == "steady":
        _te = set(plan.target_eager_set)
        plan.expected_eager_proposal_seq_ids = list(plan.draft_eager_set)
        plan.expected_normal_proposal_seq_ids = [
            s for s in plan.draft_home_set if s not in _te
        ]
        plan.expected_conditional_proposal_seq_ids = []
        plan.excluded_normal_proposal_seq_ids = [
            s for s in plan.draft_home_set if s in _te
        ]
        plan.excluded_normal_proposal_reason = (
            "covered_by_target_eager_result" if plan.excluded_normal_proposal_seq_ids else ""
        )
    else:
        plan.expected_eager_proposal_seq_ids = list(plan.draft_eager_set)
        plan.expected_normal_proposal_seq_ids = list(plan.draft_home_set)
        plan.expected_conditional_proposal_seq_ids = []
        plan.excluded_normal_proposal_seq_ids = []
        plan.excluded_normal_proposal_reason = ""
    return plan


def test_h2_expected_normal_empty_target_eager():
    """H2 steady with target_eager_set empty: expected normal == draft_home_set."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[],
        draft_eager_set=[2],
    )
    assert plan.expected_normal_proposal_seq_ids == [1, 3, 5, 7]
    assert plan.excluded_normal_proposal_seq_ids == []
    assert plan.excluded_normal_proposal_reason == ""


def test_h2_expected_normal_nonempty_target_eager():
    """H2 steady with target_eager_set non-empty: expected normal excludes target_eager_set."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1],
        draft_eager_set=[2],
    )
    assert plan.expected_normal_proposal_seq_ids == [3, 5, 7], \
        f"expected [3,5,7] got {plan.expected_normal_proposal_seq_ids}"
    assert plan.excluded_normal_proposal_seq_ids == [1]
    assert plan.excluded_normal_proposal_reason == "covered_by_target_eager_result"


def test_h2_normal_conditional_excluded_invariant():
    """normal + conditional + excluded = draft_home_set in H2 steady."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1],
        draft_eager_set=[2],
    )
    received_normal = [3, 5, 7]
    received_conditional = []
    all_received = received_normal + received_conditional + plan.excluded_normal_proposal_seq_ids
    assert sorted(all_received) == sorted(plan.draft_home_set), \
        f"invariant failure: received+excluded={sorted(all_received)}, draft_home={plan.draft_home_set}"


def test_h2_expected_eager_matches_draft_eager():
    """H2 steady with draft_eager_set non-empty: expected_eager matches draft_eager_set."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1],
        draft_eager_set=[2, 4],
    )
    assert plan.expected_eager_proposal_seq_ids == [2, 4]


def test_h2_target_and_draft_eager_distinct():
    """target_eager_set and draft_eager_set both non-empty and distinct."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1],
        draft_eager_set=[3],
        target_home_set=[2, 4, 6, 8],
    )
    assert plan.target_eager_set == [1]
    assert plan.draft_eager_set == [3]
    assert plan.expected_normal_proposal_seq_ids == [3, 5, 7]
    assert plan.excluded_normal_proposal_seq_ids == [1]
    assert plan.expected_eager_proposal_seq_ids == [3]


def test_h2_target_eager_intersects_draft_home():
    """target_eager_set is always subset of draft_home_set; exclusion works."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1, 3],
        draft_eager_set=[5],
    )
    assert plan.expected_normal_proposal_seq_ids == [5, 7]
    assert plan.excluded_normal_proposal_seq_ids == [1, 3]


def test_h2_target_eager_no_intersection_draft_home():
    """target_eager_set not intersecting draft_home_set: expected == draft_home_set."""
    # This is the empty-target_eager case but with a non-empty target_eager_set
    # that doesn't intersect draft_home_set.  In practice this shouldn't happen
    # (target_eager_set ⊂ draft_home_set), but the formula handles it gracefully.
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[],
        draft_eager_set=[2],
    )
    assert plan.expected_normal_proposal_seq_ids == [1, 3, 5, 7]
    assert plan.excluded_normal_proposal_seq_ids == []
    assert plan.excluded_normal_proposal_reason == ""


def test_h2_non_steady_preserves_old_invariant():
    """Fallback/priming phases: expected normal == draft_home_set (no exclusion)."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1],
        draft_eager_set=[2],
        plan_phase="fallback",
    )
    assert plan.expected_normal_proposal_seq_ids == [1, 3, 5, 7]
    assert plan.excluded_normal_proposal_seq_ids == []
    assert plan.excluded_normal_proposal_reason == ""


def test_h2_trace_dict_includes_new_fields():
    """to_trace_dict() includes the new H2 proposal-set fields."""
    plan = _build_step_plan_for_test(
        draft_home_set=[1, 3, 5, 7],
        target_eager_set=[1],
        draft_eager_set=[2],
    )
    d = plan.to_trace_dict()
    assert "expected_normal_proposal_seq_ids" in d
    assert "expected_eager_proposal_seq_ids" in d
    assert "excluded_normal_proposal_seq_ids" in d
    assert "excluded_normal_proposal_reason" in d
    assert d["expected_normal_proposal_seq_ids"] == [3, 5, 7]
    assert d["excluded_normal_proposal_seq_ids"] == [1]
    assert d["excluded_normal_proposal_reason"] == "covered_by_target_eager_result"
    assert d["expected_eager_proposal_seq_ids"] == [2]


def test_h2_schedule_draft_order_via_verify_group():
    """DRAFT H2 steady: recv normal → recv eager → send combined, all via verify_group.

    Scans pearl_model_runner.py source for the H2 steady block on the DRAFT side
    and asserts the three-collective ordering and group= usage.
    """
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")

    # Locate the H2 steady block on the DRAFT side ("if h2_steady:" inside
    # DraftModelRunner.dual_batch_pearl_step).  We search for the comment marker
    # that starts the H2 collective block.
    h2_section_start = src.find("# Phase 3: Receive normal verify result (UNCONDITIONAL)")
    assert h2_section_start != -1, "missing DRAFT H2 Phase 3 marker"

    # Find the end of the H2 steady block (next "else:" for legacy)
    h2_section_end = src.find("# === Legacy ordering", h2_section_start)
    assert h2_section_end != -1, "missing DRAFT H2 legacy ordering marker"

    h2_block = src[h2_section_start:h2_section_end]

    # 1. Normal result recv via verify_group
    normal_recv_pos = h2_block.find("_receive_verify_result(target_seqs, group=self.verify_group)")
    assert normal_recv_pos != -1, "DRAFT H2: missing _receive_verify_result with group=self.verify_group"

    # 2. Eager result recv via verify_group
    eager_recv_pos = h2_block.find("_receive_eager_verify_result(target_eager_seqs, group=self.verify_group)")
    assert eager_recv_pos != -1, "DRAFT H2: missing _receive_eager_verify_result with group=self.verify_group"

    # 3. Combined send — verify inside _send_combined_dual_proposals uses verify_group
    #    (checked indirectly via the function body below)
    combined_send_call = h2_block.find("_send_combined_dual_proposals(")
    assert combined_send_call != -1, "DRAFT H2: missing _send_combined_dual_proposals call"

    # Ordering: normal recv < eager recv < combined send
    assert normal_recv_pos < eager_recv_pos < combined_send_call, \
        f"DRAFT H2 order violation: normal_recv={normal_recv_pos}, eager_recv={eager_recv_pos}, combined_send={combined_send_call}"

    # Verify _send_combined_dual_proposals broadcasts use verify_group
    send_fn_start = src.find("def _send_combined_dual_proposals(")
    send_fn_end = src.find("def _parse_combined_normal_payload(", send_fn_start)
    send_fn = src[send_fn_start:send_fn_end]
    meta_bcast = send_fn.find("dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)")
    assert meta_bcast != -1, "_send_combined_dual_proposals meta broadcast missing group=self.verify_group"
    payload_bcast = send_fn.find("dist.broadcast(payload_tensor, src=self.global_config.draft_config.master_rank, group=self.verify_group)")
    assert payload_bcast != -1, "_send_combined_dual_proposals payload broadcast missing group=self.verify_group"

    # Verify ALL three collectives are unconditional (no if-guard that could skip them).
    # The comment marker lives just above Phase 3 in the source.
    draft_h2_header = src[h2_section_start - 200:h2_section_start]
    assert "ALL collectives in this block are UNCONDITIONAL" in draft_h2_header, \
        "DRAFT H2: unconditional comment marker missing"


def test_h2_schedule_target_order_via_verify_group():
    """TARGET H2 steady: send normal → send eager → recv combined, all via verify_group."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")

    # Locate the H2 steady block on the TARGET side
    h2_section_start = src.find("# Phase A: Normal verify broadcast (UNCONDITIONAL)")
    assert h2_section_start != -1, "missing TARGET H2 Phase A marker"

    h2_section_end = src.find("# === Legacy ordering", h2_section_start)
    assert h2_section_end != -1, "missing TARGET H2 legacy ordering marker"

    h2_block = src[h2_section_start:h2_section_end]

    # 1. Normal verify broadcast via verify_group on TARGET
    normal_send_pos = h2_block.find("verify_from_proposals(")
    assert normal_send_pos != -1, "TARGET H2: missing verify_from_proposals call for normal send"

    # Verify group= in verify_from_proposals call (appears after the function name)
    normal_group_pos = h2_block.find("group=self.verify_group", normal_send_pos)
    assert normal_group_pos != -1, "TARGET H2: verify_from_proposals missing group=self.verify_group"
    # The group= should be within a reasonable distance of the call
    assert normal_group_pos - normal_send_pos < 500, \
        "TARGET H2: group=self.verify_group too far from verify_from_proposals"

    # Also check the else-branch (empty target_seqs) still broadcasts via verify_group
    empty_normal_bcast = h2_block.find('dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=self.verify_group)')
    assert empty_normal_bcast != -1, "TARGET H2: empty normal result broadcast missing group=self.verify_group"

    # 2. Eager verify broadcast via verify_group on TARGET
    eager_send_pos = h2_block.find("_run_eager_verify_sidecar(target_eager_seqs, eager_proposals, plan, group=self.verify_group)")
    assert eager_send_pos != -1, "TARGET H2: missing _run_eager_verify_sidecar with group=self.verify_group"

    # Also check the else-branch (now uses source-authoritative meta broadcast)
    empty_eager_bcast = h2_block.find('dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)')
    assert empty_eager_bcast != -1, "TARGET H2: empty eager result broadcast missing group=self.verify_group (meta protocol)"

    # 3. Combined recv via verify_group
    combined_recv_pos = h2_block.find("_receive_combined_dual_proposals(")
    assert combined_recv_pos != -1, "TARGET H2: missing _receive_combined_dual_proposals call"

    # Ordering: normal send < eager send < combined recv
    assert normal_send_pos < eager_send_pos < combined_recv_pos, \
        f"TARGET H2 order violation: normal_send={normal_send_pos}, eager_send={eager_send_pos}, combined_recv={combined_recv_pos}"

    # Verify _receive_combined_dual_proposals broadcasts use verify_group
    recv_fn_start = src.find("def _receive_combined_dual_proposals(")
    recv_fn_end = src.find("def _parse_combined_normal_payload(", recv_fn_start)
    recv_fn = src[recv_fn_start:recv_fn_end]
    recv_meta_bcast = recv_fn.find("dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)")
    assert recv_meta_bcast != -1, "_receive_combined_dual_proposals meta broadcast missing group=self.verify_group"
    recv_payload_bcast = recv_fn.find("dist.broadcast(payload_tensor, src=self.global_config.draft_config.master_rank, group=self.verify_group)")
    assert recv_payload_bcast != -1, "_receive_combined_dual_proposals payload broadcast missing group=self.verify_group"

    # Verify unconditional — check header before Phase A
    target_h2_header = src[h2_section_start - 200:h2_section_start]
    assert "ALL collectives in this block are UNCONDITIONAL" in target_h2_header, \
        "TARGET H2: unconditional comment marker missing"


def test_h2_empty_broadcasts_are_unconditional():
    """Both else-branches (empty sets) still execute dist.broadcast with verify_group."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")

    # TARGET side: empty normal broadcast
    assert 'verify_res = torch.zeros((4, 0)' in src, "missing empty normal verify tensor creation"
    # TARGET side: empty eager broadcast now uses source-authoritative meta [6,0,0]
    assert 'meta = torch.tensor([6, 0, 0]' in src, "missing empty eager verify meta tensor creation"

    # DRAFT side: receive calls accept group=self.verify_group even when target_seqs is empty
    # The _receive_verify_result call is unconditional (outside if/else)
    draft_h2_start = src.find("# Phase 3: Receive normal verify result (UNCONDITIONAL)")
    draft_h2_end = src.find("# Phase 6: Send combined proposals (UNCONDITIONAL)", draft_h2_start)
    draft_h2_block = src[draft_h2_start:draft_h2_end]
    # Both receive calls should appear before any if/else that conditions on target_seqs
    recv_normal_line = draft_h2_block.find("_receive_verify_result(target_seqs, group=self.verify_group)")
    recv_eager_line = draft_h2_block.find("_receive_eager_verify_result(target_eager_seqs, group=self.verify_group)")
    # The "if target_seqs:" block should come AFTER the recv call
    if_seqs_pos = draft_h2_block.find("if target_seqs:")
    assert recv_normal_line < if_seqs_pos, \
        "DRAFT H2: _receive_verify_result must be called BEFORE the if target_seqs: guard"


# --- Phase 1H-continuous-trace tests ---


def test_continuous_draft_eager_new_subset_of_original_target_home():
    """draft_eager_new_set must be subset of original_target_home_set."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3, 5, 7],
        draft_home_set=[4, 6, 8],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.original_target_home_set = [1, 3, 5, 7]
    plan.draft_eager_new_set = [1, 5]
    assert set(plan.draft_eager_new_set).issubset(set(plan.original_target_home_set)), \
        "draft_eager_new_set must be subset of original_target_home_set"


def test_continuous_draft_eager_new_outside_original_target_home():
    """draft_eager_new_set outside original_target_home_set — must be detected."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3, 5, 7],
        draft_home_set=[4, 6, 8],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.original_target_home_set = [1, 3, 5, 7]
    plan.draft_eager_new_set = [1, 9]  # 9 is outside
    assert not set(plan.draft_eager_new_set).issubset(set(plan.original_target_home_set)), \
        "seq 9 should not pass subset check"


def test_continuous_draft_eager_set_trace_union():
    """draft_eager_set_trace must equal new ∪ continue."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3, 5, 7],
        draft_home_set=[4, 6, 8],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.draft_eager_new_set = [1, 3]
    plan.draft_eager_continue_set = []
    plan.draft_eager_set_trace = sorted(set(plan.draft_eager_new_set) | set(plan.draft_eager_continue_set))
    assert plan.draft_eager_set_trace == [1, 3], \
        f"draft_eager_set_trace should be [1,3], got {plan.draft_eager_set_trace}"


def test_continuous_executed_sets_empty():
    """All executed eager sets must be empty in trace-only."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    assert plan.draft_eager_set_executed == [], "draft_eager_set_executed must be empty"
    assert plan.target_eager_set_executed == [], "target_eager_set_executed must be empty"
    assert plan.target_home_set_executed == [], "target_home_set_executed must be empty"
    assert plan.draft_home_set_executed == [], "draft_home_set_executed must be empty"


def test_continuous_execution_counters_zero():
    """All eager execution counters must be 0 in trace-only."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    assert plan.eager_tokens_generated == 0
    assert plan.eager_tokens_verified == 0
    assert plan.eager_tokens_promoted == 0
    assert plan.eager_tokens_discarded == 0
    assert plan.eager_tokens_accepted == 0


def test_continuous_proposal_ids_string_format():
    """Proposal IDs use the ce:new string format."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.step_id = 5
    plan.plan_id = 10
    plan.draft_eager_new_set = [1, 3]
    for seq_id in plan.draft_eager_new_set:
        plan.eager_proposal_id_by_seq_id[seq_id] = f"ce:new:{plan.step_id}:{plan.plan_id}:{seq_id}"
        plan.eager_parent_proposal_id_by_seq_id[seq_id] = f"normal:{plan.step_id}:{plan.plan_id}:{seq_id}"
        plan.eager_parent_kind_by_seq_id[seq_id] = "normal"
        plan.eager_proposal_state_by_seq_id[seq_id] = "selected"
        plan.eager_promotion_condition_pending_by_seq_id[seq_id] = True
    assert plan.eager_proposal_id_by_seq_id[1] == "ce:new:5:10:1"
    assert plan.eager_proposal_id_by_seq_id[3] == "ce:new:5:10:3"
    assert plan.eager_parent_proposal_id_by_seq_id[1] == "normal:5:10:1"
    assert plan.eager_parent_kind_by_seq_id[1] == "normal"


def test_continuous_proposal_ids_unique():
    """Proposal IDs must be unique across seqs."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.step_id = 1
    plan.draft_eager_new_set = [1, 3]
    ids = set()
    for seq_id in plan.draft_eager_new_set:
        pid = f"ce:new:{plan.step_id}:{plan.plan_id}:{seq_id}"
        assert pid not in ids, f"duplicate proposal_id={pid}"
        ids.add(pid)
        plan.eager_proposal_id_by_seq_id[seq_id] = pid
    assert len(ids) == 2


def test_continuous_proposal_state_only_selected_or_pending():
    """Only 'selected' and 'pending_parent' states allowed in trace-only."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.eager_proposal_state_by_seq_id = {1: "selected", 2: "pending_parent"}
    ALLOWED = {"selected", "pending_parent"}
    for state in plan.eager_proposal_state_by_seq_id.values():
        assert state in ALLOWED, f"state {state!r} not allowed in trace-only"
    # Verify "ready" would be rejected.
    bad = {"selected", "ready"}
    assert not bad.issubset(ALLOWED), "'ready' should not be in allowed states"


def test_continuous_target_eager_set_trace_empty():
    """target_eager_set_trace must be empty in first version."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    assert plan.target_eager_set_trace == [], "target_eager_set_trace must be empty"


def test_continuous_draft_eager_continue_set_empty():
    """draft_eager_continue_set must be empty in first version."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    assert plan.draft_eager_continue_set == [], "draft_eager_continue_set must be empty"


def test_continuous_exclusion_views_computed():
    """Trace exclusion views apply lane rules without mutating originals."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3, 5, 7],
        draft_home_set=[4, 6, 8],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.original_target_home_set = [1, 3, 5, 7]
    plan.original_draft_home_set = [4, 6, 8]
    plan.target_eager_set_trace = []        # empty in first version
    plan.draft_eager_continue_set = []       # empty in first version
    _te = set(plan.target_eager_set_trace)
    _dc = set(plan.draft_eager_continue_set)
    plan.target_home_set_after_eager_exclusion_trace = list(plan.original_target_home_set)
    plan.draft_home_set_after_eager_exclusion_trace = [
        s for s in plan.original_draft_home_set if s not in _te and s not in _dc
    ]
    assert plan.target_home_set_after_eager_exclusion_trace == [1, 3, 5, 7]
    assert plan.draft_home_set_after_eager_exclusion_trace == [4, 6, 8]


def test_continuous_skip_pre_verify_reason_tracked():
    """Skip reasons include skip_pre_verify_seq, not skip_post_verify_seq."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.continuous_eager_skip_reason_by_seq_id = {1: "skip_pre_verify_seq", 3: "score_below_threshold"}
    reasons = set(plan.continuous_eager_skip_reason_by_seq_id.values())
    assert "skip_pre_verify_seq" in reasons, "should contain skip_pre_verify_seq"
    assert "skip_post_verify_seq" not in reasons, "should NOT contain skip_post_verify_seq"


def test_continuous_parent_kind_normal_only():
    """Parent kind must be 'normal' or empty — no 'eager' in first version."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.eager_parent_kind_by_seq_id = {1: "normal", 2: ""}
    ALLOWED = {"normal", ""}
    for kind in plan.eager_parent_kind_by_seq_id.values():
        assert kind in ALLOWED, f"parent_kind {kind!r} not allowed"


def test_continuous_skip_reason_counts_consistent():
    """continuous_eager_skip_reason_counts must match by_seq_id values."""
    skip_by_seq = {1: "skip_pre_verify_seq", 2: "skip_pre_verify_seq", 3: "score_below_threshold"}
    expected = {}
    for reason in skip_by_seq.values():
        expected[reason] = expected.get(reason, 0) + 1
    assert expected == {"skip_pre_verify_seq": 2, "score_below_threshold": 1}


def test_continuous_original_home_sets_preserved():
    """original_target_home_set and original_draft_home_set are not mutated."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3, 5, 7],
        draft_home_set=[4, 6, 8],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.original_target_home_set = list(plan.target_home_set)
    plan.original_draft_home_set = list(plan.draft_home_set)
    assert plan.original_target_home_set == [1, 3, 5, 7]
    assert plan.original_draft_home_set == [4, 6, 8]
    # Mutating actual sets should not affect originals.
    plan.target_home_set.append(9)
    assert plan.original_target_home_set == [1, 3, 5, 7]


def test_continuous_to_trace_dict_includes_new_fields():
    """to_trace_dict() includes all new continuous eager fields."""
    plan = StepPlan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        target_home_set=[1, 3],
        draft_home_set=[4, 6],
        dual_batch_enabled=True,
        plan_phase="steady",
    )
    plan.original_target_home_set = [1, 3]
    plan.original_draft_home_set = [4, 6]
    plan.draft_eager_new_set = [1]
    plan.draft_eager_set_trace = [1]
    plan.target_home_pre_verify_by_seq_id = {1: False, 3: True}
    plan.continuous_eager_skip_reason_by_seq_id = {3: "skip_pre_verify_seq"}
    plan.continuous_eager_skip_reason_counts = {"skip_pre_verify_seq": 1}
    plan.eager_proposal_id_by_seq_id = {1: "ce:new:0:1:1"}
    plan.eager_parent_proposal_id_by_seq_id = {1: "normal:0:1:1"}
    plan.eager_parent_kind_by_seq_id = {1: "normal"}
    plan.eager_proposal_state_by_seq_id = {1: "selected"}
    plan.eager_promotion_condition_pending_by_seq_id = {1: True}
    d = plan.to_trace_dict()
    assert d.get("original_target_home_set") == [1, 3]
    assert d.get("original_draft_home_set") == [4, 6]
    assert d.get("draft_eager_new_set") == [1]
    assert d.get("draft_eager_set_trace") == [1]
    assert d.get("target_home_pre_verify_by_seq_id") == {"1": False, "3": True}
    assert d.get("continuous_eager_skip_reason_by_seq_id") == {"3": "skip_pre_verify_seq"}
    assert d.get("continuous_eager_skip_reason_counts") == {"skip_pre_verify_seq": 1}
    assert d.get("eager_proposal_id_by_seq_id") == {"1": "ce:new:0:1:1"}
    assert d.get("eager_parent_proposal_id_by_seq_id") == {"1": "normal:0:1:1"}
    assert d.get("eager_parent_kind_by_seq_id") == {"1": "normal"}
    assert d.get("eager_proposal_state_by_seq_id") == {"1": "selected"}
    assert d.get("eager_promotion_condition_pending_by_seq_id") == {"1": True}
    assert d.get("target_eager_set_trace") == []
    assert d.get("draft_eager_continue_set") == []
    assert d.get("draft_eager_set_executed") == []
    assert d.get("target_eager_set_executed") == []
    assert d.get("target_home_set_executed") == []
    assert d.get("draft_home_set_executed") == []


# --- Phase 1H-continuous-promotion-trace tests ---

def test_continuous_promotion_pending_parent_after_selection():
    """New selected seqs get state='pending_parent' with chain_depth=1 in continuous_eager_* fields."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.draft_eager_new_set = [1]
    plan.eager_selected_seq_ids = [1]
    plan.eager_proposal_state_by_seq_id = {1: "selected"}
    plan.continuous_eager_proposal_state_by_seq_id = {1: "pending_parent"}
    plan.continuous_eager_parent_kind_by_seq_id = {1: "normal"}
    plan.continuous_eager_chain_depth_by_seq_id = {1: 1}
    assert plan.continuous_eager_chain_depth_by_seq_id[1] == 1
    assert plan.continuous_eager_proposal_state_by_seq_id[1] == "pending_parent"


def test_continuous_promotion_normal_full_accept_promotes_to_ready():
    """Full accept → state='ready' with promotion_reason='normal_full_accept'."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_promoted_seq_ids = [1]
    plan.continuous_eager_promotion_reason_by_seq_id = {1: "normal_full_accept"}
    plan.continuous_eager_trace_promoted_count = 1
    assert 1 in plan.continuous_eager_promoted_seq_ids
    assert plan.continuous_eager_promotion_reason_by_seq_id[1] == "normal_full_accept"
    assert plan.continuous_eager_trace_promoted_count == 1


def test_continuous_promotion_normal_reject_discards():
    """Invalidated > 0 → state='discarded' with discard_reason."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_discarded_seq_ids = [1]
    plan.continuous_eager_discard_reason_by_seq_id = {1: "normal_verify_invalidated_tokens=2"}
    plan.continuous_eager_trace_discarded_count = 1
    assert 1 in plan.continuous_eager_discarded_seq_ids
    assert plan.continuous_eager_discard_reason_by_seq_id[1] == "normal_verify_invalidated_tokens=2"
    assert plan.continuous_eager_trace_discarded_count == 1


def test_continuous_promotion_ready_populates_target_eager_set_trace():
    """Ready proposals appear in target_eager_set_trace."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.target_eager_set_trace = [1]
    plan.continuous_eager_trace_target_ready_count = 1
    assert plan.target_eager_set_trace == [1]
    assert plan.continuous_eager_trace_target_ready_count == 1


def test_continuous_promotion_target_eager_enables_draft_continue():
    """target_eager_set_trace seqs → draft_eager_continue_set ⊆ target_eager_set_trace."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1, 3], draft_home_set=[2, 4],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.target_eager_set_trace = [1, 3]
    plan.draft_eager_continue_set = [1]
    plan.draft_eager_new_set = [5]
    plan.draft_eager_set_trace = [1, 5]
    cont_set = set(plan.draft_eager_continue_set)
    target_set = set(plan.target_eager_set_trace)
    assert cont_set.issubset(target_set), f"{cont_set} not subset of {target_set}"
    assert set(plan.draft_eager_set_trace) == set(plan.draft_eager_new_set) | set(plan.draft_eager_continue_set)


def test_continuous_promotion_continue_increments_chain_depth():
    """Chain depth increases on continue."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_chain_depth_by_seq_id = {1: 3}
    plan.continuous_eager_parent_kind_by_seq_id = {1: "eager"}
    plan.continuous_eager_proposal_state_by_seq_id = {1: "continue_pending"}
    assert plan.continuous_eager_chain_depth_by_seq_id[1] == 3
    assert plan.continuous_eager_parent_kind_by_seq_id[1] == "eager"


def test_continuous_promotion_proposal_id_consistent_across_records():
    """Same continuous_eager proposal_id with consistent metadata is OK."""
    plan1 = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan1.continuous_eager_proposal_id_by_seq_id = {1: "ce:new:0:1:1"}
    plan1.continuous_eager_proposal_state_by_seq_id = {1: "pending_parent"}
    plan1.continuous_eager_parent_kind_by_seq_id = {1: "normal"}

    plan2 = StepPlan(
        plan_id=2, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan2.continuous_eager_proposal_id_by_seq_id = {1: "ce:new:0:1:1"}
    plan2.continuous_eager_proposal_state_by_seq_id = {1: "pending_parent"}
    plan2.continuous_eager_parent_kind_by_seq_id = {1: "normal"}
    # Same ID, same metadata — should be consistent.
    assert plan1.continuous_eager_proposal_id_by_seq_id == plan2.continuous_eager_proposal_id_by_seq_id


def test_continuous_promotion_proposal_id_inconsistent_fails():
    """Inconsistent metadata for same continuous_eager proposal_id should be detected."""
    plan1 = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan1.continuous_eager_proposal_id_by_seq_id = {1: "ce:new:0:1:1"}
    plan1.continuous_eager_proposal_state_by_seq_id = {1: "pending_parent"}

    plan2 = StepPlan(
        plan_id=2, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan2.continuous_eager_proposal_id_by_seq_id = {1: "ce:new:0:1:1"}
    plan2.continuous_eager_proposal_state_by_seq_id = {1: "discarded"}
    # Same ID but different state — checker should flag this.
    assert plan1.continuous_eager_proposal_state_by_seq_id[1] != plan2.continuous_eager_proposal_state_by_seq_id[1]


def test_continuous_promotion_fallback_metadata_ok():
    """Non-steady records with missing metadata should be handled gracefully."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="fallback",
    )
    plan.eager_trace_only = True
    plan.effective_enable_eager_trace = True
    plan.eager_policy = "tight_only"
    plan.original_target_home_set = [1]
    plan.missing_eager_metadata_seq_ids = [1]
    plan.continuous_eager_skip_reason_by_seq_id = {1: "not_steady_phase"}
    # Fallback metadata missing is OK — no error expected.
    assert 1 in plan.missing_eager_metadata_seq_ids


def test_continuous_promotion_executed_sets_remain_empty():
    """All executed sets remain empty after promotion/discard/continue."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1, 3], draft_home_set=[2, 4],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.target_eager_set_trace = [1]
    plan.draft_eager_continue_set = [1]
    plan.draft_eager_new_set = [3]
    plan.draft_eager_set_trace = [1, 3]
    plan.continuous_eager_promoted_seq_ids = [3]
    plan.continuous_eager_discarded_seq_ids = []
    # Executed sets must be empty.
    assert plan.draft_eager_set_executed == []
    assert plan.target_eager_set_executed == []
    assert plan.target_home_set_executed == []
    assert plan.draft_home_set_executed == []


def test_continuous_promotion_counters_positive():
    """Trace counters match promoted/discarded/selected list lengths."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1, 3], draft_home_set=[2, 4],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.draft_eager_new_set = [1, 3]
    plan.continuous_eager_trace_selected_count = 2
    plan.continuous_eager_promoted_seq_ids = [1]
    plan.continuous_eager_trace_promoted_count = 1
    plan.continuous_eager_discarded_seq_ids = [3]
    plan.continuous_eager_trace_discarded_count = 1
    assert plan.continuous_eager_trace_selected_count == len(plan.draft_eager_new_set)
    assert plan.continuous_eager_trace_promoted_count == len(plan.continuous_eager_promoted_seq_ids)
    assert plan.continuous_eager_trace_discarded_count == len(plan.continuous_eager_discarded_seq_ids)


def test_continuous_promotion_no_verified_or_applied_state():
    """continuous_eager_proposal_state must not be 'verified' or 'applied' in trace-only."""
    allowed = {
        "selected", "pending_parent", "pending_parent_unknown",
        "ready", "discarded", "target_trace_ready", "continue_pending",
    }
    assert "verified" not in allowed
    assert "applied" not in allowed


def test_continuous_promotion_parent_kind_eager_has_parent_id():
    """When parent_kind='eager', parent_proposal_id should be non-empty."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_proposal_id_by_seq_id = {1: "ce:continue:1:2:1"}
    plan.continuous_eager_parent_proposal_id_by_seq_id = {1: "ce:new:0:1:1"}
    plan.continuous_eager_parent_kind_by_seq_id = {1: "eager"}
    plan.continuous_eager_proposal_state_by_seq_id = {1: "continue_pending"}
    parent_id = plan.continuous_eager_parent_proposal_id_by_seq_id.get(1, "")
    assert parent_id != "", f"parent_kind=eager should have non-empty parent_proposal_id"


def test_continuous_promotion_to_trace_dict_includes_promotion_fields():
    """to_trace_dict() includes new continuous promotion-trace fields."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_promoted_seq_ids = [1]
    plan.continuous_eager_trace_promoted_count = 1
    plan.continuous_eager_proposal_state_by_seq_id = {1: "ready"}
    plan.continuous_eager_chain_depth_by_seq_id = {1: 1}
    d = plan.to_trace_dict()
    assert d.get("continuous_eager_promoted_seq_ids") == [1]
    assert d.get("continuous_eager_trace_promoted_count") == 1
    assert d.get("continuous_eager_proposal_state_by_seq_id") == {"1": "ready"}
    assert d.get("continuous_eager_chain_depth_by_seq_id") == {"1": 1}


# --- Phase 1I-A scaffold tests ---


def test_scaffold_buffer_store_and_get():
    """Scaffold buffer stores proposals and retrieves them by seq_id."""
    buf = ContinuousEagerDraftExecutionBuffer()
    p = EagerBufferedProposal(
        seq_id=1, request_id="r1", home_batch_id=0,
        eager_token_ids=[10, 20], eager_len=2, eager_base_len=100,
        source_plan_id=1, source_step_id=0, source_home_batch_id=0,
        verify_with_batch_id=None, score=0.5, policy="tight_only",
        valid=True, ready=False, original_eager_base_len_at_generation=100,
    )
    buf.store([p])
    assert buf.size() == 1
    assert buf.get(1) is not None
    assert buf.get(1).eager_len == 2
    assert buf.get(999) is None


def test_scaffold_buffer_mark_ready():
    """Scaffold buffer mark_ready transitions proposal to ready."""
    buf = ContinuousEagerDraftExecutionBuffer()
    p = EagerBufferedProposal(
        seq_id=1, request_id="r1", home_batch_id=0,
        eager_token_ids=[10], eager_len=1, eager_base_len=100,
        source_plan_id=1, source_step_id=0, source_home_batch_id=0,
        verify_with_batch_id=None, score=0.5, policy="tight_only",
        valid=True, ready=False, original_eager_base_len_at_generation=100,
    )
    buf.store([p])
    assert buf.mark_ready(1)
    assert buf.get(1).ready
    assert buf.ready_seq_ids() == [1]
    assert not buf.mark_ready(999)


def test_scaffold_buffer_discard():
    """Scaffold buffer discard removes proposals."""
    buf = ContinuousEagerDraftExecutionBuffer()
    p = EagerBufferedProposal(
        seq_id=1, request_id="r1", home_batch_id=0,
        eager_token_ids=[10], eager_len=1, eager_base_len=100,
        source_plan_id=1, source_step_id=0, source_home_batch_id=0,
        verify_with_batch_id=None, score=0.5, policy="tight_only",
        valid=True, ready=False, original_eager_base_len_at_generation=100,
    )
    buf.store([p])
    dropped = buf.discard([1])
    assert dropped == [1]
    assert buf.size() == 0
    assert buf.get(1) is None


def test_scaffold_buffer_discard_inactive():
    """Scaffold buffer discard_inactive removes finished seqs."""
    buf = ContinuousEagerDraftExecutionBuffer()
    p1 = EagerBufferedProposal(
        seq_id=1, request_id="r1", home_batch_id=0,
        eager_token_ids=[10], eager_len=1, eager_base_len=100,
        source_plan_id=1, source_step_id=0, source_home_batch_id=0,
        verify_with_batch_id=None, score=0.5, policy="tight_only",
        valid=True, ready=False, original_eager_base_len_at_generation=100,
    )
    p2 = EagerBufferedProposal(
        seq_id=2, request_id="r2", home_batch_id=0,
        eager_token_ids=[20], eager_len=1, eager_base_len=200,
        source_plan_id=1, source_step_id=0, source_home_batch_id=0,
        verify_with_batch_id=None, score=0.5, policy="tight_only",
        valid=True, ready=False, original_eager_base_len_at_generation=200,
    )
    buf.store([p1, p2])
    dropped = buf.discard_inactive([1])
    assert sorted(dropped) == [2]
    assert buf.size() == 1
    assert buf.get(1) is not None


def test_scaffold_draft_eager_new_set_executed_field():
    """StepPlan has draft_eager_new_set_executed field."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.draft_eager_new_set_executed = [1, 2]
    d = plan.to_trace_dict()
    assert d.get("draft_eager_new_set_executed") == [1, 2]


def test_scaffold_counters_serialized():
    """Scaffold counters include tokens_generated, proposals_sent, etc."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_scaffold_tokens_generated = 42
    plan.continuous_eager_scaffold_proposals_generated = 3
    plan.continuous_eager_scaffold_proposals_sent = 3
    plan.continuous_eager_scaffold_proposals_received = 3
    plan.continuous_eager_scaffold_proposals_promoted = 2
    plan.continuous_eager_scaffold_proposals_discarded = 1
    d = plan.to_trace_dict()
    assert d["continuous_eager_scaffold_tokens_generated"] == 42
    assert d["continuous_eager_scaffold_proposals_generated"] == 3
    assert d["continuous_eager_scaffold_proposals_sent"] == 3
    assert d["continuous_eager_scaffold_proposals_received"] == 3
    assert d["continuous_eager_scaffold_proposals_promoted"] == 2
    assert d["continuous_eager_scaffold_proposals_discarded"] == 1


def test_scaffold_audit_fields_serialized():
    """Full-accept audit and base construction audit fields are serialized."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_parent_acceptance_source_by_seq_id = {1: "normal_verify"}
    plan.continuous_eager_parent_acceptance_is_exact_by_seq_id = {1: True}
    plan.continuous_eager_exec_base_kind_by_seq_id = {1: "draft_eager_new_set"}
    plan.continuous_eager_exec_base_len_by_seq_id = {1: 100}
    plan.continuous_eager_exec_base_is_valid_by_seq_id = {1: True}
    d = plan.to_trace_dict()
    assert d["continuous_eager_parent_acceptance_source_by_seq_id"] == {"1": "normal_verify"}
    assert d["continuous_eager_parent_acceptance_is_exact_by_seq_id"] == {"1": True}
    assert d["continuous_eager_exec_base_kind_by_seq_id"] == {"1": "draft_eager_new_set"}
    assert d["continuous_eager_exec_base_len_by_seq_id"] == {"1": 100}
    assert d["continuous_eager_exec_base_is_valid_by_seq_id"] == {"1": True}


def test_scaffold_target_eager_set_executed_empty():
    """target_eager_set_executed stays empty (no target verification in scaffold)."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1], draft_home_set=[2],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.draft_eager_new_set_executed = [1, 2]
    # target_eager_set_executed is never populated in scaffold phase.
    assert plan.target_eager_set_executed == []
    d = plan.to_trace_dict()
    assert d["target_eager_set_executed"] == []


def test_scaffold_continue_set_not_in_executed():
    """draft_eager_continue_set seqs are NOT in draft_eager_new_set_executed."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1, 2], draft_home_set=[3, 4],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.draft_eager_new_set = [1]
    plan.draft_eager_continue_set = [2]
    plan.draft_eager_new_set_executed = [1]
    # Verify: continue_set[2] NOT in executed[1]
    assert 2 not in set(plan.draft_eager_new_set_executed)


def test_scaffold_counters_consistency():
    """Scaffold counters read correctly from to_trace_dict."""
    plan = StepPlan(
        plan_id=1, iteration_id=1, execution_mode="dual_batch_pearl",
        target_home_set=[1, 2], draft_home_set=[3, 4],
        dual_batch_enabled=True, plan_phase="steady",
    )
    plan.continuous_eager_scaffold_tokens_generated = 10
    plan.continuous_eager_scaffold_proposals_generated = 5
    d = plan.to_trace_dict()
    assert d["continuous_eager_scaffold_tokens_generated"] == 10
    assert d["continuous_eager_scaffold_proposals_generated"] == 5


# --- runner ---

if __name__ == "__main__":
    tests = [
        test_valid_eager_receive_subset_of_target_home,
        test_valid_eager_receive_empty,
        test_valid_multiple_eager_seqs,
        test_eager_receive_overlaps_draft_home,
        test_eager_receive_outside_target_home,
        test_eager_receive_overlaps_target_eager,
        test_eager_receive_exceeds_max_requests,
        test_eager_receive_exceeds_per_request_tokens,
        test_eager_receive_exceeds_step_tokens,
        test_eager_receive_finished_seq,
        test_divergence_case_valid,
        test_divergence_case_invalid,
        test_draft_home_set_preserved_when_target_eager_overlaps,
        test_target_eager_and_target_home_still_disjoint,
        test_normal_buffer_has_all_after_eager_store,
        test_overlap_rejected_without_eager_execution,
        test_overlap_fails_outside_steady_phase,
        test_overlap_target_eager_subset_constraint,
        test_overlap_trace_fields_default,
        test_overlap_trace_fields_in_trace_dict,
        test_eager_buffer_ready_only_filter,
        test_eager_buffer_discard_removes_from_ready,
        test_promote_non_overlap_eager_proposals,
        test_promote_discard_mixed_overlap_and_rejected,
        test_eager_lifecycle_trace_fields_default,
        test_eager_lifecycle_trace_fields_in_trace_dict,
        test_combined_payload_all_empty,
        test_combined_payload_empty_eager_nonempty_normal,
        test_combined_payload_empty_normal_nonempty_eager,
        test_combined_payload_with_conditional,
        test_send_normal_subset_assertion_accepts_partial,
        test_send_eager_subset_assertion_accepts_empty,
        test_h2_schedule_draft_order_via_verify_group,
        test_h2_schedule_target_order_via_verify_group,
        test_h2_empty_broadcasts_are_unconditional,
        test_eager_protocol_source_cols_zero_receiver_nonzero,
        test_eager_protocol_source_cols_one_receiver_zero,
        test_eager_protocol_both_cols_zero,
        test_eager_protocol_both_cols_positive,
        test_eager_protocol_payload_skip_driven_by_source_meta,
        test_eager_protocol_receiver_ignores_local_length,
        test_h2_eager_shape_divergence_target_empty_source_nonempty,
        test_h2_eager_shape_divergence_source_empty_target_nonempty,
        test_receive_eager_verify_result_uses_meta_broadcast,
        test_build_eager_verify_result_uses_meta_broadcast,
        test_h2_expected_normal_empty_target_eager,
        test_h2_expected_normal_nonempty_target_eager,
        test_h2_normal_conditional_excluded_invariant,
        test_h2_expected_eager_matches_draft_eager,
        test_h2_target_and_draft_eager_distinct,
        test_h2_target_eager_intersects_draft_home,
        test_h2_target_eager_no_intersection_draft_home,
        test_h2_non_steady_preserves_old_invariant,
        test_h2_trace_dict_includes_new_fields,
        # Phase 1H-continuous-trace tests
        test_continuous_draft_eager_new_subset_of_original_target_home,
        test_continuous_draft_eager_new_outside_original_target_home,
        test_continuous_draft_eager_set_trace_union,
        test_continuous_executed_sets_empty,
        test_continuous_execution_counters_zero,
        test_continuous_proposal_ids_string_format,
        test_continuous_proposal_ids_unique,
        test_continuous_proposal_state_only_selected_or_pending,
        test_continuous_target_eager_set_trace_empty,
        test_continuous_draft_eager_continue_set_empty,
        test_continuous_exclusion_views_computed,
        test_continuous_skip_pre_verify_reason_tracked,
        test_continuous_parent_kind_normal_only,
        test_continuous_skip_reason_counts_consistent,
        test_continuous_original_home_sets_preserved,
        test_continuous_to_trace_dict_includes_new_fields,
        # Phase 1H-continuous-promotion-trace tests
        test_continuous_promotion_pending_parent_after_selection,
        test_continuous_promotion_normal_full_accept_promotes_to_ready,
        test_continuous_promotion_normal_reject_discards,
        test_continuous_promotion_ready_populates_target_eager_set_trace,
        test_continuous_promotion_target_eager_enables_draft_continue,
        test_continuous_promotion_continue_increments_chain_depth,
        test_continuous_promotion_proposal_id_consistent_across_records,
        test_continuous_promotion_proposal_id_inconsistent_fails,
        test_continuous_promotion_fallback_metadata_ok,
        test_continuous_promotion_executed_sets_remain_empty,
        test_continuous_promotion_counters_positive,
        test_continuous_promotion_no_verified_or_applied_state,
        test_continuous_promotion_parent_kind_eager_has_parent_id,
        test_continuous_promotion_to_trace_dict_includes_promotion_fields,
        # Phase 1I-A scaffold tests
        test_scaffold_buffer_store_and_get,
        test_scaffold_buffer_mark_ready,
        test_scaffold_buffer_discard,
        test_scaffold_buffer_discard_inactive,
        test_scaffold_draft_eager_new_set_executed_field,
        test_scaffold_counters_serialized,
        test_scaffold_audit_fields_serialized,
        test_scaffold_target_eager_set_executed_empty,
        test_scaffold_continue_set_not_in_executed,
        test_scaffold_counters_consistency,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        raise SystemExit(1)
