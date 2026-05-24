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
    return mod.EagerBufferedProposal, mod.BufferedProposal, mod.ProposalBuffer, mod.EagerProposalBuffer


StepPlan, RequestBudget = _load_step_plan()
EagerBufferedProposal, BufferedProposal, ProposalBuffer, EagerProposalBuffer = _load_dual_batch()


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
