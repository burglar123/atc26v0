"""CPU-safe V4O mailbox payload consume/invalidate lifecycle tests."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NANO_PREFIX = "nano_pearl"
for name in list(sys.modules):
    if name == _NANO_PREFIX or name.startswith(f"{_NANO_PREFIX}."):
        del sys.modules[name]

nano_pkg = types.ModuleType(_NANO_PREFIX)
nano_pkg.__path__ = [os.path.join(REPO_ROOT, "nano_pearl")]
sys.modules[_NANO_PREFIX] = nano_pkg
engine_pkg_name = f"{_NANO_PREFIX}.pearl_engine"
engine_pkg = types.ModuleType(engine_pkg_name)
engine_pkg.__path__ = [os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine")]
sys.modules[engine_pkg_name] = engine_pkg

mailbox_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox")
transport_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_transport")
apply_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_verify_apply")

MailboxPayload = mailbox_mod.MailboxPayload
STSpecPayloadMailbox = mailbox_mod.STSpecPayloadMailbox
TargetForwardFromMailboxInput = transport_mod.TargetForwardFromMailboxInput
MailboxVerifyApplyError = apply_mod.MailboxVerifyApplyError
build_mailbox_kv_commit_plan = apply_mod.build_mailbox_kv_commit_plan
build_mailbox_payload_consume_plan = apply_mod.build_mailbox_payload_consume_plan
build_mailbox_verify_apply_plan = apply_mod.build_mailbox_verify_apply_plan
build_mailbox_verify_commit_plan = apply_mod.build_mailbox_verify_commit_plan
build_mailbox_verify_result = apply_mod.build_mailbox_verify_result
run_mailbox_payload_consume_probe = apply_mod.run_mailbox_payload_consume_probe
run_mailbox_verify_commit_probe = apply_mod.run_mailbox_verify_commit_probe


class FakeSeq:
    def __init__(self, seq_id: int, tokens: list[int], home_batch_id: int = 1):
        self.seq_id = seq_id
        self.request_id = f"r{seq_id}"
        self.token_ids = list(tokens)
        self.num_tokens = len(tokens)
        self.num_prompt_tokens = len(tokens)
        self.last_token = tokens[-1]
        self.home_batch_id = home_batch_id
        self.block_table = [seq_id]
        self.is_finished = False
        self.ignore_eos = False
        self.first_token_ts = None
        self.finish_ts = None
        self.trace_stats = {"scheduled_iterations": [], "accepted_tokens": 0, "invalidated_predraft_tokens": 0}

    def __len__(self):
        return self.num_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(int(token_id))
        self.last_token = int(token_id)
        self.num_tokens += 1

    def record_accepted(self, accepted_len: int):
        self.trace_stats["accepted_tokens"] += int(accepted_len)

    def record_invalidated_predraft(self, invalidated_len: int):
        self.trace_stats["invalidated_predraft_tokens"] += int(invalidated_len)


def payload(seq_id: int, tokens: list[int], *, home_batch_id=1, offset=0, target_home_batch_id=1):
    return MailboxPayload(
        plan_id=9,
        producer_role="draft",
        producer_home_batch_id=home_batch_id,
        target_home_batch_id=target_home_batch_id,
        draft_home_batch_id=0,
        seq_id=seq_id,
        request_id=f"r{seq_id}",
        home_batch_id=home_batch_id,
        gamma=4,
        layout_kind="variable_offsets",
        protocol_version=1,
        draft_token_ids=list(tokens),
        per_seq_length=len(tokens),
        offset=offset,
        logical_step=1,
        producer_actual_exec_seq_ids=[seq_id],
        producer_draft_message_seq_ids=[seq_id],
    )


def v4u_payload(seq_id: int, tokens: list[int], *, plan_id=8, home_batch_id=0):
    return MailboxPayload(
        plan_id=plan_id,
        producer_role="draft",
        producer_home_batch_id=home_batch_id,
        target_home_batch_id=home_batch_id,
        draft_home_batch_id=home_batch_id,
        seq_id=seq_id,
        request_id=f"r{seq_id}",
        home_batch_id=home_batch_id,
        gamma=4,
        layout_kind="variable_offsets",
        protocol_version=1,
        draft_token_ids=list(tokens),
        per_seq_length=len(tokens),
        offset=0,
        logical_step=plan_id,
        producer_actual_exec_seq_ids=[seq_id],
        producer_draft_message_seq_ids=[seq_id],
        metadata={"payload_id": f"{plan_id}:{home_batch_id}:{seq_id}:0:{len(tokens)}"},
    )


def test_duplicate_put_same_available_payload_is_idempotent_skip():
    mailbox = STSpecPayloadMailbox()
    first = v4u_payload(0, [101], plan_id=8)
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    stats = mailbox.stats()
    assert stats["duplicate_put_count"] == 1
    assert stats["duplicate_put_idempotent_skip_count"] == 1
    assert stats["duplicate_put_conflict_count"] == 0
    assert stats["put_count"] == 1


def test_duplicate_put_same_key_different_payload_is_conflict():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(0, [v4u_payload(0, [101], plan_id=8)], plan_id=8, producer_role="draft")
    with pytest.raises(mailbox_mod.STSpecMailboxError, match="duplicate_put_conflict"):
        mailbox.put_payloads(0, [v4u_payload(0, [202], plan_id=8)], plan_id=8, producer_role="draft")
    stats = mailbox.stats()
    assert stats["duplicate_put_conflict_count"] == 1


def test_active_continuation_new_plan_can_reuse_consumed_home_seq_key():
    mailbox = STSpecPayloadMailbox()
    first = v4u_payload(0, [101], plan_id=8)
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    mailbox.apply_payload_lifecycle(
        plan_id=8,
        target_home_batch_id=0,
        consumed_token_count_by_payload_id={first.payload_id: 1},
        invalidated_token_count_by_payload_id={},
    )
    second = v4u_payload(0, [202], plan_id=9)
    mailbox.put_payloads(0, [second], plan_id=9, producer_role="draft")
    result = mailbox.get_payloads(0, [0], plan_id=9, consumer_role="target")
    assert result.success is True
    assert result.payloads[0].payload_id == second.payload_id
    assert mailbox.stats()["put_count"] == 2


def test_active_continuation_new_plan_can_reuse_stale_home_seq_key():
    mailbox = STSpecPayloadMailbox()
    first = v4u_payload(0, [101], plan_id=4)
    mailbox.put_payloads(0, [first], plan_id=4, producer_role="draft")
    mailbox.mark_payloads_stale([first.payload_id], plan_id=4, reason="draft_transport_envelope_recorded")
    second = v4u_payload(0, [202], plan_id=8)
    mailbox.put_payloads(0, [second], plan_id=8, producer_role="draft")
    result = mailbox.get_payloads(0, [0], plan_id=8, consumer_role="target")
    assert result.success is True
    assert result.payloads[0].payload_id == second.payload_id


def test_outstanding_available_payload_context_blocks_new_plan_before_put():
    mailbox = STSpecPayloadMailbox()
    existing = v4u_payload(0, [101], plan_id=4)
    incoming = v4u_payload(0, [202], plan_id=8)
    mailbox.put_payloads(0, [existing], plan_id=4, producer_role="draft")
    contexts = mailbox.available_payload_contexts_for([incoming])
    assert contexts[0]["existing_plan_id"] == 4
    assert contexts[0]["incoming_plan_id"] == 8
    assert contexts[0]["same_payload"] is False
    assert contexts[0]["lifecycle_state"] == "available"


def make_mailbox():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(1, [payload(1, [101], offset=0), payload(3, [301, 302, 303, 304], offset=1)])
    return mailbox


def make_input(home_batch_id: int = 1):
    return TargetForwardFromMailboxInput(
        plan_id=9,
        target_home_batch_id=home_batch_id,
        seq_ids=[1, 3],
        request_ids=["a", "b"],
        input_token_ids=[101, 301, 302, 303, 304],
        per_seq_lengths=[1, 4],
        offsets=[0, 1],
        total_tokens=5,
        gamma=4,
        positions=[2, 4, 5, 6, 7],
        kv_slot_ids=[1010, 3010, 3020, 3030, 3040],
        source_mailbox_payload_ids=["1:1:0:1", "1:3:1:4"],
        source_draft_plan_id=9,
    )


def make_step_plan(home_batch_id: int = 1, seq_ids=None):
    return SimpleNamespace(plan_id=9, target_home_batch_id=home_batch_id, actual_target_exec_seq_ids=[1, 3] if seq_ids is None else seq_ids)


def make_seqs():
    return [FakeSeq(1, [7, 101]), FakeSeq(3, [8, 9, 10, 301])]


def build_all(predicted, *, mailbox=None, seqs=None, step_plan=None):
    mailbox = make_mailbox() if mailbox is None else mailbox
    seqs = make_seqs() if seqs is None else seqs
    step_plan = make_step_plan() if step_plan is None else step_plan
    verify_result = build_mailbox_verify_result(make_input(step_plan.target_home_batch_id), target_token_ids=predicted, output_owner_rank=10)
    apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, step_plan, max_model_len=32)
    commit_plan = build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, step_plan, commit_allowed=True)
    kv_plan = build_mailbox_kv_commit_plan(verify_result, commit_plan, seqs, step_plan, max_model_len=32, commit_allowed=True)
    consume_plan = build_mailbox_payload_consume_plan(commit_plan, mailbox)
    return mailbox, seqs, commit_plan, kv_plan, consume_plan


def test_all_accepted_consume_and_next_pipeline_metadata():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 303, 304])
    result = run_mailbox_verify_commit_probe(
        commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan,
        payload_consume_plan=consume_plan, mailbox=mailbox, next_pipeline_plan_id=10,
    )

    assert result.success is True
    assert result.mailbox_payload_consume_success is True
    assert result.mailbox_payload_invalidate_success is True
    assert result.mailbox_payload_consumed_payload_ids == ["1:1:0:1", "1:3:1:4"]
    assert result.mailbox_payload_consumed_token_count == 5
    assert result.next_required_feature == "next_pipeline_step_after_mailbox_commit"
    assert result.next_pipeline_step_attempted is True
    assert result.next_pipeline_plan_id == 10
    after = mailbox.lifecycle_snapshot(["1:1:0:1", "1:3:1:4"])
    assert {row["lifecycle_state"] for row in after.values()} == {"consumed"}


def test_partial_accepted_consumes_prefix_and_invalidates_suffix():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 999, 999])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan, payload_consume_plan=consume_plan, mailbox=mailbox)

    assert result.success is True
    assert result.mailbox_payload_consumed_payload_ids == ["1:1:0:1", "1:3:1:4"]
    assert result.mailbox_payload_invalidated_payload_ids == ["1:3:1:4"]
    assert result.mailbox_payload_consumed_token_count == 3
    assert result.mailbox_payload_invalidated_token_count == 2
    state = mailbox.lifecycle_snapshot(["1:3:1:4"])["1:3:1:4"]
    assert state["lifecycle_state"] == "invalidated"
    assert state["consumed_token_count"] == 2
    assert state["invalidated_token_count"] == 2


def test_all_rejected_invalidates_payloads_without_consume():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([0, 0, 0, 0, 0])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan, payload_consume_plan=consume_plan, mailbox=mailbox)

    assert result.success is True
    assert result.mailbox_payload_consumed_payload_ids == []
    assert set(result.mailbox_payload_invalidated_payload_ids) == {"1:1:0:1", "1:3:1:4"}
    assert result.mailbox_payload_invalidated_token_count == 5
    assert {row["lifecycle_state"] for row in mailbox.lifecycle_snapshot().values()} == {"invalidated"}


def test_duplicate_consume_fails_and_consumed_payload_cannot_be_consumed_again():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 303, 304])
    first = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan, payload_consume_plan=consume_plan, mailbox=mailbox)
    assert first.success is True
    with pytest.raises(MailboxVerifyApplyError, match="duplicate consume"):
        build_mailbox_payload_consume_plan(commit_plan, mailbox)


def test_wrong_home_batch_and_wrong_seq_id_fail_plan_validation():
    mailbox = make_mailbox()
    _, _, commit_plan, _, _ = build_all([101, 301, 302, 303, 304], mailbox=mailbox)
    bad_home = type(commit_plan)(**{**commit_plan.__dict__, "target_home_batch_id": 2})
    with pytest.raises(MailboxVerifyApplyError, match="home_batch_id mismatch"):
        build_mailbox_payload_consume_plan(bad_home, mailbox)
    bad_seq = type(commit_plan)(**{**commit_plan.__dict__, "seq_ids": [1, 4]})
    with pytest.raises(MailboxVerifyApplyError, match="seq_id mismatch|missing lifecycle"):
        build_mailbox_payload_consume_plan(bad_seq, mailbox)


def test_non_owner_skip_does_not_mutate_mailbox():
    mailbox, _, _, _, consume_plan = build_all([101, 301, 302, 303, 304])
    before = mailbox.lifecycle_snapshot()
    result = run_mailbox_payload_consume_probe(consume_plan, mailbox, current_rank=11, output_owner_rank=10)

    assert result.success is True
    assert result.skipped_non_owner is True
    assert result.invalidate_skipped_non_owner is True
    assert mailbox.lifecycle_snapshot() == before


def test_rollback_restores_mailbox_lifecycle_after_consume_failure():
    class FailingMailbox(STSpecPayloadMailbox):
        def apply_payload_lifecycle(self, **kwargs):
            super().apply_payload_lifecycle(**kwargs)
            raise RuntimeError("injected consume failure")

    mailbox = FailingMailbox()
    mailbox.put_payloads(1, [payload(1, [101], offset=0), payload(3, [301, 302, 303, 304], offset=1)])
    _, _, _, _, consume_plan = build_all([101, 301, 302, 303, 304], mailbox=mailbox)
    before = mailbox.lifecycle_snapshot()
    result = run_mailbox_payload_consume_probe(consume_plan, mailbox, current_rank=10, output_owner_rank=10)

    assert result.success is False
    assert result.rollback_attempted is True
    assert result.rollback_success is True
    assert mailbox.lifecycle_snapshot() == before


def test_consume_plan_and_result_json_serializable():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 999, 999])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan, payload_consume_plan=consume_plan, mailbox=mailbox)

    json.dumps(consume_plan.to_dict(), sort_keys=True, default=str)
    json.dumps(result.to_dict(), sort_keys=True, default=str)


def test_continue_after_commit_missing_payload_advances_diagnostic():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 303, 304])
    seqs[0].is_finished = False
    seqs[1].is_finished = False
    result = run_mailbox_verify_commit_probe(
        commit_plan,
        seqs,
        current_rank=10,
        output_owner_rank=10,
        kv_commit_plan=kv_plan,
        payload_consume_plan=consume_plan,
        mailbox=mailbox,
        continue_after_commit=True,
        continuation_context={"active_seq_ids": [1, 3]},
    )
    assert result.success is True
    assert result.second_step_state_check_attempted is True
    assert result.second_step_state_check_success is True
    assert result.next_required_feature in {
        "active_request_continuation_after_breadth_only_step",
        "result_finalization_after_breadth_only_completion",
    }
    assert result.next_pipeline_step_error_kind == "active_request_continuation_after_breadth_only_step"


def test_continue_after_commit_all_finished_sets_breadth_completed():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 303, 304])
    result = run_mailbox_verify_commit_probe(
        commit_plan,
        seqs,
        current_rank=10,
        output_owner_rank=10,
        kv_commit_plan=kv_plan,
        payload_consume_plan=consume_plan,
        mailbox=mailbox,
        continue_after_commit=True,
        continuation_context={"active_seq_ids": []},
    )
    assert result.success is True
    assert result.breadth_only_completed is True
    assert result.breadth_only_completion_reason == "all_requests_finished"
    assert result.next_required_feature == "end_to_end_breadth_only_completion"


def test_continue_after_commit_pending_payload_sets_mailbox_drain_diagnostic():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 303, 304])
    # inject a pending available payload not in consumed set
    mailbox.put_payloads(1, [payload(9, [901], offset=5)])
    result = run_mailbox_verify_commit_probe(
        commit_plan,
        seqs,
        current_rank=10,
        output_owner_rank=10,
        kv_commit_plan=kv_plan,
        payload_consume_plan=consume_plan,
        mailbox=mailbox,
        continue_after_commit=True,
        continuation_context={"active_seq_ids": [], "pending_mailbox_payload_ids": ["1:9:5:1"]},
    )
    assert result.success is True
    assert result.breadth_only_completed is False
    assert result.next_required_feature == "mailbox_drain_after_breadth_only_completion"


def test_continue_after_commit_scheduler_mismatch_sets_specific_diagnostic():
    mailbox, seqs, commit_plan, kv_plan, consume_plan = build_all([101, 301, 302, 303, 304])
    result = run_mailbox_verify_commit_probe(
        commit_plan,
        seqs,
        current_rank=10,
        output_owner_rank=10,
        kv_commit_plan=kv_plan,
        payload_consume_plan=consume_plan,
        mailbox=mailbox,
        continue_after_commit=True,
        continuation_context={"active_seq_ids": [1], "scheduler_active_seq_ids": [3]},
    )
    assert result.success is True
    assert result.next_required_feature == "scheduler_state_after_breadth_only_completion"
