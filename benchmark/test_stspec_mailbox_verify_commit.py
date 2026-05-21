"""CPU-safe V4M mailbox verify guarded commit/rollback probe tests."""

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

transport_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_transport")
apply_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_verify_apply")

TargetForwardFromMailboxInput = transport_mod.TargetForwardFromMailboxInput
MailboxVerifyApplyError = apply_mod.MailboxVerifyApplyError
build_mailbox_kv_commit_plan = apply_mod.build_mailbox_kv_commit_plan
build_mailbox_verify_apply_plan = apply_mod.build_mailbox_verify_apply_plan
build_mailbox_verify_commit_plan = apply_mod.build_mailbox_verify_commit_plan
build_mailbox_verify_result = apply_mod.build_mailbox_verify_result
run_mailbox_verify_apply_no_commit_probe = apply_mod.run_mailbox_verify_apply_no_commit_probe
run_mailbox_verify_commit_probe = apply_mod.run_mailbox_verify_commit_probe


class FakeSeq:
    def __init__(self, seq_id: int, tokens: list[int], home_batch_id: int = 1):
        self.seq_id = seq_id
        self.request_id = f"r{seq_id}"
        self.token_ids = list(tokens)
        self.num_tokens = len(tokens)
        self.num_prompt_tokens = len(tokens)
        self.last_token = tokens[-1]
        self.block_table = [seq_id]
        self.home_batch_id = home_batch_id
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


class FailingSeq(FakeSeq):
    def __init__(self, *args, fail_on_token: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_on_token = fail_on_token

    def append_token(self, token_id: int):
        if int(token_id) == self.fail_on_token:
            raise MailboxVerifyApplyError(
                "injected append failure",
                next_required_feature="mailbox_commit_rollback_validation",
                error_kind="injected_append_failure",
            )
        super().append_token(token_id)


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
        positions=[1, 3, 4, 5, 6],
        kv_slot_ids=[1010, 3010, 3020, 3030, 3040],
        source_mailbox_payload_ids=["1:1:0:1", "1:3:1:4"],
        source_draft_plan_id=9,
    )


def make_step_plan(home_batch_id: int = 1, seq_ids=None):
    return SimpleNamespace(
        plan_id=9,
        target_home_batch_id=home_batch_id,
        actual_target_exec_seq_ids=[1, 3] if seq_ids is None else seq_ids,
    )


def make_seqs(seq2_cls=FakeSeq, seq2_home_batch_id: int = 1):
    if seq2_cls is FailingSeq:
        return [FakeSeq(1, [7, 101]), FailingSeq(3, [8, 9, 10, 301], home_batch_id=seq2_home_batch_id, fail_on_token=303)]
    return [FakeSeq(1, [7, 101]), seq2_cls(3, [8, 9, 10, 301], home_batch_id=seq2_home_batch_id)]


def build_commit(predicted, *, seqs=None, step_plan=None):
    seqs = make_seqs() if seqs is None else seqs
    step_plan = make_step_plan() if step_plan is None else step_plan
    verify_result = build_mailbox_verify_result(make_input(step_plan.target_home_batch_id), target_token_ids=predicted, output_owner_rank=10)
    apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, step_plan, max_model_len=32)
    commit_plan = build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, step_plan, commit_allowed=True)
    kv_commit_plan = build_mailbox_kv_commit_plan(
        verify_result, commit_plan, seqs, step_plan, max_model_len=32, commit_allowed=True
    )
    return verify_result, apply_plan, commit_plan, kv_commit_plan, seqs


def test_all_accepted_commit_mutates_sequences_and_points_to_kv_next_feature():
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([101, 301, 302, 303, 304])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_commit_plan)

    assert result.attempted is True
    assert result.success is True
    assert result.sequence_state_commit_success is True
    assert result.next_required_feature == "mailbox_payload_consume_invalidate"
    assert result.kv_commit_success is True
    assert result.kv_commit_shadow_only is True
    assert seqs[0].token_ids == [7, 101, 101]
    assert seqs[1].token_ids == [8, 9, 10, 301, 301, 302, 303, 304]
    assert result.kv_commit_attempted is True


def test_partial_accepted_reject_appends_target_correction_and_plans_invalidate():
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([101, 301, 302, 999, 999])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_commit_plan)

    assert result.success is True
    assert commit_plan.accepted_lengths_by_seq == {1: 1, 3: 2}
    assert commit_plan.rejected_seq_ids == [3]
    assert commit_plan.rejected_token_ids_by_seq[3] == [303, 304]
    assert commit_plan.target_correction_token_ids_by_seq[3] == [999]
    assert commit_plan.mailbox_payloads_to_consume == ["1:1:0:1"]
    assert commit_plan.mailbox_payloads_to_invalidate == ["1:3:1:4"]
    assert seqs[1].token_ids == [8, 9, 10, 301, 301, 302, 999]


def test_all_rejected_commit_appends_target_correction_not_rejected_draft():
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([0, 0, 0, 0, 0])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_commit_plan)

    assert result.success is True
    assert commit_plan.accepted_lengths_by_seq == {1: 0, 3: 0}
    assert [seq.token_ids for seq in seqs] == [[7, 101, 0], [8, 9, 10, 301, 0]]
    assert set(commit_plan.mailbox_payloads_to_invalidate) == {"1:1:0:1", "1:3:1:4"}


def test_wrong_seq_id_and_wrong_home_batch_fail_validation():
    seqs = make_seqs()
    verify_result = build_mailbox_verify_result(make_input(), target_token_ids=[101, 301, 302, 303, 304])
    with pytest.raises(MailboxVerifyApplyError, match="seq ids mismatch"):
        apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan(seq_ids=[3, 1]))
        build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, make_step_plan(seq_ids=[3, 1]), commit_allowed=True)
    with pytest.raises(MailboxVerifyApplyError, match="home_batch_id mismatch"):
        build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan(home_batch_id=2))


def test_sequence_home_batch_mismatch_fails_commit_plan():
    seqs = make_seqs(seq2_home_batch_id=2)
    verify_result = build_mailbox_verify_result(make_input(), target_token_ids=[101, 301, 302, 303, 304])
    apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan())
    with pytest.raises(MailboxVerifyApplyError, match="Sequence home_batch_id mismatch"):
        build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, make_step_plan(), commit_allowed=True)


def test_accepted_length_larger_than_drafted_length_fails():
    seqs = make_seqs()
    verify_result = build_mailbox_verify_result(make_input(), target_token_ids=[101, 301, 302, 303, 304])
    verify_result = type(verify_result)(**{**verify_result.__dict__, "accepted_lengths_by_seq": {1: 2, 3: 4}})
    with pytest.raises(MailboxVerifyApplyError, match="accepted length out of range"):
        build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan())


def test_no_commit_mode_does_not_mutate():
    _, apply_plan, _, _, seqs = build_commit([101, 301, 302, 303, 304])
    before = [list(seq.token_ids) for seq in seqs]
    result = run_mailbox_verify_apply_no_commit_probe(apply_plan, seqs)

    assert result.success is True
    assert [seq.token_ids for seq in seqs] == before


def test_commit_disabled_does_not_mutate():
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([101, 301, 302, 303, 304])
    before = [list(seq.token_ids) for seq in seqs]
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, commit_enabled=False, kv_commit_plan=kv_commit_plan)

    assert result.attempted is False
    assert result.success is True
    assert [seq.token_ids for seq in seqs] == before


def test_rollback_restores_sequence_after_mid_commit_failure():
    seqs = make_seqs(seq2_cls=FailingSeq)
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([101, 301, 302, 303, 304], seqs=seqs)
    before = [list(seq.token_ids) for seq in seqs]
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_commit_plan)

    assert result.success is False
    assert result.rollback_attempted is True
    assert result.rollback_success is True
    assert [seq.token_ids for seq in seqs] == before
    assert result.next_required_feature == "mailbox_commit_rollback_validation"


def test_non_owner_skip_does_not_mutate_or_error():
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([101, 301, 302, 303, 304])
    before = [list(seq.token_ids) for seq in seqs]
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=11, output_owner_rank=10, kv_commit_plan=kv_commit_plan)

    assert result.success is True
    assert result.skipped_non_owner is True
    assert result.attempted is False
    assert [seq.token_ids for seq in seqs] == before


def test_commit_plan_and_result_are_json_serializable():
    _, _, commit_plan, kv_commit_plan, seqs = build_commit([101, 301, 302, 999, 999])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_commit_plan)

    json.dumps(commit_plan.to_dict(), sort_keys=True, default=str)
    json.dumps(result.to_dict(), sort_keys=True, default=str)
