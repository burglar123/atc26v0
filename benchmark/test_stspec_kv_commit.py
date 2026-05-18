"""CPU-safe V4N mailbox KV shadow commit/rollback probe tests."""

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
run_mailbox_verify_commit_probe = apply_mod.run_mailbox_verify_commit_probe


class FakeSeq:
    def __init__(self, seq_id: int, tokens: list[int], *, home_batch_id: int = 1, block_table=None):
        self.seq_id = seq_id
        self.request_id = f"r{seq_id}"
        self.token_ids = list(tokens)
        self.num_tokens = len(tokens)
        self.num_prompt_tokens = len(tokens)
        self.last_token = tokens[-1]
        self.home_batch_id = home_batch_id
        self.block_table = [seq_id] if block_table is None else list(block_table)
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


def make_input(home_batch_id: int = 1, *, kv_slot_ids=None):
    return TargetForwardFromMailboxInput(
        plan_id=17,
        target_home_batch_id=home_batch_id,
        seq_ids=[1, 3],
        request_ids=["a", "b"],
        input_token_ids=[101, 301, 302, 303, 304],
        per_seq_lengths=[1, 4],
        offsets=[0, 1],
        total_tokens=5,
        gamma=4,
        positions=[2, 4, 5, 6, 7],
        kv_slot_ids=[1010, 3010, 3020, 3030, 3040] if kv_slot_ids is None else kv_slot_ids,
        source_mailbox_payload_ids=["1:1:0:1", "1:3:1:4"],
        source_draft_plan_id=17,
    )


def make_step_plan(home_batch_id: int = 1, seq_ids=None):
    return SimpleNamespace(
        plan_id=17,
        target_home_batch_id=home_batch_id,
        actual_target_exec_seq_ids=[1, 3] if seq_ids is None else seq_ids,
    )


def make_seqs(*, seq2_home_batch_id=1, block_table=True):
    bt1 = [11] if block_table else []
    bt3 = [33] if block_table else []
    return [FakeSeq(1, [7, 101], block_table=bt1), FakeSeq(3, [8, 9, 10, 301], home_batch_id=seq2_home_batch_id, block_table=bt3)]


def build_plans(predicted, *, seqs=None, step_plan=None, kv_slot_ids=None):
    seqs = make_seqs() if seqs is None else seqs
    step_plan = make_step_plan() if step_plan is None else step_plan
    verify_result = build_mailbox_verify_result(
        make_input(step_plan.target_home_batch_id, kv_slot_ids=kv_slot_ids),
        target_token_ids=predicted,
        output_owner_rank=10,
    )
    apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, step_plan, max_model_len=32)
    commit_plan = build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, step_plan, commit_allowed=True)
    kv_plan = build_mailbox_kv_commit_plan(verify_result, commit_plan, seqs, step_plan, max_model_len=32, commit_allowed=True)
    return verify_result, apply_plan, commit_plan, kv_plan, seqs


def test_all_accepted_kv_commit_plan_and_shadow_success():
    _, _, commit_plan, kv_plan, seqs = build_plans([101, 301, 302, 303, 304])
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan)

    assert kv_plan.accepted_lengths_by_seq == {1: 1, 3: 4}
    assert kv_plan.append_start_positions_by_seq == {1: 0, 3: 0}
    assert kv_plan.append_end_positions_by_seq == {1: 1, 3: 4}
    assert kv_plan.sequence_length_after_by_seq == {1: 3, 3: 8}
    assert result.success is True
    assert result.kv_commit_success is True
    assert result.kv_commit_shadow_only is True
    assert result.next_required_feature == "mailbox_payload_consume_invalidate"
    assert getattr(seqs[1], "stspec_mailbox_kv_shadow")["length"] == 8


def test_partial_accept_and_all_reject_kv_plans():
    _, _, _, partial, _ = build_plans([101, 301, 302, 999, 999])
    _, _, _, rejected, _ = build_plans([0, 0, 0, 0, 0])

    assert partial.accepted_lengths_by_seq == {1: 1, 3: 2}
    assert partial.kv_positions_by_seq[3] == [4, 5]
    assert rejected.accepted_lengths_by_seq == {1: 0, 3: 0}
    assert rejected.kv_positions_by_seq == {1: [], 3: []}


def test_wrong_seq_id_and_home_batch_fail():
    seqs = make_seqs()
    verify_result = build_mailbox_verify_result(make_input(), target_token_ids=[101, 301, 302, 303, 304])
    with pytest.raises(MailboxVerifyApplyError, match="seq ids mismatch"):
        apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan(seq_ids=[3, 1]))
        commit_plan = build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, make_step_plan(seq_ids=[3, 1]), commit_allowed=True)
        build_mailbox_kv_commit_plan(verify_result, commit_plan, seqs, make_step_plan(seq_ids=[3, 1]), commit_allowed=True)
    with pytest.raises(MailboxVerifyApplyError, match="home_batch_id mismatch"):
        build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan(home_batch_id=2))


def test_accepted_length_and_append_position_validation_fail():
    verify_result, apply_plan, commit_plan, _, seqs = build_plans([101, 301, 302, 303, 304])
    bad_verify = type(verify_result)(**{**verify_result.__dict__, "accepted_lengths_by_seq": {1: 2, 3: 4}})
    with pytest.raises(MailboxVerifyApplyError, match="accepted length out of range"):
        build_mailbox_verify_apply_plan(bad_verify, seqs, make_step_plan())
    bad_commit = type(commit_plan)(**{**commit_plan.__dict__, "accepted_lengths_by_seq": {1: 1, 3: 5}})
    with pytest.raises(MailboxVerifyApplyError, match="accepted length out of range"):
        build_mailbox_kv_commit_plan(verify_result, bad_commit, seqs, make_step_plan(), commit_allowed=True)
    seqs[0].append_token(999)
    with pytest.raises(MailboxVerifyApplyError, match="append position mismatch|stale Sequence snapshot"):
        # Sequence commit rejects the stale append position before KV commit can run.
        build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, make_step_plan(), commit_allowed=True)


def test_missing_slot_mapping_gets_specific_diagnostic():
    seqs = make_seqs(block_table=False)
    verify_result = build_mailbox_verify_result(make_input(kv_slot_ids=[]), target_token_ids=[101, 301, 302, 303, 304])
    apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan())
    commit_plan = build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, make_step_plan(), commit_allowed=True)
    with pytest.raises(MailboxVerifyApplyError) as exc:
        build_mailbox_kv_commit_plan(verify_result, commit_plan, seqs, make_step_plan(), commit_allowed=True)
    assert exc.value.next_required_feature == "kv_slot_mapping_after_mailbox_verify"


def test_kv_failure_rolls_back_sequence_and_shadow_metadata():
    _, _, commit_plan, kv_plan, seqs = build_plans([101, 301, 302, 303, 304])
    seqs[1].fail_kv_commit = True
    before_tokens = [list(seq.token_ids) for seq in seqs]
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=10, output_owner_rank=10, kv_commit_plan=kv_plan)

    assert result.success is False
    assert result.kv_commit_rollback_attempted is True
    assert result.rollback_success is True
    assert [seq.token_ids for seq in seqs] == before_tokens
    assert not hasattr(seqs[0], "stspec_mailbox_kv_shadow")
    assert not hasattr(seqs[1], "stspec_mailbox_kv_shadow")
    assert result.next_required_feature == "kv_commit_rollback_validation"


def test_non_owner_skip_and_json_serialization():
    _, _, commit_plan, kv_plan, seqs = build_plans([101, 301, 302, 303, 304])
    before_tokens = [list(seq.token_ids) for seq in seqs]
    result = run_mailbox_verify_commit_probe(commit_plan, seqs, current_rank=11, output_owner_rank=10, kv_commit_plan=kv_plan)

    assert result.success is True
    assert result.skipped_non_owner is True
    assert result.kv_commit_skipped_non_owner is True
    assert [seq.token_ids for seq in seqs] == before_tokens
    json.dumps(kv_plan.to_dict(), sort_keys=True, default=str)
    json.dumps(result.to_dict(), sort_keys=True, default=str)
