"""CPU-safe V4L mailbox verify apply no-commit probe tests."""

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
build_mailbox_verify_apply_plan = apply_mod.build_mailbox_verify_apply_plan
build_mailbox_verify_result = apply_mod.build_mailbox_verify_result
extract_target_token_ids_from_logits = apply_mod.extract_target_token_ids_from_logits
run_mailbox_verify_apply_no_commit_probe = apply_mod.run_mailbox_verify_apply_no_commit_probe


class FakeSeq:
    def __init__(self, seq_id: int, tokens: list[int], home_batch_id: int = 1):
        self.seq_id = seq_id
        self.request_id = f"r{seq_id}"
        self.token_ids = list(tokens)
        self.num_tokens = len(tokens)
        self.last_token = tokens[-1]
        self.block_table = [seq_id]
        self.home_batch_id = home_batch_id
        self.is_finished = False

    def __len__(self):
        return len(self.token_ids)


class FakeArgmax:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class FakeLogits:
    def __init__(self, predicted):
        self.predicted = list(predicted)
        self.shape = (len(self.predicted), 10)

    def argmax(self, dim=-1):
        return FakeArgmax(self.predicted)


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


def make_seqs():
    return [FakeSeq(1, [7, 101]), FakeSeq(3, [8, 9, 10, 301])]


def build_plan_for_predictions(predicted):
    target_input = make_input()
    verify_result = build_mailbox_verify_result(
        target_input,
        target_token_ids=predicted,
        output_owner_rank=10,
    )
    apply_plan = build_mailbox_verify_apply_plan(verify_result, make_seqs(), make_step_plan(), max_model_len=32)
    return verify_result, apply_plan


def test_all_accepted_prefix_consumes_payloads_json():
    verify_result, apply_plan = build_plan_for_predictions([101, 301, 302, 303, 304])

    assert verify_result.accepted_lengths_by_seq == {1: 1, 3: 4}
    assert verify_result.rejected_seq_ids == []
    assert verify_result.total_accepted_tokens == 5
    assert verify_result.total_rejected_tokens == 0
    assert apply_plan.accepted_token_ids_by_seq == {1: [101], 3: [301, 302, 303, 304]}
    assert apply_plan.rejected_token_ids_by_seq == {1: [], 3: []}
    assert apply_plan.mailbox_payloads_to_consume == ["1:1:0:1", "1:3:1:4"]
    assert apply_plan.mailbox_payloads_to_invalidate == []
    json.dumps(verify_result.to_dict(), sort_keys=True)
    json.dumps(apply_plan.to_dict(), sort_keys=True)


def test_first_token_mismatch_rejects_entire_payload():
    verify_result, apply_plan = build_plan_for_predictions([999, 301, 302, 303, 304])

    assert verify_result.accepted_lengths_by_seq[1] == 0
    assert 1 in verify_result.rejected_seq_ids
    assert apply_plan.rejected_token_ids_by_seq[1] == [101]
    assert "1:1:0:1" in apply_plan.mailbox_payloads_to_invalidate


def test_middle_token_mismatch_rejects_suffix():
    verify_result, apply_plan = build_plan_for_predictions([101, 301, 302, 999, 304])

    assert verify_result.accepted_lengths_by_seq[3] == 2
    assert verify_result.rejected_token_positions_by_seq[3] == [2, 3]
    assert apply_plan.accepted_token_ids_by_seq[3] == [301, 302]
    assert apply_plan.rejected_token_ids_by_seq[3] == [303, 304]


def test_all_reject():
    verify_result, apply_plan = build_plan_for_predictions([0, 0, 0, 0, 0])

    assert verify_result.accepted_lengths_by_seq == {1: 0, 3: 0}
    assert set(verify_result.rejected_seq_ids) == {1, 3}
    assert verify_result.total_rejected_tokens == 5
    assert apply_plan.mailbox_payloads_to_consume == []
    assert set(apply_plan.mailbox_payloads_to_invalidate) == {"1:1:0:1", "1:3:1:4"}


def test_accepted_lengths_never_exceed_drafted_length_and_wrong_ids_fail():
    target_input = make_input()
    with pytest.raises(MailboxVerifyApplyError, match="target token length mismatch"):
        build_mailbox_verify_result(target_input, target_token_ids=[101])

    verify_result = build_mailbox_verify_result(target_input, target_token_ids=[101, 301, 302, 303, 304])
    with pytest.raises(MailboxVerifyApplyError, match="seq ids mismatch"):
        build_mailbox_verify_apply_plan(verify_result, make_seqs(), make_step_plan(seq_ids=[3, 1]))
    with pytest.raises(MailboxVerifyApplyError, match="home_batch_id mismatch"):
        build_mailbox_verify_apply_plan(verify_result, make_seqs(), make_step_plan(home_batch_id=2))


def test_no_commit_probe_does_not_mutate_fake_sequence_state():
    verify_result = build_mailbox_verify_result(make_input(), target_token_ids=[101, 301, 0, 0, 0])
    seqs = make_seqs()
    before = [(seq.seq_id, list(seq.token_ids), seq.num_tokens, list(seq.block_table)) for seq in seqs]
    apply_plan = build_mailbox_verify_apply_plan(verify_result, seqs, make_step_plan())
    probe = run_mailbox_verify_apply_no_commit_probe(apply_plan, seqs)
    after = [(seq.seq_id, list(seq.token_ids), seq.num_tokens, list(seq.block_table)) for seq in seqs]

    assert probe.success is True
    assert probe.state_mutation_committed is False
    assert before == after


def test_metadata_only_token_decision_points_to_backend():
    verify_result = build_mailbox_verify_result(make_input(), metadata_only=True, output_owner_rank=10)
    apply_plan = build_mailbox_verify_apply_plan(verify_result, make_seqs(), make_step_plan())

    assert verify_result.metadata_only is True
    assert verify_result.next_required_feature == "mailbox_verify_token_decision_backend"
    assert apply_plan.no_commit is True


def test_non_owner_rank_skip_trace_shape():
    trace = {}
    trace["mailbox_verify_apply_skipped_non_owner"] = True
    trace["mailbox_forward_state_mutation_committed"] = False
    json.dumps(trace, sort_keys=True)


def test_extract_target_token_ids_from_logits():
    assert extract_target_token_ids_from_logits(FakeLogits([1, 2, 3]), 3) == [1, 2, 3]
    assert extract_target_token_ids_from_logits(object(), 3) is None
