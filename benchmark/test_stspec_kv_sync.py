"""CPU-safe V4I mailbox KV/state sync plan tests."""

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
protocol_mod = importlib.import_module("nano_pearl.pearl_engine.pearl_protocol")
transport_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_transport")
kv_sync_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_kv_sync")

payloads_from_draft_message = mailbox_mod.payloads_from_draft_message
encode_variable_draft_message = protocol_mod.encode_variable_draft_message
TargetForwardFromMailboxInput = transport_mod.TargetForwardFromMailboxInput
build_verification_input_from_mailbox_payload = transport_mod.build_verification_input_from_mailbox_payload
apply_mailbox_kv_sync_plan_probe = kv_sync_mod.apply_mailbox_kv_sync_plan_probe
build_mailbox_kv_sync_plan = kv_sync_mod.build_mailbox_kv_sync_plan
validate_mailbox_kv_sync_plan = kv_sync_mod.validate_mailbox_kv_sync_plan


class Seq:
    block_size = 256

    def __init__(self, seq_id: int, request_id: str, tokens: list[int], slots: list[int] | None = None):
        self.seq_id = seq_id
        self.request_id = request_id
        self.token_ids = list(tokens)
        self._slots = slots or list(range(len(tokens)))

    def __len__(self) -> int:
        return len(self.token_ids)

    def token_to_slot(self, token_index: int) -> int:
        return self._slots[token_index]


def make_payloads(home_batch_id: int = 1):
    seqs = [
        SimpleNamespace(seq_id=1, request_id="a", pre_verify=True),
        SimpleNamespace(seq_id=3, request_id="b", pre_verify=False),
    ]
    message = encode_variable_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[101, 301, 302, 303, 304],
        next_round_input=[11, 12, 13, 14, 31, 32, 33, 34],
        plan_id=9,
        runner_role="draft",
        scheduled_seq_ids=[0, 1, 2, 3],
        actual_exec_seq_ids=[1, 3],
        target_batch_seq_ids=[0, 2],
        draft_home_batch_seq_ids=[1, 3],
    )
    return payloads_from_draft_message(
        message,
        home_batch_id=home_batch_id,
        target_home_batch_id=home_batch_id,
        draft_home_batch_id=home_batch_id,
        producer_home_batch_id=home_batch_id,
        logical_step=9,
    )


def make_step_plan():
    return SimpleNamespace(
        plan_id=9,
        target_home_batch_id=1,
        scheduled_seq_ids=[0, 1, 2, 3],
        actual_target_exec_seq_ids=[1, 3],
    )


def make_seqs():
    return [
        Seq(1, "a", [7, 101], [70, 1010]),
        Seq(3, "b", [8, 9, 10, 301, 302, 303, 304], [80, 90, 100, 3010, 3020, 3030, 3040]),
    ]


def make_input():
    return build_verification_input_from_mailbox_payload(make_payloads(), make_seqs(), make_step_plan(), 4)


def test_build_mailbox_kv_sync_plan_positions_and_json():
    plan = build_mailbox_kv_sync_plan(make_input(), make_seqs(), state_sync_mode="metadata_only", max_model_len=32)

    assert plan.seq_ids == [1, 3]
    assert plan.current_seq_lengths == {1: 2, 3: 7}
    assert plan.current_kv_positions == {1: 1, 3: 6}
    assert plan.append_start_positions == {1: 1, 3: 3}
    assert plan.append_end_positions == {1: 2, 3: 7}
    assert plan.mailbox_token_positions == [1, 3, 4, 5, 6]
    assert plan.kv_slot_ids == [1010, 3010, 3020, 3030, 3040]
    assert plan.can_append_variable_offsets is True
    assert plan.requires_kv_append is True
    validate_mailbox_kv_sync_plan(plan, expected_seq_ids=[1, 3])
    json.dumps(plan.to_dict(), sort_keys=True)
    assert plan.to_json().startswith("{")


def test_bad_offsets_and_wrong_seq_ids_fail_validation():
    plan = build_mailbox_kv_sync_plan(make_input(), make_seqs())
    bad_offsets = kv_sync_mod.MailboxKVSyncPlan(**{**plan.to_dict(), "offsets": [0, 2]})
    with pytest.raises(RuntimeError, match="offset"):
        validate_mailbox_kv_sync_plan(bad_offsets, expected_seq_ids=[1, 3])
    with pytest.raises(RuntimeError, match="seq mismatch"):
        validate_mailbox_kv_sync_plan(plan, expected_seq_ids=[3, 1])


def test_missing_sequence_state_reports_missing_seq_ids():
    plan = build_mailbox_kv_sync_plan(make_input(), [make_seqs()[0]])
    result = apply_mailbox_kv_sync_plan_probe(plan)

    assert result.attempted is True
    assert result.success is False
    assert result.missing_seq_ids == [3]
    assert result.error_kind == "missing_target_kv_state_for_mailbox_seq"


def test_metadata_only_mode_succeeds_without_mutation():
    before = [list(seq.token_ids) for seq in make_seqs()]
    seqs = make_seqs()
    plan = build_mailbox_kv_sync_plan(make_input(), seqs, state_sync_mode="metadata_only")
    result = apply_mailbox_kv_sync_plan_probe(plan, commit_enabled=False)

    assert result.success is True
    assert result.mutation_attempted is False
    assert result.mutation_committed is False
    assert [seq.token_ids for seq in seqs] == before


def test_guarded_forward_backend_unavailable_fails_clearly():
    plan = build_mailbox_kv_sync_plan(make_input(), make_seqs(), state_sync_mode="guarded_forward")
    result = apply_mailbox_kv_sync_plan_probe(plan, forward_backend_available=False)

    assert result.success is False
    assert result.error_kind == "target_forward_backend_unavailable"
    assert result.mutation_attempted is False


def test_no_commit_probe_does_not_commit_state():
    seqs = make_seqs()
    before = [list(seq.token_ids) for seq in seqs]
    plan = build_mailbox_kv_sync_plan(make_input(), seqs, state_sync_mode="no_commit_probe")
    result = apply_mailbox_kv_sync_plan_probe(plan, commit_enabled=False, forward_backend_available=True)

    assert result.success is True
    assert result.mutation_attempted is False
    assert result.mutation_committed is False
    assert [seq.token_ids for seq in seqs] == before


def test_illegal_legacy_fallback_still_detected_before_kv_sync():
    target_input = TargetForwardFromMailboxInput(
        plan_id=9,
        target_home_batch_id=1,
        seq_ids=[0, 1, 2, 3],
        request_ids=["z", "a", "y", "b"],
        input_token_ids=[1, 101, 2, 301],
        per_seq_lengths=[1, 1, 1, 1],
        offsets=[0, 1, 2, 3],
        total_tokens=4,
        gamma=4,
        positions=[0, 0, 0, 0],
        kv_slot_ids=[0, 1, 2, 3],
        source_mailbox_payload_ids=["0", "1", "2", "3"],
        source_draft_plan_id=9,
    )
    with pytest.raises(RuntimeError, match="Illegal legacy fallback"):
        transport_mod.validate_target_forward_from_mailbox_input(
            target_input,
            actual_target_exec_seq_ids=[1, 3],
            scheduled_seq_ids=[0, 1, 2, 3],
        )
