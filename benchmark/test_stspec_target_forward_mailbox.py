"""CPU-safe V4H target-forward-from-mailbox scaffolding tests."""

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

payloads_from_draft_message = mailbox_mod.payloads_from_draft_message
encode_variable_draft_message = protocol_mod.encode_variable_draft_message
TargetForwardFromMailboxInput = transport_mod.TargetForwardFromMailboxInput
build_verification_input_from_mailbox_payload = transport_mod.build_verification_input_from_mailbox_payload
interpret_target_forward_from_mailbox_output = transport_mod.interpret_target_forward_from_mailbox_output
map_target_forward_output_rows_to_seq_offsets = transport_mod.map_target_forward_output_rows_to_seq_offsets
validate_kv_state_sync_for_mailbox_forward = transport_mod.validate_kv_state_sync_for_mailbox_forward
validate_target_forward_from_mailbox_input = transport_mod.validate_target_forward_from_mailbox_input


class Seq:
    def __init__(self, seq_id: int, request_id: str, tokens: list[int], slots: list[int] | None = None):
        self.seq_id = seq_id
        self.request_id = request_id
        self.token_ids = list(tokens)
        self._slots = slots

    def __len__(self) -> int:
        return len(self.token_ids)

    def token_to_slot(self, token_index: int) -> int:
        if self._slots is None:
            raise RuntimeError("no kv mapping")
        return self._slots[token_index]


def make_message():
    seqs = [
        SimpleNamespace(seq_id=1, request_id="a", pre_verify=True),
        SimpleNamespace(seq_id=3, request_id="b", pre_verify=False),
    ]
    return encode_variable_draft_message(
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


def make_payloads(home_batch_id: int = 1):
    return payloads_from_draft_message(
        make_message(),
        home_batch_id=home_batch_id,
        target_home_batch_id=home_batch_id,
        draft_home_batch_id=home_batch_id,
        producer_home_batch_id=home_batch_id,
        logical_step=9,
    )


def make_step_plan(home_batch_id: int = 1):
    return SimpleNamespace(
        plan_id=9,
        target_home_batch_id=home_batch_id,
        scheduled_seq_ids=[0, 1, 2, 3],
        actual_target_exec_seq_ids=[1, 3],
    )


def make_seqs(with_slots: bool = True):
    return [
        Seq(1, "a", [7, 101], [70, 1010] if with_slots else None),
        Seq(3, "b", [8, 9, 10, 301, 302, 303, 304], [80, 90, 100, 3010, 3020, 3030, 3040] if with_slots else None),
    ]


def test_build_target_forward_input_from_mailbox_payload_json_serializable():
    target_input = build_verification_input_from_mailbox_payload(make_payloads(), make_seqs(), make_step_plan(), 4)

    assert isinstance(target_input, TargetForwardFromMailboxInput)
    assert target_input.plan_id == 9
    assert target_input.target_home_batch_id == 1
    assert target_input.seq_ids == [1, 3]
    assert target_input.request_ids == ["a", "b"]
    assert target_input.input_token_ids == [101, 301, 302, 303, 304]
    assert target_input.per_seq_lengths == [1, 4]
    assert target_input.offsets == [0, 1]
    assert target_input.total_tokens == 5
    assert target_input.gamma == 4
    assert target_input.positions == [1, 3, 4, 5, 6]
    assert target_input.kv_slot_ids == [1010, 3010, 3020, 3030, 3040]
    assert target_input.layout_kind == "variable_offsets"
    json.dumps(target_input.to_dict(), sort_keys=True)


def test_target_forward_input_seq_match_and_wrong_seq_validation():
    target_input = build_verification_input_from_mailbox_payload(make_payloads(), make_seqs(), make_step_plan(), 4)
    validate_target_forward_from_mailbox_input(
        target_input,
        actual_target_exec_seq_ids=[1, 3],
        target_scheduler_seq_ids=[0, 1, 2, 3],
        scheduled_seq_ids=[0, 1, 2, 3],
        target_home_batch_id=1,
    )

    with pytest.raises(RuntimeError, match="actual_target_exec_seq_ids"):
        validate_target_forward_from_mailbox_input(target_input, actual_target_exec_seq_ids=[3, 1])

    with pytest.raises(RuntimeError, match="missing from target scheduler"):
        validate_target_forward_from_mailbox_input(
            target_input,
            actual_target_exec_seq_ids=[1, 3],
            target_scheduler_seq_ids=[1],
        )


def test_wrong_home_batch_and_bad_offsets_fail():
    target_input = build_verification_input_from_mailbox_payload(make_payloads(), make_seqs(), make_step_plan(), 4)

    with pytest.raises(RuntimeError, match="home_batch_id mismatch"):
        validate_target_forward_from_mailbox_input(
            target_input,
            actual_target_exec_seq_ids=[1, 3],
            target_home_batch_id=2,
        )

    bad = TargetForwardFromMailboxInput(**{**target_input.to_dict(), "offsets": [0, 2]})
    with pytest.raises(RuntimeError, match="offset"):
        validate_target_forward_from_mailbox_input(bad, actual_target_exec_seq_ids=[1, 3])


def test_missing_kv_mapping_classified_as_kv_state_sync_error():
    target_input = build_verification_input_from_mailbox_payload(make_payloads(), make_seqs(with_slots=False), make_step_plan(), 4)
    status = validate_kv_state_sync_for_mailbox_forward(target_input, make_seqs(with_slots=False))

    assert status.attempted is True
    assert status.success is False
    assert status.error_kind == "kv_position_mapping_not_implemented"
    assert "KV/state synchronization" in status.error
    json.dumps(status.to_dict(), sort_keys=True)


def test_output_row_to_seq_offset_mapping_scaffold_and_interpretation_guard():
    target_input = build_verification_input_from_mailbox_payload(make_payloads(), make_seqs(), make_step_plan(), 4)
    mapping = map_target_forward_output_rows_to_seq_offsets(target_input, output_shape=[5, 32000])

    assert mapping == [
        {"seq_id": 1, "offset": 0, "length": 1, "row_start": 0, "row_end": 1},
        {"seq_id": 3, "offset": 1, "length": 4, "row_start": 1, "row_end": 5},
    ]
    with pytest.raises(RuntimeError, match="output interpretation is not implemented"):
        interpret_target_forward_from_mailbox_output(target_input, output_shape=[5, 32000])


def test_illegal_legacy_fallback_detection():
    legacy_full_batch_input = TargetForwardFromMailboxInput(
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
        validate_target_forward_from_mailbox_input(
            legacy_full_batch_input,
            actual_target_exec_seq_ids=[1, 3],
            scheduled_seq_ids=[0, 1, 2, 3],
        )
