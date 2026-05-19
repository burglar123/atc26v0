"""CPU-safe V4J mailbox target-forward context builder tests."""

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
context_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_forward_context")

payloads_from_draft_message = mailbox_mod.payloads_from_draft_message
encode_variable_draft_message = protocol_mod.encode_variable_draft_message
TargetForwardFromMailboxInput = transport_mod.TargetForwardFromMailboxInput
build_verification_input_from_mailbox_payload = transport_mod.build_verification_input_from_mailbox_payload
build_mailbox_kv_sync_plan = kv_sync_mod.build_mailbox_kv_sync_plan
build_target_forward_context_from_mailbox_input = context_mod.build_target_forward_context_from_mailbox_input
validate_target_forward_mailbox_context = context_mod.validate_target_forward_mailbox_context


class Seq:
    block_size = 10

    def __init__(self, seq_id: int, request_id: str, tokens: list[int], slots: list[int] | None = None):
        self.seq_id = seq_id
        self.request_id = request_id
        self.token_ids = list(tokens)
        self.num_tokens = len(tokens)
        self.block_table = [seq_id * 10]
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


def make_step_plan(home_batch_id: int = 1, actual=None, scheduled=None):
    return SimpleNamespace(
        plan_id=9,
        target_home_batch_id=home_batch_id,
        scheduled_seq_ids=[0, 1, 2, 3] if scheduled is None else scheduled,
        actual_target_exec_seq_ids=[1, 3] if actual is None else actual,
    )


def make_seqs(with_slots: bool = True):
    return [
        Seq(1, "a", [7, 101], [70, 1010] if with_slots else None),
        Seq(3, "b", [8, 9, 10, 301, 302, 303, 304], [80, 90, 100, 3010, 3020, 3030, 3040] if with_slots else None),
    ]


def make_input_and_plan(with_slots: bool = True):
    seqs = make_seqs(with_slots=with_slots)
    step_plan = make_step_plan()
    target_input = build_verification_input_from_mailbox_payload(make_payloads(), seqs, step_plan, 4)
    kv_plan = build_mailbox_kv_sync_plan(target_input, seqs, state_sync_mode="no_commit_probe")
    return target_input, kv_plan, seqs, step_plan


def test_build_context_shapes_and_json():
    target_input, kv_plan, seqs, step_plan = make_input_and_plan()
    ctx = build_target_forward_context_from_mailbox_input(target_input, kv_plan, seqs, step_plan, runner_state=SimpleNamespace(graph_bs=[1, 2, 4, 8]))

    assert ctx.can_run_model is True
    assert ctx.seq_ids == [1, 3]
    assert ctx.input_ids == [101, 301, 302, 303, 304]
    assert ctx.positions == [1, 3, 4, 5, 6]
    assert ctx.slot_mapping == [1010, 3010, 3020, 3030, 3040]
    assert ctx.input_ids_shape == [5]
    assert ctx.positions_shape == [5]
    assert ctx.slot_mapping_shape == [5]
    assert ctx.context_lens == [2, 4, 5, 6, 7]
    assert len(ctx.block_tables) == 5
    assert ctx.cuda_graph_batch_size == 8
    validate_target_forward_mailbox_context(ctx)
    json.dumps(ctx.to_dict(), sort_keys=True)


def test_missing_slot_mapping_classified_without_run_model():
    target_input, kv_plan, seqs, step_plan = make_input_and_plan(with_slots=False)
    ctx = build_target_forward_context_from_mailbox_input(target_input, kv_plan, seqs, step_plan)

    assert ctx.can_run_model is False
    assert ctx.slot_mapping_available is False
    assert ctx.cannot_run_reason == "mailbox_slot_mapping_backend"
    assert "slot_mapping" in ctx.error_message
    with pytest.raises(RuntimeError, match="mailbox_slot_mapping_backend"):
        validate_target_forward_mailbox_context(ctx)


def test_wrong_seq_ids_and_duplicate_seq_ids_fail():
    target_input, kv_plan, seqs, step_plan = make_input_and_plan()
    wrong = make_step_plan(actual=[3, 1])
    ctx = build_target_forward_context_from_mailbox_input(target_input, kv_plan, seqs, wrong)
    assert ctx.can_run_model is False
    assert ctx.cannot_run_reason == "target_forward_seq_mismatch"

    duplicate_input = TargetForwardFromMailboxInput(**{**target_input.to_dict(), "seq_ids": [1, 1]})
    duplicate_plan = SimpleNamespace(**{**kv_plan.to_dict(), "seq_ids": [1, 1]})
    dup_ctx = build_target_forward_context_from_mailbox_input(duplicate_input, duplicate_plan, [seqs[0], seqs[0]], make_step_plan(actual=[1, 1]))
    assert dup_ctx.can_run_model is False
    assert dup_ctx.cannot_run_reason == "duplicate_target_forward_seq_ids"


def test_legacy_scheduled_fallback_detection():
    target_input, kv_plan, seqs, _ = make_input_and_plan()
    legacy = TargetForwardFromMailboxInput(
        **{
            **target_input.to_dict(),
            "seq_ids": [0, 1, 2, 3],
            "request_ids": ["z", "a", "y", "b"],
            "input_token_ids": [1, 101, 2, 301],
            "per_seq_lengths": [1, 1, 1, 1],
            "offsets": [0, 1, 2, 3],
            "total_tokens": 4,
            "positions": [0, 1, 0, 3],
            "kv_slot_ids": [0, 1010, 2, 3010],
        }
    )
    legacy_kv_plan = build_mailbox_kv_sync_plan(legacy, [SimpleNamespace(seq_id=i, request_id=str(i), token_ids=[1], block_table=[i]) for i in [0, 1, 2, 3]], state_sync_mode="no_commit_probe")
    ctx = build_target_forward_context_from_mailbox_input(legacy, legacy_kv_plan, seqs, make_step_plan())
    assert ctx.can_run_model is False
    assert ctx.cannot_run_reason == "illegal_legacy_fallback"


def test_no_commit_builder_does_not_mutate_sequence_state():
    target_input, kv_plan, seqs, step_plan = make_input_and_plan()
    before = [(list(seq.token_ids), list(seq.block_table), len(seq)) for seq in seqs]
    ctx = build_target_forward_context_from_mailbox_input(target_input, kv_plan, seqs, step_plan)
    after = [(list(seq.token_ids), list(seq.block_table), len(seq)) for seq in seqs]

    assert ctx.can_run_model is True
    assert after == before


def test_contextual_runtime_error_message_for_low_level_typeerror():
    target_input, kv_plan, seqs, step_plan = make_input_and_plan()
    ctx = build_target_forward_context_from_mailbox_input(target_input, kv_plan, seqs, step_plan)
    try:
        raise TypeError("can't assign a NoneType to a torch.cuda.IntTensor")
    except Exception as exc:
        message = (
            f"target forward from mailbox input failed during guarded probe; plan_id={step_plan.plan_id}, "
            f"target_seq_ids={ctx.seq_ids}, input_shape={ctx.input_ids_shape}, "
            f"positions_shape={ctx.positions_shape}, slot_mapping_shape={ctx.slot_mapping_shape}, "
            f"error={exc}; next_required_feature=target_forward_from_mailbox_guarded_forward_backend"
        )
    assert "plan_id=9" in message
    assert "slot_mapping_shape=[5]" in message
    assert "next_required_feature=target_forward_from_mailbox_guarded_forward_backend" in message
