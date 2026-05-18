"""CPU-safe V4K mailbox target-forward output normalization tests."""

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
context_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_forward_context")

TargetForwardFromMailboxInput = transport_mod.TargetForwardFromMailboxInput
interpret_target_forward_from_mailbox_output = transport_mod.interpret_target_forward_from_mailbox_output
normalize_target_forward_from_mailbox_output = context_mod.normalize_target_forward_from_mailbox_output
is_target_forward_output_owner = context_mod.is_target_forward_output_owner


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


def make_input(total_tokens: int = 5):
    return TargetForwardFromMailboxInput(
        plan_id=9,
        target_home_batch_id=1,
        seq_ids=[1, 3],
        request_ids=["a", "b"],
        input_token_ids=[101, 301, 302, 303, 304][:total_tokens],
        per_seq_lengths=[1, total_tokens - 1],
        offsets=[0, 1],
        total_tokens=total_tokens,
        gamma=4,
        positions=list(range(total_tokens)),
        kv_slot_ids=list(range(100, 100 + total_tokens)),
        source_mailbox_payload_ids=["1:1:0:1", f"1:3:1:{total_tokens - 1}"],
        source_draft_plan_id=9,
    )


def make_runner(local_rank: int, rank: int | None = None):
    rank = local_rank if rank is None else rank
    return SimpleNamespace(
        rank=rank,
        tp_params=SimpleNamespace(local_rank=local_rank, master_rank=10),
        group_config=SimpleNamespace(master_rank=10),
    )


def test_output_owner_detection_uses_tp_local_rank_zero():
    assert is_target_forward_output_owner(tp_params=SimpleNamespace(local_rank=0, master_rank=10)) is True
    assert is_target_forward_output_owner(tp_params=SimpleNamespace(local_rank=1, master_rank=10)) is False


def test_none_output_on_non_owner_is_expected_and_skips_interpretation():
    output = normalize_target_forward_from_mailbox_output(None, make_input(), SimpleNamespace(plan_id=9), make_runner(local_rank=1, rank=11))

    assert output.output_available is False
    assert output.output_owner is False
    assert output.output_owner_rank == 10
    assert output.current_rank == 11
    assert output.output_none_expected is True
    assert output.output_none_unexpected is False
    assert output.can_interpret is False
    assert output.next_required_feature is None
    json.dumps(output.to_dict(), sort_keys=True)


def test_none_output_on_owner_fails_with_output_ownership():
    output = normalize_target_forward_from_mailbox_output(None, make_input(), SimpleNamespace(plan_id=9), make_runner(local_rank=0, rank=10))

    assert output.output_available is False
    assert output.output_owner is True
    assert output.output_none_expected is False
    assert output.output_none_unexpected is True
    assert output.can_interpret is False
    assert output.next_required_feature == "target_forward_output_ownership"
    assert output.error_kind == "target_forward_output_ownership"


def test_tensor_output_shape_and_metadata_map_are_recorded():
    output = normalize_target_forward_from_mailbox_output(FakeTensor([5, 32000]), make_input(), SimpleNamespace(plan_id=9), make_runner(local_rank=0, rank=10))

    assert output.output_available is True
    assert output.output_owner is True
    assert output.output_shape == [5, 32000]
    assert output.output_num_rows == 5
    assert output.output_num_tokens == 5
    assert output.can_interpret is True
    assert output.interpretation_map == [
        {"seq_id": 1, "offset": 0, "length": 1, "row_start": 0, "row_end": 1, "token_range": [0, 1]},
        {"seq_id": 3, "offset": 1, "length": 4, "row_start": 1, "row_end": 5, "token_range": [1, 5]},
    ]
    assert interpret_target_forward_from_mailbox_output(make_input(), output.output_shape) == output.interpretation_map
    json.dumps(output.to_dict(), sort_keys=True)


def test_tuple_dict_and_model_output_extraction_paths():
    target_input = make_input()
    tuple_output = normalize_target_forward_from_mailbox_output(("ignore", FakeTensor([5, 7])), target_input, SimpleNamespace(plan_id=9), make_runner(0, 10))
    dict_output = normalize_target_forward_from_mailbox_output({"logits": FakeTensor([5, 9])}, target_input, SimpleNamespace(plan_id=9), make_runner(0, 10))
    model_output = normalize_target_forward_from_mailbox_output(FakeModelOutput(FakeTensor([5, 11])), target_input, SimpleNamespace(plan_id=9), make_runner(0, 10))

    assert tuple_output.extraction_path == "tuple[1]"
    assert dict_output.extraction_path == "dict.logits"
    assert model_output.extraction_path == "attr.logits"
    assert tuple_output.can_interpret and dict_output.can_interpret and model_output.can_interpret


def test_wrong_output_row_count_fails_clearly():
    output = normalize_target_forward_from_mailbox_output(FakeTensor([4, 32000]), make_input(), SimpleNamespace(plan_id=9), make_runner(local_rank=0, rank=10))

    assert output.output_available is True
    assert output.can_interpret is False
    assert output.next_required_feature == "target_forward_output_normalization"
    assert output.error_kind == "target_forward_output_row_count_mismatch"
    assert "rows 4" in output.error_message
    with pytest.raises(RuntimeError, match="target_forward_output_normalization"):
        interpret_target_forward_from_mailbox_output(make_input(), output_shape=[4, 32000])


def test_unsupported_output_type_fails_with_normalization_feature():
    output = normalize_target_forward_from_mailbox_output(object(), make_input(), SimpleNamespace(plan_id=9), make_runner(local_rank=0, rank=10))

    assert output.output_available is False
    assert output.can_interpret is False
    assert output.next_required_feature == "target_forward_output_normalization"
    assert output.error_kind == "target_forward_output_normalization"


def test_target_tp_mailbox_role_classification_json():
    role = context_mod.classify_target_tp_rank_role_for_mailbox_forward(make_runner(local_rank=1, rank=11))

    assert role.is_output_owner is False
    assert role.is_payload_owner is False
    assert role.should_run_target_forward is True
    assert role.should_interpret_output is False
    assert role.should_apply_verify_result is False
    assert role.should_skip_non_owner is True
    json.dumps(role.to_dict(), sort_keys=True)
