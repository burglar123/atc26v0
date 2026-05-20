"""CPU-safe V4S result finalization helper tests."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from types import SimpleNamespace


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

apply_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_verify_apply")


class FakeSeq:
    def __init__(self, seq_id: int, request_id: str, tokens: list[int], prompt_len: int = 2):
        self.seq_id = seq_id
        self.request_id = request_id
        self.token_ids = list(tokens)
        self.num_prompt_tokens = int(prompt_len)


def config(**overrides):
    values = {
        "enable_stspec_two_batch_execution": True,
        "stspec_two_batch_dryrun": False,
        "stspec_two_batch_probe": True,
        "stspec_mailbox_commit_probe": True,
        "stspec_continue_after_mailbox_commit": True,
        "pearl_protocol_layout": "variable_offsets",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def commit_result(**overrides):
    values = {
        "success": True,
        "request_completion_check_attempted": True,
        "request_completion_check_success": True,
        "breadth_only_completed": True,
        "result_finalization_attempted": True,
        "next_required_feature": "result_finalization_after_breadth_only_completion",
        "unfinished_seq_ids_at_completion_check": [],
        "active_seq_ids_at_completion_check": [],
        "scheduler_active_seq_ids_at_completion": [],
        "mailbox_pending_payload_ids_at_completion": [],
        "sequence_state_completion_valid": True,
        "scheduler_state_completion_valid": True,
        "mailbox_state_completion_valid": True,
        "finished_seq_ids_at_completion_check": [1, 3],
        "committed_seq_ids": [1, 3],
        "request_completion_error": None,
        "request_completion_error_kind": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def seqs():
    return [
        FakeSeq(1, "r1", [10, 11, 101], prompt_len=2),
        FakeSeq(3, "r3", [30, 31, 301, 302], prompt_len=2),
    ]


def test_completed_commit_result_enters_finalization():
    result = commit_result()
    assert apply_mod.can_enter_v4s_result_finalization(result, config(), is_output_owner=True)
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        result,
        seqs(),
        is_output_owner=True,
        scheduler_active_seq_ids=[],
        mailbox_pending_payload_ids=[],
    )
    assert metadata["result_finalization_success"] is True
    assert metadata["finalized_request_ids"] == ["r1", "r3"]
    assert metadata["finalized_seq_ids"] == [1, 3]
    assert metadata["finalized_output_token_counts"] == {1: 1, 3: 2}
    assert metadata["next_required_feature"] == "end_to_end_breadth_only_completion"
    assert metadata["v4s_finalization_metadata_complete"] is True
    assert metadata["v4s_completion_gate_snapshot"]["request_completion_check_success"] is True


def test_completion_metadata_can_infer_breadth_completion():
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        commit_result(breadth_only_completed=False, breadth_only_completion_reason=None),
        seqs(),
        is_output_owner=True,
        scheduler_active_seq_ids=[],
        mailbox_pending_payload_ids=[],
    )
    assert metadata["result_finalization_success"] is True
    assert metadata["v4s_completion_gate_reason"] == "completion_metadata_proves_breadth_only_complete"


def test_active_request_does_not_finalize():
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        commit_result(unfinished_seq_ids_at_completion_check=[3]),
        seqs(),
        is_output_owner=True,
        scheduler_active_seq_ids=[],
        mailbox_pending_payload_ids=[],
    )
    assert metadata["result_finalization_success"] is False
    assert metadata["result_finalization_error_kind"] == "unfinished_requests_at_finalization"
    assert metadata["next_required_feature"] == "active_request_continuation_after_breadth_only_step"


def test_invalid_scheduler_state_gets_specific_diagnostic():
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        commit_result(scheduler_state_completion_valid=False),
        seqs(),
        is_output_owner=True,
        scheduler_active_seq_ids=[],
        mailbox_pending_payload_ids=[],
    )
    assert metadata["result_finalization_success"] is False
    assert metadata["result_finalization_error_kind"] == "scheduler_state_completion_invalid"
    assert metadata["next_required_feature"] == "scheduler_state_after_breadth_only_completion"


def test_non_owner_skips_finalization():
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        commit_result(),
        seqs(),
        is_output_owner=False,
    )
    assert metadata["result_finalization_skipped_non_owner"] is True
    assert metadata["result_finalization_attempted"] is False
    assert metadata["result_finalization_success"] is False


def test_missing_output_tokens_gives_text_assembly_diagnostic():
    bad = [FakeSeq(1, "r1", [10], prompt_len=2), FakeSeq(3, "r3", [30, 31, 301], prompt_len=2)]
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        commit_result(),
        bad,
        is_output_owner=True,
    )
    assert metadata["result_finalization_success"] is False
    assert metadata["result_finalization_error_kind"] == "finalized_token_state_invalid"
    assert metadata["next_required_feature"] == "result_text_assembly_after_breadth_only_completion"


def test_finalized_metadata_json_serializable():
    metadata = apply_mod.build_v4s_result_finalization_metadata(
        commit_result(),
        seqs(),
        is_output_owner=True,
    )
    json.dumps(metadata, sort_keys=True, default=str)


def test_normal_and_dryrun_do_not_trigger_finalization():
    result = commit_result()
    assert not apply_mod.can_enter_v4s_result_finalization(
        result,
        config(enable_stspec_two_batch_execution=False),
        is_output_owner=True,
    )
    assert not apply_mod.can_enter_v4s_result_finalization(
        result,
        config(stspec_two_batch_dryrun=True),
        is_output_owner=True,
    )
    assert not apply_mod.can_enter_v4s_result_finalization(
        result,
        config(pearl_protocol_layout="legacy_fixed"),
        is_output_owner=True,
    )


def main() -> None:
    test_completed_commit_result_enters_finalization()
    test_completion_metadata_can_infer_breadth_completion()
    test_active_request_does_not_finalize()
    test_invalid_scheduler_state_gets_specific_diagnostic()
    test_non_owner_skips_finalization()
    test_missing_output_tokens_gives_text_assembly_diagnostic()
    test_finalized_metadata_json_serializable()
    test_normal_and_dryrun_do_not_trigger_finalization()


if __name__ == "__main__":
    main()
