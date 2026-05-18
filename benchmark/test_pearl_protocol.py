#!/usr/bin/env python3
"""CPU-only tests for the explicit V4B PEARL protocol envelopes."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL_PATH = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_protocol.py")

spec = importlib.util.spec_from_file_location("pearl_protocol_under_test", PROTOCOL_PATH)
assert spec is not None and spec.loader is not None
pearl_protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pearl_protocol
spec.loader.exec_module(pearl_protocol)


def make_seqs():
    return [
        SimpleNamespace(seq_id=10, request_id="req-a", pre_verify=True),
        SimpleNamespace(seq_id=11, request_id="req-b", pre_verify=False),
    ]


def test_build_offsets_and_validation() -> None:
    assert pearl_protocol.build_offsets([1, 4, 2]) == [0, 1, 5]
    pearl_protocol.validate_offsets([1, 4, 2], [0, 1, 5], 7)
    with pytest.raises(RuntimeError, match="Invalid PEARL protocol offsets"):
        pearl_protocol.validate_offsets([1, 4, 2], [0, 2, 5], 7)


def test_encode_decode_legacy_draft_message_json_metadata() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_legacy_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[100, 200, 201, 202, 203],
        next_round_input=[300, 301, 302, 303, 400, 401, 402, 403],
        plan_id=7,
        runner_role="draft",
        scheduled_seq_ids=[10, 11],
        actual_exec_seq_ids=[10, 11],
        target_batch_seq_ids=[10],
        draft_home_batch_seq_ids=[11],
    )
    assert message.protocol_version == 1
    assert message.message_type == "draft_tokens"
    assert message.layout_kind == "legacy_fixed"
    assert message.per_seq_draft_lengths == [1, 4]
    assert message.draft_offsets == [0, 1]
    assert message.total_draft_tokens == 5
    assert pearl_protocol.decode_legacy_draft_message(message) == (
        [100, 200, 201, 202, 203],
        [300, 301, 302, 303, 400, 401, 402, 403],
    )
    json.dumps(message.to_trace_dict(), sort_keys=True)
    assert len(message.digest()) == 16


def test_encode_decode_legacy_verify_result() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_legacy_verify_result(
        seqs=seqs,
        gamma=4,
        acc=[True, False],
        rollout=[0, 2],
        revise_token=[-1, 999],
        finish=[False, True],
        plan_id=8,
        runner_role="verify",
    )
    assert message.message_type == "verify_result"
    assert message.per_seq_accepted_lengths == [1, 2]
    assert message.accepted_offsets == [0, 1]
    assert message.total_accepted_tokens == 3
    assert pearl_protocol.decode_legacy_verify_result(message) == (
        [True, False],
        [0, 2],
        [-1, 999],
        [False, True],
    )
    json.dumps(message.to_trace_dict(), sort_keys=True)
    assert len(message.digest()) == 16


def test_protocol_validation_rejects_mismatched_seq_ids() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_legacy_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[100, 200, 201, 202, 203],
        next_round_input=[300, 301, 302, 303, 400, 401, 402, 403],
    )
    with pytest.raises(RuntimeError, match="seq alignment failed"):
        pearl_protocol.validate_legacy_fixed_layout(message, [11, 10], gamma=4)


def test_protocol_validation_rejects_bad_offsets() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_legacy_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[100, 200, 201, 202, 203],
        next_round_input=[300, 301, 302, 303, 400, 401, 402, 403],
    )
    bad_message = replace(message, draft_offsets=[0, 2])
    with pytest.raises(RuntimeError, match="Invalid PEARL protocol offsets"):
        pearl_protocol.validate_legacy_fixed_layout(bad_message, [10, 11], gamma=4)



def test_variable_offsets_build_offsets_with_zero_length() -> None:
    assert pearl_protocol.build_offsets([1, 1, 1]) == [0, 1, 2]
    assert pearl_protocol.build_offsets([4, 0, 2]) == [0, 4, 4]


def test_encode_decode_variable_draft_message_uniform_and_heterogeneous() -> None:
    seqs = make_seqs()
    uniform = pearl_protocol.encode_variable_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        next_round_input=[11, 12, 13, 14, 21, 22, 23, 24],
        per_seq_draft_lengths=[4, 4],
        plan_id=9,
    )
    assert uniform.layout_kind == "variable_offsets"
    assert uniform.draft_offsets == [0, 4]
    assert uniform.total_draft_tokens == 8
    assert pearl_protocol.decode_variable_draft_message(uniform)[0] == [1, 2, 3, 4, 5, 6, 7, 8]

    heterogeneous = pearl_protocol.encode_variable_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[1, 2, 3, 4, 5],
        next_round_input=[11, 12, 13, 14, 21, 22, 23, 24],
        per_seq_draft_lengths=[1, 4],
        plan_id=10,
    )
    assert heterogeneous.draft_offsets == [0, 1]
    assert heterogeneous.per_seq_draft_lengths == [1, 4]
    json.dumps(heterogeneous.to_trace_dict(), sort_keys=True)


def test_encode_decode_variable_verify_result_and_zero_length() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_variable_verify_result(
        seqs=seqs,
        gamma=4,
        acc=[False, True],
        rollout=[1, 0],
        revise_token=[101, -1],
        finish=[False, False],
        per_seq_accepted_lengths=[0, 4],
        plan_id=11,
    )
    assert message.layout_kind == "variable_offsets"
    assert message.per_seq_accepted_lengths == [0, 4]
    assert message.accepted_offsets == [0, 0]
    assert message.total_accepted_tokens == 4
    assert pearl_protocol.decode_variable_verify_result(message) == (
        [False, True],
        [1, 0],
        [101, -1],
        [False, False],
    )
    json.dumps(message.to_trace_dict(), sort_keys=True)


def test_variable_offsets_validation_failures() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_variable_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[1, 2, 3, 4, 5],
        next_round_input=[11, 12, 13, 14, 21, 22, 23, 24],
        per_seq_draft_lengths=[1, 4],
    )
    bad_offsets = replace(message, draft_offsets=[0, 2])
    with pytest.raises(RuntimeError, match="Invalid PEARL protocol offsets"):
        pearl_protocol.validate_variable_offsets_layout(bad_offsets, [10, 11])

    duplicate = replace(message, seq_ids=[10, 10])
    with pytest.raises(RuntimeError, match="duplicate seq_ids"):
        pearl_protocol.validate_variable_offsets_layout(duplicate, [10, 10])

    bad_payload = replace(message, draft_token_ids=[1, 2])
    with pytest.raises(RuntimeError, match="payload length"):
        pearl_protocol.validate_variable_offsets_layout(bad_payload, [10, 11])


def test_variable_offsets_no_longer_reserved() -> None:
    seqs = make_seqs()
    message = pearl_protocol.encode_variable_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[1, 2, 3, 4, 5],
        next_round_input=[11, 12, 13, 14, 21, 22, 23, 24],
        per_seq_draft_lengths=[1, 4],
    )
    pearl_protocol.validate_variable_offsets_layout(message, [10, 11])
    with pytest.raises(NotImplementedError, match="not legacy_fixed"):
        pearl_protocol.ensure_legacy_fixed_layout("variable_offsets")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
