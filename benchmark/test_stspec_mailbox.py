"""CPU-safe diagnostics for the V4D ST-Spec mailbox scaffold."""

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

pearl_protocol = importlib.import_module("nano_pearl.pearl_engine.pearl_protocol")
stspec_mailbox = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox")
encode_variable_draft_message = pearl_protocol.encode_variable_draft_message
MailboxPayload = stspec_mailbox.MailboxPayload
STSpecMailboxError = stspec_mailbox.STSpecMailboxError
STSpecPayloadMailbox = stspec_mailbox.STSpecPayloadMailbox
payloads_from_draft_message = stspec_mailbox.payloads_from_draft_message


def make_payload(seq_id: int, home_batch_id: int = 7, *, tokens: list[int] | None = None) -> MailboxPayload:
    tokens = [] if tokens is None else list(tokens)
    return MailboxPayload(
        plan_id=11,
        producer_role="draft",
        producer_home_batch_id=home_batch_id,
        target_home_batch_id=3,
        draft_home_batch_id=home_batch_id,
        seq_id=seq_id,
        request_id=f"req-{seq_id}",
        home_batch_id=home_batch_id,
        gamma=4,
        layout_kind="variable_offsets",
        protocol_version=1,
        draft_token_ids=tokens,
        per_seq_length=len(tokens),
        offset=0,
        logical_step=11,
        producer_actual_exec_seq_ids=[seq_id],
        producer_draft_message_seq_ids=[seq_id],
        metadata={"debug": True},
    )


def test_put_get_by_home_batch_and_seq_id():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(7, [make_payload(1, tokens=[10]), make_payload(3, tokens=[30, 31])])

    result = mailbox.get_payloads(7, [1, 3], plan_id=12, consumer_role="verify")

    assert result.success
    assert result.hit_count == 2
    assert result.miss_count == 0
    assert [payload.seq_id for payload in result.payloads] == [1, 3]
    assert [payload.draft_token_ids for payload in result.payloads] == [[10], [30, 31]]


def test_miss_for_absent_home_batch_and_absent_seq_id():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(7, [make_payload(1, tokens=[10])])

    wrong_batch = mailbox.get_payloads(8, [1])
    missing_seq = mailbox.get_payloads(7, [1, 2])

    assert not wrong_batch.success
    assert wrong_batch.missing_seq_ids == [1]
    assert not missing_seq.success
    assert missing_seq.hit_count == 1
    assert missing_seq.missing_seq_ids == [2]


def test_duplicate_put_fails_explicitly():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(7, [make_payload(1, tokens=[10])])

    with pytest.raises(STSpecMailboxError, match="duplicate_put"):
        mailbox.put_payloads(7, [make_payload(1, tokens=[11])])

    assert mailbox.stats()["duplicate_put_count"] == 1


def test_pop_consume_once_semantics():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(7, [make_payload(1, tokens=[10])])

    popped = mailbox.pop_payloads(7, [1])
    after = mailbox.get_payloads(7, [1])

    assert popped.success
    assert popped.payloads[0].draft_token_ids == [10]
    assert not after.success
    assert mailbox.stats()["pop_count"] == 1


def test_clear_finished_removes_payloads_across_batches():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(7, [make_payload(1, tokens=[10]), make_payload(2, tokens=[20])])
    mailbox.put_payloads(8, [make_payload(1, home_batch_id=8, tokens=[30])])

    removed = mailbox.clear_finished([1])

    assert removed == 2
    assert not mailbox.get_payloads(7, [1]).success
    assert mailbox.get_payloads(7, [2]).success
    assert not mailbox.get_payloads(8, [1]).success


def test_stats_and_payload_metadata_are_json_serializable():
    mailbox = STSpecPayloadMailbox()
    payload = make_payload(1, tokens=[10])
    mailbox.put_payloads(7, [payload])

    json.dumps(mailbox.stats(), sort_keys=True)
    json.dumps(payload.to_dict(), sort_keys=True)
    json.dumps(mailbox.get_payloads(7, [1]).to_trace_dict(), sort_keys=True)


def test_zero_length_payload_can_be_stored():
    mailbox = STSpecPayloadMailbox()
    mailbox.put_payloads(7, [make_payload(1, tokens=[])])

    result = mailbox.get_payloads(7, [1])

    assert result.success
    assert result.payloads[0].per_seq_length == 0
    assert result.payloads[0].draft_token_ids == []


def test_variable_offsets_message_converts_to_mailbox_payloads():
    seqs = [
        SimpleNamespace(seq_id=1, request_id="a", pre_verify=True),
        SimpleNamespace(seq_id=3, request_id="b", pre_verify=False),
    ]
    message = encode_variable_draft_message(
        seqs=seqs,
        gamma=4,
        draft_token_ids=[101, 301, 302, 303, 304],
        next_round_input=[11, 12, 13, 14, 31, 32, 33, 34],
        plan_id=22,
        runner_role="draft",
        scheduled_seq_ids=[0, 1, 2, 3],
        actual_exec_seq_ids=[1, 3],
        target_batch_seq_ids=[0, 2],
        draft_home_batch_seq_ids=[1, 3],
    )

    payloads = payloads_from_draft_message(
        message,
        home_batch_id=9,
        target_home_batch_id=8,
        draft_home_batch_id=9,
        producer_home_batch_id=9,
        logical_step=22,
    )

    assert [payload.seq_id for payload in payloads] == [1, 3]
    assert [payload.draft_token_ids for payload in payloads] == [[101], [301, 302, 303, 304]]
    assert [payload.per_seq_length for payload in payloads] == [1, 4]
    assert [payload.offset for payload in payloads] == [0, 1]
    assert all(payload.home_batch_id == 9 for payload in payloads)
