"""CPU-safe tests for V4G mailbox payload tensor and verification input scaffolds."""

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

STSpecPayloadMailbox = mailbox_mod.STSpecPayloadMailbox
payloads_from_draft_message = mailbox_mod.payloads_from_draft_message
encode_variable_draft_message = protocol_mod.encode_variable_draft_message
MailboxPayloadTensorEnvelope = transport_mod.MailboxPayloadTensorEnvelope
build_verification_input_from_mailbox_payload = transport_mod.build_verification_input_from_mailbox_payload
encode_payload_tensor_envelope_from_payloads = transport_mod.encode_payload_tensor_envelope_from_payloads
payload_tensor_envelope_to_mailbox_payloads = transport_mod.payload_tensor_envelope_to_mailbox_payloads
validate_payload_tensor_envelope = transport_mod.validate_payload_tensor_envelope


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


def test_variable_offsets_message_to_payload_tensor_envelope_json():
    payloads = make_payloads(1)
    envelope = encode_payload_tensor_envelope_from_payloads(payloads, home_batch_id=1, gamma=4)

    assert envelope.seq_ids == [1, 3]
    assert envelope.per_seq_lengths == [1, 4]
    assert envelope.offsets == [0, 1]
    assert envelope.total_tokens == 5
    assert envelope.payload_tensor_ids == [101, 301, 302, 303, 304]
    assert envelope.payload_shape == [5]
    json.dumps(envelope.to_dict(), sort_keys=True)
    assert len(envelope.digest()) == 16


def test_payload_tensor_envelope_validation_errors():
    envelope = encode_payload_tensor_envelope_from_payloads(make_payloads(1), home_batch_id=1, gamma=4)
    with pytest.raises(RuntimeError, match="payload_tensor_ids length"):
        validate_payload_tensor_envelope(
            MailboxPayloadTensorEnvelope(**{**envelope.to_dict(), "payload_tensor_ids": [1]})
        )
    with pytest.raises(RuntimeError, match="duplicate seq_ids"):
        validate_payload_tensor_envelope(
            MailboxPayloadTensorEnvelope(**{**envelope.to_dict(), "seq_ids": [1, 1]})
        )
    with pytest.raises(RuntimeError, match="requires variable_offsets"):
        validate_payload_tensor_envelope(
            MailboxPayloadTensorEnvelope(**{**envelope.to_dict(), "layout_kind": "legacy_fixed"})
        )


def test_target_mailbox_insert_consume_exact_set_and_wrong_batch_miss():
    mailbox = STSpecPayloadMailbox()
    envelope = encode_payload_tensor_envelope_from_payloads(make_payloads(1), home_batch_id=1, gamma=4)
    mailbox.put_payloads(1, payload_tensor_envelope_to_mailbox_payloads(envelope))

    hit = mailbox.get_payloads(1, [1, 3])
    wrong_batch = mailbox.get_payloads(0, [1, 3])
    missing = mailbox.get_payloads(1, [1, 2])

    assert hit.success
    assert [payload.seq_id for payload in hit.payloads] == [1, 3]
    assert not wrong_batch.success
    assert wrong_batch.missing_seq_ids == [1, 3]
    assert not missing.success
    assert missing.missing_seq_ids == [2]


def test_verification_input_metadata_construction_and_mismatch_detection():
    payloads = make_payloads(1)
    seqs = [SimpleNamespace(seq_id=1), SimpleNamespace(seq_id=3)]
    step_plan = SimpleNamespace(target_home_batch_id=1)

    metadata = build_verification_input_from_mailbox_payload(payloads, seqs, step_plan, gamma=4)

    assert metadata.seq_ids == [1, 3]
    assert metadata.input_token_ids == [101, 301, 302, 303, 304]
    assert metadata.offsets == [0, 1]
    assert metadata.per_seq_lengths == [1, 4]
    assert metadata.total_tokens == 5
    json.dumps(metadata.to_dict(), sort_keys=True)

    with pytest.raises(RuntimeError, match="seq mismatch"):
        build_verification_input_from_mailbox_payload(payloads, [SimpleNamespace(seq_id=3), SimpleNamespace(seq_id=1)], step_plan, 4)
    with pytest.raises(RuntimeError, match="home batch mismatch"):
        build_verification_input_from_mailbox_payload(payloads, seqs, SimpleNamespace(target_home_batch_id=0), 4)


def test_illegal_legacy_fallback_detection_shape():
    scheduled_seq_ids = [0, 1, 2, 3]
    actual_target_exec_seq_ids = [1, 3]
    record = {"illegal_legacy_fallback": False}
    if actual_target_exec_seq_ids != scheduled_seq_ids:
        record["illegal_legacy_fallback"] = True
    assert record["illegal_legacy_fallback"] is True
