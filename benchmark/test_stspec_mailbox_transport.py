"""CPU-safe tests for V4E ST-Spec mailbox transport envelopes."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types

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
transport_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_transport")

MailboxPayload = mailbox_mod.MailboxPayload
STSpecMailboxError = mailbox_mod.STSpecMailboxError
STSpecPayloadMailbox = mailbox_mod.STSpecPayloadMailbox
MailboxTransportMode = transport_mod.MailboxTransportMode
classify_mailbox_miss = transport_mod.classify_mailbox_miss
decode_mailbox_transport_envelope = transport_mod.decode_mailbox_transport_envelope
encode_mailbox_transport_envelope = transport_mod.encode_mailbox_transport_envelope
envelope_to_mailbox_payloads = transport_mod.envelope_to_mailbox_payloads
insert_transport_envelope_into_mailbox = transport_mod.insert_transport_envelope_into_mailbox
validate_mailbox_transport_envelope = transport_mod.validate_mailbox_transport_envelope


def make_payload(seq_id: int, home_batch_id: int = 9, tokens: list[int] | None = None) -> MailboxPayload:
    tokens = [seq_id * 10] if tokens is None else list(tokens)
    return MailboxPayload(
        plan_id=77,
        producer_role="draft",
        producer_home_batch_id=home_batch_id,
        target_home_batch_id=8,
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
        logical_step=77,
        producer_actual_exec_seq_ids=[seq_id],
        producer_draft_message_seq_ids=[seq_id],
        metadata={"source": "test"},
    )


def test_encode_decode_transport_envelope_json_serializable():
    envelope = encode_mailbox_transport_envelope(
        payloads=[make_payload(1, tokens=[10]), make_payload(3, tokens=[30, 31])],
        transport_mode=MailboxTransportMode.DIAGNOSTIC_ONLY,
        producer_rank=0,
        source_plan_signature_hash="abc123",
        payload_metadata={"gamma": 4},
    )

    decoded = decode_mailbox_transport_envelope(envelope.to_json())

    assert decoded.seq_ids == [1, 3]
    assert decoded.per_seq_lengths == [1, 2]
    assert decoded.offsets == [0, 1]
    assert decoded.total_tokens == 3
    assert decoded.draft_token_ids == [10, 30, 31]
    assert decoded.payload_available is True
    json.dumps(decoded.to_dict(), sort_keys=True)
    assert len(decoded.digest()) == 16


def test_validation_rejects_bad_offsets_payload_length_and_duplicate_seq_ids():
    envelope = encode_mailbox_transport_envelope(payloads=[make_payload(1), make_payload(2)])

    with pytest.raises(RuntimeError, match="Invalid PEARL protocol offsets"):
        validate_mailbox_transport_envelope(
            transport_mod.MailboxTransportEnvelope(**{**envelope.to_dict(), "offsets": [0, 99]})
        )
    with pytest.raises(RuntimeError, match="draft_token_ids length"):
        validate_mailbox_transport_envelope(
            transport_mod.MailboxTransportEnvelope(**{**envelope.to_dict(), "draft_token_ids": [1]})
        )
    with pytest.raises(RuntimeError, match="duplicate seq_ids"):
        validate_mailbox_transport_envelope(
            transport_mod.MailboxTransportEnvelope(**{**envelope.to_dict(), "seq_ids": [1, 1]})
        )


def test_insert_transported_payloads_into_mailbox_and_retrieve_by_batch():
    mailbox = STSpecPayloadMailbox()
    envelope = encode_mailbox_transport_envelope(
        payloads=[make_payload(1, tokens=[10]), make_payload(3, tokens=[30])],
        payload_metadata={"gamma": 4},
    )

    inserted = insert_transport_envelope_into_mailbox(mailbox, envelope, consumer_role="verify")
    hit = mailbox.get_payloads(9, [1, 3])
    wrong_batch = mailbox.get_payloads(8, [1, 3])

    assert inserted == 2
    assert hit.success
    assert [payload.draft_token_ids for payload in hit.payloads] == [[10], [30]]
    assert not wrong_batch.success
    assert wrong_batch.missing_seq_ids == [1, 3]


def test_metadata_only_transport_marks_payload_unavailable():
    envelope = encode_mailbox_transport_envelope(
        payloads=[make_payload(1, tokens=[10])],
        payload_available=False,
        payload_metadata={"gamma": 4},
    )
    payloads = envelope_to_mailbox_payloads(envelope)

    assert envelope.payload_available is False
    assert envelope.draft_token_ids == []
    assert payloads[0].draft_token_ids == []
    assert payloads[0].per_seq_length == 1
    assert payloads[0].metadata["mailbox_transport_payload_available"] is False


def test_wrong_seq_id_not_returned_and_duplicate_payload_handling():
    mailbox = STSpecPayloadMailbox()
    envelope = encode_mailbox_transport_envelope(payloads=[make_payload(1, tokens=[10])])
    insert_transport_envelope_into_mailbox(mailbox, envelope)

    assert not mailbox.get_payloads(9, [2]).success
    with pytest.raises(STSpecMailboxError, match="duplicate_put"):
        insert_transport_envelope_into_mailbox(mailbox, envelope)


def test_warmup_miss_classification_and_allow_warmup_skip():
    assert classify_mailbox_miss(
        target_home_batch_id=8,
        available_home_batch_ids=[],
        allow_warmup_miss=False,
    ) == ("mailbox_warmup_miss", "pipeline_warmup_schedule", True)
    assert classify_mailbox_miss(
        target_home_batch_id=8,
        available_home_batch_ids=[],
        allow_warmup_miss=True,
    ) == ("mailbox_warmup_skip_not_implemented", "pipeline_warmup_schedule", True)
    assert classify_mailbox_miss(
        target_home_batch_id=8,
        available_home_batch_ids=[8],
        allow_warmup_miss=True,
    ) == ("mailbox_missing_payload", "mailbox_payload_tensor_transport", False)
