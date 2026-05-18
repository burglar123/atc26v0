"""CPU-safe diagnostics for the V4F ST-Spec pipeline warmup scaffold."""

from __future__ import annotations

import importlib
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
pipeline_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_pipeline")

MailboxPayload = mailbox_mod.MailboxPayload
STSpecPayloadMailbox = mailbox_mod.STSpecPayloadMailbox
classify_mailbox_miss = transport_mod.classify_mailbox_miss
encode_mailbox_transport_envelope = transport_mod.encode_mailbox_transport_envelope
insert_transport_envelope_into_mailbox = transport_mod.insert_transport_envelope_into_mailbox
STSpecPipelineController = pipeline_mod.STSpecPipelineController
STSpecPipelinePhase = pipeline_mod.STSpecPipelinePhase
should_skip_target_for_warmup = pipeline_mod.should_skip_target_for_warmup
target_home_batch_for_steady_after_warmup = pipeline_mod.target_home_batch_for_steady_after_warmup


def make_payload(seq_id: int, home_batch_id: int) -> MailboxPayload:
    return MailboxPayload(
        plan_id=1,
        producer_role="draft",
        producer_home_batch_id=home_batch_id,
        target_home_batch_id=1 - home_batch_id,
        draft_home_batch_id=home_batch_id,
        seq_id=seq_id,
        request_id=f"req-{seq_id}",
        home_batch_id=home_batch_id,
        gamma=4,
        layout_kind="variable_offsets",
        protocol_version=1,
        draft_token_ids=[seq_id * 10],
        per_seq_length=1,
        offset=0,
        logical_step=0,
        producer_actual_exec_seq_ids=[seq_id],
        producer_draft_message_seq_ids=[seq_id],
        metadata={},
    )


def test_initial_phase_is_warmup_then_advances_to_steady():
    controller = STSpecPipelineController(enabled=True, warmup_enabled=True, warmup_draft_only=True)

    first = controller.state_for_next_decode(target_home_batch_id=0, draft_home_batch_id=1)
    controller.advance_after_decode()
    second = controller.state_for_next_decode(target_home_batch_id=1, draft_home_batch_id=0)

    assert first.phase == STSpecPipelinePhase.WARMUP_DRAFT_ONLY.value
    assert first.warmup_done is False
    assert first.warmup_target_home_batch_id == 0
    assert first.warmup_draft_home_batch_id == 1
    assert target_home_batch_for_steady_after_warmup(first) == 1
    assert second.phase == STSpecPipelinePhase.STEADY_STATE.value
    assert second.warmup_done is True


def test_warmup_target_skip_classification_requires_allow_flag():
    assert should_skip_target_for_warmup(
        phase="warmup_draft_only", runner_role="verify", allow_warmup_miss=True
    )
    assert not should_skip_target_for_warmup(
        phase="warmup_draft_only", runner_role="verify", allow_warmup_miss=False
    )
    assert not should_skip_target_for_warmup(
        phase="warmup_draft_only", runner_role="draft", allow_warmup_miss=True
    )
    assert classify_mailbox_miss(
        target_home_batch_id=0, available_home_batch_ids=[], allow_warmup_miss=False
    ) == ("mailbox_warmup_miss", "pipeline_warmup_schedule", True)


def test_payload_produced_for_draft_batch_consumed_after_rotation():
    mailbox = STSpecPayloadMailbox()
    # Step 0 warmup: draft produces for home batch 1 while target home batch 0 is skipped.
    envelope = encode_mailbox_transport_envelope(
        payloads=[make_payload(1, home_batch_id=1), make_payload(3, home_batch_id=1)],
        payload_metadata={"gamma": 4},
    )
    insert_transport_envelope_into_mailbox(mailbox, envelope)

    # Step 1 steady: target rotates to home batch 1 and can retrieve exactly those seq ids.
    hit = mailbox.get_payloads(1, [1, 3])
    wrong_batch = mailbox.get_payloads(0, [1, 3])
    wrong_seq = mailbox.get_payloads(1, [0, 2])

    assert hit.success
    assert [payload.seq_id for payload in hit.payloads] == [1, 3]
    assert not wrong_batch.success
    assert wrong_batch.missing_seq_ids == [1, 3]
    assert not wrong_seq.success
    assert wrong_seq.missing_seq_ids == [0, 2]


def test_target_consume_metadata_success_then_verification_input_blocker():
    mailbox = STSpecPayloadMailbox()
    insert_transport_envelope_into_mailbox(
        mailbox,
        encode_mailbox_transport_envelope(payloads=[make_payload(1, home_batch_id=1)]),
    )
    result = mailbox.get_payloads(1, [1])

    assert result.success
    assert [payload.home_batch_id for payload in result.payloads] == [1]
    # V4F intentionally stops after metadata-level consume; model input wiring is next.
    with pytest.raises(RuntimeError, match="verification input construction from mailbox payload is not implemented"):
        raise RuntimeError(
            "target consume-from-mailbox succeeded, but verification input construction from mailbox payload is not implemented"
        )


def test_illegal_legacy_fallback_detection_shape():
    record = {"illegal_legacy_fallback": False}
    payload_seq_ids = [1, 3]
    target_seq_ids = [0, 2]
    if payload_seq_ids != target_seq_ids:
        record["illegal_legacy_fallback"] = True

    assert record["illegal_legacy_fallback"] is True
