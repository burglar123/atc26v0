"""CPU-safe V4T active continuation metadata tests."""

from __future__ import annotations

import importlib
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
mailbox_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox")
transport_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_mailbox_transport")


def commit_result(**overrides):
    values = {
        "request_completion_reason": "active_requests_remaining",
        "breadth_only_completion_reason": "second_step_metadata_built",
        "unfinished_seq_ids_at_completion_check": [1, 3],
        "active_seq_ids_at_completion_check": [1, 3],
        "scheduler_active_seq_ids_at_completion": [1, 3],
        "mailbox_pending_payload_ids_at_completion": [],
        "duplicate_payload_consume_after_continue": False,
        "repeated_verify_after_commit_detected": False,
        "breadth_only_step_count": 2,
        "next_required_feature": "active_request_continuation_after_breadth_only_step",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_unfinished_requests_attempt_active_continuation():
    metadata = apply_mod.build_v4t_active_continuation_metadata(commit_result(), max_steps=1, step_count=1)
    assert metadata["active_continuation_attempted"] is True
    assert metadata["active_continuation_success"] is True
    assert metadata["active_continuation_seq_ids"] == [1, 3]
    assert metadata["next_required_feature"] == "active_request_continuation_handoff"


def test_fake_runner_state_initializes_and_resets():
    runner = SimpleNamespace()
    apply_mod.initialize_v4t_active_continuation_runner_state(runner)
    assert runner.stspec_active_continuation_step_count == 0
    assert runner.stspec_active_continuation_last_snapshot is None
    assert runner.stspec_active_continuation_plan_id_history == []
    assert runner.stspec_active_continuation_progress_by_step == []
    runner.stspec_active_continuation_step_count = 2
    runner.stspec_active_continuation_last_snapshot = {"plan_id": 4}
    runner.stspec_active_continuation_plan_id_history = [4]
    runner.stspec_active_continuation_progress_by_step = [{"plan_id": 4}]
    apply_mod.initialize_v4t_active_continuation_runner_state(runner)
    assert runner.stspec_active_continuation_step_count == 2
    apply_mod.reset_v4t_active_continuation_runner_state(runner)
    assert runner.stspec_active_continuation_step_count == 0
    assert runner.stspec_active_continuation_last_snapshot is None
    assert runner.stspec_active_continuation_plan_id_history == []
    assert runner.stspec_active_continuation_progress_by_step == []


def test_active_continuation_max_step_reached():
    metadata = apply_mod.build_v4t_active_continuation_metadata(commit_result(), max_steps=1, step_count=2)
    assert metadata["active_continuation_attempted"] is True
    assert metadata["active_continuation_success"] is False
    assert metadata["active_continuation_limit_reached"] is True
    assert metadata["active_continuation_error_kind"] == "active_request_continuation_limit_reached"
    assert metadata["next_required_feature"] == "active_request_continuation_limit_reached"


def test_active_continuation_allows_multiple_steps_before_limit():
    metadata = apply_mod.build_v4t_active_continuation_metadata(commit_result(), max_steps=8, step_count=3)
    assert metadata["active_continuation_attempted"] is True
    assert metadata["active_continuation_success"] is True
    assert metadata["active_continuation_step_count"] == 3
    assert metadata["next_required_feature"] == "active_request_continuation_handoff"


def test_duplicate_consume_detection():
    metadata = apply_mod.build_v4t_active_continuation_metadata(
        commit_result(duplicate_payload_consume_after_continue=True),
        max_steps=1,
        step_count=1,
    )
    assert metadata["active_continuation_success"] is False
    assert metadata["active_request_continuation_error_kind"] == "duplicate_payload_consume_after_continue"
    assert metadata["next_required_feature"] == "mailbox_state_after_active_continuation"


def test_repeated_verify_detection():
    metadata = apply_mod.build_v4t_active_continuation_metadata(
        commit_result(repeated_verify_after_commit_detected=True),
        max_steps=1,
        step_count=1,
    )
    assert metadata["active_continuation_success"] is False
    assert metadata["active_request_continuation_error_kind"] == "repeated_verify_after_commit_detected"
    assert metadata["next_required_feature"] == "scheduler_state_after_active_continuation"


def test_no_active_seq_skips_continuation():
    metadata = apply_mod.build_v4t_active_continuation_metadata(
        commit_result(
            unfinished_seq_ids_at_completion_check=[],
            active_seq_ids_at_completion_check=[],
            scheduler_active_seq_ids_at_completion=[],
            next_required_feature="result_finalization_after_breadth_only_completion",
        ),
        max_steps=1,
        step_count=1,
    )
    assert metadata["active_continuation_attempted"] is False
    assert metadata["active_continuation_success"] is False


def test_pending_mailbox_payload_gets_drain_diagnostic():
    metadata = apply_mod.build_v4t_active_continuation_metadata(
        commit_result(mailbox_pending_payload_ids_at_completion=["1:3:0:4"]),
        max_steps=1,
        step_count=1,
    )
    assert metadata["active_continuation_success"] is False
    assert metadata["next_required_feature"] == "mailbox_payload_after_active_continuation"


def test_active_diagnostic_uses_fallback_seq_ids():
    metadata = apply_mod.build_v4t_active_continuation_metadata(
        commit_result(
            unfinished_seq_ids_at_completion_check=[],
            active_seq_ids_at_completion_check=[],
            scheduler_active_seq_ids_at_completion=[],
        ),
        max_steps=1,
        step_count=1,
        fallback_active_seq_ids=[7],
    )
    assert metadata["active_continuation_attempted"] is True
    assert metadata["active_continuation_seq_ids"] == [7]
    assert metadata["active_continuation_success"] is True


def test_target_mailbox_route_wires_active_diagnostic_before_generic_raise():
    runner_path = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_model_runner.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        source = f.read()
    active_branch = 'current_next_required == "active_request_continuation_after_breadth_only_step"'
    continue_call = "self._try_continue_v4t_active_requests(exec_seqs, step_plan, trace_record, commit_result, output)"
    generic_raise = "mailbox verify guarded commit probe reached next explicit diagnostic"
    assert active_branch in source
    assert source.index(active_branch) < source.index(generic_raise)
    assert source.index(continue_call) < source.index(generic_raise)
    assert '"active_request_continuation_limit_reached"' in source[source.index(active_branch) : source.index(generic_raise)]


def mailbox_payload(
    *,
    plan_id: int,
    seq_id: int = 0,
    home_batch_id: int = 0,
    token_ids=None,
):
    token_ids = [11] if token_ids is None else list(token_ids)
    return mailbox_mod.MailboxPayload(
        plan_id=plan_id,
        producer_role="draft",
        producer_home_batch_id=home_batch_id,
        target_home_batch_id=home_batch_id,
        draft_home_batch_id=home_batch_id,
        seq_id=seq_id,
        request_id=seq_id,
        home_batch_id=home_batch_id,
        gamma=4,
        layout_kind="variable_offsets",
        protocol_version=1,
        draft_token_ids=token_ids,
        per_seq_length=len(token_ids),
        offset=0,
        logical_step=plan_id,
        metadata={"payload_id": f"{plan_id}:{home_batch_id}:{seq_id}:0:{len(token_ids)}"},
    )


def test_same_available_payload_duplicate_put_is_idempotent_skip():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    payload = mailbox_payload(plan_id=8, token_ids=[101])
    mailbox.put_payloads(0, [payload], plan_id=8, producer_role="draft")
    mailbox.put_payloads(0, [payload], plan_id=8, producer_role="draft")
    stats = mailbox.stats()
    assert stats["duplicate_put_count"] == 1
    assert stats["duplicate_put_idempotent_skip_count"] == 1
    assert stats["duplicate_put_conflict_count"] == 0
    assert stats["put_count"] == 1


def test_same_key_different_available_payload_hard_fails_as_conflict():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    mailbox.put_payloads(0, [mailbox_payload(plan_id=8, token_ids=[101])], plan_id=8, producer_role="draft")
    try:
        mailbox.put_payloads(0, [mailbox_payload(plan_id=8, token_ids=[202])], plan_id=8, producer_role="draft")
    except mailbox_mod.STSpecMailboxError as exc:
        assert exc.kind == "duplicate_put_conflict"
        assert exc.context["home_batch_id"] == 0
        assert exc.context["seq_id"] == 0
        assert exc.context["plan_id"] == 8
    else:
        raise AssertionError("conflicting duplicate put should fail")


def test_available_payload_context_reports_semantic_conflict_before_put():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    existing = mailbox_payload(plan_id=4, token_ids=[101])
    incoming = mailbox_payload(plan_id=8, token_ids=[202])
    mailbox.put_payloads(0, [existing], plan_id=4, producer_role="draft")
    contexts = mailbox.available_payload_contexts_for([incoming])
    assert len(contexts) == 1
    assert contexts[0]["same_payload"] is False
    assert contexts[0]["existing_plan_id"] == 4
    assert contexts[0]["incoming_plan_id"] == 8
    assert contexts[0]["lifecycle_state"] == "available"


def test_consumed_lifecycle_allows_new_active_continuation_plan_for_same_home_seq():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    first = mailbox_payload(plan_id=8, token_ids=[101])
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    mailbox.apply_payload_lifecycle(
        plan_id=8,
        target_home_batch_id=0,
        consumed_token_count_by_payload_id={first.payload_id: 1},
        invalidated_token_count_by_payload_id={},
    )
    second = mailbox_payload(plan_id=9, token_ids=[202])
    mailbox.put_payloads(0, [second], plan_id=9, producer_role="draft")
    result = mailbox.get_payloads(0, [0], plan_id=9, consumer_role="target")
    assert result.success is True
    assert result.payloads[0].payload_id == second.payload_id
    stats = mailbox.stats()
    assert stats["duplicate_put_count"] == 0
    assert stats["put_count"] == 2


def test_stale_producer_payload_allows_later_plan_for_same_home_seq():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    first = mailbox_payload(plan_id=4, token_ids=[101])
    mailbox.put_payloads(0, [first], plan_id=4, producer_role="draft")
    stale = mailbox.mark_payloads_stale(
        [first.payload_id],
        plan_id=4,
        reason="draft_transport_envelope_recorded",
    )
    assert stale[first.payload_id]["lifecycle_state"] == "stale"
    second = mailbox_payload(plan_id=8, token_ids=[202])
    assert mailbox.available_payload_contexts_for([second]) == []
    mailbox.put_payloads(0, [second], plan_id=8, producer_role="draft")
    result = mailbox.get_payloads(0, [0], plan_id=8, consumer_role="target")
    assert result.success is True
    assert result.payloads[0].payload_id == second.payload_id


def test_same_plan_stale_same_payload_is_idempotent_skip_in_mailbox():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    first = mailbox_payload(plan_id=8, token_ids=[101])
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    mailbox.mark_payloads_stale([first.payload_id], plan_id=8, reason="draft_transport_envelope_recorded")
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    stats = mailbox.stats()
    assert stats["duplicate_put_count"] == 1
    assert stats["duplicate_put_idempotent_skip_count"] == 1
    assert stats["put_count"] == 1


def test_same_plan_stale_different_payload_is_conflict_in_mailbox():
    mailbox = mailbox_mod.STSpecPayloadMailbox()
    first = mailbox_payload(plan_id=8, token_ids=[101])
    second = mailbox_payload(plan_id=8, token_ids=[202])
    mailbox.put_payloads(0, [first], plan_id=8, producer_role="draft")
    mailbox.mark_payloads_stale([first.payload_id], plan_id=8, reason="draft_transport_envelope_recorded")
    try:
        mailbox.put_payloads(0, [second], plan_id=8, producer_role="draft")
    except mailbox_mod.STSpecMailboxError as exc:
        assert exc.kind == "duplicate_put_conflict"
    else:
        raise AssertionError("same-plan different payload should fail")


def test_draft_mailbox_route_guards_outstanding_available_before_put():
    runner_path = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_model_runner.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        source = f.read()
    guard_call = "self._guard_outstanding_draft_mailbox_payloads("
    put_call = "self.stspec_mailbox.put_payloads("
    stale_call = "self.stspec_mailbox.mark_payloads_stale("
    record_start = source.index("def _record_draft_mailbox_payloads")
    record_source = source[record_start:]
    assert guard_call in record_source
    assert record_source.index(guard_call) < record_source.index(put_call)
    assert stale_call in record_source
    assert "mailbox_payload_record_guard_hit" in source
    assert "mailbox_payload_put_skipped" in source


def test_draft_mailbox_record_guard_resets_at_generation_boundaries():
    runner_path = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_model_runner.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        source = f.read()
    assert "def _reset_draft_mailbox_record_guard" in source
    assert source.count("self._reset_draft_mailbox_record_guard()") >= 4


def test_active_continuation_trace_has_v4v_progress_fields():
    runner_path = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_model_runner.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        source = f.read()
    for needle in (
        "stspec_active_continuation_max_steps_effective",
        "stspec_active_continuation_max_steps_source",
        "active_continuation_no_progress",
        "active_continuation_no_progress_reason",
        "active_continuation_progress_too_slow",
        "active_request_continuation_no_effective_token_progress",
        "active_request_continuation_acceptance_too_low",
        "active_request_continuation_target_correction_shadow_only",
        "active_request_continuation_target_correction_wrong_sequence",
        "active_request_continuation_target_correction_rolled_back",
        "active_request_continuation_output_snapshot_mismatch",
        "active_request_continuation_completion_token_export_missing",
        "active_request_continuation_partial_batch_starvation",
        "active_request_continuation_output_not_committed",
        "active_request_continuation_completion_gate_mismatch",
        "active_request_continuation_steps_insufficient",
        "active_continuation_effective_token_progress_by_step",
        "active_continuation_bookkeeping_progress_by_step",
        "active_continuation_output_tokens_before_after_by_step",
        "active_continuation_accepted_tokens_before_after_by_step",
        "active_continuation_zero_accept_step_count",
        "active_continuation_average_acceptance_rate",
        "active_continuation_zero_accept_correction_available_by_step",
        "active_continuation_zero_accept_correction_token_ids_by_step",
        "active_continuation_zero_accept_correction_commit_attempted_by_step",
        "active_continuation_zero_accept_correction_commit_success_by_step",
        "active_continuation_zero_accept_correction_failure_reason_by_step",
        "active_continuation_correction_diagnostic_priority",
        "active_continuation_acceptance_too_low_after_correction_checked",
        "active_continuation_target_correction_missing",
        "active_continuation_reject_recovery_missing",
        "active_continuation_target_correction_not_committed",
        "active_continuation_zero_accept_correction_checked",
        "active_continuation_zero_accept_correction_failure",
        "active_continuation_zero_accept_correction_rows_count",
        "active_continuation_next_step_prefix_token_ids_by_step",
        "active_continuation_next_step_prefix_len_by_step",
        "active_continuation_next_step_contains_correction_by_step",
        "active_continuation_target_draft_prefix_divergence",
        "active_continuation_target_correction_not_in_next_prefix",
        "active_continuation_kv_state_mismatch",
        "active_continuation_prefix_len_mismatch",
        "active_continuation_position_mismatch",
        "active_continuation_slot_mapping_mismatch",
        "active_continuation_target_correction_available_by_step",
        "active_continuation_target_correction_committed_by_step",
        "active_continuation_target_correction_commit_seq_ids",
        "active_continuation_target_correction_commit_request_ids",
        "active_continuation_target_correction_output_delta_by_step",
        "active_continuation_target_correction_shadow_only",
        "active_continuation_target_correction_wrong_sequence",
        "active_continuation_target_correction_rolled_back",
        "active_continuation_output_snapshot_mismatch",
        "active_continuation_completion_token_export_missing",
        "active_continuation_sequence_output_len_before_after_by_step",
        "active_continuation_prefix_len_before_after_by_step",
        "active_continuation_starving_seq_ids",
        "active_continuation_last_advanced_step_by_seq",
        "active_continuation_output_not_committed",
        "active_continuation_completion_gate_mismatch",
        "active_continuation_steps_insufficient",
        "active_continuation_recommended_min_steps",
        "active_request_continuation_no_progress",
        "active_continuation_step_history",
        "active_continuation_home_batch_history",
        "active_continuation_total_output_token_delta",
        "active_continuation_total_accepted_token_delta",
        "active_continuation_remaining_output_tokens_by_seq",
        "active_continuation_remaining_tokens_to_max_by_seq",
        "active_continuation_plan_id_history",
        "active_continuation_progress_by_step",
        "_build_v4v_active_continuation_snapshot",
        "_classify_v4v_active_continuation_progress",
        "_classify_v4v_active_continuation_limit",
    ):
        assert needle in source


def test_max_steps_cli_config_is_not_hard_coded_in_runner():
    eval_path = os.path.join(REPO_ROOT, "benchmark", "eval_multi_slo.py")
    runner_path = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_model_runner.py")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_source = f.read()
    with open(runner_path, "r", encoding="utf-8") as f:
        runner_source = f.read()
    assert "--stspec-active-continuation-max-steps" in eval_source
    assert '"stspec_active_continuation_max_steps": args.stspec_active_continuation_max_steps' in eval_source
    assert "def _v4v_active_continuation_max_steps" in runner_source
    assert 'return int(getattr(self.global_config, "stspec_active_continuation_max_steps")), "config"' in runner_source
    assert "max_steps, max_steps_source = self._v4v_active_continuation_max_steps()" in runner_source


def test_limit_reached_is_reclassified_to_specific_progress_diagnostic():
    runner_path = os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "pearl_model_runner.py")
    with open(runner_path, "r", encoding="utf-8") as f:
        source = f.read()
    limit_branch = 'next_feature == "active_request_continuation_limit_reached"'
    classifier = "self._classify_v4v_active_continuation_limit("
    generic_raise = "V4T active request continuation failed; error={error}"
    assert limit_branch in source
    assert classifier in source
    assert source.index(limit_branch) < source.index(generic_raise)
    assert "active_request_continuation_steps_insufficient" in source
    assert "active_request_continuation_completion_gate_mismatch" in source
    assert "active_request_continuation_output_not_committed" in source
    assert "active_request_continuation_acceptance_too_low" in source
    assert "active_continuation_acceptance_too_low_after_correction_checked" in source
    classify_source = source[source.index("def _classify_v4v_active_continuation_limit"):]
    assert "build_v4w_zero_accept_correction_diagnostics(progress_history)" in classify_source
    assert classify_source.index("build_v4w_zero_accept_correction_diagnostics(progress_history)") < classify_source.index("active_request_continuation_no_effective_token_progress")
    assert "active_request_continuation_partial_batch_starvation" in source
    assert "active_request_continuation_no_effective_token_progress" in source
    assert "mailbox_payload_after_active_continuation" in source


class RejectRecoverySeq:
    def __init__(self):
        self.seq_id = 1
        self.request_id = "r1"
        self.token_ids = [7, 101]
        self.num_tokens = 2
        self.num_prompt_tokens = 2
        self.last_token = 101
        self.block_table = [1]
        self.home_batch_id = 0
        self.ignore_eos = False
        self.first_token_ts = None
        self.finish_ts = None
        self.is_finished = False
        self.max_tokens = 32
        self.num_acc_tokens = []
        self.cur_acc_tokens = 0
        self.trace_stats = {"scheduled_iterations": [], "accepted_tokens": 0, "invalidated_predraft_tokens": 0}

    def __len__(self):
        return self.num_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(int(token_id))
        self.last_token = int(token_id)
        self.num_tokens += 1

    def record_accepted(self, accepted_len: int):
        self.trace_stats["accepted_tokens"] += int(accepted_len)

    def record_invalidated_predraft(self, invalidated_len: int):
        self.trace_stats["invalidated_predraft_tokens"] += int(invalidated_len)


def test_zero_accept_reject_recovery_commits_target_correction_token():
    seqs = [RejectRecoverySeq()]
    verify_input = transport_mod.TargetForwardFromMailboxInput(
        plan_id=12,
        target_home_batch_id=0,
        seq_ids=[1],
        request_ids=["r1"],
        input_token_ids=[111],
        per_seq_lengths=[1],
        offsets=[0],
        total_tokens=1,
        gamma=4,
        positions=[2],
        kv_slot_ids=[2002],
        source_mailbox_payload_ids=["12:1:0:0:1"],
        source_draft_plan_id=12,
    )
    step_plan = SimpleNamespace(plan_id=12, target_home_batch_id=0, actual_target_exec_seq_ids=[1])
    verify_result = apply_mod.build_mailbox_verify_result(verify_input, target_token_ids=[999], output_owner_rank=0)
    apply_plan = apply_mod.build_mailbox_verify_apply_plan(verify_result, seqs, step_plan, max_model_len=32)
    commit_plan = apply_mod.build_mailbox_verify_commit_plan(verify_result, apply_plan, seqs, step_plan, commit_allowed=True)
    kv_plan = apply_mod.build_mailbox_kv_commit_plan(verify_result, commit_plan, seqs, step_plan, max_model_len=32, commit_allowed=True)
    result = apply_mod.run_mailbox_verify_commit_probe(
        commit_plan,
        seqs,
        current_rank=0,
        output_owner_rank=0,
        kv_commit_plan=kv_plan,
    )
    assert verify_result.accepted_lengths_by_seq == {1: 0}
    assert commit_plan.target_correction_token_ids_by_seq == {1: [999]}
    assert result.success is True
    assert result.sequence_state_after[1]["output_token_count"] == result.sequence_state_before[1]["output_token_count"] + 1
    assert result.sequence_state_after[1]["token_ids"][-1] == 999
    assert seqs[0].completion_token_ids == [999]
    assert 111 not in seqs[0].completion_token_ids
    assert seqs[0].trace_stats["invalidated_predraft_tokens"] == 1
    assert seqs[0].trace_stats["accepted_tokens"] == 0


def _zero_accept_progress_item(**overrides):
    item = {
        "step_count": 1,
        "plan_id": 8,
        "request_ids_by_seq": {"1": "r1"},
        "expected_len_by_seq": {"1": 1},
        "accepted_len_by_seq": {"1": 0},
        "rejected_len_by_seq": {"1": 1},
        "rejected_draft_token_ids_by_seq": {"1": [111]},
        "target_correction_token_ids_by_seq": {"1": [999]},
        "target_correction_available_by_seq": {"1": True},
        "target_correction_commit_attempted_by_seq": {"1": True},
        "target_correction_committed_by_seq": {"1": True},
        "target_correction_output_delta_by_seq": {"1": 1},
        "sequence_object_identity_by_seq": {"1": 12345},
        "sequence_output_len_before_after_by_seq": {"1": {"before": 0, "after": 1}},
        "sequence_token_ids_before_after_by_seq": {"1": {"before": [7, 101], "after": [7, 101, 999]}},
        "completion_token_ids_by_seq": {"1": [999]},
        "completion_token_len_before_after_by_seq": {"1": {"before": 0, "after": 1}},
        "service_metadata_num_output_tokens_before_after_by_seq": {"1": {"before": 0, "after": 1}},
    }
    item.update(overrides)
    return item


def test_zero_accept_correction_missing_has_priority_over_acceptance_too_low():
    diagnostic = apply_mod.build_v4w_zero_accept_correction_diagnostics([
        _zero_accept_progress_item(
            target_correction_token_ids_by_seq={"1": []},
            target_correction_available_by_seq={"1": False},
            target_correction_commit_attempted_by_seq={"1": False},
            target_correction_committed_by_seq={"1": False},
            target_correction_output_delta_by_seq={"1": 0},
        )
    ])
    assert diagnostic["selected_next_required_feature"] == "active_request_continuation_target_correction_missing"
    assert diagnostic["active_continuation_target_correction_missing"] is True
    assert diagnostic["active_continuation_reject_recovery_missing"] is True


def test_zero_accept_correction_available_but_not_committed_is_specific():
    diagnostic = apply_mod.build_v4w_zero_accept_correction_diagnostics([
        _zero_accept_progress_item(
            target_correction_committed_by_seq={"1": False},
            target_correction_output_delta_by_seq={"1": 0},
        )
    ])
    assert diagnostic["selected_next_required_feature"] == "active_request_continuation_target_correction_not_committed"
    assert diagnostic["active_continuation_target_correction_not_committed"] is True


def test_zero_accept_correction_shadow_wrong_rollback_and_export_diagnostics():
    cases = [
        ("target_correction_shadow_only_by_seq", "active_request_continuation_target_correction_shadow_only"),
        ("target_correction_wrong_sequence_by_seq", "active_request_continuation_target_correction_wrong_sequence"),
        ("target_correction_rolled_back_by_seq", "active_request_continuation_target_correction_rolled_back"),
        ("completion_token_export_missing_by_seq", "active_request_continuation_completion_token_export_missing"),
    ]
    for field, expected in cases:
        diagnostic = apply_mod.build_v4w_zero_accept_correction_diagnostics([
            _zero_accept_progress_item(**{field: {"1": True}, "target_correction_committed_by_seq": {"1": False}})
        ])
        assert diagnostic["selected_next_required_feature"] == expected


def test_zero_accept_correction_committed_allows_acceptance_fallback():
    diagnostic = apply_mod.build_v4w_zero_accept_correction_diagnostics([
        _zero_accept_progress_item()
    ])
    assert diagnostic["zero_accept_correction_checked"] is True
    assert diagnostic["zero_accept_correction_failure"] is False
    assert diagnostic["selected_next_required_feature"] is None
    assert diagnostic["active_continuation_zero_accept_correction_commit_success_by_step"][0]["values"] == {"1": True}


def _next_prefix_progress_item(**overrides):
    item = {
        "step_count": 2,
        "plan_id": 12,
        "request_ids_by_seq": {"1": "r1"},
        "sequence_token_ids_before_after_by_seq": {"1": {"before": [7, 101, 999], "after": [7, 101, 999, 222]}},
        "prefix_len_before_after_by_seq": {"1": {"before": 3, "after": 4}},
        "kv_length_before_after_by_seq": {"1": {"before": 3, "after": 4}},
        "position_ids_by_seq": {"1": [3]},
        "slot_mapping_prefix_len_by_seq": {"1": 3},
    }
    item.update(overrides)
    return item


def test_correction_committed_but_missing_from_next_prefix_is_specific():
    diagnostic = apply_mod.build_v4w_next_step_prefix_diagnostics([
        _zero_accept_progress_item(),
        _next_prefix_progress_item(
            sequence_token_ids_before_after_by_seq={"1": {"before": [7, 101], "after": [7, 101, 222]}},
            prefix_len_before_after_by_seq={"1": {"before": 2, "after": 3}},
        ),
    ])
    assert diagnostic["selected_next_required_feature"] == "active_request_continuation_target_correction_not_in_next_prefix"
    assert diagnostic["active_continuation_target_correction_not_in_next_prefix"] is True


def test_target_draft_prefix_divergence_is_specific():
    diagnostic = apply_mod.build_v4w_next_step_prefix_diagnostics([
        _zero_accept_progress_item(),
        _next_prefix_progress_item(draft_side_sequence_token_ids_by_seq={"1": [7, 101]}),
    ])
    assert diagnostic["selected_next_required_feature"] == "active_request_continuation_target_draft_prefix_divergence"


def test_prefix_and_kv_mismatch_are_specific():
    prefix_diagnostic = apply_mod.build_v4w_next_step_prefix_diagnostics([
        _zero_accept_progress_item(),
        _next_prefix_progress_item(prefix_len_before_after_by_seq={"1": {"before": 2, "after": 4}}),
    ])
    assert prefix_diagnostic["selected_next_required_feature"] == "active_request_continuation_prefix_len_mismatch"
    kv_diagnostic = apply_mod.build_v4w_next_step_prefix_diagnostics([
        _zero_accept_progress_item(),
        _next_prefix_progress_item(kv_length_before_after_by_seq={"1": {"before": 2, "after": 4}}),
    ])
    assert kv_diagnostic["selected_next_required_feature"] == "active_request_continuation_kv_state_mismatch"


def test_next_prefix_aligned_allows_acceptance_fallback():
    diagnostic = apply_mod.build_v4w_next_step_prefix_diagnostics([
        _zero_accept_progress_item(),
        _next_prefix_progress_item(),
    ])
    assert diagnostic["next_step_prefix_checked"] is True
    assert diagnostic["next_step_prefix_failure"] is False
    assert diagnostic["active_continuation_next_step_contains_correction_by_step"][0]["values"] == {"1": True}


def main() -> None:
    test_unfinished_requests_attempt_active_continuation()
    test_fake_runner_state_initializes_and_resets()
    test_active_continuation_max_step_reached()
    test_active_continuation_allows_multiple_steps_before_limit()
    test_duplicate_consume_detection()
    test_repeated_verify_detection()
    test_no_active_seq_skips_continuation()
    test_pending_mailbox_payload_gets_drain_diagnostic()
    test_active_diagnostic_uses_fallback_seq_ids()
    test_target_mailbox_route_wires_active_diagnostic_before_generic_raise()
    test_same_available_payload_duplicate_put_is_idempotent_skip()
    test_same_key_different_available_payload_hard_fails_as_conflict()
    test_available_payload_context_reports_semantic_conflict_before_put()
    test_consumed_lifecycle_allows_new_active_continuation_plan_for_same_home_seq()
    test_stale_producer_payload_allows_later_plan_for_same_home_seq()
    test_same_plan_stale_same_payload_is_idempotent_skip_in_mailbox()
    test_same_plan_stale_different_payload_is_conflict_in_mailbox()
    test_draft_mailbox_route_guards_outstanding_available_before_put()
    test_draft_mailbox_record_guard_resets_at_generation_boundaries()
    test_active_continuation_trace_has_v4v_progress_fields()
    test_max_steps_cli_config_is_not_hard_coded_in_runner()
    test_limit_reached_is_reclassified_to_specific_progress_diagnostic()
    test_zero_accept_reject_recovery_commits_target_correction_token()
    test_zero_accept_correction_missing_has_priority_over_acceptance_too_low()
    test_zero_accept_correction_available_but_not_committed_is_specific()
    test_zero_accept_correction_shadow_wrong_rollback_and_export_diagnostics()
    test_zero_accept_correction_committed_allows_acceptance_fallback()
    test_correction_committed_but_missing_from_next_prefix_is_specific()
    test_target_draft_prefix_divergence_is_specific()
    test_prefix_and_kv_mismatch_are_specific()
    test_next_prefix_aligned_allows_acceptance_fallback()


if __name__ == "__main__":
    main()
