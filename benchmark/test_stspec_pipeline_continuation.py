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
    runner.stspec_active_continuation_step_count = 2
    apply_mod.initialize_v4t_active_continuation_runner_state(runner)
    assert runner.stspec_active_continuation_step_count == 2
    apply_mod.reset_v4t_active_continuation_runner_state(runner)
    assert runner.stspec_active_continuation_step_count == 0


def test_active_continuation_max_step_reached():
    metadata = apply_mod.build_v4t_active_continuation_metadata(commit_result(), max_steps=1, step_count=2)
    assert metadata["active_continuation_attempted"] is True
    assert metadata["active_continuation_success"] is False
    assert metadata["active_continuation_limit_reached"] is True
    assert metadata["active_continuation_error_kind"] == "active_request_continuation_limit_reached"
    assert metadata["next_required_feature"] == "active_request_continuation_limit_reached"


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


def main() -> None:
    test_unfinished_requests_attempt_active_continuation()
    test_fake_runner_state_initializes_and_resets()
    test_active_continuation_max_step_reached()
    test_duplicate_consume_detection()
    test_repeated_verify_detection()
    test_no_active_seq_skips_continuation()
    test_pending_mailbox_payload_gets_drain_diagnostic()


if __name__ == "__main__":
    main()
