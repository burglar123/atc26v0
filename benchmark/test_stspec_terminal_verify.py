"""CPU-safe V4U terminal verify tuple tests."""

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


def seq(seq_id: int, *, pre_verify: bool = False):
    return SimpleNamespace(seq_id=seq_id, pre_verify=pre_verify)


def plan(accepted, targets):
    return SimpleNamespace(
        accepted_lengths_by_seq=dict(accepted),
        target_token_ids_by_seq={int(k): list(v) for k, v in targets.items()},
    )


def test_zero_accept_preverify_is_representable():
    rows, metadata = apply_mod.build_terminal_verify_tuple_rows(
        plan({1: 0}, {1: [901]}),
        [seq(1, pre_verify=True)],
        gamma=4,
    )
    assert metadata["terminal_verify_tuple_success"] is True
    assert metadata["terminal_verify_zero_accept_supported"] is True
    assert rows == [[0], [4], [901], [0]]


def test_partial_accept_decode_is_representable():
    rows, metadata = apply_mod.build_terminal_verify_tuple_rows(
        plan({1: 2}, {1: [101, 102, 903, 904]}),
        [seq(1)],
        gamma=4,
    )
    assert metadata["terminal_verify_tuple_success"] is True
    assert metadata["terminal_verify_partial_accept_supported"] is True
    assert rows == [[0], [2], [903], [0]]


def test_all_accept_decode_is_representable():
    rows, metadata = apply_mod.build_terminal_verify_tuple_rows(
        plan({1: 4}, {1: [101, 102, 103, 104]}),
        [seq(1)],
        gamma=4,
    )
    assert metadata["terminal_verify_tuple_success"] is True
    assert rows == [[1], [0], [-1], [0]]


def test_mixed_accept_lengths_are_representable():
    rows, metadata = apply_mod.build_terminal_verify_tuple_rows(
        plan({1: 0, 3: 2, 5: 4}, {1: [901], 3: [301, 302, 903, 904], 5: [501, 502, 503, 504]}),
        [seq(1, pre_verify=True), seq(3), seq(5)],
        gamma=4,
    )
    assert metadata["terminal_verify_tuple_success"] is True
    assert rows == [[0, 0, 1], [4, 2, 0], [901, 903, -1], [0, 0, 0]]


def test_missing_revise_token_gets_explicit_diagnostic():
    rows, metadata = apply_mod.build_terminal_verify_tuple_rows(
        plan({1: 0}, {1: []}),
        [seq(1, pre_verify=True)],
        gamma=4,
    )
    assert rows == [[], [], [], []]
    assert metadata["terminal_verify_tuple_success"] is False
    assert metadata["next_required_feature"] == "draft_verify_receiver_partial_accept_after_breadth_only"


def test_terminal_verify_tuple_json_serializable():
    rows, metadata = apply_mod.build_terminal_verify_tuple_rows(
        plan({1: 0, 3: 4}, {1: [901], 3: [301, 302, 303, 304]}),
        [seq(1, pre_verify=True), seq(3)],
        gamma=4,
    )
    json.dumps({"rows": rows, "metadata": metadata}, sort_keys=True, default=str)


def main() -> None:
    test_zero_accept_preverify_is_representable()
    test_partial_accept_decode_is_representable()
    test_all_accept_decode_is_representable()
    test_mixed_accept_lengths_are_representable()
    test_missing_revise_token_gets_explicit_diagnostic()
    test_terminal_verify_tuple_json_serializable()


if __name__ == "__main__":
    main()
