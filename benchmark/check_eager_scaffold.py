#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from nano_pearl.pearl_config import (  # type: ignore  # noqa: E402
        PEARLConfig,
        PHASE_1H0_EAGER_NOT_IMPLEMENTED,
        validate_eager_gamma,
    )
    from nano_pearl.pearl_engine.dual_batch import (  # type: ignore  # noqa: E402
        EAGER_STATE_CONSUMED,
        EAGER_STATE_DISCARDED,
        EAGER_STATE_DRAFTED_PENDING_PARENT,
        EAGER_STATE_READY_TO_VERIFY,
        EagerProposal,
        EagerProposalBuffer,
        LANE_EAGER,
        LANE_NORMAL,
        deserialize_eager_proposal_meta,
        eager_proposal_to_trace_dict,
        serialize_eager_proposal_meta,
    )
    from nano_pearl.pearl_engine.step_plan import RequestBudget, StepPlan  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    for module_name in list(sys.modules):
        if module_name == "nano_pearl" or module_name.startswith("nano_pearl."):
            sys.modules.pop(module_name, None)

    nano_pearl_stub = types.ModuleType("nano_pearl")
    nano_pearl_stub.__path__ = [os.path.join(REPO_ROOT, "nano_pearl")]
    pearl_engine_stub = types.ModuleType("nano_pearl.pearl_engine")
    pearl_engine_stub.__path__ = [os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine")]
    sys.modules["nano_pearl"] = nano_pearl_stub
    sys.modules["nano_pearl.pearl_engine"] = pearl_engine_stub

    utils_stub = types.ModuleType("nano_pearl.utils")
    logger_stub = types.ModuleType("nano_pearl.utils.pearl_logger")

    class _NoopLogger:
        def info(self, *args: Any, **kwargs: Any) -> None:
            return None

        def warning(self, *args: Any, **kwargs: Any) -> None:
            return None

    logger_stub.logger = _NoopLogger()
    logger_stub.get_model_name = lambda model: str(model)
    sys.modules["nano_pearl.utils"] = utils_stub
    sys.modules["nano_pearl.utils.pearl_logger"] = logger_stub

    if importlib.util.find_spec("transformers") is None:
        transformers_stub = types.ModuleType("transformers")

        class _AutoConfig:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("synthetic checker AutoConfig stub should not be used")

        transformers_stub.AutoConfig = _AutoConfig
        sys.modules["transformers"] = transformers_stub

    if importlib.util.find_spec("torch") is None:
        torch_stub = types.ModuleType("torch")
        dist_stub = types.ModuleType("torch.distributed")

        class _ProcessGroup:
            pass

        dist_stub.ProcessGroup = _ProcessGroup
        torch_stub.distributed = dist_stub
        sys.modules["torch"] = torch_stub
        sys.modules["torch.distributed"] = dist_stub

    sequence_stub = types.ModuleType("nano_pearl.pearl_engine.sequence")

    class _Sequence:
        pass

    sequence_stub.Sequence = _Sequence
    sys.modules["nano_pearl.pearl_engine.sequence"] = sequence_stub

    def _load_module(module_name: str, path: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module {module_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    step_plan_module = _load_module(
        "nano_pearl.pearl_engine.step_plan",
        os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "step_plan.py"),
    )
    dual_batch_module = _load_module(
        "nano_pearl.pearl_engine.dual_batch",
        os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine", "dual_batch.py"),
    )
    pearl_config_module = _load_module(
        "nano_pearl.pearl_config",
        os.path.join(REPO_ROOT, "nano_pearl", "pearl_config.py"),
    )

    PEARLConfig = pearl_config_module.PEARLConfig
    PHASE_1H0_EAGER_NOT_IMPLEMENTED = pearl_config_module.PHASE_1H0_EAGER_NOT_IMPLEMENTED
    validate_eager_gamma = pearl_config_module.validate_eager_gamma
    EAGER_STATE_CONSUMED = dual_batch_module.EAGER_STATE_CONSUMED
    EAGER_STATE_DISCARDED = dual_batch_module.EAGER_STATE_DISCARDED
    EAGER_STATE_DRAFTED_PENDING_PARENT = dual_batch_module.EAGER_STATE_DRAFTED_PENDING_PARENT
    EAGER_STATE_READY_TO_VERIFY = dual_batch_module.EAGER_STATE_READY_TO_VERIFY
    EagerProposal = dual_batch_module.EagerProposal
    EagerProposalBuffer = dual_batch_module.EagerProposalBuffer
    LANE_EAGER = dual_batch_module.LANE_EAGER
    LANE_NORMAL = dual_batch_module.LANE_NORMAL
    deserialize_eager_proposal_meta = dual_batch_module.deserialize_eager_proposal_meta
    eager_proposal_to_trace_dict = dual_batch_module.eager_proposal_to_trace_dict
    serialize_eager_proposal_meta = dual_batch_module.serialize_eager_proposal_meta
    RequestBudget = step_plan_module.RequestBudget
    StepPlan = step_plan_module.StepPlan


EAGER_ZERO_COUNTER_FIELDS = [
    "eager_tokens_generated",
    "eager_tokens_promoted",
    "eager_tokens_discarded",
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]

EAGER_EMPTY_LIST_FIELDS = [
    "eager_ready_seq_ids",
    "eager_active_seq_ids",
    "eager_promoted_seq_ids",
    "eager_discarded_seq_ids",
    "eager_verified_seq_ids",
    "eager_accepted_seq_ids",
    "eager_rejected_seq_ids",
]

REQUIRED_EAGER_META_KEYS = [
    "num_proposals",
    "payload_length",
    "gamma",
    "plan_id",
    "proposal_ids",
    "seq_ids",
    "parent_ids",
    "base_lens",
    "base_pre_verify",
    "proposal_lens",
    "to_verify_lens",
    "lane_kinds",
]


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError(
        "Unsupported trace format. Expected a raw list or a dict with "
        "'traces', 'records', or 'trace_records'."
    )


def expect_raises(fn, exc_type: type[BaseException], label: str) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"{label}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"{label}: expected {exc_type.__name__}")


def check_not_implemented_message(fn, expected_message: str) -> None:
    try:
        fn()
    except NotImplementedError as exc:
        assert str(exc) == expected_message, (
            f"NotImplementedError message mismatch: expected={expected_message!r}, got={str(exc)!r}"
        )
        return
    except Exception as exc:
        raise AssertionError(f"expected NotImplementedError, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError("expected NotImplementedError")


def make_eager_proposal(proposal_id: int, seq_id: int, *, parent_id: int | None = None) -> EagerProposal:
    return EagerProposal(
        proposal_id=proposal_id,
        seq_id=seq_id,
        request_id=f"request-{seq_id}",
        lane=LANE_EAGER,
        parent_proposal_id=parent_id,
        parent_kind=LANE_EAGER if parent_id is not None else LANE_NORMAL,
        parent_step_id=None if parent_id is None else 11,
        source_step_id=12,
        source_plan_id=34,
        home_batch_id=seq_id % 2,
        base_len=20 + seq_id,
        base_pre_verify=seq_id % 2 == 1,
        base_num_completion_tokens=seq_id,
        proposal_token_ids=[100 + proposal_id, 101 + proposal_id, 102 + proposal_id, 103 + proposal_id],
        to_be_verified_token_ids=[100 + proposal_id],
        proposal_len=4,
        state=EAGER_STATE_DRAFTED_PENDING_PARENT,
        valid=True,
    )


def check_buffer_lifecycle() -> None:
    buffer = EagerProposalBuffer()
    first = make_eager_proposal(10, 1)
    second = make_eager_proposal(11, 1, parent_id=10)

    buffer.store(first)
    assert buffer.size() == 1
    assert buffer.pending_seq_ids() == [1]
    expect_raises(lambda: buffer.store(make_eager_proposal(10, 2)), ValueError, "duplicate proposal_id")

    ready = buffer.mark_ready(10)
    assert ready.state == EAGER_STATE_READY_TO_VERIFY
    assert buffer.ready_seq_ids() == [1]
    assert [proposal.proposal_id for proposal in buffer.get_ready_by_seq_ids([1])] == [10]

    consumed = buffer.mark_consumed(10)
    assert consumed.state == EAGER_STATE_CONSUMED
    assert consumed.valid is False
    assert buffer.size() == 0

    buffer.store(second)
    assert buffer.size() == 1
    assert buffer.discard_by_seq_id(1, "synthetic parent rejected") == [11]
    assert second.state == EAGER_STATE_DISCARDED
    assert second.valid is False
    assert buffer.size() == 0

    inspected = buffer.inspect()
    assert inspected["state_counts"][EAGER_STATE_CONSUMED] == 1
    assert inspected["state_counts"][EAGER_STATE_DISCARDED] == 1


def check_metadata_roundtrip() -> None:
    proposals = [make_eager_proposal(20, 2), make_eager_proposal(21, 3, parent_id=20)]
    proposals[0].state = EAGER_STATE_READY_TO_VERIFY
    proposals[1].state = EAGER_STATE_DISCARDED
    proposals[1].valid = False

    meta, payload = serialize_eager_proposal_meta(proposals, gamma=4, plan_id=77)
    missing = [key for key in REQUIRED_EAGER_META_KEYS if key not in meta]
    assert not missing, f"metadata missing keys: {missing}"
    assert meta["num_proposals"] == 2
    assert meta["payload_length"] == len(payload)
    assert meta["gamma"] == 4
    assert meta["plan_id"] == 77
    assert meta["proposal_ids"] == [20, 21]
    assert meta["parent_ids"] == [None, 20]
    assert meta["base_lens"] == [22, 23]
    assert meta["base_pre_verify"] == [False, True]
    assert meta["lane_kinds"] == [LANE_EAGER, LANE_EAGER]

    roundtrip = deserialize_eager_proposal_meta(meta, payload)
    assert [eager_proposal_to_trace_dict(p) for p in roundtrip] == [
        eager_proposal_to_trace_dict(p) for p in proposals
    ]
    bad_meta = dict(meta)
    bad_meta["payload_length"] += 1
    expect_raises(lambda: deserialize_eager_proposal_meta(bad_meta, payload), ValueError, "payload length mismatch")


def make_step_plan(**overrides: Any) -> StepPlan:
    kwargs = {
        "plan_id": 1,
        "iteration_id": 1,
        "execution_mode": "dual_batch_pearl",
        "target_home_set": [1, 2],
        "target_eager_set": [10],
        "draft_home_set": [3, 4],
        "draft_eager_set": [10],
        "budgets": {
            1: RequestBudget(normal_gamma=4, eager_gamma=4),
            2: RequestBudget(normal_gamma=4, eager_gamma=0),
            3: RequestBudget(normal_gamma=4, eager_gamma=0),
            4: RequestBudget(normal_gamma=4, eager_gamma=0),
            10: RequestBudget(normal_gamma=4, eager_gamma=4),
        },
        "normal_gamma": 4,
        "eager_gamma": 4,
        "eager_new_selected_set": [1],
        "eager_continuing_set": [10],
        "eager_active_seq_ids": [10],
        "eager_ready_seq_ids": [10],
        "eager_proposal_ids_by_seq_id": {10: [20]},
        "eager_parent_proposal_ids_by_seq_id": {10: None},
        "eager_base_len_by_seq_id": {10: 31},
        "eager_base_pre_verify_by_seq_id": {10: True},
        "eager_parent_kind_by_seq_id": {10: LANE_NORMAL},
    }
    kwargs.update(overrides)
    return StepPlan(**kwargs)


def check_step_plan_validation() -> None:
    allowed = make_step_plan()
    allowed.validate_phase1h_eager_scaffold(enable_eager_execution=True)
    trace = allowed.to_trace_dict()
    assert trace["target_eager_set"] == [10]
    assert trace["draft_eager_set"] == [10]
    assert trace["eager_parent_proposal_ids_by_seq_id"] == {"10": None}
    assert trace["eager_base_len_by_seq_id"] == {"10": 31}
    assert trace["eager_base_pre_verify_by_seq_id"] == {"10": True}

    expect_raises(
        lambda: allowed.validate_phase1h_eager_scaffold(enable_eager_execution=False),
        AssertionError,
        "non-empty eager sets without enable",
    )
    expect_raises(
        lambda: make_step_plan(target_eager_set=[1]).validate_phase1h_eager_scaffold(True),
        AssertionError,
        "target eager overlaps target home",
    )
    expect_raises(
        lambda: make_step_plan(target_eager_set=[3]).validate_phase1h_eager_scaffold(True),
        AssertionError,
        "target eager overlaps draft home",
    )
    expect_raises(
        lambda: make_step_plan(draft_eager_set=[3]).validate_phase1h_eager_scaffold(True),
        AssertionError,
        "draft eager overlaps draft home",
    )
    bad_budget = make_step_plan(
        budgets={10: RequestBudget(normal_gamma=4, eager_gamma=2)},
        eager_new_selected_set=[],
        eager_continuing_set=[10],
    )
    expect_raises(
        lambda: bad_budget.validate_phase1h_eager_scaffold(True),
        AssertionError,
        "eager budget must be 0 or gamma",
    )
    expect_raises(
        lambda: make_step_plan(eager_new_selected_set=[99]).validate_phase1h_eager_scaffold(True),
        AssertionError,
        "new eager subset",
    )
    expect_raises(
        lambda: make_step_plan(eager_continuing_set=[99]).validate_phase1h_eager_scaffold(True),
        AssertionError,
        "continuing eager subset",
    )


def check_eager_gamma_validation() -> None:
    valid = SimpleNamespace(
        eager_gamma=4,
        max_eager_tokens_per_request=4,
        max_eager_tokens_per_step=8,
        max_eager_requests_per_step=2,
    )
    assert validate_eager_gamma(valid, 4) is True

    expect_raises(
        lambda: validate_eager_gamma(
            SimpleNamespace(
                eager_gamma=2,
                max_eager_tokens_per_request=2,
                max_eager_tokens_per_step=4,
                max_eager_requests_per_step=1,
            ),
            4,
        ),
        ValueError,
        "eager_gamma less than gamma",
    )
    expect_raises(
        lambda: validate_eager_gamma(
            SimpleNamespace(
                eager_gamma=4,
                max_eager_tokens_per_request=2,
                max_eager_tokens_per_step=8,
                max_eager_requests_per_step=1,
            ),
            4,
        ),
        ValueError,
        "max tokens per request",
    )
    expect_raises(
        lambda: validate_eager_gamma(
            SimpleNamespace(
                eager_gamma=4,
                max_eager_tokens_per_request=4,
                max_eager_tokens_per_step=6,
                max_eager_requests_per_step=1,
            ),
            4,
        ),
        ValueError,
        "step budget multiple",
    )
    expect_raises(
        lambda: validate_eager_gamma(
            SimpleNamespace(
                eager_gamma=4,
                max_eager_tokens_per_request=4,
                max_eager_tokens_per_step=4,
                max_eager_requests_per_step=2,
            ),
            4,
        ),
        ValueError,
        "request budget exceeds token budget",
    )
    check_not_implemented_message(
        lambda: PEARLConfig(
            draft_model_path="/synthetic/draft",
            target_model_path="/synthetic/target",
            enable_eager_execution=True,
        ),
        PHASE_1H0_EAGER_NOT_IMPLEMENTED,
    )


def run_synthetic_checks() -> None:
    check_buffer_lifecycle()
    check_metadata_roundtrip()
    check_step_plan_validation()
    check_eager_gamma_validation()
    print("Synthetic eager scaffold checks passed.")


def check_trace(path: Path) -> None:
    records = load_trace(path)
    dual_records = [
        record for record in records
        if record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    ]
    errors = []
    for idx, record in enumerate(dual_records):
        eager_trace_enabled = bool(record.get("eager_trace_enabled", False))
        if record.get("enable_eager_execution") not in (False, 0, None):
            errors.append(f"dual_record[{idx}] enable_eager_execution must be false")
        if record.get("eager_execution_enabled") not in (False, 0, None):
            errors.append(f"dual_record[{idx}] eager_execution_enabled must be false")
        if not eager_trace_enabled and record.get("target_eager_set"):
            errors.append(f"dual_record[{idx}] has non-empty target_eager_set")
        if not eager_trace_enabled and record.get("draft_eager_set"):
            errors.append(f"dual_record[{idx}] has non-empty draft_eager_set")
        for field in EAGER_EMPTY_LIST_FIELDS:
            if record.get(field):
                errors.append(f"dual_record[{idx}] {field} must be empty, got {record.get(field)}")
        for field in EAGER_ZERO_COUNTER_FIELDS:
            if int(record.get(field, 0) or 0) != 0:
                errors.append(f"dual_record[{idx}] {field} must be 0, got {record.get(field)}")
        for field in ("eager_buffer_size_before", "eager_buffer_size_after"):
            if int(record.get(field, 0) or 0) != 0:
                errors.append(f"dual_record[{idx}] {field} must be 0, got {record.get(field)}")
        if float(record.get("eager_waste_rate", 0.0) or 0.0) != 0.0:
            errors.append(f"dual_record[{idx}] eager_waste_rate must be 0")

    print(f"trace_records={len(records)}")
    print(f"dual_batch_pearl_records={len(dual_records)}")
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Eager scaffold trace check passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-0 eager scaffolding.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Only validate the provided trace.",
    )
    args = parser.parse_args()

    if not args.skip_synthetic:
        run_synthetic_checks()
    if args.trace is not None:
        check_trace(args.trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
