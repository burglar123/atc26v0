#!/usr/bin/env python3
"""CPU-safe diagnostics for the legacy-equivalent ST-Spec StepPlan scaffold.

The scheduler modules used here are imported under temporary lightweight package
stubs so this diagnostic does not pull in GPU model modules. The stubs are
strictly scoped to the diagnostic context and are removed before returning, so
pytest collection of other tests can import the real ``nano_pearl`` package.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
import importlib.util
import pickle
import sys
import types
from types import SimpleNamespace
from typing import Any, Iterator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

_NANO_PREFIX = "nano_pearl"


@contextmanager
def cpu_safe_stspec_modules() -> Iterator[SimpleNamespace]:
    """Import scheduler-only ST-Spec modules without polluting sys.modules.

    Importing ``nano_pearl.pearl_engine.scheduler`` normally executes the public
    package ``__init__`` first, which imports GPU-heavy engine modules. This
    context temporarily installs package stubs for just the package namespace,
    imports the scheduler-only modules, and then restores every pre-existing
    ``nano_pearl`` entry exactly as it was.
    """

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == _NANO_PREFIX or name.startswith(f"{_NANO_PREFIX}.")
    }
    try:
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

        scheduler_mod = importlib.import_module("nano_pearl.pearl_engine.scheduler")
        sequence_mod = importlib.import_module("nano_pearl.pearl_engine.sequence")
        stspec_plan_mod = importlib.import_module("nano_pearl.pearl_engine.stspec_plan")
        yield SimpleNamespace(
            Scheduler=scheduler_mod.Scheduler,
            Sequence=sequence_mod.Sequence,
            stspec_plan=stspec_plan_mod,
            PlanRole=stspec_plan_mod.PlanRole,
        )
    finally:
        for name in list(sys.modules):
            if name == _NANO_PREFIX or name.startswith(f"{_NANO_PREFIX}."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def make_scheduler(modules: SimpleNamespace) -> Any:
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        eos=-1,
        num_kvcache_blocks=16,
        kvcache_block_size=256,
    )
    return modules.Scheduler(config)


def add_dummy_requests(scheduler: Any, modules: SimpleNamespace) -> list[Any]:
    seqs = [
        modules.Sequence(
            [1, 2, 3],
            request_id="req-a",
            slo_class="tight",
            slo_tpot_ms=25.0,
            per_request_gamma=2,
        ),
        modules.Sequence(
            [4, 5],
            request_id="req-b",
            slo_class="relaxed",
            slo_tpot_ms=100.0,
            per_request_gamma=8,
        ),
    ]
    for seq in seqs:
        assert seq.home_batch_id is None
        scheduler.add(seq)
    assert [seq.home_batch_id for seq in seqs] == [0, 1]
    return seqs


def assert_legacy_plan_fields(
    step_plan: Any,
    seqs: list[Any],
    modules: SimpleNamespace,
    *,
    is_prefill: bool,
    expected_budget: int,
) -> None:
    seq_ids = [seq.seq_id for seq in seqs]
    assert step_plan.legacy_equivalent is True
    assert step_plan.scheduled_seq_ids == seq_ids
    assert step_plan.eager_seq_ids == []
    assert all(request.is_eager is False for request in step_plan.requests)
    expected_home_batch_ids = {seq.seq_id: idx % 2 for idx, seq in enumerate(seqs)}
    assert {request.seq_id: request.home_batch_id for request in step_plan.requests} == expected_home_batch_ids
    assert all(request.eager_budget == 0 for request in step_plan.requests)
    assert all(request.draft_budget == expected_budget for request in step_plan.requests)
    assert all(request.effective_gamma == 4 for request in step_plan.requests)
    if is_prefill:
        assert all(request.role == modules.PlanRole.PREFILL for request in step_plan.requests)
    else:
        assert all(request.role == modules.PlanRole.DRAFT for request in step_plan.requests)
        assert step_plan.draft_home_seq_ids == seq_ids
    assert step_plan.is_eager_per_seq == {seq.seq_id: False for seq in seqs}
    assert step_plan.home_batch_id_per_seq == expected_home_batch_ids
    assert step_plan.effective_gamma_per_seq == {seq.seq_id: 4 for seq in seqs}
    signature = step_plan.signature()
    assert signature["plan_id"] == step_plan.plan_id
    assert signature["legacy_equivalent"] is True
    assert signature["scheduled_seq_ids"] == seq_ids
    assert signature["request_ids"] == [seq.request_id for seq in seqs]
    assert signature["effective_gamma_per_seq"] == {str(seq.seq_id): 4 for seq in seqs}
    assert signature["home_batch_id_per_seq"] == {
        str(seq_id): home_batch_id
        for seq_id, home_batch_id in expected_home_batch_ids.items()
    }
    assert signature["is_eager_per_seq"] == {str(seq.seq_id): False for seq in seqs}
    assert modules.stspec_plan.step_plan_digest(step_plan) == step_plan.digest()
    if is_prefill:
        assert step_plan.plan_two_batch_shadow is False
        assert step_plan.target_home_batch_id is None
        assert step_plan.draft_home_batch_id is None
        assert step_plan.target_batch_seq_ids == []
        assert step_plan.draft_home_batch_seq_ids == []
    else:
        assert step_plan.plan_two_batch_shadow is True
        assert step_plan.target_home_batch_id in {0, 1}
        assert step_plan.draft_home_batch_id in {0, 1}
        assert step_plan.target_home_batch_id != step_plan.draft_home_batch_id
        assert step_plan.target_batch_seq_ids == [
            seq.seq_id for seq in seqs if seq.home_batch_id == step_plan.target_home_batch_id
        ]
        assert step_plan.draft_home_batch_seq_ids == [
            seq.seq_id for seq in seqs if seq.home_batch_id == step_plan.draft_home_batch_id
        ]
        flipped_plan = modules.stspec_plan.build_legacy_step_plan(
            plan_id=step_plan.plan_id,
            seqs=seqs,
            is_prefill=False,
            runner_role=step_plan.runner_role,
            execution_mode=step_plan.execution_mode,
            decode_ready_mode=step_plan.decode_ready_mode,
            default_gamma=4,
            target_home_batch_id=step_plan.draft_home_batch_id,
            draft_home_batch_id=step_plan.target_home_batch_id,
        )
        assert flipped_plan.digest() != step_plan.digest()


def run_stspec_plan_scaffold_diagnostics(modules: SimpleNamespace) -> None:
    legacy_scheduler = make_scheduler(modules)
    planned_scheduler = make_scheduler(modules)
    legacy_seqs = add_dummy_requests(legacy_scheduler, modules)
    planned_seqs = add_dummy_requests(planned_scheduler, modules)

    legacy_prefill, legacy_is_prefill = legacy_scheduler.schedule()
    planned_prefill, planned_is_prefill, prefill_plan = planned_scheduler.schedule_with_plan(
        runner_role="draft_prefill",
        execution_mode="parallel_pearl",
        decode_ready_mode=False,
        default_gamma=4,
    )

    assert [seq.home_batch_id for seq in planned_seqs] == [0, 1]
    assert [seq.service_metadata()["home_batch_id"] for seq in planned_seqs] == [0, 1]
    assert [pickle.loads(pickle.dumps(seq)).home_batch_id for seq in planned_seqs] == [0, 1]

    assert [seq.request_id for seq in planned_prefill] == [seq.request_id for seq in legacy_prefill]
    assert planned_is_prefill == legacy_is_prefill == True
    assert planned_prefill == planned_seqs
    assert_legacy_plan_fields(
        prefill_plan,
        planned_prefill,
        modules,
        is_prefill=True,
        expected_budget=1,
    )

    legacy_decode, legacy_decode_is_prefill = legacy_scheduler.schedule()
    planned_decode, planned_decode_is_prefill, decode_plan = planned_scheduler.schedule_with_plan(
        runner_role="draft",
        execution_mode="parallel_pearl",
        decode_ready_mode=False,
        default_gamma=4,
    )

    assert [seq.request_id for seq in planned_decode] == [seq.request_id for seq in legacy_decode]
    assert planned_decode_is_prefill == legacy_decode_is_prefill == False
    assert planned_decode == planned_seqs
    assert_legacy_plan_fields(
        decode_plan,
        planned_decode,
        modules,
        is_prefill=False,
        expected_budget=4,
    )
    assert decode_plan.target_home_batch_id == 0
    assert decode_plan.draft_home_batch_id == 1

    second_decode, second_is_prefill, second_plan = planned_scheduler.schedule_with_plan(
        runner_role="draft",
        execution_mode="parallel_pearl",
        decode_ready_mode=False,
        default_gamma=4,
    )
    assert second_is_prefill is False
    assert second_decode == planned_seqs
    assert second_plan.target_home_batch_id == 1
    assert second_plan.draft_home_batch_id == 0

    ar_scheduler = make_scheduler(modules)
    add_dummy_requests(ar_scheduler, modules)
    ar_scheduler.schedule()
    ar_decode, ar_is_prefill, ar_plan = ar_scheduler.schedule_with_plan(
        runner_role="verify",
        execution_mode="ar",
        decode_ready_mode=False,
        default_gamma=4,
    )
    assert ar_is_prefill is False
    assert all(request.role == modules.PlanRole.TARGET for request in ar_plan.requests)
    assert all(request.draft_budget == 1 for request in ar_plan.requests)
    assert ar_plan.target_seq_ids == [seq.seq_id for seq in ar_decode]
    assert ar_plan.plan_two_batch_shadow is True


def run_stspec_two_batch_dryrun_semantics(modules: SimpleNamespace) -> None:
    scheduler = make_scheduler(modules)
    scheduler.enable_stspec_two_batch_execution = True
    scheduler.stspec_two_batch_dryrun = True
    seqs = add_dummy_requests(scheduler, modules)

    scheduler.schedule()
    _, is_prefill, draft_plan = scheduler.schedule_with_plan(
        runner_role="draft",
        execution_mode="parallel_pearl",
        decode_ready_mode=False,
        default_gamma=4,
    )

    scheduled_seq_ids = [seq.seq_id for seq in seqs]
    draft_home_seq_ids = [seq.seq_id for seq in seqs if seq.home_batch_id == draft_plan.draft_home_batch_id]
    assert is_prefill is False
    assert draft_plan.two_batch_execution_enabled is True
    assert draft_plan.two_batch_execution_dryrun is True
    assert draft_plan.two_batch_execution_mode == "dryrun"
    assert draft_plan.actual_target_exec_seq_ids == scheduled_seq_ids
    assert draft_plan.actual_draft_exec_seq_ids == scheduled_seq_ids
    assert draft_plan.dryrun_target_exec_seq_ids == []
    assert draft_plan.dryrun_draft_exec_seq_ids == draft_home_seq_ids


def main() -> None:
    with cpu_safe_stspec_modules() as modules:
        run_stspec_plan_scaffold_diagnostics(modules)
        run_stspec_two_batch_dryrun_semantics(modules)
    print("ST-Spec StepPlan scaffold diagnostics passed")


def test_stspec_plan_scaffold_diagnostics() -> None:
    with cpu_safe_stspec_modules() as modules:
        run_stspec_plan_scaffold_diagnostics(modules)


def test_stspec_two_batch_dryrun_semantics() -> None:
    with cpu_safe_stspec_modules() as modules:
        run_stspec_two_batch_dryrun_semantics(modules)


def test_stspec_diagnostic_does_not_pollute_nano_pearl_imports() -> None:
    before_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == _NANO_PREFIX or name.startswith(f"{_NANO_PREFIX}.")
    }

    main()

    after_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == _NANO_PREFIX or name.startswith(f"{_NANO_PREFIX}.")
    }
    assert after_modules == before_modules

    if importlib.util.find_spec("flash_attn") is None:
        import pytest

        pytest.skip("real nano_pearl package import requires optional flash_attn in this environment")

    from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams

    assert PEARLConfig is not None
    assert PEARLEngine is not None
    assert SamplingParams is not None


if __name__ == "__main__":
    main()
