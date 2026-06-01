import json
import pickle
import subprocess
import sys
import types
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_sequence_symbols():
    seq_path = ROOT / "nano_pearl/pearl_engine/sequence.py"
    src = seq_path.read_text(encoding="utf-8")
    src = src.replace("from __future__ import annotations\n\n", "")
    src = src.replace("from ..layers.sampler import SamplingParams", "")
    module = types.ModuleType("seq_test_module")
    ns = module.__dict__
    exec(
        "from __future__ import annotations\n"
        "class SamplingParams:\n"
        "    def __init__(self, temperature=0.0, max_tokens=16, ignore_eos=False):\n"
        "        self.temperature=temperature\n"
        "        self.max_tokens=max_tokens\n"
        "        self.ignore_eos=ignore_eos\n"
        + src,
        ns,
    )
    sys.modules[module.__name__] = module
    ns["Sequence"].__module__ = module.__name__
    return ns["Sequence"], ns["SequenceStatus"], ns["SamplingParams"]


def _load_scheduler_symbol(sequence_cls, sequence_status_cls):
    sch_path = ROOT / "nano_pearl/pearl_engine/scheduler.py"
    src = sch_path.read_text(encoding="utf-8")
    src = src.replace("from nano_pearl.pearl_config import PEARLConfig", "")
    src = src.replace("from nano_pearl.pearl_engine.sequence import Sequence, SequenceStatus", "")
    src = src.replace("from nano_pearl.pearl_engine.block_manager import BlockManager", "")
    src = src.replace("from nano_pearl.pearl_engine.pearl_model_runner import logger", "")

    module = types.ModuleType("sched_test_module")
    ns = module.__dict__
    ns["Sequence"] = sequence_cls
    ns["SequenceStatus"] = sequence_status_cls
    ns["PEARLConfig"] = type("PEARLConfig", (), {})

    exec(
        "from __future__ import annotations\n"
        "class _L:\n"
        "  def warning(self,*a,**k):\n"
        "   pass\n"
        "logger=_L()\n"
        "class BlockManager:\n"
        "  def __init__(self,*a,**k):\n"
        "   self.blocks=[]\n"
        "   self.hash_to_block_id={}\n"
        "  def can_allocate(self,*a,**k): return True\n"
        "  def allocate(self,*a,**k): pass\n"
        "  def can_append(self,*a,**k): return True\n"
        "  def may_append(self,*a,**k): pass\n"
        "  def deallocate(self,*a,**k): pass\n"
        "  def rollback(self,*a,**k): pass\n"
        + src,
        ns,
    )
    return ns["Scheduler"]


Sequence, SequenceStatus, SamplingParams = _load_sequence_symbols()
Scheduler = _load_scheduler_symbol(Sequence, SequenceStatus)


def _load_dual_batch_symbols(sequence_cls, sequence_status_cls):
    step_path = ROOT / "nano_pearl/pearl_engine/step_plan.py"
    step_module = types.ModuleType("step_plan_test_module")
    sys.modules[step_module.__name__] = step_module
    step_ns = step_module.__dict__
    exec(step_path.read_text(encoding="utf-8"), step_ns)

    dual_path = ROOT / "nano_pearl/pearl_engine/dual_batch.py"
    src = dual_path.read_text(encoding="utf-8")
    src = src.replace("from nano_pearl.pearl_engine.sequence import Sequence, SequenceStatus\n", "")
    src = src.replace("from nano_pearl.pearl_engine.step_plan import RequestBudget, StepPlan\n", "")

    module = types.ModuleType("dual_batch_test_module")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns["Sequence"] = sequence_cls
    ns["SequenceStatus"] = sequence_status_cls
    ns["RequestBudget"] = step_ns["RequestBudget"]
    ns["StepPlan"] = step_ns["StepPlan"]
    exec(src, ns)
    return ns["DualBatchManager"]


DualBatchManager = _load_dual_batch_symbols(Sequence, SequenceStatus)


class _FakeConfig:
    max_num_seqs = 4
    max_num_batched_tokens = 256
    eos = 0
    num_kvcache_blocks = 8
    kvcache_block_size = 256


def test_workload_generator_determinism(tmp_path):
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "benchmark/generate_multislo_workload.py"),
        "--num-requests",
        "8",
        "--seed",
        "7",
        "--out",
        str(out1),
    ]
    subprocess.check_call(cmd)
    cmd[-1] = str(out2)
    subprocess.check_call(cmd)
    assert out1.read_text() == out2.read_text()


def test_workload_fields(tmp_path):
    out = tmp_path / "w.jsonl"
    subprocess.check_call([
        sys.executable,
        str(ROOT / "benchmark/generate_multislo_workload.py"),
        "--num-requests",
        "3",
        "--seed",
        "1",
        "--out",
        str(out),
    ])
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    for r in rows:
        for k in ["request_id", "arrival_offset_sec", "slo_tpot_ms", "slo_class", "max_tokens", "temperature", "ignore_eos", "category"]:
            assert k in r
        assert ("prompt" in r) or ("input_ids" in r)


def test_sequence_admit_ts_pickle_roundtrip():
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    seq.admit_ts = 1234.567
    cloned = pickle.loads(pickle.dumps(seq))
    assert cloned.admit_ts == seq.admit_ts


def test_sequence_cached_kv_materialized_pickle_roundtrip():
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    seq.mark_cached_prefill_metadata(mode="in_memory_kv", cache_key="r0")
    seq.mark_cached_materialized()
    cloned = pickle.loads(pickle.dumps(seq))
    assert cloned.cached_admission_enabled is True
    assert cloned.cached_prefill_mode == "in_memory_kv"
    assert cloned.cached_kv_ready is True
    assert cloned.cached_kv_materialized is True


def test_cached_snapshot_preserves_seq_id_in_source():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")
    assert '"seq_id": int(seq.seq_id)' in src
    assert 'seq.seq_id = int(snapshot.get("seq_id", seq.seq_id))' in src


def test_ar_cached_control_dispatch_uses_string_method_names():
    tree = ast.parse((ROOT / "nano_pearl/pearl_engine/pearl_engine.py").read_text(encoding="utf-8"))
    cached_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cached_decode_ready_generate":
            cached_fn = node
            break
    assert cached_fn is not None
    calls = [
        node
        for node in ast.walk(cached_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"write_draft_shm", "write_target_shm"}
    ]
    assert calls
    for call in calls:
        assert call.args, "shared-memory control call must pass method_name"
        assert isinstance(call.args[0], ast.Constant)
        assert isinstance(call.args[0].value, str)


def test_ar_cached_decode_dispatch_executes_with_string_method_names():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    engine_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PEARLEngine")
    cached_fn = next(
        node
        for node in engine_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "cached_decode_ready_generate"
    )
    fake_cls = ast.ClassDef(
        name="FakeEngine",
        bases=[],
        keywords=[],
        body=[cached_fn],
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                fake_cls,
            ],
            type_ignores=[],
        )
    )
    ns: dict = {}
    exec(compile(module_ast, str(ROOT / "nano_pearl/pearl_engine/pearl_engine.py"), "exec"), ns)

    class _Config:
        execution_mode = "ar"
        ALLOWED_EXECUTION_MODES = {"ar", "serialized_pearl", "parallel_pearl", "dual_batch_pearl"}

    class _Event:
        def wait(self):
            pass

        def clear(self):
            pass

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return ",".join(str(token_id) for token_id in token_ids)

    class _Controller:
        def __init__(self):
            self.calls = []

        def write_draft_shm(self, method_name, *args):
            assert isinstance(method_name, str)
            self.calls.append(("draft", method_name, args))

        def write_target_shm(self, method_name, *args):
            assert isinstance(method_name, str)
            self.calls.append(("target", method_name, args))

        def read_output(self):
            return [], 0.0, [], []

        def read_all_traces(self):
            raise AssertionError("AR cached decode must not read draft traces")

    engine = ns["FakeEngine"]()
    engine.config = _Config()
    engine.controller = _Controller()
    engine.control_event = _Event()
    engine.tokenizer = _Tokenizer()

    output_text, num_tokens, num_acc_tokens, elapsed = engine.cached_decode_ready_generate(
        4,
        execution_mode="ar",
        arrival_field="arrival_offset_sec",
    )

    assert (output_text, num_tokens, num_acc_tokens, elapsed) == ([], [], None, 0.0)
    assert engine.controller.calls == [
        ("draft", "cached_decode_ready_generate", ("ar", 4, "arrival_offset_sec")),
        ("target", "cached_decode_ready_generate", ("ar", 4, "arrival_offset_sec")),
    ]


def test_ar_draft_noop_does_not_write_result_payload_into_command_shm():
    def _is_execution_mode_ar(node):
        return (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "execution_mode"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == "ar"
        )

    def _is_self_is_draft(node):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "is_draft"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def _condition_has_ar_draft(test):
        nodes = list(ast.walk(test))
        return any(_is_execution_mode_ar(node) for node in nodes) and any(
            _is_self_is_draft(node) for node in nodes
        )

    tree = ast.parse((ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8"))
    cached_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "cached_decode_ready_generate"
    )
    ar_noop_if = next(
        node
        for node in ast.walk(cached_fn)
        if isinstance(node, ast.If) and _condition_has_ar_draft(node.test)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_write_payload"
        for stmt in ar_noop_if.body
        for node in ast.walk(stmt)
    )


def test_ar_draft_cached_decode_noop_runtime_does_not_write_payload():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    runner_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerBase"
    )
    cached_fn = next(
        node
        for node in runner_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "cached_decode_ready_generate"
    )
    fake_cls = ast.ClassDef(
        name="FakeRunner",
        bases=[],
        keywords=[],
        body=[cached_fn],
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(ast.Module(body=[fake_cls], type_ignores=[]))

    class _Dist:
        def __init__(self):
            self.barriers = 0

        def barrier(self):
            self.barriers += 1

    dist = _Dist()
    ns = {"dist": dist}
    exec(compile(module_ast, str(ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py"), "exec"), ns)

    class _Config:
        ALLOWED_EXECUTION_MODES = {"ar", "serialized_pearl", "parallel_pearl", "dual_batch_pearl"}

    runner = ns["FakeRunner"]()
    runner.global_config = _Config()
    runner.is_draft = True
    runner._set_execution_mode = lambda mode: setattr(runner, "execution_mode", mode)
    runner._write_payload = lambda *args: (_ for _ in ()).throw(AssertionError("unexpected payload write"))

    assert runner.cached_decode_ready_generate("ar", 4, "arrival_offset_sec") is None
    assert runner.execution_mode == "ar"
    assert runner.active_decode_ready_mode is True
    assert dist.barriers == 2


def test_ar_cached_admission_completion_is_signaled_by_target_master():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    runner_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerBase"
    )
    methods = [
        node
        for node in runner_cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_command_execution_mode", "_should_signal_control_event"}
    ]
    fake_cls = ast.ClassDef(
        name="FakeRunner",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                fake_cls,
            ],
            type_ignores=[],
        )
    )
    ns: dict = {}
    exec(compile(module_ast, str(ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py"), "exec"), ns)

    class _TP:
        def __init__(self, local_rank):
            self.local_rank = local_rank

    def make_runner(rank, is_draft, local_rank):
        runner = ns["FakeRunner"]()
        runner.rank = rank
        runner.is_draft = is_draft
        runner.tp_params = _TP(local_rank)
        return runner

    draft_master = make_runner(rank=0, is_draft=True, local_rank=0)
    target_master = make_runner(rank=1, is_draft=False, local_rank=0)
    target_peer = make_runner(rank=2, is_draft=False, local_rank=1)

    ar_commands = [
        ("cache_build_prepare", [[], 8, "ar"]),
        ("add_cached_request", [object(), "ar"]),
        ("cached_decode_ready_generate", ["ar", 4, "arrival_offset_sec"]),
    ]
    for method_name, args in ar_commands:
        assert draft_master._should_signal_control_event(method_name, args) is False
        assert target_master._should_signal_control_event(method_name, args) is True
        assert target_peer._should_signal_control_event(method_name, args) is False

    assert draft_master._should_signal_control_event(
        "cache_build_prepare", [[], 8, "parallel_pearl"]
    ) is True
    assert target_master._should_signal_control_event(
        "cache_build_prepare", [[], 8, "parallel_pearl"]
    ) is False


def test_runner_malformed_method_guard_is_clear():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")
    assert "runner control method_name must be str" in src
    assert "method_type=" in src
    assert "method_repr=" in src
    assert "rank=" in src
    assert "group=" in src


def test_dual_batch_new_cached_seq_goes_to_non_pending_batch():
    manager = DualBatchManager(gamma=4)
    seq4 = Sequence([1, 2], SamplingParams(), request_id="r4")
    seq5 = Sequence([1, 3], SamplingParams(), request_id="r5")
    seq6 = Sequence([1, 4], SamplingParams(), request_id="r6")
    seq4.seq_id = 4
    seq5.seq_id = 5
    seq6.seq_id = 6
    manager.assign(seq4, 0)
    manager.assign(seq5, 1)

    manager.update_running([seq4, seq5, seq6], pending_batch_ids=[0])
    assert seq6.home_batch_id == 1

    plan = manager.build_step_plan(
        plan_id=1,
        iteration_id=1,
        execution_mode="dual_batch_pearl",
        decode_ready_mode=True,
        pending_proposal_seq_ids=[4],
        pending_batch_ids=[0],
        running_seqs=[seq4, seq5, seq6],
    )
    assert plan.target_normal_verify_seq_ids == [4]
    assert 6 not in plan.target_normal_verify_seq_ids
    assert 6 in plan.draft_home_set


def test_cached_seq_id_consistency_mismatch_helper():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "cached_seq_id_consistency_mismatches"
    )
    module = types.ModuleType("cached_consistency_helper_test_module")
    helper_module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(helper_module, str(ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py"), "exec"), module.__dict__)
    mismatches = module.cached_seq_id_consistency_mismatches(
        [
            {"rank": 0, "group": "draft", "runner_role": "draft", "request_seq_id": {"r0": 7}},
            {"rank": 1, "group": "target", "runner_role": "target", "request_seq_id": {"r0": 8}},
        ]
    )
    assert mismatches
    assert mismatches[0][0] == "r0"


def test_add_cached_sets_pending_cached_status():
    scheduler = Scheduler(_FakeConfig())
    seq = Sequence([1, 2], SamplingParams())
    scheduler.add_cached(seq)
    assert seq.status == SequenceStatus.PENDING_CACHED


def test_pending_cached_queue_is_distinct_from_waiting():
    scheduler = Scheduler(_FakeConfig())
    a = Sequence([1], SamplingParams())
    b = Sequence([2], SamplingParams())
    scheduler.add(a)
    scheduler.add_cached(b)
    assert len(scheduler.waiting) == 1
    assert len(scheduler.pending_cached) == 1
