import json
import pickle
import subprocess
import sys
import types
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
