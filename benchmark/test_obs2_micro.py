"""Tests for Observation 2 micro-profiling instrumentation and merge logic."""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_field_presence():
    runner = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    engine = (ROOT / "nano_pearl/pearl_engine/pearl_engine.py").read_text()
    assert '"trace_type"' in runner
    assert '"drafted_tokens_total"' in runner
    assert '"rejected_tokens_by_request"' in runner
    assert '"slo_class_by_request"' in runner
    assert '"slo_tpot_ms_by_request"' in runner
    assert '"decode_iteration_group"' in runner
    assert '_decode_iteration_group += 1' in runner
    assert '_decode_iteration_group = 0' in runner
    assert '_merge_decode_iterations' in engine
    assert 'draft_verify_overlap_ms' in engine
    assert '"trace_type"' in engine


def test_merge_logic_comprehensive():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_engine.py").read_text()
    ms = src.index('def _merge_decode_iterations')
    rest = src[ms:]
    lines = rest.split('\n')
    fn_lines = []
    for line in lines:
        fn_lines.append(line)
        if fn_lines and line.startswith('    def ') and len(fn_lines) > 1:
            fn_lines.pop()
            break
    fn_src = '\n'.join(fn_lines).replace('    @staticmethod\n', '')
    env = {}
    exec(fn_src, env)
    merge = env['_merge_decode_iterations']

    # --- Serialized merge ---
    drafts = [{
        'trace_type': 'decode_iteration', 'execution_mode': 'serialized_pearl',
        'decode_iteration_group': 0, 'runner_role': 'serialized_draft',
        'scheduled_seq_ids': [0, 1], 'request_ids': ['a', 'b'], 'num_seqs_in_batch': 2,
        'is_prefill': False, 'draft_start_ts': 10.0, 'draft_end_ts': 10.01,
        'drafted_tokens_total': 2, 'slo_class_by_request': {'a': 'tight', 'b': 'normal'},
        'slo_tpot_ms_by_request': {'a': 50, 'b': 40},
    } for _ in range(4)]
    verify = {
        'trace_type': 'decode_iteration', 'execution_mode': 'serialized_pearl',
        'decode_iteration_group': 0, 'runner_role': 'serialized_verify',
        'scheduled_seq_ids': [0, 1], 'request_ids': ['a', 'b'], 'num_seqs_in_batch': 2,
        'is_prefill': False, 'verify_start_ts': 10.02, 'verify_end_ts': 10.03,
        'verify_time_ms': 10.0,
        'total_accepted_tokens': 6, 'accepted_tokens_per_seq': {'0': 3, '1': 3},
        'rejected_tokens_by_request': {'0': 1, '1': 1},
        'per_seq_invalidated_predraft_len': {},
        'slo_class_by_request': {'a': 'tight', 'b': 'normal'},
        'slo_tpot_ms_by_request': {'a': 50, 'b': 40},
    }
    r = merge(drafts + [verify])
    assert len(r) == 1
    m = r[0]
    assert m['trace_type'] == 'decode_iteration'
    assert m['execution_mode'] == 'serialized_pearl'
    assert m['active_batch_size'] == 2
    assert abs(m['draft_time_ms'] - 10.0) < 0.01
    assert abs(m['verify_time_ms'] - 10.0) < 0.01
    assert m['drafted_tokens_total'] == 8
    assert m['accepted_tokens_total'] == 6
    assert m['accepted_tokens_by_request'] == {'a': 3, 'b': 3}
    assert m['rejected_tokens_by_request'] == {'a': 1, 'b': 1}
    assert m['slo_class_by_request'] == {'a': 'tight', 'b': 'normal'}
    assert m['slo_tpot_ms_by_request'] == {'a': 50, 'b': 40}
    assert m['draft_verify_overlap_ms'] == 0.0

    # --- Parallel overlap ---
    d = [{
        'trace_type': 'decode_iteration', 'execution_mode': 'parallel_pearl',
        'decode_iteration_group': 0, 'runner_role': 'draft',
        'scheduled_seq_ids': [0], 'request_ids': ['x'], 'num_seqs_in_batch': 1,
        'is_prefill': False, 'draft_start_ts': 100.0, 'draft_end_ts': 100.015,
        'drafted_tokens_total': 4, 'slo_class_by_request': {'x': 'loose'},
        'slo_tpot_ms_by_request': {'x': 150},
    }]
    v = {
        'trace_type': 'decode_iteration', 'execution_mode': 'parallel_pearl',
        'decode_iteration_group': 0, 'runner_role': 'verify',
        'scheduled_seq_ids': [0], 'request_ids': ['x'], 'num_seqs_in_batch': 1,
        'is_prefill': False, 'verify_start_ts': 100.008, 'verify_end_ts': 100.020,
        'verify_time_ms': 12.0,
        'total_accepted_tokens': 3, 'accepted_tokens_per_seq': {'0': 3},
        'rejected_tokens_by_request': {}, 'per_seq_invalidated_predraft_len': {},
        'slo_class_by_request': {'x': 'loose'}, 'slo_tpot_ms_by_request': {'x': 150},
    }
    r2 = merge(d + [v])
    assert len(r2) == 1
    m2 = r2[0]
    assert abs(m2['draft_verify_overlap_ms'] - 7.0) < 0.001
    assert abs(m2['iter_time_ms'] - 20.0) < 0.001

    # --- Prefill skipped ---
    assert len(merge([{'trace_type': 'prefill', 'is_prefill': True,
                       'decode_iteration_group': -1}])) == 0


def test_profile_obs2_micro():
    with tempfile.TemporaryDirectory() as d:
        trace = {'traces': [
            {'trace_type': 'decode_iteration', 'execution_mode': 'serialized_pearl',
             'active_batch_size': 3, 'draft_time_ms': 5, 'verify_time_ms': 10,
             'iter_time_ms': 15, 'drafted_tokens_total': 12, 'accepted_tokens_total': 8,
             'accepted_tokens_by_request': {'a': 4, 'b': 3, 'c': 1}},
            {'trace_type': 'decode_iteration', 'execution_mode': 'serialized_pearl',
             'active_batch_size': 3, 'draft_time_ms': 6, 'verify_time_ms': 11,
             'iter_time_ms': 17, 'drafted_tokens_total': 12, 'accepted_tokens_total': 7,
             'accepted_tokens_by_request': {'a': 3, 'b': 3, 'c': 1}},
            {'trace_type': 'prefill', 'is_prefill': True},
        ]}
        tp = os.path.join(d, 'trace.json')
        with open(tp, 'w') as f:
            json.dump(trace, f)
        out = os.path.join(d, 'out.csv')
        r = subprocess.run([sys.executable, str(ROOT / 'benchmark/profile_obs2_micro.py'),
                            '--engine-trace', tp, '--out', out],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row['execution_mode'] == 'serialized_pearl'
        assert int(row['num_iterations']) == 2
        assert float(row['avg_active_batch_size']) == 3.0
        assert float(row['mean_draft_time_ms']) == 5.5
        assert abs(float(row['mean_accepted_tokens_per_request_per_iter']) - 2.5) < 0.01


def test_ar_step_counter():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    step_fn = src.split('def step(self):')[1].split('\n    def ')[0]
    assert '_decode_iteration_group += 1' in step_fn


def test_no_semantic_changes():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    assert 'def verify(self' in src
    assert '.verify(seqs)' in src
    assert 'dist.barrier()' in src
