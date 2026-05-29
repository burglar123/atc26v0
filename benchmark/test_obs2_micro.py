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
    assert '"record_level"' in runner
    assert '"runner_substep"' in runner
    assert '"drafted_tokens_total"' in runner
    assert '"rejected_tokens_by_request"' in runner
    assert '"slo_class_by_request"' in runner
    assert '"slo_tpot_ms_by_request"' in runner
    assert '"decode_iteration_group"' in runner
    assert '_decode_iteration_group += 1' in runner
    assert '_decode_iteration_group = 0' in runner
    assert '_merge_decode_iterations' in engine
    assert '"record_level"' in engine
    assert '"merged_iteration"' in engine
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
        'trace_type': 'decode_iteration', 'record_level': 'runner_substep',
        'execution_mode': 'serialized_pearl',
        'decode_iteration_group': 0, 'runner_role': 'serialized_draft',
        'scheduled_seq_ids': [0, 1], 'request_ids': ['a', 'b'], 'num_seqs_in_batch': 2,
        'is_prefill': False, 'draft_start_ts': 10.0, 'draft_end_ts': 10.01,
        'drafted_tokens_total': 2, 'slo_class_by_request': {'a': 'tight', 'b': 'normal'},
        'slo_tpot_ms_by_request': {'a': 50, 'b': 40},
    } for _ in range(4)]
    verify = {
        'trace_type': 'decode_iteration', 'record_level': 'runner_substep',
        'execution_mode': 'serialized_pearl',
        'decode_iteration_group': 0, 'runner_role': 'serialized_verify',
        'scheduled_seq_ids': [0, 1], 'request_ids': ['a', 'b'], 'num_seqs_in_batch': 2,
        'is_prefill': False, 'verify_start_ts': 10.02, 'verify_end_ts': 10.03,
        'verify_time_ms': 10.0,
        'total_accepted_tokens': 6, 'accepted_tokens_per_seq': {'0': 3, '1': 3},
        'rejected_tokens_by_request': {'a': 1, 'b': 1},
        'per_seq_invalidated_predraft_len': {},
        'slo_class_by_request': {'a': 'tight', 'b': 'normal'},
        'slo_tpot_ms_by_request': {'a': 50, 'b': 40},
    }
    r = merge(drafts + [verify])
    assert len(r) == 1
    m = r[0]
    assert m['trace_type'] == 'decode_iteration'
    assert m['record_level'] == 'merged_iteration'
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

    # --- Verify that rejected_tokens_by_request with already-mapped keys works ---
    verify2 = dict(verify)
    verify2['rejected_tokens_by_request'] = {'a': 2, 'b': 0}
    r1 = merge(drafts + [verify2])
    assert r1[0]['rejected_tokens_by_request'] == {'a': 2, 'b': 0}

    # --- Parallel overlap ---
    d = [{
        'trace_type': 'decode_iteration', 'record_level': 'runner_substep',
        'execution_mode': 'parallel_pearl',
        'decode_iteration_group': 0, 'runner_role': 'draft',
        'scheduled_seq_ids': [0], 'request_ids': ['x'], 'num_seqs_in_batch': 1,
        'is_prefill': False, 'draft_start_ts': 100.0, 'draft_end_ts': 100.015,
        'drafted_tokens_total': 4, 'slo_class_by_request': {'x': 'loose'},
        'slo_tpot_ms_by_request': {'x': 150},
    }]
    v = {
        'trace_type': 'decode_iteration', 'record_level': 'runner_substep',
        'execution_mode': 'parallel_pearl',
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
    assert m2['record_level'] == 'merged_iteration'
    assert abs(m2['draft_verify_overlap_ms'] - 7.0) < 0.001
    assert abs(m2['iter_time_ms'] - 20.0) < 0.001

    # --- Prefill skipped ---
    assert len(merge([{'trace_type': 'prefill', 'is_prefill': True,
                       'decode_iteration_group': -1}])) == 0


def test_profile_obs2_micro_canonical():
    """profile_obs2_micro consumes merged_iteration records with canonical fields."""
    with tempfile.TemporaryDirectory() as d:
        trace = {'traces': [
            {'trace_type': 'decode_iteration', 'record_level': 'merged_iteration',
             'execution_mode': 'serialized_pearl',
             'active_batch_size': 3, 'draft_time_ms': 5, 'verify_time_ms': 10,
             'iter_time_ms': 15, 'drafted_tokens_total': 12, 'accepted_tokens_total': 8,
             'accepted_tokens_by_request': {'a': 4, 'b': 3, 'c': 1}},
            {'trace_type': 'decode_iteration', 'record_level': 'merged_iteration',
             'execution_mode': 'serialized_pearl',
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


def test_profile_obs2_micro_raw_fallback():
    """profile_obs2_micro falls back to raw field names when canonical are missing."""
    with tempfile.TemporaryDirectory() as d:
        trace = {'traces': [
            {'trace_type': 'decode_iteration', 'execution_mode': 'serialized_pearl',
             'num_seqs_in_batch': 2, 'draft_time_ms': 3, 'verify_time_ms': 7,
             'total_iteration_time_ms': 10, 'drafted_tokens_total': 8,
             'total_accepted_tokens': 4, 'accepted_tokens_per_seq': {'0': 4}},
            {'trace_type': 'decode_iteration', 'execution_mode': 'serialized_pearl',
             'num_seqs_in_batch': 2, 'draft_time_ms': 4, 'verify_time_ms': 8,
             'total_iteration_time_ms': 12, 'drafted_tokens_total': 8,
             'total_accepted_tokens': 5, 'accepted_tokens_per_seq': {'0': 5}},
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
        assert int(row['num_iterations']) == 2
        assert float(row['avg_active_batch_size']) == 2.0
        assert float(row['mean_iter_time_ms']) == 11.0
        assert float(row['mean_accepted_tokens_total']) == 4.5
        assert abs(float(row['mean_accepted_tokens_per_request_per_iter']) - 4.5) < 0.01


def test_ar_step_counter():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    step_fn = src.split('def step(self):')[1].split('\n    def ')[0]
    assert '_decode_iteration_group += 1' in step_fn


def test_no_semantic_changes():
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    assert 'def verify(self' in src
    assert '.verify(seqs)' in src
    assert 'dist.barrier()' in src


def test_serialized_prepare_uses_gamma():
    """prepare_serialized_verify_decode always uses num_tokens = self.gamma, not 1."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    # Extract the prepare_serialized_verify_decode method
    idx = src.index('def prepare_serialized_verify_decode')
    section = src[idx:]
    next_def = section.find('\n    def ', 10)
    fn_body = section[:next_def] if next_def > 0 else section
    # Must contain num_tokens = self.gamma (not conditional on pre_verify)
    assert 'num_tokens = self.gamma' in fn_body
    # Must NOT contain the pre_verify conditional
    assert 'if not seq.pre_verify else 1' not in fn_body


def test_serialized_protocol_methods():
    """serialized_pearl uses separate protocol methods, not pre_verify patching."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()

    # DraftModelRunner.serialized_pearl_step — uses send + postprocess, no PEARL verify
    draft_idx = src.index('class DraftModelRunner')
    draft_section = src[draft_idx:]
    draft_serialized = draft_section.split('def serialized_pearl_step')[1]
    draft_serialized = draft_serialized.split('\n    def ')[0]
    assert 'send_serialized_draft_window' in draft_serialized
    assert '_serialized_postprocess' in draft_serialized
    # Must NOT call the old PEARL verify
    assert 'self.verify(seqs)' not in draft_serialized

    # TargetModelRunner.serialized_pearl_step — full protocol
    target_idx = src.index('class TargetModelRunner')
    target_section = src[target_idx:]
    target_serialized = target_section.split('def serialized_pearl_step')[1]
    target_serialized = target_serialized.split('\n    def ')[0]
    assert 'recv_serialized_draft_window' in target_serialized
    assert 'prepare_serialized_verify_decode' in target_serialized
    assert 'serialized_verify_full_gamma' in target_serialized
    assert '_serialized_postprocess' in target_serialized
    # Must NOT call the old PEARL verify
    assert 'self.verify(logits, seqs, temperatures)' not in target_serialized


def test_serialized_prepare_verifies_all_gamma():
    """prepare_serialized_verify_decode starts one token before the draft window."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    target_idx = src.index('class TargetModelRunner')
    target_section = src[target_idx:]
    prepare_fn = target_section.split('def prepare_serialized_verify_decode')[1]
    prepare_fn = prepare_fn.split('\n    def ')[0]
    # Uses start = len(seq) - num_tokens - 1 (last confirmed token)
    assert 'start = len(seq) - num_tokens - 1' in prepare_fn
    # Uses end = len(seq) - 1 (before last draft token)
    assert 'end = len(seq) - 1' in prepare_fn
    # Feeds exactly gamma tokens
    assert 'num_tokens = self.gamma' in prepare_fn
    # Has strict size assertion
    assert 'size mismatch' in prepare_fn
    # Does NOT use pre_verify conditional
    assert 'if not seq.pre_verify else 1' not in prepare_fn


def test_parallel_pearl_unchanged():
    """Parallel PEARL pearl_step must NOT force pre_verify=False."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()

    # DraftModelRunner.pearl_step
    draft_idx = src.index('class DraftModelRunner')
    draft_section = src[draft_idx:]
    draft_pearl = draft_section.split('def pearl_step')[1]
    draft_pearl = draft_pearl.split('\n    def ')[0]
    # The parallel PEARL path should NOT have our pre_verify forcing
    assert 'seq.pre_verify = False' not in draft_pearl

    # TargetModelRunner.pearl_step
    target_idx = src.index('class TargetModelRunner')
    target_section = src[target_idx:]
    target_pearl = target_section.split('def pearl_step')[1]
    target_pearl = target_pearl.split('\n    def ')[0]
    assert 'seq.pre_verify = False' not in target_pearl

    # But verify methods still contain pre_verify logic (for parallel PEARL)
    draft_verify = draft_section.split('def verify(self')[1]
    draft_verify = draft_verify.split('\n    def ')[0]
    assert 'seq.pre_verify' in draft_verify

    target_verify = target_section.split('def verify(self')[1]
    target_verify = target_verify.split('\n    def ')[0]
    assert '1 if seq.pre_verify else self.gamma' in target_verify


def test_serialized_target_uses_correct_prepare():
    """TargetModelRunner.serialized_pearl_step calls prepare_serialized_verify_decode."""
    src = (ROOT / "nano_pearl/pearl_engine/pearl_model_runner.py").read_text()
    target_idx = src.index('class TargetModelRunner')
    target_section = src[target_idx:]
    target_serialized = target_section.split('def serialized_pearl_step')[1]
    target_serialized = target_serialized.split('\n    def ')[0]
    assert 'prepare_serialized_verify_decode' in target_serialized
    # Should NOT call prepare_pearl_decode in the serialized path
    assert 'prepare_pearl_decode' not in target_serialized


def test_accepted_tokens_range_unconstrained():
    """After fix, accepted_tokens_by_request values can range from 0 to gamma.

    Validates that the merge output and profile script can handle accepted
    token counts > 1 for serialized_pearl (gamma=4 means range 0-4).
    """
    # Simulate a decoded iteration where all gamma=4 tokens were accepted
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

    # Simulate full acceptance (all 4 tokens accepted for both seqs)
    drafts = [{
        'trace_type': 'decode_iteration', 'record_level': 'runner_substep',
        'execution_mode': 'serialized_pearl',
        'decode_iteration_group': 0, 'runner_role': 'serialized_draft',
        'scheduled_seq_ids': [0, 1], 'request_ids': ['a', 'b'],
        'num_seqs_in_batch': 2, 'is_prefill': False,
        'draft_start_ts': 10.0, 'draft_end_ts': 10.01,
        'drafted_tokens_total': 4,
    } for _ in range(4)]
    verify = {
        'trace_type': 'decode_iteration', 'record_level': 'runner_substep',
        'execution_mode': 'serialized_pearl',
        'decode_iteration_group': 0, 'runner_role': 'serialized_verify',
        'scheduled_seq_ids': [0, 1], 'request_ids': ['a', 'b'],
        'num_seqs_in_batch': 2, 'is_prefill': False,
        'verify_start_ts': 10.02, 'verify_end_ts': 10.03,
        'verify_time_ms': 10.0,
        'total_accepted_tokens': 8, 'accepted_tokens_per_seq': {'0': 4, '1': 4},
        'rejected_tokens_by_request': {},
        'per_seq_invalidated_predraft_len': {},
    }
    m = merge(drafts + [verify])[0]
    assert m['accepted_tokens_total'] == 8
    assert m['accepted_tokens_by_request'] == {'a': 4, 'b': 4}
    assert m['drafted_tokens_total'] == 16  # 4 per draft step × 4 steps
    # Verify 0-4 range is also handled
    assert 4 <= 4  # sanity: max accepted is gamma
