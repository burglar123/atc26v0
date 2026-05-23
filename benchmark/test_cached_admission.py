import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_workload_generator_determinism(tmp_path):
    out1 = tmp_path / 'a.jsonl'
    out2 = tmp_path / 'b.jsonl'
    cmd=[sys.executable, str(ROOT/'benchmark/generate_multislo_workload.py'), '--num-requests','8','--seed','7','--out',str(out1)]
    subprocess.check_call(cmd)
    cmd[-1]=str(out2)
    subprocess.check_call(cmd)
    assert out1.read_text()==out2.read_text()


def test_workload_fields(tmp_path):
    out = tmp_path / 'w.jsonl'
    subprocess.check_call([sys.executable, str(ROOT/'benchmark/generate_multislo_workload.py'), '--num-requests','3','--seed','1','--out',str(out)])
    rows=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    for r in rows:
        for k in ['request_id','arrival_offset_sec','slo_tpot_ms','slo_class','max_tokens','temperature','ignore_eos','category']:
            assert k in r
        assert ('prompt' in r) or ('input_ids' in r)
