#!/usr/bin/env python3
"""Self-contained test for profile_obs1_goodput.py using synthetic data.

Generates synthetic result JSON + trace JSONL files, runs the profiler, and
asserts correctness of all computed metrics.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def make_synthetic_data(
    tmpdir: str,
    label: str,
    obs: str,
    x_value: float,
    requests: List[Dict[str, Any]],
    engine_elapsed_s: float = 10.0,
    execution_mode: str = "parallel_pearl",
    decode_ready: bool = False,
) -> tuple:
    """Create synthetic result JSON and trace JSONL files.

    Returns (result_json_path, trace_jsonl_path).
    """
    result_path = os.path.join(tmpdir, f"test_{label}.json")
    trace_path = os.path.join(tmpdir, f"test_{label}.jsonl")

    # --- result JSON ---
    total_tokens = sum(r["tokens"] for r in requests)
    attained_tokens = sum(r["tokens"] for r in requests if r.get("slo_attained", False))
    tpot_vals = [r["observed_tpot_ms"] for r in requests if r["observed_tpot_ms"] is not None]
    mean_tpot = sum(tpot_vals) / len(tpot_vals) if tpot_vals else None

    result = {
        "args": {
            "execution_mode": execution_mode,
            "decode_ready": decode_ready,
        },
        "metrics": {
            "overall": {
                "num_requests": len(requests),
                "num_attained": sum(1 for r in requests if r.get("slo_attained", False)),
                "slo_attainment": sum(1 for r in requests if r.get("slo_attained", False)) / len(requests),
                "goodput_tokens_per_s": attained_tokens / engine_elapsed_s,
                "mean_tpot_ms": mean_tpot,
                "total_output_tokens": total_tokens,
                "attained_output_tokens": attained_tokens,
            },
            "engine_elapsed_s": engine_elapsed_s,
            "goodput_denominator_mode": "engine_elapsed",
            "goodput_denominator_s": engine_elapsed_s,
            "execution_mode": execution_mode,
            "decode_ready_mode": decode_ready,
        },
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # --- trace JSONL ---
    with open(trace_path, "w") as f:
        for i, req in enumerate(requests):
            record = {
                "request_id": f"test_{label}_{i}",
                "category": req.get("category"),
                "slo_class": req.get("slo_class"),
                "slo_tpot_ms": req.get("slo_tpot_ms"),
                "per_request_gamma": req.get("gamma", 4),
                "execution_mode": execution_mode,
                "decode_ready_mode": decode_ready,
                "arrival_offset_sec": i * 0.5,
                "arrival_ts": i * 0.5,
                "admit_ts": i * 0.5 + 0.01 if req.get("has_admit", False) else None,
                "num_output_tokens": req["tokens"],
                "num_decode_output_tokens": req.get("num_decode_output_tokens", req["tokens"]),
                "decode_elapsed_ms": req["observed_tpot_ms"] * req["tokens"] if req["observed_tpot_ms"] is not None else None,
                "observed_tpot_ms": req.get("observed_tpot_ms"),
                "finish_ts": i * 0.5 + (req["observed_tpot_ms"] * req["tokens"]) / 1000.0 if req["observed_tpot_ms"] is not None else i * 0.5 + 1.0,
                "decode_start_ts": i * 0.5 + 0.001,
            }
            f.write(json.dumps(record) + "\n")

    return result_path, trace_path


def run_profiler(runs: List[tuple], out_csv: str, out_md: str) -> subprocess.CompletedProcess:
    """Run profile_obs1_goodput.py as a subprocess."""
    script = os.path.join(os.path.dirname(__file__), "profile_obs1_goodput.py")
    cmd = [sys.executable, script]
    for obs, label, xv, rj, tj in runs:
        cmd.extend(["--run", obs, label, str(xv), rj, tj])
    cmd.extend(["--out-csv", out_csv, "--out-md", out_md])
    return subprocess.run(cmd, capture_output=True, text=True)


def approx(a: float, b: float, rel: float = 1e-6) -> bool:
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        return abs(a - b) < 1e-9
    return abs(a - b) / max(abs(a), abs(b)) < rel


def test_core_metrics() -> None:
    """Verify all core metrics with carefully constructed synthetic data."""
    requests = [
        # Tight requests (slo=50ms)
        {"category": "coding", "slo_class": "tight", "tokens": 100,
         "observed_tpot_ms": 45, "slo_tpot_ms": 50, "slo_attained": True,
         "has_admit": True},
        {"category": "coding", "slo_class": "tight", "tokens": 80,
         "observed_tpot_ms": 55, "slo_tpot_ms": 50, "slo_attained": False,
         "has_admit": False},
        {"category": "coding", "slo_class": "tight", "tokens": 120,
         "observed_tpot_ms": 48, "slo_tpot_ms": 50, "slo_attained": True,
         "has_admit": True},
        # Normal requests (slo=40ms)
        {"category": "chat", "slo_class": "normal", "tokens": 200,
         "observed_tpot_ms": 35, "slo_tpot_ms": 40, "slo_attained": True,
         "has_admit": True},
        {"category": "chat", "slo_class": "normal", "tokens": 150,
         "observed_tpot_ms": 42, "slo_tpot_ms": 40, "slo_attained": False,
         "has_admit": False},
        {"category": "chat", "slo_class": "normal", "tokens": 180,
         "observed_tpot_ms": 38, "slo_tpot_ms": 40, "slo_attained": True,
         "has_admit": True},
        {"category": "chat", "slo_class": "normal", "tokens": 170,
         "observed_tpot_ms": 36, "slo_tpot_ms": 40, "slo_attained": True,
         "has_admit": True},
        # Loose requests (slo=150ms)
        {"category": "summarization", "slo_class": "loose", "tokens": 300,
         "observed_tpot_ms": 120, "slo_tpot_ms": 150, "slo_attained": True,
         "has_admit": True},
        {"category": "summarization", "slo_class": "loose", "tokens": 250,
         "observed_tpot_ms": 160, "slo_tpot_ms": 150, "slo_attained": False,
         "has_admit": False},
        {"category": "summarization", "slo_class": "loose", "tokens": 280,
         "observed_tpot_ms": 140, "slo_tpot_ms": 150, "slo_attained": True,
         "has_admit": True},
    ]

    engine_elapsed_s = 10.0

    with tempfile.TemporaryDirectory() as tmpdir:
        rj, tj = make_synthetic_data(
            tmpdir, "core", "1A", 4.0, requests,
            engine_elapsed_s=engine_elapsed_s,
        )
        out_csv = os.path.join(tmpdir, "out.csv")
        out_md = os.path.join(tmpdir, "out.md")

        result = run_profiler(
            [(("1A", "core", 4.0, rj, tj))],
            out_csv, out_md,
        )
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            raise AssertionError(f"Profiler exited with code {result.returncode}")

        # Read the CSV to verify
        with open(out_csv, "r") as f:
            import csv
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        r = rows[0]

        # --- Basic metadata ---
        assert r["obs"] == "1A"
        assert r["label"] == "core"
        assert float(r["x_value"]) == 4.0
        assert int(r["num_requests"]) == 10
        assert float(r["engine_elapsed_s"]) == engine_elapsed_s

        # --- Token counts ---
        # Total raw tokens: 100+80+120+200+150+180+170+300+250+280 = 1830
        assert int(r["raw_tokens"]) == 1830, f"raw_tokens: {r['raw_tokens']}"
        # Good tokens (SLO satisfied): 100+120+200+180+170+300+280 = 1350
        assert int(r["good_tokens"]) == 1350, f"good_tokens: {r['good_tokens']}"
        # Bad tokens: 80+150+250 = 480
        assert int(r["bad_tokens"]) == 480, f"bad_tokens: {r['bad_tokens']}"

        # --- Throughput ---
        raw_tp = float(r["raw_throughput_tokens_per_s"])
        assert approx(raw_tp, 183.0), f"raw_throughput: {raw_tp}"
        goodput = float(r["slo_goodput_tokens_per_s"])
        assert approx(goodput, 135.0), f"slo_goodput: {goodput}"
        gap = float(r["goodput_gap_tokens_per_s"])
        assert approx(gap, 48.0), f"goodput_gap: {gap}"
        ratio = float(r["goodput_ratio"])
        assert approx(ratio, 135.0 / 183.0), f"goodput_ratio: {ratio}"

        # --- SLO attainment ---
        assert approx(float(r["overall_slo_attainment"]), 0.7), \
            f"overall_slo_attainment: {r['overall_slo_attainment']}"
        assert approx(float(r["tight_slo_attainment"]), 2.0 / 3.0), \
            f"tight_slo_attainment: {r['tight_slo_attainment']}"
        assert approx(float(r["normal_slo_attainment"]), 3.0 / 4.0), \
            f"normal_slo_attainment: {r['normal_slo_attainment']}"
        assert approx(float(r["loose_slo_attainment"]), 2.0 / 3.0), \
            f"loose_slo_attainment: {r['loose_slo_attainment']}"

        # --- Per-class tokens ---
        assert int(r["tight_raw_tokens"]) == 300, f"tight_raw_tokens: {r['tight_raw_tokens']}"
        assert int(r["tight_good_tokens"]) == 220, f"tight_good_tokens: {r['tight_good_tokens']}"
        assert int(r["normal_raw_tokens"]) == 700, f"normal_raw_tokens: {r['normal_raw_tokens']}"
        assert int(r["normal_good_tokens"]) == 550, f"normal_good_tokens: {r['normal_good_tokens']}"
        assert int(r["loose_raw_tokens"]) == 830, f"loose_raw_tokens: {r['loose_raw_tokens']}"
        assert int(r["loose_good_tokens"]) == 580, f"loose_good_tokens: {r['loose_good_tokens']}"

        # Per-class goodput
        assert approx(float(r["tight_slo_goodput_tokens_per_s"]), 22.0)
        assert approx(float(r["normal_slo_goodput_tokens_per_s"]), 55.0)
        assert approx(float(r["loose_slo_goodput_tokens_per_s"]), 58.0)

        # --- Normalized TPOT ---
        # Values: 45/50=0.9, 55/50=1.1, 48/50=0.96,
        #          35/40=0.875, 42/40=1.05, 38/40=0.95, 36/40=0.9,
        #          120/150=0.8, 160/150≈1.0667, 140/150≈0.9333
        # p50 should be around 0.94
        p50_val = float(r["overall_normalized_tpot_p50"])
        assert 0.93 < p50_val < 0.95, f"p50: {p50_val}"

        # p90 should be around 1.07
        p90_val = float(r["overall_normalized_tpot_p90"])
        assert 1.06 < p90_val < 1.08, f"p90: {p90_val}"

        # --- Admit delay ---
        # 7 requests have has_admit=True
        p50_admit = float(r["admit_delay_ms_p50"])
        assert p50_admit == 10.0, f"admit_delay_ms_p50: {p50_admit} (expected 10.0ms)"

        # --- Skipped rows ---
        assert int(r["skipped_missing_tpot"]) == 0
        assert int(r["skipped_missing_slo"]) == 0
        assert int(r["overall_valid_for_slo"]) == 10

        print("[PASS] test_core_metrics")


def test_missing_fields() -> None:
    """Test handling of missing observed_tpot_ms and slo_tpot_ms."""
    requests = [
        {"category": "coding", "slo_class": "tight", "tokens": 100,
         "observed_tpot_ms": 45, "slo_tpot_ms": 50, "slo_attained": True},
        {"category": "chat", "slo_class": "normal", "tokens": 80,
         "observed_tpot_ms": None, "slo_tpot_ms": 40, "slo_attained": False},
        {"category": "summarization", "slo_class": "loose", "tokens": 120,
         "observed_tpot_ms": 100, "slo_tpot_ms": None, "slo_attained": False},
        {"category": "chat", "slo_class": "normal", "tokens": 90,
         "observed_tpot_ms": None, "slo_tpot_ms": None, "slo_attained": False},
    ]
    engine_elapsed_s = 5.0

    with tempfile.TemporaryDirectory() as tmpdir:
        rj, tj = make_synthetic_data(
            tmpdir, "missing", "1B", 8.0, requests,
            engine_elapsed_s=engine_elapsed_s,
        )
        out_csv = os.path.join(tmpdir, "out.csv")

        result = run_profiler(
            [(("1B", "missing", 8.0, rj, tj))],
            out_csv, os.path.join(tmpdir, "out.md"),
        )
        # Should still succeed (only warnings, not errors)
        assert result.returncode == 0, f"Profiler exited with {result.returncode}: {result.stderr}"

        with open(out_csv, "r") as f:
            import csv
            reader = csv.DictReader(f)
            rows = list(reader)
        r = rows[0]

        # Only 1 valid SLO request
        assert int(r["overall_valid_for_slo"]) == 1
        assert int(r["skipped_missing_tpot"]) == 2
        assert int(r["skipped_missing_slo"]) == 2
        # raw_tokens = all 4 requests = 390
        assert int(r["raw_tokens"]) == 390
        # good_tokens = only the first request (100 tokens)
        assert int(r["good_tokens"]) == 100
        # bad_tokens = the remaining 3 requests = 290
        assert int(r["bad_tokens"]) == 290
        # overall attainment = 1/1 = 1.0
        assert approx(float(r["overall_slo_attainment"]), 1.0)

        print("[PASS] test_missing_fields")


def test_num_decode_output_tokens_preferred() -> None:
    """Test that num_decode_output_tokens is preferred over num_output_tokens."""
    requests = [
        {
            "category": "coding", "slo_class": "tight",
            "tokens": 100,  # num_output_tokens=100
            "num_decode_output_tokens": 60,  # should be preferred
            "observed_tpot_ms": 45, "slo_tpot_ms": 50,
            "slo_attained": True,
        },
        {
            "category": "chat", "slo_class": "normal",
            "tokens": 200,
            "num_decode_output_tokens": 180,
            "observed_tpot_ms": 35, "slo_tpot_ms": 40,
            "slo_attained": True,
        },
    ]
    engine_elapsed_s = 2.0

    with tempfile.TemporaryDirectory() as tmpdir:
        rj, tj = make_synthetic_data(
            tmpdir, "decode_tokens", "1C", 1.2, requests,
            engine_elapsed_s=engine_elapsed_s,
        )
        out_csv = os.path.join(tmpdir, "out.csv")

        result = run_profiler(
            [(("1C", "decode_tokens", 1.2, rj, tj))],
            out_csv, os.path.join(tmpdir, "out.md"),
        )
        assert result.returncode == 0, result.stderr

        with open(out_csv, "r") as f:
            import csv
            reader = csv.DictReader(f)
            rows = list(reader)
        r = rows[0]

        # raw_tokens should use num_decode_output_tokens: 60 + 180 = 240
        assert int(r["raw_tokens"]) == 240, f"raw_tokens should be 240 (60+180), got {r['raw_tokens']}"
        # good_tokens: both satisfied, so 240
        assert int(r["good_tokens"]) == 240
        # raw_throughput = 240/2 = 120
        assert approx(float(r["raw_throughput_tokens_per_s"]), 120.0)

        print("[PASS] test_num_decode_output_tokens_preferred")


def test_slo_class_fallback() -> None:
    """Test category→slo_class fallback when slo_class is missing."""
    requests = [
        {"category": "coding", "tokens": 50,
         "observed_tpot_ms": 30, "slo_tpot_ms": 50, "slo_attained": True},
        {"category": "chat", "tokens": 60,
         "observed_tpot_ms": 50, "slo_tpot_ms": 40, "slo_attained": False},
        {"category": "summarization", "tokens": 70,
         "observed_tpot_ms": 100, "slo_tpot_ms": 150, "slo_attained": True},
    ]
    engine_elapsed_s = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        rj, tj = make_synthetic_data(
            tmpdir, "fallback", "1A", 2.0, requests,
            engine_elapsed_s=engine_elapsed_s,
        )
        # Override the JSONL to omit slo_class
        with open(tj, "r") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for line in lines:
            line.pop("slo_class", None)
        with open(tj, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        out_csv = os.path.join(tmpdir, "out.csv")
        result = run_profiler(
            [(("1A", "fallback", 2.0, rj, tj))],
            out_csv, os.path.join(tmpdir, "out.md"),
        )
        assert result.returncode == 0, result.stderr

        with open(out_csv, "r") as f:
            import csv
            reader = csv.DictReader(f)
            rows = list(reader)
        r = rows[0]

        # coding→tight, chat→normal, summarization→loose
        assert int(r["tight_raw_tokens"]) == 50
        assert int(r["normal_raw_tokens"]) == 60
        assert int(r["loose_raw_tokens"]) == 70
        assert approx(float(r["tight_slo_attainment"]), 1.0)
        assert approx(float(r["normal_slo_attainment"]), 0.0)
        assert approx(float(r["loose_slo_attainment"]), 1.0)

        print("[PASS] test_slo_class_fallback")


def test_grouped_aggregation() -> None:
    """Test grouped CSV with repeated seeds for same OBS+X_VALUE."""
    requests_a = [
        {"category": "coding", "slo_class": "tight", "tokens": 100,
         "observed_tpot_ms": 45, "slo_tpot_ms": 50, "slo_attained": True},
        {"category": "chat", "slo_class": "normal", "tokens": 80,
         "observed_tpot_ms": 35, "slo_tpot_ms": 40, "slo_attained": True},
    ]
    requests_b = [
        {"category": "coding", "slo_class": "tight", "tokens": 110,
         "observed_tpot_ms": 47, "slo_tpot_ms": 50, "slo_attained": True},
        {"category": "chat", "slo_class": "normal", "tokens": 90,
         "observed_tpot_ms": 38, "slo_tpot_ms": 40, "slo_attained": True},
    ]

    engine_elapsed_s = 2.0

    with tempfile.TemporaryDirectory() as tmpdir:
        rj_a, tj_a = make_synthetic_data(tmpdir, "seed0", "1A", 4.0, requests_a, engine_elapsed_s)
        rj_b, tj_b = make_synthetic_data(tmpdir, "seed1", "1A", 4.0, requests_b, engine_elapsed_s)

        out_csv = os.path.join(tmpdir, "out.csv")
        out_gcsv = os.path.join(tmpdir, "grouped.csv")

        result = run_profiler(
            [
                ("1A", "seed0", 4.0, rj_a, tj_a),
                ("1A", "seed1", 4.0, rj_b, tj_b),
            ],
            out_csv, os.path.join(tmpdir, "out.md"),
        )
        assert result.returncode == 0, result.stderr

        # Now run with grouped output
        script = os.path.join(os.path.dirname(__file__), "profile_obs1_goodput.py")
        cmd = [
            sys.executable, script,
            "--run", "1A", "seed0", "4", rj_a, tj_a,
            "--run", "1A", "seed1", "4", rj_b, tj_b,
            "--out-csv", out_csv,
            "--out-grouped-csv", out_gcsv,
        ]
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr

        with open(out_gcsv, "r") as f:
            import csv
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert len(rows) == 1  # one group: (1A, 4.0)
        assert rows[0][0] == "1A"
        assert rows[0][1] == "4"
        assert int(rows[0][2]) == 2  # n_seeds

        # raw_throughput mean: seed0 = 180/2=90, seed1 = 200/2=100 → mean=95
        raw_tp_idx = header.index("raw_throughput_tokens_per_s_mean")
        assert approx(float(rows[0][raw_tp_idx]), 95.0), \
            f"raw_tp mean: {rows[0][raw_tp_idx]}"

        print("[PASS] test_grouped_aggregation")


def test_normalized_class_names() -> None:
    """Test that medium→normal, relaxed→loose, relax→loose work."""
    requests = [
        {"category": "coding", "slo_class": "medium", "tokens": 50,
         "observed_tpot_ms": 30, "slo_tpot_ms": 40, "slo_attained": True},
        {"category": "chat", "slo_class": "relaxed", "tokens": 60,
         "observed_tpot_ms": 100, "slo_tpot_ms": 150, "slo_attained": True},
        {"category": "summarization", "slo_class": "relax", "tokens": 70,
         "observed_tpot_ms": 120, "slo_tpot_ms": 150, "slo_attained": True},
    ]
    engine_elapsed_s = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        rj, tj = make_synthetic_data(tmpdir, "norms", "1C", 1.5, requests, engine_elapsed_s)
        out_csv = os.path.join(tmpdir, "out.csv")

        result = run_profiler(
            [(("1C", "norms", 1.5, rj, tj))],
            out_csv, os.path.join(tmpdir, "out.md"),
        )
        assert result.returncode == 0, result.stderr

        with open(out_csv, "r") as f:
            import csv
            reader = csv.DictReader(f)
            rows = list(reader)
        r = rows[0]

        # medium→normal: raw=50, good=50
        assert int(r["normal_raw_tokens"]) == 50, f"normal_raw_tokens: {r['normal_raw_tokens']}"
        # relaxed→loose: raw=60, good=60
        # relax→loose: raw=70, good=70
        assert int(r["loose_raw_tokens"]) == 130, f"loose_raw_tokens: {r['loose_raw_tokens']}"
        assert int(r["tight_raw_tokens"]) == 0

        print("[PASS] test_normalized_class_names")


def test_missing_engine_elapsed() -> None:
    """Test that missing engine_elapsed_s fails loudly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rj = os.path.join(tmpdir, "bad.json")
        tj = os.path.join(tmpdir, "bad.jsonl")
        with open(rj, "w") as f:
            json.dump({"metrics": {"engine_elapsed_s": 0}}, f)
        with open(tj, "w") as f:
            f.write(json.dumps({"request_id": "x", "observed_tpot_ms": 10,
                                "slo_tpot_ms": 20, "num_output_tokens": 5}) + "\n")

        script = os.path.join(os.path.dirname(__file__), "profile_obs1_goodput.py")
        cmd = [
            sys.executable, script,
            "--run", "1A", "bad", "1", rj, tj,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode != 0, f"Should fail on zero engine_elapsed_s, got code {r.returncode}"
        assert "engine_elapsed_s" in r.stderr.lower()

        print("[PASS] test_missing_engine_elapsed")


def main() -> None:
    print("=== Running profile_obs1_goodput tests ===\n")
    tests = [
        test_core_metrics,
        test_missing_fields,
        test_num_decode_output_tokens_preferred,
        test_slo_class_fallback,
        test_grouped_aggregation,
        test_normalized_class_names,
        test_missing_engine_elapsed,
    ]
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            raise

    print(f"\n=== All {len(tests)} tests passed ===")


if __name__ == "__main__":
    main()
