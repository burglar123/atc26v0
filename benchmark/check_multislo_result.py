#!/usr/bin/env python3
"""Validate multi-SLO result or request-trace files.

Examples:
  python benchmark/check_multislo_result.py results/multislo/*.json
  python benchmark/check_multislo_result.py results/multislo/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def infer_tokens(row: Dict[str, Any]) -> int:
    value = first_present(
        row,
        [
            "num_decode_output_tokens",
            "num_output_tokens",
            "num_completion_tokens",
            "completion_tokens",
            "output_tokens",
            "num_generated_tokens",
            "num_tokens",
        ],
    )
    if isinstance(value, list):
        return len(value)
    try:
        return int(value or 0)
    except Exception:
        return 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_file(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if path.suffix == ".jsonl":
        return {}, load_jsonl(path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {}, [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return {}, []
    for key in ("traces", "requests", "request_traces", "request_summaries"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return payload, [row for row in rows if isinstance(row, dict)]
    return payload, []


def quantiles(values: List[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None
    return min(values), statistics.median(values), max(values)


def fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summarize(path: Path) -> int:
    payload, rows = load_file(path)
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}

    execution_mode = args.get("execution_mode") or metrics.get("execution_mode")
    decode_ready_mode = args.get("decode_ready")
    if decode_ready_mode is None:
        decode_ready_mode = metrics.get("decode_ready_mode")

    engine_elapsed_s = metrics.get("engine_elapsed_s")
    total_output_tokens = overall.get("total_output_tokens")
    goodput = overall.get("goodput_tokens_per_s")
    mean_tpot_ms = overall.get("mean_tpot_ms")
    wall_decode_tokens_per_s = first_present(
        metrics,
        ["wall_decode_tokens_per_s"],
    )
    if wall_decode_tokens_per_s is None:
        wall_decode_tokens_per_s = overall.get("wall_decode_tokens_per_s")
    wall_engine_tokens_per_s = first_present(
        metrics,
        ["wall_engine_tokens_per_s"],
    )
    if wall_engine_tokens_per_s is None:
        wall_engine_tokens_per_s = overall.get("wall_engine_tokens_per_s")

    decode_elapsed_values: List[float] = []
    observed_tpot_values: List[float] = []
    arrival_after_finish = 0
    long_fast = 0
    elapsed_mismatch = 0
    tpot_mismatch = 0

    for row in rows:
        tokens = infer_tokens(row)
        decode_elapsed_ms = to_float(row.get("decode_elapsed_ms"))
        if decode_elapsed_ms is not None:
            decode_elapsed_values.append(decode_elapsed_ms)
        observed_tpot_ms = to_float(row.get("observed_tpot_ms"))
        if observed_tpot_ms is not None:
            observed_tpot_values.append(observed_tpot_ms)

        arrival_ts = to_float(row.get("arrival_ts"))
        finish_ts = to_float(first_present(row, ["finish_ts", "finished_ts", "end_ts", "end_time"]))
        decode_start_ts = to_float(first_present(row, ["decode_start_ts", "decoding_start_ts", "first_decode_ts", "first_token_ts", "start_decode_ts"]))
        if arrival_ts is not None and finish_ts is not None and arrival_ts > finish_ts:
            arrival_after_finish += 1
        if tokens > 200 and decode_elapsed_ms is not None and decode_elapsed_ms < 1000:
            long_fast += 1
        if finish_ts is not None and decode_start_ts is not None and decode_elapsed_ms is not None:
            expected = (finish_ts - decode_start_ts) * 1000.0
            if abs(expected - decode_elapsed_ms) > 1e-3:
                elapsed_mismatch += 1
        if tokens > 0 and decode_elapsed_ms is not None and observed_tpot_ms is not None:
            expected = decode_elapsed_ms / tokens
            if abs(expected - observed_tpot_ms) > 1e-3:
                tpot_mismatch += 1

    dec_min, dec_med, dec_max = quantiles(decode_elapsed_values)
    tpot_min, tpot_med, tpot_max = quantiles(observed_tpot_values)

    print(f"\n== {path} ==")
    print(f"execution_mode: {fmt(execution_mode)}")
    print(f"decode_ready_mode: {fmt(decode_ready_mode)}")
    print(f"engine_elapsed_s: {fmt(engine_elapsed_s)}")
    print(f"total_output_tokens: {fmt(total_output_tokens)}")
    print(f"goodput_tokens_per_s: {fmt(goodput)}")
    print(f"mean_tpot_ms: {fmt(mean_tpot_ms)}")
    print(f"wall_decode_tokens_per_s: {fmt(wall_decode_tokens_per_s)}")
    print(f"wall_engine_tokens_per_s: {fmt(wall_engine_tokens_per_s)}")
    for key in (
        "goodput_metric_numerator_tokens",
        "goodput_metric_denominator_s",
        "goodput_metric_request_filter",
        "goodput_metric_includes_queue_wait",
        "goodput_metric_includes_decode_only",
        "goodput_metric_includes_cache_build",
        "goodput_metric_definition_version",
    ):
        print(f"{key}: {fmt(metrics.get(key, overall.get(key)))}")
    print(f"decode_elapsed_ms min/median/max: {fmt(dec_min)} / {fmt(dec_med)} / {fmt(dec_max)}")
    print(f"observed_tpot_ms min/median/max: {fmt(tpot_min)} / {fmt(tpot_med)} / {fmt(tpot_max)}")
    print(f"arrival_ts > finish_ts rows: {arrival_after_finish}")
    print(f"num_decode_output_tokens > 200 and decode_elapsed_ms < 1000 rows: {long_fast}")
    print(f"decode_elapsed_ms timestamp mismatches: {elapsed_mismatch}")
    print(f"observed_tpot_ms arithmetic mismatches: {tpot_mismatch}")

    anomalous = arrival_after_finish + elapsed_mismatch + tpot_mismatch
    if long_fast:
        print("WARNING: long/fast rows found; inspect whether timestamps reflect true wall-clock completion.")
    if anomalous:
        print("ERROR: timing invariants failed.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Result JSON or request trace JSONL files")
    args = parser.parse_args()

    status = 0
    for raw_path in args.paths:
        try:
            status = max(status, summarize(Path(raw_path)))
        except Exception as exc:
            print(f"ERROR: {raw_path}: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
