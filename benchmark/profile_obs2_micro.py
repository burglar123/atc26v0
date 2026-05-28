#!/usr/bin/env python3
"""Micro-profiling summary for Observation 2: per-iteration decode metrics.

Reads an --engine-trace-out JSON file and produces a CSV summary of
decode-iteration records (trace_type == "decode_iteration").

Usage:
  python benchmark/profile_obs2_micro.py \\
    --engine-trace /tmp/obs2_micro_serial_sanity_engine_trace.json \\
    --out /tmp/obs2_micro_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = (p / 100.0) * (len(sorted_vals) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def load_engine_trace(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        # Try common container keys.
        for key in ("traces", "events", "iterations"):
            cand = obj.get(key)
            if isinstance(cand, list):
                return cand
        # Fallback: return the whole dict wrapped.
        return [obj]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unexpected engine trace format in {path}: {type(obj).__name__}")


def _get(r: Dict[str, Any], canonical: str, *raw_aliases: str, default=None):
    """Return the first present value from canonical or raw-alias keys."""
    if canonical in r:
        return r[canonical]
    for alias in raw_aliases:
        if alias in r:
            return r[alias]
    return default


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Prefer merged iteration-level records, then fall back to raw decode_iteration.
    iters = [r for r in records if r.get("record_level") == "merged_iteration"]
    if not iters:
        iters = [r for r in records if r.get("trace_type") == "decode_iteration"]
    if not iters:
        # Fall back to records that look like decode steps.
        iters = [r for r in records if not r.get("is_prefill") and r.get("runner_role", "") in ("draft", "verify", "serialized_draft", "serialized_verify")]
    if not iters:
        return {"num_iterations": 0}

    mode = iters[0].get("execution_mode", "unknown")
    n = len(iters)

    batch_sizes = [_get(r, "active_batch_size", "num_seqs_in_batch", default=0) for r in iters]
    draft_times = [to_float(_get(r, "draft_time_ms")) for r in iters if to_float(_get(r, "draft_time_ms")) is not None]
    verify_times = [to_float(_get(r, "verify_time_ms")) for r in iters if to_float(_get(r, "verify_time_ms")) is not None]
    iter_times = [to_float(_get(r, "iter_time_ms", "total_iteration_time_ms")) for r in iters if to_float(_get(r, "iter_time_ms", "total_iteration_time_ms")) is not None]
    accepted_totals = [_get(r, "accepted_tokens_total", "total_accepted_tokens", default=0) for r in iters]
    drafted_totals = [_get(r, "drafted_tokens_total", default=0) for r in iters]
    verified_totals = [_get(r, "verified_tokens_total", default=0) for r in iters]
    invalidated_totals = [
        sum(_get(r, "invalidated_predraft_tokens_by_request", default={}).values()) for r in iters
    ]

    # Accepted tokens per request per iteration.
    accepted_per_req_per_iter: List[float] = []
    for r in iters:
        by_req = _get(r, "accepted_tokens_by_request", "accepted_tokens_per_seq", default={})
        if by_req:
            accepted_per_req_per_iter.extend(float(v) for v in by_req.values())

    iter_times_sorted = sorted(iter_times)

    return {
        "execution_mode": mode,
        "num_iterations": n,
        "avg_active_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0.0,
        "mean_draft_time_ms": statistics.mean(draft_times) if draft_times else None,
        "mean_verify_time_ms": statistics.mean(verify_times) if verify_times else None,
        "mean_iter_time_ms": statistics.mean(iter_times) if iter_times else None,
        "p50_iter_time_ms": percentile(iter_times_sorted, 50),
        "p90_iter_time_ms": percentile(iter_times_sorted, 90),
        "mean_accepted_tokens_total": statistics.mean(accepted_totals) if accepted_totals else 0.0,
        "mean_accepted_tokens_per_request_per_iter": (
            statistics.mean(accepted_per_req_per_iter) if accepted_per_req_per_iter else 0.0
        ),
        "mean_drafted_tokens_total": statistics.mean(drafted_totals) if drafted_totals else None,
        "mean_verified_tokens_total": statistics.mean(verified_totals) if verified_totals else None,
        "mean_invalidated_predraft_tokens_total": statistics.mean(invalidated_totals) if invalidated_totals else None,
    }


def write_csv(summary: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "execution_mode",
        "num_iterations",
        "avg_active_batch_size",
        "mean_draft_time_ms",
        "mean_verify_time_ms",
        "mean_iter_time_ms",
        "p50_iter_time_ms",
        "p90_iter_time_ms",
        "mean_accepted_tokens_total",
        "mean_accepted_tokens_per_request_per_iter",
        "mean_drafted_tokens_total",
        "mean_verified_tokens_total",
        "mean_invalidated_predraft_tokens_total",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in summary.items()})
    print(f"[OK] Wrote micro summary: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-profiling summary for Observation 2."
    )
    parser.add_argument("--engine-trace", type=str, required=True,
                        help="Path to engine_trace JSON file.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path.")
    args = parser.parse_args()

    records = load_engine_trace(args.engine_trace)
    summary = compute_summary(records)

    if summary.get("num_iterations", 0) == 0:
        print(f"[WARN] No decode_iteration records found in {args.engine_trace}", file=sys.stderr)
        sys.exit(0)

    print(f"execution_mode:           {summary['execution_mode']}")
    print(f"num_iterations:           {summary['num_iterations']}")
    print(f"avg_active_batch_size:    {summary['avg_active_batch_size']:.2f}")
    if summary["mean_draft_time_ms"] is not None:
        print(f"mean_draft_time_ms:       {summary['mean_draft_time_ms']:.3f}")
    if summary["mean_verify_time_ms"] is not None:
        print(f"mean_verify_time_ms:      {summary['mean_verify_time_ms']:.3f}")
    print(f"mean_iter_time_ms:        {summary['mean_iter_time_ms']:.3f}")
    print(f"p50_iter_time_ms:         {summary['p50_iter_time_ms']:.3f}")
    print(f"p90_iter_time_ms:         {summary['p90_iter_time_ms']:.3f}")
    print(f"mean_accepted_tokens_total: {summary['mean_accepted_tokens_total']:.2f}")
    print(f"mean_accepted_per_req_per_iter: {summary['mean_accepted_tokens_per_request_per_iter']:.3f}")
    if summary["mean_drafted_tokens_total"] is not None:
        print(f"mean_drafted_tokens_total:  {summary['mean_drafted_tokens_total']:.2f}")
    if summary["mean_invalidated_predraft_tokens_total"] is not None:
        print(f"mean_invalidated_predraft:   {summary['mean_invalidated_predraft_tokens_total']:.2f}")

    if args.out:
        write_csv(summary, args.out)


if __name__ == "__main__":
    main()
