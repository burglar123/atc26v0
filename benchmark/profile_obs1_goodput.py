#!/usr/bin/env python3
"""Offline profiling script for Observation 1: raw throughput != SLO-constrained goodput.

Reads existing result JSON + request-level trace JSONL files produced by
benchmark/eval_multi_slo.py and recomputes SLO-constrained metrics from
request-level traces.

Profiling groups:
  1A – active concurrency / batch pressure sweep
  1B – RPS sweep
  1C – SLO tightness sweep

Usage:
  python benchmark/profile_obs1_goodput.py \
    --run 1A active1 1 /tmp/obs1_active1.json /tmp/obs1_active1.jsonl \
    --run 1A active2 2 /tmp/obs1_active2.json /tmp/obs1_active2.jsonl \
    --out-csv /tmp/obs1_summary.csv \
    --out-md /tmp/obs1_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

VALID_OBS = frozenset({"1A", "1B", "1C"})


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        v = row.get(key)
        if v is not None:
            return v
    return None


def infer_tokens(row: Dict[str, Any]) -> int:
    """Prefer num_decode_output_tokens, then num_output_tokens, else 0."""
    n = first_present(row, ["num_decode_output_tokens", "num_output_tokens",
                             "num_completion_tokens", "completion_tokens",
                             "output_tokens", "num_generated_tokens", "num_tokens"])
    if isinstance(n, list):
        return len(n)
    try:
        return int(n or 0)
    except (TypeError, ValueError):
        return 0


def normalize_slo_class(raw: Optional[str], category: Optional[str] = None) -> str:
    """Normalize slo_class names. Fall back to category mapping if missing."""
    if raw is not None:
        s = str(raw).strip().lower()
        if s in ("tight",):
            return "tight"
        if s in ("normal", "medium"):
            return "normal"
        if s in ("loose", "relaxed", "relax"):
            return "loose"
    # fallback: category → slo_class
    if category is not None:
        c = str(category).strip().lower()
        if c == "coding":
            return "tight"
        if c == "chat":
            return "normal"
        if c == "summarization":
            return "loose"
    return "unknown"


def slo_satisfied(observed_tpot_ms: Optional[float],
                  slo_tpot_ms: Optional[float]) -> bool:
    if observed_tpot_ms is None or slo_tpot_ms is None:
        return False
    return observed_tpot_ms <= slo_tpot_ms


def percentile(sorted_values: List[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    k = (p / 100.0) * (len(sorted_values) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def p50(values: List[float]) -> Optional[float]:
    return percentile(sorted(values), 50)


def p90(values: List[float]) -> Optional[float]:
    return percentile(sorted(values), 90)


def p99(values: List[float]) -> Optional[float]:
    return percentile(sorted(values), 99)


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, statistics.stdev(values)


def safe_div(a: float, b: float) -> float:
    return a / max(b, 1e-9)

# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
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


def load_result_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return payload

# ---------------------------------------------------------------------------
# per-run metric computation
# ---------------------------------------------------------------------------


def compute_run_metrics(
    obs: str,
    label: str,
    x_value: float,
    result_json_path: str,
    trace_jsonl_path: str,
) -> Dict[str, Any]:
    # --- load result JSON for metadata ---
    result = load_result_json(result_json_path)
    metrics_block = result.get("metrics", {}) if isinstance(result, dict) else {}
    args_block = result.get("args", {}) if isinstance(result, dict) else {}

    engine_elapsed_s = to_float(metrics_block.get("engine_elapsed_s"))
    if engine_elapsed_s is None or engine_elapsed_s <= 0:
        raise ValueError(
            f"{result_json_path}: engine_elapsed_s is missing or non-positive; "
            f"cannot compute throughput."
        )

    execution_mode = str(
        args_block.get("execution_mode")
        or metrics_block.get("execution_mode")
        or "unknown"
    )
    decode_ready_mode = args_block.get("decode_ready")
    if decode_ready_mode is None:
        decode_ready_mode = metrics_block.get("decode_ready_mode", False)
    decode_ready_mode = bool(decode_ready_mode)

    # --- load trace JSONL ---
    rows = load_jsonl(trace_jsonl_path)
    num_requests = len(rows)

    # --- per-request classification ---
    skipped_missing_tpot = 0
    skipped_missing_slo = 0

    raw_tokens = 0
    good_tokens = 0
    bad_tokens = 0

    # per-class accumulators
    class_raw: Dict[str, int] = defaultdict(int)
    class_good: Dict[str, int] = defaultdict(int)

    # normalized TPOT lists (per request, per class)
    overall_norm_tpot: List[float] = []
    class_norm_tpot: Dict[str, List[float]] = defaultdict(list)

    # admit delay lists (ms)
    overall_admit_delay: List[float] = []
    class_admit_delay: Dict[str, List[float]] = defaultdict(list)

    # SLO attainment counters
    overall_attained = 0
    overall_valid = 0  # requests with valid tpot + slo
    class_attained: Dict[str, int] = defaultdict(int)
    class_total: Dict[str, int] = defaultdict(int)

    for row in rows:
        tokens = infer_tokens(row)
        raw_tokens += tokens

        observed_tpot_ms = to_float(row.get("observed_tpot_ms"))
        slo_tpot_ms = to_float(row.get("slo_tpot_ms"))

        if observed_tpot_ms is None or slo_tpot_ms is None:
            if observed_tpot_ms is None:
                skipped_missing_tpot += 1
            if slo_tpot_ms is None:
                skipped_missing_slo += 1
            # still count tokens as "bad" for goodput
            bad_tokens += tokens
            continue

        overall_valid += 1

        # normalized TPOT
        norm = safe_div(observed_tpot_ms, slo_tpot_ms)
        overall_norm_tpot.append(norm)

        # admit delay
        arrival_ts = to_float(row.get("arrival_ts"))
        admit_ts = to_float(row.get("admit_ts"))
        if arrival_ts is not None and admit_ts is not None:
            delay_ms = (admit_ts - arrival_ts) * 1000.0
            overall_admit_delay.append(delay_ms)

        # SLO class
        slo_cls = normalize_slo_class(
            row.get("slo_class"),
            row.get("category"),
        )

        # SLO check
        ok = slo_satisfied(observed_tpot_ms, slo_tpot_ms)
        if ok:
            overall_attained += 1
            good_tokens += tokens
            class_attained[slo_cls] += 1
            class_good[slo_cls] += tokens
        else:
            bad_tokens += tokens

        class_raw[slo_cls] += tokens
        class_total[slo_cls] += 1
        class_norm_tpot[slo_cls].append(norm)

        if arrival_ts is not None and admit_ts is not None:
            class_admit_delay[slo_cls].append(
                (admit_ts - arrival_ts) * 1000.0
            )

    # --- compute derived metrics ---
    raw_throughput = safe_div(raw_tokens, engine_elapsed_s)
    slo_goodput = safe_div(good_tokens, engine_elapsed_s)
    goodput_gap = raw_throughput - slo_goodput
    goodput_ratio = safe_div(slo_goodput, raw_throughput)

    overall_slo_attainment = safe_div(overall_attained, overall_valid) if overall_valid > 0 else 0.0

    def class_attainment(cls: str) -> float:
        return safe_div(class_attained.get(cls, 0), class_total.get(cls, 0)) if class_total.get(cls, 0) > 0 else 0.0

    def class_goodput(cls: str) -> float:
        return safe_div(class_good.get(cls, 0), engine_elapsed_s)

    # normalized TPOT percentiles
    overall_norm_tpot.sort()
    class_norm_tpot_sorted = {k: sorted(v) for k, v in class_norm_tpot.items()}

    def class_norm_p(cls: str, fn):
        vals = class_norm_tpot_sorted.get(cls, [])
        return fn(vals)

    # admit delay percentiles
    overall_admit_delay.sort()
    class_admit_delay_sorted = {k: sorted(v) for k, v in class_admit_delay.items()}

    def admit_p(vals: List[float], fn):
        return fn(vals)

    def class_admit_p(cls: str, fn):
        vals = class_admit_delay_sorted.get(cls, [])
        return fn(vals)

    # --- assemble result ---
    record: Dict[str, Any] = {
        # metadata
        "obs": obs,
        "label": label,
        "x_value": x_value,
        "result_json": result_json_path,
        "trace_jsonl": trace_jsonl_path,
        "num_requests": num_requests,
        "engine_elapsed_s": engine_elapsed_s,
        "execution_mode": execution_mode,
        "decode_ready_mode": decode_ready_mode,
        # token counts
        "raw_tokens": raw_tokens,
        "good_tokens": good_tokens,
        "bad_tokens": bad_tokens,
        # throughput
        "raw_throughput_tokens_per_s": raw_throughput,
        "slo_goodput_tokens_per_s": slo_goodput,
        "goodput_gap_tokens_per_s": goodput_gap,
        "goodput_ratio": goodput_ratio,
        # SLO attainment
        "overall_slo_attainment": overall_slo_attainment,
        "tight_slo_attainment": class_attainment("tight"),
        "normal_slo_attainment": class_attainment("normal"),
        "loose_slo_attainment": class_attainment("loose"),
        # per-class token goodput
        "tight_raw_tokens": class_raw.get("tight", 0),
        "tight_good_tokens": class_good.get("tight", 0),
        "tight_slo_goodput_tokens_per_s": class_goodput("tight"),
        "normal_raw_tokens": class_raw.get("normal", 0),
        "normal_good_tokens": class_good.get("normal", 0),
        "normal_slo_goodput_tokens_per_s": class_goodput("normal"),
        "loose_raw_tokens": class_raw.get("loose", 0),
        "loose_good_tokens": class_good.get("loose", 0),
        "loose_slo_goodput_tokens_per_s": class_goodput("loose"),
        # overall normalized TPOT
        "overall_normalized_tpot_p50": p50(overall_norm_tpot),
        "overall_normalized_tpot_p90": p90(overall_norm_tpot),
        "overall_normalized_tpot_p99": p99(overall_norm_tpot),
        # per-class normalized TPOT
        "tight_normalized_tpot_p50": class_norm_p("tight", p50),
        "tight_normalized_tpot_p90": class_norm_p("tight", p90),
        "tight_normalized_tpot_p99": class_norm_p("tight", p99),
        "normal_normalized_tpot_p50": class_norm_p("normal", p50),
        "normal_normalized_tpot_p90": class_norm_p("normal", p90),
        "normal_normalized_tpot_p99": class_norm_p("normal", p99),
        "loose_normalized_tpot_p50": class_norm_p("loose", p50),
        "loose_normalized_tpot_p90": class_norm_p("loose", p90),
        "loose_normalized_tpot_p99": class_norm_p("loose", p99),
        # admit delay (overall)
        "admit_delay_ms_p50": admit_p(overall_admit_delay, p50),
        "admit_delay_ms_p90": admit_p(overall_admit_delay, p90),
        "admit_delay_ms_p99": admit_p(overall_admit_delay, p99),
        # admit delay (per-class)
        "tight_admit_delay_ms_p50": class_admit_p("tight", p50),
        "tight_admit_delay_ms_p90": class_admit_p("tight", p90),
        "tight_admit_delay_ms_p99": class_admit_p("tight", p99),
        "normal_admit_delay_ms_p50": class_admit_p("normal", p50),
        "normal_admit_delay_ms_p90": class_admit_p("normal", p90),
        "normal_admit_delay_ms_p99": class_admit_p("normal", p99),
        "loose_admit_delay_ms_p50": class_admit_p("loose", p50),
        "loose_admit_delay_ms_p90": class_admit_p("loose", p90),
        "loose_admit_delay_ms_p99": class_admit_p("loose", p99),
        # diagnostics
        "skipped_missing_tpot": skipped_missing_tpot,
        "skipped_missing_slo": skipped_missing_slo,
        "overall_valid_for_slo": overall_valid,
    }

    if skipped_missing_tpot > 0 or skipped_missing_slo > 0:
        print(
            f"[WARN] {label}: skipped {skipped_missing_tpot} rows missing observed_tpot_ms, "
            f"{skipped_missing_slo} rows missing slo_tpot_ms",
            file=sys.stderr,
        )

    return record

# ---------------------------------------------------------------------------
# CSV / Markdown output
# ---------------------------------------------------------------------------

# All metric fields in output order.
METRIC_FIELDS = [
    "obs",
    "label",
    "x_value",
    "result_json",
    "trace_jsonl",
    "num_requests",
    "engine_elapsed_s",
    "execution_mode",
    "decode_ready_mode",
    "raw_tokens",
    "good_tokens",
    "bad_tokens",
    "raw_throughput_tokens_per_s",
    "slo_goodput_tokens_per_s",
    "goodput_gap_tokens_per_s",
    "goodput_ratio",
    "overall_slo_attainment",
    "tight_slo_attainment",
    "normal_slo_attainment",
    "loose_slo_attainment",
    "tight_raw_tokens",
    "tight_good_tokens",
    "tight_slo_goodput_tokens_per_s",
    "normal_raw_tokens",
    "normal_good_tokens",
    "normal_slo_goodput_tokens_per_s",
    "loose_raw_tokens",
    "loose_good_tokens",
    "loose_slo_goodput_tokens_per_s",
    "overall_normalized_tpot_p50",
    "overall_normalized_tpot_p90",
    "overall_normalized_tpot_p99",
    "tight_normalized_tpot_p50",
    "tight_normalized_tpot_p90",
    "tight_normalized_tpot_p99",
    "normal_normalized_tpot_p50",
    "normal_normalized_tpot_p90",
    "normal_normalized_tpot_p99",
    "loose_normalized_tpot_p50",
    "loose_normalized_tpot_p90",
    "loose_normalized_tpot_p99",
    "admit_delay_ms_p50",
    "admit_delay_ms_p90",
    "admit_delay_ms_p99",
    "tight_admit_delay_ms_p50",
    "tight_admit_delay_ms_p90",
    "tight_admit_delay_ms_p99",
    "normal_admit_delay_ms_p50",
    "normal_admit_delay_ms_p90",
    "normal_admit_delay_ms_p99",
    "loose_admit_delay_ms_p50",
    "loose_admit_delay_ms_p90",
    "loose_admit_delay_ms_p99",
    "skipped_missing_tpot",
    "skipped_missing_slo",
    "overall_valid_for_slo",
]


def fmt_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, bool):
        return str(v)
    return str(v)


def write_csv(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: fmt_val(rec.get(k)) for k in METRIC_FIELDS})
    print(f"[OK] Wrote per-run CSV: {path}")


# --- Markdown table (selected key columns) ---
MD_COLUMNS = [
    ("obs", "Obs"),
    ("label", "Label"),
    ("x_value", "X"),
    ("num_requests", "N"),
    ("engine_elapsed_s", "Elapsed(s)"),
    ("raw_throughput_tokens_per_s", "Raw(tok/s)"),
    ("slo_goodput_tokens_per_s", "SLO-Goodput"),
    ("goodput_ratio", "Ratio"),
    ("overall_slo_attainment", "Attain"),
    ("tight_slo_attainment", "Tight-Att"),
    ("normal_slo_attainment", "Norm-Att"),
    ("loose_slo_attainment", "Loose-Att"),
    ("overall_normalized_tpot_p50", "nTPOT-p50"),
    ("overall_normalized_tpot_p90", "nTPOT-p90"),
    ("overall_normalized_tpot_p99", "nTPOT-p99"),
    ("tight_slo_goodput_tokens_per_s", "Tight-GP"),
    ("normal_slo_goodput_tokens_per_s", "Norm-GP"),
    ("loose_slo_goodput_tokens_per_s", "Loose-GP"),
]


def write_markdown(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # header
        headers = [h for _, h in MD_COLUMNS]
        keys = [k for k, _ in MD_COLUMNS]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(" --- " for _ in headers) + "|\n")
        for rec in records:
            vals = [fmt_val(rec.get(k)) for k in keys]
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"[OK] Wrote Markdown summary: {path}")


# ---------------------------------------------------------------------------
# grouped aggregation (mean/std across repeated seeds)
# ---------------------------------------------------------------------------

GROUP_AGG_FIELDS = [
    "raw_throughput_tokens_per_s",
    "slo_goodput_tokens_per_s",
    "overall_slo_attainment",
    "tight_slo_attainment",
    "normal_slo_attainment",
    "loose_slo_attainment",
    "overall_normalized_tpot_p50",
    "overall_normalized_tpot_p90",
    "overall_normalized_tpot_p99",
    "tight_normalized_tpot_p50",
    "tight_normalized_tpot_p90",
    "tight_normalized_tpot_p99",
    "normal_normalized_tpot_p50",
    "normal_normalized_tpot_p90",
    "normal_normalized_tpot_p99",
    "loose_normalized_tpot_p50",
    "loose_normalized_tpot_p90",
    "loose_normalized_tpot_p99",
]


def write_grouped_csv(records: List[Dict[str, Any]], path: str) -> None:
    # Group by (obs, x_value)
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec["obs"], rec["x_value"])].append(rec)

    header = ["obs", "x_value", "n_seeds"] + [
        f"{field}_mean" for field in GROUP_AGG_FIELDS
    ] + [
        f"{field}_std" for field in GROUP_AGG_FIELDS
    ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # sort by obs then x_value
        for (obs, xv), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            means: List[str] = []
            stds: List[str] = []
            for field in GROUP_AGG_FIELDS:
                vals = [to_float(r.get(field)) for r in recs]
                vals = [v for v in vals if v is not None]
                if vals:
                    m, s = mean_std(vals)
                    means.append(fmt_val(m))
                    stds.append(fmt_val(s))
                else:
                    means.append("")
                    stds.append("")
            writer.writerow([obs, fmt_val(xv), len(recs)] + means + stds)

    print(f"[OK] Wrote grouped CSV: {path}")

# ---------------------------------------------------------------------------
# optional plotting
# ---------------------------------------------------------------------------


def make_plots(records: List[Dict[str, Any]], plot_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plots.", file=sys.stderr)
        return

    os.makedirs(plot_dir, exist_ok=True)

    # Partition records by obs
    obs_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        obs_groups[rec["obs"]].append(rec)

    for obs, recs in obs_groups.items():
        recs.sort(key=lambda r: r["x_value"])
        xs = [r["x_value"] for r in recs]

        x_label = {"1A": "Active Concurrency", "1B": "RPS", "1C": "SLO Scale"}.get(obs, "X")

        # ----- Plot 1: raw throughput vs SLO goodput -----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, [r["raw_throughput_tokens_per_s"] for r in recs],
                "o-", label="Raw Throughput (tok/s)")
        ax.plot(xs, [r["slo_goodput_tokens_per_s"] for r in recs],
                "s-", label="SLO Goodput (tok/s)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Tokens / s")
        ax.set_title(f"Obs {obs}: Raw Throughput vs SLO Goodput")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, f"obs{obs}_throughput_vs_goodput.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ----- Plot 2: SLO attainment by class -----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, [r["overall_slo_attainment"] for r in recs],
                "o-", label="Overall")
        ax.plot(xs, [r["tight_slo_attainment"] for r in recs],
                "s-", label="Tight")
        ax.plot(xs, [r["normal_slo_attainment"] for r in recs],
                "D-", label="Normal")
        ax.plot(xs, [r["loose_slo_attainment"] for r in recs],
                "^-", label="Loose")
        ax.set_xlabel(x_label)
        ax.set_ylabel("SLO Attainment")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Obs {obs}: SLO Attainment by Class")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, f"obs{obs}_slo_attainment.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ----- Plot 3: normalized TPOT p90 by class -----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(xs, [r["tight_normalized_tpot_p90"] for r in recs],
                "s-", label="Tight nTPOT p90")
        ax.plot(xs, [r["normal_normalized_tpot_p90"] for r in recs],
                "D-", label="Normal nTPOT p90")
        ax.plot(xs, [r["loose_normalized_tpot_p90"] for r in recs],
                "^-", label="Loose nTPOT p90")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Normalized TPOT p90")
        ax.set_title(f"Obs {obs}: Normalized TPOT p90 by Class (y=1 = SLO)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plot_dir, f"obs{obs}_norm_tpot_p90.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Wrote plots to: {plot_dir}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_run_arg(run_spec: List[str]) -> Tuple[str, str, float, str, str]:
    if len(run_spec) != 5:
        raise ValueError(
            f"--run expects 5 values: OBS LABEL X_VALUE RESULT_JSON TRACE_JSONL; "
            f"got {len(run_spec)}: {run_spec}"
        )
    obs, label, x_str, result_json, trace_jsonl = run_spec
    obs = obs.upper()
    if obs not in VALID_OBS:
        raise ValueError(f"OBS must be one of {sorted(VALID_OBS)}; got {obs!r}")
    try:
        x_value = float(x_str)
    except ValueError:
        raise ValueError(f"X_VALUE must be numeric; got {x_str!r}")
    return obs, label, x_value, result_json, trace_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline SLO-constrained goodput profiling for Observation 1."
    )
    parser.add_argument(
        "--run", action="append", nargs=5, dest="runs", default=[],
        metavar=("OBS", "LABEL", "X_VALUE", "RESULT_JSON", "TRACE_JSONL"),
        help="Add a run (repeatable). OBS ∈ {1A,1B,1C}.",
    )
    parser.add_argument("--out-csv", type=str, default=None, help="Per-run CSV output path.")
    parser.add_argument("--out-md", type=str, default=None, help="Markdown summary output path.")
    parser.add_argument("--out-grouped-csv", type=str, default=None,
                        help="Grouped (mean/std) CSV output path.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="If set, generate matplotlib plots in this directory.")
    args = parser.parse_args()

    if not args.runs:
        print("ERROR: at least one --run is required.", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(2)

    # Parse all runs
    parsed_runs: List[Tuple[str, str, float, str, str]] = []
    for run_spec in args.runs:
        try:
            parsed_runs.append(parse_run_arg(run_spec))
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    # Compute metrics
    records: List[Dict[str, Any]] = []
    errors = 0
    for obs, label, xv, res_json, trace_jsonl in parsed_runs:
        try:
            rec = compute_run_metrics(obs, label, xv, res_json, trace_jsonl)
            records.append(rec)
            print(
                f"[{obs}] {label} (x={xv}): "
                f"raw={rec['raw_throughput_tokens_per_s']:.1f} tok/s, "
                f"goodput={rec['slo_goodput_tokens_per_s']:.1f} tok/s, "
                f"ratio={rec['goodput_ratio']:.3f}, "
                f"attain={rec['overall_slo_attainment']:.3f}"
            )
        except Exception as exc:
            print(f"ERROR [{label}]: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"ERROR: {errors} run(s) failed.", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("ERROR: no valid records produced.", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    if args.out_csv:
        write_csv(records, args.out_csv)
    if args.out_md:
        write_markdown(records, args.out_md)
    if args.out_grouped_csv:
        write_grouped_csv(records, args.out_grouped_csv)
    if args.plot_dir:
        make_plots(records, args.plot_dir)

    # Terminal summary
    print()
    print(f"=== Summary: {len(records)} run(s) across {len(set(r['obs'] for r in records))} obs group(s) ===")
    for rec in records:
        print(
            f"  [{rec['obs']}] {rec['label']:20s} x={rec['x_value']:6g}  "
            f"raw={rec['raw_throughput_tokens_per_s']:8.1f} tok/s  "
            f"goodput={rec['slo_goodput_tokens_per_s']:8.1f} tok/s  "
            f"ratio={rec['goodput_ratio']:.3f}  "
            f"attain={rec['overall_slo_attainment']:.3f}"
        )


if __name__ == "__main__":
    main()
