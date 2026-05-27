# Observation 1 Profiling

Goal: show that raw token throughput is not equivalent to SLO-constrained goodput under heterogeneous TPOT-SLO serving.

System configuration:
- Draft model: Qwen3-0.6B
- Target model: Qwen3-32B
- Draft TP: 1
- Target TP: 3
- Execution mode: parallel_pearl
- Decode-ready: true
- Cached admission: true
- Cache build batch size: 32
- Gamma: 4
- Max tokens: 256
- GPU memory utilization: 0.85

Profiling groups:
- 1A: active concurrency / batch pressure sweep
- 1B: RPS sweep
- 1C: SLO tightness sweep

Metrics:
- raw_throughput_tokens_per_s = all decode-stage output tokens / engine_elapsed_s
- slo_goodput_tokens_per_s = SLO-satisfied decode-stage output tokens / engine_elapsed_s
- SLO satisfied iff observed_tpot_ms <= slo_tpot_ms

Note:
The raw request-level JSONL traces and engine traces are not committed here. This directory keeps only aggregated summaries and plots.
