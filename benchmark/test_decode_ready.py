#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


def main():
    from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams
    DRAFT_MODEL = "/root/autodl-tmp/models/Qwen3-0.6B"
    TARGET_MODEL = "/root/autodl-tmp/models/Qwen3-8B"

    config = PEARLConfig(
        DRAFT_MODEL,
        TARGET_MODEL,
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        execution_mode="parallel_pearl",
    )

    engine = PEARLEngine(config)

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=False,
        max_tokens=16,
    )

    engine.add_request("Hello, my name is", sampling_params)
    engine.prepare_decode_ready()
    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.decode_ready_generate("parallel_pearl")

    print("output_text:", output_text)
    print("num_tokens:", num_tokens)
    print("num_acc_tokens:", num_acc_tokens)
    print("elapsed_time:", elapsed_time)

    trace_path = "decode_ready_trace.json"
    engine.dump_traces_json(trace_path)
    print(f"trace saved to: {trace_path}")

if __name__ == "__main__":
    main()