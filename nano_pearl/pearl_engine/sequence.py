from __future__ import annotations

from copy import copy
from enum import Enum, auto
from itertools import count
import time
from typing import Any

from ..layers.sampler import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    PENDING_CACHED = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        request_id: str | int | None = None,
        arrival_ts: float | None = None,
        slo_tpot_ms: float | None = None,
        slo_class: str | None = None,
        per_request_gamma: int | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.request_id = self.seq_id if request_id is None else request_id
        self.arrival_ts = time.time() if arrival_ts is None else arrival_ts
        self.arrival_offset_sec = None
        self.first_token_ts = None
        self.admit_ts = None
        self.finish_ts = None
        self.decode_ready_ts = None
        self.decode_start_ts = None
        self.decode_ready_mode = False
        self.num_decode_ready_prefill_tokens = 0
        self.cached_admission_enabled = False
        self.cached_kv_ready = False
        self.cached_prefill_mode = None
        self.cache_key = None
        self.cached_prefill_skipped = False
        self.cached_admission_status = None
        self.cached_kv_materialized = False
        self.slo_tpot_ms = slo_tpot_ms
        self.slo_class = slo_class
        self.per_request_gamma = per_request_gamma
        self.prompt_format_used = None
        self.tokenized_prompt_len = len(token_ids)
        self.home_batch_id = None
        self.trace_stats = {
            "scheduled_iterations": [],
            "accepted_tokens": 0,
            "invalidated_predraft_tokens": 0,
        }

        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.pre_verify = True
        self.num_acc_tokens = []
        self.cur_acc_tokens = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        if self.num_completion_tokens == 0 and self.first_token_ts is None:
            self.first_token_ts = time.time()
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # PEARL KV cache Rollback
    def rollback_tokens(self, n: int):
        assert n > 0
        self.token_ids = self.token_ids[:-n]
        self.last_token = self.token_ids[-1]
        self.num_tokens -= n

    def token_to_slot(self, token_index: int):
        block_id = token_index // self.block_size
        block_offset = token_index % self.block_size
        slot = self.block_table[block_id] * self.block_size + block_offset
        return slot

    def mark_scheduled(self, iteration_id: int, batch_id: str, is_prefill: bool, runner_role: str):
        self.trace_stats["scheduled_iterations"].append(
            {
                "iteration_id": iteration_id,
                "batch_id": batch_id,
                "is_prefill": is_prefill,
                "runner_role": runner_role,
            }
        )

    def record_accepted(self, accepted_len: int):
        self.trace_stats["accepted_tokens"] += int(accepted_len)

    def record_invalidated_predraft(self, invalidated_len: int):
        self.trace_stats["invalidated_predraft_tokens"] += int(invalidated_len)

    def mark_decode_ready(self, ts: float | None = None):
        """Mark this request as decode-ready after an unmeasured prefill phase."""
        self.decode_ready_ts = time.time() if ts is None else ts
        self.decode_ready_mode = True
        self.num_decode_ready_prefill_tokens = self.num_completion_tokens

    def mark_decode_started(self, ts: float | None = None):
        if self.decode_start_ts is None:
            self.decode_start_ts = time.time() if ts is None else ts

    def mark_cached_prefill_metadata(
        self,
        mode: str = "metadata_only",
        cache_key: str | int | None = None,
    ):
        self.cached_admission_enabled = True
        self.cached_kv_ready = True
        self.cached_prefill_mode = mode
        self.cache_key = self.request_id if cache_key is None else cache_key
        self.cached_prefill_skipped = True
        self.cached_admission_status = "pending"

    def mark_cached_admitted(self, ts: float | None = None):
        admit_ts = time.time() if ts is None else ts
        self.admit_ts = admit_ts
        self.cached_admission_status = "running"
        self.mark_decode_ready(admit_ts)

    def mark_cached_materialized(self):
        self.cached_kv_ready = True
        self.cached_kv_materialized = True

    def mark_finished(self, record_finish_ts: bool = True):
        self.status = SequenceStatus.FINISHED
        if record_finish_ts and self.finish_ts is None:
            self.finish_ts = time.time()
        if self.cached_admission_enabled:
            self.cached_admission_status = "completed"

    def service_metadata(self):
        num_decode_output_tokens = max(
            self.num_completion_tokens - self.num_decode_ready_prefill_tokens,
            0,
        )
        decode_elapsed_ms = None
        observed_tpot_ms = None
        queue_wait_ms = None
        if self.decode_start_ts is not None and self.finish_ts is not None:
            decode_elapsed_ms = (self.finish_ts - self.decode_start_ts) * 1000
            if num_decode_output_tokens > 0:
                observed_tpot_ms = decode_elapsed_ms / num_decode_output_tokens
        if self.admit_ts is not None and self.arrival_ts is not None:
            queue_wait_ms = (self.admit_ts - self.arrival_ts) * 1000
        metadata = {
            "seq_id": self.seq_id,
            "request_id": self.request_id,
            "arrival_ts": self.arrival_ts,
            "arrival_offset_sec": self.arrival_offset_sec,
            "first_token_ts": self.first_token_ts,
            "finish_ts": self.finish_ts,
            "admit_ts": self.admit_ts,
            "decode_ready_ts": self.decode_ready_ts,
            "decode_start_ts": self.decode_start_ts,
            "decode_ready_mode": self.decode_ready_mode,
            "num_decode_ready_prefill_tokens": self.num_decode_ready_prefill_tokens,
            "num_decode_output_tokens": num_decode_output_tokens,
            "decode_elapsed_ms": decode_elapsed_ms,
            "observed_tpot_ms": observed_tpot_ms,
            "slo_tpot_ms": self.slo_tpot_ms,
            "slo_class": self.slo_class,
            "per_request_gamma": self.per_request_gamma,
            "prompt_format_used": self.prompt_format_used,
            "tokenized_prompt_len": self.tokenized_prompt_len,
            "home_batch_id": self.home_batch_id,
            "trace_stats": self.trace_stats,
        }
        if self.cached_admission_enabled:
            metadata.update(
                {
                    "cached_admission_enabled": True,
                    "cached_admission_status": self.cached_admission_status,
                    "admission_ts": self.admit_ts,
                    "queue_wait_ms": queue_wait_ms,
                    "cached_kv_ready": self.cached_kv_ready,
                    "cached_kv_materialized": self.cached_kv_materialized,
                    "cached_prefill_mode": self.cached_prefill_mode,
                    "cache_key": self.cache_key,
                    "cached_prefill_skipped": self.cached_prefill_skipped,
                }
            )
        return metadata

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
                self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
                self.arrival_offset_sec,
                self.first_token_ts, self.admit_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
                self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
                self.slo_tpot_ms, self.slo_class, self.per_request_gamma, self.home_batch_id, self.trace_stats,
                self.cached_admission_enabled, self.cached_kv_ready, self.cached_prefill_mode,
                self.cache_key, self.cached_prefill_skipped, self.cached_admission_status,
                self.cached_kv_materialized,
                self.prompt_format_used, self.tokenized_prompt_len,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        fields = state[:-1]
        self.cached_admission_enabled = False
        self.cached_kv_ready = False
        self.cached_prefill_mode = None
        self.cache_key = None
        self.cached_prefill_skipped = False
        self.cached_admission_status = None
        self.cached_kv_materialized = False
        self.prompt_format_used = None
        self.tokenized_prompt_len = None
        if len(fields) == 25:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
             self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
             self.arrival_offset_sec,
             self.first_token_ts, self.admit_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
             self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
             self.slo_tpot_ms, self.slo_class, self.per_request_gamma, self.trace_stats) = fields
            self.home_batch_id = None
        elif len(fields) == 26:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
             self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
             self.arrival_offset_sec,
             self.first_token_ts, self.admit_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
             self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
             self.slo_tpot_ms, self.slo_class, self.per_request_gamma, self.home_batch_id, self.trace_stats) = fields
        else:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
             self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
             self.arrival_offset_sec,
             self.first_token_ts, self.admit_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
             self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
             self.slo_tpot_ms, self.slo_class, self.per_request_gamma, self.home_batch_id, self.trace_stats,
            self.cached_admission_enabled, self.cached_kv_ready, self.cached_prefill_mode,
            self.cache_key, self.cached_prefill_skipped, self.cached_admission_status,
             *rest_cached_fields) = fields
            if rest_cached_fields:
                self.cached_kv_materialized = bool(rest_cached_fields[0])
            if len(rest_cached_fields) >= 3:
                self.prompt_format_used = rest_cached_fields[1]
                self.tokenized_prompt_len = rest_cached_fields[2]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
        if self.tokenized_prompt_len is None:
            self.tokenized_prompt_len = self.num_prompt_tokens


def _sequence_status_name(seq: Sequence) -> str:
    status = getattr(seq, "status", None)
    return status.name if isinstance(status, SequenceStatus) else str(status)


def make_sequence_checkpoint(seq: Sequence) -> dict[str, Any]:
    return {
        "seq_id": int(seq.seq_id),
        "request_id": seq.request_id,
        "len": int(len(seq)),
        "pre_verify": bool(seq.pre_verify),
        "num_completion_tokens": int(seq.num_completion_tokens),
        "cur_acc_tokens": int(seq.cur_acc_tokens),
        "status": _sequence_status_name(seq),
        "home_batch_id": seq.home_batch_id,
    }


def assert_sequence_matches_checkpoint(seq: Sequence, checkpoint: dict[str, Any]) -> None:
    current = make_sequence_checkpoint(seq)
    mismatches = []
    for key, expected in checkpoint.items():
        actual = current.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected={expected!r}, actual={actual!r}")
    assert not mismatches, (
        f"sequence checkpoint mismatch for seq_id={current.get('seq_id')}: "
        + "; ".join(mismatches)
    )


def trace_sequence_checkpoint(seq: Sequence) -> dict[str, Any]:
    return make_sequence_checkpoint(seq)
