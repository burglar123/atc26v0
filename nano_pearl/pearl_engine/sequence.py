from copy import copy
from enum import Enum, auto
from itertools import count
import time

from ..layers.sampler import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
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
        home_batch_id: int | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.request_id = self.seq_id if request_id is None else request_id
        self.arrival_ts = time.time() if arrival_ts is None else arrival_ts
        self.first_token_ts = None
        self.finish_ts = None
        self.decode_ready_ts = None
        self.decode_start_ts = None
        self.decode_ready_mode = False
        self.num_decode_ready_prefill_tokens = 0
        self.slo_tpot_ms = slo_tpot_ms
        self.slo_class = slo_class
        self.per_request_gamma = per_request_gamma
        # ST-Spec V3A shadow metadata. This is only a stable home-batch
        # membership tag for future two-batch scheduling; it does not affect
        # scheduling, verification, rollback, or token generation today.
        self.home_batch_id = home_batch_id
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

    def mark_finished(self, record_finish_ts: bool = True):
        self.status = SequenceStatus.FINISHED
        if record_finish_ts and self.finish_ts is None:
            self.finish_ts = time.time()

    def service_metadata(self):
        num_decode_output_tokens = max(
            self.num_completion_tokens - self.num_decode_ready_prefill_tokens,
            0,
        )
        decode_elapsed_ms = None
        observed_tpot_ms = None
        if self.decode_start_ts is not None and self.finish_ts is not None:
            decode_elapsed_ms = (self.finish_ts - self.decode_start_ts) * 1000
            if num_decode_output_tokens > 0:
                observed_tpot_ms = decode_elapsed_ms / num_decode_output_tokens
        return {
            "seq_id": self.seq_id,
            "request_id": self.request_id,
            "arrival_ts": self.arrival_ts,
            "first_token_ts": self.first_token_ts,
            "finish_ts": self.finish_ts,
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
            "home_batch_id": self.home_batch_id,
            "trace_stats": self.trace_stats,
        }

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
                self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
                self.first_token_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
                self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
                self.slo_tpot_ms, self.slo_class, self.per_request_gamma,
                self.home_batch_id, self.trace_stats,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        metadata = state[:-1]
        token_state = state[-1]
        if len(metadata) == 24:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
             self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
             self.first_token_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
             self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
             self.slo_tpot_ms, self.slo_class, self.per_request_gamma,
             self.home_batch_id, self.trace_stats) = metadata
        else:
            # Backward compatibility for older pickles without ST-Spec V3A
            # shadow home-batch metadata. New requests will be assigned by
            # Scheduler.add() if this remains None.
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.temperature, self.ignore_eos, self.max_tokens, self.seq_id, self.pre_verify,
             self.num_acc_tokens, self.cur_acc_tokens, self.request_id, self.arrival_ts,
             self.first_token_ts, self.finish_ts, self.decode_ready_ts, self.decode_start_ts,
             self.decode_ready_mode, self.num_decode_ready_prefill_tokens,
             self.slo_tpot_ms, self.slo_class, self.per_request_gamma, self.trace_stats) = metadata
            self.home_batch_id = None
        if self.num_completion_tokens == 0:
            self.token_ids = token_state
        else:
            self.last_token = token_state
