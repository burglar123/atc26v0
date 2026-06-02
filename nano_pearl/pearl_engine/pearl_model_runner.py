import pickle
import torch
import time
import random
import tempfile
import os
from collections import deque
from abc import abstractmethod
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from nano_pearl.utils.pearl_logger import logger
from nano_pearl.pearl_config import PEARLConfig
from dataclasses import dataclass
from nano_pearl.models import model_dict
from nano_pearl.utils.loader import load_model
from nano_pearl.pearl_config import TPParams
from nano_pearl.layers.sampler import Sampler, norm_logits, SamplingParams
from nano_pearl.utils.context import set_context, reset_context, get_context
from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.pearl_engine.scheduler import Scheduler, is_eos
from nano_pearl.pearl_engine.sequence import SequenceStatus
from transformers import AutoTokenizer
from tqdm import trange


class ModelRunnerBase:
    """
    Different from ModelRunner in nano-vllm, 
    all the ModelRunner sub-processes are forked from the main process.
    we will define a controller to control the sub-processes and shared memory.
    """
    def __init__(self, config: PEARLConfig, rank: int, event: Event, control_event: Event):
        self.rank = rank
        self.event = event
        self.is_draft = rank in config.draft_config.devices
        # global config for PEARL, group config for the draft / target group
        self.global_config = config
        self.group_config = config.draft_config if self.is_draft else config.target_config
        self.hf_config = self.group_config.hf_config
        self.control_event = control_event if rank == 0  else None

        self.block_size = self.global_config.kvcache_block_size
        self.tensor_parallel_size = self.group_config.tensor_parallel_size
        self.group_name = self.group_config.group_name
        self.gamma = self.global_config.gamma

        self.init_dist()
        self.init_model_and_kvcache()
        if self.gamma == -1:
            self.auto_set_gamma()
        self.init_shared_memory()
        
    def init_dist(self):
        """
        We use a global process group to initialize the dist.
        Create 3 sub-groups for the draft and target group and verify group.
        """
        dist.init_process_group("nccl", 
                                f"tcp://localhost:2333", 
                                world_size=self.global_config.world_size,
                                rank=self.rank)
        draft_group = dist.new_group(self.global_config.draft_config.devices)
        target_group = dist.new_group(self.global_config.target_config.devices)
        verify_group = dist.new_group([self.global_config.draft_config.master_rank] + self.global_config.target_config.devices )
        self.group = draft_group if self.is_draft else target_group
        self.verify_group = verify_group

        # IMPORTANT: tp_params is used to specify the TP settings everywhere.
        self.tp_params = TPParams(
            rank=self.rank,
            group=self.group,
            group_name=self.group_name,
            local_rank=self.rank if self.is_draft else self.rank - self.global_config.draft_config.tensor_parallel_size,
            master_rank=self.group_config.master_rank,
            is_draft=self.is_draft,
            tp_size=self.tensor_parallel_size,
            valid_vocab_size= getattr(self.hf_config, "valid_vocab_size", self.hf_config.vocab_size)
        )
        dist.barrier()
        if self.rank == 0:
            logger.info("initialized dist.", color="blue")
        
    def init_shared_memory(self):
        """
        Initialize the shared memory for the sub-processes.
        we do not use the main model runner to create the shared memory.
        """
        dist.barrier()
        self.shm = SharedMemory(name=self.group_name)
        if self.rank == 0:
            logger.info(f"[Sub-Process] Draft Model and Target Model initialized. Starting to run the model...", color="yellow")
            self.control_event.set()
        self.loop()
    
    def init_model_and_kvcache(self):
        """
        Initialize the model and kvcache for the sub-processes.
        note that in nano-PEARL, the model requires a tp_params to specify the TP settings.
        """
        self.default_dtype = torch.get_default_dtype()
        torch.cuda.set_device(self.rank)
        torch.set_default_dtype(self.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = model_dict[self.hf_config.architectures[0]](self.hf_config, self.tp_params)
        load_model(self.model, self.group_config.model)
        dist.barrier()
        self.sampler = Sampler()
        self.warmup_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.group_config.model)
        self.allocate_kv_cache()
        self.scheduler = Scheduler(self.global_config)
        self.trace_records = []
        self._decode_iteration_group = 0
        self.active_execution_mode = self.global_config.execution_mode
        self.active_decode_ready_mode = False
        self.cached_kv_store = {}
        self.cached_admission_log_interval = 32
        self.last_result_used_file_fallback = False
        if not self.global_config.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_dtype(self.default_dtype)
        torch.set_default_device("cpu")
        dist.barrier()
        model_device = next(self.model.parameters()).device
        if self.rank == 0:
            logger.info(f"initialized model, kvcache and scheduler. ", color="green")
        
    def allocate_kv_cache(self):
        hf_config = self.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.tensor_parallel_size
        head_dim = (
            hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            else hf_config.hidden_size // hf_config.num_attention_heads
        )
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        self.global_config.num_kvcache_blocks = int(total * self.global_config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert self.global_config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, self.global_config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        dist.barrier()
        if self.tp_params.local_rank == 0:
            logger.info(f"[Rank {self.rank}: {self.group_name}] allocated GPU memory {self.global_config.num_kvcache_blocks * block_bytes / 2**30} GiB for kvcache.", color="green")

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if self.rank == 0 and method_name != "exit":
                self.control_event.set()
            
            if method_name == "exit":
                break

    def read_shm(self):
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)

    def exit(self):
        self.shm.close()
        if not self.global_config.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def log(self, content: str):
        logger.info(f"[Rank {self.rank}: {self.group_name}] Log: {content}")
    
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(self.tp_params, True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(self.tp_params, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.global_config.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context(self.tp_params)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.global_config
        hf_config = self.hf_config
        max_bs = min(self.global_config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(self.tp_params, False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context(self.tp_params)

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        logger.info("CUDA graph captured.", color="blue")
        dist.barrier()

    def add_request(self, seq: Sequence):
        self.scheduler.add(seq)
        dist.barrier()

    def add_cached_request(self, seq: Sequence):
        self.scheduler.add_cached(seq)
        dist.barrier()

    def _build_cached_seq_snapshot(self, seq: Sequence, arrival_offset_sec: float | None = None) -> dict:
        return {
            "request_id": seq.request_id,
            "token_ids": list(seq.token_ids),
            "num_prompt_tokens": int(seq.num_prompt_tokens),
            "num_tokens": int(seq.num_tokens),
            "last_token": int(seq.last_token),
            "pre_verify": bool(seq.pre_verify),
            "pending_draft_tokens": int(seq.pending_draft_tokens),
            "max_tokens": int(seq.max_tokens),
            "temperature": float(seq.temperature),
            "ignore_eos": bool(seq.ignore_eos),
            "arrival_ts": float(seq.arrival_ts),
            "arrival_offset_sec": arrival_offset_sec if arrival_offset_sec is not None else getattr(seq, "arrival_offset_sec", None),
            "slo_tpot_ms": seq.slo_tpot_ms,
            "slo_class": seq.slo_class,
            "per_request_gamma": seq.per_request_gamma,
            "num_decode_ready_prefill_tokens": int(seq.num_decode_ready_prefill_tokens),
            "decode_ready_mode": bool(seq.decode_ready_mode),
            "trace_stats": seq.trace_stats,
        }

    def _restore_sequence_from_snapshot(self, snapshot: dict) -> Sequence:
        sampling_params = SamplingParams(
            temperature=float(snapshot["temperature"]),
            max_tokens=int(snapshot["max_tokens"]),
            ignore_eos=bool(snapshot["ignore_eos"]),
        )
        seq = Sequence(
            list(snapshot["token_ids"]),
            sampling_params,
            request_id=snapshot["request_id"],
            arrival_ts=float(snapshot["arrival_ts"]),
            slo_tpot_ms=snapshot.get("slo_tpot_ms"),
            slo_class=snapshot.get("slo_class"),
            per_request_gamma=snapshot.get("per_request_gamma"),
        )
        seq.num_prompt_tokens = int(snapshot["num_prompt_tokens"])
        seq.num_tokens = int(snapshot["num_tokens"])
        seq.last_token = int(snapshot["last_token"])
        seq.pre_verify = bool(snapshot["pre_verify"])
        seq.pending_draft_tokens = int(snapshot.get("pending_draft_tokens", 0))
        seq.decode_ready_mode = bool(snapshot.get("decode_ready_mode", False))
        seq.num_decode_ready_prefill_tokens = int(snapshot.get("num_decode_ready_prefill_tokens", 0))
        seq.trace_stats = snapshot.get("trace_stats", seq.trace_stats)
        seq.arrival_offset_sec = snapshot.get("arrival_offset_sec")
        return seq

    def _allocate_cached_blocks(self, seq: Sequence, num_blocks: int) -> list[int]:
        assert len(self.scheduler.block_manager.free_block_ids) >= num_blocks, (
            f"Insufficient free blocks: need={num_blocks}, free={len(self.scheduler.block_manager.free_block_ids)}"
        )
        seq.block_table = []
        for _ in range(num_blocks):
            block_id = self.scheduler.block_manager.free_block_ids[0]
            self.scheduler.block_manager._allocate_block(block_id)
            seq.block_table.append(block_id)
        return list(seq.block_table)

    def cache_build_prepare(self, seqs: list[Sequence], cache_build_batch_size: int):
        self.cached_kv_store = {}
        total_cached_kv_cpu_bytes = 0
        num_blocks_list = []
        for i in range(0, len(seqs), cache_build_batch_size):
            chunk = seqs[i:i+cache_build_batch_size]
            for seq in chunk:
                self.scheduler.add(seq)
            dist.barrier()
            # TODO: refactor to a narrower cache_build_prefill_chunk() helper to
            # avoid coupling cache build to broader decode-ready helper side effects.
            self.prepare_decode_ready()
            for seq in list(self.scheduler.running):
                block_ids = list(seq.block_table)
                kv_cpu = self.kv_cache[:, :, block_ids].detach().cpu().clone()
                total_cached_kv_cpu_bytes += kv_cpu.element_size() * kv_cpu.nelement()
                num_blocks_list.append(len(block_ids))
                self.cached_kv_store[seq.request_id] = {
                    "snapshot": self._build_cached_seq_snapshot(
                        seq,
                        arrival_offset_sec=getattr(seq, "arrival_offset_sec", None),
                    ),
                    "kv_cpu": kv_cpu,
                    "num_blocks": len(block_ids),
                }
            dist.barrier()
            self.clear_requests()
        key_set = set(self.cached_kv_store.keys())
        gathered = [None for _ in range(self.tensor_parallel_size)]
        dist.all_gather_object(gathered, key_set, group=self.group)
        assert all(g == key_set for g in gathered), "cached_kv_store key-set mismatch across TP ranks"
        if self.tp_params.local_rank == 0:
            avg_blocks = (sum(num_blocks_list) / len(num_blocks_list)) if num_blocks_list else 0.0
            max_blocks = max(num_blocks_list) if num_blocks_list else 0
            logger.info(
                f"[Rank {self.rank}: {self.group_name}] cache_build_prepare stored {len(key_set)} requests, "
                f"cached_kv_cpu_bytes={total_cached_kv_cpu_bytes}, avg_blocks={avg_blocks:.2f}, max_blocks={max_blocks}",
                color="green",
            )
        dist.barrier()

    def materialize_cached_request(self, request_id: str, admit_ts: float):
        assert request_id in self.cached_kv_store, (
            f"[Rank {self.rank}: {self.group_name}] missing cached request_id={request_id}"
        )
        cached = self.cached_kv_store[request_id]
        seq: Sequence = self._restore_sequence_from_snapshot(cached["snapshot"])
        assert hasattr(seq, "token_ids"), "restored sequence missing token_ids"
        seq.status = SequenceStatus.RUNNING
        seq.admit_ts = admit_ts
        seq.mark_decode_ready(admit_ts)
        seq.decode_start_ts = None
        seq.block_table = []
        seq.num_cached_tokens = 0
        new_block_ids = self._allocate_cached_blocks(seq, int(cached["num_blocks"]))
        assert len(new_block_ids) == int(cached["num_blocks"])
        seq.num_cached_tokens = seq.num_prompt_tokens
        kv_cpu = cached["kv_cpu"].to(self.kv_cache.device)
        assert kv_cpu.size(2) == int(cached["num_blocks"])
        self.kv_cache[:, :, new_block_ids] = kv_cpu
        self.scheduler.running.append(seq)

    def _runner_role(self):
        return "draft" if self.is_draft else "verify"

    def _set_execution_mode(self, execution_mode: str):
        if execution_mode not in self.global_config.ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode={execution_mode!r}. "
                f"Expected one of {sorted(self.global_config.ALLOWED_EXECUTION_MODES)}."
            )
        self.active_execution_mode = execution_mode

    def _trace_schedule(self, seqs: list[Sequence], is_prefill: bool, runner_role: str):
        iteration_id, batch_id = self.scheduler.next_batch_id(runner_role)
        for seq in seqs:
            seq.mark_scheduled(iteration_id, batch_id, is_prefill, runner_role)
        per_seq_zeros = {seq.seq_id: 0 for seq in seqs}
        record = {
            "trace_type": "prefill" if is_prefill else "decode_iteration",
            "record_level": "runner_substep",
            "execution_mode": self.active_execution_mode,
            "decode_ready_mode": self.active_decode_ready_mode,
            "decode_iteration_group": self._decode_iteration_group,
            "iteration_id": iteration_id,
            "batch_id": batch_id,
            "runner_role": runner_role,
            "scheduled_seq_ids": [seq.seq_id for seq in seqs],
            "request_ids": [seq.request_id for seq in seqs],
            "num_seqs_in_batch": len(seqs),
            "is_prefill": is_prefill,
            "draft_start_ts": None,
            "draft_end_ts": None,
            "verify_start_ts": None,
            "verify_end_ts": None,
            "total_iteration_start_ts": None,
            "total_iteration_end_ts": None,
            "draft_time_ms": 0.0,
            "verify_time_ms": 0.0,
            "total_iteration_time_ms": 0.0,
            "per_seq_accepted_len": dict(per_seq_zeros),
            "accepted_tokens_per_seq": dict(per_seq_zeros),
            "per_seq_invalidated_predraft_len": dict(per_seq_zeros),
            "total_accepted_tokens": 0,
            "drafted_tokens_total": 0,
            "verified_tokens_total": 0,
            "rejected_tokens_by_request": {},
            "slo_class_by_request": {
                seq.request_id: getattr(seq, "slo_class", None)
                for seq in seqs
            },
            "slo_tpot_ms_by_request": {
                seq.request_id: getattr(seq, "slo_tpot_ms", None)
                for seq in seqs
            },
        }
        self.trace_records.append(record)
        return record

    def _mark_trace_start(self, record: dict):
        if self.tp_params.local_rank != 0:
            return
        now = time.time()
        key = "draft_start_ts" if self.is_draft else "verify_start_ts"
        record[key] = now
        if record["total_iteration_start_ts"] is None:
            record["total_iteration_start_ts"] = now

    def _update_trace_token_stats(self, record: dict, accepted_lens: dict[int, int] | None = None, invalidated_lens: dict[int, int] | None = None):
        if accepted_lens:
            accepted_lens = {seq_id: int(accepted_len) for seq_id, accepted_len in accepted_lens.items()}
            record["per_seq_accepted_len"].update(accepted_lens)
            record["accepted_tokens_per_seq"].update(accepted_lens)
            record["total_accepted_tokens"] = sum(record["accepted_tokens_per_seq"].values())
            # Compute rejected tokens: for PEARL, gamma tokens were drafted per seq;
            # rejected = gamma - accepted for each seq that was verified.
            gamma = getattr(self, "gamma", 4)
            rejected = {}
            seq_to_req = dict(zip(record["scheduled_seq_ids"], record.get("request_ids", [])))
            for seq_id in accepted_lens:
                if seq_id in record["scheduled_seq_ids"]:
                    req_id = seq_to_req.get(seq_id, seq_id)
                    rejected[req_id] = max(gamma - accepted_lens[seq_id], 0)
            if rejected:
                record["rejected_tokens_by_request"].update(rejected)
        if invalidated_lens:
            invalidated_lens = {seq_id: int(invalidated_len) for seq_id, invalidated_len in invalidated_lens.items()}
            record["per_seq_invalidated_predraft_len"].update(invalidated_lens)

    def _mark_trace_end(self, record: dict, accepted_lens: dict[int, int] | None = None, invalidated_lens: dict[int, int] | None = None, drafted_tokens: int = 0, verified_tokens: int = 0):
        if self.tp_params.local_rank == 0:
            now = time.time()
            key = "draft_end_ts" if self.is_draft else "verify_end_ts"
            record[key] = now
            record["total_iteration_end_ts"] = now
            if record["draft_start_ts"] is not None and record["draft_end_ts"] is not None:
                record["draft_time_ms"] = (record["draft_end_ts"] - record["draft_start_ts"]) * 1000
            if record["verify_start_ts"] is not None and record["verify_end_ts"] is not None:
                record["verify_time_ms"] = (record["verify_end_ts"] - record["verify_start_ts"]) * 1000
            if record["total_iteration_start_ts"] is not None:
                record["total_iteration_time_ms"] = (now - record["total_iteration_start_ts"]) * 1000
        if drafted_tokens:
            record["drafted_tokens_total"] = record.get("drafted_tokens_total", 0) + drafted_tokens
        if verified_tokens:
            record["verified_tokens_total"] = record.get("verified_tokens_total", 0) + verified_tokens
        self._update_trace_token_stats(record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def _service_metadata(self):
        seqs = list(self.scheduler.waiting) + list(self.scheduler.running) + list(self.scheduler.finished)
        return [seq.service_metadata() for seq in seqs]

    def _mark_decode_ready(self):
        ts = time.time()
        for seq in self.scheduler.running:
            seq.mark_decode_ready(ts)

    def _mark_decode_started(self):
        ts = time.time()
        for seq in self.scheduler.running:
            seq.mark_decode_started(ts)

    def _write_generation_result(self, output, elapsed_time):
        data = pickle.dumps([output, elapsed_time, self.trace_records, self._service_metadata()])
        n = len(data)
        shm_capacity = self.shm.size
        total_output_tokens = sum(len(tokens) for _, tokens, _ in output) if output else 0
        self.last_result_used_file_fallback = False
        if self.tp_params.local_rank == 0:
            logger.info(
                f"[Rank {self.rank}: {self.group_name}] result payload bytes={n}, shm bytes={shm_capacity}, "
                f"num_output_reqs={len(output)}, total_output_tokens={total_output_tokens}",
                color="yellow",
            )
        if n + 4 <= shm_capacity:
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4:n+4] = data
            return
        fd, path = tempfile.mkstemp(prefix=f"pearl_result_{self.group_name}_{self.rank}_", suffix=".pkl")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(data)
        ctrl = pickle.dumps(["__PAYLOAD_FILE__", path])
        ctrl_n = len(ctrl)
        if ctrl_n + 4 > shm_capacity:
            raise RuntimeError(
                f"Control payload does not fit shared memory: ctrl={ctrl_n}, shm={shm_capacity}"
            )
        if self.tp_params.local_rank == 0:
            logger.warning(
                f"[Rank {self.rank}: {self.group_name}] payload exceeds shm; using file fallback: {path}",
            )
        self.last_result_used_file_fallback = True
        self.shm.buf[0:4] = ctrl_n.to_bytes(4, "little")
        self.shm.buf[4:ctrl_n+4] = ctrl

    def prefill(self):
        seqs, is_prefill = self.scheduler.schedule()
        trace_record = self._trace_schedule(seqs, is_prefill, f"{self._runner_role()}_prefill")
        assert is_prefill, "wrong match. current stage is decode."
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, True)
        sample_tokens = self.sampler(logits, temperatures) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
        torch.cuda.synchronize()
        token_ids = sample_tokens.tolist()
        reset_context(self.tp_params)
        self.scheduler.postprocess(seqs, token_ids)
        accepted_lens = {seq.seq_id: 1 for seq in seqs}
        for seq in seqs:
            seq.record_accepted(1)
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens)

    def step(self):
        self._decode_iteration_group += 1
        seqs, is_prefill = self.scheduler.schedule()
        trace_record = self._trace_schedule(seqs, is_prefill, self._runner_role())
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_tokens = self.sampler(logits, temperatures) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
        torch.cuda.synchronize()
        token_ids = sample_tokens.tolist()
        reset_context(self.tp_params)
        self.scheduler.postprocess(seqs, token_ids)
        accepted_lens = {seq.seq_id: 1 for seq in seqs}
        for seq in seqs:
            seq.record_accepted(1)
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
    
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.global_config.max_num_batched_tokens, self.global_config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.global_config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        input_ids, positions = self.prepare_prefill(seqs)
        logits = self.run_model(input_ids, positions, True)
        torch.cuda.empty_cache()
        dist.barrier()
        if self.tp_params.local_rank == 0:
            logger.info(f"[Rank {self.rank}: {self.group_name}] Num seqs: {num_seqs} Warmup finished.", color="green")

    def auto_set_gamma(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # TODO: will support to customize these parameters in the future.
        PROFILE_STEPS = 30
        SKIP_FIRST_STEPS = 5
        bs = [1, 2, 4, 8, 16, 32]
        MAX_SEQ_LEN = 256 # should be set to lower value if the memory is not enough.
        speed = torch.zeros(len(bs), dtype=torch.float32, device="cuda")
        for idx in trange(len(bs), desc="Auto Set Gamma", disable=self.rank != 0):
            seqs = [Sequence([0] * MAX_SEQ_LEN) for _ in range(bs[idx])]
            bs_speed = []
            for seq in seqs:
                self.add_request(seq)
            dist.barrier()
            for _ in range(PROFILE_STEPS):
                torch.cuda.synchronize()
                start_time = time.time()
                outputs, num_tokens = self.step()
                torch.cuda.synchronize()
                end_time = time.time()
                bs_speed.append(1 / (end_time - start_time))
            bs_speed = bs_speed[SKIP_FIRST_STEPS:]
            speed[idx] = sum(bs_speed) / len(bs_speed)
            self.clear_requests()
        
        global_speed = torch.zeros((self.global_config.world_size, len(bs)), dtype=torch.float32, device="cuda")
        global_speed[self.rank] = speed
        dist.all_reduce(global_speed, op=dist.ReduceOp.SUM)

        split_rank = self.global_config.draft_config.tensor_parallel_size
        draft_speed = global_speed[:split_rank].mean(dim=0)
        target_speed = global_speed[split_rank:].mean(dim=0)
        gamma_list = torch.round(draft_speed / target_speed).long().tolist()
        self.gamma_list = {b: g for b, g in zip(bs, gamma_list)}
        if self.rank == 0:
            for idx, b in enumerate(bs):
                logger.info(f"batch size: {b}, draft speed: {draft_speed[idx].item():.2f} tok/s, target speed: {target_speed[idx].item():.2f} tok/s, gamma: {self.gamma_list[b]}")

        reset_context(self.tp_params)
        torch.cuda.empty_cache()

    def clear_requests(self):
        self.scheduler.clear()
        self.trace_records.clear()
        self.active_decode_ready_mode = False
        dist.barrier()

    def prepare_decode_ready(self):
        """Materialize prompt KV before a decoder-only benchmark measurement.

        This in-memory helper intentionally does not persist KV caches. It runs
        the existing prompt prefill once, marks requests as decode-ready, and
        leaves scheduler/KV state resident for a later decode_ready_* command.
        The evaluator should start timing only after this method returns.
        """
        self.active_decode_ready_mode = True
        dist.barrier()
        self.prefill()
        self._mark_decode_ready()
        dist.barrier()

    def _finish_decode_ready_generation(self, output, elapsed_time):
        if self.tp_params.local_rank == 0:
            self._write_generation_result(output, elapsed_time)
        self.clear_requests()

    def decode_ready_parallel_generate(self):
        """Decode-only AR generation after prepare_decode_ready()."""
        self._set_execution_mode("ar")
        self.active_decode_ready_mode = True
        dist.barrier()
        self._mark_decode_started()
        torch.cuda.synchronize()
        start_time = time.time()
        while not self.scheduler.is_finished():
            outputs, num_tokens = self.step()
        torch.cuda.synchronize()
        end_time = time.time()
        dist.barrier()

        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        self._finish_decode_ready_generation(output, end_time - start_time)

    def decode_ready_pearl_generate(self):
        """Decode-only parallel PEARL after prepare_decode_ready()."""
        self._set_execution_mode("parallel_pearl")
        self.active_decode_ready_mode = True
        dist.barrier()
        self._mark_decode_started()
        torch.cuda.synchronize()
        start_time = time.time()
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        while not self.scheduler.is_finished():
            self.pearl_step()
        torch.cuda.synchronize()
        end_time = time.time()

        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        self._finish_decode_ready_generation(output, end_time - start_time)

    def _cached_decode_ready_generate_loop(self, execution_mode: str, decode_step_fn, max_active_cached_seqs: int = 0):
        self._set_execution_mode(execution_mode)
        self.active_decode_ready_mode = True
        if max_active_cached_seqs <= 0:
            max_active_cached_seqs = self.scheduler.max_num_seqs
        pending = sorted(list(self.scheduler.pending_cached), key=lambda s: s.arrival_ts)
        self.scheduler.pending_cached = deque()
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        serving_start_tensor = torch.tensor(
            [start_time if self.rank == 0 else 0.0],
            dtype=torch.float64,
            device="cuda",
        )
        dist.broadcast(serving_start_tensor, src=0)
        serving_start_ts = float(serving_start_tensor.item())
        base_offset = min([float(getattr(s, "arrival_offset_sec", 0.0) or 0.0) for s in pending], default=0.0)
        materialized_count = 0
        min_free_blocks = len(self.scheduler.block_manager.free_block_ids)
        last_logged_materialized_bucket = -1
        last_logged_pending_bucket = -1
        last_logged_running = -1
        while pending or self.scheduler.running:
            now_tensor = torch.tensor(
                [time.time() if self.rank == 0 else 0.0],
                dtype=torch.float64,
                device="cuda",
            )
            dist.broadcast(now_tensor, src=0)
            now = float(now_tensor.item())
            free_blocks_before = len(self.scheduler.block_manager.free_block_ids)
            gpu_free_before, gpu_total = torch.cuda.mem_get_info()
            guard_triggered = False
            eligible = 0
            while eligible < len(pending):
                seq = pending[eligible]
                seq_arrival = serving_start_ts + (float(getattr(seq, "arrival_offset_sec", 0.0) or 0.0) - base_offset)
                if seq_arrival <= now:
                    eligible += 1
                else:
                    break
            local_active_capacity = max(max_active_cached_seqs - len(self.scheduler.running), 0)
            if eligible > 0 and local_active_capacity > 0:
                first_need_blocks = int(self.cached_kv_store[pending[0].request_id]["num_blocks"])
                local_block_capacity = len(self.scheduler.block_manager.free_block_ids) // max(first_need_blocks, 1)
            else:
                local_block_capacity = 0
            gpu_free_now, _ = torch.cuda.mem_get_info()
            local_mem_capacity = local_active_capacity if gpu_free_now >= 256 * 1024 * 1024 else 0
            local_k = min(eligible, local_active_capacity, local_block_capacity, local_mem_capacity)
            k_tensor = torch.tensor([local_k], dtype=torch.int64, device="cuda")
            dist.all_reduce(k_tensor, op=dist.ReduceOp.MIN)
            global_k = int(k_tensor.item())
            if global_k == 0 and eligible > 0 and local_active_capacity > 0:
                guard_triggered = True
            for _ in range(global_k):
                seq = pending.pop(0)
                self.materialize_cached_request(seq.request_id, now)
                materialized_count += 1
            free_blocks_after = len(self.scheduler.block_manager.free_block_ids)
            min_free_blocks = min(min_free_blocks, free_blocks_after)
            running_count = len(self.scheduler.running)
            pending_count = len(pending)
            sync_vec = torch.tensor([materialized_count, pending_count, running_count], dtype=torch.int64, device="cuda")
            gathered = [torch.zeros_like(sync_vec) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, sync_vec)
            assert all(torch.equal(g, gathered[0]) for g in gathered), (
                "cached admission divergence across ranks: "
                + ", ".join(str(g.tolist()) for g in gathered)
            )
            mat_bucket = materialized_count // self.cached_admission_log_interval
            pending_bucket = pending_count // self.cached_admission_log_interval
            should_log = (
                guard_triggered
                or mat_bucket != last_logged_materialized_bucket
                or pending_bucket != last_logged_pending_bucket
                or running_count != last_logged_running
            )
            if self.tp_params.local_rank == 0 and should_log:
                logger.info(
                    f"[Rank {self.rank}: {self.group_name}] cached loop: max_active={max_active_cached_seqs}, "
                    f"running={running_count}, materialized={materialized_count}, pending={pending_count}, "
                    f"free_blocks_before={free_blocks_before}, free_blocks_after={free_blocks_after}, "
                    f"gpu_free_before={gpu_free_before}, gpu_total={gpu_total}",
                    color="yellow",
                )
            last_logged_materialized_bucket = mat_bucket
            last_logged_pending_bucket = pending_bucket
            last_logged_running = running_count
            if self.scheduler.running:
                if self.gamma == -1:
                    self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
                for seq in self.scheduler.running:
                    seq.mark_decode_started()
                decode_step_fn()
            elif pending:
                next_arrival = serving_start_ts + (float(getattr(pending[0], "arrival_offset_sec", 0.0) or 0.0) - base_offset)
                time.sleep(min(max(next_arrival - now, 0.0), 0.01))
        torch.cuda.synchronize()
        end_time = time.time()
        seqs = self.scheduler.finished
        if self.tp_params.local_rank == 0:
            logger.info(
                f"[Rank {self.rank}: {self.group_name}] cached final summary: "
                f"materialized={materialized_count}, finished={len(seqs)}, elapsed_s={end_time - start_time:.4f}, "
                f"min_free_blocks={min_free_blocks}, file_fallback_used={self.last_result_used_file_fallback}",
                color="green",
            )
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        self._finish_decode_ready_generation(output, end_time - start_time)

    def cached_decode_ready_pearl_generate(self, max_active_cached_seqs: int = 0):
        self._cached_decode_ready_generate_loop("parallel_pearl", self.pearl_step, max_active_cached_seqs)

    def cached_decode_ready_serialized_pearl_generate(self, max_active_cached_seqs: int = 0):
        self._cached_decode_ready_generate_loop("serialized_pearl", self.serialized_pearl_step, max_active_cached_seqs)

    def ar_step(self):
        """Single AR decode step for cached admission.

        Uses the normal scheduler decode path: schedule() calls
        block_manager.may_append(seq) to allocate KV-cache blocks BEFORE
        prepare_decode builds slot_mapping. Without schedule(), the
        block_table may not cover the next token position, causing CUDA
        illegal memory access in run_model.

        Target does the full AR decode. Draft calls schedule() for block
        consistency, then receives seq_ids + token_ids from the target
        master via global broadcast so both sides stay in sync.
        """
        self._decode_iteration_group += 1

        if not self.is_draft:
            # ── Target side: full AR decode via scheduler ──
            free_before = len(self.scheduler.block_manager.free_block_ids)
            running_before = len(self.scheduler.running)
            seqs, is_prefill = self.scheduler.schedule()
            assert not is_prefill, "cached AR decode must not trigger prefill"
            scheduled_ids = [s.seq_id for s in seqs]

            trace_record = self._trace_schedule(seqs, is_prefill, self._runner_role())
            input_ids, positions = self.prepare_decode(seqs)

            # Defensive assertions: block_table must cover current seq length.
            assert len(input_ids) == len(positions), (
                f"ar_step target: input_ids={len(input_ids)} != positions={len(positions)}"
            )
            for seq in seqs:
                assert len(seq.block_table) >= seq.num_blocks, (
                    f"ar_step target: seq {seq.seq_id} req={seq.request_id} "
                    f"block_table_len={len(seq.block_table)} < num_blocks={seq.num_blocks}"
                )

            temperatures = self.prepare_sample(seqs) if self.tp_params.local_rank == 0 else None
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, False)
            sample_tokens = self.sampler(logits, temperatures) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            if self.tp_params.local_rank == 0:
                logger.info(
                    f"[Rank {self.rank}: {self.group_name}] cached AR step: "
                    f"execution_mode=ar scheduled={len(seqs)} "
                    f"running_before={running_before} free_before={free_before} "
                    f"scheduled_seq_ids={scheduled_ids}",
                    color="cyan",
                )

            self.scheduler.postprocess(seqs, token_ids)
            accepted_lens = {seq.seq_id: 1 for seq in seqs}
            for seq in seqs:
                seq.record_accepted(1)
            self._mark_trace_end(trace_record, accepted_lens=accepted_lens)

            if self.tp_params.local_rank == 0:
                finished_ids = [s.seq_id for s in self.scheduler.finished
                                if s.seq_id in scheduled_ids]
                logger.info(
                    f"[Rank {self.rank}: {self.group_name}] cached AR postprocess: "
                    f"running_after={len(self.scheduler.running)} "
                    f"finished_total={len(self.scheduler.finished)} "
                    f"finished_this_step={finished_ids} "
                    f"free_after={len(self.scheduler.block_manager.free_block_ids)}",
                    color="cyan",
                )

            # Broadcast seq count + seq_ids + token_ids for draft sync.
            n_seqs = torch.tensor([len(seqs)], dtype=torch.int64, device="cuda")
            dist.broadcast(n_seqs, src=self.global_config.target_config.master_rank)
            seq_ids_t = torch.tensor(scheduled_ids, dtype=torch.int64, device="cuda")
            dist.broadcast(seq_ids_t, src=self.global_config.target_config.master_rank)
            dist.broadcast(sample_tokens, src=self.global_config.target_config.master_rank)
        else:
            # ── Draft side: schedule for block consistency, then sync ──
            draft_seqs, is_prefill = self.scheduler.schedule()
            assert not is_prefill, "cached AR decode must not trigger prefill"

            # Receive seq count, seq_ids, and token_ids from target master.
            n_seqs = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.broadcast(n_seqs, src=self.global_config.target_config.master_rank)
            num_scheduled = int(n_seqs.item())

            seq_ids_t = torch.zeros(num_scheduled, dtype=torch.int64, device="cuda")
            sample_tokens = torch.zeros(num_scheduled, dtype=torch.int64, device="cuda")
            dist.broadcast(seq_ids_t, src=self.global_config.target_config.master_rank)
            dist.broadcast(sample_tokens, src=self.global_config.target_config.master_rank)
            target_seq_ids = seq_ids_t.tolist()
            token_ids = sample_tokens.tolist()

            # Apply postprocess in target-determined order.
            seq_by_id = {s.seq_id: s for s in draft_seqs}
            ordered_seqs = [seq_by_id[sid] for sid in target_seq_ids]
            self.scheduler.postprocess(ordered_seqs, token_ids)
            for seq in ordered_seqs:
                seq.record_accepted(1)

    def cached_decode_ready_ar_generate(self, max_active_cached_seqs: int = 0):
        self._cached_decode_ready_generate_loop("ar", self.ar_step, max_active_cached_seqs)

    def decode_ready_serialized_pearl_generate(self):
        """Decode-only serialized-PEARL approximation after prepare_decode_ready()."""
        self._set_execution_mode("serialized_pearl")
        self.active_decode_ready_mode = True
        dist.barrier()
        self._mark_decode_started()
        torch.cuda.synchronize()
        start_time = time.time()
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        while not self.scheduler.is_finished():
            self.serialized_pearl_step()
        torch.cuda.synchronize()
        end_time = time.time()

        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        self._finish_decode_ready_generation(output, end_time - start_time)

    def parallel_generate(self):
        self._set_execution_mode("ar")
        dist.barrier()

        torch.cuda.synchronize()
        start_time = time.time()
        while not self.scheduler.is_finished():
            outputs, num_tokens = self.step()
        torch.cuda.synchronize()
        end_time = time.time()
        dist.barrier()

        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        if self.tp_params.local_rank == 0:
            self._write_generation_result(output, end_time - start_time)
        
        self.clear_requests()

    def pearl_generate(self):
        self._set_execution_mode("parallel_pearl")
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()

        # determine the gamma for each batch size
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]

        while not self.scheduler.is_finished():
            self.pearl_step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]

        if self.tp_params.local_rank == 0:
            self._write_generation_result(output, end_time - start_time)
            
        self.clear_requests()

    def pearl_bench_generate(self, num_pearl_steps: int = 100):
        """Benchmark the real-world throughput of the PEARL algorithm.

        For speculative decoding, either setting the max tokens or ignore eos
        tokens is not fair! As there always exists some seqs that have higher
        MAT and early finished. Therefore, we must set a fixed PEARL steps to
        ensure all the sequences are running at any time.
        """
        self._set_execution_mode("parallel_pearl")
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()

        # set max tokens to a large value to ensure all the sequences are running at any time
        for seq in self.scheduler.running:
            seq.max_tokens = 1e8
            seq.ignore_eos = True
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]

        for _ in range(num_pearl_steps):
            self.pearl_step()

        torch.cuda.synchronize()
        end_time = time.time()
        seqs = self.scheduler.running
        
        for seq in seqs:
            # acc tokens are not properly appended in the pearl_step function, so we append it here.
            seq.num_acc_tokens.append(seq.cur_acc_tokens)

        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]

        if self.tp_params.local_rank == 0:
            self._write_generation_result(output, end_time - start_time)
            
        self.clear_requests()


    def serialized_pearl_generate(self):
        """Run a serialized-PEARL approximation baseline.

        This is intentionally *not* a strict vanilla serial speculative decoding
        implementation. It reuses PEARL's existing verification semantics but
        adds runner-side synchronization so draft and target verification compute
        do not overlap. A larger refactor would be required for textbook serial
        speculative decoding.
        """
        self._set_execution_mode("serialized_pearl")
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()

        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]

        while not self.scheduler.is_finished():
            self.serialized_pearl_step()

        torch.cuda.synchronize()
        end_time = time.time()
        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]

        if self.tp_params.local_rank == 0:
            self._write_generation_result(output, end_time - start_time)

        self.clear_requests()

    def serialized_pearl_bench_generate(self, num_pearl_steps: int = 100):
        """Benchmark serialized-PEARL with fixed PEARL steps.

        This is an approximation baseline that disables draft/verify overlap in
        the current PEARL pipeline; it is not a strict vanilla serial
        speculative decoding implementation.
        """
        self._set_execution_mode("serialized_pearl")
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()

        for seq in self.scheduler.running:
            seq.max_tokens = 1e8
            seq.ignore_eos = True
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]

        for _ in range(num_pearl_steps):
            self.serialized_pearl_step()

        torch.cuda.synchronize()
        end_time = time.time()
        seqs = self.scheduler.running

        for seq in seqs:
            seq.num_acc_tokens.append(seq.cur_acc_tokens)

        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]

        if self.tp_params.local_rank == 0:
            self._write_generation_result(output, end_time - start_time)

        self.clear_requests()

    def _full_gamma_postprocess(self, seqs: list[Sequence], verify_res: torch.Tensor):
        """Full-gamma speculative decoding postprocess — shared by draft and target.

        Full-gamma semantics: accepted_len ∈ [0, gamma]. No pre_verify state.
        On rejection, rollback the unaccepted draft suffix and append the
        target correction token. Both sides must end the iteration with
        consistent seq.token_ids lengths.
        """
        acc, rollout, revise_token, finish = verify_res.tolist()
        accepted_lens: dict[int, int] = {}
        invalidated_lens: dict[int, int] = {}

        for idx, seq in enumerate(seqs):
            accepted_len = self.gamma if acc[idx] else self.gamma - rollout[idx]
            invalidated_len = 0 if acc[idx] else rollout[idx]
            accepted_lens[seq.seq_id] = accepted_len
            invalidated_lens[seq.seq_id] = invalidated_len
            seq.record_accepted(accepted_len)
            seq.record_invalidated_predraft(invalidated_len)

            if finish[idx]:
                seq.mark_finished()
                seq.num_acc_tokens.append(seq.cur_acc_tokens)
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)
                continue

            if not acc[idx]:
                # Rejection: rollback unaccepted draft suffix + correction token.
                unaccepted = self.gamma - accepted_len
                self.scheduler.rollback(seq, unaccepted)
                seq.append_token(revise_token[idx])
                self._ensure_block_table_covers_sequence(seq)

        return accepted_lens, invalidated_lens

    def _ensure_block_table_covers_sequence(self, seq: Sequence):
        """Allocate additional KV-cache blocks so block_table covers seq.num_blocks.

        Must be called after any serialized_pearl-specific append that occurs
        outside of scheduler.schedule() (which normally calls may_append).
        """
        missing = seq.num_blocks - len(seq.block_table)
        assert missing >= 0, (
            f"_ensure_block_table_covers_sequence: seq {seq.seq_id} "
            f"num_blocks={seq.num_blocks} block_table_len={len(seq.block_table)} "
            f"missing={missing}"
        )
        while len(seq.block_table) < seq.num_blocks:
            self.scheduler.block_manager.may_append(seq)
        assert len(seq.block_table) >= seq.num_blocks, (
            f"_ensure_block_table_covers_sequence: failed to allocate blocks "
            f"for seq {seq.seq_id} (need {seq.num_blocks}, have {len(seq.block_table)})"
        )

    @abstractmethod
    def pearl_step(self):
        pass

    @abstractmethod
    def serialized_pearl_step(self):
        pass


class DraftModelRunner(ModelRunnerBase):
    def __init__(self, config: PEARLConfig, rank: int, event: Event, control_event: Event):
        super().__init__(config, rank, event, control_event)

    def prepare_pearl_decode(self, seqs: list[Sequence]):
        return super().prepare_decode(seqs)

    def send_serialized_draft_window(self, seqs: list[Sequence]):
        """Broadcast the current iteration's gamma-token draft window to verify_group.

        Only the draft master sends. All target devices (members of verify_group)
        receive via recv_serialized_draft_window.
        """
        if self.tp_params.local_rank == 0:
            flat = []
            for seq in seqs:
                toks = seq.token_ids[-self.gamma:]
                assert len(toks) == self.gamma, (
                    f"send_serialized_draft_window: seq {seq.seq_id} "
                    f"has {len(toks)} tokens in draft window, expected {self.gamma}"
                )
                flat.extend(toks)
            msg = torch.tensor(flat, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.rank, group=self.verify_group)

    def send_parallel_draft_window(self, verify_seqs: list[Sequence]):
        """Broadcast verify windows to target using two-phase protocol.

        Phase 1: count (number of verify_seqs).
        Phase 2: seq_ids (for ordering check on target side).
        Phase 3: gamma draft tokens per verify_seq (seq.token_ids[-gamma:]).

        Asserts every verify_seq has pending_draft_tokens == gamma and the
        window tokens are exactly gamma long.
        """
        if self.tp_params.local_rank == 0:
            # Phase 1: count
            count_t = torch.tensor([len(verify_seqs)], dtype=torch.int64, device="cuda")
            dist.broadcast(count_t, src=self.rank, group=self.verify_group)

            if verify_seqs:
                # Phase 2: seq_ids
                ids_t = torch.tensor([s.seq_id for s in verify_seqs], dtype=torch.int64, device="cuda")
                dist.broadcast(ids_t, src=self.rank, group=self.verify_group)
                # Phase 3: draft tokens
                flat = []
                for s in verify_seqs:
                    assert s.pending_draft_tokens == self.gamma, (
                        f"send_parallel_draft_window: seq {s.seq_id} "
                        f"pending_draft_tokens={s.pending_draft_tokens}, expected gamma={self.gamma}"
                    )
                    toks = s.token_ids[-self.gamma:]
                    assert len(toks) == self.gamma, (
                        f"send_parallel_draft_window: seq {s.seq_id} "
                        f"has {len(toks)} tokens in draft window, expected {self.gamma}"
                    )
                    flat.extend(toks)
                msg_t = torch.tensor(flat, dtype=torch.int64, device="cuda")
                dist.broadcast(msg_t, src=self.rank, group=self.verify_group)

    def _parallel_postprocess(self, verify_seqs: list[Sequence], verify_res: torch.Tensor,
                              drafted_this_iter: dict[int, int]):
        """Draft-side full-gamma postprocess with pending-window awareness.

        For each verified seq:
        - Applies accepted_len from W_{k-1} verification.
        - On full accept: W_k stays as pending (pending_draft_tokens = gamma).
        - On rejection: rollback rejected suffix of W_{k-1} + all of W_k,
          append correction token (pending_draft_tokens = 0).
        - On finish: W_k is invalidated (pending_draft_tokens = 0).

        invalidated_predraft_tokens = gamma when W_k is discarded (rejection
        or finish), else 0. rejected_tokens (suffix of W_{k-1}) is tracked
        separately in trace via _update_trace_token_stats.
        """
        acc, rollout, revise_token, finish = verify_res.tolist()
        accepted_lens: dict[int, int] = {}
        invalidated_lens: dict[int, int] = {}

        for idx, seq in enumerate(verify_seqs):
            accepted_len = self.gamma if acc[idx] else self.gamma - rollout[idx]
            accepted_lens[seq.seq_id] = accepted_len
            seq.record_accepted(accepted_len)

            w_k_generated = drafted_this_iter.get(seq.seq_id, 0)
            assert w_k_generated == self.gamma, (
                f"_parallel_postprocess: seq {seq.seq_id} "
                f"drafted_this_iter={w_k_generated}, expected gamma={self.gamma}"
            )

            if finish[idx]:
                seq.mark_finished()
                seq.num_acc_tokens.append(seq.cur_acc_tokens)
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)
                # W_k was generated but request finished: invalidate W_k
                invalidated_predraft = self.gamma
                seq.pending_draft_tokens = 0
                seq.record_invalidated_predraft(invalidated_predraft)
                invalidated_lens[seq.seq_id] = invalidated_predraft
                continue

            if acc[idx]:
                # W_{k-1} fully accepted. W_k stays as pending.
                invalidated_predraft = 0
                seq.pending_draft_tokens = self.gamma
            else:
                # W_{k-1} rejected at position accepted_len.
                # Rollback: rejected suffix of W_{k-1} + all of W_k
                rejected_tokens = self.gamma - accepted_len
                total_rollback = rejected_tokens + self.gamma
                self.scheduler.rollback(seq, total_rollback)
                seq.append_token(revise_token[idx])
                self._ensure_block_table_covers_sequence(seq)
                invalidated_predraft = self.gamma
                seq.pending_draft_tokens = 0
                # Rejection ≠ finish. Do NOT call seq.mark_finished().

            seq.record_invalidated_predraft(invalidated_predraft)
            invalidated_lens[seq.seq_id] = invalidated_predraft

        return accepted_lens, invalidated_lens

    def _first_iteration_sync_verify_and_build_pending(self, all_seqs: list[Sequence],
                                                        trace_record: dict,
                                                        drafted_this_iter: dict[int, int]):
        """First-iteration handler: verify W₀ synchronously, then build W₁ as pending.

        W₀ was already generated on all_seqs (tracked in drafted_this_iter).
        After postprocessing W₀, resets pending_draft_tokens=0, then generates
        W₁ for ALL non-finished seqs so they have pending windows for the next
        parallel iteration.

        Returns (accepted_lens, invalidated_lens) for the caller to record in trace.
        """
        # Send W₀ for all seqs
        self.send_parallel_draft_window(all_seqs)

        # Receive verify_res
        verify_res = torch.zeros((4, len(all_seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Postprocess W₀ (synchronous, no W_k to invalidate → target-side semantics)
        accepted_lens, invalidated_lens = self._full_gamma_postprocess(all_seqs, verify_res)

        # Reset pending_draft_tokens: W₀ was verified, no longer pending
        for s in all_seqs:
            s.pending_draft_tokens = 0

        # Build W₁ as pending for ALL non-finished seqs
        non_finished = [s for s in all_seqs if not s.is_finished]
        if non_finished:
            for _ in range(self.gamma):
                seqs, is_prefill = self.scheduler.schedule()
                assert not is_prefill, "wrong match. current stage is prefill."
                input_ids, positions = self.prepare_pearl_decode(seqs)
                torch.cuda.synchronize()
                logits = self.run_model(input_ids, positions, is_prefill)
                sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
                dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
                torch.cuda.synchronize()
                token_ids = sample_tokens.tolist()
                reset_context(self.tp_params)
                for seq, token_id in zip(seqs, token_ids):
                    seq.append_token(token_id)
                    seq.pending_draft_tokens += 1

        # Assert invariant: pending_draft_tokens ∈ {0, gamma} for all running
        for s in self.scheduler.running:
            assert s.pending_draft_tokens in (0, self.gamma), (
                f"_first_iteration: seq {s.seq_id} pending_draft_tokens={s.pending_draft_tokens}, "
                f"expected 0 or {self.gamma}"
            )

        return accepted_lens, invalidated_lens

    def _validate_full_gamma_pearl_step(self):
        """Synchronous full-gamma PEARL step for Phase 1 validation.

        Reuses serialized full-gamma protocol. Keeps parallel_pearl execution
        mode for traces. NOT the performance path.
        """
        self._decode_iteration_group += 1
        trace_record = None
        for _ in range(self.gamma):
            seqs, is_prefill = self.scheduler.schedule()
            trace_record = self._trace_schedule(seqs, is_prefill, "draft")
            assert not is_prefill, "wrong match. current stage is prefill."
            input_ids, positions = self.prepare_pearl_decode(seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, is_prefill)
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
            self._mark_trace_end(trace_record, drafted_tokens=len(seqs))

        dist.barrier()
        self.send_serialized_draft_window(seqs)

        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        accepted_lens, invalidated_lens = self._full_gamma_postprocess(seqs, verify_res)
        if trace_record is not None:
            self._update_trace_token_stats(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def pearl_step(self):
        self._decode_iteration_group += 1

        # Phase 1 toggle: synchronous full-gamma validation mode
        if os.environ.get("PEARL_FULL_GAMMA_VALIDATE"):
            return self._validate_full_gamma_pearl_step()

        trace_record = None
        all_seqs = list(self.scheduler.running)

        # Partition: verify_seqs have full pending windows
        verify_seqs = [s for s in all_seqs if s.pending_draft_tokens >= self.gamma]
        draft_seqs = all_seqs

        # Assert: every verify_seq has exactly gamma pending tokens
        for s in verify_seqs:
            assert s.pending_draft_tokens == self.gamma, (
                f"pearl_step: seq {s.seq_id} pending_draft_tokens={s.pending_draft_tokens}, "
                f"expected gamma={self.gamma}"
            )

        drafted_this_iter: dict[int, int] = {}

        # Phase A: send pending windows
        if verify_seqs:
            self.send_parallel_draft_window(verify_seqs)

        # Phase B: generate gamma tokens for all draft_seqs
        for _ in range(self.gamma):
            seqs, is_prefill = self.scheduler.schedule()
            if trace_record is None:
                trace_record = self._trace_schedule(seqs, is_prefill, "draft")
            assert not is_prefill, "wrong match. current stage is prefill."
            input_ids, positions = self.prepare_pearl_decode(seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, is_prefill)
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                seq.pending_draft_tokens += 1
                drafted_this_iter[seq.seq_id] = drafted_this_iter.get(seq.seq_id, 0) + 1
            self._mark_trace_end(trace_record, drafted_tokens=len(seqs))

        # Assert: every draft_seq got exactly gamma new tokens
        for s in draft_seqs:
            if not s.is_finished:
                assert drafted_this_iter.get(s.seq_id, 0) == self.gamma, (
                    f"pearl_step: seq {s.seq_id} drafted_this_iter={drafted_this_iter.get(s.seq_id, 0)}, "
                    f"expected gamma={self.gamma}"
                )

        # Phase C: receive verify_res
        accepted_lens: dict[int, int] = {}
        invalidated_lens: dict[int, int] = {}
        if verify_seqs:
            verify_res = torch.zeros((4, len(verify_seqs)), dtype=torch.int64, device="cuda")
            dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
            accepted_lens, invalidated_lens = self._parallel_postprocess(
                verify_seqs, verify_res, drafted_this_iter
            )
        else:
            # First iteration: verify W₀ synchronously, then build W₁ as pending
            accepted_lens, invalidated_lens = self._first_iteration_sync_verify_and_build_pending(
                all_seqs, trace_record, drafted_this_iter
            )

        # Strict invariant: pending_draft_tokens ∈ {0, gamma} for all running seqs
        for s in self.scheduler.running:
            assert s.pending_draft_tokens in (0, self.gamma), (
                f"pearl_step end: seq {s.seq_id} pending_draft_tokens={s.pending_draft_tokens}, "
                f"expected 0 or {self.gamma}"
            )

        if trace_record is not None:
            self._update_trace_token_stats(trace_record,
                accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def serialized_pearl_step(self):
        """Serialized speculative decoding draft phase.

        Serial speculative decoding baseline: draft generates gamma tokens per
        active request, broadcasts the draft window to the target verify group,
        then receives verification results from the target. Draft and target
        verification are serialized (no overlap). This intentionally does NOT
        use PEARL's pre_verify one-token shortcut — every iteration verifies
        the full gamma-token draft window.
        """
        self._decode_iteration_group += 1
        trace_record = None
        for _ in range(self.gamma):
            seqs, is_prefill = self.scheduler.schedule()
            trace_record = self._trace_schedule(seqs, is_prefill, "serialized_draft")
            assert not is_prefill, "wrong match. current stage is prefill."
            input_ids, positions = self.prepare_pearl_decode(seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, is_prefill)
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
            self._mark_trace_end(trace_record, drafted_tokens=len(seqs))

        # Global barrier pairs with TargetModelRunner.serialized_pearl_step().
        dist.barrier()

        # Broadcast the gamma-token draft window to the target verify group.
        self.send_serialized_draft_window(seqs)

        # Receive verify_res from target (global broadcast).
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Apply serialized full-gamma postprocess.
        accepted_lens, invalidated_lens = self._full_gamma_postprocess(seqs, verify_res)
        if trace_record is not None:
            self._update_trace_token_stats(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    @torch.inference_mode()
    def verify(self, seqs: list[Sequence]):
        if self.tp_params.local_rank == 0:
            to_be_verified_tokens = []
            next_round_input = []
            for seq in seqs:
                if seq.pre_verify:
                    to_be_verified_tokens.append(seq.token_ids[-self.gamma])
                else:
                    to_be_verified_tokens.extend(seq.token_ids[-2*self.gamma+1:-self.gamma+1])
                next_round_input.extend(seq.token_ids[-self.gamma:])
            msg = torch.tensor(to_be_verified_tokens + next_round_input, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.rank, group=self.verify_group)
        
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
        
        # post-process the seqs according to the verify_res.
        acc, rollout, revise_token, finish = verify_res.tolist()
        accepted_lens = {}
        invalidated_lens = {}
        for idx, seq in enumerate(seqs):
            was_pre_verify = seq.pre_verify
            accepted_len = 1 if was_pre_verify and acc[idx] else 0
            if not was_pre_verify:
                accepted_len = self.gamma if acc[idx] else self.gamma - rollout[idx]
            invalidated_len = 0 if acc[idx] else rollout[idx]
            accepted_lens[seq.seq_id] = accepted_len
            invalidated_lens[seq.seq_id] = invalidated_len
            seq.record_accepted(accepted_len)
            seq.record_invalidated_predraft(invalidated_len)

            if finish[idx]:
                seq.mark_finished()
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)
                continue
            
            if seq.pre_verify:
                if acc[idx]:
                    seq.pre_verify = False
                else:
                    seq.pre_verify = True
                    self.scheduler.rollback(seq, self.gamma)
                    seq.append_token(revise_token[idx])
            else:
                if acc[idx]:
                    seq.pre_verify = False
                else:
                    seq.pre_verify = True
                    self.scheduler.rollback(seq, self.gamma)
                    if rollout[idx] > 1:
                        self.scheduler.rollback(seq, rollout[idx] - 1)
                    seq.append_token(revise_token[idx])
        return accepted_lens, invalidated_lens


class TargetModelRunner(ModelRunnerBase):
    def __init__(self, config: PEARLConfig, rank: int, event: Event, control_event: Event):
        super().__init__(config, rank, event, control_event)

    def prepare_pearl_decode(self, seqs: list[Sequence]):
        """
        Behavior of the target model pre-processing.
        For a sequence in pre-verify, the input tokens are the last token (1 token).
        For a sequence in post-verify, the input tokens are the last gamma tokens. (gamma tokens)
        To conduct efficient batching inference, we pack all the input tokens together.
        Viewing each token as an independent sample, and use slot_mapping and context_lens to instruct the attention network to use correct KV cache.
        Note that the num of input tokens is not equal to the num of seqs.
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        temp_seqs = []
        for seq in seqs:
            num_tokens = self.gamma if not seq.pre_verify else 1
            to_append_tokens = seq.token_ids[-num_tokens:]
            input_ids.extend(to_append_tokens)
            positions.extend(list(range(len(seq) - num_tokens, len(seq))))
            context_lens.extend(list(range(len(seq) - num_tokens + 1, len(seq) + 1)))
            slot_mapping.extend([seq.token_to_slot(token_index) for token_index in range(len(seq) - num_tokens, len(seq))])
            temp_seqs.extend([seq] * num_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(temp_seqs)
        set_context(self.tp_params, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions, temp_seqs

    def recv_serialized_draft_window(self, seqs: list[Sequence]):
        """Receive gamma draft tokens from draft master and append to local sequences.

        Must be called after the global barrier (so the draft has finished its
        gamma-step loop) and before prepare_serialized_verify_decode.
        """
        total_tokens = self.gamma * len(seqs)
        msg = torch.empty(total_tokens, dtype=torch.int64, device="cuda")
        dist.broadcast(msg, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        flat = msg.tolist()
        for i, seq in enumerate(seqs):
            toks = flat[i * self.gamma : (i + 1) * self.gamma]
            assert len(toks) == self.gamma, (
                f"recv_serialized_draft_window: seq {seq.seq_id} "
                f"expected {self.gamma} tokens, got {len(toks)}"
            )
            for tok in toks:
                seq.append_token(tok)
            self._ensure_block_table_covers_sequence(seq)

    def prepare_serialized_verify_decode(self, seqs: list[Sequence]):
        """Target preparation for serialized speculative decoding baseline.

        Feeds gamma input tokens per sequence: [last_confirmed, d_0, …, d_{γ-2}].
        This produces gamma logits that predict d_0 through d_{γ-1}, enabling
        full-gamma verification of all γ draft tokens.

        serialized_pearl intentionally disables PEARL's pre_verify one-token
        shortcut. Input positions start one token before the draft window so
        the model predicts every draft token.
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        temp_seqs = []
        for seq in seqs:
            num_tokens = self.gamma
            start = len(seq) - num_tokens - 1   # last confirmed token position
            end = len(seq) - 1                    # position before the last draft token
            to_append_tokens = seq.token_ids[start:end]
            assert len(to_append_tokens) == num_tokens, (
                f"prepare_serialized_verify_decode: seq {seq.seq_id} request_id={seq.request_id} "
                f"len(seq)={len(seq)} num_prompt_tokens={seq.num_prompt_tokens} "
                f"num_completion_tokens={seq.num_completion_tokens} "
                f"gamma={self.gamma} start={start} end={end} "
                f"expected {num_tokens} tokens, got {len(to_append_tokens)}"
            )
            input_ids.extend(to_append_tokens)
            positions.extend(range(start, end))
            context_lens.extend(range(start + 1, end + 1))
            assert len(seq.block_table) >= seq.num_blocks, (
                f"prepare_serialized_verify_decode: block_table too short for seq {seq.seq_id} "
                f"request_id={seq.request_id} len(seq)={len(seq)} "
                f"num_blocks={seq.num_blocks} block_table_len={len(seq.block_table)}"
            )
            slot_mapping.extend([seq.token_to_slot(i) for i in range(start, end)])
            temp_seqs.extend([seq] * num_tokens)

        n = len(input_ids)
        assert n == len(positions) == len(slot_mapping) == len(context_lens) == len(temp_seqs), (
            f"prepare_serialized_verify_decode: size mismatch "
            f"input_ids={n} positions={len(positions)} slot_mapping={len(slot_mapping)} "
            f"context_lens={len(context_lens)} temp_seqs={len(temp_seqs)}"
        )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(temp_seqs)
        set_context(self.tp_params, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions, temp_seqs

    @torch.inference_mode()
    def serialized_verify_full_gamma(self, logits: torch.Tensor, seqs: list[Sequence],
                                     temperatures: torch.Tensor):
        """Full-gamma verification for serialized speculative decoding.

        Verifies exactly gamma draft tokens per active sequence. Does NOT use
        PEARL's pre_verify state machine. accepted_len ∈ [0, gamma].

        Returns verify_res tensor [4, len(seqs)]: acc, rollout, revise_token, finish.
        The caller must broadcast verify_res to the draft group and call
        _serialized_postprocess on both sides.
        """
        # to_be_verified_tokens = gamma draft tokens per seq
        num_to_verify = self.gamma * len(seqs)
        to_be_verified_tokens = []
        for seq in seqs:
            toks = seq.token_ids[-self.gamma:]
            assert len(toks) == self.gamma, (
                f"serialized_verify_full_gamma: seq {seq.seq_id} "
                f"has {len(toks)} draft tokens, expected {self.gamma}"
            )
            to_be_verified_tokens.extend(toks)

        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")

        if self.tp_params.local_rank == 0:
            r = torch.rand(num_to_verify, device="cuda")
            target_logits = norm_logits(logits, temperatures)

            draft_t = torch.tensor(to_be_verified_tokens, dtype=torch.int64, device="cuda")
            target_prob = target_logits.gather(dim=1, index=draft_t.unsqueeze(1)).squeeze(1)
            judge = (r <= target_prob).tolist()

            logits.scatter_(1, draft_t.unsqueeze(1), -float("inf"))
            revised_tokens = self.sampler(logits, temperatures)

            acc, rollout, revise_token, finish = [], [], [], []

            for i, seq in enumerate(seqs):
                n = self.gamma
                finish_flag = False
                offset = i * self.gamma
                for j in range(self.gamma):
                    tok = to_be_verified_tokens[offset + j]
                    if not seq.ignore_eos and judge[offset + j] and is_eos(tok, self.scheduler.eos):
                        finish_flag = True
                    if not judge[offset + j]:
                        n = j
                        break
                acc.append(n == self.gamma)
                rollout.append(self.gamma - n)
                revise_token.append(revised_tokens[offset + n] if n < self.gamma else -1)
                finish.append(finish_flag or seq.num_completion_tokens >= seq.max_tokens - min(n + 1, self.gamma))

                if n == self.gamma:
                    seq.cur_acc_tokens += n
                else:
                    seq.num_acc_tokens.append(seq.cur_acc_tokens + n + 1)
                    seq.cur_acc_tokens = 0

            verify_res = torch.tensor([acc, rollout, revise_token, finish], dtype=torch.int64, device="cuda")

        return verify_res

    def prepare_parallel_verify_decode(self, seqs: list[Sequence]):
        """Target preparation for parallel full-gamma speculative decoding.

        Feeds gamma input tokens per sequence: [last_confirmed, d_0, …, d_{γ-2}].
        This produces gamma logits that predict d_0 through d_{γ-1}, enabling
        full-gamma verification of all γ draft tokens.

        Identical semantics to prepare_serialized_verify_decode. Does NOT use
        PEARL's pre_verify state machine.
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        temp_seqs = []
        for seq in seqs:
            num_tokens = self.gamma
            start = len(seq) - num_tokens - 1   # last confirmed token position
            end = len(seq) - 1                    # position before the last draft token
            to_append_tokens = seq.token_ids[start:end]
            assert len(to_append_tokens) == num_tokens, (
                f"prepare_parallel_verify_decode: seq {seq.seq_id} request_id={seq.request_id} "
                f"len(seq)={len(seq)} num_prompt_tokens={seq.num_prompt_tokens} "
                f"num_completion_tokens={seq.num_completion_tokens} "
                f"gamma={self.gamma} start={start} end={end} "
                f"expected {num_tokens} tokens, got {len(to_append_tokens)}"
            )
            input_ids.extend(to_append_tokens)
            positions.extend(range(start, end))
            context_lens.extend(range(start + 1, end + 1))
            assert len(seq.block_table) >= seq.num_blocks, (
                f"prepare_parallel_verify_decode: block_table too short for seq {seq.seq_id} "
                f"request_id={seq.request_id} len(seq)={len(seq)} "
                f"num_blocks={seq.num_blocks} block_table_len={len(seq.block_table)}"
            )
            slot_mapping.extend([seq.token_to_slot(i) for i in range(start, end)])
            temp_seqs.extend([seq] * num_tokens)

        n = len(input_ids)
        assert n == len(positions) == len(slot_mapping) == len(context_lens) == len(temp_seqs), (
            f"prepare_parallel_verify_decode: size mismatch "
            f"input_ids={n} positions={len(positions)} slot_mapping={len(slot_mapping)} "
            f"context_lens={len(context_lens)} temp_seqs={len(temp_seqs)}"
        )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(temp_seqs)
        set_context(self.tp_params, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions, temp_seqs

    @torch.inference_mode()
    def parallel_verify_full_gamma(self, logits: torch.Tensor, seqs: list[Sequence],
                                   temperatures: torch.Tensor):
        """Full-gamma verification for parallel speculative decoding.

        Verifies exactly gamma draft tokens per active sequence. Does NOT use
        PEARL's pre_verify state machine. accepted_len ∈ [0, gamma].

        Returns verify_res tensor [4, len(seqs)]: acc, rollout, revise_token, finish.
        The caller must broadcast verify_res to the draft group and call
        _full_gamma_postprocess on both sides.
        """
        num_to_verify = self.gamma * len(seqs)
        to_be_verified_tokens = []
        for seq in seqs:
            toks = seq.token_ids[-self.gamma:]
            assert len(toks) == self.gamma, (
                f"parallel_verify_full_gamma: seq {seq.seq_id} "
                f"has {len(toks)} draft tokens, expected {self.gamma}"
            )
            to_be_verified_tokens.extend(toks)

        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")

        if self.tp_params.local_rank == 0:
            r = torch.rand(num_to_verify, device="cuda")
            target_logits = norm_logits(logits, temperatures)

            draft_t = torch.tensor(to_be_verified_tokens, dtype=torch.int64, device="cuda")
            target_prob = target_logits.gather(dim=1, index=draft_t.unsqueeze(1)).squeeze(1)
            judge = (r <= target_prob).tolist()

            logits.scatter_(1, draft_t.unsqueeze(1), -float("inf"))
            revised_tokens = self.sampler(logits, temperatures)

            acc, rollout, revise_token, finish = [], [], [], []

            for i, seq in enumerate(seqs):
                n = self.gamma
                finish_flag = False
                offset = i * self.gamma
                for j in range(self.gamma):
                    tok = to_be_verified_tokens[offset + j]
                    if not seq.ignore_eos and judge[offset + j] and is_eos(tok, self.scheduler.eos):
                        finish_flag = True
                    if not judge[offset + j]:
                        n = j
                        break
                acc.append(n == self.gamma)
                rollout.append(self.gamma - n)
                revise_token.append(revised_tokens[offset + n] if n < self.gamma else -1)
                finish.append(finish_flag or seq.num_completion_tokens >= seq.max_tokens - min(n + 1, self.gamma))

                if n == self.gamma:
                    seq.cur_acc_tokens += n
                else:
                    seq.num_acc_tokens.append(seq.cur_acc_tokens + n + 1)
                    seq.cur_acc_tokens = 0

            verify_res = torch.tensor([acc, rollout, revise_token, finish], dtype=torch.int64, device="cuda")

        return verify_res

    def recv_parallel_draft_window(self, seqs: list[Sequence]):
        """Receive gamma draft tokens from draft master and build verify_seqs.

        Uses two-phase protocol: count, then seq_ids, then tokens.
        Builds verify_seqs by looking up received seq_ids in running set.
        Preserves received order. Asserts all received ids exist in running.
        Does NOT set pending_draft_tokens on target side.
        """
        count_t = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(count_t, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        num_verify = int(count_t.item())

        if num_verify == 0:
            return []

        ids_t = torch.zeros(num_verify, dtype=torch.int64, device="cuda")
        dist.broadcast(ids_t, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        received_ids = ids_t.tolist()

        running_by_id = {s.seq_id: s for s in self.scheduler.running}
        verify_seqs = []
        for sid in received_ids:
            seq = running_by_id.get(sid)
            assert seq is not None, (
                f"recv_parallel_draft_window: seq_id {sid} not in running set "
                f"(running_ids={sorted(running_by_id.keys())})"
            )
            verify_seqs.append(seq)

        total_tokens = self.gamma * num_verify
        msg_t = torch.empty(total_tokens, dtype=torch.int64, device="cuda")
        dist.broadcast(msg_t, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        flat = msg_t.tolist()
        for i, seq in enumerate(verify_seqs):
            toks = flat[i * self.gamma : (i + 1) * self.gamma]
            assert len(toks) == self.gamma, (
                f"recv_parallel_draft_window: seq {seq.seq_id} "
                f"expected {self.gamma} tokens, got {len(toks)}"
            )
            for tok in toks:
                seq.append_token(tok)
            self._ensure_block_table_covers_sequence(seq)
            # Do NOT set pending_draft_tokens on target side.

        return verify_seqs

    def _validate_full_gamma_pearl_step(self):
        """Synchronous full-gamma PEARL step for Phase 1 validation.

        Reuses serialized full-gamma preparation, verification, and postprocess.
        Keeps parallel_pearl execution mode for traces. NOT the performance path.
        """
        self._decode_iteration_group += 1
        dist.barrier()
        seqs, is_prefill = self.scheduler.schedule()
        trace_record = self._trace_schedule(seqs, is_prefill, "verify")
        assert not is_prefill, "wrong match. current stage is prefill."

        # Receive draft window (reuses serialized recv — same protocol).
        self.recv_serialized_draft_window(seqs)

        # Full-gamma preparation and verification.
        input_ids, positions, temp_seqs = self.prepare_serialized_verify_decode(seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        verify_res = self.serialized_verify_full_gamma(logits, seqs, temperatures)

        # Broadcast verify_res so draft ranks can apply postprocess.
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Full-gamma postprocess.
        accepted_lens, invalidated_lens = self._full_gamma_postprocess(seqs, verify_res)

        verified_tokens = self.gamma * len(seqs)
        trace_record["verified_tokens_total"] = verified_tokens
        torch.cuda.synchronize()
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def pearl_step(self):
        self._decode_iteration_group += 1

        # Phase 1 toggle: synchronous full-gamma validation mode
        if os.environ.get("PEARL_FULL_GAMMA_VALIDATE"):
            return self._validate_full_gamma_pearl_step()

        seqs, is_prefill = self.scheduler.schedule()
        assert not is_prefill, "wrong match. current stage is prefill."

        # Phase A: receive verify windows from draft (two-phase broadcast)
        verify_seqs = self.recv_parallel_draft_window(seqs)

        if not verify_seqs:
            # First iteration: nothing to verify yet
            return

        # Build trace from verify_seqs (not the broader scheduled seqs)
        trace_record = self._trace_schedule(verify_seqs, is_prefill, "verify")

        # Phase B: full-gamma verification
        input_ids, positions, temp_seqs = self.prepare_parallel_verify_decode(verify_seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None

        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        verify_res = self.parallel_verify_full_gamma(logits, verify_seqs, temperatures)

        # Phase C: broadcast results back to draft
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Phase D: postprocess (target-side: invalidated_predraft=0 by definition)
        accepted_lens, invalidated_lens = self._full_gamma_postprocess(verify_seqs, verify_res)

        # Runtime assert
        verified_tokens = self.gamma * len(verify_seqs)
        assert verified_tokens == self.gamma * len(verify_seqs)
        trace_record["verified_tokens_total"] = verified_tokens

        torch.cuda.synchronize()
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def serialized_pearl_step(self):
        """Serialized speculative decoding target verification phase.

        This is a serial speculative decoding baseline, NOT PEARL's two-stage
        pre-verify behavior. Protocol:

        1. Global barrier (paired with draft).
        2. Receive the gamma-token draft window from the draft runner.
        3. Prepare target inputs spanning the last confirmed token + draft window.
        4. Run the target model over all gamma positions.
        5. Full-gamma verification → verify_res broadcast to draft group.
        6. Serialized postprocess (shared with draft side).

        Every iteration verifies exactly gamma draft tokens per active request.
        accepted_len ∈ [0, gamma]. Does not use PEARL's pre_verify state machine.
        """
        self._decode_iteration_group += 1
        dist.barrier()
        seqs, is_prefill = self.scheduler.schedule()
        trace_record = self._trace_schedule(seqs, is_prefill, "serialized_verify")
        assert not is_prefill, "wrong match. current stage is prefill."

        # Receive the current iteration's gamma-token draft window.
        self.recv_serialized_draft_window(seqs)

        # Prepare inputs for full-gamma verification.
        input_ids, positions, temp_seqs = self.prepare_serialized_verify_decode(seqs)
        assert input_ids.size(0) == positions.size(0), (
            f"serialized_pearl_step: input_ids={input_ids.size(0)} != positions={positions.size(0)}"
        )
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)

        # Full-gamma verification.
        verify_res = self.serialized_verify_full_gamma(logits, seqs, temperatures)

        # Broadcast verify_res so draft ranks can apply postprocess.
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Postprocess (same logic as draft side).
        accepted_lens, invalidated_lens = self._full_gamma_postprocess(seqs, verify_res)
        torch.cuda.synchronize()
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    @torch.inference_mode()
    def verify(self, logits: torch.Tensor, seqs: list[Sequence], temperatures: torch.Tensor):
        """Refer to the verification logic in the draft model verification function."""
        # verify_res will be sent to the sub-process in the target group.
        num_to_be_verified_tokens = sum([1 if seq.pre_verify else self.gamma for seq in seqs])
        num_next_round_input = self.gamma * len(seqs)
        msg = torch.zeros(num_to_be_verified_tokens + num_next_round_input, dtype=torch.int64, device="cuda")
        dist.broadcast(msg, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        to_be_verified_tokens = msg[:num_to_be_verified_tokens].tolist()
        next_round_input = msg[num_to_be_verified_tokens:].tolist()
        
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")

        if self.tp_params.local_rank == 0:
            r = torch.rand(num_to_be_verified_tokens, device="cuda")
            target_logits = norm_logits(logits, temperatures)
            target_prob = target_logits.gather(dim=1, index=msg[:num_to_be_verified_tokens].unsqueeze(1)).squeeze(1)
            judge = (r <= target_prob).tolist()

            # keep original logic; add logs around sampling
            logits.scatter_(1, msg[:num_to_be_verified_tokens].unsqueeze(1), -float("inf"))
            revised_tokens = self.sampler(logits, temperatures)

            acc, rollout, revise_token, finish = [], [], [], []

            v_idx = 0
            for i, seq in enumerate(seqs):
                if seq.pre_verify:
                    acc.append(judge[v_idx])
                    rollout.append(0 if judge[v_idx] else self.gamma)
                    revise_token.append(revised_tokens[v_idx])

                    if judge[v_idx]:
                        seq.cur_acc_tokens += 1
                        finish.append((not seq.ignore_eos and is_eos(to_be_verified_tokens[v_idx], self.scheduler.eos)) or seq.num_completion_tokens >= seq.max_tokens - 1)
                    else:
                        seq.num_acc_tokens.append(seq.cur_acc_tokens + 1)
                        seq.cur_acc_tokens = 0
                        finish.append((not seq.ignore_eos and is_eos(revise_token[-1], self.scheduler.eos)) or seq.num_completion_tokens >= seq.max_tokens - 1)
                else:
                    n = self.gamma
                    finish_flag = False
                    for j in range(v_idx, v_idx + self.gamma):
                        if not seq.ignore_eos and judge[j] and is_eos(to_be_verified_tokens[j], self.scheduler.eos):
                            finish_flag = True

                        if not judge[j]:
                            n = j - v_idx
                            break
                    acc.append(n == self.gamma)
                    rollout.append(self.gamma - n)
                    revise_token.append(revised_tokens[n + v_idx] if n < self.gamma else -1)
                    finish.append(finish_flag or seq.num_completion_tokens >= seq.max_tokens - min(n + 1, self.gamma))

                    if n == self.gamma:
                        seq.cur_acc_tokens += n
                    else:
                        seq.num_acc_tokens.append(seq.cur_acc_tokens + n + 1)
                        seq.cur_acc_tokens = 0
                    
                v_idx += 1 if seq.pre_verify else self.gamma
        
            verify_res = torch.tensor([acc, rollout, revise_token, finish], dtype=torch.int64, device="cuda")
        
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # post-process the seqs according to the verify_res.
        acc, rollout, revise_token, finish = verify_res.tolist()
        accepted_lens = {}
        invalidated_lens = {}

        for idx, seq in enumerate(seqs):
            was_pre_verify = seq.pre_verify
            accepted_len = 1 if was_pre_verify and acc[idx] else 0
            if not was_pre_verify:
                accepted_len = self.gamma if acc[idx] else self.gamma - rollout[idx]
            invalidated_len = 0 if acc[idx] else rollout[idx]
            accepted_lens[seq.seq_id] = accepted_len
            invalidated_lens[seq.seq_id] = invalidated_len
            seq.record_accepted(accepted_len)
            seq.record_invalidated_predraft(invalidated_len)
            
            if seq.pre_verify:
                if acc[idx]:
                    seq.pre_verify = False
                    for token in next_round_input[self.gamma * idx:self.gamma * (idx + 1)]:
                        seq.append_token(token)
                else:
                    seq.pre_verify = True
                    seq.append_token(revise_token[idx])
            else:
                if acc[idx]:
                    seq.pre_verify = False
                    for token in next_round_input[self.gamma * idx:self.gamma * (idx + 1)]:
                        seq.append_token(token)
                else:
                    seq.pre_verify = True
                    if rollout[idx] > 1:
                        self.scheduler.rollback(seq, rollout[idx] - 1)
                    # A verification rejection ends the current speculative span,
                    # not the request. Do not stamp request-level finish_ts until
                    # the scheduler actually moves the sequence to finished.
                    seq.mark_finished(record_finish_ts=False)

            if finish[idx]:
                seq.mark_finished()
                seq.num_acc_tokens.append(seq.cur_acc_tokens)
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)
                continue
        return accepted_lens, invalidated_lens
