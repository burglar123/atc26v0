from __future__ import annotations

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
from nano_pearl.pearl_config import PEARLConfig, validate_eager_gamma
from dataclasses import dataclass
from nano_pearl.models import model_dict
from nano_pearl.utils.loader import load_model
from nano_pearl.pearl_config import TPParams
from nano_pearl.layers.sampler import Sampler, norm_logits, SamplingParams
from nano_pearl.utils.context import set_context, reset_context, get_context
from nano_pearl.pearl_engine.sequence import (
    Sequence,
    assert_sequence_matches_checkpoint,
    make_sequence_checkpoint,
)
from nano_pearl.pearl_engine.scheduler import Scheduler, is_eos
from nano_pearl.pearl_engine.sequence import SequenceStatus
from nano_pearl.pearl_engine.step_plan import RequestBudget, StepPlan
from nano_pearl.pearl_engine.dual_batch import (
    BufferedProposal,
    DualBatchManager,
    EAGER_STATE_DISCARDED,
    EAGER_STATE_DRAFTED_DRY_RUN,
    EAGER_STATE_READY_TO_VERIFY,
    EagerProposal,
    EagerProposalBuffer,
    LANE_EAGER,
    LANE_NORMAL,
    ProposalBuffer,
)
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
        self._trace_plan_id = 0
        self.active_execution_mode = self.global_config.execution_mode
        self.active_decode_ready_mode = False
        self.dual_batch_manager = DualBatchManager(self.gamma)
        self.dual_proposal_buffer = ProposalBuffer()
        self.eager_proposal_buffer = EagerProposalBuffer()
        self._eager_proposal_id = 0
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
            "max_tokens": int(seq.max_tokens),
            "temperature": float(seq.temperature),
            "ignore_eos": bool(seq.ignore_eos),
            "arrival_ts": float(seq.arrival_ts),
            "arrival_offset_sec": arrival_offset_sec if arrival_offset_sec is not None else getattr(seq, "arrival_offset_sec", None),
            "slo_tpot_ms": seq.slo_tpot_ms,
            "slo_class": seq.slo_class,
            "per_request_gamma": seq.per_request_gamma,
            "home_batch_id": seq.home_batch_id,
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
        seq.decode_ready_mode = bool(snapshot.get("decode_ready_mode", False))
        seq.num_decode_ready_prefill_tokens = int(snapshot.get("num_decode_ready_prefill_tokens", 0))
        seq.home_batch_id = snapshot.get("home_batch_id")
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

    def _build_step_plan_from_scheduled_batch(self, seqs: list[Sequence], is_prefill: bool, runner_role: str, batch_id: str, iteration_id: int) -> StepPlan:
        seq_ids = [seq.seq_id for seq in seqs]
        budgets = {seq_id: RequestBudget(normal_gamma=self.gamma, eager_gamma=0) for seq_id in seq_ids}
        is_draft_role = "draft" in runner_role
        plan = StepPlan(
            plan_id=self._trace_plan_id,
            iteration_id=iteration_id,
            execution_mode=self.active_execution_mode,
            target_home_set=[] if is_draft_role else list(seq_ids),
            target_eager_set=[],
            draft_home_set=list(seq_ids) if is_draft_role else [],
            draft_eager_set=[],
            budgets=budgets,
            target_batch_id=None if is_draft_role else batch_id,
            draft_batch_id=batch_id if is_draft_role else None,
            decode_ready_mode=self.active_decode_ready_mode,
            is_prefill=is_prefill,
        )
        plan.validate_phase1b(self.gamma, runner_role)
        return plan

    def _resolve_plan_seqs(self, plan: StepPlan, runner_role: str) -> list[Sequence]:
        plan.validate_phase1b(self.gamma, runner_role)
        seq_ids = plan.role_seq_ids(runner_role)
        resolved = self.scheduler.find_by_seq_ids(seq_ids)
        resolved_ids = [seq.seq_id for seq in resolved]
        assert resolved_ids == seq_ids, (
            f"Resolved seq_id mismatch for runner_role={runner_role}: expected={seq_ids}, resolved={resolved_ids}"
        )
        return resolved

    def _profile_defaults(self, seqs: list[Sequence], step_plan: StepPlan) -> dict:
        target_home_size = len(step_plan.target_home_set)
        draft_home_size = len(step_plan.draft_home_set)
        active_seq_count = int(step_plan.active_seq_count) if step_plan.active_seq_count else len(seqs)
        active_batch_count = int(step_plan.active_batch_count) if step_plan.active_batch_count else (1 if seqs else 0)
        target_fraction_of_active = target_home_size / max(1, active_seq_count)
        draft_fraction_of_active = draft_home_size / max(1, active_seq_count)
        split_imbalance = abs(target_home_size - draft_home_size) / max(1, target_home_size + draft_home_size)
        target_to_draft_size_ratio = target_home_size / max(1, draft_home_size)
        return {
            "step_start_ts": None,
            "step_end_ts": None,
            "step_time_ms": 0.0,
            "overlap_time_ms": 0.0,
            "overlap_ratio": None,
            "exposed_draft_time_ms": None,
            "exposed_verify_time_ms": None,
            "pipeline_bubble_ms": None,
            "draft_tokens_generated": 0,
            "proposal_tokens_available": 0,
            "proposal_tokens_verified": 0,
            "accepted_tokens": 0,
            "invalidated_predraft_tokens": 0,
            "wasted_draft_tokens": 0,
            "draft_waste_rate": None,
            "acceptance_rate": None,
            "target_home_size": target_home_size,
            "draft_home_size": draft_home_size,
            "active_batch_size": len(seqs),
            "running_queue_size": len(self.scheduler.running),
            "target_batch_size": target_home_size,
            "draft_batch_size": draft_home_size,
            "active_seq_count": active_seq_count,
            "active_batch_count": active_batch_count,
            "target_fraction_of_active": target_fraction_of_active,
            "draft_fraction_of_active": draft_fraction_of_active,
            "split_imbalance": split_imbalance,
            "target_to_draft_size_ratio": target_to_draft_size_ratio,
            "enable_eager_execution": bool(getattr(self.global_config, "enable_eager_execution", False)),
            "eager_execution_enabled": False,
            "enable_eager_draft_dry_run": bool(
                getattr(self.global_config, "enable_eager_draft_dry_run", False)
            ),
            "eager_draft_dry_run_enabled": False,
            "enable_eager_promotion_dry_run": bool(
                getattr(self.global_config, "enable_eager_promotion_dry_run", False)
            ),
            "eager_promotion_dry_run_enabled": False,
            "eager_buffer_size_before": self.eager_proposal_buffer.size(),
            "eager_buffer_size_after": self.eager_proposal_buffer.size(),
            "eager_parent_seq_ids": [],
            "eager_parent_accepted_len_by_seq_id": {},
            "eager_parent_invalidated_len_by_seq_id": {},
            "eager_parent_full_accept_by_seq_id": {},
            "eager_parent_finished_by_seq_id": {},
            "eager_promoted_seq_ids": [],
            "eager_promoted_proposal_ids": [],
            "eager_discarded_seq_ids": [],
            "eager_discarded_proposal_ids": [],
            "eager_promotion_reason_by_seq_id": {},
            "eager_discard_reason_by_seq_id": {},
            "eager_promotion_base_len_by_seq_id": {},
            "eager_promotion_current_len_by_seq_id": {},
            "eager_promotion_base_match_by_seq_id": {},
            "eager_verified_seq_ids": [],
            "eager_accepted_seq_ids": [],
            "eager_rejected_seq_ids": [],
            "eager_tokens_generated": 0,
            "eager_tokens_promoted": 0,
            "eager_tokens_discarded": 0,
            "eager_tokens_verified": 0,
            "eager_tokens_accepted": 0,
            "eager_tokens_rejected": 0,
            "eager_tokens_invalidated": 0,
            "eager_dry_run_tokens_generated": 0,
            "eager_waste_rate": 0.0,
        }

    def _finalize_record_profile(self, record: dict):
        draft_start = record.get("draft_start_ts")
        draft_end = record.get("draft_end_ts")
        verify_start = record.get("verify_start_ts")
        verify_end = record.get("verify_end_ts")
        draft_time_ms = float(record.get("draft_time_ms") or 0.0)
        verify_time_ms = float(record.get("verify_time_ms") or 0.0)
        step_start = record.get("step_start_ts")
        step_end = record.get("step_end_ts")
        if step_start is not None and step_end is not None:
            record["step_time_ms"] = max(0.0, (step_end - step_start) * 1000)

        overlap_time_ms = 0.0
        if draft_start is not None and draft_end is not None and verify_start is not None and verify_end is not None:
            overlap_s = max(0.0, min(draft_end, verify_end) - max(draft_start, verify_start))
            overlap_time_ms = overlap_s * 1000
        record["overlap_time_ms"] = overlap_time_ms

        min_stage_ms = min(draft_time_ms, verify_time_ms)
        record["overlap_ratio"] = None if min_stage_ms <= 0 else overlap_time_ms / max(1e-9, min_stage_ms)
        record["exposed_draft_time_ms"] = max(0.0, draft_time_ms - overlap_time_ms)
        record["exposed_verify_time_ms"] = max(0.0, verify_time_ms - overlap_time_ms)
        # Phase 1D uses the inclusion-exclusion bubble:
        # wall step time minus union(draft interval, verify interval).
        record["pipeline_bubble_ms"] = max(
            0.0,
            float(record.get("step_time_ms") or 0.0) - draft_time_ms - verify_time_ms + overlap_time_ms,
        )

        accepted_tokens = int(record.get("accepted_tokens") or 0)
        invalidated_tokens = int(record.get("invalidated_predraft_tokens") or 0)
        dropped_tokens = int(record.get("proposal_buffer_dropped_count") or 0) * max(int(self.gamma), 0)
        wasted_tokens = invalidated_tokens + dropped_tokens
        record["wasted_draft_tokens"] = wasted_tokens
        generated = int(record.get("draft_tokens_generated") or 0)
        verified = int(record.get("proposal_tokens_verified") or 0)
        record["draft_waste_rate"] = wasted_tokens / max(1, generated) if generated else None
        record["acceptance_rate"] = accepted_tokens / max(1, verified) if verified else None

    def _proposal_verify_token_count(self, seqs: list[Sequence]) -> int:
        return sum(1 if seq.pre_verify else self.gamma for seq in seqs)

    def _trace_schedule(self, seqs: list[Sequence], is_prefill: bool, runner_role: str):
        iteration_id, batch_id = self.scheduler.next_batch_id(runner_role)
        self._trace_plan_id += 1
        for seq in seqs:
            seq.mark_scheduled(iteration_id, batch_id, is_prefill, runner_role)
        seq_ids = [seq.seq_id for seq in seqs]
        step_plan = self._build_step_plan_from_scheduled_batch(seqs, is_prefill, runner_role, batch_id, iteration_id)
        per_seq_zeros = {seq.seq_id: 0 for seq in seqs}
        record = {
            "execution_mode": self.active_execution_mode,
            "decode_ready_mode": self.active_decode_ready_mode,
            "iteration_id": iteration_id,
            "batch_id": batch_id,
            "runner_role": runner_role,
            "scheduled_seq_ids": seq_ids,
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
        }
        record.update(step_plan.to_trace_dict())
        record.update(self._profile_defaults(seqs, step_plan))
        self.trace_records.append(record)
        return record, step_plan

    def _prepare_dual_batch_state(self):
        self.dual_batch_manager.gamma = int(self.gamma)
        self.dual_batch_manager.update_running(self.scheduler.running)
        active_seq_ids = [seq.seq_id for seq in self.scheduler.running]
        return self.dual_proposal_buffer.discard_inactive(active_seq_ids)

    def _eager_plan_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_plan_dry_run", False))

    def _eager_draft_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_draft_dry_run", False))

    def _eager_promotion_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_promotion_dry_run", False))

    def _pending_eager_seq_ids(self) -> set[int]:
        return {
            int(seq_id)
            for seq_id in (
                self.eager_proposal_buffer.pending_seq_ids()
                + self.eager_proposal_buffer.ready_seq_ids()
            )
        }

    def _apply_eager_plan_dry_run(self, plan: StepPlan) -> None:
        promotion_dry_run_enabled = self._eager_promotion_dry_run_enabled()
        draft_dry_run_enabled = self._eager_draft_dry_run_enabled() or promotion_dry_run_enabled
        dry_run_enabled = self._eager_plan_dry_run_enabled() or draft_dry_run_enabled
        policy = str(getattr(self.global_config, "eager_policy", "none"))
        gamma = int(self.gamma)
        plan.enable_eager_plan_dry_run = dry_run_enabled
        plan.enable_eager_draft_dry_run = draft_dry_run_enabled
        plan.enable_eager_promotion_dry_run = promotion_dry_run_enabled
        plan.eager_policy = policy
        plan.eager_post_verify_only = True
        plan.eager_gamma_equals_global_gamma = (
            int(getattr(self.global_config, "max_eager_tokens_per_request", 0) or 0) == gamma
        )
        plan.eager_active_seq_ids = []
        plan.eager_ready_seq_ids = self.eager_proposal_buffer.ready_seq_ids()
        plan.eager_continuing_set = []
        plan.target_eager_set = []

        if not dry_run_enabled:
            return

        if draft_dry_run_enabled:
            if self.active_execution_mode != "dual_batch_pearl":
                raise ValueError("enable_eager_draft_dry_run requires execution_mode='dual_batch_pearl'")
            if policy == "none":
                raise ValueError("enable_eager_draft_dry_run requires eager_policy != 'none'")
            if int(getattr(self.global_config, "max_eager_requests_per_step", 0) or 0) <= 0:
                raise ValueError("enable_eager_draft_dry_run requires max_eager_requests_per_step > 0")
            validate_eager_gamma(self.global_config, gamma)

        if policy == "none":
            return
        if policy != "tight_only":
            raise ValueError(f"Unsupported eager_policy={policy!r}; Phase 1H-1 supports only 'none' and 'tight_only'")

        if plan.execution_mode != "dual_batch_pearl" or plan.plan_phase != "steady":
            plan.validate_phase1h_eager_scaffold(
                enable_eager_execution=False,
                enable_eager_plan_dry_run=True,
                enable_eager_draft_dry_run=draft_dry_run_enabled,
                enable_eager_promotion_dry_run=promotion_dry_run_enabled,
                global_gamma=gamma,
            )
            return

        target_seq_ids = [int(seq_id) for seq_id in plan.target_home_set]
        target_home_set = set(target_seq_ids)
        target_seqs = self.scheduler.find_by_seq_ids(target_seq_ids) if target_seq_ids else []
        seq_by_id = {int(seq.seq_id): seq for seq in target_seqs}
        running_seq_ids = {int(seq.seq_id) for seq in self.scheduler.running}
        pending_eager_seq_ids = self._pending_eager_seq_ids()

        plan.eager_candidate_seq_ids = list(target_seq_ids)
        plan.eager_skipped_not_in_target_home_set_seq_ids = []
        selected_candidates = []
        reject_reasons: dict[int, str] = {}
        skipped_pre_verify = []
        skipped_non_tight = []
        pre_verify_candidate_count = 0
        post_verify_candidate_count = 0

        for seq_id in target_seq_ids:
            seq = seq_by_id.get(seq_id)
            reason = None
            if seq_id not in target_home_set:
                reason = "not_in_target_home_set"
            elif seq is None:
                reason = "missing_sequence"
            elif int(seq.seq_id) not in running_seq_ids or getattr(seq, "status", None) != SequenceStatus.RUNNING:
                reason = "not_running"
            elif seq.is_finished:
                reason = "finished"
            elif seq_id in plan.eager_active_seq_ids:
                reason = "already_eager_active"
            elif seq_id in pending_eager_seq_ids:
                reason = "pending_eager_proposal"
            elif bool(seq.pre_verify):
                reason = "pre_verify"
                pre_verify_candidate_count += 1
                skipped_pre_verify.append(seq_id)
            else:
                post_verify_candidate_count += 1
                if getattr(seq, "slo_class", None) != "tight":
                    reason = "non_tight"
                    skipped_non_tight.append(seq_id)

            if reason is None:
                selected_candidates.append(seq)
            else:
                reject_reasons[seq_id] = reason

        plan.eager_pre_verify_candidate_count = pre_verify_candidate_count
        plan.eager_post_verify_candidate_count = post_verify_candidate_count
        plan.eager_skipped_pre_verify_seq_ids = sorted(skipped_pre_verify)
        plan.eager_skipped_non_tight_seq_ids = sorted(skipped_non_tight)
        plan.eager_candidate_reject_reason_by_seq_id = reject_reasons

        selected_candidates = sorted(selected_candidates, key=lambda seq: int(seq.seq_id))
        if selected_candidates and int(getattr(self.global_config, "max_eager_requests_per_step", 0) or 0) > 0:
            validate_eager_gamma(self.global_config, gamma)

        max_requests = int(getattr(self.global_config, "max_eager_requests_per_step", 0) or 0)
        selected = selected_candidates[:max_requests]
        selected_seq_ids = [int(seq.seq_id) for seq in selected]

        plan.eager_new_selected_set = list(selected_seq_ids)
        plan.draft_eager_set = list(selected_seq_ids)
        plan.target_eager_set = []
        plan.eager_selected_seq_ids = list(selected_seq_ids)
        plan.eager_budget_by_seq_id = {seq_id: gamma for seq_id in selected_seq_ids}
        plan.eager_total_budget = len(selected_seq_ids) * gamma
        for seq in selected:
            seq_id = int(seq.seq_id)
            if seq_id not in plan.budgets:
                plan.budgets[seq_id] = RequestBudget(normal_gamma=gamma, eager_gamma=0)
            plan.budgets[seq_id].eager_gamma = gamma
            plan.eager_base_len_by_seq_id[seq_id] = len(seq)
            plan.eager_base_pre_verify_by_seq_id[seq_id] = bool(seq.pre_verify)
            plan.eager_parent_kind_by_seq_id[seq_id] = LANE_NORMAL

        plan.validate_phase1h_eager_scaffold(
            enable_eager_execution=False,
            enable_eager_plan_dry_run=True,
            enable_eager_draft_dry_run=draft_dry_run_enabled,
            enable_eager_promotion_dry_run=promotion_dry_run_enabled,
            global_gamma=gamma,
        )

    def _build_dual_batch_step_plan(self) -> StepPlan:
        proposal_buffer_size_before = self.dual_proposal_buffer.size()
        dropped_seq_ids = self._prepare_dual_batch_state()
        iteration_id, _ = self.scheduler.next_batch_id("dual_batch")
        self._trace_plan_id += 1
        plan_id = self._trace_plan_id
        plan = self.dual_batch_manager.build_step_plan(
            plan_id=plan_id,
            iteration_id=iteration_id,
            execution_mode=self.active_execution_mode,
            decode_ready_mode=self.active_decode_ready_mode,
            pending_proposal_seq_ids=self.dual_proposal_buffer.pending_seq_ids(),
            pending_batch_ids=self.dual_proposal_buffer.pending_batch_ids(),
            enable_eager_execution=bool(getattr(self.global_config, "enable_eager_execution", False)),
        )
        self._apply_eager_plan_dry_run(plan)
        buffer_inspect = self.dual_proposal_buffer.inspect(plan.target_home_set)
        plan.proposal_buffer_size_before = int(proposal_buffer_size_before)
        plan.proposal_buffer_size_after = self.dual_proposal_buffer.size()
        plan.proposal_buffer_requested_seq_ids = buffer_inspect["requested_seq_ids"]
        plan.proposal_buffer_hit_seq_ids = buffer_inspect["hit_seq_ids"]
        plan.proposal_buffer_miss_seq_ids = buffer_inspect["miss_seq_ids"]
        plan.proposal_buffer_invalid_seq_ids = buffer_inspect["invalid_seq_ids"]
        plan.proposal_buffer_dropped_seq_ids = [int(seq_id) for seq_id in dropped_seq_ids]
        plan.proposal_buffer_hit_count = len(plan.proposal_buffer_hit_seq_ids)
        plan.proposal_buffer_miss_count = len(plan.proposal_buffer_miss_seq_ids)
        plan.proposal_buffer_invalid_count = len(plan.proposal_buffer_invalid_seq_ids)
        plan.proposal_buffer_dropped_count = len(plan.proposal_buffer_dropped_seq_ids)
        if plan.plan_phase == "fallback":
            plan.fallback_buffer_hit_count = plan.proposal_buffer_hit_count
            plan.fallback_buffer_miss_count = plan.proposal_buffer_miss_count
            if plan.proposal_buffer_miss_count and plan.fallback_reason == "single_active_batch_with_buffered_proposals":
                plan.fallback_reason = "missing_target_proposals"
            if plan.fallback_reason is None:
                if not plan.target_home_set and plan.draft_home_set:
                    plan.fallback_reason = "empty_target_batch"
                elif plan.target_home_set and not plan.draft_home_set:
                    plan.fallback_reason = "empty_draft_batch"
                elif not plan.target_home_set and not plan.draft_home_set:
                    plan.fallback_reason = "no_active_batch"
                else:
                    plan.fallback_reason = "unknown_fallback_condition"
            if plan.dual_batch_state is not None:
                plan.dual_batch_state.fallback_reason = plan.fallback_reason
        return plan

    def _resolve_dual_seq_ids(self, seq_ids: list[int], plan: StepPlan, label: str) -> list[Sequence]:
        if not seq_ids:
            return []
        seqs = self.scheduler.find_by_seq_ids(seq_ids)
        resolved_ids = [seq.seq_id for seq in seqs]
        assert resolved_ids == list(seq_ids), (
            f"Dual-batch resolved seq_id mismatch for {label}: plan_id={plan.plan_id}, "
            f"target_home_set={plan.target_home_set}, draft_home_set={plan.draft_home_set}, "
            f"expected={seq_ids}, resolved={resolved_ids}"
        )
        running_seq_ids = {seq.seq_id for seq in self.scheduler.running}
        for seq in seqs:
            assert seq.seq_id in running_seq_ids, (
                f"Dual-batch attempted to schedule inactive seq_id={seq.seq_id}: "
                f"plan_id={plan.plan_id}, label={label}, status={seq.status}"
            )
        return seqs

    def _allocate_decode_slots_for_dual(self, seqs: list[Sequence], plan: StepPlan, label: str):
        for seq in seqs:
            if not self.scheduler.block_manager.can_append(seq):
                raise RuntimeError(
                    f"Dual-batch cannot append KV slot for seq_id={seq.seq_id}: "
                    f"plan_id={plan.plan_id}, label={label}, free_blocks="
                    f"{len(self.scheduler.block_manager.free_block_ids)}"
                )
            self.scheduler.block_manager.may_append(seq)

    def _trace_dual_batch_schedule(self, seqs: list[Sequence], plan: StepPlan, runner_role: str):
        batch_id = f"{runner_role}-{plan.iteration_id}"
        for seq in seqs:
            seq.mark_scheduled(plan.iteration_id, batch_id, False, runner_role)
        seq_ids = [seq.seq_id for seq in seqs]
        per_seq_zeros = {seq.seq_id: 0 for seq in seqs}
        record = {
            "execution_mode": self.active_execution_mode,
            "decode_ready_mode": self.active_decode_ready_mode,
            "iteration_id": plan.iteration_id,
            "batch_id": batch_id,
            "runner_role": runner_role,
            "scheduled_seq_ids": seq_ids,
            "resolved_seq_ids": list(seq_ids),
            "request_ids": [seq.request_id for seq in seqs],
            "num_seqs_in_batch": len(seqs),
            "is_prefill": False,
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
        }
        record.update(plan.to_trace_dict())
        record.update(self._profile_defaults(seqs, plan))
        self.trace_records.append(record)
        return record

    def _proposal_assertion_message(self, plan: StepPlan, detail: str) -> str:
        return (
            f"{detail}: plan_id={plan.plan_id}, target_home_set={plan.target_home_set}, "
            f"draft_home_set={plan.draft_home_set}, buffered_proposal_seq_ids="
            f"{self.dual_proposal_buffer.pending_seq_ids()}"
        )

    def _build_buffered_proposals(self, seqs: list[Sequence], plan: StepPlan) -> list[BufferedProposal]:
        proposals = []
        for seq in seqs:
            proposal_tokens = list(seq.token_ids[-self.gamma:])
            if seq.pre_verify:
                to_be_verified = [int(seq.token_ids[-self.gamma])]
            else:
                to_be_verified = [int(x) for x in seq.token_ids[-2 * self.gamma + 1:-self.gamma + 1]]
            assert len(proposal_tokens) == self.gamma, self._proposal_assertion_message(
                plan,
                f"proposal length mismatch for seq_id={seq.seq_id}",
            )
            expected_verify_len = 1 if seq.pre_verify else self.gamma
            assert len(to_be_verified) == expected_verify_len, self._proposal_assertion_message(
                plan,
                f"to_be_verified length mismatch for seq_id={seq.seq_id}",
            )
            proposals.append(
                BufferedProposal(
                    seq_id=int(seq.seq_id),
                    request_id=seq.request_id,
                    home_batch_id=int(seq.home_batch_id),
                    proposal_token_ids=[int(x) for x in proposal_tokens],
                    to_be_verified_token_ids=to_be_verified,
                    proposal_len=int(self.gamma),
                    pre_verify=bool(seq.pre_verify),
                    plan_id=int(plan.plan_id),
                    valid=True,
                )
            )
        return proposals

    def _serialize_proposals(self, proposals: list[BufferedProposal], plan: StepPlan) -> tuple[torch.Tensor, torch.Tensor]:
        header = []
        tokens = []
        for proposal in proposals:
            header.extend(
                [
                    int(proposal.seq_id),
                    int(proposal.home_batch_id),
                    int(proposal.pre_verify),
                    int(len(proposal.to_be_verified_token_ids)),
                    int(proposal.proposal_len),
                ]
            )
            tokens.extend(int(token) for token in proposal.to_be_verified_token_ids)
            tokens.extend(int(token) for token in proposal.proposal_token_ids)
        payload = header + tokens
        meta = torch.tensor(
            [
                len(proposals),
                len(payload),
                int(self.gamma),
                int(plan.plan_id),
                -1 if plan.draft_batch_id is None else int(plan.draft_batch_id),
            ],
            dtype=torch.int64,
            device="cuda",
        )
        payload_tensor = torch.tensor(payload, dtype=torch.int64, device="cuda")
        return meta, payload_tensor

    def _send_dual_proposals(self, proposals: list[BufferedProposal], plan: StepPlan):
        if self.tp_params.local_rank != 0:
            return
        meta, payload = self._serialize_proposals(proposals, plan)
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta[1].item()) > 0:
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)

    def _receive_dual_proposals(self, expected_seq_ids: list[int], plan: StepPlan) -> list[BufferedProposal]:
        meta = torch.zeros(5, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        n, payload_len, gamma, proposal_plan_id, batch_id = [int(x) for x in meta.tolist()]
        assert gamma == int(self.gamma), self._proposal_assertion_message(
            plan,
            f"proposal gamma mismatch: expected={self.gamma}, got={gamma}",
        )
        payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
        if payload_len > 0:
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        data = payload.tolist()
        header_len = n * 5
        headers = data[:header_len]
        token_data = data[header_len:]
        proposals = []
        token_offset = 0
        seq_lookup = {seq.seq_id: seq for seq in self.scheduler.find_by_seq_ids(expected_seq_ids)} if expected_seq_ids else {}
        for idx in range(n):
            base = idx * 5
            seq_id, home_batch_id, pre_verify, to_verify_len, proposal_len = headers[base:base + 5]
            to_verify = [int(x) for x in token_data[token_offset:token_offset + to_verify_len]]
            token_offset += to_verify_len
            proposal_tokens = [int(x) for x in token_data[token_offset:token_offset + proposal_len]]
            token_offset += proposal_len
            seq = seq_lookup.get(seq_id)
            proposals.append(
                BufferedProposal(
                    seq_id=int(seq_id),
                    request_id=seq.request_id if seq is not None else int(seq_id),
                    home_batch_id=int(home_batch_id),
                    proposal_token_ids=proposal_tokens,
                    to_be_verified_token_ids=to_verify,
                    proposal_len=int(proposal_len),
                    pre_verify=bool(pre_verify),
                    plan_id=int(proposal_plan_id),
                    valid=True,
                )
            )
        received_seq_ids = [proposal.seq_id for proposal in proposals]
        assert received_seq_ids == list(expected_seq_ids), self._proposal_assertion_message(
            plan,
            f"proposal seq_id mismatch: expected={expected_seq_ids}, received={received_seq_ids}, batch_id={batch_id}",
        )
        return proposals

    def _validate_proposals_for_target(self, proposals: list[BufferedProposal], seqs: list[Sequence], plan: StepPlan):
        proposal_seq_ids = [proposal.seq_id for proposal in proposals]
        seq_ids = [seq.seq_id for seq in seqs]
        assert proposal_seq_ids == seq_ids, self._proposal_assertion_message(
            plan,
            f"target proposal seq_id mismatch: expected={seq_ids}, got={proposal_seq_ids}",
        )
        for proposal, seq in zip(proposals, seqs):
            assert proposal.valid, self._proposal_assertion_message(
                plan,
                f"target proposal invalid for seq_id={seq.seq_id}",
            )
            assert seq.seq_id in {running_seq.seq_id for running_seq in self.scheduler.running}, self._proposal_assertion_message(
                plan,
                f"target proposal used for inactive seq_id={seq.seq_id}",
            )
            assert bool(proposal.pre_verify) == bool(seq.pre_verify), self._proposal_assertion_message(
                plan,
                f"proposal pre_verify mismatch for seq_id={seq.seq_id}: "
                f"proposal={proposal.pre_verify}, seq={seq.pre_verify}",
            )
            assert int(proposal.home_batch_id) == int(seq.home_batch_id), self._proposal_assertion_message(
                plan,
                f"proposal home_batch_id mismatch for seq_id={seq.seq_id}: "
                f"proposal={proposal.home_batch_id}, seq={seq.home_batch_id}",
            )
            assert int(proposal.proposal_len) == int(self.gamma), self._proposal_assertion_message(
                plan,
                f"proposal_len mismatch for seq_id={seq.seq_id}: proposal={proposal.proposal_len}, gamma={self.gamma}",
            )

    def _mark_trace_start(self, record: dict):
        if self.tp_params.local_rank != 0:
            return
        now = time.time()
        key = "draft_start_ts" if self.is_draft else "verify_start_ts"
        record[key] = now
        if record.get("step_start_ts") is None:
            record["step_start_ts"] = now
        if record["total_iteration_start_ts"] is None:
            record["total_iteration_start_ts"] = now

    def _update_trace_token_stats(self, record: dict, accepted_lens: dict[int, int] | None = None, invalidated_lens: dict[int, int] | None = None):
        if accepted_lens:
            accepted_lens = {seq_id: int(accepted_len) for seq_id, accepted_len in accepted_lens.items()}
            record["per_seq_accepted_len"].update(accepted_lens)
            record["accepted_tokens_per_seq"].update(accepted_lens)
            record["total_accepted_tokens"] = sum(record["accepted_tokens_per_seq"].values())
            record["accepted_tokens"] = record["total_accepted_tokens"]
        if invalidated_lens:
            invalidated_lens = {seq_id: int(invalidated_len) for seq_id, invalidated_len in invalidated_lens.items()}
            record["per_seq_invalidated_predraft_len"].update(invalidated_lens)
            record["invalidated_predraft_tokens"] = sum(record["per_seq_invalidated_predraft_len"].values())

    def _mark_trace_end(self, record: dict, accepted_lens: dict[int, int] | None = None, invalidated_lens: dict[int, int] | None = None):
        if self.tp_params.local_rank == 0:
            now = time.time()
            key = "draft_end_ts" if self.is_draft else "verify_end_ts"
            record[key] = now
            record["step_end_ts"] = now
            record["total_iteration_end_ts"] = now
            if record["draft_start_ts"] is not None and record["draft_end_ts"] is not None:
                record["draft_time_ms"] = (record["draft_end_ts"] - record["draft_start_ts"]) * 1000
            if record["verify_start_ts"] is not None and record["verify_end_ts"] is not None:
                record["verify_time_ms"] = (record["verify_end_ts"] - record["verify_start_ts"]) * 1000
            if record["total_iteration_start_ts"] is not None:
                record["total_iteration_time_ms"] = (now - record["total_iteration_start_ts"]) * 1000
        self._update_trace_token_stats(record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
        self._finalize_record_profile(record)

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
        trace_record, step_plan = self._trace_schedule(seqs, is_prefill, f"{self._runner_role()}_prefill")
        seqs = self._resolve_plan_seqs(step_plan, f"{self._runner_role()}_prefill")
        trace_record["resolved_seq_ids"] = [seq.seq_id for seq in seqs]
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
        seqs, is_prefill = self.scheduler.schedule()
        trace_record, step_plan = self._trace_schedule(seqs, is_prefill, self._runner_role())
        seqs = self._resolve_plan_seqs(step_plan, self._runner_role())
        trace_record["resolved_seq_ids"] = [seq.seq_id for seq in seqs]
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
        self.dual_batch_manager.reset()
        self.dual_proposal_buffer.clear()
        self.eager_proposal_buffer.clear()
        self._eager_proposal_id = 0
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

    def decode_ready_dual_batch_pearl_generate(self):
        """Decode-only Phase 1C dual-batch PEARL after prepare_decode_ready()."""
        self._set_execution_mode("dual_batch_pearl")
        self.active_decode_ready_mode = True
        dist.barrier()
        self._mark_decode_started()
        torch.cuda.synchronize()
        start_time = time.time()
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        self.dual_batch_manager.gamma = int(self.gamma)
        while not self.scheduler.is_finished():
            self.dual_batch_pearl_step()
        torch.cuda.synchronize()
        end_time = time.time()

        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        self._finish_decode_ready_generation(output, end_time - start_time)

    def cached_decode_ready_pearl_generate(self, max_active_cached_seqs: int = 0):
        if self.active_execution_mode == "dual_batch_pearl" or self.global_config.execution_mode == "dual_batch_pearl":
            raise NotImplementedError("cached-admission is not yet supported for dual_batch_pearl")
        self._set_execution_mode("parallel_pearl")
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
                self.pearl_step()
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

    def dual_batch_pearl_generate(self):
        self._set_execution_mode("dual_batch_pearl")
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()

        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        self.dual_batch_manager.gamma = int(self.gamma)

        while not self.scheduler.is_finished():
            self.dual_batch_pearl_step()

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

    @abstractmethod
    def pearl_step(self):
        pass

    @abstractmethod
    def serialized_pearl_step(self):
        pass

    @abstractmethod
    def dual_batch_pearl_step(self):
        pass


class DraftModelRunner(ModelRunnerBase):
    def __init__(self, config: PEARLConfig, rank: int, event: Event, control_event: Event):
        super().__init__(config, rank, event, control_event)

    def prepare_pearl_decode(self, seqs: list[Sequence]):
        return super().prepare_decode(seqs)

    def _draft_dual_batch_proposals(self, seqs: list[Sequence], plan: StepPlan) -> tuple[list[BufferedProposal], list[dict]]:
        draft_records = []
        for _ in range(self.gamma):
            self._allocate_decode_slots_for_dual(seqs, plan, "dual_draft")
            trace_record = self._trace_dual_batch_schedule(seqs, plan, "dual_draft")
            draft_records.append(trace_record)
            trace_record["draft_tokens_generated"] = len(seqs)
            input_ids, positions = self.prepare_pearl_decode(seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, False)
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
            self._mark_trace_end(trace_record)
        proposals = self._build_buffered_proposals(seqs, plan)
        for trace_record in draft_records:
            trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
        return proposals, draft_records

    def _next_eager_proposal_id(self) -> int:
        self._eager_proposal_id += 1
        return int(self._eager_proposal_id)

    def _run_eager_draft_dry_run(
        self,
        seqs: list[Sequence],
        plan: StepPlan,
        trace_record: dict,
    ) -> list[EagerProposal]:
        if not seqs:
            return []

        gamma = int(self.gamma)
        buffer_size_before = self.eager_proposal_buffer.size()
        checkpoints = {int(seq.seq_id): make_sequence_checkpoint(seq) for seq in seqs}
        generated_by_seq_id: dict[int, list[int]] = {int(seq.seq_id): [] for seq in seqs}
        proposals: list[EagerProposal] = []
        rollback_ok_by_seq_id: dict[int, bool] = {}
        rollback_seq_ids: list[int] = []
        draft_error: BaseException | None = None

        for seq in seqs:
            checkpoint = checkpoints[int(seq.seq_id)]
            assert checkpoint["pre_verify"] is False, (
                f"Phase 1H-2 eager draft dry-run requires post_verify seq_id={seq.seq_id}"
            )

        try:
            for _ in range(gamma):
                self._allocate_decode_slots_for_dual(seqs, plan, "eager_draft_dry_run")
                input_ids, positions = self.prepare_pearl_decode(seqs)
                torch.cuda.synchronize()
                logits = self.run_model(input_ids, positions, False)
                if self.tp_params.local_rank == 0:
                    sample_tokens = logits.argmax(dim=-1)
                else:
                    sample_tokens = torch.zeros(
                        len(seqs),
                        dtype=torch.int64,
                        pin_memory=True,
                    ).cuda(non_blocking=True)
                dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
                torch.cuda.synchronize()
                token_ids = sample_tokens.tolist()
                reset_context(self.tp_params)
                for seq, token_id in zip(seqs, token_ids):
                    int_token_id = int(token_id)
                    seq.append_token(int_token_id)
                    generated_by_seq_id[int(seq.seq_id)].append(int_token_id)

            for seq in seqs:
                seq_id = int(seq.seq_id)
                checkpoint = checkpoints[seq_id]
                proposal_tokens = [int(token_id) for token_id in generated_by_seq_id[seq_id]]
                to_be_verified = [int(token_id) for token_id in seq.token_ids[-2 * gamma + 1:-gamma + 1]]
                assert len(proposal_tokens) == gamma, (
                    f"eager proposal length mismatch for seq_id={seq_id}: "
                    f"expected={gamma}, got={len(proposal_tokens)}"
                )
                assert len(to_be_verified) == gamma, (
                    f"eager to_be_verified length mismatch for seq_id={seq_id}: "
                    f"expected={gamma}, got={len(to_be_verified)}"
                )
                assert int(checkpoint["len"]) == len(seq) - gamma, (
                    f"eager base_len mismatch for seq_id={seq_id}: "
                    f"base_len={checkpoint['len']}, len_after_eager={len(seq)}"
                )
                proposals.append(
                    EagerProposal(
                        proposal_id=self._next_eager_proposal_id(),
                        seq_id=seq_id,
                        request_id=seq.request_id,
                        lane=LANE_EAGER,
                        parent_proposal_id=None,
                        parent_kind=LANE_NORMAL,
                        parent_step_id=None,
                        source_step_id=0 if plan.step_id is None else int(plan.step_id),
                        source_plan_id=int(plan.plan_id),
                        home_batch_id=int(seq.home_batch_id),
                        base_len=int(checkpoint["len"]),
                        base_pre_verify=bool(checkpoint["pre_verify"]),
                        base_num_completion_tokens=int(checkpoint["num_completion_tokens"]),
                        proposal_token_ids=proposal_tokens,
                        to_be_verified_token_ids=to_be_verified,
                        proposal_len=gamma,
                        state=EAGER_STATE_DRAFTED_DRY_RUN,
                        valid=True,
                    )
                )
        except BaseException as exc:
            draft_error = exc
        finally:
            for seq in seqs:
                seq_id = int(seq.seq_id)
                checkpoint = checkpoints[seq_id]
                rollback_seq_ids.append(seq_id)
                try:
                    rollback_len = len(seq) - int(checkpoint["len"])
                    if rollback_len > 0:
                        self.scheduler.rollback(seq, rollback_len)
                    assert_sequence_matches_checkpoint(seq, checkpoint)
                    rollback_ok_by_seq_id[seq_id] = True
                except BaseException:
                    rollback_ok_by_seq_id[seq_id] = False
                    raise

        proposal_ids = [int(proposal.proposal_id) for proposal in proposals]
        proposal_ids_by_seq_id = {int(proposal.seq_id): int(proposal.proposal_id) for proposal in proposals}
        base_len_by_seq_id = {
            seq_id: int(checkpoint["len"])
            for seq_id, checkpoint in checkpoints.items()
        }
        base_pre_verify_by_seq_id = {
            seq_id: bool(checkpoint["pre_verify"])
            for seq_id, checkpoint in checkpoints.items()
        }
        proposal_len_by_seq_id = {
            int(proposal.seq_id): int(proposal.proposal_len)
            for proposal in proposals
        }
        to_verify_len_by_seq_id = {
            int(proposal.seq_id): len(proposal.to_be_verified_token_ids)
            for proposal in proposals
        }
        discard_reason_by_seq_id = {
            int(seq.seq_id): "phase1h2_dry_run_rollback"
            for seq in seqs
        }
        total_generated = sum(len(tokens) for tokens in generated_by_seq_id.values())

        trace_record["enable_eager_draft_dry_run"] = True
        trace_record["eager_draft_dry_run_enabled"] = True
        trace_record["eager_draft_seq_ids"] = [int(seq.seq_id) for seq in seqs]
        trace_record["eager_draft_proposal_ids"] = proposal_ids
        trace_record["eager_draft_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_draft_base_pre_verify_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_pre_verify_by_seq_id.items()
        }
        trace_record["eager_draft_to_verify_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in to_verify_len_by_seq_id.items()
        }
        trace_record["eager_draft_proposal_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in proposal_len_by_seq_id.items()
        }
        trace_record["eager_draft_rollback_seq_ids"] = rollback_seq_ids
        trace_record["eager_draft_rollback_ok_by_seq_id"] = {
            str(seq_id): ok for seq_id, ok in rollback_ok_by_seq_id.items()
        }
        trace_record["eager_draft_discard_reason_by_seq_id"] = {
            str(seq_id): reason for seq_id, reason in discard_reason_by_seq_id.items()
        }
        trace_record["eager_proposal_ids_by_seq_id"] = {
            str(seq_id): [proposal_id]
            for seq_id, proposal_id in proposal_ids_by_seq_id.items()
        }
        trace_record["eager_tokens_generated"] = total_generated
        trace_record["eager_dry_run_tokens_generated"] = total_generated
        trace_record["eager_buffer_size_before"] = int(buffer_size_before)
        trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()

        if draft_error is not None:
            raise draft_error
        return proposals

    def _parent_discard_reason(
        self,
        proposal: EagerProposal,
        seq: Sequence | None,
        accepted_len: int | None,
        invalidated_len: int | None,
    ) -> str | None:
        if proposal.base_pre_verify:
            return "selected_seq_pre_verify"
        if seq is None or accepted_len is None or invalidated_len is None:
            return "missing_parent_result"
        if seq.is_finished:
            return "parent_finished"
        if getattr(seq, "status", None) != SequenceStatus.RUNNING:
            return "selected_seq_not_running"
        if int(accepted_len) != int(self.gamma) or int(invalidated_len) != 0:
            return "parent_rejected" if int(accepted_len) == 0 else "parent_partial_accept"
        if bool(seq.pre_verify):
            return "parent_pre_verify_after_apply"
        if len(seq) != int(proposal.base_len):
            return "base_len_mismatch"
        return None

    def _evaluate_eager_promotion_dry_run(
        self,
        proposals: list[EagerProposal],
        plan: StepPlan,
        target_seqs: list[Sequence],
        accepted_lens: dict[int, int],
        invalidated_lens: dict[int, int],
        trace_record: dict,
    ) -> None:
        if not proposals:
            return

        seq_by_id = {int(seq.seq_id): seq for seq in target_seqs}
        target_home = {int(seq_id) for seq_id in plan.target_home_set}
        promoted_seq_ids: list[int] = []
        promoted_proposal_ids: list[int] = []
        discarded_seq_ids: list[int] = []
        discarded_proposal_ids: list[int] = []
        parent_seq_ids: list[int] = []
        parent_accepted_len_by_seq_id: dict[int, int] = {}
        parent_invalidated_len_by_seq_id: dict[int, int] = {}
        parent_full_accept_by_seq_id: dict[int, bool] = {}
        parent_finished_by_seq_id: dict[int, bool] = {}
        promotion_reason_by_seq_id: dict[int, str] = {}
        discard_reason_by_seq_id: dict[int, str] = {}
        base_len_by_seq_id: dict[int, int] = {}
        current_len_by_seq_id: dict[int, int] = {}
        base_match_by_seq_id: dict[int, bool] = {}
        promoted_tokens = 0
        discarded_tokens = 0

        for proposal in proposals:
            seq_id = int(proposal.seq_id)
            seq = seq_by_id.get(seq_id)
            accepted_len = accepted_lens.get(seq_id)
            invalidated_len = invalidated_lens.get(seq_id)
            parent_seq_ids.append(seq_id)
            if accepted_len is not None:
                parent_accepted_len_by_seq_id[seq_id] = int(accepted_len)
            if invalidated_len is not None:
                parent_invalidated_len_by_seq_id[seq_id] = int(invalidated_len)
            parent_finished_by_seq_id[seq_id] = bool(seq.is_finished) if seq is not None else False
            base_len_by_seq_id[seq_id] = int(proposal.base_len)
            current_len_by_seq_id[seq_id] = -1 if seq is None else int(len(seq))
            base_match_by_seq_id[seq_id] = seq is not None and int(len(seq)) == int(proposal.base_len)

            if seq_id not in target_home:
                discard_reason = "missing_parent_result"
            else:
                discard_reason = self._parent_discard_reason(
                    proposal,
                    seq,
                    accepted_len,
                    invalidated_len,
                )
            parent_full_accept = discard_reason is None
            parent_full_accept_by_seq_id[seq_id] = parent_full_accept

            if parent_full_accept:
                proposal.state = EAGER_STATE_READY_TO_VERIFY
                promoted_seq_ids.append(seq_id)
                promoted_proposal_ids.append(int(proposal.proposal_id))
                promoted_tokens += int(proposal.proposal_len)
                promotion_reason_by_seq_id[seq_id] = "parent_normal_full_accept"
            else:
                proposal.state = EAGER_STATE_DISCARDED
                proposal.valid = False
                discarded_seq_ids.append(seq_id)
                discarded_proposal_ids.append(int(proposal.proposal_id))
                discarded_tokens += int(proposal.proposal_len)
                discard_reason_by_seq_id[seq_id] = str(discard_reason)

        trace_record["enable_eager_promotion_dry_run"] = True
        trace_record["eager_promotion_dry_run_enabled"] = True
        trace_record["eager_parent_seq_ids"] = parent_seq_ids
        trace_record["eager_parent_accepted_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in parent_accepted_len_by_seq_id.items()
        }
        trace_record["eager_parent_invalidated_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in parent_invalidated_len_by_seq_id.items()
        }
        trace_record["eager_parent_full_accept_by_seq_id"] = {
            str(seq_id): value for seq_id, value in parent_full_accept_by_seq_id.items()
        }
        trace_record["eager_parent_finished_by_seq_id"] = {
            str(seq_id): value for seq_id, value in parent_finished_by_seq_id.items()
        }
        trace_record["eager_promoted_seq_ids"] = promoted_seq_ids
        trace_record["eager_promoted_proposal_ids"] = promoted_proposal_ids
        trace_record["eager_discarded_seq_ids"] = discarded_seq_ids
        trace_record["eager_discarded_proposal_ids"] = discarded_proposal_ids
        trace_record["eager_promotion_reason_by_seq_id"] = {
            str(seq_id): reason for seq_id, reason in promotion_reason_by_seq_id.items()
        }
        trace_record["eager_discard_reason_by_seq_id"] = {
            str(seq_id): reason for seq_id, reason in discard_reason_by_seq_id.items()
        }
        trace_record["eager_promotion_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_promotion_current_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_promotion_base_match_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_match_by_seq_id.items()
        }
        trace_record["eager_tokens_promoted"] = promoted_tokens
        trace_record["eager_tokens_discarded"] = discarded_tokens
        trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()

    def _receive_verify_result(self, seqs: list[Sequence]) -> torch.Tensor:
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
        return verify_res

    def _apply_verify_result(self, seqs: list[Sequence], verify_res: torch.Tensor):
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

    def dual_batch_pearl_step(self):
        plan = self._build_dual_batch_step_plan()
        target_seqs = self._resolve_dual_seq_ids(plan.target_home_set, plan, "draft_apply_verify")
        draft_seqs = self._resolve_dual_seq_ids(plan.draft_home_set, plan, "dual_draft")
        eager_draft_seqs = self._resolve_dual_seq_ids(
            plan.draft_eager_set,
            plan,
            "eager_draft_dry_run",
        )

        proposals = []
        draft_records = []
        eager_proposals = []
        eager_trace_record = None
        if draft_seqs:
            proposals, draft_records = self._draft_dual_batch_proposals(draft_seqs, plan)
            if plan.plan_phase in {"priming", "steady"}:
                self.dual_proposal_buffer.store(proposals)
            for trace_record in draft_records:
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                self._finalize_record_profile(trace_record)
            self._send_dual_proposals(proposals, plan)

        if self._eager_draft_dry_run_enabled() and eager_draft_seqs:
            if draft_records:
                eager_trace_record = draft_records[-1]
            else:
                eager_trace_record = self._trace_dual_batch_schedule([], plan, "eager_draft_dry_run")
            eager_proposals = self._run_eager_draft_dry_run(eager_draft_seqs, plan, eager_trace_record)
            self._finalize_record_profile(eager_trace_record)

        if target_seqs:
            trace_record = self._trace_dual_batch_schedule(target_seqs, plan, "draft_apply_verify")
            trace_record["proposal_tokens_verified"] = self._proposal_verify_token_count(target_seqs)
            trace_record["proposal_tokens_available"] = trace_record["proposal_tokens_verified"]
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            verify_res = self._receive_verify_result(target_seqs)
            accepted_lens, invalidated_lens = self._apply_verify_result(target_seqs, verify_res)
            if self._eager_promotion_dry_run_enabled() and eager_proposals:
                self._evaluate_eager_promotion_dry_run(
                    eager_proposals,
                    plan,
                    target_seqs,
                    accepted_lens,
                    invalidated_lens,
                    eager_trace_record or trace_record,
                )
            consumed_seq_ids = self.dual_proposal_buffer.discard([seq.seq_id for seq in target_seqs])
            trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
            trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
            trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
            torch.cuda.synchronize()
            self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
    
    def pearl_step(self):
        trace_record = None
        for _ in range(self.gamma):
            seqs, is_prefill = self.scheduler.schedule()
            trace_record, step_plan = self._trace_schedule(seqs, is_prefill, "draft")
            seqs = self._resolve_plan_seqs(step_plan, "draft")
            trace_record["resolved_seq_ids"] = [seq.seq_id for seq in seqs]
            trace_record["draft_tokens_generated"] = len(seqs)
            assert not is_prefill, "wrong match. current stage is prefill."
            input_ids, positions = self.prepare_pearl_decode(seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, is_prefill)
            # Currently, the temperature of the draft model is set to 0 to avoid communication overhead.
            # We will support temperature in the future.
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            # append the sample tokens to the seqs. Do not use postprocess to avoid early exiting when the draft tokens contain EOS.
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
            self._mark_trace_end(trace_record)

        accepted_lens, invalidated_lens = self.verify(seqs)
        if trace_record is not None:
            self._update_trace_token_stats(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
            self._finalize_record_profile(trace_record)

    def serialized_pearl_step(self):
        """Serialized-PEARL draft phase.

        Approximation baseline: draft generates with existing PEARL semantics, then
        all ranks synchronize before target verification compute is allowed to run.
        This disables draft/verify overlap without claiming vanilla serial
        speculative decoding equivalence.
        """
        trace_record = None
        for _ in range(self.gamma):
            seqs, is_prefill = self.scheduler.schedule()
            trace_record, step_plan = self._trace_schedule(seqs, is_prefill, "serialized_draft")
            seqs = self._resolve_plan_seqs(step_plan, "serialized_draft")
            trace_record["resolved_seq_ids"] = [seq.seq_id for seq in seqs]
            trace_record["draft_tokens_generated"] = len(seqs)
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
            self._mark_trace_end(trace_record)

        # Global barrier pairs with TargetModelRunner.serialized_pearl_step().
        # It prevents target verification compute from overlapping this draft phase.
        dist.barrier()
        accepted_lens, invalidated_lens = self.verify(seqs)
        if trace_record is not None:
            self._update_trace_token_stats(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
            self._finalize_record_profile(trace_record)

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
        verify_res = self._receive_verify_result(seqs)
        return self._apply_verify_result(seqs, verify_res)


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

    def pearl_step(self):
        seqs, is_prefill = self.scheduler.schedule()
        trace_record, step_plan = self._trace_schedule(seqs, is_prefill, "verify")
        seqs = self._resolve_plan_seqs(step_plan, "verify")
        trace_record["resolved_seq_ids"] = [seq.seq_id for seq in seqs]
        assert not is_prefill, "wrong match. current stage is prefill."
        trace_record["proposal_tokens_available"] = self._proposal_verify_token_count(seqs)
        trace_record["proposal_tokens_verified"] = trace_record["proposal_tokens_available"]
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        accepted_lens, invalidated_lens = self.verify(logits, seqs, temperatures)
        torch.cuda.synchronize()
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def dual_batch_pearl_step(self):
        plan = self._build_dual_batch_step_plan()
        target_seqs = self._resolve_dual_seq_ids(plan.target_home_set, plan, "dual_verify")
        draft_seq_ids = list(plan.draft_home_set)
        target_seq_ids = [seq.seq_id for seq in target_seqs]
        fallback_same_batch = bool(target_seq_ids) and target_seq_ids == draft_seq_ids and plan.plan_phase == "fallback"

        target_proposals = []
        if target_seqs and not fallback_same_batch:
            assert self.dual_proposal_buffer.has_all(target_seq_ids), self._proposal_assertion_message(
                plan,
                f"missing buffered proposals for target seq_ids={target_seq_ids}",
            )
            target_proposals = self.dual_proposal_buffer.get_many(target_seq_ids)

        trace_record = None
        logits = None
        temperatures = None
        if target_seqs:
            self._allocate_decode_slots_for_dual(target_seqs, plan, "dual_verify")
            trace_record = self._trace_dual_batch_schedule(target_seqs, plan, "dual_verify")
            trace_record["proposal_tokens_available"] = sum(len(p.to_be_verified_token_ids) for p in target_proposals)
            trace_record["proposal_tokens_verified"] = trace_record["proposal_tokens_available"]
            input_ids, positions, temp_seqs = self.prepare_pearl_decode(target_seqs)
            temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, False)

        received_proposals = []
        if draft_seq_ids:
            received_proposals = self._receive_dual_proposals(draft_seq_ids, plan)
            if fallback_same_batch:
                target_proposals = received_proposals
                if trace_record is not None:
                    trace_record["proposal_tokens_available"] = sum(len(p.to_be_verified_token_ids) for p in target_proposals)
                    trace_record["proposal_tokens_verified"] = trace_record["proposal_tokens_available"]
            else:
                self.dual_proposal_buffer.store(received_proposals)

        if target_seqs:
            self._validate_proposals_for_target(target_proposals, target_seqs, plan)
            consumed_seq_ids = self.dual_proposal_buffer.discard(target_seq_ids)
            if fallback_same_batch:
                consumed_seq_ids = [proposal.seq_id for proposal in target_proposals]
            trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
            trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
            trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
            accepted_lens, invalidated_lens = self.verify_from_proposals(
                logits,
                target_seqs,
                temperatures,
                target_proposals,
                plan,
            )
            torch.cuda.synchronize()
            self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
        elif received_proposals:
            priming_record = self._trace_dual_batch_schedule([], plan, "dual_verify_idle")
            priming_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
            self._finalize_record_profile(priming_record)

    def serialized_pearl_step(self):
        """Serialized-PEARL target verification phase.

        Approximation baseline: target ranks wait for draft ranks at a global
        barrier before running the existing PEARL verification compute. This
        disables draft/verify overlap but does not implement strict vanilla
        serial speculative decoding.
        """
        # Global barrier pairs with DraftModelRunner.serialized_pearl_step().
        # Do not move this below target compute, or draft/verify will overlap.
        dist.barrier()
        seqs, is_prefill = self.scheduler.schedule()
        trace_record, step_plan = self._trace_schedule(seqs, is_prefill, "serialized_verify")
        seqs = self._resolve_plan_seqs(step_plan, "serialized_verify")
        trace_record["resolved_seq_ids"] = [seq.seq_id for seq in seqs]
        assert not is_prefill, "wrong match. current stage is prefill."
        trace_record["proposal_tokens_available"] = self._proposal_verify_token_count(seqs)
        trace_record["proposal_tokens_verified"] = trace_record["proposal_tokens_available"]
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        accepted_lens, invalidated_lens = self.verify(logits, seqs, temperatures)
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
        return self._verify_from_message(logits, seqs, temperatures, msg, num_to_be_verified_tokens)

    def verify_from_proposals(
        self,
        logits: torch.Tensor,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
        proposals: list[BufferedProposal],
        plan: StepPlan,
    ):
        self._validate_proposals_for_target(proposals, seqs, plan)
        to_be_verified_tokens = []
        next_round_input = []
        for proposal in proposals:
            to_be_verified_tokens.extend(proposal.to_be_verified_token_ids)
            next_round_input.extend(proposal.proposal_token_ids)
        msg = torch.tensor(to_be_verified_tokens + next_round_input, dtype=torch.int64, device="cuda")
        return self._verify_from_message(logits, seqs, temperatures, msg, len(to_be_verified_tokens))

    @torch.inference_mode()
    def _verify_from_message(self, logits: torch.Tensor, seqs: list[Sequence], temperatures: torch.Tensor, msg: torch.Tensor, num_to_be_verified_tokens: int):
        """Refer to the verification logic in the draft model verification function."""
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
