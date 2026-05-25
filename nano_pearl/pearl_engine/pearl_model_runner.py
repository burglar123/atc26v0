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
from nano_pearl.pearl_engine.step_plan import RequestBudget, StepPlan
from nano_pearl.pearl_engine.dual_batch import (
    BufferedProposal,
    DualBatchManager,
    EagerBufferedProposal,
    EagerProposalBuffer,
    ProposalBuffer,
)
from nano_pearl.pearl_engine.proposal_payload import build_combined_proposal_payload
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
        if self.global_config.enable_eager_execution and execution_mode != "dual_batch_pearl":
            raise ValueError("enable_eager_execution is only supported with execution_mode='dual_batch_pearl'")
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
        eager_generated = int(record.get("eager_tokens_generated") or 0)
        eager_discarded = int(record.get("eager_tokens_discarded") or 0)
        eager_rejected = int(record.get("eager_tokens_rejected") or 0)
        eager_invalidated = int(record.get("eager_tokens_invalidated") or 0)
        eager_wasted = eager_discarded + eager_rejected + eager_invalidated
        record["eager_waste_rate"] = eager_wasted / max(1, eager_generated) if eager_generated else None

    def _proposal_verify_token_count(self, seqs: list[Sequence]) -> int:
        return sum(1 if seq.pre_verify else self.gamma for seq in seqs)

    def _has_pending_eager_state(self, seq: Sequence) -> bool:
        return bool(
            getattr(seq, "pending_eager_state", False)
            or getattr(seq, "pending_eager_proposal", False)
            or getattr(seq, "eager_pending", False)
        )

    def _eager_candidate_score(self, seq: Sequence, now: float) -> tuple[float, str]:
        policy = self.global_config.eager_policy
        if policy == "tight_only":
            if getattr(seq, "slo_class", None) == "tight":
                return 1.0, "tight_only:slo_class=tight"
            return 0.0, "tight_only:not_tight"
        if policy == "urgency":
            slo_tpot_ms = getattr(seq, "slo_tpot_ms", None)
            decode_start_ts = getattr(seq, "decode_start_ts", None)
            if slo_tpot_ms is None or float(slo_tpot_ms) <= 0.0 or decode_start_ts is None:
                return 0.0, "urgency:missing_metadata"
            expected_elapsed_ms = seq.num_completion_tokens * float(slo_tpot_ms)
            actual_elapsed_ms = max(0.0, (now - float(decode_start_ts)) * 1000)
            debt_ms = actual_elapsed_ms - expected_elapsed_ms
            return max(0.0, debt_ms), f"urgency:debt_ms={max(0.0, debt_ms):.3f}"
        return 0.0, "policy_none"

    def _annotate_eager_execution_plan(self, plan: StepPlan) -> None:
        plan.enable_eager_execution = bool(self.global_config.enable_eager_execution)
        plan.eager_execution_enabled = bool(self.global_config.enable_eager_execution)
        if not plan.eager_execution_enabled:
            return

        assert self.active_execution_mode == "dual_batch_pearl", self._proposal_assertion_message(
            plan,
            "eager execution is only supported for dual_batch_pearl",
        )
        if plan.plan_phase != "steady":
            return

        ready = set(self.eager_proposal_buffer.ready_seq_ids())
        target_eager_set = [seq_id for seq_id in plan.draft_home_set if seq_id in ready]
        if not target_eager_set:
            return

        target_home = set(plan.target_home_set)
        overlap_home = sorted(target_home & set(target_eager_set))
        assert not overlap_home, self._proposal_assertion_message(
            plan,
            f"target_eager_set overlaps target_home_set: {overlap_home}",
        )
        plan.target_eager_set = list(target_eager_set)
        # Phase 1H-lite: do NOT remove target_eager_set seqs from
        # draft_home_set.  These seqs still need normal proposals
        # generated and sent to the target rank so they are available
        # in the next steady step when the batch rotates.  Eager
        # verification runs after normal drafting in the same step
        # and uses eager_proposal_buffer, not dual_proposal_buffer.
        target_home_size = len(plan.target_home_set)
        draft_home_size = len(plan.draft_home_set)
        plan.target_fraction_of_active = target_home_size / max(1, int(plan.active_seq_count))
        plan.draft_fraction_of_active = draft_home_size / max(1, int(plan.active_seq_count))
        plan.split_imbalance = abs(target_home_size - draft_home_size) / max(1, target_home_size + draft_home_size)
        plan.target_to_draft_size_ratio = target_home_size / max(1, draft_home_size)
        for seq_id in target_eager_set:
            proposal = self.eager_proposal_buffer.get(seq_id)
            assert proposal is not None and proposal.ready, self._proposal_assertion_message(
                plan,
                f"missing ready eager proposal for seq_id={seq_id}",
            )
            plan.budgets.setdefault(seq_id, RequestBudget(normal_gamma=self.gamma, eager_gamma=0))
            plan.budgets[seq_id].eager_gamma = int(proposal.eager_len)
        plan.eager_ready_seq_ids = self.eager_proposal_buffer.ready_seq_ids()

    def _handle_missing_normal_proposals_after_eager(self, plan: StepPlan) -> None:
        if not self.global_config.enable_eager_execution or plan.plan_phase != "steady":
            return

        inspect = self.dual_proposal_buffer.inspect(plan.target_home_set)
        missing = sorted(set(inspect["miss_seq_ids"]) | set(inspect["invalid_seq_ids"]))
        plan.missing_normal_proposal_seq_ids = list(missing)
        if not missing:
            plan.proposal_buffer_requested_seq_ids = inspect["requested_seq_ids"]
            plan.proposal_buffer_hit_seq_ids = inspect["hit_seq_ids"]
            plan.proposal_buffer_miss_seq_ids = inspect["miss_seq_ids"]
            plan.proposal_buffer_invalid_seq_ids = inspect["invalid_seq_ids"]
            plan.proposal_buffer_hit_count = len(plan.proposal_buffer_hit_seq_ids)
            plan.proposal_buffer_miss_count = len(plan.proposal_buffer_miss_seq_ids)
            plan.proposal_buffer_invalid_count = len(plan.proposal_buffer_invalid_seq_ids)
            return

        # Phase 1H-lite: record missing normal proposals for diagnostics,
        # but do NOT rewrite plan.phase / target_home_set / draft_home_set /
        # target_eager_set / draft_eager_set based on local buffer state.
        # Local buffer inspection can differ between draft and target ranks;
        # mutating IPC-relevant plan fields here causes rank divergence.
        # If normal proposals are genuinely missing at verification time,
        # the has_all assertion in the verify path will fire with a clear
        # diagnostic message.
        plan.missing_normal_proposal_reason = (
            "eager_sidecar_skip: missing buffered normal proposals after eager selection"
        )
        plan.normal_proposal_refresh_seq_ids = list(missing)
        plan.fallback_buffer_hit_count = len(inspect["hit_seq_ids"])
        plan.fallback_buffer_miss_count = len(missing)

    def _annotate_eager_trace_plan(self, plan: StepPlan) -> None:
        plan.eager_trace_enabled = bool(self.global_config.enable_eager_trace or self.global_config.enable_eager_execution)
        plan.effective_enable_eager_trace = plan.eager_trace_enabled
        plan.eager_trace_only = bool(plan.eager_trace_enabled and not self.global_config.enable_eager_execution)
        plan.eager_policy = self.global_config.eager_policy
        plan.max_eager_requests_per_step = max(0, int(self.global_config.max_eager_requests_per_step))
        plan.max_eager_tokens_per_step = max(0, int(self.global_config.max_eager_tokens_per_step))
        plan.max_eager_tokens_per_request = max(0, int(self.global_config.max_eager_tokens_per_request))

        if (
            not plan.eager_trace_enabled
            or plan.eager_policy == "none"
            or plan.plan_phase != "steady"
        ):
            return

        # Populate rank-safe per-seq metadata snapshots for every seq in
        # target_home_set BEFORE scoring.  This guarantees the trace always
        # contains metadata diagnostics, even when local scheduler lookup
        # fails on non-leader target ranks under TP>1.
        running_seq_ids = {seq.seq_id for seq in self.scheduler.running}
        target_home_set = set(plan.target_home_set)
        for seq_id in sorted(target_home_set):
            seqs_found = self.scheduler.find_by_seq_ids([seq_id])
            if seqs_found:
                seq = seqs_found[0]
                plan.target_home_request_id_by_seq_id[seq_id] = str(getattr(seq, "request_id", None))
                plan.target_home_slo_class_by_seq_id[seq_id] = str(getattr(seq, "slo_class", None))
                plan.target_home_slo_tpot_ms_by_seq_id[seq_id] = float(getattr(seq, "slo_tpot_ms", None) or 0.0)
                plan.eager_metadata_lookup_source_by_seq_id[seq_id] = "scheduler"
            else:
                plan.missing_eager_metadata_seq_ids.append(int(seq_id))
                plan.eager_metadata_lookup_source_by_seq_id[seq_id] = "missing"

        now = time.time()
        scored = []
        threshold = float(self.global_config.eager_accept_threshold)
        eager_candidate_debug = []
        for seq in self.scheduler.find_by_seq_ids(plan.target_home_set):
            _sid = int(seq.seq_id)
            _cand = {
                "seq_id": _sid,
                "pre_verify": bool(seq.pre_verify),
                "len_seq": int(len(seq)),
                "running": seq.seq_id in running_seq_ids,
                "finished": seq.is_finished,
                "has_pending_eager": self._has_pending_eager_state(seq),
                "in_target_home_set": _sid in target_home_set,
            }
            if seq.seq_id not in running_seq_ids or seq.is_finished or self._has_pending_eager_state(seq):
                _cand["selected"] = False
                _cand["skip_reason"] = "not_running_or_finished_or_pending_eager"
                eager_candidate_debug.append(_cand)
                continue
            # Skip post-verify seqs: DRAFT has already rolled forward
            # and generated the next speculative window while TARGET may
            # still be verifying the previous one.
            if not seq.pre_verify:
                plan.eager_draft_skipped_seq_ids.append(_sid)
                plan.eager_draft_skipped_reason_by_seq_id[_sid] = "skip_post_verify_seq"
                _cand["selected"] = False
                _cand["skip_reason"] = "skip_post_verify_seq"
                eager_candidate_debug.append(_cand)
                continue
            score, reason = self._eager_candidate_score(seq, now)
            if score <= 0.0 or score <= threshold:
                continue
            seq_id = int(seq.seq_id)
            plan.eager_candidate_seq_ids.append(seq_id)
            plan.eager_score_by_seq_id[seq_id] = float(score)
            plan.eager_selection_reason_by_seq_id[seq_id] = reason
            plan.eager_slo_class_by_seq_id[seq_id] = str(getattr(seq, "slo_class", None))
            scored.append((float(score), seq_id))

        if (
            plan.max_eager_requests_per_step <= 0
            or plan.max_eager_tokens_per_step <= 0
            or plan.max_eager_tokens_per_request <= 0
        ):
            return

        remaining_tokens = plan.max_eager_tokens_per_step
        for _, seq_id in sorted(scored, key=lambda item: (-item[0], item[1])):
            if len(plan.eager_selected_seq_ids) >= plan.max_eager_requests_per_step:
                break
            if remaining_tokens <= 0:
                break
            budget = min(plan.max_eager_tokens_per_request, remaining_tokens)
            if budget <= 0:
                break
            plan.eager_selected_seq_ids.append(seq_id)
            plan.eager_budget_by_seq_id[seq_id] = int(budget)
            plan.eager_total_budget += int(budget)
            plan.budgets.setdefault(seq_id, RequestBudget(normal_gamma=self.gamma, eager_gamma=0))
            plan.budgets[seq_id].eager_gamma = int(budget)
            remaining_tokens -= int(budget)

        plan.draft_eager_set = list(plan.eager_selected_seq_ids)
        # Attach per-candidate debug info so the smoke test can verify the filter.
        selected_set = set(plan.eager_selected_seq_ids)
        for _p in scored:
            _p_sid = _p[1]
            _p_entry = next((c for c in eager_candidate_debug if c["seq_id"] == _p_sid), None)
            if _p_entry is None:
                _seq = next((s for s in self.scheduler.find_by_seq_ids([_p_sid])), None)
                _p_entry = {
                    "seq_id": _p_sid,
                    "pre_verify": bool(_seq.pre_verify) if _seq is not None else None,
                    "len_seq": int(len(_seq)) if _seq is not None else -1,
                    "running": True,
                    "finished": False,
                    "has_pending_eager": False,
                }
                eager_candidate_debug.append(_p_entry)
            _p_entry["score"] = _p[0]
            _p_entry["selected"] = _p_sid in selected_set
            _p_entry["skip_reason"] = "" if _p_sid in selected_set else (
                "budget_or_limit_exceeded"
            )
        plan.eager_candidate_debug = eager_candidate_debug

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
        # H2 repair lane migration: move repair_required seqs to repair_lane
        for seq in self.scheduler.running:
            if seq.repair_required and seq.repair_lane_id is not None and seq.repair_lane_id != seq.home_batch_id:
                old_batch = seq.home_batch_id
                self.dual_batch_manager.assign(seq, int(seq.repair_lane_id))
                if old_batch is not None and old_batch in self.dual_batch_manager.batches:
                    if seq.seq_id in self.dual_batch_manager.batches[old_batch].seq_ids:
                        self.dual_batch_manager.batches[old_batch].seq_ids.remove(seq.seq_id)
                seq.home_batch_id = int(seq.repair_lane_id)
        active_seq_ids = [seq.seq_id for seq in self.scheduler.running]
        dropped_normal = self.dual_proposal_buffer.discard_inactive(active_seq_ids)
        dropped_eager = self.eager_proposal_buffer.discard_inactive(active_seq_ids)
        # Discard normal proposals whose home_batch_id no longer matches the
        # seq's current home_batch_id (stale after repair lane migration).
        seq_home = {int(s.seq_id): int(s.home_batch_id) for s in self.scheduler.running if s.home_batch_id is not None}
        stale_normal = []
        for p in self.dual_proposal_buffer.get_many(self.dual_proposal_buffer.pending_seq_ids()):
            current_home = seq_home.get(int(p.seq_id))
            if current_home is not None and int(p.home_batch_id) != current_home:
                stale_normal.append(int(p.seq_id))
        if stale_normal:
            self.dual_proposal_buffer.discard(stale_normal)
            dropped_normal = list(set(dropped_normal) | set(stale_normal))
        return dropped_normal, dropped_eager

    def _build_dual_batch_step_plan(self) -> StepPlan:
        proposal_buffer_size_before = self.dual_proposal_buffer.size()
        eager_buffer_size_before = self.eager_proposal_buffer.size()
        dropped_seq_ids, dropped_eager_seq_ids = self._prepare_dual_batch_state()
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
        )
        buffer_inspect = self.dual_proposal_buffer.inspect(plan.target_home_set)
        plan.proposal_buffer_size_before = int(proposal_buffer_size_before)
        plan.proposal_buffer_size_after = self.dual_proposal_buffer.size()
        plan.proposal_buffer_keys_before_eager_selection = self.dual_proposal_buffer.pending_seq_ids()
        plan.proposal_buffer_requested_seq_ids = buffer_inspect["requested_seq_ids"]
        plan.proposal_buffer_hit_seq_ids = buffer_inspect["hit_seq_ids"]
        plan.proposal_buffer_miss_seq_ids = buffer_inspect["miss_seq_ids"]
        plan.proposal_buffer_invalid_seq_ids = buffer_inspect["invalid_seq_ids"]
        plan.proposal_buffer_dropped_seq_ids = [int(seq_id) for seq_id in dropped_seq_ids]
        plan.proposal_buffer_hit_count = len(plan.proposal_buffer_hit_seq_ids)
        plan.proposal_buffer_miss_count = len(plan.proposal_buffer_miss_seq_ids)
        plan.proposal_buffer_invalid_count = len(plan.proposal_buffer_invalid_seq_ids)
        plan.proposal_buffer_dropped_count = len(plan.proposal_buffer_dropped_seq_ids)
        plan.eager_buffer_size_before = int(eager_buffer_size_before)
        plan.eager_buffer_size_after = self.eager_proposal_buffer.size()
        plan.eager_ready_seq_ids = self.eager_proposal_buffer.ready_seq_ids()
        plan.eager_discarded_seq_ids = [int(seq_id) for seq_id in dropped_eager_seq_ids]
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
        # H2 repair exclusion: exclude repair_required seqs from target_home_set
        # and discard their stale buffered proposals.  Retained proposals
        # pollute pending_batch_ids and prevent batch rotation, causing the
        # next plan to select the same target_batch_id with now-unbuffered seqs.
        repair_excluded = []
        for seq_id in list(plan.target_home_set):
            seqs = self.scheduler.find_by_seq_ids([seq_id])
            if seqs and seqs[0].repair_required:
                plan.target_home_set.remove(seq_id)
                repair_excluded.append(int(seq_id))
                self.dual_proposal_buffer.discard([seq_id])
        if repair_excluded:
            plan.repair_scheduled_seq_ids = repair_excluded
        self._annotate_eager_execution_plan(plan)
        # _annotate_eager_trace_plan MUST run before expected-proposal-set
        # computation: it is the authoritative source for plan.draft_eager_set
        # and applies the post-verify filter (H2 eager sidecar is incompatible
        # with pre_verify=False seqs).  expected_eager_proposal_seq_ids must
        # reflect the filtered draft_eager_set.
        self._annotate_eager_trace_plan(plan)
        # Fail-fast: no post-verify seq in draft_eager_set after filtering.
        if plan.draft_eager_set:
            _running = self.scheduler.running
            _running_by_id = {int(s.seq_id): s for s in _running}
            for _sid in plan.draft_eager_set:
                _seq = _running_by_id.get(int(_sid))
                assert _seq is not None and _seq.pre_verify, self._proposal_assertion_message(
                    plan,
                    f"post-verify seq in draft_eager_set after filter: seq_id={_sid}, "
                    f"pre_verify={getattr(_seq, 'pre_verify', None)}, "
                    f"len_seq={getattr(_seq, '__len__', lambda: -1)() if _seq is not None else -1}",
                )
        plan.authoritative_draft_eager_set_after_filter = list(plan.draft_eager_set)
        # --- H2-aware expected proposal sets ---
        # In H2 steady, seqs in target_eager_set are covered by the eager verify
        # result broadcast (h2_eager_result_bcast) and must be excluded from
        # normal/conditional proposal expectations.  Without this exclusion the
        # receiver-side assertion normal+conditional==draft_home_set fails.
        if plan.plan_phase == "steady" and self.global_config.enable_eager_execution:
            _target_eager = set(plan.target_eager_set)
            plan.expected_eager_proposal_seq_ids = list(plan.draft_eager_set)
            plan.expected_normal_proposal_seq_ids = [
                s for s in plan.draft_home_set if s not in _target_eager
            ]
            plan.expected_conditional_proposal_seq_ids = []
            plan.excluded_normal_proposal_seq_ids = [
                s for s in plan.draft_home_set if s in _target_eager
            ]
            plan.excluded_normal_proposal_reason = (
                "covered_by_target_eager_result" if plan.excluded_normal_proposal_seq_ids else ""
            )
        else:
            plan.expected_eager_proposal_seq_ids = (
                list(plan.draft_eager_set) if self.global_config.enable_eager_execution else []
            )
            plan.expected_normal_proposal_seq_ids = list(plan.draft_home_set)
            plan.expected_conditional_proposal_seq_ids = []
            plan.excluded_normal_proposal_seq_ids = []
            plan.excluded_normal_proposal_reason = ""
        plan.proposal_buffer_keys_after_eager_selection = self.dual_proposal_buffer.pending_seq_ids()
        self._handle_missing_normal_proposals_after_eager(plan)
        # H2 steady invariant: every seq scheduled for normal verify MUST have
        # a buffered proposal.  Remove any that don't (safety filter) and log
        # diagnostics so the root cause can be traced.
        if plan.plan_phase == "steady":
            buffer_keys = set(self.dual_proposal_buffer.pending_seq_ids())
            unbuffered = [s for s in plan.target_home_set if s not in buffer_keys]
            if unbuffered:
                # Build detailed diagnostics before mutating the plan
                seq_home = {int(s.seq_id): int(s.home_batch_id) for s in self.scheduler.running if s.home_batch_id is not None}
                repair_by_seq = {int(s.seq_id): s.repair_required for s in self.scheduler.running}
                eager_by_seq = {int(s.seq_id): getattr(s, "last_eager_result", "none") for s in self.scheduler.running}
                prev_consumed = getattr(plan, "proposal_buffer_consumed_seq_ids", [])
                prev_received = getattr(plan, "received_normal_seq_ids", [])
                logger.warning(
                    "H2 steady target_home_set has seqs without buffered proposals: "
                    "plan_id=%s step_id=%s unbuffered=%s target_home=%s draft_home=%s "
                    "buffer_keys=%s eager_keys=%s home_batches=%s repair=%s eager_result=%s "
                    "prev_consumed=%s prev_received=%s",
                    plan.plan_id, plan.step_id, unbuffered,
                    list(plan.target_home_set), list(plan.draft_home_set),
                    sorted(buffer_keys), self.eager_proposal_buffer.keys(),
                    {str(k): v for k, v in seq_home.items()},
                    {str(k): v for k, v in repair_by_seq.items()},
                    {str(k): v for k, v in eager_by_seq.items()},
                    prev_consumed, prev_received,
                )
                for seq_id in unbuffered:
                    plan.target_home_set.remove(seq_id)
                plan.missing_normal_proposal_seq_ids = list(
                    set(plan.missing_normal_proposal_seq_ids) | set(unbuffered)
                )
            # Debug invariant: after filtering, target_home_set must be subset of buffer
            _post_filter_buffer = set(self.dual_proposal_buffer.pending_seq_ids())
            _post_filter_target = set(plan.target_home_set)
            assert _post_filter_target.issubset(_post_filter_buffer), \
                self._proposal_assertion_message(
                    plan,
                    f"H2 steady invariant violation: target_home_set not subset of buffer_keys: "
                    f"target_home={sorted(_post_filter_target)}, buffer_keys={sorted(_post_filter_buffer)}, "
                    f"unbuffered_in_target={sorted(_post_filter_target - _post_filter_buffer)}",
                )
        if self.global_config.enable_eager_execution:
            plan.validate_phase1h_eager_execution()
        else:
            plan.validate_phase1c()
        return plan

    def _log_h2_steady_lane_transition(self, plan: StepPlan) -> None:
        """Log H2 steady lane transition state at end of step for debugging.

        Records current plan sets, buffer contents, eager outcomes, and
        the expected lane migration due to repair_required flags so that
        target_home_set / buffer consistency can be verified across steps.
        """
        if plan.plan_phase != "steady" or not self.global_config.enable_eager_execution:
            return
        role = "draft" if self.is_draft else "target"
        running = self.scheduler.running
        repair_migrations = []
        for seq in running:
            if seq.repair_required and seq.repair_lane_id is not None and seq.repair_lane_id != seq.home_batch_id:
                repair_migrations.append({
                    "seq_id": int(seq.seq_id),
                    "from_batch": int(seq.home_batch_id) if seq.home_batch_id is not None else -1,
                    "to_batch": int(seq.repair_lane_id),
                    "last_eager_result": getattr(seq, "last_eager_result", "unknown"),
                })
        home_batch_ids = self.dual_batch_manager.home_batch_ids()
        _step_id = plan.step_id if plan.step_id is not None else -1
        logger.info(
            "H2 lane transition: role=%s plan_id=%s step_id=%s "
            "target_home=%s draft_home=%s target_eager=%s draft_eager=%s "
            "buffer_keys=%s eager_keys=%s repair_migrations=%s home_batches=%s",
            role, plan.plan_id, _step_id,
            [int(s) for s in plan.target_home_set], [int(s) for s in plan.draft_home_set],
            [int(s) for s in plan.target_eager_set], [int(s) for s in plan.draft_eager_set],
            self.dual_proposal_buffer.pending_seq_ids(), self.eager_proposal_buffer.keys(),
            repair_migrations, {str(k): int(v) for k, v in home_batch_ids.items()},
        )

    def _sync_h2_plan_fields(self, plan: StepPlan) -> None:
        """Broadcast authoritative H2 plan fields from DRAFT to TARGET.

        DRAFT is the sole authority for fields that determine what proposals
        are generated and sent: draft_eager_set, expected_eager_proposal_seq_ids,
        expected_normal_proposal_seq_ids, excluded_normal_proposal_seq_ids.
        TARGET must not derive these independently from local buffer state.
        """
        if plan.plan_phase != "steady" or not self.global_config.enable_eager_execution:
            return

        src_rank = self.global_config.draft_config.master_rank
        is_source = self.is_draft and self.tp_params.local_rank == 0

        # Serialize four lists: [num_fields, len0, data0..., len1, data1..., ...]
        if is_source:
            fields = [
                [int(s) for s in plan.draft_eager_set],
                [int(s) for s in plan.expected_eager_proposal_seq_ids],
                [int(s) for s in plan.expected_normal_proposal_seq_ids],
                [int(s) for s in plan.excluded_normal_proposal_seq_ids],
            ]
            packed = [len(fields)]
            for f in fields:
                packed.append(len(f))
                packed.extend(f)
        else:
            packed = []

        # Meta broadcast: total packed size
        if is_source:
            meta = torch.tensor([len(packed)], dtype=torch.int64, device="cuda")
        else:
            meta = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=src_rank, group=self.verify_group)
        total = int(meta.item())
        if total == 0:
            return

        # Payload broadcast
        if is_source:
            data = torch.tensor(packed, dtype=torch.int64, device="cuda")
        else:
            data = torch.zeros(total, dtype=torch.int64, device="cuda")
        dist.broadcast(data, src=src_rank, group=self.verify_group)

        # TARGET overwrites plan fields
        if not is_source:
            arr = data.tolist()
            num_fields = arr[0]
            idx = 1
            if num_fields >= 1:
                n = arr[idx]; idx += 1
                plan.draft_eager_set = arr[idx:idx + n]; idx += n
            if num_fields >= 2:
                n = arr[idx]; idx += 1
                plan.expected_eager_proposal_seq_ids = arr[idx:idx + n]; idx += n
            if num_fields >= 3:
                n = arr[idx]; idx += 1
                plan.expected_normal_proposal_seq_ids = arr[idx:idx + n]; idx += n
            if num_fields >= 4:
                n = arr[idx]; idx += 1
                plan.excluded_normal_proposal_seq_ids = arr[idx:idx + n]; idx += n
                plan.excluded_normal_proposal_reason = (
                    "covered_by_target_eager_result" if arr[idx - n:idx] else ""
                )

    def _debug_check_h2_plan_consistency(self, plan: StepPlan) -> None:
        """Fail-fast debug check: all_gather key plan fields and assert equality.

        Activated by TORCH_DISTRIBUTED_DEBUG=DETAIL.  Catches plan divergence
        between DRAFT and TARGET ranks immediately after construction instead
        of letting it cascade into obscure buffer/proposal mismatches later.
        """
        if plan.plan_phase != "steady" or not self.global_config.enable_eager_execution:
            return
        debug_env = os.environ.get("TORCH_DISTRIBUTED_DEBUG", "")
        if debug_env != "DETAIL":
            return

        # Pack a fixed-size tensor: each rank writes its key fields into a
        # row, then all_gather so every rank can compare.
        max_seqs = max(1, len(self.scheduler.running))
        row_len = 4 + 4 * max_seqs  # 4 lengths + 4 padded lists
        local = torch.zeros(row_len, dtype=torch.int64, device="cuda")
        local[0] = len(plan.draft_eager_set)
        local[1] = len(plan.expected_eager_proposal_seq_ids)
        local[2] = len(plan.expected_normal_proposal_seq_ids)
        local[3] = len(plan.excluded_normal_proposal_seq_ids)
        for offset, field in [
            (4, plan.draft_eager_set),
            (4 + max_seqs, plan.expected_eager_proposal_seq_ids),
            (4 + 2 * max_seqs, plan.expected_normal_proposal_seq_ids),
            (4 + 3 * max_seqs, plan.excluded_normal_proposal_seq_ids),
        ]:
            for i, v in enumerate(field):
                if i < max_seqs:
                    local[offset + i] = int(v)

        world = dist.get_world_size(group=self.verify_group)
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local, group=self.verify_group)

        # Compare every pair of ranks — all must be identical
        for i in range(world):
            for j in range(i + 1, world):
                if not torch.equal(gathered[i], gathered[j]):
                    logger.warning(
                        "H2 plan divergence: rank %d != rank %d: "
                        "draft_eager=%s vs %s, expected_normal=%s vs %s",
                        i, j,
                        gathered[i][4:4 + int(gathered[i][0])].tolist(),
                        gathered[j][4:4 + int(gathered[j][0])].tolist(),
                        gathered[i][4 + 2 * max_seqs:4 + 2 * max_seqs + int(gathered[i][2])].tolist(),
                        gathered[j][4 + 2 * max_seqs:4 + 2 * max_seqs + int(gathered[j][2])].tolist(),
                    )

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
            f"{detail}: plan_id={plan.plan_id}, step_id={plan.step_id}, "
            f"plan_phase={plan.plan_phase}, target_home_set={plan.target_home_set}, "
            f"draft_home_set={plan.draft_home_set}, target_eager_set={plan.target_eager_set}, "
            f"draft_eager_set={plan.draft_eager_set}, "
            f"missing_normal_proposal_seq_ids={plan.missing_normal_proposal_seq_ids}, "
            f"missing_normal_proposal_reason={plan.missing_normal_proposal_reason}, "
            f"buffered_proposal_seq_ids="
            f"{self.dual_proposal_buffer.pending_seq_ids()}, eager_buffer_keys={self.eager_proposal_buffer.keys()}"
        )

    def _build_buffered_proposals(self, seqs: list[Sequence], plan: StepPlan,
                                  base_lens: dict[int, int] | None = None) -> list[BufferedProposal]:
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
            base_len = int(base_lens.get(seq.seq_id, len(seq) - self.gamma)) if base_lens else int(len(seq) - self.gamma)
            proposals.append(
                BufferedProposal(
                    seq_id=int(seq.seq_id),
                    request_id=seq.request_id,
                    home_batch_id=int(seq.home_batch_id),
                    proposal_token_ids=[int(x) for x in proposal_tokens],
                    to_be_verified_token_ids=to_be_verified,
                    proposal_len=int(self.gamma),
                    base_len=base_len,
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

    def _serialize_eager_proposals(self, proposals: list[EagerBufferedProposal], plan: StepPlan) -> tuple[torch.Tensor, torch.Tensor]:
        header = []
        tokens = []
        for proposal in proposals:
            header.extend(
                [
                    int(proposal.seq_id),
                    int(proposal.home_batch_id),
                    int(proposal.eager_len),
                    int(proposal.eager_base_len),
                    int(proposal.source_plan_id),
                    int(proposal.source_step_id),
                    int(proposal.source_home_batch_id),
                ]
            )
            tokens.extend(int(token) for token in proposal.eager_token_ids)
        payload = header + tokens
        meta = torch.tensor(
            [
                len(proposals),
                len(payload),
                int(plan.plan_id),
                -1 if plan.step_id is None else int(plan.step_id),
                -1 if plan.target_batch_id is None else int(plan.target_batch_id),
            ],
            dtype=torch.int64,
            device="cuda",
        )
        payload_tensor = torch.tensor(payload, dtype=torch.int64, device="cuda")
        return meta, payload_tensor

    def _send_eager_proposals(self, proposals: list[EagerBufferedProposal], plan: StepPlan):
        if self.tp_params.local_rank != 0:
            return
        meta, payload = self._serialize_eager_proposals(proposals, plan)
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta[1].item()) > 0:
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)

    def _receive_eager_proposals(self, expected_seq_ids: list[int], plan: StepPlan) -> list[EagerBufferedProposal]:
        meta = torch.zeros(5, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        n, payload_len, proposal_plan_id, proposal_step_id, batch_id = [int(x) for x in meta.tolist()]
        payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
        if payload_len > 0:
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        data = payload.tolist()
        header_len = n * 7
        headers = data[:header_len]
        token_data = data[header_len:]
        proposals = []
        token_offset = 0
        seq_lookup = {seq.seq_id: seq for seq in self.scheduler.find_by_seq_ids(expected_seq_ids)} if expected_seq_ids else {}
        for idx in range(n):
            base = idx * 7
            seq_id, home_batch_id, eager_len, eager_base_len, source_plan_id, source_step_id, source_home_batch_id = headers[base:base + 7]
            eager_token_ids = [int(x) for x in token_data[token_offset:token_offset + eager_len]]
            token_offset += eager_len
            seq = seq_lookup.get(seq_id)
            proposals.append(
                EagerBufferedProposal(
                    seq_id=int(seq_id),
                    request_id=seq.request_id if seq is not None else int(seq_id),
                    home_batch_id=int(home_batch_id),
                    eager_token_ids=eager_token_ids,
                    eager_len=int(eager_len),
                    eager_base_len=int(eager_base_len),
                    source_plan_id=int(source_plan_id),
                    source_step_id=int(source_step_id),
                    source_home_batch_id=int(source_home_batch_id),
                    verify_with_batch_id=None if batch_id < 0 else int(batch_id),
                    score=float(plan.eager_score_by_seq_id.get(int(seq_id), 0.0)),
                    policy=plan.eager_policy,
                    valid=True,
                    ready=False,
                )
            )
        received_seq_ids = [proposal.seq_id for proposal in proposals]
        assert set(received_seq_ids).issubset(set(expected_seq_ids)), self._proposal_assertion_message(
            plan,
            f"eager proposal seq_ids not subset of expected: expected={expected_seq_ids}, received={received_seq_ids}, "
            f"source_plan_id={proposal_plan_id}, source_step_id={proposal_step_id}",
        )
        return proposals

    def _combined_normal_payload(self, proposals: list[BufferedProposal]) -> list[int]:
        return build_combined_proposal_payload(
            normal_proposals=proposals,
            eager_proposals=[],
            plan_id=0,
            step_id=None,
            draft_batch_id=None,
            gamma=self.gamma,
        )["normal_payload"]

    def _combined_eager_payload(self, proposals: list[EagerBufferedProposal]) -> list[int]:
        return build_combined_proposal_payload(
            normal_proposals=[],
            eager_proposals=proposals,
            plan_id=0,
            step_id=None,
            draft_batch_id=None,
            gamma=self.gamma,
        )["eager_payload"]

    def _send_combined_dual_proposals(
        self,
        normal_proposals: list[BufferedProposal],
        conditional_normal_proposals: list[BufferedProposal],
        eager_proposals: list[EagerBufferedProposal],
        plan: StepPlan,
        expected_normal_seq_ids: list[int],
        expected_eager_seq_ids: list[int],
    ) -> None:
        payload = build_combined_proposal_payload(
            normal_proposals=normal_proposals,
            conditional_normal_proposals=conditional_normal_proposals,
            eager_proposals=eager_proposals,
            plan_id=plan.plan_id,
            step_id=plan.step_id,
            draft_batch_id=plan.draft_batch_id,
            gamma=self.gamma,
        )
        actual_normal_seq_ids = list(payload["normal_seq_ids"])
        actual_conditional_seq_ids = list(payload["conditional_normal_seq_ids"])
        actual_eager_seq_ids = list(payload["eager_seq_ids"])
        plan.send_expected_normal_seq_ids = list(expected_normal_seq_ids)
        plan.send_actual_normal_seq_ids = list(actual_normal_seq_ids)
        plan.send_expected_eager_seq_ids = list(expected_eager_seq_ids)
        plan.send_actual_eager_seq_ids = list(actual_eager_seq_ids)
        plan.eager_sent_seq_ids = list(actual_eager_seq_ids)
        plan.send_combined_payload_kind = payload["kind"]
        plan.send_combined_payload_plan_id = int(payload["plan_id"])
        plan.send_combined_payload_step_id = None if payload["step_id"] < 0 else int(payload["step_id"])
        if expected_eager_seq_ids and not actual_eager_seq_ids:
            plan.eager_draft_empty_reason = "draft_eager_set_nonempty_but_no_eager_proposals"
        assert payload["kind"] == "combined", self._proposal_assertion_message(
            plan,
            f"send-side payload kind mismatch: expected=combined, actual={payload['kind']}",
        )
        # normal + conditional must be subset of expected (repair seqs intentionally excluded)
        _all_normal_conditional = actual_normal_seq_ids + actual_conditional_seq_ids
        assert set(_all_normal_conditional).issubset(set(expected_normal_seq_ids)), \
            self._proposal_assertion_message(
                plan,
                f"send normal/conditional proposal seq_ids not subset of expected: expected={expected_normal_seq_ids}, "
                f"normal={actual_normal_seq_ids}, conditional={actual_conditional_seq_ids}, "
                f"eager={actual_eager_seq_ids}",
            )
        assert set(actual_eager_seq_ids).issubset(set(expected_eager_seq_ids)), self._proposal_assertion_message(
            plan,
            f"send eager proposal seq_ids not subset of expected: expected={expected_eager_seq_ids}, actual={actual_eager_seq_ids}, "
            f"normal={actual_normal_seq_ids}",
        )
        if expected_eager_seq_ids and not actual_eager_seq_ids:
            plan.eager_draft_empty_reason = "all_eager_proposals_discarded_during_promote_or_discard"
        if self.tp_params.local_rank != 0:
            return
        normal_payload = payload["normal_payload"]
        conditional_payload = payload["conditional_normal_payload"]
        eager_payload = payload["eager_payload"]
        flat_payload = payload["flat_payload"]
        meta = torch.tensor(
            [
                1,  # proposal_message_kind="combined"
                len(flat_payload),
                int(self.gamma),
                int(plan.plan_id),
                -1 if plan.step_id is None else int(plan.step_id),
                -1 if plan.draft_batch_id is None else int(plan.draft_batch_id),
                len(normal_proposals),
                len(normal_payload),
                len(conditional_normal_proposals),
                len(conditional_payload),
                len(eager_proposals),
                len(eager_payload),
            ],
            dtype=torch.int64,
            device="cuda",
        )
        payload_tensor = torch.tensor(flat_payload, dtype=torch.int64, device="cuda")
        self._trace_collective("h2_combined_meta_bcast", plan, prefix="before",
                              tensor_numel=12, tensor_dtype="int64")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        self._trace_collective("h2_combined_meta_bcast", plan, prefix="after",
                              tensor_numel=12, tensor_dtype="int64")
        if int(meta[1].item()) > 0:
            _payload_len = int(meta[1].item())
            self._trace_collective("h2_combined_payload_bcast", plan, prefix="before",
                                  tensor_numel=_payload_len, tensor_dtype="int64")
            dist.broadcast(payload_tensor, src=self.global_config.draft_config.master_rank, group=self.verify_group)
            self._trace_collective("h2_combined_payload_bcast", plan, prefix="after",
                                  tensor_numel=_payload_len, tensor_dtype="int64")

    def _parse_combined_normal_payload(
        self,
        payload: list[int],
        n: int,
        proposal_plan_id: int,
        expected_seq_ids: list[int],
    ) -> list[BufferedProposal]:
        header_len = n * 6
        headers = payload[:header_len]
        token_data = payload[header_len:]
        proposals = []
        token_offset = 0
        seq_lookup = {seq.seq_id: seq for seq in self.scheduler.find_by_seq_ids(expected_seq_ids)} if expected_seq_ids else {}
        for idx in range(n):
            base = idx * 6
            seq_id, home_batch_id, base_len, pre_verify, to_verify_len, proposal_len = headers[base:base + 6]
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
                    base_len=int(base_len),
                    pre_verify=bool(pre_verify),
                    plan_id=int(proposal_plan_id),
                    valid=True,
                )
            )
        return proposals

    def _parse_combined_eager_payload(
        self,
        payload: list[int],
        n: int,
        expected_seq_ids: list[int],
        plan: StepPlan,
    ) -> list[EagerBufferedProposal]:
        header_fields = 8
        header_len = n * header_fields
        headers = payload[:header_len]
        token_data = payload[header_len:]
        proposals = []
        token_offset = 0
        seq_lookup = {seq.seq_id: seq for seq in self.scheduler.find_by_seq_ids(expected_seq_ids)} if expected_seq_ids else {}
        for idx in range(n):
            base = idx * header_fields
            seq_id, home_batch_id, eager_len, eager_base_len, source_plan_id, source_step_id, source_home_batch_id, original_eager_base_len_at_generation = headers[base:base + header_fields]
            eager_token_ids = [int(x) for x in token_data[token_offset:token_offset + eager_len]]
            token_offset += eager_len
            seq = seq_lookup.get(seq_id)
            proposals.append(
                EagerBufferedProposal(
                    seq_id=int(seq_id),
                    request_id=seq.request_id if seq is not None else int(seq_id),
                    home_batch_id=int(home_batch_id),
                    eager_token_ids=eager_token_ids,
                    eager_len=int(eager_len),
                    eager_base_len=int(eager_base_len),
                    source_plan_id=int(source_plan_id),
                    source_step_id=int(source_step_id),
                    source_home_batch_id=int(source_home_batch_id),
                    verify_with_batch_id=None if plan.draft_batch_id is None else int(plan.draft_batch_id),
                    score=float(plan.eager_score_by_seq_id.get(int(seq_id), 0.0)),
                    policy=plan.eager_policy,
                    valid=True,
                    ready=False,
                    original_eager_base_len_at_generation=int(original_eager_base_len_at_generation),
                )
            )
        return proposals

    def _validate_received_eager_seq_ids(
        self,
        received_eager_seq_ids: list[int],
        eager_proposals: list[EagerBufferedProposal],
        plan: StepPlan,
    ) -> str | None:
        """Validate producer-authoritative eager seq_ids on the receive side.

        Returns an error string if validation fails, None if it passes.
        """
        received_set = set(received_eager_seq_ids)
        target_home = set(plan.target_home_set)
        draft_home = set(plan.draft_home_set)
        target_eager = set(plan.target_eager_set)
        running_seq_ids = {seq.seq_id for seq in self.scheduler.running}

        if not received_set.issubset(target_home):
            return (
                f"received_eager_seq_ids must be subset of target_home_set: "
                f"received={sorted(received_set)}, "
                f"outside_target_home={sorted(received_set - target_home)}"
            )
        if received_set & draft_home:
            return (
                f"received_eager_seq_ids must not overlap draft_home_set: "
                f"overlap={sorted(received_set & draft_home)}"
            )
        if received_set & target_eager:
            return (
                f"received_eager_seq_ids must not overlap target_eager_set: "
                f"overlap={sorted(received_set & target_eager)}"
            )
        if len(received_eager_seq_ids) > plan.max_eager_requests_per_step:
            return (
                f"received eager count {len(received_eager_seq_ids)} exceeds "
                f"max_eager_requests_per_step={plan.max_eager_requests_per_step}"
            )

        total_eager_tokens = 0
        for proposal in eager_proposals:
            seq_id = int(proposal.seq_id)
            eager_len = int(proposal.eager_len)
            total_eager_tokens += eager_len
            if eager_len > plan.max_eager_tokens_per_request:
                return (
                    f"eager proposal for seq_id={seq_id} has length {eager_len} "
                    f"above max_eager_tokens_per_request={plan.max_eager_tokens_per_request}"
                )
            if seq_id not in running_seq_ids:
                return f"eager proposal seq_id={seq_id} is not in running set (finished or missing)"
        if total_eager_tokens > plan.max_eager_tokens_per_step:
            return (
                f"total eager tokens {total_eager_tokens} exceeds "
                f"max_eager_tokens_per_step={plan.max_eager_tokens_per_step}"
            )

        return None

    def _receive_combined_dual_proposals(
        self,
        expected_normal_seq_ids: list[int],
        expected_eager_seq_ids: list[int],
        plan: StepPlan,
    ) -> tuple[list[BufferedProposal], list[BufferedProposal], list[EagerBufferedProposal]]:
        meta = torch.zeros(12, dtype=torch.int64, device="cuda")
        self._trace_collective("h2_combined_meta_bcast", plan, prefix="before",
                              tensor_numel=12, tensor_dtype="int64")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        self._trace_collective("h2_combined_meta_bcast", plan, prefix="after",
                              tensor_numel=12, tensor_dtype="int64")
        (
            proposal_kind,
            payload_len,
            gamma,
            proposal_plan_id,
            proposal_step_id,
            batch_id,
            normal_n,
            normal_payload_len,
            conditional_n,
            conditional_payload_len,
            eager_n,
            eager_payload_len,
        ) = [int(x) for x in meta.tolist()]
        assert proposal_kind == 1, self._proposal_assertion_message(
            plan,
            f"proposal message kind mismatch: expected combined=1, got {proposal_kind}",
        )
        assert gamma == int(self.gamma), self._proposal_assertion_message(
            plan,
            f"combined proposal gamma mismatch: expected={self.gamma}, got={gamma}",
        )
        payload_tensor = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
        if payload_len > 0:
            self._trace_collective("h2_combined_payload_bcast", plan, prefix="before",
                                  tensor_numel=int(payload_len), tensor_dtype="int64")
            dist.broadcast(payload_tensor, src=self.global_config.draft_config.master_rank, group=self.verify_group)
            self._trace_collective("h2_combined_payload_bcast", plan, prefix="after",
                                  tensor_numel=int(payload_len), tensor_dtype="int64")
        payload = payload_tensor.tolist()
        normal_payload = payload[:normal_payload_len]
        conditional_start = normal_payload_len
        conditional_payload = payload[conditional_start:conditional_start + conditional_payload_len]
        eager_start = conditional_start + conditional_payload_len
        eager_payload = payload[eager_start:eager_start + eager_payload_len]
        normal_proposals = self._parse_combined_normal_payload(
            normal_payload,
            normal_n,
            proposal_plan_id,
            expected_normal_seq_ids,
        )
        conditional_proposals = self._parse_combined_normal_payload(
            conditional_payload,
            conditional_n,
            proposal_plan_id,
            expected_normal_seq_ids,
        )
        eager_proposals = self._parse_combined_eager_payload(
            eager_payload,
            eager_n,
            expected_eager_seq_ids,
            plan,
        )
        received_normal_seq_ids = [proposal.seq_id for proposal in normal_proposals]
        received_conditional_seq_ids = [proposal.seq_id for proposal in conditional_proposals]
        received_eager_seq_ids = [proposal.seq_id for proposal in eager_proposals]
        plan.expected_normal_receive_seq_ids = list(expected_normal_seq_ids)
        plan.received_normal_seq_ids = list(received_normal_seq_ids)
        plan.received_conditional_normal_seq_ids = list(received_conditional_seq_ids)
        plan.received_eager_seq_ids = list(received_eager_seq_ids)
        plan.proposal_message_kind = "combined"
        plan.proposal_message_plan_id = int(proposal_plan_id)
        plan.proposal_message_step_id = int(proposal_step_id)
        _received_all_normal = received_normal_seq_ids + received_conditional_seq_ids
        _expected_all = list(expected_normal_seq_ids)
        assert _received_all_normal == _expected_all, \
            self._proposal_assertion_message(
                plan,
                f"normal+conditional proposal seq_id mismatch: expected={expected_normal_seq_ids}, "
                f"normal={received_normal_seq_ids}, conditional={received_conditional_seq_ids}, "
                f"received_eager_seq_ids={received_eager_seq_ids}, batch_id={batch_id}, "
                f"proposal_plan_id={proposal_plan_id}, proposal_step_id={proposal_step_id}, "
                f"excluded_normal={plan.excluded_normal_proposal_seq_ids}, "
                f"excluded_reason={plan.excluded_normal_proposal_reason}, "
                f"draft_home_set={plan.draft_home_set}, "
                f"target_eager_set={plan.target_eager_set}",
            )

        # --- eager receive: producer-authoritative for Phase 1H-lite ---
        plan.target_local_expected_eager_seq_ids = list(expected_eager_seq_ids)
        plan.local_plan_draft_eager_set_before_receive = list(plan.draft_eager_set)
        plan.eager_receive_policy = "producer_authoritative"

        if received_eager_seq_ids:
            if not self.global_config.enable_eager_execution:
                plan.eager_receive_validation_passed = False
                plan.eager_receive_validation_error = (
                    "received eager proposals but eager execution is disabled"
                )
                plan.eager_receive_validation_ok = False
                plan.eager_receive_validation_reason = plan.eager_receive_validation_error
                assert False, self._proposal_assertion_message(
                    plan,
                    f"eager receive validation failed: {plan.eager_receive_validation_error}, "
                    f"received={received_eager_seq_ids}",
                )
            if plan.plan_phase != "steady":
                plan.eager_receive_validation_passed = False
                plan.eager_receive_validation_error = (
                    f"received eager proposals outside steady phase: phase={plan.plan_phase}"
                )
                plan.eager_receive_validation_ok = False
                plan.eager_receive_validation_reason = plan.eager_receive_validation_error
                assert False, self._proposal_assertion_message(
                    plan,
                    f"eager receive validation failed: {plan.eager_receive_validation_error}, "
                    f"received={received_eager_seq_ids}",
                )
            validation_error = self._validate_received_eager_seq_ids(
                received_eager_seq_ids, eager_proposals, plan
            )
            if validation_error:
                plan.eager_receive_validation_passed = False
                plan.eager_receive_validation_error = validation_error
                plan.eager_receive_validation_ok = False
                plan.eager_receive_validation_reason = validation_error
                assert False, self._proposal_assertion_message(
                    plan,
                    f"eager receive validation failed: {validation_error}, "
                    f"received={received_eager_seq_ids}",
                )
            plan.eager_receive_validation_passed = True
            plan.eager_receive_validation_ok = True
            plan.eager_receive_validation_reason = "all validations passed"
            plan.draft_eager_set = list(received_eager_seq_ids)
        else:
            # No eager receive event — validation is not applicable.
            plan.eager_receive_validation_ok = None
            plan.eager_receive_validation_reason = "no_eager_receive_event"

        plan.local_plan_draft_eager_set_after_receive = list(plan.draft_eager_set)
        # --- end eager receive ---

        return normal_proposals, conditional_proposals, eager_proposals

    def _apply_combined_proposal_trace_fields(self, trace_record: dict, plan: StepPlan) -> None:
        trace_record["expected_normal_receive_seq_ids"] = list(plan.expected_normal_receive_seq_ids)
        trace_record["received_normal_seq_ids"] = list(plan.received_normal_seq_ids)
        trace_record["received_conditional_normal_seq_ids"] = list(plan.received_conditional_normal_seq_ids)
        trace_record["received_eager_seq_ids"] = list(plan.received_eager_seq_ids)
        trace_record["target_local_expected_eager_seq_ids"] = list(plan.target_local_expected_eager_seq_ids)
        trace_record["eager_receive_policy"] = plan.eager_receive_policy
        trace_record["eager_receive_validation_passed"] = bool(plan.eager_receive_validation_passed)
        trace_record["eager_receive_validation_error"] = plan.eager_receive_validation_error
        trace_record["eager_receive_validation_ok"] = plan.eager_receive_validation_ok
        trace_record["eager_receive_validation_reason"] = plan.eager_receive_validation_reason
        trace_record["local_plan_draft_eager_set_before_receive"] = list(plan.local_plan_draft_eager_set_before_receive)
        trace_record["local_plan_draft_eager_set_after_receive"] = list(plan.local_plan_draft_eager_set_after_receive)
        trace_record["proposal_message_kind"] = plan.proposal_message_kind
        trace_record["proposal_message_plan_id"] = plan.proposal_message_plan_id
        trace_record["proposal_message_step_id"] = plan.proposal_message_step_id
        trace_record["normal_proposal_buffer_keys_after_receive"] = self.dual_proposal_buffer.pending_seq_ids()
        trace_record["eager_buffer_keys_after_receive"] = self.eager_proposal_buffer.keys()

    def _apply_combined_send_trace_fields(self, trace_record: dict, plan: StepPlan) -> None:
        trace_record["send_expected_normal_seq_ids"] = list(plan.send_expected_normal_seq_ids)
        trace_record["send_actual_normal_seq_ids"] = list(plan.send_actual_normal_seq_ids)
        trace_record["send_expected_eager_seq_ids"] = list(plan.send_expected_eager_seq_ids)
        trace_record["send_actual_eager_seq_ids"] = list(plan.send_actual_eager_seq_ids)
        trace_record["send_combined_payload_kind"] = plan.send_combined_payload_kind
        trace_record["send_combined_payload_plan_id"] = plan.send_combined_payload_plan_id
        trace_record["send_combined_payload_step_id"] = plan.send_combined_payload_step_id
        trace_record["eager_draft_skipped_reason"] = plan.eager_draft_skipped_reason
        trace_record["eager_draft_failed_seq_ids"] = list(plan.eager_draft_failed_seq_ids)
        trace_record["eager_draft_empty_reason"] = plan.eager_draft_empty_reason

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

    def _validate_eager_proposals(self, proposals: list[EagerBufferedProposal], seqs: list[Sequence], plan: StepPlan):
        proposal_seq_ids = [proposal.seq_id for proposal in proposals]
        seq_ids = [seq.seq_id for seq in seqs]
        assert proposal_seq_ids == seq_ids, self._proposal_assertion_message(
            plan,
            f"eager proposal seq_id mismatch: expected={seq_ids}, got={proposal_seq_ids}",
        )
        running_seq_ids = {seq.seq_id for seq in self.scheduler.running}
        for proposal, seq in zip(proposals, seqs):
            assert proposal.valid and proposal.ready and not proposal.consumed, self._proposal_assertion_message(
                plan,
                f"eager proposal is not ready/valid for seq_id={seq.seq_id}",
            )
            assert seq.seq_id in running_seq_ids, self._proposal_assertion_message(
                plan,
                f"eager proposal used for inactive seq_id={seq.seq_id}",
            )
            assert int(proposal.eager_base_len) == int(len(seq)), self._proposal_assertion_message(
                plan,
                f"eager proposal base length mismatch for seq_id={seq.seq_id}: "
                f"proposal_base={proposal.eager_base_len}, current_len={len(seq)}, "
                f"eager_len={proposal.eager_len}, "
                f"home_batch_id={proposal.home_batch_id}, "
                f"source_plan_id={proposal.source_plan_id}, "
                f"source_step_id={proposal.source_step_id}, "
                f"source_home_batch_id={proposal.source_home_batch_id}, "
                f"eager_base_len_fixups={getattr(plan, 'eager_base_len_fixups', None)}",
            )
            assert int(proposal.eager_len) == len(proposal.eager_token_ids), self._proposal_assertion_message(
                plan,
                f"eager proposal length mismatch for seq_id={seq.seq_id}",
            )
            assert int(proposal.eager_len) <= int(self.gamma), self._proposal_assertion_message(
                plan,
                f"Phase 1H-lite eager_len must be <= gamma for seq_id={seq.seq_id}: "
                f"eager_len={proposal.eager_len}, gamma={self.gamma}",
            )

    def _promote_or_discard_eager_after_normal(
        self,
        plan: StepPlan,
        accepted_lens: dict[int, int],
        invalidated_lens: dict[int, int],
        trace_record: dict | None,
        count_tokens: bool,
    ) -> None:
        if not self.global_config.enable_eager_execution or not plan.draft_eager_set:
            return
        running_seq_ids = {seq.seq_id for seq in self.scheduler.running}
        promoted_seq_ids = []
        discarded_seq_ids = []
        promoted_tokens = 0
        discarded_tokens = 0
        for seq_id in plan.draft_eager_set:
            proposal = self.eager_proposal_buffer.get(seq_id)
            assert proposal is not None, self._proposal_assertion_message(
                plan,
                f"missing eager proposal for selected seq_id={seq_id}",
            )
            in_normal_verify = int(seq_id) in accepted_lens or int(seq_id) in invalidated_lens
            if not in_normal_verify:
                full_accept = int(seq_id) in running_seq_ids
            else:
                full_accept = (
                    int(seq_id) in running_seq_ids
                    and int(invalidated_lens.get(seq_id, 0)) == 0
                    and int(accepted_lens.get(seq_id, 0)) > 0
                )
            if full_accept:
                self.eager_proposal_buffer.mark_ready(seq_id)
                promoted_seq_ids.append(int(seq_id))
                promoted_tokens += int(proposal.eager_len)
            else:
                self.eager_proposal_buffer.discard([seq_id])
                discarded_seq_ids.append(int(seq_id))
                discarded_tokens += int(proposal.eager_len)

        if trace_record is not None:
            trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()
            trace_record["eager_promoted_seq_ids"] = promoted_seq_ids
            trace_record["eager_discarded_seq_ids"] = discarded_seq_ids
            if count_tokens:
                trace_record["eager_tokens_promoted"] = promoted_tokens
                trace_record["eager_tokens_discarded"] = discarded_tokens
            self._finalize_record_profile(trace_record)

    def _receive_eager_verify_result(self, seqs: list[Sequence], *, group=None) -> torch.Tensor:
        # Source-authoritative meta+payload protocol: receiver allocates based on
        # the broadcast meta, NOT on local len(seqs).  This prevents shape mismatch
        # when rank0 and rank1 disagree on target_eager_set contents.
        meta = torch.zeros(3, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=group)
        rows, cols, numel = [int(x) for x in meta.tolist()]
        if numel > 0:
            verify_res = torch.empty((rows, cols), dtype=torch.int64, device="cuda")
            dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=group)
        else:
            verify_res = torch.empty((rows, cols), dtype=torch.int64, device="cuda")
        return verify_res

    def _apply_eager_verify_result(
        self,
        seqs: list[Sequence],
        proposals: list[EagerBufferedProposal],
        verify_res: torch.Tensor,
        plan: StepPlan,
        trace_record: dict | None,
        count_tokens: bool,
    ) -> None:
        if not seqs:
            return
        self._validate_eager_proposals(proposals, seqs, plan)
        seq_ids, accepted, eager_lens, accepted_lens, rejected_lens, finish = verify_res.tolist()
        verified_seq_ids = []
        accepted_seq_ids = []
        rejected_seq_ids = []
        verified_tokens = 0
        accepted_tokens = 0
        rejected_tokens = 0
        invalidated_tokens = 0
        for idx, (seq, proposal) in enumerate(zip(seqs, proposals)):
            assert int(seq_ids[idx]) == int(seq.seq_id), self._proposal_assertion_message(
                plan,
                f"eager verify result seq_id mismatch for seq_id={seq.seq_id}, got={seq_ids[idx]}",
            )
            assert int(eager_lens[idx]) == int(proposal.eager_len), self._proposal_assertion_message(
                plan,
                f"eager verify result length mismatch for seq_id={seq.seq_id}",
            )
            verified_seq_ids.append(int(seq.seq_id))
            verified_tokens += int(proposal.eager_len)
            if int(accepted[idx]):
                assert int(proposal.eager_base_len) == int(len(seq)), self._proposal_assertion_message(
                    plan,
                    f"eager apply base length mismatch for seq_id={seq.seq_id}: "
                    f"proposal_base={proposal.eager_base_len}, current_len={len(seq)}, "
                    f"eager_len={proposal.eager_len}, "
                    f"home_batch_id={proposal.home_batch_id}, "
                    f"source_plan_id={proposal.source_plan_id}, "
                    f"source_step_id={proposal.source_step_id}, "
                    f"source_home_batch_id={proposal.source_home_batch_id}, "
                    f"eager_base_len_fixups={getattr(plan, 'eager_base_len_fixups', None)}",
                )
                for token_id in proposal.eager_token_ids:
                    seq.append_token(int(token_id))
                seq.cur_acc_tokens += int(proposal.eager_len)
                seq.record_accepted(int(accepted_lens[idx]))
                accepted_seq_ids.append(int(seq.seq_id))
                accepted_tokens += int(accepted_lens[idx])
                if int(finish[idx]):
                    seq.mark_finished()
                    seq.num_acc_tokens.append(seq.cur_acc_tokens)
                    if seq in self.scheduler.running:
                        self.scheduler.block_manager.deallocate(seq)
                        self.scheduler.running.remove(seq)
                        self.scheduler.finished.append(seq)
            else:
                seq.record_invalidated_predraft(int(rejected_lens[idx]))
                rejected_seq_ids.append(int(seq.seq_id))
                rejected_tokens += int(rejected_lens[idx])
                invalidated_tokens += int(rejected_lens[idx])
        self.eager_proposal_buffer.discard([seq.seq_id for seq in seqs])
        if trace_record is not None:
            trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()
            trace_record["eager_verified_seq_ids"] = verified_seq_ids
            trace_record["eager_accepted_seq_ids"] = accepted_seq_ids
            trace_record["eager_rejected_seq_ids"] = rejected_seq_ids
            if count_tokens:
                trace_record["eager_tokens_verified"] = verified_tokens
                trace_record["eager_tokens_accepted"] = accepted_tokens
                trace_record["eager_tokens_rejected"] = rejected_tokens
                trace_record["eager_tokens_invalidated"] = invalidated_tokens
            self._finalize_record_profile(trace_record)

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

    def _trace_collective(
        self,
        event: str,
        plan: StepPlan,
        *,
        prefix: str = "before",
        tensor_numel: int = 0,
        tensor_dtype: str = "int64",
        group_label: str = "verify_group",
        eager_result_empty: bool = False,
        normal_seq_ids: list[int] | None = None,
        conditional_normal_seq_ids: list[int] | None = None,
        eager_seq_ids: list[int] | None = None,
        meta_rows: int | None = None,
        meta_cols: int | None = None,
        meta_numel: int | None = None,
        local_pre_meta_target_eager_set: list[int] | None = None,
    ) -> None:
        """Log collective trace for H2 NCCL ordering debugging.

        Each call site pairs a BEFORE log (prefix=\"before\") with an AFTER log
        (prefix=\"after\") around the dist.broadcast, with handler flush so the
        record survives a subsequent hang.
        """
        try:
            tag = f"[H2_COLLECTIVE_{prefix.upper()}]"
            src_rank = (
                self.global_config.target_config.master_rank
                if "eager_result" in event or "normal_result" in event
                else self.global_config.draft_config.master_rank
            )
            meta_parts = []
            if meta_rows is not None:
                meta_parts.append(f"meta_rows={meta_rows}")
            if meta_cols is not None:
                meta_parts.append(f"meta_cols={meta_cols}")
            if meta_numel is not None:
                meta_parts.append(f"meta_numel={meta_numel}")
            if local_pre_meta_target_eager_set is not None:
                meta_parts.append(f"local_pre_meta_target_eager_set={local_pre_meta_target_eager_set}")
            meta_str = " ".join(meta_parts)
            logger.info(
                f"{tag} {event} rank={self.rank} step={plan.step_id} "
                f"plan={plan.plan_id} phase={plan.plan_phase} src={src_rank} "
                f"group={group_label} tensor_numel={int(tensor_numel)} "
                f"tensor_dtype={tensor_dtype} "
                f"target_home_set={[int(s) for s in plan.target_home_set]} "
                f"draft_home_set={[int(s) for s in plan.draft_home_set]} "
                f"target_eager_set={[int(s) for s in plan.target_eager_set]} "
                f"draft_eager_set={[int(s) for s in plan.draft_eager_set]}"
                f"{' ' + meta_str if meta_str else ''}"
                f"{' eager_result_empty=True' if eager_result_empty else ''}"
                f"{' normal_seq_ids=' + str([int(s) for s in normal_seq_ids]) if normal_seq_ids is not None else ''}"
                f"{' conditional_seq_ids=' + str([int(s) for s in conditional_normal_seq_ids]) if conditional_normal_seq_ids is not None else ''}"
                f"{' eager_seq_ids=' + str([int(s) for s in eager_seq_ids]) if eager_seq_ids is not None else ''}",
                color="cyan",
            )
            for handler in logger.handlers:
                try:
                    handler.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _debug_check_eager_bcast_consistency(self, plan: StepPlan, local_cols: int) -> None:
        """Debug-only gather of both ranks' target_eager_set before eager broadcast.

        Uses a fixed-size all_gather so it cannot itself cause a shape mismatch.
        Only active when TORCH_DISTRIBUTED_DEBUG=DETAIL is set.
        """
        import os
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG", "") != "DETAIL":
            return
        try:
            max_sets = 32
            tensor = torch.zeros(3 + max_sets, dtype=torch.int64, device="cuda")
            tensor[0] = int(plan.step_id)
            tensor[1] = int(plan.plan_id)
            tensor[2] = int(local_cols)
            for i, sid in enumerate(sorted(plan.target_eager_set)[:max_sets]):
                tensor[3 + i] = int(sid)
            gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, tensor)
            rows_data = []
            for rank_idx, t in enumerate(gathered):
                tlist = t.tolist()
                rows_data.append({
                    "rank": rank_idx,
                    "step_id": int(tlist[0]),
                    "plan_id": int(tlist[1]),
                    "cols": int(tlist[2]),
                    "target_eager_set": [int(x) for x in tlist[3:] if int(x) != 0],
                })
            cols_vals = sorted({r["cols"] for r in rows_data})
            if len(cols_vals) > 1:
                logger.warning(
                    f"[H2_EAGER_DEBUG] eager broadcast shape MISMATCH detected: "
                    f"per_rank={rows_data}",
                    color="red",
                )
                for handler in logger.handlers:
                    try:
                        handler.flush()
                    except Exception:
                        pass
        except Exception:
            pass

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

    def _draft_eager_proposals(self, seqs: list[Sequence], plan: StepPlan) -> tuple[list[EagerBufferedProposal], dict | None]:
        if not self.global_config.enable_eager_execution or not seqs:
            return [], None

        trace_record = self._trace_dual_batch_schedule(seqs, plan, "dual_eager_draft")
        budgets = {seq.seq_id: int(plan.eager_budget_by_seq_id.get(seq.seq_id, 0)) for seq in seqs}
        base_lens = {seq.seq_id: len(seq) for seq in seqs}
        generated: dict[int, list[int]] = {seq.seq_id: [] for seq in seqs}

        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        trace_record["eager_draft_start_ts"] = trace_record.get("draft_start_ts")
        max_budget = max(budgets.values(), default=0)
        for token_idx in range(max_budget):
            active = [seq for seq in seqs if token_idx < budgets[seq.seq_id]]
            if not active:
                continue
            self._allocate_decode_slots_for_dual(active, plan, "dual_eager_draft")
            input_ids, positions = self.prepare_pearl_decode(active)
            logits = self.run_model(input_ids, positions, False)
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(active), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)
            for seq, token_id in zip(active, token_ids):
                token_id = int(token_id)
                seq.append_token(token_id)
                generated[seq.seq_id].append(token_id)

        proposals = []
        for seq in seqs:
            eager_token_ids = generated[seq.seq_id]
            eager_len = len(eager_token_ids)
            assert eager_len == budgets[seq.seq_id], self._proposal_assertion_message(
                plan,
                f"eager generated length mismatch for seq_id={seq.seq_id}: generated={eager_len}, budget={budgets[seq.seq_id]}",
            )
            proposals.append(
                EagerBufferedProposal(
                    seq_id=int(seq.seq_id),
                    request_id=seq.request_id,
                    home_batch_id=int(seq.home_batch_id),
                    eager_token_ids=eager_token_ids,
                    eager_len=int(eager_len),
                    eager_base_len=int(base_lens[seq.seq_id]),
                    source_plan_id=int(plan.plan_id),
                    source_step_id=-1 if plan.step_id is None else int(plan.step_id),
                    source_home_batch_id=-1 if plan.target_batch_id is None else int(plan.target_batch_id),
                    verify_with_batch_id=None if plan.draft_batch_id is None else int(plan.draft_batch_id),
                    score=float(plan.eager_score_by_seq_id.get(seq.seq_id, 0.0)),
                    policy=plan.eager_policy,
                    valid=True,
                    ready=False,
                    original_eager_base_len_at_generation=int(base_lens[seq.seq_id]),
                )
            )

        for seq in seqs:
            if generated[seq.seq_id]:
                self.scheduler.rollback(seq, len(generated[seq.seq_id]))
            assert len(seq) == base_lens[seq.seq_id], self._proposal_assertion_message(
                plan,
                f"eager draft rollback failed for seq_id={seq.seq_id}: expected_len={base_lens[seq.seq_id]}, got={len(seq)}",
            )
        # Critical invariant: original_eager_base_len_at_generation == len(seq)
        # at the moment the snapshot was taken. This must hold for every proposal.
        for proposal in proposals:
            _gen_len = base_lens[proposal.seq_id]
            assert int(proposal.original_eager_base_len_at_generation) == int(_gen_len), \
                self._proposal_assertion_message(
                    plan,
                    f"original_eager_base_len_at_generation invariant violated for seq_id={proposal.seq_id}: "
                    f"stored={proposal.original_eager_base_len_at_generation}, "
                    f"base_lens_at_generation={_gen_len}",
                )

        torch.cuda.synchronize()
        self._mark_trace_end(trace_record)
        trace_record["eager_draft_end_ts"] = trace_record.get("draft_end_ts")
        if trace_record["eager_draft_start_ts"] is not None and trace_record["eager_draft_end_ts"] is not None:
            trace_record["eager_draft_time_ms"] = (
                trace_record["eager_draft_end_ts"] - trace_record["eager_draft_start_ts"]
            ) * 1000
        trace_record["eager_tokens_generated"] = sum(len(tokens) for tokens in generated.values())
        trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()
        self._finalize_record_profile(trace_record)
        return proposals, trace_record

    def _receive_verify_result(self, seqs: list[Sequence], *, group=None) -> torch.Tensor:
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=group)
        return verify_res

    @staticmethod
    def _classify_len_delta(delta: int, acc: bool | None, was_pre_verify: bool | None, gamma: int) -> str:
        """Classify seq length delta from normal verify into semantic categories."""
        if delta == 0:
            if acc is True:
                return "full_accept_no_bonus_draft_side"
            elif acc is False:
                return "reject_no_net_change"
            return "no_change_unknown_acc"
        if delta > 0:
            if acc is True and delta == gamma and was_pre_verify is False:
                return "full_accept_bonus_tokens_target_side"
            if acc is True and delta == 1 and was_pre_verify is True:
                return "pre_verify_full_accept_single_token"
            if acc is False:
                return f"partial_reject_net_positive_delta_{delta}"
            return f"positive_delta_{delta}_acc_{acc}_pre_verify_{was_pre_verify}"
        if delta < 0:
            if acc is False:
                return f"reject_with_rollback_delta_{delta}"
            return f"negative_delta_{delta}_unexpected_acc_{acc}"
        return f"unclassified_delta_{delta}"

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

    def _draft_conditional_after_eager(
        self, seqs: list[Sequence], plan: StepPlan
    ) -> tuple[list[BufferedProposal], list[dict]]:
        """Re-draft normal proposals from post-eager prefix for overlap seqs after eager full-accept."""
        if not seqs:
            return [], []
        base_lens = {seq.seq_id: len(seq) for seq in seqs}
        draft_records = []
        for _ in range(self.gamma):
            self._allocate_decode_slots_for_dual(seqs, plan, "dual_conditional_draft")
            trace_record = self._trace_dual_batch_schedule(seqs, plan, "dual_conditional_draft")
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
        proposals = self._build_buffered_proposals(seqs, plan, base_lens=base_lens)
        for trace_record in draft_records:
            self._finalize_record_profile(trace_record)
        return proposals, draft_records

    def dual_batch_pearl_step(self):
        plan = self._build_dual_batch_step_plan()
        self._sync_h2_plan_fields(plan)
        self._debug_check_h2_plan_consistency(plan)
        target_seqs = self._resolve_dual_seq_ids(plan.target_home_set, plan, "draft_apply_verify")
        target_eager_seqs = self._resolve_dual_seq_ids(plan.target_eager_set, plan, "draft_apply_eager_verify")
        draft_seqs = self._resolve_dual_seq_ids(plan.draft_home_set, plan, "dual_draft")
        draft_eager_seqs = self._resolve_dual_seq_ids(plan.draft_eager_set, plan, "dual_eager_draft") if self.global_config.enable_eager_execution else []

        # Debug: log all skipped eager candidates with reason and seq state.
        _skipped = list(plan.eager_draft_skipped_seq_ids)
        if _skipped:
            _running_map = {int(s.seq_id): s for s in self.scheduler.running}
            plan.eager_skip_debug = []
            for _sid in _skipped:
                _seq = _running_map.get(int(_sid))
                plan.eager_skip_debug.append({
                    "seq_id": int(_sid),
                    "pre_verify": bool(_seq.pre_verify) if _seq is not None else None,
                    "len_seq": int(len(_seq)) if _seq is not None else -1,
                    "skip_reason": plan.eager_draft_skipped_reason_by_seq_id.get(int(_sid), "unknown"),
                })

        is_normal_refresh_fallback = plan.plan_phase == "fallback" and bool(plan.normal_proposal_refresh_seq_ids)
        # Draft-side validation: ALL draft_home_set seqs get normal proposals drafted.
        # H2 steady may later filter out overlap seqs before sending, but the draft
        # step itself covers every seq in draft_home_set.
        expected_normal_seq_ids = (
            list(plan.normal_proposal_refresh_seq_ids)
            if is_normal_refresh_fallback
            else list(plan.draft_home_set)
        )
        expected_eager_seq_ids = (
            []
            if is_normal_refresh_fallback
            else list(plan.draft_eager_set) if self.global_config.enable_eager_execution else []
        )
        normal_seq_ids = [int(seq.seq_id) for seq in draft_seqs]
        eager_seq_ids = [int(seq.seq_id) for seq in draft_eager_seqs]
        assert normal_seq_ids == expected_normal_seq_ids, self._proposal_assertion_message(
            plan,
            f"draft normal seq_id mismatch before send packaging: expected={expected_normal_seq_ids}, actual={normal_seq_ids}",
        )
        assert eager_seq_ids == expected_eager_seq_ids, self._proposal_assertion_message(
            plan,
            f"draft eager seq_id mismatch before send packaging: expected={expected_eager_seq_ids}, actual={eager_seq_ids}",
        )

        # H2: detect overlap between target_eager_set and draft_home_set (steady only)
        h2_steady = plan.plan_phase == "steady" and self.global_config.enable_eager_execution
        target_eager_seq_ids = [int(seq.seq_id) for seq in target_eager_seqs]
        draft_seq_ids = [int(seq.seq_id) for seq in draft_seqs]
        overlap_seq_ids = sorted(set(target_eager_seq_ids) & set(draft_seq_ids))
        if overlap_seq_ids:
            plan.target_eager_draft_home_overlap_seq_ids = overlap_seq_ids
        overlap_set = set(overlap_seq_ids)

        # === Phase 1: Normal draft (local) ===
        normal_proposals = []
        draft_records = []
        if draft_seqs:
            normal_proposals, draft_records = self._draft_dual_batch_proposals(draft_seqs, plan)
            # H2: split normal proposals — exclude overlap seqs from buffer
            if h2_steady:
                store_proposals = [p for p in normal_proposals if p.seq_id not in overlap_set]
            else:
                store_proposals = normal_proposals
            if plan.plan_phase in {"priming", "steady"}:
                self.dual_proposal_buffer.store(store_proposals)
            for trace_record in draft_records:
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                self._finalize_record_profile(trace_record)

        # === Phase 2: Eager draft (local) ===
        eager_proposals = []
        eager_trace_record = None
        if draft_eager_seqs and self.global_config.enable_eager_execution:
            plan.eager_draft_attempted_seq_ids = list(plan.draft_eager_set)
            plan.eager_buffer_keys_before_eager_draft = self.eager_proposal_buffer.keys()
            _before_keys = self.dual_proposal_buffer.pending_seq_ids()
            _before_has = self.dual_proposal_buffer.has_all(plan.target_home_set)
            assert _before_has, self._proposal_assertion_message(
                plan,
                f"dual_proposal_buffer missing target_home_set proposals BEFORE eager draft: "
                f"target_home_set={plan.target_home_set}, buffer_keys={_before_keys}",
            )

            eager_proposals, eager_trace_record = self._draft_eager_proposals(draft_eager_seqs, plan)
            self.eager_proposal_buffer.store(eager_proposals)

            _generated_eager_seq_ids = sorted(p.seq_id for p in eager_proposals)
            _expected_eager_seq_ids = sorted(plan.draft_eager_set)
            plan.eager_draft_generated_seq_ids = _generated_eager_seq_ids
            assert _generated_eager_seq_ids == _expected_eager_seq_ids, self._proposal_assertion_message(
                plan,
                f"eager draft generated seq_ids mismatch: "
                f"expected={_expected_eager_seq_ids}, generated={_generated_eager_seq_ids}",
            )

            _after_keys = self.dual_proposal_buffer.pending_seq_ids()
            _after_has = self.dual_proposal_buffer.has_all(plan.target_home_set)
            assert _after_has, self._proposal_assertion_message(
                plan,
                f"dual_proposal_buffer missing target_home_set proposals AFTER eager draft: "
                f"target_home_set={plan.target_home_set}, "
                f"draft_eager_set={plan.draft_eager_set}, "
                f"draft_home_set={plan.draft_home_set}, "
                f"buffer_keys_before={_before_keys}, "
                f"buffer_keys_after={_after_keys}, "
                f"eager_buffer_keys_before={self.eager_proposal_buffer.keys()}",
            )

            if eager_trace_record is not None:
                eager_trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()
                eager_trace_record["proposal_buffer_keys_after_eager_draft"] = self.dual_proposal_buffer.pending_seq_ids()
                # Per-proposal debug instrumentation: capture full lifecycle state
                _draft_eager_seq_map = {int(s.seq_id): s for s in draft_eager_seqs}
                eager_trace_record["eager_proposals_generated"] = [
                    {
                        "seq_id": int(p.seq_id),
                        "step_id": int(plan.step_id) if plan.step_id is not None else -1,
                        "plan_id": int(plan.plan_id),
                        "len_seq_before_generation": int(len(_draft_eager_seq_map.get(int(p.seq_id)))) if int(p.seq_id) in _draft_eager_seq_map else -1,
                        "eager_base_len_written": int(p.eager_base_len),
                        "original_eager_base_len_at_generation": int(p.original_eager_base_len_at_generation),
                        "eager_len": int(p.eager_len),
                        "home_batch_id": int(p.home_batch_id),
                        "source_home_batch_id": int(p.source_home_batch_id),
                        "normal_verify_already_applied": False,
                        "seq_pre_verify": bool(_draft_eager_seq_map.get(int(p.seq_id)).pre_verify) if int(p.seq_id) in _draft_eager_seq_map else None,
                    }
                    for p in eager_proposals
                ]
                self._finalize_record_profile(eager_trace_record)
        elif plan.draft_eager_set and not self.global_config.enable_eager_execution:
            plan.eager_draft_skipped_reason = "eager_execution_disabled"
            plan.eager_draft_failed_seq_ids = list(plan.draft_eager_set)
            plan.eager_draft_skipped_seq_ids = list(plan.draft_eager_set)
            plan.eager_draft_skipped_reason_by_seq_id = {
                int(seq_id): "eager_execution_disabled" for seq_id in plan.draft_eager_set
            }

        conditional_proposals = []
        cond_draft_records = []

        if h2_steady:
            # === H2 steady ordering: recv verify → recv eager → draft conditional → send ===
            # ALL collectives in this block are UNCONDITIONAL — every H2 steady step
            # executes the same three collective pairs regardless of empty sets.

            # Phase 3: Receive normal verify result (UNCONDITIONAL)
            self._trace_collective("h2_normal_result_bcast", plan, prefix="before",
                                  tensor_numel=4 * len(target_seqs), tensor_dtype="int64")
            verify_res = self._receive_verify_result(target_seqs, group=self.verify_group)
            self._trace_collective("h2_normal_result_bcast", plan, prefix="after",
                                  tensor_numel=4 * len(target_seqs), tensor_dtype="int64")
            if target_seqs:
                trace_record = self._trace_dual_batch_schedule(target_seqs, plan, "draft_apply_verify")
                trace_record["proposal_tokens_verified"] = self._proposal_verify_token_count(target_seqs)
                trace_record["proposal_tokens_available"] = trace_record["proposal_tokens_verified"]
                # --- instrumentation: before normal verify ---
                _pre_verify_snapshot = {}
                _draft_eager_seq_ids = {int(s.seq_id) for s in draft_eager_seqs}
                _target_seq_map = {int(s.seq_id): s for s in target_seqs}
                _verify_tensor = verify_res.tolist()
                _acc, _rollout, _revise, _finish = _verify_tensor
                for idx, seq in enumerate(target_seqs):
                    _sid = int(seq.seq_id)
                    if _sid in _draft_eager_seq_ids:
                        _pre_verify_snapshot[_sid] = {
                            "seq_id": _sid,
                            "len_before_verify": int(len(seq)),
                            "verify_acc": bool(_acc[idx]),
                            "verify_rollout": int(_rollout[idx]),
                            "verify_revise_token": int(_revise[idx]),
                            "verify_finish": bool(_finish[idx]),
                            "pre_verify": bool(seq.pre_verify),
                        }
                if _pre_verify_snapshot:
                    trace_record["pre_verify_snapshot_draft_eager"] = _pre_verify_snapshot
                # --- end instrumentation ---
                torch.cuda.synchronize()
                self._mark_trace_start(trace_record)
                accepted_lens, invalidated_lens = self._apply_verify_result(target_seqs, verify_res)
                # --- instrumentation: after normal verify ---
                _post_verify_deltas = {}
                for seq in target_seqs:
                    _sid = int(seq.seq_id)
                    if _sid in _draft_eager_seq_ids:
                        _pre = _pre_verify_snapshot.get(_sid, {})
                        _old_len = _pre.get("len_before_verify", -1)
                        _new_len = int(len(seq))
                        _delta = _new_len - _old_len
                        _acc_flag = _pre.get("verify_acc", None)
                        _was_pre_verify = _pre.get("pre_verify", None)
                        _classification = self._classify_len_delta(
                            _delta, _acc_flag, _was_pre_verify, int(self.gamma),
                        )
                        _post_verify_deltas[_sid] = {
                            "seq_id": _sid,
                            "old_len": _old_len,
                            "new_len": _new_len,
                            "delta_len": _delta,
                            "classification": _classification,
                            "verify_acc": _acc_flag,
                            "accepted_len": int(accepted_lens.get(_sid, -1)),
                            "invalidated_len": int(invalidated_lens.get(_sid, -1)),
                        }
                if _post_verify_deltas:
                    trace_record["post_verify_deltas_draft_eager"] = _post_verify_deltas
                # --- end instrumentation ---
                consumed_seq_ids = self.dual_proposal_buffer.discard([seq.seq_id for seq in target_seqs])
                trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
                trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                _pre_promote_keys = self.dual_proposal_buffer.pending_seq_ids()
                self._promote_or_discard_eager_after_normal(
                    plan, accepted_lens, invalidated_lens, trace_record, count_tokens=False,
                )
                _post_promote_keys = self.dual_proposal_buffer.pending_seq_ids()
                assert _pre_promote_keys == _post_promote_keys, self._proposal_assertion_message(
                    plan,
                    f"dual_proposal_buffer keys changed during eager promote/discard: "
                    f"before={_pre_promote_keys}, after={_post_promote_keys}",
                )
                # Fixup eager_base_len: gated behind --disable-eager-base-len-fixup.
                # When disabled (default), the original eager_base_len from Phase 2
                # is preserved so we can compare against TARGET local seq length.
                if (draft_eager_seqs and self.global_config.enable_eager_execution
                        and not self.global_config.disable_eager_base_len_fixup):
                    fixups = []
                    for seq in draft_eager_seqs:
                        proposal = self.eager_proposal_buffer.get(seq.seq_id)
                        if proposal is not None and not proposal.consumed:
                            old_base = int(proposal.eager_base_len)
                            new_base = int(len(seq))
                            if old_base != new_base:
                                proposal.eager_base_len = new_base
                                fixups.append({
                                    "seq_id": int(seq.seq_id),
                                    "old_eager_base_len": old_base,
                                    "new_eager_base_len": new_base,
                                    "original_eager_base_len_at_generation": int(proposal.original_eager_base_len_at_generation),
                                })
                    if fixups:
                        trace_record["eager_base_len_fixups"] = fixups
                        plan.eager_base_len_fixups = fixups
                elif draft_eager_seqs and self.global_config.enable_eager_execution:
                    # Instrumentation-only path: record what WOULD have been fixed up
                    _would_fixup = []
                    for seq in draft_eager_seqs:
                        proposal = self.eager_proposal_buffer.get(seq.seq_id)
                        if proposal is not None and not proposal.consumed:
                            old_base = int(proposal.eager_base_len)
                            new_base = int(len(seq))
                            if old_base != new_base:
                                _would_fixup.append({
                                    "seq_id": int(seq.seq_id),
                                    "current_eager_base_len": old_base,
                                    "would_fixup_to": new_base,
                                    "original_eager_base_len_at_generation": int(proposal.original_eager_base_len_at_generation),
                                    "fixup_skipped": True,
                                })
                    if _would_fixup:
                        trace_record["eager_base_len_fixup_skipped"] = _would_fixup
                        plan.eager_base_len_fixup_skipped = _would_fixup
                torch.cuda.synchronize()
                self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
            # Phase 4: Receive eager verify result (UNCONDITIONAL)
            _n_target_eager = len(target_eager_seqs)
            self._debug_check_eager_bcast_consistency(plan, _n_target_eager)
            self._trace_collective("h2_eager_result_bcast", plan, prefix="before",
                                  tensor_numel=6 * _n_target_eager, tensor_dtype="int64",
                                  local_pre_meta_target_eager_set=[int(s) for s in plan.target_eager_set])
            eager_verify_res = self._receive_eager_verify_result(target_eager_seqs, group=self.verify_group)
            _actual_rows, _actual_cols = eager_verify_res.shape
            self._trace_collective("h2_eager_result_bcast", plan, prefix="after",
                                  tensor_numel=int(_actual_rows * _actual_cols), tensor_dtype="int64",
                                  meta_rows=int(_actual_rows), meta_cols=int(_actual_cols),
                                  meta_numel=int(_actual_rows * _actual_cols))
            if target_eager_seqs:
                trace_record = self._trace_dual_batch_schedule(target_eager_seqs, plan, "draft_apply_eager_verify")

                eager_ready = self.eager_proposal_buffer.get_many(
                    [seq.seq_id for seq in target_eager_seqs], ready_only=True
                )
                _eager_base_by_seq = {int(p.seq_id): int(p.eager_base_len) for p in eager_ready}

                overlap_seqs = [s for s in target_eager_seqs if s.seq_id in overlap_set]
                _overlap_rolled_back = {}
                for s in overlap_seqs:
                    _overlap_rolled_back[s.seq_id] = list(s.token_ids[-self.gamma:])
                    s.rollback_tokens(self.gamma)
                    _expected_base = _eager_base_by_seq.get(int(s.seq_id))
                    assert _expected_base is not None, self._proposal_assertion_message(
                        plan, f"overlap seq_id={s.seq_id} missing eager_base_len in eager_ready",
                    )
                    assert len(s) == _expected_base, self._proposal_assertion_message(
                        plan,
                        f"overlap rollback failed for seq_id={s.seq_id}: "
                        f"expected_len={_expected_base}, got={len(s)}",
                    )

                self._apply_eager_verify_result(
                    target_eager_seqs, eager_ready, eager_verify_res, plan, trace_record, count_tokens=False,
                )

                # Phase 5: H2 overlap handling — conditional draft + repair marking
                if overlap_seqs:
                    accepted_set = set(trace_record.get("eager_accepted_seq_ids", []))
                    conditional_seqs = [s for s in overlap_seqs if s.seq_id in accepted_set]
                    repair_seqs = [s for s in overlap_seqs if s.seq_id not in accepted_set]

                    self.dual_proposal_buffer.discard([s.seq_id for s in overlap_seqs])

                    for s in repair_seqs:
                        s.repair_required = True
                        s.repair_lane_id = int(plan.target_batch_id) if plan.target_batch_id is not None else None
                        s.last_eager_result = "reject"
                    plan.repair_scheduled_seq_ids = [int(s.seq_id) for s in repair_seqs]
                    plan.repair_lane_by_seq_id = {int(s.seq_id): int(plan.target_batch_id) if plan.target_batch_id is not None else -1 for s in repair_seqs}
                    plan.eager_partial_or_reject_seq_ids = [int(s.seq_id) for s in repair_seqs]
                    plan.overlap_normal_proposal_discarded_seq_ids = [int(s.seq_id) for s in overlap_seqs]
                    plan.overlap_normal_proposal_discard_reason = "H2_eager_verify_outcome"

                    if conditional_seqs:
                        for s in conditional_seqs:
                            s.last_eager_result = "full_accept"
                        plan.eager_full_accept_seq_ids = [int(s.seq_id) for s in conditional_seqs]
                        plan.conditional_normal_seq_ids = [int(s.seq_id) for s in conditional_seqs]
                        plan.conditional_normal_base_len_by_seq_id = {int(s.seq_id): len(s) for s in conditional_seqs}
                        conditional_proposals, cond_draft_records = self._draft_conditional_after_eager(
                            conditional_seqs, plan
                        )

                    trace_record["overlap_normal_proposal_discarded_seq_ids"] = [int(s.seq_id) for s in overlap_seqs]
                    trace_record["overlap_normal_proposal_discard_reason"] = "H2_eager_verify_outcome"
                    trace_record["repair_scheduled_seq_ids"] = [int(s.seq_id) for s in repair_seqs]
                    trace_record["conditional_normal_seq_ids"] = [int(s.seq_id) for s in conditional_seqs]
                torch.cuda.synchronize()
                self._mark_trace_end(trace_record)

            # Phase 6: Send combined proposals (UNCONDITIONAL)
            unconditional_normal = [p for p in normal_proposals if p.seq_id not in overlap_set]
            eager_ready_for_send = self.eager_proposal_buffer.get_many(
                plan.draft_eager_set, ready_only=True,
            ) if plan.draft_eager_set else []
            _eager_ready_seq_ids = sorted(p.seq_id for p in eager_ready_for_send)
            plan.eager_ready_seq_ids_before_send = _eager_ready_seq_ids
            # --- instrumentation: before send ---
            _draft_running_map = {int(s.seq_id): s for s in self.scheduler.running}
            _before_send_eager_info = []
            for p in eager_ready_for_send:
                _sid = int(p.seq_id)
                _seq = _draft_running_map.get(_sid)
                _before_send_eager_info.append({
                    "seq_id": _sid,
                    "eager_base_len_current": int(p.eager_base_len),
                    "original_eager_base_len_at_generation": int(p.original_eager_base_len_at_generation),
                    "current_seq_len": int(len(_seq)) if _seq is not None else -1,
                    "fixup_applied": int(p.eager_base_len) != int(p.original_eager_base_len_at_generation),
                    "eager_len": int(p.eager_len),
                })
            if _before_send_eager_info:
                plan.before_send_eager_info = _before_send_eager_info
            # --- end instrumentation ---
            # Fail-fast: no eager proposal encoded for any post-verify seq.
            for p in eager_ready_for_send:
                _sid = int(p.seq_id)
                _seq = _draft_running_map.get(_sid)
                assert _seq is not None and _seq.pre_verify, self._proposal_assertion_message(
                    plan,
                    f"eager proposal encoded for post-verify seq at send: seq_id={_sid}, "
                    f"pre_verify={getattr(_seq, 'pre_verify', None)}, "
                    f"eager_base_len={p.eager_base_len}, "
                    f"original_eager_base_len_at_generation={p.original_eager_base_len_at_generation}, "
                    f"draft_eager_set={plan.draft_eager_set}",
                )
            encoded_eager_seq_ids = [int(p.seq_id) for p in eager_ready_for_send]
            plan.encoded_eager_seq_ids = encoded_eager_seq_ids
            self._send_combined_dual_proposals(
                normal_proposals=unconditional_normal,
                conditional_normal_proposals=conditional_proposals,
                eager_proposals=eager_ready_for_send,
                plan=plan,
                expected_normal_seq_ids=list(plan.expected_normal_proposal_seq_ids),
                expected_eager_seq_ids=expected_eager_seq_ids,
            )
            for trace_record in draft_records:
                self._apply_combined_send_trace_fields(trace_record, plan)
            if eager_trace_record is not None:
                self._apply_combined_send_trace_fields(eager_trace_record, plan)
            for trace_record in cond_draft_records:
                self._apply_combined_send_trace_fields(trace_record, plan)
        else:
            # === Legacy ordering (fallback/priming): send → recv verify ===
            if draft_seqs or expected_eager_seq_ids:
                plan.encoded_eager_seq_ids = [int(p.seq_id) for p in eager_proposals]
                self._send_combined_dual_proposals(
                    normal_proposals=normal_proposals,
                    conditional_normal_proposals=[],
                    eager_proposals=eager_proposals,
                    plan=plan,
                    expected_normal_seq_ids=expected_normal_seq_ids,
                    expected_eager_seq_ids=expected_eager_seq_ids,
                )
                for trace_record in draft_records:
                    self._apply_combined_send_trace_fields(trace_record, plan)
                if eager_trace_record is not None:
                    self._apply_combined_send_trace_fields(eager_trace_record, plan)

            if target_seqs:
                trace_record = self._trace_dual_batch_schedule(target_seqs, plan, "draft_apply_verify")
                trace_record["proposal_tokens_verified"] = self._proposal_verify_token_count(target_seqs)
                trace_record["proposal_tokens_available"] = trace_record["proposal_tokens_verified"]
                torch.cuda.synchronize()
                self._mark_trace_start(trace_record)
                verify_res = self._receive_verify_result(target_seqs)
                # --- instrumentation: before normal verify (legacy) ---
                _pre_verify_snapshot_l = {}
                _draft_eager_seq_ids_l = {int(s.seq_id) for s in draft_eager_seqs}
                _verify_tensor_l = verify_res.tolist()
                _acc_l, _rollout_l, _revise_l, _finish_l = _verify_tensor_l
                for idx, seq in enumerate(target_seqs):
                    _sid = int(seq.seq_id)
                    if _sid in _draft_eager_seq_ids_l:
                        _pre_verify_snapshot_l[_sid] = {
                            "seq_id": _sid,
                            "len_before_verify": int(len(seq)),
                            "verify_acc": bool(_acc_l[idx]),
                            "verify_rollout": int(_rollout_l[idx]),
                            "verify_revise_token": int(_revise_l[idx]),
                            "verify_finish": bool(_finish_l[idx]),
                            "pre_verify": bool(seq.pre_verify),
                        }
                if _pre_verify_snapshot_l:
                    trace_record["pre_verify_snapshot_draft_eager"] = _pre_verify_snapshot_l
                # --- end instrumentation ---
                accepted_lens, invalidated_lens = self._apply_verify_result(target_seqs, verify_res)
                # --- instrumentation: after normal verify (legacy) ---
                _post_verify_deltas_l = {}
                for seq in target_seqs:
                    _sid = int(seq.seq_id)
                    if _sid in _draft_eager_seq_ids_l:
                        _pre = _pre_verify_snapshot_l.get(_sid, {})
                        _old_len = _pre.get("len_before_verify", -1)
                        _new_len = int(len(seq))
                        _delta = _new_len - _old_len
                        _acc_flag = _pre.get("verify_acc", None)
                        _was_pre_verify = _pre.get("pre_verify", None)
                        _classification = self._classify_len_delta(
                            _delta, _acc_flag, _was_pre_verify, int(self.gamma),
                        )
                        _post_verify_deltas_l[_sid] = {
                            "seq_id": _sid,
                            "old_len": _old_len,
                            "new_len": _new_len,
                            "delta_len": _delta,
                            "classification": _classification,
                            "verify_acc": _acc_flag,
                            "accepted_len": int(accepted_lens.get(_sid, -1)),
                            "invalidated_len": int(invalidated_lens.get(_sid, -1)),
                        }
                if _post_verify_deltas_l:
                    trace_record["post_verify_deltas_draft_eager"] = _post_verify_deltas_l
                # --- end instrumentation ---
                consumed_seq_ids = self.dual_proposal_buffer.discard([seq.seq_id for seq in target_seqs])
                trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
                trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                _pre_promote_keys = self.dual_proposal_buffer.pending_seq_ids()
                self._promote_or_discard_eager_after_normal(
                    plan, accepted_lens, invalidated_lens, trace_record, count_tokens=False,
                )
                _post_promote_keys = self.dual_proposal_buffer.pending_seq_ids()
                assert _pre_promote_keys == _post_promote_keys, self._proposal_assertion_message(
                    plan,
                    f"dual_proposal_buffer keys changed during eager promote/discard: "
                    f"before={_pre_promote_keys}, after={_post_promote_keys}",
                )
                # Fixup eager_base_len: gated behind --disable-eager-base-len-fixup.
                if (draft_eager_seqs and self.global_config.enable_eager_execution
                        and not self.global_config.disable_eager_base_len_fixup):
                    fixups = []
                    for seq in draft_eager_seqs:
                        proposal = self.eager_proposal_buffer.get(seq.seq_id)
                        if proposal is not None and not proposal.consumed:
                            old_base = int(proposal.eager_base_len)
                            new_base = int(len(seq))
                            if old_base != new_base:
                                proposal.eager_base_len = new_base
                                fixups.append({
                                    "seq_id": int(seq.seq_id),
                                    "old_eager_base_len": old_base,
                                    "new_eager_base_len": new_base,
                                    "original_eager_base_len_at_generation": int(proposal.original_eager_base_len_at_generation),
                                })
                    if fixups:
                        trace_record["eager_base_len_fixups"] = fixups
                        plan.eager_base_len_fixups = fixups
                elif draft_eager_seqs and self.global_config.enable_eager_execution:
                    _would_fixup_l = []
                    for seq in draft_eager_seqs:
                        proposal = self.eager_proposal_buffer.get(seq.seq_id)
                        if proposal is not None and not proposal.consumed:
                            old_base = int(proposal.eager_base_len)
                            new_base = int(len(seq))
                            if old_base != new_base:
                                _would_fixup_l.append({
                                    "seq_id": int(seq.seq_id),
                                    "current_eager_base_len": old_base,
                                    "would_fixup_to": new_base,
                                    "original_eager_base_len_at_generation": int(proposal.original_eager_base_len_at_generation),
                                    "fixup_skipped": True,
                                })
                    if _would_fixup_l:
                        trace_record["eager_base_len_fixup_skipped"] = _would_fixup_l
                        plan.eager_base_len_fixup_skipped = _would_fixup_l
                torch.cuda.synchronize()
                self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

            if target_eager_seqs:
                trace_record = self._trace_dual_batch_schedule(target_eager_seqs, plan, "draft_apply_eager_verify")
                eager_ready = self.eager_proposal_buffer.get_many(
                    [seq.seq_id for seq in target_eager_seqs], ready_only=True
                )
                _eager_base_by_seq = {int(p.seq_id): int(p.eager_base_len) for p in eager_ready}

                overlap_seqs = [s for s in target_eager_seqs if s.seq_id in overlap_set]
                _overlap_rolled_back = {}
                for s in overlap_seqs:
                    _overlap_rolled_back[s.seq_id] = list(s.token_ids[-self.gamma:])
                    s.rollback_tokens(self.gamma)
                    _expected_base = _eager_base_by_seq.get(int(s.seq_id))
                    assert _expected_base is not None, self._proposal_assertion_message(
                        plan, f"overlap seq_id={s.seq_id} missing eager_base_len in eager_ready",
                    )
                    assert len(s) == _expected_base, self._proposal_assertion_message(
                        plan,
                        f"overlap rollback failed for seq_id={s.seq_id}: "
                        f"expected_len={_expected_base}, got={len(s)}",
                    )

                eager_verify_res = self._receive_eager_verify_result(target_eager_seqs)
                self._apply_eager_verify_result(
                    target_eager_seqs, eager_ready, eager_verify_res, plan, trace_record, count_tokens=False,
                )

                if overlap_seqs:
                    accepted_set = set(trace_record.get("eager_accepted_seq_ids", []))
                    discarded = []
                    for s in overlap_seqs:
                        self.dual_proposal_buffer.discard([s.seq_id])
                        discarded.append(int(s.seq_id))
                        if s.seq_id not in accepted_set:
                            for token_id in _overlap_rolled_back.get(s.seq_id, []):
                                s.append_token(int(token_id))
                    plan.overlap_normal_proposal_kept_seq_ids = []
                    plan.overlap_normal_proposal_discarded_seq_ids = discarded
                    if set(discarded) & accepted_set:
                        plan.overlap_normal_proposal_discard_reason = "eager_accepted_base_mismatch"
                    else:
                        plan.overlap_normal_proposal_discard_reason = "eager_rejected_or_partial"
                    trace_record["overlap_normal_proposal_kept_seq_ids"] = []
                    trace_record["overlap_normal_proposal_discarded_seq_ids"] = discarded
                    trace_record["overlap_normal_proposal_discard_reason"] = (
                        plan.overlap_normal_proposal_discard_reason
                    )
                torch.cuda.synchronize()
                self._mark_trace_end(trace_record)

        self._log_h2_steady_lane_transition(plan)

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

    def _build_eager_verify_result(
        self,
        logits: torch.Tensor,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
        proposals: list[EagerBufferedProposal],
        plan: StepPlan,
        *,
        group=None,
    ) -> torch.Tensor:
        cols = len(seqs)
        verify_res = torch.zeros((6, cols), dtype=torch.int64, device="cuda")
        if self.tp_params.local_rank == 0:
            row_indices = []
            token_ids = []
            for idx, (seq, proposal) in enumerate(zip(seqs, proposals)):
                assert not seq.pre_verify, self._proposal_assertion_message(
                    plan,
                    f"Phase 1H-lite eager verify requires post-normal-accept state for seq_id={seq.seq_id}",
                )
                row_indices.extend(range(idx * self.gamma, idx * self.gamma + int(proposal.eager_len)))
                token_ids.extend(int(token_id) for token_id in proposal.eager_token_ids)
            if token_ids:
                row_tensor = torch.tensor(row_indices, dtype=torch.long, device="cuda")
                token_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")
                selected_logits = logits.index_select(0, row_tensor)
                selected_temperatures = temperatures.index_select(0, row_tensor)
                target_logits = norm_logits(selected_logits, selected_temperatures)
                target_prob = target_logits.gather(dim=1, index=token_tensor.unsqueeze(1)).squeeze(1)
                judge = (torch.rand(len(token_ids), device="cuda") <= target_prob).tolist()
            else:
                judge = []

            offset = 0
            rows = []
            for seq, proposal in zip(seqs, proposals):
                eager_len = int(proposal.eager_len)
                accepted = bool(judge[offset:offset + eager_len]) and all(judge[offset:offset + eager_len])
                finish = False
                if accepted:
                    finish = any(
                        (not seq.ignore_eos and is_eos(token_id, self.scheduler.eos))
                        for token_id in proposal.eager_token_ids
                    ) or seq.num_completion_tokens + eager_len >= seq.max_tokens
                rows.append(
                    [
                        int(seq.seq_id),
                        int(accepted),
                        eager_len,
                        eager_len if accepted else 0,
                        0 if accepted else eager_len,
                        int(finish),
                    ]
                )
                offset += eager_len
            if rows:
                verify_res = torch.tensor(rows, dtype=torch.int64, device="cuda").T.contiguous()

        # Source-authoritative meta+payload broadcast: receiver must allocate
        # from the source meta, not from local target_eager_set length.
        is_source = self.rank == self.global_config.target_config.master_rank
        numel = int(6 * cols) if is_source else 0
        meta = torch.tensor([6, cols, numel], dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=group)
        if numel > 0:
            dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=group)
        return verify_res

    def _run_eager_verify_sidecar(
        self,
        seqs: list[Sequence],
        proposals: list[EagerBufferedProposal],
        plan: StepPlan,
        *,
        group=None,
    ) -> None:
        if not seqs:
            return
        self._validate_eager_proposals(proposals, seqs, plan)
        self._allocate_decode_slots_for_dual(seqs, plan, "dual_eager_verify")
        trace_record = self._trace_dual_batch_schedule(seqs, plan, "dual_eager_verify")
        trace_record["eager_buffer_size_before"] = self.eager_proposal_buffer.size()
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        trace_record["eager_verify_start_ts"] = trace_record.get("verify_start_ts")
        logits = self.run_model(input_ids, positions, False)
        verify_res = self._build_eager_verify_result(logits, seqs, temperatures, proposals, plan, group=group)
        self._apply_eager_verify_result(
            seqs,
            proposals,
            verify_res,
            plan,
            trace_record,
            count_tokens=True,
        )
        torch.cuda.synchronize()
        self._mark_trace_end(trace_record)
        trace_record["eager_verify_end_ts"] = trace_record.get("verify_end_ts")
        if trace_record["eager_verify_start_ts"] is not None and trace_record["eager_verify_end_ts"] is not None:
            trace_record["eager_verify_time_ms"] = (
                trace_record["eager_verify_end_ts"] - trace_record["eager_verify_start_ts"]
            ) * 1000

        # Phase 1H-lite: conditional invalidation for overlapping seqs
        # (target_eager_set ∩ draft_home_set).  On the TARGET side the
        # sequence was not mutated by normal draft, but the normal
        # proposal sent by the draft rank is stale regardless of eager
        # accept/reject outcome.
        overlap_seq_ids = set(plan.target_eager_draft_home_overlap_seq_ids)
        if overlap_seq_ids:
            accepted_set = set(trace_record.get("eager_accepted_seq_ids", []))
            target_overlap = overlap_seq_ids & {int(s.seq_id) for s in seqs}
            discarded = []
            seq_by_id = {int(s.seq_id): s for s in seqs}
            for seq_id in target_overlap:
                self.dual_proposal_buffer.discard([seq_id])
                discarded.append(int(seq_id))
                # H2 lane transition: rejected overlap seqs must be
                # marked repair_required on TARGET so both ranks agree
                # on lane migration in the next step.  Without this the
                # TARGET rank keeps the seq in target_home_set while
                # the DRAFT rank migrates it, causing target_home_set
                # divergence and missing-proposal assertion failures.
                if seq_id not in accepted_set:
                    s = seq_by_id.get(seq_id)
                    if s is not None:
                        s.repair_required = True
                        s.repair_lane_id = int(plan.target_batch_id) if plan.target_batch_id is not None else None
                        s.last_eager_result = "reject"
            rejected_overlap = [seq_id for seq_id in target_overlap if seq_id not in accepted_set]
            if rejected_overlap:
                plan.repair_scheduled_seq_ids = list(
                    set(plan.repair_scheduled_seq_ids) | set(rejected_overlap)
                )
                for seq_id in rejected_overlap:
                    plan.repair_lane_by_seq_id[int(seq_id)] = int(plan.target_batch_id) if plan.target_batch_id is not None else -1
                plan.eager_partial_or_reject_seq_ids = list(
                    set(plan.eager_partial_or_reject_seq_ids) | set(rejected_overlap)
                )
            plan.overlap_normal_proposal_discarded_seq_ids = list(
                set(plan.overlap_normal_proposal_discarded_seq_ids) | set(discarded)
            )
            if discarded and not plan.overlap_normal_proposal_discard_reason:
                if set(discarded) & accepted_set:
                    plan.overlap_normal_proposal_discard_reason = "eager_accepted_base_mismatch"
                else:
                    plan.overlap_normal_proposal_discard_reason = "eager_rejected_or_partial"
            trace_record["overlap_normal_proposal_kept_seq_ids"] = []
            trace_record["overlap_normal_proposal_discarded_seq_ids"] = discarded
            trace_record["overlap_normal_proposal_discard_reason"] = (
                plan.overlap_normal_proposal_discard_reason
            )

        self._finalize_record_profile(trace_record)

    def dual_batch_pearl_step(self):
        plan = self._build_dual_batch_step_plan()
        self._sync_h2_plan_fields(plan)
        self._debug_check_h2_plan_consistency(plan)
        target_seqs = self._resolve_dual_seq_ids(plan.target_home_set, plan, "dual_verify")
        target_eager_seqs = self._resolve_dual_seq_ids(plan.target_eager_set, plan, "dual_eager_verify")
        draft_seq_ids = list(plan.draft_home_set)
        target_seq_ids = [seq.seq_id for seq in target_seqs]
        target_eager_seq_ids = [seq.seq_id for seq in target_eager_seqs]
        assert set(target_eager_seq_ids).isdisjoint(target_seq_ids), self._proposal_assertion_message(
            plan,
            f"target_eager_set overlaps target_home_set: {target_eager_seq_ids}",
        )
        # H2: detect overlap between target_eager_set and draft_home_set
        h2_steady = plan.plan_phase == "steady" and self.global_config.enable_eager_execution
        overlap_eager_draft = sorted(set(target_eager_seq_ids) & set(draft_seq_ids))
        if overlap_eager_draft:
            plan.target_eager_draft_home_overlap_seq_ids = overlap_eager_draft
        fallback_same_batch = bool(target_seq_ids) and target_seq_ids == draft_seq_ids and plan.plan_phase == "fallback"

        target_proposals = []
        if target_seqs and not fallback_same_batch:
            assert self.dual_proposal_buffer.has_all(target_seq_ids), self._proposal_assertion_message(
                plan,
                f"missing buffered proposals for target seq_ids={target_seq_ids}",
            )
            target_proposals = self.dual_proposal_buffer.get_many(target_seq_ids)
        eager_proposals = self.eager_proposal_buffer.get_many(target_eager_seq_ids, ready_only=True)

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

        expected_eager_seq_ids = list(plan.draft_eager_set) if self.global_config.enable_eager_execution else []

        if h2_steady:
            # === H2 steady ordering: verify → broadcast → eager verify → broadcast → recv ===
            # ALL collectives in this block are UNCONDITIONAL — every H2 steady step
            # executes the same three collective pairs regardless of empty sets.

            # Phase A: Normal verify broadcast (UNCONDITIONAL)
            if target_seqs:
                self._validate_proposals_for_target(target_proposals, target_seqs, plan)
                consumed_seq_ids = self.dual_proposal_buffer.discard(target_seq_ids)
                trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
                trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                self._trace_collective("h2_normal_result_bcast", plan, prefix="before",
                                      tensor_numel=4 * len(target_seqs), tensor_dtype="int64")
                accepted_lens, invalidated_lens = self.verify_from_proposals(
                    logits, target_seqs, temperatures, target_proposals, plan,
                    group=self.verify_group,
                )
                self._trace_collective("h2_normal_result_bcast", plan, prefix="after",
                                      tensor_numel=4 * len(target_seqs), tensor_dtype="int64")
                torch.cuda.synchronize()
                self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
            else:
                verify_res = torch.zeros((4, 0), dtype=torch.int64, device="cuda")
                self._trace_collective("h2_normal_result_bcast", plan, prefix="before",
                                      tensor_numel=0, tensor_dtype="int64")
                dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=self.verify_group)
                self._trace_collective("h2_normal_result_bcast", plan, prefix="after",
                                      tensor_numel=0, tensor_dtype="int64")

            # Phase B: Eager verify broadcast (UNCONDITIONAL)
            self._debug_check_eager_bcast_consistency(plan, len(target_eager_seqs))
            if target_eager_seqs:
                _n_eager_cols = len(target_eager_seqs)
                self._trace_collective("h2_eager_result_bcast", plan, prefix="before",
                                      tensor_numel=6 * _n_eager_cols, tensor_dtype="int64",
                                      meta_rows=6, meta_cols=_n_eager_cols, meta_numel=6 * _n_eager_cols,
                                      local_pre_meta_target_eager_set=[int(s) for s in plan.target_eager_set])
                self._run_eager_verify_sidecar(target_eager_seqs, eager_proposals, plan, group=self.verify_group)
                self._trace_collective("h2_eager_result_bcast", plan, prefix="after",
                                      tensor_numel=6 * _n_eager_cols, tensor_dtype="int64",
                                      meta_rows=6, meta_cols=_n_eager_cols, meta_numel=6 * _n_eager_cols)
            else:
                # Source-authoritative meta+payload: send meta [6,0,0] so
                # receiver allocates [6,0] regardless of its local state.
                self._trace_collective("h2_eager_result_bcast", plan, prefix="before",
                                      tensor_numel=0, tensor_dtype="int64",
                                      meta_rows=6, meta_cols=0, meta_numel=0,
                                      local_pre_meta_target_eager_set=[int(s) for s in plan.target_eager_set])
                meta = torch.tensor([6, 0, 0], dtype=torch.int64, device="cuda")
                dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
                self._trace_collective("h2_eager_result_bcast", plan, prefix="after",
                                      tensor_numel=0, tensor_dtype="int64",
                                      meta_rows=6, meta_cols=0, meta_numel=0)

            # Phase C: Receive combined proposals (UNCONDITIONAL)
            received_normal, received_conditional, received_eager = self._receive_combined_dual_proposals(
                list(plan.expected_normal_proposal_seq_ids), expected_eager_seq_ids, plan,
            )
            if received_normal:
                self.dual_proposal_buffer.store(received_normal)
            if received_conditional:
                self.dual_proposal_buffer.store(received_conditional)
            if received_eager:
                self.eager_proposal_buffer.store(received_eager)
                # Mark received eager proposals as ready: the DRAFT rank
                # already filtered for ready_only before sending, so every
                # received eager proposal is implicitly ready for the next
                # step's _populate_eager_fields.  Without this the TARGET
                # eager_proposal_buffer diverges from the DRAFT side.
                for ep in received_eager:
                    self.eager_proposal_buffer.mark_ready(int(ep.seq_id))
                # Instrumentation + fail-fast: record full lifecycle state at receive.
                _running_ids = {int(s.seq_id): s for s in self.scheduler.running}
                _target_receive_eager_info = []
                for ep in received_eager:
                    _sid = int(ep.seq_id)
                    _seq = _running_ids.get(_sid)
                    _ep_base = int(ep.eager_base_len)
                    _seq_len = int(len(_seq)) if _seq is not None else -1
                    _orig_base = int(ep.original_eager_base_len_at_generation)
                    _info = {
                        "seq_id": _sid,
                        "local_seq_len_at_receive": _seq_len,
                        "received_eager_base_len": _ep_base,
                        "received_original_eager_base_len_at_generation": _orig_base,
                        "received_eager_len": int(ep.eager_len),
                        "source_step_id": int(ep.source_step_id),
                        "source_plan_id": int(ep.source_plan_id),
                        "source_home_batch_id": int(ep.source_home_batch_id),
                        "base_len_diverged_from_original": _ep_base != _orig_base,
                        "pre_verify": bool(_seq.pre_verify) if _seq is not None else None,
                    }
                    _target_receive_eager_info.append(_info)
                if _target_receive_eager_info:
                    plan.t_eager_receive_info = _target_receive_eager_info
                # Fail-fast 1: no eager proposal for post-verify seq on TARGET.
                for ep in received_eager:
                    _sid = int(ep.seq_id)
                    _seq = _running_ids.get(_sid)
                    if _seq is not None and not _seq.pre_verify:
                        assert False, self._proposal_assertion_message(
                            plan,
                            f"eager proposal received for post-verify seq on TARGET: "
                            f"seq_id={_sid}, "
                            f"target_seq_pre_verify_at_receive={bool(_seq.pre_verify)}, "
                            f"draft_eager_set={plan.draft_eager_set}, "
                            f"expected_eager_proposal_seq_ids={plan.expected_eager_proposal_seq_ids}, "
                            f"source_step_id={ep.source_step_id}, "
                            f"source_plan_id={ep.source_plan_id}, "
                            f"source_home_batch_id={ep.source_home_batch_id}, "
                            f"eager_base_len={ep.eager_base_len}, "
                            f"original_eager_base_len_at_generation={ep.original_eager_base_len_at_generation}, "
                            f"local_seq_len={len(_seq)}",
                        )
                # Fail-fast 2: verify eager_base_len matches local seq length.
                for ep in received_eager:
                    _sid = int(ep.seq_id)
                    _seq = _running_ids.get(_sid)
                    if _seq is not None:
                        _ep_base = int(ep.eager_base_len)
                        _seq_len = int(len(_seq))
                        _orig_base = int(ep.original_eager_base_len_at_generation)
                        assert _ep_base == _seq_len, self._proposal_assertion_message(
                            plan,
                            f"eager_base_len mismatch at TARGET receive for seq_id={_sid}: "
                            f"eager_base_len={_ep_base}, local_seq_len={_seq_len}, "
                            f"original_eager_base_len_at_generation={_orig_base}, "
                            f"eager_len={ep.eager_len}, "
                            f"source_plan_id={ep.source_plan_id}, "
                            f"source_step_id={ep.source_step_id}, "
                            f"source_home_batch_id={ep.source_home_batch_id}, "
                            f"pre_verify={bool(_seq.pre_verify)}, "
                            f"fixup_skipped={getattr(plan, 'eager_base_len_fixup_skipped', None)}, "
                            f"before_send_info={getattr(plan, 'before_send_eager_info', None)}",
                        )
            plan.normal_proposal_buffer_keys_after_receive = self.dual_proposal_buffer.pending_seq_ids()
            plan.eager_buffer_keys_after_receive = self.eager_proposal_buffer.keys()
        else:
            # === Legacy ordering (fallback/priming): recv → verify → broadcast ===
            received_normal = []
            received_conditional = []
            received_eager = []
            if draft_seq_ids or expected_eager_seq_ids:
                received_normal, received_conditional, received_eager = self._receive_combined_dual_proposals(
                    draft_seq_ids, expected_eager_seq_ids, plan,
                )
                if fallback_same_batch:
                    target_proposals = received_normal + received_conditional
                    if trace_record is not None:
                        trace_record["proposal_tokens_available"] = sum(len(p.to_be_verified_token_ids) for p in target_proposals)
                        trace_record["proposal_tokens_verified"] = trace_record["proposal_tokens_available"]
                else:
                    if received_normal:
                        self.dual_proposal_buffer.store(received_normal)
                    if received_conditional:
                        self.dual_proposal_buffer.store(received_conditional)
                if received_eager:
                    self.eager_proposal_buffer.store(received_eager)
                    # Instrumentation + fail-fast at receive (legacy).
                    _running_ids_l = {int(s.seq_id): s for s in self.scheduler.running}
                    _legacy_receive_info = []
                    for ep in received_eager:
                        _sid = int(ep.seq_id)
                        _seq = _running_ids_l.get(_sid)
                        _ep_base = int(ep.eager_base_len)
                        _seq_len = int(len(_seq)) if _seq is not None else -1
                        _orig_base = int(ep.original_eager_base_len_at_generation)
                        _legacy_receive_info.append({
                            "seq_id": _sid,
                            "local_seq_len_at_receive": _seq_len,
                            "received_eager_base_len": _ep_base,
                            "received_original_eager_base_len_at_generation": _orig_base,
                            "received_eager_len": int(ep.eager_len),
                            "source_step_id": int(ep.source_step_id),
                            "source_plan_id": int(ep.source_plan_id),
                            "source_home_batch_id": int(ep.source_home_batch_id),
                            "base_len_diverged_from_original": _ep_base != _orig_base,
                            "pre_verify": bool(_seq.pre_verify) if _seq is not None else None,
                        })
                    if _legacy_receive_info:
                        plan.t_eager_receive_info = _legacy_receive_info
                    for ep in received_eager:
                        _sid = int(ep.seq_id)
                        _seq = _running_ids_l.get(_sid)
                        if _seq is not None:
                            _ep_base = int(ep.eager_base_len)
                            _seq_len = int(len(_seq))
                            _orig_base = int(ep.original_eager_base_len_at_generation)
                            assert _ep_base == _seq_len, self._proposal_assertion_message(
                                plan,
                                f"eager_base_len mismatch at TARGET receive for seq_id={_sid}: "
                                f"eager_base_len={_ep_base}, local_seq_len={_seq_len}, "
                                f"original_eager_base_len_at_generation={_orig_base}, "
                                f"eager_len={ep.eager_len}, "
                                f"source_plan_id={ep.source_plan_id}, "
                                f"source_step_id={ep.source_step_id}, "
                                f"source_home_batch_id={ep.source_home_batch_id}, "
                                f"pre_verify={bool(_seq.pre_verify)}",
                            )
                plan.normal_proposal_buffer_keys_after_receive = self.dual_proposal_buffer.pending_seq_ids()
                plan.eager_buffer_keys_after_receive = self.eager_proposal_buffer.keys()
                if trace_record is not None:
                    self._apply_combined_proposal_trace_fields(trace_record, plan)

            if target_seqs:
                self._validate_proposals_for_target(target_proposals, target_seqs, plan)
                consumed_seq_ids = self.dual_proposal_buffer.discard(target_seq_ids)
                if fallback_same_batch:
                    consumed_seq_ids = [proposal.seq_id for proposal in target_proposals]
                trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
                trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                accepted_lens, invalidated_lens = self.verify_from_proposals(
                    logits, target_seqs, temperatures, target_proposals, plan,
                )
                _pre_promote_keys = self.dual_proposal_buffer.pending_seq_ids()
                self._promote_or_discard_eager_after_normal(
                    plan, accepted_lens, invalidated_lens, trace_record, count_tokens=True,
                )
                _post_promote_keys = self.dual_proposal_buffer.pending_seq_ids()
                assert _pre_promote_keys == _post_promote_keys, self._proposal_assertion_message(
                    plan,
                    f"dual_proposal_buffer keys changed during eager promote/discard: "
                    f"before={_pre_promote_keys}, after={_post_promote_keys}",
                )
                torch.cuda.synchronize()
                self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
            elif received_normal or received_conditional or received_eager:
                priming_record = self._trace_dual_batch_schedule([], plan, "dual_verify_idle")
                priming_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                self._apply_combined_proposal_trace_fields(priming_record, plan)
                self._finalize_record_profile(priming_record)

            if target_eager_seqs:
                self._run_eager_verify_sidecar(target_eager_seqs, eager_proposals, plan)

        self._log_h2_steady_lane_transition(plan)

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
        *,
        group=None,
    ):
        self._validate_proposals_for_target(proposals, seqs, plan)
        to_be_verified_tokens = []
        next_round_input = []
        for proposal in proposals:
            to_be_verified_tokens.extend(proposal.to_be_verified_token_ids)
            next_round_input.extend(proposal.proposal_token_ids)
        msg = torch.tensor(to_be_verified_tokens + next_round_input, dtype=torch.int64, device="cuda")
        return self._verify_from_message(logits, seqs, temperatures, msg, len(to_be_verified_tokens), group=group)

    @torch.inference_mode()
    def _verify_from_message(self, logits: torch.Tensor, seqs: list[Sequence], temperatures: torch.Tensor, msg: torch.Tensor, num_to_be_verified_tokens: int, *, group=None):
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
        
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=group)

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
