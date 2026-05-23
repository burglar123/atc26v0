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
        plan.draft_home_set = [seq_id for seq_id in plan.draft_home_set if seq_id not in set(target_eager_set)]
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

        # Phase 1H-lite keeps normal and eager buffers separate. If a ready
        # eager sidecar caused a seq to skip normal drafting in the previous
        # round, refresh its normal proposal explicitly instead of verifying a
        # partially populated target batch.
        plan.plan_phase = "fallback"
        plan.steady_step = False
        plan.priming_step = False
        plan.fallback_reason = "missing_normal_proposal_after_eager"
        plan.target_batch_id = None
        plan.target_home_set = []
        plan.target_eager_set = []
        plan.draft_batch_id = plan.home_batch_ids.get(missing[0]) if missing else None
        plan.draft_home_set = list(missing)
        plan.draft_eager_set = []
        plan.normal_proposal_refresh_seq_ids = list(missing)
        dropped_refresh = self.dual_proposal_buffer.discard(missing)
        plan.proposal_buffer_dropped_seq_ids = sorted(set(plan.proposal_buffer_dropped_seq_ids) | set(dropped_refresh))
        plan.proposal_buffer_dropped_count = len(plan.proposal_buffer_dropped_seq_ids)
        plan.fallback_has_target_batch = False
        plan.fallback_has_draft_batch = bool(missing)
        plan.fallback_active_batch_count = int(plan.active_batch_count)
        plan.fallback_active_seq_count = int(plan.active_seq_count)
        plan.fallback_pending_proposal_count = self.dual_proposal_buffer.size()
        plan.fallback_target_seq_count = 0
        plan.fallback_draft_seq_count = len(missing)
        plan.fallback_buffer_hit_count = len(inspect["hit_seq_ids"])
        plan.fallback_buffer_miss_count = len(missing)
        for seq_id in list(plan.budgets):
            plan.budgets[seq_id].eager_gamma = 0
        for seq_id in missing:
            plan.budgets.setdefault(seq_id, RequestBudget(normal_gamma=self.gamma, eager_gamma=0))
        target_home_size = len(plan.target_home_set)
        draft_home_size = len(plan.draft_home_set)
        plan.target_fraction_of_active = target_home_size / max(1, int(plan.active_seq_count))
        plan.draft_fraction_of_active = draft_home_size / max(1, int(plan.active_seq_count))
        plan.split_imbalance = abs(target_home_size - draft_home_size) / max(1, target_home_size + draft_home_size)
        plan.target_to_draft_size_ratio = target_home_size / max(1, draft_home_size)

        refreshed_inspect = self.dual_proposal_buffer.inspect(plan.target_home_set)
        plan.proposal_buffer_requested_seq_ids = refreshed_inspect["requested_seq_ids"]
        plan.proposal_buffer_hit_seq_ids = refreshed_inspect["hit_seq_ids"]
        plan.proposal_buffer_miss_seq_ids = refreshed_inspect["miss_seq_ids"]
        plan.proposal_buffer_invalid_seq_ids = refreshed_inspect["invalid_seq_ids"]
        plan.proposal_buffer_hit_count = len(plan.proposal_buffer_hit_seq_ids)
        plan.proposal_buffer_miss_count = len(plan.proposal_buffer_miss_seq_ids)
        plan.proposal_buffer_invalid_count = len(plan.proposal_buffer_invalid_seq_ids)

    def _annotate_eager_trace_plan(self, plan: StepPlan) -> None:
        plan.eager_trace_enabled = bool(self.global_config.enable_eager_trace or self.global_config.enable_eager_execution)
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

        running_seq_ids = {seq.seq_id for seq in self.scheduler.running}
        now = time.time()
        scored = []
        threshold = float(self.global_config.eager_accept_threshold)
        for seq in self.scheduler.find_by_seq_ids(plan.target_home_set):
            if seq.seq_id not in running_seq_ids or seq.is_finished or self._has_pending_eager_state(seq):
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
        dropped_normal = self.dual_proposal_buffer.discard_inactive(active_seq_ids)
        dropped_eager = self.eager_proposal_buffer.discard_inactive(active_seq_ids)
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
        self._annotate_eager_execution_plan(plan)
        plan.proposal_buffer_keys_after_eager_selection = self.dual_proposal_buffer.pending_seq_ids()
        self._handle_missing_normal_proposals_after_eager(plan)
        self._annotate_eager_trace_plan(plan)
        if self.global_config.enable_eager_execution:
            plan.validate_phase1h_eager_execution()
        else:
            plan.validate_phase1c()
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
            f"draft_home_set={plan.draft_home_set}, target_eager_set={plan.target_eager_set}, "
            f"draft_eager_set={plan.draft_eager_set}, buffered_proposal_seq_ids="
            f"{self.dual_proposal_buffer.pending_seq_ids()}, eager_buffer_keys={self.eager_proposal_buffer.keys()}"
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
        assert received_seq_ids == list(expected_seq_ids), self._proposal_assertion_message(
            plan,
            f"eager proposal seq_id mismatch: expected={expected_seq_ids}, received={received_seq_ids}, "
            f"source_plan_id={proposal_plan_id}, source_step_id={proposal_step_id}",
        )
        return proposals

    def _combined_normal_payload(self, proposals: list[BufferedProposal]) -> list[int]:
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
        return header + tokens

    def _combined_eager_payload(self, proposals: list[EagerBufferedProposal]) -> list[int]:
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
        return header + tokens

    def _send_combined_dual_proposals(
        self,
        normal_proposals: list[BufferedProposal],
        eager_proposals: list[EagerBufferedProposal],
        plan: StepPlan,
        expected_normal_seq_ids: list[int],
        expected_eager_seq_ids: list[int],
    ) -> None:
        actual_normal_seq_ids = [int(proposal.seq_id) for proposal in normal_proposals]
        actual_eager_seq_ids = [int(proposal.seq_id) for proposal in eager_proposals]
        plan.send_expected_normal_seq_ids = list(expected_normal_seq_ids)
        plan.send_actual_normal_seq_ids = list(actual_normal_seq_ids)
        plan.send_expected_eager_seq_ids = list(expected_eager_seq_ids)
        plan.send_actual_eager_seq_ids = list(actual_eager_seq_ids)
        plan.send_combined_payload_kind = "combined"
        plan.send_combined_payload_plan_id = int(plan.plan_id)
        plan.send_combined_payload_step_id = None if plan.step_id is None else int(plan.step_id)
        if expected_eager_seq_ids and not actual_eager_seq_ids:
            plan.eager_draft_empty_reason = "draft_eager_set_nonempty_but_no_eager_proposals"
        assert actual_normal_seq_ids == list(expected_normal_seq_ids), self._proposal_assertion_message(
            plan,
            f"send normal proposal seq_id mismatch: expected={expected_normal_seq_ids}, actual={actual_normal_seq_ids}, "
            f"actual_eager_seq_ids={actual_eager_seq_ids}",
        )
        assert actual_eager_seq_ids == list(expected_eager_seq_ids), self._proposal_assertion_message(
            plan,
            f"send eager proposal seq_id mismatch: expected={expected_eager_seq_ids}, actual={actual_eager_seq_ids}, "
            f"actual_normal_seq_ids={actual_normal_seq_ids}",
        )
        if expected_eager_seq_ids and not actual_eager_seq_ids:
            assert False, self._proposal_assertion_message(
                plan,
                f"eager draft produced no proposals for expected_eager_seq_ids={expected_eager_seq_ids}",
            )
        if self.tp_params.local_rank != 0:
            return
        normal_payload = self._combined_normal_payload(normal_proposals)
        eager_payload = self._combined_eager_payload(eager_proposals)
        payload = normal_payload + eager_payload
        meta = torch.tensor(
            [
                1,  # proposal_message_kind="combined"
                len(payload),
                int(self.gamma),
                int(plan.plan_id),
                -1 if plan.step_id is None else int(plan.step_id),
                -1 if plan.draft_batch_id is None else int(plan.draft_batch_id),
                len(normal_proposals),
                len(normal_payload),
                len(eager_proposals),
                len(eager_payload),
            ],
            dtype=torch.int64,
            device="cuda",
        )
        payload_tensor = torch.tensor(payload, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta[1].item()) > 0:
            dist.broadcast(payload_tensor, src=self.global_config.draft_config.master_rank, group=self.verify_group)

    def _parse_combined_normal_payload(
        self,
        payload: list[int],
        n: int,
        proposal_plan_id: int,
        expected_seq_ids: list[int],
    ) -> list[BufferedProposal]:
        header_len = n * 5
        headers = payload[:header_len]
        token_data = payload[header_len:]
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
        return proposals

    def _parse_combined_eager_payload(
        self,
        payload: list[int],
        n: int,
        expected_seq_ids: list[int],
        plan: StepPlan,
    ) -> list[EagerBufferedProposal]:
        header_len = n * 7
        headers = payload[:header_len]
        token_data = payload[header_len:]
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
                    verify_with_batch_id=None if plan.draft_batch_id is None else int(plan.draft_batch_id),
                    score=float(plan.eager_score_by_seq_id.get(int(seq_id), 0.0)),
                    policy=plan.eager_policy,
                    valid=True,
                    ready=False,
                )
            )
        return proposals

    def _receive_combined_dual_proposals(
        self,
        expected_normal_seq_ids: list[int],
        expected_eager_seq_ids: list[int],
        plan: StepPlan,
    ) -> tuple[list[BufferedProposal], list[EagerBufferedProposal]]:
        meta = torch.zeros(10, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        (
            proposal_kind,
            payload_len,
            gamma,
            proposal_plan_id,
            proposal_step_id,
            batch_id,
            normal_n,
            normal_payload_len,
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
            dist.broadcast(payload_tensor, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        payload = payload_tensor.tolist()
        normal_payload = payload[:normal_payload_len]
        eager_payload = payload[normal_payload_len:normal_payload_len + eager_payload_len]
        normal_proposals = self._parse_combined_normal_payload(
            normal_payload,
            normal_n,
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
        received_eager_seq_ids = [proposal.seq_id for proposal in eager_proposals]
        plan.expected_normal_receive_seq_ids = list(expected_normal_seq_ids)
        plan.received_normal_seq_ids = list(received_normal_seq_ids)
        plan.received_eager_seq_ids = list(received_eager_seq_ids)
        plan.proposal_message_kind = "combined"
        plan.proposal_message_plan_id = int(proposal_plan_id)
        plan.proposal_message_step_id = int(proposal_step_id)
        assert received_normal_seq_ids == list(expected_normal_seq_ids), self._proposal_assertion_message(
            plan,
            f"normal proposal seq_id mismatch: expected={expected_normal_seq_ids}, received={received_normal_seq_ids}, "
            f"received_eager_seq_ids={received_eager_seq_ids}, batch_id={batch_id}, "
            f"proposal_plan_id={proposal_plan_id}, proposal_step_id={proposal_step_id}",
        )
        assert received_eager_seq_ids == list(expected_eager_seq_ids), self._proposal_assertion_message(
            plan,
            f"eager proposal seq_id mismatch: expected={expected_eager_seq_ids}, received={received_eager_seq_ids}, "
            f"received_normal_seq_ids={received_normal_seq_ids}, batch_id={batch_id}, "
            f"proposal_plan_id={proposal_plan_id}, proposal_step_id={proposal_step_id}",
        )
        return normal_proposals, eager_proposals

    def _apply_combined_proposal_trace_fields(self, trace_record: dict, plan: StepPlan) -> None:
        trace_record["expected_normal_receive_seq_ids"] = list(plan.expected_normal_receive_seq_ids)
        trace_record["received_normal_seq_ids"] = list(plan.received_normal_seq_ids)
        trace_record["received_eager_seq_ids"] = list(plan.received_eager_seq_ids)
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
                f"proposal_base={proposal.eager_base_len}, current_len={len(seq)}",
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

    def _receive_eager_verify_result(self, seqs: list[Sequence]) -> torch.Tensor:
        verify_res = torch.zeros((6, len(seqs)), dtype=torch.int64, device="cuda")
        if seqs:
            dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
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
                    f"proposal_base={proposal.eager_base_len}, current_len={len(seq)}",
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
                )
            )

        for seq in seqs:
            if generated[seq.seq_id]:
                self.scheduler.rollback(seq, len(generated[seq.seq_id]))
            assert len(seq) == base_lens[seq.seq_id], self._proposal_assertion_message(
                plan,
                f"eager draft rollback failed for seq_id={seq.seq_id}: expected_len={base_lens[seq.seq_id]}, got={len(seq)}",
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
        target_eager_seqs = self._resolve_dual_seq_ids(plan.target_eager_set, plan, "draft_apply_eager_verify")
        draft_seqs = self._resolve_dual_seq_ids(plan.draft_home_set, plan, "dual_draft")
        draft_eager_seqs = self._resolve_dual_seq_ids(plan.draft_eager_set, plan, "dual_eager_draft")

        expected_normal_seq_ids = list(plan.normal_proposal_refresh_seq_ids or plan.draft_home_set)
        expected_eager_seq_ids = list(plan.draft_eager_set) if self.global_config.enable_eager_execution else []
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

        normal_proposals = []
        draft_records = []
        if draft_seqs:
            normal_proposals, draft_records = self._draft_dual_batch_proposals(draft_seqs, plan)
            if plan.plan_phase in {"priming", "steady"}:
                self.dual_proposal_buffer.store(normal_proposals)
            for trace_record in draft_records:
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                self._finalize_record_profile(trace_record)

        eager_proposals = []
        eager_trace_record = None
        if draft_eager_seqs and self.global_config.enable_eager_execution:
            eager_proposals, eager_trace_record = self._draft_eager_proposals(draft_eager_seqs, plan)
            self.eager_proposal_buffer.store(eager_proposals)
            if eager_trace_record is not None:
                eager_trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()
                eager_trace_record["proposal_buffer_keys_after_eager_draft"] = self.dual_proposal_buffer.pending_seq_ids()
                self._finalize_record_profile(eager_trace_record)
        elif plan.draft_eager_set and not self.global_config.enable_eager_execution:
            plan.eager_draft_skipped_reason = "eager_execution_disabled"
            plan.eager_draft_failed_seq_ids = list(plan.draft_eager_set)

        if draft_seqs or expected_eager_seq_ids:
            self._send_combined_dual_proposals(
                normal_proposals,
                eager_proposals,
                plan,
                expected_normal_seq_ids,
                expected_eager_seq_ids,
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
            accepted_lens, invalidated_lens = self._apply_verify_result(target_seqs, verify_res)
            consumed_seq_ids = self.dual_proposal_buffer.discard([seq.seq_id for seq in target_seqs])
            trace_record["proposal_buffer_consumed_seq_ids"] = consumed_seq_ids
            trace_record["proposal_buffer_consumed_count"] = len(consumed_seq_ids)
            trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
            self._promote_or_discard_eager_after_normal(
                plan,
                accepted_lens,
                invalidated_lens,
                trace_record,
                count_tokens=False,
            )
            torch.cuda.synchronize()
            self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

        if target_eager_seqs:
            trace_record = self._trace_dual_batch_schedule(target_eager_seqs, plan, "draft_apply_eager_verify")
            eager_verify_res = self._receive_eager_verify_result(target_eager_seqs)
            eager_ready = self.eager_proposal_buffer.get_many([seq.seq_id for seq in target_eager_seqs], ready_only=True)
            self._apply_eager_verify_result(
                target_eager_seqs,
                eager_ready,
                eager_verify_res,
                plan,
                trace_record,
                count_tokens=False,
            )
    
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
    ) -> torch.Tensor:
        verify_res = torch.zeros((6, len(seqs)), dtype=torch.int64, device="cuda")
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
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
        return verify_res

    def _run_eager_verify_sidecar(
        self,
        seqs: list[Sequence],
        proposals: list[EagerBufferedProposal],
        plan: StepPlan,
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
        verify_res = self._build_eager_verify_result(logits, seqs, temperatures, proposals, plan)
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
        self._finalize_record_profile(trace_record)

    def dual_batch_pearl_step(self):
        plan = self._build_dual_batch_step_plan()
        target_seqs = self._resolve_dual_seq_ids(plan.target_home_set, plan, "dual_verify")
        target_eager_seqs = self._resolve_dual_seq_ids(plan.target_eager_set, plan, "dual_eager_verify")
        draft_seq_ids = list(plan.draft_home_set)
        target_seq_ids = [seq.seq_id for seq in target_seqs]
        target_eager_seq_ids = [seq.seq_id for seq in target_eager_seqs]
        assert set(target_eager_seq_ids).isdisjoint(draft_seq_ids), self._proposal_assertion_message(
            plan,
            f"target_eager_set overlaps draft_home_set: {target_eager_seq_ids}",
        )
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

        received_proposals = []
        received_eager_proposals = []
        expected_eager_seq_ids = list(plan.draft_eager_set) if self.global_config.enable_eager_execution else []
        if draft_seq_ids or expected_eager_seq_ids:
            received_proposals, received_eager_proposals = self._receive_combined_dual_proposals(
                draft_seq_ids,
                expected_eager_seq_ids,
                plan,
            )
            if fallback_same_batch:
                target_proposals = received_proposals
                if trace_record is not None:
                    trace_record["proposal_tokens_available"] = sum(len(p.to_be_verified_token_ids) for p in target_proposals)
                    trace_record["proposal_tokens_verified"] = trace_record["proposal_tokens_available"]
            else:
                if received_proposals:
                    self.dual_proposal_buffer.store(received_proposals)
            if received_eager_proposals:
                self.eager_proposal_buffer.store(received_eager_proposals)
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
                logits,
                target_seqs,
                temperatures,
                target_proposals,
                plan,
            )
            self._promote_or_discard_eager_after_normal(
                plan,
                accepted_lens,
                invalidated_lens,
                trace_record,
                count_tokens=True,
            )
            torch.cuda.synchronize()
            self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
        elif received_proposals or received_eager_proposals:
            priming_record = self._trace_dual_batch_schedule([], plan, "dual_verify_idle")
            priming_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
            self._apply_combined_proposal_trace_fields(priming_record, plan)
            self._finalize_record_profile(priming_record)

        if target_eager_seqs:
            self._run_eager_verify_sidecar(target_eager_seqs, eager_proposals, plan)

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
