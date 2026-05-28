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
    EAGER_STATE_PENDING_BASE_REACHED,
    EAGER_STATE_READY_TO_VERIFY,
    EAGER_STATE_READY_TO_VERIFY_DRY_RUN,
    EAGER_STATE_SCHEDULED_DRY_RUN,
    EAGER_STATE_TRANSFERRED_DRY_RUN,
    EagerProposal,
    EagerProposalBuffer,
    LANE_EAGER,
    LANE_NORMAL,
    ProposalBuffer,
    READY_EAGER_STATE_CONSUMED_APPLIED,
    READY_EAGER_TRANSFER_HEADER_LEN,
    ReadyEagerProposal,
    deserialize_eager_transfer_payload,
    deserialize_ready_eager_proposals,
    ready_eager_proposal_from_eager_proposal,
    serialize_eager_transfer_payload,
    serialize_ready_eager_proposals,
)
from transformers import AutoTokenizer
from tqdm import trange


EAGER_TAKEOVER_DRY_RUN_SOURCE = "phase1h5e3_takeover_lane"
CONTINUOUS_EAGER_DRY_RUN_SOURCE = "continuous_shadow"
CONTINUOUS_EAGER_PARENT_SOURCE = "phase1h6a_one_shot_commit"
ROLLING_CONTINUOUS_EAGER_DRY_RUN_SOURCE = "rolling_continuous_shadow"
ROLLING_CONTINUOUS_STAGE = "overlap_dry_run"
CONTINUOUS_EAGER_TRANSFER_MAGIC = 0x1A70B
CONTINUOUS_EAGER_TRANSFER_OP_DRY_RUN = 0x1A70B1
CONTINUOUS_EAGER_TRANSFER_META_LEN = 7
CONTINUOUS_EAGER_COMMIT_SOURCE = "continuous_depth1_ready_only"
EAGER_LEGACY_RESULT_TRANSFER_SOURCE = "scheduled_target_eager_lane"
EAGER_RESULT_TRANSFER_MAGIC = 0x1A5E3
EAGER_RESULT_TRANSFER_OP_DRY_RUN = 0x1A5E35
EAGER_RESULT_TRANSFER_META_LEN = 7
EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH = 31
CONTINUOUS_EAGER_RESULT_TRANSFER_MAGIC = 0x1A72D
CONTINUOUS_EAGER_RESULT_TRANSFER_OP_COMPACT_V1 = 0x1A72D1
CONTINUOUS_EAGER_RESULT_TRANSFER_PROTOCOL_COMPACT_V1 = "compact_v1"
CONTINUOUS_EAGER_RESULT_TRANSFER_META_LEN = 7
CONTINUOUS_EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH = 13
EAGER_COMMIT_READY_ONLY_MAGIC = 0x1A60A
EAGER_COMMIT_READY_ONLY_OP = 0x1A60A1
EAGER_COMMIT_READY_ONLY_META_LEN = 6
EAGER_COMMIT_READY_ONLY_PAYLOAD_WIDTH = 4
CONTINUOUS_EAGER_COMMIT_DEPTH1_MAGIC = 0x1A71C
CONTINUOUS_EAGER_COMMIT_DEPTH1_OP = 0x1A71C1
CONTINUOUS_EAGER_COMMIT_DEPTH1_META_LEN = 6
CONTINUOUS_EAGER_COMMIT_DEPTH1_PAYLOAD_WIDTH = 4


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
        self._draft_sent_eager_proposals_by_id = {}
        self._eager_result_transfer_received_proposal_ids = set()
        self._eager_committed_proposal_ids = set()
        self._continuous_eager_committed_proposal_ids = set()
        self._rolling_continuous_shadow_proposals_by_id = {}
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
            "normal_proposal_transfer_called": False,
            "eager_proposal_transfer_called": False,
            "result_transfer_called": False,
            "result_transfer_zero_result": False,
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
            "enable_eager_transfer_dry_run": bool(
                getattr(self.global_config, "enable_eager_transfer_dry_run", False)
            ),
            "eager_transfer_dry_run_enabled": False,
            "enable_eager_schedule_dry_run": bool(
                getattr(self.global_config, "enable_eager_schedule_dry_run", False)
            ),
            "eager_schedule_dry_run_enabled": False,
            "enable_eager_verify_dry_run": bool(
                getattr(self.global_config, "enable_eager_verify_dry_run", False)
            ),
            "eager_verify_dry_run_enabled": False,
            "eager_verify_dry_run_source": None,
            "eager_verify_dry_run_step_id": None,
            "eager_verify_dry_run_plan_id": None,
            "eager_verify_dry_run_candidate_proposal_ids": [],
            "eager_verify_dry_run_candidate_seq_ids": [],
            "eager_verify_dry_run_executed_proposal_ids": [],
            "eager_verify_dry_run_executed_seq_ids": [],
            "eager_verify_dry_run_skipped_proposal_ids": [],
            "eager_verify_dry_run_skip_reason_by_proposal_id": {},
            "eager_verify_dry_run_seq_id_by_proposal_id": {},
            "eager_verify_dry_run_proposal_len_by_proposal_id": {},
            "eager_verify_dry_run_to_verify_len_by_proposal_id": {},
            "eager_verify_dry_run_base_len_by_proposal_id": {},
            "eager_verify_dry_run_takeover_step_by_proposal_id": {},
            "eager_verify_dry_run_accept_len_by_proposal_id": {},
            "eager_verify_dry_run_result_by_proposal_id": {},
            "eager_verify_dry_run_reject_position_by_proposal_id": {},
            "eager_verify_dry_run_full_accept_proposal_ids": [],
            "eager_verify_dry_run_rejected_proposal_ids": [],
            "eager_verify_dry_run_partial_accept_proposal_ids": [],
            "eager_verify_dry_run_mutation_detected": False,
            "eager_verify_dry_run_checkpoint_failed": False,
            "eager_verify_dry_run_sequence_len_before_by_seq_id": {},
            "eager_verify_dry_run_sequence_len_after_by_seq_id": {},
            "eager_verify_dry_run_executed_proposal_count": 0,
            "eager_verify_dry_run_skipped_proposal_count": 0,
            "eager_verify_candidate_proposal_ids": [],
            "eager_verify_candidate_seq_ids": [],
            "eager_verify_skipped_proposal_ids": [],
            "eager_verify_skip_reason_by_proposal_id": {},
            "eager_verify_executed_proposal_ids": [],
            "eager_verify_executed_seq_ids": [],
            "eager_verify_accepted_len_by_seq_id": {},
            "eager_verify_full_accept_by_seq_id": {},
            "eager_verify_reject_position_by_seq_id": {},
            "eager_verify_invalidated_len_by_seq_id": {},
            "eager_verify_revised_token_by_seq_id": {},
            "eager_verify_base_len_by_seq_id": {},
            "eager_verify_current_len_by_seq_id": {},
            "eager_verify_base_match_by_seq_id": {},
            "eager_verify_seq_pre_verify_by_seq_id": {},
            "eager_verify_seq_status_before_by_seq_id": {},
            "eager_verify_seq_status_after_by_seq_id": {},
            "eager_verify_mutation_detected_by_seq_id": {},
            "eager_verify_checkpoint_ok_by_seq_id": {},
            "enable_eager_apply_dry_run": bool(
                getattr(self.global_config, "enable_eager_apply_dry_run", False)
            ),
            "eager_apply_dry_run_enabled": False,
            "eager_apply_dry_run_source": None,
            "eager_apply_dry_run_step_id": None,
            "eager_apply_dry_run_plan_id": None,
            "eager_apply_dry_run_candidate_proposal_ids": [],
            "eager_apply_dry_run_candidate_seq_ids": [],
            "eager_apply_dry_run_executed_proposal_ids": [],
            "eager_apply_dry_run_executed_seq_ids": [],
            "eager_apply_dry_run_skipped_proposal_ids": [],
            "eager_apply_dry_run_skip_reason_by_proposal_id": {},
            "eager_apply_dry_run_from_verify_proposal_ids": [],
            "eager_apply_dry_run_verify_result_by_proposal_id": {},
            "eager_apply_dry_run_accept_len_by_proposal_id": {},
            "eager_apply_dry_run_seq_id_by_proposal_id": {},
            "eager_apply_dry_run_proposal_len_by_proposal_id": {},
            "eager_apply_dry_run_action_by_proposal_id": {},
            "eager_apply_dry_run_full_accept_proposal_ids": [],
            "eager_apply_dry_run_discarded_proposal_ids": [],
            "eager_apply_dry_run_append_tokens_by_proposal_id": {},
            "eager_apply_dry_run_discarded_tokens_by_proposal_id": {},
            "eager_apply_dry_run_rollback_ok_by_proposal_id": {},
            "eager_apply_dry_run_mutation_detected_by_proposal_id": {},
            "eager_apply_dry_run_checkpoint_failed_by_proposal_id": {},
            "eager_apply_dry_run_sequence_len_before_by_seq_id": {},
            "eager_apply_dry_run_sequence_len_after_by_seq_id": {},
            "eager_apply_dry_run_pre_verify_before_by_seq_id": {},
            "eager_apply_dry_run_pre_verify_after_by_seq_id": {},
            "eager_apply_dry_run_status_before_by_seq_id": {},
            "eager_apply_dry_run_status_after_by_seq_id": {},
            "eager_apply_dry_run_executed_proposal_count": 0,
            "eager_apply_dry_run_skipped_proposal_count": 0,
            "eager_apply_candidate_proposal_ids": [],
            "eager_apply_candidate_seq_ids": [],
            "eager_apply_executed_proposal_ids": [],
            "eager_apply_executed_seq_ids": [],
            "eager_apply_skipped_proposal_ids": [],
            "eager_apply_skip_reason_by_proposal_id": {},
            "eager_apply_action_by_seq_id": {},
            "eager_apply_accepted_len_by_seq_id": {},
            "eager_apply_full_accept_by_seq_id": {},
            "eager_apply_base_len_by_seq_id": {},
            "eager_apply_current_len_before_by_seq_id": {},
            "eager_apply_current_len_after_simulated_apply_by_seq_id": {},
            "eager_apply_current_len_after_rollback_by_seq_id": {},
            "eager_apply_pre_verify_before_by_seq_id": {},
            "eager_apply_pre_verify_after_simulated_apply_by_seq_id": {},
            "eager_apply_pre_verify_after_rollback_by_seq_id": {},
            "eager_apply_status_before_by_seq_id": {},
            "eager_apply_status_after_simulated_apply_by_seq_id": {},
            "eager_apply_status_after_rollback_by_seq_id": {},
            "eager_apply_checkpoint_ok_by_seq_id": {},
            "eager_apply_rollback_ok_by_seq_id": {},
            "eager_apply_mutation_remaining_by_seq_id": {},
            "eager_apply_dry_run_appended_token_count_by_seq_id": {},
            "enable_eager_result_transfer_dry_run": bool(
                getattr(self.global_config, "enable_eager_result_transfer_dry_run", False)
            ),
            "eager_result_transfer_dry_run_enabled": False,
            "eager_result_transfer_dry_run_source": None,
            "eager_result_transfer_step_id": None,
            "eager_result_transfer_plan_id": None,
            "eager_result_transfer_send_step": None,
            "eager_result_transfer_send_plan": None,
            "eager_result_transfer_num_results": 0,
            "eager_result_transfer_payload_len": 0,
            "eager_result_transfer_sent_count": 0,
            "eager_result_transfer_sent_result_count": 0,
            "eager_result_transfer_received_result_count": 0,
            "eager_result_transfer_validated_result_count": 0,
            "eager_result_transfer_invalid_result_count": 0,
            "eager_result_transfer_zero_result_step": False,
            "eager_result_transfer_sent_proposal_ids": [],
            "eager_result_transfer_sent_seq_ids": [],
            "eager_result_transfer_action_by_proposal_id": {},
            "eager_result_transfer_verify_result_by_proposal_id": {},
            "eager_result_transfer_accept_len_by_proposal_id": {},
            "eager_result_transfer_append_tokens_by_proposal_id": {},
            "eager_result_transfer_discarded_tokens_by_proposal_id": {},
            "eager_result_transfer_rollback_ok_by_proposal_id": {},
            "eager_result_transfer_mutation_detected_by_proposal_id": {},
            "eager_result_transfer_checkpoint_failed_by_proposal_id": {},
            "eager_result_sent_proposal_ids": [],
            "eager_result_sent_seq_ids": [],
            "eager_result_sent_accepted_len_by_seq_id": {},
            "eager_result_sent_full_accept_by_seq_id": {},
            "eager_result_sent_reject_position_by_seq_id": {},
            "eager_result_sent_invalidated_len_by_seq_id": {},
            "eager_result_sent_revised_token_by_seq_id": {},
            "eager_result_sent_apply_action_by_seq_id": {},
            "eager_result_sent_proposal_len_by_proposal_id": {},
            "eager_result_sent_to_verify_len_by_proposal_id": {},
            "eager_result_received_num_results": 0,
            "eager_result_received_payload_len": 0,
            "eager_result_transfer_received_proposal_ids": [],
            "eager_result_transfer_received_seq_ids": [],
            "eager_result_transfer_received_count": 0,
            "eager_result_transfer_validated_proposal_ids": [],
            "eager_result_transfer_invalid_proposal_ids": [],
            "eager_result_transfer_validation_reason_by_proposal_id": {},
            "eager_result_transfer_duplicate_proposal_ids": [],
            "eager_result_transfer_missing_local_proposal_ids": [],
            "eager_result_transfer_seq_mismatch_proposal_ids": [],
            "eager_result_transfer_bad_action_proposal_ids": [],
            "eager_result_transfer_bad_accept_len_proposal_ids": [],
            "eager_result_transfer_draft_mutation_detected": False,
            "eager_result_transfer_draft_checkpoint_failed": False,
            "eager_result_received_proposal_ids": [],
            "eager_result_received_seq_ids": [],
            "eager_result_validated_proposal_ids": [],
            "eager_result_invalid_proposal_ids": [],
            "eager_result_validation_reason_by_proposal_id": {},
            "eager_result_received_accepted_len_by_seq_id": {},
            "eager_result_received_full_accept_by_seq_id": {},
            "eager_result_received_reject_position_by_seq_id": {},
            "eager_result_received_invalidated_len_by_seq_id": {},
            "eager_result_received_revised_token_by_seq_id": {},
            "eager_result_received_proposal_len_by_proposal_id": {},
            "eager_result_received_to_verify_len_by_proposal_id": {},
            "eager_result_draft_current_len_by_seq_id": {},
            "eager_result_base_len_by_seq_id": {},
            "eager_result_draft_len_matches_base_by_seq_id": {},
            "eager_result_draft_seq_pre_verify_by_seq_id": {},
            "eager_result_draft_status_before_by_seq_id": {},
            "eager_result_draft_status_after_by_seq_id": {},
            "eager_result_draft_checkpoint_ok_by_seq_id": {},
            "eager_result_draft_mutation_detected_by_seq_id": {},
            "eager_result_zero_result_step": False,
            "enable_eager_sync_apply_dry_run": bool(
                getattr(self.global_config, "enable_eager_sync_apply_dry_run", False)
            ),
            "eager_sync_apply_dry_run_enabled": False,
            "eager_sync_apply_dry_run_source": None,
            "eager_sync_apply_step_id": None,
            "eager_sync_apply_plan_id": None,
            "eager_sync_apply_candidate_proposal_ids": [],
            "eager_sync_apply_candidate_seq_ids": [],
            "eager_sync_apply_executed_proposal_ids": [],
            "eager_sync_apply_executed_seq_ids": [],
            "eager_sync_apply_skipped_proposal_ids": [],
            "eager_sync_apply_skip_reason_by_proposal_id": {},
            "eager_sync_apply_action_by_seq_id": {},
            "eager_sync_apply_accepted_len_by_seq_id": {},
            "eager_sync_apply_full_accept_by_seq_id": {},
            "eager_sync_apply_dry_run_candidate_proposal_ids": [],
            "eager_sync_apply_dry_run_candidate_seq_ids": [],
            "eager_sync_apply_dry_run_from_result_transfer_proposal_ids": [],
            "eager_sync_apply_dry_run_from_result_transfer_seq_ids": [],
            "eager_sync_apply_dry_run_executed_proposal_ids": [],
            "eager_sync_apply_dry_run_executed_seq_ids": [],
            "eager_sync_apply_dry_run_skipped_proposal_ids": [],
            "eager_sync_apply_dry_run_skip_reason_by_proposal_id": {},
            "eager_sync_apply_dry_run_target_action_by_proposal_id": {},
            "eager_sync_apply_dry_run_draft_action_by_proposal_id": {},
            "eager_sync_apply_dry_run_target_verify_result_by_proposal_id": {},
            "eager_sync_apply_dry_run_draft_verify_result_by_proposal_id": {},
            "eager_sync_apply_dry_run_target_accept_len_by_proposal_id": {},
            "eager_sync_apply_dry_run_draft_accept_len_by_proposal_id": {},
            "eager_sync_apply_dry_run_action_match_by_proposal_id": {},
            "eager_sync_apply_dry_run_accept_len_match_by_proposal_id": {},
            "eager_sync_apply_dry_run_result_match_by_proposal_id": {},
            "eager_sync_apply_dry_run_append_tokens_by_proposal_id": {},
            "eager_sync_apply_dry_run_discarded_tokens_by_proposal_id": {},
            "eager_sync_apply_dry_run_rollback_ok_by_proposal_id": {},
            "eager_sync_apply_dry_run_mutation_detected_by_proposal_id": {},
            "eager_sync_apply_dry_run_checkpoint_failed_by_proposal_id": {},
            "eager_sync_apply_dry_run_sequence_len_before_by_seq_id": {},
            "eager_sync_apply_dry_run_sequence_len_after_by_seq_id": {},
            "eager_sync_apply_dry_run_pre_verify_before_by_seq_id": {},
            "eager_sync_apply_dry_run_pre_verify_after_by_seq_id": {},
            "eager_sync_apply_dry_run_status_before_by_seq_id": {},
            "eager_sync_apply_dry_run_status_after_by_seq_id": {},
            "eager_sync_apply_dry_run_consistent_proposal_ids": [],
            "eager_sync_apply_dry_run_inconsistent_proposal_ids": [],
            "eager_sync_apply_dry_run_missing_local_proposal_ids": [],
            "eager_sync_apply_dry_run_duplicate_proposal_ids": [],
            "enable_eager_commit_readiness_dry_run": bool(
                getattr(self.global_config, "enable_eager_commit_readiness_dry_run", False)
            ),
            "eager_commit_readiness_dry_run_enabled": False,
            "eager_commit_readiness_dry_run_source": None,
            "eager_commit_readiness_candidate_proposal_ids": [],
            "eager_commit_readiness_candidate_seq_ids": [],
            "eager_commit_readiness_from_sync_apply_proposal_ids": [],
            "eager_commit_readiness_from_result_transfer_proposal_ids": [],
            "eager_commit_readiness_from_apply_proposal_ids": [],
            "eager_commit_readiness_from_verify_proposal_ids": [],
            "eager_commit_ready_proposal_ids": [],
            "eager_commit_ready_seq_ids": [],
            "eager_commit_ready_token_count_by_proposal_id": {},
            "eager_commit_ready_action_by_proposal_id": {},
            "eager_commit_ready_accept_len_by_proposal_id": {},
            "eager_commit_ready_verify_result_by_proposal_id": {},
            "eager_commit_not_ready_proposal_ids": [],
            "eager_commit_not_ready_seq_ids": [],
            "eager_commit_not_ready_reason_by_proposal_id": {},
            "eager_commit_readiness_verify_ok_by_proposal_id": {},
            "eager_commit_readiness_apply_ok_by_proposal_id": {},
            "eager_commit_readiness_result_transfer_ok_by_proposal_id": {},
            "eager_commit_readiness_sync_apply_ok_by_proposal_id": {},
            "eager_commit_readiness_frontier_ok_by_proposal_id": {},
            "eager_commit_readiness_token_payload_ok_by_proposal_id": {},
            "eager_commit_readiness_no_mutation_by_proposal_id": {},
            "eager_commit_readiness_actual_counters_zero": True,
            "eager_commit_readiness_real_target_eager_empty": True,
            "eager_commit_readiness_candidate_count": 0,
            "eager_commit_ready_count": 0,
            "eager_commit_not_ready_count": 0,
            "eager_commit_ready_token_count": 0,
            "eager_commit_not_ready_reason_counts": {},
            "eager_commit_readiness_full_accept_count": 0,
            "eager_commit_readiness_partial_reject_count": 0,
            "enable_eager_commit_ready_only": bool(
                getattr(self.global_config, "enable_eager_commit_ready_only", False)
            ),
            "eager_commit_enabled": False,
            "eager_commit_source": None,
            "eager_commit_candidate_proposal_ids": [],
            "eager_commit_candidate_seq_ids": [],
            "eager_commit_from_readiness_proposal_ids": [],
            "eager_committed_proposal_ids": [],
            "eager_committed_seq_ids": [],
            "eager_committed_token_count_by_proposal_id": {},
            "eager_committed_accept_len_by_proposal_id": {},
            "eager_committed_action_by_proposal_id": {},
            "eager_committed_verify_result_by_proposal_id": {},
            "eager_commit_skipped_proposal_ids": [],
            "eager_commit_skip_reason_by_proposal_id": {},
            "eager_commit_precondition_ok_by_proposal_id": {},
            "eager_commit_precondition_failed_by_proposal_id": {},
            "eager_commit_precondition_failure_reason_by_proposal_id": {},
            "eager_commit_duplicate_proposal_ids": [],
            "eager_commit_duplicate_seq_ids": [],
            "eager_commit_target_seq_len_before_by_seq_id": {},
            "eager_commit_target_seq_len_after_by_seq_id": {},
            "eager_commit_draft_seq_len_before_by_seq_id": {},
            "eager_commit_draft_seq_len_after_by_seq_id": {},
            "eager_commit_target_draft_len_match_by_seq_id": {},
            "eager_commit_target_draft_token_match_by_seq_id": {},
            "eager_commit_frontier_ok_by_proposal_id": {},
            "eager_commit_token_payload_ok_by_proposal_id": {},
            "enable_continuous_eager_dry_run": bool(
                getattr(self.global_config, "enable_continuous_eager_dry_run", False)
            ),
            "enable_continuous_eager_verify_apply_dry_run": bool(
                getattr(self.global_config, "enable_continuous_eager_verify_apply_dry_run", False)
            ),
            "enable_continuous_eager_commit_depth1_ready_only": bool(
                getattr(self.global_config, "enable_continuous_eager_commit_depth1_ready_only", False)
            ),
            "enable_rolling_continuous_eager_dry_run": bool(
                getattr(self.global_config, "enable_rolling_continuous_eager_dry_run", False)
            ),
            "continuous_eager_dry_run_enabled": False,
            "continuous_eager_source": None,
            "continuous_eager_parent_source": None,
            "continuous_eager_execution_stage": None,
            "continuous_shadow_stage": None,
            "max_continuous_eager_chain_depth": int(
                getattr(self.global_config, "max_continuous_eager_chain_depth", 0) or 0
            ),
            "max_continuous_depth_configured": int(
                getattr(self.global_config, "max_continuous_eager_chain_depth", 0) or 0
            ),
            "max_continuous_depth_observed": 0,
            "max_continuous_eager_requests_per_step": int(
                getattr(self.global_config, "max_continuous_eager_requests_per_step", 0) or 0
            ),
            "max_continuous_eager_tokens_per_step": int(
                getattr(self.global_config, "max_continuous_eager_tokens_per_step", 0) or 0
            ),
            "max_continuous_eager_tokens_per_request": int(
                getattr(self.global_config, "max_continuous_eager_tokens_per_request", 0) or 0
            ),
            "continuous_eager_candidate_proposal_ids": [],
            "continuous_eager_candidate_seq_ids": [],
            "continuous_eager_candidate_token_count_by_proposal_id": {},
            "continuous_eager_parent_proposal_id_by_proposal_id": {},
            "continuous_eager_chain_depth_by_proposal_id": {},
            "continuous_eager_chain_index_by_proposal_id": {},
            "continuous_eager_root_proposal_id_by_proposal_id": {},
            "continuous_eager_parent_source_by_proposal_id": {},
            "continuous_eager_verified_proposal_ids": [],
            "continuous_eager_full_accept_proposal_ids": [],
            "continuous_eager_partial_reject_proposal_ids": [],
            "continuous_eager_accept_len_by_proposal_id": {},
            "continuous_eager_verify_result_by_proposal_id": {},
            "continuous_eager_verify_dry_run_candidate_proposal_ids": [],
            "continuous_eager_verify_dry_run_executed_proposal_ids": [],
            "continuous_eager_verify_dry_run_skipped_proposal_ids": [],
            "continuous_eager_verify_skip_reason_by_proposal_id": {},
            "continuous_eager_verified_token_count": 0,
            "continuous_eager_full_accept_token_count": 0,
            "continuous_eager_partial_reject_token_count": 0,
            "continuous_eager_apply_dry_run_candidate_proposal_ids": [],
            "continuous_eager_apply_dry_run_executed_proposal_ids": [],
            "continuous_eager_apply_action_by_proposal_id": {},
            "continuous_eager_apply_append_tokens_by_proposal_id": {},
            "continuous_eager_apply_discarded_tokens_by_proposal_id": {},
            "continuous_eager_apply_rollback_ok_by_proposal_id": {},
            "continuous_eager_apply_mutation_detected_by_proposal_id": {},
            "continuous_eager_apply_checkpoint_failed_by_proposal_id": {},
            "continuous_eager_result_transfer_sent_proposal_ids": [],
            "continuous_eager_result_transfer_received_proposal_ids": [],
            "continuous_eager_result_transfer_validated_proposal_ids": [],
            "continuous_eager_result_transfer_invalid_proposal_ids": [],
            "continuous_eager_result_transfer_validation_reason_by_proposal_id": {},
            "continuous_eager_result_transfer_payload_len_units": 0,
            "continuous_eager_result_transfer_zero_steps": 0,
            "continuous_eager_sync_apply_candidate_proposal_ids": [],
            "continuous_eager_sync_apply_executed_proposal_ids": [],
            "continuous_eager_sync_apply_action_match_by_proposal_id": {},
            "continuous_eager_sync_apply_result_match_by_proposal_id": {},
            "continuous_eager_sync_apply_accept_len_match_by_proposal_id": {},
            "continuous_eager_sync_apply_rollback_ok_by_proposal_id": {},
            "continuous_eager_sync_apply_mutation_detected_by_proposal_id": {},
            "continuous_eager_sync_apply_checkpoint_failed_by_proposal_id": {},
            "continuous_eager_sync_apply_target_action_by_proposal_id": {},
            "continuous_eager_sync_apply_draft_action_by_proposal_id": {},
            "continuous_eager_sync_apply_target_verify_result_by_proposal_id": {},
            "continuous_eager_sync_apply_draft_verify_result_by_proposal_id": {},
            "continuous_eager_sync_apply_target_accept_len_by_proposal_id": {},
            "continuous_eager_sync_apply_draft_accept_len_by_proposal_id": {},
            "continuous_eager_sync_apply_append_tokens_by_proposal_id": {},
            "continuous_eager_sync_apply_discarded_tokens_by_proposal_id": {},
            "continuous_eager_commit_ready_shadow_proposal_ids": [],
            "continuous_eager_commit_ready_shadow_seq_ids": [],
            "continuous_eager_commit_ready_shadow_token_count_by_proposal_id": {},
            "continuous_eager_not_ready_shadow_proposal_ids": [],
            "continuous_eager_not_ready_shadow_reason_by_proposal_id": {},
            "continuous_eager_stale_proposal_ids": [],
            "continuous_eager_duplicate_proposal_ids": [],
            "continuous_eager_parent_not_ready_proposal_ids": [],
            "continuous_eager_parent_shadow_not_committed_proposal_ids": [],
            "continuous_eager_true_frontier_mismatch_proposal_ids": [],
            "continuous_eager_frontier_mismatch_proposal_ids": [],
            "continuous_eager_seq_finished_proposal_ids": [],
            "continuous_eager_overshot_proposal_ids": [],
            "continuous_eager_invalidated_proposal_ids": [],
            "continuous_eager_mutation_detected_count": 0,
            "continuous_eager_real_commit_count": 0,
            "continuous_depth2_real_commit_count": 0,
            "continuous_eager_commit_enabled": False,
            "continuous_eager_commit_source": None,
            "continuous_eager_commit_candidate_proposal_ids": [],
            "continuous_eager_commit_candidate_seq_ids": [],
            "continuous_eager_commit_ready_source_proposal_ids": [],
            "continuous_eager_real_committed_proposal_ids": [],
            "continuous_eager_real_committed_seq_ids": [],
            "continuous_eager_real_committed_token_count_by_proposal_id": {},
            "continuous_eager_real_committed_accept_len_by_proposal_id": {},
            "continuous_eager_real_commit_action_by_proposal_id": {},
            "continuous_eager_real_commit_verify_result_by_proposal_id": {},
            "continuous_eager_real_commit_skipped_proposal_ids": [],
            "continuous_eager_real_commit_skip_reason_by_proposal_id": {},
            "continuous_eager_real_commit_precondition_ok_by_proposal_id": {},
            "continuous_eager_real_commit_precondition_failed_by_proposal_id": {},
            "continuous_eager_real_commit_precondition_failure_reason_by_proposal_id": {},
            "continuous_eager_real_commit_duplicate_proposal_ids": [],
            "continuous_eager_real_commit_duplicate_seq_ids": [],
            "continuous_eager_target_seq_len_before_by_seq_id": {},
            "continuous_eager_target_seq_len_after_by_seq_id": {},
            "continuous_eager_draft_seq_len_before_by_seq_id": {},
            "continuous_eager_draft_seq_len_after_by_seq_id": {},
            "continuous_eager_target_draft_len_match_by_seq_id": {},
            "continuous_eager_target_draft_token_match_by_seq_id": {},
            "continuous_eager_tokens_verified": 0,
            "continuous_eager_tokens_accepted": 0,
            "continuous_eager_tokens_committed": 0,
            "continuous_eager_tokens_rejected": 0,
            "continuous_eager_tokens_invalidated": 0,
            "continuous_eager_real_committed_proposal_count": 0,
            "continuous_eager_real_committed_token_count": 0,
            "continuous_eager_unexpected_lane_overlap_count": 0,
            "continuous_eager_unexpected_takeover_overlap_count": 0,
            "continuous_eager_candidate_proposal_count": 0,
            "continuous_eager_candidate_token_count": 0,
            "continuous_eager_verified_proposal_count": 0,
            "continuous_eager_full_accept_proposal_count": 0,
            "continuous_eager_commit_ready_shadow_proposal_count": 0,
            "continuous_eager_commit_ready_shadow_token_count": 0,
            "continuous_eager_not_ready_shadow_proposal_count": 0,
            "continuous_eager_chain_length_distribution": {},
            "continuous_eager_drop_reason_counts": {},
            "continuous_parent_shadow_not_committed_count": 0,
            "continuous_parent_shadow_not_ready_count": 0,
            "continuous_true_frontier_mismatch_count": 0,
            "continuous_eager_estimated_committed_token_share_of_output": 0.0,
            "continuous_eager_payload_len_units_per_ready_token": 0.0,
            "continuous_eager_overhead_time_ms": 0.0,
            "rolling_continuous_eager_dry_run_enabled": False,
            "rolling_continuous_stage": None,
            "rolling_continuous_source": None,
            "max_rolling_continuous_depth": int(
                getattr(self.global_config, "max_rolling_continuous_depth", 0) or 0
            ),
            "max_rolling_continuous_depth_observed": 0,
            "max_rolling_continuous_draft_children_per_step": int(
                getattr(self.global_config, "max_rolling_continuous_draft_children_per_step", 0) or 0
            ),
            "max_rolling_continuous_seqs_per_step": int(
                getattr(self.global_config, "max_rolling_continuous_seqs_per_step", 0) or 0
            ),
            "target_rolling_eager_verify_proposal_ids": [],
            "target_rolling_eager_verify_seq_ids": [],
            "draft_rolling_eager_draft_proposal_ids": [],
            "draft_rolling_eager_draft_seq_ids": [],
            "rolling_same_seq_overlap_count": 0,
            "rolling_same_seq_overlap_seq_ids": [],
            "rolling_normal_lane_excluded_seq_ids": [],
            "rolling_normal_lane_conflict_seq_ids": [],
            "rolling_chain_proposal_ids": [],
            "rolling_chain_seq_ids": [],
            "rolling_chain_parent_by_proposal_id": {},
            "rolling_chain_children_by_proposal_id": {},
            "rolling_chain_root_by_proposal_id": {},
            "rolling_chain_depth_by_proposal_id": {},
            "rolling_chain_index_by_proposal_id": {},
            "rolling_chain_base_len_by_proposal_id": {},
            "rolling_chain_parent_base_len_by_proposal_id": {},
            "rolling_chain_parent_expected_accept_len_by_proposal_id": {},
            "rolling_chain_parent_source_step_id_by_proposal_id": {},
            "rolling_chain_parent_source_plan_id_by_proposal_id": {},
            "rolling_chain_parent_source_by_proposal_id": {},
            "rolling_chain_status_by_proposal_id": {},
            "rolling_chain_status_reason_by_proposal_id": {},
            "rolling_parent_verified_proposal_ids": [],
            "rolling_parent_full_accept_proposal_ids": [],
            "rolling_parent_partial_reject_proposal_ids": [],
            "rolling_parent_invalidated_proposal_ids": [],
            "rolling_child_generated_proposal_ids": [],
            "rolling_child_ready_after_parent_full_accept_proposal_ids": [],
            "rolling_child_invalidated_proposal_ids": [],
            "rolling_child_invalidated_reason_by_proposal_id": {},
            "rolling_cascade_discard_root_proposal_ids": [],
            "rolling_cascade_discarded_proposal_ids": [],
            "rolling_cascade_discard_reason_by_proposal_id": {},
            "rolling_cascade_discard_depth_by_proposal_id": {},
            "rolling_cascade_discard_count": 0,
            "rolling_depth2_real_commit_count": 0,
            "rolling_depth_gt1_real_commit_count": 0,
            "rolling_child_verified_without_parent_full_accept_count": 0,
            "rolling_child_committed_without_parent_full_accept_count": 0,
            "rolling_child_drafted_without_valid_parent_count": 0,
            "rolling_normal_lane_conflict_count": 0,
            "rolling_duplicate_child_count": 0,
            "rolling_frontier_mismatch_count": 0,
            "rolling_child_candidate_proposal_count": 0,
            "rolling_child_candidate_token_count": 0,
            "rolling_child_ready_shadow_proposal_count": 0,
            "rolling_child_ready_shadow_token_count": 0,
            "rolling_child_invalidated_count": 0,
            "rolling_max_depth_observed": 0,
            "rolling_drop_reason_counts": {},
            "target_sync_apply_checkpoint_ok_by_seq_id": {},
            "target_sync_apply_rollback_ok_by_seq_id": {},
            "target_sync_apply_mutation_remaining_by_seq_id": {},
            "target_sync_apply_len_before_by_seq_id": {},
            "target_sync_apply_len_after_simulated_by_seq_id": {},
            "target_sync_apply_len_after_restore_by_seq_id": {},
            "target_sync_apply_status_before_by_seq_id": {},
            "target_sync_apply_status_after_restore_by_seq_id": {},
            "draft_sync_apply_checkpoint_ok_by_seq_id": {},
            "draft_sync_apply_rollback_ok_by_seq_id": {},
            "draft_sync_apply_mutation_remaining_by_seq_id": {},
            "draft_sync_apply_len_before_by_seq_id": {},
            "draft_sync_apply_base_len_by_seq_id": {},
            "draft_sync_apply_len_after_simulated_by_seq_id": {},
            "draft_sync_apply_len_after_restore_by_seq_id": {},
            "draft_sync_apply_status_before_by_seq_id": {},
            "draft_sync_apply_status_after_restore_by_seq_id": {},
            "draft_sync_apply_expected_normal_draft_conflict_by_seq_id": {},
            "draft_sync_apply_original_draft_home_intersection_by_seq_id": {},
            "draft_sync_apply_adjusted_draft_home_exclusion_by_seq_id": {},
            "enable_eager_lane_exclusion_dry_run": bool(
                getattr(self.global_config, "enable_eager_lane_exclusion_dry_run", False)
            ),
            "eager_lane_exclusion_dry_run_enabled": bool(
                step_plan.eager_lane_exclusion_dry_run_enabled
            ),
            "original_draft_home_set": list(step_plan.original_draft_home_set or step_plan.draft_home_set),
            "actual_draft_home_set_for_normal_draft": list(
                step_plan.actual_draft_home_set_for_normal_draft or step_plan.draft_home_set
            ),
            "excluded_from_actual_draft_home_for_eager": list(step_plan.lane_excluded_seq_ids),
            "lane_excluded_seq_ids": list(step_plan.lane_excluded_seq_ids),
            "lane_exclusion_decision_available_before_draft": bool(
                step_plan.lane_exclusion_decision_available_before_draft
            ),
            "lane_exclusion_deferred_until_next_step": bool(
                step_plan.lane_exclusion_deferred_until_next_step
            ),
            "lane_exclusion_defer_reason": step_plan.lane_exclusion_defer_reason,
            "eager_lane_exclusion_proposal_ids": list(
                getattr(step_plan, "eager_lane_exclusion_proposal_ids", [])
            ),
            "eager_lane_exclusion_seq_ids": list(
                getattr(step_plan, "eager_lane_exclusion_seq_ids", step_plan.lane_excluded_seq_ids)
            ),
            "eager_lane_exclusion_reason_by_seq_id": dict(
                getattr(step_plan, "eager_lane_exclusion_reason_by_seq_id", {})
            ),
            "lane_exclusion_dry_run_done": bool(
                getattr(step_plan, "lane_exclusion_dry_run_done", False)
            ),
            "lane_exclusion_dry_run_done_proposal_ids": list(
                getattr(step_plan, "lane_exclusion_dry_run_done_proposal_ids", [])
            ),
            "lane_exclusion_dry_run_done_seq_ids": list(
                getattr(step_plan, "lane_exclusion_dry_run_done_seq_ids", step_plan.lane_excluded_seq_ids)
            ),
            "normal_proposal_expected_seq_ids_after_lane_exclusion": list(
                step_plan.actual_draft_home_set_for_normal_draft or step_plan.draft_home_set
            ),
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [],
            "normal_proposal_missing_excluded_seq_ids": [],
            "adjusted_normal_proposal_expected_seq_ids": list(
                step_plan.actual_draft_home_set_for_normal_draft or step_plan.draft_home_set
            ),
            "adjusted_normal_proposal_received_seq_ids": [],
            "missing_normal_proposal_after_lane_exclusion": [],
            "missing_normal_proposal_handled_by_fallback": [],
            "excluded_eager_seq_ids": list(step_plan.lane_excluded_seq_ids),
            "excluded_eager_seq_rejoin_required": bool(step_plan.lane_excluded_seq_ids),
            "lane_exclusion_expected_normal_draft_conflict_count": len(step_plan.lane_excluded_seq_ids),
            "lane_exclusion_resolved_normal_draft_conflict_count": len(step_plan.lane_excluded_seq_ids),
            "lane_exclusion_unexpected_missing_proposal_count": 0,
            "lane_exclusion_decision_late_count": 0,
            "ready_eager_proposal_created_ids": list(step_plan.ready_eager_proposal_created_ids),
            "ready_eager_proposal_created_seq_ids": list(step_plan.ready_eager_proposal_created_seq_ids),
            "ready_eager_proposal_synced_ids": list(step_plan.ready_eager_proposal_synced_ids),
            "ready_eager_proposal_registry_ids_before_plan": list(
                step_plan.ready_eager_proposal_registry_ids_before_plan
            ),
            "ready_eager_proposal_seen_by_scheduler_ids": list(
                step_plan.ready_eager_proposal_seen_by_scheduler_ids
            ),
            "ready_eager_proposal_in_target_home_ids": list(
                step_plan.ready_eager_proposal_in_target_home_ids
            ),
            "ready_eager_proposal_in_draft_home_ids": list(
                step_plan.ready_eager_proposal_in_draft_home_ids
            ),
            "ready_eager_proposal_applied_ids": list(step_plan.ready_eager_proposal_applied_ids),
            "ready_eager_proposal_stale_ids": list(step_plan.ready_eager_proposal_stale_ids),
            "ready_eager_proposal_expired_ids": list(step_plan.ready_eager_proposal_expired_ids),
            "ready_eager_proposal_invalidated_ids": list(
                step_plan.ready_eager_proposal_invalidated_ids
            ),
            "ready_eager_proposal_state_by_id": dict(step_plan.ready_eager_proposal_state_by_id),
            "ready_eager_proposal_skip_reason_by_id": dict(
                step_plan.ready_eager_proposal_skip_reason_by_id
            ),
            "ready_eager_proposal_stale_reason_by_id": dict(
                step_plan.ready_eager_proposal_stale_reason_by_id
            ),
            "ready_eager_proposal_age_by_id": dict(step_plan.ready_eager_proposal_age_by_id),
            "ready_eager_proposal_seq_id_by_id": dict(step_plan.ready_eager_proposal_seq_id_by_id),
            "ready_eager_proposal_base_len_by_id": dict(step_plan.ready_eager_proposal_base_len_by_id),
            "ready_eager_proposal_current_len_by_id": dict(
                step_plan.ready_eager_proposal_current_len_by_id
            ),
            "ready_eager_proposal_current_pre_verify_by_id": dict(
                step_plan.ready_eager_proposal_current_pre_verify_by_id
            ),
            "ready_eager_proposal_current_status_by_id": dict(
                step_plan.ready_eager_proposal_current_status_by_id
            ),
            "ready_eager_proposal_apply_step_by_id": dict(
                step_plan.ready_eager_proposal_apply_step_by_id
            ),
            "ready_eager_proposal_takeover_routed_step_by_id": dict(
                step_plan.ready_eager_proposal_takeover_routed_step_by_id
            ),
            "ready_eager_proposal_takeover_routed_ids": list(
                step_plan.ready_eager_proposal_takeover_routed_ids
            ),
            "ready_eager_proposal_takeover_routed_seq_ids": list(
                step_plan.ready_eager_proposal_takeover_routed_seq_ids
            ),
            "ready_eager_proposal_pending_takeover_ids": list(
                step_plan.ready_eager_proposal_pending_takeover_ids
            ),
            "ready_eager_proposal_pending_takeover_proposal_ids": list(
                step_plan.ready_eager_proposal_pending_takeover_proposal_ids
            ),
            "ready_eager_proposal_pending_takeover_seq_ids": list(
                step_plan.ready_eager_proposal_pending_takeover_seq_ids
            ),
            "ready_eager_proposal_takeover_waiting_for_target_home_ids": list(
                step_plan.ready_eager_proposal_takeover_waiting_for_target_home_ids
            ),
            "ready_eager_proposal_already_takeover_routed_ids": list(
                step_plan.ready_eager_proposal_already_takeover_routed_ids
            ),
            "repeated_takeover_proposal_ids": list(step_plan.repeated_takeover_proposal_ids),
            "ready_eager_proposals_synchronized_before_plan": bool(
                step_plan.ready_eager_proposals_synchronized_before_plan
            ),
            "ready_eager_proposal_transfer_called": bool(step_plan.ready_eager_proposal_transfer_called),
            "ready_eager_proposal_sent_ids": list(step_plan.ready_eager_proposal_sent_ids),
            "ready_eager_proposal_received_ids": list(step_plan.ready_eager_proposal_received_ids),
            "ready_eager_proposal_num_proposals": int(step_plan.ready_eager_proposal_num_proposals),
            "ready_eager_proposal_payload_len": int(step_plan.ready_eager_proposal_payload_len),
            "ready_eager_proposal_zero_proposal": bool(step_plan.ready_eager_proposal_zero_proposal),
            "lane_exclusion_applied_proposal_ids": list(step_plan.lane_exclusion_applied_proposal_ids),
            "lane_exclusion_applied_seq_ids": list(step_plan.lane_exclusion_applied_seq_ids),
            "lane_exclusion_apply_reason_by_proposal_id": dict(
                step_plan.lane_exclusion_apply_reason_by_proposal_id
            ),
            "raw_target_home_set_for_normal_verify": list(
                step_plan.raw_target_home_set_for_normal_verify or step_plan.target_home_set
            ),
            "target_normal_verify_seq_ids": list(
                step_plan.target_normal_verify_ids()
                if hasattr(step_plan, "target_normal_verify_ids")
                else (step_plan.target_normal_verify_seq_ids or step_plan.target_home_set)
            ),
            "target_eager_verify_seq_ids_dry_run": list(step_plan.target_eager_verify_seq_ids_dry_run),
            "target_eager_verify_proposal_ids_dry_run": list(
                step_plan.target_eager_verify_proposal_ids_dry_run
            ),
            "target_eager_verify_reason_by_seq_id_dry_run": dict(
                step_plan.target_eager_verify_reason_by_seq_id_dry_run
            ),
            "excluded_from_target_normal_verify_for_eager_dry_run": list(
                step_plan.excluded_from_target_normal_verify_for_eager_dry_run
            ),
            "missing_normal_proposal_allowed_by_eager_dry_run": bool(
                step_plan.missing_normal_proposal_allowed_by_eager_dry_run
            ),
            "missing_normal_proposal_allowed_seq_ids_dry_run": list(
                step_plan.missing_normal_proposal_allowed_seq_ids_dry_run
            ),
            "missing_buffered_proposal_seq_ids": list(step_plan.missing_buffered_proposal_seq_ids),
            "missing_buffered_proposal_allowed_by_eager_seq_ids": list(
                step_plan.missing_buffered_proposal_allowed_by_eager_seq_ids
            ),
            "missing_buffered_proposal_unexpected_seq_ids": list(
                step_plan.missing_buffered_proposal_unexpected_seq_ids
            ),
            "fallback_same_batch": bool(step_plan.fallback_same_batch),
            "fallback_pending_receive_seq_ids": list(step_plan.fallback_pending_receive_seq_ids),
            "fallback_received_seq_ids": list(step_plan.fallback_received_seq_ids),
            "fallback_missing_after_receive_seq_ids": list(
                step_plan.fallback_missing_after_receive_seq_ids
            ),
            "eager_schedule_step_id": None,
            "eager_schedule_plan_id": None,
            "target_eager_set_dry_run": list(step_plan.target_eager_set_dry_run),
            "scheduled_target_eager_set_dry_run": list(step_plan.scheduled_target_eager_set_dry_run),
            "scheduled_target_eager_proposal_ids_dry_run": list(
                step_plan.scheduled_target_eager_proposal_ids_dry_run
            ),
            "scheduled_target_eager_seq_ids_dry_run": list(
                step_plan.scheduled_target_eager_seq_ids_dry_run
            ),
            "adjusted_draft_home_set_dry_run": list(step_plan.adjusted_draft_home_set_dry_run),
            "excluded_from_draft_home_for_eager_dry_run": list(
                step_plan.excluded_from_draft_home_for_eager_dry_run
            ),
            "eager_schedule_candidate_proposal_ids": [],
            "eager_schedule_candidate_seq_ids": [],
            "eager_scheduled_proposal_ids": [],
            "eager_scheduled_seq_ids": [],
            "eager_schedule_deferred_proposal_ids": [],
            "eager_schedule_deferred_seq_ids": [],
            "eager_schedule_defer_reason_by_proposal_id": {},
            "eager_schedule_skipped_proposal_ids": [],
            "eager_schedule_skip_reason_by_proposal_id": {},
            "eager_schedule_clear_reason_by_proposal_id": {},
            "eager_schedule_proposal_id_by_seq_id": {},
            "eager_schedule_base_len_by_seq_id": {},
            "eager_schedule_current_len_by_seq_id": {},
            "eager_schedule_base_match_by_seq_id": {},
            "eager_schedule_seq_pre_verify_by_seq_id": {},
            "eager_schedule_base_pre_verify_by_proposal_id": {},
            "eager_schedule_proposal_len_by_proposal_id": {},
            "eager_schedule_to_verify_len_by_proposal_id": {},
            "eager_schedule_intersects_target_home": False,
            "eager_schedule_intersects_original_draft_home": False,
            "eager_schedule_intersects_adjusted_draft_home": False,
            "eager_schedule_intersects_draft_home": False,
            "eager_schedule_ready_buffer_size_before": self.eager_proposal_buffer.size(),
            "eager_schedule_ready_buffer_size_after": self.eager_proposal_buffer.size(),
            "eager_schedule_ready_buffer_size_after_clear": self.eager_proposal_buffer.size(),
            "eager_ready_buffer_size_before_schedule": self.eager_proposal_buffer.size(),
            "eager_ready_buffer_size_after_schedule": self.eager_proposal_buffer.size(),
            "eager_ready_buffer_size_after_clear": self.eager_proposal_buffer.size(),
            "eager_transfer_step_id": None,
            "eager_transfer_plan_id": None,
            "eager_transfer_num_proposals": 0,
            "eager_transfer_payload_len": 0,
            "eager_transfer_sent_proposal_ids": [],
            "eager_transfer_sent_seq_ids": [],
            "eager_transfer_received_proposal_ids": [],
            "eager_transfer_received_seq_ids": [],
            "eager_transfer_validated_proposal_ids": [],
            "eager_transfer_pending_proposal_ids": [],
            "eager_transfer_dropped_proposal_ids": [],
            "eager_transfer_drop_reason_by_proposal_id": {},
            "eager_transfer_base_len_by_seq_id": {},
            "eager_transfer_base_pre_verify_by_seq_id": {},
            "eager_transfer_current_len_by_seq_id": {},
            "eager_transfer_base_delta_by_seq_id": {},
            "eager_transfer_base_match_by_seq_id": {},
            "eager_transfer_seq_raw_status_by_seq_id": {},
            "eager_transfer_seq_is_finished_raw_by_seq_id": {},
            "eager_transfer_seq_request_finished_by_seq_id": {},
            "eager_transfer_seq_span_invalidated_by_seq_id": {},
            "eager_transfer_seq_in_scheduled_by_seq_id": {},
            "eager_transfer_seq_in_resolved_by_seq_id": {},
            "eager_transfer_seq_in_target_home_by_seq_id": {},
            "eager_transfer_seq_in_draft_home_by_seq_id": {},
            "eager_transfer_classification_by_proposal_id": {},
            "eager_transfer_proposal_len_by_proposal_id": {},
            "eager_transfer_to_verify_len_by_proposal_id": {},
            "eager_pending_received_proposal_ids": [],
            "eager_pending_received_seq_ids": [],
            "eager_pending_base_not_reached_proposal_ids": [],
            "eager_pending_base_not_reached_seq_ids": [],
            "eager_pending_ready_proposal_ids": [],
            "eager_pending_ready_seq_ids": [],
            "eager_pending_dropped_proposal_ids": [],
            "eager_pending_drop_reason_by_proposal_id": {},
            "eager_pending_buffer_size_before_update": self.eager_proposal_buffer.size(),
            "eager_pending_buffer_size_after_receive": self.eager_proposal_buffer.size(),
            "eager_pending_buffer_size_after_update": self.eager_proposal_buffer.size(),
            "eager_pending_buffer_size_after_clear": self.eager_proposal_buffer.size(),
            "eager_pending_current_len_by_seq_id": {},
            "eager_pending_base_len_by_seq_id": {},
            "eager_pending_base_delta_by_seq_id": {},
            "eager_pending_state_by_proposal_id": {},
            "draft_transfer_buffer_size_before_send": 0,
            "draft_transfer_buffer_size_after_send": 0,
            "draft_eager_buffer_size_before_transfer": 0,
            "draft_eager_buffer_size_after_transfer": 0,
            "target_ready_buffer_size_after_receive": 0,
            "target_ready_buffer_size_after_schedule": 0,
            "target_eager_buffer_size_before_receive": self.eager_proposal_buffer.size(),
            "target_eager_buffer_size_after_receive": self.eager_proposal_buffer.size(),
            "target_eager_buffer_size_after_clear": self.eager_proposal_buffer.size(),
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
            "eager_tokens_transferred": 0,
            "eager_tokens_transfer_pending": 0,
            "eager_tokens_transfer_validated": 0,
            "eager_tokens_transfer_dropped": 0,
            "eager_tokens_schedule_candidates": 0,
            "eager_tokens_scheduled_dry_run": 0,
            "eager_tokens_deferred_dry_run": 0,
            "eager_tokens_verify_dry_run": 0,
            "eager_tokens_verify_dry_run_full_accept": 0,
            "eager_tokens_verify_dry_run_rejected": 0,
            "eager_tokens_verify_dry_run_partial_accept": 0,
            "eager_tokens_apply_dry_run": 0,
            "eager_tokens_apply_dry_run_full_accept": 0,
            "eager_tokens_apply_dry_run_discarded": 0,
            "eager_apply_dry_run_append_tokens": 0,
            "eager_apply_dry_run_rollback_failure_count": 0,
            "eager_tokens_result_transfer_sent": 0,
            "eager_tokens_result_transfer_received": 0,
            "eager_tokens_result_transfer_validated": 0,
            "eager_tokens_result_transfer_invalid": 0,
            "eager_tokens_result_transfer_dry_run": 0,
            "eager_tokens_result_transfer_full_accept": 0,
            "eager_tokens_result_transfer_discarded": 0,
            "eager_tokens_sync_apply_dry_run": 0,
            "eager_tokens_sync_apply_dry_run_full_accept": 0,
            "eager_tokens_sync_apply_dry_run_discarded": 0,
            "eager_sync_apply_dry_run_append_tokens": 0,
            "eager_tokens_sync_apply_dry_run_target_side": 0,
            "eager_tokens_sync_apply_dry_run_draft_side": 0,
            "eager_sync_apply_target_mutation_remaining_count": 0,
            "eager_sync_apply_draft_mutation_remaining_count": 0,
            "eager_tokens_committed": 0,
            "eager_tokens_committed_full_accept": 0,
            "eager_commit_candidate_count": 0,
            "eager_commit_committed_count": 0,
            "eager_commit_skipped_count": 0,
            "eager_commit_skip_reason_counts": {},
            "eager_tokens_lane_excluded_dry_run": len(step_plan.lane_excluded_seq_ids) * int(self.gamma),
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
        self._prune_trace_record_for_level(record)

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

    def _eager_transfer_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_transfer_dry_run", False))

    def _eager_schedule_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_schedule_dry_run", False))

    def _eager_verify_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_verify_dry_run", False))

    def _eager_apply_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_apply_dry_run", False))

    def _eager_result_transfer_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_result_transfer_dry_run", False))

    def _eager_sync_apply_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_sync_apply_dry_run", False))

    def _eager_commit_readiness_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_commit_readiness_dry_run", False))

    def _eager_commit_ready_only_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_commit_ready_only", False))

    def _eager_lane_exclusion_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_eager_lane_exclusion_dry_run", False))

    def _continuous_eager_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_continuous_eager_dry_run", False))

    def _continuous_eager_verify_apply_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_continuous_eager_verify_apply_dry_run", False))

    def _continuous_eager_commit_depth1_ready_only_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_continuous_eager_commit_depth1_ready_only", False))

    def _rolling_continuous_eager_dry_run_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_rolling_continuous_eager_dry_run", False))

    def _eager_trace_level(self) -> str:
        level = str(getattr(self.global_config, "eager_trace_level", "full") or "full")
        return level if level in {"full", "summary", "minimal"} else "full"

    def _record_elapsed_ms(self, trace_record: dict | None, field_name: str, start_time: float) -> None:
        if trace_record is None:
            return
        elapsed_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
        trace_record[field_name] = float(trace_record.get(field_name) or 0.0) + elapsed_ms

    def _trace_value_is_empty_debug_default(self, value) -> bool:
        return value is None or value == [] or value == {}

    def _prune_trace_record_for_level(self, record: dict) -> None:
        level = self._eager_trace_level()
        record["eager_trace_level"] = level
        if level == "full":
            record["eager_trace_pruned_key_count"] = 0
            return

        eager_prefixes = (
            "eager_",
            "ready_eager_",
            "lane_exclusion_",
            "target_eager_",
            "scheduled_target_eager_",
            "adjusted_draft_home_set_dry_run",
            "excluded_from_draft_home_for_eager_dry_run",
            "missing_buffered_proposal_",
            "missing_normal_proposal_",
            "fallback_",
            "continuous_eager_",
        )
        pruned = 0
        for key in list(record.keys()):
            if key in {"eager_trace_level", "eager_trace_pruned_key_count"}:
                continue
            if key.startswith(eager_prefixes) and self._trace_value_is_empty_debug_default(record.get(key)):
                del record[key]
                pruned += 1

        if level == "minimal":
            # These are high-cardinality debug diagnostics that are useful when
            # chasing stale spans, but they are not needed by the 6a/6c commit
            # and accounting checkers. Keep proposal-id token/action maps.
            minimal_drop_prefixes = (
                "eager_transfer_base_",
                "eager_transfer_current_",
                "eager_transfer_seq_",
                "eager_pending_current_",
                "eager_pending_base_",
                "eager_verify_base_len_by_seq_id",
                "eager_verify_current_len_by_seq_id",
                "eager_verify_seq_",
                "eager_apply_current_len_",
                "eager_apply_pre_verify_",
                "eager_apply_status_",
                "eager_result_draft_",
                "ready_eager_proposal_current_",
            )
            for key in list(record.keys()):
                if key.startswith(minimal_drop_prefixes):
                    del record[key]
                    pruned += 1
        record["eager_trace_pruned_key_count"] = pruned

    def _pending_eager_seq_ids(self) -> set[int]:
        return {
            int(seq_id)
            for seq_id in (
                self.eager_proposal_buffer.pending_seq_ids()
                + self.eager_proposal_buffer.ready_seq_ids()
            )
        }

    def _validate_phase1h_plan(self, plan: StepPlan) -> None:
        plan.validate_phase1h_eager_scaffold(
            enable_eager_execution=False,
            enable_eager_plan_dry_run=bool(plan.enable_eager_plan_dry_run),
            enable_eager_draft_dry_run=bool(plan.enable_eager_draft_dry_run),
            enable_eager_promotion_dry_run=bool(plan.enable_eager_promotion_dry_run),
            enable_eager_transfer_dry_run=bool(plan.enable_eager_transfer_dry_run),
            enable_eager_schedule_dry_run=bool(plan.enable_eager_schedule_dry_run),
            enable_eager_verify_dry_run=bool(plan.enable_eager_verify_dry_run),
            enable_eager_apply_dry_run=bool(plan.enable_eager_apply_dry_run),
            enable_eager_result_transfer_dry_run=bool(plan.enable_eager_result_transfer_dry_run),
            enable_eager_sync_apply_dry_run=bool(plan.enable_eager_sync_apply_dry_run),
            enable_eager_commit_readiness_dry_run=bool(plan.enable_eager_commit_readiness_dry_run),
            enable_eager_commit_ready_only=bool(plan.enable_eager_commit_ready_only),
            enable_eager_lane_exclusion_dry_run=bool(plan.enable_eager_lane_exclusion_dry_run),
            global_gamma=int(self.gamma),
        )

    def _actual_normal_draft_seq_ids(self, plan: StepPlan) -> list[int]:
        return [
            int(seq_id)
            for seq_id in (plan.actual_draft_home_set_for_normal_draft or plan.draft_home_set)
        ]

    def _target_normal_verify_seq_ids(self, plan: StepPlan) -> list[int]:
        if hasattr(plan, "target_normal_verify_ids"):
            return [int(seq_id) for seq_id in plan.target_normal_verify_ids()]
        return [int(seq_id) for seq_id in (plan.target_normal_verify_seq_ids or plan.target_home_set)]

    def _lane_exclusion_step_id_before_plan(self) -> int:
        return int(self.dual_batch_manager.step_id)

    def _empty_ready_eager_proposal_sync_info(self) -> dict:
        return {
            "called": False,
            "sent_proposal_ids": [],
            "received_proposal_ids": [],
            "sent_seq_ids": [],
            "received_seq_ids": [],
            "num_proposals": 0,
            "payload_len": 0,
            "zero_proposal": True,
            "plan_id": None,
            "step_id": None,
        }

    def _sync_ready_eager_proposals_before_step_plan(self, plan_id: int) -> dict:
        info = self._empty_ready_eager_proposal_sync_info()
        if not self._eager_lane_exclusion_dry_run_enabled():
            return info

        step_id = self._lane_exclusion_step_id_before_plan()
        info.update({"called": True, "plan_id": int(plan_id), "step_id": int(step_id)})
        meta_len = 5

        if self.is_draft:
            meta_values = [0, 0, int(plan_id), int(step_id), READY_EAGER_TRANSFER_HEADER_LEN]
            payload_values: list[int] = []
            if self.tp_params.local_rank == 0:
                meta = torch.zeros(meta_len, dtype=torch.int64, device="cuda")
                dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
                meta_values = [int(value) for value in meta.tolist()]
                payload_len = int(meta_values[1])
                if payload_len > 0:
                    payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
                    dist.broadcast(
                        payload,
                        src=self.global_config.target_config.master_rank,
                        group=self.verify_group,
                    )
                    payload_values = [int(value) for value in payload.tolist()]

            draft_meta = (
                torch.tensor(meta_values, dtype=torch.int64, device="cuda")
                if self.tp_params.local_rank == 0
                else torch.zeros(meta_len, dtype=torch.int64, device="cuda")
            )
            dist.broadcast(draft_meta, src=self.global_config.draft_config.master_rank, group=self.group)
            meta_values = [int(value) for value in draft_meta.tolist()]
            payload_len = int(meta_values[1])
            if payload_len > 0:
                draft_payload = (
                    torch.tensor(payload_values, dtype=torch.int64, device="cuda")
                    if self.tp_params.local_rank == 0
                    else torch.zeros(payload_len, dtype=torch.int64, device="cuda")
                )
                dist.broadcast(
                    draft_payload,
                    src=self.global_config.draft_config.master_rank,
                    group=self.group,
                )
                payload_values = [int(value) for value in draft_payload.tolist()]
            proposals = deserialize_ready_eager_proposals(meta_values, payload_values)
            self.dual_batch_manager.receive_ready_eager_proposals(proposals)
            info["received_proposal_ids"] = [int(proposal.proposal_id) for proposal in proposals]
            info["received_seq_ids"] = [int(proposal.seq_id) for proposal in proposals]
            info["num_proposals"] = int(meta_values[0])
            info["payload_len"] = int(meta_values[1])
            info["zero_proposal"] = len(proposals) == 0
            info["plan_id"] = None if int(meta_values[2]) < 0 else int(meta_values[2])
            info["step_id"] = None if int(meta_values[3]) < 0 else int(meta_values[3])
            return info

        proposals = self.dual_batch_manager.pop_ready_eager_proposals_for_sync()
        meta_values, payload_values = serialize_ready_eager_proposals(
            proposals,
            plan_id=int(plan_id),
            step_id=step_id,
        )
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
        broadcast_meta_values = [int(value) for value in meta.tolist()]
        payload_len = int(broadcast_meta_values[1])
        if payload_len > 0:
            payload = (
                torch.tensor(payload_values, dtype=torch.int64, device="cuda")
                if self.rank == self.global_config.target_config.master_rank
                else torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            )
            dist.broadcast(payload, src=self.global_config.target_config.master_rank, group=self.verify_group)

        info["sent_proposal_ids"] = [int(proposal.proposal_id) for proposal in proposals]
        info["sent_seq_ids"] = [int(proposal.seq_id) for proposal in proposals]
        info["num_proposals"] = int(meta_values[0])
        info["payload_len"] = int(meta_values[1])
        info["zero_proposal"] = len(proposals) == 0
        return info

    def _attach_ready_eager_proposal_sync_info(self, plan: StepPlan, sync_info: dict) -> None:
        called = bool(sync_info.get("called", False))
        sent_ids = [
            int(proposal_id) for proposal_id in sync_info.get("sent_proposal_ids", [])
        ]
        received_ids = [
            int(proposal_id) for proposal_id in sync_info.get("received_proposal_ids", [])
        ]
        sent_seq_ids = [
            int(seq_id) for seq_id in sync_info.get("sent_seq_ids", [])
        ]
        received_seq_ids = [
            int(seq_id) for seq_id in sync_info.get("received_seq_ids", [])
        ]
        plan.ready_eager_proposals_synchronized_before_plan = called
        plan.ready_eager_proposal_transfer_called = called
        plan.ready_eager_proposal_sent_ids = list(sent_ids)
        plan.ready_eager_proposal_received_ids = list(received_ids)
        plan.ready_eager_proposal_sent_seq_ids = list(sent_seq_ids)
        plan.ready_eager_proposal_received_seq_ids = list(received_seq_ids)
        plan.ready_eager_proposal_synced_ids = sorted(set(sent_ids) | set(received_ids))
        plan.ready_eager_proposal_num_proposals = int(sync_info.get("num_proposals", 0))
        plan.ready_eager_proposal_payload_len = int(sync_info.get("payload_len", 0))
        plan.ready_eager_proposal_zero_proposal = bool(sync_info.get("zero_proposal", True))
        plan.ready_eager_proposal_sync_plan_id = sync_info.get("plan_id")
        plan.ready_eager_proposal_sync_step_id = sync_info.get("step_id")

        # Deprecated 1H-5e2 decision sync fields stay empty in 1H-5e3; the
        # synchronized cross-step object is ReadyEagerProposal.
        plan.lane_exclusion_decisions_synchronized_before_plan = called
        plan.lane_exclusion_decision_transfer_called = called
        plan.lane_exclusion_decision_sent_proposal_ids = []
        plan.lane_exclusion_decision_received_proposal_ids = []
        plan.lane_exclusion_decision_sent_seq_ids = []
        plan.lane_exclusion_decision_received_seq_ids = []
        plan.lane_exclusion_decision_num_decisions = 0
        plan.lane_exclusion_decision_payload_len = int(sync_info.get("payload_len", 0))
        plan.lane_exclusion_decision_zero_decision = True
        plan.lane_exclusion_decision_sync_plan_id = sync_info.get("plan_id")
        plan.lane_exclusion_decision_sync_step_id = sync_info.get("step_id")

    def _apply_eager_plan_dry_run(self, plan: StepPlan) -> None:
        commit_ready_only_enabled = self._eager_commit_ready_only_enabled()
        commit_readiness_dry_run_enabled = (
            self._eager_commit_readiness_dry_run_enabled() or commit_ready_only_enabled
        )
        sync_apply_dry_run_enabled = self._eager_sync_apply_dry_run_enabled() or commit_readiness_dry_run_enabled
        lane_exclusion_dry_run_enabled = self._eager_lane_exclusion_dry_run_enabled()
        result_transfer_dry_run_enabled = self._eager_result_transfer_dry_run_enabled() or sync_apply_dry_run_enabled
        apply_dry_run_enabled = self._eager_apply_dry_run_enabled() or result_transfer_dry_run_enabled
        verify_dry_run_enabled = self._eager_verify_dry_run_enabled() or apply_dry_run_enabled
        schedule_dry_run_enabled = self._eager_schedule_dry_run_enabled() or verify_dry_run_enabled or lane_exclusion_dry_run_enabled
        transfer_dry_run_enabled = self._eager_transfer_dry_run_enabled() or schedule_dry_run_enabled
        promotion_dry_run_enabled = self._eager_promotion_dry_run_enabled() or transfer_dry_run_enabled
        draft_dry_run_enabled = self._eager_draft_dry_run_enabled() or promotion_dry_run_enabled
        dry_run_enabled = self._eager_plan_dry_run_enabled() or draft_dry_run_enabled
        policy = str(getattr(self.global_config, "eager_policy", "none"))
        gamma = int(self.gamma)
        plan.enable_eager_plan_dry_run = dry_run_enabled
        plan.enable_eager_draft_dry_run = draft_dry_run_enabled
        plan.enable_eager_promotion_dry_run = promotion_dry_run_enabled
        plan.enable_eager_transfer_dry_run = transfer_dry_run_enabled
        plan.enable_eager_schedule_dry_run = schedule_dry_run_enabled
        plan.enable_eager_verify_dry_run = verify_dry_run_enabled
        plan.enable_eager_apply_dry_run = apply_dry_run_enabled
        plan.enable_eager_result_transfer_dry_run = result_transfer_dry_run_enabled
        plan.enable_eager_sync_apply_dry_run = sync_apply_dry_run_enabled
        plan.enable_eager_commit_readiness_dry_run = commit_readiness_dry_run_enabled
        plan.enable_eager_commit_ready_only = commit_ready_only_enabled
        plan.eager_commit_enabled = commit_ready_only_enabled
        plan.enable_eager_lane_exclusion_dry_run = lane_exclusion_dry_run_enabled
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
                enable_eager_transfer_dry_run=transfer_dry_run_enabled,
                enable_eager_schedule_dry_run=schedule_dry_run_enabled,
                enable_eager_verify_dry_run=verify_dry_run_enabled,
                enable_eager_apply_dry_run=apply_dry_run_enabled,
                enable_eager_result_transfer_dry_run=result_transfer_dry_run_enabled,
                enable_eager_sync_apply_dry_run=sync_apply_dry_run_enabled,
                enable_eager_commit_readiness_dry_run=commit_readiness_dry_run_enabled,
                enable_eager_commit_ready_only=commit_ready_only_enabled,
                enable_eager_lane_exclusion_dry_run=lane_exclusion_dry_run_enabled,
                global_gamma=gamma,
            )
            return

        target_seq_ids = [int(seq_id) for seq_id in plan.target_home_set]
        target_home_set = set(target_seq_ids)
        target_seqs = self.scheduler.find_by_seq_ids(target_seq_ids) if target_seq_ids else []
        seq_by_id = {int(seq.seq_id): seq for seq in target_seqs}
        running_seq_ids = {int(seq.seq_id) for seq in self.scheduler.running}
        pending_eager_seq_ids = self._pending_eager_seq_ids()
        target_eager_takeover_seq_ids = {
            int(seq_id) for seq_id in getattr(plan, "target_eager_verify_seq_ids_dry_run", [])
        }

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
            elif seq_id in target_eager_takeover_seq_ids:
                reason = "target_eager_verify_takeover"
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
            enable_eager_transfer_dry_run=transfer_dry_run_enabled,
            enable_eager_schedule_dry_run=schedule_dry_run_enabled,
            enable_eager_verify_dry_run=verify_dry_run_enabled,
            enable_eager_apply_dry_run=apply_dry_run_enabled,
            enable_eager_result_transfer_dry_run=result_transfer_dry_run_enabled,
            enable_eager_sync_apply_dry_run=sync_apply_dry_run_enabled,
            enable_eager_commit_readiness_dry_run=commit_readiness_dry_run_enabled,
            enable_eager_commit_ready_only=commit_ready_only_enabled,
            enable_eager_lane_exclusion_dry_run=lane_exclusion_dry_run_enabled,
            global_gamma=gamma,
        )

    def _build_dual_batch_step_plan(self) -> StepPlan:
        proposal_buffer_size_before = self.dual_proposal_buffer.size()
        iteration_id, _ = self.scheduler.next_batch_id("dual_batch")
        self._trace_plan_id += 1
        plan_id = self._trace_plan_id
        lane_sync_info = self._sync_ready_eager_proposals_before_step_plan(plan_id)
        dropped_seq_ids = self._prepare_dual_batch_state()
        plan = self.dual_batch_manager.build_step_plan(
            plan_id=plan_id,
            iteration_id=iteration_id,
            execution_mode=self.active_execution_mode,
            decode_ready_mode=self.active_decode_ready_mode,
            pending_proposal_seq_ids=self.dual_proposal_buffer.pending_seq_ids(),
            pending_batch_ids=self.dual_proposal_buffer.pending_batch_ids(),
            enable_eager_execution=bool(getattr(self.global_config, "enable_eager_execution", False)),
            enable_eager_lane_exclusion_dry_run=self._eager_lane_exclusion_dry_run_enabled(),
            running_seqs=list(self.scheduler.running),
        )
        self._attach_ready_eager_proposal_sync_info(plan, lane_sync_info)
        self._apply_eager_plan_dry_run(plan)
        if bool(getattr(plan, "enable_eager_plan_dry_run", False)):
            self._validate_phase1h_plan(plan)
        raw_buffer_inspect = self.dual_proposal_buffer.inspect(plan.target_home_set)
        target_normal_verify_seq_ids = self._target_normal_verify_seq_ids(plan)
        actual_normal_draft_seq_ids = self._actual_normal_draft_seq_ids(plan)
        buffer_inspect = self.dual_proposal_buffer.inspect(target_normal_verify_seq_ids)
        allowed_missing = sorted(
            set(raw_buffer_inspect["miss_seq_ids"])
            & set(getattr(plan, "target_eager_verify_seq_ids_dry_run", []))
        )
        fallback_same_batch = (
            bool(target_normal_verify_seq_ids)
            and plan.plan_phase == "fallback"
            and target_normal_verify_seq_ids == actual_normal_draft_seq_ids
        )
        fallback_pending_receive = (
            sorted(set(raw_buffer_inspect["miss_seq_ids"]) & set(target_normal_verify_seq_ids))
            if fallback_same_batch
            else []
        )
        unexpected_missing = sorted(
            set(raw_buffer_inspect["miss_seq_ids"])
            - set(allowed_missing)
            - set(fallback_pending_receive)
        )
        plan.raw_target_home_set_for_normal_verify = [int(seq_id) for seq_id in plan.target_home_set]
        plan.missing_buffered_proposal_seq_ids = [int(seq_id) for seq_id in raw_buffer_inspect["miss_seq_ids"]]
        plan.missing_buffered_proposal_allowed_by_eager_seq_ids = [int(seq_id) for seq_id in allowed_missing]
        plan.missing_buffered_proposal_unexpected_seq_ids = [int(seq_id) for seq_id in unexpected_missing]
        plan.missing_normal_proposal_allowed_by_eager_dry_run = bool(allowed_missing)
        plan.missing_normal_proposal_allowed_seq_ids_dry_run = [int(seq_id) for seq_id in allowed_missing]
        plan.fallback_same_batch = bool(fallback_same_batch)
        plan.fallback_pending_receive_seq_ids = [int(seq_id) for seq_id in fallback_pending_receive]
        plan.fallback_received_seq_ids = []
        plan.fallback_missing_after_receive_seq_ids = []
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
            if (
                plan.proposal_buffer_miss_count
                and not fallback_same_batch
                and plan.fallback_reason == "single_active_batch_with_buffered_proposals"
            ):
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
            f"draft_home_set={plan.draft_home_set}, original_draft_home_set="
            f"{plan.original_draft_home_set}, actual_draft_home_set_for_normal_draft="
            f"{self._actual_normal_draft_seq_ids(plan)}, buffered_proposal_seq_ids="
            f"{self.dual_proposal_buffer.pending_seq_ids()}, target_normal_verify_seq_ids="
            f"{self._target_normal_verify_seq_ids(plan)}, target_eager_verify_seq_ids_dry_run="
            f"{getattr(plan, 'target_eager_verify_seq_ids_dry_run', [])}, "
            f"missing_buffered_proposal_allowed_by_eager_seq_ids="
            f"{getattr(plan, 'missing_buffered_proposal_allowed_by_eager_seq_ids', [])}, "
            f"missing_buffered_proposal_unexpected_seq_ids="
            f"{getattr(plan, 'missing_buffered_proposal_unexpected_seq_ids', [])}"
        )

    def _update_lane_exclusion_proposal_trace(
        self,
        trace_record: dict,
        plan: StepPlan,
        sent_proposals: list[BufferedProposal] | None = None,
        received_proposals: list[BufferedProposal] | None = None,
    ) -> None:
        expected_seq_ids = self._actual_normal_draft_seq_ids(plan)
        sent_seq_ids = (
            [int(proposal.seq_id) for proposal in sent_proposals]
            if sent_proposals is not None
            else list(trace_record.get("normal_proposal_sent_seq_ids_after_lane_exclusion", []))
        )
        received_seq_ids = (
            [int(proposal.seq_id) for proposal in received_proposals]
            if received_proposals is not None
            else list(trace_record.get("normal_proposal_received_seq_ids_after_lane_exclusion", []))
        )
        excluded_seq_ids = [int(seq_id) for seq_id in plan.lane_excluded_seq_ids]
        missing_sent = [seq_id for seq_id in expected_seq_ids if seq_id not in set(sent_seq_ids)]
        missing_received = [seq_id for seq_id in expected_seq_ids if seq_id not in set(received_seq_ids)]
        trace_record["original_draft_home_set"] = list(plan.original_draft_home_set or plan.draft_home_set)
        trace_record["actual_draft_home_set_for_normal_draft"] = list(expected_seq_ids)
        if (
            self._eager_lane_exclusion_dry_run_enabled()
            or plan.adjusted_draft_home_set_dry_run
            or plan.excluded_from_draft_home_for_eager_dry_run
        ):
            trace_record["adjusted_draft_home_set_dry_run"] = list(
                plan.adjusted_draft_home_set_dry_run or expected_seq_ids
            )
            trace_record["excluded_from_draft_home_for_eager_dry_run"] = list(
                plan.excluded_from_draft_home_for_eager_dry_run
            )
        trace_record["excluded_from_actual_draft_home_for_eager"] = list(excluded_seq_ids)
        trace_record["lane_excluded_seq_ids"] = list(excluded_seq_ids)
        trace_record["raw_target_home_set_for_normal_verify"] = list(
            plan.raw_target_home_set_for_normal_verify or plan.target_home_set
        )
        trace_record["target_normal_verify_seq_ids"] = self._target_normal_verify_seq_ids(plan)
        trace_record["target_eager_verify_seq_ids_dry_run"] = list(
            plan.target_eager_verify_seq_ids_dry_run
        )
        trace_record["target_eager_verify_proposal_ids_dry_run"] = list(
            plan.target_eager_verify_proposal_ids_dry_run
        )
        trace_record["target_eager_verify_reason_by_seq_id_dry_run"] = dict(
            plan.target_eager_verify_reason_by_seq_id_dry_run
        )
        trace_record["excluded_from_target_normal_verify_for_eager_dry_run"] = list(
            plan.excluded_from_target_normal_verify_for_eager_dry_run
        )
        trace_record["missing_normal_proposal_allowed_by_eager_dry_run"] = bool(
            plan.missing_normal_proposal_allowed_by_eager_dry_run
        )
        trace_record["missing_normal_proposal_allowed_seq_ids_dry_run"] = list(
            plan.missing_normal_proposal_allowed_seq_ids_dry_run
        )
        trace_record["missing_buffered_proposal_seq_ids"] = list(plan.missing_buffered_proposal_seq_ids)
        trace_record["missing_buffered_proposal_allowed_by_eager_seq_ids"] = list(
            plan.missing_buffered_proposal_allowed_by_eager_seq_ids
        )
        trace_record["missing_buffered_proposal_unexpected_seq_ids"] = list(
            plan.missing_buffered_proposal_unexpected_seq_ids
        )
        trace_record["fallback_same_batch"] = bool(plan.fallback_same_batch)
        trace_record["fallback_pending_receive_seq_ids"] = list(plan.fallback_pending_receive_seq_ids)
        trace_record["fallback_received_seq_ids"] = list(plan.fallback_received_seq_ids)
        trace_record["fallback_missing_after_receive_seq_ids"] = list(
            plan.fallback_missing_after_receive_seq_ids
        )
        trace_record["normal_proposal_expected_seq_ids_after_lane_exclusion"] = list(expected_seq_ids)
        trace_record["adjusted_normal_proposal_expected_seq_ids"] = list(expected_seq_ids)
        trace_record["normal_proposal_sent_seq_ids_after_lane_exclusion"] = list(sent_seq_ids)
        trace_record["normal_proposal_received_seq_ids_after_lane_exclusion"] = list(received_seq_ids)
        trace_record["adjusted_normal_proposal_received_seq_ids"] = list(received_seq_ids)
        trace_record["normal_proposal_missing_excluded_seq_ids"] = list(excluded_seq_ids)
        trace_record["missing_normal_proposal_after_lane_exclusion"] = (
            list(missing_sent) if sent_proposals is not None else list(missing_received)
        )
        trace_record["missing_normal_proposal_handled_by_fallback"] = list(excluded_seq_ids)
        trace_record["lane_exclusion_expected_normal_draft_conflict_count"] = len(excluded_seq_ids)
        trace_record["lane_exclusion_resolved_normal_draft_conflict_count"] = len(excluded_seq_ids)
        trace_record["lane_exclusion_unexpected_missing_proposal_count"] = len(
            list(missing_sent) if sent_proposals is not None else list(missing_received)
        )
        trace_record["eager_tokens_lane_excluded_dry_run"] = len(excluded_seq_ids) * int(self.gamma)

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

    def _trace_eager_transfer_send(
        self,
        trace_record: dict,
        plan: StepPlan,
        proposals: list[EagerProposal],
        meta_values: list[int],
        buffer_size_before: int,
    ) -> None:
        trace_record["enable_eager_transfer_dry_run"] = True
        trace_record["eager_transfer_dry_run_enabled"] = True
        trace_record["eager_transfer_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_transfer_plan_id"] = int(plan.plan_id)
        trace_record["eager_transfer_num_proposals"] = int(meta_values[0])
        trace_record["eager_transfer_payload_len"] = int(meta_values[1])
        trace_record["eager_transfer_sent_proposal_ids"] = [int(proposal.proposal_id) for proposal in proposals]
        trace_record["eager_transfer_sent_seq_ids"] = [int(proposal.seq_id) for proposal in proposals]
        trace_record["draft_transfer_buffer_size_before_send"] = int(buffer_size_before)
        trace_record["draft_transfer_buffer_size_after_send"] = self.eager_proposal_buffer.size()
        trace_record["draft_eager_buffer_size_before_transfer"] = int(buffer_size_before)
        trace_record["draft_eager_buffer_size_after_transfer"] = self.eager_proposal_buffer.size()
        trace_record["eager_tokens_transferred"] = sum(int(proposal.proposal_len) for proposal in proposals)
        trace_record["eager_buffer_size_after"] = self.eager_proposal_buffer.size()
        trace_record["eager_proposal_transfer_called"] = True

    def _send_eager_transfer_dry_run(
        self,
        proposals: list[EagerProposal],
        plan: StepPlan,
        trace_record: dict,
    ) -> None:
        timer_start = time.perf_counter()
        buffer_size_before = self.eager_proposal_buffer.size()
        ready_proposals = [
            proposal for proposal in proposals
            if proposal.valid and proposal.state == EAGER_STATE_READY_TO_VERIFY
        ]
        for proposal in ready_proposals:
            self._draft_sent_eager_proposals_by_id[int(proposal.proposal_id)] = proposal
        meta_values, payload_values = serialize_eager_transfer_payload(
            ready_proposals,
            gamma=int(self.gamma),
            plan_id=int(plan.plan_id),
            step_id=plan.step_id,
        )
        for proposal in ready_proposals:
            proposal.state = EAGER_STATE_TRANSFERRED_DRY_RUN
            proposal.valid = False
        self.eager_proposal_buffer.clear()
        self._trace_eager_transfer_send(
            trace_record,
            plan,
            ready_proposals,
            meta_values,
            buffer_size_before,
        )
        if self.tp_params.local_rank != 0:
            self._record_elapsed_ms(trace_record, "eager_transfer_time_ms", timer_start)
            return
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta_values[1]) > 0:
            payload = torch.tensor(payload_values, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        self._record_elapsed_ms(trace_record, "eager_transfer_time_ms", timer_start)

    def _result_action_code(self, action: str) -> int:
        return {
            "append_full_accept_then_rollback": 1,
            "discard_partial_no_mutation": 2,
            "discard_reject_no_mutation": 3,
            "skipped_invalid_no_mutation": 4,
        }.get(str(action), 0)

    def _result_action_from_code(self, action_code: int) -> str:
        return {
            1: "append_full_accept_then_rollback",
            2: "discard_partial_no_mutation",
            3: "discard_reject_no_mutation",
            4: "skipped_invalid_no_mutation",
        }.get(int(action_code), "unknown")

    def _result_verify_code(self, result: str) -> int:
        return {
            "full_accept": 1,
            "partial_accept": 2,
            "reject_at_first_token": 3,
            "skipped_invalid": 4,
        }.get(str(result), 0)

    def _result_verify_from_code(self, result_code: int) -> str:
        return {
            1: "full_accept",
            2: "partial_accept",
            3: "reject_at_first_token",
            4: "skipped_invalid",
        }.get(int(result_code), "unknown")

    def _verify_result_from_accept_len(self, accepted_len: int, proposal_len: int) -> str:
        accepted_len = int(accepted_len)
        proposal_len = int(proposal_len)
        if accepted_len >= proposal_len:
            return "full_accept"
        if accepted_len <= 0:
            return "reject_at_first_token"
        return "partial_accept"

    def _sync_apply_action(self, accepted_len: int, full_accept: bool) -> str:
        if bool(full_accept):
            return "append_full_accept_then_rollback"
        return "discard_partial_no_mutation" if int(accepted_len) > 0 else "discard_reject_no_mutation"

    def _numeric_request_id(self, request_id) -> int:
        try:
            return int(request_id)
        except Exception:
            return -1

    def _trace_map_get(self, mapping: dict, key: int, default=None):
        if not isinstance(mapping, dict):
            return default
        if key in mapping:
            return mapping[key]
        return mapping.get(str(key), default)

    def _build_takeover_eager_result_transfer_results(self, plan: StepPlan, trace_record: dict) -> list[dict]:
        proposal_ids = [
            int(proposal_id)
            for proposal_id in trace_record.get("eager_apply_dry_run_executed_proposal_ids", [])
        ]
        seq_id_by_proposal = trace_record.get("eager_apply_dry_run_seq_id_by_proposal_id", {})
        verify_result_by_proposal = trace_record.get("eager_apply_dry_run_verify_result_by_proposal_id", {})
        accept_len_by_proposal = trace_record.get("eager_apply_dry_run_accept_len_by_proposal_id", {})
        proposal_len_by_proposal = trace_record.get("eager_apply_dry_run_proposal_len_by_proposal_id", {})
        action_by_proposal = trace_record.get("eager_apply_dry_run_action_by_proposal_id", {})
        append_tokens_by_proposal = trace_record.get("eager_apply_dry_run_append_tokens_by_proposal_id", {})
        discarded_tokens_by_proposal = trace_record.get("eager_apply_dry_run_discarded_tokens_by_proposal_id", {})
        rollback_ok_by_proposal = trace_record.get("eager_apply_dry_run_rollback_ok_by_proposal_id", {})
        mutation_by_proposal = trace_record.get("eager_apply_dry_run_mutation_detected_by_proposal_id", {})
        checkpoint_failed_by_proposal = trace_record.get("eager_apply_dry_run_checkpoint_failed_by_proposal_id", {})
        len_before_by_seq = trace_record.get("eager_apply_dry_run_sequence_len_before_by_seq_id", {})
        len_after_by_seq = trace_record.get("eager_apply_dry_run_sequence_len_after_by_seq_id", {})
        apply_step_id = trace_record.get("eager_apply_dry_run_step_id")
        apply_plan_id = trace_record.get("eager_apply_dry_run_plan_id")
        verify_step_id = trace_record.get("eager_verify_dry_run_step_id")
        verify_plan_id = trace_record.get("eager_verify_dry_run_plan_id")

        results: list[dict] = []
        for proposal_id in proposal_ids:
            ready_proposal = self.dual_batch_manager.ready_eager_proposals.by_id(proposal_id)
            seq_id = int(self._trace_map_get(seq_id_by_proposal, proposal_id, -1))
            proposal_len = int(self._trace_map_get(proposal_len_by_proposal, proposal_id, int(self.gamma)))
            accept_len = int(self._trace_map_get(accept_len_by_proposal, proposal_id, -1))
            verify_result = str(self._trace_map_get(verify_result_by_proposal, proposal_id, "unknown"))
            apply_action = str(self._trace_map_get(action_by_proposal, proposal_id, "unknown"))
            append_token_count = int(self._trace_map_get(append_tokens_by_proposal, proposal_id, 0))
            discarded_token_count = int(self._trace_map_get(discarded_tokens_by_proposal, proposal_id, 0))
            rollback_ok = bool(self._trace_map_get(rollback_ok_by_proposal, proposal_id, False))
            mutation_detected = bool(self._trace_map_get(mutation_by_proposal, proposal_id, True))
            checkpoint_failed = bool(self._trace_map_get(checkpoint_failed_by_proposal, proposal_id, True))
            source_plan_id = int(ready_proposal.source_plan_id) if ready_proposal is not None else -1
            source_step_id = int(ready_proposal.source_step_id) if ready_proposal is not None else -1
            takeover_step_id = (
                -1
                if ready_proposal is None or ready_proposal.takeover_routed_step_id is None
                else int(ready_proposal.takeover_routed_step_id)
            )
            base_len = int(ready_proposal.base_len) if ready_proposal is not None else -1
            base_pre_verify = bool(ready_proposal.base_pre_verify) if ready_proposal is not None else True
            to_verify_len = int(ready_proposal.to_verify_len) if ready_proposal is not None else proposal_len
            request_id = self._numeric_request_id(ready_proposal.request_id) if ready_proposal is not None else -1
            sequence_len_before = int(self._trace_map_get(len_before_by_seq, seq_id, -1))
            sequence_len_after = int(self._trace_map_get(len_after_by_seq, seq_id, -1))
            full_accept = verify_result == "full_accept"
            reject_position = -1 if full_accept else max(0, accept_len)
            results.append(
                {
                    "proposal_id": proposal_id,
                    "seq_id": seq_id,
                    "request_id": request_id,
                    "source_plan_id": source_plan_id,
                    "source_step_id": source_step_id,
                    "schedule_plan_id": int(apply_plan_id if apply_plan_id is not None else plan.plan_id),
                    "schedule_step_id": int(apply_step_id if apply_step_id is not None else (-1 if plan.step_id is None else plan.step_id)),
                    "takeover_step_id": takeover_step_id,
                    "verify_plan_id": int(verify_plan_id if verify_plan_id is not None else plan.plan_id),
                    "verify_step_id": int(verify_step_id if verify_step_id is not None else (-1 if plan.step_id is None else plan.step_id)),
                    "apply_plan_id": int(apply_plan_id if apply_plan_id is not None else plan.plan_id),
                    "apply_step_id": int(apply_step_id if apply_step_id is not None else (-1 if plan.step_id is None else plan.step_id)),
                    "verify_result": verify_result,
                    "accepted_len": accept_len,
                    "full_accept": full_accept,
                    "reject_position": reject_position,
                    "invalidated_len": max(0, proposal_len - max(0, accept_len)),
                    "revised_token": -1,
                    "proposal_len": proposal_len,
                    "to_verify_len": to_verify_len,
                    "gamma": int(self.gamma),
                    "base_len": base_len,
                    "base_pre_verify": base_pre_verify,
                    "target_seq_len_at_verify": sequence_len_before,
                    "target_seq_pre_verify_at_verify": False,
                    "apply_action": apply_action,
                    "append_token_count": append_token_count,
                    "discarded_token_count": discarded_token_count,
                    "rollback_ok": rollback_ok,
                    "mutation_detected": mutation_detected,
                    "checkpoint_failed": checkpoint_failed,
                    "sequence_len_before": sequence_len_before,
                    "sequence_len_after": sequence_len_after,
                    "source": EAGER_TAKEOVER_DRY_RUN_SOURCE,
                }
            )
        return results

    def _build_eager_result_transfer_results(
        self,
        plan: StepPlan,
        trace_record: dict,
        scheduled_proposals: list[EagerProposal],
    ) -> list[dict]:
        if trace_record.get("eager_apply_dry_run_source") == EAGER_TAKEOVER_DRY_RUN_SOURCE:
            return self._build_takeover_eager_result_transfer_results(plan, trace_record)

        verify_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_verify_executed_proposal_ids", [])
        }
        accepted_by_seq = trace_record.get("eager_verify_accepted_len_by_seq_id", {})
        full_accept_by_seq = trace_record.get("eager_verify_full_accept_by_seq_id", {})
        reject_position_by_seq = trace_record.get("eager_verify_reject_position_by_seq_id", {})
        invalidated_by_seq = trace_record.get("eager_verify_invalidated_len_by_seq_id", {})
        revised_by_seq = trace_record.get("eager_verify_revised_token_by_seq_id", {})
        target_len_by_seq = trace_record.get("eager_verify_current_len_by_seq_id", {})
        target_pre_verify_by_seq = trace_record.get("eager_verify_seq_pre_verify_by_seq_id", {})
        apply_action_by_seq = trace_record.get("eager_apply_action_by_seq_id", {})
        results = []
        for proposal in scheduled_proposals:
            proposal_id = int(proposal.proposal_id)
            if proposal_id not in verify_ids:
                continue
            seq_id = int(proposal.seq_id)
            accepted_len = int(accepted_by_seq.get(str(seq_id), accepted_by_seq.get(seq_id, -1)))
            full_accept = bool(full_accept_by_seq.get(str(seq_id), full_accept_by_seq.get(seq_id, False)))
            reject_position = int(reject_position_by_seq.get(str(seq_id), reject_position_by_seq.get(seq_id, -1)))
            invalidated_len = int(invalidated_by_seq.get(str(seq_id), invalidated_by_seq.get(seq_id, -1)))
            revised_token = int(revised_by_seq.get(str(seq_id), revised_by_seq.get(seq_id, -1)))
            target_len = int(target_len_by_seq.get(str(seq_id), target_len_by_seq.get(seq_id, -1)))
            target_pre_verify = bool(
                target_pre_verify_by_seq.get(str(seq_id), target_pre_verify_by_seq.get(seq_id, True))
            )
            apply_action = str(apply_action_by_seq.get(str(seq_id), apply_action_by_seq.get(seq_id, "unknown")))
            verify_result = self._verify_result_from_accept_len(accepted_len, int(proposal.proposal_len))
            results.append(
                {
                    "proposal_id": proposal_id,
                    "seq_id": seq_id,
                    "request_id": self._numeric_request_id(proposal.request_id),
                    "source_plan_id": int(proposal.source_plan_id),
                    "source_step_id": int(proposal.source_step_id),
                    "schedule_plan_id": int(plan.plan_id),
                    "schedule_step_id": -1 if plan.step_id is None else int(plan.step_id),
                    "takeover_step_id": -1,
                    "verify_plan_id": int(trace_record.get("eager_verify_dry_run_plan_id") or plan.plan_id),
                    "verify_step_id": -1
                    if trace_record.get("eager_verify_dry_run_step_id") is None
                    else int(trace_record.get("eager_verify_dry_run_step_id")),
                    "apply_plan_id": int(trace_record.get("eager_apply_dry_run_plan_id") or plan.plan_id),
                    "apply_step_id": -1
                    if trace_record.get("eager_apply_dry_run_step_id") is None
                    else int(trace_record.get("eager_apply_dry_run_step_id")),
                    "verify_result": verify_result,
                    "accepted_len": accepted_len,
                    "full_accept": full_accept,
                    "reject_position": reject_position,
                    "invalidated_len": invalidated_len,
                    "revised_token": revised_token,
                    "proposal_len": int(proposal.proposal_len),
                    "to_verify_len": len(proposal.to_be_verified_token_ids),
                    "gamma": int(self.gamma),
                    "base_len": int(proposal.base_len),
                    "base_pre_verify": bool(proposal.base_pre_verify),
                    "target_seq_len_at_verify": target_len,
                    "target_seq_pre_verify_at_verify": target_pre_verify,
                    "apply_action": apply_action,
                    "append_token_count": int(proposal.proposal_len) if full_accept else 0,
                    "discarded_token_count": 0 if full_accept else int(proposal.proposal_len),
                    "rollback_ok": True,
                    "mutation_detected": False,
                    "checkpoint_failed": False,
                    "sequence_len_before": target_len,
                    "sequence_len_after": target_len,
                    "source": EAGER_LEGACY_RESULT_TRANSFER_SOURCE,
                }
            )
        return results

    def _serialize_eager_result_transfer_payload(
        self,
        results: list[dict],
        plan: StepPlan,
    ) -> tuple[list[int], list[int]]:
        payload_values = []
        for result in results:
            payload_values.extend(
                [
                    int(result["proposal_id"]),
                    int(result["seq_id"]),
                    int(result["request_id"]),
                    int(result["source_plan_id"]),
                    int(result["source_step_id"]),
                    int(result["schedule_plan_id"]),
                    int(result["schedule_step_id"]),
                    int(result.get("takeover_step_id", -1)),
                    int(result["verify_plan_id"]),
                    int(result["verify_step_id"]),
                    int(result.get("apply_plan_id", result.get("schedule_plan_id", -1))),
                    int(result.get("apply_step_id", result.get("schedule_step_id", -1))),
                    int(self._result_verify_code(result.get("verify_result", "unknown"))),
                    int(result["accepted_len"]),
                    int(bool(result["full_accept"])),
                    int(result["reject_position"]),
                    int(result["invalidated_len"]),
                    int(result["revised_token"]),
                    int(result["proposal_len"]),
                    int(result["to_verify_len"]),
                    int(result["gamma"]),
                    int(result["base_len"]),
                    int(bool(result["base_pre_verify"])),
                    int(result["target_seq_len_at_verify"]),
                    int(bool(result["target_seq_pre_verify_at_verify"])),
                    int(self._result_action_code(result["apply_action"])),
                    int(result.get("append_token_count", 0)),
                    int(result.get("discarded_token_count", 0)),
                    int(bool(result.get("rollback_ok", False))),
                    int(bool(result.get("mutation_detected", True))),
                    int(bool(result.get("checkpoint_failed", True))),
                ]
            )
        meta_values = [
            int(EAGER_RESULT_TRANSFER_MAGIC),
            int(EAGER_RESULT_TRANSFER_OP_DRY_RUN),
            len(results),
            len(payload_values),
            int(self.gamma),
            int(plan.plan_id),
            -1 if plan.step_id is None else int(plan.step_id),
        ]
        return meta_values, payload_values

    def _deserialize_eager_result_transfer_payload(self, meta_values: list[int], payload_values: list[int]) -> list[dict]:
        meta_values = [int(value) for value in meta_values]
        if len(meta_values) >= EAGER_RESULT_TRANSFER_META_LEN:
            magic, op_type, num_results, payload_len, _gamma, _plan_id, _step_id = meta_values[:EAGER_RESULT_TRANSFER_META_LEN]
            if int(magic) != int(EAGER_RESULT_TRANSFER_MAGIC):
                raise ValueError(f"eager result transfer magic mismatch: got={magic}")
            if int(op_type) != int(EAGER_RESULT_TRANSFER_OP_DRY_RUN):
                raise ValueError(f"eager result transfer op mismatch: got={op_type}")
            header_width = EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH
        else:
            num_results, payload_len, _gamma, _plan_id, _step_id = meta_values[:5]
            header_width = 22
        if int(payload_len) != len(payload_values):
            raise ValueError(
                f"eager result transfer payload length mismatch: meta={payload_len}, actual={len(payload_values)}"
            )
        if payload_len != num_results * header_width:
            raise ValueError(
                f"malformed eager result payload: num_results={num_results}, payload_len={payload_len}"
            )
        results = []
        for idx in range(num_results):
            base = idx * header_width
            (
                proposal_id,
                seq_id,
                request_id,
                source_plan_id,
                source_step_id,
                schedule_plan_id,
                schedule_step_id,
                *rest,
            ) = payload_values[base:base + header_width]
            if header_width == EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH:
                (
                    takeover_step_id,
                    verify_plan_id,
                    verify_step_id,
                    apply_plan_id,
                    apply_step_id,
                    verify_result_code,
                    accepted_len,
                    full_accept,
                    reject_position,
                    invalidated_len,
                    revised_token,
                    proposal_len,
                    to_verify_len,
                    gamma,
                    base_len,
                    base_pre_verify,
                    target_seq_len_at_verify,
                    target_seq_pre_verify_at_verify,
                    apply_action_code,
                    append_token_count,
                    discarded_token_count,
                    rollback_ok,
                    mutation_detected,
                    checkpoint_failed,
                ) = rest
                verify_result = self._result_verify_from_code(int(verify_result_code))
            else:
                (
                    verify_plan_id,
                    verify_step_id,
                    accepted_len,
                    full_accept,
                    reject_position,
                    invalidated_len,
                    revised_token,
                    proposal_len,
                    to_verify_len,
                    gamma,
                    base_len,
                    base_pre_verify,
                    target_seq_len_at_verify,
                    target_seq_pre_verify_at_verify,
                    apply_action_code,
                ) = rest
                takeover_step_id = -1
                apply_plan_id = schedule_plan_id
                apply_step_id = schedule_step_id
                verify_result = self._verify_result_from_accept_len(int(accepted_len), int(proposal_len))
                append_token_count = int(proposal_len) if bool(full_accept) else 0
                discarded_token_count = 0 if bool(full_accept) else int(proposal_len)
                rollback_ok = 1
                mutation_detected = 0
                checkpoint_failed = 0
            results.append(
                {
                    "proposal_id": int(proposal_id),
                    "seq_id": int(seq_id),
                    "request_id": int(request_id),
                    "source_plan_id": int(source_plan_id),
                    "source_step_id": int(source_step_id),
                    "schedule_plan_id": int(schedule_plan_id),
                    "schedule_step_id": int(schedule_step_id),
                    "takeover_step_id": int(takeover_step_id),
                    "verify_plan_id": int(verify_plan_id),
                    "verify_step_id": int(verify_step_id),
                    "apply_plan_id": int(apply_plan_id),
                    "apply_step_id": int(apply_step_id),
                    "verify_result": str(verify_result),
                    "accepted_len": int(accepted_len),
                    "full_accept": bool(full_accept),
                    "reject_position": int(reject_position),
                    "invalidated_len": int(invalidated_len),
                    "revised_token": int(revised_token),
                    "proposal_len": int(proposal_len),
                    "to_verify_len": int(to_verify_len),
                    "gamma": int(gamma),
                    "base_len": int(base_len),
                    "base_pre_verify": bool(base_pre_verify),
                    "target_seq_len_at_verify": int(target_seq_len_at_verify),
                    "target_seq_pre_verify_at_verify": bool(target_seq_pre_verify_at_verify),
                    "apply_action": self._result_action_from_code(int(apply_action_code)),
                    "append_token_count": int(append_token_count),
                    "discarded_token_count": int(discarded_token_count),
                    "rollback_ok": bool(rollback_ok),
                    "mutation_detected": bool(mutation_detected),
                    "checkpoint_failed": bool(checkpoint_failed),
                }
            )
        return results

    def _send_eager_result_transfer_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        scheduled_proposals: list[EagerProposal],
    ) -> None:
        timer_start = time.perf_counter()
        results = self._build_eager_result_transfer_results(plan, trace_record, scheduled_proposals)
        meta_values, payload_values = self._serialize_eager_result_transfer_payload(results, plan)
        result_source = (
            EAGER_TAKEOVER_DRY_RUN_SOURCE
            if (
                trace_record.get("eager_apply_dry_run_source") == EAGER_TAKEOVER_DRY_RUN_SOURCE
                or self._eager_lane_exclusion_dry_run_enabled()
            )
            else EAGER_LEGACY_RESULT_TRANSFER_SOURCE
        )
        num_results = int(meta_values[2])
        payload_len = int(meta_values[3])
        sent_tokens = sum(int(result["proposal_len"]) for result in results)
        full_accept_tokens = sum(
            int(result["proposal_len"])
            for result in results
            if str(result.get("verify_result")) == "full_accept"
        )
        discarded_tokens = sum(
            int(result.get("discarded_token_count", 0)) for result in results
        )
        trace_record["enable_eager_result_transfer_dry_run"] = True
        trace_record["eager_result_transfer_dry_run_enabled"] = True
        trace_record["eager_result_transfer_dry_run_source"] = result_source
        trace_record["eager_result_transfer_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_result_transfer_plan_id"] = int(plan.plan_id)
        trace_record["eager_result_transfer_send_step"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_result_transfer_send_plan"] = int(plan.plan_id)
        trace_record["eager_result_transfer_num_results"] = num_results
        trace_record["eager_result_transfer_payload_len"] = payload_len
        trace_record["eager_result_transfer_sent_count"] = num_results
        trace_record["eager_result_transfer_sent_result_count"] = num_results
        trace_record["eager_result_transfer_zero_result_step"] = len(results) == 0
        trace_record["eager_result_transfer_sent_proposal_ids"] = [int(result["proposal_id"]) for result in results]
        trace_record["eager_result_transfer_sent_seq_ids"] = [int(result["seq_id"]) for result in results]
        trace_record["eager_result_transfer_action_by_proposal_id"] = {
            str(result["proposal_id"]): str(result["apply_action"]) for result in results
        }
        trace_record["eager_result_transfer_verify_result_by_proposal_id"] = {
            str(result["proposal_id"]): str(result.get("verify_result", "unknown")) for result in results
        }
        trace_record["eager_result_transfer_accept_len_by_proposal_id"] = {
            str(result["proposal_id"]): int(result["accepted_len"]) for result in results
        }
        trace_record["eager_result_transfer_append_tokens_by_proposal_id"] = {
            str(result["proposal_id"]): int(result.get("append_token_count", 0)) for result in results
        }
        trace_record["eager_result_transfer_discarded_tokens_by_proposal_id"] = {
            str(result["proposal_id"]): int(result.get("discarded_token_count", 0)) for result in results
        }
        trace_record["eager_result_transfer_rollback_ok_by_proposal_id"] = {
            str(result["proposal_id"]): bool(result.get("rollback_ok", False)) for result in results
        }
        trace_record["eager_result_transfer_mutation_detected_by_proposal_id"] = {
            str(result["proposal_id"]): bool(result.get("mutation_detected", True)) for result in results
        }
        trace_record["eager_result_transfer_checkpoint_failed_by_proposal_id"] = {
            str(result["proposal_id"]): bool(result.get("checkpoint_failed", True)) for result in results
        }
        trace_record["eager_result_sent_proposal_ids"] = [int(result["proposal_id"]) for result in results]
        trace_record["eager_result_sent_seq_ids"] = [int(result["seq_id"]) for result in results]
        trace_record["eager_result_sent_accepted_len_by_seq_id"] = {
            str(result["seq_id"]): int(result["accepted_len"]) for result in results
        }
        trace_record["eager_result_sent_full_accept_by_seq_id"] = {
            str(result["seq_id"]): bool(result["full_accept"]) for result in results
        }
        trace_record["eager_result_sent_reject_position_by_seq_id"] = {
            str(result["seq_id"]): int(result["reject_position"]) for result in results
        }
        trace_record["eager_result_sent_invalidated_len_by_seq_id"] = {
            str(result["seq_id"]): int(result["invalidated_len"]) for result in results
        }
        trace_record["eager_result_sent_revised_token_by_seq_id"] = {
            str(result["seq_id"]): int(result["revised_token"]) for result in results
        }
        trace_record["eager_result_sent_apply_action_by_seq_id"] = {
            str(result["seq_id"]): str(result["apply_action"]) for result in results
        }
        trace_record["eager_result_sent_proposal_len_by_proposal_id"] = {
            str(result["proposal_id"]): int(result["proposal_len"]) for result in results
        }
        trace_record["eager_result_sent_to_verify_len_by_proposal_id"] = {
            str(result["proposal_id"]): int(result["to_verify_len"]) for result in results
        }
        trace_record["eager_tokens_result_transfer_sent"] = sent_tokens
        trace_record["eager_tokens_result_transfer_dry_run"] = sent_tokens
        trace_record["eager_tokens_result_transfer_full_accept"] = full_accept_tokens
        trace_record["eager_tokens_result_transfer_discarded"] = discarded_tokens
        trace_record["eager_result_zero_result_step"] = len(results) == 0
        trace_record["result_transfer_called"] = True
        trace_record["result_transfer_zero_result"] = len(results) == 0
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
        broadcast_meta_values = [int(value) for value in meta.tolist()]
        payload_len = int(broadcast_meta_values[3])
        if payload_len > 0:
            if self.rank == self.global_config.target_config.master_rank:
                payload = torch.tensor(payload_values, dtype=torch.int64, device="cuda")
            else:
                payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.target_config.master_rank, group=self.verify_group)
        self._record_elapsed_ms(trace_record, "eager_result_transfer_time_ms", timer_start)

    def _validate_eager_result_on_draft(
        self,
        result: dict,
        known_proposals: dict[int, EagerProposal | ReadyEagerProposal],
        seq: Sequence | None,
        meta_gamma: int,
        result_source: str,
    ) -> str:
        proposal_id = int(result["proposal_id"])
        gamma = int(self.gamma)
        known_proposal = known_proposals.get(proposal_id)
        if known_proposal is None:
            return "unknown_proposal_id"
        if int(getattr(known_proposal, "seq_id", -1)) != int(result["seq_id"]):
            return "seq_id_mismatch"
        if (
            result_source == EAGER_TAKEOVER_DRY_RUN_SOURCE
            and str(getattr(known_proposal, "state", "")) != READY_EAGER_STATE_CONSUMED_APPLIED
        ):
            return "not_lane_exclusion_applied"
        if (
            result_source == EAGER_TAKEOVER_DRY_RUN_SOURCE
            and getattr(known_proposal, "takeover_routed_step_id", None) is None
        ):
            return "not_takeover_routed"
        if result_source == CONTINUOUS_EAGER_DRY_RUN_SOURCE:
            parent_id = int(result.get("parent_proposal_id", -1))
            known_parent_id = -1 if getattr(known_proposal, "parent_proposal_id", None) is None else int(known_proposal.parent_proposal_id)
            if parent_id != known_parent_id:
                return "parent_proposal_mismatch"
            if int(result.get("chain_depth", 0)) != 1:
                return "invalid_chain_depth"
            if getattr(known_proposal, "parent_kind", LANE_EAGER) != LANE_EAGER:
                return "invalid_parent_kind"
        if seq is None:
            return "seq_not_found"
        if int(meta_gamma) != gamma or int(result["gamma"]) != gamma:
            return "gamma_mismatch"
        proposal_len = int(result["proposal_len"])
        if proposal_len != gamma:
            return "invalid_proposal_len"
        if int(result["to_verify_len"]) != gamma:
            return "invalid_to_verify_len"
        accepted_len = int(result["accepted_len"])
        if not (0 <= accepted_len <= proposal_len):
            return "invalid_accepted_len"
        verify_result = str(result.get("verify_result", "unknown"))
        apply_action = str(result.get("apply_action", "unknown"))
        if bool(result["full_accept"]) != (verify_result == "full_accept"):
            return "full_accept_mismatch"
        if verify_result == "full_accept" and accepted_len != proposal_len:
            return "invalid_accepted_len"
        if verify_result == "partial_accept" and not (0 < accepted_len < proposal_len):
            return "invalid_accepted_len"
        if verify_result == "reject_at_first_token" and accepted_len != 0:
            return "invalid_accepted_len"
        if verify_result == "skipped_invalid" and apply_action != "skipped_invalid_no_mutation":
            return "bad_apply_action"
        if verify_result not in {
            "full_accept",
            "partial_accept",
            "reject_at_first_token",
            "skipped_invalid",
        }:
            return "unknown_verify_result"
        if bool(result["full_accept"]) != (accepted_len == proposal_len):
            return "full_accept_mismatch"
        if int(result["invalidated_len"]) != proposal_len - accepted_len:
            return "invalidated_len_mismatch"
        expected_action = {
            "full_accept": "append_full_accept_then_rollback",
            "partial_accept": "discard_partial_no_mutation",
            "reject_at_first_token": "discard_reject_no_mutation",
            "skipped_invalid": "skipped_invalid_no_mutation",
        }[verify_result]
        if verify_result == "reject_at_first_token" and apply_action == "discard_partial_no_mutation":
            expected_action = apply_action
        if apply_action != expected_action:
            return "bad_apply_action"
        if verify_result == "full_accept" and not bool(result.get("rollback_ok", False)):
            return "rollback_not_ok"
        if bool(result.get("mutation_detected", True)):
            return "mutation_detected"
        if bool(result.get("checkpoint_failed", True)):
            return "checkpoint_failed"
        if bool(result["base_pre_verify"]):
            return "invalid_base_pre_verify"
        for field_name in (
            "source_plan_id",
            "source_step_id",
            "schedule_plan_id",
            "schedule_step_id",
            "takeover_step_id",
            "verify_plan_id",
            "verify_step_id",
            "apply_plan_id",
            "apply_step_id",
        ):
            if (
                result_source in {EAGER_LEGACY_RESULT_TRANSFER_SOURCE, CONTINUOUS_EAGER_DRY_RUN_SOURCE}
                and field_name == "takeover_step_id"
            ):
                continue
            if int(result[field_name]) < 0:
                return "missing_plan_or_step_id"
        return "ok"

    def _receive_eager_result_transfer_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        known_proposals: list[EagerProposal],
    ) -> None:
        timer_start = time.perf_counter()
        known_by_id = dict(self._draft_sent_eager_proposals_by_id)
        known_by_id.update({int(proposal.proposal_id): proposal for proposal in known_proposals})
        known_by_id.update(
            {
                int(proposal.proposal_id): proposal
                for proposal in self.dual_batch_manager.ready_eager_proposals.proposals()
            }
        )
        if self.tp_params.local_rank != 0:
            self._record_elapsed_ms(trace_record, "eager_result_transfer_time_ms", timer_start)
            return
        meta = torch.zeros(EAGER_RESULT_TRANSFER_META_LEN, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
        meta_values = [int(value) for value in meta.tolist()]
        if int(meta_values[0]) != int(EAGER_RESULT_TRANSFER_MAGIC):
            raise ValueError(f"eager result transfer magic mismatch: got={meta_values[0]}")
        if int(meta_values[1]) != int(EAGER_RESULT_TRANSFER_OP_DRY_RUN):
            raise ValueError(f"eager result transfer op mismatch: got={meta_values[1]}")
        num_results, payload_len, gamma, result_plan_id, result_step_id = meta_values[2:7]
        result_source = (
            EAGER_TAKEOVER_DRY_RUN_SOURCE
            if self._eager_lane_exclusion_dry_run_enabled()
            else EAGER_LEGACY_RESULT_TRANSFER_SOURCE
        )
        payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
        if payload_len > 0:
            dist.broadcast(payload, src=self.global_config.target_config.master_rank, group=self.verify_group)
        results = self._deserialize_eager_result_transfer_payload(meta_values, payload.tolist())
        seq_by_id = self._local_sequence_by_id()
        validated: list[dict] = []
        invalid: list[dict] = []
        validation_reason_by_proposal_id: dict[int, str] = {}
        current_len_by_seq_id: dict[int, int] = {}
        base_len_by_seq_id: dict[int, int] = {}
        len_match_by_seq_id: dict[int, bool] = {}
        pre_verify_by_seq_id: dict[int, bool] = {}
        status_before_by_seq_id: dict[int, str] = {}
        status_after_by_seq_id: dict[int, str] = {}
        checkpoint_ok_by_seq_id: dict[int, bool] = {}
        mutation_detected_by_seq_id: dict[int, bool] = {}
        checkpoints: dict[int, dict] = {}
        duplicate_proposal_ids: list[int] = []
        seen_result_ids: set[int] = set()

        for result in results:
            seq_id = int(result["seq_id"])
            proposal_id = int(result["proposal_id"])
            seq = seq_by_id.get(seq_id)
            if seq is not None:
                checkpoints[seq_id] = make_sequence_checkpoint(seq)
            if proposal_id in seen_result_ids or proposal_id in self._eager_result_transfer_received_proposal_ids:
                reason = "duplicate_result"
                duplicate_proposal_ids.append(proposal_id)
            else:
                reason = self._validate_eager_result_on_draft(
                    result,
                    known_by_id,
                    seq,
                    gamma,
                    result_source,
                )
            validation_reason_by_proposal_id[proposal_id] = reason
            if reason == "ok":
                validated.append(result)
            else:
                invalid.append(result)
            seen_result_ids.add(proposal_id)
            current_len_by_seq_id[seq_id] = -1 if seq is None else int(len(seq))
            base_len_by_seq_id[seq_id] = int(result["base_len"])
            len_match_by_seq_id[seq_id] = seq is not None and int(len(seq)) == int(result["base_len"])
            pre_verify_by_seq_id[seq_id] = bool(getattr(seq, "pre_verify", True)) if seq is not None else True
            status_before_by_seq_id[seq_id] = self._sequence_status_name(seq)

        for result in results:
            seq_id = int(result["seq_id"])
            seq = seq_by_id.get(seq_id)
            status_after_by_seq_id[seq_id] = self._sequence_status_name(seq)
            if seq is None or seq_id not in checkpoints:
                continue
            try:
                assert_sequence_matches_checkpoint(seq, checkpoints[seq_id])
                checkpoint_ok_by_seq_id[seq_id] = True
                mutation_detected_by_seq_id[seq_id] = False
            except AssertionError:
                checkpoint_ok_by_seq_id[seq_id] = False
                mutation_detected_by_seq_id[seq_id] = True

        self._eager_result_transfer_received_proposal_ids.update(
            int(result["proposal_id"]) for result in results
        )
        missing_local_ids = [
            int(result["proposal_id"])
            for result in invalid
            if validation_reason_by_proposal_id.get(int(result["proposal_id"])) == "unknown_proposal_id"
        ]
        seq_mismatch_ids = [
            int(result["proposal_id"])
            for result in invalid
            if validation_reason_by_proposal_id.get(int(result["proposal_id"])) == "seq_id_mismatch"
        ]
        bad_action_ids = [
            int(result["proposal_id"])
            for result in invalid
            if validation_reason_by_proposal_id.get(int(result["proposal_id"])) in {
                "bad_apply_action",
                "unknown_verify_result",
                "rollback_not_ok",
                "mutation_detected",
                "checkpoint_failed",
            }
        ]
        bad_accept_len_ids = [
            int(result["proposal_id"])
            for result in invalid
            if validation_reason_by_proposal_id.get(int(result["proposal_id"])) in {
                "invalid_accepted_len",
                "full_accept_mismatch",
                "invalidated_len_mismatch",
            }
        ]
        received_tokens = sum(int(result["proposal_len"]) for result in results)
        validated_tokens = sum(int(result["proposal_len"]) for result in validated)
        invalid_tokens = sum(int(result["proposal_len"]) for result in invalid)
        full_accept_tokens = sum(
            int(result["proposal_len"]) for result in results if result.get("verify_result") == "full_accept"
        )
        discarded_tokens = sum(int(result.get("discarded_token_count", 0)) for result in results)

        trace_record["enable_eager_result_transfer_dry_run"] = True
        trace_record["eager_result_transfer_dry_run_enabled"] = True
        trace_record["eager_result_transfer_dry_run_source"] = result_source
        trace_record["eager_result_transfer_step_id"] = None if result_step_id < 0 else int(result_step_id)
        trace_record["eager_result_transfer_plan_id"] = int(result_plan_id)
        trace_record["eager_result_transfer_received_result_count"] = int(num_results)
        trace_record["eager_result_transfer_validated_result_count"] = len(validated)
        trace_record["eager_result_transfer_invalid_result_count"] = len(invalid)
        trace_record["eager_result_transfer_zero_result_step"] = len(results) == 0
        trace_record["eager_result_received_num_results"] = int(num_results)
        trace_record["eager_result_received_payload_len"] = int(payload_len)
        trace_record["eager_result_transfer_received_proposal_ids"] = [
            int(result["proposal_id"]) for result in results
        ]
        trace_record["eager_result_transfer_received_seq_ids"] = [int(result["seq_id"]) for result in results]
        trace_record["eager_result_transfer_received_count"] = int(num_results)
        trace_record["eager_result_transfer_validated_proposal_ids"] = [
            int(result["proposal_id"]) for result in validated
        ]
        trace_record["eager_result_transfer_invalid_proposal_ids"] = [
            int(result["proposal_id"]) for result in invalid
        ]
        trace_record["eager_result_transfer_validation_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in validation_reason_by_proposal_id.items()
        }
        trace_record["eager_result_transfer_duplicate_proposal_ids"] = sorted(set(duplicate_proposal_ids))
        trace_record["eager_result_transfer_missing_local_proposal_ids"] = missing_local_ids
        trace_record["eager_result_transfer_seq_mismatch_proposal_ids"] = seq_mismatch_ids
        trace_record["eager_result_transfer_bad_action_proposal_ids"] = bad_action_ids
        trace_record["eager_result_transfer_bad_accept_len_proposal_ids"] = bad_accept_len_ids
        trace_record["eager_result_transfer_draft_mutation_detected"] = any(
            bool(value) for value in mutation_detected_by_seq_id.values()
        )
        trace_record["eager_result_transfer_draft_checkpoint_failed"] = any(
            not bool(value) for value in checkpoint_ok_by_seq_id.values()
        )
        trace_record["eager_result_transfer_action_by_proposal_id"] = {
            str(result["proposal_id"]): str(result["apply_action"]) for result in results
        }
        trace_record["eager_result_transfer_verify_result_by_proposal_id"] = {
            str(result["proposal_id"]): str(result.get("verify_result", "unknown")) for result in results
        }
        trace_record["eager_result_transfer_accept_len_by_proposal_id"] = {
            str(result["proposal_id"]): int(result["accepted_len"]) for result in results
        }
        trace_record["eager_result_transfer_append_tokens_by_proposal_id"] = {
            str(result["proposal_id"]): int(result.get("append_token_count", 0)) for result in results
        }
        trace_record["eager_result_transfer_discarded_tokens_by_proposal_id"] = {
            str(result["proposal_id"]): int(result.get("discarded_token_count", 0)) for result in results
        }
        trace_record["eager_result_transfer_rollback_ok_by_proposal_id"] = {
            str(result["proposal_id"]): bool(result.get("rollback_ok", False)) for result in results
        }
        trace_record["eager_result_transfer_mutation_detected_by_proposal_id"] = {
            str(result["proposal_id"]): bool(result.get("mutation_detected", True)) for result in results
        }
        trace_record["eager_result_transfer_checkpoint_failed_by_proposal_id"] = {
            str(result["proposal_id"]): bool(result.get("checkpoint_failed", True)) for result in results
        }
        trace_record["eager_result_received_proposal_ids"] = [int(result["proposal_id"]) for result in results]
        trace_record["eager_result_received_seq_ids"] = [int(result["seq_id"]) for result in results]
        trace_record["eager_result_validated_proposal_ids"] = [int(result["proposal_id"]) for result in validated]
        trace_record["eager_result_invalid_proposal_ids"] = [int(result["proposal_id"]) for result in invalid]
        trace_record["eager_result_validation_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in validation_reason_by_proposal_id.items()
        }
        trace_record["eager_result_received_accepted_len_by_seq_id"] = {
            str(result["seq_id"]): int(result["accepted_len"]) for result in results
        }
        trace_record["eager_result_received_full_accept_by_seq_id"] = {
            str(result["seq_id"]): bool(result["full_accept"]) for result in results
        }
        trace_record["eager_result_received_reject_position_by_seq_id"] = {
            str(result["seq_id"]): int(result["reject_position"]) for result in results
        }
        trace_record["eager_result_received_invalidated_len_by_seq_id"] = {
            str(result["seq_id"]): int(result["invalidated_len"]) for result in results
        }
        trace_record["eager_result_received_revised_token_by_seq_id"] = {
            str(result["seq_id"]): int(result["revised_token"]) for result in results
        }
        trace_record["eager_result_received_proposal_len_by_proposal_id"] = {
            str(result["proposal_id"]): int(result["proposal_len"]) for result in results
        }
        trace_record["eager_result_received_to_verify_len_by_proposal_id"] = {
            str(result["proposal_id"]): int(result["to_verify_len"]) for result in results
        }
        trace_record["eager_result_draft_current_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_result_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_result_draft_len_matches_base_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_match_by_seq_id.items()
        }
        trace_record["eager_result_draft_seq_pre_verify_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_by_seq_id.items()
        }
        trace_record["eager_result_draft_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_before_by_seq_id.items()
        }
        trace_record["eager_result_draft_status_after_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_after_by_seq_id.items()
        }
        trace_record["eager_result_draft_checkpoint_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in checkpoint_ok_by_seq_id.items()
        }
        trace_record["eager_result_draft_mutation_detected_by_seq_id"] = {
            str(seq_id): value for seq_id, value in mutation_detected_by_seq_id.items()
        }
        trace_record["eager_tokens_result_transfer_received"] = received_tokens
        trace_record["eager_tokens_result_transfer_validated"] = validated_tokens
        trace_record["eager_tokens_result_transfer_invalid"] = invalid_tokens
        trace_record["eager_tokens_result_transfer_dry_run"] = received_tokens
        trace_record["eager_tokens_result_transfer_full_accept"] = full_accept_tokens
        trace_record["eager_tokens_result_transfer_discarded"] = discarded_tokens
        trace_record["eager_result_zero_result_step"] = len(results) == 0
        trace_record["result_transfer_called"] = True
        trace_record["result_transfer_zero_result"] = len(results) == 0
        self._record_elapsed_ms(trace_record, "eager_result_transfer_time_ms", timer_start)
        if self._eager_sync_apply_dry_run_enabled():
            plan.eager_sync_apply_dry_run_enabled = True
            self._run_draft_sync_apply_dry_run(
                plan,
                trace_record,
                validated,
                known_by_id,
                seq_by_id,
            )
        if self._eager_commit_ready_only_enabled():
            self._send_eager_commit_ready_only_decision(
                plan,
                trace_record,
                known_by_id,
                seq_by_id,
            )

    def _local_sequence_by_id(self) -> dict[int, Sequence]:
        seq_by_id = {}
        for seq in list(self.scheduler.running) + list(self.scheduler.waiting) + list(self.scheduler.pending_cached) + list(self.scheduler.finished):
            seq_by_id[int(seq.seq_id)] = seq
        return seq_by_id

    def _eager_transfer_metadata_drop_reason(self, proposal: EagerProposal) -> str | None:
        if bool(proposal.base_pre_verify):
            return "invalid_base_pre_verify"
        if int(proposal.proposal_len) != int(self.gamma):
            return "invalid_proposal_len"
        if len(proposal.to_be_verified_token_ids) != int(self.gamma):
            return "invalid_to_verify_len"
        if len(proposal.proposal_token_ids) != int(self.gamma):
            return "invalid_proposal_token_len"
        if proposal.parent_kind != LANE_NORMAL:
            return "parent_dependency_invalidated"
        return None

    def _eager_transfer_plan_context(self, plan: StepPlan, trace_record: dict | None) -> dict[str, set[int]]:
        trace_record = trace_record or {}
        scheduled = {int(seq_id) for seq_id in trace_record.get("scheduled_seq_ids", [])}
        resolved = {int(seq_id) for seq_id in trace_record.get("resolved_seq_ids", [])}
        target_home = {int(seq_id) for seq_id in plan.target_home_set}
        draft_home = {int(seq_id) for seq_id in plan.draft_home_set}
        target_eager = {int(seq_id) for seq_id in plan.target_eager_set}
        draft_eager = {int(seq_id) for seq_id in plan.draft_eager_set}
        return {
            "scheduled_seq_ids": scheduled,
            "resolved_seq_ids": resolved,
            "target_home_set": target_home,
            "draft_home_set": draft_home,
            "target_eager_set": target_eager,
            "draft_eager_set": draft_eager,
            "active_plan_seq_ids": scheduled | resolved | target_home | draft_home | target_eager | draft_eager,
        }

    def _sequence_status_name(self, seq: Sequence | None) -> str:
        if seq is None:
            return "MISSING"
        status = getattr(seq, "status", None)
        return status.name if isinstance(status, SequenceStatus) else str(status)

    def _seq_in_plan_context(self, seq_id: int, plan_context: dict[str, set[int]]) -> bool:
        return int(seq_id) in plan_context.get("active_plan_seq_ids", set())

    def is_request_level_finished(self, seq: Sequence | None, plan_context: dict[str, set[int]]) -> bool:
        if seq is None or not bool(getattr(seq, "is_finished", False)):
            return False
        if self._seq_in_plan_context(int(seq.seq_id), plan_context):
            return False
        if getattr(seq, "finish_ts", None) is not None:
            return True
        return True

    def is_speculative_span_invalidated(
        self,
        seq: Sequence | None,
        plan_context: dict[str, set[int]],
    ) -> bool:
        if seq is None:
            return False
        return bool(getattr(seq, "is_finished", False)) and self._seq_in_plan_context(
            int(seq.seq_id),
            plan_context,
        )

    def _classify_eager_transfer_proposal(
        self,
        proposal: EagerProposal,
        seq: Sequence | None,
        plan_context: dict[str, set[int]],
        *,
        pending_update: bool = False,
    ) -> tuple[str, str]:
        metadata_reason = self._eager_transfer_metadata_drop_reason(proposal)
        if metadata_reason is not None:
            return "drop", metadata_reason
        if seq is None:
            return "drop", "seq_not_found_later" if pending_update else "seq_not_found"
        if self.is_request_level_finished(seq, plan_context):
            return "drop", "seq_finished_before_base"
        if self.is_speculative_span_invalidated(seq, plan_context):
            return "drop", "seq_span_invalidated_before_base"
        if getattr(seq, "status", None) != SequenceStatus.RUNNING:
            return "drop", "seq_not_running"
        if bool(seq.pre_verify):
            return "drop", "seq_returned_pre_verify_before_base"
        current_len = int(len(seq))
        base_len = int(proposal.base_len)
        if current_len < base_len:
            return "pending", "pending_base_not_reached"
        if current_len == base_len:
            return "ready", "base_reached"
        return "drop", "base_overshot_or_stale"

    def _record_eager_transfer_seq_state(
        self,
        proposal: EagerProposal,
        seq: Sequence | None,
        base_len_by_seq_id: dict[int, int],
        base_pre_verify_by_seq_id: dict[int, bool],
        current_len_by_seq_id: dict[int, int],
        base_match_by_seq_id: dict[int, bool],
        proposal_len_by_proposal_id: dict[int, int],
        to_verify_len_by_proposal_id: dict[int, int],
        base_delta_by_seq_id: dict[int, int],
        seq_raw_status_by_seq_id: dict[int, str],
        seq_is_finished_raw_by_seq_id: dict[int, bool],
        seq_request_finished_by_seq_id: dict[int, bool],
        seq_span_invalidated_by_seq_id: dict[int, bool],
        seq_in_scheduled_by_seq_id: dict[int, bool],
        seq_in_resolved_by_seq_id: dict[int, bool],
        seq_in_target_home_by_seq_id: dict[int, bool],
        seq_in_draft_home_by_seq_id: dict[int, bool],
        plan_context: dict[str, set[int]],
    ) -> None:
        seq_id = int(proposal.seq_id)
        proposal_id = int(proposal.proposal_id)
        current_len = -1 if seq is None else int(len(seq))
        base_len = int(proposal.base_len)
        base_len_by_seq_id[seq_id] = base_len
        base_pre_verify_by_seq_id[seq_id] = bool(proposal.base_pre_verify)
        current_len_by_seq_id[seq_id] = current_len
        base_match_by_seq_id[seq_id] = seq is not None and current_len == base_len
        base_delta_by_seq_id[seq_id] = base_len - current_len if seq is not None else base_len
        proposal_len_by_proposal_id[proposal_id] = int(proposal.proposal_len)
        to_verify_len_by_proposal_id[proposal_id] = len(proposal.to_be_verified_token_ids)
        seq_raw_status_by_seq_id[seq_id] = self._sequence_status_name(seq)
        seq_is_finished_raw_by_seq_id[seq_id] = bool(getattr(seq, "is_finished", False)) if seq is not None else False
        seq_request_finished_by_seq_id[seq_id] = self.is_request_level_finished(seq, plan_context)
        seq_span_invalidated_by_seq_id[seq_id] = self.is_speculative_span_invalidated(seq, plan_context)
        seq_in_scheduled_by_seq_id[seq_id] = seq_id in plan_context.get("scheduled_seq_ids", set())
        seq_in_resolved_by_seq_id[seq_id] = seq_id in plan_context.get("resolved_seq_ids", set())
        seq_in_target_home_by_seq_id[seq_id] = seq_id in plan_context.get("target_home_set", set())
        seq_in_draft_home_by_seq_id[seq_id] = seq_id in plan_context.get("draft_home_set", set())

    def _update_target_pending_eager_buffer(
        self,
        seq_by_id: dict[int, Sequence],
        base_len_by_seq_id: dict[int, int],
        base_pre_verify_by_seq_id: dict[int, bool],
        current_len_by_seq_id: dict[int, int],
        base_match_by_seq_id: dict[int, bool],
        proposal_len_by_proposal_id: dict[int, int],
        to_verify_len_by_proposal_id: dict[int, int],
        base_delta_by_seq_id: dict[int, int],
        seq_raw_status_by_seq_id: dict[int, str],
        seq_is_finished_raw_by_seq_id: dict[int, bool],
        seq_request_finished_by_seq_id: dict[int, bool],
        seq_span_invalidated_by_seq_id: dict[int, bool],
        seq_in_scheduled_by_seq_id: dict[int, bool],
        seq_in_resolved_by_seq_id: dict[int, bool],
        seq_in_target_home_by_seq_id: dict[int, bool],
        seq_in_draft_home_by_seq_id: dict[int, bool],
        plan_context: dict[str, set[int]],
    ) -> tuple[list[EagerProposal], list[EagerProposal], list[EagerProposal], dict[int, str]]:
        ready: list[EagerProposal] = []
        still_pending: list[EagerProposal] = []
        dropped: list[EagerProposal] = []
        state_by_proposal_id: dict[int, str] = {}

        pending_proposals = [
            proposal
            for proposal in self.eager_proposal_buffer.proposals()
            if proposal.valid and proposal.state == EAGER_STATE_PENDING_BASE_REACHED
        ]
        for proposal in pending_proposals:
            seq = seq_by_id.get(int(proposal.seq_id))
            self._record_eager_transfer_seq_state(
                proposal,
                seq,
                base_len_by_seq_id,
                base_pre_verify_by_seq_id,
                current_len_by_seq_id,
                base_match_by_seq_id,
                proposal_len_by_proposal_id,
                to_verify_len_by_proposal_id,
                base_delta_by_seq_id,
                seq_raw_status_by_seq_id,
                seq_is_finished_raw_by_seq_id,
                seq_request_finished_by_seq_id,
                seq_span_invalidated_by_seq_id,
                seq_in_scheduled_by_seq_id,
                seq_in_resolved_by_seq_id,
                seq_in_target_home_by_seq_id,
                seq_in_draft_home_by_seq_id,
                plan_context,
            )
            action, reason = self._classify_eager_transfer_proposal(
                proposal,
                seq,
                plan_context,
                pending_update=True,
            )
            state_by_proposal_id[int(proposal.proposal_id)] = reason
            if action == "ready":
                proposal.state = EAGER_STATE_READY_TO_VERIFY_DRY_RUN
                proposal.valid = True
                ready.append(proposal)
            elif action == "pending":
                proposal.state = EAGER_STATE_PENDING_BASE_REACHED
                proposal.valid = True
                still_pending.append(proposal)
            else:
                proposal.state = EAGER_STATE_DISCARDED
                proposal.valid = False
                dropped.append(proposal)

        self.eager_proposal_buffer.remove_many(
            [proposal.proposal_id for proposal in dropped]
        )
        return ready, still_pending, dropped, state_by_proposal_id

    def _ready_eager_proposals(self) -> list[EagerProposal]:
        return sorted(
            [
                proposal
                for proposal in self.eager_proposal_buffer.proposals()
                if proposal.valid and proposal.state == EAGER_STATE_READY_TO_VERIFY_DRY_RUN
            ],
            key=lambda proposal: (int(proposal.proposal_id), int(proposal.seq_id)),
        )

    @torch.inference_mode()
    def compute_pearl_verify_result_no_apply(
        self,
        logits: torch.Tensor,
        seqs: list[Sequence],
        temperatures: torch.Tensor | None,
        proposals: list[EagerProposal],
        gamma: int,
        lane: str = LANE_EAGER,
    ) -> dict[str, dict[int, int | bool]]:
        """Compute PEARL verification results without mutating Sequence state."""
        if lane != LANE_EAGER:
            raise ValueError(f"unsupported no-apply verify lane={lane!r}")
        to_be_verified_tokens: list[int] = []
        for proposal in proposals:
            to_be_verified_tokens.extend(int(token_id) for token_id in proposal.to_be_verified_token_ids)
        num_to_be_verified_tokens = len(to_be_verified_tokens)
        msg = torch.tensor(to_be_verified_tokens, dtype=torch.int64, device="cuda")
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")

        if self.tp_params.local_rank == 0 and seqs:
            r = torch.rand(num_to_be_verified_tokens, device="cuda")
            target_logits = norm_logits(logits, temperatures)
            target_prob = target_logits.gather(dim=1, index=msg.unsqueeze(1)).squeeze(1)
            judge = (r <= target_prob).tolist()

            revised_logits = logits.clone()
            revised_logits.scatter_(1, msg.unsqueeze(1), -float("inf"))
            revised_tokens = self.sampler(revised_logits, temperatures)

            acc, rollout, revise_token, finish = [], [], [], []
            v_idx = 0
            for seq in seqs:
                if seq.pre_verify:
                    n = 1 if judge[v_idx] else 0
                    acc.append(bool(judge[v_idx]))
                    rollout.append(0 if judge[v_idx] else int(gamma))
                    revise_token.append(int(revised_tokens[v_idx]))
                    finish.append(
                        (not seq.ignore_eos and judge[v_idx] and is_eos(to_be_verified_tokens[v_idx], self.scheduler.eos))
                        or seq.num_completion_tokens >= seq.max_tokens - 1
                    )
                    v_idx += 1
                    continue

                n = int(gamma)
                finish_flag = False
                for j in range(v_idx, v_idx + int(gamma)):
                    if not seq.ignore_eos and judge[j] and is_eos(to_be_verified_tokens[j], self.scheduler.eos):
                        finish_flag = True
                    if not judge[j]:
                        n = j - v_idx
                        break
                acc.append(n == int(gamma))
                rollout.append(int(gamma) - n)
                revise_token.append(int(revised_tokens[n + v_idx]) if n < int(gamma) else -1)
                finish.append(finish_flag or seq.num_completion_tokens >= seq.max_tokens - min(n + 1, int(gamma)))
                v_idx += int(gamma)

            verify_res = torch.tensor([acc, rollout, revise_token, finish], dtype=torch.int64, device="cuda")

        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank, group=self.group)
        acc, rollout, revise_token, _finish = verify_res.tolist()

        accepted_len_by_seq_id: dict[int, int] = {}
        invalidated_len_by_seq_id: dict[int, int] = {}
        reject_position_by_seq_id: dict[int, int] = {}
        revised_token_by_seq_id: dict[int, int] = {}
        full_accept_by_seq_id: dict[int, bool] = {}
        for idx, seq in enumerate(seqs):
            seq_id = int(seq.seq_id)
            if seq.pre_verify:
                accepted_len = 1 if acc[idx] else 0
            else:
                accepted_len = int(gamma) if acc[idx] else int(gamma) - int(rollout[idx])
            invalidated_len = 0 if acc[idx] else int(rollout[idx])
            accepted_len_by_seq_id[seq_id] = int(accepted_len)
            invalidated_len_by_seq_id[seq_id] = int(invalidated_len)
            reject_position_by_seq_id[seq_id] = -1 if acc[idx] else int(accepted_len)
            revised_token_by_seq_id[seq_id] = int(revise_token[idx])
            full_accept_by_seq_id[seq_id] = bool(accepted_len == int(gamma))

        return {
            "accepted_len_by_seq_id": accepted_len_by_seq_id,
            "invalidated_len_by_seq_id": invalidated_len_by_seq_id,
            "reject_position_by_seq_id": reject_position_by_seq_id,
            "revised_token_by_seq_id": revised_token_by_seq_id,
            "full_accept_by_seq_id": full_accept_by_seq_id,
        }

    def _run_eager_verify_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        scheduled_proposals: list[EagerProposal],
        plan_context: dict[str, set[int]],
    ) -> None:
        seq_by_id = self._local_sequence_by_id()
        gamma = int(self.gamma)
        scheduled_proposal_ids = {int(proposal.proposal_id) for proposal in scheduled_proposals}
        scheduled_seq_ids = {int(seq_id) for seq_id in plan.scheduled_target_eager_seq_ids_dry_run}
        adjusted_draft_home = {int(seq_id) for seq_id in plan.adjusted_draft_home_set_dry_run}

        executed_proposals: list[EagerProposal] = []
        executed_seqs: list[Sequence] = []
        skipped_proposal_ids: list[int] = []
        skip_reason_by_proposal_id: dict[int, str] = {}
        base_len_by_seq_id: dict[int, int] = {}
        current_len_by_seq_id: dict[int, int] = {}
        base_match_by_seq_id: dict[int, bool] = {}
        seq_pre_verify_by_seq_id: dict[int, bool] = {}
        seq_status_before_by_seq_id: dict[int, str] = {}

        for proposal in scheduled_proposals:
            proposal_id = int(proposal.proposal_id)
            seq_id = int(proposal.seq_id)
            seq = seq_by_id.get(seq_id)
            current_len = -1 if seq is None else int(len(seq))
            base_len_by_seq_id[seq_id] = int(proposal.base_len)
            current_len_by_seq_id[seq_id] = current_len
            base_match_by_seq_id[seq_id] = seq is not None and current_len == int(proposal.base_len)
            seq_pre_verify_by_seq_id[seq_id] = bool(getattr(seq, "pre_verify", True)) if seq is not None else True
            seq_status_before_by_seq_id[seq_id] = self._sequence_status_name(seq)

            reason = None
            if proposal_id not in scheduled_proposal_ids:
                reason = "not_scheduled_proposal"
            elif seq_id not in scheduled_seq_ids:
                reason = "not_scheduled_seq"
            elif seq_id in {int(seq_id) for seq_id in plan.target_home_set}:
                reason = "intersects_target_home"
            elif seq_id in adjusted_draft_home:
                reason = "intersects_adjusted_draft_home"
            elif proposal.lane != LANE_EAGER:
                reason = "invalid_lane"
            elif proposal.state not in {EAGER_STATE_READY_TO_VERIFY_DRY_RUN, EAGER_STATE_SCHEDULED_DRY_RUN}:
                reason = "invalid_state"
            elif bool(proposal.base_pre_verify):
                reason = "invalid_base_pre_verify"
            elif int(proposal.proposal_len) != gamma:
                reason = "invalid_proposal_len"
            elif len(proposal.to_be_verified_token_ids) != gamma:
                reason = "invalid_to_verify_len"
            elif len(proposal.proposal_token_ids) != gamma:
                reason = "invalid_proposal_token_len"
            elif seq is None:
                reason = "seq_not_found"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished_before_verify"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "seq_span_invalidated_before_verify"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_returned_pre_verify_before_verify"
            elif int(len(seq)) != int(proposal.base_len):
                reason = "base_mismatch_before_verify"

            if reason is None:
                proposal.state = EAGER_STATE_SCHEDULED_DRY_RUN
                executed_proposals.append(proposal)
                executed_seqs.append(seq)
            else:
                skipped_proposal_ids.append(proposal_id)
                skip_reason_by_proposal_id[proposal_id] = reason

        checkpoints = {int(seq.seq_id): make_sequence_checkpoint(seq) for seq in executed_seqs}
        results: dict[str, dict[int, int | bool]] = {
            "accepted_len_by_seq_id": {},
            "invalidated_len_by_seq_id": {},
            "reject_position_by_seq_id": {},
            "revised_token_by_seq_id": {},
            "full_accept_by_seq_id": {},
        }
        if executed_seqs:
            input_ids, positions, temp_seqs = self.prepare_pearl_decode(executed_seqs)
            temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
            torch.cuda.synchronize()
            logits = self.run_model(input_ids, positions, False)
            results = self.compute_pearl_verify_result_no_apply(
                logits,
                executed_seqs,
                temperatures,
                executed_proposals,
                gamma=gamma,
                lane=LANE_EAGER,
            )
            torch.cuda.synchronize()

        seq_status_after_by_seq_id: dict[int, str] = {}
        mutation_detected_by_seq_id: dict[int, bool] = {}
        checkpoint_ok_by_seq_id: dict[int, bool] = {}
        for seq in executed_seqs:
            seq_id = int(seq.seq_id)
            seq_status_after_by_seq_id[seq_id] = self._sequence_status_name(seq)
            try:
                assert_sequence_matches_checkpoint(seq, checkpoints[seq_id])
                checkpoint_ok_by_seq_id[seq_id] = True
                mutation_detected_by_seq_id[seq_id] = False
            except AssertionError:
                checkpoint_ok_by_seq_id[seq_id] = False
                mutation_detected_by_seq_id[seq_id] = True

        trace_record["enable_eager_verify_dry_run"] = True
        trace_record["eager_verify_dry_run_enabled"] = True
        trace_record["eager_verify_dry_run_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_verify_dry_run_plan_id"] = int(plan.plan_id)
        trace_record["eager_verify_candidate_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in scheduled_proposals
        ]
        trace_record["eager_verify_candidate_seq_ids"] = [int(proposal.seq_id) for proposal in scheduled_proposals]
        trace_record["eager_verify_skipped_proposal_ids"] = [int(proposal_id) for proposal_id in skipped_proposal_ids]
        trace_record["eager_verify_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in skip_reason_by_proposal_id.items()
        }
        trace_record["eager_verify_executed_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in executed_proposals
        ]
        trace_record["eager_verify_executed_seq_ids"] = [int(seq.seq_id) for seq in executed_seqs]
        trace_record["eager_verify_accepted_len_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["accepted_len_by_seq_id"].items()
        }
        trace_record["eager_verify_full_accept_by_seq_id"] = {
            str(seq_id): bool(value)
            for seq_id, value in results["full_accept_by_seq_id"].items()
        }
        trace_record["eager_verify_reject_position_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["reject_position_by_seq_id"].items()
        }
        trace_record["eager_verify_invalidated_len_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["invalidated_len_by_seq_id"].items()
        }
        trace_record["eager_verify_revised_token_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["revised_token_by_seq_id"].items()
        }
        trace_record["eager_verify_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_verify_current_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_verify_base_match_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_match_by_seq_id.items()
        }
        trace_record["eager_verify_seq_pre_verify_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_pre_verify_by_seq_id.items()
        }
        trace_record["eager_verify_seq_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_status_before_by_seq_id.items()
        }
        trace_record["eager_verify_seq_status_after_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_status_after_by_seq_id.items()
        }
        trace_record["eager_verify_mutation_detected_by_seq_id"] = {
            str(seq_id): value for seq_id, value in mutation_detected_by_seq_id.items()
        }
        trace_record["eager_verify_checkpoint_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in checkpoint_ok_by_seq_id.items()
        }
        trace_record["eager_tokens_verify_dry_run"] = sum(
            int(proposal.proposal_len) for proposal in executed_proposals
        )
        trace_record["eager_tokens_verify_dry_run_full_accept"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if bool(results["full_accept_by_seq_id"].get(int(proposal.seq_id), False))
        )
        trace_record["eager_tokens_verify_dry_run_rejected"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if not bool(results["full_accept_by_seq_id"].get(int(proposal.seq_id), False))
        )

    def _ready_takeover_to_eager_proposal(
        self,
        proposal: ReadyEagerProposal,
        seq: Sequence,
    ) -> EagerProposal:
        return EagerProposal(
            proposal_id=int(proposal.proposal_id),
            seq_id=int(proposal.seq_id),
            request_id=proposal.request_id if proposal.request_id is not None else seq.request_id,
            lane=LANE_EAGER,
            parent_proposal_id=proposal.parent_proposal_id,
            parent_kind=LANE_NORMAL,
            parent_step_id=None,
            source_step_id=int(proposal.source_step_id),
            source_plan_id=int(proposal.source_plan_id),
            home_batch_id=-1 if seq.home_batch_id is None else int(seq.home_batch_id),
            base_len=int(proposal.base_len),
            base_pre_verify=bool(proposal.base_pre_verify),
            base_num_completion_tokens=int(getattr(seq, "num_completion_tokens", 0)),
            proposal_token_ids=[int(token_id) for token_id in proposal.proposal_token_ids],
            to_be_verified_token_ids=[
                int(token_id) for token_id in proposal.to_be_verified_token_ids
            ],
            proposal_len=int(proposal.proposal_len),
            state=EAGER_STATE_SCHEDULED_DRY_RUN,
            valid=True,
        )

    def _run_takeover_eager_verify_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        plan_context: dict[str, set[int]],
    ) -> None:
        candidate_proposal_ids = [
            int(proposal_id) for proposal_id in plan.target_eager_verify_proposal_ids_dry_run
        ]
        candidate_seq_ids = [
            int(seq_id) for seq_id in plan.target_eager_verify_seq_ids_dry_run
        ]
        assert len(candidate_proposal_ids) == len(candidate_seq_ids), (
            self._proposal_assertion_message(
                plan,
                "target eager verify dry-run proposal/seq length mismatch",
            )
        )

        seq_by_id = self._local_sequence_by_id()
        target_home = {int(seq_id) for seq_id in plan.target_home_set}
        target_normal = {int(seq_id) for seq_id in self._target_normal_verify_seq_ids(plan)}
        current_step_id = int(plan.step_id if plan.step_id is not None else self.dual_batch_manager.step_id)
        gamma = int(self.gamma)

        executed_ready_proposals: list[ReadyEagerProposal] = []
        executed_proposals: list[EagerProposal] = []
        executed_seqs: list[Sequence] = []
        skipped_proposal_ids: list[int] = []
        skip_reason_by_proposal_id: dict[int, str] = {}
        seq_id_by_proposal_id: dict[int, int] = {}
        proposal_len_by_proposal_id: dict[int, int] = {}
        to_verify_len_by_proposal_id: dict[int, int] = {}
        base_len_by_proposal_id: dict[int, int] = {}
        takeover_step_by_proposal_id: dict[int, int] = {}
        base_len_by_seq_id: dict[int, int] = {}
        current_len_by_seq_id: dict[int, int] = {}
        base_match_by_seq_id: dict[int, bool] = {}
        seq_pre_verify_by_seq_id: dict[int, bool] = {}
        seq_status_before_by_seq_id: dict[int, str] = {}

        for proposal_id, routed_seq_id in zip(candidate_proposal_ids, candidate_seq_ids):
            ready_proposal = self.dual_batch_manager.ready_eager_proposals.by_id(proposal_id)
            seq_id = int(routed_seq_id)
            seq = seq_by_id.get(seq_id)
            proposal_seq_id = seq_id if ready_proposal is None else int(ready_proposal.seq_id)
            seq_id_by_proposal_id[proposal_id] = proposal_seq_id
            proposal_len_by_proposal_id[proposal_id] = (
                -1 if ready_proposal is None else int(ready_proposal.proposal_len)
            )
            to_verify_len_by_proposal_id[proposal_id] = (
                -1 if ready_proposal is None else int(ready_proposal.to_verify_len)
            )
            base_len_by_proposal_id[proposal_id] = (
                -1 if ready_proposal is None else int(ready_proposal.base_len)
            )
            takeover_step = (
                -1
                if ready_proposal is None or ready_proposal.takeover_routed_step_id is None
                else int(ready_proposal.takeover_routed_step_id)
            )
            takeover_step_by_proposal_id[proposal_id] = takeover_step
            current_len = -1 if seq is None else int(len(seq))
            base_len_by_seq_id[proposal_seq_id] = (
                -1 if ready_proposal is None else int(ready_proposal.base_len)
            )
            current_len_by_seq_id[proposal_seq_id] = current_len
            base_match_by_seq_id[proposal_seq_id] = (
                ready_proposal is not None
                and seq is not None
                and current_len == int(ready_proposal.base_len)
            )
            seq_pre_verify_by_seq_id[proposal_seq_id] = (
                True if seq is None else bool(getattr(seq, "pre_verify", True))
            )
            seq_status_before_by_seq_id[proposal_seq_id] = self._sequence_status_name(seq)

            reason = None
            if ready_proposal is None:
                reason = "ready_proposal_not_found"
            elif int(ready_proposal.seq_id) != seq_id:
                reason = "routed_seq_mismatch"
            elif seq_id not in target_home:
                reason = "seq_not_in_target_home"
            elif seq_id in target_normal:
                reason = "seq_still_in_target_normal_verify"
            elif ready_proposal.state != READY_EAGER_STATE_CONSUMED_APPLIED:
                reason = "invalid_ready_proposal_state"
            elif ready_proposal.takeover_routed_step_id is None:
                reason = "takeover_not_routed"
            elif int(ready_proposal.takeover_routed_step_id) != current_step_id:
                reason = "takeover_routed_in_different_step"
            elif ready_proposal.verify_dry_run_step_id is not None:
                reason = "verify_dry_run_already_executed"
            elif bool(ready_proposal.base_pre_verify):
                reason = "invalid_base_pre_verify"
            elif int(ready_proposal.proposal_len) != gamma:
                reason = "invalid_proposal_len"
            elif int(ready_proposal.to_verify_len) != gamma:
                reason = "invalid_to_verify_len"
            elif len(ready_proposal.proposal_token_ids) != gamma:
                reason = "invalid_proposal_token_len"
            elif len(ready_proposal.to_be_verified_token_ids) != gamma:
                reason = "invalid_to_verify_token_len"
            elif seq is None:
                reason = "seq_not_found"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished_before_verify"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "seq_span_invalidated_before_verify"
            elif getattr(seq, "status", None) != SequenceStatus.RUNNING:
                reason = "seq_not_running"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_returned_pre_verify_before_verify"
            elif int(len(seq)) < int(ready_proposal.base_len):
                reason = "base_not_reached_before_verify"
            elif int(len(seq)) > int(ready_proposal.base_len):
                reason = "base_overshot_before_verify"

            if reason is None:
                eager_proposal = self._ready_takeover_to_eager_proposal(ready_proposal, seq)
                executed_ready_proposals.append(ready_proposal)
                executed_proposals.append(eager_proposal)
                executed_seqs.append(seq)
            else:
                skipped_proposal_ids.append(proposal_id)
                skip_reason_by_proposal_id[proposal_id] = reason

        checkpoints = {int(seq.seq_id): make_sequence_checkpoint(seq) for seq in executed_seqs}
        token_ids_before_by_seq_id = {
            int(seq.seq_id): [int(token_id) for token_id in seq.token_ids]
            for seq in executed_seqs
        }
        len_before_by_seq_id = {int(seq.seq_id): int(len(seq)) for seq in executed_seqs}
        results: dict[str, dict[int, int | bool]] = {
            "accepted_len_by_seq_id": {},
            "invalidated_len_by_seq_id": {},
            "reject_position_by_seq_id": {},
            "revised_token_by_seq_id": {},
            "full_accept_by_seq_id": {},
        }

        if executed_seqs:
            self._allocate_decode_slots_for_dual(executed_seqs, plan, "target_eager_verify_dry_run")
            input_ids, positions, temp_seqs = self.prepare_pearl_decode(executed_seqs)
            temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
            torch.cuda.synchronize()
            logits = self.run_model(input_ids, positions, False)
            results = self.compute_pearl_verify_result_no_apply(
                logits,
                executed_seqs,
                temperatures,
                executed_proposals,
                gamma=gamma,
                lane=LANE_EAGER,
            )
            torch.cuda.synchronize()

        seq_status_after_by_seq_id: dict[int, str] = {}
        mutation_detected_by_seq_id: dict[int, bool] = {}
        checkpoint_ok_by_seq_id: dict[int, bool] = {}
        len_after_by_seq_id: dict[int, int] = {}
        for seq in executed_seqs:
            seq_id = int(seq.seq_id)
            seq_status_after_by_seq_id[seq_id] = self._sequence_status_name(seq)
            len_after_by_seq_id[seq_id] = int(len(seq))
            token_ids_match = [int(token_id) for token_id in seq.token_ids] == token_ids_before_by_seq_id[seq_id]
            try:
                assert_sequence_matches_checkpoint(seq, checkpoints[seq_id])
                checkpoint_ok = bool(token_ids_match)
            except AssertionError:
                checkpoint_ok = False
            checkpoint_ok_by_seq_id[seq_id] = checkpoint_ok
            mutation_detected_by_seq_id[seq_id] = not checkpoint_ok

        accept_len_by_proposal_id: dict[int, int] = {}
        reject_position_by_proposal_id: dict[int, int] = {}
        result_by_proposal_id: dict[int, str] = {}
        full_accept_proposal_ids: list[int] = []
        rejected_proposal_ids: list[int] = []
        partial_accept_proposal_ids: list[int] = []
        executed_proposal_count = 0
        for proposal in executed_proposals:
            proposal_id = int(proposal.proposal_id)
            seq_id = int(proposal.seq_id)
            accept_len = int(results["accepted_len_by_seq_id"].get(seq_id, 0))
            reject_position = int(results["reject_position_by_seq_id"].get(seq_id, -1))
            accept_len_by_proposal_id[proposal_id] = accept_len
            reject_position_by_proposal_id[proposal_id] = reject_position
            executed_proposal_count += 1
            if accept_len >= gamma:
                result_by_proposal_id[proposal_id] = "full_accept"
                full_accept_proposal_ids.append(proposal_id)
            elif accept_len <= 0:
                result_by_proposal_id[proposal_id] = "reject_at_first_token"
                rejected_proposal_ids.append(proposal_id)
            else:
                result_by_proposal_id[proposal_id] = "partial_accept"
                partial_accept_proposal_ids.append(proposal_id)

        for proposal_id in skipped_proposal_ids:
            result_by_proposal_id[int(proposal_id)] = "skipped_invalid"

        for proposal in executed_ready_proposals:
            self.dual_batch_manager.ready_eager_proposals.mark_verify_dry_run_executed(
                proposal.proposal_id,
                current_step_id,
            )

        trace_record["enable_eager_verify_dry_run"] = True
        trace_record["eager_verify_dry_run_enabled"] = True
        trace_record["eager_verify_dry_run_source"] = "phase1h5e3_takeover_lane"
        trace_record["eager_verify_dry_run_step_id"] = current_step_id
        trace_record["eager_verify_dry_run_plan_id"] = int(plan.plan_id)
        trace_record["eager_verify_dry_run_candidate_proposal_ids"] = list(candidate_proposal_ids)
        trace_record["eager_verify_dry_run_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_verify_dry_run_skipped_proposal_ids"] = list(skipped_proposal_ids)
        trace_record["eager_verify_dry_run_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(skip_reason_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_executed_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in executed_proposals
        ]
        trace_record["eager_verify_dry_run_executed_seq_ids"] = [
            int(seq.seq_id) for seq in executed_seqs
        ]
        trace_record["eager_verify_dry_run_seq_id_by_proposal_id"] = {
            str(proposal_id): int(seq_id)
            for proposal_id, seq_id in sorted(seq_id_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_proposal_len_by_proposal_id"] = {
            str(proposal_id): int(value)
            for proposal_id, value in sorted(proposal_len_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_to_verify_len_by_proposal_id"] = {
            str(proposal_id): int(value)
            for proposal_id, value in sorted(to_verify_len_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_base_len_by_proposal_id"] = {
            str(proposal_id): int(value)
            for proposal_id, value in sorted(base_len_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_takeover_step_by_proposal_id"] = {
            str(proposal_id): int(value)
            for proposal_id, value in sorted(takeover_step_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value)
            for proposal_id, value in sorted(accept_len_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_result_by_proposal_id"] = {
            str(proposal_id): result
            for proposal_id, result in sorted(result_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_reject_position_by_proposal_id"] = {
            str(proposal_id): int(value)
            for proposal_id, value in sorted(reject_position_by_proposal_id.items())
        }
        trace_record["eager_verify_dry_run_full_accept_proposal_ids"] = list(full_accept_proposal_ids)
        trace_record["eager_verify_dry_run_rejected_proposal_ids"] = list(rejected_proposal_ids)
        trace_record["eager_verify_dry_run_partial_accept_proposal_ids"] = list(partial_accept_proposal_ids)
        trace_record["eager_verify_dry_run_mutation_detected"] = any(
            bool(value) for value in mutation_detected_by_seq_id.values()
        )
        trace_record["eager_verify_dry_run_checkpoint_failed"] = any(
            not bool(value) for value in checkpoint_ok_by_seq_id.values()
        )
        trace_record["eager_verify_dry_run_sequence_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(len_before_by_seq_id.items())
        }
        trace_record["eager_verify_dry_run_sequence_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(len_after_by_seq_id.items())
        }
        trace_record["eager_verify_dry_run_executed_proposal_count"] = executed_proposal_count
        trace_record["eager_verify_dry_run_skipped_proposal_count"] = len(skipped_proposal_ids)

        trace_record["eager_verify_candidate_proposal_ids"] = list(candidate_proposal_ids)
        trace_record["eager_verify_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_verify_skipped_proposal_ids"] = list(skipped_proposal_ids)
        trace_record["eager_verify_skip_reason_by_proposal_id"] = dict(
            trace_record["eager_verify_dry_run_skip_reason_by_proposal_id"]
        )
        trace_record["eager_verify_executed_proposal_ids"] = list(
            trace_record["eager_verify_dry_run_executed_proposal_ids"]
        )
        trace_record["eager_verify_executed_seq_ids"] = list(
            trace_record["eager_verify_dry_run_executed_seq_ids"]
        )
        trace_record["eager_verify_accepted_len_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["accepted_len_by_seq_id"].items()
        }
        trace_record["eager_verify_full_accept_by_seq_id"] = {
            str(seq_id): bool(value)
            for seq_id, value in results["full_accept_by_seq_id"].items()
        }
        trace_record["eager_verify_reject_position_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["reject_position_by_seq_id"].items()
        }
        trace_record["eager_verify_invalidated_len_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["invalidated_len_by_seq_id"].items()
        }
        trace_record["eager_verify_revised_token_by_seq_id"] = {
            str(seq_id): int(value)
            for seq_id, value in results["revised_token_by_seq_id"].items()
        }
        trace_record["eager_verify_base_len_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_verify_current_len_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_verify_base_match_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in base_match_by_seq_id.items()
        }
        trace_record["eager_verify_seq_pre_verify_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in seq_pre_verify_by_seq_id.items()
        }
        trace_record["eager_verify_seq_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_status_before_by_seq_id.items()
        }
        trace_record["eager_verify_seq_status_after_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_status_after_by_seq_id.items()
        }
        trace_record["eager_verify_mutation_detected_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in mutation_detected_by_seq_id.items()
        }
        trace_record["eager_verify_checkpoint_ok_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in checkpoint_ok_by_seq_id.items()
        }

        trace_record["eager_tokens_verify_dry_run"] = sum(
            int(proposal.proposal_len) for proposal in executed_proposals
        )
        trace_record["eager_tokens_verify_dry_run_full_accept"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if int(proposal.proposal_id) in set(full_accept_proposal_ids)
        )
        trace_record["eager_tokens_verify_dry_run_rejected"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if int(proposal.proposal_id) in set(rejected_proposal_ids)
        )
        trace_record["eager_tokens_verify_dry_run_partial_accept"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if int(proposal.proposal_id) in set(partial_accept_proposal_ids)
        )

    def _run_takeover_eager_apply_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        plan_context: dict[str, set[int]],
    ) -> None:
        verify_executed_ids = [
            int(proposal_id)
            for proposal_id in trace_record.get("eager_verify_dry_run_executed_proposal_ids", [])
        ]
        if not verify_executed_ids:
            return
        verify_executed_seq_ids = [
            int(seq_id)
            for seq_id in trace_record.get("eager_verify_dry_run_executed_seq_ids", [])
        ]
        seq_id_by_proposal = {
            int(proposal_id): int(seq_id)
            for proposal_id, seq_id in trace_record.get(
                "eager_verify_dry_run_seq_id_by_proposal_id",
                {},
            ).items()
        }
        verify_result_by_proposal = {
            int(proposal_id): str(result)
            for proposal_id, result in trace_record.get(
                "eager_verify_dry_run_result_by_proposal_id",
                {},
            ).items()
        }
        accept_len_by_proposal = {
            int(proposal_id): int(accept_len)
            for proposal_id, accept_len in trace_record.get(
                "eager_verify_dry_run_accept_len_by_proposal_id",
                {},
            ).items()
        }
        proposal_len_by_proposal = {
            int(proposal_id): int(proposal_len)
            for proposal_id, proposal_len in trace_record.get(
                "eager_verify_dry_run_proposal_len_by_proposal_id",
                {},
            ).items()
        }

        candidate_ids = list(verify_executed_ids)
        candidate_seq_ids = [
            int(seq_id_by_proposal.get(proposal_id, -1))
            for proposal_id in candidate_ids
        ]
        seq_by_id = self._local_sequence_by_id()
        current_step_id = int(plan.step_id if plan.step_id is not None else self.dual_batch_manager.step_id)
        gamma = int(self.gamma)

        executed_ids: list[int] = []
        executed_seq_ids: list[int] = []
        skipped_ids: list[int] = []
        skip_reason_by_proposal_id: dict[int, str] = {}
        action_by_proposal_id: dict[int, str] = {}
        action_by_seq_id: dict[int, str] = {}
        full_accept_ids: list[int] = []
        discarded_ids: list[int] = []
        append_tokens_by_proposal_id: dict[int, int] = {}
        discarded_tokens_by_proposal_id: dict[int, int] = {}
        rollback_ok_by_proposal_id: dict[int, bool] = {}
        mutation_detected_by_proposal_id: dict[int, bool] = {}
        checkpoint_failed_by_proposal_id: dict[int, bool] = {}

        accepted_len_trace_by_seq: dict[int, int] = {}
        full_accept_trace_by_seq: dict[int, bool] = {}
        base_len_by_seq: dict[int, int] = {}
        len_before_by_seq: dict[int, int] = {}
        len_after_apply_by_seq: dict[int, int] = {}
        len_after_rollback_by_seq: dict[int, int] = {}
        pre_verify_before_by_seq: dict[int, bool] = {}
        pre_verify_after_apply_by_seq: dict[int, bool] = {}
        pre_verify_after_rollback_by_seq: dict[int, bool] = {}
        status_before_by_seq: dict[int, str] = {}
        status_after_apply_by_seq: dict[int, str] = {}
        status_after_rollback_by_seq: dict[int, str] = {}
        checkpoint_ok_by_seq: dict[int, bool] = {}
        rollback_ok_by_seq: dict[int, bool] = {}
        mutation_remaining_by_seq: dict[int, bool] = {}
        appended_token_count_by_seq: dict[int, int] = {}

        for proposal_id in candidate_ids:
            ready_proposal = self.dual_batch_manager.ready_eager_proposals.by_id(proposal_id)
            seq_id = int(seq_id_by_proposal.get(proposal_id, -1))
            seq = seq_by_id.get(seq_id)
            proposal_len = int(proposal_len_by_proposal.get(proposal_id, -1))
            accept_len = int(accept_len_by_proposal.get(proposal_id, -1))
            verify_result = verify_result_by_proposal.get(proposal_id, "")
            current_len = -1 if seq is None else int(len(seq))
            base_len = -1 if ready_proposal is None else int(ready_proposal.base_len)

            base_len_by_seq[seq_id] = base_len
            len_before_by_seq[seq_id] = current_len
            pre_verify_before_by_seq[seq_id] = (
                True if seq is None else bool(getattr(seq, "pre_verify", True))
            )
            status_before_by_seq[seq_id] = self._sequence_status_name(seq)
            accepted_len_trace_by_seq[seq_id] = accept_len
            full_accept = verify_result == "full_accept" and accept_len == gamma
            full_accept_trace_by_seq[seq_id] = bool(full_accept)

            reason = None
            if proposal_id not in verify_executed_ids:
                reason = "not_verify_executed"
            elif ready_proposal is None:
                reason = "ready_proposal_not_found"
            elif seq_id < 0 or int(ready_proposal.seq_id) != seq_id:
                reason = "routed_seq_mismatch"
            elif verify_result not in {
                "full_accept",
                "partial_accept",
                "reject_at_first_token",
                "skipped_invalid",
            }:
                reason = "missing_verify_result"
            elif proposal_len != gamma or int(ready_proposal.proposal_len) != gamma:
                reason = "invalid_proposal_len"
            elif len(ready_proposal.proposal_token_ids) != gamma:
                reason = "invalid_proposal_len"
            elif len(ready_proposal.to_be_verified_token_ids) != gamma:
                reason = "invalid_to_verify_len"
            elif seq is None:
                reason = "seq_not_found"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished_before_apply_dry_run"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "span_invalidated_before_apply_dry_run"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_pre_verify_before_apply_dry_run"
            elif current_len != base_len:
                reason = "base_mismatch_before_apply_dry_run"
            elif bool(ready_proposal.base_pre_verify):
                reason = "seq_pre_verify_before_apply_dry_run"
            elif verify_result == "full_accept" and accept_len != gamma:
                reason = "missing_verify_result"
            elif verify_result == "partial_accept" and not (0 < accept_len < gamma):
                reason = "missing_verify_result"
            elif verify_result == "reject_at_first_token" and accept_len != 0:
                reason = "missing_verify_result"

            if reason is not None or verify_result == "skipped_invalid":
                skipped_ids.append(proposal_id)
                skip_reason_by_proposal_id[proposal_id] = reason or "skipped_invalid"
                action = "skipped_invalid_no_mutation"
                action_by_proposal_id[proposal_id] = action
                if seq_id >= 0:
                    action_by_seq_id[seq_id] = action
                discarded_ids.append(proposal_id)
                append_tokens_by_proposal_id[proposal_id] = 0
                discarded_tokens_by_proposal_id[proposal_id] = max(0, proposal_len)
                rollback_ok_by_proposal_id[proposal_id] = True
                mutation_detected_by_proposal_id[proposal_id] = False
                checkpoint_failed_by_proposal_id[proposal_id] = False
                continue

            checkpoint = self._make_eager_apply_checkpoint(seq)
            checkpoint_ok_before = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
            appended_count = 0
            if verify_result == "full_accept":
                for token_id in ready_proposal.proposal_token_ids:
                    seq.append_token(int(token_id))
                seq.pre_verify = False
                appended_count = len(ready_proposal.proposal_token_ids)
                action = "append_full_accept_then_rollback"
                full_accept_ids.append(proposal_id)
            elif verify_result == "partial_accept":
                action = "discard_partial_no_mutation"
                discarded_ids.append(proposal_id)
            else:
                action = "discard_reject_no_mutation"
                discarded_ids.append(proposal_id)

            len_after_apply_by_seq[seq_id] = int(len(seq))
            pre_verify_after_apply_by_seq[seq_id] = bool(seq.pre_verify)
            status_after_apply_by_seq[seq_id] = self._sequence_status_name(seq)
            if appended_count > 0:
                self.scheduler.rollback(seq, appended_count)
            self._restore_eager_apply_checkpoint(seq, checkpoint)
            rollback_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
            mutation_remaining = not rollback_ok
            len_after_rollback_by_seq[seq_id] = int(len(seq))
            pre_verify_after_rollback_by_seq[seq_id] = bool(seq.pre_verify)
            status_after_rollback_by_seq[seq_id] = self._sequence_status_name(seq)
            checkpoint_ok_by_seq[seq_id] = bool(checkpoint_ok_before)
            rollback_ok_by_seq[seq_id] = bool(rollback_ok)
            mutation_remaining_by_seq[seq_id] = bool(mutation_remaining)
            appended_token_count_by_seq[seq_id] = int(appended_count)
            rollback_ok_by_proposal_id[proposal_id] = bool(rollback_ok)
            mutation_detected_by_proposal_id[proposal_id] = bool(mutation_remaining)
            checkpoint_failed_by_proposal_id[proposal_id] = not bool(checkpoint_ok_before)
            append_tokens_by_proposal_id[proposal_id] = int(appended_count)
            discarded_tokens_by_proposal_id[proposal_id] = 0 if appended_count else max(0, proposal_len)
            action_by_proposal_id[proposal_id] = action
            action_by_seq_id[seq_id] = action
            executed_ids.append(proposal_id)
            executed_seq_ids.append(seq_id)

        total_candidate_tokens = sum(
            max(0, int(proposal_len_by_proposal.get(proposal_id, 0)))
            for proposal_id in candidate_ids
        )
        full_accept_tokens = sum(
            max(0, int(proposal_len_by_proposal.get(proposal_id, 0)))
            for proposal_id in full_accept_ids
        )
        discarded_tokens = sum(int(value) for value in discarded_tokens_by_proposal_id.values())

        trace_record["enable_eager_apply_dry_run"] = True
        trace_record["eager_apply_dry_run_enabled"] = True
        trace_record["eager_apply_dry_run_source"] = "phase1h5e3_takeover_lane"
        trace_record["eager_apply_dry_run_step_id"] = current_step_id
        trace_record["eager_apply_dry_run_plan_id"] = int(plan.plan_id)
        trace_record["eager_apply_dry_run_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["eager_apply_dry_run_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_apply_dry_run_executed_proposal_ids"] = list(executed_ids)
        trace_record["eager_apply_dry_run_executed_seq_ids"] = list(executed_seq_ids)
        trace_record["eager_apply_dry_run_skipped_proposal_ids"] = list(skipped_ids)
        trace_record["eager_apply_dry_run_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(skip_reason_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_from_verify_proposal_ids"] = list(verify_executed_ids)
        trace_record["eager_apply_dry_run_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(verify_result_by_proposal.items())
        }
        trace_record["eager_apply_dry_run_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(accept_len_by_proposal.items())
        }
        trace_record["eager_apply_dry_run_seq_id_by_proposal_id"] = {
            str(proposal_id): int(seq_id) for proposal_id, seq_id in sorted(seq_id_by_proposal.items())
        }
        trace_record["eager_apply_dry_run_proposal_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(proposal_len_by_proposal.items())
        }
        trace_record["eager_apply_dry_run_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(action_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_full_accept_proposal_ids"] = list(full_accept_ids)
        trace_record["eager_apply_dry_run_discarded_proposal_ids"] = list(discarded_ids)
        trace_record["eager_apply_dry_run_append_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(append_tokens_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_discarded_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(discarded_tokens_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_rollback_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(rollback_ok_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_mutation_detected_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(mutation_detected_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_checkpoint_failed_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(checkpoint_failed_by_proposal_id.items())
        }
        trace_record["eager_apply_dry_run_sequence_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(len_before_by_seq.items())
        }
        trace_record["eager_apply_dry_run_sequence_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(len_after_rollback_by_seq.items())
        }
        trace_record["eager_apply_dry_run_pre_verify_before_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(pre_verify_before_by_seq.items())
        }
        trace_record["eager_apply_dry_run_pre_verify_after_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(pre_verify_after_rollback_by_seq.items())
        }
        trace_record["eager_apply_dry_run_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in sorted(status_before_by_seq.items())
        }
        trace_record["eager_apply_dry_run_status_after_by_seq_id"] = {
            str(seq_id): value for seq_id, value in sorted(status_after_rollback_by_seq.items())
        }
        trace_record["eager_apply_dry_run_executed_proposal_count"] = len(executed_ids)
        trace_record["eager_apply_dry_run_skipped_proposal_count"] = len(skipped_ids)

        trace_record["eager_apply_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["eager_apply_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_apply_executed_proposal_ids"] = list(executed_ids)
        trace_record["eager_apply_executed_seq_ids"] = list(executed_seq_ids)
        trace_record["eager_apply_skipped_proposal_ids"] = list(skipped_ids)
        trace_record["eager_apply_skip_reason_by_proposal_id"] = dict(
            trace_record["eager_apply_dry_run_skip_reason_by_proposal_id"]
        )
        trace_record["eager_apply_action_by_seq_id"] = {
            str(seq_id): action for seq_id, action in action_by_seq_id.items()
        }
        trace_record["eager_apply_accepted_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in accepted_len_trace_by_seq.items()
        }
        trace_record["eager_apply_full_accept_by_seq_id"] = {
            str(seq_id): value for seq_id, value in full_accept_trace_by_seq.items()
        }
        trace_record["eager_apply_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq.items()
        }
        trace_record["eager_apply_current_len_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_before_by_seq.items()
        }
        trace_record["eager_apply_current_len_after_simulated_apply_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_after_apply_by_seq.items()
        }
        trace_record["eager_apply_current_len_after_rollback_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_after_rollback_by_seq.items()
        }
        trace_record["eager_apply_pre_verify_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_before_by_seq.items()
        }
        trace_record["eager_apply_pre_verify_after_simulated_apply_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_after_apply_by_seq.items()
        }
        trace_record["eager_apply_pre_verify_after_rollback_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_after_rollback_by_seq.items()
        }
        trace_record["eager_apply_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_before_by_seq.items()
        }
        trace_record["eager_apply_status_after_simulated_apply_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_after_apply_by_seq.items()
        }
        trace_record["eager_apply_status_after_rollback_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_after_rollback_by_seq.items()
        }
        trace_record["eager_apply_checkpoint_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in checkpoint_ok_by_seq.items()
        }
        trace_record["eager_apply_rollback_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in rollback_ok_by_seq.items()
        }
        trace_record["eager_apply_mutation_remaining_by_seq_id"] = {
            str(seq_id): value for seq_id, value in mutation_remaining_by_seq.items()
        }
        trace_record["eager_apply_dry_run_appended_token_count_by_seq_id"] = {
            str(seq_id): value for seq_id, value in appended_token_count_by_seq.items()
        }
        trace_record["eager_tokens_apply_dry_run"] = int(total_candidate_tokens)
        trace_record["eager_tokens_apply_dry_run_full_accept"] = int(full_accept_tokens)
        trace_record["eager_tokens_apply_dry_run_discarded"] = int(discarded_tokens)
        trace_record["eager_apply_dry_run_append_tokens"] = sum(
            int(value) for value in append_tokens_by_proposal_id.values()
        )
        trace_record["eager_apply_dry_run_rollback_failure_count"] = sum(
            1 for value in rollback_ok_by_proposal_id.values() if not bool(value)
        )

    def _make_eager_apply_checkpoint(self, seq: Sequence) -> dict:
        checkpoint = make_sequence_checkpoint(seq)
        checkpoint.update(
            {
                "token_ids": list(seq.token_ids),
                "last_token": int(seq.last_token),
                "status_obj": seq.status,
                "num_acc_tokens": list(seq.num_acc_tokens),
                "first_token_ts": seq.first_token_ts,
            }
        )
        return checkpoint

    def _sequence_matches_eager_apply_checkpoint(self, seq: Sequence, checkpoint: dict) -> bool:
        return (
            int(seq.seq_id) == int(checkpoint["seq_id"])
            and list(seq.token_ids) == list(checkpoint["token_ids"])
            and int(len(seq)) == int(checkpoint["len"])
            and bool(seq.pre_verify) == bool(checkpoint["pre_verify"])
            and int(seq.num_completion_tokens) == int(checkpoint["num_completion_tokens"])
            and int(seq.cur_acc_tokens) == int(checkpoint["cur_acc_tokens"])
            and self._sequence_status_name(seq) == str(checkpoint["status"])
            and seq.home_batch_id == checkpoint["home_batch_id"]
            and list(seq.num_acc_tokens) == list(checkpoint["num_acc_tokens"])
            and seq.first_token_ts == checkpoint["first_token_ts"]
            and int(seq.last_token) == int(checkpoint["last_token"])
        )

    def _restore_eager_apply_checkpoint(self, seq: Sequence, checkpoint: dict) -> None:
        seq.token_ids = list(checkpoint["token_ids"])
        seq.num_tokens = int(checkpoint["len"])
        seq.last_token = int(checkpoint["last_token"])
        seq.pre_verify = bool(checkpoint["pre_verify"])
        seq.cur_acc_tokens = int(checkpoint["cur_acc_tokens"])
        seq.status = checkpoint["status_obj"]
        seq.home_batch_id = checkpoint["home_batch_id"]
        seq.num_acc_tokens = list(checkpoint["num_acc_tokens"])
        seq.first_token_ts = checkpoint["first_token_ts"]

    def _rollback_eager_apply_dry_run(
        self,
        seq: Sequence,
        checkpoint: dict,
        appended_count: int,
    ) -> tuple[bool, bool]:
        if appended_count > 0:
            self.scheduler.rollback(seq, int(appended_count))
        rollback_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
        if not rollback_ok:
            self._restore_eager_apply_checkpoint(seq, checkpoint)
        mutation_remaining = not self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
        return rollback_ok, mutation_remaining

    def _run_eager_apply_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        scheduled_proposals: list[EagerProposal],
        plan_context: dict[str, set[int]],
    ) -> None:
        seq_by_id = self._local_sequence_by_id()
        gamma = int(self.gamma)
        verify_executed_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_verify_executed_proposal_ids", [])
        }
        accepted_len_by_seq = trace_record.get("eager_verify_accepted_len_by_seq_id", {})
        full_accept_by_seq = trace_record.get("eager_verify_full_accept_by_seq_id", {})
        candidate_proposals = [
            proposal
            for proposal in scheduled_proposals
            if int(proposal.proposal_id) in verify_executed_ids
        ]

        executed_proposals: list[EagerProposal] = []
        executed_seq_ids: list[int] = []
        skipped_proposal_ids: list[int] = []
        skip_reason_by_proposal_id: dict[int, str] = {}
        action_by_seq_id: dict[int, str] = {}
        accepted_len_trace: dict[int, int] = {}
        full_accept_trace: dict[int, bool] = {}
        base_len_by_seq_id: dict[int, int] = {}
        len_before_by_seq_id: dict[int, int] = {}
        len_after_apply_by_seq_id: dict[int, int] = {}
        len_after_rollback_by_seq_id: dict[int, int] = {}
        pre_verify_before_by_seq_id: dict[int, bool] = {}
        pre_verify_after_apply_by_seq_id: dict[int, bool] = {}
        pre_verify_after_rollback_by_seq_id: dict[int, bool] = {}
        status_before_by_seq_id: dict[int, str] = {}
        status_after_apply_by_seq_id: dict[int, str] = {}
        status_after_rollback_by_seq_id: dict[int, str] = {}
        checkpoint_ok_by_seq_id: dict[int, bool] = {}
        rollback_ok_by_seq_id: dict[int, bool] = {}
        mutation_remaining_by_seq_id: dict[int, bool] = {}
        appended_token_count_by_seq_id: dict[int, int] = {}

        for proposal in candidate_proposals:
            proposal_id = int(proposal.proposal_id)
            seq_id = int(proposal.seq_id)
            seq = seq_by_id.get(seq_id)
            current_len = -1 if seq is None else int(len(seq))
            base_len_by_seq_id[seq_id] = int(proposal.base_len)
            len_before_by_seq_id[seq_id] = current_len
            pre_verify_before_by_seq_id[seq_id] = bool(getattr(seq, "pre_verify", True)) if seq is not None else True
            status_before_by_seq_id[seq_id] = self._sequence_status_name(seq)

            accepted_value = accepted_len_by_seq.get(str(seq_id), accepted_len_by_seq.get(seq_id))
            full_accept_value = full_accept_by_seq.get(str(seq_id), full_accept_by_seq.get(seq_id))
            reason = None
            if accepted_value is None or full_accept_value is None:
                reason = "missing_verify_result"
            else:
                accepted_len = int(accepted_value)
                full_accept = bool(full_accept_value)
                if not (0 <= accepted_len <= gamma) or full_accept != (accepted_len == gamma):
                    reason = "missing_verify_result"
            if reason is None and int(proposal.proposal_len) != gamma:
                reason = "invalid_proposal_len"
            elif reason is None and len(proposal.proposal_token_ids) != gamma:
                reason = "invalid_proposal_len"
            elif reason is None and len(proposal.to_be_verified_token_ids) != gamma:
                reason = "invalid_to_verify_len"
            elif reason is None and seq is None:
                reason = "missing_verify_result"
            elif reason is None and self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished_before_apply_dry_run"
            elif reason is None and self.is_speculative_span_invalidated(seq, plan_context):
                reason = "span_invalidated_before_apply_dry_run"
            elif reason is None and bool(getattr(seq, "pre_verify", True)):
                reason = "seq_pre_verify_before_apply_dry_run"
            elif reason is None and current_len != int(proposal.base_len):
                reason = "base_mismatch_before_apply_dry_run"
            elif reason is None and bool(proposal.base_pre_verify):
                reason = "seq_pre_verify_before_apply_dry_run"

            if reason is not None:
                skipped_proposal_ids.append(proposal_id)
                skip_reason_by_proposal_id[proposal_id] = reason
                continue

            accepted_len = int(accepted_value)
            full_accept = bool(full_accept_value)
            checkpoint = self._make_eager_apply_checkpoint(seq)
            checkpoint_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
            appended_count = 0
            if full_accept:
                for token_id in proposal.proposal_token_ids:
                    seq.append_token(int(token_id))
                seq.pre_verify = False
                appended_count = len(proposal.proposal_token_ids)
                action = "append_full_accept_then_rollback"
            else:
                action = "discard_partial_no_mutation" if accepted_len > 0 else "discard_reject_no_mutation"

            len_after_apply_by_seq_id[seq_id] = int(len(seq))
            pre_verify_after_apply_by_seq_id[seq_id] = bool(seq.pre_verify)
            status_after_apply_by_seq_id[seq_id] = self._sequence_status_name(seq)
            rollback_ok, mutation_remaining = self._rollback_eager_apply_dry_run(
                seq,
                checkpoint,
                appended_count,
            )
            len_after_rollback_by_seq_id[seq_id] = int(len(seq))
            pre_verify_after_rollback_by_seq_id[seq_id] = bool(seq.pre_verify)
            status_after_rollback_by_seq_id[seq_id] = self._sequence_status_name(seq)
            checkpoint_ok_by_seq_id[seq_id] = bool(checkpoint_ok)
            rollback_ok_by_seq_id[seq_id] = bool(rollback_ok)
            mutation_remaining_by_seq_id[seq_id] = bool(mutation_remaining)
            appended_token_count_by_seq_id[seq_id] = int(appended_count)
            action_by_seq_id[seq_id] = action
            accepted_len_trace[seq_id] = accepted_len
            full_accept_trace[seq_id] = full_accept
            executed_proposals.append(proposal)
            executed_seq_ids.append(seq_id)

        trace_record["enable_eager_apply_dry_run"] = True
        trace_record["eager_apply_dry_run_enabled"] = True
        trace_record["eager_apply_dry_run_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_apply_dry_run_plan_id"] = int(plan.plan_id)
        trace_record["eager_apply_candidate_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in candidate_proposals
        ]
        trace_record["eager_apply_candidate_seq_ids"] = [
            int(proposal.seq_id) for proposal in candidate_proposals
        ]
        trace_record["eager_apply_executed_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in executed_proposals
        ]
        trace_record["eager_apply_executed_seq_ids"] = [int(seq_id) for seq_id in executed_seq_ids]
        trace_record["eager_apply_skipped_proposal_ids"] = [int(proposal_id) for proposal_id in skipped_proposal_ids]
        trace_record["eager_apply_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in skip_reason_by_proposal_id.items()
        }
        trace_record["eager_apply_action_by_seq_id"] = {
            str(seq_id): action for seq_id, action in action_by_seq_id.items()
        }
        trace_record["eager_apply_accepted_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in accepted_len_trace.items()
        }
        trace_record["eager_apply_full_accept_by_seq_id"] = {
            str(seq_id): value for seq_id, value in full_accept_trace.items()
        }
        trace_record["eager_apply_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_apply_current_len_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_before_by_seq_id.items()
        }
        trace_record["eager_apply_current_len_after_simulated_apply_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_after_apply_by_seq_id.items()
        }
        trace_record["eager_apply_current_len_after_rollback_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_after_rollback_by_seq_id.items()
        }
        trace_record["eager_apply_pre_verify_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_before_by_seq_id.items()
        }
        trace_record["eager_apply_pre_verify_after_simulated_apply_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_after_apply_by_seq_id.items()
        }
        trace_record["eager_apply_pre_verify_after_rollback_by_seq_id"] = {
            str(seq_id): value for seq_id, value in pre_verify_after_rollback_by_seq_id.items()
        }
        trace_record["eager_apply_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_before_by_seq_id.items()
        }
        trace_record["eager_apply_status_after_simulated_apply_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_after_apply_by_seq_id.items()
        }
        trace_record["eager_apply_status_after_rollback_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_after_rollback_by_seq_id.items()
        }
        trace_record["eager_apply_checkpoint_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in checkpoint_ok_by_seq_id.items()
        }
        trace_record["eager_apply_rollback_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in rollback_ok_by_seq_id.items()
        }
        trace_record["eager_apply_mutation_remaining_by_seq_id"] = {
            str(seq_id): value for seq_id, value in mutation_remaining_by_seq_id.items()
        }
        trace_record["eager_apply_dry_run_appended_token_count_by_seq_id"] = {
            str(seq_id): value for seq_id, value in appended_token_count_by_seq_id.items()
        }
        trace_record["eager_tokens_apply_dry_run"] = sum(
            int(proposal.proposal_len) for proposal in executed_proposals
        )
        trace_record["eager_tokens_apply_dry_run_full_accept"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if bool(full_accept_trace.get(int(proposal.seq_id), False))
        )
        trace_record["eager_tokens_apply_dry_run_discarded"] = sum(
            int(proposal.proposal_len)
            for proposal in executed_proposals
            if not bool(full_accept_trace.get(int(proposal.seq_id), False))
        )
        trace_record["eager_apply_dry_run_append_tokens"] = sum(
            int(value) for value in appended_token_count_by_seq_id.values()
        )
        trace_record["eager_apply_dry_run_rollback_failure_count"] = sum(
            1 for value in rollback_ok_by_seq_id.values() if not bool(value)
        )
        if self._eager_sync_apply_dry_run_enabled():
            self._trace_target_sync_apply_from_apply_dry_run(plan, trace_record)

    def _trace_target_sync_apply_from_apply_dry_run(self, plan: StepPlan, trace_record: dict) -> None:
        mutation_by_seq = trace_record.get("eager_apply_mutation_remaining_by_seq_id", {})
        plan.eager_sync_apply_dry_run_enabled = True
        trace_record["enable_eager_sync_apply_dry_run"] = True
        trace_record["eager_sync_apply_dry_run_enabled"] = True
        trace_record["eager_sync_apply_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_sync_apply_plan_id"] = int(plan.plan_id)
        trace_record["eager_sync_apply_candidate_proposal_ids"] = list(
            trace_record.get("eager_apply_candidate_proposal_ids", [])
        )
        trace_record["eager_sync_apply_candidate_seq_ids"] = list(
            trace_record.get("eager_apply_candidate_seq_ids", [])
        )
        trace_record["eager_sync_apply_executed_proposal_ids"] = list(
            trace_record.get("eager_apply_executed_proposal_ids", [])
        )
        trace_record["eager_sync_apply_executed_seq_ids"] = list(
            trace_record.get("eager_apply_executed_seq_ids", [])
        )
        trace_record["eager_sync_apply_skipped_proposal_ids"] = list(
            trace_record.get("eager_apply_skipped_proposal_ids", [])
        )
        trace_record["eager_sync_apply_skip_reason_by_proposal_id"] = dict(
            trace_record.get("eager_apply_skip_reason_by_proposal_id", {})
        )
        trace_record["eager_sync_apply_action_by_seq_id"] = dict(
            trace_record.get("eager_apply_action_by_seq_id", {})
        )
        trace_record["eager_sync_apply_accepted_len_by_seq_id"] = dict(
            trace_record.get("eager_apply_accepted_len_by_seq_id", {})
        )
        trace_record["eager_sync_apply_full_accept_by_seq_id"] = dict(
            trace_record.get("eager_apply_full_accept_by_seq_id", {})
        )
        trace_record["target_sync_apply_checkpoint_ok_by_seq_id"] = dict(
            trace_record.get("eager_apply_checkpoint_ok_by_seq_id", {})
        )
        trace_record["target_sync_apply_rollback_ok_by_seq_id"] = dict(
            trace_record.get("eager_apply_rollback_ok_by_seq_id", {})
        )
        trace_record["target_sync_apply_mutation_remaining_by_seq_id"] = dict(mutation_by_seq)
        trace_record["target_sync_apply_len_before_by_seq_id"] = dict(
            trace_record.get("eager_apply_current_len_before_by_seq_id", {})
        )
        trace_record["target_sync_apply_len_after_simulated_by_seq_id"] = dict(
            trace_record.get("eager_apply_current_len_after_simulated_apply_by_seq_id", {})
        )
        trace_record["target_sync_apply_len_after_restore_by_seq_id"] = dict(
            trace_record.get("eager_apply_current_len_after_rollback_by_seq_id", {})
        )
        trace_record["target_sync_apply_status_before_by_seq_id"] = dict(
            trace_record.get("eager_apply_status_before_by_seq_id", {})
        )
        trace_record["target_sync_apply_status_after_restore_by_seq_id"] = dict(
            trace_record.get("eager_apply_status_after_rollback_by_seq_id", {})
        )
        total_tokens = int(trace_record.get("eager_tokens_apply_dry_run") or 0)
        trace_record["eager_tokens_sync_apply_dry_run"] = total_tokens
        trace_record["eager_tokens_sync_apply_dry_run_full_accept"] = int(
            trace_record.get("eager_tokens_apply_dry_run_full_accept") or 0
        )
        trace_record["eager_tokens_sync_apply_dry_run_discarded"] = int(
            trace_record.get("eager_tokens_apply_dry_run_discarded") or 0
        )
        trace_record["eager_tokens_sync_apply_dry_run_target_side"] = total_tokens
        trace_record["eager_tokens_sync_apply_dry_run_draft_side"] = 0
        trace_record["eager_sync_apply_target_mutation_remaining_count"] = sum(
            1 for value in mutation_by_seq.values() if bool(value)
        )
        trace_record["eager_sync_apply_draft_mutation_remaining_count"] = 0

    def _set_sequence_to_checkpoint_prefix(self, seq: Sequence, checkpoint: dict, base_len: int) -> None:
        token_ids = list(checkpoint["token_ids"])[: int(base_len)]
        if not token_ids:
            raise ValueError("cannot restore sequence prefix with no tokens")
        seq.token_ids = token_ids
        seq.num_tokens = int(base_len)
        seq.last_token = int(token_ids[-1])
        seq.pre_verify = False

    def _run_draft_sync_apply_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        validated_results: list[dict],
        known_by_id: dict[int, EagerProposal],
        seq_by_id: dict[int, Sequence],
    ) -> None:
        gamma = int(self.gamma)
        plan_context = self._eager_transfer_plan_context(plan, trace_record)
        result_source = str(
            trace_record.get("eager_result_transfer_dry_run_source")
            or EAGER_LEGACY_RESULT_TRANSFER_SOURCE
        )
        transfer_validated_ids = {
            int(proposal_id)
            for proposal_id in (
                trace_record.get("eager_result_transfer_validated_proposal_ids")
                or trace_record.get("eager_result_validated_proposal_ids")
                or []
            )
        }
        validation_reason_by_proposal_id = (
            trace_record.get("eager_result_transfer_validation_reason_by_proposal_id")
            or trace_record.get("eager_result_validation_reason_by_proposal_id")
            or {}
        )
        original_draft_home = {
            int(seq_id)
            for seq_id in list(plan.draft_home_set) + list(trace_record.get("draft_home_set", []))
        }
        dry_run_exclusions = {
            int(seq_id)
            for seq_id in list(getattr(plan, "excluded_from_draft_home_for_eager_dry_run", []))
            + list(trace_record.get("excluded_from_draft_home_for_eager_dry_run", []))
        }

        candidate_ids: list[int] = []
        candidate_seq_ids: list[int] = []
        executed_ids: list[int] = []
        executed_seq_ids: list[int] = []
        skipped_ids: list[int] = []
        skip_reason_by_proposal_id: dict[int, str] = {}
        target_action_by_proposal_id: dict[int, str] = {}
        draft_action_by_proposal_id: dict[int, str] = {}
        target_result_by_proposal_id: dict[int, str] = {}
        draft_result_by_proposal_id: dict[int, str] = {}
        target_accept_len_by_proposal_id: dict[int, int] = {}
        draft_accept_len_by_proposal_id: dict[int, int] = {}
        action_match_by_proposal_id: dict[int, bool] = {}
        accept_len_match_by_proposal_id: dict[int, bool] = {}
        result_match_by_proposal_id: dict[int, bool] = {}
        append_tokens_by_proposal_id: dict[int, int] = {}
        discarded_tokens_by_proposal_id: dict[int, int] = {}
        rollback_ok_by_proposal_id: dict[int, bool] = {}
        mutation_detected_by_proposal_id: dict[int, bool] = {}
        checkpoint_failed_by_proposal_id: dict[int, bool] = {}
        consistent_proposal_ids: list[int] = []
        inconsistent_proposal_ids: list[int] = []
        missing_local_proposal_ids: list[int] = []
        duplicate_proposal_ids: list[int] = []
        action_by_seq_id: dict[int, str] = {}
        accepted_len_by_seq_id: dict[int, int] = {}
        full_accept_by_seq_id: dict[int, bool] = {}
        checkpoint_ok_by_seq_id: dict[int, bool] = {}
        rollback_ok_by_seq_id: dict[int, bool] = {}
        mutation_remaining_by_seq_id: dict[int, bool] = {}
        len_before_by_seq_id: dict[int, int] = {}
        base_len_by_seq_id: dict[int, int] = {}
        len_after_simulated_by_seq_id: dict[int, int] = {}
        len_after_restore_by_seq_id: dict[int, int] = {}
        status_before_by_seq_id: dict[int, str] = {}
        status_after_restore_by_seq_id: dict[int, str] = {}
        expected_conflict_by_seq_id: dict[int, bool] = {}
        original_draft_home_by_seq_id: dict[int, bool] = {}
        adjusted_exclusion_by_seq_id: dict[int, bool] = {}
        pre_verify_before_by_seq_id: dict[int, bool] = {}
        pre_verify_after_restore_by_seq_id: dict[int, bool] = {}
        seen_candidate_ids: set[int] = set()

        def expected_actions_for_result(verify_result: str, accepted_len: int) -> set[str]:
            if verify_result == "full_accept":
                return {"append_full_accept_then_rollback"}
            if verify_result == "partial_accept":
                return {"discard_partial_no_mutation"}
            if verify_result == "reject_at_first_token":
                actions = {"discard_reject_no_mutation"}
                if int(accepted_len) == 0:
                    actions.add("discard_partial_no_mutation")
                return actions
            if verify_result == "skipped_invalid":
                return {"skipped_invalid_no_mutation"}
            return set()

        def default_action_for_result(verify_result: str, accepted_len: int) -> str:
            actions = sorted(expected_actions_for_result(verify_result, accepted_len))
            if actions:
                if verify_result == "reject_at_first_token":
                    return "discard_reject_no_mutation"
                return actions[0]
            return "unknown"

        for result in validated_results:
            proposal_id = int(result["proposal_id"])
            seq_id = int(result["seq_id"])
            proposal = known_by_id.get(proposal_id)
            seq = seq_by_id.get(seq_id)
            candidate_ids.append(proposal_id)
            candidate_seq_ids.append(seq_id)
            validation_reason = str(
                self._trace_map_get(validation_reason_by_proposal_id, proposal_id, "ok")
            )
            current_len = -1 if seq is None else int(len(seq))
            base_len = int(result.get("base_len", -1))
            accepted_len = int(result.get("accepted_len", -1))
            verify_result = str(result.get("verify_result", "unknown"))
            full_accept = verify_result == "full_accept"
            target_action = str(result.get("apply_action", "unknown"))
            expected_actions = expected_actions_for_result(verify_result, accepted_len)
            draft_action = (
                target_action
                if target_action in expected_actions
                else default_action_for_result(verify_result, accepted_len)
            )
            proposal_len = int(result.get("proposal_len", -1))
            to_verify_len = int(result.get("to_verify_len", -1))
            append_token_count = int(result.get("append_token_count", 0))
            discarded_token_count = int(result.get("discarded_token_count", 0))
            original_draft_intersection = seq_id in original_draft_home
            adjusted_exclusion = seq_id in dry_run_exclusions or original_draft_intersection
            expected_conflict = current_len > base_len and (original_draft_intersection or adjusted_exclusion)
            len_before_by_seq_id[seq_id] = current_len
            base_len_by_seq_id[seq_id] = base_len
            pre_verify_before_by_seq_id[seq_id] = bool(getattr(seq, "pre_verify", True)) if seq is not None else True
            status_before_by_seq_id[seq_id] = self._sequence_status_name(seq)
            expected_conflict_by_seq_id[seq_id] = bool(expected_conflict)
            original_draft_home_by_seq_id[seq_id] = bool(original_draft_intersection)
            adjusted_exclusion_by_seq_id[seq_id] = bool(adjusted_exclusion)
            target_action_by_proposal_id[proposal_id] = target_action
            draft_action_by_proposal_id[proposal_id] = draft_action
            target_result_by_proposal_id[proposal_id] = verify_result
            draft_result_by_proposal_id[proposal_id] = verify_result
            target_accept_len_by_proposal_id[proposal_id] = accepted_len
            draft_accept_len_by_proposal_id[proposal_id] = accepted_len
            action_match_by_proposal_id[proposal_id] = target_action == draft_action
            accept_len_match_by_proposal_id[proposal_id] = True
            result_match_by_proposal_id[proposal_id] = True
            append_tokens_by_proposal_id[proposal_id] = 0
            discarded_tokens_by_proposal_id[proposal_id] = max(0, discarded_token_count)
            rollback_ok_by_proposal_id[proposal_id] = True
            mutation_detected_by_proposal_id[proposal_id] = False
            checkpoint_failed_by_proposal_id[proposal_id] = False

            reason = None
            if proposal_id in seen_candidate_ids:
                reason = "duplicate_result"
                duplicate_proposal_ids.append(proposal_id)
            elif transfer_validated_ids and proposal_id not in transfer_validated_ids:
                reason = "not_validated_result_transfer"
            elif validation_reason != "ok":
                reason = "not_validated_result_transfer"
            elif proposal is None:
                reason = "missing_proposal_id"
                missing_local_proposal_ids.append(proposal_id)
            elif seq is None:
                reason = "missing_local_seq"
                missing_local_proposal_ids.append(proposal_id)
            elif int(getattr(proposal, "seq_id", -1)) != seq_id:
                reason = "seq_id_mismatch"
            elif proposal_len != gamma or int(proposal.proposal_len) != gamma:
                reason = "invalid_proposal_len"
            elif to_verify_len != gamma or len(proposal.to_be_verified_token_ids) != gamma:
                reason = "invalid_to_verify_len"
            elif bool(result.get("base_pre_verify", True)) or bool(proposal.base_pre_verify):
                reason = "invalid_base_pre_verify"
            elif verify_result not in {
                "full_accept",
                "partial_accept",
                "reject_at_first_token",
                "skipped_invalid",
            }:
                reason = "invalid_result_metadata"
            elif not (0 <= accepted_len <= gamma) or full_accept != (accepted_len == gamma):
                reason = "invalid_result_metadata"
            elif verify_result == "partial_accept" and not (0 < accepted_len < gamma):
                reason = "invalid_result_metadata"
            elif verify_result == "reject_at_first_token" and accepted_len != 0:
                reason = "invalid_result_metadata"
            elif target_action not in expected_actions:
                reason = "target_action_mismatch"
            elif verify_result == "full_accept" and append_token_count != gamma:
                reason = "target_token_count_mismatch"
            elif verify_result != "full_accept" and append_token_count != 0:
                reason = "target_token_count_mismatch"
            elif verify_result != "full_accept" and discarded_token_count != gamma:
                reason = "target_token_count_mismatch"
            elif verify_result == "full_accept" and discarded_token_count != 0:
                reason = "target_token_count_mismatch"
            elif verify_result == "full_accept" and not bool(result.get("rollback_ok", False)):
                reason = "target_rollback_failed"
            elif bool(result.get("mutation_detected", True)):
                reason = "target_mutation_detected"
            elif bool(result.get("checkpoint_failed", True)):
                reason = "target_checkpoint_failed"
            elif verify_result == "full_accept" and len(proposal.proposal_token_ids) != gamma:
                reason = "missing_proposal_token_payload"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished_before_sync_apply_dry_run"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "span_invalidated_before_sync_apply_dry_run"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_pre_verify_before_sync_apply_dry_run"
            elif current_len < base_len:
                reason = "draft_base_not_reached"
            elif current_len > base_len and not expected_conflict:
                reason = "unexpected_draft_base_overshot"
            seen_candidate_ids.add(proposal_id)

            if reason is not None or verify_result == "skipped_invalid":
                if reason is None:
                    reason = "skipped_invalid"
                skipped_ids.append(proposal_id)
                skip_reason_by_proposal_id[proposal_id] = reason
                if reason not in {"skipped_invalid"}:
                    inconsistent_proposal_ids.append(proposal_id)
                else:
                    consistent_proposal_ids.append(proposal_id)
                discarded_tokens_by_proposal_id[proposal_id] = max(0, proposal_len)
                continue

            checkpoint = self._make_eager_apply_checkpoint(seq)
            checkpoint_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
            action = draft_action
            appended_count = 0
            if full_accept:
                if expected_conflict:
                    self._set_sequence_to_checkpoint_prefix(seq, checkpoint, base_len)
                for token_id in proposal.proposal_token_ids:
                    seq.append_token(int(token_id))
                seq.pre_verify = False
                appended_count = len(proposal.proposal_token_ids)
            len_after_simulated_by_seq_id[seq_id] = int(len(seq))
            self._restore_eager_apply_checkpoint(seq, checkpoint)
            rollback_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
            mutation_remaining = not rollback_ok
            len_after_restore_by_seq_id[seq_id] = int(len(seq))
            pre_verify_after_restore_by_seq_id[seq_id] = bool(getattr(seq, "pre_verify", True))
            status_after_restore_by_seq_id[seq_id] = self._sequence_status_name(seq)
            checkpoint_ok_by_seq_id[seq_id] = bool(checkpoint_ok)
            rollback_ok_by_seq_id[seq_id] = bool(rollback_ok)
            mutation_remaining_by_seq_id[seq_id] = bool(mutation_remaining)
            append_tokens_by_proposal_id[proposal_id] = int(appended_count)
            discarded_tokens_by_proposal_id[proposal_id] = 0 if appended_count else max(0, proposal_len)
            rollback_ok_by_proposal_id[proposal_id] = bool(rollback_ok)
            mutation_detected_by_proposal_id[proposal_id] = bool(mutation_remaining)
            checkpoint_failed_by_proposal_id[proposal_id] = not bool(checkpoint_ok)
            action_by_seq_id[seq_id] = action
            accepted_len_by_seq_id[seq_id] = accepted_len
            full_accept_by_seq_id[seq_id] = full_accept
            executed_ids.append(proposal_id)
            executed_seq_ids.append(seq_id)
            if (
                action_match_by_proposal_id[proposal_id]
                and accept_len_match_by_proposal_id[proposal_id]
                and result_match_by_proposal_id[proposal_id]
                and checkpoint_ok
                and rollback_ok
                and not mutation_remaining
            ):
                consistent_proposal_ids.append(proposal_id)
            else:
                inconsistent_proposal_ids.append(proposal_id)

        for proposal_id in set(candidate_ids):
            self._draft_sent_eager_proposals_by_id.pop(int(proposal_id), None)

        trace_record["enable_eager_sync_apply_dry_run"] = True
        trace_record["eager_sync_apply_dry_run_enabled"] = True
        trace_record["eager_sync_apply_dry_run_source"] = result_source
        trace_record["eager_sync_apply_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_sync_apply_plan_id"] = int(plan.plan_id)
        trace_record["eager_sync_apply_candidate_proposal_ids"] = candidate_ids
        trace_record["eager_sync_apply_candidate_seq_ids"] = candidate_seq_ids
        trace_record["eager_sync_apply_executed_proposal_ids"] = executed_ids
        trace_record["eager_sync_apply_executed_seq_ids"] = executed_seq_ids
        trace_record["eager_sync_apply_skipped_proposal_ids"] = skipped_ids
        trace_record["eager_sync_apply_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in skip_reason_by_proposal_id.items()
        }
        trace_record["eager_sync_apply_action_by_seq_id"] = {
            str(seq_id): action for seq_id, action in action_by_seq_id.items()
        }
        trace_record["eager_sync_apply_accepted_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in accepted_len_by_seq_id.items()
        }
        trace_record["eager_sync_apply_full_accept_by_seq_id"] = {
            str(seq_id): value for seq_id, value in full_accept_by_seq_id.items()
        }
        trace_record["eager_sync_apply_dry_run_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["eager_sync_apply_dry_run_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_sync_apply_dry_run_from_result_transfer_proposal_ids"] = [
            int(result["proposal_id"]) for result in validated_results
        ]
        trace_record["eager_sync_apply_dry_run_from_result_transfer_seq_ids"] = [
            int(result["seq_id"]) for result in validated_results
        ]
        trace_record["eager_sync_apply_dry_run_executed_proposal_ids"] = list(executed_ids)
        trace_record["eager_sync_apply_dry_run_executed_seq_ids"] = list(executed_seq_ids)
        trace_record["eager_sync_apply_dry_run_skipped_proposal_ids"] = list(skipped_ids)
        trace_record["eager_sync_apply_dry_run_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(skip_reason_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_target_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(target_action_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_draft_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(draft_action_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_target_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(target_result_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_draft_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(draft_result_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_target_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(target_accept_len_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_draft_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(draft_accept_len_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_action_match_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(action_match_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_accept_len_match_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(accept_len_match_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_result_match_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(result_match_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_append_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(append_tokens_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_discarded_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(discarded_tokens_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_rollback_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(rollback_ok_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_mutation_detected_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(mutation_detected_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_checkpoint_failed_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(checkpoint_failed_by_proposal_id.items())
        }
        trace_record["eager_sync_apply_dry_run_sequence_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(len_before_by_seq_id.items())
        }
        trace_record["eager_sync_apply_dry_run_sequence_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(len_after_restore_by_seq_id.items())
        }
        trace_record["eager_sync_apply_dry_run_pre_verify_before_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(pre_verify_before_by_seq_id.items())
        }
        trace_record["eager_sync_apply_dry_run_pre_verify_after_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(pre_verify_after_restore_by_seq_id.items())
        }
        trace_record["eager_sync_apply_dry_run_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in sorted(status_before_by_seq_id.items())
        }
        trace_record["eager_sync_apply_dry_run_status_after_by_seq_id"] = {
            str(seq_id): value for seq_id, value in sorted(status_after_restore_by_seq_id.items())
        }
        trace_record["eager_sync_apply_dry_run_consistent_proposal_ids"] = sorted(set(consistent_proposal_ids))
        trace_record["eager_sync_apply_dry_run_inconsistent_proposal_ids"] = sorted(set(inconsistent_proposal_ids))
        trace_record["eager_sync_apply_dry_run_missing_local_proposal_ids"] = sorted(set(missing_local_proposal_ids))
        trace_record["eager_sync_apply_dry_run_duplicate_proposal_ids"] = sorted(set(duplicate_proposal_ids))
        trace_record["draft_sync_apply_checkpoint_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in checkpoint_ok_by_seq_id.items()
        }
        trace_record["draft_sync_apply_rollback_ok_by_seq_id"] = {
            str(seq_id): value for seq_id, value in rollback_ok_by_seq_id.items()
        }
        trace_record["draft_sync_apply_mutation_remaining_by_seq_id"] = {
            str(seq_id): value for seq_id, value in mutation_remaining_by_seq_id.items()
        }
        trace_record["draft_sync_apply_len_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_before_by_seq_id.items()
        }
        trace_record["draft_sync_apply_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["draft_sync_apply_len_after_simulated_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_after_simulated_by_seq_id.items()
        }
        trace_record["draft_sync_apply_len_after_restore_by_seq_id"] = {
            str(seq_id): value for seq_id, value in len_after_restore_by_seq_id.items()
        }
        trace_record["draft_sync_apply_status_before_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_before_by_seq_id.items()
        }
        trace_record["draft_sync_apply_status_after_restore_by_seq_id"] = {
            str(seq_id): value for seq_id, value in status_after_restore_by_seq_id.items()
        }
        trace_record["draft_sync_apply_expected_normal_draft_conflict_by_seq_id"] = {
            str(seq_id): value for seq_id, value in expected_conflict_by_seq_id.items()
        }
        trace_record["draft_sync_apply_original_draft_home_intersection_by_seq_id"] = {
            str(seq_id): value for seq_id, value in original_draft_home_by_seq_id.items()
        }
        trace_record["draft_sync_apply_adjusted_draft_home_exclusion_by_seq_id"] = {
            str(seq_id): value for seq_id, value in adjusted_exclusion_by_seq_id.items()
        }
        total_tokens = sum(
            max(0, int(result.get("proposal_len", 0)))
            for result in validated_results
            if int(result["proposal_id"]) in set(executed_ids)
        )
        full_accept_tokens = sum(
            int(value) for value in append_tokens_by_proposal_id.values()
        )
        discarded_tokens = sum(
            int(value)
            for proposal_id, value in discarded_tokens_by_proposal_id.items()
            if proposal_id in set(executed_ids)
        )
        trace_record["eager_tokens_sync_apply_dry_run"] = int(total_tokens)
        trace_record["eager_tokens_sync_apply_dry_run_full_accept"] = int(full_accept_tokens)
        trace_record["eager_tokens_sync_apply_dry_run_discarded"] = int(discarded_tokens)
        trace_record["eager_sync_apply_dry_run_append_tokens"] = int(full_accept_tokens)
        trace_record["eager_tokens_sync_apply_dry_run_target_side"] = 0
        trace_record["eager_tokens_sync_apply_dry_run_draft_side"] = int(total_tokens)
        trace_record["eager_sync_apply_target_mutation_remaining_count"] = 0
        trace_record["eager_sync_apply_draft_mutation_remaining_count"] = sum(
            1 for value in mutation_remaining_by_seq_id.values() if bool(value)
        )
        if self._eager_commit_readiness_dry_run_enabled():
            plan.eager_commit_readiness_dry_run_enabled = True
            self._run_eager_commit_readiness_dry_run(
                plan,
                trace_record,
                validated_results,
                known_by_id,
                seq_by_id,
                plan_context,
            )

    def _run_eager_commit_readiness_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        validated_results: list[dict],
        known_by_id: dict[int, EagerProposal | ReadyEagerProposal],
        seq_by_id: dict[int, Sequence],
        plan_context: dict[str, set[int]],
    ) -> None:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        result_by_id = {int(result["proposal_id"]): result for result in validated_results}
        sync_executed_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_sync_apply_dry_run_executed_proposal_ids", [])
        }
        sync_consistent_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_sync_apply_dry_run_consistent_proposal_ids", [])
        }
        transfer_validated_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_result_transfer_validated_proposal_ids", [])
        }
        apply_executed_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_apply_dry_run_executed_proposal_ids", [])
        } or set(transfer_validated_ids)
        verify_executed_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_verify_dry_run_executed_proposal_ids", [])
        } or set(transfer_validated_ids)
        candidate_ids = sorted(sync_executed_ids & sync_consistent_ids & transfer_validated_ids)

        actual_counters_zero = all(
            int(trace_record.get(field, 0) or 0) == 0
            for field in (
                "eager_tokens_verified",
                "eager_tokens_accepted",
                "eager_tokens_rejected",
                "eager_tokens_invalidated",
            )
        )
        real_target_eager_empty = not bool(trace_record.get("target_eager_set") or [])
        unexpected_missing = bool(trace_record.get("missing_buffered_proposal_unexpected_seq_ids") or [])
        invalid_result_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_result_transfer_invalid_proposal_ids", [])
        }
        duplicate_result_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("eager_result_transfer_duplicate_proposal_ids", [])
        }
        validation_reason_by_id = trace_record.get("eager_result_transfer_validation_reason_by_proposal_id", {})
        result_action_by_id = trace_record.get("eager_result_transfer_action_by_proposal_id", {})
        result_verify_by_id = trace_record.get("eager_result_transfer_verify_result_by_proposal_id", {})
        result_accept_by_id = trace_record.get("eager_result_transfer_accept_len_by_proposal_id", {})
        result_append_by_id = trace_record.get("eager_result_transfer_append_tokens_by_proposal_id", {})
        result_discard_by_id = trace_record.get("eager_result_transfer_discarded_tokens_by_proposal_id", {})
        result_rollback_by_id = trace_record.get("eager_result_transfer_rollback_ok_by_proposal_id", {})
        result_mutation_by_id = trace_record.get("eager_result_transfer_mutation_detected_by_proposal_id", {})
        result_checkpoint_by_id = trace_record.get("eager_result_transfer_checkpoint_failed_by_proposal_id", {})
        sync_action_match_by_id = trace_record.get("eager_sync_apply_dry_run_action_match_by_proposal_id", {})
        sync_accept_match_by_id = trace_record.get("eager_sync_apply_dry_run_accept_len_match_by_proposal_id", {})
        sync_result_match_by_id = trace_record.get("eager_sync_apply_dry_run_result_match_by_proposal_id", {})
        sync_action_by_id = trace_record.get("eager_sync_apply_dry_run_draft_action_by_proposal_id", {})
        sync_result_by_id = trace_record.get("eager_sync_apply_dry_run_draft_verify_result_by_proposal_id", {})
        sync_accept_by_id = trace_record.get("eager_sync_apply_dry_run_draft_accept_len_by_proposal_id", {})
        sync_append_by_id = trace_record.get("eager_sync_apply_dry_run_append_tokens_by_proposal_id", {})
        sync_rollback_by_id = trace_record.get("eager_sync_apply_dry_run_rollback_ok_by_proposal_id", {})
        sync_mutation_by_id = trace_record.get("eager_sync_apply_dry_run_mutation_detected_by_proposal_id", {})
        sync_checkpoint_by_id = trace_record.get("eager_sync_apply_dry_run_checkpoint_failed_by_proposal_id", {})
        sync_len_before_by_seq = trace_record.get("eager_sync_apply_dry_run_sequence_len_before_by_seq_id", {})

        candidate_seq_ids: list[int] = []
        ready_ids: list[int] = []
        ready_seq_ids: list[int] = []
        not_ready_ids: list[int] = []
        not_ready_seq_ids: list[int] = []
        not_ready_reason_by_id: dict[int, str] = {}
        ready_token_count_by_id: dict[int, int] = {}
        ready_action_by_id: dict[int, str] = {}
        ready_accept_by_id: dict[int, int] = {}
        ready_verify_by_id: dict[int, str] = {}
        verify_ok_by_id: dict[int, bool] = {}
        apply_ok_by_id: dict[int, bool] = {}
        result_ok_by_id: dict[int, bool] = {}
        sync_ok_by_id: dict[int, bool] = {}
        frontier_ok_by_id: dict[int, bool] = {}
        token_payload_ok_by_id: dict[int, bool] = {}
        no_mutation_by_id: dict[int, bool] = {}
        reason_counts: dict[str, int] = {}
        full_accept_count = 0
        partial_reject_count = 0

        def add_reason(reason: str) -> None:
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

        for proposal_id in candidate_ids:
            result = result_by_id.get(proposal_id, {})
            proposal = known_by_id.get(proposal_id) or self.dual_batch_manager.ready_eager_proposals.by_id(proposal_id)
            seq_id = int(
                result.get(
                    "seq_id",
                    getattr(proposal, "seq_id", -1),
                )
            )
            seq = seq_by_id.get(seq_id)
            candidate_seq_ids.append(seq_id)
            proposal_len = int(result.get("proposal_len", gamma))
            base_len = int(result.get("base_len", getattr(proposal, "base_len", -1)))
            verify_result = str(
                self._trace_map_get(
                    result_verify_by_id,
                    proposal_id,
                    result.get("verify_result", "unknown"),
                )
            )
            action = str(
                self._trace_map_get(
                    result_action_by_id,
                    proposal_id,
                    result.get("apply_action", "unknown"),
                )
            )
            accept_len = int(
                self._trace_map_get(
                    result_accept_by_id,
                    proposal_id,
                    result.get("accepted_len", -1),
                )
            )
            append_tokens = int(
                self._trace_map_get(
                    result_append_by_id,
                    proposal_id,
                    result.get("append_token_count", 0),
                )
            )
            result_discarded = int(
                self._trace_map_get(
                    result_discard_by_id,
                    proposal_id,
                    result.get("discarded_token_count", 0),
                )
            )
            result_rollback_ok = bool(
                self._trace_map_get(result_rollback_by_id, proposal_id, result.get("rollback_ok", False))
            )
            result_mutation = bool(
                self._trace_map_get(result_mutation_by_id, proposal_id, result.get("mutation_detected", True))
            )
            result_checkpoint_failed = bool(
                self._trace_map_get(result_checkpoint_by_id, proposal_id, result.get("checkpoint_failed", True))
            )
            sync_action = str(self._trace_map_get(sync_action_by_id, proposal_id, action))
            sync_result = str(self._trace_map_get(sync_result_by_id, proposal_id, verify_result))
            sync_accept = int(self._trace_map_get(sync_accept_by_id, proposal_id, accept_len))
            sync_append_tokens = int(self._trace_map_get(sync_append_by_id, proposal_id, append_tokens))
            sync_rollback_ok = bool(self._trace_map_get(sync_rollback_by_id, proposal_id, False))
            sync_mutation = bool(self._trace_map_get(sync_mutation_by_id, proposal_id, True))
            sync_checkpoint_failed = bool(self._trace_map_get(sync_checkpoint_by_id, proposal_id, True))
            current_len = -1 if seq is None else int(len(seq))
            sync_len_before = int(self._trace_map_get(sync_len_before_by_seq, seq_id, current_len))
            proposal_state = str(getattr(proposal, "state", "")) if proposal is not None else ""

            verify_ok = (
                proposal_id in verify_executed_ids
                and verify_result == "full_accept"
                and accept_len == proposal_len == gamma
                and not bool(trace_record.get("eager_verify_dry_run_mutation_detected", False))
                and not bool(trace_record.get("eager_verify_dry_run_checkpoint_failed", False))
            )
            apply_ok = (
                proposal_id in apply_executed_ids
                and action == "append_full_accept_then_rollback"
                and append_tokens == proposal_len == gamma
                and result_rollback_ok
                and not result_mutation
                and not result_checkpoint_failed
            )
            result_token_accounting_ok = (
                (
                    verify_result == "full_accept"
                    and append_tokens == proposal_len == gamma
                    and result_discarded == 0
                )
                or (
                    verify_result != "full_accept"
                    and append_tokens == 0
                    and result_discarded == proposal_len == gamma
                )
            )
            result_ok = (
                proposal_id in transfer_validated_ids
                and proposal_id not in invalid_result_ids
                and proposal_id not in duplicate_result_ids
                and str(self._trace_map_get(validation_reason_by_id, proposal_id, "ok")) == "ok"
                and result_token_accounting_ok
            )
            sync_ok = (
                proposal_id in sync_executed_ids
                and proposal_id in sync_consistent_ids
                and sync_action == action
                and sync_result == verify_result
                and sync_accept == accept_len
                and bool(self._trace_map_get(sync_action_match_by_id, proposal_id, False))
                and bool(self._trace_map_get(sync_accept_match_by_id, proposal_id, False))
                and bool(self._trace_map_get(sync_result_match_by_id, proposal_id, False))
                and sync_append_tokens == append_tokens
                and sync_rollback_ok
                and not sync_mutation
                and not sync_checkpoint_failed
            )
            frontier_ok = (
                seq is not None
                and getattr(seq, "status", None) == SequenceStatus.RUNNING
                and not self.is_request_level_finished(seq, plan_context)
                and not self.is_speculative_span_invalidated(seq, plan_context)
                and not bool(getattr(seq, "pre_verify", True))
                and sync_len_before >= base_len >= 0
            )
            token_payload_ok = (
                proposal is not None
                and len(getattr(proposal, "proposal_token_ids", [])) == proposal_len == gamma
            )
            no_mutation = (
                actual_counters_zero
                and real_target_eager_empty
                and not result_mutation
                and not result_checkpoint_failed
                and not sync_mutation
                and not sync_checkpoint_failed
            )
            verify_ok_by_id[proposal_id] = bool(verify_ok)
            apply_ok_by_id[proposal_id] = bool(apply_ok)
            result_ok_by_id[proposal_id] = bool(result_ok)
            sync_ok_by_id[proposal_id] = bool(sync_ok)
            frontier_ok_by_id[proposal_id] = bool(frontier_ok)
            token_payload_ok_by_id[proposal_id] = bool(token_payload_ok)
            no_mutation_by_id[proposal_id] = bool(no_mutation)

            reason = None
            if proposal is None:
                reason = "missing_local_proposal"
            elif int(getattr(proposal, "seq_id", -1)) != seq_id:
                reason = "result_metadata_mismatch"
            elif proposal_state == "STALE":
                reason = "proposal_stale"
            elif proposal_state == "EXPIRED":
                reason = "proposal_expired"
            elif proposal_state == "INVALIDATED":
                reason = "proposal_invalidated"
            elif proposal_state and proposal_state != READY_EAGER_STATE_CONSUMED_APPLIED:
                reason = "proposal_stale"
            elif getattr(proposal, "takeover_routed_step_id", None) is None:
                reason = "result_metadata_mismatch"
            elif unexpected_missing:
                reason = "unexpected_missing_normal_proposal"
            elif not actual_counters_zero:
                reason = "actual_eager_counter_nonzero"
            elif not real_target_eager_empty:
                reason = "real_target_eager_nonempty"
            elif proposal_id not in verify_executed_ids:
                reason = "verify_not_executed"
            elif verify_result != "full_accept":
                reason = "not_full_accept"
            elif result_mutation:
                reason = "apply_mutation_detected"
            elif result_checkpoint_failed:
                reason = "apply_checkpoint_failed"
            elif not result_rollback_ok:
                reason = "apply_rollback_failed"
            elif proposal_id not in apply_executed_ids:
                reason = "apply_not_executed"
            elif action != "append_full_accept_then_rollback":
                reason = "apply_action_not_full_accept"
            elif proposal_id not in transfer_validated_ids:
                reason = "result_not_validated"
            elif proposal_id in invalid_result_ids:
                reason = "result_invalid"
            elif proposal_id in duplicate_result_ids:
                reason = "result_duplicate"
            elif not result_ok:
                reason = "result_metadata_mismatch"
            elif proposal_id not in sync_executed_ids:
                reason = "sync_apply_not_executed"
            elif not sync_ok:
                reason = "sync_apply_inconsistent"
            elif sync_mutation:
                reason = "sync_apply_mutation_detected"
            elif not sync_rollback_ok:
                reason = "sync_apply_rollback_failed"
            elif seq is None:
                reason = "missing_local_proposal"
            elif getattr(seq, "status", None) != SequenceStatus.RUNNING:
                reason = "seq_not_running"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_pre_verify"
            elif sync_len_before < base_len:
                reason = "base_len_mismatch"
            elif not frontier_ok:
                reason = "frontier_mismatch"
            elif not token_payload_ok:
                reason = "token_payload_missing"

            if verify_result == "full_accept":
                full_accept_count += 1
            else:
                partial_reject_count += 1

            if reason is None:
                ready_ids.append(proposal_id)
                ready_seq_ids.append(seq_id)
                ready_token_count_by_id[proposal_id] = proposal_len
                ready_action_by_id[proposal_id] = action
                ready_accept_by_id[proposal_id] = accept_len
                ready_verify_by_id[proposal_id] = verify_result
            else:
                not_ready_ids.append(proposal_id)
                not_ready_seq_ids.append(seq_id)
                not_ready_reason_by_id[proposal_id] = reason
                add_reason(reason)

        trace_record["enable_eager_commit_readiness_dry_run"] = True
        trace_record["eager_commit_readiness_dry_run_enabled"] = True
        trace_record["eager_commit_readiness_dry_run_source"] = EAGER_TAKEOVER_DRY_RUN_SOURCE
        trace_record["eager_commit_readiness_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["eager_commit_readiness_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_commit_readiness_from_sync_apply_proposal_ids"] = sorted(sync_executed_ids)
        trace_record["eager_commit_readiness_from_result_transfer_proposal_ids"] = sorted(transfer_validated_ids)
        trace_record["eager_commit_readiness_from_apply_proposal_ids"] = sorted(apply_executed_ids)
        trace_record["eager_commit_readiness_from_verify_proposal_ids"] = sorted(verify_executed_ids)
        trace_record["eager_commit_ready_proposal_ids"] = list(ready_ids)
        trace_record["eager_commit_ready_seq_ids"] = list(ready_seq_ids)
        trace_record["eager_commit_ready_token_count_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(ready_token_count_by_id.items())
        }
        trace_record["eager_commit_ready_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(ready_action_by_id.items())
        }
        trace_record["eager_commit_ready_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(ready_accept_by_id.items())
        }
        trace_record["eager_commit_ready_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(ready_verify_by_id.items())
        }
        trace_record["eager_commit_not_ready_proposal_ids"] = list(not_ready_ids)
        trace_record["eager_commit_not_ready_seq_ids"] = list(not_ready_seq_ids)
        trace_record["eager_commit_not_ready_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(not_ready_reason_by_id.items())
        }
        trace_record["eager_commit_readiness_verify_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(verify_ok_by_id.items())
        }
        trace_record["eager_commit_readiness_apply_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(apply_ok_by_id.items())
        }
        trace_record["eager_commit_readiness_result_transfer_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(result_ok_by_id.items())
        }
        trace_record["eager_commit_readiness_sync_apply_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(sync_ok_by_id.items())
        }
        trace_record["eager_commit_readiness_frontier_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(frontier_ok_by_id.items())
        }
        trace_record["eager_commit_readiness_token_payload_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(token_payload_ok_by_id.items())
        }
        trace_record["eager_commit_readiness_no_mutation_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(no_mutation_by_id.items())
        }
        trace_record["eager_commit_readiness_actual_counters_zero"] = bool(actual_counters_zero)
        trace_record["eager_commit_readiness_real_target_eager_empty"] = bool(real_target_eager_empty)
        trace_record["eager_commit_readiness_candidate_count"] = len(candidate_ids)
        trace_record["eager_commit_ready_count"] = len(ready_ids)
        trace_record["eager_commit_not_ready_count"] = len(not_ready_ids)
        trace_record["eager_commit_ready_token_count"] = sum(int(value) for value in ready_token_count_by_id.values())
        trace_record["eager_commit_not_ready_reason_counts"] = dict(sorted(reason_counts.items()))
        trace_record["eager_commit_readiness_full_accept_count"] = int(full_accept_count)
        trace_record["eager_commit_readiness_partial_reject_count"] = int(partial_reject_count)
        self._record_elapsed_ms(trace_record, "eager_commit_readiness_time_ms", timer_start)

    def _commit_ready_only_decisions_from_trace(self, trace_record: dict) -> list[dict]:
        ready_ids = [int(proposal_id) for proposal_id in trace_record.get("eager_commit_ready_proposal_ids", [])]
        ready_seq_ids = [int(seq_id) for seq_id in trace_record.get("eager_commit_ready_seq_ids", [])]
        token_by_id = trace_record.get("eager_commit_ready_token_count_by_proposal_id", {})
        accept_by_id = trace_record.get("eager_commit_ready_accept_len_by_proposal_id", {})
        action_by_id = trace_record.get("eager_commit_ready_action_by_proposal_id", {})
        result_by_id = trace_record.get("eager_commit_ready_verify_result_by_proposal_id", {})
        decisions = []
        for index, proposal_id in enumerate(ready_ids):
            seq_id = ready_seq_ids[index] if index < len(ready_seq_ids) else -1
            decisions.append(
                {
                    "proposal_id": int(proposal_id),
                    "seq_id": int(seq_id),
                    "token_count": int(self._trace_map_get(token_by_id, proposal_id, self.gamma)),
                    "accept_len": int(self._trace_map_get(accept_by_id, proposal_id, self.gamma)),
                    "action": str(
                        self._trace_map_get(
                            action_by_id,
                            proposal_id,
                            "append_full_accept_then_rollback",
                        )
                    ),
                    "verify_result": str(self._trace_map_get(result_by_id, proposal_id, "full_accept")),
                }
            )
        return decisions

    def _serialize_eager_commit_ready_only_payload(
        self,
        decisions: list[dict],
        plan: StepPlan,
    ) -> tuple[list[int], list[int]]:
        payload_values: list[int] = []
        for decision in decisions:
            payload_values.extend(
                [
                    int(decision["proposal_id"]),
                    int(decision["seq_id"]),
                    int(decision["token_count"]),
                    int(decision["accept_len"]),
                ]
            )
        meta_values = [
            int(EAGER_COMMIT_READY_ONLY_MAGIC),
            int(EAGER_COMMIT_READY_ONLY_OP),
            int(plan.plan_id),
            -1 if plan.step_id is None else int(plan.step_id),
            len(decisions),
            len(payload_values),
        ]
        return meta_values, payload_values

    def _deserialize_eager_commit_ready_only_payload(
        self,
        meta_values: list[int],
        payload_values: list[int],
    ) -> list[dict]:
        if len(meta_values) != EAGER_COMMIT_READY_ONLY_META_LEN:
            raise ValueError(f"eager commit-ready-only meta length mismatch: {len(meta_values)}")
        if int(meta_values[0]) != int(EAGER_COMMIT_READY_ONLY_MAGIC):
            raise ValueError(f"eager commit-ready-only magic mismatch: got={meta_values[0]}")
        if int(meta_values[1]) != int(EAGER_COMMIT_READY_ONLY_OP):
            raise ValueError(f"eager commit-ready-only op mismatch: got={meta_values[1]}")
        num_decisions = int(meta_values[4])
        payload_len = int(meta_values[5])
        expected_len = num_decisions * EAGER_COMMIT_READY_ONLY_PAYLOAD_WIDTH
        if payload_len != expected_len or len(payload_values) != expected_len:
            raise ValueError(
                "eager commit-ready-only payload length mismatch: "
                f"num_decisions={num_decisions}, payload_len={payload_len}, actual={len(payload_values)}"
            )
        decisions = []
        for index in range(num_decisions):
            base = index * EAGER_COMMIT_READY_ONLY_PAYLOAD_WIDTH
            proposal_id, seq_id, token_count, accept_len = payload_values[base:base + 4]
            decisions.append(
                {
                    "proposal_id": int(proposal_id),
                    "seq_id": int(seq_id),
                    "token_count": int(token_count),
                    "accept_len": int(accept_len),
                    "action": "append_full_accept_then_rollback",
                    "verify_result": "full_accept",
                }
            )
        return decisions

    def _send_eager_commit_ready_only_decision(
        self,
        plan: StepPlan,
        trace_record: dict,
        known_by_id: dict[int, EagerProposal | ReadyEagerProposal],
        seq_by_id: dict[int, Sequence],
    ) -> None:
        decisions = self._commit_ready_only_decisions_from_trace(trace_record)
        meta_values, payload_values = self._serialize_eager_commit_ready_only_payload(decisions, plan)
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta_values[5]) > 0:
            payload = torch.tensor(payload_values, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        self._run_eager_commit_ready_only(
            plan,
            trace_record,
            decisions,
            known_by_id,
            seq_by_id,
            self._eager_transfer_plan_context(plan, trace_record),
            side="draft",
        )
        if self._continuous_eager_verify_apply_dry_run_enabled():
            plan_context = self._eager_transfer_plan_context(plan, trace_record)
            committed_ids = [int(proposal_id) for proposal_id in trace_record.get("eager_committed_proposal_ids", [])]
            committed_seq_ids = [int(seq_id) for seq_id in trace_record.get("eager_committed_seq_ids", [])]
            continuous_proposals = self._run_continuous_eager_draft_shadow_dry_run(
                plan,
                trace_record,
                committed_ids,
                committed_seq_ids,
                known_by_id,
                seq_by_id,
                plan_context,
            )
            self._send_continuous_eager_transfer_dry_run(continuous_proposals, plan, trace_record)
            validated_results = self._receive_continuous_eager_result_transfer_dry_run(plan, trace_record)
            self._run_continuous_eager_sync_apply_dry_run(plan, trace_record, validated_results)
            self._resolve_rolling_continuous_overlap_after_parent_result(trace_record)
            if self._continuous_eager_commit_depth1_ready_only_enabled():
                self._send_continuous_eager_commit_depth1_decision(
                    plan,
                    trace_record,
                    dict(self._draft_sent_eager_proposals_by_id),
                    self._local_sequence_by_id(),
                )

    def _receive_eager_commit_ready_only_decision(
        self,
        plan: StepPlan,
        trace_record: dict,
    ) -> None:
        meta = torch.zeros(EAGER_COMMIT_READY_ONLY_META_LEN, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        meta_values = [int(value) for value in meta.tolist()]
        payload_len = int(meta_values[5])
        payload_values: list[int] = []
        if payload_len > 0:
            payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
            payload_values = [int(value) for value in payload.tolist()]
        decisions = self._deserialize_eager_commit_ready_only_payload(meta_values, payload_values)
        known_by_id = {
            int(proposal.proposal_id): proposal
            for proposal in self.dual_batch_manager.ready_eager_proposals.proposals()
        }
        self._run_eager_commit_ready_only(
            plan,
            trace_record,
            decisions,
            known_by_id,
            self._local_sequence_by_id(),
            self._eager_transfer_plan_context(plan, trace_record),
            side="target",
        )
        if self._continuous_eager_verify_apply_dry_run_enabled():
            plan_context = self._eager_transfer_plan_context(plan, trace_record)
            continuous_proposals = self._receive_continuous_eager_transfer_dry_run(plan, trace_record)
            continuous_results = self._run_continuous_eager_target_verify_apply_dry_run(
                plan,
                trace_record,
                continuous_proposals,
                plan_context,
            )
            self._send_continuous_eager_result_transfer_dry_run(plan, trace_record, continuous_results)
            if self._continuous_eager_commit_depth1_ready_only_enabled():
                self._receive_continuous_eager_commit_depth1_decision(
                    plan,
                    trace_record,
                    {int(proposal.proposal_id): proposal for proposal in continuous_proposals},
                )

    def _mark_eager_commit_finished_if_needed(self, seq: Sequence, proposal_tokens: list[int]) -> None:
        if not proposal_tokens:
            return
        hit_eos = (not seq.ignore_eos) and any(is_eos(int(token_id), self.scheduler.eos) for token_id in proposal_tokens)
        hit_max_tokens = int(seq.num_completion_tokens) >= int(seq.max_tokens)
        if not (hit_eos or hit_max_tokens):
            return
        if seq in self.scheduler.running:
            self.scheduler.block_manager.deallocate(seq)
            self.scheduler.running.remove(seq)
        if seq not in self.scheduler.finished:
            self.scheduler.finished.append(seq)
        seq.mark_finished()

    def _continuous_shadow_proposal_id(self, root_proposal_id: int, chain_depth: int) -> int:
        return 900_000_000 + int(root_proposal_id) * 100 + int(chain_depth)

    def _run_continuous_eager_shadow_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        committed_ids: list[int],
        committed_seq_ids: list[int],
        known_by_id: dict[int, EagerProposal | ReadyEagerProposal],
        seq_by_id: dict[int, Sequence],
        plan_context: dict[str, set[int]],
    ) -> None:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        max_depth = int(getattr(self.global_config, "max_continuous_eager_chain_depth", 1) or 1)
        max_requests = int(getattr(self.global_config, "max_continuous_eager_requests_per_step", 1) or 1)
        token_per_request = int(
            getattr(self.global_config, "max_continuous_eager_tokens_per_request", gamma) or gamma
        )
        token_per_request = max(1, min(token_per_request, gamma))
        max_tokens_per_step = int(
            getattr(
                self.global_config,
                "max_continuous_eager_tokens_per_step",
                max_requests * token_per_request,
            )
            or (max_requests * token_per_request)
        )

        parent_by_id: dict[int, int] = {}
        root_by_id: dict[int, int] = {}
        depth_by_id: dict[int, int] = {}
        chain_index_by_id: dict[int, int] = {}
        token_count_by_id: dict[int, int] = {}
        parent_source_by_id: dict[int, str] = {}
        candidate_ids: list[int] = []
        candidate_seq_ids: list[int] = []
        not_ready_ids: list[int] = []
        not_ready_reason_by_id: dict[int, str] = {}
        stale_ids: list[int] = []
        duplicate_ids: list[int] = []
        frontier_mismatch_ids: list[int] = []
        true_frontier_mismatch_ids: list[int] = []
        parent_shadow_not_committed_ids: list[int] = []
        parent_not_ready_ids: list[int] = []
        seq_finished_ids: list[int] = []
        overshot_ids: list[int] = []
        invalidated_ids: list[int] = []
        chain_distribution: dict[int, int] = {}
        reason_counts: dict[str, int] = {}
        seen_seq_depth: set[tuple[int, int]] = set()
        token_budget = 0

        def add_not_ready(proposal_id: int, reason: str) -> None:
            not_ready_ids.append(proposal_id)
            not_ready_reason_by_id[proposal_id] = reason
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            if reason in {"seq_not_running", "seq_finished"}:
                seq_finished_ids.append(proposal_id)
            elif reason == "frontier_mismatch":
                frontier_mismatch_ids.append(proposal_id)
                true_frontier_mismatch_ids.append(proposal_id)
            elif reason == "parent_shadow_not_committed":
                parent_shadow_not_committed_ids.append(proposal_id)
            elif reason in {"parent_not_committed", "parent_shadow_not_ready"}:
                parent_not_ready_ids.append(proposal_id)
            elif reason == "duplicate_continuous_candidate":
                duplicate_ids.append(proposal_id)
            elif reason == "base_overshot":
                overshot_ids.append(proposal_id)
            elif reason == "span_invalidated":
                invalidated_ids.append(proposal_id)
            elif reason in {"missing_parent_proposal", "seq_pre_verify"}:
                stale_ids.append(proposal_id)

        parent_pairs = list(dict.fromkeys(zip(committed_ids, committed_seq_ids)))[:max_requests]
        for chain_index, (root_proposal_id, seq_id) in enumerate(parent_pairs):
            root_proposal_id = int(root_proposal_id)
            seq_id = int(seq_id)
            parent_id = root_proposal_id
            parent = known_by_id.get(root_proposal_id) or self.dual_batch_manager.ready_eager_proposals.by_id(root_proposal_id)
            seq = seq_by_id.get(seq_id)

            for depth in range(1, max_depth + 1):
                proposal_id = self._continuous_shadow_proposal_id(root_proposal_id, depth)
                parent_by_id[proposal_id] = int(parent_id)
                root_by_id[proposal_id] = root_proposal_id
                depth_by_id[proposal_id] = int(depth)
                chain_index_by_id[proposal_id] = int(chain_index)
                token_count_by_id[proposal_id] = int(token_per_request)
                parent_source_by_id[proposal_id] = (
                    CONTINUOUS_EAGER_PARENT_SOURCE if depth == 1 else CONTINUOUS_EAGER_DRY_RUN_SOURCE
                )

                reason = None
                current_len = -1 if seq is None else int(len(seq))
                if (seq_id, depth) in seen_seq_depth:
                    reason = "duplicate_continuous_candidate"
                elif depth > 1:
                    reason = "parent_shadow_not_committed"
                elif token_budget + token_per_request > max_tokens_per_step:
                    reason = "token_budget_exhausted"
                elif parent is None:
                    reason = "missing_parent_proposal"
                elif seq is None:
                    reason = "seq_not_found"
                elif getattr(seq, "status", None) != SequenceStatus.RUNNING:
                    reason = "seq_not_running"
                elif self.is_request_level_finished(seq, plan_context):
                    reason = "seq_finished"
                elif self.is_speculative_span_invalidated(seq, plan_context):
                    reason = "span_invalidated"
                elif bool(getattr(seq, "pre_verify", True)):
                    reason = "seq_pre_verify"
                elif getattr(parent, "real_commit_step_id", None) is None:
                    reason = "parent_not_committed"
                elif current_len < int(getattr(parent, "base_len", 0)) + int(getattr(parent, "proposal_len", 0)):
                    reason = "frontier_mismatch"

                seen_seq_depth.add((seq_id, depth))
                if reason is not None:
                    add_not_ready(proposal_id, reason)
                    parent_id = proposal_id
                    continue

                candidate_ids.append(proposal_id)
                candidate_seq_ids.append(seq_id)
                chain_distribution[depth] = int(chain_distribution.get(depth, 0)) + 1
                token_budget += token_per_request
                add_not_ready(proposal_id, "shadow_verify_not_executed")
                parent_id = proposal_id

        trace_record["enable_continuous_eager_dry_run"] = True
        trace_record["continuous_eager_dry_run_enabled"] = True
        trace_record["continuous_eager_source"] = CONTINUOUS_EAGER_DRY_RUN_SOURCE
        trace_record["continuous_eager_parent_source"] = CONTINUOUS_EAGER_PARENT_SOURCE
        trace_record["continuous_eager_execution_stage"] = "candidate_shadow"
        trace_record["continuous_shadow_stage"] = "candidate_shadow"
        trace_record["max_continuous_eager_chain_depth"] = max_depth
        trace_record["max_continuous_depth_configured"] = max_depth
        trace_record["max_continuous_depth_observed"] = max(depth_by_id.values(), default=0)
        trace_record["max_continuous_eager_requests_per_step"] = max_requests
        trace_record["max_continuous_eager_tokens_per_step"] = max_tokens_per_step
        trace_record["max_continuous_eager_tokens_per_request"] = token_per_request
        trace_record["continuous_eager_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["continuous_eager_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["continuous_eager_candidate_token_count_by_proposal_id"] = {
            str(proposal_id): int(token_count_by_id[proposal_id]) for proposal_id in sorted(candidate_ids)
        }
        trace_record["continuous_eager_parent_proposal_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_by_id.items())
        }
        trace_record["continuous_eager_chain_depth_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(depth_by_id.items())
        }
        trace_record["continuous_eager_chain_index_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(chain_index_by_id.items())
        }
        trace_record["continuous_eager_root_proposal_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(root_by_id.items())
        }
        trace_record["continuous_eager_parent_source_by_proposal_id"] = {
            str(proposal_id): str(value) for proposal_id, value in sorted(parent_source_by_id.items())
        }
        trace_record["continuous_eager_verified_proposal_ids"] = []
        trace_record["continuous_eager_full_accept_proposal_ids"] = []
        trace_record["continuous_eager_partial_reject_proposal_ids"] = []
        trace_record["continuous_eager_accept_len_by_proposal_id"] = {}
        trace_record["continuous_eager_verify_result_by_proposal_id"] = {}
        trace_record["continuous_eager_commit_ready_shadow_proposal_ids"] = []
        trace_record["continuous_eager_commit_ready_shadow_seq_ids"] = []
        trace_record["continuous_eager_commit_ready_shadow_token_count_by_proposal_id"] = {}
        trace_record["continuous_eager_not_ready_shadow_proposal_ids"] = sorted(set(not_ready_ids))
        trace_record["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(not_ready_reason_by_id.items())
        }
        trace_record["continuous_eager_stale_proposal_ids"] = sorted(set(stale_ids))
        trace_record["continuous_eager_duplicate_proposal_ids"] = sorted(set(duplicate_ids))
        trace_record["continuous_eager_parent_not_ready_proposal_ids"] = sorted(set(parent_not_ready_ids))
        trace_record["continuous_eager_parent_shadow_not_committed_proposal_ids"] = sorted(
            set(parent_shadow_not_committed_ids)
        )
        trace_record["continuous_eager_true_frontier_mismatch_proposal_ids"] = sorted(
            set(true_frontier_mismatch_ids)
        )
        trace_record["continuous_eager_frontier_mismatch_proposal_ids"] = sorted(set(frontier_mismatch_ids))
        trace_record["continuous_eager_seq_finished_proposal_ids"] = sorted(set(seq_finished_ids))
        trace_record["continuous_eager_overshot_proposal_ids"] = sorted(set(overshot_ids))
        trace_record["continuous_eager_invalidated_proposal_ids"] = sorted(set(invalidated_ids))
        trace_record["continuous_eager_mutation_detected_count"] = 0
        trace_record["continuous_eager_real_commit_count"] = 0
        trace_record["continuous_eager_candidate_proposal_count"] = len(candidate_ids)
        trace_record["continuous_eager_candidate_token_count"] = sum(
            int(token_count_by_id[proposal_id]) for proposal_id in candidate_ids
        )
        trace_record["continuous_eager_verified_proposal_count"] = 0
        trace_record["continuous_eager_full_accept_proposal_count"] = 0
        trace_record["continuous_eager_commit_ready_shadow_proposal_count"] = 0
        trace_record["continuous_eager_commit_ready_shadow_token_count"] = 0
        trace_record["continuous_eager_not_ready_shadow_proposal_count"] = len(set(not_ready_ids))
        trace_record["continuous_eager_chain_length_distribution"] = {
            str(depth): count for depth, count in sorted(chain_distribution.items())
        }
        trace_record["continuous_eager_drop_reason_counts"] = dict(sorted(reason_counts.items()))
        trace_record["continuous_parent_shadow_not_committed_count"] = len(set(parent_shadow_not_committed_ids))
        trace_record["continuous_parent_shadow_not_ready_count"] = len(set(parent_not_ready_ids))
        trace_record["continuous_true_frontier_mismatch_count"] = len(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_estimated_committed_token_share_of_output"] = 0.0
        trace_record["continuous_eager_payload_len_units_per_ready_token"] = 0.0
        self._record_elapsed_ms(trace_record, "continuous_eager_overhead_time_ms", timer_start)

    def _continuous_eager_limits(self) -> tuple[int, int, int, int]:
        gamma = int(self.gamma)
        max_depth = int(getattr(self.global_config, "max_continuous_eager_chain_depth", 1) or 1)
        max_requests = int(getattr(self.global_config, "max_continuous_eager_requests_per_step", 1) or 1)
        token_per_request = int(
            getattr(self.global_config, "max_continuous_eager_tokens_per_request", gamma) or gamma
        )
        token_per_request = max(1, min(token_per_request, gamma))
        max_tokens_per_step = int(
            getattr(
                self.global_config,
                "max_continuous_eager_tokens_per_step",
                max_requests * token_per_request,
            )
            or (max_requests * token_per_request)
        )
        return max_depth, max_requests, token_per_request, max_tokens_per_step

    def _continuous_proposal_seq_id_by_id(self, trace_record: dict) -> dict[int, int]:
        proposal_ids = [int(proposal_id) for proposal_id in trace_record.get("continuous_eager_candidate_proposal_ids", [])]
        seq_ids = [int(seq_id) for seq_id in trace_record.get("continuous_eager_candidate_seq_ids", [])]
        return {proposal_id: seq_id for proposal_id, seq_id in zip(proposal_ids, seq_ids)}

    def _rolling_continuous_limits(self) -> tuple[int, int, int]:
        max_depth = int(getattr(self.global_config, "max_rolling_continuous_depth", 2) or 2)
        max_children = int(getattr(self.global_config, "max_rolling_continuous_draft_children_per_step", 1) or 1)
        max_seqs = int(getattr(self.global_config, "max_rolling_continuous_seqs_per_step", 1) or 1)
        return max(2, max_depth), max(1, max_children), max(1, max_seqs)

    def _set_rolling_common_trace(self, trace_record: dict, max_depth: int) -> None:
        trace_record["enable_rolling_continuous_eager_dry_run"] = True
        trace_record["rolling_continuous_eager_dry_run_enabled"] = True
        trace_record["rolling_continuous_stage"] = ROLLING_CONTINUOUS_STAGE
        trace_record["rolling_continuous_source"] = ROLLING_CONTINUOUS_EAGER_DRY_RUN_SOURCE
        trace_record["max_rolling_continuous_depth"] = int(max_depth)

    def _rolling_normal_lane_conflicts(self, trace_record: dict, seq_ids: set[int]) -> tuple[list[int], list[int]]:
        normal_draft_source = trace_record.get("actual_draft_home_set_for_normal_draft")
        if normal_draft_source is None:
            normal_draft_source = trace_record.get("draft_home_set") or []
        normal_draft = {int(seq_id) for seq_id in normal_draft_source}
        target_normal = {int(seq_id) for seq_id in trace_record.get("target_normal_verify_seq_ids", [])}
        conflicts = sorted(seq_ids & (normal_draft | target_normal))
        return sorted(seq_ids - set(conflicts)), conflicts

    def _run_rolling_continuous_child_draft_overlap_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        parent_proposals: list[EagerProposal],
        seq_by_id: dict[int, Sequence],
        checkpoints: dict[int, dict],
    ) -> None:
        if not self._rolling_continuous_eager_dry_run_enabled():
            return
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        max_depth, max_children, max_seqs = self._rolling_continuous_limits()
        self._set_rolling_common_trace(trace_record, max_depth)

        selected: list[tuple[EagerProposal, Sequence, int]] = []
        seen_seq_ids: set[int] = set()
        for parent in parent_proposals:
            seq_id = int(parent.seq_id)
            if len(selected) >= max_children or len(seen_seq_ids) >= max_seqs:
                break
            seq = seq_by_id.get(seq_id)
            if seq is None or seq_id in seen_seq_ids:
                continue
            selected.append((parent, seq, len(selected)))
            seen_seq_ids.add(seq_id)

        target_parent_ids = [int(parent.proposal_id) for parent, _seq, _idx in selected]
        target_parent_seq_ids = [int(parent.seq_id) for parent, _seq, _idx in selected]
        child_ids: list[int] = []
        child_seq_ids: list[int] = []
        chain_proposal_ids: list[int] = []
        chain_seq_ids: list[int] = []
        parent_by_id: dict[int, int] = {}
        children_by_id: dict[int, list[int]] = {}
        root_by_id: dict[int, int] = {}
        depth_by_id: dict[int, int] = {}
        chain_index_by_id: dict[int, int] = {}
        base_len_by_id: dict[int, int] = {}
        parent_base_len_by_id: dict[int, int] = {}
        parent_expected_accept_len_by_id: dict[int, int] = {}
        parent_source_step_by_id: dict[int, int] = {}
        parent_source_plan_by_id: dict[int, int] = {}
        parent_source_by_id: dict[int, str] = {}
        status_by_id: dict[int, str] = {}
        status_reason_by_id: dict[int, str] = {}
        token_count_by_id: dict[int, int] = {}
        valid_children: list[tuple[int, EagerProposal, Sequence, dict]] = []
        generated_by_child_id: dict[int, list[int]] = {}
        duplicate_child_ids: list[int] = []
        frontier_mismatch_ids: list[int] = []
        drafted_without_parent_ids: list[int] = []

        for parent, seq, chain_index in selected:
            parent_id = int(parent.proposal_id)
            root_id = -1 if parent.parent_proposal_id is None else int(parent.parent_proposal_id)
            child_id = self._continuous_shadow_proposal_id(root_id, 2)
            child_depth = 2
            parent_seq_id = int(parent.seq_id)
            parent_base = int(parent.base_len)
            child_base = parent_base + int(parent.proposal_len)
            chain_proposal_ids.extend([parent_id, child_id])
            chain_seq_ids.extend([parent_seq_id, parent_seq_id])
            parent_by_id[parent_id] = root_id
            parent_by_id[child_id] = parent_id
            children_by_id.setdefault(parent_id, []).append(child_id)
            root_by_id[parent_id] = root_id
            root_by_id[child_id] = root_id
            depth_by_id[parent_id] = 1
            depth_by_id[child_id] = child_depth
            chain_index_by_id[parent_id] = chain_index
            chain_index_by_id[child_id] = chain_index
            base_len_by_id[parent_id] = parent_base
            base_len_by_id[child_id] = child_base
            parent_base_len_by_id[parent_id] = int(getattr(parent, "base_len", -1))
            parent_base_len_by_id[child_id] = parent_base
            parent_expected_accept_len_by_id[parent_id] = int(parent.proposal_len)
            parent_expected_accept_len_by_id[child_id] = int(parent.proposal_len)
            parent_source_step_by_id[parent_id] = -1 if parent.parent_step_id is None else int(parent.parent_step_id)
            parent_source_step_by_id[child_id] = -1 if plan.step_id is None else int(plan.step_id)
            parent_source_plan_by_id[parent_id] = int(parent.source_plan_id)
            parent_source_plan_by_id[child_id] = int(parent.source_plan_id)
            parent_source_by_id[parent_id] = CONTINUOUS_EAGER_PARENT_SOURCE
            parent_source_by_id[child_id] = CONTINUOUS_EAGER_COMMIT_SOURCE
            token_count_by_id[child_id] = gamma
            status_by_id[parent_id] = "PARENT_VERIFY_PENDING"

            reason = None
            if child_depth > max_depth:
                reason = "max_depth_exceeded"
            elif child_id in self._rolling_continuous_shadow_proposals_by_id:
                reason = "duplicate_child"
                duplicate_child_ids.append(child_id)
            elif root_id < 0:
                reason = "missing_root_proposal"
                drafted_without_parent_ids.append(child_id)
            elif int(parent.seq_id) != int(seq.seq_id):
                reason = "parent_seq_mismatch"
                drafted_without_parent_ids.append(child_id)
            elif int(parent.proposal_len) != gamma:
                reason = "invalid_parent_len"
            elif int(len(seq)) != child_base:
                reason = "frontier_mismatch"
                frontier_mismatch_ids.append(child_id)

            if reason is not None:
                status_by_id[child_id] = "CHILD_DROPPED"
                status_reason_by_id[child_id] = reason
                continue

            child_ids.append(child_id)
            child_seq_ids.append(parent_seq_id)
            status_by_id[child_id] = "DRAFTED_CHILD_SHADOW"
            valid_children.append((child_id, parent, seq, checkpoints[parent_seq_id]))
            generated_by_child_id[child_id] = []

        valid_child_seqs = [seq for _child_id, _parent, seq, _checkpoint in valid_children]
        for _ in range(gamma):
            if not valid_child_seqs:
                break
            self._allocate_decode_slots_for_dual(valid_child_seqs, plan, "rolling_continuous_eager_draft_shadow")
            input_ids, positions = self.prepare_pearl_decode(valid_child_seqs)
            torch.cuda.synchronize()
            logits = self.run_model(input_ids, positions, False)
            if self.tp_params.local_rank == 0:
                sample_tokens = logits.argmax(dim=-1)
            else:
                sample_tokens = torch.zeros(
                    len(valid_child_seqs),
                    dtype=torch.int64,
                    pin_memory=True,
                ).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            reset_context(self.tp_params)
            for (child_id, _parent, seq, _checkpoint), token_id in zip(valid_children, sample_tokens.tolist()):
                int_token_id = int(token_id)
                seq.append_token(int_token_id)
                generated_by_child_id[int(child_id)].append(int_token_id)

        for child_id, parent, seq, checkpoint in valid_children:
            child_tokens = [int(token_id) for token_id in generated_by_child_id[child_id]]
            if len(child_tokens) != gamma:
                status_by_id[child_id] = "CHILD_DROPPED"
                status_reason_by_id[child_id] = "invalid_child_token_span"
                continue
            child_base = int(parent.base_len) + int(parent.proposal_len)
            to_be_verified = [int(token_id) for token_id in seq.token_ids[-2 * gamma + 1:-gamma + 1]]
            proposal = EagerProposal(
                proposal_id=int(child_id),
                seq_id=int(seq.seq_id),
                request_id=seq.request_id,
                lane=LANE_EAGER,
                parent_proposal_id=int(parent.proposal_id),
                parent_kind=LANE_EAGER,
                parent_step_id=None if plan.step_id is None else int(plan.step_id),
                source_step_id=0 if plan.step_id is None else int(plan.step_id),
                source_plan_id=int(plan.plan_id),
                home_batch_id=-1 if seq.home_batch_id is None else int(seq.home_batch_id),
                base_len=child_base,
                base_pre_verify=bool(checkpoint["pre_verify"]),
                base_num_completion_tokens=int(checkpoint["num_completion_tokens"]) + int(parent.proposal_len),
                proposal_token_ids=child_tokens,
                to_be_verified_token_ids=to_be_verified,
                proposal_len=gamma,
                state=EAGER_STATE_READY_TO_VERIFY,
                valid=True,
            )
            self._rolling_continuous_shadow_proposals_by_id[int(child_id)] = proposal

        rolling_seq_ids = set(target_parent_seq_ids) | set(child_seq_ids)
        normal_excluded, normal_conflicts = self._rolling_normal_lane_conflicts(trace_record, rolling_seq_ids)
        overlap_seq_ids = sorted(set(target_parent_seq_ids) & set(child_seq_ids))
        trace_record["target_rolling_eager_verify_proposal_ids"] = target_parent_ids
        trace_record["target_rolling_eager_verify_seq_ids"] = target_parent_seq_ids
        trace_record["draft_rolling_eager_draft_proposal_ids"] = list(child_ids)
        trace_record["draft_rolling_eager_draft_seq_ids"] = list(child_seq_ids)
        trace_record["rolling_same_seq_overlap_seq_ids"] = overlap_seq_ids
        trace_record["rolling_same_seq_overlap_count"] = len(overlap_seq_ids)
        trace_record["rolling_normal_lane_excluded_seq_ids"] = normal_excluded
        trace_record["rolling_normal_lane_conflict_seq_ids"] = normal_conflicts
        trace_record["rolling_normal_lane_conflict_count"] = len(normal_conflicts)
        trace_record["rolling_chain_proposal_ids"] = sorted(set(chain_proposal_ids))
        trace_record["rolling_chain_seq_ids"] = list(chain_seq_ids)
        trace_record["rolling_chain_parent_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_by_id.items())
        }
        trace_record["rolling_chain_children_by_proposal_id"] = {
            str(proposal_id): sorted(children) for proposal_id, children in sorted(children_by_id.items())
        }
        trace_record["rolling_chain_root_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(root_by_id.items())
        }
        trace_record["rolling_chain_depth_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(depth_by_id.items())
        }
        trace_record["rolling_chain_index_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(chain_index_by_id.items())
        }
        trace_record["rolling_chain_base_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(base_len_by_id.items())
        }
        trace_record["rolling_chain_parent_base_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_base_len_by_id.items())
        }
        trace_record["rolling_chain_parent_expected_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_expected_accept_len_by_id.items())
        }
        trace_record["rolling_chain_parent_source_step_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_source_step_by_id.items())
        }
        trace_record["rolling_chain_parent_source_plan_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_source_plan_by_id.items())
        }
        trace_record["rolling_chain_parent_source_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in sorted(parent_source_by_id.items())
        }
        trace_record["rolling_chain_status_by_proposal_id"] = {
            str(proposal_id): status for proposal_id, status in sorted(status_by_id.items())
        }
        trace_record["rolling_chain_status_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(status_reason_by_id.items())
        }
        trace_record["rolling_child_generated_proposal_ids"] = list(child_ids)
        trace_record["rolling_child_candidate_proposal_count"] = len(child_ids)
        trace_record["rolling_child_candidate_token_count"] = len(child_ids) * gamma
        trace_record["rolling_duplicate_child_count"] = len(set(duplicate_child_ids))
        trace_record["rolling_frontier_mismatch_count"] = len(set(frontier_mismatch_ids))
        trace_record["rolling_child_drafted_without_valid_parent_count"] = len(set(drafted_without_parent_ids))
        trace_record["rolling_max_depth_observed"] = max(depth_by_id.values(), default=0)
        trace_record["max_rolling_continuous_depth_observed"] = max(depth_by_id.values(), default=0)
        self._record_elapsed_ms(trace_record, "rolling_continuous_overhead_time_ms", timer_start)

    def _resolve_rolling_continuous_overlap_after_parent_result(self, trace_record: dict) -> None:
        if not self._rolling_continuous_eager_dry_run_enabled():
            return
        if not trace_record.get("rolling_continuous_eager_dry_run_enabled"):
            return
        parent_by_id = {
            int(key): int(value)
            for key, value in (trace_record.get("rolling_chain_parent_by_proposal_id") or {}).items()
        }
        depth_by_id = {
            int(key): int(value)
            for key, value in (trace_record.get("rolling_chain_depth_by_proposal_id") or {}).items()
        }
        root_by_id = {
            int(key): int(value)
            for key, value in (trace_record.get("rolling_chain_root_by_proposal_id") or {}).items()
        }
        status_by_id = {
            int(key): str(value)
            for key, value in (trace_record.get("rolling_chain_status_by_proposal_id") or {}).items()
        }
        reason_by_id = {
            int(key): str(value)
            for key, value in (trace_record.get("rolling_chain_status_reason_by_proposal_id") or {}).items()
        }
        child_ids = [int(proposal_id) for proposal_id in trace_record.get("rolling_child_generated_proposal_ids", [])]
        parent_verified = set(int(proposal_id) for proposal_id in trace_record.get("continuous_eager_verified_proposal_ids", []))
        parent_full = set(int(proposal_id) for proposal_id in trace_record.get("continuous_eager_full_accept_proposal_ids", []))
        parent_partial = set(int(proposal_id) for proposal_id in trace_record.get("continuous_eager_partial_reject_proposal_ids", []))
        verify_result_by_id = {
            int(key): str(value)
            for key, value in (trace_record.get("continuous_eager_verify_result_by_proposal_id") or {}).items()
        }

        ready_child_ids: list[int] = []
        ready_seq_ids: list[int] = []
        invalidated_ids: list[int] = []
        invalidated_reason_by_id: dict[int, str] = {}
        cascade_ids: list[int] = []
        cascade_reason_by_id: dict[int, str] = {}
        cascade_depth_by_id: dict[int, int] = {}
        parent_invalidated_ids: list[int] = []
        child_verified_without_parent = 0
        child_committed_without_parent = 0

        proposal_seq_by_id = self._continuous_proposal_seq_id_by_id(trace_record)
        proposal_seq_by_id.update(
            {
                int(proposal_id): int(seq_id)
                for proposal_id, seq_id in zip(
                    trace_record.get("draft_rolling_eager_draft_proposal_ids", []),
                    trace_record.get("draft_rolling_eager_draft_seq_ids", []),
                )
            }
        )
        for parent_id in parent_verified:
            if parent_id in parent_full:
                status_by_id[parent_id] = "PARENT_FULL_ACCEPT"
            elif parent_id in parent_partial:
                status_by_id[parent_id] = "PARENT_NOT_FULL_ACCEPT"
            else:
                status_by_id[parent_id] = "PARENT_VERIFY_PENDING"

        for child_id in child_ids:
            parent_id = int(parent_by_id.get(child_id, -1))
            depth = int(depth_by_id.get(child_id, 0))
            existing_reason = reason_by_id.get(child_id)
            if existing_reason:
                status_by_id[child_id] = "CHILD_DROPPED"
                invalidated_ids.append(child_id)
                invalidated_reason_by_id[child_id] = existing_reason
                cascade_ids.append(child_id)
                cascade_reason_by_id[child_id] = existing_reason
                cascade_depth_by_id[child_id] = depth
                continue
            if parent_id in parent_full:
                status_by_id[child_id] = "CHILD_READY_AFTER_PARENT_FULL_ACCEPT"
                ready_child_ids.append(child_id)
                ready_seq_ids.append(int(proposal_seq_by_id.get(child_id, -1)))
                continue
            verify_result = verify_result_by_id.get(parent_id, "not_executed")
            if verify_result == "partial_accept":
                reason = "parent_partial_accept"
            elif verify_result == "reject_at_first_token":
                reason = "parent_rejected"
            elif verify_result == "full_accept":
                reason = "parent_full_accept_missing_ready"
            elif verify_result == "not_executed":
                reason = "parent_verify_pending"
            else:
                reason = "parent_not_full_accept"
            if reason != "parent_verify_pending":
                parent_invalidated_ids.append(parent_id)
            status_by_id[child_id] = "CHILD_INVALIDATED_PARENT_REJECT" if reason in {
                "parent_partial_accept",
                "parent_rejected",
                "parent_not_full_accept",
            } else "CHILD_DROPPED"
            reason_by_id[child_id] = reason
            invalidated_ids.append(child_id)
            invalidated_reason_by_id[child_id] = reason
            if reason != "parent_verify_pending":
                cascade_ids.append(child_id)
                cascade_reason_by_id[child_id] = reason
                cascade_depth_by_id[child_id] = depth

        trace_record["rolling_parent_verified_proposal_ids"] = sorted(parent_verified)
        trace_record["rolling_parent_full_accept_proposal_ids"] = sorted(parent_full)
        trace_record["rolling_parent_partial_reject_proposal_ids"] = sorted(parent_partial)
        trace_record["rolling_parent_invalidated_proposal_ids"] = sorted(set(parent_invalidated_ids))
        trace_record["rolling_child_ready_after_parent_full_accept_proposal_ids"] = sorted(set(ready_child_ids))
        trace_record["rolling_child_invalidated_proposal_ids"] = sorted(set(invalidated_ids))
        trace_record["rolling_child_invalidated_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(invalidated_reason_by_id.items())
        }
        trace_record["rolling_cascade_discard_root_proposal_ids"] = sorted(
            set(root_by_id.get(pid, parent_by_id.get(pid, -1)) for pid in cascade_ids)
        )
        trace_record["rolling_cascade_discarded_proposal_ids"] = sorted(set(cascade_ids))
        trace_record["rolling_cascade_discard_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(cascade_reason_by_id.items())
        }
        trace_record["rolling_cascade_discard_depth_by_proposal_id"] = {
            str(proposal_id): int(depth) for proposal_id, depth in sorted(cascade_depth_by_id.items())
        }
        trace_record["rolling_cascade_discard_count"] = len(set(cascade_ids))
        trace_record["rolling_child_ready_shadow_proposal_count"] = len(set(ready_child_ids))
        trace_record["rolling_child_ready_shadow_token_count"] = len(set(ready_child_ids)) * int(self.gamma)
        trace_record["rolling_child_invalidated_count"] = len(set(invalidated_ids))
        trace_record["rolling_child_verified_without_parent_full_accept_count"] = child_verified_without_parent
        trace_record["rolling_child_committed_without_parent_full_accept_count"] = child_committed_without_parent
        trace_record["rolling_depth2_real_commit_count"] = 0
        trace_record["rolling_depth_gt1_real_commit_count"] = 0
        trace_record["rolling_chain_status_by_proposal_id"] = {
            str(proposal_id): status for proposal_id, status in sorted(status_by_id.items())
        }
        trace_record["rolling_chain_status_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(reason_by_id.items())
        }
        reason_counts: dict[str, int] = {}
        for reason in invalidated_reason_by_id.values():
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        trace_record["rolling_drop_reason_counts"] = dict(sorted(reason_counts.items()))

    def _continuous_set_common_trace(
        self,
        trace_record: dict,
        max_depth: int,
        max_requests: int,
        token_per_request: int,
        max_tokens_per_step: int,
        depth_by_id: dict[int, int],
    ) -> None:
        trace_record["enable_continuous_eager_dry_run"] = True
        trace_record["enable_continuous_eager_verify_apply_dry_run"] = bool(
            self._continuous_eager_verify_apply_dry_run_enabled()
        )
        trace_record["continuous_eager_dry_run_enabled"] = True
        trace_record["continuous_eager_source"] = CONTINUOUS_EAGER_DRY_RUN_SOURCE
        trace_record["continuous_eager_parent_source"] = CONTINUOUS_EAGER_PARENT_SOURCE
        trace_record["continuous_eager_execution_stage"] = "verify_apply_dry_run"
        trace_record["continuous_shadow_stage"] = "verify_apply_dry_run"
        trace_record["max_continuous_eager_chain_depth"] = int(max_depth)
        trace_record["max_continuous_depth_configured"] = int(max_depth)
        trace_record["max_continuous_depth_observed"] = max(depth_by_id.values(), default=0)
        trace_record["max_continuous_eager_requests_per_step"] = int(max_requests)
        trace_record["max_continuous_eager_tokens_per_step"] = int(max_tokens_per_step)
        trace_record["max_continuous_eager_tokens_per_request"] = int(token_per_request)

    def _run_continuous_eager_draft_shadow_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        committed_ids: list[int],
        committed_seq_ids: list[int],
        known_by_id: dict[int, EagerProposal | ReadyEagerProposal],
        seq_by_id: dict[int, Sequence],
        plan_context: dict[str, set[int]],
    ) -> list[EagerProposal]:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        max_depth, max_requests, token_per_request, max_tokens_per_step = self._continuous_eager_limits()
        proposals: list[EagerProposal] = []
        parent_by_id: dict[int, int] = {}
        root_by_id: dict[int, int] = {}
        depth_by_id: dict[int, int] = {}
        chain_index_by_id: dict[int, int] = {}
        token_count_by_id: dict[int, int] = {}
        parent_source_by_id: dict[int, str] = {}
        candidate_ids: list[int] = []
        candidate_seq_ids: list[int] = []
        not_ready_ids: list[int] = []
        not_ready_reason_by_id: dict[int, str] = {}
        duplicate_ids: list[int] = []
        parent_not_ready_ids: list[int] = []
        parent_shadow_not_committed_ids: list[int] = []
        true_frontier_mismatch_ids: list[int] = []
        frontier_mismatch_ids: list[int] = []
        seq_finished_ids: list[int] = []
        stale_ids: list[int] = []
        invalidated_ids: list[int] = []
        chain_distribution: dict[int, int] = {}
        reason_counts: dict[str, int] = {}
        seen_seq_depth: set[tuple[int, int]] = set()
        token_budget = 0

        def add_not_ready(proposal_id: int, reason: str) -> None:
            not_ready_ids.append(int(proposal_id))
            not_ready_reason_by_id[int(proposal_id)] = str(reason)
            reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0)) + 1
            if reason == "duplicate_continuous_candidate":
                duplicate_ids.append(int(proposal_id))
            elif reason == "parent_shadow_not_committed":
                parent_shadow_not_committed_ids.append(int(proposal_id))
            elif reason in {"parent_not_committed", "parent_shadow_not_ready"}:
                parent_not_ready_ids.append(int(proposal_id))
            elif reason == "frontier_mismatch":
                true_frontier_mismatch_ids.append(int(proposal_id))
                frontier_mismatch_ids.append(int(proposal_id))
            elif reason in {"seq_not_running", "seq_finished"}:
                seq_finished_ids.append(int(proposal_id))
            elif reason in {"span_invalidated"}:
                invalidated_ids.append(int(proposal_id))
            elif reason in {"seq_pre_verify", "missing_parent_proposal", "seq_not_found"}:
                stale_ids.append(int(proposal_id))

        valid_depth1: list[tuple[int, int, int, Sequence, ReadyEagerProposal | EagerProposal]] = []
        parent_pairs = list(dict.fromkeys(zip(committed_ids, committed_seq_ids)))[:max_requests]
        for chain_index, (root_proposal_id, seq_id) in enumerate(parent_pairs):
            root_proposal_id = int(root_proposal_id)
            seq_id = int(seq_id)
            parent = known_by_id.get(root_proposal_id) or self.dual_batch_manager.ready_eager_proposals.by_id(root_proposal_id)
            seq = seq_by_id.get(seq_id)
            for depth in range(1, max_depth + 1):
                proposal_id = self._continuous_shadow_proposal_id(root_proposal_id, depth)
                parent_id = root_proposal_id if depth == 1 else self._continuous_shadow_proposal_id(root_proposal_id, depth - 1)
                parent_by_id[proposal_id] = int(parent_id)
                root_by_id[proposal_id] = int(root_proposal_id)
                depth_by_id[proposal_id] = int(depth)
                chain_index_by_id[proposal_id] = int(chain_index)
                token_count_by_id[proposal_id] = int(token_per_request)
                parent_source_by_id[proposal_id] = (
                    CONTINUOUS_EAGER_PARENT_SOURCE if depth == 1 else CONTINUOUS_EAGER_DRY_RUN_SOURCE
                )

                current_len = -1 if seq is None else int(len(seq))
                reason = None
                if (seq_id, depth) in seen_seq_depth:
                    reason = "duplicate_continuous_candidate"
                elif depth > 1:
                    reason = "parent_shadow_not_committed"
                elif token_per_request != gamma:
                    reason = "continuous_token_limit_not_gamma"
                elif token_budget + token_per_request > max_tokens_per_step:
                    reason = "token_budget_exhausted"
                elif parent is None:
                    reason = "missing_parent_proposal"
                elif seq is None:
                    reason = "seq_not_found"
                elif getattr(seq, "status", None) != SequenceStatus.RUNNING:
                    reason = "seq_not_running"
                elif self.is_request_level_finished(seq, plan_context):
                    reason = "seq_finished"
                elif self.is_speculative_span_invalidated(seq, plan_context):
                    reason = "span_invalidated"
                elif bool(getattr(seq, "pre_verify", True)):
                    reason = "seq_pre_verify"
                elif getattr(parent, "real_commit_step_id", None) is None:
                    reason = "parent_not_committed"
                elif current_len != int(getattr(parent, "base_len", 0)) + int(getattr(parent, "proposal_len", 0)):
                    reason = "frontier_mismatch"

                seen_seq_depth.add((seq_id, depth))
                if reason is not None:
                    add_not_ready(proposal_id, reason)
                    continue
                valid_depth1.append((proposal_id, root_proposal_id, chain_index, seq, parent))
                token_budget += token_per_request

        checkpoints = {
            int(seq.seq_id): self._make_eager_apply_checkpoint(seq)
            for _pid, _root, _idx, seq, _parent in valid_depth1
        }
        generated_by_seq_id: dict[int, list[int]] = {int(seq.seq_id): [] for _pid, _root, _idx, seq, _parent in valid_depth1}
        draft_error: BaseException | None = None
        valid_seqs = [seq for _pid, _root, _idx, seq, _parent in valid_depth1]
        try:
            for _ in range(gamma):
                if not valid_seqs:
                    break
                self._allocate_decode_slots_for_dual(valid_seqs, plan, "continuous_eager_draft_shadow")
                input_ids, positions = self.prepare_pearl_decode(valid_seqs)
                torch.cuda.synchronize()
                logits = self.run_model(input_ids, positions, False)
                if self.tp_params.local_rank == 0:
                    sample_tokens = logits.argmax(dim=-1)
                else:
                    sample_tokens = torch.zeros(
                        len(valid_seqs),
                        dtype=torch.int64,
                        pin_memory=True,
                    ).cuda(non_blocking=True)
                dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
                torch.cuda.synchronize()
                reset_context(self.tp_params)
                for seq, token_id in zip(valid_seqs, sample_tokens.tolist()):
                    int_token_id = int(token_id)
                    seq.append_token(int_token_id)
                    generated_by_seq_id[int(seq.seq_id)].append(int_token_id)

            for proposal_id, root_proposal_id, chain_index, seq, parent in valid_depth1:
                seq_id = int(seq.seq_id)
                checkpoint = checkpoints[seq_id]
                proposal_tokens = [int(token_id) for token_id in generated_by_seq_id[seq_id]]
                to_be_verified = [int(token_id) for token_id in seq.token_ids[-2 * gamma + 1:-gamma + 1]]
                if len(proposal_tokens) != gamma or len(to_be_verified) != gamma:
                    add_not_ready(proposal_id, "invalid_continuous_token_span")
                    continue
                candidate_ids.append(int(proposal_id))
                candidate_seq_ids.append(seq_id)
                chain_distribution[1] = int(chain_distribution.get(1, 0)) + 1
                proposal = EagerProposal(
                    proposal_id=int(proposal_id),
                    seq_id=seq_id,
                    request_id=seq.request_id,
                    lane=LANE_EAGER,
                    parent_proposal_id=int(root_proposal_id),
                    parent_kind=LANE_EAGER,
                    parent_step_id=getattr(parent, "real_commit_step_id", None),
                    source_step_id=0 if plan.step_id is None else int(plan.step_id),
                    source_plan_id=int(plan.plan_id),
                    home_batch_id=-1 if seq.home_batch_id is None else int(seq.home_batch_id),
                    base_len=int(checkpoint["len"]),
                    base_pre_verify=bool(checkpoint["pre_verify"]),
                    base_num_completion_tokens=int(checkpoint["num_completion_tokens"]),
                    proposal_token_ids=proposal_tokens,
                    to_be_verified_token_ids=to_be_verified,
                    proposal_len=gamma,
                    state=EAGER_STATE_READY_TO_VERIFY,
                    valid=True,
                )
                proposals.append(proposal)
                self._draft_sent_eager_proposals_by_id[int(proposal.proposal_id)] = proposal
            self._run_rolling_continuous_child_draft_overlap_dry_run(
                plan,
                trace_record,
                proposals,
                seq_by_id,
                checkpoints,
            )
        except BaseException as exc:
            draft_error = exc
        finally:
            for seq in valid_seqs:
                seq_id = int(seq.seq_id)
                checkpoint = checkpoints[seq_id]
                rollback_len = int(len(seq)) - int(checkpoint["len"])
                if rollback_len > 0:
                    self.scheduler.rollback(seq, rollback_len)
                if not self._sequence_matches_eager_apply_checkpoint(seq, checkpoint):
                    self._restore_eager_apply_checkpoint(seq, checkpoint)

        self._continuous_set_common_trace(trace_record, max_depth, max_requests, token_per_request, max_tokens_per_step, depth_by_id)
        trace_record["continuous_eager_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["continuous_eager_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["continuous_eager_candidate_token_count_by_proposal_id"] = {
            str(proposal_id): int(token_count_by_id[proposal_id]) for proposal_id in sorted(candidate_ids)
        }
        trace_record["continuous_eager_parent_proposal_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_by_id.items())
        }
        trace_record["continuous_eager_chain_depth_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(depth_by_id.items())
        }
        trace_record["continuous_eager_chain_index_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(chain_index_by_id.items())
        }
        trace_record["continuous_eager_root_proposal_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(root_by_id.items())
        }
        trace_record["continuous_eager_parent_source_by_proposal_id"] = {
            str(proposal_id): str(value) for proposal_id, value in sorted(parent_source_by_id.items())
        }
        trace_record["continuous_eager_not_ready_shadow_proposal_ids"] = sorted(set(not_ready_ids))
        trace_record["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(not_ready_reason_by_id.items())
        }
        trace_record["continuous_eager_duplicate_proposal_ids"] = sorted(set(duplicate_ids))
        trace_record["continuous_eager_parent_not_ready_proposal_ids"] = sorted(set(parent_not_ready_ids))
        trace_record["continuous_eager_parent_shadow_not_committed_proposal_ids"] = sorted(
            set(parent_shadow_not_committed_ids)
        )
        trace_record["continuous_eager_true_frontier_mismatch_proposal_ids"] = sorted(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_frontier_mismatch_proposal_ids"] = sorted(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_seq_finished_proposal_ids"] = sorted(set(seq_finished_ids))
        trace_record["continuous_eager_invalidated_proposal_ids"] = sorted(set(invalidated_ids))
        trace_record["continuous_eager_stale_proposal_ids"] = sorted(set(stale_ids))
        trace_record["continuous_eager_candidate_proposal_count"] = len(candidate_ids)
        trace_record["continuous_eager_candidate_token_count"] = sum(
            int(token_count_by_id[proposal_id]) for proposal_id in candidate_ids
        )
        trace_record["continuous_eager_not_ready_shadow_proposal_count"] = len(set(not_ready_ids))
        trace_record["continuous_eager_chain_length_distribution"] = {
            str(depth): count for depth, count in sorted(chain_distribution.items())
        }
        trace_record["continuous_eager_drop_reason_counts"] = dict(sorted(reason_counts.items()))
        trace_record["continuous_parent_shadow_not_committed_count"] = len(set(parent_shadow_not_committed_ids))
        trace_record["continuous_parent_shadow_not_ready_count"] = len(set(parent_not_ready_ids))
        trace_record["continuous_true_frontier_mismatch_count"] = len(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_real_commit_count"] = 0
        self._record_elapsed_ms(trace_record, "continuous_eager_overhead_time_ms", timer_start)
        if draft_error is not None:
            raise draft_error
        return proposals

    def _serialize_continuous_eager_transfer_payload(
        self,
        proposals: list[EagerProposal],
        plan: StepPlan,
    ) -> tuple[list[int], list[int]]:
        base_meta, payload_values = serialize_eager_transfer_payload(
            proposals,
            gamma=int(self.gamma),
            plan_id=int(plan.plan_id),
            step_id=plan.step_id,
        )
        return [
            int(CONTINUOUS_EAGER_TRANSFER_MAGIC),
            int(CONTINUOUS_EAGER_TRANSFER_OP_DRY_RUN),
            *[int(value) for value in base_meta],
        ], payload_values

    def _deserialize_continuous_eager_transfer_payload(
        self,
        meta_values: list[int],
        payload_values: list[int],
    ) -> list[EagerProposal]:
        if len(meta_values) != CONTINUOUS_EAGER_TRANSFER_META_LEN:
            raise ValueError(f"continuous eager transfer meta length mismatch: {len(meta_values)}")
        if int(meta_values[0]) != int(CONTINUOUS_EAGER_TRANSFER_MAGIC):
            raise ValueError(f"continuous eager transfer magic mismatch: got={meta_values[0]}")
        if int(meta_values[1]) != int(CONTINUOUS_EAGER_TRANSFER_OP_DRY_RUN):
            raise ValueError(f"continuous eager transfer op mismatch: got={meta_values[1]}")
        return deserialize_eager_transfer_payload(meta_values[2:], payload_values)

    def _send_continuous_eager_transfer_dry_run(
        self,
        proposals: list[EagerProposal],
        plan: StepPlan,
        trace_record: dict,
    ) -> None:
        meta_values, payload_values = self._serialize_continuous_eager_transfer_payload(proposals, plan)
        trace_record["continuous_eager_result_transfer_zero_steps"] = int(len(proposals) == 0)
        trace_record["continuous_eager_proposal_transfer_payload_len_units"] = int(meta_values[3])
        if self.tp_params.local_rank != 0:
            return
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta_values[3]) > 0:
            payload = torch.tensor(payload_values, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)

    def _receive_continuous_eager_transfer_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
    ) -> list[EagerProposal]:
        meta = torch.zeros(CONTINUOUS_EAGER_TRANSFER_META_LEN, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        meta_values = [int(value) for value in meta.tolist()]
        payload_len = int(meta_values[3])
        payload_values: list[int] = []
        if payload_len > 0:
            payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
            payload_values = [int(value) for value in payload.tolist()]
        proposals = self._deserialize_continuous_eager_transfer_payload(meta_values, payload_values)
        trace_record["continuous_eager_proposal_transfer_payload_len_units"] = payload_len
        return proposals

    def _validate_continuous_target_proposal(
        self,
        proposal: EagerProposal,
        seq: Sequence | None,
        plan_context: dict[str, set[int]],
    ) -> str | None:
        gamma = int(self.gamma)
        parent_id = -1 if proposal.parent_proposal_id is None else int(proposal.parent_proposal_id)
        parent = self.dual_batch_manager.ready_eager_proposals.by_id(parent_id)
        if proposal.parent_kind != LANE_EAGER:
            return "invalid_parent_kind"
        if parent is None:
            return "missing_parent_proposal"
        if getattr(parent, "real_commit_step_id", None) is None:
            return "parent_not_committed"
        if int(parent.seq_id) != int(proposal.seq_id):
            return "parent_seq_mismatch"
        if int(proposal.proposal_len) != gamma or len(proposal.proposal_token_ids) != gamma:
            return "invalid_proposal_len"
        if len(proposal.to_be_verified_token_ids) != gamma:
            return "invalid_to_verify_len"
        if bool(proposal.base_pre_verify):
            return "invalid_base_pre_verify"
        if seq is None:
            return "seq_not_found"
        if getattr(seq, "status", None) != SequenceStatus.RUNNING:
            return "seq_not_running"
        if self.is_request_level_finished(seq, plan_context):
            return "seq_finished"
        if self.is_speculative_span_invalidated(seq, plan_context):
            return "span_invalidated"
        if bool(getattr(seq, "pre_verify", True)):
            return "seq_pre_verify"
        if int(len(seq)) != int(proposal.base_len):
            return "frontier_mismatch"
        return None

    def _run_continuous_eager_target_verify_apply_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        proposals: list[EagerProposal],
        plan_context: dict[str, set[int]],
    ) -> list[dict]:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        max_depth, max_requests, token_per_request, max_tokens_per_step = self._continuous_eager_limits()
        seq_by_id = self._local_sequence_by_id()
        candidate_ids = [int(proposal.proposal_id) for proposal in proposals]
        candidate_seq_ids = [int(proposal.seq_id) for proposal in proposals]
        parent_by_id = {
            int(proposal.proposal_id): (-1 if proposal.parent_proposal_id is None else int(proposal.parent_proposal_id))
            for proposal in proposals
        }
        root_by_id = dict(parent_by_id)
        depth_by_id = {int(proposal.proposal_id): 1 for proposal in proposals}
        token_by_id = {int(proposal.proposal_id): int(proposal.proposal_len) for proposal in proposals}
        parent_source_by_id = {int(proposal.proposal_id): CONTINUOUS_EAGER_PARENT_SOURCE for proposal in proposals}

        executed_proposals: list[EagerProposal] = []
        executed_seqs: list[Sequence] = []
        skipped_ids: list[int] = []
        skip_reason_by_id: dict[int, str] = {}
        not_ready_ids: list[int] = []
        not_ready_reason_by_id: dict[int, str] = {}
        true_frontier_mismatch_ids: list[int] = []
        seq_finished_ids: list[int] = []
        invalidated_ids: list[int] = []
        stale_ids: list[int] = []
        parent_not_ready_ids: list[int] = []
        reason_counts: dict[str, int] = {}

        for proposal in proposals:
            proposal_id = int(proposal.proposal_id)
            seq = seq_by_id.get(int(proposal.seq_id))
            reason = self._validate_continuous_target_proposal(proposal, seq, plan_context)
            if reason is None:
                executed_proposals.append(proposal)
                executed_seqs.append(seq)
            else:
                skipped_ids.append(proposal_id)
                skip_reason_by_id[proposal_id] = reason
                not_ready_ids.append(proposal_id)
                not_ready_reason_by_id[proposal_id] = reason
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                if reason == "frontier_mismatch":
                    true_frontier_mismatch_ids.append(proposal_id)
                elif reason in {"seq_finished", "seq_not_running"}:
                    seq_finished_ids.append(proposal_id)
                elif reason == "span_invalidated":
                    invalidated_ids.append(proposal_id)
                elif reason in {"parent_not_committed", "missing_parent_proposal"}:
                    parent_not_ready_ids.append(proposal_id)
                else:
                    stale_ids.append(proposal_id)

        checkpoints = {int(seq.seq_id): self._make_eager_apply_checkpoint(seq) for seq in executed_seqs}
        results: dict[str, dict[int, int | bool]] = {
            "accepted_len_by_seq_id": {},
            "invalidated_len_by_seq_id": {},
            "reject_position_by_seq_id": {},
            "revised_token_by_seq_id": {},
            "full_accept_by_seq_id": {},
        }
        if executed_seqs:
            self._allocate_decode_slots_for_dual(executed_seqs, plan, "continuous_eager_verify_dry_run")
            input_ids, positions, temp_seqs = self.prepare_pearl_decode(executed_seqs)
            temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
            torch.cuda.synchronize()
            logits = self.run_model(input_ids, positions, False)
            results = self.compute_pearl_verify_result_no_apply(
                logits,
                executed_seqs,
                temperatures,
                executed_proposals,
                gamma=gamma,
                lane=LANE_EAGER,
            )
            torch.cuda.synchronize()

        accept_len_by_id: dict[int, int] = {}
        verify_result_by_id: dict[int, str] = {}
        full_accept_ids: list[int] = []
        partial_reject_ids: list[int] = []
        apply_candidate_ids: list[int] = []
        apply_executed_ids: list[int] = []
        action_by_id: dict[int, str] = {}
        append_by_id: dict[int, int] = {}
        discard_by_id: dict[int, int] = {}
        rollback_ok_by_id: dict[int, bool] = {}
        mutation_by_id: dict[int, bool] = {}
        checkpoint_failed_by_id: dict[int, bool] = {}
        result_items: list[dict] = []

        for proposal, seq in zip(executed_proposals, executed_seqs):
            proposal_id = int(proposal.proposal_id)
            seq_id = int(seq.seq_id)
            accept_len = int(results["accepted_len_by_seq_id"].get(seq_id, 0))
            verify_result = self._verify_result_from_accept_len(accept_len, gamma)
            accept_len_by_id[proposal_id] = accept_len
            verify_result_by_id[proposal_id] = verify_result
            apply_candidate_ids.append(proposal_id)
            if verify_result == "full_accept":
                full_accept_ids.append(proposal_id)
            else:
                partial_reject_ids.append(proposal_id)
                not_ready_ids.append(proposal_id)
                not_ready_reason_by_id[proposal_id] = "continuous_not_full_accept"
                reason_counts["continuous_not_full_accept"] = int(reason_counts.get("continuous_not_full_accept", 0)) + 1

            checkpoint = checkpoints[seq_id]
            appended_count = 0
            checkpoint_ok = True
            rollback_ok = True
            mutation_remaining = False
            if verify_result == "full_accept":
                checkpoint_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
                for token_id in proposal.proposal_token_ids:
                    seq.append_token(int(token_id))
                seq.pre_verify = False
                appended_count = len(proposal.proposal_token_ids)
                action = "append_full_accept_then_rollback"
                apply_executed_ids.append(proposal_id)
                rollback_ok, mutation_remaining = self._rollback_eager_apply_dry_run(seq, checkpoint, appended_count)
            elif verify_result == "partial_accept":
                action = "discard_partial_no_mutation"
            else:
                action = "discard_reject_no_mutation"
            action_by_id[proposal_id] = action
            append_by_id[proposal_id] = int(appended_count)
            discard_by_id[proposal_id] = 0 if appended_count else gamma
            rollback_ok_by_id[proposal_id] = bool(rollback_ok)
            mutation_by_id[proposal_id] = bool(mutation_remaining)
            checkpoint_failed_by_id[proposal_id] = not bool(checkpoint_ok)
            if verify_result == "full_accept" and (not rollback_ok or mutation_remaining or not checkpoint_ok):
                not_ready_ids.append(proposal_id)
                not_ready_reason_by_id[proposal_id] = "continuous_apply_guard_failed"
                reason_counts["continuous_apply_guard_failed"] = int(reason_counts.get("continuous_apply_guard_failed", 0)) + 1

            result_items.append(
                {
                    "proposal_id": proposal_id,
                    "seq_id": seq_id,
                    "request_id": self._numeric_request_id(proposal.request_id),
                    "source_plan_id": int(proposal.source_plan_id),
                    "source_step_id": int(proposal.source_step_id),
                    "schedule_plan_id": int(plan.plan_id),
                    "schedule_step_id": -1 if plan.step_id is None else int(plan.step_id),
                    "takeover_step_id": -1,
                    "verify_plan_id": int(plan.plan_id),
                    "verify_step_id": -1 if plan.step_id is None else int(plan.step_id),
                    "apply_plan_id": int(plan.plan_id),
                    "apply_step_id": -1 if plan.step_id is None else int(plan.step_id),
                    "verify_result": verify_result,
                    "accepted_len": accept_len,
                    "full_accept": verify_result == "full_accept",
                    "reject_position": -1 if verify_result == "full_accept" else accept_len,
                    "invalidated_len": max(0, gamma - accept_len),
                    "revised_token": -1,
                    "proposal_len": gamma,
                    "to_verify_len": gamma,
                    "gamma": gamma,
                    "base_len": int(proposal.base_len),
                    "base_pre_verify": bool(proposal.base_pre_verify),
                    "target_seq_len_at_verify": int(checkpoint["len"]),
                    "target_seq_pre_verify_at_verify": bool(checkpoint["pre_verify"]),
                    "apply_action": action,
                    "append_token_count": int(appended_count),
                    "discarded_token_count": 0 if appended_count else gamma,
                    "rollback_ok": bool(rollback_ok),
                    "mutation_detected": bool(mutation_remaining),
                    "checkpoint_failed": not bool(checkpoint_ok),
                    "source": CONTINUOUS_EAGER_DRY_RUN_SOURCE,
                    "parent_proposal_id": -1 if proposal.parent_proposal_id is None else int(proposal.parent_proposal_id),
                    "chain_depth": 1,
                }
            )

        # Target-side full accepts are only shadow-ready after result transfer
        # and draft sync-apply validation complete. Keep this target record at
        # the verify/apply stage so accounting does not count a pre-sync result.
        ready_ids: list[int] = []
        seq_by_proposal_id = {int(proposal.proposal_id): int(proposal.seq_id) for proposal in proposals}
        ready_seq_ids = [int(seq_by_proposal_id.get(proposal_id, -1)) for proposal_id in ready_ids]
        ready_token_by_id = {proposal_id: gamma for proposal_id in ready_ids}
        self._continuous_set_common_trace(trace_record, max_depth, max_requests, token_per_request, max_tokens_per_step, depth_by_id)
        trace_record["continuous_eager_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["continuous_eager_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["continuous_eager_candidate_token_count_by_proposal_id"] = {
            str(proposal_id): int(token_by_id.get(proposal_id, gamma)) for proposal_id in sorted(candidate_ids)
        }
        trace_record["continuous_eager_parent_proposal_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(parent_by_id.items())
        }
        trace_record["continuous_eager_chain_depth_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(depth_by_id.items())
        }
        trace_record["continuous_eager_root_proposal_id_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(root_by_id.items())
        }
        trace_record["continuous_eager_parent_source_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in sorted(parent_source_by_id.items())
        }
        trace_record["continuous_eager_verify_dry_run_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["continuous_eager_verify_dry_run_executed_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in executed_proposals
        ]
        trace_record["continuous_eager_verify_dry_run_skipped_proposal_ids"] = list(skipped_ids)
        trace_record["continuous_eager_verify_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(skip_reason_by_id.items())
        }
        trace_record["continuous_eager_verified_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in executed_proposals
        ]
        trace_record["continuous_eager_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(verify_result_by_id.items())
        }
        trace_record["continuous_eager_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(accept_len_by_id.items())
        }
        trace_record["continuous_eager_full_accept_proposal_ids"] = list(full_accept_ids)
        trace_record["continuous_eager_partial_reject_proposal_ids"] = list(partial_reject_ids)
        trace_record["continuous_eager_verified_token_count"] = len(executed_proposals) * gamma
        trace_record["continuous_eager_full_accept_token_count"] = len(full_accept_ids) * gamma
        trace_record["continuous_eager_partial_reject_token_count"] = len(partial_reject_ids) * gamma
        trace_record["continuous_eager_apply_dry_run_candidate_proposal_ids"] = list(apply_candidate_ids)
        trace_record["continuous_eager_apply_dry_run_executed_proposal_ids"] = list(apply_executed_ids)
        trace_record["continuous_eager_apply_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(action_by_id.items())
        }
        trace_record["continuous_eager_apply_append_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(append_by_id.items())
        }
        trace_record["continuous_eager_apply_discarded_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(discard_by_id.items())
        }
        trace_record["continuous_eager_apply_rollback_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(rollback_ok_by_id.items())
        }
        trace_record["continuous_eager_apply_mutation_detected_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(mutation_by_id.items())
        }
        trace_record["continuous_eager_apply_checkpoint_failed_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(checkpoint_failed_by_id.items())
        }
        trace_record["continuous_eager_commit_ready_shadow_proposal_ids"] = list(ready_ids)
        trace_record["continuous_eager_commit_ready_shadow_seq_ids"] = list(ready_seq_ids)
        trace_record["continuous_eager_commit_ready_shadow_token_count_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(ready_token_by_id.items())
        }
        trace_record["continuous_eager_not_ready_shadow_proposal_ids"] = sorted(set(not_ready_ids))
        trace_record["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(not_ready_reason_by_id.items())
        }
        trace_record["continuous_eager_parent_not_ready_proposal_ids"] = sorted(set(parent_not_ready_ids))
        trace_record["continuous_eager_true_frontier_mismatch_proposal_ids"] = sorted(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_frontier_mismatch_proposal_ids"] = sorted(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_seq_finished_proposal_ids"] = sorted(set(seq_finished_ids))
        trace_record["continuous_eager_invalidated_proposal_ids"] = sorted(set(invalidated_ids))
        trace_record["continuous_eager_stale_proposal_ids"] = sorted(set(stale_ids))
        trace_record["continuous_eager_candidate_proposal_count"] = len(candidate_ids)
        trace_record["continuous_eager_candidate_token_count"] = len(candidate_ids) * gamma
        trace_record["continuous_eager_verified_proposal_count"] = len(executed_proposals)
        trace_record["continuous_eager_full_accept_proposal_count"] = len(full_accept_ids)
        trace_record["continuous_eager_commit_ready_shadow_proposal_count"] = len(ready_ids)
        trace_record["continuous_eager_commit_ready_shadow_token_count"] = len(ready_ids) * gamma
        trace_record["continuous_eager_not_ready_shadow_proposal_count"] = len(set(not_ready_ids))
        trace_record["continuous_eager_chain_length_distribution"] = {"1": len(candidate_ids)} if candidate_ids else {}
        trace_record["continuous_eager_drop_reason_counts"] = dict(sorted(reason_counts.items()))
        trace_record["continuous_parent_shadow_not_committed_count"] = 0
        trace_record["continuous_parent_shadow_not_ready_count"] = len(set(parent_not_ready_ids))
        trace_record["continuous_true_frontier_mismatch_count"] = len(set(true_frontier_mismatch_ids))
        trace_record["continuous_eager_mutation_detected_count"] = sum(1 for value in mutation_by_id.values() if bool(value))
        trace_record["continuous_eager_real_commit_count"] = 0
        trace_record["continuous_eager_verify_apply_zero_candidate_steps"] = int(len(proposals) == 0)
        if self._rolling_continuous_eager_dry_run_enabled():
            max_rolling_depth, _max_children, _max_seqs = self._rolling_continuous_limits()
            self._set_rolling_common_trace(trace_record, max_rolling_depth)
            rolling_seq_ids = set(candidate_seq_ids)
            normal_excluded, normal_conflicts = self._rolling_normal_lane_conflicts(trace_record, rolling_seq_ids)
            trace_record["target_rolling_eager_verify_proposal_ids"] = list(candidate_ids)
            trace_record["target_rolling_eager_verify_seq_ids"] = list(candidate_seq_ids)
            trace_record["rolling_parent_verified_proposal_ids"] = list(accept_len_by_id)
            trace_record["rolling_parent_full_accept_proposal_ids"] = list(full_accept_ids)
            trace_record["rolling_parent_partial_reject_proposal_ids"] = list(partial_reject_ids)
            trace_record["rolling_parent_invalidated_proposal_ids"] = sorted(set(skipped_ids))
            trace_record["rolling_normal_lane_excluded_seq_ids"] = normal_excluded
            trace_record["rolling_normal_lane_conflict_seq_ids"] = normal_conflicts
            trace_record["rolling_normal_lane_conflict_count"] = len(normal_conflicts)
            trace_record["rolling_depth2_real_commit_count"] = 0
            trace_record["rolling_depth_gt1_real_commit_count"] = 0
        self._record_elapsed_ms(trace_record, "continuous_eager_overhead_time_ms", timer_start)
        return result_items

    def _serialize_continuous_eager_result_transfer_payload(
        self,
        results: list[dict],
        plan: StepPlan,
    ) -> tuple[list[int], list[int]]:
        payload_values: list[int] = []
        for result in results:
            payload_values.extend(
                [
                    int(result["proposal_id"]),
                    int(result["seq_id"]),
                    int(result.get("parent_proposal_id", -1)),
                    int(result.get("chain_depth", 1)),
                    int(result["proposal_len"]),
                    int(self._result_verify_code(result.get("verify_result", "unknown"))),
                    int(result["accepted_len"]),
                    int(self._result_action_code(result.get("apply_action", "unknown"))),
                    int(result.get("append_token_count", 0)),
                    int(result.get("discarded_token_count", 0)),
                    int(bool(result.get("rollback_ok", False))),
                    int(bool(result.get("mutation_detected", True))),
                    int(bool(result.get("checkpoint_failed", True))),
                ]
            )
        meta_values = [
            int(CONTINUOUS_EAGER_RESULT_TRANSFER_MAGIC),
            int(CONTINUOUS_EAGER_RESULT_TRANSFER_OP_COMPACT_V1),
            len(results),
            len(payload_values),
            int(self.gamma),
            int(plan.plan_id),
            -1 if plan.step_id is None else int(plan.step_id),
        ]
        return meta_values, payload_values

    def _deserialize_continuous_eager_result_transfer_payload(
        self,
        meta_values: list[int],
        payload_values: list[int],
        known_by_id: dict[int, EagerProposal],
    ) -> list[dict]:
        meta_values = [int(value) for value in meta_values]
        if len(meta_values) != CONTINUOUS_EAGER_RESULT_TRANSFER_META_LEN:
            raise ValueError(f"continuous eager result transfer meta length mismatch: {len(meta_values)}")
        magic, op_type, num_results, payload_len, gamma, plan_id, step_id = meta_values
        if int(magic) != int(CONTINUOUS_EAGER_RESULT_TRANSFER_MAGIC):
            raise ValueError(f"continuous eager result transfer magic mismatch: got={magic}")
        if int(op_type) != int(CONTINUOUS_EAGER_RESULT_TRANSFER_OP_COMPACT_V1):
            raise ValueError(f"continuous eager result transfer op mismatch: got={op_type}")
        expected_len = int(num_results) * CONTINUOUS_EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH
        if int(payload_len) != len(payload_values) or len(payload_values) != expected_len:
            raise ValueError(
                "continuous eager result transfer payload length mismatch: "
                f"num_results={num_results}, payload_len={payload_len}, actual={len(payload_values)}"
            )
        results: list[dict] = []
        for idx in range(int(num_results)):
            base = idx * CONTINUOUS_EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH
            (
                proposal_id,
                seq_id,
                parent_proposal_id,
                chain_depth,
                proposal_len,
                verify_result_code,
                accepted_len,
                apply_action_code,
                append_token_count,
                discarded_token_count,
                rollback_ok,
                mutation_detected,
                checkpoint_failed,
            ) = payload_values[base:base + CONTINUOUS_EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH]
            proposal = known_by_id.get(int(proposal_id))
            request_id = self._numeric_request_id(proposal.request_id) if proposal is not None else -1
            base_len = int(getattr(proposal, "base_len", -1)) if proposal is not None else -1
            base_pre_verify = bool(getattr(proposal, "base_pre_verify", False)) if proposal is not None else False
            verify_result = self._result_verify_from_code(int(verify_result_code))
            full_accept = verify_result == "full_accept"
            accepted_len = int(accepted_len)
            proposal_len = int(proposal_len)
            raw_source_plan_id = getattr(proposal, "source_plan_id", plan_id) if proposal is not None else plan_id
            raw_source_step_id = getattr(proposal, "source_step_id", step_id) if proposal is not None else step_id
            source_plan_id = int(plan_id if raw_source_plan_id is None else raw_source_plan_id)
            source_step_id = int(step_id if raw_source_step_id is None else raw_source_step_id)
            results.append(
                {
                    "proposal_id": int(proposal_id),
                    "seq_id": int(seq_id),
                    "parent_proposal_id": int(parent_proposal_id),
                    "chain_depth": int(chain_depth),
                    "request_id": request_id,
                    "source_plan_id": int(source_plan_id),
                    "source_step_id": int(source_step_id),
                    "schedule_plan_id": int(plan_id),
                    "schedule_step_id": int(step_id),
                    "takeover_step_id": -1,
                    "verify_plan_id": int(plan_id),
                    "verify_step_id": int(step_id),
                    "apply_plan_id": int(plan_id),
                    "apply_step_id": int(step_id),
                    "verify_result": str(verify_result),
                    "accepted_len": accepted_len,
                    "full_accept": bool(full_accept),
                    "reject_position": -1 if full_accept else max(0, accepted_len),
                    "invalidated_len": max(0, proposal_len - max(0, accepted_len)),
                    "revised_token": -1,
                    "proposal_len": proposal_len,
                    "to_verify_len": proposal_len,
                    "gamma": int(gamma),
                    "base_len": base_len,
                    "base_pre_verify": bool(base_pre_verify),
                    "target_seq_len_at_verify": base_len,
                    "target_seq_pre_verify_at_verify": bool(base_pre_verify),
                    "apply_action": self._result_action_from_code(int(apply_action_code)),
                    "append_token_count": int(append_token_count),
                    "discarded_token_count": int(discarded_token_count),
                    "rollback_ok": bool(rollback_ok),
                    "mutation_detected": bool(mutation_detected),
                    "checkpoint_failed": bool(checkpoint_failed),
                    "source": CONTINUOUS_EAGER_DRY_RUN_SOURCE,
                }
            )
        return results

    def _send_continuous_eager_result_transfer_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        results: list[dict],
    ) -> None:
        timer_start = time.perf_counter()
        meta_values, payload_values = self._serialize_continuous_eager_result_transfer_payload(results, plan)
        num_results = int(meta_values[2])
        payload_len = int(meta_values[3])
        trace_record["continuous_eager_result_transfer_sent_proposal_ids"] = [
            int(result["proposal_id"]) for result in results
        ]
        trace_record["continuous_eager_result_transfer_protocol"] = CONTINUOUS_EAGER_RESULT_TRANSFER_PROTOCOL_COMPACT_V1
        trace_record["continuous_eager_result_transfer_compacted"] = True
        trace_record["continuous_eager_result_transfer_payload_len_units_before_compact"] = (
            int(num_results) * EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH
        )
        trace_record["continuous_eager_result_transfer_payload_len_units"] = payload_len
        trace_record["continuous_eager_result_transfer_zero_steps"] = int(num_results == 0)
        trace_record["continuous_zero_result_fast_path_count"] = int(num_results == 0)
        trace_record["continuous_eager_result_transfer_sent_count"] = num_results
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
        if payload_len > 0:
            if self.rank == self.global_config.target_config.master_rank:
                payload = torch.tensor(payload_values, dtype=torch.int64, device="cuda")
            else:
                payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.target_config.master_rank, group=self.verify_group)
        self._record_elapsed_ms(trace_record, "continuous_eager_result_transfer_time_ms", timer_start)

    def _receive_continuous_eager_result_transfer_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
    ) -> list[dict]:
        meta = torch.zeros(CONTINUOUS_EAGER_RESULT_TRANSFER_META_LEN, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.target_config.master_rank, group=self.verify_group)
        meta_values = [int(value) for value in meta.tolist()]
        payload_len = int(meta_values[3])
        payload_values: list[int] = []
        if payload_len > 0:
            payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.target_config.master_rank, group=self.verify_group)
            payload_values = [int(value) for value in payload.tolist()]
        known_by_id = dict(self._draft_sent_eager_proposals_by_id)
        results = self._deserialize_continuous_eager_result_transfer_payload(meta_values, payload_values, known_by_id)
        seq_by_id = self._local_sequence_by_id()
        validated: list[dict] = []
        invalid: list[dict] = []
        validation_reason_by_id: dict[int, str] = {}
        for result in results:
            proposal_id = int(result["proposal_id"])
            reason = self._validate_eager_result_on_draft(
                result,
                known_by_id,
                seq_by_id.get(int(result["seq_id"])),
                int(meta_values[4]),
                CONTINUOUS_EAGER_DRY_RUN_SOURCE,
            )
            validation_reason_by_id[proposal_id] = reason
            if reason == "ok":
                validated.append(result)
            else:
                invalid.append(result)
        trace_record["continuous_eager_result_transfer_received_proposal_ids"] = [
            int(result["proposal_id"]) for result in results
        ]
        trace_record["continuous_eager_result_transfer_validated_proposal_ids"] = [
            int(result["proposal_id"]) for result in validated
        ]
        trace_record["continuous_eager_result_transfer_invalid_proposal_ids"] = [
            int(result["proposal_id"]) for result in invalid
        ]
        trace_record["continuous_eager_result_transfer_validation_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(validation_reason_by_id.items())
        }
        trace_record["continuous_eager_result_transfer_protocol"] = CONTINUOUS_EAGER_RESULT_TRANSFER_PROTOCOL_COMPACT_V1
        trace_record["continuous_eager_result_transfer_compacted"] = True
        trace_record["continuous_eager_result_transfer_payload_len_units_before_compact"] = (
            int(len(results)) * EAGER_RESULT_TRANSFER_PAYLOAD_WIDTH
        )
        trace_record["continuous_eager_result_transfer_payload_len_units"] = payload_len
        trace_record["continuous_eager_result_transfer_zero_steps"] = int(len(results) == 0)
        trace_record["continuous_zero_result_fast_path_count"] = int(len(results) == 0)
        trace_record["continuous_eager_result_transfer_received_count"] = len(results)
        trace_record["continuous_eager_result_transfer_validated_count"] = len(validated)
        trace_record["continuous_eager_result_transfer_invalid_count"] = len(invalid)
        return validated

    def _run_continuous_eager_sync_apply_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        validated_results: list[dict],
    ) -> None:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        seq_by_id = self._local_sequence_by_id()
        known_by_id = dict(self._draft_sent_eager_proposals_by_id)
        candidate_ids = [int(result["proposal_id"]) for result in validated_results]
        candidate_seq_ids = [int(result["seq_id"]) for result in validated_results]
        executed_ids: list[int] = []
        ready_ids: list[int] = []
        ready_seq_ids: list[int] = []
        not_ready_ids: list[int] = []
        not_ready_reason_by_id: dict[int, str] = {}
        target_action_by_id: dict[int, str] = {}
        draft_action_by_id: dict[int, str] = {}
        target_result_by_id: dict[int, str] = {}
        draft_result_by_id: dict[int, str] = {}
        target_accept_by_id: dict[int, int] = {}
        draft_accept_by_id: dict[int, int] = {}
        action_match_by_id: dict[int, bool] = {}
        result_match_by_id: dict[int, bool] = {}
        accept_match_by_id: dict[int, bool] = {}
        rollback_ok_by_id: dict[int, bool] = {}
        mutation_by_id: dict[int, bool] = {}
        checkpoint_failed_by_id: dict[int, bool] = {}
        append_by_id: dict[int, int] = {}
        discard_by_id: dict[int, int] = {}
        reason_counts: dict[str, int] = {}

        for result in validated_results:
            proposal_id = int(result["proposal_id"])
            seq_id = int(result["seq_id"])
            proposal = known_by_id.get(proposal_id)
            seq = seq_by_id.get(seq_id)
            verify_result = str(result.get("verify_result", "unknown"))
            accept_len = int(result.get("accepted_len", -1))
            target_action = str(result.get("apply_action", "unknown"))
            draft_action = target_action
            target_action_by_id[proposal_id] = target_action
            draft_action_by_id[proposal_id] = draft_action
            target_result_by_id[proposal_id] = verify_result
            draft_result_by_id[proposal_id] = verify_result
            target_accept_by_id[proposal_id] = accept_len
            draft_accept_by_id[proposal_id] = accept_len
            action_match_by_id[proposal_id] = True
            result_match_by_id[proposal_id] = True
            accept_match_by_id[proposal_id] = True
            append_by_id[proposal_id] = 0
            discard_by_id[proposal_id] = int(result.get("discarded_token_count", 0))
            rollback_ok_by_id[proposal_id] = True
            mutation_by_id[proposal_id] = False
            checkpoint_failed_by_id[proposal_id] = False

            reason = None
            if proposal is None:
                reason = "missing_local_proposal"
            elif seq is None:
                reason = "seq_not_found"
            elif int(getattr(proposal, "seq_id", -1)) != seq_id:
                reason = "seq_id_mismatch"
            elif verify_result == "full_accept" and target_action != "append_full_accept_then_rollback":
                reason = "action_mismatch"
            elif verify_result != "full_accept":
                reason = "continuous_not_full_accept"
            elif not bool(result.get("rollback_ok", False)):
                reason = "target_rollback_failed"
            elif bool(result.get("mutation_detected", True)):
                reason = "target_mutation_detected"
            elif bool(result.get("checkpoint_failed", True)):
                reason = "target_checkpoint_failed"
            elif int(result.get("proposal_len", -1)) != gamma or accept_len != gamma:
                reason = "invalid_accept_len"

            if reason is None:
                checkpoint = self._make_eager_apply_checkpoint(seq)
                checkpoint_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
                for token_id in proposal.proposal_token_ids:
                    seq.append_token(int(token_id))
                seq.pre_verify = False
                appended_count = len(proposal.proposal_token_ids)
                self._restore_eager_apply_checkpoint(seq, checkpoint)
                rollback_ok = self._sequence_matches_eager_apply_checkpoint(seq, checkpoint)
                mutation_remaining = not rollback_ok
                append_by_id[proposal_id] = appended_count
                discard_by_id[proposal_id] = 0
                rollback_ok_by_id[proposal_id] = rollback_ok
                mutation_by_id[proposal_id] = mutation_remaining
                checkpoint_failed_by_id[proposal_id] = not checkpoint_ok
                if checkpoint_ok and rollback_ok and not mutation_remaining:
                    ready_ids.append(proposal_id)
                    ready_seq_ids.append(seq_id)
                    executed_ids.append(proposal_id)
                else:
                    reason = "continuous_sync_apply_guard_failed"

            if reason is not None:
                not_ready_ids.append(proposal_id)
                not_ready_reason_by_id[proposal_id] = reason
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                if proposal_id not in executed_ids and proposal is not None:
                    executed_ids.append(proposal_id)

        trace_record["continuous_eager_sync_apply_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["continuous_eager_sync_apply_executed_proposal_ids"] = list(executed_ids)
        trace_record["continuous_eager_sync_apply_action_match_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(action_match_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_result_match_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(result_match_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_accept_len_match_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(accept_match_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_rollback_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(rollback_ok_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_mutation_detected_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(mutation_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_checkpoint_failed_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(checkpoint_failed_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_target_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(target_action_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_draft_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(draft_action_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_target_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(target_result_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_draft_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(draft_result_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_target_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(target_accept_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_draft_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(draft_accept_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_append_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(append_by_id.items())
        }
        trace_record["continuous_eager_sync_apply_discarded_tokens_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(discard_by_id.items())
        }
        trace_record["continuous_eager_commit_ready_shadow_proposal_ids"] = list(ready_ids)
        trace_record["continuous_eager_commit_ready_shadow_seq_ids"] = list(ready_seq_ids)
        trace_record["continuous_eager_commit_ready_shadow_token_count_by_proposal_id"] = {
            str(proposal_id): gamma for proposal_id in ready_ids
        }
        existing_not_ready = set(int(proposal_id) for proposal_id in trace_record.get("continuous_eager_not_ready_shadow_proposal_ids", []))
        existing_reason = dict(trace_record.get("continuous_eager_not_ready_shadow_reason_by_proposal_id", {}))
        for proposal_id, reason in not_ready_reason_by_id.items():
            existing_not_ready.add(proposal_id)
            existing_reason[str(proposal_id)] = reason
        trace_record["continuous_eager_not_ready_shadow_proposal_ids"] = sorted(existing_not_ready)
        trace_record["continuous_eager_not_ready_shadow_reason_by_proposal_id"] = existing_reason
        merged_counts = dict(trace_record.get("continuous_eager_drop_reason_counts", {}))
        for reason, count in reason_counts.items():
            merged_counts[reason] = int(merged_counts.get(reason, 0)) + int(count)
        trace_record["continuous_eager_drop_reason_counts"] = dict(sorted(merged_counts.items()))
        trace_record["continuous_eager_commit_ready_shadow_proposal_count"] = len(ready_ids)
        trace_record["continuous_eager_commit_ready_shadow_token_count"] = len(ready_ids) * gamma
        trace_record["continuous_eager_not_ready_shadow_proposal_count"] = len(existing_not_ready)
        trace_record["continuous_eager_mutation_detected_count"] = sum(1 for value in mutation_by_id.values() if bool(value))
        trace_record["continuous_eager_real_commit_count"] = 0
        trace_record["continuous_eager_sync_apply_zero_steps"] = int(len(validated_results) == 0)
        self._record_elapsed_ms(trace_record, "continuous_eager_sync_apply_dry_run_time_ms", timer_start)


    def _continuous_commit_depth1_decisions_from_trace(self, trace_record: dict) -> list[dict]:
        ready_ids = [
            int(proposal_id)
            for proposal_id in trace_record.get("continuous_eager_commit_ready_shadow_proposal_ids", [])
        ]
        ready_seq_ids = [
            int(seq_id)
            for seq_id in trace_record.get("continuous_eager_commit_ready_shadow_seq_ids", [])
        ]
        token_by_id = trace_record.get("continuous_eager_commit_ready_shadow_token_count_by_proposal_id", {})
        accept_by_id = trace_record.get("continuous_eager_sync_apply_draft_accept_len_by_proposal_id", {})
        result_by_id = trace_record.get("continuous_eager_sync_apply_draft_verify_result_by_proposal_id", {})
        action_by_id = trace_record.get("continuous_eager_sync_apply_draft_action_by_proposal_id", {})
        decisions = []
        for index, proposal_id in enumerate(ready_ids):
            seq_id = ready_seq_ids[index] if index < len(ready_seq_ids) else -1
            decisions.append(
                {
                    "proposal_id": int(proposal_id),
                    "seq_id": int(seq_id),
                    "token_count": int(self._trace_map_get(token_by_id, proposal_id, self.gamma)),
                    "accept_len": int(self._trace_map_get(accept_by_id, proposal_id, self.gamma)),
                    "action": str(
                        self._trace_map_get(
                            action_by_id,
                            proposal_id,
                            "append_full_accept_then_rollback",
                        )
                    ),
                    "verify_result": str(self._trace_map_get(result_by_id, proposal_id, "full_accept")),
                }
            )
        return decisions

    def _serialize_continuous_eager_commit_depth1_payload(
        self,
        decisions: list[dict],
        plan: StepPlan,
    ) -> tuple[list[int], list[int]]:
        payload_values: list[int] = []
        for decision in decisions:
            payload_values.extend(
                [
                    int(decision["proposal_id"]),
                    int(decision["seq_id"]),
                    int(decision["token_count"]),
                    int(decision["accept_len"]),
                ]
            )
        meta_values = [
            int(CONTINUOUS_EAGER_COMMIT_DEPTH1_MAGIC),
            int(CONTINUOUS_EAGER_COMMIT_DEPTH1_OP),
            int(plan.plan_id),
            -1 if plan.step_id is None else int(plan.step_id),
            len(decisions),
            len(payload_values),
        ]
        return meta_values, payload_values

    def _deserialize_continuous_eager_commit_depth1_payload(
        self,
        meta_values: list[int],
        payload_values: list[int],
    ) -> list[dict]:
        if len(meta_values) != CONTINUOUS_EAGER_COMMIT_DEPTH1_META_LEN:
            raise ValueError(f"continuous eager commit meta length mismatch: {len(meta_values)}")
        if int(meta_values[0]) != int(CONTINUOUS_EAGER_COMMIT_DEPTH1_MAGIC):
            raise ValueError(f"continuous eager commit magic mismatch: got={meta_values[0]}")
        if int(meta_values[1]) != int(CONTINUOUS_EAGER_COMMIT_DEPTH1_OP):
            raise ValueError(f"continuous eager commit op mismatch: got={meta_values[1]}")
        num_decisions = int(meta_values[4])
        payload_len = int(meta_values[5])
        expected_len = num_decisions * CONTINUOUS_EAGER_COMMIT_DEPTH1_PAYLOAD_WIDTH
        if payload_len != expected_len or len(payload_values) != expected_len:
            raise ValueError(
                "continuous eager commit payload length mismatch: "
                f"num_decisions={num_decisions}, payload_len={payload_len}, actual={len(payload_values)}"
            )
        decisions = []
        for index in range(num_decisions):
            base = index * CONTINUOUS_EAGER_COMMIT_DEPTH1_PAYLOAD_WIDTH
            proposal_id, seq_id, token_count, accept_len = payload_values[base:base + 4]
            decisions.append(
                {
                    "proposal_id": int(proposal_id),
                    "seq_id": int(seq_id),
                    "token_count": int(token_count),
                    "accept_len": int(accept_len),
                    "action": "append_full_accept_then_rollback",
                    "verify_result": "full_accept",
                }
            )
        return decisions

    def _send_continuous_eager_commit_depth1_decision(
        self,
        plan: StepPlan,
        trace_record: dict,
        known_by_id: dict[int, EagerProposal],
        seq_by_id: dict[int, Sequence],
    ) -> None:
        timer_start = time.perf_counter()
        decisions = self._continuous_commit_depth1_decisions_from_trace(trace_record)
        meta_values, payload_values = self._serialize_continuous_eager_commit_depth1_payload(decisions, plan)
        trace_record["continuous_eager_commit_decision_broadcast_payload_len_units"] = int(meta_values[5])
        trace_record["continuous_eager_commit_decision_broadcast_count"] = int(meta_values[4])
        trace_record["continuous_eager_commit_decision_broadcast_zero_steps"] = int(int(meta_values[4]) == 0)
        trace_record["continuous_zero_decision_fast_path_count"] = int(int(meta_values[4]) == 0)
        meta = torch.tensor(meta_values, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        if int(meta_values[5]) > 0:
            payload = torch.tensor(payload_values, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        self._record_elapsed_ms(trace_record, "continuous_eager_commit_decision_broadcast_time_ms", timer_start)
        self._run_continuous_eager_commit_depth1_ready_only(
            plan,
            trace_record,
            decisions,
            known_by_id,
            seq_by_id,
            self._eager_transfer_plan_context(plan, trace_record),
            side="draft",
        )

    def _receive_continuous_eager_commit_depth1_decision(
        self,
        plan: StepPlan,
        trace_record: dict,
        known_by_id: dict[int, EagerProposal],
    ) -> None:
        meta = torch.zeros(CONTINUOUS_EAGER_COMMIT_DEPTH1_META_LEN, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        meta_values = [int(value) for value in meta.tolist()]
        payload_len = int(meta_values[5])
        payload_values: list[int] = []
        if payload_len > 0:
            payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
            payload_values = [int(value) for value in payload.tolist()]
        decisions = self._deserialize_continuous_eager_commit_depth1_payload(meta_values, payload_values)
        self._run_continuous_eager_commit_depth1_ready_only(
            plan,
            trace_record,
            decisions,
            known_by_id,
            self._local_sequence_by_id(),
            self._eager_transfer_plan_context(plan, trace_record),
            side="target",
        )

    def _run_continuous_eager_commit_depth1_ready_only(
        self,
        plan: StepPlan,
        trace_record: dict,
        decisions: list[dict],
        known_by_id: dict[int, EagerProposal],
        seq_by_id: dict[int, Sequence],
        plan_context: dict[str, set[int]],
        side: str,
    ) -> None:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        ready_ids = {
            int(proposal_id)
            for proposal_id in trace_record.get("continuous_eager_commit_ready_shadow_proposal_ids", [])
        }
        decision_ids = {int(decision["proposal_id"]) for decision in decisions}
        if not ready_ids and decision_ids:
            ready_ids = set(decision_ids)
            seq_ids_by_decision = [int(decision["seq_id"]) for decision in decisions]
            trace_record["continuous_eager_commit_ready_shadow_proposal_ids"] = sorted(ready_ids)
            trace_record["continuous_eager_commit_ready_shadow_seq_ids"] = seq_ids_by_decision
            trace_record["continuous_eager_commit_ready_shadow_token_count_by_proposal_id"] = {
                str(decision["proposal_id"]): int(decision.get("token_count", gamma))
                for decision in decisions
            }

        token_by_id = trace_record.get("continuous_eager_candidate_token_count_by_proposal_id", {})
        depth_by_id = trace_record.get("continuous_eager_chain_depth_by_proposal_id", {})
        parent_by_id = trace_record.get("continuous_eager_parent_proposal_id_by_proposal_id", {})
        parent_source_by_id = trace_record.get("continuous_eager_parent_source_by_proposal_id", {})
        verify_result_by_id = trace_record.get("continuous_eager_verify_result_by_proposal_id", {})
        accept_len_by_id = trace_record.get("continuous_eager_accept_len_by_proposal_id", {})
        apply_action_by_id = trace_record.get("continuous_eager_apply_action_by_proposal_id", {})
        apply_rollback_by_id = trace_record.get("continuous_eager_apply_rollback_ok_by_proposal_id", {})
        apply_mutation_by_id = trace_record.get("continuous_eager_apply_mutation_detected_by_proposal_id", {})
        apply_checkpoint_by_id = trace_record.get("continuous_eager_apply_checkpoint_failed_by_proposal_id", {})
        sync_action_by_id = trace_record.get("continuous_eager_sync_apply_action_match_by_proposal_id", {})
        sync_result_by_id = trace_record.get("continuous_eager_sync_apply_result_match_by_proposal_id", {})
        sync_accept_by_id = trace_record.get("continuous_eager_sync_apply_accept_len_match_by_proposal_id", {})
        sync_rollback_by_id = trace_record.get("continuous_eager_sync_apply_rollback_ok_by_proposal_id", {})
        sync_mutation_by_id = trace_record.get("continuous_eager_sync_apply_mutation_detected_by_proposal_id", {})
        sync_checkpoint_by_id = trace_record.get("continuous_eager_sync_apply_checkpoint_failed_by_proposal_id", {})
        target_action_by_id = trace_record.get("continuous_eager_sync_apply_target_action_by_proposal_id", {})
        target_result_by_id = trace_record.get("continuous_eager_sync_apply_target_verify_result_by_proposal_id", {})
        target_accept_by_id = trace_record.get("continuous_eager_sync_apply_target_accept_len_by_proposal_id", {})

        candidate_ids = [int(decision["proposal_id"]) for decision in decisions]
        candidate_seq_ids = [int(decision["seq_id"]) for decision in decisions]
        committed_ids: list[int] = []
        committed_seq_ids: list[int] = []
        skipped_ids: list[int] = []
        skip_reason_by_id: dict[int, str] = {}
        precondition_ok_by_id: dict[int, bool] = {}
        precondition_failed_by_id: dict[int, bool] = {}
        precondition_failure_reason_by_id: dict[int, str] = {}
        duplicate_proposal_ids: list[int] = []
        duplicate_seq_ids: list[int] = []
        token_count_by_id: dict[int, int] = {}
        accept_by_id: dict[int, int] = {}
        action_by_id: dict[int, str] = {}
        result_by_id: dict[int, str] = {}
        target_len_before_by_seq: dict[int, int] = {}
        target_len_after_by_seq: dict[int, int] = {}
        draft_len_before_by_seq: dict[int, int] = {}
        draft_len_after_by_seq: dict[int, int] = {}
        len_match_by_seq: dict[int, bool] = {}
        token_match_by_seq: dict[int, bool] = {}
        seen_seq_depth: set[tuple[int, int]] = set()
        depth2_real_commit_count = 0
        unexpected_missing = bool(trace_record.get("missing_buffered_proposal_unexpected_seq_ids") or [])

        for decision in decisions:
            proposal_id = int(decision["proposal_id"])
            seq_id = int(decision["seq_id"])
            proposal = known_by_id.get(proposal_id)
            seq = seq_by_id.get(seq_id)
            token_count = int(decision.get("token_count", self._trace_map_get(token_by_id, proposal_id, gamma)))
            accept_len = int(decision.get("accept_len", self._trace_map_get(target_accept_by_id, proposal_id, gamma)))
            action = str(decision.get("action", self._trace_map_get(target_action_by_id, proposal_id, "append_full_accept_then_rollback")))
            verify_result = str(decision.get("verify_result", self._trace_map_get(target_result_by_id, proposal_id, "full_accept")))
            if proposal_id in ready_ids:
                token_count = int(self._trace_map_get(token_by_id, proposal_id, token_count))
                accept_len = int(self._trace_map_get(accept_len_by_id, proposal_id, accept_len))
                action = str(self._trace_map_get(apply_action_by_id, proposal_id, action))
                verify_result = str(self._trace_map_get(verify_result_by_id, proposal_id, verify_result))
            token_count_by_id[proposal_id] = int(token_count)
            accept_by_id[proposal_id] = int(accept_len)
            action_by_id[proposal_id] = action
            result_by_id[proposal_id] = verify_result

            current_len = -1 if seq is None else int(len(seq))
            target_len_before_by_seq[seq_id] = current_len
            draft_len_before_by_seq[seq_id] = current_len
            proposal_len = -1 if proposal is None else int(getattr(proposal, "proposal_len", -1))
            base_len = -1 if proposal is None else int(getattr(proposal, "base_len", -1))
            proposal_tokens = [] if proposal is None else [int(token_id) for token_id in proposal.proposal_token_ids]
            depth = int(self._trace_map_get(depth_by_id, proposal_id, 0))
            parent_id = int(self._trace_map_get(parent_by_id, proposal_id, -1))
            parent_source = str(self._trace_map_get(parent_source_by_id, proposal_id, ""))
            local_frontier_ok = bool(seq is not None and current_len == base_len)
            local_token_payload_ok = bool(len(proposal_tokens) == proposal_len == token_count == gamma)
            seq_depth_key = (seq_id, depth)

            reason = None
            if proposal_id in self._continuous_eager_committed_proposal_ids:
                reason = "duplicate_proposal_commit"
                duplicate_proposal_ids.append(proposal_id)
            elif seq_depth_key in seen_seq_depth:
                reason = "duplicate_seq_depth_commit"
                duplicate_seq_ids.append(seq_id)
            elif proposal_id not in ready_ids:
                reason = "not_shadow_ready"
            elif depth != 1:
                reason = "depth_not_one"
                if depth > 1:
                    depth2_real_commit_count += 1
            elif parent_source != CONTINUOUS_EAGER_PARENT_SOURCE:
                reason = "bad_parent_source"
            elif parent_id not in self._eager_committed_proposal_ids:
                reason = "parent_not_committed"
            elif proposal is None:
                reason = "missing_local_proposal"
            elif int(getattr(proposal, "seq_id", -1)) != seq_id:
                reason = "seq_id_mismatch"
            elif verify_result != "full_accept":
                reason = "not_full_accept"
            elif action != "append_full_accept_then_rollback":
                reason = "apply_action_not_full_accept"
            elif accept_len != token_count or token_count != proposal_len or token_count != gamma:
                reason = "accept_len_mismatch"
            elif self._trace_map_get(apply_rollback_by_id, proposal_id, True) is not True:
                reason = "apply_rollback_failed"
            elif bool(self._trace_map_get(apply_mutation_by_id, proposal_id, False)):
                reason = "apply_mutation_detected"
            elif bool(self._trace_map_get(apply_checkpoint_by_id, proposal_id, False)):
                reason = "apply_checkpoint_failed"
            elif self._trace_map_get(sync_action_by_id, proposal_id, True) is not True:
                reason = "sync_action_mismatch"
            elif self._trace_map_get(sync_result_by_id, proposal_id, True) is not True:
                reason = "sync_result_mismatch"
            elif self._trace_map_get(sync_accept_by_id, proposal_id, True) is not True:
                reason = "sync_accept_len_mismatch"
            elif self._trace_map_get(sync_rollback_by_id, proposal_id, True) is not True:
                reason = "sync_rollback_failed"
            elif bool(self._trace_map_get(sync_mutation_by_id, proposal_id, False)):
                reason = "sync_mutation_detected"
            elif bool(self._trace_map_get(sync_checkpoint_by_id, proposal_id, False)):
                reason = "sync_checkpoint_failed"
            elif unexpected_missing:
                reason = "unexpected_missing_normal_proposal"
            elif not local_token_payload_ok:
                reason = "token_payload_missing"
            elif seq is None:
                reason = "seq_not_found"
            elif getattr(seq, "status", None) != SequenceStatus.RUNNING:
                reason = "seq_not_running"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_pre_verify"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "span_invalidated"
            elif not local_frontier_ok:
                reason = "frontier_mismatch"

            seen_seq_depth.add(seq_depth_key)
            if reason is not None:
                skipped_ids.append(proposal_id)
                skip_reason_by_id[proposal_id] = reason
                precondition_ok_by_id[proposal_id] = False
                precondition_failed_by_id[proposal_id] = True
                precondition_failure_reason_by_id[proposal_id] = reason
                target_len_after_by_seq[seq_id] = current_len
                draft_len_after_by_seq[seq_id] = current_len
                len_match_by_seq[seq_id] = True
                token_match_by_seq[seq_id] = True
                continue

            for token_id in proposal_tokens:
                seq.append_token(int(token_id))
                self.scheduler.block_manager.may_append(seq)
            seq.pre_verify = False
            seq.record_accepted(token_count)
            setattr(proposal, "real_continuous_commit_step_id", None if plan.step_id is None else int(plan.step_id))
            setattr(proposal, "real_continuous_commit_plan_id", int(plan.plan_id))
            self._continuous_eager_committed_proposal_ids.add(proposal_id)
            self._mark_eager_commit_finished_if_needed(seq, proposal_tokens)
            len_after = int(len(seq))
            target_len_after_by_seq[seq_id] = len_after
            draft_len_after_by_seq[seq_id] = len_after
            len_match_by_seq[seq_id] = len_after == current_len + token_count
            token_match_by_seq[seq_id] = list(seq.token_ids[-token_count:]) == proposal_tokens
            committed_ids.append(proposal_id)
            committed_seq_ids.append(seq_id)
            precondition_ok_by_id[proposal_id] = True
            precondition_failed_by_id[proposal_id] = False

        committed_tokens = sum(int(token_count_by_id.get(proposal_id, 0)) for proposal_id in committed_ids)
        reason_counts: dict[str, int] = {}
        for reason in skip_reason_by_id.values():
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

        committed_set = set(committed_ids)
        trace_record["enable_continuous_eager_commit_depth1_ready_only"] = True
        trace_record["continuous_eager_commit_enabled"] = True
        trace_record["continuous_eager_commit_source"] = CONTINUOUS_EAGER_COMMIT_SOURCE
        trace_record["continuous_eager_commit_side"] = side
        trace_record["continuous_eager_commit_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["continuous_eager_commit_plan_id"] = int(plan.plan_id)
        trace_record["continuous_eager_commit_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["continuous_eager_commit_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["continuous_eager_commit_ready_source_proposal_ids"] = sorted(ready_ids)
        trace_record["continuous_eager_real_committed_proposal_ids"] = list(committed_ids)
        trace_record["continuous_eager_real_committed_seq_ids"] = list(committed_seq_ids)
        trace_record["continuous_eager_real_committed_token_count_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(token_count_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["continuous_eager_real_committed_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(accept_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["continuous_eager_real_commit_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(action_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["continuous_eager_real_commit_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(result_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["continuous_eager_real_commit_skipped_proposal_ids"] = sorted(set(skipped_ids))
        trace_record["continuous_eager_real_commit_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(skip_reason_by_id.items())
        }
        trace_record["continuous_eager_real_commit_precondition_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(precondition_ok_by_id.items())
        }
        trace_record["continuous_eager_real_commit_precondition_failed_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(precondition_failed_by_id.items())
        }
        trace_record["continuous_eager_real_commit_precondition_failure_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(precondition_failure_reason_by_id.items())
        }
        trace_record["continuous_eager_real_commit_duplicate_proposal_ids"] = sorted(set(duplicate_proposal_ids))
        trace_record["continuous_eager_real_commit_duplicate_seq_ids"] = sorted(set(duplicate_seq_ids))
        trace_record["continuous_eager_target_seq_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(target_len_before_by_seq.items())
        }
        trace_record["continuous_eager_target_seq_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(target_len_after_by_seq.items())
        }
        trace_record["continuous_eager_draft_seq_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(draft_len_before_by_seq.items())
        }
        trace_record["continuous_eager_draft_seq_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(draft_len_after_by_seq.items())
        }
        trace_record["continuous_eager_target_draft_len_match_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(len_match_by_seq.items())
        }
        trace_record["continuous_eager_target_draft_token_match_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(token_match_by_seq.items())
        }
        trace_record["continuous_eager_tokens_verified"] = int(committed_tokens)
        trace_record["continuous_eager_tokens_accepted"] = int(committed_tokens)
        trace_record["continuous_eager_tokens_committed"] = int(committed_tokens)
        trace_record["continuous_eager_tokens_rejected"] = 0
        trace_record["continuous_eager_tokens_invalidated"] = 0
        trace_record["continuous_eager_real_committed_proposal_count"] = len(committed_ids)
        trace_record["continuous_eager_real_committed_token_count"] = int(committed_tokens)
        trace_record["continuous_eager_real_commit_count"] = len(committed_ids)
        trace_record["continuous_depth2_real_commit_count"] = int(depth2_real_commit_count)
        trace_record["continuous_eager_real_commit_skip_reason_counts"] = dict(sorted(reason_counts.items()))
        self._record_elapsed_ms(trace_record, "continuous_eager_commit_time_ms", timer_start)



    def _run_eager_commit_ready_only(
        self,
        plan: StepPlan,
        trace_record: dict,
        decisions: list[dict],
        known_by_id: dict[int, EagerProposal | ReadyEagerProposal],
        seq_by_id: dict[int, Sequence],
        plan_context: dict[str, set[int]],
        side: str,
    ) -> None:
        timer_start = time.perf_counter()
        gamma = int(self.gamma)
        readiness_ids = {int(proposal_id) for proposal_id in trace_record.get("eager_commit_ready_proposal_ids", [])}
        decision_ids = {int(decision["proposal_id"]) for decision in decisions}
        if not readiness_ids:
            readiness_ids = set(decision_ids)
        readiness_token_by_id = trace_record.get("eager_commit_ready_token_count_by_proposal_id", {})
        readiness_accept_by_id = trace_record.get("eager_commit_ready_accept_len_by_proposal_id", {})
        readiness_action_by_id = trace_record.get("eager_commit_ready_action_by_proposal_id", {})
        readiness_result_by_id = trace_record.get("eager_commit_ready_verify_result_by_proposal_id", {})
        not_ready_ids = [int(proposal_id) for proposal_id in trace_record.get("eager_commit_not_ready_proposal_ids", [])]
        not_ready_reason_by_id = trace_record.get("eager_commit_not_ready_reason_by_proposal_id", {})
        verify_ok_by_id = trace_record.get("eager_commit_readiness_verify_ok_by_proposal_id", {})
        apply_ok_by_id = trace_record.get("eager_commit_readiness_apply_ok_by_proposal_id", {})
        result_ok_by_id = trace_record.get("eager_commit_readiness_result_transfer_ok_by_proposal_id", {})
        sync_ok_by_id = trace_record.get("eager_commit_readiness_sync_apply_ok_by_proposal_id", {})
        frontier_ok_by_id = trace_record.get("eager_commit_readiness_frontier_ok_by_proposal_id", {})
        token_payload_ok_by_id = trace_record.get("eager_commit_readiness_token_payload_ok_by_proposal_id", {})
        no_mutation_by_id = trace_record.get("eager_commit_readiness_no_mutation_by_proposal_id", {})
        unexpected_missing = bool(trace_record.get("missing_buffered_proposal_unexpected_seq_ids") or [])

        candidate_ids = [int(decision["proposal_id"]) for decision in decisions]
        candidate_seq_ids = [int(decision["seq_id"]) for decision in decisions]
        if side == "draft":
            candidate_ids = list(dict.fromkeys(candidate_ids + not_ready_ids))
            not_ready_seq_ids = [int(seq_id) for seq_id in trace_record.get("eager_commit_not_ready_seq_ids", [])]
            candidate_seq_ids = list(dict.fromkeys(candidate_seq_ids + not_ready_seq_ids))

        committed_ids: list[int] = []
        committed_seq_ids: list[int] = []
        skipped_ids: list[int] = []
        skip_reason_by_id: dict[int, str] = {}
        precondition_ok_by_id: dict[int, bool] = {}
        precondition_failed_by_id: dict[int, bool] = {}
        precondition_failure_reason_by_id: dict[int, str] = {}
        duplicate_proposal_ids: list[int] = []
        duplicate_seq_ids: list[int] = []
        token_count_by_id: dict[int, int] = {}
        accept_len_by_id: dict[int, int] = {}
        action_by_id: dict[int, str] = {}
        result_by_id: dict[int, str] = {}
        target_len_before_by_seq: dict[int, int] = {}
        target_len_after_by_seq: dict[int, int] = {}
        draft_len_before_by_seq: dict[int, int] = {}
        draft_len_after_by_seq: dict[int, int] = {}
        target_draft_len_match_by_seq: dict[int, bool] = {}
        target_draft_token_match_by_seq: dict[int, bool] = {}
        frontier_ok_commit_by_id: dict[int, bool] = {}
        token_payload_ok_commit_by_id: dict[int, bool] = {}
        seen_seq_ids: set[int] = set()

        for proposal_id in not_ready_ids:
            reason = str(self._trace_map_get(not_ready_reason_by_id, proposal_id, "not_ready"))
            skipped_ids.append(proposal_id)
            skip_reason_by_id[proposal_id] = reason
            precondition_ok_by_id[proposal_id] = False
            precondition_failed_by_id[proposal_id] = True
            precondition_failure_reason_by_id[proposal_id] = reason

        for decision in decisions:
            proposal_id = int(decision["proposal_id"])
            seq_id = int(decision["seq_id"])
            proposal = known_by_id.get(proposal_id) or self.dual_batch_manager.ready_eager_proposals.by_id(proposal_id)
            seq = seq_by_id.get(seq_id)
            token_count = int(decision.get("token_count", gamma))
            accept_len = int(decision.get("accept_len", gamma))
            action = str(decision.get("action", "append_full_accept_then_rollback"))
            verify_result = str(decision.get("verify_result", "full_accept"))
            if proposal_id in readiness_ids:
                token_count = int(self._trace_map_get(readiness_token_by_id, proposal_id, token_count))
                accept_len = int(self._trace_map_get(readiness_accept_by_id, proposal_id, accept_len))
                action = str(self._trace_map_get(readiness_action_by_id, proposal_id, action))
                verify_result = str(self._trace_map_get(readiness_result_by_id, proposal_id, verify_result))
            token_count_by_id[proposal_id] = int(token_count)
            accept_len_by_id[proposal_id] = int(accept_len)
            action_by_id[proposal_id] = action
            result_by_id[proposal_id] = verify_result

            current_len = -1 if seq is None else int(len(seq))
            target_len_before_by_seq[seq_id] = current_len
            draft_len_before_by_seq[seq_id] = current_len
            proposal_len = -1 if proposal is None else int(getattr(proposal, "proposal_len", -1))
            base_len = -1 if proposal is None else int(getattr(proposal, "base_len", -1))
            proposal_tokens = [] if proposal is None else [int(token_id) for token_id in proposal.proposal_token_ids]
            local_frontier_ok = bool(seq is not None and current_len == base_len)
            local_token_payload_ok = bool(len(proposal_tokens) == proposal_len == token_count == gamma)
            frontier_ok_commit_by_id[proposal_id] = bool(local_frontier_ok)
            token_payload_ok_commit_by_id[proposal_id] = bool(local_token_payload_ok)

            reason = None
            readiness_guards = (
                verify_ok_by_id,
                apply_ok_by_id,
                result_ok_by_id,
                sync_ok_by_id,
                frontier_ok_by_id,
                token_payload_ok_by_id,
                no_mutation_by_id,
            )
            if proposal_id in self._eager_committed_proposal_ids:
                reason = "duplicate_proposal_commit"
                duplicate_proposal_ids.append(proposal_id)
            elif seq_id in seen_seq_ids:
                reason = "duplicate_seq_commit"
                duplicate_seq_ids.append(seq_id)
            elif proposal_id not in readiness_ids:
                reason = "not_commit_ready"
            elif proposal is None:
                reason = "missing_local_proposal"
            elif int(getattr(proposal, "seq_id", -1)) != seq_id:
                reason = "seq_id_mismatch"
            elif str(getattr(proposal, "state", "")) != READY_EAGER_STATE_CONSUMED_APPLIED:
                reason = "proposal_not_consumed_applied"
            elif getattr(proposal, "takeover_routed_step_id", None) is None:
                reason = "takeover_not_routed"
            elif verify_result != "full_accept":
                reason = "not_full_accept"
            elif action != "append_full_accept_then_rollback":
                reason = "apply_action_not_full_accept"
            elif accept_len != token_count or token_count != proposal_len or token_count != gamma:
                reason = "accept_len_mismatch"
            elif any(self._trace_map_get(mapping, proposal_id, True) is not True for mapping in readiness_guards):
                reason = "readiness_guard_failed"
            elif unexpected_missing:
                reason = "unexpected_missing_normal_proposal"
            elif not local_token_payload_ok:
                reason = "token_payload_missing"
            elif seq is None:
                reason = "seq_not_found"
            elif getattr(seq, "status", None) != SequenceStatus.RUNNING:
                reason = "seq_not_running"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_pre_verify"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "span_invalidated"
            elif not local_frontier_ok:
                reason = "frontier_mismatch"

            seen_seq_ids.add(seq_id)
            if reason is not None:
                skipped_ids.append(proposal_id)
                skip_reason_by_id[proposal_id] = reason
                precondition_ok_by_id[proposal_id] = False
                precondition_failed_by_id[proposal_id] = True
                precondition_failure_reason_by_id[proposal_id] = reason
                target_len_after_by_seq[seq_id] = current_len
                draft_len_after_by_seq[seq_id] = current_len
                target_draft_len_match_by_seq[seq_id] = True
                target_draft_token_match_by_seq[seq_id] = True
                continue

            for token_id in proposal_tokens:
                seq.append_token(int(token_id))
                self.scheduler.block_manager.may_append(seq)
            seq.pre_verify = False
            seq.record_accepted(token_count)
            setattr(proposal, "real_commit_step_id", None if plan.step_id is None else int(plan.step_id))
            self._eager_committed_proposal_ids.add(proposal_id)
            self._mark_eager_commit_finished_if_needed(seq, proposal_tokens)
            len_after = int(len(seq))
            target_len_after_by_seq[seq_id] = len_after
            draft_len_after_by_seq[seq_id] = len_after
            target_draft_len_match_by_seq[seq_id] = len_after == current_len + token_count
            target_draft_token_match_by_seq[seq_id] = list(seq.token_ids[-token_count:]) == proposal_tokens
            committed_ids.append(proposal_id)
            committed_seq_ids.append(seq_id)
            precondition_ok_by_id[proposal_id] = True
            precondition_failed_by_id[proposal_id] = False

        committed_tokens = sum(int(token_count_by_id.get(proposal_id, 0)) for proposal_id in committed_ids)
        reason_counts: dict[str, int] = {}
        for reason in skip_reason_by_id.values():
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

        trace_record["enable_eager_commit_ready_only"] = True
        trace_record["eager_commit_enabled"] = True
        trace_record["eager_commit_source"] = EAGER_TAKEOVER_DRY_RUN_SOURCE
        trace_record["eager_commit_side"] = side
        trace_record["eager_commit_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_commit_plan_id"] = int(plan.plan_id)
        trace_record["eager_commit_candidate_proposal_ids"] = list(candidate_ids)
        trace_record["eager_commit_candidate_seq_ids"] = list(candidate_seq_ids)
        trace_record["eager_commit_from_readiness_proposal_ids"] = sorted(readiness_ids)
        trace_record["eager_committed_proposal_ids"] = list(committed_ids)
        trace_record["eager_committed_seq_ids"] = list(committed_seq_ids)
        committed_set = set(committed_ids)
        trace_record["eager_committed_token_count_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(token_count_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["eager_committed_accept_len_by_proposal_id"] = {
            str(proposal_id): int(value) for proposal_id, value in sorted(accept_len_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["eager_committed_action_by_proposal_id"] = {
            str(proposal_id): action for proposal_id, action in sorted(action_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["eager_committed_verify_result_by_proposal_id"] = {
            str(proposal_id): result for proposal_id, result in sorted(result_by_id.items())
            if proposal_id in committed_set
        }
        trace_record["eager_commit_skipped_proposal_ids"] = sorted(set(skipped_ids))
        trace_record["eager_commit_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(skip_reason_by_id.items())
        }
        trace_record["eager_commit_precondition_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(precondition_ok_by_id.items())
        }
        trace_record["eager_commit_precondition_failed_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(precondition_failed_by_id.items())
        }
        trace_record["eager_commit_precondition_failure_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in sorted(precondition_failure_reason_by_id.items())
        }
        trace_record["eager_commit_duplicate_proposal_ids"] = sorted(set(duplicate_proposal_ids))
        trace_record["eager_commit_duplicate_seq_ids"] = sorted(set(duplicate_seq_ids))
        trace_record["eager_commit_target_seq_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(target_len_before_by_seq.items())
        }
        trace_record["eager_commit_target_seq_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(target_len_after_by_seq.items())
        }
        trace_record["eager_commit_draft_seq_len_before_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(draft_len_before_by_seq.items())
        }
        trace_record["eager_commit_draft_seq_len_after_by_seq_id"] = {
            str(seq_id): int(value) for seq_id, value in sorted(draft_len_after_by_seq.items())
        }
        trace_record["eager_commit_target_draft_len_match_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(target_draft_len_match_by_seq.items())
        }
        trace_record["eager_commit_target_draft_token_match_by_seq_id"] = {
            str(seq_id): bool(value) for seq_id, value in sorted(target_draft_token_match_by_seq.items())
        }
        trace_record["eager_commit_frontier_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(frontier_ok_commit_by_id.items())
        }
        trace_record["eager_commit_token_payload_ok_by_proposal_id"] = {
            str(proposal_id): bool(value) for proposal_id, value in sorted(token_payload_ok_commit_by_id.items())
        }
        trace_record["eager_tokens_committed"] = int(committed_tokens)
        trace_record["eager_tokens_committed_full_accept"] = int(committed_tokens)
        trace_record["eager_commit_candidate_count"] = len(candidate_ids)
        trace_record["eager_commit_committed_count"] = len(committed_ids)
        trace_record["eager_commit_skipped_count"] = len(set(skipped_ids))
        trace_record["eager_commit_skip_reason_counts"] = dict(sorted(reason_counts.items()))
        trace_record["eager_tokens_verified"] = int(committed_tokens)
        trace_record["eager_tokens_accepted"] = int(committed_tokens)
        trace_record["eager_tokens_rejected"] = 0
        trace_record["eager_tokens_invalidated"] = 0
        if (
            self._continuous_eager_dry_run_enabled()
            and not self._continuous_eager_verify_apply_dry_run_enabled()
            and side == "target"
        ):
            self._run_continuous_eager_shadow_dry_run(
                plan,
                trace_record,
                committed_ids,
                committed_seq_ids,
                known_by_id,
                seq_by_id,
                plan_context,
            )
        self._record_elapsed_ms(trace_record, "eager_commit_time_ms", timer_start)

    def _schedule_ready_eager_dry_run(
        self,
        plan: StepPlan,
        trace_record: dict,
        plan_context: dict[str, set[int]],
    ) -> list[EagerProposal]:
        ready_proposals = self._ready_eager_proposals()
        ready_size_before = len(ready_proposals)
        seq_by_id = self._local_sequence_by_id()
        target_home = {int(seq_id) for seq_id in plan.target_home_set}
        draft_home = {int(seq_id) for seq_id in plan.draft_home_set}
        max_requests = int(getattr(self.global_config, "max_eager_requests_per_step", 0) or 0)
        gamma = int(self.gamma)

        candidates: list[EagerProposal] = []
        scheduled: list[EagerProposal] = []
        deferred: list[EagerProposal] = []
        skipped: list[EagerProposal] = []
        defer_reason_by_proposal_id: dict[int, str] = {}
        skip_reason_by_proposal_id: dict[int, str] = {}
        seen_seq_ids: set[int] = set()
        proposal_id_by_seq_id: dict[int, int] = {}
        base_len_by_seq_id: dict[int, int] = {}
        current_len_by_seq_id: dict[int, int] = {}
        base_match_by_seq_id: dict[int, bool] = {}
        seq_pre_verify_by_seq_id: dict[int, bool] = {}
        base_pre_verify_by_proposal_id: dict[int, bool] = {}
        proposal_len_by_proposal_id: dict[int, int] = {}
        to_verify_len_by_proposal_id: dict[int, int] = {}

        for proposal in ready_proposals:
            proposal_id = int(proposal.proposal_id)
            seq_id = int(proposal.seq_id)
            seq = seq_by_id.get(seq_id)
            current_len = -1 if seq is None else int(len(seq))
            candidates.append(proposal)
            base_len_by_seq_id[seq_id] = int(proposal.base_len)
            current_len_by_seq_id[seq_id] = current_len
            base_match_by_seq_id[seq_id] = seq is not None and current_len == int(proposal.base_len)
            seq_pre_verify_by_seq_id[seq_id] = bool(getattr(seq, "pre_verify", True)) if seq is not None else True
            base_pre_verify_by_proposal_id[proposal_id] = bool(proposal.base_pre_verify)
            proposal_len_by_proposal_id[proposal_id] = int(proposal.proposal_len)
            to_verify_len_by_proposal_id[proposal_id] = len(proposal.to_be_verified_token_ids)

            reason = None
            if plan.plan_phase != "steady":
                reason = "defer_non_steady_phase"
            elif proposal.state != EAGER_STATE_READY_TO_VERIFY_DRY_RUN:
                reason = "not_ready"
            elif bool(proposal.base_pre_verify):
                reason = "invalid_base_pre_verify"
            elif int(proposal.proposal_len) != gamma:
                reason = "invalid_proposal_len"
            elif len(proposal.to_be_verified_token_ids) != gamma:
                reason = "invalid_to_verify_len"
            elif len(proposal.proposal_token_ids) != gamma:
                reason = "invalid_proposal_token_len"
            elif seq is None:
                reason = "seq_not_found"
            elif self.is_request_level_finished(seq, plan_context):
                reason = "seq_finished_before_schedule"
            elif self.is_speculative_span_invalidated(seq, plan_context):
                reason = "seq_span_invalidated_before_schedule"
            elif bool(getattr(seq, "pre_verify", True)):
                reason = "seq_returned_pre_verify_before_schedule"
            elif int(len(seq)) != int(proposal.base_len):
                reason = (
                    "base_mismatch_before_schedule"
                    if int(len(seq)) < int(proposal.base_len)
                    else "base_overshot_before_schedule"
                )
            elif seq_id in target_home:
                reason = "defer_intersects_target_home"
            elif seq_id in seen_seq_ids:
                reason = "duplicate_ready_seq"

            if reason is None:
                seen_seq_ids.add(seq_id)
                scheduled.append(proposal)
                proposal_id_by_seq_id[seq_id] = proposal_id
            elif reason.startswith("defer_"):
                deferred.append(proposal)
                defer_reason_by_proposal_id[proposal_id] = reason
            else:
                skipped.append(proposal)
                skip_reason_by_proposal_id[proposal_id] = reason

        if max_requests > 0 and len(scheduled) > max_requests:
            overflow = scheduled[max_requests:]
            scheduled = scheduled[:max_requests]
            for proposal in overflow:
                deferred.append(proposal)
                defer_reason_by_proposal_id[int(proposal.proposal_id)] = "defer_max_eager_requests_per_step"

        scheduled_seq_ids = [int(proposal.seq_id) for proposal in scheduled]
        scheduled_proposal_ids = [int(proposal.proposal_id) for proposal in scheduled]
        deferred_seq_ids = [int(proposal.seq_id) for proposal in deferred]
        deferred_proposal_ids = [int(proposal.proposal_id) for proposal in deferred]
        skipped_proposal_ids = [int(proposal.proposal_id) for proposal in skipped]
        actual_draft_home_for_lane = set(self._actual_normal_draft_seq_ids(plan))
        registry_ready_proposals: list[tuple[EagerProposal, bool]] = []
        if self._eager_lane_exclusion_dry_run_enabled():
            registry_ready_proposals.extend((proposal, True) for proposal in scheduled)
            registry_ready_proposals.extend(
                (proposal, False)
                for proposal in deferred
                if defer_reason_by_proposal_id.get(int(proposal.proposal_id)) == "defer_intersects_target_home"
            )
        created_ready_proposals = []
        for proposal, scheduled_for_eager in registry_ready_proposals:
            ready_proposal = ready_eager_proposal_from_eager_proposal(
                proposal,
                created_step_id=0 if plan.step_id is None else int(plan.step_id),
                scheduled=scheduled_for_eager,
            )
            if self.dual_batch_manager.emit_ready_eager_proposal(ready_proposal):
                created_ready_proposals.append(ready_proposal)
        scheduled_target_eager_set_dry_run = list(scheduled_seq_ids)
        adjusted_draft_home_set_dry_run = [
            int(seq_id) for seq_id in plan.draft_home_set if int(seq_id) not in set(scheduled_seq_ids)
        ]
        excluded_from_draft_home_for_eager_dry_run = sorted(set(scheduled_seq_ids) & draft_home)
        plan.target_eager_set_dry_run = list(scheduled_target_eager_set_dry_run)
        plan.scheduled_target_eager_set_dry_run = list(scheduled_target_eager_set_dry_run)
        plan.scheduled_target_eager_proposal_ids_dry_run = list(scheduled_proposal_ids)
        plan.scheduled_target_eager_seq_ids_dry_run = list(scheduled_seq_ids)
        plan.adjusted_draft_home_set_dry_run = list(adjusted_draft_home_set_dry_run)
        plan.excluded_from_draft_home_for_eager_dry_run = list(excluded_from_draft_home_for_eager_dry_run)
        plan.eager_schedule_dry_run_enabled = True
        plan.enable_eager_verify_dry_run = self._eager_verify_dry_run_enabled()
        plan.enable_eager_apply_dry_run = self._eager_apply_dry_run_enabled()
        if self._eager_verify_dry_run_enabled() and not self._eager_lane_exclusion_dry_run_enabled():
            plan.eager_verify_dry_run_enabled = True
            self._run_eager_verify_dry_run(
                plan,
                trace_record,
                scheduled,
                plan_context,
            )
        if self._eager_apply_dry_run_enabled() and not self._eager_lane_exclusion_dry_run_enabled():
            plan.eager_apply_dry_run_enabled = True
            self._run_eager_apply_dry_run(
                plan,
                trace_record,
                scheduled,
                plan_context,
            )

        clear_reason_by_proposal_id = {
            str(proposal.proposal_id): "phase1h4c_scheduled_dry_run_no_verify_yet"
            for proposal in scheduled
        }
        clear_reason_by_proposal_id.update(
            {
                str(proposal.proposal_id): skip_reason_by_proposal_id.get(
                    int(proposal.proposal_id),
                    "phase1h4c_invalid_ready_no_verify_yet",
                )
                for proposal in skipped
            }
        )
        for proposal in scheduled:
            proposal.state = EAGER_STATE_TRANSFERRED_DRY_RUN
            proposal.valid = False
        for proposal in skipped:
            proposal.state = EAGER_STATE_DISCARDED
            proposal.valid = False
        ready_size_after_schedule = len(self._ready_eager_proposals())
        self.eager_proposal_buffer.remove_many(
            [proposal.proposal_id for proposal in scheduled + skipped]
        )
        ready_size_after_clear = len(self._ready_eager_proposals())

        trace_record["enable_eager_schedule_dry_run"] = True
        trace_record["eager_schedule_dry_run_enabled"] = True
        trace_record["eager_schedule_step_id"] = None if plan.step_id is None else int(plan.step_id)
        trace_record["eager_schedule_plan_id"] = int(plan.plan_id)
        trace_record["target_eager_set_dry_run"] = scheduled_target_eager_set_dry_run
        trace_record["scheduled_target_eager_set_dry_run"] = scheduled_target_eager_set_dry_run
        trace_record["scheduled_target_eager_proposal_ids_dry_run"] = scheduled_proposal_ids
        trace_record["scheduled_target_eager_seq_ids_dry_run"] = scheduled_seq_ids
        trace_record["adjusted_draft_home_set_dry_run"] = adjusted_draft_home_set_dry_run
        trace_record["excluded_from_draft_home_for_eager_dry_run"] = excluded_from_draft_home_for_eager_dry_run
        trace_record["eager_schedule_candidate_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in candidates
        ]
        trace_record["eager_schedule_candidate_seq_ids"] = [int(proposal.seq_id) for proposal in candidates]
        trace_record["eager_scheduled_proposal_ids"] = scheduled_proposal_ids
        trace_record["eager_scheduled_seq_ids"] = scheduled_seq_ids
        trace_record["eager_schedule_deferred_proposal_ids"] = deferred_proposal_ids
        trace_record["eager_schedule_deferred_seq_ids"] = deferred_seq_ids
        trace_record["eager_schedule_defer_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in defer_reason_by_proposal_id.items()
        }
        trace_record["eager_schedule_skipped_proposal_ids"] = skipped_proposal_ids
        trace_record["eager_schedule_skip_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in skip_reason_by_proposal_id.items()
        }
        trace_record["eager_schedule_clear_reason_by_proposal_id"] = clear_reason_by_proposal_id
        trace_record["eager_schedule_proposal_id_by_seq_id"] = {
            str(seq_id): proposal_id for seq_id, proposal_id in proposal_id_by_seq_id.items()
        }
        trace_record["eager_schedule_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_schedule_current_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_schedule_base_match_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_match_by_seq_id.items()
        }
        trace_record["eager_schedule_seq_pre_verify_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_pre_verify_by_seq_id.items()
        }
        trace_record["eager_schedule_base_pre_verify_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in base_pre_verify_by_proposal_id.items()
        }
        trace_record["eager_schedule_proposal_len_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in proposal_len_by_proposal_id.items()
        }
        trace_record["eager_schedule_to_verify_len_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in to_verify_len_by_proposal_id.items()
        }
        adjusted_draft_home = set(adjusted_draft_home_set_dry_run)
        trace_record["eager_schedule_intersects_target_home"] = bool(set(scheduled_seq_ids) & target_home)
        trace_record["eager_schedule_intersects_original_draft_home"] = bool(set(scheduled_seq_ids) & draft_home)
        trace_record["eager_schedule_intersects_adjusted_draft_home"] = bool(
            set(scheduled_seq_ids) & adjusted_draft_home
        )
        trace_record["eager_schedule_intersects_draft_home"] = bool(set(scheduled_seq_ids) & adjusted_draft_home)
        trace_record["eager_schedule_ready_buffer_size_before"] = int(ready_size_before)
        trace_record["eager_schedule_ready_buffer_size_after"] = int(ready_size_after_schedule)
        trace_record["eager_schedule_ready_buffer_size_after_clear"] = int(ready_size_after_clear)
        trace_record["target_ready_buffer_size_after_schedule"] = int(ready_size_after_schedule)
        trace_record["eager_ready_buffer_size_before_schedule"] = int(ready_size_before)
        trace_record["eager_ready_buffer_size_after_schedule"] = int(ready_size_after_schedule)
        trace_record["eager_ready_buffer_size_after_clear"] = int(ready_size_after_clear)
        if self._eager_lane_exclusion_dry_run_enabled() and created_ready_proposals:
            created_seq_ids = [int(proposal.seq_id) for proposal in created_ready_proposals]
            created_proposal_ids = [int(proposal.proposal_id) for proposal in created_ready_proposals]
            same_step_normal_draft_seq_ids = sorted(set(created_seq_ids) & actual_draft_home_for_lane)
            applied_excluded_seq_ids = list(plan.lane_excluded_seq_ids)
            plan.ready_eager_proposal_created_ids = list(created_proposal_ids)
            plan.ready_eager_proposal_created_seq_ids = list(created_seq_ids)
            trace_record["eager_lane_exclusion_dry_run_enabled"] = True
            trace_record["ready_eager_proposal_created_ids"] = list(created_proposal_ids)
            trace_record["ready_eager_proposal_created_seq_ids"] = list(created_seq_ids)
            trace_record["lane_exclusion_decision_available_before_draft"] = bool(
                plan.lane_exclusion_decision_available_before_draft
            )
            trace_record["lane_exclusion_deferred_until_next_step"] = bool(
                same_step_normal_draft_seq_ids
            )
            trace_record["lane_exclusion_defer_reason"] = (
                "ready_eager_proposal_created_after_plan"
                if same_step_normal_draft_seq_ids
                else None
            )
            trace_record["lane_exclusion_decision_late_count"] = 0
            trace_record["pending_lane_exclusion_decision_ids_after_emit"] = []
            trace_record["lane_exclusion_dry_run_done"] = bool(
                getattr(plan, "lane_exclusion_dry_run_done", False)
            )
            trace_record["excluded_from_actual_draft_home_for_eager"] = list(applied_excluded_seq_ids)
            trace_record["lane_excluded_seq_ids"] = list(applied_excluded_seq_ids)
            trace_record["actual_draft_home_set_for_normal_draft"] = list(
                plan.actual_draft_home_set_for_normal_draft or plan.draft_home_set
            )
        trace_record["eager_tokens_schedule_candidates"] = sum(
            int(proposal.proposal_len) for proposal in candidates
        )
        trace_record["eager_tokens_scheduled_dry_run"] = sum(
            int(proposal.proposal_len) for proposal in scheduled
        )
        trace_record["eager_tokens_deferred_dry_run"] = sum(
            int(proposal.proposal_len) for proposal in deferred
        )
        return scheduled

    def _receive_eager_transfer_dry_run(self, plan: StepPlan, trace_record: dict) -> list[EagerProposal]:
        timer_start = time.perf_counter()
        seq_by_id = self._local_sequence_by_id()
        plan_context = self._eager_transfer_plan_context(plan, trace_record)
        buffer_size_before_update = self.eager_proposal_buffer.size()
        pending_size_before_update = len(
            [
                proposal
                for proposal in self.eager_proposal_buffer.proposals()
                if proposal.valid and proposal.state == EAGER_STATE_PENDING_BASE_REACHED
            ]
        )
        base_len_by_seq_id: dict[int, int] = {}
        base_pre_verify_by_seq_id: dict[int, bool] = {}
        current_len_by_seq_id: dict[int, int] = {}
        base_match_by_seq_id: dict[int, bool] = {}
        proposal_len_by_proposal_id: dict[int, int] = {}
        to_verify_len_by_proposal_id: dict[int, int] = {}
        base_delta_by_seq_id: dict[int, int] = {}
        seq_raw_status_by_seq_id: dict[int, str] = {}
        seq_is_finished_raw_by_seq_id: dict[int, bool] = {}
        seq_request_finished_by_seq_id: dict[int, bool] = {}
        seq_span_invalidated_by_seq_id: dict[int, bool] = {}
        seq_in_scheduled_by_seq_id: dict[int, bool] = {}
        seq_in_resolved_by_seq_id: dict[int, bool] = {}
        seq_in_target_home_by_seq_id: dict[int, bool] = {}
        seq_in_draft_home_by_seq_id: dict[int, bool] = {}

        pending_ready, still_pending, pending_dropped, pending_state_by_proposal_id = (
            self._update_target_pending_eager_buffer(
                seq_by_id,
                base_len_by_seq_id,
                base_pre_verify_by_seq_id,
                current_len_by_seq_id,
                base_match_by_seq_id,
                proposal_len_by_proposal_id,
                to_verify_len_by_proposal_id,
                base_delta_by_seq_id,
                seq_raw_status_by_seq_id,
                seq_is_finished_raw_by_seq_id,
                seq_request_finished_by_seq_id,
                seq_span_invalidated_by_seq_id,
                seq_in_scheduled_by_seq_id,
                seq_in_resolved_by_seq_id,
                seq_in_target_home_by_seq_id,
                seq_in_draft_home_by_seq_id,
                plan_context,
            )
        )
        buffer_size_after_update = self.eager_proposal_buffer.size()
        pending_size_after_update = len(
            [
                proposal
                for proposal in self.eager_proposal_buffer.proposals()
                if proposal.valid and proposal.state == EAGER_STATE_PENDING_BASE_REACHED
            ]
        )

        meta = torch.zeros(5, dtype=torch.int64, device="cuda")
        dist.broadcast(meta, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        meta_values = [int(value) for value in meta.tolist()]
        num_proposals, payload_len, gamma, transfer_plan_id, transfer_step_id = meta_values
        if int(gamma) != int(self.gamma):
            raise ValueError(f"eager transfer gamma mismatch: expected={self.gamma}, got={gamma}")
        payload = torch.zeros(payload_len, dtype=torch.int64, device="cuda")
        if payload_len > 0:
            dist.broadcast(payload, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        proposals = deserialize_eager_transfer_payload(meta_values, payload.tolist())

        validated: list[EagerProposal] = []
        pending_received: list[EagerProposal] = []
        dropped: list[EagerProposal] = []
        drop_reason_by_proposal_id: dict[int, str] = {}
        state_by_proposal_id: dict[int, str] = dict(pending_state_by_proposal_id)

        for proposal in proposals:
            seq = seq_by_id.get(int(proposal.seq_id))
            self._record_eager_transfer_seq_state(
                proposal,
                seq,
                base_len_by_seq_id,
                base_pre_verify_by_seq_id,
                current_len_by_seq_id,
                base_match_by_seq_id,
                proposal_len_by_proposal_id,
                to_verify_len_by_proposal_id,
                base_delta_by_seq_id,
                seq_raw_status_by_seq_id,
                seq_is_finished_raw_by_seq_id,
                seq_request_finished_by_seq_id,
                seq_span_invalidated_by_seq_id,
                seq_in_scheduled_by_seq_id,
                seq_in_resolved_by_seq_id,
                seq_in_target_home_by_seq_id,
                seq_in_draft_home_by_seq_id,
                plan_context,
            )
            action, reason = self._classify_eager_transfer_proposal(proposal, seq, plan_context)
            state_by_proposal_id[int(proposal.proposal_id)] = reason
            if action == "ready":
                proposal.state = EAGER_STATE_READY_TO_VERIFY_DRY_RUN
                proposal.valid = True
                self.eager_proposal_buffer.store(proposal)
                validated.append(proposal)
            elif action == "pending":
                proposal.state = EAGER_STATE_PENDING_BASE_REACHED
                proposal.valid = True
                self.eager_proposal_buffer.store(proposal)
                pending_received.append(proposal)
            else:
                proposal.state = EAGER_STATE_DISCARDED
                proposal.valid = False
                dropped.append(proposal)
                drop_reason_by_proposal_id[int(proposal.proposal_id)] = reason

        buffer_size_after_receive = self.eager_proposal_buffer.size()
        pending_size_after_receive = len(
            [
                proposal
                for proposal in self.eager_proposal_buffer.proposals()
                if proposal.valid and proposal.state == EAGER_STATE_PENDING_BASE_REACHED
            ]
        )
        target_ready_size_after_receive = len(self._ready_eager_proposals())
        trace_record["target_ready_buffer_size_after_receive"] = int(target_ready_size_after_receive)
        trace_record["target_ready_buffer_size_after_schedule"] = int(target_ready_size_after_receive)
        scheduled_for_result_transfer: list[EagerProposal] = []
        if self._eager_schedule_dry_run_enabled():
            scheduled_for_result_transfer = self._schedule_ready_eager_dry_run(plan, trace_record, plan_context)
        else:
            self.eager_proposal_buffer.remove_many(
                [proposal.proposal_id for proposal in validated + pending_ready]
            )
        buffer_size_after_clear = self.eager_proposal_buffer.size()
        active_pending = [
            proposal
            for proposal in self.eager_proposal_buffer.proposals()
            if proposal.valid and proposal.state == EAGER_STATE_PENDING_BASE_REACHED
        ]

        trace_record["enable_eager_transfer_dry_run"] = True
        trace_record["eager_transfer_dry_run_enabled"] = True
        trace_record["eager_transfer_step_id"] = None if transfer_step_id < 0 else int(transfer_step_id)
        trace_record["eager_transfer_plan_id"] = int(transfer_plan_id)
        trace_record["eager_transfer_num_proposals"] = int(num_proposals)
        trace_record["eager_transfer_payload_len"] = int(payload_len)
        trace_record["eager_transfer_received_proposal_ids"] = [int(proposal.proposal_id) for proposal in proposals]
        trace_record["eager_transfer_received_seq_ids"] = [int(proposal.seq_id) for proposal in proposals]
        trace_record["eager_transfer_validated_proposal_ids"] = [int(proposal.proposal_id) for proposal in validated]
        trace_record["eager_transfer_pending_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in pending_received
        ]
        trace_record["eager_transfer_dropped_proposal_ids"] = [int(proposal.proposal_id) for proposal in dropped]
        trace_record["eager_transfer_drop_reason_by_proposal_id"] = {
            str(proposal_id): reason for proposal_id, reason in drop_reason_by_proposal_id.items()
        }
        trace_record["eager_transfer_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_transfer_base_pre_verify_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_pre_verify_by_seq_id.items()
        }
        trace_record["eager_transfer_current_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_transfer_base_delta_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_delta_by_seq_id.items()
        }
        trace_record["eager_transfer_base_match_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_match_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_raw_status_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_raw_status_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_is_finished_raw_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_is_finished_raw_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_request_finished_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_request_finished_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_span_invalidated_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_span_invalidated_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_in_scheduled_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_in_scheduled_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_in_resolved_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_in_resolved_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_in_target_home_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_in_target_home_by_seq_id.items()
        }
        trace_record["eager_transfer_seq_in_draft_home_by_seq_id"] = {
            str(seq_id): value for seq_id, value in seq_in_draft_home_by_seq_id.items()
        }
        trace_record["eager_transfer_classification_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in state_by_proposal_id.items()
        }
        trace_record["eager_transfer_proposal_len_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in proposal_len_by_proposal_id.items()
        }
        trace_record["eager_transfer_to_verify_len_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in to_verify_len_by_proposal_id.items()
        }
        trace_record["eager_pending_received_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in pending_received
        ]
        trace_record["eager_pending_received_seq_ids"] = [int(proposal.seq_id) for proposal in pending_received]
        trace_record["eager_pending_base_not_reached_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in active_pending
        ]
        trace_record["eager_pending_base_not_reached_seq_ids"] = [
            int(proposal.seq_id) for proposal in active_pending
        ]
        trace_record["eager_pending_ready_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in pending_ready
        ]
        trace_record["eager_pending_ready_seq_ids"] = [int(proposal.seq_id) for proposal in pending_ready]
        trace_record["eager_pending_dropped_proposal_ids"] = [
            int(proposal.proposal_id) for proposal in pending_dropped
        ]
        trace_record["eager_pending_drop_reason_by_proposal_id"] = {
            str(proposal.proposal_id): pending_state_by_proposal_id.get(int(proposal.proposal_id), "discarded")
            for proposal in pending_dropped
        }
        trace_record["eager_pending_buffer_size_before_update"] = int(pending_size_before_update)
        trace_record["eager_pending_buffer_size_after_update"] = int(pending_size_after_update)
        trace_record["eager_pending_buffer_size_after_receive"] = int(pending_size_after_receive)
        trace_record["eager_pending_buffer_size_after_clear"] = int(len(active_pending))
        trace_record["eager_pending_current_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in current_len_by_seq_id.items()
        }
        trace_record["eager_pending_base_len_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_len_by_seq_id.items()
        }
        trace_record["eager_pending_base_delta_by_seq_id"] = {
            str(seq_id): value for seq_id, value in base_delta_by_seq_id.items()
        }
        trace_record["eager_pending_state_by_proposal_id"] = {
            str(proposal_id): value for proposal_id, value in state_by_proposal_id.items()
        }
        trace_record["target_eager_buffer_size_before_receive"] = int(buffer_size_before_update)
        trace_record["target_eager_buffer_size_after_receive"] = int(buffer_size_after_receive)
        trace_record["target_eager_buffer_size_after_clear"] = int(buffer_size_after_clear)
        trace_record["eager_tokens_transfer_pending"] = sum(
            int(proposal.proposal_len) for proposal in pending_received
        )
        trace_record["eager_tokens_transfer_validated"] = sum(
            int(proposal.proposal_len) for proposal in validated + pending_ready
        )
        trace_record["eager_tokens_transfer_dropped"] = sum(
            int(proposal.proposal_len) for proposal in dropped + pending_dropped
        )
        trace_record["eager_buffer_size_after"] = int(buffer_size_after_clear)
        trace_record["eager_proposal_transfer_called"] = True
        self._record_elapsed_ms(trace_record, "eager_transfer_time_ms", timer_start)
        return scheduled_for_result_transfer

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
                proposal.valid = True
                promoted_seq_ids.append(seq_id)
                promoted_proposal_ids.append(int(proposal.proposal_id))
                promoted_tokens += int(proposal.proposal_len)
                promotion_reason_by_seq_id[seq_id] = "parent_normal_full_accept"
                if self._eager_transfer_dry_run_enabled():
                    self.eager_proposal_buffer.store(proposal)
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
        target_seqs = self._resolve_dual_seq_ids(
            self._target_normal_verify_seq_ids(plan),
            plan,
            "draft_apply_verify",
        )
        normal_draft_seq_ids = self._actual_normal_draft_seq_ids(plan)
        draft_seqs = self._resolve_dual_seq_ids(normal_draft_seq_ids, plan, "dual_draft")
        eager_draft_seqs = self._resolve_dual_seq_ids(
            plan.draft_eager_set,
            plan,
            "eager_draft_dry_run",
        )

        proposals = []
        draft_records = []
        eager_proposals = []
        eager_trace_record = None
        target_trace_record = None
        if draft_seqs:
            proposals, draft_records = self._draft_dual_batch_proposals(draft_seqs, plan)
            if plan.plan_phase in {"priming", "steady"}:
                self.dual_proposal_buffer.store(proposals)
            for trace_record in draft_records:
                trace_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
                self._update_lane_exclusion_proposal_trace(trace_record, plan, sent_proposals=proposals)
                self._finalize_record_profile(trace_record)
            self._send_dual_proposals(proposals, plan)
            for trace_record in draft_records:
                trace_record["normal_proposal_transfer_called"] = True
        elif self._eager_lane_exclusion_dry_run_enabled() and plan.lane_excluded_seq_ids:
            trace_record = self._trace_dual_batch_schedule([], plan, "dual_draft_lane_exclusion")
            self._update_lane_exclusion_proposal_trace(trace_record, plan, sent_proposals=[])
            self._finalize_record_profile(trace_record)

        if self._eager_draft_dry_run_enabled() and eager_draft_seqs:
            if draft_records:
                eager_trace_record = draft_records[-1]
            else:
                eager_trace_record = self._trace_dual_batch_schedule([], plan, "eager_draft_dry_run")
            eager_proposals = self._run_eager_draft_dry_run(eager_draft_seqs, plan, eager_trace_record)
            self._finalize_record_profile(eager_trace_record)

        if target_seqs:
            trace_record = self._trace_dual_batch_schedule(target_seqs, plan, "draft_apply_verify")
            target_trace_record = trace_record
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

        if self._eager_transfer_dry_run_enabled():
            # Fixed verify_group collective order for eager dry-runs:
            # normal proposals, eager proposal transfer/ready sync, local
            # takeover verify/apply dry-runs on target, then result metadata
            # transfer from target back to draft. Result transfer is zero-safe.
            transfer_trace_record = eager_trace_record or target_trace_record
            if transfer_trace_record is None and draft_records:
                transfer_trace_record = draft_records[-1]
            if transfer_trace_record is None:
                transfer_trace_record = self._trace_dual_batch_schedule([], plan, "eager_transfer_dry_run")
            self._send_eager_transfer_dry_run(
                eager_proposals,
                plan,
                transfer_trace_record,
            )
            if self._eager_result_transfer_dry_run_enabled():
                self._receive_eager_result_transfer_dry_run(
                    plan,
                    transfer_trace_record,
                    eager_proposals,
                )
            self._finalize_record_profile(transfer_trace_record)
    
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
        target_normal_seq_ids = self._target_normal_verify_seq_ids(plan)
        target_seqs = self._resolve_dual_seq_ids(target_normal_seq_ids, plan, "dual_verify")
        draft_seq_ids = self._actual_normal_draft_seq_ids(plan)
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
        priming_record = None
        eager_verify_dry_run_ran = False
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
                received_seq_ids = [int(proposal.seq_id) for proposal in received_proposals]
                received_seq_id_set = set(received_seq_ids)
                missing_after_receive = [
                    int(seq_id) for seq_id in target_seq_ids if int(seq_id) not in received_seq_id_set
                ]
                plan.fallback_received_seq_ids = list(received_seq_ids)
                plan.fallback_missing_after_receive_seq_ids = list(missing_after_receive)
            if trace_record is not None:
                trace_record["normal_proposal_transfer_called"] = True
                self._update_lane_exclusion_proposal_trace(
                    trace_record,
                    plan,
                    received_proposals=received_proposals,
                )
            if fallback_same_batch:
                target_proposals = received_proposals
                assert not plan.fallback_missing_after_receive_seq_ids, self._proposal_assertion_message(
                    plan,
                    "fallback same-batch missing proposals after receive for target seq_ids="
                    f"{target_seq_ids}",
                )
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
            if self._eager_verify_dry_run_enabled() and plan.target_eager_verify_proposal_ids_dry_run:
                plan_context = self._eager_transfer_plan_context(plan, trace_record)
                self._run_takeover_eager_verify_dry_run(
                    plan,
                    trace_record,
                    plan_context,
                )
                if (
                    self._eager_apply_dry_run_enabled()
                    and trace_record.get("eager_verify_dry_run_executed_proposal_ids")
                ):
                    self._run_takeover_eager_apply_dry_run(
                        plan,
                        trace_record,
                        plan_context,
                    )
                eager_verify_dry_run_ran = True
            torch.cuda.synchronize()
            self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)
        elif received_proposals:
            priming_record = self._trace_dual_batch_schedule([], plan, "dual_verify_idle")
            priming_record["normal_proposal_transfer_called"] = True
            priming_record["proposal_buffer_size_after"] = self.dual_proposal_buffer.size()
            self._finalize_record_profile(priming_record)
        elif trace_record is not None and self._eager_lane_exclusion_dry_run_enabled():
            self._update_lane_exclusion_proposal_trace(trace_record, plan, received_proposals=[])

        if (
            self._eager_verify_dry_run_enabled()
            and plan.target_eager_verify_proposal_ids_dry_run
            and not eager_verify_dry_run_ran
        ):
            eager_verify_seqs = self._resolve_dual_seq_ids(
                plan.target_eager_verify_seq_ids_dry_run,
                plan,
                "target_eager_verify_dry_run",
            )
            trace_record = self._trace_dual_batch_schedule(
                eager_verify_seqs,
                plan,
                "target_eager_verify_dry_run",
            )
            trace_record["proposal_tokens_available"] = 0
            trace_record["proposal_tokens_verified"] = 0
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            self._run_takeover_eager_verify_dry_run(
                plan,
                trace_record,
                self._eager_transfer_plan_context(plan, trace_record),
            )
            if (
                self._eager_apply_dry_run_enabled()
                and trace_record.get("eager_verify_dry_run_executed_proposal_ids")
            ):
                self._run_takeover_eager_apply_dry_run(
                    plan,
                    trace_record,
                    self._eager_transfer_plan_context(plan, trace_record),
                )
            torch.cuda.synchronize()
            self._mark_trace_end(trace_record, accepted_lens={}, invalidated_lens={})

        if self._eager_transfer_dry_run_enabled():
            # Fixed verify_group collective order pairs with DraftModelRunner:
            # normal proposals, eager proposal transfer/ready sync, local
            # takeover verify/apply dry-runs on target, then result metadata
            # transfer from target back to draft. Result transfer is zero-safe.
            transfer_trace_record = trace_record or priming_record
            if transfer_trace_record is None:
                transfer_trace_record = self._trace_dual_batch_schedule([], plan, "eager_transfer_dry_run")
            scheduled_for_result_transfer = self._receive_eager_transfer_dry_run(plan, transfer_trace_record)
            if self._eager_result_transfer_dry_run_enabled():
                plan.eager_result_transfer_dry_run_enabled = True
                self._send_eager_result_transfer_dry_run(
                    plan,
                    transfer_trace_record,
                    scheduled_for_result_transfer,
                )
                if self._eager_commit_ready_only_enabled():
                    self._receive_eager_commit_ready_only_decision(
                        plan,
                        transfer_trace_record,
                    )
            self._finalize_record_profile(transfer_trace_record)

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
