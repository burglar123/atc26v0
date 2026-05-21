import pickle
import torch
import time
import random
import hashlib
import json
from abc import abstractmethod
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from nano_pearl.utils.pearl_logger import logger
from nano_pearl.pearl_config import PEARLConfig
from dataclasses import dataclass, replace
from nano_pearl.models import model_dict
from nano_pearl.utils.loader import load_model
from nano_pearl.pearl_config import TPParams
from nano_pearl.layers.sampler import Sampler, norm_logits
from nano_pearl.utils.context import set_context, reset_context, get_context
from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.pearl_engine.scheduler import Scheduler, is_eos
from nano_pearl.pearl_engine.sequence import SequenceStatus
from nano_pearl.pearl_engine.stspec_plan import (
    StepPlan,
    select_exec_seqs_for_plan,
    stspec_protocol_alignment_error,
)
from nano_pearl.pearl_engine.pearl_protocol import (
    PearlMessageType,
    PearlLayoutKind,
    decode_legacy_draft_message,
    decode_legacy_verify_result,
    encode_legacy_draft_message,
    encode_legacy_verify_result,
    encode_variable_draft_message,
    encode_variable_verify_result,
    decode_variable_draft_message,
    decode_variable_verify_result,
    ensure_supported_protocol,
    normalize_layout_kind,
    validate_legacy_fixed_layout,
    validate_variable_offsets_layout,
    PearlDraftMessage,
    PearlVerifyResultMessage,
)
from nano_pearl.pearl_engine.stspec_mailbox import (
    MailboxPayload,
    STSpecMailboxError,
    STSpecPayloadMailbox,
    payloads_from_draft_message,
)
from nano_pearl.pearl_engine.stspec_mailbox_transport import (
    MailboxTransportMode,
    TargetForwardMailboxError,
    build_verification_input_from_mailbox_payload,
    classify_mailbox_miss,
    encode_mailbox_transport_envelope,
    encode_payload_tensor_envelope_from_payloads,
    interpret_target_forward_from_mailbox_output,
    payload_tensor_envelope_to_mailbox_payloads,
    validate_target_forward_from_mailbox_input,
)
from nano_pearl.pearl_engine.stspec_kv_sync import (
    MailboxKVSyncMode,
    apply_mailbox_kv_sync_plan_probe,
    build_mailbox_kv_sync_plan,
    normalize_kv_sync_mode,
)
from nano_pearl.pearl_engine.stspec_mailbox_forward_context import (
    build_target_forward_context_from_mailbox_input,
    classify_target_tp_rank_role_for_mailbox_forward,
    is_stspec_real_probe_enabled,
    normalize_target_forward_from_mailbox_output,
    validate_target_forward_mailbox_context,
)
from nano_pearl.pearl_engine.stspec_pipeline import (
    STSpecPipelinePhase,
    should_skip_target_for_warmup,
)
from nano_pearl.pearl_engine.stspec_mailbox_verify_apply import (
    MailboxVerifyApplyError,
    build_v4t_active_continuation_metadata,
    build_v4w_zero_accept_correction_diagnostics,
    build_v4w_next_step_prefix_diagnostics,
    build_v4s_result_finalization_metadata,
    build_mailbox_kv_commit_plan,
    build_mailbox_payload_consume_plan,
    build_mailbox_verify_apply_plan,
    build_mailbox_verify_commit_plan,
    build_mailbox_verify_result,
    can_enter_v4s_result_finalization,
    extract_target_token_ids_from_logits,
    initialize_v4t_active_continuation_runner_state,
    reset_v4t_active_continuation_runner_state,
    build_terminal_verify_tuple_rows,
    run_mailbox_verify_apply_no_commit_probe,
    run_mailbox_verify_commit_probe,
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
        initialize_v4t_active_continuation_runner_state(self)
        self._reset_draft_mailbox_record_guard()
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
        # V4D mailbox is local diagnostic state only. Draft/target runners are
        # separate processes, so cross-process payload delivery is deliberately
        # reported as not implemented instead of assuming shared Python memory.
        self.stspec_mailbox = STSpecPayloadMailbox()
        self.active_execution_mode = self.global_config.execution_mode
        self.active_decode_ready_mode = False
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

    def _runner_role(self):
        return "draft" if self.is_draft else "verify"

    def _set_execution_mode(self, execution_mode: str):
        if execution_mode not in self.global_config.ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode={execution_mode!r}. "
                f"Expected one of {sorted(self.global_config.ALLOWED_EXECUTION_MODES)}."
            )
        self.active_execution_mode = execution_mode

    def _schedule_with_plan(self, runner_role: str):
        return self.scheduler.schedule_with_plan(
            runner_role=runner_role,
            execution_mode=self.active_execution_mode,
            decode_ready_mode=self.active_decode_ready_mode,
            default_gamma=self.gamma,
        )

    def _trace_schedule(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        runner_role: str,
        step_plan: StepPlan,
    ):
        iteration_id, batch_id = self.scheduler.next_batch_id(runner_role)
        if step_plan.plan_id != iteration_id:
            logger.warning(
                f"StepPlan plan_id={step_plan.plan_id} does not match "
                f"trace iteration_id={iteration_id}; keeping legacy trace id."
            )
        for seq in seqs:
            seq.mark_scheduled(iteration_id, batch_id, is_prefill, runner_role)
        per_seq_zeros = {seq.seq_id: 0 for seq in seqs}
        scheduled_seq_ids = [seq.seq_id for seq in seqs]
        assert step_plan.scheduled_seq_ids == scheduled_seq_ids
        plan_signature = step_plan.signature()
        plan_digest = step_plan.digest()
        actual_exec_seq_ids = (
            list(step_plan.actual_draft_exec_seq_ids)
            if "draft" in runner_role
            else list(step_plan.actual_target_exec_seq_ids)
        )
        dryrun_exec_seq_ids = (
            list(step_plan.dryrun_draft_exec_seq_ids)
            if "draft" in runner_role
            else list(step_plan.dryrun_target_exec_seq_ids)
        )
        filtered_out_seq_ids = [seq_id for seq_id in scheduled_seq_ids if seq_id not in set(actual_exec_seq_ids)]
        actual_exec_fraction = (
            float(len(actual_exec_seq_ids)) / float(len(scheduled_seq_ids))
            if scheduled_seq_ids
            else 1.0
        )
        protocol_alignment_error = stspec_protocol_alignment_error(
            step_plan, runner_role, self.gamma, self._pearl_protocol_layout()
        )
        protocol_alignment_ok = protocol_alignment_error is None
        record = {
            "execution_mode": self.active_execution_mode,
            "decode_ready_mode": self.active_decode_ready_mode,
            "iteration_id": iteration_id,
            "batch_id": batch_id,
            "runner_role": runner_role,
            "scheduled_seq_ids": [seq.seq_id for seq in seqs],
            "request_ids": [seq.request_id for seq in seqs],
            "plan_id": step_plan.plan_id,
            "plan_signature": plan_signature,
            "plan_signature_hash": plan_digest,
            "plan_digest": plan_digest,
            "plan_num_requests": len(step_plan.requests),
            "plan_request_ids": list(step_plan.request_ids),
            "plan_two_batch_shadow": step_plan.plan_two_batch_shadow,
            "target_home_batch_id": step_plan.target_home_batch_id,
            "draft_home_batch_id": step_plan.draft_home_batch_id,
            "target_batch_seq_ids": list(step_plan.target_batch_seq_ids),
            "draft_home_batch_seq_ids": list(step_plan.draft_home_batch_seq_ids),
            "off_batch_seq_ids": list(step_plan.off_batch_seq_ids),
            "two_batch_execution_enabled": step_plan.two_batch_execution_enabled,
            "two_batch_execution_dryrun": step_plan.two_batch_execution_dryrun,
            "two_batch_execution_mode": step_plan.two_batch_execution_mode,
            "stspec_probe_enabled": step_plan.stspec_probe_enabled,
            "stspec_probe_fail_fast": step_plan.stspec_probe_fail_fast,
            "stspec_probe_local_only": step_plan.stspec_probe_local_only,
            "real_probe_attempted": step_plan.real_probe_attempted,
            "real_probe_applied": step_plan.real_probe_applied and protocol_alignment_ok,
            "real_probe_blocked": bool(step_plan.real_probe_blocked or protocol_alignment_error),
            "real_probe_block_reason": step_plan.real_probe_block_reason or protocol_alignment_error,
            "actual_target_exec_seq_ids": list(step_plan.actual_target_exec_seq_ids),
            "actual_draft_exec_seq_ids": list(step_plan.actual_draft_exec_seq_ids),
            "dryrun_target_exec_seq_ids": list(step_plan.dryrun_target_exec_seq_ids),
            "dryrun_draft_exec_seq_ids": list(step_plan.dryrun_draft_exec_seq_ids),
            "actual_exec_seq_ids": actual_exec_seq_ids,
            "dryrun_exec_seq_ids": dryrun_exec_seq_ids,
            "filtered_out_seq_ids": filtered_out_seq_ids,
            "filtered_out_seq_count": len(filtered_out_seq_ids),
            "actual_exec_fraction": actual_exec_fraction,
            "stspec_pipeline_enabled": step_plan.stspec_pipeline_enabled,
            "stspec_pipeline_phase": step_plan.stspec_pipeline_phase,
            "stspec_pipeline_step": step_plan.stspec_pipeline_step,
            "stspec_pipeline_warmup_done": step_plan.stspec_pipeline_warmup_done,
            "stspec_warmup_target_home_batch_id": step_plan.stspec_warmup_target_home_batch_id,
            "stspec_warmup_draft_home_batch_id": step_plan.stspec_warmup_draft_home_batch_id,
            "warmup_draft_payload_produced": False,
            "warmup_target_verify_skipped": False,
            "pipeline_phase_advanced": False,
            "verification_input_from_mailbox_attempted": False,
            "verification_input_from_mailbox_success": False,
            "verification_input_from_mailbox_seq_ids": [],
            "verification_input_from_mailbox_total_tokens": 0,
            "verification_input_from_mailbox_error": None,
            "target_forward_from_mailbox_input_built": False,
            "target_forward_from_mailbox_input_seq_ids": [],
            "target_forward_from_mailbox_input_total_tokens": 0,
            "target_forward_from_mailbox_input_shape": [],
            "stspec_kv_sync_probe_enabled": False,
            "stspec_kv_sync_mode": None,
            "mailbox_kv_sync_plan_built": False,
            "mailbox_kv_sync_plan_seq_ids": [],
            "mailbox_kv_sync_plan_total_tokens": 0,
            "mailbox_kv_sync_current_seq_lengths": {},
            "mailbox_kv_sync_append_start_positions": {},
            "mailbox_kv_sync_append_end_positions": {},
            "mailbox_kv_sync_position_ids": [],
            "kv_state_sync_check_attempted": False,
            "kv_state_sync_check_success": False,
            "kv_state_sync_missing_seq_ids": [],
            "kv_state_sync_error": None,
            "kv_state_sync_error_kind": None,
            "kv_state_sync_plan_json": None,
            "target_forward_from_mailbox_attempted": False,
            "target_forward_from_mailbox_success": False,
            "target_forward_from_mailbox_seq_ids": [],
            "target_forward_from_mailbox_total_tokens": 0,
            "target_forward_from_mailbox_input_shape": [],
            "target_forward_from_mailbox_output_shape": [],
            "target_forward_from_mailbox_error": None,
            "target_forward_from_mailbox_error_kind": None,
            "target_forward_from_mailbox_latency_ms": None,
            "target_forward_from_mailbox_output_interpretation_attempted": False,
            "target_forward_from_mailbox_output_interpretation_success": False,
            "target_forward_from_mailbox_output_interpretation_error": None,
            "accepted_lengths_by_seq": {},
            "rejected_seq_ids": [],
            "invalidated_mailbox_payload_count": 0,
            "stspec_mailbox_commit_probe_enabled": bool(getattr(self.global_config, "stspec_mailbox_commit_probe", False)),
            "stspec_continue_after_mailbox_commit_enabled": bool(getattr(self.global_config, "stspec_continue_after_mailbox_commit", False)),
            "mailbox_verify_commit_attempted": False,
            "mailbox_verify_commit_success": False,
            "mailbox_verify_commit_error": None,
            "mailbox_verify_commit_error_kind": None,
            "mailbox_verify_commit_seq_ids": [],
            "mailbox_verify_commit_accepted_lengths_by_seq": {},
            "mailbox_verify_commit_target_correction_token_ids_by_seq": {},
            "mailbox_verify_commit_rejected_seq_ids": [],
            "mailbox_verify_commit_total_accepted_tokens": 0,
            "mailbox_verify_commit_total_rejected_tokens": 0,
            "sequence_state_commit_attempted": False,
            "sequence_state_commit_success": False,
            "sequence_state_before": {},
            "sequence_state_after": {},
            "kv_commit_plan_built": False,
            "kv_commit_attempted": False,
            "kv_commit_success": False,
            "kv_commit_shadow_only": False,
            "kv_commit_error": None,
            "kv_commit_error_kind": None,
            "kv_commit_seq_ids": [],
            "kv_commit_accepted_lengths_by_seq": {},
            "kv_commit_target_correction_token_ids_by_seq": {},
            "kv_commit_append_start_positions_by_seq": {},
            "kv_commit_append_end_positions_by_seq": {},
            "kv_commit_sequence_length_before_by_seq": {},
            "kv_commit_sequence_length_after_by_seq": {},
            "kv_commit_rollback_attempted": False,
            "kv_commit_rollback_success": True,
            "kv_commit_skipped_non_owner": False,
            "mailbox_payload_consume_plan_built": False,
            "mailbox_payload_consume_attempted": False,
            "mailbox_payload_consume_success": False,
            "mailbox_payload_consume_error": None,
            "mailbox_payload_consume_error_kind": None,
            "mailbox_payload_consumed_payload_ids": [],
            "mailbox_payload_consumed_token_count": 0,
            "mailbox_payload_invalidate_attempted": False,
            "mailbox_payload_invalidate_success": False,
            "mailbox_payload_invalidate_error": None,
            "mailbox_payload_invalidated_payload_ids": [],
            "mailbox_payload_invalidated_token_count": 0,
            "mailbox_payload_lifecycle_before": {},
            "mailbox_payload_lifecycle_after": {},
            "mailbox_payload_duplicate_consume_detected": False,
            "mailbox_payload_consume_rollback_attempted": False,
            "mailbox_payload_consume_rollback_success": True,
            "mailbox_payload_consume_skipped_non_owner": False,
            "mailbox_payload_invalidate_skipped_non_owner": False,
            "next_pipeline_step_attempted": False,
            "next_pipeline_step_success": False,
            "next_pipeline_step_error": None,
            "next_pipeline_step_error_kind": None,
            "next_pipeline_plan_id": None,
            "next_pipeline_target_home_batch_id": None,
            "next_pipeline_draft_home_batch_id": None,
            "next_pipeline_actual_target_seq_ids": [],
            "next_pipeline_actual_draft_seq_ids": [],
            "previous_committed_plan_id": None,
            "previous_consumed_payload_ids": [],
            "previous_invalidated_payload_ids": [],
            "duplicate_payload_consume_after_continue": False,
            "pipeline_state_after_commit_valid": False,
            "scheduler_state_after_commit_valid": False,
            "breadth_only_step_count": 0,
            "breadth_only_completed": False,
            "breadth_only_completion_reason": None,
            "second_step_state_check_attempted": False,
            "second_step_state_check_success": False,
            "second_step_state_error": None,
            "second_step_state_error_kind": None,
            "current_pipeline_step": 0,
            "current_plan_id": None,
            "next_plan_id": None,
            "previous_target_home_batch_id": None,
            "previous_draft_home_batch_id": None,
            "current_target_home_batch_id": None,
            "current_draft_home_batch_id": None,
            "active_seq_ids_before_second_step": [],
            "active_seq_ids_after_second_step": [],
            "committed_seq_ids": [],
            "consumed_payload_ids": [],
            "invalidated_payload_ids": [],
            "available_mailbox_payload_ids": [],
            "pending_mailbox_payload_ids": [],
            "repeated_verify_after_commit_detected": False,
            "scheduler_state_after_second_step_valid": False,
            "sequence_state_after_second_step_valid": False,
            "mailbox_state_after_second_step_valid": False,
            "request_completion_check_attempted": False,
            "request_completion_check_success": False,
            "request_completion_reason": None,
            "active_seq_ids_at_completion_check": [],
            "finished_seq_ids_at_completion_check": [],
            "unfinished_seq_ids_at_completion_check": [],
            "max_tokens_reached_seq_ids": [],
            "eos_reached_seq_ids": [],
            "mailbox_pending_payload_ids_at_completion": [],
            "mailbox_consumed_payload_ids_at_completion": [],
            "scheduler_active_seq_ids_at_completion": [],
            "sequence_state_completion_valid": False,
            "scheduler_state_completion_valid": False,
            "mailbox_state_completion_valid": False,
            "request_completion_error": None,
            "request_completion_error_kind": None,
            "result_finalization_attempted": False,
            "result_finalization_success": False,
            "result_finalization_error": None,
            "result_finalization_error_kind": None,
            "result_finalization_skipped_non_owner": False,
            "v4s_finalization_metadata_complete": False,
            "v4s_finalization_missing_fields": [],
            "v4s_finalization_invalid_fields": [],
            "v4s_completion_gate_reason": None,
            "v4s_completion_gate_snapshot": {},
            "finalized_request_ids": [],
            "finalized_seq_ids": [],
            "finalized_output_token_counts": {},
            "finalized_output_text_available": False,
            "finalized_trace_rows": 0,
            "v4s_terminal_verify_broadcast_attempted": False,
            "v4s_terminal_verify_broadcast_success": False,
            "v4s_terminal_verify_broadcast_error": None,
            "terminal_verify_tuple_attempted": False,
            "terminal_verify_tuple_success": False,
            "terminal_verify_tuple_error": None,
            "terminal_verify_tuple_error_kind": None,
            "terminal_verify_partial_accept_supported": False,
            "terminal_verify_zero_accept_supported": False,
            "terminal_verify_seq_ids": [],
            "terminal_verify_expected_lengths_by_seq": {},
            "terminal_verify_accepted_lengths_by_seq": {},
            "terminal_verify_rejected_lengths_by_seq": {},
            "evaluator_return_attempted": False,
            "evaluator_return_success": False,
            "evaluator_return_error": None,
            "evaluator_return_error_kind": None,
            "active_continuation_attempted": False,
            "active_continuation_success": False,
            "active_continuation_step_count": 0,
            "active_continuation_seq_ids": [],
            "active_continuation_reason": None,
            "active_continuation_limit_reached": False,
            "stspec_active_continuation_max_steps": int(getattr(self.global_config, "stspec_active_continuation_max_steps", 2)),
            "stspec_active_continuation_max_steps_effective": int(
                getattr(self.global_config, "stspec_active_continuation_max_steps", 2)
            ),
            "stspec_active_continuation_max_steps_source": "config"
            if hasattr(self.global_config, "stspec_active_continuation_max_steps")
            else "default",
            "active_continuation_no_progress": False,
            "active_continuation_no_progress_reason": None,
            "active_continuation_progress_too_slow": False,
            "active_continuation_effective_token_progress_by_step": [],
            "active_continuation_bookkeeping_progress_by_step": [],
            "active_continuation_output_tokens_before_after_by_step": [],
            "active_continuation_accepted_tokens_before_after_by_step": [],
            "active_continuation_expected_len_by_step": [],
            "active_continuation_accepted_len_by_step": [],
            "active_continuation_rejected_len_by_step": [],
            "active_continuation_zero_accept_correction_available_by_step": [],
            "active_continuation_zero_accept_correction_token_ids_by_step": [],
            "active_continuation_zero_accept_correction_commit_attempted_by_step": [],
            "active_continuation_zero_accept_correction_commit_success_by_step": [],
            "active_continuation_zero_accept_correction_failure_reason_by_step": [],
            "active_continuation_correction_diagnostic_priority": [],
            "active_continuation_acceptance_too_low_after_correction_checked": False,
            "active_continuation_target_correction_missing": False,
            "active_continuation_reject_recovery_missing": False,
            "active_continuation_target_correction_not_committed": False,
            "active_continuation_zero_accept_correction_checked": False,
            "active_continuation_zero_accept_correction_failure": False,
            "active_continuation_zero_accept_correction_rows_count": 0,
            "active_continuation_next_step_prefix_token_ids_by_step": [],
            "active_continuation_next_step_prefix_len_by_step": [],
            "active_continuation_next_step_contains_correction_by_step": [],
            "active_continuation_target_draft_prefix_divergence": False,
            "active_continuation_target_correction_not_in_next_prefix": False,
            "active_continuation_kv_state_mismatch": False,
            "active_continuation_prefix_len_mismatch": False,
            "active_continuation_position_mismatch": False,
            "active_continuation_slot_mapping_mismatch": False,
            "active_continuation_scheduler_sequence_state_mismatch": False,
            "active_continuation_next_step_prefix_source_by_step": {},
            "active_continuation_target_prefix_token_ids_by_step": {},
            "active_continuation_draft_prefix_token_ids_by_step": {},
            "active_continuation_pending_correction_prefix_by_seq": {},
            "active_continuation_target_correction_token_ids_by_step": [],
            "active_continuation_target_correction_available_by_step": [],
            "active_continuation_target_correction_committed_by_step": [],
            "active_continuation_target_correction_commit_seq_ids": [],
            "active_continuation_target_correction_commit_request_ids": [],
            "active_continuation_target_correction_output_delta_by_step": [],
            "active_continuation_target_correction_shadow_only": False,
            "active_continuation_target_correction_wrong_sequence": False,
            "active_continuation_target_correction_rolled_back": False,
            "active_continuation_output_snapshot_mismatch": False,
            "active_continuation_completion_token_export_missing": False,
            "active_continuation_sequence_output_len_before_after_by_step": [],
            "active_continuation_prefix_len_before_after_by_step": [],
            "active_continuation_rejected_draft_token_ids_by_step": [],
            "active_continuation_prefix_len_by_step": [],
            "active_continuation_position_ids_by_step": [],
            "active_continuation_slot_mapping_summary_by_step": [],
            "active_continuation_alignment_check_by_step": [],
            "active_continuation_alignment_error": None,
            "active_continuation_reject_recovery_attempted": False,
            "active_continuation_reject_recovery_success": False,
            "active_continuation_zero_accept_step_count": 0,
            "active_continuation_average_acceptance_rate": 0.0,
            "active_continuation_starving_seq_ids": [],
            "active_continuation_last_advanced_step_by_seq": {},
            "active_continuation_output_not_committed": False,
            "active_continuation_completion_gate_mismatch": False,
            "active_continuation_steps_insufficient": False,
            "active_continuation_recommended_min_steps": None,
            "active_continuation_remaining_seq_ids": [],
            "active_continuation_remaining_output_tokens_by_seq": {},
            "active_continuation_pending_payload_ids": [],
            "active_continuation_latest_plan_id": None,
            "active_continuation_plan_id_history": [],
            "active_continuation_home_batch_history": [],
            "active_continuation_step_history": [],
            "active_continuation_progress_by_step": [],
            "active_continuation_total_output_token_delta": 0,
            "active_continuation_total_accepted_token_delta": 0,
            "active_continuation_remaining_tokens_to_max_by_seq": {},
            "active_continuation_completion_rechecked": False,
            "active_continuation_finalization_attempted": False,
            "active_continuation_finalization_success": False,
            "active_continuation_error": None,
            "active_continuation_error_kind": None,
            "active_continuation_skipped_due_to_outstanding_payload": False,
            "active_continuation_plan_id": None,
            "active_request_continuation_error": None,
            "active_request_continuation_error_kind": None,
            "finalized_after_active_continuation": False,
            "second_step_rollback_attempted": False,
            "second_step_rollback_success": False,
            "next_pipeline_step_skipped_non_owner": False,
            "mailbox_verify_commit_rollback_attempted": False,
            "mailbox_verify_commit_rollback_success": True,
            "mailbox_verify_commit_skipped_non_owner": False,
            "illegal_legacy_fallback": False,
            "protocol_alignment_ok": protocol_alignment_ok,
            "protocol_alignment_error": protocol_alignment_error,
            "pearl_protocol_version": int(getattr(self.global_config, "pearl_protocol_version", 1)),
            "pearl_protocol_layout": getattr(self.global_config, "pearl_protocol_layout", "legacy_fixed"),
            "pearl_protocol_envelope_enabled": bool(getattr(self.global_config, "enable_pearl_protocol_envelope", True)),
            "pearl_protocol_validate_enabled": bool(getattr(self.global_config, "pearl_protocol_validate", True)),
            "pearl_protocol_trace_enabled": bool(getattr(self.global_config, "pearl_protocol_trace", True)),
            "draft_message_seq_ids": None,
            "draft_message_per_seq_lengths": None,
            "draft_message_offsets": None,
            "draft_message_total_tokens": None,
            "verify_result_seq_ids": None,
            "verify_result_per_seq_accepted_lengths": None,
            "verify_result_offsets": None,
            "verify_result_total_tokens": None,
            "protocol_validation_ok": True,
            "protocol_validation_error": None,
            "protocol_message_type": None,
            "protocol_layout_kind": None,
            "variable_offsets_enabled": self._pearl_protocol_layout() == "variable_offsets",
            "variable_draft_message_seq_ids": None,
            "variable_draft_message_per_seq_lengths": None,
            "variable_draft_message_offsets": None,
            "variable_draft_message_total_tokens": None,
            "variable_verify_result_seq_ids": None,
            "variable_verify_result_per_seq_lengths": None,
            "variable_verify_result_offsets": None,
            "variable_verify_result_total_tokens": None,
            "variable_offsets_validation_ok": None,
            "variable_offsets_validation_error": None,
            "cross_batch_routing_ok": True,
            "cross_batch_routing_error": None,
            "stspec_mailbox_enabled": False,
            "mailbox_put_attempted": False,
            "mailbox_put_success": False,
            "mailbox_put_count": 0,
            "mailbox_put_home_batch_id": None,
            "mailbox_put_seq_ids": [],
            "mailbox_payload_put_attempted": False,
            "mailbox_payload_put_success": False,
            "mailbox_payload_put_skipped": False,
            "mailbox_payload_duplicate_put_detected": False,
            "mailbox_payload_duplicate_put_idempotent_skip": False,
            "mailbox_payload_duplicate_put_conflict": False,
            "mailbox_payload_duplicate_put_context": {},
            "mailbox_payload_put_key": None,
            "mailbox_payload_put_payload_hash": None,
            "mailbox_payload_record_guard_hit": False,
            "mailbox_payload_record_guard_size": 0,
            "mailbox_payload_recorded_plan_ids": [],
            "mailbox_payload_outstanding_available_detected": False,
            "mailbox_payload_outstanding_context": {},
            "mailbox_payload_existing_plan_id": None,
            "mailbox_payload_incoming_plan_id": None,
            "mailbox_payload_existing_payload_id": None,
            "mailbox_payload_incoming_payload_id": None,
            "mailbox_payload_lifecycle_state": None,
            "mailbox_payload_put_plan_id": None,
            "mailbox_payload_put_seq_ids": [],
            "mailbox_payload_put_home_batch_ids": [],
            "mailbox_get_attempted": False,
            "mailbox_get_success": False,
            "mailbox_get_hit_count": 0,
            "mailbox_get_miss_count": 0,
            "mailbox_get_home_batch_id": None,
            "mailbox_get_seq_ids": [],
            "mailbox_missing_seq_ids": [],
            "mailbox_available_home_batch_ids": [],
            "mailbox_available_seq_ids_by_batch": {},
            "mailbox_cross_process_delivery": None,
            "mailbox_error": None,
            "mailbox_error_kind": None,
            "mailbox_warmup_miss": False,
            "mailbox_routing_ok": True,
            "stspec_mailbox_transport_enabled": False,
            "mailbox_transport_mode": None,
            "mailbox_transport_send_attempted": False,
            "mailbox_transport_send_success": False,
            "mailbox_transport_send_seq_ids": [],
            "mailbox_transport_send_home_batch_id": None,
            "mailbox_transport_recv_attempted": False,
            "mailbox_transport_recv_success": False,
            "mailbox_transport_recv_seq_ids": [],
            "mailbox_transport_recv_home_batch_id": None,
            "mailbox_transport_payload_available": False,
            "mailbox_transport_error": None,
            "mailbox_transport_error_kind": None,
            "target_mailbox_insert_count": 0,
            "target_mailbox_insert_seq_ids": [],
            "target_mailbox_insert_home_batch_id": None,
            "target_mailbox_available_home_batch_ids": [],
            "target_mailbox_available_seq_ids_by_batch": {},
            "mailbox_payload_tensor_transport_attempted": False,
            "mailbox_payload_tensor_transport_success": False,
            "mailbox_payload_tensor_transport_backend": None,
            "mailbox_payload_tensor_transport_error": None,
            "mailbox_payload_tensor_transport_error_kind": None,
            "mailbox_payload_tensor_seq_ids": [],
            "mailbox_payload_tensor_home_batch_id": None,
            "mailbox_payload_tensor_total_tokens": 0,
            "mailbox_payload_tensor_shape": [],
            "mailbox_payload_tensor_device": None,
            "mailbox_payload_envelope_available": False,
            "mailbox_payload_token_ids_available": False,
            "mailbox_payload_tensor_available": False,
            "mailbox_payload_available_for_seq_ids": False,
            "mailbox_payload_local_to_rank": False,
            "mailbox_payload_owner_rank": None,
            "mailbox_payload_current_rank": None,
            "mailbox_payload_missing_reason": None,
            "target_tp_current_rank": None,
            "target_tp_output_owner_rank": None,
            "target_tp_is_output_owner": False,
            "target_tp_is_payload_owner": False,
            "target_tp_should_run_forward": False,
            "target_tp_should_interpret_output": False,
            "target_tp_should_apply_verify_result": False,
            "target_tp_skipped_non_owner": False,
            "mailbox_verify_apply_skipped_non_owner": False,
            "mailbox_warmup_skip": False,
            "target_verify_skipped_for_warmup": False,
            "target_consume_from_mailbox_attempted": False,
            "target_consume_from_mailbox_success": False,
            "target_consume_from_mailbox_payload_seq_ids": [],
            "target_consume_from_mailbox_payload_lengths": [],
            "target_consume_from_mailbox_payload_total_tokens": 0,
            "target_consume_from_mailbox_payload_home_batch_id": None,
            "target_consume_from_mailbox_error": None,
            "next_required_feature": None,
            "plan_legacy_equivalent": step_plan.legacy_equivalent,
            "plan_runner_role": step_plan.runner_role,
            "plan_scheduled_seq_ids": list(step_plan.scheduled_seq_ids),
            "plan_target_seq_ids": list(step_plan.target_seq_ids),
            "plan_draft_home_seq_ids": list(step_plan.draft_home_seq_ids),
            "plan_eager_seq_ids": list(step_plan.eager_seq_ids),
            "effective_gamma_per_seq": dict(step_plan.effective_gamma_per_seq),
            "home_batch_id_per_seq": dict(step_plan.home_batch_id_per_seq),
            "is_eager_per_seq": dict(step_plan.is_eager_per_seq),
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
        if not self._is_stspec_real_probe_enabled(step_plan):
            self._strip_stspec_real_probe_only_trace_fields(record)
        self.trace_records.append(record)
        return record

    def _strip_stspec_real_probe_only_trace_fields(self, record: dict) -> None:
        for key in (
            "target_tp_current_rank",
            "target_tp_output_owner_rank",
            "target_tp_is_output_owner",
            "target_tp_is_payload_owner",
            "target_tp_should_run_forward",
            "target_tp_should_interpret_output",
            "target_tp_should_apply_verify_result",
            "target_tp_skipped_non_owner",
            "mailbox_payload_envelope_available",
            "mailbox_payload_token_ids_available",
            "mailbox_payload_tensor_available",
            "mailbox_payload_available_for_seq_ids",
            "mailbox_payload_local_to_rank",
            "mailbox_payload_owner_rank",
            "mailbox_payload_current_rank",
            "mailbox_payload_missing_reason",
            "output_interpretation_skipped_non_owner",
            "mailbox_verify_apply_skipped_non_owner",
            "target_forward_output_none_expected",
            "target_forward_output_none_unexpected",
            "mailbox_verify_apply_attempted",
            "mailbox_verify_apply_success",
            "mailbox_verify_apply_error",
            "mailbox_forward_state_mutation_attempted",
            "mailbox_forward_state_mutation_committed",
            "mailbox_forward_state_mutation_rollback_success",
            "stspec_mailbox_commit_probe_enabled",
            "stspec_continue_after_mailbox_commit_enabled",
            "mailbox_verify_commit_attempted",
            "mailbox_verify_commit_success",
            "mailbox_verify_commit_error",
            "mailbox_verify_commit_error_kind",
            "mailbox_verify_commit_seq_ids",
            "mailbox_verify_commit_accepted_lengths_by_seq",
            "mailbox_verify_commit_target_correction_token_ids_by_seq",
            "mailbox_verify_commit_rejected_seq_ids",
            "mailbox_verify_commit_total_accepted_tokens",
            "mailbox_verify_commit_total_rejected_tokens",
            "sequence_state_commit_attempted",
            "sequence_state_commit_success",
            "sequence_state_before",
            "sequence_state_after",
            "kv_commit_plan_built",
            "kv_commit_attempted",
            "kv_commit_success",
            "kv_commit_shadow_only",
            "kv_commit_error",
            "kv_commit_error_kind",
            "kv_commit_seq_ids",
            "kv_commit_accepted_lengths_by_seq",
            "kv_commit_target_correction_token_ids_by_seq",
            "kv_commit_append_start_positions_by_seq",
            "kv_commit_append_end_positions_by_seq",
            "kv_commit_sequence_length_before_by_seq",
            "kv_commit_sequence_length_after_by_seq",
            "kv_commit_rollback_attempted",
            "kv_commit_rollback_success",
            "kv_commit_skipped_non_owner",
            "mailbox_payload_consume_plan_built",
            "mailbox_payload_consume_attempted",
            "mailbox_payload_consume_success",
            "mailbox_payload_consume_error",
            "mailbox_payload_consume_error_kind",
            "mailbox_payload_consumed_payload_ids",
            "mailbox_payload_consumed_token_count",
            "mailbox_payload_invalidate_attempted",
            "mailbox_payload_invalidate_success",
            "mailbox_payload_invalidate_error",
            "mailbox_payload_invalidated_payload_ids",
            "mailbox_payload_invalidated_token_count",
            "mailbox_payload_lifecycle_before",
            "mailbox_payload_lifecycle_after",
            "mailbox_payload_duplicate_consume_detected",
            "mailbox_payload_consume_rollback_attempted",
            "mailbox_payload_consume_rollback_success",
            "mailbox_payload_consume_skipped_non_owner",
            "mailbox_payload_invalidate_skipped_non_owner",
            "next_pipeline_step_attempted",
            "next_pipeline_step_success",
            "next_pipeline_step_error",
            "next_pipeline_step_error_kind",
            "next_pipeline_plan_id",
            "next_pipeline_target_home_batch_id",
            "next_pipeline_draft_home_batch_id",
            "next_pipeline_actual_target_seq_ids",
            "next_pipeline_actual_draft_seq_ids",
            "previous_committed_plan_id",
            "previous_consumed_payload_ids",
            "previous_invalidated_payload_ids",
            "duplicate_payload_consume_after_continue",
            "pipeline_state_after_commit_valid",
            "scheduler_state_after_commit_valid",
            "breadth_only_step_count",
            "breadth_only_completed",
            "breadth_only_completion_reason",
            "next_pipeline_step_skipped_non_owner",
            "second_step_state_check_attempted",
            "second_step_state_check_success",
            "second_step_state_error",
            "second_step_state_error_kind",
            "current_pipeline_step",
            "current_plan_id",
            "next_plan_id",
            "previous_target_home_batch_id",
            "previous_draft_home_batch_id",
            "current_target_home_batch_id",
            "current_draft_home_batch_id",
            "active_seq_ids_before_second_step",
            "active_seq_ids_after_second_step",
            "committed_seq_ids",
            "consumed_payload_ids",
            "invalidated_payload_ids",
            "available_mailbox_payload_ids",
            "pending_mailbox_payload_ids",
            "repeated_verify_after_commit_detected",
            "scheduler_state_after_second_step_valid",
            "sequence_state_after_second_step_valid",
            "mailbox_state_after_second_step_valid",
            "request_completion_check_attempted",
            "request_completion_check_success",
            "request_completion_reason",
            "active_seq_ids_at_completion_check",
            "finished_seq_ids_at_completion_check",
            "unfinished_seq_ids_at_completion_check",
            "max_tokens_reached_seq_ids",
            "eos_reached_seq_ids",
            "mailbox_pending_payload_ids_at_completion",
            "mailbox_consumed_payload_ids_at_completion",
            "scheduler_active_seq_ids_at_completion",
            "sequence_state_completion_valid",
            "scheduler_state_completion_valid",
            "mailbox_state_completion_valid",
            "request_completion_error",
            "request_completion_error_kind",
            "result_finalization_attempted",
            "result_finalization_success",
            "result_finalization_error",
            "result_finalization_error_kind",
            "result_finalization_skipped_non_owner",
            "v4s_finalization_metadata_complete",
            "v4s_finalization_missing_fields",
            "v4s_finalization_invalid_fields",
            "v4s_completion_gate_reason",
            "v4s_completion_gate_snapshot",
            "finalized_request_ids",
            "finalized_seq_ids",
            "finalized_output_token_counts",
            "finalized_output_text_available",
            "finalized_trace_rows",
            "v4s_terminal_verify_broadcast_attempted",
            "v4s_terminal_verify_broadcast_success",
            "v4s_terminal_verify_broadcast_error",
            "terminal_verify_tuple_attempted",
            "terminal_verify_tuple_success",
            "terminal_verify_tuple_error",
            "terminal_verify_tuple_error_kind",
            "terminal_verify_partial_accept_supported",
            "terminal_verify_zero_accept_supported",
            "terminal_verify_seq_ids",
            "terminal_verify_expected_lengths_by_seq",
            "terminal_verify_accepted_lengths_by_seq",
            "terminal_verify_rejected_lengths_by_seq",
            "evaluator_return_attempted",
            "evaluator_return_success",
            "evaluator_return_error",
            "evaluator_return_error_kind",
            "active_continuation_attempted",
            "active_continuation_success",
            "active_continuation_step_count",
            "active_continuation_seq_ids",
            "active_continuation_reason",
            "active_continuation_limit_reached",
            "stspec_active_continuation_max_steps",
            "stspec_active_continuation_max_steps_effective",
            "stspec_active_continuation_max_steps_source",
            "active_continuation_no_progress",
            "active_continuation_no_progress_reason",
            "active_continuation_progress_too_slow",
            "active_continuation_effective_token_progress_by_step",
            "active_continuation_bookkeeping_progress_by_step",
            "active_continuation_output_tokens_before_after_by_step",
            "active_continuation_accepted_tokens_before_after_by_step",
            "active_continuation_expected_len_by_step",
            "active_continuation_accepted_len_by_step",
            "active_continuation_rejected_len_by_step",
            "active_continuation_zero_accept_correction_available_by_step",
            "active_continuation_zero_accept_correction_token_ids_by_step",
            "active_continuation_zero_accept_correction_commit_attempted_by_step",
            "active_continuation_zero_accept_correction_commit_success_by_step",
            "active_continuation_zero_accept_correction_failure_reason_by_step",
            "active_continuation_correction_diagnostic_priority",
            "active_continuation_acceptance_too_low_after_correction_checked",
            "active_continuation_target_correction_missing",
            "active_continuation_reject_recovery_missing",
            "active_continuation_target_correction_not_committed",
            "active_continuation_zero_accept_correction_checked",
            "active_continuation_zero_accept_correction_failure",
            "active_continuation_zero_accept_correction_rows_count",
            "active_continuation_next_step_prefix_token_ids_by_step",
            "active_continuation_next_step_prefix_len_by_step",
            "active_continuation_next_step_contains_correction_by_step",
            "active_continuation_target_draft_prefix_divergence",
            "active_continuation_target_correction_not_in_next_prefix",
            "active_continuation_kv_state_mismatch",
            "active_continuation_prefix_len_mismatch",
            "active_continuation_position_mismatch",
            "active_continuation_slot_mapping_mismatch",
            "active_continuation_scheduler_sequence_state_mismatch",
            "active_continuation_next_step_prefix_source_by_step",
            "active_continuation_target_prefix_token_ids_by_step",
            "active_continuation_draft_prefix_token_ids_by_step",
            "active_continuation_pending_correction_prefix_by_seq",
            "active_continuation_target_correction_token_ids_by_step",
            "active_continuation_target_correction_available_by_step",
            "active_continuation_target_correction_committed_by_step",
            "active_continuation_target_correction_commit_seq_ids",
            "active_continuation_target_correction_commit_request_ids",
            "active_continuation_target_correction_output_delta_by_step",
            "active_continuation_target_correction_shadow_only",
            "active_continuation_target_correction_wrong_sequence",
            "active_continuation_target_correction_rolled_back",
            "active_continuation_output_snapshot_mismatch",
            "active_continuation_completion_token_export_missing",
            "active_continuation_sequence_output_len_before_after_by_step",
            "active_continuation_prefix_len_before_after_by_step",
            "active_continuation_rejected_draft_token_ids_by_step",
            "active_continuation_prefix_len_by_step",
            "active_continuation_position_ids_by_step",
            "active_continuation_slot_mapping_summary_by_step",
            "active_continuation_alignment_check_by_step",
            "active_continuation_alignment_error",
            "active_continuation_reject_recovery_attempted",
            "active_continuation_reject_recovery_success",
            "active_continuation_zero_accept_step_count",
            "active_continuation_average_acceptance_rate",
            "active_continuation_starving_seq_ids",
            "active_continuation_last_advanced_step_by_seq",
            "active_continuation_output_not_committed",
            "active_continuation_completion_gate_mismatch",
            "active_continuation_steps_insufficient",
            "active_continuation_recommended_min_steps",
            "active_continuation_remaining_seq_ids",
            "active_continuation_remaining_output_tokens_by_seq",
            "active_continuation_pending_payload_ids",
            "active_continuation_latest_plan_id",
            "active_continuation_plan_id_history",
            "active_continuation_home_batch_history",
            "active_continuation_step_history",
            "active_continuation_progress_by_step",
            "active_continuation_total_output_token_delta",
            "active_continuation_total_accepted_token_delta",
            "active_continuation_remaining_tokens_to_max_by_seq",
            "active_continuation_completion_rechecked",
            "active_continuation_finalization_attempted",
            "active_continuation_finalization_success",
            "active_continuation_error",
            "active_continuation_error_kind",
            "active_continuation_skipped_due_to_outstanding_payload",
            "active_continuation_plan_id",
            "active_request_continuation_error",
            "active_request_continuation_error_kind",
            "finalized_after_active_continuation",
            "second_step_rollback_attempted",
            "second_step_rollback_success",
            "mailbox_verify_commit_rollback_attempted",
            "mailbox_verify_commit_rollback_success",
            "mailbox_verify_commit_skipped_non_owner",
            "mailbox_payload_put_attempted",
            "mailbox_payload_put_success",
            "mailbox_payload_put_skipped",
            "mailbox_payload_duplicate_put_detected",
            "mailbox_payload_duplicate_put_idempotent_skip",
            "mailbox_payload_duplicate_put_conflict",
            "mailbox_payload_duplicate_put_context",
            "mailbox_payload_put_key",
            "mailbox_payload_put_payload_hash",
            "mailbox_payload_record_guard_hit",
            "mailbox_payload_record_guard_size",
            "mailbox_payload_recorded_plan_ids",
            "mailbox_payload_outstanding_available_detected",
            "mailbox_payload_outstanding_context",
            "mailbox_payload_existing_plan_id",
            "mailbox_payload_incoming_plan_id",
            "mailbox_payload_existing_payload_id",
            "mailbox_payload_incoming_payload_id",
            "mailbox_payload_lifecycle_state",
            "mailbox_payload_put_plan_id",
            "mailbox_payload_put_seq_ids",
            "mailbox_payload_put_home_batch_ids",
        ):
            record.pop(key, None)


    def _pearl_protocol_enabled(self) -> bool:
        return bool(getattr(self.global_config, "enable_pearl_protocol_envelope", True))

    def _pearl_protocol_validate_enabled(self) -> bool:
        return bool(getattr(self.global_config, "pearl_protocol_validate", True))

    def _pearl_protocol_trace_enabled(self) -> bool:
        return bool(getattr(self.global_config, "pearl_protocol_trace", True))

    def _pearl_protocol_version(self) -> int:
        return int(getattr(self.global_config, "pearl_protocol_version", 1))

    def _pearl_protocol_layout(self) -> str:
        return getattr(self.global_config, "pearl_protocol_layout", "legacy_fixed")

    def _check_pearl_protocol_config(self) -> None:
        ensure_supported_protocol(self._pearl_protocol_version())
        normalize_layout_kind(self._pearl_protocol_layout())


    def _encode_draft_protocol_message(self, **kwargs) -> PearlDraftMessage:
        if self._pearl_protocol_layout() == PearlLayoutKind.VARIABLE_OFFSETS.value:
            return encode_variable_draft_message(**kwargs)
        return encode_legacy_draft_message(**kwargs)

    def _decode_draft_protocol_message(self, message: PearlDraftMessage) -> tuple[list[int], list[int]]:
        if message.layout_kind == PearlLayoutKind.VARIABLE_OFFSETS.value:
            return decode_variable_draft_message(message)
        return decode_legacy_draft_message(message)

    def _encode_verify_protocol_message(self, **kwargs) -> PearlVerifyResultMessage:
        if self._pearl_protocol_layout() == PearlLayoutKind.VARIABLE_OFFSETS.value:
            return encode_variable_verify_result(**kwargs)
        return encode_legacy_verify_result(**kwargs)

    def _decode_verify_protocol_message(self, message: PearlVerifyResultMessage):
        if message.layout_kind == PearlLayoutKind.VARIABLE_OFFSETS.value:
            return decode_variable_verify_result(message)
        return decode_legacy_verify_result(message)

    def _trace_pearl_protocol_message(self, record: dict | None, message) -> None:
        if record is None or not self._pearl_protocol_trace_enabled():
            return
        record["pearl_protocol_version"] = message.protocol_version
        record["pearl_protocol_layout"] = message.layout_kind
        record["protocol_message_type"] = message.message_type
        record["protocol_layout_kind"] = message.layout_kind
        is_variable = message.layout_kind == PearlLayoutKind.VARIABLE_OFFSETS.value
        record["variable_offsets_enabled"] = is_variable
        if message.message_type == PearlMessageType.DRAFT_TOKENS.value:
            record["draft_message_seq_ids"] = list(message.seq_ids)
            record["draft_message_per_seq_lengths"] = list(message.per_seq_draft_lengths)
            record["draft_message_offsets"] = list(message.draft_offsets)
            record["draft_message_total_tokens"] = int(message.total_draft_tokens)
            if is_variable:
                record["variable_draft_message_seq_ids"] = list(message.seq_ids)
                record["variable_draft_message_per_seq_lengths"] = list(message.per_seq_draft_lengths)
                record["variable_draft_message_offsets"] = list(message.draft_offsets)
                record["variable_draft_message_total_tokens"] = int(message.total_draft_tokens)
        elif message.message_type == PearlMessageType.VERIFY_RESULT.value:
            record["verify_result_seq_ids"] = list(message.seq_ids)
            record["verify_result_per_seq_accepted_lengths"] = list(message.per_seq_accepted_lengths)
            record["verify_result_offsets"] = list(message.accepted_offsets)
            record["verify_result_total_tokens"] = int(message.total_accepted_tokens)
            if is_variable:
                record["variable_verify_result_seq_ids"] = list(message.seq_ids)
                record["variable_verify_result_per_seq_lengths"] = list(message.per_seq_accepted_lengths)
                record["variable_verify_result_offsets"] = list(message.accepted_offsets)
                record["variable_verify_result_total_tokens"] = int(message.total_accepted_tokens)

    def _validate_and_trace_pearl_protocol(self, record: dict | None, message, expected_seq_ids: list[int]) -> None:
        if not self._pearl_protocol_enabled():
            return
        try:
            self._check_pearl_protocol_config()
            if self._pearl_protocol_validate_enabled():
                if message.layout_kind == PearlLayoutKind.VARIABLE_OFFSETS.value:
                    validate_variable_offsets_layout(message, expected_seq_ids)
                else:
                    validate_legacy_fixed_layout(message, expected_seq_ids, self.gamma)
        except Exception as exc:
            if record is not None:
                record["protocol_validation_ok"] = False
                record["protocol_validation_error"] = str(exc)
                record["protocol_message_type"] = getattr(message, "message_type", None)
                record["protocol_layout_kind"] = getattr(message, "layout_kind", None)
                if getattr(message, "layout_kind", None) == PearlLayoutKind.VARIABLE_OFFSETS.value:
                    record["variable_offsets_validation_ok"] = False
                    record["variable_offsets_validation_error"] = str(exc)
            raise
        if record is not None:
            record["protocol_validation_ok"] = True
            record["protocol_validation_error"] = None
            if getattr(message, "layout_kind", None) == PearlLayoutKind.VARIABLE_OFFSETS.value:
                record["variable_offsets_validation_ok"] = True
                record["variable_offsets_validation_error"] = None
        self._trace_pearl_protocol_message(record, message)

    def _select_exec_seqs_for_plan(
        self,
        seqs: list[Sequence],
        step_plan: StepPlan,
        runner_role: str,
        trace_record: dict,
    ) -> list[Sequence]:
        exec_seqs = select_exec_seqs_for_plan(seqs, step_plan, runner_role)
        actual_exec_seq_ids = [seq.seq_id for seq in exec_seqs]
        trace_record["actual_exec_seq_ids"] = actual_exec_seq_ids
        trace_record["filtered_out_seq_ids"] = [
            seq.seq_id for seq in seqs if seq.seq_id not in set(actual_exec_seq_ids)
        ]
        trace_record["filtered_out_seq_count"] = len(trace_record["filtered_out_seq_ids"])
        trace_record["actual_exec_fraction"] = (
            float(len(actual_exec_seq_ids)) / float(len(seqs)) if seqs else 1.0
        )
        if not step_plan.real_probe_attempted:
            assert exec_seqs == seqs
        return exec_seqs

    def _is_stspec_real_probe_enabled(self, step_plan: StepPlan | None = None) -> bool:
        return is_stspec_real_probe_enabled(self.global_config, step_plan)

    def _stspec_mailbox_enabled(self, step_plan: StepPlan | None) -> bool:
        return bool(
            step_plan is not None
            and self._is_stspec_real_probe_enabled(step_plan)
            and not step_plan.stspec_probe_local_only
            and not step_plan.is_prefill
            and step_plan.execution_mode in {"parallel_pearl", "serialized_pearl"}
        )

    def _mailbox_available_seq_ids_by_batch(self) -> dict[str, list[int]]:
        return self.stspec_mailbox.available_seq_ids_by_batch()

    def _reset_draft_mailbox_record_guard(self) -> None:
        self.stspec_draft_mailbox_record_guard: dict[tuple, dict] = {}

    @staticmethod
    def _draft_mailbox_payload_hash(payload: MailboxPayload) -> str:
        encoded = json.dumps(payload.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _draft_mailbox_recorded_plan_ids(self) -> list[int]:
        plan_ids = {
            int(key[0])
            for key in getattr(self, "stspec_draft_mailbox_record_guard", {})
            if key and key[0] is not None
        }
        return sorted(plan_ids)

    def _stspec_kv_sync_probe_enabled(self, step_plan: StepPlan | None) -> bool:
        return bool(
            step_plan is not None
            and step_plan.real_probe_attempted
            and getattr(self.global_config, "stspec_kv_sync_probe", True)
        )

    def _stspec_kv_sync_mode(self) -> str:
        return normalize_kv_sync_mode(getattr(self.global_config, "stspec_kv_sync_mode", "metadata_only")).value

    def _mailbox_forward_commit_disabled(self) -> bool:
        return bool(getattr(self.global_config, "stspec_disable_mailbox_forward_commit", True))

    def _trace_mailbox_availability(self, trace_record: dict) -> None:
        home_batch_ids = self.stspec_mailbox.available_home_batch_ids()
        seq_ids_by_batch = self._mailbox_available_seq_ids_by_batch()
        trace_record["mailbox_available_home_batch_ids"] = home_batch_ids
        trace_record["mailbox_available_seq_ids_by_batch"] = seq_ids_by_batch
        trace_record["target_mailbox_available_home_batch_ids"] = home_batch_ids
        trace_record["target_mailbox_available_seq_ids_by_batch"] = seq_ids_by_batch

    def _record_mailbox_error(
        self,
        trace_record: dict,
        *,
        kind: str,
        message: str,
        next_required_feature: str | None = None,
        warmup_miss: bool = False,
    ) -> None:
        trace_record["mailbox_error"] = message
        trace_record["mailbox_error_kind"] = kind
        trace_record["mailbox_warmup_miss"] = bool(warmup_miss)
        trace_record["mailbox_routing_ok"] = False
        trace_record["cross_batch_routing_ok"] = False
        trace_record["cross_batch_routing_error"] = message
        if next_required_feature:
            trace_record["next_required_feature"] = next_required_feature
        self._trace_mailbox_availability(trace_record)

    def _record_mailbox_transport_error(
        self,
        trace_record: dict,
        *,
        kind: str,
        message: str,
        next_required_feature: str | None = None,
    ) -> None:
        trace_record["mailbox_transport_error"] = message
        trace_record["mailbox_transport_error_kind"] = kind
        if next_required_feature:
            trace_record["next_required_feature"] = next_required_feature

    def _mailbox_transport_mode(self) -> str:
        # V4E exposes a validated transport envelope. Runtime delivery remains
        # diagnostic-only until a safe cross-process object/tensor side channel is
        # selected; warmup and target-consume diagnostics are reported separately.
        return MailboxTransportMode.DIAGNOSTIC_ONLY.value

    def _mailbox_allow_warmup_miss(self) -> bool:
        return bool(getattr(self.global_config, "stspec_mailbox_allow_warmup_miss", False))

    def _receive_mailbox_payload_tensor_probe(
        self,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        runner_role: str,
        trace_record: dict,
    ) -> None:
        """V4G local-list payload transport backend for the guarded probe.

        The existing PEARL verification path broadcasts a flat CUDA tensor after
        target preflight; it does not yet carry variable-offset Python metadata.
        For V4G we explicitly construct a CPU/list token payload envelope on the
        target side using the target StepPlan's exact seq ids/home batch. This
        proves envelope validation, mailbox insertion, and target consume
        plumbing before failing later at target-forward/KV wiring.
        """
        seq_by_id = {int(seq.seq_id): seq for seq in exec_seqs}
        payloads: list[MailboxPayload] = []
        for seq_id in step_plan.actual_target_exec_seq_ids:
            seq = seq_by_id.get(int(seq_id))
            if seq is None:
                trace_record["illegal_legacy_fallback"] = True
                raise RuntimeError(
                    "ST-Spec target payload transport cannot find target exec seq; "
                    f"seq_id={seq_id}, actual_target_exec_seq_ids={step_plan.actual_target_exec_seq_ids}, "
                    f"available_exec_seq_ids={list(seq_by_id)}"
                )
            length = 1 if getattr(seq, "pre_verify", False) else int(self.gamma)
            token_ids = [int(token) for token in list(seq.token_ids)[-length:]]
            if len(token_ids) < length:
                token_ids = ([0] * (length - len(token_ids))) + token_ids
            payloads.append(
                MailboxPayload(
                    plan_id=step_plan.plan_id,
                    producer_role="draft_payload_tensor_probe",
                    producer_home_batch_id=step_plan.target_home_batch_id,
                    target_home_batch_id=step_plan.target_home_batch_id,
                    draft_home_batch_id=step_plan.draft_home_batch_id,
                    seq_id=int(seq.seq_id),
                    request_id=seq.request_id,
                    home_batch_id=step_plan.target_home_batch_id,
                    gamma=int(self.gamma),
                    layout_kind=PearlLayoutKind.VARIABLE_OFFSETS.value,
                    protocol_version=self._pearl_protocol_version(),
                    draft_token_ids=token_ids,
                    per_seq_length=length,
                    offset=0,
                    logical_step=step_plan.plan_id,
                    producer_actual_exec_seq_ids=list(step_plan.actual_target_exec_seq_ids),
                    producer_draft_message_seq_ids=list(step_plan.actual_target_exec_seq_ids),
                    metadata={"mailbox_payload_tensor_backend": "local_list_probe"},
                )
            )
        trace_record["mailbox_payload_tensor_transport_attempted"] = True
        trace_record["mailbox_payload_tensor_transport_backend"] = "local_list_probe"
        try:
            envelope = encode_payload_tensor_envelope_from_payloads(
                payloads,
                home_batch_id=step_plan.target_home_batch_id,
                source_plan_id=step_plan.plan_id,
                source_runner_role="draft_payload_tensor_probe",
                source_draft_home_batch_id=step_plan.target_home_batch_id,
                target_home_batch_id=step_plan.target_home_batch_id,
                gamma=self.gamma,
                logical_step=step_plan.plan_id,
            )
            transported_payloads = payload_tensor_envelope_to_mailbox_payloads(envelope)
            self.stspec_mailbox.put_payloads(
                step_plan.target_home_batch_id,
                transported_payloads,
                plan_id=step_plan.plan_id,
                producer_role="draft_payload_tensor_probe",
            )
        except Exception as exc:
            trace_record["mailbox_payload_tensor_transport_success"] = False
            trace_record["mailbox_payload_tensor_transport_error"] = str(exc)
            trace_record["mailbox_payload_tensor_transport_error_kind"] = type(exc).__name__
            trace_record["next_required_feature"] = "mailbox_payload_tensor_transport_backend"
            raise
        trace_record["mailbox_payload_tensor_transport_success"] = True
        trace_record["mailbox_payload_tensor_transport_error"] = None
        trace_record["mailbox_payload_tensor_transport_error_kind"] = None
        trace_record["mailbox_payload_tensor_seq_ids"] = list(envelope.seq_ids)
        trace_record["mailbox_payload_tensor_home_batch_id"] = envelope.home_batch_id
        trace_record["mailbox_payload_tensor_total_tokens"] = int(envelope.total_tokens)
        trace_record["mailbox_payload_tensor_shape"] = list(envelope.payload_shape)
        trace_record["mailbox_payload_tensor_device"] = envelope.payload_device
        trace_record["mailbox_transport_recv_attempted"] = True
        trace_record["mailbox_transport_recv_success"] = True
        trace_record["mailbox_transport_recv_seq_ids"] = list(envelope.seq_ids)
        trace_record["mailbox_transport_recv_home_batch_id"] = envelope.home_batch_id
        trace_record["mailbox_transport_payload_available"] = True
        trace_record["target_mailbox_insert_count"] = len(transported_payloads)
        trace_record["target_mailbox_insert_seq_ids"] = [payload.seq_id for payload in transported_payloads]
        trace_record["target_mailbox_insert_home_batch_id"] = step_plan.target_home_batch_id
        self._trace_mailbox_availability(trace_record)

    def _raise_illegal_legacy_fallback(
        self,
        trace_record: dict,
        *,
        step_plan: StepPlan,
        input_seq_ids: list[int],
        exec_seq_ids: list[int],
    ) -> None:
        trace_record["illegal_legacy_fallback"] = True
        trace_record["target_forward_from_mailbox_error_kind"] = "illegal_legacy_fallback"
        trace_record["next_required_feature"] = "strict_mailbox_target_seq_routing"
        message = (
            "illegal legacy fallback detected in ST-Spec real probe; target forward from mailbox "
            "must use actual_target_exec_seq_ids rather than scheduled full batch"
        )
        trace_record["target_forward_from_mailbox_error"] = message
        raise RuntimeError(
            f"{message}; plan_id={step_plan.plan_id}, scheduled_seq_ids={list(step_plan.scheduled_seq_ids)}, "
            f"actual_target_exec_seq_ids={list(step_plan.actual_target_exec_seq_ids)}, "
            f"input_seq_ids={input_seq_ids}, exec_seq_ids={exec_seq_ids}, "
            "next_required_feature=strict_mailbox_target_seq_routing"
        )

    def _v4s_real_probe_finalization_mode(self, step_plan: StepPlan | None = None) -> bool:
        return bool(
            step_plan is not None
            and self._is_stspec_real_probe_enabled(step_plan)
            and getattr(self.global_config, "stspec_mailbox_commit_probe", False)
            and getattr(self.global_config, "stspec_continue_after_mailbox_commit", False)
            and self._pearl_protocol_layout() == PearlLayoutKind.VARIABLE_OFFSETS.value
        )

    def _record_v4x_pending_corrections(self, commit_result, trace_record: dict) -> None:
        plan = getattr(commit_result, "plan", None)
        if plan is None or not bool(getattr(commit_result, "success", False)):
            return
        pending = dict(getattr(self, "stspec_active_continuation_pending_corrections", {}) or {})
        request_ids = list(getattr(plan, "request_ids", []) or [])
        request_ids_by_seq = {
            int(seq_id): request_ids[index]
            for index, seq_id in enumerate(list(getattr(plan, "seq_ids", []) or []))
            if index < len(request_ids)
        }
        recorded: dict[str, dict] = {}
        for raw_seq_id, raw_tokens in dict(getattr(plan, "target_correction_token_ids_by_seq", {}) or {}).items():
            seq_id = int(raw_seq_id)
            tokens = [int(token) for token in list(raw_tokens or [])]
            if not tokens:
                continue
            after_state = (
                dict(getattr(commit_result, "sequence_state_after", {}) or {}).get(seq_id)
                or dict(getattr(commit_result, "sequence_state_after", {}) or {}).get(str(seq_id))
                or {}
            )
            token_ids_after = [int(token) for token in list(after_state.get("token_ids") or [])]
            pending[seq_id] = {
                "seq_id": seq_id,
                "request_id": request_ids_by_seq.get(seq_id),
                "correction_token_ids": tokens,
                "source_plan_id": getattr(plan, "plan_id", None),
                "sequence_token_ids_after": token_ids_after,
                "prefix_len_after": int(after_state.get("num_tokens") or len(token_ids_after) or 0),
                "output_token_count_after": int(after_state.get("output_token_count") or 0),
            }
            recorded[str(seq_id)] = dict(pending[seq_id])
        self.stspec_active_continuation_pending_corrections = pending
        if recorded:
            trace_record["active_continuation_pending_correction_prefix_by_seq"] = recorded

    def _propagate_v4x_pending_correction_prefix(
        self,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        trace_record: dict,
    ) -> None:
        if not self._v4s_real_probe_finalization_mode(step_plan):
            return
        pending = dict(getattr(self, "stspec_active_continuation_pending_corrections", {}) or {})
        if not pending:
            return
        seq_by_id = {int(seq.seq_id): seq for seq in exec_seqs}
        applied: dict[str, dict] = {}
        inherited: dict[str, dict] = {}
        mismatches: dict[str, dict] = {}
        remaining: dict[int, dict] = {}
        for seq_id, info in pending.items():
            seq_id = int(seq_id)
            seq = seq_by_id.get(seq_id)
            if seq is None:
                remaining[seq_id] = info
                continue
            tokens = [int(token) for token in list(info.get("correction_token_ids") or [])]
            if not tokens:
                continue
            current_tokens = [int(token) for token in list(getattr(seq, "token_ids", []) or [])]
            expected_prefix = [int(token) for token in list(info.get("sequence_token_ids_after") or [])]
            if expected_prefix and current_tokens[: len(expected_prefix)] == expected_prefix:
                inherited[str(seq_id)] = {
                    "source_plan_id": info.get("source_plan_id"),
                    "correction_token_ids": tokens,
                    "prefix_len": len(current_tokens),
                    "source": "target_sequence",
                }
                continue
            if current_tokens[-len(tokens):] == tokens:
                inherited[str(seq_id)] = {
                    "source_plan_id": info.get("source_plan_id"),
                    "correction_token_ids": tokens,
                    "prefix_len": len(current_tokens),
                    "source": "target_sequence_suffix",
                }
                continue
            request_id = info.get("request_id")
            if request_id is not None and getattr(seq, "request_id", None) != request_id:
                mismatches[str(seq_id)] = {
                    "reason": "request_id_mismatch",
                    "expected_request_id": request_id,
                    "actual_request_id": getattr(seq, "request_id", None),
                }
                remaining[seq_id] = info
                continue
            before_len = len(current_tokens)
            for token in tokens:
                seq.append_token(int(token))
            after_tokens = [int(token) for token in list(getattr(seq, "token_ids", []) or [])]
            applied[str(seq_id)] = {
                "source_plan_id": info.get("source_plan_id"),
                "correction_token_ids": tokens,
                "prefix_len_before": before_len,
                "prefix_len_after": len(after_tokens),
                "source": "target_pending_correction_prefix",
            }
        self.stspec_active_continuation_pending_corrections = remaining
        if applied or inherited or mismatches:
            trace_record["active_continuation_next_step_prefix_source_by_step"] = {
                "plan_id": getattr(step_plan, "plan_id", None),
                "applied": applied,
                "inherited": inherited,
                "mismatches": mismatches,
            }
            trace_record["active_continuation_target_prefix_token_ids_by_step"] = {
                str(seq.seq_id): list(getattr(seq, "token_ids", []) or [])
                for seq in exec_seqs
                if int(seq.seq_id) in {int(key) for key in list(applied) + list(inherited)}
            }
        if mismatches:
            trace_record["active_continuation_scheduler_sequence_state_mismatch"] = True
            trace_record["next_required_feature"] = "active_request_continuation_scheduler_sequence_state_mismatch"

    def _build_v4s_terminal_verify_rows(self, exec_seqs: list[Sequence], commit_result) -> list[list[int]]:
        if commit_result.plan is None:
            raise RuntimeError("V4S terminal verify requires a mailbox commit plan")
        verify_rows, metadata = build_terminal_verify_tuple_rows(
            commit_result.plan,
            exec_seqs,
            gamma=int(self.gamma),
            eos_token_id=getattr(self.global_config, "eos", None),
        )
        if not metadata.get("terminal_verify_tuple_success"):
            raise RuntimeError(str(metadata.get("terminal_verify_tuple_error")))
        verify_rows[3] = [1 for _ in exec_seqs]
        return verify_rows

    def _build_v4t_active_verify_rows(self, exec_seqs: list[Sequence], commit_result) -> tuple[list[list[int]], dict]:
        if commit_result.plan is None:
            raise RuntimeError("V4T active continuation requires a mailbox commit plan")
        verify_rows, metadata = build_terminal_verify_tuple_rows(
            commit_result.plan,
            exec_seqs,
            gamma=int(self.gamma),
            eos_token_id=getattr(self.global_config, "eos", None),
        )
        return verify_rows, metadata

    def _participate_v4s_terminal_verify_broadcast(
        self,
        exec_seqs: list[Sequence],
        trace_record: dict,
        *,
        verify_rows: list[list[int]] | None = None,
    ) -> None:
        trace_record["v4s_terminal_verify_broadcast_attempted"] = True
        try:
            num_to_be_verified_tokens = sum(1 if bool(getattr(seq, "pre_verify", False)) else int(self.gamma) for seq in exec_seqs)
            num_next_round_input = int(self.gamma) * len(exec_seqs)
            msg = torch.zeros(num_to_be_verified_tokens + num_next_round_input, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.global_config.draft_config.master_rank, group=self.verify_group)
            if self.rank == self.global_config.target_config.master_rank and verify_rows is not None:
                verify_res = torch.tensor(verify_rows, dtype=torch.int64, device="cuda")
            else:
                verify_res = torch.zeros((4, len(exec_seqs)), dtype=torch.int64, device="cuda")
            dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
            trace_record["v4s_terminal_verify_broadcast_success"] = True
            trace_record["v4s_terminal_verify_broadcast_error"] = None
        except Exception as exc:
            trace_record["v4s_terminal_verify_broadcast_success"] = False
            trace_record["v4s_terminal_verify_broadcast_error"] = str(exc)
            trace_record["result_finalization_error"] = str(exc)
            trace_record["result_finalization_error_kind"] = "evaluator_return_after_breadth_only_completion"
            trace_record["next_required_feature"] = "evaluator_return_after_breadth_only_completion"
            raise

    def _scheduler_active_seq_ids_excluding_finalized(self, finalized_seq_ids: list[int]) -> list[int]:
        finalized = {int(seq_id) for seq_id in finalized_seq_ids}
        return [int(seq.seq_id) for seq in self.scheduler.running if int(seq.seq_id) not in finalized]

    def _mailbox_available_payload_ids(self) -> list[str]:
        if not hasattr(self.stspec_mailbox, "lifecycle_snapshot"):
            return []
        snapshot = self.stspec_mailbox.lifecycle_snapshot()
        return sorted(
            [
                str(payload_id)
                for payload_id, row in snapshot.items()
                if isinstance(row, dict) and str(row.get("lifecycle_state")) == "available"
            ],
            key=str,
        )

    def _finalize_v4s_scheduler_state(
        self,
        exec_seqs: list[Sequence],
        finalized_seq_ids: list[int],
    ) -> None:
        finalized = {int(seq_id) for seq_id in finalized_seq_ids}
        now = time.time()
        for seq in list(exec_seqs):
            if int(seq.seq_id) not in finalized:
                continue
            if hasattr(seq, "finish_ts") and getattr(seq, "finish_ts", None) is None:
                seq.finish_ts = now
            if not bool(getattr(seq, "is_finished", False)):
                seq.mark_finished(record_finish_ts=False)
            if hasattr(seq, "num_acc_tokens") and isinstance(seq.num_acc_tokens, list) and not seq.num_acc_tokens:
                seq.num_acc_tokens.append(int(getattr(seq, "cur_acc_tokens", 0) or 0))
            if seq in self.scheduler.running:
                self.scheduler.running.remove(seq)
            if seq not in self.scheduler.finished:
                self.scheduler.finished.append(seq)

    def _try_finalize_v4s_result(
        self,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        trace_record: dict,
        commit_result,
        output,
    ) -> bool:
        is_output_owner = bool(getattr(output, "output_owner", False))
        if not can_enter_v4s_result_finalization(
            commit_result,
            self.global_config,
            is_output_owner=is_output_owner,
        ):
            return False

        finalized_seq_ids = [int(seq_id) for seq_id in commit_result.finished_seq_ids_at_completion_check or commit_result.committed_seq_ids]
        scheduler_active = self._scheduler_active_seq_ids_excluding_finalized(finalized_seq_ids)
        mailbox_pending = self._mailbox_available_payload_ids()
        metadata = build_v4s_result_finalization_metadata(
            commit_result,
            exec_seqs,
            is_output_owner=is_output_owner,
            scheduler_active_seq_ids=scheduler_active,
            mailbox_pending_payload_ids=mailbox_pending,
        )
        trace_record.update(metadata)
        if not metadata.get("result_finalization_success"):
            next_feature = metadata.get("next_required_feature") or "evaluator_return_after_breadth_only_completion"
            trace_record["next_required_feature"] = next_feature
            if next_feature == "active_request_continuation_after_breadth_only_step":
                return False
            raise RuntimeError(
                f"V4S result finalization failed; error={metadata.get('result_finalization_error')}; "
                f"next_required_feature={next_feature}"
            )

        try:
            if commit_result.plan is None:
                raise RuntimeError("V4S terminal verify requires a mailbox commit plan")
            verify_rows, tuple_metadata = build_terminal_verify_tuple_rows(
                commit_result.plan,
                exec_seqs,
                gamma=int(self.gamma),
                eos_token_id=getattr(self.global_config, "eos", None),
            )
            trace_record.update(tuple_metadata)
            if not tuple_metadata.get("terminal_verify_tuple_success"):
                raise RuntimeError(str(tuple_metadata.get("terminal_verify_tuple_error")))
            verify_rows[3] = [1 for _ in exec_seqs]
        except Exception as exc:
            trace_record["result_finalization_success"] = False
            trace_record["result_finalization_error"] = str(exc)
            trace_record["result_finalization_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["terminal_verify_tuple_attempted"] = True
            trace_record["terminal_verify_tuple_success"] = False
            trace_record["terminal_verify_tuple_error"] = str(exc)
            trace_record["terminal_verify_tuple_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["evaluator_return_attempted"] = True
            trace_record["evaluator_return_success"] = False
            trace_record["evaluator_return_error"] = str(exc)
            trace_record["evaluator_return_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["next_required_feature"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            raise RuntimeError(
                f"{exc}; next_required_feature=terminal_verify_tuple_partial_accept_after_breadth_only"
            ) from exc
        self._participate_v4s_terminal_verify_broadcast(exec_seqs, trace_record, verify_rows=verify_rows)
        self._finalize_v4s_scheduler_state(exec_seqs, list(metadata.get("finalized_seq_ids") or []))
        trace_record["evaluator_return_attempted"] = True
        trace_record["evaluator_return_success"] = True
        trace_record["evaluator_return_error"] = None
        trace_record["evaluator_return_error_kind"] = None
        trace_record["result_finalization_success"] = True
        trace_record["result_finalization_error"] = None
        trace_record["result_finalization_error_kind"] = None
        trace_record["breadth_only_completed"] = True
        trace_record["finalized_after_active_continuation"] = self.stspec_active_continuation_step_count > 0
        reset_v4t_active_continuation_runner_state(self)
        trace_record["next_required_feature"] = "end_to_end_breadth_only_completion"
        return True

    def _v4v_active_continuation_max_steps(self) -> tuple[int, str]:
        if hasattr(self.global_config, "stspec_active_continuation_max_steps"):
            return int(getattr(self.global_config, "stspec_active_continuation_max_steps")), "config"
        return 2, "default"

    def _build_v4v_active_continuation_snapshot(
        self,
        *,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        commit_result,
        active_seq_ids: list[int],
        step_count: int,
        max_steps: int,
    ) -> dict:
        active_set = {int(seq_id) for seq_id in active_seq_ids}
        request_ids_by_seq = {
            int(seq.seq_id): getattr(seq, "request_id", None)
            for seq in exec_seqs
            if int(seq.seq_id) in active_set
        }
        output_tokens_by_seq = {
            int(seq.seq_id): int(getattr(seq, "num_completion_tokens", 0) or 0)
            for seq in exec_seqs
            if int(seq.seq_id) in active_set
        }
        accepted_tokens_by_seq = {
            int(seq.seq_id): int((getattr(seq, "trace_stats", {}) or {}).get("accepted_tokens") or 0)
            for seq in exec_seqs
            if int(seq.seq_id) in active_set
        }
        finished_by_seq = {
            int(seq.seq_id): bool(getattr(seq, "is_finished", False))
            for seq in exec_seqs
            if int(seq.seq_id) in active_set
        }
        max_tokens_by_seq = {
            int(seq.seq_id): int(getattr(seq, "max_tokens", 0) or 0)
            for seq in exec_seqs
            if int(seq.seq_id) in active_set and getattr(seq, "max_tokens", None) is not None
        }
        remaining_to_max_by_seq = {
            str(seq_id): max(int(max_tokens_by_seq.get(seq_id, 0) or 0) - int(token_count or 0), 0)
            for seq_id, token_count in output_tokens_by_seq.items()
            if int(max_tokens_by_seq.get(seq_id, 0) or 0) > 0
        }
        pending_payload_ids = [str(payload_id) for payload_id in getattr(commit_result, "mailbox_pending_payload_ids_at_completion", []) or []]
        consumed_payload_ids = [str(payload_id) for payload_id in getattr(commit_result, "mailbox_payload_consumed_payload_ids", []) or []]
        invalidated_payload_ids = [str(payload_id) for payload_id in getattr(commit_result, "mailbox_payload_invalidated_payload_ids", []) or []]
        commit_plan = getattr(commit_result, "plan", None)
        accepted_lengths = {int(key): int(value or 0) for key, value in dict(getattr(commit_plan, "accepted_lengths_by_seq", {}) or {}).items()}
        rejected_token_ids_by_seq = dict(getattr(commit_plan, "rejected_token_ids_by_seq", {}) or {})
        accepted_token_ids_by_seq = dict(getattr(commit_plan, "accepted_token_ids_by_seq", {}) or {})
        target_correction_token_ids_by_seq = dict(getattr(commit_plan, "target_correction_token_ids_by_seq", {}) or {})
        drafted_token_ids_by_seq = dict(getattr(getattr(commit_result, "plan", None), "drafted_token_ids_by_seq", {}) or {})
        seq_by_id = {int(seq.seq_id): seq for seq in exec_seqs}
        sequence_state_before = getattr(commit_result, "sequence_state_before", {}) or {}
        sequence_state_after = getattr(commit_result, "sequence_state_after", {}) or {}
        commit_request_ids = list(getattr(commit_plan, "request_ids", []) or [])
        commit_request_ids_by_seq = {
            int(seq_id): commit_request_ids[index]
            for index, seq_id in enumerate(list(getattr(commit_plan, "seq_ids", []) or []))
            if index < len(commit_request_ids)
        }

        def _state_for(mapping: dict, seq_id: int) -> dict:
            return dict(mapping.get(seq_id) or mapping.get(str(seq_id)) or {})

        target_correction_available_by_seq: dict[int, bool] = {}
        target_correction_committed_by_seq: dict[int, bool] = {}
        target_correction_output_delta_by_seq: dict[int, int] = {}
        target_correction_commit_request_ids_by_seq: dict[int, object] = {}
        sequence_output_len_before_after_by_seq: dict[int, dict[str, int]] = {}
        prefix_len_before_after_by_seq: dict[int, dict[str, int]] = {}
        sequence_token_ids_before_after_by_seq: dict[int, dict[str, list[int]]] = {}
        completion_token_ids_by_seq: dict[int, list[int]] = {}
        target_correction_shadow_only_by_seq: dict[int, bool] = {}
        target_correction_wrong_sequence_by_seq: dict[int, bool] = {}
        target_correction_rolled_back_by_seq: dict[int, bool] = {}
        output_snapshot_mismatch_by_seq: dict[int, bool] = {}
        completion_token_export_missing_by_seq: dict[int, bool] = {}
        target_correction_commit_attempted_by_seq: dict[int, bool] = {}
        sequence_object_identity_by_seq: dict[int, int | None] = {}
        completion_token_len_before_after_by_seq: dict[int, dict[str, int]] = {}
        service_metadata_num_output_tokens_before_after_by_seq: dict[int, dict[str, int | None]] = {}
        for raw_seq_id, raw_tokens in target_correction_token_ids_by_seq.items():
            seq_id = int(raw_seq_id)
            correction_tokens = [int(token) for token in list(raw_tokens or [])]
            target_correction_available_by_seq[seq_id] = bool(correction_tokens)
            target_correction_commit_attempted_by_seq[seq_id] = bool(
                correction_tokens and getattr(commit_result, "sequence_state_commit_attempted", False)
            )
            before_state = _state_for(sequence_state_before, seq_id)
            after_state = _state_for(sequence_state_after, seq_id)
            before_token_ids = [int(token) for token in list(before_state.get("token_ids") or [])]
            after_token_ids = [int(token) for token in list(after_state.get("token_ids") or [])]
            before_output_len = int(before_state.get("output_token_count") or 0)
            after_output_len = int(after_state.get("output_token_count") or 0)
            before_prefix_len = int(before_state.get("num_tokens") or 0)
            after_prefix_len = int(after_state.get("num_tokens") or 0)
            output_delta = int(after_output_len - before_output_len)
            prefix_delta = int(after_prefix_len - before_prefix_len)
            expected_suffix = correction_tokens[-len(correction_tokens):] if correction_tokens else []
            suffix_matches = bool(
                correction_tokens
                and output_delta >= len(correction_tokens)
                and len(after_token_ids) >= len(correction_tokens)
                and after_token_ids[-len(correction_tokens):] == expected_suffix
            )
            seq = seq_by_id.get(seq_id)
            current_completion_tokens = [
                int(token) for token in list(getattr(seq, "completion_token_ids", []) or [])
            ] if seq is not None else []
            sequence_token_ids_before_after_by_seq[seq_id] = {
                "before": before_token_ids,
                "after": after_token_ids,
            }
            completion_token_ids_by_seq[seq_id] = list(current_completion_tokens)
            current_output_len = int(getattr(seq, "num_completion_tokens", 0) or 0) if seq is not None else 0
            sequence_object_identity_by_seq[seq_id] = id(seq) if seq is not None else None
            completion_token_len_before_after_by_seq[seq_id] = {
                "before": before_output_len,
                "after": len(current_completion_tokens),
            }
            service_after = None
            service_before = None
            if seq is not None and hasattr(seq, "service_metadata"):
                service_metadata = seq.service_metadata()
                service_after = service_metadata.get("num_decode_output_tokens")
                decode_ready_prefill = int(service_metadata.get("num_decode_ready_prefill_tokens") or 0)
                service_before = max(before_output_len - decode_ready_prefill, 0)
            service_metadata_num_output_tokens_before_after_by_seq[seq_id] = {
                "before": service_before,
                "after": service_after,
            }
            commit_request_id = commit_request_ids_by_seq.get(seq_id)
            target_correction_commit_request_ids_by_seq[seq_id] = commit_request_id
            wrong_sequence = bool(
                seq is None
                or (commit_request_id is not None and getattr(seq, "request_id", None) != commit_request_id)
                or (after_state.get("request_id") is not None and commit_request_id is not None and after_state.get("request_id") != commit_request_id)
            )
            committed = bool(correction_tokens and suffix_matches and output_delta >= len(correction_tokens) and not wrong_sequence)
            export_missing = bool(
                committed
                and (
                    current_output_len < after_output_len
                    or len(current_completion_tokens) < len(correction_tokens)
                    or current_completion_tokens[-len(correction_tokens):] != correction_tokens
                )
            )
            output_snapshot_mismatch = bool(committed and seq is not None and current_output_len != after_output_len)
            rolled_back = bool(committed and seq is not None and current_output_len < after_output_len)
            shadow_only = bool(correction_tokens and not committed and output_delta <= 0 and prefix_delta <= 0)

            target_correction_committed_by_seq[seq_id] = committed
            target_correction_output_delta_by_seq[seq_id] = output_delta
            sequence_output_len_before_after_by_seq[seq_id] = {"before": before_output_len, "after": after_output_len}
            prefix_len_before_after_by_seq[seq_id] = {"before": before_prefix_len, "after": after_prefix_len}
            target_correction_shadow_only_by_seq[seq_id] = shadow_only
            target_correction_wrong_sequence_by_seq[seq_id] = wrong_sequence
            target_correction_rolled_back_by_seq[seq_id] = rolled_back
            output_snapshot_mismatch_by_seq[seq_id] = output_snapshot_mismatch
            completion_token_export_missing_by_seq[seq_id] = export_missing
        rejected_lengths = {
            int(key): len(value or [])
            for key, value in rejected_token_ids_by_seq.items()
        }
        expected_lengths = {
            int(seq_id): int(accepted_lengths.get(int(seq_id), 0)) + int(rejected_lengths.get(int(seq_id), 0))
            for seq_id in set(accepted_lengths) | set(rejected_lengths)
        }
        total_accepted = int(getattr(commit_result, "total_accepted_tokens", 0) or 0)
        if not total_accepted and accepted_lengths:
            total_accepted = sum(int(value or 0) for value in accepted_lengths.values())
        return {
            "step_count": int(step_count),
            "max_steps": int(max_steps),
            "plan_id": int(getattr(step_plan, "plan_id", 0) or 0),
            "next_plan_id": getattr(commit_result, "next_pipeline_plan_id", None),
            "target_home_batch_id": getattr(step_plan, "target_home_batch_id", None),
            "draft_home_batch_id": getattr(step_plan, "draft_home_batch_id", None),
            "remaining_seq_ids": sorted(active_set),
            "request_ids_by_seq": {str(key): value for key, value in request_ids_by_seq.items()},
            "unfinished_seq_ids": list(getattr(commit_result, "unfinished_seq_ids_at_completion_check", []) or []),
            "scheduler_active_seq_ids": list(getattr(commit_result, "scheduler_active_seq_ids_at_completion", []) or []),
            "remaining_output_tokens_by_seq": {str(key): value for key, value in output_tokens_by_seq.items()},
            "accepted_tokens_by_seq": {str(key): value for key, value in accepted_tokens_by_seq.items()},
            "remaining_tokens_to_max_by_seq": remaining_to_max_by_seq,
            "finished_by_seq": {str(key): value for key, value in finished_by_seq.items()},
            "max_tokens_by_seq": {str(key): value for key, value in max_tokens_by_seq.items()},
            "expected_len_by_seq": {str(key): value for key, value in expected_lengths.items()},
            "accepted_len_by_seq": {str(key): value for key, value in accepted_lengths.items()},
            "rejected_len_by_seq": {str(key): value for key, value in rejected_lengths.items()},
            "accepted_token_count_by_seq": {
                str(key): len(value or []) for key, value in accepted_token_ids_by_seq.items()
            },
            "target_correction_token_ids_by_seq": {
                str(key): list(value or []) for key, value in target_correction_token_ids_by_seq.items()
            },
            "target_correction_available_by_seq": {
                str(key): value for key, value in target_correction_available_by_seq.items()
            },
            "target_correction_committed_by_seq": {
                str(key): value for key, value in target_correction_committed_by_seq.items()
            },
            "target_correction_commit_attempted_by_seq": {
                str(key): value for key, value in target_correction_commit_attempted_by_seq.items()
            },
            "target_correction_commit_request_ids_by_seq": {
                str(key): value for key, value in target_correction_commit_request_ids_by_seq.items()
            },
            "target_correction_output_delta_by_seq": {
                str(key): value for key, value in target_correction_output_delta_by_seq.items()
            },
            "target_correction_shadow_only_by_seq": {
                str(key): value for key, value in target_correction_shadow_only_by_seq.items()
            },
            "target_correction_wrong_sequence_by_seq": {
                str(key): value for key, value in target_correction_wrong_sequence_by_seq.items()
            },
            "target_correction_rolled_back_by_seq": {
                str(key): value for key, value in target_correction_rolled_back_by_seq.items()
            },
            "output_snapshot_mismatch_by_seq": {
                str(key): value for key, value in output_snapshot_mismatch_by_seq.items()
            },
            "completion_token_export_missing_by_seq": {
                str(key): value for key, value in completion_token_export_missing_by_seq.items()
            },
            "sequence_output_len_before_after_by_seq": {
                str(key): value for key, value in sequence_output_len_before_after_by_seq.items()
            },
            "prefix_len_before_after_by_seq": {
                str(key): value for key, value in prefix_len_before_after_by_seq.items()
            },
            "sequence_token_ids_before_after_by_seq": {
                str(key): value for key, value in sequence_token_ids_before_after_by_seq.items()
            },
            "completion_token_ids_by_seq": {
                str(key): value for key, value in completion_token_ids_by_seq.items()
            },
            "sequence_object_identity_by_seq": {
                str(key): value for key, value in sequence_object_identity_by_seq.items()
            },
            "completion_token_len_before_after_by_seq": {
                str(key): value for key, value in completion_token_len_before_after_by_seq.items()
            },
            "service_metadata_num_output_tokens_before_after_by_seq": {
                str(key): value for key, value in service_metadata_num_output_tokens_before_after_by_seq.items()
            },
            "rejected_draft_token_ids_by_seq": {
                str(key): list(value or []) for key, value in rejected_token_ids_by_seq.items()
            },
            "draft_token_ids_by_seq": {
                str(key): list(value or []) for key, value in drafted_token_ids_by_seq.items()
            },
            "pending_payload_ids": pending_payload_ids,
            "consumed_payload_ids": consumed_payload_ids,
            "invalidated_payload_ids": invalidated_payload_ids,
            "total_accepted_tokens": total_accepted,
            "finished_seq_ids": list(getattr(commit_result, "finished_seq_ids_at_completion_check", []) or []),
            "breadth_only_completed": bool(getattr(commit_result, "breadth_only_completed", False)),
        }

    def _classify_v4v_active_continuation_progress(self, snapshot: dict) -> tuple[bool, dict]:
        previous = getattr(self, "stspec_active_continuation_last_snapshot", None)
        progress_reasons: list[str] = []
        output_token_delta = 0
        accepted_token_delta = 0
        pending_payload_delta = 0
        consumed_payload_delta = 0
        invalidated_payload_delta = 0
        output_delta_by_seq: dict[str, int] = {}
        accepted_delta_by_seq: dict[str, int] = {}
        if previous is None:
            progress_reasons.append("initial_active_continuation_snapshot")
        else:
            if int(snapshot.get("plan_id") or 0) > int(previous.get("plan_id") or 0):
                progress_reasons.append("plan_id_advanced")
            current_tokens = snapshot.get("remaining_output_tokens_by_seq") or {}
            previous_tokens = previous.get("remaining_output_tokens_by_seq") or {}
            output_increased = False
            for seq_id, token_count in current_tokens.items():
                token_delta = int(token_count or 0) - int(previous_tokens.get(seq_id, 0) or 0)
                output_delta_by_seq[str(seq_id)] = int(token_delta)
                output_token_delta += max(token_delta, 0)
                if token_delta > 0:
                    output_increased = True
            if output_increased:
                progress_reasons.append("output_token_increased")
            current_accepted = snapshot.get("accepted_tokens_by_seq") or {}
            previous_accepted = previous.get("accepted_tokens_by_seq") or {}
            accepted_delta_by_seq = {
                str(seq_id): int(value or 0) - int(previous_accepted.get(seq_id, 0) or 0)
                for seq_id, value in current_accepted.items()
            }
            accepted_token_delta = sum(max(delta, 0) for delta in accepted_delta_by_seq.values())
            if not accepted_token_delta:
                accepted_token_delta = max(
                    int(snapshot.get("total_accepted_tokens") or 0)
                    - int(previous.get("total_accepted_tokens") or 0),
                    0,
                )
            if accepted_token_delta > 0:
                progress_reasons.append("accepted_token_progress")
            current_finished = snapshot.get("finished_by_seq") or {}
            previous_finished = previous.get("finished_by_seq") or {}
            finished_transitions = [
                str(seq_id)
                for seq_id, finished in current_finished.items()
                if bool(finished) and not bool(previous_finished.get(seq_id, False))
            ]
            if finished_transitions or set(snapshot.get("finished_seq_ids") or []) - set(previous.get("finished_seq_ids") or []):
                progress_reasons.append("sequence_finished")
            if set(snapshot.get("consumed_payload_ids") or []) - set(previous.get("consumed_payload_ids") or []):
                progress_reasons.append("payload_consumed")
            if set(snapshot.get("invalidated_payload_ids") or []) - set(previous.get("invalidated_payload_ids") or []):
                progress_reasons.append("payload_invalidated")
            pending_payload_delta = len(previous.get("pending_payload_ids") or []) - len(snapshot.get("pending_payload_ids") or [])
            consumed_payload_delta = len(set(snapshot.get("consumed_payload_ids") or []) - set(previous.get("consumed_payload_ids") or []))
            invalidated_payload_delta = len(
                set(snapshot.get("invalidated_payload_ids") or []) - set(previous.get("invalidated_payload_ids") or [])
            )
            if set(previous.get("remaining_seq_ids") or []) - set(snapshot.get("remaining_seq_ids") or []):
                progress_reasons.append("unfinished_seq_count_decreased")
            if pending_payload_delta > 0 and "payload_consumed" not in progress_reasons:
                progress_reasons.append("pending_payload_decreased")
        snapshot = dict(snapshot)
        snapshot["progress_reasons"] = progress_reasons
        snapshot["made_progress"] = bool(progress_reasons)
        snapshot["output_token_delta"] = int(output_token_delta)
        snapshot["accepted_token_delta"] = int(accepted_token_delta)
        snapshot["output_token_delta_by_seq"] = output_delta_by_seq if previous is not None else {}
        snapshot["accepted_token_delta_by_seq"] = accepted_delta_by_seq if previous is not None else {}
        snapshot["pending_payload_delta"] = int(pending_payload_delta)
        snapshot["consumed_payload_delta"] = int(consumed_payload_delta)
        snapshot["invalidated_payload_delta"] = int(invalidated_payload_delta)
        effective_reasons = [
            reason
            for reason in progress_reasons
            if reason in {"output_token_increased", "accepted_token_progress", "sequence_finished", "unfinished_seq_count_decreased"}
        ]
        bookkeeping_reasons = [reason for reason in progress_reasons if reason not in set(effective_reasons)]
        snapshot["effective_token_progress_reasons"] = effective_reasons
        snapshot["bookkeeping_progress_reasons"] = bookkeeping_reasons
        snapshot["made_effective_token_progress"] = bool(effective_reasons)
        if previous is not None:
            snapshot["before_output_tokens_by_seq"] = dict(previous.get("remaining_output_tokens_by_seq") or {})
            snapshot["after_output_tokens_by_seq"] = dict(snapshot.get("remaining_output_tokens_by_seq") or {})
            snapshot["before_accepted_tokens_by_seq"] = dict(previous.get("accepted_tokens_by_seq") or {})
            snapshot["after_accepted_tokens_by_seq"] = dict(snapshot.get("accepted_tokens_by_seq") or {})
            snapshot["before_pending_payload_ids"] = list(previous.get("pending_payload_ids") or [])
            snapshot["after_pending_payload_ids"] = list(snapshot.get("pending_payload_ids") or [])
            snapshot["before_consumed_payload_ids"] = list(previous.get("consumed_payload_ids") or [])
            snapshot["after_consumed_payload_ids"] = list(snapshot.get("consumed_payload_ids") or [])
            snapshot["before_invalidated_payload_ids"] = list(previous.get("invalidated_payload_ids") or [])
            snapshot["after_invalidated_payload_ids"] = list(snapshot.get("invalidated_payload_ids") or [])
            snapshot["before_finished_seq_ids"] = list(previous.get("finished_seq_ids") or [])
            snapshot["after_finished_seq_ids"] = list(snapshot.get("finished_seq_ids") or [])
            snapshot["before_finished_by_seq"] = dict(previous.get("finished_by_seq") or {})
            snapshot["after_finished_by_seq"] = dict(snapshot.get("finished_by_seq") or {})
        substantive_progress = [reason for reason in progress_reasons if reason != "plan_id_advanced"]
        no_progress = (
            previous is not None
            and not substantive_progress
            and bool(snapshot.get("remaining_seq_ids"))
        )
        if no_progress:
            snapshot["no_progress_reason"] = "only_plan_id_advanced" if progress_reasons else "no_observable_state_delta"
        return no_progress, snapshot

    def _remember_v4v_active_continuation_progress(self, snapshot: dict) -> None:
        self.stspec_active_continuation_last_snapshot = dict(snapshot)
        history = list(getattr(self, "stspec_active_continuation_plan_id_history", []) or [])
        plan_id = snapshot.get("plan_id")
        if plan_id is not None:
            history.append(int(plan_id))
        self.stspec_active_continuation_plan_id_history = history[-64:]
        progress = list(getattr(self, "stspec_active_continuation_progress_by_step", []) or [])
        progress.append(dict(snapshot))
        self.stspec_active_continuation_progress_by_step = progress[-64:]

    def _record_v4v_active_continuation_snapshot(
        self,
        trace_record: dict,
        snapshot: dict,
        *,
        max_steps: int,
        max_steps_source: str = "config",
    ) -> None:
        trace_record["stspec_active_continuation_max_steps"] = int(max_steps)
        trace_record["stspec_active_continuation_max_steps_effective"] = int(max_steps)
        trace_record["stspec_active_continuation_max_steps_source"] = str(max_steps_source)
        trace_record["active_continuation_remaining_seq_ids"] = list(snapshot.get("remaining_seq_ids") or [])
        trace_record["active_continuation_remaining_output_tokens_by_seq"] = dict(
            snapshot.get("remaining_output_tokens_by_seq") or {}
        )
        trace_record["active_continuation_remaining_tokens_to_max_by_seq"] = dict(
            snapshot.get("remaining_tokens_to_max_by_seq") or {}
        )
        trace_record["active_continuation_pending_payload_ids"] = list(snapshot.get("pending_payload_ids") or [])
        trace_record["active_continuation_latest_plan_id"] = snapshot.get("plan_id")
        trace_record["active_continuation_plan_id_history"] = list(
            getattr(self, "stspec_active_continuation_plan_id_history", []) or []
        )
        if (
            snapshot.get("plan_id") is not None
            and (
                not trace_record["active_continuation_plan_id_history"]
                or trace_record["active_continuation_plan_id_history"][-1] != snapshot.get("plan_id")
            )
        ):
            trace_record["active_continuation_plan_id_history"].append(snapshot.get("plan_id"))
        home_batch_history = list(trace_record.get("active_continuation_home_batch_history") or [])
        home_batch_snapshot = {
            "step_count": snapshot.get("step_count"),
            "target_home_batch_id": snapshot.get("target_home_batch_id"),
            "draft_home_batch_id": snapshot.get("draft_home_batch_id"),
        }
        if not home_batch_history or home_batch_history[-1] != home_batch_snapshot:
            home_batch_history.append(home_batch_snapshot)
        trace_record["active_continuation_home_batch_history"] = home_batch_history[-64:]
        step_history = list(
            getattr(self, "stspec_active_continuation_progress_by_step", []) or []
        )
        if (
            not step_history
            or step_history[-1].get("step_count") != snapshot.get("step_count")
            or step_history[-1].get("plan_id") != snapshot.get("plan_id")
        ):
            step_history.append(dict(snapshot))
        trace_record["active_continuation_step_history"] = step_history
        trace_record["active_continuation_progress_by_step"] = step_history
        trace_record["active_continuation_effective_token_progress_by_step"] = [
            {
                "step_count": item.get("step_count"),
                "plan_id": item.get("plan_id"),
                "seq_ids": list(item.get("remaining_seq_ids") or []),
                "reasons": list(item.get("effective_token_progress_reasons") or []),
                "output_token_delta_by_seq": dict(item.get("output_token_delta_by_seq") or {}),
                "accepted_token_delta_by_seq": dict(item.get("accepted_token_delta_by_seq") or {}),
            }
            for item in step_history
        ]
        trace_record["active_continuation_bookkeeping_progress_by_step"] = [
            {
                "step_count": item.get("step_count"),
                "plan_id": item.get("plan_id"),
                "reasons": list(item.get("bookkeeping_progress_reasons") or []),
            }
            for item in step_history
        ]
        trace_record["active_continuation_output_tokens_before_after_by_step"] = [
            {
                "step_count": item.get("step_count"),
                "before": dict(item.get("before_output_tokens_by_seq") or {}),
                "after": dict(item.get("after_output_tokens_by_seq") or item.get("remaining_output_tokens_by_seq") or {}),
                "delta": dict(item.get("output_token_delta_by_seq") or {}),
            }
            for item in step_history
        ]
        trace_record["active_continuation_accepted_tokens_before_after_by_step"] = [
            {
                "step_count": item.get("step_count"),
                "before": dict(item.get("before_accepted_tokens_by_seq") or {}),
                "after": dict(item.get("after_accepted_tokens_by_seq") or item.get("accepted_tokens_by_seq") or {}),
                "delta": dict(item.get("accepted_token_delta_by_seq") or {}),
            }
            for item in step_history
        ]
        trace_record["active_continuation_expected_len_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("expected_len_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_accepted_len_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("accepted_len_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_rejected_len_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("rejected_len_by_seq") or {})}
            for item in step_history
        ]
        zero_accept_correction = build_v4w_zero_accept_correction_diagnostics(step_history)
        trace_record["active_continuation_zero_accept_correction_available_by_step"] = list(
            zero_accept_correction.get("active_continuation_zero_accept_correction_available_by_step") or []
        )
        trace_record["active_continuation_zero_accept_correction_token_ids_by_step"] = list(
            zero_accept_correction.get("active_continuation_zero_accept_correction_token_ids_by_step") or []
        )
        trace_record["active_continuation_zero_accept_correction_commit_attempted_by_step"] = list(
            zero_accept_correction.get("active_continuation_zero_accept_correction_commit_attempted_by_step") or []
        )
        trace_record["active_continuation_zero_accept_correction_commit_success_by_step"] = list(
            zero_accept_correction.get("active_continuation_zero_accept_correction_commit_success_by_step") or []
        )
        trace_record["active_continuation_zero_accept_correction_failure_reason_by_step"] = list(
            zero_accept_correction.get("active_continuation_zero_accept_correction_failure_reason_by_step") or []
        )
        trace_record["active_continuation_correction_diagnostic_priority"] = list(
            zero_accept_correction.get("active_continuation_correction_diagnostic_priority") or []
        )
        trace_record["active_continuation_target_correction_missing"] = bool(
            zero_accept_correction.get("active_continuation_target_correction_missing", False)
        )
        trace_record["active_continuation_reject_recovery_missing"] = bool(
            zero_accept_correction.get("active_continuation_reject_recovery_missing", False)
        )
        trace_record["active_continuation_target_correction_not_committed"] = bool(
            zero_accept_correction.get("active_continuation_target_correction_not_committed", False)
        )
        trace_record["active_continuation_zero_accept_correction_checked"] = bool(
            zero_accept_correction.get("zero_accept_correction_checked", False)
        )
        trace_record["active_continuation_zero_accept_correction_failure"] = bool(
            zero_accept_correction.get("zero_accept_correction_failure", False)
        )
        trace_record["active_continuation_zero_accept_correction_rows_count"] = len(
            zero_accept_correction.get("zero_accept_correction_rows") or []
        )
        next_prefix_diagnostic = build_v4w_next_step_prefix_diagnostics(step_history)
        trace_record["active_continuation_next_step_prefix_token_ids_by_step"] = list(
            next_prefix_diagnostic.get("active_continuation_next_step_prefix_token_ids_by_step") or []
        )
        trace_record["active_continuation_next_step_prefix_len_by_step"] = list(
            next_prefix_diagnostic.get("active_continuation_next_step_prefix_len_by_step") or []
        )
        trace_record["active_continuation_next_step_contains_correction_by_step"] = list(
            next_prefix_diagnostic.get("active_continuation_next_step_contains_correction_by_step") or []
        )
        for key in (
            "active_continuation_target_correction_not_in_next_prefix",
            "active_continuation_target_draft_prefix_divergence",
            "active_continuation_kv_state_mismatch",
            "active_continuation_prefix_len_mismatch",
            "active_continuation_position_mismatch",
            "active_continuation_slot_mapping_mismatch",
        ):
            trace_record[key] = bool(next_prefix_diagnostic.get(key, False))
        trace_record["active_continuation_target_correction_token_ids_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("target_correction_token_ids_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_target_correction_available_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("target_correction_available_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_target_correction_committed_by_step"] = [
            {
                "step_count": item.get("step_count"),
                "values": dict(item.get("target_correction_committed_by_seq") or {}),
            }
            for item in step_history
        ]
        trace_record["active_continuation_target_correction_commit_seq_ids"] = sorted(
            {
                int(seq_id)
                for item in step_history
                for seq_id, committed in (item.get("target_correction_committed_by_seq") or {}).items()
                if bool(committed)
            }
        )
        trace_record["active_continuation_target_correction_commit_request_ids"] = [
            request_id
            for item in step_history
            for seq_id, request_id in (item.get("target_correction_commit_request_ids_by_seq") or {}).items()
            if bool((item.get("target_correction_committed_by_seq") or {}).get(str(seq_id)))
        ]
        trace_record["active_continuation_target_correction_output_delta_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("target_correction_output_delta_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_target_correction_shadow_only"] = any(
            any(bool(value) for value in (item.get("target_correction_shadow_only_by_seq") or {}).values())
            for item in step_history
        )
        trace_record["active_continuation_target_correction_wrong_sequence"] = any(
            any(bool(value) for value in (item.get("target_correction_wrong_sequence_by_seq") or {}).values())
            for item in step_history
        )
        trace_record["active_continuation_target_correction_rolled_back"] = any(
            any(bool(value) for value in (item.get("target_correction_rolled_back_by_seq") or {}).values())
            for item in step_history
        )
        trace_record["active_continuation_output_snapshot_mismatch"] = any(
            any(bool(value) for value in (item.get("output_snapshot_mismatch_by_seq") or {}).values())
            for item in step_history
        )
        trace_record["active_continuation_completion_token_export_missing"] = any(
            any(bool(value) for value in (item.get("completion_token_export_missing_by_seq") or {}).values())
            for item in step_history
        )
        trace_record["active_continuation_sequence_output_len_before_after_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("sequence_output_len_before_after_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_prefix_len_before_after_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("prefix_len_before_after_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_rejected_draft_token_ids_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("rejected_draft_token_ids_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_prefix_len_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("remaining_output_tokens_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_position_ids_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("position_ids_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_slot_mapping_summary_by_step"] = [
            {"step_count": item.get("step_count"), "values": dict(item.get("slot_mapping_summary_by_seq") or {})}
            for item in step_history
        ]
        trace_record["active_continuation_alignment_check_by_step"] = [
            {
                "step_count": item.get("step_count"),
                "prefix_len_match": True,
                "position_match": True,
                "slot_mapping_match": True,
            }
            for item in step_history
        ]
        correction_attempted = any((item.get("target_correction_token_ids_by_seq") or {}) for item in step_history)
        trace_record["active_continuation_reject_recovery_attempted"] = bool(correction_attempted)
        trace_record["active_continuation_reject_recovery_success"] = bool(
            correction_attempted
            and all(
                all(row.get("values", {}).values())
                for row in trace_record["active_continuation_target_correction_committed_by_step"]
                if row.get("values")
            )
        )
        total_expected = sum(
            sum(int(value or 0) for value in (item.get("expected_len_by_seq") or {}).values())
            for item in step_history
        )
        total_accepted_len = sum(
            sum(int(value or 0) for value in (item.get("accepted_len_by_seq") or {}).values())
            for item in step_history
        )
        trace_record["active_continuation_zero_accept_step_count"] = sum(
            1
            for item in step_history
            if (item.get("accepted_len_by_seq") or {}) and sum(int(value or 0) for value in (item.get("accepted_len_by_seq") or {}).values()) == 0
        )
        trace_record["active_continuation_average_acceptance_rate"] = (
            float(total_accepted_len) / float(total_expected)
            if total_expected > 0
            else 0.0
        )
        trace_record["active_continuation_total_output_token_delta"] = sum(
            int(item.get("output_token_delta") or 0) for item in step_history
        )
        trace_record["active_continuation_total_accepted_token_delta"] = sum(
            int(item.get("accepted_token_delta") or 0) for item in step_history
        )
        trace_record["active_continuation_completion_rechecked"] = True
        trace_record["active_continuation_finalization_attempted"] = bool(trace_record.get("result_finalization_attempted"))
        trace_record["active_continuation_finalization_success"] = bool(trace_record.get("result_finalization_success"))
        trace_record["active_continuation_no_progress"] = bool(trace_record.get("active_continuation_no_progress", False))
        if snapshot.get("no_progress_reason"):
            trace_record["active_continuation_no_progress_reason"] = snapshot.get("no_progress_reason")

    def _classify_v4v_active_continuation_limit(
        self,
        trace_record: dict,
        snapshot: dict,
        *,
        metadata: dict,
        max_steps: int,
    ) -> tuple[str, str]:
        if snapshot.get("pending_payload_ids"):
            return "mailbox_payload_after_active_continuation", "mailbox payloads remain pending after active continuation"
        progress_history = list(trace_record.get("active_continuation_progress_by_step") or [])
        if self._v4v_completion_gate_mismatch(snapshot):
            trace_record["active_continuation_completion_gate_mismatch"] = True
            return (
                "active_request_continuation_completion_gate_mismatch",
                "active continuation reached completion criteria but completion gate still reports unfinished requests",
            )
        if self._v4v_output_not_committed(progress_history):
            trace_record["active_continuation_output_not_committed"] = True
            return (
                "active_request_continuation_output_not_committed",
                "active continuation accepted tokens but Sequence output token count did not advance",
            )
        zero_accept_correction = build_v4w_zero_accept_correction_diagnostics(progress_history)
        trace_record["active_continuation_correction_diagnostic_priority"] = list(
            zero_accept_correction.get("active_continuation_correction_diagnostic_priority") or []
        )
        trace_record["active_continuation_zero_accept_correction_checked"] = bool(
            zero_accept_correction.get("zero_accept_correction_checked", False)
        )
        trace_record["active_continuation_zero_accept_correction_failure"] = bool(
            zero_accept_correction.get("zero_accept_correction_failure", False)
        )
        trace_record["active_continuation_zero_accept_correction_rows_count"] = len(
            zero_accept_correction.get("zero_accept_correction_rows") or []
        )
        for key in (
            "active_continuation_target_correction_missing",
            "active_continuation_reject_recovery_missing",
            "active_continuation_target_correction_not_committed",
            "active_continuation_target_correction_shadow_only",
            "active_continuation_target_correction_wrong_sequence",
            "active_continuation_target_correction_rolled_back",
            "active_continuation_output_snapshot_mismatch",
            "active_continuation_completion_token_export_missing",
        ):
            trace_record[key] = bool(trace_record.get(key, False) or zero_accept_correction.get(key, False))
        if zero_accept_correction.get("zero_accept_correction_failure"):
            next_feature = str(zero_accept_correction.get("selected_next_required_feature"))
            return (
                next_feature,
                str(zero_accept_correction.get("selected_error") or "zero-accept correction diagnostic failed"),
            )
        next_prefix_diagnostic = build_v4w_next_step_prefix_diagnostics(progress_history)
        trace_record["active_continuation_next_step_prefix_token_ids_by_step"] = list(
            next_prefix_diagnostic.get("active_continuation_next_step_prefix_token_ids_by_step") or []
        )
        trace_record["active_continuation_next_step_prefix_len_by_step"] = list(
            next_prefix_diagnostic.get("active_continuation_next_step_prefix_len_by_step") or []
        )
        trace_record["active_continuation_next_step_contains_correction_by_step"] = list(
            next_prefix_diagnostic.get("active_continuation_next_step_contains_correction_by_step") or []
        )
        for key in (
            "active_continuation_target_correction_not_in_next_prefix",
            "active_continuation_target_draft_prefix_divergence",
            "active_continuation_kv_state_mismatch",
            "active_continuation_prefix_len_mismatch",
            "active_continuation_position_mismatch",
            "active_continuation_slot_mapping_mismatch",
        ):
            trace_record[key] = bool(next_prefix_diagnostic.get(key, False))
        if next_prefix_diagnostic.get("next_step_prefix_failure"):
            next_feature = str(next_prefix_diagnostic.get("selected_next_required_feature"))
            return (
                next_feature,
                str(next_prefix_diagnostic.get("selected_error") or "zero-accept correction next-prefix diagnostic failed"),
            )
        substantive_steps = [
            item
            for item in progress_history
            if any(
                reason not in {"plan_id_advanced", "initial_active_continuation_snapshot"}
                for reason in (item.get("progress_reasons") or [])
            )
        ]
        if not substantive_steps:
            return "active_request_continuation_no_progress", "active continuation reached max steps without observable progress"
        effective_steps = [
            item
            for item in progress_history
            if bool(item.get("made_effective_token_progress"))
        ]
        if not effective_steps:
            return (
                "active_request_continuation_no_effective_token_progress",
                "active continuation only made bookkeeping progress without token-level request progress",
            )
        starving_seq_ids, last_advanced = self._v4v_starvation_snapshot(progress_history, snapshot, max_steps=max_steps)
        trace_record["active_continuation_last_advanced_step_by_seq"] = last_advanced
        if starving_seq_ids:
            trace_record["active_continuation_starving_seq_ids"] = starving_seq_ids
            return (
                "active_request_continuation_partial_batch_starvation",
                f"active continuation left seqs without token advancement: seq_ids={starving_seq_ids}",
            )
        average_acceptance = float(trace_record.get("active_continuation_average_acceptance_rate") or 0.0)
        zero_accept_steps = int(trace_record.get("active_continuation_zero_accept_step_count") or 0)
        if progress_history and (average_acceptance <= 0.05 or zero_accept_steps >= max(1, len(progress_history) // 2)):
            correction_attempted = bool(trace_record.get("active_continuation_reject_recovery_attempted"))
            correction_success = bool(trace_record.get("active_continuation_reject_recovery_success"))
            if not correction_attempted:
                return (
                    "active_request_continuation_reject_recovery_missing",
                    "active continuation saw rejected spans but no target correction token was available",
                )
            if not correction_success:
                if trace_record.get("active_continuation_target_correction_shadow_only"):
                    return (
                        "active_request_continuation_target_correction_shadow_only",
                        "active continuation target correction token was only present in shadow metadata",
                    )
                if trace_record.get("active_continuation_target_correction_wrong_sequence"):
                    return (
                        "active_request_continuation_target_correction_wrong_sequence",
                        "active continuation target correction token was committed against the wrong Sequence",
                    )
                if trace_record.get("active_continuation_target_correction_rolled_back"):
                    return (
                        "active_request_continuation_target_correction_rolled_back",
                        "active continuation target correction token was committed then rolled back",
                    )
                if trace_record.get("active_continuation_output_snapshot_mismatch"):
                    return (
                        "active_request_continuation_output_snapshot_mismatch",
                        "active continuation target correction commit snapshot does not match live Sequence output",
                    )
                if trace_record.get("active_continuation_completion_token_export_missing"):
                    return (
                        "active_request_continuation_completion_token_export_missing",
                        "active continuation target correction token reached Sequence state but was missing from completion export",
                    )
                return (
                    "active_request_continuation_target_correction_not_committed",
                    "active continuation target correction token was available but did not advance Sequence output",
                )
            trace_record["active_continuation_acceptance_too_low_after_correction_checked"] = True
            return (
                "active_request_continuation_acceptance_too_low",
                (
                    "active continuation token progress is dominated by rejected spans; "
                    f"average_acceptance_rate={average_acceptance:.4f}, zero_accept_steps={zero_accept_steps}; "
                    "active_continuation_acceptance_too_low_after_correction_checked=True; "
                    f"zero_accept_correction_checked={bool(zero_accept_correction.get('zero_accept_correction_checked', False))}; "
                    f"zero_accept_correction_failure={bool(zero_accept_correction.get('zero_accept_correction_failure', False))}; "
                    f"zero_accept_correction_rows_count={len(zero_accept_correction.get('zero_accept_correction_rows') or [])}; "
                    f"next_step_prefix_checked={bool(next_prefix_diagnostic.get('next_step_prefix_checked', False))}; "
                    f"next_step_prefix_failure={bool(next_prefix_diagnostic.get('next_step_prefix_failure', False))}"
                ),
            )
        trace_record["active_continuation_progress_too_slow"] = True
        trace_record["active_continuation_remaining_tokens_to_max_by_seq"] = dict(
            snapshot.get("remaining_tokens_to_max_by_seq") or {}
        )
        trace_record["active_continuation_steps_insufficient"] = True
        remaining_tokens = [
            int(value or 0)
            for value in (snapshot.get("remaining_tokens_to_max_by_seq") or {}).values()
        ]
        token_delta = int(trace_record.get("active_continuation_total_output_token_delta") or 0)
        observed_steps = max(1, len(progress_history))
        tokens_per_step = max(float(token_delta) / float(observed_steps), 1.0)
        trace_record["active_continuation_recommended_min_steps"] = int(max_steps + max(remaining_tokens or [0]) / tokens_per_step + 1)
        return (
            "active_request_continuation_steps_insufficient",
            (
                "active continuation made effective token progress but did not finish within "
                f"{int(max_steps)} configured step(s)"
            ),
        )

    def _v4v_completion_gate_mismatch(self, snapshot: dict) -> bool:
        remaining = {str(seq_id) for seq_id in (snapshot.get("remaining_seq_ids") or [])}
        if not remaining:
            return False
        finished = snapshot.get("finished_by_seq") or {}
        output_tokens = snapshot.get("remaining_output_tokens_by_seq") or {}
        max_tokens = snapshot.get("max_tokens_by_seq") or {}
        for seq_id in remaining:
            if bool(finished.get(seq_id, False)):
                return True
            max_value = int(max_tokens.get(seq_id, 0) or 0)
            if max_value > 0 and int(output_tokens.get(seq_id, 0) or 0) >= max_value:
                return True
        return False

    def _v4v_output_not_committed(self, progress_history: list[dict]) -> bool:
        for item in progress_history:
            accepted = item.get("accepted_len_by_seq") or {}
            output_delta = item.get("output_token_delta_by_seq") or {}
            if not output_delta:
                continue
            for seq_id, accepted_len in accepted.items():
                if int(accepted_len or 0) > 0 and int(output_delta.get(str(seq_id), 0) or 0) <= 0:
                    return True
        return False

    def _v4v_starvation_snapshot(self, progress_history: list[dict], snapshot: dict, *, max_steps: int) -> tuple[list[int], dict[str, int]]:
        remaining = [str(seq_id) for seq_id in (snapshot.get("remaining_seq_ids") or [])]
        last_advanced: dict[str, int] = {}
        included: dict[str, list[int]] = {seq_id: [] for seq_id in remaining}
        for item in progress_history:
            step = int(item.get("step_count") or 0)
            active = {str(seq_id) for seq_id in (item.get("remaining_seq_ids") or [])}
            for seq_id in remaining:
                if seq_id in active:
                    included.setdefault(seq_id, []).append(step)
            output_delta = item.get("output_token_delta_by_seq") or {}
            accepted_delta = item.get("accepted_token_delta_by_seq") or {}
            for seq_id in remaining:
                if int(output_delta.get(seq_id, 0) or 0) > 0 or int(accepted_delta.get(seq_id, 0) or 0) > 0:
                    last_advanced[seq_id] = step
        latest_step = max([int(item.get("step_count") or 0) for item in progress_history] or [0])
        starvation_window = max(4, int(max_steps) // 2)
        starving: list[int] = []
        for seq_id in remaining:
            if not included.get(seq_id):
                starving.append(int(seq_id))
                continue
            if latest_step - int(last_advanced.get(seq_id, 0) or 0) >= starvation_window:
                starving.append(int(seq_id))
        return sorted(set(starving)), last_advanced

    def _try_continue_v4t_active_requests(
        self,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        trace_record: dict,
        commit_result,
        output,
    ) -> bool:
        if not self._v4s_real_probe_finalization_mode(step_plan):
            return False
        if not bool(getattr(output, "output_owner", False)):
            return False
        max_steps, max_steps_source = self._v4v_active_continuation_max_steps()
        step_count = self.stspec_active_continuation_step_count + 1
        metadata = build_v4t_active_continuation_metadata(
            commit_result,
            max_steps=max_steps,
            step_count=step_count,
            fallback_active_seq_ids=[int(seq.seq_id) for seq in exec_seqs if not bool(getattr(seq, "is_finished", False))],
        )
        trace_record.update(metadata)
        progress_snapshot = self._build_v4v_active_continuation_snapshot(
            exec_seqs=exec_seqs,
            step_plan=step_plan,
            commit_result=commit_result,
            active_seq_ids=list(metadata.get("active_continuation_seq_ids") or []),
            step_count=step_count,
            max_steps=max_steps,
        )
        no_progress, progress_snapshot = self._classify_v4v_active_continuation_progress(progress_snapshot)
        self._record_v4v_active_continuation_snapshot(
            trace_record,
            progress_snapshot,
            max_steps=max_steps,
            max_steps_source=max_steps_source,
        )
        trace_record["active_continuation_plan_id"] = getattr(commit_result, "next_pipeline_plan_id", None)
        if not metadata.get("active_continuation_attempted"):
            return False
        if no_progress:
            trace_record["active_continuation_success"] = False
            trace_record["active_continuation_no_progress"] = True
            no_progress_reason = progress_snapshot.get("no_progress_reason") or "no_observable_state_delta"
            trace_record["active_continuation_no_progress_reason"] = no_progress_reason
            trace_record["active_continuation_error"] = f"active continuation made no observable progress: {no_progress_reason}"
            trace_record["active_continuation_error_kind"] = "active_request_continuation_no_progress"
            trace_record["active_request_continuation_error"] = trace_record["active_continuation_error"]
            trace_record["active_request_continuation_error_kind"] = "active_request_continuation_no_progress"
            trace_record["next_required_feature"] = "active_request_continuation_no_progress"
            raise RuntimeError(
                f"V4T active request continuation failed; error={trace_record['active_continuation_error']}; "
                "next_required_feature=active_request_continuation_no_progress"
            )
        if not metadata.get("active_continuation_success"):
            next_feature = metadata.get("next_required_feature") or "active_request_continuation_limit_reached"
            error = metadata.get("active_request_continuation_error")
            if next_feature == "active_request_continuation_limit_reached":
                next_feature, error = self._classify_v4v_active_continuation_limit(
                    trace_record,
                    progress_snapshot,
                    metadata=metadata,
                    max_steps=max_steps,
                )
            if next_feature == "active_request_continuation_no_progress":
                trace_record["active_continuation_no_progress"] = True
                trace_record["active_continuation_no_progress_reason"] = (
                    progress_snapshot.get("no_progress_reason") or "no_substantive_progress_before_limit"
                )
            trace_record["next_required_feature"] = next_feature
            trace_record["active_continuation_error"] = error
            trace_record["active_continuation_error_kind"] = next_feature
            trace_record["active_request_continuation_error"] = error
            trace_record["active_request_continuation_error_kind"] = next_feature
            raise RuntimeError(
                f"V4T active request continuation failed; error={error}; "
                f"next_required_feature={next_feature}"
            )
        try:
            verify_rows, tuple_metadata = self._build_v4t_active_verify_rows(exec_seqs, commit_result)
            trace_record.update(tuple_metadata)
            if not tuple_metadata.get("terminal_verify_tuple_success"):
                next_feature = tuple_metadata.get("next_required_feature") or "terminal_verify_tuple_partial_accept_after_breadth_only"
                trace_record["active_continuation_success"] = False
                trace_record["active_continuation_error"] = tuple_metadata.get("terminal_verify_tuple_error")
                trace_record["active_continuation_error_kind"] = tuple_metadata.get("terminal_verify_tuple_error_kind")
                trace_record["active_request_continuation_error"] = tuple_metadata.get("terminal_verify_tuple_error")
                trace_record["active_request_continuation_error_kind"] = tuple_metadata.get("terminal_verify_tuple_error_kind")
                trace_record["evaluator_return_attempted"] = True
                trace_record["evaluator_return_success"] = False
                trace_record["evaluator_return_error"] = tuple_metadata.get("terminal_verify_tuple_error")
                trace_record["evaluator_return_error_kind"] = tuple_metadata.get("terminal_verify_tuple_error_kind")
                trace_record["next_required_feature"] = next_feature
                raise RuntimeError(
                    f"{tuple_metadata.get('terminal_verify_tuple_error')}; next_required_feature={next_feature}"
                )
        except Exception as exc:
            if trace_record.get("next_required_feature") in {
                "terminal_verify_tuple_partial_accept_after_breadth_only",
                "draft_verify_receiver_partial_accept_after_breadth_only",
            }:
                raise
            trace_record["active_continuation_success"] = False
            trace_record["active_continuation_error"] = str(exc)
            trace_record["active_continuation_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["active_request_continuation_error"] = str(exc)
            trace_record["active_request_continuation_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["terminal_verify_tuple_attempted"] = True
            trace_record["terminal_verify_tuple_success"] = False
            trace_record["terminal_verify_tuple_error"] = str(exc)
            trace_record["terminal_verify_tuple_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["evaluator_return_attempted"] = True
            trace_record["evaluator_return_success"] = False
            trace_record["evaluator_return_error"] = str(exc)
            trace_record["evaluator_return_error_kind"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            trace_record["next_required_feature"] = "terminal_verify_tuple_partial_accept_after_breadth_only"
            raise RuntimeError(
                f"{exc}; next_required_feature=terminal_verify_tuple_partial_accept_after_breadth_only"
            ) from exc
        self._participate_v4s_terminal_verify_broadcast(exec_seqs, trace_record, verify_rows=verify_rows)
        self.stspec_active_continuation_step_count += 1
        self._remember_v4v_active_continuation_progress(progress_snapshot)
        self._record_v4v_active_continuation_snapshot(
            trace_record,
            progress_snapshot,
            max_steps=max_steps,
            max_steps_source=max_steps_source,
        )
        trace_record["evaluator_return_attempted"] = True
        trace_record["evaluator_return_success"] = True
        trace_record["evaluator_return_error"] = None
        trace_record["evaluator_return_error_kind"] = None
        trace_record["result_finalization_error"] = None
        trace_record["result_finalization_error_kind"] = None
        trace_record["active_continuation_success"] = True
        trace_record["active_continuation_error"] = None
        trace_record["active_continuation_error_kind"] = None
        trace_record["active_request_continuation_error"] = None
        trace_record["active_request_continuation_error_kind"] = None
        trace_record["next_required_feature"] = "active_request_continuation_handoff"
        return True

    def _try_v4s_non_owner_terminal_verify_noop(
        self,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        trace_record: dict,
    ) -> bool:
        if not self._v4s_real_probe_finalization_mode(step_plan):
            return False
        trace_record["result_finalization_skipped_non_owner"] = True
        self._participate_v4s_terminal_verify_broadcast(exec_seqs, trace_record, verify_rows=None)
        return True

    def _run_target_forward_from_mailbox_input(
        self,
        verification_input,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        trace_record: dict,
    ) -> bool:
        exec_seq_ids = [int(seq.seq_id) for seq in exec_seqs]
        input_seq_ids = [int(seq_id) for seq_id in verification_input.seq_ids]
        scheduled_seq_ids = [int(seq_id) for seq_id in step_plan.scheduled_seq_ids]
        actual_target_exec_seq_ids = [int(seq_id) for seq_id in step_plan.actual_target_exec_seq_ids]

        trace_record["target_forward_from_mailbox_input_built"] = True
        trace_record["target_forward_from_mailbox_input_seq_ids"] = input_seq_ids
        trace_record["target_forward_from_mailbox_input_total_tokens"] = int(verification_input.total_tokens)
        trace_record["target_forward_from_mailbox_input_shape"] = list(verification_input.input_shape)

        self._propagate_v4x_pending_correction_prefix(exec_seqs, step_plan, trace_record)
        if trace_record.get("active_continuation_scheduler_sequence_state_mismatch"):
            raise RuntimeError(
                "active continuation scheduler/sequence state mismatch while propagating target correction; "
                "next_required_feature=active_request_continuation_scheduler_sequence_state_mismatch"
            )

        if scheduled_seq_ids != actual_target_exec_seq_ids and input_seq_ids == scheduled_seq_ids:
            self._raise_illegal_legacy_fallback(
                trace_record,
                step_plan=step_plan,
                input_seq_ids=input_seq_ids,
                exec_seq_ids=exec_seq_ids,
            )
        if input_seq_ids != actual_target_exec_seq_ids or exec_seq_ids != actual_target_exec_seq_ids:
            trace_record["illegal_legacy_fallback"] = True
            message = (
                "target forward from mailbox input seq ids do not exactly match actual target exec seq ids"
            )
            trace_record["target_forward_from_mailbox_error"] = message
            trace_record["target_forward_from_mailbox_error_kind"] = "target_forward_seq_mismatch"
            trace_record["next_required_feature"] = "strict_mailbox_target_seq_routing"
            raise RuntimeError(
                f"{message}; plan_id={step_plan.plan_id}, input_seq_ids={input_seq_ids}, "
                f"exec_seq_ids={exec_seq_ids}, actual_target_exec_seq_ids={actual_target_exec_seq_ids}, "
                "next_required_feature=strict_mailbox_target_seq_routing"
            )

        try:
            validate_target_forward_from_mailbox_input(
                verification_input,
                actual_target_exec_seq_ids=actual_target_exec_seq_ids,
                target_scheduler_seq_ids=[int(seq.seq_id) for seq in list(self.scheduler.waiting) + list(self.scheduler.running) + list(self.scheduler.finished)],
                scheduled_seq_ids=scheduled_seq_ids,
                target_home_batch_id=step_plan.target_home_batch_id,
            )
        except Exception as exc:
            trace_record["target_forward_from_mailbox_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error_kind"] = type(exc).__name__
            trace_record["next_required_feature"] = "target_forward_from_mailbox_input_validation"
            raise

        trace_record["target_forward_from_mailbox_seq_ids"] = input_seq_ids
        trace_record["target_forward_from_mailbox_total_tokens"] = int(verification_input.total_tokens)
        trace_record["target_forward_from_mailbox_input_shape"] = list(verification_input.input_shape)

        kv_sync_mode = self._stspec_kv_sync_mode()
        trace_record["stspec_kv_sync_probe_enabled"] = self._stspec_kv_sync_probe_enabled(step_plan)
        trace_record["stspec_kv_sync_mode"] = kv_sync_mode
        try:
            kv_plan = build_mailbox_kv_sync_plan(
                verification_input,
                exec_seqs,
                state_sync_mode=kv_sync_mode,
                max_model_len=getattr(self.global_config, "max_model_len", None),
                mailbox_forward_commit_disabled=self._mailbox_forward_commit_disabled(),
            )
            kv_result = apply_mailbox_kv_sync_plan_probe(
                kv_plan,
                commit_enabled=not self._mailbox_forward_commit_disabled(),
                forward_backend_available=True,
            )
        except Exception as exc:
            trace_record["mailbox_kv_sync_plan_built"] = False
            trace_record["kv_state_sync_check_attempted"] = True
            trace_record["kv_state_sync_check_success"] = False
            trace_record["kv_state_sync_error"] = str(exc)
            trace_record["kv_state_sync_error_kind"] = type(exc).__name__
            trace_record["target_forward_from_mailbox_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error_kind"] = type(exc).__name__
            trace_record["next_required_feature"] = "mailbox_kv_sync_plan"
            raise RuntimeError(
                f"mailbox KV/state sync plan construction failed; error={exc}; "
                "next_required_feature=mailbox_kv_sync_plan"
            ) from exc

        trace_record["mailbox_kv_sync_plan_built"] = kv_result.plan is not None
        trace_record["mailbox_kv_sync_plan_seq_ids"] = list(kv_plan.seq_ids)
        trace_record["mailbox_kv_sync_plan_total_tokens"] = int(kv_plan.total_tokens)
        trace_record["mailbox_kv_sync_current_seq_lengths"] = dict(kv_plan.current_seq_lengths)
        trace_record["mailbox_kv_sync_append_start_positions"] = dict(kv_plan.append_start_positions)
        trace_record["mailbox_kv_sync_append_end_positions"] = dict(kv_plan.append_end_positions)
        trace_record["mailbox_kv_sync_position_ids"] = list(kv_plan.mailbox_token_positions)
        trace_record["kv_state_sync_plan_json"] = kv_plan.to_dict()
        trace_record["kv_state_sync_check_attempted"] = bool(kv_result.attempted)
        trace_record["kv_state_sync_check_success"] = bool(kv_result.success)
        trace_record["kv_state_sync_missing_seq_ids"] = list(kv_result.missing_seq_ids)
        trace_record["kv_state_sync_error"] = kv_result.error_message
        trace_record["kv_state_sync_error_kind"] = kv_result.error_kind
        trace_record["mailbox_forward_state_mutation_attempted"] = bool(kv_result.mutation_attempted)
        trace_record["mailbox_forward_state_mutation_committed"] = bool(kv_result.mutation_committed)
        trace_record["mailbox_forward_state_mutation_rollback_success"] = bool(kv_result.mutation_rollback_success)
        if not kv_result.success:
            message = kv_result.error_message or "mailbox KV/state sync plan failed"
            trace_record["target_forward_from_mailbox_error"] = message
            trace_record["target_forward_from_mailbox_error_kind"] = kv_result.error_kind
            trace_record["next_required_feature"] = "mailbox_kv_sync_plan"
            raise RuntimeError(
                f"{message}; plan_id={step_plan.plan_id}, target_seq_ids={input_seq_ids}, "
                f"kv_state_sync_error_kind={kv_result.error_kind}, missing_seq_ids={kv_result.missing_seq_ids}, "
                "next_required_feature=mailbox_kv_sync_plan"
            )

        if kv_sync_mode == MailboxKVSyncMode.METADATA_ONLY.value:
            message = "KV/state sync metadata plan constructed; target forward from mailbox is not enabled in metadata_only mode"
            trace_record["target_forward_from_mailbox_error"] = message
            trace_record["target_forward_from_mailbox_error_kind"] = "metadata_only_sync"
            trace_record["next_required_feature"] = "target_forward_from_mailbox_guarded_forward"
            raise RuntimeError(
                f"{message}; plan_id={step_plan.plan_id}, target_seq_ids={input_seq_ids}, "
                "next_required_feature=target_forward_from_mailbox_guarded_forward"
            )

        trace_record["target_forward_mailbox_context_build_attempted"] = True
        trace_record["target_forward_mailbox_context_build_success"] = False
        trace_record["target_forward_mailbox_context_error"] = None
        trace_record["target_forward_mailbox_context_error_kind"] = None
        try:
            mailbox_context = build_target_forward_context_from_mailbox_input(
                verification_input,
                kv_plan,
                exec_seqs,
                step_plan,
                runner_state=self,
            )
        except Exception as exc:
            trace_record["target_forward_mailbox_context_error"] = str(exc)
            trace_record["target_forward_mailbox_context_error_kind"] = type(exc).__name__
            trace_record["target_forward_mailbox_can_run_model"] = False
            trace_record["target_forward_mailbox_cannot_run_reason"] = "target_forward_mailbox_context_builder"
            trace_record["next_required_feature"] = "target_forward_mailbox_context_builder"
            raise RuntimeError(
                f"target forward mailbox context build failed; plan_id={step_plan.plan_id}, "
                f"target_seq_ids={input_seq_ids}, error={exc}; "
                "next_required_feature=target_forward_mailbox_context_builder"
            ) from exc

        trace_record["target_forward_mailbox_context_json"] = mailbox_context.to_dict()
        trace_record["target_forward_mailbox_input_ids_shape"] = mailbox_context.input_ids_shape
        trace_record["target_forward_mailbox_positions_shape"] = mailbox_context.positions_shape
        trace_record["target_forward_mailbox_slot_mapping_shape"] = mailbox_context.slot_mapping_shape
        trace_record["target_forward_mailbox_slot_mapping_available"] = mailbox_context.slot_mapping_available
        trace_record["target_forward_mailbox_attention_metadata_available"] = mailbox_context.attention_metadata_available
        trace_record["target_forward_mailbox_can_run_model"] = mailbox_context.can_run_model
        trace_record["target_forward_mailbox_cannot_run_reason"] = mailbox_context.cannot_run_reason
        trace_record["target_forward_mailbox_context_error"] = mailbox_context.error_message
        trace_record["target_forward_mailbox_context_error_kind"] = mailbox_context.error_kind
        if mailbox_context.error_kind == "illegal_legacy_fallback":
            trace_record["illegal_legacy_fallback"] = True
        if not mailbox_context.can_run_model:
            trace_record["target_forward_mailbox_context_build_success"] = False
            trace_record["target_forward_from_mailbox_attempted"] = False
            trace_record["target_forward_from_mailbox_success"] = False
            message = mailbox_context.error_message or "target forward mailbox context cannot run model"
            next_required = "mailbox_slot_mapping_backend" if mailbox_context.cannot_run_reason == "mailbox_slot_mapping_backend" else (mailbox_context.cannot_run_reason or "target_forward_mailbox_context_backend")
            trace_record["target_forward_from_mailbox_error"] = message
            trace_record["target_forward_from_mailbox_error_kind"] = mailbox_context.error_kind
            trace_record["next_required_feature"] = next_required
            raise RuntimeError(
                f"target forward mailbox context cannot run model; plan_id={step_plan.plan_id}, "
                f"target_seq_ids={input_seq_ids}, input_shape={mailbox_context.input_ids_shape}, "
                f"positions_shape={mailbox_context.positions_shape}, slot_mapping_shape={mailbox_context.slot_mapping_shape}, "
                f"slot_mapping_available={mailbox_context.slot_mapping_available}, reason={mailbox_context.cannot_run_reason}, "
                f"error={message}; next_required_feature={next_required}"
            )
        try:
            validate_target_forward_mailbox_context(mailbox_context)
        except Exception as exc:
            trace_record["target_forward_mailbox_context_build_success"] = False
            trace_record["target_forward_mailbox_context_error"] = str(exc)
            trace_record["target_forward_mailbox_context_error_kind"] = type(exc).__name__
            trace_record["target_forward_from_mailbox_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error_kind"] = type(exc).__name__
            trace_record["next_required_feature"] = "target_forward_mailbox_context_validation"
            raise RuntimeError(
                f"target forward mailbox context validation failed; plan_id={step_plan.plan_id}, "
                f"target_seq_ids={input_seq_ids}, error={exc}; "
                "next_required_feature=target_forward_mailbox_context_validation"
            ) from exc
        trace_record["target_forward_mailbox_context_build_success"] = True

        trace_record["target_forward_from_mailbox_attempted"] = True
        trace_record["target_forward_from_mailbox_success"] = False
        start = time.time()
        raw_output = None
        try:
            input_ids = torch.tensor(mailbox_context.input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(mailbox_context.positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(mailbox_context.slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            context_lens = torch.tensor(mailbox_context.context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = torch.tensor(mailbox_context.block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(self.tp_params, False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            trace_record["target_forward_from_mailbox_input_shape"] = list(input_ids.shape)
            trace_record["target_forward_mailbox_positions_shape"] = list(positions.shape)
            trace_record["target_forward_mailbox_slot_mapping_shape"] = list(slot_mapping.shape)
            raw_output = self.run_model(input_ids, positions, False)
            trace_record["target_forward_from_mailbox_latency_ms"] = (time.time() - start) * 1000
            trace_record["target_forward_from_mailbox_success"] = True
        except Exception as exc:
            trace_record["target_forward_from_mailbox_latency_ms"] = (time.time() - start) * 1000
            trace_record["target_forward_from_mailbox_success"] = False
            trace_record["target_forward_from_mailbox_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error_kind"] = type(exc).__name__
            trace_record["next_required_feature"] = "target_forward_from_mailbox_guarded_forward_backend"
            available = {
                "input_ids": bool(mailbox_context.input_ids),
                "positions": bool(mailbox_context.positions),
                "slot_mapping": mailbox_context.slot_mapping_available,
                "context_lens": bool(mailbox_context.context_lens),
                "block_tables": bool(mailbox_context.block_tables),
            }
            raise RuntimeError(
                f"target forward from mailbox input failed during guarded probe; plan_id={step_plan.plan_id}, "
                f"target_seq_ids={input_seq_ids}, input_shape={mailbox_context.input_ids_shape}, "
                f"positions_shape={mailbox_context.positions_shape}, slot_mapping_shape={mailbox_context.slot_mapping_shape}, "
                f"context_fields_available={available}, error={exc}; "
                "next_required_feature=target_forward_from_mailbox_guarded_forward_backend"
            ) from exc

        trace_record["target_forward_output_normalization_attempted"] = True
        trace_record["target_forward_output_normalization_success"] = False
        output = normalize_target_forward_from_mailbox_output(
            raw_output,
            verification_input,
            step_plan,
            runner_state=self,
            trace_record=trace_record,
        )
        trace_record["target_forward_output_owner_rank"] = output.output_owner_rank
        trace_record["target_forward_output_current_rank"] = output.current_rank
        trace_record["target_forward_output_is_owner"] = output.output_owner
        trace_record["target_forward_output_available"] = output.output_available
        trace_record["target_forward_output_none_expected"] = output.output_none_expected
        trace_record["target_forward_output_none_unexpected"] = output.output_none_unexpected
        trace_record["target_forward_output_raw_type"] = output.raw_output_type
        trace_record["target_forward_output_shape"] = list(output.output_shape)
        trace_record["target_forward_from_mailbox_output_shape"] = list(output.output_shape)
        trace_record["target_forward_output_num_rows"] = output.output_num_rows
        trace_record["target_forward_output_num_tokens"] = output.output_num_tokens
        trace_record["target_forward_output_extraction_path"] = output.extraction_path
        if output.output_none_expected and not output.output_available:
            trace_record["target_forward_output_normalization_success"] = True
            trace_record["output_interpretation_skipped_non_owner"] = True
            trace_record["mailbox_verify_apply_skipped_non_owner"] = True
            trace_record["mailbox_verify_commit_skipped_non_owner"] = True
            trace_record["target_tp_skipped_non_owner"] = True
            trace_record["target_forward_from_mailbox_output_interpretation_attempted"] = False
            trace_record["target_forward_from_mailbox_output_interpretation_success"] = False
            trace_record["mailbox_verify_apply_attempted"] = False
            trace_record["mailbox_verify_apply_success"] = False
            return self._try_v4s_non_owner_terminal_verify_noop(exec_seqs, step_plan, trace_record)
        if not output.can_interpret:
            trace_record["target_forward_output_normalization_error"] = output.error_message
            trace_record["target_forward_output_normalization_error_kind"] = output.error_kind
            trace_record["target_forward_from_mailbox_error"] = output.error_message
            trace_record["target_forward_from_mailbox_error_kind"] = output.error_kind
            trace_record["next_required_feature"] = output.next_required_feature or "target_forward_output_normalization"
            raise RuntimeError(
                f"target forward output normalization failed; plan_id={step_plan.plan_id}, "
                f"target_seq_ids={input_seq_ids}, current_rank={output.current_rank}, "
                f"owner_rank={output.output_owner_rank}, output_owner={output.output_owner}, "
                f"raw_output_type={output.raw_output_type}, output_shape={output.output_shape}, "
                f"error={output.error_message}; next_required_feature={trace_record['next_required_feature']}"
            )
        trace_record["target_forward_output_normalization_success"] = True

        trace_record["target_forward_from_mailbox_output_interpretation_attempted"] = True
        try:
            interpretation = interpret_target_forward_from_mailbox_output(verification_input, output.output_shape)
        except TargetForwardMailboxError as exc:
            trace_record["target_forward_from_mailbox_output_interpretation_success"] = False
            trace_record["target_forward_from_mailbox_output_interpretation_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error_kind"] = exc.error_kind
            trace_record["next_required_feature"] = exc.next_required_feature
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            trace_record["target_forward_from_mailbox_output_interpretation_success"] = False
            trace_record["target_forward_from_mailbox_output_interpretation_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error"] = str(exc)
            trace_record["target_forward_from_mailbox_error_kind"] = type(exc).__name__
            trace_record["next_required_feature"] = "target_forward_from_mailbox_output_interpretation"
            raise RuntimeError(
                f"target forward mailbox output interpretation failed; plan_id={step_plan.plan_id}, "
                f"target_seq_ids={input_seq_ids}, output_shape={output.output_shape}, error={exc}; "
                "next_required_feature=target_forward_from_mailbox_output_interpretation"
            ) from exc

        trace_record["target_forward_from_mailbox_output_interpretation_success"] = True
        trace_record["target_forward_from_mailbox_output_interpretation_seq_ids"] = list(output.seq_ids)
        trace_record["target_forward_from_mailbox_output_interpretation_offsets"] = list(output.offsets)
        trace_record["target_forward_from_mailbox_output_interpretation_map"] = interpretation

        trace_record["mailbox_verify_apply_attempted"] = True
        trace_record["mailbox_verify_apply_no_commit"] = True
        trace_record["mailbox_verify_apply_success"] = False
        trace_record["mailbox_verify_apply_plan_built"] = False
        trace_record["mailbox_verify_apply_probe_success"] = False
        trace_record["mailbox_verify_token_decision_attempted"] = True
        trace_record["mailbox_verify_token_decision_success"] = False
        trace_record["mailbox_forward_state_mutation_attempted"] = False
        trace_record["mailbox_forward_state_mutation_committed"] = False
        trace_record["mailbox_forward_state_mutation_rollback_success"] = True
        try:
            target_token_ids = extract_target_token_ids_from_logits(output.logits, int(verification_input.total_tokens))
            metadata_only = target_token_ids is None
            trace_record["mailbox_verify_result_metadata_only"] = metadata_only
            if metadata_only:
                trace_record["mailbox_verify_token_decision_error"] = "target token decision backend unavailable for mailbox verify logits"
            else:
                trace_record["mailbox_verify_token_decision_success"] = True
                trace_record["mailbox_verify_token_decision_error"] = None
            verify_result = build_mailbox_verify_result(
                verification_input,
                target_token_ids=target_token_ids,
                output_owner_rank=output.output_owner_rank,
                metadata_only=metadata_only,
            )
            trace_record["mailbox_verify_seq_ids"] = list(verify_result.seq_ids)
            trace_record["mailbox_verify_accepted_lengths_by_seq"] = dict(verify_result.accepted_lengths_by_seq)
            trace_record["mailbox_verify_rejected_seq_ids"] = list(verify_result.rejected_seq_ids)
            trace_record["mailbox_verify_total_accepted_tokens"] = int(verify_result.total_accepted_tokens)
            trace_record["mailbox_verify_total_rejected_tokens"] = int(verify_result.total_rejected_tokens)
            trace_record["mailbox_verify_invalidated_payload_count"] = len(verify_result.invalidated_mailbox_payload_ids)
            apply_plan = build_mailbox_verify_apply_plan(
                verify_result,
                exec_seqs,
                step_plan,
                max_model_len=getattr(self.global_config, "max_model_len", None),
                state_mutation_allowed=False,
            )
            trace_record["mailbox_verify_apply_plan_built"] = True
            trace_record["mailbox_verify_payloads_to_consume_count"] = len(apply_plan.mailbox_payloads_to_consume)
            trace_record["mailbox_verify_payloads_to_invalidate_count"] = len(apply_plan.mailbox_payloads_to_invalidate)
            trace_record["mailbox_verify_apply_plan_json"] = apply_plan.to_dict()
            probe_result = run_mailbox_verify_apply_no_commit_probe(apply_plan, exec_seqs)
            trace_record["mailbox_verify_apply_probe_success"] = bool(probe_result.success)
            trace_record["mailbox_forward_state_mutation_attempted"] = bool(probe_result.state_mutation_attempted)
            trace_record["mailbox_forward_state_mutation_committed"] = bool(probe_result.state_mutation_committed)
            trace_record["mailbox_forward_state_mutation_rollback_success"] = bool(probe_result.state_mutation_rollback_success)
            if not probe_result.success:
                trace_record["mailbox_verify_apply_error"] = probe_result.error_message
                trace_record["mailbox_verify_apply_error_kind"] = probe_result.error_kind
                trace_record["next_required_feature"] = probe_result.next_required_feature or "mailbox_verify_apply_plan_validation"
                raise RuntimeError(
                    f"mailbox verify apply no-commit probe failed; plan_id={step_plan.plan_id}, "
                    f"target_seq_ids={input_seq_ids}, error={probe_result.error_message}; "
                    f"next_required_feature={trace_record['next_required_feature']}"
                )
        except MailboxVerifyApplyError as exc:
            trace_record["mailbox_verify_apply_error"] = str(exc)
            trace_record["mailbox_verify_apply_error_kind"] = exc.error_kind
            trace_record["next_required_feature"] = exc.next_required_feature
            raise RuntimeError(str(exc)) from exc

        if trace_record.get("mailbox_verify_result_metadata_only"):
            message = "mailbox verify token decision backend is not implemented for no-commit apply probe"
            trace_record["mailbox_verify_apply_error"] = message
            trace_record["mailbox_verify_apply_error_kind"] = "mailbox_verify_token_decision_backend"
            trace_record["next_required_feature"] = "mailbox_verify_token_decision_backend"
            raise RuntimeError(f"{message}; next_required_feature=mailbox_verify_token_decision_backend")

        trace_record["mailbox_verify_apply_success"] = True
        commit_probe_enabled = bool(getattr(self.global_config, "stspec_mailbox_commit_probe", False))
        trace_record["stspec_mailbox_commit_probe_enabled"] = commit_probe_enabled
        if not commit_probe_enabled:
            message = "mailbox verify no-commit apply probe succeeded; state commit after mailbox verify is not implemented"
            trace_record["mailbox_verify_apply_error"] = message
            trace_record["next_required_feature"] = "state_commit_after_mailbox_verify"
            raise RuntimeError(f"{message}; next_required_feature=state_commit_after_mailbox_verify")

        try:
            commit_plan = build_mailbox_verify_commit_plan(
                verify_result,
                apply_plan,
                exec_seqs,
                step_plan,
                commit_allowed=True,
                commit_mode="guarded_probe",
                eos_token_id=getattr(self.global_config, "eos", None),
            )
            trace_record["mailbox_verify_commit_seq_ids"] = list(commit_plan.seq_ids)
            trace_record["mailbox_verify_commit_accepted_lengths_by_seq"] = dict(commit_plan.accepted_lengths_by_seq)
            trace_record["mailbox_verify_commit_target_correction_token_ids_by_seq"] = dict(
                commit_plan.target_correction_token_ids_by_seq
            )
            trace_record["mailbox_verify_commit_rejected_seq_ids"] = list(commit_plan.rejected_seq_ids)
            trace_record["mailbox_verify_commit_total_accepted_tokens"] = sum(
                len(tokens) for tokens in commit_plan.accepted_token_ids_by_seq.values()
            )
            trace_record["mailbox_verify_commit_total_rejected_tokens"] = sum(
                len(tokens) for tokens in commit_plan.rejected_token_ids_by_seq.values()
            )
            kv_commit_plan = build_mailbox_kv_commit_plan(
                verify_result,
                commit_plan,
                exec_seqs,
                step_plan,
                max_model_len=getattr(self.global_config, "max_model_len", None),
                commit_allowed=True,
                commit_mode="shadow_only",
            )
            trace_record["kv_commit_plan_built"] = True
            trace_record["kv_commit_seq_ids"] = list(kv_commit_plan.seq_ids)
            trace_record["kv_commit_accepted_lengths_by_seq"] = dict(kv_commit_plan.accepted_lengths_by_seq)
            trace_record["kv_commit_target_correction_token_ids_by_seq"] = dict(
                kv_commit_plan.target_correction_token_ids_by_seq
            )
            trace_record["kv_commit_append_start_positions_by_seq"] = dict(kv_commit_plan.append_start_positions_by_seq)
            trace_record["kv_commit_append_end_positions_by_seq"] = dict(kv_commit_plan.append_end_positions_by_seq)
            trace_record["kv_commit_sequence_length_before_by_seq"] = dict(kv_commit_plan.sequence_length_before_by_seq)
            trace_record["kv_commit_sequence_length_after_by_seq"] = dict(kv_commit_plan.sequence_length_after_by_seq)
            payload_consume_plan = build_mailbox_payload_consume_plan(commit_plan, self.stspec_mailbox)
            trace_record["mailbox_payload_consume_plan_built"] = True
            trace_record["mailbox_payload_consumed_payload_ids"] = list(payload_consume_plan.consumed_payload_ids)
            trace_record["mailbox_payload_invalidated_payload_ids"] = list(payload_consume_plan.invalidated_payload_ids)
            trace_record["mailbox_payload_lifecycle_before"] = payload_consume_plan.mailbox_state_before
            continue_after_commit = bool(getattr(self.global_config, "stspec_continue_after_mailbox_commit", False))
            trace_record["stspec_continue_after_mailbox_commit_enabled"] = continue_after_commit
            commit_result = run_mailbox_verify_commit_probe(
                commit_plan,
                exec_seqs,
                current_rank=output.current_rank,
                output_owner_rank=output.output_owner_rank,
                eos_token_id=getattr(self.global_config, "eos", None),
                commit_enabled=True,
                kv_commit_plan=kv_commit_plan,
                payload_consume_plan=payload_consume_plan,
                mailbox=self.stspec_mailbox,
                next_pipeline_plan_id=int(getattr(step_plan, "plan_id", 0) or 0) + 1,
                continue_after_commit=continue_after_commit,
                continuation_context={
                    "active_seq_ids": [int(seq.seq_id) for seq in exec_seqs if not bool(getattr(seq, "is_finished", False))],
                },
            )
            trace_record["mailbox_verify_commit_attempted"] = bool(commit_result.attempted)
            trace_record["mailbox_verify_commit_success"] = bool(commit_result.success)
            trace_record["mailbox_verify_commit_error"] = commit_result.error_message
            trace_record["mailbox_verify_commit_error_kind"] = commit_result.error_kind
            trace_record["sequence_state_commit_attempted"] = bool(commit_result.sequence_state_commit_attempted)
            trace_record["sequence_state_commit_success"] = bool(commit_result.sequence_state_commit_success)
            trace_record["sequence_state_before"] = commit_result.sequence_state_before
            trace_record["sequence_state_after"] = commit_result.sequence_state_after
            trace_record["kv_commit_plan_built"] = bool(commit_result.kv_commit_plan_built) or trace_record.get("kv_commit_plan_built", False)
            trace_record["kv_commit_attempted"] = bool(commit_result.kv_commit_attempted)
            trace_record["kv_commit_success"] = bool(commit_result.kv_commit_success)
            trace_record["kv_commit_shadow_only"] = bool(commit_result.kv_commit_shadow_only)
            trace_record["kv_commit_error"] = commit_result.kv_commit_error
            trace_record["kv_commit_error_kind"] = commit_result.kv_commit_error_kind
            trace_record["kv_commit_rollback_attempted"] = bool(commit_result.kv_commit_rollback_attempted)
            trace_record["kv_commit_rollback_success"] = bool(commit_result.kv_commit_rollback_success)
            trace_record["kv_commit_skipped_non_owner"] = bool(commit_result.kv_commit_skipped_non_owner)
            trace_record["mailbox_payload_consume_plan_built"] = bool(commit_result.mailbox_payload_consume_plan_built) or trace_record.get("mailbox_payload_consume_plan_built", False)
            trace_record["mailbox_payload_consume_attempted"] = bool(commit_result.mailbox_payload_consume_attempted)
            trace_record["mailbox_payload_consume_success"] = bool(commit_result.mailbox_payload_consume_success)
            trace_record["mailbox_payload_consume_error"] = commit_result.mailbox_payload_consume_error
            trace_record["mailbox_payload_consume_error_kind"] = commit_result.mailbox_payload_consume_error_kind
            trace_record["mailbox_payload_consumed_payload_ids"] = list(commit_result.mailbox_payload_consumed_payload_ids)
            trace_record["mailbox_payload_consumed_token_count"] = int(commit_result.mailbox_payload_consumed_token_count)
            trace_record["mailbox_payload_invalidate_attempted"] = bool(commit_result.mailbox_payload_invalidate_attempted)
            trace_record["mailbox_payload_invalidate_success"] = bool(commit_result.mailbox_payload_invalidate_success)
            trace_record["mailbox_payload_invalidate_error"] = commit_result.mailbox_payload_invalidate_error
            trace_record["mailbox_payload_invalidated_payload_ids"] = list(commit_result.mailbox_payload_invalidated_payload_ids)
            trace_record["mailbox_payload_invalidated_token_count"] = int(commit_result.mailbox_payload_invalidated_token_count)
            trace_record["mailbox_payload_lifecycle_before"] = commit_result.mailbox_payload_lifecycle_before or trace_record.get("mailbox_payload_lifecycle_before", {})
            trace_record["mailbox_payload_lifecycle_after"] = commit_result.mailbox_payload_lifecycle_after
            trace_record["mailbox_payload_duplicate_consume_detected"] = bool(commit_result.mailbox_payload_duplicate_consume_detected)
            trace_record["mailbox_payload_consume_rollback_attempted"] = bool(commit_result.mailbox_payload_consume_rollback_attempted)
            trace_record["mailbox_payload_consume_rollback_success"] = bool(commit_result.mailbox_payload_consume_rollback_success)
            trace_record["mailbox_payload_consume_skipped_non_owner"] = bool(commit_result.mailbox_payload_consume_skipped_non_owner)
            trace_record["mailbox_payload_invalidate_skipped_non_owner"] = bool(commit_result.mailbox_payload_invalidate_skipped_non_owner)
            trace_record["next_pipeline_step_attempted"] = bool(commit_result.next_pipeline_step_attempted)
            trace_record["next_pipeline_step_success"] = bool(commit_result.next_pipeline_step_success)
            trace_record["next_pipeline_step_error"] = commit_result.next_pipeline_step_error
            trace_record["next_pipeline_step_error_kind"] = commit_result.next_pipeline_step_error_kind
            trace_record["next_pipeline_plan_id"] = commit_result.next_pipeline_plan_id
            trace_record["next_pipeline_target_home_batch_id"] = commit_result.next_pipeline_target_home_batch_id
            trace_record["next_pipeline_draft_home_batch_id"] = commit_result.next_pipeline_draft_home_batch_id
            trace_record["next_pipeline_actual_target_seq_ids"] = list(commit_result.next_pipeline_actual_target_seq_ids)
            trace_record["next_pipeline_actual_draft_seq_ids"] = list(commit_result.next_pipeline_actual_draft_seq_ids)
            trace_record["previous_committed_plan_id"] = commit_result.previous_committed_plan_id
            trace_record["previous_consumed_payload_ids"] = list(commit_result.previous_consumed_payload_ids)
            trace_record["previous_invalidated_payload_ids"] = list(commit_result.previous_invalidated_payload_ids)
            trace_record["duplicate_payload_consume_after_continue"] = bool(commit_result.duplicate_payload_consume_after_continue)
            trace_record["pipeline_state_after_commit_valid"] = bool(commit_result.pipeline_state_after_commit_valid)
            trace_record["scheduler_state_after_commit_valid"] = bool(commit_result.scheduler_state_after_commit_valid)
            trace_record["breadth_only_step_count"] = int(commit_result.breadth_only_step_count)
            trace_record["breadth_only_completed"] = bool(commit_result.breadth_only_completed)
            trace_record["breadth_only_completion_reason"] = commit_result.breadth_only_completion_reason
            trace_record["second_step_state_check_attempted"] = commit_result.second_step_state_check_attempted
            trace_record["second_step_state_check_success"] = commit_result.second_step_state_check_success
            trace_record["second_step_state_error"] = commit_result.second_step_state_error
            trace_record["second_step_state_error_kind"] = commit_result.second_step_state_error_kind
            trace_record["current_pipeline_step"] = commit_result.current_pipeline_step
            trace_record["current_plan_id"] = commit_result.current_plan_id
            trace_record["next_plan_id"] = commit_result.next_plan_id
            trace_record["previous_target_home_batch_id"] = commit_result.previous_target_home_batch_id
            trace_record["previous_draft_home_batch_id"] = commit_result.previous_draft_home_batch_id
            trace_record["current_target_home_batch_id"] = commit_result.current_target_home_batch_id
            trace_record["current_draft_home_batch_id"] = commit_result.current_draft_home_batch_id
            trace_record["active_seq_ids_before_second_step"] = commit_result.active_seq_ids_before_second_step
            trace_record["active_seq_ids_after_second_step"] = commit_result.active_seq_ids_after_second_step
            trace_record["committed_seq_ids"] = commit_result.committed_seq_ids
            trace_record["consumed_payload_ids"] = commit_result.consumed_payload_ids
            trace_record["invalidated_payload_ids"] = commit_result.invalidated_payload_ids
            trace_record["available_mailbox_payload_ids"] = commit_result.available_mailbox_payload_ids
            trace_record["pending_mailbox_payload_ids"] = commit_result.pending_mailbox_payload_ids
            trace_record["repeated_verify_after_commit_detected"] = commit_result.repeated_verify_after_commit_detected
            trace_record["scheduler_state_after_second_step_valid"] = commit_result.scheduler_state_after_second_step_valid
            trace_record["sequence_state_after_second_step_valid"] = commit_result.sequence_state_after_second_step_valid
            trace_record["mailbox_state_after_second_step_valid"] = commit_result.mailbox_state_after_second_step_valid
            trace_record["request_completion_check_attempted"] = commit_result.request_completion_check_attempted
            trace_record["request_completion_check_success"] = commit_result.request_completion_check_success
            trace_record["request_completion_reason"] = commit_result.request_completion_reason
            trace_record["active_seq_ids_at_completion_check"] = list(commit_result.active_seq_ids_at_completion_check)
            trace_record["finished_seq_ids_at_completion_check"] = list(commit_result.finished_seq_ids_at_completion_check)
            trace_record["unfinished_seq_ids_at_completion_check"] = list(commit_result.unfinished_seq_ids_at_completion_check)
            trace_record["max_tokens_reached_seq_ids"] = list(commit_result.max_tokens_reached_seq_ids)
            trace_record["eos_reached_seq_ids"] = list(commit_result.eos_reached_seq_ids)
            trace_record["mailbox_pending_payload_ids_at_completion"] = list(commit_result.mailbox_pending_payload_ids_at_completion)
            trace_record["mailbox_consumed_payload_ids_at_completion"] = list(commit_result.mailbox_consumed_payload_ids_at_completion)
            trace_record["scheduler_active_seq_ids_at_completion"] = list(commit_result.scheduler_active_seq_ids_at_completion)
            trace_record["sequence_state_completion_valid"] = bool(commit_result.sequence_state_completion_valid)
            trace_record["scheduler_state_completion_valid"] = bool(commit_result.scheduler_state_completion_valid)
            trace_record["mailbox_state_completion_valid"] = bool(commit_result.mailbox_state_completion_valid)
            trace_record["request_completion_error"] = commit_result.request_completion_error
            trace_record["request_completion_error_kind"] = commit_result.request_completion_error_kind
            trace_record["result_finalization_attempted"] = bool(commit_result.result_finalization_attempted)
            trace_record["result_finalization_success"] = bool(commit_result.result_finalization_success)
            trace_record["result_finalization_error"] = commit_result.result_finalization_error
            trace_record["result_finalization_error_kind"] = getattr(commit_result, "result_finalization_error_kind", None)
            trace_record["second_step_rollback_attempted"] = commit_result.second_step_rollback_attempted
            trace_record["second_step_rollback_success"] = commit_result.second_step_rollback_success
            trace_record["next_pipeline_step_skipped_non_owner"] = bool(commit_result.next_pipeline_step_skipped_non_owner)
            trace_record["mailbox_verify_commit_rollback_attempted"] = bool(commit_result.rollback_attempted)
            trace_record["mailbox_verify_commit_rollback_success"] = bool(commit_result.rollback_success)
            trace_record["mailbox_verify_commit_skipped_non_owner"] = bool(commit_result.skipped_non_owner)
            if commit_result.kv_commit_skipped_non_owner:
                trace_record["kv_commit_skipped_non_owner"] = True
            if commit_result.skipped_non_owner:
                return False
            if not commit_result.success:
                trace_record["next_required_feature"] = commit_result.next_required_feature or "mailbox_commit_rollback_validation"
                raise RuntimeError(
                    f"mailbox verify guarded commit probe failed; plan_id={step_plan.plan_id}, "
                    f"target_seq_ids={input_seq_ids}, error={commit_result.error_message}; "
                    f"next_required_feature={trace_record['next_required_feature']}"
                )
            self._record_v4x_pending_corrections(commit_result, trace_record)
            if self._try_finalize_v4s_result(exec_seqs, step_plan, trace_record, commit_result, output):
                return True
            current_next_required = (
                trace_record.get("next_required_feature")
                or commit_result.next_required_feature
                or "next_pipeline_step_after_mailbox_commit"
            )
            if current_next_required == "active_request_continuation_after_breadth_only_step":
                if self._try_continue_v4t_active_requests(exec_seqs, step_plan, trace_record, commit_result, output):
                    return True
                if trace_record.get("next_required_feature") == "active_request_continuation_after_breadth_only_step":
                    trace_record["next_required_feature"] = "active_request_continuation_limit_reached"
                current_next_required = trace_record.get("next_required_feature") or "active_request_continuation_limit_reached"
            trace_record["next_required_feature"] = commit_result.next_required_feature or "next_pipeline_step_after_mailbox_commit"
            if current_next_required != "active_request_continuation_after_breadth_only_step":
                trace_record["next_required_feature"] = current_next_required
            if trace_record["next_required_feature"] == "result_finalization_after_breadth_only_completion":
                trace_record["next_required_feature"] = "evaluator_return_after_breadth_only_completion"
            message = "mailbox verify guarded commit probe reached next explicit diagnostic"
            trace_record["mailbox_verify_commit_error"] = message
            raise RuntimeError(f"{message}; next_required_feature={trace_record['next_required_feature']}")
        except MailboxVerifyApplyError as exc:
            trace_record["mailbox_verify_commit_attempted"] = True
            trace_record["mailbox_verify_commit_success"] = False
            trace_record["mailbox_verify_commit_error"] = str(exc)
            trace_record["mailbox_verify_commit_error_kind"] = exc.error_kind
            if str(exc.error_kind).startswith("mailbox_kv_commit"):
                trace_record["kv_commit_error"] = str(exc)
                trace_record["kv_commit_error_kind"] = exc.error_kind
            trace_record["next_required_feature"] = exc.next_required_feature
            raise RuntimeError(str(exc)) from exc
        return False

    def _prepare_stspec_mailbox_route(
        self,
        step_plan: StepPlan,
        runner_role: str,
        trace_record: dict,
        exec_seqs: list[Sequence] | None = None,
    ) -> bool:
        """V4E mailbox transport/consume preflight for real variable-offset probes.

        Draft runners now continue to produce a validated mailbox transport
        envelope after the draft payload exists. Target runners first classify
        pipeline warmup separately from transport/payload misses; if payloads are
        present, the probe stops at the next explicit blocker: verification input
        construction from mailbox payloads is not wired yet.
        """
        if not self._stspec_mailbox_enabled(step_plan):
            return False
        trace_record["stspec_mailbox_enabled"] = True
        trace_record["stspec_mailbox_transport_enabled"] = True
        trace_record["mailbox_transport_mode"] = self._mailbox_transport_mode()
        self._trace_mailbox_availability(trace_record)

        if "draft" in runner_role:
            # The draft payload is not available until after gamma draft steps;
            # _record_draft_mailbox_payloads() will encode the transport envelope.
            trace_record["mailbox_transport_send_attempted"] = False
            trace_record["mailbox_transport_send_seq_ids"] = list(step_plan.actual_draft_exec_seq_ids)
            trace_record["mailbox_transport_send_home_batch_id"] = step_plan.draft_home_batch_id
            return False

        trace_record["mailbox_transport_recv_attempted"] = True
        trace_record["mailbox_transport_recv_success"] = False
        trace_record["mailbox_transport_recv_home_batch_id"] = step_plan.target_home_batch_id
        trace_record["mailbox_transport_recv_seq_ids"] = []
        trace_record["mailbox_transport_payload_available"] = False

        target_seq_ids = list(step_plan.actual_target_exec_seq_ids)
        target_tp_role = classify_target_tp_rank_role_for_mailbox_forward(self)
        trace_record["target_tp_current_rank"] = target_tp_role.current_rank
        trace_record["target_tp_output_owner_rank"] = target_tp_role.owner_rank
        trace_record["target_tp_is_output_owner"] = target_tp_role.is_output_owner
        trace_record["target_tp_is_payload_owner"] = target_tp_role.is_payload_owner
        trace_record["target_tp_should_run_forward"] = target_tp_role.should_run_target_forward
        trace_record["target_tp_should_interpret_output"] = target_tp_role.should_interpret_output
        trace_record["target_tp_should_apply_verify_result"] = target_tp_role.should_apply_verify_result
        trace_record["target_tp_skipped_non_owner"] = False
        trace_record["mailbox_payload_owner_rank"] = target_tp_role.owner_rank
        trace_record["mailbox_payload_current_rank"] = target_tp_role.current_rank
        if (
            step_plan.stspec_pipeline_phase == STSpecPipelinePhase.STEADY_STATE.value
            and step_plan.target_home_batch_id not in self.stspec_mailbox.available_home_batch_ids()
            and exec_seqs is not None
        ):
            self._receive_mailbox_payload_tensor_probe(exec_seqs, step_plan, runner_role, trace_record)
        result = self.stspec_mailbox.get_payloads(
            step_plan.target_home_batch_id,
            target_seq_ids,
            plan_id=step_plan.plan_id,
            consumer_role=runner_role,
        )
        trace_record["mailbox_get_attempted"] = True
        trace_record["mailbox_get_success"] = result.success
        trace_record["mailbox_get_hit_count"] = result.hit_count
        trace_record["mailbox_get_miss_count"] = result.miss_count
        trace_record["mailbox_get_home_batch_id"] = step_plan.target_home_batch_id
        trace_record["mailbox_get_seq_ids"] = target_seq_ids
        trace_record["mailbox_missing_seq_ids"] = list(result.missing_seq_ids)
        self._trace_mailbox_availability(trace_record)
        available_by_batch = trace_record.get("mailbox_available_seq_ids_by_batch") or {}
        available_seq_ids = [int(seq_id) for seq_id in available_by_batch.get(str(step_plan.target_home_batch_id), [])]
        payload_available_for_seq_ids = all(int(seq_id) in set(available_seq_ids) for seq_id in target_seq_ids)
        trace_record["mailbox_payload_envelope_available"] = step_plan.target_home_batch_id in self.stspec_mailbox.available_home_batch_ids()
        trace_record["mailbox_payload_available_for_seq_ids"] = payload_available_for_seq_ids
        trace_record["mailbox_payload_missing_reason"] = None if result.success else (
            "missing_seq_ids" if result.missing_seq_ids else "payload_metadata_available_but_local_payload_unavailable"
        )
        if result.success:
            payload_seq_ids = [int(payload.seq_id) for payload in result.payloads]
            payload_home_batch_ids = {payload.home_batch_id for payload in result.payloads}
            if payload_seq_ids != target_seq_ids or payload_home_batch_ids != {step_plan.target_home_batch_id}:
                trace_record["illegal_legacy_fallback"] = True
                message = (
                    "target consume-from-mailbox payload validation failed; payload seq/home batch ids "
                    "do not match target StepPlan"
                )
                trace_record["target_consume_from_mailbox_attempted"] = True
                trace_record["target_consume_from_mailbox_success"] = False
                trace_record["target_consume_from_mailbox_payload_seq_ids"] = payload_seq_ids
                trace_record["target_consume_from_mailbox_payload_home_batch_id"] = list(payload_home_batch_ids)
                trace_record["target_consume_from_mailbox_error"] = message
                trace_record["next_required_feature"] = "target_consume_payload_validation"
                raise RuntimeError(
                    f"{message}; plan_id={step_plan.plan_id}, runner_role={runner_role}, "
                    f"target_seq_ids={target_seq_ids}, payload_seq_ids={payload_seq_ids}, "
                    f"target_home_batch_id={step_plan.target_home_batch_id}, payload_home_batch_ids={payload_home_batch_ids}, "
                    "next_required_feature=target_consume_payload_validation"
                )
            trace_record["mailbox_transport_recv_success"] = True
            trace_record["mailbox_transport_recv_seq_ids"] = target_seq_ids
            token_ids_available = all(
                len(payload.draft_token_ids) == int(payload.per_seq_length) for payload in result.payloads
            )
            trace_record["mailbox_transport_payload_available"] = token_ids_available
            trace_record["mailbox_payload_envelope_available"] = True
            trace_record["mailbox_payload_token_ids_available"] = token_ids_available
            trace_record["mailbox_payload_tensor_available"] = token_ids_available
            trace_record["mailbox_payload_available_for_seq_ids"] = True
            trace_record["mailbox_payload_local_to_rank"] = token_ids_available
            trace_record["mailbox_payload_missing_reason"] = None if token_ids_available else "mailbox_payload_token_ids_unavailable"
            trace_record["mailbox_routing_ok"] = True
            trace_record["mailbox_error"] = None
            trace_record["mailbox_error_kind"] = None
            trace_record["target_consume_from_mailbox_attempted"] = True
            trace_record["target_consume_from_mailbox_success"] = True
            trace_record["target_consume_from_mailbox_payload_seq_ids"] = payload_seq_ids
            trace_record["target_consume_from_mailbox_payload_lengths"] = [int(payload.per_seq_length) for payload in result.payloads]
            trace_record["target_consume_from_mailbox_payload_total_tokens"] = sum(
                int(payload.per_seq_length) for payload in result.payloads
            )
            trace_record["target_consume_from_mailbox_payload_home_batch_id"] = step_plan.target_home_batch_id
            trace_record["target_consume_from_mailbox_error"] = None
            if not token_ids_available:
                if target_tp_role.should_skip_non_owner:
                    trace_record["target_tp_skipped_non_owner"] = True
                    trace_record["output_interpretation_skipped_non_owner"] = True
                    trace_record["mailbox_verify_apply_skipped_non_owner"] = True
                    trace_record["mailbox_verify_commit_skipped_non_owner"] = True
                    trace_record["mailbox_payload_missing_reason"] = "mailbox_payload_tensor_unavailable_on_non_owner"
                    trace_record["target_forward_output_none_expected"] = True
                    return False
                message = "ST-Spec mailbox payload tensor unavailable on output owner rank"
                trace_record["target_consume_from_mailbox_success"] = False
                trace_record["target_consume_from_mailbox_error"] = message
                self._record_mailbox_transport_error(
                    trace_record,
                    kind="mailbox_payload_tensor_backend_unavailable",
                    message=message,
                    next_required_feature="mailbox_payload_tensor_backend",
                )
                trace_record["next_required_feature"] = "mailbox_payload_tensor_backend"
                raise RuntimeError(
                    f"{message}; plan_id={step_plan.plan_id}, runner_role={runner_role}, "
                    f"owner_rank={target_tp_role.owner_rank}, current_rank={target_tp_role.current_rank}, "
                    f"target_home_batch_id={step_plan.target_home_batch_id}, target_seq_ids={target_seq_ids}, "
                    f"available_mailbox_seq_ids_by_batch={trace_record['mailbox_available_seq_ids_by_batch']}, "
                    "next_required_feature=mailbox_payload_tensor_backend"
                )
            trace_record["verification_input_from_mailbox_attempted"] = True
            verification_input = build_verification_input_from_mailbox_payload(
                result.payloads, exec_seqs or [], step_plan, self.gamma
            )
            trace_record["verification_input_from_mailbox_success"] = True
            trace_record["verification_input_from_mailbox_seq_ids"] = list(verification_input.seq_ids)
            trace_record["verification_input_from_mailbox_total_tokens"] = int(verification_input.total_tokens)
            trace_record["verification_input_from_mailbox_error"] = None
            return self._run_target_forward_from_mailbox_input(
                verification_input,
                exec_seqs or [],
                step_plan,
                trace_record,
            )

        error_kind, next_feature, warmup_miss = classify_mailbox_miss(
            target_home_batch_id=step_plan.target_home_batch_id,
            available_home_batch_ids=self.stspec_mailbox.available_home_batch_ids(),
            allow_warmup_miss=self._mailbox_allow_warmup_miss(),
        )
        if step_plan.stspec_pipeline_phase == STSpecPipelinePhase.STEADY_STATE.value and warmup_miss:
            error_kind, next_feature, warmup_miss = "mailbox_missing_payload", "mailbox_payload_tensor_transport", False
        if not result.missing_seq_ids and payload_available_for_seq_ids and not result.success:
            trace_record["mailbox_payload_envelope_available"] = True
            trace_record["mailbox_payload_available_for_seq_ids"] = True
            if target_tp_role.should_skip_non_owner:
                trace_record["target_tp_skipped_non_owner"] = True
                trace_record["output_interpretation_skipped_non_owner"] = True
                trace_record["mailbox_verify_apply_skipped_non_owner"] = True
                trace_record["mailbox_payload_missing_reason"] = "mailbox_payload_tensor_unavailable_on_non_owner"
                trace_record["target_forward_output_none_expected"] = True
                return False
            error_kind, next_feature, warmup_miss = "mailbox_payload_tensor_backend_unavailable", "mailbox_payload_tensor_backend", False
            trace_record["mailbox_payload_missing_reason"] = "mailbox_payload_tensor_backend_unavailable"
        if warmup_miss and should_skip_target_for_warmup(
            phase=step_plan.stspec_pipeline_phase,
            runner_role=runner_role,
            allow_warmup_miss=self._mailbox_allow_warmup_miss(),
        ):
            trace_record["mailbox_warmup_skip"] = True
            trace_record["target_verify_skipped_for_warmup"] = True
            trace_record["warmup_target_verify_skipped"] = True
            trace_record["pipeline_phase_advanced"] = True
            message = "ST-Spec warmup target verify skipped; draft-only pipeline fill recorded"
            self._record_mailbox_error(
                trace_record,
                kind=error_kind,
                message=message,
                next_required_feature="verification_input_from_mailbox",
                warmup_miss=True,
            )
            return False
        elif warmup_miss:
            message = "ST-Spec mailbox warmup miss for target batch; pipeline warmup schedule is required"
        else:
            message = "ST-Spec mailbox payload missing for target batch after transport receive attempt"
        self._record_mailbox_error(
            trace_record,
            kind=error_kind,
            message=message,
            next_required_feature=next_feature,
            warmup_miss=warmup_miss,
        )
        if not warmup_miss:
            self._record_mailbox_transport_error(
                trace_record,
                kind="mailbox_payload_tensor_transport_unavailable",
                message=message,
                next_required_feature=next_feature,
            )
        raise RuntimeError(
            f"{message}; mailbox_error_kind={error_kind}, plan_id={step_plan.plan_id}, "
            f"runner_role={runner_role}, target_home_batch_id={step_plan.target_home_batch_id}, "
            f"target_seq_ids={target_seq_ids}, mailbox_missing_seq_ids={result.missing_seq_ids}, "
            f"available_mailbox_home_batch_ids={trace_record['mailbox_available_home_batch_ids']}, "
            f"available_mailbox_seq_ids_by_batch={trace_record['mailbox_available_seq_ids_by_batch']}, "
            f"allow_warmup_miss={self._mailbox_allow_warmup_miss()}, "
            f"next_required_feature={next_feature}"
        )

    def _guard_outstanding_draft_mailbox_payloads(
        self,
        *,
        trace_record: dict,
        step_plan: StepPlan,
        payloads: list[MailboxPayload],
    ) -> None:
        contexts = self.stspec_mailbox.available_payload_contexts_for(payloads)
        conflicts = [context for context in contexts if not bool(context.get("same_payload"))]
        if not conflicts:
            if contexts:
                trace_record["mailbox_payload_duplicate_put_detected"] = True
                trace_record["mailbox_payload_duplicate_put_idempotent_skip"] = True
                trace_record["mailbox_payload_outstanding_available_detected"] = True
                trace_record["mailbox_payload_outstanding_context"] = dict(contexts[0])
            return

        context = dict(conflicts[0])
        message = "outstanding mailbox payload remains before next draft production"
        trace_record["mailbox_payload_put_success"] = False
        trace_record["mailbox_put_success"] = False
        trace_record["mailbox_payload_duplicate_put_detected"] = True
        trace_record["mailbox_payload_duplicate_put_conflict"] = True
        trace_record["mailbox_payload_duplicate_put_context"] = dict(context)
        trace_record["mailbox_payload_outstanding_available_detected"] = True
        trace_record["mailbox_payload_outstanding_context"] = dict(context)
        trace_record["mailbox_payload_existing_plan_id"] = context.get("existing_plan_id")
        trace_record["mailbox_payload_incoming_plan_id"] = context.get("incoming_plan_id")
        trace_record["mailbox_payload_existing_payload_id"] = context.get("existing_payload_id")
        trace_record["mailbox_payload_incoming_payload_id"] = context.get("incoming_payload_id")
        trace_record["mailbox_payload_lifecycle_state"] = context.get("lifecycle_state")
        trace_record["active_continuation_skipped_due_to_outstanding_payload"] = True
        trace_record["active_continuation_plan_id"] = step_plan.plan_id
        trace_record["active_continuation_attempted"] = bool(
            getattr(self.global_config, "stspec_continue_after_mailbox_commit", False)
        )
        trace_record["active_continuation_success"] = False
        trace_record["active_continuation_error"] = message
        trace_record["active_continuation_error_kind"] = "mailbox_payload_after_active_continuation"
        trace_record["active_request_continuation_error"] = message
        trace_record["active_request_continuation_error_kind"] = "mailbox_payload_after_active_continuation"
        self._record_mailbox_error(
            trace_record,
            kind="mailbox_payload_after_active_continuation",
            message=message,
            next_required_feature="mailbox_payload_after_active_continuation",
        )
        raise RuntimeError(
            f"{message}; home_batch_id={context.get('home_batch_id')}, seq_id={context.get('seq_id')}, "
            f"existing_plan_id={context.get('existing_plan_id')}, incoming_plan_id={context.get('incoming_plan_id')}; "
            "next_required_feature=mailbox_payload_after_active_continuation"
        )

    def _record_draft_mailbox_guard_skip_or_conflict(
        self,
        *,
        trace_record: dict,
        step_plan: StepPlan,
        payloads: list[MailboxPayload],
    ) -> bool:
        guard = getattr(self, "stspec_draft_mailbox_record_guard", None)
        if guard is None:
            self._reset_draft_mailbox_record_guard()
            guard = self.stspec_draft_mailbox_record_guard

        payload_hashes = [self._draft_mailbox_payload_hash(payload) for payload in payloads]
        put_keys = [
            (payload.plan_id, payload.home_batch_id, payload.seq_id, payload.producer_role)
            for payload in payloads
        ]
        trace_record["mailbox_payload_put_key"] = [list(key) for key in put_keys]
        trace_record["mailbox_payload_put_payload_hash"] = payload_hashes
        trace_record["mailbox_payload_record_guard_size"] = len(guard)
        trace_record["mailbox_payload_recorded_plan_ids"] = self._draft_mailbox_recorded_plan_ids()

        for payload, payload_hash, key in zip(payloads, payload_hashes, put_keys):
            existing = guard.get(key)
            if existing is None:
                continue
            trace_record["mailbox_payload_record_guard_hit"] = True
            context = {
                "plan_id": payload.plan_id,
                "home_batch_id": payload.home_batch_id,
                "seq_id": payload.seq_id,
                "producer_role": payload.producer_role,
                "payload_id": payload.payload_id,
                "payload_hash": payload_hash,
                "existing_payload_id": existing.get("payload_id"),
                "existing_payload_hash": existing.get("payload_hash"),
            }
            if existing.get("payload_hash") == payload_hash:
                trace_record["mailbox_payload_put_success"] = False
                trace_record["mailbox_payload_put_skipped"] = True
                trace_record["mailbox_payload_duplicate_put_detected"] = True
                trace_record["mailbox_payload_duplicate_put_idempotent_skip"] = True
                trace_record["mailbox_payload_duplicate_put_conflict"] = False
                trace_record["mailbox_payload_duplicate_put_context"] = context
                trace_record["mailbox_put_count"] = 0
                return True
            trace_record["mailbox_payload_put_success"] = False
            trace_record["mailbox_payload_put_skipped"] = False
            trace_record["mailbox_payload_duplicate_put_detected"] = True
            trace_record["mailbox_payload_duplicate_put_idempotent_skip"] = False
            trace_record["mailbox_payload_duplicate_put_conflict"] = True
            trace_record["mailbox_payload_duplicate_put_context"] = context
            raise STSpecMailboxError(
                "Conflicting repeated draft mailbox payload recording",
                kind="duplicate_put_conflict",
                context=context,
            )
        return False

    def _remember_draft_mailbox_recorded_payloads(self, payloads: list[MailboxPayload]) -> None:
        guard = getattr(self, "stspec_draft_mailbox_record_guard", None)
        if guard is None:
            self._reset_draft_mailbox_record_guard()
            guard = self.stspec_draft_mailbox_record_guard
        for payload in payloads:
            key = (payload.plan_id, payload.home_batch_id, payload.seq_id, payload.producer_role)
            guard[key] = {
                "payload_id": payload.payload_id,
                "payload_hash": self._draft_mailbox_payload_hash(payload),
            }
        # Keep this probe-only cache bounded while preserving recent plan retry detection.
        if len(guard) > 1024:
            recent_plan_ids = sorted({key[0] for key in guard if key[0] is not None})[-32:]
            keep_plans = set(recent_plan_ids)
            self.stspec_draft_mailbox_record_guard = {
                key: value for key, value in guard.items() if key[0] in keep_plans
            }

    def _record_draft_mailbox_payloads(
        self,
        trace_record: dict | None,
        step_plan: StepPlan | None,
        draft_message: PearlDraftMessage,
    ) -> None:
        if trace_record is None or not self._stspec_mailbox_enabled(step_plan):
            return
        payloads = payloads_from_draft_message(
            draft_message,
            home_batch_id=step_plan.draft_home_batch_id,
            target_home_batch_id=step_plan.target_home_batch_id,
            draft_home_batch_id=step_plan.draft_home_batch_id,
            producer_home_batch_id=step_plan.draft_home_batch_id,
            producer_role="draft",
            logical_step=step_plan.plan_id,
            metadata={"mailbox_locality": "diagnostic_local"},
        )
        payloads = [
            replace(
                payload,
                metadata={
                    **dict(payload.metadata or {}),
                    "source_plan_id": step_plan.plan_id,
                    "payload_id": (
                        f"{step_plan.plan_id}:{payload.home_batch_id}:"
                        f"{payload.seq_id}:{payload.offset}:{payload.per_seq_length}"
                    ),
                },
            )
            for payload in payloads
        ]
        trace_record["stspec_mailbox_transport_enabled"] = True
        trace_record["mailbox_transport_mode"] = self._mailbox_transport_mode()
        trace_record["mailbox_put_attempted"] = True
        trace_record["mailbox_payload_put_attempted"] = True
        trace_record["mailbox_put_home_batch_id"] = step_plan.draft_home_batch_id
        trace_record["mailbox_put_seq_ids"] = [payload.seq_id for payload in payloads]
        trace_record["mailbox_payload_put_plan_id"] = step_plan.plan_id
        trace_record["mailbox_payload_put_seq_ids"] = [payload.seq_id for payload in payloads]
        trace_record["mailbox_payload_put_home_batch_ids"] = [payload.home_batch_id for payload in payloads]
        if self._record_draft_mailbox_guard_skip_or_conflict(
            trace_record=trace_record,
            step_plan=step_plan,
            payloads=payloads,
        ):
            return
        self._guard_outstanding_draft_mailbox_payloads(
            trace_record=trace_record,
            step_plan=step_plan,
            payloads=payloads,
        )
        before_stats = self.stspec_mailbox.stats() if hasattr(self.stspec_mailbox, "stats") else {}
        try:
            self.stspec_mailbox.put_payloads(
                step_plan.draft_home_batch_id,
                payloads,
                plan_id=step_plan.plan_id,
                producer_role="draft",
            )
            envelope = encode_mailbox_transport_envelope(
                payloads=payloads,
                transport_mode=self._mailbox_transport_mode(),
                plan_id=step_plan.plan_id,
                producer_rank=self.rank,
                producer_role="draft",
                producer_home_batch_id=step_plan.draft_home_batch_id,
                target_home_batch_id=step_plan.target_home_batch_id,
                draft_home_batch_id=step_plan.draft_home_batch_id,
                produced_for_home_batch_id=step_plan.draft_home_batch_id,
                logical_step=step_plan.plan_id,
                source_plan_signature_hash=trace_record.get("plan_signature_hash"),
                payload_available=True,
                payload_metadata={"gamma": self.gamma, "mailbox_transport_scope": "diagnostic_envelope"},
            )
        except STSpecMailboxError as exc:
            trace_record["mailbox_put_success"] = False
            trace_record["mailbox_payload_put_success"] = False
            trace_record["mailbox_put_count"] = 0
            trace_record["mailbox_payload_duplicate_put_detected"] = str(exc.kind).startswith("duplicate_put")
            trace_record["mailbox_payload_duplicate_put_idempotent_skip"] = False
            trace_record["mailbox_payload_duplicate_put_conflict"] = exc.kind == "duplicate_put_conflict"
            trace_record["mailbox_payload_duplicate_put_context"] = dict(exc.context)
            self._record_mailbox_error(
                trace_record,
                kind=exc.kind,
                message=str(exc),
                next_required_feature="mailbox_payload_tensor_transport",
            )
            raise
        self._remember_draft_mailbox_recorded_payloads(payloads)
        trace_record["mailbox_payload_record_guard_size"] = len(self.stspec_draft_mailbox_record_guard)
        trace_record["mailbox_payload_recorded_plan_ids"] = self._draft_mailbox_recorded_plan_ids()
        after_stats = self.stspec_mailbox.stats() if hasattr(self.stspec_mailbox, "stats") else {}
        trace_record["mailbox_put_success"] = True
        trace_record["mailbox_payload_put_success"] = True
        put_delta = int(after_stats.get("put_count", 0) or 0) - int(before_stats.get("put_count", 0) or 0)
        trace_record["mailbox_put_count"] = put_delta
        idempotent_skips = int(after_stats.get("duplicate_put_idempotent_skip_count", 0) or 0) - int(
            before_stats.get("duplicate_put_idempotent_skip_count", 0) or 0
        )
        duplicate_puts = int(after_stats.get("duplicate_put_count", 0) or 0) - int(
            before_stats.get("duplicate_put_count", 0) or 0
        )
        trace_record["mailbox_payload_duplicate_put_detected"] = duplicate_puts > 0
        trace_record["mailbox_payload_duplicate_put_idempotent_skip"] = idempotent_skips > 0
        trace_record["mailbox_payload_duplicate_put_conflict"] = False
        trace_record["mailbox_payload_duplicate_put_context"] = {}
        if step_plan.stspec_pipeline_phase == STSpecPipelinePhase.WARMUP_DRAFT_ONLY.value:
            trace_record["warmup_draft_payload_produced"] = True
            trace_record["pipeline_phase_advanced"] = True
        trace_record["mailbox_cross_process_delivery"] = "transport_envelope_visible"
        trace_record["mailbox_transport_send_attempted"] = True
        trace_record["mailbox_transport_send_success"] = True
        trace_record["mailbox_transport_send_seq_ids"] = list(envelope.seq_ids)
        trace_record["mailbox_transport_send_home_batch_id"] = envelope.produced_for_home_batch_id
        trace_record["mailbox_transport_payload_available"] = bool(envelope.payload_available)
        trace_record["mailbox_transport_error"] = None
        trace_record["mailbox_transport_error_kind"] = None
        trace_record["mailbox_transport_envelope_digest"] = envelope.digest()
        self.stspec_mailbox.mark_payloads_stale(
            [payload.payload_id for payload in payloads],
            plan_id=step_plan.plan_id,
            reason="draft_transport_envelope_recorded",
        )
        trace_record["next_required_feature"] = "target_consume_from_mailbox"
        self._trace_mailbox_availability(trace_record)

    def _validate_stspec_probe_alignment(
        self,
        step_plan: StepPlan,
        runner_role: str,
        trace_record: dict,
    ) -> None:
        error = stspec_protocol_alignment_error(step_plan, runner_role, self.gamma, self._pearl_protocol_layout())
        if error is None:
            trace_record["protocol_alignment_ok"] = True
            trace_record["protocol_alignment_error"] = None
            if step_plan.real_probe_attempted:
                trace_record["real_probe_applied"] = bool(trace_record.get("filtered_out_seq_ids"))
                trace_record["real_probe_blocked"] = False
            return

        trace_record["protocol_alignment_ok"] = False
        trace_record["protocol_alignment_error"] = error
        trace_record["real_probe_applied"] = False
        trace_record["real_probe_blocked"] = True
        trace_record["real_probe_block_reason"] = error
        if self._pearl_protocol_layout() == "variable_offsets":
            trace_record["cross_batch_routing_ok"] = False
            trace_record["cross_batch_routing_error"] = error
            trace_record["next_required_feature"] = "mailbox_payload_tensor_transport"
        # V4A is explicitly a feasibility probe. The current PEARL protocol packs
        # draft/verify tensors by common sequence index, so a mismatch must stop
        # before distributed communication can hang or corrupt request state.
        raise RuntimeError(error)

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
        if invalidated_lens:
            invalidated_lens = {seq_id: int(invalidated_len) for seq_id, invalidated_len in invalidated_lens.items()}
            record["per_seq_invalidated_predraft_len"].update(invalidated_lens)

    def _mark_trace_end(self, record: dict, accepted_lens: dict[int, int] | None = None, invalidated_lens: dict[int, int] | None = None):
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
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data

    def prefill(self):
        runner_role = f"{self._runner_role()}_prefill"
        seqs, is_prefill, step_plan = self._schedule_with_plan(runner_role)
        trace_record = self._trace_schedule(seqs, is_prefill, runner_role, step_plan)
        exec_seqs = self._select_exec_seqs_for_plan(seqs, step_plan, runner_role, trace_record)
        self._validate_stspec_probe_alignment(step_plan, runner_role, trace_record)
        assert is_prefill, "wrong match. current stage is decode."
        input_ids, positions = self.prepare_prefill(exec_seqs)
        temperatures = self.prepare_sample(exec_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, True)
        sample_tokens = self.sampler(logits, temperatures) if self.tp_params.local_rank == 0 else torch.zeros(len(exec_seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
        torch.cuda.synchronize()
        token_ids = sample_tokens.tolist()
        reset_context(self.tp_params)
        self.scheduler.postprocess(exec_seqs, token_ids)
        accepted_lens = {seq.seq_id: 1 for seq in exec_seqs}
        for seq in exec_seqs:
            seq.record_accepted(1)
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens)

    def step(self):
        runner_role = self._runner_role()
        seqs, is_prefill, step_plan = self._schedule_with_plan(runner_role)
        trace_record = self._trace_schedule(seqs, is_prefill, runner_role, step_plan)
        exec_seqs = self._select_exec_seqs_for_plan(seqs, step_plan, runner_role, trace_record)
        self._validate_stspec_probe_alignment(step_plan, runner_role, trace_record)
        input_ids, positions = self.prepare_prefill(exec_seqs) if is_prefill else self.prepare_decode(exec_seqs)
        temperatures = self.prepare_sample(exec_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_tokens = self.sampler(logits, temperatures) if self.tp_params.local_rank == 0 else torch.zeros(len(exec_seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
        torch.cuda.synchronize()
        token_ids = sample_tokens.tolist()
        reset_context(self.tp_params)
        self.scheduler.postprocess(exec_seqs, token_ids)
        accepted_lens = {seq.seq_id: 1 for seq in exec_seqs}
        for seq in exec_seqs:
            seq.record_accepted(1)
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in exec_seqs]
        num_tokens = sum(len(seq) for seq in exec_seqs) if is_prefill else -len(exec_seqs)
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
        reset_v4t_active_continuation_runner_state(self)
        dist.barrier()

    def prepare_decode_ready(self):
        """Materialize prompt KV before a decoder-only benchmark measurement.

        This in-memory helper intentionally does not persist KV caches. It runs
        the existing prompt prefill once, marks requests as decode-ready, and
        leaves scheduler/KV state resident for a later decode_ready_* command.
        The evaluator should start timing only after this method returns.
        """
        self.active_decode_ready_mode = True
        reset_v4t_active_continuation_runner_state(self)
        self._reset_draft_mailbox_record_guard()
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
        reset_v4t_active_continuation_runner_state(self)
        self._reset_draft_mailbox_record_guard()
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

    def decode_ready_serialized_pearl_generate(self):
        """Decode-only serialized-PEARL approximation after prepare_decode_ready()."""
        self._set_execution_mode("serialized_pearl")
        self.active_decode_ready_mode = True
        self._reset_draft_mailbox_record_guard()
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
        reset_v4t_active_continuation_runner_state(self)
        self._reset_draft_mailbox_record_guard()
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
        self._reset_draft_mailbox_record_guard()
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


class DraftModelRunner(ModelRunnerBase):
    def __init__(self, config: PEARLConfig, rank: int, event: Event, control_event: Event):
        super().__init__(config, rank, event, control_event)

    def prepare_pearl_decode(self, seqs: list[Sequence]):
        return super().prepare_decode(seqs)
    
    def pearl_step(self):
        trace_record = None
        for _ in range(self.gamma):
            seqs, is_prefill, step_plan = self._schedule_with_plan("draft")
            trace_record = self._trace_schedule(seqs, is_prefill, "draft", step_plan)
            exec_seqs = self._select_exec_seqs_for_plan(seqs, step_plan, "draft", trace_record)
            self._validate_stspec_probe_alignment(step_plan, "draft", trace_record)
            self._prepare_stspec_mailbox_route(step_plan, "draft", trace_record, exec_seqs)
            assert not is_prefill, "wrong match. current stage is prefill."
            input_ids, positions = self.prepare_pearl_decode(exec_seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, is_prefill)
            # Currently, the temperature of the draft model is set to 0 to avoid communication overhead.
            # We will support temperature in the future.
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(exec_seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            # append the sample tokens to the seqs. Do not use postprocess to avoid early exiting when the draft tokens contain EOS.
            for seq, token_id in zip(exec_seqs, token_ids):
                seq.append_token(token_id)
            self._mark_trace_end(trace_record)

        accepted_lens, invalidated_lens = self.verify(exec_seqs, trace_record=trace_record, step_plan=step_plan)
        if trace_record is not None:
            self._update_trace_token_stats(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    def serialized_pearl_step(self):
        """Serialized-PEARL draft phase.

        Approximation baseline: draft generates with existing PEARL semantics, then
        all ranks synchronize before target verification compute is allowed to run.
        This disables draft/verify overlap without claiming vanilla serial
        speculative decoding equivalence.
        """
        trace_record = None
        for _ in range(self.gamma):
            seqs, is_prefill, step_plan = self._schedule_with_plan("serialized_draft")
            trace_record = self._trace_schedule(seqs, is_prefill, "serialized_draft", step_plan)
            exec_seqs = self._select_exec_seqs_for_plan(seqs, step_plan, "serialized_draft", trace_record)
            self._validate_stspec_probe_alignment(step_plan, "serialized_draft", trace_record)
            self._prepare_stspec_mailbox_route(step_plan, "serialized_draft", trace_record, exec_seqs)
            assert not is_prefill, "wrong match. current stage is prefill."
            input_ids, positions = self.prepare_pearl_decode(exec_seqs)
            torch.cuda.synchronize()
            self._mark_trace_start(trace_record)
            logits = self.run_model(input_ids, positions, is_prefill)
            sample_tokens = logits.argmax(dim=-1) if self.tp_params.local_rank == 0 else torch.zeros(len(exec_seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            torch.cuda.synchronize()
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)

            for seq, token_id in zip(exec_seqs, token_ids):
                seq.append_token(token_id)
            self._mark_trace_end(trace_record)

        # Global barrier pairs with TargetModelRunner.serialized_pearl_step().
        # It prevents target verification compute from overlapping this draft phase.
        dist.barrier()
        accepted_lens, invalidated_lens = self.verify(exec_seqs, trace_record=trace_record, step_plan=step_plan)
        if trace_record is not None:
            self._update_trace_token_stats(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    @torch.inference_mode()
    def verify(self, seqs: list[Sequence], trace_record: dict | None = None, step_plan: StepPlan | None = None):
        if self.tp_params.local_rank == 0:
            to_be_verified_tokens = []
            next_round_input = []
            for seq in seqs:
                if seq.pre_verify:
                    to_be_verified_tokens.append(seq.token_ids[-self.gamma])
                else:
                    to_be_verified_tokens.extend(seq.token_ids[-2*self.gamma+1:-self.gamma+1])
                next_round_input.extend(seq.token_ids[-self.gamma:])
            if self._pearl_protocol_enabled():
                draft_message = self._encode_draft_protocol_message(
                    seqs=seqs,
                    gamma=self.gamma,
                    draft_token_ids=to_be_verified_tokens,
                    next_round_input=next_round_input,
                    plan_id=step_plan.plan_id if step_plan is not None else None,
                    runner_role="draft",
                    scheduled_seq_ids=list(step_plan.scheduled_seq_ids) if step_plan is not None else [seq.seq_id for seq in seqs],
                    actual_exec_seq_ids=[seq.seq_id for seq in seqs],
                    target_batch_seq_ids=list(step_plan.target_batch_seq_ids) if step_plan is not None else [],
                    draft_home_batch_seq_ids=list(step_plan.draft_home_batch_seq_ids) if step_plan is not None else [],
                    protocol_version=self._pearl_protocol_version(),
                    layout_kind=self._pearl_protocol_layout(),
                )
                self._validate_and_trace_pearl_protocol(trace_record, draft_message, [seq.seq_id for seq in seqs])
                self._record_draft_mailbox_payloads(trace_record, step_plan, draft_message)
                to_be_verified_tokens, next_round_input = self._decode_draft_protocol_message(draft_message)
            msg = torch.tensor(to_be_verified_tokens + next_round_input, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.rank, group=self.verify_group)
        
        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)
        
        # post-process the seqs according to the verify_res.
        acc, rollout, revise_token, finish = verify_res.tolist()
        if self._pearl_protocol_enabled():
            verify_message = self._encode_verify_protocol_message(
                seqs=seqs,
                gamma=self.gamma,
                acc=acc,
                rollout=rollout,
                revise_token=revise_token,
                finish=finish,
                plan_id=step_plan.plan_id if step_plan is not None else None,
                runner_role="draft",
                scheduled_seq_ids=list(step_plan.scheduled_seq_ids) if step_plan is not None else [seq.seq_id for seq in seqs],
                actual_exec_seq_ids=[seq.seq_id for seq in seqs],
                target_batch_seq_ids=list(step_plan.target_batch_seq_ids) if step_plan is not None else [],
                draft_home_batch_seq_ids=list(step_plan.draft_home_batch_seq_ids) if step_plan is not None else [],
                protocol_version=self._pearl_protocol_version(),
                layout_kind=self._pearl_protocol_layout(),
            )
            self._validate_and_trace_pearl_protocol(trace_record, verify_message, [seq.seq_id for seq in seqs])
            acc, rollout, revise_token, finish = self._decode_verify_protocol_message(verify_message)
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
        initialize_v4t_active_continuation_runner_state(self)

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
        seqs, is_prefill, step_plan = self._schedule_with_plan("verify")
        trace_record = self._trace_schedule(seqs, is_prefill, "verify", step_plan)
        exec_seqs = self._select_exec_seqs_for_plan(seqs, step_plan, "verify", trace_record)
        self._validate_stspec_probe_alignment(step_plan, "verify", trace_record)
        if self._prepare_stspec_mailbox_route(step_plan, "verify", trace_record, exec_seqs):
            self._mark_trace_end(trace_record)
            return
        if trace_record.get("target_verify_skipped_for_warmup"):
            self._mark_trace_end(trace_record)
            return
        assert not is_prefill, "wrong match. current stage is prefill."
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(exec_seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        accepted_lens, invalidated_lens = self.verify(logits, exec_seqs, temperatures, trace_record=trace_record, step_plan=step_plan)
        torch.cuda.synchronize()
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

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
        seqs, is_prefill, step_plan = self._schedule_with_plan("serialized_verify")
        trace_record = self._trace_schedule(seqs, is_prefill, "serialized_verify", step_plan)
        exec_seqs = self._select_exec_seqs_for_plan(seqs, step_plan, "serialized_verify", trace_record)
        self._validate_stspec_probe_alignment(step_plan, "serialized_verify", trace_record)
        if self._prepare_stspec_mailbox_route(step_plan, "serialized_verify", trace_record, exec_seqs):
            self._mark_trace_end(trace_record)
            return
        if trace_record.get("target_verify_skipped_for_warmup"):
            self._mark_trace_end(trace_record)
            return
        assert not is_prefill, "wrong match. current stage is prefill."
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(exec_seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        accepted_lens, invalidated_lens = self.verify(logits, exec_seqs, temperatures, trace_record=trace_record, step_plan=step_plan)
        torch.cuda.synchronize()
        self._mark_trace_end(trace_record, accepted_lens=accepted_lens, invalidated_lens=invalidated_lens)

    @torch.inference_mode()
    def verify(self, logits: torch.Tensor, seqs: list[Sequence], temperatures: torch.Tensor, trace_record: dict | None = None, step_plan: StepPlan | None = None):
        """Refer to the verification logic in the draft model verification function."""
        # verify_res will be sent to the sub-process in the target group.
        num_to_be_verified_tokens = sum([1 if seq.pre_verify else self.gamma for seq in seqs])
        num_next_round_input = self.gamma * len(seqs)
        msg = torch.zeros(num_to_be_verified_tokens + num_next_round_input, dtype=torch.int64, device="cuda")
        dist.broadcast(msg, src=self.global_config.draft_config.master_rank, group=self.verify_group)
        to_be_verified_tokens = msg[:num_to_be_verified_tokens].tolist()
        next_round_input = msg[num_to_be_verified_tokens:].tolist()
        if self._pearl_protocol_enabled():
            draft_message = self._encode_draft_protocol_message(
                seqs=seqs,
                gamma=self.gamma,
                draft_token_ids=to_be_verified_tokens,
                next_round_input=next_round_input,
                plan_id=step_plan.plan_id if step_plan is not None else None,
                runner_role="verify",
                scheduled_seq_ids=list(step_plan.scheduled_seq_ids) if step_plan is not None else [seq.seq_id for seq in seqs],
                actual_exec_seq_ids=[seq.seq_id for seq in seqs],
                target_batch_seq_ids=list(step_plan.target_batch_seq_ids) if step_plan is not None else [],
                draft_home_batch_seq_ids=list(step_plan.draft_home_batch_seq_ids) if step_plan is not None else [],
                protocol_version=self._pearl_protocol_version(),
                layout_kind=self._pearl_protocol_layout(),
            )
            self._validate_and_trace_pearl_protocol(trace_record, draft_message, [seq.seq_id for seq in seqs])
            to_be_verified_tokens, next_round_input = self._decode_draft_protocol_message(draft_message)
        
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
        if self._pearl_protocol_enabled():
            verify_message = self._encode_verify_protocol_message(
                seqs=seqs,
                gamma=self.gamma,
                acc=acc,
                rollout=rollout,
                revise_token=revise_token,
                finish=finish,
                plan_id=step_plan.plan_id if step_plan is not None else None,
                runner_role="verify",
                scheduled_seq_ids=list(step_plan.scheduled_seq_ids) if step_plan is not None else [seq.seq_id for seq in seqs],
                actual_exec_seq_ids=[seq.seq_id for seq in seqs],
                target_batch_seq_ids=list(step_plan.target_batch_seq_ids) if step_plan is not None else [],
                draft_home_batch_seq_ids=list(step_plan.draft_home_batch_seq_ids) if step_plan is not None else [],
                protocol_version=self._pearl_protocol_version(),
                layout_kind=self._pearl_protocol_layout(),
            )
            self._validate_and_trace_pearl_protocol(trace_record, verify_message, [seq.seq_id for seq in seqs])
            acc, rollout, revise_token, finish = self._decode_verify_protocol_message(verify_message)
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
