import pickle
import torch
import time
import random
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
        self.trace_records.append(record)
        return record


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

    def _run_target_forward_from_mailbox_input(
        self,
        verification_input,
        exec_seqs: list[Sequence],
        step_plan: StepPlan,
        trace_record: dict,
    ) -> None:
        exec_seq_ids = [int(seq.seq_id) for seq in exec_seqs]
        input_seq_ids = [int(seq_id) for seq_id in verification_input.seq_ids]
        scheduled_seq_ids = [int(seq_id) for seq_id in step_plan.scheduled_seq_ids]
        actual_target_exec_seq_ids = [int(seq_id) for seq_id in step_plan.actual_target_exec_seq_ids]

        trace_record["target_forward_from_mailbox_input_built"] = True
        trace_record["target_forward_from_mailbox_input_seq_ids"] = input_seq_ids
        trace_record["target_forward_from_mailbox_input_total_tokens"] = int(verification_input.total_tokens)
        trace_record["target_forward_from_mailbox_input_shape"] = list(verification_input.input_shape)

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
            trace_record["target_tp_skipped_non_owner"] = True
            trace_record["target_forward_from_mailbox_output_interpretation_attempted"] = False
            trace_record["target_forward_from_mailbox_output_interpretation_success"] = False
            trace_record["mailbox_verify_apply_attempted"] = False
            trace_record["mailbox_verify_apply_success"] = False
            return
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
        trace_record["mailbox_verify_apply_success"] = False
        message = "mailbox verification apply path is not implemented"
        trace_record["mailbox_verify_apply_error"] = message
        trace_record["next_required_feature"] = "mailbox_verify_apply_path"
        raise RuntimeError(f"{message}; next_required_feature=mailbox_verify_apply_path")

    def _prepare_stspec_mailbox_route(
        self,
        step_plan: StepPlan,
        runner_role: str,
        trace_record: dict,
        exec_seqs: list[Sequence] | None = None,
    ) -> None:
        """V4E mailbox transport/consume preflight for real variable-offset probes.

        Draft runners now continue to produce a validated mailbox transport
        envelope after the draft payload exists. Target runners first classify
        pipeline warmup separately from transport/payload misses; if payloads are
        present, the probe stops at the next explicit blocker: verification input
        construction from mailbox payloads is not wired yet.
        """
        if not self._stspec_mailbox_enabled(step_plan):
            return
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
            return

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
                    trace_record["mailbox_payload_missing_reason"] = "mailbox_payload_tensor_unavailable_on_non_owner"
                    trace_record["target_forward_output_none_expected"] = True
                    return
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
            self._run_target_forward_from_mailbox_input(
                verification_input,
                exec_seqs or [],
                step_plan,
                trace_record,
            )
            return

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
                return
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
            return
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
        trace_record["stspec_mailbox_transport_enabled"] = True
        trace_record["mailbox_transport_mode"] = self._mailbox_transport_mode()
        trace_record["mailbox_put_attempted"] = True
        trace_record["mailbox_put_home_batch_id"] = step_plan.draft_home_batch_id
        trace_record["mailbox_put_seq_ids"] = [payload.seq_id for payload in payloads]
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
            trace_record["mailbox_put_count"] = 0
            self._record_mailbox_error(
                trace_record,
                kind=exc.kind,
                message=str(exc),
                next_required_feature="mailbox_payload_tensor_transport",
            )
            raise
        trace_record["mailbox_put_success"] = True
        trace_record["mailbox_put_count"] = len(payloads)
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
        self._prepare_stspec_mailbox_route(step_plan, "verify", trace_record, exec_seqs)
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
        self._prepare_stspec_mailbox_route(step_plan, "serialized_verify", trace_record, exec_seqs)
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
