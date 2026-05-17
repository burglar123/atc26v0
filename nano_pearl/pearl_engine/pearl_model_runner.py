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
from nano_pearl.pearl_engine.stspec_plan import StepPlan
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
        assert step_plan.actual_target_exec_seq_ids == scheduled_seq_ids
        assert step_plan.actual_draft_exec_seq_ids == scheduled_seq_ids
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
            "actual_target_exec_seq_ids": list(step_plan.actual_target_exec_seq_ids),
            "actual_draft_exec_seq_ids": list(step_plan.actual_draft_exec_seq_ids),
            "dryrun_target_exec_seq_ids": list(step_plan.dryrun_target_exec_seq_ids),
            "dryrun_draft_exec_seq_ids": list(step_plan.dryrun_draft_exec_seq_ids),
            "actual_exec_seq_ids": actual_exec_seq_ids,
            "dryrun_exec_seq_ids": dryrun_exec_seq_ids,
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
        runner_role = self._runner_role()
        seqs, is_prefill, step_plan = self._schedule_with_plan(runner_role)
        trace_record = self._trace_schedule(seqs, is_prefill, runner_role, step_plan)
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

    def pearl_step(self):
        seqs, is_prefill, step_plan = self._schedule_with_plan("verify")
        trace_record = self._trace_schedule(seqs, is_prefill, "verify", step_plan)
        assert not is_prefill, "wrong match. current stage is prefill."
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        torch.cuda.synchronize()
        self._mark_trace_start(trace_record)
        logits = self.run_model(input_ids, positions, is_prefill)
        accepted_lens, invalidated_lens = self.verify(logits, seqs, temperatures)
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
        assert not is_prefill, "wrong match. current stage is prefill."
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
