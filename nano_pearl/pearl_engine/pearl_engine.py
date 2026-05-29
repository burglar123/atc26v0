import atexit
import json
from dataclasses import fields
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import pickle
import os
import torch
from nano_pearl.pearl_config import PEARLConfig
from nano_pearl.pearl_engine.pearl_model_runner import DraftModelRunner, TargetModelRunner
from nano_pearl.utils.pearl_logger import logger
from multiprocessing.synchronize import Event
from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.layers.sampler import SamplingParams


class Controller:
    def __init__(self, config: PEARLConfig, control_event: Event):
        self.config = config
        self.draft_event = []
        self.target_event = []
        self.control_event = control_event
        self.draft_shm = SharedMemory(name=config.draft_config.group_name, create=True, size=2**24)
        self.target_shm = SharedMemory(name=config.target_config.group_name, create=True, size=2**24)

    def add_event(self, rank, event):
        if rank in self.config.draft_config.devices:
            self.draft_event.append(event)
        else:
            self.target_event.append(event)

    def write_draft_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.draft_shm.buf[0:4] = n.to_bytes(4, "little")
        self.draft_shm.buf[4:n+4] = data
        for event in self.draft_event:
            event.set()
        
    def write_target_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.target_shm.buf[0:4] = n.to_bytes(4, "little")
        self.target_shm.buf[4:n+4] = data
        for event in self.target_event:
            event.set()

    def read_payload(self, shm: SharedMemory):
        n = int.from_bytes(shm.buf[0:4], "little")
        data = shm.buf[4:n+4]
        payload = pickle.loads(data)
        if (
            isinstance(payload, list)
            and len(payload) == 2
            and payload[0] == "__PAYLOAD_FILE__"
            and isinstance(payload[1], str)
        ):
            path = payload[1]
            with open(path, "rb") as f:
                payload = pickle.load(f)
            try:
                os.remove(path)
            except Exception:
                pass
        if len(payload) == 2:
            output, elapsed_time = payload
            return output, elapsed_time, [], []
        output, elapsed_time, traces, service_metadata = payload
        return output, elapsed_time, traces, service_metadata

    def read_output(self):
        return self.read_payload(self.target_shm)

    @staticmethod
    def _merge_decode_iterations(traces):
        """Merge per-step draft/verify records into consolidated decode-iteration records.

        Groups records by ``decode_iteration_group`` and merges draft-phase timing
        with verify-phase token statistics into a single record per logical iteration.
        """
        groups: dict = {}
        for rec in traces:
            if rec.get("is_prefill"):
                continue
            gid = rec.get("decode_iteration_group", -1)
            if gid not in groups:
                groups[gid] = {"draft": [], "verify": []}
            role = rec.get("runner_role", "")
            if "draft" in role:
                groups[gid]["draft"].append(rec)
            elif "verify" in role:
                groups[gid]["verify"].append(rec)

        merged = []
        for gid in sorted(groups.keys()):
            drafts = groups[gid]["draft"]
            verifies = groups[gid]["verify"]
            if not drafts and not verifies:
                continue

            # Use the verify record as the primary source for token stats.
            primary = verifies[0] if verifies else drafts[-1]
            mode = primary.get("execution_mode", "")

            # Draft timing: span across all draft sub-records.
            draft_starts = [r["draft_start_ts"] for r in drafts if r.get("draft_start_ts") is not None]
            draft_ends = [r["draft_end_ts"] for r in drafts if r.get("draft_end_ts") is not None]
            draft_start_ts = min(draft_starts) if draft_starts else None
            draft_end_ts = max(draft_ends) if draft_ends else None
            draft_time_ms = (draft_end_ts - draft_start_ts) * 1000.0 if (draft_start_ts is not None and draft_end_ts is not None) else 0.0

            # Verify timing: from the verify record.
            verify_start_ts = primary.get("verify_start_ts")
            verify_end_ts = primary.get("verify_end_ts")
            verify_time_ms = primary.get("verify_time_ms")
            if verify_time_ms is None and verify_start_ts is not None and verify_end_ts is not None:
                verify_time_ms = (verify_end_ts - verify_start_ts) * 1000.0
            elif verify_time_ms is None:
                verify_time_ms = 0.0

            # Iteration span.
            all_starts = [t for t in [draft_start_ts, verify_start_ts] if t is not None]
            all_ends = [t for t in [draft_end_ts, verify_end_ts] if t is not None]
            iter_start_ts = min(all_starts) if all_starts else None
            iter_end_ts = max(all_ends) if all_ends else None
            iter_time_ms = (iter_end_ts - iter_start_ts) * 1000.0 if (iter_start_ts is not None and iter_end_ts is not None) else 0.0

            # Overlap (parallel_pearl only).
            overlap_time_ms = 0.0
            if "parallel_pearl" in mode and draft_start_ts is not None and verify_end_ts is not None:
                overlap_start = max(draft_start_ts, verify_start_ts or 0)
                overlap_end = min(draft_end_ts or float("inf"), verify_end_ts or float("inf"))
                if overlap_end > overlap_start:
                    overlap_time_ms = (overlap_end - overlap_start) * 1000.0

            # Token stats: aggregate across all draft records + use verify record's stats.
            drafted_tokens_total = sum(r.get("drafted_tokens_total", 0) for r in drafts)
            accepted_tokens_total = primary.get("total_accepted_tokens", 0)
            verified_tokens_total = drafted_tokens_total if drafted_tokens_total > 0 else (
                primary.get("num_seqs_in_batch", 0) * 4  # fallback: gamma * batch
            )

            # Per-request maps: prefer verify side (has accepted + rejected),
            # fall back to draft side for slo info.
            accepted_by_req = primary.get("accepted_tokens_per_seq", {})
            rejected_by_req = primary.get("rejected_tokens_by_request", {})
            invalidated_by_req = primary.get("per_seq_invalidated_predraft_len", {})
            slo_class_by_req = primary.get("slo_class_by_request", {})
            slo_tpot_by_req = primary.get("slo_tpot_ms_by_request", {})
            # Fill SLO info from draft records if verify side is missing them.
            if not slo_class_by_req and drafts:
                slo_class_by_req = drafts[-1].get("slo_class_by_request", {})
            if not slo_tpot_by_req and drafts:
                slo_tpot_by_req = drafts[-1].get("slo_tpot_ms_by_request", {})

            # Map seq_ids to request_ids.
            seq_to_req = {}
            for r in drafts + verifies:
                for sid, rid in zip(r.get("scheduled_seq_ids", []), r.get("request_ids", [])):
                    seq_to_req[str(sid)] = rid

            def _seq_to_req(d):
                return {seq_to_req.get(str(k), str(k)): v for k, v in d.items()}

            merged.append({
                "trace_type": "decode_iteration",
                "record_level": "merged_iteration",
                "execution_mode": mode,
                "decode_ready_mode": primary.get("decode_ready_mode", False),
                "decode_iteration_group": gid,
                "iteration_id": primary.get("iteration_id"),
                "active_batch_size": primary.get("num_seqs_in_batch", 0),
                "active_request_ids": primary.get("request_ids", []),
                "draft_start_ts": draft_start_ts,
                "draft_end_ts": draft_end_ts,
                "draft_time_ms": draft_time_ms,
                "verify_start_ts": verify_start_ts,
                "verify_end_ts": verify_end_ts,
                "verify_time_ms": verify_time_ms,
                "iter_start_ts": iter_start_ts,
                "iter_end_ts": iter_end_ts,
                "iter_time_ms": iter_time_ms,
                "draft_verify_overlap_ms": overlap_time_ms,
                "drafted_tokens_total": drafted_tokens_total,
                "verified_tokens_total": verified_tokens_total,
                "accepted_tokens_total": accepted_tokens_total,
                "accepted_tokens_by_request": _seq_to_req(accepted_by_req),
                "rejected_tokens_by_request": _seq_to_req(rejected_by_req),
                "invalidated_predraft_tokens_by_request": _seq_to_req(invalidated_by_req),
                "slo_class_by_request": _seq_to_req(slo_class_by_req),
                "slo_tpot_ms_by_request": _seq_to_req(slo_tpot_by_req),
            })

        return merged

    def read_all_traces(self):
        _, _, draft_traces, draft_requests = self.read_payload(self.draft_shm)
        _, _, target_traces, target_requests = self.read_payload(self.target_shm)
        traces = draft_traces + target_traces
        traces.sort(
            key=lambda record: (
                record.get("draft_start_ts")
                or record.get("verify_start_ts")
                or float("inf"),
                record.get("runner_role", ""),
                record.get("iteration_id", -1),
            )
        )
        # Merge per-step records into consolidated decode-iteration records.
        merged = self._merge_decode_iterations(traces)
        if merged:
            traces = merged
        requests = {req["seq_id"]: req for req in draft_requests + target_requests}.values()
        return traces, list(requests)


class PEARLEngine:    
    def __init__(self, config: PEARLConfig):
        self.config = config
        self.ps = []
        self._exited = False
        
        ctx = mp.get_context("spawn")
        # the control event is used to wait for the sub-processes to be ready
        self.control_event = ctx.Event()
        self.controller = Controller(config, self.control_event)
        self.last_traces = []
        self.last_request_metadata = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.draft_config.model, use_fast=True)
        config.eos = self.config.draft_config.eos
        logger.info(f"[Main Process] EOS token id: {config.eos}, EOS tokens: {self.tokenizer.decode(config.eos)}")   

        for i in range(config.world_size):
            event = ctx.Event()
            process = ctx.Process(target=DraftModelRunner if i in config.draft_config.devices else TargetModelRunner, args=(config, i, event, self.control_event))
            process.daemon = True        
            process.start()
            self.ps.append(process)
            self.controller.add_event(i, event)
        
        # wait for the initialization of the draft and target TP models
        logger.info("[Main Process] Waiting for the initialization of the draft and target TP models...", color="red")
        self.control_event.wait()
        self.control_event.clear()
        
        atexit.register(self.exit)
    

    def log(self, content: str):
        logger.info(f"[Main Process] Running log function, waiting for the sub-processes", color="red")
        self.controller.write_draft_shm("log", content)
        self.controller.write_target_shm("log", content)
        self.control_event.wait()
        self.control_event.clear()

    def run_model(self, seqs: list[Sequence], is_prefill: bool):        
        self.controller.write_draft_shm("run_model", seqs, is_prefill)
        self.controller.write_target_shm("run_model", seqs, is_prefill)
        self.control_event.wait()
        self.control_event.clear()

    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            self.controller.write_draft_shm("exit")
            self.controller.write_target_shm("exit")
        except Exception as exc:
            logger.warning(f"[Main Process] Failed to send exit command during cleanup: {exc}")
        for p in self.ps:
            p.join()
        for shm in (self.controller.draft_shm, self.controller.target_shm):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | int | None = None,
        arrival_ts: float | None = None,
        slo_tpot_ms: float | None = None,
        slo_class: str | None = None,
        per_request_gamma: int | None = None,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(
            prompt,
            sampling_params,
            request_id=request_id,
            arrival_ts=arrival_ts,
            slo_tpot_ms=slo_tpot_ms,
            slo_class=slo_class,
            per_request_gamma=per_request_gamma,
        )
        self.controller.write_draft_shm("add_request", seq)
        self.controller.write_target_shm("add_request", seq)
        self.control_event.wait()
        self.control_event.clear()
    
    def generate(self):
        self.controller.write_draft_shm("pearl_generate")
        self.controller.write_target_shm("pearl_generate")
        self.control_event.wait()
        self.control_event.clear()

        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]
        
        return output_text, num_tokens, num_acc_tokens, time

    def serialized_pearl_generate(self):
        """Run the serialized-PEARL approximation baseline.

        This is not strict vanilla serial speculative decoding. It reuses the
        PEARL runner semantics while disabling draft/verify overlap.
        """
        self.controller.write_draft_shm("serialized_pearl_generate")
        self.controller.write_target_shm("serialized_pearl_generate")
        self.control_event.wait()
        self.control_event.clear()

        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, num_acc_tokens, time

    def AR_generate(self):
        """Only use target model for Auto-Regressive generation."""
        self.controller.write_draft_shm("parallel_generate")
        self.controller.write_target_shm("parallel_generate")
        self.control_event.wait()
        self.control_event.clear()

        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, _ = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, None, time
    
    def bench_generate(self, num_pearl_steps: int = 100):
        self.controller.write_draft_shm("pearl_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("pearl_bench_generate", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()

        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, num_acc_tokens, time

    def serialized_pearl_bench_generate(self, num_pearl_steps: int = 100):
        """Benchmark the serialized-PEARL approximation baseline."""
        self.controller.write_draft_shm("serialized_pearl_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("serialized_pearl_bench_generate", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()

        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, num_acc_tokens, time


    def prepare_decode_ready(self):
        """Run unmeasured in-memory prefill for decoder-only benchmarking.

        This materializes prompt KV in the current runner processes and leaves
        requests resident for a subsequent decode_ready_generate() call. It does
        not persist KV cache to disk or across engine runs.
        """
        self.controller.write_draft_shm("prepare_decode_ready")
        self.controller.write_target_shm("prepare_decode_ready")
        self.control_event.wait()
        self.control_event.clear()

    def decode_ready_generate(self, execution_mode: str | None = None):
        """Run a decode-only measured phase after prepare_decode_ready().

        The returned elapsed time excludes the prefill done by
        prepare_decode_ready(). Traces/request metadata include
        decode_ready_mode=True and decode_start_ts/decode_ready_ts markers.
        """
        execution_mode = self.config.execution_mode if execution_mode is None else execution_mode
        if execution_mode not in self.config.ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode={execution_mode!r}. "
                f"Expected one of {sorted(self.config.ALLOWED_EXECUTION_MODES)}."
            )
        method_name = {
            "ar": "decode_ready_parallel_generate",
            "parallel_pearl": "decode_ready_pearl_generate",
            "serialized_pearl": "decode_ready_serialized_pearl_generate",
        }[execution_mode]
        self.controller.write_draft_shm(method_name)
        self.controller.write_target_shm(method_name)
        self.control_event.wait()
        self.control_event.clear()

        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        prefill_tokens = {
            req["seq_id"]: req.get("num_decode_ready_prefill_tokens", 0)
            for req in self.last_request_metadata
        }
        # Decode-ready mode excludes the unmeasured prefill token(s) from the
        # returned token counts so downstream TPOT uses decode-stage tokens only.
        num_tokens = [max(len(t) - prefill_tokens.get(seq, 0), 0) for seq, t in zip(seq_id, token_ids)]
        if execution_mode == "ar":
            num_acc_tokens = None

        return output_text, num_tokens, num_acc_tokens, time

    def cached_build_from_sequences(self, seqs: list[Sequence], cache_build_batch_size: int):
        self.controller.write_draft_shm("cache_build_prepare", seqs, cache_build_batch_size)
        self.controller.write_target_shm("cache_build_prepare", seqs, cache_build_batch_size)
        self.control_event.wait()
        self.control_event.clear()

    def add_cached_sequence(self, seq: Sequence):
        self.controller.write_draft_shm("add_cached_request", seq)
        self.controller.write_target_shm("add_cached_request", seq)
        self.control_event.wait()
        self.control_event.clear()

    def cached_decode_ready_generate(self, max_active_cached_seqs: int, execution_mode: str = "parallel_pearl"):
        if execution_mode not in self.config.ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode={execution_mode!r} for cached admission. "
                f"Expected one of {sorted(self.config.ALLOWED_EXECUTION_MODES)}."
            )
        if execution_mode not in ("ar", "parallel_pearl", "serialized_pearl"):
            raise ValueError(
                f"--cached-admission currently supports only --execution-mode "
                f"ar, parallel_pearl, or serialized_pearl, got {execution_mode!r}"
            )
        method_name = {
            "ar": "cached_decode_ready_ar_generate",
            "parallel_pearl": "cached_decode_ready_pearl_generate",
            "serialized_pearl": "cached_decode_ready_serialized_pearl_generate",
        }[execution_mode]
        self.controller.write_draft_shm(method_name, max_active_cached_seqs)
        self.controller.write_target_shm(method_name, max_active_cached_seqs)
        self.control_event.wait()
        self.control_event.clear()
        output, time, target_traces, target_request_metadata = self.controller.read_output()
        try:
            self.last_traces, self.last_request_metadata = self.controller.read_all_traces()
        except Exception:
            self.last_traces = target_traces
            self.last_request_metadata = target_request_metadata
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        prefill_tokens = {
            req["seq_id"]: req.get("num_decode_ready_prefill_tokens", 0)
            for req in self.last_request_metadata
        }
        num_tokens = [max(len(t) - prefill_tokens.get(seq, 0), 0) for seq, t in zip(seq_id, token_ids)]
        if execution_mode == "ar":
            num_acc_tokens = None
        return output_text, num_tokens, num_acc_tokens, time

    def get_traces(self):
        return {
            "traces": self.last_traces,
            "requests": self.last_request_metadata,
        }

    def dump_traces_json(self, path: str | os.PathLike | None = None, indent: int = 2):
        payload = self.get_traces()
        trace_json = json.dumps(payload, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(trace_json)
        return trace_json
