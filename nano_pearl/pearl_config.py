from __future__ import annotations

import os
from nano_pearl.utils.pearl_logger import logger, get_model_name
from dataclasses import dataclass
from typing import ClassVar
from transformers import AutoConfig
import torch.distributed as dist


PHASE_1H0_EAGER_NOT_IMPLEMENTED = (
    "Phase 1H-0 only adds eager scaffolding; eager execution is not implemented yet."
)
EAGER_POLICIES = {"none", "tight_only"}
EAGER_TRACE_LEVELS = {"full", "summary", "minimal"}


def validate_eager_gamma(config, gamma: int) -> bool:
    gamma = int(gamma)
    if gamma <= 0:
        raise ValueError(f"global gamma must be positive for eager execution, got {gamma}")

    eager_gamma = getattr(config, "eager_gamma", None)
    if eager_gamma is None:
        eager_gamma = getattr(config, "max_eager_tokens_per_request", None)
    if eager_gamma is None:
        raise ValueError("eager_gamma is required for eager execution validation")
    eager_gamma = int(eager_gamma)

    max_tokens_per_request = int(getattr(config, "max_eager_tokens_per_request", 0) or 0)
    max_tokens_per_step = int(getattr(config, "max_eager_tokens_per_step", 0) or 0)
    max_requests_per_step = int(getattr(config, "max_eager_requests_per_step", 0) or 0)

    if eager_gamma != gamma:
        raise ValueError(f"eager_gamma must equal global gamma={gamma}, got {eager_gamma}")
    if max_tokens_per_request != gamma:
        raise ValueError(
            f"max_eager_tokens_per_request must equal global gamma={gamma}, got {max_tokens_per_request}"
        )
    if max_tokens_per_step <= 0 or max_tokens_per_step % gamma != 0:
        raise ValueError(
            f"max_eager_tokens_per_step must be a positive multiple of gamma={gamma}, "
            f"got {max_tokens_per_step}"
        )
    if max_requests_per_step * gamma > max_tokens_per_step:
        raise ValueError(
            "max_eager_requests_per_step * gamma must be <= max_eager_tokens_per_step, "
            f"got {max_requests_per_step} * {gamma} > {max_tokens_per_step}"
        )
    return True


@dataclass
class TPParams:
    rank: int
    group: dist.ProcessGroup
    group_name: str
    local_rank: int
    master_rank: int
    is_draft: bool
    tp_size: int
    valid_vocab_size: int


class BaseConfig:
    def __init__(self, model: str, tensor_parallel_size: int, devices: list[int], group_name: str):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.devices = devices
        self.group_name = group_name
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.eos = self.hf_config.eos_token_id
        self.master_rank = self.devices[0]
        logger.info(f"Model={get_model_name(self.model)}")
        logger.info(f"TP={self.tensor_parallel_size}")
        logger.info(f"Devices={self.devices}")
        logger.info(f"GroupName={self.group_name}")
        logger.info(f"Architectures={self.hf_config.architectures[0]}")
        logger.info(f"Vocab_Size={self.hf_config.vocab_size}")
        logger.info(f"Eos={self.eos}")

        # Dynamic TP: Padding Parameters
        if self.tensor_parallel_size not in [1, 2, 4, 8]:
            logger.warning(
                f"Currently, non-2-power TP is a developing feature, and you may encounter some unexpected errors."
            )
            num_heads = self.hf_config.num_attention_heads
            num_kv_heads = self.hf_config.num_key_value_heads
            intermediate_size = self.hf_config.intermediate_size
            vocab_size = self.hf_config.vocab_size
            tp = self.tensor_parallel_size
            
            from math import ceil
            
            gqa_ratio = num_heads // num_kv_heads
            padded_num_kv_heads = ceil(num_kv_heads / tp) * tp
            padded_num_heads = padded_num_kv_heads * gqa_ratio
            # Ensure per-rank intermediate shard is Tensor Core friendly (multiple of 128).
            # Pad total intermediate to a multiple of tp*128 so (intermediate/tp) % 128 == 0.
            TC_TILE = 128
            padded_intermediate_size = ceil(intermediate_size / (tp * TC_TILE)) * (tp * TC_TILE)
            padded_vocab_size = ceil(vocab_size / tp) * tp
            
            logger.info(f"Pad num heads from {num_heads} to {padded_num_heads}", color="red")
            logger.info(f"Pad num kv heads from {num_kv_heads} to {padded_num_kv_heads}", color="red")
            logger.info(f"Pad intermediate size from {intermediate_size} to {padded_intermediate_size} (per-rank {padded_intermediate_size // tp}, aligned to {TC_TILE})", color="red")
            logger.info(f"Pad vocab size from {vocab_size} to {padded_vocab_size}", color="red")
            self.hf_config.num_key_value_heads = padded_num_kv_heads
            self.hf_config.num_attention_heads = padded_num_heads
            self.hf_config.intermediate_size = padded_intermediate_size
            self.hf_config.valid_vocab_size = self.hf_config.vocab_size
            self.hf_config.vocab_size = padded_vocab_size

@dataclass
class PEARLConfig:
    ALLOWED_EXECUTION_MODES: ClassVar[set[str]] = {"ar", "serialized_pearl", "parallel_pearl", "dual_batch_pearl"}
    draft_model_path: str
    target_model_path: str
    draft_tensor_parallel_size: int = 2
    target_tensor_parallel_size: int = 2
    draft_group_name: str = "draft_group"
    target_group_name: str = "target_group"
    max_num_batched_tokens: int = 16384 # 8192 for 40GB GPUs
    max_num_seqs: int = 512 # 128 or 256 for 40GB GPUs
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enforce_eager: bool = False
    gamma: int = -1
    execution_mode: str = "parallel_pearl"
    enable_eager_execution: bool = False
    enable_eager_plan_dry_run: bool = False
    enable_eager_draft_dry_run: bool = False
    enable_eager_promotion_dry_run: bool = False
    enable_eager_transfer_dry_run: bool = False
    enable_eager_schedule_dry_run: bool = False
    enable_eager_verify_dry_run: bool = False
    enable_eager_apply_dry_run: bool = False
    enable_eager_result_transfer_dry_run: bool = False
    enable_eager_sync_apply_dry_run: bool = False
    enable_eager_commit_readiness_dry_run: bool = False
    enable_eager_commit_ready_only: bool = False
    enable_eager_lane_exclusion_dry_run: bool = False
    enable_continuous_eager_dry_run: bool = False
    enable_continuous_eager_verify_apply_dry_run: bool = False
    enable_continuous_eager_commit_depth1_ready_only: bool = False
    enable_rolling_continuous_eager_dry_run: bool = False
    enable_rolling_continuous_depth2_commit_ready_only: bool = False
    enable_rolling_continuous_depth3_shadow_dry_run: bool = False
    eager_policy: str = "none"
    max_eager_requests_per_step: int = 0
    max_eager_tokens_per_step: int = 0
    max_eager_tokens_per_request: int = 0
    eager_trace_level: str = "full"
    max_continuous_eager_chain_depth: int = 1
    max_continuous_eager_requests_per_step: int = 2
    max_continuous_eager_tokens_per_step: int = 0
    max_continuous_eager_tokens_per_request: int = 0
    max_rolling_continuous_depth: int = 2
    max_rolling_continuous_draft_children_per_step: int = 2
    max_rolling_continuous_seqs_per_step: int = 2

    def __post_init__(self):
        if self.execution_mode not in self.ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode={self.execution_mode!r}. "
                f"Expected one of {sorted(self.ALLOWED_EXECUTION_MODES)}."
            )
        self.enable_eager_execution = bool(self.enable_eager_execution)
        self.enable_eager_plan_dry_run = bool(self.enable_eager_plan_dry_run)
        self.enable_eager_draft_dry_run = bool(self.enable_eager_draft_dry_run)
        self.enable_eager_promotion_dry_run = bool(self.enable_eager_promotion_dry_run)
        self.enable_eager_transfer_dry_run = bool(self.enable_eager_transfer_dry_run)
        self.enable_eager_schedule_dry_run = bool(self.enable_eager_schedule_dry_run)
        self.enable_eager_verify_dry_run = bool(self.enable_eager_verify_dry_run)
        self.enable_eager_apply_dry_run = bool(self.enable_eager_apply_dry_run)
        self.enable_eager_result_transfer_dry_run = bool(self.enable_eager_result_transfer_dry_run)
        self.enable_eager_sync_apply_dry_run = bool(self.enable_eager_sync_apply_dry_run)
        self.enable_eager_commit_readiness_dry_run = bool(self.enable_eager_commit_readiness_dry_run)
        self.enable_eager_commit_ready_only = bool(self.enable_eager_commit_ready_only)
        self.enable_eager_lane_exclusion_dry_run = bool(self.enable_eager_lane_exclusion_dry_run)
        self.enable_continuous_eager_dry_run = bool(self.enable_continuous_eager_dry_run)
        self.enable_continuous_eager_verify_apply_dry_run = bool(self.enable_continuous_eager_verify_apply_dry_run)
        self.enable_continuous_eager_commit_depth1_ready_only = bool(
            self.enable_continuous_eager_commit_depth1_ready_only
        )
        self.enable_rolling_continuous_eager_dry_run = bool(self.enable_rolling_continuous_eager_dry_run)
        self.enable_rolling_continuous_depth2_commit_ready_only = bool(
            self.enable_rolling_continuous_depth2_commit_ready_only
        )
        self.enable_rolling_continuous_depth3_shadow_dry_run = bool(
            self.enable_rolling_continuous_depth3_shadow_dry_run
        )
        if self.enable_rolling_continuous_depth3_shadow_dry_run:
            self.enable_rolling_continuous_depth2_commit_ready_only = True
        if self.enable_rolling_continuous_depth2_commit_ready_only:
            self.enable_rolling_continuous_eager_dry_run = True
        if self.enable_rolling_continuous_eager_dry_run:
            self.enable_continuous_eager_commit_depth1_ready_only = True
            self.enable_continuous_eager_verify_apply_dry_run = True
        if self.enable_continuous_eager_commit_depth1_ready_only:
            self.enable_eager_commit_ready_only = True
            self.enable_continuous_eager_verify_apply_dry_run = True
        if self.enable_continuous_eager_verify_apply_dry_run:
            self.enable_continuous_eager_dry_run = True
        if self.enable_eager_commit_ready_only:
            self.enable_eager_commit_readiness_dry_run = True
        if self.enable_eager_commit_readiness_dry_run:
            self.enable_eager_sync_apply_dry_run = True
        if self.enable_eager_sync_apply_dry_run:
            self.enable_eager_result_transfer_dry_run = True
        if self.enable_eager_result_transfer_dry_run:
            self.enable_eager_apply_dry_run = True
        if self.enable_eager_apply_dry_run:
            self.enable_eager_verify_dry_run = True
        if self.enable_eager_verify_dry_run:
            self.enable_eager_lane_exclusion_dry_run = True
            self.enable_eager_schedule_dry_run = True
        if self.enable_eager_schedule_dry_run:
            self.enable_eager_transfer_dry_run = True
        if self.enable_eager_lane_exclusion_dry_run:
            self.enable_eager_schedule_dry_run = True
            self.enable_eager_transfer_dry_run = True
        if self.enable_eager_transfer_dry_run:
            self.enable_eager_promotion_dry_run = True
        if self.enable_eager_promotion_dry_run:
            self.enable_eager_draft_dry_run = True
        if self.enable_eager_draft_dry_run:
            self.enable_eager_plan_dry_run = True
        self.eager_policy = str(self.eager_policy)
        if self.eager_policy not in EAGER_POLICIES:
            raise ValueError(
                f"Invalid eager_policy={self.eager_policy!r}. "
                f"Expected one of {sorted(EAGER_POLICIES)}."
            )
        self.eager_trace_level = str(self.eager_trace_level)
        if self.eager_trace_level not in EAGER_TRACE_LEVELS:
            raise ValueError(
                f"Invalid eager_trace_level={self.eager_trace_level!r}. "
                f"Expected one of {sorted(EAGER_TRACE_LEVELS)}."
            )
        for field_name in (
            "max_eager_requests_per_step",
            "max_eager_tokens_per_step",
            "max_eager_tokens_per_request",
            "max_continuous_eager_chain_depth",
            "max_continuous_eager_requests_per_step",
            "max_continuous_eager_tokens_per_step",
            "max_continuous_eager_tokens_per_request",
            "max_rolling_continuous_depth",
            "max_rolling_continuous_draft_children_per_step",
            "max_rolling_continuous_seqs_per_step",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
            setattr(self, field_name, value)
        if self.max_continuous_eager_tokens_per_request <= 0 and int(self.gamma) > 0:
            self.max_continuous_eager_tokens_per_request = int(self.gamma)
        if self.max_continuous_eager_tokens_per_step <= 0:
            if self.max_eager_tokens_per_step > 0:
                self.max_continuous_eager_tokens_per_step = int(self.max_eager_tokens_per_step)
            else:
                self.max_continuous_eager_tokens_per_step = (
                    int(self.max_continuous_eager_requests_per_step)
                    * int(self.max_continuous_eager_tokens_per_request)
                )
        if self.enable_continuous_eager_dry_run and self.max_continuous_eager_chain_depth <= 0:
            raise ValueError("enable_continuous_eager_dry_run requires max_continuous_eager_chain_depth > 0")
        if self.enable_continuous_eager_dry_run and self.max_continuous_eager_requests_per_step <= 0:
            raise ValueError("enable_continuous_eager_dry_run requires max_continuous_eager_requests_per_step > 0")
        if self.enable_rolling_continuous_eager_dry_run:
            if self.max_rolling_continuous_depth <= 1:
                raise ValueError("enable_rolling_continuous_eager_dry_run requires max_rolling_continuous_depth > 1")
            if self.enable_rolling_continuous_depth3_shadow_dry_run and self.max_rolling_continuous_depth < 3:
                raise ValueError(
                    "enable_rolling_continuous_depth3_shadow_dry_run requires "
                    "max_rolling_continuous_depth >= 3"
                )
            if self.max_rolling_continuous_draft_children_per_step <= 0:
                raise ValueError(
                    "enable_rolling_continuous_eager_dry_run requires "
                    "max_rolling_continuous_draft_children_per_step > 0"
                )
            if self.max_rolling_continuous_seqs_per_step <= 0:
                raise ValueError(
                    "enable_rolling_continuous_eager_dry_run requires max_rolling_continuous_seqs_per_step > 0"
                )
        if self.enable_eager_execution:
            raise NotImplementedError(PHASE_1H0_EAGER_NOT_IMPLEMENTED)
        if self.enable_eager_draft_dry_run:
            if self.execution_mode != "dual_batch_pearl":
                raise ValueError(
                    "enable_eager_draft_dry_run requires execution_mode='dual_batch_pearl'"
                )
            if self.eager_policy == "none":
                raise ValueError("enable_eager_draft_dry_run requires eager_policy != 'none'")
            if self.max_eager_requests_per_step <= 0:
                raise ValueError(
                    "enable_eager_draft_dry_run requires max_eager_requests_per_step > 0"
                )
            if int(self.gamma) > 0:
                validate_eager_gamma(self, int(self.gamma))
        logger.info("="*50)
        logger.info(f"Loading Draft Config:")
        draft_devices = list(range(self.draft_tensor_parallel_size))
        self.draft_config = BaseConfig(self.draft_model_path, self.draft_tensor_parallel_size, draft_devices, self.draft_group_name)
        logger.info("="*50)
        logger.info(f"Loading Target Config:")
        target_devices = list(range(len(draft_devices), len(draft_devices) + self.target_tensor_parallel_size))
        self.target_config = BaseConfig(self.target_model_path, self.target_tensor_parallel_size, target_devices, self.target_group_name)
        logger.info("="*50)
        logger.info(f"Global_Config:")
        logger.info(f"Max_Num_Batched_Tokens={self.max_num_batched_tokens}")
        logger.info(f"Max_Num_Seqs={self.max_num_seqs}")
        logger.info(f"Max_Model_Len={self.max_model_len}")
        logger.info(f"GPU_Memory_Utilization={self.gpu_memory_utilization}")
        logger.info(f"Enforce_Eager={self.enforce_eager}")
        logger.info(f"Gamma (Window_Size)={self.gamma}, [-1 means auto-set]")
        logger.info(f"Execution_Mode={self.execution_mode}")
        logger.info(f"Enable_Eager_Execution={self.enable_eager_execution}")
        logger.info(f"Enable_Eager_Plan_Dry_Run={self.enable_eager_plan_dry_run}")
        logger.info(f"Enable_Eager_Draft_Dry_Run={self.enable_eager_draft_dry_run}")
        logger.info(f"Enable_Eager_Promotion_Dry_Run={self.enable_eager_promotion_dry_run}")
        logger.info(f"Enable_Eager_Transfer_Dry_Run={self.enable_eager_transfer_dry_run}")
        logger.info(f"Enable_Eager_Schedule_Dry_Run={self.enable_eager_schedule_dry_run}")
        logger.info(f"Enable_Eager_Verify_Dry_Run={self.enable_eager_verify_dry_run}")
        logger.info(f"Enable_Eager_Apply_Dry_Run={self.enable_eager_apply_dry_run}")
        logger.info(f"Enable_Eager_Result_Transfer_Dry_Run={self.enable_eager_result_transfer_dry_run}")
        logger.info(f"Enable_Eager_Sync_Apply_Dry_Run={self.enable_eager_sync_apply_dry_run}")
        logger.info(f"Enable_Eager_Commit_Readiness_Dry_Run={self.enable_eager_commit_readiness_dry_run}")
        logger.info(f"Enable_Eager_Commit_Ready_Only={self.enable_eager_commit_ready_only}")
        logger.info(f"Enable_Eager_Lane_Exclusion_Dry_Run={self.enable_eager_lane_exclusion_dry_run}")
        logger.info(f"Enable_Continuous_Eager_Dry_Run={self.enable_continuous_eager_dry_run}")
        logger.info(
            f"Enable_Continuous_Eager_Verify_Apply_Dry_Run={self.enable_continuous_eager_verify_apply_dry_run}"
        )
        logger.info(
            "Enable_Continuous_Eager_Commit_Depth1_Ready_Only="
            f"{self.enable_continuous_eager_commit_depth1_ready_only}"
        )
        logger.info(f"Enable_Rolling_Continuous_Eager_Dry_Run={self.enable_rolling_continuous_eager_dry_run}")
        logger.info(
            "Enable_Rolling_Continuous_Depth2_Commit_Ready_Only="
            f"{self.enable_rolling_continuous_depth2_commit_ready_only}"
        )
        logger.info(
            "Enable_Rolling_Continuous_Depth3_Shadow_Dry_Run="
            f"{self.enable_rolling_continuous_depth3_shadow_dry_run}"
        )
        logger.info(f"Eager_Policy={self.eager_policy}")
        logger.info(f"Eager_Trace_Level={self.eager_trace_level}")
        logger.info(f"Max_Eager_Requests_Per_Step={self.max_eager_requests_per_step}")
        logger.info(f"Max_Eager_Tokens_Per_Step={self.max_eager_tokens_per_step}")
        logger.info(f"Max_Eager_Tokens_Per_Request={self.max_eager_tokens_per_request}")
        logger.info(f"Max_Continuous_Eager_Chain_Depth={self.max_continuous_eager_chain_depth}")
        logger.info(f"Max_Continuous_Eager_Requests_Per_Step={self.max_continuous_eager_requests_per_step}")
        logger.info(f"Max_Continuous_Eager_Tokens_Per_Step={self.max_continuous_eager_tokens_per_step}")
        logger.info(f"Max_Continuous_Eager_Tokens_Per_Request={self.max_continuous_eager_tokens_per_request}")
        logger.info(f"Max_Rolling_Continuous_Depth={self.max_rolling_continuous_depth}")
        logger.info(
            "Max_Rolling_Continuous_Draft_Children_Per_Step="
            f"{self.max_rolling_continuous_draft_children_per_step}"
        )
        logger.info(f"Max_Rolling_Continuous_Seqs_Per_Step={self.max_rolling_continuous_seqs_per_step}")
        assert self.draft_config.eos == self.target_config.eos
        assert (self.draft_config.tensor_parallel_size + self.target_config.tensor_parallel_size) <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
        self.world_size = self.draft_config.tensor_parallel_size + self.target_config.tensor_parallel_size
        logger.info(f"World_Size={self.world_size}")
        logger.info("="*50)
    
