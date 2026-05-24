import os
from nano_pearl.utils.pearl_logger import logger, get_model_name
from dataclasses import dataclass
from typing import ClassVar
from transformers import AutoConfig
import torch.distributed as dist


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
    enable_eager_trace: bool = False
    max_eager_requests_per_step: int = 0
    max_eager_tokens_per_step: int = 0
    max_eager_tokens_per_request: int = 0
    eager_policy: str = "none"
    eager_accept_threshold: float = 0.0
    enable_eager_execution: bool = False
    disable_eager_base_len_fixup: bool = True

    def __post_init__(self):
        if self.enable_eager_execution:
            self.enable_eager_trace = True
        if self.execution_mode not in self.ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Invalid execution_mode={self.execution_mode!r}. "
                f"Expected one of {sorted(self.ALLOWED_EXECUTION_MODES)}."
            )
        if self.eager_policy not in {"none", "tight_only", "urgency"}:
            raise ValueError(
                f"Invalid eager_policy={self.eager_policy!r}. "
                "Expected one of ['none', 'tight_only', 'urgency']."
            )
        if self.enable_eager_execution:
            if self.execution_mode != "dual_batch_pearl":
                raise ValueError("--enable-eager-execution requires --execution-mode dual_batch_pearl")
            if self.eager_policy == "none":
                raise ValueError("--enable-eager-execution requires --eager-policy tight_only or urgency")
            if int(self.max_eager_requests_per_step) <= 0:
                raise ValueError("--enable-eager-execution requires --max-eager-requests-per-step > 0")
            if int(self.max_eager_tokens_per_step) <= 0:
                raise ValueError("--enable-eager-execution requires --max-eager-tokens-per-step > 0")
            if int(self.max_eager_tokens_per_request) <= 0:
                raise ValueError("--enable-eager-execution requires --max-eager-tokens-per-request > 0")
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
        logger.info(f"Enable_Eager_Trace={self.enable_eager_trace}")
        logger.info(f"Enable_Eager_Execution={self.enable_eager_execution}")
        logger.info(f"Eager_Policy={self.eager_policy}")
        logger.info(f"Max_Eager_Requests_Per_Step={self.max_eager_requests_per_step}")
        logger.info(f"Max_Eager_Tokens_Per_Step={self.max_eager_tokens_per_step}")
        logger.info(f"Max_Eager_Tokens_Per_Request={self.max_eager_tokens_per_request}")
        logger.info(f"Eager_Accept_Threshold={self.eager_accept_threshold}")
        assert self.draft_config.eos == self.target_config.eos
        assert (self.draft_config.tensor_parallel_size + self.target_config.tensor_parallel_size) <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
        self.world_size = self.draft_config.tensor_parallel_size + self.target_config.tensor_parallel_size
        logger.info(f"World_Size={self.world_size}")
        logger.info("="*50)
    
