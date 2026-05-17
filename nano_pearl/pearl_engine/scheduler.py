from collections import deque

from nano_pearl.pearl_config import PEARLConfig
from nano_pearl.pearl_engine.sequence import Sequence, SequenceStatus
from nano_pearl.pearl_engine.block_manager import BlockManager
from nano_pearl.utils.pearl_logger import logger
from nano_pearl.pearl_engine.stspec_plan import StepPlan, build_legacy_step_plan


def is_eos(token_id: int, eos_token_id: int | list[int]):
    if isinstance(eos_token_id, int):
        return token_id == eos_token_id
    else:
        return token_id in eos_token_id

class Scheduler:

    def __init__(self, config: PEARLConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []
        self.iteration_id = 0
        # ST-Spec V3A shadow metadata only: assign deterministic 0/1 home
        # membership when requests enter the scheduler. This does not change
        # scheduling order or execution behavior; actual two-batch scheduling is
        # deferred to a later PR.
        self.next_home_batch_id = 0
        # ST-Spec V3B shadow-only two-batch rhythm. Decode plans alternate
        # conceptual target/draft home batches for diagnostics; schedule()
        # still returns the full legacy batch and no execution is filtered.
        self.two_batch_shadow_step = 0
        self.current_target_home_batch_id = 0
        self.current_draft_home_batch_id = 1
        self.enable_stspec_two_batch_execution = bool(
            getattr(config, "enable_stspec_two_batch_execution", False)
        )
        self.stspec_two_batch_dryrun = bool(
            getattr(config, "stspec_two_batch_dryrun", True)
        )

    def next_batch_id(self, runner_role: str) -> tuple[int, str]:
        iteration_id = self.iteration_id
        self.iteration_id += 1
        return iteration_id, f"{runner_role}-{iteration_id}"

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.home_batch_id is None:
            seq.home_batch_id = self.next_home_batch_id
            self.next_home_batch_id = 1 - self.next_home_batch_id
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                logger.warning(f"num_batched_tokens + len(seq): {num_batched_tokens + len(seq)}, max_num_batched_tokens: {self.max_num_batched_tokens}, self.block_manager.can_allocate(seq): {self.block_manager.can_allocate(seq)}")
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def schedule_with_plan(
        self,
        runner_role: str,
        execution_mode: str,
        decode_ready_mode: bool,
        default_gamma: int,
    ) -> tuple[list[Sequence], bool, StepPlan]:
        """Return the legacy schedule plus a scaffold StepPlan.

        This wrapper deliberately delegates to schedule() and does not alter the
        scheduler decision. The plan_id mirrors the next trace iteration id that
        _trace_schedule() will consume, preserving current iteration accounting.
        """
        if self.enable_stspec_two_batch_execution and not self.stspec_two_batch_dryrun:
            raise NotImplementedError(
                "Real ST-Spec two-batch execution is not implemented in V3C."
            )
        plan_id = self.iteration_id
        seqs, is_prefill = self.schedule()
        target_home_batch_id = None
        draft_home_batch_id = None
        if not is_prefill:
            target_home_batch_id = self.two_batch_shadow_step % 2
            draft_home_batch_id = 1 - target_home_batch_id
            self.current_target_home_batch_id = target_home_batch_id
            self.current_draft_home_batch_id = draft_home_batch_id
            self.two_batch_shadow_step += 1
        step_plan = build_legacy_step_plan(
            plan_id=plan_id,
            seqs=seqs,
            is_prefill=is_prefill,
            runner_role=runner_role,
            execution_mode=execution_mode,
            decode_ready_mode=decode_ready_mode,
            default_gamma=default_gamma,
            target_home_batch_id=target_home_batch_id,
            draft_home_batch_id=draft_home_batch_id,
            two_batch_execution_enabled=self.enable_stspec_two_batch_execution,
            two_batch_execution_dryrun=self.stspec_two_batch_dryrun,
        )
        return seqs, is_prefill, step_plan

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and is_eos(token_id, self.eos)) or seq.num_completion_tokens == seq.max_tokens:
                seq.mark_finished()
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self.finished.append(seq)

    def rollback(self, seq: Sequence, n: int):
        self.block_manager.rollback(seq, n)

    def clear(self):
        while self.waiting:
            seq = self.waiting.pop()
            self.block_manager.deallocate(seq)
        while self.running:
            seq = self.running.pop()
            self.block_manager.deallocate(seq)
        while self.finished:
            seq = self.finished.pop()
            self.block_manager.deallocate(seq)
        self.iteration_id = 0
        self.next_home_batch_id = 0
        self.two_batch_shadow_step = 0
        self.current_target_home_batch_id = 0
        self.current_draft_home_batch_id = 1
        self.block_manager.hash_to_block_id.clear()
        for block in self.block_manager.blocks:
            block.hash = -1
            block.token_ids = []