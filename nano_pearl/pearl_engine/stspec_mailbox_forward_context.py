"""V4J guarded target-forward context wiring for ST-Spec mailbox inputs.

The helpers here are intentionally CPU-safe and side-effect free.  Runtime
model runners can turn the validated Python lists into CUDA tensors and install
nano-PEARL's low-level decode context immediately before calling run_model().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from nano_pearl.pearl_engine.pearl_protocol import validate_offsets


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TargetForwardMailboxContext:
    plan_id: int | None
    target_home_batch_id: int | str | None
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int | None]
    seq_ids: list[int]
    request_ids: list[Any]
    per_seq_lengths: list[int]
    offsets: list[int]
    total_tokens: int
    batch_size: int
    num_tokens: int
    cuda_graph_batch_size: int | None = None
    context_lens: list[int] = field(default_factory=list)
    block_tables: list[list[int]] = field(default_factory=list)
    kv_cache_slots: list[int | None] = field(default_factory=list)
    block_ids: list[int | None] = field(default_factory=list)
    attention_metadata_available: bool = False
    slot_mapping_available: bool = False
    can_run_model: bool = False
    cannot_run_reason: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    @property
    def input_ids_shape(self) -> list[int]:
        return [len(self.input_ids)]

    @property
    def positions_shape(self) -> list[int]:
        return [len(self.positions)]

    @property
    def slot_mapping_shape(self) -> list[int]:
        return [len(self.slot_mapping)] if self.slot_mapping else []

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


def build_target_forward_context_from_mailbox_input(
    verification_input: Any,
    kv_sync_plan: Any,
    exec_seqs: Iterable[Any],
    step_plan: Any,
    runner_state: Any | None = None,
) -> TargetForwardMailboxContext:
    """Build the decode context metadata needed for a mailbox target forward.

    This mirrors TargetModelRunner.prepare_pearl_decode for the mailbox subset:
    tokens are packed flat, each token gets its target position, KV slot, context
    length, and a per-token block table row for paged attention.  No Sequence or
    KV state is mutated.
    """

    seq_list = list(exec_seqs)
    seq_by_id = {int(seq.seq_id): seq for seq in seq_list}
    input_seq_ids = [int(seq_id) for seq_id in getattr(verification_input, "seq_ids", [])]
    actual_seq_ids = [int(seq_id) for seq_id in getattr(step_plan, "actual_target_exec_seq_ids", input_seq_ids)]
    scheduled_seq_ids = [int(seq_id) for seq_id in getattr(step_plan, "scheduled_seq_ids", [])]
    errors: list[str] = []
    error_kind: str | None = None

    input_ids = [int(token) for token in getattr(verification_input, "input_token_ids", [])]
    positions = [int(pos) for pos in getattr(kv_sync_plan, "mailbox_token_positions", [])]
    if not positions:
        positions = [int(pos) for pos in getattr(verification_input, "positions", []) if pos is not None]
    slot_mapping = list(getattr(kv_sync_plan, "kv_slot_ids", []) or [])
    if not slot_mapping:
        slot_mapping = list(getattr(verification_input, "kv_slot_ids", []) or [])
    slot_mapping = [None if slot is None else int(slot) for slot in slot_mapping]

    lengths = [int(length) for length in getattr(verification_input, "per_seq_lengths", [])]
    offsets = [int(offset) for offset in getattr(verification_input, "offsets", [])]
    total_tokens = int(getattr(verification_input, "total_tokens", len(input_ids)) or 0)
    request_ids = list(getattr(verification_input, "request_ids", []) or [])
    target_home_batch_id = getattr(step_plan, "target_home_batch_id", None)
    input_home_batch_id = getattr(verification_input, "target_home_batch_id", None)

    if scheduled_seq_ids and scheduled_seq_ids != actual_seq_ids and input_seq_ids == scheduled_seq_ids:
        error_kind = "illegal_legacy_fallback"
        errors.append("mailbox context input uses scheduled full batch instead of actual target exec seq ids")
    if input_seq_ids != actual_seq_ids:
        error_kind = error_kind or "target_forward_seq_mismatch"
        errors.append(f"input_seq_ids={input_seq_ids} do not match actual_target_exec_seq_ids={actual_seq_ids}")
    if [int(seq.seq_id) for seq in seq_list] != actual_seq_ids:
        error_kind = error_kind or "target_forward_exec_seq_mismatch"
        errors.append(f"exec_seq_ids={[int(seq.seq_id) for seq in seq_list]} do not match actual_target_exec_seq_ids={actual_seq_ids}")
    if len(set(input_seq_ids)) != len(input_seq_ids):
        error_kind = error_kind or "duplicate_target_forward_seq_ids"
        errors.append(f"duplicate seq ids in mailbox context: {input_seq_ids}")
    if input_home_batch_id != target_home_batch_id:
        error_kind = error_kind or "target_forward_home_batch_mismatch"
        errors.append(f"home_batch_id mismatch: input={input_home_batch_id}, target={target_home_batch_id}")

    try:
        validate_offsets(lengths, offsets, total_tokens, message_type="target_forward_mailbox_context", layout_kind="variable_offsets", seq_ids=input_seq_ids)
    except Exception as exc:  # keep structured, CPU-safe diagnostics.
        error_kind = error_kind or type(exc).__name__
        errors.append(str(exc))

    if not input_ids:
        error_kind = error_kind or "missing_mailbox_input_ids"
        errors.append("input_ids are unavailable for mailbox target forward")
    if not positions:
        error_kind = error_kind or "missing_mailbox_positions"
        errors.append("positions are unavailable for mailbox target forward")
    if len(input_ids) != total_tokens:
        error_kind = error_kind or "mailbox_input_length_mismatch"
        errors.append(f"input_ids length {len(input_ids)} != total_tokens {total_tokens}")
    if len(positions) != len(input_ids):
        error_kind = error_kind or "mailbox_position_length_mismatch"
        errors.append(f"positions length {len(positions)} != input_ids length {len(input_ids)}")
    if len(slot_mapping) != len(input_ids):
        error_kind = error_kind or "mailbox_slot_mapping_backend"
        errors.append(f"slot_mapping length {len(slot_mapping)} != input_ids length {len(input_ids)}")
    if any(slot is None for slot in slot_mapping):
        error_kind = error_kind or "mailbox_slot_mapping_backend"
        errors.append("slot_mapping contains unavailable KV slots")

    context_lens = [int(pos) + 1 for pos in positions] if len(positions) == len(input_ids) else []
    block_tables: list[list[int]] = []
    if not errors and slot_mapping:
        block_tables = _build_per_token_block_tables(seq_by_id, input_seq_ids, offsets, lengths)
        if len(block_tables) != len(input_ids):
            error_kind = error_kind or "mailbox_attention_metadata_backend"
            errors.append(f"block_tables rows {len(block_tables)} != input_ids length {len(input_ids)}")
    elif slot_mapping:
        # Still expose whether block tables are derivable for diagnostics.
        block_tables = _build_per_token_block_tables(seq_by_id, input_seq_ids, offsets, lengths)

    attention_metadata_available = bool(context_lens and len(context_lens) == len(input_ids) and block_tables and len(block_tables) == len(input_ids))
    slot_mapping_available = bool(slot_mapping and len(slot_mapping) == len(input_ids) and not any(slot is None for slot in slot_mapping))
    if slot_mapping_available and not attention_metadata_available and not errors:
        error_kind = "mailbox_attention_metadata_backend"
        errors.append("attention metadata/block tables are unavailable for mailbox target forward")

    can_run_model = bool(not errors and slot_mapping_available and attention_metadata_available)
    cannot_run_reason = None if can_run_model else (error_kind or "target_forward_mailbox_context_invalid")
    cuda_graph_batch_size = _next_cuda_graph_batch_size(len(input_ids), runner_state)
    return TargetForwardMailboxContext(
        plan_id=getattr(verification_input, "plan_id", None),
        target_home_batch_id=input_home_batch_id,
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        seq_ids=input_seq_ids,
        request_ids=request_ids,
        per_seq_lengths=lengths,
        offsets=offsets,
        total_tokens=total_tokens,
        batch_size=len(input_seq_ids),
        num_tokens=len(input_ids),
        cuda_graph_batch_size=cuda_graph_batch_size,
        context_lens=context_lens,
        block_tables=block_tables,
        kv_cache_slots=list(slot_mapping),
        block_ids=[None if block is None else int(block) for block in getattr(kv_sync_plan, "block_ids", [])],
        attention_metadata_available=attention_metadata_available,
        slot_mapping_available=slot_mapping_available,
        can_run_model=can_run_model,
        cannot_run_reason=cannot_run_reason,
        error_kind=error_kind,
        error_message="; ".join(errors) if errors else None,
        metadata={
            "actual_target_exec_seq_ids": actual_seq_ids,
            "scheduled_seq_ids": scheduled_seq_ids,
            "no_state_mutation": True,
            "layout_kind": getattr(verification_input, "layout_kind", "variable_offsets"),
        },
    )


def validate_target_forward_mailbox_context(context: TargetForwardMailboxContext) -> None:
    if not context.input_ids:
        raise RuntimeError("target forward mailbox context missing input_ids")
    if not context.positions:
        raise RuntimeError("target forward mailbox context missing positions")
    if not context.slot_mapping_available or not context.slot_mapping:
        raise RuntimeError("target forward mailbox context missing slot_mapping; next_required_feature=mailbox_slot_mapping_backend")
    if len(context.input_ids) != len(context.positions):
        raise RuntimeError("target forward mailbox context input_ids/positions length mismatch")
    if len(context.input_ids) != len(context.slot_mapping):
        raise RuntimeError("target forward mailbox context input_ids/slot_mapping length mismatch")
    if len(set(context.seq_ids)) != len(context.seq_ids):
        raise RuntimeError(f"target forward mailbox context duplicate seq_ids={context.seq_ids}")
    if not context.can_run_model:
        raise RuntimeError(
            f"target forward mailbox context cannot run model: {context.cannot_run_reason}; {context.error_message}"
        )


def _build_per_token_block_tables(
    seq_by_id: dict[int, Any],
    seq_ids: list[int],
    offsets: list[int],
    lengths: list[int],
) -> list[list[int]]:
    rows: list[list[int]] = []
    max_len = 0
    raw_rows: list[list[int]] = []
    for seq_id, length in zip(seq_ids, lengths):
        seq = seq_by_id.get(int(seq_id))
        table = [int(block) for block in list(getattr(seq, "block_table", []) or [])]
        max_len = max(max_len, len(table))
        raw_rows.extend([table] * int(length))
    if max_len == 0:
        return []
    for table in raw_rows:
        rows.append(table + [-1] * (max_len - len(table)))
    return rows


def _next_cuda_graph_batch_size(num_tokens: int, runner_state: Any | None) -> int | None:
    graph_bs = list(getattr(runner_state, "graph_bs", []) or [])
    for graph_size in graph_bs:
        if int(graph_size) >= int(num_tokens):
            return int(graph_size)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(inner) for inner in value]
    return value
