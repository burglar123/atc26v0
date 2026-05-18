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


@dataclass(frozen=True)
class TargetTPMailboxRole:
    is_output_owner: bool
    is_payload_owner: bool
    should_run_target_forward: bool
    should_interpret_output: bool
    should_apply_verify_result: bool
    should_skip_non_owner: bool
    owner_rank: int | None
    current_rank: int | None
    reason: str | None = None

    def to_dict(self) -> JsonDict:
        return _jsonable(asdict(self))


def classify_target_tp_rank_role_for_mailbox_forward(runner_state: Any | None = None) -> TargetTPMailboxRole:
    """Classify target TP role for mailbox-forward probe decisions.

    nano-PEARL gathers logits to TP local rank 0.  All TP ranks may run the
    forward backend, but only the owner rank should interpret logits or advance
    the mailbox verify/apply probe boundary.
    """

    tp_params = getattr(runner_state, "tp_params", None)
    group_config = getattr(runner_state, "group_config", None)
    current_rank = getattr(runner_state, "rank", None)
    current_rank = None if current_rank is None else int(current_rank)
    owner_rank = target_forward_output_owner_rank(runner_state)
    is_owner = is_target_forward_output_owner(
        current_rank,
        group_config,
        tp_params,
        runner_role="verify",
    )
    return TargetTPMailboxRole(
        is_output_owner=is_owner,
        is_payload_owner=is_owner,
        should_run_target_forward=True,
        should_interpret_output=is_owner,
        should_apply_verify_result=is_owner,
        should_skip_non_owner=not is_owner,
        owner_rank=owner_rank,
        current_rank=current_rank,
        reason="output_owner" if is_owner else "non_owner_skip_interpret_apply",
    )


@dataclass(frozen=True)
class TargetForwardMailboxOutput:
    output_available: bool
    output_owner: bool
    output_owner_rank: int | None
    current_rank: int | None
    raw_output_type: str
    logits: Any = None
    logits_shape: list[int] = field(default_factory=list)
    output_shape: list[int] = field(default_factory=list)
    output_num_rows: int | None = None
    output_num_tokens: int | None = None
    seq_ids: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    per_seq_lengths: list[int] = field(default_factory=list)
    total_tokens: int = 0
    can_interpret: bool = False
    cannot_interpret_reason: str | None = None
    next_required_feature: str | None = None
    extraction_path: str | None = None
    output_none_expected: bool = False
    output_none_unexpected: bool = False
    error_kind: str | None = None
    error_message: str | None = None
    interpretation_map: list[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "output_available": self.output_available,
            "output_owner": self.output_owner,
            "output_owner_rank": self.output_owner_rank,
            "current_rank": self.current_rank,
            "raw_output_type": self.raw_output_type,
            "logits_shape": list(self.logits_shape),
            "output_shape": list(self.output_shape),
            "output_num_rows": self.output_num_rows,
            "output_num_tokens": self.output_num_tokens,
            "seq_ids": list(self.seq_ids),
            "offsets": list(self.offsets),
            "per_seq_lengths": list(self.per_seq_lengths),
            "total_tokens": self.total_tokens,
            "can_interpret": self.can_interpret,
            "cannot_interpret_reason": self.cannot_interpret_reason,
            "next_required_feature": self.next_required_feature,
            "extraction_path": self.extraction_path,
            "output_none_expected": self.output_none_expected,
            "output_none_unexpected": self.output_none_unexpected,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "interpretation_map": _jsonable(self.interpretation_map),
        }


def is_target_forward_output_owner(
    rank: int | None = None,
    target_config: Any | None = None,
    tp_params: Any | None = None,
    runner_role: str | None = None,
) -> bool:
    """Return whether this rank owns gathered target logits.

    The normal target verification path only samples/interprets logits on TP
    local rank 0; embed_head gathers sharded logits to tp_params.master_rank and
    returns None on other TP ranks.  Mailbox verification mirrors that behavior.
    """

    del runner_role  # reserved for future role-specific ownership rules.
    local_rank = getattr(tp_params, "local_rank", None)
    if local_rank is not None:
        return int(local_rank) == 0
    master_rank = getattr(tp_params, "master_rank", None)
    if master_rank is None and target_config is not None:
        master_rank = getattr(target_config, "master_rank", None)
    if rank is not None and master_rank is not None:
        return int(rank) == int(master_rank)
    return True


def target_forward_output_owner_rank(runner_state: Any | None = None) -> int | None:
    tp_params = getattr(runner_state, "tp_params", None)
    group_config = getattr(runner_state, "group_config", None)
    owner_rank = getattr(tp_params, "master_rank", None)
    if owner_rank is None and group_config is not None:
        owner_rank = getattr(group_config, "master_rank", None)
    return None if owner_rank is None else int(owner_rank)


def normalize_target_forward_from_mailbox_output(
    raw_output: Any,
    verification_input: Any,
    step_plan: Any,
    runner_state: Any | None = None,
    trace_record: dict | None = None,
) -> TargetForwardMailboxOutput:
    """Normalize TP-aware mailbox target-forward output for interpretation.

    Non-owner TP ranks are allowed to receive None because the normal LM head
    gathers logits only to TP local rank 0.  Owner ranks must produce a
    tensor-like logits object with a first dimension matching total_tokens.
    """

    del trace_record  # normalization is pure; caller exports fields.
    tp_params = getattr(runner_state, "tp_params", None)
    group_config = getattr(runner_state, "group_config", None)
    current_rank = getattr(runner_state, "rank", None)
    current_rank = None if current_rank is None else int(current_rank)
    owner_rank = target_forward_output_owner_rank(runner_state)
    expected_owner = is_target_forward_output_owner(
        current_rank,
        group_config,
        tp_params,
        runner_role="verify",
    )
    seq_ids = [int(seq_id) for seq_id in getattr(verification_input, "seq_ids", [])]
    offsets = [int(offset) for offset in getattr(verification_input, "offsets", [])]
    lengths = [int(length) for length in getattr(verification_input, "per_seq_lengths", [])]
    total_tokens = int(getattr(verification_input, "total_tokens", 0) or 0)
    raw_type = type(raw_output).__name__ if raw_output is not None else "NoneType"

    if raw_output is None:
        none_expected = not expected_owner
        none_unexpected = expected_owner
        return TargetForwardMailboxOutput(
            output_available=False,
            output_owner=expected_owner,
            output_owner_rank=owner_rank,
            current_rank=current_rank,
            raw_output_type=raw_type,
            seq_ids=seq_ids,
            offsets=offsets,
            per_seq_lengths=lengths,
            total_tokens=total_tokens,
            can_interpret=False,
            cannot_interpret_reason="non_owner_rank_no_output" if none_expected else "owner_rank_missing_output",
            next_required_feature=None if none_expected else "target_forward_output_ownership",
            output_none_expected=none_expected,
            output_none_unexpected=none_unexpected,
            error_kind=None if none_expected else "target_forward_output_ownership",
            error_message=None if none_expected else "target forward output owner rank received None logits",
        )

    logits, extraction_path = _extract_tensor_like_output(raw_output)
    if logits is None:
        return TargetForwardMailboxOutput(
            output_available=False,
            output_owner=expected_owner,
            output_owner_rank=owner_rank,
            current_rank=current_rank,
            raw_output_type=raw_type,
            seq_ids=seq_ids,
            offsets=offsets,
            per_seq_lengths=lengths,
            total_tokens=total_tokens,
            can_interpret=False,
            cannot_interpret_reason="unsupported_output_type",
            next_required_feature="target_forward_output_normalization",
            error_kind="target_forward_output_normalization",
            error_message=f"unsupported target forward output type {raw_type}",
        )

    shape = _shape_list(logits)
    rows = int(shape[0]) if shape else 0
    if rows != total_tokens:
        return TargetForwardMailboxOutput(
            output_available=True,
            output_owner=True,
            output_owner_rank=owner_rank,
            current_rank=current_rank,
            raw_output_type=raw_type,
            logits=logits,
            logits_shape=shape,
            output_shape=shape,
            output_num_rows=rows,
            output_num_tokens=total_tokens,
            seq_ids=seq_ids,
            offsets=offsets,
            per_seq_lengths=lengths,
            total_tokens=total_tokens,
            can_interpret=False,
            cannot_interpret_reason="target_forward_output_row_count_mismatch",
            next_required_feature="target_forward_output_normalization",
            extraction_path=extraction_path,
            error_kind="target_forward_output_row_count_mismatch",
            error_message=f"target forward output rows {rows} != mailbox total_tokens {total_tokens}",
        )

    interpretation_map = build_target_forward_output_interpretation_map(verification_input, shape)
    return TargetForwardMailboxOutput(
        output_available=True,
        output_owner=True,
        output_owner_rank=owner_rank,
        current_rank=current_rank,
        raw_output_type=raw_type,
        logits=logits,
        logits_shape=shape,
        output_shape=shape,
        output_num_rows=rows,
        output_num_tokens=total_tokens,
        seq_ids=seq_ids,
        offsets=offsets,
        per_seq_lengths=lengths,
        total_tokens=total_tokens,
        can_interpret=True,
        extraction_path=extraction_path,
        interpretation_map=interpretation_map,
    )


def build_target_forward_output_interpretation_map(
    verification_input: Any,
    output_shape: Iterable[int] | None = None,
) -> list[JsonDict]:
    shape = list(output_shape or [])
    total_tokens = int(getattr(verification_input, "total_tokens", 0) or 0)
    if shape:
        rows = int(shape[0])
        if rows != total_tokens:
            raise RuntimeError(
                "Target forward mailbox output row count does not match input tokens: "
                f"output_shape={shape}, total_tokens={total_tokens}; "
                "next_required_feature=target_forward_output_normalization"
            )
    mapping: list[JsonDict] = []
    for seq_id, offset, length in zip(
        getattr(verification_input, "seq_ids", []),
        getattr(verification_input, "offsets", []),
        getattr(verification_input, "per_seq_lengths", []),
    ):
        row_start = int(offset)
        row_end = row_start + int(length)
        mapping.append(
            {
                "seq_id": int(seq_id),
                "offset": int(offset),
                "length": int(length),
                "row_start": row_start,
                "row_end": row_end,
                "token_range": [row_start, row_end],
            }
        )
    return mapping


def _extract_tensor_like_output(raw_output: Any) -> tuple[Any | None, str | None]:
    if _is_tensor_like(raw_output):
        return raw_output, "self"
    if isinstance(raw_output, dict):
        for key in ("logits", "output", "outputs", "last_hidden_state"):
            value = raw_output.get(key)
            if _is_tensor_like(value):
                return value, f"dict.{key}"
        for key, value in raw_output.items():
            if _is_tensor_like(value):
                return value, f"dict.{key}"
        return None, None
    if isinstance(raw_output, (tuple, list)):
        for idx, value in enumerate(raw_output):
            if _is_tensor_like(value):
                return value, f"{type(raw_output).__name__}[{idx}]"
        return None, None
    for attr in ("logits", "output", "last_hidden_state"):
        value = getattr(raw_output, attr, None)
        if _is_tensor_like(value):
            return value, f"attr.{attr}"
    return None, None


def _is_tensor_like(value: Any) -> bool:
    return value is not None and hasattr(value, "shape")


def _shape_list(value: Any) -> list[int]:
    return [int(dim) for dim in list(getattr(value, "shape"))]


def classify_mailbox_payload_availability(
    *,
    target_home_batch_id: int | str | None,
    target_seq_ids: Iterable[int],
    available_home_batch_ids: Iterable[int | str | None],
    available_seq_ids_by_batch: dict[str, Iterable[int]] | None,
    missing_seq_ids: Iterable[int],
    payloads: Iterable[Any] | None = None,
    is_payload_owner: bool = True,
    owner_rank: int | None = None,
    current_rank: int | None = None,
) -> JsonDict:
    """Classify mailbox payload availability without conflating metadata and tensors."""

    seq_ids = [int(seq_id) for seq_id in target_seq_ids]
    missing = [int(seq_id) for seq_id in missing_seq_ids]
    available_batches = set(available_home_batch_ids)
    available_by_batch = available_seq_ids_by_batch or {}
    available_seq_ids = {int(seq_id) for seq_id in available_by_batch.get(str(target_home_batch_id), [])}
    payload_list = list(payloads or [])
    envelope_available = target_home_batch_id in available_batches
    available_for_seq_ids = bool(seq_ids) and all(seq_id in available_seq_ids for seq_id in seq_ids)
    token_ids_available = bool(payload_list) and all(
        len(getattr(payload, "draft_token_ids", []) or []) == int(getattr(payload, "per_seq_length", 0) or 0)
        for payload in payload_list
    )
    tensor_available = token_ids_available
    missing_reason = None
    error_kind = None
    next_required_feature = None
    should_skip_non_owner = False
    if missing:
        missing_reason = "missing_seq_ids"
        error_kind = "mailbox_missing_payload"
        next_required_feature = "mailbox_payload_tensor_transport"
    elif not envelope_available:
        missing_reason = "missing_home_batch"
        error_kind = "mailbox_warmup_miss"
        next_required_feature = "pipeline_warmup_schedule"
    elif not available_for_seq_ids:
        missing_reason = "seq_ids_not_available_in_home_batch"
        error_kind = "mailbox_missing_payload"
        next_required_feature = "mailbox_payload_tensor_transport"
    elif not tensor_available:
        if not is_payload_owner:
            missing_reason = "mailbox_payload_tensor_unavailable_on_non_owner"
            error_kind = "mailbox_payload_tensor_unavailable_on_non_owner"
            should_skip_non_owner = True
        else:
            missing_reason = "mailbox_payload_tensor_backend_unavailable"
            error_kind = "mailbox_payload_tensor_backend_unavailable"
            next_required_feature = "mailbox_payload_tensor_backend"
    return {
        "mailbox_payload_envelope_available": envelope_available,
        "mailbox_payload_token_ids_available": token_ids_available,
        "mailbox_payload_tensor_available": tensor_available,
        "mailbox_payload_available_for_seq_ids": available_for_seq_ids,
        "mailbox_payload_local_to_rank": tensor_available,
        "mailbox_payload_owner_rank": owner_rank,
        "mailbox_payload_current_rank": current_rank,
        "mailbox_payload_missing_reason": missing_reason,
        "mailbox_error_kind": error_kind,
        "next_required_feature": next_required_feature,
        "should_skip_non_owner": should_skip_non_owner,
    }
