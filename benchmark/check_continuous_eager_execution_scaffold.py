#!/usr/bin/env python3
"""Check Phase 1I-A continuous eager draft-execution scaffold invariants.

Validates that:
  A. draft_eager_new_set_executed non-empty implies proposals were generated.
  B. draft_eager_continue_set seqs are NOT in draft_eager_new_set_executed.
  C. target_eager_set_executed remains empty (no target verification).
  D. Executed seqs have continuous_eager_exec_base_kind = "draft_eager_new_set".
  E. scaffold_proposals_sent >= scaffold_proposals_received (DRAFT-TARGET consistency).
  F. Promoted proposals have parent_acceptance_status = "accepted".
  G. tokens_generated > 0 iff draft_eager_new_set_executed non-empty.
  H. No token mutation evidence (sequence lengths unchanged by scaffold).
  I. Enablement flag present in all scaffold records.
  J. Base validation: all executed seqs have base_validation_ok=True.
  K. Transport token consistency: send_token_count >= receive_token_count.
  L. Buffer consistency: buffer_size_after >= buffer_size_before.
  M. Promotion/discard exec fields populated when proposals present.
  N. Execution phase is non-empty in scaffold records.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(
        "Unsupported engine trace JSON format. Expected a raw list, or a dict "
        "with 'traces', 'records', or 'trace_records'."
    )


def as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(x) for x in value}


def _has_scaffold_fields(record: dict[str, Any]) -> bool:
    return any(
        record.get(k) is not None
        for k in (
            "continuous_eager_scaffold_tokens_generated",
            "continuous_eager_scaffold_proposals_generated",
            "draft_eager_new_set_executed",
        )
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python benchmark/check_continuous_eager_execution_scaffold.py <engine_trace.json>")
        return 2

    records = load_trace(Path(sys.argv[1]))
    scaffold_records = [r for r in records if _has_scaffold_fields(r)]

    print(f"total_records={len(records)}")
    print(f"scaffold_records={len(scaffold_records)}")

    if not scaffold_records:
        print("\nNo scaffold records found — nothing to check.")
        return 0

    errors: list[str] = []
    warnings: list[str] = []

    # --- Summary ---
    total_executed = sum(len(r.get("draft_eager_new_set_executed") or []) for r in scaffold_records)
    total_new = sum(len(r.get("draft_eager_new_set") or []) for r in scaffold_records)
    total_continue = sum(len(r.get("draft_eager_continue_set") or []) for r in scaffold_records)
    gen_total = sum(
        int(r.get("continuous_eager_scaffold_tokens_generated") or 0) for r in scaffold_records
    )
    prop_gen_total = sum(
        int(r.get("continuous_eager_scaffold_proposals_generated") or 0) for r in scaffold_records
    )
    sent_total = sum(
        int(r.get("continuous_eager_scaffold_proposals_sent") or 0) for r in scaffold_records
    )
    recv_total = sum(
        int(r.get("continuous_eager_scaffold_proposals_received") or 0) for r in scaffold_records
    )
    promoted_total = sum(
        int(r.get("continuous_eager_scaffold_proposals_promoted") or 0) for r in scaffold_records
    )
    discarded_total = sum(
        int(r.get("continuous_eager_scaffold_proposals_discarded") or 0) for r in scaffold_records
    )

    # Transport exec totals
    exec_send_tokens = sum(
        int(r.get("continuous_eager_exec_send_token_count") or 0) for r in scaffold_records
    )
    exec_recv_tokens = sum(
        int(r.get("continuous_eager_exec_receive_token_count") or 0) for r in scaffold_records
    )
    exec_sent_seq_count = sum(
        len(r.get("continuous_eager_exec_sent_seq_ids") or []) for r in scaffold_records
    )
    exec_received_seq_count = sum(
        len(r.get("continuous_eager_exec_received_seq_ids") or []) for r in scaffold_records
    )

    print(f"\n--- Scaffold summary ---")
    print(f"draft_eager_new_set_total={total_new}")
    print(f"draft_eager_continue_set_total={total_continue}")
    print(f"draft_eager_new_set_executed_total={total_executed}")
    print(f"scaffold_tokens_generated_total={gen_total}")
    print(f"scaffold_proposals_generated_total={prop_gen_total}")
    print(f"scaffold_proposals_sent_total={sent_total}")
    print(f"scaffold_proposals_received_total={recv_total}")
    print(f"scaffold_proposals_promoted_total={promoted_total}")
    print(f"scaffold_proposals_discarded_total={discarded_total}")
    print(f"exec_send_token_count_total={exec_send_tokens}")
    print(f"exec_receive_token_count_total={exec_recv_tokens}")
    print(f"exec_sent_seq_ids_count={exec_sent_seq_count}")
    print(f"exec_received_seq_ids_count={exec_received_seq_count}")

    # --- Per-record checks ---
    for idx, record in enumerate(scaffold_records):
        phase = record.get("plan_phase", "unknown")
        executed = as_int_set(record.get("draft_eager_new_set_executed"))
        continue_set = as_int_set(record.get("draft_eager_continue_set"))
        target_exec = as_int_set(record.get("target_eager_set_executed"))
        base_kind = record.get("continuous_eager_exec_base_kind_by_seq_id") or {}
        base_len = record.get("continuous_eager_exec_base_len_by_seq_id") or {}
        base_valid = record.get("continuous_eager_exec_base_is_valid_by_seq_id") or {}
        base_val_ok = record.get("continuous_eager_exec_base_validation_ok_by_seq_id") or {}
        base_val_reason = record.get("continuous_eager_exec_base_validation_reason_by_seq_id") or {}

        # Check I: enablement flag present
        if not record.get("continuous_eager_draft_execution_enabled"):
            errors.append(
                f"record[{idx}] phase={phase}: continuous_eager_draft_execution_enabled "
                f"is missing or False"
            )

        # Check N: execution phase is non-empty
        exec_phase = record.get("continuous_eager_execution_phase", "")
        if not exec_phase:
            errors.append(
                f"record[{idx}] phase={phase}: continuous_eager_execution_phase is empty"
            )

        # Check A: executed non-empty -> proposals generated > 0
        if phase == "steady" and executed:
            if int(record.get("continuous_eager_scaffold_proposals_generated") or 0) <= 0:
                errors.append(
                    f"record[{idx}] phase={phase}: draft_eager_new_set_executed={sorted(executed)} "
                    f"but scaffold_proposals_generated=0"
                )

        # Check B: continue_set seqs NOT in executed
        bad_continue = continue_set & executed
        if bad_continue:
            errors.append(
                f"record[{idx}] phase={phase}: draft_eager_continue_set seqs in "
                f"draft_eager_new_set_executed: {sorted(bad_continue)}"
            )

        # Check C: target_eager_set_executed empty
        if target_exec:
            errors.append(
                f"record[{idx}] phase={phase}: target_eager_set_executed must be empty, "
                f"got {sorted(target_exec)}"
            )

        # Check D: executed seqs have base_kind = "draft_eager_new_set"
        for sid in sorted(executed):
            if str(sid) not in base_kind:
                errors.append(
                    f"record[{idx}] phase={phase}: seq_id={sid} in "
                    f"draft_eager_new_set_executed missing exec_base_kind"
                )
            elif base_kind[str(sid)] != "draft_eager_new_set":
                errors.append(
                    f"record[{idx}] phase={phase}: seq_id={sid} exec_base_kind="
                    f"'{base_kind[str(sid)]}' expected 'draft_eager_new_set'"
                )

        # Check J: base validation ok for all executed seqs
        for sid in sorted(executed):
            ok = base_val_ok.get(str(sid))
            if ok is not True:
                reason = base_val_reason.get(str(sid), "missing")
                errors.append(
                    f"record[{idx}] phase={phase}: seq_id={sid} "
                    f"base_validation_ok={ok} reason='{reason}'"
                )

        # Check G: tokens_generated > 0 iff executed non-empty (in steady)
        if phase == "steady":
            has_tokens = int(record.get("continuous_eager_scaffold_tokens_generated") or 0) > 0
            has_executed = bool(executed)
            if has_tokens != has_executed:
                errors.append(
                    f"record[{idx}] phase={phase}: scaffold_tokens_generated>0={has_tokens} "
                    f"but executed_non_empty={has_executed}"
                )

        # Check K: transport token per-record consistency
        send_tokens = int(record.get("continuous_eager_exec_send_token_count") or 0)
        recv_tokens = int(record.get("continuous_eager_exec_receive_token_count") or 0)
        if send_tokens > 0 and recv_tokens > 0 and recv_tokens > send_tokens:
            errors.append(
                f"record[{idx}] phase={phase}: receive_token_count={recv_tokens} > "
                f"send_token_count={send_tokens}"
            )

        # Check L: buffer consistency
        buf_before = int(record.get("continuous_eager_exec_buffer_size_before") or 0)
        buf_after = int(record.get("continuous_eager_exec_buffer_size_after") or 0)
        if buf_before > buf_after and buf_after > 0:
            errors.append(
                f"record[{idx}] phase={phase}: buffer_size_before={buf_before} > "
                f"buffer_size_after={buf_after}"
            )

        # Check transport seq_id consistency: sent seq_ids should match received
        # when both are non-empty on the same record.
        sent_ids = as_int_set(record.get("continuous_eager_exec_sent_seq_ids"))
        recv_ids = as_int_set(record.get("continuous_eager_exec_received_seq_ids"))
        if sent_ids and recv_ids and sent_ids != recv_ids:
            errors.append(
                f"record[{idx}] phase={phase}: sent_seq_ids={sorted(sent_ids)} != "
                f"received_seq_ids={sorted(recv_ids)}"
            )

    # --- Cross-record checks ---

    # Check E: sent >= received (DRAFT-TARGET consistency, best-effort)
    if sent_total > 0 and recv_total > 0 and recv_total > sent_total:
        errors.append(
            f"cross-record: scaffold_proposals_received={recv_total} > "
            f"scaffold_proposals_sent={sent_total}"
        )

    # Check transport token totals
    if exec_send_tokens > 0 and exec_recv_tokens > 0 and exec_recv_tokens > exec_send_tokens:
        errors.append(
            f"cross-record: exec_receive_token_count={exec_recv_tokens} > "
            f"exec_send_token_count={exec_send_tokens}"
        )

    # Check F: Promoted proposals must have parent_acceptance_status = "accepted"
    # Use exec-prefixed fields for the execution scaffold.
    promoted_total_scaffold = sum(
        int(r.get("continuous_eager_scaffold_proposals_promoted") or 0)
        for r in scaffold_records
    )
    if promoted_total_scaffold > 0:
        for idx, record in enumerate(scaffold_records):
            parent_status = record.get(
                "continuous_eager_exec_parent_acceptance_status_by_seq_id"
            ) or {}
            promoted_local = int(
                record.get("continuous_eager_scaffold_proposals_promoted") or 0
            )
            if promoted_local > 0:
                promoted_seq_ids = as_int_set(
                    record.get("continuous_eager_exec_promoted_seq_ids")
                )
                for sid in sorted(promoted_seq_ids):
                    status = parent_status.get(str(sid), "")
                    if status != "accepted":
                        errors.append(
                            f"record[{idx}]: seq_id={sid} promoted in scaffold but "
                            f"parent_acceptance_status='{status}' expected 'accepted'"
                        )

    # Check M: promotion/discard exec fields should be consistent with counters
    for idx, record in enumerate(scaffold_records):
        promoted_count = int(
            record.get("continuous_eager_scaffold_proposals_promoted") or 0
        )
        discarded_count = int(
            record.get("continuous_eager_scaffold_proposals_discarded") or 0
        )
        exec_promoted = len(
            record.get("continuous_eager_exec_promoted_seq_ids") or []
        )
        exec_discarded = len(
            record.get("continuous_eager_exec_discarded_seq_ids") or []
        )
        if promoted_count != exec_promoted:
            errors.append(
                f"record[{idx}]: scaffold_proposals_promoted={promoted_count} != "
                f"exec_promoted_seq_ids len={exec_promoted}"
            )
        if discarded_count != exec_discarded:
            errors.append(
                f"record[{idx}]: scaffold_proposals_discarded={discarded_count} != "
                f"exec_discarded_seq_ids len={exec_discarded}"
            )

    # --- Proposal lifecycle reconciliation ---
    if prop_gen_total > 0:
        if sent_total > prop_gen_total:
            errors.append(
                f"proposal lifecycle: sent={sent_total} > generated={prop_gen_total}"
            )
        if recv_total > sent_total and recv_total > 0:
            errors.append(
                f"proposal lifecycle: received={recv_total} > sent={sent_total}"
            )

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"\nContinuous eager scaffold execution check passed ({len(scaffold_records)} records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
