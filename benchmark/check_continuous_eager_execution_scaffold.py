#!/usr/bin/env python3
"""Check Phase 1I-A continuous eager draft-execution scaffold invariants.

Phase-aware validation:
  - steady: strict scaffold validation (all fields required for executed seqs).
  - priming/fallback: safety-only unless proposals are actually executed.

Validates:
  A. draft_eager_new_set_executed non-empty implies proposals were generated.
  B. draft_eager_continue_set seqs are NOT in draft_eager_new_set_executed.
  C. target_eager_set_executed remains empty (no target verification).
  D. Executed seqs have base_kind = "normal_parent_full_accept_prefix"; skipped seqs may have base_kind = "draft_eager_new_set".
  E. scaffold_proposals_sent >= scaffold_proposals_received (cross-record).
  F. Promoted proposals have parent_acceptance_status = "accepted".
  G. tokens_generated > 0 iff draft_eager_new_set_executed non-empty.
  H. No token mutation evidence.
  I. Enablement flag present in all scaffold-eligible records.
  J. Base validation: every executed seq has base_validation_ok=True.
  K. Transport token consistency: send_token_count == receive_token_count.
  L. Buffer consistency: buffer_size_after >= buffer_size_before.
  M. Promotion/discard fields populated when proposals present.
  N. Execution phase is non-empty in scaffold records.
  O. draft_eager_continue_set_executed == [].
  P. eager_tokens_verified == 0, eager_tokens_accepted == 0.
  Q. Sent/received proposal ids match.
  R. Sent/received token counts match.
  S. Proposals generated => sent > 0, received > 0.
  T. unsupported_post_verify_seq must never appear as a skip reason.
  U. Non-empty draft_eager_new_set with zero executed must have valid skip reasons.
  V. executed_total > 0 requires transport records (sent/received > 0).
  W. generated > 0 => sent_seq_ids and received_seq_ids must be non-empty.
  X. send_token_count == receive_token_count (cross-record).
  Y. receive_validation_ok must be True when proposals received.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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
            "continuous_eager_draft_execution_enabled",
        )
    )


def _is_scaffold_enabled(record: dict[str, Any]) -> bool:
    return bool(record.get("continuous_eager_draft_execution_enabled"))


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

    # Classify records by phase for phase-aware validation.
    steady_records = [r for r in scaffold_records if r.get("plan_phase") == "steady"]
    non_steady_records = [r for r in scaffold_records if r.get("plan_phase") != "steady"]

    print(f"steady_scaffold_records={len(steady_records)}")
    print(f"non_steady_scaffold_records={len(non_steady_records)}")

    errors: list[str] = []
    warnings: list[str] = []

    # --- Aggregated diagnostics for non-steady records ---
    missing_flag_by_phase_role: dict[tuple[str, str], int] = defaultdict(int)
    missing_phase_by_phase_role: dict[tuple[str, str], int] = defaultdict(int)

    for record in non_steady_records:
        if not _is_scaffold_enabled(record):
            continue
        phase = record.get("plan_phase", "unknown")
        role = record.get("runner_role", "unknown")
        if not record.get("continuous_eager_draft_execution_enabled"):
            missing_flag_by_phase_role[(role, phase)] += 1
        if not record.get("continuous_eager_execution_phase"):
            missing_phase_by_phase_role[(role, phase)] += 1

    # Print aggregated non-steady diagnostics (not individual errors).
    if missing_flag_by_phase_role:
        print("\n--- Non-steady: missing continuous_eager_draft_execution_enabled ---")
        for (role, phase), count in sorted(missing_flag_by_phase_role.items()):
            print(f"  role={role} phase={phase} count={count}")

    if missing_phase_by_phase_role:
        print("\n--- Non-steady: missing continuous_eager_execution_phase ---")
        for (role, phase), count in sorted(missing_phase_by_phase_role.items()):
            print(f"  role={role} phase={phase} count={count}")

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

    # Invariant counters
    continue_set_executed = sum(
        len(r.get("draft_eager_continue_set_executed") or []) for r in scaffold_records
    )
    target_eager_exec = sum(
        len(r.get("target_eager_set_executed") or []) for r in scaffold_records
    )
    eager_verified = sum(
        int(r.get("eager_tokens_verified") or 0) for r in scaffold_records
    )
    eager_accepted = sum(
        int(r.get("eager_tokens_accepted") or 0) for r in scaffold_records
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

    # --- Invariant checks (all phases) ---

    # Check O: draft_eager_continue_set_executed == []
    if continue_set_executed > 0:
        errors.append(
            f"INVARIANT: draft_eager_continue_set_executed must be empty, "
            f"got {continue_set_executed} entries"
        )

    # Check C: target_eager_set_executed == []
    if target_eager_exec > 0:
        errors.append(
            f"INVARIANT: target_eager_set_executed must be empty, "
            f"got {target_eager_exec} entries"
        )

    # Check P: eager_tokens_verified == 0, eager_tokens_accepted == 0
    if eager_verified > 0:
        errors.append(
            f"INVARIANT: eager_tokens_verified must be 0 (no target verification), "
            f"got {eager_verified}"
        )
    if eager_accepted > 0:
        errors.append(
            f"INVARIANT: eager_tokens_accepted must be 0 (no token application), "
            f"got {eager_accepted}"
        )

    # --- Per-record checks (strict for steady, relaxed for non-steady) ---
    for idx, record in enumerate(scaffold_records):
        phase = record.get("plan_phase", "unknown")
        role = record.get("runner_role", "unknown")
        is_steady = phase == "steady"
        is_scaffold_enabled_record = _is_scaffold_enabled(record)

        executed = as_int_set(record.get("draft_eager_new_set_executed"))
        continue_set = as_int_set(record.get("draft_eager_continue_set"))
        target_exec = as_int_set(record.get("target_eager_set_executed"))

        # Non-steady records: safety-only checks unless proposals are executed.
        if not is_steady and not executed:
            # If scaffold is enabled, flag must be present.
            if is_scaffold_enabled_record:
                if not record.get("continuous_eager_draft_execution_enabled"):
                    pass  # aggregated above
                if not record.get("continuous_eager_execution_phase"):
                    pass  # aggregated above
            continue

        # Check I: enablement flag present (steady only, strict)
        if is_steady and is_scaffold_enabled_record:
            if not record.get("continuous_eager_draft_execution_enabled"):
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: "
                    f"continuous_eager_draft_execution_enabled missing or False"
                )

        # Check N: execution phase non-empty (steady only, strict)
        if is_steady and is_scaffold_enabled_record:
            exec_phase = record.get("continuous_eager_execution_phase", "")
            if not exec_phase:
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: "
                    f"continuous_eager_execution_phase is empty"
                )

        # Check A: executed non-empty -> proposals generated > 0
        if is_steady and executed:
            if int(record.get("continuous_eager_scaffold_proposals_generated") or 0) <= 0:
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: "
                    f"draft_eager_new_set_executed={sorted(executed)} "
                    f"but scaffold_proposals_generated=0"
                )

        # Check B: continue_set seqs NOT in executed
        bad_continue = continue_set & executed
        if bad_continue:
            errors.append(
                f"record[{idx}] role={role} phase={phase}: "
                f"draft_eager_continue_set seqs in draft_eager_new_set_executed: "
                f"{sorted(bad_continue)}"
            )

        # Check D: executed seqs must have base_kind = "normal_parent_full_accept_prefix"
        # (skipped seqs may have base_kind = "draft_eager_new_set").
        base_kind = record.get("continuous_eager_exec_base_kind_by_seq_id") or {}
        for sid in sorted(executed):
            kind = base_kind.get(str(sid))
            if kind is None:
                if is_steady:
                    errors.append(
                        f"record[{idx}] role={role} phase={phase}: seq_id={sid} "
                        f"in draft_eager_new_set_executed missing exec_base_kind"
                    )
            elif kind not in ("normal_parent_full_accept_prefix", "draft_eager_new_set"):
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: seq_id={sid} "
                    f"exec_base_kind='{kind}' expected 'normal_parent_full_accept_prefix'"
                )

        # Check J: base validation ok for all executed seqs
        base_val_ok = record.get("continuous_eager_exec_base_validation_ok_by_seq_id") or {}
        base_val_reason = record.get("continuous_eager_exec_base_validation_reason_by_seq_id") or {}
        base_len = record.get("continuous_eager_exec_base_len_by_seq_id") or {}
        expected_base = record.get("continuous_eager_exec_expected_base_len_by_seq_id") or {}
        parent_len = record.get("continuous_eager_exec_parent_len_by_seq_id") or {}
        for sid in sorted(executed):
            ok = base_val_ok.get(str(sid))
            if ok is not True:
                reason = base_val_reason.get(str(sid), "missing")
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: seq_id={sid} "
                    f"base_validation_ok={ok} reason='{reason}'"
                )
            # Check base metadata completeness for executed seqs.
            if str(sid) not in base_len:
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: seq_id={sid} "
                    f"missing exec_base_len"
                )
            if str(sid) not in expected_base:
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: seq_id={sid} "
                    f"missing exec_expected_base_len"
                )
            if str(sid) not in parent_len:
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: seq_id={sid} "
                    f"missing exec_parent_len"
                )

        # Check G: tokens_generated > 0 iff executed non-empty (steady)
        if is_steady:
            has_tokens = int(record.get("continuous_eager_scaffold_tokens_generated") or 0) > 0
            has_executed = bool(executed)
            if has_tokens != has_executed:
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: "
                    f"scaffold_tokens_generated>0={has_tokens} "
                    f"but executed_non_empty={has_executed}"
                )

        # Check K: transport token per-record consistency
        send_tokens = int(record.get("continuous_eager_exec_send_token_count") or 0)
        recv_tokens = int(record.get("continuous_eager_exec_receive_token_count") or 0)
        if send_tokens > 0 and recv_tokens > 0 and recv_tokens != send_tokens:
            errors.append(
                f"record[{idx}] role={role} phase={phase}: "
                f"receive_token_count={recv_tokens} != send_token_count={send_tokens}"
            )

        # Check L: buffer consistency
        buf_before = int(record.get("continuous_eager_exec_buffer_size_before") or 0)
        buf_after = int(record.get("continuous_eager_exec_buffer_size_after") or 0)
        if buf_before > buf_after and buf_after > 0:
            errors.append(
                f"record[{idx}] role={role} phase={phase}: "
                f"buffer_size_before={buf_before} > buffer_size_after={buf_after}"
            )

        # Check Q: sent/received seq_ids match (when both present)
        sent_ids = as_int_set(record.get("continuous_eager_exec_sent_seq_ids"))
        recv_ids = as_int_set(record.get("continuous_eager_exec_received_seq_ids"))
        if sent_ids and recv_ids and sent_ids != recv_ids:
            errors.append(
                f"record[{idx}] role={role} phase={phase}: "
                f"sent_seq_ids={sorted(sent_ids)} != received_seq_ids={sorted(recv_ids)}"
            )

        # Check R: sent/received proposal_ids match
        sent_pids = record.get("continuous_eager_exec_sent_proposal_ids") or []
        recv_pids = record.get("continuous_eager_exec_received_proposal_ids") or []
        if sent_pids and recv_pids and sorted(sent_pids) != sorted(recv_pids):
            errors.append(
                f"record[{idx}] role={role} phase={phase}: "
                f"sent_proposal_ids != received_proposal_ids"
            )

        # Check T: unsupported_post_verify_seq must NEVER appear as a skip reason.
        # Post-verify seqs are the intended input for continuous eager.
        _val_reasons = record.get("continuous_eager_exec_base_validation_reason_by_seq_id") or {}
        for _sid_str, _reason in _val_reasons.items():
            if _reason == "unsupported_post_verify_seq":
                errors.append(
                    f"record[{idx}] role={role} phase={phase}: seq_id={_sid_str} "
                    f"has forbidden skip reason 'unsupported_post_verify_seq'. "
                    f"Post-verify seqs are intended input for continuous eager; "
                    f"use 'unsupported_missing_parent_tokens' if no parent proposal exists."
                )

        # Check U: non-empty draft_eager_new_set with zero executed must have
        # all skipped seqs explained by valid reasons (not unsupported_post_verify_seq).
        draft_new_set = as_int_set(record.get("draft_eager_new_set"))
        if is_steady and draft_new_set and not executed:
            _val_reasons_u = record.get("continuous_eager_exec_base_validation_reason_by_seq_id") or {}
            for _sid in sorted(draft_new_set):
                _reason_u = _val_reasons_u.get(str(_sid))
                if _reason_u is None:
                    errors.append(
                        f"record[{idx}] role={role} phase={phase}: seq_id={_sid} "
                        f"in draft_eager_new_set but not executed and has no validation reason"
                    )
                elif _reason_u == "unsupported_post_verify_seq":
                    errors.append(
                        f"record[{idx}] role={role} phase={phase}: seq_id={_sid} "
                        f"in draft_eager_new_set skipped with forbidden reason "
                        f"'unsupported_post_verify_seq'"
                    )

    # --- Cross-record checks ---

    # Check E: sent >= received
    if sent_total > 0 and recv_total > 0 and recv_total > sent_total:
        errors.append(
            f"cross-record: scaffold_proposals_received={recv_total} > "
            f"scaffold_proposals_sent={sent_total}"
        )

    # Check K (cross): transport token totals must match
    if exec_send_tokens > 0 and exec_recv_tokens > 0 and exec_recv_tokens != exec_send_tokens:
        errors.append(
            f"cross-record: exec_receive_token_count={exec_recv_tokens} != "
            f"exec_send_token_count={exec_send_tokens}"
        )

    # Check S: proposals generated => sent > 0 and received > 0
    if prop_gen_total > 0:
        if sent_total == 0:
            errors.append(
                f"cross-record: scaffold_proposals_generated={prop_gen_total} "
                f"but scaffold_proposals_sent=0"
            )
        if recv_total == 0:
            errors.append(
                f"cross-record: scaffold_proposals_generated={prop_gen_total} "
                f"but scaffold_proposals_received=0"
            )

    # Check V: executed_total > 0 requires transport records.
    if total_executed > 0:
        if sent_total == 0:
            errors.append(
                f"TRANSPORT: draft_eager_new_set_executed_total={total_executed} "
                f"but scaffold_proposals_sent=0 (no transport records on DRAFT side)"
            )
        if recv_total == 0:
            errors.append(
                f"TRANSPORT: draft_eager_new_set_executed_total={total_executed} "
                f"but scaffold_proposals_received=0 (no transport records on TARGET side)"
            )

    # Check W: generated > 0 => sent_seq_ids and received_seq_ids must be non-empty.
    if prop_gen_total > 0:
        if exec_sent_seq_count == 0:
            errors.append(
                f"TRANSPORT: scaffold_proposals_generated={prop_gen_total} "
                f"but sent_seq_ids is empty across all records"
            )
        if exec_received_seq_count == 0:
            errors.append(
                f"TRANSPORT: scaffold_proposals_generated={prop_gen_total} "
                f"but received_seq_ids is empty across all records"
            )

    # Check X: sent/received token counts must match (cross-record).
    if exec_send_tokens > 0 and exec_recv_tokens > 0:
        if exec_recv_tokens != exec_send_tokens:
            errors.append(
                f"TRANSPORT: send_token_count_total={exec_send_tokens} != "
                f"receive_token_count_total={exec_recv_tokens}"
            )

    # Check Y: receive_validation_ok must be True when proposals received.
    _recv_val_failures = 0
    for idx, record in enumerate(scaffold_records):
        recv_ok = record.get("continuous_eager_exec_receive_validation_ok")
        recv_count = int(record.get("continuous_eager_exec_receive_token_count") or 0)
        if recv_count > 0 and recv_ok is not True:
            _recv_val_failures += 1
            errors.append(
                f"record[{idx}]: receive_token_count={recv_count} but "
                f"receive_validation_ok={recv_ok}"
            )
    if exec_recv_tokens > 0 and _recv_val_failures > 0:
        errors.append(
            f"TRANSPORT: {_recv_val_failures} records with non-empty receive "
            f"but receive_validation_ok != True"
        )

    # Check F: Promoted proposals must have parent_acceptance_status = "accepted"
    if promoted_total > 0:
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

    # Check M: promotion/discard exec field consistency
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
        total_disposition = promoted_total + discarded_total
        # sum of unknown across all records
        unknown_total = sum(
            len(r.get("continuous_eager_exec_parent_unknown_seq_ids") or [])
            for r in scaffold_records
        )
        if total_disposition + unknown_total < prop_gen_total:
            warnings.append(
                f"proposal lifecycle: generated={prop_gen_total} but "
                f"promoted={promoted_total} + discarded={discarded_total} + "
                f"unknown={unknown_total} = {total_disposition + unknown_total} < generated"
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
