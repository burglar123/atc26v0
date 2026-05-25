#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]


def load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("traces", "records", "trace_records"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    raise ValueError(
        "Unsupported trace format. Expected a raw list or a dict with "
        "'traces', 'records', or 'trace_records'."
    )


def is_dual_record(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    )


def as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(item) for item in value}


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def dict_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def step_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int_value(record.get("eager_result_transfer_plan_id"), -1),
        int_value(record.get("eager_result_transfer_step_id"), -1),
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_result_transfer_enabled = 0
    sent_by_step: dict[tuple[int, int], set[int]] = defaultdict(set)
    received_by_step: dict[tuple[int, int], set[int]] = defaultdict(set)
    sent_num_by_step: dict[tuple[int, int], int] = {}
    received_num_by_step: dict[tuple[int, int], int] = {}
    sent_payload_by_step: dict[tuple[int, int], int] = {}
    received_payload_by_step: dict[tuple[int, int], int] = {}
    zero_steps: set[tuple[int, int]] = set()
    accepted_len_distribution: Counter[int] = Counter()
    full_accept_count = 0
    reject_partial_count = 0
    draft_mutation_detected_count = 0
    draft_checkpoint_failure_count = 0
    actual_eager_verified_counter_rows = 0
    real_target_eager_non_empty_count = 0
    validated_result_ids: set[int] = set()
    invalid_result_ids: set[int] = set()
    sent_result_ids: set[int] = set()
    received_result_ids: set[int] = set()
    validation_reason_counts: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        result_enabled = bool(record.get("enable_eager_result_transfer_dry_run", False))
        result_active = bool(record.get("eager_result_transfer_dry_run_enabled", False))
        apply_enabled = bool(record.get("enable_eager_apply_dry_run", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        sent_ids = as_int_set(record.get("eager_result_sent_proposal_ids"))
        received_ids = as_int_set(record.get("eager_result_received_proposal_ids"))
        received_seq_ids = as_int_list(record.get("eager_result_received_seq_ids"))
        validated_ids = as_int_set(record.get("eager_result_validated_proposal_ids"))
        invalid_ids = as_int_set(record.get("eager_result_invalid_proposal_ids"))
        reasons = record.get("eager_result_validation_reason_by_proposal_id", {})
        accepted_len_by_seq = record.get("eager_result_received_accepted_len_by_seq_id", {})
        full_accept_by_seq = record.get("eager_result_received_full_accept_by_seq_id", {})
        invalidated_len_by_seq = record.get("eager_result_received_invalidated_len_by_seq_id", {})
        proposal_len_by_id = record.get("eager_result_received_proposal_len_by_proposal_id", {})
        to_verify_len_by_id = record.get("eager_result_received_to_verify_len_by_proposal_id", {})
        checkpoint_ok_by_seq = record.get("eager_result_draft_checkpoint_ok_by_seq_id", {})
        mutation_by_seq = record.get("eager_result_draft_mutation_detected_by_seq_id", {})
        gamma = int_value(record.get("normal_gamma"), None)

        if target_eager:
            real_target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_actual = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_verified_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must stay zero: {nonzero_actual}")

        if not result_enabled:
            populated = (
                result_active
                or sent_ids
                or received_ids
                or int_value(record.get("eager_tokens_result_transfer_sent"), 0)
                or int_value(record.get("eager_tokens_result_transfer_received"), 0)
                or int_value(record.get("eager_tokens_result_transfer_validated"), 0)
                or int_value(record.get("eager_tokens_result_transfer_invalid"), 0)
            )
            if populated:
                errors.append(f"record[{idx}] eager result transfer fields populated while disabled")
            continue

        records_with_result_transfer_enabled += 1
        if not apply_enabled:
            errors.append(f"record[{idx}] result transfer dry-run must imply apply dry-run")
        if not result_active and not (sent_ids or received_ids):
            continue

        key = step_key(record)
        runner_role = str(record.get("runner_role", ""))
        sent_tokens = int_value(record.get("eager_tokens_result_transfer_sent"), 0)
        received_tokens = int_value(record.get("eager_tokens_result_transfer_received"), 0)
        is_sender_row = bool(sent_ids or sent_tokens or ("verify" in runner_role and not received_ids and not received_tokens))
        is_receiver_row = bool(received_ids or received_tokens or ("draft" in runner_role and not sent_ids and not sent_tokens))
        if is_sender_row:
            sent_by_step[key].update(sent_ids)
            sent_num_by_step[key] = int_value(record.get("eager_result_transfer_num_results"), 0)
            sent_payload_by_step[key] = int_value(record.get("eager_result_transfer_payload_len"), 0)
        if is_receiver_row:
            received_by_step[key].update(received_ids)
            received_num_by_step[key] = int_value(record.get("eager_result_received_num_results"), 0)
            received_payload_by_step[key] = int_value(record.get("eager_result_received_payload_len"), 0)
        if bool(record.get("eager_result_zero_result_step", False)):
            zero_steps.add(key)
            sent_by_step.setdefault(key, set())
            received_by_step.setdefault(key, set())

        sent_result_ids.update(sent_ids)
        received_result_ids.update(received_ids)
        validated_result_ids.update(validated_ids)
        invalid_result_ids.update(invalid_ids)

        classified = validated_ids | invalid_ids
        if received_ids and classified != received_ids:
            errors.append(
                f"record[{idx}] validated + invalid must equal received ids: "
                f"classified={sorted(classified)}, received={sorted(received_ids)}"
            )
        if validated_ids & invalid_ids:
            errors.append(f"record[{idx}] result ids both validated and invalid: {sorted(validated_ids & invalid_ids)}")

        proposal_to_seq = {
            proposal_id: seq_id
            for proposal_id, seq_id in zip(as_int_list(record.get("eager_result_received_proposal_ids")), received_seq_ids)
        }
        for proposal_id in sorted(received_ids):
            seq_id = proposal_to_seq.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] received proposal_id={proposal_id} missing seq mapping")
                continue
            accepted_len = int_value(dict_get(accepted_len_by_seq, seq_id), -1)
            full_accept = bool(dict_get(full_accept_by_seq, seq_id, False))
            invalidated_len = int_value(dict_get(invalidated_len_by_seq, seq_id), -1)
            proposal_len = int_value(dict_get(proposal_len_by_id, proposal_id), -1)
            to_verify_len = int_value(dict_get(to_verify_len_by_id, proposal_id), -1)
            if gamma is not None and not (0 <= accepted_len <= gamma):
                errors.append(f"record[{idx}] accepted_len out of range for seq_id={seq_id}: {accepted_len}")
            if gamma is not None and full_accept != (accepted_len == gamma):
                errors.append(f"record[{idx}] full_accept mismatch for seq_id={seq_id}")
            if gamma is not None and invalidated_len != gamma - accepted_len:
                errors.append(f"record[{idx}] invalidated_len mismatch for seq_id={seq_id}")
            if gamma is not None and proposal_len != gamma:
                errors.append(f"record[{idx}] proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and to_verify_len != gamma:
                errors.append(f"record[{idx}] proposal_id={proposal_id} to_verify_len != gamma")
            accepted_len_distribution[accepted_len] += 1
            if full_accept:
                full_accept_count += 1
            else:
                reject_partial_count += 1

        for proposal_id in sorted(validated_ids | invalid_ids):
            reason = dict_get(reasons, proposal_id)
            validation_reason_counts[str(reason)] += 1
            if proposal_id in validated_ids and reason != "ok":
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} reason must be ok, got {reason!r}")
            if proposal_id in invalid_ids and reason in (None, "", "ok"):
                errors.append(f"record[{idx}] invalid proposal_id={proposal_id} missing invalid reason")

        for seq_id, checkpoint_ok in checkpoint_ok_by_seq.items():
            if checkpoint_ok is not True:
                draft_checkpoint_failure_count += 1
                errors.append(f"record[{idx}] draft checkpoint failed for seq_id={seq_id}")
        for seq_id, mutation in mutation_by_seq.items():
            if mutation is True:
                draft_mutation_detected_count += 1
                errors.append(f"record[{idx}] draft mutation detected for seq_id={seq_id}")

    bad_steps = 0
    all_steps = set(sent_by_step) | set(received_by_step)
    for key in sorted(all_steps):
        sent = sent_by_step.get(key, set())
        received = received_by_step.get(key, set())
        if sent != received:
            bad_steps += 1
            errors.append(f"step{key} sent/received result ids mismatch: sent={sorted(sent)}, received={sorted(received)}")
        sent_num = sent_num_by_step.get(key)
        received_num = received_num_by_step.get(key)
        if sent_num is not None and received_num is not None and sent_num != received_num:
            bad_steps += 1
            errors.append(f"step{key} num_results mismatch: sent={sent_num}, received={received_num}")
        sent_payload = sent_payload_by_step.get(key)
        received_payload = received_payload_by_step.get(key)
        if sent_payload is not None and received_payload is not None and sent_payload != received_payload:
            bad_steps += 1
            errors.append(f"step{key} payload_len mismatch: sent={sent_payload}, received={received_payload}")

    summary = {
        "total_trace_records": len(records),
        "records_with_result_transfer_enabled": records_with_result_transfer_enabled,
        "result_transfer_steps": len(all_steps),
        "zero_result_transfer_steps": len(zero_steps),
        "sent_result_count": len(sent_result_ids),
        "received_result_count": len(received_result_ids),
        "validated_result_count": len(validated_result_ids),
        "invalid_result_count": len(invalid_result_ids),
        "bad_sent_received_step_count": bad_steps,
        "accepted_len_distribution": dict(accepted_len_distribution),
        "full_accept_count": full_accept_count,
        "reject_partial_count": reject_partial_count,
        "draft_mutation_detected_count": draft_mutation_detected_count,
        "draft_checkpoint_failure_count": draft_checkpoint_failure_count,
        "actual_eager_verified_counter_rows": actual_eager_verified_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_non_empty_count,
        "validation_reason_counts": dict(validation_reason_counts),
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_result_transfer_enabled",
        "result_transfer_steps",
        "zero_result_transfer_steps",
        "sent_result_count",
        "received_result_count",
        "validated_result_count",
        "invalid_result_count",
        "bad_sent_received_step_count",
        "accepted_len_distribution",
        "full_accept_count",
        "reject_partial_count",
        "draft_mutation_detected_count",
        "draft_checkpoint_failure_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "validation_reason_counts",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "enable_eager_transfer_dry_run": False,
        "enable_eager_schedule_dry_run": False,
        "enable_eager_verify_dry_run": False,
        "enable_eager_apply_dry_run": False,
        "enable_eager_result_transfer_dry_run": False,
        "eager_result_transfer_dry_run_enabled": False,
        "normal_gamma": 4,
        "target_eager_set": [],
        "eager_tokens_result_transfer_sent": 0,
        "eager_tokens_result_transfer_received": 0,
        "eager_tokens_result_transfer_validated": 0,
        "eager_tokens_result_transfer_invalid": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_sender_record(num_results: int = 1, accepted_len: int = 4, proposal_id: int = 101) -> dict[str, Any]:
    record = synthetic_base_record()
    full_accept = accepted_len == 4
    sent_ids = [proposal_id] if num_results else []
    seq_ids = [3] if num_results else []
    payload_len = 22 * num_results
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "enable_eager_draft_dry_run": True,
            "enable_eager_promotion_dry_run": True,
            "enable_eager_transfer_dry_run": True,
            "enable_eager_schedule_dry_run": True,
            "enable_eager_verify_dry_run": True,
            "enable_eager_apply_dry_run": True,
            "enable_eager_result_transfer_dry_run": True,
            "eager_result_transfer_dry_run_enabled": True,
            "runner_role": "dual_verify",
            "eager_result_transfer_step_id": 7,
            "eager_result_transfer_plan_id": 11,
            "eager_result_transfer_num_results": num_results,
            "eager_result_transfer_payload_len": payload_len,
            "eager_result_sent_proposal_ids": sent_ids,
            "eager_result_sent_seq_ids": seq_ids,
            "eager_result_sent_accepted_len_by_seq_id": {"3": accepted_len} if num_results else {},
            "eager_result_sent_full_accept_by_seq_id": {"3": full_accept} if num_results else {},
            "eager_result_sent_reject_position_by_seq_id": {"3": -1 if full_accept else accepted_len} if num_results else {},
            "eager_result_sent_invalidated_len_by_seq_id": {"3": 4 - accepted_len} if num_results else {},
            "eager_result_sent_revised_token_by_seq_id": {"3": -1 if full_accept else 200} if num_results else {},
            "eager_result_sent_apply_action_by_seq_id": {
                "3": "append_full_accept_then_rollback" if full_accept else "discard_partial_no_mutation"
            } if num_results else {},
            "eager_result_sent_proposal_len_by_proposal_id": {str(proposal_id): 4} if num_results else {},
            "eager_result_sent_to_verify_len_by_proposal_id": {str(proposal_id): 4} if num_results else {},
            "eager_tokens_result_transfer_sent": 4 if num_results else 0,
            "eager_result_zero_result_step": num_results == 0,
        }
    )
    return record


def synthetic_receiver_record(num_results: int = 1, accepted_len: int = 4, proposal_id: int = 101) -> dict[str, Any]:
    record = synthetic_sender_record(num_results, accepted_len, proposal_id)
    full_accept = accepted_len == 4
    received_ids = [proposal_id] if num_results else []
    seq_ids = [3] if num_results else []
    payload_len = 22 * num_results
    record.update(
        {
            "runner_role": "dual_draft",
            "eager_result_transfer_num_results": 0,
            "eager_result_transfer_payload_len": 0,
            "eager_result_sent_proposal_ids": [],
            "eager_result_sent_seq_ids": [],
            "eager_tokens_result_transfer_sent": 0,
            "eager_result_received_num_results": num_results,
            "eager_result_received_payload_len": payload_len,
            "eager_result_received_proposal_ids": received_ids,
            "eager_result_received_seq_ids": seq_ids,
            "eager_result_validated_proposal_ids": received_ids,
            "eager_result_invalid_proposal_ids": [],
            "eager_result_validation_reason_by_proposal_id": {str(proposal_id): "ok"} if num_results else {},
            "eager_result_received_accepted_len_by_seq_id": {"3": accepted_len} if num_results else {},
            "eager_result_received_full_accept_by_seq_id": {"3": full_accept} if num_results else {},
            "eager_result_received_reject_position_by_seq_id": {"3": -1 if full_accept else accepted_len} if num_results else {},
            "eager_result_received_invalidated_len_by_seq_id": {"3": 4 - accepted_len} if num_results else {},
            "eager_result_received_revised_token_by_seq_id": {"3": -1 if full_accept else 200} if num_results else {},
            "eager_result_received_proposal_len_by_proposal_id": {str(proposal_id): 4} if num_results else {},
            "eager_result_received_to_verify_len_by_proposal_id": {str(proposal_id): 4} if num_results else {},
            "eager_result_draft_current_len_by_seq_id": {"3": 12} if num_results else {},
            "eager_result_base_len_by_seq_id": {"3": 12} if num_results else {},
            "eager_result_draft_len_matches_base_by_seq_id": {"3": True} if num_results else {},
            "eager_result_draft_seq_pre_verify_by_seq_id": {"3": False} if num_results else {},
            "eager_result_draft_status_before_by_seq_id": {"3": "RUNNING"} if num_results else {},
            "eager_result_draft_status_after_by_seq_id": {"3": "RUNNING"} if num_results else {},
            "eager_result_draft_checkpoint_ok_by_seq_id": {"3": True} if num_results else {},
            "eager_result_draft_mutation_detected_by_seq_id": {"3": False} if num_results else {},
            "eager_tokens_result_transfer_received": 4 if num_results else 0,
            "eager_tokens_result_transfer_validated": 4 if num_results else 0,
            "eager_tokens_result_transfer_invalid": 0,
            "eager_result_zero_result_step": num_results == 0,
        }
    )
    return record


def synthetic_invalid_receiver_record(reason: str = "seq_not_found") -> dict[str, Any]:
    record = synthetic_receiver_record(1, 2, 102)
    record.update(
        {
            "eager_result_validated_proposal_ids": [],
            "eager_result_invalid_proposal_ids": [102],
            "eager_result_validation_reason_by_proposal_id": {"102": reason},
            "eager_tokens_result_transfer_validated": 0,
            "eager_tokens_result_transfer_invalid": 4,
        }
    )
    if reason == "seq_not_found":
        record["eager_result_draft_checkpoint_ok_by_seq_id"] = {}
        record["eager_result_draft_mutation_detected_by_seq_id"] = {}
    return record


def run_synthetic_tests() -> None:
    zero_sender = synthetic_sender_record(0, 4, 201)
    zero_receiver = synthetic_receiver_record(0, 4, 201)
    zero_sender["eager_result_transfer_plan_id"] = 12
    zero_sender["eager_result_transfer_step_id"] = 8
    zero_receiver["eager_result_transfer_plan_id"] = 12
    zero_receiver["eager_result_transfer_step_id"] = 8
    valid_records = [
        synthetic_base_record(),
        synthetic_sender_record(1, 4, 101),
        synthetic_receiver_record(1, 4, 101),
        synthetic_sender_record(1, 2, 102),
        synthetic_invalid_receiver_record("seq_not_found"),
        zero_sender,
        zero_receiver,
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager result transfer records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_received_accepted_len_by_seq_id"] = {"3": 5}
    errors, _ = validate_records(invalid)
    assert any("accepted_len out of range" in error for error in errors), "checker missed accepted_len range"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_received_full_accept_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("full_accept mismatch" in error for error in errors), "checker missed full_accept mismatch"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_draft_mutation_detected_by_seq_id"] = {"3": True}
    errors, _ = validate_records(invalid)
    assert any("draft mutation detected" in error for error in errors), "checker missed draft mutation"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_draft_checkpoint_ok_by_seq_id"] = {"3": False}
    errors, _ = validate_records(invalid)
    assert any("draft checkpoint failed" in error for error in errors), "checker missed checkpoint failure"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[2]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[2]["eager_result_received_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("sent/received result ids mismatch" in error for error in errors), (
        "checker missed sent/received mismatch"
    )

    print("Synthetic eager result transfer dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5c eager result transfer dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        if args.trace is None:
            return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nEager result transfer dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
