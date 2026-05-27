#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


TAKEOVER_SOURCE = "phase1h5e3_takeover_lane"
FULL_ACCEPT_ACTION = "append_full_accept_then_rollback"
ALWAYS_ZERO_COUNTER_FIELDS = [
    "eager_tokens_verified",
    "eager_tokens_accepted",
    "eager_tokens_rejected",
    "eager_tokens_invalidated",
]
VALID_NOT_READY_REASONS = {
    "not_full_accept",
    "verify_not_executed",
    "verify_mutation_detected",
    "verify_checkpoint_failed",
    "apply_not_executed",
    "apply_action_not_full_accept",
    "apply_rollback_failed",
    "apply_mutation_detected",
    "apply_checkpoint_failed",
    "result_not_transferred",
    "result_not_validated",
    "result_invalid",
    "result_duplicate",
    "result_metadata_mismatch",
    "sync_apply_not_executed",
    "sync_apply_inconsistent",
    "sync_apply_rollback_failed",
    "sync_apply_mutation_detected",
    "missing_local_proposal",
    "proposal_stale",
    "proposal_expired",
    "proposal_invalidated",
    "seq_not_running",
    "seq_finished",
    "seq_pre_verify",
    "base_len_mismatch",
    "frontier_mismatch",
    "token_payload_missing",
    "token_digest_mismatch",
    "unexpected_missing_normal_proposal",
    "actual_eager_counter_nonzero",
    "real_target_eager_nonempty",
}


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


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def as_int_set(value: Any) -> set[int]:
    return set(as_int_list(value))


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


def first_mapping(record: dict[str, Any], *keys: str) -> dict:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def commit_active(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("eager_commit_readiness_dry_run_enabled", False))
        or record.get("eager_commit_readiness_dry_run_source") == TAKEOVER_SOURCE
        or bool(as_int_set(record.get("eager_commit_readiness_candidate_proposal_ids")))
        or bool(as_int_set(record.get("eager_commit_ready_proposal_ids")))
        or bool(as_int_set(record.get("eager_commit_not_ready_proposal_ids")))
        or int_value(record.get("eager_commit_readiness_candidate_count"), 0) > 0
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_commit_enabled = 0
    commit_active_records = 0
    candidate_ids_seen: set[int] = set()
    ready_ids_seen: set[int] = set()
    not_ready_ids_seen: set[int] = set()
    candidate_result_by_id_seen: dict[int, str] = {}
    ready_token_by_id_seen: dict[int, int] = {}
    not_ready_reason_by_id_seen: dict[int, str] = {}
    actual_eager_counter_rows = 0
    real_target_eager_nonempty_count = 0
    missing_unexpected_count = 0
    reason_counts: Counter[str] = Counter()

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        target_eager = as_int_set(record.get("target_eager_set"))
        if target_eager:
            real_target_eager_nonempty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty: {sorted(target_eager)}")

        nonzero_actual = [
            field for field in ALWAYS_ZERO_COUNTER_FIELDS if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must remain zero: {nonzero_actual}")

        missing_unexpected = as_int_set(record.get("missing_buffered_proposal_unexpected_seq_ids"))
        if missing_unexpected:
            missing_unexpected_count += len(missing_unexpected)
            errors.append(
                f"record[{idx}] unexpected missing buffered normal proposals: {sorted(missing_unexpected)}"
            )

        enabled = bool(record.get("enable_eager_commit_readiness_dry_run", False))
        active = commit_active(record)
        if not enabled:
            if active:
                errors.append(f"record[{idx}] commit-readiness fields populated while disabled")
            continue

        records_with_commit_enabled += 1
        if not bool(record.get("enable_eager_sync_apply_dry_run", False)):
            errors.append(f"record[{idx}] commit-readiness dry-run must imply sync apply dry-run")
        if not active:
            continue

        commit_active_records += 1
        if record.get("eager_commit_readiness_dry_run_source") != TAKEOVER_SOURCE:
            errors.append(f"record[{idx}] commit-readiness source must be {TAKEOVER_SOURCE!r}")

        candidate_ids = as_int_set(record.get("eager_commit_readiness_candidate_proposal_ids"))
        ready_ids = as_int_set(record.get("eager_commit_ready_proposal_ids"))
        not_ready_ids = as_int_set(record.get("eager_commit_not_ready_proposal_ids"))
        from_sync_ids = as_int_set(record.get("eager_commit_readiness_from_sync_apply_proposal_ids"))
        from_result_ids = as_int_set(record.get("eager_commit_readiness_from_result_transfer_proposal_ids"))
        from_apply_ids = as_int_set(record.get("eager_commit_readiness_from_apply_proposal_ids"))
        from_verify_ids = as_int_set(record.get("eager_commit_readiness_from_verify_proposal_ids"))
        sync_consistent_ids = as_int_set(record.get("eager_sync_apply_dry_run_consistent_proposal_ids"))
        sync_executed_ids = as_int_set(record.get("eager_sync_apply_dry_run_executed_proposal_ids"))
        result_validated_ids = as_int_set(record.get("eager_result_transfer_validated_proposal_ids"))
        classified = ready_ids | not_ready_ids

        candidate_ids_seen.update(candidate_ids)
        ready_ids_seen.update(ready_ids)
        not_ready_ids_seen.update(not_ready_ids)
        if ready_ids & not_ready_ids:
            errors.append(f"record[{idx}] proposals both ready and not-ready: {sorted(ready_ids & not_ready_ids)}")
        if candidate_ids and classified != candidate_ids:
            errors.append(
                f"record[{idx}] every commit-readiness candidate must be classified exactly once: "
                f"classified={sorted(classified)}, candidates={sorted(candidate_ids)}"
            )
        if from_sync_ids and not candidate_ids <= from_sync_ids:
            errors.append(f"record[{idx}] candidates not from sync apply: {sorted(candidate_ids - from_sync_ids)}")
        if sync_executed_ids and not candidate_ids <= sync_executed_ids:
            errors.append(f"record[{idx}] candidates not sync-apply executed: {sorted(candidate_ids - sync_executed_ids)}")
        if sync_consistent_ids and not candidate_ids <= sync_consistent_ids:
            errors.append(f"record[{idx}] candidates not sync-apply consistent: {sorted(candidate_ids - sync_consistent_ids)}")
        if from_result_ids and not candidate_ids <= from_result_ids:
            errors.append(f"record[{idx}] candidates not from result transfer: {sorted(candidate_ids - from_result_ids)}")
        if result_validated_ids and not candidate_ids <= result_validated_ids:
            errors.append(f"record[{idx}] candidates not result-transfer validated: {sorted(candidate_ids - result_validated_ids)}")
        if from_apply_ids and not candidate_ids <= from_apply_ids:
            errors.append(f"record[{idx}] candidates not from apply dry-run: {sorted(candidate_ids - from_apply_ids)}")
        if from_verify_ids and not candidate_ids <= from_verify_ids:
            errors.append(f"record[{idx}] candidates not from verify dry-run: {sorted(candidate_ids - from_verify_ids)}")

        reason_by_id = first_mapping(record, "eager_commit_not_ready_reason_by_proposal_id")
        ready_token_by_id = first_mapping(record, "eager_commit_ready_token_count_by_proposal_id")
        ready_action_by_id = first_mapping(record, "eager_commit_ready_action_by_proposal_id")
        ready_accept_by_id = first_mapping(record, "eager_commit_ready_accept_len_by_proposal_id")
        ready_result_by_id = first_mapping(record, "eager_commit_ready_verify_result_by_proposal_id")
        verify_ok_by_id = first_mapping(record, "eager_commit_readiness_verify_ok_by_proposal_id")
        apply_ok_by_id = first_mapping(record, "eager_commit_readiness_apply_ok_by_proposal_id")
        result_ok_by_id = first_mapping(record, "eager_commit_readiness_result_transfer_ok_by_proposal_id")
        sync_ok_by_id = first_mapping(record, "eager_commit_readiness_sync_apply_ok_by_proposal_id")
        frontier_ok_by_id = first_mapping(record, "eager_commit_readiness_frontier_ok_by_proposal_id")
        token_payload_ok_by_id = first_mapping(record, "eager_commit_readiness_token_payload_ok_by_proposal_id")
        no_mutation_by_id = first_mapping(record, "eager_commit_readiness_no_mutation_by_proposal_id")
        sync_result_by_id = first_mapping(record, "eager_sync_apply_dry_run_draft_verify_result_by_proposal_id")
        sync_action_by_id = first_mapping(record, "eager_sync_apply_dry_run_draft_action_by_proposal_id")
        sync_accept_by_id = first_mapping(record, "eager_sync_apply_dry_run_draft_accept_len_by_proposal_id")
        gamma = int_value(record.get("normal_gamma"), 0)

        for proposal_id in sorted(ready_ids):
            guards = {
                "verify": dict_get(verify_ok_by_id, proposal_id),
                "apply": dict_get(apply_ok_by_id, proposal_id),
                "result_transfer": dict_get(result_ok_by_id, proposal_id),
                "sync_apply": dict_get(sync_ok_by_id, proposal_id),
                "frontier": dict_get(frontier_ok_by_id, proposal_id),
                "token_payload": dict_get(token_payload_ok_by_id, proposal_id),
                "no_mutation": dict_get(no_mutation_by_id, proposal_id),
            }
            failed = [name for name, value in guards.items() if value is not True]
            if failed:
                errors.append(f"record[{idx}] ready proposal_id={proposal_id} failed guards: {failed}")
            action = str(dict_get(ready_action_by_id, proposal_id, ""))
            verify_result = str(dict_get(ready_result_by_id, proposal_id, ""))
            accept_len = int_value(dict_get(ready_accept_by_id, proposal_id), -1)
            token_count = int_value(dict_get(ready_token_by_id, proposal_id), 0)
            if action != FULL_ACCEPT_ACTION:
                errors.append(f"record[{idx}] ready proposal_id={proposal_id} bad action={action!r}")
            if verify_result != "full_accept":
                errors.append(f"record[{idx}] ready proposal_id={proposal_id} is not full_accept")
            if gamma > 0 and accept_len != gamma:
                errors.append(f"record[{idx}] ready proposal_id={proposal_id} accept_len != gamma")
            if gamma > 0 and token_count != gamma:
                errors.append(f"record[{idx}] ready proposal_id={proposal_id} token count != gamma")
            ready_token_by_id_seen[proposal_id] = token_count

        for proposal_id in sorted(not_ready_ids):
            reason = str(dict_get(reason_by_id, proposal_id, ""))
            if not reason:
                errors.append(f"record[{idx}] not-ready proposal_id={proposal_id} missing reason")
            elif reason not in VALID_NOT_READY_REASONS:
                errors.append(f"record[{idx}] not-ready proposal_id={proposal_id} bad reason={reason!r}")
            not_ready_reason_by_id_seen[proposal_id] = reason
            verify_result = str(dict_get(sync_result_by_id, proposal_id, ""))
            if verify_result != "full_accept" and reason != "not_full_accept":
                errors.append(
                    f"record[{idx}] non-full proposal_id={proposal_id} must be not_ready reason=not_full_accept"
                )

        for proposal_id in sorted(candidate_ids):
            verify_result = str(dict_get(sync_result_by_id, proposal_id, ""))
            candidate_result_by_id_seen[proposal_id] = verify_result
            action = str(dict_get(sync_action_by_id, proposal_id, ""))
            accept_len = int_value(dict_get(sync_accept_by_id, proposal_id), -1)
            all_commit_guards_ok = all(
                dict_get(mapping, proposal_id) is True
                for mapping in (
                    verify_ok_by_id,
                    apply_ok_by_id,
                    result_ok_by_id,
                    sync_ok_by_id,
                    frontier_ok_by_id,
                    token_payload_ok_by_id,
                    no_mutation_by_id,
                )
            )
            if verify_result == "full_accept":
                if proposal_id not in ready_ids and proposal_id in not_ready_ids:
                    reason = str(dict_get(reason_by_id, proposal_id, ""))
                    if reason == "not_full_accept":
                        errors.append(f"record[{idx}] full-accept proposal_id={proposal_id} marked not_full_accept")
                if all_commit_guards_ok and proposal_id not in ready_ids:
                    errors.append(f"record[{idx}] full-accept proposal_id={proposal_id} passed all guards but is not ready")
            else:
                if proposal_id in ready_ids:
                    errors.append(f"record[{idx}] non-full proposal_id={proposal_id} marked ready")
            if action == FULL_ACCEPT_ACTION and verify_result != "full_accept":
                errors.append(f"record[{idx}] non-full proposal_id={proposal_id} has full-accept action")
            if verify_result == "full_accept" and gamma > 0 and accept_len != gamma:
                errors.append(f"record[{idx}] full-accept proposal_id={proposal_id} accept_len mismatch")

        if int_value(record.get("eager_commit_readiness_candidate_count"), 0) != len(candidate_ids):
            errors.append(f"record[{idx}] candidate count field mismatch")
        if int_value(record.get("eager_commit_ready_count"), 0) != len(ready_ids):
            errors.append(f"record[{idx}] ready count field mismatch")
        if int_value(record.get("eager_commit_not_ready_count"), 0) != len(not_ready_ids):
            errors.append(f"record[{idx}] not-ready count field mismatch")
        if int_value(record.get("eager_commit_ready_token_count"), 0) != sum(
            int_value(dict_get(ready_token_by_id, proposal_id), 0) for proposal_id in ready_ids
        ):
            errors.append(f"record[{idx}] ready token count field mismatch")
        if record.get("eager_commit_readiness_actual_counters_zero") is not True:
            errors.append(f"record[{idx}] actual counter guard is false")
        if record.get("eager_commit_readiness_real_target_eager_empty") is not True:
            errors.append(f"record[{idx}] real target eager guard is false")

    if records_with_commit_enabled and not candidate_ids_seen:
        errors.append("commit-readiness dry-run enabled but no candidates were audited")
    reason_counts = Counter(not_ready_reason_by_id_seen.values())
    full_accept_count = sum(1 for value in candidate_result_by_id_seen.values() if value == "full_accept")
    partial_reject_count = len(candidate_result_by_id_seen) - full_accept_count
    ready_token_count = sum(int(value) for value in ready_token_by_id_seen.values())

    summary = {
        "total_trace_records": len(records),
        "records_with_commit_readiness_dry_run_enabled": records_with_commit_enabled,
        "commit_readiness_active_records": commit_active_records,
        "candidate_count": len(candidate_ids_seen),
        "ready_count": len(ready_ids_seen),
        "not_ready_count": len(not_ready_ids_seen),
        "ready_token_count": ready_token_count,
        "not_ready_reason_counts": dict(reason_counts),
        "full_accept_count": full_accept_count,
        "partial_reject_count": partial_reject_count,
        "actual_eager_verified_counter_rows": actual_eager_counter_rows,
        "real_target_eager_non_empty_count": real_target_eager_nonempty_count,
        "missing_buffered_proposal_unexpected_count": missing_unexpected_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_commit_readiness_dry_run_enabled",
        "commit_readiness_active_records",
        "candidate_count",
        "ready_count",
        "not_ready_count",
        "ready_token_count",
        "not_ready_reason_counts",
        "full_accept_count",
        "partial_reject_count",
        "actual_eager_verified_counter_rows",
        "real_target_eager_non_empty_count",
        "missing_buffered_proposal_unexpected_count",
    ):
        print(f"{key}={summary[key]}")


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "normal_gamma": 4,
        "target_eager_set": [],
        "enable_eager_sync_apply_dry_run": False,
        "enable_eager_commit_readiness_dry_run": False,
        "eager_commit_readiness_dry_run_enabled": False,
        "eager_commit_readiness_dry_run_source": None,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_commit_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_sync_apply_dry_run": True,
            "enable_eager_commit_readiness_dry_run": True,
            "eager_commit_readiness_dry_run_enabled": True,
            "eager_commit_readiness_dry_run_source": TAKEOVER_SOURCE,
            "eager_commit_readiness_candidate_proposal_ids": [101, 102],
            "eager_commit_readiness_candidate_seq_ids": [7, 8],
            "eager_commit_readiness_from_sync_apply_proposal_ids": [101, 102],
            "eager_commit_readiness_from_result_transfer_proposal_ids": [101, 102],
            "eager_commit_readiness_from_apply_proposal_ids": [101, 102],
            "eager_commit_readiness_from_verify_proposal_ids": [101, 102],
            "eager_sync_apply_dry_run_executed_proposal_ids": [101, 102],
            "eager_sync_apply_dry_run_consistent_proposal_ids": [101, 102],
            "eager_result_transfer_validated_proposal_ids": [101, 102],
            "eager_sync_apply_dry_run_draft_verify_result_by_proposal_id": {
                "101": "full_accept",
                "102": "partial_accept",
            },
            "eager_sync_apply_dry_run_draft_action_by_proposal_id": {
                "101": FULL_ACCEPT_ACTION,
                "102": "discard_partial_no_mutation",
            },
            "eager_sync_apply_dry_run_draft_accept_len_by_proposal_id": {
                "101": 4,
                "102": 2,
            },
            "eager_commit_ready_proposal_ids": [101],
            "eager_commit_ready_seq_ids": [7],
            "eager_commit_ready_token_count_by_proposal_id": {"101": 4},
            "eager_commit_ready_action_by_proposal_id": {"101": FULL_ACCEPT_ACTION},
            "eager_commit_ready_accept_len_by_proposal_id": {"101": 4},
            "eager_commit_ready_verify_result_by_proposal_id": {"101": "full_accept"},
            "eager_commit_not_ready_proposal_ids": [102],
            "eager_commit_not_ready_seq_ids": [8],
            "eager_commit_not_ready_reason_by_proposal_id": {"102": "not_full_accept"},
            "eager_commit_readiness_verify_ok_by_proposal_id": {"101": True, "102": False},
            "eager_commit_readiness_apply_ok_by_proposal_id": {"101": True, "102": False},
            "eager_commit_readiness_result_transfer_ok_by_proposal_id": {"101": True, "102": True},
            "eager_commit_readiness_sync_apply_ok_by_proposal_id": {"101": True, "102": True},
            "eager_commit_readiness_frontier_ok_by_proposal_id": {"101": True, "102": True},
            "eager_commit_readiness_token_payload_ok_by_proposal_id": {"101": True, "102": True},
            "eager_commit_readiness_no_mutation_by_proposal_id": {"101": True, "102": True},
            "eager_commit_readiness_actual_counters_zero": True,
            "eager_commit_readiness_real_target_eager_empty": True,
            "eager_commit_readiness_candidate_count": 2,
            "eager_commit_ready_count": 1,
            "eager_commit_not_ready_count": 1,
            "eager_commit_ready_token_count": 4,
            "eager_commit_not_ready_reason_counts": {"not_full_accept": 1},
            "eager_commit_readiness_full_accept_count": 1,
            "eager_commit_readiness_partial_reject_count": 1,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid_records = [synthetic_base_record(), synthetic_commit_record()]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic commit-readiness records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_not_ready_reason_by_proposal_id"] = {}
    errors, _ = validate_records(invalid)
    assert any("missing reason" in error for error in errors), "checker missed not-ready reason"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_ready_proposal_ids"] = [101, 102]
    invalid[1]["eager_commit_not_ready_proposal_ids"] = []
    invalid[1]["eager_commit_ready_count"] = 2
    invalid[1]["eager_commit_not_ready_count"] = 0
    errors, _ = validate_records(invalid)
    assert any("non-full" in error and "marked ready" in error for error in errors), (
        "checker missed partial proposal marked ready"
    )

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_readiness_apply_ok_by_proposal_id"] = {"101": False, "102": False}
    errors, _ = validate_records(invalid)
    assert any("failed guards" in error for error in errors), "checker missed ready guard failure"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_ready_token_count_by_proposal_id"] = {"101": 3}
    errors, _ = validate_records(invalid)
    assert any("token count" in error for error in errors), "checker missed ready token mismatch"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = deepcopy(valid_records)
    invalid[1]["target_eager_set"] = [7]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = deepcopy(valid_records)
    invalid[1]["missing_buffered_proposal_unexpected_seq_ids"] = [7]
    errors, _ = validate_records(invalid)
    assert any("unexpected missing" in error for error in errors), "checker missed unexpected missing normal proposal"

    invalid = deepcopy(valid_records)
    invalid[1]["eager_commit_readiness_from_sync_apply_proposal_ids"] = [101]
    errors, _ = validate_records(invalid)
    assert any("not from sync apply" in error for error in errors), "checker missed invalid candidate source"

    print("Synthetic eager commit-readiness dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5j eager commit-readiness dry-run traces.")
    parser.add_argument("trace", nargs="?", type=Path, help="Optional engine trace JSON to validate.")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic checker tests.")
    args = parser.parse_args()

    if args.synthetic or args.trace is None:
        run_synthetic_tests()
        return 0

    records = load_trace(args.trace)
    errors, summary = validate_records(records)
    print_summary(summary)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Eager commit-readiness dry-run trace checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
