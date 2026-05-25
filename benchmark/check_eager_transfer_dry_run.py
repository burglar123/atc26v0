#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from nano_pearl.pearl_engine.dual_batch import (  # type: ignore  # noqa: E402
        EAGER_STATE_READY_TO_VERIFY,
        EagerProposal,
        deserialize_eager_transfer_payload,
        serialize_eager_transfer_payload,
    )
except ModuleNotFoundError:
    from benchmark.check_eager_scaffold import (  # type: ignore  # noqa: E402
        EAGER_STATE_READY_TO_VERIFY,
        EagerProposal,
        deserialize_eager_transfer_payload,
        serialize_eager_transfer_payload,
    )


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


def is_dual_record(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_mode") == "dual_batch_pearl"
        and record.get("dual_batch_enabled") is True
    )


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    transfer_records = 0
    transfer_steps = 0
    zero_proposal_steps = 0
    sent_count = 0
    received_count = 0
    validated_count = 0
    dropped_count = 0
    drop_reason_counts: Counter[str] = Counter()
    generated_tokens = 0
    promoted_tokens = 0
    discarded_tokens = 0
    transferred_tokens = 0
    transfer_validated_tokens = 0
    target_eager_non_empty_count = 0
    verified_counter_rows = 0
    base_mismatch_count = 0

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        plan_enabled = bool(record.get("enable_eager_plan_dry_run", False))
        draft_enabled = bool(record.get("enable_eager_draft_dry_run", False))
        promotion_enabled = bool(record.get("enable_eager_promotion_dry_run", False))
        transfer_enabled = bool(record.get("enable_eager_transfer_dry_run", False))
        transfer_active = bool(record.get("eager_transfer_dry_run_enabled", False))
        target_eager = as_int_set(record.get("target_eager_set"))
        draft_eager = as_int_set(record.get("draft_eager_set"))
        sent_ids = as_int_set(record.get("eager_transfer_sent_proposal_ids"))
        received_ids = as_int_set(record.get("eager_transfer_received_proposal_ids"))
        validated_ids = as_int_set(record.get("eager_transfer_validated_proposal_ids"))
        dropped_ids = as_int_set(record.get("eager_transfer_dropped_proposal_ids"))

        if target_eager:
            target_eager_non_empty_count += 1
            errors.append(f"record[{idx}] target_eager_set must remain empty, got {sorted(target_eager)}")

        nonzero_verified = [
            field
            for field in ALWAYS_ZERO_COUNTER_FIELDS
            if int_value(record.get(field), 0) != 0
        ]
        if nonzero_verified:
            verified_counter_rows += 1
            errors.append(f"record[{idx}] eager verification counters must stay zero: {nonzero_verified}")

        generated = int_value(record.get("eager_tokens_generated"), 0)
        promoted = int_value(record.get("eager_tokens_promoted"), 0)
        discarded = int_value(record.get("eager_tokens_discarded"), 0)
        transferred = int_value(record.get("eager_tokens_transferred"), 0)
        transfer_validated = int_value(record.get("eager_tokens_transfer_validated"), 0)
        transfer_dropped = int_value(record.get("eager_tokens_transfer_dropped"), 0)
        generated_tokens += generated
        promoted_tokens += promoted
        discarded_tokens += discarded
        transferred_tokens += transferred
        transfer_validated_tokens += transfer_validated

        if not plan_enabled and not draft_enabled and not promotion_enabled and not transfer_enabled:
            if draft_eager or generated or promoted or discarded or transferred or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] default run has eager activity")
            continue

        if plan_enabled and not draft_enabled and not promotion_enabled and not transfer_enabled:
            if generated or promoted or discarded or transferred or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] plan dry-run has eager execution/transfer counters")
            continue

        if draft_enabled and not promotion_enabled and not transfer_enabled:
            if transferred or promoted or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] draft dry-run has promotion/transfer counters")
            continue

        if promotion_enabled and not transfer_enabled:
            if transferred or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] promotion dry-run has transfer counters")
            continue

        if transfer_enabled:
            if not (plan_enabled and draft_enabled and promotion_enabled):
                errors.append(f"record[{idx}] transfer dry-run must imply plan/draft/promotion dry-runs")
            if transfer_active:
                transfer_records += 1
                transfer_steps += 1

        if not transfer_active:
            if transferred or transfer_validated or transfer_dropped:
                errors.append(f"record[{idx}] transfer counters outside active transfer dry-run row")
            continue

        num_proposals = int_value(record.get("eager_transfer_num_proposals"), 0)
        if num_proposals == 0:
            zero_proposal_steps += 1
        if num_proposals != len(sent_ids) and sent_ids:
            errors.append(f"record[{idx}] eager_transfer_num_proposals does not match sent ids")

        sent_count += len(sent_ids)
        received_count += len(received_ids)
        validated_count += len(validated_ids)
        dropped_count += len(dropped_ids)
        if sent_ids != received_ids:
            errors.append(
                f"record[{idx}] received proposal ids must match sent proposal ids: "
                f"sent={sorted(sent_ids)}, received={sorted(received_ids)}"
            )
        if validated_ids | dropped_ids != received_ids:
            errors.append(
                f"record[{idx}] validated+dropped proposal ids must classify received ids: "
                f"classified={sorted(validated_ids | dropped_ids)}, received={sorted(received_ids)}"
            )
        if validated_ids & dropped_ids:
            errors.append(f"record[{idx}] proposal ids both validated and dropped: {sorted(validated_ids & dropped_ids)}")
        if transferred != promoted:
            errors.append(
                f"record[{idx}] eager_tokens_transferred must equal eager_tokens_promoted: "
                f"{transferred} != {promoted}"
            )
        if transfer_validated + transfer_dropped != transferred:
            errors.append(
                f"record[{idx}] transfer validated+dropped tokens must equal transferred tokens: "
                f"{transfer_validated} + {transfer_dropped} != {transferred}"
            )
        if int_value(record.get("draft_eager_buffer_size_after_transfer"), 0) != 0:
            errors.append(f"record[{idx}] draft_eager_buffer_size_after_transfer must be 0")
        if int_value(record.get("target_eager_buffer_size_after_clear"), 0) != 0:
            errors.append(f"record[{idx}] target_eager_buffer_size_after_clear must be 0")

        base_match = record.get("eager_transfer_base_match_by_seq_id", {})
        base_pre_verify = record.get("eager_transfer_base_pre_verify_by_seq_id", {})
        proposal_len_by_id = record.get("eager_transfer_proposal_len_by_proposal_id", {})
        to_verify_len_by_id = record.get("eager_transfer_to_verify_len_by_proposal_id", {})
        drop_reasons = record.get("eager_transfer_drop_reason_by_proposal_id", {})
        gamma = int_value(record.get("normal_gamma"), None)

        proposal_to_seq = {
            proposal_id: seq_id
            for proposal_id, seq_id in zip(
                as_int_list(record.get("eager_transfer_received_proposal_ids")),
                as_int_list(record.get("eager_transfer_received_seq_ids")),
            )
        }
        for proposal_id in sorted(validated_ids):
            seq_id = proposal_to_seq.get(proposal_id)
            if seq_id is None:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} missing seq id")
                continue
            if dict_get(base_match, seq_id) is not True:
                base_mismatch_count += 1
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} lacks base_len match")
            if dict_get(base_pre_verify, seq_id) is not False:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} base_pre_verify is not false")
            if gamma is not None and int_value(dict_get(proposal_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} proposal_len != gamma")
            if gamma is not None and int_value(dict_get(to_verify_len_by_id, proposal_id), -1) != gamma:
                errors.append(f"record[{idx}] validated proposal_id={proposal_id} to_verify_len != gamma")

        for proposal_id in sorted(dropped_ids):
            reason = dict_get(drop_reasons, proposal_id)
            drop_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] dropped proposal_id={proposal_id} missing drop reason")
            seq_id = proposal_to_seq.get(proposal_id)
            if seq_id is not None and dict_get(base_match, seq_id) is False:
                base_mismatch_count += 1

    summary = {
        "total_trace_records": len(records),
        "records_with_transfer_dry_run_enabled": transfer_records,
        "transfer_steps": transfer_steps,
        "zero_proposal_transfer_steps": zero_proposal_steps,
        "sent_proposal_count": sent_count,
        "received_proposal_count": received_count,
        "validated_proposal_count": validated_count,
        "dropped_proposal_count": dropped_count,
        "drop_reason_counts": dict(drop_reason_counts),
        "eager_tokens_generated": generated_tokens,
        "eager_tokens_promoted": promoted_tokens,
        "eager_tokens_discarded": discarded_tokens,
        "eager_tokens_transferred": transferred_tokens,
        "eager_tokens_transfer_validated": transfer_validated_tokens,
        "target_eager_non_empty_count": target_eager_non_empty_count,
        "eager_verified_counter_rows": verified_counter_rows,
        "base_mismatch_count": base_mismatch_count,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "total_trace_records",
        "records_with_transfer_dry_run_enabled",
        "transfer_steps",
        "zero_proposal_transfer_steps",
        "sent_proposal_count",
        "received_proposal_count",
        "validated_proposal_count",
        "dropped_proposal_count",
        "drop_reason_counts",
        "eager_tokens_generated",
        "eager_tokens_promoted",
        "eager_tokens_discarded",
        "eager_tokens_transferred",
        "eager_tokens_transfer_validated",
        "target_eager_non_empty_count",
        "eager_verified_counter_rows",
        "base_mismatch_count",
    ):
        print(f"{key}={summary[key]}")


def make_proposal(proposal_id: int = 101, seq_id: int = 1) -> EagerProposal:
    return EagerProposal(
        proposal_id=proposal_id,
        seq_id=seq_id,
        request_id=f"req-{seq_id}",
        parent_kind="normal",
        parent_proposal_id=None,
        source_step_id=7,
        source_plan_id=11,
        home_batch_id=0,
        base_len=12,
        base_pre_verify=False,
        base_num_completion_tokens=8,
        proposal_token_ids=[201, 202, 203, 204],
        to_be_verified_token_ids=[111, 112, 113, 201],
        proposal_len=4,
        state=EAGER_STATE_READY_TO_VERIFY,
        valid=True,
    )


def synthetic_base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "enable_eager_plan_dry_run": False,
        "enable_eager_draft_dry_run": False,
        "enable_eager_promotion_dry_run": False,
        "enable_eager_transfer_dry_run": False,
        "eager_transfer_dry_run_enabled": False,
        "target_home_set": [1],
        "draft_home_set": [2],
        "target_eager_set": [],
        "draft_eager_set": [],
        "normal_gamma": 4,
        "eager_tokens_generated": 0,
        "eager_tokens_promoted": 0,
        "eager_tokens_discarded": 0,
        "eager_tokens_transferred": 0,
        "eager_tokens_transfer_validated": 0,
        "eager_tokens_transfer_dropped": 0,
    }
    for field in ALWAYS_ZERO_COUNTER_FIELDS:
        record[field] = 0
    return record


def synthetic_plan_record() -> dict[str, Any]:
    record = synthetic_base_record()
    record.update(
        {
            "enable_eager_plan_dry_run": True,
            "target_home_set": [1],
            "draft_eager_set": [1],
        }
    )
    return record


def synthetic_draft_record() -> dict[str, Any]:
    record = synthetic_plan_record()
    record.update(
        {
            "enable_eager_draft_dry_run": True,
            "eager_tokens_generated": 4,
        }
    )
    return record


def synthetic_promotion_record() -> dict[str, Any]:
    record = synthetic_draft_record()
    record.update(
        {
            "enable_eager_promotion_dry_run": True,
            "eager_tokens_promoted": 4,
            "eager_tokens_discarded": 0,
        }
    )
    return record


def synthetic_transfer_record(num_proposals: int = 1) -> dict[str, Any]:
    record = synthetic_promotion_record()
    sent_ids = [101] if num_proposals else []
    seq_ids = [1] if num_proposals else []
    transferred = 4 if num_proposals else 0
    record.update(
        {
            "enable_eager_transfer_dry_run": True,
            "eager_transfer_dry_run_enabled": True,
            "eager_transfer_step_id": 7,
            "eager_transfer_plan_id": 11,
            "eager_transfer_num_proposals": num_proposals,
            "eager_transfer_payload_len": 20 if num_proposals else 0,
            "eager_transfer_sent_proposal_ids": sent_ids,
            "eager_transfer_sent_seq_ids": seq_ids,
            "eager_transfer_received_proposal_ids": sent_ids,
            "eager_transfer_received_seq_ids": seq_ids,
            "eager_transfer_validated_proposal_ids": sent_ids,
            "eager_transfer_dropped_proposal_ids": [],
            "eager_transfer_drop_reason_by_proposal_id": {},
            "eager_transfer_base_len_by_seq_id": {"1": 12} if num_proposals else {},
            "eager_transfer_base_pre_verify_by_seq_id": {"1": False} if num_proposals else {},
            "eager_transfer_current_len_by_seq_id": {"1": 12} if num_proposals else {},
            "eager_transfer_base_match_by_seq_id": {"1": True} if num_proposals else {},
            "eager_transfer_proposal_len_by_proposal_id": {"101": 4} if num_proposals else {},
            "eager_transfer_to_verify_len_by_proposal_id": {"101": 4} if num_proposals else {},
            "draft_eager_buffer_size_after_transfer": 0,
            "target_eager_buffer_size_after_clear": 0,
            "eager_tokens_transferred": transferred,
            "eager_tokens_transfer_validated": transferred,
            "eager_tokens_transfer_dropped": 0,
        }
    )
    if not num_proposals:
        record["eager_tokens_generated"] = 0
        record["eager_tokens_promoted"] = 0
    return record


def run_synthetic_tests() -> None:
    proposal = make_proposal()
    meta, payload = serialize_eager_transfer_payload([proposal], gamma=4, plan_id=11, step_id=7)
    roundtrip = deserialize_eager_transfer_payload(meta, payload)
    assert len(roundtrip) == 1
    assert roundtrip[0].proposal_id == proposal.proposal_id
    assert roundtrip[0].proposal_token_ids == proposal.proposal_token_ids
    assert roundtrip[0].to_be_verified_token_ids == proposal.to_be_verified_token_ids

    zero_meta, zero_payload = serialize_eager_transfer_payload([], gamma=4, plan_id=11, step_id=7)
    assert zero_meta[0] == 0 and zero_meta[1] == 0 and zero_payload == []
    assert deserialize_eager_transfer_payload(zero_meta, zero_payload) == []

    bad_meta = list(meta)
    bad_meta[1] += 1
    try:
        deserialize_eager_transfer_payload(bad_meta, payload)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed eager transfer payload length was accepted")

    valid_records = [
        synthetic_base_record(),
        synthetic_plan_record(),
        synthetic_draft_record(),
        synthetic_promotion_record(),
        synthetic_transfer_record(0),
        synthetic_transfer_record(1),
    ]
    errors, _ = validate_records(valid_records)
    assert not errors, f"valid synthetic eager transfer records failed: {errors}"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_received_proposal_ids"] = [999]
    errors, _ = validate_records(invalid)
    assert any("received proposal ids" in error for error in errors), "checker missed sent/received mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_base_match_by_seq_id"] = {"1": False}
    errors, _ = validate_records(invalid)
    assert any("base_len match" in error for error in errors), "checker missed base_len mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_base_pre_verify_by_seq_id"] = {"1": True}
    errors, _ = validate_records(invalid)
    assert any("base_pre_verify" in error for error in errors), "checker missed base_pre_verify mismatch"

    invalid = deepcopy(valid_records)
    invalid[-1]["eager_transfer_validated_proposal_ids"] = []
    invalid[-1]["eager_transfer_dropped_proposal_ids"] = [101]
    invalid[-1]["eager_transfer_drop_reason_by_proposal_id"] = {"101": "seq_not_running"}
    invalid[-1]["eager_tokens_transfer_validated"] = 0
    invalid[-1]["eager_tokens_transfer_dropped"] = 4
    errors, _ = validate_records(invalid)
    assert not errors, f"valid seq_not_running drop failed: {errors}"

    print("Synthetic eager transfer dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-4 eager transfer dry-run traces.")
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
    print("\nEager transfer dry-run trace check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
