#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any


ACTUAL_EAGER_COUNTERS = [
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
    raise ValueError("Unsupported trace format")


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


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def validate_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    records_with_lane_enabled = 0
    scheduled_target_eager_count = 0
    excluded_from_actual_count = 0
    deferred_count = 0
    decision_late_count = 0
    sent_received_mismatch_count = 0
    excluded_expected_count = 0
    missing_after_lane_count = 0
    fallback_handled_count = 0
    actual_eager_counter_rows = 0
    real_target_eager_nonempty_count = 0
    original_draft_sizes: list[int] = []
    adjusted_draft_sizes: list[int] = []
    done_proposal_counts: Counter[int] = Counter()
    defer_reason_counts: Counter[str] = Counter()
    drop_reason_counts: Counter[str] = Counter()
    coverage_candidates = 0
    created_ids: set[int] = set()
    activated_ids: set[int] = set()
    expired_ids: set[int] = set()
    dropped_ids: set[int] = set()
    created_step_by_id: dict[int, int] = {}
    activated_step_by_id: dict[int, int] = {}
    activation_source_by_id: dict[int, str] = {}
    same_step_activation_count = 0
    activated_without_deferred_count = 0
    decision_available_before_draft_count = 0
    transfer_by_step: dict[tuple[int, int], dict[str, set[int]]] = {}
    role_by_step: dict[int, dict[str, Any]] = {}

    for idx, record in enumerate(records):
        if not is_dual_record(record):
            continue

        lane_enabled = bool(record.get("enable_eager_lane_exclusion_dry_run", False))
        lane_active = bool(record.get("eager_lane_exclusion_dry_run_enabled", False))
        target_home = as_int_set(record.get("target_home_set"))
        real_target_eager = as_int_set(record.get("target_eager_set"))
        original = as_int_set(record.get("original_draft_home_set")) or as_int_set(record.get("draft_home_set"))
        actual = (
            as_int_set(record.get("actual_draft_home_set_for_normal_draft"))
            or as_int_set(record.get("draft_home_set"))
        )
        scheduled = as_int_set(record.get("scheduled_target_eager_set_dry_run")) or as_int_set(
            record.get("target_eager_set_dry_run")
        )
        excluded = as_int_set(record.get("excluded_from_actual_draft_home_for_eager")) or as_int_set(
            record.get("lane_excluded_seq_ids")
        )
        excluded_dry_run = as_int_set(record.get("excluded_from_draft_home_for_eager_dry_run"))
        adjusted_dry_run = as_int_set(record.get("adjusted_draft_home_set_dry_run"))
        expected_seq_ids = as_int_set(record.get("normal_proposal_expected_seq_ids_after_lane_exclusion")) or actual
        adjusted_expected = as_int_set(record.get("adjusted_normal_proposal_expected_seq_ids")) or actual
        sent_seq_ids = as_int_set(record.get("normal_proposal_sent_seq_ids_after_lane_exclusion"))
        received_seq_ids = as_int_set(record.get("normal_proposal_received_seq_ids_after_lane_exclusion"))
        decision_available = bool(record.get("lane_exclusion_decision_available_before_draft", False))
        deferred = bool(record.get("lane_exclusion_deferred_until_next_step", False))
        defer_reason = record.get("lane_exclusion_defer_reason")
        done = bool(record.get("lane_exclusion_dry_run_done", False))
        done_proposal_ids = as_int_list(record.get("lane_exclusion_dry_run_done_proposal_ids"))
        missing_after_lane = as_int_set(record.get("missing_normal_proposal_after_lane_exclusion"))
        fallback_handled = as_int_set(record.get("missing_normal_proposal_handled_by_fallback"))
        lane_tokens = int_value(record.get("eager_tokens_lane_excluded_dry_run"), 0)
        step_id = int_value(record.get("step_id"), int_value(record.get("lane_exclusion_decision_shared_step_id"), -1))
        runner_role = str(record.get("runner_role", ""))
        created = as_int_set(record.get("deferred_lane_exclusion_created_proposal_ids"))
        activated = as_int_set(record.get("deferred_lane_exclusion_activated_proposal_ids")) or set(done_proposal_ids)
        expired = as_int_set(record.get("deferred_lane_exclusion_expired_proposal_ids"))
        dropped = as_int_set(record.get("deferred_lane_exclusion_dropped_proposal_ids"))
        created_steps = record.get("deferred_lane_exclusion_created_step_by_proposal_id", {})
        activated_steps = record.get("deferred_lane_exclusion_activated_step_by_proposal_id", {})
        activation_source = str(record.get("lane_exclusion_activation_source") or "")
        activated_without_deferred = int_value(record.get("activated_without_deferred_count"), 0)
        activated_without_deferred_count += activated_without_deferred

        if real_target_eager:
            real_target_eager_nonempty_count += 1
            errors.append(f"record[{idx}] real target_eager_set must remain empty: {sorted(real_target_eager)}")

        nonzero_actual = [
            field for field in ACTUAL_EAGER_COUNTERS if int_value(record.get(field), 0) != 0
        ]
        if nonzero_actual:
            actual_eager_counter_rows += 1
            errors.append(f"record[{idx}] actual eager counters must remain zero: {nonzero_actual}")

        if not lane_enabled:
            if lane_active or excluded or lane_tokens:
                errors.append(f"record[{idx}] lane-exclusion fields populated while lane dry-run disabled")
            continue

        records_with_lane_enabled += 1
        original_draft_sizes.append(len(original))
        adjusted_draft_sizes.append(len(actual))
        scheduled_target_eager_count += len(scheduled)
        excluded_from_actual_count += len(excluded)
        missing_after_lane_count += len(missing_after_lane)
        fallback_handled_count += len(fallback_handled)
        decision_late_count += int_value(record.get("lane_exclusion_decision_late_count"), 0)
        coverage_candidates += len(scheduled & original) + len(
            as_int_set(record.get("eager_lane_exclusion_seq_ids")) & original
        )
        created_ids.update(created)
        activated_ids.update(activated)
        expired_ids.update(expired)
        dropped_ids.update(dropped)
        for proposal_id in created:
            created_step_by_id.setdefault(
                proposal_id,
                int_value(created_steps.get(str(proposal_id), created_steps.get(proposal_id, step_id)), step_id),
            )
        for proposal_id in activated:
            activated_step = int_value(
                activated_steps.get(str(proposal_id), activated_steps.get(proposal_id, step_id)),
                step_id,
            )
            activated_step_by_id[proposal_id] = activated_step
            activation_source_by_id[proposal_id] = activation_source
            created_step = created_step_by_id.get(
                proposal_id,
                int_value(created_steps.get(str(proposal_id), created_steps.get(proposal_id, -1)), -1),
            )
            if created_step >= 0 and activated_step <= created_step:
                same_step_activation_count += 1
                errors.append(
                    f"record[{idx}] proposal_id={proposal_id} activated at step {activated_step} "
                    f"but created at step {created_step}"
                )
            if activation_source != "deferred_previous_step":
                errors.append(
                    f"record[{idx}] activated proposal_id={proposal_id} has bad activation source={activation_source!r}"
                )
            if proposal_id not in created_ids and proposal_id not in created_step_by_id:
                errors.append(f"record[{idx}] proposal_id={proposal_id} activated without deferred creation")
        drop_reasons = record.get("deferred_lane_exclusion_drop_reason_by_proposal_id", {})
        for proposal_id in dropped:
            reason = drop_reasons.get(str(proposal_id), drop_reasons.get(proposal_id, ""))
            drop_reason_counts[str(reason)] += 1
            if not reason:
                errors.append(f"record[{idx}] dropped deferred proposal_id={proposal_id} missing reason")

        if bool(record.get("lane_exclusion_decision_transfer_called", False)):
            shared_key = (
                int_value(record.get("lane_exclusion_decision_shared_plan_id"), -1),
                int_value(record.get("lane_exclusion_decision_shared_step_id"), -1),
            )
            step = transfer_by_step.setdefault(shared_key, {"sent": set(), "received": set()})
            step["sent"].update(as_int_set(record.get("lane_exclusion_decision_sent_proposal_ids")))
            step["received"].update(as_int_set(record.get("lane_exclusion_decision_received_proposal_ids")))

        role_step = role_by_step.setdefault(
            step_id,
            {
                "target_deferred": set(),
                "draft_deferred": set(),
                "target_activated": set(),
                "draft_activated": set(),
                "target_excluded": set(),
                "draft_excluded": set(),
                "target_actual": None,
                "draft_actual": None,
            },
        )
        if "verify" in runner_role or "target" in runner_role:
            role_step["target_deferred"].update(as_int_set(record.get("target_deferred_decision_ids")))
            role_step["target_activated"].update(as_int_set(record.get("target_activated_decision_ids")))
            role_step["target_excluded"].update(as_int_set(record.get("target_lane_excluded_seq_ids")))
            target_actual = as_int_set(record.get("target_actual_draft_home_set_for_normal_draft"))
            if target_actual:
                role_step["target_actual"] = target_actual
        if "draft" in runner_role:
            role_step["draft_deferred"].update(as_int_set(record.get("draft_deferred_decision_ids")))
            role_step["draft_activated"].update(as_int_set(record.get("draft_activated_decision_ids")))
            role_step["draft_excluded"].update(as_int_set(record.get("draft_lane_excluded_seq_ids")))
            draft_actual = as_int_set(record.get("draft_actual_draft_home_set_for_normal_draft"))
            if draft_actual:
                role_step["draft_actual"] = draft_actual

        if scheduled & target_home:
            errors.append(
                f"record[{idx}] scheduled target eager dry-run intersects target_home_set: "
                f"{sorted(scheduled & target_home)}"
            )

        if decision_available:
            decision_available_before_draft_count += 1
            expected_excluded = scheduled & original if scheduled else excluded
            if excluded != expected_excluded:
                errors.append(
                    f"record[{idx}] excluded seqs must match scheduled/original intersection: "
                    f"excluded={sorted(excluded)}, expected={sorted(expected_excluded)}"
                )
            expected_actual = original - excluded
            if actual != expected_actual:
                errors.append(
                    f"record[{idx}] actual normal draft set must equal original minus excluded: "
                    f"actual={sorted(actual)}, expected={sorted(expected_actual)}"
                )
            if adjusted_dry_run and adjusted_dry_run != expected_actual:
                errors.append(
                    f"record[{idx}] adjusted_draft_home_set_dry_run must equal actual adjusted set"
                )
            if excluded_dry_run and excluded_dry_run != excluded:
                errors.append(
                    f"record[{idx}] excluded_from_draft_home_for_eager_dry_run must match actual excluded seqs"
                )
            if expected_seq_ids != actual or adjusted_expected != actual:
                errors.append(f"record[{idx}] normal proposal expected seq ids must use adjusted set")
            if sent_seq_ids and sent_seq_ids != actual:
                sent_received_mismatch_count += 1
                errors.append(
                    f"record[{idx}] sent normal proposal seq ids must match adjusted set: "
                    f"sent={sorted(sent_seq_ids)}, expected={sorted(actual)}"
                )
            if received_seq_ids and received_seq_ids != actual:
                sent_received_mismatch_count += 1
                errors.append(
                    f"record[{idx}] received normal proposal seq ids must match adjusted set: "
                    f"received={sorted(received_seq_ids)}, expected={sorted(actual)}"
                )
            if excluded & actual:
                errors.append(f"record[{idx}] excluded seqs still present in actual draft set")
            if excluded & sent_seq_ids or excluded & received_seq_ids:
                errors.append(f"record[{idx}] excluded seqs appeared in normal proposal metadata")
            if not done:
                errors.append(f"record[{idx}] actual lane exclusion must mark lane_exclusion_dry_run_done")
            if not activated:
                errors.append(f"record[{idx}] actual lane exclusion must cite activated deferred proposal ids")
            if lane_tokens != len(excluded) * int_value(record.get("normal_gamma"), 0):
                errors.append(f"record[{idx}] eager_tokens_lane_excluded_dry_run has bad token accounting")
            for proposal_id in done_proposal_ids:
                done_proposal_counts[int(proposal_id)] += 1
        else:
            if actual != original:
                errors.append(
                    f"record[{idx}] late/unavailable decision must not retroactively alter draft set: "
                    f"actual={sorted(actual)}, original={sorted(original)}"
                )
            if excluded:
                errors.append(f"record[{idx}] unavailable decision must not exclude actual draft seqs")
            if scheduled & original:
                if not deferred:
                    errors.append(f"record[{idx}] late scheduled draft-home candidate must be deferred")
                if defer_reason != "decision_not_available_before_normal_draft":
                    errors.append(
                        f"record[{idx}] bad lane exclusion defer reason={defer_reason!r}"
                    )
            if deferred:
                deferred_count += 1
                defer_reason_counts[str(defer_reason)] += 1

        if excluded & expected_seq_ids:
            excluded_expected_count += len(excluded & expected_seq_ids)
            errors.append(f"record[{idx}] excluded seqs were still expected by normal proposal receive")

    repeated_done = [proposal_id for proposal_id, count in done_proposal_counts.items() if count > 2]
    if repeated_done:
        errors.append(f"lane exclusion dry-run done repeated for proposal ids={sorted(repeated_done)}")

    for step, values in sorted(transfer_by_step.items()):
        if values["sent"] != values["received"]:
            errors.append(
                f"lane decision transfer mismatch for step={step}: "
                f"sent={sorted(values['sent'])}, received={sorted(values['received'])}"
            )

    for step_id, values in sorted(role_by_step.items()):
        target_deferred = values["target_deferred"]
        draft_deferred = values["draft_deferred"]
        target_activated = values["target_activated"]
        draft_activated = values["draft_activated"]
        if target_deferred and draft_deferred and target_deferred != draft_deferred:
            errors.append(
                f"step={step_id} target/draft deferred decisions diverged: "
                f"target={sorted(target_deferred)}, draft={sorted(draft_deferred)}"
            )
        if target_activated or draft_activated:
            if target_activated != draft_activated:
                errors.append(
                    f"step={step_id} target/draft activated decisions diverged: "
                    f"target={sorted(target_activated)}, draft={sorted(draft_activated)}"
                )
            if values["target_excluded"] != values["draft_excluded"]:
                errors.append(
                    f"step={step_id} target/draft lane excluded seqs diverged: "
                    f"target={sorted(values['target_excluded'])}, draft={sorted(values['draft_excluded'])}"
                )
            if (
                values["target_actual"] is not None
                and values["draft_actual"] is not None
                and values["target_actual"] != values["draft_actual"]
            ):
                errors.append(
                    f"step={step_id} target/draft adjusted draft sets diverged: "
                    f"target={sorted(values['target_actual'])}, draft={sorted(values['draft_actual'])}"
                )

    if activated_without_deferred_count:
        errors.append(f"activated_without_deferred_count must be 0, got {activated_without_deferred_count}")

    coverage_caveat = ""
    if records_with_lane_enabled and (created_ids or coverage_candidates) and excluded_from_actual_count == 0:
        terminal_ids = expired_ids | dropped_ids
        if created_ids and created_ids <= terminal_ids:
            coverage_caveat = "pass with coverage caveat: all deferred lane decisions expired or dropped"
        else:
            errors.append("deferred lane-exclusion decisions existed but no actual exclusion was activated")

    summary = {
        "total_trace_records": len(records),
        "records_with_lane_exclusion_enabled": records_with_lane_enabled,
        "scheduled_target_eager_dry_run_count": scheduled_target_eager_count,
        "deferred_created_count": len(created_ids),
        "deferred_activated_count": len(activated_ids),
        "excluded_from_actual_draft_home_count": excluded_from_actual_count,
        "expired_count": len(expired_ids),
        "drop_reason_counts": dict(drop_reason_counts),
        "lane_exclusion_deferred_count": deferred_count,
        "lane_exclusion_decision_late_count": decision_late_count,
        "decision_available_before_draft_count": decision_available_before_draft_count,
        "original_draft_home_size_mean": mean(original_draft_sizes) if original_draft_sizes else 0.0,
        "adjusted_draft_home_size_mean": mean(adjusted_draft_sizes) if adjusted_draft_sizes else 0.0,
        "normal_proposal_sent_received_mismatch_count": sent_received_mismatch_count,
        "excluded_seqs_in_normal_expected_count": excluded_expected_count,
        "missing_normal_proposal_after_lane_exclusion_count": missing_after_lane_count,
        "fallback_handled_missing_proposal_count": fallback_handled_count,
        "actual_eager_verified_counter_rows": actual_eager_counter_rows,
        "real_target_eager_nonempty_count": real_target_eager_nonempty_count,
        "activated_without_deferred_count": activated_without_deferred_count,
        "same_step_activation_count": same_step_activation_count,
        "lane_exclusion_defer_reason_counts": dict(defer_reason_counts),
        "coverage_caveat": coverage_caveat,
    }
    return errors, summary


def print_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        print(f"{key}={value}")


def base_record() -> dict[str, Any]:
    record = {
        "execution_mode": "dual_batch_pearl",
        "dual_batch_enabled": True,
        "plan_phase": "steady",
        "normal_gamma": 4,
        "step_id": 1,
        "runner_role": "dual_draft",
        "enable_eager_lane_exclusion_dry_run": True,
        "eager_lane_exclusion_dry_run_enabled": True,
        "target_home_set": [0, 2],
        "draft_home_set": [1, 3],
        "original_draft_home_set": [1, 3],
        "actual_draft_home_set_for_normal_draft": [1, 3],
        "target_eager_set": [],
        "target_eager_set_dry_run": [],
        "scheduled_target_eager_set_dry_run": [],
        "excluded_from_actual_draft_home_for_eager": [],
        "excluded_from_draft_home_for_eager_dry_run": [],
        "lane_excluded_seq_ids": [],
        "lane_exclusion_decision_available_before_draft": False,
        "lane_exclusion_deferred_until_next_step": False,
        "lane_exclusion_defer_reason": None,
        "lane_exclusion_dry_run_done": False,
        "lane_exclusion_dry_run_done_proposal_ids": [],
        "deferred_lane_exclusion_created_proposal_ids": [],
        "deferred_lane_exclusion_created_seq_ids": [],
        "deferred_lane_exclusion_activated_proposal_ids": [],
        "deferred_lane_exclusion_activated_seq_ids": [],
        "deferred_lane_exclusion_dropped_proposal_ids": [],
        "deferred_lane_exclusion_drop_reason_by_proposal_id": {},
        "deferred_lane_exclusion_expired_proposal_ids": [],
        "deferred_lane_exclusion_created_step_by_proposal_id": {},
        "deferred_lane_exclusion_activated_step_by_proposal_id": {},
        "lane_exclusion_activation_source": "",
        "activated_without_deferred_count": 0,
        "target_deferred_decision_ids": [],
        "draft_deferred_decision_ids": [],
        "target_activated_decision_ids": [],
        "draft_activated_decision_ids": [],
        "target_lane_excluded_seq_ids": [],
        "draft_lane_excluded_seq_ids": [],
        "target_actual_draft_home_set_for_normal_draft": [],
        "draft_actual_draft_home_set_for_normal_draft": [1, 3],
        "lane_exclusion_decision_transfer_called": False,
        "lane_exclusion_decision_sent_proposal_ids": [],
        "lane_exclusion_decision_received_proposal_ids": [],
        "lane_exclusion_decision_shared_plan_id": None,
        "lane_exclusion_decision_shared_step_id": None,
        "normal_proposal_expected_seq_ids_after_lane_exclusion": [1, 3],
        "normal_proposal_sent_seq_ids_after_lane_exclusion": [1, 3],
        "normal_proposal_received_seq_ids_after_lane_exclusion": [1, 3],
        "adjusted_normal_proposal_expected_seq_ids": [1, 3],
        "adjusted_normal_proposal_received_seq_ids": [1, 3],
        "missing_normal_proposal_after_lane_exclusion": [],
        "missing_normal_proposal_handled_by_fallback": [],
        "eager_tokens_lane_excluded_dry_run": 0,
    }
    for field in ACTUAL_EAGER_COUNTERS:
        record[field] = 0
    return record


def actual_exclusion_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "step_id": 2,
            "actual_draft_home_set_for_normal_draft": [1],
            "adjusted_draft_home_set_dry_run": [1],
            "target_eager_set_dry_run": [3],
            "scheduled_target_eager_set_dry_run": [3],
            "scheduled_target_eager_proposal_ids_dry_run": [101],
            "scheduled_target_eager_seq_ids_dry_run": [3],
            "excluded_from_actual_draft_home_for_eager": [3],
            "excluded_from_draft_home_for_eager_dry_run": [3],
            "lane_excluded_seq_ids": [3],
            "lane_exclusion_decision_available_before_draft": True,
            "lane_exclusion_activation_source": "deferred_previous_step",
            "lane_exclusion_dry_run_done": True,
            "lane_exclusion_dry_run_done_proposal_ids": [101],
            "lane_exclusion_dry_run_done_seq_ids": [3],
            "deferred_lane_exclusion_activated_proposal_ids": [101],
            "deferred_lane_exclusion_activated_seq_ids": [3],
            "deferred_lane_exclusion_created_step_by_proposal_id": {"101": 1},
            "deferred_lane_exclusion_activated_step_by_proposal_id": {"101": 2},
            "draft_activated_decision_ids": [101],
            "draft_lane_excluded_seq_ids": [3],
            "draft_actual_draft_home_set_for_normal_draft": [1],
            "normal_proposal_expected_seq_ids_after_lane_exclusion": [1],
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [1],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [1],
            "adjusted_normal_proposal_expected_seq_ids": [1],
            "adjusted_normal_proposal_received_seq_ids": [1],
            "normal_proposal_missing_excluded_seq_ids": [3],
            "missing_normal_proposal_after_lane_exclusion": [],
            "missing_normal_proposal_handled_by_fallback": [3],
            "eager_tokens_lane_excluded_dry_run": 4,
        }
    )
    return record


def actual_exclusion_target_record() -> dict[str, Any]:
    record = actual_exclusion_record()
    record.update(
        {
            "runner_role": "dual_verify",
            "draft_activated_decision_ids": [],
            "draft_lane_excluded_seq_ids": [],
            "draft_actual_draft_home_set_for_normal_draft": [],
            "target_activated_decision_ids": [101],
            "target_lane_excluded_seq_ids": [3],
            "target_actual_draft_home_set_for_normal_draft": [1],
            "normal_proposal_sent_seq_ids_after_lane_exclusion": [],
            "normal_proposal_received_seq_ids_after_lane_exclusion": [1],
            "adjusted_normal_proposal_received_seq_ids": [1],
        }
    )
    return record


def late_decision_record() -> dict[str, Any]:
    record = base_record()
    record.update(
        {
            "step_id": 1,
            "target_eager_set_dry_run": [3],
            "scheduled_target_eager_set_dry_run": [3],
            "scheduled_target_eager_proposal_ids_dry_run": [101],
            "scheduled_target_eager_seq_ids_dry_run": [3],
            "lane_exclusion_deferred_until_next_step": True,
            "lane_exclusion_defer_reason": "decision_not_available_before_normal_draft",
            "lane_exclusion_decision_late_count": 1,
            "eager_lane_exclusion_proposal_ids": [101],
            "eager_lane_exclusion_seq_ids": [3],
            "deferred_lane_exclusion_created_proposal_ids": [101],
            "deferred_lane_exclusion_created_seq_ids": [3],
            "deferred_lane_exclusion_created_step_by_proposal_id": {"101": 1},
            "deferred_lane_exclusion_state_by_proposal_id": {"101": "DEFERRED_LANE_EXCLUSION"},
            "draft_deferred_decision_ids": [101],
            "lane_exclusion_decision_transfer_called": True,
            "lane_exclusion_decision_sent_proposal_ids": [101],
            "lane_exclusion_decision_received_proposal_ids": [101],
            "lane_exclusion_decision_shared_plan_id": 1,
            "lane_exclusion_decision_shared_step_id": 1,
        }
    )
    return record


def run_synthetic_tests() -> None:
    valid = [late_decision_record(), actual_exclusion_record(), actual_exclusion_target_record()]
    errors, _ = validate_records(valid)
    assert not errors, f"valid synthetic lane-exclusion records failed: {errors}"

    invalid = [deepcopy(actual_exclusion_record())]
    invalid[0]["normal_proposal_received_seq_ids_after_lane_exclusion"] = [1, 3]
    errors, _ = validate_records(invalid)
    assert any("received normal proposal seq ids" in error for error in errors), (
        "checker missed target receive using original draft set"
    )

    invalid = [deepcopy(late_decision_record())]
    invalid[0]["actual_draft_home_set_for_normal_draft"] = [1]
    errors, _ = validate_records(invalid)
    assert any("must not retroactively alter draft set" in error for error in errors), (
        "checker missed retroactive removal after late decision"
    )

    invalid = [deepcopy(actual_exclusion_record())]
    invalid[0]["normal_proposal_expected_seq_ids_after_lane_exclusion"] = [1, 3]
    errors, _ = validate_records(invalid)
    assert any("excluded seqs were still expected" in error for error in errors), (
        "checker missed excluded seq expected by receive"
    )

    invalid = [deepcopy(actual_exclusion_record())]
    invalid[0]["eager_tokens_verified"] = 4
    errors, _ = validate_records(invalid)
    assert any("actual eager counters" in error for error in errors), "checker missed actual eager counter"

    invalid = [deepcopy(actual_exclusion_record())]
    invalid[0]["target_eager_set"] = [3]
    errors, _ = validate_records(invalid)
    assert any("real target_eager_set" in error for error in errors), "checker missed real target eager set"

    invalid = [actual_exclusion_record(), actual_exclusion_target_record(), actual_exclusion_record()]
    errors, _ = validate_records(invalid)
    assert any("done repeated" in error for error in errors), "checker missed repeated proposal exclusion"

    print("Synthetic eager lane-exclusion dry-run checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 1H-5e eager lane-exclusion dry-run traces.")
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
    print("\nEager lane-exclusion dry-run trace check passed.")
    if summary.get("coverage_caveat"):
        print(summary["coverage_caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
