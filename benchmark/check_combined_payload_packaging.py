#!/usr/bin/env python3
"""CPU-only synthetic check for Phase 1H combined proposal packaging."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util


def load_payload_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "nano_pearl" / "pearl_engine" / "proposal_payload.py"
    spec = importlib.util.spec_from_file_location("proposal_payload", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normal(seq_id: int):
    return SimpleNamespace(
        seq_id=seq_id,
        home_batch_id=1,
        pre_verify=False,
        to_be_verified_token_ids=[seq_id * 10],
        proposal_token_ids=[seq_id * 100, seq_id * 100 + 1],
        proposal_len=2,
    )


def eager(seq_id: int):
    return SimpleNamespace(
        seq_id=seq_id,
        home_batch_id=0,
        eager_token_ids=[seq_id * 1000],
        eager_len=1,
        eager_base_len=16,
        source_plan_id=24,
        source_step_id=4,
        source_home_batch_id=0,
    )


def main():
    payload_mod = load_payload_module()
    payload = payload_mod.build_combined_proposal_payload(
        normal_proposals=[normal(4), normal(6), normal(8)],
        eager_proposals=[eager(1)],
        plan_id=24,
        step_id=4,
        draft_batch_id=1,
        gamma=2,
    )

    assert payload["kind"] == "combined"
    assert payload["normal_seq_ids"] == [4, 6, 8], payload
    assert payload["eager_seq_ids"] == [1], payload
    assert payload["normal_payload"] != payload["eager_payload"], payload
    print("combined payload packaging check passed")
    print(f"normal_seq_ids={payload['normal_seq_ids']}")
    print(f"eager_seq_ids={payload['eager_seq_ids']}")


if __name__ == "__main__":
    raise SystemExit(main())
