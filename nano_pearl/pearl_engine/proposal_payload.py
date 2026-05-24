from __future__ import annotations

"""Pure helpers for dual-batch normal/eager/conditional proposal message packaging."""


def _proposal_seq_ids(proposals) -> list[int]:
    return [int(proposal.seq_id) for proposal in proposals]


def encode_normal_proposals(proposals) -> list[int]:
    header = []
    tokens = []
    for proposal in proposals:
        to_verify = [int(token) for token in proposal.to_be_verified_token_ids]
        proposal_tokens = [int(token) for token in proposal.proposal_token_ids]
        header.extend(
            [
                int(proposal.seq_id),
                int(proposal.home_batch_id),
                int(proposal.base_len),
                int(proposal.pre_verify),
                int(len(to_verify)),
                int(proposal.proposal_len),
            ]
        )
        tokens.extend(to_verify)
        tokens.extend(proposal_tokens)
    return header + tokens


def encode_eager_proposals(proposals) -> list[int]:
    header = []
    tokens = []
    for proposal in proposals:
        eager_tokens = [int(token) for token in proposal.eager_token_ids]
        header.extend(
            [
                int(proposal.seq_id),
                int(proposal.home_batch_id),
                int(proposal.eager_len),
                int(proposal.eager_base_len),
                int(proposal.source_plan_id),
                int(proposal.source_step_id),
                int(proposal.source_home_batch_id),
            ]
        )
        tokens.extend(eager_tokens)
    return header + tokens


def build_combined_proposal_payload(
    *,
    normal_proposals,
    conditional_normal_proposals,
    eager_proposals,
    plan_id: int,
    step_id: int | None,
    draft_batch_id: int | None,
    gamma: int,
) -> dict:
    normal_payload = encode_normal_proposals(normal_proposals)
    conditional_payload = encode_normal_proposals(conditional_normal_proposals)
    eager_payload = encode_eager_proposals(eager_proposals)
    return {
        "kind": "combined",
        "plan_id": int(plan_id),
        "step_id": -1 if step_id is None else int(step_id),
        "draft_batch_id": -1 if draft_batch_id is None else int(draft_batch_id),
        "gamma": int(gamma),
        "normal_seq_ids": _proposal_seq_ids(normal_proposals),
        "conditional_normal_seq_ids": _proposal_seq_ids(conditional_normal_proposals),
        "eager_seq_ids": _proposal_seq_ids(eager_proposals),
        "normal_payload": normal_payload,
        "conditional_normal_payload": conditional_payload,
        "eager_payload": eager_payload,
        "flat_payload": normal_payload + conditional_payload + eager_payload,
    }
