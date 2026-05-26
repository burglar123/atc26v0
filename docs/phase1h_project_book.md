# Phase 1H Project Book: Eager Speculative Path for dual-batch PEARL

## 0. Current branch and baseline

Project repository: atc26v0

Current restart base:
- Commit: 6c5c0a2a4db88c90c9236843541581eb497085cd
- Phase: 1H-5e initial lane-exclusion dry-run
- Intended new direction: scheduler-owned lane-exclusion planning

Do not continue from dac1a671 as the main implementation base. That commit is useful as an experiment/reference, but it implemented runner-level deferred/drop activation and exposed architectural issues.

## 1. Research goal

This project extends nano-PEARL dual-batch speculative decoding with an eager child-proposal path.

Baseline dual-batch PEARL:
- draft model generates normal proposals for draft_home_set;
- target model verifies normal proposals for target_home_set;
- draft and target alternate over disjoint home sets.

New eager path:
- while target verifies a normal parent proposal, draft may generate eager child proposals for selected post-verify target_home_set sequences;
- if the parent normal proposal is full-accepted, the eager child proposal may become valid;
- later, the eager proposal can be scheduled, verified, and eventually committed.

The long-term goal is to reduce pipeline bubbles and improve decoding throughput/capacity.

## 2. Key correctness principle

A sequence must not be advanced by normal draft lane and eager lane at the same time.

If a ready/scheduled eager proposal reserves seq_id S, then S must be excluded from normal draft_home_set before normal draft execution.

Correct behavior:
- original_draft_home_set = [1,3,5,7]
- ready eager seq = [1]
- actual_draft_home_set_for_normal_draft = [3,5,7]

Incorrect behavior:
- normal draft executes [1,3,5,7]
- eager lane also reserves [1]

This causes proposal conflict, stale base_len, draft frontier overshot, and target/draft state divergence.

## 3. Pre/post verify semantics

pre_verify / post_verify is a Sequence state-machine property.

Rules:
- Only post-verify sequences, i.e. seq.pre_verify == False, can be used as stable eager proposal base.
- draft_eager_set must only include safe post-verify candidates.
- eager proposal base_pre_verify must be False.
- If a deferred lane-exclusion decision later sees seq.pre_verify == True, it must not activate.
- If seq is finished, span invalidated, base overshot, or pre_verify, the pending decision must be dropped or kept pending with explicit reason.

Lane exclusion itself does not change Sequence tokens or pre_verify state.
It only changes which seqs are run by normal draft.

## 4. Completed phases

### Phase 1H-1: eager plan dry-run
Select draft_eager_set only. No eager generation.
Expected:
- draft_eager_set may be non-empty;
- target_eager_set remains empty;
- actual eager counters are zero.

### Phase 1H-2: eager draft dry-run with rollback
Generate eager child proposal on draft side, then immediately rollback.
Expected:
- eager_tokens_generated > 0;
- rollback_ok;
- no target verify;
- no mutation remains.

### Phase 1H-3: promotion/discard dry-run
Classify eager proposals based on parent normal verification.
Expected:
- parent full accept -> promoted;
- reject/partial/finished/pre_verify/base mismatch -> discarded;
- promoted + discarded == generated;
- target_eager_set remains empty.

### Phase 1H-4 / 1H-4b: eager proposal transfer and pending-base handling
Transfer promoted eager proposal from draft to target.
Handle:
- base reached -> ready;
- current_len < base_len -> pending or invalidated;
- span invalidated -> drop;
- request finished -> drop.
Observed:
- transfer works;
- span invalidation classification was fixed;
- healthy pending path often not exercised by current workload.

### Phase 1H-4c: ready eager scheduling dry-run
Target side schedules READY eager proposals into target_eager_set_dry_run.
Actual target_eager_set remains empty.
If ready proposal intersects current target_home_set, defer.
If it intersects draft_home_set, schedule dry-run and compute adjusted_draft_home_set_dry_run.
Expected:
- scheduled_target_eager_set_dry_run > 0;
- adjusted_draft_home_intersection_count = 0;
- real target_eager_set remains empty.

### Phase 1H-5a: target eager verify dry-run
Target verifies scheduled eager proposal without applying.
Expected:
- eager_tokens_verify_dry_run > 0;
- actual eager_tokens_verified/accepted/rejected/invalidated remain zero;
- no sequence mutation.

### Phase 1H-5b: target eager apply dry-run with rollback
Target simulates eager apply:
- full accept -> append then rollback;
- partial/reject -> discard-only.
Expected:
- rollback_ok;
- mutation_remaining = 0;
- actual eager counters remain zero.

### Phase 1H-5c: eager verify-result transfer dry-run
Target sends eager verify dry-run result metadata to draft.
Draft receives and validates only; no apply.
Important lesson:
- New collectives can break normal proposal transfer if not fixed-order and zero-safe.

### Phase 1H-5d: synchronized apply dry-run
Target and draft both simulate apply/discard with rollback.
Expected:
- target and draft execute same eager result;
- rollback ok on both sides;
- actual eager counters remain zero.
Observation:
- expected_normal_draft_conflict > 0.
This exposed the need for actual normal-lane exclusion.

### Phase 1H-5e initial: lane-exclusion dry-run
Attempted to make actual normal draft set exclude scheduled eager seqs.
Result:
- safe but only coverage caveat;
- scheduled_total > 0;
- excluded_total = 0;
- decisions were late and deferred.

### Phase 1H-5e-fix runner-level attempt
Commit dac1a671 introduced:
- runner-level deferred lane-exclusion buffer;
- target-to-draft decision transfer;
- runner-level activation/drop.
Result:
- decision transfer appeared to work;
- checker/trace active-vs-terminal decision semantics became confusing;
- actual activation still did not occur;
- deferred entries often dropped by draft_base_overshot_before_lane_exclusion, pre_verify, or span invalidation.
Conclusion:
- do not continue this as the main architecture;
- use it only as reference.

## 5. New architecture direction: scheduler-owned lane-exclusion planning

The correct architecture should be:

step N verify/apply
    |
    | emit LaneExclusionDecision
    v
step N boundary transfer
    |
    | synchronized pending decisions
    v
step N+1 scheduler / StepPlan build
    |
    | validate pending decisions
    | produce lane_excluded_seq_ids
    | produce actual_draft_home_set_for_normal_draft
    v
step N+1 runner
    |
    | execute only actual_draft_home_set_for_normal_draft
    v
checker
    |
    | verify lifecycle + actual execution consistency

Key principle:
lane exclusion should be a plan-time decision, not a runner-time patch.

## 6. Responsibility split

verify/apply or schedule stage:
- only discovers ready/scheduled eager proposals;
- emits LaneExclusionDecision;
- does not modify draft_home_set.

boundary transfer / buffer:
- synchronizes pending decisions between target and draft;
- zero-decision safe;
- fixed-order if collective is used.

scheduler / StepPlan:
- owns validation of pending decisions;
- decides which decisions apply to this step;
- emits:
  - original_draft_home_set;
  - lane_excluded_seq_ids;
  - actual_draft_home_set_for_normal_draft;
  - applied/stale/drop reasons.

runner:
- consumes actual_draft_home_set_for_normal_draft only;
- does not independently guess lane exclusions;
- sends normal proposals only for the adjusted set;
- target expects normal proposals only for the adjusted set.

checker:
- validates decision lifecycle;
- validates target/draft synchronized pending decisions;
- validates actual execution consistency.

## 7. LaneExclusionDecision schema

A LaneExclusionDecision should contain:
- proposal_id
- seq_id
- request_id if available
- source_plan_id
- source_step_id
- created_step_id
- base_len
- base_pre_verify
- proposal_len
- to_verify_len
- proposal_token_ids or token length metadata
- reason = ready_eager_scheduled
- state:
  - PENDING
  - APPLIED
  - STALE_DROPPED
  - EXPIRED

## 8. StepPlan fields

StepPlan should include:
- original_draft_home_set
- lane_excluded_seq_ids
- actual_draft_home_set_for_normal_draft
- pending_lane_exclusion_decision_ids_before_plan
- applied_lane_exclusion_decision_ids
- stale_lane_exclusion_decision_ids
- lane_exclusion_drop_reason_by_decision_id
- normal_proposal_expected_seq_ids_after_lane_exclusion

## 9. Validation rules at StepPlan build

For each pending LaneExclusionDecision:

Apply only if:
- plan phase is steady;
- seq_id is in candidate draft_home_set;
- seq_id is not in candidate target_home_set;
- seq exists and is running;
- request is not finished;
- seq.pre_verify == False;
- proposal.base_pre_verify == False;
- proposal_len == gamma;
- to_verify_len == gamma;
- span not invalidated;
- current length is compatible with base_len.

If seq_id is still in target_home_set:
- keep pending if not expired;
- reason = still_in_target_home.

If seq_id in neither set:
- keep pending or drop with explicit reason.

If seq.pre_verify == True:
- drop with seq_returned_pre_verify_before_lane_exclusion.

If len(seq) > base_len:
- drop with draft_base_overshot_before_lane_exclusion.

If span invalidated:
- drop with seq_span_invalidated_before_lane_exclusion.

If max age exceeded:
- expire.

## 10. Execution invariants

When lane exclusion is applied:
- actual_draft_home_set_for_normal_draft = original_draft_home_set - lane_excluded_seq_ids;
- draft normal execution uses actual_draft_home_set_for_normal_draft;
- normal proposal construction uses actual_draft_home_set_for_normal_draft;
- target expected normal proposals use actual_draft_home_set_for_normal_draft;
- missing proposal for excluded seq is correct;
- target_home_set is unchanged;
- real target_eager_set is empty in this dry-run phase;
- actual eager verified/accepted/rejected/invalidated counters remain zero.

## 11. Checker requirements

The checker should validate:

Default path:
- no lane exclusion;
- real target_eager_set empty;
- eager counters zero.

Lane-exclusion planning:
- pending decisions are synchronized before StepPlan build;
- StepPlan uses pending decisions to compute lane_excluded_seq_ids;
- lane_excluded_seq_ids are not same-step late decisions;
- actual_draft_home_set_for_normal_draft matches original - excluded;
- draft sent normal proposals match actual set;
- target expected normal proposals match actual set;
- excluded seqs are not expected in normal receive;
- no target/draft divergence;
- no actual eager verify/apply/result-transfer required;
- actual eager counters remain zero.

Avoid ambiguous fields:
- active_pending_decision_ids: only PENDING;
- terminal_decision_ids: DROPPED / EXPIRED / APPLIED;
- touched_decision_ids: all decisions touched in this step.

Do not mix active and terminal ids in comparisons.

## 12. Next implementation phase

New phase name:
Phase 1H-5e2: scheduler-owned lane-exclusion planning

Goal:
Refactor lane-exclusion activation from runner-level deferred/drop logic into StepPlan construction.

Do not implement:
- target eager verification;
- permanent eager apply;
- continuous eager;
- full 1H-5f commit.

