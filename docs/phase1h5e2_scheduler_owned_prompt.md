# Agent Handoff Prompt: Phase 1H-5e2 Scheduler-Owned Lane Exclusion

Read first:
- docs/phase1h_project_book.md

You are working on branch:
phase1h5e2-scheduler-owned-lane-exclusion

Base commit:
6c5c0a2a4db88c90c9236843541581eb497085cd

Do not continue the runner-level deferred/drop design from dac1a671 as the main implementation. Use it only as reference.

Task:
Implement Phase 1H-5e2: scheduler-owned lane-exclusion planning.

Core architecture:

step N verify/apply or schedule stage:
    emit LaneExclusionDecision only

step N boundary:
    synchronize pending decisions between target and draft

step N+1 scheduler / StepPlan build:
    validate pending decisions
    produce lane_excluded_seq_ids
    produce actual_draft_home_set_for_normal_draft

step N+1 runner:
    execute only actual_draft_home_set_for_normal_draft

checker:
    verify decision lifecycle and actual normal proposal consistency

Hard rules:
- lane exclusion is a plan-time decision, not a runner-time patch;
- runner must not independently guess excluded seqs;
- runner only consumes StepPlan.actual_draft_home_set_for_normal_draft;
- no target eager verify;
- no permanent eager apply;
- no continuous eager;
- real target_eager_set remains empty;
- actual eager counters remain zero.

Implementation targets:
1. Add LaneExclusionDecision data model or equivalent metadata structure.
2. Add a pending decision buffer that is visible to StepPlan construction.
3. Emit LaneExclusionDecision when target-side ready/scheduled eager proposal is discovered.
4. Synchronize pending decisions across target/draft at step boundary.
5. Make StepPlan build consume pending decisions and produce:
   - original_draft_home_set
   - lane_excluded_seq_ids
   - actual_draft_home_set_for_normal_draft
   - applied_lane_exclusion_decision_ids
   - stale/dropped decision ids and reasons
6. Make normal draft execution use actual_draft_home_set_for_normal_draft.
7. Make target normal proposal expected seq ids use actual_draft_home_set_for_normal_draft.
8. Update checker to validate scheduler-owned semantics.

Avoid ambiguous trace fields:
- active_pending_decision_ids: only PENDING decisions
- terminal_decision_ids: DROPPED / EXPIRED / APPLIED
- touched_decision_ids: all decisions touched in current step

Do not compare active and terminal decisions together.

Acceptance criteria:
- default dual-batch unchanged;
- previous 1H dry-run checkers still pass;
- pending decisions are visible before StepPlan build;
- StepPlan produces lane_excluded_seq_ids > 0 if eligible;
- actual_draft_home_set_for_normal_draft excludes those seqs;
- normal proposal send/receive uses adjusted set;
- no normal proposal mismatch;
- real target_eager_set remains empty;
- actual eager counters remain zero.

Required CPU checks:
python -m py_compile \
  nano_pearl/pearl_engine/dual_batch.py \
  nano_pearl/pearl_engine/step_plan.py \
  nano_pearl/pearl_engine/sequence.py \
  nano_pearl/pearl_engine/pearl_model_runner.py \
  nano_pearl/pearl_config.py \
  benchmark/eval_multi_slo.py \
  benchmark/check_eager_lane_exclusion_dry_run.py \
  benchmark/check_eager_schedule_dry_run.py \
  benchmark/check_eager_transfer_dry_run.py \
  benchmark/check_eager_promotion_dry_run.py \
  benchmark/check_eager_draft_dry_run.py \
  benchmark/check_eager_plan_dry_run.py \
  benchmark/check_eager_scaffold.py \
  benchmark/check_dual_batch_trace.py

Run:
git diff --check

If GPU validation is not run, clearly say so.
