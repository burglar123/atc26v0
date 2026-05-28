# Phase 1H Project Book: Eager / Continuous / Rolling Speculative Path for dual-batch PEARL

Last updated: 2026-05-28

Repository: `burglar123/atc26v0`
Working branch: `before1Eactual1D1prepare-phase1h5estartnew`

## 0. Current state

The project has moved far beyond the old Phase 1H-5e lane-exclusion restart point.

The current latest implemented phase is:

**Phase 1H-8b: guarded rolling depth-2 real commit**

Important recent commits:

* `05e6e81d26778b84248d6a1f41cc558775d1b3e5`

  * Phase 1H-8a rolling continuous overlap dry-run.
* `d36553e9bb7a576fd6adab32b210d92f25bf0e52`

  * Phase 1H-8a parent-resolution fix.
* `a187b347bcbedc5756409ad31a5ac9a40f1b2245`

  * Phase 1H-8b guarded rolling depth-2 real commit.

The immediate next task is **not** to rewrite runtime logic. The immediate next task is:

**Phase 1H-8b-checker-fix: upgrade `check_rolling_continuous_eager_dry_run.py` from record-local validation to 8b-compatible global lifecycle validation.**

Current observed state:

* 8b runtime already produces rolling depth-2 real commits.
* The new 8b checker `check_rolling_continuous_depth2_commit_ready_only.py` passes.
* `check_eager_performance_accounting.py` passes.
* `check_multislo_result.py` passes.
* The runner still fails because the older 8a rolling checker `check_rolling_continuous_eager_dry_run.py` assumes parent full-accept evidence must appear in the same record as child-ready evidence.

The current failure should be treated as a checker compatibility issue unless further inspection proves a runtime trace-field deficiency.

---

## 1. Research goal

This project extends nano-PEARL dual-batch speculative decoding with an eager / continuous / rolling eager path.

Baseline dual-batch PEARL:

* draft model generates normal proposals for `draft_home_set`;
* target model verifies normal proposals for `target_home_set`;
* draft and target alternate over disjoint home sets.

New eager / continuous / rolling path:

* one-shot eager proposal `P0` can be committed if full-accepted;
* continuous depth-1 proposal `P1` can be generated after `P0` and committed if full-accepted;
* rolling depth-2 child proposal `P2` can be drafted while `P1` is being verified;
* if `P1` full-accepts, `P2` becomes ready shadow;
* if `P1` partial/rejects, `P2` is invalidated;
* 8b allows ready-shadow `P2` to real commit under strict guards;
* depth > 2 remains forbidden for now.

Long-term goal:

**Build a PEARL-style rolling continuous speculative pipeline where target verifies eager proposal `Pk` while draft speculatively drafts `P{k+1}`, with safe chain dependency and cascade discard.**

---

## 2. Core correctness invariants

### 2.1 Normal lane and eager lane cannot advance the same sequence frontier independently

A sequence must not be advanced by both the normal draft lane and eager / rolling lane in incompatible ways.

If an eager or rolling proposal reserves `seq_id = S`, then `S` must be excluded from the normal draft home set before normal draft execution.

Correct behavior:

```text
original_draft_home_set = [1, 3, 5, 7]
ready eager seq = [1]
actual_draft_home_set_for_normal_draft = [3, 5, 7]
```

Incorrect behavior:

```text
normal draft executes [1, 3, 5, 7]
eager lane also advances [1]
```

This causes stale base length, proposal conflict, frontier overshoot, and target/draft divergence.

### 2.2 Same-seq overlap is allowed only inside rolling eager lanes

For rolling continuous eager, this is allowed:

```text
same seq_id:
  target rolling eager verify lane: verify Pk
  draft rolling eager draft lane: draft P{k+1}
```

The same `seq_id` can appear in both rolling eager lanes in the same step, but:

* proposal IDs must differ;
* `P{k+1}.parent_proposal_id == Pk.proposal_id`;
* `P{k+1}.chain_depth == Pk.chain_depth + 1`;
* the sequence must not also appear in a conflicting normal lane.

This is the central PEARL-style rolling behavior.

### 2.3 Parent-child dependency is mandatory

For every rolling child proposal:

```text
P0 -> P1 -> P2 -> ...
```

The child is valid only if its parent is valid.

Rules:

* child parent must exist;
* child and parent must have the same `seq_id`;
* child depth must equal parent depth + 1;
* child root must match parent root;
* child base/frontier must match parent accepted frontier;
* child cannot become ready unless parent full-accepts;
* child cannot commit unless parent is full-accept / ready / real-committed;
* child must be invalidated if parent partial/rejects.

### 2.4 Cascade discard

If a parent proposal is:

* partial-accepted;
* rejected;
* invalidated;
* stale;
* expired;
* frontier-mismatched;
* missing;
* not committed when required;

then all known descendants must be invalidated or cascade-discarded.

For current depth-2-only 8b:

* direct `P2` child invalidation does not necessarily count as cascade discard;
* cascade discard remains reserved for descendants beyond the direct failed child;
* depth > 2 remains disabled.

### 2.5 Current depth boundary

Current allowed real commits:

```text
P0 one-shot real commit: allowed
P1 continuous depth-1 real commit: allowed
P2 rolling depth-2 real commit: allowed in 8b with flag
P3 or deeper: forbidden
```

Hard safety expectations:

```text
rolling_depth3_real_commit_count = 0
rolling_depth_gt2_real_commit_count = 0
```

Partial commit remains forbidden.

---

## 3. Phase history summary

### Phase 1H-5e3: scheduler-owned lane exclusion

Main architecture correction:

* carry ready proposal opportunities across steps;
* apply lane exclusion during `StepPlan` construction;
* keep apply records local to the plan;
* do not maintain runner-level pending lane-exclusion decisions.

Important semantic shift:

```text
carry opportunities, not pending decisions
```

This phase solved the earlier confusion from runner-level deferred lane-exclusion state.

### Phase 1H-5f to 1H-5j: dry-run target verify / apply / result-transfer / sync-apply / commit-readiness

Implemented a staged dry-run path:

* target eager verify dry-run;
* target apply dry-run;
* result-transfer dry-run;
* draft-side sync-apply dry-run;
* commit-readiness audit.

These phases preserved no-mutation semantics while proving transfer and validation plumbing.

### Phase 1H-6a: one-shot real commit

Implemented guarded real commit for one-shot eager proposals.

Observed validated result after fixes:

```text
eager committed proposals = 5
eager committed tokens = 20
target actual eager verified = 20
draft actual eager verified = 20
no repeated commit
no unexpected missing normal proposals
```

### Phase 1H-6b / 6c / 6d: regression, accounting, diagnosis

Added:

* regression matrix;
* commit-ready-only checks;
* performance accounting;
* overhead diagnosis;
* candidate / ready / committed token accounting;
* goodput / TPOT comparison;
* timing availability warnings.

Important conclusion:

* one-shot eager commit coverage was too small;
* committed token share was very low;
* fixed overhead dominated;
* this motivated continuous and rolling eager.

### Phase 1H-7a: continuous eager shadow prototype

Implemented conservative continuous shadow generation.

Outcome:

* continuous candidates were generated;
* no real continuous commit;
* depth-1 shadow existed but verify/apply was not executed.

### Phase 1H-7b: continuous verify/apply dry-run

Implemented continuous depth-1 verify/apply dry-run.

Observed minimum GPU result:

```text
baseline_one_shot:
  one_shot_committed_tokens = 20
  continuous_verified_tokens = 0
  continuous_ready_shadow_tokens = 0

continuous_depth1_verify:
  one_shot_committed_tokens = 20
  continuous_candidate_tokens = 20
  continuous_verified_tokens = 20
  continuous_full_accept_tokens = 16
  continuous_commit_ready_shadow_tokens = 16
```

This proved continuous depth-1 candidates could be verified and become ready shadow.

### Phase 1H-7c: guarded continuous depth-1 real commit

Implemented real commit for continuous depth-1 ready proposals.

Observed result:

```text
one_shot_committed_tokens = 20
continuous_shadow_ready_tokens = 12
continuous_real_committed_tokens = 12
combined_real_committed_tokens = 32
continuous_depth2_real_commit_count = 0
```

This established the first real continuous commit beyond one-shot.

### Phase 1H-7c timing instrumentation

Added scoped timing fields:

* `continuous_eager_commit_decision_broadcast_time_ms`
* `continuous_eager_result_transfer_time_ms`
* `continuous_eager_sync_apply_dry_run_time_ms`
* `continuous_eager_commit_time_ms`
* `continuous_eager_verify_apply_dry_run_time_ms`

Observed timing breakdown:

```text
total_continuous_overhead_time_ms ≈ 267 ms
continuous_eager_verify_apply_dry_run_time_ms ≈ 158 ms
continuous_eager_commit_decision_broadcast_time_ms ≈ 42 ms
continuous_eager_result_transfer_time_ms ≈ 42 ms
continuous_eager_sync_apply_dry_run_time_ms ≈ 9 ms
continuous_eager_commit_time_ms ≈ 16 ms
```

Conclusion:

* verify/apply dry-run was the largest component;
* result-transfer payload was not the only bottleneck.

### Phase 1H-7d: overhead reduction / compact result transfer

Added `compact_v1` continuous result-transfer protocol.

Observed minimum GPU result:

```text
continuous_eager_result_transfer_protocol = compact_v1
continuous_eager_result_transfer_payload_len_units_before_compact = 310
continuous_eager_result_transfer_payload_len_units = 130
continuous_real_committed_tokens = 12
combined_real_committed_tokens = 32
```

Conclusion:

* compacting payload worked;
* timing did not improve much;
* fixed collective / Python overhead and verify/apply dry-run remained dominant;
* do not keep deep-diving payload compression at this point.

### Phase 1H-8a: rolling continuous overlap dry-run

Goal:

```text
target verifies P1 while draft drafts P2
```

8a added rolling shadow child generation:

* `P1` remains the continuous depth-1 verified/committed proposal;
* `P2` is drafted as `rolling_continuous_shadow` during the `P1` shadow window;
* `P2` resolves after parent `P1` result becomes available;
* if `P1` full-accepts, `P2` becomes ready shadow;
* if `P1` partial/rejects, `P2` is invalidated;
* depth > 1 real commit stayed zero in 8a.

Initial 8a minimum GPU result showed:

```text
rolling_child_candidate_tokens = 20
rolling_same_seq_overlap_count = 5
rolling_child_ready_shadow_tokens = 0
rolling_child_invalidated_count = 5
rolling_drop_reason_counts = {"parent_verify_pending": 5}
```

Diagnosis:

* overlap generation worked;
* parent resolution was not connected correctly;
* full-accept parents still left children dropped as `parent_verify_pending`.

### Phase 1H-8a-fix: parent resolution fix

Fixed rolling parent resolution.

After fix, minimum result became:

```text
rolling_child_candidate_tokens = 20
rolling_child_ready_shadow_tokens = 12
rolling_child_invalidated_count = 2
rolling_parent_full_accept_count = 3
rolling_parent_partial_reject_count = 2
rolling_parent_resolution_pending_count = 0
rolling_drop_reason_counts = {"parent_partial_accept": 2}
```

This proved:

* parent full-accept -> child ready shadow;
* parent partial/reject -> child invalidated with concrete reason;
* `parent_verify_pending` no longer appeared as a final reason for resolved parents.

### Phase 1H-8a full matrix

Validated cases:

```text
baseline_7d_depth1:
  pass
  combined_real_committed_tokens = 32
  rolling_child_candidate_tokens = 0
  rolling_child_ready_shadow_tokens = 0
  rolling_same_seq_overlap_count = 0
  rolling_depth2_real_commit_count = 0

rolling_depth2_shadow:
  pass
  combined_real_committed_tokens = 32
  rolling_child_candidate_tokens = 20
  rolling_child_ready_shadow_tokens = 12
  rolling_same_seq_overlap_count = 5
  rolling_depth2_real_commit_count = 0

rolling_depth2_pressure_shadow:
  pass
  combined_real_committed_tokens = 44
  rolling_child_candidate_tokens = 28
  rolling_child_ready_shadow_tokens = 16
  rolling_same_seq_overlap_count = 7
  rolling_depth2_real_commit_count = 0

rolling_gamma2_depth2_shadow:
  pass
  combined_real_committed_tokens = 24
  rolling_child_candidate_tokens = 16
  rolling_child_ready_shadow_tokens = 8
  rolling_same_seq_overlap_count = 8
  rolling_depth2_real_commit_count = 0
```

Interpretation:

8a proved:

* P2 child generation;
* same-seq P1 target-verify / P2 draft overlap;
* parent full-accept -> child ready shadow;
* parent partial/reject -> child invalidation;
* normal lane conflict remained zero;
* depth-2 real commit remained disabled.

### Phase 1H-8b: guarded rolling depth-2 real commit

Added:

```text
--enable-rolling-continuous-depth2-commit-ready-only
```

Goal:

* commit only ready-shadow P2 children;
* require full-accept / committed parent P1;
* keep depth > 2 forbidden;
* keep partial commit forbidden;
* preserve one-shot and depth-1 semantics.

Current 8b minimum result:

#### baseline_8a_shadow

```text
check_status = pass
depth2_commit_enabled = False
one_shot_committed_tokens = 20
continuous_depth1_real_committed_tokens = 12
rolling_child_candidate_tokens = 20
rolling_child_ready_shadow_tokens = 12
rolling_depth2_real_committed_tokens = 0
combined_real_committed_tokens = 32
rolling_depth2_real_commit_count = 0
rolling_depth_gt2_real_commit_count = 0
rolling_parent_full_accept_count = 3
rolling_parent_partial_reject_count = 2
rolling_parent_resolution_pending_count = 0
rolling_drop_reason_counts = {"parent_partial_accept": 2}
```

#### rolling_depth2_commit

Runner currently fails at:

```text
check_status = failed:benchmark/check_rolling_continuous_eager_dry_run.py
```

But the actual depth-2 commit path shows:

```text
one-shot committed tokens = 16
continuous depth-1 real committed tokens = 12
rolling depth-2 real committed tokens = 12
combined real committed tokens = 40

rolling_depth2_real_committed_proposal_count = 3
rolling_depth2_target_actual_verified_token_increment_sum = 12
rolling_depth2_draft_actual_verified_token_increment_sum = 12

rolling_depth3_real_commit_count = 0
rolling_depth_gt2_real_commit_count = 0
rolling_normal_lane_conflict_count = 0
missing_buffered_proposal_unexpected_count = 0
```

Passing checkers in the 8b minimum run:

```text
check_eager_commit_ready_only.py passed
check_continuous_eager_dry_run.py passed
check_continuous_eager_commit_depth1_ready_only.py passed
check_rolling_continuous_depth2_commit_ready_only.py passed
check_eager_performance_accounting.py passed
check_multislo_result.py passed
```

Failing checker:

```text
check_rolling_continuous_eager_dry_run.py
```

Error:

```text
record[42] rolling child 900000602 ready without full-accept parent
record[66] rolling child 900001402 ready without full-accept parent
record[72] rolling child 900001502 ready without full-accept parent
```

---

## 4. Current failure interpretation

Current failure is likely **checker compatibility**, not runtime semantic failure.

### 4.1 Why the old rolling checker fails

`check_rolling_continuous_eager_dry_run.py` still validates ready child -> full-accept parent in a record-local way.

It builds `full_accept_parent_ids` per record from:

```text
rolling_parent_full_accept_proposal_ids
continuous_eager_full_accept_proposal_ids
continuous_eager_commit_ready_shadow_proposal_ids
continuous_eager_real_committed_proposal_ids
continuous_eager_verify_result_by_proposal_id
continuous_eager_sync_apply_*_verify_result_by_proposal_id
continuous_eager_real_commit_verify_result_by_proposal_id
```

Then it checks every ready child in the same record:

```text
if child_id in ready_ids:
    parent_id must be in same-record full_accept_parent_ids
```

This was acceptable for 8a shadow traces, but 8b introduces real depth-2 commit records and target/draft side records. In 8b, evidence may be distributed:

```text
child ready evidence: record A
parent full-accept evidence: record B
depth2 real commit evidence: record C
```

So the checker reports:

```text
ready without full-accept parent
```

even though the new depth-2 checker can verify globally that the committed child has a full-accept parent.

### 4.2 Why this is likely not a runtime bug

The new 8b checker `check_rolling_continuous_depth2_commit_ready_only.py` uses global lifecycle aggregation. It verifies:

* committed P2 is ready shadow;
* committed P2 has full-accept parent;
* committed P2 is not invalidated;
* committed P2 is not cascade-discarded;
* committed proposal depth is 2;
* verify result is full-accept;
* action is real commit append;
* target/draft length and token checks pass;
* depth3/depth>2 remain zero.

This checker passes on the current 8b minimum trace.

Performance accounting also shows:

```text
rolling_depth2_real_committed_proposal_count = 3
rolling_depth2_real_committed_token_count = 12
rolling_depth2_commit_rate_by_token = 1.0
rolling_depth2_target_actual_verified_token_increment_sum = 12
rolling_depth2_draft_actual_verified_token_increment_sum = 12
combined_real_committed_token_count = 40
rolling_depth3_real_commit_count = 0
rolling_depth_gt2_real_commit_count = 0
```

Therefore the next fix should start from the checker, not from runtime.

---

## 5. Immediate next task

### Phase 1H-8b-checker-fix

Upgrade `benchmark/check_rolling_continuous_eager_dry_run.py` from record-local validation to 8b-compatible global lifecycle validation.

Do not rewrite runtime first.

### Required changes

#### 5.1 Add a first global pass

Collect global lifecycle evidence across all records:

```text
global_parent_full_ids
global_parent_partial_ids
global_parent_resolved_ids

global_ready_child_ids
global_invalidated_child_ids
global_cascade_ids

global_parent_by_child
global_depth_by_id
global_root_by_id
global_status_by_id
global_status_reason_by_id
global_invalid_reason_by_id

global_depth2_committed_child_ids
global_depth2_commit_enabled
```

Parent full evidence should include:

```text
rolling_parent_full_accept_proposal_ids
continuous_eager_full_accept_proposal_ids
continuous_eager_commit_ready_shadow_proposal_ids
continuous_eager_real_committed_proposal_ids
continuous_eager_verify_result_by_proposal_id == full_accept
continuous_eager_sync_apply_*_verify_result_by_proposal_id == full_accept
continuous_eager_real_commit_verify_result_by_proposal_id == full_accept
rolling_depth2_real_commit_parent_by_proposal_id values if the committed child has passed the new depth2 checker semantics
```

Parent partial/reject evidence should include:

```text
rolling_parent_partial_reject_proposal_ids
continuous_eager_partial_reject_proposal_ids
verify_result != full_accept and not unknown/not_executed
```

#### 5.2 Validate ready children globally

Replace strict same-record logic with global lifecycle logic.

Correct rule:

```text
If child is ready anywhere in the trace,
then its parent must be full-accept somewhere in the trace.
```

Do not require that the parent full-accept evidence appears in the same record.

Still fail if:

```text
child is ready but parent is not full-accept anywhere
child is ready and invalidated
child is ready and cascade-discarded
child has missing parent
child depth != parent depth + 1
child root != parent root
```

#### 5.3 Keep 8a bad-case detection

The checker must still fail the old bad 8a case:

```text
parent is full-accept / real-committed,
but child is dropped with parent_verify_pending
```

So this condition remains invalid globally:

```text
child_reason == parent_verify_pending
and parent_id in global_parent_full_ids or global_parent_partial_ids
```

#### 5.4 Keep 8b safety

For 8b traces:

* allow depth-2 real commit only when depth2 commit flag is enabled;
* forbid depth3/depth>2;
* do not treat side-record count as unique proposal count;
* do not require every record to repeat full parent evidence;
* keep normal lane conflict zero;
* keep unexpected missing normal proposals zero.

#### 5.5 Fix duplicate token accounting in the old rolling checker

Current old rolling checker may report:

```text
rolling_child_ready_shadow_proposal_count = 3
rolling_child_ready_shadow_token_count = 24
```

But accounting correctly reports:

```text
rolling_child_ready_shadow_proposal_count = 3
rolling_child_ready_shadow_token_count = 12
```

Cause:

* old checker accumulates `rolling_child_ready_shadow_token_count` record-by-record;
* 8b trace may contain target/draft side duplicates.

Fix:

* compute ready shadow token count from unique ready proposal IDs;
* use `rolling_child_token_count_by_proposal_id` if available;
* otherwise use proposal length / gamma fallback;
* do not double-count target/draft side rows.

#### 5.6 Clarify depth2 count semantics

Old checker reports:

```text
rolling_depth2_real_commit_count = 6
```

This is likely side-record count:

```text
3 committed proposals × 2 sides = 6
```

Do not confuse this with unique committed proposal count.

Prefer summary fields:

```text
rolling_depth2_real_committed_proposal_count
rolling_depth2_real_committed_token_count
```

If `check_rolling_continuous_eager_dry_run.py` keeps `rolling_depth2_real_commit_count`, document or treat it as side-record count.

---

## 6. Do-not-do list

Do not:

* rewrite depth-2 runtime commit path unless checker fix proves trace evidence is insufficient;
* change one-shot commit semantics;
* change depth-1 continuous commit semantics;
* implement depth-3;
* implement unbounded rolling;
* implement partial commit;
* disable `check_rolling_continuous_eager_dry_run.py` in the 8b runner;
* weaken the new depth-2 checker;
* hide errors by ignoring ready-parent validation.

---

## 7. Validation commands

### CPU

```bash
python3 -m py_compile \
  benchmark/check_rolling_continuous_eager_dry_run.py \
  benchmark/check_rolling_continuous_depth2_commit_ready_only.py \
  benchmark/check_eager_performance_accounting.py \
  benchmark/run_phase1h8b_rolling_depth2_commit.py \
  benchmark/check_continuous_eager_commit_depth1_ready_only.py \
  benchmark/check_continuous_eager_dry_run.py

git diff --check
```

### Synthetic

```bash
python3 benchmark/check_rolling_continuous_eager_dry_run.py --synthetic
python3 benchmark/check_rolling_continuous_depth2_commit_ready_only.py --synthetic
python3 benchmark/check_eager_performance_accounting.py --synthetic
```

### GPU validation after checker fix

```bash
python3 benchmark/run_phase1h8b_rolling_depth2_commit.py \
  --draft-model "$DRAFT_MODEL" \
  --target-model "$TARGET_MODEL" \
  --workload-in "$WORKLOAD" \
  --out-root results/multislo/phase1h8b_depth2_commit_min_fix \
  --cases baseline_8a_shadow,rolling_depth2_commit
```

Expected:

```text
baseline_8a_shadow:
  check_status = pass
  rolling_depth2_real_committed_tokens = 0
  combined_real_committed_tokens ≈ 32

rolling_depth2_commit:
  check_status = pass
  rolling_depth2_real_committed_tokens > 0
  expected around 12 in current min workload
  combined_real_committed_tokens > baseline_8a_shadow
  expected around 40 in current min workload
  rolling_depth_gt2_real_commit_count = 0
```

Then run:

```bash
TRACE=results/multislo/phase1h8b_depth2_commit_min_fix/rolling_depth2_commit/engine_trace.json
RESULT=results/multislo/phase1h8b_depth2_commit_min_fix/rolling_depth2_commit/result.json

python3 benchmark/check_eager_commit_ready_only.py "$TRACE"
python3 benchmark/check_continuous_eager_dry_run.py "$TRACE"
python3 benchmark/check_continuous_eager_commit_depth1_ready_only.py "$TRACE"
python3 benchmark/check_rolling_continuous_eager_dry_run.py "$TRACE"
python3 benchmark/check_rolling_continuous_depth2_commit_ready_only.py "$TRACE"
python3 benchmark/check_eager_performance_accounting.py "$TRACE" "$RESULT"
python3 benchmark/check_multislo_result.py "$RESULT"
```

Pass criteria:

```text
check_rolling_continuous_eager_dry_run.py passes
check_rolling_continuous_depth2_commit_ready_only.py passes
check_eager_performance_accounting.py passes
check_multislo_result.py passes

rolling_depth2_real_committed_token_count > 0
combined_real_committed_token_count > baseline_8a_shadow
rolling_depth_gt2_real_commit_count = 0
rolling_normal_lane_conflict_count = 0
missing_buffered_proposal_unexpected_count = 0
```

---

## 8. Next phase after 8b checker fix

Only after the 8b minimum and full matrix pass should the project proceed.

Likely next phase:

**Phase 1H-8c: bounded rolling depth-3 shadow / dry-run**

Do not jump directly to unbounded rolling. The next safe step should mirror the 8a/8b pattern:

```text
8c: generate P3 as shadow while target verifies/commits P2
8d: guarded real commit for P3 only if 8c passes
```

Depth-3 real commit must remain disabled until the depth-3 shadow parent-child lifecycle passes.
