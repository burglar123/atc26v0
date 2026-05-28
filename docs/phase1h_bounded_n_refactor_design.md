# Phase 1H-8f: Generic Bounded-N Rolling Refactor Readiness Design

## Status And Scope

Phase 1H-8e validated the current bounded N=3 rolling chain:

```text
P0 one-shot eager full-accept -> real commit
P1 continuous eager depth-1 full-accept -> real commit
P2 rolling continuous depth-2 full-accept -> real commit
P3 rolling continuous depth-3 full-accept -> real commit
```

The 8e stability regression passed with:

- `combined_accounting_ok = True`
- `target_draft_accounting_ok = True`
- `depth4_real_commit_count = 0`
- `depth_gt3_real_commit_count = 0`
- `normal_lane_conflict_count = 0`
- `missing_buffered_proposal_unexpected_count = 0`
- `duplicate_commit_count = 0`
- `invalid_committed_child_count = 0`
- `cascade_committed_child_count = 0`
- `parent_missing_committed_child_count = 0`

This phase is design/readiness only. It does not implement depth-4, unbounded rolling, partial commit, or any runtime semantic change. Existing depth-specific trace fields, checkers, runners, and accounting remain the compatibility surface until a later migration replaces their internals.

The current performance warnings remain important inputs for the generic design:

- `committed_token_share_below_0.01`
- `payload_bytes_unavailable`
- `proposal_payload_len_units_high_per_committed_token`
- `suppressed_slots_exceed_committed_tokens`

They indicate that the mechanism is semantically correct through N=3, but control-plane overhead and coverage must be managed before extending depth.

## A. Current Implementation Map

### Configuration And CLI

- `nano_pearl/pearl_config.py`
  - Defines feature flags for one-shot eager commit, continuous eager depth-1, rolling depth-2, rolling depth-3 shadow, and rolling depth-3 commit.
  - Performs cascading enablement, for example depth-3 commit implies depth-3 shadow, depth-3 shadow implies depth-2 commit, and depth-2 commit implies rolling eager.
  - Validates `max_rolling_continuous_depth`, including the current requirement that depth-3 modes require a configured max depth of at least 3.

- `benchmark/eval_multi_slo.py`
  - Owns user-facing CLI flags such as `--enable-continuous-eager-commit-depth1-ready-only`, `--enable-rolling-continuous-depth2-commit-ready-only`, `--enable-rolling-continuous-depth3-shadow-dry-run`, `--enable-rolling-continuous-depth3-commit-ready-only`, and `--max-rolling-continuous-depth`.
  - Passes feature settings into the engine runner and result metadata.

- `nano_pearl/pearl_engine/step_plan.py`
  - Carries eager and continuous eager planning flags through step-plan construction.
  - Emits plan-level trace metadata used by existing checker paths.

### Runtime

- `nano_pearl/pearl_engine/pearl_model_runner.py`
  - Defines source labels and payload constants:
    - one-shot eager takeover dry run
    - continuous eager dry run
    - rolling depth-2 commit
    - rolling depth-3 shadow
    - rolling depth-3 commit
  - Owns proposal registries and committed-proposal sets:
    - `_continuous_eager_committed_proposal_ids`
    - `_rolling_depth2_committed_proposal_ids`
    - `_rolling_depth3_committed_proposal_ids`
    - `_rolling_depth3_shadow_proposals_by_id`
  - Emits the depth-specific trace schema.
  - Implements helper gates:
    - `_continuous_eager_commit_depth1_ready_only_enabled`
    - `_rolling_depth2_commit_ready_only_enabled`
    - `_rolling_depth3_shadow_dry_run_enabled`
    - `_rolling_depth3_commit_ready_only_enabled`

### P0: One-Shot Eager Commit

Primary runtime behavior lives in `pearl_model_runner.py` one-shot eager takeover and target apply paths. Trace fields include:

- `eager_committed_token_count`
- `target_actual_eager_verified_token_increment_sum`
- `target_actual_eager_accepted_token_increment_sum`
- one-shot verify/action/result fields consumed by `benchmark/check_eager_commit_ready_only.py`

P0 acts as the root of the validated chain when later continuous/rolling proposals reference root evidence.

### P1: Continuous Eager Depth-1

Primary runtime paths in `pearl_model_runner.py` include:

- `_run_continuous_eager_shadow_dry_run`
- `_run_continuous_eager_draft_shadow_dry_run`
- `_send_continuous_eager_transfer_dry_run`
- `_run_continuous_eager_target_verify_apply_dry_run`
- `_send_continuous_eager_result_transfer_dry_run`
- `_run_continuous_eager_sync_apply_dry_run`
- `_send_continuous_eager_commit_depth1_decision`
- `_run_continuous_eager_commit_depth1_ready_only`

Trace and accounting are consumed by:

- `benchmark/check_continuous_eager_dry_run.py`
- `benchmark/check_continuous_eager_commit_depth1_ready_only.py`
- `benchmark/check_eager_performance_accounting.py`

### P2: Rolling Continuous Depth-2

Primary runtime paths in `pearl_model_runner.py` include:

- `_rolling_continuous_limits`
- `_set_rolling_common_trace`
- `_run_rolling_continuous_child_draft_overlap_dry_run`
- `_rolling_depth2_commit_decisions_from_trace`
- `_run_rolling_depth2_commit_ready_only`

Trace and accounting are consumed by:

- `benchmark/check_rolling_continuous_eager_dry_run.py`
- `benchmark/check_rolling_continuous_depth2_commit_ready_only.py`
- `benchmark/check_eager_performance_accounting.py`
- `benchmark/check_bounded_rolling_readiness_audit.py`

### P3: Rolling Continuous Depth-3 Shadow And Commit

Primary runtime paths in `pearl_model_runner.py` include:

- `_run_rolling_depth3_shadow_dry_run`
- `_run_rolling_depth3_commit_ready_only`

Trace and accounting are consumed by:

- `benchmark/check_rolling_continuous_depth3_shadow_dry_run.py`
- `benchmark/check_rolling_continuous_depth3_commit_ready_only.py`
- `benchmark/check_eager_performance_accounting.py`
- `benchmark/check_bounded_rolling_readiness_audit.py`

### Runner Summaries

- `benchmark/run_phase1h8d_rolling_depth3_commit.py` validates the guarded P3 real-commit path.
- `benchmark/run_phase1h8e_bounded_readiness_regression.py` runs the current N=3 stability matrix and includes `check_bounded_rolling_readiness_audit.py`.
- `benchmark/compare_phase1h8e_bounded_regression.py` prints compact summary comparisons.

## B. Repeated Pattern Inventory

The current implementation repeats the same logical phases across P1, P2, and P3 with depth-specific field names.

Repeated proposal identity fields:

- proposal id
- seq id
- parent proposal id
- root proposal id
- chain depth
- source
- side
- plan id
- step id

Repeated generation fields:

- generated proposal ids
- generated seq ids
- token count by proposal id
- base length by proposal id when available
- skip reason by proposal id

Repeated parent-resolution fields:

- parent full-accept ids
- parent real-committed ids
- parent skipped ids
- parent invalidated ids
- parent resolution pending ids
- parent pending count

Repeated lifecycle fields:

- ready shadow proposal ids
- ready shadow seq ids
- invalidated proposal ids
- invalidated reason by proposal id
- cascade-discarded proposal ids
- drop reason counts
- frontier mismatch counts

Repeated real-commit fields:

- commit candidate proposal ids
- precondition ok/failed maps
- real committed proposal ids
- real committed seq ids
- token count by committed proposal id
- accept length by committed proposal id
- action by proposal id
- verify result by proposal id
- parent/root/depth maps for committed proposals
- duplicate proposal ids
- duplicate seq ids

Repeated safety fields:

- normal lane conflict count
- missing buffered proposal unexpected count
- committed without ready shadow
- committed without parent commit/full accept
- committed invalidated child
- committed cascade-discarded child
- committed non-full-accept proposal
- target/draft length match
- target/draft token match
- max observed depth
- real commit count above allowed depth

Repeated accounting fields:

- committed proposal count
- committed token count
- target verified/accepted/rejected/invalidated increments
- draft verified/accepted/rejected/invalidated increments
- combined real committed token count
- combined actual verified token increment sum
- combined actual accepted token increment sum

Repeated tests:

- good parent -> child ready/commit
- missing parent -> fail
- parent partial/reject -> invalidate
- child committed without ready shadow -> fail
- child committed while invalidated/cascade-discarded -> fail
- duplicate side rows dedupe by proposal id
- distributed target/draft evidence validates globally

## C. Proposed Generic Data Structures

The runtime should eventually represent rolling proposals as data. The initial migration should use these models in read-only parsers/checkers before runtime paths depend on them.

### RollingProposalNode

```python
@dataclass
class RollingProposalNode:
    proposal_id: int
    seq_id: int
    depth: int
    parent_id: int | None
    root_id: int | None
    source: str
    status: str
    status_reason: str | None
    base_len: int | None
    token_span: tuple[int, ...] | None
    token_count: int
    accept_len: int | None
    verify_result: str | None
    apply_action: str | None
    commit_state: str
    side: str | None
    plan_id: int | None
    step_id: int | None
```

Required invariants:

- `depth >= 0`
- `depth == 0` for one-shot/root proposals when represented in the generic chain
- `parent_id is not None` for committed rolling proposals at depth greater than 0
- child depth is exactly `parent.depth + 1`
- committed full-accept-only paths have `token_count == accept_len`

### RollingChainRegistry

```python
@dataclass
class RollingChainRegistry:
    nodes_by_id: dict[int, RollingProposalNode]
    children_by_parent: dict[int, set[int]]
    committed_by_depth: dict[int, set[int]]
    ready_by_depth: dict[int, set[int]]
    invalidated_by_depth: dict[int, set[int]]
    cascade_by_depth: dict[int, set[int]]
    seq_frontier_by_seq: dict[int, int]
    root_by_proposal: dict[int, int]
    parent_by_proposal: dict[int, int]
```

The registry should support:

- idempotent ingestion from multiple target/draft trace records
- side-record dedupe by proposal id
- global parent lookup across steps
- global status resolution when evidence is distributed across records

### RollingDepthConfig

```python
@dataclass
class RollingDepthConfig:
    max_rolling_depth: int
    shadow_enabled_by_depth: dict[int, bool]
    commit_enabled_by_depth: dict[int, bool]
    per_depth_budget: dict[int, int]
    token_budget: int | None
    trace_level: str
```

Depth enablement must remain explicit. A generic loop must not infer permission to commit depth `k` from the existence of a proposal at depth `k`.

## D. Proposed Generic State Machine

States:

- `GENERATED`
- `VERIFY_PENDING`
- `VERIFY_RESULT_READY`
- `READY_SHADOW`
- `COMMIT_CANDIDATE`
- `COMMITTED`
- `INVALIDATED`
- `CASCADE_DISCARDED`
- `STALE`
- `FRONTIER_MISMATCH`
- `FINISHED`
- `SKIPPED`

Transitions:

```text
GENERATED
  -> VERIFY_PENDING
  -> VERIFY_RESULT_READY

VERIFY_RESULT_READY + parent committed/full-accept
  -> READY_SHADOW

VERIFY_RESULT_READY + parent partial/reject/not committed
  -> INVALIDATED

READY_SHADOW + commit enabled for depth
  -> COMMIT_CANDIDATE

COMMIT_CANDIDATE + guards pass
  -> COMMITTED

COMMIT_CANDIDATE + guards fail
  -> SKIPPED

parent invalidated/stale/frontier mismatch
  -> child INVALIDATED
  -> descendant CASCADE_DISCARDED
```

Commit guards:

- child is ready shadow
- parent exists
- parent depth is `child.depth - 1`
- parent is committed/full-accept according to that depth's semantics
- child is not invalidated
- child is not cascade-discarded
- child is not stale/expired/frontier-mismatched
- seq is still active
- normal lane conflict count is zero
- no duplicate commit by proposal id
- no duplicate commit by `(seq_id, depth, step_id)` unless explicitly modeled
- target/draft length and token evidence match when available
- depth commit is enabled by config
- `child.depth <= max_rolling_depth`

## E. Stop Conditions For Future Unbounded-Style Rolling

"Unbounded" should be implemented as a bounded runtime loop with explicit stop conditions. It must never mean unconstrained recursive proposal generation.

Required stop conditions:

- reject
- partial accept
- request finished or EOS
- `max_new_tokens`
- `max_rolling_depth`
- `max_eager_chain_tokens`
- `max_eager_chain_steps`
- max rolling proposals per step
- per-step token budget
- per-step payload budget
- SLO or latency budget
- scheduler/batch pressure
- memory budget
- target/draft frontier mismatch
- target/draft token mismatch
- stale parent
- normal lane takeover
- missing buffered proposal
- duplicate proposal/commit detection

The generic design should preserve the possibility of larger bounded N values by making depth a data field, while all runtime execution remains constrained by these stop conditions.

## F. Generic Trace Schema Proposal

Existing depth-specific fields must remain until all downstream tools are migrated. Generic trace fields should be additive.

### `rolling_chain_nodes`

List of node dictionaries:

```json
{
  "proposal_id": 900000103,
  "seq_id": 7,
  "depth": 3,
  "parent_id": 900000102,
  "root_id": 900000100,
  "source": "rolling_depth3_ready_only",
  "status": "COMMITTED",
  "status_reason": null,
  "base_len": 128,
  "token_count": 4,
  "accept_len": 4,
  "verify_result": "full_accept",
  "apply_action": "append_full_accept_real_commit",
  "commit_state": "real_committed",
  "side": "target",
  "plan_id": 12,
  "step_id": 34
}
```

A compact tuple representation is acceptable if trace size becomes a problem, but the first implementation should favor debuggability.

### `rolling_chain_edges`

Map from child proposal id to parent proposal id:

```json
{
  "900000101": 900000100,
  "900000102": 900000101,
  "900000103": 900000102
}
```

### `rolling_depth_summary`

Map from depth to lifecycle counts:

```json
{
  "1": {
    "candidate_proposals": 3,
    "ready_proposals": 3,
    "committed_proposals": 3,
    "invalidated_proposals": 0,
    "cascade_discarded_proposals": 0,
    "skipped_proposals": 0,
    "candidate_tokens": 12,
    "ready_tokens": 12,
    "committed_tokens": 12
  }
}
```

### `rolling_chain_commit_summary`

Map from depth to committed tokens and target/draft increments:

```json
{
  "3": {
    "committed_proposals": [900000103],
    "committed_tokens": 4,
    "target_verified": 4,
    "target_accepted": 4,
    "target_rejected": 0,
    "target_invalidated": 0,
    "draft_verified": 4,
    "draft_accepted": 4,
    "draft_rejected": 0,
    "draft_invalidated": 0
  }
}
```

### `rolling_chain_safety_summary`

```json
{
  "max_configured_depth": 3,
  "max_observed_depth": 3,
  "max_real_committed_depth": 3,
  "depth_gt_configured_real_commit_count": 0,
  "depth_gt3_real_commit_count": 0,
  "depth4_real_commit_count": 0,
  "normal_lane_conflict_count": 0,
  "missing_buffered_proposal_unexpected_count": 0,
  "duplicate_commit_count": 0,
  "parent_missing_count": 0,
  "invalid_committed_count": 0,
  "cascade_committed_count": 0,
  "target_draft_mismatch_count": 0
}
```

Backward compatibility rule:

- legacy fields remain the source of truth during migration
- generic fields are generated alongside legacy fields
- checkers compare legacy-derived and generic-derived summaries before switching ownership
- runners continue emitting existing summary keys

## G. Generic Accounting Proposal

The accounting implementation should be a depth loop over unique committed proposal ids:

```python
for depth in committed_depths:
    committed_tokens[depth] = sum_unique_committed_token_counts(depth)
    target_verified[depth] = sum_target_verified_increments(depth)
    target_accepted[depth] = sum_target_accepted_increments(depth)
    draft_verified[depth] = sum_draft_verified_increments(depth)
    draft_accepted[depth] = sum_draft_accepted_increments(depth)
```

Combined formula:

```python
combined_real_committed_tokens = sum(committed_tokens.values())
combined_actual_verified_token_increment_sum = sum(target_verified.values())
combined_actual_accepted_token_increment_sum = sum(target_accepted.values())
```

Per-depth invariants:

- `target_verified[depth] == committed_tokens[depth]`
- `target_accepted[depth] == committed_tokens[depth]`
- `draft_verified[depth] == committed_tokens[depth]`
- `draft_accepted[depth] == committed_tokens[depth]`
- target rejected increment is zero for full-accept-only real commits
- target invalidated increment is zero for committed proposals
- draft rejected increment is zero for full-accept-only real commits
- draft invalidated increment is zero for committed proposals

The current combined accounting for N=3 is:

```text
combined =
  one_shot_committed_tokens
  + continuous_depth1_real_committed_tokens
  + rolling_depth2_real_committed_tokens
  + rolling_depth3_real_committed_tokens
```

Future depth-4 or bounded-N tokens should enter this formula only after those depths have legal real-commit semantics and checker coverage.

## H. Generic Checker Proposal

Future checker:

```text
benchmark/check_generic_bounded_rolling_chain.py
```

Responsibilities:

- parse generic fields when present
- fall back to legacy per-depth fields
- dedupe target/draft side rows by proposal id
- build a `RollingChainRegistry`
- validate parent-child consistency for all depths
- validate no real commit beyond configured max depth
- validate no real commit beyond supported phase depth
- validate committed proposal legality
- validate ready-shadow precondition for committed proposals
- validate parent committed/full-accept precondition
- validate invalidated and cascade-discarded proposals are not committed
- validate target/draft accounting by depth
- validate combined accounting across committed depths
- validate normal lane conflict count is zero
- validate missing buffered proposal unexpected count is zero
- validate duplicate proposal commit count is zero
- validate duplicate `(seq_id, depth, step_id)` commit count is zero unless modeled

Parsing strategy:

1. Build nodes from legacy one-shot/depth1/depth2/depth3 fields.
2. Merge side-record duplicates into one logical proposal node.
3. Attach parent/root/depth/status evidence globally.
4. Build per-depth summaries.
5. Compare summaries with legacy aggregate fields.
6. If generic runtime fields exist, compare generic-derived and legacy-derived registries.

This checker should eventually replace depth-local combined-accounting assumptions in lower-depth checkers, but it should not remove the depth-specific safety checkers until at least one full phase validates parity.

## I. Migration Plan

### 8f-a: Design Document Only

- Add this design.
- Do not change runtime or checker behavior.
- Preserve the 8e N=3 golden baseline.

### 8f-b: Legacy-To-Generic Audit Parser

- Add a read-only parser that ingests existing legacy trace fields into `RollingProposalNode` and `RollingChainRegistry`.
- Run it from a new generic audit checker.
- Do not emit new runtime trace fields yet.
- Validate that parser-derived accounting matches existing 8e audit output.

### 8f-c: Checker/Accounting Internal Refactor

- Update existing checkers and performance accounting to use the generic parser internally.
- Keep all existing input/output fields.
- Preserve the current checker chain.
- Run the full 8e regression as a no-behavior-change proof.

### 8f-d: Add Generic Summary Fields

- Emit additive generic trace fields such as `rolling_chain_nodes`, `rolling_depth_summary`, and `rolling_chain_safety_summary`.
- Keep all legacy depth-specific fields.
- Add parity checks comparing generic fields against legacy fields.

### 8f-e: Runtime Helper Refactor

- Extract common helper functions for:
  - proposal node creation
  - parent resolution
  - ready/invalid/cascade transitions
  - real-commit guard evaluation
  - target/draft sync
  - per-depth accounting emission
- Keep depth2 and depth3 behavior identical.
- Do not add depth4 in this step.

### 8f-f: N=3 Regression Proof

- Re-run the full 8e matrix.
- Require identical legality results:
  - `max_real_committed_depth <= 3`
  - `depth_gt3_real_commit_count = 0`
  - `normal_lane_conflict_count = 0`
  - combined accounting remains correct

### 8g: Depth-4 Shadow Candidate

- Use the generic path to add depth-4 shadow only.
- No depth-4 real commit until shadow lifecycle passes the same standard used for depth-3.

### 8h: Bounded-N Runtime

- Generalize runtime loop for `max_rolling_depth = N`.
- Keep explicit stop conditions.
- Make commit enablement per depth explicit.
- Add SLO and payload budget gates before any larger-N rollout.

## J. Risk Analysis

### Trace Size Explosion

Depth-as-data can increase trace size quickly, especially if every side record emits full node dictionaries. Mitigations:

- emit compact node deltas after initial implementation
- dedupe node records by proposal id
- summarize large token spans by length/hash unless payload-level debugging is enabled
- keep detailed payload fields behind trace level

### Control-Plane Overhead

Current 8e warnings show payload and suppression overhead may dominate useful committed tokens. Mitigations:

- track committed tokens per candidate token
- track committed tokens per payload unit
- enforce per-step proposal and payload budgets
- stop chain expansion when expected useful commit share is too low

### Checker Complexity

Per-depth checkers became fragile when higher-depth commits were added. Mitigations:

- centralize parent-chain and combined-accounting validation in the generic checker
- keep lower-depth checkers focused on local depth legality
- use one parser for trace normalization

### Scheduler Fairness And SLO Risk

Longer rolling chains can starve normal scheduling or violate latency budgets. Mitigations:

- enforce SLO budget stops
- cap proposals per step
- track normal lane suppression explicitly
- keep normal lane conflict checks hard-zero

### Frontier Mismatch

Target/draft divergence is the main correctness risk for deeper chains. Mitigations:

- make frontier fields first-class in `RollingProposalNode`
- validate target/draft length and token matches before commit
- cascade discard descendants on mismatch

### Duplicate Commits

Distributed target/draft records can double-count unless proposal identity is authoritative. Mitigations:

- commit dedupe by proposal id
- optional duplicate guard by `(seq_id, depth, step_id)`
- side-record dedupe in parser and checkers

### Parent Chain Stale State

Higher depth proposals can outlive their parent frontier. Mitigations:

- parent status resolution must be global and monotonic
- stale/expired/frontier-mismatch states must invalidate or cascade
- pending must not become terminal if the parent later resolves

### Low Coverage

Current committed token share is low. Mitigations:

- preserve 8e golden baseline
- add workload cases with enough chainable tokens
- measure candidate/ready/committed conversion by depth
- require depth-N value before increasing N

## K. Recommendation

Do not implement depth-4 next as another hard-coded phase.

Recommended next step:

1. Build a generic legacy-to-node parser over current N=3 fields.
2. Add `check_generic_bounded_rolling_chain.py` using that parser.
3. Prove it agrees with the existing 8e checker chain.
4. Move existing accounting/checkers to the parser internally while preserving output fields.
5. Only then refactor runtime helpers.
6. Only after the no-behavior-change N=3 regression passes should depth-4 shadow be considered.

This keeps the current N=3 validated baseline intact while making the next runtime phase depend on a generic model instead of another depth-specific copy.
