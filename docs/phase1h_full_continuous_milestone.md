# Phase 1H Full-Continuous Milestone

Phase 1H-8u-b is the first validated full-continuous milestone for the rolling eager path.

## What 8u-b Achieved

- `max_rolling_continuous_depth=100` is accepted in full-continuous mode.
- Full continuous remains bounded by the configured max depth. It is not unbounded recursion.
- Real generic P5+ candidate, ready, commit, and apply activity is present.
- The current minimal workload reaches depth 60:
  - `generic_full_continuous_max_observed_depth = 60`
  - `generic_full_continuous_max_real_committed_depth = 60`
  - `generic_rolling_apply_depths = [2, 3, 4, ..., 60]`
- Partial-prefix recovery is compatible with full-continuous accounting.
- `depth_gt_max = 0`, `normal_lane_conflict = 0`, and target/draft mismatch count remains zero.

## Feature Boundary

Included:

- Bounded full continuous with `max_depth=100`.
- Real depth greater than 4 activity through the generic runtime/apply path.
- Depth-4 legacy fields remain available for compatibility.
- Partial-prefix recovery accounting works with full-continuous output totals.

Not included:

- Cached-admission.
- Dynamic gamma or dynamic draft length.
- Unbounded rolling recursion.
- Removal of legacy depth1/depth2/depth3/depth4 checker fields.
- SLO-aware gamma policy.

## Validated Cases

- `baseline_8t_generic_apply_depth4`
- `full_continuous_depth100_baseline`
- `full_continuous_depth100_partial_recovery`
- `full_continuous_depth100_stress`
- `full_continuous_depth100_must_exceed4`

Current reference values from the validated minimal run:

- Baseline depth4 combined output: `44`
- Full-continuous baseline combined output: `484`
- Full-continuous partial/stress combined output: `486`
- Full-continuous max observed/real depth: `60`
- `depth_gt_max = 0`

These values are reference points for the current workload shape. Future valid runs may differ in totals if scheduling or workload shape changes, but output totals must remain internally consistent.

## Required Checker Chain

Run the following for each case:

```bash
python3 benchmark/check_eager_partial_prefix_recovery.py "$TRACE" "$RESULT"
python3 benchmark/check_bounded_rolling_readiness_audit.py "$TRACE" "$RESULT" --check-generic-parity
python3 benchmark/check_generic_bounded_rolling_chain.py "$TRACE" "$RESULT"
python3 benchmark/check_generic_rolling_runtime_parity.py "$TRACE" "$RESULT"
python3 benchmark/check_generic_rolling_apply_path_parity.py "$TRACE" "$RESULT"
python3 benchmark/check_full_continuous_max_depth.py "$TRACE" "$RESULT"
python3 benchmark/check_eager_performance_accounting.py "$TRACE" "$RESULT" --check-generic-chain
python3 benchmark/check_multislo_result.py "$RESULT"
```

For `full_continuous_depth100_must_exceed4`, also run:

```bash
python3 benchmark/check_full_continuous_max_depth.py "$TRACE" "$RESULT" --require-depth-gt4-activity
```

## Compact Validation

Summary only:

```bash
python3 benchmark/run_phase1h8v_full_continuous_validation.py \
  --root results/multislo/phase1h8u_full_continuous_depth100_real_depthgt4_min_fix3 \
  --summary-only --strict
```

Checker chain:

```bash
python3 benchmark/run_phase1h8v_full_continuous_validation.py \
  --root results/multislo/phase1h8u_full_continuous_depth100_real_depthgt4_min_fix3 \
  --run-checkers --strict
```

Machine-readable compact summary:

```bash
python3 benchmark/summarize_phase1h8u_full_continuous.py \
  --root results/multislo/phase1h8u_full_continuous_depth100_real_depthgt4_min_fix3 \
  --json --strict
```

## Remaining Work

- Clean up historical depth-specific validation surfaces.
- Integrate cached-admission.
- Run fixed-gamma sweeps.
- Add per-request SLO-aware gamma policy.
- Perform performance analysis on full-continuous depth and recovery behavior.
