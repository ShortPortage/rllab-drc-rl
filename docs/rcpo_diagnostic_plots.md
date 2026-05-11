# RCPO Diagnostic Plots

This document explains how to use the RCPO/PCPO diagnostics added for plotting reward, raw constraint cost, and projected update size over time.

## What was added

### Logging helper

` sandbox/cpo/optimizers/pcpo_diagnostics.py`

This helper logs how much of the reward-only policy update remains after the safety projection/correction.

It writes these columns into each run's `progress.csv`:

- `ProjectedStepDirectionalRetentionPct`
  - Progress along the original reward-update direction after projection.
  - Formula: `100 * dot(final_step, reward_step) / (||reward_step||^2 + eps)`
- `ProjectedStepNormRatioPct`
  - Total final projected update norm compared to reward-only update norm.
  - Formula: `100 * ||final_step|| / (||reward_step|| + eps)`
- `RewardStepNorm`
- `ProjectedStepNorm`

Important: either percentage can be above 100%. `ProjectedStepDirectionalRetentionPct` can be negative if projection reverses reward-direction progress.

### Optimizer integration

The metrics are logged by:

- `sandbox/cpo/optimizers/conjugate_constraint_optimizer_pcpo_kl.py`
- `sandbox/cpo/optimizers/conjugate_constraint_optimizer_pcpo_l2.py`

So they should work for:

- `sandbox/cpo/experiments/RCPO_KL_point_gather.py`
- `sandbox/cpo/experiments/RCPO_L2_point_gather.py`

### Safety threshold logging

`SafetyThreshold` is now logged in `sandbox/cpo/algos/safe/sampler_safe.py`, alongside:

- `SafetyEval`
- `MeanSafety[U]Return`

This lets the plotting script draw raw cost with the safety threshold.

## Run experiments

From the repository root:

```bash
python sandbox/cpo/experiments/RCPO_KL_point_gather.py
python sandbox/cpo/experiments/RCPO_L2_point_gather.py
```

Local runs are written under:

```text
data/local/RCPO-KL-PointGather/<run-name>/progress.csv
data/local/RCPO-L2-PointGather/<run-name>/progress.csv
```

## Plot a KL run

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
  --labels RCPO-KL \
  --output-dir plots/rcpo_kl_point_gather
```

## Plot an L2 run

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
  --labels RCPO-L2 \
  --output-dir plots/rcpo_l2_point_gather
```

## Compare KL and L2

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
  'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
  --labels RCPO-KL RCPO-L2 \
  --output-dir plots/rcpo_point_gather_compare
```

## RCPO policy/adversary alternation

The target scripts alternate between optimizing the main policy and optimizing an adversary:

- even outer-loop iterations (`i % 2 == 0`) optimize `policy` in `PointGatherAdvEnv(..., adv_policy=adv_policy)`;
- odd outer-loop iterations optimize `adv_policy` in `PointGatherNegEnv(..., adv_policy=policy)`.

Each phase creates a `PCPO_KL` or `PCPO_L2` trainer with `n_itr=2`, so the log rows are arranged as:

```text
policy row, policy row, adversary row, adversary row, policy row, policy row, ...
```

For that reason, the plotting script writes separate policy and adversary reward plots by default. The separated plots use compact per-role update indices, rather than the shared/reset `Iteration` values in `progress.csv`.

## Output files

The plotting script saves six `.png` files by default:

```text
reward.png
reward_policy.png
reward_adversary.png
raw_cost_with_threshold.png
projected_step_directional_retention_pct.png
projected_step_norm_ratio_pct.png
```

It does not open interactive matplotlib windows.

## Useful options

Use discounted reward instead of undiscounted reward:

```bash
python scripts/plot_rcpo_diagnostics.py <progress.csv> \
  --reward-key AverageDiscountedReturn \
  --output-dir plots/my_run
```

Use a specific x-axis column, if it is monotonic for your experiment:

```bash
python scripts/plot_rcpo_diagnostics.py <progress.csv> \
  --x-key TotalEnvInteracts \
  --output-dir plots/my_run
```

By default, the script uses row/update index because the target RCPO PointGather scripts repeatedly instantiate short `PCPO_*` trainers, which resets `Iteration` and `TotalEnvInteracts`.

Disable separate policy/adversary reward plots:

```bash
python scripts/plot_rcpo_diagnostics.py <progress.csv> \
  --no-phase-reward-plots \
  --output-dir plots/my_run
```

If you change `n_itr` in the RCPO experiment scripts, update the phase length used for policy/adversary splitting:

```bash
python scripts/plot_rcpo_diagnostics.py <progress.csv> \
  --rcpo-phase-length 5 \
  --output-dir plots/my_run
```

The raw cost plot uses a horizontal dashed threshold line. Plot old logs that do not have `SafetyThreshold`:

```bash
python scripts/plot_rcpo_diagnostics.py <old-progress.csv> \
  --threshold 0.1 \
  --output-dir plots/old_run
```

Use raw `SafetyEval` as the cost curve instead of `MeanSafety[U]Return`:

```bash
python scripts/plot_rcpo_diagnostics.py <progress.csv> \
  --cost-key SafetyEval \
  --output-dir plots/my_run
```

## Reading the step-size plots

- `projected_step_directional_retention_pct.png`
  - Best for showing reward-direction slowdown due to constraint projection.
  - Low values mean little of the reward update survives in the original reward direction.
  - Negative values mean the final projected update moves opposite the reward-only direction.

- `projected_step_norm_ratio_pct.png`
  - Shows how large the final projected update is compared to the reward-only update.
  - A high norm ratio does not necessarily mean good reward progress; it may be mostly sideways safety correction.

For the user's intended comparison, the directional retention plot is usually the most direct evidence of update slowdown from reward/constraint direction conflict.
