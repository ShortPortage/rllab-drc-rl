# RCPO PointGather Diagnostics Summary

Generated from:

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
  'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
  --labels RCPO-KL RCPO-L2 \
  --output-dir plots/rcpo_point_gather_compare
```

Input runs:

- RCPO-KL: `data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_2026_05_11_12_28_05_0001/progress.csv`
- RCPO-L2: `data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_2026_05_11_13_42_15_0001/progress.csv`

All summary statistics below are averages over the last 150 rows/iterations where available.

## Summary table

| Metric | RCPO-KL | RCPO-L2 | Notes |
|---|---:|---:|---|
| AverageReturn | -0.134631 | -0.0860488 | Aggregate reward across alternating policy/adversary phases. |
| Policy AverageReturn | 8.37931 | 6.16881 | Policy phase only. |
| Adversary AverageReturn | -8.5181 | -6.29416 | Adversary phase only. |
| MeanSafety[U]Return | 0.125441 | 0.229732 | Safety/cost; threshold is `0.1`. |
| ProjectedStepDirectionalRetentionPct | 91.9557 | 99.3605 | Reward-direction progress in reward-step units; can exceed 100%. |
| ProjectedStepDirectionalAlignmentPct | 72.8154 | 75.3028 | Cosine alignment with reward-step direction; naturally bounded. |
| ProjectedStepNormRatioPct | 129.423 | 138.636 | Final projected-step norm relative to reward-step norm; can exceed 100%. |

## Interpretation

### Directional retention is fairly high

- RCPO-KL: `91.96%`
- RCPO-L2: `99.36%`

In reward-step units, both methods mostly preserve reward-direction progress. L2 is especially close to full directional retention.

### Directional alignment is only moderate

- RCPO-KL: `72.82%`
- RCPO-L2: `75.30%`

This means the final projected/corrected update is substantially angled away from the pure reward update. A cosine alignment around `0.73–0.75` corresponds to an angle of roughly `41–43` degrees from the reward-step direction.

### Projection increases step norm

- RCPO-KL: `129.42%`
- RCPO-L2: `138.64%`

The final projected/corrected step is larger than the reward-only step on average. This explains how directional retention can stay near `100%` while alignment is only around `75%`: the optimizer is adding a sizable safety/cost correction component rather than merely shrinking the reward step.

### Safety performance differs substantially

- RCPO-KL cost: `0.125441`, slightly above the `0.1` threshold.
- RCPO-L2 cost: `0.229732`, well above the `0.1` threshold.

KL is closer to satisfying the safety constraint. L2 has slightly better reward-direction retention, but worse safety performance.

### Phase-split reward is more informative than aggregate reward

The aggregate reward is near zero/negative because the RCPO setup alternates policy and adversary phases. The phase-specific reward plots are more useful:

- Policy reward:
  - RCPO-KL: `8.37931`
  - RCPO-L2: `6.16881`
- Adversary reward:
  - RCPO-KL: `-8.5181`
  - RCPO-L2: `-6.29416`

## Implications for active-set optimization

These diagnostics do not suggest that projection is massively slowing reward-direction progress according to `ProjectedStepDirectionalRetentionPct`. The larger issue appears to be that projection changes the update direction and increases the update norm by adding a substantial orthogonal/safety correction component.

An active-set approach would be compelling if it can achieve:

- `ProjectedStepDirectionalRetentionPct` near `95–100%`,
- `ProjectedStepDirectionalAlignmentPct` higher than the current `~73–75%`,
- `ProjectedStepNormRatioPct` closer to `100%`,
- `MeanSafety[U]Return <= 0.1`,
- policy-phase reward at least competitive with the current baselines.

If active-set optimization raises directional alignment toward `90%` while keeping safety near or below threshold, that would be strong evidence that it reduces unnecessary projection distortion.
