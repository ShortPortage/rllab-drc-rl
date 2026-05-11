# Constrained Policy Optimization for rllab

Constrained Policy Optimization (CPO) is an algorithm for learning policies that should satisfy behavioral constraints throughout training. [1]

This module was designed for [rllab](https://github.com/openai/rllab) [2], and includes the implementations of
- [Constrained Policy Optimization](https://github.com/jachiam/cpo/blob/master/algos/safe/cpo.py)
- [Primal-Dual Optimization](https://github.com/jachiam/cpo/blob/master/algos/safe/pdo.py)
- [Fixed Penalty Optimization](https://github.com/jachiam/cpo/blob/master/algos/safe/fpo.py)

described in our paper [1]. 

To configure, run the following command in the root folder of `rllab`:

```bash
git submodule add -f https://github.com/jachiam/cpo sandbox/cpo
```

Run CPO in the Point-Gather environment with
```bash
python sandbox/cpo/experiments/CPO_point_gather.py 
```

## RCPO diagnostic plots

See `docs/rcpo_diagnostic_plots.md` for detailed usage instructions.

PCPO/RCPO runs log projected-step diagnostics in `progress.csv`:

- `ProjectedStepDirectionalRetentionPct`: final projected step progress along the reward-step direction.
- `ProjectedStepNormRatioPct`: final projected step norm relative to the reward-step norm.

Both percentages can exceed 100%; directional retention can be negative if projection reverses reward-direction progress.

Example PointGather plotting commands:

```bash
python scripts/plot_rcpo_diagnostics.py \
  data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv \
  --labels RCPO-KL \
  --output-dir plots/rcpo_kl_point_gather

python scripts/plot_rcpo_diagnostics.py \
  data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv \
  --labels RCPO-L2 \
  --output-dir plots/rcpo_l2_point_gather

python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
  'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
  --labels RCPO-KL RCPO-L2 \
  --output-dir plots/rcpo_point_gather_compare
```

Use `--reward-key AverageDiscountedReturn` to plot discounted return instead of `AverageReturn`. The target RCPO PointGather scripts alternate policy/adversary phases, so the plotting script also writes `reward_policy.png` and `reward_adversary.png` by default. Old logs without `SafetyThreshold` can pass `--threshold <value>`.

***

1. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. "[Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)". _Proceedings of the 34th International Conference on Machine Learning (ICML), 2017._ 
2. Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

