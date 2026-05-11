# RCPO Run and Plotting Notes

Run these commands from the repository root.

## Run RCPO experiments

Run the KL version:

```bash
python sandbox/cpo/experiments/RCPO_KL_point_gather.py
```

Run the L2 version:

```bash
python sandbox/cpo/experiments/RCPO_L2_point_gather.py
```

These should write experiment logs under paths like:

```text
data/local/RCPO-KL-PointGather/.../progress.csv
data/local/RCPO-L2-PointGather/.../progress.csv
```

## Generate comparison plots

After both runs finish, generate comparison plots with:

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
  'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
  --labels RCPO-KL RCPO-L2 \
  --output-dir plots/rcpo_point_gather_compare
```

Plots will be written to:

```text
plots/rcpo_point_gather_compare/
```

Expected outputs include reward, raw cost, and projected-step diagnostic plots.

## Plot only KL

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
  --labels RCPO-KL \
  --output-dir plots/rcpo_kl_point_gather
```

## Plot only L2

```bash
python scripts/plot_rcpo_diagnostics.py \
  'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
  --labels RCPO-L2 \
  --output-dir plots/rcpo_l2_point_gather
```

## Environment note

This old rllab code may require the intended conda environment and a working MuJoCo license/config before the experiments will run successfully.
