#!/usr/bin/env python
"""Plot RCPO reward, raw cost, and PCPO projected-step diagnostics.

Examples for the target PointGather experiments::

    python scripts/plot_rcpo_diagnostics.py \
        data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv \
        --output-dir plots/rcpo_kl_point_gather --labels RCPO-KL

    python scripts/plot_rcpo_diagnostics.py \
        data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv \
        --output-dir plots/rcpo_l2_point_gather --labels RCPO-L2

Compare KL and L2 on shared figures::

    python scripts/plot_rcpo_diagnostics.py \
        'data/local/RCPO-KL-PointGather/RCPO_KL-PointGather_*/progress.csv' \
        'data/local/RCPO-L2-PointGather/RCPO_L2-PointGather_*/progress.csv' \
        --labels RCPO-KL RCPO-L2 \
        --output-dir plots/rcpo_point_gather_compare

The target RCPO PointGather experiments alternate two policy updates, then two
adversary updates, because each outer-loop phase creates a PCPO trainer with
``n_itr=2``. By default this script also writes separate reward plots for those
policy/adversary phases.

The script saves PNG files only and does not open matplotlib windows.
"""

from __future__ import print_function

import argparse
import csv
import glob
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIRECTIONAL_KEY = "ProjectedStepDirectionalRetentionPct"
ALIGNMENT_KEY = "ProjectedStepDirectionalAlignmentPct"
NORM_RATIO_KEY = "ProjectedStepNormRatioPct"
DEFAULT_COST_KEY = "MeanSafety[U]Return"
FALLBACK_COST_KEY = "SafetyEval"
THRESHOLD_KEY = "SafetyThreshold"
SUMMARY_WINDOW = 150


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _looks_like_progress_csv(path):
    return os.path.isfile(path) and os.path.basename(path) == "progress.csv"


def resolve_progress_paths(inputs):
    """Resolve globs, experiment directories, and progress.csv paths.

    Returns ``(progress_csv_path, input_index)`` pairs so labels can be supplied
    either per resolved CSV or per original input/glob.
    """
    resolved = []
    seen = set()
    for input_index, item in enumerate(inputs):
        matches = glob.glob(item)
        if not matches:
            matches = [item]
        for match in sorted(matches):
            candidates = []
            if _looks_like_progress_csv(match):
                candidates.append(match)
            elif os.path.isdir(match):
                direct = os.path.join(match, "progress.csv")
                if os.path.isfile(direct):
                    candidates.append(direct)
                else:
                    candidates.extend(sorted(glob.glob(os.path.join(match, "*", "progress.csv"))))
            else:
                print("warning: skipping missing input: %s" % match, file=sys.stderr)

            for candidate in candidates:
                abspath = os.path.abspath(candidate)
                if abspath not in seen:
                    resolved.append((abspath, input_index))
                    seen.add(abspath)
    return resolved


def _to_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_progress_csv(path):
    rows = []
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def column(rows, key):
    values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
    if key == ALIGNMENT_KEY and not has_finite(values):
        retention = np.asarray([_to_float(row.get(DIRECTIONAL_KEY)) for row in rows], dtype=np.float64)
        reward_norm = np.asarray([_to_float(row.get("RewardStepNorm")) for row in rows], dtype=np.float64)
        projected_norm = np.asarray([_to_float(row.get("ProjectedStepNorm")) for row in rows], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = retention * reward_norm / projected_norm
        values[~np.isfinite(values)] = np.nan
    return values


def has_finite(values):
    return values.size > 0 and np.any(np.isfinite(values))


def last_window_mean(values, window=SUMMARY_WINDOW):
    tail = values[-window:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:
        return np.nan, 0
    return float(np.mean(finite)), int(finite.size)


def print_last_window_mean(label, key, values, output_path, window=SUMMARY_WINDOW):
    mean, count = last_window_mean(values, window=window)
    plot_name = os.path.basename(output_path)
    if np.isfinite(mean):
        print("summary: %s | %s | last %d rows avg %s = %.6g (n=%d)" % (
            plot_name, label, window, key, mean, count))
    else:
        print("summary: %s | %s | last %d rows avg %s = nan (n=0)" % (
            plot_name, label, window, key))


def default_label(path):
    run_dir = os.path.basename(os.path.dirname(path))
    exp_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if run_dir:
        return run_dir
    return exp_dir or path


def label_for_run(labels, resolved_index, input_index, path, num_resolved, num_inputs):
    if not labels:
        return default_label(path)
    if len(labels) == num_resolved:
        return labels[resolved_index]
    if len(labels) == num_inputs:
        return labels[input_index]
    if len(labels) == 1:
        return labels[0]
    return default_label(path)


def load_runs(resolved, labels, num_inputs):
    runs = []
    num_resolved = len(resolved)
    for idx, (path, input_index) in enumerate(resolved):
        rows = read_progress_csv(path)
        label = label_for_run(labels, idx, input_index, path, num_resolved, num_inputs)
        runs.append({"path": path, "label": label, "rows": rows})
    return runs


def x_values(run, x_key):
    rows = run["rows"]
    if x_key:
        values = column(rows, x_key)
        if has_finite(values):
            return values, x_key
        print("warning: %s missing finite x-key '%s'; using update index" % (run["path"], x_key), file=sys.stderr)
    return np.arange(len(rows), dtype=np.float64), "Update index"


def plot_series(runs, y_key, title, ylabel, output_path, x_key=None, fallback_key=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    xlabel = None
    for run in runs:
        x, this_xlabel = x_values(run, x_key)
        xlabel = xlabel or this_xlabel
        key_used = y_key
        y = column(run["rows"], key_used)
        if not has_finite(y) and fallback_key:
            key_used = fallback_key
            y = column(run["rows"], key_used)
        if not has_finite(y):
            print("warning: %s missing finite y-key '%s'" % (run["path"], y_key), file=sys.stderr)
            continue
        label = run["label"]
        if key_used != y_key:
            label = "%s (%s)" % (label, key_used)
        ax.plot(x, y, label=label)
        print_last_window_mean(label, key_used, y, output_path)
        plotted = True

    ax.set_title(title)
    ax.set_xlabel(xlabel or "Update index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("wrote %s" % output_path)


def phase_mask(num_rows, phase_length, phase_offset):
    indices = np.arange(num_rows)
    return ((indices // phase_length) % 2) == phase_offset


def plot_phase_reward_series(runs, reward_key, phase_length, phase_offset, phase_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for run in runs:
        y_all = column(run["rows"], reward_key)
        if not has_finite(y_all):
            print("warning: %s missing finite reward key '%s'" % (run["path"], reward_key), file=sys.stderr)
            continue
        mask = phase_mask(len(y_all), phase_length, phase_offset)
        y = y_all[mask]
        x = np.arange(len(y), dtype=np.float64)
        if not has_finite(y):
            print("warning: %s has no finite %s reward rows" % (run["path"], phase_name), file=sys.stderr)
            continue
        ax.plot(x, y, label=run["label"])
        print_last_window_mean(run["label"], reward_key, y, output_path)
        plotted = True

    ax.set_title("%s reward over time (%s)" % (phase_name, reward_key))
    ax.set_xlabel("%s update index" % phase_name)
    ax.set_ylabel(reward_key)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("wrote %s" % output_path)


def threshold_constant(threshold_values, threshold):
    if has_finite(threshold_values):
        finite = threshold_values[np.isfinite(threshold_values)]
        if finite.size > 0:
            return float(finite[0])
    return threshold


def plot_cost_with_threshold(runs, cost_key, output_path, x_key=None, threshold=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    xlabel = None
    for run in runs:
        x, this_xlabel = x_values(run, x_key)
        xlabel = xlabel or this_xlabel
        key_used = cost_key
        cost = column(run["rows"], key_used)
        if not has_finite(cost) and key_used != FALLBACK_COST_KEY:
            key_used = FALLBACK_COST_KEY
            cost = column(run["rows"], key_used)
        if not has_finite(cost):
            print("warning: %s missing finite cost key '%s' and fallback '%s'" % (
                run["path"], cost_key, FALLBACK_COST_KEY), file=sys.stderr)
            continue

        cost_label = run["label"]
        if key_used != cost_key:
            cost_label = "%s (%s)" % (cost_label, key_used)
        line = ax.plot(x, cost, label=cost_label)[0]
        print_last_window_mean(cost_label, key_used, cost, output_path)
        plotted = True

        threshold_values = column(run["rows"], THRESHOLD_KEY)
        threshold_y = threshold_constant(threshold_values, threshold)
        if threshold_y is not None and np.isfinite(threshold_y):
            ax.axhline(
                threshold_y,
                linestyle="--",
                color=line.get_color(),
                alpha=0.8,
                label="%s threshold" % run["label"],
            )
        else:
            print("warning: %s has no SafetyThreshold column; pass --threshold for old logs" % run["path"], file=sys.stderr)

    ax.set_title("Raw safety/cost over time with threshold")
    ax.set_xlabel(xlabel or "Update index")
    ax.set_ylabel("Raw safety/cost")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("wrote %s" % output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save PNG plots for RCPO reward, raw cost, directional retention, directional alignment, and norm ratio diagnostics."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Experiment directories, progress.csv paths, or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        default="rcpo_diagnostics_plots",
        help="Directory for output PNG files. Default: %(default)s",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels, one per resolved progress.csv. Defaults to run directory names.",
    )
    parser.add_argument(
        "--reward-key",
        default="AverageReturn",
        help="Reward column to plot. Use AverageDiscountedReturn if desired. Default: %(default)s",
    )
    parser.add_argument(
        "--cost-key",
        default=DEFAULT_COST_KEY,
        help="Raw cost column to plot. Falls back to SafetyEval. Default: %(default)s",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fallback safety/cost threshold for old logs without SafetyThreshold.",
    )
    parser.add_argument(
        "--x-key",
        default=None,
        help="Optional x-axis column. Default uses row/update index, which is safest for RCPO runs that reset Iteration.",
    )
    parser.add_argument(
        "--rcpo-phase-length",
        type=int,
        default=2,
        help="Rows per alternating RCPO phase. Target PointGather scripts use n_itr=2, so default is 2.",
    )
    parser.add_argument(
        "--no-phase-reward-plots",
        action="store_true",
        help="Disable separate policy/adversary reward plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resolved = resolve_progress_paths(args.inputs)
    if not resolved:
        raise SystemExit("No progress.csv files found.")
    if args.labels and len(args.labels) not in (1, len(args.inputs), len(resolved)):
        raise SystemExit(
            "--labels count (%d) must be 1, match original input count (%d), "
            "or match resolved progress.csv count (%d)." % (
                len(args.labels), len(args.inputs), len(resolved)))

    _ensure_dir(args.output_dir)
    runs = load_runs(resolved, args.labels, len(args.inputs))

    print("plotting the following progress files:")
    for run in runs:
        print("  %s -> %s" % (run["label"], run["path"]))

    plot_series(
        runs,
        args.reward_key,
        "Reward over time (%s)" % args.reward_key,
        args.reward_key,
        os.path.join(args.output_dir, "reward.png"),
        x_key=args.x_key,
    )
    if not args.no_phase_reward_plots:
        plot_phase_reward_series(
            runs,
            args.reward_key,
            args.rcpo_phase_length,
            0,
            "Policy",
            os.path.join(args.output_dir, "reward_policy.png"),
        )
        plot_phase_reward_series(
            runs,
            args.reward_key,
            args.rcpo_phase_length,
            1,
            "Adversary",
            os.path.join(args.output_dir, "reward_adversary.png"),
        )
    plot_cost_with_threshold(
        runs,
        args.cost_key,
        os.path.join(args.output_dir, "raw_cost_with_threshold.png"),
        x_key=args.x_key,
        threshold=args.threshold,
    )
    plot_series(
        runs,
        DIRECTIONAL_KEY,
        "Projected-step directional retention (%)\n(final projected step progress along reward-step direction)",
        "Directional retention (% of reward-direction step)",
        os.path.join(args.output_dir, "projected_step_directional_retention_pct.png"),
        x_key=args.x_key,
    )
    plot_series(
        runs,
        ALIGNMENT_KEY,
        "Projected-step directional alignment (%)\n(cosine alignment with reward-step direction)",
        "Directional alignment (% cosine similarity)",
        os.path.join(args.output_dir, "projected_step_directional_alignment_pct.png"),
        x_key=args.x_key,
    )
    plot_series(
        runs,
        NORM_RATIO_KEY,
        "Projected-step norm ratio (%)\n(total final step norm / reward step norm)",
        "Norm ratio (% of reward-step norm)",
        os.path.join(args.output_dir, "projected_step_norm_ratio_pct.png"),
        x_key=args.x_key,
    )


if __name__ == "__main__":
    main()
