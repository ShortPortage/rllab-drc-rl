"""Diagnostics for projection-based constrained policy optimizers.

The helpers in this module log how much of a reward-only policy update remains
after the update is projected/corrected for safety feasibility.

Projected-step percentages are intentionally logged because they answer different questions:

* ``ProjectedStepDirectionalRetentionPct`` measures progress along the original
  reward step direction::

      100 * dot(final_step, reward_step) / (||reward_step||^2 + eps)

  This value can exceed 100% if the projected step moves farther than the reward
  step along the reward direction. It can be negative if projection reverses
  reward-direction progress.

* ``ProjectedStepDirectionalAlignmentPct`` measures cosine alignment with the
  original reward step direction::

      100 * dot(final_step, reward_step) / (||final_step|| * ||reward_step|| + eps)

  This value is naturally bounded between -100% and 100% up to numerical error.

* ``ProjectedStepNormRatioPct`` measures total final movement size relative to
  the reward step size::

      100 * ||final_step|| / (||reward_step|| + eps)

  This value can also exceed 100%, especially when projection adds a large
  safety/cost correction component.
"""

import numpy as np

from rllab.misc import logger


DEFAULT_EPS = 1e-8


def _as_flat_float_array(step):
    """Return *step* as a flat float64 numpy array."""
    return np.asarray(step, dtype=np.float64).reshape(-1)


def _safe_nan():
    return float("nan")


def record_projected_step_metrics(reward_step, final_step, eps=DEFAULT_EPS, prefix=""):
    """Record projected-step diagnostics to the rllab tabular logger.

    Args:
        reward_step: The reward-only step vector. Use the same sign convention as
            ``final_step``. For the PCPO optimizers, passing parameter deltas
            ``-flat_descent_step_tr`` and ``-flat_descent_step`` is clearest.
        final_step: The final projected/corrected step vector, with the same sign
            convention as ``reward_step``.
        eps: Small denominator stabilizer for the percentage formulas.
        prefix: Optional logger key prefix.

    The percentage metrics are undefined when the reward step has zero norm, and
    all metrics are undefined if either input has non-finite values or mismatched
    shape. In those cases, this helper logs ``nan`` for the affected percentages
    rather than raising during training.
    """
    reward = _as_flat_float_array(reward_step)
    final = _as_flat_float_array(final_step)

    reward_norm = _safe_nan()
    final_norm = _safe_nan()
    directional_retention_pct = _safe_nan()
    directional_alignment_pct = _safe_nan()
    norm_ratio_pct = _safe_nan()

    valid = reward.shape == final.shape and reward.size > 0
    if valid and np.all(np.isfinite(reward)) and np.all(np.isfinite(final)):
        reward_norm_sq = float(np.dot(reward, reward))
        final_norm_sq = float(np.dot(final, final))
        dot = float(np.dot(final, reward))
        reward_norm = float(np.sqrt(max(reward_norm_sq, 0.0)))
        final_norm = float(np.sqrt(max(final_norm_sq, 0.0)))

        if reward_norm > eps:
            directional_retention_pct = 100.0 * dot / (reward_norm_sq + eps)
            norm_ratio_pct = 100.0 * final_norm / (reward_norm + eps)
            if final_norm > eps:
                directional_alignment_pct = 100.0 * dot / (final_norm * reward_norm + eps)

    logger.record_tabular(prefix + "ProjectedStepDirectionalRetentionPct", directional_retention_pct)
    logger.record_tabular(prefix + "ProjectedStepDirectionalAlignmentPct", directional_alignment_pct)
    logger.record_tabular(prefix + "ProjectedStepNormRatioPct", norm_ratio_pct)
    logger.record_tabular(prefix + "RewardStepNorm", reward_norm)
    logger.record_tabular(prefix + "ProjectedStepNorm", final_norm)
