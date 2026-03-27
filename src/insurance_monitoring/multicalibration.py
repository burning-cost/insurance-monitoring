"""
MulticalibrationMonitor: subgroup-level calibration monitoring for insurance pricing.

PSI and A/E ratio are portfolio-level tools. PSI tells you whether the score
distribution has shifted; A/E tells you whether overall claims are on budget.
Neither tells you whether the model is mispricing a specific *combination* of
subgroup and premium band — the question multicalibration answers.

This is the monitoring analogue of insurance-fairness MulticalibrationAudit.
Where the audit is a one-shot sign-off test, the monitor runs every period and
accumulates evidence using the same credibility-shrinkage approach as classical
actuarial GLM monitoring.

Reference: Denuit, Michaelides & Trufin (2026), arXiv:2603.16317.

Algorithm
---------
1. **fit()**: establish exposure-weighted quantile bin boundaries from reference
   data and store them. Boundaries are frozen — never recomputed — so (bin, group)
   cells are comparable across periods.

2. **update()**: for each (bin_k, group_l) cell, compute:

   - ``observed``: exposure-weighted mean actual in that cell
   - ``expected``: exposure-weighted mean predicted
   - ``AE_ratio = observed / expected``
   - ``relative_bias = (observed - expected) / expected`` (= AE_ratio - 1)
   - ``z_stat``: Poisson-based test statistic

   The z-statistic uses the formula:

   .. math::

       z = \\frac{\\hat{b}_{kl} \\cdot \\sqrt{w_{kl}}}{\\sqrt{\\bar{\\pi}_{kl}}}

   where :math:`\\hat{b}_{kl} = \\bar{y}_{kl} - \\bar{\\pi}_{kl}` is the
   exposure-weighted mean residual, :math:`w_{kl}` is total exposure, and
   :math:`\\bar{\\pi}_{kl}` is the exposure-weighted mean prediction.

   This is asymptotically standard normal under the Poisson null
   (no miscalibration in cell kl).

3. **Alert gating**: alert fires only when BOTH conditions hold:

   - ``|relative_bias| >= min_relative_bias``
   - ``|z_stat| >= min_z_abs``

   The conjunction prevents noisy small-cell alerts (z-gate) while also
   filtering out statistically significant but economically trivial bias (bias-gate).

Design decisions
----------------
- Bin boundaries fixed at fit() time. Changing them mid-stream would make
  period-over-period comparison meaningless.
- No credibility shrinkage in v1.0. The Bühlmann formula from KB entry 3094 is
  implemented but the KB spec for monitoring v0.10.0 uses raw b_hat. Shrinkage
  will be added when we have empirical evidence on the optimal c parameter.
- Polars used for cell_table output — consistent with the rest of this library.
- No scipy dependency. z-stat is computed directly; no need for norm.cdf.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class MulticalibCell:
    """One (bin, group) cell from a single update() call.

    Attributes
    ----------
    bin_idx:
        Zero-based prediction quantile bin index (0 = lowest predicted rate).
    group:
        Subgroup label (string or int as passed to update).
    n_exposure:
        Total exposure in this cell for the current period.
    observed:
        Exposure-weighted mean actual in this cell.
    expected:
        Exposure-weighted mean prediction in this cell.
    AE_ratio:
        observed / expected.
    relative_bias:
        (observed - expected) / expected. Equivalent to AE_ratio - 1.
    z_stat:
        Poisson z-statistic. Under H0 (no miscalibration) asymptotically N(0,1).
    alert:
        True when |relative_bias| >= threshold AND |z_stat| >= min_z_abs.
    """

    bin_idx: int
    group: Any
    n_exposure: float
    observed: float
    expected: float
    AE_ratio: float
    relative_bias: float
    z_stat: float
    alert: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_idx": self.bin_idx,
            "group": self.group,
            "n_exposure": self.n_exposure,
            "observed": self.observed,
            "expected": self.expected,
            "AE_ratio": self.AE_ratio,
            "relative_bias": self.relative_bias,
            "z_stat": self.z_stat,
            "alert": self.alert,
        }


@dataclass
class MulticalibrationResult:
    """Result returned by :meth:`MulticalibrationMonitor.update`.

    Attributes
    ----------
    alerts:
        Cells where BOTH |relative_bias| >= threshold AND |z_stat| >= min_z_abs.
        Empty list means the portfolio passed for this period.
    cell_table:
        Polars DataFrame with one row per evaluated (bin, group) cell.
        Columns: bin_idx, group, n_exposure, observed, expected, AE_ratio,
        relative_bias, z_stat, alert.
    n_cells_evaluated:
        Total cells with sufficient exposure to be evaluated.
    n_cells_skipped:
        Cells with exposure below min_exposure (not evaluated, not alerted).
    period_index:
        Monotonically incrementing counter for the update number (1-based).
    """

    alerts: list[MulticalibCell]
    cell_table: pl.DataFrame
    n_cells_evaluated: int
    n_cells_skipped: int
    period_index: int

    def summary(self) -> dict[str, Any]:
        """Return a concise governance-ready summary dict.

        Returns
        -------
        dict with keys:
            - ``n_alerts``: number of alerting cells
            - ``overall_pass``: True when n_alerts == 0
            - ``worst_cell``: dict for the cell with the largest |relative_bias|
              among alerting cells, or None if no alerts
            - ``n_cells_evaluated``: cells with enough exposure to evaluate
            - ``n_cells_skipped``: cells with insufficient exposure
            - ``period_index``: update counter
        """
        worst: dict[str, Any] | None = None
        if self.alerts:
            worst_cell = max(self.alerts, key=lambda c: abs(c.relative_bias))
            worst = worst_cell.to_dict()

        return {
            "n_alerts": len(self.alerts),
            "overall_pass": len(self.alerts) == 0,
            "worst_cell": worst,
            "n_cells_evaluated": self.n_cells_evaluated,
            "n_cells_skipped": self.n_cells_skipped,
            "period_index": self.period_index,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation of the full result.

        Suitable for JSON serialisation, MLflow logging, or database storage.
        """
        return {
            "alerts": [a.to_dict() for a in self.alerts],
            "cell_table": self.cell_table.to_dicts(),
            "n_cells_evaluated": self.n_cells_evaluated,
            "n_cells_skipped": self.n_cells_skipped,
            "period_index": self.period_index,
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Threshold dataclass (follows PSIThresholds / AERatioThresholds pattern)
# ---------------------------------------------------------------------------


@dataclass
class MulticalibThresholds:
    """Alert thresholds for MulticalibrationMonitor.

    Attributes
    ----------
    min_relative_bias:
        Minimum |relative_bias| required for an alert. Default 0.05 (5%).
        A cell with 4.9% bias and high z-stat will not alert.
    min_z_abs:
        Minimum |z_stat| required for an alert. Default 1.96 (two-sided 95% CI).
        Use 1.645 for 90% CI in higher-frequency monitoring.
    min_exposure:
        Minimum total cell exposure (car-years) before the cell is evaluated.
        Cells below this threshold are skipped. Default 50.
    """

    min_relative_bias: float = 0.05
    min_z_abs: float = 1.96
    min_exposure: float = 50.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MulticalibrationMonitor:
    """Monitor per-(subgroup, premium-band) calibration in production.

    Answers: "is the model currently well-calibrated for subgroup g at premium
    band k?" — a question that neither PSI (distributional drift) nor A/E ratio
    (overall lift) can answer.

    The monitor bins predictions into exposure-weighted quantile bands at fit()
    time. These boundaries are frozen for the lifetime of the monitor, ensuring
    that (bin, group) cells are comparable across periods.

    Parameters
    ----------
    n_bins:
        Number of exposure-weighted prediction quantile bins. Default 10.
        More bins = finer granularity but fewer policies per cell.
    min_z_abs:
        Minimum |z-statistic| required for an alert. Default 1.96.
    min_relative_bias:
        Minimum |relative_bias| = |(observed - expected) / expected| required
        for an alert. Default 0.05 (5%).
    min_exposure:
        Minimum exposure (car-years) per cell before it is evaluated.
        Cells below this are skipped. Default 50.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 5000
    >>> y_pred = rng.gamma(2, 0.05, n)
    >>> y_true = rng.poisson(y_pred * 1.0)  # well-calibrated
    >>> groups = rng.choice(["A", "B", "C"], n)
    >>> exposure = rng.uniform(0.5, 2.0, n)
    >>> monitor = MulticalibrationMonitor(n_bins=5, min_exposure=20)
    >>> monitor.fit(y_true, y_pred, groups, exposure=exposure)
    MulticalibrationMonitor(n_bins=5, fitted, 0 periods)
    >>> result = monitor.update(y_true, y_pred, groups, exposure=exposure)
    >>> result.summary()["overall_pass"]
    True
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_z_abs: float = 1.96,
        min_relative_bias: float = 0.05,
        min_exposure: float = 50.0,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if min_exposure <= 0:
            raise ValueError(f"min_exposure must be > 0, got {min_exposure}")

        self.n_bins = n_bins
        self.min_z_abs = min_z_abs
        self.min_relative_bias = min_relative_bias
        self.min_exposure = min_exposure

        self._thresholds = MulticalibThresholds(
            min_relative_bias=min_relative_bias,
            min_z_abs=min_z_abs,
            min_exposure=min_exposure,
        )

        # Set at fit() time — never modified after that
        self._bin_edges: np.ndarray | None = None  # shape (n_bins + 1,)
        self._is_fitted: bool = False
        self._period_index: int = 0
        self._history: list[MulticalibrationResult] = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        groups: npt.ArrayLike,
        exposure: npt.ArrayLike | None = None,
    ) -> "MulticalibrationMonitor":
        """Establish bin boundaries from reference data.

        The exposure-weighted quantile boundaries of y_pred are computed here
        and frozen for all subsequent update() calls. This is what makes
        cross-period comparison valid — a policy in bin 3 this month is in the
        same premium band as a policy in bin 3 last month.

        Parameters
        ----------
        y_true:
            Observed loss rates or claim counts. Shape (n,).
        y_pred:
            Model predictions (rates). Shape (n,). Must be strictly positive.
        groups:
            Subgroup labels. Shape (n,). Any hashable type.
        exposure:
            Policy durations (car-years). Shape (n,). If None, assumed uniform.

        Returns
        -------
        MulticalibrationMonitor
            Self, for method chaining.
        """
        y_arr, y_pred_arr, w = _validate_monitor_inputs(y_true, y_pred, groups, exposure)
        self._bin_edges = _exposure_weighted_quantile_edges(y_pred_arr, w, self.n_bins)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        groups: npt.ArrayLike,
        exposure: npt.ArrayLike | None = None,
    ) -> MulticalibrationResult:
        """Evaluate multicalibration for a new period batch.

        For each (bin_k, group_l) cell with sufficient exposure, computes the
        observed/expected ratio and a Poisson z-statistic. Fires an alert when
        both the relative bias and z-stat exceed their respective thresholds.

        Parameters
        ----------
        y_true:
            Observed loss rates or claim counts for the new period. Shape (n,).
        y_pred:
            Model predictions for the new period. Shape (n,).
        groups:
            Subgroup labels for the new period. Shape (n,).
        exposure:
            Policy durations for the new period. If None, assumed uniform.

        Returns
        -------
        MulticalibrationResult
            Cell-level statistics, alert list, and summary.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "MulticalibrationMonitor must be fitted before calling update(). "
                "Call fit() with reference period data first."
            )

        y_arr, y_pred_arr, w = _validate_monitor_inputs(y_true, y_pred, groups, exposure)
        groups_arr = np.asarray(groups)

        # Assign each observation to a bin using fixed reference boundaries
        bin_indices = _assign_bins(y_pred_arr, self._bin_edges)

        unique_groups = sorted(set(groups_arr.tolist()), key=lambda g: str(g))
        cells: list[MulticalibCell] = []
        n_evaluated = 0
        n_skipped = 0

        for b in range(self.n_bins):
            for g in unique_groups:
                mask = (bin_indices == b) & (groups_arr == g)
                cell_w = w[mask]
                total_w = float(cell_w.sum())

                if total_w < self._thresholds.min_exposure:
                    n_skipped += 1
                    continue

                cell_y = y_arr[mask]
                cell_yhat = y_pred_arr[mask]

                observed = float(np.sum(cell_w * cell_y) / total_w)
                expected = float(np.sum(cell_w * cell_yhat) / total_w)

                if expected <= 0.0:
                    # Degenerate cell — skip with a warning
                    warnings.warn(
                        f"Cell (bin={b}, group={g}) has expected <= 0 after "
                        "exposure weighting. Skipping.",
                        stacklevel=2,
                    )
                    n_skipped += 1
                    continue

                ae_ratio = observed / expected
                relative_bias = (observed - expected) / expected

                # Poisson z-statistic: b_hat * sqrt(w / pi_bar)
                # This is asymptotically N(0,1) under no-miscalibration H0.
                z_stat = (observed - expected) * np.sqrt(total_w / expected)

                alert = (
                    abs(relative_bias) >= self._thresholds.min_relative_bias
                    and abs(z_stat) >= self._thresholds.min_z_abs
                )

                cells.append(MulticalibCell(
                    bin_idx=b,
                    group=g,
                    n_exposure=total_w,
                    observed=observed,
                    expected=expected,
                    AE_ratio=ae_ratio,
                    relative_bias=relative_bias,
                    z_stat=z_stat,
                    alert=alert,
                ))
                n_evaluated += 1

        alerts = [c for c in cells if c.alert]

        # Build Polars DataFrame
        cell_table = _cells_to_polars(cells)

        self._period_index += 1
        result = MulticalibrationResult(
            alerts=alerts,
            cell_table=cell_table,
            n_cells_evaluated=n_evaluated,
            n_cells_skipped=n_skipped,
            period_index=self._period_index,
        )
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def history(self) -> list[MulticalibrationResult]:
        """Return the list of all update() results in chronological order."""
        return list(self._history)

    def period_summary(self) -> pl.DataFrame:
        """Return a Polars DataFrame with one row per update() call.

        Columns: period_index, n_alerts, n_cells_evaluated, n_cells_skipped,
        overall_pass.
        """
        if not self._history:
            return pl.DataFrame(schema={
                "period_index": pl.Int64,
                "n_alerts": pl.Int64,
                "n_cells_evaluated": pl.Int64,
                "n_cells_skipped": pl.Int64,
                "overall_pass": pl.Boolean,
            })

        rows = [
            {
                "period_index": r.period_index,
                "n_alerts": len(r.alerts),
                "n_cells_evaluated": r.n_cells_evaluated,
                "n_cells_skipped": r.n_cells_skipped,
                "overall_pass": len(r.alerts) == 0,
            }
            for r in self._history
        ]
        return pl.DataFrame(rows)

    @property
    def bin_edges(self) -> np.ndarray | None:
        """The frozen prediction quantile bin boundaries, or None before fit()."""
        return self._bin_edges

    @property
    def is_fitted(self) -> bool:
        """True after fit() has been called."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"MulticalibrationMonitor(n_bins={self.n_bins}, "
            f"{status}, {self._period_index} periods)"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_monitor_inputs(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    groups: npt.ArrayLike,
    exposure: npt.ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce all inputs to float64 / object arrays."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    groups_arr = np.asarray(groups)

    if y_arr.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got shape {y_arr.shape}")
    if y_pred_arr.ndim != 1:
        raise ValueError(f"y_pred must be 1-dimensional, got shape {y_pred_arr.shape}")
    if groups_arr.ndim != 1:
        raise ValueError(f"groups must be 1-dimensional, got shape {groups_arr.shape}")
    n = len(y_arr)
    if len(y_pred_arr) != n:
        raise ValueError(
            f"y_true and y_pred must have the same length: {n} vs {len(y_pred_arr)}"
        )
    if len(groups_arr) != n:
        raise ValueError(
            f"y_true and groups must have the same length: {n} vs {len(groups_arr)}"
        )
    if n < 2:
        raise ValueError("At least 2 observations are required")

    if np.any(y_pred_arr <= 0):
        bad = int(np.sum(y_pred_arr <= 0))
        raise ValueError(
            f"All y_pred values must be strictly positive. "
            f"Found {bad} values <= 0."
        )

    if exposure is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(exposure, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError(f"exposure must be 1-dimensional, got shape {w.shape}")
        if len(w) != n:
            raise ValueError(
                f"exposure must have the same length as y_true: {len(w)} vs {n}"
            )
        if np.any(w <= 0):
            raise ValueError("All exposure values must be strictly positive.")

    return y_arr, y_pred_arr, w


def _exposure_weighted_quantile_edges(
    y_pred: np.ndarray,
    w: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Compute exposure-weighted quantile bin boundaries for y_pred.

    Returns an array of shape (n_bins + 1,) where edges[0] = -inf and
    edges[-1] = +inf. This matches the approach in
    ``calibration._autocal._compute_per_bin``.

    Parameters
    ----------
    y_pred:
        Predictions. Shape (n,).
    w:
        Exposure weights. Shape (n,).
    n_bins:
        Number of bins.

    Returns
    -------
    np.ndarray
        Bin edges of shape (n_bins + 1,). First element is -inf,
        last element is +inf, so every prediction maps into exactly one bin.
    """
    sort_idx = np.argsort(y_pred)
    cumulative_w = np.cumsum(w[sort_idx])
    total_w = cumulative_w[-1]

    # Quantile cut-points at evenly spaced fractions of total exposure
    # (same logic as _compute_per_bin in calibration)
    fractions = np.linspace(0.0, total_w, n_bins + 1)

    # For each interior cut-point, find the y_pred value at that cumulative weight
    interior_edges = []
    y_sorted = y_pred[sort_idx]
    for frac in fractions[1:-1]:
        idx = int(np.searchsorted(cumulative_w, frac, side="left"))
        idx = min(idx, len(y_sorted) - 1)
        interior_edges.append(y_sorted[idx])

    edges = np.concatenate([[-np.inf], interior_edges, [np.inf]])
    return edges


def _assign_bins(y_pred: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each prediction to a bin index using frozen edges.

    Uses np.searchsorted on the interior edges (edges[1:-1]) so that
    bin 0 corresponds to edges[0] <= y_pred < edges[1], etc.

    Parameters
    ----------
    y_pred:
        Predictions to bin.
    edges:
        Bin edges of shape (n_bins + 1,) as returned by
        _exposure_weighted_quantile_edges. First must be -inf, last +inf.

    Returns
    -------
    np.ndarray of int
        Bin indices in [0, n_bins - 1].
    """
    interior = edges[1:-1]
    # searchsorted gives the index in interior where y_pred would be inserted.
    # This maps directly to bin index (0-based).
    bins = np.searchsorted(interior, y_pred, side="left")
    return bins


def _cells_to_polars(cells: list[MulticalibCell]) -> pl.DataFrame:
    """Convert a list of MulticalibCell to a Polars DataFrame."""
    if not cells:
        return pl.DataFrame(schema={
            "bin_idx": pl.Int64,
            "group": pl.String,
            "n_exposure": pl.Float64,
            "observed": pl.Float64,
            "expected": pl.Float64,
            "AE_ratio": pl.Float64,
            "relative_bias": pl.Float64,
            "z_stat": pl.Float64,
            "alert": pl.Boolean,
        })

    rows = [c.to_dict() for c in cells]
    # Polars infers types from the data. Cast group to String explicitly so the
    # dtype is consistent regardless of whether groups are ints or strings.
    df = pl.DataFrame(rows)
    if "group" in df.columns:
        df = df.with_columns(pl.col("group").cast(pl.String))
    return df
