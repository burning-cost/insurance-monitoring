"""
Model calibration monitoring for insurance pricing models.

Calibration is the most directly business-relevant performance dimension for
insurance pricing: a model that is systematically underpredicting by 5% will
underprice by 5%, which shows up directly in loss ratio.

The Actual/Expected (A/E) ratio is the universal actuarial calibration metric.
It has one crucial complication that generic ML calibration tools miss: IBNR
(Incurred But Not Reported). Recent accident periods are systematically
under-reported because claims take months or years to reach their ultimate
development. Always check whether your actuals are adequately developed before
comparing against these metrics — a 12-month minimum development lag is the
safe default for motor.

The Murphy score decomposition (arXiv 2510.04556) frames calibration as one
of three separable components: Calibration + Resolution + Uncertainty = Score.
A model with good calibration but poor resolution (Gini) needs a refit.
A model with poor calibration only needs a recalibration (cheap).

References
----------
- arXiv 2510.04556, Section 4: Murphy decomposition and calibration tests
- Milliman P&C actuarial reporting article: separate A/E for frequency vs severity
- Hosmer-Lemeshow test: Hosmer & Lemeshow (2000), Applied Logistic Regression
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats


ArrayLike = Union[np.ndarray, pl.Series]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_numpy_optional(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return _to_numpy(x)


def ae_ratio(
    actual: ArrayLike,
    predicted: ArrayLike,
    exposure: Optional[ArrayLike] = None,
    segments: Optional[ArrayLike] = None,
) -> Union[float, pl.DataFrame]:
    """Compute Actual/Expected ratio, optionally segmented.

    A/E = sum(actual) / sum(predicted * exposure)

    For frequency models, ``actual`` is claim count and ``predicted`` is the
    predicted frequency (claims per unit exposure). ``exposure`` is earned
    car-years (or policy-years). For severity models, ``exposure`` can be
    omitted and ``predicted`` is the predicted average severity.

    When ``segments`` is provided, returns a per-segment breakdown as a
    Polars DataFrame — the key diagnostic for identifying which rating factors
    have concept drift.

    Parameters
    ----------
    actual:
        Observed values (claim counts for frequency, claim amounts for severity).
    predicted:
        Model predictions. For frequency models this is predicted rate
        (per unit exposure). For severity models this is predicted average cost.
    exposure:
        Optional earned exposure weights (e.g., car-years). When provided,
        expected = sum(predicted * exposure).
    segments:
        Optional categorical array for grouping. When provided, returns
        per-segment A/E table as a Polars DataFrame.

    Returns
    -------
    float
        Aggregate A/E ratio when no segments provided.
    pl.DataFrame
        Per-segment table with columns ``segment``, ``actual``,
        ``expected``, ``ae_ratio``, ``n_policies`` when segments provided.

    Examples
    --------
    Aggregate A/E for a frequency model::

        from insurance_monitoring.calibration import ae_ratio
        import numpy as np

        actual = np.array([120, 0, 1, 0, 2, 1])      # claim counts
        predicted = np.array([0.08, 0.05, 0.12, 0.04, 0.09, 0.07])  # freq
        exposure = np.array([1.0, 0.5, 1.0, 1.0, 0.8, 0.9])
        ae = ae_ratio(actual, predicted, exposure=exposure)
        # ae ≈ 1.05 means model slightly underpredicted (actuals exceeded predictions)

    Segmented A/E by driver age band::

        segments = np.array(["17-25", "17-25", "26-50", "26-50", "51-70", "51-70"])
        ae_ratio(actual, predicted, exposure=exposure, segments=segments)
    """
    act = _to_numpy(actual)
    pred = _to_numpy(predicted)
    exp = _to_numpy_optional(exposure)

    if len(act) == 0:
        raise ValueError("actual must be non-empty")
    if len(act) != len(pred):
        raise ValueError(f"actual length ({len(act)}) != predicted length ({len(pred)})")
    if exp is not None and len(exp) != len(act):
        raise ValueError(f"exposure length ({len(exp)}) != actual length ({len(act)})")

    if segments is None:
        expected = pred * exp if exp is not None else pred
        total_actual = float(np.sum(act))
        total_expected = float(np.sum(expected))
        if total_expected == 0:
            raise ValueError("Sum of expected values is zero — check your predictions")
        return total_actual / total_expected

    # Segmented A/E
    seg = np.asarray(segments)
    if len(seg) != len(act):
        raise ValueError(f"segments length ({len(seg)}) != actual length ({len(act)})")

    expected = pred * exp if exp is not None else pred
    unique_segs = np.unique(seg)
    rows = []
    for s in unique_segs:
        mask = seg == s
        total_act = float(np.sum(act[mask]))
        total_exp = float(np.sum(expected[mask]))
        n = int(np.sum(mask))
        ratio = total_act / total_exp if total_exp > 0 else float("nan")
        rows.append({
            "segment": str(s),
            "actual": total_act,
            "expected": total_exp,
            "ae_ratio": ratio,
            "n_policies": n,
        })

    return pl.DataFrame(
        rows,
        schema={
            "segment": pl.String,
            "actual": pl.Float64,
            "expected": pl.Float64,
            "ae_ratio": pl.Float64,
            "n_policies": pl.Int64,
        },
    )


def ae_ratio_ci(
    actual: ArrayLike,
    predicted: ArrayLike,
    exposure: Optional[ArrayLike] = None,
    alpha: float = 0.05,
    method: str = "poisson",
) -> dict[str, float]:
    """Compute A/E ratio with confidence interval.

    For frequency models, the natural framework is Poisson: the sum of
    observed claim counts is approximately Poisson with mean = sum of expected
    counts. This gives an exact confidence interval via the chi-squared
    distribution, which is preferable to the normal approximation when
    the number of claims is small (< 100).

    For severity, use ``method='normal'`` which applies a standard
    bootstrap-style normal approximation.

    Parameters
    ----------
    actual:
        Observed claim counts (for Poisson) or amounts (for normal).
    predicted:
        Model predictions (frequency or severity).
    exposure:
        Optional exposure weights.
    alpha:
        Significance level. Defaults to 0.05 for 95% CI.
    method:
        'poisson' (default, exact CI for count data) or 'normal'
        (normal approximation).

    Returns
    -------
    dict with keys:
        - ``ae``: point estimate
        - ``lower``: lower CI bound
        - ``upper``: upper CI bound
        - ``n_claims``: total observed claims (for Poisson)
        - ``n_expected``: total expected claims

    Examples
    --------
    ::

        from insurance_monitoring.calibration import ae_ratio_ci
        import numpy as np

        rng = np.random.default_rng(42)
        exposure = np.ones(5000)
        predicted = rng.uniform(0.05, 0.15, 5000)
        actual = rng.poisson(predicted * exposure)
        result = ae_ratio_ci(actual, predicted, exposure=exposure)
        # result['lower'], result['ae'], result['upper']
    """
    act = _to_numpy(actual)
    pred = _to_numpy(predicted)
    exp = _to_numpy_optional(exposure)

    if len(act) == 0:
        raise ValueError("actual must be non-empty")
    if len(act) != len(pred):
        raise ValueError(f"actual and predicted must have the same length")

    expected_per_obs = pred * exp if exp is not None else pred
    n_observed = float(np.sum(act))
    n_expected = float(np.sum(expected_per_obs))

    if n_expected == 0:
        raise ValueError("Sum of expected is zero")

    ae = n_observed / n_expected

    if method == "poisson":
        # Exact Poisson CI for the observed count n_observed,
        # then scale by 1/n_expected to get CI on the ratio.
        # Uses chi-squared quantiles: Garwood (1936) intervals.
        n_obs_int = int(round(n_observed))
        if n_obs_int == 0:
            lower_count = 0.0
        else:
            lower_count = 0.5 * stats.chi2.ppf(alpha / 2, 2 * n_obs_int)
        upper_count = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (n_obs_int + 1))
        lower = lower_count / n_expected
        upper = upper_count / n_expected
    elif method == "normal":
        # Normal approximation: SE(A/E) = ae / sqrt(n_expected)
        # Under Poisson, Var(sum(Y)) = sum(mu), so Var(A/E) = sum(mu) / sum(mu)^2 = 1/n_expected.
        # SE(A/E) = ae / sqrt(n_expected) — NOT ae / sqrt(n_policies).
        # Using n_policies (number of rows) makes the CI ~4.5x too narrow
        # when average expected claims per policy is ~0.05.
        se = ae / math.sqrt(n_expected) if n_expected > 0 else float("inf")
        z = stats.norm.ppf(1 - alpha / 2)
        lower = ae - z * se
        upper = ae + z * se
    else:
        raise ValueError(f"method must be 'poisson' or 'normal', got '{method}'")

    return {
        "ae": ae,
        "lower": float(lower),
        "upper": float(upper),
        "n_claims": n_observed,
        "n_expected": n_expected,
    }


def calibration_curve(
    actual: ArrayLike,
    predicted: ArrayLike,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pl.DataFrame:
    """Compute a binned calibration curve.

    Sorts observations into bins by predicted value, then computes the mean
    observed and mean predicted within each bin. A perfectly calibrated model
    has points on the diagonal (mean predicted = mean actual in every bin).

    This is the visual diagnostic counterpart to A/E ratio: it shows whether
    miscalibration is uniform (shift the intercept) or varies across the
    predicted score range (partial refit needed).

    Parameters
    ----------
    actual:
        Observed values (0/1 binary or claim counts per exposure).
    predicted:
        Model predictions (probabilities or rates).
    n_bins:
        Number of bins. Defaults to 10.
    strategy:
        'quantile' (equal-frequency bins, default) or 'uniform' (equal-width).

    Returns
    -------
    pl.DataFrame
        Columns: ``bin``, ``mean_predicted``, ``mean_actual``, ``n``,
        ``ae_ratio`` (mean_actual / mean_predicted).

    Examples
    --------
    ::

        from insurance_monitoring.calibration import calibration_curve
        import numpy as np

        rng = np.random.default_rng(0)
        predicted = rng.beta(2, 5, 1000)
        actual = rng.binomial(1, predicted)
        calibration_curve(actual, predicted, n_bins=10)
    """
    act = _to_numpy(actual)
    pred = _to_numpy(predicted)

    if len(act) != len(pred):
        raise ValueError(f"actual and predicted must have the same length")
    if len(act) == 0:
        raise ValueError("arrays must be non-empty")

    if strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(pred, quantiles)
    elif strategy == "uniform":
        bin_edges = np.linspace(pred.min(), pred.max(), n_bins + 1)
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got '{strategy}'")

    bin_edges = np.unique(bin_edges)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    rows = []
    bin_indices = np.digitize(pred, bin_edges[1:-1])
    n_effective_bins = len(bin_edges) - 1

    for i in range(n_effective_bins):
        mask = bin_indices == i
        n = int(np.sum(mask))
        if n == 0:
            continue
        mean_pred = float(np.mean(pred[mask]))
        mean_act = float(np.mean(act[mask]))
        ratio = mean_act / mean_pred if mean_pred > 0 else float("nan")
        rows.append({
            "bin": i + 1,
            "mean_predicted": mean_pred,
            "mean_actual": mean_act,
            "n": n,
            "ae_ratio": ratio,
        })

    return pl.DataFrame(
        rows,
        schema={
            "bin": pl.Int64,
            "mean_predicted": pl.Float64,
            "mean_actual": pl.Float64,
            "n": pl.Int64,
            "ae_ratio": pl.Float64,
        },
    )


def hosmer_lemeshow(
    actual: ArrayLike,
    predicted: ArrayLike,
    n_bins: int = 10,
) -> dict[str, float]:
    """Hosmer-Lemeshow goodness-of-fit test.

    Tests whether the binned calibration curve departs significantly from the
    diagonal. Groups observations into deciles of predicted probability and
    computes a chi-squared statistic.

    Commonly used for binary outcome models (e.g., claim occurrence). For
    count models (Poisson frequency), the A/E ratio with Poisson CI is
    more appropriate.

    A small p-value (< 0.05) indicates poor calibration at the bin level —
    but note that with very large samples the test will reject even trivial
    miscalibration. Always report the H statistic alongside p-value.

    Parameters
    ----------
    actual:
        Binary observed outcomes (0/1).
    predicted:
        Predicted probabilities.
    n_bins:
        Number of groups (deciles by default). The original Hosmer-Lemeshow
        paper used 10.

    Returns
    -------
    dict with keys:
        - ``statistic``: HL chi-squared statistic
        - ``p_value``: p-value from chi-squared distribution with (n_bins - 2) df
        - ``df``: degrees of freedom

    Examples
    --------
    ::

        from insurance_monitoring.calibration import hosmer_lemeshow
        import numpy as np

        rng = np.random.default_rng(0)
        p = rng.beta(2, 5, 2000)
        y = rng.binomial(1, p)
        result = hosmer_lemeshow(y, p)
        result['p_value']  # should be large (model is well-calibrated by construction)
    """
    act = _to_numpy(actual)
    pred = _to_numpy(predicted)

    if len(act) != len(pred):
        raise ValueError("actual and predicted must have the same length")
    if len(act) == 0:
        raise ValueError("arrays must be non-empty")

    # Sort by predicted probability
    order = np.argsort(pred)
    act_sorted = act[order]
    pred_sorted = pred[order]

    # Split into bins of equal size
    bins = np.array_split(np.arange(len(act)), n_bins)

    hl_stat = 0.0
    for bin_idx in bins:
        n_bin = len(bin_idx)
        if n_bin == 0:
            continue
        observed = float(np.sum(act_sorted[bin_idx]))
        expected = float(np.sum(pred_sorted[bin_idx]))
        expected_neg = float(np.sum(1.0 - pred_sorted[bin_idx]))
        observed_neg = n_bin - observed

        if expected > 0:
            hl_stat += (observed - expected) ** 2 / expected
        if expected_neg > 0:
            hl_stat += (observed_neg - expected_neg) ** 2 / expected_neg

    df = max(n_bins - 2, 1)
    p_value = float(1.0 - stats.chi2.cdf(hl_stat, df))

    return {
        "statistic": float(hl_stat),
        "p_value": p_value,
        "df": df,
    }
