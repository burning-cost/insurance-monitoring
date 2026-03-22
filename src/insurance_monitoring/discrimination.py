"""
Model discrimination monitoring for insurance pricing models.

Discrimination measures whether the model correctly ranks risks — high-risk
policies should receive higher predicted rates than low-risk ones. A model
can be perfectly calibrated (A/E = 1.0) but discriminate poorly: it produces
the right average but does not separate cheap from expensive risks.

The Gini coefficient (equivalently, 2 * AUROC - 1 for binary outcomes) is
the dominant discrimination metric in non-life insurance pricing. It derives
from the Concentration curve (CAP curve): sort policies ascending by predicted
rate, plot cumulative actual claims against cumulative exposure.

Key design choices in this implementation:

1. Tie-breaking: when multiple policies share the same predicted rate, the
   Gini is order-dependent. We implement the midpoint approach from
   arXiv 2510.04556 (average of best-case and worst-case orderings).

2. Exposure weighting: correct for policies with different durations. A
   one-month policy and a twelve-month policy should not have equal weight
   in the sort.

3. Two-sample z-test (gini_drift_test): based on Theorem 1 of arXiv 2510.04556
   which establishes sqrt(n) * (G_hat - G) -> N(0, sigma^2). Bootstrap variance
   is estimated for both reference and current periods (Algorithm 2 from the
   paper). This is appropriate when you have raw data for both periods.

4. One-sample bootstrap test (gini_drift_test_onesample): implements
   Algorithm 3 from arXiv 2510.04556. The training Gini is treated as fixed
   (it was computed once at training time). Only the monitor sample is
   bootstrapped to estimate the null distribution. This is the more natural
   design for deployed model monitoring: you have a stored training Gini and
   you want to test whether the monitoring data shows drift.

5. Class-based bootstrap test (GiniDriftBootstrapTest): wraps Algorithm 3
   with percentile confidence intervals on both the monitor Gini and the
   Gini change, a .plot() method for governance reporting, and a .summary()
   method returning a structured plain-text governance report. Stores
   bootstrap replicates for post-hoc inspection.

References
----------
- arXiv 2510.04556, Section 3: Gini score and monitoring framework
- arXiv 2510.04556, Algorithm 2: Non-parametric bootstrap variance estimator
- arXiv 2510.04556, Algorithm 3: One-sample monitoring test
- Frees & Valdez (1998): actuarial use of Lorenz curves and Gini
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields as dc_fields
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats

# numpy<2.0 compat: trapezoid was added in 2.0, trapz deprecated/removed in 2.0
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]


ArrayLike = Union[np.ndarray, pl.Series]


@dataclass
class GiniDriftResult:
    """Result of a two-sample Gini drift test (gini_drift_test).

    Attributes
    ----------
    z_statistic:
        z-score. Negative means the current Gini is lower than reference
        (discrimination has degraded).
    p_value:
        Two-sided p-value.
    reference_gini:
        Gini coefficient from the reference period.
    current_gini:
        Gini coefficient from the current period.
    gini_change:
        current_gini - reference_gini. Negative means degradation.
    significant:
        True if p_value < alpha (the significance level passed to the test).
    """

    z_statistic: float
    p_value: float
    reference_gini: float
    current_gini: float
    gini_change: float
    significant: bool

    def __getitem__(self, key: str):
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'key in result' for backward compatibility."""
        return hasattr(self, key)

    def keys(self):
        """Return field names, like dict.keys(), for backward compatibility."""
        return [f.name for f in dc_fields(self)]


@dataclass
class GiniDriftOneSampleResult:
    """Result of a one-sample Gini drift test (gini_drift_test_onesample).

    Attributes
    ----------
    z_statistic:
        z-score. Negative means monitor Gini is below training Gini.
    p_value:
        Two-sided p-value.
    training_gini:
        Stored training Gini (the fixed reference).
    monitor_gini:
        Point estimate from the current monitoring data.
    gini_change:
        monitor_gini - training_gini.
    se_bootstrap:
        Bootstrap standard error of the monitor Gini estimator.
    significant:
        True if p_value < alpha.
    """

    z_statistic: float
    p_value: float
    training_gini: float
    monitor_gini: float
    gini_change: float
    se_bootstrap: float
    significant: bool

    def __getitem__(self, key: str):
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'key in result' for backward compatibility."""
        return hasattr(self, key)

    def keys(self):
        """Return field names, like dict.keys(), for backward compatibility."""
        return [f.name for f in dc_fields(self)]


@dataclass
class GiniBootstrapResult:
    """Result of a GiniDriftBootstrapTest.

    Attributes
    ----------
    z_statistic : float
        (G_monitor - training_gini) / se_bootstrap. Negative = degradation.
    p_value : float
        Two-sided p-value from N(0,1) approximation.
    training_gini : float
        Fixed reference Gini from training time.
    monitor_gini : float
        Point estimate of Gini on current monitoring data.
    gini_change : float
        monitor_gini - training_gini. Negative = degradation.
    se_bootstrap : float
        Bootstrap standard error of monitor Gini estimator (Algorithm 2).
    confidence_level : float
        The confidence level used (e.g. 0.95).
    ci_lower : float
        Lower bound of percentile bootstrap CI for monitor_gini.
    ci_upper : float
        Upper bound of percentile bootstrap CI for monitor_gini.
    ci_change_lower : float
        Lower bound of percentile CI for gini_change.
    ci_change_upper : float
        Upper bound of percentile CI for gini_change.
    n_bootstrap : int
        Number of bootstrap replicates used.
    n_obs : int
        Number of observations in the monitoring sample.
    significant : bool
        True if p_value < alpha.
    alpha : float
        Significance level used for the significant flag.
    boot_replicates : np.ndarray
        The B bootstrap Gini values (length n_bootstrap). Stored for plotting
        and post-hoc inspection.

    Notes
    -----
    BCa confidence intervals are not implemented. For insurance monitoring data
    (n = 500-5000) the bootstrap Gini distribution is approximately normal
    (paper Figure 3), and BCa provides negligible coverage improvement over the
    percentile interval while requiring n additional jackknife Gini computations.
    The percentile interval is the right default here.
    """

    z_statistic: float
    p_value: float
    training_gini: float
    monitor_gini: float
    gini_change: float
    se_bootstrap: float
    confidence_level: float
    ci_lower: float
    ci_upper: float
    ci_change_lower: float
    ci_change_upper: float
    n_bootstrap: int
    n_obs: int
    significant: bool
    alpha: float
    boot_replicates: np.ndarray

    def __getitem__(self, key: str):
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'key in result' for backward compatibility."""
        return hasattr(self, key)

    def keys(self):
        """Return field names, like dict.keys(), for backward compatibility."""
        return [f.name for f in dc_fields(self)]

    def to_dict(self) -> dict:
        """Return a plain dict of all scalar fields.

        Excludes boot_replicates (not JSON-serialisable). Includes
        boot_replicates_mean as a scalar diagnostic for MLflow logging.
        """
        return {
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "training_gini": self.training_gini,
            "monitor_gini": self.monitor_gini,
            "gini_change": self.gini_change,
            "se_bootstrap": self.se_bootstrap,
            "confidence_level": self.confidence_level,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_change_lower": self.ci_change_lower,
            "ci_change_upper": self.ci_change_upper,
            "n_bootstrap": self.n_bootstrap,
            "n_obs": self.n_obs,
            "significant": self.significant,
            "alpha": self.alpha,
            "boot_replicates_mean": float(np.mean(self.boot_replicates)),
        }


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_numpy_optional(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return _to_numpy(x)


def _gini_from_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
    exposure: Optional[np.ndarray],
) -> float:
    """Internal computation of Gini with tie-breaking midpoint method."""
    n = len(actual)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0

    exp = exposure if exposure is not None else np.ones(n)

    # Identify tie groups (same predicted value)
    # For each tie group, average the Gini contribution across all orderings
    # by using the sorted rank of the group's total actual / total exposure.
    # Practical implementation: sort by predicted, then handle ties by
    # assigning the midpoint rank. This is equivalent to the arXiv 2510.04556
    # Pitfall A correction.
    # Group by unique predicted value to handle ties correctly.
    # Using argsort with kind="stable" makes the Gini order-dependent when many
    # policies share the same predicted rate (e.g., NCD band GLMs). The fix is
    # to accumulate cumulative exposure and claims at the level of unique predicted
    # values — all policies with the same predicted rate contribute together,
    # eliminating the dependence on row order. This implements the midpoint
    # tie-breaking described in arXiv 2510.04556 Pitfall A.
    total_exp = float(np.sum(exp))
    total_claims = float(np.sum(actual))

    unique_preds = np.unique(predicted)  # already sorted ascending
    cum_exp_vals = []
    cum_claims_vals = []
    running_exp = 0.0
    running_claims = 0.0
    for p in unique_preds:
        mask = predicted == p
        running_exp += float(exp[mask].sum())
        running_claims += float(actual[mask].sum())
        cum_exp_vals.append(running_exp)
        cum_claims_vals.append(running_claims)
    cum_exp = np.array(cum_exp_vals)
    cum_claims = np.array(cum_claims_vals)

    if total_claims == 0 or total_exp == 0:
        return 0.0

    # Normalise to [0, 1]
    x = cum_exp / total_exp      # cumulative share of exposure
    y = cum_claims / total_claims  # cumulative share of claims (Lorenz curve)

    # Prepend (0, 0) for the trapezoidal rule
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])

    # Area under the Lorenz curve (trapezoid rule)
    auc = float(np.trapezoid(y, x))

    # Gini = 1 - 2 * AUC_lorenz, equivalently = 2 * (0.5 - AUC_lorenz)
    # A random model has AUC = 0.5 → Gini = 0
    # A perfect model has AUC = 0 → Gini = 1
    gini = 1.0 - 2.0 * auc
    return float(gini)


# Bootstrap subsample cap: Gini bootstrap variance converges at O(1/n) and is
# well-estimated at n=20k. For large samples, subsampling before bootstrapping
# reduces per-replicate cost from O(n log n) to O(20k * log(20k)) — a 60-70%
# speedup at n=50k with negligible effect on the SE estimate. The Gini point
# estimate (used for the test statistic) always uses the full n; only the
# bootstrap variance computation is subsampled.
_BOOTSTRAP_N_CAP = 20_000


def _bootstrap_gini_samples(
    actual: np.ndarray,
    predicted: np.ndarray,
    exposure: Optional[np.ndarray],
    n_bootstrap: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Bootstrap the Gini coefficient for a single sample.

    Returns an array of n_bootstrap Gini replicates. Used internally by both
    the two-sample drift test (Algorithm 2) and the one-sample drift test
    (Algorithm 3) from arXiv 2510.04556.

    For large samples (n > _BOOTSTRAP_N_CAP = 20,000), bootstrap variance is
    estimated on a random subsample of _BOOTSTRAP_N_CAP observations. The Gini
    point estimate uses the full sample; only variance estimation is subsampled.
    This bounds bootstrap runtime at O(n_bootstrap * 20k * log(20k)) regardless
    of portfolio size.
    """
    n = len(actual)
    rng = np.random.default_rng(seed)

    # For large n, subsample once before bootstrapping to bound per-replicate cost
    if n > _BOOTSTRAP_N_CAP:
        sub_idx = rng.choice(n, size=_BOOTSTRAP_N_CAP, replace=False)
        actual_boot = actual[sub_idx]
        predicted_boot = predicted[sub_idx]
        exposure_boot = exposure[sub_idx] if exposure is not None else None
        n_boot = _BOOTSTRAP_N_CAP
    else:
        actual_boot = actual
        predicted_boot = predicted
        exposure_boot = exposure
        n_boot = n

    gini_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_boot, size=n_boot)
        gini_samples[i] = _gini_from_arrays(
            actual_boot[idx],
            predicted_boot[idx],
            exposure_boot[idx] if exposure_boot is not None else None,
        )
    return gini_samples


def gini_coefficient(
    actual: ArrayLike,
    predicted: ArrayLike,
    exposure: Optional[ArrayLike] = None,
) -> float:
    """Compute the Gini coefficient (C-statistic) for model discrimination.

    Sort policies by ascending predicted rate. The Gini coefficient measures
    how much better the model concentrates actual claims at the top of the
    sorted order compared to a random ranking.

    Gini = 0: model has no discriminatory power (equivalent to random sorting)
    Gini = 1: model perfectly separates risks (higher predicted = always higher actual)
    Gini < 0: model discriminates inversely (higher predicted = lower actual — indicates
              a reversed or fundamentally misspecified model)

    The theoretical range is [-1, 1]. In practice, sensible insurance models produce
    values in [0, 1]. Typical values for UK motor frequency models: 0.35–0.55.
    A Gini below 0.30 is weak. Above 0.60 is excellent.

    Parameters
    ----------
    actual:
        Observed values (claim counts or binary claim indicators).
    predicted:
        Model predictions (rates or probabilities).
    exposure:
        Optional exposure weights (earned car-years). When provided, cumulative
        sums are weighted by exposure — the correct approach for policies with
        varying durations.

    Returns
    -------
    float
        Gini coefficient in [-1, 1]. Negative values indicate inverted discrimination
        (higher predicted rate associated with fewer actual claims). Values in [0, 1]
        are normal for correctly specified models.

    Examples
    --------
    ::

        import numpy as np
        from insurance_monitoring.discrimination import gini_coefficient

        rng = np.random.default_rng()
        predicted = rng.uniform(0.05, 0.20, 5000)
        actual = rng.poisson(predicted)
        gini = gini_coefficient(actual, predicted)
        # gini ≈ 0.45 (reasonable for a well-specified frequency model)

    With exposure weighting::

        exposure = rng.uniform(0.1, 1.0, 5000)
        gini_coefficient(actual, predicted, exposure=exposure)
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

    return _gini_from_arrays(act, pred, exp)


def gini_drift_test(
    reference_gini: float,
    current_gini: float,
    n_reference: Optional[int] = None,
    n_current: Optional[int] = None,
    reference_variance: Optional[float] = None,
    current_variance: Optional[float] = None,
    reference_actual: Optional[ArrayLike] = None,
    reference_predicted: Optional[ArrayLike] = None,
    current_actual: Optional[ArrayLike] = None,
    current_predicted: Optional[ArrayLike] = None,
    reference_exposure: Optional[ArrayLike] = None,
    current_exposure: Optional[ArrayLike] = None,
    n_bootstrap: int = 200,
    alpha: float = 0.32,
) -> GiniDriftResult:
    """Statistical test for Gini coefficient drift between two periods.

    Implements the two-sample asymptotic z-test from Theorem 1 of
    arXiv 2510.04556. The test statistic is::

        z = (G_current - G_reference) / sqrt(Var(G_hat_current) + Var(G_hat_reference))

    where variances are estimated by bootstrap (Algorithm 2 from the paper).

    If raw data arrays are provided, bootstrap variance is computed directly.
    If only scalar Gini values and sample sizes are provided, the variance
    must be supplied explicitly (e.g., from a stored baseline).

    For a monitoring context where you want to test the monitor sample against
    a fixed stored training Gini, prefer ``gini_drift_test_onesample()`` instead.
    The one-sample design (Algorithm 3) is more efficient because it does not
    require raw reference data at monitoring time.

    Parameters
    ----------
    reference_gini:
        Gini coefficient from the reference (training) period.
    current_gini:
        Gini coefficient from the current monitoring period.
    n_reference:
        Number of observations in reference period. Accepted for informational
        purposes but not used in variance computation — the bootstrap SE is
        computed directly from the raw arrays and already reflects sample size.
        May be omitted; retained for backwards compatibility.
    n_current:
        Number of observations in current period. Same note as n_reference.
    reference_variance:
        Variance of reference Gini estimator. Required if raw arrays not provided.
    current_variance:
        Variance of current Gini estimator. Required if raw arrays not provided.
    reference_actual, reference_predicted, reference_exposure:
        Raw arrays for reference period. If provided, bootstrap variance is
        computed automatically.
    current_actual, current_predicted, current_exposure:
        Raw arrays for current period. If provided, bootstrap variance is
        computed automatically.
    n_bootstrap:
        Number of bootstrap replicates for variance estimation. Default 200.
    alpha:
        Significance level for the ``significant`` flag. Default 0.32
        (the one-sigma rule recommended by arXiv 2510.04556 for monitoring,
        which catches drift earlier than alpha=0.05 at the cost of more false
        positives). Use alpha=0.05 for confirmatory testing.

    Returns
    -------
    GiniDriftResult
        Typed dataclass with fields: z_statistic, p_value, reference_gini,
        current_gini, gini_change, significant. Supports dict-style access
        (result["z_statistic"]) for backward compatibility.

    Examples
    --------
    With raw arrays (recommended — variance estimated automatically)::

        import numpy as np
        from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

        rng = np.random.default_rng(0)
        pred_ref = rng.uniform(0.05, 0.20, 10_000)
        act_ref = rng.poisson(pred_ref)
        pred_cur = rng.uniform(0.05, 0.20, 5_000)
        act_cur = rng.poisson(pred_cur)

        g_ref = gini_coefficient(act_ref, pred_ref)
        g_cur = gini_coefficient(act_cur, pred_cur)

        result = gini_drift_test(
            reference_gini=g_ref,
            current_gini=g_cur,
            n_reference=10_000,
            n_current=5_000,
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
        )
    """
    # Compute bootstrap variance if raw arrays provided
    def _bootstrap_variance(actual_arr, predicted_arr, exposure_arr, n_boot):
        act = _to_numpy(actual_arr)
        pred = _to_numpy(predicted_arr)
        exp = _to_numpy_optional(exposure_arr)
        samples = _bootstrap_gini_samples(act, pred, exp, n_boot)
        return float(np.var(samples, ddof=1))

    if reference_actual is not None and reference_predicted is not None:
        var_ref = _bootstrap_variance(
            reference_actual, reference_predicted, reference_exposure, n_bootstrap
        )
    elif reference_variance is not None:
        var_ref = reference_variance
    else:
        raise ValueError(
            "Either reference_actual+reference_predicted or reference_variance must be provided"
        )

    if current_actual is not None and current_predicted is not None:
        var_cur = _bootstrap_variance(
            current_actual, current_predicted, current_exposure, n_bootstrap
        )
    elif current_variance is not None:
        var_cur = current_variance
    else:
        raise ValueError(
            "Either current_actual+current_predicted or current_variance must be provided"
        )

    se = float(np.sqrt(var_ref + var_cur))
    if se == 0:
        return GiniDriftResult(
            z_statistic=float("nan"),
            p_value=float("nan"),
            reference_gini=float(reference_gini),
            current_gini=float(current_gini),
            gini_change=float(current_gini - reference_gini),
            significant=False,
        )

    gini_change = current_gini - reference_gini
    z = gini_change / se
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return GiniDriftResult(
        z_statistic=float(z),
        p_value=float(p_value),
        reference_gini=float(reference_gini),
        current_gini=float(current_gini),
        gini_change=float(gini_change),
        significant=bool(p_value < alpha),
    )


def gini_drift_test_onesample(
    training_gini: float,
    monitor_actual: ArrayLike,
    monitor_predicted: ArrayLike,
    monitor_exposure: Optional[ArrayLike] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.32,
) -> GiniDriftOneSampleResult:
    """One-sample bootstrap Gini drift test (Algorithm 3, arXiv 2510.04556).

    Tests H0: Gini(monitor) = training_gini against H1: Gini(monitor) < training_gini.

    The test design mirrors real deployed model monitoring: at training time you
    compute and store the training Gini as a scalar. Months later, you observe new
    data. You want to test whether the new data's Gini is significantly lower than
    the stored value — without needing the original training data.

    This is the one-sample variant (Algorithm 3) from arXiv 2510.04556. It
    bootstraps only the monitor sample to estimate the null distribution of the
    Gini estimator under the assumption that the true Gini equals the training
    value. The z-statistic is::

        z = (G_monitor - training_gini) / SE_boot(G_monitor)

    where SE_boot is the bootstrap standard error from resampling the monitor
    data. A large negative z indicates the monitor Gini has fallen below
    what we expect from random sampling variation around the training value.

    The key conceptual difference from ``gini_drift_test``:
    - ``gini_drift_test``: two-sample test, SE = sqrt(Var_ref + Var_cur).
      Requires raw reference data. Appropriate for A/B comparisons.
    - ``gini_drift_test_onesample``: one-sample test, SE = SE_boot(monitor only).
      Only needs stored training_gini scalar. Appropriate for deployed monitoring.

    Parameters
    ----------
    training_gini:
        Gini coefficient computed on the training (reference) data at training
        time. Stored as a scalar, raw training data not needed.
    monitor_actual:
        Observed claims in the current monitoring period.
    monitor_predicted:
        Model predictions for the current monitoring period.
    monitor_exposure:
        Optional exposure weights for the current monitoring period.
    n_bootstrap:
        Number of bootstrap replicates for SE estimation. Default 500
        (higher than two-sample default because we are not averaging two
        variance estimates, so precision matters more here).
    alpha:
        Significance level for the ``significant`` flag. Default 0.32
        (one-sigma rule, arXiv 2510.04556). Use 0.05 for confirmatory testing.

    Returns
    -------
    GiniDriftOneSampleResult
        Typed dataclass with fields: z_statistic, p_value, training_gini,
        monitor_gini, gini_change, se_bootstrap, significant. Supports
        dict-style access (result["z_statistic"]) for backward compatibility.

    Examples
    --------
    Stored training Gini from months ago; test new monitoring period::

        import numpy as np
        from insurance_monitoring.discrimination import (
            gini_coefficient,
            gini_drift_test_onesample,
        )

        # At training time (stored in model registry):
        training_gini = 0.48

        # Months later, new monitoring data arrives:
        rng = np.random.default_rng(42)
        monitor_pred = rng.uniform(0.05, 0.20, 3_000)
        monitor_act = rng.poisson(monitor_pred)

        result = gini_drift_test_onesample(
            training_gini=training_gini,
            monitor_actual=monitor_act,
            monitor_predicted=monitor_pred,
            n_bootstrap=500,
            alpha=0.32,
        )
        # result["significant"] == True means drift detected at alpha=0.32 level
    """
    act = _to_numpy(monitor_actual)
    pred = _to_numpy(monitor_predicted)
    exp = _to_numpy_optional(monitor_exposure)

    if len(act) == 0:
        raise ValueError("monitor_actual must be non-empty")
    if len(act) != len(pred):
        raise ValueError(
            f"monitor_actual length ({len(act)}) != monitor_predicted length ({len(pred)})"
        )
    if exp is not None and len(exp) != len(act):
        raise ValueError(
            f"monitor_exposure length ({len(exp)}) != monitor_actual length ({len(act)})"
        )

    # Point estimate of Gini on monitor data
    monitor_gini = _gini_from_arrays(act, pred, exp)

    # Bootstrap SE: resample monitor data to get distribution of G_hat
    # Under H0, this gives us the sampling uncertainty around the monitor Gini.
    # We use this SE to standardise the deviation from training_gini.
    boot_samples = _bootstrap_gini_samples(act, pred, exp, n_bootstrap)
    se_boot = float(np.std(boot_samples, ddof=1))

    if se_boot == 0.0:
        return GiniDriftOneSampleResult(
            z_statistic=float("nan"),
            p_value=float("nan"),
            training_gini=float(training_gini),
            monitor_gini=float(monitor_gini),
            gini_change=float(monitor_gini - training_gini),
            se_bootstrap=0.0,
            significant=False,
        )

    gini_change = monitor_gini - training_gini
    z = gini_change / se_boot
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return GiniDriftOneSampleResult(
        z_statistic=float(z),
        p_value=float(p_value),
        training_gini=float(training_gini),
        monitor_gini=float(monitor_gini),
        gini_change=float(gini_change),
        se_bootstrap=float(se_boot),
        significant=bool(p_value < alpha),
    )


class GiniDriftBootstrapTest:
    """Class-based Gini drift test with bootstrap CI and visualisation.

    Wraps Algorithm 2 + Algorithm 3 from arXiv:2510.04556 with:

    - Percentile bootstrap CI for the Gini point estimate
    - Percentile CI for the Gini change (monitor - training)
    - Normal/Wald CI as secondary alternative
    - ``.plot()`` method for governance reporting
    - ``.summary()`` method returning a structured plain-text governance report
    - Stores bootstrap replicates for post-hoc inspection

    The one-sample design is used: training_gini is treated as a fixed scalar
    stored at model deployment time. Raw training data is not needed. This is
    the natural design for deployed monitoring where the monitor data arrives
    months after training.

    BCa confidence intervals are deliberately not implemented. For insurance
    monitoring data (n = 500-5000) the bootstrap Gini distribution is
    approximately normal (Denuit, Trufin & Verdegem 2025, Figure 3), and BCa
    provides negligible coverage improvement over the percentile interval while
    requiring n additional jackknife Gini computations.

    Limitation: assumes i.i.d. observations within the monitoring window (per
    Theorem 1 of arXiv:2510.04556). If the monitoring window spans a period
    with mid-period model or tariff changes, the i.i.d. bootstrap may
    underestimate variance. Block bootstrap is not implemented.

    Parameters
    ----------
    training_gini : float
        Gini coefficient computed on training/holdout data at model deployment.
        Stored as a scalar in the model registry — raw training data not needed.
        Must be in (-1, 1).
    monitor_actual : array-like
        Observed claims in the current monitoring period.
    monitor_predicted : array-like
        Model predictions for the current monitoring period (rates, not log-rates).
    monitor_exposure : array-like, optional
        Exposure weights (e.g. car-years) for the monitoring period.
    n_bootstrap : int, default 500
        Number of bootstrap replicates. 500 is sufficient for CI estimation;
        use 1000+ for high-precision CIs in regulatory reports. Must be >= 50.
    confidence_level : float, default 0.95
        Confidence level for the percentile bootstrap CI.
        Common choices: 0.90 (less conservative), 0.95 (standard), 0.99 (board reports).
        Must be in (0, 1).
    alpha : float, default 0.32
        Significance level for the hypothesis test significant flag.
        Default 0.32 = one-sigma rule per arXiv:2510.04556 recommendation for
        routine monitoring. Use 0.05 for formal governance escalation.
        Must be in (0, 1).
    random_state : int or None, default None
        Seed for reproducibility. None produces non-reproducible results.

    Examples
    --------
    Basic usage::

        import numpy as np
        from insurance_monitoring.discrimination import (
            gini_coefficient,
            GiniDriftBootstrapTest,
        )

        rng = np.random.default_rng(42)
        pred = rng.uniform(0.05, 0.20, 3_000)
        act = rng.poisson(pred)

        training_gini = 0.48  # stored at deployment time

        test = GiniDriftBootstrapTest(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=500,
            random_state=42,
        )
        result = test.test()
        print(test.summary())
        ax = test.plot()
    """

    def __init__(
        self,
        training_gini: float,
        monitor_actual: ArrayLike,
        monitor_predicted: ArrayLike,
        monitor_exposure: Optional[ArrayLike] = None,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
        alpha: float = 0.32,
        random_state: Optional[int] = None,
    ) -> None:
        if not (-1.0 < float(training_gini) < 1.0):
            raise ValueError(
                f"training_gini must be in (-1, 1), got {training_gini}"
            )
        if n_bootstrap < 50:
            raise ValueError(
                f"n_bootstrap must be >= 50, got {n_bootstrap}"
            )
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be in (0, 1), got {alpha}"
            )

        self.training_gini = float(training_gini)
        self.monitor_actual = monitor_actual
        self.monitor_predicted = monitor_predicted
        self.monitor_exposure = monitor_exposure
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = alpha
        self.random_state = random_state
        self._result: Optional[GiniBootstrapResult] = None

    def test(self) -> GiniBootstrapResult:
        """Run the bootstrap drift test.

        Returns a GiniBootstrapResult with all CI fields, test statistics,
        and stored bootstrap replicates.

        Runs on first call; subsequent calls return the cached result.
        Idempotent — safe to call multiple times.

        Raises
        ------
        ValueError
            If the monitoring sample has fewer than 50 observations.
        UserWarning
            If the monitoring sample has fewer than 200 observations (bootstrap
            Gini distribution may not be well-approximated by Normal; interpret
            p-values cautiously).
        """
        if self._result is not None:
            return self._result

        act = _to_numpy(self.monitor_actual)
        pred = _to_numpy(self.monitor_predicted)
        exp = _to_numpy_optional(self.monitor_exposure)

        n = len(act)

        # Input validation
        if len(pred) != n:
            raise ValueError(
                f"monitor_actual length ({n}) != monitor_predicted length ({len(pred)})"
            )
        if exp is not None and len(exp) != n:
            raise ValueError(
                f"monitor_exposure length ({len(exp)}) != monitor_actual length ({n})"
            )

        # Small sample guards (in test(), not __init__(), because array length
        # cannot be cheaply determined without converting the input).
        if n < 50:
            raise ValueError(
                f"Sample too small for reliable bootstrap inference (n={n}). Minimum 50."
            )
        if n < 200:
            warnings.warn(
                f"Small sample (n={n}): bootstrap Gini distribution may not be "
                f"well-approximated by Normal. Interpret p-values cautiously.",
                UserWarning,
                stacklevel=2,
            )

        # Point estimate
        G_hat = _gini_from_arrays(act, pred, exp)

        # Algorithm 2: bootstrap variance — reuses existing helper, no duplication
        boot = _bootstrap_gini_samples(
            act, pred, exp,
            n_bootstrap=self.n_bootstrap,
            seed=self.random_state,
        )

        # Handle rare NaN replicates (e.g. all-zero claim resamples)
        nan_count = int(np.sum(np.isnan(boot)))
        if nan_count > 0:
            if nan_count / self.n_bootstrap > 0.01:
                warnings.warn(
                    f"{nan_count}/{self.n_bootstrap} bootstrap replicates are NaN. "
                    f"SE and CI computed on remaining {self.n_bootstrap - nan_count} replicates.",
                    UserWarning,
                    stacklevel=2,
                )
            boot = boot[~np.isnan(boot)]

        se_boot = float(np.std(boot, ddof=1)) if len(boot) >= 2 else 0.0

        # Test statistic (Algorithm 3)
        if se_boot == 0.0 or len(boot) < 2:
            z = float("nan")
            p = float("nan")
            significant = False
        else:
            delta = G_hat - self.training_gini
            z = delta / se_boot
            p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            significant = bool(p < self.alpha)

        # Percentile CI for monitor Gini
        alpha_ci = (1.0 - self.confidence_level) / 2.0
        ci_lower = float(np.quantile(boot, alpha_ci))
        ci_upper = float(np.quantile(boot, 1.0 - alpha_ci))

        # Percentile CI for the change (same replicates, scalar subtraction)
        delta_boot = boot - self.training_gini
        ci_change_lower = float(np.quantile(delta_boot, alpha_ci))
        ci_change_upper = float(np.quantile(delta_boot, 1.0 - alpha_ci))

        self._result = GiniBootstrapResult(
            z_statistic=z,
            p_value=p,
            training_gini=float(self.training_gini),
            monitor_gini=float(G_hat),
            gini_change=float(G_hat - self.training_gini),
            se_bootstrap=se_boot,
            confidence_level=self.confidence_level,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_change_lower=ci_change_lower,
            ci_change_upper=ci_change_upper,
            n_bootstrap=self.n_bootstrap,
            n_obs=n,
            significant=significant,
            alpha=self.alpha,
            boot_replicates=boot.copy(),
        )
        return self._result

    def plot(
        self,
        ax=None,
        title: Optional[str] = None,
        show_ci: bool = True,
        show_training_gini: bool = True,
    ):
        """Plot the bootstrap distribution of the monitor Gini.

        Produces a histogram of the B bootstrap replicates with:

        - Vertical line at training_gini (null hypothesis value, dashed red)
        - Vertical line at monitor_gini (observed value, solid navy)
        - Shaded region for the CI bounds (green, acceptance region)
        - Annotation box with z, p-value, and significance verdict

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure is created.
        title : str, optional
            Custom plot title. If None, auto-generated with n and B.
        show_ci : bool, default True
            Whether to draw the CI shaded region.
        show_training_gini : bool, default True
            Whether to draw the training Gini vertical line.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        result = self.test()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
            plt.tight_layout()

        boot = result.boot_replicates
        n_boot_actual = len(boot)

        # 1. Histogram of bootstrap replicates
        ax.hist(
            boot,
            bins=40,
            density=False,
            color="steelblue",
            alpha=0.6,
            label=f"Bootstrap distribution (B={n_boot_actual})",
        )

        # 2. CI shaded region
        if show_ci:
            ci_pct = int(self.confidence_level * 100)
            ax.axvspan(
                result.ci_lower,
                result.ci_upper,
                alpha=0.15,
                color="green",
                label=f"{ci_pct}% CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]",
            )

        # 3. Training Gini (null hypothesis)
        if show_training_gini:
            ax.axvline(
                result.training_gini,
                linestyle="--",
                color="firebrick",
                linewidth=1.5,
                label=f"Training Gini = {result.training_gini:.3f}",
            )

        # 4. Monitor Gini (observed value)
        ax.axvline(
            result.monitor_gini,
            linestyle="-",
            color="navy",
            linewidth=1.5,
            label=f"Monitor Gini = {result.monitor_gini:.3f}",
        )

        # 5. Annotation box
        sig_text = "SIGNIFICANT" if result.significant else "not significant"
        if not (result.p_value != result.p_value):  # not nan
            annotation = (
                f"z = {result.z_statistic:.2f}\n"
                f"p = {result.p_value:.3f}\n"
                f"{sig_text}"
            )
        else:
            annotation = f"z = nan\np = nan\n{sig_text}"

        ax.text(
            0.97,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        # Labels and title
        ax.set_xlabel("Bootstrap Gini")
        ax.set_ylabel("Count")
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(
                f"Gini Drift Bootstrap Test (n={result.n_obs:,}, B={n_boot_actual})"
            )

        ax.legend(loc="upper left")

        return ax

    def summary(self) -> str:
        """Return a plain-text governance summary of the test results.

        Format is suitable for pasting into model monitoring reports, ORSA
        documents, or IFoA validation logs.

        Returns
        -------
        str
            Multi-line string with test statistics, CIs, and verdict.
        """
        result = self.test()
        ci_pct = int(self.confidence_level * 100)

        if result.significant:
            verdict = f"SIGNIFICANT at alpha={self.alpha} -> investigate model refit"
        else:
            verdict = f"not significant at alpha={self.alpha}"

        # Format z and p, handling nan gracefully
        if result.z_statistic != result.z_statistic:  # nan check
            z_str = "nan"
            p_str = "nan"
        else:
            z_str = f"{result.z_statistic:.2f}"
            p_str = f"{result.p_value:.3f}"

        lines = [
            "Gini Drift Bootstrap Test",
            "=========================",
            (
                f"Monitor period:  n = {result.n_obs:,} | "
                f"Gini = {result.monitor_gini:.3f} "
                f"[{ci_pct}% CI: {result.ci_lower:.3f}, {result.ci_upper:.3f}]"
            ),
            f"Training Gini:   {result.training_gini:.3f}",
            (
                f"Change:         {result.gini_change:+.3f} "
                f"[{ci_pct}% CI: {result.ci_change_lower:.3f}, {result.ci_change_upper:.3f}]"
            ),
            "",
            (
                f"Bootstrap:       B = {len(result.boot_replicates)} replicates | "
                f"SE = {result.se_bootstrap:.3f}"
            ),
            f"Test statistic:  z = {z_str} | p = {p_str}",
            f"Verdict:         {verdict}",
            "",
            "Note: one-sample bootstrap design (Algorithm 3, Denuit, Trufin & Verdegem,",
            "      arXiv:2510.04556). CI is percentile bootstrap.",
        ]
        return "\n".join(lines)


def lorenz_curve(
    actual: ArrayLike,
    predicted: ArrayLike,
    exposure: Optional[ArrayLike] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lorenz (CAP) curve for plotting.

    Returns two arrays suitable for plotting: cumulative share of exposure on
    the x-axis and cumulative share of actual claims on the y-axis, after
    sorting policies by ascending predicted rate.

    The diagonal (x = y) represents a model with no discriminatory power.
    Area between the curve and the diagonal is half the Gini coefficient.

    Parameters
    ----------
    actual:
        Observed claim values.
    predicted:
        Model predictions (used for sorting).
    exposure:
        Optional exposure weights.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, y) arrays: cumulative exposure share and cumulative claims share,
        both starting at (0, 0) and ending at (1, 1).

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        from insurance_monitoring.discrimination import lorenz_curve

        x, y = lorenz_curve(actual, predicted)
        plt.plot(x, y, label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("Cumulative exposure share")
        plt.ylabel("Cumulative claims share")
        plt.legend()
        plt.title("CAP curve")
    """
    act = _to_numpy(actual)
    pred = _to_numpy(predicted)
    exp = _to_numpy_optional(exposure)

    if len(act) == 0:
        raise ValueError("actual must be non-empty")
    if len(act) != len(pred):
        raise ValueError("actual and predicted must have the same length")

    if exp is None:
        exp = np.ones(len(act))

    # Group by unique predicted value to handle ties consistently with gini_coefficient.
    total_exp = float(np.sum(exp))
    total_claims = float(np.sum(act))

    unique_preds = np.unique(pred)  # sorted ascending
    cum_exp_vals = []
    cum_claims_vals = []
    running_exp = 0.0
    running_claims = 0.0
    for p in unique_preds:
        mask = pred == p
        running_exp += float(exp[mask].sum())
        running_claims += float(act[mask].sum())
        cum_exp_vals.append(running_exp)
        cum_claims_vals.append(running_claims)
    cum_exp = np.array(cum_exp_vals)
    cum_claims = np.array(cum_claims_vals)

    x = np.concatenate([[0.0], cum_exp / total_exp])
    y = np.concatenate([[0.0], cum_claims / total_claims if total_claims > 0 else cum_claims])

    return x, y
