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

References
----------
- arXiv 2510.04556, Section 3: Gini score and monitoring framework
- arXiv 2510.04556, Algorithm 2: Non-parametric bootstrap variance estimator
- arXiv 2510.04556, Algorithm 3: One-sample monitoring test
- Frees & Valdez (1998): actuarial use of Lorenz curves and Gini
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats

# numpy<2.0 compat: trapezoid was added in 2.0, trapz deprecated/removed in 2.0
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]


ArrayLike = Union[np.ndarray, pl.Series]


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
    """
    n = len(actual)
    rng = np.random.default_rng(seed)
    gini_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gini_samples[i] = _gini_from_arrays(
            actual[idx],
            predicted[idx],
            exposure[idx] if exposure is not None else None,
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
    n_reference: int,
    n_current: int,
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
) -> dict[str, float]:
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
        Number of observations in reference period (used for variance scaling).
    n_current:
        Number of observations in current period.
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
    dict with keys:
        - ``z_statistic``: z-score (negative = current Gini degraded)
        - ``p_value``: two-sided p-value
        - ``reference_gini``: as supplied
        - ``current_gini``: as supplied
        - ``gini_change``: current_gini - reference_gini
        - ``significant``: bool, True if p < alpha

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
        return {
            "z_statistic": float("nan"),
            "p_value": float("nan"),
            "reference_gini": reference_gini,
            "current_gini": current_gini,
            "gini_change": current_gini - reference_gini,
            "significant": False,
        }

    gini_change = current_gini - reference_gini
    z = gini_change / se
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "z_statistic": float(z),
        "p_value": p_value,
        "reference_gini": float(reference_gini),
        "current_gini": float(current_gini),
        "gini_change": float(gini_change),
        "significant": bool(p_value < alpha),
    }


def gini_drift_test_onesample(
    training_gini: float,
    monitor_actual: ArrayLike,
    monitor_predicted: ArrayLike,
    monitor_exposure: Optional[ArrayLike] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.32,
) -> dict[str, float]:
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
    dict with keys:
        - ``z_statistic``: z-score (negative = monitor Gini below training Gini)
        - ``p_value``: two-sided p-value
        - ``training_gini``: as supplied
        - ``monitor_gini``: point estimate from monitor data
        - ``gini_change``: monitor_gini - training_gini
        - ``se_bootstrap``: bootstrap standard error of the monitor Gini estimator
        - ``significant``: bool, True if p < alpha

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
        return {
            "z_statistic": float("nan"),
            "p_value": float("nan"),
            "training_gini": float(training_gini),
            "monitor_gini": float(monitor_gini),
            "gini_change": float(monitor_gini - training_gini),
            "se_bootstrap": 0.0,
            "significant": False,
        }

    gini_change = monitor_gini - training_gini
    z = gini_change / se_boot
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "z_statistic": float(z),
        "p_value": p_value,
        "training_gini": float(training_gini),
        "monitor_gini": float(monitor_gini),
        "gini_change": float(gini_change),
        "se_bootstrap": float(se_boot),
        "significant": bool(p_value < alpha),
    }


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
