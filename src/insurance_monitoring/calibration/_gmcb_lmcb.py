"""Separate bootstrap p-values for GMCB (global) and LMCB (local) miscalibration.

Implements Algorithm 4a (GMCB) and Algorithm 4b (LMCB) from Brauer, Menzel &
Wüthrich (2025), arXiv:2510.04556.

**Time-splitting pitfall**: always pass data aggregated to policyholder level.
ETL pipelines that split multi-claim policies into one row per claim event will
inflate deviance by a factor of 10x and deflate Gini. This module emits a
UserWarning when np.median(exposure) < 0.05, which is the most reliable
indicator of row-level splitting.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from ._deviance import deviance
from ._rectify import isotonic_recalibrate
from ._types import GMCBResult, LMCBResult
from ._utils import validate_inputs


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def check_gmcb(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
    distribution: str = "poisson",
    bootstrap_n: int = 999,
    significance_level: float = 0.32,
    seed: int | None = None,
    tweedie_power: float = 1.5,
) -> GMCBResult:
    """Bootstrap test for global miscalibration (Algorithm 4a, arXiv:2510.04556).

    Tests H_0: GMCB = 0. Rejects if the global balance cannot be achieved by
    applying a single multiplicative correction (balance factor) to all predictions.
    This detects portfolio-wide inflation/deflation such as claims trend or
    medical cost inflation that shifts every cohort by the same proportion.

    The test statistic is::

        GMCB = D(y, y_hat) - D(y, alpha * y_hat)

    where alpha = sum(w * y) / sum(w * y_hat) is the balance factor. The
    bootstrap simulates datasets under H_0 (model is correctly globally
    calibrated) and computes the null distribution of GMCB. A low p-value
    means the observed GMCB exceeds what we would expect under a globally
    balanced model.

    **Alpha recommendation**: use significance_level=0.32 (one-sigma rule)
    for ongoing monitoring. At alpha=0.05, Type II error is high for realistic
    drift magnitudes. See Figure 8 of arXiv:2510.04556 — the paper demonstrates
    that alpha=0.05 misses mild-to-moderate global shifts that alpha=0.32 catches.

    Parameters
    ----------
    y
        Observed loss rates (claims per year, losses per unit exposure). Shape (n,).
        Must be aggregated to policyholder level — not row-per-claim.
    y_hat
        Model predictions (rates). Shape (n,). Must be strictly positive.
    exposure
        Policy durations in years. If None, assumed uniform = 1.0.
    distribution
        EDF family: 'poisson', 'gamma', 'tweedie', 'normal'.
    bootstrap_n
        Number of bootstrap replicates. Default 999. Minimum 99.
    significance_level
        Significance level alpha for the is_significant flag. Default 0.32.
        See note above.
    seed
        Random seed for reproducibility. None gives non-reproducible results.
    tweedie_power
        Tweedie variance power. Only used when distribution='tweedie'.

    Returns
    -------
    GMCBResult
        Bootstrap p-value, significance flag, balance factor, and GMCB score.

    Raises
    ------
    ValueError
        If sum(w * y_hat) is near zero or the balance factor is extreme.

    Warns
    -----
    UserWarning
        If np.median(exposure) < 0.05 (time-splitting pitfall detected).
    UserWarning
        If balance_factor < 0.01 or > 100 (extreme balance suggests data issue).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> exposure = rng.uniform(0.5, 2.0, 2000)
    >>> y_hat = rng.gamma(2, 0.05, 2000)
    >>> y = rng.poisson(exposure * y_hat) / exposure  # balanced
    >>> result = check_gmcb(y, y_hat, exposure, seed=0)
    >>> result.p_value > 0.32  # should not reject under no drift
    True
    """
    y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)
    _warn_time_split(w)

    # Balance factor and observed GMCB
    pred_total = float(np.sum(w * y_hat_arr))
    if pred_total < 1e-12:
        raise ValueError(
            "sum(w * y_hat) is near zero. Cannot compute balance factor."
        )
    obs_total = float(np.sum(w * y_arr))
    balance_factor = obs_total / pred_total

    if balance_factor < 0.01 or balance_factor > 100.0:
        warnings.warn(
            f"Extreme balance factor ({balance_factor:.4f}) suggests a data issue. "
            "Check that y and y_hat are on the same scale (rates, not counts).",
            UserWarning,
            stacklevel=2,
        )

    y_hat_bc = balance_factor * y_hat_arr
    y_hat_bc = np.maximum(y_hat_bc, 1e-10)

    gmcb_obs = deviance(y_arr, y_hat_arr, w, distribution, tweedie_power) - \
               deviance(y_arr, y_hat_bc, w, distribution, tweedie_power)
    gmcb_obs = max(gmcb_obs, 0.0)  # floor at zero; GMCB is guaranteed >= 0

    # Bootstrap variance estimation: isotonic regression of squared residuals
    var_hat = _estimate_variance(y_arr, y_hat_arr, w)

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_gmcb = _bootstrap_gmcb(
        y_hat_arr=y_hat_arr,
        var_hat=var_hat,
        w=w,
        distribution=distribution,
        tweedie_power=tweedie_power,
        bootstrap_n=bootstrap_n,
        rng=rng,
    )

    p_value = float(np.mean(boot_gmcb >= gmcb_obs))

    return GMCBResult(
        gmcb_score=gmcb_obs,
        p_value=p_value,
        is_significant=bool(p_value < significance_level),
        significance_level=significance_level,
        n_bootstrap=bootstrap_n,
        balance_factor=balance_factor,
    )


def check_lmcb(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
    distribution: str = "poisson",
    bootstrap_n: int = 999,
    significance_level: float = 0.32,
    seed: int | None = None,
    tweedie_power: float = 1.5,
) -> LMCBResult:
    """Bootstrap test for local miscalibration (Algorithm 4b, arXiv:2510.04556).

    Tests H_0: LMCB = 0. Rejects if there are cohort-level level shifts beyond
    what a global balance correction can fix. This is the structural test —
    a significant LMCB indicates that some risk cohorts are systematically
    over-priced and others under-priced in a pattern the model fails to capture.

    The test statistic is::

        LMCB = D(y, y_hat_bc) - D(y, y_hat_bc_rc)

    where y_hat_bc is the balance-corrected prediction (alpha * y_hat) and
    y_hat_bc_rc is the isotonically recalibrated version of y_hat_bc.

    **Note on sign**: LMCB can be numerically negative when the model is
    poorly ranked (isotonic recalibration hurts), which happens when rank
    correlation between y and y_hat is negative. This is not suppressed —
    it is a meaningful signal.

    Parameters
    ----------
    y
        Observed loss rates. Shape (n,).
    y_hat
        Model predictions (rates). Shape (n,).
    exposure
        Policy durations. If None, assumed uniform = 1.0.
    distribution
        EDF family: 'poisson', 'gamma', 'tweedie', 'normal'.
    bootstrap_n
        Number of bootstrap replicates. Default 999.
    significance_level
        Significance level alpha for the is_significant flag. Default 0.32.
    seed
        Random seed for reproducibility.
    tweedie_power
        Tweedie variance power. Only used when distribution='tweedie'.

    Returns
    -------
    LMCBResult
        Bootstrap p-value, significance flag, and LMCB score.

    Warns
    -----
    UserWarning
        If np.median(exposure) < 0.05 (time-splitting pitfall).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> exposure = rng.uniform(0.5, 2.0, 2000)
    >>> y_hat = rng.gamma(2, 0.05, 2000)
    >>> y = rng.poisson(exposure * y_hat) / exposure  # well-ranked
    >>> result = check_lmcb(y, y_hat, exposure, seed=0)
    >>> result.p_value > 0.32  # should not reject under no drift
    True
    """
    y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)
    _warn_time_split(w)

    # Balance-corrected predictions
    pred_total = float(np.sum(w * y_hat_arr))
    if pred_total < 1e-12:
        raise ValueError(
            "sum(w * y_hat) is near zero. Cannot compute balance factor."
        )
    balance_factor = float(np.sum(w * y_arr)) / pred_total
    y_hat_bc = np.maximum(balance_factor * y_hat_arr, 1e-10)

    # Isotonic recalibration of balance-corrected predictions
    y_hat_bc_rc = isotonic_recalibrate(y_arr, y_hat_bc, w)
    y_hat_bc_rc = np.maximum(y_hat_bc_rc, 1e-10)

    d_bc = deviance(y_arr, y_hat_bc, w, distribution, tweedie_power)
    d_bc_rc = deviance(y_arr, y_hat_bc_rc, w, distribution, tweedie_power)
    lmcb_obs = d_bc - d_bc_rc
    # Do NOT floor LMCB at zero — negative LMCB signals poor ranking ability

    # Bootstrap variance estimation using balance-corrected residuals
    var_hat_bc = _estimate_variance(y_arr, y_hat_bc, w)

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_lmcb = _bootstrap_lmcb(
        y_hat_bc=y_hat_bc,
        var_hat_bc=var_hat_bc,
        w=w,
        distribution=distribution,
        tweedie_power=tweedie_power,
        bootstrap_n=bootstrap_n,
        rng=rng,
    )

    p_value = float(np.mean(boot_lmcb >= lmcb_obs))

    return LMCBResult(
        lmcb_score=lmcb_obs,
        p_value=p_value,
        is_significant=bool(p_value < significance_level),
        significance_level=significance_level,
        n_bootstrap=bootstrap_n,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _warn_time_split(w: np.ndarray) -> None:
    """Warn if median exposure suggests row-level (per-claim) data splitting."""
    med = float(np.median(w))
    if med < 0.05:
        warnings.warn(
            f"Median exposure is {med:.4f}, which is below 0.05. This often "
            "indicates that the data has been split to one row per claim event "
            "(time-splitting). The Murphy decomposition and Gini require data "
            "aggregated to policyholder level. Row-level splitting inflates "
            "deviance ~10x and deflates Gini. "
            "See Section 3.3 of arXiv:2510.04556.",
            UserWarning,
            stacklevel=3,
        )


def _estimate_variance(
    y: np.ndarray,
    y_hat: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Estimate Var(Y_i | x_i) by isotonic regression of clipped squared residuals.

    Clips residuals before squaring to prevent extreme outliers (e.g. severity
    catastrophes) from destabilising the isotonic variance fit. The clip threshold
    is 10 * median(|r|), which leaves the bulk of the distribution untouched.

    Parameters
    ----------
    y, y_hat, w
        Observations, predictions, and exposure weights.

    Returns
    -------
    np.ndarray
        Estimated conditional variance at each observation. Shape (n,). Floored
        at a small positive value to prevent degenerate Gamma/Normal bootstrap.
    """
    r = y - y_hat
    abs_r = np.abs(r)
    med_abs_r = float(np.median(abs_r))
    clip_threshold = 10.0 * med_abs_r if med_abs_r > 1e-12 else 1.0
    r_clipped = np.clip(r, -clip_threshold, clip_threshold)
    sq_resid = r_clipped ** 2

    # Isotonic regression of squared residuals on y_hat.
    # Use the same solver hierarchy as _rectify.py (_get_isotonic).
    from ._rectify import _get_isotonic

    _iso = _get_isotonic()
    sort_idx = np.argsort(y_hat)
    sq_sorted = sq_resid[sort_idx]
    w_sorted = w[sort_idx]

    var_sorted = _iso(sq_sorted, w_sorted)

    var_hat = np.empty_like(sq_resid)
    var_hat[sort_idx] = var_sorted

    # Floor to prevent division-by-zero in Gamma/Normal sampling
    min_var = 1e-10
    var_hat = np.maximum(var_hat, min_var)
    return var_hat


def _sample_bootstrap_y(
    y_hat: np.ndarray,
    var_hat: np.ndarray,
    w: np.ndarray,
    distribution: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one bootstrap dataset Y* ~ F(y_hat, var_hat).

    For Poisson: counts ~ Poisson(w * y_hat), rates = counts / w.
    For Gamma: shape = y_hat^2 / var_hat, scale = var_hat / y_hat.
    For Normal: Y* ~ Normal(y_hat, sqrt(var_hat)).
    For Tweedie: approximate as Poisson (standard practice).
    """
    if distribution in ("poisson", "tweedie"):
        lam = y_hat * w
        counts_b = rng.poisson(lam)
        y_b = np.where(w > 0, counts_b / w, 0.0)
        return np.maximum(y_b, 0.0)

    elif distribution == "gamma":
        # Gamma(a, scale=b): mean = a*b = y_hat, var = a*b^2 = var_hat
        # => a = y_hat^2 / var_hat, b = var_hat / y_hat
        a = y_hat ** 2 / var_hat
        # Clip a to prevent instability when var >> mean
        a = np.clip(a, 1e-3, 1e6)
        scale = var_hat / y_hat
        y_b = rng.gamma(shape=a, scale=scale)
        return np.maximum(y_b, 1e-10)

    elif distribution == "normal":
        sigma = np.sqrt(var_hat)
        return rng.normal(loc=y_hat, scale=sigma)

    else:
        raise ValueError(f"Unknown distribution '{distribution}'.")


def _bootstrap_gmcb(
    y_hat_arr: np.ndarray,
    var_hat: np.ndarray,
    w: np.ndarray,
    distribution: str,
    tweedie_power: float,
    bootstrap_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Algorithm 4a bootstrap loop.

    For each replicate:
      1. Sample Y* ~ F(y_hat, var_hat).
      2. Compute balance factor alpha* = sum(w * Y*) / sum(w * y_hat).
      3. y_hat_bc* = alpha* * y_hat.
      4. gmcb* = D(Y*, y_hat) - D(Y*, y_hat_bc*).
    """
    y_hat_safe = np.maximum(y_hat_arr, 1e-10)
    sum_w_yhat = float(np.sum(w * y_hat_safe))
    boot = np.empty(bootstrap_n, dtype=np.float64)

    for b in range(bootstrap_n):
        y_b = _sample_bootstrap_y(y_hat_safe, var_hat, w, distribution, rng)
        alpha_b = float(np.sum(w * y_b)) / sum_w_yhat
        y_hat_bc_b = np.maximum(alpha_b * y_hat_safe, 1e-10)

        d_full = deviance(y_b, y_hat_safe, w, distribution, tweedie_power)
        d_bc = deviance(y_b, y_hat_bc_b, w, distribution, tweedie_power)
        boot[b] = max(d_full - d_bc, 0.0)  # floor: GMCB >= 0

    return boot


def _bootstrap_lmcb(
    y_hat_bc: np.ndarray,
    var_hat_bc: np.ndarray,
    w: np.ndarray,
    distribution: str,
    tweedie_power: float,
    bootstrap_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Algorithm 4b bootstrap loop.

    For each replicate:
      1. Sample Y* ~ F(y_hat_bc, var_hat_bc).
      2. Fit isotonic regression on (Y*, y_hat_bc, w) to get y_hat_bc_rc*.
      3. lmcb* = D(Y*, y_hat_bc) - D(Y*, y_hat_bc_rc*).
    """
    boot = np.empty(bootstrap_n, dtype=np.float64)
    y_hat_bc_safe = np.maximum(y_hat_bc, 1e-10)

    for b in range(bootstrap_n):
        y_b = _sample_bootstrap_y(y_hat_bc_safe, var_hat_bc, w, distribution, rng)
        y_hat_bc_rc_b = isotonic_recalibrate(y_b, y_hat_bc_safe, w)
        y_hat_bc_rc_b = np.maximum(y_hat_bc_rc_b, 1e-10)

        d_bc = deviance(y_b, y_hat_bc_safe, w, distribution, tweedie_power)
        d_bc_rc = deviance(y_b, y_hat_bc_rc_b, w, distribution, tweedie_power)
        boot[b] = d_bc - d_bc_rc  # do NOT floor: negative LMCB is informative

    return boot
