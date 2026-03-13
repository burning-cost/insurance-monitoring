"""Auto-calibration test: binning test and bootstrap MCB test."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats

from ._deviance import deviance
from ._rectify import isotonic_recalibrate
from ._types import AutoCalibResult
from ._utils import validate_inputs


def _estimate_dispersion(
    y: np.ndarray,
    y_hat: np.ndarray,
    w: np.ndarray,
    distribution: str,
) -> float:
    """Estimate dispersion parameter (phi) for Gamma bootstrap.

    Uses the method-of-moments estimator: phi = mean squared Pearson residual.
    """
    pearson_residuals = (y - y_hat) / y_hat
    phi = float(np.sum(w * pearson_residuals ** 2) / np.sum(w))
    return max(phi, 1e-6)


def check_auto_calibration(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
    distribution: str = "poisson",
    n_bins: int = 10,
    method: str = "bootstrap",
    bootstrap_n: int = 999,
    significance_level: float = 0.05,
    seed: int | None = None,
    tweedie_power: float = 1.5,
) -> AutoCalibResult:
    """Test whether E[Y | mu_hat(X)] = mu_hat(X) holds within prediction cohorts.

    Auto-calibration is a stronger property than global balance. It requires
    that every price cohort (group of policies with similar predicted rates)
    is on average self-financing — no systematic cross-subsidisation between
    low-risk and high-risk cohorts.

    GLMs with canonical links satisfy auto-calibration on the training set by
    construction. GBMs and neural networks generally do not.

    Two test methods are available:

    **bootstrap** (default, recommended): Following Algorithm 1 from Brauer
    et al. (arXiv:2510.04556), simulates datasets under the null hypothesis
    (y ~ F(y_hat)) and computes the distribution of the MCB statistic. The
    p-value is the fraction of bootstrap MCB values exceeding the observed MCB.

    **hosmer_lemeshow**: Bins predictions into quantile groups, computes
    chi-squared statistic comparing observed vs expected counts per bin.
    Simpler but less powerful and sensitive to bin choice.

    Parameters
    ----------
    y
        Observed loss rates. Shape (n,).
    y_hat
        Model predictions. Shape (n,).
    exposure
        Policy durations. If None, assumed uniform.
    distribution
        Loss distribution: 'poisson', 'gamma', 'tweedie', 'normal'.
    n_bins
        Number of prediction quantile bins for binning test and per_bin output.
    method
        'bootstrap' or 'hosmer_lemeshow'.
    bootstrap_n
        Number of bootstrap replicates (only used when method='bootstrap').
    significance_level
        Significance level for the is_calibrated flag.

        .. note::
            Brauer et al. (2025) recommend alpha = 0.32 for ongoing monitoring
            (not 0.05). At alpha = 0.05, a one-standard-deviation deterioration
            in calibration has very low detection probability. The default here
            is 0.05 for initial model validation; use 0.32 for routine monitoring.

    seed
        Random seed for reproducibility.
    tweedie_power
        Variance power for Tweedie distribution (1 < p < 2).

    Returns
    -------
    AutoCalibResult
        All auto-calibration diagnostics.
    """
    y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)
    n = len(y_arr)

    # Compute per-bin diagnostics (used for both methods and for plots)
    per_bin_df = _compute_per_bin(y_arr, y_hat_arr, w, n_bins)

    # MCB score via Murphy decomposition (used in both tests)
    y_hat_rc = isotonic_recalibrate(y_arr, y_hat_arr, w)
    mcb_obs = deviance(y_arr, y_hat_arr, w, distribution, tweedie_power) - \
               deviance(y_arr, y_hat_rc, w, distribution, tweedie_power)
    mcb_obs = max(mcb_obs, 0.0)  # numerical noise can give tiny negatives

    # Count isotonic steps for complexity check
    sort_idx = np.argsort(y_hat_arr)
    fitted_sorted = y_hat_rc[sort_idx]
    diffs = np.diff(fitted_sorted)
    n_steps = int(np.sum(np.abs(diffs) > 1e-12)) + 1

    # Worst bin ratio
    import polars as pl
    ratios = per_bin_df["ratio"].to_numpy()
    worst_bin_ratio = float(np.max(np.abs(ratios - 1.0)))

    if method == "bootstrap":
        p_value = _bootstrap_mcb_test(
            y_arr, y_hat_arr, w, distribution, tweedie_power,
            mcb_obs, bootstrap_n, seed
        )
    elif method == "hosmer_lemeshow":
        p_value = _hosmer_lemeshow_test(per_bin_df, n_bins)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'bootstrap', 'hosmer_lemeshow'."
        )

    is_calibrated = p_value > significance_level

    return AutoCalibResult(
        p_value=p_value,
        is_calibrated=is_calibrated,
        per_bin=per_bin_df,
        mcb_score=mcb_obs,
        worst_bin_ratio=worst_bin_ratio,
        n_isotonic_steps=n_steps,
    )


def _compute_per_bin(
    y: np.ndarray,
    y_hat: np.ndarray,
    w: np.ndarray,
    n_bins: int,
) -> "pl.DataFrame":
    """Compute exposure-weighted per-bin diagnostics for the reliability diagram."""
    import polars as pl

    # Assign each observation to a quantile bin based on y_hat
    # Use exposure-weighted quantiles
    sort_idx = np.argsort(y_hat)
    cumulative_w = np.cumsum(w[sort_idx])
    total_w = cumulative_w[-1]
    bin_edges = np.linspace(0, total_w, n_bins + 1)

    bin_assignments = np.zeros(len(y), dtype=np.int32)
    for i, idx in enumerate(sort_idx):
        bin_id = int(np.searchsorted(bin_edges[1:], cumulative_w[i], side="left"))
        bin_assignments[idx] = min(bin_id, n_bins - 1)

    bins_list = []
    for b in range(n_bins):
        mask = bin_assignments == b
        if not np.any(mask):
            continue
        w_b = w[mask]
        y_b = y[mask]
        y_hat_b = y_hat[mask]
        total_w_b = float(np.sum(w_b))
        obs_mean = float(np.sum(w_b * y_b) / total_w_b)
        pred_mean = float(np.sum(w_b * y_hat_b) / total_w_b)
        ratio = obs_mean / pred_mean if pred_mean > 0 else float("nan")
        bins_list.append({
            "bin": b + 1,
            "pred_mean": pred_mean,
            "obs_mean": obs_mean,
            "ratio": ratio,
            "exposure": total_w_b,
            "n_policies": int(np.sum(mask)),
        })

    return pl.DataFrame(bins_list, schema={
        "bin": pl.Int32,
        "pred_mean": pl.Float64,
        "obs_mean": pl.Float64,
        "ratio": pl.Float64,
        "exposure": pl.Float64,
        "n_policies": pl.Int32,
    })


def _bootstrap_mcb_test(
    y: np.ndarray,
    y_hat: np.ndarray,
    w: np.ndarray,
    distribution: str,
    tweedie_power: float,
    mcb_obs: float,
    bootstrap_n: int,
    seed: int | None,
) -> float:
    """Bootstrap MCB test following Algorithm 1 of arXiv:2510.04556.

    Under H0 (model is correctly calibrated), simulate y_b from the assumed
    distribution with mean y_hat, then compute MCB of y_b vs y_hat. The
    p-value is the fraction of bootstrap MCB values >= observed MCB.
    """
    rng = np.random.default_rng(seed)
    mcb_boot = np.empty(bootstrap_n, dtype=np.float64)

    if distribution == "poisson":
        # y is a rate, so counts = Poisson(exposure * y_hat); rate = counts / exposure
        for i in range(bootstrap_n):
            counts_b = rng.poisson(w * y_hat)
            y_b = np.where(w > 0, counts_b / w, 0.0)
            y_b = np.maximum(y_b, 0.0)
            # Use a small floor to avoid log(0) issues
            y_hat_safe = np.maximum(y_hat, 1e-10)
            y_hat_rc_b = isotonic_recalibrate(y_b, y_hat_safe, w)
            d_full = deviance(y_b, y_hat_safe, w, distribution, tweedie_power)
            d_rc = deviance(y_b, y_hat_rc_b, w, distribution, tweedie_power)
            mcb_boot[i] = max(d_full - d_rc, 0.0)

    elif distribution == "gamma":
        phi = _estimate_dispersion(y, y_hat, w, distribution)
        shape = 1.0 / phi
        for i in range(bootstrap_n):
            # Gamma(shape, scale=y_hat * phi)
            y_b = rng.gamma(shape=shape, scale=y_hat * phi)
            y_b = np.maximum(y_b, 1e-10)
            y_hat_rc_b = isotonic_recalibrate(y_b, y_hat, w)
            d_full = deviance(y_b, y_hat, w, distribution, tweedie_power)
            d_rc = deviance(y_b, y_hat_rc_b, w, distribution, tweedie_power)
            mcb_boot[i] = max(d_full - d_rc, 0.0)

    elif distribution == "normal":
        # Estimate sigma from residuals
        residuals = y - y_hat
        sigma = float(np.sqrt(np.average(residuals ** 2, weights=w)))
        for i in range(bootstrap_n):
            y_b = rng.normal(loc=y_hat, scale=sigma)
            y_hat_rc_b = isotonic_recalibrate(y_b, y_hat, w)
            d_full = deviance(y_b, y_hat, w, distribution, tweedie_power)
            d_rc = deviance(y_b, y_hat_rc_b, w, distribution, tweedie_power)
            mcb_boot[i] = max(d_full - d_rc, 0.0)

    elif distribution == "tweedie":
        # Approximate as Poisson for bootstrap (compound Poisson-Gamma)
        # This is a pragmatic approximation; proper Tweedie simulation is complex.
        for i in range(bootstrap_n):
            counts_b = rng.poisson(w * y_hat)
            y_b = np.where(w > 0, counts_b / w, 0.0)
            y_b = np.maximum(y_b, 0.0)
            y_hat_safe = np.maximum(y_hat, 1e-10)
            y_hat_rc_b = isotonic_recalibrate(y_b, y_hat_safe, w)
            d_full = deviance(y_b, y_hat_safe, w, distribution, tweedie_power)
            d_rc = deviance(y_b, y_hat_rc_b, w, distribution, tweedie_power)
            mcb_boot[i] = max(d_full - d_rc, 0.0)

    else:
        raise ValueError(f"Unknown distribution '{distribution}'.")

    p_value = float(np.mean(mcb_boot >= mcb_obs))
    return p_value


def _hosmer_lemeshow_test(
    per_bin_df: "pl.DataFrame",
    n_bins: int,
) -> float:
    """Hosmer-Lemeshow style chi-squared test on per-bin obs vs expected.

    Uses chi-squared with df = n_bins - 2. This is a simple alternative to
    the bootstrap test — faster but less powerful and sensitive to bin choice.
    """
    import polars as pl

    obs = per_bin_df["obs_mean"].to_numpy()
    exp = per_bin_df["pred_mean"].to_numpy()
    exposure = per_bin_df["exposure"].to_numpy()

    # Convert rates to counts for chi-squared
    obs_counts = obs * exposure
    exp_counts = exp * exposure

    # Avoid division by zero for empty bins
    valid = exp_counts > 0
    chi2 = float(np.sum(((obs_counts[valid] - exp_counts[valid]) ** 2) / exp_counts[valid]))

    df = max(1, n_bins - 2)
    p_value = float(1.0 - scipy.stats.chi2.cdf(chi2, df=df))
    return p_value
