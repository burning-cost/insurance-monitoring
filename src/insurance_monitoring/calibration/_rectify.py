"""Rectification methods: multiplicative balance correction, affine GLM, isotonic."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.optimize

from ._utils import validate_inputs


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm (PAVA) for weighted isotonic regression.

    Solves the weighted least-squares isotonic regression problem::

        argmin_m sum_i w_i * (y_i - m_i)^2  subject to  m_1 <= m_2 <= ... <= m_n

    This is a pure numpy implementation requiring only numpy. No scipy >= 1.12
    or scikit-learn required. Runtime is O(n) amortised.

    Parameters
    ----------
    y
        Target values, already sorted by the predictor. Shape (n,).
    w
        Non-negative weights. Shape (n,).

    Returns
    -------
    np.ndarray
        Isotonically non-decreasing fitted values. Same shape as y.
    """
    n = len(y)
    # Each block stores (sum_w, sum_wy) — weighted mean = sum_wy / sum_w
    sum_w = w.copy().astype(np.float64)
    sum_wy = (w * y).astype(np.float64)
    # Block boundaries: block_end[i] = end index (exclusive) of block i
    # We track blocks as a stack
    block_start = list(range(n))  # start index of each block
    block_sw = list(sum_w)        # sum of weights in block
    block_swy = list(sum_wy)      # sum of w*y in block

    i = 1
    while i < len(block_start):
        mean_prev = block_swy[i - 1] / block_sw[i - 1]
        mean_curr = block_swy[i] / block_sw[i]
        if mean_prev > mean_curr:
            # Merge block i into block i-1
            block_sw[i - 1] += block_sw[i]
            block_swy[i - 1] += block_swy[i]
            del block_start[i]
            del block_sw[i]
            del block_swy[i]
            # Check if merged block violates constraint with previous block
            if i > 1:
                i -= 1
        else:
            i += 1

    # Expand blocks back to per-observation values
    result = np.empty(n, dtype=np.float64)
    pos = 0
    for k in range(len(block_start)):
        start = block_start[k]
        end = block_start[k + 1] if k + 1 < len(block_start) else n
        mean_val = block_swy[k] / block_sw[k]
        result[start:end] = mean_val
    return result


def _get_isotonic():
    """Return the best available isotonic regression solver.

    Preference order:
    1. scipy.stats.isotonic_regression (scipy >= 1.12) — fastest
    2. sklearn.isotonic.IsotonicRegression — widely available
    3. _pava — pure numpy fallback, always works
    """
    try:
        from scipy.stats import isotonic_regression as _scipy_iso

        def _solve_scipy(y: np.ndarray, w: np.ndarray) -> np.ndarray:
            result = _scipy_iso(y, weights=w, increasing=True)
            return np.asarray(result.x, dtype=np.float64)

        return _solve_scipy
    except (ImportError, AttributeError):
        pass

    try:
        from sklearn.isotonic import IsotonicRegression

        def _solve_sk(y: np.ndarray, w: np.ndarray) -> np.ndarray:
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            x = np.arange(len(y), dtype=np.float64)
            return ir.fit_transform(x, y, sample_weight=w)

        return _solve_sk
    except (ImportError, AttributeError):
        pass

    # Pure numpy PAVA — always available
    return _pava


def isotonic_recalibrate(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Apply isotonic recalibration to restore auto-calibration empirically.

    Solves the weighted isotonic regression problem::

        argmin_{m: order-preserving} sum_i w_i * (y_i - m_i)^2
        subject to: m_k <= m_j whenever y_hat_k <= y_hat_j

    This preserves the ranking implied by ``y_hat`` whilst correcting the
    price levels at each quantile of the prediction distribution.

    A warning is raised if the number of isotonic steps exceeds sqrt(n),
    following Wüthrich & Ziegel (SAJ 2024). This suggests the recalibration
    is fitting noise rather than signal — the holdout sample may be too small,
    or the model may lack genuine discriminatory power.

    .. warning::
        Isotonic recalibration trivially achieves in-sample auto-calibration.
        Always apply this to **holdout** data. In-sample use is meaningless.

    Parameters
    ----------
    y
        Observed loss rates. Shape (n,).
    y_hat
        Model predictions (determines the ordering constraint). Shape (n,).
    exposure
        Policy durations. Used as weights in the isotonic regression. If None,
        uniform weights are used.

    Returns
    -------
    np.ndarray
        Isotonically recalibrated predictions. Same shape as y_hat, preserving
        the original ordering of observations (not sorted).
    """
    from ._utils import check_isotonic_complexity

    y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)
    n = len(y_arr)
    _iso = _get_isotonic()

    # Sort by prediction to apply isotonic regression in the right order
    sort_idx = np.argsort(y_hat_arr, kind="stable")
    y_sorted = y_arr[sort_idx]
    w_sorted = w[sort_idx]

    # Apply isotonic regression to y values in prediction order
    fitted_sorted = _iso(y_sorted, w_sorted)

    # Count distinct levels (isotonic steps)
    diffs = np.diff(fitted_sorted)
    n_steps = int(np.sum(np.abs(diffs) > 1e-12)) + 1
    check_isotonic_complexity(n_steps, n)

    # Map back to original observation order
    result = np.empty(n, dtype=np.float64)
    result[sort_idx] = fitted_sorted
    return result


def rectify_balance(
    y_hat: npt.ArrayLike,
    y: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
    method: str = "multiplicative",
    distribution: str = "poisson",
) -> np.ndarray:
    """Correct predictions to restore the global balance property.

    Two methods are available:

    **multiplicative** — scales all predictions by a single constant::

        y_hat_corrected = alpha * y_hat
        where alpha = sum(v * y) / sum(v * y_hat)

    This is the simplest correction. It does not change the model's ranking
    or shape — it merely shifts the overall level.

    **affine** — fits a one-variable GLM: log(mu) = beta_0 + beta_1 * log(y_hat),
    minimising the weighted Poisson deviance. This corrects for both a global
    level shift (beta_0 != 0) and a scale error (beta_1 != 1). For Poisson
    this corresponds to the affine recalibration from Lindholm & Wüthrich (2025).

    Parameters
    ----------
    y_hat
        Original model predictions (rates). Shape (n,).
    y
        Observed loss rates. Shape (n,).
    exposure
        Policy durations. If None, assumed uniform = 1.0.
    method
        'multiplicative' or 'affine'.
    distribution
        Loss distribution for affine rectification deviance loss.
        Only 'poisson' is fully supported for affine; multiplicative is
        distribution-agnostic.

    Returns
    -------
    np.ndarray
        Corrected predictions. Same shape as y_hat.

    Raises
    ------
    ValueError
        If method is not recognised.
    """
    y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)

    if method == "multiplicative":
        alpha = np.sum(w * y_arr) / np.sum(w * y_hat_arr)
        return alpha * y_hat_arr

    if method == "affine":
        return _affine_rectify(y_arr, y_hat_arr, w, distribution)

    raise ValueError(
        f"Unknown rectification method '{method}'. "
        "Supported: 'multiplicative', 'affine'."
    )


def _affine_rectify(
    y: np.ndarray,
    y_hat: np.ndarray,
    w: np.ndarray,
    distribution: str,
) -> np.ndarray:
    """Fit affine recalibration: log(mu) = b0 + b1 * log(y_hat).

    Minimises the weighted Poisson deviance (appropriate for frequency and
    pure premium models). The link function is log, matching the canonical
    link for Poisson and Gamma GLMs.

    The affine parameterisation is more general than multiplicative:
    - b1 = 1, b0 = log(alpha) reduces to multiplicative correction
    - b1 != 1 corrects for slope errors (model over/under-responsive to risk factors)
    """
    from ._deviance import deviance

    log_y_hat = np.log(y_hat)

    def _loss(params: np.ndarray) -> float:
        b0, b1 = params
        mu = np.exp(b0 + b1 * log_y_hat)
        # Clip to avoid numerical issues
        mu = np.clip(mu, 1e-10, None)
        return deviance(y, mu, w, distribution)

    # Initialise at multiplicative correction (b1=1, b0=log(alpha))
    alpha = np.sum(w * y) / np.sum(w * y_hat)
    x0 = np.array([np.log(alpha), 1.0])

    result = scipy.optimize.minimize(
        _loss,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10_000},
    )
    b0, b1 = result.x
    return np.exp(b0 + b1 * log_y_hat)
