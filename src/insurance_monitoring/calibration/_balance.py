"""Global balance property test and bootstrap confidence interval."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats

from ._types import BalanceResult
from ._utils import validate_inputs


def check_balance(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
    distribution: str = "poisson",
    bootstrap_n: int = 999,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BalanceResult:
    """Test whether exposure-weighted predicted totals equal observed totals.

    The balance ratio is defined as::

        alpha = sum(v_i * y_i) / sum(v_i * mu_hat_i)

    A well-calibrated model has alpha = 1.0. Values above 1.0 indicate
    systematic under-prediction (the model charges less than claims cost).
    Values below 1.0 indicate systematic over-prediction.

    The parametric p-value uses a Poisson approximation: under H0, the
    observed count total is approximately Normal(expected, expected), giving
    a z-statistic. This is appropriate for frequency models. For Gamma or
    Normal models, interpret the bootstrap CI rather than the p-value.

    The bootstrap CI resamples observation indices with replacement, preserving
    the joint dependence structure of (y, y_hat, exposure).

    Parameters
    ----------
    y
        Observed loss rates (claims per year). Shape (n,).
    y_hat
        Model predictions (rates, not counts). Shape (n,).
    exposure
        Policy durations in years. If None, assumed uniform = 1.0.
    distribution
        Loss distribution. Used to label the parametric p-value.
        'poisson', 'gamma', 'tweedie', 'normal'.
    bootstrap_n
        Number of bootstrap replicates for the confidence interval on alpha.
    confidence_level
        Coverage for the bootstrap CI. Default 0.95.
    seed
        Random seed for reproducibility.

    Returns
    -------
    BalanceResult
        All balance diagnostics in a single dataclass.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> exposure = rng.uniform(0.5, 2.0, 1000)
    >>> y_hat = rng.gamma(2, 0.05, 1000)
    >>> y = rng.poisson(exposure * y_hat) / exposure
    >>> result = check_balance(y, y_hat, exposure, seed=42)
    >>> abs(result.balance_ratio - 1.0) < 0.05
    True
    """
    y, y_hat, w = validate_inputs(y, y_hat, exposure)
    n = len(y)

    # Point estimate
    obs_total = float(np.sum(w * y))
    pred_total = float(np.sum(w * y_hat))
    alpha = obs_total / pred_total

    # Parametric p-value (Poisson approximation)
    # H0: observed total ~ Poisson(predicted total)
    # By CLT: z ~ N(0,1) under H0
    z = (obs_total - pred_total) / np.sqrt(pred_total)
    p_value = float(2.0 * (1.0 - scipy.stats.norm.cdf(abs(z))))

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    bootstrap_alphas = np.empty(bootstrap_n, dtype=np.float64)
    for i in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        w_b = w[idx]
        obs_b = float(np.sum(w_b * y[idx]))
        pred_b = float(np.sum(w_b * y_hat[idx]))
        bootstrap_alphas[i] = obs_b / pred_b

    tail = (1.0 - confidence_level) / 2.0
    ci_lower, ci_upper = np.quantile(bootstrap_alphas, [tail, 1.0 - tail])

    is_balanced = bool(ci_lower <= 1.0 <= ci_upper)

    return BalanceResult(
        balance_ratio=alpha,
        observed_total=obs_total,
        predicted_total=pred_total,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=p_value,
        is_balanced=is_balanced,
        n_policies=n,
        total_exposure=float(np.sum(w)),
    )
