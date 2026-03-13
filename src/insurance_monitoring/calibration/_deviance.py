"""Exposure-weighted deviance functions for Poisson, Gamma, Tweedie, and Normal."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def poisson_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None = None,
) -> float:
    """Exposure-weighted mean Poisson deviance.

    The unit deviance is::

        d(y, mu) = 2 * [y * log(y / mu) - (y - mu)]

    with the convention 0 * log(0) = 0, which ensures the term vanishes when
    the observed rate is zero (no claim).

    Parameters
    ----------
    y
        Observed rates (claims per unit exposure). Shape (n,).
    mu
        Predicted rates. Must be strictly positive. Shape (n,).
    exposure
        Policy durations. If None, uniform weighting is used.

    Returns
    -------
    float
        Exposure-weighted mean deviance.
    """
    w = np.ones(len(y), dtype=np.float64) if exposure is None else np.asarray(exposure, dtype=np.float64)
    log_term = np.where(y > 0, y * np.log(y / mu), 0.0)
    unit_dev = 2.0 * (log_term - (y - mu))
    return float(np.sum(w * unit_dev) / np.sum(w))


def gamma_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None = None,
) -> float:
    """Exposure-weighted mean Gamma deviance.

    The unit deviance is::

        d(y, mu) = 2 * [log(mu / y) + y / mu - 1]

    Gamma deviance is undefined when y <= 0. This is the correct actuarial
    behaviour — Gamma is appropriate for claim severity (given a claim occurred),
    not for frequency data that includes zeros.

    Parameters
    ----------
    y
        Observed severity values. Must be strictly positive. Shape (n,).
    mu
        Predicted severity. Must be strictly positive. Shape (n,).
    exposure
        Claim counts or policy durations used as weights. If None, uniform.

    Returns
    -------
    float
        Exposure-weighted mean deviance.

    Raises
    ------
    ValueError
        If any y <= 0.
    """
    w = np.ones(len(y), dtype=np.float64) if exposure is None else np.asarray(exposure, dtype=np.float64)
    if np.any(y <= 0):
        raise ValueError(
            "Gamma deviance is undefined for y <= 0. "
            "Use Gamma distribution only for severity models (y = claim amount given claim occurred). "
            "For frequency models with zeros, use Poisson or Tweedie."
        )
    unit_dev = 2.0 * (np.log(mu / y) + y / mu - 1.0)
    return float(np.sum(w * unit_dev) / np.sum(w))


def tweedie_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None = None,
    power: float = 1.5,
) -> float:
    """Exposure-weighted mean Tweedie deviance for compound Poisson-Gamma (1 < p < 2).

    The unit deviance is::

        d(y, mu) = 2 * [y^(2-p) / ((1-p)(2-p)) - y * mu^(1-p) / (1-p) + mu^(2-p) / (2-p)]

    When y = 0, the first term y^(2-p) is replaced by 0 (the limit as y -> 0+).
    This is appropriate for pure premium models where some policies have no claims.

    Parameters
    ----------
    y
        Observed pure premiums. Non-negative. Shape (n,).
    mu
        Predicted pure premiums. Must be strictly positive. Shape (n,).
    exposure
        Policy durations. If None, uniform weighting is used.
    power
        Tweedie variance power. Must satisfy 1 < power < 2.

    Returns
    -------
    float
        Exposure-weighted mean deviance.

    Raises
    ------
    ValueError
        If power is not in (1, 2).
    """
    if not (1.0 < power < 2.0):
        raise ValueError(
            f"Tweedie power must satisfy 1 < p < 2 for compound Poisson-Gamma. "
            f"Got power={power}. "
            "For p=1 use Poisson; for p=2 use Gamma."
        )
    w = np.ones(len(y), dtype=np.float64) if exposure is None else np.asarray(exposure, dtype=np.float64)
    p = power
    eps = np.finfo(np.float64).tiny  # smallest positive float64, ~5e-324

    # When y=0, the term y^(2-p) -> 0 as y -> 0+, so we can safely use max(y, eps)
    # only for the logarithm-like term; the y*mu^(1-p)/(1-p) term handles y=0 directly.
    y_safe = np.where(y > 0, y, eps)
    t1 = y_safe ** (2 - p) / ((1 - p) * (2 - p))
    t2 = y * mu ** (1 - p) / (1 - p)
    t3 = mu ** (2 - p) / (2 - p)
    unit_dev = 2.0 * (t1 - t2 + t3)
    # Zero-claim rows: y=0 means t2=0, t1 approaches 0, leaving only t3
    unit_dev = np.where(y > 0, unit_dev, 2.0 * mu ** (2 - p) / (2 - p))
    return float(np.sum(w * unit_dev) / np.sum(w))


def normal_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None = None,
) -> float:
    """Exposure-weighted mean squared error (Normal / Gaussian deviance).

    The unit deviance is simply::

        d(y, mu) = (y - mu)^2

    Parameters
    ----------
    y
        Observed values. Shape (n,).
    mu
        Predicted values. Shape (n,).
    exposure
        Observation weights. If None, uniform weighting.

    Returns
    -------
    float
        Exposure-weighted mean squared error.
    """
    w = np.ones(len(y), dtype=np.float64) if exposure is None else np.asarray(exposure, dtype=np.float64)
    unit_dev = (y - mu) ** 2
    return float(np.sum(w * unit_dev) / np.sum(w))


_DEVIANCE_FNS = {
    "poisson": poisson_deviance,
    "gamma": gamma_deviance,
    "normal": normal_deviance,
}


def deviance(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None = None,
    distribution: str = "poisson",
    tweedie_power: float = 1.5,
) -> float:
    """Dispatch to the appropriate deviance function by distribution name.

    Parameters
    ----------
    y
        Observed values.
    mu
        Predicted values.
    exposure
        Observation weights / exposures.
    distribution
        One of 'poisson', 'gamma', 'tweedie', 'normal'.
    tweedie_power
        Variance power for Tweedie (only used when distribution='tweedie').

    Returns
    -------
    float
        Exposure-weighted mean deviance.

    Raises
    ------
    ValueError
        If distribution is not recognised.
    """
    dist = distribution.lower()
    if dist == "tweedie":
        return tweedie_deviance(y, mu, exposure, power=tweedie_power)
    if dist in _DEVIANCE_FNS:
        return _DEVIANCE_FNS[dist](y, mu, exposure)
    raise ValueError(
        f"Unknown distribution '{distribution}'. "
        "Supported: 'poisson', 'gamma', 'tweedie', 'normal'."
    )
