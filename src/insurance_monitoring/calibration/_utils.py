"""Shared utilities: input validation, exposure normalisation, tie-breaking."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def validate_inputs(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce inputs to float64 arrays.

    Parameters
    ----------
    y
        Observed loss rates or frequencies. Shape (n,).
    y_hat
        Model predictions (rates). Shape (n,).
    exposure
        Policy durations in years. If None, set to ones.

    Returns
    -------
    tuple of (y, y_hat, exposure) as float64 arrays.

    Raises
    ------
    ValueError
        If shapes do not match, if any exposure <= 0, or if any y_hat <= 0.
    """
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    if y_hat.ndim != 1:
        raise ValueError(f"y_hat must be 1-dimensional, got shape {y_hat.shape}")
    if len(y) != len(y_hat):
        raise ValueError(
            f"y and y_hat must have the same length: {len(y)} vs {len(y_hat)}"
        )
    if len(y) < 2:
        raise ValueError("At least 2 observations are required")

    if exposure is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(exposure, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError(f"exposure must be 1-dimensional, got shape {w.shape}")
        if len(w) != len(y):
            raise ValueError(
                f"exposure must have the same length as y: {len(w)} vs {len(y)}"
            )
        if np.any(w <= 0):
            raise ValueError(
                "All exposure values must be strictly positive. "
                "Found exposure <= 0 at positions: "
                f"{np.where(w <= 0)[0].tolist()}"
            )

    if np.any(y_hat <= 0):
        raise ValueError(
            "All y_hat values must be strictly positive (predictions are rates). "
            f"Found y_hat <= 0 at positions: {np.where(y_hat <= 0)[0].tolist()}"
        )

    return y, y_hat, w


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Exposure-weighted mean of x."""
    return float(np.sum(w * x) / np.sum(w))


def jitter_for_ties(x: np.ndarray, rng: np.random.Generator, scale: float = 1e-10) -> np.ndarray:
    """Add tiny random noise to break ties in predictions.

    Isotonic regression requires strict ordering to avoid degenerate step functions
    when many predictions are identical (e.g., from a GLM with few rating factors).
    """
    return x + rng.uniform(-scale, scale, size=len(x))


def check_isotonic_complexity(n_steps: int, n_obs: int) -> None:
    """Warn if the isotonic step function is too complex relative to sample size.

    Following Wüthrich & Ziegel (SAJ 2024): under low signal-to-noise ratio,
    isotonic recalibration on holdout data may produce many small steps that
    fit noise rather than signal.
    """
    import warnings

    threshold = int(np.sqrt(n_obs))
    if n_steps > threshold:
        warnings.warn(
            f"Isotonic regression produced {n_steps} steps on {n_obs} observations. "
            f"Under low signal-to-noise ratio this may overfit the holdout sample "
            f"(threshold: sqrt({n_obs}) = {threshold} steps). "
            "Consider using check_auto_calibration with method='hosmer_lemeshow' "
            "or increasing the number of holdout observations.",
            stacklevel=3,
        )
