"""Murphy score decomposition: UNC - DSC + MCB (with GMCB/LMCB split)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._deviance import deviance
from ._rectify import isotonic_recalibrate
from ._types import MurphyResult
from ._utils import validate_inputs


def murphy_decomposition(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    exposure: npt.ArrayLike | None = None,
    distribution: str = "poisson",
    tweedie_power: float = 1.5,
    seed: int | None = None,
) -> MurphyResult:
    """Decompose the deviance loss into UNC, DSC, MCB (= GMCB + LMCB).

    The Murphy decomposition identity is::

        D(y, y_hat) = UNC - DSC + MCB

    where:

    - **UNC** (Uncertainty): D(y, y_bar) — baseline deviance from the
      intercept-only (grand mean) model. This is determined by the data
      difficulty, not the model.

    - **DSC** (Discrimination): UNC - D(y, y_hat_rc) — improvement in deviance
      from having a well-ranked model vs the grand mean. A model with no
      discriminatory power has DSC = 0.

    - **MCB** (Miscalibration): D(y, y_hat) - D(y, y_hat_rc) — excess deviance
      from having wrong price levels independent of the ranking. A well-calibrated
      model has MCB = 0.

    MCB decomposes further into::

        MCB = GMCB + LMCB

    - **GMCB** (Global MCB): D(y, y_hat) - D(y, alpha * y_hat) — portion
      removable by balance correction (multiplying all predictions by alpha).

    - **LMCB** (Local MCB): residual miscalibration after balance correction.
      Requires model refit or isotonic recalibration to fix.

    The verdict is::

        'OK'          if MCB / UNC < 1% and DSC > 0
        'RECALIBRATE' if GMCB > LMCB (global shift dominates)
        'REFIT'       if LMCB >= GMCB (local structure is wrong)

    Parameters
    ----------
    y
        Observed loss rates. Shape (n,).
    y_hat
        Model predictions (rates). Shape (n,).
    exposure
        Policy durations. If None, assumed uniform.
    distribution
        Loss distribution: 'poisson', 'gamma', 'tweedie', 'normal'.
    tweedie_power
        Tweedie variance power (only used when distribution='tweedie').
    seed
        Random seed (unused currently; reserved for bootstrap CI extension).

    Returns
    -------
    MurphyResult
        All Murphy components and the diagnostic verdict.

    Notes
    -----
    The ``y_hat_rc`` (isotonically recalibrated predictions) is computed on the
    same data passed in. Always pass holdout data, never training data.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> exposure = np.ones(n)
    >>> y_hat = rng.gamma(2, 0.05, n)
    >>> y = rng.poisson(y_hat) / 1.0  # perfectly calibrated
    >>> result = murphy_decomposition(y, y_hat, exposure)
    >>> result.miscalibration < result.discrimination  # good model: DSC >> MCB
    True
    """
    y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)

    D = lambda a, b: deviance(a, b, w, distribution, tweedie_power)

    # Grand mean (intercept-only model)
    y_bar = float(np.sum(w * y_arr) / np.sum(w))
    y_bar_arr = np.full(len(y_arr), y_bar, dtype=np.float64)

    # Isotonically recalibrated predictions
    y_hat_rc = isotonic_recalibrate(y_arr, y_hat_arr, w)
    # Clip to avoid numerical issues in deviance
    y_hat_rc = np.maximum(y_hat_rc, 1e-10)

    # Balance-corrected predictions (for GMCB split)
    alpha = float(np.sum(w * y_arr) / np.sum(w * y_hat_arr))
    y_hat_bc = alpha * y_hat_arr
    y_hat_bc_rc = isotonic_recalibrate(y_arr, y_hat_bc, w)
    y_hat_bc_rc = np.maximum(y_hat_bc_rc, 1e-10)

    # Murphy components
    d_y_yhat = D(y_arr, y_hat_arr)
    d_y_ybar = D(y_arr, y_bar_arr)
    d_y_yhat_rc = D(y_arr, y_hat_rc)
    d_y_yhat_bc = D(y_arr, y_hat_bc)

    unc = d_y_ybar
    dsc = d_y_ybar - d_y_yhat_rc   # how much better than grand mean after recalibration
    mcb = d_y_yhat - d_y_yhat_rc   # excess due to wrong levels
    gmcb = d_y_yhat - d_y_yhat_bc  # portion fixed by balance correction

    # LMCB: residual local miscalibration after balance correction
    # LMCB = MCB - GMCB, but compute directly for numerical stability
    d_y_yhat_bc_rc = D(y_arr, y_hat_bc_rc)
    lmcb = d_y_yhat_bc - d_y_yhat_bc_rc

    # Apply floor at zero for numerical noise
    dsc = max(dsc, 0.0)
    mcb = max(mcb, 0.0)
    gmcb = max(gmcb, 0.0)
    lmcb = max(lmcb, 0.0)

    # Percentage contributions relative to total deviance
    total_dev = d_y_yhat
    dsc_pct = 100.0 * dsc / total_dev if total_dev > 0 else 0.0
    mcb_pct = 100.0 * mcb / total_dev if total_dev > 0 else 0.0

    # Verdict logic
    if unc > 0 and mcb / unc < 0.01 and dsc > 0:
        verdict = "OK"
    elif gmcb >= lmcb:
        verdict = "RECALIBRATE"
    else:
        verdict = "REFIT"

    return MurphyResult(
        total_deviance=total_dev,
        uncertainty=unc,
        discrimination=dsc,
        miscalibration=mcb,
        global_mcb=gmcb,
        local_mcb=lmcb,
        discrimination_pct=dsc_pct,
        miscalibration_pct=mcb_pct,
        verdict=verdict,
    )
