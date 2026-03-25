"""
Business value translation for insurance model improvements.

Converts model accuracy (Pearson correlation rho) into expected loss ratio
impact using the closed-form framework from Evans Hedges (2025),
arXiv:2512.03242.

The key insight is that a pricing model with Pearson correlation rho < 1
systematically produces a portfolio loss ratio above the perfect-model
baseline 1/M. Theorem 1 gives the closed-form expression. This module
implements the core formulas plus tooling for comparing two models and
recovering the implied elasticity from historical data.

Practical framing: if your current model has rho=0.92 and a candidate
model has rho=0.95, how many basis points of loss ratio improvement can
you expect? That is what ``lre_compare`` answers.

Reference
---------
Evans Hedges, C. (2025). "A Theoretical Framework Bridging Model
Validation and Loss Ratio in Insurance." arXiv:2512.03242.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core formula helpers (private)
# ---------------------------------------------------------------------------


def _ratio_factor(rho: float, cv: float) -> float:
    """
    Compute the ratio inside the power in Theorem 1.

        R = (1 + rho^2 * CV^{-2}) / (rho^2 * (1 + CV^{-2}))

    At rho=1 this reduces to exactly 1.0.
    """
    rho2 = rho ** 2
    cv2_inv = 1.0 / (cv ** 2)
    numerator = 1.0 + rho2 * cv2_inv
    denominator = rho2 * (1.0 + cv2_inv)
    return numerator / denominator


def _validate_inputs(
    rho: float,
    cv: float,
    eta: float,
    margin: float = 1.0,
) -> None:
    """Raise ValueError if inputs are outside valid ranges."""
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if cv <= 0.0:
        raise ValueError(f"cv must be > 0, got {cv}")
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}")
    if margin <= 0.0:
        raise ValueError(f"margin must be > 0, got {margin}")
    if eta <= 0.5:
        warnings.warn(
            f"eta={eta} <= 0.5. The Evans Hedges (2025) formula requires "
            "eta > 0.5 for the exponent (2*eta-1)/2 to be positive. "
            "Results below this threshold may be counterintuitive. "
            "Typical UK motor/home elasticity estimates are in (0.5, 3.0).",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def loss_ratio_error(rho: float, cv: float, eta: float) -> float:
    """
    Expected loss ratio error: uplift above perfect-model baseline (Definition 2).

    A perfect model (rho=1) achieves LR = 1.0 when the pricing margin M=1.
    Any rho < 1 produces a systematic upward bias:

        E_LR = ((1 + rho^2 * CV^{-2}) / (rho^2 * (1 + CV^{-2})))^{(2*eta-1)/2} - 1

    Returns 0.0 for rho >= 0.999 (numerically perfect model).

    Parameters
    ----------
    rho:
        Pearson correlation between model predictions and true loss cost,
        in (0, 1].
    cv:
        Coefficient of variation of the true loss cost distribution. Must
        be > 0. Typical range: 0.5 (homogeneous book) to 3.0 (heavy-tailed
        liability). Use the true loss cost CV, not the predicted CV.
    eta:
        Price elasticity of demand. Must be > 0; paper requires eta > 0.5
        for the formula to be well-behaved. Common values for UK personal
        lines are in (0.8, 2.0).

    Returns
    -------
    float
        Loss ratio error (non-negative). E_LR = 0.0 for a perfect model.

    Reference
    ---------
    Evans Hedges (2025), arXiv:2512.03242, Definition 2.
    """
    _validate_inputs(rho, cv, eta, margin=1.0)
    if rho >= 0.999:
        return 0.0
    exponent = (2.0 * eta - 1.0) / 2.0
    return _ratio_factor(rho, cv) ** exponent - 1.0


def loss_ratio(
    rho: float,
    cv: float,
    eta: float,
    margin: float = 1.0,
) -> float:
    """
    Expected portfolio loss ratio at Pearson correlation rho (Theorem 1).

    At M=1 (break-even pricing), a perfect model achieves LR=1.0. Any
    imperfection (rho < 1) inflates the LR above that baseline.

        LR = (1/M) * ((1 + rho^2 * CV^{-2}) / (rho^2 * (1 + CV^{-2})))^{(2*eta-1)/2}

    Parameters
    ----------
    rho:
        Pearson correlation between model predictions and true loss cost,
        in (0, 1].
    cv:
        Coefficient of variation of the true loss cost distribution.
    eta:
        Price elasticity of demand (> 0; paper requires > 0.5).
    margin:
        Pricing margin factor M. The perfect-model LR is 1/M. Default 1.0
        (break-even). For a 70% loss ratio target use M = 1/0.70 ≈ 1.4286.

    Returns
    -------
    float
        Expected portfolio loss ratio.

    Reference
    ---------
    Evans Hedges (2025), arXiv:2512.03242, Theorem 1.
    """
    _validate_inputs(rho, cv, eta, margin)
    if rho >= 0.999:
        return 1.0 / margin
    exponent = (2.0 * eta - 1.0) / 2.0
    return (1.0 / margin) * (_ratio_factor(rho, cv) ** exponent)


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclass
class LREResult:
    """
    Structured comparison of two models' expected loss ratio impact.

    Generated by ``lre_compare()``. All monetary-scale interpretation assumes
    you apply the ``delta_lr_bps`` figure to your gross written premium.

    Attributes
    ----------
    rho_old:
        Pearson correlation of the current model.
    rho_new:
        Pearson correlation of the candidate model.
    cv:
        Coefficient of variation of true loss costs.
    eta:
        Price elasticity parameter.
    margin:
        Pricing margin factor M used.
    lr_old:
        Expected portfolio LR at rho_old.
    lr_new:
        Expected portfolio LR at rho_new.
    delta_lr:
        lr_new - lr_old (negative means improvement).
    delta_lr_bps:
        delta_lr expressed in basis points (delta_lr * 10_000).
    e_lr_old:
        Loss ratio error of the current model.
    e_lr_new:
        Loss ratio error of the candidate model.
    """

    rho_old: float
    rho_new: float
    cv: float
    eta: float
    margin: float
    lr_old: float
    lr_new: float
    delta_lr: float
    delta_lr_bps: float
    e_lr_old: float
    e_lr_new: float

    def __repr__(self) -> str:
        direction = "improvement" if self.delta_lr_bps < 0 else "deterioration"
        return (
            f"LREResult("
            f"rho {self.rho_old:.3f}->{self.rho_new:.3f}, "
            f"delta_lr={self.delta_lr_bps:+.1f}bps [{direction}], "
            f"lr_old={self.lr_old:.4f}, lr_new={self.lr_new:.4f}"
            f")"
        )


def lre_compare(
    rho_old: float,
    rho_new: float,
    cv: float,
    eta: float,
    margin: float = 1.0,
) -> LREResult:
    """
    Compare two models by their expected loss ratio impact.

    Returns the LR and loss ratio error for both models plus the improvement
    delta in loss ratio points and basis points. Use this to build a business
    case for a model upgrade: given £X GWP, multiply by ``delta_lr`` to get
    the annual financial impact.

    Parameters
    ----------
    rho_old:
        Pearson correlation of the current deployed model, in (0, 1].
    rho_new:
        Pearson correlation of the candidate model, in (0, 1].
    cv:
        Coefficient of variation of true loss costs. Both models operate
        on the same book, so a shared CV is appropriate.
    eta:
        Price elasticity of demand.
    margin:
        Pricing margin factor M. Default 1.0 (break-even target).

    Returns
    -------
    LREResult
        Structured comparison with lr_old, lr_new, delta_lr, delta_lr_bps,
        e_lr_old, e_lr_new.

    Examples
    --------
    >>> result = lre_compare(rho_old=0.92, rho_new=0.95, cv=1.2, eta=1.5)
    >>> print(f"Improvement: {result.delta_lr_bps:.1f} bps")
    """
    # Validate both rho values (other params validated inside lr/lre calls)
    if not (0.0 < rho_old <= 1.0):
        raise ValueError(f"rho_old must be in (0, 1], got {rho_old}")
    if not (0.0 < rho_new <= 1.0):
        raise ValueError(f"rho_new must be in (0, 1], got {rho_new}")

    lr_old = loss_ratio(rho_old, cv, eta, margin)
    lr_new = loss_ratio(rho_new, cv, eta, margin)
    e_lr_old = loss_ratio_error(rho_old, cv, eta)
    e_lr_new = loss_ratio_error(rho_new, cv, eta)
    delta_lr = lr_new - lr_old

    return LREResult(
        rho_old=rho_old,
        rho_new=rho_new,
        cv=cv,
        eta=eta,
        margin=margin,
        lr_old=lr_old,
        lr_new=lr_new,
        delta_lr=delta_lr,
        delta_lr_bps=delta_lr * 10_000.0,
        e_lr_old=e_lr_old,
        e_lr_new=e_lr_new,
    )


def calibrate_eta(
    rho_observed: float,
    cv: float,
    lr_observed: float,
    margin: float = 1.0,
    eta_bounds: tuple[float, float] = (0.5, 5.0),
) -> Optional[float]:
    """
    Reverse-solve Theorem 1 to recover implied price elasticity from data.

    Given an observed portfolio loss ratio and known model correlation and
    loss cost CV, this finds the eta that makes Theorem 1 consistent with
    the observation. Useful when no direct demand model estimate is available
    but you have historical LR data alongside model validation statistics.

    Uses scipy.optimize.brentq for reliable bracketed root-finding. Returns
    None if no solution exists within ``eta_bounds``.

    Parameters
    ----------
    rho_observed:
        Observed Pearson correlation of the pricing model, in (0, 1].
    cv:
        Coefficient of variation of the loss cost distribution.
    lr_observed:
        Observed portfolio loss ratio (e.g. 0.74 for a 74% LR). Must be
        positive and achievable within the given eta_bounds.
    margin:
        Pricing margin factor M used when writing the business. Default 1.0.
    eta_bounds:
        (eta_low, eta_high) search interval for brentq. Default (0.5, 5.0).
        Widen this if ``calibrate_eta`` returns None unexpectedly.

    Returns
    -------
    float or None
        Implied eta, or None if the observed LR cannot be reproduced within
        ``eta_bounds``.

    Reference
    ---------
    Evans Hedges (2025), arXiv:2512.03242, Theorem 1 (inverted).
    """
    if lr_observed <= 0.0:
        raise ValueError(f"lr_observed must be > 0, got {lr_observed}")
    # Validate the non-eta inputs
    _validate_inputs(rho_observed, cv, eta=1.0, margin=margin)

    eta_low, eta_high = eta_bounds

    def _residual(eta: float) -> float:
        return loss_ratio(rho_observed, cv, eta, margin) - lr_observed

    try:
        f_low = _residual(eta_low)
        f_high = _residual(eta_high)
    except (ValueError, ZeroDivisionError):
        return None

    if f_low * f_high > 0:
        warnings.warn(
            f"No eta in {eta_bounds} reproduces lr_observed={lr_observed:.4f} "
            f"at rho={rho_observed:.3f}, cv={cv:.3f}, margin={margin:.4f}. "
            "Returning None. Try widening eta_bounds.",
            UserWarning,
            stacklevel=2,
        )
        return None

    eta_star = brentq(_residual, eta_low, eta_high, xtol=1e-8, rtol=1e-8)
    return float(eta_star)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "loss_ratio_error",
    "loss_ratio",
    "lre_compare",
    "calibrate_eta",
    "LREResult",
]
