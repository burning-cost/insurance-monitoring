"""
Model calibration for insurance pricing — A/E monitoring plus the full calibration suite.

This package consolidates two calibration layers:

1. **A/E monitoring** (from insurance-monitoring v0.1–0.2):
   - :func:`ae_ratio` — Actual/Expected ratio, optionally segmented
   - :func:`ae_ratio_ci` — A/E with Poisson or normal confidence interval
   - :func:`calibration_curve` — Binned reliability diagram
   - :func:`hosmer_lemeshow` — Hosmer-Lemeshow goodness-of-fit test

2. **Calibration suite** (absorbed from insurance-calibration):
   - :func:`check_balance` — Global balance property test (Lindholm & Wüthrich 2025)
   - :func:`check_auto_calibration` — Per-cohort auto-calibration test
   - :func:`murphy_decomposition` — UNC/DSC/MCB decomposition (arXiv:2510.04556)
   - :func:`rectify_balance` — Multiplicative or affine balance correction
   - :func:`isotonic_recalibrate` — Empirical auto-calibration (holdout only)
   - :func:`deviance` and family — Exposure-weighted Poisson/Gamma/Tweedie/Normal deviance
   - :class:`CalibrationChecker` — Fit/check pipeline class
   - Result types: :class:`BalanceResult`, :class:`AutoCalibResult`,
     :class:`MurphyResult`, :class:`CalibrationReport`

The A/E layer is appropriate for routine monitoring (monthly/quarterly dashboards).
The calibration suite is for model sign-off, root-cause diagnosis, and rectification.

All functions are exposure-weighted and model-agnostic — they accept any prediction array.

Supported distributions: ``'poisson'``, ``'gamma'``, ``'tweedie'``, ``'normal'``.

Quick start
-----------
::

    from insurance_monitoring.calibration import ae_ratio, check_balance, murphy_decomposition
    import numpy as np

    # Monthly A/E monitoring
    ae = ae_ratio(actual_claims, predicted_freq, exposure=car_years)

    # Model sign-off: full calibration check
    from insurance_monitoring.calibration import CalibrationChecker
    checker = CalibrationChecker(distribution='poisson')
    report = checker.check(y_holdout, y_hat_holdout, exposure_holdout)
    print(report.verdict())  # 'OK', 'RECALIBRATE', or 'REFIT'
"""

from __future__ import annotations

# ------------------------------------------------------------------
# A/E monitoring layer (original insurance-monitoring calibration.py)
# ------------------------------------------------------------------
from insurance_monitoring.calibration._ae import (
    ae_ratio,
    ae_ratio_ci,
    calibration_curve,
    hosmer_lemeshow,
)

# ------------------------------------------------------------------
# Calibration suite (absorbed from insurance-calibration)
# ------------------------------------------------------------------
from insurance_monitoring.calibration._balance import check_balance
from insurance_monitoring.calibration._autocal import check_auto_calibration
from insurance_monitoring.calibration._murphy import murphy_decomposition
from insurance_monitoring.calibration._rectify import rectify_balance, isotonic_recalibrate
from insurance_monitoring.calibration._deviance import (
    deviance,
    poisson_deviance,
    gamma_deviance,
    tweedie_deviance,
    normal_deviance,
)
from insurance_monitoring.calibration._types import (
    BalanceResult,
    AutoCalibResult,
    MurphyResult,
    CalibrationReport,
)
from insurance_monitoring.calibration._checker import CalibrationChecker
from insurance_monitoring.calibration._plots import (
    plot_auto_calibration,
    plot_murphy,
    plot_balance_over_time,
    plot_calibration_report,
)

__all__ = [
    # A/E monitoring
    "ae_ratio",
    "ae_ratio_ci",
    "calibration_curve",
    "hosmer_lemeshow",
    # Calibration suite — functional API
    "check_balance",
    "check_auto_calibration",
    "murphy_decomposition",
    "rectify_balance",
    "isotonic_recalibrate",
    # Deviance functions
    "deviance",
    "poisson_deviance",
    "gamma_deviance",
    "tweedie_deviance",
    "normal_deviance",
    # Result types
    "BalanceResult",
    "AutoCalibResult",
    "MurphyResult",
    "CalibrationReport",
    # Pipeline class
    "CalibrationChecker",
    # Plots
    "plot_auto_calibration",
    "plot_murphy",
    "plot_balance_over_time",
    "plot_calibration_report",
]
