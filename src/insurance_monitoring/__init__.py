"""
insurance-monitoring: model drift detection, monitoring, and calibration for insurance pricing.

Modules
-------
drift
    PSI, CSI, KS test, Wasserstein distance for feature distribution monitoring.
drift_attribution
    DriftAttributor (TRIPODD): feature-interaction-aware drift attribution with
    Type I error control. Identifies which features explain model performance drift.
calibration
    A/E ratio, Hosmer-Lemeshow, and the full calibration suite:
    balance property test, auto-calibration, Murphy decomposition, rectification.
discrimination
    Gini coefficient, Gini drift tests (arXiv 2510.04556), Lorenz curves.
report
    MonitoringReport — combined check with traffic-light output.
thresholds
    Configurable threshold defaults (PSI, A/E, Gini).

Quick start
-----------
::

    from insurance_monitoring import MonitoringReport
    from insurance_monitoring.drift import psi
    from insurance_monitoring.drift_attribution import DriftAttributor
    from insurance_monitoring.calibration import ae_ratio, check_balance, CalibrationChecker
    from insurance_monitoring.discrimination import gini_coefficient

v0.4.0 changes
--------------
- ``DriftAttributor`` and ``DriftAttributionResult`` added to
  ``insurance_monitoring.drift_attribution``. Implements TRIPODD
  (Panda et al. 2025, arXiv:2503.06606): feature-interaction-aware drift
  detection with bootstrap Bonferroni Type I error control.
  Goes beyond PSI/KS — tells you *which* features (and interactions) explain
  why model performance has degraded. Exported from top-level package.

v0.3.3 changes
--------------
- ``GiniDriftResult`` and ``GiniDriftOneSampleResult`` typed dataclasses added to
  ``insurance_monitoring.discrimination``. The drift test functions now return these
  typed objects instead of plain dicts — attribute access (``result.significant``,
  ``result.gini_change``) works. Dict-style access (``result["significant"]``) is
  also supported for backward compatibility via ``__getitem__``. Both types exported
  from the top-level package.

v0.3.0 changes
--------------
- insurance-calibration absorbed as ``insurance_monitoring.calibration`` sub-package.
  Adds: ``check_balance``, ``check_auto_calibration``, ``murphy_decomposition``,
  ``rectify_balance``, ``isotonic_recalibrate``, ``CalibrationChecker``, all
  deviance functions, result types (``BalanceResult``, ``AutoCalibResult``,
  ``MurphyResult``, ``CalibrationReport``), and plot helpers.
- ``insurance-calibration`` is no longer a separate optional dependency — these
  capabilities are now built in. ``murphy_distribution`` in ``MonitoringReport``
  now always works.
- matplotlib added to core dependencies (required for calibration plots).
- All existing imports from ``insurance_monitoring.calibration`` unchanged.

v0.2.0 additions
----------------
- ``gini_drift_test_onesample``: one-sample bootstrap design (Algorithm 3,
  arXiv 2510.04556). Tests monitor data against a stored training Gini scalar
  without needing raw reference data. More natural for deployed model monitoring.
- Murphy decomposition in ``MonitoringReport``: set ``murphy_distribution``
  to enable MCB/DSC decomposition. Sharpens the RECALIBRATE vs REFIT
  recommendation.
- ``alpha=0.32`` default in ``GiniDriftThresholds`` and all drift tests, per
  arXiv 2510.04556 recommendation for monitoring (one-sigma rule gives earlier
  signals, reducing catastrophic misses).
"""

from insurance_monitoring.calibration import (
    ae_ratio,
    ae_ratio_ci,
    calibration_curve,
    hosmer_lemeshow,
    # Calibration suite
    check_balance,
    check_auto_calibration,
    murphy_decomposition,
    rectify_balance,
    isotonic_recalibrate,
    deviance,
    BalanceResult,
    AutoCalibResult,
    MurphyResult,
    CalibrationReport,
    CalibrationChecker,
)
from insurance_monitoring.discrimination import (
    gini_coefficient,
    gini_drift_test,
    gini_drift_test_onesample,
    GiniDriftResult,
    GiniDriftOneSampleResult,
    lorenz_curve,
)
from insurance_monitoring.drift import (
    csi,
    ks_test,
    psi,
    wasserstein_distance,
)
from insurance_monitoring.drift_attribution import (
    DriftAttributor,
    DriftAttributionResult,
)
from insurance_monitoring.report import MonitoringReport
from insurance_monitoring.thresholds import (
    AERatioThresholds,
    GiniDriftThresholds,
    MonitoringThresholds,
    PSIThresholds,
)

__version__ = "0.4.0"

__all__ = [
    # drift
    "psi",
    "csi",
    "ks_test",
    "wasserstein_distance",
    # drift attribution (TRIPODD)
    "DriftAttributor",
    "DriftAttributionResult",
    # calibration — A/E monitoring
    "ae_ratio",
    "ae_ratio_ci",
    "calibration_curve",
    "hosmer_lemeshow",
    # calibration suite
    "check_balance",
    "check_auto_calibration",
    "murphy_decomposition",
    "rectify_balance",
    "isotonic_recalibrate",
    "deviance",
    "BalanceResult",
    "AutoCalibResult",
    "MurphyResult",
    "CalibrationReport",
    "CalibrationChecker",
    # discrimination
    "gini_coefficient",
    "gini_drift_test",
    "gini_drift_test_onesample",
    "GiniDriftResult",
    "GiniDriftOneSampleResult",
    "lorenz_curve",
    # report
    "MonitoringReport",
    # thresholds
    "MonitoringThresholds",
    "PSIThresholds",
    "AERatioThresholds",
    "GiniDriftThresholds",
]
