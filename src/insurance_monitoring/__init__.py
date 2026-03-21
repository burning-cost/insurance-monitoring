"""
insurance-monitoring: model drift detection, monitoring, and calibration for insurance pricing.

Modules
-------
drift
    PSI, CSI, KS test, Wasserstein distance for feature distribution monitoring.
drift_attribution
    DriftAttributor (TRIPODD): feature-interaction-aware drift attribution with
    Type I error control. Identifies which features explain model performance drift.
interpretable_drift
    InterpretableDriftDetector: upgraded TRIPODD implementation with exposure
    weighting, FDR control, single-loop bootstrap, cached subset risks, and
    Poisson deviance loss. The right choice for high-d feature sets or when
    earned exposure weighting is required.
calibration
    A/E ratio, Hosmer-Lemeshow, and the full calibration suite:
    balance property test, auto-calibration, Murphy decomposition, rectification.
    PITMonitor for anytime-valid calibration change detection in production.
discrimination
    Gini coefficient, Gini drift tests (arXiv 2510.04556), Lorenz curves,
    and GiniDriftBootstrapTest with percentile CIs and governance plot.
report
    MonitoringReport — combined check with traffic-light output.
sequential
    Anytime-valid A/B testing for champion/challenger experiments using mSPRT
    (Johari et al. 2022). Valid type I error control at every interim check.
thresholds
    Configurable threshold defaults (PSI, A/E, Gini).

Quick start
-----------
::

    from insurance_monitoring import MonitoringReport
    from insurance_monitoring.drift import psi
    from insurance_monitoring.drift_attribution import DriftAttributor
    from insurance_monitoring.interpretable_drift import InterpretableDriftDetector
    from insurance_monitoring.calibration import ae_ratio, check_balance, CalibrationChecker
    from insurance_monitoring.discrimination import gini_coefficient, GiniDriftBootstrapTest
    from insurance_monitoring.sequential import SequentialTest
    from insurance_monitoring import PITMonitor

v0.7.0 changes
--------------
- ``PITMonitor``, ``PITAlarm``, ``PITSummary`` added to
  ``insurance_monitoring.calibration``. Anytime-valid calibration change detection
  via probability integral transforms and mixture e-processes. Reference:
  Henzi, Murph, Ziegel (2025), arXiv:2603.13156.

  Guarantee: P(ever alarm | model calibrated) <= alpha, for all t, forever.
  Solves the repeated-testing inflation in monthly H-L / A/E monitoring.

  New dependency: ``sortedcontainers>=2.4`` (pure Python, zero transitive deps).

- ``InterpretableDriftDetector`` and ``InterpretableDriftResult`` added. Seven
  improvements over ``DriftAttributor``:

  - Exposure weighting via ``weights`` parameter (correct for mixed policy terms).
  - Benjamini-Hochberg FDR control alongside Bonferroni. Use ``error_control='fdr'``
    for d >= 10 rating factors where Bonferroni is too conservative.
  - Single bootstrap loop: thresholds and p-values in one pass, halving cost
    versus the two-loop design in DriftAttributor.
  - Subset risk caching at fit_reference(): reference-side model calls saved
    across all subsequent test() calls.
  - Polars-native API: accepts pl.DataFrame and pl.Series directly.
  - Poisson deviance loss: canonical GLM goodness-of-fit for frequency models.
  - Explicit update_reference(): no auto-retrain on drift detection.

  ``DriftAttributor`` is unchanged — both classes coexist.

v0.6.0 changes
--------------
- ``GiniDriftBootstrapTest`` and ``GiniBootstrapResult`` added to
  ``insurance_monitoring.discrimination``. Class-based one-sample bootstrap
  Gini drift test with:

  - Percentile bootstrap CI for both the monitor Gini and the Gini change
  - ``.plot()`` method producing a bootstrap histogram with CI shading, training
    Gini line, monitor Gini line, and z/p annotation box — a standard IFoA/PRA
    model validation deliverable
  - ``.summary()`` returning a governance-ready plain-text report paragraph
  - Bootstrap replicates stored for post-hoc inspection
  - Lazy evaluation (bootstrap runs on first ``.test()`` call, then cached)

  Wraps the existing ``_bootstrap_gini_samples()`` helper — no duplicated
  bootstrap logic. BCa CIs deliberately omitted (jackknife cost not justified
  for approximately-normal Gini bootstrap distribution at n >= 200).

- ``MonitoringReport`` gains ``gini_bootstrap: bool = False`` parameter.
  When True, uses GiniDriftBootstrapTest and adds ``ci_lower`` / ``ci_upper``
  fields to ``results_["gini"]`` and corresponding rows to ``to_polars()``.

v0.5.0 changes
--------------
- ``SequentialTest`` and ``SequentialTestResult`` added for anytime-valid A/B testing.
  Uses mixture SPRT (Johari et al. 2022). Supports frequency, severity, and loss ratio
  metrics. No pre-specified sample size or look schedule required.
- ``sequential_test_from_df()`` convenience function for DataFrame-based workflows.
- All three exports available from the top-level package.

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
    # Sequential calibration monitoring (v0.7.0)
    PITMonitor,
    PITAlarm,
    PITSummary,
)
from insurance_monitoring.discrimination import (
    gini_coefficient,
    gini_drift_test,
    gini_drift_test_onesample,
    GiniDriftResult,
    GiniDriftOneSampleResult,
    GiniBootstrapResult,
    GiniDriftBootstrapTest,
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
from insurance_monitoring.interpretable_drift import (
    InterpretableDriftDetector,
    InterpretableDriftResult,
)
from insurance_monitoring.report import MonitoringReport
from insurance_monitoring.sequential import (
    SequentialTest,
    SequentialTestResult,
    sequential_test_from_df,
)
from insurance_monitoring.thresholds import (
    AERatioThresholds,
    GiniDriftThresholds,
    MonitoringThresholds,
    PSIThresholds,
)

__version__ = "0.7.0"

__all__ = [
    # drift
    "psi",
    "csi",
    "ks_test",
    "wasserstein_distance",
    # drift attribution (TRIPODD)
    "DriftAttributor",
    "DriftAttributionResult",
    # interpretable drift detection (TRIPODD+)
    "InterpretableDriftDetector",
    "InterpretableDriftResult",
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
    # sequential calibration monitoring (v0.7.0)
    "PITMonitor",
    "PITAlarm",
    "PITSummary",
    # discrimination
    "gini_coefficient",
    "gini_drift_test",
    "gini_drift_test_onesample",
    "GiniDriftResult",
    "GiniDriftOneSampleResult",
    "GiniBootstrapResult",
    "GiniDriftBootstrapTest",
    "lorenz_curve",
    # report
    "MonitoringReport",
    # sequential A/B testing
    "SequentialTest",
    "SequentialTestResult",
    "sequential_test_from_df",
    # sequential calibration monitoring (v0.7.0)
    "PITMonitor",
    "PITAlarm",
    "PITSummary",
    # thresholds
    "MonitoringThresholds",
    "PSIThresholds",
    "AERatioThresholds",
    "GiniDriftThresholds",
]
