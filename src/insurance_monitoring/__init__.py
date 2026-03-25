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
gini_drift
    GiniDriftTest: class-based two-sample asymptotic z-test for whether
    ranking power has changed between reference and monitoring periods.
    Requires raw data for both periods. Result in GiniDriftTestResult.
business_value
    Loss Ratio Error (LRE) metric from Evans Hedges (2025), arXiv:2512.03242.
    Converts Pearson correlation rho into expected loss ratio impact. Use
    lre_compare() to quantify the financial value of a model improvement.
report
    MonitoringReport — combined check with traffic-light output.
sequential
    Anytime-valid A/B testing for champion/challenger experiments using mSPRT
    (Johari et al. 2022). Valid type I error control at every interim check.
    Supports Poisson frequency, log-normal severity, and compound loss ratio tests.
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
    from insurance_monitoring.business_value import lre_compare, loss_ratio_error

v0.9.2 changes
--------------
- ``gini_drift`` module added. ``GiniDriftTest`` class: two-sample asymptotic z-test for
  whether the Gini coefficient (ranking power) has changed between reference and monitoring
  periods. Based on Wüthrich, Merz & Noll (2025), arXiv:2510.04556 Theorem 1 + Algorithm 2.

  New public API:

  - ``GiniDriftTest(reference_actual, reference_predicted, monitor_actual,
    monitor_predicted, reference_exposure=None, monitor_exposure=None,
    n_bootstrap=200, alpha=0.32, random_state=None)`` — class with lazy evaluation.
  - ``.test()`` — returns ``GiniDriftTestResult`` with gini_reference, gini_monitor,
    delta, z_statistic, p_value, significant, se_reference, se_monitor.
  - ``.summary()`` — governance-ready plain-text report paragraph.
  - ``GiniDriftTestResult`` — typed dataclass with dict-style access.

  No new dependencies.

v0.9.1 changes
--------------
- ``business_value`` module added. Implements the Loss Ratio Error (LRE) metric
  from Evans Hedges (2025), arXiv:2512.03242.

  New public API:

  - ``loss_ratio_error(rho, cv, eta)`` — expected LR uplift above perfect-model
    baseline (E_LR from Definition 2). Returns 0 for rho=1.
  - ``loss_ratio(rho, cv, eta, margin=1.0)`` — expected portfolio loss ratio at
    correlation rho (Theorem 1). Perfect model gives 1/margin.
  - ``lre_compare(rho_old, rho_new, cv, eta, margin=1.0)`` — compare two models,
    returning ``LREResult`` with lr_old, lr_new, delta_lr, delta_lr_bps, and
    loss ratio errors for both. delta_lr_bps is the headline figure for business
    cases: multiply by GWP to get annual financial impact.
  - ``calibrate_eta(rho_observed, cv, lr_observed, margin=1.0)`` — reverse-solve
    Theorem 1 to recover the implied demand elasticity from historical data.
  - ``LREResult`` dataclass for structured output.

  No new dependencies. scipy.optimize.brentq used for calibrate_eta.

v0.9.0 changes
--------------
- ``MonitoringTracker`` added — MLflow model registry integration.
  Attaches insurance-monitoring results to registered MLflow models.
  MLflow is an optional dependency: ``pip install insurance-monitoring[mlflow]``.

v0.8.0 changes
--------------
- ``sequential`` module completed with full test coverage. All 10 test cases from
  the spec now pass: type I error control (Monte Carlo n=500), power check,
  batching idempotency, confidence sequence coverage, loss ratio e-value
  multiplication, Bayesian posterior bounds, and null behaviour.
- ``notebooks/sequential_testing.ipynb`` added: Databricks-ready demo showing
  the complete mSPRT workflow on synthetic motor book data with 35k champion /
  15k challenger policies, monthly updates, and early stopping.
- No new dependencies. numpy, scipy, polars remain the only maths stack.

v0.7.0 changes
--------------
- ``PITMonitor``, ``PITAlarm``, ``PITSummary`` added to
  ``insurance_monitoring.calibration``. Anytime-valid calibration change detection
  via probability integral transforms and mixture e-processes. Reference:
  Henzi, Murph, Ziegel (2025), arXiv:2603.13156.

  Guarantee: P(ever alarm | model calibrated) <= alpha, for all t, forever.
  Solves the repeated-testing inflation in monthly H-L / A/E monitoring.

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

from insurance_monitoring.business_value import (
    loss_ratio_error,
    loss_ratio,
    lre_compare,
    calibrate_eta,
    LREResult,
)
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
from insurance_monitoring.gini_drift import (
    GiniDriftTest,
    GiniDriftTestResult,
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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-monitoring")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

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
    # gini drift test class (v1.0.0)
    "GiniDriftTest",
    "GiniDriftTestResult",
    # business value / LRE (v0.9.1)
    "loss_ratio_error",
    "loss_ratio",
    "lre_compare",
    "calibrate_eta",
    "LREResult",
    # report
    "MonitoringReport",
    # sequential A/B testing (v0.5.0+, tests completed v0.8.0)
    "SequentialTest",
    "SequentialTestResult",
    "sequential_test_from_df",
    # thresholds
    "MonitoringThresholds",
    "PSIThresholds",
    "AERatioThresholds",
    "GiniDriftThresholds",
    # MLflow integration (optional -- requires mlflow)
    "MonitoringTracker",
]


def __getattr__(name: str):
    """Lazy import for optional-dependency modules.

    MonitoringTracker requires mlflow. We don't import it at module load time
    so that the rest of the library works without mlflow installed.
    """
    if name == "MonitoringTracker":
        from insurance_monitoring.mlflow_tracker import MonitoringTracker
        return MonitoringTracker
    raise AttributeError(f"module 'insurance_monitoring' has no attribute {name!r}")
