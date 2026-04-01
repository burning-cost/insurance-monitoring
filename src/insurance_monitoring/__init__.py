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
    Supports Poisson frequency, log-normal severity, and compound loss ratio tests.
multicalibration
    MulticalibrationMonitor: subgroup-level calibration monitoring. Dual-gate
    alerting (relative bias + z-stat) for (bin, group) cells. Based on
    Denuit, Michaelides & Trufin (2026), arXiv:2603.16317.
thresholds
    Configurable threshold defaults (PSI, A/E, Gini).
conformal_chart
    ConformalControlChart: distribution-free Shewhart-equivalent chart via
    conformal p-values (arXiv:2512.23602). Calibrated FAR replaces parametric
    3-sigma limits. No normality assumption — exact finite-sample guarantee.
    MultivariateConformalMonitor: multivariate model-health monitoring via
    conformal anomaly detection. Single p-value per reporting cycle.

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
    from insurance_monitoring import ConformalControlChart, MultivariateConformalMonitor

v0.12.0 changes
---------------
- ConformedControlChart, ConformedProcessMonitor, ConformedControlResult,
  and ConformedMonitorResult added (insurance_monitoring.conformal_spc).
  sklearn-style fit/predict API for conformal SPC, complementing the existing
  ConformalControlChart (fit/monitor) in conformal_chart.py.

  - ConformedControlChart: built-in score functions (absolute, relative,
    studentized). Accepts raw scalar observations; NCS computed internally.
    Use for univariate residual/A/E/loss-ratio monitoring where you want
    the chart to handle normalisation.
  - ConformedProcessMonitor: multivariate monitoring with a single
    X_calibration array (internal 80/20 train/calibration split). Supports
    ocsvm (OneClassSVM), isolation_forest, or any sklearn-duck-typed model.
  - Both classes accept alpha=0.0 (no signals) and alpha=1.0 (all signals)
    as edge cases. Both return typed result dataclasses with signal_rate,
    signals (bool array), and n_calibration.
  - Module is self-contained: no dependency on conformal_chart.py.

v0.11.0 changes
---------------
- ConformalControlChart, MultivariateConformalMonitor, and
  ConformalChartResult added (insurance_monitoring.conformal_chart).
  Distribution-free SPC via conformal prediction (Burger 2025, arXiv:2512.23602).

  - Calibrated false alarm rate at alpha=0.05 (default) or any user-specified
    level. No normality required — exact finite-sample coverage guarantee under
    exchangeability.
  - Four NCS helpers: absolute residual, relative residual (best for GLM/GBM
    policy-level monitoring), median deviation, studentized.
  - MultivariateConformalMonitor accepts any sklearn-duck-typed anomaly
    model (IsolationForest default, sklearn optional). Single conformal p-value
    per monitoring cycle — suitable for governance dashboards combining PSI,
    A/E, and Gini metrics.
  - ConformalChartResult.summary() returns a governance-ready paragraph
    for PRA SS3/17 model monitoring submissions.

v0.10.0 changes
---------------
- PricingDriftMonitor, PricingDriftResult, and CalibTestResult
  added to insurance_monitoring.pricing_drift. Implements the full Brauer,
  Menzel & Wüthrich (2025) two-step monitoring framework (arXiv:2510.04556):

  - Step 1: Gini bootstrap ranking drift test (Algorithm 3). The z-test
    denominator is sigma_hat[G_old] (reference bootstrap SE), not sigma_hat[G_new].
  - Step 2: GMCB + LMCB bootstrap auto-calibration tests (Algorithm 4).
    GMCB tests for global level shift (fixable by balance correction).
    LMCB tests for local cohort-level miscalibration (requires refit).
  - Structured REFIT / RECALIBRATE / OK verdict.
  - Default alpha_gini=0.32 (one-sigma early-warning rule, paper Remark 3).
  - governance-ready summary() method for PRA model validation reports.
  - Orchestrates GiniBootstrapMonitor and MurphyDecomposition internally.

- CalibrationCUSUM, CUSUMAlarm, and CUSUMSummary added to
  insurance_monitoring.cusum. Implements the calibration CUSUM chart
  from Franck, Driscoll, Szajnfarber, Woodall (2025), arXiv:2510.25573:

  - Likelihood-ratio CUSUM statistic using LLO alternative hypothesis.
  - Dynamic probability control limits (DPCLs) maintaining constant CFAR.
  - Supports binary Bernoulli outcomes (paper method) and Poisson count
    data (insurance frequency adaptation).
  - In-control ARL0 ~ 1/cfar by construction. At cfar=0.005: ARL0=200.
  - Out-of-control ARL1 ~ 20-40 for 2x miscalibration at cfar=0.005.
  - Statistic resets after alarm; monitoring continues automatically.
  - .plot() method produces the governance chart (statistic + control limits).

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
from insurance_monitoring.gini_drift import GiniDriftTest, GiniDriftTestResult
from insurance_monitoring.gini_monitoring import (
    GiniDriftMonitor,
    GiniDriftResult as GiniDriftMonitorResult,
    GiniBootstrapMonitor,
    GiniBootstrapResult as GiniBootstrapMonitorResult,
    MurphyDecomposition,
    _MurphyComponents as MurphyComponents,
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
from insurance_monitoring.multicalibration import (
    MulticalibrationMonitor,
    MulticalibrationResult,
    MulticalibCell,
    MulticalibThresholds,
)


from insurance_monitoring.conformal_chart import (
    ConformalControlChart,
    MultivariateConformalMonitor,
    ConformalChartResult,
)
from insurance_monitoring.conformal_spc import (
    ConformedControlChart,
    ConformedProcessMonitor,
    ConformedControlResult,
    ConformedMonitorResult,
)
from insurance_monitoring.pricing_drift import (
    PricingDriftMonitor,
    PricingDriftResult,
    CalibTestResult,
)
from insurance_monitoring.cusum import (
    CalibrationCUSUM,
    CUSUMAlarm,
    CUSUMSummary,
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
    "GiniDriftTest",
    "GiniDriftTestResult",
    "lorenz_curve",
    # Gini monitoring with fit/test interface (gini_monitoring module)
    "GiniDriftMonitor",
    "GiniBootstrapMonitor",
    "MurphyDecomposition",
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
    # multicalibration monitoring (v0.9.3)
    "MulticalibrationMonitor",
    "MulticalibrationResult",
    "MulticalibCell",
    "MulticalibThresholds",
    # conformal SPC (v0.11.0)
    "ConformalControlChart",
    "MultivariateConformalMonitor",
    "ConformalChartResult",
    # conformed SPC — fit/predict API (v0.12.0)
    "ConformedControlChart",
    "ConformedProcessMonitor",
    "ConformedControlResult",
    "ConformedMonitorResult",
    # pricing drift monitoring (v0.10.0)
    "PricingDriftMonitor",
    "PricingDriftResult",
    "CalibTestResult",
    # calibration CUSUM monitoring (v0.10.0)
    "CalibrationCUSUM",
    "CUSUMAlarm",
    "CUSUMSummary",
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
