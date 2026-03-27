# insurance-monitoring

[![PyPI](https://img.shields.io/pypi/v/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Tests](https://github.com/burning-cost/insurance-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-monitoring/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-monitoring/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-monitoring/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-monitoring/discussions). Found it useful? A star helps others find it.

Models drift. Regulators notice.

A pricing model that is 15% cheap on young drivers and 15% expensive on mature drivers reads 1.00 at portfolio level — and triggers no alarm. The loss ratio deteriorates twelve months later. By then, the PRA (SS1/23) expects you to have had a monitoring framework that would have caught it. insurance-monitoring detects per-feature distribution shifts and model discrimination drift, not just the headline A/E ratio, so you find the problem before it reaches the accounts or the supervisor.

---

## Part of the Burning Cost stack

Takes the outputs of any fitted pricing model and a stream of actual experience. Feeds drift signals and calibration verdicts into [insurance-governance](https://github.com/burning-cost/insurance-governance) for model risk committee packs. Pairs with [insurance-fairness](https://github.com/burning-cost/insurance-fairness) to monitor per-protected-group A/E ratios under Consumer Duty. → [See the full stack](https://burning-cost.github.io/stack/)

---

## Manual monitoring vs insurance-monitoring

| Task | Manual approach | insurance-monitoring |
|------|----------------|----------------------|
| Population Stability Index | Excel macro per factor, re-coded each quarter, unweighted | `psi()` / `csi()` — exposure-weighted, Polars-native, traffic-light band |
| Feature drift heatmap | Engineer writes one-off script; no standard thresholds | `csi()` — one call, all rating factors, PRA-aligned thresholds |
| A/E ratio with CI | Custom formula in SQL, no confidence interval | `ae_ratio_ci()` — Wilson CI, exposure-weighted, RAG status |
| Discrimination drift | Gini computed ad hoc; no test for statistical significance | `gini_drift_test()` / `GiniDriftBootstrapTest` — arXiv 2510.04556 bootstrap CI, governance plot |
| RECALIBRATE vs REFIT decision | Actuary judgment call, not documented | `MonitoringReport.recommendation` — arXiv 2510.04556 decision tree, Murphy decomposition sharpens it |
| Repeated monthly testing inflation | H-L / A/E tested afresh each month; inflated false-positive rate | `PITMonitor` — anytime-valid, P(ever false alarm | calibrated) ≤ α, forever |
| Champion/challenger A/B test | Wait for pre-specified sample size; cannot stop early | `SequentialTest` — mSPRT, valid at every interim check, supports frequency/severity/loss ratio |
| Drift attribution (which feature explains the performance change?) | PSI-by-eye; no interaction-aware method | `DriftAttributor` / `InterpretableDriftDetector` — TRIPODD, FDR control, exposure weighting |
| Monitoring report for model risk committee | Manual Word document | `MonitoringReport.to_polars()` — flat DataFrame, writes directly to Delta table |

---

## Quick start

```python
import numpy as np
import polars as pl
from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi

rng = np.random.default_rng(42)
# Training period
actual_ref = rng.poisson(0.08, 10_000).astype(float)
predicted_ref = np.full(10_000, 0.08)
exposure_ref = rng.uniform(0.5, 1.0, 10_000)
# Current period — model has drifted
actual_cur = rng.poisson(0.11, 5_000).astype(float)   # loss rate up
predicted_cur = np.full(5_000, 0.08)                   # model unchanged
exposure_cur = rng.uniform(0.5, 1.0, 5_000)

report = MonitoringReport(
    reference_actual=actual_ref, reference_predicted=predicted_ref,
    current_actual=actual_cur, current_predicted=predicted_cur,
    exposure=exposure_cur, murphy_distribution="poisson",
)
print(report.recommendation)          # => 'RECALIBRATE' or 'REFIT'
print(report.to_polars())             # flat DataFrame: metric / value / band
```

See `examples/quickstart.py` for a fully self-contained example with feature drift and CSI.

---

## Features

- **PSI / CSI** — Population Stability Index and Characteristic Stability Index across all rating factors; exposure-weighted; traffic-light bands aligned with PRA SS1/23 model risk expectations
- **A/E ratio with confidence interval** — Wilson CI, exposure-weighted, RAG status; the primary calibration metric for UK pricing teams
- **Gini drift test** — two-sample and one-sample bootstrap designs (arXiv 2510.04556); percentile CI; governance plot suitable for model validation packs
- **Decision tree** (`MonitoringReport.recommendation`) — NO_ACTION / MONITOR_CLOSELY / RECALIBRATE / REFIT / INVESTIGATE; Murphy decomposition sharpens the RECALIBRATE vs REFIT distinction
- **Murphy decomposition** — decomposes scoring loss into uncertainty, discrimination (DSC), and miscalibration (MCB); when DSC falls, REFIT; when MCB dominates, RECALIBRATE
- **Anytime-valid calibration monitoring** (`PITMonitor`) — probability integral transform e-processes (Henzi, Murph, Ziegel 2025); valid type I error control at every monthly check, forever; solves repeated-testing inflation
- **Sequential A/B testing** (`SequentialTest`) — mixture SPRT (Johari et al. 2022); supports Poisson frequency, log-normal severity, and compound loss ratio; no pre-specified sample size
- **TRIPODD drift attribution** (`DriftAttributor`, `InterpretableDriftDetector`) — feature-interaction-aware; identifies which factors explain model performance degradation; Bonferroni or Benjamini-Hochberg FDR control; exposure weighting; Poisson deviance loss
- **MLflow integration** (`MonitoringTracker`) — logs all metrics and bands to an MLflow run; requires `mlflow` (optional dependency)
- **Polars-native throughout** — no pandas required; flat `to_polars()` output writes directly to Delta tables on Databricks

---

## Installation

```bash
pip install insurance-monitoring
# or
uv add insurance-monitoring
```

**Dependencies:** polars, numpy, scipy, matplotlib

MLflow integration (optional):

```bash
pip install insurance-monitoring mlflow
```

---

## Modules

### `MonitoringReport`

The single entry point for a complete monthly or quarterly monitoring run.

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=train_claims,
    reference_predicted=train_predicted,
    current_actual=current_claims,
    current_predicted=current_predicted,
    exposure=current_exposure,
    feature_df_reference=train_features,    # pl.DataFrame
    feature_df_current=current_features,    # pl.DataFrame
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",          # Murphy decomposition — always available
    gini_bootstrap=True,                    # percentile CI on Gini (v0.6.0)
)
print(report.recommendation)   # 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'
print(report.to_dict())        # nested dict — JSON serialisable, log to MLflow
print(report.to_polars())      # flat DataFrame — write to Delta table
```

Recommendation logic follows the three-stage decision tree from arXiv 2510.04556:
- Gini OK + A/E OK → NO_ACTION
- A/E bad only → RECALIBRATE (update intercept)
- Gini bad → REFIT (rebuild on recent data)
- Murphy decomposition present: overrides the heuristic when miscalibration vs discrimination are unambiguous

### `drift` — PSI, CSI, KS, Wasserstein

```python
from insurance_monitoring.drift import psi, csi, ks_test, wasserstein_distance
import polars as pl

# PSI on a single feature — exposure-weighted
drift_val = psi(
    reference=driver_age_train,
    current=driver_age_now,
    n_bins=10,
    exposure_weights=exposure_now,    # car-years, not policy count
    reference_exposure=exposure_train,
)
print(f"PSI: {drift_val:.3f}")   # < 0.10 green | 0.10–0.25 amber | > 0.25 red

# CSI heatmap across all rating factors — returns pl.DataFrame with band column
csi_df = csi(
    reference_df=train_df,
    current_df=current_df,
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_group"],
)
print(csi_df)  # feature | csi | band
```

Use PSI/CSI for operational dashboards. Use `ks_test` for formal hypothesis testing at quarter-end (note: over-sensitive at n > 500k). Use `wasserstein_distance` when communicating to non-technical stakeholders — it reports drift in original feature units.

### `calibration` — A/E, balance, Murphy, rectification

```python
from insurance_monitoring import (
    ae_ratio, ae_ratio_ci,
    check_balance, check_auto_calibration,
    murphy_decomposition,
    rectify_balance, isotonic_recalibrate,
    CalibrationChecker,
)

# A/E ratio with Wilson CI
result = ae_ratio_ci(actual, predicted, exposure=exposure)
print(f"A/E = {result['ae']:.3f}  95% CI [{result['lower']:.3f}, {result['upper']:.3f}]")

# Murphy decomposition — distinguishes miscalibration from discrimination loss
murphy = murphy_decomposition(y=actual, y_hat=predicted, exposure=exposure, distribution="poisson")
print(f"DSC (discrimination score): {murphy.discrimination:.4f}")
print(f"MCB (miscalibration):       {murphy.miscalibration:.4f}")
print(f"Verdict: {murphy.verdict}")   # 'RECALIBRATE' or 'REFIT'

# Full calibration audit in one call
checker = CalibrationChecker(distribution="poisson")
report = checker.run(actual, predicted, exposure=exposure)
```

### `discrimination` — Gini drift

```python
from insurance_monitoring import (
    gini_coefficient,
    gini_drift_test_onesample,
    GiniDriftBootstrapTest,
)

# One-sample design: tests monitor data against a stored training Gini scalar
# (more natural for deployed model monitoring — reference data may not be available)
result = gini_drift_test_onesample(
    training_gini=0.42,
    monitor_actual=current_claims,
    monitor_predicted=current_predicted,
    monitor_exposure=current_exposure,
)
print(f"Gini change: {result.gini_change:+.3f}  p={result.p_value:.3f}  [{result.significant}]")

# Class-based API with governance plot
bt = GiniDriftBootstrapTest(
    training_gini=0.42,
    monitor_actual=current_claims,
    monitor_predicted=current_predicted,
    monitor_exposure=current_exposure,
    n_bootstrap=500,
)
bt_result = bt.test()
print(bt_result.summary())   # governance-ready paragraph
bt.plot()                    # bootstrap histogram with CI shading — IFoA/PRA deliverable
```

### `calibration.PITMonitor` — anytime-valid calibration monitoring (v0.7.0)

The standard pattern — run a Hosmer-Lemeshow test or check A/E each month — inflates the false-positive rate. After twelve months at α=0.05, the chance of a false alarm exceeds 40% even if the model is perfectly calibrated.

`PITMonitor` solves this using probability integral transform e-processes (Henzi, Murph, Ziegel 2025, arXiv:2603.13156). The guarantee is: P(ever raise an alarm | model calibrated) ≤ α, for all t, forever.

```python
from insurance_monitoring import PITMonitor

monitor = PITMonitor(alpha=0.05, distribution="poisson")
monitor.fit(train_actual, train_predicted)

for month_actual, month_predicted in monthly_data:
    alarm = monitor.update(month_actual, month_predicted)
    if alarm.triggered:
        print(f"Calibration alarm: e-value = {alarm.e_value:.2f}")
        break
```

### `sequential` — anytime-valid A/B testing (v0.8.0)

Champion/challenger pricing experiments are routinely run with a pre-specified end date. Checking early to stop a bad challenger is statistically invalid under classical hypothesis testing. `SequentialTest` uses mixture SPRT (Johari et al. 2022) — you can check at every interim update and stop early if the challenger is clearly better or worse, with full type I error control.

```python
from insurance_monitoring import SequentialTest

test = SequentialTest(metric="frequency", alpha=0.05)
test.set_null(reference_rate=0.08)

# Monthly updates — stop whenever the e-value crosses the threshold
for batch_actual, batch_predicted, batch_exposure in monthly_batches:
    result = test.update(batch_actual, batch_predicted, exposure=batch_exposure)
    print(f"e-value: {result.e_value:.2f}  stopped: {result.stopped}")
    if result.stopped:
        break
```

### `drift_attribution` — TRIPODD (v0.4.0+)

PSI tells you that driver_age has drifted. TRIPODD tells you whether that driver_age drift explains why the model's discrimination has fallen — accounting for feature interactions. Use `InterpretableDriftDetector` (v0.7.0) for high-dimensional feature sets or when exposure weighting is required.

```python
from insurance_monitoring import InterpretableDriftDetector

detector = InterpretableDriftDetector(
    error_control="fdr",    # Benjamini-Hochberg for d >= 10 factors
    loss="poisson_deviance", # canonical for frequency models
)
detector.fit_reference(X_ref, y_ref, weights=exposure_ref)
result = detector.test(X_cur, y_cur, weights=exposure_cur)
print(result.significant_features)  # list of features that explain the performance shift
```

---

## Regulatory context

**PRA SS1/23** (Model Risk Management, March 2023) requires insurers to maintain a model monitoring framework that detects deterioration in model performance. The expectation is documented thresholds, regular testing, and a governance process that escalates to the model risk committee when thresholds are breached. A/E ratio and Gini monitoring are the two metrics most commonly cited in SS1/23 supervisory discussions.

**Consumer Duty (PS22/9)** requires ongoing monitoring of whether pricing outcomes are fair across customer groups, not just at point of sale. The combination of `MonitoringReport` and [insurance-fairness](https://github.com/burning-cost/insurance-fairness) `calibration_by_group()` produces a per-protected-group A/E split suitable for Consumer Duty Outcome 4 monitoring.

---

## Expected performance

On a 50,000-policy synthetic UK motor portfolio:

| Task | Time | Notes |
|------|------|-------|
| PSI on one feature | < 0.1s | Exposure-weighted |
| CSI across 10 features | < 0.5s | Returns Polars DataFrame |
| `ae_ratio_ci()` | < 0.1s | Wilson CI |
| `MonitoringReport` (no Murphy) | < 2s | A/E + Gini + CSI |
| `MonitoringReport` with Murphy | < 5s | Adds MCB/DSC decomposition |
| `GiniDriftBootstrapTest` (n_bootstrap=500) | 5–15s | Percentile CI |
| `SequentialTest` batch update | < 0.1s | Per monthly update |
| `InterpretableDriftDetector` (10 features) | 30–90s | FDR-controlled bootstrap |

[Run the full workflow on Databricks](https://github.com/burning-cost/insurance-monitoring/blob/main/notebooks/quickstart.ipynb)

---

## Installation

```bash
pip install insurance-monitoring
# or
uv add insurance-monitoring
```

---

## Related libraries

| Library | What it does |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Per-protected-group A/E calibration, proxy detection, Consumer Duty audit reports |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double ML — establishes whether a rating factor causally drives risk or is a proxy |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model risk committee packs and FCA Consumer Duty documentation |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Interpretable GLM-style models whose factors can be monitored directly |

---

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).
