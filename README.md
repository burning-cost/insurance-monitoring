# insurance-monitoring

[![CI](https://github.com/burning-cost/insurance-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-monitoring/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Deployed insurance pricing models go stale. The portfolio ages, the claims environment shifts, regulators change the rules. Without systematic monitoring you find out about it when the loss ratio deteriorates — typically 12 to 18 months after the model started misfiring.

This library gives UK pricing teams two things in one install:

1. **Ongoing model monitoring** — exposure-weighted PSI for feature distribution, A/E ratios with Poisson confidence intervals, and the Gini drift z-test from [arXiv 2510.04556](https://arxiv.org/abs/2510.04556).
2. **Deep calibration diagnostics** — balance property testing, auto-calibration, Murphy decomposition (UNC/DSC/MCB), and rectification methods for model sign-off and root-cause analysis (Lindholm & Wüthrich, SAJ 2025).

The two layers serve the same person — the pricing actuary — at different points in the model lifecycle. Use the monitoring layer for monthly/quarterly dashboards. Use the calibration suite when a model needs to be signed off or when monitoring flags a problem you need to diagnose.

**No scikit-learn. No pandas. Polars-native throughout.**

## Installation

```bash
uv add insurance-monitoring
```

## Quick example

```python
import numpy as np
from insurance_monitoring import MonitoringReport, psi, ae_ratio, gini_coefficient

rng = np.random.default_rng(42)

# Reference period (model training window)
pred_ref = rng.uniform(0.05, 0.20, 50_000)
act_ref = rng.poisson(pred_ref).astype(float)

# Current monitoring period (18 months later)
pred_cur = rng.uniform(0.05, 0.20, 15_000)
act_cur = rng.poisson(pred_cur * 1.08).astype(float)  # model is 8% optimistic

# Combined monitoring report with traffic lights
report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    murphy_distribution="poisson",  # optional Murphy decomposition, now built in
)
print(report.recommendation)   # 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'
print(report.to_polars())      # flat DataFrame with metric / value / band columns
```

## Modules

### `calibration` - A/E ratio, calibration suite, Murphy decomposition

The calibration module has two layers. Use A/E for routine monitoring. Use the calibration suite for model sign-off.

**A/E ratio monitoring:**

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with Poisson CI (exact Garwood intervals)
result = ae_ratio_ci(actual, predicted, exposure=exposure)
# {'ae': 1.08, 'lower': 1.04, 'upper': 1.12, 'n_claims': 342, 'n_expected': 317}

# Segmented A/E: where is the model misfiring?
seg_ae = ae_ratio(
    actual, predicted, exposure=exposure,
    segments=driver_age_bands,
)
# Returns Polars DataFrame: segment | actual | expected | ae_ratio | n_policies
```

**Calibration suite — model sign-off:**

```python
from insurance_monitoring.calibration import CalibrationChecker

checker = CalibrationChecker(distribution='poisson', alpha=0.05)
report = checker.check(y_holdout, y_hat_holdout, exposure_holdout)

print(report.verdict())    # 'OK' | 'RECALIBRATE' | 'REFIT'
print(report.summary())    # human-readable diagnostic paragraph

# Individual components
print(report.balance)          # BalanceResult: global A/E ratio with bootstrap CI
print(report.auto_calibration) # AutoCalibResult: per-cohort bootstrap MCB test
print(report.murphy)           # MurphyResult: UNC/DSC/MCB/GMCB/LMCB decomposition
```

**Murphy decomposition directly:**

```python
from insurance_monitoring.calibration import murphy_decomposition

result = murphy_decomposition(y, y_hat, exposure, distribution='poisson')
# result.uncertainty     # baseline deviance (data difficulty)
# result.discrimination  # DSC: skill from ranking
# result.miscalibration  # MCB: excess from wrong price levels
# result.global_mcb      # GMCB: portion fixed by multiplying all predictions by A/E
# result.local_mcb       # LMCB: portion requiring model refit
# result.verdict         # 'OK' | 'RECALIBRATE' | 'REFIT'
```

**Why two calibration layers?** The A/E ratio answers "is the model globally right?". The Murphy decomposition answers "if it is wrong, is it wrong in a cheap way (scale factor) or an expensive way (the ranking is broken)?". You need both to make the RECALIBRATE vs REFIT decision correctly.

**On the IBNR problem**: the A/E ratio and balance test are only reliable on mature accident periods. For motor, at least 12 months of claims development. For liability, 24+ months. Apply chain-ladder factors first when monitoring recent accident months.

### `drift` - Feature distribution monitoring

```python
from insurance_monitoring.drift import psi, csi, ks_test, wasserstein_distance
import polars as pl

# PSI with exposure weighting (insurance-correct)
score_psi = psi(
    reference=score_train,
    current=score_q1_2025,
    n_bins=10,
    exposure_weights=earned_exposure,  # car-years, not policy count
)

# CSI heatmap across all rating factors
feature_ref = pl.DataFrame({"driver_age": [...], "vehicle_age": [...], "ncd_years": [...]})
feature_cur = pl.DataFrame({"driver_age": [...], "vehicle_age": [...], "ncd_years": [...]})
csi_table = csi(feature_ref, feature_cur, features=["driver_age", "vehicle_age", "ncd_years"])
# Returns: feature | csi | band

# Wasserstein: report drift in original units
d = wasserstein_distance(driver_ages_train, driver_ages_q1_2025)
print(f"Average driver age shifted by {d:.1f} years")
```

**On exposure-weighted PSI**: standard PSI treats every policy equally regardless of how long it was on risk. If your book renews quarterly and mixes 1-month and 12-month policies, unweighted PSI is wrong. The `exposure_weights` parameter weights bin proportions by earned exposure.

### `discrimination` - Gini drift test

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(act_ref, pred_ref, exposure=exp_ref)
gini_cur = gini_coefficient(act_cur, pred_cur, exposure=exp_cur)

# Statistical test: has Gini degraded significantly?
# Implements arXiv 2510.04556 Theorem 1
result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=50_000,
    n_current=15_000,
    reference_actual=act_ref, reference_predicted=pred_ref,
    current_actual=act_cur, current_predicted=pred_cur,
)
# {'z_statistic': -1.93, 'p_value': 0.054, 'gini_change': -0.03, 'significant': False}
```

The Gini drift test is the distinguishing feature of this library. Most monitoring tools tell you whether A/E has moved. This tells you whether the model's *ranking* has degraded — the difference between a cheap recalibration and a full refit.

### `report` - Combined monitoring in one call

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,  # Polars DataFrame
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",  # built in since v0.3.0, no extra install needed
)

print(report.recommendation)
# 'REFIT' | 'RECALIBRATE' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

df = report.to_polars()
# metric              | value  | band
# ae_ratio            | 1.08   | amber
# gini_current        | 0.39   | amber
# gini_p_value        | 0.054  | amber
# csi_driver_age      | 0.14   | amber
# murphy_discrimination | 0.041 | RECALIBRATE
# murphy_miscalibration | 0.003 | RECALIBRATE
# recommendation      | nan    | RECALIBRATE
```

### `thresholds` - Configurable traffic lights

```python
from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds

# Tighten PSI thresholds for a large motor book with monthly monitoring
custom = MonitoringThresholds(
    psi=PSIThresholds(green_max=0.05, amber_max=0.15),
)
report = MonitoringReport(..., thresholds=custom)
```

Default thresholds follow industry convention (PSI: 0.1/0.25 from FICO/credit scoring; A/E: 0.95–1.05 green, 0.90–1.10 amber; Gini: p < 0.32 amber, p < 0.10 red per arXiv 2510.04556 recommendation).

## Decision framework

The `recommendation` property implements the three-stage decision tree from arXiv 2510.04556, mapped to actuarial practice:

| Signal | Recommendation | Action |
|--------|---------------|--------|
| No drift in any test | NO_ACTION | Continue, schedule next review |
| A/E red, Gini stable | RECALIBRATE | Update intercept/offset (hours of work) |
| Gini red | REFIT | Rebuild model on recent data (weeks of work) |
| Both red | INVESTIGATE | Manual review — check data quality first |
| Any amber | MONITOR_CLOSELY | Increase monitoring frequency |

When `murphy_distribution` is set, the Murphy decomposition sharpens the RECALIBRATE vs REFIT distinction: if GMCB > LMCB (global shift dominates), RECALIBRATE; if LMCB >= GMCB (local structure is broken), REFIT.

## Calibration plots

The calibration module includes matplotlib visualisations for model documentation:

```python
from insurance_monitoring.calibration import (
    CalibrationChecker,
    plot_auto_calibration,
    plot_murphy,
    plot_calibration_report,
)

checker = CalibrationChecker(distribution='poisson')
report = checker.check(y, y_hat, exposure)

# Three-panel combined figure (auto-calibration + Murphy bar + per-bin heatmap)
fig = plot_calibration_report(report)
fig.savefig("model_calibration_sign_off.pdf")
```

## Databricks integration

The demo notebook at `notebooks/demo_monitoring.py` shows the full workflow on synthetic motor data and runs on Databricks serverless. Upload it to your workspace and schedule it as a monthly job against your MLflow inference table.

## What changed in v0.3.0

`insurance-calibration` was a separate package covering the three-property calibration framework (Lindholm & Wüthrich 2025) and Murphy decomposition. As of v0.3.0 it is absorbed into `insurance-monitoring` as the `calibration` sub-package.

All existing imports are unchanged. New imports follow the same pattern:

```python
# These already worked:
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# These are new in v0.3.0 (previously required separate install):
from insurance_monitoring.calibration import (
    check_balance, check_auto_calibration, murphy_decomposition,
    rectify_balance, isotonic_recalibrate, CalibrationChecker,
    BalanceResult, AutoCalibResult, MurphyResult, CalibrationReport,
    deviance, poisson_deviance, gamma_deviance,
    plot_auto_calibration, plot_murphy, plot_calibration_report,
)
```

## Background

The monitoring framework implements:
> "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing", arXiv 2510.04556 (December 2025)

The calibration suite implements:
> Lindholm & Wüthrich: "Three calibration properties for insurance pricing models" (SAJ 2025)
> Brauer et al.: arXiv:2510.04556 Section 4 — Murphy decomposition and the MCB bootstrap test

## Read more

[Your Pricing Model is Drifting (and You Probably Can't Tell)](https://burning-cost.github.io/2026/03/07/your-pricing-model-is-drifting.html) — why PSI alone is insufficient, and what it means when A/E is stable but the Gini is falling.

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs — when monitoring flags REFIT, use SHAP to diagnose which factors have drifted most |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | GLM interaction detection — a refit triggered by Gini degradation may need new interactions added |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID causal evaluation — if monitoring shows deterioration after a rate change, use this to isolate cause |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation — use monitoring outputs to decide when to retrain and validate the retrained model |
| [rate-optimiser](https://github.com/burning-cost/rate-optimiser) | Constrained rate change optimisation — monitoring informs when a rate adjustment is needed; rate-optimiser determines the right one |

[All Burning Cost libraries →](https://burning-cost.github.io)

---

## Licence

MIT
