# insurance-monitoring

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-monitoring)

Deployed insurance pricing models go stale. The portfolio ages, the claims environment shifts, regulators change the rules. Without systematic monitoring you find out about it when the loss ratio deteriorates — typically 12 to 18 months after the model started misfiring.

This library gives UK pricing teams the specific tools to catch that drift early: exposure-weighted PSI for feature distribution, A/E ratios with Poisson confidence intervals for calibration, and the Gini drift z-test from [arXiv 2510.04556](https://arxiv.org/abs/2510.04556) — currently the only statistically rigorous actuarial monitoring framework in the literature.

It produces traffic-light outputs (green/amber/red) that match how a Head of Pricing actually reads a monitoring pack, and a decision recommendation based on the Murphy score decomposition: recalibrate (update the intercept, one hour of work) or refit (rebuild the model, weeks of work).

**No scikit-learn. No pandas. Polars-native throughout.**

## Installation

```bash
uv add insurance-monitoring
```

Or with pip:

```bash
pip install insurance-monitoring
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
# Portfolio has aged, young drivers more numerous, claim rate up
pred_cur = rng.uniform(0.05, 0.20, 15_000)
act_cur = rng.poisson(pred_cur * 1.08).astype(float)  # model is 8% optimistic

# Quick check: feature drift on model score
score_psi = psi(pred_ref, pred_cur)
print(f"Score PSI: {score_psi:.3f}")  # < 0.10 = stable, > 0.25 = investigate

# A/E ratio (aggregate)
from insurance_monitoring import ae_ratio_ci
ae_result = ae_ratio_ci(act_cur, pred_cur)
print(f"A/E: {ae_result['ae']:.3f}  (95% CI: {ae_result['lower']:.3f}–{ae_result['upper']:.3f})")

# Gini coefficient (discrimination)
gini = gini_coefficient(act_cur, pred_cur)
print(f"Gini: {gini:.3f}")

# Combined monitoring report with traffic lights
report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
)
print(report.recommendation)  # 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'
print(report.to_polars())     # flat DataFrame with metric / value / band columns
```

## Modules

### `drift` — Feature distribution monitoring

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

**On exposure-weighted PSI**: standard PSI treats every policy equally regardless of how long it was on risk. If your book renews quarterly and mixes 1-month and 12-month policies, unweighted PSI is wrong. The `exposure_weights` parameter weights bin proportions by earned exposure — correct for insurance.

### `calibration` — A/E ratio and calibration checks

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with Poisson CI (exact Garwood intervals)
result = ae_ratio_ci(actual, predicted, exposure=exposure)
# {'ae': 1.08, 'lower': 1.04, 'upper': 1.12, 'n_claims': 342, 'n_expected': 317}

# Segmented A/E: where is the model misfiring?
from insurance_monitoring.calibration import ae_ratio
seg_ae = ae_ratio(
    actual, predicted, exposure=exposure,
    segments=driver_age_bands,   # np.array(['17-24', '25-39', ...])
)
# Returns Polars DataFrame: segment | actual | expected | ae_ratio | n_policies
```

**On the IBNR problem**: the A/E ratio is only reliable on mature accident periods. For motor, that means at least 12 months of claims development. For liability, 24+ months. If you run monthly monitoring on recent accident months, apply chain-ladder development factors first — otherwise you will see artificially low A/E ratios that recover as claims develop.

### `discrimination` — Gini drift test

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

The Gini drift test is the distinguishing feature of this library. Most monitoring tools will tell you whether A/E has moved. This tells you whether the model's *ranking* has degraded — the difference between a cheap recalibration and a full refit.

### `report` — Combined monitoring in one call

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
)

print(report.recommendation)
# 'REFIT' | 'RECALIBRATE' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

df = report.to_polars()
# metric              | value  | band
# ae_ratio            | 1.08   | amber
# gini_current        | 0.39   | amber
# gini_p_value        | 0.054  | amber
# csi_driver_age      | 0.14   | amber
# csi_vehicle_age     | 0.03   | green
# recommendation      | nan    | REFIT
```

### `thresholds` — Configurable traffic lights

```python
from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds

# Tighten PSI thresholds for a large motor book with monthly monitoring
custom = MonitoringThresholds(
    psi=PSIThresholds(green_max=0.05, amber_max=0.15),
)
report = MonitoringReport(..., thresholds=custom)
```

Default thresholds follow industry convention (PSI: 0.1/0.25 from FICO/credit scoring; A/E: 0.95–1.05 green, 0.90–1.10 amber; Gini: p < 0.10 amber, p < 0.05 red).

## Decision framework

The `recommendation` property implements the three-stage decision tree from arXiv 2510.04556, mapped to actuarial practice:

| Signal | Recommendation | Action |
|--------|---------------|--------|
| No drift in any test | NO_ACTION | Continue, schedule next review |
| A/E red, Gini stable | RECALIBRATE | Update intercept/offset (hours of work) |
| Gini red | REFIT | Rebuild model on recent data (weeks of work) |
| Both red | INVESTIGATE | Manual review — check data quality first |
| Any amber | MONITOR_CLOSELY | Increase monitoring frequency |

## Databricks integration

The demo notebook at `notebooks/demo_monitoring.py` shows the full workflow on synthetic motor data and runs on Databricks serverless. Upload it to your workspace and schedule it as a monthly job against your MLflow inference table.

## Background

Built by [Burning Cost](https://burningcost.com) — insurance pricing education and open-source tooling for UK actuaries and pricing teams.

The Gini drift test implements the framework from:
> "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing", arXiv 2510.04556 (December 2025)

This is the only published actuarially-specific monitoring framework with proper asymptotic theory. As of March 2026, no other Python library implements it.

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burningcost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burningcost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burningcost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burningcost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burningcost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burningcost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burningcost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burningcost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burningcost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burningcost/insurance-causal) | Double Machine Learning for causal pricing inference |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burningcost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All libraries →](https://burningcost.github.io)

---

## Licence

MIT
