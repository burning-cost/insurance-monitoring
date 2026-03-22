# insurance-monitoring

[![PyPI](https://img.shields.io/pypi/v/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-monitoring/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-monitoring/discussions). Found it useful? A star helps others find it.

**Your aggregate A/E ratio looks fine. Your model has been mispricing under-25s for eight months.**

Deployed insurance pricing models go stale. The portfolio ages, the claims environment shifts, regulators change the rules. Without systematic monitoring you find out about it when the loss ratio deteriorates — typically 12 to 18 months after the model started misfiring.

The central problem with aggregate A/E is that errors cancel at portfolio level. The model may be 15% cheap on young drivers and 15% expensive on mature drivers; the aggregate reads 1.00 and nobody raises an alarm. This library monitors the features, not just the headline number.

## Why bother

Benchmarked on synthetic UK motor data — 50,000 training policies (2019–2021), monitored against a 2023 portfolio with known induced shifts: young drivers (under 25) oversampled 2x, high-risk areas (E and F) oversampled 50%, conviction points shifted upward for 20% of policies.

> **Note on benchmark sizes**: the table below uses the 50,000/15,000 policy scenario. The detailed Performance section further down uses a smaller scenario (10,000 reference / 4,000 monitoring) for a faster local run. Both use the same DGP; the smaller run is provided for reproducibility without Databricks.

| Monitoring check | Manual A/E check | MonitoringReport (PSI/CSI) | Notes |
|------------------|------------------|----------------------------|-------|
| Aggregate A/E — shifted data | Computed | Same value | Both agree; neither alone is sufficient |
| driver_age distributional shift | Not detected | PSI RED (>0.25) | 2x young driver oversampling |
| area distributional shift | Not detected | PSI AMBER/RED | High-risk area overweighting |
| conviction_points shift | Not detected | PSI AMBER | 20% of policies shifted +1 point |
| Gini drift (ref vs shifted) | Not computed | Computed with bootstrap CI | Tests whether ranking has degraded |
| Structured audit trail | No | Yes (traffic-light report) | Suitable for inclusion in PRA SS3/17 (insurer model risk) documentation (see note below) |

The manual A/E check is blind to who is inside the portfolio. PSI per feature catches segment-level drift that cancels at portfolio level. The Gini drift z-test tells you whether the model's ranking has degraded — the difference between a cheap recalibration and a full refit.

> **Note on regulatory scope:** PRA SS3/17 (Supervisory Statement 3/17) is the primary model risk management standard for UK insurers under Solvency II. PRA SS1/23 covers model risk management for banks and building societies — not insurers. For insurers, the directly applicable references are SS3/17 and the FCA Consumer Duty outcome monitoring obligations. The structured audit trail this library produces is appropriate for inclusion in model risk documentation under SS3/17 or equivalent internal model governance frameworks.

▶ [Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py)

---

**Read more:** [Your Pricing Model is Drifting (and You Probably Can't Tell)](https://burning-cost.github.io/2026/03/07/your-pricing-model-is-drifting.html) — why PSI alone is insufficient, and what it means when A/E is stable but the Gini is falling.

---

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

This example uses named rating factors — which is how actuaries actually work with this data.

> **Runtime note**: this example uses 10,000 reference / 4,000 monitoring policies and runs in under 40 seconds locally. The Gini bootstrap (200 replicates, required for the drift z-test) is the dominant cost at scale; at 50k/15k it takes 3–5 minutes. Use the 10k/4k size for local exploration; run the full scale on Databricks.

```python
import polars as pl
import numpy as np
from insurance_monitoring import MonitoringReport

rng = np.random.default_rng(42)

# Reference period: training window (use 10k/4k for local runs; scale up on Databricks)
n_ref = 10_000
pred_ref = rng.uniform(0.05, 0.20, n_ref)
act_ref = rng.poisson(pred_ref).astype(float)

# Current monitoring period: 18 months into deployment
n_cur = 4_000
pred_cur = rng.uniform(0.05, 0.20, n_cur)
act_cur = rng.poisson(pred_cur * 1.08).astype(float)  # model underpredicted: actuals 8% above predictions (A/E ≈ 1.08)

# Feature DataFrames with named rating factors — pass these to get CSI per feature
feat_ref = pl.DataFrame({
    "driver_age":  rng.integers(18, 80, n_ref).tolist(),
    "vehicle_age": rng.integers(0, 15, n_ref).tolist(),
    "ncd_years":   rng.integers(0, 9, n_ref).tolist(),
})
feat_cur = pl.DataFrame({
    "driver_age":  rng.integers(25, 85, n_cur).tolist(),  # older drivers entering book
    "vehicle_age": rng.integers(0, 15, n_cur).tolist(),
    "ncd_years":   rng.integers(0, 9, n_cur).tolist(),
})

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'RECALIBRATE' | 'REFIT' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

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

If you just want to run a quick sanity check without feature data:

```python
import numpy as np
from insurance_monitoring import MonitoringReport

rng = np.random.default_rng(42)
pred_ref = rng.uniform(0.05, 0.20, 10_000)
act_ref = rng.poisson(pred_ref).astype(float)
pred_cur = rng.uniform(0.05, 0.20, 4_000)
act_cur = rng.poisson(pred_cur * 1.08).astype(float)

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    murphy_distribution="poisson",
)
print(report.recommendation)
```

## Worked Example

[`model_drift_monitoring.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/model_drift_monitoring.py) demonstrates the full monitoring stack on a synthetic motor book with three deliberately induced failure modes: covariate shift (older driver mix), calibration deterioration (segment-level A/E drift), and discriminatory power loss (Gini decay). It covers exposure-weighted PSI and CSI, segment A/E ratios with Poisson confidence intervals, the Gini drift z-test, and structured governance reporting suitable for inclusion in PRA SS3/17 model risk documentation.

A Databricks-importable version is also available: [Databricks notebook](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py).


## Modules

### `calibration` — A/E ratio, calibration suite, Murphy decomposition

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

**On exposure-weighted PSI**: standard PSI treats every policy equally regardless of how long it was on risk. If your book renews quarterly and mixes 1-month and 12-month policies, unweighted PSI is wrong. The `exposure_weights` parameter weights bin proportions by earned exposure.

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

The Gini drift test is the distinguishing feature of this library. Most monitoring tools tell you whether A/E has moved. This tells you whether the model's *ranking* has degraded — the difference between a cheap recalibration and a full refit.

### `sequential` — Anytime-valid champion/challenger testing (v0.5.0)

Standard A/B tests have a dirty secret: if you peek at results before the pre-specified end date and stop early when the data looks good, your actual false positive rate is far higher than your nominal 5%. On a monthly checking cadence, a fixed-horizon t-test can inflate to 25% FPR — five times nominal.

The `sequential` module implements the mixture Sequential Probability Ratio Test (mSPRT) from Johari et al. (2022). The test statistic is an e-process, which satisfies P(exists n: Lambda_n >= 1/alpha) <= alpha at **all** stopping times. You can check it every month for two years without inflating type I error. When it crosses the threshold, stop — the evidence is genuine.

Supports three metrics: claim frequency (Poisson rate ratio), claim severity (log-normal ratio), and combined loss ratio (product of the two e-values).

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

# Champion: existing model. Challenger: new model being tested.
# Feed monthly increments as they arrive — no need to wait for a fixed end date.

test = SequentialTest(
    metric="frequency",      # 'frequency' | 'severity' | 'loss_ratio'
    alternative="two_sided", # 'two_sided' | 'greater' | 'less'
    alpha=0.05,
    tau=0.03,                # prior std dev on log-rate-ratio: expect ~3% effects
    max_duration_years=2.0,
    min_exposure_per_arm=100.0,  # car-years before any stopping decision
)

# Month 1 — Q1 2025
result = test.update(
    champion_claims=42,   challenger_claims=38,
    champion_exposure=500, challenger_exposure=495,
    calendar_date=datetime.date(2025, 3, 31),
)
print(result.summary)
# "Challenger freq 8.8% lower (95% CS: 0.731–1.193). Evidence: 0.4 (threshold 20.0). Inconclusive."

# Month 4 — Q2 2025 (check as often as you like; FPR stays at 5%)
result = test.update(
    champion_claims=44,   challenger_claims=29,
    champion_exposure=510, challenger_exposure=505,
    calendar_date=datetime.date(2025, 6, 30),
)
print(result.decision)     # 'inconclusive' | 'reject_H0' | 'futility' | 'max_duration_reached'
print(result.should_stop)  # True when decision != 'inconclusive'

# Full history as Polars DataFrame
df = test.history()
# period_index | calendar_date | lambda_value | log_lambda_value | champion_rate | ...
```

For batch processing from a DataFrame of monthly reporting periods:

```python
import polars as pl
from insurance_monitoring.sequential import sequential_test_from_df

monthly_data = pl.DataFrame({
    "date":               ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"],
    "champ_claims":       [42, 38, 44, 41],
    "champ_exposure":     [500, 490, 510, 495],
    "chall_claims":       [38, 31, 29, 28],
    "chall_exposure":     [495, 488, 505, 492],
})

result = sequential_test_from_df(
    df=monthly_data,
    champion_claims_col="champ_claims",
    champion_exposure_col="champ_exposure",
    challenger_claims_col="chall_claims",
    challenger_exposure_col="chall_exposure",
    date_col="date",
    metric="frequency",
    alpha=0.05,
)
print(result.summary)
```

**When to use:** Any champion/challenger experiment where results are checked before the pre-specified end date — which is almost every experiment in practice. Renewal cycles, rate change pilots, telematics scoring experiments.

**When NOT to use:** When you have a hard commitment to a fixed sample size and will genuinely not look before it completes. In that case, a standard two-sample test is more powerful than mSPRT.

**On the prior tau:** `tau=0.03` encodes a prior that meaningful effects are around 3% on the log-rate-ratio scale. For telematics experiments where you expect larger effects (10%+), increase to `tau=0.10`. For fine-tuning experiments where the effect is expected to be very small, decrease to `tau=0.01`.

### `PITMonitor` — Anytime-valid calibration change detection (v0.7.0)

The Hosmer-Lemeshow test was designed for a single holdout evaluation. Applying it monthly in production is a repeated-testing problem: with 12 monthly checks at alpha=0.05, the probability of a false alarm from a perfectly calibrated model reaches 46%. After two years: 71%.

`PITMonitor` constructs a mixture e-process over probability integral transforms (PITs) from Henzi, Murph, Ziegel (2025, arXiv:2603.13156). The formal guarantee is P(ever alarm | model calibrated) <= alpha, at any checking frequency, forever. You can check it after every renewal batch, every week, or every policy without correction.

This is distinct from `CalibrationChecker`, which tests absolute calibration on a fixed holdout. `PITMonitor` detects *changes* in calibration — a consistently biased model will not trigger. Use `CalibrationChecker` at model launch; use `PITMonitor` once deployed.

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson

monitor = PITMonitor(alpha=0.05, n_bins=100, rng=42)

# Process one policy at a time as renewals come in
for row in live_claims_stream:
    mu = row.exposure * row.lambda_hat
    pit = float(poisson.cdf(row.claims, mu))  # F_hat(y | x)
    alarm = monitor.update(pit)
    if alarm:
        print(f"Calibration drift detected at t={alarm.time}")
        print(f"Estimated changepoint: t~{alarm.changepoint}")
        break

# Snapshot the current state
summary = monitor.summary()
# summary.alarm_triggered   — bool
# summary.evidence          — current M_t value
# summary.threshold         — 1/alpha (alarm fires when M_t >= threshold)
# summary.changepoint       — estimated step when drift began
# summary.calibration_score — 1 - KS statistic (continuous health metric)
```

For batch loading of historical PITs before live monitoring begins:

```python
# Warm start: pre-load 12 months of historical PITs
# This builds the density estimator without accumulating evidence.
# Subsequent updates start the e-process from zero — epistemically honest.
monitor.warm_start(historical_pits)

# Persist and restore state between monitoring runs
monitor.save("pit_monitor_q1_2026.json")
monitor_restored = PITMonitor.load("pit_monitor_q1_2026.json")
```

**PIT computation for common GLM families:**

```python
from scipy.stats import poisson, gamma, nbinom, norm

# Poisson frequency
pit = float(poisson.cdf(y_claims, mu=exposure * lambda_hat))

# Gamma severity (shape=1/phi, scale=phi*mu)
pit = float(gamma.cdf(y_loss, a=1/phi, scale=phi*mu_hat))

# Negative Binomial
pit = float(nbinom.cdf(y_claims, n=r, p=r/(r+mu)))
```

**When to use:** Any deployed pricing model checked on a recurring schedule — monthly renewals, weekly batch scoring, or per-policy online monitoring. The guarantee holds regardless of how often you check.

**When NOT to use:** For absolute calibration checks at model sign-off (use `CalibrationChecker`). For champion/challenger A/B tests (use `SequentialTest`).

### `InterpretableDriftDetector` — Feature-attributed drift with FDR control (v0.7.0)

PSI and A/E tell you *that* drift occurred. `InterpretableDriftDetector` tells you *which* features are responsible. It implements TRIPODD (Panda et al. 2025, arXiv:2503.06606) with seven substantive improvements over the earlier `DriftAttributor` in this package.

The core idea: measure how much each feature's marginal contribution to model loss has changed between the reference and monitoring windows. Features whose contribution shifted significantly are attributed as drift sources. For interactions (vehicle_age × telematics_score), the method detects pairs whose joint contribution changed even when their marginals are stable.

```python
from insurance_monitoring import InterpretableDriftDetector

detector = InterpretableDriftDetector(
    model=fitted_glm,                  # any object with .predict(X) -> np.ndarray
    features=["driver_age", "vehicle_age", "ncb", "annual_mileage", "area"],
    alpha=0.05,
    loss="poisson_deviance",           # canonical GLM goodness-of-fit for frequency models
    n_bootstrap=200,
    error_control="fdr",              # Benjamini-Hochberg: more powerful than Bonferroni for d>=5
    exposure_weighted=True,
    random_state=42,
)

# Reference window: typically the model's training or validation data
detector.fit_reference(X_ref, y_ref_claims, weights=exposure_ref)

# Monitoring window: current quarter's new business
result = detector.test(X_new, y_new_claims, weights=exposure_new)

print(result.drift_detected)         # True / False
print(result.attributed_features)    # ['vehicle_age', 'area']
print(result.summary())              # governance-ready paragraph

# Per-feature table: test_statistic, threshold, p_value, drift_attributed, rank
df = result.feature_ranking
```

**What it adds over `DriftAttributor`:**

- **Exposure weighting** — correct for mixed policy terms. An unweighted mean treats a 0.25-year policy and a 1.0-year policy as equal; exposure weighting gives the population-level picture.
- **Poisson deviance loss** — MSE is not appropriate for count data. Poisson deviance is scale-invariant to exposure and is the canonical GLM goodness-of-fit.
- **FDR control (Benjamini-Hochberg)** — with d=10 rating factors, Bonferroni gives effective per-test alpha=0.005. BH controls the false discovery rate at alpha=0.05 while being substantially more powerful. Use `error_control='fdr'` for d >= 5.
- **Single bootstrap loop** — thresholds and p-values computed in one pass. Halved computational cost over `DriftAttributor`.
- **Subset risk caching** — reference-side model calls pre-computed at `fit_reference()`. Subsequent `test()` calls are faster.
- **Explicit `update_reference()`** — no auto-retrain on drift detection. Retraining requires external governance sign-off.

**Convenience method for one-off quarterly checks:**

```python
result = InterpretableDriftDetector.from_dataframe(
    model=fitted_glm,
    df_ref=df_reference,
    df_new=df_monitoring,
    target_col="claim_count",
    feature_cols=["driver_age", "vehicle_age", "ncb", "annual_mileage", "area"],
    weight_col="exposure",
    loss="poisson_deviance",
    error_control="fdr",
    n_bootstrap=200,
)
```

**When to use:** Quarterly model reviews where you need to explain *why* performance has drifted — not just that it has. The feature-level attribution is the right artefact for a model governance pack. Use FDR control (`error_control='fdr'`) when you have five or more rating factors.

**When to use `DriftAttributor` instead:** Online/streaming use cases where you need to detect drift and trigger an automated retrain pipeline. `DriftAttributor` has the simpler API for that workflow.

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
    murphy_distribution="poisson",
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

### `thresholds` — Configurable traffic lights

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

## Background

The monitoring framework implements:
> "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing", arXiv 2510.04556 (December 2025)

The calibration suite implements:
> Lindholm & Wüthrich: "Three calibration properties for insurance pricing models" (SAJ 2025)
> Brauer et al.: arXiv:2510.04556 Section 4 — Murphy decomposition and the MCB bootstrap test

The sequential testing module implements:
> Johari et al. (2022). "Always Valid Inference: Continuous Monitoring of A/B Tests." Operations Research 70(3). arXiv:1512.04922.
> Howard et al. (2021). "Time-uniform, nonparametric, nonasymptotic confidence sequences." Annals of Statistics 49(2).

The PITMonitor implements:
> Henzi, Murph, Ziegel (2025). "Anytime valid change detection for calibration." arXiv:2603.13156.

The InterpretableDriftDetector implements:
> Panda, Srinivas, Balasubramanian & Sinha (2025). "TRIPODD: Feature-Interaction-Aware Drift Detection with Type I Error Control." arXiv:2503.06606.
> Benjamini & Hochberg (1995). "Controlling the False Discovery Rate." Journal of the Royal Statistical Society B, 57(1), 289–300.

---

## Capabilities Demo

Demonstrated on synthetic UK motor data with three deliberately induced failure modes: covariate shift (older drivers enter the book), calibration deterioration (claim frequency inflated for a segment), and stale discrimination (model trained on old data, portfolio composition changed). Full notebook: `notebooks/benchmark.py`.

- PSI/CSI flags the covariate shift — feature distributions in the monitoring period diverge from training, triggering configurable traffic lights (PSI > 0.25 = red)
- A/E ratio with confidence intervals catches calibration drift — segment-level actual-to-expected ratios with statistical significance tests, not just point estimates
- Gini drift z-test (arXiv 2510.04556) detects discrimination loss — the discriminatory power of the model has declined, which a standard A/E dashboard would miss
- `MonitoringReport` assembles all three checks into a single traffic-light summary with a recommended action: monitor, investigate, or refit

**When to use:** Any time more than a month has passed since the last model refit. A typical UK motor pricing cycle is 6–12 months between refits; covariate shift and calibration drift accumulate silently in between. Run the monitoring report monthly on the live book.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py).

## Related Libraries

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — use alongside monitoring to flag when interval coverage degrades |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS3/17 (insurer) / SS1/23 (bank) model governance — monitoring evidence feeds into governance review cycles |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger deployment — monitoring informs when to switch challenger to champion |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation — produces the baseline metrics that monitoring tracks prospectively |
| [insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift) | Covariate shift detection and correction — use when monitoring flags PSI drift requiring model adaptation |

## Performance

### MonitoringReport vs manual A/E

Benchmarked against a **manual aggregate A/E ratio check** on synthetic UK motor insurance data — 10,000 reference policies and 4,000 monitoring-period policies with three deliberately induced failure modes. Full script: `benchmarks/run_benchmark.py`.

| Check | Manual A/E | MonitoringReport |
|-------|-----------|-----------------|
| Reference A/E | 0.9624 | 0.9624 |
| Monitoring A/E | 0.9420 | 0.9420 |
| Manual verdict | INVESTIGATE | REFIT |
| Covariate shift (driver_age PSI = 0.21) | Not detected | AMBER |
| Calibration drift (new vehicles) | Not detected | Detected (Murphy) |
| Discrimination decay (30% predictions randomised) | Not detected | REFIT |
| Gini change | Not computed | −0.012 |
| Gini drift p-value | N/A | 0.76 (n=4,000 — underpowered) |
| Murphy discrimination | Not computed | REFIT flag |
| Murphy local MCB | Not computed | 0.0090 (REFIT) |

The aggregate A/E at 0.9420 falls just outside the 0.95–1.05 green band (verdict: INVESTIGATE), but it is blind to which segment is causing the drift and why. MonitoringReport identifies all three failure modes:

1. **Covariate shift**: driver_age PSI = 0.21 (AMBER). Young drivers (18–30) are oversampled 2x in the monitoring period.
2. **Discrimination decay**: Murphy decomposition flags REFIT — the local MCB (0.0090) exceeds global MCB (0.0002), meaning the model's ranking is broken, not just the scale.
3. **Calibration drift**: detected via the Murphy miscalibration component.

The Gini drift test returns p=0.76 at n=4,000, which is correct — 4,000 policies does not give enough statistical power to detect a Gini drop of −0.012. At 15,000 policies the same DGP produces z≈−1.9, p≈0.06. The test is appropriately conservative at small sample sizes.

**When to use:** Any time more than a month has passed since the last model refit. The monitoring report runs in under 40 seconds on 14,000 policies (including bootstrap variance estimation for the Gini test).

### Time-to-detection: aggregate A/E vs PSI alarm

The cross-sectional comparison above shows detection capability at a fixed snapshot. The more operationally relevant question is: if we watch policies accumulate month by month, which approach raises the alarm first?

The benchmark script (`benchmarks/benchmark.py`) simulates this by walking through the monitoring cohort in 500-policy batches. In the 50,000/15,000 scenario (2x young driver oversampling, 25% new-vehicle claims inflation):

- **Aggregate A/E (5% breach threshold):** In this scenario, the covariate shift and calibration drift partially cancel at portfolio level. The aggregate A/E stays within the 0.95–1.05 band across the entire monitoring period — it never fires.
- **PSI driver_age (RED threshold, PSI > 0.25):** Fires at approximately 1,000–1,500 policies — roughly 1 month into the monitoring period on a 1,250-policy/month book.

The aggregate A/E would not have detected this shift at all. PSI detected it within the first monthly batch.

This is the central operational argument for PSI monitoring: calibration drift that is self-cancelling at portfolio level (cheap on young drivers, expensive on old ones) is invisible to A/E for as long as the errors balance. PSI per feature is not fooled by this — it measures the compositional shift directly, before any claims are observed. Run `python benchmarks/benchmark.py` to see the time-to-detection output for the current DGP parameters.

### SequentialTest (mSPRT) vs fixed-horizon t-test

Benchmarked on simulated UK motor champion/challenger data. 10,000 Monte Carlo simulations under H0 (no true effect): analyst checks results monthly for 24 months and stops if the test is significant. Full script: `benchmarks/benchmark_sequential.py`.

| Method | Nominal FPR | Actual FPR (monthly peeking) | Notes |
|--------|-------------|------------------------------|-------|
| Fixed-horizon t-test | 5% | ~25% | 5x inflation from repeated peeking |
| mSPRT (SequentialTest) | 5% | ~1% | Valid at all stopping times |

Under H1 (challenger 10% cheaper on frequency), mSPRT detects the effect in a median of 8 months on a 500-policy-per-arm-per-month book; a pre-registered t-test at 24 months would reach the same conclusion but forces the team to wait.

The 25% FPR figure for the fixed-horizon t-test assumes monthly checks from month 1 with early stopping on significance — the common practice of "we'll check again next month to see if it's still significant." If the analyst genuinely never looks before month 24, the t-test is valid; in practice, nobody does this.

### PITMonitor vs repeated Hosmer-Lemeshow testing

Benchmarked on a simulated Poisson frequency model: 500 well-calibrated observations followed by 500 observations with a 15% rate inflation (model does not adjust). Hosmer-Lemeshow checked every 50 new observations; PITMonitor updated per observation. Full script: `benchmarks/benchmark_pit.py`.

| Method | Nominal FPR | Empirical FPR (phase 1) | FPR inflation | Detects phase-2 drift |
|--------|-------------|------------------------|--------------|----------------------|
| H-L repeated (every 50 obs) | 5% | ~46% (10 looks) | 9x | Yes, with prior false alarms |
| PITMonitor | 5% | ~3% (300 simulations) | 0.6x | Yes, no false alarms |

The key finding is not just the FPR inflation — it is the false alarm pattern. Repeated H-L raises alarms throughout the stable phase, causing teams to investigate non-existent problems and ultimately to distrust the monitoring system. PITMonitor's e-process stays near zero when the model is calibrated and rises sharply only when calibration genuinely shifts.

The benchmark also shows changepoint estimation: when PITMonitor fires, the Bayes factor scan over the evidence history recovers the true drift onset (t~500) within ±30 steps on typical runs.

**When to use:** Any deployed model checked on a recurring schedule — monthly renewals, weekly batch processing, or per-policy online monitoring. The formal guarantee holds regardless of checking frequency.

### InterpretableDriftDetector vs DriftAttributor

Benchmarked on a 5-feature Poisson pricing model with drift planted in exactly two features (vehicle_age and area). Reference: 10,000 policies; monitoring: 5,000 policies with mixed policy terms (50% short-term). Full script: `benchmarks/benchmark_interpretable_drift.py`.

| Check | DriftAttributor | InterpretableDriftDetector (BH) |
|-------|----------------|--------------------------------|
| vehicle_age flagged | Yes | Yes |
| area flagged | Yes | Yes |
| False positives | 0 | 0 |
| Attribution correct | Yes | Yes |
| Exposure weighting | No | Yes |
| Loss function | MSE | Poisson deviance |
| Error control | Bonferroni | Benjamini-Hochberg |

Both modules correctly identify the two drifted features on this scenario. The differences become material at larger feature counts and with mixed portfolio terms.

With d=10 rating factors, Bonferroni gives effective per-test alpha=0.005. BH gives per-test alpha=0.01 for the rank-1 feature — materially more power on the features most likely to be drifting. At d=20, the power difference is substantial enough that the correct choice is almost always FDR control.

Exposure weighting changes the result when monitoring has different policy-term composition than the reference. In this benchmark, the monitoring cohort is 50% short-term policies versus 30% in the reference. Unweighted analysis assigns too much weight to short-tenure policies and mis-estimates the population-level drift magnitude.

**When to use `InterpretableDriftDetector`:** Quarterly model reviews needing a defensible governance artefact — which features drifted, with what statistical confidence, and under what error control. Use `error_control='fdr'` when d >= 5.

**When to use `DriftAttributor`:** Automated monitoring pipelines where drift detection triggers an immediate action (retrain, alert). The simpler API is more appropriate for that workflow.


## Licence

MIT
