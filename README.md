# insurance-monitoring

[![PyPI](https://img.shields.io/pypi/v/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-monitoring)](https://pypi.org/project/insurance-monitoring/)
[![Tests](https://github.com/burning-cost/insurance-monitoring/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-monitoring/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-monitoring/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-monitoring/blob/main/notebooks/quickstart.ipynb)

**Production model monitoring for UK insurance pricing — PSI, Gini drift, Murphy decomposition, and anytime-valid A/B testing in one package.**

---

## The problem

A pricing model that is 15% cheap on young drivers and 15% expensive on mature drivers reads 1.00 at portfolio level — and triggers no alarm. The loss ratio deteriorates twelve months later.

The PRA expects regulated insurers to have model risk management frameworks (SS1/23, while primarily aimed at banks, is widely adopted as the de facto standard for insurers) — and that framework should have caught this first. Standard monitoring approaches fail in three specific ways:

- **Portfolio-level A/E hides segmental drift.** Per-feature distribution shifts and model discrimination drift are invisible to a single headline ratio.
- **Repeated monthly testing inflates false positives.** Running Hosmer-Lemeshow or A/E tests each month means a perfectly calibrated model will trigger a false alarm roughly 40% of the time within a year at α=0.05.
- **Champion/challenger comparisons are run to a pre-specified date.** Checking interim results is statistically invalid under classical hypothesis testing — but pricing teams do it anyway.

`insurance-monitoring` addresses all three. It monitors per-feature distribution shifts (PSI/CSI), discrimination and calibration separately (Murphy decomposition), and runs anytime-valid tests you can check at any point.

---

## Part of the Burning Cost stack

Takes the outputs of any fitted pricing model and a stream of actual experience. Feeds drift signals and calibration verdicts into [insurance-governance](https://github.com/burning-cost/insurance-governance) for model risk committee packs. Pairs with [insurance-fairness](https://github.com/burning-cost/insurance-fairness) to monitor per-protected-group A/E ratios under Consumer Duty. See the [full stack](https://burning-cost.github.io/stack/).

**Blog post:** [Insurance Model Monitoring: Beyond Generic Drift Detection](https://burning-cost.github.io/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)

---

## Quick start

```python
import numpy as np
import polars as pl
from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi

rng = np.random.default_rng(42)
# Training period — well-calibrated model
actual_ref = rng.poisson(0.08, 10_000).astype(float)
predicted_ref = np.full(10_000, 0.08)
# Current period — book has aged, model is stale
actual_cur = rng.poisson(0.11, 5_000).astype(float)
predicted_cur = np.full(5_000, 0.08)
exposure_cur = rng.uniform(0.5, 1.0, 5_000)

report = MonitoringReport(
    reference_actual=actual_ref, reference_predicted=predicted_ref,
    current_actual=actual_cur, current_predicted=predicted_cur,
    exposure=exposure_cur, murphy_distribution="poisson",
)
print(report.recommendation)   # => 'RECALIBRATE' or 'REFIT'
print(report.to_polars())      # flat DataFrame: metric / value / band
```

See `examples/` for fully worked scenarios: `model_monitor_quickstart.py` (the v1.0.0 three-way decision), UK motor frequency monitoring, home insurance with PITMonitor, and a champion/challenger A/B test.

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
pip install insurance-monitoring[mlflow]
```

---

## What manual monitoring looks like vs this

| Task | Manual approach | insurance-monitoring |
|------|----------------|----------------------|
| Population Stability Index | Excel macro per factor, re-coded each quarter, unweighted | `psi()` / `csi()` — exposure-weighted, Polars-native, traffic-light band |
| Feature drift heatmap | Engineer writes one-off script; no standard thresholds | `csi()` — one call, all rating factors, PRA-aligned thresholds |
| A/E ratio with CI | Custom formula in SQL, no confidence interval | `ae_ratio_ci()` — Wilson CI, exposure-weighted, RAG status |
| Discrimination drift | Gini computed ad hoc; no test for statistical significance | `gini_drift_test()` / `GiniDriftBootstrapTest` — bootstrap CI, governance plot |
| RECALIBRATE vs REFIT decision | Actuary judgment call, not documented | `ModelMonitor` — Gini + GMCB + LMCB bootstrap tests (arXiv 2510.04556), structured three-way decision |
| Repeated monthly testing inflation | H-L / A/E tested afresh each month; inflated false-positive rate | `PITMonitor` — anytime-valid, P(ever false alarm \| calibrated) ≤ α, forever |
| Champion/challenger A/B test | Wait for pre-specified sample size; cannot stop early | `SequentialTest` — mSPRT, valid at every interim check, supports frequency/severity/loss ratio |
| Drift attribution (which feature explains the performance change?) | PSI-by-eye; no interaction-aware method | `DriftAttributor` / `InterpretableDriftDetector` — TRIPODD, FDR control, exposure weighting |
| Monitoring report for model risk committee | Manual Word document | `MonitoringReport.to_polars()` — flat DataFrame, writes directly to Delta table |

---

## Features

### ModelMonitor — structured three-way decision (v1.0.0)

`ModelMonitor` is the recommended entry point for teams running quarterly or monthly model reviews. It implements the full two-step procedure from Brauer, Menzel & Wüthrich (2025), arXiv:2510.04556.

```python
import numpy as np
from insurance_monitoring import ModelMonitor

rng = np.random.default_rng(42)
n = 10_000
exposure = rng.uniform(0.5, 1.5, n)
y_hat = rng.gamma(2, 0.05, n)
y_ref = rng.poisson(exposure * y_hat) / exposure   # reference period

monitor = ModelMonitor(distribution="poisson", n_bootstrap=500, random_state=0)
monitor.fit(y_ref, y_hat, exposure)

# Quarterly monitoring run
y_new = rng.poisson(exposure * y_hat * 1.10) / exposure  # 10% claims trend
result = monitor.test(y_new, y_hat, exposure)

print(result.decision)          # 'REDEPLOY' | 'RECALIBRATE' | 'REFIT'
print(result.balance_factor)    # multiply predictions by this for RECALIBRATE
print(result.summary())         # governance-ready paragraph for MRC packs
print(result.to_dict())         # JSON-serialisable — log to MLflow or Delta
```

Decision rules:

| Test fires | Decision | Action |
|-----------|----------|--------|
| Nothing | REDEPLOY | No action. Re-run next quarter. |
| GMCB only | RECALIBRATE | Multiply predictions by `balance_factor`. Hours of work. |
| Gini or LMCB | REFIT | Rebuild the model. Weeks of work. |

The default `alpha=0.32` (one-sigma rule) is calibrated for ongoing production monitoring. See `examples/model_monitor_quickstart.py` for all three scenarios with realistic synthetic data.

### MonitoringReport — one call for a complete monitoring run

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

### PSI / CSI — feature distribution monitoring

```python
from insurance_monitoring.drift import psi, csi

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

### Calibration — A/E, Murphy decomposition, anytime-valid PITMonitor

```python
from insurance_monitoring import ae_ratio_ci, murphy_decomposition, PITMonitor
from scipy.stats import poisson

# A/E ratio with Wilson CI
result = ae_ratio_ci(actual, predicted, exposure=exposure)
print(f"A/E = {result['ae']:.3f}  95% CI [{result['lower']:.3f}, {result['upper']:.3f}]")

# Murphy decomposition — distinguishes miscalibration from discrimination loss
murphy = murphy_decomposition(y=actual, y_hat=predicted, exposure=exposure, distribution="poisson")
print(f"DSC (discrimination score): {murphy.discrimination:.4f}")
print(f"MCB (miscalibration):       {murphy.miscalibration:.4f}")
print(f"Verdict: {murphy.verdict}")   # 'RECALIBRATE' or 'REFIT'
```

**PITMonitor** — the standard pattern of running A/E tests monthly inflates the false-positive rate. After twelve months at α=0.05, the chance of a false alarm exceeds 40% even on a perfectly calibrated model. `PITMonitor` uses probability integral transform e-processes (Henzi, Murph, Ziegel 2025); the guarantee is P(ever raise an alarm | model calibrated) ≤ α, for all t, forever.

```python
monitor = PITMonitor(alpha=0.05)
for row in live_data:
    mu = row.exposure * row.lambda_hat
    pit = float(poisson.cdf(row.claims, mu))
    alarm = monitor.update(pit)
    if alarm.triggered:
        print(f"Calibration alarm: evidence = {alarm.evidence:.2f}")
        break
```

### Discrimination — Gini drift test

```python
from insurance_monitoring import gini_drift_test_onesample, GiniDriftBootstrapTest

# One-sample design: reference data not required — just the stored training Gini
result = gini_drift_test_onesample(
    training_gini=0.42,
    monitor_actual=current_claims,
    monitor_predicted=current_predicted,
    monitor_exposure=current_exposure,
)
print(f"Gini change: {result.gini_change:+.3f}  p={result.p_value:.3f}  [{result.significant}]")

# Class-based API with governance plot (IFoA/PRA deliverable)
bt = GiniDriftBootstrapTest(training_gini=0.42, monitor_actual=current_claims,
                             monitor_predicted=current_predicted, monitor_exposure=current_exposure)
bt.test()
bt.plot()       # bootstrap histogram with CI shading
bt.summary()    # governance-ready paragraph
```

### Sequential A/B testing — champion/challenger

Champion/challenger pricing experiments are routinely run with a pre-specified end date. Checking early to stop a bad challenger is statistically invalid under classical hypothesis testing. `SequentialTest` uses mixture SPRT (mSPRT, Johari et al. 2022) — check at every interim update with full type I error control.

```python
from insurance_monitoring import SequentialTest

test = SequentialTest(metric="frequency", alpha=0.05)
for batch in monthly_batches:
    result = test.update(
        champion_claims=batch.champion_claims,
        champion_exposure=batch.champion_exposure,
        challenger_claims=batch.challenger_claims,
        challenger_exposure=batch.challenger_exposure,
    )
    print(f"e-value: {result.lambda_value:.2f}  stopped: {result.should_stop}")
    if result.should_stop:
        break
```

### TRIPODD drift attribution

PSI tells you that driver_age has drifted. TRIPODD tells you whether that drift explains why the model's discrimination has fallen — accounting for feature interactions.

```python
from insurance_monitoring import InterpretableDriftDetector

detector = InterpretableDriftDetector(
    error_control="fdr",        # Benjamini-Hochberg for d >= 10 factors
    loss="poisson_deviance",    # canonical for frequency models
)
detector.fit_reference(X_ref, y_ref, weights=exposure_ref)
result = detector.test(X_cur, y_cur, weights=exposure_cur)
print(result.significant_features)  # which factors explain the performance shift
```

---


---


### ScoreDecompositionTest — asymptotic inference on Murphy decomposition components

`ScoreDecompositionTest` decomposes a scoring rule into miscalibration (MCB), discrimination (DSC), and uncertainty (UNC) components with HAC standard errors, so you can formally test whether a model is miscalibrated vs. losing discrimination power — the distinction that drives the RECALIBRATE vs. REFIT decision.

Based on Dimitriadis & Puke (arXiv:2603.04275). Supports MSE, MAE, and quantile scoring rules. For two competing forecasts, the two-sample test has higher power than a plain Diebold-Mariano test when models differ in only one dimension.

```python
import numpy as np
from insurance_monitoring.calibration import ScoreDecompositionTest

rng = np.random.default_rng(42)
n = 5_000

# Synthetic motor frequency: well-discriminating model with a calibration bias
y_hat = rng.gamma(2, 0.05, n)
exposure = rng.uniform(0.5, 1.5, n)
y = rng.poisson(exposure * y_hat * 1.08) / exposure  # 8% bias

sdi = ScoreDecompositionTest(score_type="mse")
result = sdi.fit_single(y, y_hat)
print(result.summary())
# MCB=0 test:   FAIL (miscalibrated)  p=0.0001
# DSC=0 test:   PASS (has skill)      p=0.0000

# Compare champion vs challenger on the same holdout
result_two = sdi.fit_two(y, y_hat, y_hat * 0.95)
print(result_two.summary())
```

---

## Regulatory context

**PRA SS1/23** (Model Risk Management, March 2023) requires insurers to maintain a model monitoring framework that detects deterioration in model performance. The expectation is documented thresholds, regular testing, and a governance process that escalates to the model risk committee when thresholds are breached. A/E ratio and Gini monitoring are the two metrics most commonly cited in SS1/23 supervisory discussions.

**Consumer Duty (PS22/9)** requires ongoing monitoring of whether pricing outcomes are fair across customer groups. The combination of `MonitoringReport` and [insurance-fairness](https://github.com/burning-cost/insurance-fairness) `calibration_by_group()` produces a per-protected-group A/E split suitable for Consumer Duty Outcome 4 monitoring.

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
| `ModelMonitor.fit()` (n_bootstrap=500, n=10k) | 5–15s | Bootstrap Gini SE estimation |
| `ModelMonitor.test()` (n_bootstrap=500, n=5k) | 5–15s | Gini + GMCB + LMCB bootstrap tests |

[Run the full workflow on Databricks](https://github.com/burning-cost/insurance-monitoring/blob/main/notebooks/quickstart.ipynb)

---

## References

**Regulatory instruments**

- PRA. (2023). *Model Risk Management Principles for Banks* (SS1/23). Prudential Regulation Authority, Bank of England. [www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss)
- FCA. (2023). *Consumer Duty: Final rules and guidance* (PS22/9). Financial Conduct Authority. [www.fca.org.uk/publications/policy-statements/ps22-9-new-consumer-duty](https://www.fca.org.uk/publications/policy-statements/ps22-9-new-consumer-duty)

**Population Stability Index and score monitoring**

- Yurdakul, B. (2018). "Statistical Properties of Population Stability Index." University of KwaZulu-Natal working paper. (Derivation of PSI thresholds: <0.10 no change, 0.10–0.25 moderate shift, >0.25 major shift.)
- Hand, D.J. & Anagnostopoulos, C. (2014). "When is the area under the receiver operating characteristic curve an appropriate measure of classifier performance?" *Pattern Recognition Letters*, 42, 128–132. [doi:10.1016/j.patrec.2014.01.015](https://doi.org/10.1016/j.patrec.2014.01.015)

**Forecast scoring and Murphy decomposition**

- Murphy, A.H. (1973). "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology*, 12(4), 595–600. [doi:10.1175/1520-0450(1973)012](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)
- Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association*, 102(477), 359–378. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)

**Sequential testing and drift detection**

- Page, E.S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100–115. [doi:10.2307/2333009](https://doi.org/10.2307/2333009) (CUSUM control charts.)
- Wald, A. (1945). "Sequential Tests of Statistical Hypotheses." *The Annals of Mathematical Statistics*, 16(2), 117–186. [doi:10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118)

---

## Related libraries

| Library | What it does |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Per-protected-group A/E calibration, proxy detection, Consumer Duty audit reports |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double ML — establishes whether a rating factor causally drives risk or is a proxy |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model risk committee packs and FCA Consumer Duty documentation |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Interpretable GLM-style models whose factors can be monitored directly |

---

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-monitoring/discussions). Found it useful? A star helps others find it.

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).
