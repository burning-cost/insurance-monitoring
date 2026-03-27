# Changelog

## v0.9.4 (2026-03-27)

Documentation fixes — first-run UX audit.

Four README code examples had method names or parameter names that diverged from
the actual implementation. A new user copying any of these examples would have
hit an immediate AttributeError or TypeError with no clear error message.

**Bugs fixed:**

1. `CalibrationChecker.run()` → `CalibrationChecker.check()`. The method has
   always been `check()`. `run()` does not exist on this class. Example in the
   `calibration` module section of the README was wrong.

2. `PITMonitor(alpha=0.05, distribution="poisson")` — `distribution` is not a
   constructor parameter. `PITMonitor` accepts pre-computed probability integral
   transforms, not raw actuals and predictions. The distribution-specific CDF
   (e.g. `scipy.stats.poisson.cdf`) is applied by the caller before calling
   `monitor.update(pit)`. The README example replaced the wrong constructor call
   and the fictional `monitor.fit(train_actual, train_predicted)` with a correct
   per-observation loop using `scipy.stats.poisson`.

3. `alarm.e_value` → `alarm.evidence`. `PITAlarm` stores the e-process value in
   the `evidence` field, not `e_value`.

4. `SequentialTest.set_null(reference_rate=0.08)` — this method does not exist.
   The README example was entirely wrong for `SequentialTest`: it showed
   `test.update(batch_actual, batch_predicted, exposure=batch_exposure)` but the
   actual signature is
   `update(champion_claims, champion_exposure, challenger_claims, challenger_exposure)`.
   The corrected example shows the correct positional parameters. `result.e_value`
   and `result.stopped` are also wrong; corrected to `result.lambda_value` and
   `result.should_stop`.

5. `bt_result.summary()` → `bt.summary()`. `summary()` is a method on
   `GiniDriftBootstrapTest` (the test class), not on `GiniBootstrapResult` (the
   object returned by `bt.test()`). The README called it on the result object.

No source code changes. All bugs were documentation-only. No new dependencies.


## v0.9.3 (2026-03-26)

`MulticalibrationMonitor` added — subgroup-level calibration monitoring.

Reference: Denuit, Michaelides & Trufin (2026), arXiv:2603.16317.


## v0.9.2 (2026-03-25)

`gini_drift` module added — `GiniDriftTest`: class-based two-sample asymptotic z-test
for Gini coefficient drift (Wüthrich, Merz & Noll 2025, arXiv:2510.04556).

Answers the question model validation teams need answered: has our model's ranking power
changed between the reference period and the monitoring period? The test uses the asymptotic
distribution of the Gini coefficient established in Theorem 1 of arXiv:2510.04556.
Variance is estimated by non-parametric bootstrap (Algorithm 2) for both samples
independently, then combined under the independence assumption.

New API:

```python
from insurance_monitoring.gini_drift import GiniDriftTest
# or: from insurance_monitoring import GiniDriftTest

test = GiniDriftTest(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    monitor_actual=act_mon,
    monitor_predicted=pred_mon,
    n_bootstrap=200,
    alpha=0.32,  # one-sigma rule per arXiv:2510.04556
    random_state=42,
)
result = test.test()
print(f"Delta: {result.delta:+.4f}")
print(f"z = {result.z_statistic:.2f}, p = {result.p_value:.4f}")
print(f"Drift detected: {result.significant}")
print(test.summary())
```

- `GiniDriftTest`: class-based wrapper with lazy evaluation, caching, exposure weighting
- `GiniDriftTestResult`: typed dataclass with gini_reference, gini_monitor, delta,
  z_statistic, p_value, significant, se_reference, se_monitor, n_reference, n_monitor.
  Supports dict-style access for backward compatibility.
- `summary()`: governance-ready plain-text report for model validation sign-off

Design: independent bootstrap seeds for reference and monitor SE estimates.
Large-sample cap (n > 20,000): variance estimated on subsample of 20k, point
estimate uses full sample. No new dependencies.


## v0.9.1 (2026-03-25)

`business_value` module added — Loss Ratio Error (LRE) metric from Evans Hedges (2025), arXiv:2512.03242.

Converts Pearson correlation rho into expected portfolio loss ratio impact. Answers the
question pricing teams actually care about: if we improve our model from rho=0.92 to
rho=0.95, how many basis points of loss ratio improvement should we expect?

New API:

```python
from insurance_monitoring.business_value import lre_compare, loss_ratio_error, loss_ratio, calibrate_eta

# How much is a model improvement worth?
result = lre_compare(rho_old=0.92, rho_new=0.95, cv=1.2, eta=1.5)
print(f"Improvement: {result.delta_lr_bps:.1f} bps")
# On a £100M book: £100M * abs(result.delta_lr) = annual value

# What is the current model costing us?
e_lr = loss_ratio_error(rho=0.92, cv=1.2, eta=1.5)
print(f"Current model adds {e_lr * 10000:.0f} bps to our loss ratio vs perfect pricing")

# Recover implied elasticity from historical observed LR
eta_implied = calibrate_eta(rho_observed=0.91, cv=1.2, lr_observed=0.76, margin=1/0.70)
```

- `loss_ratio_error(rho, cv, eta)` — E_LR from Definition 2. Zero for perfect model.
- `loss_ratio(rho, cv, eta, margin=1.0)` — Theorem 1 formula.
- `lre_compare(rho_old, rho_new, cv, eta, margin=1.0)` — returns `LREResult` with
  lr_old, lr_new, delta_lr, delta_lr_bps, e_lr_old, e_lr_new.
- `calibrate_eta(rho_observed, cv, lr_observed, margin=1.0)` — brentq inversion of
  Theorem 1 to recover implied demand elasticity. Returns None if unsolvable.
- `LREResult` dataclass for structured output.

No new dependencies. scipy.optimize.brentq (already a core dependency) used for calibrate_eta.

## v0.9.0 (2026-03-24)

`MonitoringTracker` added — MLflow model registry integration.

Attaches insurance-monitoring results to registered MLflow models. Each `log_*` call
creates an MLflow run under a `monitoring/<model_name>` experiment, logging scalar
metrics for time-series tracking and a JSON artifact for full audit detail.

```python
from insurance_monitoring.mlflow_tracker import MonitoringTracker

tracker = MonitoringTracker("motor_freq_glm", model_version="3", tracking_uri="databricks")
tracker.log_ae_ratios({"overall": 1.04, "young_drivers": 0.97})
tracker.log_psi({"vehicle_age": 0.06, "ncb": 0.14})
tracker.log_gini_drift(gini_result)

history = tracker.get_monitoring_history()  # pandas DataFrame
```

MLflow is an optional dependency: `pip install insurance-monitoring[mlflow]`.
The rest of the library is unaffected if mlflow is not installed.

## v0.8.2 (2026-03-23)
- fix: suppress RuntimeWarning in `poisson_deviance` when y=0 — `np.where` evaluates both branches before masking, so `log(0)` and `log(0/mu)` were computed (and warned about) on every real dataset with zero-claim observations. Wrapped in `np.errstate(divide='ignore', invalid='ignore')`. The mathematical result is unchanged; the 0*log(0)=0 convention was already correct.

## v0.7.0

`PITMonitor`, `PITAlarm`, and `PITSummary` added to `insurance_monitoring.calibration`.
Anytime-valid calibration change detection via probability integral transforms and
mixture e-processes. Reference: Henzi, Murph, Ziegel (2025), arXiv:2603.13156.

**Guarantee:** `P(ever alarm | model calibrated) <= alpha`, for all t, forever.
Solves the repeated-testing inflation in monthly H-L / A/E monitoring.

New features vs. existing `CalibrationChecker`:

- Sequential: processes one observation at a time, suitable for live deployment
- Anytime-valid: no correction needed for repeated checks at any frequency
- Changepoint estimation: identifies when calibration degraded (Bayes factor scan, O(T) time)
- Warm-start: pre-load historical PITs to avoid cold-start sensitivity loss
- Exposure weighting: integer-repetition weighting preserves formal guarantee

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson

monitor = PITMonitor(alpha=0.05, rng=42)

for claims_row in live_data:
    mu = claims_row.exposure * claims_row.lambda_hat
    alarm = monitor.update(float(poisson.cdf(claims_row.claims, mu)))
    if alarm:
        print(f"Alarm at step {alarm.time}, changepoint ~{alarm.changepoint}")
        break
```

New dependency: `sortedcontainers>=2.4` (pure Python, zero transitive dependencies).

---

## v0.3.0

`insurance-calibration` was a separate package covering the three-property calibration framework (Lindholm & Wüthrich 2025) and Murphy decomposition. As of v0.3.0 it is absorbed into `insurance-monitoring` as the `calibration` sub-package.

All existing imports are unchanged. New imports follow the same pattern:

```python
# These already worked:
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# These are new in v0.3.0 (previously required separate install of insurance-calibration):
from insurance_monitoring.calibration import (
    check_balance, check_auto_calibration, murphy_decomposition,
    rectify_balance, isotonic_recalibrate, CalibrationChecker,
    BalanceResult, AutoCalibResult, MurphyResult, CalibrationReport,
    deviance, poisson_deviance, gamma_deviance,
    plot_auto_calibration, plot_murphy, plot_calibration_report,
)
```

**Migration**: uninstall `insurance-calibration` and update your imports to use `insurance_monitoring.calibration`. The function signatures are identical.
