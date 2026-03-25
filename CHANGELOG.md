# Changelog

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
