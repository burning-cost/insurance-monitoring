# Changelog

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
