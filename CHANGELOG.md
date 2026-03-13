# Changelog

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
