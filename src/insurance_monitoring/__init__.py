"""
insurance-monitoring: model drift detection and monitoring for insurance pricing.

Modules
-------
drift
    PSI, CSI, KS test, Wasserstein distance for feature distribution monitoring.
calibration
    A/E ratio, Hosmer-Lemeshow, calibration curves for model calibration checks.
discrimination
    Gini coefficient, Gini drift test (arXiv 2510.04556), Lorenz curves.
report
    MonitoringReport — combined check with traffic-light output.
thresholds
    Configurable threshold defaults (PSI, A/E, Gini).

Quick start
-----------
::

    from insurance_monitoring import MonitoringReport
    from insurance_monitoring.drift import psi
    from insurance_monitoring.calibration import ae_ratio
    from insurance_monitoring.discrimination import gini_coefficient
"""

from insurance_monitoring.calibration import (
    ae_ratio,
    ae_ratio_ci,
    calibration_curve,
    hosmer_lemeshow,
)
from insurance_monitoring.discrimination import (
    gini_coefficient,
    gini_drift_test,
    lorenz_curve,
)
from insurance_monitoring.drift import (
    csi,
    ks_test,
    psi,
    wasserstein_distance,
)
from insurance_monitoring.report import MonitoringReport
from insurance_monitoring.thresholds import (
    AERatioThresholds,
    GiniDriftThresholds,
    MonitoringThresholds,
    PSIThresholds,
)

__version__ = "0.1.0"

__all__ = [
    # drift
    "psi",
    "csi",
    "ks_test",
    "wasserstein_distance",
    # calibration
    "ae_ratio",
    "ae_ratio_ci",
    "calibration_curve",
    "hosmer_lemeshow",
    # discrimination
    "gini_coefficient",
    "gini_drift_test",
    "lorenz_curve",
    # report
    "MonitoringReport",
    # thresholds
    "MonitoringThresholds",
    "PSIThresholds",
    "AERatioThresholds",
    "GiniDriftThresholds",
]
