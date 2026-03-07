"""
Configurable thresholds for insurance model monitoring metrics.

Defaults are based on industry practice from credit scoring (PSI) and
actuarial convention (A/E ratios). Override them per deployment to match
your portfolio size and monitoring frequency.

Notes on calibration:
- PSI thresholds (0.1 / 0.25) originate from FICO credit score monitoring and
  have been adopted wholesale by insurance. They were derived empirically for
  monthly monitoring of large credit portfolios. For quarterly insurance
  monitoring at smaller volumes they may be too sensitive or too lenient
  depending on your portfolio size.
- A/E thresholds assume fully developed claims. Apply development factors before
  comparing against these bands.
- Gini drift thresholds correspond to the z-test p-values from arXiv 2510.04556.
"""

from dataclasses import dataclass, field


@dataclass
class PSIThresholds:
    """Traffic light thresholds for Population Stability Index.

    Attributes
    ----------
    green_max:
        PSI below this value: no significant population shift.
    amber_max:
        PSI between green_max and amber_max: investigate, likely some drift.
    red_min:
        PSI at or above amber_max (= red_min): significant shift, action required.
    """
    green_max: float = 0.10
    amber_max: float = 0.25

    def classify(self, psi: float) -> str:
        """Return 'green', 'amber', or 'red' for a PSI value."""
        if psi < self.green_max:
            return "green"
        elif psi < self.amber_max:
            return "amber"
        else:
            return "red"


@dataclass
class AERatioThresholds:
    """Traffic light thresholds for Actual/Expected ratio.

    Industry convention for mature accident periods (12+ months developed).
    Tighter bands trigger investigation before escalation.

    Attributes
    ----------
    green_lower, green_upper:
        A/E within this band: model calibrated, no action.
    amber_lower, amber_upper:
        A/E outside green but within amber: investigate.
    red_lower, red_upper:
        A/E outside amber: escalate (board-level model risk event).
    """
    green_lower: float = 0.95
    green_upper: float = 1.05
    amber_lower: float = 0.90
    amber_upper: float = 1.10
    red_lower: float = 0.80
    red_upper: float = 1.20

    def classify(self, ae: float) -> str:
        """Return 'green', 'amber', or 'red' for an A/E ratio value."""
        if self.green_lower <= ae <= self.green_upper:
            return "green"
        elif self.amber_lower <= ae <= self.amber_upper:
            return "amber"
        else:
            return "red"


@dataclass
class GiniDriftThresholds:
    """Traffic light thresholds for Gini drift z-test.

    Based on the asymptotic normality result from arXiv 2510.04556 (Theorem 1).
    The z-test compares the current Gini coefficient against the reference
    period distribution.

    Attributes
    ----------
    amber_p_value:
        Two-sided p-value below which we flag amber (monitor closely).
    red_p_value:
        Two-sided p-value below which we flag red (Gini has significantly degraded).
    """
    amber_p_value: float = 0.10
    red_p_value: float = 0.05

    def classify(self, p_value: float) -> str:
        """Return 'green', 'amber', or 'red' for a Gini drift p-value."""
        if p_value >= self.amber_p_value:
            return "green"
        elif p_value >= self.red_p_value:
            return "amber"
        else:
            return "red"


@dataclass
class MonitoringThresholds:
    """Aggregate threshold configuration passed to MonitoringReport.

    Example
    -------
    Tighten PSI thresholds for a large motor book with monthly monitoring::

        from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds
        thresholds = MonitoringThresholds(
            psi=PSIThresholds(green_max=0.05, amber_max=0.15),
        )
    """
    psi: PSIThresholds = field(default_factory=PSIThresholds)
    ae_ratio: AERatioThresholds = field(default_factory=AERatioThresholds)
    gini_drift: GiniDriftThresholds = field(default_factory=GiniDriftThresholds)
