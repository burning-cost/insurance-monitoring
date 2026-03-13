"""Result dataclasses for insurance-calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


@dataclass
class BalanceResult:
    """Result of the global balance property test.

    Attributes
    ----------
    balance_ratio
        alpha = sum(v * y) / sum(v * y_hat). Values above 1.0 indicate
        under-prediction; below 1.0 indicate over-prediction.
    observed_total
        Exposure-weighted sum of observed losses.
    predicted_total
        Exposure-weighted sum of predicted losses.
    ci_lower
        Lower bound of bootstrap confidence interval on balance ratio.
    ci_upper
        Upper bound of bootstrap confidence interval on balance ratio.
    p_value
        Two-sided p-value from Poisson z-test approximation. Small values
        indicate global imbalance.
    is_balanced
        True if the confidence interval includes 1.0.
    n_policies
        Number of observations (policies or records).
    total_exposure
        Sum of exposure weights.
    """

    balance_ratio: float
    observed_total: float
    predicted_total: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_balanced: bool
    n_policies: int
    total_exposure: float

    def __repr__(self) -> str:
        status = "BALANCED" if self.is_balanced else "IMBALANCED"
        return (
            f"BalanceResult({status}: alpha={self.balance_ratio:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f})"
        )

    def to_polars(self) -> "pl.DataFrame":
        """Return a one-row Polars DataFrame of all balance diagnostics."""
        import polars as pl

        return pl.DataFrame(
            {
                "balance_ratio": [self.balance_ratio],
                "observed_total": [self.observed_total],
                "predicted_total": [self.predicted_total],
                "ci_lower": [self.ci_lower],
                "ci_upper": [self.ci_upper],
                "p_value": [self.p_value],
                "is_balanced": [self.is_balanced],
                "n_policies": [self.n_policies],
                "total_exposure": [self.total_exposure],
            }
        )


@dataclass
class AutoCalibResult:
    """Result of the auto-calibration test.

    Attributes
    ----------
    p_value
        Bootstrap MCB or Hosmer-Lemeshow p-value. Small values indicate
        systematic miscalibration within prediction cohorts.
    is_calibrated
        True if p_value > significance level (default 0.05).
    per_bin
        Polars DataFrame with one row per prediction bin, containing columns:
        bin, pred_mean, obs_mean, ratio, exposure, n_policies.
    mcb_score
        Miscalibration component of the Murphy decomposition (raw deviance units).
    worst_bin_ratio
        Maximum absolute deviation from 1.0 of obs_mean / pred_mean across bins.
        A value of 0.25 means the worst cohort is 25% mis-priced.
    n_isotonic_steps
        Number of distinct levels in the isotonic step function. High values
        relative to sqrt(n) suggest overfitting noise (see Wüthrich & Ziegel 2024).
    """

    p_value: float
    is_calibrated: bool
    per_bin: "pl.DataFrame"
    mcb_score: float
    worst_bin_ratio: float
    n_isotonic_steps: int

    def __repr__(self) -> str:
        status = "CALIBRATED" if self.is_calibrated else "NOT CALIBRATED"
        return (
            f"AutoCalibResult({status}: p={self.p_value:.4f}, "
            f"MCB={self.mcb_score:.6f}, "
            f"worst bin ratio={self.worst_bin_ratio:.3f}, "
            f"isotonic steps={self.n_isotonic_steps})"
        )

    def to_polars(self) -> "pl.DataFrame":
        """Return summary as a one-row Polars DataFrame."""
        import polars as pl

        return pl.DataFrame(
            {
                "p_value": [self.p_value],
                "is_calibrated": [self.is_calibrated],
                "mcb_score": [self.mcb_score],
                "worst_bin_ratio": [self.worst_bin_ratio],
                "n_isotonic_steps": [self.n_isotonic_steps],
            }
        )


@dataclass
class MurphyResult:
    """Result of the Murphy score decomposition.

    The decomposition identity is::

        total_deviance = uncertainty - discrimination + miscalibration

    equivalently::

        S(Y, mu_hat) = UNC - DSC + MCB

    A well-ranked, well-calibrated model has high DSC and low MCB.

    Attributes
    ----------
    total_deviance
        D(y, y_hat): the raw deviance of the model.
    uncertainty
        UNC = D(y, y_bar): baseline deviance from the intercept-only model.
        This is determined by the data, not the model.
    discrimination
        DSC = UNC - D(y, y_hat_rc): improvement from ranking correctly.
        A model with no discriminatory power has DSC = 0.
    miscalibration
        MCB = D(y, y_hat) - D(y, y_hat_rc): excess deviance due to wrong
        price levels (independent of ranking ability).
    global_mcb
        GMCB: portion of MCB removed by balance correction (multiplying all
        predictions by the balance ratio). Cheap to fix.
    local_mcb
        LMCB: portion of MCB remaining after balance correction. Requires
        model refit or isotonic recalibration.
    discrimination_pct
        DSC as a percentage of total_deviance.
    miscalibration_pct
        MCB as a percentage of total_deviance.
    verdict
        'OK', 'RECALIBRATE', or 'REFIT' based on MCB magnitude and GMCB/LMCB split.
    """

    total_deviance: float
    uncertainty: float
    discrimination: float
    miscalibration: float
    global_mcb: float
    local_mcb: float
    discrimination_pct: float
    miscalibration_pct: float
    verdict: str

    def __repr__(self) -> str:
        return (
            f"MurphyResult({self.verdict}: "
            f"D={self.total_deviance:.4f}, "
            f"UNC={self.uncertainty:.4f}, "
            f"DSC={self.discrimination:.4f} ({self.discrimination_pct:.1f}%), "
            f"MCB={self.miscalibration:.4f} ({self.miscalibration_pct:.1f}%), "
            f"GMCB={self.global_mcb:.4f}, LMCB={self.local_mcb:.4f})"
        )

    def to_polars(self) -> "pl.DataFrame":
        """Return a one-row Polars DataFrame of all Murphy components."""
        import polars as pl

        return pl.DataFrame(
            {
                "total_deviance": [self.total_deviance],
                "uncertainty": [self.uncertainty],
                "discrimination": [self.discrimination],
                "miscalibration": [self.miscalibration],
                "global_mcb": [self.global_mcb],
                "local_mcb": [self.local_mcb],
                "discrimination_pct": [self.discrimination_pct],
                "miscalibration_pct": [self.miscalibration_pct],
                "verdict": [self.verdict],
            }
        )


@dataclass
class CalibrationReport:
    """Combined calibration report tying together balance, auto-calibration, and Murphy.

    Produced by :class:`CalibrationChecker` after calling :meth:`check`.

    Attributes
    ----------
    balance
        Result of the global balance property test.
    auto_calibration
        Result of the auto-calibration test.
    murphy
        Result of the Murphy score decomposition.
    distribution
        Loss distribution used ('poisson', 'gamma', 'tweedie', 'normal').
    n_policies
        Number of observations.
    total_exposure
        Sum of exposure weights.
    """

    balance: BalanceResult
    auto_calibration: AutoCalibResult
    murphy: MurphyResult
    distribution: str
    n_policies: int
    total_exposure: float

    def verdict(self) -> str:
        """Return the overall diagnostic verdict.

        Combines balance, auto-calibration, and Murphy signals into a single
        action recommendation.

        Returns
        -------
        str
            One of 'OK', 'MONITOR', 'RECALIBRATE', 'REFIT'.
        """
        if self.murphy.verdict == "REFIT":
            return "REFIT"
        if self.murphy.verdict == "RECALIBRATE" or not self.balance.is_balanced:
            return "RECALIBRATE"
        if not self.auto_calibration.is_calibrated:
            return "MONITOR"
        return "OK"

    def summary(self) -> str:
        """Return a human-readable paragraph summarising the diagnostic findings."""
        b = self.balance
        ac = self.auto_calibration
        m = self.murphy
        v = self.verdict()

        bal_status = "globally balanced" if b.is_balanced else "globally imbalanced"
        cal_status = "holds" if ac.is_calibrated else "is violated"

        lines = [
            f"Model calibration report ({self.distribution}, n={self.n_policies:,}, "
            f"exposure={self.total_exposure:,.1f}).",
            "",
            f"Balance: model is {bal_status} (ratio={b.balance_ratio:.3f}, "
            f"95% CI=[{b.ci_lower:.3f}, {b.ci_upper:.3f}], p={b.p_value:.3f}).",
            "",
            f"Auto-calibration {cal_status} (p={ac.p_value:.3f}). "
            f"Worst bin ratio: {ac.worst_bin_ratio:.3f}.",
            "",
            f"Murphy decomposition: UNC={m.uncertainty:.4f}, "
            f"DSC={m.discrimination:.4f} ({m.discrimination_pct:.1f}%), "
            f"MCB={m.miscalibration:.4f} ({m.miscalibration_pct:.1f}%). "
            f"Of miscalibration, GMCB={m.global_mcb:.4f} (global, fixable by recalibration) "
            f"and LMCB={m.local_mcb:.4f} (local, requires investigation).",
            "",
            f"Verdict: {v}.",
        ]
        return "\n".join(lines)

    def to_polars(self) -> "pl.DataFrame":
        """Return all diagnostics as a single flat Polars DataFrame."""
        import polars as pl

        return pl.concat(
            [
                self.balance.to_polars().rename(
                    {c: f"balance_{c}" for c in self.balance.to_polars().columns}
                ),
                self.auto_calibration.to_polars().rename(
                    {c: f"ac_{c}" for c in self.auto_calibration.to_polars().columns}
                ),
                self.murphy.to_polars().rename(
                    {c: f"murphy_{c}" for c in self.murphy.to_polars().columns}
                ),
                pl.DataFrame(
                    {
                        "distribution": [self.distribution],
                        "n_policies": [self.n_policies],
                        "total_exposure": [self.total_exposure],
                        "verdict": [self.verdict()],
                    }
                ),
            ],
            how="horizontal",
        )
