"""
Combined monitoring report for insurance pricing models.

MonitoringReport is the entry point for a complete monthly or quarterly
monitoring run. It runs drift, calibration, and discrimination checks in one
pass and produces a structured summary with traffic-light status per metric.

Design rationale:
- Single function call replaces manual orchestration of 5+ metrics
- Traffic lights use configurable thresholds (see thresholds.py)
- Output is a plain dict or Polars DataFrame — no opaque report objects,
  easy to write to a Delta table or log to MLflow
- Follows the three-stage decision tree from arXiv 2510.04556:
  Gini OK + A/E OK -> no action
  A/E bad only -> recalibrate
  Gini bad -> refit
- Murphy decomposition is now built in (insurance-calibration absorbed into this
  package as of v0.3.0). Set ``murphy_distribution`` to enable it. The Murphy
  verdict sharpens the RECALIBRATE vs REFIT distinction.

Usage
-----
::

    from insurance_monitoring import MonitoringReport
    import polars as pl

    report = MonitoringReport(
        reference_actual=train_claims,
        reference_predicted=train_predicted,
        current_actual=current_claims,
        current_predicted=current_predicted,
        exposure=current_exposure,
        feature_df_reference=train_features,
        feature_df_current=current_features,
        features=["driver_age", "vehicle_age", "ncd_years"],
        murphy_distribution="poisson",  # Murphy decomposition now always available
        gini_bootstrap=True,            # add percentile CI on Gini (v0.6.0)
    )
    print(report.to_dict())
    print(report.to_polars())
    print(report.recommendation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import polars as pl

from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci
from insurance_monitoring.calibration._murphy import murphy_decomposition
from insurance_monitoring.discrimination import (
    gini_coefficient,
    gini_drift_test,
    GiniDriftBootstrapTest,
)
from insurance_monitoring.drift import csi, psi
from insurance_monitoring.thresholds import MonitoringThresholds


ArrayLike = Union[np.ndarray, pl.Series]


def _try_murphy(
    actual: np.ndarray,
    predicted: np.ndarray,
    exposure: Optional[np.ndarray],
    distribution: str,
) -> Optional[dict]:
    """Run Murphy decomposition using the built-in calibration module.

    Returns a dict of Murphy components, or None on any error. Never raises —
    a monitoring run should not fail because of a Murphy computation error.
    """
    try:
        result = murphy_decomposition(
            y=actual,
            y_hat=predicted,
            exposure=exposure,
            distribution=distribution,
        )
        return {
            "uncertainty": result.uncertainty,
            "discrimination": result.discrimination,
            "miscalibration": result.miscalibration,
            "global_mcb": result.global_mcb,
            "local_mcb": result.local_mcb,
            "discrimination_pct": result.discrimination_pct,
            "miscalibration_pct": result.miscalibration_pct,
            "verdict": result.verdict,
        }
    except Exception:
        return None


@dataclass
class MonitoringReport:
    """Run a full model monitoring check and produce a traffic-light summary.

    Parameters
    ----------
    reference_actual:
        Observed values from the reference (training/baseline) period.
    reference_predicted:
        Model predictions from the reference period.
    current_actual:
        Observed values from the current monitoring period.
    current_predicted:
        Model predictions from the current monitoring period.
    exposure:
        Optional exposure weights for the current period (e.g., car-years).
    reference_exposure:
        Optional exposure weights for the reference period.
    feature_df_reference:
        Optional Polars DataFrame of features from the reference period.
        Required for CSI computation.
    feature_df_current:
        Optional Polars DataFrame of features from the current period.
        Required for CSI computation.
    features:
        List of feature names to include in CSI monitoring.
    score_reference:
        Optional model score (e.g., log-rate) array for reference period,
        used for score-level PSI.
    score_current:
        Optional model score array for current period.
    thresholds:
        MonitoringThresholds instance. Defaults to industry-standard settings.
    n_bootstrap:
        Bootstrap replicates for Gini variance estimation. Default 200.
    murphy_distribution:
        If provided, compute Murphy decomposition on the current period data
        using this distribution. Supported: 'poisson', 'gamma', 'tweedie',
        'normal'. Default None (Murphy section omitted from report).

        The Murphy decomposition distinguishes calibration drift (MCB high,
        GMCB > LMCB -> RECALIBRATE) from discrimination drift (DSC low ->
        REFIT). When set, it sharpens the recommendation logic beyond the
        simpler Gini/A/E heuristic.
    gini_bootstrap : bool, default False
        When True, compute GiniDriftBootstrapTest instead of the two-sample
        gini_drift_test. Uses the reference period Gini as training_gini and
        bootstraps the current (monitor) period to produce percentile CIs.
        Adds ``ci_lower`` and ``ci_upper`` fields to ``results_["gini"]``.

        When False (default), existing two-sample gini_drift_test behaviour
        is preserved unchanged.

    Attributes
    ----------
    results_ : dict
        Populated after fitting. Contains all metrics and traffic-light bands.
    recommendation : str
        Decision recommendation from the arXiv 2510.04556 decision tree.
    murphy_available : bool
        True if Murphy decomposition was successfully computed.
    """

    reference_actual: ArrayLike
    reference_predicted: ArrayLike
    current_actual: ArrayLike
    current_predicted: ArrayLike
    exposure: Optional[ArrayLike] = None
    reference_exposure: Optional[ArrayLike] = None
    feature_df_reference: Optional[pl.DataFrame] = None
    feature_df_current: Optional[pl.DataFrame] = None
    features: list[str] = field(default_factory=list)
    score_reference: Optional[ArrayLike] = None
    score_current: Optional[ArrayLike] = None
    thresholds: MonitoringThresholds = field(default_factory=MonitoringThresholds)
    n_bootstrap: int = 200
    murphy_distribution: Optional[str] = None
    gini_bootstrap: bool = False

    def __post_init__(self) -> None:
        self.results_: dict = {}
        self._fitted: bool = False
        self.murphy_available: bool = False
        self._run()

    def _run(self) -> None:
        """Execute all monitoring checks."""
        results: dict = {}

        # --- A/E ratio ---
        ae_result = ae_ratio_ci(
            self.current_actual,
            self.current_predicted,
            exposure=self.exposure,
        )
        ae_band = self.thresholds.ae_ratio.classify(ae_result["ae"])
        results["ae_ratio"] = {
            "value": ae_result["ae"],
            "lower_ci": ae_result["lower"],
            "upper_ci": ae_result["upper"],
            "n_claims": ae_result["n_claims"],
            "n_expected": ae_result["n_expected"],
            "band": ae_band,
        }

        # --- Gini coefficient ---
        gini_ref = gini_coefficient(
            self.reference_actual,
            self.reference_predicted,
            exposure=self.reference_exposure,
        )
        gini_cur = gini_coefficient(
            self.current_actual,
            self.current_predicted,
            exposure=self.exposure,
        )

        if self.gini_bootstrap:
            # One-sample bootstrap design: treat reference Gini as fixed scalar.
            # Produces percentile CIs on both monitor Gini and the change.
            bt = GiniDriftBootstrapTest(
                training_gini=gini_ref,
                monitor_actual=self.current_actual,
                monitor_predicted=self.current_predicted,
                monitor_exposure=self.exposure,
                n_bootstrap=self.n_bootstrap,
                random_state=None,
            )
            bt_result = bt.test()
            gini_band = self.thresholds.gini_drift.classify(bt_result.p_value)
            results["gini"] = {
                "reference": gini_ref,
                "current": gini_cur,
                "change": bt_result.gini_change,
                "z_statistic": bt_result.z_statistic,
                "p_value": bt_result.p_value,
                "ci_lower": bt_result.ci_lower,
                "ci_upper": bt_result.ci_upper,
                "band": gini_band,
            }
        else:
            n_ref = len(np.asarray(self.reference_actual))
            n_cur = len(np.asarray(self.current_actual))

            drift_result = gini_drift_test(
                reference_gini=gini_ref,
                current_gini=gini_cur,
                n_reference=n_ref,
                n_current=n_cur,
                reference_actual=self.reference_actual,
                reference_predicted=self.reference_predicted,
                current_actual=self.current_actual,
                current_predicted=self.current_predicted,
                reference_exposure=self.reference_exposure,
                current_exposure=self.exposure,
                n_bootstrap=self.n_bootstrap,
            )
            gini_band = self.thresholds.gini_drift.classify(drift_result["p_value"])
            results["gini"] = {
                "reference": gini_ref,
                "current": gini_cur,
                "change": drift_result["gini_change"],
                "z_statistic": drift_result["z_statistic"],
                "p_value": drift_result["p_value"],
                "band": gini_band,
            }

        # --- Score PSI ---
        if self.score_reference is not None and self.score_current is not None:
            score_psi = psi(self.score_reference, self.score_current, n_bins=10)
            score_band = self.thresholds.psi.classify(score_psi)
            results["score_psi"] = {
                "value": score_psi,
                "band": score_band,
            }

        # --- CSI per feature ---
        if (
            self.feature_df_reference is not None
            and self.feature_df_current is not None
            and self.features
        ):
            csi_df = csi(
                self.feature_df_reference,
                self.feature_df_current,
                self.features,
            )
            results["csi"] = csi_df.to_dicts()
            # Overall CSI summary: max CSI across features
            max_csi = float(csi_df["csi"].max())
            results["max_csi"] = {
                "value": max_csi,
                "band": self.thresholds.psi.classify(max_csi),
                "worst_feature": csi_df.filter(pl.col("csi") == max_csi)["feature"][0],
            }

        # --- Murphy decomposition ---
        if self.murphy_distribution is not None:
            act_cur = np.asarray(self.current_actual, dtype=np.float64)
            pred_cur = np.asarray(self.current_predicted, dtype=np.float64)
            exp_cur = (
                np.asarray(self.exposure, dtype=np.float64)
                if self.exposure is not None
                else None
            )
            murphy_result = _try_murphy(
                act_cur, pred_cur, exp_cur, self.murphy_distribution
            )
            if murphy_result is not None:
                results["murphy"] = murphy_result
                self.murphy_available = True

        self.results_ = results
        self._fitted = True

    @property
    def recommendation(self) -> str:
        """Decision recommendation based on arXiv 2510.04556 three-stage framework.

        When Murphy decomposition is available (murphy_distribution set), the
        recommendation uses the Murphy verdict to sharpen the RECALIBRATE vs
        REFIT distinction:

        - Murphy says REFIT (DSC degraded, local miscalibration dominates) -> REFIT
        - Murphy says RECALIBRATE (global miscalibration dominates) -> RECALIBRATE
          even if the Gini z-test has not crossed red yet

        Without Murphy, falls back to the simpler Gini/A/E heuristic.

        Returns one of:
        - 'NO_ACTION': no significant drift detected
        - 'RECALIBRATE': A/E ratio drifted but Gini stable -- update intercept
        - 'REFIT': Gini has degraded -- rebuild model on recent data
        - 'INVESTIGATE': multiple conflicting signals -- manual review required
        - 'MONITOR_CLOSELY': amber signals but no red -- watch the trend
        """
        if not self._fitted:
            return "NOT_RUN"

        ae_red = self.results_["ae_ratio"]["band"] == "red"
        ae_amber = self.results_["ae_ratio"]["band"] == "amber"
        gini_red = self.results_["gini"]["band"] == "red"
        gini_amber = self.results_["gini"]["band"] == "amber"

        # Murphy verdict takes precedence when available: it directly tests
        # whether the degradation is a calibration issue (cheap to fix) or
        # a discrimination issue (requires refit).
        if self.murphy_available and "murphy" in self.results_:
            murphy_verdict = self.results_["murphy"]["verdict"]
            if murphy_verdict == "REFIT" or gini_red:
                return "REFIT"
            elif murphy_verdict == "RECALIBRATE" or (ae_red and not gini_red):
                return "RECALIBRATE"
            elif gini_amber and ae_red:
                return "INVESTIGATE"
            elif ae_amber or gini_amber:
                return "MONITOR_CLOSELY"
            else:
                return "NO_ACTION"

        # Fallback: simpler Gini/A/E heuristic (original v0.1.0 logic)
        if gini_red:
            return "REFIT"
        elif gini_amber and ae_red:
            return "INVESTIGATE"
        elif ae_red and not gini_red:
            return "RECALIBRATE"
        elif ae_amber or gini_amber:
            return "MONITOR_CLOSELY"
        else:
            return "NO_ACTION"

    def to_dict(self) -> dict:
        """Return all monitoring results as a nested dictionary.

        Suitable for JSON serialisation or logging to MLflow as run metrics.
        """
        return {
            "results": self.results_,
            "recommendation": self.recommendation,
            "murphy_available": self.murphy_available,
        }

    def to_polars(self) -> pl.DataFrame:
        """Return a flat Polars DataFrame with one row per metric.

        Columns: ``metric``, ``value``, ``band``.

        CSI per-feature rows are included with metric names like 'csi_driver_age'.
        Murphy rows are included as 'murphy_discrimination', 'murphy_miscalibration',
        etc. when murphy_distribution is set.
        When gini_bootstrap=True, adds 'gini_ci_lower' and 'gini_ci_upper' rows.
        """
        rows = []

        ae = self.results_["ae_ratio"]
        rows.append({"metric": "ae_ratio", "value": ae["value"], "band": ae["band"]})
        rows.append({"metric": "ae_ratio_lower_ci", "value": ae["lower_ci"], "band": ae["band"]})
        rows.append({"metric": "ae_ratio_upper_ci", "value": ae["upper_ci"], "band": ae["band"]})

        gini = self.results_["gini"]
        rows.append({"metric": "gini_current", "value": gini["current"], "band": gini["band"]})
        rows.append({"metric": "gini_reference", "value": gini["reference"], "band": "green"})
        rows.append({"metric": "gini_change", "value": gini["change"], "band": gini["band"]})
        rows.append({"metric": "gini_p_value", "value": gini["p_value"], "band": gini["band"]})

        # CI rows present only when gini_bootstrap=True
        if "ci_lower" in gini:
            rows.append({
                "metric": "gini_ci_lower",
                "value": gini["ci_lower"],
                "band": gini["band"],
            })
            rows.append({
                "metric": "gini_ci_upper",
                "value": gini["ci_upper"],
                "band": gini["band"],
            })

        if "score_psi" in self.results_:
            sp = self.results_["score_psi"]
            rows.append({"metric": "score_psi", "value": sp["value"], "band": sp["band"]})

        if "csi" in self.results_:
            for csi_row in self.results_["csi"]:
                rows.append({
                    "metric": f"csi_{csi_row['feature']}",
                    "value": csi_row["csi"],
                    "band": csi_row["band"],
                })

        # Murphy decomposition rows
        if "murphy" in self.results_:
            m = self.results_["murphy"]
            murphy_verdict = m["verdict"]
            rows.append({
                "metric": "murphy_discrimination",
                "value": m["discrimination"],
                "band": murphy_verdict,
            })
            rows.append({
                "metric": "murphy_miscalibration",
                "value": m["miscalibration"],
                "band": murphy_verdict,
            })
            rows.append({
                "metric": "murphy_discrimination_pct",
                "value": m["discrimination_pct"],
                "band": murphy_verdict,
            })
            rows.append({
                "metric": "murphy_miscalibration_pct",
                "value": m["miscalibration_pct"],
                "band": murphy_verdict,
            })
            rows.append({
                "metric": "murphy_global_mcb",
                "value": m["global_mcb"],
                "band": murphy_verdict,
            })
            rows.append({
                "metric": "murphy_local_mcb",
                "value": m["local_mcb"],
                "band": murphy_verdict,
            })

        rows.append({
            "metric": "recommendation",
            "value": float("nan"),
            "band": self.recommendation,
        })

        return pl.DataFrame(
            rows,
            schema={"metric": pl.String, "value": pl.Float64, "band": pl.String},
        )
