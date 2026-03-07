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
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test
from insurance_monitoring.drift import csi, psi
from insurance_monitoring.thresholds import MonitoringThresholds


ArrayLike = Union[np.ndarray, pl.Series]


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

    Attributes
    ----------
    results_ : dict
        Populated after fitting. Contains all metrics and traffic-light bands.
    recommendation : str
        Decision recommendation from the arXiv 2510.04556 decision tree.
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

    def __post_init__(self) -> None:
        self.results_: dict = {}
        self._fitted: bool = False
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

        self.results_ = results
        self._fitted = True

    @property
    def recommendation(self) -> str:
        """Decision recommendation based on arXiv 2510.04556 three-stage framework.

        Returns one of:
        - 'NO_ACTION': no significant drift detected
        - 'RECALIBRATE': A/E ratio drifted but Gini stable — update intercept
        - 'REFIT': Gini has degraded — rebuild model on recent data
        - 'INVESTIGATE': multiple signals — manual review required
        """
        if not self._fitted:
            return "NOT_RUN"

        ae_red = self.results_["ae_ratio"]["band"] == "red"
        ae_amber = self.results_["ae_ratio"]["band"] == "amber"
        gini_red = self.results_["gini"]["band"] == "red"
        gini_amber = self.results_["gini"]["band"] == "amber"

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
        }

    def to_polars(self) -> pl.DataFrame:
        """Return a flat Polars DataFrame with one row per metric.

        Columns: ``metric``, ``value``, ``band``.

        CSI per-feature rows are included with metric names like 'csi_driver_age'.
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

        rows.append({
            "metric": "recommendation",
            "value": float("nan"),
            "band": self.recommendation,
        })

        return pl.DataFrame(
            rows,
            schema={"metric": pl.Utf8, "value": pl.Float64, "band": pl.Utf8},
        )
