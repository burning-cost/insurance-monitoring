"""Tests for insurance_monitoring.report module."""

import numpy as np
import polars as pl
import pytest

from insurance_monitoring import MonitoringReport
from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds


def _make_data(n_ref=5_000, n_cur=2_000, seed=0):
    """Generate synthetic insurance monitoring data."""
    rng = np.random.default_rng(seed)
    # Reference period
    pred_ref = rng.uniform(0.05, 0.20, n_ref)
    act_ref = rng.poisson(pred_ref).astype(float)
    exp_ref = np.ones(n_ref)
    # Current period (same distribution — no drift)
    pred_cur = rng.uniform(0.05, 0.20, n_cur)
    act_cur = rng.poisson(pred_cur).astype(float)
    exp_cur = np.ones(n_cur)
    return act_ref, pred_ref, exp_ref, act_cur, pred_cur, exp_cur


class TestMonitoringReport:
    """MonitoringReport integration tests."""

    def test_report_runs_without_error(self):
        """MonitoringReport should complete without raising."""
        act_ref, pred_ref, exp_ref, act_cur, pred_cur, exp_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        assert report._fitted is True

    def test_to_dict_has_results_and_recommendation(self):
        """to_dict should have 'results' and 'recommendation' keys."""
        act_ref, pred_ref, _, act_cur, pred_cur, _ = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        d = report.to_dict()
        assert "results" in d
        assert "recommendation" in d

    def test_to_polars_returns_dataframe(self):
        """to_polars should return a Polars DataFrame."""
        act_ref, pred_ref, _, act_cur, pred_cur, _ = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        df = report.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "metric" in df.columns
        assert "value" in df.columns
        assert "band" in df.columns

    def test_recommendation_valid_value(self):
        """recommendation should be one of the expected strings."""
        act_ref, pred_ref, _, act_cur, pred_cur, _ = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        valid = {"NO_ACTION", "RECALIBRATE", "REFIT", "INVESTIGATE", "MONITOR_CLOSELY"}
        assert report.recommendation in valid

    def test_report_with_features(self):
        """Report should include CSI when feature dataframes provided."""
        act_ref, pred_ref, _, act_cur, pred_cur, _ = _make_data(n_ref=3_000, n_cur=1_000)
        rng = np.random.default_rng(99)
        feat_ref = pl.DataFrame({
            "driver_age": rng.normal(35, 8, 3_000).tolist(),
            "vehicle_age": rng.uniform(0, 15, 3_000).tolist(),
        })
        feat_cur = pl.DataFrame({
            "driver_age": rng.normal(35, 8, 1_000).tolist(),
            "vehicle_age": rng.uniform(0, 15, 1_000).tolist(),
        })
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            feature_df_reference=feat_ref,
            feature_df_current=feat_cur,
            features=["driver_age", "vehicle_age"],
            n_bootstrap=20,
        )
        assert "csi" in report.results_
        df = report.to_polars()
        csi_metrics = [m for m in df["metric"].to_list() if m.startswith("csi_")]
        assert len(csi_metrics) == 2

    def test_ae_ratio_green_for_calibrated_model(self):
        """A well-calibrated model should show green A/E."""
        rng = np.random.default_rng(100)
        n = 10_000
        pred = rng.uniform(0.05, 0.15, n)
        act = rng.poisson(pred).astype(float)
        report = MonitoringReport(
            reference_actual=act,
            reference_predicted=pred,
            current_actual=act,
            current_predicted=pred,
            n_bootstrap=20,
        )
        assert report.results_["ae_ratio"]["band"] == "green"

    def test_custom_thresholds_accepted(self):
        """Custom thresholds should be accepted and used."""
        act_ref, pred_ref, _, act_cur, pred_cur, _ = _make_data()
        custom = MonitoringThresholds(psi=PSIThresholds(green_max=0.01, amber_max=0.05))
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            thresholds=custom,
            n_bootstrap=20,
        )
        assert report._fitted is True

    def test_report_with_exposure(self):
        """Report should accept and use exposure arrays."""
        act_ref, pred_ref, exp_ref, act_cur, pred_cur, exp_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            exposure=exp_cur,
            reference_exposure=exp_ref,
            n_bootstrap=20,
        )
        assert report._fitted is True
        assert "ae_ratio" in report.results_
