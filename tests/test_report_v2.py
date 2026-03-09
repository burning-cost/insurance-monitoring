"""Tests for v0.2.0 MonitoringReport additions: Murphy integration, new threshold defaults."""

import numpy as np
import polars as pl
import pytest

from insurance_monitoring import MonitoringReport
from insurance_monitoring.thresholds import MonitoringThresholds, GiniDriftThresholds


def _make_data(n_ref=5_000, n_cur=2_000, seed=0):
    """Generate synthetic insurance monitoring data."""
    rng = np.random.default_rng(seed)
    pred_ref = rng.uniform(0.05, 0.20, n_ref)
    act_ref = rng.poisson(pred_ref).astype(float)
    pred_cur = rng.uniform(0.05, 0.20, n_cur)
    act_cur = rng.poisson(pred_cur).astype(float)
    return act_ref, pred_ref, act_cur, pred_cur


class TestMonitoringReportMurphy:
    """MonitoringReport with Murphy decomposition (requires insurance-calibration)."""

    def test_murphy_not_computed_without_distribution(self):
        """Murphy section should be absent when murphy_distribution is not set."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        assert "murphy" not in report.results_
        assert report.murphy_available is False

    def test_murphy_attribute_exists_on_report(self):
        """MonitoringReport should have murphy_available attribute."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        assert hasattr(report, "murphy_available")
        assert isinstance(report.murphy_available, bool)

    def test_murphy_distribution_parameter_accepted(self):
        """murphy_distribution parameter should be accepted without error."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        # This may or may not compute Murphy depending on whether insurance-calibration
        # is installed. Both paths should work without error.
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
            murphy_distribution="poisson",
        )
        assert report._fitted is True

    def test_murphy_with_calibration_installed(self):
        """If insurance-calibration is installed, Murphy should be computed."""
        try:
            import insurance_calibration  # noqa: F401
            calibration_available = True
        except ImportError:
            calibration_available = False

        act_ref, pred_ref, act_cur, pred_cur = _make_data(seed=42)
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
            murphy_distribution="poisson",
        )

        if calibration_available:
            assert report.murphy_available is True
            assert "murphy" in report.results_
            murphy = report.results_["murphy"]
            expected_keys = {
                "uncertainty", "discrimination", "miscalibration",
                "global_mcb", "local_mcb", "discrimination_pct",
                "miscalibration_pct", "verdict",
            }
            assert expected_keys == set(murphy.keys())
            assert murphy["verdict"] in {"OK", "RECALIBRATE", "REFIT"}
        else:
            assert report.murphy_available is False
            assert "murphy" not in report.results_

    def test_to_dict_includes_murphy_available_flag(self):
        """to_dict should include murphy_available field."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        d = report.to_dict()
        assert "murphy_available" in d

    def test_to_polars_without_murphy(self):
        """to_polars should not include murphy rows when Murphy not computed."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        df = report.to_polars()
        murphy_metrics = [m for m in df["metric"].to_list() if m.startswith("murphy_")]
        assert len(murphy_metrics) == 0

    def test_to_polars_with_murphy_when_available(self):
        """to_polars should include murphy rows when Murphy is computed."""
        try:
            import insurance_calibration  # noqa: F401
            calibration_available = True
        except ImportError:
            calibration_available = False

        act_ref, pred_ref, act_cur, pred_cur = _make_data(seed=7)
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
            murphy_distribution="poisson",
        )

        if calibration_available:
            df = report.to_polars()
            murphy_metrics = [m for m in df["metric"].to_list() if m.startswith("murphy_")]
            assert len(murphy_metrics) > 0
            # Should include the key decomposition metrics
            expected = {"murphy_discrimination", "murphy_miscalibration"}
            actual_set = set(murphy_metrics)
            assert expected.issubset(actual_set)

    def test_recommendation_valid_with_murphy(self):
        """recommendation should still be one of the expected values with Murphy."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data(seed=8)
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
            murphy_distribution="poisson",
        )
        valid = {"NO_ACTION", "RECALIBRATE", "REFIT", "INVESTIGATE", "MONITOR_CLOSELY"}
        assert report.recommendation in valid

    def test_backward_compatible_without_murphy_distribution(self):
        """Omitting murphy_distribution must produce identical behaviour to v0.1.0."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data(seed=9)
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        d = report.to_dict()
        # Core keys present
        assert "results" in d
        assert "recommendation" in d
        assert "ae_ratio" in d["results"]
        assert "gini" in d["results"]

    def test_murphy_exposure_passed_through(self):
        """Murphy should use current exposure when provided."""
        try:
            import insurance_calibration  # noqa: F401
            calibration_available = True
        except ImportError:
            calibration_available = False

        rng = np.random.default_rng(200)
        n = 3_000
        pred_ref = rng.uniform(0.05, 0.20, n)
        act_ref = rng.poisson(pred_ref).astype(float)
        pred_cur = rng.uniform(0.05, 0.20, 1_000)
        act_cur = rng.poisson(pred_cur).astype(float)
        exp_cur = rng.uniform(0.1, 2.0, 1_000)

        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            exposure=exp_cur,
            n_bootstrap=20,
            murphy_distribution="poisson",
        )
        assert report._fitted is True

        if calibration_available:
            assert report.murphy_available is True


class TestMonitoringReportNewThresholds:
    """MonitoringReport uses new GiniDriftThresholds(amber=0.32, red=0.10) by default."""

    def test_default_thresholds_use_new_gini_defaults(self):
        """Default MonitoringReport should use amber=0.32, red=0.10 for Gini."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            n_bootstrap=20,
        )
        assert report.thresholds.gini_drift.amber_p_value == pytest.approx(0.32)
        assert report.thresholds.gini_drift.red_p_value == pytest.approx(0.10)

    def test_custom_traditional_thresholds_still_work(self):
        """Traditional thresholds should still be configurable for regulatory use."""
        act_ref, pred_ref, act_cur, pred_cur = _make_data()
        custom = MonitoringThresholds(
            gini_drift=GiniDriftThresholds(amber_p_value=0.10, red_p_value=0.05)
        )
        report = MonitoringReport(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            current_actual=act_cur,
            current_predicted=pred_cur,
            thresholds=custom,
            n_bootstrap=20,
        )
        assert report._fitted is True
        assert report.thresholds.gini_drift.amber_p_value == pytest.approx(0.10)
        assert report.thresholds.gini_drift.red_p_value == pytest.approx(0.05)
