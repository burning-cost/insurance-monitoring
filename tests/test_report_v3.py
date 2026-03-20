"""Tests for v0.6.0 MonitoringReport additions: gini_bootstrap parameter."""

import numpy as np
import pytest

from insurance_monitoring import MonitoringReport


def _make_report_data(n_ref: int = 2000, n_cur: int = 1500, seed: int = 0):
    """Return reference and current arrays for MonitoringReport tests."""
    rng = np.random.default_rng(seed)
    ref_pred = rng.uniform(0.05, 0.20, n_ref)
    ref_act = rng.poisson(ref_pred).astype(float)
    cur_pred = rng.uniform(0.05, 0.20, n_cur)
    cur_act = rng.poisson(cur_pred).astype(float)
    return ref_act, ref_pred, cur_act, cur_pred


class TestMonitoringReportGiniBootstrap:
    """Integration tests for MonitoringReport with gini_bootstrap=True."""

    def test_gini_bootstrap_false_no_ci_fields(self):
        """Default (gini_bootstrap=False) should not add ci_lower/ci_upper."""
        ref_act, ref_pred, cur_act, cur_pred = _make_report_data(seed=100)
        report = MonitoringReport(
            reference_actual=ref_act,
            reference_predicted=ref_pred,
            current_actual=cur_act,
            current_predicted=cur_pred,
            n_bootstrap=50,
            gini_bootstrap=False,
        )
        gini_results = report.results_["gini"]
        assert "ci_lower" not in gini_results
        assert "ci_upper" not in gini_results

    def test_gini_bootstrap_true_adds_ci_fields(self):
        """gini_bootstrap=True should add ci_lower and ci_upper to results_['gini']."""
        ref_act, ref_pred, cur_act, cur_pred = _make_report_data(seed=101)
        report = MonitoringReport(
            reference_actual=ref_act,
            reference_predicted=ref_pred,
            current_actual=cur_act,
            current_predicted=cur_pred,
            n_bootstrap=100,
            gini_bootstrap=True,
        )
        gini_results = report.results_["gini"]
        assert "ci_lower" in gini_results, "ci_lower missing from results_['gini']"
        assert "ci_upper" in gini_results, "ci_upper missing from results_['gini']"
        # CIs should be finite floats in a sensible range
        assert -1.0 < gini_results["ci_lower"] < 1.0
        assert -1.0 < gini_results["ci_upper"] < 1.0
        assert gini_results["ci_lower"] <= gini_results["ci_upper"]

    def test_gini_bootstrap_true_ci_in_polars_output(self):
        """gini_bootstrap=True should add gini_ci_lower and gini_ci_upper rows in to_polars()."""
        ref_act, ref_pred, cur_act, cur_pred = _make_report_data(seed=102)
        report = MonitoringReport(
            reference_actual=ref_act,
            reference_predicted=ref_pred,
            current_actual=cur_act,
            current_predicted=cur_pred,
            n_bootstrap=100,
            gini_bootstrap=True,
        )
        df = report.to_polars()
        metrics = df["metric"].to_list()
        assert "gini_ci_lower" in metrics, "gini_ci_lower not in to_polars() output"
        assert "gini_ci_upper" in metrics, "gini_ci_upper not in to_polars() output"

    def test_gini_bootstrap_false_no_ci_in_polars_output(self):
        """gini_bootstrap=False should not add CI rows to to_polars()."""
        ref_act, ref_pred, cur_act, cur_pred = _make_report_data(seed=103)
        report = MonitoringReport(
            reference_actual=ref_act,
            reference_predicted=ref_pred,
            current_actual=cur_act,
            current_predicted=cur_pred,
            n_bootstrap=50,
            gini_bootstrap=False,
        )
        df = report.to_polars()
        metrics = df["metric"].to_list()
        assert "gini_ci_lower" not in metrics
        assert "gini_ci_upper" not in metrics

    def test_gini_bootstrap_recommendation_still_works(self):
        """recommendation property must still return a valid string with gini_bootstrap=True."""
        ref_act, ref_pred, cur_act, cur_pred = _make_report_data(seed=104)
        report = MonitoringReport(
            reference_actual=ref_act,
            reference_predicted=ref_pred,
            current_actual=cur_act,
            current_predicted=cur_pred,
            n_bootstrap=100,
            gini_bootstrap=True,
        )
        rec = report.recommendation
        valid = {"NO_ACTION", "RECALIBRATE", "REFIT", "INVESTIGATE", "MONITOR_CLOSELY"}
        assert rec in valid, f"Unexpected recommendation: {rec}"
