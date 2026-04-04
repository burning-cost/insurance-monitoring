"""
Test coverage expansion batch 7.

Adds ~60 targeted tests to push the total to >= 1763 (50% above baseline 1175).
Covers remaining gaps: SequentialTest edge cases, ConformalChartResult to_polars
dtypes, conformal_chart math properties, DriftAttributor psi_comparison,
_bayesian_prob, _build_summary, and parametric top-ups.
"""

from __future__ import annotations

import datetime
import math
import warnings

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# _bayesian_prob
# ---------------------------------------------------------------------------

class TestBayesianProb:
    def test_import(self):
        from insurance_monitoring.sequential import _bayesian_prob
        assert callable(_bayesian_prob)

    def test_equal_rates_near_half(self):
        from insurance_monitoring.sequential import _bayesian_prob
        # Same claims and exposure -> P(B < A) ~ 0.5
        p = _bayesian_prob(200, 1000, 200, 1000)
        assert 0.3 <= p <= 0.7

    def test_challenger_much_lower_rate(self):
        from insurance_monitoring.sequential import _bayesian_prob
        # C_B very small vs C_A -> P(lambda_B < lambda_A | data) should be high
        p = _bayesian_prob(1000, 1000, 10, 1000)
        assert p > 0.9

    def test_challenger_much_higher_rate(self):
        from insurance_monitoring.sequential import _bayesian_prob
        # C_B very large vs C_A -> P(lambda_B < lambda_A | data) should be near 0
        p = _bayesian_prob(10, 1000, 1000, 1000)
        assert p < 0.1

    def test_in_range(self):
        from insurance_monitoring.sequential import _bayesian_prob
        p = _bayesian_prob(50, 500, 50, 500)
        assert 0.0 <= p <= 1.0

    def test_small_counts_exact_path(self):
        from insurance_monitoring.sequential import _bayesian_prob
        # C_A=10, C_B=10 -> uses exact integration path (not normal approx)
        p = _bayesian_prob(10, 100, 10, 100)
        assert 0.0 <= p <= 1.0

    def test_large_counts_normal_approx_path(self):
        from insurance_monitoring.sequential import _bayesian_prob
        # C_A > 50 and C_B > 50 -> uses normal approximation path
        p = _bayesian_prob(200, 1000, 200, 1000)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_import(self):
        from insurance_monitoring.sequential import _build_summary
        assert callable(_build_summary)

    def test_frequency_label(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("inconclusive", 1.05, 0.9, 1.2, 5.0, 20.0, "frequency")
        assert "freq" in s

    def test_severity_label(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("inconclusive", 1.05, 0.9, 1.2, 5.0, 20.0, "severity")
        assert "sev" in s

    def test_loss_ratio_label(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("inconclusive", 1.05, 0.9, 1.2, 5.0, 20.0, "loss_ratio")
        assert "LR" in s

    def test_cs_insufficient_data(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("inconclusive", 1.0, 0.0, float("inf"), 1.0, 20.0, "frequency")
        assert "insufficient" in s

    def test_reject_h0_in_summary(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("reject_H0", 1.1, 1.0, 1.2, 100.0, 20.0, "frequency")
        assert "Reject H0" in s or "reject" in s.lower()

    def test_higher_direction(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("inconclusive", 1.1, 0.9, 1.3, 5.0, 20.0, "frequency")
        assert "higher" in s

    def test_lower_direction(self):
        from insurance_monitoring.sequential import _build_summary
        s = _build_summary("inconclusive", 0.9, 0.8, 1.0, 5.0, 20.0, "frequency")
        assert "lower" in s


# ---------------------------------------------------------------------------
# ConformalChartResult to_polars data types
# ---------------------------------------------------------------------------

class TestConformalChartResultPolarsTypes:
    def _make_result(self):
        from insurance_monitoring.conformal_chart import ConformalChartResult
        scores = np.array([1.0, 2.0, 3.0])
        p_values = np.array([0.8, 0.3, 0.04])
        is_alarm = p_values < 0.05
        return ConformalChartResult(
            scores=scores, p_values=p_values, threshold=2.5,
            is_alarm=is_alarm, alpha=0.05, n_cal=50
        )

    def test_obs_index_dtype(self):
        r = self._make_result()
        df = r.to_polars()
        assert df["obs_index"].dtype == pl.Int64

    def test_score_dtype_float(self):
        r = self._make_result()
        df = r.to_polars()
        assert df["score"].dtype in (pl.Float32, pl.Float64)

    def test_p_value_dtype_float(self):
        r = self._make_result()
        df = r.to_polars()
        assert df["p_value"].dtype in (pl.Float32, pl.Float64)

    def test_is_alarm_dtype_bool(self):
        r = self._make_result()
        df = r.to_polars()
        assert df["is_alarm"].dtype == pl.Boolean

    def test_p_values_in_data_match_result(self):
        r = self._make_result()
        df = r.to_polars()
        np.testing.assert_allclose(df["p_value"].to_numpy(), r.p_values)

    def test_is_alarm_correct_values(self):
        r = self._make_result()
        df = r.to_polars()
        assert df["is_alarm"].to_list() == [False, False, True]


# ---------------------------------------------------------------------------
# ConformalControlChart: full workflow with NCS helper
# ---------------------------------------------------------------------------

class TestConformalChartWorkflow:
    def test_full_workflow_with_relative_residual(self):
        from insurance_monitoring.conformal_chart import ConformalControlChart
        rng = np.random.default_rng(42)
        # Calibration data: in-control predictions vs actuals
        n_cal = 200
        pred_cal = rng.exponential(0.1, n_cal)
        actual_cal = rng.poisson(pred_cal)
        ncs_cal = ConformalControlChart.ncs_relative_residual(actual_cal, pred_cal)
        chart = ConformalControlChart(alpha=0.05).fit(ncs_cal)

        # Monitor data: in-control
        n_mon = 100
        pred_mon = rng.exponential(0.1, n_mon)
        actual_mon = rng.poisson(pred_mon)
        ncs_mon = ConformalControlChart.ncs_relative_residual(actual_mon, pred_mon)
        result = chart.monitor(ncs_mon)

        assert len(result.scores) == n_mon
        # Alarm rate under null should be approximately 5%
        assert result.alarm_rate < 0.20

    def test_full_workflow_with_absolute_residual(self):
        from insurance_monitoring.conformal_chart import ConformalControlChart
        rng = np.random.default_rng(99)
        pred = rng.normal(0, 1, 300)
        actual = pred + rng.normal(0, 0.1, 300)
        ncs_cal = ConformalControlChart.ncs_absolute_residual(actual[:200], pred[:200])
        chart = ConformalControlChart(alpha=0.05).fit(ncs_cal)
        ncs_mon = ConformalControlChart.ncs_absolute_residual(actual[200:], pred[200:])
        result = chart.monitor(ncs_mon)
        assert result.n_cal == 200

    def test_full_workflow_with_studentized(self):
        from insurance_monitoring.conformal_chart import ConformalControlChart
        rng = np.random.default_rng(77)
        actual = rng.normal(0, 1, 200)
        pred = actual + rng.normal(0, 0.1, 200)
        vol = np.abs(rng.normal(1, 0.1, 200)) + 0.01
        ncs = ConformalControlChart.ncs_studentized(actual, pred, vol)
        chart = ConformalControlChart(alpha=0.05).fit(ncs)
        result = chart.monitor(ncs)
        assert isinstance(result.alarm_rate, float)


# ---------------------------------------------------------------------------
# DriftAttributor psi_comparison
# ---------------------------------------------------------------------------

class TestDriftAttributorPsiComparison:
    def test_psi_comparison_returns_df(self):
        from insurance_monitoring.drift_attribution import DriftAttributor, DriftAttributionResult
        import polars as pl

        # Minimal DriftAttributionResult
        features = ["age", "ncb"]
        ranking_df = pl.DataFrame({
            "feature": features,
            "test_statistic": [1.0, 0.5],
            "threshold": [0.8, 0.8],
            "ratio": [1.25, 0.625],
            "p_value": [0.02, 0.4],
            "drift_attributed": [True, False],
        })
        result = DriftAttributionResult(
            drift_detected=True,
            attributed_features=["age"],
            test_statistics={"age": 1.0, "ncb": 0.5},
            thresholds={"age": 0.8, "ncb": 0.8},
            p_values={"age": 0.02, "ncb": 0.4},
            window_ref_size=100,
            window_new_size=100,
            alpha=0.05,
            feature_ranking=ranking_df,
            interaction_pairs=None,
            subset_risks_ref={"age": 0.1, "ncb": 0.2},
            subset_risks_new={"age": 0.15, "ncb": 0.2},
            model_retrained=False,
        )

        # Build minimal reference and current DataFrames
        rng = np.random.default_rng(42)
        ref = pl.DataFrame({
            "age": rng.normal(35, 10, 200).tolist(),
            "ncb": rng.integers(0, 5, 200).astype(float).tolist(),
        })
        cur = pl.DataFrame({
            "age": rng.normal(35, 10, 100).tolist(),
            "ncb": rng.integers(0, 5, 100).astype(float).tolist(),
        })

        merged = DriftAttributor.psi_comparison(result, ref, cur, features=features)
        assert isinstance(merged, pl.DataFrame)
        assert "feature" in merged.columns


# ---------------------------------------------------------------------------
# Additional parametric tests: SequentialTest with calendar dates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_periods", [1, 3, 6, 12])
def test_sequential_test_n_updates_matches_periods(n_periods):
    from insurance_monitoring.sequential import SequentialTest
    test = SequentialTest()
    result = None
    for i in range(n_periods):
        result = test.update(
            champion_claims=10, champion_exposure=100,
            challenger_claims=10, challenger_exposure=100,
        )
    assert result.n_updates == n_periods
    assert len(test.history()) == n_periods


@pytest.mark.parametrize("c_a,c_b,expected_direction", [
    (100, 50, "lower"),    # challenger lower rate
    (50, 100, "higher"),   # challenger higher rate
    (100, 100, None),      # equal - either direction
])
def test_sequential_summary_direction(c_a, c_b, expected_direction):
    from insurance_monitoring.sequential import SequentialTest
    test = SequentialTest(alpha=0.05, min_exposure_per_arm=0.0)
    result = test.update(
        champion_claims=c_a, champion_exposure=1000,
        challenger_claims=c_b, challenger_exposure=1000,
    )
    if expected_direction is not None:
        assert expected_direction in result.summary


@pytest.mark.parametrize("score_val,alpha,expect_alarm", [
    (0.0, 0.05, False),    # very low score -> high p-value -> no alarm
    (1000.0, 0.05, True),  # extreme score -> alarm
])
def test_conformal_chart_alarm_behavior(score_val, alpha, expect_alarm):
    from insurance_monitoring.conformal_chart import ConformalControlChart
    rng = np.random.default_rng(42)
    cal = rng.exponential(2.0, 500)
    chart = ConformalControlChart(alpha=alpha).fit(cal)
    result = chart.monitor(np.array([score_val]))
    assert result.is_alarm[0] == expect_alarm


@pytest.mark.parametrize("field_name", [
    "drift_detected", "attributed_features", "test_statistics", "thresholds",
    "p_values", "window_ref_size", "window_new_size", "alpha",
    "feature_ranking", "interaction_pairs", "subset_risks_ref", "subset_risks_new",
    "model_retrained",
])
def test_drift_attribution_result_has_field(field_name):
    from insurance_monitoring.drift_attribution import DriftAttributionResult
    r = DriftAttributionResult(
        drift_detected=False,
        attributed_features=[],
        test_statistics={"f": 0.1},
        thresholds={"f": 1.0},
        p_values={"f": 0.5},
        window_ref_size=100,
        window_new_size=100,
        alpha=0.05,
        feature_ranking=pl.DataFrame({"feature": ["f"], "ratio": [0.1]}),
        interaction_pairs=None,
        subset_risks_ref={"f": 0.1},
        subset_risks_new={"f": 0.1},
        model_retrained=False,
    )
    assert hasattr(r, field_name)
    val = r[field_name]
    expected_val = getattr(r, field_name)
    if isinstance(val, pl.DataFrame):
        assert val.equals(expected_val)
    else:
        assert val == expected_val


@pytest.mark.parametrize("alpha,n_cal,expect_warning", [
    (0.05, 10, True),   # n < 20, alpha >= 0.05 -> warn
    (0.05, 25, False),  # n >= 20 -> no warning from small-cal branch
    (0.003, 100, True), # alpha <= 0.003, n < 370 -> warn
])
def test_conformal_chart_fit_warnings(alpha, n_cal, expect_warning):
    from insurance_monitoring.conformal_chart import ConformalControlChart
    chart = ConformalControlChart(alpha=alpha)
    rng = np.random.default_rng(42)
    cal = rng.exponential(1, n_cal)
    with warnings.catch_warnings(record=True) as w:
        import warnings as _warnings
        _warnings.simplefilter("always")
        chart.fit(cal)
        warned = len(w) > 0
        assert warned == expect_warning


@pytest.mark.parametrize("metric_label,metric", [
    ("freq", "frequency"),
    ("sev", "severity"),
    ("LR", "loss_ratio"),
])
def test_build_summary_metric_labels(metric_label, metric):
    from insurance_monitoring.sequential import _build_summary
    s = _build_summary("inconclusive", 1.0, 0.9, 1.1, 5.0, 20.0, metric)
    assert metric_label in s
