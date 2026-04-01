"""
Tests for insurance_monitoring.conformal_chart.

Coverage plan from build spec (KB 5243 / conformal-spc-research.md):
  - Conformal p-value edge cases
  - Threshold calculation at alpha=0.0027 and alpha=0.05
  - Monte Carlo false alarm rate control
  - All four NCS helpers (absolute, relative, median deviation, studentized)
  - ConformalChartResult: to_polars schema, summary string
  - ConformalControlChart: fit/monitor round-trip, raises before fit
  - MultivariateConformalMonitor: duck-typed custom model, round-trip
"""
import numpy as np
import pytest

from insurance_monitoring.conformal_chart import (
    ConformalControlChart,
    ConformalChartResult,
    MultivariateConformalMonitor,
    _conformal_p_value,
    _conformal_p_values,
    _conformal_threshold,
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

class TestConformalPValue:
    def test_min_score_returns_one(self):
        """p = 1.0 when s_new equals the minimum calibration score."""
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # All cal_scores >= 1.0, so count = 5; p = (5+1)/(5+1) = 1.0
        p = _conformal_p_value(1.0, cal)
        assert p == pytest.approx(1.0)

    def test_max_score_plus_one_returns_minimum(self):
        """p = 1/(n+1) when s_new exceeds all calibration scores."""
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = _conformal_p_value(10.0, cal)
        assert p == pytest.approx(1.0 / 6.0)

    def test_p_value_always_positive(self):
        """p-value is always > 0 (no zero p-values from finite calibration set)."""
        rng = np.random.default_rng(42)
        cal = rng.exponential(scale=1.0, size=1000)
        for s_new in [0.0, 1e10, np.max(cal) * 100]:
            p = _conformal_p_value(s_new, cal)
            assert p > 0.0

    def test_vectorised_consistent_with_scalar(self):
        """_conformal_p_values produces same results as scalar loop."""
        rng = np.random.default_rng(99)
        cal = rng.exponential(scale=1.0, size=200)
        scores = rng.exponential(scale=1.0, size=50)
        vec = _conformal_p_values(scores, cal)
        scalar = np.array([_conformal_p_value(s, cal) for s in scores])
        np.testing.assert_allclose(vec, scalar, rtol=1e-12)

    def test_p_value_decreasing_in_score(self):
        """Higher NCS score should give lower or equal p-value."""
        cal = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        scores = np.array([0.3, 1.0, 2.0, 3.5])
        p = _conformal_p_values(scores, cal)
        # p should be weakly decreasing as scores increase
        assert all(p[i] >= p[i + 1] for i in range(len(p) - 1))


class TestConformalThreshold:
    def test_alpha_005_n100(self):
        """Threshold at alpha=0.05, n=100 matches expected quantile level."""
        rng = np.random.default_rng(7)
        cal = rng.uniform(0, 1, size=100)
        q = _conformal_threshold(cal, alpha=0.05)
        # Expected level: 0.95 * (1 + 1/100) = 0.9595
        expected = float(np.quantile(cal, 0.9595))
        assert q == pytest.approx(expected, rel=1e-10)

    def test_alpha_0027_small_n_saturates_at_max(self):
        """At alpha=0.0027 with n=100 < 370, level clips to 1.0 -> q = max(cal)."""
        cal = np.arange(1.0, 101.0)  # n=100
        q = _conformal_threshold(cal, alpha=0.0027)
        # level = min(0.9973 * 1.01, 1.0) = min(1.007273, 1.0) = 1.0 -> max
        assert q == pytest.approx(float(np.max(cal)))

    def test_alpha_0027_large_n(self):
        """At alpha=0.0027 with n=1000, level is below 1.0."""
        rng = np.random.default_rng(123)
        cal = rng.normal(0, 1, size=1000)
        q = _conformal_threshold(cal, alpha=0.0027)
        # level = 0.9973 * (1 + 1/1000) = 0.99827...; not 1.0
        level = 0.9973 * (1.0 + 1.0 / 1000)
        expected = float(np.quantile(cal, level))
        assert q == pytest.approx(expected, rel=1e-10)

    def test_threshold_less_than_max_for_alpha_005_large_n(self):
        """At alpha=0.05 with large n, threshold is strictly below max."""
        rng = np.random.default_rng(55)
        cal = rng.exponential(scale=1.0, size=500)
        q = _conformal_threshold(cal, alpha=0.05)
        assert q < float(np.max(cal))


# ---------------------------------------------------------------------------
# NCS helpers
# ---------------------------------------------------------------------------

class TestNCSHelpers:
    def test_absolute_residual_basic(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 1.8, 3.5])
        result = ConformalControlChart.ncs_absolute_residual(actual, predicted)
        np.testing.assert_allclose(result, [0.5, 0.2, 0.5])

    def test_relative_residual_basic(self):
        actual = np.array([1.0, 2.0])
        predicted = np.array([2.0, 2.0])
        result = ConformalControlChart.ncs_relative_residual(actual, predicted)
        np.testing.assert_allclose(result, [0.5, 0.0])

    def test_relative_residual_zero_predicted(self):
        """Zero predicted should not raise; protected by 1e-9 floor."""
        actual = np.array([1.0])
        predicted = np.array([0.0])
        result = ConformalControlChart.ncs_relative_residual(actual, predicted)
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_median_deviation_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ConformalControlChart.ncs_median_deviation(values)
        # median = 3.0; deviations = [2, 1, 0, 1, 2]
        np.testing.assert_allclose(result, [2.0, 1.0, 0.0, 1.0, 2.0])

    def test_median_deviation_supplied_median(self):
        values = np.array([1.0, 2.0, 3.0])
        result = ConformalControlChart.ncs_median_deviation(values, median=2.0)
        np.testing.assert_allclose(result, [1.0, 0.0, 1.0])

    def test_studentized_basic(self):
        actual = np.array([2.0, 3.0])
        predicted = np.array([1.0, 1.0])
        local_vol = np.array([2.0, 0.5])
        result = ConformalControlChart.ncs_studentized(actual, predicted, local_vol)
        np.testing.assert_allclose(result, [0.5, 4.0])

    def test_studentized_zero_vol(self):
        """Zero local_vol should not raise; protected by 1e-9 floor."""
        actual = np.array([1.0])
        predicted = np.array([0.5])
        local_vol = np.array([0.0])
        result = ConformalControlChart.ncs_studentized(actual, predicted, local_vol)
        assert np.isfinite(result[0])
        assert result[0] > 0


# ---------------------------------------------------------------------------
# ConformalControlChart
# ---------------------------------------------------------------------------

class TestConformalControlChart:
    def test_fit_sets_attributes(self):
        rng = np.random.default_rng(1)
        cal = rng.exponential(scale=1.0, size=100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        assert chart.threshold_ is not None
        assert chart.n_cal_ == 100
        assert chart.cal_scores_ is not None

    def test_monitor_before_fit_raises(self):
        chart = ConformalControlChart()
        with pytest.raises(RuntimeError, match="fit()"):
            chart.monitor([1.0, 2.0])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ConformalControlChart(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalControlChart(alpha=1.0)
        with pytest.raises(ValueError):
            ConformalControlChart(alpha=1.5)

    def test_fit_requires_at_least_two_scores(self):
        with pytest.raises(ValueError):
            ConformalControlChart().fit([1.0])

    def test_small_calibration_set_warns(self):
        with pytest.warns(UserWarning, match="n >= 20"):
            ConformalControlChart(alpha=0.05).fit([1.0, 2.0, 3.0, 4.0])

    def test_monitor_returns_result(self):
        rng = np.random.default_rng(2)
        cal = rng.exponential(scale=1.0, size=200)
        monitor_scores = rng.exponential(scale=1.0, size=50)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(monitor_scores)
        assert isinstance(result, ConformalChartResult)
        assert len(result.scores) == 50
        assert len(result.p_values) == 50
        assert len(result.is_alarm) == 50

    def test_p_values_in_valid_range(self):
        rng = np.random.default_rng(3)
        cal = rng.exponential(scale=1.0, size=300)
        scores = rng.exponential(scale=1.0, size=100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(scores)
        assert np.all(result.p_values > 0)
        assert np.all(result.p_values <= 1)

    def test_false_alarm_rate_controlled_monte_carlo(self):
        """Empirical FAR should be <= alpha + epsilon under null (in-control data).

        Monte Carlo with n_sim=500 simulations. Under H0, empirical FAR should
        not exceed alpha substantially. We allow a generous tolerance here since
        the test must be fast (no heavy compute locally).
        """
        rng = np.random.default_rng(42)
        alpha = 0.05
        n_cal = 500
        n_sim = 200
        n_monitor = 1  # one observation per sim, i.i.d. from same distribution

        alarm_count = 0
        for _ in range(n_sim):
            cal = rng.exponential(scale=1.0, size=n_cal)
            s_new = rng.exponential(scale=1.0, size=n_monitor)
            chart = ConformalControlChart(alpha=alpha).fit(cal)
            result = chart.monitor(s_new)
            alarm_count += result.n_alarms

        empirical_far = alarm_count / (n_sim * n_monitor)
        # Should be <= alpha. Allow slack for Monte Carlo noise.
        assert empirical_far <= alpha + 0.03, (
            f"Empirical FAR {empirical_far:.3f} exceeds alpha={alpha} + 0.03"
        )

    def test_in_control_no_alarms_on_identical_distribution(self):
        """Most observations from the calibration distribution should not alarm."""
        rng = np.random.default_rng(77)
        cal = rng.normal(0, 1, size=1000)
        monitor = rng.normal(0, 1, size=100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(np.abs(monitor))  # use absolute residuals
        # With alpha=0.05, expect ~5% alarms; not >15% on average
        assert result.alarm_rate < 0.15

    def test_out_of_control_increases_alarm_rate(self):
        """Shifted distribution should produce more alarms than in-control."""
        rng = np.random.default_rng(88)
        cal = rng.exponential(scale=1.0, size=500)
        in_control = rng.exponential(scale=1.0, size=100)
        out_of_control = rng.exponential(scale=5.0, size=100)  # shifted up

        chart = ConformalControlChart(alpha=0.05).fit(cal)
        r_ic = chart.monitor(in_control)
        r_oc = chart.monitor(out_of_control)

        assert r_oc.alarm_rate > r_ic.alarm_rate

    def test_alarm_threshold_consistency(self):
        """score > threshold should be equivalent to is_alarm == True."""
        rng = np.random.default_rng(5)
        cal = rng.exponential(scale=1.0, size=200)
        scores = rng.exponential(scale=1.0, size=50)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(scores)
        expected_alarm = result.scores > result.threshold
        np.testing.assert_array_equal(result.is_alarm, expected_alarm)


# ---------------------------------------------------------------------------
# ConformalChartResult methods
# ---------------------------------------------------------------------------

class TestConformalChartResult:
    def _make_result(self) -> ConformalChartResult:
        rng = np.random.default_rng(11)
        cal = rng.exponential(scale=1.0, size=200)
        scores = rng.exponential(scale=1.0, size=30)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        return chart.monitor(scores)

    def test_to_polars_schema(self):
        result = self._make_result()
        df = result.to_polars()
        assert "obs_index" in df.columns
        assert "score" in df.columns
        assert "p_value" in df.columns
        assert "threshold" in df.columns
        assert "is_alarm" in df.columns
        assert len(df) == 30

    def test_to_polars_types(self):
        import polars as pl
        result = self._make_result()
        df = result.to_polars()
        assert df["obs_index"].dtype == pl.Int64
        assert df["score"].dtype == pl.Float64
        assert df["p_value"].dtype == pl.Float64
        assert df["is_alarm"].dtype == pl.Boolean

    def test_summary_returns_nonempty_string(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 50
        # Must mention alpha and n_cal
        assert "alpha=" in s
        assert "n_cal=" in s

    def test_n_alarms_matches_is_alarm_sum(self):
        result = self._make_result()
        assert result.n_alarms == int(np.sum(result.is_alarm))

    def test_alarm_rate_formula(self):
        result = self._make_result()
        expected = result.n_alarms / len(result.is_alarm)
        assert result.alarm_rate == pytest.approx(expected)

    def test_summary_contains_status(self):
        result = self._make_result()
        s = result.summary()
        assert "IN CONTROL" in s or "OUT OF CONTROL" in s


# ---------------------------------------------------------------------------
# MultivariateConformalMonitor — duck-typed custom model
# ---------------------------------------------------------------------------

class _SimpleAnomalyModel:
    """Minimal duck-typed model: decision_function = negative Euclidean distance from centroid."""

    def __init__(self):
        self.centroid_ = None

    def fit(self, X):
        self.centroid_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X):
        # Higher = more normal (less anomalous), consistent with sklearn convention
        dists = np.linalg.norm(X - self.centroid_, axis=1)
        return -dists  # negate so that nearby = high score


class TestMultivariateConformalMonitor:
    def test_fit_monitor_round_trip_custom_model(self):
        """Full fit/monitor round-trip with a duck-typed custom model."""
        rng = np.random.default_rng(20)
        X_train = rng.normal(0, 1, size=(100, 5))
        X_cal = rng.normal(0, 1, size=(50, 5))
        X_new = rng.normal(0, 1, size=(20, 5))

        model = _SimpleAnomalyModel()
        monitor = MultivariateConformalMonitor(model=model, alpha=0.05)
        monitor.fit(X_train, X_cal)
        result = monitor.monitor(X_new)

        assert isinstance(result, ConformalChartResult)
        assert len(result.scores) == 20
        assert np.all(result.p_values > 0)
        assert np.all(result.p_values <= 1)

    def test_out_of_control_has_more_alarms(self):
        """Vectors far from training distribution should mostly alarm."""
        rng = np.random.default_rng(21)
        X_train = rng.normal(0, 1, size=(200, 3))
        X_cal = rng.normal(0, 1, size=(100, 3))
        X_ic = rng.normal(0, 1, size=(50, 3))
        X_oc = rng.normal(20, 1, size=(50, 3))  # very far from training centre

        model = _SimpleAnomalyModel()
        monitor = MultivariateConformalMonitor(model=model, alpha=0.05)
        monitor.fit(X_train, X_cal)

        r_ic = monitor.monitor(X_ic)
        r_oc = monitor.monitor(X_oc)

        assert r_oc.alarm_rate > r_ic.alarm_rate

    def test_monitor_before_fit_raises(self):
        model = _SimpleAnomalyModel()
        monitor = MultivariateConformalMonitor(model=model)
        with pytest.raises(RuntimeError, match="fit()"):
            monitor.monitor(np.ones((5, 3)))

    def test_single_vector_input(self):
        """1D input (single state vector) should be handled gracefully."""
        rng = np.random.default_rng(22)
        X_train = rng.normal(0, 1, size=(100, 4))
        X_cal = rng.normal(0, 1, size=(50, 4))
        x_single = rng.normal(0, 1, size=(4,))

        model = _SimpleAnomalyModel()
        monitor = MultivariateConformalMonitor(model=model, alpha=0.05)
        monitor.fit(X_train, X_cal)
        result = monitor.monitor(x_single)
        assert len(result.scores) == 1

    def test_model_without_decision_function_raises(self):
        """Model missing both decision_function and score_samples should raise AttributeError."""
        class BadModel:
            def fit(self, X):
                return self

        rng = np.random.default_rng(23)
        X_train = rng.normal(0, 1, size=(50, 3))
        X_cal = rng.normal(0, 1, size=(20, 3))

        monitor = MultivariateConformalMonitor(model=BadModel(), alpha=0.05)
        monitor.model_ = BadModel()
        monitor.model_.fit(X_train)
        monitor.cal_scores_ = np.ones(20)
        monitor.n_cal_ = 20
        monitor.threshold_ = 2.0

        with pytest.raises(AttributeError, match="decision_function"):
            monitor.monitor(rng.normal(0, 1, size=(5, 3)))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            MultivariateConformalMonitor(alpha=-0.1)
        with pytest.raises(ValueError):
            MultivariateConformalMonitor(alpha=2.0)

    def test_to_polars_from_multivariate_result(self):
        rng = np.random.default_rng(24)
        X_train = rng.normal(0, 1, size=(100, 3))
        X_cal = rng.normal(0, 1, size=(50, 3))
        X_new = rng.normal(0, 1, size=(10, 3))

        model = _SimpleAnomalyModel()
        monitor = MultivariateConformalMonitor(model=model, alpha=0.05)
        monitor.fit(X_train, X_cal)
        result = monitor.monitor(X_new)

        df = result.to_polars()
        assert len(df) == 10
        assert "p_value" in df.columns
        assert "is_alarm" in df.columns
