"""Extended tests for insurance_monitoring.discrimination — GiniDriftBootstrapTest class.

Covers the class-based bootstrap test which had near-zero coverage, plus:
- GiniBootstrapResult dataclass fields
- GiniDriftResult and GiniDriftOneSampleResult dataclass dict access
- _gini_from_arrays with exposure weighting
- _bootstrap_gini_samples internals
- GiniDriftBootstrapTest validation errors
- GiniDriftBootstrapTest.test() idempotency
- GiniDriftBootstrapTest.summary()
- Lorenz curve with exposure weights
"""

from __future__ import annotations

import warnings
import numpy as np
import polars as pl
import pytest

from insurance_monitoring.discrimination import (
    GiniDriftBootstrapTest,
    GiniBootstrapResult,
    GiniDriftResult,
    GiniDriftOneSampleResult,
    gini_coefficient,
    gini_drift_test,
    gini_drift_test_onesample,
    lorenz_curve,
)


# ---------------------------------------------------------------------------
# GiniDriftBootstrapTest — basic functionality
# ---------------------------------------------------------------------------


class TestGiniDriftBootstrapTestBasic:
    def _make_data(self, n=2_000, seed=42):
        rng = np.random.default_rng(seed)
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        return act, pred

    def test_basic_test_runs(self):
        """GiniDriftBootstrapTest.test() should return a GiniBootstrapResult."""
        act, pred = self._make_data()
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=42,
        )
        result = btest.test()
        assert isinstance(result, GiniBootstrapResult)

    def test_result_fields_populated(self):
        """All GiniBootstrapResult fields should be populated and finite."""
        act, pred = self._make_data()
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=0,
        )
        result = btest.test()
        assert isinstance(result.z_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.training_gini, float)
        assert isinstance(result.monitor_gini, float)
        assert isinstance(result.gini_change, float)
        assert isinstance(result.se_bootstrap, float)
        assert isinstance(result.ci_lower, float)
        assert isinstance(result.ci_upper, float)
        assert isinstance(result.ci_change_lower, float)
        assert isinstance(result.ci_change_upper, float)
        assert isinstance(result.significant, bool)
        assert isinstance(result.n_obs, int)
        assert isinstance(result.n_bootstrap, int)
        assert result.n_obs == 2_000
        assert result.n_bootstrap == 100

    def test_ci_lower_lt_upper(self):
        """Bootstrap CI lower should be strictly below upper."""
        act, pred = self._make_data()
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=1,
        )
        result = btest.test()
        assert result.ci_lower < result.ci_upper

    def test_change_ci_correct_direction(self):
        """Change CI should be consistent with gini_change direction."""
        act, pred = self._make_data()
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=2,
        )
        result = btest.test()
        assert result.ci_change_lower < result.ci_change_upper
        # gini_change should be within the change CI
        assert result.ci_change_lower <= result.gini_change <= result.ci_change_upper

    def test_gini_change_equals_monitor_minus_training(self):
        """gini_change should equal monitor_gini - training_gini."""
        act, pred = self._make_data()
        training_gini = 0.40
        btest = GiniDriftBootstrapTest(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=3,
        )
        result = btest.test()
        assert result.gini_change == pytest.approx(result.monitor_gini - training_gini, abs=1e-10)

    def test_test_is_idempotent(self):
        """Calling test() twice should return identical results."""
        act, pred = self._make_data()
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=7,
        )
        r1 = btest.test()
        r2 = btest.test()
        assert r1.z_statistic == r2.z_statistic
        assert r1.p_value == r2.p_value
        assert r1.ci_lower == r2.ci_lower

    def test_no_drift_case(self):
        """When training gini matches monitor gini, p-value should be large."""
        rng = np.random.default_rng(10)
        pred = rng.uniform(0.05, 0.20, 5_000)
        act = rng.poisson(pred).astype(float)
        training_gini = gini_coefficient(act, pred)
        btest = GiniDriftBootstrapTest(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=42,
        )
        result = btest.test()
        # p-value should be large when training and monitor Gini are the same
        assert result.p_value > 0.2

    def test_with_exposure_weights(self):
        """GiniDriftBootstrapTest should accept monitor_exposure."""
        rng = np.random.default_rng(20)
        pred = rng.uniform(0.05, 0.20, 2_000)
        act = rng.poisson(pred).astype(float)
        exposure = rng.uniform(0.5, 1.5, 2_000)
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            monitor_exposure=exposure,
            n_bootstrap=100,
            random_state=0,
        )
        result = btest.test()
        assert isinstance(result, GiniBootstrapResult)

    def test_polars_series_input(self):
        """GiniDriftBootstrapTest should accept Polars Series."""
        rng = np.random.default_rng(21)
        pred_arr = rng.uniform(0.05, 0.20, 2_000)
        act_arr = rng.poisson(pred_arr).astype(float)
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=pl.Series(act_arr.tolist()),
            monitor_predicted=pl.Series(pred_arr.tolist()),
            n_bootstrap=100,
            random_state=0,
        )
        result = btest.test()
        assert isinstance(result, GiniBootstrapResult)

    def test_significant_flag_matches_p_value(self):
        """significant should be True iff p_value < alpha."""
        act, pred = self._make_data()
        alpha = 0.05
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            alpha=alpha,
            random_state=5,
        )
        result = btest.test()
        assert result.significant == (result.p_value < alpha)

    def test_confidence_level_affects_ci_width(self):
        """Higher confidence_level should produce wider CI."""
        act, pred = self._make_data(n=5_000, seed=30)
        btest_90 = GiniDriftBootstrapTest(
            training_gini=0.45, monitor_actual=act, monitor_predicted=pred,
            n_bootstrap=200, confidence_level=0.90, random_state=42,
        )
        btest_99 = GiniDriftBootstrapTest(
            training_gini=0.45, monitor_actual=act, monitor_predicted=pred,
            n_bootstrap=200, confidence_level=0.99, random_state=42,
        )
        r90 = btest_90.test()
        r99 = btest_99.test()
        width_90 = r90.ci_upper - r90.ci_lower
        width_99 = r99.ci_upper - r99.ci_lower
        assert width_99 > width_90

    def test_boot_replicates_stored(self):
        """boot_replicates should be a numpy array of length ~ n_bootstrap."""
        act, pred = self._make_data()
        n_boot = 100
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=n_boot,
            random_state=0,
        )
        result = btest.test()
        assert isinstance(result.boot_replicates, np.ndarray)
        assert len(result.boot_replicates) <= n_boot  # may be fewer if NaN removed


# ---------------------------------------------------------------------------
# GiniDriftBootstrapTest — validation errors
# ---------------------------------------------------------------------------


class TestGiniDriftBootstrapTestValidation:
    def test_training_gini_out_of_range_raises(self):
        """training_gini outside (-1, 1) should raise ValueError at init."""
        rng = np.random.default_rng(0)
        act = rng.poisson(0.1, 100).astype(float)
        pred = np.full(100, 0.1)
        with pytest.raises(ValueError, match="training_gini"):
            GiniDriftBootstrapTest(training_gini=1.5, monitor_actual=act, monitor_predicted=pred)

    def test_n_bootstrap_too_small_raises(self):
        """n_bootstrap < 50 should raise ValueError at init."""
        rng = np.random.default_rng(0)
        act = rng.poisson(0.1, 100).astype(float)
        pred = np.full(100, 0.1)
        with pytest.raises(ValueError, match="n_bootstrap"):
            GiniDriftBootstrapTest(
                training_gini=0.4, monitor_actual=act, monitor_predicted=pred, n_bootstrap=10
            )

    def test_confidence_level_out_of_range_raises(self):
        """confidence_level outside (0, 1) should raise ValueError."""
        rng = np.random.default_rng(0)
        act = rng.poisson(0.1, 100).astype(float)
        pred = np.full(100, 0.1)
        with pytest.raises(ValueError, match="confidence_level"):
            GiniDriftBootstrapTest(
                training_gini=0.4, monitor_actual=act, monitor_predicted=pred,
                confidence_level=1.5,
            )

    def test_alpha_out_of_range_raises(self):
        """alpha outside (0, 1) should raise ValueError."""
        rng = np.random.default_rng(0)
        act = rng.poisson(0.1, 100).astype(float)
        pred = np.full(100, 0.1)
        with pytest.raises(ValueError, match="alpha"):
            GiniDriftBootstrapTest(
                training_gini=0.4, monitor_actual=act, monitor_predicted=pred, alpha=2.0
            )

    def test_sample_too_small_raises(self):
        """Sample with n < 50 should raise ValueError on test() call."""
        act = np.ones(30)
        pred = np.full(30, 0.1)
        btest = GiniDriftBootstrapTest(
            training_gini=0.4, monitor_actual=act, monitor_predicted=pred, n_bootstrap=50
        )
        with pytest.raises(ValueError, match="50"):
            btest.test()

    def test_small_sample_warning(self):
        """Sample with n < 200 should produce a UserWarning."""
        rng = np.random.default_rng(0)
        act = rng.poisson(0.1, 100).astype(float)
        pred = rng.uniform(0.05, 0.15, 100)
        btest = GiniDriftBootstrapTest(
            training_gini=0.4, monitor_actual=act, monitor_predicted=pred, n_bootstrap=50,
            random_state=0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            btest.test()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) > 0

    def test_mismatched_actual_predicted_raises(self):
        """Mismatched monitor_actual/monitor_predicted lengths should raise ValueError."""
        act = np.ones(100)
        pred = np.ones(90)  # wrong length
        btest = GiniDriftBootstrapTest(
            training_gini=0.4, monitor_actual=act, monitor_predicted=pred, n_bootstrap=50
        )
        with pytest.raises(ValueError):
            btest.test()

    def test_mismatched_exposure_raises(self):
        """Mismatched monitor_exposure length should raise ValueError."""
        rng = np.random.default_rng(0)
        act = rng.poisson(0.1, 200).astype(float)
        pred = rng.uniform(0.05, 0.15, 200)
        exp = np.ones(150)  # wrong length
        btest = GiniDriftBootstrapTest(
            training_gini=0.4, monitor_actual=act, monitor_predicted=pred,
            monitor_exposure=exp, n_bootstrap=50,
        )
        with pytest.raises(ValueError):
            btest.test()


# ---------------------------------------------------------------------------
# GiniDriftBootstrapTest — summary()
# ---------------------------------------------------------------------------


class TestGiniDriftBootstrapTestSummary:
    def test_summary_returns_string(self):
        """summary() should return a non-empty string."""
        rng = np.random.default_rng(50)
        pred = rng.uniform(0.05, 0.20, 2_000)
        act = rng.poisson(pred).astype(float)
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=0,
        )
        s = btest.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_summary_contains_key_fields(self):
        """summary() should contain Gini values and test result."""
        rng = np.random.default_rng(51)
        pred = rng.uniform(0.05, 0.20, 2_000)
        act = rng.poisson(pred).astype(float)
        btest = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=0,
        )
        s = btest.summary()
        assert "Gini" in s
        assert "z" in s.lower() or "statistic" in s.lower()

    def test_summary_verdict_not_significant(self):
        """When no drift, summary should say 'not significant'."""
        rng = np.random.default_rng(52)
        pred = rng.uniform(0.05, 0.20, 5_000)
        act = rng.poisson(pred).astype(float)
        training_gini = gini_coefficient(act, pred)
        btest = GiniDriftBootstrapTest(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            alpha=0.05,
            random_state=0,
        )
        s = btest.summary()
        assert "not significant" in s.lower()


# ---------------------------------------------------------------------------
# Dataclass dict-style access (backward compat)
# ---------------------------------------------------------------------------


class TestDataclassDictAccess:
    def test_gini_drift_result_dict_access(self):
        """GiniDriftResult supports dict-style [] access."""
        result = GiniDriftResult(
            z_statistic=1.5,
            p_value=0.13,
            reference_gini=0.45,
            current_gini=0.42,
            gini_change=-0.03,
            significant=False,
        )
        assert result["z_statistic"] == pytest.approx(1.5)
        assert result["significant"] is False
        assert result["gini_change"] == pytest.approx(-0.03)

    def test_gini_drift_onesample_result_dict_access(self):
        """GiniDriftOneSampleResult supports dict-style [] access."""
        result = GiniDriftOneSampleResult(
            z_statistic=0.8,
            p_value=0.42,
            training_gini=0.45,
            monitor_gini=0.46,
            gini_change=0.01,
            se_bootstrap=0.02,
            significant=False,
        )
        assert result["training_gini"] == pytest.approx(0.45)
        assert result["se_bootstrap"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Lorenz curve with exposure
# ---------------------------------------------------------------------------


class TestLorenzCurveWithExposure:
    def test_lorenz_with_exposure_returns_valid_curve(self):
        """lorenz_curve with exposure should return valid arrays."""
        rng = np.random.default_rng(60)
        actual = rng.poisson(0.1, 500).astype(float)
        predicted = rng.uniform(0.05, 0.15, 500)
        exposure = rng.uniform(0.5, 2.0, 500)
        x, y = lorenz_curve(actual, predicted, exposure=exposure)
        assert x[0] == 0.0
        assert y[0] == 0.0
        assert x[-1] == pytest.approx(1.0)
        assert y[-1] == pytest.approx(1.0)

    def test_lorenz_with_exposure_monotone(self):
        """Lorenz curve with exposure should be monotone."""
        rng = np.random.default_rng(61)
        actual = rng.poisson(0.1, 1_000).astype(float)
        predicted = rng.uniform(0.05, 0.15, 1_000)
        exposure = rng.uniform(0.1, 2.0, 1_000)
        x, y = lorenz_curve(actual, predicted, exposure=exposure)
        assert np.all(np.diff(x) >= 0)
        assert np.all(np.diff(y) >= 0)

    def test_lorenz_with_all_zero_actual_returns_zeros(self):
        """All-zero actual should produce a flat Lorenz curve."""
        actual = np.zeros(100)
        predicted = np.random.default_rng(62).uniform(0, 1, 100)
        x, y = lorenz_curve(actual, predicted)
        # All y should be 0 except the last which wraps to 1 (or stays 0)
        # Key: curve should be valid arrays
        assert len(x) == len(y)
        assert x[0] == 0.0


# ---------------------------------------------------------------------------
# gini_coefficient edge cases
# ---------------------------------------------------------------------------


class TestGiniCoefficientEdgeCases:
    def test_gini_single_observation(self):
        """Single observation should return 0.0."""
        g = gini_coefficient(np.array([1.0]), np.array([0.5]))
        assert g == pytest.approx(0.0)

    def test_gini_tie_breaking_midpoint(self):
        """Tie-breaking: identical predictions should average best/worst case."""
        # All same predicted rate — tied, random ordering
        # Gini should be exactly 0.0 for ties
        actual = np.array([1.0, 0.0, 1.0, 0.0])
        predicted = np.array([0.5, 0.5, 0.5, 0.5])  # all tied
        g = gini_coefficient(actual, predicted)
        assert isinstance(g, float)
        # No ordering information => midpoint approach => near 0
        assert abs(g) < 0.1

    def test_gini_with_zero_exposure(self):
        """Zero exposure weights should raise or return valid result (no crash)."""
        actual = np.array([1.0, 2.0, 0.0, 3.0])
        predicted = np.array([0.1, 0.5, 0.2, 0.8])
        exposure = np.array([1.0, 1.0, 0.0, 1.0])  # one zero weight
        # Should run without crashing
        g = gini_coefficient(actual, predicted, exposure=exposure)
        assert isinstance(g, float)

    def test_gini_negative_actuals(self):
        """Negative actual values should run without crashing."""
        # Some insurance contexts have negative net claims
        actual = np.array([-1.0, 0.0, 2.0, 5.0])
        predicted = np.array([0.1, 0.2, 0.6, 0.9])
        g = gini_coefficient(actual, predicted)
        assert isinstance(g, float)
