"""Tests for v0.6.0 discrimination additions: GiniDriftBootstrapTest, GiniBootstrapResult."""

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.discrimination import (
    gini_coefficient,
    GiniBootstrapResult,
    GiniDriftBootstrapTest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int, seed: int = 0):
    """Return (actual, predicted) arrays for n policies."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.05, 0.20, n)
    act = rng.poisson(pred).astype(float)
    return act, pred


# ---------------------------------------------------------------------------
# T01 — instantiation_no_fit
# ---------------------------------------------------------------------------

class TestInstantiationNoFit:
    """T01: _result is None before test() is called (lazy evaluation)."""

    def test_result_is_none_before_test(self):
        act, pred = _make_data(500, seed=1)
        t = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
        )
        assert t._result is None


# ---------------------------------------------------------------------------
# T02 — test_returns_gini_bootstrap_result
# ---------------------------------------------------------------------------

class TestTestReturnsGiniBootstrapResult:
    """T02: test() returns a GiniBootstrapResult instance."""

    def test_returns_correct_type(self):
        act, pred = _make_data(500, seed=2)
        t = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
            random_state=2,
        )
        result = t.test()
        assert isinstance(result, GiniBootstrapResult)


# ---------------------------------------------------------------------------
# T03 — test_is_idempotent
# ---------------------------------------------------------------------------

class TestTestIsIdempotent:
    """T03: Repeated test() calls return the same object (caching)."""

    def test_same_object_on_second_call(self):
        act, pred = _make_data(500, seed=3)
        t = GiniDriftBootstrapTest(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
            random_state=3,
        )
        r1 = t.test()
        r2 = t.test()
        assert r1 is r2


# ---------------------------------------------------------------------------
# T04 — boot_replicates_shape
# ---------------------------------------------------------------------------

class TestBootReplicatesShape:
    """T04: boot_replicates has the right length and valid Gini range."""

    def test_replicates_length_and_range(self):
        act, pred = _make_data(500, seed=4)
        n_boot = 80
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=n_boot,
            random_state=4,
        )
        result = t.test()
        assert len(result.boot_replicates) == n_boot
        assert all(isinstance(v, (float, np.floating)) for v in result.boot_replicates)
        assert np.all(result.boot_replicates > -1.0)
        assert np.all(result.boot_replicates < 1.0)


# ---------------------------------------------------------------------------
# T05 — no_drift_large_pvalue
# ---------------------------------------------------------------------------

class TestNoDriftLargePvalue:
    """T05: No drift scenario — z small, not significant, gini_change ~ 0."""

    def test_no_drift(self):
        # Use a large n to reduce sampling variance
        rng = np.random.default_rng(5)
        n = 3000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        training_gini = gini_coefficient(act, pred)

        t = GiniDriftBootstrapTest(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=200,
            random_state=5,
        )
        result = t.test()
        assert abs(result.z_statistic) < 2.0, (
            f"z should be near 0 for identical data, got {result.z_statistic:.3f}"
        )
        assert result.significant is False
        assert result.gini_change == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# T06 — severe_drift_detected
# ---------------------------------------------------------------------------

class TestSevereDriftDetected:
    """T06: Large drift (training 0.50, monitor ~0) detected as significant."""

    def test_severe_drift(self):
        rng = np.random.default_rng(6)
        n = 2000
        # Random predictions -> Gini near zero
        pred_random = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(rng.uniform(0.05, 0.20, n)).astype(float)

        t = GiniDriftBootstrapTest(
            training_gini=0.50,
            monitor_actual=act,
            monitor_predicted=pred_random,
            n_bootstrap=200,
            alpha=0.32,
            random_state=6,
        )
        result = t.test()
        assert result.gini_change < 0, "Monitor Gini should be below training Gini"
        assert result.significant is True, (
            f"Should detect drift: training=0.50, monitor={result.monitor_gini:.3f}"
        )


# ---------------------------------------------------------------------------
# T07 — gini_change_arithmetic
# ---------------------------------------------------------------------------

class TestGiniChangeArithmetic:
    """T07: gini_change == monitor_gini - training_gini exactly."""

    def test_gini_change_arithmetic(self):
        act, pred = _make_data(500, seed=7)
        t = GiniDriftBootstrapTest(
            training_gini=0.42,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
            random_state=7,
        )
        result = t.test()
        expected = result.monitor_gini - result.training_gini
        assert result.gini_change == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# T08 — ci_coverage_monotone
# ---------------------------------------------------------------------------

class TestCICoverageMonotone:
    """T08: Higher confidence level produces wider CI."""

    def test_ci_width_monotone_with_confidence(self):
        act, pred = _make_data(1000, seed=8)
        t90 = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=200,
            confidence_level=0.90,
            random_state=8,
        )
        t99 = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=200,
            confidence_level=0.99,
            random_state=8,
        )
        r90 = t90.test()
        r99 = t99.test()
        assert r90.ci_lower >= r99.ci_lower, "90% lower bound should be >= 99% lower bound"
        assert r90.ci_upper <= r99.ci_upper, "90% upper bound should be <= 99% upper bound"


# ---------------------------------------------------------------------------
# T09 — ci_change_contains_point_estimate
# ---------------------------------------------------------------------------

class TestCIChangeContainsPointEstimate:
    """T09: ci_change_lower <= gini_change <= ci_change_upper."""

    def test_ci_change_contains_gini_change(self):
        act, pred = _make_data(500, seed=9)
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=9,
        )
        result = t.test()
        assert result.ci_change_lower <= result.gini_change <= result.ci_change_upper, (
            f"gini_change={result.gini_change:.4f} not in "
            f"[{result.ci_change_lower:.4f}, {result.ci_change_upper:.4f}]"
        )


# ---------------------------------------------------------------------------
# T10 — se_bootstrap_is_sd_of_replicates
# ---------------------------------------------------------------------------

class TestSEBootstrapIsSDOfReplicates:
    """T10: se_bootstrap == std(boot_replicates, ddof=1) exactly."""

    def test_se_matches_std_of_replicates(self):
        act, pred = _make_data(600, seed=10)
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
            random_state=10,
        )
        result = t.test()
        expected_se = float(np.std(result.boot_replicates, ddof=1))
        assert result.se_bootstrap == pytest.approx(expected_se, abs=1e-10)


# ---------------------------------------------------------------------------
# T11 — exposure_weighted_accepted
# ---------------------------------------------------------------------------

class TestExposureWeightedAccepted:
    """T11: Exposure weights accepted; se_bootstrap is positive."""

    def test_exposure_weighted(self):
        rng = np.random.default_rng(11)
        n = 600
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        exp = rng.uniform(0.1, 2.0, n)

        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            monitor_exposure=exp,
            n_bootstrap=80,
            random_state=11,
        )
        result = t.test()
        assert isinstance(result, GiniBootstrapResult)
        assert result.se_bootstrap > 0


# ---------------------------------------------------------------------------
# T12 — polars_series_input
# ---------------------------------------------------------------------------

class TestPolarsSeriesInput:
    """T12: Polars Series accepted for actual and predicted."""

    def test_polars_series(self):
        rng = np.random.default_rng(12)
        n = 400
        pred = pl.Series(rng.uniform(0.05, 0.20, n).tolist())
        act = pl.Series(rng.poisson(0.10, n).astype(float).tolist())

        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
            random_state=12,
        )
        result = t.test()
        assert isinstance(result, GiniBootstrapResult)


# ---------------------------------------------------------------------------
# T13 — small_n_warning
# ---------------------------------------------------------------------------

class TestSmallNWarning:
    """T13: n=100 triggers UserWarning but still returns result."""

    def test_small_n_warns(self):
        rng = np.random.default_rng(13)
        n = 100
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
            random_state=13,
        )
        with pytest.warns(UserWarning, match="Small sample"):
            result = t.test()
        assert isinstance(result, GiniBootstrapResult)


# ---------------------------------------------------------------------------
# T14 — too_small_n_raises
# ---------------------------------------------------------------------------

class TestTooSmallNRaises:
    """T14: n=30 raises ValueError."""

    def test_too_small_n_raises(self):
        rng = np.random.default_rng(14)
        n = 30
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
            random_state=14,
        )
        with pytest.raises(ValueError, match="Sample too small"):
            t.test()


# ---------------------------------------------------------------------------
# T15 — plot_returns_axes
# ---------------------------------------------------------------------------

class TestPlotReturnsAxes:
    """T15: plot() returns matplotlib Axes with lines and legend."""

    def test_plot_returns_axes(self):
        import matplotlib.axes

        act, pred = _make_data(500, seed=15)
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=80,
            random_state=15,
        )
        ax = t.plot()
        assert isinstance(ax, matplotlib.axes.Axes)
        # Should have at least 2 vertical lines (training_gini and monitor_gini)
        assert len(ax.lines) >= 2
        # Legend should be populated
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) > 0


# ---------------------------------------------------------------------------
# Additional: constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    """Constructor raises ValueError for out-of-range parameters."""

    def test_training_gini_below_minus_one_raises(self):
        act, pred = _make_data(200, seed=20)
        with pytest.raises(ValueError, match="training_gini"):
            GiniDriftBootstrapTest(training_gini=-1.5, monitor_actual=act, monitor_predicted=pred)

    def test_training_gini_above_one_raises(self):
        act, pred = _make_data(200, seed=21)
        with pytest.raises(ValueError, match="training_gini"):
            GiniDriftBootstrapTest(training_gini=1.0, monitor_actual=act, monitor_predicted=pred)

    def test_n_bootstrap_below_50_raises(self):
        act, pred = _make_data(200, seed=22)
        with pytest.raises(ValueError, match="n_bootstrap"):
            GiniDriftBootstrapTest(
                training_gini=0.40,
                monitor_actual=act,
                monitor_predicted=pred,
                n_bootstrap=10,
            )

    def test_confidence_level_out_of_range_raises(self):
        act, pred = _make_data(200, seed=23)
        with pytest.raises(ValueError, match="confidence_level"):
            GiniDriftBootstrapTest(
                training_gini=0.40,
                monitor_actual=act,
                monitor_predicted=pred,
                confidence_level=1.5,
            )

    def test_alpha_out_of_range_raises(self):
        act, pred = _make_data(200, seed=24)
        with pytest.raises(ValueError, match="alpha"):
            GiniDriftBootstrapTest(
                training_gini=0.40,
                monitor_actual=act,
                monitor_predicted=pred,
                alpha=0.0,
            )


# ---------------------------------------------------------------------------
# Additional: to_dict excludes boot_replicates, includes boot_replicates_mean
# ---------------------------------------------------------------------------

class TestToDict:
    """GiniBootstrapResult.to_dict() contract."""

    def test_to_dict_excludes_boot_replicates_array(self):
        act, pred = _make_data(300, seed=30)
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=60,
            random_state=30,
        )
        result = t.test()
        d = result.to_dict()
        assert "boot_replicates" not in d
        assert "boot_replicates_mean" in d
        assert isinstance(d["boot_replicates_mean"], float)

    def test_to_dict_contains_all_scalar_fields(self):
        act, pred = _make_data(300, seed=31)
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=60,
            random_state=31,
        )
        result = t.test()
        d = result.to_dict()
        for key in [
            "z_statistic", "p_value", "training_gini", "monitor_gini",
            "gini_change", "se_bootstrap", "confidence_level",
            "ci_lower", "ci_upper", "ci_change_lower", "ci_change_upper",
            "n_bootstrap", "n_obs", "significant", "alpha",
        ]:
            assert key in d, f"Expected key '{key}' in to_dict() output"


# ---------------------------------------------------------------------------
# Additional: summary() output
# ---------------------------------------------------------------------------

class TestSummary:
    """summary() returns a non-empty string with expected substrings."""

    def test_summary_is_string_with_key_content(self):
        act, pred = _make_data(500, seed=40)
        t = GiniDriftBootstrapTest(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=80,
            random_state=40,
        )
        s = t.summary()
        assert isinstance(s, str)
        assert "Gini Drift Bootstrap Test" in s
        assert "Monitor period" in s
        assert "Training Gini" in s
        assert "arXiv:2510.04556" in s
