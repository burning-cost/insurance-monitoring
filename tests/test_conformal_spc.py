"""
Tests for insurance_monitoring.conformal_spc.

Coverage:
  - ConformedControlResult and ConformedMonitorResult dataclasses
  - Threshold computation matches manual calculation
  - All three score_fn options (absolute, relative, studentized)
  - In-control signal_rate ≈ alpha for normal, Poisson-like, and gamma-like data
  - Out-of-control (mean-shifted) has signal_rate >> alpha
  - ConformedProcessMonitor: multivariate in-control vs shifted test
  - Edge cases: n=1 calibration, alpha=0, alpha=1
  - Custom duck-typed detector
  - Both result dataclasses populated correctly
"""
import warnings

import numpy as np
import pytest

from insurance_monitoring.conformal_spc import (
    ConformedControlChart,
    ConformedControlResult,
    ConformedMonitorResult,
    ConformedProcessMonitor,
    _threshold_from_calibration,
    _conformal_p_values_vec,
    _score_absolute,
    _score_relative,
    _score_studentized,
)


# ---------------------------------------------------------------------------
# Score function tests
# ---------------------------------------------------------------------------

class TestScoreFunctions:
    def test_absolute_basic(self):
        values = np.array([1.0, 3.0, 5.0])
        result = _score_absolute(values, ref_median=3.0)
        np.testing.assert_allclose(result, [2.0, 0.0, 2.0])

    def test_relative_basic(self):
        values = np.array([1.0, 3.0, 5.0])
        result = _score_relative(values, ref_median=2.0)
        # denom = max(|2.0|, 1e-8) = 2.0
        np.testing.assert_allclose(result, [0.5, 0.5, 1.5])

    def test_relative_zero_median_uses_floor(self):
        """Zero median should use 1e-8 floor to avoid division by zero."""
        values = np.array([1.0, 2.0])
        result = _score_relative(values, ref_median=0.0)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_studentized_basic(self):
        values = np.array([1.0, 5.0])
        result = _score_studentized(values, ref_median=3.0, ref_mad=2.0)
        # |1 - 3| / 2 = 1.0; |5 - 3| / 2 = 1.0
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_studentized_zero_mad_uses_floor(self):
        """Zero MAD (constant series) should not raise; 1e-8 floor applied."""
        values = np.array([1.0, 2.0])
        result = _score_studentized(values, ref_median=1.5, ref_mad=0.0)
        assert np.all(np.isfinite(result))

    def test_absolute_non_negative(self):
        rng = np.random.default_rng(1)
        values = rng.normal(5, 2, size=100)
        result = _score_absolute(values, ref_median=5.0)
        assert np.all(result >= 0)

    def test_relative_non_negative(self):
        rng = np.random.default_rng(2)
        values = rng.gamma(2, 2, size=100)
        result = _score_relative(values, ref_median=4.0)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

class TestThresholdComputation:
    def test_matches_manual_quantile(self):
        rng = np.random.default_rng(10)
        cal = rng.exponential(1.0, size=100)
        alpha = 0.05
        q = _threshold_from_calibration(cal, alpha)
        level = min((1.0 - alpha) * (1.0 + 1.0 / 100), 1.0)
        expected = float(np.quantile(cal, level))
        assert q == pytest.approx(expected, rel=1e-10)

    def test_level_clips_at_one_for_3sigma(self):
        """alpha=0.0027 with n=100 < 370: level clips to 1.0 -> max(cal)."""
        cal = np.arange(1.0, 101.0)
        q = _threshold_from_calibration(cal, alpha=0.0027)
        assert q == pytest.approx(float(np.max(cal)))

    def test_large_n_3sigma_below_max(self):
        """alpha=0.0027 with n=1000: level < 1.0 -> threshold < max."""
        rng = np.random.default_rng(11)
        cal = rng.normal(0, 1, size=1000)
        q = _threshold_from_calibration(cal, alpha=0.0027)
        assert q < float(np.max(cal))

    def test_alpha_005_n200_below_max(self):
        rng = np.random.default_rng(12)
        cal = rng.exponential(1.0, size=200)
        q = _threshold_from_calibration(cal, alpha=0.05)
        assert q < float(np.max(cal))


# ---------------------------------------------------------------------------
# ConformedControlChart — construction and validation
# ---------------------------------------------------------------------------

class TestConformedControlChartInit:
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ConformedControlChart(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            ConformedControlChart(alpha=1.5)

    def test_invalid_score_fn_raises(self):
        with pytest.raises(ValueError, match="score_fn"):
            ConformedControlChart(score_fn="unknown")

    def test_valid_score_fns_accepted(self):
        for fn in ("absolute", "relative", "studentized"):
            chart = ConformedControlChart(score_fn=fn)
            assert chart.score_fn == fn

    def test_alpha_zero_accepted(self):
        chart = ConformedControlChart(alpha=0.0)
        assert chart.alpha == 0.0

    def test_alpha_one_accepted(self):
        chart = ConformedControlChart(alpha=1.0)
        assert chart.alpha == 1.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            ConformedControlChart().predict([1.0, 2.0])


# ---------------------------------------------------------------------------
# ConformedControlChart — fit()
# ---------------------------------------------------------------------------

class TestConformedControlChartFit:
    def test_fit_sets_internal_state(self):
        rng = np.random.default_rng(20)
        cal = rng.exponential(1.0, size=100)
        chart = ConformedControlChart(alpha=0.05).fit(cal)
        assert chart._threshold is not None
        assert chart._n_calibration == 100
        assert chart._ref_median is not None
        assert chart._cal_scores is not None

    def test_fit_requires_at_least_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConformedControlChart().fit([1.0])

    def test_fit_single_value_raises(self):
        with pytest.raises(ValueError):
            ConformedControlChart().fit(np.array([3.14]))

    def test_small_n_warns(self):
        with pytest.warns(UserWarning, match="n >= 20"):
            ConformedControlChart(alpha=0.05).fit([1.0, 2.0, 3.0, 4.0])

    def test_fit_returns_self(self):
        chart = ConformedControlChart()
        result = chart.fit(np.ones(30))
        assert result is chart


# ---------------------------------------------------------------------------
# ConformedControlChart — predict()
# ---------------------------------------------------------------------------

class TestConformedControlChartPredict:
    def test_result_type(self):
        rng = np.random.default_rng(30)
        cal = rng.exponential(1.0, size=200)
        chart = ConformedControlChart(alpha=0.05).fit(cal)
        result = chart.predict(rng.exponential(1.0, size=50))
        assert isinstance(result, ConformedControlResult)

    def test_result_fields_populated(self):
        rng = np.random.default_rng(31)
        cal = rng.exponential(1.0, size=200)
        test = rng.exponential(1.0, size=50)
        chart = ConformedControlChart(alpha=0.05).fit(cal)
        r = chart.predict(test)
        assert len(r.scores) == 50
        assert len(r.signals) == 50
        assert r.signals.dtype == bool
        assert 0.0 <= r.signal_rate <= 1.0
        assert r.alpha == 0.05
        assert r.n_calibration == 200
        assert isinstance(r.threshold, float)

    def test_signal_rate_matches_signals(self):
        rng = np.random.default_rng(32)
        cal = rng.exponential(1.0, size=200)
        test = rng.exponential(1.0, size=100)
        chart = ConformedControlChart(alpha=0.05).fit(cal)
        r = chart.predict(test)
        expected = r.signals.sum() / len(r.signals)
        assert r.signal_rate == pytest.approx(expected)

    def test_signals_consistent_with_threshold(self):
        """scores > threshold <=> signals == True."""
        rng = np.random.default_rng(33)
        cal = rng.exponential(1.0, size=200)
        test = rng.exponential(1.0, size=50)
        chart = ConformedControlChart(alpha=0.05).fit(cal)
        r = chart.predict(test)
        np.testing.assert_array_equal(r.signals, r.scores > r.threshold)

    def test_all_score_fns_run(self):
        rng = np.random.default_rng(34)
        cal = rng.normal(5.0, 1.0, size=100)
        test = rng.normal(5.0, 1.0, size=20)
        for fn in ("absolute", "relative", "studentized"):
            chart = ConformedControlChart(alpha=0.05, score_fn=fn).fit(cal)
            r = chart.predict(test)
            assert isinstance(r, ConformedControlResult)
            assert len(r.scores) == 20


# ---------------------------------------------------------------------------
# ConformedControlChart — statistical properties
# ---------------------------------------------------------------------------

class TestConformedControlChartStats:
    def test_incontrol_normal_signal_rate_near_alpha(self):
        """In-control normal data: signal_rate should be close to alpha."""
        rng = np.random.default_rng(40)
        alpha = 0.05
        cal = rng.normal(0, 1, size=500)
        test = rng.normal(0, 1, size=500)
        chart = ConformedControlChart(alpha=alpha, score_fn="absolute").fit(cal)
        r = chart.predict(test)
        # Conformal guarantee: signal_rate <= alpha. Allow Monte Carlo slack.
        assert r.signal_rate <= alpha + 0.05, (
            f"In-control signal_rate {r.signal_rate:.3f} exceeds alpha+0.05"
        )

    def test_incontrol_gamma_signal_rate_near_alpha(self):
        """In-control gamma data: signal_rate <= alpha + tolerance."""
        rng = np.random.default_rng(41)
        alpha = 0.05
        cal = rng.gamma(shape=2, scale=3, size=500)
        test = rng.gamma(shape=2, scale=3, size=500)
        chart = ConformedControlChart(alpha=alpha, score_fn="absolute").fit(cal)
        r = chart.predict(test)
        assert r.signal_rate <= alpha + 0.05

    def test_incontrol_poisson_signal_rate_near_alpha(self):
        """In-control Poisson-like (float) data."""
        rng = np.random.default_rng(42)
        alpha = 0.05
        cal = rng.poisson(lam=5, size=500).astype(float)
        test = rng.poisson(lam=5, size=500).astype(float)
        chart = ConformedControlChart(alpha=alpha, score_fn="absolute").fit(cal)
        r = chart.predict(test)
        assert r.signal_rate <= alpha + 0.06  # slightly more slack for discrete dist

    def test_outofcontrol_mean_shift_more_signals(self):
        """Mean-shifted test data should have higher signal_rate than in-control."""
        rng = np.random.default_rng(43)
        alpha = 0.05
        cal = rng.normal(0, 1, size=500)
        in_ctrl = rng.normal(0, 1, size=200)
        out_ctrl = rng.normal(5, 1, size=200)  # large mean shift

        chart = ConformedControlChart(alpha=alpha, score_fn="absolute").fit(cal)
        r_ic = chart.predict(in_ctrl)
        r_oc = chart.predict(out_ctrl)

        assert r_oc.signal_rate > r_ic.signal_rate
        assert r_oc.signal_rate > 0.5, (
            f"Expected strong signal for 5-sigma shift, got {r_oc.signal_rate:.3f}"
        )

    def test_outofcontrol_gamma_shift(self):
        """Gamma with larger scale should produce more signals than calibration scale."""
        rng = np.random.default_rng(44)
        cal = rng.gamma(2, scale=1, size=300)
        ic = rng.gamma(2, scale=1, size=100)
        oc = rng.gamma(2, scale=8, size=100)  # 8x scale shift
        chart = ConformedControlChart(alpha=0.05, score_fn="absolute").fit(cal)
        r_ic = chart.predict(ic)
        r_oc = chart.predict(oc)
        assert r_oc.signal_rate > r_ic.signal_rate

    def test_relative_score_fn_detects_shift(self):
        rng = np.random.default_rng(45)
        cal = rng.normal(10, 1, size=300)
        oc = rng.normal(15, 1, size=100)
        chart = ConformedControlChart(alpha=0.05, score_fn="relative").fit(cal)
        r = chart.predict(oc)
        assert r.signal_rate > 0.3

    def test_studentized_score_fn_detects_shift(self):
        rng = np.random.default_rng(46)
        cal = rng.normal(0, 1, size=300)
        oc = rng.normal(4, 1, size=100)
        chart = ConformedControlChart(alpha=0.05, score_fn="studentized").fit(cal)
        r = chart.predict(oc)
        assert r.signal_rate > 0.3


# ---------------------------------------------------------------------------
# Edge cases: alpha = 0 and alpha = 1
# ---------------------------------------------------------------------------

class TestConformedControlChartEdgeCases:
    def test_alpha_zero_no_signals(self):
        """alpha=0: threshold set to near-infinity, no observations should signal."""
        rng = np.random.default_rng(50)
        cal = rng.normal(0, 1, size=100)
        chart = ConformedControlChart(alpha=0.0, score_fn="absolute").fit(cal)
        test = rng.normal(0, 1, size=50)
        r = chart.predict(test)
        assert r.signal_rate == 0.0

    def test_alpha_one_all_signals(self):
        """alpha=1: threshold=0, all observations should signal."""
        rng = np.random.default_rng(51)
        cal = rng.normal(0, 1, size=100)
        chart = ConformedControlChart(alpha=1.0, score_fn="absolute").fit(cal)
        # Use values != median to ensure scores > 0 > threshold
        test = rng.normal(5, 1, size=50)
        r = chart.predict(test)
        assert r.signal_rate == 1.0

    def test_minimum_calibration_two_obs(self):
        """Two calibration observations: should fit without error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            chart = ConformedControlChart(alpha=0.05).fit([1.0, 2.0])
        r = chart.predict([1.5])
        assert isinstance(r, ConformedControlResult)

    def test_single_test_observation(self):
        rng = np.random.default_rng(52)
        cal = rng.normal(0, 1, size=50)
        chart = ConformedControlChart(alpha=0.05).fit(cal)
        r = chart.predict([1.0])
        assert len(r.scores) == 1
        assert len(r.signals) == 1


# ---------------------------------------------------------------------------
# ConformedProcessMonitor — construction and validation
# ---------------------------------------------------------------------------

class TestConformedProcessMonitorInit:
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ConformedProcessMonitor(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            ConformedProcessMonitor(alpha=1.5)

    def test_invalid_detector_string_raises(self):
        with pytest.raises(ValueError, match="detector"):
            ConformedProcessMonitor(detector="unsupported_method")

    def test_valid_detector_strings(self):
        for det in ("ocsvm", "isolation_forest"):
            m = ConformedProcessMonitor(detector=det)
            assert m.detector == det

    def test_predict_before_fit_raises(self):
        m = ConformedProcessMonitor()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.ones((5, 3)))

    def test_alpha_zero_accepted(self):
        m = ConformedProcessMonitor(alpha=0.0)
        assert m.alpha == 0.0

    def test_alpha_one_accepted(self):
        m = ConformedProcessMonitor(alpha=1.0)
        assert m.alpha == 1.0


# ---------------------------------------------------------------------------
# ConformedProcessMonitor — fit()
# ---------------------------------------------------------------------------

class TestConformedProcessMonitorFit:
    def _make_monitor(self) -> ConformedProcessMonitor:
        """Build a monitor with a fast custom duck-typed model."""
        return ConformedProcessMonitor(alpha=0.05, detector=_FastCentroidModel())

    def test_fit_requires_at_least_ten(self):
        with pytest.raises(ValueError, match="at least 10"):
            ConformedProcessMonitor(detector=_FastCentroidModel()).fit(
                np.ones((5, 3))
            )

    def test_fit_sets_internal_state(self):
        rng = np.random.default_rng(60)
        X = rng.normal(0, 1, size=(50, 4))
        monitor = self._make_monitor().fit(X)
        assert monitor._cal_scores is not None
        assert monitor._n_calibration is not None
        assert monitor._model is not None

    def test_fit_returns_self(self):
        rng = np.random.default_rng(61)
        X = rng.normal(0, 1, size=(50, 4))
        m = ConformedProcessMonitor(detector=_FastCentroidModel())
        result = m.fit(X)
        assert result is m

    def test_1d_input_handled(self):
        """1D calibration input (univariate) should be reshaped gracefully."""
        rng = np.random.default_rng(62)
        X = rng.normal(0, 1, size=50)
        m = ConformedProcessMonitor(detector=_FastCentroidModel()).fit(X)
        assert m._cal_scores is not None

    def test_small_cal_set_warns(self):
        """Small 80/20 split producing < 20 calibration points should warn."""
        rng = np.random.default_rng(63)
        X = rng.normal(0, 1, size=(15, 3))  # 20% of 15 = 3 cal points
        m = ConformedProcessMonitor(alpha=0.05, detector=_FastCentroidModel())
        with pytest.warns(UserWarning, match="calibration"):
            m.fit(X)


# ---------------------------------------------------------------------------
# ConformedProcessMonitor — predict()
# ---------------------------------------------------------------------------

class TestConformedProcessMonitorPredict:
    def _fitted_monitor(self, rng, n=100, d=5, alpha=0.05) -> ConformedProcessMonitor:
        X = rng.normal(0, 1, size=(n, d))
        return ConformedProcessMonitor(
            alpha=alpha, detector=_FastCentroidModel()
        ).fit(X)

    def test_result_type(self):
        rng = np.random.default_rng(70)
        m = self._fitted_monitor(rng)
        X_test = rng.normal(0, 1, size=(20, 5))
        r = m.predict(X_test)
        assert isinstance(r, ConformedMonitorResult)

    def test_result_fields_populated(self):
        rng = np.random.default_rng(71)
        m = self._fitted_monitor(rng, n=100, d=4)
        X_test = rng.normal(0, 1, size=(30, 4))
        r = m.predict(X_test)
        assert len(r.p_values) == 30
        assert len(r.signals) == 30
        assert r.signals.dtype == bool
        assert np.all(r.p_values > 0)
        assert np.all(r.p_values <= 1.0)
        assert 0.0 <= r.signal_rate <= 1.0
        assert r.alpha == 0.05
        assert r.n_calibration is not None

    def test_signal_rate_matches_signals(self):
        rng = np.random.default_rng(72)
        m = self._fitted_monitor(rng)
        X_test = rng.normal(0, 1, size=(50, 5))
        r = m.predict(X_test)
        expected = r.signals.sum() / len(r.signals)
        assert r.signal_rate == pytest.approx(expected)

    def test_1d_test_input_handled(self):
        """Single observation (1D array) should return result with 1 row."""
        rng = np.random.default_rng(73)
        m = self._fitted_monitor(rng, n=100, d=5)
        r = m.predict(rng.normal(0, 1, size=5))
        assert len(r.p_values) == 1
        assert len(r.signals) == 1

    def test_alpha_zero_no_signals(self):
        rng = np.random.default_rng(74)
        m = self._fitted_monitor(rng, alpha=0.0)
        X_test = rng.normal(100, 1, size=(20, 5))  # very out-of-control
        r = m.predict(X_test)
        assert r.signal_rate == 0.0

    def test_alpha_one_all_signals(self):
        rng = np.random.default_rng(75)
        m = self._fitted_monitor(rng, alpha=1.0)
        X_test = rng.normal(0, 1, size=(20, 5))
        r = m.predict(X_test)
        assert r.signal_rate == 1.0


# ---------------------------------------------------------------------------
# ConformedProcessMonitor — statistical properties
# ---------------------------------------------------------------------------

class TestConformedProcessMonitorStats:
    def test_incontrol_multivariate_normal_signal_rate_near_alpha(self):
        """In-control multivariate normal: signal_rate should be <= alpha + slack."""
        rng = np.random.default_rng(80)
        alpha = 0.05
        X_cal = rng.normal(0, 1, size=(200, 5))
        X_test = rng.normal(0, 1, size=(200, 5))
        m = ConformedProcessMonitor(
            alpha=alpha, detector=_FastCentroidModel()
        ).fit(X_cal)
        r = m.predict(X_test)
        # Under exchangeability the conformal guarantee applies
        assert r.signal_rate <= alpha + 0.08, (
            f"In-control signal_rate {r.signal_rate:.3f} exceeds alpha + 0.08"
        )

    def test_outofcontrol_shifted_has_more_signals(self):
        """Observations from a far-shifted distribution should mostly alarm."""
        rng = np.random.default_rng(81)
        X_cal = rng.normal(0, 1, size=(200, 4))
        X_ic = rng.normal(0, 1, size=(50, 4))
        X_oc = rng.normal(10, 1, size=(50, 4))  # very far shift

        m = ConformedProcessMonitor(
            alpha=0.05, detector=_FastCentroidModel()
        ).fit(X_cal)
        r_ic = m.predict(X_ic)
        r_oc = m.predict(X_oc)

        assert r_oc.signal_rate > r_ic.signal_rate
        assert r_oc.signal_rate > 0.5, (
            f"Expected strong OOC signal rate, got {r_oc.signal_rate:.3f}"
        )

    def test_custom_duck_typed_detector(self):
        """Any object with .fit() and .decision_function() should work."""
        rng = np.random.default_rng(82)
        X_cal = rng.normal(0, 1, size=(100, 3))
        X_test = rng.normal(0, 1, size=(30, 3))

        m = ConformedProcessMonitor(
            alpha=0.05, detector=_FastCentroidModel()
        ).fit(X_cal)
        r = m.predict(X_test)
        assert isinstance(r, ConformedMonitorResult)
        assert len(r.p_values) == 30


# ---------------------------------------------------------------------------
# Conformal p-value vectorised function
# ---------------------------------------------------------------------------

class TestConformalPValuesVec:
    def test_basic(self):
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = np.array([0.5, 3.0, 6.0])
        p = _conformal_p_values_vec(scores, cal)
        # score=0.5: #{cal >= 0.5} = 5 -> (5+1)/(5+1)=1.0
        # score=3.0: #{cal >= 3.0} = 3 -> (3+1)/(5+1)=4/6
        # score=6.0: #{cal >= 6.0} = 0 -> (0+1)/(5+1)=1/6
        assert p[0] == pytest.approx(1.0)
        assert p[1] == pytest.approx(4.0 / 6.0)
        assert p[2] == pytest.approx(1.0 / 6.0)

    def test_all_p_positive(self):
        rng = np.random.default_rng(90)
        cal = rng.exponential(1.0, size=200)
        scores = rng.exponential(1.0, size=50)
        p = _conformal_p_values_vec(scores, cal)
        assert np.all(p > 0)

    def test_p_weakly_decreasing_in_score(self):
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = np.sort(np.array([0.5, 2.0, 4.5, 6.0]))
        p = _conformal_p_values_vec(scores, cal)
        for i in range(len(p) - 1):
            assert p[i] >= p[i + 1]


# ---------------------------------------------------------------------------
# Duck-typed test model (used in multiple test classes)
# ---------------------------------------------------------------------------

class _FastCentroidModel:
    """Minimal duck-typed anomaly model: decision_function = -distance from centroid.

    Faster and dependency-free alternative to sklearn for tests.
    """

    def __init__(self):
        self.centroid_ = None

    def fit(self, X):
        self.centroid_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X):
        # Higher = more normal (closer to centroid), consistent with sklearn convention
        dists = np.linalg.norm(X - self.centroid_, axis=1)
        return -dists
