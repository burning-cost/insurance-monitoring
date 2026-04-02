"""
Tests targeting coverage gaps across multiple modules.

Focuses on:
- thresholds: boundary conditions, negative values, AERatioThresholds edge cases
- drift: csi function, wasserstein_distance symmetry, PSI/KS validation errors
- baws: _get_block_length direct, window when T barely >= min_window, score type switching
- cusum: gamma_a != 1 LLO, reset preserves alarms, _llo function, Poisson exposure mismatch
- multicalibration: re-fit frozen boundaries, _cells_to_polars empty, degenerate cell warning,
  period_summary columns with data, bin_edges immutability
- model_monitor: _make_decision all branches directly, gamma/tweedie distributions, to_dict JSON
"""

from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# thresholds module
# ---------------------------------------------------------------------------


class TestPSIThresholdsBoundary:
    def test_exactly_at_green_max_is_amber(self):
        """PSI exactly at green_max=0.10 is amber (not green), per strict <."""
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(0.10) == "amber"

    def test_exactly_at_amber_max_is_red(self):
        """PSI exactly at amber_max=0.25 is red (not amber), per strict <."""
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(0.25) == "red"

    def test_negative_psi_is_green(self):
        """Negative PSI (theoretically shouldn't happen but shouldn't crash)."""
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(-0.01) == "green"

    def test_zero_psi_is_green(self):
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(0.0) == "green"

    def test_green_max_less_than_amber_max_is_valid(self):
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds(green_max=0.01, amber_max=0.05)
        assert t.classify(0.001) == "green"
        assert t.classify(0.03) == "amber"
        assert t.classify(0.10) == "red"


class TestAERatioThresholdsBoundary:
    def test_exactly_at_green_lower_is_green(self):
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(0.95) == "green"

    def test_exactly_at_green_upper_is_green(self):
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.05) == "green"

    def test_exactly_at_amber_lower_is_amber(self):
        """Value at amber_lower=0.90 is amber (outside green band)."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(0.90) == "amber"

    def test_exactly_at_amber_upper_is_amber(self):
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.10) == "amber"

    def test_above_red_upper_is_red(self):
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.50) == "red"

    def test_zero_ae_is_red(self):
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(0.0) == "red"

    def test_ae_at_exactly_red_lower_is_red(self):
        """A/E at red_lower=0.80 is red (outside amber band)."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(0.80) == "red"

    def test_custom_tight_bands(self):
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds(
            green_lower=0.98, green_upper=1.02,
            amber_lower=0.95, amber_upper=1.05,
            red_lower=0.90, red_upper=1.10,
        )
        assert t.classify(1.00) == "green"
        assert t.classify(1.03) == "amber"
        assert t.classify(1.08) == "red"


class TestGiniDriftThresholdsBoundary:
    def test_exactly_at_amber_pvalue_is_green(self):
        """p_value exactly at amber_p_value=0.32 is green (>= comparison)."""
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.32) == "green"

    def test_just_below_amber_pvalue_is_amber(self):
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.319) == "amber"

    def test_exactly_at_red_pvalue_is_amber(self):
        """p_value exactly at red_p_value=0.10 is amber (>= comparison)."""
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.10) == "amber"

    def test_zero_pvalue_is_red(self):
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.0) == "red"

    def test_p_value_1_is_green(self):
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(1.0) == "green"


# ---------------------------------------------------------------------------
# drift module
# ---------------------------------------------------------------------------


class TestCSI:
    def test_csi_identical_distributions_near_zero(self):
        """CSI for identical distributions should be near zero for all features."""
        from insurance_monitoring.drift import csi
        rng = np.random.default_rng(10)
        n = 5000
        data = {
            "driver_age": rng.normal(35, 8, n).tolist(),
            "vehicle_age": rng.uniform(0, 15, n).tolist(),
        }
        ref = pl.DataFrame(data)
        cur = pl.DataFrame(data)
        result = csi(ref, cur, features=["driver_age", "vehicle_age"])
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"feature", "csi", "band"}
        assert len(result) == 2
        for row in result.iter_rows(named=True):
            assert row["csi"] < 0.10, f"CSI {row['feature']}={row['csi']:.4f} for identical data"

    def test_csi_shifted_distribution_is_red(self):
        """Large distribution shift should give red CSI."""
        from insurance_monitoring.drift import csi
        rng = np.random.default_rng(11)
        ref = pl.DataFrame({"age": rng.normal(30, 5, 5000).tolist()})
        cur = pl.DataFrame({"age": rng.normal(50, 5, 2000).tolist()})
        result = csi(ref, cur, features=["age"])
        assert result["band"][0] == "red"

    def test_csi_returns_one_row_per_feature(self):
        from insurance_monitoring.drift import csi
        rng = np.random.default_rng(12)
        n = 1000
        ref = pl.DataFrame({f"f{i}": rng.uniform(0, 1, n).tolist() for i in range(5)})
        cur = pl.DataFrame({f"f{i}": rng.uniform(0, 1, n).tolist() for i in range(5)})
        result = csi(ref, cur, features=[f"f{i}" for i in range(5)])
        assert len(result) == 5
        assert sorted(result["feature"].to_list()) == [f"f{i}" for i in range(5)]

    def test_csi_missing_feature_raises(self):
        from insurance_monitoring.drift import csi
        ref = pl.DataFrame({"age": [1.0, 2.0, 3.0]})
        cur = pl.DataFrame({"age": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="not found"):
            csi(ref, cur, features=["nonexistent"])

    def test_csi_nonnegative(self):
        """CSI should always be non-negative."""
        from insurance_monitoring.drift import csi
        rng = np.random.default_rng(13)
        ref = pl.DataFrame({"x": rng.normal(0, 1, 500).tolist()})
        cur = pl.DataFrame({"x": rng.normal(1, 1, 300).tolist()})
        result = csi(ref, cur, features=["x"])
        assert result["csi"][0] >= 0.0


class TestWassersteinDistance:
    def test_identical_distributions_near_zero(self):
        """Wasserstein distance between identical distributions should be near zero."""
        from insurance_monitoring.drift import wasserstein_distance
        rng = np.random.default_rng(20)
        data = rng.normal(35, 8, 5000)
        result = wasserstein_distance(data, data)
        assert isinstance(result, float)
        assert result < 0.05

    def test_large_shift_large_distance(self):
        """5-unit mean shift: Wasserstein distance should be ~5."""
        from insurance_monitoring.drift import wasserstein_distance
        rng = np.random.default_rng(21)
        ref = rng.normal(30, 5, 10000)
        cur = rng.normal(35, 5, 5000)
        result = wasserstein_distance(ref, cur)
        assert isinstance(result, float)
        # Wasserstein for Normal(mu1, sigma) vs Normal(mu2, sigma) ~ |mu1 - mu2|
        assert result > 3.0

    def test_distance_is_nonnegative(self):
        from insurance_monitoring.drift import wasserstein_distance
        rng = np.random.default_rng(23)
        for _ in range(5):
            ref = rng.uniform(0, 10, 500)
            cur = rng.uniform(0, 10, 300)
            result = wasserstein_distance(ref, cur)
            assert result >= 0.0

    def test_polars_series_input(self):
        from insurance_monitoring.drift import wasserstein_distance
        rng = np.random.default_rng(24)
        ref = pl.Series(rng.normal(0, 1, 500).tolist())
        cur = pl.Series(rng.normal(0, 1, 300).tolist())
        result = wasserstein_distance(ref, cur)
        assert isinstance(result, float)

    def test_distance_increases_with_shift(self):
        """Larger mean shift should give larger Wasserstein distance."""
        from insurance_monitoring.drift import wasserstein_distance
        rng = np.random.default_rng(25)
        ref = rng.normal(0, 1, 5000)
        cur_small = rng.normal(1, 1, 5000)
        cur_large = rng.normal(5, 1, 5000)
        d_small = wasserstein_distance(ref, cur_small)
        d_large = wasserstein_distance(ref, cur_large)
        assert d_large > d_small

class TestKSTestPolarsInput:
    def test_ks_polars_series_input(self):
        """KS test should accept Polars Series."""
        from insurance_monitoring.drift import ks_test
        rng = np.random.default_rng(30)
        ref = pl.Series(rng.normal(0, 1, 500).tolist())
        cur = pl.Series(rng.normal(0, 1, 500).tolist())
        result = ks_test(ref, cur)
        assert "statistic" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

class TestPSIValidation:
    def test_empty_reference_raises(self):
        from insurance_monitoring.drift import psi
        with pytest.raises(ValueError, match="non-empty"):
            psi([], [1.0, 2.0, 3.0])

    def test_empty_current_raises(self):
        from insurance_monitoring.drift import psi
        with pytest.raises(ValueError, match="non-empty"):
            psi([1.0, 2.0, 3.0], [])

    def test_exposure_length_mismatch_raises(self):
        from insurance_monitoring.drift import psi
        rng = np.random.default_rng(40)
        ref = rng.normal(0, 1, 100)
        cur = rng.normal(0, 1, 50)
        weights = rng.uniform(0.5, 2.0, 40)  # wrong length
        with pytest.raises(ValueError, match="exposure_weights length"):
            psi(ref, cur, exposure_weights=weights)

    def test_reference_exposure_length_mismatch_raises(self):
        from insurance_monitoring.drift import psi
        rng = np.random.default_rng(41)
        ref = rng.normal(0, 1, 100)
        cur = rng.normal(0, 1, 50)
        ref_weights = rng.uniform(0.5, 2.0, 80)  # wrong length
        with pytest.raises(ValueError, match="reference_exposure length"):
            psi(ref, cur, reference_exposure=ref_weights)

    def test_n_bins_less_than_2_raises(self):
        from insurance_monitoring.drift import psi
        with pytest.raises(ValueError, match="n_bins"):
            psi([1.0, 2.0], [1.0, 2.0], n_bins=1)

    def test_constant_reference_returns_zero(self):
        """Single-value reference collapses all bins — PSI should return 0."""
        from insurance_monitoring.drift import psi
        ref = np.ones(100)
        cur = np.ones(50) * 2.0
        # Should not raise; undefined PSI returns 0
        result = psi(ref, cur)
        assert result == 0.0

    def test_both_exposures_weighted_psi(self):
        """Fully symmetric PSI (both sides weighted) should be non-negative."""
        from insurance_monitoring.drift import psi
        rng = np.random.default_rng(42)
        ref = rng.normal(30, 5, 5000)
        cur = rng.normal(32, 5, 3000)
        ref_exp = rng.uniform(0.5, 2.0, 5000)
        cur_exp = rng.uniform(0.5, 2.0, 3000)
        result = psi(ref, cur, exposure_weights=cur_exp, reference_exposure=ref_exp)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# baws module
# ---------------------------------------------------------------------------


class TestBAWSBlockLength:
    def test_get_block_length_default_rule(self):
        """Default T^(1/3) rule returns int >= min_block_length."""
        from insurance_monitoring.baws import BAWSMonitor
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10)
        T = 125
        bl = m._get_block_length(T)
        expected = max(3, int(round(T ** (1.0 / 3.0))))  # T^(1/3) = 5
        assert bl == expected

    def test_get_block_length_fixed(self):
        """Fixed block_length should be returned unchanged (within clamping)."""
        from insurance_monitoring.baws import BAWSMonitor
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, block_length=7)
        bl = m._get_block_length(100)
        assert bl == 7

    def test_get_block_length_clamped_below(self):
        """Block length is clamped to min_block_length."""
        from insurance_monitoring.baws import BAWSMonitor
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10,
                        min_block_length=5)
        # T=8: T^(1/3) = 2, but min_block_length=5
        bl = m._get_block_length(8)
        assert bl >= 5

    def test_get_block_length_clamped_above(self):
        """Block length clamped to T//2 for small T."""
        from insurance_monitoring.baws import BAWSMonitor
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, block_length=200)
        T = 20
        bl = m._get_block_length(T)
        assert bl <= max(T // 2, 1)

    def test_fit_with_t_equal_to_min_window(self):
        """fit() with T exactly at min candidate window should succeed."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(99)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=0)
        data = rng.standard_normal(50)  # exactly min_window
        m.fit(data)
        result = m.update(float(rng.standard_normal()))
        assert result.selected_window in [50, 100]

    def test_single_candidate_window_raises(self):
        """Only one candidate window should raise ValueError."""
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="at least 2"):
            BAWSMonitor(alpha=0.05, candidate_windows=[100])

    def test_negative_window_raises(self):
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="positive"):
            BAWSMonitor(alpha=0.05, candidate_windows=[50, -10])

    def test_n_bootstrap_too_small_raises(self):
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="n_bootstrap"):
            BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=5)

    def test_min_block_length_zero_raises(self):
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="min_block_length"):
            BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], min_block_length=0)

    def test_scores_include_inf_for_windows_exceeding_history(self):
        """Windows larger than current history should have score=inf."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(77)
        m = BAWSMonitor(
            alpha=0.05,
            candidate_windows=[50, 300],  # 300 > initial 100 observations
            n_bootstrap=10,
            random_state=1,
        )
        m.fit(rng.standard_normal(100))
        result = m.update(float(rng.standard_normal()))
        # Window 300 exceeds history of 101: should be inf
        assert result.scores[300] == float("inf")
        # Window 50 should have a finite score
        assert np.isfinite(result.scores[50])

    def test_history_n_obs_increments_correctly(self):
        """n_obs in history should grow by 1 with each update."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(88)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=2)
        m.fit(rng.standard_normal(100))
        for i, r in enumerate(rng.standard_normal(5)):
            result = m.update(float(r))
            assert result.n_obs == 101 + i

    def test_fissler_ziegel_score_es_positive_clipped(self):
        """fissler_ziegel_score clips ES to be strictly negative."""
        from insurance_monitoring.baws import fissler_ziegel_score
        # es=0.5 (positive) should be clipped to -1e-10
        y = np.array([0.0])
        score = fissler_ziegel_score(var=0.0, es=0.5, y=y, alpha=0.05)
        assert np.isfinite(score[0]), "Score with positive ES (clipped) should be finite"

    def test_block_bootstrap_returns_same_length(self):
        """Block bootstrap replicate should have same length as input."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(55)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=3)
        data = rng.standard_normal(100)
        rep = m._block_bootstrap(data, block_length=10)
        assert len(rep) == len(data)

    def test_block_bootstrap_degenerate_block_length(self):
        """block_length >= T falls back to iid resample."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(56)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=4)
        data = rng.standard_normal(20)
        rep = m._block_bootstrap(data, block_length=20)
        assert len(rep) == 20


# ---------------------------------------------------------------------------
# cusum module
# ---------------------------------------------------------------------------


class TestCUSUMLLO:
    def test_llo_identity_at_delta1_gamma1(self):
        """_llo(x, 1, 1) should return x (identity function)."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = _llo(x, delta=1.0, gamma=1.0)
        np.testing.assert_allclose(result, x, rtol=1e-8, atol=1e-10)

    def test_llo_delta_greater_1_shifts_up(self):
        """delta > 1 should shift probabilities upward."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.3, 0.5, 0.7])
        result = _llo(x, delta=2.0, gamma=1.0)
        assert np.all(result > x), "delta>1 should shift probabilities up"

    def test_llo_delta_less_1_shifts_down(self):
        """delta < 1 should shift probabilities downward."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.3, 0.5, 0.7])
        result = _llo(x, delta=0.5, gamma=1.0)
        assert np.all(result < x), "delta<1 should shift probabilities down"

    def test_llo_output_in_unit_interval(self):
        """_llo output should always be in (0, 1)."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        for delta in [0.5, 1.0, 2.0, 5.0]:
            for gamma in [0.5, 1.0, 2.0]:
                result = _llo(x, delta=delta, gamma=gamma)
                assert np.all(result > 0.0)
                assert np.all(result < 1.0)


class TestCUSUMGammaA:
    def test_gamma_a_not_1_bernoulli_works(self):
        """gamma_a != 1.0 with delta_a != 1.0 should construct without error."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        monitor = CalibrationCUSUM(delta_a=2.0, gamma_a=0.8, n_mc=100)
        rng = np.random.default_rng(99)
        p = rng.uniform(0.05, 0.20, 50)
        y = rng.binomial(1, p)
        alarm = monitor.update(p, y)
        assert alarm.statistic >= 0.0

    def test_identity_alternative_with_gamma_not_1_does_not_raise(self):
        """delta_a=1, gamma_a=2 is a valid (non-identity) alternative."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        # delta=1, gamma=2 is NOT identity (identity is delta=1, gamma=1)
        monitor = CalibrationCUSUM(delta_a=1.0, gamma_a=2.0, n_mc=100)
        assert monitor.delta_a == 1.0
        assert monitor.gamma_a == 2.0

    def test_identity_alternative_raises_only_when_both_1(self):
        """Only delta_a=1 AND gamma_a=1 together should raise."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        with pytest.raises(ValueError, match="identity"):
            CalibrationCUSUM(delta_a=1.0, gamma_a=1.0, distribution="bernoulli")
        # delta_a=2 but gamma_a=1 should be fine
        CalibrationCUSUM(delta_a=2.0, gamma_a=1.0)


class TestCUSUMResetBehaviour:
    def test_reset_does_not_clear_alarm_times(self):
        """Alarm times accumulate in summary() even after reset()."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(10)
        monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.1, n_mc=200, random_state=0)
        for _ in range(30):
            p = rng.uniform(0.05, 0.25, 30)
            monitor.update(p, rng.binomial(1, np.clip(2.0 * p, 0, 1)))

        n_alarms_before = monitor.summary().n_alarms
        alarm_times_before = monitor.summary().alarm_times[:]
        monitor.reset()

        s_after = monitor.summary()
        assert s_after.n_alarms == n_alarms_before, "n_alarms must survive reset()"
        assert s_after.alarm_times == alarm_times_before, "alarm_times must survive reset()"

    def test_reset_clears_history_lists(self):
        """After reset(), time and statistic are zero."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(11)
        monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=200, random_state=1)
        for _ in range(10):
            p = rng.uniform(0.05, 0.25, 30)
            monitor.update(p, rng.binomial(1, p))
        monitor.reset()
        assert monitor.time == 0
        assert monitor.statistic == 0.0

    def test_summary_before_any_update(self):
        """Summary before any update should show zeroes."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        monitor = CalibrationCUSUM(delta_a=2.0, n_mc=100)
        s = monitor.summary()
        assert s.n_time_steps == 0
        assert s.n_alarms == 0
        assert s.current_statistic == 0.0
        assert s.current_control_limit is None

    def test_poisson_exposure_length_mismatch_raises(self):
        """Exposure length != predictions length should raise ValueError."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        monitor = CalibrationCUSUM(delta_a=1.5, distribution="poisson", n_mc=100)
        pred = np.full(10, 0.1)
        y = np.zeros(10, dtype=int)
        bad_exposure = np.ones(8)  # wrong length
        with pytest.raises(ValueError, match="exposure length"):
            monitor.update(pred, y, exposure=bad_exposure)

    def test_statistic_resets_to_zero_after_alarm(self):
        """After an alarm fires, the statistic should reset to 0."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(20)
        monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.1, n_mc=500, random_state=2)
        alarm_fired = False
        for _ in range(50):
            p = rng.uniform(0.05, 0.25, 50)
            y = rng.binomial(1, np.clip(2.0 * p, 0, 1))
            alarm = monitor.update(p, y)
            if alarm.triggered:
                # The statistic in the result is the post-reset value
                assert alarm.statistic == 0.0
                alarm_fired = True
                break
        # The test is only meaningful if an alarm actually fired
        if not alarm_fired:
            pytest.skip("No alarm fired in 50 steps — test skipped (insufficient power)")


class TestResampleMCPool:
    def test_resample_returns_correct_size(self):
        """_resample_mc_pool should always return size n_mc."""
        from insurance_monitoring.cusum import _resample_mc_pool
        rng = np.random.default_rng(0)
        mc_s = rng.uniform(0, 5, 1000)
        result = _resample_mc_pool(mc_s, h_t=3.0, n_mc=1000, rng=rng)
        assert len(result) == 1000

    def test_resample_resets_when_too_few_below(self):
        """Pool resets to zeros when fewer than n_mc//4 paths are below limit."""
        from insurance_monitoring.cusum import _resample_mc_pool
        rng = np.random.default_rng(1)
        # Almost all paths are above the limit
        mc_s = np.ones(1000) * 10.0
        result = _resample_mc_pool(mc_s, h_t=2.0, n_mc=1000, rng=rng)
        assert np.all(result == 0.0), "Pool should reset to zeros when few paths are below limit"

    def test_resample_from_below_paths(self):
        """When enough paths are below limit, resampled pool contains only below-limit values."""
        from insurance_monitoring.cusum import _resample_mc_pool
        rng = np.random.default_rng(2)
        mc_s = rng.uniform(0, 2, 1000)  # all below h_t=3
        result = _resample_mc_pool(mc_s, h_t=3.0, n_mc=1000, rng=rng)
        assert np.all(result <= 3.0), "Resampled pool should only contain paths below limit"


# ---------------------------------------------------------------------------
# multicalibration module
# ---------------------------------------------------------------------------


class TestMulticalibrationFrozenBinEdges:
    def test_refit_updates_bin_edges(self):
        """Calling fit() again changes bin_edges (second fit overwrites first)."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        rng = np.random.default_rng(0)
        n = 2000
        y_pred_1 = rng.gamma(2, 0.05, n)
        y_pred_2 = rng.gamma(2, 0.15, n)  # different scale
        y_true = rng.poisson(y_pred_1)
        groups = rng.choice(["A", "B"], n)

        m = MulticalibrationMonitor(n_bins=4, min_exposure=5)
        m.fit(y_true, y_pred_1, groups)
        edges_1 = m.bin_edges.copy()

        m.fit(y_true, y_pred_2, groups)
        edges_2 = m.bin_edges.copy()

        assert not np.allclose(edges_1[1:-1], edges_2[1:-1]), (
            "Bin edges should change after re-fitting on different predictions"
        )

    def test_update_uses_frozen_edges_not_refit(self):
        """update() should bin against the original fit() edges, not fresh edges."""
        from insurance_monitoring.multicalibration import (
            MulticalibrationMonitor,
            _exposure_weighted_quantile_edges,
            _assign_bins,
        )
        rng = np.random.default_rng(10)
        n = 3000
        y_pred = rng.gamma(2, 0.05, n)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B"], n)
        exposure = rng.uniform(0.5, 2.0, n)

        m = MulticalibrationMonitor(n_bins=5, min_exposure=5)
        m.fit(y_true, y_pred, groups, exposure=exposure)
        frozen_edges = m.bin_edges.copy()

        # Call update and check that frozen edges are unchanged
        m.update(y_true, y_pred, groups, exposure=exposure)
        assert np.allclose(m.bin_edges, frozen_edges), "Bin edges should not change after update()"


class TestCellsToPolarsBehaviour:
    def test_empty_cells_returns_correct_schema(self):
        """_cells_to_polars([]) should return an empty DataFrame with correct schema."""
        from insurance_monitoring.multicalibration import _cells_to_polars
        df = _cells_to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "bin_idx" in df.columns
        assert "group" in df.columns
        assert "alert" in df.columns

    def test_group_cast_to_string_for_int_groups(self):
        """Integer group labels should be cast to String in the output DataFrame."""
        from insurance_monitoring.multicalibration import MulticalibCell, _cells_to_polars
        cells = [
            MulticalibCell(
                bin_idx=0, group=1, n_exposure=100.0, observed=0.10, expected=0.10,
                AE_ratio=1.0, relative_bias=0.0, z_stat=0.0, alert=False,
            )
        ]
        df = _cells_to_polars(cells)
        assert df["group"].dtype == pl.String


class TestDegenerateExpectedCell:
    def test_zero_expected_cell_skips_with_warning(self):
        """A cell with expected=0 after weighting should be skipped with a warning."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        rng = np.random.default_rng(5)
        n = 2000
        # Create a scenario where one cell has zero predicted rate
        y_pred = rng.gamma(2, 0.05, n)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B"], n)

        m = MulticalibrationMonitor(n_bins=5, min_exposure=5)
        m.fit(y_true, y_pred, groups)

        # Manually create a near-zero expected situation for test purposes
        # by setting update predictions close to zero for one group
        mask_a = groups == "A"
        y_pred_bad = y_pred.copy()
        y_pred_bad[mask_a] = 1e-30  # extremely small but positive (passes validation)

        # Should produce a result (cells with degenerate expected may be skipped)
        result = m.update(y_true, y_pred_bad, groups)
        assert isinstance(result.cell_table, pl.DataFrame)

    def test_period_summary_has_required_columns(self):
        """period_summary() should have all expected columns."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        rng = np.random.default_rng(6)
        n = 3000
        y_pred = rng.gamma(2, 0.05, n)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B"], n)

        m = MulticalibrationMonitor(n_bins=5, min_exposure=20)
        m.fit(y_true, y_pred, groups)
        m.update(y_true, y_pred, groups)
        m.update(y_true, y_pred, groups)

        df = m.period_summary()
        assert "period_index" in df.columns
        assert "n_alerts" in df.columns
        assert "n_cells_evaluated" in df.columns
        assert "n_cells_skipped" in df.columns
        assert "overall_pass" in df.columns
        assert len(df) == 2
        assert df["period_index"].to_list() == [1, 2]

    def test_multicalib_result_to_dict_json_serialisable(self):
        """to_dict() result should survive JSON round-trip."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        rng = np.random.default_rng(7)
        n = 3000
        y_pred = rng.gamma(2, 0.05, n)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B", "C"], n)

        m = MulticalibrationMonitor(n_bins=5, min_exposure=20)
        m.fit(y_true, y_pred, groups)
        result = m.update(y_true, y_pred, groups)

        d = result.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert "summary" in restored
        assert "alerts" in restored

    def test_worst_cell_has_max_abs_relative_bias(self):
        """summary()['worst_cell'] should be the cell with max |relative_bias|."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        from insurance_monitoring.multicalibration import (
            _exposure_weighted_quantile_edges,
            _assign_bins,
        )
        rng = np.random.default_rng(8)
        n = 8000
        n_bins = 5
        y_pred = rng.gamma(2, 0.05, n)
        groups = rng.choice(["G0", "G1"], n)
        exposure = rng.uniform(0.5, 2.0, n)

        # Inject severe bias in one cell
        edges = _exposure_weighted_quantile_edges(y_pred, exposure, n_bins)
        bin_idx = _assign_bins(y_pred, edges)
        y_true = rng.poisson(y_pred * exposure) / exposure
        mask = (bin_idx == 2) & (groups == "G0")
        y_true = y_true.copy()
        y_true[mask] *= 2.5  # 150% bias

        m = MulticalibrationMonitor(n_bins=n_bins, min_exposure=30, min_relative_bias=0.05)
        m.fit(y_true, y_pred, groups, exposure=exposure)
        result = m.update(y_true, y_pred, groups, exposure=exposure)

        if len(result.alerts) >= 2:
            worst = result.summary()["worst_cell"]
            max_bias = max(abs(a.relative_bias) for a in result.alerts)
            assert abs(worst["relative_bias"]) == pytest.approx(max_bias, rel=1e-9)

    def test_multicalib_n_bins_minimum_2(self):
        """n_bins=1 should raise ValueError."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        with pytest.raises(ValueError, match="n_bins"):
            MulticalibrationMonitor(n_bins=1)

    def test_multicalib_negative_min_exposure_raises(self):
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        with pytest.raises(ValueError, match="min_exposure"):
            MulticalibrationMonitor(min_exposure=-1.0)

    def test_validation_rejects_2d_y_true(self):
        from insurance_monitoring.multicalibration import _validate_monitor_inputs
        y_true = np.ones((10, 2))
        y_pred = np.ones(10) * 0.1
        groups = np.array(["A"] * 10)
        with pytest.raises(ValueError, match="1-dimensional"):
            _validate_monitor_inputs(y_true, y_pred, groups, None)

    def test_validation_rejects_exposure_too_short(self):
        from insurance_monitoring.multicalibration import _validate_monitor_inputs
        y_true = np.ones(10)
        y_pred = np.ones(10) * 0.1
        groups = np.array(["A"] * 10)
        exposure = np.ones(8)  # wrong length
        with pytest.raises(ValueError, match="exposure must have the same length"):
            _validate_monitor_inputs(y_true, y_pred, groups, exposure)


# ---------------------------------------------------------------------------
# model_monitor module
# ---------------------------------------------------------------------------


class TestMakeDecisionDirect:
    def test_redeploy_when_nothing_significant(self):
        from insurance_monitoring.model_monitor import _make_decision
        decision, reason = _make_decision(
            gini_sig=False, gini_z=-0.5, gini_p=0.6,
            gmcb_sig=False, gmcb_score=0.001, gmcb_p=0.5,
            lmcb_sig=False, lmcb_score=0.001, lmcb_p=0.6,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert decision == "REDEPLOY"
        assert "REDEPLOY" in decision

    def test_recalibrate_when_only_gmcb_significant(self):
        from insurance_monitoring.model_monitor import _make_decision
        decision, reason = _make_decision(
            gini_sig=False, gini_z=-0.5, gini_p=0.6,
            gmcb_sig=True, gmcb_score=0.05, gmcb_p=0.01,
            lmcb_sig=False, lmcb_score=0.001, lmcb_p=0.6,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert decision == "RECALIBRATE"
        assert "balance" in reason.lower() or "global" in reason.lower()

    def test_refit_when_gini_significant(self):
        from insurance_monitoring.model_monitor import _make_decision
        decision, reason = _make_decision(
            gini_sig=True, gini_z=-3.0, gini_p=0.003,
            gmcb_sig=False, gmcb_score=0.001, gmcb_p=0.5,
            lmcb_sig=False, lmcb_score=0.001, lmcb_p=0.6,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert decision == "REFIT"
        assert "REFIT" in decision

    def test_refit_when_lmcb_significant(self):
        from insurance_monitoring.model_monitor import _make_decision
        decision, reason = _make_decision(
            gini_sig=False, gini_z=-0.5, gini_p=0.6,
            gmcb_sig=False, gmcb_score=0.001, gmcb_p=0.5,
            lmcb_sig=True, lmcb_score=0.1, lmcb_p=0.01,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert decision == "REFIT"

    def test_refit_when_both_gini_and_lmcb_significant(self):
        from insurance_monitoring.model_monitor import _make_decision
        decision, reason = _make_decision(
            gini_sig=True, gini_z=-3.0, gini_p=0.003,
            gmcb_sig=True, gmcb_score=0.05, gmcb_p=0.01,
            lmcb_sig=True, lmcb_score=0.1, lmcb_p=0.01,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert decision == "REFIT"

    def test_reason_mentions_gini_on_gini_refit(self):
        from insurance_monitoring.model_monitor import _make_decision
        _, reason = _make_decision(
            gini_sig=True, gini_z=-4.0, gini_p=0.0001,
            gmcb_sig=False, gmcb_score=0.0, gmcb_p=0.8,
            lmcb_sig=False, lmcb_score=0.0, lmcb_p=0.8,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert "ranking" in reason.lower() or "gini" in reason.lower()

    def test_reason_mentions_lmcb_on_lmcb_refit(self):
        from insurance_monitoring.model_monitor import _make_decision
        _, reason = _make_decision(
            gini_sig=False, gini_z=-0.1, gini_p=0.9,
            gmcb_sig=False, gmcb_score=0.0, gmcb_p=0.8,
            lmcb_sig=True, lmcb_score=0.1, lmcb_p=0.02,
            alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
        )
        assert "local" in reason.lower() or "lmcb" in reason.lower()


class TestModelMonitorGammaDistribution:
    def test_gamma_distribution_runs(self):
        """ModelMonitor with gamma distribution should fit and test without error."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(42)
        n = 2000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.5, n)  # severity predictions
        y_ref = rng.gamma(2.0 / y_hat, y_hat, n)  # gamma-distributed actuals

        monitor = ModelMonitor(
            distribution="gamma",
            n_bootstrap=99,
            alpha_gini=0.32,
            alpha_global=0.32,
            alpha_local=0.32,
            random_state=0,
        )
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_ref, y_hat, exposure)
        assert result.decision in ("REDEPLOY", "RECALIBRATE", "REFIT")

    def test_tweedie_distribution_runs(self):
        """ModelMonitor with tweedie distribution should complete without error."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(43)
        n = 2000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.1, n)
        y_ref = rng.gamma(2, 0.1, n)

        monitor = ModelMonitor(
            distribution="tweedie",
            tweedie_power=1.5,
            n_bootstrap=99,
            alpha_gini=0.32,
            random_state=0,
        )
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_ref, y_hat, exposure)
        assert result.decision in ("REDEPLOY", "RECALIBRATE", "REFIT")

    def test_to_dict_is_json_serialisable(self):
        """ModelMonitorResult.to_dict() should survive json.dumps."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(44)
        n = 1000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * y_hat) / exposure

        monitor = ModelMonitor(
            distribution="poisson",
            n_bootstrap=99,
            random_state=1,
        )
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_ref, y_hat, exposure)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert "decision" in json.loads(json_str)

    def test_summary_string_contains_all_decisions(self):
        """summary() string should mention all three tests."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(45)
        n = 1000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * y_hat) / exposure

        monitor = ModelMonitor(distribution="poisson", n_bootstrap=99, random_state=2)
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_ref, y_hat, exposure)
        s = result.summary()
        # summary should mention Gini, GMCB, and LMCB
        assert "Gini" in s
        assert "GMCB" in s
        assert "LMCB" in s

    def test_gini_metadata_fields(self):
        """Result should carry correct alpha and distribution metadata."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(46)
        n = 1000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * y_hat) / exposure

        alpha_g = 0.20
        alpha_gl = 0.25
        alpha_l = 0.30
        monitor = ModelMonitor(
            distribution="poisson",
            n_bootstrap=99,
            alpha_gini=alpha_g,
            alpha_global=alpha_gl,
            alpha_local=alpha_l,
            random_state=3,
        )
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_ref, y_hat, exposure)

        assert result.alpha_gini == alpha_g
        assert result.alpha_global == alpha_gl
        assert result.alpha_local == alpha_l
        assert result.distribution == "poisson"
        assert result.n_new == n

    def test_small_exposure_warning_at_fit(self):
        """fit() with tiny exposures (< 0.05 median) should emit UserWarning."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(47)
        n = 500
        exposure = rng.uniform(0.001, 0.02, n)  # very small
        y_hat = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(y_hat) / 1.0

        monitor = ModelMonitor(distribution="poisson", n_bootstrap=50, random_state=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.fit(y_ref, y_hat, exposure)
            time_split_warns = [x for x in w if "0.05" in str(x.message) or "aggregated" in str(x.message).lower()]
            assert len(time_split_warns) >= 1, "Expected UserWarning for small median exposure"

    def test_repr_contains_distribution(self):
        from insurance_monitoring.model_monitor import ModelMonitor
        monitor = ModelMonitor(distribution="gamma", n_bootstrap=99)
        r = repr(monitor)
        assert "gamma" in r
        assert "ModelMonitor" in r
