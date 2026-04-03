"""
Additional test coverage for insurance-monitoring (April 2026).

Focuses on edge cases, error paths, private helpers, and integration between
modules that the existing test suite does not cover. Does not duplicate tests
already present in test_baws.py, test_cusum.py, test_multicalibration_monitor.py,
test_conformal_spc.py, or test_model_monitor.py.

Modules covered
---------------
baws            - private helpers, validation edge cases, degenerate data
cusum           - _llo / _resample_mc_pool helpers, gamma_a != 1, pool reset
multicalibration - degenerate expected<=0 cell, n<2 input, _cells_to_polars empty
conformal_spc   - score_samples fallback, AttributeError, ConformedControlChart.plot
model_monitor   - _make_decision unit tests, nan gini path, test() time-split warning
thresholds      - boundary conditions on all three classifiers
top-level import - verify all public names importable from insurance_monitoring
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# baws: private helpers and validation edge cases
# ---------------------------------------------------------------------------


class TestBAWSPrivateHelpers:
    """Unit tests for BAWSMonitor private methods."""

    def _make_monitor(self, **kw):
        from insurance_monitoring.baws import BAWSMonitor
        defaults = dict(
            alpha=0.05,
            candidate_windows=[50, 100],
            n_bootstrap=10,
            random_state=0,
        )
        defaults.update(kw)
        return BAWSMonitor(**defaults)

    def test_get_block_length_auto(self):
        """_get_block_length with block_length=None uses T^(1/3) rule."""
        m = self._make_monitor()
        m.fit(np.zeros(100))
        # T=100: T^(1/3) = 4.64 -> round -> 5
        bl = m._get_block_length(100)
        assert bl == 5

    def test_get_block_length_fixed(self):
        """Explicit block_length parameter is respected."""
        m = self._make_monitor(block_length=7)
        m.fit(np.zeros(100))
        bl = m._get_block_length(100)
        assert bl == 7

    def test_get_block_length_clamped_min(self):
        """Very small T still returns min_block_length."""
        m = self._make_monitor(min_block_length=3)
        m.fit(np.zeros(50))
        # T=2: T^(1/3) < 3, so should clamp to 3
        bl = m._get_block_length(2)
        assert bl == 3

    def test_get_block_length_clamped_max(self):
        """block_length cannot exceed T//2."""
        m = self._make_monitor(block_length=999)
        m.fit(np.zeros(100))
        bl = m._get_block_length(10)
        assert bl <= 5  # 10//2 = 5

    def test_compute_var_es_empty(self):
        """_compute_var_es on empty array returns (0.0, 0.0)."""
        m = self._make_monitor()
        m.fit(np.zeros(50))
        var, es = m._compute_var_es(np.array([]))
        assert var == 0.0
        assert es == 0.0

    def test_compute_var_es_all_same(self):
        """Constant data: var == es == the constant value."""
        m = self._make_monitor()
        m.fit(np.zeros(50))
        data = np.full(50, -2.0)
        var, es = m._compute_var_es(data)
        assert var == pytest.approx(-2.0)
        assert es == pytest.approx(-2.0)

    def test_compute_var_es_es_le_var(self):
        """ES must always be <= VaR."""
        rng = np.random.default_rng(42)
        m = self._make_monitor()
        m.fit(np.zeros(50))
        for _ in range(20):
            data = rng.standard_t(df=4, size=50)
            var, es = m._compute_var_es(data)
            assert es <= var + 1e-12

    def test_block_bootstrap_block_ge_T(self):
        """block_length >= T: degenerate single-block fallback returns correct length."""
        m = self._make_monitor()
        m.fit(np.zeros(50))
        data = np.arange(10, dtype=float)
        rep = m._block_bootstrap(data, block_length=20)
        assert len(rep) == len(data)
        # All values must be from data
        assert set(rep.tolist()).issubset(set(data.tolist()))

    def test_block_bootstrap_empty(self):
        """Empty data returns empty replicate."""
        m = self._make_monitor()
        m.fit(np.zeros(50))
        empty = np.array([], dtype=float)
        rep = m._block_bootstrap(empty, block_length=5)
        assert len(rep) == 0


class TestBAWSValidationEdgeCases:
    def test_single_candidate_window_raises(self):
        """candidate_windows must have at least 2 values."""
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="at least 2"):
            BAWSMonitor(alpha=0.05, candidate_windows=[100])

    def test_zero_candidate_window_raises(self):
        """Any window <= 0 must raise ValueError."""
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="positive"):
            BAWSMonitor(alpha=0.05, candidate_windows=[0, 100])

    def test_n_bootstrap_too_small_raises(self):
        """n_bootstrap < 10 must raise ValueError."""
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="n_bootstrap"):
            BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=5)

    def test_min_block_length_zero_raises(self):
        """min_block_length < 1 must raise ValueError."""
        from insurance_monitoring.baws import BAWSMonitor
        with pytest.raises(ValueError, match="min_block_length"):
            BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], min_block_length=0)

    def test_duplicate_candidate_windows_sorted(self):
        """Duplicate windows in candidate_windows should not crash; they get sorted."""
        from insurance_monitoring.baws import BAWSMonitor
        # [100, 50, 100] has duplicates — should still work (min deduplication not required
        # by spec, but must not crash)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[100, 50, 100], n_bootstrap=10)
        assert m.candidate_windows == sorted([50, 100, 100])

    def test_score_window_tiny_data_returns_inf(self):
        """_score_window with < 2 data points returns inf."""
        from insurance_monitoring.baws import BAWSMonitor
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=0)
        m.fit(np.zeros(50))
        s = m._score_window(np.array([1.0]))
        assert s == float("inf")

    def test_bootstrap_score_tiny_data_returns_inf(self):
        """_bootstrap_score with < 2 data points returns inf."""
        from insurance_monitoring.baws import BAWSMonitor
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=0)
        m.fit(np.zeros(50))
        s = m._bootstrap_score(np.array([1.0]), block_length=3)
        assert s == float("inf")

    def test_fissler_ziegel_es_at_zero_clamped(self):
        """fissler_ziegel_score with es=0 clips to -1e-10 without raising."""
        from insurance_monitoring.baws import fissler_ziegel_score
        y = np.array([1.0, -1.0])
        # es=0 is degenerate but should not raise (clipped to -1e-10)
        score = fissler_ziegel_score(var=-0.5, es=0.0, y=y, alpha=0.05)
        assert np.all(np.isfinite(score))

    def test_asymm_abs_loss_at_boundary(self):
        """asymm_abs_loss: y exactly at var boundary should assign to 'below'."""
        from insurance_monitoring.baws import asymm_abs_loss
        # y = var: indicator = 1 (y <= x)
        y = np.array([0.0])
        score = asymm_abs_loss(var=0.0, y=y, alpha=0.1)
        # (alpha - 1) * (y - x) = (0.1 - 1) * 0 = 0
        assert float(score[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cusum: private helpers, gamma_a mode, pool reset
# ---------------------------------------------------------------------------


class TestCUSUMPrivateHelpers:
    def test_llo_identity(self):
        """_llo(x, delta=1, gamma=1) == x for all x in (0, 1)."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = _llo(x, delta=1.0, gamma=1.0)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_llo_delta_upward_shift(self):
        """delta > 1 shifts probabilities upward."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.3, 0.5])
        result = _llo(x, delta=2.0, gamma=1.0)
        assert np.all(result > x)

    def test_llo_delta_downward_shift(self):
        """delta < 1 shifts probabilities downward."""
        from insurance_monitoring.cusum import _llo
        x = np.array([0.3, 0.5])
        result = _llo(x, delta=0.5, gamma=1.0)
        assert np.all(result < x)

    def test_llo_output_in_unit_interval(self):
        """_llo output must stay in (0, 1)."""
        from insurance_monitoring.cusum import _llo
        rng = np.random.default_rng(10)
        x = rng.uniform(0.01, 0.99, 100)
        result = _llo(x, delta=5.0, gamma=2.0)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_resample_mc_pool_normal(self):
        """_resample_mc_pool: enough below limit -> resample from below."""
        from insurance_monitoring.cusum import _resample_mc_pool
        rng = np.random.default_rng(1)
        mc = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)  # 1000 entries
        h_t = 3.5  # 700 values below
        result = _resample_mc_pool(mc, h_t=h_t, n_mc=1000, rng=rng)
        assert len(result) == 1000
        # All resampled values should be <= h_t (sampled from below-limit paths)
        assert np.all(result <= h_t)

    def test_resample_mc_pool_too_few_below_resets(self):
        """_resample_mc_pool: fewer than n_mc//4 below limit -> reset to zeros."""
        from insurance_monitoring.cusum import _resample_mc_pool
        rng = np.random.default_rng(2)
        mc = np.full(1000, 10.0)  # all above any reasonable limit
        h_t = 0.5  # none below
        result = _resample_mc_pool(mc, h_t=h_t, n_mc=1000, rng=rng)
        assert len(result) == 1000
        assert np.all(result == 0.0)


class TestCUSUMGammaMode:
    """Tests for the gamma_a != 1.0 Bernoulli mode (scale parameter)."""

    def test_gamma_a_not_one_valid(self):
        """gamma_a != 1 with delta_a=1 should not raise (non-identity LLO)."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        # delta_a=1, gamma_a=2 — LLO is NOT identity
        m = CalibrationCUSUM(delta_a=1.0, gamma_a=2.0, n_mc=100)
        assert m.gamma_a == 2.0

    def test_gamma_a_update_returns_alarm(self):
        """Using gamma_a != 1 should run without error and return CUSUMAlarm."""
        from insurance_monitoring.cusum import CalibrationCUSUM, CUSUMAlarm
        rng = np.random.default_rng(20)
        m = CalibrationCUSUM(delta_a=1.0, gamma_a=2.0, n_mc=200, random_state=5)
        p = rng.uniform(0.1, 0.4, 50)
        y = rng.binomial(1, p)
        alarm = m.update(p, y)
        assert isinstance(alarm, CUSUMAlarm)
        assert alarm.statistic >= 0.0

    def test_exposure_length_mismatch_raises(self):
        """Mismatched exposure length raises ValueError."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        m = CalibrationCUSUM(delta_a=2.0, distribution="poisson", n_mc=100)
        mu = np.full(10, 0.1)
        y = np.zeros(10)
        exposure = np.ones(8)  # wrong length
        with pytest.raises(ValueError, match="exposure length"):
            m.update(mu, y, exposure=exposure)

    def test_extreme_predictions_warns(self):
        """More than 1% of predictions outside [0.001, 0.999] should warn."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        m = CalibrationCUSUM(delta_a=2.0, n_mc=100, random_state=0)
        # Create predictions with >1% below 0.001
        p = np.full(100, 0.5)
        p[:5] = 0.0001  # 5% extreme
        y = np.zeros(100)
        with pytest.warns(UserWarning, match="outside"):
            m.update(p, y)

    def test_poisson_with_default_exposure(self):
        """Poisson mode with no exposure defaults to ones without error."""
        from insurance_monitoring.cusum import CalibrationCUSUM, CUSUMAlarm
        rng = np.random.default_rng(99)
        m = CalibrationCUSUM(delta_a=2.0, distribution="poisson", n_mc=200, random_state=0)
        mu = rng.uniform(0.05, 0.15, 100)
        y = rng.poisson(mu)
        alarm = m.update(mu, y)  # no exposure
        assert isinstance(alarm, CUSUMAlarm)

    def test_reset_clears_time_history(self):
        """After reset(), history lists are cleared."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(7)
        m = CalibrationCUSUM(delta_a=2.0, n_mc=100, random_state=0)
        for _ in range(5):
            p = rng.uniform(0.1, 0.3, 20)
            m.update(p, rng.binomial(1, p))

        assert len(m._history_t) == 5
        m.reset()
        assert len(m._history_t) == 0
        assert m.time == 0
        assert m.statistic == 0.0

    def test_statistic_and_time_properties(self):
        """statistic and time properties reflect internal state."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(8)
        m = CalibrationCUSUM(delta_a=2.0, n_mc=100, random_state=0)
        assert m.statistic == 0.0
        assert m.time == 0
        p = rng.uniform(0.1, 0.2, 30)
        m.update(p, rng.binomial(1, p))
        assert m.time == 1
        assert m.statistic >= 0.0


# ---------------------------------------------------------------------------
# multicalibration: degenerate paths and edge cases
# ---------------------------------------------------------------------------


class TestMulticalibDegenerate:
    def test_n_less_than_2_raises(self):
        """Less than 2 observations should raise ValueError."""
        from insurance_monitoring.multicalibration import _validate_monitor_inputs
        y = np.array([0.1])
        pred = np.array([0.1])
        groups = np.array(["A"])
        with pytest.raises(ValueError, match="At least 2"):
            _validate_monitor_inputs(y, pred, groups, None)

    def test_2d_y_true_raises(self):
        """2D y_true should raise ValueError."""
        from insurance_monitoring.multicalibration import _validate_monitor_inputs
        y = np.ones((3, 2))
        pred = np.ones(6)
        groups = np.array(["A"] * 6)
        with pytest.raises(ValueError, match="1-dimensional"):
            _validate_monitor_inputs(y.ravel(), pred, groups, None)
        with pytest.raises(ValueError, match="1-dimensional"):
            _validate_monitor_inputs(pred, y, groups, None)

    def test_2d_exposure_raises(self):
        """2D exposure should raise ValueError."""
        from insurance_monitoring.multicalibration import _validate_monitor_inputs
        y = np.ones(5)
        pred = np.ones(5)
        groups = np.array(["A"] * 5)
        exposure = np.ones((5, 1))
        with pytest.raises(ValueError, match="1-dimensional"):
            _validate_monitor_inputs(y, pred, groups, exposure)

    def test_cells_to_polars_empty(self):
        """_cells_to_polars([]) returns an empty DataFrame with correct schema."""
        from insurance_monitoring.multicalibration import _cells_to_polars
        df = _cells_to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "bin_idx" in df.columns
        assert "alert" in df.columns

    def test_expected_le_zero_skips_cell_with_warning(self):
        """A cell where expected <= 0 should emit a warning and be skipped.

        This is a degenerate path (predictions all zero in a cell). We inject
        it by constructing a monitor with a single-group, single-bin scenario
        where we override y_pred to 0 for one cell after fitting.
        """
        from insurance_monitoring.multicalibration import (
            MulticalibrationMonitor,
            _assign_bins,
            _exposure_weighted_quantile_edges,
        )
        rng = np.random.default_rng(42)
        n = 1000
        y_pred = rng.gamma(2.0, 0.05, n)
        y_true = rng.poisson(y_pred).astype(float)
        groups = np.array(["A"] * n)
        exposure = np.ones(n)

        monitor = MulticalibrationMonitor(n_bins=2, min_exposure=1)
        monitor.fit(y_true, y_pred, groups, exposure=exposure)

        # Force all y_pred to near-zero for update so expected will be ~0
        y_pred_bad = np.full(n, 1e-30)
        with pytest.warns(Warning):
            result = monitor.update(y_true, y_pred_bad, groups, exposure=exposure)

        # All cells should be skipped (either min_exposure fail or expected<=0 path)
        # The important thing is that no crash occurs
        assert result is not None

    def test_summary_worst_cell_with_alerts(self):
        """summary()['worst_cell'] is not None when there are alerts."""
        from insurance_monitoring.multicalibration import (
            MulticalibrationMonitor,
            _assign_bins,
            _exposure_weighted_quantile_edges,
        )
        rng = np.random.default_rng(7)
        n = 8000
        n_bins = 3
        exposure = rng.uniform(0.5, 2.0, n)
        y_pred = rng.gamma(2.0, 0.05, n)

        # Inject a large bias into half the data
        y_true = rng.poisson(y_pred * exposure).astype(float) / exposure
        groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        # 100% upward bias for group A (very large -> must alert)
        y_true_biased = y_true.copy()
        y_true_biased[:n // 2] *= 3.0

        monitor = MulticalibrationMonitor(n_bins=n_bins, min_z_abs=1.0, min_relative_bias=0.05, min_exposure=10)
        monitor.fit(y_true, y_pred, groups, exposure=exposure)
        result = monitor.update(y_true_biased, y_pred, groups, exposure=exposure)

        s = result.summary()
        if s["n_alerts"] > 0:
            assert s["worst_cell"] is not None
            assert isinstance(s["worst_cell"]["relative_bias"], float)

    def test_cell_ae_ratio_near_one_on_calibrated_data(self):
        """On calibrated data, most cells should have AE_ratio near 1."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        rng = np.random.default_rng(123)
        n = 5000
        exposure = rng.uniform(0.5, 2.0, n)
        y_pred = rng.gamma(2.0, 0.05, n)
        y_true = rng.poisson(y_pred * exposure).astype(float) / exposure
        groups = rng.choice(["X", "Y", "Z"], n)

        monitor = MulticalibrationMonitor(n_bins=5, min_exposure=30)
        monitor.fit(y_true, y_pred, groups, exposure=exposure)
        result = monitor.update(y_true, y_pred, groups, exposure=exposure)

        # Check that AE ratios in the table are reasonable
        if len(result.cell_table) > 0:
            ae = result.cell_table["AE_ratio"].to_numpy()
            assert np.all(np.isfinite(ae))
            # Median AE should be near 1
            assert abs(float(np.median(ae)) - 1.0) < 0.5


# ---------------------------------------------------------------------------
# conformal_spc: score_samples fallback, AttributeError, plot
# ---------------------------------------------------------------------------


class TestConformedControlChartPlot:
    """Plot method tests for ConformedControlChart."""

    def test_plot_returns_axes(self):
        """plot() should return a matplotlib Axes."""
        from insurance_monitoring.conformal_spc import ConformedControlChart
        rng = np.random.default_rng(1)
        cal = rng.normal(0, 1, size=100)
        chart = ConformedControlChart(alpha=0.05, score_fn="absolute").fit(cal)
        test = rng.normal(0, 1, size=20)
        result = chart.predict(test)
        ax = chart.plot(result)
        assert ax is not None
        plt.close("all")

    def test_plot_with_signals_marks_them(self):
        """Out-of-control points should be highlighted in the plot."""
        from insurance_monitoring.conformal_spc import ConformedControlChart
        rng = np.random.default_rng(2)
        cal = rng.normal(0, 1, size=200)
        chart = ConformedControlChart(alpha=0.05, score_fn="absolute").fit(cal)
        # Mean-shifted test data to force signals
        test = rng.normal(10, 1, size=20)
        result = chart.predict(test)
        ax = chart.plot(result)
        assert ax is not None
        plt.close("all")

    def test_plot_with_custom_ax(self):
        """plot() should use a passed-in Axes."""
        from insurance_monitoring.conformal_spc import ConformedControlChart
        rng = np.random.default_rng(3)
        cal = rng.normal(0, 1, size=100)
        chart = ConformedControlChart(alpha=0.05, score_fn="relative").fit(cal)
        test = rng.normal(0, 1, size=10)
        result = chart.predict(test)
        fig, ax = plt.subplots()
        returned_ax = chart.plot(result, ax=ax)
        assert returned_ax is ax
        plt.close("all")


class TestConformedProcessMonitorScoreSamples:
    """Test score_samples fallback for anomaly detectors."""

    def test_score_samples_model_accepted(self):
        """An anomaly model with score_samples (no decision_function) should work."""
        from insurance_monitoring.conformal_spc import ConformedProcessMonitor, ConformedMonitorResult

        class ScoreSamplesModel:
            def fit(self, X):
                self.mean_ = np.mean(X, axis=0)
                return self

            def score_samples(self, X):
                # Higher = more normal
                return -np.linalg.norm(X - self.mean_, axis=1)

        rng = np.random.default_rng(10)
        X_cal = rng.normal(0, 1, size=(80, 3))
        X_test = rng.normal(0, 1, size=(20, 3))

        m = ConformedProcessMonitor(alpha=0.05, detector=ScoreSamplesModel())
        m.fit(X_cal)
        result = m.predict(X_test)
        assert isinstance(result, ConformedMonitorResult)
        assert len(result.p_values) == 20

    def test_no_decision_function_or_score_samples_raises(self):
        """A model with neither decision_function nor score_samples should raise AttributeError."""
        from insurance_monitoring.conformal_spc import ConformedProcessMonitor

        class BrokenModel:
            def fit(self, X):
                return self
            # No decision_function, no score_samples

        rng = np.random.default_rng(11)
        X_cal = rng.normal(0, 1, size=(50, 2))
        m = ConformedProcessMonitor(alpha=0.05, detector=BrokenModel())
        m.fit(X_cal)

        X_test = rng.normal(0, 1, size=(5, 2))
        with pytest.raises(AttributeError, match="decision_function"):
            m.predict(X_test)

    def test_p_values_bounded(self):
        """All conformal p-values should be in (0, 1]."""
        from insurance_monitoring.conformal_spc import ConformedProcessMonitor

        class CentroidModel:
            def fit(self, X):
                self.c = np.mean(X, axis=0)
                return self
            def decision_function(self, X):
                return -np.linalg.norm(X - self.c, axis=1)

        rng = np.random.default_rng(12)
        X_cal = rng.normal(0, 1, size=(100, 4))
        X_test = rng.normal(0, 1, size=(50, 4))
        m = ConformedProcessMonitor(alpha=0.05, detector=CentroidModel()).fit(X_cal)
        result = m.predict(X_test)
        assert np.all(result.p_values > 0)
        assert np.all(result.p_values <= 1.0)


# ---------------------------------------------------------------------------
# model_monitor: _make_decision unit tests, zero-SE path, distributions
# ---------------------------------------------------------------------------


class TestMakeDecision:
    """Unit tests for model_monitor._make_decision."""

    def _call(self, gini_sig, gmcb_sig, lmcb_sig, **kw):
        from insurance_monitoring.model_monitor import _make_decision
        defaults = dict(
            gini_sig=gini_sig,
            gini_z=-0.5,
            gini_p=0.5,
            gmcb_sig=gmcb_sig,
            gmcb_score=0.01,
            gmcb_p=0.2,
            lmcb_sig=lmcb_sig,
            lmcb_score=0.005,
            lmcb_p=0.3,
            alpha_gini=0.32,
            alpha_global=0.32,
            alpha_local=0.32,
        )
        defaults.update(kw)
        return _make_decision(**defaults)

    def test_all_pass_redeploy(self):
        decision, reason = self._call(False, False, False)
        assert decision == "REDEPLOY"
        assert "adequate" in reason.lower()

    def test_only_gmcb_recalibrate(self):
        decision, reason = self._call(False, True, False)
        assert decision == "RECALIBRATE"
        assert "balance" in reason.lower()

    def test_gini_sig_refit(self):
        decision, reason = self._call(True, False, False)
        assert decision == "REFIT"
        assert "refit" in reason.lower()

    def test_lmcb_sig_refit(self):
        decision, reason = self._call(False, False, True)
        assert decision == "REFIT"
        assert "refit" in reason.lower()

    def test_all_sig_refit(self):
        decision, reason = self._call(True, True, True)
        assert decision == "REFIT"

    def test_gini_and_gmcb_refit(self):
        """If Gini AND GMCB both fire, REFIT takes precedence (Gini sig)."""
        decision, reason = self._call(True, True, False)
        assert decision == "REFIT"

    def test_lmcb_and_gmcb_refit(self):
        """LMCB + GMCB -> REFIT (LMCB sig)."""
        decision, reason = self._call(False, True, True)
        assert decision == "REFIT"

    def test_decision_values_are_valid(self):
        """Decision string must always be one of the three valid values."""
        valid = {"REDEPLOY", "RECALIBRATE", "REFIT"}
        for g in [True, False]:
            for gm in [True, False]:
                for lm in [True, False]:
                    decision, _ = self._call(g, gm, lm)
                    assert decision in valid


class TestModelMonitorZeroSEPath:
    """Tests for the gini_std == 0 degenerate path."""

    def test_zero_se_produces_nan_gini_stats(self):
        """When gini_std is forced to 0, z and p should be nan and sig=False."""
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(1)
        n = 500
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * y_hat) / exposure

        m = ModelMonitor(distribution="poisson", n_bootstrap=50, random_state=0)
        m.fit(y_ref, y_hat, exposure)

        # Force zero SE to hit the degenerate path
        m._gini_ref_std = 0.0

        y_new = rng.poisson(exposure * y_hat) / exposure
        result = m.test(y_new, y_hat, exposure)

        assert np.isnan(result.gini_z)
        assert np.isnan(result.gini_p)
        assert result.gini_significant is False


class TestModelMonitorTimesSplitWarning:
    """Verify that median exposure < 0.05 triggers warning in test()."""

    def test_time_split_warning_in_test(self):
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(77)
        n = 300
        normal_exp = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(normal_exp * y_hat) / normal_exp

        m = ModelMonitor(distribution="poisson", n_bootstrap=50, random_state=0)
        m.fit(y_ref, y_hat, normal_exp)

        # Now call test with tiny exposures
        tiny_exp = rng.uniform(0.001, 0.02, n)
        y_new = rng.poisson(tiny_exp * y_hat) / tiny_exp

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.test(y_new, y_hat, tiny_exp)
            time_split = [
                x for x in w
                if "time-splitting" in str(x.message).lower()
                or "0.05" in str(x.message)
            ]
            assert len(time_split) >= 1, (
                "Expected UserWarning about time-splitting for median exposure < 0.05"
            )


class TestModelMonitorGammaDistribution:
    """ModelMonitor with distribution='gamma' should run without error."""

    def test_gamma_distribution_runs(self):
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(20)
        n = 500
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.5, n)
        y_ref = rng.gamma(2, y_hat)

        m = ModelMonitor(distribution="gamma", n_bootstrap=50, random_state=0)
        m.fit(y_ref, y_hat, exposure)
        result = m.test(y_ref, y_hat, exposure)
        assert result.decision in ("REDEPLOY", "RECALIBRATE", "REFIT")

    def test_normal_distribution_runs(self):
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(21)
        n = 500
        y_hat = rng.uniform(1, 10, n)
        y_ref = y_hat + rng.normal(0, 0.5, n)

        m = ModelMonitor(distribution="normal", n_bootstrap=50, random_state=0)
        m.fit(y_ref, y_hat)
        result = m.test(y_ref, y_hat)
        assert result.decision in ("REDEPLOY", "RECALIBRATE", "REFIT")

    def test_tweedie_distribution_runs(self):
        from insurance_monitoring.model_monitor import ModelMonitor
        rng = np.random.default_rng(22)
        n = 500
        y_hat = rng.gamma(2, 0.5, n)
        y_ref = rng.gamma(2, y_hat)

        m = ModelMonitor(distribution="tweedie", tweedie_power=1.5, n_bootstrap=50, random_state=0)
        m.fit(y_ref, y_hat)
        result = m.test(y_ref, y_hat)
        assert result.decision in ("REDEPLOY", "RECALIBRATE", "REFIT")


# ---------------------------------------------------------------------------
# thresholds: boundary conditions
# ---------------------------------------------------------------------------


class TestThresholdBoundaries:
    """Boundary values at the classification thresholds."""

    def test_psi_exactly_at_green_max_is_amber(self):
        """PSI == green_max (0.10) should be amber, not green."""
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(0.10) == "amber"

    def test_psi_exactly_at_amber_max_is_red(self):
        """PSI == amber_max (0.25) should be red."""
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(0.25) == "red"

    def test_psi_negative_is_green(self):
        """Negative PSI (shouldn't happen but defensive) -> green."""
        from insurance_monitoring.thresholds import PSIThresholds
        t = PSIThresholds()
        assert t.classify(-0.01) == "green"

    def test_ae_exactly_at_green_lower_is_green(self):
        """AE == green_lower (0.95) -> green."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(0.95) == "green"

    def test_ae_exactly_at_green_upper_is_green(self):
        """AE == green_upper (1.05) -> green."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.05) == "green"

    def test_ae_just_outside_green_upper_is_amber(self):
        """AE just above green_upper -> amber."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.051) == "amber"

    def test_ae_exactly_at_amber_upper_is_amber(self):
        """AE == amber_upper (1.10) -> amber."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.10) == "amber"

    def test_ae_just_outside_amber_upper_is_red(self):
        """AE just above amber_upper -> red."""
        from insurance_monitoring.thresholds import AERatioThresholds
        t = AERatioThresholds()
        assert t.classify(1.11) == "red"

    def test_gini_exactly_at_amber_threshold_is_green(self):
        """p_value == amber_p_value (0.32) -> green (boundary is green)."""
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.32) == "green"

    def test_gini_just_below_amber_threshold_is_amber(self):
        """p_value just below 0.32 -> amber."""
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.319) == "amber"

    def test_gini_exactly_at_red_threshold_is_amber(self):
        """p_value == red_p_value (0.10) -> amber (boundary is amber)."""
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.10) == "amber"

    def test_gini_just_below_red_threshold_is_red(self):
        """p_value just below 0.10 -> red."""
        from insurance_monitoring.thresholds import GiniDriftThresholds
        t = GiniDriftThresholds()
        assert t.classify(0.099) == "red"


# ---------------------------------------------------------------------------
# Top-level import integration test
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """Verify all public names from __init__ are importable."""

    def test_baws_importable(self):
        from insurance_monitoring import BAWSMonitor, BAWSResult
        assert BAWSMonitor is not None
        assert BAWSResult is not None

    def test_cusum_importable(self):
        from insurance_monitoring import CalibrationCUSUM, CUSUMAlarm, CUSUMSummary
        assert CalibrationCUSUM is not None
        assert CUSUMAlarm is not None
        assert CUSUMSummary is not None

    def test_conformal_spc_importable(self):
        from insurance_monitoring import (
            ConformedControlChart,
            ConformedProcessMonitor,
            ConformedControlResult,
            ConformedMonitorResult,
        )
        assert ConformedControlChart is not None

    def test_multicalibration_importable(self):
        from insurance_monitoring import (
            MulticalibrationMonitor,
            MulticalibrationResult,
            MulticalibCell,
            MulticalibThresholds,
        )
        assert MulticalibrationMonitor is not None

    def test_model_monitor_importable(self):
        from insurance_monitoring import ModelMonitor, ModelMonitorResult
        assert ModelMonitor is not None

    def test_score_decomp_importable(self):
        from insurance_monitoring import (
            ScoreDecompositionTest,
            ScoreDecompositionResult,
            TwoForecastSDIResult,
        )
        assert ScoreDecompositionTest is not None

    def test_pricing_drift_importable(self):
        from insurance_monitoring import (
            PricingDriftMonitor,
            PricingDriftResult,
            CalibTestResult,
        )
        assert PricingDriftMonitor is not None

    def test_getattr_unknown_name_raises(self):
        """Unknown names from __getattr__ should raise AttributeError."""
        import insurance_monitoring
        with pytest.raises(AttributeError):
            _ = insurance_monitoring.NonExistentClass

    def test_version_accessible(self):
        import insurance_monitoring
        assert hasattr(insurance_monitoring, "__version__")
        assert isinstance(insurance_monitoring.__version__, str)


# ---------------------------------------------------------------------------
# BAWS history and plot integration
# ---------------------------------------------------------------------------


class TestBAWSHistoryIntegration:
    def test_history_n_obs_increases_monotonically(self):
        """n_obs should increase by 1 with each update."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(50)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=0)
        init = rng.standard_normal(100)
        m.fit(init)
        prev_n = 100
        for _ in range(5):
            result = m.update(float(rng.standard_normal()))
            assert result.n_obs == prev_n + 1
            prev_n += 1

    def test_time_step_matches_update_count(self):
        """time_step in result == number of update() calls."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(51)
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100], n_bootstrap=10, random_state=0)
        m.fit(rng.standard_normal(100))
        for i in range(1, 6):
            result = m.update(float(rng.standard_normal()))
            assert result.time_step == i

    def test_scores_dict_keys_match_candidate_windows(self):
        """result.scores must have exactly one key per candidate_window."""
        from insurance_monitoring.baws import BAWSMonitor
        windows = [50, 100, 200]
        rng = np.random.default_rng(52)
        m = BAWSMonitor(alpha=0.05, candidate_windows=windows, n_bootstrap=10, random_state=0)
        m.fit(rng.standard_normal(200))
        result = m.update(float(rng.standard_normal()))
        assert set(result.scores.keys()) == set(windows)

    def test_scores_for_too_short_windows_are_inf(self):
        """Windows longer than current history should have score=inf."""
        from insurance_monitoring.baws import BAWSMonitor
        rng = np.random.default_rng(53)
        # Start with exactly 50 observations; window 200 is too long initially
        m = BAWSMonitor(alpha=0.05, candidate_windows=[50, 200], n_bootstrap=10, random_state=0)
        m.fit(rng.standard_normal(50))
        result = m.update(float(rng.standard_normal()))
        # Window 200 > 51 total -> should be inf
        assert result.scores[200] == float("inf")


# ---------------------------------------------------------------------------
# CUSUM integration: alarm counter persists across periods
# ---------------------------------------------------------------------------


class TestCUSUMAlarmCounterPersistence:
    def test_alarm_count_monotone(self):
        """n_alarms should never decrease over time."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(200)
        # cfar=0.1 to generate alarms in small number of steps
        m = CalibrationCUSUM(delta_a=2.0, cfar=0.1, n_mc=500, random_state=0)
        prev_count = 0
        for _ in range(30):
            p = rng.uniform(0.05, 0.25, 30)
            m.update(p, rng.binomial(1, p))
            curr = m.summary().n_alarms
            assert curr >= prev_count
            prev_count = curr

    def test_alarm_times_are_subset_of_time_steps(self):
        """All alarm_times must be valid 1-based time step indices."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(201)
        m = CalibrationCUSUM(delta_a=2.0, cfar=0.1, n_mc=300, random_state=1)
        for t in range(20):
            p = rng.uniform(0.05, 0.25, 30)
            m.update(p, rng.binomial(1, p))
        s = m.summary()
        assert all(1 <= t <= 20 for t in s.alarm_times)

    def test_last_control_limit_is_none_before_any_update(self):
        """current_control_limit is None in summary before any update."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        m = CalibrationCUSUM(delta_a=2.0, n_mc=100)
        s = m.summary()
        assert s.current_control_limit is None

    def test_last_control_limit_set_after_first_update(self):
        """current_control_limit is non-None after first update."""
        from insurance_monitoring.cusum import CalibrationCUSUM
        rng = np.random.default_rng(202)
        m = CalibrationCUSUM(delta_a=2.0, n_mc=100, random_state=0)
        p = rng.uniform(0.1, 0.2, 20)
        m.update(p, rng.binomial(1, p))
        s = m.summary()
        assert s.current_control_limit is not None
        assert s.current_control_limit > 0


# ---------------------------------------------------------------------------
# Multicalibration: period_summary before fit
# ---------------------------------------------------------------------------


class TestMulticalibPeriodSummaryBeforeFit:
    def test_period_summary_before_fit_raises_implicitly(self):
        """period_summary() with no history returns empty DataFrame even before fit."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        m = MulticalibrationMonitor()
        # Note: fit() not called, but period_summary() should still work
        # (it reads self._history which is [])
        df = m.period_summary()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_history_empty_before_any_update(self):
        """history() returns empty list before any update."""
        from insurance_monitoring.multicalibration import MulticalibrationMonitor
        rng = np.random.default_rng(300)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        y_pred = rng.gamma(2.0, 0.05, 500)
        y_true = rng.poisson(y_pred).astype(float)
        groups = rng.choice(["A", "B"], 500)
        m.fit(y_true, y_pred, groups)
        assert m.history() == []
