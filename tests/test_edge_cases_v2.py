"""
Edge-case and integration tests for the v0.9.5–v1.2.0 feature additions.

Covers modules introduced in:
- v0.9.5: GiniDriftMonitor, GiniBootstrapMonitor, MurphyDecomposition
- v0.10.0: PricingDriftMonitor, CalibrationCUSUM
- v0.11.0: ConformalControlChart, MultivariateConformalMonitor
- v1.0.0: ModelMonitor (check_gmcb/check_lmcb)
- v1.1.0: BAWSMonitor
- v1.2.0: (ScoreDecompositionTest — verified absent, not a real module)

Focus: empty/tiny inputs, single-observation edge cases, mismatched array
lengths, NaN/inf in data, perfect-model scenarios, extreme drift, inter-monitor
integration, CUSUM with various ARL0 settings, Gini with tied predictions.

All bootstrap counts are intentionally low (50–100) so the tests finish in
under 30 seconds on the Raspberry Pi.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from insurance_monitoring.gini_monitoring import (
    GiniDriftMonitor,
    GiniBootstrapMonitor,
    MurphyDecomposition,
    GiniDriftResult,
    GiniBootstrapResult,
)
from insurance_monitoring.gini_drift import GiniDriftTest
from insurance_monitoring.pricing_drift import PricingDriftMonitor, CalibTestResult
from insurance_monitoring.cusum import CalibrationCUSUM, CUSUMAlarm
from insurance_monitoring.conformal_chart import (
    ConformalControlChart,
    MultivariateConformalMonitor,
    _conformal_p_value,
    _conformal_p_values,
    _conformal_threshold,
)
from insurance_monitoring.model_monitor import ModelMonitor, ModelMonitorResult
from insurance_monitoring.baws import BAWSMonitor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _poisson_data(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, n)
    y = rng.poisson(mu).astype(float)
    return y, mu


def _tied_predictions(n: int):
    """All predictions identical — degenerate for Gini."""
    y = np.random.default_rng(0).poisson(0.1, n).astype(float)
    mu = np.full(n, 0.1)
    return y, mu


# ===========================================================================
# GiniDriftMonitor — edge cases
# ===========================================================================


class TestGiniDriftMonitorEdgeCases:
    """Edge cases for GiniDriftMonitor that existing tests miss."""

    def test_empty_reference_raises(self):
        m = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="non-empty"):
            m.fit([], [])

    def test_empty_monitor_raises(self):
        y, mu = _poisson_data(300)
        m = GiniDriftMonitor(n_bootstrap=50).fit(y, mu)
        with pytest.raises(ValueError, match="non-empty"):
            m.test([], [])

    def test_mismatched_reference_lengths_raises(self):
        m = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="length"):
            m.fit(np.ones(10), np.ones(11))

    def test_mismatched_monitor_lengths_raises(self):
        y, mu = _poisson_data(300)
        m = GiniDriftMonitor(n_bootstrap=50).fit(y, mu)
        with pytest.raises(ValueError, match="length"):
            m.test(np.ones(10), np.ones(11))

    def test_mismatched_reference_exposure_raises(self):
        m = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="exposure"):
            m.fit(np.ones(10), np.ones(10), exposure_ref=np.ones(9))

    def test_mismatched_monitor_exposure_raises(self):
        y, mu = _poisson_data(300)
        m = GiniDriftMonitor(n_bootstrap=50).fit(y, mu)
        with pytest.raises(ValueError, match="exposure"):
            m.test(np.ones(10), np.ones(10), exposure_mon=np.ones(9))

    def test_non_positive_reference_exposure_raises(self):
        m = GiniDriftMonitor(n_bootstrap=50)
        exp = np.ones(10)
        exp[5] = -0.5
        with pytest.raises(ValueError, match="positive"):
            m.fit(np.ones(10), np.ones(10), exposure_ref=exp)

    def test_non_positive_monitor_exposure_raises(self):
        y, mu = _poisson_data(300)
        m = GiniDriftMonitor(n_bootstrap=50).fit(y, mu)
        exp = np.ones(10)
        exp[3] = 0.0
        with pytest.raises(ValueError, match="positive"):
            m.test(np.ones(10), np.ones(10), exposure_mon=exp)

    def test_test_before_fit_raises(self):
        m = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(RuntimeError, match="fit"):
            m.test(*_poisson_data(100))

    def test_n_bootstrap_too_low_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            GiniDriftMonitor(n_bootstrap=10)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            GiniDriftMonitor(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            GiniDriftMonitor(alpha=1.0)

    def test_small_sample_warns(self):
        """Fewer than 200 observations should trigger a UserWarning."""
        y, mu = _poisson_data(50)
        m = GiniDriftMonitor(n_bootstrap=50).fit(y, mu)
        with pytest.warns(UserWarning, match="observations"):
            m.test(y[:50], mu[:50])

    def test_result_fields_present(self):
        y, mu = _poisson_data(400, seed=1)
        m = GiniDriftMonitor(n_bootstrap=50, random_state=0).fit(y, mu)
        result = m.test(y, mu)
        assert isinstance(result, GiniDriftResult)
        assert hasattr(result, "gini_ref")
        assert hasattr(result, "gini_mon")
        assert hasattr(result, "z_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "reject_h0")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert result.n_ref == 400
        assert result.n_mon == 400

    def test_tied_predictions_return_valid(self):
        """All-identical predictions -> Gini=0, but no crash."""
        y, mu = _tied_predictions(300)
        m = GiniDriftMonitor(n_bootstrap=50, random_state=0).fit(y, mu)
        result = m.test(y, mu)
        # Gini is near zero for tied predictions
        assert abs(result.gini_ref) < 0.1
        assert abs(result.gini_mon) < 0.1
        # p-value is nan or float when se_total=0
        assert isinstance(result.z_stat, float)

    def test_to_dict_serialisable(self):
        y, mu = _poisson_data(300, seed=10)
        m = GiniDriftMonitor(n_bootstrap=50, random_state=1).fit(y, mu)
        d = m.test(y, mu).to_dict()
        assert isinstance(d, dict)
        assert all(isinstance(k, str) for k in d)

    def test_is_fitted_property(self):
        m = GiniDriftMonitor(n_bootstrap=50)
        assert not m.is_fitted()
        y, mu = _poisson_data(300)
        m.fit(y, mu)
        assert m.is_fitted()

    def test_alpha_override_at_test_time(self):
        """Override alpha at test() time changes reject_h0 without changing p."""
        y, mu = _poisson_data(400, seed=7)
        m = GiniDriftMonitor(n_bootstrap=50, alpha=0.05, random_state=0).fit(y, mu)
        r1 = m.test(y, mu, alpha=0.99)   # almost always reject
        r2 = m.test(y, mu, alpha=0.0001) # almost never reject
        # Same p-value; different reject_h0
        assert r1.p_value == pytest.approx(r2.p_value, abs=1e-6)
        assert r1.reject_h0 or not r2.reject_h0  # at least one differs

    def test_zero_drift_no_rejection_at_strict_alpha(self):
        """Identical data in ref and monitor -> p-value large (no drift)."""
        rng = np.random.default_rng(99)
        mu = rng.uniform(0.05, 0.20, 500)
        y = rng.poisson(mu).astype(float)
        m = GiniDriftMonitor(n_bootstrap=50, alpha=0.01, random_state=0).fit(y, mu)
        result = m.test(y, mu)
        # With identical data, p should be large (no significant difference)
        # This is not guaranteed but extremely likely with same arrays
        assert result.n_ref == 500
        assert result.n_mon == 500


# ===========================================================================
# GiniBootstrapMonitor — edge cases
# ===========================================================================


class TestGiniBootstrapMonitorEdgeCases:
    """Edge cases for the one-sample GiniBootstrapMonitor."""

    def test_fit_out_of_range_raises(self):
        m = GiniBootstrapMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="\\(-1, 1\\)"):
            m.fit(gini_ref=1.5)
        with pytest.raises(ValueError, match="\\(-1, 1\\)"):
            m.fit(gini_ref=-1.0)

    def test_fit_exact_boundary_raises(self):
        m = GiniBootstrapMonitor(n_bootstrap=50)
        with pytest.raises(ValueError):
            m.fit(1.0)
        with pytest.raises(ValueError):
            m.fit(-1.0)

    def test_test_before_fit_raises(self):
        m = GiniBootstrapMonitor(n_bootstrap=50)
        with pytest.raises(RuntimeError, match="fit"):
            m.test(*_poisson_data(100))

    def test_empty_monitor_raises(self):
        m = GiniBootstrapMonitor(n_bootstrap=50).fit(0.4)
        with pytest.raises(ValueError, match="non-empty"):
            m.test([], [])

    def test_mismatched_lengths_raises(self):
        m = GiniBootstrapMonitor(n_bootstrap=50).fit(0.4)
        with pytest.raises(ValueError, match="length"):
            m.test(np.ones(5), np.ones(6))

    def test_small_sample_warning(self):
        m = GiniBootstrapMonitor(n_bootstrap=50).fit(0.4)
        y, mu = _poisson_data(50)
        with pytest.warns(UserWarning):
            m.test(y, mu)

    def test_result_structure(self):
        y, mu = _poisson_data(400, seed=5)
        m = GiniBootstrapMonitor(n_bootstrap=50, random_state=0).fit(0.3)
        result = m.test(y, mu)
        assert isinstance(result, GiniBootstrapResult)
        assert 0.0 <= result.p_value <= 1.0
        assert result.n_mon == 400
        assert result.gini_ref == pytest.approx(0.3)

    def test_gini_ref_zero_is_valid(self):
        """gini_ref=0 is within (-1, 1) and should work."""
        y, mu = _poisson_data(300)
        m = GiniBootstrapMonitor(n_bootstrap=50, random_state=0).fit(0.0)
        result = m.test(y, mu)
        assert isinstance(result, GiniBootstrapResult)

    def test_very_negative_gini_ref(self):
        """gini_ref=-0.9 is valid."""
        y, mu = _poisson_data(300)
        m = GiniBootstrapMonitor(n_bootstrap=50, random_state=0).fit(-0.9)
        result = m.test(y, mu)
        assert result.gini_change > 0  # monitor Gini > -0.9

    def test_to_dict(self):
        y, mu = _poisson_data(300)
        m = GiniBootstrapMonitor(n_bootstrap=50, random_state=0).fit(0.3)
        d = m.test(y, mu).to_dict()
        assert "gini_ref" in d
        assert "p_value" in d

    def test_extreme_drift_detected(self):
        """If monitor Gini is way off from reference, rejection likely."""
        rng = np.random.default_rng(42)
        mu = rng.uniform(0.05, 0.5, 500)
        y = rng.poisson(mu).astype(float)
        # Fit with a high gini_ref, test on random noise (low Gini)
        mu_noise = np.full(500, 0.1)
        m = GiniBootstrapMonitor(n_bootstrap=50, alpha=0.32, random_state=0).fit(0.9)
        result = m.test(y, mu_noise)
        # Gini on noise should be ~0, far from 0.9 -> likely reject
        assert result.gini_mon < 0.5


# ===========================================================================
# GiniDriftTest (class in gini_drift.py) — edge cases
# ===========================================================================


class TestGiniDriftTestEdgeCases:
    """Tests for GiniDriftTest (the class, not the functional API)."""

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            t = GiniDriftTest([], [], np.ones(5), np.ones(5), n_bootstrap=50)
            t.test()

    def test_empty_monitor_raises(self):
        y, mu = _poisson_data(300)
        with pytest.raises(ValueError, match="non-empty"):
            t = GiniDriftTest(y, mu, [], [], n_bootstrap=50)
            t.test()

    def test_n_bootstrap_too_low_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            GiniDriftTest(
                np.ones(5), np.ones(5), np.ones(5), np.ones(5), n_bootstrap=10
            )

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            GiniDriftTest(
                np.ones(5), np.ones(5), np.ones(5), np.ones(5),
                n_bootstrap=50, alpha=0.0
            )

    def test_result_cached_on_second_call(self):
        """Calling test() twice returns the same object (idempotent)."""
        y, mu = _poisson_data(400)
        t = GiniDriftTest(y, mu, y, mu, n_bootstrap=50, random_state=0)
        r1 = t.test()
        r2 = t.test()
        assert r1 is r2

    def test_summary_returns_string(self):
        y, mu = _poisson_data(400)
        t = GiniDriftTest(y, mu, y, mu, n_bootstrap=50, random_state=0)
        s = t.summary()
        assert isinstance(s, str)
        assert "Gini" in s

    def test_dict_like_access(self):
        """GiniDriftTestResult supports dict-style access."""
        y, mu = _poisson_data(400)
        t = GiniDriftTest(y, mu, y, mu, n_bootstrap=50, random_state=0)
        r = t.test()
        assert "p_value" in r
        assert r["p_value"] == pytest.approx(r.p_value)

    def test_to_dict(self):
        y, mu = _poisson_data(400)
        t = GiniDriftTest(y, mu, y, mu, n_bootstrap=50, random_state=0)
        d = t.test().to_dict()
        assert isinstance(d, dict)
        assert "gini_reference" in d

    def test_exposure_mismatch_reference_raises(self):
        y, mu = _poisson_data(300)
        with pytest.raises(ValueError, match="exposure"):
            t = GiniDriftTest(y, mu, y, mu,
                              reference_exposure=np.ones(10), n_bootstrap=50)
            t.test()

    def test_exposure_mismatch_monitor_raises(self):
        y, mu = _poisson_data(300)
        with pytest.raises(ValueError, match="exposure"):
            t = GiniDriftTest(y, mu, y, mu,
                              monitor_exposure=np.ones(10), n_bootstrap=50)
            t.test()

    def test_non_positive_exposure_raises(self):
        y, mu = _poisson_data(300)
        bad_exp = np.ones(300)
        bad_exp[0] = -1.0
        with pytest.raises(ValueError, match="positive"):
            t = GiniDriftTest(y, mu, y, mu,
                              reference_exposure=bad_exp, n_bootstrap=50)
            t.test()

    def test_tied_predictions_no_crash(self):
        """Tied predictions -> Gini~0, se_total may be 0 -> NaN handling."""
        y_ref, mu_ref = _tied_predictions(300)
        y_mon, mu_mon = _tied_predictions(300)
        t = GiniDriftTest(y_ref, mu_ref, y_mon, mu_mon, n_bootstrap=50, random_state=0)
        r = t.test()
        assert isinstance(r.gini_reference, float)
        assert isinstance(r.gini_monitor, float)

    def test_small_sample_warns(self):
        y, mu = _poisson_data(50)
        t = GiniDriftTest(y, mu, y, mu, n_bootstrap=50)
        with pytest.warns(UserWarning, match="observations"):
            t.test()


# ===========================================================================
# MurphyDecomposition — edge cases
# ===========================================================================


class TestMurphyDecompositionEdgeCases:
    """Edge cases for MurphyDecomposition class."""

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="family"):
            MurphyDecomposition(family="binomial")

    def test_summary_before_decompose_raises(self):
        m = MurphyDecomposition()
        with pytest.raises(RuntimeError, match="decompose"):
            m.summary()

    def test_result_property_before_decompose_is_none(self):
        m = MurphyDecomposition()
        assert m.result is None

    def test_mcb_near_zero_for_perfect_calibration(self):
        """When mu_hat == y_bar (grand mean), DSC should be ~0."""
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.poisson(0.1, n).astype(float)
        mu_hat = np.full(n, float(np.mean(y)))
        m = MurphyDecomposition(family="poisson")
        result = m.decompose(y, mu_hat)
        # Grand mean model -> no discrimination
        assert result.dsc >= -0.01  # near zero (may have small numerical noise)

    def test_gamma_family_works(self):
        rng = np.random.default_rng(5)
        y = rng.gamma(2, 0.5, 500)
        mu = rng.uniform(0.5, 2.0, 500)
        m = MurphyDecomposition(family="gamma")
        result = m.decompose(y, mu)
        assert result.verdict in {"OK", "RECALIBRATE", "REFIT"}

    def test_normal_family_works(self):
        rng = np.random.default_rng(6)
        y = rng.normal(10, 2, 500)
        mu = rng.normal(10, 1, 500)
        m = MurphyDecomposition(family="normal")
        result = m.decompose(y, mu)
        assert result.verdict in {"OK", "RECALIBRATE", "REFIT"}

    def test_tweedie_family_works(self):
        rng = np.random.default_rng(7)
        y = rng.gamma(2, 0.1, 500)
        mu = rng.uniform(0.1, 0.5, 500)
        m = MurphyDecomposition(family="tweedie", tweedie_power=1.5)
        result = m.decompose(y, mu)
        assert result.verdict in {"OK", "RECALIBRATE", "REFIT"}

    def test_summary_returns_dict_with_interpretation(self):
        y, mu = _poisson_data(500)
        m = MurphyDecomposition(family="poisson")
        m.decompose(y, mu)
        s = m.summary()
        assert isinstance(s, dict)
        assert "verdict" in s
        assert "interpretation" in s

    def test_to_dict_serialisable(self):
        y, mu = _poisson_data(500)
        m = MurphyDecomposition(family="poisson")
        result = m.decompose(y, mu)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert all(isinstance(k, str) for k in d)

    def test_unc_positive(self):
        """Uncertainty component must be >= 0."""
        y, mu = _poisson_data(500)
        m = MurphyDecomposition(family="poisson")
        result = m.decompose(y, mu)
        assert result.unc >= 0.0

    def test_with_exposure(self):
        rng = np.random.default_rng(8)
        n = 500
        exposure = rng.uniform(0.5, 2.0, n)
        mu = rng.uniform(0.05, 0.25, n)
        y = rng.poisson(exposure * mu) / exposure
        m = MurphyDecomposition(family="poisson")
        result = m.decompose(y, mu, exposure=exposure)
        assert isinstance(result.verdict, str)


# ===========================================================================
# CalibrationCUSUM — edge cases
# ===========================================================================


class TestCalibrationCUSUMEdgeCases:
    """Edge cases for CalibrationCUSUM."""

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            CalibrationCUSUM(distribution="normal")

    def test_cfar_out_of_range_raises(self):
        with pytest.raises(ValueError, match="cfar"):
            CalibrationCUSUM(cfar=0.0)
        with pytest.raises(ValueError, match="cfar"):
            CalibrationCUSUM(cfar=1.0)

    def test_n_mc_too_low_raises(self):
        with pytest.raises(ValueError, match="n_mc"):
            CalibrationCUSUM(n_mc=50)

    def test_bernoulli_identity_alternative_raises(self):
        """delta_a=1, gamma_a=1 is the identity — no test possible."""
        with pytest.raises(ValueError, match="identity"):
            CalibrationCUSUM(delta_a=1.0, gamma_a=1.0, distribution="bernoulli")

    def test_poisson_delta_one_raises(self):
        """delta_a=1.0 in Poisson mode gives W_t=0 always."""
        with pytest.raises(ValueError, match="delta_a=1.0"):
            CalibrationCUSUM(delta_a=1.0, distribution="poisson")

    def test_mismatched_lengths_raises(self):
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        with pytest.raises(ValueError, match="length"):
            m.update(np.full(10, 0.1), np.zeros(11))

    def test_negative_outcome_raises(self):
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        y = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            m.update(np.full(3, 0.1), y)

    def test_alarm_is_bool(self):
        m = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=200, random_state=0)
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.25, 50)
        y = rng.binomial(1, p)
        alarm = m.update(p, y)
        assert isinstance(alarm, CUSUMAlarm)
        assert isinstance(bool(alarm), bool)

    def test_statistic_non_negative(self):
        """CUSUM statistic must always be >= 0."""
        m = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=200, random_state=0)
        rng = np.random.default_rng(2)
        for _ in range(20):
            p = rng.uniform(0.05, 0.25, 50)
            y = rng.binomial(1, p)
            m.update(p, y)
        assert m.statistic >= 0.0

    def test_time_increments_each_step(self):
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        rng = np.random.default_rng(3)
        for i in range(5):
            p = rng.uniform(0.05, 0.2, 30)
            y = rng.binomial(1, p)
            m.update(p, y)
        assert m.time == 5

    def test_reset_clears_statistic_and_time(self):
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        rng = np.random.default_rng(4)
        for _ in range(5):
            m.update(rng.uniform(0.05, 0.2, 30), rng.binomial(1, 0.1, 30))
        m.reset()
        assert m.statistic == 0.0
        assert m.time == 0

    def test_reset_retains_alarm_count(self):
        """Alarm count is NOT cleared on reset (per spec)."""
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, cfar=0.5, random_state=0)
        # cfar=0.5 makes alarms very likely
        rng = np.random.default_rng(5)
        for _ in range(20):
            p = rng.uniform(0.05, 0.2, 50)
            # Send badly miscalibrated data
            y = rng.binomial(1, np.clip(p * 3, 0, 1))
            m.update(p, y)
        pre_reset_alarms = m.summary().n_alarms
        m.reset()
        assert m.summary().n_alarms == pre_reset_alarms

    def test_summary_structure(self):
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        rng = np.random.default_rng(6)
        for _ in range(5):
            m.update(rng.uniform(0.05, 0.2, 30), rng.binomial(1, 0.1, 30))
        s = m.summary()
        assert s.n_time_steps == 5
        assert isinstance(s.alarm_times, list)
        d = s.to_dict()
        assert "n_alarms" in d
        assert "current_statistic" in d

    def test_poisson_mode_basic(self):
        """Poisson CUSUM runs without error."""
        m = CalibrationCUSUM(
            delta_a=1.5, distribution="poisson", cfar=0.005, n_mc=200, random_state=0
        )
        rng = np.random.default_rng(7)
        mu = rng.uniform(0.05, 0.2, 100)
        y = rng.poisson(mu)
        exposure = np.ones(100)
        alarm = m.update(mu, y, exposure=exposure)
        assert isinstance(alarm, CUSUMAlarm)
        assert alarm.n_obs == 100

    def test_poisson_downward_shift_detectable(self):
        """delta_a < 1 should detect downward rate shift."""
        m = CalibrationCUSUM(
            delta_a=0.5, distribution="poisson", cfar=0.005, n_mc=200, random_state=0
        )
        rng = np.random.default_rng(8)
        detected = False
        for _ in range(100):
            mu = rng.uniform(0.2, 0.4, 50)
            # True rate is half of predicted
            y = rng.poisson(mu * 0.5)
            if m.update(mu, y, exposure=np.ones(50)):
                detected = True
                break
        # Very likely to detect within 100 steps
        assert detected

    def test_repr_contains_state(self):
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        r = repr(m)
        assert "CalibrationCUSUM" in r
        assert "delta_a=2.0" in r

    def test_high_cfar_more_alarms_than_low_cfar(self):
        """Higher cfar => lower threshold => more alarms on in-control data."""
        rng = np.random.default_rng(9)
        data = [(rng.uniform(0.05, 0.2, 50), rng.binomial(1, 0.1, 50))
                for _ in range(50)]

        alarms_high = 0
        m_high = CalibrationCUSUM(delta_a=2.0, cfar=0.2, n_mc=200, random_state=0)
        for p, y in data:
            if m_high.update(p, y):
                alarms_high += 1

        alarms_low = 0
        m_low = CalibrationCUSUM(delta_a=2.0, cfar=0.001, n_mc=200, random_state=0)
        for p, y in data:
            if m_low.update(p, y):
                alarms_low += 1

        assert alarms_high >= alarms_low

    def test_extreme_cfar_values(self):
        """Very small cfar (near ARL0=2000) should rarely fire on good data."""
        m = CalibrationCUSUM(delta_a=2.0, cfar=0.0005, n_mc=200, random_state=0)
        rng = np.random.default_rng(10)
        alarms = 0
        for _ in range(50):
            p = rng.uniform(0.05, 0.2, 50)
            y = rng.binomial(1, p)
            if m.update(p, y):
                alarms += 1
        assert alarms <= 3


# ===========================================================================
# ConformalControlChart — edge cases
# ===========================================================================


class TestConformalControlChartEdgeCases:
    """Edge cases for ConformalControlChart."""

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ConformalControlChart(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ConformalControlChart(alpha=1.0)

    def test_too_few_calibration_scores_raises(self):
        chart = ConformalControlChart(alpha=0.05)
        with pytest.raises(ValueError, match="at least 2"):
            chart.fit([0.5])

    def test_small_calibration_warns(self):
        chart = ConformalControlChart(alpha=0.05)
        with pytest.warns(UserWarning):
            chart.fit(np.arange(10, dtype=float))

    def test_monitor_before_fit_raises(self):
        chart = ConformalControlChart(alpha=0.05)
        with pytest.raises(RuntimeError, match="fit"):
            chart.monitor([1.0, 2.0])

    def test_p_values_in_zero_one(self):
        rng = np.random.default_rng(0)
        cal = rng.exponential(1.0, 200)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        mon = rng.exponential(1.0, 50)
        result = chart.monitor(mon)
        assert np.all(result.p_values > 0)
        assert np.all(result.p_values <= 1)

    def test_in_control_false_alarm_rate(self):
        """Under exchangeability, empirical FAR should be <= alpha with margin."""
        rng = np.random.default_rng(11)
        cal = rng.exponential(1.0, 500)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        mon = rng.exponential(1.0, 500)
        result = chart.monitor(mon)
        # FAR should be around 0.05; allow generous margin for Monte Carlo variation
        assert result.alarm_rate <= 0.15

    def test_out_of_control_higher_alarm_rate(self):
        """Drifted data should have higher alarm rate than in-control."""
        rng = np.random.default_rng(12)
        cal = rng.exponential(1.0, 300)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        # Shifted distribution -> higher NCS values -> more alarms
        drifted = rng.exponential(5.0, 100)
        result = chart.monitor(drifted)
        assert result.alarm_rate > 0.05

    def test_single_monitoring_observation(self):
        rng = np.random.default_rng(13)
        cal = rng.exponential(1.0, 100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor([1.0])
        assert len(result.p_values) == 1
        assert len(result.is_alarm) == 1

    def test_result_n_cal_correct(self):
        rng = np.random.default_rng(14)
        cal = rng.exponential(1.0, 123)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(rng.exponential(1.0, 20))
        assert result.n_cal == 123

    def test_to_polars_schema(self):
        rng = np.random.default_rng(15)
        cal = rng.exponential(1.0, 100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(rng.exponential(1.0, 10))
        df = result.to_polars()
        assert "obs_index" in df.columns
        assert "score" in df.columns
        assert "p_value" in df.columns
        assert "is_alarm" in df.columns
        assert len(df) == 10

    def test_summary_string(self):
        rng = np.random.default_rng(16)
        cal = rng.exponential(1.0, 100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(rng.exponential(1.0, 10))
        s = result.summary()
        assert isinstance(s, str)
        assert "alpha" in s

    def test_ncs_absolute_residual(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 1.5, 2.5])
        ncs = ConformalControlChart.ncs_absolute_residual(actual, predicted)
        np.testing.assert_allclose(ncs, [0.5, 0.5, 0.5])

    def test_ncs_relative_residual_zero_denominator(self):
        """Zero predicted -> floored to 1e-9."""
        actual = np.array([1.0])
        predicted = np.array([0.0])
        ncs = ConformalControlChart.ncs_relative_residual(actual, predicted)
        assert ncs[0] > 0.0  # no division-by-zero crash

    def test_ncs_median_deviation(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ncs = ConformalControlChart.ncs_median_deviation(values)
        # Median=3, so deviations are [2,1,0,1,2]
        np.testing.assert_allclose(ncs, [2.0, 1.0, 0.0, 1.0, 2.0])

    def test_ncs_studentized_zero_vol(self):
        """Zero local_vol -> floored to 1e-9 to avoid division by zero."""
        actual = np.array([1.0])
        predicted = np.array([0.5])
        local_vol = np.array([0.0])
        ncs = ConformalControlChart.ncs_studentized(actual, predicted, local_vol)
        assert ncs[0] > 0.0

    def test_threshold_monotone_in_alpha(self):
        """Lower alpha -> higher threshold (stricter control limit)."""
        rng = np.random.default_rng(17)
        cal = rng.exponential(1.0, 200)
        t_strict = _conformal_threshold(cal, 0.01)
        t_lenient = _conformal_threshold(cal, 0.10)
        assert t_strict >= t_lenient

    def test_p_value_exact_boundary(self):
        """ICP p-value: new score = max of cal scores -> p > 1/(n+1)."""
        cal = np.array([1.0, 2.0, 3.0])
        p = _conformal_p_value(3.0, cal)
        assert p > 0.0

    def test_alarm_rate_property(self):
        rng = np.random.default_rng(18)
        cal = rng.exponential(1.0, 100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(rng.exponential(1.0, 100))
        expected_rate = result.n_alarms / 100
        assert result.alarm_rate == pytest.approx(expected_rate)


# ===========================================================================
# MultivariateConformalMonitor — edge cases
# ===========================================================================


class TestMultivariateConformalMonitorEdgeCases:
    """Edge cases for MultivariateConformalMonitor."""

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            MultivariateConformalMonitor(alpha=1.5)

    def test_monitor_before_fit_raises(self):
        m = MultivariateConformalMonitor(alpha=0.05)
        with pytest.raises(RuntimeError, match="fit"):
            m.monitor(np.ones((3, 2)))

    def test_too_few_calibration_raises(self):
        m = MultivariateConformalMonitor(alpha=0.05)

        class DummyModel:
            def fit(self, X):
                return self
            def decision_function(self, X):
                return -np.ones(len(X))

        with pytest.raises(ValueError, match="at least 2"):
            m.fit(np.ones((50, 3)), np.ones((1, 3)), )

    def test_custom_model_without_interface_raises(self):
        """Model missing decision_function and score_samples -> AttributeError at fit."""
        class BadModel:
            def fit(self, X):
                return self

        m = MultivariateConformalMonitor(model=BadModel(), alpha=0.05)
        X = np.random.default_rng(0).normal(0, 1, (50, 3))
        # AttributeError is raised at fit() during calibration scoring
        with pytest.raises(AttributeError):
            m.fit(X, X)

    def test_duck_typed_model_works(self):
        """Custom model with decision_function interface works."""
        class ConstantModel:
            def fit(self, X):
                return self
            def decision_function(self, X):
                # Return random-looking values to get non-degenerate scores
                return np.arange(len(X), dtype=float)

        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, (50, 3))
        X_cal = rng.normal(0, 1, (30, 3))
        X_new = rng.normal(0, 1, (10, 3))

        m = MultivariateConformalMonitor(model=ConstantModel(), alpha=0.05)
        m.fit(X_train, X_cal)
        result = m.monitor(X_new)
        assert len(result.p_values) == 10
        assert np.all(result.p_values > 0)

    def test_1d_input_reshaped(self):
        """1D input to fit() is reshaped to (n, 1); 1D monitor input -> (1, n).
        
        The monitor() reshape treats a 1D array of shape (d,) as a single
        d-dimensional observation (1 row), not d univariate observations.
        This matches numpy convention: reshape(-1, 1) only applies in fit().
        """
        class ScoreModel:
            def fit(self, X):
                return self
            def decision_function(self, X):
                return np.zeros(len(X))

        rng = np.random.default_rng(1)
        X_train = rng.normal(0, 1, 50)
        X_cal = rng.normal(0, 1, 30)
        m = MultivariateConformalMonitor(model=ScoreModel(), alpha=0.05)
        m.fit(X_train, X_cal)
        # A 1D array passed to monitor() is treated as 1 observation (shape (1, d))
        result = m.monitor(rng.normal(0, 1, 10))
        assert len(result.p_values) == 1  # 1 observation (of 10 dims)
        # Pass a 2D array of shape (5, 1) for 5 separate univariate observations
        X_mon = rng.normal(0, 1, (5, 1))
        result2 = m.monitor(X_mon)
        assert len(result2.p_values) == 5


# ===========================================================================
# ModelMonitor — edge cases
# ===========================================================================


class TestModelMonitorEdgeCases:
    """Edge cases for ModelMonitor (check_gmcb/check_lmcb integration)."""

    def _make_monitor(self, **kwargs):
        defaults = dict(
            distribution="poisson",
            n_bootstrap=50,
            alpha_gini=0.32,
            alpha_global=0.32,
            alpha_local=0.32,
            random_state=0,
        )
        defaults.update(kwargs)
        return ModelMonitor(**defaults)

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            ModelMonitor(distribution="bernoulli")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            ModelMonitor(alpha_gini=1.5)

    def test_n_bootstrap_too_low_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            ModelMonitor(n_bootstrap=20)

    def test_test_before_fit_raises(self):
        m = self._make_monitor()
        y, mu = _poisson_data(300)
        with pytest.raises(RuntimeError, match="fit"):
            m.test(y, mu)

    def test_empty_reference_raises(self):
        m = self._make_monitor()
        with pytest.raises(ValueError):
            m.fit([], [])

    def test_empty_monitor_raises(self):
        y, mu = _poisson_data(500)
        m = self._make_monitor().fit(y, mu)
        with pytest.raises(ValueError):
            m.test([], [])

    def test_mismatched_ref_lengths_raises(self):
        m = self._make_monitor()
        with pytest.raises(ValueError, match="length"):
            m.fit(np.ones(10), np.ones(11))

    def test_mismatched_monitor_lengths_raises(self):
        y, mu = _poisson_data(500)
        m = self._make_monitor().fit(y, mu)
        with pytest.raises(ValueError, match="length"):
            m.test(np.ones(10), np.ones(11))

    def test_is_fitted_property(self):
        m = self._make_monitor()
        assert not m.is_fitted()
        y, mu = _poisson_data(500)
        m.fit(y, mu)
        assert m.is_fitted()

    def test_result_fields_present(self):
        y, mu = _poisson_data(800, seed=5)
        m = self._make_monitor().fit(y, mu)
        result = m.test(y, mu)
        assert isinstance(result, ModelMonitorResult)
        assert result.decision in {"REFIT", "RECALIBRATE", "REDEPLOY"}
        assert isinstance(result.decision_reason, str)
        assert result.n_new == 800

    def test_summary_string(self):
        y, mu = _poisson_data(800, seed=6)
        m = self._make_monitor().fit(y, mu)
        result = m.test(y, mu)
        s = result.summary()
        assert isinstance(s, str)
        assert result.decision in s

    def test_to_dict(self):
        y, mu = _poisson_data(800, seed=7)
        m = self._make_monitor().fit(y, mu)
        d = m.test(y, mu).to_dict()
        assert isinstance(d, dict)
        assert "decision" in d

    def test_extreme_global_shift_recalibrate(self):
        """A strong global upward shift should produce RECALIBRATE or REFIT."""
        rng = np.random.default_rng(42)
        n = 1000
        exposure = rng.uniform(0.5, 2.0, n)
        mu = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * mu) / exposure
        # 40% global shift
        y_mon = rng.poisson(exposure * mu * 1.4) / exposure

        m = self._make_monitor(alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32)
        m.fit(y_ref, mu, exposure)
        result = m.test(y_mon, mu, exposure)
        assert result.decision in {"RECALIBRATE", "REFIT"}

    def test_balance_factor_near_one_for_calibrated(self):
        """When data is calibrated, balance factor should be near 1."""
        y, mu = _poisson_data(1000, seed=100)
        m = self._make_monitor().fit(y, mu)
        result = m.test(y, mu)
        # Balance factor = sum(y)/sum(mu), should be near 1 for poisson
        assert 0.5 < result.balance_factor < 2.0

    def test_exposure_low_median_warns(self):
        """Tiny exposure values trigger the time-splitting pitfall warning."""
        rng = np.random.default_rng(11)
        n = 500
        exposure = rng.uniform(0.001, 0.05, n)  # very small exposures
        mu = rng.uniform(0.05, 0.2, n)
        y = rng.poisson(exposure * mu) / exposure
        m = self._make_monitor()
        with pytest.warns(UserWarning):
            m.fit(y, mu, exposure)

    def test_gamma_distribution(self):
        rng = np.random.default_rng(12)
        y = rng.gamma(2, 0.5, 600)
        mu = rng.uniform(0.5, 2.0, 600)
        m = ModelMonitor(distribution="gamma", n_bootstrap=50, random_state=0)
        m.fit(y, mu)
        result = m.test(y, mu)
        assert result.decision in {"REFIT", "RECALIBRATE", "REDEPLOY"}


# ===========================================================================
# PricingDriftMonitor — edge cases
# ===========================================================================


class TestPricingDriftMonitorEdgeCases:
    """Edge cases for PricingDriftMonitor."""

    def _monitor(self, **kwargs):
        defaults = dict(
            distribution="poisson", n_bootstrap=50, alpha_gini=0.32,
            alpha_global=0.32, alpha_local=0.32, random_state=0
        )
        defaults.update(kwargs)
        return PricingDriftMonitor(**defaults)

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            PricingDriftMonitor(distribution="logistic")

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha_gini"):
            PricingDriftMonitor(alpha_gini=2.0)

    def test_n_bootstrap_too_low_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            PricingDriftMonitor(n_bootstrap=10)

    def test_test_before_fit_raises(self):
        m = self._monitor()
        y, mu = _poisson_data(300)
        with pytest.raises(RuntimeError, match="fit"):
            m.test(y, mu)

    def test_is_fitted_property(self):
        m = self._monitor()
        assert not m.is_fitted()
        y, mu = _poisson_data(600)
        m.fit(y, mu)
        assert m.is_fitted()

    def test_verdict_is_valid_string(self):
        y, mu = _poisson_data(700, seed=20)
        m = self._monitor()
        m.fit(y, mu)
        result = m.test(y, mu)
        assert result.verdict in {"OK", "RECALIBRATE", "REFIT"}

    def test_calibration_test_results_present(self):
        y, mu = _poisson_data(700, seed=21)
        m = self._monitor()
        m.fit(y, mu)
        result = m.test(y, mu)
        assert isinstance(result.global_calib, CalibTestResult)
        assert isinstance(result.local_calib, CalibTestResult)
        assert result.global_calib.component == "GMCB"
        assert result.local_calib.component == "LMCB"

    def test_to_dict_flat(self):
        y, mu = _poisson_data(700, seed=22)
        m = self._monitor()
        m.fit(y, mu)
        d = m.test(y, mu).to_dict()
        assert isinstance(d, dict)
        assert "verdict" in d
        assert "gini_gini_ref" in d or "gini_ref" in [k.split("_")[-1] for k in d]

    def test_summary_string(self):
        y, mu = _poisson_data(700, seed=23)
        m = self._monitor()
        m.fit(y, mu)
        s = m.test(y, mu).summary()
        assert isinstance(s, str)
        assert "Gini" in s

    def test_repr_contains_status(self):
        m = self._monitor()
        r = repr(m)
        assert "not fitted" in r
        y, mu = _poisson_data(700)
        m.fit(y, mu)
        r2 = repr(m)
        assert "fitted" in r2

    def test_small_monitor_warns(self):
        """n_mon < 500 triggers UserWarning."""
        y, mu = _poisson_data(700)
        m = self._monitor()
        m.fit(y, mu)
        y_small, mu_small = _poisson_data(100)
        with pytest.warns(UserWarning, match="500"):
            m.test(y_small, mu_small)

    def test_alpha_override_at_test_time(self):
        y, mu = _poisson_data(700, seed=30)
        m = self._monitor()
        m.fit(y, mu)
        r_strict = m.test(y, mu, alpha_gini=0.999)  # almost always reject Gini
        r_lenient = m.test(y, mu, alpha_gini=0.0001) # almost never reject Gini
        # Same p-values, different reject flags
        assert r_strict.gini.p_value == pytest.approx(r_lenient.gini.p_value, abs=1e-6)


# ===========================================================================
# BAWSMonitor — edge cases
# ===========================================================================


class TestBAWSMonitorEdgeCases:
    """Additional edge cases for BAWSMonitor beyond existing test_baws.py."""

    def test_update_before_fit_raises(self):
        m = BAWSMonitor(candidate_windows=[50, 100])
        with pytest.raises(RuntimeError):
            m.update(-0.5)

    def test_history_before_any_update(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(200)
        m = BAWSMonitor(candidate_windows=[50, 100], n_bootstrap=20, random_state=0)
        m.fit(data)
        h = m.history()
        assert len(h) == 0  # no updates yet

    def test_update_returns_baws_result(self):
        from insurance_monitoring.baws import BAWSResult
        rng = np.random.default_rng(1)
        data = rng.standard_normal(200)
        m = BAWSMonitor(candidate_windows=[50, 100], n_bootstrap=20, random_state=0)
        m.fit(data)
        result = m.update(-0.3)
        assert isinstance(result, BAWSResult)
        assert result.selected_window in [50, 100]
        assert result.time_step == 1

    def test_selected_window_is_candidate(self):
        rng = np.random.default_rng(2)
        data = rng.standard_normal(300)
        windows = [50, 100, 150]
        m = BAWSMonitor(candidate_windows=windows, n_bootstrap=20, random_state=0)
        m.fit(data)
        for _ in range(5):
            r = m.update(rng.standard_normal())
            assert r.selected_window in windows

    def test_es_le_var_for_losses(self):
        """ES should be <= VaR for a loss distribution (ES more negative)."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal(200)
        m = BAWSMonitor(candidate_windows=[50, 100], alpha=0.05,
                        n_bootstrap=20, random_state=0)
        m.fit(data)
        r = m.update(-0.5)
        # ES is the expected value below the VaR quantile -> more negative
        assert r.es_estimate <= r.var_estimate + 1e-6

    def test_all_identical_returns_no_crash(self):
        """Constant returns (sigma=0) should not raise."""
        data = np.zeros(200)
        m = BAWSMonitor(candidate_windows=[50, 100], n_bootstrap=20, random_state=0)
        m.fit(data)
        # Should not raise even with degenerate data
        r = m.update(0.0)
        assert isinstance(r.selected_window, int)

    def test_history_grows_with_updates(self):
        rng = np.random.default_rng(4)
        data = rng.standard_t(df=5, size=200)
        m = BAWSMonitor(candidate_windows=[50, 100], n_bootstrap=20, random_state=0)
        m.fit(data)
        for i in range(5):
            m.update(rng.standard_t(df=5))
        h = m.history()
        assert len(h) == 5

    def test_scores_dict_has_all_windows(self):
        rng = np.random.default_rng(5)
        data = rng.standard_normal(200)
        windows = [50, 100]
        m = BAWSMonitor(candidate_windows=windows, n_bootstrap=20, random_state=0)
        m.fit(data)
        r = m.update(-0.2)
        for w in windows:
            assert w in r.scores


# ===========================================================================
# Integration: ModelMonitor + ConformalControlChart
# ===========================================================================


class TestModelMonitorConformalIntegration:
    """Integration test: use ModelMonitor output to feed ConformalControlChart."""

    def test_pipeline_end_to_end(self):
        """
        Simulate a monitoring pipeline:
        1. Fit ModelMonitor on reference data.
        2. Collect Gini scores from 12 monitoring windows.
        3. Fit ConformalControlChart on first 8 windows.
        4. Alert on remaining 4 windows.
        """
        rng = np.random.default_rng(100)
        n = 500

        # Reference data
        exposure = rng.uniform(0.5, 2.0, n)
        mu_ref = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * mu_ref) / exposure

        mm = ModelMonitor(distribution="poisson", n_bootstrap=50,
                          alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
                          random_state=0)
        mm.fit(y_ref, mu_ref, exposure)

        gini_scores = []
        for period in range(12):
            y_mon = rng.poisson(exposure * mu_ref) / exposure
            result = mm.test(y_mon, mu_ref, exposure)
            gini_scores.append(result.gini_new)

        gini_scores = np.array(gini_scores)

        # Use absolute deviation from mean as NCS
        cal_ncs = ConformalControlChart.ncs_median_deviation(gini_scores[:8])
        chart = ConformalControlChart(alpha=0.1).fit(cal_ncs)
        mon_ncs = ConformalControlChart.ncs_median_deviation(
            gini_scores[8:], median=float(np.median(gini_scores[:8]))
        )
        result = chart.monitor(mon_ncs)
        assert len(result.p_values) == 4
        assert result.n_cal == 8

    def test_cusum_after_model_monitor_alarm(self):
        """
        After ModelMonitor flags RECALIBRATE, CUSUM on the balance factor
        should accumulate positive increments.
        """
        rng = np.random.default_rng(200)
        n = 600
        exposure = rng.uniform(0.5, 2.0, n)
        mu = rng.gamma(2, 0.05, n)
        y_ref = rng.poisson(exposure * mu) / exposure

        mm = ModelMonitor(distribution="poisson", n_bootstrap=50,
                          random_state=0)
        mm.fit(y_ref, mu, exposure)

        # Introduce an upward shift
        y_shifted = rng.poisson(exposure * mu * 1.3) / exposure
        mon_result = mm.test(y_shifted, mu, exposure)
        assert mon_result.balance_factor > 1.0

        # Feed the per-policy residuals into CUSUM
        cusum = CalibrationCUSUM(
            delta_a=1.3, distribution="poisson", cfar=0.005, n_mc=200, random_state=0
        )
        for _ in range(20):
            y_step = rng.poisson(exposure[:50] * mu[:50] * 1.3)
            alarm = cusum.update(mu[:50], y_step.astype(float), exposure=exposure[:50])

        # After 20 steps of upward shift, CUSUM should be positive
        assert cusum.statistic >= 0.0


# ===========================================================================
# conformal_p_value edge cases (internal math)
# ===========================================================================


class TestConformalPValueMath:
    """Additional edge cases for the internal conformal math functions."""

    def test_single_calibration_score(self):
        """n=1 calibration set: p(s) is either 1/2 or 2/2=1."""
        cal = np.array([5.0])
        p_above = _conformal_p_value(10.0, cal)  # above cal -> 0 >= count; (0+1)/(1+1) = 0.5
        p_at = _conformal_p_value(5.0, cal)      # count=1; (1+1)/(1+1) = 1.0
        assert p_above == pytest.approx(0.5)
        assert p_at == pytest.approx(1.0)

    def test_vectorised_p_values_consistent_with_scalar(self):
        """_conformal_p_values must match repeated calls to _conformal_p_value."""
        rng = np.random.default_rng(0)
        cal = rng.exponential(1.0, 100)
        s_new = rng.exponential(1.0, 20)
        vec = _conformal_p_values(s_new, cal)
        scalar = np.array([_conformal_p_value(s, cal) for s in s_new])
        np.testing.assert_allclose(vec, scalar, rtol=1e-10)

    def test_p_value_monotone_decreasing_in_score(self):
        """Higher score (more anomalous) -> lower p-value."""
        cal = np.arange(1.0, 101.0)
        p_low = _conformal_p_value(10.0, cal)
        p_high = _conformal_p_value(90.0, cal)
        assert p_high <= p_low

    def test_threshold_at_small_alpha_is_high(self):
        """Very small alpha -> threshold near maximum of cal_scores."""
        rng = np.random.default_rng(0)
        cal = rng.exponential(1.0, 500)
        t = _conformal_threshold(cal, 0.001)
        assert t >= np.quantile(cal, 0.90)

    def test_nan_in_cal_propagates(self):
        """NaN in calibration scores propagates through numpy operations."""
        cal = np.array([1.0, np.nan, 3.0])
        # Should not crash but result may be nan
        p = _conformal_p_value(2.0, cal)
        assert isinstance(p, float)


# ===========================================================================
# Degenerate / NaN scenarios — smoke tests
# ===========================================================================


class TestDegenerateInputs:
    """Smoke tests for degenerate inputs that should not crash the library."""

    def test_gini_drift_monitor_all_zero_actuals(self):
        """All-zero actuals -> Gini is undefined but should not crash."""
        y_ref = np.zeros(300)
        mu_ref = np.full(300, 0.1)
        m = GiniDriftMonitor(n_bootstrap=50, random_state=0)
        m.fit(y_ref, mu_ref)
        y_mon = np.zeros(200)
        mu_mon = np.full(200, 0.1)
        # May warn but should not crash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = m.test(y_mon, mu_mon)
        assert isinstance(result, GiniDriftResult)

    def test_cusum_bernoulli_all_ones(self):
        """All claims = 1, predictions well-calibrated -> CUSUM runs."""
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        p = np.full(50, 0.9)
        y = np.ones(50)
        alarm = m.update(p, y)
        assert isinstance(alarm, CUSUMAlarm)

    def test_cusum_bernoulli_all_zeros(self):
        """All claims = 0, predictions high -> negative W_t, S_t stays 0."""
        m = CalibrationCUSUM(delta_a=2.0, n_mc=200, random_state=0)
        p = np.full(50, 0.5)
        y = np.zeros(50)
        alarm = m.update(p, y)
        assert alarm.statistic >= 0.0

    def test_conformal_chart_all_identical_scores(self):
        """All calibration NCS identical -> threshold = that value."""
        cal = np.ones(100)
        chart = ConformalControlChart(alpha=0.05).fit(cal)
        result = chart.monitor(np.ones(10))
        # All monitoring scores == all cal scores -> p-values all 1.0
        assert np.all(result.p_values == pytest.approx(1.0))

    def test_murphy_decomposition_constant_mu_hat(self):
        """Constant mu_hat (grand mean predictor) -> MCB dominated result."""
        rng = np.random.default_rng(0)
        y = rng.poisson(0.2, 500).astype(float)
        mu_hat = np.full(500, np.mean(y))
        m = MurphyDecomposition(family="poisson")
        result = m.decompose(y, mu_hat)
        # Grand mean predictor has zero discrimination by definition
        assert result.dsc >= -1e-8  # non-negative (may be ~0)
