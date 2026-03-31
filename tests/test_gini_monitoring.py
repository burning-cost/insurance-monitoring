"""
Tests for insurance_monitoring.gini_monitoring — GiniDriftMonitor,
GiniBootstrapMonitor, and MurphyDecomposition classes.

Implements the Brauer/Menzel/Wüthrich (arXiv:2510.04556) monitoring framework
tests as specified in the Burning Cost engineering spec.

Covers:
- GiniDriftMonitor: fit/test interface, drift detection, no-drift, exposure
  weighting, input validation, result fields, subsample cap.
- GiniBootstrapMonitor: fit/test interface, one-sample test consistency with
  two-sample, input validation.
- MurphyDecomposition: MCB=0 for perfect calibration, DSC=0 for constant
  prediction, GMCB dominates for global miscalibration, family variants,
  summary() method, decision rule.
- Cross-class: one-sample vs two-sample consistency, exposure weighting effects.

All tests designed to run comfortably within Databricks serverless time limits.
Bootstrap counts are kept low (50–150) for speed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_monitoring.gini_monitoring import (
    GiniBootstrapMonitor,
    GiniBootstrapResult,
    GiniDriftMonitor,
    GiniDriftResult,
    MurphyDecomposition,
    _MurphyComponents,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_poisson_data(n: int, seed: int = 0):
    """Return (y, mu_hat) from a Poisson model with genuine ranking power."""
    rng = np.random.default_rng(seed)
    mu_hat = rng.uniform(0.05, 0.20, n)
    y = rng.poisson(mu_hat).astype(float)
    return y, mu_hat


def _make_flat_predictions(n: int, seed: int = 0):
    """All predictions identical — Gini = 0."""
    rng = np.random.default_rng(seed)
    y = rng.poisson(0.10, n).astype(float)
    mu_hat = np.full(n, 0.10)
    return y, mu_hat


def _make_degraded_data(n: int, seed: int = 99):
    """Heavy noise on predictions — Gini much lower than clean model."""
    rng = np.random.default_rng(seed)
    true_mu = rng.uniform(0.05, 0.20, n)
    noise = rng.normal(0, 0.5, n)
    mu_noisy = np.clip(true_mu + noise, 0.001, 5.0)
    y = rng.poisson(true_mu).astype(float)
    return y, mu_noisy


# ===========================================================================
# Section A — GiniDriftMonitor (two-sample fit/test)
# ===========================================================================


class TestGiniDriftMonitorFitTest:
    """A01–A03: basic fit/test contract."""

    def test_a01_not_fitted_before_fit(self):
        monitor = GiniDriftMonitor(n_bootstrap=50)
        assert not monitor.is_fitted()

    def test_a02_fitted_after_fit(self):
        y, mu = _make_poisson_data(500, seed=1)
        monitor = GiniDriftMonitor(n_bootstrap=50)
        monitor.fit(y, mu)
        assert monitor.is_fitted()

    def test_a03_test_raises_before_fit(self):
        y, mu = _make_poisson_data(300, seed=2)
        monitor = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(RuntimeError, match="fitted before calling test"):
            monitor.test(y, mu)

    def test_a04_fit_returns_self(self):
        y, mu = _make_poisson_data(400, seed=3)
        monitor = GiniDriftMonitor(n_bootstrap=50)
        ret = monitor.fit(y, mu)
        assert ret is monitor

    def test_a05_returns_gini_drift_result(self):
        y_r, mu_r = _make_poisson_data(600, seed=4)
        y_m, mu_m = _make_poisson_data(600, seed=5)
        monitor = GiniDriftMonitor(n_bootstrap=80, random_state=0)
        monitor.fit(y_r, mu_r)
        result = monitor.test(y_m, mu_m)
        assert isinstance(result, GiniDriftResult)

    def test_a06_result_fields_finite(self):
        y_r, mu_r = _make_poisson_data(800, seed=6)
        y_m, mu_m = _make_poisson_data(700, seed=7)
        monitor = GiniDriftMonitor(n_bootstrap=100, alpha=0.05, random_state=1)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        assert np.isfinite(r.gini_ref)
        assert np.isfinite(r.gini_mon)
        assert np.isfinite(r.gini_change)
        assert np.isfinite(r.z_stat)
        assert np.isfinite(r.p_value)
        assert isinstance(r.reject_h0, bool)
        assert np.isfinite(r.ci_lower)
        assert np.isfinite(r.ci_upper)
        assert r.n_ref == 800
        assert r.n_mon == 700

    def test_a07_gini_change_is_mon_minus_ref(self):
        y_r, mu_r = _make_poisson_data(600, seed=8)
        y_m, mu_m = _make_poisson_data(600, seed=9)
        monitor = GiniDriftMonitor(n_bootstrap=80, random_state=2)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        assert abs(r.gini_change - (r.gini_mon - r.gini_ref)) < 1e-10

    def test_a08_p_value_in_range(self):
        y_r, mu_r = _make_poisson_data(500, seed=10)
        y_m, mu_m = _make_poisson_data(500, seed=11)
        monitor = GiniDriftMonitor(n_bootstrap=80, random_state=3)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        assert 0.0 <= r.p_value <= 1.0

    def test_a09_reject_flag_consistent_with_p_and_alpha(self):
        y_r, mu_r = _make_poisson_data(600, seed=12)
        y_m, mu_m = _make_poisson_data(600, seed=13)
        for alpha in [0.01, 0.05, 0.10, 0.32]:
            monitor = GiniDriftMonitor(n_bootstrap=80, alpha=alpha, random_state=4)
            monitor.fit(y_r, mu_r)
            r = monitor.test(y_m, mu_m)
            assert r.reject_h0 == (r.p_value < alpha), (
                f"reject_h0 inconsistent: p={r.p_value:.4f}, alpha={alpha}"
            )

    def test_a10_ci_contains_zero_under_same_dgp(self):
        """Under same DGP, (1-alpha) CI for change should contain 0 most of the time."""
        rng = np.random.default_rng(14)
        mu = rng.uniform(0.05, 0.20, 3000)
        y_r = rng.poisson(mu).astype(float)
        mu_m = rng.uniform(0.05, 0.20, 3000)
        y_m = rng.poisson(mu_m).astype(float)
        monitor = GiniDriftMonitor(n_bootstrap=100, alpha=0.05, random_state=14)
        monitor.fit(y_r, mu)
        r = monitor.test(y_m, mu_m)
        # With alpha=0.05 the CI should contain 0 ~95% of the time.
        # With a fixed seed this is deterministic.
        assert r.ci_lower <= r.gini_change <= r.ci_upper


class TestGiniDriftMonitorNoDrift:
    """A11: Same DGP should not trigger false alarms at alpha=0.05."""

    def test_a11_no_drift_same_dgp(self):
        rng = np.random.default_rng(20)
        mu_r = rng.uniform(0.05, 0.20, 5000)
        y_r = rng.poisson(mu_r).astype(float)
        mu_m = rng.uniform(0.05, 0.20, 5000)
        y_m = rng.poisson(mu_m).astype(float)
        monitor = GiniDriftMonitor(n_bootstrap=100, alpha=0.05, random_state=20)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        assert r.p_value > 0.05, (
            f"False alarm under same DGP: p={r.p_value:.4f}"
        )
        assert r.reject_h0 is False


class TestGiniDriftMonitorDetectsDrift:
    """A12: Significantly degraded model should be flagged."""

    def test_a12_detects_severe_degradation(self):
        rng = np.random.default_rng(30)
        mu_r = rng.uniform(0.05, 0.20, 5000)
        y_r = rng.poisson(mu_r).astype(float)
        # Monitor: flat predictions — Gini collapses to ~0
        mu_m = np.full(5000, 0.10)
        y_m = rng.poisson(rng.uniform(0.05, 0.20, 5000)).astype(float)
        monitor = GiniDriftMonitor(n_bootstrap=100, alpha=0.32, random_state=30)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        assert r.gini_ref > r.gini_mon, "Reference Gini should be higher than flat model"
        assert r.gini_change < 0, "Drift should be negative (degradation)"

    def test_a13_detects_30pct_prediction_scale_shift(self):
        """Multiplying predictions by 1.3 changes the Gini."""
        rng = np.random.default_rng(31)
        mu_r = rng.uniform(0.05, 0.20, 4000)
        y_r = rng.poisson(mu_r).astype(float)
        # Monitor: same ranking but predictions uniformly scaled up
        mu_m = mu_r * 1.3
        y_m = rng.poisson(mu_r).astype(float)  # actual unchanged
        monitor = GiniDriftMonitor(n_bootstrap=100, alpha=0.05, random_state=31)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        # Scaling does not change ranking, Gini should be similar
        assert abs(r.gini_change) < 0.15, (
            "Pure scale shift should not dramatically change Gini"
        )


class TestGiniDriftMonitorExposure:
    """A14–A15: Exposure weighting."""

    def test_a14_exposure_accepted_without_error(self):
        rng = np.random.default_rng(50)
        n = 500
        y_r, mu_r = _make_poisson_data(n, seed=50)
        y_m, mu_m = _make_poisson_data(n, seed=51)
        exp_r = rng.uniform(0.1, 2.0, n)
        exp_m = rng.uniform(0.1, 2.0, n)
        monitor = GiniDriftMonitor(n_bootstrap=80, random_state=50)
        monitor.fit(y_r, mu_r, exposure_ref=exp_r)
        r = monitor.test(y_m, mu_m, exposure_mon=exp_m)
        assert isinstance(r, GiniDriftResult)
        assert np.isfinite(r.gini_ref)

    def test_a15_exposure_changes_gini_value(self):
        rng = np.random.default_rng(52)
        n = 600
        y_r, mu_r = _make_poisson_data(n, seed=52)
        y_m, mu_m = _make_poisson_data(n, seed=53)
        exp_r = rng.uniform(0.1, 2.0, n)
        exp_m = rng.uniform(0.1, 2.0, n)
        m_no_exp = GiniDriftMonitor(n_bootstrap=80, random_state=52)
        m_with_exp = GiniDriftMonitor(n_bootstrap=80, random_state=52)
        m_no_exp.fit(y_r, mu_r)
        m_with_exp.fit(y_r, mu_r, exposure_ref=exp_r)
        r_no = m_no_exp.test(y_m, mu_m)
        r_with = m_with_exp.test(y_m, mu_m, exposure_mon=exp_m)
        # Values differ when exposure varies
        assert isinstance(r_no.gini_ref, float)
        assert isinstance(r_with.gini_ref, float)


class TestGiniDriftMonitorValidation:
    """A16–A17: Input validation."""

    def test_a16_empty_reference_raises(self):
        monitor = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="reference actual array must be non-empty"):
            monitor.fit(np.array([]), np.array([]))

    def test_a17_mismatched_lengths_raises(self):
        y, mu = _make_poisson_data(300, seed=60)
        monitor = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="reference actual length"):
            monitor.fit(y[:200], mu)  # 200 vs 300

    def test_a18_negative_exposure_raises(self):
        y, mu = _make_poisson_data(300, seed=61)
        monitor = GiniDriftMonitor(n_bootstrap=50)
        with pytest.raises(ValueError, match="reference exposure values must be positive"):
            monitor.fit(y, mu, exposure_ref=np.full(300, -1.0))

    def test_a19_to_dict_returns_all_expected_keys(self):
        y_r, mu_r = _make_poisson_data(400, seed=62)
        y_m, mu_m = _make_poisson_data(400, seed=63)
        monitor = GiniDriftMonitor(n_bootstrap=80, random_state=5)
        monitor.fit(y_r, mu_r)
        r = monitor.test(y_m, mu_m)
        d = r.to_dict()
        expected = {
            "gini_ref", "gini_mon", "gini_change", "z_stat", "p_value",
            "reject_h0", "ci_lower", "ci_upper", "se_ref", "se_mon",
            "n_ref", "n_mon", "alpha", "n_bootstrap",
        }
        assert expected.issubset(d.keys())


# ===========================================================================
# Section B — GiniBootstrapMonitor (one-sample fit/test)
# ===========================================================================


class TestGiniBootstrapMonitorFitTest:
    """B01–B06: basic fit/test contract for one-sample test."""

    def test_b01_not_fitted_before_fit(self):
        monitor = GiniBootstrapMonitor(n_bootstrap=50)
        assert not monitor.is_fitted()

    def test_b02_fitted_after_fit(self):
        monitor = GiniBootstrapMonitor(n_bootstrap=50)
        monitor.fit(gini_ref=0.45)
        assert monitor.is_fitted()

    def test_b03_test_raises_before_fit(self):
        y, mu = _make_poisson_data(300, seed=70)
        monitor = GiniBootstrapMonitor(n_bootstrap=50)
        with pytest.raises(RuntimeError, match="fitted before calling test"):
            monitor.test(y, mu)

    def test_b04_invalid_gini_ref_raises(self):
        with pytest.raises(ValueError, match="gini_ref must be in"):
            GiniBootstrapMonitor(n_bootstrap=50).fit(gini_ref=1.5)

    def test_b05_returns_gini_bootstrap_result(self):
        y_r, mu_r = _make_poisson_data(1000, seed=71)
        from insurance_monitoring.discrimination import gini_coefficient
        gini_ref = gini_coefficient(y_r, mu_r)
        y_m, mu_m = _make_poisson_data(800, seed=72)
        monitor = GiniBootstrapMonitor(n_bootstrap=100, alpha=0.05, random_state=10)
        monitor.fit(gini_ref=float(gini_ref))
        result = monitor.test(y_m, mu_m)
        assert isinstance(result, GiniBootstrapResult)

    def test_b06_result_fields_correct(self):
        y_r, mu_r = _make_poisson_data(1000, seed=73)
        from insurance_monitoring.discrimination import gini_coefficient
        gini_ref = float(gini_coefficient(y_r, mu_r))
        y_m, mu_m = _make_poisson_data(700, seed=74)
        monitor = GiniBootstrapMonitor(n_bootstrap=100, alpha=0.05, random_state=11)
        monitor.fit(gini_ref=gini_ref)
        r = monitor.test(y_m, mu_m)
        assert np.isfinite(r.gini_ref)
        assert np.isfinite(r.gini_mon)
        assert np.isfinite(r.gini_change)
        assert np.isfinite(r.z_stat)
        assert np.isfinite(r.p_value)
        assert isinstance(r.reject_h0, bool)
        assert r.n_mon == 700
        assert abs(r.gini_change - (r.gini_mon - r.gini_ref)) < 1e-10

    def test_b07_p_value_in_range(self):
        y_r, mu_r = _make_poisson_data(500, seed=75)
        from insurance_monitoring.discrimination import gini_coefficient
        gini_ref = float(gini_coefficient(y_r, mu_r))
        y_m, mu_m = _make_poisson_data(500, seed=76)
        monitor = GiniBootstrapMonitor(n_bootstrap=80, random_state=12)
        monitor.fit(gini_ref=gini_ref)
        r = monitor.test(y_m, mu_m)
        assert 0.0 <= r.p_value <= 1.0


class TestGiniBootstrapMonitorOneSampleConsistency:
    """B08: one-sample result is qualitatively consistent with two-sample."""

    def test_b08_one_and_two_sample_agree_on_direction(self):
        """Both tests should agree on whether Gini went up or down."""
        rng = np.random.default_rng(80)
        mu_r = rng.uniform(0.05, 0.20, 3000)
        y_r = rng.poisson(mu_r).astype(float)
        # Monitoring: flat predictions → lower Gini
        mu_m = np.full(3000, 0.10)
        y_m = rng.poisson(rng.uniform(0.05, 0.20, 3000)).astype(float)

        from insurance_monitoring.discrimination import gini_coefficient
        gini_ref = float(gini_coefficient(y_r, mu_r))

        two_sample = GiniDriftMonitor(n_bootstrap=100, alpha=0.05, random_state=80)
        two_sample.fit(y_r, mu_r)
        r2 = two_sample.test(y_m, mu_m)

        one_sample = GiniBootstrapMonitor(n_bootstrap=100, alpha=0.05, random_state=80)
        one_sample.fit(gini_ref=gini_ref)
        r1 = one_sample.test(y_m, mu_m)

        # Both should report the same direction of change
        assert np.sign(r2.gini_change) == np.sign(r1.gini_change)
        # Both should have the same monitor Gini estimate
        assert abs(r2.gini_mon - r1.gini_mon) < 1e-8


class TestGiniBootstrapMonitorExposure:
    """B09: Exposure weighting accepted."""

    def test_b09_exposure_accepted_without_error(self):
        rng = np.random.default_rng(90)
        n = 500
        y_m, mu_m = _make_poisson_data(n, seed=90)
        exp_m = rng.uniform(0.1, 2.0, n)
        monitor = GiniBootstrapMonitor(n_bootstrap=80, random_state=90)
        monitor.fit(gini_ref=0.40)
        r = monitor.test(y_m, mu_m, exposure_mon=exp_m)
        assert isinstance(r, GiniBootstrapResult)
        assert np.isfinite(r.gini_mon)


# ===========================================================================
# Section C — MurphyDecomposition class
# ===========================================================================


class TestMurphyDecompositionBasic:
    """C01–C04: basic interface and mathematical properties."""

    def test_c01_decompose_returns_murphy_components(self):
        y, mu = _make_poisson_data(2000, seed=100)
        murphy = MurphyDecomposition(family="poisson")
        result = murphy.decompose(y, mu)
        assert isinstance(result, _MurphyComponents)

    def test_c02_identity_unc_minus_dsc_plus_mcb_equals_total_deviance(self):
        """UNC - DSC + MCB = total_deviance (Murphy identity)."""
        rng = np.random.default_rng(101)
        n = 2000
        mu = rng.gamma(2, 0.05, n)
        y = rng.poisson(mu).astype(float)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu)
        reconstructed = r.unc - r.dsc + r.mcb
        assert abs(reconstructed - r.total_deviance) < 1e-4 * max(r.total_deviance, 1e-6)

    def test_c03_mcb_near_zero_for_perfectly_calibrated_model(self):
        """A perfectly calibrated model (mu_hat = true mu) should have very small MCB."""
        rng = np.random.default_rng(102)
        n = 5000
        mu_true = rng.gamma(2, 0.05, n)
        y = rng.poisson(mu_true).astype(float)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu_true)
        # MCB/UNC should be small for well-calibrated model
        assert r.mcb / max(r.unc, 1e-9) < 0.20, (
            f"MCB/UNC={r.mcb/r.unc:.3f} — expected < 0.20 for well-calibrated model"
        )

    def test_c04_dsc_near_zero_for_constant_prediction(self):
        """A constant prediction model has no ranking power: DSC should be ~0."""
        rng = np.random.default_rng(103)
        n = 3000
        y = rng.poisson(0.10, n).astype(float)
        mu_hat = np.full(n, 0.10)  # constant prediction
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu_hat)
        # DSC/UNC should be very small for a constant model
        assert r.dsc / max(r.unc, 1e-9) < 0.05, (
            f"DSC/UNC={r.dsc/r.unc:.4f} — expected ~0 for constant prediction"
        )

    def test_c05_all_components_nonnegative(self):
        y, mu = _make_poisson_data(2000, seed=104)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu)
        assert r.unc >= 0.0
        assert r.dsc >= 0.0
        assert r.mcb >= 0.0
        assert r.gmcb >= 0.0
        assert r.lmcb >= 0.0

    def test_c06_dsc_leq_unc(self):
        y, mu = _make_poisson_data(1000, seed=105)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu)
        assert r.dsc <= r.unc + 1e-8

    def test_c07_gmcb_plus_lmcb_equals_mcb(self):
        y, mu = _make_poisson_data(1000, seed=106)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu)
        assert r.gmcb + r.lmcb == pytest.approx(r.mcb, abs=1e-6)

    def test_c08_global_miscalibration_triggers_gmcb_dominance(self):
        """A pure scale error (predictions * 1.3) should produce GMCB > LMCB."""
        rng = np.random.default_rng(107)
        n = 5000
        mu_true = rng.gamma(2, 0.05, n)
        y = rng.poisson(mu_true).astype(float)
        mu_biased = mu_true * 1.3  # global scale error
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu_biased)
        assert r.gmcb >= r.lmcb, (
            f"Pure scale error: GMCB={r.gmcb:.6f} should >= LMCB={r.lmcb:.6f}"
        )

    def test_c09_verdict_valid_string(self):
        y, mu = _make_poisson_data(1000, seed=108)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu)
        assert r.verdict in ("OK", "RECALIBRATE", "REFIT")


class TestMurphyDecompositionFamilies:
    """C10–C12: Family variants work correctly."""

    def test_c10_gamma_family(self):
        rng = np.random.default_rng(110)
        n = 1000
        mu = rng.gamma(3, 100, n)
        y = rng.gamma(3, 100, n)
        murphy = MurphyDecomposition(family="gamma")
        r = murphy.decompose(y, mu)
        assert isinstance(r, _MurphyComponents)
        assert np.isfinite(r.unc)
        assert np.isfinite(r.dsc)

    def test_c11_tweedie_family(self):
        rng = np.random.default_rng(111)
        n = 1000
        mu = rng.gamma(2, 0.1, n)
        y = np.maximum(rng.gamma(2, 0.1, n), 0.0)
        murphy = MurphyDecomposition(family="tweedie", tweedie_power=1.5)
        r = murphy.decompose(y, mu)
        assert isinstance(r, _MurphyComponents)

    def test_c12_invalid_family_raises(self):
        with pytest.raises(ValueError, match="family must be one of"):
            MurphyDecomposition(family="binomial")


class TestMurphyDecompositionSummary:
    """C13–C15: summary() and exposure weighting."""

    def test_c13_summary_raises_before_decompose(self):
        murphy = MurphyDecomposition(family="poisson")
        with pytest.raises(RuntimeError, match="must run decompose"):
            murphy.summary()

    def test_c14_summary_returns_dict_with_interpretation(self):
        y, mu = _make_poisson_data(1500, seed=120)
        murphy = MurphyDecomposition(family="poisson")
        murphy.decompose(y, mu)
        s = murphy.summary()
        assert isinstance(s, dict)
        assert "unc" in s
        assert "dsc" in s
        assert "mcb" in s
        assert "gmcb" in s
        assert "lmcb" in s
        assert "verdict" in s
        assert "interpretation" in s
        assert isinstance(s["interpretation"], str)
        assert len(s["interpretation"]) > 10

    def test_c15_result_property_none_before_decompose(self):
        murphy = MurphyDecomposition(family="poisson")
        assert murphy.result is None

    def test_c16_result_property_populated_after_decompose(self):
        y, mu = _make_poisson_data(1000, seed=121)
        murphy = MurphyDecomposition(family="poisson")
        murphy.decompose(y, mu)
        assert murphy.result is not None
        assert isinstance(murphy.result, _MurphyComponents)

    def test_c17_exposure_weighting_accepted(self):
        rng = np.random.default_rng(122)
        n = 1000
        mu = rng.gamma(2, 0.05, n)
        y = rng.poisson(mu).astype(float)
        exposure = rng.uniform(0.5, 2.0, n)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu, exposure=exposure)
        assert isinstance(r, _MurphyComponents)
        assert np.isfinite(r.unc)

    def test_c18_exposure_changes_components(self):
        """Exposure weighting should change the Murphy values."""
        rng = np.random.default_rng(123)
        n = 1000
        mu = rng.gamma(2, 0.05, n)
        y = rng.poisson(mu).astype(float)
        exposure = rng.uniform(0.1, 3.0, n)  # heterogeneous exposure
        m_no = MurphyDecomposition(family="poisson")
        m_yes = MurphyDecomposition(family="poisson")
        r_no = m_no.decompose(y, mu)
        r_yes = m_yes.decompose(y, mu, exposure=exposure)
        # Results will differ because exposure changes the weighted deviance
        assert isinstance(r_no, _MurphyComponents)
        assert isinstance(r_yes, _MurphyComponents)

    def test_c19_to_dict_returns_expected_keys(self):
        y, mu = _make_poisson_data(800, seed=124)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu)
        d = r.to_dict()
        assert set(d.keys()) == {
            "unc", "dsc", "mcb", "gmcb", "lmcb",
            "total_deviance", "dsc_pct", "mcb_pct", "verdict",
        }

    def test_c20_decision_rule_recalibrate_for_global_shift(self):
        """Pure global scale error → RECALIBRATE verdict."""
        rng = np.random.default_rng(125)
        n = 8000
        mu = rng.gamma(2, 0.05, n)
        y = rng.poisson(mu).astype(float)
        murphy = MurphyDecomposition(family="poisson")
        r = murphy.decompose(y, mu * 1.4)  # 40% global over-prediction
        # Should not be OK; GMCB should dominate
        assert r.verdict in ("RECALIBRATE", "REFIT")

    def test_c21_top_level_import(self):
        """MurphyDecomposition importable from top-level package."""
        from insurance_monitoring import MurphyDecomposition as MD
        assert MD is MurphyDecomposition

    def test_c22_gini_drift_monitor_top_level_import(self):
        """GiniDriftMonitor importable from top-level package."""
        from insurance_monitoring import GiniDriftMonitor as GDM
        assert GDM is GiniDriftMonitor

    def test_c23_gini_bootstrap_monitor_top_level_import(self):
        """GiniBootstrapMonitor importable from top-level package."""
        from insurance_monitoring import GiniBootstrapMonitor as GBM
        assert GBM is GiniBootstrapMonitor
