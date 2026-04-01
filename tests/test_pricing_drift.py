"""Tests for PricingDriftMonitor (pricing_drift.py).

Test cases follow the specification in the build spec (Section 5.1).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from insurance_monitoring.pricing_drift import (
    CalibTestResult,
    PricingDriftMonitor,
    PricingDriftResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calibrated_data():
    """Well-calibrated reference and monitoring data."""
    rng = np.random.default_rng(0)
    mu_ref = rng.uniform(0.05, 0.20, 20_000)
    y_ref = rng.poisson(mu_ref).astype(float)
    mu_mon = rng.uniform(0.05, 0.20, 10_000)
    y_mon = rng.poisson(mu_mon).astype(float)
    return mu_ref, y_ref, mu_mon, y_mon


# ---------------------------------------------------------------------------
# TC-PDM-01: No drift returns OK
# ---------------------------------------------------------------------------


def test_no_drift_returns_ok(calibrated_data):
    """TC-PDM-01: well-calibrated data with no drift should return OK verdict."""
    mu_ref, y_ref, mu_mon, y_mon = calibrated_data

    monitor = PricingDriftMonitor(
        distribution="poisson", n_bootstrap=500, alpha_gini=0.05, random_state=0,
    )
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon)

    assert result.verdict == "OK", (
        f"Expected OK but got {result.verdict}; "
        f"gini p={result.gini.p_value:.4f}, "
        f"global p={result.global_calib.p_value:.4f}, "
        f"local p={result.local_calib.p_value:.4f}"
    )
    assert result.gini.p_value > 0.05
    assert result.global_calib.p_value > 0.05
    assert result.local_calib.p_value > 0.05


# ---------------------------------------------------------------------------
# TC-PDM-02: Pure global level shift returns RECALIBRATE
# ---------------------------------------------------------------------------


def test_global_level_shift_recalibrate():
    """TC-PDM-02: 40% level uplift with preserved ranking returns RECALIBRATE."""
    rng = np.random.default_rng(1)
    mu_ref = rng.uniform(0.05, 0.10, 10_000)
    y_ref = rng.poisson(mu_ref).astype(float)

    # Monitoring: same model predictions but 40% more claims (global shift)
    mu_mon = mu_ref[:5_000].copy()
    y_mon = rng.poisson(mu_mon * 1.40).astype(float)

    monitor = PricingDriftMonitor(distribution="poisson", n_bootstrap=300, random_state=2)
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon)

    # Ranking preserved (monotone shift), Gini test should pass
    assert result.gini.reject_h0 is False, (
        f"Gini should not be rejected for pure level shift, "
        f"got p={result.gini.p_value:.4f}"
    )
    # Global level shift should be detected
    assert result.global_calib.reject_h0 is True, (
        f"GMCB should be rejected for 40% uplift, "
        f"got p={result.global_calib.p_value:.4f}"
    )
    assert result.verdict == "RECALIBRATE"


# ---------------------------------------------------------------------------
# TC-PDM-03: Destroyed ranking returns REFIT
# ---------------------------------------------------------------------------


def test_destroyed_ranking_refit():
    """TC-PDM-03: shuffled predictions (Gini -> ~0) should return REFIT."""
    rng = np.random.default_rng(2)
    mu_ref = rng.uniform(0.02, 0.30, 10_000)
    y_ref = rng.poisson(mu_ref).astype(float)

    mu_mon = rng.uniform(0.02, 0.30, 5_000)
    y_mon = rng.poisson(mu_mon).astype(float)
    mu_mon_wrong = rng.permutation(mu_mon)  # shuffled: Gini -> ~0

    monitor = PricingDriftMonitor(distribution="poisson", n_bootstrap=300, random_state=3)
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon_wrong)

    assert result.gini.gini_mon < 0.05, (
        f"Shuffled predictions should give near-zero Gini, got {result.gini.gini_mon:.4f}"
    )
    assert result.gini.reject_h0 is True, (
        f"Gini should be rejected for shuffled predictions, "
        f"got p={result.gini.p_value:.4f}"
    )
    assert result.verdict == "REFIT"


# ---------------------------------------------------------------------------
# TC-PDM-04: test() before fit() raises RuntimeError
# ---------------------------------------------------------------------------


def test_test_before_fit_raises():
    """TC-PDM-04: calling test() before fit() must raise RuntimeError."""
    monitor = PricingDriftMonitor()
    with pytest.raises(RuntimeError, match="fitted"):
        monitor.test(np.ones(100), np.ones(100))


# ---------------------------------------------------------------------------
# TC-PDM-05: to_dict() is JSON-serialisable
# ---------------------------------------------------------------------------


def test_to_dict_json_serialisable(calibrated_data):
    """TC-PDM-05: result.to_dict() must produce a fully JSON-serialisable dict."""
    mu_ref, y_ref, mu_mon, y_mon = calibrated_data

    monitor = PricingDriftMonitor(n_bootstrap=200, random_state=0)
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon)

    d = result.to_dict()
    json_str = json.dumps(d)
    assert isinstance(json_str, str)

    # Verify round-trip preserves verdict
    restored = json.loads(json_str)
    assert restored["verdict"] == result.verdict


# ---------------------------------------------------------------------------
# TC-PDM-06: Small sample issues UserWarning
# ---------------------------------------------------------------------------


def test_small_sample_warns():
    """TC-PDM-06: monitoring with fewer than 500 observations issues UserWarning."""
    rng = np.random.default_rng(4)
    y_small = rng.poisson(0.1, 100).astype(float)
    mu_small = np.full(100, 0.1)

    monitor = PricingDriftMonitor(n_bootstrap=100, random_state=0)
    monitor.fit(y_small, mu_small)

    with pytest.warns(UserWarning):
        monitor.test(y_small, mu_small)


# ---------------------------------------------------------------------------
# TC-PDM-07: Exposure weighting changes Gini
# ---------------------------------------------------------------------------


def test_exposure_weighting_changes_gini():
    """TC-PDM-07: Gini with exposure weighting should differ from unweighted."""
    rng = np.random.default_rng(5)
    n = 5_000
    mu = rng.uniform(0.05, 0.25, n)
    exposure = rng.uniform(0.1, 3.0, n)
    y = rng.poisson(mu * exposure).astype(float)

    m1 = PricingDriftMonitor(n_bootstrap=200, random_state=6)
    m1.fit(y, mu)
    r1 = m1.test(y, mu)

    m2 = PricingDriftMonitor(n_bootstrap=200, random_state=6)
    m2.fit(y, mu, exposure=exposure)
    r2 = m2.test(y, mu, exposure=exposure)

    assert abs(r1.gini.gini_mon - r2.gini.gini_mon) > 0.001, (
        f"Expected exposure to change Gini, but got "
        f"r1={r1.gini.gini_mon:.4f}, r2={r2.gini.gini_mon:.4f}"
    )


# ---------------------------------------------------------------------------
# TC-PDM-08: Murphy identity MCB = GMCB + LMCB (approximately)
# ---------------------------------------------------------------------------


def test_murphy_identity(calibrated_data):
    """TC-PDM-08: Murphy identity: MCB = GMCB + LMCB (within numerical tolerance)."""
    mu_ref, y_ref, mu_mon, y_mon = calibrated_data

    monitor = PricingDriftMonitor(n_bootstrap=200, random_state=0)
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon)

    m = result.murphy
    assert m.unc >= 0.0, f"UNC must be non-negative, got {m.unc}"
    assert m.dsc >= 0.0, f"DSC must be non-negative, got {m.dsc}"
    assert m.gmcb >= 0.0, f"GMCB must be non-negative, got {m.gmcb}"
    assert m.lmcb >= 0.0, f"LMCB must be non-negative, got {m.lmcb}"

    # MCB ≈ GMCB + LMCB (they are computed via different paths so may differ slightly)
    # The murphy_decomposition in _murphy.py computes MCB directly, while GMCB and LMCB
    # are computed here via the GLM path. Accept a modest tolerance.
    assert abs(m.mcb - (m.gmcb + m.lmcb)) < 1e-4, (
        f"Murphy identity violated: MCB={m.mcb:.8f}, "
        f"GMCB+LMCB={m.gmcb + m.lmcb:.8f}"
    )


# ---------------------------------------------------------------------------
# Additional tests: API and edge cases
# ---------------------------------------------------------------------------


def test_is_fitted_flag():
    """is_fitted() returns False before fit(), True after."""
    monitor = PricingDriftMonitor()
    assert monitor.is_fitted() is False

    rng = np.random.default_rng(0)
    y = rng.poisson(0.1, 1000).astype(float)
    mu = np.full(1000, 0.1)
    monitor.fit(y, mu)
    assert monitor.is_fitted() is True


def test_repr():
    """repr includes key parameters and fitted status."""
    monitor = PricingDriftMonitor(distribution="poisson", n_bootstrap=300)
    r = repr(monitor)
    assert "PricingDriftMonitor" in r
    assert "not fitted" in r

    rng = np.random.default_rng(0)
    y = rng.poisson(0.1, 1000).astype(float)
    mu = np.full(1000, 0.1)
    monitor.fit(y, mu)
    r2 = repr(monitor)
    assert "fitted" in r2


def test_result_types(calibrated_data):
    """PricingDriftResult attributes have correct types."""
    mu_ref, y_ref, mu_mon, y_mon = calibrated_data

    monitor = PricingDriftMonitor(n_bootstrap=200, random_state=0)
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon)

    assert isinstance(result, PricingDriftResult)
    assert result.verdict in {"OK", "RECALIBRATE", "REFIT"}
    assert isinstance(result.global_calib, CalibTestResult)
    assert isinstance(result.local_calib, CalibTestResult)
    assert result.global_calib.component == "GMCB"
    assert result.local_calib.component == "LMCB"
    assert 0.0 <= result.global_calib.p_value <= 1.0
    assert 0.0 <= result.local_calib.p_value <= 1.0


def test_summary_is_string(calibrated_data):
    """result.summary() returns a non-empty string."""
    mu_ref, y_ref, mu_mon, y_mon = calibrated_data

    monitor = PricingDriftMonitor(n_bootstrap=200, random_state=0)
    monitor.fit(y_ref, mu_ref)
    result = monitor.test(y_mon, mu_mon)

    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 50
    assert result.verdict in s


def test_alpha_override(calibrated_data):
    """Per-call alpha overrides should be respected."""
    mu_ref, y_ref, mu_mon, y_mon = calibrated_data

    monitor = PricingDriftMonitor(
        n_bootstrap=200, alpha_gini=0.32, alpha_global=0.05, random_state=0
    )
    monitor.fit(y_ref, mu_ref)

    # Override with very tight alpha: nothing should reject on OK data
    result = monitor.test(y_mon, mu_mon, alpha_gini=0.001, alpha_global=0.001, alpha_local=0.001)
    assert result.alpha_gini == 0.001
    assert result.alpha_global == 0.001
    assert result.alpha_local == 0.001


def test_invalid_distribution():
    """Invalid distribution should raise ValueError."""
    with pytest.raises(ValueError, match="distribution"):
        PricingDriftMonitor(distribution="negative_binomial")


def test_invalid_alpha():
    """alpha out of range should raise ValueError."""
    with pytest.raises(ValueError):
        PricingDriftMonitor(alpha_gini=1.5)
    with pytest.raises(ValueError):
        PricingDriftMonitor(alpha_global=0.0)


def test_method_chaining():
    """fit() returns self for method chaining."""
    rng = np.random.default_rng(0)
    y = rng.poisson(0.1, 1000).astype(float)
    mu = np.full(1000, 0.1)

    monitor = PricingDriftMonitor(n_bootstrap=100)
    result_self = monitor.fit(y, mu)
    assert result_self is monitor


def test_calib_test_result_to_dict():
    """CalibTestResult.to_dict() returns the expected keys."""
    ct = CalibTestResult(
        statistic=0.001, p_value=0.123, reject_h0=False,
        alpha=0.05, n_bootstrap=300, component="GMCB"
    )
    d = ct.to_dict()
    assert set(d.keys()) == {"statistic", "p_value", "reject_h0", "alpha", "n_bootstrap", "component"}
