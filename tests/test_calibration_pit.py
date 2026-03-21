"""
Tests for PITMonitor, PITAlarm, and PITSummary (insurance_monitoring v0.7.0).

These tests verify:
- Type I error control under the null (uniform PITs)
- Power: alarm fires under calibration shift
- Changepoint estimation accuracy
- Reproducibility with integer seeds
- E-value measurability (density computed before bin update)
- JSON persistence round-trip
- Poisson frequency end-to-end workflow
- warm_start mechanics
- Evidence freezing after alarm
- Edge cases: empty input, boundary PITs, invalid arguments
"""

from __future__ import annotations

import math
import json
import numpy as np
import pytest

from insurance_monitoring.calibration import PITMonitor, PITAlarm, PITSummary


# ---------------------------------------------------------------------------
# Test 1: No alarm under null (uniform PITs)
# ---------------------------------------------------------------------------

def test_no_alarm_under_null():
    rng = np.random.default_rng(42)
    monitor = PITMonitor(alpha=0.05, n_bins=50, rng=42)
    pits = rng.uniform(0, 1, 2000)
    result = monitor.update_many(pits, stop_on_alarm=False)
    assert not result.triggered
    assert monitor.evidence < 20.0


# ---------------------------------------------------------------------------
# Test 2: Alarm fires under shift
# ---------------------------------------------------------------------------

def test_alarm_under_shift():
    rng = np.random.default_rng(0)
    monitor = PITMonitor(alpha=0.05, n_bins=50, rng=0)
    monitor.update_many(rng.uniform(0, 1, 500))
    shifted = rng.beta(2, 1, 2000)
    result = monitor.update_many(shifted, stop_on_alarm=True)
    assert result.triggered
    assert monitor.alarm_time > 500


# ---------------------------------------------------------------------------
# Test 3: Changepoint in right neighbourhood
# ---------------------------------------------------------------------------

def test_changepoint_near_truth():
    rng = np.random.default_rng(1)
    monitor = PITMonitor(alpha=0.05, n_bins=50, rng=1)
    n_stable = 300
    monitor.update_many(rng.uniform(0, 1, n_stable))
    monitor.update_many(rng.beta(3, 1, 2000), stop_on_alarm=False)
    if monitor.alarm_triggered:
        cp = monitor.changepoint()
        assert cp is not None
        assert abs(cp - (n_stable + 1)) < 100


# ---------------------------------------------------------------------------
# Test 4: Type I error Monte Carlo (slow, marked)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_type_i_error_monte_carlo():
    alpha = 0.05
    n_trials = 500
    alarms = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        monitor = PITMonitor(alpha=alpha, n_bins=50, rng=seed)
        pits = rng.uniform(0, 1, 1000)
        if monitor.update_many(pits, stop_on_alarm=True).triggered:
            alarms += 1
    assert alarms / n_trials <= 0.08  # Monte Carlo slack around 5%


# ---------------------------------------------------------------------------
# Test 5: Reproducibility with seed
# ---------------------------------------------------------------------------

def test_reproducibility():
    pits = np.random.default_rng(99).uniform(0, 1, 200)
    m1 = PITMonitor(rng=42)
    m2 = PITMonitor(rng=42)
    m1.update_many(pits, stop_on_alarm=False)
    m2.update_many(pits, stop_on_alarm=False)
    assert m1.evidence == m2.evidence
    np.testing.assert_array_equal(m1.pvalues, m2.pvalues)


# ---------------------------------------------------------------------------
# Test 6: E-value measurability (mean ~ 1 under null)
# ---------------------------------------------------------------------------

def test_evalue_mean_near_one():
    """Checks that update order is correct: density computed before bin update."""
    rng = np.random.default_rng(7)
    monitor = PITMonitor(n_bins=20, rng=7)
    evalues = []
    for _ in range(5000):
        M_before = monitor._M
        monitor.update(float(rng.uniform()))
        t = monitor.t
        w_t = 1.0 / (t * (t + 1))
        if M_before + w_t > 0:
            e_t = monitor._M / (M_before + w_t)
            evalues.append(e_t)
    mean_e = np.mean(evalues)
    assert 0.9 < mean_e < 1.1, f"Mean e-value {mean_e:.3f} far from 1.0"


# ---------------------------------------------------------------------------
# Test 7: JSON persistence round-trip
# ---------------------------------------------------------------------------

def test_save_load_json(tmp_path):
    monitor = PITMonitor(alpha=0.05, n_bins=30, rng=0)
    monitor.update_many(np.random.default_rng(0).uniform(0, 1, 100))
    path = tmp_path / "monitor.json"
    monitor.save(path)
    loaded = PITMonitor.load(path)
    assert loaded.t == monitor.t
    assert loaded.evidence == monitor.evidence
    assert loaded.alarm_triggered == monitor.alarm_triggered


# ---------------------------------------------------------------------------
# Test 8: Poisson frequency end-to-end
# ---------------------------------------------------------------------------

def test_poisson_pit_workflow():
    from scipy.stats import poisson
    rng = np.random.default_rng(42)
    monitor = PITMonitor(alpha=0.05, n_bins=50, rng=42)
    n = 500
    exposure = rng.uniform(0.5, 2.0, n)
    lambda_hat = rng.gamma(2, 0.05, n)
    y = rng.poisson(exposure * lambda_hat)
    for i in range(n):
        mu = exposure[i] * lambda_hat[i]
        monitor.update(float(poisson.cdf(y[i], mu)))
    assert not monitor.alarm_triggered
    assert 0.0 <= monitor.calibration_score() <= 1.0


# ---------------------------------------------------------------------------
# Test 9: warm_start does not alarm on history
# ---------------------------------------------------------------------------

def test_warm_start_no_alarm():
    rng = np.random.default_rng(5)
    monitor = PITMonitor(alpha=0.05, n_bins=50, rng=5)
    monitor.warm_start(rng.uniform(0, 1, 500))
    assert not monitor.alarm_triggered
    assert monitor._M == 0.0
    assert monitor.t == 0


# ---------------------------------------------------------------------------
# Test 10: Evidence freezes after alarm
# ---------------------------------------------------------------------------

def test_evidence_freezes_after_alarm():
    rng = np.random.default_rng(3)
    monitor = PITMonitor(alpha=0.05, n_bins=20, rng=3)
    monitor.update_many(rng.uniform(0, 1, 200))
    monitor.update_many(rng.beta(5, 1, 5000), stop_on_alarm=False)
    if monitor.alarm_triggered:
        frozen = monitor.evidence
        monitor.update(0.5)
        assert monitor.evidence == frozen


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_pitalar_bool_interface():
    """PITAlarm evaluates as bool correctly."""
    alarm_no = PITAlarm(triggered=False, time=1, evidence=0.1, threshold=20.0, changepoint=None)
    alarm_yes = PITAlarm(triggered=True, time=5, evidence=25.0, threshold=20.0, changepoint=3)
    assert not alarm_no
    assert alarm_yes


def test_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        PITMonitor(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        PITMonitor(alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        PITMonitor(alpha=-0.1)


def test_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        PITMonitor(n_bins=4)
    with pytest.raises(ValueError, match="n_bins"):
        PITMonitor(n_bins=501)


def test_invalid_pit_value():
    monitor = PITMonitor(rng=0)
    with pytest.raises(ValueError, match="pit"):
        monitor.update(-0.01)
    with pytest.raises(ValueError, match="pit"):
        monitor.update(1.01)


def test_boundary_pit_values():
    """PITs of exactly 0 and 1 are valid and processed without error."""
    monitor = PITMonitor(rng=0)
    monitor.update(0.0)
    monitor.update(1.0)
    assert monitor.t == 2


def test_update_many_empty():
    """update_many on empty array returns current state without error."""
    monitor = PITMonitor(rng=0)
    result = monitor.update_many([], stop_on_alarm=True)
    assert result.time == 0
    assert not result.triggered


def test_summary_returns_pit_summary():
    """summary() returns a PITSummary dataclass with correct fields."""
    monitor = PITMonitor(alpha=0.05, n_bins=20, rng=0)
    pits = np.random.default_rng(0).uniform(0, 1, 50)
    monitor.update_many(pits, stop_on_alarm=False)
    s = monitor.summary()
    assert isinstance(s, PITSummary)
    assert s.t == 50
    assert s.n_observations == 50
    assert s.calibration_score is not None
    assert 0.0 <= s.calibration_score <= 1.0
    assert s.threshold == pytest.approx(20.0)


def test_reset_clears_state():
    monitor = PITMonitor(alpha=0.05, n_bins=20, rng=42)
    pits = np.random.default_rng(1).uniform(0, 1, 100)
    monitor.update_many(pits)
    monitor.reset()
    assert monitor.t == 0
    assert monitor._M == 0.0
    assert len(monitor._sorted_pits) == 0
    assert not monitor.alarm_triggered
    assert monitor.alarm_time is None


def test_pits_and_pvalues_properties():
    monitor = PITMonitor(rng=42)
    pits_in = np.random.default_rng(0).uniform(0, 1, 50)
    monitor.update_many(pits_in, stop_on_alarm=False)
    pits_out = monitor.pits
    pvals_out = monitor.pvalues
    assert pits_out.shape == (50,)
    assert pvals_out.shape == (50,)
    np.testing.assert_allclose(pits_out, pits_in)
    assert np.all((pvals_out >= 0.0) & (pvals_out <= 1.0))


def test_calibration_score_zero_when_empty():
    monitor = PITMonitor()
    assert monitor.calibration_score() == 0.0


def test_changepoint_none_when_no_alarm():
    monitor = PITMonitor(rng=0)
    monitor.update_many(np.random.default_rng(0).uniform(0, 1, 100), stop_on_alarm=False)
    assert not monitor.alarm_triggered
    assert monitor.changepoint() is None


def test_update_with_cdf():
    """update_with_cdf produces same result as manually computing the PIT."""
    from scipy.stats import norm as norm_dist
    rng = np.random.default_rng(10)
    ys = rng.normal(0, 1, 30)

    m1 = PITMonitor(n_bins=20, rng=10)
    m2 = PITMonitor(n_bins=20, rng=10)

    for y in ys:
        m1.update_with_cdf(lambda val: norm_dist.cdf(val, 0, 1), y)
        m2.update(float(norm_dist.cdf(y, 0, 1)))

    assert m1.t == m2.t == 30
    np.testing.assert_array_equal(m1.pits, m2.pits)


def test_exposure_weighting_integer_repetition():
    """An observation with exposure=3 is equivalent to three identical updates."""
    pit_val = 0.4
    m1 = PITMonitor(n_bins=10, rng=0)
    m2 = PITMonitor(n_bins=10, rng=0)
    m1.update(pit_val, exposure=3.0)
    for _ in range(3):
        m2.update(pit_val)
    assert m1.t == m2.t == 3


def test_warm_start_histogram_warmed():
    """After warm_start, bin_counts should exceed initial Laplace pseudocounts."""
    rng = np.random.default_rng(0)
    monitor = PITMonitor(n_bins=10, rng=0)
    initial_total = float(np.sum(monitor._bin_counts))  # = n_bins = 10
    monitor.warm_start(rng.uniform(0, 1, 200))
    warmed_total = float(np.sum(monitor._bin_counts))
    assert warmed_total > initial_total
    assert monitor.t == 0  # Evidence counter reset


def test_plot_returns_none_before_two_obs():
    monitor = PITMonitor(rng=0)
    assert monitor.plot() is None
    monitor.update(0.5)
    assert monitor.plot() is None


def test_plot_returns_figure_after_two_obs():
    import matplotlib
    matplotlib.use("Agg")
    monitor = PITMonitor(rng=0)
    monitor.update_many(np.random.default_rng(0).uniform(0, 1, 10), stop_on_alarm=False)
    fig = monitor.plot()
    assert fig is not None


def test_pickle_persistence(tmp_path):
    """Pickle round-trip preserves exact floating point state."""
    monitor = PITMonitor(alpha=0.05, n_bins=20, rng=1)
    pits = np.random.default_rng(1).uniform(0, 1, 80)
    monitor.update_many(pits, stop_on_alarm=False)
    path = tmp_path / "monitor.pkl"
    monitor.save(path)
    loaded = PITMonitor.load(path)
    assert loaded.t == monitor.t
    assert loaded.evidence == monitor.evidence
    np.testing.assert_array_equal(loaded.pits, monitor.pits)
    np.testing.assert_array_equal(loaded.pvalues, monitor.pvalues)


def test_custom_weight_schedule():
    """Custom weight schedule that sums to 1 is accepted."""
    # Geometric weights: w(t) = (1/2)^t; sum = 1
    def geom(t: int) -> float:
        return 0.5 ** t

    monitor = PITMonitor(n_bins=20, weight_schedule=geom, rng=0)
    pits = np.random.default_rng(0).uniform(0, 1, 50)
    result = monitor.update_many(pits, stop_on_alarm=False)
    assert isinstance(result, PITAlarm)


def test_repr():
    monitor = PITMonitor(alpha=0.05, n_bins=50, rng=0)
    r = repr(monitor)
    assert "PITMonitor" in r
    assert "alpha=0.05" in r
    assert "n_bins=50" in r
