"""Tests for CalibrationCUSUM (cusum.py).

Test cases follow the specification in the build spec (Section 5.2).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from insurance_monitoring.cusum import (
    CalibrationCUSUM,
    CUSUMAlarm,
    CUSUMSummary,
)


# ---------------------------------------------------------------------------
# TC-CC-01: In-control produces few alarms
# ---------------------------------------------------------------------------


def test_in_control_few_alarms():
    """TC-CC-01: calibrated data at cfar=0.005 should alarm ~0.5 times in 100 steps."""
    rng = np.random.default_rng(10)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=1000, random_state=0)
    alarms = 0
    for _ in range(100):
        n_t = 50
        p = rng.uniform(0.05, 0.25, n_t)
        # Well-calibrated: y ~ Bernoulli(p)
        y = rng.binomial(1, p)
        if monitor.update(p, y):
            alarms += 1
    # At cfar=0.005, expected false alarms in 100 steps ~ 0.5.
    # Allow up to 5 for statistical fluctuation.
    assert alarms <= 5, (
        f"In-control chart produced {alarms} alarms in 100 steps at cfar=0.005"
    )


# ---------------------------------------------------------------------------
# TC-CC-02: Out-of-control detects uplift within ARL1
# ---------------------------------------------------------------------------


def test_out_of_control_detects_uplift():
    """TC-CC-02: 2x miscalibration should be detected within 100 steps."""
    rng = np.random.default_rng(11)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=1000, random_state=1)
    detection = None
    for t in range(100):
        p_pred = rng.uniform(0.05, 0.20, 50)
        # True rate is 2x predicted
        y = rng.binomial(1, np.clip(2.0 * p_pred, 0, 1))
        if monitor.update(p_pred, y) and detection is None:
            detection = t + 1
            break
    # With delta=2, ARL1 ~ 37. Should detect well within 100 steps.
    assert detection is not None, (
        "Out-of-control chart failed to detect 2x miscalibration in 100 steps"
    )
    assert detection <= 80, (
        f"Detection at step {detection} is later than expected (ARL1 ~ 37)"
    )


# ---------------------------------------------------------------------------
# TC-CC-03: reset() restores state to zero
# ---------------------------------------------------------------------------


def test_reset_restores_state():
    """TC-CC-03: reset() sets statistic=0 and time=0."""
    rng = np.random.default_rng(12)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=500, random_state=2)
    for _ in range(20):
        p = rng.uniform(0.1, 0.3, 30)
        monitor.update(p, rng.binomial(1, np.clip(1.5 * p, 0, 1)))

    # After running with out-of-control data, statistic should be positive or zero
    # (may have reset if alarm fired). Time should be 20.
    assert monitor.time == 20

    monitor.reset()
    assert monitor.statistic == 0.0
    assert monitor.time == 0


# ---------------------------------------------------------------------------
# TC-CC-04: Poisson distribution mode works
# ---------------------------------------------------------------------------


def test_poisson_mode():
    """TC-CC-04: Poisson distribution mode returns correct CUSUMAlarm."""
    rng = np.random.default_rng(13)
    monitor = CalibrationCUSUM(
        delta_a=1.5, distribution="poisson", n_mc=1000, random_state=3
    )
    mu = rng.uniform(0.05, 0.15, 200)
    exposure = rng.uniform(0.5, 2.0, 200)
    y = rng.poisson(mu * exposure)

    alarm = monitor.update(mu, y, exposure=exposure)
    assert isinstance(alarm, CUSUMAlarm)
    assert alarm.statistic >= 0.0
    assert alarm.control_limit > 0.0
    assert alarm.n_obs == 200
    assert isinstance(bool(alarm), bool)


# ---------------------------------------------------------------------------
# TC-CC-05: Identity alternative raises ValueError
# ---------------------------------------------------------------------------


def test_identity_alternative_raises():
    """TC-CC-05: delta_a=1, gamma_a=1 (identity) raises ValueError."""
    with pytest.raises(ValueError, match="delta_a.*gamma_a|alternative|identity"):
        CalibrationCUSUM(delta_a=1.0, gamma_a=1.0)


# ---------------------------------------------------------------------------
# TC-CC-06: summary() tracks alarm count correctly
# ---------------------------------------------------------------------------


def test_summary_tracks_alarms():
    """TC-CC-06: summary() reports correct alarm count after multiple updates."""
    rng = np.random.default_rng(14)
    # cfar=0.1 gives ~3 false alarms in 30 steps
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.1, n_mc=500, random_state=4)
    for _ in range(30):
        p = rng.uniform(0.05, 0.25, 30)
        monitor.update(p, rng.binomial(1, p))

    s = monitor.summary()
    assert s.n_time_steps == 30
    assert isinstance(s.n_alarms, int)
    assert s.n_alarms >= 0
    assert len(s.alarm_times) == s.n_alarms
    assert all(isinstance(t, int) for t in s.alarm_times)
    assert all(1 <= t <= 30 for t in s.alarm_times)


# ---------------------------------------------------------------------------
# TC-CC-07: plot() smoke test
# ---------------------------------------------------------------------------


def test_plot_smoke():
    """TC-CC-07: plot() returns a matplotlib Axes with no error."""
    rng = np.random.default_rng(15)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=200, random_state=5)
    for _ in range(10):
        p = rng.uniform(0.05, 0.25, 20)
        monitor.update(p, rng.binomial(1, p))

    ax = monitor.plot()
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# TC-CC-08: Control limits are always positive
# ---------------------------------------------------------------------------


def test_control_limits_always_positive():
    """TC-CC-08: dynamic control limits h_t must be strictly positive at each step."""
    rng = np.random.default_rng(16)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=1000, random_state=6)
    limits = []
    for _ in range(20):
        alarm = monitor.update(
            rng.uniform(0.05, 0.25, 50),
            rng.binomial(1, rng.uniform(0.05, 0.25, 50)),
        )
        limits.append(alarm.control_limit)

    assert all(h > 0.0 for h in limits), (
        f"Some control limits were not positive: {limits}"
    )


# ---------------------------------------------------------------------------
# Additional API tests
# ---------------------------------------------------------------------------


def test_cusum_alarm_bool():
    """CUSUMAlarm evaluates as bool correctly."""
    alarm_on = CUSUMAlarm(
        triggered=True, time=5, statistic=3.2, control_limit=2.1,
        log_likelihood_ratio=1.5, n_obs=100
    )
    alarm_off = CUSUMAlarm(
        triggered=False, time=5, statistic=1.0, control_limit=2.1,
        log_likelihood_ratio=0.5, n_obs=100
    )
    assert bool(alarm_on) is True
    assert bool(alarm_off) is False


def test_summary_to_dict():
    """CUSUMSummary.to_dict() returns the expected structure."""
    rng = np.random.default_rng(20)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.01, n_mc=500, random_state=7)
    for _ in range(5):
        p = rng.uniform(0.05, 0.20, 30)
        monitor.update(p, rng.binomial(1, p))

    s = monitor.summary()
    d = s.to_dict()
    assert "n_time_steps" in d
    assert "n_alarms" in d
    assert "alarm_times" in d
    assert "current_statistic" in d
    assert "current_control_limit" in d
    assert d["n_time_steps"] == 5


def test_repr():
    """repr contains key parameters and current state."""
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=1000)
    r = repr(monitor)
    assert "CalibrationCUSUM" in r
    assert "delta_a=2.0" in r
    assert "cfar=0.005" in r


def test_time_property_increments():
    """time property increments correctly with each update."""
    rng = np.random.default_rng(30)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=200, random_state=8)
    assert monitor.time == 0
    for expected_t in range(1, 6):
        p = rng.uniform(0.1, 0.2, 20)
        monitor.update(p, rng.binomial(1, p))
        assert monitor.time == expected_t


def test_statistic_non_negative():
    """CUSUM statistic must always be non-negative."""
    rng = np.random.default_rng(40)
    monitor = CalibrationCUSUM(delta_a=0.5, cfar=0.005, n_mc=500, random_state=9)
    for _ in range(30):
        p = rng.uniform(0.05, 0.30, 40)
        monitor.update(p, rng.binomial(1, p))
    assert monitor.statistic >= 0.0


def test_alarm_history_persists_after_reset():
    """Alarm history should not be cleared by reset()."""
    rng = np.random.default_rng(50)
    # Use cfar=0.1 to force some false alarms in 20 steps
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.1, n_mc=500, random_state=10)
    for _ in range(20):
        p = rng.uniform(0.05, 0.25, 30)
        monitor.update(p, rng.binomial(1, p))

    n_alarms_before = monitor.summary().n_alarms
    monitor.reset()
    assert monitor.summary().n_alarms == n_alarms_before, (
        "Alarm count should persist after reset()"
    )


def test_length_mismatch_raises():
    """Mismatched predictions and outcomes lengths should raise ValueError."""
    monitor = CalibrationCUSUM(delta_a=2.0, n_mc=100)
    with pytest.raises(ValueError):
        monitor.update(np.ones(10), np.ones(15))


def test_negative_outcomes_raises():
    """Negative outcomes should raise ValueError."""
    monitor = CalibrationCUSUM(delta_a=2.0, n_mc=100)
    with pytest.raises(ValueError):
        monitor.update(np.full(10, 0.1), np.array([-1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))


def test_poisson_mode_no_delta1_raises():
    """delta_a=1.0 in Poisson mode should raise ValueError."""
    with pytest.raises(ValueError, match="delta_a"):
        CalibrationCUSUM(delta_a=1.0, distribution="poisson")


def test_invalid_cfar_raises():
    """cfar outside (0, 1) should raise ValueError."""
    with pytest.raises(ValueError):
        CalibrationCUSUM(delta_a=2.0, cfar=1.1)
    with pytest.raises(ValueError):
        CalibrationCUSUM(delta_a=2.0, cfar=0.0)


def test_invalid_distribution_raises():
    """Unsupported distribution should raise ValueError."""
    with pytest.raises(ValueError, match="distribution"):
        CalibrationCUSUM(delta_a=2.0, distribution="gamma")


def test_plot_empty_history():
    """plot() with no update history should return an Axes without error."""
    import matplotlib.pyplot as plt
    monitor = CalibrationCUSUM(delta_a=2.0, n_mc=100)
    ax = monitor.plot(title="Empty CUSUM")
    assert ax is not None
    plt.close("all")


def test_poisson_mode_out_of_control():
    """Poisson mode should detect 2x miscalibration within a reasonable run length."""
    rng = np.random.default_rng(60)
    monitor = CalibrationCUSUM(
        delta_a=2.0, distribution="poisson", cfar=0.005, n_mc=1000, random_state=11
    )
    detection = None
    for t in range(100):
        mu = rng.uniform(0.05, 0.15, 100)
        exposure = rng.uniform(0.5, 2.0, 100)
        # True rate is 2x predicted: y ~ Poisson(2 * mu * exposure)
        y = rng.poisson(2.0 * mu * exposure)
        if monitor.update(mu, y, exposure=exposure) and detection is None:
            detection = t + 1
            break
    assert detection is not None, (
        "Poisson CUSUM failed to detect 2x miscalibration in 100 steps"
    )
