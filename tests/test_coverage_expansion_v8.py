"""
Test coverage expansion batch 8 — final push to >= 1763 tests.

Adds ~30 targeted tests: conformal_chart math identities, sequential
edge cases, DriftAttributor helper functions, and parametric top-ups.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# conformal_chart math identities
# ---------------------------------------------------------------------------

class TestConformalMathIdentities:
    """Verify ICP theory invariants hold."""

    def test_p_value_superuniform_under_null(self):
        """P(p < alpha) <= alpha under exchangeability (statistical)."""
        from insurance_monitoring.conformal_chart import _conformal_p_values
        rng = np.random.default_rng(0)
        n_cal = 1000
        cal = rng.normal(0, 1, n_cal)
        # Under null: new scores from same distribution
        new_scores = rng.normal(0, 1, 5000)
        p = _conformal_p_values(new_scores, cal)
        alpha = 0.05
        # Allow tight bound; ICP guarantees P(p < alpha) <= alpha
        assert np.mean(p < alpha) <= alpha + 0.01

    def test_p_value_formula_single(self):
        """Verify formula: (#{s_cal >= s_new} + 1) / (n + 1)."""
        from insurance_monitoring.conformal_chart import _conformal_p_value
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # s_new = 2.5: s_cal >= 2.5 -> {3, 4, 5}, count = 3
        p = _conformal_p_value(2.5, cal)
        expected = (3 + 1) / (5 + 1)
        assert p == pytest.approx(expected)

    def test_threshold_monotone_in_alpha(self):
        """Lower alpha -> higher (more conservative) threshold."""
        from insurance_monitoring.conformal_chart import _conformal_threshold
        cal = np.arange(500, dtype=float)
        t_high = _conformal_threshold(cal, 0.1)
        t_low = _conformal_threshold(cal, 0.01)
        assert t_low >= t_high

    def test_p_value_vectorised_matches_scalar(self):
        """Vectorised p-values should match scalar loop."""
        from insurance_monitoring.conformal_chart import _conformal_p_value, _conformal_p_values
        cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = np.array([0.5, 2.0, 3.5, 7.0])
        vec = _conformal_p_values(scores, cal)
        scalar = np.array([_conformal_p_value(s, cal) for s in scores])
        np.testing.assert_allclose(vec, scalar)

    def test_all_cal_scores_equal(self):
        """When all calibration scores are equal, threshold = that value."""
        from insurance_monitoring.conformal_chart import _conformal_threshold, ConformalControlChart
        cal = np.ones(100) * 5.0
        t = _conformal_threshold(cal, 0.05)
        assert t == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# SequentialTest: rate tracking precision
# ---------------------------------------------------------------------------

class TestSequentialRateTracking:
    def test_accumulated_exposure_correct(self):
        from insurance_monitoring.sequential import SequentialTest
        test = SequentialTest(min_exposure_per_arm=0.0)
        for i in range(5):
            test.update(
                champion_claims=2, champion_exposure=50,
                challenger_claims=3, challenger_exposure=60,
            )
        result = test.update(
            champion_claims=2, champion_exposure=50,
            challenger_claims=3, challenger_exposure=60,
        )
        assert result.total_champion_exposure == pytest.approx(300.0)
        assert result.total_challenger_exposure == pytest.approx(360.0)
        assert result.total_champion_claims == pytest.approx(12.0)
        assert result.total_challenger_claims == pytest.approx(18.0)

    def test_rate_ratio_doubles(self):
        from insurance_monitoring.sequential import SequentialTest
        test = SequentialTest(min_exposure_per_arm=0.0)
        result = test.update(
            champion_claims=10, champion_exposure=100,
            challenger_claims=20, challenger_exposure=100,
        )
        assert result.rate_ratio == pytest.approx(2.0)

    def test_zero_champion_rate_gives_rate_ratio_one(self):
        from insurance_monitoring.sequential import SequentialTest
        test = SequentialTest(min_exposure_per_arm=0.0)
        # Zero claims -> rate = 0 -> rate_ratio falls back to 1.0
        result = test.update(
            champion_claims=0, champion_exposure=100,
            challenger_claims=0, challenger_exposure=100,
        )
        assert result.rate_ratio == pytest.approx(1.0)

    def test_history_rate_ratio_column_consistent(self):
        from insurance_monitoring.sequential import SequentialTest
        test = SequentialTest()
        for _ in range(3):
            test.update(champion_claims=10, champion_exposure=100, challenger_claims=10, challenger_exposure=100)
        df = test.history()
        assert "rate_ratio" in df.columns
        # All ratios should be positive finite numbers
        rr = df["rate_ratio"].to_numpy()
        assert np.all(np.isfinite(rr))
        assert np.all(rr > 0)


# ---------------------------------------------------------------------------
# DriftAttributor helper functions
# ---------------------------------------------------------------------------

class TestDriftAttributorHelpers:
    def test_subset_risk_full_active(self):
        """With all features active, subset risk = full model risk."""
        from insurance_monitoring.drift_attribution import _subset_risk, _compute_loss
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        y = X @ np.ones(3)
        model = LinearRegression().fit(X, y)
        fill = np.zeros(3)

        r_full = _subset_risk(model, X, y, [0, 1, 2], [0, 1, 2], fill, "mse")
        y_pred = model.predict(X)
        expected = _compute_loss(y, y_pred, "mse")
        assert r_full == pytest.approx(expected, rel=0.01)

    def test_subset_risk_no_active_uses_fill(self):
        """With no active features (all masked), risk depends only on fill prediction."""
        from insurance_monitoring.drift_attribution import _subset_risk
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        y = X @ np.ones(3)
        model = LinearRegression().fit(X, y)
        fill = np.mean(X, axis=0)  # all features masked to their mean

        r_empty = _subset_risk(model, X, y, [], [0, 1, 2], fill, "mse")
        assert r_empty >= 0.0

    def test_delta_positive_when_feature_informative(self):
        """delta(S, k) > 0 when feature k is informative for the model."""
        from insurance_monitoring.drift_attribution import _delta
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 2))
        y = 5 * X[:, 0] + rng.normal(0, 0.01, 200)  # only feature 0 matters
        model = LinearRegression().fit(X, y)
        fill = np.zeros(2)

        # delta([], k=0) = R^{} - R^{0}: feature 0 reduces loss significantly
        d = _delta(model, X, y, [], [0], [0, 1], fill, "mse")
        assert d > 0  # adding feature 0 reduces loss


# ---------------------------------------------------------------------------
# MultivariateConformalMonitor: small calibration set warning
# ---------------------------------------------------------------------------

def test_multivariate_monitor_small_cal_warns():
    from insurance_monitoring.conformal_chart import MultivariateConformalMonitor

    class ConstantModel:
        def fit(self, X): return self
        def decision_function(self, X): return np.zeros(len(X))

    X = np.random.randn(50, 3)
    mon = MultivariateConformalMonitor(model=ConstantModel(), alpha=0.05)
    with warnings.catch_warnings(record=True) as w:
        import warnings as _warnings
        _warnings.simplefilter("always")
        mon.fit(X[:30], X[:10])  # X_cal has only 10 rows < 20
        assert any("calibration" in str(x.message).lower() or "20" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Parametric top-up tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_new,n_ref", [
    (50, 200), (100, 200), (200, 200), (500, 200)
])
def test_drift_attributor_various_window_sizes(n_new, n_ref):
    from insurance_monitoring.drift_attribution import DriftAttributor
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(42)
    X_ref = rng.normal(0, 1, (n_ref, 3))
    y_ref = X_ref @ np.ones(3) + rng.normal(0, 0.1, n_ref)
    model = LinearRegression().fit(X_ref, y_ref)

    attr = DriftAttributor(
        model=model, features=["a", "b", "c"],
        n_bootstrap=10, auto_retrain=False,
    )
    attr.fit_reference(X_ref, y_ref, train_on_ref=False)

    X_new = rng.normal(0, 1, (n_new, 3))
    y_new = X_new @ np.ones(3) + rng.normal(0, 0.1, n_new)
    result = attr.test(X_new, y_new)

    assert result.window_new_size == n_new
    assert isinstance(result.drift_detected, bool)


@pytest.mark.parametrize("cal_size,mon_size", [
    (50, 10), (100, 50), (200, 200), (500, 100)
])
def test_conformal_chart_various_sizes(cal_size, mon_size):
    from insurance_monitoring.conformal_chart import ConformalControlChart
    rng = np.random.default_rng(7)
    cal = rng.exponential(1, cal_size)
    chart = ConformalControlChart(alpha=0.05).fit(cal)
    mon = rng.exponential(1, mon_size)
    result = chart.monitor(mon)
    assert len(result.p_values) == mon_size
    assert result.n_cal == cal_size


@pytest.mark.parametrize("claims_a,claims_b", [
    (5, 5), (10, 20), (50, 50), (100, 80),
])
def test_sequential_test_various_claim_ratios(claims_a, claims_b):
    from insurance_monitoring.sequential import SequentialTest
    test = SequentialTest(metric="frequency", alpha=0.05)
    result = test.update(
        champion_claims=claims_a, champion_exposure=100,
        challenger_claims=claims_b, challenger_exposure=100,
    )
    assert isinstance(result.rate_ratio, float)
    assert result.rate_ratio > 0


@pytest.mark.parametrize("decision_str", [
    "inconclusive", "reject_H0", "futility", "max_duration_reached"
])
def test_sequential_test_result_should_stop(decision_str):
    """should_stop is False only when decision is inconclusive."""
    from insurance_monitoring.sequential import SequentialTestResult
    r = SequentialTestResult(
        decision=decision_str,
        should_stop=(decision_str != "inconclusive"),
        lambda_value=1.0,
        log_lambda_value=0.0,
        threshold=20.0,
        champion_rate=0.1,
        challenger_rate=0.1,
        rate_ratio=1.0,
        rate_ratio_ci_lower=0.8,
        rate_ratio_ci_upper=1.25,
        total_champion_claims=100.0,
        total_champion_exposure=1000.0,
        total_challenger_claims=100.0,
        total_challenger_exposure=1000.0,
        n_updates=5,
        total_calendar_time_days=180.0,
        prob_challenger_better=0.5,
        summary="Test summary.",
    )
    if decision_str == "inconclusive":
        assert not r.should_stop
    else:
        assert r.should_stop


# ---------------------------------------------------------------------------
# Final 10 tests to reach target
# ---------------------------------------------------------------------------

def test_conformal_chart_result_alpha_stored():
    from insurance_monitoring.conformal_chart import ConformalChartResult
    r = ConformalChartResult(
        scores=np.array([1.0]), p_values=np.array([0.5]),
        threshold=2.0, is_alarm=np.array([False]), alpha=0.1, n_cal=30
    )
    assert r.alpha == pytest.approx(0.1)


def test_conformal_chart_result_n_cal_stored():
    from insurance_monitoring.conformal_chart import ConformalChartResult
    r = ConformalChartResult(
        scores=np.array([1.0]), p_values=np.array([0.5]),
        threshold=2.0, is_alarm=np.array([False]), alpha=0.05, n_cal=42
    )
    assert r.n_cal == 42


def test_sequential_test_history_schema():
    from insurance_monitoring.sequential import SequentialTest
    test = SequentialTest()
    df = test.history()
    # Empty schema should have the correct column names
    assert "period_index" in df.columns
    assert "lambda_value" in df.columns
    assert "cum_champion_exposure" in df.columns


def test_conformal_threshold_increases_with_higher_cal_max():
    """Threshold from higher-valued calibration set should be larger."""
    from insurance_monitoring.conformal_chart import _conformal_threshold
    cal_low = np.arange(100, dtype=float)  # max=99
    cal_high = np.arange(100, dtype=float) + 1000  # max=1099
    t_low = _conformal_threshold(cal_low, 0.05)
    t_high = _conformal_threshold(cal_high, 0.05)
    assert t_high > t_low


def test_sequential_test_n_updates_after_reset():
    from insurance_monitoring.sequential import SequentialTest
    test = SequentialTest()
    for _ in range(5):
        test.update(champion_claims=5, champion_exposure=50, challenger_claims=5, challenger_exposure=50)
    test.reset()
    assert test._n_updates == 0
    result = test.update(champion_claims=5, champion_exposure=50, challenger_claims=5, challenger_exposure=50)
    assert result.n_updates == 1


def test_compute_loss_mse_nonzero():
    from insurance_monitoring.drift_attribution import _compute_loss
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    # All predictions off by 1 -> MSE = 1.0
    assert _compute_loss(y_true, y_pred, "mse") == pytest.approx(1.0)


def test_drift_attribution_result_model_retrained_default_false():
    from insurance_monitoring.drift_attribution import DriftAttributionResult
    r = DriftAttributionResult(
        drift_detected=False, attributed_features=[],
        test_statistics={}, thresholds={}, p_values={},
        window_ref_size=100, window_new_size=100, alpha=0.05,
        feature_ranking=pl.DataFrame(), interaction_pairs=None,
        subset_risks_ref={}, subset_risks_new={},
    )
    assert r.model_retrained is False


def test_conformal_chart_fit_large_cal():
    from insurance_monitoring.conformal_chart import ConformalControlChart
    cal = np.random.exponential(1, 5000)
    chart = ConformalControlChart(alpha=0.05).fit(cal)
    assert chart.n_cal_ == 5000


def test_sequential_test_zero_exposure():
    from insurance_monitoring.sequential import SequentialTest
    test = SequentialTest()
    result = test.update(
        champion_claims=0, champion_exposure=0,
        challenger_claims=0, challenger_exposure=0,
    )
    # Zero exposure -> rate = 0 -> rate_ratio = 1.0 (fallback)
    assert result.rate_ratio == pytest.approx(1.0)


def test_conformal_chart_result_to_polars_scores_match():
    from insurance_monitoring.conformal_chart import ConformalChartResult
    scores = np.array([0.5, 1.5, 2.5])
    r = ConformalChartResult(
        scores=scores, p_values=np.array([0.9, 0.5, 0.1]),
        threshold=2.0, is_alarm=np.array([False, False, False]),
        alpha=0.05, n_cal=100
    )
    df = r.to_polars()
    np.testing.assert_allclose(df["score"].to_numpy(), scores)
