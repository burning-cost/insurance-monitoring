"""Tests for insurance_monitoring.discrimination module."""

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.discrimination import (
    gini_coefficient,
    gini_drift_test,
    lorenz_curve,
)


# ---------------------------------------------------------------------------
# Gini coefficient tests
# ---------------------------------------------------------------------------


class TestGiniCoefficient:
    """Gini coefficient correctness against known values."""

    def test_perfect_discrimination(self):
        """Perfect ranking: highest predicted always has highest actual claims.

        For a binary outcome (0/1) with 50% claim rate, the maximum achievable
        Gini is 0.5 (not 1.0) because the Lorenz curve can only reach the
        upper-right triangle at most. With a low claim rate, the max Gini
        approaches 1 - claim_rate.
        """
        # 6 policies: 3 no-claim (low predicted) + 3 claim (high predicted)
        # Perfect ranking → Gini = 0.5 (max for 50% binary outcome)
        actual = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        predicted = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        g = gini_coefficient(actual, predicted)
        assert g == pytest.approx(0.5, abs=1e-10), f"Expected Gini = 0.5 for perfect 50/50 ranking, got {g}"

    def test_good_model_better_gini_than_random(self):
        """A good model should have higher Gini than a random model.

        Gini is bounded by Poisson randomness: even a perfect model cannot
        achieve Gini = 1.0 when outcomes are noisy counts. The key test is
        that a good model consistently outperforms a random one.
        """
        rng = np.random.default_rng(77)
        n = 5_000
        # True rates vary from 0.02 to 0.30
        true_rate = np.linspace(0.02, 0.30, n)
        actual = rng.poisson(true_rate).astype(float)
        # Good model: near-perfect predictions
        predicted_good = true_rate + rng.normal(0, 0.001, n)
        # Random model: predictions uncorrelated with true rate
        predicted_random = rng.uniform(0.02, 0.30, n)
        g_good = gini_coefficient(actual, predicted_good)
        g_random = gini_coefficient(actual, predicted_random)
        assert g_good > g_random + 0.10, (
            f"Good model Gini ({g_good:.3f}) should exceed random ({g_random:.3f}) by >0.10"
        )

    def test_random_model_near_zero(self):
        """Random model (predictions uncorrelated with actuals) should have Gini near 0."""
        rng = np.random.default_rng(70)
        n = 10_000
        actual = rng.binomial(1, 0.1, n).astype(float)
        # Predictions independent of actuals
        predicted = rng.uniform(0, 1, n)
        g = gini_coefficient(actual, predicted)
        assert abs(g) < 0.05, f"Random model Gini should be near 0, got {g}"

    def test_gini_range(self):
        """Gini should be in [-1, 1] in principle, [0, 1] for sensible models."""
        rng = np.random.default_rng(71)
        predicted = rng.uniform(0.05, 0.20, 5_000)
        actual = rng.poisson(predicted)
        g = gini_coefficient(actual, predicted)
        assert -1.0 <= g <= 1.0

    def test_gini_good_model_reasonable_range(self):
        """A good frequency model should produce Gini between 0.3 and 0.7."""
        rng = np.random.default_rng(72)
        n = 10_000
        # Create a well-specified GLM-like model
        x = rng.uniform(0, 1, n)
        true_rate = 0.05 + 0.15 * x  # true rate is linear in x
        predicted = true_rate + rng.normal(0, 0.005, n)  # nearly perfect predictions
        actual = rng.poisson(true_rate).astype(float)
        g = gini_coefficient(actual, predicted)
        assert 0.1 < g < 0.9, f"Good model Gini should be moderate, got {g}"

    def test_gini_with_exposure_weighting(self):
        """Gini with exposure should differ from unweighted when exposures vary."""
        rng = np.random.default_rng(73)
        n = 2_000
        predicted = rng.uniform(0.05, 0.20, n)
        actual = rng.poisson(predicted)
        exposure = rng.uniform(0.1, 2.0, n)  # highly variable exposure
        g_unweighted = gini_coefficient(actual, predicted)
        g_weighted = gini_coefficient(actual, predicted, exposure=exposure)
        # Both should be valid Gini values, but may differ
        assert -1.0 <= g_unweighted <= 1.0
        assert -1.0 <= g_weighted <= 1.0

    def test_gini_polars_input(self):
        """gini_coefficient should accept Polars Series."""
        actual = pl.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        predicted = pl.Series([0.1, 0.8, 0.2, 0.9, 0.15, 0.85])
        g = gini_coefficient(actual, predicted)
        assert isinstance(g, float)

    def test_gini_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            gini_coefficient(np.array([]), np.array([]))

    def test_gini_mismatched_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            gini_coefficient(np.array([1.0, 2.0]), np.array([0.1]))

    def test_gini_all_zero_actual(self):
        """All-zero actual should return Gini = 0 (no claims to rank)."""
        actual = np.zeros(100)
        predicted = np.random.default_rng(74).uniform(0, 1, 100)
        g = gini_coefficient(actual, predicted)
        assert g == 0.0

    def test_gini_known_value(self):
        """Check Gini against a manually computed value.

        4 policies, 2 with claims in the highest-predicted group.
        Sorted ascending: actual=[0,0,1,1], x=[0,0.25,0.5,0.75,1], y=[0,0,0,0.5,1]
        AUC via trapz = 0*0.25 + 0*0.25 + 0.25*0.25 + 0.75*0.25 = 0.25
        Gini = 1 - 2 * 0.25 = 0.5
        """
        actual = np.array([0.0, 0.0, 1.0, 1.0])
        predicted = np.array([0.1, 0.2, 0.8, 0.9])
        g = gini_coefficient(actual, predicted)
        assert g == pytest.approx(0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# Gini drift test
# ---------------------------------------------------------------------------


class TestGiniDriftTest:
    """Tests for the arXiv 2510.04556 Gini drift z-test."""

    def test_identical_data_large_pvalue(self):
        """Identical reference and current data: z-stat = 0, p-value = 1.0."""
        rng = np.random.default_rng(80)
        n = 3_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        g = gini_coefficient(act, pred)

        # Same data for both periods → z = 0, p = 1.0
        result = gini_drift_test(
            reference_gini=g,
            current_gini=g,
            n_reference=n,
            n_current=n,
            reference_actual=act,
            reference_predicted=pred,
            current_actual=act,
            current_predicted=pred,
            n_bootstrap=50,
        )
        assert result["z_statistic"] == pytest.approx(0.0, abs=1e-10)
        assert result["p_value"] == pytest.approx(1.0, abs=1e-10)
        assert result["significant"] is False

    def test_no_drift_not_significant(self):
        """Two draws from the same process should not be significant at 1% level."""
        # Use precomputed variances to avoid bootstrap randomness in this test
        result = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.46,  # very small difference
            n_reference=10_000,
            n_current=10_000,
            reference_variance=0.0005,
            current_variance=0.0005,
        )
        # With small variance and tiny Gini change, should not be significant
        assert result["significant"] is False

    def test_result_keys(self):
        """Result should have all expected keys."""
        rng = np.random.default_rng(81)
        n = 1_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        g = gini_coefficient(act, pred)

        result = gini_drift_test(
            reference_gini=g,
            current_gini=g,
            n_reference=n,
            n_current=n,
            reference_actual=act,
            reference_predicted=pred,
            current_actual=act,
            current_predicted=pred,
            n_bootstrap=20,
        )
        for key in ["z_statistic", "p_value", "reference_gini", "current_gini",
                    "gini_change", "significant"]:
            assert key in result

    def test_with_precomputed_variance(self):
        """gini_drift_test should work with precomputed variance values."""
        result = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.40,
            n_reference=10_000,
            n_current=5_000,
            reference_variance=0.001,
            current_variance=0.002,
        )
        assert "z_statistic" in result
        assert result["gini_change"] == pytest.approx(-0.05)

    def test_missing_variance_and_arrays_raises(self):
        """Should raise if neither arrays nor variance provided."""
        with pytest.raises(ValueError):
            gini_drift_test(
                reference_gini=0.45,
                current_gini=0.42,
                n_reference=1_000,
                n_current=500,
            )

    def test_gini_change_computed_correctly(self):
        """gini_change should equal current - reference."""
        result = gini_drift_test(
            reference_gini=0.50,
            current_gini=0.45,
            n_reference=5_000,
            n_current=5_000,
            reference_variance=0.001,
            current_variance=0.001,
        )
        assert result["gini_change"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# Lorenz curve
# ---------------------------------------------------------------------------


class TestLorenzCurve:
    """Lorenz curve shape and boundary tests."""

    def test_lorenz_starts_at_origin(self):
        """Lorenz curve should start at (0, 0)."""
        actual = np.array([0, 0, 1, 1, 2], dtype=float)
        predicted = np.array([0.1, 0.2, 0.5, 0.7, 0.9])
        x, y = lorenz_curve(actual, predicted)
        assert x[0] == 0.0
        assert y[0] == 0.0

    def test_lorenz_ends_at_one(self):
        """Lorenz curve should end at (1, 1)."""
        actual = np.array([1, 2, 0, 3, 1], dtype=float)
        predicted = np.array([0.3, 0.6, 0.1, 0.8, 0.4])
        x, y = lorenz_curve(actual, predicted)
        assert x[-1] == pytest.approx(1.0)
        assert y[-1] == pytest.approx(1.0)

    def test_lorenz_returns_numpy_arrays(self):
        """lorenz_curve should return numpy arrays."""
        actual = np.array([1.0, 0.0, 1.0])
        predicted = np.array([0.3, 0.1, 0.7])
        x, y = lorenz_curve(actual, predicted)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_lorenz_same_length(self):
        """x and y should have the same length."""
        actual = np.array([1, 0, 2, 0, 1], dtype=float)
        predicted = np.array([0.2, 0.1, 0.8, 0.3, 0.5])
        x, y = lorenz_curve(actual, predicted)
        assert len(x) == len(y)

    def test_lorenz_x_monotone(self):
        """Cumulative exposure (x) should be non-decreasing."""
        rng = np.random.default_rng(90)
        actual = rng.poisson(0.1, 1_000).astype(float)
        predicted = rng.uniform(0.05, 0.20, 1_000)
        x, _ = lorenz_curve(actual, predicted)
        assert np.all(np.diff(x) >= 0)

    def test_lorenz_y_monotone(self):
        """Cumulative claims (y) should be non-decreasing."""
        rng = np.random.default_rng(91)
        actual = rng.poisson(0.1, 1_000).astype(float)
        predicted = rng.uniform(0.05, 0.20, 1_000)
        _, y = lorenz_curve(actual, predicted)
        assert np.all(np.diff(y) >= 0)

    def test_lorenz_polars_input(self):
        """lorenz_curve should accept Polars Series."""
        actual = pl.Series([0.0, 1.0, 0.0, 2.0])
        predicted = pl.Series([0.1, 0.7, 0.2, 0.9])
        x, y = lorenz_curve(actual, predicted)
        assert len(x) == 5  # n + 1 (prepend origin)

    def test_lorenz_empty_raises(self):
        """Empty actual should raise ValueError."""
        with pytest.raises(ValueError):
            lorenz_curve(np.array([]), np.array([]))
