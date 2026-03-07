"""Tests for insurance_monitoring.calibration module."""

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.calibration import (
    ae_ratio,
    ae_ratio_ci,
    calibration_curve,
    hosmer_lemeshow,
)


# ---------------------------------------------------------------------------
# A/E ratio tests
# ---------------------------------------------------------------------------


class TestAERatio:
    """A/E ratio correctness and segmentation."""

    def test_perfect_calibration(self):
        """When actual == expected, A/E should be 1.0."""
        actual = np.array([10.0, 20.0, 15.0, 5.0])
        predicted = np.array([10.0, 20.0, 15.0, 5.0])
        result = ae_ratio(actual, predicted)
        assert result == pytest.approx(1.0)

    def test_optimistic_model(self):
        """When predictions are too high (optimistic), A/E < 1.0."""
        actual = np.array([100.0])
        predicted = np.array([110.0])  # 10% over-predicted
        result = ae_ratio(actual, predicted)
        assert result == pytest.approx(100.0 / 110.0)

    def test_pessimistic_model(self):
        """When predictions are too low (pessimistic), A/E > 1.0."""
        actual = np.array([110.0])
        predicted = np.array([100.0])
        result = ae_ratio(actual, predicted)
        assert result == pytest.approx(1.1)

    def test_with_exposure(self):
        """A/E with exposure: expected = sum(predicted * exposure)."""
        actual = np.array([12.0, 8.0])      # claim counts
        predicted = np.array([0.10, 0.08])  # predicted frequency
        exposure = np.array([100.0, 100.0]) # car-years
        # expected = 0.10 * 100 + 0.08 * 100 = 10 + 8 = 18
        # A/E = (12 + 8) / 18 = 20 / 18
        result = ae_ratio(actual, predicted, exposure=exposure)
        assert result == pytest.approx(20.0 / 18.0)

    def test_segmented_returns_polars_dataframe(self):
        """Segmented A/E should return a Polars DataFrame."""
        actual = np.array([10, 5, 20, 3])
        predicted = np.array([10, 5, 18, 3])
        segments = np.array(["young", "young", "mature", "mature"])
        result = ae_ratio(actual, predicted, segments=segments)
        assert isinstance(result, pl.DataFrame)
        assert "segment" in result.columns
        assert "ae_ratio" in result.columns

    def test_segmented_correct_values(self):
        """Segmented A/E should compute correct per-segment ratios."""
        actual = np.array([10.0, 10.0, 20.0, 20.0])
        predicted = np.array([10.0, 10.0, 10.0, 10.0])
        segments = np.array(["A", "A", "B", "B"])
        result = ae_ratio(actual, predicted, segments=segments)
        seg_a = result.filter(pl.col("segment") == "A")["ae_ratio"][0]
        seg_b = result.filter(pl.col("segment") == "B")["ae_ratio"][0]
        assert seg_a == pytest.approx(1.0)
        assert seg_b == pytest.approx(2.0)

    def test_empty_actual_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            ae_ratio(np.array([]), np.array([]))

    def test_mismatched_lengths_raises(self):
        """Mismatched actual/predicted lengths should raise ValueError."""
        with pytest.raises(ValueError):
            ae_ratio(np.array([1.0, 2.0]), np.array([1.0]))

    def test_zero_expected_raises(self):
        """Zero total expected should raise ValueError."""
        with pytest.raises(ValueError, match="zero"):
            ae_ratio(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

    def test_polars_series_input(self):
        """ae_ratio should accept Polars Series."""
        actual = pl.Series([10.0, 20.0, 15.0])
        predicted = pl.Series([10.0, 20.0, 15.0])
        result = ae_ratio(actual, predicted)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A/E ratio with confidence intervals
# ---------------------------------------------------------------------------


class TestAERatioCI:
    """A/E confidence interval tests."""

    def test_poisson_ci_contains_true_ae(self):
        """Poisson CI should contain the true A/E ratio at 95% coverage."""
        # Set up a model that is exactly calibrated (A/E = 1.0 by design)
        rng = np.random.default_rng(42)
        exposure = np.ones(5_000)
        predicted = rng.uniform(0.05, 0.15, 5_000)
        # Generate actual from the predicted rates (so true A/E = 1.0)
        actual = rng.poisson(predicted * exposure)

        result = ae_ratio_ci(actual, predicted, exposure=exposure)
        assert result["lower"] <= result["ae"] <= result["upper"]
        # True A/E is approximately 1.0 — should be within CI
        assert result["lower"] <= 1.0 <= result["upper"]

    def test_ci_lower_lt_upper(self):
        """Lower CI bound should always be below upper bound."""
        actual = np.array([50, 60, 40, 55])
        predicted = np.array([0.10, 0.12, 0.09, 0.11])
        exposure = np.ones(4) * 500
        result = ae_ratio_ci(actual, predicted, exposure=exposure)
        assert result["lower"] < result["upper"]

    def test_wider_ci_with_alpha_01(self):
        """90% CI should be narrower than 99% CI."""
        actual = np.array([100])
        predicted = np.array([100.0])
        result_90 = ae_ratio_ci(actual, predicted, alpha=0.10)
        result_99 = ae_ratio_ci(actual, predicted, alpha=0.01)
        width_90 = result_90["upper"] - result_90["lower"]
        width_99 = result_99["upper"] - result_99["lower"]
        assert width_90 < width_99

    def test_normal_method(self):
        """Normal approximation method should return valid dict."""
        rng = np.random.default_rng(43)
        actual = rng.normal(100, 10, 1_000)
        predicted = rng.normal(100, 10, 1_000)
        result = ae_ratio_ci(actual, predicted, method="normal")
        assert "ae" in result
        assert result["lower"] < result["upper"]

    def test_invalid_method_raises(self):
        """Invalid method string should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            ae_ratio_ci(np.array([1.0]), np.array([1.0]), method="invalid")

    def test_returns_n_claims(self):
        """Result dict should include n_claims."""
        actual = np.array([5, 3, 7, 2])
        predicted = np.ones(4) * 4.25
        result = ae_ratio_ci(actual, predicted)
        assert result["n_claims"] == pytest.approx(17.0)


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    """Calibration curve binning tests."""

    def test_returns_polars_dataframe(self):
        """calibration_curve should return a Polars DataFrame."""
        rng = np.random.default_rng(50)
        predicted = rng.beta(2, 5, 500)
        actual = rng.binomial(1, predicted)
        result = calibration_curve(actual, predicted)
        assert isinstance(result, pl.DataFrame)

    def test_column_names(self):
        """Should have the expected columns."""
        rng = np.random.default_rng(51)
        p = rng.beta(2, 5, 500)
        y = rng.binomial(1, p)
        result = calibration_curve(y, p)
        for col in ["bin", "mean_predicted", "mean_actual", "n", "ae_ratio"]:
            assert col in result.columns

    def test_perfect_calibration_curve(self):
        """Mean predicted == mean actual in bins for well-calibrated model."""
        # With a very large sample and predictions = actuals, AE should be ~1.0 per bin
        rng = np.random.default_rng(52)
        n = 50_000
        predicted = rng.uniform(0.05, 0.20, n)
        # Actually draw from Poisson to get noisy actuals
        actual = rng.poisson(predicted)
        result = calibration_curve(actual, predicted, n_bins=5)
        # All A/E ratios should be close to 1.0
        for ae in result["ae_ratio"].to_list():
            assert 0.90 < ae < 1.10, f"Expected A/E near 1.0, got {ae}"

    def test_uniform_strategy(self):
        """uniform strategy should also work."""
        rng = np.random.default_rng(53)
        p = rng.beta(2, 5, 1_000)
        y = rng.binomial(1, p)
        result = calibration_curve(y, p, strategy="uniform")
        assert len(result) > 0

    def test_invalid_strategy_raises(self):
        """Invalid strategy should raise ValueError."""
        with pytest.raises(ValueError, match="strategy"):
            calibration_curve(np.ones(10), np.ones(10), strategy="bad")


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow test
# ---------------------------------------------------------------------------


class TestHosmerLemeshow:
    """Hosmer-Lemeshow goodness-of-fit tests."""

    def test_well_calibrated_model_large_pvalue(self):
        """Well-calibrated model should have large HL p-value."""
        rng = np.random.default_rng(60)
        p = rng.beta(2, 5, 2_000)
        y = rng.binomial(1, p)
        result = hosmer_lemeshow(y, p)
        # Good calibration → p-value should not be tiny
        assert result["p_value"] > 0.001

    def test_miscalibrated_model_small_pvalue(self):
        """Model with systematic bias should fail HL test."""
        rng = np.random.default_rng(61)
        n = 5_000
        # True probs are high but we predict low (systematic underestimation)
        p_true = rng.beta(5, 2, n)
        p_pred = rng.beta(2, 5, n)  # wrong distribution
        y = rng.binomial(1, p_true)
        result = hosmer_lemeshow(y, p_pred)
        assert result["p_value"] < 0.05

    def test_returns_correct_keys(self):
        """HL result should have statistic, p_value, df."""
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        p = np.array([0.2, 0.7, 0.3, 0.8, 0.6, 0.1, 0.9, 0.4, 0.2, 0.7])
        result = hosmer_lemeshow(y, p, n_bins=5)
        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result

    def test_statistic_nonnegative(self):
        """HL statistic should be non-negative."""
        rng = np.random.default_rng(62)
        p = rng.beta(2, 5, 500)
        y = rng.binomial(1, p)
        result = hosmer_lemeshow(y, p)
        assert result["statistic"] >= 0

    def test_mismatched_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            hosmer_lemeshow(np.array([0, 1, 0]), np.array([0.2, 0.7]))
