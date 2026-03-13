"""Tests for the balance property test (absorbed from insurance-calibration)."""

import numpy as np
import pytest

from insurance_monitoring.calibration import check_balance, BalanceResult


def _make_calibrated(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat = rng.gamma(2, 0.05, n)
    counts = rng.poisson(exposure * y_hat)
    y = counts / exposure
    return y, y_hat, exposure


def _make_miscalibrated(scale=1.2, n=1000, seed=42):
    y, y_hat, exposure = _make_calibrated(n=n, seed=seed)
    y_hat_biased = y_hat * scale  # systematic over-prediction
    return y, y_hat_biased, exposure


class TestBalanceResult:
    def test_returns_balance_result(self):
        y, y_hat, exp = _make_calibrated()
        result = check_balance(y, y_hat, exp, seed=0)
        assert isinstance(result, BalanceResult)

    def test_calibrated_model_is_balanced(self):
        """A perfectly calibrated model should have alpha near 1.0."""
        y, y_hat, exp = _make_calibrated(n=5000, seed=0)
        result = check_balance(y, y_hat, exp, seed=0, bootstrap_n=499)
        assert result.is_balanced
        assert abs(result.balance_ratio - 1.0) < 0.05

    def test_miscalibrated_model_is_detected(self):
        """20% scale error should be detected with 1000 policies."""
        y, y_hat, exp = _make_miscalibrated(scale=1.2, n=2000, seed=0)
        result = check_balance(y, y_hat, exp, seed=0, bootstrap_n=499)
        assert not result.is_balanced
        # Balance ratio should be ~1/1.2 ≈ 0.833 (actuals < predicted)
        assert result.balance_ratio < 0.95

    def test_over_prediction_ratio_below_one(self):
        """Over-prediction (y_hat > y on average) gives alpha < 1."""
        y, y_hat, exp = _make_miscalibrated(scale=1.5, n=2000, seed=1)
        result = check_balance(y, y_hat, exp, seed=1, bootstrap_n=499)
        assert result.balance_ratio < 1.0

    def test_under_prediction_ratio_above_one(self):
        """Under-prediction (y_hat < y on average) gives alpha > 1."""
        y, y_hat, exp = _make_miscalibrated(scale=0.7, n=2000, seed=1)
        result = check_balance(y, y_hat, exp, seed=1, bootstrap_n=499)
        assert result.balance_ratio > 1.0

    def test_ci_contains_true_ratio(self):
        """95% CI should contain the true balance ratio most of the time."""
        y, y_hat, exp = _make_calibrated(n=1000, seed=99)
        result = check_balance(y, y_hat, exp, seed=99, bootstrap_n=999)
        assert result.ci_lower < result.balance_ratio < result.ci_upper

    def test_ci_ordering(self):
        y, y_hat, exp = _make_calibrated()
        result = check_balance(y, y_hat, exp, seed=0)
        assert result.ci_lower <= result.ci_upper

    def test_totals(self):
        y, y_hat, exp = _make_calibrated(n=100)
        result = check_balance(y, y_hat, exp, seed=0)
        expected_obs = float(np.sum(exp * y))
        expected_pred = float(np.sum(exp * y_hat))
        assert result.observed_total == pytest.approx(expected_obs, rel=1e-6)
        assert result.predicted_total == pytest.approx(expected_pred, rel=1e-6)

    def test_n_policies(self):
        y, y_hat, exp = _make_calibrated(n=500)
        result = check_balance(y, y_hat, exp)
        assert result.n_policies == 500

    def test_total_exposure(self):
        y, y_hat, exp = _make_calibrated(n=100)
        result = check_balance(y, y_hat, exp)
        assert result.total_exposure == pytest.approx(float(np.sum(exp)), rel=1e-6)

    def test_no_exposure_defaults_to_uniform(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(1, 0.1, 200)
        y_hat = rng.gamma(1, 0.1, 200)
        result = check_balance(y, y_hat, exposure=None, seed=0)
        assert result.total_exposure == pytest.approx(200.0, rel=1e-6)
        assert result.n_policies == 200

    def test_to_polars(self):
        import polars as pl
        y, y_hat, exp = _make_calibrated(n=100)
        result = check_balance(y, y_hat, exp, seed=0)
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "balance_ratio" in df.columns
        assert len(df) == 1

    def test_p_value_range(self):
        y, y_hat, exp = _make_calibrated()
        result = check_balance(y, y_hat, exp, seed=0)
        assert 0.0 <= result.p_value <= 1.0

    def test_seed_reproducibility(self):
        y, y_hat, exp = _make_calibrated()
        r1 = check_balance(y, y_hat, exp, seed=42)
        r2 = check_balance(y, y_hat, exp, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


class TestBalanceValidation:
    def test_mismatched_lengths_raise(self):
        y = np.array([0.1, 0.2, 0.3])
        y_hat = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="same length"):
            check_balance(y, y_hat)

    def test_negative_exposure_raises(self):
        y = np.array([0.1, 0.2])
        y_hat = np.array([0.1, 0.2])
        exp = np.array([-1.0, 1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            check_balance(y, y_hat, exp)

    def test_zero_exposure_raises(self):
        y = np.array([0.1, 0.2])
        y_hat = np.array([0.1, 0.2])
        exp = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            check_balance(y, y_hat, exp)

    def test_zero_y_hat_raises(self):
        y = np.array([0.1, 0.2])
        y_hat = np.array([0.0, 0.2])
        with pytest.raises(ValueError, match="y_hat <= 0"):
            check_balance(y, y_hat)
