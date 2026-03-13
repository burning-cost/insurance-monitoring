"""Tests for rectification methods (absorbed from insurance-calibration)."""

import numpy as np
import pytest

from insurance_monitoring.calibration import (
    rectify_balance,
    isotonic_recalibrate,
    poisson_deviance,
)


def _make_data(n=1000, seed=42, scale=1.0):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat_true = rng.gamma(2, 0.05, n)
    counts = rng.poisson(exposure * y_hat_true)
    y = counts / exposure
    y_hat = y_hat_true * scale
    return y, y_hat, exposure


class TestRectifyBalance:
    def test_multiplicative_restores_balance(self):
        """After multiplicative rectification, sum(v*y) should equal sum(v*y_hat)."""
        y, y_hat, exp = _make_data(scale=1.3)
        y_hat_fixed = rectify_balance(y_hat, y, exp, method="multiplicative")
        obs = np.sum(exp * y)
        pred = np.sum(exp * y_hat_fixed)
        assert pred == pytest.approx(obs, rel=1e-8)

    def test_multiplicative_preserves_order(self):
        """Multiplicative correction should not change the ranking."""
        y, y_hat, exp = _make_data(scale=1.2)
        y_hat_fixed = rectify_balance(y_hat, y, exp, method="multiplicative")
        assert np.all(np.argsort(y_hat) == np.argsort(y_hat_fixed))

    def test_multiplicative_is_scalar_multiple(self):
        """Multiplicative correction multiplies by a single constant alpha."""
        y, y_hat, exp = _make_data(scale=1.5)
        y_hat_fixed = rectify_balance(y_hat, y, exp, method="multiplicative")
        ratios = y_hat_fixed / y_hat
        assert np.std(ratios) < 1e-10  # all ratios identical

    def test_affine_reduces_deviance(self):
        """Affine rectification should not increase deviance."""
        y, y_hat, exp = _make_data(scale=1.4)
        y_hat_affine = rectify_balance(y_hat, y, exp, method="affine", distribution="poisson")
        dev_before = poisson_deviance(y, y_hat, exp)
        dev_after = poisson_deviance(y, y_hat_affine, exp)
        assert dev_after <= dev_before + 1e-6

    def test_affine_nearly_restores_balance(self):
        """After affine rectification, global balance should be near 1.0."""
        y, y_hat, exp = _make_data(scale=1.4, n=2000)
        y_hat_fixed = rectify_balance(y_hat, y, exp, method="affine", distribution="poisson")
        alpha = np.sum(exp * y) / np.sum(exp * y_hat_fixed)
        assert abs(alpha - 1.0) < 0.01

    def test_affine_result_positive(self):
        """All predictions should remain positive after affine rectification."""
        y, y_hat, exp = _make_data(scale=1.3)
        y_hat_fixed = rectify_balance(y_hat, y, exp, method="affine")
        assert np.all(y_hat_fixed > 0)

    def test_unknown_method_raises(self):
        y, y_hat, exp = _make_data()
        with pytest.raises(ValueError, match="Unknown rectification method"):
            rectify_balance(y_hat, y, exp, method="isotonic")

    def test_no_exposure(self):
        y, y_hat, _ = _make_data()
        y_hat_fixed = rectify_balance(y_hat, y, exposure=None, method="multiplicative")
        obs = np.sum(y)
        pred = np.sum(y_hat_fixed)
        assert pred == pytest.approx(obs, rel=1e-8)


class TestIsotonicRecalibrate:
    def test_returns_array(self):
        y, y_hat, exp = _make_data()
        result = isotonic_recalibrate(y, y_hat, exp)
        assert isinstance(result, np.ndarray)
        assert result.shape == y_hat.shape

    def test_preserves_ordering(self):
        """Isotonic recalibration must preserve the ordering of y_hat."""
        y, y_hat, exp = _make_data(n=500)
        result = isotonic_recalibrate(y, y_hat, exp)
        sort_idx = np.argsort(y_hat)
        result_sorted = result[sort_idx]
        # Should be non-decreasing
        assert np.all(np.diff(result_sorted) >= -1e-10)

    def test_all_positive(self):
        """Isotonic result should be non-negative (for frequency data)."""
        y, y_hat, exp = _make_data()
        result = isotonic_recalibrate(y, y_hat, exp)
        assert np.all(result >= 0)

    def test_reduces_deviance_vs_original(self):
        """Isotonic fit should not be worse than the original predictions."""
        y, y_hat, exp = _make_data(scale=1.3, n=2000, seed=0)
        y_hat_rc = isotonic_recalibrate(y, y_hat, exp)
        # Clip to avoid log(0)
        y_hat_rc_clipped = np.maximum(y_hat_rc, 1e-10)
        dev_original = poisson_deviance(y, y_hat, exp)
        dev_isotonic = poisson_deviance(y, y_hat_rc_clipped, exp)
        assert dev_isotonic <= dev_original + 1e-6

    def test_perfectly_calibrated_model_low_change(self):
        """On a well-calibrated model, isotonic recalibration should change little."""
        rng = np.random.default_rng(0)
        n = 3000
        exp = np.ones(n)
        y_hat = np.linspace(0.01, 0.2, n)
        y = rng.poisson(y_hat) / 1.0
        y_hat_rc = isotonic_recalibrate(y, y_hat, exp)
        # The L2 distance should be small relative to prediction range
        l2_change = float(np.sqrt(np.mean((y_hat_rc - y_hat) ** 2)))
        assert l2_change < 0.1

    def test_no_exposure(self):
        y, y_hat, _ = _make_data(n=200)
        result = isotonic_recalibrate(y, y_hat, exposure=None)
        assert result.shape == y_hat.shape
        assert np.all(np.diff(result[np.argsort(y_hat)]) >= -1e-10)

    def test_shape_preserved(self):
        y, y_hat, exp = _make_data(n=300)
        result = isotonic_recalibrate(y, y_hat, exp)
        assert result.shape == (300,)
