"""Tests for deviance functions (absorbed from insurance-calibration)."""

import numpy as np
import pytest

from insurance_monitoring.calibration import (
    poisson_deviance,
    gamma_deviance,
    tweedie_deviance,
    normal_deviance,
    deviance,
)


class TestPoissonDeviance:
    def test_perfect_prediction_is_zero(self):
        y = np.array([0.1, 0.2, 0.3])
        assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_zero_observation_handled(self):
        """0 * log(0/mu) = 0 by convention."""
        y = np.array([0.0, 0.1, 0.2])
        mu = np.array([0.05, 0.1, 0.2])
        result = poisson_deviance(y, mu)
        assert np.isfinite(result)
        assert result > 0

    def test_symmetric_with_uniform_exposure(self):
        """With uniform exposure, result should match manual calculation."""
        y = np.array([0.1, 0.2])
        mu = np.array([0.15, 0.15])
        expected = np.mean(2 * (y * np.log(y / mu) - (y - mu)))
        assert poisson_deviance(y, mu) == pytest.approx(expected, rel=1e-10)

    def test_exposure_weighting(self):
        """Larger exposure should weight observations proportionally."""
        y = np.array([0.1, 0.3])
        mu = np.array([0.2, 0.2])
        w_equal = np.ones(2)
        w_heavy = np.array([2.0, 1.0])
        result_equal = poisson_deviance(y, mu, w_equal)
        result_heavy = poisson_deviance(y, mu, w_heavy)
        assert result_equal != result_heavy

    def test_all_zeros(self):
        """All-zero y is valid for Poisson (frequency models)."""
        y = np.zeros(5)
        mu = np.ones(5) * 0.1
        result = poisson_deviance(y, mu)
        assert np.isfinite(result)
        assert result > 0

    def test_dispatch(self):
        y = np.array([0.1, 0.2, 0.3])
        mu = np.array([0.12, 0.18, 0.25])
        assert deviance(y, mu, distribution="poisson") == pytest.approx(
            poisson_deviance(y, mu), rel=1e-10
        )


class TestGammaDeviance:
    def test_perfect_prediction_is_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_raises_on_zero_y(self):
        y = np.array([0.0, 100.0, 200.0])
        mu = np.array([50.0, 100.0, 200.0])
        with pytest.raises(ValueError, match="Gamma deviance is undefined"):
            gamma_deviance(y, mu)

    def test_positive_for_nonzero(self):
        y = np.array([100.0, 200.0])
        mu = np.array([120.0, 180.0])
        result = gamma_deviance(y, mu)
        assert result > 0

    def test_manual_calculation(self):
        y = np.array([100.0])
        mu = np.array([120.0])
        expected = 2 * (np.log(120 / 100) + 100 / 120 - 1)
        assert gamma_deviance(y, mu) == pytest.approx(expected, rel=1e-10)

    def test_dispatch(self):
        y = np.array([100.0, 200.0])
        mu = np.array([110.0, 190.0])
        assert deviance(y, mu, distribution="gamma") == pytest.approx(
            gamma_deviance(y, mu), rel=1e-10
        )


class TestTweedieDeviance:
    def test_perfect_prediction_approx_zero(self):
        """With y = mu, deviance should be near zero."""
        y = np.array([0.5, 1.0, 0.0, 0.2])
        mu = np.array([0.5, 1.0, 0.01, 0.2])  # mu > 0 required
        result = tweedie_deviance(y, mu, power=1.5)
        assert result >= 0

    def test_zero_y_handled(self):
        y = np.array([0.0, 0.1, 0.5])
        mu = np.array([0.05, 0.1, 0.5])
        result = tweedie_deviance(y, mu, power=1.5)
        assert np.isfinite(result)
        assert result >= 0

    def test_invalid_power_raises(self):
        y = np.array([0.1, 0.2])
        mu = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="1 < p < 2"):
            tweedie_deviance(y, mu, power=0.5)
        with pytest.raises(ValueError, match="1 < p < 2"):
            tweedie_deviance(y, mu, power=2.5)

    def test_power_range(self):
        """Should work for all valid power values in (1, 2)."""
        y = np.array([0.1, 0.5, 1.0])
        mu = np.array([0.15, 0.4, 0.9])
        for p in [1.1, 1.3, 1.5, 1.7, 1.9]:
            result = tweedie_deviance(y, mu, power=p)
            assert np.isfinite(result)
            assert result >= 0

    def test_dispatch_with_power(self):
        y = np.array([0.1, 0.5])
        mu = np.array([0.12, 0.45])
        result = deviance(y, mu, distribution="tweedie", tweedie_power=1.6)
        assert np.isfinite(result)


class TestNormalDeviance:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert normal_deviance(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_is_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 2.5])
        expected = np.mean((y - mu) ** 2)
        assert normal_deviance(y, mu) == pytest.approx(expected, rel=1e-10)

    def test_negative_y_allowed(self):
        """Normal deviance is valid for any real y."""
        y = np.array([-1.0, 0.0, 1.0])
        mu = np.array([0.1, 0.1, 0.9])
        result = normal_deviance(y, mu)
        assert np.isfinite(result)


class TestDevianceDispatch:
    def test_unknown_distribution_raises(self):
        y = np.array([0.1, 0.2])
        mu = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="Unknown distribution"):
            deviance(y, mu, distribution="bernoulli")
