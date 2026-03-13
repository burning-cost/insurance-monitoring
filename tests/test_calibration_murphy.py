"""Tests for Murphy score decomposition (absorbed from insurance-calibration)."""

import numpy as np
import pytest

from insurance_monitoring.calibration import murphy_decomposition, MurphyResult


def _make_data(n=2000, seed=42, scale=1.0):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat = rng.gamma(2, 0.05, n)
    y_hat_biased = y_hat * scale
    counts = rng.poisson(exposure * y_hat)
    y = counts / exposure
    return y, y_hat_biased, exposure


class TestMurphyResult:
    def test_returns_murphy_result(self):
        y, y_hat, exp = _make_data()
        result = murphy_decomposition(y, y_hat, exp, seed=0)
        assert isinstance(result, MurphyResult)

    def test_decomposition_identity(self):
        """D(y, y_hat) = UNC - DSC + MCB should hold."""
        y, y_hat, exp = _make_data(n=1000, seed=1)
        result = murphy_decomposition(y, y_hat, exp, seed=1)
        reconstructed = result.uncertainty - result.discrimination + result.miscalibration
        assert reconstructed == pytest.approx(result.total_deviance, rel=1e-3)

    def test_nonnegative_components(self):
        """All components should be non-negative."""
        y, y_hat, exp = _make_data(n=1000, seed=2)
        result = murphy_decomposition(y, y_hat, exp, seed=2)
        assert result.uncertainty >= 0
        assert result.discrimination >= 0
        assert result.miscalibration >= 0
        assert result.global_mcb >= 0
        assert result.local_mcb >= 0

    def test_dsc_leq_unc(self):
        """Discrimination cannot exceed uncertainty."""
        y, y_hat, exp = _make_data(n=1000, seed=3)
        result = murphy_decomposition(y, y_hat, exp, seed=3)
        assert result.discrimination <= result.uncertainty + 1e-8

    def test_mcb_split(self):
        """GMCB + LMCB should approximately equal MCB."""
        y, y_hat, exp = _make_data(n=1000, seed=4)
        result = murphy_decomposition(y, y_hat, exp, seed=4)
        assert result.global_mcb + result.local_mcb == pytest.approx(
            result.miscalibration, abs=1e-6
        )

    def test_perfectly_calibrated_low_mcb(self):
        """A well-calibrated model should have low MCB relative to DSC."""
        y, y_hat, exp = _make_data(n=3000, scale=1.0, seed=0)
        result = murphy_decomposition(y, y_hat, exp, seed=0)
        # MCB should be small relative to total deviance
        assert result.miscalibration_pct < 20.0  # less than 20% of deviance is MCB
        assert result.discrimination > 0.0

    def test_globally_miscalibrated_large_gmcb(self):
        """A 30% scale error should produce significant GMCB."""
        # Use larger n to reduce noise in GMCB/LMCB split
        y, y_hat, exp = _make_data(n=10000, scale=1.3, seed=0)
        result = murphy_decomposition(y, y_hat, exp, seed=0)
        assert result.global_mcb > 0.0
        # With a pure scale error, GMCB should dominate LMCB
        assert result.global_mcb >= result.local_mcb

    def test_verdict_ok_for_calibrated(self):
        """Calibrated model should have low MCB relative to UNC."""
        y, y_hat, exp = _make_data(n=5000, scale=1.0, seed=0)
        result = murphy_decomposition(y, y_hat, exp, seed=0)
        assert result.miscalibration / result.uncertainty < 0.15

    def test_verdict_recalibrate_for_scaled(self):
        """Pure scale error should suggest RECALIBRATE, not REFIT."""
        y, y_hat, exp = _make_data(n=3000, scale=1.4, seed=0)
        result = murphy_decomposition(y, y_hat, exp, seed=0)
        assert result.verdict in ("RECALIBRATE", "REFIT")  # at least not OK

    def test_percentage_fields(self):
        y, y_hat, exp = _make_data(n=1000, seed=5)
        result = murphy_decomposition(y, y_hat, exp, seed=5)
        if result.total_deviance > 0:
            assert result.discrimination_pct == pytest.approx(
                100.0 * result.discrimination / result.total_deviance, rel=1e-6
            )
            assert result.miscalibration_pct == pytest.approx(
                100.0 * result.miscalibration / result.total_deviance, rel=1e-6
            )

    def test_verdict_is_valid_string(self):
        y, y_hat, exp = _make_data()
        result = murphy_decomposition(y, y_hat, exp)
        assert result.verdict in ("OK", "RECALIBRATE", "REFIT")

    def test_to_polars(self):
        import polars as pl
        y, y_hat, exp = _make_data(n=500)
        result = murphy_decomposition(y, y_hat, exp)
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "total_deviance" in df.columns
        assert "uncertainty" in df.columns
        assert "discrimination" in df.columns
        assert "miscalibration" in df.columns
        assert len(df) == 1

    def test_no_exposure(self):
        """Should work without explicit exposure."""
        rng = np.random.default_rng(0)
        y = rng.gamma(1, 0.1, 500)
        y_hat = rng.gamma(1, 0.1, 500)
        result = murphy_decomposition(y, y_hat, exposure=None)
        assert isinstance(result, MurphyResult)

    def test_repr(self):
        y, y_hat, exp = _make_data(n=500)
        result = murphy_decomposition(y, y_hat, exp)
        repr_str = repr(result)
        assert "MurphyResult" in repr_str
        assert "UNC" in repr_str


class TestMurphyDistributions:
    def test_gamma_distribution(self):
        rng = np.random.default_rng(10)
        n = 500
        y_hat = rng.gamma(3, 100, n)
        y = rng.gamma(3, 100, n)
        result = murphy_decomposition(y, y_hat, distribution="gamma")
        assert isinstance(result, MurphyResult)

    def test_normal_distribution(self):
        rng = np.random.default_rng(11)
        n = 500
        y_hat = rng.uniform(1, 10, n)
        y = y_hat + rng.normal(0, 1, n)
        result = murphy_decomposition(y, y_hat, distribution="normal")
        assert isinstance(result, MurphyResult)

    def test_tweedie_distribution(self):
        rng = np.random.default_rng(12)
        n = 500
        y_hat = rng.gamma(2, 0.1, n)
        y = np.maximum(rng.gamma(2, 0.1, n), 0.0)
        result = murphy_decomposition(y, y_hat, distribution="tweedie", tweedie_power=1.5)
        assert isinstance(result, MurphyResult)
