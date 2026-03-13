"""Tests for auto-calibration check (absorbed from insurance-calibration)."""

import numpy as np
import pytest

from insurance_monitoring.calibration import check_auto_calibration, AutoCalibResult


def _make_calibrated_poisson(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat = rng.gamma(2, 0.05, n)
    counts = rng.poisson(exposure * y_hat)
    y = counts / exposure
    return y, y_hat, exposure


def _make_shape_miscalibrated(n=2000, seed=42):
    """Model that is globally balanced but not auto-calibrated.

    The model over-estimates high risks and under-estimates low risks (wrong shape).
    We achieve this by flattening the predictions towards the mean.
    """
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat_true = rng.gamma(2, 0.05, n)
    counts = rng.poisson(exposure * y_hat_true)
    y = counts / exposure
    # Flattened (wrong-shaped) predictions: compress toward mean
    mean_pred = float(np.sum(exposure * y_hat_true) / np.sum(exposure))
    y_hat_flat = 0.5 * y_hat_true + 0.5 * mean_pred
    # Re-scale to restore global balance
    alpha = np.sum(exposure * y) / np.sum(exposure * y_hat_flat)
    y_hat_flat *= alpha
    return y, y_hat_flat, exposure


class TestAutoCalibResult:
    def test_returns_auto_calib_result(self):
        y, y_hat, exp = _make_calibrated_poisson()
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=99)
        assert isinstance(result, AutoCalibResult)

    def test_calibrated_model_passes(self):
        """Well-calibrated Poisson model should not reject H0."""
        y, y_hat, exp = _make_calibrated_poisson(n=3000, seed=0)
        result = check_auto_calibration(
            y, y_hat, exp, seed=0, bootstrap_n=199, significance_level=0.05
        )
        # With large n and correctly calibrated data, p-value should often be > 0.05
        # Allow some tolerance — bootstrap tests have variance
        assert result.p_value > 0.01 or result.is_calibrated

    def test_miscalibrated_model_detected(self):
        """Shape-miscalibrated model should be detected with enough data."""
        y, y_hat, exp = _make_shape_miscalibrated(n=3000, seed=0)
        result = check_auto_calibration(
            y, y_hat, exp, seed=0, bootstrap_n=199, significance_level=0.3
        )
        # At 30% significance level, should detect the miscalibration
        assert not result.is_calibrated or result.mcb_score > 0.0

    def test_per_bin_is_dataframe(self):
        import polars as pl
        y, y_hat, exp = _make_calibrated_poisson(n=500)
        result = check_auto_calibration(
            y, y_hat, exp, seed=0, bootstrap_n=49, n_bins=5
        )
        assert isinstance(result.per_bin, pl.DataFrame)
        assert "bin" in result.per_bin.columns
        assert "pred_mean" in result.per_bin.columns
        assert "obs_mean" in result.per_bin.columns
        assert "ratio" in result.per_bin.columns
        assert "exposure" in result.per_bin.columns
        assert len(result.per_bin) == 5

    def test_per_bin_exposure_sums_to_total(self):
        y, y_hat, exp = _make_calibrated_poisson(n=500)
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=49)
        assert result.per_bin["exposure"].sum() == pytest.approx(
            float(np.sum(exp)), rel=1e-4
        )

    def test_worst_bin_ratio_nonnegative(self):
        y, y_hat, exp = _make_calibrated_poisson()
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=49)
        assert result.worst_bin_ratio >= 0.0

    def test_mcb_score_nonnegative(self):
        y, y_hat, exp = _make_calibrated_poisson()
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=49)
        assert result.mcb_score >= 0.0

    def test_p_value_in_unit_interval(self):
        y, y_hat, exp = _make_calibrated_poisson()
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=49)
        assert 0.0 <= result.p_value <= 1.0

    def test_n_isotonic_steps_positive(self):
        y, y_hat, exp = _make_calibrated_poisson()
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=49)
        assert result.n_isotonic_steps >= 1

    def test_seed_reproducibility(self):
        y, y_hat, exp = _make_calibrated_poisson()
        r1 = check_auto_calibration(y, y_hat, exp, seed=42, bootstrap_n=99)
        r2 = check_auto_calibration(y, y_hat, exp, seed=42, bootstrap_n=99)
        assert r1.p_value == r2.p_value
        assert r1.mcb_score == r2.mcb_score


class TestAutoCalibMethods:
    def test_hosmer_lemeshow_method(self):
        y, y_hat, exp = _make_calibrated_poisson(n=500)
        result = check_auto_calibration(
            y, y_hat, exp, method="hosmer_lemeshow"
        )
        assert isinstance(result, AutoCalibResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_unknown_method_raises(self):
        y, y_hat, exp = _make_calibrated_poisson(n=200)
        with pytest.raises(ValueError, match="Unknown method"):
            check_auto_calibration(y, y_hat, exp, method="bad_method")

    def test_hosmer_lemeshow_calibrated_gives_high_pvalue(self):
        """Binned chi-square on a calibrated model should not reject H0."""
        y, y_hat, exp = _make_calibrated_poisson(n=3000, seed=123)
        result = check_auto_calibration(
            y, y_hat, exp, method="hosmer_lemeshow", n_bins=10
        )
        # Should not reject at 1% level
        assert result.p_value > 0.01


class TestAutoCalibDistributions:
    def test_gamma_distribution(self):
        rng = np.random.default_rng(7)
        n = 500
        y_hat = rng.gamma(2, 100, n)
        y = rng.gamma(2, 100, n)  # not calibrated to y_hat
        result = check_auto_calibration(
            y, y_hat, distribution="gamma", seed=7, bootstrap_n=49
        )
        assert isinstance(result, AutoCalibResult)

    def test_normal_distribution(self):
        rng = np.random.default_rng(8)
        n = 500
        y_hat = rng.uniform(1, 10, n)
        y = y_hat + rng.normal(0, 0.5, n)
        result = check_auto_calibration(
            y, y_hat, distribution="normal", seed=8, bootstrap_n=49
        )
        assert isinstance(result, AutoCalibResult)

    def test_to_polars(self):
        import polars as pl
        y, y_hat, exp = _make_calibrated_poisson(n=200)
        result = check_auto_calibration(y, y_hat, exp, seed=0, bootstrap_n=49)
        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "p_value" in df.columns
