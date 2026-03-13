"""Tests for CalibrationReport and CalibrationChecker (absorbed from insurance-calibration)."""

import numpy as np
import pytest

from insurance_monitoring.calibration import (
    CalibrationChecker,
    BalanceResult,
    AutoCalibResult,
    MurphyResult,
    CalibrationReport,
)


def _make_data(n=2000, seed=42, scale=1.0):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat = rng.gamma(2, 0.05, n) * scale
    y_hat_true = y_hat / scale
    counts = rng.poisson(exposure * y_hat_true)
    y = counts / exposure
    return y, y_hat, exposure


class TestCalibrationChecker:
    def test_returns_calibration_report(self):
        y, y_hat, exp = _make_data()
        checker = CalibrationChecker(distribution="poisson", bootstrap_n=99)
        report = checker.check(y, y_hat, exp, seed=0)
        assert isinstance(report, CalibrationReport)

    def test_report_has_all_components(self):
        y, y_hat, exp = _make_data()
        checker = CalibrationChecker(bootstrap_n=99)
        report = checker.check(y, y_hat, exp, seed=0)
        assert isinstance(report.balance, BalanceResult)
        assert isinstance(report.auto_calibration, AutoCalibResult)
        assert isinstance(report.murphy, MurphyResult)

    def test_fit_then_check(self):
        y, y_hat, exp = _make_data()
        checker = CalibrationChecker(bootstrap_n=99)
        checker.fit(y, y_hat, exp, seed=0)
        assert checker._is_fitted
        report = checker.check(y, y_hat, exp, seed=0)
        assert isinstance(report, CalibrationReport)

    def test_verdict_is_valid(self):
        y, y_hat, exp = _make_data()
        checker = CalibrationChecker(bootstrap_n=99)
        report = checker.check(y, y_hat, exp, seed=0)
        assert report.verdict() in ("OK", "MONITOR", "RECALIBRATE", "REFIT")

    def test_summary_is_string(self):
        y, y_hat, exp = _make_data()
        checker = CalibrationChecker(bootstrap_n=99)
        report = checker.check(y, y_hat, exp, seed=0)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "balance" in summary.lower()
        assert "murphy" in summary.lower()

    def test_to_polars(self):
        import polars as pl
        y, y_hat, exp = _make_data()
        checker = CalibrationChecker(bootstrap_n=99)
        report = checker.check(y, y_hat, exp, seed=0)
        df = report.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        # Should have columns from balance, auto-cal, murphy, and report-level
        assert any("balance" in c for c in df.columns)
        assert any("murphy" in c for c in df.columns)
        assert "verdict" in df.columns

    def test_miscalibrated_model_gets_nonok_verdict(self):
        """30% scale error should not get OK verdict."""
        y, y_hat, exp = _make_data(scale=1.3, n=3000, seed=0)
        checker = CalibrationChecker(bootstrap_n=199)
        report = checker.check(y, y_hat, exp, seed=0)
        assert report.verdict() != "OK"

    def test_n_policies_and_exposure(self):
        y, y_hat, exp = _make_data(n=500)
        checker = CalibrationChecker(bootstrap_n=99)
        report = checker.check(y, y_hat, exp)
        assert report.n_policies == 500
        assert report.total_exposure == pytest.approx(float(np.sum(exp)), rel=1e-6)

    def test_repr(self):
        checker = CalibrationChecker(distribution="poisson")
        assert "CalibrationChecker" in repr(checker)
        assert "not fitted" in repr(checker)
        checker.fit(*_make_data(n=200), seed=0)
        assert "fitted" in repr(checker)

    def test_all_distributions(self):
        """CalibrationChecker should work for all distributions."""
        rng = np.random.default_rng(5)
        n = 500
        y_hat = rng.gamma(2, 0.1, n)

        for dist in ["poisson", "normal", "tweedie"]:
            y = rng.gamma(2, 0.1, n)
            checker = CalibrationChecker(
                distribution=dist, bootstrap_n=49, tweedie_power=1.5
            )
            report = checker.check(y, y_hat, seed=0)
            assert report.verdict() in ("OK", "MONITOR", "RECALIBRATE", "REFIT")

    def test_no_exposure_works(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(1, 0.1, 300)
        y_hat = rng.gamma(1, 0.1, 300)
        checker = CalibrationChecker(bootstrap_n=49)
        report = checker.check(y, y_hat, exposure=None, seed=0)
        assert report.total_exposure == pytest.approx(300.0, rel=1e-6)


class TestCalibrationReportVerdict:
    """Test verdict logic for different miscalibration patterns."""

    def test_globally_balanced_well_ranked(self):
        """Good model: balance ratio near 1.0, low MCB."""
        y, y_hat, exp = _make_data(n=5000, scale=1.0, seed=0)
        checker = CalibrationChecker(bootstrap_n=199)
        report = checker.check(y, y_hat, exp, seed=0)
        assert abs(report.balance.balance_ratio - 1.0) < 0.05
        assert report.murphy.miscalibration / report.murphy.uncertainty < 0.15

    def test_large_scale_error_gets_recalibrate(self):
        """40% global over-prediction should trigger RECALIBRATE."""
        rng = np.random.default_rng(0)
        n = 3000
        exp = np.ones(n)
        y_hat_true = rng.gamma(2, 0.05, n)
        counts = rng.poisson(y_hat_true)
        y = counts.astype(float)
        y_hat_biased = y_hat_true * 1.4  # over-predict by 40%
        checker = CalibrationChecker(bootstrap_n=199)
        report = checker.check(y, y_hat_biased, exp, seed=0)
        assert report.verdict() in ("RECALIBRATE", "REFIT")
