"""Tests for insurance_monitoring.calibration._plots module.

The plot functions use matplotlib, so we patch plt.show() to avoid
display issues. We test:
1. plot_auto_calibration() returns a Figure
2. plot_murphy() returns a Figure
3. plot_balance_over_time() returns a Figure
4. plot_calibration_report() returns a Figure (requires CalibrationChecker)

We mock matplotlib.pyplot.subplots and use a non-interactive backend
to avoid any display output.
"""

from __future__ import annotations

import numpy as np
import pytest

# Use non-interactive matplotlib backend for tests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from insurance_monitoring.calibration import (
    check_auto_calibration,
    murphy_decomposition,
    CalibrationChecker,
    plot_auto_calibration,
    plot_murphy,
    plot_balance_over_time,
    plot_calibration_report,
)


def _make_poisson_data(n=3_000, seed=0):
    """Generate synthetic Poisson frequency data."""
    rng = np.random.default_rng(seed)
    y_hat = rng.uniform(0.05, 0.20, n)
    exposure = rng.uniform(0.5, 1.5, n)
    y = rng.poisson(y_hat * exposure).astype(float)
    return y, y_hat, exposure


class TestPlotAutoCalibration:
    def test_returns_figure(self):
        """plot_auto_calibration should return a matplotlib Figure."""
        y, y_hat, exposure = _make_poisson_data()
        result = check_auto_calibration(
            y, y_hat, exposure=exposure, n_bins=5, bootstrap_n=50, seed=0
        )
        fig = plot_auto_calibration(result)
        assert fig is not None
        plt.close("all")

    def test_with_existing_axes(self):
        """plot_auto_calibration should accept an existing Axes."""
        y, y_hat, exposure = _make_poisson_data()
        result = check_auto_calibration(
            y, y_hat, exposure=exposure, n_bins=5, bootstrap_n=50, seed=0
        )
        fig_pre, ax_pre = plt.subplots()
        returned_fig = plot_auto_calibration(result, ax=ax_pre)
        assert returned_fig is not None
        plt.close("all")

    def test_log_scale_false(self):
        """plot_auto_calibration with log_scale=False should not raise."""
        y, y_hat, exposure = _make_poisson_data()
        result = check_auto_calibration(
            y, y_hat, exposure=exposure, n_bins=5, bootstrap_n=50, seed=0
        )
        fig = plot_auto_calibration(result, log_scale=False)
        assert fig is not None
        plt.close("all")

    def test_custom_title(self):
        """plot_auto_calibration should accept custom title."""
        y, y_hat, exposure = _make_poisson_data()
        result = check_auto_calibration(
            y, y_hat, exposure=exposure, n_bins=5, bootstrap_n=50, seed=0
        )
        fig = plot_auto_calibration(result, title="My Custom Title")
        assert fig is not None
        plt.close("all")


class TestPlotMurphy:
    def test_returns_figure(self):
        """plot_murphy should return a matplotlib Figure."""
        y, y_hat, exposure = _make_poisson_data()
        result = murphy_decomposition(y, y_hat, exposure=exposure, n_bins=5)
        fig = plot_murphy(result)
        assert fig is not None
        plt.close("all")

    def test_with_existing_axes(self):
        """plot_murphy should accept an existing Axes."""
        y, y_hat, exposure = _make_poisson_data()
        result = murphy_decomposition(y, y_hat, exposure=exposure, n_bins=5)
        fig_pre, ax_pre = plt.subplots()
        returned_fig = plot_murphy(result, ax=ax_pre)
        assert returned_fig is not None
        plt.close("all")

    def test_gamma_distribution(self):
        """plot_murphy should work for Gamma distribution."""
        rng = np.random.default_rng(1)
        n = 2_000
        y_hat = rng.uniform(0.5, 2.0, n)
        exposure = np.ones(n)
        # Gamma outcomes: shape=1, scale=y_hat
        from scipy.stats import gamma
        y = gamma.rvs(a=1.0, scale=y_hat, random_state=1)
        result = murphy_decomposition(y, y_hat, exposure=exposure, n_bins=5, distribution="gamma")
        fig = plot_murphy(result)
        assert fig is not None
        plt.close("all")


class TestPlotBalanceOverTime:
    def test_returns_figure(self):
        """plot_balance_over_time should return a matplotlib Figure."""
        periods = ["2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4"]
        ratios = [1.01, 0.98, 1.05, 0.97]
        ci_lowers = [0.95, 0.92, 0.99, 0.91]
        ci_uppers = [1.07, 1.04, 1.11, 1.03]
        fig = plot_balance_over_time(periods, ratios, ci_lowers, ci_uppers)
        assert fig is not None
        plt.close("all")

    def test_with_imbalanced_periods(self):
        """Plot should highlight periods where CI excludes 1.0."""
        periods = ["Q1", "Q2", "Q3"]
        ratios = [1.20, 1.00, 0.75]
        ci_lowers = [1.10, 0.95, 0.70]
        ci_uppers = [1.30, 1.05, 0.80]
        fig = plot_balance_over_time(periods, ratios, ci_lowers, ci_uppers)
        assert fig is not None
        plt.close("all")

    def test_with_existing_axes(self):
        """plot_balance_over_time should accept an existing Axes."""
        periods = ["Q1", "Q2"]
        ratios = [1.0, 1.05]
        ci_lowers = [0.95, 1.00]
        ci_uppers = [1.05, 1.10]
        fig_pre, ax_pre = plt.subplots()
        returned_fig = plot_balance_over_time(periods, ratios, ci_lowers, ci_uppers, ax=ax_pre)
        assert returned_fig is not None
        plt.close("all")

    def test_single_period(self):
        """Single-period plot should not raise."""
        fig = plot_balance_over_time(["2024"], [1.02], [0.95], [1.10])
        assert fig is not None
        plt.close("all")


class TestPlotCalibrationReport:
    def test_returns_figure(self):
        """plot_calibration_report should return a matplotlib Figure."""
        y, y_hat, exposure = _make_poisson_data()
        checker = CalibrationChecker(distribution="poisson")
        report = checker.check(y, y_hat, exposure=exposure)
        fig = plot_calibration_report(report)
        assert fig is not None
        plt.close("all")

    def test_custom_figsize(self):
        """plot_calibration_report should accept custom figsize."""
        y, y_hat, exposure = _make_poisson_data()
        checker = CalibrationChecker(distribution="poisson")
        report = checker.check(y, y_hat, exposure=exposure)
        fig = plot_calibration_report(report, figsize=(12, 8))
        assert fig is not None
        plt.close("all")
