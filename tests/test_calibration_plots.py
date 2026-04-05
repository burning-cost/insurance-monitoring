"""Smoke tests for calibration plot functions.

Verifies that plot_auto_calibration, plot_murphy, plot_balance_over_time,
and plot_calibration_report each return a Figure without raising on valid inputs,
and each raises or handles an obvious error case.

Two styles of test are used:
- Unit style: construct dataclass instances directly, no bootstrap/fitting needed.
  Fast and deterministic, suitable for CI.
- Integration style: use CalibrationChecker.check() on synthetic data for
  plot_calibration_report, which validates the full pipeline.

matplotlib Agg backend is forced so tests run headless without a display.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from insurance_monitoring.calibration._types import (
    AutoCalibResult,
    BalanceResult,
    CalibrationReport,
    MurphyResult,
)
from insurance_monitoring.calibration._plots import (
    plot_auto_calibration,
    plot_murphy,
    plot_balance_over_time,
    plot_calibration_report,
)


# ---------------------------------------------------------------------------
# Helpers: construct minimal valid result objects (unit style — no compute)
# ---------------------------------------------------------------------------

def _make_per_bin(n_bins=10):
    """Minimal per_bin Polars DataFrame as expected by AutoCalibResult."""
    rng = np.random.default_rng(0)
    pred = np.sort(rng.uniform(0.05, 0.20, n_bins))
    obs = pred * rng.uniform(0.85, 1.15, n_bins)
    exposure = rng.uniform(50, 200, n_bins).astype(float)
    ratio = obs / pred
    return pl.DataFrame(
        {
            "bin": list(range(1, n_bins + 1)),
            "pred_mean": pred.tolist(),
            "obs_mean": obs.tolist(),
            "ratio": ratio.tolist(),
            "exposure": exposure.tolist(),
            "n_policies": [int(e * 2) for e in exposure],
        }
    )


def _make_auto_calib_result(n_bins=10, calibrated=True):
    return AutoCalibResult(
        p_value=0.3 if calibrated else 0.01,
        is_calibrated=calibrated,
        per_bin=_make_per_bin(n_bins),
        mcb_score=0.0002,
        worst_bin_ratio=0.08,
        n_isotonic_steps=5,
    )


def _make_murphy_result(verdict="OK"):
    return MurphyResult(
        total_deviance=0.10,
        uncertainty=0.12,
        discrimination=0.05,
        miscalibration=0.03,
        global_mcb=0.01,
        local_mcb=0.02,
        discrimination_pct=50.0,
        miscalibration_pct=30.0,
        verdict=verdict,
    )


def _make_balance_result(balanced=True):
    return BalanceResult(
        balance_ratio=1.01 if balanced else 1.20,
        observed_total=1010.0,
        predicted_total=1000.0,
        ci_lower=0.97 if balanced else 1.15,
        ci_upper=1.05 if balanced else 1.25,
        p_value=0.4 if balanced else 0.001,
        is_balanced=balanced,
        n_policies=5000,
        total_exposure=4800.0,
    )


def _make_calibration_report():
    return CalibrationReport(
        balance=_make_balance_result(),
        auto_calibration=_make_auto_calib_result(),
        murphy=_make_murphy_result(),
        distribution="poisson",
        n_policies=5000,
        total_exposure=4800.0,
    )


# ---------------------------------------------------------------------------
# plot_auto_calibration
# ---------------------------------------------------------------------------

class TestPlotAutoCalibration:
    def test_returns_figure(self):
        result = _make_auto_calib_result()
        fig = plot_auto_calibration(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_uses_provided_ax(self):
        fig_in, ax_in = plt.subplots()
        result = _make_auto_calib_result()
        fig_out = plot_auto_calibration(result, ax=ax_in)
        assert isinstance(fig_out, matplotlib.figure.Figure)
        plt.close("all")

    def test_linear_scale(self):
        result = _make_auto_calib_result()
        fig = plot_auto_calibration(result, log_scale=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self):
        result = _make_auto_calib_result()
        fig, ax = plt.subplots()
        plot_auto_calibration(result, title="My Calib", ax=ax)
        assert ax.get_title() == "My Calib"
        plt.close("all")

    def test_not_calibrated_result(self):
        """Functions should work the same whether is_calibrated is True or False."""
        result = _make_auto_calib_result(calibrated=False)
        fig = plot_auto_calibration(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_single_bin_no_crash(self):
        """Edge case: single prediction bin."""
        result = _make_auto_calib_result(n_bins=1)
        fig = plot_auto_calibration(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_missing_per_bin_column_raises(self):
        """If per_bin is missing the exposure column, the function should raise."""
        result = _make_auto_calib_result()
        result = AutoCalibResult(
            p_value=result.p_value,
            is_calibrated=result.is_calibrated,
            per_bin=result.per_bin.drop("exposure"),
            mcb_score=result.mcb_score,
            worst_bin_ratio=result.worst_bin_ratio,
            n_isotonic_steps=result.n_isotonic_steps,
        )
        with pytest.raises(Exception):
            plot_auto_calibration(result)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_murphy
# ---------------------------------------------------------------------------

class TestPlotMurphy:
    def test_returns_figure(self):
        result = _make_murphy_result()
        fig = plot_murphy(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_uses_provided_ax(self):
        fig_in, ax_in = plt.subplots()
        result = _make_murphy_result()
        fig_out = plot_murphy(result, ax=ax_in)
        assert isinstance(fig_out, matplotlib.figure.Figure)
        plt.close("all")

    def test_verdict_recalibrate(self):
        result = _make_murphy_result(verdict="RECALIBRATE")
        fig = plot_murphy(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_verdict_refit(self):
        result = _make_murphy_result(verdict="REFIT")
        fig = plot_murphy(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_zero_mcb(self):
        """Perfectly calibrated model where MCB=0 should not crash."""
        result = MurphyResult(
            total_deviance=0.10,
            uncertainty=0.12,
            discrimination=0.05,
            miscalibration=0.0,
            global_mcb=0.0,
            local_mcb=0.0,
            discrimination_pct=50.0,
            miscalibration_pct=0.0,
            verdict="OK",
        )
        fig = plot_murphy(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_missing_attribute_raises(self):
        """Passing a non-MurphyResult object should raise AttributeError."""
        with pytest.raises(AttributeError):
            plot_murphy("not_a_result")
        plt.close("all")

    def test_from_real_decomposition(self):
        """Smoke test using murphy_decomposition on synthetic Gamma data."""
        from insurance_monitoring.calibration import murphy_decomposition
        from scipy.stats import gamma as gamma_dist

        rng = np.random.default_rng(1)
        n = 500
        y_hat = rng.uniform(0.5, 2.0, n)
        y = gamma_dist.rvs(a=1.0, scale=y_hat, random_state=1)
        result = murphy_decomposition(y, y_hat, exposure=np.ones(n), distribution="gamma")
        fig = plot_murphy(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_balance_over_time
# ---------------------------------------------------------------------------

class TestPlotBalanceOverTime:
    def _sample_inputs(self, n=6, all_balanced=True):
        rng = np.random.default_rng(1)
        periods = [f"Q{i}" for i in range(1, n + 1)]
        if all_balanced:
            ratios = (1.0 + rng.uniform(-0.03, 0.03, n)).tolist()
            lowers = (np.array(ratios) - 0.05).tolist()
            uppers = (np.array(ratios) + 0.05).tolist()
        else:
            ratios = [0.75, 1.0, 1.05, 1.30, 0.95, 1.01]
            lowers = [0.70, 0.92, 0.98, 1.22, 0.87, 0.94]
            uppers = [0.80, 1.08, 1.12, 1.38, 1.03, 1.08]
        return periods, ratios, lowers, uppers

    def test_returns_figure(self):
        periods, ratios, lowers, uppers = self._sample_inputs()
        fig = plot_balance_over_time(periods, ratios, lowers, uppers)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_uses_provided_ax(self):
        fig_in, ax_in = plt.subplots()
        periods, ratios, lowers, uppers = self._sample_inputs()
        fig_out = plot_balance_over_time(periods, ratios, lowers, uppers, ax=ax_in)
        assert isinstance(fig_out, matplotlib.figure.Figure)
        plt.close("all")

    def test_with_imbalanced_periods(self):
        """Some periods have CI excluding 1.0 — the red dot branch fires."""
        periods, ratios, lowers, uppers = self._sample_inputs(all_balanced=False)
        fig = plot_balance_over_time(periods, ratios, lowers, uppers)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_single_period(self):
        fig = plot_balance_over_time(["Q1"], [1.02], [0.95], [1.09])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_mismatched_lengths_raises(self):
        """periods and ratios of different length should raise."""
        with pytest.raises(Exception):
            plot_balance_over_time(["Q1", "Q2"], [1.0], [0.9], [1.1])
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_calibration_report (integration style — uses CalibrationChecker)
# ---------------------------------------------------------------------------

class TestPlotCalibrationReport:
    def test_returns_figure(self):
        report = _make_calibration_report()
        fig = plot_calibration_report(report)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_figsize(self):
        report = _make_calibration_report()
        fig = plot_calibration_report(report, figsize=(10, 8))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_verdict_in_title(self):
        """The overall verdict should appear in the figure suptitle."""
        report = _make_calibration_report()
        fig = plot_calibration_report(report)
        suptitle = fig.texts[0].get_text() if fig.texts else ""
        assert "Verdict" in suptitle
        plt.close("all")

    def test_report_verdict_refit(self):
        """REFIT verdict flows through without error."""
        report = CalibrationReport(
            balance=_make_balance_result(balanced=False),
            auto_calibration=_make_auto_calib_result(calibrated=False),
            murphy=_make_murphy_result(verdict="REFIT"),
            distribution="gamma",
            n_policies=2000,
            total_exposure=1900.0,
        )
        fig = plot_calibration_report(report)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_from_calibration_checker(self):
        """Integration test: build report through CalibrationChecker.check()."""
        from insurance_monitoring.calibration import CalibrationChecker

        rng = np.random.default_rng(0)
        n = 1000
        y_hat = rng.uniform(0.05, 0.20, n)
        exposure = rng.uniform(0.5, 1.5, n)
        y = rng.poisson(y_hat * exposure).astype(float)

        checker = CalibrationChecker(distribution="poisson")
        report = checker.check(y, y_hat, exposure=exposure)
        fig = plot_calibration_report(report)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_missing_attribute_raises(self):
        """Passing a non-report object should raise AttributeError."""
        with pytest.raises(AttributeError):
            plot_calibration_report({"not": "a report"})
        plt.close("all")
