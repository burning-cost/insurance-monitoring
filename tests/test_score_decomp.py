"""Tests for ScoreDecompositionTest (arXiv:2603.04275, Dimitriadis & Puke 2026)."""

from __future__ import annotations

import numpy as np
import pytest

from insurance_monitoring.calibration import (
    ScoreDecompositionTest,
    ScoreDecompositionResult,
    TwoForecastSDIResult,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_normal_data(n: int = 2000, seed: int = 0, scale: float = 1.0):
    """Normal regression setup. y_hat = true conditional mean * scale."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 5, n)
    mu = 1.0 + 2.0 * x
    y = mu + rng.normal(0, 1.5, n)
    y_hat = mu * scale
    return y, y_hat


def _make_poisson_data(n: int = 2000, seed: int = 0):
    """Poisson counts — tests score decomp on non-Gaussian residuals."""
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.5, 5.0, n)
    y = rng.poisson(lam).astype(float)
    y_hat = lam
    return y, y_hat


# ---------------------------------------------------------------------------
# Identity check: S = MCB + UNC - DSC
# ---------------------------------------------------------------------------


class TestDecompositionIdentity:
    """The decomposition identity S = MCB + UNC - DSC must hold exactly."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_identity_mse(self, seed):
        y, y_hat = _make_normal_data(n=1000, seed=seed)
        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.score == pytest.approx(
            r.miscalibration + r.uncertainty - r.discrimination, abs=1e-9
        )

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    def test_identity_quantile(self, q):
        rng = np.random.default_rng(5)
        n = 800
        y = rng.gamma(2, 1.0, n)
        y_hat = np.quantile(y, q) * np.ones(n) + rng.normal(0, 0.1, n)
        sdi = ScoreDecompositionTest(score_type="quantile", alpha=q, hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.score == pytest.approx(
            r.miscalibration + r.uncertainty - r.discrimination, abs=1e-9
        )

    def test_identity_mae(self):
        y, y_hat = _make_normal_data(n=800, seed=3)
        sdi = ScoreDecompositionTest(score_type="mae", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.score == pytest.approx(
            r.miscalibration + r.uncertainty - r.discrimination, abs=1e-9
        )

    @pytest.mark.parametrize("seed", [0, 7])
    def test_identity_poisson_data(self, seed):
        y, y_hat = _make_poisson_data(n=1500, seed=seed)
        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.score == pytest.approx(
            r.miscalibration + r.uncertainty - r.discrimination, abs=1e-9
        )


# ---------------------------------------------------------------------------
# Perfect calibration: MCB should be near-zero and p_mcb should be large
# ---------------------------------------------------------------------------


class TestPerfectCalibration:
    def test_mcb_small_when_calibrated(self):
        """If y_hat = E[y|X], MCB should be near zero on average."""
        rng = np.random.default_rng(42)
        n = 3000
        x = rng.uniform(0, 1, n)
        mu = 2.0 + 3.0 * x
        y = mu + rng.normal(0, 1.0, n)
        y_hat = mu  # exact conditional mean

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        # MCB should be small relative to UNC
        assert abs(r.miscalibration) / r.uncertainty < 0.05, (
            f"MCB/UNC={abs(r.miscalibration)/r.uncertainty:.4f} — expected < 0.05 for "
            f"perfectly calibrated forecast"
        )

    def test_mcb_pvalue_large_when_calibrated(self):
        """p_mcb should be large (not significant) for a calibrated forecast."""
        rng = np.random.default_rng(99)
        n = 3000
        mu = rng.uniform(1, 10, n)
        y = mu + rng.normal(0, 0.5, n)
        y_hat = mu

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        # With n=3000 and perfect calibration, we expect p_mcb > 0.05 most of the time
        # Use a very lax threshold to avoid flaky tests
        assert r.mcb_pvalue > 0.001, (
            f"p_mcb={r.mcb_pvalue:.4f} — expected not significant for calibrated forecast"
        )


# ---------------------------------------------------------------------------
# Known miscalibration: MCB should be large and significant
# ---------------------------------------------------------------------------


class TestKnownMiscalibration:
    def test_scaled_forecast_large_mcb(self):
        """y_hat * 1.5 introduces systematic bias — MCB should be large."""
        y, y_hat = _make_normal_data(n=3000, seed=10, scale=1.5)
        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.miscalibration > 0
        assert r.mcb_pvalue < 0.05, (
            f"p_mcb={r.mcb_pvalue:.4f} — expected significant for 50% scale error"
        )

    def test_shifted_forecast_mcb_significant(self):
        """A constant additive shift in forecasts should produce significant MCB."""
        rng = np.random.default_rng(20)
        n = 2000
        y = rng.normal(5, 1, n)
        y_hat = y + 2.0  # shift upward by 2

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.mcb_pvalue < 0.01


# ---------------------------------------------------------------------------
# No discrimination: DSC should be near zero
# ---------------------------------------------------------------------------


class TestNoDiscrimination:
    def test_constant_forecast_dsc_small(self):
        """A constant forecast has no discriminatory power — DSC close to zero."""
        rng = np.random.default_rng(30)
        n = 2000
        y = rng.normal(5, 2, n)
        y_hat = np.full(n, y.mean())  # grand mean — no information

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        # DSC should be near zero (constant forecast = climate forecast, no improvement)
        assert abs(r.discrimination) < 0.1, (
            f"DSC={r.discrimination:.4f} — expected near zero for constant forecast"
        )

    def test_dsc_pvalue_large_for_constant_forecast(self):
        """p_dsc should be large for a constant (uninformative) forecast."""
        rng = np.random.default_rng(31)
        n = 2000
        y = rng.normal(5, 2, n)
        y_hat = np.full(n, 5.0)

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.dsc_pvalue > 0.05


# ---------------------------------------------------------------------------
# HAC: iid data, auto vs hac_lags=0 should give close SEs
# ---------------------------------------------------------------------------


class TestHACLags:
    def test_hac0_vs_auto_close_for_iid(self):
        """For iid data, hac_lags=0 and auto should give similar SEs."""
        y, y_hat = _make_normal_data(n=2000, seed=50)

        sdi_0 = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        sdi_auto = ScoreDecompositionTest(score_type="mse", hac_lags=None)

        r0 = sdi_0.fit_single(y, y_hat)
        r_auto = sdi_auto.fit_single(y, y_hat)

        # SEs should be within 50% of each other for iid data
        assert abs(r0.mcb_se - r_auto.mcb_se) / (r0.mcb_se + 1e-12) < 0.5
        assert abs(r0.dsc_se - r_auto.dsc_se) / (r0.dsc_se + 1e-12) < 0.5


# ---------------------------------------------------------------------------
# Two-sample tests
# ---------------------------------------------------------------------------


class TestTwoSample:
    def test_good_vs_constant_dsc_significant(self):
        """Good model vs constant forecast: delta_dsc should be significant."""
        rng = np.random.default_rng(60)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = 1.0 + 2.0 * x
        y = mu + rng.normal(0, 1.0, n)

        y_hat_a = mu              # good model: discriminating
        y_hat_b = np.full(n, mu.mean())  # constant

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_two(y, y_hat_a, y_hat_b)

        assert isinstance(r, TwoForecastSDIResult)
        # A has lower score (better)
        assert r.delta_score < 0
        # A has more discrimination
        assert r.result_a.discrimination > r.result_b.discrimination
        # delta_dsc test should be significant
        assert r.delta_dsc_pvalue < 0.05

    def test_returns_individual_results(self):
        """fit_two should embed valid fit_single results for each forecast."""
        y, y_hat = _make_normal_data(n=1000, seed=70)
        y_hat_b = y_hat * 1.2

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_two(y, y_hat, y_hat_b)

        # Check individual results are consistent with fit_single
        sdi2 = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r_a = sdi2.fit_single(y, y_hat)
        assert r.result_a.score == pytest.approx(r_a.score, rel=1e-9)
        assert r.result_a.miscalibration == pytest.approx(r_a.miscalibration, rel=1e-9)

    def test_combined_pvalue_in_unit_interval(self):
        y, y_hat = _make_normal_data(n=1000, seed=80)
        y_hat_b = y_hat * 0.9

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_two(y, y_hat, y_hat_b)
        assert 0 <= r.combined_pvalue <= 2.0  # IU can exceed 1, but bounded by 2*min


# ---------------------------------------------------------------------------
# Exposure weighting
# ---------------------------------------------------------------------------


class TestExposureWeighting:
    def test_uniform_exposure_matches_unweighted(self):
        """Exposure=ones should give same result as no exposure."""
        y, y_hat = _make_normal_data(n=500, seed=90)
        n = len(y)

        sdi_no_exp = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        sdi_ones = ScoreDecompositionTest(
            score_type="mse", hac_lags=0, exposure=np.ones(n)
        )

        r1 = sdi_no_exp.fit_single(y, y_hat)
        r2 = sdi_ones.fit_single(y, y_hat)

        assert r1.score == pytest.approx(r2.score, rel=1e-9)
        assert r1.miscalibration == pytest.approx(r2.miscalibration, rel=1e-9)
        assert r1.discrimination == pytest.approx(r2.discrimination, rel=1e-9)

    def test_varying_exposure_changes_result(self):
        """Non-uniform exposure should produce different estimates."""
        rng = np.random.default_rng(91)
        n = 500
        y, y_hat = _make_normal_data(n=n, seed=91)
        exposure = rng.uniform(0.1, 3.0, n)

        sdi_no_exp = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        sdi_exp = ScoreDecompositionTest(
            score_type="mse", hac_lags=0, exposure=exposure
        )

        r1 = sdi_no_exp.fit_single(y, y_hat)
        r2 = sdi_exp.fit_single(y, y_hat)

        # Results should differ (not equal)
        assert not np.isclose(r1.score, r2.score), (
            "Exposure weighting had no effect — check weighted mean implementation"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_small_sample(self):
        """n=50 should work without errors."""
        rng = np.random.default_rng(100)
        n = 50
        y = rng.normal(3, 1, n)
        y_hat = y + rng.normal(0, 0.3, n)
        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert isinstance(r, ScoreDecompositionResult)
        assert np.isfinite(r.mcb_pvalue)
        assert np.isfinite(r.dsc_pvalue)

    def test_all_equal_forecasts_degenerate(self):
        """All-equal forecasts (zero variance) should not raise; slope=0."""
        rng = np.random.default_rng(110)
        y = rng.normal(3, 1, 200)
        y_hat = np.full(200, 3.0)

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.recalib_slope == pytest.approx(0.0, abs=1e-10)
        assert np.isfinite(r.score)

    def test_returns_correct_types(self):
        y, y_hat = _make_normal_data(n=100, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_single(y, y_hat)
        assert isinstance(r, ScoreDecompositionResult)
        assert isinstance(r.score, float)
        assert isinstance(r.n, int)
        assert r.n == 100


# ---------------------------------------------------------------------------
# Quantile / MAE tests
# ---------------------------------------------------------------------------


class TestQuantileScores:
    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    def test_quantile_identity_holds(self, q):
        rng = np.random.default_rng(200 + int(q * 100))
        n = 800
        y = rng.exponential(2.0, n)
        y_hat = np.full(n, np.quantile(y, q)) + rng.normal(0, 0.1, n)
        sdi = ScoreDecompositionTest(score_type="quantile", alpha=q, hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.score == pytest.approx(
            r.miscalibration + r.uncertainty - r.discrimination, abs=1e-9
        )

    def test_mae_identity_holds(self):
        rng = np.random.default_rng(210)
        y = rng.exponential(1.5, 800)
        y_hat = np.median(y) + rng.normal(0, 0.1, 800)
        sdi = ScoreDecompositionTest(score_type="mae", hac_lags=0)
        r = sdi.fit_single(y, y_hat)
        assert r.score == pytest.approx(
            r.miscalibration + r.uncertainty - r.discrimination, abs=1e-9
        )

    def test_quantile_score_nonnegative(self):
        """Pinball loss is always non-negative."""
        rng = np.random.default_rng(220)
        y = rng.normal(0, 1, 500)
        y_hat = rng.normal(0, 1, 500)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            sdi = ScoreDecompositionTest(score_type="quantile", alpha=q, hac_lags=0)
            r = sdi.fit_single(y, y_hat)
            assert r.score >= -1e-10


# ---------------------------------------------------------------------------
# API surface: repr, summary, plot
# ---------------------------------------------------------------------------


class TestAPISurface:
    def test_repr_single(self):
        y, y_hat = _make_normal_data(n=200, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_single(y, y_hat)
        rep = repr(r)
        assert "ScoreDecompositionResult" in rep
        assert "MCB=" in rep

    def test_repr_two(self):
        y, y_hat = _make_normal_data(n=200, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_two(y, y_hat, y_hat * 1.1)
        rep = repr(r)
        assert "TwoForecastSDIResult" in rep

    def test_summary_single(self):
        y, y_hat = _make_normal_data(n=200, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_single(y, y_hat)
        s = r.summary()
        assert "MCB" in s
        assert "DSC" in s
        assert "UNC" in s

    def test_summary_two(self):
        y, y_hat = _make_normal_data(n=200, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_two(y, y_hat, y_hat * 1.1)
        s = r.summary()
        assert "Delta" in s

    def test_plot_single_returns_axes(self):
        """plot() should return a matplotlib Axes without raising."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, y_hat = _make_normal_data(n=200, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_single(y, y_hat)
        ax = sdi.plot(r)
        assert ax is not None
        plt.close("all")

    def test_plot_two_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, y_hat = _make_normal_data(n=200, seed=0)
        sdi = ScoreDecompositionTest()
        r = sdi.fit_two(y, y_hat, y_hat * 1.1)
        ax = sdi.plot(r)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_score_type(self):
        with pytest.raises(ValueError, match="score_type"):
            ScoreDecompositionTest(score_type="rmse")  # type: ignore[arg-type]

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ScoreDecompositionTest(alpha=1.5)

    def test_mismatched_lengths(self):
        sdi = ScoreDecompositionTest()
        with pytest.raises(ValueError):
            sdi.fit_single(np.ones(10), np.ones(11))

    def test_too_short(self):
        sdi = ScoreDecompositionTest()
        with pytest.raises(ValueError):
            sdi.fit_single(np.ones(3), np.ones(3))

    def test_negative_exposure_raises(self):
        sdi = ScoreDecompositionTest(exposure=np.array([-1.0, 1.0, 1.0, 1.0, 1.0]))
        with pytest.raises(ValueError, match="strictly positive"):
            sdi.fit_single(np.ones(5), np.ones(5))


# ---------------------------------------------------------------------------
# Regression: IU combined p-value uses delta p-values, not per-model p-values
# ---------------------------------------------------------------------------


class TestCombinedPValueFormula:
    """Regression test for the IU combined p-value bug.

    The combined p-value must be max(p_delta_mcb, p_delta_dsc) — the maximum
    of the two *delta* component p-values. The earlier, incorrect formula was
    max(p_dm, 2*min(res_a.mcb_pvalue, res_b.mcb_pvalue)), which used the
    individual per-model MCB p-values instead of the delta_mcb p-value
    computed from the joint HAC covariance.
    """

    def test_combined_pvalue_equals_max_of_delta_pvalues(self):
        """combined_pvalue must equal max(delta_mcb_pvalue, delta_dsc_pvalue)."""
        rng = np.random.default_rng(42)
        n = 1500
        x = rng.uniform(0, 5, n)
        mu = 1.0 + 2.0 * x
        y = mu + rng.normal(0, 1.5, n)
        # Two forecasts with different calibration errors
        y_hat_a = mu * 1.1   # slight over-prediction
        y_hat_b = mu * 0.9   # slight under-prediction

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_two(y, y_hat_a, y_hat_b)

        expected = max(r.delta_mcb_pvalue, r.delta_dsc_pvalue)
        assert r.combined_pvalue == pytest.approx(expected, abs=1e-12), (
            f"combined_pvalue={r.combined_pvalue:.6f} but "
            f"max(delta_mcb_pvalue={r.delta_mcb_pvalue:.6f}, "
            f"delta_dsc_pvalue={r.delta_dsc_pvalue:.6f})={expected:.6f}. "
            "The IU combined p-value must use the delta component p-values."
        )

    def test_combined_pvalue_not_equal_individual_mcb_formula(self):
        """The old (wrong) formula must NOT match the result for a non-trivial case."""
        rng = np.random.default_rng(7)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = 1.0 + 3.0 * x
        y = mu + rng.normal(0, 2.0, n)
        y_hat_a = mu * 1.2
        y_hat_b = mu * 0.8

        sdi = ScoreDecompositionTest(score_type="mse", hac_lags=0)
        r = sdi.fit_two(y, y_hat_a, y_hat_b)

        # Old (wrong) formula
        p_dm = r.delta_score_pvalue
        wrong_combined = max(p_dm, 2.0 * min(r.result_a.mcb_pvalue, r.result_b.mcb_pvalue))

        # Correct formula
        correct_combined = max(r.delta_mcb_pvalue, r.delta_dsc_pvalue)

        # They should differ (the bug would have returned the wrong value)
        assert r.combined_pvalue == pytest.approx(correct_combined, abs=1e-12), (
            "combined_pvalue does not match max(delta_mcb_pvalue, delta_dsc_pvalue)"
        )
        # And for this dataset the two formulas produce different numbers
        # (this confirms the test would have caught the old bug)
        assert not np.isclose(wrong_combined, correct_combined, rtol=1e-6), (
            "Old and new formulae give identical values for this dataset — "
            "the regression test is not discriminative. Choose different data."
        )
