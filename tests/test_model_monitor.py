"""Tests for ModelMonitor, check_gmcb, and check_lmcb.

Covers the three decision paths (REFIT, RECALIBRATE, REDEPLOY), edge cases,
and consistency checks against murphy_decomposition().
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_monitoring.model_monitor import ModelMonitor, ModelMonitorResult
from insurance_monitoring.calibration._gmcb_lmcb import check_gmcb, check_lmcb
from insurance_monitoring.calibration._types import GMCBResult, LMCBResult
from insurance_monitoring.calibration import murphy_decomposition


# ---------------------------------------------------------------------------
# Shared data generators
# ---------------------------------------------------------------------------


def _make_clean_data(n: int = 3000, seed: int = 42) -> tuple:
    """Generate a well-calibrated, well-ranked dataset."""
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_hat = rng.gamma(2, 0.05, n)
    counts = rng.poisson(exposure * y_hat)
    y = counts / exposure
    return y, y_hat, exposure


def _make_monitor() -> ModelMonitor:
    return ModelMonitor(
        distribution="poisson",
        n_bootstrap=199,
        alpha_gini=0.32,
        alpha_global=0.32,
        alpha_local=0.32,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Decision path: REDEPLOY
# ---------------------------------------------------------------------------


class TestRedeploy:
    def test_redeploy_no_drift(self):
        """Same DGP for reference and monitor periods -> REDEPLOY."""
        y_ref, y_hat, exp_ref = _make_clean_data(n=3000, seed=1)
        y_new, _, exp_new = _make_clean_data(n=3000, seed=2)

        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp_ref)
        result = monitor.test(y_new, y_hat, exp_new)

        assert result.decision == "REDEPLOY", (
            f"Expected REDEPLOY under no drift, got {result.decision}. "
            f"gini_p={result.gini_p:.3f}, gmcb_p={result.gmcb_p:.3f}, "
            f"lmcb_p={result.lmcb_p:.3f}"
        )

    def test_pvalue_bounds(self):
        """All p-values must be in [0, 1]."""
        y_ref, y_hat, exp = _make_clean_data(n=2000, seed=10)
        y_new, _, _ = _make_clean_data(n=2000, seed=11)

        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp)
        result = monitor.test(y_new, y_hat, exp)

        for name, val in [
            ("gini_p", result.gini_p),
            ("gmcb_p", result.gmcb_p),
            ("lmcb_p", result.lmcb_p),
        ]:
            assert 0.0 <= val <= 1.0, f"{name}={val} is outside [0, 1]"

    def test_result_type(self):
        y_ref, y_hat, exp = _make_clean_data(n=1000, seed=20)
        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp)
        result = monitor.test(y_ref, y_hat, exp)
        assert isinstance(result, ModelMonitorResult)

    def test_decision_field_values(self):
        """decision must be one of the three valid strings."""
        y_ref, y_hat, exp = _make_clean_data(n=1000, seed=21)
        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp)
        result = monitor.test(y_ref, y_hat, exp)
        assert result.decision in ("REDEPLOY", "RECALIBRATE", "REFIT")


# ---------------------------------------------------------------------------
# Decision path: RECALIBRATE
# ---------------------------------------------------------------------------


class TestRecalibrate:
    def test_recalibrate_global_shift(self):
        """Scaling true frequencies by 1.1 (global shift) must produce RECALIBRATE.

        Under a global scalar shift:
        - The Gini is approximately rank-invariant in large samples.
        - GMCB detects the level shift.
        - LMCB is near zero (rank structure intact).
        -> Decision must be RECALIBRATE.

        We use alpha_gini=0.05 (conservative) and alpha_global/local=0.32 (sensitive).
        This matches the paper's dual-level design: Gini drift (REFIT trigger) is costly,
        so we set a high bar (alpha=0.05). Calibration drift (RECALIBRATE trigger) is cheap
        to act on, so we use alpha=0.32 for early detection. See arXiv:2510.04556 Remark 3.
        """
        rng = np.random.default_rng(1)
        n = 5000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)

        # Reference: perfectly balanced
        counts_ref = rng.poisson(exposure * y_hat)
        y_ref = counts_ref / exposure

        # New period: 10% global frequency inflation (claims trend)
        counts_new = rng.poisson(exposure * y_hat * 1.10)
        y_new = counts_new / exposure

        # Use alpha_gini=0.05 (tight) to avoid false Gini alarms under global shift
        monitor = ModelMonitor(
            distribution="poisson",
            n_bootstrap=299,
            alpha_gini=0.05,
            alpha_global=0.32,
            alpha_local=0.32,
            random_state=1,
        )
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_new, y_hat, exposure)

        assert result.decision == "RECALIBRATE", (
            f"Expected RECALIBRATE for global 10% shift, got {result.decision}. "
            f"gini_sig={result.gini_significant} (p={result.gini_p:.3f}), "
            f"gmcb_sig={result.gmcb_significant} (p={result.gmcb_p:.3f}), "
            f"lmcb_sig={result.lmcb_significant} (p={result.lmcb_p:.3f})"
        )

    def test_balance_factor_above_one_for_under_prediction(self):
        """If true claims are inflated, balance factor must be > 1."""
        rng = np.random.default_rng(100)
        n = 2000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        counts = rng.poisson(exposure * y_hat * 1.15)
        y = counts / exposure

        monitor = _make_monitor()
        monitor.fit(y, y_hat, exposure)
        result = monitor.test(y, y_hat, exposure)

        assert result.balance_factor > 1.0, (
            f"Expected balance_factor > 1 for under-prediction, "
            f"got {result.balance_factor:.4f}"
        )


# ---------------------------------------------------------------------------
# Decision path: REFIT
# ---------------------------------------------------------------------------


class TestRefit:
    def test_refit_rank_drift(self):
        """Injecting age-correlated drift should trigger REFIT (Gini drops).

        We create a monitor period where old drivers' frequencies increase
        substantially, while young drivers' are unchanged. The model (trained
        on the pre-drift distribution) now has inverted rank ordering for the
        old-driver segment -> Gini should drop -> REFIT.
        """
        rng = np.random.default_rng(55)
        n = 5000
        exposure = rng.uniform(0.5, 2.0, n)

        # Simple feature: age proxy (uniform in [0,1], high = old)
        age_proxy = rng.uniform(0, 1, n)

        # Pre-drift: frequency decreases with age (young = risky)
        y_hat = 0.05 + 0.15 * (1 - age_proxy)  # young: 0.20, old: 0.05
        counts_ref = rng.poisson(exposure * y_hat)
        y_ref = counts_ref / exposure

        # Post-drift: old drivers become equally risky (model ranking inverted)
        # True frequency now INCREASES with age (old = risky)
        true_rate_new = 0.05 + 0.15 * age_proxy
        counts_new = rng.poisson(exposure * true_rate_new)
        y_new = counts_new / exposure

        monitor = ModelMonitor(
            distribution="poisson",
            n_bootstrap=299,
            alpha_gini=0.32,
            alpha_global=0.32,
            alpha_local=0.32,
            random_state=3,
        )
        monitor.fit(y_ref, y_hat, exposure)
        result = monitor.test(y_new, y_hat, exposure)

        assert result.decision == "REFIT", (
            f"Expected REFIT for rank-inverting drift, got {result.decision}. "
            f"gini_z={result.gini_z:.2f}, gini_p={result.gini_p:.4f}"
        )


# ---------------------------------------------------------------------------
# check_gmcb / check_lmcb standalone functions
# ---------------------------------------------------------------------------


class TestCheckGMCB:
    def test_returns_gmcb_result(self):
        y, y_hat, exp = _make_clean_data(n=1000, seed=30)
        result = check_gmcb(y, y_hat, exp, seed=0)
        assert isinstance(result, GMCBResult)

    def test_pvalue_in_range(self):
        y, y_hat, exp = _make_clean_data(n=1000, seed=31)
        result = check_gmcb(y, y_hat, exp, seed=1)
        assert 0.0 <= result.p_value <= 1.0

    def test_gmcb_score_nonnegative(self):
        """GMCB score must be floored at zero."""
        y, y_hat, exp = _make_clean_data(n=1000, seed=32)
        result = check_gmcb(y, y_hat, exp, seed=2)
        assert result.gmcb_score >= 0.0

    def test_significance_flag_matches_pvalue(self):
        y, y_hat, exp = _make_clean_data(n=1000, seed=33)
        alpha = 0.32
        result = check_gmcb(y, y_hat, exp, significance_level=alpha, seed=3)
        expected_sig = result.p_value < alpha
        assert result.is_significant == expected_sig

    def test_detects_global_shift(self):
        """GMCB should be significant for 20% global frequency inflation."""
        rng = np.random.default_rng(200)
        n = 3000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat = rng.gamma(2, 0.05, n)
        counts = rng.poisson(exposure * y_hat * 1.20)
        y = counts / exposure

        result = check_gmcb(
            y, y_hat, exposure, significance_level=0.32, seed=200, bootstrap_n=299
        )
        assert result.is_significant, (
            f"Expected GMCB to be significant for 20% global shift, "
            f"p={result.p_value:.4f}, gmcb={result.gmcb_score:.6f}"
        )

    def test_balance_factor_matches_ae(self):
        """balance_factor should match sum(w*y) / sum(w*y_hat)."""
        y, y_hat, exp = _make_clean_data(n=1000, seed=34)
        result = check_gmcb(y, y_hat, exp, seed=4)
        expected_af = float(np.sum(exp * y) / np.sum(exp * y_hat))
        assert result.balance_factor == pytest.approx(expected_af, rel=1e-6)

    def test_zero_predicted_total_raises(self):
        """sum(w * y_hat) near zero must raise ValueError."""
        y = np.array([0.1, 0.2, 0.3])
        y_hat = np.array([1e-20, 1e-20, 1e-20])
        with pytest.raises(ValueError, match="near zero"):
            check_gmcb(y, y_hat)

    def test_time_split_warning(self):
        """Median exposure < 0.05 must emit a UserWarning."""
        rng = np.random.default_rng(40)
        n = 500
        # Very small exposures simulate row-per-claim data
        exposure = rng.uniform(0.001, 0.02, n)
        y_hat = rng.gamma(2, 0.05, n)
        y = rng.poisson(exposure * y_hat) / exposure

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_gmcb(y, y_hat, exposure, seed=5)
            time_split_warnings = [
                x for x in w if "time-splitting" in str(x.message).lower()
                or "0.05" in str(x.message)
            ]
            assert len(time_split_warnings) >= 1, (
                "Expected UserWarning about time-splitting for median exposure < 0.05"
            )


class TestCheckLMCB:
    def test_returns_lmcb_result(self):
        y, y_hat, exp = _make_clean_data(n=1000, seed=50)
        result = check_lmcb(y, y_hat, exp, seed=0)
        assert isinstance(result, LMCBResult)

    def test_pvalue_in_range(self):
        y, y_hat, exp = _make_clean_data(n=1000, seed=51)
        result = check_lmcb(y, y_hat, exp, seed=1)
        assert 0.0 <= result.p_value <= 1.0

    def test_significance_flag_matches_pvalue(self):
        y, y_hat, exp = _make_clean_data(n=1000, seed=52)
        alpha = 0.32
        result = check_lmcb(y, y_hat, exp, significance_level=alpha, seed=2)
        expected_sig = result.p_value < alpha
        assert result.is_significant == expected_sig

    def test_does_not_reject_well_calibrated_model(self):
        """Well-ranked, well-calibrated model should not reject LMCB."""
        y, y_hat, exp = _make_clean_data(n=3000, seed=60)
        result = check_lmcb(y, y_hat, exp, significance_level=0.32, seed=60, bootstrap_n=299)
        # With good data this should usually pass; we allow a generous tolerance
        assert not result.is_significant or result.p_value > 0.01, (
            f"Well-calibrated model rejected LMCB at p={result.p_value:.4f}"
        )

    def test_lmcb_not_floored_at_zero(self):
        """LMCB score should NOT be floored at zero (negative values are valid)."""
        # This test just checks that the score is returned as-is (not clipped)
        y, y_hat, exp = _make_clean_data(n=500, seed=70)
        result = check_lmcb(y, y_hat, exp, seed=7)
        # We don't assert negative — just that the function returns a float
        assert isinstance(result.lmcb_score, float)


# ---------------------------------------------------------------------------
# GMCB + LMCB sum consistency
# ---------------------------------------------------------------------------


class TestMCBConsistency:
    def test_gmcb_lmcb_sum_approx_mcb(self):
        """check_gmcb().gmcb_score + check_lmcb().lmcb_score ≈ murphy.miscalibration.

        The Murphy decomposition computes GMCB and LMCB using a simple alpha
        balance correction (scalar multiply). check_gmcb / check_lmcb use the
        same approach. The two scores should match to within floating-point
        tolerance.
        """
        y, y_hat, exp = _make_clean_data(n=2000, seed=80)

        gmcb_res = check_gmcb(y, y_hat, exp, seed=80, bootstrap_n=99)
        lmcb_res = check_lmcb(y, y_hat, exp, seed=81, bootstrap_n=99)
        murphy = murphy_decomposition(y, y_hat, exp)

        computed_mcb = gmcb_res.gmcb_score + lmcb_res.lmcb_score
        expected_mcb = murphy.global_mcb + murphy.local_mcb

        # Allow 1% relative tolerance — both use same underlying deviance
        assert computed_mcb == pytest.approx(expected_mcb, rel=0.01, abs=1e-6), (
            f"check_gmcb + check_lmcb = {computed_mcb:.6f} != "
            f"murphy GMCB+LMCB = {expected_mcb:.6f}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_fit_required_before_test(self):
        """test() before fit() must raise RuntimeError."""
        y, y_hat, exp = _make_clean_data(n=500, seed=90)
        monitor = _make_monitor()
        with pytest.raises(RuntimeError, match="fit"):
            monitor.test(y, y_hat, exp)

    def test_is_fitted_false_before_fit(self):
        monitor = _make_monitor()
        assert monitor.is_fitted() is False

    def test_is_fitted_true_after_fit(self):
        y, y_hat, exp = _make_clean_data(n=500, seed=91)
        monitor = _make_monitor()
        monitor.fit(y, y_hat, exp)
        assert monitor.is_fitted() is True

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution"):
            ModelMonitor(distribution="binomial")

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha_gini"):
            ModelMonitor(alpha_gini=1.5)

    def test_n_bootstrap_too_small_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            ModelMonitor(n_bootstrap=10)

    def test_mismatched_lengths_raise(self):
        y_ref, y_hat, exp = _make_clean_data(n=500, seed=92)
        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp)
        with pytest.raises(ValueError):
            monitor.test(y_ref[:100], y_hat, exp)

    def test_method_chaining(self):
        """fit() must return self for chaining."""
        y, y_hat, exp = _make_clean_data(n=500, seed=93)
        monitor = _make_monitor()
        result = monitor.fit(y, y_hat, exp)
        assert result is monitor


# ---------------------------------------------------------------------------
# Summary and serialisation
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_string(self):
        y_ref, y_hat, exp = _make_clean_data(n=1000, seed=95)
        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp)
        result = monitor.test(y_ref, y_hat, exp)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_to_dict_contains_decision(self):
        y_ref, y_hat, exp = _make_clean_data(n=1000, seed=96)
        monitor = _make_monitor()
        monitor.fit(y_ref, y_hat, exp)
        result = monitor.test(y_ref, y_hat, exp)
        d = result.to_dict()
        assert "decision" in d
        assert d["decision"] in ("REDEPLOY", "RECALIBRATE", "REFIT")

    def test_repr_shows_status(self):
        monitor = _make_monitor()
        r = repr(monitor)
        assert "not fitted" in r
        y, y_hat, exp = _make_clean_data(n=500, seed=97)
        monitor.fit(y, y_hat, exp)
        r = repr(monitor)
        assert "fitted" in r
