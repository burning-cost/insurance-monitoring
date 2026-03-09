"""Tests for v0.2.0 discrimination additions: gini_drift_test_onesample, alpha parameter."""

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.discrimination import (
    gini_coefficient,
    gini_drift_test,
    gini_drift_test_onesample,
)


# ---------------------------------------------------------------------------
# One-sample Gini drift test (Algorithm 3, arXiv 2510.04556)
# ---------------------------------------------------------------------------


class TestGiniDriftTestOnesample:
    """Tests for the Algorithm 3 one-sample monitoring design."""

    def test_returns_required_keys(self):
        """Result dict must have all documented keys."""
        rng = np.random.default_rng(100)
        n = 2_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        training_gini = gini_coefficient(act, pred)

        result = gini_drift_test_onesample(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
        )
        required_keys = {
            "z_statistic",
            "p_value",
            "training_gini",
            "monitor_gini",
            "gini_change",
            "se_bootstrap",
            "significant",
        }
        assert required_keys == set(result.keys())

    def test_identical_data_large_pvalue(self):
        """Same data as training should produce large p-value (no drift)."""
        rng = np.random.default_rng(101)
        n = 4_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        training_gini = gini_coefficient(act, pred)

        result = gini_drift_test_onesample(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
        )
        # No drift: p-value should be large (z close to 0)
        assert abs(result["z_statistic"]) < 2.0, (
            f"z-stat should be small for identical data, got {result['z_statistic']}"
        )
        assert result["gini_change"] == pytest.approx(0.0, abs=1e-10)
        assert result["significant"] is False

    def test_severe_drift_detects_signal(self):
        """A training_gini far above monitor data should produce significant z."""
        rng = np.random.default_rng(102)
        n = 3_000
        # Monitor data with random predictions → very low Gini
        pred_random = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(rng.uniform(0.05, 0.20, n)).astype(float)

        monitor_gini = gini_coefficient(act, pred_random)

        # training_gini is 0.50 — far above what a random model can achieve
        result = gini_drift_test_onesample(
            training_gini=0.50,
            monitor_actual=act,
            monitor_predicted=pred_random,
            n_bootstrap=200,
            alpha=0.32,
        )
        # Should flag significant drift at alpha=0.32
        assert result["gini_change"] < 0, "Monitor Gini should be below training Gini"
        assert result["z_statistic"] < 0, "z-stat should be negative (degradation)"
        assert result["significant"] is True, (
            f"Should detect drift: training_gini=0.50, monitor_gini={monitor_gini:.3f}"
        )

    def test_alpha_controls_significant_flag(self):
        """Smaller alpha should give fewer significant flags for the same data."""
        rng = np.random.default_rng(103)
        n = 2_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        # Mildly drifted training Gini (slightly above monitor)
        monitor_gini = gini_coefficient(act, pred)
        training_gini = monitor_gini + 0.05  # small drift

        result_32 = gini_drift_test_onesample(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=200,
            alpha=0.32,
        )
        result_05 = gini_drift_test_onesample(
            training_gini=training_gini,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=200,
            alpha=0.05,
        )
        # alpha=0.32 should catch more signals than alpha=0.05
        # At minimum: if alpha=0.05 is significant, alpha=0.32 must also be
        if result_05["significant"]:
            assert result_32["significant"] is True

    def test_gini_change_equals_monitor_minus_training(self):
        """gini_change must equal monitor_gini - training_gini."""
        rng = np.random.default_rng(104)
        n = 1_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        result = gini_drift_test_onesample(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
        )
        expected_change = result["monitor_gini"] - result["training_gini"]
        assert result["gini_change"] == pytest.approx(expected_change, abs=1e-10)

    def test_training_gini_stored_in_result(self):
        """training_gini in result must match what was passed."""
        rng = np.random.default_rng(105)
        n = 500
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        result = gini_drift_test_onesample(
            training_gini=0.42,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=30,
        )
        assert result["training_gini"] == pytest.approx(0.42)

    def test_polars_input_accepted(self):
        """Should accept Polars Series as monitor_actual and monitor_predicted."""
        rng = np.random.default_rng(106)
        n = 500
        pred = pl.Series(rng.uniform(0.05, 0.20, n).tolist())
        act = pl.Series(rng.poisson(0.10, n).astype(float).tolist())

        result = gini_drift_test_onesample(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=30,
        )
        assert isinstance(result, dict)
        assert "monitor_gini" in result

    def test_exposure_accepted(self):
        """Should accept optional exposure weights."""
        rng = np.random.default_rng(107)
        n = 1_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        exp = rng.uniform(0.1, 2.0, n)

        result = gini_drift_test_onesample(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            monitor_exposure=exp,
            n_bootstrap=50,
        )
        assert isinstance(result, dict)
        assert result["se_bootstrap"] > 0

    def test_empty_monitor_raises(self):
        """Empty monitor_actual should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            gini_drift_test_onesample(
                training_gini=0.45,
                monitor_actual=np.array([]),
                monitor_predicted=np.array([]),
                n_bootstrap=10,
            )

    def test_mismatched_lengths_raises(self):
        """Mismatched actual/predicted lengths should raise ValueError."""
        with pytest.raises(ValueError):
            gini_drift_test_onesample(
                training_gini=0.45,
                monitor_actual=np.array([1.0, 0.0, 1.0]),
                monitor_predicted=np.array([0.1, 0.9]),
                n_bootstrap=10,
            )

    def test_se_bootstrap_positive(self):
        """Bootstrap SE should be positive for real data."""
        rng = np.random.default_rng(108)
        n = 500
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        result = gini_drift_test_onesample(
            training_gini=0.40,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=100,
        )
        assert result["se_bootstrap"] > 0.0

    def test_p_value_in_range(self):
        """p-value should be in [0, 1]."""
        rng = np.random.default_rng(109)
        n = 1_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)

        result = gini_drift_test_onesample(
            training_gini=0.45,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=50,
        )
        assert 0.0 <= result["p_value"] <= 1.0

    def test_default_alpha_is_0_32(self):
        """Default alpha=0.32: significant flag is True iff p_value < 0.32.

        The default is verified by checking that the significant flag is
        consistent with the returned p_value and the documented default of 0.32.
        We do not compare two separate bootstrap runs (they use different random
        seeds, so p-values are not identical), but instead verify the flag logic.
        """
        rng = np.random.default_rng(110)
        n = 2_000
        pred = rng.uniform(0.05, 0.20, n)
        act = rng.poisson(pred).astype(float)
        monitor_gini = gini_coefficient(act, pred)

        # Large drift: training_gini well above monitor → p_value should be small
        result = gini_drift_test_onesample(
            training_gini=monitor_gini + 0.15,
            monitor_actual=act,
            monitor_predicted=pred,
            n_bootstrap=300,
        )
        p = result["p_value"]
        sig = result["significant"]

        # Verify the flag is internally consistent with alpha=0.32
        assert sig == (p < 0.32), (
            f"significant={sig} but p_value={p:.4f}: flag is inconsistent with alpha=0.32"
        )
        # For a 0.15 Gini drop with n=2000, p should be well below 0.32
        assert p < 0.32, f"Expected significant drift at alpha=0.32, got p={p:.4f}"
        assert sig is True


# ---------------------------------------------------------------------------
# Alpha parameter on gini_drift_test (two-sample)
# ---------------------------------------------------------------------------


class TestGiniDriftTestAlpha:
    """Tests for the new alpha parameter on the two-sample test."""

    def test_alpha_parameter_exists(self):
        """gini_drift_test should accept alpha parameter."""
        result = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.40,
            n_reference=5_000,
            n_current=5_000,
            reference_variance=0.001,
            current_variance=0.001,
            alpha=0.32,
        )
        assert "significant" in result

    def test_alpha_0_32_more_sensitive_than_0_05(self):
        """At alpha=0.32, the significant flag should trigger on weaker evidence."""
        # A small drift that is borderline significant
        result_32 = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.43,
            n_reference=5_000,
            n_current=5_000,
            reference_variance=0.0004,
            current_variance=0.0004,
            alpha=0.32,
        )
        result_05 = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.43,
            n_reference=5_000,
            n_current=5_000,
            reference_variance=0.0004,
            current_variance=0.0004,
            alpha=0.05,
        )
        # If alpha=0.05 catches it, alpha=0.32 must too
        if result_05["significant"]:
            assert result_32["significant"] is True

    def test_default_alpha_is_0_32(self):
        """Default alpha in gini_drift_test should now be 0.32."""
        # Craft a drift where p_value is between 0.05 and 0.32 (deterministic
        # because we use precomputed variance, not bootstrap).
        # z = (G_cur - G_ref) / SE = -0.0268 / sqrt(0.0002 + 0.0002) = -0.0268/0.02 = -1.34
        # p-value = 2*(1 - Phi(1.34)) ≈ 0.18 — above 0.05, below 0.32
        result = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.45 - 0.0268,
            n_reference=10_000,
            n_current=10_000,
            reference_variance=0.0002,
            current_variance=0.0002,
        )
        p = result["p_value"]
        # Verify p is in the 0.05-0.32 window to make the test meaningful
        assert 0.05 < p < 0.32, (
            f"p_value={p:.4f} is not in (0.05, 0.32); test case needs adjustment"
        )
        # Default alpha=0.32: flag should be True
        assert result["significant"] is True

    def test_significant_flag_consistent_with_p_and_alpha(self):
        """significant == (p_value < alpha) must hold exactly."""
        # Precomputed variance → deterministic p-value
        result_32 = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.42,
            n_reference=5_000,
            n_current=5_000,
            reference_variance=0.001,
            current_variance=0.001,
            alpha=0.32,
        )
        assert result_32["significant"] == (result_32["p_value"] < 0.32)

        result_05 = gini_drift_test(
            reference_gini=0.45,
            current_gini=0.42,
            n_reference=5_000,
            n_current=5_000,
            reference_variance=0.001,
            current_variance=0.001,
            alpha=0.05,
        )
        assert result_05["significant"] == (result_05["p_value"] < 0.05)
