"""
Tests for insurance_monitoring.business_value module.

Covers:
- Perfect model property: rho=1 -> E_LR=0, LR=1/M
- Monotonicity: lower rho -> higher E_LR and LR
- lre_compare field correctness and basis-points conversion
- calibrate_eta roundtrip: recover eta from LR computed with known params
- Edge cases: rho near zero, unit elasticity (eta=1), invalid inputs
- lre_compare with rho_new < rho_old (deterioration case)
"""
import math
import warnings

import pytest

from insurance_monitoring.business_value import (
    LREResult,
    calibrate_eta,
    loss_ratio,
    loss_ratio_error,
    lre_compare,
)


# ---------------------------------------------------------------------------
# loss_ratio_error
# ---------------------------------------------------------------------------


class TestLossRatioError:
    def test_perfect_model_zero_error(self):
        """rho=1 must produce E_LR=0 regardless of CV and eta."""
        assert loss_ratio_error(rho=1.0, cv=1.0, eta=1.0) == pytest.approx(0.0)
        assert loss_ratio_error(rho=1.0, cv=2.5, eta=1.5) == pytest.approx(0.0)

    def test_positive_error_for_imperfect_model(self):
        """Any rho < 1 should produce a positive loss ratio error."""
        e = loss_ratio_error(rho=0.90, cv=1.2, eta=1.5)
        assert e > 0.0

    def test_monotone_in_rho(self):
        """Lower rho must give higher error: E_LR(0.80) > E_LR(0.90) > E_LR(0.95)."""
        e_80 = loss_ratio_error(0.80, cv=1.2, eta=1.5)
        e_90 = loss_ratio_error(0.90, cv=1.2, eta=1.5)
        e_95 = loss_ratio_error(0.95, cv=1.2, eta=1.5)
        assert e_80 > e_90 > e_95

    def test_near_perfect_model(self):
        """rho=0.999 is treated as exact 1 — returns 0.0."""
        assert loss_ratio_error(rho=0.999, cv=1.0, eta=1.0) == 0.0

    def test_unit_elasticity(self):
        """eta=1 is a valid and common case; exponent = (2*1-1)/2 = 0.5."""
        e = loss_ratio_error(rho=0.90, cv=1.0, eta=1.0)
        # Manual: ratio_factor = (1 + 0.81) / (0.81 * 2) = 1.81/1.62
        ratio = (1 + 0.81 * 1.0) / (0.81 * 2.0)  # cv^{-2}=1 when cv=1
        expected = math.sqrt(ratio) - 1.0
        assert e == pytest.approx(expected, rel=1e-6)

    def test_eta_warning_at_boundary(self):
        """eta <= 0.5 should raise a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss_ratio_error(rho=0.90, cv=1.0, eta=0.4)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "eta" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# loss_ratio
# ---------------------------------------------------------------------------


class TestLossRatio:
    def test_perfect_model_equal_inverse_margin(self):
        """Perfect model (rho=1) must give LR = 1/M."""
        assert loss_ratio(1.0, cv=1.0, eta=1.5, margin=1.0) == pytest.approx(1.0)
        assert loss_ratio(1.0, cv=1.5, eta=1.5, margin=1.4286) == pytest.approx(
            1.0 / 1.4286, abs=1e-4
        )

    def test_imperfect_model_above_baseline(self):
        """LR(rho < 1) must exceed 1/M."""
        lr_perfect = 1.0 / 1.4286
        lr = loss_ratio(0.90, cv=1.2, eta=1.5, margin=1.4286)
        assert lr > lr_perfect

    def test_monotone_in_rho(self):
        """Higher rho -> lower LR (better performance)."""
        lr_80 = loss_ratio(0.80, cv=1.2, eta=1.5)
        lr_90 = loss_ratio(0.90, cv=1.2, eta=1.5)
        lr_95 = loss_ratio(0.95, cv=1.2, eta=1.5)
        assert lr_80 > lr_90 > lr_95

    def test_consistency_with_lre(self):
        """loss_ratio should equal 1 + loss_ratio_error when margin=1."""
        rho, cv, eta = 0.88, 1.3, 1.8
        lr = loss_ratio(rho, cv, eta, margin=1.0)
        e_lr = loss_ratio_error(rho, cv, eta)
        assert lr == pytest.approx(1.0 + e_lr, rel=1e-10)

    def test_rho_near_zero_large_lr(self):
        """Very low rho (but > 0) must produce a very large LR."""
        lr = loss_ratio(0.01, cv=1.0, eta=1.5)
        assert lr > 10.0

    def test_high_cv_higher_lr(self):
        """Higher CV amplifies the effect of model imperfection."""
        lr_low_cv = loss_ratio(0.88, cv=0.5, eta=1.5)
        lr_high_cv = loss_ratio(0.88, cv=2.0, eta=1.5)
        assert lr_high_cv > lr_low_cv

    def test_margin_scales_result(self):
        """LR should scale inversely with margin: doubling M halves LR."""
        lr1 = loss_ratio(0.90, cv=1.0, eta=1.5, margin=1.0)
        lr2 = loss_ratio(0.90, cv=1.0, eta=1.5, margin=2.0)
        assert lr1 == pytest.approx(2.0 * lr2, rel=1e-8)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize("rho", [-0.1, 0.0, 1.1, 2.0])
    def test_invalid_rho(self, rho):
        with pytest.raises(ValueError, match="rho"):
            loss_ratio_error(rho=rho, cv=1.0, eta=1.0)

    @pytest.mark.parametrize("cv", [0.0, -1.0, -0.001])
    def test_invalid_cv(self, cv):
        with pytest.raises(ValueError, match="cv"):
            loss_ratio_error(rho=0.9, cv=cv, eta=1.0)

    @pytest.mark.parametrize("eta", [0.0, -0.5])
    def test_invalid_eta(self, eta):
        with pytest.raises(ValueError, match="eta"):
            loss_ratio_error(rho=0.9, cv=1.0, eta=eta)

    @pytest.mark.parametrize("margin", [0.0, -1.0])
    def test_invalid_margin(self, margin):
        with pytest.raises(ValueError, match="margin"):
            loss_ratio(rho=0.9, cv=1.0, eta=1.0, margin=margin)

    def test_invalid_rho_old_lre_compare(self):
        with pytest.raises(ValueError, match="rho_old"):
            lre_compare(rho_old=0.0, rho_new=0.9, cv=1.0, eta=1.0)

    def test_invalid_rho_new_lre_compare(self):
        with pytest.raises(ValueError, match="rho_new"):
            lre_compare(rho_old=0.9, rho_new=1.1, cv=1.0, eta=1.0)


# ---------------------------------------------------------------------------
# lre_compare
# ---------------------------------------------------------------------------


class TestLreCompare:
    def test_returns_lre_result(self):
        result = lre_compare(0.90, 0.95, cv=1.2, eta=1.5)
        assert isinstance(result, LREResult)

    def test_improvement_negative_delta(self):
        """Higher rho_new -> lower LR -> negative delta_lr."""
        result = lre_compare(0.90, 0.95, cv=1.2, eta=1.5)
        assert result.delta_lr < 0.0
        assert result.delta_lr_bps < 0.0

    def test_deterioration_positive_delta(self):
        """rho_new < rho_old gives positive delta_lr (deterioration)."""
        result = lre_compare(0.95, 0.85, cv=1.2, eta=1.5)
        assert result.delta_lr > 0.0
        assert result.delta_lr_bps > 0.0

    def test_same_rho_zero_delta(self):
        result = lre_compare(0.90, 0.90, cv=1.2, eta=1.5)
        assert result.delta_lr == pytest.approx(0.0, abs=1e-12)
        assert result.delta_lr_bps == pytest.approx(0.0, abs=1e-8)

    def test_delta_lr_bps_conversion(self):
        """delta_lr_bps must equal delta_lr * 10_000."""
        result = lre_compare(0.90, 0.95, cv=1.2, eta=1.5)
        assert result.delta_lr_bps == pytest.approx(result.delta_lr * 10_000.0, rel=1e-10)

    def test_lr_consistency(self):
        """lr_old and lr_new should match standalone loss_ratio calls."""
        rho_old, rho_new, cv, eta, margin = 0.88, 0.93, 1.3, 1.8, 1.4286
        result = lre_compare(rho_old, rho_new, cv, eta, margin)
        assert result.lr_old == pytest.approx(
            loss_ratio(rho_old, cv, eta, margin), rel=1e-10
        )
        assert result.lr_new == pytest.approx(
            loss_ratio(rho_new, cv, eta, margin), rel=1e-10
        )

    def test_e_lr_consistency(self):
        """e_lr_old and e_lr_new should match standalone loss_ratio_error calls."""
        rho_old, rho_new, cv, eta = 0.88, 0.93, 1.3, 1.8
        result = lre_compare(rho_old, rho_new, cv, eta)
        assert result.e_lr_old == pytest.approx(
            loss_ratio_error(rho_old, cv, eta), rel=1e-10
        )
        assert result.e_lr_new == pytest.approx(
            loss_ratio_error(rho_new, cv, eta), rel=1e-10
        )

    def test_perfect_new_model(self):
        """rho_new=1 -> e_lr_new=0, lr_new=1/margin."""
        result = lre_compare(0.90, 1.0, cv=1.2, eta=1.5, margin=1.4286)
        assert result.e_lr_new == pytest.approx(0.0)
        assert result.lr_new == pytest.approx(1.0 / 1.4286, abs=1e-4)

    def test_magnitude_typical_uk_motor(self):
        """
        Sense check with typical UK motor parameters.
        rho=0.92->0.95, CV=1.2, eta=1.5 should give a non-trivial improvement
        of at least 10 bps.
        """
        result = lre_compare(0.92, 0.95, cv=1.2, eta=1.5)
        assert result.delta_lr_bps < -10.0, (
            f"Expected >10 bps improvement for rho 0.92->0.95, got {result.delta_lr_bps:.1f} bps"
        )

    def test_repr_contains_bps(self):
        result = lre_compare(0.90, 0.95, cv=1.2, eta=1.5)
        assert "bps" in repr(result)


# ---------------------------------------------------------------------------
# calibrate_eta
# ---------------------------------------------------------------------------


class TestCalibrateEta:
    def test_roundtrip_eta_1(self):
        """Compute LR with eta=1.0, then recover eta=1.0 from that LR."""
        true_eta = 1.0
        rho, cv, margin = 0.88, 1.2, 1.0
        lr_from_known = loss_ratio(rho, cv, true_eta, margin)
        eta_recovered = calibrate_eta(rho, cv, lr_from_known, margin)
        assert eta_recovered is not None
        assert eta_recovered == pytest.approx(true_eta, abs=1e-6)

    def test_roundtrip_eta_1_5(self):
        """Roundtrip with eta=1.5."""
        true_eta = 1.5
        rho, cv, margin = 0.91, 1.4, 1.2
        lr_from_known = loss_ratio(rho, cv, true_eta, margin)
        eta_recovered = calibrate_eta(rho, cv, lr_from_known, margin)
        assert eta_recovered is not None
        assert eta_recovered == pytest.approx(true_eta, abs=1e-5)

    def test_roundtrip_eta_2_5(self):
        """Roundtrip with a higher elasticity."""
        true_eta = 2.5
        rho, cv, margin = 0.85, 0.8, 1.0
        lr_from_known = loss_ratio(rho, cv, true_eta, margin)
        eta_recovered = calibrate_eta(rho, cv, lr_from_known, margin)
        assert eta_recovered is not None
        assert eta_recovered == pytest.approx(true_eta, abs=1e-5)

    def test_returns_none_when_out_of_bounds(self):
        """An LR that is not achievable within the default bounds returns None."""
        # LR = 1.0 at rho=1; for rho=0.9 with margin=1, LR > 1. So LR=0.5
        # is not achievable with eta in (0.5, 5.0).
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = calibrate_eta(0.90, cv=1.2, lr_observed=0.5, margin=1.0)
        assert result is None

    def test_invalid_lr_observed(self):
        with pytest.raises(ValueError, match="lr_observed"):
            calibrate_eta(0.90, cv=1.2, lr_observed=0.0)

    def test_invalid_rho(self):
        with pytest.raises(ValueError, match="rho"):
            calibrate_eta(0.0, cv=1.2, lr_observed=1.1)


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


def test_all_exports():
    """Check that all documented names are exported."""
    import insurance_monitoring.business_value as bv

    for name in ["loss_ratio_error", "loss_ratio", "lre_compare", "calibrate_eta", "LREResult"]:
        assert hasattr(bv, name), f"{name} missing from business_value"
    assert set(bv.__all__) >= {"loss_ratio_error", "loss_ratio", "lre_compare", "calibrate_eta", "LREResult"}
