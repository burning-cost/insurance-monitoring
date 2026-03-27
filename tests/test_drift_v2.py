"""Extended tests for insurance_monitoring.drift — covers gaps in existing test_drift.py.

Targets:
- PSI with reference_exposure (fully symmetric)
- PSI with n_bins=2 (minimum)
- PSI raises for n_bins < 2
- PSI raises for mismatched reference_exposure
- PSI raises for zero reference_exposure sum
- CSI with pandas DataFrames
- CSI with missing feature in current_df
- KS test returns dict structure
- KS test with polars on both sides
- Wasserstein with different distributions
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.drift import csi, ks_test, psi, wasserstein_distance


class TestPSIExtended:
    def test_reference_exposure_changes_proportions(self):
        """reference_exposure should change reference bin proportions."""
        rng = np.random.default_rng(10)
        ref = rng.normal(30, 5, 5_000)
        cur = rng.normal(35, 5, 2_000)
        ref_exp = np.where(ref > 30, 2.0, 0.5)

        psi_no_ref_exp = psi(ref, cur)
        psi_with_ref_exp = psi(ref, cur, reference_exposure=ref_exp)
        # When reference exposure is asymmetric, PSI values should differ
        assert abs(psi_no_ref_exp - psi_with_ref_exp) > 0.001

    def test_symmetric_psi_both_exposures(self):
        """Both reference_exposure and exposure_weights provided should run without error."""
        rng = np.random.default_rng(11)
        ref = rng.normal(30, 5, 3_000)
        cur = rng.normal(32, 5, 1_500)
        ref_exp = rng.uniform(0.5, 1.5, 3_000)
        cur_exp = rng.uniform(0.5, 1.5, 1_500)
        result = psi(ref, cur, exposure_weights=cur_exp, reference_exposure=ref_exp)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_n_bins_minimum_2(self):
        """n_bins=2 should work without error."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        cur = np.array([2.0, 3.0, 4.0, 5.0, 6.0] * 50)
        result = psi(ref, cur, n_bins=2)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_n_bins_less_than_2_raises(self):
        """n_bins < 2 should raise ValueError."""
        ref = np.ones(100)
        cur = np.ones(50)
        with pytest.raises(ValueError, match="n_bins"):
            psi(ref, cur, n_bins=1)

    def test_mismatched_reference_exposure_raises(self):
        """Mismatched reference_exposure length should raise ValueError."""
        ref = np.ones(100)
        cur = np.ones(50)
        ref_exp = np.ones(80)  # wrong length
        with pytest.raises(ValueError, match="reference_exposure length"):
            psi(ref, cur, reference_exposure=ref_exp)

    def test_zero_sum_reference_exposure_raises(self):
        """Zero total reference_exposure should raise ValueError."""
        ref = np.ones(50)
        cur = np.ones(30)
        ref_exp = np.zeros(50)  # zero sum
        with pytest.raises(ValueError, match="zero"):
            psi(ref, cur, reference_exposure=ref_exp)

    def test_zero_sum_exposure_weights_raises(self):
        """Zero total exposure_weights should raise ValueError."""
        ref = np.ones(50)
        cur = np.ones(30)
        weights = np.zeros(30)  # zero sum
        with pytest.raises(ValueError, match="zero"):
            psi(ref, cur, exposure_weights=weights)

    def test_large_n_bins(self):
        """n_bins=20 should produce a valid non-negative PSI."""
        rng = np.random.default_rng(12)
        ref = rng.normal(0, 1, 10_000)
        cur = rng.normal(1, 1, 5_000)
        result = psi(ref, cur, n_bins=20)
        assert result >= 0.0

    def test_psi_increases_with_shift(self):
        """Larger mean shift should produce larger PSI."""
        rng = np.random.default_rng(13)
        ref = rng.normal(30, 5, 5_000)
        cur_small = rng.normal(31, 5, 2_000)
        cur_large = rng.normal(40, 5, 2_000)
        assert psi(ref, cur_large) > psi(ref, cur_small)


class TestCSIExtended:
    def test_csi_missing_feature_in_current_raises(self):
        """CSI should raise ValueError if feature missing from current_df."""
        ref = pl.DataFrame({"driver_age": [30, 35, 40], "vehicle_age": [2, 3, 4]})
        cur = pl.DataFrame({"driver_age": [31, 36, 41]})  # missing vehicle_age
        with pytest.raises(ValueError, match="not found in current_df"):
            csi(ref, cur, features=["vehicle_age"])

    def test_csi_all_bands_valid(self):
        """All bands should be one of: green, amber, red."""
        rng = np.random.default_rng(20)
        features = ["f1", "f2", "f3"]
        ref = pl.DataFrame({f: rng.normal(0, 1, 5_000).tolist() for f in features})
        # Mix: some stable, some drifted
        cur_data = {
            "f1": rng.normal(0, 1, 2_000).tolist(),     # stable
            "f2": rng.normal(5, 1, 2_000).tolist(),     # large drift
            "f3": rng.normal(0.5, 1, 2_000).tolist(),   # small drift
        }
        cur = pl.DataFrame(cur_data)
        result = csi(ref, cur, features=features)
        valid_bands = {"green", "amber", "red"}
        for band in result["band"].to_list():
            assert band in valid_bands, f"Invalid band: {band}"

    def test_csi_n_bins_passed_through(self):
        """n_bins parameter should be accepted and propagated."""
        rng = np.random.default_rng(21)
        ref = pl.DataFrame({"x": rng.normal(0, 1, 2_000).tolist()})
        cur = pl.DataFrame({"x": rng.normal(0, 1, 1_000).tolist()})
        result_5 = csi(ref, cur, features=["x"], n_bins=5)
        result_20 = csi(ref, cur, features=["x"], n_bins=20)
        # Both should produce valid output
        assert len(result_5) == 1
        assert len(result_20) == 1

    def test_csi_with_integer_features(self):
        """CSI should work with integer-coded categorical features."""
        rng = np.random.default_rng(22)
        ref = pl.DataFrame({"region_code": rng.integers(1, 6, 2_000).tolist()})
        cur = pl.DataFrame({"region_code": rng.integers(2, 7, 1_000).tolist()})
        result = csi(ref, cur, features=["region_code"])
        assert isinstance(result, pl.DataFrame)
        assert result["csi"][0] >= 0.0


class TestKSTestExtended:
    def test_ks_result_structure(self):
        """KS test result should be a dict with exactly the right keys."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = ks_test(ref, cur)
        assert set(result.keys()) == {"statistic", "p_value", "significant"}

    def test_ks_statistic_between_0_and_1(self):
        """KS statistic should always be in [0, 1]."""
        rng = np.random.default_rng(30)
        for _ in range(5):
            ref = rng.normal(0, 1, 200)
            cur = rng.normal(1, 1, 200)
            result = ks_test(ref, cur)
            assert 0.0 <= result["statistic"] <= 1.0

    def test_ks_significant_flag_matches_pvalue(self):
        """significant should be True iff p_value < 0.05."""
        rng = np.random.default_rng(31)
        ref = rng.normal(0, 1, 1_000)
        cur = rng.normal(10, 1, 1_000)  # obvious drift
        result = ks_test(ref, cur)
        assert result["significant"] == (result["p_value"] < 0.05)

    def test_ks_empty_current_raises(self):
        """KS test with empty current should raise ValueError."""
        with pytest.raises(ValueError):
            ks_test(np.array([1.0, 2.0, 3.0]), np.array([]))

    def test_ks_returns_float_values(self):
        """All values in KS result should be Python floats or bool."""
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.5, 2.5, 3.5])
        result = ks_test(ref, cur)
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["significant"], bool)


class TestWassersteinExtended:
    def test_wasserstein_larger_shift_larger_distance(self):
        """Larger distributional shift should produce larger Wasserstein distance."""
        rng = np.random.default_rng(40)
        ref = rng.normal(0, 1, 5_000)
        cur_small = rng.normal(1, 1, 2_000)
        cur_large = rng.normal(10, 1, 2_000)
        assert wasserstein_distance(ref, cur_large) > wasserstein_distance(ref, cur_small)

    def test_wasserstein_empty_current_raises(self):
        """Empty current should raise ValueError."""
        with pytest.raises(ValueError):
            wasserstein_distance(np.array([1.0, 2.0]), np.array([]))

    def test_wasserstein_symmetric(self):
        """Wasserstein distance is symmetric: d(A, B) == d(B, A)."""
        rng = np.random.default_rng(41)
        a = rng.normal(0, 1, 1_000)
        b = rng.normal(2, 1, 1_000)
        assert wasserstein_distance(a, b) == pytest.approx(wasserstein_distance(b, a), rel=1e-10)

    def test_wasserstein_integer_inputs(self):
        """Wasserstein should accept integer arrays."""
        ref = np.array([1, 2, 3, 4, 5])
        cur = np.array([2, 3, 4, 5, 6])
        result = wasserstein_distance(ref, cur)
        assert result == pytest.approx(1.0, abs=1e-8)
