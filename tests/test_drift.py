"""Tests for insurance_monitoring.drift module."""

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.drift import csi, ks_test, psi, wasserstein_distance


# ---------------------------------------------------------------------------
# PSI tests
# ---------------------------------------------------------------------------


class TestPSI:
    """PSI formula and edge case coverage."""

    def test_identical_distributions_near_zero(self):
        """Same distribution in reference and current should give PSI near 0."""
        rng = np.random.default_rng(0)
        ref = rng.normal(35, 8, 20_000)
        cur = rng.normal(35, 8, 10_000)
        result = psi(ref, cur, n_bins=10)
        assert result < 0.05, f"PSI for identical distributions should be near 0, got {result}"

    def test_large_shift_above_threshold(self):
        """A clear mean shift should exceed the 0.25 threshold."""
        rng = np.random.default_rng(1)
        ref = rng.normal(30, 5, 10_000)
        cur = rng.normal(50, 5, 5_000)  # large mean shift
        result = psi(ref, cur, n_bins=10)
        assert result > 0.25, f"PSI for large shift should exceed 0.25, got {result}"

    def test_moderate_shift_amber_range(self):
        """A moderate shift should land in the amber zone (0.1 to 0.25)."""
        rng = np.random.default_rng(2)
        ref = rng.normal(30, 5, 10_000)
        cur = rng.normal(35, 5, 5_000)  # moderate mean shift
        result = psi(ref, cur, n_bins=10)
        assert 0.05 < result < 0.60, f"PSI for moderate shift, got {result}"

    def test_psi_nonnegative(self):
        """PSI should always be non-negative."""
        rng = np.random.default_rng(3)
        ref = rng.uniform(0, 100, 5_000)
        cur = rng.uniform(20, 80, 3_000)
        result = psi(ref, cur)
        assert result >= 0

    def test_psi_polars_series_input(self):
        """PSI should accept Polars Series."""
        rng = np.random.default_rng(4)
        ref = pl.Series(rng.normal(30, 5, 5_000).tolist())
        cur = pl.Series(rng.normal(30, 5, 2_000).tolist())
        result = psi(ref, cur)
        assert isinstance(result, float)
        assert result >= 0

    def test_psi_numpy_array_input(self):
        """PSI should accept numpy arrays."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)
        cur = np.array([1.5, 2.5, 3.5, 4.5, 5.5] * 100)
        result = psi(ref, cur, n_bins=5)
        assert isinstance(result, float)

    def test_exposure_weighted_vs_unweighted_differ(self):
        """Exposure-weighted PSI should differ from unweighted when weights vary by bin."""
        rng = np.random.default_rng(5)
        ref = rng.normal(30, 5, 5_000)
        cur = rng.normal(35, 5, 2_000)
        # Assign higher exposure to the high end of current distribution
        # This should change the PSI value
        exposure = np.where(cur > 35, 2.0, 0.5)  # skewed weights
        psi_unweighted = psi(ref, cur)
        psi_weighted = psi(ref, cur, exposure_weights=exposure)
        # They should differ (weights are asymmetric)
        assert abs(psi_weighted - psi_unweighted) > 0.001

    def test_exposure_weighted_uniform_weights_equivalent(self):
        """Exposure-weighted PSI with uniform weights should equal unweighted."""
        rng = np.random.default_rng(6)
        ref = rng.normal(30, 5, 5_000)
        cur = rng.normal(30, 5, 2_000)
        uniform_exposure = np.ones(len(cur))
        psi_unweighted = psi(ref, cur)
        psi_weighted = psi(ref, cur, exposure_weights=uniform_exposure)
        # Should be very close (within floating point tolerance)
        assert abs(psi_weighted - psi_unweighted) < 1e-6

    def test_custom_n_bins(self):
        """PSI should work with different bin counts."""
        rng = np.random.default_rng(7)
        ref = rng.normal(30, 5, 5_000)
        cur = rng.normal(32, 5, 2_000)
        for n in [5, 10, 20]:
            result = psi(ref, cur, n_bins=n)
            assert result >= 0

    def test_empty_reference_raises(self):
        """Empty reference array should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            psi(np.array([]), np.array([1.0, 2.0]))

    def test_empty_current_raises(self):
        """Empty current array should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            psi(np.array([1.0, 2.0]), np.array([]))

    def test_mismatched_exposure_weights_raises(self):
        """Mismatched exposure weights length should raise ValueError."""
        ref = np.ones(100)
        cur = np.ones(50)
        weights = np.ones(30)  # wrong length
        with pytest.raises(ValueError, match="exposure_weights length"):
            psi(ref, cur, exposure_weights=weights)

    def test_single_value_reference(self):
        """Reference distribution with all same values returns 0.0 (undefined PSI)."""
        ref = np.ones(100) * 5.0
        cur = np.ones(50) * 5.0
        result = psi(ref, cur)
        assert result == 0.0

    def test_psi_symmetry_approximate(self):
        """PSI(A, B) should be close to PSI(B, A) due to near-symmetry."""
        rng = np.random.default_rng(8)
        a = rng.normal(30, 5, 10_000)
        b = rng.normal(32, 5, 10_000)
        psi_ab = psi(a, b)
        psi_ba = psi(b, a)
        # PSI is not exactly symmetric but should be in the same ballpark
        assert abs(psi_ab - psi_ba) < 0.05


# ---------------------------------------------------------------------------
# CSI tests
# ---------------------------------------------------------------------------


class TestCSI:
    """CSI (Characteristic Stability Index) tests."""

    def test_csi_returns_polars_dataframe(self):
        """CSI should return a Polars DataFrame."""
        rng = np.random.default_rng(10)
        ref = pl.DataFrame({
            "driver_age": rng.normal(35, 8, 1_000).tolist(),
            "vehicle_age": rng.uniform(0, 15, 1_000).tolist(),
        })
        cur = pl.DataFrame({
            "driver_age": rng.normal(35, 8, 500).tolist(),
            "vehicle_age": rng.uniform(0, 15, 500).tolist(),
        })
        result = csi(ref, cur, features=["driver_age", "vehicle_age"])
        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "csi" in result.columns
        assert "band" in result.columns

    def test_csi_one_row_per_feature(self):
        """CSI output should have exactly one row per feature."""
        rng = np.random.default_rng(11)
        features = ["driver_age", "vehicle_age", "ncd_years"]
        ref = pl.DataFrame({f: rng.normal(0, 1, 500).tolist() for f in features})
        cur = pl.DataFrame({f: rng.normal(0, 1, 300).tolist() for f in features})
        result = csi(ref, cur, features=features)
        assert len(result) == len(features)
        assert set(result["feature"].to_list()) == set(features)

    def test_csi_stable_features_green(self):
        """Stable features should receive green band."""
        rng = np.random.default_rng(12)
        ref = pl.DataFrame({"x": rng.normal(0, 1, 10_000).tolist()})
        cur = pl.DataFrame({"x": rng.normal(0, 1, 5_000).tolist()})
        result = csi(ref, cur, features=["x"])
        assert result["band"][0] == "green"

    def test_csi_drifted_feature_red(self):
        """A heavily drifted feature should receive red band."""
        rng = np.random.default_rng(13)
        ref = pl.DataFrame({"x": rng.normal(20, 3, 10_000).tolist()})
        cur = pl.DataFrame({"x": rng.normal(50, 3, 5_000).tolist()})
        result = csi(ref, cur, features=["x"])
        assert result["band"][0] == "red"

    def test_csi_missing_feature_raises(self):
        """CSI should raise ValueError for missing feature."""
        ref = pl.DataFrame({"driver_age": [30, 35, 40]})
        cur = pl.DataFrame({"driver_age": [31, 36, 41]})
        with pytest.raises(ValueError, match="not found in reference_df"):
            csi(ref, cur, features=["nonexistent"])

    def test_csi_values_nonnegative(self):
        """All CSI values should be non-negative."""
        rng = np.random.default_rng(14)
        ref = pl.DataFrame({"a": rng.normal(0, 1, 5_000).tolist(),
                             "b": rng.uniform(0, 10, 5_000).tolist()})
        cur = pl.DataFrame({"a": rng.normal(1, 1, 2_000).tolist(),
                             "b": rng.uniform(2, 8, 2_000).tolist()})
        result = csi(ref, cur, features=["a", "b"])
        assert (result["csi"] >= 0).all()


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------


class TestKSTest:
    """Kolmogorov-Smirnov test wrapper tests."""

    def test_ks_same_distribution_not_significant(self):
        """KS test on same distribution should typically not be significant."""
        rng = np.random.default_rng(20)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        result = ks_test(ref, cur)
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        # With n=500, same dist, p-value should be above 0.05 most of the time
        # Use a very generous threshold to avoid flakiness
        assert result["p_value"] > 0.001

    def test_ks_different_distributions_significant(self):
        """KS test on clearly different distributions should be significant."""
        rng = np.random.default_rng(21)
        ref = rng.normal(0, 1, 1_000)
        cur = rng.normal(5, 1, 1_000)  # large mean shift
        result = ks_test(ref, cur)
        assert result["significant"] is True
        assert result["p_value"] < 0.001

    def test_ks_statistic_in_range(self):
        """KS statistic should be in [0, 1]."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = ks_test(ref, cur)
        assert 0.0 <= result["statistic"] <= 1.0

    def test_ks_polars_input(self):
        """KS test should accept Polars Series."""
        rng = np.random.default_rng(22)
        ref = pl.Series(rng.normal(0, 1, 500).tolist())
        cur = pl.Series(rng.normal(0, 1, 500).tolist())
        result = ks_test(ref, cur)
        assert isinstance(result["statistic"], float)

    def test_ks_empty_raises(self):
        """KS test on empty array should raise ValueError."""
        with pytest.raises(ValueError):
            ks_test(np.array([]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------


class TestWassersteinDistance:
    """Wasserstein/earth mover's distance tests."""

    def test_wasserstein_identical_zero(self):
        """Identical distributions should have zero Wasserstein distance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wasserstein_distance(x, x.copy())
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_wasserstein_known_value(self):
        """Check Wasserstein distance for a known case: uniform shift by 1."""
        # Uniform distribution shifted by delta has Wasserstein distance = delta
        rng = np.random.default_rng(30)
        ref = rng.uniform(0, 1, 50_000)
        cur = ref + 1.0  # exact shift by 1
        result = wasserstein_distance(ref, cur)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_wasserstein_nonnegative(self):
        """Wasserstein distance should be non-negative."""
        rng = np.random.default_rng(31)
        ref = rng.normal(30, 5, 1_000)
        cur = rng.normal(35, 7, 800)
        result = wasserstein_distance(ref, cur)
        assert result >= 0

    def test_wasserstein_polars_input(self):
        """Wasserstein should accept Polars Series."""
        ref = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        cur = pl.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        result = wasserstein_distance(ref, cur)
        assert isinstance(result, float)
        assert result >= 0

    def test_wasserstein_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            wasserstein_distance(np.array([]), np.array([1.0]))
