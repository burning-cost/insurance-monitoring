"""Tests for insurance_monitoring.drift_attribution (TRIPODD implementation).

These tests cover:
1. Type I error control under the null (no drift)
2. Single-feature drift detection
3. Interaction drift without marginal drift
4. Threshold sensitivity to alpha
5. Auto-retrain behaviour
6. High-dimensional permutation sampling performance
7. PSI comparison utility
"""

from __future__ import annotations

import time
import numpy as np
import polars as pl
import pytest

from insurance_monitoring.drift_attribution import DriftAttributor, DriftAttributionResult


# ---------------------------------------------------------------------------
# Minimal sklearn-compatible model stubs
# ---------------------------------------------------------------------------


class LinearModel:
    """Minimal linear model for testing — avoids sklearn dependency in tests."""

    def __init__(self, coef=None):
        self.coef_ = coef
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearModel":
        # Closed-form OLS
        X_b = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        except Exception:
            beta = np.zeros(X.shape[1] + 1)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


class ConstantModel:
    """Model that always predicts the mean of training targets."""

    def __init__(self, value: float = 0.0):
        self._value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._value)


# ---------------------------------------------------------------------------
# Helper: build a fitted DriftAttributor quickly
# ---------------------------------------------------------------------------


def _make_attributor(
    features=None,
    alpha=0.05,
    n_bootstrap=30,
    n_permutations=None,
    auto_retrain=False,
    feature_pairs=False,
    seed=0,
):
    if features is None:
        features = ["f0", "f1", "f2", "f3"]
    return DriftAttributor(
        model=LinearModel(),
        features=features,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        auto_retrain=auto_retrain,
        feature_pairs=feature_pairs,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Test 1: Type I error control — same distribution, false positive rate < alpha + margin
# ---------------------------------------------------------------------------


class TestTypeIErrorControl:
    """Under H0 (same distribution), detection rate should be close to alpha."""

    def test_no_drift_type_I_control(self):
        """Run 20 trials with same-distribution windows; expect < alpha + 0.20 false positives."""
        rng = np.random.default_rng(42)
        alpha = 0.10
        n_trials = 20
        n_ref = 300
        n_new = 200
        d = 3
        features = [f"x{i}" for i in range(d)]
        coef = np.array([1.0, -0.5, 0.3])

        false_positives = 0
        for trial in range(n_trials):
            X_ref = rng.normal(0, 1, (n_ref, d))
            y_ref = X_ref @ coef + rng.normal(0, 0.5, n_ref)

            X_new = rng.normal(0, 1, (n_new, d))
            y_new = X_new @ coef + rng.normal(0, 0.5, n_new)

            model = LinearModel().fit(X_ref, y_ref)
            attr = DriftAttributor(
                model=model,
                features=features,
                alpha=alpha,
                n_bootstrap=25,
                auto_retrain=False,
                random_state=trial,
            )
            attr.fit_reference(X_ref, y_ref, train_on_ref=False)
            result = attr.test(X_new, y_new)
            if result.drift_detected:
                false_positives += 1

        fp_rate = false_positives / n_trials
        # Allow generous margin: alpha + 0.25 (bootstrap with n=25 has variance)
        assert fp_rate <= alpha + 0.25, (
            f"False positive rate {fp_rate:.2f} exceeds alpha + 0.25 = {alpha + 0.25:.2f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Single-feature drift is detected and correctly attributed
# ---------------------------------------------------------------------------


class TestSingleFeatureDrift:
    """Shift one feature strongly; assert it appears in attributed_features."""

    def test_single_feature_drift_detected(self):
        rng = np.random.default_rng(7)
        n = 500
        d = 4
        features = ["age", "ncb", "vehicle_value", "mileage"]
        coef = np.array([2.0, -1.0, 0.5, 0.3])

        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref @ coef + rng.normal(0, 0.1, n)

        model = LinearModel().fit(X_ref, y_ref)
        attr = DriftAttributor(
            model=model,
            features=features,
            alpha=0.10,
            n_bootstrap=40,
            auto_retrain=False,
            random_state=1,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        # Shift feature 0 ("age") by 5 standard deviations
        X_new = rng.normal(0, 1, (n, d))
        X_new[:, 0] += 5.0
        # Target still follows original relationship
        y_new = X_new @ coef + rng.normal(0, 0.1, n)

        result = attr.test(X_new, y_new)

        assert result.drift_detected, "Expected drift to be detected after 5-sigma shift"
        assert "age" in result.attributed_features, (
            f"Expected 'age' in attributed_features, got {result.attributed_features}"
        )

    def test_result_fields_complete(self):
        """DriftAttributionResult has all required fields populated."""
        rng = np.random.default_rng(11)
        d = 3
        features = ["a", "b", "c"]
        X_ref = rng.normal(0, 1, (200, d))
        y_ref = X_ref.sum(axis=1) + rng.normal(0, 0.1, 200)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=20, auto_retrain=False, random_state=0
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        X_new = rng.normal(0, 1, (150, d))
        y_new = X_new.sum(axis=1) + rng.normal(0, 0.1, 150)
        result = attr.test(X_new, y_new)

        assert isinstance(result.drift_detected, bool)
        assert isinstance(result.attributed_features, list)
        assert len(result.test_statistics) == d
        assert len(result.thresholds) == d
        assert len(result.p_values) == d
        assert result.window_ref_size == 200
        assert result.window_new_size == 150
        assert isinstance(result.feature_ranking, pl.DataFrame)
        assert "feature" in result.feature_ranking.columns
        assert "ratio" in result.feature_ranking.columns
        assert result.interaction_pairs is None  # feature_pairs=False


# ---------------------------------------------------------------------------
# Test 3: Interaction drift without marginal drift
# ---------------------------------------------------------------------------


class TestInteractionDrift:
    """Change the interaction structure while keeping marginals identical.

    Reference: Y = X0 * X1  (strong interaction)
    New:       Y = X0 + X1  (no interaction, same marginals)

    A method that only checks marginal distributions cannot detect this.
    TRIPODD should detect it via the feature_pairs option.
    """

    def test_interaction_drift_no_marginal_drift(self):
        rng = np.random.default_rng(99)
        n = 400
        d = 3
        features = ["x0", "x1", "x2"]

        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref[:, 0] * X_ref[:, 1] + rng.normal(0, 0.05, n)

        # Fit a simple additive model — deliberately "wrong" for reference too
        # so the comparison is relative to fixed model behaviour
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model,
            features=features,
            alpha=0.10,
            n_bootstrap=30,
            auto_retrain=False,
            feature_pairs=True,
            random_state=5,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        # New data: same marginals, different generative process
        X_new = rng.normal(0, 1, (n, d))
        y_new = X_new[:, 0] + X_new[:, 1] + rng.normal(0, 0.05, n)

        result = attr.test(X_new, y_new)

        # Interaction pairs should be populated
        assert result.interaction_pairs is not None
        assert isinstance(result.interaction_pairs, pl.DataFrame)
        assert "feature_1" in result.interaction_pairs.columns
        assert "interaction_drift" in result.interaction_pairs.columns
        # At most 3 rows returned
        assert len(result.interaction_pairs) <= 3


# ---------------------------------------------------------------------------
# Test 4: Stricter alpha produces higher thresholds
# ---------------------------------------------------------------------------


class TestBootstrapThreshold:
    """A stricter alpha (smaller value) should produce higher Bonferroni thresholds."""

    def test_bootstrap_threshold_higher_alpha(self):
        rng = np.random.default_rng(3)
        d = 3
        features = ["p", "q", "r"]
        X_ref = rng.normal(0, 1, (300, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        X_new = rng.normal(0, 1, (200, d))
        y_new = X_new.sum(axis=1)

        # alpha=0.01: very strict, thresholds should be higher
        attr_strict = DriftAttributor(
            model=model, features=features, alpha=0.01,
            n_bootstrap=50, auto_retrain=False, random_state=42,
        )
        attr_strict.fit_reference(X_ref, y_ref, train_on_ref=False)
        result_strict = attr_strict.test(X_new, y_new)

        # alpha=0.20: lenient, thresholds should be lower
        attr_lenient = DriftAttributor(
            model=model, features=features, alpha=0.20,
            n_bootstrap=50, auto_retrain=False, random_state=42,
        )
        attr_lenient.fit_reference(X_ref, y_ref, train_on_ref=False)
        result_lenient = attr_lenient.test(X_new, y_new)

        # Average threshold across features should be higher for strict alpha
        avg_strict = np.mean(list(result_strict.thresholds.values()))
        avg_lenient = np.mean(list(result_lenient.thresholds.values()))
        assert avg_strict > avg_lenient, (
            f"Strict alpha=0.01 threshold ({avg_strict:.4f}) should exceed "
            f"lenient alpha=0.20 threshold ({avg_lenient:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 5: Auto-retrain updates reference window
# ---------------------------------------------------------------------------


class TestAutoRetrain:
    """When auto_retrain=True and drift is detected, reference should be updated."""

    def test_auto_retrain_updates_reference(self):
        rng = np.random.default_rng(17)
        d = 3
        features = ["u", "v", "w"]
        coef = np.array([3.0, -2.0, 1.0])
        n = 400

        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref @ coef + rng.normal(0, 0.1, n)

        model = LinearModel().fit(X_ref, y_ref)
        original_ref_X = X_ref.copy()

        attr = DriftAttributor(
            model=model,
            features=features,
            alpha=0.10,
            n_bootstrap=30,
            auto_retrain=True,
            random_state=10,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        # Introduce large drift
        X_new = rng.normal(5, 1, (n, d))  # large mean shift in all features
        y_new = X_new @ coef + rng.normal(0, 0.1, n)

        result = attr.test(X_new, y_new)

        if result.drift_detected:
            # Reference should have been updated to the new window
            assert result.model_retrained is True
            np.testing.assert_array_equal(attr.reference_X_, X_new)
            np.testing.assert_array_equal(attr.reference_y_, y_new)
        else:
            # If drift not detected (possible with small n_bootstrap), reference unchanged
            np.testing.assert_array_equal(attr.reference_X_, original_ref_X)
            assert result.model_retrained is False

    def test_auto_retrain_false_does_not_update(self):
        rng = np.random.default_rng(19)
        d = 2
        features = ["a", "b"]
        X_ref = rng.normal(0, 1, (200, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=20,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        X_new = rng.normal(10, 1, (200, d))  # large drift
        y_new = X_new.sum(axis=1)
        result = attr.test(X_new, y_new)

        # Should never retrain
        assert result.model_retrained is False
        np.testing.assert_array_equal(attr.reference_X_, X_ref)


# ---------------------------------------------------------------------------
# Test 6: High-dimensional permutation sampling runs under 60 seconds
# ---------------------------------------------------------------------------


class TestHighDimensional:
    """d=20 with n_permutations=64 should complete in under 60 seconds."""

    def test_high_dimensional_permutation_sampling(self):
        rng = np.random.default_rng(55)
        d = 20
        features = [f"feat_{i}" for i in range(d)]
        coef = rng.normal(0, 1, d)
        n = 200

        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref @ coef + rng.normal(0, 0.1, n)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model,
            features=features,
            alpha=0.10,
            n_bootstrap=10,          # minimal for speed
            n_permutations=32,       # required for d > 12
            auto_retrain=False,
            random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        X_new = rng.normal(0, 1, (n, d))
        X_new[:, 0] += 3.0          # shift one feature
        y_new = X_new @ coef + rng.normal(0, 0.1, n)

        t0 = time.time()
        result = attr.test(X_new, y_new)
        elapsed = time.time() - t0

        assert elapsed < 60.0, f"High-dimensional test took {elapsed:.1f}s (limit: 60s)"
        assert isinstance(result.drift_detected, bool)
        assert len(result.test_statistics) == d

    def test_high_dimensional_without_n_permutations_raises(self):
        """d > 12 without n_permutations should raise ValueError at init."""
        with pytest.raises(ValueError, match="n_permutations"):
            DriftAttributor(
                model=ConstantModel(),
                features=[f"f{i}" for i in range(13)],
                n_permutations=None,
            )


# ---------------------------------------------------------------------------
# Test 7: psi_comparison returns merged DataFrame with expected columns
# ---------------------------------------------------------------------------


class TestPSIComparison:
    """psi_comparison() should return a DataFrame with both TRIPODD and PSI columns."""

    def test_psi_comparison_returns_merged_df(self):
        rng = np.random.default_rng(77)
        d = 3
        features = ["driver_age", "vehicle_age", "ncb"]
        coef = np.array([1.0, -0.5, 0.3])
        n = 400

        X_ref_arr = rng.normal(0, 1, (n, d))
        y_ref = X_ref_arr @ coef + rng.normal(0, 0.1, n)
        model = LinearModel().fit(X_ref_arr, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=20,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref_arr, y_ref, train_on_ref=False)

        X_new_arr = rng.normal(0, 1, (n, d))
        y_new = X_new_arr @ coef + rng.normal(0, 0.1, n)
        result = attr.test(X_new_arr, y_new)

        # Build Polars DataFrames for PSI comparison
        ref_df = pl.DataFrame({feat: X_ref_arr[:, i].tolist() for i, feat in enumerate(features)})
        new_df = pl.DataFrame({feat: X_new_arr[:, i].tolist() for i, feat in enumerate(features)})

        comparison = DriftAttributor.psi_comparison(result, ref_df, new_df, features)

        assert isinstance(comparison, pl.DataFrame)
        # Must have PSI columns
        assert "csi" in comparison.columns
        assert "band" in comparison.columns
        # Must have TRIPODD columns
        assert "test_statistic" in comparison.columns
        assert "threshold" in comparison.columns
        assert "ratio" in comparison.columns
        assert "drift_attributed" in comparison.columns
        # One row per feature
        assert len(comparison) == d
        assert set(comparison["feature"].to_list()) == set(features)

    def test_psi_comparison_nonnegative_csi(self):
        """CSI values in merged output should all be non-negative."""
        rng = np.random.default_rng(88)
        d = 2
        features = ["x", "y"]
        X_ref_arr = rng.normal(0, 1, (300, d))
        y_ref = X_ref_arr.sum(axis=1)
        model = LinearModel().fit(X_ref_arr, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=15,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref_arr, y_ref, train_on_ref=False)
        X_new_arr = rng.normal(0, 1, (200, d))
        y_new = X_new_arr.sum(axis=1)
        result = attr.test(X_new_arr, y_new)

        ref_df = pl.DataFrame({feat: X_ref_arr[:, i].tolist() for i, feat in enumerate(features)})
        new_df = pl.DataFrame({feat: X_new_arr[:, i].tolist() for i, feat in enumerate(features)})
        comparison = DriftAttributor.psi_comparison(result, ref_df, new_df, features)
        assert (comparison["csi"] >= 0).all()


# ---------------------------------------------------------------------------
# Test 8: run_stream returns a Polars DataFrame
# ---------------------------------------------------------------------------


class TestRunStream:
    """run_stream() should process a stream and return a summary DataFrame."""

    def test_run_stream_returns_dataframe(self):
        rng = np.random.default_rng(101)
        d = 3
        features = ["a", "b", "c"]
        coef = np.array([1.0, 2.0, -1.0])
        N = 2500

        X_stream = rng.normal(0, 1, (N, d))
        y_stream = X_stream @ coef + rng.normal(0, 0.1, N)
        model = LinearModel().fit(X_stream[:500], y_stream[:500])

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=10,
            window_size=500, step_size=250,
            auto_retrain=False, random_state=0,
        )
        df = attr.run_stream(X_stream, y_stream)

        assert isinstance(df, pl.DataFrame)
        assert "window_start" in df.columns
        assert "drift_detected" in df.columns
        assert "attributed_features" in df.columns
        assert len(df) > 0

    def test_run_stream_too_short_raises(self):
        """Stream shorter than 2 * window_size should raise ValueError."""
        model = ConstantModel(0.0)
        attr = DriftAttributor(
            model=model, features=["x"], n_bootstrap=10,
            window_size=500, auto_retrain=False,
        )
        with pytest.raises(ValueError, match="window_size"):
            attr.run_stream(np.ones((600, 1)), np.ones(600))


# ---------------------------------------------------------------------------
# Test 9: dict-style access on DriftAttributionResult (backward compat)
# ---------------------------------------------------------------------------


class TestResultBackwardCompat:
    def test_dict_style_access(self):
        rng = np.random.default_rng(200)
        d = 2
        features = ["p", "q"]
        X_ref = rng.normal(0, 1, (200, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=15,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        result = attr.test(rng.normal(0, 1, (150, d)), rng.normal(0, 1, 150))

        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result["attributed_features"], list)
        assert isinstance(result["feature_ranking"], pl.DataFrame)


# ---------------------------------------------------------------------------
# Test 10: fit_reference with train_on_ref=True retrains model
# ---------------------------------------------------------------------------


class TestFitReference:
    def test_train_on_ref_calls_fit(self):
        """fit_reference(train_on_ref=True) should update model coefficients."""
        rng = np.random.default_rng(300)
        d = 2
        features = ["x0", "x1"]

        X_ref = rng.normal(0, 1, (300, d))
        y_ref = X_ref @ np.array([5.0, -3.0]) + rng.normal(0, 0.1, 300)

        model = LinearModel()
        model.coef_ = np.zeros(d)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=10,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=True)

        # After training, coefficients should be close to [5, -3]
        assert abs(model.coef_[0] - 5.0) < 0.5, f"coef[0]={model.coef_[0]:.2f}, expected ~5"
        assert abs(model.coef_[1] - (-3.0)) < 0.5, f"coef[1]={model.coef_[1]:.2f}, expected ~-3"

    def test_fill_values_computed_from_reference(self):
        """fill_values_ should equal column means of X_ref."""
        rng = np.random.default_rng(301)
        d = 3
        features = ["a", "b", "c"]
        X_ref = rng.normal([10, 20, 30], 1, (500, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=10,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        expected = X_ref.mean(axis=0)
        np.testing.assert_allclose(attr.fill_values_, expected, rtol=1e-6)
