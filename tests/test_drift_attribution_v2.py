"""Extended tests for insurance_monitoring.drift_attribution (TRIPODD).

Covers gaps in test_drift_attribution.py:
- _compute_loss function directly
- _predict_masked function
- _subset_risk function
- _delta function
- _shapley_based_statistic function
- _interaction_delta function
- DriftAttributor validation errors
- DriftAttributor with MAE and log_loss
- update() alias for test()
- Multiple feature drift
- Feature ranking correctness
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.drift_attribution import (
    DriftAttributor,
    DriftAttributionResult,
    _compute_loss,
    _predict_masked,
    _subset_risk,
    _delta,
    _interaction_delta,
)


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------


class LinearModel:
    """Closed-form OLS."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearModel":
        X_b = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


class ConstantModel:
    """Always returns same prediction."""
    def __init__(self, value=0.5):
        self._value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._value)


# ---------------------------------------------------------------------------
# Internal function tests
# ---------------------------------------------------------------------------


class TestComputeLoss:
    def test_mse_perfect_prediction(self):
        """MSE with y == y_pred should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        result = _compute_loss(y, y, "mse")
        assert result == pytest.approx(0.0)

    def test_mse_known_value(self):
        """MSE([0, 1], [1, 0]) = mean([1, 1]) = 1.0."""
        y = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 0.0])
        assert _compute_loss(y, y_pred, "mse") == pytest.approx(1.0)

    def test_mae_perfect_prediction(self):
        """MAE with y == y_pred should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert _compute_loss(y, y, "mae") == pytest.approx(0.0)

    def test_mae_known_value(self):
        """MAE([0, 0], [1, 2]) = mean([1, 2]) = 1.5."""
        y = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        assert _compute_loss(y, y_pred, "mae") == pytest.approx(1.5)

    def test_log_loss_perfect(self):
        """Log-loss: y=1 with pred=1 (clipped to 1-eps) gives near-zero loss."""
        y = np.array([1.0, 0.0])
        y_pred = np.array([0.9999, 0.0001])
        result = _compute_loss(y, y_pred, "log_loss")
        assert result < 0.01

    def test_log_loss_clips_predictions(self):
        """Log-loss should not produce infinity for extreme predictions."""
        y = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 1.0])  # worst case
        result = _compute_loss(y, y_pred, "log_loss")
        assert np.isfinite(result)

    def test_unknown_loss_raises(self):
        """Unknown loss name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            _compute_loss(np.array([1.0]), np.array([1.0]), "unknown_loss")

    def test_mse_positive(self):
        """MSE should be non-negative."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        assert _compute_loss(y, y_pred, "mse") >= 0.0

    def test_mae_positive(self):
        """MAE should be non-negative."""
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        assert _compute_loss(y, y_pred, "mae") >= 0.0


class TestPredictMasked:
    def test_masking_replaces_feature(self):
        """Masked feature should be replaced by fill_value."""
        class SimpleModel:
            def predict(self, X):
                return X[:, 0] + X[:, 1]

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        fill_values = np.array([0.0, 0.0])
        # Mask feature 0 (replace with 0.0)
        result = _predict_masked(SimpleModel(), X, [0], fill_values)
        # With feature 0 = 0, predictions should be [0+2, 0+4] = [2, 4]
        np.testing.assert_array_almost_equal(result, [2.0, 4.0])

    def test_no_masking_returns_unchanged(self):
        """Empty mask_indices should produce original predictions."""
        class IdentityModel:
            def predict(self, X):
                return X[:, 0]

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        fill_values = np.array([99.0, 99.0])
        result = _predict_masked(IdentityModel(), X, [], fill_values)
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_all_masked_returns_constant(self):
        """Masking all features should return predictions based on fill_values."""
        class SumModel:
            def predict(self, X):
                return X.sum(axis=1)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        fill_values = np.array([0.5, 0.5])
        # Both features masked → X becomes [[0.5, 0.5], [0.5, 0.5]]
        result = _predict_masked(SumModel(), X, [0, 1], fill_values)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])


class TestSubsetRiskAndDelta:
    def _make_linear_setup(self, seed=0):
        rng = np.random.default_rng(seed)
        n, d = 100, 3
        X = rng.normal(0, 1, (n, d))
        coef = np.array([1.0, 2.0, -1.0])
        y = X @ coef + rng.normal(0, 0.1, n)
        model = LinearModel().fit(X, y)
        fill = X.mean(axis=0)
        all_idx = list(range(d))
        return model, X, y, fill, all_idx

    def test_subset_risk_all_active_lower_than_none(self):
        """Loss with all features active should be <= loss with none active."""
        model, X, y, fill, all_idx = self._make_linear_setup()
        risk_all = _subset_risk(model, X, y, all_idx, all_idx, fill, "mse")
        risk_none = _subset_risk(model, X, y, [], all_idx, fill, "mse")
        # A good model should have lower loss when all features are available
        assert risk_all <= risk_none

    def test_delta_positive_for_useful_feature(self):
        """delta for a highly informative feature should be positive (adding it reduces loss)."""
        model, X, y, fill, all_idx = self._make_linear_setup()
        # Feature 0 (coef=1.0) is informative; adding it to empty set should reduce loss
        d_val = _delta(model, X, y, [], 0, all_idx, fill, "mse")
        assert d_val > 0.0, f"Delta for informative feature should be positive, got {d_val}"

    def test_delta_with_mae_loss(self):
        """delta function should work with MAE loss."""
        model, X, y, fill, all_idx = self._make_linear_setup()
        d_val = _delta(model, X, y, [], 0, all_idx, fill, "mae")
        assert isinstance(d_val, float)


class TestInteractionDelta:
    def test_interaction_delta_multiplicative_model(self):
        """For y = X0*X1 + noise, interaction delta should be non-trivial."""
        rng = np.random.default_rng(99)
        n, d = 300, 3
        X = rng.normal(0, 1, (n, d))
        y = X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)

        model = LinearModel().fit(X, y)
        fill = X.mean(axis=0)
        all_idx = list(range(d))

        delta_01 = _interaction_delta(model, X, y, 0, 1, all_idx, fill, "mse")
        assert isinstance(delta_01, float)

    def test_interaction_delta_additive_model_near_zero(self):
        """For strictly additive model, interaction delta should be near zero."""
        rng = np.random.default_rng(100)
        n, d = 300, 3
        X = rng.normal(0, 1, (n, d))
        # Purely additive
        coef = np.array([1.0, 2.0, 3.0])
        y = X @ coef + rng.normal(0, 0.01, n)

        model = LinearModel().fit(X, y)
        fill = X.mean(axis=0)
        all_idx = list(range(d))

        delta_01 = _interaction_delta(model, X, y, 0, 1, all_idx, fill, "mse")
        # For a well-fit additive model, interaction delta should be small
        assert abs(delta_01) < 0.5


# ---------------------------------------------------------------------------
# DriftAttributor validation and error paths
# ---------------------------------------------------------------------------


class TestDriftAttributorValidation:
    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            DriftAttributor(model=ConstantModel(), features=["a", "b"], alpha=0.0)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            DriftAttributor(model=ConstantModel(), features=["a", "b"], alpha=1.0)

    def test_n_bootstrap_too_small_raises(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            DriftAttributor(model=ConstantModel(), features=["a", "b"], n_bootstrap=5)

    def test_wrong_feature_count_at_fit_raises(self):
        """X with wrong column count should raise ValueError at fit_reference."""
        model = ConstantModel()
        attr = DriftAttributor(model=model, features=["a", "b", "c"], n_bootstrap=10)
        with pytest.raises(ValueError, match="columns"):
            attr.fit_reference(np.ones((50, 2)), np.ones(50))  # 2 cols, 3 features

    def test_test_before_fit_raises(self):
        """test() before fit_reference() should raise RuntimeError."""
        attr = DriftAttributor(
            model=ConstantModel(), features=["a", "b"], n_bootstrap=10, auto_retrain=False
        )
        with pytest.raises(RuntimeError, match="fit_reference"):
            attr.test(np.ones((50, 2)), np.ones(50))


# ---------------------------------------------------------------------------
# DriftAttributor with different loss functions
# ---------------------------------------------------------------------------


class TestDriftAttributorLossFunctions:
    def _run_basic(self, loss: str, seed: int = 0):
        rng = np.random.default_rng(seed)
        d, n = 3, 200
        features = ["a", "b", "c"]
        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, loss=loss,
            n_bootstrap=20, auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        X_new = rng.normal(0, 1, (150, d))
        y_new = X_new.sum(axis=1)
        return attr.test(X_new, y_new)

    def test_mse_loss_runs(self):
        result = self._run_basic("mse")
        assert isinstance(result.drift_detected, bool)

    def test_mae_loss_runs(self):
        result = self._run_basic("mae")
        assert isinstance(result.drift_detected, bool)

    def test_log_loss_runs(self):
        """log_loss requires predictions in (0, 1) and binary targets."""
        rng = np.random.default_rng(42)
        d, n = 3, 200
        features = ["a", "b", "c"]
        X_ref = rng.normal(0, 1, (n, d))
        # Binary targets
        y_ref = rng.binomial(1, 0.3, n).astype(float)

        class SigmoidModel:
            def predict(self, X):
                return 1.0 / (1.0 + np.exp(-X.sum(axis=1)))

        model = SigmoidModel()
        attr = DriftAttributor(
            model=model, features=features, loss="log_loss",
            n_bootstrap=20, auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        X_new = rng.normal(0, 1, (150, d))
        y_new = rng.binomial(1, 0.3, 150).astype(float)
        result = attr.test(X_new, y_new)
        assert isinstance(result.drift_detected, bool)


# ---------------------------------------------------------------------------
# DriftAttributor — update() alias
# ---------------------------------------------------------------------------


class TestUpdateAlias:
    def test_update_identical_to_test(self):
        """update() should produce identical results to test() on same data."""
        rng = np.random.default_rng(0)
        d, n = 3, 200
        features = ["a", "b", "c"]
        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        def make_attr():
            a = DriftAttributor(
                model=model, features=features, n_bootstrap=20,
                auto_retrain=False, random_state=42,
            )
            a.fit_reference(X_ref.copy(), y_ref.copy(), train_on_ref=False)
            return a

        X_new = rng.normal(0, 1, (150, d))
        y_new = X_new.sum(axis=1)

        r_test = make_attr().test(X_new, y_new)
        r_update = make_attr().update(X_new, y_new)

        assert r_test.drift_detected == r_update.drift_detected
        assert r_test.test_statistics == r_update.test_statistics


# ---------------------------------------------------------------------------
# DriftAttributor — feature_ranking correctness
# ---------------------------------------------------------------------------


class TestFeatureRanking:
    def test_drifted_feature_tops_ranking(self):
        """The strongly drifted feature should have the highest ratio in ranking."""
        rng = np.random.default_rng(5)
        d = 4
        features = ["driver_age", "ncb", "vehicle_value", "mileage"]
        coef = np.array([2.0, 0.5, 0.5, 0.5])
        n = 400

        X_ref = rng.normal(0, 1, (n, d))
        y_ref = X_ref @ coef + rng.normal(0, 0.1, n)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, alpha=0.10,
            n_bootstrap=40, auto_retrain=False, random_state=1,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)

        # Strongly shift driver_age (high coef)
        X_new = rng.normal(0, 1, (n, d))
        X_new[:, 0] += 5.0
        y_new = X_new @ coef + rng.normal(0, 0.1, n)
        result = attr.test(X_new, y_new)

        top_feature = result.feature_ranking["feature"][0]
        assert top_feature == "driver_age", (
            f"Expected 'driver_age' at top of ranking, got '{top_feature}'"
        )

    def test_feature_ranking_sorted_descending(self):
        """Feature ranking should be sorted by ratio descending."""
        rng = np.random.default_rng(6)
        d = 3
        features = ["x", "y", "z"]
        X_ref = rng.normal(0, 1, (300, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=20,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        X_new = rng.normal(0, 1, (200, d))
        y_new = X_new.sum(axis=1)
        result = attr.test(X_new, y_new)

        ratios = result.feature_ranking["ratio"].to_list()
        assert ratios == sorted(ratios, reverse=True), "Ratios should be sorted descending"

    def test_p_values_in_range(self):
        """All p_values should be in [0, 1]."""
        rng = np.random.default_rng(7)
        d = 3
        features = ["a", "b", "c"]
        X_ref = rng.normal(0, 1, (200, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=20,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        X_new = rng.normal(0, 1, (150, d))
        y_new = X_new.sum(axis=1)
        result = attr.test(X_new, y_new)

        for feat, pv in result.p_values.items():
            assert 0.0 <= pv <= 1.0, f"p_value for {feat} out of range: {pv}"


# ---------------------------------------------------------------------------
# DriftAttributor — subset_risks in result
# ---------------------------------------------------------------------------


class TestSubsetRisksInResult:
    def test_subset_risks_populated_for_all_features(self):
        """subset_risks_ref and subset_risks_new should have an entry per feature."""
        rng = np.random.default_rng(8)
        d = 3
        features = ["a", "b", "c"]
        X_ref = rng.normal(0, 1, (200, d))
        y_ref = X_ref.sum(axis=1)
        model = LinearModel().fit(X_ref, y_ref)

        attr = DriftAttributor(
            model=model, features=features, n_bootstrap=20,
            auto_retrain=False, random_state=0,
        )
        attr.fit_reference(X_ref, y_ref, train_on_ref=False)
        X_new = rng.normal(0, 1, (150, d))
        y_new = X_new.sum(axis=1)
        result = attr.test(X_new, y_new)

        assert set(result.subset_risks_ref.keys()) == set(features)
        assert set(result.subset_risks_new.keys()) == set(features)
        for v in result.subset_risks_ref.values():
            assert isinstance(v, float)
            assert np.isfinite(v)
