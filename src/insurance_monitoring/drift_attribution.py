"""
Feature-interaction-aware drift attribution for insurance pricing models.

PSI and KS tests tell you *that* drift occurred. This module tells you *which*
features (and which feature interactions) explain why your model's performance
has degraded. It implements TRIPODD (Panda et al. 2025, arXiv:2503.06606):
a permutation-based attribution method with Type I error control via bootstrap
Bonferroni thresholds.

The key insight: measure how much each feature contributes to the *change* in
model risk between reference and new windows. Features whose marginal risk
contribution changed significantly are flagged as drift attributors.

For interaction drift, the method compares joint contributions of feature pairs
to detect cases where two features interact differently even if their marginals
are unchanged — the classic insurance example is a telematics score that used
to interact with driver age but now does not.

References
----------
- Panda et al. (2025). TRIPODD: Feature-Interaction-Aware Drift Detection
  with Type I Error Control. arXiv:2503.06606.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Literal, Optional

import numpy as np
import polars as pl


LossName = Literal["mse", "log_loss", "mae"]
MaskingStrategy = Literal["mean"]


def _compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: LossName,
) -> float:
    """Compute scalar loss over a window."""
    if loss == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    elif loss == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    elif loss == "log_loss":
        # Binary cross-entropy, clipped for numerical stability
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose from: mse, log_loss, mae")


def _predict_masked(
    model: Any,
    X: np.ndarray,
    mask_indices: list[int],
    fill_values: np.ndarray,
) -> np.ndarray:
    """Predict with the given feature indices replaced by their reference means.

    We use mean-imputation (not zero) because zero can be well outside the
    support for many insurance features (e.g. vehicle age, sum insured).
    Masking with the reference mean preserves the model's baseline behaviour
    for those features while isolating the contribution of the unmasked ones.
    """
    X_masked = X.copy()
    for idx in mask_indices:
        X_masked[:, idx] = fill_values[idx]
    return np.asarray(model.predict(X_masked), dtype=np.float64)


def _subset_risk(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    active_indices: list[int],
    all_indices: list[int],
    fill_values: np.ndarray,
    loss: LossName,
) -> float:
    """Compute R^S(h): loss when predicting with only features in `active_indices`.

    Features NOT in active_indices are masked (mean-imputed).
    """
    masked = [i for i in all_indices if i not in active_indices]
    y_pred = _predict_masked(model, X, masked, fill_values)
    return _compute_loss(y, y_pred, loss)


def _delta(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    S: list[int],
    k: int,
    all_indices: list[int],
    fill_values: np.ndarray,
    loss: LossName,
) -> float:
    """delta(S, k) = R^S(h) - R^{S union {k}}(h).

    This is the marginal contribution of feature k given the active set S.
    A larger delta means feature k is more informative to the model.
    """
    r_s = _subset_risk(model, X, y, S, all_indices, fill_values, loss)
    r_sk = _subset_risk(model, X, y, S + [k], all_indices, fill_values, loss)
    return r_s - r_sk


def _shapley_based_statistic(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    all_indices: list[int],
    fill_values: np.ndarray,
    loss: LossName,
    n_permutations: Optional[int],
    rng: np.random.Generator,
) -> float:
    """Compute the feature attribution statistic for feature k.

    The statistic is: max_S |delta(S, k)| where S ranges over subsets of
    features not including k. We return the average over sampled subsets
    (not the max) for stability, following the TRIPODD implementation.

    For d <= 12: enumerate all 2^(d-1) subsets.
    For d > 12: sample n_permutations random subsets.
    """
    d = len(all_indices)
    others = [i for i in all_indices if i != k]

    if n_permutations is None:
        # Full enumeration
        n_subsets = 2 ** len(others)
        subset_deltas = []
        for mask in range(n_subsets):
            S = [others[j] for j in range(len(others)) if (mask >> j) & 1]
            subset_deltas.append(abs(_delta(model, X, y, S, k, all_indices, fill_values, loss)))
        return float(np.max(subset_deltas))
    else:
        # Random permutation sampling
        subset_deltas = []
        for _ in range(n_permutations):
            size = rng.integers(0, len(others) + 1)
            S = rng.choice(others, size=size, replace=False).tolist()
            subset_deltas.append(abs(_delta(model, X, y, S, k, all_indices, fill_values, loss)))
        return float(np.max(subset_deltas))


def _test_statistic_k(
    model: Any,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    k: int,
    all_indices: list[int],
    fill_values: np.ndarray,
    loss: LossName,
    n_permutations: Optional[int],
    rng: np.random.Generator,
) -> float:
    """c_hat_n^k = n_new * max_S |delta_ref(S,k) - delta_new(S,k)|."""
    n = len(y_new)
    d = len(all_indices)
    others = [i for i in all_indices if i != k]

    if n_permutations is None:
        n_subsets = 2 ** len(others)
        diffs = []
        for mask in range(n_subsets):
            S = [others[j] for j in range(len(others)) if (mask >> j) & 1]
            delta_ref = _delta(model, X_ref, y_ref, S, k, all_indices, fill_values, loss)
            delta_new = _delta(model, X_new, y_new, S, k, all_indices, fill_values, loss)
            diffs.append(abs(delta_ref - delta_new))
        stat = float(np.max(diffs))
    else:
        diffs = []
        for _ in range(n_permutations):
            size = rng.integers(0, len(others) + 1)
            S = rng.choice(others, size=size, replace=False).tolist() if others else []
            delta_ref = _delta(model, X_ref, y_ref, S, k, all_indices, fill_values, loss)
            delta_new = _delta(model, X_new, y_new, S, k, all_indices, fill_values, loss)
            diffs.append(abs(delta_ref - delta_new))
        stat = float(np.max(diffs))

    return n * stat


def _interaction_delta(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    k1: int,
    k2: int,
    all_indices: list[int],
    fill_values: np.ndarray,
    loss: LossName,
) -> float:
    """Interaction delta between features k1 and k2.

    delta_interaction(k1, k2) = R^{} - R^{k1} - R^{k2} + R^{k1,k2}
    where R^S is the loss with only features in S active.

    A large interaction delta means the two features jointly contribute
    differently from the sum of their individual contributions.
    """
    r_empty = _subset_risk(model, X, y, [], all_indices, fill_values, loss)
    r_k1 = _subset_risk(model, X, y, [k1], all_indices, fill_values, loss)
    r_k2 = _subset_risk(model, X, y, [k2], all_indices, fill_values, loss)
    r_k1k2 = _subset_risk(model, X, y, [k1, k2], all_indices, fill_values, loss)
    return r_empty - r_k1 - r_k2 + r_k1k2


@dataclass
class DriftAttributionResult:
    """Result of a TRIPODD drift attribution test.

    Attributes
    ----------
    drift_detected:
        True if any feature's test statistic exceeds its bootstrap threshold.
    attributed_features:
        List of feature names (or indices) flagged as drift attributors.
    test_statistics:
        Dict mapping feature name -> raw test statistic c_hat_n^k.
    thresholds:
        Dict mapping feature name -> bootstrap threshold T_alpha^k.
    p_values:
        Approximate p-values from bootstrap distribution (proportion of
        bootstrap statistics exceeding the observed statistic).
    window_ref_size:
        Number of observations in the reference window used.
    window_new_size:
        Number of observations in the new window tested.
    alpha:
        Significance level used (after Bonferroni correction).
    feature_ranking:
        Polars DataFrame ranking all features by test statistic / threshold ratio.
    interaction_pairs:
        Top-3 feature pairs with largest interaction drift delta (if
        feature_pairs=True was set). None otherwise.
    subset_risks_ref:
        Dict mapping feature name -> max absolute delta over subsets for ref window.
    subset_risks_new:
        Dict mapping feature name -> max absolute delta over subsets for new window.
    model_retrained:
        True if auto_retrain was triggered (drift detected and auto_retrain=True).
    """

    drift_detected: bool
    attributed_features: list[str]
    test_statistics: dict[str, float]
    thresholds: dict[str, float]
    p_values: dict[str, float]
    window_ref_size: int
    window_new_size: int
    alpha: float
    feature_ranking: pl.DataFrame
    interaction_pairs: Optional[pl.DataFrame]
    subset_risks_ref: dict[str, float]
    subset_risks_new: dict[str, float]
    model_retrained: bool = False

    # Support dict-style access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DriftAttributor:
    """Feature-interaction-aware model performance drift attributor.

    Implements TRIPODD (Panda et al. 2025, arXiv:2503.06606). Goes beyond
    PSI and KS: rather than asking 'has the input distribution shifted?', it
    asks 'which features explain the change in my model's predictive accuracy?'

    This distinction matters for insurance. A portfolio can shift in ways that
    cancel out — e.g. older vehicles with younger drivers — so the marginal
    distributions look stable while the joint distribution (and model accuracy)
    degrades. TRIPODD catches this by measuring feature contributions to model
    risk, not raw distributional differences.

    Parameters
    ----------
    model:
        Fitted model with a ``predict(X)`` method. For classification, predict
        should return probabilities (use predict_proba[:,1] or wrap accordingly).
    features:
        List of feature names (must match column order in X arrays passed to
        fit_reference / test).
    alpha:
        Significance level for drift detection. Bonferroni correction applied
        across d features. Defaults to 0.05.
    loss:
        Loss function: 'mse' (regression), 'log_loss' (classification),
        'mae' (robust regression). Defaults to 'mse'.
    n_bootstrap:
        Number of bootstrap resamples for threshold estimation. More gives
        more stable thresholds at higher cost. Defaults to 100.
    n_permutations:
        Number of random subset samples per feature statistic. Required when
        d > 12. If None and d <= 12, full enumeration is used. If None and
        d > 12, a ValueError is raised.
    window_size:
        Number of observations per window in streaming mode.
    step_size:
        Step size (observations) between windows in streaming mode.
    train_ratio:
        Fraction of new window used for retraining when auto_retrain=True.
    masking_strategy:
        How to impute masked features. Currently only 'mean' (reference mean).
    auto_retrain:
        If True and drift detected, retrain model on new window and reset
        reference. The model must have a ``fit(X, y)`` method.
    feature_pairs:
        If True, compute interaction drift for all feature pairs and report
        the top-3 in DriftAttributionResult.interaction_pairs.
    random_state:
        Seed for reproducibility.

    Examples
    --------
    Basic usage::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from insurance_monitoring.drift_attribution import DriftAttributor

        rng = np.random.default_rng(42)
        X_ref = rng.normal(0, 1, (1000, 4))
        y_ref = X_ref @ np.array([1, 2, 0.5, -1]) + rng.normal(0, 0.1, 1000)

        model = LinearRegression().fit(X_ref, y_ref)

        attributor = DriftAttributor(
            model=model,
            features=["age", "ncb", "vehicle_value", "mileage"],
            alpha=0.05,
            n_bootstrap=50,
        )
        attributor.fit_reference(X_ref, y_ref, train_on_ref=False)

        # New window with drift in feature 0
        X_new = rng.normal(0, 1, (500, 4))
        X_new[:, 0] += 3.0  # age has shifted
        y_new = X_new @ np.array([1, 2, 0.5, -1]) + rng.normal(0, 0.1, 500)

        result = attributor.test(X_new, y_new)
        print(result.drift_detected)        # True
        print(result.attributed_features)   # ['age']
    """

    def __init__(
        self,
        model: Any,
        features: list[str],
        alpha: float = 0.05,
        loss: LossName = "mse",
        n_bootstrap: int = 100,
        n_permutations: Optional[int] = None,
        window_size: int = 1000,
        step_size: int = 50,
        train_ratio: float = 0.8,
        masking_strategy: MaskingStrategy = "mean",
        auto_retrain: bool = True,
        feature_pairs: bool = False,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.features = features
        self.alpha = alpha
        self.loss = loss
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.window_size = window_size
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.masking_strategy = masking_strategy
        self.auto_retrain = auto_retrain
        self.feature_pairs = feature_pairs
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._d = len(features)
        self._all_indices = list(range(self._d))

        # Validate
        if self._d > 12 and n_permutations is None:
            raise ValueError(
                f"d={self._d} features > 12: full enumeration (2^d subsets) is infeasible. "
                "Set n_permutations (e.g. n_permutations=256) to use random subset sampling."
            )
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be in (0, 1)")
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap should be at least 10 for reliable thresholds")

        # State set by fit_reference
        self.is_fitted_: bool = False
        self.fill_values_: Optional[np.ndarray] = None
        self.reference_X_: Optional[np.ndarray] = None
        self.reference_y_: Optional[np.ndarray] = None

    def fit_reference(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        train_on_ref: bool = True,
    ) -> "DriftAttributor":
        """Fit the reference window.

        Parameters
        ----------
        X_ref:
            Reference feature matrix, shape (n_ref, d).
        y_ref:
            Reference targets, shape (n_ref,).
        train_on_ref:
            If True and model has a fit() method, retrain the model on X_ref
            before storing as reference. Set to False if you are providing a
            pre-trained model.

        Returns
        -------
        self
        """
        X_ref = np.asarray(X_ref, dtype=np.float64)
        y_ref = np.asarray(y_ref, dtype=np.float64)

        if X_ref.shape[1] != self._d:
            raise ValueError(
                f"X_ref has {X_ref.shape[1]} columns but {self._d} features were specified"
            )

        if train_on_ref and hasattr(self.model, "fit"):
            self.model.fit(X_ref, y_ref)

        # Compute fill values from reference window (used for masking)
        self.fill_values_ = np.nanmean(X_ref, axis=0)
        self.reference_X_ = X_ref
        self.reference_y_ = y_ref
        self.is_fitted_ = True
        return self

    def _compute_bootstrap_thresholds(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        n_new: int,
    ) -> dict[str, float]:
        """Bootstrap Bonferroni thresholds for each feature.

        The null hypothesis is: the reference and new windows have the same
        joint distribution. Under H0, we resample both windows from the
        combined reference pool and compute null statistics.

        Bonferroni correction: use (1 - alpha/d) quantile of the bootstrap
        distribution for each feature independently, giving family-wise error
        rate control at level alpha.
        """
        combined_X = X_ref
        combined_y = y_ref
        n_ref = len(y_ref)

        # We'll build the empirical null distribution of the test statistic
        boot_stats: dict[str, list[float]] = {f: [] for f in self.features}

        for _ in range(self.n_bootstrap):
            idx_boot_ref = self._rng.choice(n_ref, size=n_ref, replace=True)
            idx_boot_new = self._rng.choice(n_ref, size=n_new, replace=True)

            X_b_ref = combined_X[idx_boot_ref]
            y_b_ref = combined_y[idx_boot_ref]
            X_b_new = combined_X[idx_boot_new]
            y_b_new = combined_y[idx_boot_new]

            # Fill values from bootstrap reference
            fill_b = np.nanmean(X_b_ref, axis=0)

            for k_idx, feat in enumerate(self.features):
                stat = _test_statistic_k(
                    self.model,
                    X_b_ref,
                    y_b_ref,
                    X_b_new,
                    y_b_new,
                    k_idx,
                    self._all_indices,
                    fill_b,
                    self.loss,
                    self.n_permutations,
                    self._rng,
                )
                boot_stats[feat].append(stat)

        # Bonferroni: threshold at (1 - alpha/d) quantile
        bonferroni_level = 1.0 - self.alpha / self._d
        thresholds = {}
        for feat in self.features:
            arr = np.array(boot_stats[feat])
            thresholds[feat] = float(np.quantile(arr, bonferroni_level))

        return thresholds

    def _compute_p_values(
        self,
        test_stats: dict[str, float],
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        n_new: int,
    ) -> dict[str, float]:
        """Approximate p-values from bootstrap null distribution.

        p-value = fraction of bootstrap statistics >= observed statistic.
        These are approximate (dependent on n_bootstrap) but useful for ranking.
        """
        n_ref = len(y_ref)
        boot_stats: dict[str, list[float]] = {f: [] for f in self.features}

        for _ in range(self.n_bootstrap):
            idx_ref = self._rng.choice(n_ref, size=n_ref, replace=True)
            idx_new = self._rng.choice(n_ref, size=n_new, replace=True)
            X_b_ref = X_ref[idx_ref]
            y_b_ref = y_ref[idx_ref]
            X_b_new = X_ref[idx_new]
            y_b_new = y_ref[idx_new]
            fill_b = np.nanmean(X_b_ref, axis=0)

            for k_idx, feat in enumerate(self.features):
                stat = _test_statistic_k(
                    self.model, X_b_ref, y_b_ref, X_b_new, y_b_new,
                    k_idx, self._all_indices, fill_b, self.loss,
                    self.n_permutations, self._rng,
                )
                boot_stats[feat].append(stat)

        p_values = {}
        for feat in self.features:
            arr = np.array(boot_stats[feat])
            p_values[feat] = float(np.mean(arr >= test_stats[feat]))
        return p_values

    def _compute_interaction_pairs(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> pl.DataFrame:
        """Compute interaction drift for all feature pairs, return top-3."""
        pair_rows = []
        for k1, k2 in combinations(self._all_indices, 2):
            delta_ref = _interaction_delta(
                self.model, X_ref, y_ref, k1, k2,
                self._all_indices, self.fill_values_, self.loss,
            )
            delta_new = _interaction_delta(
                self.model, X_new, y_new, k1, k2,
                self._all_indices, self.fill_values_, self.loss,
            )
            pair_rows.append({
                "feature_1": self.features[k1],
                "feature_2": self.features[k2],
                "interaction_delta_ref": delta_ref,
                "interaction_delta_new": delta_new,
                "interaction_drift": abs(delta_ref - delta_new),
            })

        df = pl.DataFrame(pair_rows)
        if len(df) == 0:
            return df
        return df.sort("interaction_drift", descending=True).head(3)

    def test(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> DriftAttributionResult:
        """Test a new window for drift against the reference window.

        Parameters
        ----------
        X_new:
            New window feature matrix, shape (n_new, d).
        y_new:
            New window targets, shape (n_new,).

        Returns
        -------
        DriftAttributionResult
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit_reference() before test()")

        X_new = np.asarray(X_new, dtype=np.float64)
        y_new = np.asarray(y_new, dtype=np.float64)

        X_ref = self.reference_X_
        y_ref = self.reference_y_

        # Compute observed test statistics
        test_stats = {}
        subset_risks_ref = {}
        subset_risks_new = {}

        for k_idx, feat in enumerate(self.features):
            stat = _test_statistic_k(
                self.model, X_ref, y_ref, X_new, y_new,
                k_idx, self._all_indices, self.fill_values_, self.loss,
                self.n_permutations, self._rng,
            )
            test_stats[feat] = stat

            # Store raw shapley stats for reference (subset risk summaries)
            others = [i for i in self._all_indices if i != k_idx]
            # Use full model risk as a proxy for "subset risk" summary
            r_ref = _subset_risk(
                self.model, X_ref, y_ref, [k_idx],
                self._all_indices, self.fill_values_, self.loss,
            )
            r_new = _subset_risk(
                self.model, X_new, y_new, [k_idx],
                self._all_indices, self.fill_values_, self.loss,
            )
            subset_risks_ref[feat] = r_ref
            subset_risks_new[feat] = r_new

        # Bootstrap thresholds
        thresholds = self._compute_bootstrap_thresholds(X_ref, y_ref, len(y_new))

        # P-values (reuse bootstrap draws — separate call for cleanliness)
        p_values = self._compute_p_values(test_stats, X_ref, y_ref, len(y_new))

        # Attributed features
        attributed = [
            feat for feat in self.features
            if test_stats[feat] > thresholds[feat]
        ]
        drift_detected = len(attributed) > 0

        # Feature ranking DataFrame
        ranking_rows = []
        for feat in self.features:
            ratio = test_stats[feat] / max(thresholds[feat], 1e-12)
            ranking_rows.append({
                "feature": feat,
                "test_statistic": test_stats[feat],
                "threshold": thresholds[feat],
                "ratio": ratio,
                "p_value": p_values[feat],
                "drift_attributed": feat in attributed,
            })
        ranking_df = pl.DataFrame(ranking_rows).sort("ratio", descending=True)

        # Interaction pairs
        interaction_pairs = None
        if self.feature_pairs and self._d >= 2:
            interaction_pairs = self._compute_interaction_pairs(X_ref, y_ref, X_new, y_new)

        # Auto-retrain
        model_retrained = False
        if self.auto_retrain and drift_detected and hasattr(self.model, "fit"):
            n_train = int(len(y_new) * self.train_ratio)
            self.model.fit(X_new[:n_train], y_new[:n_train])
            self.fill_values_ = np.nanmean(X_new, axis=0)
            self.reference_X_ = X_new
            self.reference_y_ = y_new
            model_retrained = True

        return DriftAttributionResult(
            drift_detected=drift_detected,
            attributed_features=attributed,
            test_statistics=test_stats,
            thresholds=thresholds,
            p_values=p_values,
            window_ref_size=len(y_ref),
            window_new_size=len(y_new),
            alpha=self.alpha,
            feature_ranking=ranking_df,
            interaction_pairs=interaction_pairs,
            subset_risks_ref=subset_risks_ref,
            subset_risks_new=subset_risks_new,
            model_retrained=model_retrained,
        )

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> DriftAttributionResult:
        """Slide the window forward and test for drift.

        Identical to test() but named update() to make streaming pipelines
        read naturally.
        """
        return self.test(X_new, y_new)

    def run_stream(
        self,
        X_stream: np.ndarray,
        y_stream: np.ndarray,
    ) -> pl.DataFrame:
        """Process a full data stream in windows and return a summary DataFrame.

        The reference window is the first window_size observations. Subsequent
        windows of size window_size are tested at step_size intervals.

        Parameters
        ----------
        X_stream:
            Full stream feature matrix, shape (N, d).
        y_stream:
            Full stream targets, shape (N,).

        Returns
        -------
        pl.DataFrame
            One row per test window with columns: window_start, window_end,
            drift_detected, attributed_features, model_retrained.
        """
        X_stream = np.asarray(X_stream, dtype=np.float64)
        y_stream = np.asarray(y_stream, dtype=np.float64)
        N = len(y_stream)

        if N < self.window_size * 2:
            raise ValueError(
                f"Stream length {N} must be at least 2 * window_size = {self.window_size * 2}"
            )

        # Fit reference on first window
        X_ref_init = X_stream[: self.window_size]
        y_ref_init = y_stream[: self.window_size]
        self.fit_reference(X_ref_init, y_ref_init, train_on_ref=False)

        rows = []
        start = self.window_size
        while start + self.window_size <= N:
            end = start + self.window_size
            X_w = X_stream[start:end]
            y_w = y_stream[start:end]
            result = self.update(X_w, y_w)
            rows.append({
                "window_start": start,
                "window_end": end,
                "drift_detected": result.drift_detected,
                "attributed_features": ",".join(result.attributed_features),
                "model_retrained": result.model_retrained,
                "n_attributed": len(result.attributed_features),
            })
            start += self.step_size

        return pl.DataFrame(rows)

    @staticmethod
    def psi_comparison(
        result: DriftAttributionResult,
        reference: "pl.DataFrame",
        current: "pl.DataFrame",
        features: list[str],
        n_bins: int = 10,
    ) -> pl.DataFrame:
        """Merge TRIPODD attribution results with PSI/CSI values for comparison.

        Produces a combined DataFrame showing both the TRIPODD drift attribution
        statistic and the traditional PSI for each feature. This is the right
        diagnostic table to include in a monitoring pack: PSI for the audience
        that expects it, TRIPODD ratio for interpretable attribution.

        Parameters
        ----------
        result:
            DriftAttributionResult from .test().
        reference:
            Reference period DataFrame (Polars).
        current:
            Current period DataFrame (Polars).
        features:
            Feature column names.
        n_bins:
            Bins for PSI computation.

        Returns
        -------
        pl.DataFrame
            Merged DataFrame with columns: feature, csi, band, test_statistic,
            threshold, ratio, p_value, drift_attributed.
        """
        from insurance_monitoring.drift import csi as compute_csi

        csi_df = compute_csi(reference, current, features=features, n_bins=n_bins)
        merged = csi_df.join(result.feature_ranking, on="feature", how="left")
        return merged
