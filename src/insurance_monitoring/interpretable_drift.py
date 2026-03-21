"""
Interpretable drift detection with exposure weighting, FDR control, and Poisson deviance.

This module provides InterpretableDriftDetector — a substantive upgrade of DriftAttributor
(also in this package). Both coexist: DriftAttributor remains suitable for streaming
window use cases and auto-retrain workflows. InterpretableDriftDetector is the right
choice when you need:

- Exposure-weighted risk estimates (mixed policy terms: 0.25-year vs 1.0-year)
- Benjamini-Hochberg FDR control for d >= 10 rating factors
- Poisson deviance loss for frequency model monitoring
- Explicit reference management (update_reference() instead of auto-retrain)
- Polars-native DataFrame input

The statistical foundation is TRIPODD (Panda, Srinivas, Balasubramanian & Sinha,
arXiv:2503.06606): permutation-based attribution with Type I error control via
bootstrap thresholds.

Improvements over DriftAttributor
----------------------------------
1. Exposure weighting via ``weights`` parameter at fit and test time.
2. FDR control: Benjamini-Hochberg alongside Bonferroni.
3. Bootstrap efficiency: single loop for thresholds and p-values (halved cost).
4. Subset risk caching at fit_reference() — reference-side predictions saved.
5. Polars-native API: accepts pl.DataFrame / pl.Series directly.
6. Poisson deviance loss for count / frequency models.
7. Explicit update_reference() — no silent retraining on drift detection.

References
----------
- Panda et al. (2025). TRIPODD: Feature-Interaction-Aware Drift Detection
  with Type I Error Control. arXiv:2503.06606.
- Benjamini & Hochberg (1995). Controlling the False Discovery Rate.
  Journal of the Royal Statistical Society B, 57(1), 289-300.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Optional, Union

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LossName = Literal["mse", "log_loss", "mae", "poisson_deviance"]
MaskingStrategy = Literal["mean", "median", "zero"]

_ArrayLike = Union[np.ndarray, "pl.DataFrame", "pl.Series"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_elementwise_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: LossName,
) -> np.ndarray:
    """Per-observation losses (not reduced to mean).

    Returns an array of shape (n,) containing the individual loss values.
    Callers apply weighting and reduction after the fact.
    """
    if loss == "mse":
        return (y_true - y_pred) ** 2
    elif loss == "mae":
        return np.abs(y_true - y_pred)
    elif loss == "log_loss":
        eps = 1e-7
        p = np.clip(y_pred, eps, 1.0 - eps)
        return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    elif loss == "poisson_deviance":
        mu = np.clip(y_pred, 1e-8, None)
        # Use errstate to silence the benign log(0) and 0*NaN warnings that
        # arise because numpy evaluates both branches of np.where before masking.
        # The y=0 rows use 2*mu regardless — the log branch is never returned.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = 2.0 * (y_true * np.log(y_true / mu) - (y_true - mu))
        losses = np.where(y_true > 0, log_term, 2.0 * mu)
        return losses
    else:
        raise ValueError(
            f"Unknown loss '{loss}'. Choose from: mse, log_loss, mae, poisson_deviance"
        )


def _compute_fill_values(
    X: np.ndarray,
    masking_strategy: MaskingStrategy,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute fill values used to mask features not in the active set.

    Parameters
    ----------
    X:
        Feature matrix, shape (n, d).
    masking_strategy:
        'mean' uses (exposure-weighted) column means. 'median' uses column
        medians. 'zero' uses zeros (matches the TRIPODD paper exactly but
        is inappropriate when zero is outside the feature's support).
    weights:
        Exposure weights, shape (n,). Used only when masking_strategy='mean'.
    """
    if masking_strategy == "zero":
        return np.zeros(X.shape[1])
    elif masking_strategy == "median":
        return np.nanmedian(X, axis=0)
    else:  # "mean"
        if weights is not None:
            w = weights / weights.sum()
            return np.average(X, axis=0, weights=w)
        return np.nanmean(X, axis=0)


def _subset_risk(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    S: frozenset,
    d: int,
    fill_values: np.ndarray,
    loss: LossName,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Exposure-weighted subset risk R^S(h, D, e).

    Features not in S are replaced by their fill values before predicting.

    Parameters
    ----------
    model:
        Fitted model with a predict(X) method.
    X:
        Feature matrix, shape (n, d).
    y:
        Targets, shape (n,).
    S:
        Frozenset of active feature indices (0-based). Features not in S
        are masked (replaced by fill_values).
    d:
        Total number of features.
    fill_values:
        Reference fill values for masking, shape (d,).
    loss:
        Loss function name.
    weights:
        Exposure weights, shape (n,). If None, all observations are
        weighted equally.
    """
    masked_X = X.copy()
    for j in range(d):
        if j not in S:
            masked_X[:, j] = fill_values[j]
    y_pred = np.asarray(model.predict(masked_X), dtype=np.float64)
    losses = _compute_elementwise_loss(y, y_pred, loss)
    if weights is not None:
        w = weights / weights.sum()
        return float(np.dot(w, losses))
    return float(np.mean(losses))


def _enumerate_subsets(d: int, n_permutations: Optional[int], rng: np.random.Generator) -> list:
    """Enumerate all subsets of {0, ..., d-1} for cached subset risk computation.

    For d <= 12, returns all 2^d frozensets. For d > 12, samples
    n_permutations randomly-sized subsets.

    Note: this enumerates subsets of all features (not per-feature exclusion).
    The caller is responsible for skipping subsets containing feature k when
    computing statistics for feature k.
    """
    all_indices = list(range(d))
    if n_permutations is None:
        # Full enumeration: all 2^d subsets
        subsets = []
        for mask in range(2 ** d):
            S = frozenset(j for j in range(d) if (mask >> j) & 1)
            subsets.append(S)
        return subsets
    else:
        # Stratified random sampling
        subsets = set()
        # Always include empty and full sets
        subsets.add(frozenset())
        subsets.add(frozenset(all_indices))
        while len(subsets) < n_permutations:
            size = int(rng.integers(0, d + 1))
            chosen = rng.choice(all_indices, size=size, replace=False)
            subsets.add(frozenset(chosen))
        return list(subsets)


def _apply_error_control(
    test_stats: dict,
    bootstrap_stats: dict,
    features: list,
    alpha: float,
    d: int,
    error_control: str,
) -> tuple:
    """Apply multiple testing correction and return (thresholds, p_values, attributed).

    Parameters
    ----------
    test_stats:
        Observed test statistics per feature.
    bootstrap_stats:
        Bootstrap null statistics per feature (list of floats).
    features:
        Feature names in order.
    alpha:
        Significance level.
    d:
        Number of features (= len(features)).
    error_control:
        'fwer' for Bonferroni; 'fdr' for Benjamini-Hochberg.
    """
    p_values = {}
    for feat in features:
        arr = np.array(bootstrap_stats[feat])
        p_values[feat] = float(np.mean(arr >= test_stats[feat]))

    if error_control == "fwer":
        bonferroni_level = 1.0 - alpha / d
        thresholds = {}
        for feat in features:
            arr = np.array(bootstrap_stats[feat])
            thresholds[feat] = float(np.quantile(arr, bonferroni_level))
        attributed = [f for f in features if test_stats[f] > thresholds[f]]

    else:  # "fdr" — Benjamini-Hochberg
        sorted_feats = sorted(features, key=lambda f: p_values[f])
        k_star = 0
        for rank, feat in enumerate(sorted_feats, start=1):
            if p_values[feat] <= rank * alpha / d:
                k_star = rank
        attributed = sorted_feats[:k_star]
        if k_star > 0:
            bh_cutoff = p_values[sorted_feats[k_star - 1]]
        else:
            bh_cutoff = float("inf")
        thresholds = {f: bh_cutoff for f in features}

    return thresholds, p_values, attributed


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InterpretableDriftResult:
    """Result of an InterpretableDriftDetector test.

    Attributes
    ----------
    drift_detected:
        True if any feature's H0 is rejected.
    attributed_features:
        Feature names where H0 was rejected.
    test_statistics:
        Observed test statistic c_hat_n^k per feature.
    thresholds:
        Bootstrap threshold per feature (Bonferroni) or BH p-value cutoff (FDR).
    p_values:
        Bootstrap p-values per feature.
    error_control:
        Multiple testing method used: 'fwer' or 'fdr'.
    alpha:
        Significance level.
    feature_ranking:
        Polars DataFrame with one row per feature, sorted by test_statistic /
        threshold ratio descending. Columns: feature, test_statistic, threshold,
        ratio, p_value, drift_attributed, rank.
    interaction_pairs:
        Top-5 feature pairs by interaction drift (when feature_pairs=True). None
        otherwise. Columns: feature_1, feature_2, interaction_delta_ref,
        interaction_delta_new, interaction_drift, rank.
    window_ref_size:
        Number of observations in the reference window.
    window_new_size:
        Number of observations in the new window tested.
    bootstrap_iterations:
        Number of bootstrap permutations used.
    computation_time_s:
        Wall-clock seconds for the test() call.
    """

    drift_detected: bool
    attributed_features: list
    test_statistics: dict
    thresholds: dict
    p_values: dict
    error_control: str
    alpha: float
    feature_ranking: pl.DataFrame
    interaction_pairs: Optional[pl.DataFrame]
    window_ref_size: int
    window_new_size: int
    bootstrap_iterations: int
    computation_time_s: float

    def summary(self) -> str:
        """One-paragraph plain-text summary suitable for governance packs.

        Returns a concise description of the test result: whether drift was
        detected, which features were attributed, the error control method,
        and top statistics.
        """
        if not self.drift_detected:
            top = self.feature_ranking.head(1)
            top_feat = top["feature"][0]
            top_ratio = top["ratio"][0]
            return (
                f"No drift detected. "
                f"Strongest signal: {top_feat} (ratio={top_ratio:.2f}). "
                f"{self.error_control.upper()} error control at alpha={self.alpha}. "
                f"{self.bootstrap_iterations} bootstrap iterations. "
                f"Reference n={self.window_ref_size}, new n={self.window_new_size}."
            )

        attributed_strs = []
        for feat in self.attributed_features:
            pv = self.p_values[feat]
            ratio = self.test_statistics[feat] / max(self.thresholds[feat], 1e-12)
            attributed_strs.append(f"{feat} (p={pv:.3f}, ratio={ratio:.1f}x)")
        feat_summary = ", ".join(attributed_strs)

        interaction_str = ""
        if self.interaction_pairs is not None and len(self.interaction_pairs) > 0:
            top_pair = self.interaction_pairs.head(1)
            f1 = top_pair["feature_1"][0]
            f2 = top_pair["feature_2"][0]
            delta = top_pair["interaction_drift"][0]
            interaction_str = f" Top interaction drift: {f1} x {f2} (delta={delta:.4f})."

        return (
            f"Drift detected. Features attributed: {feat_summary}.{interaction_str} "
            f"{self.error_control.upper()} error control at alpha={self.alpha}. "
            f"{self.bootstrap_iterations} bootstrap iterations."
        )

    def to_monitoring_row(self) -> dict:
        """Dict compatible with MonitoringReport result schema.

        Returns one dict per detection event — suitable for appending to a
        monitoring log table. One row per feature, with drift attribution flag.
        """
        rows = self.feature_ranking.to_dicts()
        for row in rows:
            row["window_ref_size"] = self.window_ref_size
            row["window_new_size"] = self.window_new_size
            row["error_control"] = self.error_control
            row["alpha"] = self.alpha
            row["drift_detected"] = self.drift_detected
            row["computation_time_s"] = self.computation_time_s
        return rows

    def __getitem__(self, key: str) -> Any:
        """Dict-style access for backward compatibility."""
        return getattr(self, key)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InterpretableDriftDetector:
    """Interpretable model performance drift detection for insurance pricing models.

    Implements TRIPODD (Panda et al. 2025, arXiv:2503.06606) with seven
    substantive improvements over DriftAttributor:

    1. Exposure weighting — correct for mixed policy terms (e.g. 0.25-year
       and 1.0-year). Unweighted means overweight short-tenure policies.
    2. FDR control via Benjamini-Hochberg — for d >= 10 rating factors,
       Bonferroni is too conservative. BH controls false discovery rate at
       alpha instead of family-wise error rate.
    3. Single bootstrap loop — thresholds and p-values from one pass.
       Halved cost over DriftAttributor (which ran two separate loops).
    4. Subset risk caching — reference-side predictions cached at
       fit_reference(). Only new-window predictions needed per test().
    5. Polars-native API — accepts pl.DataFrame and pl.Series directly.
    6. Poisson deviance loss — the canonical GLM goodness-of-fit for count
       data. MSE is not appropriate for frequency models.
    7. Explicit update_reference() — no auto-retrain on drift detection.
       Retraining requires external sign-off in a production setting.

    Parameters
    ----------
    model:
        Fitted model with a ``predict(X) -> np.ndarray`` method. For log-loss,
        predict must return probabilities in [0, 1].
    features:
        Feature names. Length must match the number of columns in all X arrays
        passed to fit_reference() and test().
    alpha:
        Significance level. Default 0.05.
    loss:
        Loss function. 'mse' for regression, 'log_loss' for classification,
        'mae' for robust regression, 'poisson_deviance' for count / frequency
        models (natural GLM deviance, scale-invariant to exposure).
    n_bootstrap:
        Bootstrap iterations. Default 200 — doubled from DriftAttributor
        because the single loop makes this cost-equivalent.
    n_permutations:
        Required when d > 12. Raises ValueError at construction if None
        and d > 12. Ignored for d <= 12 (full enumeration used).
    error_control:
        'fwer' for Bonferroni (FWER control); 'fdr' for Benjamini-Hochberg
        (FDR control). Use 'fdr' for d >= 10.
    masking_strategy:
        How to fill masked features. 'mean' uses (exposure-weighted) column
        means. 'median' is better for skewed features (sum insured, vehicle
        value). 'zero' matches the paper exactly but is unsafe when zero is
        outside the feature's support.
    exposure_weighted:
        If True, requires weights at fit_reference() and test(). Weights
        should be earned exposure in years.
    feature_pairs:
        If True, compute interaction drift for all feature pairs and return
        the top-5 in result.interaction_pairs.
    random_state:
        RNG seed. Default 42.

    Examples
    --------
    Motor frequency monitoring::

        from insurance_monitoring import InterpretableDriftDetector

        detector = InterpretableDriftDetector(
            model=fitted_glm,
            features=["driver_age", "vehicle_age", "ncb", "annual_mileage"],
            loss="poisson_deviance",
            error_control="fdr",
            exposure_weighted=True,
            n_bootstrap=200,
        )
        detector.fit_reference(
            X_ref, y_ref_claims,
            weights=exposure_ref,
        )
        result = detector.test(X_new, y_new_claims, weights=exposure_new)
        print(result.summary())
    """

    def __init__(
        self,
        model: Any,
        features: list,
        alpha: float = 0.05,
        loss: LossName = "mse",
        n_bootstrap: int = 200,
        n_permutations: Optional[int] = None,
        error_control: Literal["fwer", "fdr"] = "fwer",
        masking_strategy: MaskingStrategy = "mean",
        exposure_weighted: bool = False,
        feature_pairs: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        self.model = model
        self.features = features
        self.alpha = alpha
        self.loss = loss
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.error_control = error_control
        self.masking_strategy = masking_strategy
        self.exposure_weighted = exposure_weighted
        self.feature_pairs = feature_pairs
        self.random_state = random_state

        self._d = len(features)
        self._rng = np.random.default_rng(random_state)
        self._model_id_at_fit: Optional[int] = None

        # Validation
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_bootstrap < 10:
            raise ValueError(
                f"n_bootstrap must be >= 10 for reliable thresholds, got {n_bootstrap}"
            )
        if self._d > 12 and n_permutations is None:
            raise ValueError(
                f"d={self._d} features > 12: full enumeration (2^d subsets) is infeasible. "
                "Set n_permutations (e.g. n_permutations=256) to use random subset sampling."
            )

        # State set by fit_reference()
        self.is_fitted_: bool = False
        self.fill_values_: Optional[np.ndarray] = None
        self.reference_X_: Optional[np.ndarray] = None
        self.reference_y_: Optional[np.ndarray] = None
        self.reference_weights_: Optional[np.ndarray] = None
        self.cached_subset_risks_: dict = {}
        self._cached_subsets_: list = []
        self.n_detections_: int = 0

    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------

    def _coerce_inputs(
        self,
        X: Any,
        y: Any,
        weights: Any,
    ) -> tuple:
        """Coerce Polars or numpy inputs to float64 numpy arrays.

        Validates shapes and weight constraints.
        """
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy().astype(np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)

        if isinstance(y, pl.Series):
            y = y.to_numpy().astype(np.float64)
        else:
            y = np.asarray(y, dtype=np.float64)

        if X.shape[1] != self._d:
            raise ValueError(
                f"X has {X.shape[1]} columns but {self._d} features were specified"
            )
        if len(y) != len(X):
            raise ValueError(
                f"X has {len(X)} rows but y has {len(y)} elements"
            )

        w: Optional[np.ndarray] = None
        if weights is not None:
            if isinstance(weights, pl.Series):
                w = weights.to_numpy().astype(np.float64)
            else:
                w = np.asarray(weights, dtype=np.float64)
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            if len(w) != len(y):
                raise ValueError(
                    f"weights has {len(w)} elements but y has {len(y)} elements"
                )

        if self.exposure_weighted and w is None:
            raise ValueError(
                "exposure_weighted=True but weights=None. "
                "Pass earned exposure as weights."
            )

        return X, y, w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_reference(
        self,
        X_ref: Any,
        y_ref: Any,
        weights: Optional[Any] = None,
    ) -> "InterpretableDriftDetector":
        """Store reference window and pre-compute subset risks.

        Parameters
        ----------
        X_ref:
            Reference features. Shape (n_ref, d). Accepts numpy array or
            Polars DataFrame.
        y_ref:
            Reference targets. Shape (n_ref,). Accepts numpy array or Polars
            Series.
        weights:
            Exposure weights. Shape (n_ref,). Required when
            exposure_weighted=True.

        Returns
        -------
        self
        """
        X_ref, y_ref, w_ref = self._coerce_inputs(X_ref, y_ref, weights)

        self.reference_X_ = X_ref
        self.reference_y_ = y_ref
        self.reference_weights_ = w_ref

        # Compute fill values from reference window
        self.fill_values_ = _compute_fill_values(X_ref, self.masking_strategy, w_ref)

        # Enumerate subsets once — shared across all features
        self._cached_subsets_ = _enumerate_subsets(self._d, self.n_permutations, self._rng)

        # Pre-compute reference subset risks (cached across test() calls)
        self.cached_subset_risks_ = {}
        for S in self._cached_subsets_:
            self.cached_subset_risks_[S] = _subset_risk(
                self.model, X_ref, y_ref, S, self._d,
                self.fill_values_, self.loss, w_ref,
            )

        self._model_id_at_fit = id(self.model)
        self.is_fitted_ = True
        return self

    def test(
        self,
        X_new: Any,
        y_new: Any,
        weights: Optional[Any] = None,
    ) -> InterpretableDriftResult:
        """Test a new window for drift against the reference window.

        Parameters
        ----------
        X_new:
            New window features. Shape (n_new, d).
        y_new:
            New window targets. Shape (n_new,).
        weights:
            Exposure weights for the new window. Required when
            exposure_weighted=True.

        Returns
        -------
        InterpretableDriftResult
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit_reference() before test()."
            )

        if id(self.model) != self._model_id_at_fit:
            import warnings
            warnings.warn(
                "model identity has changed since fit_reference() was called. "
                "cached_subset_risks_ were computed under the old model. "
                "Call update_reference() after replacing the model.",
                UserWarning,
                stacklevel=2,
            )

        t0 = time.monotonic()
        X_new, y_new, w_new = self._coerce_inputs(X_new, y_new, weights)

        d = self._d
        n_new = len(y_new)
        subsets = self._cached_subsets_

        # 1. New-window subset risks (reference risks come from cache)
        risks_new = {}
        for S in subsets:
            risks_new[S] = _subset_risk(
                self.model, X_new, y_new, S, d,
                self.fill_values_, self.loss, w_new,
            )

        # 2. Observed test statistics
        test_stats = {}
        for k_idx, feat in enumerate(self.features):
            max_diff = 0.0
            for S in subsets:
                if k_idx in S:
                    continue
                S_k = S | frozenset([k_idx])
                if S_k not in self.cached_subset_risks_ or S_k not in risks_new:
                    continue
                delta_ref = self.cached_subset_risks_[S] - self.cached_subset_risks_[S_k]
                delta_new = risks_new[S] - risks_new[S_k]
                diff = abs(delta_ref - delta_new)
                if diff > max_diff:
                    max_diff = diff
            test_stats[feat] = n_new * max_diff

        # 3. Single bootstrap loop — produces both null stats and p-values
        n_ref = len(self.reference_y_)
        Z_all_X = np.vstack([self.reference_X_, X_new])
        Z_all_y = np.concatenate([self.reference_y_, y_new])
        if self.exposure_weighted and self.reference_weights_ is not None:
            Z_all_w = np.concatenate([self.reference_weights_, w_new])
        else:
            Z_all_w = None

        null_stats: dict = {feat: [] for feat in self.features}

        for _ in range(self.n_bootstrap):
            perm = self._rng.permutation(len(Z_all_y))
            idx_R = perm[:n_ref]
            idx_N = perm[n_ref:]

            X_b_R = Z_all_X[idx_R]
            y_b_R = Z_all_y[idx_R]
            X_b_N = Z_all_X[idx_N]
            y_b_N = Z_all_y[idx_N]
            w_b_R = Z_all_w[idx_R] if Z_all_w is not None else None
            w_b_N = Z_all_w[idx_N] if Z_all_w is not None else None

            fill_b = _compute_fill_values(X_b_R, self.masking_strategy, w_b_R)

            risks_b_R: dict = {}
            risks_b_N: dict = {}
            for S in subsets:
                risks_b_R[S] = _subset_risk(
                    self.model, X_b_R, y_b_R, S, d, fill_b, self.loss, w_b_R
                )
                risks_b_N[S] = _subset_risk(
                    self.model, X_b_N, y_b_N, S, d, fill_b, self.loss, w_b_N
                )

            for k_idx, feat in enumerate(self.features):
                max_diff = 0.0
                for S in subsets:
                    if k_idx in S:
                        continue
                    S_k = S | frozenset([k_idx])
                    if S_k not in risks_b_R:
                        continue
                    diff = abs(
                        (risks_b_R[S] - risks_b_R[S_k]) - (risks_b_N[S] - risks_b_N[S_k])
                    )
                    if diff > max_diff:
                        max_diff = diff
                null_stats[feat].append(len(y_b_N) * max_diff)

        # 4. Error control
        thresholds, p_values, attributed = _apply_error_control(
            test_stats, null_stats, self.features, self.alpha, d, self.error_control
        )

        # 5. Feature ranking DataFrame
        ranking_rows = [
            {
                "feature": feat,
                "test_statistic": test_stats[feat],
                "threshold": thresholds[feat],
                "ratio": test_stats[feat] / max(thresholds[feat], 1e-12),
                "p_value": p_values[feat],
                "drift_attributed": feat in attributed,
            }
            for feat in self.features
        ]
        ranking_df = (
            pl.DataFrame(ranking_rows)
            .sort("ratio", descending=True)
            .with_row_index("rank")
            .with_columns(pl.col("rank") + 1)
        )

        # 6. Interaction pairs
        interaction_pairs = None
        if self.feature_pairs and d >= 2:
            interaction_pairs = self._compute_interaction_pairs(
                X_new, y_new, w_new, risks_new
            )

        if len(attributed) > 0:
            self.n_detections_ += 1

        return InterpretableDriftResult(
            drift_detected=len(attributed) > 0,
            attributed_features=attributed,
            test_statistics=test_stats,
            thresholds=thresholds,
            p_values=p_values,
            error_control=self.error_control,
            alpha=self.alpha,
            feature_ranking=ranking_df,
            interaction_pairs=interaction_pairs,
            window_ref_size=n_ref,
            window_new_size=n_new,
            bootstrap_iterations=self.n_bootstrap,
            computation_time_s=time.monotonic() - t0,
        )

    def update_reference(
        self,
        X_new: Any,
        y_new: Any,
        weights: Optional[Any] = None,
    ) -> None:
        """Replace the reference window and recompute fill values and cached risks.

        Call this explicitly after model retraining and governance sign-off.
        Do not rely on automatic retraining — that conflates monitoring with
        model management.

        Parameters
        ----------
        X_new:
            New reference features. Shape (n, d).
        y_new:
            New reference targets. Shape (n,).
        weights:
            Exposure weights. Required when exposure_weighted=True.
        """
        self.fit_reference(X_new, y_new, weights)

    @classmethod
    def from_dataframe(
        cls,
        model: Any,
        df_ref: pl.DataFrame,
        df_new: pl.DataFrame,
        target_col: str,
        feature_cols: list,
        weight_col: Optional[str] = None,
        **kwargs: Any,
    ) -> InterpretableDriftResult:
        """Convenience for one-off quarterly checks.

        Creates a detector, fits on df_ref, tests on df_new, returns result.

        Parameters
        ----------
        model:
            Fitted model.
        df_ref:
            Reference period Polars DataFrame.
        df_new:
            New period Polars DataFrame.
        target_col:
            Name of the target column.
        feature_cols:
            List of feature column names.
        weight_col:
            Name of the exposure weight column. If provided, exposure_weighted
            is set to True automatically.
        **kwargs:
            Additional keyword arguments passed to InterpretableDriftDetector().

        Returns
        -------
        InterpretableDriftResult
        """
        exposure_weighted = weight_col is not None
        detector = cls(
            model=model,
            features=feature_cols,
            exposure_weighted=exposure_weighted,
            **kwargs,
        )

        w_ref = df_ref[weight_col] if weight_col else None
        w_new = df_new[weight_col] if weight_col else None

        detector.fit_reference(
            df_ref.select(feature_cols),
            df_ref[target_col],
            weights=w_ref,
        )
        return detector.test(
            df_new.select(feature_cols),
            df_new[target_col],
            weights=w_new,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_interaction_pairs(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        w_new: Optional[np.ndarray],
        risks_new: dict,
    ) -> pl.DataFrame:
        """Compute interaction drift for all feature pairs, return top-5.

        Uses pre-computed risks_new where possible to avoid redundant model
        calls. For pairs not in risks_new, falls back to direct computation.
        """
        pair_rows = []
        all_indices = list(range(self._d))

        for k1, k2 in combinations(all_indices, 2):
            S_empty = frozenset()
            S_k1 = frozenset([k1])
            S_k2 = frozenset([k2])
            S_k1k2 = frozenset([k1, k2])

            # Reference: use cached risks where available
            def get_ref_risk(S: frozenset) -> float:
                if S in self.cached_subset_risks_:
                    return self.cached_subset_risks_[S]
                return _subset_risk(
                    self.model, self.reference_X_, self.reference_y_, S, self._d,
                    self.fill_values_, self.loss, self.reference_weights_,
                )

            def get_new_risk(S: frozenset) -> float:
                if S in risks_new:
                    return risks_new[S]
                return _subset_risk(
                    self.model, X_new, y_new, S, self._d,
                    self.fill_values_, self.loss, w_new,
                )

            r_empty_ref = get_ref_risk(S_empty)
            r_k1_ref = get_ref_risk(S_k1)
            r_k2_ref = get_ref_risk(S_k2)
            r_k1k2_ref = get_ref_risk(S_k1k2)
            delta_ref = r_empty_ref - r_k1_ref - r_k2_ref + r_k1k2_ref

            r_empty_new = get_new_risk(S_empty)
            r_k1_new = get_new_risk(S_k1)
            r_k2_new = get_new_risk(S_k2)
            r_k1k2_new = get_new_risk(S_k1k2)
            delta_new = r_empty_new - r_k1_new - r_k2_new + r_k1k2_new

            pair_rows.append({
                "feature_1": self.features[k1],
                "feature_2": self.features[k2],
                "interaction_delta_ref": float(delta_ref),
                "interaction_delta_new": float(delta_new),
                "interaction_drift": float(abs(delta_ref - delta_new)),
            })

        if not pair_rows:
            return pl.DataFrame(schema={
                "feature_1": pl.Utf8,
                "feature_2": pl.Utf8,
                "interaction_delta_ref": pl.Float64,
                "interaction_delta_new": pl.Float64,
                "interaction_drift": pl.Float64,
                "rank": pl.Int64,
            })

        df = (
            pl.DataFrame(pair_rows)
            .sort("interaction_drift", descending=True)
            .head(5)
            .with_row_index("rank")
            .with_columns(pl.col("rank") + 1)
        )
        return df
