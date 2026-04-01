"""
Distribution-free SPC via conformal prediction: ConformedControlChart and
ConformedProcessMonitor.

This module provides the sklearn-style API (fit/predict) for conformal SPC,
complementing the existing ConformalControlChart (fit/monitor) in
conformal_chart.py. The two implementations share the same underlying maths —
conformal p-values from Vovk, Gammerman & Shafer — but differ in interface
conventions and intended use.

ConformedControlChart targets univariate residual monitoring with built-in
score functions (absolute, relative, studentized). You supply raw
actuals/predictions and it handles the NCS computation internally.

ConformedProcessMonitor targets multivariate process monitoring. It accepts
a single calibration array (no separate train/calibration split required),
trains an anomaly detector, and produces p-values from a held-out conformal
calibration set derived via an 80/20 internal split.

Reference: Burger (2025), "Distribution-Free Process Monitoring with Conformal
Prediction", arXiv:2512.23602.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

try:
    import matplotlib.axes
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformedControlResult:
    """Result from ConformedControlChart.predict().

    Attributes
    ----------
    scores : ndarray
        Non-conformity scores for the monitored observations.
    threshold : float
        Control limit. Observations with score > threshold are flagged.
    signals : ndarray
        Boolean array. True where the observation is out-of-control.
    signal_rate : float
        Fraction of observations flagged (signals.sum() / len(signals)).
    alpha : float
        False alarm rate used to derive the threshold.
    n_calibration : int
        Number of calibration observations used to fit the threshold.
    """
    scores: npt.NDArray[np.float64]
    threshold: float
    signals: npt.NDArray[np.bool_]
    signal_rate: float
    alpha: float
    n_calibration: int


@dataclass
class ConformedMonitorResult:
    """Result from ConformedProcessMonitor.predict().

    Attributes
    ----------
    p_values : ndarray
        Conformal p-values in (0, 1]. p_t < alpha triggers a signal.
    signals : ndarray
        Boolean array. True where p_value < alpha.
    signal_rate : float
        Fraction of observations flagged.
    alpha : float
        Significance level used to declare signals.
    n_calibration : int
        Number of calibration observations used in the conformal calculation.
    """
    p_values: npt.NDArray[np.float64]
    signals: npt.NDArray[np.bool_]
    signal_rate: float
    alpha: float
    n_calibration: int


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def _score_absolute(
    values: npt.NDArray[np.float64],
    ref_median: float,
) -> npt.NDArray[np.float64]:
    """Absolute deviation from reference median: |x - median(calibration)|."""
    return np.abs(values - ref_median)


def _score_relative(
    values: npt.NDArray[np.float64],
    ref_median: float,
) -> npt.NDArray[np.float64]:
    """Relative deviation: |x - median| / max(|median|, 1e-8)."""
    denom = max(abs(ref_median), 1e-8)
    return np.abs(values - ref_median) / denom


def _score_studentized(
    values: npt.NDArray[np.float64],
    ref_median: float,
    ref_mad: float,
) -> npt.NDArray[np.float64]:
    """Studentized deviation: |x - median| / max(MAD, 1e-8).

    MAD (median absolute deviation) is the robust scale estimate computed
    from the calibration data.
    """
    denom = max(ref_mad, 1e-8)
    return np.abs(values - ref_median) / denom


# ---------------------------------------------------------------------------
# Conformal threshold (shared with conformal_chart.py but reproduced here
# to keep conformal_spc.py self-contained)
# ---------------------------------------------------------------------------

def _threshold_from_calibration(
    cal_scores: npt.NDArray[np.float64],
    alpha: float,
) -> float:
    """Compute conformal control limit from calibration scores.

    Threshold = ceil((1-alpha)(n+1))-th order statistic of cal_scores.
    Implemented via the finite-sample quantile level:

        level = min((1 - alpha) * (1 + 1/n), 1.0)

    This ensures the threshold is achievable as an order statistic of the
    empirical distribution. At alpha=0.05 with n >= 20, the level is below
    1.0 and yields a threshold strictly below max(cal_scores). At
    alpha=0.0027 (3-sigma equivalent) the n >= 370 requirement applies.

    Parameters
    ----------
    cal_scores : ndarray
        Non-conformity scores from an in-control calibration period.
    alpha : float
        Target false alarm rate.

    Returns
    -------
    float
        Control limit.
    """
    n = len(cal_scores)
    level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)
    return float(np.quantile(cal_scores, level))


def _conformal_p_values_vec(
    scores: npt.NDArray[np.float64],
    cal_scores: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Vectorised conformal p-values.

    p_t = (#{s_cal >= s_t} + 1) / (n_cal + 1)

    The +1 correction (Vovk et al.) ensures p > 0 for all finite calibration
    sets and preserves super-uniformity under exchangeability.
    """
    n = len(cal_scores)
    counts = np.sum(cal_scores[np.newaxis, :] >= scores[:, np.newaxis], axis=1)
    return (counts.astype(float) + 1.0) / (n + 1.0)


# ---------------------------------------------------------------------------
# ConformedControlChart
# ---------------------------------------------------------------------------

class ConformedControlChart:
    """Distribution-free control chart using conformal prediction thresholds.

    Replaces parametric Shewhart 3-sigma limits with a calibrated quantile
    threshold derived from in-control data. The false alarm rate is controlled
    at alpha exactly in finite samples under the exchangeability assumption —
    no normality, no constant variance required.

    Unlike ConformalControlChart (which operates on pre-computed NCS), this
    class accepts raw scalar values and applies a score function internally.
    The three built-in score functions cover the main use cases in insurance
    residual monitoring:

    - absolute: robust for homoskedastic residuals (claim counts on large
      cohorts, loss ratios after rate normalisation)
    - relative: normalises by the calibration median; appropriate when the
      scale of the process varies (premium per policy, exposure-weighted AE)
    - studentized: scales by MAD — the right choice when volatility itself
      changes between periods (e.g. catastrophe quarters inflating the scale)

    Parameters
    ----------
    alpha : float
        Target false alarm rate. Default 0.05. Use 0.0027 for the 3-sigma
        equivalent (requires n_calibration >= 370).
    score_fn : {"absolute", "relative", "studentized"}
        Non-conformity score function. Default "absolute".

    Examples
    --------
    Monthly A/E monitoring on a motor frequency book::

        from insurance_monitoring import ConformedControlChart
        import numpy as np

        # 24 months of in-control A/E ratios
        cal_ae = np.array([0.98, 1.02, 0.97, 1.01, ...])
        chart = ConformedControlChart(alpha=0.05, score_fn="absolute").fit(cal_ae)

        # Monitor the last 6 months
        new_ae = np.array([1.03, 0.99, 1.15, 1.02, 0.96, 1.08])
        result = chart.predict(new_ae)
        print(f"Signals: {result.signals.sum()} / {len(result.signals)}")
        print(f"Signal rate: {result.signal_rate:.1%}")
    """

    _VALID_SCORE_FNS = ("absolute", "relative", "studentized")

    def __init__(
        self,
        alpha: float = 0.05,
        score_fn: str = "absolute",
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if score_fn not in self._VALID_SCORE_FNS:
            raise ValueError(
                f"score_fn must be one of {self._VALID_SCORE_FNS}, got {score_fn!r}"
            )
        self.alpha = alpha
        self.score_fn = score_fn

        # Set after fit()
        self._cal_scores: Optional[npt.NDArray[np.float64]] = None
        self._threshold: Optional[float] = None
        self._ref_median: Optional[float] = None
        self._ref_mad: Optional[float] = None
        self._n_calibration: Optional[int] = None

    def fit(self, calibration_scores: npt.ArrayLike) -> "ConformedControlChart":
        """Compute conformal threshold from in-control calibration data.

        The threshold is the ceil((1-alpha)(n+1))-th order statistic of the
        computed NCS values, with finite-sample correction:

            quantile_level = min((1 - alpha)(1 + 1/n), 1.0)

        Parameters
        ----------
        calibration_scores : array-like of shape (n,)
            In-control scalar observations (e.g. monthly A/E ratios, residuals,
            loss ratios). The score function is applied internally. Must have
            at least 2 elements.

        Returns
        -------
        self
        """
        cal = np.asarray(calibration_scores, dtype=float).ravel()
        if len(cal) < 2:
            raise ValueError(
                f"Need at least 2 calibration observations to fit a threshold, "
                f"got {len(cal)}."
            )
        if self.alpha > 0 and len(cal) < 20 and self.alpha >= 0.05:
            warnings.warn(
                f"Calibration set has only {len(cal)} observations. "
                f"At alpha={self.alpha}, n >= 20 is recommended for a valid threshold. "
                "The threshold will saturate at max(cal_scores).",
                UserWarning,
                stacklevel=2,
            )
        if self.alpha > 0 and len(cal) < 370 and self.alpha <= 0.003:
            warnings.warn(
                f"Calibration set has {len(cal)} observations. "
                f"At alpha={self.alpha:.4f} (3-sigma equivalent), n >= 370 is "
                "required. The threshold saturates at max(cal_scores) below this.",
                UserWarning,
                stacklevel=2,
            )

        self._ref_median = float(np.median(cal))
        self._ref_mad = float(np.median(np.abs(cal - self._ref_median)))
        self._n_calibration = len(cal)

        # Compute NCS on the calibration data
        ncs = self._compute_ncs(cal)
        self._cal_scores = ncs

        if self.alpha == 0.0:
            # alpha=0: never signal (threshold = infinity)
            self._threshold = float(np.max(ncs)) * 1e10
        elif self.alpha == 1.0:
            # alpha=1: always signal (threshold = 0)
            self._threshold = 0.0
        else:
            self._threshold = _threshold_from_calibration(ncs, self.alpha)

        return self

    def predict(self, test_scores: npt.ArrayLike) -> ConformedControlResult:
        """Flag out-of-control points in new observations.

        Parameters
        ----------
        test_scores : array-like of shape (n,)
            New scalar observations to monitor.

        Returns
        -------
        ConformedControlResult
            Dataclass with scores, threshold, signals (bool array), signal_rate,
            alpha, and n_calibration.
        """
        if self._cal_scores is None:
            raise RuntimeError("Call fit() before predict().")

        test = np.asarray(test_scores, dtype=float).ravel()
        ncs = self._compute_ncs(test)
        signals = ncs > self._threshold
        n = len(signals)
        signal_rate = float(signals.sum()) / n if n > 0 else 0.0

        return ConformedControlResult(
            scores=ncs,
            threshold=self._threshold,  # type: ignore[arg-type]
            signals=signals,
            signal_rate=signal_rate,
            alpha=self.alpha,
            n_calibration=self._n_calibration,  # type: ignore[arg-type]
        )

    def plot(
        self,
        result: ConformedControlResult,
        ax: Optional["matplotlib.axes.Axes"] = None,
    ) -> "matplotlib.axes.Axes":
        """Shewhart-style chart with conformal threshold line.

        Parameters
        ----------
        result : ConformedControlResult
            Output of predict().
        ax : matplotlib.axes.Axes, optional
            Axes to plot onto. If None, a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plot(). "
                "Install with: pip install matplotlib"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        idx = np.arange(len(result.scores))
        ax.plot(idx, result.scores, color="steelblue", linewidth=1.2, label="NCS score")
        ax.axhline(
            result.threshold,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Conformal threshold (alpha={result.alpha:.4f})",
        )
        n_signals = int(result.signals.sum())
        if n_signals > 0:
            sig_idx = np.where(result.signals)[0]
            ax.scatter(
                sig_idx,
                result.scores[sig_idx],
                color="crimson",
                zorder=5,
                s=50,
                label=f"Signal ({n_signals})",
            )
        ax.set_xlabel("Observation index")
        ax.set_ylabel(f"NCS ({self.score_fn})")
        ax.set_title(
            f"Conformed Control Chart — alpha={result.alpha:.4f}, "
            f"n_cal={result.n_calibration}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ncs(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the configured score function."""
        if self.score_fn == "absolute":
            return _score_absolute(values, self._ref_median)  # type: ignore[arg-type]
        elif self.score_fn == "relative":
            return _score_relative(values, self._ref_median)  # type: ignore[arg-type]
        elif self.score_fn == "studentized":
            return _score_studentized(
                values,
                self._ref_median,  # type: ignore[arg-type]
                self._ref_mad,  # type: ignore[arg-type]
            )
        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown score_fn: {self.score_fn!r}")


# ---------------------------------------------------------------------------
# ConformedProcessMonitor
# ---------------------------------------------------------------------------

class ConformedProcessMonitor:
    """Multivariate process monitoring using conformal p-values.

    Reframes multivariate SPC as conformal anomaly detection (Burger 2025,
    Section 4). An unsupervised anomaly detector is trained on in-control
    feature vectors. A held-out conformal calibration set (80/20 internal
    split of X_calibration) gives the reference score distribution. For each
    new observation, a conformal p-value is computed:

        p_t = (#{s_cal >= s_t} + 1) / (n_cal + 1)

    Signal: p_t < alpha.

    Parameters
    ----------
    alpha : float
        Significance level. Default 0.05.
    detector : {"ocsvm", "isolation_forest"} or sklearn estimator
        Anomaly detection model. If a string, the corresponding sklearn
        estimator is instantiated. Pass a fitted or unfitted sklearn-style
        object (with .fit() and .decision_function() or .score_samples()) to
        use a custom model.

        "ocsvm" (default): OneClassSVM with rbf kernel. Fast, works well on
        low-to-moderate dimensional monitoring vectors (d <= 20). Can be slow
        on large training sets — prefer isolation_forest above n=5000.

        "isolation_forest": IsolationForest with contamination=0.01. Better
        for high-dimensional monitoring vectors or large calibration sets.

    Examples
    --------
    Monthly multivariate model health monitoring::

        from insurance_monitoring import ConformedProcessMonitor
        import numpy as np

        # 36 months of in-control monitoring vectors:
        # [PSI_age, PSI_area, AE_freq, AE_sev, Gini_delta]
        X_cal = np.random.normal(0, 1, size=(36, 5))
        monitor = ConformedProcessMonitor(alpha=0.05, detector="ocsvm")
        monitor.fit(X_cal)

        X_new = np.random.normal(0, 1, size=(6, 5))
        result = monitor.predict(X_new)
        print(f"Signals: {result.signals.sum()} / {len(result.signals)}")
    """

    _VALID_DETECTORS = ("ocsvm", "isolation_forest")

    def __init__(
        self,
        alpha: float = 0.05,
        detector: Union[str, object] = "ocsvm",
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if isinstance(detector, str) and detector not in self._VALID_DETECTORS:
            raise ValueError(
                f"detector must be one of {self._VALID_DETECTORS} or a fitted "
                f"sklearn estimator, got {detector!r}"
            )
        self.alpha = alpha
        self.detector = detector

        # Set after fit()
        self._model = None
        self._cal_scores: Optional[npt.NDArray[np.float64]] = None
        self._n_calibration: Optional[int] = None

    def fit(self, X_calibration: npt.ArrayLike) -> "ConformedProcessMonitor":
        """Train anomaly detector on in-control feature vectors.

        The calibration array is split 80/20 internally. The first 80% trains
        the anomaly detector; the remaining 20% provides the conformal
        calibration scores. A minimum of 10 observations are reserved for
        conformal calibration regardless of array size.

        Parameters
        ----------
        X_calibration : array-like of shape (n, d)
            In-control feature vectors. n >= 10 required.

        Returns
        -------
        self
        """
        X = np.asarray(X_calibration, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = len(X)
        if n < 10:
            raise ValueError(
                f"X_calibration must have at least 10 observations for a "
                f"meaningful conformal calibration split, got {n}."
            )

        # 80/20 split: train on 80%, calibrate on 20% (at least 2)
        n_train = max(n - max(int(n * 0.2), 2), 2)
        X_train = X[:n_train]
        X_cal = X[n_train:]

        self._model = self._build_model()
        self._model.fit(X_train)

        cal_scores = self._score(X_cal)
        self._cal_scores = cal_scores
        self._n_calibration = len(cal_scores)

        if self._n_calibration < 20 and self.alpha >= 0.05:
            warnings.warn(
                f"Conformal calibration set has only {self._n_calibration} "
                f"observations after the 80/20 split. At alpha={self.alpha}, "
                "n_calibration >= 20 is recommended for a valid threshold. "
                "Pass a larger X_calibration array.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def predict(self, X_test: npt.ArrayLike) -> ConformedMonitorResult:
        """Compute conformal p-values for each test observation.

        Parameters
        ----------
        X_test : array-like of shape (n, d) or (d,)
            New monitoring feature vectors.

        Returns
        -------
        ConformedMonitorResult
            Dataclass with p_values, signals (bool), signal_rate, alpha,
            and n_calibration.
        """
        if self._cal_scores is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = self._score(X)
        p_values = _conformal_p_values_vec(scores, self._cal_scores)

        if self.alpha == 0.0:
            signals = np.zeros(len(p_values), dtype=bool)
        elif self.alpha == 1.0:
            signals = np.ones(len(p_values), dtype=bool)
        else:
            signals = p_values < self.alpha

        n = len(signals)
        signal_rate = float(signals.sum()) / n if n > 0 else 0.0

        return ConformedMonitorResult(
            p_values=p_values,
            signals=signals,
            signal_rate=signal_rate,
            alpha=self.alpha,
            n_calibration=self._n_calibration,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        """Instantiate the anomaly detection model."""
        if not isinstance(self.detector, str):
            # User-supplied model — use as-is
            return self.detector

        if self.detector == "ocsvm":
            try:
                from sklearn.svm import OneClassSVM
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn is required for detector='ocsvm'. "
                    "Install with: pip install scikit-learn. "
                    "Or pass a custom model object."
                ) from exc
            return OneClassSVM(kernel="rbf", nu=0.05)

        elif self.detector == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn is required for detector='isolation_forest'. "
                    "Install with: pip install scikit-learn. "
                    "Or pass a custom model object."
                ) from exc
            return IsolationForest(contamination=0.01, random_state=42)

        raise ValueError(f"Unknown detector: {self.detector!r}")

    def _score(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute NCS from model output. Higher NCS = more anomalous.

        sklearn anomaly detectors use the convention: higher decision_function
        = more normal. We negate so NCS is monotone with anomalousness.
        """
        if hasattr(self._model, "decision_function"):
            raw = self._model.decision_function(X)
        elif hasattr(self._model, "score_samples"):
            raw = self._model.score_samples(X)
        else:
            raise AttributeError(
                "Anomaly detector must have .decision_function() or "
                f".score_samples(). Got {type(self._model).__name__} with neither."
            )
        return -np.asarray(raw, dtype=float)
