"""
Distribution-free SPC control charts via conformal prediction.

Classical Shewhart charts assume normal residuals. Insurance losses are not
normal — they are compound Poisson-Gamma or Tweedie, heteroskedastic, and
occasionally contaminated by large individual losses. The parametric control
limits are wrong by construction.

Conformal control charts replace parametric limits with a calibrated quantile
threshold derived from in-control data, with a finite-sample false alarm rate
guarantee:

    P(p(X_new) < alpha | in-control) <= alpha

for any exchangeable data sequence, regardless of distribution. This is exact
in finite samples, not asymptotic. The only requirement is exchangeability —
the calibration data must come from a stable, representative in-control period.

Reference: Burger (2025), "Distribution-Free Process Monitoring with Conformal
Prediction", arXiv:2512.23602. CC BY-4.0.

Classes
-------
ConformalControlChart
    Univariate distribution-free Shewhart-equivalent chart. Calibrate on
    in-control NCS scores, monitor new periods, get p-values and alarms.
    Static NCS helpers for common insurance residual types.

MultivariateConformalMonitor
    Multivariate monitoring via conformal anomaly detection. Accepts any
    sklearn-duck-typed anomaly model. Default: IsolationForest (sklearn
    optional). Produces a single conformal p-value per monitoring period.

ConformalChartResult
    Return type from both monitor() methods. Carries scores, p-values,
    alarms, threshold, and governance-ready output methods.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core math (module-private)
# ---------------------------------------------------------------------------

def _conformal_p_value(s_new: float, cal_scores: npt.NDArray[np.float64]) -> float:
    """Conformal p-value for a single new score.

    From ICP theory (Vovk, Gammerman, Shafer 2022):

        p(s_new) = (#{i : s_i >= s_new} + 1) / (n + 1)

    The +1 in numerator and denominator is the finite-sample ICP correction.
    It ensures p > 0 always — a p-value of exactly 0 would be statistically
    incoherent for a finite calibration set. Under the null (exchangeable data),
    this p-value is super-uniform: P(p < alpha) <= alpha exactly.

    Parameters
    ----------
    s_new : float
        New non-conformity score.
    cal_scores : ndarray
        Calibration NCS values (in-control reference).

    Returns
    -------
    float
        p-value in (0, 1].
    """
    n = len(cal_scores)
    return (float(np.sum(cal_scores >= s_new)) + 1.0) / (n + 1.0)


def _conformal_p_values(scores: npt.NDArray[np.float64], cal_scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorised conformal p-values for an array of new scores."""
    n = len(cal_scores)
    # Broadcasting: (len(scores), 1) >= (1, n) -> (len(scores), n)
    counts = np.sum(cal_scores[np.newaxis, :] >= scores[:, np.newaxis], axis=1)
    return (counts.astype(float) + 1.0) / (n + 1.0)


def _conformal_threshold(cal_scores: npt.NDArray[np.float64], alpha: float) -> float:
    """Control limit q for the conformal Shewhart chart.

    Equivalent to the ceil((1-alpha)(n+1))-th order statistic of cal_scores.
    The finite-sample correction (1 + 1/n) ensures the threshold level is
    achievable as a quantile of the empirical distribution.

    At alpha=0.05 with n >= 20, this gives a valid threshold.
    At alpha=0.0027 (3-sigma equivalent), requires n >= 370; below that, the
    level saturates at 1.0 and q = max(cal_scores), which is conservative.

    Parameters
    ----------
    cal_scores : ndarray
        In-control calibration NCS values.
    alpha : float
        Target false alarm rate.

    Returns
    -------
    float
        Control limit q. Flag if new score > q.
    """
    n = len(cal_scores)
    level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)
    return float(np.quantile(cal_scores, level))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConformalChartResult:
    """Result from ConformalControlChart.monitor() or MultivariateConformalMonitor.monitor().

    Attributes
    ----------
    scores : ndarray
        Non-conformity scores for the monitored observations.
    p_values : ndarray
        Conformal p-values in (0, 1]. p < alpha => alarm.
    threshold : float
        Control limit q. Equivalent signal: score > threshold.
    is_alarm : ndarray
        Boolean array. True where p_value < alpha.
    alpha : float
        False alarm rate used to calibrate the chart.
    n_cal : int
        Calibration set size. Determines threshold precision.
    """
    scores: npt.NDArray[np.float64]
    p_values: npt.NDArray[np.float64]
    threshold: float
    is_alarm: npt.NDArray[np.bool_]
    alpha: float
    n_cal: int

    @property
    def n_alarms(self) -> int:
        """Number of observations flagged as out-of-control."""
        return int(np.sum(self.is_alarm))

    @property
    def alarm_rate(self) -> float:
        """Empirical false alarm rate (alarms / total observations)."""
        n = len(self.is_alarm)
        return self.n_alarms / n if n > 0 else 0.0

    def to_polars(self) -> pl.DataFrame:
        """Return one row per monitored observation.

        Columns: obs_index (Int64), score (Float64), p_value (Float64),
        threshold (Float64), is_alarm (Boolean).
        """
        n = len(self.scores)
        return pl.DataFrame({
            "obs_index": list(range(n)),
            "score": self.scores.tolist(),
            "p_value": self.p_values.tolist(),
            "threshold": [self.threshold] * n,
            "is_alarm": self.is_alarm.tolist(),
        })

    def plot(self, title: str = "Conformal Control Chart") -> "Figure":
        """Plot the conformal p-value chart with alarm annotations.

        Two-panel figure: NCS values with control limit (top), conformal
        p-values with alpha line (bottom). Alarms highlighted in red.

        Parameters
        ----------
        title : str
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plot(). Install with: pip install matplotlib"
            )

        n = len(self.scores)
        idx = np.arange(n)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # --- Panel 1: NCS scores with threshold ---
        ax1.plot(idx, self.scores, color="steelblue", linewidth=1.2, label="NCS score")
        ax1.axhline(self.threshold, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Control limit (alpha={self.alpha:.4f})")
        if self.n_alarms > 0:
            alarm_idx = np.where(self.is_alarm)[0]
            ax1.scatter(alarm_idx, self.scores[alarm_idx], color="crimson",
                        zorder=5, s=50, label=f"Alarm ({self.n_alarms})")
        ax1.set_ylabel("Non-conformity score")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: conformal p-values ---
        ax2.plot(idx, self.p_values, color="steelblue", linewidth=1.2)
        ax2.axhline(self.alpha, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"alpha = {self.alpha:.4f}")
        if self.n_alarms > 0:
            alarm_idx = np.where(self.is_alarm)[0]
            ax2.scatter(alarm_idx, self.p_values[alarm_idx], color="crimson",
                        zorder=5, s=50)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Conformal p-value")
        ax2.set_xlabel("Observation index")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """Governance-ready plain-text paragraph describing the monitoring result.

        Suitable for inclusion in a model monitoring MI pack or validation report.

        Returns
        -------
        str
            Single paragraph summarising chart parameters, calibration, and
            monitoring outcome.
        """
        n_obs = len(self.scores)
        status = "OUT OF CONTROL" if self.n_alarms > 0 else "IN CONTROL"
        return (
            f"Conformal control chart (alpha={self.alpha:.4f}, n_cal={self.n_cal}). "
            f"Control limit: {self.threshold:.6g} (distribution-free, "
            f"arXiv:2512.23602). "
            f"Monitoring period: {n_obs} observations. "
            f"Alarms: {self.n_alarms}/{n_obs} ({self.alarm_rate:.1%}). "
            f"Status: {status}. "
            f"The false alarm rate is calibrated to {self.alpha:.4f} under the "
            f"exchangeability assumption. No distributional assumptions on residuals "
            f"are required (finite-sample guarantee, not asymptotic)."
        )


# ---------------------------------------------------------------------------
# ConformalControlChart
# ---------------------------------------------------------------------------

class ConformalControlChart:
    """Distribution-free Shewhart-equivalent control chart via conformal p-values.

    Replaces parametric 3-sigma limits with a conformal quantile threshold
    calibrated from in-control non-conformity scores (NCS). The false alarm
    rate is controlled at alpha exactly in finite samples under exchangeability
    — no normality, no constant variance required.

    Reference: Burger (2025), arXiv:2512.23602, Section 3.

    Parameters
    ----------
    alpha : float
        Target false alarm rate. Default 0.05 (5% per observation).
        Use 0.0027 for 3-sigma equivalent (requires n_cal >= 370).

    Examples
    --------
    Relative-residual monitoring on motor frequency model::

        from insurance_monitoring import ConformalControlChart

        cal_ncs = ConformalControlChart.ncs_relative_residual(
            actual=cal_df["claims"], predicted=cal_df["model_pred"]
        )
        chart = ConformalControlChart(alpha=0.05).fit(cal_ncs)

        monitor_ncs = ConformalControlChart.ncs_relative_residual(
            actual=new_df["claims"], predicted=new_df["model_pred"]
        )
        result = chart.monitor(monitor_ncs)
        print(result.summary())
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.cal_scores_: Optional[npt.NDArray[np.float64]] = None
        self.threshold_: Optional[float] = None
        self.n_cal_: Optional[int] = None

    def fit(self, calibration_scores: npt.ArrayLike) -> "ConformalControlChart":
        """Calibrate the chart from in-control NCS values.

        Parameters
        ----------
        calibration_scores : array-like
            Non-conformity scores from a representative in-control period.
            These must come from a stable window — including out-of-control
            periods will widen the threshold and reduce sensitivity.

        Returns
        -------
        self
        """
        cal = np.asarray(calibration_scores, dtype=float).ravel()
        if len(cal) < 2:
            raise ValueError("Need at least 2 calibration scores to fit a threshold.")
        if len(cal) < 20 and self.alpha >= 0.05:
            warnings.warn(
                f"Calibration set has only {len(cal)} observations. "
                f"At alpha={self.alpha}, n >= 20 is recommended for a valid threshold. "
                "The threshold will saturate at max(cal_scores).",
                UserWarning,
                stacklevel=2,
            )
        if len(cal) < 370 and self.alpha <= 0.003:
            warnings.warn(
                f"Calibration set has {len(cal)} observations. "
                f"At alpha={self.alpha:.4f} (3-sigma equivalent), n >= 370 is required. "
                "The threshold saturates at max(cal_scores) below this — "
                "consider alpha=0.05 for small calibration sets.",
                UserWarning,
                stacklevel=2,
            )
        self.cal_scores_ = cal
        self.n_cal_ = len(cal)
        self.threshold_ = _conformal_threshold(cal, self.alpha)
        return self

    def monitor(self, scores: npt.ArrayLike) -> ConformalChartResult:
        """Compute conformal p-values and alarms for new NCS values.

        Parameters
        ----------
        scores : array-like
            Non-conformity scores for the monitoring period.

        Returns
        -------
        ConformalChartResult
        """
        if self.cal_scores_ is None:
            raise RuntimeError("Call fit() before monitor().")
        s = np.asarray(scores, dtype=float).ravel()
        p_values = _conformal_p_values(s, self.cal_scores_)
        is_alarm = p_values < self.alpha
        return ConformalChartResult(
            scores=s,
            p_values=p_values,
            threshold=self.threshold_,  # type: ignore[arg-type]
            is_alarm=is_alarm,
            alpha=self.alpha,
            n_cal=self.n_cal_,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # NCS helpers — static, usable without instantiation
    # ------------------------------------------------------------------

    @staticmethod
    def ncs_absolute_residual(
        actual: npt.ArrayLike,
        predicted: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Absolute residual NCS: |actual - predicted|.

        Appropriate for homoskedastic residuals where variance is roughly
        constant across the predicted range. For insurance frequency models
        at policy level, prefer relative_residual due to heteroskedasticity.
        """
        a = np.asarray(actual, dtype=float)
        p = np.asarray(predicted, dtype=float)
        return np.abs(a - p)

    @staticmethod
    def ncs_relative_residual(
        actual: npt.ArrayLike,
        predicted: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Relative residual NCS: |actual - predicted| / predicted.

        Normalises by the predicted value so that the NCS distribution is
        comparable across policies with different exposure or base rate.
        Appropriate for Poisson frequency and Gamma severity models.

        Division by zero protected: denominator floored at 1e-9.
        """
        a = np.asarray(actual, dtype=float)
        p = np.asarray(predicted, dtype=float)
        safe_p = np.maximum(p, 1e-9)
        return np.abs(a - p) / safe_p

    @staticmethod
    def ncs_median_deviation(
        values: npt.ArrayLike,
        median: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Median absolute deviation NCS: |x - median(x)|.

        Robust to outliers. Use for univariate input monitoring (e.g. PSI
        values per feature) where the calibration median is the reference
        level. If median is None, computed from values — appropriate when
        using the monitoring data itself as calibration reference (unusual;
        normally fit() provides the calibration median implicitly).

        Parameters
        ----------
        values : array-like
            Observed values.
        median : float | None
            Reference median. If None, computed from values.
        """
        v = np.asarray(values, dtype=float)
        ref = float(np.median(v)) if median is None else float(median)
        return np.abs(v - ref)

    @staticmethod
    def ncs_studentized(
        actual: npt.ArrayLike,
        predicted: npt.ArrayLike,
        local_vol: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Studentized residual NCS: |actual - predicted| / local_vol.

        Adapts the NCS to local volatility estimates. For a GLM with
        overdispersion modelling, local_vol can be the square root of the
        predicted variance. A sudden increase in interval width
        (threshold computed from this NCS widening) is a leading indicator
        of instability before the mean has shifted.

        Division by zero protected: local_vol floored at 1e-9.
        """
        a = np.asarray(actual, dtype=float)
        p = np.asarray(predicted, dtype=float)
        v = np.asarray(local_vol, dtype=float)
        safe_v = np.maximum(v, 1e-9)
        return np.abs(a - p) / safe_v


# ---------------------------------------------------------------------------
# MultivariateConformalMonitor
# ---------------------------------------------------------------------------

class MultivariateConformalMonitor:
    """Multivariate conformal anomaly detection for model health monitoring.

    Reframes multivariate SPC as conformal anomaly detection (Burger 2025,
    Section 4). Trains an unsupervised anomaly detector on in-control state
    vectors, calibrates a conformal p-value distribution from a held-out
    in-control calibration set, and produces a single p-value per monitoring
    period.

    The anomaly detector can be any object with:
    - .fit(X: ndarray) -> self
    - .decision_function(X: ndarray) -> ndarray  (higher = more normal)
      or .score_samples(X: ndarray) -> ndarray  (higher = more normal)

    IsolationForest (scikit-learn) satisfies this interface and is used as
    the default. Pass model=None to use the default; pass any duck-typed model
    to override.

    Note on score direction: scikit-learn anomaly detectors use the convention
    where higher decision_function / score_samples = more normal (less anomalous).
    The NCS must be monotone with anomalousness, so we negate the model output:
    NCS = -decision_function(x). Higher NCS = more anomalous.

    Parameters
    ----------
    model : object | None
        Anomaly detection model. If None, uses IsolationForest with
        contamination=0.01. Must have .fit() and .decision_function() or
        .score_samples().
    alpha : float
        False alarm rate. Default 0.05.

    Examples
    --------
    Monitoring a vector of model health metrics each reporting cycle::

        from insurance_monitoring import MultivariateConformalMonitor
        import numpy as np

        # Monthly monitoring vectors: [PSI_age, PSI_area, AE_freq, AE_sev, Gini]
        X_train = ...  # 12 months in-control
        X_cal = ...    # 6 months held-out in-control
        X_new = ...    # current monitoring period

        monitor = MultivariateConformalMonitor(alpha=0.05).fit(X_train, X_cal)
        result = monitor.monitor(X_new)
        print(result.summary())
    """

    def __init__(self, model=None, alpha: float = 0.05) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._model_arg = model
        self.model_ = None
        self.cal_scores_: Optional[npt.NDArray[np.float64]] = None
        self.threshold_: Optional[float] = None
        self.n_cal_: Optional[int] = None

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        X_cal: npt.NDArray[np.float64],
    ) -> "MultivariateConformalMonitor":
        """Train anomaly model on X_train and calibrate thresholds on X_cal.

        X_train and X_cal must both come from the in-control period. Keeping
        them separate (rather than using cross-conformal) is deliberately simple
        and avoids the computational cost of cross-validation at calibration time.

        Parameters
        ----------
        X_train : ndarray of shape (n_train, d)
            In-control state vectors for training the anomaly detector.
        X_cal : ndarray of shape (n_cal, d)
            Held-out in-control state vectors for conformal calibration.
            n_cal >= 20 required for alpha=0.05.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=float)
        X_cal = np.asarray(X_cal, dtype=float)

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_cal.ndim == 1:
            X_cal = X_cal.reshape(-1, 1)

        if len(X_cal) < 2:
            raise ValueError("Need at least 2 calibration vectors.")

        # Instantiate or use provided model
        if self._model_arg is None:
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn is required for the default MultivariateConformalMonitor "
                    "model. Install with: pip install scikit-learn. "
                    "Or pass your own model with .fit() and .decision_function()."
                ) from exc
            self.model_ = IsolationForest(contamination=0.01, random_state=42)
        else:
            self.model_ = self._model_arg

        self.model_.fit(X_train)

        # Compute calibration NCS scores
        cal_scores = self._score(X_cal)
        self.cal_scores_ = cal_scores
        self.n_cal_ = len(cal_scores)
        self.threshold_ = _conformal_threshold(cal_scores, self.alpha)

        if self.n_cal_ < 20 and self.alpha >= 0.05:
            warnings.warn(
                f"Calibration set has only {self.n_cal_} vectors. "
                f"At alpha={self.alpha}, n >= 20 is recommended.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def monitor(self, X_new: npt.NDArray[np.float64]) -> ConformalChartResult:
        """Compute conformal p-values for new state vectors.

        Parameters
        ----------
        X_new : ndarray of shape (n_new, d) or (d,)
            New monitoring state vectors. One row per monitoring period.

        Returns
        -------
        ConformalChartResult
        """
        if self.cal_scores_ is None:
            raise RuntimeError("Call fit() before monitor().")

        X = np.asarray(X_new, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = self._score(X)
        p_values = _conformal_p_values(scores, self.cal_scores_)
        is_alarm = p_values < self.alpha

        return ConformalChartResult(
            scores=scores,
            p_values=p_values,
            threshold=self.threshold_,  # type: ignore[arg-type]
            is_alarm=is_alarm,
            alpha=self.alpha,
            n_cal=self.n_cal_,  # type: ignore[arg-type]
        )

    def _score(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute NCS from model. Higher = more anomalous.

        Uses decision_function if available, falling back to score_samples.
        Negates the output because sklearn anomaly detectors return higher
        values for more normal observations.
        """
        if hasattr(self.model_, "decision_function"):
            raw = self.model_.decision_function(X)
        elif hasattr(self.model_, "score_samples"):
            raw = self.model_.score_samples(X)
        else:
            raise AttributeError(
                "Model must have .decision_function() or .score_samples(). "
                f"Got {type(self.model_).__name__} with neither method."
            )
        return -np.asarray(raw, dtype=float)
