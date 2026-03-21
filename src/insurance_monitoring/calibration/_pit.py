"""
Anytime-valid calibration change detection via probability integral transforms.

PITMonitor constructs a mixture e-process over probability integral transforms
(PITs). The guarantee is:

    P(sup_t M_t >= 1/alpha | model calibrated) <= alpha

for all t, forever, with no pre-specified horizon. An insurer can check weekly,
monthly, or after every renewal without correction or inflation.

Critical distinction from CalibrationChecker
--------------------------------------------
PITMonitor detects *changes* in calibration, not absolute miscalibration.
A consistently over-predicting model will not trigger — only a model that was
stable and then drifted. The two methods are complementary:

- ``CalibrationChecker`` at model launch: absolute calibration on holdout data
- ``PITMonitor`` in production: ongoing drift detection with formal error control

PIT computation by distribution
---------------------------------
The monitor is model-agnostic — it accepts any value in [0, 1]. Canonical
computations for UK insurance GLMs::

    # Poisson frequency
    from scipy.stats import poisson
    pit = float(poisson.cdf(y_claims, mu=exposure * lambda_hat))

    # Gamma severity (shape = 1/phi, scale = phi * mu)
    from scipy.stats import gamma
    pit = float(gamma.cdf(y_loss, a=1/phi, scale=phi*mu_hat))

    # Negative Binomial
    from scipy.stats import nbinom
    pit = float(nbinom.cdf(y_claims, n=r, p=r/(r+mu)))

    # Normal
    from scipy.stats import norm
    pit = float(norm.cdf(y, loc=mu_hat, scale=sigma_hat))

    # Tweedie: no closed-form CDF; use the `tweedie` package or saddlepoint approx.

Segment monitors
-----------------
For portfolios with distinct subgroups (vehicle types, peril classes), run a
separate PITMonitor per segment. Do not pool segments — miscalibration in one
segment will be diluted by a stable segment, reducing detection power.

Reference
---------
Henzi, Murph, Ziegel (2025). Anytime valid change detection for calibration.
arXiv:2603.13156.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
from sortedcontainers import SortedList


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class PITAlarm:
    """Result returned by :meth:`PITMonitor.update`.

    Evaluates as bool: ``if alarm:`` is True when triggered.
    """

    triggered: bool
    """True when M_t >= 1/alpha."""

    time: int
    """Current step t (1-indexed)."""

    evidence: float
    """Current e-process value M_t."""

    threshold: float
    """Rejection threshold 1/alpha."""

    changepoint: int | None
    """Estimated changepoint tau_hat if triggered, else None."""

    def __bool__(self) -> bool:
        return self.triggered


@dataclass
class PITSummary:
    """Full monitoring state snapshot from :meth:`PITMonitor.summary`."""

    t: int
    """Current step count."""

    alarm_triggered: bool
    """Whether an alarm has fired."""

    alarm_time: int | None
    """Step at which alarm first fired, or None."""

    evidence: float
    """Current e-process value M_t."""

    threshold: float
    """Rejection threshold 1/alpha."""

    changepoint: int | None
    """Estimated changepoint, or None if no alarm."""

    calibration_score: float | None
    """1 - KS statistic for PIT uniformity. None if t == 0."""

    n_observations: int
    """Number of observations processed (equal to t)."""


# ---------------------------------------------------------------------------
# PITMonitor
# ---------------------------------------------------------------------------


class PITMonitor:
    """Anytime-valid calibration change detector for deployed pricing models.

    Implements the mixture e-process of Henzi, Murph, Ziegel (2025) over
    conformal p-values derived from probability integral transforms.

    Parameters
    ----------
    alpha
        False alarm rate bound. Must be in (0, 1). The monitor guarantees
        P(ever alarm | calibrated) <= alpha for any monitoring horizon.
    n_bins
        Histogram bins B for the density estimator. Range [5, 500].
        More bins give better resolution but slower warm-up. See portfolio
        scale guidance in the module docstring.
    weight_schedule
        Mixing weights w(t) for t >= 1. Must be non-negative with sum = 1.
        Default: w(t) = 1 / (t(t+1)), which satisfies sum = 1 exactly.
        Custom schedules are not persisted by :meth:`save`.
    rng
        Seed or numpy Generator for the tie-breaking randomisation in
        conformal p-value computation. Use an integer seed for reproducibility.

    Examples
    --------
    Monitoring a Poisson frequency model monthly::

        from insurance_monitoring import PITMonitor
        from scipy.stats import poisson

        monitor = PITMonitor(alpha=0.05, rng=42)

        for month_batch in monthly_claims:
            for i, row in enumerate(month_batch):
                mu = row.exposure * row.lambda_hat
                pit = float(poisson.cdf(row.claims, mu))
                alarm = monitor.update(pit)
                if alarm:
                    print(f"Alarm at step {alarm.time}, changepoint ~{alarm.changepoint}")
                    break
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins: int = 100,
        weight_schedule: Callable[[int], float] | None = None,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (5 <= n_bins <= 500):
            raise ValueError(f"n_bins must be in [5, 500], got {n_bins}")

        if weight_schedule is not None:
            _validate_weight_schedule(weight_schedule)

        self.alpha = alpha
        self.n_bins = n_bins
        self._weight_schedule = weight_schedule
        self._threshold = 1.0 / alpha

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(int(rng))

        self._rng_seed = rng if isinstance(rng, (int, type(None))) else None

        self._init_state()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Initialise all mutable monitor state to t=0."""
        self.t: int = 0
        self._sorted_pits: SortedList = SortedList()
        self._bin_counts: np.ndarray = np.ones(self.n_bins, dtype=np.float64)
        self._M: float = 0.0
        self._history: list[tuple[float, float, float]] = []  # (pit, pval, M_t)
        self.alarm_triggered: bool = False
        self.alarm_time: int | None = None

    def _w(self, t: int) -> float:
        """Mixing weight w_t for step t (1-indexed)."""
        if self._weight_schedule is not None:
            return self._weight_schedule(t)
        return 1.0 / (t * (t + 1))

    def _compute_pvalue(self, pit: float) -> float:
        """Convert a PIT to a conformal p-value with tie-breaking randomisation.

        Inserts pit into the sorted structure (including current), then computes
        rank using bisect. V_t is a CONTINUOUS Uniform(0, right-left) draw —
        this is essential for p_t ~ Uniform(0,1) exactly under H0.

        For continuous PITs (no exact ties between observations), right-left = 1
        (only U_t itself), and V_t ~ U(0,1), giving p_t ~ Uniform(0,1).
        """
        # Insert into sorted list (already incremented t before calling)
        self._sorted_pits.add(pit)
        t = self.t  # already updated by caller

        left = self._sorted_pits.bisect_left(pit)
        right = self._sorted_pits.bisect_right(pit)
        # V_t ~ Continuous Uniform(0, right - left).
        # Even when right-left = 1 (no prior ties, only U_t in the group),
        # this randomisation makes p_t exactly Uniform(0,1) under H0.
        v = float(self._rng.uniform(0.0, float(right - left)))
        return (left + v) / t

    def _compute_evalue(self, pval: float) -> float:
        """Compute histogram e-value for p-value, then update bin counts.

        The density is computed BEFORE updating bin_counts to ensure
        F_{t-1}-measurability: E[e_t | H0] = 1.
        """
        bin_idx = min(int(math.floor(pval * self.n_bins)), self.n_bins - 1)
        density = self._bin_counts[bin_idx] / float(np.sum(self._bin_counts))
        e_t = density * self.n_bins
        self._bin_counts[bin_idx] += 1.0
        return e_t

    def _step(self, pit: float) -> PITAlarm:
        """Process one PIT. Increments t and updates all state."""
        self.t += 1

        pval = self._compute_pvalue(pit)
        e_t = self._compute_evalue(pval)

        if not self.alarm_triggered:
            w_t = self._w(self.t)
            self._M = e_t * (self._M + w_t)
            if self._M >= self._threshold:
                self.alarm_triggered = True
                self.alarm_time = self.t

        self._history.append((pit, pval, self._M))

        return PITAlarm(
            triggered=self.alarm_triggered,
            time=self.t,
            evidence=self._M,
            threshold=self._threshold,
            changepoint=self.changepoint() if self.alarm_triggered else None,
        )

    # ------------------------------------------------------------------
    # Public core API
    # ------------------------------------------------------------------

    def update(self, pit: float, exposure: float = 1.0) -> PITAlarm:
        """Process one PIT value.

        Parameters
        ----------
        pit
            Probability integral transform in [0, 1].
            Compute as F_hat(y_t | x_t) using the model's predictive CDF.
        exposure
            Observation weight (default 1.0). Implemented as integer
            repetition: the PIT is processed round(max(exposure, 1)) times.
            Fractional values are rounded. Preserves the anytime-valid
            guarantee by construction.

        Returns
        -------
        PITAlarm
            Evaluates as True if alarm has been triggered.

        Notes
        -----
        Once an alarm fires, subsequent calls return the same alarm with
        frozen evidence. PITs and p-values continue to be recorded for
        changepoint estimation completeness.
        """
        if not (0.0 <= pit <= 1.0):
            raise ValueError(f"pit must be in [0, 1], got {pit}")

        n_reps = max(1, round(float(exposure)))
        result: PITAlarm | None = None

        for _ in range(n_reps):
            result = self._step(pit)
            if self.alarm_triggered and _ == 0:
                # For repeated exposures, stop at first alarm step
                break

        assert result is not None
        return result

    def update_with_cdf(self, cdf: Callable[[float], float], y: float) -> PITAlarm:
        """Convenience wrapper: compute PIT from a CDF callable and process it.

        Parameters
        ----------
        cdf
            Cumulative distribution function F(y). Must accept a single float
            and return a float in [0, 1].
        y
            Observed outcome.

        Examples
        --------
        >>> from scipy.stats import poisson
        >>> monitor = PITMonitor(rng=42)
        >>> mu = 0.15
        >>> y = 0.0
        >>> alarm = monitor.update_with_cdf(lambda val: poisson.cdf(val, mu), y)
        """
        return self.update(float(cdf(y)))

    def update_many(
        self,
        pits: npt.ArrayLike,
        stop_on_alarm: bool = True,
    ) -> PITAlarm:
        """Process a sequence of PIT values.

        Parameters
        ----------
        pits
            Array-like of floats in [0, 1].
        stop_on_alarm
            If True (default), halt processing at first alarm.
            Set False to process all observations (useful for
            retrospective analysis and changepoint estimation).

        Returns
        -------
        PITAlarm
            Alarm state after the final processed step.
        """
        arr = np.asarray(pits, dtype=np.float64).ravel()
        if arr.size == 0:
            return PITAlarm(
                triggered=self.alarm_triggered,
                time=self.t,
                evidence=self._M,
                threshold=self._threshold,
                changepoint=self.changepoint() if self.alarm_triggered else None,
            )

        result: PITAlarm | None = None
        for pit in arr:
            result = self._step(float(pit))
            if stop_on_alarm and self.alarm_triggered:
                break

        assert result is not None
        return result

    # ------------------------------------------------------------------
    # Changepoint estimation
    # ------------------------------------------------------------------

    def changepoint(self) -> int | None:
        """Estimate when calibration shifted (post-alarm only).

        Scans candidate split points k = 1, ..., T-1 and returns the one
        maximising the log Bayes factor comparing the post-split p-values
        under H1 (Dirichlet-Multinomial) versus H0 (uniform).

        Returns
        -------
        int or None
            1-indexed step in the monitoring stream, or None if no alarm.

        Complexity
        ----------
        O(T) time, O(B) space via incremental histogram update.
        """
        if not self.alarm_triggered:
            return None

        T = len(self._history)
        if T < 2:
            return 1

        pvals = [h[1] for h in self._history]
        B = self.n_bins
        kappa = 0.5  # Jeffreys prior

        # Initialise with all observations in the segment [1 .. T-1]
        # (0-indexed: indices 0 to T-2, i.e. the post-split segment when k=1)
        # The post-split segment for split k is [k, T-1] (0-indexed),
        # containing T - k observations.
        # We start with k=1 (0-indexed: segment is [1, T-1], n = T-1).

        counts = np.zeros(B, dtype=np.float64)
        for idx in range(1, T):
            b = min(int(math.floor(pvals[idx] * B)), B - 1)
            counts[b] += 1.0

        lgamma_kappa = math.lgamma(kappa)
        lgamma_B_kappa = math.lgamma(B * kappa)

        def _lgamma_sum(c: np.ndarray) -> float:
            """Sum of lgamma(c[b] + kappa) over bins using stdlib math.lgamma."""
            return sum(math.lgamma(float(cb) + kappa) for cb in c)

        def _log_bf(lgsum: float, n: int) -> float:
            """Log Bayes factor for a segment."""
            if n == 0:
                return -math.inf
            log_p_h1 = (
                lgamma_B_kappa
                - math.lgamma(n + B * kappa)
                + lgsum
                - B * lgamma_kappa
            )
            log_p_h0 = -n * math.log(B)
            return log_p_h1 - log_p_h0

        # Compute initial lgamma_sum for k=1 segment
        lgsum = _lgamma_sum(counts)

        best_k = 1
        best_lbf = _log_bf(lgsum, T - 1)

        # Walk k from 1 to T-2, removing observation k from the segment
        for k in range(1, T - 1):
            # Remove pvals[k] (0-indexed k) from segment by decrementing
            b_remove = min(int(math.floor(pvals[k] * B)), B - 1)
            # Incremental lgamma_sum update
            old_c = counts[b_remove]
            lgsum -= math.lgamma(float(old_c) + kappa)
            counts[b_remove] -= 1.0
            lgsum += math.lgamma(float(old_c - 1.0) + kappa)

            n_seg = T - k - 1
            lbf = _log_bf(lgsum, n_seg)
            if lbf > best_lbf:
                best_lbf = lbf
                best_k = k + 1  # 1-indexed

        return best_k

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def warm_start(self, historical_pits: npt.ArrayLike) -> None:
        """Pre-load historical PITs without accumulating evidence.

        Resets monitor state, processes ``historical_pits`` to build the
        histogram and sorted PIT list, then resets ``M_t = 0`` and ``t = 0``.
        Subsequent :meth:`update` calls benefit from a pre-warmed density
        estimator but cannot retroactively alarm on history.

        This is epistemically honest: we condition on the pre-monitoring period
        being stable and start the e-process fresh from monitoring start.

        Parameters
        ----------
        historical_pits
            Array-like of PITs from the stable pre-monitoring period.
        """
        arr = np.asarray(historical_pits, dtype=np.float64).ravel()
        # Reset to clean state first, then process history
        self._init_state()

        for pit in arr:
            if not (0.0 <= pit <= 1.0):
                raise ValueError(f"historical_pits must all be in [0, 1], got {pit}")
            self._sorted_pits.add(float(pit))
            b = min(int(math.floor(pit * self.n_bins)), self.n_bins - 1)
            self._bin_counts[b] += 1.0

        # Reset evidence — history does not contribute to alarm
        self._M = 0.0
        self.t = 0
        self._history = []
        self.alarm_triggered = False
        self.alarm_time = None

    # ------------------------------------------------------------------
    # Summary and diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> PITSummary:
        """Return a full snapshot of the current monitoring state."""
        cs = self.calibration_score() if self.t > 0 else None
        cp = self.changepoint() if self.alarm_triggered else None
        return PITSummary(
            t=self.t,
            alarm_triggered=self.alarm_triggered,
            alarm_time=self.alarm_time,
            evidence=self._M,
            threshold=self._threshold,
            changepoint=cp,
            calibration_score=cs,
            n_observations=self.t,
        )

    def calibration_score(self) -> float:
        """1 - KS statistic for PIT uniformity.

        Returns a value in [0, 1] where 1 = perfectly uniform PITs.
        Useful as a continuous degradation metric for dashboards, separate
        from the binary alarm decision.

        Returns
        -------
        float
            0.0 if t == 0. Otherwise 1 - max|ECDF(u) - u| over sorted PITs.
        """
        if self.t == 0:
            return 0.0
        sorted_arr = np.array(self._sorted_pits)
        n = len(sorted_arr)
        ecdf = np.arange(1, n + 1) / n
        ks_stat = float(np.max(np.abs(ecdf - sorted_arr)))
        return 1.0 - ks_stat

    def reset(self) -> None:
        """Reset monitor to initial state, preserving hyperparameters."""
        if isinstance(self._rng_seed, int):
            self._rng = np.random.default_rng(self._rng_seed)
        elif self._rng_seed is None and not isinstance(
            self._rng, np.random.Generator
        ):
            self._rng = np.random.default_rng()
        self._init_state()

    @property
    def evidence(self) -> float:
        """Current e-process value M_t."""
        return self._M

    @property
    def pits(self) -> np.ndarray:
        """All observed PITs in order, shape (t,)."""
        return np.array([h[0] for h in self._history])

    @property
    def pvalues(self) -> np.ndarray:
        """All conformal p-values in order, shape (t,)."""
        return np.array([h[1] for h in self._history])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str | Path) -> None:
        """Save monitor state to JSON or pickle.

        JSON is human-readable and does not execute code on load.
        Pickle preserves numpy array types exactly.
        Custom weight schedules cannot be serialised — a UserWarning
        is emitted if one is set.
        Extension determines format: ``.json`` or ``.pkl``.

        Parameters
        ----------
        filepath
            Target path. Extension must be ``.json`` or ``.pkl``.
        """
        filepath = Path(filepath)
        if self._weight_schedule is not None:
            warnings.warn(
                "Custom weight_schedule cannot be serialised. "
                "The default schedule will be used on load.",
                UserWarning,
                stacklevel=2,
            )

        state = {
            "alpha": self.alpha,
            "n_bins": self.n_bins,
            "rng_seed": self._rng_seed,
            "t": self.t,
            "M": self._M,
            "bin_counts": self._bin_counts.tolist(),
            "sorted_pits": list(self._sorted_pits),
            "history": self._history,
            "alarm_triggered": self.alarm_triggered,
            "alarm_time": self.alarm_time,
        }

        if filepath.suffix == ".json":
            with filepath.open("w", encoding="utf-8") as fh:
                json.dump(state, fh)
        elif filepath.suffix in (".pkl", ".pickle"):
            with filepath.open("wb") as fb:
                pickle.dump(state, fb)
        else:
            raise ValueError(
                f"Unsupported extension {filepath.suffix!r}. Use '.json' or '.pkl'."
            )

    @classmethod
    def load(cls, filepath: str | Path) -> "PITMonitor":
        """Restore monitor from saved state.

        Supports both ``.json`` and ``.pkl`` formats written by :meth:`save`.

        Parameters
        ----------
        filepath
            Path to a file previously written by :meth:`save`.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            with filepath.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
        elif filepath.suffix in (".pkl", ".pickle"):
            with filepath.open("rb") as fb:
                state = pickle.load(fb)  # noqa: S301
        else:
            raise ValueError(
                f"Unsupported extension {filepath.suffix!r}. Use '.json' or '.pkl'."
            )

        monitor = cls.__new__(cls)
        monitor.alpha = state["alpha"]
        monitor.n_bins = state["n_bins"]
        monitor._threshold = 1.0 / state["alpha"]
        monitor._weight_schedule = None
        monitor._rng_seed = state.get("rng_seed")
        if monitor._rng_seed is not None:
            monitor._rng = np.random.default_rng(int(monitor._rng_seed))
        else:
            monitor._rng = np.random.default_rng()

        monitor.t = state["t"]
        monitor._M = state["M"]
        monitor._bin_counts = np.array(state["bin_counts"], dtype=np.float64)
        monitor._sorted_pits = SortedList(state["sorted_pits"])
        monitor._history = [tuple(h) for h in state["history"]]
        monitor.alarm_triggered = state["alarm_triggered"]
        monitor.alarm_time = state["alarm_time"]

        return monitor

    # ------------------------------------------------------------------
    # Diagnostic plot
    # ------------------------------------------------------------------

    def plot(self, figsize: tuple[float, float] = (12, 4)):  # type: ignore[return]
        """Diagnostic plot: e-process trace and p-value histogram.

        Left panel: M_t on log scale with threshold line. Alarm marker and
        changepoint estimate are shown when triggered.
        Right panel: histogram of conformal p-values with uniform reference
        line (density = 1).

        Returns
        -------
        matplotlib.figure.Figure or None
            None if fewer than 2 observations have been processed.
        """
        if self.t < 2:
            return None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- Left: e-process trace ----
        evidence_trace = [h[2] for h in self._history]
        ts = np.arange(1, len(evidence_trace) + 1)
        ax = axes[0]
        ax.plot(ts, evidence_trace, color="steelblue", linewidth=1.0, label="M_t")
        ax.axhline(
            self._threshold,
            color="crimson",
            linestyle="--",
            linewidth=1.0,
            label=f"Threshold 1/\u03b1 = {self._threshold:.0f}",
        )
        if self.alarm_triggered and self.alarm_time is not None:
            ax.axvline(
                self.alarm_time,
                color="crimson",
                linestyle=":",
                linewidth=1.0,
                label=f"Alarm t={self.alarm_time}",
            )
        cp = self.changepoint()
        if cp is not None:
            ax.axvline(
                cp,
                color="darkorange",
                linestyle="--",
                linewidth=1.0,
                label=f"Changepoint t={cp}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Step t")
        ax.set_ylabel("E-process M_t (log scale)")
        ax.set_title("Mixture e-process")
        ax.legend(fontsize=8)

        # ---- Right: p-value histogram ----
        pvals = self.pvalues
        ax2 = axes[1]
        ax2.hist(
            pvals,
            bins=min(self.n_bins, 50),
            density=True,
            color="steelblue",
            alpha=0.7,
            label="Conformal p-values",
        )
        ax2.axhline(1.0, color="crimson", linestyle="--", linewidth=1.0, label="Uniform")
        ax2.set_xlabel("p-value")
        ax2.set_ylabel("Density")
        ax2.set_title("P-value distribution")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        status = "ALARM" if self.alarm_triggered else "running"
        return (
            f"PITMonitor(alpha={self.alpha}, n_bins={self.n_bins}, "
            f"t={self.t}, M={self._M:.3g}, status={status})"
        )


# ---------------------------------------------------------------------------
# Module-private helper
# ---------------------------------------------------------------------------


def _validate_weight_schedule(fn: Callable[[int], float]) -> None:
    """Validate a custom weight schedule over 10,000 terms."""
    tol = 1e-3
    partial_sum = 0.0
    for t in range(1, 10_001):
        w = fn(t)
        if w < 0:
            raise ValueError(f"weight_schedule returned negative value {w} at t={t}")
        partial_sum += w
        if partial_sum > 1.0 + tol:
            raise ValueError(
                f"weight_schedule partial sum {partial_sum:.6f} exceeds 1 + {tol} at t={t}"
            )
    if abs(partial_sum - 1.0) > tol:
        raise ValueError(
            f"weight_schedule sum over first 10,000 terms is {partial_sum:.6f}; "
            f"expected close to 1.0 (tol={tol})"
        )
