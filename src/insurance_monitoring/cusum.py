"""
Calibration CUSUM chart with dynamic probability control limits (DPCLs).

Implements the calibration monitoring CUSUM from Franck, Driscoll, Szajnfarber
and Woodall (2025), arXiv:2510.25573. The chart monitors whether model probability
predictions remain calibrated as new observations arrive, using a likelihood-ratio
CUSUM statistic and Monte Carlo control limits that maintain a constant conditional
false alarm rate (CFAR).

Design notes
------------
``CalibrationCUSUM`` is designed for continuous production monitoring, running
monthly or weekly as new data arrives. It complements ``PITMonitor`` (which gives
anytime-valid formal error control) by providing fast detection with quantified
average run length (ARL) properties.

In-control ARL0 ~ 1/cfar by construction (geometric distribution with probability
cfar). At cfar=0.005, ARL0=200. Out-of-control ARL1 depends on the alternative;
for delta=2 (rate doubled), ARL1 ~ 37 at cfar=0.005.

The CUSUM statistic resets to zero after each alarm, allowing detection of
multiple miscalibration events over the monitoring lifecycle.

**Bernoulli mode** (distribution='bernoulli'): the original paper method. Uses
the LLO (linear log-odds) alternative hypothesis parametrised by (delta_a, gamma_a).
Appropriate for binary claim indicators (0/1 per policy).

**Poisson mode** (distribution='poisson'): insurance adaptation. Uses the Poisson
log-likelihood ratio with a rate multiplier delta_a. gamma_a is not used in this
mode. Appropriate for claim count data with varying exposure.

MC pool resampling
------------------
At each step, the MC CUSUM paths that were below the control limit are resampled
(with replacement) to maintain n_mc non-alarmed paths for the next step. If fewer
than n_mc//4 paths are below the limit (rare for in-control data), the pool resets
to zero. This threshold (n_mc//4) is an engineering choice that balances
path contamination against numerical stability; it is not specified by the paper.

References
----------
Franck, Driscoll, Szajnfarber, Woodall (2025), arXiv:2510.25573.

Examples
--------
Bernoulli (binary claim indicator)::

    from insurance_monitoring.cusum import CalibrationCUSUM
    import numpy as np

    rng = np.random.default_rng(42)
    monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=5000, random_state=0)

    for t in range(36):
        n_t = rng.integers(50, 200)
        p = rng.uniform(0.05, 0.25, n_t)
        y = rng.binomial(1, p)
        alarm = monitor.update(p, y)
        if alarm:
            print(f"Alarm at t={alarm.time}, S_t={alarm.statistic:.2f}")

Poisson frequency (UK motor book)::

    monitor = CalibrationCUSUM(
        delta_a=1.3, distribution='poisson', cfar=0.005, random_state=0
    )
    alarm = monitor.update(mu_hat, y_claims, exposure=car_years)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from insurance_monitoring.discrimination import (
    _to_numpy,
    _to_numpy_optional,
    ArrayLike,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CUSUMAlarm:
    """Result returned by :meth:`CalibrationCUSUM.update`.

    Evaluates as bool: ``if alarm:`` is True when triggered.

    Attributes
    ----------
    triggered : bool
        True when S_t > h_t (control limit exceeded).
    time : int
        Current time step t (1-indexed).
    statistic : float
        CUSUM value S_t at this time step (after potential reset to 0).
    control_limit : float
        Dynamic control limit h_t at this time step.
    log_likelihood_ratio : float
        W_t: the increment to the CUSUM statistic before max(0, ...).
    n_obs : int
        Number of observations n_t at this time step.
    """

    triggered: bool
    time: int
    statistic: float
    control_limit: float
    log_likelihood_ratio: float
    n_obs: int

    def __bool__(self) -> bool:
        return self.triggered


@dataclass
class CUSUMSummary:
    """Summary of :class:`CalibrationCUSUM` history across all time steps.

    Attributes
    ----------
    n_time_steps : int
        Total number of :meth:`update` calls since initialisation (or last
        :meth:`reset`).
    n_alarms : int
        Total number of alarm events across all time steps.
    alarm_times : list[int]
        Time steps at which alarms fired (1-indexed).
    current_statistic : float
        Current CUSUM value S_t.
    current_control_limit : float or None
        Most recent dynamic control limit h_t. None before any update.
    """

    n_time_steps: int
    n_alarms: int
    alarm_times: list
    current_statistic: float
    current_control_limit: Optional[float]

    def to_dict(self) -> dict:
        """Return all fields as a plain dict."""
        return {
            "n_time_steps": self.n_time_steps,
            "n_alarms": self.n_alarms,
            "alarm_times": list(self.alarm_times),
            "current_statistic": self.current_statistic,
            "current_control_limit": self.current_control_limit,
        }


# ---------------------------------------------------------------------------
# CalibrationCUSUM
# ---------------------------------------------------------------------------


class CalibrationCUSUM:
    """Calibration monitoring CUSUM chart with dynamic probability control limits.

    Monitors whether model probability predictions remain calibrated over time.
    Uses a likelihood-ratio CUSUM statistic comparing the calibrated null
    hypothesis against a specified LLO (linear log-odds) alternative.

    Dynamic control limits (DPCLs) maintain a constant conditional false alarm
    rate (CFAR) at each time step, regardless of the varying number of
    observations per step. In-control ARL0 ~ 1/cfar by construction.

    Implements Franck, Driscoll, Szajnfarber, Woodall (2025), arXiv:2510.25573.
    For the Poisson insurance adaptation, see the class Notes.

    Parameters
    ----------
    delta_a : float, default 2.0
        Alternative shift parameter. delta_a > 1 detects upward shift (event
        rate higher than predicted); delta_a < 1 detects downward shift.
        Must not equal 1.0 when gamma_a=1.0 (that would be the identity LLO,
        making the alternative identical to the null).
    gamma_a : float, default 1.0
        Alternative scale parameter. gamma_a > 1 detects scale-up (predictions
        too compressed); gamma_a < 1 detects scale-down (predictions too
        extreme). Only used when distribution='bernoulli'.
    cfar : float, default 0.005
        Conditional false alarm rate per time step. ARL0 ~ 1/cfar.
        Paper recommendation: 0.005, giving ARL0=200. Use 0.0005 for ARL0~2000.
    n_mc : int, default 5000
        Monte Carlo draws for DPCL computation. Larger n_mc gives more stable
        control limits at higher compute cost. 5000 is the paper default.
    distribution : str, default 'bernoulli'
        Observation model:
        - 'bernoulli': binary 0/1 outcomes, predictions in (0, 1). The paper
          method using the LLO recalibration function.
        - 'poisson': count outcomes, predictions are claim rates > 0,
          exposure required. Insurance frequency adaptation.
    random_state : int or None, default None
        Seed for Monte Carlo reproducibility.

    Notes
    -----
    **Bernoulli** (distribution='bernoulli'): the log-likelihood ratio at step t is::

        W_t = sum_i [y_i * log(g(p_i; delta_a, gamma_a) / p_i)
                     + (1 - y_i) * log((1 - g(p_i; delta_a, gamma_a)) / (1 - p_i))]

    where g is the LLO function: g(x; d, g) = d*x^g / (d*x^g + (1-x)^g).

    **Poisson** (distribution='poisson'): the Poisson log-likelihood ratio is::

        W_t = sum_i [y_i * log(delta_a) - (delta_a - 1) * mu_i * v_i]

    where mu_i is the model's rate prediction and v_i is the exposure.
    gamma_a is not used in Poisson mode.

    Examples
    --------
    ::

        import numpy as np
        from insurance_monitoring.cusum import CalibrationCUSUM

        rng = np.random.default_rng(42)
        monitor = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=5000, random_state=0)

        for t in range(36):
            n_t = rng.integers(50, 200)
            p = rng.uniform(0.05, 0.25, n_t)
            y = rng.binomial(1, p)
            alarm = monitor.update(p, y)
            if alarm:
                print(f"Alarm at t={alarm.time}, S_t={alarm.statistic:.2f}")
    """

    def __init__(
        self,
        delta_a: float = 2.0,
        gamma_a: float = 1.0,
        cfar: float = 0.005,
        n_mc: int = 5000,
        distribution: str = "bernoulli",
        random_state: Optional[int] = None,
    ) -> None:
        _valid_distributions = {"bernoulli", "poisson"}
        if distribution not in _valid_distributions:
            raise ValueError(
                f"distribution must be one of {sorted(_valid_distributions)}, "
                f"got '{distribution}'"
            )
        if not (0.0 < cfar < 1.0):
            raise ValueError(f"cfar must be in (0, 1), got {cfar}")
        if n_mc < 100:
            raise ValueError(f"n_mc must be >= 100, got {n_mc}")
        # Identity alternative: g(x; 1, 1) = x — no signal possible
        if distribution == "bernoulli" and abs(delta_a - 1.0) < 1e-12 and abs(gamma_a - 1.0) < 1e-12:
            raise ValueError(
                "delta_a=1.0 and gamma_a=1.0 gives the identity LLO function. "
                "The alternative is identical to the null — no signal is possible. "
                "Set delta_a != 1.0 or gamma_a != 1.0."
            )
        if distribution == "poisson" and abs(delta_a - 1.0) < 1e-12:
            raise ValueError(
                "delta_a=1.0 in Poisson mode gives W_t=0 always. "
                "Set delta_a != 1.0 to define a meaningful alternative."
            )

        self.delta_a = float(delta_a)
        self.gamma_a = float(gamma_a)
        self.cfar = float(cfar)
        self.n_mc = int(n_mc)
        self.distribution = distribution
        self.random_state = random_state

        # Internal state
        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._S: float = 0.0
        self._t: int = 0
        self._mc_S: np.ndarray = np.zeros(self.n_mc, dtype=np.float64)

        # History
        self._history_t: list[int] = []
        self._history_S: list[float] = []
        self._history_h: list[float] = []
        self._history_alarm: list[bool] = []
        self._history_W: list[float] = []
        self._n_alarms: int = 0
        self._alarm_times: list[int] = []
        self._last_h: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        predictions: ArrayLike,
        outcomes: ArrayLike,
        exposure: Optional[ArrayLike] = None,
    ) -> CUSUMAlarm:
        """Process one time step of observations.

        Computes the log-likelihood ratio W_t, updates the CUSUM statistic
        S_t = max(0, S_{t-1} + W_t), computes the dynamic control limit h_t
        via Monte Carlo simulation, and checks for an alarm.

        After an alarm, S_t resets to 0 and monitoring continues from the
        next call. Alarm history is retained in summary().

        Parameters
        ----------
        predictions : array-like
            Model predictions at this time step.
            - 'bernoulli': probabilities in (0, 1). Shape (n_t,).
            - 'poisson': claim rates > 0. Shape (n_t,).
        outcomes : array-like
            Observed outcomes. Shape (n_t,).
            - 'bernoulli': binary {0, 1}.
            - 'poisson': non-negative integer claim counts.
        exposure : array-like, optional
            Policy durations (car-years). Required for 'poisson'. Shape (n_t,).
            If None, assumed 1.0 for all observations.

        Returns
        -------
        CUSUMAlarm
            Result with triggered, statistic, control_limit, and diagnostic info.

        Raises
        ------
        ValueError
            Predictions outside valid range, length mismatch, negative outcomes.
        """
        pred = _to_numpy(predictions)
        y = _to_numpy(outcomes)
        exp = _to_numpy_optional(exposure)

        n_t = len(pred)
        if len(y) != n_t:
            raise ValueError(
                f"predictions length ({n_t}) != outcomes length ({len(y)})"
            )
        if exp is not None and len(exp) != n_t:
            raise ValueError(
                f"exposure length ({len(exp)}) != predictions length ({n_t})"
            )
        if np.any(y < 0):
            raise ValueError("outcomes must be non-negative")

        self._t += 1

        if self.distribution == "bernoulli":
            W_t, h_t, alarmed = self._update_bernoulli(pred, y, n_t)
        else:
            v = exp if exp is not None else np.ones(n_t, dtype=np.float64)
            W_t, h_t, alarmed = self._update_poisson(pred, y, v, n_t)

        # Append history before resetting
        self._history_t.append(self._t)
        self._history_W.append(W_t)
        self._history_h.append(h_t)
        self._history_alarm.append(alarmed)

        stat_after_alarm = self._S  # already reset if alarmed
        self._history_S.append(stat_after_alarm)
        self._last_h = h_t

        return CUSUMAlarm(
            triggered=alarmed,
            time=self._t,
            statistic=stat_after_alarm,
            control_limit=h_t,
            log_likelihood_ratio=W_t,
            n_obs=n_t,
        )

    def reset(self) -> None:
        """Reset the CUSUM statistic and time index to zero.

        Call this after a planned model intervention (recalibration, refit)
        to restart monitoring. The alarm history (summary()) is retained —
        all past alarms remain visible.

        The Monte Carlo pool is also reset to zeros to avoid carryover from
        the pre-intervention period.
        """
        self._S = 0.0
        self._t = 0
        self._mc_S = np.zeros(self.n_mc, dtype=np.float64)
        self._history_t.clear()
        self._history_S.clear()
        self._history_h.clear()
        self._history_alarm.clear()
        self._history_W.clear()
        # Note: alarm count and alarm_times are NOT reset (per spec)

    def summary(self) -> CUSUMSummary:
        """Return a summary of the monitoring history.

        Returns
        -------
        CUSUMSummary
            Counts, alarm times, and current state.
        """
        return CUSUMSummary(
            n_time_steps=self._t,
            n_alarms=self._n_alarms,
            alarm_times=list(self._alarm_times),
            current_statistic=self._S,
            current_control_limit=self._last_h,
        )

    def plot(self, ax=None, title: Optional[str] = None):
        """Plot the CUSUM statistic and dynamic control limits over time.

        Requires matplotlib. Returns a matplotlib Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        title : str, optional
            Plot title. Defaults to 'Calibration CUSUM Chart'.

        Returns
        -------
        matplotlib.axes.Axes
            - Solid black line: CUSUM statistic S_t.
            - Dotted blue line: dynamic control limit h_t.
            - Red vertical lines: alarm time steps.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        t_steps = self._history_t
        if not t_steps:
            ax.set_title(title or "Calibration CUSUM Chart")
            ax.set_xlabel("Time step")
            ax.set_ylabel("CUSUM statistic")
            return ax

        ax.plot(t_steps, self._history_S, color="black", lw=1.5, label="$S_t$")
        ax.plot(
            t_steps, self._history_h, color="blue", lw=1.0, ls=":",
            label="Control limit $h_t$"
        )

        for alarm_t in self._alarm_times:
            ax.axvline(x=alarm_t, color="red", lw=0.8, alpha=0.7, ls="--")

        ax.set_title(title or "Calibration CUSUM Chart")
        ax.set_xlabel("Time step")
        ax.set_ylabel("CUSUM statistic")
        ax.legend(loc="upper left")
        return ax

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic S_t (0.0 before any update)."""
        return self._S

    @property
    def time(self) -> int:
        """Current time step (number of update() calls since last reset)."""
        return self._t

    def __repr__(self) -> str:
        return (
            f"CalibrationCUSUM(delta_a={self.delta_a}, gamma_a={self.gamma_a}, "
            f"cfar={self.cfar}, n_mc={self.n_mc}, "
            f"distribution='{self.distribution}', "
            f"t={self._t}, S_t={self._S:.4f})"
        )

    # ------------------------------------------------------------------
    # Private update implementations
    # ------------------------------------------------------------------

    def _update_bernoulli(
        self,
        pred: np.ndarray,
        y: np.ndarray,
        n_t: int,
    ) -> tuple[float, float, bool]:
        """Bernoulli CUSUM update step.

        Returns (W_t, h_t, alarmed).
        """
        # Clip predictions for numerical stability
        p = np.clip(pred, 1e-10, 1.0 - 1e-10)

        # Warn if extreme predictions are common
        extreme = np.mean((pred < 0.001) | (pred > 0.999))
        if extreme > 0.01:
            warnings.warn(
                f"{100.0 * extreme:.1f}% of predictions are outside [0.001, 0.999]. "
                "Extreme probabilities may cause numerical instability in LLR computation.",
                UserWarning,
                stacklevel=3,
            )

        # LLO alternative probabilities
        g_alt = _llo(p, self.delta_a, self.gamma_a)

        # Observed W_t
        log_ratio_y1 = np.log(g_alt) - np.log(p)
        log_ratio_y0 = np.log(1.0 - g_alt) - np.log(1.0 - p)
        W_t = float(np.sum(y * log_ratio_y1 + (1.0 - y) * log_ratio_y0))

        # Update observed CUSUM
        self._S = max(0.0, self._S + W_t)

        # Monte Carlo: simulate n_mc calibrated outcome sets
        # p_2d: (n_mc, n_t) broadcast
        p_row = p[np.newaxis, :]  # shape (1, n_t)
        sim_y = self._rng.binomial(1, np.broadcast_to(p_row, (self.n_mc, n_t)))

        g_row = g_alt[np.newaxis, :]  # shape (1, n_t)
        log_r1 = np.log(g_row) - np.log(p_row)
        log_r0 = np.log(1.0 - g_row) - np.log(1.0 - p_row)
        W_mc = np.sum(sim_y * log_r1 + (1.0 - sim_y) * log_r0, axis=1)  # (n_mc,)

        self._mc_S = np.maximum(0.0, self._mc_S + W_mc)

        # Dynamic control limit
        h_t = float(np.quantile(self._mc_S, 1.0 - self.cfar))

        # Alarm check
        alarmed = bool(self._S > h_t)
        if alarmed:
            self._S = 0.0
            self._n_alarms += 1
            self._alarm_times.append(self._t)

        # Resample MC pool from non-alarmed paths
        self._mc_S = _resample_mc_pool(self._mc_S, h_t, self.n_mc, self._rng)

        return W_t, h_t, alarmed

    def _update_poisson(
        self,
        pred: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray,
        n_t: int,
    ) -> tuple[float, float, bool]:
        """Poisson CUSUM update step.

        Returns (W_t, h_t, alarmed).
        """
        mu = np.maximum(pred, 1e-15)
        lam_0 = mu * exposure  # calibrated Poisson means

        log_delta = np.log(self.delta_a)
        W_t = float(np.sum(y * log_delta - (self.delta_a - 1.0) * lam_0))

        self._S = max(0.0, self._S + W_t)

        # MC: simulate from calibrated Poisson(lam_0)
        lam_row = lam_0[np.newaxis, :]  # (1, n_t)
        sim_y = self._rng.poisson(np.broadcast_to(lam_row, (self.n_mc, n_t)))
        W_mc = np.sum(sim_y * log_delta - (self.delta_a - 1.0) * lam_row, axis=1)

        self._mc_S = np.maximum(0.0, self._mc_S + W_mc)

        h_t = float(np.quantile(self._mc_S, 1.0 - self.cfar))

        alarmed = bool(self._S > h_t)
        if alarmed:
            self._S = 0.0
            self._n_alarms += 1
            self._alarm_times.append(self._t)

        self._mc_S = _resample_mc_pool(self._mc_S, h_t, self.n_mc, self._rng)

        return W_t, h_t, alarmed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _llo(x: np.ndarray, delta: float, gamma: float) -> np.ndarray:
    """Linear log-odds (LLO) recalibration function (paper eq. 1).

    g(x; delta, gamma) = delta * x^gamma / (delta * x^gamma + (1 - x)^gamma)

    At (delta=1, gamma=1): g(x) = x (identity, perfectly calibrated null).
    At delta > 1: upward shift in probability.
    At gamma < 1: predictions compressed toward 0.5 relative to calibrated.

    Parameters
    ----------
    x : np.ndarray
        Probability predictions in (0, 1).
    delta : float
        Log-odds shift parameter.
    gamma : float
        Log-odds scale parameter.

    Returns
    -------
    np.ndarray
        Recalibrated probabilities in (0, 1).
    """
    # Use log-space for numerical stability at extreme probabilities
    # log(numerator) = log(delta) + gamma * log(x)
    # log(denominator) = log(delta * x^gamma + (1-x)^gamma)
    #                  = log_numerator + log(1 + exp(gamma*log(1-x) - log_numerator))
    log_num = np.log(delta) + gamma * np.log(np.maximum(x, 1e-30))
    log_den_complement = gamma * np.log(np.maximum(1.0 - x, 1e-30))
    # Numerically stable: log(a + b) = log(a) + log(1 + b/a) = log_num + log(1 + exp(complement - log_num))
    log_diff = log_den_complement - log_num
    # Use sigmoid-like computation to avoid overflow
    # g = 1 / (1 + exp(log_diff))
    return 1.0 / (1.0 + np.exp(log_diff))


def _resample_mc_pool(
    mc_s: np.ndarray,
    h_t: float,
    n_mc: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample the MC CUSUM pool from paths that are below the control limit.

    This implements the DPCL conditioning: paths that signalled (above h_t)
    are discarded and replaced by resampled non-signalling paths. This mirrors
    the real statistic which resets after an alarm.

    If fewer than n_mc//4 paths are below the limit, the pool restarts from
    zero. This threshold avoids contamination of the pool with near-limit paths
    when the chart is running hot.

    Parameters
    ----------
    mc_s : np.ndarray
        MC CUSUM statistics, shape (n_mc,).
    h_t : float
        Current control limit.
    n_mc : int
        Required pool size.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Resampled pool of size n_mc.
    """
    below = mc_s[mc_s <= h_t]
    if len(below) >= n_mc // 4:
        idx = rng.choice(len(below), size=n_mc, replace=True)
        return below[idx]
    else:
        # Very few non-signalling paths — restart the pool
        return np.zeros(n_mc, dtype=np.float64)
