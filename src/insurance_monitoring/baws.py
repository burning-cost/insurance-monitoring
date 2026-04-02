"""
Bootstrap Adaptive Window Selection (BAWS) for VaR/ES monitoring.

Implements the BAWS algorithm from Li, Lyu, Wang (2026) — arXiv:2603.01157.
Provides data-driven sequential window selection for Value-at-Risk and Expected
Shortfall monitoring: at each time step the algorithm picks the window length
that minimises a block-bootstrapped estimate of the scoring rule.

This is directly relevant to Solvency II / IFRS 17 internal model backtesting
and PRA SS3/19 validation requirements, where teams need to decide how much
historical data to include in a rolling risk estimate. Too short a window
overfits recent fluctuations; too long a window fails to detect genuine regime
changes. BAWS automates that trade-off with a formal bootstrap test.

Design notes
------------
The Fissler-Ziegel score is the strictly consistent scoring rule for the joint
functional (VaR_alpha, ES_alpha). Using it means window selection is aligned
with what you actually care about — not some proxy loss. The asymmetric absolute
loss (tick loss) is the strictly consistent scoring rule for VaR alone and is
included as an alternative for simpler monitoring setups.

The block bootstrap (block_length controls the block size, default: T^(1/3))
accounts for serial dependence in the return series — important for financial
data that exhibits autocorrelation and volatility clustering.

References
----------
- Li, Lyu, Wang (2026). BAWS — Bootstrap Adaptive Window Selection.
  arXiv:2603.01157.
- Fissler & Ziegel (2016). Higher order elicitability and Osband's principle.
  Annals of Statistics 44(4).
- Patton, Ziegel, Chen (2019). Dynamic semiparametric models for expected
  shortfall (and VaR). Journal of Econometrics 211(2).

Examples
--------
::

    import numpy as np
    from insurance_monitoring.baws import BAWSMonitor

    rng = np.random.default_rng(42)
    returns = rng.standard_t(df=5, size=500)

    monitor = BAWSMonitor(alpha=0.05, candidate_windows=[50, 100, 150, 200])
    monitor.fit(returns)

    # Update with one new return:
    r = rng.standard_t(df=5)
    result = monitor.update(r)
    print(result.selected_window, result.var_estimate, result.es_estimate)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BAWSResult:
    """Result returned by :meth:`BAWSMonitor.update`.

    Attributes
    ----------
    selected_window : int
        Window length chosen by BAWS at this time step.
    var_estimate : float
        VaR_alpha estimate using the selected window.
    es_estimate : float
        ES_alpha estimate using the selected window. Always <= var_estimate
        for losses (i.e. ES is more negative than VaR for loss distributions).
    scores : dict[int, float]
        Bootstrap mean score for each candidate window. Lower is better.
        Key: window length. Value: mean bootstrapped score.
    n_obs : int
        Number of historical observations available at this time step.
    time_step : int
        1-indexed time step counter (number of update() calls since fit()).
    """

    selected_window: int
    var_estimate: float
    es_estimate: float
    scores: dict[int, float]
    n_obs: int
    time_step: int


# ---------------------------------------------------------------------------
# Scoring functions (module-level helpers)
# ---------------------------------------------------------------------------


def fissler_ziegel_score(
    var: float,
    es: float,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fissler-Ziegel strictly consistent joint score for (VaR_alpha, ES_alpha).

    S(x, v, y; alpha) = (1/alpha) * 1{y > x} * (y - x) / (-v)
                      + log(-v)
                      + (1 - 1/alpha) * x / v

    This is the negentropy form. Note that v = ES < 0 for positive alpha and
    standard loss conventions. We enforce v < 0 for numerical stability.

    The score is minimised at the true (VaR_alpha, ES_alpha) in expectation.

    Parameters
    ----------
    var : float
        VaR estimate (the alpha-quantile of the loss distribution).
    es : float
        ES estimate. Must be strictly negative (ES < 0 in this convention).
        For losses where larger = worse, ES_alpha = E[Y | Y > VaR_alpha] < 0
        when Y represents P&L returns (negative = loss).
    y : np.ndarray
        Observed returns. Shape (T,).
    alpha : float
        Coverage probability, e.g. 0.05 for 5% VaR.

    Returns
    -------
    np.ndarray
        Score values, shape (T,). Mean across T gives the period score.
    """
    # Clip es to be strictly negative to avoid log(0) or division by zero
    v = np.minimum(es, -1e-10)
    x = float(var)
    indicator = (y > x).astype(float)
    score = (
        (1.0 / alpha) * indicator * (y - x) / (-v)
        + np.log(-v)
        + (1.0 - 1.0 / alpha) * x / v
    )
    return score


def asymm_abs_loss(
    var: float,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Asymmetric absolute loss (tick loss) — strictly consistent for VaR_alpha.

    S(x, y; alpha) = (alpha - 1{y <= x}) * (y - x)

    Parameters
    ----------
    var : float
        VaR estimate.
    y : np.ndarray
        Observed returns. Shape (T,).
    alpha : float
        Coverage probability.

    Returns
    -------
    np.ndarray
        Score values, shape (T,).
    """
    x = float(var)
    indicator = (y <= x).astype(float)
    return (alpha - indicator) * (y - x)


# ---------------------------------------------------------------------------
# BAWSMonitor
# ---------------------------------------------------------------------------


class BAWSMonitor:
    """Bootstrap Adaptive Window Selection for VaR/ES monitoring.

    At each time step, selects the window length from ``candidate_windows``
    that minimises the block-bootstrapped expected score (Fissler-Ziegel for
    VaR+ES jointly, or tick loss for VaR alone). This provides a data-driven
    alternative to fixed-window or exponential-weighting approaches.

    The block bootstrap accounts for temporal dependence (volatility clustering,
    autocorrelation) that is endemic to financial return series.

    Parameters
    ----------
    alpha : float
        Coverage probability. alpha=0.05 gives 5% VaR / ES (tail coverage).
    candidate_windows : sequence of int, optional
        Window lengths to evaluate at each step. Default: [50, 100, 150, 200].
        Must have at least two candidates. All values must be positive integers.
    score_type : str, default 'fissler_ziegel'
        Scoring rule to use for window selection:
        - 'fissler_ziegel': strictly consistent for joint (VaR, ES).
        - 'asymm_abs_loss': strictly consistent for VaR only (tick loss).
    n_bootstrap : int, default 200
        Number of block bootstrap replicates. Higher values give more stable
        window selection at the cost of compute time.
    block_length : int or None, default None
        Bootstrap block length. If None, uses T^(1/3) (rounded) at each call,
        where T is the current history length. This is the standard rule of
        thumb for stationary bootstrap block length selection.
    min_block_length : int, default 3
        Minimum block length. Prevents degenerate blocks when T is small.
    random_state : int or None, default None
        Seed for reproducibility. Initialises a numpy Generator.

    Raises
    ------
    ValueError
        If candidate_windows contains fewer than 2 values, if any window
        is <= 0, or if alpha is outside (0, 1).

    Notes
    -----
    Call :meth:`fit` with the historical return series before calling
    :meth:`update`. Each :meth:`update` call appends one new observation,
    recomputes the window scores, and selects the winner.

    The history is stored internally and accessible via :meth:`history`.
    The plot via :meth:`plot` shows the selected window over time.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        candidate_windows: Optional[Sequence[int]] = None,
        score_type: str = "fissler_ziegel",
        n_bootstrap: int = 200,
        block_length: Optional[int] = None,
        min_block_length: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        _valid_scores = {"fissler_ziegel", "asymm_abs_loss"}
        if score_type not in _valid_scores:
            raise ValueError(
                f"score_type must be one of {sorted(_valid_scores)}, got {score_type!r}"
            )
        if candidate_windows is None:
            candidate_windows = [50, 100, 150, 200]
        candidate_windows = list(candidate_windows)
        if len(candidate_windows) < 2:
            raise ValueError(
                f"candidate_windows must have at least 2 values, got {len(candidate_windows)}"
            )
        if any(w <= 0 for w in candidate_windows):
            raise ValueError(
                f"All candidate_windows must be positive integers, got {candidate_windows}"
            )
        if n_bootstrap < 10:
            raise ValueError(f"n_bootstrap must be >= 10, got {n_bootstrap}")
        if min_block_length < 1:
            raise ValueError(f"min_block_length must be >= 1, got {min_block_length}")

        self.alpha = float(alpha)
        self.candidate_windows = sorted(candidate_windows)
        self.score_type = score_type
        self.n_bootstrap = int(n_bootstrap)
        self.block_length = block_length
        self.min_block_length = int(min_block_length)
        self.random_state = random_state

        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._history: np.ndarray = np.array([], dtype=np.float64)
        self._result_history: list[dict] = []
        self._t: int = 0
        self._fitted: bool = False
        self._current_window: Optional[int] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "BAWSMonitor":
        """Initialise the monitor with a historical return series.

        Parameters
        ----------
        returns : np.ndarray
            Historical return observations. Shape (T,). Returns are signed
            (positive = gain, negative = loss) so that VaR_alpha is the
            alpha-quantile (left tail).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the history length T is smaller than the smallest candidate window.
        """
        returns = np.asarray(returns, dtype=float)
        T = len(returns)
        min_window = self.candidate_windows[0]
        if T < min_window:
            raise ValueError(
                f"History length T={T} is smaller than the smallest candidate window "
                f"{min_window}. Provide at least {min_window} observations."
            )
        self._history = returns.copy()
        self._result_history = []
        self._t = 0
        self._fitted = True
        self._current_window = None
        return self

    def update(self, new_return: float) -> BAWSResult:
        """Append one new observation and select the optimal window.

        Parameters
        ----------
        new_return : float
            The latest return observation.

        Returns
        -------
        BAWSResult
            Result with the selected window, VaR/ES estimates, and per-window
            scores from the bootstrap evaluation.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError(
                "BAWSMonitor.update() called before fit(). Call fit(returns) first."
            )

        # Append new observation
        self._history = np.append(self._history, float(new_return))
        self._t += 1
        T = len(self._history)

        # Determine block length for this step
        bl = self._get_block_length(T)

        # Evaluate each candidate window via block bootstrap
        scores: dict[int, float] = {}
        valid_windows = [w for w in self.candidate_windows if w <= T]
        if not valid_windows:
            # All windows exceed current history — use shortest candidate
            valid_windows = [self.candidate_windows[0]]

        for w in valid_windows:
            window_data = self._history[-w:]
            score_val = self._bootstrap_score(window_data, bl)
            scores[w] = score_val

        # Fill in missing windows (those too long for current data) with inf
        for w in self.candidate_windows:
            if w not in scores:
                scores[w] = float("inf")

        # Select best window (minimum score); fall back to previous on tie
        best_window = min(valid_windows, key=lambda w: scores[w])

        # Compute VaR and ES from the selected window
        selected_data = self._history[-best_window:] if best_window <= T else self._history
        var_hat, es_hat = self._compute_var_es(selected_data)

        self._current_window = best_window

        result = BAWSResult(
            selected_window=best_window,
            var_estimate=var_hat,
            es_estimate=es_hat,
            scores=dict(scores),
            n_obs=T,
            time_step=self._t,
        )

        self._result_history.append({
            "time_step": self._t,
            "selected_window": best_window,
            "var_estimate": var_hat,
            "es_estimate": es_hat,
            "n_obs": T,
            **{f"score_w{w}": scores.get(w, float("inf")) for w in self.candidate_windows},
        })

        return result

    def update_batch(self, new_returns: Sequence[float]) -> list[BAWSResult]:
        """Process a sequence of new returns, calling update() for each.

        Produces identical results to calling update() sequentially.

        Parameters
        ----------
        new_returns : sequence of float
            New observations to process in order. May be empty, in which case
            an empty list is returned.

        Returns
        -------
        list[BAWSResult]
            One result per element of new_returns.
        """
        return [self.update(r) for r in new_returns]

    def current_window(self) -> int:
        """Return the most recently selected window length.

        Raises
        ------
        RuntimeError
            If called before any update() call (window not yet selected).
        """
        if self._current_window is None:
            raise RuntimeError(
                "current_window() called before any update(). "
                "Call fit() then update() first."
            )
        return self._current_window

    def history(self) -> pl.DataFrame:
        """Return one row per update() call as a Polars DataFrame.

        Columns:
        - time_step (Int64)
        - selected_window (Int64)
        - var_estimate (Float64)
        - es_estimate (Float64)
        - n_obs (Int64)
        - score_w{w} (Float64) for each candidate window

        Returns an empty DataFrame (with correct schema) if no updates have
        been made.
        """
        score_cols = {
            f"score_w{w}": pl.Float64 for w in self.candidate_windows
        }
        schema = {
            "time_step": pl.Int64,
            "selected_window": pl.Int64,
            "var_estimate": pl.Float64,
            "es_estimate": pl.Float64,
            "n_obs": pl.Int64,
            **score_cols,
        }
        if not self._result_history:
            return pl.DataFrame(schema=schema)
        df = pl.DataFrame(self._result_history)
        # Cast integer columns
        df = df.with_columns([
            pl.col("time_step").cast(pl.Int64),
            pl.col("selected_window").cast(pl.Int64),
            pl.col("n_obs").cast(pl.Int64),
        ])
        return df

    def plot(self, ax=None, title: Optional[str] = None):
        """Plot the selected window over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if None.
        title : str, optional
            Plot title. Defaults to 'BAWS: Selected Window Over Time'.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        if not self._result_history:
            ax.set_title(title or "BAWS: Selected Window Over Time")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Selected window")
            return ax

        df = self.history()
        t = df["time_step"].to_numpy()
        w = df["selected_window"].to_numpy()

        ax.step(t, w, where="post", color="steelblue", lw=1.5, label="Selected window")
        # Plot horizontal lines for each candidate
        for cw in self.candidate_windows:
            ax.axhline(y=cw, color="grey", lw=0.5, ls=":", alpha=0.5)

        ax.set_title(title or "BAWS: Selected Window Over Time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Selected window")
        ax.set_yticks(self.candidate_windows)
        ax.legend(loc="upper right")
        return ax

    def __repr__(self) -> str:
        return (
            f"BAWSMonitor(alpha={self.alpha}, "
            f"candidate_windows={self.candidate_windows}, "
            f"score_type='{self.score_type}', "
            f"n_bootstrap={self.n_bootstrap}, "
            f"t={self._t}, "
            f"current_window={self._current_window!r})"
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _get_block_length(self, T: int) -> int:
        """Compute the block length for the block bootstrap.

        Uses T^(1/3) if block_length is None (the standard rule of thumb).
        Clamps to min_block_length from below and T//2 from above.
        """
        if self.block_length is not None:
            bl = int(self.block_length)
        else:
            bl = max(self.min_block_length, int(round(T ** (1.0 / 3.0))))
        return max(self.min_block_length, min(bl, max(T // 2, 1)))

    def _block_bootstrap(self, data: np.ndarray, block_length: int) -> np.ndarray:
        """Generate one block-bootstrap replicate of ``data``.

        Draws overlapping blocks of length ``block_length`` with replacement,
        concatenates them, and trims to the original length T.

        When block_length == 1, this degenerates to an iid bootstrap.

        Parameters
        ----------
        data : np.ndarray
            Input series. Shape (T,).
        block_length : int
            Length of each block.

        Returns
        -------
        np.ndarray
            Bootstrap replicate, same length as data.
        """
        T = len(data)
        if T == 0:
            return data.copy()
        if block_length >= T:
            # Degenerate: single block — resample entire series with replacement
            idx = self._rng.integers(0, T, size=T)
            return data[idx]

        n_blocks = int(np.ceil(T / block_length))
        # Draw block starting positions uniformly from [0, T - block_length]
        max_start = T - block_length
        starts = self._rng.integers(0, max_start + 1, size=n_blocks)
        replicate = np.concatenate([data[s: s + block_length] for s in starts])
        return replicate[:T]

    def _compute_var_es(self, data: np.ndarray) -> tuple[float, float]:
        """Compute empirical VaR_alpha and ES_alpha from data.

        VaR_alpha = alpha-quantile of the distribution.
        ES_alpha = mean of observations <= VaR_alpha (left-tail conditional mean).

        For positive-is-gain convention, VaR at alpha=0.05 is the 5th percentile
        (the level exceeded with probability 1-alpha from below).

        Returns
        -------
        (var_hat, es_hat) : tuple[float, float]
            Both are typically negative for gain/loss P&L returns with alpha < 0.5.
        """
        if len(data) == 0:
            return 0.0, 0.0

        var_hat = float(np.quantile(data, self.alpha))

        # ES: mean of returns that are at or below the VaR threshold
        tail = data[data <= var_hat]
        if len(tail) == 0:
            es_hat = var_hat  # Degenerate case: no observations in tail
        else:
            es_hat = float(np.mean(tail))

        # Ensure ES <= VaR (ES is always at least as severe as VaR)
        es_hat = min(es_hat, var_hat)

        return var_hat, es_hat

    def _score_window(self, data: np.ndarray) -> float:
        """Compute the expected score for a single dataset ``data``.

        Fits VaR/ES on all-but-last observation, scores against the last.
        For window evaluation during bootstrap we score the full window.
        """
        if len(data) < 2:
            return float("inf")
        var_hat, es_hat = self._compute_var_es(data[:-1])
        y = data[-1:]
        if self.score_type == "fissler_ziegel":
            s = fissler_ziegel_score(var_hat, es_hat, y, self.alpha)
        else:
            s = asymm_abs_loss(var_hat, y, self.alpha)
        return float(np.mean(s))

    def _bootstrap_score(self, window_data: np.ndarray, block_length: int) -> float:
        """Compute the mean bootstrap score for a given window.

        Generates n_bootstrap block-bootstrap replicates of window_data and
        computes the mean score across replicates. Returns the average as the
        bootstrap estimate of expected score for this window length.

        Parameters
        ----------
        window_data : np.ndarray
            The slice of history to evaluate (length = window size).
        block_length : int
            Block length for the block bootstrap.

        Returns
        -------
        float
            Mean bootstrapped score. Lower is better (more negative = better
            fit under Fissler-Ziegel / tick loss conventions).
        """
        if len(window_data) < 2:
            return float("inf")

        replicate_scores = np.empty(self.n_bootstrap, dtype=np.float64)
        for b in range(self.n_bootstrap):
            replicate = self._block_bootstrap(window_data, block_length)
            replicate_scores[b] = self._score_window(replicate)

        # Filter out inf/nan replicates (can happen with degenerate blocks)
        finite = replicate_scores[np.isfinite(replicate_scores)]
        if len(finite) == 0:
            return float("inf")
        return float(np.mean(finite))
