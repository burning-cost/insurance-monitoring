"""Score decomposition with asymptotic inference (Dimitriadis & Puke, arXiv:2603.04275).

Decomposes any proper scoring function into:

    S(F, y) = MCB + UNC - DSC

where:
- UNC = S(y_bar, y): score of the climate forecast. Irreducible.
- DSC = UNC - S(F*, y): improvement over climate after recalibrating F.
- MCB = S(F, y) - S(F*, y): excess score from miscalibration. Zero if well-calibrated.

F* is the linear recalibration of F — from Mincer-Zarnowitz (MZ) regression for mean
forecasts, or quantile regression for quantile forecasts.

The key advance over prior work (Murphy 1977, Bröcker 2009, Wüthrich 2025) is formal
asymptotic inference on each component, with HAC standard errors that are valid under
temporal dependence. Two-sample tests enable MCB(A) vs MCB(B) and DSC(A) vs DSC(B)
comparisons with higher power than aggregate Diebold-Mariano when models differ in
only one dimension.

Reference implementation: https://github.com/marius-cp/SDI (R)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScoreDecompositionResult:
    """Score decomposition with hypothesis tests for a single forecast.

    The decomposition identity is::

        S(F, y) = MCB + UNC - DSC

    Attributes
    ----------
    score
        Mean score S(F, y) — the raw score of the forecast.
    miscalibration
        MCB = S(F, y) - S(F*, y). Zero for a perfectly calibrated forecast.
    discrimination
        DSC = S(y_bar, y) - S(F*, y). Zero for a constant (uninformative) forecast.
    uncertainty
        UNC = S(y_bar, y). Irreducible baseline determined by the data.
    mcb_pvalue
        Two-sided p-value for H0: MCB = 0. Small values indicate miscalibration.
    dsc_pvalue
        Two-sided p-value for H0: DSC = 0. Small values indicate the forecast
        carries skill beyond the climate forecast.
    mcb_se
        Asymptotic standard error of the MCB estimate (HAC).
    dsc_se
        Asymptotic standard error of the DSC estimate (HAC).
    recalib_intercept
        MZ regression intercept b (recalibrated forecast = b + a * F).
    recalib_slope
        MZ regression slope a.
    n
        Number of observations.
    """

    score: float
    miscalibration: float
    discrimination: float
    uncertainty: float
    mcb_pvalue: float
    dsc_pvalue: float
    mcb_se: float
    dsc_se: float
    recalib_intercept: float
    recalib_slope: float
    n: int

    def summary(self) -> str:
        """Return a governance-ready plain-text summary."""
        lines = [
            f"Score decomposition (n={self.n:,})",
            f"  Score (S):        {self.score:.6f}",
            f"  MCB:              {self.miscalibration:.6f}  "
            f"(SE={self.mcb_se:.6f}, p={self.mcb_pvalue:.4f})",
            f"  DSC:              {self.discrimination:.6f}  "
            f"(SE={self.dsc_se:.6f}, p={self.dsc_pvalue:.4f})",
            f"  UNC:              {self.uncertainty:.6f}",
            f"  MZ regression:    y_hat* = {self.recalib_intercept:.4f} + "
            f"{self.recalib_slope:.4f} * y_hat",
            "",
            f"  MCB=0 test:   {'FAIL (miscalibrated)' if self.mcb_pvalue < 0.05 else 'PASS'}  "
            f"p={self.mcb_pvalue:.4f}",
            f"  DSC=0 test:   {'FAIL (no skill)' if self.dsc_pvalue > 0.05 else 'PASS (has skill)'}  "
            f"p={self.dsc_pvalue:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ScoreDecompositionResult("
            f"S={self.score:.4f}, MCB={self.miscalibration:.4f} (p={self.mcb_pvalue:.3f}), "
            f"DSC={self.discrimination:.4f} (p={self.dsc_pvalue:.3f}), "
            f"UNC={self.uncertainty:.4f}, n={self.n})"
        )


@dataclass
class TwoForecastSDIResult:
    """Comparison of two forecast series using score decomposition inference.

    Tests whether models A and B differ in their MCB or DSC components. This
    is more powerful than an overall Diebold-Mariano test when the models differ
    in only one dimension — e.g., both well-calibrated but one has better
    discrimination.

    Attributes
    ----------
    result_a
        Full decomposition for forecast A.
    result_b
        Full decomposition for forecast B.
    delta_score
        S(A, y) - S(B, y). Positive means A has higher (worse) score.
    delta_score_pvalue
        Diebold-Mariano test p-value for H0: S(A) = S(B).
    delta_mcb
        MCB(A) - MCB(B).
    delta_mcb_pvalue
        Two-sided p-value for H0: MCB(A) = MCB(B).
    delta_dsc
        DSC(A) - DSC(B). Negative means A is more discriminating.
    delta_dsc_pvalue
        Two-sided p-value for H0: DSC(A) = DSC(B).
    combined_pvalue
        IU intersection-union combined p-value. Controls FWER: the combined
        null (both MCB equal AND DSC equal) is rejected when
        max(p_dm, 2*min(p_mcb_A, p_mcb_B)) < alpha.
    """

    result_a: ScoreDecompositionResult
    result_b: ScoreDecompositionResult
    delta_score: float
    delta_score_pvalue: float
    delta_mcb: float
    delta_mcb_pvalue: float
    delta_dsc: float
    delta_dsc_pvalue: float
    combined_pvalue: float

    def summary(self) -> str:
        """Return a governance-ready plain-text comparison summary."""
        lines = [
            f"Two-forecast SDI comparison (n={self.result_a.n:,})",
            "",
            "  Forecast A:",
            f"    S={self.result_a.score:.6f}, MCB={self.result_a.miscalibration:.6f}, "
            f"DSC={self.result_a.discrimination:.6f}",
            "  Forecast B:",
            f"    S={self.result_b.score:.6f}, MCB={self.result_b.miscalibration:.6f}, "
            f"DSC={self.result_b.discrimination:.6f}",
            "",
            f"  Delta score (A-B): {self.delta_score:+.6f}  (DM p={self.delta_score_pvalue:.4f})",
            f"  Delta MCB (A-B):   {self.delta_mcb:+.6f}  (p={self.delta_mcb_pvalue:.4f})",
            f"  Delta DSC (A-B):   {self.delta_dsc:+.6f}  (p={self.delta_dsc_pvalue:.4f})",
            f"  Combined IU p:     {self.combined_pvalue:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TwoForecastSDIResult("
            f"delta_S={self.delta_score:+.4f} (p={self.delta_score_pvalue:.3f}), "
            f"delta_MCB={self.delta_mcb:+.4f} (p={self.delta_mcb_pvalue:.3f}), "
            f"delta_DSC={self.delta_dsc:+.4f} (p={self.delta_dsc_pvalue:.3f}))"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ScoreDecompositionTest:
    """Asymptotic inference for score decompositions (Dimitriadis & Puke, 2026).

    Decomposes S(F, y) = MCB + UNC - DSC and tests whether each component is
    significantly non-zero. For two competing forecasts, tests H0: MCB(A) = MCB(B)
    and H0: DSC(A) = DSC(B) with higher power than the overall Diebold-Mariano
    test when forecasts differ in only one dimension.

    The recalibration step uses Mincer-Zarnowitz (MZ) regression:
    - 'mse': OLS regression of y on [1, y_hat], HAC via Newey-West.
    - 'mae': same OLS form but absolute error (quantile at 0.5).
    - 'quantile': quantile regression at the given alpha level.

    Parameters
    ----------
    score_type
        Scoring function to use. 'mse' for squared error, 'mae' for absolute
        error, 'quantile' for pinball loss at the given alpha level.
    alpha
        Quantile level for score_type='quantile'. Ignored for 'mse'/'mae'.
        Must be in (0, 1).
    hac_lags
        Newey-West lag truncation. If None (default), auto-selects using the
        Newey-West (1994) rule: floor(4*(n/100)^(2/9)). Set hac_lags=0 for
        iid data (this recovers standard OLS standard errors).
    exposure
        Observation weights (e.g., policy durations). Applied as regression
        weights in the MZ step. If None, weights are uniform.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> y_hat = rng.gamma(2, 0.5, n)
    >>> y = y_hat + rng.normal(0, 0.5, n)
    >>> sdi = ScoreDecompositionTest(score_type='mse')
    >>> result = sdi.fit_single(y, y_hat)
    >>> result.mcb_pvalue > 0.05  # well-calibrated forecast: MCB not significant
    True

    References
    ----------
    Dimitriadis & Puke (2026), arXiv:2603.04275.
    R reference: https://github.com/marius-cp/SDI
    """

    def __init__(
        self,
        score_type: Literal["mse", "quantile", "mae"] = "mse",
        alpha: float = 0.5,
        hac_lags: int | None = None,
        exposure: npt.ArrayLike | None = None,
    ) -> None:
        if score_type not in ("mse", "quantile", "mae"):
            raise ValueError(f"score_type must be 'mse', 'quantile', or 'mae', got {score_type!r}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if hac_lags is not None and hac_lags < 0:
            raise ValueError(f"hac_lags must be non-negative or None, got {hac_lags}")
        self.score_type = score_type
        self.alpha = alpha
        self.hac_lags = hac_lags
        self._exposure = exposure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_single(
        self,
        y: npt.ArrayLike,
        y_hat: npt.ArrayLike,
    ) -> ScoreDecompositionResult:
        """Decompose and test a single forecast series.

        Parameters
        ----------
        y
            Observed outcomes. Shape (n,).
        y_hat
            Forecast values. Shape (n,).

        Returns
        -------
        ScoreDecompositionResult
            Point estimates, standard errors, and p-values for MCB and DSC.
        """
        y_arr, y_hat_arr, w = self._validate(y, y_hat)
        n = len(y_arr)
        nlags = self._auto_lags(n)

        if self.score_type == "mse":
            return self._fit_single_mse(y_arr, y_hat_arr, w, n, nlags)
        else:
            q = 0.5 if self.score_type == "mae" else self.alpha
            return self._fit_single_quantile(y_arr, y_hat_arr, w, n, nlags, q)

    def fit_two(
        self,
        y: npt.ArrayLike,
        y_hat_a: npt.ArrayLike,
        y_hat_b: npt.ArrayLike,
    ) -> TwoForecastSDIResult:
        """Decompose and compare two competing forecast series.

        Runs fit_single on each series and then performs joint HAC inference
        on the differences, yielding tests for H0: MCB(A) = MCB(B) and
        H0: DSC(A) = DSC(B).

        Parameters
        ----------
        y
            Observed outcomes. Shape (n,).
        y_hat_a
            Forecast A. Shape (n,).
        y_hat_b
            Forecast B. Shape (n,).

        Returns
        -------
        TwoForecastSDIResult
        """
        y_arr = np.asarray(y, dtype=np.float64)
        y_hat_a_arr = np.asarray(y_hat_a, dtype=np.float64)
        y_hat_b_arr = np.asarray(y_hat_b, dtype=np.float64)

        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y_arr.shape}")
        if y_hat_a_arr.shape != y_arr.shape:
            raise ValueError("y and y_hat_a must have the same shape")
        if y_hat_b_arr.shape != y_arr.shape:
            raise ValueError("y and y_hat_b must have the same shape")
        if len(y_arr) < 4:
            raise ValueError("At least 4 observations are required for two-sample test")

        w = self._get_weights(len(y_arr))
        n = len(y_arr)
        nlags = self._auto_lags(n)

        if self.score_type == "mse":
            res_a, z_a = self._fit_single_mse_with_scores(y_arr, y_hat_a_arr, w, n, nlags)
            res_b, z_b = self._fit_single_mse_with_scores(y_arr, y_hat_b_arr, w, n, nlags)
        else:
            q = 0.5 if self.score_type == "mae" else self.alpha
            res_a, z_a = self._fit_single_quantile_with_scores(y_arr, y_hat_a_arr, w, n, nlags, q)
            res_b, z_b = self._fit_single_quantile_with_scores(y_arr, y_hat_b_arr, w, n, nlags, q)

        return self._two_sample_inference(res_a, res_b, z_a, z_b, n, nlags)

    def summary(self, result: ScoreDecompositionResult | TwoForecastSDIResult) -> str:
        """Return a governance-ready plain-text report."""
        return result.summary()

    def plot(
        self,
        result: ScoreDecompositionResult | TwoForecastSDIResult,
        ax=None,  # type: ignore[type-arg]
    ):
        """Stacked bar chart: MCB/DSC/UNC decomposition with p-value annotations.

        Parameters
        ----------
        result
            Result from fit_single or fit_two.
        ax
            Matplotlib axes. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        if isinstance(result, ScoreDecompositionResult):
            forecasts = ["Forecast"]
            results = [result]
        else:
            forecasts = ["A", "B"]
            results = [result.result_a, result.result_b]

        x = np.arange(len(forecasts))
        width = 0.5

        mcb_vals = [r.miscalibration for r in results]
        dsc_vals = [r.discrimination for r in results]
        unc_vals = [r.uncertainty for r in results]

        # Visual layout: S = UNC - DSC + MCB
        # Stack: UNC base, DSC reduces it (show as additive, annotate separately),
        # MCB sits on top of (UNC - DSC)
        unc_minus_dsc = [u - d for u, d in zip(unc_vals, dsc_vals)]
        ax.bar(x, unc_minus_dsc, width, label="UNC-DSC", color="#d4e6f1", edgecolor="white")
        ax.bar(x, dsc_vals, width, bottom=unc_minus_dsc, label="DSC",
               color="#2e86c1", edgecolor="white")
        ax.bar(x, mcb_vals, width, bottom=unc_minus_dsc, label="MCB",
               color="#e74c3c", edgecolor="white", alpha=0.8)

        for i, r in enumerate(results):
            p_mcb = r.mcb_pvalue
            p_dsc = r.dsc_pvalue
            ax.text(
                x[i], r.score * 1.02,
                f"MCB p={p_mcb:.3f}\nDSC p={p_dsc:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(forecasts)
        ax.set_ylabel("Score contribution")
        ax.set_title("Score decomposition: MCB / DSC / UNC")
        ax.legend(loc="upper right")
        return ax

    # ------------------------------------------------------------------
    # MSE (squared error) path
    # ------------------------------------------------------------------

    def _fit_single_mse(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        w: np.ndarray,
        n: int,
        nlags: int,
    ) -> ScoreDecompositionResult:
        result, _ = self._fit_single_mse_with_scores(y, y_hat, w, n, nlags)
        return result

    def _fit_single_mse_with_scores(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        w: np.ndarray,
        n: int,
        nlags: int,
    ) -> tuple[ScoreDecompositionResult, np.ndarray]:
        """Return (result, score_matrix) where score_matrix is (n, 3).

        Columns: [s_i, dsc_i, unc_i] — per-observation score contributions.
        The HAC covariance of their column means gives the asymptotic variance
        of (S, DSC, UNC), and hence MCB = S - UNC + DSC.
        """
        w_norm = w / w.sum()  # normalised weights summing to 1

        # Per-observation squared error scores
        s_i = (y_hat - y) ** 2

        # MZ regression: OLS of y on [1, y_hat] with weights w_norm
        a, b = self._weighted_ols(y, y_hat, w_norm)
        y_hat_star = b + a * y_hat

        # Climate forecast: weighted mean
        y_bar = float(np.sum(w_norm * y))

        # Component estimates (all exposure-weighted means)
        score = float(np.sum(w_norm * s_i))
        unc_i = (y_bar - y) ** 2
        unc = float(np.sum(w_norm * unc_i))
        dsc_i = (y_bar - y) ** 2 - (y_hat_star - y) ** 2
        dsc = float(np.sum(w_norm * dsc_i))
        mcb = score - (unc - dsc)  # MCB = S - S(F*) = S - (UNC - DSC)

        # Score matrix centred at estimates for HAC
        Z = np.column_stack([
            s_i - score,
            dsc_i - dsc,
            unc_i - unc,
        ])

        omega = self._hac_cov(Z, nlags)  # (3, 3)

        # MCB = S - UNC + DSC, gradient w.r.t. [S, DSC, UNC] is [1, 1, -1]
        g_mcb = np.array([1.0, 1.0, -1.0])
        var_mcb = float(g_mcb @ omega @ g_mcb) / n
        mcb_se = float(np.sqrt(max(var_mcb, 0.0)))

        # DSC SE directly from omega[1,1]
        var_dsc = float(omega[1, 1]) / n
        dsc_se = float(np.sqrt(max(var_dsc, 0.0)))

        # Two-sided t-tests
        t_mcb = mcb / mcb_se if mcb_se > 0 else 0.0
        t_dsc = dsc / dsc_se if dsc_se > 0 else 0.0
        p_mcb = float(2 * (1 - norm.cdf(abs(t_mcb))))
        p_dsc = float(2 * (1 - norm.cdf(abs(t_dsc))))

        result = ScoreDecompositionResult(
            score=score,
            miscalibration=mcb,
            discrimination=dsc,
            uncertainty=unc,
            mcb_pvalue=p_mcb,
            dsc_pvalue=p_dsc,
            mcb_se=mcb_se,
            dsc_se=dsc_se,
            recalib_intercept=b,
            recalib_slope=a,
            n=n,
        )
        # Return full (uncentred) score matrix for two-sample joint inference
        score_matrix = np.column_stack([s_i, dsc_i, unc_i])
        return result, score_matrix

    # ------------------------------------------------------------------
    # Quantile / MAE path
    # ------------------------------------------------------------------

    def _fit_single_quantile(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        w: np.ndarray,
        n: int,
        nlags: int,
        q: float,
    ) -> ScoreDecompositionResult:
        result, _ = self._fit_single_quantile_with_scores(y, y_hat, w, n, nlags, q)
        return result

    def _fit_single_quantile_with_scores(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        w: np.ndarray,
        n: int,
        nlags: int,
        q: float,
    ) -> tuple[ScoreDecompositionResult, np.ndarray]:
        """Quantile forecast decomposition via quantile regression MZ step."""
        from statsmodels.regression.quantile_regression import QuantReg

        w_norm = w / w.sum()

        # Per-observation pinball scores
        s_i = _pinball(y, y_hat, q)

        # MZ recalibration: quantile regression of y on [1, y_hat]
        X = np.column_stack([np.ones(n), y_hat])
        model = QuantReg(y, X)
        try:
            fit = model.fit(q=q, vcov="robust", kernel="epa", bandwidth="hsheather",
                            max_iter=2000)
        except Exception:
            # Fall back to basic fit if HAC options not supported in this statsmodels version
            fit = model.fit(q=q, max_iter=2000)

        b, a = float(fit.params[0]), float(fit.params[1])
        y_hat_star = b + a * y_hat

        # Climate: marginal quantile (unweighted, as in Dimitriadis & Puke)
        y_bar = float(np.quantile(y, q))
        y_bar_arr = np.full(n, y_bar)

        # Component estimates (exposure-weighted)
        score = float(np.sum(w_norm * s_i))
        unc_i = _pinball(y, y_bar_arr, q)
        unc = float(np.sum(w_norm * unc_i))
        dsc_i = unc_i - _pinball(y, y_hat_star, q)
        dsc = float(np.sum(w_norm * dsc_i))
        mcb = score - (unc - dsc)

        # HAC on centred score matrix
        Z = np.column_stack([
            s_i - score,
            dsc_i - dsc,
            unc_i - unc,
        ])
        omega = self._hac_cov(Z, nlags)

        g_mcb = np.array([1.0, 1.0, -1.0])
        var_mcb = float(g_mcb @ omega @ g_mcb) / n
        mcb_se = float(np.sqrt(max(var_mcb, 0.0)))
        var_dsc = float(omega[1, 1]) / n
        dsc_se = float(np.sqrt(max(var_dsc, 0.0)))

        t_mcb = mcb / mcb_se if mcb_se > 0 else 0.0
        t_dsc = dsc / dsc_se if dsc_se > 0 else 0.0
        p_mcb = float(2 * (1 - norm.cdf(abs(t_mcb))))
        p_dsc = float(2 * (1 - norm.cdf(abs(t_dsc))))

        result = ScoreDecompositionResult(
            score=score,
            miscalibration=mcb,
            discrimination=dsc,
            uncertainty=unc,
            mcb_pvalue=p_mcb,
            dsc_pvalue=p_dsc,
            mcb_se=mcb_se,
            dsc_se=dsc_se,
            recalib_intercept=b,
            recalib_slope=a,
            n=n,
        )
        score_matrix = np.column_stack([s_i, dsc_i, unc_i])
        return result, score_matrix

    # ------------------------------------------------------------------
    # Two-sample inference
    # ------------------------------------------------------------------

    def _two_sample_inference(
        self,
        res_a: ScoreDecompositionResult,
        res_b: ScoreDecompositionResult,
        z_a: np.ndarray,  # (n, 3): [s_i, dsc_i, unc_i] for A
        z_b: np.ndarray,  # (n, 3): [s_i, dsc_i, unc_i] for B
        n: int,
        nlags: int,
    ) -> TwoForecastSDIResult:
        """Joint HAC inference on the 6-dimensional score vector."""
        # Build 6-col matrix: [sA, dscA, uncA, sB, dscB, uncB], centred
        Z6 = np.column_stack([z_a, z_b])
        Z6_centred = Z6 - Z6.mean(axis=0)
        omega_6 = self._hac_cov(Z6_centred, nlags)  # (6, 6)

        # DM test: Var(delta_s) = Var(sA - sB) = omega[0,0] + omega[3,3] - 2*omega[0,3]
        delta_s = float(res_a.score - res_b.score)
        var_ds = (omega_6[0, 0] + omega_6[3, 3] - 2.0 * omega_6[0, 3]) / n
        se_ds = float(np.sqrt(max(var_ds, 0.0)))
        t_dm = delta_s / se_ds if se_ds > 0 else 0.0
        p_dm = float(2 * (1 - norm.cdf(abs(t_dm))))

        # MCB difference: delta_MCB = MCB(A) - MCB(B) = (SA - SB) - (UNCA - UNCB) + (DSCA - DSCB)
        # gradient: g = [1, 1, -1, -1, -1, 1] for cols [sA, dscA, uncA, sB, dscB, uncB]
        delta_mcb = float(res_a.miscalibration - res_b.miscalibration)
        g_dmcb = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
        var_dmcb = float(g_dmcb @ omega_6 @ g_dmcb) / n
        se_dmcb = float(np.sqrt(max(var_dmcb, 0.0)))
        t_dmcb = delta_mcb / se_dmcb if se_dmcb > 0 else 0.0
        p_dmcb = float(2 * (1 - norm.cdf(abs(t_dmcb))))

        # DSC difference: delta_DSC = DSC(A) - DSC(B)
        # gradient: g = [0, 1, 0, 0, -1, 0]
        delta_dsc = float(res_a.discrimination - res_b.discrimination)
        g_ddsc = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
        var_ddsc = float(g_ddsc @ omega_6 @ g_ddsc) / n
        se_ddsc = float(np.sqrt(max(var_ddsc, 0.0)))
        t_ddsc = delta_dsc / se_ddsc if se_ddsc > 0 else 0.0
        p_ddsc = float(2 * (1 - norm.cdf(abs(t_ddsc))))

        # IU combined p-value: reject joint null when combined < alpha
        combined_pvalue = float(max(p_dm, 2.0 * min(res_a.mcb_pvalue, res_b.mcb_pvalue)))

        return TwoForecastSDIResult(
            result_a=res_a,
            result_b=res_b,
            delta_score=delta_s,
            delta_score_pvalue=p_dm,
            delta_mcb=delta_mcb,
            delta_mcb_pvalue=p_dmcb,
            delta_dsc=delta_dsc,
            delta_dsc_pvalue=p_ddsc,
            combined_pvalue=combined_pvalue,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(
        self,
        y: npt.ArrayLike,
        y_hat: npt.ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_arr = np.asarray(y, dtype=np.float64)
        y_hat_arr = np.asarray(y_hat, dtype=np.float64)

        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y_arr.shape}")
        if y_hat_arr.ndim != 1:
            raise ValueError(f"y_hat must be 1-dimensional, got shape {y_hat_arr.shape}")
        if len(y_arr) != len(y_hat_arr):
            raise ValueError(
                f"y and y_hat must have the same length: {len(y_arr)} vs {len(y_hat_arr)}"
            )
        if len(y_arr) < 4:
            raise ValueError("At least 4 observations are required")

        w = self._get_weights(len(y_arr))
        return y_arr, y_hat_arr, w

    def _get_weights(self, n: int) -> np.ndarray:
        if self._exposure is None:
            return np.ones(n, dtype=np.float64)
        w = np.asarray(self._exposure, dtype=np.float64).ravel()
        if len(w) != n:
            raise ValueError(
                f"exposure length {len(w)} does not match data length {n}"
            )
        if np.any(w <= 0):
            raise ValueError("All exposure values must be strictly positive")
        return w

    def _auto_lags(self, n: int) -> int:
        if self.hac_lags is not None:
            return self.hac_lags
        # Newey-West (1994) automatic bandwidth: floor(4*(n/100)^(2/9))
        return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

    @staticmethod
    def _weighted_ols(
        y: np.ndarray,
        x: np.ndarray,
        w: np.ndarray,
    ) -> tuple[float, float]:
        """Weighted OLS of y on [1, x]. Returns (slope a, intercept b)."""
        sum_w = w.sum()
        x_bar = float(np.sum(w * x)) / sum_w
        y_bar = float(np.sum(w * y)) / sum_w
        cov_xy = float(np.sum(w * (x - x_bar) * (y - y_bar)))
        var_x = float(np.sum(w * (x - x_bar) ** 2))
        if var_x < 1e-15:
            # Degenerate: all forecasts identical — slope is undefined, use 0
            return 0.0, y_bar
        a = cov_xy / var_x
        b = y_bar - a * x_bar
        return float(a), float(b)

    @staticmethod
    def _hac_cov(Z: np.ndarray, nlags: int) -> np.ndarray:
        """Newey-West (Bartlett kernel) HAC covariance estimator.

        Z has shape (n, k), assumed mean-zero. Returns the (k, k) long-run
        covariance matrix Omega such that sqrt(n) * mean(Z) -> N(0, Omega).

        The estimator is:
            Omega = Gamma_0 + sum_{l=1}^{L} w_l * (Gamma_l + Gamma_l')
        where:
            Gamma_l = (1/n) * Z[l:]' Z[:-l]   (cross-autocovariance at lag l)
            w_l = 1 - l / (L + 1)             (Bartlett kernel weight)

        For nlags=0, returns only Gamma_0 = Z'Z/n (standard sandwich estimator).
        """
        n = Z.shape[0]
        n_eff = float(n)

        # Lag 0: Gamma_0 = Z' Z / n (vectorised)
        Omega: np.ndarray = Z.T @ Z / n_eff

        for lag in range(1, nlags + 1):
            w_lag = 1.0 - lag / (nlags + 1.0)  # Bartlett kernel weight
            gamma = Z[lag:].T @ Z[:-lag] / n_eff  # (k, k), vectorised
            Omega = Omega + w_lag * (gamma + gamma.T)

        return Omega


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _pinball(y: np.ndarray, q_hat: np.ndarray, alpha: float) -> np.ndarray:
    """Per-observation pinball (quantile) loss.

    pinball(y, q, alpha) = (y - q) * alpha          if y >= q
                         = (q - y) * (1 - alpha)    if y < q

    Equivalently: (y - q) * (alpha - (y < q)).
    """
    return (y - q_hat) * (alpha - (y < q_hat).astype(np.float64))
