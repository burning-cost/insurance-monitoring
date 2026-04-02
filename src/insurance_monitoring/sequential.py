"""
Sequential (anytime-valid) A/B testing for insurance champion/challenger experiments.

Uses mixture Sequential Probability Ratio Test (mSPRT) from Johari et al. (2022).
The test statistic is an e-process — valid at any stopping time — so results can
be checked weekly/monthly without inflating type I error.

References
----------
- Johari et al. (2022). Always Valid Inference. OR 70(3). arXiv:1512.04922.
- Ramdas et al. (2023). Game-Theoretic Statistics and SAVI. Stat Sci 38(4).
- Howard et al. (2021). Time-uniform confidence sequences. AoS 49(2).
- Hao, Turner, Grunwald (2024). E-values for k-Sample Tests. Sankhya A.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional
import datetime

import numpy as np
import polars as pl
from scipy.stats import norm as norm_dist
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class SequentialTestResult:
    """Result returned by :meth:`SequentialTest.update`.

    Attributes
    ----------
    decision : str
        Stopping decision: one of ``'reject_H0'``, ``'inconclusive'``,
        ``'futility'``, or ``'max_duration_reached'``.
    should_stop : bool
        True when decision is not ``'inconclusive'``. Use this as the
        stopping flag in experiment loops.
    lambda_value : float
        e-process value Lambda_n. The test rejects H0 when Lambda_n >= 1/alpha.
    log_lambda_value : float
        log(Lambda_n). Monotone with evidence — better than lambda_value for
        plotting the evidence trajectory over time.
    threshold : float
        Rejection threshold = 1/alpha. Fixed at test construction and constant
        throughout the experiment.
    champion_rate : float
        Accumulated champion frequency: total_champion_claims /
        total_champion_exposure (claims per car-year). For severity tests,
        this is the mean log-cost.
    challenger_rate : float
        Accumulated challenger frequency or mean log-cost.
    rate_ratio : float
        challenger_rate / champion_rate. Values > 1 mean the challenger has a
        higher claims rate (or cost) than the champion.
    rate_ratio_ci_lower : float
        Lower bound of the anytime-valid (1-alpha) confidence sequence for the
        rate ratio. Valid simultaneously at all interim looks — no Bonferroni
        correction required for interim peeking.
    rate_ratio_ci_upper : float
        Upper bound of the confidence sequence. Returns inf when either arm
        has fewer than 5 accumulated claims.
    total_champion_claims : float
        Cumulative claims across all champion arms of update() calls.
    total_champion_exposure : float
        Cumulative exposure (car-years) across all champion arms.
    total_challenger_claims : float
        Cumulative claims across all challenger arms.
    total_challenger_exposure : float
        Cumulative exposure across all challenger arms.
    n_updates : int
        Number of update() calls made on this test instance.
    total_calendar_time_days : float
        Elapsed calendar time between first and most recent update() call.
        Zero if no date arguments were provided.
    prob_challenger_better : float
        P(lambda_B < lambda_A | data) from a Gamma-Poisson conjugate model.
        Informational only — not used for the stopping decision.
    summary : str
        Human-readable one-line summary including rate ratio, confidence
        sequence, evidence level, and decision.
    """

    # Stopping decision
    decision: str
    should_stop: bool

    # Test statistic
    lambda_value: float
    log_lambda_value: float
    threshold: float

    # Effect estimates
    champion_rate: float
    challenger_rate: float
    rate_ratio: float
    rate_ratio_ci_lower: float
    rate_ratio_ci_upper: float

    # Accumulated state
    total_champion_claims: float
    total_champion_exposure: float
    total_challenger_claims: float
    total_challenger_exposure: float
    n_updates: int
    total_calendar_time_days: float

    # Bayesian secondary display (informational only — not used for stopping decision)
    prob_challenger_better: float

    # Human-readable summary
    summary: str


# ---------------------------------------------------------------------------
# Internal state dataclasses (private)
# ---------------------------------------------------------------------------

@dataclass
class _FreqState:
    C_A: float = 0.0
    E_A: float = 0.0
    C_B: float = 0.0
    E_B: float = 0.0


@dataclass
class _SevState:
    n_A: float = 0.0
    sum_log_A: float = 0.0
    ss_log_A: float = 0.0
    n_B: float = 0.0
    sum_log_B: float = 0.0
    ss_log_B: float = 0.0


# ---------------------------------------------------------------------------
# Core mathematics (module-private)
# ---------------------------------------------------------------------------

def _gaussian_msprt(
    theta_hat: float,
    sigma_sq: float,
    tau_sq: float,
    alternative: str,
) -> float:
    """Return log(Lambda_n) for Gaussian mSPRT.

    Integrates over a N(0, tau^2) prior on the log-rate-ratio theta.
    The resulting Lambda_n is an e-process under H0: theta = 0.

    Parameters
    ----------
    theta_hat : float
        Observed effect (log-rate-ratio or log-severity-ratio).
    sigma_sq : float
        Estimated variance of theta_hat (delta-method or pooled t-test).
    tau_sq : float
        Prior variance on theta (= tau**2 from SequentialTest).
    alternative : str
        'two_sided', 'greater', or 'less'.
    """
    if sigma_sq < 1e-12:
        return 0.0

    log_lambda = (
        0.5 * math.log(tau_sq / (tau_sq + sigma_sq))
        + theta_hat ** 2 / (2.0 * (sigma_sq + tau_sq))
    )

    if alternative == "two_sided":
        return log_lambda
    elif alternative == "greater":
        # Half-normal prior: only positive effects count
        return log_lambda + math.log(2) if theta_hat >= 0 else -math.inf
    else:  # 'less'
        # Half-normal prior: only negative effects (challenger < champion) count
        return log_lambda + math.log(2) if theta_hat <= 0 else -math.inf


def _poisson_msprt(
    C_A: float,
    E_A: float,
    C_B: float,
    E_B: float,
    tau: float,
    alternative: str,
) -> float:
    """Return log(Lambda_n) for Poisson rate ratio test via CLT approximation.

    Applies the Gaussian mSPRT to theta_hat = log(lambda_B_hat / lambda_A_hat)
    using the delta-method variance sigma_sq = 1/C_A + 1/C_B.

    CLT requires C_A >= 5 and C_B >= 5. Returns 0.0 (Lambda = 1.0) otherwise.
    """
    if C_A < 5 or C_B < 5:
        return 0.0

    theta_hat = math.log(C_B / E_B) - math.log(C_A / E_A)

    if abs(theta_hat) > math.log(100):
        warnings.warn(
            f"Rate ratio {math.exp(theta_hat):.1f}x is > 100x. Likely a data error.",
            UserWarning,
            stacklevel=4,
        )

    sigma_sq = 1.0 / C_A + 1.0 / C_B
    return _gaussian_msprt(theta_hat, sigma_sq, tau ** 2, alternative)


def _lognormal_msprt(
    n_A: float,
    sum_log_A: float,
    ss_log_A: float,
    n_B: float,
    sum_log_B: float,
    ss_log_B: float,
    tau: float,
    alternative: str,
) -> float:
    """Return log(Lambda_n) for log-normal severity ratio test.

    Applies the Gaussian mSPRT to the difference in log-means, using pooled
    variance from the log-cost observations.

    Requires n_A >= 10 and n_B >= 10. Returns 0.0 otherwise.
    """
    if n_A < 10 or n_B < 10:
        return 0.0

    mu_A = sum_log_A / n_A
    mu_B = sum_log_B / n_B
    theta_hat = mu_B - mu_A

    # Pooled within-group variance on log-scale
    numerator = ss_log_A + ss_log_B - n_A * mu_A ** 2 - n_B * mu_B ** 2
    denom = n_A + n_B - 2
    s_sq = numerator / denom if denom > 0 else 0.0
    sigma_sq = s_sq * (1.0 / n_A + 1.0 / n_B)

    return _gaussian_msprt(theta_hat, sigma_sq, tau ** 2, alternative)


def _confidence_sequence(
    C_A: float,
    E_A: float,
    C_B: float,
    E_B: float,
    alpha: float,
) -> tuple[float, float]:
    """Anytime-valid confidence sequence for the rate ratio (Howard et al. 2021).

    Returns the (1-alpha) time-uniform CI for the rate ratio challenger/champion.
    This CI is valid simultaneously for all interim looks, not just the final one.

    Returns (0.0, inf) when either arm has fewer than 5 claims.

    Derivation note
    ---------------
    With sigma_sq = 1/C_A + 1/C_B (variance of theta_hat) and
    v = C_A * C_B / (C_A + C_B) = 1/sigma_sq (intrinsic time), the
    Howard et al. normal-mixture CS half-width for theta_hat is:

        h = sqrt(2 * sigma_sq * log(sqrt(1 + v/rho) / alpha))

    Note: sigma_sq / v = sigma_sq^2 (since sigma_sq = 1/v), so the
    formula does NOT divide sigma_sq by v — that would give a much
    too-narrow interval. The factor 2 * sigma_sq gives the correct
    (1-alpha) coverage.
    """
    if C_A < 5 or C_B < 5:
        return (0.0, float("inf"))

    theta_hat = math.log(C_B / E_B) - math.log(C_A / E_A)
    sigma_sq = 1.0 / C_A + 1.0 / C_B

    # Intrinsic time: harmonic mean of claim counts = C_A * C_B / (C_A + C_B)
    v = 1.0 / sigma_sq

    # Tuning parameter: conservative for small v
    rho = max(5.0, 0.1 * v)

    # CS half-width on log scale (Howard et al. 2021, normal mixture bound)
    # sigma_sq is already the variance of theta_hat; no further division by v.
    h = math.sqrt(2.0 * sigma_sq * math.log(math.sqrt(1.0 + v / rho) / alpha))

    return (math.exp(theta_hat - h), math.exp(theta_hat + h))


def _bayesian_prob(
    C_A: float,
    E_A: float,
    C_B: float,
    E_B: float,
    alpha_prior: float = 2.0,
    beta_prior: float = 20.0,
) -> float:
    """P(lambda_B < lambda_A | data) from Gamma-Poisson conjugate model.

    Prior: lambda ~ Gamma(alpha_prior=2, beta_prior=20). Prior mean = 0.10, weak.
    Posterior: Gamma(alpha_prior + C, beta_prior + E) for each arm.

    Returns a probability in [0, 1]. Informational only — not used for stopping.
    """
    a1 = alpha_prior + C_A  # champion posterior shape
    b1 = beta_prior + E_A   # champion posterior rate
    a2 = alpha_prior + C_B  # challenger posterior shape
    b2 = beta_prior + E_B   # challenger posterior rate

    if C_A > 50 and C_B > 50:
        # Normal approximation: Gamma mean = a/b, var = a/b^2
        mu_diff = a2 / b2 - a1 / b1
        var_diff = a2 / b2 ** 2 + a1 / b1 ** 2
        return float(norm_dist.cdf(0.0, loc=mu_diff, scale=var_diff ** 0.5))

    # Exact: integrate P(X_B < x) * f_A(x) dx where X_A ~ Gamma(a1, 1/b1)
    from scipy.stats import gamma as gamma_dist

    dist_A = gamma_dist(a1, scale=1.0 / b1)
    dist_B = gamma_dist(a2, scale=1.0 / b2)

    upper = dist_A.ppf(0.9999)

    def integrand(x: float) -> float:
        return dist_B.cdf(x) * dist_A.pdf(x)

    result, _ = quad(integrand, 0.0, upper)
    return float(np.clip(result, 0.0, 1.0))


def _safe_lambda(log_lambda: float) -> float:
    """Convert log(Lambda) to Lambda with overflow guard."""
    if log_lambda > 500:
        return float("inf")
    if log_lambda == -math.inf:
        return 0.0
    return math.exp(log_lambda)


def _build_summary(
    result_decision: str,
    rate_ratio: float,
    ci_lower: float,
    ci_upper: float,
    lambda_value: float,
    threshold: float,
    metric: str,
) -> str:
    """Build a human-readable summary string."""
    pct_diff = (rate_ratio - 1.0) * 100.0
    direction = "lower" if pct_diff < 0 else "higher"
    metric_label = {"frequency": "freq", "severity": "sev", "loss_ratio": "LR"}[metric]

    if ci_lower == 0.0 and ci_upper == float("inf"):
        cs_str = "CS: insufficient data"
    else:
        cs_str = f"95% CS: {ci_lower:.3f}\u2013{ci_upper:.3f}"

    summary = (
        f"Challenger {metric_label} {abs(pct_diff):.1f}% {direction} ({cs_str}). "
        f"Evidence: {lambda_value:.1f} (threshold {threshold:.1f}). "
        f"{result_decision.replace('_', ' ').title()}."
    )
    return summary


# ---------------------------------------------------------------------------
# Main class (public)
# ---------------------------------------------------------------------------

class SequentialTest:
    """Anytime-valid sequential A/B test for insurance champion/challenger experiments.

    Uses the mixture Sequential Probability Ratio Test (mSPRT) of Johari et al.
    (2022). The e-process Lambda_n satisfies P_0(exists n: Lambda_n >= 1/alpha)
    <= alpha at ALL stopping times — no peeking penalty.

    Parameters
    ----------
    metric : str
        'frequency' (Poisson rate ratio), 'severity' (log-normal ratio),
        or 'loss_ratio' (product of frequency and severity e-values).
    method : str
        'msprt'. Reserved for future extension.
    alternative : str
        'two_sided' | 'greater' | 'less'. 'greater' means challenger has a
        higher rate than champion.
    alpha : float
        Type I error level. Decision threshold is 1/alpha.
    tau : float
        mSPRT prior std dev on the log-rate-ratio effect size. Default 0.03
        (prior expects ~3% effects). See spec Section 10.6 for calibration.
    max_duration_years : float
        Hard stop after this many years of experiment time. Declared as
        'max_duration_reached' regardless of Lambda_n.
    futility_threshold : float | None
        If Lambda_n falls below this value after min_exposure_per_arm is
        reached, declare 'futility'. Default None (disabled).
    min_exposure_per_arm : float
        Car-years per arm accumulated before any stopping decision is made.
        Prevents premature stops on early noisy data.
    """

    def __init__(
        self,
        metric: str = "frequency",
        method: str = "msprt",
        alternative: str = "two_sided",
        alpha: float = 0.05,
        tau: float = 0.03,
        max_duration_years: float = 2.0,
        futility_threshold: Optional[float] = None,
        min_exposure_per_arm: float = 100.0,
    ) -> None:
        if metric not in ("frequency", "severity", "loss_ratio"):
            raise ValueError(f"metric must be 'frequency', 'severity', or 'loss_ratio', got {metric!r}")
        if alternative not in ("two_sided", "greater", "less"):
            raise ValueError(f"alternative must be 'two_sided', 'greater', or 'less', got {alternative!r}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.metric = metric
        self.method = method
        self.alternative = alternative
        self.alpha = alpha
        self.tau = tau
        self.max_duration_years = max_duration_years
        self.futility_threshold = futility_threshold
        self.min_exposure_per_arm = min_exposure_per_arm

        self._threshold = 1.0 / alpha
        self._history: list[dict] = []
        self._first_date: Optional[datetime.date] = None
        self._last_date: Optional[datetime.date] = None
        self._n_updates: int = 0

        self._freq = _FreqState()
        self._sev = _SevState()

    def reset(self) -> None:
        """Clear accumulated state. Allows reuse of a SequentialTest in simulation loops."""
        self._history = []
        self._first_date = None
        self._last_date = None
        self._n_updates = 0
        self._freq = _FreqState()
        self._sev = _SevState()

    def update(
        self,
        champion_claims: float,
        champion_exposure: float,
        challenger_claims: float,
        challenger_exposure: float,
        calendar_date: Optional[datetime.date] = None,
        champion_severity_sum: Optional[float] = None,
        champion_severity_ss: Optional[float] = None,
        challenger_severity_sum: Optional[float] = None,
        challenger_severity_ss: Optional[float] = None,
    ) -> SequentialTestResult:
        """Process a new reporting period and return the current test result.

        All parameters are *increments* since the last update() call.
        Accumulated state is maintained internally.

        Parameters
        ----------
        champion_claims : float
            New claims in champion arm since last update.
        champion_exposure : float
            New exposure (car-years) in champion arm since last update.
        challenger_claims : float
            New claims in challenger arm since last update.
        challenger_exposure : float
            New exposure (car-years) in challenger arm since last update.
        calendar_date : datetime.date | None
            Optional date of this reporting period end. Used for time tracking.
        champion_severity_sum : float | None
            Sum of log(claim_cost) for new champion claims. Required for
            metric='severity' or 'loss_ratio'.
        champion_severity_ss : float | None
            Sum of log(claim_cost)^2 for new champion claims.
        challenger_severity_sum : float | None
            Sum of log(claim_cost) for new challenger claims.
        challenger_severity_ss : float | None
            Sum of log(claim_cost)^2 for new challenger claims.
        """
        # ---- Input validation ----
        if champion_claims < 0 or challenger_claims < 0:
            raise ValueError("Claims cannot be negative")
        if champion_exposure < 0 or challenger_exposure < 0:
            raise ValueError("Exposure cannot be negative")

        # ---- Accumulate frequency state ----
        self._freq.C_A += champion_claims
        self._freq.E_A += champion_exposure
        self._freq.C_B += challenger_claims
        self._freq.E_B += challenger_exposure

        # ---- Accumulate severity state ----
        if self.metric in ("severity", "loss_ratio"):
            if champion_severity_sum is not None:
                self._sev.n_A += champion_claims
                self._sev.sum_log_A += champion_severity_sum
                self._sev.ss_log_A += champion_severity_ss or 0.0
            if challenger_severity_sum is not None:
                self._sev.n_B += challenger_claims
                self._sev.sum_log_B += challenger_severity_sum
                self._sev.ss_log_B += challenger_severity_ss or 0.0

        # ---- Validate positive exposure totals ----
        if self._freq.E_A < 0 or self._freq.E_B < 0:
            raise ValueError("Exposure must be non-negative")

        # ---- Update time tracking ----
        if calendar_date is not None:
            if self._first_date is None:
                self._first_date = calendar_date
            self._last_date = calendar_date

        total_calendar_days = 0.0
        if self._first_date is not None and self._last_date is not None:
            total_calendar_days = float(
                (self._last_date - self._first_date).days
            )

        self._n_updates += 1

        # ---- Compute test statistics ----
        C_A = self._freq.C_A
        E_A = self._freq.E_A
        C_B = self._freq.C_B
        E_B = self._freq.E_B

        # Rates and ratio
        champ_rate = C_A / E_A if E_A > 0 else 0.0
        chall_rate = C_B / E_B if E_B > 0 else 0.0
        rate_ratio = chall_rate / champ_rate if champ_rate > 0 else 1.0

        # Confidence sequence
        ci_lower, ci_upper = _confidence_sequence(C_A, E_A, C_B, E_B, self.alpha)

        # mSPRT log-lambda
        if self.metric == "frequency":
            log_lam = _poisson_msprt(C_A, E_A, C_B, E_B, self.tau, self.alternative)
        elif self.metric == "severity":
            log_lam = _lognormal_msprt(
                self._sev.n_A, self._sev.sum_log_A, self._sev.ss_log_A,
                self._sev.n_B, self._sev.sum_log_B, self._sev.ss_log_B,
                self.tau, self.alternative,
            )
        else:  # loss_ratio: e-value product
            log_lam_freq = _poisson_msprt(C_A, E_A, C_B, E_B, self.tau, self.alternative)
            log_lam_sev = _lognormal_msprt(
                self._sev.n_A, self._sev.sum_log_A, self._sev.ss_log_A,
                self._sev.n_B, self._sev.sum_log_B, self._sev.ss_log_B,
                self.tau, self.alternative,
            )
            log_lam = log_lam_freq + log_lam_sev

        lambda_value = _safe_lambda(log_lam)

        # Bayesian secondary display
        prob_better = _bayesian_prob(C_A, E_A, C_B, E_B)

        # ---- Determine stopping decision ----
        # Check max duration
        elapsed_years = total_calendar_days / 365.25
        if calendar_date is not None and elapsed_years >= self.max_duration_years:
            decision = "max_duration_reached"
        # Check rejection threshold (only after minimum exposure)
        elif (
            E_A >= self.min_exposure_per_arm
            and E_B >= self.min_exposure_per_arm
            and (lambda_value >= self._threshold or math.isinf(lambda_value))
        ):
            decision = "reject_H0"
        # Check futility (only after minimum exposure)
        elif (
            self.futility_threshold is not None
            and E_A >= self.min_exposure_per_arm
            and E_B >= self.min_exposure_per_arm
            and lambda_value < self.futility_threshold
        ):
            decision = "futility"
        else:
            decision = "inconclusive"

        should_stop = decision != "inconclusive"

        summary = _build_summary(
            decision, rate_ratio, ci_lower, ci_upper,
            lambda_value, self._threshold, self.metric,
        )

        result = SequentialTestResult(
            decision=decision,
            should_stop=should_stop,
            lambda_value=lambda_value,
            log_lambda_value=log_lam if not math.isinf(log_lam) else float("inf"),
            threshold=self._threshold,
            champion_rate=champ_rate,
            challenger_rate=chall_rate,
            rate_ratio=rate_ratio,
            rate_ratio_ci_lower=ci_lower,
            rate_ratio_ci_upper=ci_upper,
            total_champion_claims=C_A,
            total_champion_exposure=E_A,
            total_challenger_claims=C_B,
            total_challenger_exposure=E_B,
            n_updates=self._n_updates,
            total_calendar_time_days=total_calendar_days,
            prob_challenger_better=prob_better,
            summary=summary,
        )

        # ---- Record history ----
        self._history.append({
            "period_index": self._n_updates,
            "calendar_date": calendar_date,
            "lambda_value": lambda_value,
            "log_lambda_value": result.log_lambda_value,
            "champion_rate": champ_rate,
            "challenger_rate": chall_rate,
            "rate_ratio": rate_ratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "decision": decision,
            "cum_champion_exposure": E_A,
            "cum_challenger_exposure": E_B,
            "cum_champion_claims": C_A,
            "cum_challenger_claims": C_B,
        })

        return result

    def history(self) -> pl.DataFrame:
        """Return one row per update() call as a Polars DataFrame.

        Columns: period_index, calendar_date, lambda_value, log_lambda_value,
        champion_rate, challenger_rate, rate_ratio, ci_lower, ci_upper,
        decision, cum_champion_exposure, cum_challenger_exposure,
        cum_champion_claims, cum_challenger_claims.
        """
        if not self._history:
            return pl.DataFrame(
                schema={
                    "period_index": pl.Int64,
                    "calendar_date": pl.Date,
                    "lambda_value": pl.Float64,
                    "log_lambda_value": pl.Float64,
                    "champion_rate": pl.Float64,
                    "challenger_rate": pl.Float64,
                    "rate_ratio": pl.Float64,
                    "ci_lower": pl.Float64,
                    "ci_upper": pl.Float64,
                    "decision": pl.String,
                    "cum_champion_exposure": pl.Float64,
                    "cum_challenger_exposure": pl.Float64,
                    "cum_champion_claims": pl.Float64,
                    "cum_challenger_claims": pl.Float64,
                }
            )
        return pl.DataFrame(self._history)


# ---------------------------------------------------------------------------
# Convenience function (public)
# ---------------------------------------------------------------------------

def sequential_test_from_df(
    df: pl.DataFrame,
    champion_claims_col: str,
    champion_exposure_col: str,
    challenger_claims_col: str,
    challenger_exposure_col: str,
    date_col: Optional[str] = None,
    champion_severity_sum_col: Optional[str] = None,
    champion_severity_ss_col: Optional[str] = None,
    challenger_severity_sum_col: Optional[str] = None,
    challenger_severity_ss_col: Optional[str] = None,
    metric: str = "frequency",
    alpha: float = 0.05,
    tau: float = 0.03,
    max_duration_years: float = 2.0,
    alternative: str = "two_sided",
    min_exposure_per_arm: float = 100.0,
    futility_threshold: Optional[float] = None,
) -> SequentialTestResult:
    """Run a sequential test over a Polars DataFrame of reporting periods.

    Each row in `df` represents one reporting period (month, quarter, etc.).
    Rows are processed in order, feeding each row as an update() call.

    Parameters
    ----------
    df : pl.DataFrame
        One row per reporting period, ordered chronologically.
    champion_claims_col : str
        Column name for new champion claims per period.
    champion_exposure_col : str
        Column name for new champion exposure (car-years) per period.
    challenger_claims_col : str
        Column name for new challenger claims per period.
    challenger_exposure_col : str
        Column name for new challenger exposure (car-years) per period.
    date_col : str | None
        Optional column name for period-end date (date or string).

    Returns
    -------
    SequentialTestResult
        Result from the final update() call (full history accessible via
        test.history() if you construct SequentialTest directly).
    """
    test = SequentialTest(
        metric=metric,
        alpha=alpha,
        tau=tau,
        max_duration_years=max_duration_years,
        alternative=alternative,
        min_exposure_per_arm=min_exposure_per_arm,
        futility_threshold=futility_threshold,
    )

    result: Optional[SequentialTestResult] = None

    for row in df.iter_rows(named=True):
        date_val = None
        if date_col is not None:
            raw = row[date_col]
            if isinstance(raw, datetime.date):
                date_val = raw
            elif raw is not None:
                date_val = datetime.date.fromisoformat(str(raw))

        result = test.update(
            champion_claims=float(row[champion_claims_col]),
            champion_exposure=float(row[champion_exposure_col]),
            challenger_claims=float(row[challenger_claims_col]),
            challenger_exposure=float(row[challenger_exposure_col]),
            calendar_date=date_val,
            champion_severity_sum=(
                float(row[champion_severity_sum_col])
                if champion_severity_sum_col and row.get(champion_severity_sum_col) is not None
                else None
            ),
            champion_severity_ss=(
                float(row[champion_severity_ss_col])
                if champion_severity_ss_col and row.get(champion_severity_ss_col) is not None
                else None
            ),
            challenger_severity_sum=(
                float(row[challenger_severity_sum_col])
                if challenger_severity_sum_col and row.get(challenger_severity_sum_col) is not None
                else None
            ),
            challenger_severity_ss=(
                float(row[challenger_severity_ss_col])
                if challenger_severity_ss_col and row.get(challenger_severity_ss_col) is not None
                else None
            ),
        )

    if result is None:
        raise ValueError("DataFrame is empty — no rows to process")

    return result
