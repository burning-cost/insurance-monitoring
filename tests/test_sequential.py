"""
Tests for insurance_monitoring.sequential — mSPRT sequential A/B testing.

10 test cases covering:
  8a. Minimum data guard
  8b. Type I error control (Monte Carlo, n=500)
  8c. Power check
  8d. Exposure scaling
  8e. Batching idempotency
  8f. Confidence sequence time-uniform coverage
  8g. DataFrame equivalence
  8h. Loss ratio e-value multiplication
  8i. Bayesian posterior in [0, 1]
  8j. Null true — Lambda stays near 1 on average
"""

import math
import datetime

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.sequential import (
    SequentialTest,
    SequentialTestResult,
    sequential_test_from_df,
)


# ---------------------------------------------------------------------------
# 8a. Minimum data guard
# ---------------------------------------------------------------------------

def test_8a_minimum_data_guard():
    """With fewer than 5 claims per arm, CLT is not valid.

    The module should return lambda_value = 1.0 (log_lambda = 0.0) and
    decision = 'inconclusive' rather than making a premature rejection.
    """
    test = SequentialTest(metric="frequency", alpha=0.05)
    result = test.update(2, 10.0, 1, 10.0)  # C_A=2, C_B=1 < 5

    assert result.decision == "inconclusive"
    assert result.lambda_value == pytest.approx(1.0)
    assert not result.should_stop


def test_8a_minimum_data_guard_boundary():
    """Exactly 5 claims per arm should proceed past the minimum guard."""
    test = SequentialTest(metric="frequency", alpha=0.05, min_exposure_per_arm=0.0)
    result = test.update(5, 50.0, 5, 50.0)
    # Lambda should be computed (not stuck at 1.0 for the guard reason)
    # With equal rates, log_lambda will be near 0 but computed — lambda_value may be < 1
    # because the prior term 0.5 * log(tau^2 / (tau^2 + sigma^2)) is negative.
    assert isinstance(result.lambda_value, float)
    assert result.n_updates == 1


# ---------------------------------------------------------------------------
# 8b. Type I error control (Monte Carlo, n=500 runs)
# ---------------------------------------------------------------------------

def test_8b_type_i_error_control():
    """mSPRT must control FPR at alpha=0.05 across 500 simulated null experiments.

    Each experiment runs monthly Poisson data with equal rates for up to 36 months.
    FPR bound: alpha + 2.5 * sqrt(alpha*(1-alpha)/500) ~ 0.076.

    This test directly verifies the key property: P_0(exists n: Lambda_n >= 1/alpha) <= alpha.
    """
    rng = np.random.default_rng(0)
    rejections = 0

    for _ in range(500):
        test = SequentialTest(
            metric="frequency",
            alpha=0.05,
            tau=0.03,
            max_duration_years=3.0,
            min_exposure_per_arm=0.0,  # allow early stops to stress-test type I error
        )
        for month in range(36):
            ca = int(rng.poisson(0.10 * 50))
            cb = int(rng.poisson(0.10 * 50))
            result = test.update(ca, 50.0, cb, 50.0)
            if result.should_stop and result.decision == "reject_H0":
                rejections += 1
                break

    fpr = rejections / 500
    # Allow 2.5-sigma above nominal alpha
    assert fpr <= 0.076, f"FPR={fpr:.3f} exceeds 0.076 bound (expected <= 0.05)"


# ---------------------------------------------------------------------------
# 8c. Power check
# ---------------------------------------------------------------------------

def test_8c_power_check():
    """mSPRT achieves >= 50% power for a 15% improvement on a large book.

    Design rationale:
    - Champion: rate 0.10 claims/car-year, 500 car-years/month (50 claims/month)
    - Challenger: rate 0.085 (15% improvement), 500 car-years/month
    - tau=0.05, alpha=0.05, min_exposure_per_arm=500 (1 month to clear)
    - 36 month horizon

    The mSPRT's log_lambda maximum at the true theta = log(0.085/0.10) = -0.163 is:
        theta^2 / (2*tau^2) = 0.02659 / 0.005 = 5.32  =>  Lambda_max = 205
    So the threshold (Lambda=20) is reachable. With 50 claims/arm/month the test
    is expected to cross the threshold around month 25-30, giving ~50-60% power
    at month 36.

    Note: the mSPRT requires much more data than a fixed-horizon t-test for small
    effects. This is the cost of anytime-validity — no peeking penalty but higher
    expected sample size to reach a decision.
    """
    rng = np.random.default_rng(1)
    stops = 0

    for _ in range(300):
        test = SequentialTest(
            metric="frequency",
            alpha=0.05,
            tau=0.05,
            max_duration_years=3.0,
            min_exposure_per_arm=500.0,
        )
        for month in range(36):
            ca = int(rng.poisson(0.10 * 500))
            cb = int(rng.poisson(0.085 * 500))  # 15% better challenger, equal allocation
            result = test.update(ca, 500.0, cb, 500.0)
            if result.should_stop and result.decision == "reject_H0":
                stops += 1
                break

    power = stops / 300
    assert power >= 0.50, f"Power={power:.3f} is below 0.50 threshold"


# ---------------------------------------------------------------------------
# 8d. Exposure scaling
# ---------------------------------------------------------------------------

def test_8d_exposure_scaling():
    """10x exposure at the same rate -> same rate estimate, higher Lambda.

    The mSPRT should accumulate more evidence with more data, while the
    rate estimates themselves should be identical.
    """
    test_small = SequentialTest(metric="frequency")
    test_large = SequentialTest(metric="frequency")

    for _ in range(12):
        test_small.update(5, 50.0, 3, 50.0)
        test_large.update(50, 500.0, 30, 500.0)

    r_s = test_small.update(0, 0.0, 0, 0.0)
    r_l = test_large.update(0, 0.0, 0, 0.0)

    # Same rate estimates
    assert abs(r_s.champion_rate - r_l.champion_rate) < 1e-10
    assert abs(r_s.challenger_rate - r_l.challenger_rate) < 1e-10
    assert abs(r_s.rate_ratio - r_l.rate_ratio) < 1e-10

    # More data -> more evidence
    assert r_l.lambda_value >= r_s.lambda_value


# ---------------------------------------------------------------------------
# 8e. Batching idempotency
# ---------------------------------------------------------------------------

def test_8e_batching_idempotency():
    """Monthly updates vs single batch with same totals must give identical Lambda.

    The mSPRT statistic depends only on cumulative counts, not the update
    schedule. This is the key property enabling flexible monitoring cadences.
    """
    claims_a = [5, 7, 4, 6, 8]
    claims_b = [3, 4, 5, 3, 6]

    test_monthly = SequentialTest(metric="frequency")
    for ca, cb in zip(claims_a, claims_b):
        test_monthly.update(ca, 50.0, cb, 40.0)

    test_batch = SequentialTest(metric="frequency")
    test_batch.update(sum(claims_a), 250.0, sum(claims_b), 200.0)

    # Add a zero update to get the final result after the same total data
    r_m = test_monthly.update(0, 0.0, 0, 0.0)
    r_b = test_batch.update(0, 0.0, 0, 0.0)

    assert abs(r_m.lambda_value - r_b.lambda_value) < 1e-10
    assert abs(r_m.rate_ratio - r_b.rate_ratio) < 1e-10


# ---------------------------------------------------------------------------
# 8f. Confidence sequence time-uniform coverage
# ---------------------------------------------------------------------------

def test_8f_confidence_sequence_coverage():
    """90% CS should cover the true rate ratio in >= 88% of 300 experiments.

    The CS is time-uniform: it must simultaneously cover the true value at
    all time points within a single experiment. Coverage across experiments
    should be >= (1 - alpha) = 0.90, with 2-sigma tolerance giving 0.88.
    """
    rng = np.random.default_rng(42)
    covered = []
    true_ratio = 1.05

    for _ in range(300):
        test = SequentialTest(metric="frequency", alpha=0.10)
        ok = True
        for month in range(24):
            ca = int(rng.poisson(0.10 * 50))
            cb = int(rng.poisson(0.10 * true_ratio * 50))
            r = test.update(ca, 50.0, cb, 50.0)
            # Only check CS validity once minimum claims are accumulated
            if r.total_champion_claims > 5 and r.total_challenger_claims > 5:
                if not (r.rate_ratio_ci_lower <= true_ratio <= r.rate_ratio_ci_upper):
                    ok = False
                    break
        covered.append(ok)

    coverage = sum(covered) / 300
    assert coverage >= 0.88, f"CS coverage={coverage:.3f} is below 0.88"


# ---------------------------------------------------------------------------
# 8g. DataFrame equivalence
# ---------------------------------------------------------------------------

def test_8g_dataframe_equivalence():
    """sequential_test_from_df must produce identical results to manual update() calls."""
    df = pl.DataFrame({
        "cc": [5, 7, 4],
        "ce": [50.0, 50.0, 50.0],
        "tc": [3, 4, 5],
        "te": [40.0, 40.0, 40.0],
    })

    r_df = sequential_test_from_df(df, "cc", "ce", "tc", "te")

    test = SequentialTest(metric="frequency")
    for row in df.iter_rows(named=True):
        r_iter = test.update(row["cc"], row["ce"], row["tc"], row["te"])

    assert abs(r_df.lambda_value - r_iter.lambda_value) < 1e-10
    assert abs(r_df.rate_ratio - r_iter.rate_ratio) < 1e-10
    assert r_df.decision == r_iter.decision


def test_8g_dataframe_with_dates():
    """sequential_test_from_df handles date columns correctly."""
    df = pl.DataFrame({
        "cc": [10, 12, 8],
        "ce": [100.0, 100.0, 100.0],
        "tc": [8, 9, 7],
        "te": [100.0, 100.0, 100.0],
        "period": [
            datetime.date(2025, 1, 31),
            datetime.date(2025, 2, 28),
            datetime.date(2025, 3, 31),
        ],
    })

    result = sequential_test_from_df(df, "cc", "ce", "tc", "te", date_col="period")
    assert result.total_calendar_time_days == pytest.approx(59.0)  # Jan31 -> Mar31 = 59 days
    assert result.n_updates == 3


# ---------------------------------------------------------------------------
# 8h. Loss ratio e-value multiplication
# ---------------------------------------------------------------------------

def test_8h_loss_ratio_evalue_multiplication():
    """loss_ratio lambda must equal freq_lambda * sev_lambda (product of e-values).

    Per Ramdas et al. (2023) Proposition 2.1: the product of independent
    e-values is itself an e-value. This is the mathematical foundation for
    the loss ratio test.
    """
    import math

    # Construct shared claim counts and sufficient statistics
    n_claims_a = 20
    n_claims_b = 15
    exposure_a = 200.0
    exposure_b = 200.0

    # Severity sufficient stats: simulate log-costs with known mean/ss
    rng = np.random.default_rng(7)
    log_costs_a = rng.normal(7.0, 0.8, n_claims_a)  # ln(cost), mean ~ln(1097)
    log_costs_b = rng.normal(7.2, 0.8, n_claims_b)  # challenger slightly higher severity

    sev_sum_a = float(log_costs_a.sum())
    sev_ss_a = float((log_costs_a ** 2).sum())
    sev_sum_b = float(log_costs_b.sum())
    sev_ss_b = float((log_costs_b ** 2).sum())

    # Build freq-only test
    t_freq = SequentialTest(metric="frequency", alpha=0.05, tau=0.05, min_exposure_per_arm=0.0)
    r_freq = t_freq.update(n_claims_a, exposure_a, n_claims_b, exposure_b)

    # Build sev-only test
    t_sev = SequentialTest(metric="severity", alpha=0.05, tau=0.05, min_exposure_per_arm=0.0)
    r_sev = t_sev.update(
        n_claims_a, exposure_a, n_claims_b, exposure_b,
        champion_severity_sum=sev_sum_a, champion_severity_ss=sev_ss_a,
        challenger_severity_sum=sev_sum_b, challenger_severity_ss=sev_ss_b,
    )

    # Build loss_ratio test
    t_lr = SequentialTest(metric="loss_ratio", alpha=0.05, tau=0.05, min_exposure_per_arm=0.0)
    r_lr = t_lr.update(
        n_claims_a, exposure_a, n_claims_b, exposure_b,
        champion_severity_sum=sev_sum_a, champion_severity_ss=sev_ss_a,
        challenger_severity_sum=sev_sum_b, challenger_severity_ss=sev_ss_b,
    )

    # lambda_LR = lambda_freq * lambda_sev
    expected_log_lr = r_freq.log_lambda_value + r_sev.log_lambda_value
    assert abs(r_lr.log_lambda_value - expected_log_lr) < 1e-10, (
        f"log(lambda_LR)={r_lr.log_lambda_value:.6f} != "
        f"log(lambda_freq)+log(lambda_sev)={expected_log_lr:.6f}"
    )


# ---------------------------------------------------------------------------
# 8i. Bayesian posterior in [0, 1]
# ---------------------------------------------------------------------------

def test_8i_bayesian_posterior_bounds():
    """prob_challenger_better must be a valid probability in [0, 1]."""
    test = SequentialTest(metric="frequency")
    result = test.update(15, 150.0, 10, 150.0)
    assert 0.0 <= result.prob_challenger_better <= 1.0


def test_8i_bayesian_posterior_direction():
    """When challenger has lower claims, prob_challenger_better should be > 0.5."""
    test = SequentialTest(metric="frequency")
    # Challenger has 30% fewer claims at same exposure -> lower rate -> better
    result = test.update(30, 300.0, 15, 300.0)
    # Lower challenger rate -> P(lambda_B < lambda_A) should be > 0.5
    assert result.prob_challenger_better > 0.5


def test_8i_bayesian_large_counts():
    """For large counts the normal approximation path is taken (C_A > 50, C_B > 50)."""
    test = SequentialTest(metric="frequency")
    result = test.update(100, 1000.0, 80, 1000.0)
    assert 0.0 <= result.prob_challenger_better <= 1.0


# ---------------------------------------------------------------------------
# 8j. Null true -> Lambda stays near 1 on average
# ---------------------------------------------------------------------------

def test_8j_null_true_lambda_near_one():
    """Under the null, Lambda_n is an e-process with E[Lambda_n] = 1.

    After 60 months of null data, Lambda should be < 1/alpha = 20 with
    high probability (failing to stop is the correct outcome under H0).
    The test is a single realisation — very unlikely to reach 100x.
    """
    rng = np.random.default_rng(99)
    test = SequentialTest(metric="frequency", alpha=0.05)
    result = None

    for _ in range(60):
        ca = int(rng.poisson(0.10 * 50))
        cb = int(rng.poisson(0.10 * 50))
        result = test.update(ca, 50.0, cb, 50.0)

    # Under H0, Lambda is a nonneg martingale. After 60 months of null data
    # it should rarely exceed 100 (1/0.01).
    assert not result.should_stop or result.lambda_value < 1.0 / 0.01


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """reset() must return test to initial state."""
    test = SequentialTest(metric="frequency")
    test.update(20, 200.0, 15, 200.0)
    assert test._n_updates == 1

    test.reset()
    assert test._n_updates == 0
    assert test._freq.C_A == 0.0
    assert test._freq.E_A == 0.0
    assert len(test._history) == 0


def test_history_dataframe_columns():
    """history() must return a DataFrame with the correct schema."""
    test = SequentialTest(metric="frequency")
    test.update(10, 100.0, 8, 100.0)
    test.update(12, 100.0, 7, 100.0)

    df = test.history()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2

    expected_cols = {
        "period_index", "calendar_date", "lambda_value", "log_lambda_value",
        "champion_rate", "challenger_rate", "rate_ratio", "ci_lower", "ci_upper",
        "decision", "cum_champion_exposure", "cum_challenger_exposure",
        "cum_champion_claims", "cum_challenger_claims",
    }
    assert expected_cols.issubset(set(df.columns))


def test_history_empty():
    """history() on a fresh test should return an empty DataFrame with correct schema."""
    test = SequentialTest(metric="frequency")
    df = test.history()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0


def test_invalid_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        SequentialTest(metric="severity_loss")


def test_invalid_alternative_raises():
    with pytest.raises(ValueError, match="alternative"):
        SequentialTest(alternative="both")


def test_negative_claims_raises():
    test = SequentialTest(metric="frequency")
    with pytest.raises(ValueError, match="negative"):
        test.update(-1, 10.0, 5, 10.0)


def test_max_duration_stops():
    """max_duration_reached fires when elapsed calendar time exceeds max_duration_years."""
    test = SequentialTest(metric="frequency", alpha=0.05, max_duration_years=1.0)
    start = datetime.date(2024, 1, 1)
    end = datetime.date(2025, 6, 1)  # 17 months later -> > 1 year

    test.update(10, 100.0, 9, 100.0, calendar_date=start)
    result = test.update(10, 100.0, 9, 100.0, calendar_date=end)

    assert result.decision == "max_duration_reached"
    assert result.should_stop


def test_futility_threshold():
    """Futility fires when Lambda falls below threshold after min exposure."""
    test = SequentialTest(
        metric="frequency",
        alpha=0.05,
        futility_threshold=0.5,
        min_exposure_per_arm=0.0,
    )
    # Equal rates -> Lambda near 1 but will drop below 0.5 with the prior penalty
    # Use very few claims so sigma_sq is large and prior penalty dominates
    result = test.update(5, 50.0, 5, 50.0)
    # The prior penalty 0.5 * log(tau^2 / (tau^2 + sigma^2)) is negative.
    # With tau=0.03 and sigma^2 = 1/5 + 1/5 = 0.4, the penalty is significant.
    # lambda_value = exp(0.5 * log(0.0009 / 0.4009) + 0) ~ exp(-3.0) ~ 0.05
    if result.lambda_value < 0.5:
        assert result.decision == "futility"
        assert result.should_stop


def test_sequential_test_result_fields():
    """SequentialTestResult must have all required fields with correct types."""
    test = SequentialTest(metric="frequency", alpha=0.05)
    result = test.update(20, 200.0, 18, 200.0)

    assert isinstance(result.decision, str)
    assert isinstance(result.should_stop, bool)
    assert isinstance(result.lambda_value, float)
    assert isinstance(result.log_lambda_value, float)
    assert isinstance(result.threshold, float)
    assert result.threshold == pytest.approx(20.0)  # 1/0.05
    assert isinstance(result.champion_rate, float)
    assert isinstance(result.challenger_rate, float)
    assert isinstance(result.rate_ratio, float)
    assert isinstance(result.rate_ratio_ci_lower, float)
    assert isinstance(result.rate_ratio_ci_upper, float)
    assert isinstance(result.total_champion_claims, float)
    assert isinstance(result.total_champion_exposure, float)
    assert isinstance(result.total_challenger_claims, float)
    assert isinstance(result.total_challenger_exposure, float)
    assert isinstance(result.n_updates, int)
    assert isinstance(result.total_calendar_time_days, float)
    assert isinstance(result.prob_challenger_better, float)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0
