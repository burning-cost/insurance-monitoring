"""
Additional tests for insurance_monitoring.sequential covering paths not
exercised by test_sequential.py.

Gaps addressed:
1. _gaussian_msprt with alternative='greater' and alternative='less'
   (one-sided mSPRT paths — only 'two_sided' is tested in the main suite)
2. _lognormal_msprt with insufficient data guard (n < 10 -> returns 0.0)
3. sequential_test_from_df with severity columns (loss_ratio metric path)
4. SequentialTest with alternative='greater' and 'less' — stopping behaviour
5. _bayesian_prob exact integration path (C_A <= 50, C_B <= 50 branch)

The one-sided tests matter most: UK actuaries running challenger tests
against a worse model (e.g. checking that a new telematics model has
strictly lower frequency, not just different) use alternative='less'.
The current suite only covers the null ('two_sided'). The one-sided paths
contain an explicit `if theta_hat >= 0: return -inf` branch that flips the
e-process to zero for effects in the wrong direction — untested behaviour
that, if wrong, would silently give a test with no power at all.
"""

from __future__ import annotations

import math
import datetime

import numpy as np
import polars as pl
import pytest

from insurance_monitoring.sequential import (
    SequentialTest,
    SequentialTestResult,
    sequential_test_from_df,
    _gaussian_msprt,
    _lognormal_msprt,
    _poisson_msprt,
)


# ---------------------------------------------------------------------------
# _gaussian_msprt: one-sided alternatives
# ---------------------------------------------------------------------------


class TestGaussianMsprtOneSided:
    """Tests for alternative='greater' and 'less' in the core formula."""

    def test_greater_positive_effect_positive_log_lambda(self):
        """
        alternative='greater': positive theta_hat (challenger > champion)
        should give a positive log_lambda (evidence for greater).
        """
        log_lam = _gaussian_msprt(
            theta_hat=0.2,   # large positive effect
            sigma_sq=0.01,
            tau_sq=0.04,
            alternative="greater",
        )
        assert log_lam > 0.0

    def test_greater_negative_effect_returns_neg_inf(self):
        """
        alternative='greater': negative theta_hat (challenger < champion)
        should return -inf (zero evidence in the wrong direction).
        """
        log_lam = _gaussian_msprt(
            theta_hat=-0.2,  # challenger is worse
            sigma_sq=0.01,
            tau_sq=0.04,
            alternative="greater",
        )
        assert log_lam == -math.inf

    def test_less_negative_effect_positive_log_lambda(self):
        """
        alternative='less': negative theta_hat (challenger < champion, i.e. better)
        should give a positive log_lambda.
        """
        log_lam = _gaussian_msprt(
            theta_hat=-0.2,  # challenger is better (fewer claims)
            sigma_sq=0.01,
            tau_sq=0.04,
            alternative="less",
        )
        assert log_lam > 0.0

    def test_less_positive_effect_returns_neg_inf(self):
        """
        alternative='less': positive theta_hat (challenger > champion)
        should return -inf.
        """
        log_lam = _gaussian_msprt(
            theta_hat=0.2,
            sigma_sq=0.01,
            tau_sq=0.04,
            alternative="less",
        )
        assert log_lam == -math.inf

    def test_greater_and_less_sum_to_two_sided(self):
        """
        For any positive theta_hat:
          log_lam_greater + log(0.5) = log_lam_two_sided
        (since the half-normal prior doubles the density on one side)
        The two-sided test uses the full normal; each one-sided uses a half-normal
        that adds log(2). So for a positive theta_hat:
          lam_greater = 2 * lam_two_sided
          log_lam_greater = log(2) + log_lam_two_sided
        """
        theta_hat, sigma_sq, tau_sq = 0.15, 0.05, 0.04
        log_lam_ts = _gaussian_msprt(theta_hat, sigma_sq, tau_sq, "two_sided")
        log_lam_gt = _gaussian_msprt(theta_hat, sigma_sq, tau_sq, "greater")
        assert abs(log_lam_gt - (log_lam_ts + math.log(2))) < 1e-12

    def test_tiny_sigma_sq_returns_zero(self):
        """sigma_sq < 1e-12 returns 0.0 (degenerate case guard)."""
        assert _gaussian_msprt(0.1, sigma_sq=1e-15, tau_sq=0.04, alternative="two_sided") == 0.0


# ---------------------------------------------------------------------------
# _lognormal_msprt: insufficient data guard
# ---------------------------------------------------------------------------


class TestLognormalMsprtDataGuard:
    """Tests for the n < 10 guard in _lognormal_msprt."""

    def test_too_few_claims_returns_zero(self):
        """n_A < 10 or n_B < 10 -> log_lambda = 0.0 (no evidence)."""
        log_lam = _lognormal_msprt(
            n_A=5, sum_log_A=35.0, ss_log_A=250.0,
            n_B=5, sum_log_B=37.0, ss_log_B=275.0,
            tau=0.05, alternative="two_sided",
        )
        assert log_lam == 0.0

    def test_exactly_ten_claims_proceeds(self):
        """n_A == 10 and n_B == 10 is the boundary — should compute (not return 0)."""
        log_lam = _lognormal_msprt(
            n_A=10, sum_log_A=70.0, ss_log_A=500.0,
            n_B=10, sum_log_B=72.0, ss_log_B=530.0,
            tau=0.05, alternative="two_sided",
        )
        # Not guaranteed to be non-zero (prior shrinkage), but must be computable
        assert isinstance(log_lam, float)
        assert not math.isinf(log_lam) or log_lam == -math.inf  # -inf allowed for one-sided


# ---------------------------------------------------------------------------
# SequentialTest: one-sided alternatives end-to-end
# ---------------------------------------------------------------------------


class TestSequentialTestOneSided:
    """End-to-end tests for SequentialTest with one-sided alternatives."""

    def test_alternative_less_rejects_when_challenger_better(self):
        """
        alternative='less': accumulate data where challenger has clearly fewer
        claims than champion. Should eventually reject H0 (reject_H0 decision).
        """
        rng = np.random.default_rng(42)
        test = SequentialTest(
            metric="frequency",
            alternative="less",
            alpha=0.05,
            tau=0.10,
            min_exposure_per_arm=0.0,
        )
        # Champion: 0.10 claims/car-year; Challenger: 0.065 (35% better)
        rejected = False
        for _ in range(60):
            ca = int(rng.poisson(0.10 * 500))
            cb = int(rng.poisson(0.065 * 500))
            result = test.update(ca, 500.0, cb, 500.0)
            if result.decision == "reject_H0":
                rejected = True
                break

        assert rejected, (
            "alternative='less' should reject when challenger has 35% fewer claims"
        )

    def test_alternative_greater_does_not_reject_when_challenger_better(self):
        """
        alternative='greater' means H1: challenger rate > champion rate.
        When challenger is better (lower rate), this alternative should NOT reject.
        """
        rng = np.random.default_rng(99)
        test = SequentialTest(
            metric="frequency",
            alternative="greater",
            alpha=0.05,
            tau=0.10,
            min_exposure_per_arm=0.0,
        )
        # Challenger clearly better -> lambda_value = 0 for 'greater' alternative
        for _ in range(30):
            ca = int(rng.poisson(0.10 * 500))
            cb = int(rng.poisson(0.065 * 500))
            result = test.update(ca, 500.0, cb, 500.0)

        # The challenger has lower rate -> 'greater' alternative finds no evidence
        assert result.lambda_value < test._threshold, (
            "alternative='greater' should not reject when challenger has lower rate"
        )

    def test_alternative_less_lambda_direction(self):
        """
        After accumulating data where challenger is better (lower rate),
        alternative='less' should give a higher lambda than alternative='greater'.
        """
        rng = np.random.default_rng(7)
        test_less = SequentialTest(
            metric="frequency",
            alternative="less",
            tau=0.10,
            min_exposure_per_arm=0.0,
        )
        test_greater = SequentialTest(
            metric="frequency",
            alternative="greater",
            tau=0.10,
            min_exposure_per_arm=0.0,
        )
        # Feed the same data to both
        for _ in range(20):
            ca = int(rng.poisson(0.10 * 200))
            cb = int(rng.poisson(0.07 * 200))  # challenger clearly better
            test_less.update(ca, 200.0, cb, 200.0)
            test_greater.update(ca, 200.0, cb, 200.0)

        result_less = test_less.update(0, 0.0, 0, 0.0)
        result_greater = test_greater.update(0, 0.0, 0, 0.0)

        # 'less' alternative accumulates evidence; 'greater' gives near-zero lambda
        assert result_less.lambda_value > result_greater.lambda_value

    def test_invalid_alpha(self):
        """alpha outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            SequentialTest(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            SequentialTest(alpha=1.0)

    def test_invalid_tau(self):
        """tau <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="tau"):
            SequentialTest(tau=0.0)
        with pytest.raises(ValueError, match="tau"):
            SequentialTest(tau=-0.01)


# ---------------------------------------------------------------------------
# sequential_test_from_df: loss_ratio metric with severity columns
# ---------------------------------------------------------------------------


class TestSequentialTestFromDfSeverity:
    """Tests for sequential_test_from_df with severity-related columns."""

    def _make_severity_df(self, n_periods=12, seed=0):
        """Build a DataFrame with log-severity sufficient statistics per period."""
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_periods):
            n_claims_a = int(rng.poisson(25))
            n_claims_b = int(rng.poisson(22))
            # Log-costs: champion mean ~7.0, challenger slightly lower ~6.9
            log_costs_a = rng.normal(7.0, 0.5, max(n_claims_a, 1))
            log_costs_b = rng.normal(6.9, 0.5, max(n_claims_b, 1))
            rows.append({
                "champ_claims": float(n_claims_a),
                "champ_exposure": 200.0,
                "chall_claims": float(n_claims_b),
                "chall_exposure": 200.0,
                "champ_sev_sum": float(log_costs_a.sum()),
                "champ_sev_ss": float((log_costs_a ** 2).sum()),
                "chall_sev_sum": float(log_costs_b.sum()),
                "chall_sev_ss": float((log_costs_b ** 2).sum()),
            })
        return pl.DataFrame(rows)

    def test_loss_ratio_metric_runs(self):
        """loss_ratio metric with severity columns should run without error."""
        df = self._make_severity_df()
        result = sequential_test_from_df(
            df,
            champion_claims_col="champ_claims",
            champion_exposure_col="champ_exposure",
            challenger_claims_col="chall_claims",
            challenger_exposure_col="chall_exposure",
            champion_severity_sum_col="champ_sev_sum",
            champion_severity_ss_col="champ_sev_ss",
            challenger_severity_sum_col="chall_sev_sum",
            challenger_severity_ss_col="chall_sev_ss",
            metric="loss_ratio",
            min_exposure_per_arm=0.0,
        )
        assert isinstance(result, SequentialTestResult)
        assert result.n_updates == 12

    def test_loss_ratio_lambda_equals_product(self):
        """
        loss_ratio lambda = freq_lambda * sev_lambda.

        This is identical to test_8h in the main suite but exercises
        sequential_test_from_df (rather than direct update() calls),
        ensuring the severity column plumbing is correct.
        """
        df = self._make_severity_df(n_periods=1, seed=3)

        result_lr = sequential_test_from_df(
            df,
            champion_claims_col="champ_claims",
            champion_exposure_col="champ_exposure",
            challenger_claims_col="chall_claims",
            challenger_exposure_col="chall_exposure",
            champion_severity_sum_col="champ_sev_sum",
            champion_severity_ss_col="champ_sev_ss",
            challenger_severity_sum_col="chall_sev_sum",
            challenger_severity_ss_col="chall_sev_ss",
            metric="loss_ratio",
            min_exposure_per_arm=0.0,
        )

        result_freq = sequential_test_from_df(
            df,
            champion_claims_col="champ_claims",
            champion_exposure_col="champ_exposure",
            challenger_claims_col="chall_claims",
            challenger_exposure_col="chall_exposure",
            metric="frequency",
            min_exposure_per_arm=0.0,
        )

        result_sev = sequential_test_from_df(
            df,
            champion_claims_col="champ_claims",
            champion_exposure_col="champ_exposure",
            challenger_claims_col="chall_claims",
            challenger_exposure_col="chall_exposure",
            champion_severity_sum_col="champ_sev_sum",
            champion_severity_ss_col="champ_sev_ss",
            challenger_severity_sum_col="chall_sev_sum",
            challenger_severity_ss_col="chall_sev_ss",
            metric="severity",
            min_exposure_per_arm=0.0,
        )

        expected_log = result_freq.log_lambda_value + result_sev.log_lambda_value
        assert abs(result_lr.log_lambda_value - expected_log) < 1e-10

    def test_empty_df_raises(self):
        """Empty DataFrame should raise ValueError."""
        df = pl.DataFrame({
            "cc": pl.Series([], dtype=pl.Float64),
            "ce": pl.Series([], dtype=pl.Float64),
            "tc": pl.Series([], dtype=pl.Float64),
            "te": pl.Series([], dtype=pl.Float64),
        })
        with pytest.raises(ValueError, match="[Ee]mpty"):
            sequential_test_from_df(df, "cc", "ce", "tc", "te")


# ---------------------------------------------------------------------------
# _bayesian_prob: exact integration path (small claim counts)
# ---------------------------------------------------------------------------


class TestBayesianProbSmallCounts:
    """Tests for the exact Gamma-Poisson integration path (C_A, C_B <= 50)."""

    def test_small_counts_probability_valid(self):
        """Small counts (< 50) trigger the exact integration path; result must be in [0, 1]."""
        test = SequentialTest(metric="frequency")
        result = test.update(20, 200.0, 15, 200.0)
        assert 0.0 <= result.prob_challenger_better <= 1.0

    def test_small_counts_direction(self):
        """When challenger clearly has fewer claims, P(challenger better) > 0.5."""
        test = SequentialTest(metric="frequency")
        # 30 claims vs 10 claims at same exposure -> challenger much better
        result = test.update(30, 200.0, 10, 200.0)
        assert result.prob_challenger_better > 0.5

    def test_exact_path_vs_normal_approximation_consistent(self):
        """
        At the boundary (around 50 claims), both the exact and normal paths
        should give similar results.

        We test this by comparing 49 vs 51 claims — should not discontinuously jump.
        """
        test_exact = SequentialTest(metric="frequency")
        test_normal = SequentialTest(metric="frequency")

        r_exact = test_exact.update(49, 490.0, 35, 490.0)
        r_normal = test_normal.update(51, 510.0, 37, 510.0)

        # Both should put P(challenger better) > 0.5 (challenger has lower rate ~0.071 vs 0.10)
        assert r_exact.prob_challenger_better > 0.5
        assert r_normal.prob_challenger_better > 0.5
        # And they should be in the same ballpark (within 0.20)
        assert abs(r_exact.prob_challenger_better - r_normal.prob_challenger_better) < 0.20
