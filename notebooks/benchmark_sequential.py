# Databricks notebook source
# MAGIC %md
# MAGIC # Sequential Testing Benchmark — insurance-monitoring v0.5.0
# MAGIC
# MAGIC **Burning Cost | March 2026**
# MAGIC
# MAGIC This notebook demonstrates the core problem with fixed-horizon t-tests in
# MAGIC insurance champion/challenger experiments, and shows how the mSPRT from
# MAGIC `insurance-monitoring` solves it.
# MAGIC
# MAGIC **What we test:**
# MAGIC - 500 experiments under H0 (same rate) — FPR comparison
# MAGIC - 500 experiments under H1 (challenger 5% better) — power and stopping time
# MAGIC
# MAGIC **Expected output:**
# MAGIC - t-test with monthly peeking: FPR ~15-20% (vs nominal 5%)
# MAGIC - mSPRT with monthly checking: FPR ~5% (controlled)
# MAGIC - mSPRT stops earlier when effect is real

# COMMAND ----------

# MAGIC %pip install insurance-monitoring>=0.5.0

# COMMAND ----------

import math
import numpy as np
from scipy import stats
from insurance_monitoring.sequential import SequentialTest
import insurance_monitoring

print(f"insurance-monitoring version: {insurance_monitoring.__version__}")
print("Imports successful.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: simulation parameters

# COMMAND ----------

N_SIMS = 500
RATE_A = 0.10          # Champion: 10 claims per 100 car-years
RATE_B_H0 = 0.10       # H0: equal rates
RATE_B_H1 = 0.095      # H1: challenger 5% better
EXPOSURE_PER_MONTH = 50.0   # car-years per arm per month
N_MONTHS = 36          # max 3 years
ALPHA = 0.05

print(f"Simulations per scenario: {N_SIMS}")
print(f"Champion rate: {RATE_A} claims/car-year")
print(f"H0 challenger rate: {RATE_B_H0} (no effect)")
print(f"H1 challenger rate: {RATE_B_H1} ({(1 - RATE_B_H1/RATE_A)*100:.0f}% improvement)")
print(f"Monthly exposure per arm: {EXPOSURE_PER_MONTH} car-years")
print(f"Max follow-up: {N_MONTHS} months ({N_MONTHS//12} years)")
print(f"Alpha: {ALPHA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def run_fixed_horizon_ttest(rng, rate_a, rate_b, exposure_per_month, n_months, alpha=0.05):
    """Normal-approximation rate comparison checked monthly — invalid peeking."""
    cum_ca, cum_cb = 0, 0
    cum_ea = cum_eb = 0.0
    for month in range(n_months):
        ca = int(rng.poisson(rate_a * exposure_per_month))
        cb = int(rng.poisson(rate_b * exposure_per_month))
        cum_ca += ca; cum_cb += cb
        cum_ea += exposure_per_month; cum_eb += exposure_per_month
        if cum_ca < 5 or cum_cb < 5:
            continue
        lam_a = cum_ca / cum_ea
        lam_b = cum_cb / cum_eb
        se = math.sqrt(lam_a / cum_ea + lam_b / cum_eb)
        if se == 0:
            continue
        z = (lam_b - lam_a) / se
        if 2 * stats.norm.sf(abs(z)) < alpha:
            return True, month + 1
    return False, None


def run_msprt(rng, rate_a, rate_b, exposure_per_month, n_months, alpha=0.05, tau=0.03):
    """mSPRT sequential test checked monthly — valid anytime."""
    test = SequentialTest(
        metric="frequency", alpha=alpha, tau=tau,
        max_duration_years=n_months / 12, min_exposure_per_arm=0.0,
    )
    for month in range(n_months):
        ca = int(rng.poisson(rate_a * exposure_per_month))
        cb = int(rng.poisson(rate_b * exposure_per_month))
        result = test.update(ca, exposure_per_month, cb, exposure_per_month)
        if result.should_stop and result.decision == "reject_H0":
            return True, month + 1
    return False, None

print("Helper functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 1: H0 (null true) — False Positive Rate

# COMMAND ----------

print(f"Running {N_SIMS} experiments under H0 (equal rates)...")
print("This tests type I error control — the core property of the mSPRT.\n")

ttest_h0_rejects = 0
msprt_h0_rejects = 0

for i in range(N_SIMS):
    rng1 = np.random.default_rng(42 + i)
    rejected_t, _ = run_fixed_horizon_ttest(rng1, RATE_A, RATE_B_H0, EXPOSURE_PER_MONTH, N_MONTHS, ALPHA)

    rng2 = np.random.default_rng(42 + i)
    rejected_m, _ = run_msprt(rng2, RATE_A, RATE_B_H0, EXPOSURE_PER_MONTH, N_MONTHS, ALPHA)

    if rejected_t:
        ttest_h0_rejects += 1
    if rejected_m:
        msprt_h0_rejects += 1

ttest_fpr = ttest_h0_rejects / N_SIMS
msprt_fpr = msprt_h0_rejects / N_SIMS

print("=" * 55)
print(f"{'Method':<35} {'FPR':>8} {'vs alpha':>10}")
print("-" * 55)
print(f"{'t-test (monthly peeking)  [WRONG]':<35} {ttest_fpr:>8.1%} {ttest_fpr/ALPHA:>9.1f}x")
print(f"{'mSPRT  (monthly checking) [VALID]':<35} {msprt_fpr:>8.1%} {msprt_fpr/ALPHA:>9.1f}x")
print(f"{'Nominal alpha':<35} {ALPHA:>8.1%} {'1.0x':>10}")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2: H1 (effect present) — Power and Stopping Time

# COMMAND ----------

print(f"Running {N_SIMS} experiments under H1 (challenger {(1-RATE_B_H1/RATE_A)*100:.0f}% better)...")

ttest_h1_rejects = 0
msprt_h1_rejects = 0
ttest_stop_months = []
msprt_stop_months = []

for i in range(N_SIMS):
    rng1 = np.random.default_rng(1000 + i)
    rejected_t, stop_t = run_fixed_horizon_ttest(rng1, RATE_A, RATE_B_H1, EXPOSURE_PER_MONTH, N_MONTHS, ALPHA)

    rng2 = np.random.default_rng(1000 + i)
    rejected_m, stop_m = run_msprt(rng2, RATE_A, RATE_B_H1, EXPOSURE_PER_MONTH, N_MONTHS, ALPHA)

    if rejected_t:
        ttest_h1_rejects += 1
        ttest_stop_months.append(stop_t)
    if rejected_m:
        msprt_h1_rejects += 1
        msprt_stop_months.append(stop_m)

ttest_power = ttest_h1_rejects / N_SIMS
msprt_power = msprt_h1_rejects / N_SIMS
ttest_avg = np.mean(ttest_stop_months) if ttest_stop_months else float("nan")
msprt_avg = np.mean(msprt_stop_months) if msprt_stop_months else float("nan")

print("=" * 65)
print(f"{'Method':<35} {'Power':>8} {'Avg stop month':>15}")
print("-" * 65)
print(f"{'t-test (monthly peeking)  [WRONG]':<35} {ttest_power:>8.1%} {ttest_avg:>15.1f}")
print(f"{'mSPRT  (monthly checking) [VALID]':<35} {msprt_power:>8.1%} {msprt_avg:>15.1f}")
print("=" * 65)
print(f"\nt-test inflated power comes partly from excess false positives.")
print(f"mSPRT stops ~{ttest_avg - msprt_avg:.0f} months earlier when the effect is genuine.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("SUMMARY")
print("=" * 65)
print(f"\nProblem: monthly peeking with t-test")
print(f"  FPR inflated from {ALPHA:.0%} to {ttest_fpr:.0%} "
      f"({ttest_fpr / ALPHA:.1f}x nominal)")
print(f"  Over 3 years of an insurance experiment, you'd falsely deploy")
print(f"  a useless challenger {ttest_fpr:.0%} of the time instead of {ALPHA:.0%}.")
print(f"\nSolution: mSPRT (insurance-monitoring v0.5.0)")
print(f"  FPR: {msprt_fpr:.0%} ({msprt_fpr / ALPHA:.1f}x nominal) — controlled")
print(f"  Power under H1: {msprt_power:.0%}")
print(f"  Stops ~{ttest_avg - msprt_avg:.0f} months earlier when challenger is genuinely better.")
print(f"\nUsage:")
print(f"  from insurance_monitoring.sequential import SequentialTest")
print(f"  test = SequentialTest(metric='frequency', alpha=0.05, tau=0.03)")
print(f"  result = test.update(champion_claims=12, champion_exposure=150,")
print(f"                       challenger_claims=9, challenger_exposure=75)")
print(f"  if result.should_stop:")
print(f"      print(result.summary)")
