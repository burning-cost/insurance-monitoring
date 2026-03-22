# Databricks notebook source
# MAGIC %md
# MAGIC # Sequential Testing — insurance-monitoring v0.8.0
# MAGIC
# MAGIC **Burning Cost | March 2026**
# MAGIC
# MAGIC Champion/challenger experiments in insurance pricing have a timing problem.
# MAGIC Your commercial instinct is correct: check results monthly. Your statistics
# MAGIC textbook says you can't. The peeking problem inflates false positive rates
# MAGIC from 5% to 25% or higher on a monthly cadence over two years.
# MAGIC
# MAGIC The `sequential` module uses the mixture Sequential Probability Ratio Test
# MAGIC (mSPRT) from Johari et al. (2022). The test statistic is an e-process —
# MAGIC a nonneg martingale under H0 — which gives exact type I error control at
# MAGIC every interim check, with no pre-specified sample size or look schedule.
# MAGIC
# MAGIC **This notebook demonstrates:**
# MAGIC - Full mSPRT workflow on a synthetic UK motor book (35k champion / 15k challenger)
# MAGIC - Monthly updates, anytime-valid confidence sequences, and early stopping
# MAGIC - Frequency test (Poisson rate ratio), severity test (log-normal), loss ratio
# MAGIC - History DataFrame for monitoring dashboards
# MAGIC
# MAGIC **References:**
# MAGIC - Johari et al. (2022). Always Valid Inference. OR 70(3). arXiv:1512.04922.
# MAGIC - Howard et al. (2021). Time-uniform confidence sequences. AoS 49(2).
# MAGIC - Ramdas et al. (2023). Game-Theoretic Statistics and SAVI. Stat Sci 38(4).

# COMMAND ----------

# MAGIC %pip install insurance-monitoring>=0.8.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import datetime
import numpy as np
import polars as pl
import insurance_monitoring
from insurance_monitoring.sequential import SequentialTest, sequential_test_from_df

print(f"insurance-monitoring version: {insurance_monitoring.__version__}")
print("Imports successful.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Quickstart — frequency test
# MAGIC
# MAGIC Experiment setup from the spec (Section 11):
# MAGIC - Champion: 35,000 policies/year at 0.100 claims/car-year
# MAGIC - Challenger: 15,000 policies/year (30% allocation), 0.094 (6% improvement)
# MAGIC - Monthly updates

# COMMAND ----------

rng = np.random.default_rng(42)

test = SequentialTest(
    metric="frequency",
    alpha=0.05,
    tau=0.04,            # prior expects ~4% effects on log-rate-ratio scale
    max_duration_years=2.0,
    min_exposure_per_arm=200.0,  # wait for 200 car-years before any stopping decision
)

print(f"{'Month':>5}  {'Lambda':>8}  {'Ratio':>7}  {'CS lower':>8}  {'CS upper':>8}  {'P(chall better)':>16}  {'Decision'}")
print("-" * 80)

final_result = None
for month in range(24):
    champ_exp = 35_000 * 0.8 / 12    # ~2333 car-years/month (80% avg annual duration)
    chall_exp = 15_000 * 0.8 / 12    # ~1000 car-years/month
    champ_claims = int(rng.poisson(0.100 * champ_exp))
    chall_claims = int(rng.poisson(0.094 * chall_exp))

    result = test.update(
        champion_claims=champ_claims,
        champion_exposure=champ_exp,
        challenger_claims=chall_claims,
        challenger_exposure=chall_exp,
        calendar_date=datetime.date(2025, 1, 1) + datetime.timedelta(days=30 * month),
    )

    print(
        f"{month+1:>5}  {result.lambda_value:>8.2f}  "
        f"{result.rate_ratio:>7.4f}  "
        f"{result.rate_ratio_ci_lower:>8.4f}  "
        f"{result.rate_ratio_ci_upper:>8.4f}  "
        f"{result.prob_challenger_better:>16.3f}  "
        f"{result.decision}"
    )

    final_result = result
    if result.should_stop:
        break

print()
print("=" * 80)
print(final_result.summary)
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. History DataFrame
# MAGIC
# MAGIC Every `update()` call is stored. Access the full history for dashboards or
# MAGIC Lambda trajectory plots.

# COMMAND ----------

history = test.history()
print(f"History rows: {len(history)}")
print(history.select([
    "period_index", "lambda_value", "log_lambda_value",
    "champion_rate", "challenger_rate", "rate_ratio",
    "ci_lower", "ci_upper", "decision"
]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch processing from a DataFrame
# MAGIC
# MAGIC If your data lives in a Delta table, read it into a Polars DataFrame and
# MAGIC use `sequential_test_from_df()` instead of calling `update()` in a loop.

# COMMAND ----------

# Build a representative 12-month DataFrame
rng2 = np.random.default_rng(7)
monthly_records = []
for m in range(12):
    monthly_records.append({
        "period_end": datetime.date(2025, 1, 31) + datetime.timedelta(days=30 * m),
        "champ_claims": int(rng2.poisson(0.10 * 500)),
        "champ_exp": 500.0,
        "chall_claims": int(rng2.poisson(0.092 * 200)),  # 8% improvement, 28% allocation
        "chall_exp": 200.0,
    })

df = pl.DataFrame(monthly_records)
print("Monthly data:")
print(df)

result = sequential_test_from_df(
    df=df,
    champion_claims_col="champ_claims",
    champion_exposure_col="champ_exp",
    challenger_claims_col="chall_claims",
    challenger_exposure_col="chall_exp",
    date_col="period_end",
    metric="frequency",
    alpha=0.05,
    tau=0.04,
    max_duration_years=2.0,
)

print(f"\nAfter {result.n_updates} monthly updates:")
print(result.summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Two-sided vs one-sided tests
# MAGIC
# MAGIC For a new model where you know the direction (challenger should be cheaper),
# MAGIC use `alternative='less'`. This concentrates power on the direction of interest.
# MAGIC
# MAGIC - `'two_sided'` (default): detects effects in either direction. Appropriate
# MAGIC   when you are genuinely uncertain which model is better.
# MAGIC - `'less'`: detects challenger lower than champion. More powerful for
# MAGIC   one-directional model improvements.
# MAGIC - `'greater'`: detects challenger higher. Useful for testing whether a new
# MAGIC   model inflates claims (a regulatory concern).

# COMMAND ----------

rng3 = np.random.default_rng(99)

test_one_sided = SequentialTest(
    metric="frequency",
    alternative="less",   # challenger rate < champion rate
    alpha=0.05,
    tau=0.04,
    max_duration_years=2.0,
    min_exposure_per_arm=200.0,
)

for month in range(24):
    ca = int(rng3.poisson(0.10 * 2333))
    cb = int(rng3.poisson(0.094 * 1000))
    r = test_one_sided.update(ca, 2333.0, cb, 1000.0)
    if r.should_stop:
        print(f"One-sided test stopped at month {month + 1}: {r.decision}")
        print(r.summary)
        break
else:
    print(f"One-sided test inconclusive after 24 months. Lambda={r.lambda_value:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Loss ratio test (compound metric)
# MAGIC
# MAGIC The loss ratio combines frequency and severity via the e-value product rule
# MAGIC (Ramdas et al. 2023, Proposition 2.1). Pass per-claim log-cost sufficient
# MAGIC statistics alongside claims and exposure.
# MAGIC
# MAGIC Required per update:
# MAGIC - `champion_severity_sum`: sum of log(claim_cost) for new champion claims
# MAGIC - `champion_severity_ss`: sum of log(claim_cost)^2 for new champion claims
# MAGIC - Same for challenger

# COMMAND ----------

rng4 = np.random.default_rng(55)

test_lr = SequentialTest(
    metric="loss_ratio",
    alpha=0.05,
    tau=0.04,
    max_duration_years=2.0,
    min_exposure_per_arm=200.0,
)

for month in range(24):
    # Frequency
    ca = int(rng4.poisson(0.10 * 1000))
    cb = int(rng4.poisson(0.094 * 500))

    # Severity: challenger also has lower average severity (log-mean 6.9 vs 7.0)
    log_costs_a = rng4.normal(7.0, 0.8, max(1, ca))
    log_costs_b = rng4.normal(6.9, 0.8, max(1, cb))

    r = test_lr.update(
        champion_claims=ca,
        champion_exposure=1000.0,
        challenger_claims=cb,
        challenger_exposure=500.0,
        champion_severity_sum=float(log_costs_a.sum()) if ca > 0 else None,
        champion_severity_ss=float((log_costs_a ** 2).sum()) if ca > 0 else None,
        challenger_severity_sum=float(log_costs_b.sum()) if cb > 0 else None,
        challenger_severity_ss=float((log_costs_b ** 2).sum()) if cb > 0 else None,
    )

    if r.should_stop:
        print(f"Loss ratio test stopped at month {month + 1}: {r.decision}")
        print(r.summary)
        break
else:
    print(f"Loss ratio test inconclusive after 24 months. Lambda={r.lambda_value:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Key parameters — calibrating tau
# MAGIC
# MAGIC `tau` is the prior standard deviation on the log-rate-ratio effect size.
# MAGIC
# MAGIC | tau  | Optimal for detecting | Approx months to stop (50k-policy book, 30% challenger) |
# MAGIC |------|----------------------|----------------------------------------------------------|
# MAGIC | 0.02 | 2% improvement       | 20-28 months                                             |
# MAGIC | 0.03 | 3% improvement       | 14-20 months                                             |
# MAGIC | 0.05 | 5% improvement       | 9-13 months                                              |
# MAGIC | 0.10 | 10%+ improvement     | 4-7 months                                               |
# MAGIC
# MAGIC Default `tau=0.03` is conservative. Use `tau=0.05` for large re-rates.
# MAGIC Setting tau too small wastes time on real effects; too large inflates Lambda
# MAGIC for small-effect nulls (still valid, but stops sooner than alpha implies).

# COMMAND ----------

print("Tau sensitivity: stopping time for 5% effect, 500 car-years/arm/month")
print(f"{'tau':>6}  {'stops (of 200 runs)':>20}  {'median stop month':>18}")
print("-" * 50)

for tau_val in [0.02, 0.03, 0.05, 0.10]:
    rng_t = np.random.default_rng(200)
    stop_months = []
    for _ in range(200):
        t = SequentialTest(
            metric="frequency", alpha=0.05, tau=tau_val,
            max_duration_years=3.0, min_exposure_per_arm=500.0,
        )
        for m in range(36):
            ca = int(rng_t.poisson(0.10 * 500))
            cb = int(rng_t.poisson(0.095 * 500))  # 5% improvement
            r = t.update(ca, 500.0, cb, 500.0)
            if r.should_stop and r.decision == "reject_H0":
                stop_months.append(m + 1)
                break
    n_stops = len(stop_months)
    median_stop = float(np.median(stop_months)) if stop_months else float("nan")
    print(f"{tau_val:>6.2f}  {n_stops:>20}  {median_stop:>18.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. reset() for simulation loops
# MAGIC
# MAGIC `reset()` clears accumulated state without creating a new object.
# MAGIC Useful inside Monte Carlo loops where you are reusing the same test instance.

# COMMAND ----------

test_reuse = SequentialTest(metric="frequency", alpha=0.05, tau=0.03, min_exposure_per_arm=0.0)
rng5 = np.random.default_rng(300)
rejections = 0

for sim in range(100):
    test_reuse.reset()
    for _ in range(24):
        ca = int(rng5.poisson(0.10 * 50))
        cb = int(rng5.poisson(0.10 * 50))  # H0: equal rates
        r = test_reuse.update(ca, 50.0, cb, 50.0)
        if r.should_stop and r.decision == "reject_H0":
            rejections += 1
            break

print(f"100 null simulations (H0 true): {rejections} rejections (FPR = {rejections/100:.1%})")
print("Expected: <= 5% (mSPRT type I error control)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `sequential` module gives insurance pricing teams what they actually need:
# MAGIC a champion/challenger test they can check every month without inflating FPR.
# MAGIC
# MAGIC **Key properties:**
# MAGIC - Valid at every interim check — no peeking penalty, no Bonferroni correction
# MAGIC - No pre-specified sample size or look schedule
# MAGIC - Stops as soon as evidence is sufficient — not at a fixed end date
# MAGIC - Supports frequency, severity, and loss ratio (product of e-values)
# MAGIC - Anytime-valid confidence sequences alongside the stopping rule
# MAGIC - Bayesian posterior as secondary display (not used for stopping)
# MAGIC
# MAGIC **When NOT to use it:**
# MAGIC Hard commitment to a fixed sample size you will genuinely never peek before
# MAGIC it completes. In that case, a standard two-sample z-test for Poisson rates
# MAGIC is more powerful than mSPRT. In practice, nobody does this.
# MAGIC
# MAGIC **On IBNR:** for short-tail lines (motor AD, home buildings), use reported
# MAGIC paid claims — 80-90% reported within 3 months. For bodily injury and liability,
# MAGIC restrict to claim frequency only (not incurred cost) or run for 18+ months.
