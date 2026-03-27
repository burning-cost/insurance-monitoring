"""
UK home insurance — anytime-valid calibration monitoring with PITMonitor.

Scenario:
    A home contents frequency model was validated in January and deployed.
    The pricing team wants ongoing calibration monitoring without inflating
    type I error across monthly checks. The standard approach — running
    an A/E test or Hosmer-Lemeshow test each month — gives a ~40% chance
    of a false alarm within a year even on a perfectly calibrated model.

    PITMonitor uses probability integral transform e-processes
    (Henzi, Murph, Ziegel 2025, arXiv:2603.13156). The guarantee:

        P(ever raise an alarm | model calibrated) <= alpha, for all t, forever.

    You can check the monitor every week, every month, whenever you like,
    and the type I error guarantee holds without any correction for
    multiple looks.

Commercial interpretation:
    This is the correct monitoring tool when:
    - You want to detect calibration shift as soon as possible after it occurs.
    - Your monitoring cadence is monthly or more frequent.
    - You are reporting to a model risk committee that expects documented
      false-positive rates.

    When the alarm triggers, use MonitoringReport to distinguish between
    RECALIBRATE and REFIT before taking action.
"""

import numpy as np
from scipy.stats import poisson as poisson_dist
from insurance_monitoring import PITMonitor, MonitoringReport

rng = np.random.default_rng(2025)

# ── Model setup ───────────────────────────────────────────────────────────────
# A Poisson frequency model for home contents.
# lambda_hat is the model's predicted claim rate (claims per policy-year).
# We simulate monthly batches of new business.

TRUE_LAMBDA = 0.07       # true claim rate post-deployment
MODEL_LAMBDA = 0.07      # model prediction — initially well-calibrated
DRIFT_MONTH = 8          # calibration starts drifting at month 8
DRIFT_AMOUNT = 0.025     # model becomes cheap by 2.5pp after drift

n_policies_per_month = 800

monitor = PITMonitor(alpha=0.05)

print("UK home contents — PITMonitor monthly check")
print("=" * 60)
print(f"{'Month':<8} {'Policies':<12} {'e-value':>10} {'Alarm':>8}")
print("-" * 60)

for month in range(1, 25):
    # Introduce calibration drift at DRIFT_MONTH
    true_lambda = TRUE_LAMBDA + (DRIFT_AMOUNT if month >= DRIFT_MONTH else 0.0)
    model_lambda = MODEL_LAMBDA  # model does not update — it is stale

    # Simulate new policies this month
    n = n_policies_per_month
    exposure = rng.uniform(0.5, 1.0, n)       # partial-year policies
    true_mu = exposure * true_lambda           # true expected claims per policy
    model_mu = exposure * model_lambda         # model's expected claims per policy

    actual_claims = rng.poisson(true_mu).astype(float)

    # Compute probability integral transforms
    # PIT = CDF of the predictive distribution evaluated at the observed value
    # For Poisson(mu): PIT = P(X <= x | X ~ Poisson(mu))
    for i in range(n):
        pit = float(poisson_dist.cdf(int(actual_claims[i]), model_mu[i]))
        alarm = monitor.update(pit)

    # Report the e-value and alarm status after each month's batch
    summary = monitor.summary()
    alarm_flag = "ALARM" if summary.alarm_triggered else ""
    print(f"{month:<8} {n:<12} {summary.e_value:>10.2f} {alarm_flag:>8}")

    if summary.alarm_triggered:
        print()
        print(f"Calibration alarm triggered at month {month}.")
        print(f"E-value: {summary.e_value:.2f} (threshold: {1/0.05:.0f})")
        print(f"Drift was introduced at month {DRIFT_MONTH}.")
        print(f"Detection lag: {month - DRIFT_MONTH} months.")
        break

print()
print("── What to do when the alarm triggers ─────────────────────────────────")
print("""
  1. Run a MonitoringReport on the last N months of live data vs the
     reference period to get the recommendation (RECALIBRATE or REFIT).

  2. Check the Murphy decomposition:
     - MCB% dominant → model is systematically mispriced → apply offset
     - DSC% dominant → model's rank-ordering has degraded → refit

  3. If RECALIBRATE: fit a new intercept on recent data and re-deploy.
     This is a single-parameter update — fast, low model risk risk.

  4. If REFIT: full redevelopment cycle. Escalate to model risk committee
     with the PITMonitor e-value series and the Gini drift test result
     from GiniDriftBootstrapTest as supporting evidence.
""")

# ── Null behaviour check ──────────────────────────────────────────────────────
# Demonstrate the false-positive guarantee: run 1,000 simulations
# with a perfectly calibrated model and count false alarms.

print("── False positive rate verification (500 simulations) ──────────────────")
false_alarms = 0
n_sims = 500
n_obs_per_sim = 10_000

for _ in range(n_sims):
    null_monitor = PITMonitor(alpha=0.05)
    for _ in range(n_obs_per_sim):
        # Under the null: model is perfectly calibrated, PIT ~ Uniform(0,1)
        pit = float(rng.uniform(0, 1))
        alarm = null_monitor.update(pit)
    if null_monitor.summary().alarm_triggered:
        false_alarms += 1

fpr = false_alarms / n_sims
print(f"  Alpha:         0.05")
print(f"  Observations:  {n_obs_per_sim:,} per simulation")
print(f"  Simulations:   {n_sims}")
print(f"  False alarms:  {false_alarms}")
print(f"  Empirical FPR: {fpr:.3f}  (expected <= 0.05)")
