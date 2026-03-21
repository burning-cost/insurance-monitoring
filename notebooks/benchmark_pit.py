# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: PITMonitor vs repeated Hosmer-Lemeshow testing
# MAGIC
# MAGIC **Library:** `insurance-monitoring` v0.7.0 — `PITMonitor` for anytime-valid calibration
# MAGIC change detection in deployed pricing models
# MAGIC
# MAGIC **Baseline:** Repeated Hosmer-Lemeshow test applied monthly — the common practice that
# MAGIC turns a valid single-comparison test into an invalid sequential procedure
# MAGIC
# MAGIC **Date:** 2026-03-21
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## The problem
# MAGIC
# MAGIC The Hosmer-Lemeshow test was designed for a single holdout evaluation after model fitting.
# MAGIC Applying it every month in production — checking whether the current month's A/E pattern
# MAGIC deviates significantly from the model's predicted distribution — is a repeated-testing
# MAGIC problem with no correction applied.
# MAGIC
# MAGIC With 12 monthly checks per year at alpha=0.05, the probability of raising at least one
# MAGIC false alarm under a well-calibrated model is roughly 1 - (1-0.05)^12 = 46%. After two
# MAGIC years of monthly monitoring: 71%. The team thinks the model is misfiring when it is not.
# MAGIC
# MAGIC `PITMonitor` uses an e-process over probability integral transforms (Henzi, Murph, Ziegel
# MAGIC 2025, arXiv:2603.13156) that gives a formal bound: P(ever alarm | model calibrated) <= alpha,
# MAGIC at any checking frequency, forever.
# MAGIC
# MAGIC ## DGP
# MAGIC
# MAGIC Poisson frequency model. Two phases:
# MAGIC - Phase 1 (t=1..500): y ~ Poisson(exposure * lambda_hat) — model is correct
# MAGIC - Phase 2 (t=501..1000): y ~ Poisson(exposure * lambda_hat * 1.15) — true rate 15% higher
# MAGIC
# MAGIC The model does not know about the phase-2 inflation. PITMonitor detects the accumulating
# MAGIC right-skew in the PITs. Repeated H-L raises false alarms throughout phase 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-monitoring

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import math
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import poisson

from insurance_monitoring import PITMonitor

warnings.filterwarnings("ignore")
print(f"Run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data-generating process

# COMMAND ----------

RNG = np.random.default_rng(42)

N_PHASE1 = 500    # well-calibrated
N_PHASE2 = 500    # 15% rate drift introduced
N_TOTAL = N_PHASE1 + N_PHASE2

DRIFT_FACTOR = 1.15
ALPHA = 0.05

# Log-linear frequency model (stale — does not drift with phase 2)
exposure = RNG.uniform(0.5, 1.0, N_TOTAL)
driver_age = RNG.integers(18, 80, N_TOTAL)
vehicle_age = RNG.integers(0, 15, N_TOTAL)
ncd = RNG.integers(0, 9, N_TOTAL)

log_lambda = (
    -2.5
    - 0.012 * (driver_age - 40)
    + 0.06 * vehicle_age / 14
    + 0.05 * (4 - ncd)
)
lambda_hat = np.exp(log_lambda)
mu_hat = lambda_hat * exposure

# True claims
true_mu = mu_hat.copy()
true_mu[N_PHASE1:] *= DRIFT_FACTOR   # phase 2 drift
y = RNG.poisson(true_mu).astype(float)

# Compute PITs: F(y | mu_hat) using Poisson CDF
pits = np.array([float(poisson.cdf(int(y[i]), mu=mu_hat[i])) for i in range(N_TOTAL)])

print(f"Phase 1 (t=1..{N_PHASE1}):   A/E = {y[:N_PHASE1].sum() / mu_hat[:N_PHASE1].sum():.4f}")
print(f"Phase 2 (t={N_PHASE1+1}..{N_TOTAL}): A/E = {y[N_PHASE1:].sum() / mu_hat[N_PHASE1:].sum():.4f}  (drift: {DRIFT_FACTOR}x)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Repeated Hosmer-Lemeshow testing

# COMMAND ----------

from scipy.stats import chi2 as chi2_dist


def hosmer_lemeshow_pvalue(y_actual, mu_expected, n_bins=10):
    """Chi-squared H-L test for Poisson counts. Invalid when called repeatedly."""
    if len(y_actual) < n_bins * 5:
        return 1.0
    quantiles = np.percentile(mu_expected, np.linspace(0, 100, n_bins + 1))
    quantiles[0] -= 1e-6
    quantiles[-1] += 1e-6
    hl_stat = 0.0
    df = 0
    for b in range(n_bins):
        mask = (mu_expected > quantiles[b]) & (mu_expected <= quantiles[b + 1])
        if mask.sum() == 0:
            continue
        obs = float(y_actual[mask].sum())
        exp = float(mu_expected[mask].sum())
        if exp > 0:
            hl_stat += (obs - exp) ** 2 / exp
            df += 1
    if df <= 2:
        return 1.0
    return float(chi2_dist.sf(hl_stat, df - 2))


CHECK_INTERVAL = 50   # check every 50 new observations

hl_alarms_phase1 = []
hl_alarms_phase2 = []

for t in range(CHECK_INTERVAL, N_TOTAL + 1, CHECK_INTERVAL):
    p = hosmer_lemeshow_pvalue(y[:t], mu_hat[:t])
    if p < ALPHA:
        if t <= N_PHASE1:
            hl_alarms_phase1.append(t)
        else:
            hl_alarms_phase2.append(t)

n_checks_phase1 = N_PHASE1 // CHECK_INTERVAL
hl_fpr = len(hl_alarms_phase1) / n_checks_phase1
expected_fpr = 1.0 - (1.0 - ALPHA) ** n_checks_phase1

print("Repeated Hosmer-Lemeshow (checked every 50 observations)")
print(f"  Checks in phase 1 (no drift): {n_checks_phase1}")
print(f"  False alarms in phase 1:      {len(hl_alarms_phase1)}  (rate: {hl_fpr:.1%})")
print(f"  Nominal alpha:                {ALPHA:.0%}")
print(f"  Theoretical FPR ({n_checks_phase1} looks):  {expected_fpr:.0%}")
print(f"  Detections in phase 2:        {len(hl_alarms_phase2)}")
if hl_alarms_phase1:
    print(f"  First false alarm at t:       {hl_alarms_phase1[0]}")
if hl_alarms_phase2:
    print(f"  First true detection at t:    {hl_alarms_phase2[0]}  (delay: {hl_alarms_phase2[0]-N_PHASE1})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PITMonitor

# COMMAND ----------

monitor = PITMonitor(alpha=ALPHA, n_bins=50, rng=42)

pit_alarm_t = None
evidence_trace = []

for t in range(N_TOTAL):
    alarm = monitor.update(pits[t])
    evidence_trace.append(alarm.evidence)
    if alarm.triggered and pit_alarm_t is None:
        pit_alarm_t = t + 1

summary = monitor.summary()

print(f"PITMonitor result:")
print(f"  Alarm triggered:       {summary.alarm_triggered}")
if pit_alarm_t:
    phase = "PHASE 1 (false alarm!)" if pit_alarm_t <= N_PHASE1 else "PHASE 2 (correct detection)"
    print(f"  Alarm at t:            {pit_alarm_t}  [{phase}]")
    if pit_alarm_t > N_PHASE1:
        print(f"  Detection delay:       {pit_alarm_t - N_PHASE1} observations after drift began")
    print(f"  Estimated changepoint: t={summary.changepoint}")
else:
    print("  No alarm raised within monitoring window")
print(f"  Final evidence M_t:    {summary.evidence:.4f}  (threshold: {summary.threshold:.1f})")
print(f"  Calibration score:     {summary.calibration_score:.4f}  (1 = perfect uniformity)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evidence trajectory

# COMMAND ----------

# Print evidence at key checkpoints
print("Evidence M_t at selected steps:")
print(f"  {'t':>5}  {'M_t':>10}  {'Phase':>15}  {'Note'}")
print(f"  {'─'*5}  {'─'*10}  {'─'*15}  {'─'*20}")

threshold = monitor._threshold
checkpoints = [50, 100, 200, 400, 500, 550, 600, 700, 800, 900, 1000]
for cp in checkpoints:
    if cp <= N_TOTAL:
        ev = evidence_trace[cp - 1]
        phase = "pre-drift" if cp <= N_PHASE1 else "post-drift"
        note = ">>> ALARM" if ev >= threshold else ("threshold: {:.0f}".format(threshold))
        print(f"  {cp:>5}  {ev:>10.4f}  {phase:>15}  {note}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. FPR simulation under H0

# COMMAND ----------

N_SIM_OBS = 500
N_SIM = 300
pit_false_alarms = 0

for sim in range(N_SIM):
    sim_rng = np.random.default_rng(1000 + sim)
    sim_exp = sim_rng.uniform(0.5, 1.0, N_SIM_OBS)
    sim_lam = np.exp(-2.5 + 0.01 * sim_rng.standard_normal(N_SIM_OBS))
    sim_mu = sim_lam * sim_exp
    sim_y = sim_rng.poisson(sim_mu).astype(float)
    sim_pits = np.array([float(poisson.cdf(int(sim_y[i]), mu=sim_mu[i])) for i in range(N_SIM_OBS)])

    sim_monitor = PITMonitor(alpha=ALPHA, n_bins=50, rng=2000 + sim)
    final = sim_monitor.update_many(sim_pits, stop_on_alarm=True)
    if final.triggered:
        pit_false_alarms += 1

pit_sim_fpr = pit_false_alarms / N_SIM
print(f"PITMonitor FPR ({N_SIM} simulations, {N_SIM_OBS} obs each under H0):")
print(f"  False alarms:  {pit_false_alarms}/{N_SIM}")
print(f"  Empirical FPR: {pit_sim_fpr:.1%}  (nominal: {ALPHA:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary table

# COMMAND ----------

print("=" * 68)
print("SUMMARY: PITMonitor vs repeated Hosmer-Lemeshow")
print("=" * 68)
print(f"  {'Metric':<43} {'H-L':>12} {'PITMonitor':>10}")
print(f"  {'-'*43} {'-'*12} {'-'*10}")

rows = [
    ("False alarm rate under H0",
     f"{hl_fpr:.1%}", f"{pit_sim_fpr:.1%}"),
    ("Nominal alpha",
     f"{ALPHA:.0%}", f"{ALPHA:.0%}"),
    ("FPR / nominal (inflation)",
     f"{hl_fpr/ALPHA:.1f}x", f"{pit_sim_fpr/ALPHA:.1f}x"),
    ("Theoretical FPR (10 looks)",
     f"{expected_fpr:.0%}", "<=5%"),
    ("Detects drift in phase 2",
     "YES" if hl_alarms_phase2 else "NO",
     "YES" if pit_alarm_t and pit_alarm_t > N_PHASE1 else "NO"),
    ("Detection delay (obs after drift)",
     f"{hl_alarms_phase2[0]-N_PHASE1}" if hl_alarms_phase2 else "N/A",
     f"{pit_alarm_t - N_PHASE1}" if pit_alarm_t and pit_alarm_t > N_PHASE1 else "N/A"),
    ("False alarms before drift",
     str(len(hl_alarms_phase1)),
     "0" if not pit_alarm_t or pit_alarm_t > N_PHASE1 else "1"),
    ("Anytime-valid guarantee",
     "NO", "YES"),
    ("Changepoint estimation",
     "NO", f"t~{summary.changepoint}" if summary.changepoint else "N/A"),
]

for metric, hl, pit in rows:
    print(f"  {metric:<43} {hl:>12} {pit:>10}")

print()
print("  Drift: 15% rate inflation from t=501")
print(f"  PITMonitor: P(ever alarm | calibrated) <= {ALPHA:.0%}")
print(f"  H-L repeated: P(ever alarm | calibrated) ~{expected_fpr:.0%} (10 looks)")
