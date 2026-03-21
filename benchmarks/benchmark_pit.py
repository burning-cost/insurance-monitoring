"""
Benchmark: PITMonitor vs repeated Hosmer-Lemeshow testing for calibration change detection.

The problem with repeated H-L tests
------------------------------------
The Hosmer-Lemeshow test was designed for a single holdout check after model fitting.
Applying it monthly in production — checking whether the current month's A/E pattern
deviates significantly — is a repeated-testing problem. With 12 monthly checks per year
at alpha=0.05, the probability of raising at least one false alarm under a well-calibrated
model is roughly 1 - (1-0.05)^12 = 46%. After two years of monthly monitoring: 71%.

PITMonitor uses an e-process (mixture over probability integral transforms) that gives a
formal bound: P(ever alarm | model calibrated) <= alpha, for all t, forever. You can check
it every week or every policy without correction.

DGP
---
Poisson frequency model. Two phases:

- Phase 1 (t=1..500):   y_i ~ Poisson(exposure_i * lambda_hat_i)       — model is correct
- Phase 2 (t=501..1000): y_i ~ Poisson(exposure_i * lambda_hat_i * 1.15) — true rate 15% higher

The model does not know about the phase-2 inflation. PITMonitor detects it from the
accumulating right-skew in the PITs. Repeated H-L raises false alarms throughout phase 1
because of multiple-testing inflation.

Run:
    uv run python benchmarks/benchmark_pit.py
"""

from __future__ import annotations

import math
import sys
import time
import warnings

import numpy as np
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: PITMonitor vs repeated Hosmer-Lemeshow testing")
print("=" * 70)
print()

try:
    from insurance_monitoring import PITMonitor
    print("insurance-monitoring imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-monitoring: {e}")
    print("Install with: pip install insurance-monitoring")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Hosmer-Lemeshow helper (chi-squared variant, 10 bins)
# ---------------------------------------------------------------------------


def hosmer_lemeshow_pvalue(
    y_actual: np.ndarray,
    mu_expected: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Chi-squared Hosmer-Lemeshow test for Poisson counts.

    Bins observations by predicted mean, computes (O-E)^2/E per bin,
    returns p-value from chi-squared(n_bins - 2) approximation.

    This is invalid when called repeatedly — each call is a fresh test
    on the same growing dataset.
    """
    if len(y_actual) < n_bins * 5:
        return 1.0  # too few observations to bin reliably

    # Quantile-based bins on predicted rate
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

    from scipy.stats import chi2
    return float(chi2.sf(hl_stat, df - 2))


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_PHASE1 = 500   # well-calibrated
N_PHASE2 = 500   # 15% rate drift introduced
N_TOTAL = N_PHASE1 + N_PHASE2

DRIFT_FACTOR = 1.15  # true rate in phase 2 = lambda_hat * 1.15
ALPHA = 0.05

print(f"DGP: {N_PHASE1} well-calibrated observations, then {N_PHASE2} with 15% rate inflation")
print(f"Alpha: {ALPHA}")
print()

# Model predicted rates (stable throughout — model does not know about the drift)
exposure = RNG.uniform(0.5, 1.0, N_TOTAL)

# Log-linear frequency model with 3 rating factors
driver_age = RNG.integers(18, 80, N_TOTAL)
vehicle_age = RNG.integers(0, 15, N_TOTAL)
ncd = RNG.integers(0, 9, N_TOTAL)

log_lambda = (
    -2.5
    - 0.012 * (driver_age - 40)
    + 0.06 * vehicle_age / 14
    + 0.05 * (4 - ncd)
)
lambda_hat = np.exp(log_lambda)          # model's predicted rate (per unit exposure)
mu_hat = lambda_hat * exposure           # model's predicted mean claims

# True claims: phase 1 = well-calibrated, phase 2 = 15% inflation
true_mu = mu_hat.copy()
true_mu[N_PHASE1:] *= DRIFT_FACTOR

y = RNG.poisson(true_mu).astype(float)  # observed claims

print(f"Average predicted mu (phase 1): {mu_hat[:N_PHASE1].mean():.4f}")
print(f"Average predicted mu (phase 2): {mu_hat[N_PHASE1:].mean():.4f}  (same — model unchanged)")
print(f"Average true mu (phase 1):      {true_mu[:N_PHASE1].mean():.4f}")
print(f"Average true mu (phase 2):      {true_mu[N_PHASE1:].mean():.4f}  (inflated by {DRIFT_FACTOR}x)")
print()

# ---------------------------------------------------------------------------
# PIT computation (Poisson)
# ---------------------------------------------------------------------------

pits = np.array([
    float(poisson.cdf(int(y[i]), mu=mu_hat[i]))
    for i in range(N_TOTAL)
])

# ---------------------------------------------------------------------------
# BASELINE: Repeated Hosmer-Lemeshow with peeking
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE: Repeated Hosmer-Lemeshow test (checked every 50 observations)")
print("-" * 70)
print()

CHECK_INTERVAL = 50   # check frequency

hl_alarms_phase1 = []   # false alarms (before drift)
hl_alarms_phase2 = []   # true alarms (after drift)

for t in range(CHECK_INTERVAL, N_TOTAL + 1, CHECK_INTERVAL):
    p = hosmer_lemeshow_pvalue(y[:t], mu_hat[:t])
    if p < ALPHA:
        if t <= N_PHASE1:
            hl_alarms_phase1.append(t)
        else:
            hl_alarms_phase2.append(t)

n_checks_phase1 = N_PHASE1 // CHECK_INTERVAL
n_checks_phase2 = N_PHASE2 // CHECK_INTERVAL
hl_fpr = len(hl_alarms_phase1) / n_checks_phase1

print(f"Checks in phase 1 (no drift):  {n_checks_phase1}")
print(f"  H-L false alarms:            {len(hl_alarms_phase1)}")
print(f"  H-L false alarm rate:        {hl_fpr:.1%}  (nominal: {ALPHA:.0%})")
if hl_alarms_phase1:
    print(f"  First false alarm at t:      {hl_alarms_phase1[0]}")
print()
print(f"Checks in phase 2 (drift active): {n_checks_phase2}")
print(f"  H-L detections:               {len(hl_alarms_phase2)}")
if hl_alarms_phase2:
    print(f"  First detection at t:         {hl_alarms_phase2[0]}  "
          f"(delay from drift start: {hl_alarms_phase2[0] - N_PHASE1})")
else:
    print("  H-L failed to detect drift within monitoring window")
print()

# Theoretical FPR inflation: 1 - (1-alpha)^k
expected_fpr = 1.0 - (1.0 - ALPHA) ** n_checks_phase1
print(f"Theoretical FPR with {n_checks_phase1} uncorrected tests: {expected_fpr:.1%}")
print(f"(This is why repeated H-L monitoring is statistically invalid)")
print()

# ---------------------------------------------------------------------------
# LIBRARY: PITMonitor
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: PITMonitor (anytime-valid, alpha=0.05)")
print("-" * 70)
print()

monitor = PITMonitor(alpha=ALPHA, n_bins=50, rng=42)

pit_alarm_t = None
evidence_trace = []

for t in range(N_TOTAL):
    alarm = monitor.update(pits[t])
    evidence_trace.append(alarm.evidence)
    if alarm.triggered and pit_alarm_t is None:
        pit_alarm_t = t + 1  # 1-indexed

summary = monitor.summary()

print(f"Alarm triggered:  {summary.alarm_triggered}")
if pit_alarm_t is not None:
    phase = "PHASE 1 (false alarm!)" if pit_alarm_t <= N_PHASE1 else "PHASE 2 (true detection)"
    delay = pit_alarm_t - N_PHASE1 if pit_alarm_t > N_PHASE1 else None
    print(f"Alarm at t:       {pit_alarm_t}  [{phase}]")
    if delay is not None:
        print(f"Detection delay:  {delay} observations after drift began")
    print(f"Estimated changepoint: {summary.changepoint}")
else:
    print("No alarm raised")
print(f"Final evidence M_t: {summary.evidence:.4f}  (threshold: {summary.threshold:.1f})")
print(f"Calibration score (1 - KS): {summary.calibration_score:.4f}")
print()

# Evidence trajectory at key points
print("Evidence trajectory M_t (selected steps):")
checkpoints = [50, 100, 200, 400, 500, 550, 600, 700, 800, 900, 1000]
for cp in checkpoints:
    if cp <= N_TOTAL:
        ev = evidence_trace[cp - 1]
        phase_label = "pre-drift" if cp <= N_PHASE1 else "post-drift"
        alarm_marker = " <<< ALARM" if ev >= monitor._threshold else ""
        print(f"  t={cp:<4}  M_t = {ev:>8.4f}  [{phase_label}]{alarm_marker}")
print()

# ---------------------------------------------------------------------------
# Monte Carlo FPR simulation for PITMonitor
# ---------------------------------------------------------------------------

print("-" * 70)
print("FPR simulation: 300 experiments under H0 (no drift)")
print("-" * 70)
print()

N_SIM_OBS = 500   # observations per simulation under H0
N_SIM = 300

pit_false_alarms = 0

for sim in range(N_SIM):
    sim_rng = np.random.default_rng(1000 + sim)
    sim_exposure = sim_rng.uniform(0.5, 1.0, N_SIM_OBS)
    sim_lambda = np.exp(-2.5 + 0.01 * sim_rng.standard_normal(N_SIM_OBS))
    sim_mu = sim_lambda * sim_exposure
    sim_y = sim_rng.poisson(sim_mu).astype(float)
    sim_pits = np.array([float(poisson.cdf(int(sim_y[i]), mu=sim_mu[i])) for i in range(N_SIM_OBS)])

    sim_monitor = PITMonitor(alpha=ALPHA, n_bins=50, rng=2000 + sim)
    final_alarm = sim_monitor.update_many(sim_pits, stop_on_alarm=True)
    if final_alarm.triggered:
        pit_false_alarms += 1

pit_sim_fpr = pit_false_alarms / N_SIM
print(f"PITMonitor FPR ({N_SIM} simulations, {N_SIM_OBS} obs each): {pit_sim_fpr:.1%}  (nominal: {ALPHA:.0%})")
print(f"  False alarms: {pit_false_alarms}/{N_SIM}")
print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  {'Metric':<45} {'H-L (repeated)':>14} {'PITMonitor':>12}")
print(f"  {'-'*45} {'-'*14} {'-'*12}")

rows = [
    ("False alarm rate under H0",
     f"{hl_fpr:.1%}",
     f"{pit_sim_fpr:.1%}"),
    ("Nominal alpha",
     f"{ALPHA:.0%}",
     f"{ALPHA:.0%}"),
    ("FPR / nominal (inflation factor)",
     f"{hl_fpr / ALPHA:.1f}x",
     f"{pit_sim_fpr / ALPHA:.1f}x"),
    ("Detects drift in phase 2",
     "YES" if hl_alarms_phase2 else "NO",
     "YES" if pit_alarm_t and pit_alarm_t > N_PHASE1 else "NO"),
    ("Detection delay (obs after drift)",
     f"{hl_alarms_phase2[0] - N_PHASE1}" if hl_alarms_phase2 else "N/A",
     f"{pit_alarm_t - N_PHASE1}" if pit_alarm_t and pit_alarm_t > N_PHASE1 else "N/A"),
    ("Alarms before drift (false positives)",
     str(len(hl_alarms_phase1)),
     "0" if not pit_alarm_t or pit_alarm_t > N_PHASE1 else "1"),
    ("Anytime-valid guarantee",
     "NO",
     "YES"),
    ("Valid with any check schedule",
     "NO",
     "YES"),
    ("Changepoint estimate",
     "NO",
     f"t~{summary.changepoint}" if summary.changepoint else "N/A"),
]

for metric, hl, pit in rows:
    print(f"  {metric:<45} {hl:>14} {pit:>12}")

print()
print("  DGP: Poisson frequency, 15% rate drift from t=501")
print(f"  PITMonitor guarantee: P(ever alarm | calibrated) <= {ALPHA:.0%}")
print(f"  H-L repeated testing: P(ever alarm | calibrated) ~{expected_fpr:.0%} ({n_checks_phase1} looks)")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
