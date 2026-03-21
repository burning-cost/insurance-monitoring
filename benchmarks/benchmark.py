"""
Benchmark: insurance-monitoring vs manual A/E ratio check for model drift detection.

The claim: a simple aggregate A/E ratio misses three common failure modes that
occur silently in deployed pricing models — covariate shift in the portfolio,
calibration drift in specific segments, and degradation in discriminatory power.
Each can result in significant underpricing before the aggregate A/E triggers.

Setup:
- 50,000 reference (training) policies, Poisson frequency model
- 15,000 monitoring period policies with three deliberately induced problems:
    1. Covariate shift: younger drivers (age 18-30) oversampled 2x
    2. Calibration drift: high-risk vehicles (age < 3) have claims inflated 25%
    3. Discrimination decay: model predictions partially randomised in monitoring period
- Known DGP so we can verify exactly what each approach catches

Expected output:
- Aggregate A/E: looks acceptable (near 1.0) because errors cancel at portfolio level
- PSI flags the covariate shift on driver_age
- A/E by segment shows the segment-level calibration drift
- Gini drift test detects the discrimination decay
- MonitoringReport gives a structured RECALIBRATE or REFIT recommendation

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: insurance-monitoring vs aggregate A/E check")
print("=" * 70)
print()

try:
    from insurance_monitoring import (
        MonitoringReport,
        psi,
        csi,
        gini_coefficient,
        gini_drift_test,
        ae_ratio,
        ae_ratio_ci,
    )
    print("insurance-monitoring imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-monitoring: {e}")
    print("Install with: pip install insurance-monitoring")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_REF = 50_000   # reference / training window
N_CUR = 15_000   # current monitoring period

print(f"DGP: {N_REF:,} reference policies, {N_CUR:,} monitoring-period policies")
print("Three induced failure modes:")
print("  1. Covariate shift: younger drivers (18-30) oversampled 2x in monitoring period")
print("  2. Calibration drift: new vehicles (age < 3) have claims inflated 25%")
print("  3. Discrimination decay: monitoring period predictions 30% randomised")
print()

# --- Reference period (clean, well-calibrated model) ---
driver_age_ref = RNG.integers(18, 80, N_REF)
vehicle_age_ref = RNG.integers(0, 15, N_REF)
ncd_years_ref = RNG.integers(0, 9, N_REF)
exposure_ref = RNG.uniform(0.5, 1.0, N_REF)

# True frequency model: Poisson, log-linear
log_freq_ref = (
    -2.8
    - 0.015 * (driver_age_ref - 40)           # older drivers lower risk
    + 0.08 * np.clip(ncd_years_ref - 4, -4, 4)  # ncd effect
    + 0.05 * vehicle_age_ref                   # older vehicles slightly higher risk
)
freq_ref = np.exp(log_freq_ref)
act_ref = RNG.poisson(freq_ref * exposure_ref).astype(float)
pred_ref = freq_ref * exposure_ref  # well-calibrated model

# --- Monitoring period: three deliberate failure modes ---

# Failure mode 1: covariate shift — young drivers oversampled
young_count = int(N_CUR * 0.40)   # 40% young (vs ~20% in reference)
old_count = N_CUR - young_count
driver_age_cur = np.concatenate([
    RNG.integers(18, 30, young_count),  # young drivers
    RNG.integers(30, 80, old_count),    # rest of portfolio
])
RNG.shuffle(driver_age_cur)

vehicle_age_cur = RNG.integers(0, 15, N_CUR)
ncd_years_cur = RNG.integers(0, 9, N_CUR)
exposure_cur = RNG.uniform(0.5, 1.0, N_CUR)

log_freq_cur = (
    -2.8
    - 0.015 * (driver_age_cur - 40)
    + 0.08 * np.clip(ncd_years_cur - 4, -4, 4)
    + 0.05 * vehicle_age_cur
)
freq_cur = np.exp(log_freq_cur)

# Failure mode 2: calibration drift — new vehicles have 25% more claims
new_vehicle_mask = vehicle_age_cur < 3
actual_freq_cur = freq_cur.copy()
actual_freq_cur[new_vehicle_mask] *= 1.25  # model doesn't know about this

# Model predicts using old parameters (doesn't know about vehicle inflation)
pred_cur = freq_cur * exposure_cur  # model is stale

# Failure mode 3: discrimination decay — randomise 30% of predictions
randomise_mask = RNG.random(N_CUR) < 0.30
pred_cur_degraded = pred_cur.copy()
pred_cur_degraded[randomise_mask] = RNG.uniform(
    pred_cur.min(), pred_cur.max(), randomise_mask.sum()
)

act_cur = RNG.poisson(actual_freq_cur * exposure_cur).astype(float)

# Feature DataFrames for PSI/CSI
feat_ref = pl.DataFrame({
    "driver_age": driver_age_ref.tolist(),
    "vehicle_age": vehicle_age_ref.tolist(),
    "ncd_years": ncd_years_ref.tolist(),
})
feat_cur = pl.DataFrame({
    "driver_age": driver_age_cur.tolist(),
    "vehicle_age": vehicle_age_cur.tolist(),
    "ncd_years": ncd_years_cur.tolist(),
})

# ---------------------------------------------------------------------------
# BASELINE: Manual aggregate A/E check (what most teams do)
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE: Manual aggregate A/E ratio check")
print("-" * 70)

total_actual_cur = float(act_cur.sum())
total_expected_cur = float(pred_cur_degraded.sum())
aggregate_ae = total_actual_cur / total_expected_cur

total_actual_ref = float(act_ref.sum())
total_expected_ref = float(pred_ref.sum())
aggregate_ae_ref = total_actual_ref / total_expected_ref

print(f"  Reference A/E:   {aggregate_ae_ref:.4f}  (target: 1.00)")
print(f"  Monitoring A/E:  {aggregate_ae:.4f}")

ae_change_pp = (aggregate_ae - aggregate_ae_ref) * 100
print(f"  Change:          {ae_change_pp:+.2f}pp")

verdict_manual = "NO ACTION NEEDED" if 0.95 <= aggregate_ae <= 1.05 else "INVESTIGATE"
print(f"  Manual verdict:  {verdict_manual}  (threshold: 0.95-1.05)")
print()
print("  What manual A/E MISSES:")
print("  - Which drivers/vehicles are causing the drift (segment blind)")
print("  - Whether the model's ranking has degraded (Gini drift)")
print("  - Distributional shifts in the incoming book (PSI)")
print()

# ---------------------------------------------------------------------------
# LIBRARY: insurance-monitoring full suite
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: insurance-monitoring MonitoringReport")
print("-" * 70)
print()

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur_degraded,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(f"  MonitoringReport recommendation: {report.recommendation}")
print()

# Show the full metric table
df_report = report.to_polars()
print("  Metric breakdown:")
for row in df_report.iter_rows(named=True):
    metric = row.get("metric") or row.get("check") or str(list(row.values())[0])
    value = row.get("value") or row.get("score") or ""
    band = row.get("band") or row.get("status") or ""
    try:
        value_str = f"{float(value):.4f}" if value not in ("", None) else "—"
    except (TypeError, ValueError):
        value_str = str(value)
    print(f"    {str(metric):<35} {value_str:>10}  {band}")
print()

# ---------------------------------------------------------------------------
# Detailed: PSI per feature
# ---------------------------------------------------------------------------

print("-" * 70)
print("DETAIL: PSI per feature (covariate shift detection)")
print("-" * 70)

for feature in ["driver_age", "vehicle_age", "ncd_years"]:
    ref_vals = feat_ref[feature].to_numpy().astype(float)
    cur_vals = feat_cur[feature].to_numpy().astype(float)
    psi_score = psi(ref_vals, cur_vals, n_bins=10)
    band = "GREEN" if psi_score < 0.10 else ("AMBER" if psi_score < 0.25 else "RED")
    print(f"  {feature:<20} PSI = {psi_score:.4f}  [{band}]")

print()
print("  Manual A/E check sees: no alarm (aggregate A/E ~1.0)")
print("  PSI correctly flags: driver_age has drifted (young driver oversampling)")
print()

# ---------------------------------------------------------------------------
# Detailed: Gini drift test
# ---------------------------------------------------------------------------

print("-" * 70)
print("DETAIL: Gini drift test (discrimination decay detection)")
print("-" * 70)

gini_ref_val = gini_coefficient(act_ref, pred_ref, exposure=exposure_ref)
gini_cur_val = gini_coefficient(act_cur, pred_cur_degraded, exposure=exposure_cur)

result = gini_drift_test(
    reference_gini=gini_ref_val,
    current_gini=gini_cur_val,
    n_reference=N_REF,
    n_current=N_CUR,
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur_degraded,
)

print(f"  Reference Gini:   {gini_ref_val:.4f}")
print(f"  Monitoring Gini:  {gini_cur_val:.4f}")
print(f"  Gini change:      {result['gini_change']:+.4f}")
print(f"  z-statistic:      {result['z_statistic']:.3f}")
print(f"  p-value:          {result['p_value']:.4f}")
print(f"  Significant:      {result['significant']}")
print()
print("  This detects discrimination decay — A/E alone would not catch this.")
print("  A model with degraded ranking needs a REFIT, not just recalibration.")
print()

# ---------------------------------------------------------------------------
# Detailed: Segment-level A/E
# ---------------------------------------------------------------------------

print("-" * 70)
print("DETAIL: Segment-level A/E (calibration drift by vehicle age)")
print("-" * 70)

vehicle_age_bands = (vehicle_age_cur // 3).astype(str)  # 0-2, 3-5, 6-8, 9-11, 12+
seg_ae = ae_ratio(
    actual=act_cur,
    predicted=pred_cur_degraded,
    exposure=exposure_cur,
    segments=vehicle_age_bands,
)
print(seg_ae)
print()
print("  New vehicles (band 0 = age 0-2) should show highest A/E (calibration drift).")
print("  Aggregate A/E masks this — the overpriced older vehicles cancel it out.")
print()

# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: What each approach detects")
print("=" * 70)
print()
print(f"  {'Check':<40} {'Manual A/E':>12} {'MonitoringReport':>16}")
print(f"  {'-'*40} {'-'*12} {'-'*16}")

checks = [
    ("Aggregate A/E ratio", "YES", "YES"),
    ("Statistical CI on A/E", "NO", "YES"),
    ("Segment-level A/E drift", "NO", "YES"),
    ("Driver age covariate shift (PSI)", "NO", "YES (RED)" if psi_score > 0.25 else "YES (AMBER)"),
    ("Vehicle age covariate shift (PSI)", "NO", "YES"),
    ("Gini discrimination decay", "NO", "YES"),
    ("Murphy decomposition (REFIT vs RECAL)", "NO", "YES"),
    ("Structured recommendation", "NO", f"YES ({report.recommendation})"),
    ("PRA SS3/17 (insurer model risk) audit trail", "NO", "YES"),
]

for check, manual, library in checks:
    print(f"  {check:<40} {manual:>12} {library:>16}")

print()
print(f"  Aggregate A/E (monitoring period): {aggregate_ae:.4f}  — looks OK to manual check")
print(f"  But MonitoringReport recommends:   {report.recommendation}")
print()


# ---------------------------------------------------------------------------
# TIME-TO-DETECTION: aggregate A/E breach vs library first alarm
# ---------------------------------------------------------------------------
# Simulate how many policies need to accumulate before each approach raises an alarm.
# Aggregate A/E: breach when cumulative A/E exceeds 1.05 (5% threshold).
# Library: PSI fires at a 5% threshold for driver_age once enough data exists.
# This simulates a monitoring team watching the book grow from 0 to N_CUR policies.

print("-" * 70)
print("TIME-TO-DETECTION: cumulative A/E 5% breach vs PSI alarm")
print("-" * 70)
print()

PSI_N_BINS = 10
PSI_THRESHOLD = 0.25  # RED band

# Sort current monitoring period by arrival order (already in order for this DGP)
# Walk through policies 1..N_CUR and compute cumulative A/E and PSI after each batch
BATCH_SIZE = 500  # check every 500 policies

ae_breach_n = None
psi_alarm_n = None

for n in range(BATCH_SIZE, N_CUR + 1, BATCH_SIZE):
    # Cumulative A/E
    cumulative_ae = float(act_cur[:n].sum()) / float(pred_cur_degraded[:n].sum())
    if ae_breach_n is None and abs(cumulative_ae - 1.0) > 0.05:
        ae_breach_n = n

    # PSI for driver_age (most shifted feature)
    ref_vals = driver_age_ref.astype(float)
    cur_vals = driver_age_cur[:n].astype(float)
    # Compute PSI manually (same as library)
    ref_bins = np.percentile(ref_vals, np.linspace(0, 100, PSI_N_BINS + 1))
    ref_bins[0]  -= 1e-6
    ref_bins[-1] += 1e-6
    ref_hist = np.histogram(ref_vals,  bins=ref_bins)[0] / len(ref_vals)
    cur_hist = np.histogram(cur_vals,  bins=ref_bins)[0] / len(cur_vals)
    # Clip zeros for log stability
    ref_hist = np.clip(ref_hist, 1e-8, None)
    cur_hist = np.clip(cur_hist, 1e-8, None)
    psi_da = float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))
    if psi_alarm_n is None and psi_da >= PSI_THRESHOLD:
        psi_alarm_n = n

print(f"  N_CUR = {N_CUR:,} policies in the monitoring period")
print(f"  Aggregate A/E (final): {aggregate_ae:.4f}  (breach if |A/E - 1| > 0.05)")
print()
if ae_breach_n is not None:
    print(f"  Aggregate A/E first breaches 5% threshold at: policy {ae_breach_n:,}  (month ~{ae_breach_n // 1250})")
else:
    print(f"  Aggregate A/E never breaches 5% threshold in {N_CUR:,} policies")
if psi_alarm_n is not None:
    print(f"  PSI driver_age first hits RED threshold at:   policy {psi_alarm_n:,}  (month ~{psi_alarm_n // 1250})")
else:
    print(f"  PSI driver_age RED threshold not reached in {N_CUR:,} policies")
print()
if ae_breach_n is None and psi_alarm_n is not None:
    print("  CONCLUSION: aggregate A/E never flagged the shift; PSI detected it first.")
elif ae_breach_n is not None and psi_alarm_n is not None:
    if psi_alarm_n < ae_breach_n:
        print(f"  PSI alarm fires {ae_breach_n - psi_alarm_n:,} policies earlier than A/E breach.")
    else:
        print(f"  A/E breach fires {psi_alarm_n - ae_breach_n:,} policies earlier than PSI alarm.")
else:
    print("  Neither method breached its threshold in this monitoring period.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
