"""
UK motor frequency model — monthly monitoring scenario.

Scenario:
    A personal lines GBM was trained on 2022–2023 data. It is now Q1 2025.
    The book has aged: the average driver is older, vehicles are newer, and
    NCD distributions have shifted because of two years of claim-free renewals.
    The model's frequency predictions are increasingly stale.

    This script simulates one quarterly monitoring run as a UK pricing team
    would run it: A/E ratio with CI, CSI across all rating factors, Murphy
    decomposition to distinguish calibration drift from discrimination drift,
    and a governance-ready Gini bootstrap test.

Commercial interpretation:
    - A/E > 1.10 (red): the model is underpricing. This needs actioning before
      the next renewal cycle, not at year-end.
    - Gini drift significant (red): the model has lost rank-ordering ability.
      Recalibrating the intercept will not fix this — a refit is required.
    - CSI red on driver_age: the shift in the portfolio age profile is the
      likely root cause. TRIPODD attribution (see tripodd_attribution.py)
      confirms which features are driving the performance loss.
"""

import numpy as np
import polars as pl
from insurance_monitoring import MonitoringReport, GiniDriftBootstrapTest
from insurance_monitoring.drift import psi, csi, wasserstein_distance
from insurance_monitoring.calibration import ae_ratio_ci

rng = np.random.default_rng(2024)

# ── Training period (2022–2023, n=40,000 policies) ──────────────────────────

n_ref = 40_000

driver_age_ref = rng.normal(38, 10, n_ref).clip(17, 80)
vehicle_age_ref = rng.uniform(1, 10, n_ref)
ncd_years_ref = rng.integers(0, 10, n_ref).astype(float)
vehicle_group_ref = rng.integers(1, 20, n_ref).astype(float)

# Frequency model: younger drivers and older vehicles cost more
lam_ref = np.exp(
    -2.8
    + 0.03 * np.maximum(30 - driver_age_ref, 0)
    + 0.04 * vehicle_age_ref
    - 0.05 * ncd_years_ref
    + 0.01 * vehicle_group_ref
)
actual_ref = rng.poisson(lam_ref).astype(float)
predicted_ref = lam_ref  # perfect calibration on training data
exposure_ref = rng.uniform(0.5, 1.0, n_ref)

# ── Current period (Q1 2025, n=12,000 policies) ──────────────────────────────
# Two shifts: (1) book has aged — drivers older, vehicles newer
# (2) model is stale — predictions are from the old calibration

n_cur = 12_000

driver_age_cur = rng.normal(44, 10, n_cur).clip(17, 80)   # +6 years shift
vehicle_age_cur = rng.uniform(1, 6, n_cur)                 # newer fleet
ncd_years_cur = rng.integers(2, 10, n_cur).astype(float)   # more NCD after 2 clean years
vehicle_group_cur = rng.integers(1, 20, n_cur).astype(float)

lam_cur = np.exp(
    -2.8
    + 0.03 * np.maximum(30 - driver_age_cur, 0)
    + 0.04 * vehicle_age_cur
    - 0.05 * ncd_years_cur
    + 0.01 * vehicle_group_cur
)
actual_cur = rng.poisson(lam_cur).astype(float)
# Stale model: still predicts the training-period mean for the old portfolio mix
predicted_cur = np.full(n_cur, lam_ref.mean())
exposure_cur = rng.uniform(0.5, 1.0, n_cur)

# ── Feature DataFrames for CSI ────────────────────────────────────────────────

ref_df = pl.DataFrame({
    "driver_age": driver_age_ref,
    "vehicle_age": vehicle_age_ref,
    "ncd_years": ncd_years_ref,
    "vehicle_group": vehicle_group_ref,
})
cur_df = pl.DataFrame({
    "driver_age": driver_age_cur,
    "vehicle_age": vehicle_age_cur,
    "ncd_years": ncd_years_cur,
    "vehicle_group": vehicle_group_cur,
})

features = ["driver_age", "vehicle_age", "ncd_years", "vehicle_group"]

# ── Step 1: A/E ratio ─────────────────────────────────────────────────────────

ae = ae_ratio_ci(actual_cur, predicted_cur, exposure=exposure_cur)
print("── A/E ratio ──────────────────────────────────────────────────────────")
print(f"  A/E = {ae['ae']:.3f}  95% CI [{ae['lower']:.3f}, {ae['upper']:.3f}]")
print(f"  Actual claims:   {ae['n_claims']:.1f}")
print(f"  Expected claims: {ae['n_expected']:.1f}")
# > 1.0 means the book is experiencing more claims than predicted.
# > 1.05 is typically amber; > 1.10 is red and requires actioning.

# ── Step 2: PSI on key rating factors ─────────────────────────────────────────

print("\n── PSI per factor ─────────────────────────────────────────────────────")
for feat in features:
    val = psi(
        ref_df[feat].to_numpy(),
        cur_df[feat].to_numpy(),
        exposure_weights=exposure_cur,
        reference_exposure=exposure_ref,
    )
    band = "RED" if val > 0.25 else "amber" if val > 0.10 else "green"
    print(f"  {feat:<20} PSI = {val:.3f}  [{band}]")

# Wasserstein gives a more interpretable number for non-technical stakeholders:
wa = wasserstein_distance(driver_age_ref, driver_age_cur)
print(f"\n  driver_age Wasserstein = {wa:.1f} years  (average shift in original units)")

# ── Step 3: CSI heatmap ───────────────────────────────────────────────────────

print("\n── CSI heatmap ────────────────────────────────────────────────────────")
csi_df = csi(ref_df, cur_df, features)
print(csi_df)

# ── Step 4: Gini drift test (governance-ready) ────────────────────────────────

print("\n── Gini discrimination drift ──────────────────────────────────────────")
bt = GiniDriftBootstrapTest(
    training_gini=None,  # will be computed from reference data
    monitor_actual=actual_cur,
    monitor_predicted=predicted_cur,
    monitor_exposure=exposure_cur,
    n_bootstrap=300,
    random_state=42,
)
# Compute reference Gini manually and pass it in
from insurance_monitoring import gini_coefficient
ref_gini = gini_coefficient(actual_ref, predicted_ref, exposure=exposure_ref)
bt = GiniDriftBootstrapTest(
    training_gini=ref_gini,
    monitor_actual=actual_cur,
    monitor_predicted=predicted_cur,
    monitor_exposure=exposure_cur,
    n_bootstrap=300,
    random_state=42,
)
bt_result = bt.test()
print(f"  Reference Gini:  {ref_gini:.3f}")
print(f"  Monitor Gini:    {bt_result.monitor_gini:.3f}  "
      f"CI [{bt_result.ci_lower:.3f}, {bt_result.ci_upper:.3f}]")
print(f"  Change:          {bt_result.gini_change:+.3f}  p={bt_result.p_value:.3f}")
print(f"  Significant:     {bt_result.significant}")
print(f"\n  Governance summary:\n  {bt.summary()}")

# ── Step 5: Full MonitoringReport ─────────────────────────────────────────────

print("\n── Full MonitoringReport ──────────────────────────────────────────────")
report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=predicted_ref,
    current_actual=actual_cur,
    current_predicted=predicted_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=ref_df,
    feature_df_current=cur_df,
    features=features,
    murphy_distribution="poisson",
    gini_bootstrap=True,
    n_bootstrap=300,
)

print(f"\n  Recommendation: {report.recommendation}")
print("\n  Key metrics:")
key_metrics = ["ae_ratio", "gini_current", "gini_change", "murphy_discrimination_pct",
               "murphy_miscalibration_pct", "recommendation"]
print(report.to_polars().filter(pl.col("metric").is_in(key_metrics)))

# ── Interpretation ────────────────────────────────────────────────────────────
print("""
── Commercial interpretation ──────────────────────────────────────────────────
  If A/E is red and Gini drift is not significant:
    → RECALIBRATE: apply a flat loading to the current predictions.
      No full refit required. This is a calibration issue, not a rank-ordering issue.

  If Gini drift is significant (discrimination falling):
    → REFIT: the model's rank-ordering is deteriorating because the book
      mix has shifted. A flat recalibration cannot fix this. Retrain on
      recent data and re-validate before the next rating action.

  Murphy decomposition confirmation:
    MCB% > DSC%  → calibration issue dominates → RECALIBRATE
    DSC%  high   → discrimination loss dominates → REFIT
""")
