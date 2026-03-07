# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-monitoring: Model Drift Detection for Insurance Pricing
# MAGIC
# MAGIC This notebook demonstrates the full monitoring workflow for a UK motor
# MAGIC insurance frequency model.
# MAGIC
# MAGIC **Scenario**: A Poisson frequency model was fitted on 2022-2023 data.
# MAGIC We now run monitoring on Q1 2025 data to check for drift.
# MAGIC
# MAGIC Three checks:
# MAGIC 1. Feature drift (CSI heatmap across rating factors)
# MAGIC 2. Calibration (A/E ratio overall and by segment)
# MAGIC 3. Discrimination (Gini drift test, arXiv 2510.04556)

# COMMAND ----------
# MAGIC %pip install insurance-monitoring polars scipy

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import numpy as np
import polars as pl

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic motor insurance dataset
# ---------------------------------------------------------------------------
# Training (reference) period: 50,000 policy-years
# Monitoring (current) period: 15,000 policy-years
# We introduce drift in Q1 2025:
#   - Portfolio has aged (driver_age +3 years on average)
#   - Vehicle age distribution has shifted (newer vehicles)
#   - True claim rate has increased for young drivers (concept drift)

N_REF = 50_000
N_CUR = 15_000

print("Generating synthetic motor insurance data...")

# --- Reference period ---
driver_age_ref = rng.normal(38, 10, N_REF).clip(17, 80)
vehicle_age_ref = rng.exponential(5, N_REF).clip(0, 20)
ncd_years_ref = rng.integers(0, 10, N_REF).astype(float)
exposure_ref = rng.uniform(0.5, 1.0, N_REF)  # earned car-years

# True Poisson rate depends on rating factors
log_rate_ref = (
    -3.0
    - 0.02 * (driver_age_ref - 38)   # age: older = lower rate
    + 0.03 * vehicle_age_ref          # vehicle age: older = higher rate
    - 0.05 * ncd_years_ref            # NCD: more = lower rate
)
true_rate_ref = np.exp(log_rate_ref)
actual_ref = rng.poisson(true_rate_ref * exposure_ref).astype(float)
predicted_ref = true_rate_ref  # model is perfectly calibrated at training

# --- Current period (with drift) ---
# Portfolio has aged by 3 years on average
driver_age_cur = rng.normal(41, 10, N_CUR).clip(17, 80)  # +3 years
vehicle_age_cur = rng.exponential(3, N_CUR).clip(0, 20)   # newer vehicles
ncd_years_cur = rng.integers(0, 10, N_CUR).astype(float)
exposure_cur = rng.uniform(0.5, 1.0, N_CUR)

# True rate has also changed (concept drift: young drivers more risky now)
log_true_rate_cur = (
    -3.0
    - 0.02 * (driver_age_cur - 38)
    + 0.03 * vehicle_age_cur
    - 0.05 * ncd_years_cur
    + np.where(driver_age_cur < 25, 0.15, 0.0)  # concept drift: young +15%
)
true_rate_cur = np.exp(log_true_rate_cur)
actual_cur = rng.poisson(true_rate_cur * exposure_cur).astype(float)

# Model predictions use the ORIGINAL formula (no concept drift correction)
log_pred_cur = (
    -3.0
    - 0.02 * (driver_age_cur - 38)
    + 0.03 * vehicle_age_cur
    - 0.05 * ncd_years_cur
)
predicted_cur = np.exp(log_pred_cur)  # model doesn't know about concept drift

print(f"Reference period: {N_REF:,} policies, {actual_ref.sum():.0f} claims")
print(f"Current period: {N_CUR:,} policies, {actual_cur.sum():.0f} claims")
print(f"Ref A/E check: {actual_ref.sum() / (predicted_ref * exposure_ref).sum():.3f}")
print(f"Current A/E check: {actual_cur.sum() / (predicted_cur * exposure_cur).sum():.3f}")

# COMMAND ----------
# MAGIC %md ## 1. Feature Drift: CSI Heatmap

# COMMAND ----------
from insurance_monitoring.drift import csi, psi, wasserstein_distance

# Build feature DataFrames
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

csi_result = csi(feat_ref, feat_cur, features=["driver_age", "vehicle_age", "ncd_years"])
print("CSI Heatmap:")
print(csi_result)
print()

# Wasserstein distances for interpretability
for feature in ["driver_age", "vehicle_age", "ncd_years"]:
    d = wasserstein_distance(feat_ref[feature], feat_cur[feature])
    print(f"  {feature}: Wasserstein distance = {d:.2f} (original units)")

# COMMAND ----------
# MAGIC %md ## 2. Calibration: A/E Ratio

# COMMAND ----------
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci, calibration_curve

# Overall A/E with Poisson confidence interval
ae_result = ae_ratio_ci(actual_cur, predicted_cur, exposure=exposure_cur, alpha=0.05)
print("Overall A/E ratio:")
print(f"  A/E = {ae_result['ae']:.4f}  (95% CI: {ae_result['lower']:.4f} – {ae_result['upper']:.4f})")
print(f"  Observed claims: {ae_result['n_claims']:.0f}, Expected: {ae_result['n_expected']:.1f}")
print()

# Segmented A/E by driver age band
age_bands = np.select(
    [driver_age_cur < 25, driver_age_cur < 40, driver_age_cur < 60],
    ["17-24", "25-39", "40-59"],
    default="60+",
)
seg_ae = ae_ratio(actual_cur, predicted_cur, exposure=exposure_cur, segments=age_bands)
print("Segmented A/E by driver age band:")
print(seg_ae.sort("segment"))
print()
print("Note: young drivers (17-24) show elevated A/E — concept drift from whiplash reform")

# Calibration curve
cal_curve = calibration_curve(actual_cur, predicted_cur, n_bins=10)
print("\nCalibration curve (first 5 bins):")
print(cal_curve.head(5))

# COMMAND ----------
# MAGIC %md ## 3. Discrimination: Gini Drift Test

# COMMAND ----------
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test, lorenz_curve

gini_ref = gini_coefficient(actual_ref, predicted_ref, exposure=exposure_ref)
gini_cur = gini_coefficient(actual_cur, predicted_cur, exposure=exposure_cur)
print(f"Reference Gini: {gini_ref:.4f}")
print(f"Current Gini:   {gini_cur:.4f}")
print(f"Change:         {gini_cur - gini_ref:+.4f}")
print()

# Statistical test for Gini drift (arXiv 2510.04556 Theorem 1)
drift_result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=N_REF,
    n_current=N_CUR,
    reference_actual=actual_ref,
    reference_predicted=predicted_ref,
    current_actual=actual_cur,
    current_predicted=predicted_cur,
    reference_exposure=exposure_ref,
    current_exposure=exposure_cur,
    n_bootstrap=200,
)
print("Gini Drift Test (arXiv 2510.04556):")
print(f"  z-statistic: {drift_result['z_statistic']:.3f}")
print(f"  p-value:     {drift_result['p_value']:.4f}")
print(f"  Significant: {drift_result['significant']}")

# COMMAND ----------
# MAGIC %md ## 4. Combined Monitoring Report

# COMMAND ----------
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=predicted_ref,
    current_actual=actual_cur,
    current_predicted=predicted_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    n_bootstrap=200,
)

print("=== MONITORING REPORT ===")
print(f"\nRecommendation: {report.recommendation}")
print("\nFull results table:")
print(report.to_polars())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The monitoring report flags:
# MAGIC - **CSI**: driver_age has drifted (portfolio has aged by ~3 years)
# MAGIC - **A/E ratio**: aggregate near 1.0, but segmented A/E shows young drivers ~15% elevated
# MAGIC - **Gini drift**: possible degradation — concept drift has reduced discrimination
# MAGIC
# MAGIC Decision (arXiv 2510.04556 framework):
# MAGIC - Gini has degraded → **REFIT** recommended
# MAGIC - In practice: first check if recalibration of the young driver band restores A/E,
# MAGIC   then decide whether a partial refit of the age interaction is sufficient or
# MAGIC   whether a full refit on recent data is needed.
# MAGIC
# MAGIC **Next steps**:
# MAGIC 1. Confirm IBNR development is complete for this accident period
# MAGIC 2. Pull SHAP values for young driver band to quantify feature importance shift
# MAGIC 3. Escalate to Head of Pricing with this report

# COMMAND ----------
dbutils.notebook.exit("Demo complete.")
