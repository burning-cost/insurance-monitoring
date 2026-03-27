"""
insurance-monitoring quickstart — fully self-contained example.

Generates a synthetic UK motor pricing book, introduces calibration and
feature drift, then runs a full MonitoringReport to detect it.
"""

import numpy as np
import polars as pl
from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi, csi

rng = np.random.default_rng(42)
n_ref, n_cur = 10_000, 5_000

# --- Reference (training) period ---
driver_age_ref = rng.normal(38, 10, n_ref)
vehicle_age_ref = rng.uniform(1, 10, n_ref)
ncd_years_ref = rng.integers(0, 9, n_ref).astype(float)

lam_ref = np.exp(-2.5 + 0.02 * np.maximum(30 - driver_age_ref, 0) + 0.05 * vehicle_age_ref)
actual_ref = rng.poisson(lam_ref).astype(float)
predicted_ref = lam_ref                             # perfect calibration at training
exposure_ref = rng.uniform(0.5, 1.0, n_ref)

# --- Current monitoring period (drift introduced) ---
# Book has aged: drivers are older and vehicles are newer
driver_age_cur = rng.normal(43, 10, n_cur)          # +5 years shift
vehicle_age_cur = rng.uniform(1, 5, n_cur)           # newer fleet
ncd_years_cur = rng.integers(0, 9, n_cur).astype(float)

lam_cur = np.exp(-2.5 + 0.02 * np.maximum(30 - driver_age_cur, 0) + 0.05 * vehicle_age_cur)
actual_cur = rng.poisson(lam_cur).astype(float)
predicted_cur = lam_ref.mean() * np.ones(n_cur)     # stale model: flat prediction
exposure_cur = rng.uniform(0.5, 1.0, n_cur)

# --- Feature DataFrames for CSI ---
ref_df = pl.DataFrame({
    "driver_age": driver_age_ref,
    "vehicle_age": vehicle_age_ref,
    "ncd_years": ncd_years_ref,
})
cur_df = pl.DataFrame({
    "driver_age": driver_age_cur,
    "vehicle_age": vehicle_age_cur,
    "ncd_years": ncd_years_cur,
})

# --- PSI on a single feature ---
drift = psi(driver_age_ref, driver_age_cur)
print(f"PSI (driver_age): {drift:.3f}  {'RED — significant shift' if drift > 0.25 else 'amber' if drift > 0.10 else 'green'}")

# --- Full monitoring report ---
report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=predicted_ref,
    current_actual=actual_cur,
    current_predicted=predicted_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=ref_df,
    feature_df_current=cur_df,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(f"\nRecommendation: {report.recommendation}")
print("\nMetric summary:")
print(report.to_polars().filter(
    pl.col("metric").is_in(["ae_ratio", "gini_current", "gini_change", "recommendation"])
))
