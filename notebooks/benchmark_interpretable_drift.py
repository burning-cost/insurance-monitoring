# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: InterpretableDriftDetector vs DriftAttributor
# MAGIC
# MAGIC **Library:** `insurance-monitoring` v0.7.0 — `InterpretableDriftDetector` for
# MAGIC feature-attributed model performance drift with exposure weighting and FDR control
# MAGIC
# MAGIC **Baseline:** `DriftAttributor` — the earlier module in the same library
# MAGIC (MSE loss, Bonferroni FWER, no exposure weighting)
# MAGIC
# MAGIC **Date:** 2026-03-21
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## The scenario
# MAGIC
# MAGIC A Poisson frequency pricing model uses five rating factors. Drift is induced in exactly
# MAGIC two (vehicle_age and area) in the monitoring period. The model is stale — its predictions
# MAGIC do not change. The aggregate A/E is elevated, but which features are responsible?
# MAGIC
# MAGIC Both modules implement TRIPODD (Panda et al. 2025, arXiv:2503.06606): permutation-based
# MAGIC attribution with bootstrap Type I error control. InterpretableDriftDetector adds:
# MAGIC
# MAGIC - **Exposure weighting** — correct for mixed policy terms (0.25-year vs 1.0-year policies)
# MAGIC - **FDR control** — Benjamini-Hochberg is more powerful than Bonferroni for d=5 features
# MAGIC - **Poisson deviance loss** — the canonical GLM goodness-of-fit metric
# MAGIC - **Single bootstrap loop** — thresholds and p-values in one pass (halved cost)
# MAGIC - **Subset risk caching** — reference-side risks pre-computed at fit_reference()
# MAGIC
# MAGIC ## DGP
# MAGIC
# MAGIC - Reference: 10,000 policies, well-calibrated Poisson frequency model
# MAGIC - Monitoring: 5,000 policies with drift in vehicle_age (+20% for new vehicles, age<3)
# MAGIC   and area (+25% for urban area, area>0.6). No drift in driver_age, ncb, annual_mileage.
# MAGIC - Exposure mix: monitoring period has 50% short-term policies (0.2-0.4yr),
# MAGIC   reference has 30%. Unweighted analysis overweights short-term policies.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-monitoring

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
from datetime import datetime

import numpy as np
import polars as pl

from insurance_monitoring import InterpretableDriftDetector, DriftAttributor

warnings.filterwarnings("ignore")
print(f"Run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic model and DGP

# COMMAND ----------

class PoissonFrequencyModel:
    """Log-linear Poisson GLM stub. Stale — does not know about monitoring-period drift."""
    INTERCEPT = -2.5
    COEF_DRIVER_AGE = -0.008
    COEF_VEHICLE_AGE = 0.06
    COEF_NCB = -0.07
    COEF_ANNUAL_MILEAGE = 0.15
    COEF_AREA = 0.20

    def predict(self, X):
        log_mu = (
            self.INTERCEPT
            + self.COEF_DRIVER_AGE * X[:, 0]
            + self.COEF_VEHICLE_AGE * X[:, 1]
            + self.COEF_NCB * X[:, 2]
            + self.COEF_ANNUAL_MILEAGE * X[:, 3]
            + self.COEF_AREA * X[:, 4]
        )
        return np.exp(log_mu)


MODEL = PoissonFrequencyModel()
FEATURES = ["driver_age", "vehicle_age", "ncb", "annual_mileage", "area"]
DRIFTED_FEATURES = {"vehicle_age", "area"}

RNG = np.random.default_rng(42)
N_REF = 10_000
N_MON = 5_000
VEHICLE_AGE_DRIFT = 1.20
AREA_DRIFT = 1.25


def generate_features(n, rng):
    return np.column_stack([
        rng.integers(18, 80, n).astype(float),   # driver_age
        rng.integers(0, 15, n).astype(float),     # vehicle_age
        rng.integers(0, 9, n).astype(float),      # ncb
        rng.uniform(0.0, 1.0, n),                 # annual_mileage (normalised)
        rng.uniform(0.0, 1.0, n),                 # area (0=rural, 1=urban)
    ])


# Reference period
X_ref = generate_features(N_REF, RNG)
exposure_ref = np.where(
    RNG.random(N_REF) < 0.3,
    RNG.uniform(0.2, 0.4, N_REF),   # 30% short-term
    RNG.uniform(0.75, 1.0, N_REF),
)
mu_ref = MODEL.predict(X_ref) * exposure_ref
y_ref = RNG.poisson(mu_ref).astype(float)

# Monitoring period
X_mon = generate_features(N_MON, RNG)
exposure_mon = np.where(
    RNG.random(N_MON) < 0.5,
    RNG.uniform(0.2, 0.4, N_MON),   # 50% short-term (more in monitoring)
    RNG.uniform(0.75, 1.0, N_MON),
)
mu_pred_mon = MODEL.predict(X_mon) * exposure_mon

# True risk with drift
true_mu_mon = mu_pred_mon.copy()
new_vehicle_mask = X_mon[:, 1] < 3
urban_mask = X_mon[:, 4] > 0.6
true_mu_mon[new_vehicle_mask] *= VEHICLE_AGE_DRIFT
true_mu_mon[urban_mask] *= AREA_DRIFT
y_mon = RNG.poisson(true_mu_mon).astype(float)

print(f"Reference: n={N_REF:,}, mean exposure={exposure_ref.mean():.3f}yr, A/E={y_ref.sum()/mu_ref.sum():.4f}")
print(f"Monitoring: n={N_MON:,}, mean exposure={exposure_mon.mean():.3f}yr, A/E={y_mon.sum()/mu_pred_mon.sum():.4f}")
print(f"  New vehicles (age<3): {new_vehicle_mask.mean():.1%}, drifted {VEHICLE_AGE_DRIFT}x")
print(f"  Urban (area>0.6):     {urban_mask.mean():.1%}, drifted {AREA_DRIFT}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: DriftAttributor

# COMMAND ----------

import time

t0 = time.time()
attributor = DriftAttributor(
    model=MODEL,
    feature_names=FEATURES,
    loss="mse",
    alpha=0.05,
    n_bootstrap=100,
    random_state=42,
)
attributor.fit(X_ref, y_ref)
result_da = attributor.detect(X_mon, y_mon)
elapsed_da = time.time() - t0

print(f"DriftAttributor ({elapsed_da:.1f}s)")
print(f"  Drift detected:      {result_da.drift_detected}")
print(f"  Attributed features: {result_da.attributed_features}")
print(f"  Error control:       FWER (Bonferroni)")
print()
print(f"  {'Feature':<18} {'Test stat':>10} {'Threshold':>10} {'p-value':>8} {'Drift':>6}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
for feat in FEATURES:
    ts = result_da.test_statistics.get(feat, float("nan"))
    th = result_da.thresholds.get(feat, float("nan"))
    pv = result_da.p_values.get(feat, float("nan"))
    flag = "YES" if feat in result_da.attributed_features else "no"
    truth = " (PLANTED)" if feat in DRIFTED_FEATURES else ""
    print(f"  {feat:<18} {ts:>10.4f} {th:>10.4f} {pv:>8.3f} {flag:>6}{truth}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. InterpretableDriftDetector (exposure-weighted, FDR)

# COMMAND ----------

t0 = time.time()
detector = InterpretableDriftDetector(
    model=MODEL,
    features=FEATURES,
    alpha=0.05,
    loss="poisson_deviance",
    n_bootstrap=200,
    error_control="fdr",
    masking_strategy="mean",
    exposure_weighted=True,
    random_state=42,
)
detector.fit_reference(X_ref, y_ref, weights=exposure_ref)
result_idd = detector.test(X_mon, y_mon, weights=exposure_mon)
elapsed_idd = time.time() - t0

print(f"InterpretableDriftDetector ({elapsed_idd:.1f}s)")
print(f"  Drift detected:      {result_idd.drift_detected}")
print(f"  Attributed features: {result_idd.attributed_features}")
print(f"  Error control:       {result_idd.error_control.upper()} (Benjamini-Hochberg)")
print()
print(f"  {'Feature':<18} {'Test stat':>10} {'BH thresh':>10} {'p-value':>8} {'Drift':>6}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
for row in result_idd.feature_ranking.iter_rows(named=True):
    feat = row["feature"]
    ts = row["test_statistic"]
    th = row["threshold"]
    pv = row["p_value"]
    flag = "YES" if row["drift_attributed"] else "no"
    truth = " (PLANTED)" if feat in DRIFTED_FEATURES else ""
    print(f"  {feat:<18} {ts:>10.4f} {th:>10.4f} {pv:>8.3f} {flag:>6}{truth}")

print()
print(f"Summary: {result_idd.summary()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Exposure weighting comparison

# COMMAND ----------

t0 = time.time()
detector_unw = InterpretableDriftDetector(
    model=MODEL,
    features=FEATURES,
    alpha=0.05,
    loss="poisson_deviance",
    n_bootstrap=200,
    error_control="fdr",
    masking_strategy="mean",
    exposure_weighted=False,
    random_state=42,
)
detector_unw.fit_reference(X_ref, y_ref)
result_unw = detector_unw.test(X_mon, y_mon)
elapsed_unw = time.time() - t0

print(f"Weighted IDD ({elapsed_idd:.1f}s):    attributed = {result_idd.attributed_features}")
print(f"Unweighted IDD ({elapsed_unw:.1f}s):  attributed = {result_unw.attributed_features}")
print()
print("P-value comparison (weighted vs unweighted):")
print(f"  {'Feature':<18} {'Weighted p':>12} {'Unweighted p':>14}")
print(f"  {'-'*18} {'-'*12} {'-'*14}")
for feat in FEATURES:
    pv_w = result_idd.p_values[feat]
    pv_u = result_unw.p_values[feat]
    truth = " (PLANTED)" if feat in DRIFTED_FEATURES else ""
    print(f"  {feat:<18} {pv_w:>12.3f} {pv_u:>14.3f}{truth}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. FDR vs Bonferroni

# COMMAND ----------

t0 = time.time()
detector_bonf = InterpretableDriftDetector(
    model=MODEL,
    features=FEATURES,
    alpha=0.05,
    loss="poisson_deviance",
    n_bootstrap=200,
    error_control="fwer",
    masking_strategy="mean",
    exposure_weighted=True,
    random_state=42,
)
detector_bonf.fit_reference(X_ref, y_ref, weights=exposure_ref)
result_bonf = detector_bonf.test(X_mon, y_mon, weights=exposure_mon)
elapsed_bonf = time.time() - t0

print("Multiple testing comparison (d=5 features, alpha=0.05):")
print(f"  Bonferroni per-test alpha: {0.05/5:.3f}")
print(f"  BH per-test alpha (rank-1): {1 * 0.05/5:.3f}")
print()
print(f"  IDD FDR (BH):          attributed = {result_idd.attributed_features}")
print(f"  IDD FWER (Bonferroni): attributed = {result_bonf.attributed_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary

# COMMAND ----------

print("=" * 68)
print("SUMMARY: DriftAttributor vs InterpretableDriftDetector")
print("=" * 68)
print(f"  Ground truth: drift in {sorted(DRIFTED_FEATURES)}")
print()

da_correct = set(result_da.attributed_features) == DRIFTED_FEATURES
idd_correct = set(result_idd.attributed_features) == DRIFTED_FEATURES

print(f"  {'Check':<40} {'DriftAttributor':>15} {'IDD (BH)':>10}")
print(f"  {'-'*40} {'-'*15} {'-'*10}")

rows = [
    ("Drift detected",
     str(result_da.drift_detected), str(result_idd.drift_detected)),
    ("vehicle_age flagged",
     "YES" if "vehicle_age" in result_da.attributed_features else "no",
     "YES" if "vehicle_age" in result_idd.attributed_features else "no"),
    ("area flagged",
     "YES" if "area" in result_da.attributed_features else "no",
     "YES" if "area" in result_idd.attributed_features else "no"),
    ("False positives",
     str(len([f for f in result_da.attributed_features if f not in DRIFTED_FEATURES])),
     str(len([f for f in result_idd.attributed_features if f not in DRIFTED_FEATURES]))),
    ("Attribution correct",
     "YES" if da_correct else "NO",
     "YES" if idd_correct else "NO"),
    ("Exposure weighting", "NO", "YES"),
    ("Loss function", "MSE", "Poisson deviance"),
    ("Error control", "Bonferroni", "BH (FDR)"),
    ("Run time", f"{elapsed_da:.1f}s", f"{elapsed_idd:.1f}s"),
]

for check, da, idd in rows:
    print(f"  {check:<40} {da:>15} {idd:>10}")

print()
print(f"  Reference: {N_REF:,} policies  |  Monitoring: {N_MON:,} policies")
print(f"  Drift: vehicle_age +{int((VEHICLE_AGE_DRIFT-1)*100)}% (age<3), area +{int((AREA_DRIFT-1)*100)}% (area>0.6)")
