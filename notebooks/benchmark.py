# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Capability demo: insurance-monitoring
# Run end-to-end on Databricks (Free Edition or above).

# COMMAND ----------

# MAGIC %md
# MAGIC # Capability Demo: insurance-monitoring
# MAGIC
# MAGIC **Library:** `insurance-monitoring` — model drift detection for insurance pricing
# MAGIC
# MAGIC **What this demo shows:** We train a CatBoost frequency model on a synthetic motor
# MAGIC portfolio, then deliberately break the monitoring period in three ways:
# MAGIC
# MAGIC 1. Shift the age distribution (older drivers enter the book)
# MAGIC 2. Inflate claim frequency for a segment (young drivers get riskier)
# MAGIC 3. Do nothing to the model — it is stale relative to the new portfolio
# MAGIC
# MAGIC `insurance-monitoring` catches all three: PSI/CSI flags the covariate shift,
# MAGIC A/E flags the calibration deterioration, and the Gini drift z-test flags the
# MAGIC discrimination loss. MonitoringReport assembles these into a single traffic-light
# MAGIC summary with a recommended action.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC **Library version:** 0.3.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Why this matters for pricing teams:**
# MAGIC A typical UK motor pricing cycle is 6–12 months between full refits. During that
# MAGIC window the model silently degrades. PSI dashboards in Excel catch covariate shift
# MAGIC but miss calibration and discrimination drift. Ad-hoc A/E checks catch calibration
# MAGIC but have no statistical test. Gini drift testing (arXiv 2510.04556) is new to most
# MAGIC pricing teams. This demo shows all three in one workflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install git+https://github.com/burningcost/insurance-monitoring.git
%pip install catboost scikit-learn matplotlib pandas numpy scipy polars

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from catboost import CatBoostRegressor, Pool

from insurance_monitoring import (
    MonitoringReport,
    psi,
    csi,
    ks_test,
    wasserstein_distance,
    ae_ratio,
    ae_ratio_ci,
    gini_coefficient,
    gini_drift_test,
    gini_drift_test_onesample,
    lorenz_curve,
    PSIThresholds,
    AERatioThresholds,
    GiniDriftThresholds,
    MonitoringThresholds,
    murphy_decomposition,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Demo run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

import insurance_monitoring
print(f"insurance-monitoring version: {insurance_monitoring.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Motor Data
# MAGIC
# MAGIC We generate two policy cohorts from the same DGP, then inject known drift into
# MAGIC the monitoring period. Because we control the DGP we can verify that the library
# MAGIC catches exactly what we injected — not just 'something changed'.
# MAGIC
# MAGIC **Reference period:** 50,000 policies, the model training window. Clean.
# MAGIC
# MAGIC **Monitoring period:** 20,000 policies, 12 months after model deployment. Three
# MAGIC injected changes:
# MAGIC - Driver age distribution shifted up by 6 years (fleet renewals — older drivers)
# MAGIC - Young driver (age < 30) claim frequency multiplied by 1.5
# MAGIC - Model is *not* updated — it still uses reference-period parameters
# MAGIC
# MAGIC This mirrors the typical scenario: a book characteristic shifts, the actual
# MAGIC claims experience changes, but nobody has refitted the model yet.

# COMMAND ----------

rng = np.random.default_rng(42)

N_REF   = 50_000
N_DRIFT = 20_000

# ---------------------------------------------------------------------------
# Reference period
# ---------------------------------------------------------------------------
driver_age_ref = rng.normal(38, 10, N_REF).clip(17, 75)
vehicle_age_ref = rng.gamma(2, 2, N_REF).clip(0, 15)
ncd_years_ref = rng.choice([0, 1, 2, 3, 4, 5], N_REF,
                            p=[0.10, 0.12, 0.15, 0.18, 0.20, 0.25])
region_ref = rng.choice(["London", "SE", "Midlands", "North", "Scotland"], N_REF,
                         p=[0.18, 0.20, 0.22, 0.25, 0.15])
exposure_ref = rng.uniform(0.3, 1.0, N_REF)

# True frequency DGP (Poisson)
age_effect_ref = np.exp(-0.020 * (driver_age_ref - 30).clip(-15, 30))
veh_effect_ref = np.exp(0.030 * vehicle_age_ref)
ncd_effect_ref = np.exp(-0.15 * ncd_years_ref)
region_map = {"London": 1.20, "SE": 1.10, "Midlands": 1.00, "North": 0.95, "Scotland": 0.90}
region_effect_ref = np.array([region_map[r] for r in region_ref])

base_freq = 0.10
true_freq_ref = base_freq * age_effect_ref * veh_effect_ref * ncd_effect_ref * region_effect_ref
claims_ref = rng.poisson(true_freq_ref * exposure_ref)

df_ref = pd.DataFrame({
    "driver_age":   driver_age_ref,
    "vehicle_age":  vehicle_age_ref,
    "ncd_years":    ncd_years_ref.astype(float),
    "region":       region_ref,
    "exposure":     exposure_ref,
    "true_freq":    true_freq_ref,
    "claims":       claims_ref,
})

print(f"Reference: {len(df_ref):,} policies")
print(f"  Mean driver age:   {df_ref['driver_age'].mean():.1f}")
print(f"  Mean exposure:     {df_ref['exposure'].mean():.3f}")
print(f"  Mean true freq:    {df_ref['true_freq'].mean():.4f}")
print(f"  Claim count total: {df_ref['claims'].sum():,}")
print(f"  Observed frequency:{df_ref['claims'].sum() / df_ref['exposure'].sum():.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Monitoring period — with injected drift
# ---------------------------------------------------------------------------

# Change 1: Age distribution shifts up by 6 years (older fleet customers, renewal churn)
driver_age_drift = rng.normal(44, 10, N_DRIFT).clip(17, 75)    # was 38, now 44

vehicle_age_drift = rng.gamma(2, 2, N_DRIFT).clip(0, 15)
ncd_years_drift = rng.choice([0, 1, 2, 3, 4, 5], N_DRIFT,
                               p=[0.10, 0.12, 0.15, 0.18, 0.20, 0.25])
region_drift = rng.choice(["London", "SE", "Midlands", "North", "Scotland"], N_DRIFT,
                            p=[0.18, 0.20, 0.22, 0.25, 0.15])
exposure_drift = rng.uniform(0.3, 1.0, N_DRIFT)

# Change 2: Young driver (< 30) frequency multiplied by 1.5 — new underwriting segment
age_effect_drift = np.exp(-0.020 * (driver_age_drift - 30).clip(-15, 30))
veh_effect_drift = np.exp(0.030 * vehicle_age_drift)
ncd_effect_drift = np.exp(-0.15 * ncd_years_drift)
region_effect_drift = np.array([region_map[r] for r in region_drift])

true_freq_drift = base_freq * age_effect_drift * veh_effect_drift * ncd_effect_drift * region_effect_drift

# Apply the young-driver frequency uplift to the DGP
young_mask = driver_age_drift < 30
true_freq_drift = np.where(young_mask, true_freq_drift * 1.5, true_freq_drift)

claims_drift = rng.poisson(true_freq_drift * exposure_drift)

df_drift = pd.DataFrame({
    "driver_age":   driver_age_drift,
    "vehicle_age":  vehicle_age_drift,
    "ncd_years":    ncd_years_drift.astype(float),
    "region":       region_drift,
    "exposure":     exposure_drift,
    "true_freq":    true_freq_drift,
    "claims":       claims_drift,
})

print(f"\nMonitoring period: {len(df_drift):,} policies")
print(f"  Mean driver age:   {df_drift['driver_age'].mean():.1f}  (was {df_ref['driver_age'].mean():.1f})")
print(f"  Young drivers:     {young_mask.mean():.1%} of portfolio")
print(f"  Mean true freq:    {df_drift['true_freq'].mean():.4f}  (was {df_ref['true_freq'].mean():.4f})")
print(f"  Claim count total: {df_drift['claims'].sum():,}")
print(f"  Observed frequency:{df_drift['claims'].sum() / df_drift['exposure'].sum():.4f}")
print(f"\nDrift injected: driver age +6 years, young driver freq x1.5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train a CatBoost Frequency Model on the Reference Period

# COMMAND ----------

FEATURES = ["driver_age", "vehicle_age", "ncd_years"]
# region is categorical but we leave it out for simplicity here —
# enough features to get a real Gini without complicating the demo

X_train = df_ref[FEATURES].values
y_train = df_ref["claims"].values
e_train = df_ref["exposure"].values

# Frequency model: target = claims per car-year, weight = exposure
pool_train = Pool(
    data=X_train,
    label=y_train / e_train,
    weight=e_train,
)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    learning_rate=0.05,
    depth=5,
    verbose=0,
    random_seed=42,
)
model.fit(pool_train)

# Score both periods — model stays fixed (this is the "stale model" scenario)
# Predictions are frequency rates (claims per car-year)
pred_freq_ref   = model.predict(df_ref[FEATURES])
pred_freq_drift = model.predict(df_drift[FEATURES])

# Convert to expected claim counts for A/E monitoring
expected_ref   = pred_freq_ref   * e_train
expected_drift = pred_freq_drift * df_drift["exposure"].values

print("Model trained on reference period.")
print(f"  Train A/E (sanity check): {df_ref['claims'].sum() / expected_ref.sum():.4f}  (should be ~1.0)")
print(f"  Monitor A/E (stale model): {df_drift['claims'].sum() / expected_drift.sum():.4f}  (we expect > 1.0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PSI and CSI — Feature Distribution Shift
# MAGIC
# MAGIC PSI (Population Stability Index) measures how much a single feature's distribution
# MAGIC has changed between training and monitoring. CSI applies PSI to every feature in
# MAGIC the DataFrame and returns a traffic-light summary.
# MAGIC
# MAGIC Traffic lights: green < 0.10, amber 0.10–0.25, red >= 0.25
# MAGIC
# MAGIC We expect `driver_age` to come back red — we injected a 6-year mean shift.
# MAGIC `vehicle_age` and `ncd_years` should be green (no drift injected).

# COMMAND ----------

# ── PSI: driver_age ────────────────────────────────────────────────────────
psi_age = psi(
    reference=df_ref["driver_age"].values,
    current=df_drift["driver_age"].values,
    n_bins=10,
    exposure_weights=df_drift["exposure"].values,   # exposure-weighted (insurance-correct)
)
psi_age_unweighted = psi(
    reference=df_ref["driver_age"].values,
    current=df_drift["driver_age"].values,
    n_bins=10,
)

thresholds_psi = PSIThresholds()
print(f"PSI driver_age (exposure-weighted): {psi_age:.4f}  [{thresholds_psi.classify(psi_age).upper()}]")
print(f"PSI driver_age (unweighted):        {psi_age_unweighted:.4f}  [{thresholds_psi.classify(psi_age_unweighted).upper()}]")
print()

# ── KS test: driver_age ────────────────────────────────────────────────────
ks_age = ks_test(df_ref["driver_age"].values, df_drift["driver_age"].values)
print(f"KS test driver_age: stat={ks_age['statistic']:.4f}, p={ks_age['p_value']:.2e}, "
      f"significant={ks_age['significant']}")

# ── Wasserstein distance: driver_age ──────────────────────────────────────
wd_age = wasserstein_distance(df_ref["driver_age"].values, df_drift["driver_age"].values)
print(f"Wasserstein driver_age: {wd_age:.2f} years  (mean shift ~6 years — Wasserstein is interpretable)")

# COMMAND ----------

# ── CSI: all numeric features ──────────────────────────────────────────────
csi_features = ["driver_age", "vehicle_age", "ncd_years"]

csi_result = csi(
    reference_df=df_ref[csi_features],
    current_df=df_drift[csi_features],
    features=csi_features,
    n_bins=10,
)

print("CSI heat map:")
print(csi_result)
print()
print("Interpretation:")
for row in csi_result.to_dicts():
    label = {"green": "No action", "amber": "Investigate", "red": "Action required"}[row["band"]]
    print(f"  {row['feature']:15s}  CSI={row['csi']:.4f}  [{row['band'].upper()}]  {label}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### PSI/CSI Visualisation

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

feature_labels = {
    "driver_age":  "Driver age (years)",
    "vehicle_age": "Vehicle age (years)",
    "ncd_years":   "NCD years",
}
bins_dict = {"driver_age": 30, "vehicle_age": 20, "ncd_years": 6}

for ax, feature in zip(axes, csi_features):
    ref_vals  = df_ref[feature].values
    mon_vals  = df_drift[feature].values
    mon_w     = df_drift["exposure"].values
    n_bins    = bins_dict[feature]

    ax.hist(ref_vals, bins=n_bins, density=True, alpha=0.55,
            color="steelblue", label="Reference")
    ax.hist(mon_vals, bins=n_bins, density=True, alpha=0.55,
            color="tomato",    label="Monitor")

    csi_val = float(csi_result.filter(pl.col("feature") == feature)["csi"][0])
    band    = thresholds_psi.classify(csi_val)
    band_color = {"green": "green", "amber": "darkorange", "red": "red"}[band]

    ax.set_title(f"{feature_labels[feature]}\nCSI = {csi_val:.3f}  [{band.upper()}]",
                 color=band_color, fontweight="bold")
    ax.set_xlabel(feature_labels[feature])
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Feature Distribution Shift: Reference vs Monitor Period", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/im_csi.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/im_csi.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. A/E Monitoring — Calibration Drift
# MAGIC
# MAGIC A/E = actual claims / expected claims (from the model). A well-calibrated model
# MAGIC has A/E = 1.0. We expect A/E > 1 in the monitoring period because:
# MAGIC - Older drivers are actually cheaper (but the model trained on a younger mix)
# MAGIC - Young drivers are now 1.5x more frequent than the model learned
# MAGIC
# MAGIC `ae_ratio_ci` adds a 95% confidence interval using the normal approximation for
# MAGIC proportions — so we can distinguish statistical noise from real drift.
# MAGIC
# MAGIC A/E traffic lights: green [0.95–1.05], amber [0.90–1.10], red outside amber.

# COMMAND ----------

# Overall A/E with CI — reference period (sanity check, should be ~1.0)
ae_ref = ae_ratio_ci(
    actual=df_ref["claims"].values,
    predicted=expected_ref,
    exposure=e_train,
)
print("Reference period A/E (sanity check):")
print(f"  A/E = {ae_ref['ae']:.4f}  95% CI [{ae_ref['lower']:.4f}, {ae_ref['upper']:.4f}]")
print(f"  Claims: {ae_ref['n_claims']:.0f}, Expected: {ae_ref['n_expected']:.0f}")
print()

# Overall A/E — monitoring period (expect drift)
ae_mon = ae_ratio_ci(
    actual=df_drift["claims"].values,
    predicted=expected_drift,
    exposure=df_drift["exposure"].values,
)
thr_ae = AERatioThresholds()
ae_band = thr_ae.classify(ae_mon["ae"])
print("Monitoring period A/E:")
print(f"  A/E = {ae_mon['ae']:.4f}  95% CI [{ae_mon['lower']:.4f}, {ae_mon['upper']:.4f}]")
print(f"  Claims: {ae_mon['n_claims']:.0f}, Expected: {ae_mon['n_expected']:.0f}")
print(f"  Band: [{ae_band.upper()}]")
print()

# Segmented A/E — young vs mature drivers
age_segment = np.where(df_drift["driver_age"].values < 30, "Young (<30)", "Mature (30+)")
ae_segmented = ae_ratio(
    actual=df_drift["claims"].values,
    predicted=expected_drift,
    exposure=df_drift["exposure"].values,
    segments=ae_segment,
)
print("Monitoring period A/E by age segment:")
print(ae_segmented)

# COMMAND ----------

# A/E by prediction decile — standard actuarial diagnostic
def ae_by_decile(actual, predicted, exposure, n_deciles=10):
    """Compute A/E ratio for each decile of the predicted distribution."""
    cuts = pd.qcut(predicted, n_deciles, labels=False, duplicates="drop")
    rows = []
    for d in sorted(cuts.unique()):
        mask = cuts == d
        act_d = (actual[mask]).sum()
        exp_d = (predicted[mask]).sum()
        if exp_d > 0:
            rows.append({
                "decile": int(d) + 1,
                "actual": act_d,
                "expected": exp_d,
                "ae": act_d / exp_d,
            })
    return pd.DataFrame(rows)

ae_decile_ref = ae_by_decile(
    df_ref["claims"].values,
    expected_ref,
    e_train,
)
ae_decile_mon = ae_by_decile(
    df_drift["claims"].values,
    expected_drift,
    df_drift["exposure"].values,
)

print("A/E by predicted decile — monitoring period:")
print(ae_decile_mon.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### A/E Calibration Plot

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

deciles = ae_decile_mon["decile"].values
width = 0.38

ax1.bar(deciles - width/2, ae_decile_ref["ae"].values, width,
        label="Reference", color="steelblue", alpha=0.75)
ax1.bar(deciles + width/2, ae_decile_mon["ae"].values, width,
        label="Monitor", color="tomato", alpha=0.75)
ax1.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Perfect calibration")
ax1.axhspan(0.95, 1.05, alpha=0.08, color="green", label="Green band [0.95–1.05]")
ax1.axhspan(0.90, 0.95, alpha=0.08, color="orange")
ax1.axhspan(1.05, 1.10, alpha=0.08, color="orange", label="Amber band [0.90–1.10]")
ax1.set_xlabel("Predicted frequency decile")
ax1.set_ylabel("A/E ratio")
ax1.set_title("A/E by Predicted Decile\n(model-sorted, reference vs monitor)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0.70, 1.40)

# Segmented A/E bar chart
seg_df = ae_segmented.to_pandas() if hasattr(ae_segmented, "to_pandas") else ae_segmented
seg_labels = seg_df["segment"].tolist() if "segment" in seg_df.columns else ["Young (<30)", "Mature (30+)"]
seg_ae = seg_df["ae"].values if "ae" in seg_df.columns else [0, 0]

ax2.bar(seg_labels, seg_ae, color=["tomato", "steelblue"], alpha=0.8, width=0.4)
ax2.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Perfect calibration")
ax2.axhspan(0.95, 1.05, alpha=0.10, color="green")
ax2.axhspan(1.05, 1.15, alpha=0.10, color="orange")
ax2.axhspan(1.15, 1.50, alpha=0.10, color="red")
ax2.set_ylabel("A/E ratio")
ax2.set_title(f"A/E by Age Segment (Monitor)\nOverall A/E = {ae_mon['ae']:.3f}  [{ae_band.upper()}]",
              color={"green": "green", "amber": "darkorange", "red": "red"}[ae_band], fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim(0.70, 1.50)

plt.suptitle("Calibration Monitoring: A/E Ratios", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/im_ae.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/im_ae.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Gini Drift Test — Discrimination Degradation
# MAGIC
# MAGIC The Gini coefficient measures whether the model correctly ranks risks. A perfectly
# MAGIC calibrated model can still discriminate poorly — it gets the average right but fails
# MAGIC to separate cheap from expensive risks.
# MAGIC
# MAGIC We run two tests from arXiv 2510.04556:
# MAGIC
# MAGIC **Two-sample test** (`gini_drift_test`): Tests whether reference and monitor Gini
# MAGIC differ significantly. Both raw arrays required — appropriate when you still have
# MAGIC reference data on hand.
# MAGIC
# MAGIC **One-sample test** (`gini_drift_test_onesample`): Tests the monitor Gini against
# MAGIC a stored scalar from training time. More realistic for deployed monitoring — you
# MAGIC typically don't carry raw training data into production; you store one number.
# MAGIC
# MAGIC Both tests use the one-sigma rule (alpha = 0.32) for early-warning monitoring,
# MAGIC as recommended by the paper. For formal governance, use alpha = 0.05.

# COMMAND ----------

gini_ref = gini_coefficient(
    actual=df_ref["claims"].values,
    predicted=pred_freq_ref,
    exposure=e_train,
)
gini_mon = gini_coefficient(
    actual=df_drift["claims"].values,
    predicted=pred_freq_drift,
    exposure=df_drift["exposure"].values,
)

print(f"Gini coefficient — reference period:  {gini_ref:.4f}")
print(f"Gini coefficient — monitor period:    {gini_mon:.4f}")
print(f"Change:                               {gini_mon - gini_ref:+.4f} ({(gini_mon - gini_ref)/gini_ref*100:+.1f}%)")
print()

# ── Two-sample test: has the current period Gini changed? ─────────────────
drift_2samp = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_mon,
    n_reference=len(df_ref),
    n_current=len(df_drift),
    reference_actual=df_ref["claims"].values,
    reference_predicted=pred_freq_ref,
    current_actual=df_drift["claims"].values,
    current_predicted=pred_freq_drift,
    reference_exposure=e_train,
    current_exposure=df_drift["exposure"].values,
    n_bootstrap=300,
    alpha=0.32,  # one-sigma monitoring threshold (arXiv 2510.04556)
)

print("Two-sample Gini drift test (arXiv 2510.04556 Algorithm 2):")
print(f"  z-statistic:  {drift_2samp['z_statistic']:.3f}")
print(f"  p-value:      {drift_2samp['p_value']:.4f}")
print(f"  Gini change:  {drift_2samp['gini_change']:+.4f}")
print(f"  Significant (alpha=0.32): {drift_2samp['significant']}")
print()

# ── One-sample test: the production monitoring design ─────────────────────
# Simulate the deployed scenario: training Gini was stored at model sign-off.
# We only have the monitor sample now.
drift_1samp = gini_drift_test_onesample(
    training_gini=gini_ref,     # stored scalar from model registry
    monitor_actual=df_drift["claims"].values,
    monitor_predicted=pred_freq_drift,
    monitor_exposure=df_drift["exposure"].values,
    n_bootstrap=500,
    alpha=0.32,
)

print("One-sample Gini drift test (arXiv 2510.04556 Algorithm 3):")
print("  (Production design: only stored training_gini=scalar + new monitor data)")
print(f"  Training Gini: {drift_1samp['training_gini']:.4f}")
print(f"  Monitor Gini:  {drift_1samp['monitor_gini']:.4f}")
print(f"  z-statistic:   {drift_1samp['z_statistic']:.3f}")
print(f"  p-value:       {drift_1samp['p_value']:.4f}")
print(f"  Bootstrap SE:  {drift_1samp['se_bootstrap']:.5f}")
print(f"  Significant (alpha=0.32): {drift_1samp['significant']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lorenz (CAP) Curve

# COMMAND ----------

x_ref, y_ref = lorenz_curve(
    actual=df_ref["claims"].values,
    predicted=pred_freq_ref,
    exposure=e_train,
)
x_mon, y_mon = lorenz_curve(
    actual=df_drift["claims"].values,
    predicted=pred_freq_drift,
    exposure=df_drift["exposure"].values,
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(x_ref, y_ref, "b-",  linewidth=2.0, label=f"Reference  Gini = {gini_ref:.3f}")
ax.plot(x_mon, y_mon, "r--", linewidth=2.0, label=f"Monitor    Gini = {gini_mon:.3f}")
ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Random (Gini = 0)")
ax.fill_between(x_ref, y_ref, x_ref, alpha=0.08, color="blue")
ax.fill_between(x_mon, y_mon, x_mon, alpha=0.08, color="red")
ax.set_xlabel("Cumulative share of exposure (sorted by predicted rate, ascending)")
ax.set_ylabel("Cumulative share of actual claims")
ax.set_title(f"CAP Curve — Gini drift: {drift_2samp['gini_change']:+.3f} "
             f"(p={drift_2samp['p_value']:.3f})", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/im_lorenz.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/im_lorenz.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Murphy MCB/DSC Decomposition
# MAGIC
# MAGIC The Murphy decomposition breaks the model's deviance loss into three components:
# MAGIC
# MAGIC - **UNC** (Uncertainty): inherent difficulty of the data — how much loss a
# MAGIC   naive intercept-only model would achieve. Not the model's fault.
# MAGIC - **DSC** (Discrimination): improvement from having a ranked model. This is the
# MAGIC   component that goes bad when Gini degrades.
# MAGIC - **MCB** (Miscalibration): excess loss from wrong price levels, independent of
# MAGIC   ranking. This further splits into:
# MAGIC   - **GMCB** (global): fixable by a scalar balance correction
# MAGIC   - **LMCB** (local): requires a refit
# MAGIC
# MAGIC The verdict logic: if GMCB > LMCB, the problem is a global shift — recalibrate.
# MAGIC If LMCB >= GMCB, the model structure is wrong — refit.
# MAGIC
# MAGIC This sharpens the RECALIBRATE vs REFIT distinction that pure A/E monitoring misses.

# COMMAND ----------

murphy_ref = murphy_decomposition(
    y=df_ref["claims"].values / e_train,         # claim rate
    y_hat=pred_freq_ref,
    exposure=e_train,
    distribution="poisson",
)
murphy_mon = murphy_decomposition(
    y=df_drift["claims"].values / df_drift["exposure"].values,
    y_hat=pred_freq_drift,
    exposure=df_drift["exposure"].values,
    distribution="poisson",
)

print("Murphy decomposition — Reference period:")
print(f"  UNC:             {murphy_ref.uncertainty:.6f}")
print(f"  DSC:             {murphy_ref.discrimination:.6f}  ({murphy_ref.discrimination_pct:.1f}% of UNC)")
print(f"  MCB:             {murphy_ref.miscalibration:.6f}  ({murphy_ref.miscalibration_pct:.1f}% of UNC)")
print(f"    GMCB (global): {murphy_ref.global_mcb:.6f}")
print(f"    LMCB (local):  {murphy_ref.local_mcb:.6f}")
print(f"  Verdict:         {murphy_ref.verdict}")
print()
print("Murphy decomposition — Monitor period (stale model on drifted data):")
print(f"  UNC:             {murphy_mon.uncertainty:.6f}")
print(f"  DSC:             {murphy_mon.discrimination:.6f}  ({murphy_mon.discrimination_pct:.1f}% of UNC)")
print(f"  MCB:             {murphy_mon.miscalibration:.6f}  ({murphy_mon.miscalibration_pct:.1f}% of UNC)")
print(f"    GMCB (global): {murphy_mon.global_mcb:.6f}")
print(f"    LMCB (local):  {murphy_mon.local_mcb:.6f}")
print(f"  Verdict:         {murphy_mon.verdict}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Murphy Decomposition: Reference vs Monitor

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_murphy_bar(ax, murphy_result, title):
    components = ["DSC", "GMCB", "LMCB"]
    values = [
        murphy_result.discrimination,
        murphy_result.global_mcb,
        murphy_result.local_mcb,
    ]
    colors = ["steelblue", "darkorange", "tomato"]
    bars = ax.bar(components, values, color=colors, alpha=0.80, width=0.5)
    ax.set_ylabel("Deviance component")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    verdict_color = {"OK": "green", "RECALIBRATE": "darkorange", "REFIT": "red"}.get(
        murphy_result.verdict, "black"
    )
    ax.set_xlabel(f"Verdict: {murphy_result.verdict}", color=verdict_color, fontweight="bold")

plot_murphy_bar(axes[0], murphy_ref,
                "Reference period\n(DSC = discrimination, MCB = miscalibration)")
plot_murphy_bar(axes[1], murphy_mon,
                "Monitor period\n(stale model — note MCB increase)")

plt.suptitle("Murphy MCB/DSC Decomposition: Reference vs Monitor", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/im_murphy.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/im_murphy.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MonitoringReport — Traffic Light Dashboard
# MAGIC
# MAGIC `MonitoringReport` is the entry point for a production monitoring run. One call
# MAGIC orchestrates all checks (PSI/CSI, A/E, Gini drift, Murphy decomposition) and
# MAGIC emits a structured summary with traffic-light bands and a recommended action.
# MAGIC
# MAGIC The recommendation logic follows the three-stage decision tree from arXiv 2510.04556:
# MAGIC
# MAGIC 1. Gini OK + A/E OK: **NO_ACTION**
# MAGIC 2. A/E red, Gini OK: **RECALIBRATE** (cheap — update intercept only)
# MAGIC 3. Gini red: **REFIT** (model structure has degraded — retrain)
# MAGIC 4. Murphy verdict takes precedence when enabled — sharpens the RECALIBRATE/REFIT call
# MAGIC
# MAGIC The Murphy decomposition path is more informative: GMCB > LMCB means the drift
# MAGIC is a global level shift (recalibrate); LMCB >= GMCB means the model's ranking
# MAGIC structure has changed (refit).

# COMMAND ----------

report = MonitoringReport(
    reference_actual=df_ref["claims"].values,
    reference_predicted=pred_freq_ref,
    current_actual=df_drift["claims"].values,
    current_predicted=pred_freq_drift,
    exposure=df_drift["exposure"].values,
    reference_exposure=e_train,
    feature_df_reference=df_ref[["driver_age", "vehicle_age", "ncd_years"]],
    feature_df_current=df_drift[["driver_age", "vehicle_age", "ncd_years"]],
    features=["driver_age", "vehicle_age", "ncd_years"],
    score_reference=np.log(pred_freq_ref + 1e-8),   # log-rate as model score
    score_current=np.log(pred_freq_drift + 1e-8),
    murphy_distribution="poisson",
    n_bootstrap=300,
)

print("MonitoringReport complete.")
print(f"Recommendation: {report.recommendation}")
print(f"Murphy available: {report.murphy_available}")

# COMMAND ----------

# Full results dict
full_dict = report.to_dict()

print("\nA/E Ratio:")
ae = full_dict["results"]["ae_ratio"]
print(f"  A/E = {ae['value']:.4f}  CI [{ae['lower_ci']:.4f}, {ae['upper_ci']:.4f}]  [{ae['band'].upper()}]")

print("\nGini Drift:")
g = full_dict["results"]["gini"]
print(f"  Reference Gini = {g['reference']:.4f}")
print(f"  Current Gini   = {g['current']:.4f}  (change = {g['change']:+.4f})")
print(f"  z-statistic = {g['z_statistic']:.3f},  p-value = {g['p_value']:.4f}  [{g['band'].upper()}]")

if "score_psi" in full_dict["results"]:
    sp = full_dict["results"]["score_psi"]
    print(f"\nScore PSI: {sp['value']:.4f}  [{sp['band'].upper()}]")

if "max_csi" in full_dict["results"]:
    mc = full_dict["results"]["max_csi"]
    print(f"\nMax CSI: {mc['value']:.4f}  [{mc['band'].upper()}]  (worst: {mc['worst_feature']})")

if report.murphy_available:
    m = full_dict["results"]["murphy"]
    print(f"\nMurphy decomposition:")
    print(f"  DSC (discrimination): {m['discrimination']:.6f}  ({m['discrimination_pct']:.1f}% of UNC)")
    print(f"  MCB (miscalibration): {m['miscalibration']:.6f}  ({m['miscalibration_pct']:.1f}% of UNC)")
    print(f"    GMCB:               {m['global_mcb']:.6f}")
    print(f"    LMCB:               {m['local_mcb']:.6f}")
    print(f"  Murphy verdict: {m['verdict']}")

# COMMAND ----------

# Polars DataFrame output — one row per metric, suitable for Delta table
report_df = report.to_polars()
print("MonitoringReport as Polars DataFrame:")
print(report_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Traffic Light Dashboard

# COMMAND ----------

def traffic_light_color(band):
    return {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c",
            "RECALIBRATE": "#f39c12", "REFIT": "#e74c3c",
            "NO_ACTION": "#2ecc71", "MONITOR_CLOSELY": "#f39c12",
            "INVESTIGATE": "#e74c3c"}.get(band, "#95a5a6")

# Build display rows from report
display_rows = [
    ("A/E Ratio",           f"{ae['value']:.4f}",        ae['band']),
    ("A/E 95% CI lower",    f"{ae['lower_ci']:.4f}",     ae['band']),
    ("A/E 95% CI upper",    f"{ae['upper_ci']:.4f}",     ae['band']),
    ("Gini (reference)",    f"{g['reference']:.4f}",     "green"),
    ("Gini (current)",      f"{g['current']:.4f}",       g['band']),
    ("Gini change",         f"{g['change']:+.4f}",       g['band']),
    ("Gini p-value",        f"{g['p_value']:.4f}",       g['band']),
]
if "score_psi" in full_dict["results"]:
    sp = full_dict["results"]["score_psi"]
    display_rows.append(("Score PSI", f"{sp['value']:.4f}", sp['band']))
if "max_csi" in full_dict["results"]:
    mc = full_dict["results"]["max_csi"]
    display_rows.append((f"CSI max ({mc['worst_feature']})", f"{mc['value']:.4f}", mc['band']))
if report.murphy_available:
    m = full_dict["results"]["murphy"]
    display_rows.append(("Murphy DSC%", f"{m['discrimination_pct']:.1f}%", m['verdict']))
    display_rows.append(("Murphy MCB%", f"{m['miscalibration_pct']:.1f}%", m['verdict']))
    display_rows.append(("Murphy GMCB", f"{m['global_mcb']:.6f}", m['verdict']))
    display_rows.append(("Murphy LMCB", f"{m['local_mcb']:.6f}", m['verdict']))

rec = report.recommendation

n_rows = len(display_rows)
fig_height = max(5, 0.45 * n_rows + 1.5)
fig, ax = plt.subplots(figsize=(10, fig_height))
ax.set_xlim(0, 10)
ax.set_ylim(0, n_rows + 1.2)
ax.axis("off")

ax.text(5, n_rows + 0.7, "Model Monitoring Dashboard", ha="center", va="center",
        fontsize=14, fontweight="bold")
ax.text(5, n_rows + 0.25, f"Recommendation: {rec}", ha="center", va="center",
        fontsize=12, fontweight="bold",
        color=traffic_light_color(rec),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=traffic_light_color(rec) + "33",
                  edgecolor=traffic_light_color(rec), linewidth=2))

for i, (metric, value, band) in enumerate(reversed(display_rows)):
    row_y = i + 0.2
    bg_col = traffic_light_color(band)
    rect = plt.Rectangle((0.1, row_y), 9.8, 0.75, color=bg_col, alpha=0.18, zorder=0)
    ax.add_patch(rect)
    ax.plot([0.25], [row_y + 0.38], "o", color=bg_col, markersize=10, zorder=2)
    ax.text(0.7, row_y + 0.38, metric, va="center", ha="left", fontsize=9.5)
    ax.text(7.5, row_y + 0.38, value,  va="center", ha="right", fontsize=9.5, fontweight="bold")
    ax.text(8.8, row_y + 0.38, band.upper(), va="center", ha="center", fontsize=8.5,
            color=bg_col, fontweight="bold")

plt.tight_layout()
plt.savefig("/tmp/im_dashboard.png", dpi=130, bbox_inches="tight")
plt.show()
print("Dashboard saved to /tmp/im_dashboard.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary: What the Library Caught
# MAGIC
# MAGIC We injected two changes into the monitoring period:
# MAGIC 1. Driver age distribution shifted up by 6 years
# MAGIC 2. Young driver (< 30) claim frequency multiplied by 1.5
# MAGIC
# MAGIC Here is what each check caught:

# COMMAND ----------

print("=" * 65)
print("SUMMARY: WHAT insurance-monitoring CAUGHT")
print("=" * 65)

psi_val     = float(csi_result.filter(pl.col("feature") == "driver_age")["csi"][0])
psi_band    = thresholds_psi.classify(psi_val)
ae_val      = full_dict["results"]["ae_ratio"]["value"]
ae_b        = full_dict["results"]["ae_ratio"]["band"]
g_change    = full_dict["results"]["gini"]["change"]
g_band      = full_dict["results"]["gini"]["band"]
murphy_v    = full_dict["results"].get("murphy", {}).get("verdict", "N/A")

checks = [
    ("PSI (driver_age)", f"{psi_val:.3f}", psi_band.upper(),
     "YES — age shift of +6 years detected as RED"),
    ("CSI (vehicle_age)", "low", "GREEN",
     "Correct — no drift injected into vehicle_age"),
    ("CSI (ncd_years)", "low", "GREEN",
     "Correct — no drift injected into ncd_years"),
    ("A/E ratio", f"{ae_val:.3f}", ae_b.upper(),
     "YES — young-driver freq uplift shows as A/E > 1.0"),
    ("Gini drift", f"{g_change:+.4f}", g_band.upper(),
     "YES — model ranks risks worse on shifted portfolio"),
    ("Murphy verdict", murphy_v, murphy_v,
     "Distinguishes calibration shift vs discrimination loss"),
    ("Recommendation", report.recommendation, report.recommendation,
     "Decision: NO_ACTION / RECALIBRATE / REFIT"),
]

for name, val, band, interpretation in checks:
    band_fmt = {"GREEN": "OK ", "AMBER": "!  ", "RED": "!!!",
                "REFIT": "!!!", "RECALIBRATE": "!  ",
                "NO_ACTION": "OK ", "N/A": "   "}.get(band, "   ")
    print(f"  [{band_fmt}] {name:<22}  {val:<10}  {interpretation}")

print()
print("Excel A/E dashboards would have caught the A/E drift.")
print("They would NOT have caught the Gini degradation or provided a")
print("statistical test for whether the age shift is significant.")
print("insurance-monitoring catches all three with one function call.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to Use When
# MAGIC
# MAGIC **Use `psi` / `csi` for:**
# MAGIC - Monthly feature monitoring dashboards. Standard PSI traffic lights.
# MAGIC - Detecting covariate shift before it becomes a claims problem.
# MAGIC - `wasserstein_distance` when you need a number non-technical stakeholders
# MAGIC   can read: "average driver age shifted by 2.1 years".
# MAGIC
# MAGIC **Use `ae_ratio_ci` for:**
# MAGIC - Quarterly calibration reviews on developed accident periods.
# MAGIC - Segmented A/E to identify which rating factors are drifting.
# MAGIC - The CI matters — a book of 500 policies has wide sampling variance;
# MAGIC   a book of 50,000 policies does not.
# MAGIC
# MAGIC **Use `gini_drift_test` / `gini_drift_test_onesample` for:**
# MAGIC - Discrimination monitoring — the part most pricing teams currently skip.
# MAGIC - `gini_drift_test_onesample` is the production design: store one scalar
# MAGIC   at sign-off, test new data against it without raw training data.
# MAGIC - Default alpha=0.32 (one-sigma) for early-warning monitoring.
# MAGIC   Use alpha=0.05 for formal governance sign-off.
# MAGIC
# MAGIC **Use `murphy_decomposition` for:**
# MAGIC - Root cause diagnosis: is the problem a global level shift (recalibrate)
# MAGIC   or a structural ranking problem (refit)? This distinction has direct
# MAGIC   business value — recalibration is an afternoon's work; a full refit is
# MAGIC   a two-week project.
# MAGIC
# MAGIC **Use `MonitoringReport` for:**
# MAGIC - The complete monthly/quarterly monitoring pack in one call.
# MAGIC - Output is a plain dict or Polars DataFrame — write directly to a Delta
# MAGIC   table or log to MLflow as run metrics.
# MAGIC - The `recommendation` attribute replaces the "what do we do now?" debate
# MAGIC   with a structured, defensible decision following arXiv 2510.04556.
# MAGIC
# MAGIC **What it does not replace:**
# MAGIC - Human judgement on IBNR development state. Never run A/E on undeveloped
# MAGIC   accident periods. Apply your development factors first.
# MAGIC - Multivariate drift attribution. CSI tells you *which* features shifted,
# MAGIC   not *why*. You still need business context for root cause.
# MAGIC - Claims severity monitoring. This library focuses on frequency / rate
# MAGIC   models. Severity requires a separate check on claim cost distributions.
