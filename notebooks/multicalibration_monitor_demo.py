# Databricks notebook source
# MAGIC # MulticalibrationMonitor Demo
# MAGIC
# MAGIC ## The problem it solves
# MAGIC
# MAGIC A/E ratio tells you whether the *portfolio* is on budget. PSI tells you
# MAGIC whether the *score distribution* has shifted. Neither tells you whether the
# MAGIC model is mispricing a specific combination of subgroup and premium band.
# MAGIC
# MAGIC That is the question MulticalibrationMonitor answers. For each cell
# MAGIC (bin_k, group_l), it checks whether the observed loss rate matches the
# MAGIC model's prediction, with a Poisson z-test and a relative bias gate to
# MAGIC filter out statistically significant but economically trivial deviations.
# MAGIC
# MAGIC Reference: Denuit, Michaelides & Trufin (2026), arXiv:2603.16317.

# COMMAND ----------
# MAGIC %pip install insurance-monitoring>=0.9.3 polars

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import numpy as np
import polars as pl
from insurance_monitoring import MulticalibrationMonitor, MulticalibrationResult

print("insurance-monitoring imported successfully")

# COMMAND ----------
# MAGIC ## Synthetic motor book data
# MAGIC
# MAGIC 50,000 policies split across 4 rating subgroups (e.g. vehicle age bands).
# MAGIC We start with a well-calibrated model, then inject a 30% uplift into one
# MAGIC cell to simulate model degradation in a specific segment.

# COMMAND ----------
rng = np.random.default_rng(42)
N_REF = 50_000
N_MON = 20_000

# Rating subgroups — e.g. vehicle age band
groups_ref = rng.choice(["0-3yr", "4-7yr", "8-12yr", "13yr+"], N_REF)
groups_mon = rng.choice(["0-3yr", "4-7yr", "8-12yr", "13yr+"], N_MON)

# Exposures (car-years, mixed-term policies)
exposure_ref = rng.uniform(0.25, 1.5, N_REF)
exposure_mon = rng.uniform(0.25, 1.5, N_MON)

# Model predictions (claim frequency)
y_pred_ref = rng.gamma(2.0, 0.04, N_REF)
y_pred_mon = rng.gamma(2.0, 0.04, N_MON)

# Reference period: well-calibrated Poisson draws
y_true_ref = rng.poisson(y_pred_ref * exposure_ref).astype(float) / exposure_ref

# Monitor period: well-calibrated EXCEPT for "8-12yr" group at low-risk end
# Simulate the model underestimating claims for older vehicles in the cheap-car band
y_true_mon = rng.poisson(y_pred_mon * exposure_mon).astype(float) / exposure_mon

# Find a specific bin to corrupt — use a rough quantile threshold
LOW_RISK_THRESHOLD = np.quantile(y_pred_mon, 0.3)
biased_mask = (groups_mon == "8-12yr") & (y_pred_mon < LOW_RISK_THRESHOLD)
y_true_mon = y_true_mon.copy()
y_true_mon[biased_mask] *= 1.35  # 35% uplift in this cell

print(f"Reference: {N_REF:,} policies, {biased_mask.sum()} biased in monitor period")
print(f"Biased cell: 'bin ~0-2, group=8-12yr', uplift=35%")

# COMMAND ----------
# MAGIC ## Fit the monitor on reference data

# COMMAND ----------
monitor = MulticalibrationMonitor(
    n_bins=5,             # 5 prediction quantile bands (quintiles)
    min_z_abs=1.96,       # 95% CI z-gate
    min_relative_bias=0.05,  # 5% relative bias gate
    min_exposure=30.0,    # minimum 30 car-years per cell
)

monitor.fit(y_true_ref, y_pred_ref, groups_ref, exposure=exposure_ref)
print(monitor)
print(f"\nBin edges (prediction quantile boundaries):")
for i, e in enumerate(monitor.bin_edges[1:-1]):
    print(f"  Edge {i+1}: {e:.4f}")

# COMMAND ----------
# MAGIC ## Run the monitor on the production period

# COMMAND ----------
result = monitor.update(y_true_mon, y_pred_mon, groups_mon, exposure=exposure_mon)

summary = result.summary()
print(f"Period {summary['period_index']} result:")
print(f"  Cells evaluated: {summary['n_cells_evaluated']}")
print(f"  Cells skipped (insufficient exposure): {summary['n_cells_skipped']}")
print(f"  Alerts: {summary['n_alerts']}")
print(f"  Overall pass: {summary['overall_pass']}")
if summary['worst_cell']:
    wc = summary['worst_cell']
    print(f"\n  Worst cell: bin={wc['bin_idx']}, group={wc['group']}")
    print(f"    Observed:      {wc['observed']:.4f}")
    print(f"    Expected:      {wc['expected']:.4f}")
    print(f"    A/E ratio:     {wc['AE_ratio']:.3f}")
    print(f"    Relative bias: {wc['relative_bias']:+.1%}")
    print(f"    Z-statistic:   {wc['z_stat']:.2f}")

# COMMAND ----------
# MAGIC ## Cell table — all evaluated cells

# COMMAND ----------
# Sort by absolute relative bias to see worst cells first
cell_df = result.cell_table.sort(pl.col("relative_bias").abs(), descending=True)

# Show alerting cells highlighted
print("=== ALERTING CELLS ===")
alerting = cell_df.filter(pl.col("alert"))
if len(alerting) > 0:
    print(alerting.select(["bin_idx", "group", "n_exposure", "observed", "expected",
                            "AE_ratio", "relative_bias", "z_stat"]))
else:
    print("  None")

print("\n=== ALL CELLS (sorted by |relative_bias|) ===")
print(cell_df.select(["bin_idx", "group", "n_exposure", "AE_ratio",
                       "relative_bias", "z_stat", "alert"]).head(20))

# COMMAND ----------
# MAGIC ## Multiple monitoring periods

# COMMAND ----------
print("Simulating 12 monthly monitoring periods...")
print(f"{'Period':>8} {'Alerts':>8} {'Pass':>8}")
print("-" * 28)

for month in range(12):
    rng_m = np.random.default_rng(100 + month)
    n_m = 5000
    groups_m = rng_m.choice(["0-3yr", "4-7yr", "8-12yr", "13yr+"], n_m)
    exposure_m = rng_m.uniform(0.25, 1.5, n_m)
    y_pred_m = rng_m.gamma(2.0, 0.04, n_m)
    y_true_m = rng_m.poisson(y_pred_m * exposure_m).astype(float) / exposure_m

    # Inject bias starting from month 6
    if month >= 6:
        threshold_m = np.quantile(y_pred_m, 0.3)
        biased_m = (groups_m == "8-12yr") & (y_pred_m < threshold_m)
        y_true_m = y_true_m.copy()
        y_true_m[biased_m] *= 1.35

    r_m = monitor.update(y_true_m, y_pred_m, groups_m, exposure=exposure_m)
    s_m = r_m.summary()
    indicator = " <-- BIAS INJECTED" if month >= 6 else ""
    print(f"{s_m['period_index']:>8} {s_m['n_alerts']:>8} {'PASS' if s_m['overall_pass'] else 'FAIL':>8}{indicator}")

# COMMAND ----------
# MAGIC ## Period summary DataFrame

# COMMAND ----------
period_df = monitor.period_summary()
print("Period summary (all updates including initial demo period):")
print(period_df)

# COMMAND ----------
# MAGIC ## Serialisation (for MLflow logging, databases, etc.)

# COMMAND ----------
import json

# to_dict() gives fully JSON-serialisable output
d = result.to_dict()
json_str = json.dumps(d, indent=2)
print(f"to_dict() serialises to {len(json_str):,} characters of JSON")
print(f"Keys: {list(d.keys())}")

# COMMAND ----------
# MAGIC ## Key points for practitioners
# MAGIC
# MAGIC 1. **Bin edges are frozen at fit() time.** The reference period establishes
# MAGIC    what each premium band means. If you refit the model, create a new monitor.
# MAGIC
# MAGIC 2. **The dual gate prevents alert fatigue.** A 6% bias in a 25-car-year cell
# MAGIC    won't alert (z below threshold). A 0.3% bias in a 10,000-car-year cell
# MAGIC    won't alert (below 5% relative bias threshold). Only the combination of
# MAGIC    economically meaningful AND statistically significant deviations alerts.
# MAGIC
# MAGIC 3. **This is complementary to A/E monitoring.** A/E ratio catches overall
# MAGIC    book-level drift. MulticalibrationMonitor catches cross-subsidisation within
# MAGIC    the book — the model might be perfectly balanced overall while quietly
# MAGIC    undercharging one segment and overcharging another.
# MAGIC
# MAGIC 4. **The z-statistic is Poisson-based.** Under the null hypothesis of correct
# MAGIC    calibration, z ~ N(0,1) asymptotically. For small cells (< 100 car-years)
# MAGIC    the normal approximation is weaker; the min_exposure gate is the main
# MAGIC    protection there.

print("Demo complete.")
dbutils.notebook.exit("MulticalibrationMonitor demo completed successfully.")
