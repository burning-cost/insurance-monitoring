# Databricks notebook source
# MAGIC %md
# MAGIC # Conformal Control Charts for Insurance Model Monitoring
# MAGIC
# MAGIC **Reference:** Burger (2025), "Distribution-Free Process Monitoring with Conformal Prediction", arXiv:2512.23602
# MAGIC
# MAGIC Classical Shewhart charts assume normal residuals. Insurance losses are not normal —
# MAGIC they are heteroskedastic Poisson-Gamma or Tweedie. The parametric control limits are
# MAGIC wrong by construction.
# MAGIC
# MAGIC Conformal control charts replace parametric 3-sigma limits with a calibrated quantile
# MAGIC threshold derived from in-control data. The false alarm rate is controlled at alpha
# MAGIC exactly in finite samples under exchangeability. No distributional assumptions required.
# MAGIC
# MAGIC This notebook covers:
# MAGIC 1. Univariate relative-residual monitoring on a simulated motor frequency book
# MAGIC 2. Segment-level A/E monitoring replacing the informal "AE > 1.15" convention
# MAGIC 3. Multivariate model health monitoring combining PSI, A/E, and Gini metrics
# MAGIC 4. Governance output: summary text and export to Polars DataFrame

# COMMAND ----------

# MAGIC %pip install polars>=1.0 scipy>=1.12 matplotlib scikit-learn

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Databricks

from insurance_monitoring import (
    ConformalControlChart,
    MultivariateConformalMonitor,
    ConformalChartResult,
)

print(f"insurance_monitoring loaded successfully")
print(f"ConformalControlChart: {ConformalControlChart}")
print(f"MultivariateConformalMonitor: {MultivariateConformalMonitor}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Motor Frequency Book
# MAGIC
# MAGIC We simulate 24 months of policy-level frequency data. The first 18 months are in-control
# MAGIC (calibration period). Months 19-21 are in-control (testing the chart holds). Months 22-24
# MAGIC introduce a 30% mean shift (claims inflation).

# COMMAND ----------

rng = np.random.default_rng(42)

# 2,000 policies per month; Poisson frequency with mean 0.08
n_policies = 2000
n_cal_months = 18
n_test_months = 6  # 3 in-control + 3 out-of-control

# Simulate calibration period: 18 months in-control
# NCS = relative residual |actual - predicted| / predicted, aggregated per month
cal_ae_ratios = []
for _ in range(n_cal_months):
    # Policy-level actual claims
    actual = rng.poisson(lam=0.08, size=n_policies).astype(float)
    predicted = np.full(n_policies, 0.08)
    monthly_ncs = ConformalControlChart.ncs_relative_residual(actual, predicted)
    # Use the mean NCS across policies as the segment-level statistic
    cal_ae_ratios.append(float(np.mean(monthly_ncs)))

cal_scores = np.array(cal_ae_ratios)
print(f"Calibration NCS: mean={cal_scores.mean():.4f}, std={cal_scores.std():.4f}")

# Fit the conformal control chart at 5% FAR
chart = ConformalControlChart(alpha=0.05).fit(cal_scores)
print(f"\nControl limit (threshold): {chart.threshold_:.6f}")
print(f"This replaces the ad-hoc 'AE > 1.15' convention with a calibrated threshold.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Monitor New Periods
# MAGIC
# MAGIC Months 19-21: in-control (same DGP as calibration).
# MAGIC Months 22-24: out-of-control (30% mean shift — claims inflation).

# COMMAND ----------

# Months 19-21: in-control
test_in_control = []
for _ in range(3):
    actual = rng.poisson(lam=0.08, size=n_policies).astype(float)
    predicted = np.full(n_policies, 0.08)
    monthly_ncs = ConformalControlChart.ncs_relative_residual(actual, predicted)
    test_in_control.append(float(np.mean(monthly_ncs)))

# Months 22-24: out-of-control (30% claims inflation)
test_out_of_control = []
for _ in range(3):
    actual = rng.poisson(lam=0.104, size=n_policies).astype(float)  # 0.08 * 1.30
    predicted = np.full(n_policies, 0.08)
    monthly_ncs = ConformalControlChart.ncs_relative_residual(actual, predicted)
    test_out_of_control.append(float(np.mean(monthly_ncs)))

monitor_scores = np.array(test_in_control + test_out_of_control)
result = chart.monitor(monitor_scores)

print("Monthly NCS and p-values:")
print(f"{'Month':>6} {'NCS':>8} {'p-value':>8} {'Alarm':>6}")
period_labels = [f"M{i+19}" for i in range(6)]
for i, (month, ncs, pval, alarm) in enumerate(zip(
    period_labels, result.scores, result.p_values, result.is_alarm
)):
    flag = "*** ALARM ***" if alarm else ""
    print(f"{month:>6} {ncs:>8.4f} {pval:>8.4f} {str(alarm):>6}  {flag}")

print(f"\nControl limit: {result.threshold:.4f}")
print(f"Alarms: {result.n_alarms}/{len(result.scores)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Governance Output

# COMMAND ----------

print("=== GOVERNANCE SUMMARY ===\n")
print(result.summary())
print()

# Export to Polars for MI pack
df = result.to_polars()
df = df.with_columns(pl.lit(period_labels).alias("period"))
print("\nExport DataFrame:")
print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Segment-Level A/E Monitoring
# MAGIC
# MAGIC The informal "AE > 1.15 = investigate" convention uses a fixed threshold unrelated
# MAGIC to actual A/E variability in your book. The conformal approach calibrates the threshold
# MAGIC to historical A/E variability — a segment with naturally high volatility gets a wider
# MAGIC band; a stable segment gets a tight one.

# COMMAND ----------

# Simulate segment-level A/E ratios over 24 months
# Calibration: 18 months with mean A/E near 1.0 and natural variability
cal_ae = 1.0 + rng.normal(0, 0.05, size=18)  # realistic +/- 5% variation

# Fit using median deviation NCS: |AE - median(AE_cal)|
cal_ncs_ae = ConformalControlChart.ncs_median_deviation(cal_ae)
ae_chart = ConformalControlChart(alpha=0.05).fit(cal_ncs_ae)

cal_median_ae = float(np.median(cal_ae))
print(f"Calibration A/E: mean={cal_ae.mean():.4f}, std={cal_ae.std():.4f}")
print(f"Calibration median: {cal_median_ae:.4f}")
print(f"Conformal control limit on |AE - median|: {ae_chart.threshold_:.4f}")
print(f"  (equivalent to flagging |AE - 1.0| > {ae_chart.threshold_:.4f})")
print(f"  vs. the ad-hoc |AE - 1.0| > 0.15 convention\n")

# Monitor 6 new months
new_ae = np.concatenate([
    1.0 + rng.normal(0, 0.05, size=3),    # in-control
    np.array([1.22, 1.26, 1.30]),          # out-of-control (sharp deterioration)
])
new_ncs_ae = ConformalControlChart.ncs_median_deviation(new_ae, median=cal_median_ae)
ae_result = ae_chart.monitor(new_ncs_ae)

print("A/E monitoring results:")
print(f"{'Month':>6} {'A/E':>6} {'p-value':>8} {'Alarm':>6}")
for i, (ae, pval, alarm) in enumerate(zip(new_ae, ae_result.p_values, ae_result.is_alarm)):
    flag = "*** ALARM ***" if alarm else ""
    print(f"M{i+19:>5} {ae:>6.3f} {pval:>8.4f} {str(alarm):>6}  {flag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multivariate Model Health Monitoring
# MAGIC
# MAGIC Combine PSI, A/E frequency, A/E severity, and Gini coefficient into a single
# MAGIC model health vector. Train an IsolationForest on 12 months of in-control vectors,
# MAGIC then produce a single conformal p-value per month as the overall model health score.
# MAGIC
# MAGIC This is Use Case D from the insurance applicability assessment: a governance dashboard
# MAGIC metric that drops below 0.05 when the model needs formal investigation.

# COMMAND ----------

# Monthly model health vectors: [PSI_age, PSI_area, AE_freq, AE_sev, Gini]
# In-control: all metrics near their stable values

n_train = 12  # months used to train the anomaly detector
n_cal_mv = 6   # held-out months for conformal calibration

# Simulate in-control health vectors
def make_in_control(n, rng):
    """Simulate n months of in-control model health metrics."""
    psi_age  = rng.normal(0.03, 0.01, n)   # PSI around 0.03 (low, in-control)
    psi_area = rng.normal(0.04, 0.01, n)
    ae_freq  = rng.normal(1.00, 0.02, n)   # A/E around 1.0
    ae_sev   = rng.normal(1.00, 0.03, n)
    gini     = rng.normal(0.55, 0.01, n)   # Gini stable
    return np.column_stack([psi_age, psi_area, ae_freq, ae_sev, gini])

def make_out_of_control(n, rng):
    """Simulate n months of deteriorating model health."""
    psi_age  = rng.normal(0.15, 0.02, n)   # PSI spiked (feature drift)
    psi_area = rng.normal(0.18, 0.02, n)
    ae_freq  = rng.normal(1.20, 0.03, n)   # A/E elevated (miscalibration)
    ae_sev   = rng.normal(1.15, 0.04, n)
    gini     = rng.normal(0.48, 0.02, n)   # Gini degraded
    return np.column_stack([psi_age, psi_area, ae_freq, ae_sev, gini])

X_train = make_in_control(n_train, rng)
X_cal_mv = make_in_control(n_cal_mv, rng)

# Fit multivariate monitor (IsolationForest default)
mv_monitor = MultivariateConformalMonitor(alpha=0.05).fit(X_train, X_cal_mv)
print(f"Multivariate monitor fitted.")
print(f"Calibration set size: {mv_monitor.n_cal_}")
print(f"Control limit: {mv_monitor.threshold_:.4f}")

# Monitor 6 months: 3 in-control, 3 out-of-control
X_ic  = make_in_control(3, rng)
X_oc  = make_out_of_control(3, rng)
X_new = np.vstack([X_ic, X_oc])

mv_result = mv_monitor.monitor(X_new)

print("\nMultivariate model health monitoring:")
print(f"{'Month':>6} {'Anomaly NCS':>12} {'p-value':>8} {'Status':>10}")
for i, (ncs, pval, alarm) in enumerate(zip(
    mv_result.scores, mv_result.p_values, mv_result.is_alarm
)):
    status = "OUT OF CONTROL" if alarm else "in control"
    print(f"M{i+1:>5} {ncs:>12.4f} {pval:>8.4f} {status}")

print(f"\n{mv_result.summary()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Key Design Notes for Practitioners
# MAGIC
# MAGIC **Calibration window selection matters.** The statistical guarantee (FAR <= alpha) is
# MAGIC exact only when the calibration data comes from a genuinely stable in-control period.
# MAGIC Including a claims-inflation period in calibration widens the threshold and reduces
# MAGIC sensitivity. This is not automatable — it requires pricing team judgement.
# MAGIC
# MAGIC **Exchangeability, not i.i.d.** The method requires exchangeability (permutation
# MAGIC invariance of the joint distribution), not independence. Monthly statistics on a stable
# MAGIC book satisfy this. Monthly statistics spanning a premium cycle turn or a COVID period
# MAGIC do not.
# MAGIC
# MAGIC **No ARL guarantee.** The paper provides no average run length analysis. We do not know
# MAGIC how many out-of-control periods are needed to achieve a given detection probability.
# MAGIC CUSUM has known ARL formulas; conformal charts do not. Use this alongside CalibrationCUSUM
# MAGIC for complementary coverage.
# MAGIC
# MAGIC **Minimum calibration set size.** At alpha=0.05, n >= 20 is needed for a valid threshold.
# MAGIC At alpha=0.0027 (3-sigma equivalent), n >= 370. Below these thresholds, the threshold
# MAGIC saturates at max(cal_scores) which is conservative.
# MAGIC
# MAGIC **p-value interpretation.** p < 0.05 does not mean "the model has definitely degraded."
# MAGIC It means "this observation is in the most anomalous 5% of what we saw during
# MAGIC calibration." Multiple alarms in a row, or alarms accompanied by business context
# MAGIC (large account, CAT event), are stronger signals.

# COMMAND ----------

print("Demo complete.")
print("\nKey classes demonstrated:")
print("  ConformalControlChart   — univariate relative-residual and A/E monitoring")
print("  MultivariateConformalMonitor — multivariate model health dashboard metric")
print("  ConformalChartResult    — to_polars(), summary() for governance output")
print("\nSee insurance_monitoring.conformal_chart for full API documentation.")
