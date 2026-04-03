# Databricks notebook source
# MAGIC %md
# MAGIC # ScoreDecompositionTest Demo
# MAGIC
# MAGIC ## The problem it solves
# MAGIC
# MAGIC A pricing actuary runs a quarterly model performance review. The production GLM's
# MAGIC Gini has dropped 1.2 points. There are two plausible explanations:
# MAGIC
# MAGIC **1. Calibration drift.** Claims inflation moved the mean loss and the model's
# MAGIC absolute level is now wrong. Fix: apply a credibility-weighted A/E adjustment.
# MAGIC Cheap. No governance drama.
# MAGIC
# MAGIC **2. Discrimination degradation.** Competitor behaviour or mix-of-business change
# MAGIC means the risk relativities are no longer correctly ordered. Fix: refit with new
# MAGIC features, potentially full redevelopment. Expensive. Six months minimum.
# MAGIC
# MAGIC An aggregate score (Gini, MSE, loss ratio) cannot distinguish these two. They
# MAGIC have identical symptoms — worse numbers — but completely different remediation paths.
# MAGIC
# MAGIC `ScoreDecompositionTest` separates them with p-values. It decomposes any proper
# MAGIC scoring rule into:
# MAGIC
# MAGIC ```
# MAGIC S(F, y) = MCB + UNC - DSC
# MAGIC ```
# MAGIC
# MAGIC - **MCB** (miscalibration): how much worse the model is than its own linearly
# MAGIC   recalibrated version. Zero for a well-calibrated forecast.
# MAGIC - **DSC** (discrimination/resolution): how much better the recalibrated model is
# MAGIC   than the "climate" forecast (the grand mean). Measures ranking skill.
# MAGIC - **UNC** (uncertainty): the inherent variability of the outcome. Irreducible.
# MAGIC
# MAGIC The inference layer — HAC asymptotic standard errors valid under temporal
# MAGIC dependence, two-sample delta tests for MCB(A) vs MCB(B) and DSC(A) vs DSC(B),
# MAGIC and the intersection-union combined test — is what is new (Dimitriadis & Puke,
# MAGIC arXiv:2603.04275). The point estimates alone have existed since Murphy (1977).
# MAGIC
# MAGIC Reference: Dimitriadis & Puke (2026), arXiv:2603.04275.
# MAGIC R reference implementation: https://github.com/marius-cp/SDI

# COMMAND ----------
# MAGIC %pip install "insurance-monitoring>=1.2.0" polars matplotlib statsmodels

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_monitoring.calibration import ScoreDecompositionTest

print("insurance-monitoring imported successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic UK motor data
# MAGIC
# MAGIC We generate a realistic pure-premium dataset. Each policy has a risk factor x
# MAGIC (e.g., a composite of vehicle age and driver age) and an exposure (earned
# MAGIC car-years, varying from 0.1 to 1.2 as in a genuine portfolio with mid-term
# MAGIC adjustments and cancellations).
# MAGIC
# MAGIC The true expected loss is:
# MAGIC ```
# MAGIC E[y | x] = exposure * (300 + 80 * x)
# MAGIC ```
# MAGIC
# MAGIC Claims are Poisson-gamma (a compound distribution approximating aggregate
# MAGIC loss in a UK motor book). We generate both a well-specified model and a
# MAGIC deliberately miscalibrated challenger.

# COMMAND ----------

rng = np.random.default_rng(42)
N = 8000

# Risk factors and exposures
x = rng.uniform(0, 5, N)
exposure = rng.uniform(0.1, 1.2, N)

# True conditional mean
true_mu = exposure * (300 + 80 * x)

# Observed claims: Poisson-gamma compound (overdispersed vs Poisson)
# Shape of individual claims ~ Gamma(shape=1.5)
claim_counts = rng.poisson(exposure * 0.35)  # frequency part
claim_sizes = rng.gamma(shape=1.5, scale=(300 + 80 * x) / 0.35 / 1.5, size=N)
y = claim_counts * claim_sizes * (exposure / exposure.mean())

# Well-calibrated champion model
y_hat_champion = true_mu * (1 + rng.normal(0, 0.03, N))  # small noise, correct level

# Miscalibrated challenger: systematic 20% over-prediction (level wrong)
# but ranking is preserved (discrimination intact)
y_hat_challenger_miscal = y_hat_champion * 1.20

# Undiscriminating challenger: correct mean but poor ranking
# (e.g., only uses age band, ignores vehicle factors)
y_hat_challenger_undiscrim = exposure * 430 * (1 + 0.2 * (x > 2.5).astype(float))
y_hat_challenger_undiscrim *= (1 + rng.normal(0, 0.05, N))

print(f"Portfolio: n={N:,} policies")
print(f"Mean actual: {y.mean():.1f}")
print(f"Mean champion prediction: {y_hat_champion.mean():.1f}")
print(f"Mean miscalibrated prediction: {y_hat_challenger_miscal.mean():.1f}")
print(f"Mean undiscriminating prediction: {y_hat_challenger_undiscrim.mean():.1f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Single-model health check: champion
# MAGIC
# MAGIC First pass: just check whether the champion is calibrated and discriminating.
# MAGIC This is the quarterly single-model review.

# COMMAND ----------

sdi = ScoreDecompositionTest(score_type="mse", exposure=exposure)
result_champion = sdi.fit_single(y, y_hat_champion)

print(result_champion.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC The champion should show:
# MAGIC - MCB p-value large (not significant) — well-calibrated
# MAGIC - DSC p-value small (significant) — has real discriminatory skill

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Miscalibrated challenger vs champion
# MAGIC
# MAGIC The first challenger has correct ranking but is 20% over-predicting.
# MAGIC The aggregate DM test on MSE may or may not be significant. The
# MAGIC delta-MCB test will be. The delta-DSC test will not be.

# COMMAND ----------

result_vs_miscal = sdi.fit_two(y, y_hat_champion, y_hat_challenger_miscal)

print("Champion vs Miscalibrated Challenger")
print("=" * 50)
print(result_vs_miscal.summary())
print()
print("Interpretation:")
if result_vs_miscal.delta_mcb_pvalue < 0.05 and result_vs_miscal.delta_dsc_pvalue > 0.05:
    print("  -> delta_MCB significant, delta_DSC not significant.")
    print("     Models differ in CALIBRATION only.")
    print("     Challenger can be fixed by recalibration — no need to refit.")
elif result_vs_miscal.delta_mcb_pvalue > 0.05 and result_vs_miscal.delta_dsc_pvalue < 0.05:
    print("  -> delta_DSC significant, delta_MCB not significant.")
    print("     Models differ in DISCRIMINATION only.")
    print("     Challenger has better/worse ranking skill.")
else:
    print("  -> Mixed signal. Both components may differ.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Undiscriminating challenger vs champion
# MAGIC
# MAGIC The second challenger has the right aggregate level but poor risk ranking.
# MAGIC The delta-DSC test should detect this; delta-MCB should not be significant.
# MAGIC The DM aggregate test may have lower power than the targeted DSC test.

# COMMAND ----------

result_vs_undiscrim = sdi.fit_two(y, y_hat_champion, y_hat_challenger_undiscrim)

print("Champion vs Undiscriminating Challenger")
print("=" * 50)
print(result_vs_undiscrim.summary())
print()
print("Interpretation:")
if result_vs_undiscrim.delta_dsc_pvalue < 0.05:
    print("  -> delta_DSC significant: champion has materially better discrimination.")
    print("     Undiscriminating challenger should not replace the champion.")
    print("     DM-only test may have missed this if aggregate score difference was small.")
else:
    print("  -> Discrimination difference not statistically significant at 5%.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Temporal monitoring: rolling window decomposition
# MAGIC
# MAGIC The real use case is quarterly monitoring over time. We split the portfolio
# MAGIC into 8 quarters and track MCB and DSC through time.
# MAGIC
# MAGIC In quarters 5-8 we introduce a calibration drift (claims inflation pushes
# MAGIC the mean up by 15%) without touching the ranking. We expect MCB to increase
# MAGIC in those quarters while DSC remains stable.

# COMMAND ----------

rng2 = np.random.default_rng(123)
N_TOTAL = 12000
N_Q = N_TOTAL // 8  # observations per quarter

x_t = rng2.uniform(0, 5, N_TOTAL)
exposure_t = rng2.uniform(0.2, 1.0, N_TOTAL)
true_mu_t = exposure_t * (300 + 80 * x_t)

# Calibration drift starts at quarter 5 (claims inflation +15%)
drift_factor = np.ones(N_TOTAL)
drift_factor[4 * N_Q:] = 1.15

y_t = true_mu_t * drift_factor + rng2.normal(0, 80, N_TOTAL) * np.sqrt(exposure_t)
y_hat_t = true_mu_t * (1 + rng2.normal(0, 0.04, N_TOTAL))  # model unaware of drift

quarters = []
for q in range(8):
    sl = slice(q * N_Q, (q + 1) * N_Q)
    sdi_q = ScoreDecompositionTest(score_type="mse", exposure=exposure_t[sl])
    res_q = sdi_q.fit_single(y_t[sl], y_hat_t[sl])
    quarters.append({
        "quarter": q + 1,
        "mcb": res_q.miscalibration,
        "dsc": res_q.discrimination,
        "unc": res_q.uncertainty,
        "score": res_q.score,
        "mcb_pvalue": res_q.mcb_pvalue,
        "dsc_pvalue": res_q.dsc_pvalue,
    })

print("Quarterly MCB/DSC decomposition:")
print(f"{'Q':>3}  {'MCB':>10}  {'MCB p':>8}  {'DSC':>10}  {'DSC p':>8}  {'UNC':>10}")
print("-" * 60)
for row in quarters:
    sig_mcb = "*" if row["mcb_pvalue"] < 0.05 else " "
    sig_dsc = " " if row["dsc_pvalue"] > 0.05 else "*"
    print(
        f"Q{row['quarter']:1d}  {row['mcb']:10.1f}  "
        f"{row['mcb_pvalue']:7.4f}{sig_mcb}  "
        f"{row['dsc']:10.1f}  "
        f"{row['dsc_pvalue']:7.4f}{sig_dsc}  "
        f"{row['unc']:10.1f}"
    )

print()
print("* = significant at 5%")
print("MCB should rise from Q5; DSC should remain stable.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Visualisation: MCB / DSC / UNC over time

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

qs = [r["quarter"] for r in quarters]
mcbs = [r["mcb"] for r in quarters]
dscs = [r["dsc"] for r in quarters]
uncs = [r["unc"] for r in quarters]
mcb_pvals = [r["mcb_pvalue"] for r in quarters]

ax1 = axes[0]
ax1.plot(qs, mcbs, "o-", color="#e74c3c", label="MCB (miscalibration)")
ax1.plot(qs, dscs, "s--", color="#2e86c1", label="DSC (discrimination)")
ax1.axvline(4.5, color="gray", linestyle=":", alpha=0.7, label="Drift starts")
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Score component")
ax1.set_title("MCB and DSC over time\n(drift introduced at Q5)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in mcb_pvals]
ax2.bar(qs, mcb_pvals, color=colors, edgecolor="white")
ax2.axhline(0.05, color="black", linestyle="--", alpha=0.7, label="alpha=0.05")
ax2.axvline(4.5, color="gray", linestyle=":", alpha=0.7, label="Drift starts")
ax2.set_xlabel("Quarter")
ax2.set_ylabel("MCB p-value")
ax2.set_title("MCB test p-values over time\n(red = significant miscalibration)")
ax2.legend()
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/tmp/score_decomp_temporal.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure saved to /tmp/score_decomp_temporal.png")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Quantile score decomposition for reserve adequacy
# MAGIC
# MAGIC Solvency II Article 120-126 requires internal model validation including VaR/ES backtesting.
# MAGIC For a 75th-percentile reserve adequacy test (whether the model's 75th
# MAGIC percentile correctly covers observed losses), we can use the quantile
# MAGIC score decomposition.
# MAGIC
# MAGIC - MCB = 0 test: standard coverage test (is the 75th percentile correct?)
# MAGIC - DSC = 0 test: new — does the model rank tail scenarios correctly?
# MAGIC
# MAGIC A model can pass the coverage test (correct overall frequency) while failing
# MAGIC the discrimination test (unable to rank which risks are in the tail).

# COMMAND ----------

rng3 = np.random.default_rng(99)
N_RESERVE = 3000

# Claims severity — heavy-tailed (Pareto-like via log-normal)
x_r = rng3.uniform(0, 3, N_RESERVE)
true_75pct = np.exp(3.5 + 0.6 * x_r)  # true 75th percentile

# Good reserve model: approximately correct 75th percentile for each risk
reserve_hat_good = true_75pct * np.exp(rng3.normal(0, 0.1, N_RESERVE))

# Poor model: correct average level but no discrimination (flat reserves)
reserve_hat_flat = np.full(N_RESERVE, np.median(true_75pct)) * np.exp(rng3.normal(0, 0.05, N_RESERVE))

# Observed losses (lognormal with known 75th percentile)
y_reserve = true_75pct * np.exp(rng3.normal(0, 0.5, N_RESERVE) - 0.5 * 0.25)

sdi_q = ScoreDecompositionTest(score_type="quantile", alpha=0.75)

print("Reserve model (75th percentile) — good discriminating model:")
print(sdi_q.fit_single(y_reserve, reserve_hat_good).summary())
print()
print("Reserve model (75th percentile) — flat (undiscriminating) model:")
print(sdi_q.fit_single(y_reserve, reserve_hat_flat).summary())
print()
print("Comparison (good vs flat):")
print(sdi_q.fit_two(y_reserve, reserve_hat_good, reserve_hat_flat).summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. FCA Consumer Duty (PROD 4) context
# MAGIC
# MAGIC Under Consumer Duty, firms must demonstrate models are not producing outcomes
# MAGIC to the detriment of customers. A model showing significant MCB (systematic
# MAGIC over-prediction = overcharging) with non-zero DSC (it can rank risks) is a
# MAGIC clearer governance finding than "Gini dropped 1%".
# MAGIC
# MAGIC The MCB p-value is an auditable artefact: significant or not, with a clear
# MAGIC remediation path. The `.summary()` output is designed to paste directly into
# MAGIC governance documentation.

# COMMAND ----------

# Simulate a model with systematic overcharging
rng4 = np.random.default_rng(7)
N_DUTY = 5000
exposure_duty = rng4.uniform(0.3, 1.0, N_DUTY)
x_duty = rng4.uniform(0, 4, N_DUTY)
true_premium = exposure_duty * (400 + 60 * x_duty)
y_duty = true_premium + rng4.normal(0, 100, N_DUTY) * np.sqrt(exposure_duty)

# Model with 12% systematic overcharge
y_hat_duty = true_premium * 1.12

sdi_duty = ScoreDecompositionTest(score_type="mse", exposure=exposure_duty)
result_duty = sdi_duty.fit_single(y_duty, y_hat_duty)

print("Consumer Duty monitoring — governance output:")
print("=" * 55)
print(result_duty.summary())
print()
print("Recommendation:")
if result_duty.mcb_pvalue < 0.05 and result_duty.dsc_pvalue < 0.05:
    print("  MCB significant: model is systematically miscalibrated.")
    print("  DSC significant: model retains discrimination skill.")
    print("  Action: recalibrate (apply A/E factor). Do NOT refit.")
    print("  This is a PROD 4 finding — document in the Consumer Duty monitoring pack.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Scenario | MCB | DSC | Conclusion | Action |
# MAGIC |----------|-----|-----|------------|--------|
# MAGIC | Calibration drift | High (sig) | Normal | Systematic bias | Recalibrate |
# MAGIC | Discrimination loss | Normal | Low (not sig) | Model lost ranking skill | Refit |
# MAGIC | Both degraded | High | Low | Full model failure | Refit + recal |
# MAGIC | Healthy model | Low (not sig) | Normal (sig) | All good | No action |
# MAGIC
# MAGIC The decomposition is exact: S = MCB + UNC - DSC. The HAC standard errors
# MAGIC handle temporal dependence in quarterly monitoring windows. The two-sample
# MAGIC tests are more powerful than aggregate Diebold-Mariano when models differ
# MAGIC in only one dimension.
# MAGIC
# MAGIC **API:**
# MAGIC ```python
# MAGIC from insurance_monitoring.calibration import ScoreDecompositionTest
# MAGIC
# MAGIC sdi = ScoreDecompositionTest(score_type="mse", exposure=earned_durations)
# MAGIC result = sdi.fit_single(y_actual, y_predicted)
# MAGIC print(result.summary())  # paste into governance doc
# MAGIC
# MAGIC comparison = sdi.fit_two(y_actual, y_champion, y_challenger)
# MAGIC print(comparison.delta_mcb_pvalue)   # MCB different?
# MAGIC print(comparison.delta_dsc_pvalue)   # DSC different?
# MAGIC ```
