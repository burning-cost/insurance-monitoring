# Databricks notebook source
# MAGIC %md
# MAGIC # BAWSMonitor Demo
# MAGIC
# MAGIC ## The problem it solves
# MAGIC
# MAGIC When you run a rolling VaR or ES estimate, you need to decide how many years
# MAGIC of data to include. Use too short a window and you overfit recent volatility.
# MAGIC Use too long a window and you miss genuine regime shifts. Both errors cost you
# MAGIC in a Solvency II backtesting context.
# MAGIC
# MAGIC The standard answer — pick 250 trading days and move on — is arbitrary. BAWS
# MAGIC (Bootstrap Adaptive Window Selection, Li, Lyu, Wang 2026) gives you a
# MAGIC data-driven alternative: at each time step it selects the window that
# MAGIC minimises a block-bootstrapped scoring rule. The scoring rule is the
# MAGIC Fissler-Ziegel score, which is strictly consistent for the joint
# MAGIC (VaR_alpha, ES_alpha) functional. That alignment is the point: you are
# MAGIC selecting a window by the criterion you actually care about.
# MAGIC
# MAGIC Reference: Li, Lyu, Wang (2026). arXiv:2603.01157.

# COMMAND ----------
# MAGIC %pip install insurance-monitoring>=1.1.0 polars matplotlib

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from insurance_monitoring import BAWSMonitor, BAWSResult
from insurance_monitoring.baws import fissler_ziegel_score, asymm_abs_loss

print("insurance-monitoring imported successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic return series with a structural break
# MAGIC
# MAGIC We simulate a 600-observation return series. The first 400 observations come
# MAGIC from a t(8) distribution (moderate tails). At t=401 the distribution shifts
# MAGIC to t(3) — much heavier tails, simulating a transition to a stressed regime.
# MAGIC
# MAGIC The key question: does BAWS shorten its window after the break to respond
# MAGIC to the new regime?

# COMMAND ----------

rng = np.random.default_rng(42)

N_REFERENCE = 300  # initial history fed to fit()
N_UPDATES = 300    # online updates (break at update 100)

# Pre-break: t(8) returns, volatility ~ 2%/day
returns_pre = rng.standard_t(df=8, size=N_REFERENCE + 100) * 0.02

# Post-break: t(3), much heavier tails
returns_post = rng.standard_t(df=3, size=200) * 0.035

update_series = np.concatenate([returns_pre[-100:], returns_post])

print(f"Reference period: {N_REFERENCE} observations (pre-break)")
print(f"Update period:    {len(update_series)} observations")
print(f"  First 100: t(8), sigma=2%  (pre-break)")
print(f"  Last 200: t(3), sigma=3.5% (post-break)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Initialise and run the monitor

# COMMAND ----------

monitor = BAWSMonitor(
    alpha=0.05,
    candidate_windows=[50, 100, 150, 200],  # window lengths to evaluate
    score_type="fissler_ziegel",             # strictly consistent for (VaR, ES)
    n_bootstrap=200,
    random_state=42,
)

# fit() loads the reference history
monitor.fit(returns_pre)
print(f"Monitor fitted on {N_REFERENCE} reference observations")
print(f"Candidate windows: {monitor.candidate_windows}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Step through the update series

# COMMAND ----------

results = monitor.update_batch(update_series)

# Summary
final = results[-1]
print(f"Final step: t={final.time_step}")
print(f"  Selected window:  {final.selected_window}")
print(f"  VaR (5%):         {final.var_estimate:.4f}")
print(f"  ES  (5%):         {final.es_estimate:.4f}")
print(f"  N observations:   {final.n_obs}")
print(f"\nWindow scores at final step:")
for w, s in sorted(final.scores.items()):
    marker = " <-- selected" if w == final.selected_window else ""
    print(f"  w={w:>4}: score={s:.6f}{marker}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Selected window over time
# MAGIC
# MAGIC After the structural break (vertical line), the monitor should tend to
# MAGIC prefer shorter windows — the heavy-tailed post-break data scores better
# MAGIC on recent windows.

# COMMAND ----------

hist = monitor.history()
print(f"History: {len(hist)} rows")
print(hist.head(5))

# COMMAND ----------

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

t = hist["time_step"].to_numpy()
w = hist["selected_window"].to_numpy()
var_series = hist["var_estimate"].to_numpy()
es_series = hist["es_estimate"].to_numpy()

# Panel 1: selected window
axes[0].step(t, w, where="post", color="steelblue", lw=2, label="Selected window")
for cw in monitor.candidate_windows:
    axes[0].axhline(cw, color="grey", lw=0.5, ls="--", alpha=0.5)
axes[0].axvline(100, color="red", lw=1.5, ls="--", label="Structural break (t=100)")
axes[0].set_ylabel("Window length")
axes[0].set_ylim(0, max(monitor.candidate_windows) + 30)
axes[0].legend(loc="upper right")
axes[0].set_title("BAWS: Selected Window Over Time")

# Panel 2: VaR and ES
axes[1].plot(t, var_series, color="navy", lw=1.5, label="VaR (5%)")
axes[1].plot(t, es_series, color="crimson", lw=1.5, alpha=0.8, label="ES (5%)")
axes[1].axvline(100, color="red", lw=1.5, ls="--", label="Structural break")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Return")
axes[1].legend(loc="lower left")
axes[1].set_title("VaR/ES Estimates from BAWS-Selected Window")

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Window frequency summary
# MAGIC
# MAGIC What fraction of time does BAWS spend in each window? In the pre-break
# MAGIC period we expect longer windows to dominate (the distribution is stable).
# MAGIC Post-break we expect a shift toward shorter windows.

# COMMAND ----------

pre_break = hist.filter(pl.col("time_step") <= 100)
post_break = hist.filter(pl.col("time_step") > 100)

def window_freqs(df: pl.DataFrame, label: str):
    total = len(df)
    if total == 0:
        print(f"{label}: no data")
        return
    counts = df.group_by("selected_window").agg(pl.len().alias("count")).sort("selected_window")
    print(f"\n{label} (n={total}):")
    for row in counts.iter_rows(named=True):
        pct = row["count"] / total * 100
        print(f"  w={row['selected_window']:>4}: {row['count']:>5} ({pct:.1f}%)")

window_freqs(pre_break, "Pre-break (t=1-100)")
window_freqs(post_break, "Post-break (t=101-300)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. VaR coverage backtesting
# MAGIC
# MAGIC At 5% VaR, we expect the actual return to exceed VaR in roughly 5% of
# MAGIC periods. Here we check coverage over the full update window.
# MAGIC
# MAGIC Note: this is the standard Solvency II internal model backtesting check.
# MAGIC The PRA requires at least 99 non-exceedances over 500 trading days at the
# MAGIC 99% level. The same logic applies at the 95% level.

# COMMAND ----------

# To check coverage we need the actual return that was observed alongside
# the VaR estimate made the period before
# We can reconstruct: at each step t, VaR is estimated, and update_series[t]
# is the realised return

n_updates = len(update_series)
var_estimates = hist["var_estimate"].to_numpy()

# Exceedance: actual return < VaR (loss exceeds the threshold)
# In our sign convention: negative return = loss, VaR is the alpha-quantile
# (left tail), so exceedance means actual < var_estimate
exceedances = (update_series[:n_updates] < var_estimates[:n_updates])
exceedance_rate = exceedances.mean()

print(f"VaR coverage backtest (alpha={monitor.alpha}):")
print(f"  Exceedance rate: {exceedance_rate:.3f}  (target: {monitor.alpha:.3f})")
print(f"  Number of exceedances: {exceedances.sum()} / {n_updates}")

pre_ex = (update_series[:100] < var_estimates[:100]).mean()
post_ex = (update_series[100:] < var_estimates[100:]).mean()
print(f"\n  Pre-break exceedance rate:  {pre_ex:.3f}")
print(f"  Post-break exceedance rate: {post_ex:.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Score functions — what's being optimised
# MAGIC
# MAGIC The Fissler-Ziegel score is the kernel of the window selection. For a
# MAGIC given window, we compute the mean score on block-bootstrap replicates.
# MAGIC Here we compute it directly on observed data to illustrate the scoring.

# COMMAND ----------

# Score a fixed-window 100-day estimate against the last 200 observations
obs_returns = update_series[-200:]
var_fixed = np.quantile(obs_returns[:100], monitor.alpha)
es_fixed = obs_returns[:100][obs_returns[:100] <= var_fixed].mean()

scores_fz = fissler_ziegel_score(var_fixed, es_fixed, obs_returns, monitor.alpha)

print("Fissler-Ziegel score illustration (fixed 100-day window):")
print(f"  VaR estimate:    {var_fixed:.4f}")
print(f"  ES estimate:     {es_fixed:.4f}")
print(f"  Mean score:      {scores_fz.mean():.6f}")
print(f"  Score shape:     {scores_fz.shape}")

# Tick loss (asymm abs) — for VaR only
scores_al = asymm_abs_loss(var_fixed, obs_returns, monitor.alpha)
print(f"\nAsymmetric abs loss (VaR only):")
print(f"  Mean score:      {scores_al.mean():.6f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Practical notes for Solvency II / IFRS 17 use
# MAGIC
# MAGIC 1. **The window selection criterion and the risk measure target are aligned.**
# MAGIC    Using the Fissler-Ziegel score means you are selecting the window that
# MAGIC    best predicts the joint (VaR, ES) target — not some proxy loss. This is
# MAGIC    the methodologically correct approach.
# MAGIC
# MAGIC 2. **Block bootstrap is essential for financial data.** Return series
# MAGIC    exhibit volatility clustering (GARCH effects) and autocorrelation.
# MAGIC    Treating bootstrap replicates as iid understates the uncertainty in
# MAGIC    short-window estimates. BAWSMonitor uses block bootstrap by default,
# MAGIC    with block length T^(1/3).
# MAGIC
# MAGIC 3. **The selected window adapts to regime changes automatically.** You
# MAGIC    do not need to re-specify the candidate_windows after a market event.
# MAGIC    The scoring function provides the signal.
# MAGIC
# MAGIC 4. **PRA SS3/19 evidence.** For internal model validation, document the
# MAGIC    window selection methodology in your validation report. BAWS provides
# MAGIC    a principled, reproducible audit trail via the history() DataFrame and
# MAGIC    the per-window score columns.
# MAGIC
# MAGIC 5. **Computational cost.** Each update() call runs n_bootstrap replicates
# MAGIC    for each candidate window. With 4 windows and 200 bootstrap replicates,
# MAGIC    that is 800 score evaluations per step. This is fast (milliseconds) for
# MAGIC    typical monitoring window sizes.

# COMMAND ----------

print("Demo complete.")
dbutils.notebook.exit("BAWSMonitor demo completed successfully.")
