"""
ModelMonitor — the v1.0.0 integrated monitoring framework.

Scenario:
    A UK motor frequency GBM was trained and validated on 2022–2023 data.
    It is now mid-2025. The pricing team runs their quarterly monitoring check.

    This example covers three scenarios:
    1. No drift — model remains fit for purpose (REDEPLOY)
    2. Global frequency trend — claims inflation, ranking intact (RECALIBRATE)
    3. Book mix shift — old drivers now behave differently, ranking broken (REFIT)

ModelMonitor implements the two-step procedure from Brauer, Menzel & Wüthrich
(2025), arXiv:2510.04556:

    Step 1: Gini ranking drift test (Algorithm 3).
            Has the model's rank-ordering degraded?

    Step 2: GMCB global calibration test (Algorithm 4a).
            Is there a flat level shift fixable by a multiplier?

    Step 3: LMCB local calibration test (Algorithm 4b).
            Are specific cohorts badly miscalibrated?

Decision logic:
    Gini OR LMCB significant  ->  REFIT
    Only GMCB significant     ->  RECALIBRATE
    Nothing significant       ->  REDEPLOY

Commercial context:
    REDEPLOY  — no action needed. Re-run next quarter.
    RECALIBRATE — multiply all predictions by A/E ratio. One-parameter fix.
                  Fast, low model risk. Appropriate for claims inflation.
    REFIT     — rebuild the model on recent data. Weeks of work.
                Only do this when the rank structure is genuinely broken.

    The alpha=0.32 default (one-sigma rule) gives early warnings. A false alarm
    triggers a model review — not an automatic refit. Calibrate to your team's
    cost tolerance. See arXiv:2510.04556 Remark 3.
"""

import numpy as np
from insurance_monitoring import ModelMonitor

rng = np.random.default_rng(42)

# ── Training data ─────────────────────────────────────────────────────────────
# A realistic UK motor portfolio: 15,000 policies, exposure-weighted.
# True frequency driven by driver age and vehicle age.

n_ref = 15_000
exposure_ref = rng.uniform(0.5, 1.5, n_ref)
driver_age_ref = rng.normal(40, 10, n_ref).clip(17, 80)
vehicle_age_ref = rng.uniform(1, 10, n_ref)

# Frequency model: younger drivers, older vehicles → higher claims
true_lam_ref = np.exp(
    -2.8
    + 0.025 * np.maximum(30 - driver_age_ref, 0)
    + 0.04 * vehicle_age_ref
)
y_hat_ref = true_lam_ref                           # model was well-calibrated at training
counts_ref = rng.poisson(exposure_ref * true_lam_ref)
y_ref = counts_ref / exposure_ref                  # observed claim rates (per policy-year)

# Fit the monitor on reference data. Algorithm 2: bootstrap Gini distribution.
monitor = ModelMonitor(
    distribution="poisson",
    n_bootstrap=400,    # 400 is fast; use 1000+ for formal governance reporting
    alpha_gini=0.32,    # one-sigma rule — early warning for ranking drift
    alpha_global=0.32,
    alpha_local=0.32,
    random_state=0,
)
monitor.fit(y_ref, y_hat_ref, exposure_ref)

print("ModelMonitor fitted on reference data.")
print(f"  n_ref = {n_ref:,} policies")
print(f"  mean exposure = {exposure_ref.mean():.2f}")
print(f"  mean frequency = {y_ref.mean():.4f}")
print()


# ── Scenario 1: No drift (REDEPLOY) ──────────────────────────────────────────

n_new = 8_000
exposure_new = rng.uniform(0.5, 1.5, n_new)
driver_age_new = rng.normal(40, 10, n_new).clip(17, 80)
vehicle_age_new = rng.uniform(1, 10, n_new)

true_lam_new = np.exp(
    -2.8
    + 0.025 * np.maximum(30 - driver_age_new, 0)
    + 0.04 * vehicle_age_new
)
y_hat_new = np.exp(
    -2.8
    + 0.025 * np.maximum(30 - driver_age_new, 0)
    + 0.04 * vehicle_age_new
)  # model predictions are still correct
counts_new = rng.poisson(exposure_new * true_lam_new)
y_new = counts_new / exposure_new

result = monitor.test(y_new, y_hat_new, exposure_new)

print("── Scenario 1: No drift ────────────────────────────────────────────────")
print(f"  Decision:   {result.decision}")
print(f"  Gini z:     {result.gini_z:.2f}  (p={result.gini_p:.3f})")
print(f"  GMCB score: {result.gmcb_score:.5f}  (p={result.gmcb_p:.3f})")
print(f"  LMCB score: {result.lmcb_score:.5f}  (p={result.lmcb_p:.3f})")
print(f"  Balance:    {result.balance_factor:.3f}")
print()


# ── Scenario 2: Global frequency inflation (RECALIBRATE) ─────────────────────
# Claims trend: all frequencies up 12%, but rank ordering unchanged.
# The fix is a simple multiplier on all predictions — no model rebuild needed.

counts_inflated = rng.poisson(exposure_new * true_lam_new * 1.12)
y_inflated = counts_inflated / exposure_new

result2 = monitor.test(y_inflated, y_hat_new, exposure_new)

print("── Scenario 2: 12% global frequency inflation ──────────────────────────")
print(f"  Decision:   {result2.decision}")
print(f"  Gini z:     {result2.gini_z:.2f}  (p={result2.gini_p:.3f})")
print(f"  GMCB score: {result2.gmcb_score:.5f}  (p={result2.gmcb_p:.3f})")
print(f"  LMCB score: {result2.lmcb_score:.5f}  (p={result2.lmcb_p:.3f})")
print(f"  Balance:    {result2.balance_factor:.3f}")
if result2.decision == "RECALIBRATE":
    print(f"  Action: multiply all predictions by {result2.balance_factor:.3f}")
print()


# ── Scenario 3: Structural drift — rank ordering broken (REFIT) ───────────────
# Old drivers (>55) have become dramatically riskier due to a change in their
# driving behaviour. The model still predicts them as low-risk.
# This is not fixable by a multiplier — the rank structure is wrong.

driver_age_drift = rng.normal(40, 10, n_new).clip(17, 80)
vehicle_age_drift = rng.uniform(1, 10, n_new)

# New regime: old drivers are high-risk (post-population shift)
# True rate now INCREASES with age above 55
true_lam_drift = np.exp(
    -2.8
    + 0.025 * np.maximum(30 - driver_age_drift, 0)  # original young-driver risk
    + 0.04 * vehicle_age_drift
    + 0.04 * np.maximum(driver_age_drift - 55, 0)   # new: old driver risk premium
)
# Model still uses the old structure (no old-driver premium)
y_hat_drift = np.exp(
    -2.8
    + 0.025 * np.maximum(30 - driver_age_drift, 0)
    + 0.04 * vehicle_age_drift
)
counts_drift = rng.poisson(exposure_new * true_lam_drift)
y_drift = counts_drift / exposure_new

result3 = monitor.test(y_drift, y_hat_drift, exposure_new)

print("── Scenario 3: Structural drift (old-driver risk premium) ──────────────")
print(f"  Decision:   {result3.decision}")
print(f"  Gini z:     {result3.gini_z:.2f}  (p={result3.gini_p:.3f})")
print(f"  GMCB score: {result3.gmcb_score:.5f}  (p={result3.gmcb_p:.3f})")
print(f"  LMCB score: {result3.lmcb_score:.5f}  (p={result3.lmcb_p:.3f})")
print()


# ── Governance summary ────────────────────────────────────────────────────────
# summary() returns a PRA-ready paragraph for model risk committee packs.

print("── Governance summary (Scenario 3) ────────────────────────────────────")
print(result3.summary())
print()

# ── Serialisation ─────────────────────────────────────────────────────────────
# to_dict() gives a JSON-serialisable dict for MLflow or Delta table logging.

d = result3.to_dict()
print("── Serialised result (selected fields) ─────────────────────────────────")
for k in ["decision", "gini_new", "gini_z", "gini_p", "gmcb_score", "gmcb_p",
          "lmcb_score", "lmcb_p", "balance_factor", "n_new"]:
    print(f"  {k}: {d[k]}")
