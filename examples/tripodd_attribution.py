"""
TRIPODD drift attribution — which rating factors explain the performance loss?

Scenario:
    Following the motor monitoring run (see motor_frequency_monitoring.py),
    the monitoring team knows the model is underperforming. PSI shows driver_age
    and vehicle_age have both shifted. But which shift actually explains why
    the model's deviance has deteriorated? They are correlated — adjusting for
    one may explain away the other.

    InterpretableDriftDetector implements TRIPODD (Panda et al. 2025,
    arXiv:2503.06606) with several extensions:
    - Exposure weighting (correct for mixed policy terms)
    - Poisson deviance loss (canonical for frequency models)
    - Benjamini-Hochberg FDR control (better than Bonferroni for d >= 10)
    - Single bootstrap loop (halves cost vs the two-loop DriftAttributor design)

Commercial interpretation:
    The output is a list of features that, when their distribution shifts,
    explain the observed performance degradation. This is what you bring to
    the refit conversation:

    "driver_age shift explains 68% of the performance loss (p=0.004).
     vehicle_age shift is not significant once driver_age is accounted for.
     Recommendation: retrain with driver_age given appropriate weight in the
     sampling strategy for the refreshed training set."
"""

import numpy as np
import polars as pl
from insurance_monitoring import InterpretableDriftDetector

rng = np.random.default_rng(777)

# ── Reference data (training period) ─────────────────────────────────────────

n_ref = 20_000

driver_age_ref = rng.normal(38, 10, n_ref).clip(17, 80)
vehicle_age_ref = rng.uniform(1, 10, n_ref)
ncd_years_ref = rng.integers(0, 10, n_ref).astype(float)
vehicle_group_ref = rng.integers(1, 20, n_ref).astype(float)
region_ref = rng.integers(1, 12, n_ref).astype(float)  # UK regions

X_ref = pl.DataFrame({
    "driver_age": driver_age_ref,
    "vehicle_age": vehicle_age_ref,
    "ncd_years": ncd_years_ref,
    "vehicle_group": vehicle_group_ref,
    "region": region_ref,
})

lam_ref = np.exp(
    -2.8
    + 0.030 * np.maximum(30 - driver_age_ref, 0)
    + 0.040 * vehicle_age_ref
    - 0.050 * ncd_years_ref
    + 0.008 * vehicle_group_ref
    + 0.002 * region_ref
)
y_ref = rng.poisson(lam_ref).astype(float)
exposure_ref = rng.uniform(0.5, 1.0, n_ref)

# ── Current data (monitoring period) ──────────────────────────────────────────
# Only driver_age and ncd_years have genuinely shifted.
# vehicle_age, vehicle_group, and region are stable.

n_cur = 8_000

driver_age_cur = rng.normal(44, 10, n_cur).clip(17, 80)   # +6 years — meaningful shift
vehicle_age_cur = rng.uniform(1, 10, n_cur)                # STABLE — same as reference
ncd_years_cur = rng.integers(3, 10, n_cur).astype(float)   # shifted — more NCD
vehicle_group_cur = rng.integers(1, 20, n_cur).astype(float)  # STABLE
region_cur = rng.integers(1, 12, n_cur).astype(float)      # STABLE

X_cur = pl.DataFrame({
    "driver_age": driver_age_cur,
    "vehicle_age": vehicle_age_cur,
    "ncd_years": ncd_years_cur,
    "vehicle_group": vehicle_group_cur,
    "region": region_cur,
})

lam_cur = np.exp(
    -2.8
    + 0.030 * np.maximum(30 - driver_age_cur, 0)
    + 0.040 * vehicle_age_cur
    - 0.050 * ncd_years_cur
    + 0.008 * vehicle_group_cur
    + 0.002 * region_cur
)
y_cur = rng.poisson(lam_cur).astype(float)
exposure_cur = rng.uniform(0.5, 1.0, n_cur)

# ── Run InterpretableDriftDetector ────────────────────────────────────────────

print("TRIPODD drift attribution")
print("=" * 60)
print(f"Reference: {n_ref:,} policies  |  Current: {n_cur:,} policies")
print(f"True shifts: driver_age (+6 yrs), ncd_years (+3 yrs)")
print(f"True stable: vehicle_age, vehicle_group, region")
print()

detector = InterpretableDriftDetector(
    error_control="fdr",        # Benjamini-Hochberg — appropriate for 5 features
    loss="poisson_deviance",    # canonical for frequency models
    n_bootstrap=200,
    random_state=42,
)

# Fit on reference data
detector.fit_reference(X_ref, y_ref, weights=exposure_ref)

# Test on current data
result = detector.test(X_cur, y_cur, weights=exposure_cur)

# ── Results ───────────────────────────────────────────────────────────────────

print("Feature attribution results:")
print("-" * 60)
print(f"{'Feature':<20} {'p-value':>10} {'Significant':>14}")
print("-" * 60)
for feat, pval in zip(result.features, result.p_values):
    sig = "YES" if feat in result.significant_features else "no"
    print(f"{feat:<20} {pval:>10.4f} {sig:>14}")

print()
print(f"Significant features ({result.error_control.upper()} controlled):")
for f in result.significant_features:
    print(f"  - {f}")

print()
print("── What to do with this ────────────────────────────────────────────────")
print("""
  1. The significant features tell you WHERE to focus the refit effort.
     In this example, driver_age and ncd_years should be the primary
     variables in the refreshed training sample.

  2. Features that are stable (not significant) can be held at their
     current model estimates — no evidence that their relativities
     have moved.

  3. If a significant feature cannot be addressed in the current model
     structure (e.g., a new driver age band), escalate to model risk
     committee with this result as supporting evidence for a structural
     redevelopment.

  4. Call detector.update_reference(X_cur, y_cur, weights=exposure_cur)
     after a refit to reset the reference baseline. The detector does NOT
     auto-update — explicit control required.
""")
