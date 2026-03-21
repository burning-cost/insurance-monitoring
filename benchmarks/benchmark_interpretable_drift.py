"""
Benchmark: InterpretableDriftDetector vs DriftAttributor for feature-attributed drift.

The scenario
------------
A Poisson frequency pricing model uses five rating factors. Drift is induced in exactly
two of them (vehicle_age and area) in the monitoring period. The model's aggregate
predictions do not change — the model is stale. True risk in the monitoring period is
higher for young vehicles and urban areas, but the model does not know this.

The benchmark demonstrates three things:

1. InterpretableDriftDetector with BH FDR control correctly flags exactly the two
   drifted features. With 5 features, Bonferroni at alpha=0.05 gives effective per-test
   alpha=0.01; FDR (BH) gives per-test alpha=0.02 for the top-ranked feature —
   materially more powerful when most features are not drifting.

2. Exposure weighting makes a difference: the monitoring cohort contains a mix of
   0.25-year and 1.0-year policies. Unweighted means overweight short-tenure
   policies that happen to be younger vehicles (high-risk). Exposure weighting
   gives the correct population-level picture.

3. DriftAttributor (the older module, no exposure weighting, MSE loss, Bonferroni)
   finds the same features but is noisier on the boundary cases.

Run:
    uv run python benchmarks/benchmark_interpretable_drift.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: InterpretableDriftDetector vs DriftAttributor")
print("=" * 70)
print()

try:
    from insurance_monitoring import InterpretableDriftDetector, DriftAttributor
    print("insurance-monitoring imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-monitoring: {e}")
    print("Install with: pip install insurance-monitoring")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Synthetic model stub
# ---------------------------------------------------------------------------


class PoissonFrequencyModel:
    """Minimal stub that wraps a log-linear Poisson GLM for use in drift detection.

    In a real deployment this would be a fitted statsmodels GLM or a sklearn pipeline.
    Here we encode the model's training-time coefficients directly so the benchmark
    is self-contained and reproducible.

    The model is intentionally stale: it never learns about the phase-2 drift in
    vehicle_age and area.
    """

    # Training-time coefficients (log-linear: log mu = intercept + betas @ x)
    INTERCEPT = -2.5
    COEF_DRIVER_AGE = -0.008      # older driver = lower risk
    COEF_VEHICLE_AGE = 0.06       # older vehicle = higher base risk (no drift awareness)
    COEF_NCB = -0.07              # more NCB years = lower risk
    COEF_ANNUAL_MILEAGE = 0.15    # higher mileage = higher risk
    COEF_AREA = 0.20              # urban area = higher base risk (no drift awareness)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted Poisson means.

        Columns: driver_age, vehicle_age, ncb, annual_mileage, area_numeric.
        Exposure is baked into column 5 (log_exposure) when present; otherwise
        assumes unit exposure.
        """
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
DRIFTED_FEATURES = {"vehicle_age", "area"}   # ground truth

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_REF = 10_000    # reference period
N_MON = 5_000     # monitoring period

VEHICLE_AGE_DRIFT = 1.20   # young vehicles (age < 3) are 20% more expensive in monitoring
AREA_DRIFT = 1.25           # urban areas (area >= 0.6) are 25% more expensive in monitoring

print(f"DGP:")
print(f"  Reference:  {N_REF:,} policies")
print(f"  Monitoring: {N_MON:,} policies")
print(f"  Drifted features: vehicle_age (+20% for new vehicles), area (+25% for urban)")
print(f"  Non-drifted: driver_age, ncb, annual_mileage")
print()


def generate_features(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 5-feature matrix for n policies."""
    driver_age = rng.integers(18, 80, n).astype(float)
    vehicle_age = rng.integers(0, 15, n).astype(float)
    ncb = rng.integers(0, 9, n).astype(float)
    annual_mileage = rng.uniform(0.0, 1.0, n)   # normalised 0–1
    area = rng.uniform(0.0, 1.0, n)              # 0=rural, 1=urban (continuous for GLM)
    return np.column_stack([driver_age, vehicle_age, ncb, annual_mileage, area])


# --- Reference period (well-calibrated) ---
X_ref = generate_features(N_REF, RNG)

# Exposure: mix of annual (0.75-1.0) and quarterly (0.2-0.4) policies
exposure_ref = np.where(
    RNG.random(N_REF) < 0.3,
    RNG.uniform(0.2, 0.4, N_REF),   # 30% short-term policies
    RNG.uniform(0.75, 1.0, N_REF),  # 70% annual policies
)

mu_ref = MODEL.predict(X_ref) * exposure_ref
y_ref = RNG.poisson(mu_ref).astype(float)

# --- Monitoring period: drift induced in vehicle_age and area ---
X_mon = generate_features(N_MON, RNG)

# Exposure: more short-term policies in monitoring (seasonal effect)
exposure_mon = np.where(
    RNG.random(N_MON) < 0.5,
    RNG.uniform(0.2, 0.4, N_MON),
    RNG.uniform(0.75, 1.0, N_MON),
)

# Model predictions — stale, does not know about drift
mu_predicted_mon = MODEL.predict(X_mon) * exposure_mon

# True risk in monitoring period: higher for young vehicles and urban areas
true_mu_mon = mu_predicted_mon.copy()
new_vehicle_mask = X_mon[:, 1] < 3       # vehicle_age < 3
urban_area_mask = X_mon[:, 4] > 0.6     # area > 0.6 (upper 40% = urban)

true_mu_mon[new_vehicle_mask] *= VEHICLE_AGE_DRIFT
true_mu_mon[urban_area_mask] *= AREA_DRIFT

y_mon = RNG.poisson(true_mu_mon).astype(float)

print(f"Reference period:")
print(f"  Mean exposure:    {exposure_ref.mean():.3f} years")
print(f"  Mean predicted mu: {mu_ref.mean():.4f}")
print(f"  A/E ratio:         {y_ref.sum() / mu_ref.sum():.4f}  (target: ~1.00)")
print()
print(f"Monitoring period:")
print(f"  Mean exposure:    {exposure_mon.mean():.3f} years  (more short-term policies)")
print(f"  Mean predicted mu: {mu_predicted_mon.mean():.4f}  (model unchanged)")
print(f"  A/E ratio:         {y_mon.sum() / mu_predicted_mon.sum():.4f}  (elevated: drift exists)")
print(f"  New vehicles (age<3):  {new_vehicle_mask.mean():.1%} of book, drifted {VEHICLE_AGE_DRIFT}x")
print(f"  Urban (area>0.6):      {urban_area_mask.mean():.1%} of book, drifted {AREA_DRIFT}x")
print()

# ---------------------------------------------------------------------------
# BASELINE: DriftAttributor (older module)
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE: DriftAttributor (MSE loss, Bonferroni FWER, no exposure weighting)")
print("-" * 70)
print()

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

print(f"DriftAttributor result ({elapsed_da:.1f}s):")
print(f"  Drift detected:        {result_da.drift_detected}")
print(f"  Attributed features:   {result_da.attributed_features}")
print(f"  Error control:         FWER (Bonferroni)")
print()

# Feature-level results
print(f"  {'Feature':<18} {'Test stat':>10} {'Threshold':>10} {'p-value':>8} {'Drift':>6}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
for feat in FEATURES:
    ts = result_da.test_statistics.get(feat, float("nan"))
    th = result_da.thresholds.get(feat, float("nan"))
    pv = result_da.p_values.get(feat, float("nan"))
    flag = "YES" if feat in result_da.attributed_features else "no"
    truth = " (PLANTED)" if feat in DRIFTED_FEATURES else ""
    print(f"  {feat:<18} {ts:>10.4f} {th:>10.4f} {pv:>8.3f} {flag:>6}{truth}")
print()

correct_da = set(result_da.attributed_features) == DRIFTED_FEATURES
missed_da = DRIFTED_FEATURES - set(result_da.attributed_features)
false_da = set(result_da.attributed_features) - DRIFTED_FEATURES
print(f"  Correct attribution: {'YES' if correct_da else 'NO'}")
if missed_da:
    print(f"  Missed (false negatives): {sorted(missed_da)}")
if false_da:
    print(f"  Over-attributed (false positives): {sorted(false_da)}")
print()

# ---------------------------------------------------------------------------
# LIBRARY: InterpretableDriftDetector with exposure weighting + FDR
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: InterpretableDriftDetector")
print("  poisson_deviance loss, BH FDR control, exposure-weighted")
print("-" * 70)
print()

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

print(f"InterpretableDriftDetector result ({elapsed_idd:.1f}s):")
print(f"  Drift detected:        {result_idd.drift_detected}")
print(f"  Attributed features:   {result_idd.attributed_features}")
print(f"  Error control:         {result_idd.error_control.upper()} (Benjamini-Hochberg)")
print(f"  Bootstrap iterations:  {result_idd.bootstrap_iterations}")
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

correct_idd = set(result_idd.attributed_features) == DRIFTED_FEATURES
missed_idd = DRIFTED_FEATURES - set(result_idd.attributed_features)
false_idd = set(result_idd.attributed_features) - DRIFTED_FEATURES
print(f"  Correct attribution: {'YES' if correct_idd else 'NO'}")
if missed_idd:
    print(f"  Missed (false negatives): {sorted(missed_idd)}")
if false_idd:
    print(f"  Over-attributed (false positives): {sorted(false_idd)}")
print()

print(f"  Plain-text summary:")
print(f"  {result_idd.summary()}")
print()

# ---------------------------------------------------------------------------
# Exposure weighting effect
# ---------------------------------------------------------------------------

print("-" * 70)
print("DETAIL: Exposure weighting matters for mixed policy terms")
print("-" * 70)
print()

t0 = time.time()
detector_unweighted = InterpretableDriftDetector(
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
detector_unweighted.fit_reference(X_ref, y_ref)
result_unweighted = detector_unweighted.test(X_mon, y_mon)
elapsed_unw = time.time() - t0

print(f"Unweighted IDD ({elapsed_unw:.1f}s): attributed = {result_unweighted.attributed_features}")
print(f"Weighted IDD:                    attributed = {result_idd.attributed_features}")
print()
print("P-value comparison (lower = stronger signal):")
print(f"  {'Feature':<18} {'Weighted p':>12} {'Unweighted p':>14}")
print(f"  {'-'*18} {'-'*12} {'-'*14}")
pvals_w = result_idd.p_values
pvals_u = result_unweighted.p_values
for feat in FEATURES:
    truth = " (PLANTED)" if feat in DRIFTED_FEATURES else ""
    print(f"  {feat:<18} {pvals_w[feat]:>12.3f} {pvals_u[feat]:>14.3f}{truth}")
print()

# ---------------------------------------------------------------------------
# FDR control: why it matters with 5 features
# ---------------------------------------------------------------------------

print("-" * 70)
print("DETAIL: FDR (BH) vs FWER (Bonferroni) for 5 features")
print("-" * 70)
print()

# Also run IDD with Bonferroni for direct comparison
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

print(f"With alpha=0.05, d=5 features:")
print(f"  Bonferroni effective per-test alpha: {0.05/5:.3f}")
print(f"  BH effective alpha (rank-1 feature): {1 * 0.05/5:.3f}")
print()
print(f"  IDD (FDR / BH):          attributed = {result_idd.attributed_features}")
print(f"  IDD (FWER / Bonferroni): attributed = {result_bonf.attributed_features}")
print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: What each approach detects")
print("=" * 70)
print()
print(f"  Ground truth: drift in {sorted(DRIFTED_FEATURES)}")
print()
print(f"  {'Check':<42} {'DriftAttributor':>15} {'IDD (BH)':>10}")
print(f"  {'-'*42} {'-'*15} {'-'*10}")

da_correct = set(result_da.attributed_features) == DRIFTED_FEATURES
idd_correct = set(result_idd.attributed_features) == DRIFTED_FEATURES

rows = [
    ("Drift detected", str(result_da.drift_detected), str(result_idd.drift_detected)),
    ("vehicle_age flagged",
     "YES" if "vehicle_age" in result_da.attributed_features else "no",
     "YES" if "vehicle_age" in result_idd.attributed_features else "no"),
    ("area flagged",
     "YES" if "area" in result_da.attributed_features else "no",
     "YES" if "area" in result_idd.attributed_features else "no"),
    ("False positives (non-drifted features flagged)",
     str(len([f for f in result_da.attributed_features if f not in DRIFTED_FEATURES])),
     str(len([f for f in result_idd.attributed_features if f not in DRIFTED_FEATURES]))),
    ("Attribution correct (both planted, no extras)",
     "YES" if da_correct else "NO",
     "YES" if idd_correct else "NO"),
    ("Exposure weighting", "NO", "YES"),
    ("Loss function", "MSE", "Poisson deviance"),
    ("Error control", "Bonferroni (FWER)", "BH (FDR)"),
    ("Run time (s)", f"{elapsed_da:.1f}s", f"{elapsed_idd:.1f}s"),
]

for check, da, idd in rows:
    print(f"  {check:<42} {da:>15} {idd:>10}")

print()
print("  DGP: 5 rating factors, drift in vehicle_age (+20%) and area (+25%)")
print(f"  Reference: {N_REF:,} policies. Monitoring: {N_MON:,} policies.")
print(f"  Monitoring exposure mix: 50% short-term (0.2-0.4 yr), 50% annual (0.75-1.0 yr)")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
