"""
Champion/challenger A/B test — sequential (anytime-valid) design.

Scenario:
    A UK motor pricing team has developed a new telematics-informed frequency
    model (the challenger). The current GBM (the champion) has been in production
    for 18 months. New business is randomly split 70/30 champion/challenger.

    The classical design says: choose a sample size, wait, test once. In
    practice, pricing teams check interim results — which inflates the
    false-positive rate. By month 6 of a 12-month trial, it is tempting
    to conclude the challenger is worse even when it is not.

    SequentialTest uses mixture SPRT (Johari et al. 2022). You can check
    at every monthly update. The type I error guarantee holds regardless
    of when you look or stop. If the challenger is clearly worse, you can
    stop early and redeploy the champion — without invalidating the test.

Commercial interpretation:
    - The test statistic is an e-value (Lambda_n). Reject H0 when Lambda_n >= 1/alpha.
    - Stopping early to remove a bad challenger is valid. Stopping early
      because you like the interim numbers (challenger looks better) is
      also valid.
    - The confidence sequence on the rate ratio is anytime-valid — it is
      a valid interval at every look, not just at the planned end date.
"""

import numpy as np
import polars as pl
from insurance_monitoring import SequentialTest

rng = np.random.default_rng(9999)

# ── Experimental setup ────────────────────────────────────────────────────────
CHAMPION_RATE = 0.080    # claims per car-year
CHALLENGER_RATE = 0.095  # challenger is worse by ~19% — meaningful deterioration
TRIAL_MONTHS = 18        # maximum trial duration
SPLIT = 0.70             # 70% champion, 30% challenger

new_business_per_month = 2_000   # new policies written per month

# ── Run the sequential test ───────────────────────────────────────────────────

test = SequentialTest(
    metric="frequency",
    alpha=0.05,
    tau=0.3,         # prior on log-rate-ratio — moderate effect expected
    alternative="two_sided",
)

print("Monthly champion/challenger monitoring")
print("=" * 72)
print(f"{'Month':<8} {'C claims':>10} {'C exp':>10} {'X claims':>10} "
      f"{'X exp':>10} {'e-value':>10} {'Decision':<14}")
print("-" * 72)

stopped_at = None

for month in range(1, TRIAL_MONTHS + 1):
    n = new_business_per_month
    n_champion = int(n * SPLIT)
    n_challenger = n - n_champion

    # Simulate one month's claims experience
    exp_champ = rng.uniform(0.5, 1.0, n_champion)
    exp_chall = rng.uniform(0.5, 1.0, n_challenger)

    claims_champ = rng.poisson(exp_champ * CHAMPION_RATE).astype(float)
    claims_chall = rng.poisson(exp_chall * CHALLENGER_RATE).astype(float)

    result = test.update(
        champion_claims=claims_champ,
        champion_exposure=exp_champ,
        challenger_claims=claims_chall,
        challenger_exposure=exp_chall,
    )

    decision = result.decision if result.should_stop else "—"
    print(f"{month:<8} {claims_champ.sum():>10.0f} {exp_champ.sum():>10.1f} "
          f"{claims_chall.sum():>10.0f} {exp_chall.sum():>10.1f} "
          f"{result.lambda_value:>10.2f} {decision:<14}")

    if result.should_stop:
        stopped_at = month
        print()
        print(f"Test stopped at month {month}: {result.decision}")
        break

print()
print("── Final result ────────────────────────────────────────────────────────")
print(f"  Champion rate:    {result.champion_rate:.4f} claims/car-year")
print(f"  Challenger rate:  {result.challenger_rate:.4f} claims/car-year")
print(f"  Rate ratio:       {result.rate_ratio:.3f}  "
      f"CI [{result.rate_ratio_ci_lower:.3f}, {result.rate_ratio_ci_upper:.3f}]")
print(f"  e-value:          {result.lambda_value:.2f}  (threshold: {result.threshold:.1f})")
print(f"  P(challenger worse): {1 - result.prob_challenger_better:.2%}")
print()
print(result.summary)

# ── Power comparison: sequential vs fixed-n design ────────────────────────────
print()
print("── Power comparison (100 simulations) ─────────────────────────────────")
print("  True effect: challenger 19% worse. How often does each design catch it?")
print()

n_sims = 100
seq_detections = 0
fixed_detections = 0

# Fixed design: classical test at month 12 only
from scipy.stats import poisson as poisson_dist

for _ in range(n_sims):
    seq_test = SequentialTest(metric="frequency", alpha=0.05, tau=0.3)
    detected_sequential = False
    total_champ_claims = 0.0
    total_champ_exp = 0.0
    total_chall_claims = 0.0
    total_chall_exp = 0.0

    for month in range(1, 13):  # 12-month common horizon
        n = new_business_per_month
        n_c = int(n * SPLIT)
        n_x = n - n_c
        ec = rng.uniform(0.5, 1.0, n_c)
        ex = rng.uniform(0.5, 1.0, n_x)
        cc = rng.poisson(ec * CHAMPION_RATE).astype(float)
        cx = rng.poisson(ex * CHALLENGER_RATE).astype(float)

        r = seq_test.update(champion_claims=cc, champion_exposure=ec,
                            challenger_claims=cx, challenger_exposure=ex)

        total_champ_claims += cc.sum()
        total_champ_exp += ec.sum()
        total_chall_claims += cx.sum()
        total_chall_exp += ex.sum()

        if r.should_stop and not detected_sequential:
            detected_sequential = True

    if detected_sequential:
        seq_detections += 1

    # Fixed design: z-test for rate difference at month 12 only
    champ_rate = total_champ_claims / total_champ_exp
    chall_rate = total_chall_claims / total_chall_exp
    # Pooled rate under H0
    pooled = (total_champ_claims + total_chall_claims) / (total_champ_exp + total_chall_exp)
    se = np.sqrt(pooled / total_champ_exp + pooled / total_chall_exp)
    z = (chall_rate - champ_rate) / se if se > 0 else 0.0
    if abs(z) > 1.96:
        fixed_detections += 1

print(f"  Sequential (check monthly, stop early):  {seq_detections}/{n_sims} detected")
print(f"  Fixed design (check once at month 12):   {fixed_detections}/{n_sims} detected")
print()
print("  Sequential power is higher because it can accumulate evidence across")
print("  months and stop as soon as the signal is clear, whereas the fixed")
print("  design commits the full sample to a single end-point test.")
