"""
Benchmark: mSPRT vs fixed-horizon t-test on simulated insurance A/B data.

Demonstrates:
1. Fixed-horizon t-test with monthly peeking inflates FPR to ~15-20%
2. mSPRT maintains FPR at ~5% with the same monthly checking
3. mSPRT stops earlier when a real effect exists

Run on Databricks (not locally — computation is non-trivial).

Usage:
    # On Databricks cluster:
    %pip install insurance-monitoring
    # then run this script as a notebook cell or %run
"""
import math
import numpy as np
from scipy import stats

try:
    from insurance_monitoring.sequential import SequentialTest
except ImportError:
    raise ImportError("Run: pip install insurance-monitoring>=0.5.0")


def run_fixed_horizon_ttest(rng, rate_a, rate_b, exposure_per_arm_per_month, n_months, alpha=0.05):
    """Simulate a fixed-horizon t-test analyst who peeks every month.

    The analyst computes a two-sample rate comparison using a normal approximation
    each month. This is the invalid practice the mSPRT replaces.
    Returns (rejected: bool, stop_month: int | None).
    """
    cum_ca, cum_cb = 0, 0
    cum_ea = cum_eb = 0.0

    for month in range(n_months):
        ca = int(rng.poisson(rate_a * exposure_per_arm_per_month))
        cb = int(rng.poisson(rate_b * exposure_per_arm_per_month))
        cum_ca += ca
        cum_cb += cb
        cum_ea += exposure_per_arm_per_month
        cum_eb += exposure_per_arm_per_month

        if cum_ca < 5 or cum_cb < 5:
            continue

        # Normal approximation for rate difference
        lambda_a_hat = cum_ca / cum_ea
        lambda_b_hat = cum_cb / cum_eb
        se_a = math.sqrt(lambda_a_hat / cum_ea)
        se_b = math.sqrt(lambda_b_hat / cum_eb)
        se_diff = math.sqrt(se_a ** 2 + se_b ** 2)

        if se_diff == 0:
            continue

        z = (lambda_b_hat - lambda_a_hat) / se_diff
        p_value = 2 * stats.norm.sf(abs(z))

        if p_value < alpha:
            return True, month + 1

    return False, None


def run_msprt(rng, rate_a, rate_b, exposure_per_arm_per_month, n_months, alpha=0.05, tau=0.03):
    """Simulate the mSPRT with monthly checks.

    Returns (rejected: bool, stop_month: int | None).
    """
    test = SequentialTest(
        metric="frequency",
        alpha=alpha,
        tau=tau,
        max_duration_years=n_months / 12,
        min_exposure_per_arm=0.0,
    )

    for month in range(n_months):
        ca = int(rng.poisson(rate_a * exposure_per_arm_per_month))
        cb = int(rng.poisson(rate_b * exposure_per_arm_per_month))
        result = test.update(ca, exposure_per_arm_per_month, cb, exposure_per_arm_per_month)

        if result.should_stop and result.decision == "reject_H0":
            return True, month + 1

    return False, None


def benchmark_h0(n_sims=500, rate=0.10, exposure_per_arm_per_month=50.0, n_months=36, alpha=0.05):
    """Simulate n_sims experiments under H0 (same rate, no effect).

    Measures false positive rate (FPR) for both methods at each monthly peek.
    """
    print(f"\n{'='*60}")
    print(f"H0 SIMULATION: {n_sims} experiments, rate_A = rate_B = {rate}")
    print(f"Monthly exposure: {exposure_per_arm_per_month} car-years/arm, {n_months} months")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    ttest_rejects = 0
    msprt_rejects = 0
    ttest_stop_months = []
    msprt_stop_months = []

    for i in range(n_sims):
        # Reset RNG state each experiment (deterministic but independent)
        exp_rng = np.random.default_rng(42 + i)

        rejected_t, stop_t = run_fixed_horizon_ttest(
            exp_rng, rate, rate, exposure_per_arm_per_month, n_months, alpha
        )
        exp_rng2 = np.random.default_rng(42 + i)
        rejected_m, stop_m = run_msprt(
            exp_rng2, rate, rate, exposure_per_arm_per_month, n_months, alpha
        )

        if rejected_t:
            ttest_rejects += 1
            ttest_stop_months.append(stop_t)
        if rejected_m:
            msprt_rejects += 1
            msprt_stop_months.append(stop_m)

    ttest_fpr = ttest_rejects / n_sims
    msprt_fpr = msprt_rejects / n_sims

    print(f"\nFixed-horizon t-test (monthly peeking):")
    print(f"  False positives: {ttest_rejects}/{n_sims} = {ttest_fpr:.1%}")
    print(f"  Nominal alpha: {alpha:.0%}")
    print(f"  FPR inflation: {ttest_fpr / alpha:.1f}x")

    print(f"\nmSPRT (monthly, tau=0.03):")
    print(f"  False positives: {msprt_rejects}/{n_sims} = {msprt_fpr:.1%}")
    print(f"  FPR vs nominal: {msprt_fpr / alpha:.1f}x")

    return {
        "ttest_fpr": ttest_fpr,
        "msprt_fpr": msprt_fpr,
        "ttest_rejects": ttest_rejects,
        "msprt_rejects": msprt_rejects,
    }


def benchmark_h1(n_sims=500, rate_a=0.10, rate_b=0.095, exposure_per_arm_per_month=50.0,
                 n_months=36, alpha=0.05):
    """Simulate n_sims experiments under H1 (challenger 5% better).

    Measures power and average stopping month for both methods.
    """
    true_improvement = (1 - rate_b / rate_a) * 100
    print(f"\n{'='*60}")
    print(f"H1 SIMULATION: {n_sims} experiments, challenger {true_improvement:.0f}% better")
    print(f"  rate_A={rate_a}, rate_B={rate_b}")
    print(f"Monthly exposure: {exposure_per_arm_per_month} car-years/arm, {n_months} months")
    print(f"{'='*60}")

    ttest_rejects = 0
    msprt_rejects = 0
    ttest_stop_months_list = []
    msprt_stop_months_list = []

    for i in range(n_sims):
        exp_rng = np.random.default_rng(1000 + i)
        rejected_t, stop_t = run_fixed_horizon_ttest(
            exp_rng, rate_a, rate_b, exposure_per_arm_per_month, n_months, alpha
        )
        exp_rng2 = np.random.default_rng(1000 + i)
        rejected_m, stop_m = run_msprt(
            exp_rng2, rate_a, rate_b, exposure_per_arm_per_month, n_months, alpha
        )

        if rejected_t:
            ttest_rejects += 1
            ttest_stop_months_list.append(stop_t)
        if rejected_m:
            msprt_rejects += 1
            msprt_stop_months_list.append(stop_m)

    ttest_power = ttest_rejects / n_sims
    msprt_power = msprt_rejects / n_sims

    ttest_avg_stop = np.mean(ttest_stop_months_list) if ttest_stop_months_list else float("nan")
    msprt_avg_stop = np.mean(msprt_stop_months_list) if msprt_stop_months_list else float("nan")

    print(f"\nFixed-horizon t-test (monthly peeking):")
    print(f"  Power: {ttest_rejects}/{n_sims} = {ttest_power:.1%}")
    print(f"  Average stop month (when stopped): {ttest_avg_stop:.1f}")
    print(f"  Note: peeking inflates power too — FPR and power both wrong")

    print(f"\nmSPRT (monthly, tau=0.03):")
    print(f"  Power: {msprt_rejects}/{n_sims} = {msprt_power:.1%}")
    print(f"  Average stop month (when stopped): {msprt_avg_stop:.1f}")
    print(f"  Earlier stopping: {(ttest_avg_stop - msprt_avg_stop):.1f} months sooner")

    return {
        "ttest_power": ttest_power,
        "msprt_power": msprt_power,
        "ttest_avg_stop": ttest_avg_stop,
        "msprt_avg_stop": msprt_avg_stop,
    }


def main():
    print("insurance-monitoring v0.5.0 — Sequential Test Benchmark")
    print("Comparing mSPRT vs fixed-horizon t-test with monthly peeking\n")

    h0_results = benchmark_h0(n_sims=500)
    h1_results = benchmark_h1(n_sims=500)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nUnder H0 (null true):")
    print(f"  t-test FPR with peeking: {h0_results['ttest_fpr']:.1%} "
          f"({h0_results['ttest_fpr'] / 0.05:.1f}x nominal)")
    print(f"  mSPRT FPR:               {h0_results['msprt_fpr']:.1%} "
          f"({h0_results['msprt_fpr'] / 0.05:.1f}x nominal)")

    print(f"\nUnder H1 (5% challenger improvement):")
    print(f"  t-test power:  {h1_results['ttest_power']:.1%}, avg stop month: {h1_results['ttest_avg_stop']:.1f}")
    print(f"  mSPRT power:   {h1_results['msprt_power']:.1%}, avg stop month: {h1_results['msprt_avg_stop']:.1f}")

    print(f"\nConclusion: mSPRT maintains valid type I error control while the")
    print(f"t-test with monthly peeking inflates FPR to "
          f"{h0_results['ttest_fpr']:.0%}. When effects are real, mSPRT")
    print(f"stops approximately {h1_results['ttest_avg_stop'] - h1_results['msprt_avg_stop']:.0f} "
          f"months earlier on average.")


if __name__ == "__main__":
    main()
