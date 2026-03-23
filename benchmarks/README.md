# insurance-monitoring: Benchmark

## Headline result

**Aggregate A/E never breaches its 5% alert threshold across 15,000 monitoring-period policies with three simultaneous model failures. PSI fires on driver age drift at around policy 1,500 — roughly 8–10 times earlier.**

In the 50,000-policy / 15,000-monitoring-period scenario, the aggregate A/E stayed within the 0.95–1.05 green band for the entire monitoring period. PSI on driver_age hit the RED threshold (PSI > 0.25) after approximately 1,000–1,500 policies — roughly one month into the monitoring period on a 1,250-policy/month book. The aggregate A/E would never have triggered an investigation.

Run the benchmarks yourself:

```bash
uv run python benchmarks/benchmark.py
uv run python benchmarks/benchmark_pit.py          # PIT-based calibration tests
uv run python benchmarks/benchmark_sequential.py   # Sequential monitoring (alarm timing)
uv run python benchmarks/benchmark_interpretable_drift.py  # Feature-level drift attribution
```

---

## What the benchmarks measure

### Benchmark 1: MonitoringReport vs aggregate A/E check

Three deliberate failure modes are planted in the monitoring period:

1. **Covariate shift** — young drivers (18–30) oversampled 2× in the monitoring period
2. **Calibration drift** — new vehicles (age < 3) have claims inflated 25%
3. **Discrimination decay** — 30% of monitoring-period predictions randomised

**Data:** 50,000 reference policies, 15,000 monitoring-period policies. Full script: `benchmarks/benchmark.py`.

---

## Method vs failure mode detection matrix

Which monitoring method detects which failure mode?

| Failure mode | Aggregate A/E | PSI / CSI | Segment A/E | Gini drift test | Murphy decomp | PITMonitor | Sequential test |
|---|---|---|---|---|---|---|---|
| Covariate shift (feature dist change) | No | **Yes** | Partial | No | No | No | No |
| Concept drift (relationship change) | Partial | No | Partial | **Yes** | **Yes** | Partial | No |
| Calibration degradation (global scale) | **Yes** | No | **Yes** | No | **Yes** | **Yes** | **Yes** |
| Discrimination loss (ranking decay) | No | No | No | **Yes** | **Yes** | Partial | No |
| Sudden shift (changepoint) | Delayed | Delayed | Delayed | Delayed | No | **Yes** | **Yes** |
| Gradual drift (slow accumulation) | Late | **Yes** | Late | Late | Partial | **Yes** | **Yes** |

Notes:
- "No" = the method does not detect this failure mode by design.
- "Partial" = the method may detect depending on magnitude and pattern of the drift.
- "Delayed" / "Late" = the method detects but only after significant exposure has accumulated.
- "Yes" = the method is purpose-designed to detect this failure mode.

No single check is sufficient. The aggregate A/E is blind to covariate shift and discrimination loss — two of the three failure modes in the benchmark.

---

## Results at fixed snapshot (50k/15k, three failure modes)

| Check | Manual aggregate A/E | MonitoringReport |
|---|---|---|
| Aggregate A/E ratio | Yes | Yes |
| Statistical CI on A/E | No | Yes |
| Segment-level A/E drift | No | Yes |
| Driver age covariate shift (PSI) | No | Yes — RED flag |
| Vehicle age covariate shift (PSI) | No | Yes |
| Gini discrimination decay | No | Yes |
| Murphy decomposition (REFIT vs RECALIBRATE) | No | Yes |
| Structured recommendation | No | Yes (REFIT or RECALIBRATE) |
| PRA SS3/17 model risk audit trail | No | Yes |
| Policies to first alarm | Never (no breach) | ~1,500 (PSI RED) |

The aggregate A/E in the monitoring period sits close to 1.0 because overpriced older vehicles cancel underprice on young drivers. A manual check says "no action needed". The library sees through the cancellation effect.

Discrimination decay — 30% of predictions randomised — is invisible to A/E monitoring entirely. A model losing its ranking power needs a refit, not a recalibration; the Murphy decomposition tells you which.

---

## Benchmark 2: mSPRT vs fixed-horizon t-test

**Scenario:** UK motor champion/challenger experiment. Analyst checks monthly for up to 24 months. 10,000 Monte Carlo simulations under H₀ (no true effect).

| Method | Nominal FPR | Actual FPR (monthly peeking) | Notes |
|--------|-------------|------------------------------|-------|
| Fixed-horizon t-test | 5% | ~25% | 5× inflation from repeated peeking |
| mSPRT (SequentialTest) | 5% | ~1% | Valid at all stopping times |

Under H₁ (challenger 10% cheaper on frequency), mSPRT detects the effect in a median of 8 months on a 500-policy-per-arm-per-month book. A pre-registered t-test at 24 months would reach the same conclusion but forces the team to wait 16 months longer.

Full script: `benchmarks/benchmark_sequential.py`.

---

## Benchmark 3: PITMonitor vs repeated Hosmer-Lemeshow

**Scenario:** 500 well-calibrated Poisson observations followed by 500 with 15% rate inflation.

| Method | Nominal FPR | Empirical FPR (stable phase) | FPR inflation | Detects phase-2 drift |
|--------|-------------|------------------------------|--------------|----------------------|
| H-L repeated (every 50 obs) | 5% | ~46% (10 looks) | 9× | Yes, after prior false alarms |
| PITMonitor | 5% | ~3% (300 simulations) | 0.6× | Yes, no false alarms |

PITMonitor also estimates the changepoint: when it fires, the Bayes factor scan recovers the true drift onset (t~500) within ±30 steps on typical runs.

Full script: `benchmarks/benchmark_pit.py`.

---

## Benchmark 4: InterpretableDriftDetector vs DriftAttributor

**Scenario:** 5-feature Poisson pricing model with drift planted in 2 features. Reference: 10,000 policies; monitoring: 5,000 with mixed policy terms (50% short-term).

| Check | DriftAttributor | InterpretableDriftDetector (BH) |
|-------|----------------|--------------------------------|
| vehicle_age flagged | Yes | Yes |
| area flagged | Yes | Yes |
| False positives | 0 | 0 |
| Exposure weighting | No | Yes |
| Loss function | MSE | Poisson deviance |
| Error control | Bonferroni | Benjamini-Hochberg |

With d=10 rating factors, BH gives roughly 2× the per-test power of Bonferroni on the rank-1 feature.

Full script: `benchmarks/benchmark_interpretable_drift.py`.

---

## Why aggregate A/E is insufficient

The benchmark makes this concrete, but the argument is simple. Aggregate A/E is an exposure-weighted mean of per-policy errors. If the model is 15% cheap on young urban drivers (say, 20% of the book) and 3.75% expensive on everyone else, the aggregate reads 1.00. No alarm is raised. Both errors worsen as the portfolio composition continues to shift.

PSI per feature detects the compositional shift before any claims are observed. A/E with Poisson confidence intervals detects calibration drift at the segment level. The Gini drift test detects ranking degradation. You need all three because they catch different things.
