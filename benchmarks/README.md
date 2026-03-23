# Benchmarks — insurance-monitoring

**Headline:** Aggregate A/E ratio never breaches its 5% alert threshold across 15,000 monitoring-period policies with three simultaneous model failures; PSI fires on driver age drift at around policy 1,500 — roughly 8–10 times earlier.

---

## Comparison table

50,000 reference policies, 15,000 monitoring-period policies. Three deliberately induced failure modes: 2× young driver oversampling, 25% calibration inflation on new vehicles, 30% prediction randomisation (discrimination decay).

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

## How to run

```bash
uv run python benchmarks/benchmark.py
```

Additional benchmarks:

```bash
uv run python benchmarks/benchmark_pit.py          # PIT-based calibration tests
uv run python benchmarks/benchmark_sequential.py   # Sequential monitoring (alarm timing)
uv run python benchmarks/benchmark_interpretable_drift.py  # Feature-level drift attribution
```

### Databricks

```bash
databricks workspace import-dir benchmarks /Workspace/insurance-monitoring/benchmarks
```

Dependencies: `insurance-monitoring`, `numpy`, `polars`.

The main benchmark runs in under 60 seconds. The sequential benchmark walks through the monitoring period in 500-policy batches to show alarm timing.
