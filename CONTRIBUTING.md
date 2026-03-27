# Contributing

Contributions are welcome. Here is how the project is set up and what we expect.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-monitoring
cd insurance-monitoring
uv sync --dev
```

Tests run on Databricks (the library is designed for that environment). To run locally for fast iteration:

```bash
uv run pytest tests/ -x -q
```

## What we are looking for

- **New monitoring methods** backed by a published reference. The library follows arXiv 2510.04556 as its methodological backbone. New additions should cite their source.
- **Bug fixes** with a regression test that reproduces the failure.
- **Improved docstrings or examples** — especially UK-specific insurance scenarios.

We are not looking for:

- Pandas support (Polars-native is a design decision, not an oversight)
- Generic ML monitoring utilities that do not account for insurance-specific structure (exposure weighting, Poisson/Gamma distributions, regulatory thresholds)

## Workflow

1. Open an issue to discuss your idea before writing code.
2. Fork the repo and create a branch (`git checkout -b your-feature`).
3. Write tests alongside your code. The test suite lives in `tests/`.
4. Submit a pull request. The PR description should explain *why* the change is needed, not just what it does.

## Code standards

- Type annotations on all public functions and classes.
- Docstrings on all public symbols following the NumPy format used throughout the codebase.
- No new hard dependencies without discussion. The current stack (numpy, scipy, polars, matplotlib) is intentionally minimal.
- Polars-native: inputs accept `pl.Series` or `np.ndarray`; outputs return `pl.DataFrame` where tabular.

## Questions

Start a [Discussion](https://github.com/burning-cost/insurance-monitoring/discussions) rather than opening an issue for general questions.
