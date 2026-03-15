"""
Feature drift detection for insurance pricing models.

Three complementary tools:

- PSI (Population Stability Index): the industry standard for operational
  dashboards. Heuristic, no p-value, but consistent across portfolio sizes.
  Use the exposure-weighted variant for insurance — unweighted PSI treats a
  policy with 0.1 car-years the same as one with 1.0 car-years.

- KS test: non-parametric hypothesis test. Best for formal quarterly testing
  where you want a p-value. Can be over-sensitive at large n (>500k policies).

- Wasserstein distance: reports drift in the original feature units ('average
  driver age shifted by 1.4 years'). More interpretable than PSI for
  communicating to underwriters and heads of pricing.

References
----------
- arXiv 2510.04556 (monitoring framework)
- Evidently AI blog: comparison of drift detection methods (2024)
- PSI formula from FICO/banking practice, standard since the 1990s
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats


ArrayLike = Union[np.ndarray, pl.Series]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert Polars Series or numpy array to a 1-D float64 numpy array."""
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_numpy_optional(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return _to_numpy(x)


def psi(
    reference: ArrayLike,
    current: ArrayLike,
    n_bins: int = 10,
    exposure_weights: Optional[ArrayLike] = None,
    reference_exposure: Optional[ArrayLike] = None,
) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI measures how much a distribution has shifted between a reference
    period (typically the model training window) and a current monitoring
    period. Higher values indicate greater drift.

    Formula::

        PSI = sum_i [ (A_i - E_i) * ln(A_i / E_i) ]

    where E_i is the proportion in bin i for the reference distribution and
    A_i is the proportion in bin i for the current distribution.

    Bins are determined by equal-frequency quantiles of the reference
    distribution (not equal-width). This preserves sensitivity where the
    reference distribution has mass.

    Parameters
    ----------
    reference:
        Reference period values (training window). Accepts Polars Series or
        numpy array.
    current:
        Current monitoring period values.
    n_bins:
        Number of bins. Defaults to 10, which is standard practice.
    exposure_weights:
        Optional array of exposures (e.g., earned car-years) for the current
        period. When provided, bin proportions are computed as
        sum(exposure in bin) / sum(total exposure) rather than policy count.
        This is the correct approach for insurance pricing monitoring — a bin
        containing 1,000 one-month policies should not be treated identically
        to a bin containing 100 annual policies.
    reference_exposure:
        Optional array of exposures for the reference period. When provided,
        reference bin proportions are also exposure-weighted — the same logic
        applied symmetrically. If omitted, reference proportions are count-based
        (the v0.3 default, maintained for backward compatibility). Providing
        reference_exposure when exposure_weights is also provided gives a fully
        symmetric PSI: both sides weighted by actual exposure.

    Returns
    -------
    float
        PSI value. Interpret as:
        - < 0.10: no significant shift
        - 0.10–0.25: moderate shift, investigate
        - >= 0.25: significant shift, model likely stale

    Examples
    --------
    No drift (same distribution)::

        import numpy as np
        from insurance_monitoring.drift import psi
        rng = np.random.default_rng(0)
        ref = rng.normal(30, 5, 10_000)   # driver ages, training
        cur = rng.normal(30, 5, 5_000)    # driver ages, current
        psi(ref, cur)  # => near 0.0

    Significant shift::

        cur_shift = rng.normal(35, 5, 5_000)  # older drivers
        psi(ref, cur_shift)  # => > 0.25

    Exposure-weighted::

        exposure = rng.uniform(0.1, 1.0, 5_000)
        psi(ref, cur, exposure_weights=exposure)
    """
    ref = _to_numpy(reference)
    cur = _to_numpy(current)
    weights = _to_numpy_optional(exposure_weights)
    ref_weights = _to_numpy_optional(reference_exposure)

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must both be non-empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if weights is not None and len(weights) != len(cur):
        raise ValueError(
            f"exposure_weights length ({len(weights)}) must match current length ({len(cur)})"
        )
    if ref_weights is not None and len(ref_weights) != len(ref):
        raise ValueError(
            f"reference_exposure length ({len(ref_weights)}) must match reference length ({len(ref)})"
        )

    # Bin edges from equal-frequency quantiles of the reference distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    # Deduplicate edges that collapse due to ties (e.g., many identical values)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # Entire reference is a single value — PSI is undefined; return 0
        return 0.0

    # Force boundaries to include all data
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Reference proportions: exposure-weighted if reference_exposure provided, else counts
    if ref_weights is not None:
        ref_props = np.zeros(len(bin_edges) - 1)
        ref_bin_indices = np.digitize(ref, bin_edges[1:-1])
        for idx in range(len(ref_props)):
            mask = ref_bin_indices == idx
            ref_props[idx] = ref_weights[mask].sum()
        total_ref_weight = ref_props.sum()
        if total_ref_weight == 0:
            raise ValueError("Sum of reference_exposure is zero")
        ref_props = ref_props / total_ref_weight
    else:
        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        ref_props = ref_counts / ref_counts.sum()

    # Current proportions: exposure-weighted if weights provided, else counts
    if weights is not None:
        cur_props = np.zeros(len(bin_edges) - 1)
        bin_indices = np.digitize(cur, bin_edges[1:-1])  # which bin each obs falls in
        for idx in range(len(cur_props)):
            mask = bin_indices == idx
            cur_props[idx] = weights[mask].sum()
        total_weight = cur_props.sum()
        if total_weight == 0:
            raise ValueError("Sum of exposure_weights is zero")
        cur_props = cur_props / total_weight
    else:
        cur_counts, _ = np.histogram(cur, bins=bin_edges)
        cur_props = cur_counts / cur_counts.sum()

    # Compute PSI, adding a small epsilon to avoid log(0)
    # epsilon = 0.0001 is the standard fix from credit scoring practice
    eps = 1e-4
    ref_props = np.where(ref_props == 0, eps, ref_props)
    cur_props = np.where(cur_props == 0, eps, cur_props)

    psi_value = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return psi_value


def csi(
    reference_df: Union[pl.DataFrame, "pd.DataFrame"],  # noqa: F821
    current_df: Union[pl.DataFrame, "pd.DataFrame"],  # noqa: F821
    features: list[str],
    n_bins: int = 10,
) -> pl.DataFrame:
    """Compute Characteristic Stability Index for each feature in a dataframe.

    CSI applies PSI independently to each feature column. The result is a
    one-row-per-feature summary with the CSI value and a traffic-light band.

    This is the standard 'CSI heat map' used in monthly monitoring packs:
    identify which rating factors have drifted and by how much.

    Parameters
    ----------
    reference_df:
        DataFrame from the reference (training) period. Accepts Polars or
        pandas DataFrames.
    current_df:
        DataFrame from the current monitoring period.
    features:
        List of column names to monitor. Typically your continuous rating
        factors: driver age, vehicle age, sum insured, etc. For categorical
        features, convert to integer codes first.
    n_bins:
        Passed to :func:`psi` for each feature.

    Returns
    -------
    pl.DataFrame
        One row per feature with columns:
        - ``feature``: feature name
        - ``csi``: CSI value
        - ``band``: 'green' / 'amber' / 'red'

    Examples
    --------
    ::

        import polars as pl
        from insurance_monitoring.drift import csi

        ref = pl.DataFrame({"driver_age": [25, 30, 35, 40, 45, 50]})
        cur = pl.DataFrame({"driver_age": [28, 33, 38, 43, 48, 53]})
        csi(ref, cur, features=["driver_age"])
    """
    # Convert pandas to polars if needed
    if not isinstance(reference_df, pl.DataFrame):
        reference_df = pl.from_pandas(reference_df)
    if not isinstance(current_df, pl.DataFrame):
        current_df = pl.from_pandas(current_df)

    from insurance_monitoring.thresholds import PSIThresholds
    thresholds = PSIThresholds()

    rows = []
    for feature in features:
        if feature not in reference_df.columns:
            raise ValueError(f"Feature '{feature}' not found in reference_df")
        if feature not in current_df.columns:
            raise ValueError(f"Feature '{feature}' not found in current_df")

        ref_col = reference_df[feature]
        cur_col = current_df[feature]

        csi_value = psi(ref_col, cur_col, n_bins=n_bins)
        band = thresholds.classify(csi_value)
        rows.append({"feature": feature, "csi": csi_value, "band": band})

    return pl.DataFrame(rows, schema={"feature": pl.Utf8, "csi": pl.Float64, "band": pl.Utf8})


def ks_test(
    reference: ArrayLike,
    current: ArrayLike,
) -> dict[str, float]:
    """Kolmogorov-Smirnov two-sample test for distribution shift.

    Nonparametric test of whether reference and current come from the same
    distribution. Returns both the KS statistic and p-value.

    Use this for formal hypothesis testing at quarter-end. For large portfolios
    (>500k policies) the test will detect economically trivial shifts; in that
    case prefer PSI for operational dashboards and interpret the KS p-value
    with caution.

    Parameters
    ----------
    reference:
        Reference period values.
    current:
        Current monitoring period values.

    Returns
    -------
    dict with keys:
        - ``statistic``: KS test statistic (max absolute difference in CDFs)
        - ``p_value``: two-sided p-value
        - ``significant``: bool, True if p < 0.05

    Examples
    --------
    ::

        from insurance_monitoring.drift import ks_test
        import numpy as np

        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0.5, 1, 1000)  # shifted mean
        result = ks_test(ref, cur)
        result["significant"]  # => True
    """
    ref = _to_numpy(reference)
    cur = _to_numpy(current)

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must both be non-empty")

    result = stats.ks_2samp(ref, cur)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }


def wasserstein_distance(
    reference: ArrayLike,
    current: ArrayLike,
) -> float:
    """Wasserstein (earth mover's) distance between two distributions.

    Reports the minimum 'work' needed to transform the reference distribution
    into the current one, measured in the same units as the input data.

    This interpretability advantage over PSI is useful when communicating
    drift to non-technical stakeholders: 'average driver age shifted by 1.4
    years' is more actionable than 'PSI = 0.18'.

    Parameters
    ----------
    reference:
        Reference period values.
    current:
        Current monitoring period values.

    Returns
    -------
    float
        Wasserstein-1 distance (units = original feature units).

    Examples
    --------
    ::

        from insurance_monitoring.drift import wasserstein_distance
        import numpy as np

        rng = np.random.default_rng(0)
        driver_ages_train = rng.normal(35, 8, 10_000)
        driver_ages_now = rng.normal(37, 8, 5_000)  # slight ageing of book
        d = wasserstein_distance(driver_ages_train, driver_ages_now)
        # d ≈ 2.0 years  →  'book has aged by ~2 years on average'
    """
    ref = _to_numpy(reference)
    cur = _to_numpy(current)

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must both be non-empty")

    return float(stats.wasserstein_distance(ref, cur))
