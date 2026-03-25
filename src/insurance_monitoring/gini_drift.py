"""
Class-based Gini drift test based on the asymptotic distribution established
in Wüthrich, Merz & Noll (2025), arXiv:2510.04556.

This module provides ``GiniDriftTest``: a two-sample z-test for whether the
Gini coefficient (ranking power) has changed between a reference period and a
monitoring period. Both raw datasets are required — if you only have a stored
training Gini scalar, use ``GiniDriftBootstrapTest`` in
``insurance_monitoring.discrimination`` instead.

Design rationale
----------------
The asymptotic theory (Theorem 1 of arXiv:2510.04556) establishes that::

    sqrt(n) * (G_hat - G) -> N(0, sigma^2)

where sigma^2 is estimated by bootstrap. For two independent samples, the
variance of the difference is additive::

    Var(G_ref - G_mon) = Var(G_ref) + Var(G_mon)

so the z-statistic is::

    z = (G_ref - G_mon) / sqrt(Var(G_ref) + Var(G_mon))

This class wraps that logic with:

- Lazy evaluation: bootstrap runs on first ``.test()`` call, then cached
- Exposure weighting for policies with varying durations
- A ``GiniDriftTestResult`` dataclass with all test diagnostics
- A ``.summary()`` method returning a governance-ready plain-text report
- Input validation with informative error messages

When to use this vs the alternatives
--------------------------------------
- ``GiniDriftTest`` (this class): use when you have raw data for both periods
  and want a class interface. The two-sample design is correct for A/B
  comparisons and period-over-period validation reviews.

- ``gini_drift_test()`` function: functional equivalent, use in pipelines
  where you do not need the class interface.

- ``GiniDriftBootstrapTest``: use for deployed monitoring where you only have
  a stored training Gini scalar — raw reference data not needed.

References
----------
- Wüthrich, Merz & Noll (2025), arXiv:2510.04556, Theorem 1, Algorithm 2
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields as dc_fields
from typing import Optional, Union

import numpy as np
from scipy import stats

from insurance_monitoring.discrimination import (
    _gini_from_arrays,
    _bootstrap_gini_samples,
    _to_numpy,
    _to_numpy_optional,
    ArrayLike,
)


@dataclass
class GiniDriftTestResult:
    """Result of a ``GiniDriftTest`` two-sample asymptotic z-test.

    Attributes
    ----------
    gini_reference : float
        Gini coefficient computed on the reference period data.
    gini_monitor : float
        Gini coefficient computed on the monitoring period data.
    delta : float
        gini_monitor - gini_reference. Negative means ranking power has
        degraded in the monitoring period.
    z_statistic : float
        (gini_monitor - gini_reference) / sqrt(Var_ref + Var_mon).
        Negative z means discrimination has declined.
    p_value : float
        Two-sided p-value under the N(0, 1) asymptotic null distribution.
    significant : bool
        True if p_value < alpha (the significance level passed to the test).
    se_reference : float
        Bootstrap standard error of the reference Gini estimator.
    se_monitor : float
        Bootstrap standard error of the monitor Gini estimator.
    n_reference : int
        Number of observations in the reference period.
    n_monitor : int
        Number of observations in the monitoring period.
    alpha : float
        Significance level used for the ``significant`` flag.
    n_bootstrap : int
        Number of bootstrap replicates used for variance estimation.

    Notes
    -----
    Both SE estimates use the bootstrap (Algorithm 2 from arXiv:2510.04556).
    For large samples (n > 20,000) the variance is estimated on a random
    subsample of 20,000 observations — the Gini point estimates still use the
    full sample. This bounds runtime to O(n_bootstrap * 20k * log(20k)).
    """

    gini_reference: float
    gini_monitor: float
    delta: float
    z_statistic: float
    p_value: float
    significant: bool
    se_reference: float
    se_monitor: float
    n_reference: int
    n_monitor: int
    alpha: float
    n_bootstrap: int

    def __getitem__(self, key: str):
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'key in result' checks for backward compatibility."""
        return hasattr(self, key)

    def keys(self):
        """Return field names, like dict.keys(), for backward compatibility."""
        return [f.name for f in dc_fields(self)]

    def to_dict(self) -> dict:
        """Return a plain dict of all fields (JSON-serialisable)."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


class GiniDriftTest:
    """Two-sample asymptotic z-test for Gini coefficient drift.

    Tests H0: Gini(reference) = Gini(monitor) against the two-sided
    alternative H1: Gini(reference) != Gini(monitor).

    The variance of each Gini estimator is estimated by non-parametric
    bootstrap (Algorithm 2, arXiv:2510.04556). The two variances are combined
    under the independence assumption::

        SE_total = sqrt(Var_bootstrap(G_ref) + Var_bootstrap(G_mon))
        z = (G_mon - G_ref) / SE_total
        p = 2 * (1 - Phi(|z|))

    Negative z means the monitor Gini is lower (discrimination has degraded).

    Both raw datasets are required. If you only have a stored training Gini
    scalar, use ``GiniDriftBootstrapTest`` instead.

    Parameters
    ----------
    reference_actual : array-like
        Observed losses or claim counts in the reference period.
    reference_predicted : array-like
        Model predictions for the reference period.
    monitor_actual : array-like
        Observed losses or claim counts in the monitoring period.
    monitor_predicted : array-like
        Model predictions for the monitoring period.
    reference_exposure : array-like, optional
        Exposure weights for the reference period (e.g. earned car-years).
        Use this when policies have varying durations to avoid giving equal
        weight to a one-month policy and a full-year policy.
    monitor_exposure : array-like, optional
        Exposure weights for the monitoring period.
    n_bootstrap : int, default 200
        Number of bootstrap replicates for variance estimation. 200 is
        sufficient for routine monitoring; use 500+ for regulatory reports.
        Must be >= 50.
    alpha : float, default 0.32
        Significance level for the ``significant`` flag. Default 0.32 is the
        one-sigma rule recommended by arXiv:2510.04556 for routine monitoring:
        it gives earlier signals at the cost of more false positives. Use
        alpha=0.05 for formal confirmatory testing or governance escalation.
    random_state : int or None, default None
        Seed for reproducibility. None gives non-reproducible results.

    Examples
    --------
    Basic usage — test whether the model's ranking power has changed::

        import numpy as np
        from insurance_monitoring.gini_drift import GiniDriftTest

        rng = np.random.default_rng(0)
        pred_ref = rng.uniform(0.05, 0.20, 10_000)
        act_ref = rng.poisson(pred_ref).astype(float)
        pred_mon = rng.uniform(0.05, 0.20, 5_000)
        act_mon = rng.poisson(pred_mon).astype(float)

        test = GiniDriftTest(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            monitor_actual=act_mon,
            monitor_predicted=pred_mon,
            n_bootstrap=200,
            random_state=42,
        )
        result = test.test()
        print(f"Gini ref: {result.gini_reference:.3f}")
        print(f"Gini mon: {result.gini_monitor:.3f}")
        print(f"Delta: {result.delta:+.3f}")
        print(f"z = {result.z_statistic:.2f}, p = {result.p_value:.4f}")
        print(f"Significant at alpha={result.alpha}: {result.significant}")

    With exposure weighting::

        exposure_ref = rng.uniform(0.1, 1.0, 10_000)
        exposure_mon = rng.uniform(0.1, 1.0, 5_000)
        test = GiniDriftTest(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            monitor_actual=act_mon,
            monitor_predicted=pred_mon,
            reference_exposure=exposure_ref,
            monitor_exposure=exposure_mon,
        )
        result = test.test()

    Governance text report::

        print(test.summary())
    """

    def __init__(
        self,
        reference_actual: ArrayLike,
        reference_predicted: ArrayLike,
        monitor_actual: ArrayLike,
        monitor_predicted: ArrayLike,
        reference_exposure: Optional[ArrayLike] = None,
        monitor_exposure: Optional[ArrayLike] = None,
        n_bootstrap: int = 200,
        alpha: float = 0.32,
        random_state: Optional[int] = None,
    ) -> None:
        if n_bootstrap < 50:
            raise ValueError(f"n_bootstrap must be >= 50, got {n_bootstrap}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.reference_actual = reference_actual
        self.reference_predicted = reference_predicted
        self.monitor_actual = monitor_actual
        self.monitor_predicted = monitor_predicted
        self.reference_exposure = reference_exposure
        self.monitor_exposure = monitor_exposure
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self._result: Optional[GiniDriftTestResult] = None

    def test(self) -> GiniDriftTestResult:
        """Run the asymptotic Gini drift z-test.

        Returns a ``GiniDriftTestResult`` with Gini estimates, z-statistic,
        p-value, SEs, and the ``significant`` flag.

        Runs on first call; subsequent calls return the cached result
        (idempotent — safe to call multiple times).

        Returns
        -------
        GiniDriftTestResult

        Raises
        ------
        ValueError
            If inputs are empty, have mismatched lengths, or exposure arrays
            have mismatched lengths.
        UserWarning
            If either sample has fewer than 200 observations. The bootstrap
            Gini distribution may not be well-approximated by Normal at
            small n; interpret p-values cautiously.
        """
        if self._result is not None:
            return self._result

        # Convert inputs
        act_ref = _to_numpy(self.reference_actual)
        pred_ref = _to_numpy(self.reference_predicted)
        exp_ref = _to_numpy_optional(self.reference_exposure)

        act_mon = _to_numpy(self.monitor_actual)
        pred_mon = _to_numpy(self.monitor_predicted)
        exp_mon = _to_numpy_optional(self.monitor_exposure)

        # Validate reference arrays
        n_ref = len(act_ref)
        if n_ref == 0:
            raise ValueError("reference_actual must be non-empty")
        if len(pred_ref) != n_ref:
            raise ValueError(
                f"reference_actual length ({n_ref}) != "
                f"reference_predicted length ({len(pred_ref)})"
            )
        if exp_ref is not None and len(exp_ref) != n_ref:
            raise ValueError(
                f"reference_exposure length ({len(exp_ref)}) != "
                f"reference_actual length ({n_ref})"
            )

        # Validate monitor arrays
        n_mon = len(act_mon)
        if n_mon == 0:
            raise ValueError("monitor_actual must be non-empty")
        if len(pred_mon) != n_mon:
            raise ValueError(
                f"monitor_actual length ({n_mon}) != "
                f"monitor_predicted length ({len(pred_mon)})"
            )
        if exp_mon is not None and len(exp_mon) != n_mon:
            raise ValueError(
                f"monitor_exposure length ({len(exp_mon)}) != "
                f"monitor_actual length ({n_mon})"
            )

        # Exposure non-negativity checks
        if exp_ref is not None and np.any(exp_ref <= 0):
            raise ValueError("All reference_exposure values must be positive")
        if exp_mon is not None and np.any(exp_mon <= 0):
            raise ValueError("All monitor_exposure values must be positive")

        # Small-sample warnings
        if n_ref < 200:
            warnings.warn(
                f"reference sample has only {n_ref} observations. "
                "Bootstrap Gini distribution may not be well-approximated "
                "by Normal; interpret p-values cautiously.",
                UserWarning,
                stacklevel=2,
            )
        if n_mon < 200:
            warnings.warn(
                f"monitor sample has only {n_mon} observations. "
                "Bootstrap Gini distribution may not be well-approximated "
                "by Normal; interpret p-values cautiously.",
                UserWarning,
                stacklevel=2,
            )

        # Gini point estimates (always full sample)
        g_ref = _gini_from_arrays(act_ref, pred_ref, exp_ref)
        g_mon = _gini_from_arrays(act_mon, pred_mon, exp_mon)

        # Bootstrap variance for reference period
        # Use two independent seeds derived from random_state so variance
        # estimates are independent even if arrays happen to be identical.
        seed_ref: Optional[int] = None
        seed_mon: Optional[int] = None
        if self.random_state is not None:
            rng_meta = np.random.default_rng(self.random_state)
            seed_ref = int(rng_meta.integers(0, 2**31))
            seed_mon = int(rng_meta.integers(0, 2**31))

        boot_ref = _bootstrap_gini_samples(
            act_ref, pred_ref, exp_ref, self.n_bootstrap, seed=seed_ref
        )
        boot_mon = _bootstrap_gini_samples(
            act_mon, pred_mon, exp_mon, self.n_bootstrap, seed=seed_mon
        )

        var_ref = float(np.var(boot_ref, ddof=1))
        var_mon = float(np.var(boot_mon, ddof=1))
        se_ref = float(np.sqrt(var_ref))
        se_mon = float(np.sqrt(var_mon))

        se_total = float(np.sqrt(var_ref + var_mon))

        delta = g_mon - g_ref

        if se_total == 0.0:
            # Both bootstrap distributions collapsed — degenerate data
            self._result = GiniDriftTestResult(
                gini_reference=float(g_ref),
                gini_monitor=float(g_mon),
                delta=float(delta),
                z_statistic=float("nan"),
                p_value=float("nan"),
                significant=False,
                se_reference=se_ref,
                se_monitor=se_mon,
                n_reference=n_ref,
                n_monitor=n_mon,
                alpha=self.alpha,
                n_bootstrap=self.n_bootstrap,
            )
            return self._result

        z = delta / se_total
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

        self._result = GiniDriftTestResult(
            gini_reference=float(g_ref),
            gini_monitor=float(g_mon),
            delta=float(delta),
            z_statistic=float(z),
            p_value=float(p_value),
            significant=bool(p_value < self.alpha),
            se_reference=se_ref,
            se_monitor=se_mon,
            n_reference=n_ref,
            n_monitor=n_mon,
            alpha=self.alpha,
            n_bootstrap=self.n_bootstrap,
        )
        return self._result

    def summary(self) -> str:
        """Return a governance-ready plain-text summary of the test result.

        Runs ``test()`` if not already called.

        Returns
        -------
        str
            A short paragraph suitable for model validation reports or
            monitoring dashboards.
        """
        r = self.test()

        direction = "declined" if r.delta < 0 else "improved"
        verdict = "DRIFT DETECTED" if r.significant else "no significant drift"

        lines = [
            "Gini Drift Test (arXiv:2510.04556 asymptotic z-test)",
            "-" * 52,
            f"Reference period : Gini = {r.gini_reference:.4f}  (n={r.n_reference:,}, "
            f"SE={r.se_reference:.4f})",
            f"Monitor period   : Gini = {r.gini_monitor:.4f}  (n={r.n_monitor:,}, "
            f"SE={r.se_monitor:.4f})",
            f"Delta (mon-ref)  : {r.delta:+.4f}  ({direction})",
            f"z-statistic      : {r.z_statistic:.3f}",
            f"p-value          : {r.p_value:.4f}",
            f"Alpha            : {r.alpha}",
            f"Verdict          : {verdict}",
            "",
            f"Bootstrap replicates: {r.n_bootstrap} per period.",
        ]

        if r.significant:
            lines.append(
                f"The Gini coefficient has {direction} by "
                f"{abs(r.delta):.4f} ({abs(r.delta)/max(r.gini_reference, 1e-9)*100:.1f}% "
                f"relative). The change is statistically significant at "
                f"alpha={r.alpha} (z={r.z_statistic:.2f}, p={r.p_value:.4f}). "
                "Investigate for model or population drift."
            )
        else:
            lines.append(
                f"The Gini coefficient has {direction} by "
                f"{abs(r.delta):.4f} but the change is not statistically "
                f"significant at alpha={r.alpha} (z={r.z_statistic:.2f}, "
                f"p={r.p_value:.4f}). No action required at this threshold."
            )

        return "\n".join(lines)
