"""
Integrated model monitoring framework (Brauer, Menzel & Wüthrich 2025).

Implements the two-step monitoring procedure from arXiv:2510.04556:

  Step 1: Gini-based ranking drift test (Algorithm 3).
  Step 2: GMCB global miscalibration test (Algorithm 4a).
          LMCB local miscalibration test (Algorithm 4b).

Three-way decision:
  - Gini fires OR LMCB fires  -> REFIT
  - Only GMCB fires           -> RECALIBRATE
  - Nothing fires             -> REDEPLOY

**Alpha default: 0.32**. The paper (arXiv:2510.04556, Section 3.2.1 and
Figure 8) shows that alpha=0.05 has high Type II error for realistic drift
magnitudes in insurance. The default 0.32 (one-sigma equivalent) is
recommended for ongoing production monitoring. A false alarm triggers a model
review, not an automatic refit — the cost is low. A missed detection allows
a degraded model to continue pricing, which is costly. Calibrate alpha to
your specific Type I/II cost ratio.

**Time-splitting pitfall**: always pass data aggregated to policyholder level.
Row-per-claim ETL pipelines inflate deviance ~10x and deflate Gini by ~4%.
A UserWarning is emitted when np.median(exposure) < 0.05.

References
----------
Brauer, Menzel & Wüthrich (2025), arXiv:2510.04556v2.

Examples
--------
::

    import numpy as np
    from insurance_monitoring import ModelMonitor

    rng = np.random.default_rng(42)
    exposure = rng.uniform(0.5, 2.0, 10_000)
    y_hat = rng.gamma(2, 0.05, 10_000)
    y_ref = rng.poisson(exposure * y_hat) / exposure

    monitor = ModelMonitor(distribution='poisson', n_bootstrap=500,
                           alpha_gini=0.32, alpha_global=0.32, alpha_local=0.32,
                           random_state=0)
    monitor.fit(y_ref, y_hat, exposure)

    # Global shift scenario: true rates inflated by 10%
    y_new = rng.poisson(exposure * y_hat * 1.1) / exposure
    result = monitor.test(y_new, y_hat, exposure)
    print(result.decision)   # 'RECALIBRATE'
    print(result.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields as dc_fields
from typing import Optional

import numpy as np
from scipy import stats

from insurance_monitoring.discrimination import (
    _gini_from_arrays,
    _bootstrap_gini_samples,
    _to_numpy,
    _to_numpy_optional,
    ArrayLike,
)
from insurance_monitoring.gini_monitoring import (
    _validate_pair,
    _warn_small,
    _derive_seeds,
)
from insurance_monitoring.calibration._gmcb_lmcb import check_gmcb, check_lmcb
from insurance_monitoring.calibration._types import GMCBResult, LMCBResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelMonitorResult:
    """Full monitoring result combining Gini drift + calibration tests.

    Produced by :meth:`ModelMonitor.test`. Contains all per-test results and
    the integrated decision.

    Attributes
    ----------
    gini_reference
        E_hat[G] from the reference period bootstrap (Algorithm 2).
    gini_se
        sigma_hat[G] from the reference period bootstrap (Algorithm 2).
    gini_new
        hat_G on the new monitoring period data (Algorithm 3).
    gini_z
        (gini_new - gini_reference) / gini_se. Negative means degraded ranking.
    gini_p
        Two-sided p-value under N(0,1) null. Small values indicate Gini drift.
    gini_significant
        True if gini_p < alpha_gini.
    gmcb_score
        Observed GMCB = D(y, y_hat) - D(y, alpha * y_hat). Always >= 0.
    gmcb_p
        Bootstrap p-value for H_0: GMCB = 0.
    gmcb_significant
        True if gmcb_p < alpha_global.
    lmcb_score
        Observed LMCB = D(y, y_hat_bc) - D(y, y_hat_bc_rc).
    lmcb_p
        Bootstrap p-value for H_0: LMCB = 0.
    lmcb_significant
        True if lmcb_p < alpha_local.
    decision
        'REDEPLOY', 'RECALIBRATE', or 'REFIT'. See :class:`ModelMonitor`
        for the decision rules.
    decision_reason
        Plain-text explanation of the decision.
    balance_factor
        Global balance correction factor alpha = sum(w*y) / sum(w*y_hat).
    n_new
        Number of observations in the monitoring period.
    distribution
        EDF distribution family used.
    alpha_gini
        Significance level used for the Gini test.
    alpha_global
        Significance level used for the GMCB test.
    alpha_local
        Significance level used for the LMCB test.
    """

    # Algorithm 3: Gini drift
    gini_reference: float
    gini_se: float
    gini_new: float
    gini_z: float
    gini_p: float
    gini_significant: bool

    # Algorithm 4a: global calibration
    gmcb_score: float
    gmcb_p: float
    gmcb_significant: bool

    # Algorithm 4b: local calibration
    lmcb_score: float
    lmcb_p: float
    lmcb_significant: bool

    # Decision
    decision: str
    decision_reason: str
    balance_factor: float

    # Metadata
    n_new: int
    distribution: str
    alpha_gini: float
    alpha_global: float
    alpha_local: float

    def summary(self) -> str:
        """Return a governance-ready paragraph summarising the result.

        Returns
        -------
        str
            Plain-text summary suitable for PRA model validation reports.
        """
        gini_line = (
            f"Gini ranking test: G_new={self.gini_new:.4f} vs "
            f"G_ref={self.gini_reference:.4f} "
            f"(SE={self.gini_se:.4f}), z={self.gini_z:.2f}, "
            f"p={self.gini_p:.4f} "
            f"({'rejected' if self.gini_significant else 'not rejected'} "
            f"at alpha={self.alpha_gini})."
        )
        gmcb_line = (
            f"GMCB test: score={self.gmcb_score:.6f}, p={self.gmcb_p:.4f} "
            f"({'rejected' if self.gmcb_significant else 'not rejected'} "
            f"at alpha={self.alpha_global}). "
            f"Balance factor={self.balance_factor:.4f}."
        )
        lmcb_line = (
            f"LMCB test: score={self.lmcb_score:.6f}, p={self.lmcb_p:.4f} "
            f"({'rejected' if self.lmcb_significant else 'not rejected'} "
            f"at alpha={self.alpha_local})."
        )
        decision_line = f"Decision: {self.decision}. {self.decision_reason}"
        return f"{gini_line} {gmcb_line} {lmcb_line} {decision_line}"

    def to_dict(self) -> dict:
        """Return all fields as a JSON-serialisable dict."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ---------------------------------------------------------------------------
# ModelMonitor
# ---------------------------------------------------------------------------


class ModelMonitor:
    """Integrated model monitoring framework (Brauer, Menzel & Wüthrich 2025).

    Implements the two-step monitoring procedure from arXiv:2510.04556:

    - Step 1 (Algorithm 3): Gini-based ranking drift test. Detects whether the
      model's discriminatory power has degraded in the new period.
    - Step 2 (Algorithms 4a + 4b): Bootstrap auto-calibration tests. GMCB
      tests for global level shift (fixable by balance correction); LMCB tests
      for local cohort-level miscalibration (requires refit).

    Decision rules (from paper Section 3.1)::

        gini_sig OR lmcb_sig             -> REFIT
        not gini_sig AND not lmcb_sig AND gmcb_sig -> RECALIBRATE
        not gini_sig AND not gmcb_sig AND not lmcb_sig -> REDEPLOY

    **Alpha default 0.32** for all three tests. The paper demonstrates (Figure 8)
    that alpha=0.05 has high Type II error for realistic insurance drift magnitudes.
    Using 0.32 (one-sigma rule) gives earlier warnings. A false alarm costs a model
    review; a missed detection costs mis-pricing.

    Parameters
    ----------
    distribution : str, default 'poisson'
        EDF distribution family. One of 'poisson', 'gamma', 'tweedie', 'normal'.
    tweedie_power : float, default 1.5
        Tweedie variance power. Only used when distribution='tweedie'.
    n_bootstrap : int, default 500
        Bootstrap replicates for Gini SE estimation at fit time and for
        GMCB/LMCB tests at test time. B >= 200 gives stable p-values.
        B = 500 is recommended for governance reporting.
    alpha_gini : float, default 0.32
        Significance level for the Gini ranking drift test.
    alpha_global : float, default 0.32
        Significance level for the GMCB (global calibration) test.
    alpha_local : float, default 0.32
        Significance level for the LMCB (local calibration) test.
    random_state : int or None, default None
        Seed for bootstrap reproducibility.

    Raises
    ------
    ValueError
        If distribution is not recognised or if alpha values are out of (0, 1).

    Examples
    --------
    ::

        import numpy as np
        from insurance_monitoring import ModelMonitor

        rng = np.random.default_rng(42)
        exposure = rng.uniform(0.5, 2.0, 10_000)
        y_hat = rng.gamma(2, 0.05, 10_000)
        y_ref = rng.poisson(exposure * y_hat) / exposure

        monitor = ModelMonitor(distribution='poisson', n_bootstrap=500,
                               random_state=0)
        monitor.fit(y_ref, y_hat, exposure)

        y_new = rng.poisson(exposure * y_hat * 1.1) / exposure  # 10% inflation
        result = monitor.test(y_new, y_hat, exposure)
        print(result.decision)  # 'RECALIBRATE'
    """

    _VALID_DISTRIBUTIONS = frozenset({"poisson", "gamma", "tweedie", "normal"})

    def __init__(
        self,
        distribution: str = "poisson",
        tweedie_power: float = 1.5,
        n_bootstrap: int = 500,
        alpha_gini: float = 0.32,
        alpha_global: float = 0.32,
        alpha_local: float = 0.32,
        random_state: Optional[int] = None,
    ) -> None:
        if distribution not in self._VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of "
                f"{sorted(self._VALID_DISTRIBUTIONS)}, got '{distribution}'"
            )
        if n_bootstrap < 50:
            raise ValueError(f"n_bootstrap must be >= 50, got {n_bootstrap}")
        for name, val in [
            ("alpha_gini", alpha_gini),
            ("alpha_global", alpha_global),
            ("alpha_local", alpha_local),
        ]:
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {val}")

        self.distribution = distribution
        self.tweedie_power = tweedie_power
        self.n_bootstrap = n_bootstrap
        self.alpha_gini = alpha_gini
        self.alpha_global = alpha_global
        self.alpha_local = alpha_local
        self.random_state = random_state

        # Set at fit time (Algorithm 2)
        self._gini_ref_mean: Optional[float] = None
        self._gini_ref_std: Optional[float] = None
        self._is_fitted: bool = False
        self._last_result: Optional[ModelMonitorResult] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        y_ref: ArrayLike,
        y_hat_ref: ArrayLike,
        exposure_ref: Optional[ArrayLike] = None,
    ) -> "ModelMonitor":
        """Run Algorithm 2 on reference data. Stores bootstrap Gini distribution.

        Must be called before :meth:`test`. Uses the bootstrap to estimate the
        mean and standard deviation of the reference Gini. These are used in
        Algorithm 3 to form the z-statistic for the new period.

        For n > 20,000, the bootstrap is computed on a random subsample of
        20,000 observations. The point estimate in test() uses the full sample.

        Parameters
        ----------
        y_ref : array-like
            Observed loss rates in the reference period. Shape (n,).
        y_hat_ref : array-like
            Model predictions for the reference period. Shape (n,).
        exposure_ref : array-like, optional
            Exposure weights. If None, assumed uniform = 1.0.

        Returns
        -------
        ModelMonitor
            self (for method chaining).

        Raises
        ------
        ValueError
            If arrays are empty, have mismatched lengths, or exposures are
            non-positive.
        UserWarning
            If np.median(exposure) < 0.05 (time-splitting pitfall).
        """
        y = _to_numpy(y_ref)
        mu = _to_numpy(y_hat_ref)
        exp = _to_numpy_optional(exposure_ref)
        _validate_pair(y, mu, exp, "reference")

        if exp is not None and float(np.median(exp)) < 0.05:
            warnings.warn(
                "Median reference exposure is below 0.05. Check that data has "
                "been aggregated to policyholder level before calling fit(). "
                "Row-per-claim data inflates deviance ~10x. "
                "See Section 3.3 of arXiv:2510.04556.",
                UserWarning,
                stacklevel=2,
            )

        seed, _ = _derive_seeds(self.random_state)
        boot_samples = _bootstrap_gini_samples(y, mu, exp, self.n_bootstrap, seed=seed)
        self._gini_ref_mean = float(np.mean(boot_samples))
        self._gini_ref_std = float(np.std(boot_samples, ddof=1))
        self._is_fitted = True
        return self

    def test(
        self,
        y_new: ArrayLike,
        y_hat_new: ArrayLike,
        exposure_new: Optional[ArrayLike] = None,
    ) -> ModelMonitorResult:
        """Run Algorithms 3 and 4 on new period data. Returns decision result.

        Runs all three tests regardless of intermediate outcomes; the decision
        is determined after all p-values are available.

        Parameters
        ----------
        y_new : array-like
            Observed loss rates in the monitoring period. Shape (m,).
        y_hat_new : array-like
            Model predictions for the monitoring period. Shape (m,).
        exposure_new : array-like, optional
            Exposure weights for the monitoring period. Shape (m,).

        Returns
        -------
        ModelMonitorResult
            All test results, p-values, and the three-way decision.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        UserWarning
            If np.median(exposure) < 0.05 (time-splitting pitfall).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ModelMonitor must be fitted before calling test(). "
                "Call fit(y_ref, y_hat_ref) first."
            )

        y = _to_numpy(y_new)
        mu = _to_numpy(y_hat_new)
        exp = _to_numpy_optional(exposure_new)
        _validate_pair(y, mu, exp, "monitor")

        n_new = len(y)
        _warn_small(n_new, "monitor")

        # ------------------------------------------------------------------
        # Algorithm 3: Gini ranking drift test
        # Uses sigma_hat[G_old] (reference bootstrap SE) in the denominator,
        # not the monitor bootstrap SE. This is the one-sample test design
        # from the paper: the reference distribution is fixed at fit time.
        # ------------------------------------------------------------------
        g_new = float(_gini_from_arrays(y, mu, exp))
        ref_mean = self._gini_ref_mean
        ref_std = self._gini_ref_std

        if ref_std is None or ref_std == 0.0:
            gini_z = float("nan")
            gini_p = float("nan")
            gini_sig = False
        else:
            gini_z = (g_new - ref_mean) / ref_std
            gini_p = float(2.0 * (1.0 - stats.norm.cdf(abs(gini_z))))
            gini_sig = bool(gini_p < self.alpha_gini)

        # ------------------------------------------------------------------
        # Algorithm 4a: GMCB bootstrap test
        # ------------------------------------------------------------------
        _, seed_test = _derive_seeds(self.random_state)
        gmcb_result: GMCBResult = check_gmcb(
            y=y,
            y_hat=mu,
            exposure=exp,
            distribution=self.distribution,
            bootstrap_n=self.n_bootstrap,
            significance_level=self.alpha_global,
            seed=seed_test,
            tweedie_power=self.tweedie_power,
        )

        # ------------------------------------------------------------------
        # Algorithm 4b: LMCB bootstrap test
        # ------------------------------------------------------------------
        # Use a different seed for the LMCB bootstrap to ensure independence
        if seed_test is not None:
            lmcb_seed = seed_test + 1
        else:
            lmcb_seed = None

        lmcb_result: LMCBResult = check_lmcb(
            y=y,
            y_hat=mu,
            exposure=exp,
            distribution=self.distribution,
            bootstrap_n=self.n_bootstrap,
            significance_level=self.alpha_local,
            seed=lmcb_seed,
            tweedie_power=self.tweedie_power,
        )

        # ------------------------------------------------------------------
        # Decision logic (Section 3.1 of arXiv:2510.04556)
        # ------------------------------------------------------------------
        gmcb_sig = gmcb_result.is_significant
        lmcb_sig = lmcb_result.is_significant

        decision, reason = _make_decision(
            gini_sig=gini_sig,
            gini_z=gini_z,
            gini_p=gini_p,
            gmcb_sig=gmcb_sig,
            gmcb_score=gmcb_result.gmcb_score,
            gmcb_p=gmcb_result.p_value,
            lmcb_sig=lmcb_sig,
            lmcb_score=lmcb_result.lmcb_score,
            lmcb_p=lmcb_result.p_value,
            alpha_gini=self.alpha_gini,
            alpha_global=self.alpha_global,
            alpha_local=self.alpha_local,
        )

        result = ModelMonitorResult(
            gini_reference=float(ref_mean) if ref_mean is not None else float("nan"),
            gini_se=float(ref_std) if ref_std is not None else float("nan"),
            gini_new=g_new,
            gini_z=gini_z,
            gini_p=gini_p,
            gini_significant=gini_sig,
            gmcb_score=gmcb_result.gmcb_score,
            gmcb_p=gmcb_result.p_value,
            gmcb_significant=gmcb_sig,
            lmcb_score=lmcb_result.lmcb_score,
            lmcb_p=lmcb_result.p_value,
            lmcb_significant=lmcb_sig,
            decision=decision,
            decision_reason=reason,
            balance_factor=gmcb_result.balance_factor,
            n_new=n_new,
            distribution=self.distribution,
            alpha_gini=self.alpha_gini,
            alpha_global=self.alpha_global,
            alpha_local=self.alpha_local,
        )
        self._last_result = result
        return result

    def is_fitted(self) -> bool:
        """Return True if fit() has been called."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ModelMonitor(distribution='{self.distribution}', "
            f"n_bootstrap={self.n_bootstrap}, "
            f"alpha_gini={self.alpha_gini}, "
            f"alpha_global={self.alpha_global}, "
            f"alpha_local={self.alpha_local}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_decision(
    gini_sig: bool,
    gini_z: float,
    gini_p: float,
    gmcb_sig: bool,
    gmcb_score: float,
    gmcb_p: float,
    lmcb_sig: bool,
    lmcb_score: float,
    lmcb_p: float,
    alpha_gini: float,
    alpha_global: float,
    alpha_local: float,
) -> tuple[str, str]:
    """Apply the three-way decision rule from paper Section 3.1.

    Returns
    -------
    tuple[str, str]
        (decision, reason_text).
    """
    if not gini_sig and not gmcb_sig and not lmcb_sig:
        decision = "REDEPLOY"
        reason = (
            "No significant drift detected in ranking or calibration. "
            "The model remains adequate for deployment."
        )

    elif not gini_sig and not lmcb_sig and gmcb_sig:
        decision = "RECALIBRATE"
        reason = (
            f"Global level shift detected (GMCB={gmcb_score:.6f}, "
            f"p={gmcb_p:.4f}, alpha={alpha_global}). "
            "Apply balance correction h^{-1}(beta_0 + beta_1 * h(y_hat)). "
            "Ranking and local calibration remain intact — no model refit required."
        )

    else:
        parts = []
        if gini_sig:
            parts.append(
                f"ranking drift (Gini z={gini_z:.2f}, p={gini_p:.4f}, "
                f"alpha={alpha_gini})"
            )
        if lmcb_sig:
            parts.append(
                f"local miscalibration (LMCB={lmcb_score:.6f}, p={lmcb_p:.4f}, "
                f"alpha={alpha_local})"
            )
        decision = "REFIT"
        reason = (
            "Structural drift detected: " + " and ".join(parts) + ". "
            "Model refit required. Balance correction alone is insufficient."
        )

    return decision, reason
