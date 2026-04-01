"""
Two-step annual pricing model monitoring framework (Brauer, Menzel, Wüthrich 2025).

This module implements the full monitoring workflow from arXiv:2510.04556:

- **Step 1 — Gini ranking drift test** (Algorithm 3): tests whether the model's
  discriminatory power has changed. The null hypothesis is that the monitor Gini
  comes from the same bootstrap distribution as the reference Gini. A significant
  result means rank structure has degraded and a model refit is required.

- **Step 2 — Auto-calibration tests** (Algorithm 4): decomposes miscalibration
  into GMCB (global level shift, fixable by balance correction) and LMCB (local
  cohort-level structure, requires refit). Formal bootstrap p-values are produced
  for each component.

- **Decision logic**:
  - Gini rejected OR LMCB rejected → REFIT
  - Only GMCB rejected → RECALIBRATE
  - All pass → OK

Design notes
------------
``PricingDriftMonitor`` orchestrates the existing ``GiniBootstrapMonitor`` and
``MurphyDecomposition`` classes — it does not duplicate their code. The Gini
z-test here uses the reference bootstrap distribution (sigma_hat[G_old]) in the
denominator, not a one-sample test treating G_old as fixed. This is the critical
distinction from ``GiniBootstrapMonitor.test()``, which uses sigma_hat[G_new].

The balance-correction GLM (mu_bc) is fitted by ``scipy.optimize.minimize`` using
the Poisson negative log-likelihood. This is a 2-parameter log-linear GLM; no
new dependencies are required.

References
----------
Brauer, Menzel & Wüthrich (2025), arXiv:2510.04556 (v2, Dec 2025).

Examples
--------
::

    import numpy as np
    from insurance_monitoring.pricing_drift import PricingDriftMonitor

    rng = np.random.default_rng(42)
    mu_ref = rng.uniform(0.05, 0.20, 10_000)
    y_ref = rng.poisson(mu_ref).astype(float)
    exposure_ref = rng.uniform(0.5, 2.0, 10_000)

    monitor = PricingDriftMonitor(distribution='poisson', n_bootstrap=500,
                                   alpha_gini=0.32, random_state=0)
    monitor.fit(y_ref, mu_ref, exposure=exposure_ref)

    mu_mon = rng.uniform(0.05, 0.20, 5_000)
    y_mon = rng.poisson(mu_mon).astype(float)
    result = monitor.test(y_mon, mu_mon, exposure=rng.uniform(0.5, 2.0, 5_000))
    print(result.verdict)    # 'OK', 'RECALIBRATE', or 'REFIT'
    print(result.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields as dc_fields
from typing import Optional

import numpy as np
import scipy.optimize
from scipy import stats

from insurance_monitoring.gini_monitoring import (
    GiniBootstrapResult,
    _MurphyComponents,
    _validate_pair,
    _warn_small,
    _derive_seeds,
)
from insurance_monitoring.discrimination import (
    _gini_from_arrays,
    _bootstrap_gini_samples,
    _to_numpy,
    _to_numpy_optional,
    ArrayLike,
)
from insurance_monitoring.calibration._murphy import murphy_decomposition
from insurance_monitoring.calibration._deviance import deviance
from insurance_monitoring.calibration._rectify import isotonic_recalibrate


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CalibTestResult:
    """Result of a single bootstrap auto-calibration test (GMCB or LMCB).

    Attributes
    ----------
    statistic : float
        Observed Murphy component value (GMCB or LMCB).
    p_value : float
        Bootstrap p-value. Fraction of simulated H0 values >= observed.
    reject_h0 : bool
        True when p_value < alpha.
    alpha : float
        Significance level used.
    n_bootstrap : int
        Number of bootstrap replicates.
    component : str
        'GMCB' or 'LMCB'.
    """

    statistic: float
    p_value: float
    reject_h0: bool
    alpha: float
    n_bootstrap: int
    component: str

    def to_dict(self) -> dict:
        """Return all fields as a plain dict."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


@dataclass
class PricingDriftResult:
    """Full result from :meth:`PricingDriftMonitor.test`.

    Attributes
    ----------
    verdict : str
        'OK', 'RECALIBRATE', or 'REFIT'. See decision logic in class docstring.
    gini : GiniBootstrapResult
        Gini ranking drift test result (Algorithm 3 of arXiv:2510.04556).
        The z-statistic denominator is sigma_hat[G_old] (reference bootstrap SE).
    murphy : _MurphyComponents
        Murphy decomposition components: UNC, DSC, MCB, GMCB, LMCB.
    global_calib : CalibTestResult
        GMCB bootstrap auto-calibration test.
    local_calib : CalibTestResult
        LMCB bootstrap auto-calibration test.
    n_mon : int
        Number of monitoring observations.
    distribution : str
        EDF distribution family used.
    alpha_gini : float
        Significance level used for the Gini test.
    alpha_global : float
        Significance level used for the GMCB test.
    alpha_local : float
        Significance level used for the LMCB test.
    """

    verdict: str
    gini: GiniBootstrapResult
    murphy: _MurphyComponents
    global_calib: CalibTestResult
    local_calib: CalibTestResult
    n_mon: int
    distribution: str
    alpha_gini: float
    alpha_global: float
    alpha_local: float

    def summary(self) -> str:
        """Return a governance-ready paragraph summarising the verdict and evidence.

        Suitable for inclusion in PRA model validation reports. Covers the Gini
        test, calibration test results, Murphy components, and the verdict with
        interpretation.

        Returns
        -------
        str
            One-paragraph plain-text summary.
        """
        g = self.gini
        m = self.murphy
        gc = self.global_calib
        lc = self.local_calib

        gini_line = (
            f"Gini ranking test: G_mon={g.gini_mon:.4f} vs G_ref={g.gini_ref:.4f} "
            f"(change {g.gini_change:+.4f}), z={g.z_stat:.2f}, p={g.p_value:.4f} "
            f"({'rejected' if g.reject_h0 else 'not rejected'} at alpha={self.alpha_gini})."
        )
        murphy_line = (
            f"Murphy decomposition: UNC={m.unc:.4f}, DSC={m.dsc:.4f} "
            f"({m.dsc_pct:.1f}%), MCB={m.mcb:.4f} ({m.mcb_pct:.1f}%), "
            f"GMCB={m.gmcb:.4f}, LMCB={m.lmcb:.4f}."
        )
        calib_line = (
            f"Global calibration test (GMCB): statistic={gc.statistic:.6f}, "
            f"p={gc.p_value:.4f} ({'rejected' if gc.reject_h0 else 'not rejected'} "
            f"at alpha={self.alpha_global}). "
            f"Local calibration test (LMCB): statistic={lc.statistic:.6f}, "
            f"p={lc.p_value:.4f} ({'rejected' if lc.reject_h0 else 'not rejected'} "
            f"at alpha={self.alpha_local})."
        )
        verdict_line = _interpret_verdict(self.verdict, g, m, gc, lc)
        return f"{gini_line} {murphy_line} {calib_line} {verdict_line}"

    def to_dict(self) -> dict:
        """Return all fields as a JSON-serialisable dict.

        Nested dataclasses (gini, murphy, global_calib, local_calib) are
        flattened with their field names prefixed.

        Returns
        -------
        dict
            Flat dict with all numeric values. Suitable for Delta table writes
            or JSON logging.
        """
        d: dict = {
            "verdict": self.verdict,
            "n_mon": self.n_mon,
            "distribution": self.distribution,
            "alpha_gini": self.alpha_gini,
            "alpha_global": self.alpha_global,
            "alpha_local": self.alpha_local,
        }
        for f in dc_fields(self.gini):
            d[f"gini_{f.name}"] = getattr(self.gini, f.name)
        for f in dc_fields(self.murphy):
            d[f"murphy_{f.name}"] = getattr(self.murphy, f.name)
        for f in dc_fields(self.global_calib):
            d[f"global_calib_{f.name}"] = getattr(self.global_calib, f.name)
        for f in dc_fields(self.local_calib):
            d[f"local_calib_{f.name}"] = getattr(self.local_calib, f.name)
        return d


# ---------------------------------------------------------------------------
# PricingDriftMonitor
# ---------------------------------------------------------------------------


class PricingDriftMonitor:
    """Two-step annual pricing model monitoring framework (Brauer, Menzel, Wüthrich 2025).

    Implements Algorithm 3 (Gini ranking drift test) followed by Algorithm 4
    (global + local auto-calibration test) from arXiv:2510.04556. Outputs a
    structured REFIT / RECALIBRATE / OK verdict for use in annual model
    governance reviews (SR 11/7, PRA SS1/23).

    Decision logic (hierarchical):

    1. **Gini test** (Algorithm 3): if the monitor Gini falls outside the
       reference bootstrap distribution, rank structure has degraded → REFIT.
    2. **LMCB test** (Algorithm 4b): if local cohort-level miscalibration is
       significant, the model's rank structure is locally wrong → REFIT.
    3. **GMCB test** (Algorithm 4a): if only global level shift is significant,
       a balance correction can fix the problem → RECALIBRATE.
    4. All pass → OK.

    **Critical distinction from GiniBootstrapMonitor**: The Gini z-test
    denominator here is ``sigma_hat[G_old]`` (the bootstrap standard deviation
    of the reference Gini), not ``sigma_hat[G_new]``. This implements the
    one-sample test against the reference distribution, as per paper Algorithm 3
    step 3.

    Parameters
    ----------
    distribution : str, default 'poisson'
        EDF family for deviance scoring. One of 'poisson', 'gamma',
        'tweedie', 'normal'.
    n_bootstrap : int, default 500
        Bootstrap replicates for the Gini test (at fit time) and the MCB
        auto-calibration bootstrap (at test time). B >= 200 gives stable
        p-values; B = 500 is recommended for governance reporting.
    alpha_gini : float, default 0.32
        Significance level for the Gini ranking test. The default is the
        one-sigma rule from arXiv:2510.04556 Remark 3: it catches moderate
        drift that alpha=0.05 misses. Use 0.05 for conservative governance.
    alpha_global : float, default 0.05
        Significance level for the GMCB (global calibration) test.
    alpha_local : float, default 0.05
        Significance level for the LMCB (local calibration) test.
    tweedie_power : float, default 1.5
        Tweedie variance power. Only used when distribution='tweedie'.
    random_state : int or None, default None
        Seed for bootstrap reproducibility.

    Examples
    --------
    ::

        import numpy as np
        from insurance_monitoring.pricing_drift import PricingDriftMonitor

        rng = np.random.default_rng(42)
        mu_ref = rng.uniform(0.05, 0.20, 10_000)
        y_ref = rng.poisson(mu_ref).astype(float)

        monitor = PricingDriftMonitor(
            distribution='poisson', n_bootstrap=500, alpha_gini=0.32, random_state=0
        )
        monitor.fit(y_ref, mu_ref)

        mu_mon = rng.uniform(0.05, 0.20, 5_000)
        y_mon = rng.poisson(mu_mon).astype(float)
        result = monitor.test(y_mon, mu_mon)

        print(result.verdict)    # 'OK', 'RECALIBRATE', or 'REFIT'
        print(result.summary())
    """

    def __init__(
        self,
        distribution: str = "poisson",
        n_bootstrap: int = 500,
        alpha_gini: float = 0.32,
        alpha_global: float = 0.05,
        alpha_local: float = 0.05,
        tweedie_power: float = 1.5,
        random_state: Optional[int] = None,
    ) -> None:
        _valid_distributions = {"poisson", "gamma", "tweedie", "normal"}
        if distribution not in _valid_distributions:
            raise ValueError(
                f"distribution must be one of {sorted(_valid_distributions)}, "
                f"got '{distribution}'"
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
        self.n_bootstrap = n_bootstrap
        self.alpha_gini = alpha_gini
        self.alpha_global = alpha_global
        self.alpha_local = alpha_local
        self.tweedie_power = tweedie_power
        self.random_state = random_state

        # Set at fit time
        self._gini_ref_mean: Optional[float] = None
        self._gini_ref_std: Optional[float] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        y_ref: ArrayLike,
        mu_hat_ref: ArrayLike,
        exposure: Optional[ArrayLike] = None,
    ) -> "PricingDriftMonitor":
        """Store reference period data and compute bootstrap Gini distribution.

        Parameters
        ----------
        y_ref : array-like
            Observed claims (Poisson counts) or loss rates in the reference
            period. Shape (n,).
        mu_hat_ref : array-like
            Model predictions for the reference period. Shape (n,). Must be
            strictly positive.
        exposure : array-like, optional
            Exposure weights (car-years). Shape (n,). If None, uniform.

        Returns
        -------
        PricingDriftMonitor
            self (for method chaining).

        Raises
        ------
        ValueError
            If arrays are empty, have mismatched lengths, or exposures contain
            non-positive values.
        """
        y = _to_numpy(y_ref)
        mu = _to_numpy(mu_hat_ref)
        exp = _to_numpy_optional(exposure)
        _validate_pair(y, mu, exp, "reference")

        seed, _ = _derive_seeds(self.random_state)
        boot_samples = _bootstrap_gini_samples(y, mu, exp, self.n_bootstrap, seed=seed)
        self._gini_ref_mean = float(np.mean(boot_samples))
        self._gini_ref_std = float(np.std(boot_samples, ddof=1))
        self._is_fitted = True
        return self

    def test(
        self,
        y_mon: ArrayLike,
        mu_hat_mon: ArrayLike,
        exposure: Optional[ArrayLike] = None,
        alpha_gini: Optional[float] = None,
        alpha_global: Optional[float] = None,
        alpha_local: Optional[float] = None,
    ) -> PricingDriftResult:
        """Run the two-step drift test against the stored reference.

        Both calibration tests always run regardless of the Gini test outcome;
        the verdict is determined after all three test results are available.

        Parameters
        ----------
        y_mon : array-like
            Observed claims in the monitoring period. Shape (m,).
        mu_hat_mon : array-like
            Model predictions for the monitoring period. Shape (m,).
        exposure : array-like, optional
            Exposure weights for the monitoring period. Shape (m,).
        alpha_gini : float, optional
            Override the Gini significance level for this call.
        alpha_global : float, optional
            Override the GMCB significance level for this call.
        alpha_local : float, optional
            Override the LMCB significance level for this call.

        Returns
        -------
        PricingDriftResult
            Structured result with verdict, component test results, and Murphy
            decomposition breakdown.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        UserWarning
            If the monitoring sample has fewer than 500 observations. The Murphy
            bootstrap isotonic regression may be unstable on very small samples.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PricingDriftMonitor must be fitted before calling test(). "
                "Call fit(y_ref, mu_hat_ref) first."
            )

        alpha_g = alpha_gini if alpha_gini is not None else self.alpha_gini
        alpha_gl = alpha_global if alpha_global is not None else self.alpha_global
        alpha_lo = alpha_local if alpha_local is not None else self.alpha_local

        y = _to_numpy(y_mon)
        mu = _to_numpy(mu_hat_mon)
        exp = _to_numpy_optional(exposure)
        _validate_pair(y, mu, exp, "monitor")

        n_mon = len(y)
        if n_mon < 500:
            warnings.warn(
                f"Monitoring sample has only {n_mon} observations. "
                "The MCB bootstrap relies on isotonic regression, which may be "
                "unstable for n < 500. Interpret p-values cautiously.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 1: Gini ranking drift test (Algorithm 3)
        # ------------------------------------------------------------------
        g_mon = float(_gini_from_arrays(y, mu, exp))

        ref_std = self._gini_ref_std
        ref_mean = self._gini_ref_mean

        if ref_std is None or ref_std == 0.0:
            z = float("nan")
            p_gini = float("nan")
            gini_reject = False
        else:
            z = (g_mon - ref_mean) / ref_std
            p_gini = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            gini_reject = bool(p_gini < alpha_g)

        gini_result = GiniBootstrapResult(
            gini_ref=float(ref_mean),
            gini_mon=g_mon,
            gini_change=g_mon - float(ref_mean),
            z_stat=z,
            p_value=p_gini,
            reject_h0=gini_reject,
            se_bootstrap=float(ref_std) if ref_std is not None else float("nan"),
            n_mon=n_mon,
            alpha=alpha_g,
            n_bootstrap=self.n_bootstrap,
        )

        # ------------------------------------------------------------------
        # Step 2: Murphy decomposition
        # ------------------------------------------------------------------
        murphy_raw = murphy_decomposition(
            y=y,
            y_hat=mu,
            exposure=exp,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
        )
        murphy_components = _MurphyComponents(
            unc=murphy_raw.uncertainty,
            dsc=murphy_raw.discrimination,
            mcb=murphy_raw.miscalibration,
            gmcb=murphy_raw.global_mcb,
            lmcb=murphy_raw.local_mcb,
            total_deviance=murphy_raw.total_deviance,
            dsc_pct=murphy_raw.discrimination_pct,
            mcb_pct=murphy_raw.miscalibration_pct,
            verdict=murphy_raw.verdict,
        )

        # ------------------------------------------------------------------
        # Step 2b: GMCB and LMCB bootstrap p-values
        # ------------------------------------------------------------------
        _, seed_test = _derive_seeds(self.random_state)
        rng = np.random.default_rng(seed_test)

        exp_arr = exp if exp is not None else np.ones(n_mon, dtype=np.float64)

        p_gmcb = _mcb_component_bootstrap_pvalue(
            y=y,
            mu_hat=mu,
            exposure=exp_arr,
            component="GMCB",
            observed_value=murphy_raw.global_mcb,
            n_bootstrap=self.n_bootstrap,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
            rng=rng,
        )
        p_lmcb = _mcb_component_bootstrap_pvalue(
            y=y,
            mu_hat=mu,
            exposure=exp_arr,
            component="LMCB",
            observed_value=murphy_raw.local_mcb,
            n_bootstrap=self.n_bootstrap,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
            rng=rng,
        )

        global_result = CalibTestResult(
            statistic=murphy_raw.global_mcb,
            p_value=p_gmcb,
            reject_h0=bool(p_gmcb < alpha_gl),
            alpha=alpha_gl,
            n_bootstrap=self.n_bootstrap,
            component="GMCB",
        )
        local_result = CalibTestResult(
            statistic=murphy_raw.local_mcb,
            p_value=p_lmcb,
            reject_h0=bool(p_lmcb < alpha_lo),
            alpha=alpha_lo,
            n_bootstrap=self.n_bootstrap,
            component="LMCB",
        )

        # ------------------------------------------------------------------
        # Decision logic
        # ------------------------------------------------------------------
        if gini_result.reject_h0 or local_result.reject_h0:
            verdict = "REFIT"
        elif global_result.reject_h0:
            verdict = "RECALIBRATE"
        else:
            verdict = "OK"

        return PricingDriftResult(
            verdict=verdict,
            gini=gini_result,
            murphy=murphy_components,
            global_calib=global_result,
            local_calib=local_result,
            n_mon=n_mon,
            distribution=self.distribution,
            alpha_gini=alpha_g,
            alpha_global=alpha_gl,
            alpha_local=alpha_lo,
        )

    def is_fitted(self) -> bool:
        """Return True if fit() has been called."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"PricingDriftMonitor(distribution='{self.distribution}', "
            f"n_bootstrap={self.n_bootstrap}, alpha_gini={self.alpha_gini}, "
            f"alpha_global={self.alpha_global}, alpha_local={self.alpha_local}, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fit_balance_correction_glm(
    y: np.ndarray,
    mu_hat: np.ndarray,
    exposure: np.ndarray,
    distribution: str,
) -> np.ndarray:
    """Fit the balance-correction GLM and return mu_bc.

    Fits mu_bc = h^{-1}(beta_0 + beta_1 * h(mu_hat)) by MLE, where h is the
    canonical link of the EDF family. For Poisson this is log(mu_hat); for
    Gamma the reciprocal link 1/mu_hat; for Normal the identity.

    Parameters
    ----------
    y : np.ndarray
        Observed rates.
    mu_hat : np.ndarray
        Model predictions.
    exposure : np.ndarray
        Exposure weights.
    distribution : str
        EDF family: 'poisson', 'gamma', 'normal', 'tweedie'.

    Returns
    -------
    np.ndarray
        Balance-corrected predictions mu_bc.
    """
    if distribution in ("poisson", "tweedie"):
        # Log-link: log(mu_bc) = b0 + b1 * log(mu_hat)
        log_mu = np.log(np.maximum(mu_hat, 1e-15))
        lam_base = mu_hat * exposure

        def neg_ll(params: np.ndarray) -> float:
            b0, b1 = params
            log_mu_bc = b0 + b1 * log_mu
            lam = np.exp(log_mu_bc) * exposure
            lam = np.maximum(lam, 1e-15)
            # Poisson negative log-likelihood (ignoring log-factorial constants)
            return float(-np.sum(y * exposure * log_mu_bc - lam))

        # Initialise at multiplicative correction (b1=1, b0=log(A/E))
        ae = float(np.sum(exposure * y) / np.sum(lam_base)) if np.sum(lam_base) > 0 else 1.0
        x0 = np.array([np.log(max(ae, 1e-10)), 1.0])
        res = scipy.optimize.minimize(
            neg_ll, x0, method="L-BFGS-B",
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
        )
        b0, b1 = res.x
        return np.exp(b0 + b1 * log_mu)

    elif distribution == "gamma":
        # Reciprocal link: 1/mu_bc = b0 + b1 * (1/mu_hat)
        inv_mu = 1.0 / np.maximum(mu_hat, 1e-15)

        def neg_ll_gamma(params: np.ndarray) -> float:
            b0, b1 = params
            inv_mu_bc = b0 + b1 * inv_mu
            # Keep mu_bc positive
            if np.any(inv_mu_bc <= 0):
                return 1e15
            mu_bc = 1.0 / inv_mu_bc
            # Gamma log-likelihood proportional to: sum w * (-log(mu_bc) - y/mu_bc)
            return float(-np.sum(exposure * (-np.log(mu_bc + 1e-15) - y / mu_bc)))

        ae = float(np.sum(exposure * y) / np.sum(exposure * mu_hat))
        x0 = np.array([1.0 / (ae * float(np.mean(mu_hat))), 1.0])
        res = scipy.optimize.minimize(
            neg_ll_gamma, x0, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
        )
        b0, b1 = res.x
        inv_mu_bc = b0 + b1 * inv_mu
        return 1.0 / np.maximum(inv_mu_bc, 1e-15)

    else:  # normal / identity link
        # Identity link: mu_bc = b0 + b1 * mu_hat
        def neg_ll_normal(params: np.ndarray) -> float:
            b0, b1 = params
            mu_bc = b0 + b1 * mu_hat
            return float(np.sum(exposure * (y - mu_bc) ** 2))

        ae = float(np.sum(exposure * y) / np.sum(exposure * mu_hat))
        x0 = np.array([0.0, ae])
        res = scipy.optimize.minimize(
            neg_ll_normal, x0, method="L-BFGS-B",
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
        )
        b0, b1 = res.x
        return b0 + b1 * mu_hat


def _compute_gmcb_lmcb(
    y: np.ndarray,
    mu_hat: np.ndarray,
    exposure: np.ndarray,
    distribution: str,
    tweedie_power: float,
) -> tuple[float, float]:
    """Compute GMCB and LMCB components directly.

    GMCB = S(y, mu_hat) - S(y, mu_bc)
    LMCB = S(y, mu_bc) - S(y, mu_rc)

    where mu_bc is the balance-correction GLM prediction and mu_rc is the
    isotonically recalibrated prediction of mu_bc.

    Parameters
    ----------
    y, mu_hat, exposure : np.ndarray
        Observations, predictions, and exposures.
    distribution : str
        EDF family.
    tweedie_power : float
        Tweedie power (only used for 'tweedie').

    Returns
    -------
    tuple[float, float]
        (gmcb, lmcb), both >= 0.
    """
    mu_hat_safe = np.maximum(mu_hat, 1e-10)

    # Balance-corrected predictions
    mu_bc = _fit_balance_correction_glm(y, mu_hat_safe, exposure, distribution)
    mu_bc = np.maximum(mu_bc, 1e-10)

    # Isotonic recalibration of mu_bc
    mu_bc_rc = isotonic_recalibrate(y, mu_bc, exposure)
    mu_bc_rc = np.maximum(mu_bc_rc, 1e-10)

    d_yhat = deviance(y, mu_hat_safe, exposure, distribution, tweedie_power)
    d_bc = deviance(y, mu_bc, exposure, distribution, tweedie_power)
    d_bc_rc = deviance(y, mu_bc_rc, exposure, distribution, tweedie_power)

    gmcb = max(d_yhat - d_bc, 0.0)
    lmcb = max(d_bc - d_bc_rc, 0.0)
    return gmcb, lmcb


def _mcb_component_bootstrap_pvalue(
    y: np.ndarray,
    mu_hat: np.ndarray,
    exposure: np.ndarray,
    component: str,
    observed_value: float,
    n_bootstrap: int,
    distribution: str,
    tweedie_power: float,
    rng: np.random.Generator,
) -> float:
    """Bootstrap p-value for H0: GMCB=0 or H0: LMCB=0.

    Following Algorithm 1 of arXiv:2510.04556. Under H0, simulate y_b from
    the assumed distribution with mean mu_hat, then compute the MCB component.
    The p-value is the fraction of bootstrap values >= observed.

    For Poisson: Y_b ~ Poisson(mu_hat * exposure), then rate = Y_b / exposure.
    For Gamma: Y_b ~ Gamma(shape=1/phi, scale=mu_hat*phi).
    For Normal: Y_b ~ Normal(mu_hat, sigma).

    Parameters
    ----------
    y : np.ndarray
        Observed rates.
    mu_hat : np.ndarray
        Model predictions.
    exposure : np.ndarray
        Exposure weights.
    component : str
        'GMCB' or 'LMCB'.
    observed_value : float
        Observed value of the component.
    n_bootstrap : int
        Number of bootstrap replicates.
    distribution : str
        EDF family.
    tweedie_power : float
        Tweedie power.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    float
        Bootstrap p-value in [0, 1].
    """
    mu_hat_safe = np.maximum(mu_hat, 1e-10)
    boot_values = np.empty(n_bootstrap, dtype=np.float64)

    if distribution in ("poisson", "tweedie"):
        lam = mu_hat_safe * exposure
        for b in range(n_bootstrap):
            counts_b = rng.poisson(lam)
            y_b = np.where(exposure > 0, counts_b / exposure, 0.0)
            y_b = np.maximum(y_b, 0.0)
            gmcb_b, lmcb_b = _compute_gmcb_lmcb(
                y_b, mu_hat_safe, exposure, distribution, tweedie_power
            )
            boot_values[b] = gmcb_b if component == "GMCB" else lmcb_b

    elif distribution == "gamma":
        # Estimate Gamma dispersion by method-of-moments
        pearson = (y - mu_hat_safe) / mu_hat_safe
        phi = float(np.average(pearson ** 2, weights=exposure))
        phi = max(phi, 1e-6)
        shape = 1.0 / phi
        for b in range(n_bootstrap):
            y_b = rng.gamma(shape=shape, scale=mu_hat_safe * phi)
            y_b = np.maximum(y_b, 1e-10)
            gmcb_b, lmcb_b = _compute_gmcb_lmcb(
                y_b, mu_hat_safe, exposure, distribution, tweedie_power
            )
            boot_values[b] = gmcb_b if component == "GMCB" else lmcb_b

    elif distribution == "normal":
        residuals = y - mu_hat_safe
        sigma = float(np.sqrt(np.average(residuals ** 2, weights=exposure)))
        sigma = max(sigma, 1e-10)
        for b in range(n_bootstrap):
            y_b = rng.normal(loc=mu_hat_safe, scale=sigma)
            gmcb_b, lmcb_b = _compute_gmcb_lmcb(
                y_b, mu_hat_safe, exposure, distribution, tweedie_power
            )
            boot_values[b] = gmcb_b if component == "GMCB" else lmcb_b

    else:
        raise ValueError(f"Unknown distribution '{distribution}'.")

    p_value = float(np.mean(boot_values >= observed_value))
    return p_value


def _interpret_verdict(
    verdict: str,
    g: GiniBootstrapResult,
    m: _MurphyComponents,
    gc: CalibTestResult,
    lc: CalibTestResult,
) -> str:
    """Return a plain-text interpretation of the combined verdict."""
    if verdict == "OK":
        return (
            "Verdict: OK. The model's ranking ability and calibration are "
            "consistent with the reference period. No action required."
        )
    elif verdict == "RECALIBRATE":
        return (
            f"Verdict: RECALIBRATE. Rank structure is stable (Gini p={g.p_value:.4f}) "
            f"and local cohort calibration passes (LMCB p={lc.p_value:.4f}), but a "
            f"significant global level shift is detected (GMCB={gc.statistic:.6f}, "
            f"p={gc.p_value:.4f}). Apply a balance correction to the deployed model. "
            "No model refit required."
        )
    else:  # REFIT
        reasons = []
        if g.reject_h0:
            reasons.append(
                f"Gini ranking has degraded (G_mon={g.gini_mon:.4f} "
                f"vs G_ref={g.gini_ref:.4f}, p={g.p_value:.4f})"
            )
        if lc.reject_h0:
            reasons.append(
                f"local cohort-level miscalibration detected "
                f"(LMCB={lc.statistic:.6f}, p={lc.p_value:.4f})"
            )
        return (
            "Verdict: REFIT. " + "; ".join(reasons) + ". "
            "The model requires a full refit; balance correction alone is insufficient."
        )
