"""
Gini drift tests and Murphy decomposition via fit/test class interface.

This module provides three class-based monitors that implement the Brauer,
Menzel & Wüthrich (arXiv:2510.04556) model monitoring framework:

- ``GiniDriftMonitor``: two-sample Gini drift test with a scikit-learn-style
  ``fit`` / ``test`` interface. Stores the reference period at fit time; the
  monitoring data is passed at test time. Use this when raw reference data is
  available.

- ``GiniBootstrapMonitor``: one-sample variant that stores only a scalar
  training Gini at fit time — no raw reference data needed. The practical
  choice for deployed monitoring where the reference data cannot be shipped
  with the model.

- ``MurphyDecomposition``: class-based Murphy score decomposition with a
  ``decompose`` / ``summary`` interface. Wraps the functional
  :func:`insurance_monitoring.calibration.murphy_decomposition` with a
  stateful API, caching results and adding a decision-rule interpretation.

Design rationale
----------------
The functional API (``gini_drift_test``, ``murphy_decomposition``) is well
suited to one-shot pipeline scripts. The class API here is better for:

- Periodic monitoring loops where the reference is stored once and
  monitoring data arrives month-by-month.
- Governance reporting where the object retains provenance (fit parameters,
  input shapes, n_bootstrap used).
- Integration with monitoring dashboards that call ``.test()`` idempotently.

All classes are exposure-weighted — UK motor and home books always have
varying policy durations and the Gini and Murphy statistics are materially
affected by unequal exposure.

References
----------
- Brauer, Menzel & Wüthrich (2025), arXiv:2510.04556
- Wüthrich (2024), *Actuarial Data Science*

Examples
--------
Two-sample Gini drift test::

    import numpy as np
    from insurance_monitoring.gini_monitoring import GiniDriftMonitor

    rng = np.random.default_rng(0)
    y_ref = rng.poisson(rng.uniform(0.05, 0.20, 5_000)).astype(float)
    mu_ref = rng.uniform(0.05, 0.20, 5_000)

    monitor = GiniDriftMonitor(n_bootstrap=300, alpha=0.05, random_state=42)
    monitor.fit(y_ref, mu_ref)

    y_mon = rng.poisson(rng.uniform(0.05, 0.20, 3_000)).astype(float)
    mu_mon = rng.uniform(0.05, 0.20, 3_000)
    result = monitor.test(y_mon, mu_mon)
    print(result.p_value)

One-sample bootstrap test::

    from insurance_monitoring.gini_monitoring import GiniBootstrapMonitor

    boot_monitor = GiniBootstrapMonitor(n_bootstrap=500, alpha=0.05)
    boot_monitor.fit(gini_ref=0.48)
    result = boot_monitor.test(y_mon, mu_mon)

Murphy decomposition::

    from insurance_monitoring.gini_monitoring import MurphyDecomposition

    murphy = MurphyDecomposition(family="poisson")
    result = murphy.decompose(y, mu_hat, exposure)
    print(murphy.summary())
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
from insurance_monitoring.calibration._murphy import murphy_decomposition
from insurance_monitoring.calibration._types import MurphyResult

# ---------------------------------------------------------------------------
# Result dataclass for GiniDriftMonitor
# ---------------------------------------------------------------------------


@dataclass
class GiniDriftResult:
    """Result of a :class:`GiniDriftMonitor` two-sample z-test.

    Attributes
    ----------
    gini_ref : float
        Gini coefficient computed on the reference period data.
    gini_mon : float
        Gini coefficient computed on the monitoring period data.
    gini_change : float
        gini_mon - gini_ref. Negative indicates degraded ranking power.
    z_stat : float
        (gini_mon - gini_ref) / sqrt(Var_ref + Var_mon).
    p_value : float
        Two-sided p-value under N(0, 1) asymptotic null.
    reject_h0 : bool
        True if p_value < alpha.
    ci_lower : float
        Lower bound of the (1 - alpha) confidence interval for gini_change.
    ci_upper : float
        Upper bound of the (1 - alpha) confidence interval for gini_change.
    se_ref : float
        Bootstrap standard error for the reference Gini estimator.
    se_mon : float
        Bootstrap standard error for the monitor Gini estimator.
    n_ref : int
        Number of reference observations.
    n_mon : int
        Number of monitoring observations.
    alpha : float
        Significance level used for reject_h0.
    n_bootstrap : int
        Bootstrap replicates used for variance estimation.
    """

    gini_ref: float
    gini_mon: float
    gini_change: float
    z_stat: float
    p_value: float
    reject_h0: bool
    ci_lower: float
    ci_upper: float
    se_ref: float
    se_mon: float
    n_ref: int
    n_mon: int
    alpha: float
    n_bootstrap: int

    def to_dict(self) -> dict:
        """Return all fields as a plain dict (JSON-serialisable)."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ---------------------------------------------------------------------------
# Result dataclass for GiniBootstrapMonitor
# ---------------------------------------------------------------------------


@dataclass
class GiniBootstrapResult:
    """Result of a :class:`GiniBootstrapMonitor` one-sample bootstrap test.

    Attributes
    ----------
    gini_ref : float
        Stored scalar reference Gini (from fit time).
    gini_mon : float
        Point estimate of Gini on monitoring data.
    gini_change : float
        gini_mon - gini_ref. Negative indicates degradation.
    z_stat : float
        (gini_mon - gini_ref) / se_bootstrap.
    p_value : float
        Two-sided p-value under N(0, 1) approximation.
    reject_h0 : bool
        True if p_value < alpha.
    se_bootstrap : float
        Bootstrap standard error of the monitor Gini estimator.
    n_mon : int
        Number of monitoring observations.
    alpha : float
        Significance level used.
    n_bootstrap : int
        Bootstrap replicates used.
    """

    gini_ref: float
    gini_mon: float
    gini_change: float
    z_stat: float
    p_value: float
    reject_h0: bool
    se_bootstrap: float
    n_mon: int
    alpha: float
    n_bootstrap: int

    def to_dict(self) -> dict:
        """Return all fields as a plain dict (JSON-serialisable)."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ---------------------------------------------------------------------------
# GiniDriftMonitor — two-sample fit / test interface
# ---------------------------------------------------------------------------

class GiniDriftMonitor:
    """Two-sample Gini drift monitor with fit / test interface.

    Stores the reference period at ``fit`` time; monitoring data is passed at
    ``test`` time. Suitable for periodic monitoring loops where reference data
    is fixed once at model sign-off and new monitoring data arrives each month.

    The test statistic is the z-score::

        z = (G_mon - G_ref) / sqrt(Var_boot(G_ref) + Var_boot(G_mon))

    Under H0 (no change), z ~ N(0, 1) asymptotically. The variance of each
    Gini estimator is estimated by bootstrap (Algorithm 2 of arXiv:2510.04556).
    For n > 20,000 the bootstrap is computed on a random subsample of 20,000
    observations — the Gini point estimates still use the full sample.

    Parameters
    ----------
    n_bootstrap : int, default 500
        Number of bootstrap replicates for variance estimation. Must be >= 50.
    alpha : float, default 0.05
        Significance level for the ``reject_h0`` flag. Use 0.32 for the
        one-sigma early-warning rule recommended by arXiv:2510.04556.
    random_state : int or None, default None
        Seed for the bootstrap RNG. None gives non-reproducible results.

    Examples
    --------
    ::

        monitor = GiniDriftMonitor(n_bootstrap=300, alpha=0.05, random_state=0)
        monitor.fit(y_ref, mu_ref, exposure_ref)
        result = monitor.test(y_mon, mu_mon, exposure_mon)
        print(f"p={result.p_value:.4f}, reject={result.reject_h0}")
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        if n_bootstrap < 50:
            raise ValueError(f"n_bootstrap must be >= 50, got {n_bootstrap}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

        # Set at fit time
        self._y_ref: Optional[np.ndarray] = None
        self._mu_ref: Optional[np.ndarray] = None
        self._exp_ref: Optional[np.ndarray] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        y_ref: ArrayLike,
        mu_hat_ref: ArrayLike,
        exposure_ref: Optional[ArrayLike] = None,
    ) -> "GiniDriftMonitor":
        """Store reference period data.

        Parameters
        ----------
        y_ref : array-like
            Observed claims or loss rates in the reference period.
        mu_hat_ref : array-like
            Model predictions for the reference period.
        exposure_ref : array-like, optional
            Exposure weights (e.g. earned car-years) for the reference period.

        Returns
        -------
        GiniDriftMonitor
            self (for method chaining).

        Raises
        ------
        ValueError
            If arrays are empty, have mismatched lengths, or exposures contain
            non-positive values.
        """
        y = _to_numpy(y_ref)
        mu = _to_numpy(mu_hat_ref)
        exp = _to_numpy_optional(exposure_ref)
        _validate_pair(y, mu, exp, "reference")
        self._y_ref = y
        self._mu_ref = mu
        self._exp_ref = exp
        self._fitted = True
        return self

    def test(
        self,
        y_mon: ArrayLike,
        mu_hat_mon: ArrayLike,
        exposure_mon: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
        n_bootstrap: Optional[int] = None,
    ) -> GiniDriftResult:
        """Run the two-sample Gini drift test against the stored reference.

        Parameters
        ----------
        y_mon : array-like
            Observed claims in the monitoring period.
        mu_hat_mon : array-like
            Model predictions for the monitoring period.
        exposure_mon : array-like, optional
            Exposure weights for the monitoring period.
        alpha : float, optional
            Override significance level for this call. If None, uses the
            alpha set at construction time.
        n_bootstrap : int, optional
            Override bootstrap replicates for this call. If None, uses
            n_bootstrap from construction.

        Returns
        -------
        GiniDriftResult

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        ValueError
            If monitor arrays are empty, have mismatched lengths, or
            exposures contain non-positive values.
        UserWarning
            If either sample has fewer than 200 observations.
        """
        if not self._fitted:
            raise RuntimeError(
                "GiniDriftMonitor must be fitted before calling test(). "
                "Call fit(y_ref, mu_hat_ref) first."
            )

        alpha_ = alpha if alpha is not None else self.alpha
        n_boot = n_bootstrap if n_bootstrap is not None else self.n_bootstrap
        if not (0.0 < alpha_ < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha_}")
        if n_boot < 50:
            raise ValueError(f"n_bootstrap must be >= 50, got {n_boot}")

        act_mon = _to_numpy(y_mon)
        pred_mon = _to_numpy(mu_hat_mon)
        exp_mon = _to_numpy_optional(exposure_mon)
        _validate_pair(act_mon, pred_mon, exp_mon, "monitor")

        act_ref = self._y_ref
        pred_ref = self._mu_ref
        exp_ref = self._exp_ref

        n_ref = len(act_ref)
        n_mon = len(act_mon)

        _warn_small(n_ref, "reference")
        _warn_small(n_mon, "monitor")

        # Gini point estimates use full samples
        g_ref = float(_gini_from_arrays(act_ref, pred_ref, exp_ref))
        g_mon = float(_gini_from_arrays(act_mon, pred_mon, exp_mon))

        # Bootstrap variance (subsampling for n > 20,000 is handled internally
        # by _bootstrap_gini_samples).
        seed_ref, seed_mon = _derive_seeds(self.random_state)

        boot_ref = _bootstrap_gini_samples(
            act_ref, pred_ref, exp_ref, n_boot, seed=seed_ref
        )
        boot_mon = _bootstrap_gini_samples(
            act_mon, pred_mon, exp_mon, n_boot, seed=seed_mon
        )

        var_ref = float(np.var(boot_ref, ddof=1))
        var_mon = float(np.var(boot_mon, ddof=1))
        se_ref = float(np.sqrt(var_ref))
        se_mon = float(np.sqrt(var_mon))
        se_total = float(np.sqrt(var_ref + var_mon))

        delta = g_mon - g_ref

        if se_total == 0.0:
            z = float("nan")
            p_value = float("nan")
            reject = False
        else:
            z = delta / se_total
            p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            reject = bool(p_value < alpha_)

        # (1 - alpha) CI for delta = g_mon - g_ref
        # Computed from bootstrap deltas, centering on g_ref and g_mon
        # Use normal approximation CI: delta +/- z_{alpha/2} * se_total
        if se_total > 0.0:
            z_crit = float(stats.norm.ppf(1.0 - alpha_ / 2.0))
            ci_lower = delta - z_crit * se_total
            ci_upper = delta + z_crit * se_total
        else:
            ci_lower = delta
            ci_upper = delta

        return GiniDriftResult(
            gini_ref=g_ref,
            gini_mon=g_mon,
            gini_change=delta,
            z_stat=float(z),
            p_value=p_value,
            reject_h0=reject,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se_ref=se_ref,
            se_mon=se_mon,
            n_ref=n_ref,
            n_mon=n_mon,
            alpha=alpha_,
            n_bootstrap=n_boot,
        )

    def is_fitted(self) -> bool:
        """Return True if ``fit`` has been called."""
        return self._fitted


# ---------------------------------------------------------------------------
# GiniBootstrapMonitor — one-sample fit / test interface
# ---------------------------------------------------------------------------


class GiniBootstrapMonitor:
    """One-sample Gini drift monitor: stores only the reference scalar Gini.

    More practical for deployed monitoring than :class:`GiniDriftMonitor`
    because no raw reference data is needed — only the scalar training Gini is
    stored in the model registry at deployment time.

    The test statistic uses only the bootstrap variance of the monitor Gini::

        z = (G_mon - G_ref) / sigma_boot(G_mon)

    The one-sample design treats G_ref as a fixed constant with no estimation
    uncertainty. This is the correct assumption when G_ref was computed on a
    large training set and the monitoring window is the uncertain quantity.

    Parameters
    ----------
    n_bootstrap : int, default 500
        Bootstrap replicates for the monitor variance estimate. Must be >= 50.
    alpha : float, default 0.05
        Significance level for the reject_h0 flag.
    random_state : int or None, default None
        Seed for reproducibility.

    Examples
    --------
    ::

        boot = GiniBootstrapMonitor(n_bootstrap=500, alpha=0.05, random_state=0)
        boot.fit(gini_ref=0.48)
        result = boot.test(y_mon, mu_mon)
        print(f"Gini change: {result.gini_change:+.4f}, p={result.p_value:.4f}")
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        if n_bootstrap < 50:
            raise ValueError(f"n_bootstrap must be >= 50, got {n_bootstrap}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

        self._gini_ref: Optional[float] = None
        self._fitted: bool = False

    def fit(self, gini_ref: float) -> "GiniBootstrapMonitor":
        """Store the scalar reference Gini.

        Parameters
        ----------
        gini_ref : float
            Gini coefficient computed on training / holdout data at model
            deployment time. Must be in (-1, 1).

        Returns
        -------
        GiniBootstrapMonitor
            self.

        Raises
        ------
        ValueError
            If gini_ref is not in (-1, 1).
        """
        gini_ref = float(gini_ref)
        if not (-1.0 < gini_ref < 1.0):
            raise ValueError(
                f"gini_ref must be in (-1, 1), got {gini_ref}"
            )
        self._gini_ref = gini_ref
        self._fitted = True
        return self

    def test(
        self,
        y_mon: ArrayLike,
        mu_hat_mon: ArrayLike,
        exposure_mon: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
        n_bootstrap: Optional[int] = None,
    ) -> GiniBootstrapResult:
        """Run the one-sample Gini drift test against the stored reference Gini.

        Parameters
        ----------
        y_mon : array-like
            Observed claims in the monitoring period.
        mu_hat_mon : array-like
            Model predictions for the monitoring period.
        exposure_mon : array-like, optional
            Exposure weights for the monitoring period.
        alpha : float, optional
            Override significance level for this call.
        n_bootstrap : int, optional
            Override bootstrap replicates for this call.

        Returns
        -------
        GiniBootstrapResult

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        UserWarning
            If n_mon < 200.
        """
        if not self._fitted:
            raise RuntimeError(
                "GiniBootstrapMonitor must be fitted before calling test(). "
                "Call fit(gini_ref) first."
            )

        alpha_ = alpha if alpha is not None else self.alpha
        n_boot = n_bootstrap if n_bootstrap is not None else self.n_bootstrap
        if not (0.0 < alpha_ < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha_}")
        if n_boot < 50:
            raise ValueError(f"n_bootstrap must be >= 50, got {n_boot}")

        act_mon = _to_numpy(y_mon)
        pred_mon = _to_numpy(mu_hat_mon)
        exp_mon = _to_numpy_optional(exposure_mon)
        _validate_pair(act_mon, pred_mon, exp_mon, "monitor")

        n_mon = len(act_mon)
        _warn_small(n_mon, "monitor")

        g_mon = float(_gini_from_arrays(act_mon, pred_mon, exp_mon))
        g_ref = self._gini_ref

        _, seed_mon = _derive_seeds(self.random_state)

        boot_mon = _bootstrap_gini_samples(
            act_mon, pred_mon, exp_mon, n_boot, seed=seed_mon
        )

        se = float(np.std(boot_mon, ddof=1))
        delta = g_mon - g_ref

        if se == 0.0:
            z = float("nan")
            p_value = float("nan")
            reject = False
        else:
            z = delta / se
            p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            reject = bool(p_value < alpha_)

        return GiniBootstrapResult(
            gini_ref=g_ref,
            gini_mon=g_mon,
            gini_change=delta,
            z_stat=float(z),
            p_value=p_value,
            reject_h0=reject,
            se_bootstrap=se,
            n_mon=n_mon,
            alpha=alpha_,
            n_bootstrap=n_boot,
        )

    def is_fitted(self) -> bool:
        """Return True if ``fit`` has been called."""
        return self._fitted


# ---------------------------------------------------------------------------
# MurphyDecomposition — class-based wrapper around murphy_decomposition()
# ---------------------------------------------------------------------------


class MurphyDecomposition:
    """Class-based Murphy score decomposition with fit / decompose interface.

    Wraps the functional :func:`insurance_monitoring.calibration.murphy_decomposition`
    with a stateful class API that:

    - Caches the last decomposition result for repeated inspection.
    - Provides a ``summary()`` method that interprets the components and
      applies the recalibrate-vs-refit decision rule.
    - Accepts ``family`` as a constructor argument (matching GLM convention)
      alongside the ``distribution`` kwarg of the underlying function.

    The Murphy identity::

        D(y, mu_hat) = UNC - DSC + MCB

    where:

    - **UNC** — uncertainty: baseline deviance from the grand mean model.
      Measures data difficulty; independent of the model.
    - **DSC** — discrimination: deviance improvement from correct ranking.
      A model that cannot rank risks at all has DSC = 0.
    - **MCB** — miscalibration: excess deviance from wrong price levels
      (independent of ranking order). Splits further into GMCB (global,
      fixable by a single recalibration factor) and LMCB (local, requires
      model refit).

    Decision rule (``summary()``)::

        If MCB dominates (GMCB > LMCB): recalibrate.
        If DSC drops: refit.
        If both MCB ≈ 0 and DSC > 0: model is OK.

    Parameters
    ----------
    family : str, default 'poisson'
        Loss distribution family. One of ``'poisson'``, ``'gamma'``,
        ``'tweedie'``, ``'normal'``. Maps directly to the ``distribution``
        argument of :func:`murphy_decomposition`.
    tweedie_power : float, default 1.5
        Tweedie variance power. Only used when ``family='tweedie'``.
    dsc_threshold : float, default 0.01
        Minimum DSC/UNC ratio required for a verdict of OK. A calibrated
        grand-mean model (DSC ≈ 0) should not receive an OK verdict.

    Examples
    --------
    ::

        import numpy as np
        from insurance_monitoring.gini_monitoring import MurphyDecomposition

        rng = np.random.default_rng(0)
        y_hat = rng.gamma(2, 0.05, 2000)
        y = rng.poisson(y_hat).astype(float)
        exposure = np.ones(2000)

        murphy = MurphyDecomposition(family="poisson")
        result = murphy.decompose(y, y_hat, exposure)
        print(f"UNC={result.unc:.4f}, DSC={result.dsc:.4f}, MCB={result.mcb:.4f}")
        print(murphy.summary())
    """

    def __init__(
        self,
        family: str = "poisson",
        tweedie_power: float = 1.5,
        dsc_threshold: float = 0.01,
    ) -> None:
        _valid_families = {"poisson", "gamma", "tweedie", "normal"}
        if family not in _valid_families:
            raise ValueError(
                f"family must be one of {sorted(_valid_families)}, got '{family}'"
            )
        self.family = family
        self.tweedie_power = tweedie_power
        self.dsc_threshold = dsc_threshold
        self._result: Optional[_MurphyComponents] = None
        self._raw: Optional[MurphyResult] = None

    def decompose(
        self,
        y: ArrayLike,
        mu_hat: ArrayLike,
        exposure: Optional[ArrayLike] = None,
    ) -> "_MurphyComponents":
        """Compute the Murphy decomposition and cache the result.

        Parameters
        ----------
        y : array-like
            Observed losses or claim rates.
        mu_hat : array-like
            Model predictions.
        exposure : array-like, optional
            Exposure weights. If None, all policies are equally weighted.

        Returns
        -------
        _MurphyComponents
            Dataclass with ``unc``, ``dsc``, ``mcb``, ``gmcb``, ``lmcb``
            and ``verdict`` fields. Calling ``to_dict()`` returns a flat dict
            suitable for logging or writing to a Delta table.

        Raises
        ------
        ValueError
            If arrays are empty, have mismatched lengths, or contain NaN/inf
            values (propagated from the underlying deviance functions).
        """
        raw = murphy_decomposition(
            y=y,
            y_hat=mu_hat,
            exposure=exposure,
            distribution=self.family,
            tweedie_power=self.tweedie_power,
            dsc_threshold=self.dsc_threshold,
        )
        self._raw = raw
        self._result = _MurphyComponents(
            unc=raw.uncertainty,
            dsc=raw.discrimination,
            mcb=raw.miscalibration,
            gmcb=raw.global_mcb,
            lmcb=raw.local_mcb,
            total_deviance=raw.total_deviance,
            dsc_pct=raw.discrimination_pct,
            mcb_pct=raw.miscalibration_pct,
            verdict=raw.verdict,
        )
        return self._result

    def summary(self) -> dict:
        """Return a structured summary of the last decomposition result.

        Returns a flat dict with all Murphy components and a human-readable
        interpretation. Suitable for logging, governance reports, and
        Delta table writes.

        Returns
        -------
        dict
            Keys: ``unc``, ``dsc``, ``mcb``, ``gmcb``, ``lmcb``,
            ``total_deviance``, ``dsc_pct``, ``mcb_pct``, ``verdict``,
            ``interpretation``.

        Raises
        ------
        RuntimeError
            If called before ``decompose``.
        """
        if self._result is None:
            raise RuntimeError(
                "MurphyDecomposition must run decompose() before calling summary(). "
                "Call decompose(y, mu_hat) first."
            )
        r = self._result
        interp = _interpret_murphy(r)
        d = r.to_dict()
        d["interpretation"] = interp
        return d

    @property
    def result(self) -> Optional["_MurphyComponents"]:
        """Cached decomposition result (None until ``decompose`` is called)."""
        return self._result


# ---------------------------------------------------------------------------
# _MurphyComponents — lightweight result dataclass
# ---------------------------------------------------------------------------


@dataclass
class _MurphyComponents:
    """Murphy decomposition components (result of :class:`MurphyDecomposition`).

    Attributes
    ----------
    unc : float
        Uncertainty: D(y, y_bar) — baseline deviance from grand mean.
    dsc : float
        Discrimination: UNC - D(y, mu_hat_rc). How much better than the
        grand mean after isotonic recalibration.
    mcb : float
        Miscalibration: D(y, mu_hat) - D(y, mu_hat_rc). Excess deviance
        due to wrong price levels.
    gmcb : float
        Global MCB: fixable by multiplying all predictions by the balance
        ratio (single global recalibration factor).
    lmcb : float
        Local MCB: residual after balance correction. Requires model refit
        or isotonic recalibration to fix.
    total_deviance : float
        D(y, mu_hat): raw model deviance.
    dsc_pct : float
        DSC as a percentage of total_deviance.
    mcb_pct : float
        MCB as a percentage of total_deviance.
    verdict : str
        ``'OK'``, ``'RECALIBRATE'``, or ``'REFIT'``.
    """

    unc: float
    dsc: float
    mcb: float
    gmcb: float
    lmcb: float
    total_deviance: float
    dsc_pct: float
    mcb_pct: float
    verdict: str

    def to_dict(self) -> dict:
        """Return all fields as a plain dict."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pair(
    y: np.ndarray,
    mu: np.ndarray,
    exp: Optional[np.ndarray],
    label: str,
) -> None:
    """Validate that y, mu (and optionally exp) are consistent."""
    n = len(y)
    if n == 0:
        raise ValueError(f"{label} actual array must be non-empty")
    if len(mu) != n:
        raise ValueError(
            f"{label} actual length ({n}) != {label} predicted length ({len(mu)})"
        )
    if exp is not None:
        if len(exp) != n:
            raise ValueError(
                f"{label} exposure length ({len(exp)}) != {label} actual length ({n})"
            )
        if np.any(exp <= 0):
            raise ValueError(
                f"All {label} exposure values must be positive"
            )


def _warn_small(n: int, label: str) -> None:
    """Issue UserWarning if n is below 200."""
    if n < 200:
        warnings.warn(
            f"{label} sample has only {n} observations. "
            "Bootstrap Gini distribution may not be well-approximated "
            "by Normal; interpret p-values cautiously.",
            UserWarning,
            stacklevel=3,
        )


def _derive_seeds(random_state: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    """Derive two independent seeds from a single random state."""
    if random_state is None:
        return None, None
    rng = np.random.default_rng(random_state)
    return int(rng.integers(0, 2**31)), int(rng.integers(0, 2**31))



def _interpret_murphy(r: "_MurphyComponents") -> str:
    """Return a plain-text interpretation of the Murphy decomposition."""
    if r.verdict == "OK":
        return (
            f"Model is well calibrated and discriminates well. "
            f"DSC={r.dsc:.4f} ({r.dsc_pct:.1f}% of deviance), "
            f"MCB={r.mcb:.4f} ({r.mcb_pct:.1f}%). No action required."
        )
    elif r.verdict == "RECALIBRATE":
        return (
            f"Global miscalibration dominates (GMCB={r.gmcb:.4f} > "
            f"LMCB={r.lmcb:.4f}). Apply a multiplicative recalibration "
            f"factor. No model refit required."
        )
    else:  # REFIT
        return (
            f"Local miscalibration dominates (LMCB={r.lmcb:.4f} >= "
            f"GMCB={r.gmcb:.4f}) or discrimination is insufficient "
            f"(DSC={r.dsc:.4f}). Model refit or isotonic recalibration required."
        )
