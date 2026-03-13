"""CalibrationChecker: pipeline class tying together all diagnostics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._balance import check_balance
from ._autocal import check_auto_calibration
from ._murphy import murphy_decomposition
from ._types import CalibrationReport
from ._utils import validate_inputs


class CalibrationChecker:
    """End-to-end calibration diagnostic pipeline.

    Combines balance property test, auto-calibration test, and Murphy
    decomposition into a single workflow. Designed for use in model
    sign-off processes and ongoing monitoring pipelines.

    Usage follows a fit/check pattern to support the monitoring use case
    where reference diagnostics from holdout data are retained for later
    comparison against new period data. For a single-shot validation,
    ``fit`` and ``check`` can be called with the same data.

    Parameters
    ----------
    distribution
        Loss distribution: 'poisson', 'gamma', 'tweedie', 'normal'.
    alpha
        Significance level for all tests. Default 0.05.
        For ongoing monitoring, Brauer et al. (2025) recommend 0.32 to
        improve detection power for one-SD calibration deteriorations.
    n_bins
        Number of prediction quantile bins for auto-calibration test.
    bootstrap_n
        Bootstrap replicates for balance CI and MCB test.
    tweedie_power
        Tweedie variance power (only relevant when distribution='tweedie').
    autocal_method
        Method for auto-calibration test: 'bootstrap' or 'hosmer_lemeshow'.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> exposure = rng.uniform(0.5, 2.0, n)
    >>> y_hat = rng.gamma(2, 0.05, n)
    >>> y = rng.poisson(exposure * y_hat).astype(float) / exposure
    >>> checker = CalibrationChecker(distribution='poisson', alpha=0.05)
    >>> checker.fit(y, y_hat, exposure)
    >>> report = checker.check(y, y_hat, exposure)
    >>> isinstance(report.verdict(), str)
    True
    """

    def __init__(
        self,
        distribution: str = "poisson",
        alpha: float = 0.05,
        n_bins: int = 10,
        bootstrap_n: int = 999,
        tweedie_power: float = 1.5,
        autocal_method: str = "bootstrap",
    ) -> None:
        self.distribution = distribution
        self.alpha = alpha
        self.n_bins = n_bins
        self.bootstrap_n = bootstrap_n
        self.tweedie_power = tweedie_power
        self.autocal_method = autocal_method
        self._is_fitted = False
        self._reference_report: CalibrationReport | None = None

    def fit(
        self,
        y: npt.ArrayLike,
        y_hat: npt.ArrayLike,
        exposure: npt.ArrayLike | None = None,
        seed: int | None = None,
    ) -> "CalibrationChecker":
        """Compute reference diagnostics from holdout data.

        These are stored for comparison when :meth:`check` is called on
        a new period. For single-shot use, the output of :meth:`check`
        is sufficient.

        Parameters
        ----------
        y
            Observed loss rates (holdout / reference period).
        y_hat
            Model predictions.
        exposure
            Policy durations.
        seed
            Random seed for bootstrap.

        Returns
        -------
        CalibrationChecker
            Self, for method chaining.
        """
        self._reference_report = self.check(y, y_hat, exposure, seed=seed)
        self._is_fitted = True
        return self

    def check(
        self,
        y: npt.ArrayLike,
        y_hat: npt.ArrayLike,
        exposure: npt.ArrayLike | None = None,
        seed: int | None = None,
    ) -> CalibrationReport:
        """Run all calibration diagnostics and return a combined report.

        Parameters
        ----------
        y
            Observed loss rates.
        y_hat
            Model predictions.
        exposure
            Policy durations. If None, assumed uniform.
        seed
            Random seed for bootstrap methods.

        Returns
        -------
        CalibrationReport
            Combined report with balance, auto-calibration, and Murphy results,
            plus a ``verdict()`` method summarising the recommended action.
        """
        y_arr, y_hat_arr, w = validate_inputs(y, y_hat, exposure)

        balance_result = check_balance(
            y_arr, y_hat_arr, w,
            distribution=self.distribution,
            bootstrap_n=self.bootstrap_n,
            confidence_level=1.0 - self.alpha,
            seed=seed,
        )

        autocal_result = check_auto_calibration(
            y_arr, y_hat_arr, w,
            distribution=self.distribution,
            n_bins=self.n_bins,
            method=self.autocal_method,
            bootstrap_n=self.bootstrap_n,
            significance_level=self.alpha,
            seed=seed,
            tweedie_power=self.tweedie_power,
        )

        murphy_result = murphy_decomposition(
            y_arr, y_hat_arr, w,
            distribution=self.distribution,
            tweedie_power=self.tweedie_power,
            seed=seed,
        )

        return CalibrationReport(
            balance=balance_result,
            auto_calibration=autocal_result,
            murphy=murphy_result,
            distribution=self.distribution,
            n_policies=len(y_arr),
            total_exposure=float(np.sum(w)),
        )

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"CalibrationChecker({self.distribution}, alpha={self.alpha}, "
            f"n_bins={self.n_bins}, {fitted_str})"
        )
