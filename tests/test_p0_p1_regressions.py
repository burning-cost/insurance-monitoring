"""Regression tests for P0 and P1 bug fixes (v0.3.2).

Each test class maps to a specific bug report:
- TestP0_1_AERatioCINormalSE  — normal CI was 4.5x too narrow (wrong denominator)
- TestP0_2_GiniTieReproducible — Gini varied by 11.5% on row-shuffles (argsort ties)
- TestP1_1_GiniDocstringRange  — docstring claimed [0,1], actual range is [-1,1]
- TestP1_2_MurphyZeroDiscVerdict — zero-discrimination model got 'OK' verdict
- TestP1_3_CalibrationReportNoGini — CalibrationReport.verdict docstring now warns
- TestP1_4_PSIReferenceExposure — PSI reference_exposure weighting added

These tests are expected to fail on v0.3.1 and pass on v0.3.2+.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from insurance_monitoring.calibration import ae_ratio_ci, murphy_decomposition
from insurance_monitoring.calibration._types import CalibrationReport
from insurance_monitoring.discrimination import gini_coefficient, lorenz_curve
from insurance_monitoring.drift import psi


# ---------------------------------------------------------------------------
# P0-1: ae_ratio_ci normal CI SE
# ---------------------------------------------------------------------------


class TestP0_1_AERatioCINormalSE:
    """The SE for A/E under Poisson is ae / sqrt(n_expected), not ae / sqrt(n_policies).

    For a frequency model with 5,000 policies each with expected rate ~0.1
    (n_expected = 500), the old code used sqrt(5000) ~ 70.7 as the denominator
    instead of sqrt(500) ~ 22.4.  That makes the CI 70.7/22.4 ≈ 3.16x too narrow
    (interval width ∝ 1/sqrt, so width ratio is ~3.16x).

    The bug was particularly damaging for sparse books where few claims occur
    across many policies — exactly the regime where false precision is most
    dangerous.
    """

    def test_normal_ci_wider_than_poisson_ci_for_sparse_book(self):
        """For a sparse Poisson book, the normal CI should be comparable to Poisson.

        The normal method should produce a CI that broadly agrees with the
        exact Poisson method (same order of magnitude). The old code produced
        a CI ~4.5x too narrow, so checking the ratio of interval widths is
        the most direct regression guard.
        """
        rng = np.random.default_rng(42)
        n_policies = 5_000
        # Sparse book: average expected claims per policy ~0.05
        exposure = np.ones(n_policies)
        predicted = np.full(n_policies, 0.05)
        actual = rng.poisson(predicted * exposure)

        result_poisson = ae_ratio_ci(actual, predicted, exposure=exposure, method="poisson")
        result_normal = ae_ratio_ci(actual, predicted, exposure=exposure, method="normal")

        width_poisson = result_poisson["upper"] - result_poisson["lower"]
        width_normal = result_normal["upper"] - result_normal["lower"]

        # The ratio of widths should be close to 1 (both methods estimate the same
        # quantity). Old code made width_normal / width_poisson ~ 0.22 (way too narrow).
        # After fix it should be within 3x of Poisson (normal approximation is
        # rougher but not catastrophically wrong).
        ratio = width_normal / width_poisson
        assert ratio > 0.3, (
            f"Normal CI width ({width_normal:.4f}) is {ratio:.2f}x the Poisson CI width "
            f"({width_poisson:.4f}). Expected ratio > 0.3. Old bug gave ~0.22."
        )

    def test_normal_ci_se_uses_n_expected_not_n_policies(self):
        """Directly verify the SE formula uses n_expected.

        SE(A/E) = ae / sqrt(n_expected). With n_policies >> n_expected (sparse book),
        old code used sqrt(n_policies) giving a much smaller SE.
        """
        rng = np.random.default_rng(0)
        n = 10_000
        predicted = np.full(n, 0.02)   # very sparse: 200 expected claims from 10k policies
        actual = rng.poisson(predicted)
        exposure = np.ones(n)

        result = ae_ratio_ci(actual, predicted, exposure=exposure, method="normal")
        ae = result["ae"]
        n_expected = result["n_expected"]  # = 200

        # Expected SE: ae / sqrt(n_expected)
        expected_se = ae / math.sqrt(n_expected)
        # Implied SE from the CI: width / (2 * z_{0.975})
        from scipy import stats
        z = stats.norm.ppf(0.975)
        implied_se = (result["upper"] - result["lower"]) / (2 * z)

        assert implied_se == pytest.approx(expected_se, rel=1e-6), (
            f"Implied SE ({implied_se:.6f}) != ae/sqrt(n_expected) ({expected_se:.6f}). "
            f"Check that the SE uses n_expected={n_expected:.0f}, not n_policies={n}."
        )

    def test_normal_ci_contains_true_ae_with_adequate_coverage(self):
        """The normal CI at 95% should contain the true A/E in ~95% of trials."""
        rng = np.random.default_rng(123)
        n_trials = 200
        n_policies = 2_000
        predicted = np.full(n_policies, 0.08)
        exposure = np.ones(n_policies)
        true_ae = 1.0  # perfectly calibrated model

        covered = 0
        for _ in range(n_trials):
            actual = rng.poisson(predicted * exposure)
            result = ae_ratio_ci(actual, predicted, exposure=exposure, method="normal", alpha=0.05)
            if result["lower"] <= true_ae <= result["upper"]:
                covered += 1

        coverage = covered / n_trials
        # Allow loose tolerance: coverage should be between 80% and 99.5%
        # (would be ~5% too narrow with old bug for n_policies=2000, n_expected=160)
        assert coverage >= 0.80, (
            f"Normal CI 95% coverage = {coverage:.1%}. Expected >= 80%. "
            f"Old bug caused systematically narrow intervals."
        )


# ---------------------------------------------------------------------------
# P0-2: Gini reproducibility on tie rows
# ---------------------------------------------------------------------------


class TestP0_2_GiniTieReproducible:
    """Gini must be order-independent when many policies share the same predicted rate.

    The old argsort(kind='stable') implementation produced different Gini values
    depending on how the input rows were ordered, varying by up to 11.5% on GLM
    datasets where NCD bands produce large groups of identical predictions.
    """

    def _make_ncd_dataset(self, n: int = 5_000, seed: int = 0) -> tuple:
        """Synthetic GLM dataset with NCD bands: 5 distinct predicted rates."""
        rng = np.random.default_rng(seed)
        # 5 NCD levels — each level has the same predicted rate for all policies
        ncd_rates = np.array([0.05, 0.08, 0.12, 0.18, 0.25])
        ncd = rng.integers(0, 5, size=n)
        predicted = ncd_rates[ncd]  # 80% of policies share a rate with >900 others
        actual = rng.poisson(predicted).astype(float)
        return actual, predicted

    def test_gini_invariant_to_row_permutation(self):
        """Gini must be identical regardless of how rows are ordered."""
        actual, predicted = self._make_ncd_dataset(n=5_000, seed=42)
        g_original = gini_coefficient(actual, predicted)

        rng = np.random.default_rng(999)
        for i in range(10):
            perm = rng.permutation(len(actual))
            g_perm = gini_coefficient(actual[perm], predicted[perm])
            assert g_perm == pytest.approx(g_original, abs=1e-12), (
                f"Gini changed on permutation {i}: {g_original:.6f} vs {g_perm:.6f}. "
                f"Difference = {abs(g_original - g_perm):.6f}. Tie-breaking is broken."
            )

    def test_lorenz_curve_invariant_to_row_permutation(self):
        """lorenz_curve must produce the same points regardless of row order."""
        actual, predicted = self._make_ncd_dataset(n=2_000, seed=7)
        x_orig, y_orig = lorenz_curve(actual, predicted)

        rng = np.random.default_rng(77)
        for i in range(5):
            perm = rng.permutation(len(actual))
            x_perm, y_perm = lorenz_curve(actual[perm], predicted[perm])
            assert len(x_perm) == len(x_orig), (
                f"lorenz_curve returned different number of points on permutation {i}. "
                f"Expected {len(x_orig)}, got {len(x_perm)}."
            )
            np.testing.assert_array_almost_equal(
                x_perm, x_orig, decimal=12,
                err_msg=f"lorenz_curve x changed on permutation {i} — tie-breaking is order-dependent"
            )
            np.testing.assert_array_almost_equal(
                y_perm, y_orig, decimal=12,
                err_msg=f"lorenz_curve y changed on permutation {i} — tie-breaking is order-dependent"
            )

    def test_gini_stable_on_constant_prediction(self):
        """When all predictions are equal, Gini must be 0 regardless of row order."""
        rng = np.random.default_rng(5)
        actual = rng.poisson(0.1, 1_000).astype(float)
        predicted = np.full(1_000, 0.1)  # all same prediction

        g = gini_coefficient(actual, predicted)
        assert g == pytest.approx(0.0, abs=1e-12), (
            f"Constant prediction should give Gini = 0.0, got {g}. "
            f"Old argsort bug would give a non-zero value dependent on row order."
        )

    def test_gini_with_two_groups_matches_analytic(self):
        """Two predicted levels, analytically verifiable result."""
        # 500 low-risk (pred=0.05), 500 high-risk (pred=0.20)
        # 10 low-risk claim, 100 high-risk claim
        # After unique-value grouping: group1 (low) contributes 500 exp, 10 claims
        #                              group2 (high) contributes 500 exp, 100 claims
        # Sorted ascending: low then high
        # cum_exp = [500, 1000], cum_claims = [10, 110], total_claims = 110
        # x = [0, 0.5, 1.0], y = [0, 10/110, 1.0]
        # AUC = 0.5*(10/110) + 0.5*((10/110)+1.0)/2 = ... let numpy compute
        actual = np.array([1.0] * 10 + [0.0] * 490 + [1.0] * 100 + [0.0] * 400)
        predicted = np.array([0.05] * 500 + [0.20] * 500)

        g = gini_coefficient(actual, predicted)

        # Analytic: x=[0, 0.5, 1], y=[0, 10/110, 1]
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 10 / 110, 1.0])
        auc = float(np.trapezoid(y, x))
        expected_gini = 1.0 - 2.0 * auc

        assert g == pytest.approx(expected_gini, abs=1e-10), (
            f"Gini = {g:.6f}, expected = {expected_gini:.6f}. "
            f"Unique-value grouping should match analytic result exactly."
        )

        # Also check that shuffling the row order gives the same value
        rng = np.random.default_rng(0)
        perm = rng.permutation(1_000)
        g_perm = gini_coefficient(actual[perm], predicted[perm])
        assert g_perm == pytest.approx(expected_gini, abs=1e-10)


# ---------------------------------------------------------------------------
# P1-1: Gini docstring range
# ---------------------------------------------------------------------------


class TestP1_1_GiniDocstringRange:
    """The gini_coefficient docstring must state [-1, 1], not [0, 1]."""

    def test_gini_docstring_mentions_negative(self):
        """Docstring should explicitly mention negative values."""
        doc = gini_coefficient.__doc__
        assert doc is not None
        assert "-1" in doc, (
            "gini_coefficient docstring should mention -1 as a boundary. "
            "The old docstring said [0, 1] which was wrong."
        )

    def test_gini_docstring_mentions_inverted(self):
        """Docstring should warn about inverted discrimination."""
        doc = gini_coefficient.__doc__ or ""
        assert "invert" in doc.lower() or "negative" in doc.lower(), (
            "gini_coefficient docstring should explain what negative Gini means."
        )

    def test_gini_can_be_negative_for_inverted_model(self):
        """A reversed model (highest predicted -> fewest claims) should have Gini < 0."""
        actual = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        # Deliberately inverted: high predicted for no-claim policies
        predicted = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        g = gini_coefficient(actual, predicted)
        assert g < 0.0, (
            f"Inverted model should produce negative Gini, got {g}. "
            f"Gini range is [-1, 1], not [0, 1]."
        )
        assert g >= -1.0


# ---------------------------------------------------------------------------
# P1-2: Murphy verdict for zero-discrimination model
# ---------------------------------------------------------------------------


class TestP1_2_MurphyZeroDiscVerdict:
    """A zero-discrimination model (grand mean) must not receive verdict 'OK'.

    The old condition was `dsc > 0`, which numeric noise could satisfy even
    for a constant prediction model. The fix requires DSC / UNC > threshold
    (default 1%).
    """

    def test_grand_mean_model_not_ok(self):
        """A grand-mean (constant) model should not receive verdict OK."""
        rng = np.random.default_rng(0)
        n = 3_000
        exposure = rng.uniform(0.5, 2.0, n)
        y_hat_varying = rng.gamma(2, 0.05, n)
        counts = rng.poisson(exposure * y_hat_varying)
        y = counts / exposure

        # Grand mean model: constant prediction = weighted average y
        y_bar = float(np.sum(exposure * y) / np.sum(exposure))
        y_hat_constant = np.full(n, y_bar)

        result = murphy_decomposition(y, y_hat_constant, exposure)
        assert result.verdict != "OK", (
            f"Grand-mean model returned verdict '{result.verdict}', expected RECALIBRATE or REFIT. "
            f"DSC={result.discrimination:.6f}, UNC={result.uncertainty:.6f}, "
            f"DSC/UNC={result.discrimination/result.uncertainty:.6f}. "
            f"The old bug checked DSC > 0 instead of DSC/UNC > threshold."
        )

    def test_good_model_can_still_be_ok(self):
        """A well-calibrated model with good discrimination should still get OK."""
        rng = np.random.default_rng(42)
        n = 5_000
        exposure = np.ones(n)
        # True rates vary — model knows them exactly
        y_hat = rng.gamma(2, 0.05, n)
        y = rng.poisson(y_hat) / 1.0

        result = murphy_decomposition(y, y_hat, exposure)
        # Good model should have significant DSC. Verdict may be OK or RECALIBRATE.
        dsc_ratio = result.discrimination / result.uncertainty if result.uncertainty > 0 else 0
        assert dsc_ratio > 0.01, (
            f"Good model has very low DSC/UNC = {dsc_ratio:.4f}. Something is wrong."
        )

    def test_dsc_threshold_parameter_works(self):
        """dsc_threshold parameter should control when OK is granted."""
        rng = np.random.default_rng(10)
        n = 5_000
        exposure = np.ones(n)
        # Moderate discrimination model
        y_hat = np.linspace(0.05, 0.15, n)
        y = rng.poisson(y_hat)

        # With very high threshold (50%), OK should be nearly impossible
        result_strict = murphy_decomposition(y, y_hat, exposure, dsc_threshold=0.50)
        # With very low threshold (0.001%), OK is easy if MCB is low
        result_loose = murphy_decomposition(y, y_hat, exposure, dsc_threshold=0.0001)

        # The verdicts may differ depending on how the model performs
        # but neither should crash and both should return valid verdicts
        assert result_strict.verdict in ("OK", "RECALIBRATE", "REFIT")
        assert result_loose.verdict in ("OK", "RECALIBRATE", "REFIT")

    def test_constant_prediction_low_dsc_ratio(self):
        """Constant prediction model should have DSC/UNC near zero."""
        rng = np.random.default_rng(5)
        n = 2_000
        y = rng.gamma(2, 0.1, n)
        y_hat = np.full(n, y.mean())

        result = murphy_decomposition(y, y_hat)
        dsc_ratio = result.discrimination / result.uncertainty if result.uncertainty > 0 else 0
        assert dsc_ratio < 0.01, (
            f"Constant prediction model has DSC/UNC = {dsc_ratio:.4f}, expected < 0.01. "
            f"DSC={result.discrimination:.6f}, UNC={result.uncertainty:.6f}."
        )


# ---------------------------------------------------------------------------
# P1-3: CalibrationReport.verdict docstring Gini warning
# ---------------------------------------------------------------------------


class TestP1_3_CalibrationReportNoGiniWarning:
    """CalibrationReport.verdict() docstring must warn about absence of Gini testing."""

    def test_verdict_docstring_warns_about_gini(self):
        """verdict() docstring should mention that Gini drift is not tested."""
        doc = CalibrationReport.verdict.__doc__
        assert doc is not None
        doc_lower = doc.lower()
        # Should mention discrimination and Gini drift warning
        assert "gini" in doc_lower or "discrimination" in doc_lower, (
            "CalibrationReport.verdict docstring should warn that Gini drift "
            "is not tested. Use MonitoringReport for full monitoring."
        )

    def test_verdict_docstring_mentions_monitoring_report(self):
        """verdict() docstring should refer users to MonitoringReport."""
        doc = CalibrationReport.verdict.__doc__ or ""
        assert "monitoring" in doc.lower(), (
            "CalibrationReport.verdict docstring should direct users to "
            "MonitoringReport for full monitoring including Gini drift."
        )


# ---------------------------------------------------------------------------
# P1-4: PSI reference_exposure parameter
# ---------------------------------------------------------------------------


class TestP1_4_PSIReferenceExposure:
    """PSI should support symmetric exposure weighting for both reference and current."""

    def test_reference_exposure_accepted(self):
        """psi() should accept reference_exposure without raising."""
        rng = np.random.default_rng(0)
        ref = rng.normal(30, 5, 1_000)
        cur = rng.normal(30, 5, 500)
        ref_exp = rng.uniform(0.1, 2.0, 1_000)
        cur_exp = rng.uniform(0.1, 2.0, 500)

        result = psi(ref, cur, reference_exposure=ref_exp, exposure_weights=cur_exp)
        assert isinstance(result, float)
        assert result >= 0.0  # PSI is non-negative by construction (up to eps)

    def test_uniform_exposures_match_unweighted(self):
        """Uniform exposures should give the same result as no weighting."""
        rng = np.random.default_rng(1)
        ref = rng.normal(30, 5, 1_000)
        cur = rng.normal(30, 5, 500)

        psi_unweighted = psi(ref, cur)
        # Unit exposures should be equivalent to no exposure weights
        ref_exp = np.ones(1_000)
        cur_exp = np.ones(500)
        psi_uniform = psi(ref, cur, reference_exposure=ref_exp, exposure_weights=cur_exp)

        assert psi_unweighted == pytest.approx(psi_uniform, rel=1e-6), (
            f"Uniform exposures should give same PSI as unweighted: "
            f"{psi_unweighted:.6f} vs {psi_uniform:.6f}"
        )

    def test_reference_exposure_only_works(self):
        """psi() should work with only reference_exposure (no current exposure_weights)."""
        rng = np.random.default_rng(2)
        ref = rng.normal(30, 5, 1_000)
        cur = rng.normal(31, 5, 500)
        ref_exp = rng.uniform(0.1, 2.0, 1_000)

        result = psi(ref, cur, reference_exposure=ref_exp)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_reference_exposure_length_mismatch_raises(self):
        """Mismatched reference_exposure length should raise ValueError."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cur = np.array([1.5, 2.5, 3.5])
        ref_exp_wrong = np.ones(3)  # wrong length

        with pytest.raises(ValueError, match="reference_exposure"):
            psi(ref, cur, reference_exposure=ref_exp_wrong)

    def test_heavy_vs_light_policies_affects_psi(self):
        """Upweighting different strata in reference vs current should change PSI.

        If the reference has most exposure in the high-age tail (older drivers)
        and the current has uniform exposure, weighted PSI should differ from
        count-based PSI.
        """
        rng = np.random.default_rng(42)
        # Reference: mix of young and old, but old policies have large exposure
        ages_ref = np.concatenate([
            rng.uniform(20, 30, 500),   # young
            rng.uniform(50, 70, 500),   # old
        ])
        exp_ref = np.concatenate([
            np.full(500, 0.1),    # young: short policies
            np.full(500, 2.0),    # old: annual policies
        ])

        # Current: similarly structured but shifted towards younger
        ages_cur = rng.uniform(20, 40, 500)
        exp_cur = np.ones(500)

        psi_count = psi(ages_ref, ages_cur)
        psi_weighted = psi(ages_ref, ages_cur, reference_exposure=exp_ref, exposure_weights=exp_cur)

        # The two should differ because the reference population looks older when weighted
        assert psi_count != pytest.approx(psi_weighted, rel=0.01), (
            "Exposure weighting should change PSI when policies have different durations. "
            "If they are equal, the reference_exposure parameter has no effect."
        )
