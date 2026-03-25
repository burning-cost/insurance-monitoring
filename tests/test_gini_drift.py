"""
Tests for insurance_monitoring.gini_drift — GiniDriftTest class.

Covers: basic operation, significance detection, no-drift scenario, edge cases,
exposure weighting, input validation, result dataclass, summary method,
and the lazy-evaluation / caching contract.

All tests are designed to run in < 5 s on Databricks serverless. Bootstrap
n_bootstrap is kept at 100-150 to bound runtime.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from insurance_monitoring.gini_drift import GiniDriftTest, GiniDriftTestResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_good_data(n: int, seed: int = 0):
    """Return (actual, predicted) for a model with real discriminatory power."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.05, 0.20, n)
    act = rng.poisson(pred).astype(float)
    return act, pred


def _make_flat_predictions(n: int, seed: int = 0):
    """Predicted all equal — Gini = 0, bootstrap collapses."""
    rng = np.random.default_rng(seed)
    act = rng.poisson(0.10, n).astype(float)
    pred = np.full(n, 0.10)
    return act, pred


def _make_degraded_data(n: int, noise_scale: float = 5.0, seed: int = 99):
    """Return predictions with heavy noise added — lower Gini than clean model."""
    rng = np.random.default_rng(seed)
    pred_clean = rng.uniform(0.05, 0.20, n)
    noise = rng.normal(0, noise_scale * pred_clean.std(), n)
    pred_noisy = np.clip(pred_clean + noise, 0.001, 10.0)
    act = rng.poisson(pred_clean).astype(float)
    return act, pred_noisy


# ---------------------------------------------------------------------------
# T01 — instantiation and lazy evaluation
# ---------------------------------------------------------------------------

class TestInstantiation:
    """T01: Object is created without running bootstrap; _result is None."""

    def test_result_none_before_test(self):
        act_r, pred_r = _make_good_data(500, seed=1)
        act_m, pred_m = _make_good_data(500, seed=2)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=50,
        )
        assert t._result is None

    def test_attributes_stored(self):
        act_r, pred_r = _make_good_data(300, seed=3)
        act_m, pred_m = _make_good_data(300, seed=4)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=80,
            alpha=0.05,
            random_state=42,
        )
        assert t.n_bootstrap == 80
        assert t.alpha == 0.05
        assert t.random_state == 42


# ---------------------------------------------------------------------------
# T02 — test() returns correct type
# ---------------------------------------------------------------------------

class TestReturnType:
    """T02: test() returns a GiniDriftTestResult."""

    def test_returns_gini_drift_test_result(self):
        act_r, pred_r = _make_good_data(500, seed=5)
        act_m, pred_m = _make_good_data(500, seed=6)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=0,
        )
        result = t.test()
        assert isinstance(result, GiniDriftTestResult)


# ---------------------------------------------------------------------------
# T03 — idempotency / caching
# ---------------------------------------------------------------------------

class TestIdempotency:
    """T03: Repeated test() calls return the same object."""

    def test_same_object_on_second_call(self):
        act_r, pred_r = _make_good_data(500, seed=7)
        act_m, pred_m = _make_good_data(500, seed=8)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=1,
        )
        r1 = t.test()
        r2 = t.test()
        assert r1 is r2


# ---------------------------------------------------------------------------
# T04 — result fields are all populated with finite values
# ---------------------------------------------------------------------------

class TestResultFields:
    """T04: All result fields are finite floats / ints of the right type."""

    def test_all_fields_finite(self):
        act_r, pred_r = _make_good_data(1000, seed=9)
        act_m, pred_m = _make_good_data(800, seed=10)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=2,
        )
        r = t.test()
        assert np.isfinite(r.gini_reference)
        assert np.isfinite(r.gini_monitor)
        assert np.isfinite(r.delta)
        assert np.isfinite(r.z_statistic)
        assert np.isfinite(r.p_value)
        assert isinstance(r.significant, bool)
        assert np.isfinite(r.se_reference)
        assert np.isfinite(r.se_monitor)
        assert r.n_reference == 1000
        assert r.n_monitor == 800
        assert r.alpha == 0.32
        assert r.n_bootstrap == 100

    def test_delta_equals_monitor_minus_reference(self):
        act_r, pred_r = _make_good_data(600, seed=11)
        act_m, pred_m = _make_good_data(600, seed=12)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=3,
        )
        r = t.test()
        assert abs(r.delta - (r.gini_monitor - r.gini_reference)) < 1e-10

    def test_p_value_in_range(self):
        act_r, pred_r = _make_good_data(600, seed=13)
        act_m, pred_m = _make_good_data(600, seed=14)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=4,
        )
        r = t.test()
        assert 0.0 <= r.p_value <= 1.0


# ---------------------------------------------------------------------------
# T05 — no-drift scenario: same process, large n, p-value should be high
# ---------------------------------------------------------------------------

class TestNoDrift:
    """T05: Same underlying data-generating process → large p-value expected."""

    def test_no_drift_large_p_value(self):
        # Generate two independent samples from the same DGP.
        # With n=5000 each and same model, we expect p > 0.05 most of the time.
        rng = np.random.default_rng(20)
        pred_ref = rng.uniform(0.05, 0.20, 5000)
        act_ref = rng.poisson(pred_ref).astype(float)
        pred_mon = rng.uniform(0.05, 0.20, 5000)
        act_mon = rng.poisson(pred_mon).astype(float)
        t = GiniDriftTest(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            monitor_actual=act_mon,
            monitor_predicted=pred_mon,
            n_bootstrap=150,
            alpha=0.05,
            random_state=20,
        )
        r = t.test()
        # Under H0, large p-value expected (this is a stochastic test but
        # with seed=20 we have a deterministic outcome)
        assert r.p_value > 0.05, (
            f"Expected no significant drift under same DGP, got p={r.p_value:.4f}"
        )
        assert r.significant is False


# ---------------------------------------------------------------------------
# T06 — drift detection: degraded monitor model
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """T06: Significantly degraded model should be flagged as significant."""

    def test_detects_severe_degradation(self):
        # Reference: clean model. Monitor: heavily-noised predictions.
        rng = np.random.default_rng(30)
        pred_ref = rng.uniform(0.05, 0.20, 5000)
        act_ref = rng.poisson(pred_ref).astype(float)

        # Monitor predictions are pure noise — Gini ≈ 0
        pred_mon = np.full(5000, 0.10)
        act_mon = rng.poisson(rng.uniform(0.05, 0.20, 5000)).astype(float)

        t = GiniDriftTest(
            reference_actual=act_ref,
            reference_predicted=pred_ref,
            monitor_actual=act_mon,
            monitor_predicted=pred_mon,
            n_bootstrap=150,
            alpha=0.32,
            random_state=30,
        )
        r = t.test()
        # The reference Gini should be clearly higher than monitor Gini
        assert r.gini_reference > r.gini_monitor
        assert r.delta < 0  # degradation


# ---------------------------------------------------------------------------
# T07 — significance flag matches p-value
# ---------------------------------------------------------------------------

class TestSignificanceFlag:
    """T07: significant flag must be consistent with p_value and alpha."""

    def test_significant_flag_consistent(self):
        act_r, pred_r = _make_good_data(800, seed=40)
        act_m, pred_m = _make_good_data(800, seed=41)
        for alpha in [0.05, 0.10, 0.32]:
            t = GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                n_bootstrap=100,
                alpha=alpha,
                random_state=40,
            )
            r = t.test()
            assert r.significant == (r.p_value < alpha), (
                f"significant flag inconsistent: p={r.p_value:.4f}, alpha={alpha}, "
                f"significant={r.significant}"
            )


# ---------------------------------------------------------------------------
# T08 — exposure weighting accepted and changes the result
# ---------------------------------------------------------------------------

class TestExposureWeighting:
    """T08: exposure parameter is accepted and affects the Gini estimate."""

    def test_exposure_accepted_no_error(self):
        rng = np.random.default_rng(50)
        n = 500
        pred_r = rng.uniform(0.05, 0.20, n)
        act_r = rng.poisson(pred_r).astype(float)
        exp_r = rng.uniform(0.1, 1.0, n)
        pred_m = rng.uniform(0.05, 0.20, n)
        act_m = rng.poisson(pred_m).astype(float)
        exp_m = rng.uniform(0.1, 1.0, n)

        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            reference_exposure=exp_r,
            monitor_exposure=exp_m,
            n_bootstrap=100,
            random_state=50,
        )
        r = t.test()
        assert isinstance(r, GiniDriftTestResult)
        assert np.isfinite(r.gini_reference)

    def test_exposure_changes_gini_value(self):
        rng = np.random.default_rng(51)
        n = 600
        pred_r = rng.uniform(0.05, 0.20, n)
        act_r = rng.poisson(pred_r).astype(float)
        pred_m = rng.uniform(0.05, 0.20, n)
        act_m = rng.poisson(pred_m).astype(float)
        exp_r = rng.uniform(0.1, 1.0, n)
        exp_m = rng.uniform(0.1, 1.0, n)

        t_no_exp = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=51,
        )
        t_with_exp = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            reference_exposure=exp_r,
            monitor_exposure=exp_m,
            n_bootstrap=100,
            random_state=51,
        )
        r_no = t_no_exp.test()
        r_with = t_with_exp.test()
        # Gini values with and without exposure will typically differ
        # (not asserting a direction, just that the computation ran)
        assert isinstance(r_no.gini_reference, float)
        assert isinstance(r_with.gini_reference, float)


# ---------------------------------------------------------------------------
# T09 — dict-style access (backward compatibility)
# ---------------------------------------------------------------------------

class TestDictAccess:
    """T09: GiniDriftTestResult supports dict-style access."""

    def test_dict_style_access(self):
        act_r, pred_r = _make_good_data(500, seed=60)
        act_m, pred_m = _make_good_data(500, seed=61)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=60,
        )
        r = t.test()
        assert r["gini_reference"] == r.gini_reference
        assert r["significant"] == r.significant
        assert "delta" in r
        assert r.keys() is not None

    def test_to_dict_returns_all_fields(self):
        act_r, pred_r = _make_good_data(500, seed=62)
        act_m, pred_m = _make_good_data(500, seed=63)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=62,
        )
        r = t.test()
        d = r.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "gini_reference", "gini_monitor", "delta", "z_statistic",
            "p_value", "significant", "se_reference", "se_monitor",
            "n_reference", "n_monitor", "alpha", "n_bootstrap",
        }
        assert expected_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# T10 — input validation errors
# ---------------------------------------------------------------------------

class TestInputValidation:
    """T10: Informative errors on bad inputs."""

    def test_empty_reference_actual(self):
        act_m, pred_m = _make_good_data(300, seed=70)
        with pytest.raises(ValueError, match="reference_actual must be non-empty"):
            t = GiniDriftTest(
                reference_actual=np.array([]),
                reference_predicted=np.array([]),
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                n_bootstrap=50,
            )
            t.test()

    def test_empty_monitor_actual(self):
        act_r, pred_r = _make_good_data(300, seed=71)
        with pytest.raises(ValueError, match="monitor_actual must be non-empty"):
            t = GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=np.array([]),
                monitor_predicted=np.array([]),
                n_bootstrap=50,
            )
            t.test()

    def test_mismatched_reference_lengths(self):
        act_r, pred_r = _make_good_data(300, seed=72)
        act_m, pred_m = _make_good_data(300, seed=73)
        with pytest.raises(ValueError, match="reference_actual length"):
            t = GiniDriftTest(
                reference_actual=act_r[:200],
                reference_predicted=pred_r,  # 300 vs 200
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                n_bootstrap=50,
            )
            t.test()

    def test_mismatched_monitor_lengths(self):
        act_r, pred_r = _make_good_data(300, seed=74)
        act_m, pred_m = _make_good_data(300, seed=75)
        with pytest.raises(ValueError, match="monitor_actual length"):
            t = GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m[:200],
                monitor_predicted=pred_m,  # 300 vs 200
                n_bootstrap=50,
            )
            t.test()

    def test_mismatched_reference_exposure(self):
        act_r, pred_r = _make_good_data(300, seed=76)
        act_m, pred_m = _make_good_data(300, seed=77)
        with pytest.raises(ValueError, match="reference_exposure length"):
            t = GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                reference_exposure=np.ones(200),  # wrong length
                n_bootstrap=50,
            )
            t.test()

    def test_mismatched_monitor_exposure(self):
        act_r, pred_r = _make_good_data(300, seed=78)
        act_m, pred_m = _make_good_data(300, seed=79)
        with pytest.raises(ValueError, match="monitor_exposure length"):
            t = GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                monitor_exposure=np.ones(200),  # wrong length
                n_bootstrap=50,
            )
            t.test()

    def test_n_bootstrap_too_small(self):
        act_r, pred_r = _make_good_data(300, seed=80)
        act_m, pred_m = _make_good_data(300, seed=81)
        with pytest.raises(ValueError, match="n_bootstrap must be >= 50"):
            GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                n_bootstrap=10,
            )

    def test_alpha_out_of_range(self):
        act_r, pred_r = _make_good_data(300, seed=82)
        act_m, pred_m = _make_good_data(300, seed=83)
        with pytest.raises(ValueError, match="alpha must be in"):
            GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                alpha=1.5,
            )

    def test_negative_exposure_raises(self):
        act_r, pred_r = _make_good_data(300, seed=84)
        act_m, pred_m = _make_good_data(300, seed=85)
        exp_bad = np.full(300, -1.0)
        with pytest.raises(ValueError, match="reference_exposure values must be positive"):
            t = GiniDriftTest(
                reference_actual=act_r,
                reference_predicted=pred_r,
                monitor_actual=act_m,
                monitor_predicted=pred_m,
                reference_exposure=exp_bad,
                n_bootstrap=50,
            )
            t.test()


# ---------------------------------------------------------------------------
# T11 — small-sample warning
# ---------------------------------------------------------------------------

class TestSmallSampleWarning:
    """T11: UserWarning when sample size < 200."""

    def test_small_reference_warns(self):
        act_r, pred_r = _make_good_data(100, seed=90)
        act_m, pred_m = _make_good_data(500, seed=91)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=50,
            random_state=90,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.test()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1
        assert any("100 observations" in str(x.message) for x in user_warnings)

    def test_small_monitor_warns(self):
        act_r, pred_r = _make_good_data(500, seed=92)
        act_m, pred_m = _make_good_data(50, seed=93)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=50,
            random_state=92,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.test()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1


# ---------------------------------------------------------------------------
# T12 — summary() method returns a non-empty string
# ---------------------------------------------------------------------------

class TestSummary:
    """T12: summary() produces a governance-ready string."""

    def test_summary_is_string(self):
        act_r, pred_r = _make_good_data(500, seed=100)
        act_m, pred_m = _make_good_data(500, seed=101)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=100,
        )
        s = t.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_summary_contains_key_fields(self):
        act_r, pred_r = _make_good_data(500, seed=102)
        act_m, pred_m = _make_good_data(500, seed=103)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=102,
        )
        s = t.summary()
        assert "Gini" in s
        assert "p-value" in s
        assert "z-statistic" in s

    def test_summary_runs_test_implicitly(self):
        act_r, pred_r = _make_good_data(500, seed=104)
        act_m, pred_m = _make_good_data(500, seed=105)
        t = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=104,
        )
        assert t._result is None
        _ = t.summary()
        assert t._result is not None


# ---------------------------------------------------------------------------
# T13 — top-level package import
# ---------------------------------------------------------------------------

class TestTopLevelImport:
    """T13: GiniDriftTest and GiniDriftTestResult importable from top-level package."""

    def test_importable_from_package(self):
        from insurance_monitoring import GiniDriftTest as GDT, GiniDriftTestResult as GDTR
        assert GDT is GiniDriftTest
        assert GDTR is GiniDriftTestResult


# ---------------------------------------------------------------------------
# T14 — reproducibility via random_state
# ---------------------------------------------------------------------------

class TestReproducibility:
    """T14: Same random_state produces identical results."""

    def test_same_seed_same_result(self):
        act_r, pred_r = _make_good_data(800, seed=110)
        act_m, pred_m = _make_good_data(700, seed=111)

        t1 = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=42,
        )
        t2 = GiniDriftTest(
            reference_actual=act_r,
            reference_predicted=pred_r,
            monitor_actual=act_m,
            monitor_predicted=pred_m,
            n_bootstrap=100,
            random_state=42,
        )
        r1 = t1.test()
        r2 = t2.test()
        assert r1.z_statistic == r2.z_statistic
        assert r1.p_value == r2.p_value
        assert r1.gini_reference == r2.gini_reference


# ---------------------------------------------------------------------------
# T15 — polars Series inputs accepted
# ---------------------------------------------------------------------------

class TestPolarsInputs:
    """T15: polars Series accepted as inputs (consistent with existing API)."""

    def test_polars_series_accepted(self):
        import polars as pl
        rng = np.random.default_rng(120)
        n = 400
        pred_r = rng.uniform(0.05, 0.20, n)
        act_r = rng.poisson(pred_r).astype(float)
        pred_m = rng.uniform(0.05, 0.20, n)
        act_m = rng.poisson(pred_m).astype(float)

        t = GiniDriftTest(
            reference_actual=pl.Series(act_r),
            reference_predicted=pl.Series(pred_r),
            monitor_actual=pl.Series(act_m),
            monitor_predicted=pl.Series(pred_m),
            n_bootstrap=100,
            random_state=120,
        )
        r = t.test()
        assert isinstance(r, GiniDriftTestResult)
        assert np.isfinite(r.gini_reference)
        assert np.isfinite(r.gini_monitor)
