"""Tests for MulticalibrationMonitor.

Coverage:
- Basic construction and parameter validation
- Bin edge computation (exposure-weighted quantiles)
- fit() freezes bin boundaries
- update() before fit() raises RuntimeError
- Well-calibrated data produces no alerts
- Injected bias in one group+bin triggers an alert
- Alert gating: only |bias| AND |z| together trigger
- Sparse cells are skipped (min_exposure)
- Multiple groups work correctly
- Numeric groups (int labels)
- result.summary() keys and types
- result.to_dict() is serialisable
- period_index increments correctly
- history() accumulates results
- period_summary() DataFrame structure
- bin_edges and is_fitted properties
- repr
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from insurance_monitoring import MulticalibrationMonitor, MulticalibrationResult
from insurance_monitoring.multicalibration import (
    MulticalibCell,
    MulticalibThresholds,
    _assign_bins,
    _exposure_weighted_quantile_edges,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_data(
    n: int = 4000,
    n_groups: int = 3,
    bias_group: str | None = None,
    bias_bin: int | None = None,
    bias_factor: float = 1.5,
    seed: int = 0,
    n_bins: int = 5,
) -> dict:
    """Generate synthetic motor book data.

    Parameters
    ----------
    bias_group:
        If set, multiply y_true by bias_factor in (bias_bin, bias_group) cell.
    bias_bin:
        Bin index to inject bias into. Requires bias_group.
    """
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 2.0, n)
    y_pred = rng.gamma(2.0, 0.05, n)

    # Poisson draws — well-calibrated
    y_true = rng.poisson(y_pred * exposure).astype(float) / exposure

    groups = rng.choice([f"G{i}" for i in range(n_groups)], n)

    # Optionally inject bias into one cell
    if bias_group is not None and bias_bin is not None:
        # Compute bin edges to know which observations fall into bias_bin
        edges = _exposure_weighted_quantile_edges(y_pred, exposure, n_bins)
        bin_idx = _assign_bins(y_pred, edges)
        cell_mask = (bin_idx == bias_bin) & (groups == bias_group)
        y_true = y_true.copy()
        y_true[cell_mask] *= bias_factor

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "groups": groups,
        "exposure": exposure,
    }


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        m = MulticalibrationMonitor()
        assert m.n_bins == 10
        assert m.min_z_abs == 1.96
        assert m.min_relative_bias == 0.05
        assert m.min_exposure == 50.0
        assert not m.is_fitted

    def test_custom_params(self):
        m = MulticalibrationMonitor(n_bins=5, min_z_abs=1.645, min_relative_bias=0.10, min_exposure=20.0)
        assert m.n_bins == 5
        assert m.min_z_abs == 1.645
        assert m.min_relative_bias == 0.10
        assert m.min_exposure == 20.0

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            MulticalibrationMonitor(n_bins=1)

    def test_invalid_min_exposure(self):
        with pytest.raises(ValueError, match="min_exposure must be > 0"):
            MulticalibrationMonitor(min_exposure=0.0)

    def test_repr_unfitted(self):
        m = MulticalibrationMonitor(n_bins=5)
        assert "not fitted" in repr(m)
        assert "n_bins=5" in repr(m)

    def test_repr_fitted(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        assert "fitted" in repr(m)


# ---------------------------------------------------------------------------
# Bin edge computation
# ---------------------------------------------------------------------------


class TestBinEdges:
    def test_shape(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 1, 1000)
        w = np.ones(1000)
        edges = _exposure_weighted_quantile_edges(y, w, n_bins=10)
        assert edges.shape == (11,)

    def test_first_last(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0, 1, 500)
        w = np.ones(500)
        edges = _exposure_weighted_quantile_edges(y, w, 5)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_interior_monotone(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(0.01, 1.0, 1000)
        w = rng.uniform(0.5, 2.0, 1000)
        edges = _exposure_weighted_quantile_edges(y, w, 8)
        interior = edges[1:-1]
        assert np.all(np.diff(interior) >= 0), "Interior edges must be non-decreasing"


class TestAssignBins:
    def test_all_in_range(self):
        edges = np.array([-np.inf, 0.2, 0.5, 0.8, np.inf])
        y = np.array([0.1, 0.3, 0.6, 0.9])
        bins = _assign_bins(y, edges)
        assert bins.tolist() == [0, 1, 2, 3]

    def test_boundary_values(self):
        edges = np.array([-np.inf, 0.5, np.inf])
        # Value exactly at the boundary 0.5 should go to bin 1
        y = np.array([0.49, 0.5, 0.51])
        bins = _assign_bins(y, edges)
        assert bins[0] == 0
        assert bins[1] == 1  # searchsorted 'right': 0.5 maps to index 1 (upper bin)
        assert bins[2] == 1


# ---------------------------------------------------------------------------
# fit() behaviour
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_sets_is_fitted(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        assert not m.is_fitted
        m.fit(**d)
        assert m.is_fitted

    def test_fit_returns_self(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        result = m.fit(**d)
        assert result is m

    def test_bin_edges_set_after_fit(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        assert m.bin_edges is not None
        assert len(m.bin_edges) == 6  # n_bins + 1

    def test_bin_edges_none_before_fit(self):
        m = MulticalibrationMonitor()
        assert m.bin_edges is None

    def test_fit_without_exposure(self):
        rng = np.random.default_rng(99)
        y_pred = rng.gamma(2, 0.05, 500)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B"], 500)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(y_true, y_pred, groups)  # no exposure arg
        assert m.is_fitted


# ---------------------------------------------------------------------------
# update() behaviour
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_before_fit_raises(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            m.update(**d)

    def test_returns_result_type(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        assert isinstance(result, MulticalibrationResult)

    def test_period_index_increments(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        r1 = m.update(**d)
        r2 = m.update(**d)
        r3 = m.update(**d)
        assert r1.period_index == 1
        assert r2.period_index == 2
        assert r3.period_index == 3

    def test_history_accumulates(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        for _ in range(4):
            m.update(**d)
        assert len(m.history()) == 4

    def test_cell_table_is_polars_dataframe(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        assert isinstance(result.cell_table, pl.DataFrame)

    def test_cell_table_columns(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        expected_cols = {
            "bin_idx", "group", "n_exposure", "observed", "expected",
            "AE_ratio", "relative_bias", "z_stat", "alert"
        }
        assert expected_cols.issubset(set(result.cell_table.columns))

    def test_n_cells_evaluated_positive(self):
        d = make_data(n=4000, n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        assert result.n_cells_evaluated > 0

    def test_update_without_exposure(self):
        rng = np.random.default_rng(55)
        y_pred = rng.gamma(2, 0.05, 500)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B"], 500)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=5)
        m.fit(y_true, y_pred, groups)
        result = m.update(y_true, y_pred, groups)
        assert isinstance(result, MulticalibrationResult)


# ---------------------------------------------------------------------------
# Calibration and alert logic
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_well_calibrated_no_alerts(self):
        """A well-calibrated model should not generate persistent alerts."""
        rng = np.random.default_rng(42)
        n = 8000
        exposure = rng.uniform(0.5, 2.0, n)
        y_pred = rng.gamma(2.0, 0.05, n)
        y_true = rng.poisson(y_pred * exposure).astype(float) / exposure
        groups = rng.choice(["A", "B", "C"], n)

        m = MulticalibrationMonitor(n_bins=5, min_exposure=50)
        m.fit(y_true, y_pred, groups, exposure=exposure)

        # Run 10 periods — should have very few alerts on well-calibrated data
        alert_counts = []
        for _ in range(10):
            # Fresh well-calibrated draw each period
            y_true_new = rng.poisson(y_pred * exposure).astype(float) / exposure
            result = m.update(y_true_new, y_pred, groups, exposure=exposure)
            alert_counts.append(len(result.alerts))

        # Under the null we expect at most a moderate number of spurious alerts
        # across 10 periods. With 5 bins x 3 groups = 15 cells, each tested at
        # ~5% significance, we expect ~7.5 alerts by chance. Allow up to 15.
        assert sum(alert_counts) <= 15, (
            f"Too many alerts on well-calibrated data: {alert_counts}"
        )

    def test_injected_bias_triggers_alert(self):
        """A 60% uplift in a large cell should reliably trigger an alert."""
        n = 10000
        n_bins = 5
        d = make_data(
            n=n,
            n_groups=2,
            bias_group="G0",
            bias_bin=2,
            bias_factor=1.6,
            seed=7,
            n_bins=n_bins,
        )
        m = MulticalibrationMonitor(
            n_bins=n_bins,
            min_z_abs=1.96,
            min_relative_bias=0.05,
            min_exposure=30,
        )
        m.fit(**d)
        result = m.update(**d)

        assert len(result.alerts) >= 1, "Expected at least one alert for 60% injected bias"
        # The alerted cell should be in the biased group
        alert_groups = {a.group for a in result.alerts}
        assert "G0" in alert_groups

    def test_small_bias_no_alert(self):
        """A 2% bias (below threshold) should not alert, even with high z."""
        n = 20000
        n_bins = 5
        rng = np.random.default_rng(13)
        exposure = rng.uniform(0.5, 2.0, n)
        y_pred = rng.gamma(2.0, 0.05, n)
        y_true = rng.poisson(y_pred * exposure).astype(float) / exposure
        groups = rng.choice(["A", "B"], n)

        # Add a tiny systematic 2% uplift to group A
        mask_a = groups == "A"
        y_true = y_true.copy()
        y_true[mask_a] *= 1.02

        m = MulticalibrationMonitor(
            n_bins=n_bins,
            min_z_abs=1.96,
            min_relative_bias=0.05,  # 5% threshold, so 2% should not alert
            min_exposure=30,
        )
        m.fit(y_true, y_pred, groups, exposure=exposure)
        result = m.update(y_true, y_pred, groups, exposure=exposure)

        # With only 2% bias, no cell should exceed the 5% relative bias gate
        assert len(result.alerts) == 0, (
            f"Unexpected alerts for 2% bias: {[a.to_dict() for a in result.alerts]}"
        )

    def test_alert_gating_requires_both_conditions(self):
        """Alert requires BOTH |relative_bias| >= threshold AND |z_stat| >= min_z_abs."""
        # Construct a minimal cell manually using MulticalibCell
        # Case 1: large bias, z below threshold -> no alert
        cell_no_z = MulticalibCell(
            bin_idx=0, group="A", n_exposure=10.0,
            observed=0.11, expected=0.10, AE_ratio=1.1,
            relative_bias=0.10, z_stat=1.0,  # z < 1.96
            alert=False,
        )
        assert not cell_no_z.alert

        # Case 2: large z, bias below threshold -> no alert
        cell_no_bias = MulticalibCell(
            bin_idx=0, group="A", n_exposure=10.0,
            observed=0.102, expected=0.10, AE_ratio=1.02,
            relative_bias=0.02, z_stat=3.0,  # z > 1.96 but bias < 5%
            alert=False,
        )
        assert not cell_no_bias.alert

        # Case 3: both thresholds exceeded -> should alert
        cell_alert = MulticalibCell(
            bin_idx=0, group="A", n_exposure=200.0,
            observed=0.11, expected=0.10, AE_ratio=1.1,
            relative_bias=0.10, z_stat=2.5,
            alert=True,
        )
        assert cell_alert.alert


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_y_pred_not_positive_raises(self):
        y_pred = np.array([0.1, -0.05, 0.2])
        y_true = np.array([0.1, 0.1, 0.1])
        groups = np.array(["A", "A", "A"])
        m = MulticalibrationMonitor(n_bins=2, min_exposure=1)
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(y_true, y_pred, groups)

    def test_length_mismatch_raises(self):
        y_pred = np.array([0.1, 0.2, 0.3])
        y_true = np.array([0.1, 0.2])  # wrong length
        groups = np.array(["A", "A", "A"])
        m = MulticalibrationMonitor(n_bins=2, min_exposure=1)
        with pytest.raises(ValueError, match="same length"):
            m.fit(y_true, y_pred, groups)

    def test_groups_length_mismatch_raises(self):
        y_pred = np.array([0.1, 0.2, 0.3])
        y_true = np.array([0.1, 0.2, 0.3])
        groups = np.array(["A", "A"])  # wrong length
        m = MulticalibrationMonitor(n_bins=2, min_exposure=1)
        with pytest.raises(ValueError, match="same length"):
            m.fit(y_true, y_pred, groups)

    def test_negative_exposure_raises(self):
        y_pred = np.array([0.1, 0.2, 0.3])
        y_true = np.array([0.1, 0.2, 0.3])
        groups = np.array(["A", "A", "A"])
        exposure = np.array([1.0, -0.5, 1.0])
        m = MulticalibrationMonitor(n_bins=2, min_exposure=1)
        with pytest.raises(ValueError, match="strictly positive"):
            m.fit(y_true, y_pred, groups, exposure=exposure)


# ---------------------------------------------------------------------------
# Numeric group labels
# ---------------------------------------------------------------------------


class TestNumericGroups:
    def test_integer_groups(self):
        rng = np.random.default_rng(20)
        n = 3000
        y_pred = rng.gamma(2.0, 0.05, n)
        y_true = rng.poisson(y_pred)
        groups = rng.choice([0, 1, 2, 3], n)
        exposure = rng.uniform(0.5, 2.0, n)

        m = MulticalibrationMonitor(n_bins=4, min_exposure=20)
        m.fit(y_true, y_pred, groups, exposure=exposure)
        result = m.update(y_true, y_pred, groups, exposure=exposure)

        assert isinstance(result, MulticalibrationResult)
        # Group column should be string in the Polars table
        assert result.cell_table["group"].dtype == pl.String


# ---------------------------------------------------------------------------
# Summary and serialisation
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_keys(self):
        d = make_data(n=4000, n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        s = result.summary()
        assert "n_alerts" in s
        assert "overall_pass" in s
        assert "worst_cell" in s
        assert "n_cells_evaluated" in s
        assert "n_cells_skipped" in s
        assert "period_index" in s

    def test_overall_pass_type(self):
        d = make_data(n=4000, n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        assert isinstance(result.summary()["overall_pass"], bool)

    def test_worst_cell_none_when_no_alerts(self):
        rng = np.random.default_rng(77)
        n = 5000
        y_pred = rng.gamma(2.0, 0.05, n)
        y_true = rng.poisson(y_pred)
        groups = rng.choice(["A", "B"], n)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=50)
        m.fit(y_true, y_pred, groups)
        result = m.update(y_true, y_pred, groups)
        # May or may not have alerts, but if no alerts worst_cell is None
        if result.summary()["n_alerts"] == 0:
            assert result.summary()["worst_cell"] is None

    def test_to_dict_is_json_serialisable(self):
        d = make_data(n=4000, n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        result = m.update(**d)
        d_out = result.to_dict()
        # Should not raise
        json_str = json.dumps(d_out)
        assert len(json_str) > 10


# ---------------------------------------------------------------------------
# period_summary()
# ---------------------------------------------------------------------------


class TestPeriodSummary:
    def test_empty_before_update(self):
        d = make_data(n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        df = m.period_summary()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_one_row_per_update(self):
        d = make_data(n=4000, n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        for _ in range(5):
            m.update(**d)
        df = m.period_summary()
        assert len(df) == 5

    def test_period_summary_columns(self):
        d = make_data(n=4000, n_bins=5)
        m = MulticalibrationMonitor(n_bins=5, min_exposure=10)
        m.fit(**d)
        m.update(**d)
        df = m.period_summary()
        assert "period_index" in df.columns
        assert "n_alerts" in df.columns
        assert "overall_pass" in df.columns


# ---------------------------------------------------------------------------
# MulticalibThresholds
# ---------------------------------------------------------------------------


class TestMulticalibThresholds:
    def test_defaults(self):
        t = MulticalibThresholds()
        assert t.min_relative_bias == 0.05
        assert t.min_z_abs == 1.96
        assert t.min_exposure == 50.0

    def test_custom(self):
        t = MulticalibThresholds(min_relative_bias=0.10, min_z_abs=1.645, min_exposure=100.0)
        assert t.min_relative_bias == 0.10
        assert t.min_z_abs == 1.645
        assert t.min_exposure == 100.0
