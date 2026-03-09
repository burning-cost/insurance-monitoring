"""Tests for v0.2.0 threshold changes: alpha=0.32 defaults in GiniDriftThresholds."""

import pytest

from insurance_monitoring.thresholds import (
    GiniDriftThresholds,
    MonitoringThresholds,
)


class TestGiniDriftThresholdsV2:
    """GiniDriftThresholds with new alpha=0.32 defaults."""

    def test_default_amber_p_value_is_0_32(self):
        """Default amber_p_value must now be 0.32 (alpha=0.32 monitoring rule)."""
        t = GiniDriftThresholds()
        assert t.amber_p_value == pytest.approx(0.32)

    def test_default_red_p_value_is_0_10(self):
        """Default red_p_value must now be 0.10 for monitoring context."""
        t = GiniDriftThresholds()
        assert t.red_p_value == pytest.approx(0.10)

    def test_classify_green_above_amber(self):
        """p-value >= amber_p_value (0.32) should be green."""
        t = GiniDriftThresholds()
        assert t.classify(0.40) == "green"
        assert t.classify(0.32) == "green"
        assert t.classify(1.00) == "green"

    def test_classify_amber_between_red_and_amber(self):
        """p-value between red_p_value (0.10) and amber_p_value (0.32) should be amber."""
        t = GiniDriftThresholds()
        assert t.classify(0.20) == "amber"
        assert t.classify(0.15) == "amber"
        assert t.classify(0.10) == "amber"

    def test_classify_red_below_red(self):
        """p-value < red_p_value (0.10) should be red."""
        t = GiniDriftThresholds()
        assert t.classify(0.05) == "red"
        assert t.classify(0.01) == "red"
        assert t.classify(0.00) == "red"

    def test_traditional_thresholds_configurable(self):
        """Traditional alpha=0.05 thresholds must still be configurable."""
        t = GiniDriftThresholds(amber_p_value=0.10, red_p_value=0.05)
        assert t.amber_p_value == pytest.approx(0.10)
        assert t.red_p_value == pytest.approx(0.05)
        assert t.classify(0.15) == "green"
        assert t.classify(0.07) == "amber"
        assert t.classify(0.03) == "red"

    def test_monitoring_thresholds_default_uses_new_gini(self):
        """MonitoringThresholds() default should have amber=0.32, red=0.10."""
        mt = MonitoringThresholds()
        assert mt.gini_drift.amber_p_value == pytest.approx(0.32)
        assert mt.gini_drift.red_p_value == pytest.approx(0.10)

    def test_green_band_at_boundary(self):
        """p-value exactly at amber_p_value should be green (>= boundary)."""
        t = GiniDriftThresholds(amber_p_value=0.32, red_p_value=0.10)
        # classify uses >= for green
        assert t.classify(0.32) == "green"

    def test_amber_band_at_red_boundary(self):
        """p-value exactly at red_p_value should be amber (>= boundary)."""
        t = GiniDriftThresholds(amber_p_value=0.32, red_p_value=0.10)
        assert t.classify(0.10) == "amber"
