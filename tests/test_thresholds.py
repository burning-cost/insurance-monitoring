"""Tests for insurance_monitoring.thresholds module."""

import pytest

from insurance_monitoring.thresholds import (
    AERatioThresholds,
    GiniDriftThresholds,
    MonitoringThresholds,
    PSIThresholds,
)


class TestPSIThresholds:
    def test_green(self):
        t = PSIThresholds()
        assert t.classify(0.05) == "green"
        assert t.classify(0.0) == "green"

    def test_amber(self):
        t = PSIThresholds()
        assert t.classify(0.10) == "amber"
        assert t.classify(0.20) == "amber"

    def test_red(self):
        t = PSIThresholds()
        assert t.classify(0.25) == "red"
        assert t.classify(1.0) == "red"

    def test_custom_thresholds(self):
        t = PSIThresholds(green_max=0.05, amber_max=0.15)
        assert t.classify(0.03) == "green"
        assert t.classify(0.08) == "amber"
        assert t.classify(0.20) == "red"


class TestAERatioThresholds:
    def test_green(self):
        t = AERatioThresholds()
        assert t.classify(1.0) == "green"
        assert t.classify(0.95) == "green"
        assert t.classify(1.05) == "green"

    def test_amber(self):
        t = AERatioThresholds()
        assert t.classify(0.92) == "amber"
        assert t.classify(1.08) == "amber"

    def test_red(self):
        t = AERatioThresholds()
        assert t.classify(0.75) == "red"
        assert t.classify(1.30) == "red"


class TestGiniDriftThresholds:
    def test_green(self):
        """v0.2.0: default amber_p_value=0.32. Values at or above 0.32 are green."""
        t = GiniDriftThresholds()
        assert t.classify(0.50) == "green"
        assert t.classify(0.32) == "green"   # boundary is green
        assert t.classify(0.40) == "green"

    def test_amber(self):
        """v0.2.0: amber band is between red_p_value=0.10 and amber_p_value=0.32."""
        t = GiniDriftThresholds()
        assert t.classify(0.20) == "amber"
        assert t.classify(0.15) == "amber"
        assert t.classify(0.10) == "amber"   # boundary is amber

    def test_red(self):
        """v0.2.0: red_p_value=0.10. Values strictly below 0.10 are red."""
        t = GiniDriftThresholds()
        assert t.classify(0.05) == "red"
        assert t.classify(0.01) == "red"
        assert t.classify(0.0) == "red"

    def test_traditional_thresholds_still_work(self):
        """Traditional alpha=0.05 thresholds configurable for regulatory use."""
        t = GiniDriftThresholds(amber_p_value=0.10, red_p_value=0.05)
        assert t.classify(0.50) == "green"
        assert t.classify(0.10) == "green"   # boundary green under old thresholds
        assert t.classify(0.07) == "amber"
        assert t.classify(0.01) == "red"


class TestMonitoringThresholds:
    def test_default_construction(self):
        t = MonitoringThresholds()
        assert isinstance(t.psi, PSIThresholds)
        assert isinstance(t.ae_ratio, AERatioThresholds)
        assert isinstance(t.gini_drift, GiniDriftThresholds)

    def test_custom_psi(self):
        t = MonitoringThresholds(psi=PSIThresholds(green_max=0.05, amber_max=0.15))
        assert t.psi.green_max == 0.05
