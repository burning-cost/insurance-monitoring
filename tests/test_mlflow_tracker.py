"""Tests for insurance_monitoring.mlflow_tracker.

All MLflow interactions are mocked — we test the bridge logic (data preparation,
tag generation, artifact serialisation) without needing a real tracking server.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeGiniResult:
    """Minimal stand-in for GiniDriftResult."""
    z_statistic: float = 1.8
    p_value: float = 0.072
    gini_change: float = -0.03
    significant: bool = False
    reference_gini: float = 0.45
    current_gini: float = 0.42


@dataclass
class FakeBalanceResult:
    """Minimal stand-in for BalanceResult."""
    ratio: float = 1.04
    p_value: float = 0.21
    significant: bool = False


def _make_mlflow_mock():
    """Build a realistic MLflow mock with the right structure."""
    mlflow_mock = MagicMock()

    # Experiment returned by get_experiment_by_name
    fake_exp = MagicMock()
    fake_exp.experiment_id = "exp-001"
    mlflow_mock.get_experiment_by_name.return_value = fake_exp
    mlflow_mock.create_experiment.return_value = "exp-002"

    # Run returned by start_run context manager
    fake_run_info = MagicMock()
    fake_run_info.run_id = "run-abc123"

    fake_run = MagicMock()
    fake_run.info = fake_run_info
    fake_run.__enter__ = MagicMock(return_value=fake_run)
    fake_run.__exit__ = MagicMock(return_value=False)
    mlflow_mock.start_run.return_value = fake_run

    # search_runs returns empty DataFrame by default
    mlflow_mock.search_runs.return_value = pd.DataFrame()

    return mlflow_mock, fake_run


@contextmanager
def _patch_mlflow(mlflow_mock):
    """Patch both the module-level import and the lazy loader."""
    with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
        with patch(
            "insurance_monitoring.mlflow_tracker._require_mlflow",
            return_value=mlflow_mock,
        ):
            yield mlflow_mock


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestMonitoringTrackerInit:
    def test_basic_init(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        tracker = MonitoringTracker("motor_freq_glm", model_version="3")
        assert tracker.model_name == "motor_freq_glm"
        assert tracker.model_version == "3"

    def test_version_int_converted_to_str(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        tracker = MonitoringTracker("motor_freq_glm", model_version=5)
        assert tracker.model_version == "5"

    def test_version_none_allowed(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        tracker = MonitoringTracker("motor_freq_glm")
        assert tracker.model_version is None

    def test_empty_model_name_raises(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        with pytest.raises(ValueError, match="non-empty"):
            MonitoringTracker("")

    def test_whitespace_model_name_raises(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        with pytest.raises(ValueError, match="non-empty"):
            MonitoringTracker("   ")

    def test_model_name_stripped(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        tracker = MonitoringTracker("  motor_freq_glm  ")
        assert tracker.model_name == "motor_freq_glm"

    def test_tracking_uri_stored(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        tracker = MonitoringTracker("model", tracking_uri="databricks")
        assert tracker._tracking_uri == "databricks"


# ---------------------------------------------------------------------------
# ImportError behaviour when mlflow is absent
# ---------------------------------------------------------------------------


class TestMlflowImportError:
    def test_import_error_raised_with_helpful_message(self):
        """Importing MonitoringTracker is fine; using it raises ImportError
        with install instructions if mlflow is missing."""
        from insurance_monitoring.mlflow_tracker import MonitoringTracker, _require_mlflow

        # Simulate mlflow not installed
        with patch(
            "insurance_monitoring.mlflow_tracker._require_mlflow",
            side_effect=ImportError(
                "MLflow is required for MonitoringTracker but is not installed. "
                "Install it with: pip install insurance-monitoring[mlflow]"
            ),
        ):
            tracker = MonitoringTracker("test_model")
            with pytest.raises(ImportError, match="pip install insurance-monitoring"):
                tracker.log_ae_ratios(1.05)

    def test_require_mlflow_error_message(self, monkeypatch):
        """_require_mlflow raises ImportError when mlflow is missing."""
        from insurance_monitoring import mlflow_tracker

        # Remove mlflow from sys.modules and block re-import
        monkeypatch.setitem(sys.modules, "mlflow", None)  # type: ignore[arg-type]

        with pytest.raises(ImportError, match="pip install insurance-monitoring"):
            mlflow_tracker._require_mlflow()


# ---------------------------------------------------------------------------
# log_ae_ratios
# ---------------------------------------------------------------------------


class TestLogAeRatios:
    def test_scalar_ae_creates_run(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm", model_version="3")
            run_id = tracker.log_ae_ratios(1.05)

        assert run_id == "run-abc123"
        mlflow_mock.start_run.assert_called_once()

    def test_scalar_ae_logs_metric(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            tracker.log_ae_ratios(1.05)

        # Should log "ae_ratio/value" (dots replaced with slashes)
        calls = [str(c) for c in mlflow_mock.log_metric.call_args_list]
        assert any("ae_ratio" in c for c in calls)

    def test_dict_ae_creates_run(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            run_id = tracker.log_ae_ratios({"overall": 1.05, "young_drivers": 0.98})

        assert run_id == "run-abc123"

    def test_run_tagged_with_metric_type(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm", model_version="2")
            tracker.log_ae_ratios(1.02)

        call_kwargs = mlflow_mock.start_run.call_args[1]
        tags = call_kwargs["tags"]
        assert tags["monitoring.metric_type"] == "ae_ratio"
        assert tags["monitoring.model_name"] == "motor_freq_glm"
        assert tags["monitoring.model_version"] == "2"
        assert "monitoring.timestamp" in tags
        assert "monitoring.library_version" in tags

    def test_custom_run_name_used(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            tracker.log_ae_ratios(1.02, run_name="march_2026_check")

        call_kwargs = mlflow_mock.start_run.call_args[1]
        assert call_kwargs["run_name"] == "march_2026_check"

    def test_artifact_logged(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            tracker.log_ae_ratios({"overall": 1.05})

        mlflow_mock.log_artifact.assert_called_once()
        artifact_call = mlflow_mock.log_artifact.call_args
        assert artifact_call[1]["artifact_path"] == "monitoring"

    def test_dataclass_result_handled(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        gini_result = FakeGiniResult()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            tracker.log_ae_ratios({"gini_like": gini_result})

        # Should log numeric fields as metrics
        metric_names = [c[0][0] for c in mlflow_mock.log_metric.call_args_list]
        assert any("z_statistic" in n or "p_value" in n for n in metric_names)


# ---------------------------------------------------------------------------
# log_psi
# ---------------------------------------------------------------------------


class TestLogPsi:
    def test_float_psi_creates_run(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("freq_model")
            run_id = tracker.log_psi(0.12)

        assert run_id == "run-abc123"

    def test_run_tagged_psi(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("freq_model")
            tracker.log_psi({"vehicle_age": 0.05, "ncb": 0.18})

        tags = mlflow_mock.start_run.call_args[1]["tags"]
        assert tags["monitoring.metric_type"] == "psi"


# ---------------------------------------------------------------------------
# log_gini_drift
# ---------------------------------------------------------------------------


class TestLogGiniDrift:
    def test_dataclass_result_logged(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        result = FakeGiniResult()
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("freq_model")
            run_id = tracker.log_gini_drift(result)

        assert run_id == "run-abc123"
        tags = mlflow_mock.start_run.call_args[1]["tags"]
        assert tags["monitoring.metric_type"] == "gini_drift"

    def test_gini_scalars_logged_as_metrics(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        result = FakeGiniResult(p_value=0.04, gini_change=-0.05, significant=True)
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("freq_model")
            tracker.log_gini_drift(result)

        logged_metrics = {c[0][0]: c[0][1] for c in mlflow_mock.log_metric.call_args_list}
        # p_value and gini_change should be in the logged metrics
        assert any("p_value" in k for k in logged_metrics)
        assert any("gini_change" in k for k in logged_metrics)


# ---------------------------------------------------------------------------
# log_monitoring_report
# ---------------------------------------------------------------------------


class TestLogMonitoringReport:
    def test_report_dict_logged(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, fake_run = _make_mlflow_mock()
        report = {
            "ae_ratio": 1.03,
            "psi": 0.09,
            "gini": FakeGiniResult(),
        }
        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("freq_model")
            run_id = tracker.log_monitoring_report(report)

        assert run_id == "run-abc123"
        tags = mlflow_mock.start_run.call_args[1]["tags"]
        assert tags["monitoring.metric_type"] == "monitoring_report"


# ---------------------------------------------------------------------------
# get_monitoring_history
# ---------------------------------------------------------------------------


class TestGetMonitoringHistory:
    def test_returns_empty_dataframe_when_no_experiment(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, _ = _make_mlflow_mock()
        # Experiment doesn't exist yet
        mlflow_mock.get_experiment_by_name.return_value = None

        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("new_model")
            result = tracker.get_monitoring_history()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_dataframe_when_no_runs(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, _ = _make_mlflow_mock()
        mlflow_mock.search_runs.return_value = pd.DataFrame()

        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            result = tracker.get_monitoring_history()

        assert isinstance(result, pd.DataFrame)

    def test_filter_string_includes_model_name(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, _ = _make_mlflow_mock()
        mlflow_mock.search_runs.return_value = pd.DataFrame()

        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm", model_version="3")
            tracker.get_monitoring_history(metric_type="ae_ratio")

        call_kwargs = mlflow_mock.search_runs.call_args[1]
        filter_str = call_kwargs["filter_string"]
        assert "motor_freq_glm" in filter_str
        assert "ae_ratio" in filter_str
        assert "3" in filter_str

    def test_returns_dataframe_with_runs(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, _ = _make_mlflow_mock()
        fake_runs = pd.DataFrame({
            "run_id": ["run-001", "run-002"],
            "run_name": ["ae_ratio_20260301", "ae_ratio_20260201"],
            "start_time": pd.to_datetime(["2026-03-01", "2026-02-01"]),
            "status": ["FINISHED", "FINISHED"],
            "tags.monitoring.metric_type": ["ae_ratio", "ae_ratio"],
            "tags.monitoring.model_version": ["3", "3"],
            "tags.monitoring.timestamp": ["2026-03-01T10:00:00", "2026-02-01T10:00:00"],
            "tags.monitoring.library_version": ["0.8.3", "0.8.3"],
            "metrics.ae_ratio/value": [1.05, 1.02],
        })
        mlflow_mock.search_runs.return_value = fake_runs

        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm", model_version="3")
            result = tracker.get_monitoring_history()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "run_id" in result.columns
        assert "metric_type" in result.columns  # renamed from tags.monitoring.metric_type

    def test_no_metric_type_filter_omitted(self):
        from insurance_monitoring.mlflow_tracker import MonitoringTracker

        mlflow_mock, _ = _make_mlflow_mock()
        mlflow_mock.search_runs.return_value = pd.DataFrame()

        with _patch_mlflow(mlflow_mock):
            tracker = MonitoringTracker("motor_freq_glm")
            tracker.get_monitoring_history()  # no metric_type

        call_kwargs = mlflow_mock.search_runs.call_args[1]
        filter_str = call_kwargs["filter_string"]
        assert "metric_type" not in filter_str


# ---------------------------------------------------------------------------
# _result_to_dict helper
# ---------------------------------------------------------------------------


class TestResultToDict:
    def test_scalar_float(self):
        from insurance_monitoring.mlflow_tracker import _result_to_dict

        assert _result_to_dict(1.05) == {"value": 1.05}

    def test_plain_dict(self):
        from insurance_monitoring.mlflow_tracker import _result_to_dict

        d = {"a": 1.0, "b": 2.0}
        assert _result_to_dict(d) == d

    def test_dataclass(self):
        from insurance_monitoring.mlflow_tracker import _result_to_dict

        result = FakeGiniResult()
        d = _result_to_dict(result)
        assert d["p_value"] == pytest.approx(0.072)
        assert d["significant"] is False

    def test_polars_dataframe(self):
        from insurance_monitoring.mlflow_tracker import _result_to_dict

        df = pl.DataFrame({"feature": ["age", "ncb"], "csi": [0.05, 0.12]})
        d = _result_to_dict(df)
        assert "feature" in d
        assert "csi" in d

    def test_pandas_dataframe(self):
        from insurance_monitoring.mlflow_tracker import _result_to_dict

        df = pd.DataFrame({"segment": ["young", "old"], "ae": [1.1, 0.95]})
        d = _result_to_dict(df)
        assert "segment" in d


# ---------------------------------------------------------------------------
# _extract_scalar_metrics helper
# ---------------------------------------------------------------------------


class TestExtractScalarMetrics:
    def test_flat_dict(self):
        from insurance_monitoring.mlflow_tracker import _extract_scalar_metrics

        d = {"p_value": 0.04, "gini_change": -0.03, "significant": True}
        metrics = _extract_scalar_metrics(d, prefix="gini_drift")
        assert "gini_drift.p_value" in metrics
        assert "gini_drift.gini_change" in metrics
        # Booleans should be excluded
        assert "gini_drift.significant" not in metrics

    def test_nested_dict(self):
        from insurance_monitoring.mlflow_tracker import _extract_scalar_metrics

        d = {"overall": {"ae": 1.05, "ci_lower": 0.98}}
        metrics = _extract_scalar_metrics(d, prefix="ae_ratio")
        assert "ae_ratio.overall.ae" in metrics
        assert metrics["ae_ratio.overall.ae"] == pytest.approx(1.05)

    def test_lists_excluded(self):
        from insurance_monitoring.mlflow_tracker import _extract_scalar_metrics

        d = {"p_value": 0.05, "bootstrap_samples": [0.04, 0.06, 0.05]}
        metrics = _extract_scalar_metrics(d, prefix="test")
        assert "test.p_value" in metrics
        assert "test.bootstrap_samples" not in metrics
