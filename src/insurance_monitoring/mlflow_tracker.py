"""
MLflow integration for insurance-monitoring.

Attaches monitoring metrics to registered MLflow models so that A/E ratios,
PSI/CSI values, and Gini drift results accumulate as time-series alongside
the model version they describe.

The canonical UK pricing workflow on Databricks: you fit a GLM, register it in
MLflow, and then run monthly monitoring. Without this module, monitoring results
sit in spreadsheets or Databricks notebooks disconnected from the model. With it,
every monitoring run is a child run logged against the registered model version —
queryable, auditable, and visible in the MLflow UI.

MLflow is an optional dependency. The rest of insurance-monitoring works fine
without it. If you try to use MonitoringTracker without mlflow installed, you'll
get a clear error telling you how to fix it.

pandas is also an optional dependency. The module imports gracefully without it;
pandas is only needed at runtime inside get_monitoring_history() (which calls
mlflow.search_runs(), itself returning a pandas DataFrame).

Usage
-----
::

    from insurance_monitoring.mlflow_tracker import MonitoringTracker
    from insurance_monitoring import ae_ratio, psi

    tracker = MonitoringTracker(
        model_name="motor_freq_glm",
        model_version="3",
        tracking_uri="databricks",
    )

    ae = ae_ratio(actual, predicted)
    tracker.log_ae_ratios({"overall": ae, "region": ae_by_region})

    psi_val = psi(reference, current)
    tracker.log_psi({"premium": psi_val})

    history = tracker.get_monitoring_history()  # returns a DataFrame
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

import polars as pl

from importlib.metadata import version, PackageNotFoundError

try:
    _LIB_VERSION = version("insurance-monitoring")
except PackageNotFoundError:
    _LIB_VERSION = "0.0.0"


def _require_mlflow():
    """Lazy MLflow import — raises a clear error if not installed."""
    try:
        import mlflow  # noqa: F401
        return mlflow
    except ImportError as e:
        raise ImportError(
            "MLflow is required for MonitoringTracker but is not installed. "
            "Install it with: pip install insurance-monitoring[mlflow]\n"
            "Or directly: pip install mlflow>=3.0"
        ) from e


def _require_pandas():
    """Raise a clear error if pandas is not installed."""
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for get_monitoring_history() but is not installed. "
            "Install it with: pip install pandas"
        )


def _result_to_dict(result: Any) -> dict:
    """Convert a metric result object to a plain dict for JSON serialisation.

    Handles:
    - dataclasses (GiniDriftResult, BalanceResult, etc.)
    - plain dicts
    - scalars (float/int from ae_ratio without segments)
    - polars DataFrames
    - pandas DataFrames (only when pandas is installed)
    """
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    if isinstance(result, dict):
        return result
    if isinstance(result, (float, int)):
        return {"value": result}
    if isinstance(result, pl.DataFrame):
        return result.to_dict(as_series=False)
    if _PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
        return result.to_dict(orient="list")
    # Fallback: attempt dict-style access (GiniDriftResult supports this)
    try:
        return dict(result)
    except (TypeError, ValueError):
        return {"value": str(result)}


def _extract_scalar_metrics(result_dict: dict, prefix: str) -> dict[str, float]:
    """Flatten a result dict into MLflow-loggable scalar metrics.

    Only numeric leaf values are included. Lists are skipped (they can't be
    logged as a single MLflow metric step). The full detail goes into the JSON
    artifact.
    """
    out: dict[str, float] = {}
    for k, v in result_dict.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_extract_scalar_metrics(v, key))
    return out


class MonitoringTracker:
    """Attaches insurance-monitoring metrics to MLflow registered models.

    Each call to a ``log_*`` method creates a new MLflow run tagged to the
    model version. The experiment is named after the model to keep runs
    discoverable in the MLflow UI.

    Scalar metrics are logged as MLflow metrics (enabling time-series charts).
    Full result objects are serialised to a JSON artifact on the run for
    complete audit detail.

    Args:
        model_name: MLflow registered model name, e.g. ``"motor_freq_glm"``.
        model_version: Version string or int. Pass ``None`` to log against the
            model name without a specific version tag (useful when monitoring
            the latest production alias).
        tracking_uri: MLflow tracking URI. Defaults to the
            ``MLFLOW_TRACKING_URI`` environment variable. Pass
            ``"databricks"`` to use Databricks-managed MLflow.
    """

    def __init__(
        self,
        model_name: str,
        model_version: Optional[Union[str, int]] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name.strip()
        self.model_version = str(model_version) if model_version is not None else None
        self._tracking_uri = tracking_uri

    def _mlflow(self):
        """Return the mlflow module, raising if not installed."""
        return _require_mlflow()

    def _get_experiment_name(self) -> str:
        return f"monitoring/{self.model_name}"

    def _ensure_experiment(self, mlflow) -> str:
        """Get or create the monitoring experiment, return its ID."""
        exp_name = self._get_experiment_name()
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = mlflow.create_experiment(exp_name)
        else:
            exp_id = exp.experiment_id
        return exp_id

    def _base_tags(self, metric_type: str) -> dict[str, str]:
        tags = {
            "monitoring.metric_type": metric_type,
            "monitoring.timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "monitoring.library_version": _LIB_VERSION,
            "monitoring.model_name": self.model_name,
        }
        if self.model_version is not None:
            tags["monitoring.model_version"] = self.model_version
        return tags

    def _log_result(
        self,
        mlflow,
        metric_type: str,
        results: dict[str, Any],
        run_name: Optional[str],
    ) -> str:
        """Core logging logic. Returns the run ID."""
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        exp_id = self._ensure_experiment(mlflow)
        tags = self._base_tags(metric_type)

        if run_name is None:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"{metric_type}_{ts}"

        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            tags=tags,
        ) as run:
            # Flatten scalars and log as MLflow metrics
            for key, result in results.items():
                result_dict = _result_to_dict(result)
                scalars = _extract_scalar_metrics(result_dict, prefix=key)
                for metric_name, metric_value in scalars.items():
                    # Replace dots with slashes — MLflow metric names can't have dots
                    safe_name = metric_name.replace(".", "/")
                    mlflow.log_metric(safe_name, metric_value)

            # Log full results as JSON artifact for complete audit trail
            artifact_payload = {}
            for key, result in results.items():
                artifact_payload[key] = _result_to_dict(result)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"monitoring_{metric_type}_",
                delete=False,
            ) as f:
                json.dump(artifact_payload, f, indent=2, default=str)
                tmp_path = f.name

            try:
                mlflow.log_artifact(tmp_path, artifact_path="monitoring")
            finally:
                os.unlink(tmp_path)

            return run.info.run_id

    def log_ae_ratios(
        self,
        ae_results: Union[float, dict[str, Any]],
        run_name: Optional[str] = None,
    ) -> str:
        """Log Actual/Expected ratio results to MLflow.

        Args:
            ae_results: Either a single float (from ``ae_ratio()`` without
                segments) or a dict mapping names to results. Results can be
                floats, polars DataFrames (segmented output), or any other
                object returned by ``ae_ratio`` or ``ae_ratio_ci``.
            run_name: Optional name for the MLflow run. Auto-generated if not
                provided.

        Returns:
            The MLflow run ID.
        """
        mlflow = self._mlflow()
        if not isinstance(ae_results, dict):
            ae_results = {"ae_ratio": ae_results}
        return self._log_result(mlflow, "ae_ratio", ae_results, run_name)

    def log_psi(
        self,
        psi_results: Union[float, dict[str, Any]],
        run_name: Optional[str] = None,
    ) -> str:
        """Log PSI/CSI results to MLflow.

        Args:
            psi_results: Float or dict mapping feature/segment names to PSI
                values or CSI DataFrames.
            run_name: Optional run name.

        Returns:
            The MLflow run ID.
        """
        mlflow = self._mlflow()
        if not isinstance(psi_results, dict):
            psi_results = {"psi": psi_results}
        return self._log_result(mlflow, "psi", psi_results, run_name)

    def log_gini_drift(
        self,
        gini_results: Union[Any, dict[str, Any]],
        run_name: Optional[str] = None,
    ) -> str:
        """Log Gini drift test results to MLflow.

        Args:
            gini_results: A ``GiniDriftResult``, ``GiniDriftOneSampleResult``,
                ``GiniBootstrapResult``, or a dict mapping names to such objects.
            run_name: Optional run name.

        Returns:
            The MLflow run ID.
        """
        mlflow = self._mlflow()
        if not isinstance(gini_results, dict):
            gini_results = {"gini_drift": gini_results}
        return self._log_result(mlflow, "gini_drift", gini_results, run_name)

    def log_monitoring_report(
        self,
        report_dict: dict[str, Any],
        run_name: Optional[str] = None,
    ) -> str:
        """Log a full MonitoringReport output to MLflow.

        The expected input is the dict returned by
        ``MonitoringReport.run()``, but any dict of named results works.

        Args:
            report_dict: Dict of monitoring results, keyed by metric name.
            run_name: Optional run name.

        Returns:
            The MLflow run ID.
        """
        mlflow = self._mlflow()
        return self._log_result(mlflow, "monitoring_report", report_dict, run_name)

    def get_monitoring_history(
        self,
        metric_type: Optional[str] = None,
    ) -> Any:
        """Query past monitoring runs and return a summary DataFrame.

        Each row is one monitoring run. Columns include the run ID, start time,
        metric type, model version, and all scalar metrics logged.

        Requires pandas to be installed (pandas is returned by
        ``mlflow.search_runs()``).

        Args:
            metric_type: Filter to runs of a specific type (e.g. ``"ae_ratio"``,
                ``"psi"``, ``"gini_drift"``). Returns all types if not specified.

        Returns:
            pandas DataFrame with columns: ``run_id``, ``run_name``,
            ``start_time``, ``status``, ``metric_type``, ``model_version``, plus
            one column per logged metric. Returns an empty DataFrame if no runs
            found.

        Raises:
            ImportError: If pandas is not installed.
        """
        _require_pandas()
        mlflow = self._mlflow()

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        exp_name = self._get_experiment_name()
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            return pd.DataFrame()

        filter_parts = [f"tags.`monitoring.model_name` = '{self.model_name}'"]
        if metric_type:
            filter_parts.append(f"tags.`monitoring.metric_type` = '{metric_type}'")
        if self.model_version:
            filter_parts.append(
                f"tags.`monitoring.model_version` = '{self.model_version}'"
            )

        filter_string = " AND ".join(filter_parts)

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
        )

        if runs.empty:
            return pd.DataFrame()

        # Rename tag columns to friendlier names
        rename = {
            "tags.monitoring.metric_type": "metric_type",
            "tags.monitoring.model_version": "model_version",
            "tags.monitoring.timestamp": "monitoring_timestamp",
            "tags.monitoring.library_version": "library_version",
        }
        runs = runs.rename(columns={k: v for k, v in rename.items() if k in runs.columns})

        # Keep a tidy core set of columns, then append all metrics
        core_cols = ["run_id", "run_name", "start_time", "status"]
        friendly_cols = list(rename.values())
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]

        keep = [c for c in core_cols + friendly_cols + metric_cols if c in runs.columns]
        return runs[keep].reset_index(drop=True)
