# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib pytest pytest-cov sortedcontainers pandas

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys
import os

# Add the src directory to sys.path so insurance_monitoring is importable
sys.path.insert(0, "/Workspace/insurance-monitoring/src")

# Verify import works
import insurance_monitoring
print(f"insurance_monitoring loaded from: {insurance_monitoring.__file__}")
print(f"Version: {insurance_monitoring.__version__}")

# Verify new classes are importable from top-level package
from insurance_monitoring import (
    PricingDriftMonitor,
    PricingDriftResult,
    CalibTestResult,
    CalibrationCUSUM,
    CUSUMAlarm,
    CUSUMSummary,
)
print(f"PricingDriftMonitor importable: {PricingDriftMonitor}")
print(f"CalibrationCUSUM importable: {CalibrationCUSUM}")

# Quick smoke test
import numpy as np

rng = np.random.default_rng(42)
mu_ref = rng.uniform(0.05, 0.20, 2000)
y_ref = rng.poisson(mu_ref).astype(float)
mu_mon = rng.uniform(0.05, 0.20, 1000)
y_mon = rng.poisson(mu_mon).astype(float)

monitor = PricingDriftMonitor(n_bootstrap=100, random_state=0)
monitor.fit(y_ref, mu_ref)
result = monitor.test(y_mon, mu_mon)
print(f"PricingDriftMonitor smoke test verdict: {result.verdict}")

cusum = CalibrationCUSUM(delta_a=2.0, cfar=0.005, n_mc=500, random_state=0)
p = rng.uniform(0.05, 0.25, 50)
y = rng.binomial(1, p)
alarm = cusum.update(p, y)
print(f"CalibrationCUSUM smoke test: S_t={alarm.statistic:.4f}, h_t={alarm.control_limit:.4f}")

# COMMAND ----------
# Run the new test files
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-monitoring/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_pricing_drift.py",
        "/Workspace/insurance-monitoring/tests/test_cusum.py",
        "-v", "--tb=long",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        f"--rootdir=/Workspace/insurance-monitoring",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output[-8000:] if len(output) > 8000 else output)
dbutils.notebook.exit(output[-5000:] if len(output) > 5000 else output)
