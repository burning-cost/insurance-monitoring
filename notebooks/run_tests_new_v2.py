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

# COMMAND ----------
# Run the new test files (v2 tests)
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-monitoring/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

new_test_files = [
    "/Workspace/insurance-monitoring/tests/test_drift_v2.py",
    "/Workspace/insurance-monitoring/tests/test_discrimination_v4.py",
    "/Workspace/insurance-monitoring/tests/test_drift_attribution_v2.py",
    "/Workspace/insurance-monitoring/tests/test_interpretable_drift_v2.py",
    "/Workspace/insurance-monitoring/tests/test_calibration_plots.py",
]

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        *new_test_files,
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
print(output)
dbutils.notebook.exit(output[-8000:] if len(output) > 8000 else output)
