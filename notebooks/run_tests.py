# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib pytest pytest-cov sortedcontainers

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

# Verify PITMonitor is available (v0.7.0)
from insurance_monitoring.calibration import PITMonitor, PITAlarm, PITSummary
print("PITMonitor available (v0.7.0)")
from insurance_monitoring.calibration import murphy_decomposition, CalibrationChecker
print("CalibrationChecker available (built in since v0.3.0)")

# COMMAND ----------
# Databricks workspace filesystem does not support __pycache__ creation.
# Disable bytecode writing and use a /tmp cache directory for pytest.
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-monitoring/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests",
        "-v", "--tb=long",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        "-m", "not slow",
        f"--rootdir=/Workspace/insurance-monitoring",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-5000:] if len(output) > 5000 else output)
