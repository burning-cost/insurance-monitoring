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

# Verify new classes are importable
from insurance_monitoring import GiniDriftMonitor, GiniBootstrapMonitor, MurphyDecomposition
print(f"GiniDriftMonitor: {GiniDriftMonitor}")
print(f"GiniBootstrapMonitor: {GiniBootstrapMonitor}")
print(f"MurphyDecomposition: {MurphyDecomposition}")

# COMMAND ----------
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-monitoring/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_gini_monitoring.py",
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
